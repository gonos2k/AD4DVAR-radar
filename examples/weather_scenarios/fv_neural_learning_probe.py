"""Tiny neural observation-error calibration on the exact small FV oracle.

Research evidence only: this does not use the FV neural-prior path or change
any production eligibility contract.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.func import functional_call

from advar import variational as v
import fv_learning_probe as base
import fv_sensitivity_probe as oracle


class TinyObservationStd(nn.Module):
    """Three-parameter bounded log-standard-deviation model."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        return 0.25 * torch.tanh(self.linear(features)).reshape(-1)


def _flat(model: TinyObservationStd) -> Tensor:
    return torch.cat((model.linear.weight.reshape(-1), model.linear.bias.reshape(-1)))


def _set_flat(model: TinyObservationStd, value: Tensor) -> None:
    with torch.no_grad():
        model.linear.weight.copy_(value[:2].reshape_as(model.linear.weight))
        model.linear.bias.copy_(value[2:3])


def _features(observations: object) -> Tensor:
    values = observations.dbz
    scale = values.std(unbiased=False).clamp_min(1.0e-6)
    centered = (values - values.mean()) / scale
    temporal_change = (values - values[0]) / scale
    return torch.stack((centered, temporal_change), dim=-1).reshape(-1, 2)


def _std_observations(
    observations: object, params: Tensor, model: TinyObservationStd, features: Tensor
) -> object:
    names = {"linear.weight": params[:2].reshape_as(model.linear.weight),
             "linear.bias": params[2:3]}
    bounded_log_std = functional_call(model, names, (features,))
    std = 0.1 * torch.exp(bounded_log_std).reshape_as(observations.dbz)
    return replace(observations, std_dbz=std)


def _objective(observations, frozen, model, features):
    def objective(control: Tensor, params: Tensor) -> Tensor:
        return v.robust_objective(
            control, _std_observations(observations, params, model, features), frozen
        )
    return objective


def _run(observations, frozen, model, features, params, *, polish: bool):
    objective = _objective(observations, frozen, model, features)
    result = v.solve_analysis(
        _std_observations(observations, params, model, features), frozen
    )
    control = result.control
    if polish:
        control = oracle.polish(
            objective, control, params,
            check_step=lambda start, stop: oracle.face_branch_margin(start, stop, frozen),
        )
    return result, control


def _response(observations, frozen, future, support, verification, weight,
              model, features, params, control):
    objective = _objective(observations, frozen, model, features)
    score = lambda value, _: base._score(value, frozen, future, support, verification, weight)
    response, _, adjoint_residual = oracle.stationary_sensitivity(
        objective, score, control, params
    )
    gradient = torch.func.grad(objective, argnums=0)(control, params)
    hessian = oracle.dense_hessian(torch.func.grad(objective, argnums=0), control, params)
    return response, float(gradient.abs().max()), adjoint_residual, float(torch.linalg.eigvalsh(hessian).min())


def _local_error(observations, frozen, future, support, verification, weight,
                 model, features, params, control):
    objective = _objective(observations, frozen, model, features)
    gradient = torch.func.grad(objective, argnums=0)
    residual = gradient(control, params)
    hessian = oracle.dense_hessian(gradient, control, params)
    eigenvalue = torch.linalg.eigvalsh(hessian).min()
    score_grad = torch.func.grad(
        lambda value: base._score(value, frozen, future, support, verification, weight)
    )(control)
    return {
        "gradient_max": float(residual.abs().max()),
        "hessian_min_eigenvalue": float(eigenvalue),
        "local_score_error_scale": float(
            torch.linalg.vector_norm(score_grad) * torch.linalg.vector_norm(residual) / eigenvalue
        ),
    }


def _windows():
    observations, frozen, future, support = oracle.make_case()
    frozen = replace(frozen, analysis_config=base._analysis_config(frozen))
    coordinates = torch.arange(observations.dbz.numel(), dtype=torch.float64).reshape_as(observations.dbz)
    train_obs, train_frozen = base._window(observations, frozen, 0.02 * torch.sin(0.37 * coordinates))
    train_truth = base._trajectory_dbz(frozen, future, support, observations.dbz[0])[3:]
    holdout_initial = observations.dbz[0] + 0.08 * torch.cos(torch.arange(observations.dbz[0].numel(), dtype=torch.float64).reshape_as(observations.dbz[0]))
    holdout_truth_all = base._trajectory_dbz(frozen, future, support, holdout_initial)
    holdout_obs, holdout_frozen = base._window(
        replace(observations, dbz=holdout_truth_all[:3]), frozen,
        0.02 * torch.cos(0.61 * coordinates + 0.4),
    )
    return train_obs, train_frozen, train_truth, holdout_obs, holdout_frozen, holdout_truth_all[3:], future, support


def run_probe(*, checkpoint_path: str | Path | None = None) -> dict[str, object]:
    train_obs, train_frozen, train_truth, hold_obs, hold_frozen, hold_truth, future, support = _windows()
    model = TinyObservationStd()
    params = _flat(model).detach()
    train_features, hold_features = _features(train_obs), _features(hold_obs)
    weight, hold_weight = torch.ones_like(train_truth), torch.ones_like(hold_truth)
    _, base_control = _run(train_obs, train_frozen, model, train_features, params, polish=True)
    response, stationarity, adjoint_residual, eigenvalue = _response(
        train_obs, train_frozen, future, support, train_truth, weight,
        model, train_features, params, base_control,
    )
    direction = torch.tensor([1.0, -0.3, 0.7], dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    fd = []
    fd_stationarity = []
    for step in (1.0e-3, 5.0e-4):
        plus, minus = params + step * direction, params - step * direction
        _, cp = _run(train_obs, train_frozen, model, train_features, plus, polish=True)
        _, cm = _run(train_obs, train_frozen, model, train_features, minus, polish=True)
        fd.append(float((base._score(cp, train_frozen, future, support, train_truth, weight)
                         - base._score(cm, train_frozen, future, support, train_truth, weight)) / (2.0 * step)))
        fd_stationarity.extend([float(torch.func.grad(_objective(train_obs, train_frozen, model, train_features), argnums=0)(cp, plus).abs().max()), float(torch.func.grad(_objective(train_obs, train_frozen, model, train_features), argnums=0)(cm, minus).abs().max())])
    learning_rate = 0.25
    updated_params = params - learning_rate * response
    _, updated_control = _run(train_obs, train_frozen, model, train_features, updated_params, polish=True)
    _, hold_base = _run(hold_obs, hold_frozen, model, hold_features, params, polish=True)
    _, hold_updated = _run(hold_obs, hold_frozen, model, hold_features, updated_params, polish=True)
    hold_before = base._score(hold_base, hold_frozen, future, support, hold_truth, hold_weight)
    hold_after = base._score(hold_updated, hold_frozen, future, support, hold_truth, hold_weight)
    hold_base_diag = _local_error(hold_obs, hold_frozen, future, support, hold_truth, hold_weight, model, hold_features, params, hold_base)
    hold_updated_diag = _local_error(hold_obs, hold_frozen, future, support, hold_truth, hold_weight, model, hold_features, updated_params, hold_updated)
    reload_errors = {"score": None, "control": None, "forecast": None}
    if checkpoint_path is not None:
        path = Path(checkpoint_path); path.parent.mkdir(parents=True, exist_ok=True)
        updated_model = TinyObservationStd(); _set_flat(updated_model, updated_params)
        torch.save(updated_model.state_dict(), path)
        loaded_model = TinyObservationStd(); loaded_model.load_state_dict(torch.load(path, weights_only=True))
        loaded_params = _flat(loaded_model); hold_loaded_features = _features(hold_obs)
        _, hold_loaded = _run(hold_obs, hold_frozen, loaded_model, hold_loaded_features, loaded_params, polish=True)
        loaded_score = base._score(hold_loaded, hold_frozen, future, support, hold_truth, hold_weight)
        reload_errors = {"score": float((loaded_score - hold_after).abs()), "control": float((hold_loaded - hold_updated).abs().max()), "forecast": float((base._forecast(hold_loaded, hold_frozen, future, support) - base._forecast(hold_updated, hold_frozen, future, support)).abs().max())}
    gain = float(hold_before - hold_after)
    roundoff = torch.finfo(torch.float64).eps * max(abs(float(hold_before)), abs(float(hold_after)), 1.0e-30)
    budget = 10.0 * (hold_base_diag["local_score_error_scale"] + hold_updated_diag["local_score_error_scale"] + (reload_errors["score"] or 0.0) + roundoff)
    return {"scope": "4x5 CPU FP64 donorcell tiny neural observation-error update; research-only", "production_learning_eligible": False, "control_count": int(base_control.numel()), "parameter_count": int(params.numel()), "parameter_gradient": response.tolist(), "predicted_directional_slope": float(torch.dot(response, direction)), "finite_difference_slopes": fd, "finite_difference_stationarity_max": max(fd_stationarity), "stationarity_max": stationarity, "adjoint_relative_residual": adjoint_residual, "hessian_min_eigenvalue": eigenvalue, "train_score_before": float(base._score(base_control, train_frozen, future, support, train_truth, weight)), "train_score_after": float(base._score(updated_control, train_frozen, future, support, train_truth, weight)), "holdout_score_before": float(hold_before), "holdout_score_after": float(hold_after), "holdout_gain": gain, "holdout_error_budget": budget, "holdout_gain_above_numerical_error": gain > budget if checkpoint_path is not None else None, "numerical_error_scope": "local linearized estimate, not rigorous bound", "initial_parameters": params.tolist(), "updated_parameters": updated_params.tolist(), "holdout_base_diagnostics": hold_base_diag, "holdout_updated_diagnostics": hold_updated_diag, "reload_errors": reload_errors, "same_face_signs": bool(torch.equal(oracle.face_signs(base_control, train_frozen), oracle.face_signs(updated_control, train_frozen))), "face_margin": oracle.face_branch_margin(base_control, updated_control, train_frozen)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); report = run_probe(checkpoint_path=args.checkpoint)
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(encoded + "\n")
    print(encoded)
