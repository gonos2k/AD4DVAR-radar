"""Persistent tiny neural observation-error learning on the exact FV oracle.

This is a bounded research experiment.  It updates only a three-parameter
observation-error model, uses the exact stationary response as the optimizer
gradient, and never uses the held-out window to form an update.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fv_neural_learning_probe as neural
import fv_sensitivity_probe as oracle


DTYPE = torch.float64
LEARNING_RATE = 0.25
MOMENTUM = 0.5
FD_STEPS = (1.0e-3, 5.0e-4)
WINDOW_SCHEDULE = ("train_a", "train_b", "train_a")
CHECKPOINT_STEP = 1


def _training_windows() -> tuple[dict[str, tuple[Any, Any, torch.Tensor]], Any, Any, Any, Any, Any]:
    """Return two deterministic training windows and one immutable holdout."""

    train_a, frozen_a, train_truth, holdout, holdout_frozen, holdout_truth, future, support = neural._windows()
    raw_observations, raw_frozen, _, _ = oracle.make_case()
    raw_frozen = replace(raw_frozen, analysis_config=frozen_a.analysis_config)
    coordinates = torch.arange(
        raw_observations.dbz.numel(), dtype=DTYPE
    ).reshape_as(raw_observations.dbz)
    train_b, frozen_b = neural.base._window(
        raw_observations,
        raw_frozen,
        0.017 * torch.cos(0.53 * coordinates + 0.8),
    )
    windows = {
        "train_a": (train_a, frozen_a, train_truth),
        "train_b": (train_b, frozen_b, train_truth),
    }
    if torch.equal(train_a.dbz, train_b.dbz):
        raise AssertionError("training windows must be distinct")
    return (
        windows,
        holdout,
        holdout_frozen,
        holdout_truth,
        future,
        support,
    )


def _optimizer(model: neural.TinyObservationStd) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
    )


def _assign_gradient(model: neural.TinyObservationStd, gradient: torch.Tensor) -> None:
    cursor = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.grad = gradient[cursor : cursor + count].reshape_as(parameter).detach().clone()
        cursor += count
    if cursor != gradient.numel():
        raise AssertionError("parameter gradient size mismatch")


def _score(
    control: torch.Tensor,
    frozen: Any,
    future: Any,
    support: Any,
    verification: torch.Tensor,
) -> torch.Tensor:
    return neural.base._score(
        control,
        frozen,
        future,
        support,
        verification,
        torch.ones_like(verification),
    )


def _stationary_control(
    model: neural.TinyObservationStd,
    observations: Any,
    frozen: Any,
    params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = neural._features(observations)
    _, control = neural._run(
        observations, frozen, model, features, params, polish=True
    )
    return features, control


def _directional_check_with_support(
    model: neural.TinyObservationStd,
    observations: Any,
    frozen: Any,
    future: Any,
    support: Any,
    verification: torch.Tensor,
    features: torch.Tensor,
    params: torch.Tensor,
    control: torch.Tensor,
    response: torch.Tensor,
) -> dict[str, Any]:
    response_norm = float(torch.linalg.vector_norm(response))
    if not bool(torch.isfinite(response).all()) or response_norm == 0.0:
        raise RuntimeError("persistent update has a zero or nonfinite response")
    direction = response / response_norm
    predicted = float(torch.dot(response, direction))
    finite_difference: list[float] = []
    errors: list[float] = []
    endpoint_stationarity: list[float] = []
    margins: list[float] = []
    nominal_faces = oracle.face_signs(control, frozen)
    same_faces = True
    objective = neural._objective(observations, frozen, model, features)
    for step in FD_STEPS:
        plus_params = params + params.new_tensor(step) * direction
        minus_params = params - params.new_tensor(step) * direction
        _, plus_control = neural._run(
            observations, frozen, model, features, plus_params, polish=True
        )
        _, minus_control = neural._run(
            observations, frozen, model, features, minus_params, polish=True
        )
        plus_score = _score(plus_control, frozen, future, support, verification)
        minus_score = _score(minus_control, frozen, future, support, verification)
        slope = float((plus_score - minus_score) / (2.0 * step))
        finite_difference.append(slope)
        errors.append(abs(slope - predicted))
        endpoint_stationarity.extend(
            [
                float(torch.func.grad(objective, argnums=0)(plus_control, plus_params).abs().max()),
                float(torch.func.grad(objective, argnums=0)(minus_control, minus_params).abs().max()),
            ]
        )
        margins.extend(
            [
                oracle.face_branch_margin(control, plus_control, frozen),
                oracle.face_branch_margin(control, minus_control, frozen),
            ]
        )
        same_faces &= bool(torch.equal(nominal_faces, oracle.face_signs(plus_control, frozen)))
        same_faces &= bool(torch.equal(nominal_faces, oracle.face_signs(minus_control, frozen)))
    if errors[-1] > 1.0e-5 * abs(predicted):
        raise RuntimeError("stationary response direction failed finite-difference check")
    return {
        "response_norm": response_norm,
        "direction": direction.tolist(),
        "predicted_directional_slope": predicted,
        "finite_difference_slopes": finite_difference,
        "finite_difference_errors": errors,
        "endpoint_stationarity_max": max(endpoint_stationarity),
        "face_branch_margins": margins,
        "same_face_signs": same_faces,
    }


def _run_updates(
    model: neural.TinyObservationStd,
    optimizer: torch.optim.Optimizer,
    windows: dict[str, tuple[Any, Any, torch.Tensor]],
    future: Any,
    support: Any,
    *,
    start_step: int,
    stop_step: int,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step in range(start_step, stop_step):
        label = WINDOW_SCHEDULE[step]
        observations, frozen, verification = windows[label]
        params_before = neural._flat(model).detach().clone()
        features, control = _stationary_control(model, observations, frozen, params_before)
        response, stationarity, adjoint_residual, eigenvalue = neural._response(
            observations,
            frozen,
            future,
            support,
            verification,
            torch.ones_like(verification),
            model,
            features,
            params_before,
            control,
        )
        direction_check = _directional_check_with_support(
            model,
            observations,
            frozen,
            future,
            support,
            verification,
            features,
            params_before,
            control,
            response,
        )
        score_before = float(_score(control, frozen, future, support, verification))
        optimizer.zero_grad(set_to_none=True)
        _assign_gradient(model, response)
        optimizer.step()
        params_after = neural._flat(model).detach().clone()
        update = params_after - params_before
        if not bool(torch.dot(response, update) < 0):
            raise RuntimeError("optimizer update is not a descent direction")
        _, after_control = _stationary_control(model, observations, frozen, params_after)
        score_after = float(
            _score(after_control, frozen, future, support, verification)
        )
        if not score_after < score_before:
            raise RuntimeError("reanalysis training score did not decrease")
        record = {
            "step": step + 1,
            "window": label,
            "parameter_before": params_before.tolist(),
            "parameter_after": params_after.tolist(),
            "parameter_update": update.tolist(),
            "parameter_gradient": response.tolist(),
            "stationarity_max": stationarity,
            "adjoint_relative_residual": adjoint_residual,
            "hessian_min_eigenvalue": eigenvalue,
            "train_score_before": score_before,
            "train_score_after": score_after,
            "gradient_direction_check": direction_check,
        }
        records.append(record)
        if checkpoint_path is not None and step + 1 == CHECKPOINT_STEP:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 1,
                    "next_step": step + 1,
                    "window_schedule": WINDOW_SCHEDULE,
                    "learning_rate": LEARNING_RATE,
                    "momentum": MOMENTUM,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )
    return records


def _evaluate_holdout(
    model: neural.TinyObservationStd,
    holdout: Any,
    holdout_frozen: Any,
    holdout_truth: torch.Tensor,
    future: Any,
    support: Any,
    params: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    features, control = _stationary_control(
        model, holdout, holdout_frozen, params
    )
    del features
    forecast = neural.base._forecast(control, holdout_frozen, future, support)
    score = _score(control, holdout_frozen, future, support, holdout_truth)
    return float(score), forecast.detach().clone()


def _nested_max_abs(first: Any, second: Any) -> float:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return float((first - second).abs().max()) if first.numel() else 0.0
    if isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            return float("inf")
        return max((_nested_max_abs(first[key], second[key]) for key in first), default=0.0)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return float("inf")
        return max((_nested_max_abs(a, b) for a, b in zip(first, second)), default=0.0)
    return 0.0 if first == second else float("inf")


def run_probe(*, checkpoint_path: str | Path) -> dict[str, Any]:
    windows, holdout, holdout_frozen, holdout_truth, future, support = _training_windows()
    model = neural.TinyObservationStd()
    model.eval()
    optimizer = _optimizer(model)
    initial_params = neural._flat(model).detach().clone()
    holdout_snapshot = (
        holdout.dbz.detach().clone(),
        holdout.std_dbz.detach().clone(),
        holdout_frozen.input_frames_dbz.detach().clone(),
        holdout_truth.detach().clone(),
    )
    initial_holdout_score, initial_holdout_forecast = _evaluate_holdout(
        model,
        holdout,
        holdout_frozen,
        holdout_truth,
        future,
        support,
        initial_params,
    )
    uninterrupted_records = _run_updates(
        model,
        optimizer,
        windows,
        future,
        support,
        start_step=0,
        stop_step=len(WINDOW_SCHEDULE),
        checkpoint_path=Path(checkpoint_path),
    )
    uninterrupted_params = neural._flat(model).detach().clone()
    uninterrupted_optimizer_state = optimizer.state_dict()
    uninterrupted_holdout_score, uninterrupted_holdout_forecast = _evaluate_holdout(
        model,
        holdout,
        holdout_frozen,
        holdout_truth,
        future,
        support,
        uninterrupted_params,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    resumed_model = neural.TinyObservationStd()
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_model.eval()
    resumed_optimizer = _optimizer(resumed_model)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint["next_step"] != CHECKPOINT_STEP:
        raise AssertionError("checkpoint does not resume at step 1")
    resumed_records = _run_updates(
        resumed_model,
        resumed_optimizer,
        windows,
        future,
        support,
        start_step=int(checkpoint["next_step"]),
        stop_step=len(WINDOW_SCHEDULE),
    )
    resumed_params = neural._flat(resumed_model).detach().clone()
    resumed_holdout_score, resumed_holdout_forecast = _evaluate_holdout(
        resumed_model,
        holdout,
        holdout_frozen,
        holdout_truth,
        future,
        support,
        resumed_params,
    )
    holdout_unchanged = all(
        torch.equal(current, snapshot)
        for current, snapshot in zip(
            (
                holdout.dbz,
                holdout.std_dbz,
                holdout_frozen.input_frames_dbz,
                holdout_truth,
            ),
            holdout_snapshot,
        )
    )
    full_records = uninterrupted_records
    return {
        "scope": "4x5 CPU FP64 donorcell persistent NN observation-error learning; research-only",
        "production_learning_eligible": False,
        "neural_prior_eligible": False,
        "parameter_count": int(initial_params.numel()),
        "window_schedule": list(WINDOW_SCHEDULE),
        "distinct_training_windows": len(set(WINDOW_SCHEDULE)) >= 2,
        "optimizer": {
            "algorithm": "SGD",
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "fixed_schedule": True,
        },
        "heldout_used_for_updates": False,
        "heldout_data_immutable": holdout_unchanged,
        "initial_parameters": initial_params.tolist(),
        "uninterrupted_steps": full_records,
        "resumed_steps": resumed_records,
        "checkpoint": {
            "step": CHECKPOINT_STEP,
            "path": str(checkpoint_path),
            "fresh_model_optimizer": True,
            "next_step": int(checkpoint["next_step"]),
        },
        "resume_max_parameter_difference": float(
            (uninterrupted_params - resumed_params).abs().max()
        ),
        "resume_max_optimizer_state_difference": _nested_max_abs(
            uninterrupted_optimizer_state, resumed_optimizer.state_dict()
        ),
        "holdout_score_before": initial_holdout_score,
        "holdout_score_after_uninterrupted": uninterrupted_holdout_score,
        "holdout_score_after_resumed": resumed_holdout_score,
        "holdout_gain_uninterrupted": initial_holdout_score - uninterrupted_holdout_score,
        "holdout_gain_above_roundoff": (
            initial_holdout_score - uninterrupted_holdout_score
            > 10.0
            * torch.finfo(DTYPE).eps
            * max(abs(initial_holdout_score), abs(uninterrupted_holdout_score), 1.0e-30)
        ),
        "final_forecast_max_difference": float(
            (uninterrupted_holdout_forecast - resumed_holdout_forecast).abs().max()
        ),
        "initial_holdout_forecast_norm": float(initial_holdout_forecast.norm()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe(checkpoint_path=args.checkpoint)
    encoded = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
    print(encoded)
