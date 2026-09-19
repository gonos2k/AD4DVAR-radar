"""Bounded Phase 1 proof for an externally learned FV background.

This research probe composes the real 4x5 FV objective/forecast with
``B_theta(y)`` supplied outside the typed neural-prior API.  It verifies the
stationary chain rule and one fixed training update on an independent window.
It intentionally does not construct a NeuralPriorApplication, an FSOI result,
or a production learning/certificate record.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from advar import variational as v
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.transport import finite_volume_trajectory

import fv_sensitivity_probe as oracle


DTYPE = torch.float64
FD_STEPS = (1.0e-3, 5.0e-4)
LEARNING_RATE = 5.0e-2
INITIAL_THETA = (0.1, -0.2, 0.15)


class TinyMeanBackground(nn.Module):
    """Three-parameter causal mean correction, used only for checkpointing."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.lag1 = nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.lag2 = nn.Parameter(torch.zeros((), dtype=DTYPE))

    def vector(self) -> Tensor:
        return torch.stack((self.bias, self.lag1, self.lag2))


def background(theta: Tensor, observations_dbz: Tensor) -> Tensor:
    """B_theta(y): y[0] plus a bounded spatial correction from first observation."""

    if theta.shape != (3,):
        raise ValueError("three background parameters are required")
    first = observations_dbz[0]
    west = torch.cat((first[..., :1], first[..., :-1]), dim=-1)
    north = torch.cat((first[..., :1, :], first[..., :-1, :]), dim=-2)
    lag1 = torch.tanh(first - west)
    lag2 = torch.tanh(first - north)
    correction = torch.tanh(theta[0] + theta[1] * lag1 + theta[2] * lag2)
    return observations_dbz[0] + 0.1 * correction


def _analysis_config(frozen: Any) -> Any:
    return replace(
        frozen.analysis_config,
        maximum_outer_iterations=16,
        maximum_pcg_iterations=96,
        gradient_tolerance=1e-10,
        step_tolerance=1e-12,
        pcg_relative_tolerance=1e-10,
    )


def _trajectory_dbz(
    frozen: Any, future: Any, support: Any, initial_dbz: Tensor
) -> Tensor:
    spec = frozen.fv_transport
    if spec is None:
        raise RuntimeError("FV fixture is missing transport")
    frames, _ = finite_volume_trajectory(
        dbz_to_echo(initial_dbz, min_dbz=-10.0),
        torch.ones_like(initial_dbz),
        spec.coefficient_limits * initial_dbz.new_tensor([0.2, -0.1, 0.15]).tanh(),
        initial_dbz.new_tensor(0.01),
        psi_basis=spec.psi_basis,
        leads=4,
        substeps_per_interval=spec.substeps_per_interval,
        interval_seconds=60.0,
        spacing_yx=spec.spacing_yx,
        boundary_echo=spec.boundary_echo + future,
        boundary_support=spec.boundary_support + support,
        reconstruction="donorcell",
    )
    return echo_to_dbz(frames, min_dbz=-10.0)


def _frozen_with_background(frozen: Any, value: Tensor) -> Any:
    # Support, sigma, transport, and all typed-prior fields remain unchanged.
    return replace(frozen, initial_background_dbz=value)


def _forecast(
    control: Tensor,
    frozen: Any,
    future: Any,
    support: Any,
) -> Tensor:
    trajectory = v.forecast_fv_analysis(
        control,
        frozen,
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future,
        boundary_support=support,
    )
    return echo_to_dbz(trajectory.frames_linear[1:], min_dbz=-10.0)


def _score(
    control: Tensor,
    frozen: Any,
    future: Any,
    support: Any,
    verification: Tensor,
    weight: Tensor,
) -> Tensor:
    difference = _forecast(control, frozen, future, support) - verification
    return (weight * difference.square()).sum() / weight.sum()


def _objective(
    control: Tensor, observations: Any, frozen: Any, theta: Tensor
) -> Tensor:
    return v.robust_objective(
        control,
        observations,
        _frozen_with_background(frozen, background(theta, observations.dbz)),
    )


def _score_theta(
    control: Tensor,
    observations: Any,
    frozen: Any,
    theta: Tensor,
    future: Any,
    support: Any,
    verification: Tensor,
    weight: Tensor,
) -> Tensor:
    return _score(
        control,
        _frozen_with_background(frozen, background(theta, observations.dbz)),
        future,
        support,
        verification,
        weight,
    )


def _solve(
    observations: Any,
    frozen: Any,
    theta: Tensor,
    future: Any,
    support: Any,
    verification: Tensor,
    weight: Tensor,
    *,
    start: Tensor | None = None,
) -> tuple[Any, Tensor]:
    injected = _frozen_with_background(frozen, background(theta, observations.dbz))
    result = v.solve_analysis(observations, injected) if start is None else None
    initial = result.control if result is not None else start
    objective = lambda c, p: _objective(c, observations, frozen, p)
    control = oracle.polish(
        objective,
        initial,
        theta,
        check_step=lambda left, right: oracle.face_branch_margin(left, right, frozen),
    )
    return result, control


def _stationary_response(
    objective: Callable[[Tensor, Tensor], Tensor],
    score: Callable[[Tensor, Tensor], Tensor],
    control: Tensor,
    parameter: Tensor,
) -> tuple[Tensor, float, float]:
    response, _, residual = oracle.stationary_sensitivity(
        objective, score, control, parameter
    )
    stationarity = torch.func.grad(objective, argnums=0)(control, parameter)
    return response, float(stationarity.abs().max()), residual


def _finite_reanalysis(
    objective: Callable[[Tensor, Tensor], Tensor],
    score: Callable[[Tensor, Tensor], Tensor],
    control: Tensor,
    parameter: Tensor,
    direction: Tensor,
    check_frozen: Any,
) -> list[dict[str, float]]:
    baseline = score(control, parameter)
    values: list[dict[str, float]] = []
    for h in FD_STEPS:
        plus_parameter = parameter + h * direction
        minus_parameter = parameter - h * direction
        plus = oracle.polish(
            objective,
            control,
            plus_parameter,
            check_step=lambda left, right: oracle.face_branch_margin(
                left, right, check_frozen
            ),
        )
        minus = oracle.polish(
            objective,
            control,
            minus_parameter,
            check_step=lambda left, right: oracle.face_branch_margin(
                left, right, check_frozen
            ),
        )
        if not (
            torch.equal(oracle.face_signs(plus, check_frozen), oracle.face_signs(control, check_frozen))
            and torch.equal(oracle.face_signs(minus, check_frozen), oracle.face_signs(control, check_frozen))
        ):
            raise RuntimeError("finite reanalysis left the donorcell face branch")
        centered = (score(plus, plus_parameter) - score(minus, minus_parameter)) / (2.0 * h)
        actual = score(plus, plus_parameter) - baseline
        values.append(
            {
                "h": h,
                "actual_change": float(actual),
                "centered_slope": float(centered),
                "plus_gradient_max": float(torch.func.grad(objective, argnums=0)(plus, plus_parameter).abs().max()),
                "minus_gradient_max": float(torch.func.grad(objective, argnums=0)(minus, minus_parameter).abs().max()),
                "face_box_margin": oracle.face_branch_margin(control, plus, check_frozen),
            }
        )
    return values


def _chain_rule(
    observations: Any,
    frozen: Any,
    theta: Tensor,
    control: Tensor,
    future: Any,
    support: Any,
    verification: Tensor,
    weight: Tensor,
) -> dict[str, Any]:
    y = observations.dbz
    b0 = background(theta, y)

    def objective_b(c: Tensor, b: Tensor) -> Tensor:
        return v.robust_objective(c, observations, _frozen_with_background(frozen, b))

    def score_b(c: Tensor, b: Tensor) -> Tensor:
        return _score(c, _frozen_with_background(frozen, b), future, support, verification, weight)

    g_b, b_stationarity, b_adjoint = _stationary_response(objective_b, score_b, control, b0)

    def objective_y_fixed(c: Tensor, value: Tensor) -> Tensor:
        return v.robust_objective(c, replace(observations, dbz=value), _frozen_with_background(frozen, b0))

    def score_y_fixed(c: Tensor, value: Tensor) -> Tensor:
        return _score(c, _frozen_with_background(frozen, b0), future, support, verification, weight)

    direct_y, y_fixed_stationarity, y_fixed_adjoint = _stationary_response(
        objective_y_fixed, score_y_fixed, control, y
    )

    def objective_y_total(c: Tensor, value: Tensor) -> Tensor:
        obs_value = replace(observations, dbz=value)
        return v.robust_objective(c, obs_value, _frozen_with_background(frozen, background(theta, value)))

    def score_y_total(c: Tensor, value: Tensor) -> Tensor:
        obs_value = replace(observations, dbz=value)
        return _score(c, _frozen_with_background(frozen, background(theta, value)), future, support, verification, weight)

    total_y, y_stationarity, y_adjoint = _stationary_response(
        objective_y_total, score_y_total, control, y
    )
    _, pullback_y = torch.func.vjp(lambda value: background(theta, value), y)
    chain_y = pullback_y(g_b)[0]
    identity_y = torch.zeros_like(y)
    identity_y[0] = g_b
    background_y_nonidentity = chain_y - identity_y

    def objective_theta(c: Tensor, value: Tensor) -> Tensor:
        return _objective(c, observations, frozen, value)

    def score_theta(c: Tensor, value: Tensor) -> Tensor:
        return _score_theta(c, observations, frozen, value, future, support, verification, weight)

    total_theta, theta_stationarity, theta_adjoint = _stationary_response(
        objective_theta, score_theta, control, theta
    )
    _, pullback_theta = torch.func.vjp(lambda value: background(value, y), theta)
    chain_theta = pullback_theta(g_b)[0]

    return {
        "background_gradient_max": float(g_b.abs().max()),
        "background_stationarity_max": b_stationarity,
        "background_adjoint_relative_residual": b_adjoint,
        "direct_y_gradient_max": float(direct_y.abs().max()),
        "direct_y_stationarity_max": y_fixed_stationarity,
        "direct_y_adjoint_relative_residual": y_fixed_adjoint,
        "total_y_gradient_max": float(total_y.abs().max()),
        "total_y_stationarity_max": y_stationarity,
        "total_y_adjoint_relative_residual": y_adjoint,
        "total_theta": total_theta.tolist(),
        "chain_theta": chain_theta.tolist(),
        "theta_chain_max_error": float((total_theta - chain_theta).abs().max()),
        "total_y_chain_max_error": float((total_y - direct_y - chain_y).abs().max()),
        "total_y_norm": float(total_y.norm()),
        "chain_y_norm": float(chain_y.norm()),
        "identity_y_chain_norm": float(identity_y.norm()),
        "background_y_nonidentity_norm": float(background_y_nonidentity.norm()),
        "background_norm": float(b0.norm()),
        "gB_norm": float(g_b.norm()),
        "total_y_response": total_y,
        "total_theta_response": total_theta,
        "objective_theta": objective_theta,
        "score_theta": score_theta,
        "objective_y_total": objective_y_total,
        "score_y_total": score_y_total,
    }


def run_probe(*, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    observations, frozen, future, support = oracle.make_case()
    frozen = replace(frozen, analysis_config=_analysis_config(frozen))
    coordinate = torch.arange(observations.dbz.numel(), dtype=DTYPE).reshape_as(observations.dbz)
    train_noise = 0.02 * torch.sin(0.37 * coordinate)
    holdout_noise = 0.02 * torch.cos(0.61 * coordinate + 0.4)
    train_obs = replace(observations, dbz=observations.dbz + train_noise)
    train_truth = _trajectory_dbz(frozen, future, support, observations.dbz[0])
    train_verification = train_truth[3:]
    holdout_initial = observations.dbz[0] + 0.08 * torch.cos(
        torch.arange(observations.dbz[0].numel(), dtype=DTYPE).reshape_as(observations.dbz[0])
    )
    holdout_truth = _trajectory_dbz(frozen, future, support, holdout_initial)
    holdout_obs = replace(observations, dbz=holdout_truth[:3] + holdout_noise)
    holdout_verification = holdout_truth[3:]
    train_weight = torch.ones_like(train_verification)
    holdout_weight = torch.ones_like(holdout_verification)

    model = TinyMeanBackground()
    # Fixed before evaluating either training or heldout scores; this keeps the
    # y-Jacobian probe away from the identity-only theta=0 special case.
    with torch.no_grad():
        for parameter, value in zip(model.parameters(), INITIAL_THETA):
            parameter.copy_(torch.tensor(value, dtype=DTYPE))
    theta = model.vector().detach().clone().requires_grad_()
    _, train_control = _solve(
        train_obs, frozen, theta, future, support, train_verification, train_weight
    )
    chain = _chain_rule(
        train_obs, frozen, theta, train_control, future, support, train_verification, train_weight
    )
    theta_response = chain["total_theta_response"]
    y_response = chain["total_y_response"]
    theta_direction = torch.tensor((0.6, -0.8, 0.5), dtype=DTYPE)
    theta_direction /= theta_direction.norm()
    y_direction = torch.sin(torch.arange(train_obs.dbz.numel(), dtype=DTYPE)).reshape_as(train_obs.dbz)
    y_direction /= y_direction.norm()
    theta_fd = _finite_reanalysis(
        chain["objective_theta"], chain["score_theta"], train_control, theta,
        theta_direction, frozen
    )
    y_fd = _finite_reanalysis(
        chain["objective_y_total"], chain["score_y_total"], train_control, train_obs.dbz,
        y_direction, frozen
    )
    theta_predicted = float((theta_response * theta_direction).sum())
    y_predicted = float((y_response * y_direction).sum())
    for row in theta_fd:
        row["predicted_slope"] = theta_predicted
        row["slope_error"] = abs(row["centered_slope"] - theta_predicted)
    for row in y_fd:
        row["predicted_slope"] = y_predicted
        row["slope_error"] = abs(row["centered_slope"] - y_predicted)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    with torch.no_grad():
        for parameter, gradient in zip(model.parameters(), theta_response):
            parameter.grad = gradient.detach().clone()
    optimizer.step()
    updated_theta = model.vector().detach().clone()
    _, updated_train_control = _solve(
        train_obs, frozen, updated_theta, future, support, train_verification, train_weight
    )
    train_before = _score_theta(train_control, train_obs, frozen, theta, future, support, train_verification, train_weight)
    train_after = _score_theta(updated_train_control, train_obs, frozen, updated_theta, future, support, train_verification, train_weight)
    _, holdout_before_control = _solve(
        holdout_obs, frozen, theta, future, support, holdout_verification, holdout_weight
    )
    _, holdout_after_control = _solve(
        holdout_obs, frozen, updated_theta, future, support, holdout_verification, holdout_weight
    )
    holdout_before = _score_theta(holdout_before_control, holdout_obs, frozen, theta, future, support, holdout_verification, holdout_weight)
    holdout_after = _score_theta(holdout_after_control, holdout_obs, frozen, updated_theta, future, support, holdout_verification, holdout_weight)

    reload_errors: dict[str, float] | None = None
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        reloaded_model = TinyMeanBackground()
        reloaded_optimizer = torch.optim.SGD(reloaded_model.parameters(), lr=LEARNING_RATE)
        payload = torch.load(path, weights_only=True)
        reloaded_model.load_state_dict(payload["model"])
        reloaded_optimizer.load_state_dict(payload["optimizer"])
        reloaded_theta = reloaded_model.vector().detach()
        _, reload_control = _solve(
            holdout_obs, frozen, reloaded_theta, future, support, holdout_verification, holdout_weight
        )
        reload_score = _score_theta(reload_control, holdout_obs, frozen, reloaded_theta, future, support, holdout_verification, holdout_weight)
        reload_forecast = _forecast(
            reload_control,
            _frozen_with_background(frozen, background(reloaded_theta, holdout_obs.dbz)),
            future,
            support,
        )
        updated_forecast = _forecast(
            holdout_after_control,
            _frozen_with_background(frozen, background(updated_theta, holdout_obs.dbz)),
            future,
            support,
        )
        reload_errors = {
            "theta_max_abs": float((reloaded_theta - updated_theta).abs().max()),
            "score_abs": float((reload_score - holdout_after).abs()),
            "control_max_abs": float((reload_control - holdout_after_control).abs().max()),
            "forecast_max_abs": float((reload_forecast - updated_forecast).abs().max()),
        }

    chain.pop("total_y_response")
    chain.pop("total_theta_response")
    chain.pop("objective_theta")
    chain.pop("score_theta")
    chain.pop("objective_y_total")
    chain.pop("score_y_total")
    train_base_signs = oracle.face_signs(train_control, frozen)
    train_updated_signs = oracle.face_signs(updated_train_control, frozen)
    holdout_base_signs = oracle.face_signs(holdout_before_control, frozen)
    holdout_updated_signs = oracle.face_signs(holdout_after_control, frozen)
    same_face_signs = all(
        bool(torch.equal(left, right))
        for left, right in (
            (train_base_signs, train_updated_signs),
            (holdout_base_signs, holdout_updated_signs),
        )
    )
    injected_supports = [
        _frozen_with_background(frozen, background(value, obs.dbz)).initial_support_mask
        for value, obs in ((theta, train_obs), (updated_theta, train_obs), (theta, holdout_obs), (updated_theta, holdout_obs))
    ]
    support_unchanged = all(bool(torch.equal(frozen.initial_support_mask, value)) for value in injected_supports)
    if not same_face_signs or not support_unchanged:
        raise RuntimeError("external-background update changed fixed FV support/face contract")
    return {
        "scope": "external learned background composition on 4x5 CPU FP64 donorcell FV; research-only",
        "production_learning_eligible": False,
        "typed_neural_prior_application": False,
        "legacy_fsoi_eligible": False,
        "truth_seeded_background": False,
        "fixed_support_and_sigma": True,
        "parameter_count": 3,
        "learning_rate": LEARNING_RATE,
        "chain_rule": chain,
        "theta_directional_prediction": theta_predicted,
        "y_directional_prediction": y_predicted,
        "theta_finite_reanalysis": theta_fd,
        "y_finite_reanalysis": y_fd,
        "theta_initial": theta.detach().tolist(),
        "theta_updated": updated_theta.tolist(),
        "train_score_before": float(train_before),
        "train_score_after": float(train_after),
        "train_score_change": float(train_after - train_before),
        "holdout_score_before": float(holdout_before),
        "holdout_score_after": float(holdout_after),
        "holdout_gain": float(holdout_before - holdout_after),
        "checkpoint_reload": reload_errors,
        "same_face_signs": same_face_signs,
        "support_digest_unchanged": support_unchanged,
        "evidence_limits": "local exact-adjoint/FD checks only; no global Hessian, typed-prior, FSOI, or production claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(checkpoint_path=args.checkpoint)
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
