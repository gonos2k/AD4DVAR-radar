"""Bounded exact-stationary FV observation-scale learning demonstration.

This is a small research probe only.  A positive scalar observation-error scale
is fitted through an exact dense/HVP stationary response on the existing 4x5 FV
fixture.  It does not create a production FSO/FSOI record or alter eligibility.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from advar import variational as v
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.transport import finite_volume_trajectory

import fv_sensitivity_probe as oracle


DTYPE = torch.float64
FD_STEPS = (1.0e-3, 5.0e-4, 2.5e-4)
LEARNING_RATE = 0.25


def _analysis_config(frozen: Any) -> Any:
    return replace(
        frozen.analysis_config,
        maximum_outer_iterations=16,
        maximum_pcg_iterations=96,
        gradient_tolerance=1.0e-10,
        step_tolerance=1.0e-12,
        pcg_relative_tolerance=1.0e-10,
    )


def _trajectory_dbz(
    frozen: Any, future: Any, support: Any, initial_dbz: Tensor
) -> Tensor:
    spec = frozen.fv_transport
    if spec is None:
        raise RuntimeError("FV fixture is missing transport")
    initial = dbz_to_echo(initial_dbz, min_dbz=-10.0)
    frames, _ = finite_volume_trajectory(
        initial,
        torch.ones_like(initial),
        spec.coefficient_limits * initial.new_tensor([0.2, -0.1, 0.15]).tanh(),
        initial.new_tensor(0.01),
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


def _window(observations: Any, frozen: Any, noise: Tensor) -> tuple[Any, Any]:
    dbz = observations.dbz + noise
    return replace(observations, dbz=dbz), replace(
        frozen, input_frames_dbz=dbz, initial_background_dbz=dbz[0]
    )


def _scaled_observations(observations: Any, theta: Tensor) -> Any:
    scale = torch.exp(theta)
    return replace(observations, std_dbz=observations.std_dbz * scale)


def _run(
    observations: Any,
    frozen: Any,
    theta: Tensor,
    *,
    polish: bool,
) -> tuple[Any, Tensor]:
    scaled = _scaled_observations(observations, theta)
    result = v.solve_analysis(scaled, frozen)

    def objective(control: Tensor, parameter: Tensor) -> Tensor:
        return v.robust_objective(
            control,
            _scaled_observations(observations, parameter),
            frozen,
        )

    if polish:
        control = oracle.polish(
            objective,
            result.control,
            theta,
            check_step=lambda start, stop: oracle.face_branch_margin(
                start, stop, frozen
            ),
        )
    else:
        control = result.control
    return result, control


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
    prediction = _forecast(control, frozen, future, support)
    difference = prediction - verification
    return (weight * difference.square()).sum() / weight.sum()


def _stationary_response(
    observations: Any,
    frozen: Any,
    future: Any,
    support: Any,
    verification: Tensor,
    weight: Tensor,
    control: Tensor,
    theta: Tensor,
) -> tuple[Tensor, float, float]:
    def objective(value: Tensor, parameter: Tensor) -> Tensor:
        return v.robust_objective(
            value,
            _scaled_observations(observations, parameter),
            frozen,
        )

    def score(value: Tensor, parameter: Tensor) -> Tensor:
        return _score(value, frozen, future, support, verification, weight)

    response, _, adjoint_residual = oracle.stationary_sensitivity(
        objective, score, control, theta
    )
    stationarity = torch.func.grad(objective, argnums=0)(control, theta)
    return response, float(stationarity.abs().max()), adjoint_residual


def _local_score_error(
    observations: Any,
    frozen: Any,
    future: Any,
    support: Any,
    verification: Tensor,
    weight: Tensor,
    control: Tensor,
    theta: Tensor,
) -> dict[str, float]:
    """Report a local stationarity-to-score error scale, not a bound."""

    def objective(value: Tensor, parameter: Tensor) -> Tensor:
        return v.robust_objective(
            value,
            _scaled_observations(observations, parameter),
            frozen,
        )

    gradient = torch.func.grad(objective, argnums=0)
    stationarity = gradient(control, theta)
    hessian = oracle.dense_hessian(gradient, control, theta)
    minimum_eigenvalue = torch.linalg.eigvalsh(hessian).min()
    if not bool(torch.isfinite(minimum_eigenvalue)) or minimum_eigenvalue <= 0:
        raise RuntimeError("holdout exact Hessian is not positive definite")
    score_gradient = torch.func.grad(
        lambda value: _score(
            value, frozen, future, support, verification, weight
        )
    )(control)
    error_scale = (
        torch.linalg.vector_norm(score_gradient)
        * torch.linalg.vector_norm(stationarity)
        / minimum_eigenvalue
    )
    return {
        "gradient_max": float(stationarity.abs().max()),
        "gradient_norm": float(torch.linalg.vector_norm(stationarity)),
        "hessian_min_eigenvalue": float(minimum_eigenvalue),
        "local_score_error_scale": float(error_scale),
    }


def run_probe(*, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    observations, frozen, future, support = oracle.make_case()
    frozen = replace(frozen, analysis_config=_analysis_config(frozen))
    coordinate = torch.arange(observations.dbz.numel(), dtype=DTYPE).reshape_as(
        observations.dbz
    )
    # Independent deterministic training/holdout noise; all perturbations stay
    # well inside the fixture's dBZ, detection, support and branch margins.
    train_noise = 0.02 * torch.sin(0.37 * coordinate)
    holdout_noise = 0.02 * torch.cos(0.61 * coordinate + 0.4)
    train_obs, train_frozen = _window(observations, frozen, train_noise)
    train_trajectory = _trajectory_dbz(frozen, future, support, observations.dbz[0])
    train_verification = train_trajectory[3:]
    holdout_initial = observations.dbz[0] + 0.08 * torch.cos(
        torch.arange(observations.dbz[0].numel(), dtype=DTYPE).reshape_as(
            observations.dbz[0]
        )
    )
    holdout_trajectory = _trajectory_dbz(
        frozen, future, support, holdout_initial
    )
    holdout_clean_observations = replace(
        observations, dbz=holdout_trajectory[:3]
    )
    holdout_obs, holdout_frozen = _window(
        holdout_clean_observations, frozen, holdout_noise
    )
    holdout_verification = holdout_trajectory[3:]
    weight = torch.ones_like(train_verification)
    holdout_weight = torch.ones_like(holdout_verification)
    theta = torch.zeros((), dtype=DTYPE)
    base_result, base_control = _run(
        train_obs, train_frozen, theta, polish=True
    )
    response, stationarity, adjoint_residual = _stationary_response(
        train_obs,
        train_frozen,
        future,
        support,
        train_verification,
        weight,
        base_control,
        theta,
    )
    theta_gradient = response.reshape(()).clone()
    delta_theta = -LEARNING_RATE * theta_gradient
    updated_theta = theta + delta_theta
    updated_result, updated_control = _run(
        train_obs, train_frozen, updated_theta, polish=True
    )

    fd_values: list[float] = []
    fd_stationarity: list[float] = []
    for step in FD_STEPS:
        plus_theta = theta + theta.new_tensor(step)
        minus_theta = theta - theta.new_tensor(step)
        plus_result, plus_control = _run(
            train_obs, train_frozen, plus_theta, polish=True
        )
        minus_result, minus_control = _run(
            train_obs, train_frozen, minus_theta, polish=True
        )
        plus_score = _score(
            plus_control, train_frozen, future, support, train_verification, weight
        )
        minus_score = _score(
            minus_control, train_frozen, future, support, train_verification, weight
        )
        fd_values.append(float((plus_score - minus_score) / (2.0 * step)))
        fd_stationarity.extend(
            [
                float(
                    torch.func.grad(
                        lambda c: v.robust_objective(
                            c, _scaled_observations(train_obs, plus_theta), train_frozen
                        )
                    )(plus_control).abs().max()
                ),
                float(
                    torch.func.grad(
                        lambda c: v.robust_objective(
                            c, _scaled_observations(train_obs, minus_theta), train_frozen
                        )
                    )(minus_control).abs().max()
                ),
            ]
        )

    base_train = _score(
        base_control, train_frozen, future, support, train_verification, weight
    )
    updated_train = _score(
        updated_control, train_frozen, future, support, train_verification, weight
    )
    holdout_base_result, holdout_base_control = _run(
        holdout_obs, holdout_frozen, theta, polish=True
    )
    holdout_updated_result, holdout_updated_control = _run(
        holdout_obs, holdout_frozen, updated_theta, polish=True
    )
    holdout_before = _score(
        holdout_base_control,
        holdout_frozen,
        future,
        support,
        holdout_verification,
        holdout_weight,
    )
    holdout_after = _score(
        holdout_updated_control,
        holdout_frozen,
        future,
        support,
        holdout_verification,
        holdout_weight,
    )
    holdout_base_diagnostics = _local_score_error(
        holdout_obs,
        holdout_frozen,
        future,
        support,
        holdout_verification,
        holdout_weight,
        holdout_base_control,
        theta,
    )
    holdout_updated_diagnostics = _local_score_error(
        holdout_obs,
        holdout_frozen,
        future,
        support,
        holdout_verification,
        holdout_weight,
        holdout_updated_control,
        updated_theta,
    )

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"log_scale": updated_theta}, path)
        reloaded = torch.load(path, weights_only=True)["log_scale"]
        reload_train_control = _run(
            train_obs, train_frozen, reloaded, polish=True
        )[1]
        reload_train_score = _score(
            reload_train_control,
            train_frozen,
            future,
            support,
            train_verification,
            weight,
        )
        reload_holdout_control = _run(
            holdout_obs, holdout_frozen, reloaded, polish=True
        )[1]
        reload_holdout_score = _score(
            reload_holdout_control,
            holdout_frozen,
            future,
            support,
            holdout_verification,
            holdout_weight,
        )
        updated_holdout_forecast = _forecast(
            holdout_updated_control, holdout_frozen, future, support
        )
        reload_holdout_forecast = _forecast(
            reload_holdout_control, holdout_frozen, future, support
        )
        reload_train_score_error = float((reload_train_score - updated_train).abs())
        reload_holdout_score_error = float(
            (reload_holdout_score - holdout_after).abs()
        )
        reload_holdout_control_max_abs = float(
            (reload_holdout_control - holdout_updated_control).abs().max()
        )
        reload_holdout_forecast_max_abs = float(
            (reload_holdout_forecast - updated_holdout_forecast).abs().max()
        )
    else:
        reload_train_score_error = None
        reload_holdout_score_error = None
        reload_holdout_control_max_abs = None
        reload_holdout_forecast_max_abs = None

    signs = [
        oracle.face_signs(control, train_frozen)
        for control in (base_control, updated_control)
    ]
    branch_margin = oracle.face_branch_margin(base_control, updated_control, train_frozen)
    fd_error = abs(float(theta_gradient) - fd_values[-1])
    fd_step_error = abs(fd_values[-1] - fd_values[-2])
    train_change = float(updated_train - base_train)
    holdout_gain = float(holdout_before - holdout_after)
    score_roundoff_scale = torch.finfo(DTYPE).eps * max(
        abs(float(holdout_before)), abs(float(holdout_after)), 1.0e-30
    )
    # Local first-order error estimate with a safety factor, not a rigorous bound.
    holdout_error_budget = 10.0 * (
        holdout_base_diagnostics["local_score_error_scale"]
        + holdout_updated_diagnostics["local_score_error_scale"]
        + (reload_holdout_score_error or 0.0)
        + score_roundoff_scale
    )
    return {
        "scope": "4x5 CPU FP64 donorcell exact stationary observation-scale learning; research-only",
        "production_learning_eligible": False,
        "stationarity_verified": False,
        "control_count": base_control.numel(),
        "initial_log_scale": float(theta),
        "updated_log_scale": float(updated_theta),
        "parameter_gradient": float(theta_gradient),
        "predicted_impact": float(theta_gradient * delta_theta),
        "train_score_before": float(base_train),
        "train_score_after": float(updated_train),
        "train_score_change": train_change,
        "holdout_score_before": float(holdout_before),
        "holdout_score_after": float(holdout_after),
        "holdout_gain": holdout_gain,
        "score_roundoff_scale": score_roundoff_scale,
        "holdout_base_diagnostics": holdout_base_diagnostics,
        "holdout_updated_diagnostics": holdout_updated_diagnostics,
        "holdout_error_budget": holdout_error_budget,
        "holdout_gain_above_error_budget": (
            holdout_gain > holdout_error_budget if checkpoint_path is not None else None
        ),
        "numerical_error_scope": "local linearized estimate, not a rigorous neighborhood bound",
        "checkpoint_scope": "parameter-only reload with the same prescribed in-memory contract",
        "finite_difference_steps": FD_STEPS,
        "finite_difference_gradients": tuple(fd_values),
        "finite_difference_last_error": fd_error,
        "finite_difference_step_error": fd_step_error,
        "finite_difference_stationarity_max": max(fd_stationarity),
        "stationarity_max": stationarity,
        "adjoint_relative_residual": adjoint_residual,
        "branch_margin": branch_margin,
        "same_face_signs": bool(torch.equal(signs[0], signs[1])),
        "base_solver_reason": base_result.reason,
        "updated_solver_reason": updated_result.reason,
        "holdout_base_solver_reason": holdout_base_result.reason,
        "holdout_updated_solver_reason": holdout_updated_result.reason,
        "reload_train_score_error": reload_train_score_error,
        "reload_holdout_score_error": reload_holdout_score_error,
        "reload_holdout_control_max_abs": reload_holdout_control_max_abs,
        "reload_holdout_forecast_max_abs": reload_holdout_forecast_max_abs,
        "curvature": "exact_robust_hessian",
    }


if __name__ == "__main__":
    import argparse
    import json

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
