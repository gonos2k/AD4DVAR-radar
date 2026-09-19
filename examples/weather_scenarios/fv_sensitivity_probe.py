"""Small CPU/FP64 oracle for FV observation sensitivity and reanalysis impact.

Uses the actual robust objective and FV forecast. Dense Newton refinement is a
bounded verification oracle, not a replacement solver or a production FSO API.
Only the fully detected, uncapped, donorcell synthetic case below is supported.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from advar import variational as v
from advar.matrix_free import hvp, pcg
from advar.nowcast import NowcastConfig
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.transport import finite_volume_trajectory
from advar.fv_sensitivity import (
    compute_fv_observation_response,
    face_signs,
    face_branch_margin,
)


def make_case():
    y, x = torch.meshgrid(
        torch.arange(5, dtype=torch.float64),
        torch.arange(6, dtype=torch.float64),
        indexing="ij",
    )
    basis = torch.stack((y, -x, -0.5 * (x.square() + y.square())))
    limits = y.new_tensor([0.15, 0.15, 0.02])
    initial_dbz = y.new_tensor(
        [
            [18, 21, 19, 23, 20],
            [22, 18, 24, 19, 21],
            [19, 24, 20, 22, 18],
            [21, 20, 23, 18, 22],
        ]
    )
    initial = dbz_to_echo(initial_dbz, min_dbz=-10.0)
    edges = tuple(
        y.new_full((n,), value) for n, value in zip((4, 4, 5, 5), (60, 70, 80, 90))
    )
    ones = tuple(torch.ones_like(edge) for edge in edges)
    boundary = tuple(
        (
            tuple(e * (1 + 0.01 * i) for e in edges),
            tuple(e * (1 + 0.01 * (i + 1)) for e in edges),
        )
        for i in range(8)
    )
    support = ((ones, ones),) * 8
    spec = v.FVAnalysisTransport(
        psi_basis=basis,
        coefficient_limits=limits,
        substeps_per_interval=2,
        spacing_yx=(10.0, 10.0),
        boundary_echo=boundary[:4],
        boundary_support=support[:4],
        reconstruction="donorcell",
        replay=True,
    )
    # Synthetic truth creates observations only; never seed the inverse with it.
    truth, _ = finite_volume_trajectory(
        initial,
        torch.ones_like(initial),
        limits * y.new_tensor([0.2, -0.1, 0.15]).tanh(),
        y.new_tensor(0.01),
        psi_basis=basis,
        leads=2,
        substeps_per_interval=2,
        interval_seconds=60.0,
        spacing_yx=(10.0, 10.0),
        boundary_echo=boundary[:4],
        boundary_support=support[:4],
        reconstruction="donorcell",
    )
    obs, frozen = v.prepare_analysis(
        echo_to_dbz(truth, min_dbz=-10.0),
        nowcast_config=NowcastConfig(
            interval_minutes=1, horizon_minutes=1, min_dbz=-10.0
        ),
        analysis_config=v.AnalysisConfig(field_smoothness_weight=0.0),
        observation_std_dbz=0.1,
        fv_transport=spec,
    )
    return obs, frozen, boundary[4:], support[4:]


def dense_hessian(gradient, control, observations):
    if control.numel() > 32:
        raise ValueError("dense verification oracle is limited to 32 controls")
    matrix = torch.stack(
        [
            torch.func.jvp(lambda c: gradient(c, observations), (control,), (e,))[1]
            for e in torch.eye(control.numel(), dtype=control.dtype)
        ],
        dim=1,
    )
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("verification Hessian is not finite")
    tolerance = 128 * torch.finfo(matrix.dtype).eps * matrix.abs().max()
    if (matrix - matrix.T).abs().max() > tolerance:
        raise ValueError("verification Hessian is not symmetric")
    return matrix


def polish(objective, control, observations, *, check_step=None):
    """Solve the local gradient root; this is not a global cost minimizer.

    The merit function is ||grad J||² / 2. Cost subtraction cannot resolve
    Newton improvements below its evaluation precision. Positive curvature
    and branch checks constrain this small, already initialized oracle.
    """
    gradient = torch.func.grad(objective, argnums=0)
    control = control.detach().clone()
    for _ in range(8):
        if not bool(torch.isfinite(objective(control, observations))):
            raise RuntimeError("verification objective is not finite")
        g = gradient(control, observations)
        if not bool(torch.isfinite(g).all()):
            raise RuntimeError("verification gradient is not finite")
        H = dense_hessian(gradient, control, observations)
        # A positive Hessian is checked, not manufactured with damping.
        factor = torch.linalg.cholesky(H)
        if g.abs().max() < 1e-10:
            return control
        step = torch.cholesky_solve(-g[:, None], factor)[:, 0]
        for power in range(16):
            scale = 2.0 ** (-power)
            trial = control + scale * step
            if check_step is not None:
                try:
                    check_step(control, trial)
                except ValueError:
                    continue
            trial_gradient = gradient(trial, observations)
            if (
                bool(torch.isfinite(objective(trial, observations)))
                and bool(torch.isfinite(trial_gradient).all())
                and (trial_gradient.norm() / g.norm()).square() <= 1 - 2e-4 * scale
            ):
                control = trial.detach()
                break
        else:
            raise RuntimeError("verification Newton refinement did not decrease")
    final_gradient = gradient(control, observations)
    torch.linalg.cholesky(dense_hessian(gradient, control, observations))
    if not bool(torch.isfinite(final_gradient).all()) or final_gradient.abs().max() >= 1e-10:
        raise RuntimeError("verification point is not sufficiently stationary")
    return control


def stationary_sensitivity(objective, score, control, observations):
    """E_y - (D_y grad_c J)^T H^{-T} E_c, with an exact robust HVP."""
    gradient = torch.func.grad(objective, argnums=0)
    g = gradient(control, observations)
    if not bool(torch.isfinite(g).all()) or g.abs().max() >= 1e-8:
        raise ValueError("stationary sensitivity requires a refined point")
    rhs, direct = torch.func.grad(score, argnums=(0, 1))(control, observations)
    product = lambda direction: hvp(
        lambda c: objective(c, observations), control, direction
    )
    solve = pcg(product, rhs, rtol=1e-10, max_iterations=4 * control.numel())
    if not solve.converged:
        raise RuntimeError("exact-Hessian adjoint did not converge")
    _, pullback = torch.func.vjp(lambda y: gradient(control, y), observations)
    return direct - pullback(solve.solution)[0], direct, solve.relative_residual


def run_probe():
    started = time.monotonic()
    obs, frozen, future, future_support = make_case()
    assert bool(obs.detected_mask.all())
    assert bool(
        (
            (obs.dbz > frozen.nowcast_config.min_dbz)
            & (obs.dbz < frozen.nowcast_config.max_dbz)
        ).all()
    )
    result = v.solve_analysis(obs, frozen)

    # In this fully detected interior domain prepare_analysis uses B = y[0].
    # Masks, uncertainties, basis, and prescribed boundaries remain fixed.
    def contract(y):
        return replace(frozen, initial_background_dbz=y[0])

    def objective(c, y):
        return v.robust_objective(c, replace(obs, dbz=y), contract(y))

    def forecast(c, y):
        trajectory = v.forecast_fv_analysis(
            c,
            contract(y),
            leads=2,
            boundary_start_interval=2,
            boundary_echo=future,
            boundary_support=future_support,
        )
        return echo_to_dbz(trajectory.frames_linear[1:], min_dbz=-10.0)

    check_step = lambda start, stop: face_branch_margin(start, stop, frozen)
    control = polish(objective, result.control, obs.dbz, check_step=check_step)
    signs = face_signs(control, frozen)
    polish_box_margin = face_branch_margin(result.control, control, frozen)
    prediction = forecast(control, obs.dbz).detach()
    weights = torch.linspace(
        0.2, 1.0, prediction.numel(), dtype=control.dtype
    ).reshape_as(prediction)
    verification = prediction + 0.3 * torch.cos(
        torch.arange(prediction.numel(), dtype=control.dtype)
    ).reshape_as(prediction)

    def score(c, y):
        return (
            weights * (forecast(c, y) - verification).square()
        ).sum() / weights.sum()

    gradient = torch.func.grad(objective, argnums=0)
    H = dense_hessian(gradient, control, obs.dbz)
    torch.linalg.cholesky(H)
    eigenvalues = torch.linalg.eigvalsh(H)
    sensitivity, direct, linear_residual = stationary_sensitivity(
        objective, score, control, obs.dbz
    )
    # Dense solve is independent of the PCG iteration used above.
    rhs = torch.func.grad(score, argnums=0)(control, obs.dbz)
    _, pullback = torch.func.vjp(lambda y: gradient(control, y), obs.dbz)
    dense_sensitivity = direct - pullback(torch.linalg.solve(H.T, rhs))[0]
    frozen_objective = lambda c, y: v.robust_objective(c, replace(obs, dbz=y), frozen)
    frozen_score = lambda c, y: score(c, obs.dbz)
    partial, _, _ = stationary_sensitivity(
        frozen_objective, frozen_score, control, obs.dbz
    )
    direction = torch.sin(
        torch.arange(obs.dbz.numel(), dtype=control.dtype)
    ).reshape_as(obs.dbz)
    linear = (sensitivity * direction).sum()
    measurements = []
    for h in (0.01, 0.005, 0.0025):
        plus_y, minus_y = obs.dbz + h * direction, obs.dbz - h * direction
        plus, minus = (
            polish(objective, control, plus_y, check_step=check_step),
            polish(objective, control, minus_y, check_step=check_step),
        )
        if not (
            torch.equal(face_signs(plus, frozen), signs)
            and torch.equal(face_signs(minus, frozen), signs)
        ):
            raise RuntimeError("reanalysis left the donorcell sign branch")
        change = score(plus, plus_y) - score(control, obs.dbz)
        centered = (score(plus, plus_y) - score(minus, minus_y)) / (2 * h)
        box_margin = min(
            face_branch_margin(control, plus, frozen),
            face_branch_margin(control, minus, frozen),
        )
        measurements.append(
            dict(
                h=h,
                face_box_margin=box_margin,
                actual_change=float(change),
                linear_change=float(h * linear),
                taylor_error=float((change - h * linear).abs()),
                central_error=float((centered - linear).abs()),
                gradient_max=float(
                    torch.maximum(
                        gradient(plus, plus_y).abs().max(),
                        gradient(minus, minus_y).abs().max(),
                    )
                ),
            )
        )
    irls = v.freeze_irls_weights(control, obs, frozen)
    residual = lambda c: v.residual_vector(c, obs, irls)
    J = torch.stack(
        [
            torch.func.jvp(residual, (control,), (e,))[1]
            for e in torch.eye(control.numel(), dtype=control.dtype)
        ],
        dim=1,
    )
    response_arguments = dict(
        verification_dbz=verification,
        metric_weight=weights,
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future,
        boundary_support=future_support,
        background_dependency="first_observation",
    )
    response = compute_fv_observation_response(
        control, obs, frozen, **response_arguments
    )
    gn_adjoint = torch.linalg.solve(J.T @ J, rhs)
    gn_dense = direct - pullback(gn_adjoint)[0]
    gn_linear = (response.sensitivity_dbz * direction).sum()
    for measurement in measurements:
        estimate = float(measurement["h"] * gn_linear)
        measurement["gn_linear_change"] = estimate
        measurement["gn_taylor_error"] = abs(measurement["actual_change"] - estimate)
    try:
        compute_fv_observation_response(
            control, obs, frozen, **response_arguments, maximum_normal_products=1
        )
    except RuntimeError as error:
        if "budget exhausted" not in str(error):
            raise
    else:
        raise AssertionError("normal-product budget did not reject")
    fixed_response = compute_fv_observation_response(
        control, obs, frozen, **dict(response_arguments, background_dependency="frozen")
    )
    _, fixed_pullback = torch.func.vjp(
        lambda y: torch.func.grad(frozen_objective, argnums=0)(control, y), obs.dbz
    )
    fixed_dense = -fixed_pullback(gn_adjoint)[0]
    data = J[: obs.dbz.numel()]
    field_size = frozen.active_field_index.numel()
    field_space = torch.linalg.qr(data[:, :field_size], mode="reduced").Q
    dynamics = data[:, field_size:]
    conditional_dynamics = dynamics - field_space @ (field_space.T @ dynamics)
    singular_values = torch.linalg.svdvals(conditional_dynamics)
    threshold = (
        max(conditional_dynamics.shape)
        * torch.finfo(control.dtype).eps
        * singular_values.max()
    )
    report = dict(
        scope="CPU FP64 4x5 same-model donorcell stationary-point oracle; no production FSO eligibility or learning claim",
        background_dependency="B=y[0]; direct forecast term plus implicit optimum term",
        branch_scope="positive face margins over boxes joining evaluated controls; no unknown stationary-path enclosure claim",
        polish_face_box_margin=polish_box_margin,
        field_conditioned_data_dynamics_singular_values=singular_values.tolist(),
        field_conditioned_data_dynamics_rank=int((singular_values > threshold).sum()),
        data_rank_scope="local IRLS-weighted observation Jacobian, projected off initial-field columns; no control-prior rows",
        scored_pixels_per_lead=20,
        control_count=control.numel(),
        gn_api_curvature=response.curvature,
        gn_api_score_error=abs(response.score - float(score(control, obs.dbz))),
        gn_api_vs_dense_max=float((response.sensitivity_dbz - gn_dense).abs().max()),
        gn_api_fixed_background_vs_dense_max=float(
            (fixed_response.sensitivity_dbz - fixed_dense).abs().max()
        ),
        gn_api_vs_exact_relative=float(
            (response.sensitivity_dbz - sensitivity).norm() / sensitivity.norm()
        ),
        gn_api_normal_products=response.normal_products,
        gn_api_adjoint_relative_residual=response.adjoint_relative_residual,
        gn_api_direct_max_error=float(
            (response.direct_sensitivity_dbz - direct).abs().max()
        ),
        gradient_max=float(gradient(control, obs.dbz).abs().max()),
        hessian_min_eigenvalue=float(eigenvalues.min()),
        hessian_condition_number=float(eigenvalues.max() / eigenvalues.min()),
        local_linearized_control_error_estimate=float(
            gradient(control, obs.dbz).norm() / eigenvalues.min()
        ),
        hessian_symmetry_max=float((H - H.T).abs().max()),
        hessian_vs_irls_gn_relative=float((H - J.T @ J).norm() / H.norm()),
        adjoint_relative_residual=linear_residual,
        pcg_vs_dense_sensitivity_max=float(
            (sensitivity - dense_sensitivity).abs().max()
        ),
        total_vs_frozen_background_max=float((sensitivity - partial).abs().max()),
        direct_first_frame_norm=float(direct[0].norm()),
        later_frame_partial_difference_max=float(
            (sensitivity[1:] - partial[1:]).abs().max()
        ),
        directional_derivative=float(linear),
        perturbations=measurements,
        elapsed_seconds=time.monotonic() - started,
    )
    return report


def run_solver_response():
    """Existing matrix-free solver followed by GN response; no dense polish."""
    started = time.monotonic()
    observations, frozen, future, support = make_case()
    config = replace(
        frozen.analysis_config,
        maximum_outer_iterations=16,
        maximum_pcg_iterations=96,
        gradient_tolerance=1e-10,
        step_tolerance=1e-12,
        pcg_relative_tolerance=1e-10,
    )
    frozen = replace(frozen, analysis_config=config)
    result = v.solve_analysis(observations, frozen)
    shape = (2, *observations.dbz.shape[-2:])
    response = compute_fv_observation_response(
        result.control,
        observations,
        frozen,
        verification_dbz=observations.dbz.new_full(shape, 20.0),
        metric_weight=observations.dbz.new_ones(shape),
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future,
        boundary_support=support,
        background_dependency="first_observation",
    )
    return dict(
        scope="existing matrix-free solver; no dense polish; same-model CPU FP64 donorcell; GN approximate response",
        reason=result.reason,
        selected_gradient_norm=result.selected_gradient_norm,
        outer_iterations=result.outer_iterations,
        pcg_iterations=result.pcg_iterations,
        initial_objective=result.initial_objective,
        final_objective=result.final_objective,
        stationarity_verified=result.stationarity_verified,
        response_gradient_max=response.gradient_max,
        normal_products=response.normal_products,
        adjoint_relative_residual=response.adjoint_relative_residual,
        score=response.score,
        sensitivity_norm=float(response.sensitivity_dbz.norm()),
        elapsed_seconds=time.monotonic() - started,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--solver-only",
        action="store_true",
        help="run the shared solver and GN response without a dense oracle",
    )
    args = parser.parse_args()
    report = run_solver_response() if args.solver_only else run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
