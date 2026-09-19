"""Research FV observation response with selectable local curvature.

This module does not construct legacy FSO/FSOI or learning-eligibility records.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import math
import logging
from typing import Literal, cast

import torch
from torch import Tensor

from .matrix_free import pcg
from .physics import echo_to_dbz
from .transport import BoundarySchedule, face_volume_fluxes
from .variational import (
    AnalysisObservations,
    FVAnalysisResult,
    FrozenOuterState,
    _evaluate_control,
    _initial_analysis_dbz,
    _validate_control,
    _validate_observations,
    forecast_fv_analysis,
    freeze_irls_weights,
    residual_vector,
    robust_objective,
)


@dataclass(frozen=True)
class FVObservationResponse:
    """Stationary response using the selected local curvature operator.

    The default normal solve uses frozen IRLS GN.  Exact mode uses matrix-free
    HVPs of the robust objective and therefore remains a local implicit
    derivative only; neither mode makes an observational-rank or learning
    claim.
    """

    sensitivity_dbz: Tensor
    direct_sensitivity_dbz: Tensor
    score: float
    gradient_max: float
    normal_products: int
    adjoint_relative_residual: float
    face_margin: float
    curvature: Literal["irls_gauss_newton", "exact_robust_hessian"] = (
        "irls_gauss_newton"
    )


def _face_rows(frozen: FrozenOuterState) -> Tensor:
    spec = frozen.fv_transport
    if spec is None or spec.reconstruction != "donorcell":
        raise ValueError("this oracle supports donorcell only")
    basis = spec.psi_basis
    rows = torch.cat(
        (
            (basis[:, 1:, :] - basis[:, :-1, :]).flatten(1),
            -(basis[:, :, 1:] - basis[:, :, :-1]).flatten(1),
        ),
        dim=1,
    ).T
    return rows


def _face_flux_and_roundoff(
    coefficients: Tensor, radius: Tensor, frozen: FrozenOuterState
) -> tuple[Tensor, Tensor]:
    assert frozen.fv_transport is not None
    basis = frozen.fv_transport.psi_basis
    psi = torch.einsum("k,kij->ij", coefficients, basis)
    qx, qy = face_volume_fluxes(psi)
    flux = torch.cat((qx.flatten(), qy.flatten()))
    # Match the backend's vertex sum THEN difference: small fluxes can be
    # ill-resolved even when the algebraically differenced basis looks benign.
    vertices = torch.einsum("k,kij->ij", coefficients.abs() + radius, basis.abs())
    scale = torch.cat(
        (
            (vertices[1:] + vertices[:-1]).flatten(),
            (vertices[:, 1:] + vertices[:, :-1]).flatten(),
        )
    )
    allowance = (
        64 * (coefficients.numel() + 1) * torch.finfo(coefficients.dtype).eps * scale
    )
    return flux, allowance


def face_signs(control: Tensor, frozen: FrozenOuterState) -> Tensor:
    """Reject coefficient-sensitive ties; structural zero rows remain harmless."""
    rows = _face_rows(frozen)
    assert frozen.fv_transport is not None
    size = frozen.active_field_index.numel()
    coefficients = frozen.fv_transport.coefficient_limits * control[size:-1].tanh()
    flux, roundoff = _face_flux_and_roundoff(
        coefficients, torch.zeros_like(coefficients), frozen
    )
    variable = (rows != 0).any(dim=1)
    if (
        not bool(torch.isfinite(flux).all())
        or not bool(torch.isfinite(roundoff).all())
        or bool(torch.any(variable & (flux.abs() <= roundoff)))
    ):
        raise ValueError("coefficient-sensitive face sign tie")
    return flux[variable].sign()


def face_branch_margin(start: Tensor, stop: Tensor, frozen: FrozenOuterState) -> float:
    """Sufficient donorcell sign margin over the latent box between two points.

    tanh is monotone and each face flux is linear in physical coefficients.
    This bounds the entire box, not only samples along its diagonal. It does
    not assert that an unknown stationary solution curve stays in that box.
    """
    rows = _face_rows(frozen)
    assert frozen.fv_transport is not None
    size = frozen.active_field_index.numel()
    limits = frozen.fv_transport.coefficient_limits
    a, b = limits * start[size:-1].tanh(), limits * stop[size:-1].tanh()
    middle, radius = (a + b) * 0.5, (a - b).abs() * 0.5
    flux, roundoff = _face_flux_and_roundoff(middle, radius, frozen)
    margin = flux.abs() - rows.abs() @ radius - roundoff
    variable = (rows != 0).any(dim=1)
    if not bool(variable.any()):
        return float("inf")
    if not bool(torch.isfinite(margin).all()) or bool((margin[variable] <= 0).any()):
        raise ValueError("latent box can cross a coefficient-sensitive face sign")
    return float(margin[variable].min())


def _known_donorcell_step(
    known: Tensor,
    qx: Tensor,
    qy: Tensor,
    support_edges: tuple[Tensor, Tensor, Tensor, Tensor],
) -> Tensor:
    """Propagate a conservative fixed-known mask through one donorcell step."""
    left, right, bottom, top = support_edges
    incoming = known.clone()
    positive_x = qx > 0
    negative_x = qx < 0
    positive_y = qy > 0
    negative_y = qy < 0
    incoming[:, 1:] &= (~positive_x[:, 1:-1]) | known[:, :-1]
    incoming[:, :-1] &= (~negative_x[:, 1:-1]) | known[:, 1:]
    incoming[1:, :] &= (~positive_y[1:-1, :]) | known[:-1, :]
    incoming[:-1, :] &= (~negative_y[1:-1, :]) | known[1:, :]
    # A fractional trace remains an unknown contribution even when it is
    # within roundoff of one; only the exact fixed value one is known.
    incoming[:, 0] &= (~positive_x[:, 0]) | (left == 1)
    incoming[:, -1] &= (~negative_x[:, -1]) | (right == 1)
    incoming[0, :] &= (~positive_y[0, :]) | (bottom == 1)
    incoming[-1, :] &= (~negative_y[-1, :]) | (top == 1)
    return incoming


def _known_support_frames(
    frozen: FrozenOuterState,
    *,
    leads: int,
    boundary_support: BoundarySchedule,
    psi_coefficients: Tensor,
) -> Tensor:
    """Return support cells stable under the current donorcell sign stencil."""
    assert frozen.fv_transport is not None
    qx, qy = face_volume_fluxes(
        torch.einsum("k,kij->ij", psi_coefficients, frozen.fv_transport.psi_basis)
    )
    known = frozen.initial_support_mask.clone()
    substeps = frozen.fv_transport.substeps_per_interval

    def advance(schedule: BoundarySchedule, known_state: Tensor) -> Tensor:
        state = known_state
        for stages in schedule:
            stage0, stage1 = stages
            state1 = _known_donorcell_step(state, qx, qy, stage0)
            state2 = _known_donorcell_step(state1, qx, qy, stage1)
            state = state & state2
        return state

    frames = [known]
    analysis_interval = substeps
    for interval in range(2):
        start = interval * analysis_interval
        stop = start + analysis_interval
        known = advance(frozen.fv_transport.boundary_support[start:stop], known)
        frames.append(known)
    for lead in range(leads):
        start = lead * substeps
        stop = start + substeps
        known = advance(boundary_support[start:stop], known)
        frames.append(known)
    return torch.stack(frames)


def _initial_observation_diagonal(
    control: Tensor, observations: AnalysisObservations, frozen: FrozenOuterState
) -> Tensor:
    """Positive initial-observation scaling, not an approximation used in J.

    The initial field transform is pointwise, so a ones JVP is its diagonal.
    Later observations, robust weights and error correlations remain in the
    exact operator; this cheap preconditioner need not reproduce them.
    """
    index = frozen.active_field_index
    field = torch.zeros_like(frozen.initial_background_dbz).flatten().scatter(
        0, index, control[:index.numel()]).reshape_as(frozen.initial_background_dbz)
    derivative = torch.func.jvp(lambda x: _initial_analysis_dbz(x, frozen),
                                (field,), (torch.ones_like(field),))[1]
    active = observations.valid_mask[0] & observations.detected_mask[0]
    sigma = torch.where(active, observations.std_dbz[0], torch.ones_like(derivative))
    derivative = torch.where(active, derivative, torch.zeros_like(derivative))
    precision = 1 + (derivative / sigma).square() * observations.quality_weight[0]
    diagonal = torch.ones_like(control)
    diagonal[:index.numel()] = precision.flatten()[index]
    if not bool(torch.isfinite(diagonal).all()) or not bool((diagonal > 0).all()):
        raise ValueError("nonfinite FV initial-observation preconditioner")
    return diagonal.detach()


def refine_fv_stationarity(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    gradient_tolerance: float = 1e-9,
    maximum_iterations: int = 4,
    maximum_normal_products: int = 64,
) -> tuple[Tensor, list[dict[str, float | int]]]:
    """Refine a nearby FV stationary point using exact matrix-free Newton steps.

    Acceptance decreases ||grad J||, so it remains meaningful when changes in
    separately evaluated objective values cannot be resolved. This is a local
    root solve, not a global minimum or implicit-path certificate.
    """
    if control.dtype != torch.float64 or control.device.type != "cpu":
        raise ValueError("FV root refinement requires CPU FP64")
    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be positive and finite")
    if maximum_iterations <= 0 or maximum_normal_products < 3:
        raise ValueError("FV root refinement requires positive iteration budgets")
    _validate_control(control, frozen)
    _validate_observations(observations)
    current = control.detach().clone()
    face_branch_margin(current, current, frozen)
    gradient = torch.func.grad(lambda c: robust_objective(c, observations, frozen))
    records: list[dict[str, float | int]] = []
    for iteration in range(maximum_iterations + 1):
        cost, _ = _evaluate_control(current, observations, frozen)
        g = gradient(current)
        norm = float(torch.linalg.vector_norm(g))
        logging.getLogger(__name__).info("FV root iteration %d: gradient norm %.17g", iteration, norm)
        if not bool(torch.isfinite(cost)) or not math.isfinite(norm):
            raise ValueError("nonfinite FV stationary residual")
        if norm <= gradient_tolerance:
            return current, records
        if iteration == maximum_iterations:
            break
        products = 0

        def normal(direction: Tensor) -> Tensor:
            nonlocal products
            if products >= maximum_normal_products:
                raise RuntimeError("FV refinement normal-product budget exhausted")
            products += 1
            return torch.func.jvp(gradient, (current,), (direction,))[1]

        # Resolve the Newton equation more tightly than the requested outer
        # residual, without solving tiny right-hand sides to an unrelated scale.
        linear_atol = 0.1 * gradient_tolerance
        diagonal = _initial_observation_diagonal(current, observations, frozen)
        step = pcg(normal, -g, preconditioner=lambda x: x/diagonal,
                   rtol=1e-6, atol=linear_atol,
                   max_iterations=maximum_normal_products - 2)
        if not step.converged:
            raise ValueError("FV stationary Newton system did not converge")
        trial_norm = math.inf
        for backtrack in range(12):
            alpha = 0.5 ** backtrack
            candidate = current + alpha * step.solution
            # An unknown implicit curve is not enclosed by this trial-step box.
            try:
                margin = face_branch_margin(current, candidate, frozen)
                candidate_cost, _ = _evaluate_control(candidate, observations, frozen)
            except ValueError:
                continue
            trial_norm = float(torch.linalg.vector_norm(gradient(candidate)))
            if bool(torch.isfinite(candidate_cost)) and math.isfinite(trial_norm) and (
                trial_norm <= (1 - 1e-4 * alpha) * norm
            ):
                records.append(dict(iteration=iteration, gradient_before=norm,
                                    gradient_after=trial_norm, step_scale=alpha,
                                    normal_products=products,
                                    linear_relative_residual=step.relative_residual,
                                    linear_absolute_tolerance=linear_atol,
                                    face_margin=margin))
                current = candidate.detach()
                break
        else:
            raise ValueError(
                f"FV stationary residual did not decrease: gradient_norm={norm}; "
                f"last_trial_norm={trial_norm}; steps={records}"
            )
    raise ValueError(
        f"FV stationary refinement iteration budget exhausted: gradient_norm={norm}; steps={records}"
    )


def verify_fv_stationarity(
    result: FVAnalysisResult,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> tuple[FVAnalysisResult, dict[str, object]]:
    """Verify a local first-order CPU FP64 donorcell result, not a minimum.

    The full AD gradient is tested against the configured tolerance. Independent
    central residual differences sample one spatial direction and each dynamics
    control. Frozen IRLS weights include the same prior and weighting terms as
    the robust objective. No Hessian or unknown solution-path claim is made.
    """
    fv = frozen.fv_transport
    if not isinstance(result, FVAnalysisResult) or fv is None:
        raise ValueError("FV stationarity verification requires an FV result")
    control = result.control
    if (
        control.dtype != torch.float64
        or control.device.type != "cpu"
        or fv.reconstruction != "donorcell"
    ):
        raise ValueError("local FV verification requires CPU FP64 donorcell")
    _validate_control(control, frozen)
    _validate_observations(observations)
    if not bool(observations.valid_mask.all() & observations.detected_mask.all()):
        raise ValueError("local FV verification requires fully detected observations")
    if not bool(frozen.initial_support_mask.all()) or any(
        not bool((edge == 1).all())
        for stages in fv.boundary_support
        for edges in stages
        for edge in edges
    ):
        raise ValueError("local FV verification requires full prescribed support")

    def evaluate(value: Tensor) -> Tensor:
        cost, trajectory = _evaluate_control(value, observations, frozen)
        prediction = echo_to_dbz(
            trajectory.frames_linear,
            min_dbz=frozen.nowcast_config.min_dbz,
        )
        if not bool(torch.isfinite(cost)) or not bool(
            (trajectory.frames_linear > 0).all()
            & (prediction > frozen.nowcast_config.min_dbz).all()
            & (prediction < frozen.nowcast_config.max_dbz).all()
        ):
            raise ValueError("local verification left the finite positive interior")
        if trajectory.support_frames is None or not bool(
            (trajectory.support_frames >= 1 - 128 * torch.finfo(control.dtype).eps).all()
        ):
            raise ValueError("local verification left full trajectory support")
        return cost

    cost = evaluate(control)
    gradient = torch.func.grad(robust_objective)(control, observations, frozen)
    norm = float(torch.linalg.vector_norm(gradient))
    tolerance = frozen.analysis_config.gradient_tolerance
    if not math.isfinite(norm) or norm > tolerance:
        raise ValueError("FV robust gradient exceeds the stationarity tolerance")
    # The supplied record must describe this exact analysis contract.
    _, trajectory = _evaluate_control(control, observations, frozen)
    for name in (
        "frames_linear",
        "displacement_yx",
        "log_growth_per_step",
        "psi_coefficients",
        "support_frames",
    ):
        actual = getattr(result.trajectory, name)
        expected = getattr(trajectory, name)
        if (actual is None) != (expected is None) or (
            actual is not None and not torch.equal(actual, expected)
        ):
            raise ValueError(
                f"FV result trajectory {name} does not match the supplied contract"
            )
    size = frozen.active_field_index.numel()
    spatial = torch.zeros_like(control)
    spatial[:size] = torch.cos(torch.arange(size, dtype=control.dtype))
    spatial = spatial / spatial.norm()
    directions = [("spatial_cosine", spatial)]
    for index in range(size, control.numel()):
        direction = torch.zeros_like(control)
        direction[index] = 1
        directions.append((f"dynamics_{index-size}", direction))
    linearized = freeze_irls_weights(control, observations, frozen)
    residual_fn = lambda value: residual_vector(value, observations, linearized)
    vjp_result = torch.func.vjp(residual_fn, control)
    residual = cast(Tensor, vjp_result[0])
    pullback = cast(Callable[[Tensor], tuple[Tensor, ...]], vjp_result[1])
    frozen_residual_gradient = pullback(residual)[0]
    epsilon = torch.finfo(control.dtype).eps
    checks: list[dict[str, object]] = []
    for name, direction in directions:
        tangent = torch.func.jvp(residual_fn, (control,), (direction,))[1]
        scale = float(tangent.norm())
        if not math.isfinite(scale) or scale == 0:
            raise ValueError(f"unresolved zero/nonfinite residual direction: {name}")
        frozen_slope = float(frozen_residual_gradient @ direction)
        robust_slope = float(gradient @ direction)
        chain_slope = float(residual @ tangent)
        allowance = 256 * epsilon * float(
            (residual * tangent).abs().sum()
            + (frozen_residual_gradient * direction).abs().sum()
            + (gradient * direction).abs().sum()
        )
        if max(
            abs(chain_slope - frozen_slope), abs(robust_slope - frozen_slope)
        ) > allowance:
            raise ValueError(f"inconsistent robust/frozen gradient direction: {name}")
        for h in (1e-3, 1e-4, 1e-5, 1e-6):
            try:
                margin = face_branch_margin(
                    control - h * direction,
                    control + h * direction,
                    frozen,
                )
                samples = []
                with torch.no_grad():
                    for step in (h, -h, h / 2, -h / 2):
                        candidate = control + step * direction
                        evaluate(candidate)
                        samples.append(residual_fn(candidate))
                plus, minus, half_plus, half_minus = samples
                full = (plus - minus) / (2 * h)
                half = (half_plus - half_minus) / h
                relative_error = float((half - tangent).norm()) / scale
                truncation = float((half - full).norm()) / (3 * scale)
                roundoff = 256 * epsilon * max(
                    float(x.norm()) for x in samples
                ) / (h / 2 * scale)
                if relative_error <= epsilon ** (1 / 3) and relative_error <= (
                    4 * truncation + roundoff
                ):
                    checks.append(
                        dict(
                            direction=name,
                            step=h,
                            robust_slope=robust_slope,
                            frozen_slope=frozen_slope,
                            chain_slope=chain_slope,
                            chain_roundoff=allowance,
                            jvp_relative_error=relative_error,
                            estimated_truncation=truncation,
                            estimated_roundoff=roundoff,
                            face_box_margin=margin,
                        )
                    )
                    break
            except (ArithmeticError, ValueError):
                continue
        else:
            raise ValueError(f"unresolved FV stationarity direction: {name}")
    evidence: dict[str, object] = dict(
        scope="local numerical first-order check; sampled directions, no minimum or FSO eligibility",
        gradient_norm=norm,
        tolerance=tolerance,
        field_gradient_max=float(gradient[:size].abs().max()),
        dynamics_gradient_max=float(gradient[size:].abs().max()),
        checks=checks,
    )
    return (
        replace(
            result,
            stationarity_verified=True,
            selected_gradient_norm=norm,
            reason=result.reason.replace("_unverified", "_locally_verified"),
        ),
        evidence,
    )


def compute_fv_observation_response(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    verification_dbz: Tensor,
    metric_weight: Tensor,
    leads: int,
    boundary_start_interval: int,
    boundary_echo: BoundarySchedule,
    boundary_support: BoundarySchedule,
    background_dependency: Literal["frozen", "first_observation"],
    maximum_normal_products: int = 128,
    curvature: Literal["irls_gauss_newton", "exact_robust_hessian"] = (
        "irls_gauss_newton"
    ),
) -> FVObservationResponse:
    """Compute ``E_y - (D_y grad_c J)^T A^-T E_c`` for local curvature ``A``.

    CPU FP64 and donorcell with fixed geometry/error statistics, masks, and
    prescribed boundaries. Missing observations and censored observations are
    permitted; detected values must remain in the strict finite interior and
    at least one valid sample is required. Partial support is permitted for
    donorcell when every fixed support trace remains in [0, 1].
    The dimensionless robust control gradient must be <= 1e-8. This numerical
    gate is not a proof of an exact stationary point or a regular optimizer
    path. The weighted future dBZ MSE is fixed throughout the calculation.

    ``first_observation`` includes B(y)=y[0] and its direct forecast term;
    ``frozen`` holds the initial background fixed. Masks and all other prepared
    inputs remain fixed. Requires-grad tensors for verification/weights/future
    boundaries are rejected. Detached dependencies cannot be inferred from raw
    tensors: their exogenous meaning remains the caller's responsibility.
    Positive metric weights must also exclude future cells whose base
    trajectory support is below the fixed-known threshold. Partial support is
    a fixed known-contribution policy, not a physical forecast.
    ``curvature="irls_gauss_newton"`` uses the frozen-IRLS normal operator.
    ``curvature="exact_robust_hessian"`` uses matrix-free HVPs of the robust
    objective and rejects any non-positive curvature encountered by the solve,
    or any failed solve.  This is
    a first-order research response, not a differentiated training API, an
    exact implicit FSO, or a finite impact.

    The budget counts normal products, including true-residual checks, not
    total runtime/memory or every derivative. Budget 1 with a nonzero RHS fails
    closed because a fresh residual needs another product. No dense Hessian is constructed.
    """
    fv = frozen.fv_transport
    if fv is None or fv.reconstruction != "donorcell":
        raise ValueError("FV observation response requires donorcell")
    if control.dtype != torch.float64 or control.device.type != "cpu":
        raise ValueError("FV observation response supports CPU FP64 only")
    _validate_control(control, frozen)
    _validate_observations(observations)
    if (
        observations.dbz.dtype != control.dtype
        or observations.dbz.device != control.device
        or observations.dbz.shape != frozen.input_frames_dbz.shape
    ):
        raise ValueError("observations must match the frozen CPU FP64 analysis grid")
    if (
        frozen.neural_prior_std_dbz is not None
        or frozen.neural_prior_valid_mask is not None
    ):
        raise ValueError("FV GN response requires the identity control prior")
    if not bool(observations.valid_mask.any()):
        raise ValueError("FV observation response requires at least one valid observation")
    detected = observations.valid_mask & observations.detected_mask
    if bool(detected.any()) and not bool(
        (
            (
                (observations.dbz > frozen.analysis_config.detection_limit_dbz)
                & (observations.dbz > frozen.nowcast_config.min_dbz)
                & (observations.dbz < frozen.nowcast_config.max_dbz)
            )[detected]
        ).all()
    ):
        raise ValueError(
            "detected observations must be strictly inside detection and dBZ limits"
        )
    if frozen.initial_support_mask.dtype is not torch.bool:
        raise ValueError("FV response requires a fixed boolean initial support mask")

    def validate_support_schedule(schedule: BoundarySchedule, name: str) -> None:
        for stages in schedule:
            for edges in stages:
                for edge in edges:
                    if not bool(torch.isfinite(edge).all()) or bool(
                        (edge < 0).any() | (edge > 1).any()
                    ):
                        raise ValueError(
                            f"{name} must remain in the fixed support range [0, 1]"
                        )

    validate_support_schedule(fv.boundary_support, "FV prescribed support")
    validate_support_schedule(boundary_support, "future prescribed support")
    if background_dependency not in ("frozen", "first_observation"):
        raise ValueError("unknown background_dependency")
    if background_dependency == "first_observation" and not bool(
        observations.detected_mask[0].all()
    ):
        raise ValueError(
            "first_observation requires a fully detected initial background"
        )
    if background_dependency == "first_observation" and not torch.equal(
        frozen.initial_background_dbz, observations.dbz[0]
    ):
        raise ValueError("first_observation requires initial background B=y[0]")
    if type(maximum_normal_products) is not int or maximum_normal_products <= 0:
        raise ValueError("maximum_normal_products must be a positive integer")
    if curvature not in ("irls_gauss_newton", "exact_robust_hessian"):
        raise ValueError("unsupported FV response curvature")
    if type(leads) is not int or leads <= 0:
        raise ValueError("leads must be a positive integer")
    if type(boundary_start_interval) is not int or boundary_start_interval != 2:
        raise ValueError("future boundaries must start at analysis interval 2")
    expected = (leads, *observations.dbz.shape[-2:])
    for name, value in (
        ("verification_dbz", verification_dbz),
        ("metric_weight", metric_weight),
    ):
        if (
            value.shape != expected
            or value.dtype != control.dtype
            or value.device != control.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{name} must be finite CPU FP64 with shape {expected}")
    if bool((metric_weight < 0).any()) or not bool((metric_weight > 0).any()):
        raise ValueError("metric_weight must be nonnegative with positive total weight")
    fixed_inputs = [verification_dbz, metric_weight]
    fixed_inputs.extend(
        edge
        for schedule in (boundary_echo, boundary_support)
        for stages in schedule
        for edges in stages
        for edge in edges
    )
    if any(value.requires_grad for value in fixed_inputs):
        raise ValueError(
            "verification, weights and future boundaries must be fixed tensors"
        )
    if observations.std_dbz.requires_grad or observations.quality_weight.requires_grad:
        raise ValueError("observation scales and weights must be fixed tensors")
    support_check = forecast_fv_analysis(
        control,
        frozen,
        leads=leads,
        boundary_start_interval=boundary_start_interval,
        boundary_echo=boundary_echo,
        boundary_support=boundary_support,
    )
    if support_check.support_frames is None:
        raise ValueError("FV response requires fixed support diagnostics")
    support_known = support_check.support_frames[1:] >= (
        1 - 128 * torch.finfo(control.dtype).eps
    )
    face_margin = face_branch_margin(control, control, frozen)
    if support_check.psi_coefficients is None:
        raise ValueError("FV response requires fixed transport coefficients")
    stencil_known = _known_support_frames(
        frozen,
        leads=leads,
        boundary_support=boundary_support,
        psi_coefficients=support_check.psi_coefficients,
    )
    active_observation = observations.valid_mask & (
        observations.quality_weight > 0
    )
    if bool((active_observation & ~stencil_known[:3]).any()):
        raise ValueError(
            "active observations must lie in the fixed known support domain"
        )
    if bool(((metric_weight > 0) & (~support_known | ~stencil_known[3:])).any()):
        raise ValueError(
            "metric_weight must exclude future cells without fixed known support"
        )
    # Normalize fixed weights before their sum, avoiding overflow from units.
    weights = metric_weight / metric_weight.max()
    weights = weights / weights.sum()
    if bool(((metric_weight > 0) & (weights == 0)).any()):
        raise ValueError("metric_weight dynamic range underflows during normalization")
    def contract(y: Tensor) -> FrozenOuterState:
        return (
            replace(frozen, initial_background_dbz=y[0])
            if background_dependency == "first_observation"
            else frozen
        )

    def objective(c: Tensor, y: Tensor) -> Tensor:
        return robust_objective(c, replace(observations, dbz=y), contract(y))

    def score(c: Tensor, y: Tensor) -> Tensor:
        trajectory = forecast_fv_analysis(
            c,
            contract(y),
            leads=leads,
            boundary_start_interval=boundary_start_interval,
            boundary_echo=boundary_echo,
            boundary_support=boundary_support,
        )
        prediction = echo_to_dbz(
            trajectory.frames_linear[1:], min_dbz=frozen.nowcast_config.min_dbz
        )
        difference = torch.where(
            weights > 0, prediction - verification_dbz, torch.zeros_like(prediction)
        )
        return (weights * difference.square()).sum()

    gradient = torch.func.grad(objective, argnums=0)
    g = gradient(control, observations.dbz)
    gradient_max = float(g.abs().max())
    if not bool(torch.isfinite(g).all()) or gradient_max > 1e-8:
        raise ValueError("FV response requires a refined robust stationary point")
    metric_value = score(control, observations.dbz)
    if not bool(torch.isfinite(metric_value)):
        raise ValueError("nonfinite FV score")
    rhs, direct = torch.func.grad(score, argnums=(0, 1))(control, observations.dbz)
    if not bool(torch.isfinite(metric_value)) or not bool(torch.isfinite(direct).all()):
        raise ValueError("nonfinite FV score or direct sensitivity")
    products = 0

    if curvature == "irls_gauss_newton":
        linearized = freeze_irls_weights(control, observations, frozen)

        def residual(c: Tensor) -> Tensor:
            return residual_vector(c, observations, linearized)

        pullback = cast(
            Callable[[Tensor], tuple[Tensor]], torch.func.vjp(residual, control)[1]
        )

        def normal(direction: Tensor) -> Tensor:
            nonlocal products
            if products >= maximum_normal_products:
                raise RuntimeError("FV normal-product budget exhausted")
            products += 1
            tangent = torch.func.jvp(residual, (control,), (direction,))[1]
            return pullback(tangent)[0]

    else:
        def normal(direction: Tensor) -> Tensor:
            nonlocal products
            if products >= maximum_normal_products:
                raise RuntimeError("FV normal-product budget exhausted")
            products += 1
            hessian = torch.func.jvp(
                gradient,
                (control, observations.dbz),
                (direction, torch.zeros_like(observations.dbz)),
            )[1]
            # H(0)=0 is valid; a zero direction carries no curvature
            # information and must not be treated as a positive-definiteness
            # certificate.
            if not bool(torch.any(direction != 0)):
                return hessian
            curvature_value = torch.dot(direction, hessian)
            if not bool(torch.isfinite(curvature_value)) or not bool(
                curvature_value > 0
            ):
                raise ValueError("exact robust Hessian has non-positive curvature")
            return hessian

    # The identity control-prior rows guarantee A=J.T J >= I in GN mode.
    # Exact mode checks positive curvature on each HVP direction.  Reserve a
    # final product; true-residual checks remain inside the same hard budget.
    diagonal = _initial_observation_diagonal(control, observations, frozen)
    adjoint = pcg(
        normal, rhs, preconditioner=lambda x: x / diagonal,
        rtol=1e-10, max_iterations=max(1, maximum_normal_products - 1)
    )
    if not adjoint.converged:
        raise ValueError("FV curvature adjoint did not converge")
    observation_pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        torch.func.vjp(lambda y: gradient(control, y), observations.dbz)[1],
    )
    sensitivity = direct - observation_pullback(adjoint.solution)[0]
    if not bool(torch.isfinite(sensitivity).all()):
        raise ValueError("nonfinite FV observation sensitivity")
    return FVObservationResponse(
        sensitivity_dbz=sensitivity,
        direct_sensitivity_dbz=direct,
        score=float(metric_value),
        gradient_max=gradient_max,
        normal_products=products,
        adjoint_relative_residual=adjoint.relative_residual,
        face_margin=face_margin,
        curvature=curvature,
    )
