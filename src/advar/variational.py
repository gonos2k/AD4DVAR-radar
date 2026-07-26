"""Minimal three-frame variational analysis.

P1 jointly adjusts the initial (-20 minute) echo and three global dynamics
controls.  The existing FFT nowcast remains the numerical fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .matrix_free import gauss_newton_hvp, pcg
from .nowcast import (
    NowcastConfig,
    RadarState,
    RemapCell,
    advect,
    dbz_to_linear,
    estimate_state,
    freeze_remap_cell,
    forecast_from_state,
    linear_to_dbz,
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Small, explicit P1 analysis and solver configuration."""

    detection_limit_dbz: float = 5.0
    censor_temperature_dbz: float = 1.0
    observation_std_dbz: float = 2.0
    minimum_observation_std_dbz: float = 0.1
    pseudo_huber_delta: float = 2.0
    echo_transform_scale_dbz: float = 1.0
    transform_epsilon: float = 1.0e-6
    initial_increment_scale_dbz: float = 4.0
    motion_increment_scale_px: float = 1.0
    growth_increment_scale: float = 0.04
    maximum_outer_iterations: int = 4
    maximum_pcg_iterations: int = 40
    maximum_damping_retries: int = 2
    pcg_relative_tolerance: float = 1.0e-5
    gradient_tolerance: float = 1.0e-5
    step_tolerance: float = 1.0e-4
    initial_damping: float = 1.0e-2
    minimum_damping: float = 1.0e-6
    maximum_damping: float = 1.0e6

    def __post_init__(self) -> None:
        positive = {
            "censor_temperature_dbz": self.censor_temperature_dbz,
            "observation_std_dbz": self.observation_std_dbz,
            "minimum_observation_std_dbz": (
                self.minimum_observation_std_dbz
            ),
            "pseudo_huber_delta": self.pseudo_huber_delta,
            "echo_transform_scale_dbz": self.echo_transform_scale_dbz,
            "transform_epsilon": self.transform_epsilon,
            "initial_increment_scale_dbz": (
                self.initial_increment_scale_dbz
            ),
            "motion_increment_scale_px": self.motion_increment_scale_px,
            "growth_increment_scale": self.growth_increment_scale,
            "pcg_relative_tolerance": self.pcg_relative_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
            "step_tolerance": self.step_tolerance,
            "initial_damping": self.initial_damping,
            "minimum_damping": self.minimum_damping,
            "maximum_damping": self.maximum_damping,
        }
        if not math.isfinite(self.detection_limit_dbz):
            raise ValueError("detection_limit_dbz must be finite")
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            type(self.maximum_outer_iterations) is not int
            or self.maximum_outer_iterations <= 0
        ):
            raise ValueError("maximum_outer_iterations must be positive")
        if (
            type(self.maximum_pcg_iterations) is not int
            or self.maximum_pcg_iterations <= 0
        ):
            raise ValueError("maximum_pcg_iterations must be positive")
        if (
            type(self.maximum_damping_retries) is not int
            or self.maximum_damping_retries < 0
        ):
            raise ValueError("maximum_damping_retries cannot be negative")
        if self.minimum_damping > self.initial_damping:
            raise ValueError("minimum_damping cannot exceed initial_damping")
        if self.initial_damping > self.maximum_damping:
            raise ValueError("initial_damping cannot exceed maximum_damping")


@dataclass(frozen=True)
class AnalysisObservations:
    """Three observation frames and their fixed quality contract."""

    dbz: Tensor
    std_dbz: Tensor
    quality_weight: Tensor
    valid_mask: Tensor
    detected_mask: Tensor
    censored_mask: Tensor


@dataclass(frozen=True)
class FrozenOuterState:
    """Quantities that must not change inside one Krylov solve."""

    initial_background_dbz: Tensor
    initial_support_mask: Tensor
    baseline_state: RadarState
    irls_sqrt_weight: Tensor
    nowcast_config: NowcastConfig
    analysis_config: AnalysisConfig
    analysis_remap_cells: tuple[RemapCell, RemapCell] | None = None


@dataclass(frozen=True)
class AnalysisTrajectory:
    """Decoded control and its -20/-10/0 minute model trajectory."""

    frames_linear: Tensor
    displacement_yx: Tensor
    log_growth_per_step: Tensor


@dataclass(frozen=True)
class AnalysisResult:
    """Safe result returned by the P1 LM-PCG analysis."""

    control: Tensor
    state: RadarState
    analyzed_frames_linear: Tensor
    initial_objective: float
    final_objective: float
    outer_iterations: int
    pcg_iterations: int
    converged: bool
    used_fallback: bool
    reason: str


def prepare_analysis(
    frames_dbz: Tensor,
    *,
    nowcast_config: NowcastConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    observation_std_dbz: float | Tensor | None = None,
    quality_weight: float | Tensor | None = None,
    qc_mask: Tensor | None = None,
) -> tuple[AnalysisObservations, FrozenOuterState]:
    """Freeze observation masks, errors, FFT background, and initial IRLS."""

    nowcast_config = nowcast_config or NowcastConfig()
    analysis_config = analysis_config or AnalysisConfig()
    _validate_frames(frames_dbz)
    if not (
        nowcast_config.min_dbz
        < analysis_config.detection_limit_dbz
        < nowcast_config.max_dbz
    ):
        raise ValueError("detection_limit_dbz must be inside the dBZ range")

    finite = torch.isfinite(frames_dbz)
    if qc_mask is None:
        qc = torch.ones_like(frames_dbz, dtype=torch.bool)
    else:
        if qc_mask.shape != frames_dbz.shape or qc_mask.dtype != torch.bool:
            raise ValueError("qc_mask must be boolean with the frame shape")
        qc = qc_mask.to(device=frames_dbz.device)

    std = _observation_std(
        frames_dbz,
        observation_std_dbz,
        analysis_config,
    )
    quality = _quality_weight(frames_dbz, quality_weight)
    valid = finite & qc & (quality > 0)
    clean = torch.nan_to_num(
        frames_dbz,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    ).clamp(nowcast_config.min_dbz, nowcast_config.max_dbz)
    clean = torch.where(
        valid,
        clean,
        clean.new_full((), nowcast_config.min_dbz),
    )
    detected = valid & (clean >= analysis_config.detection_limit_dbz)
    censored = valid & ~detected
    observations = AnalysisObservations(
        dbz=clean.detach().clone(),
        std_dbz=std.detach().clone(),
        quality_weight=quality.detach().clone(),
        valid_mask=valid.detach().clone(),
        detected_mask=detected.detach().clone(),
        censored_mask=censored.detach().clone(),
    )

    baseline_state = _detach_state(estimate_state(clean, nowcast_config))
    remap_cells = tuple(
        freeze_remap_cell(step * baseline_state.displacement_yx)
        for step in (1, 2)
    )
    initial_frozen = FrozenOuterState(
        initial_background_dbz=clean[0].detach().clone(),
        initial_support_mask=detected[0].detach().clone(),
        baseline_state=baseline_state,
        analysis_remap_cells=remap_cells,
        irls_sqrt_weight=valid.to(dtype=frames_dbz.dtype).detach().clone(),
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
    )
    control = initial_control(observations)
    return observations, freeze_irls_weights(
        control,
        observations,
        initial_frozen,
    )


def initial_control(observations: AnalysisObservations) -> Tensor:
    """Return zero standardized increments for q(-20), motion, and growth."""

    _validate_observations(observations)
    height, width = observations.dbz.shape[-2:]
    return observations.dbz.new_zeros(height * width + 3)


def analysis_trajectory(
    control: Tensor,
    frozen_outer_state: FrozenOuterState,
) -> AnalysisTrajectory:
    """Decode a standardized control into the three analysis times."""

    _validate_control(control, frozen_outer_state)
    height, width = frozen_outer_state.initial_background_dbz.shape
    field_control = control[: height * width].reshape(height, width)
    dynamics_control = control[height * width :]
    config = frozen_outer_state.analysis_config
    nowcast = frozen_outer_state.nowcast_config
    baseline = frozen_outer_state.baseline_state

    floor_dbz = nowcast.min_dbz
    background_offset = (
        frozen_outer_state.initial_background_dbz - floor_dbz
    ) / config.echo_transform_scale_dbz
    background_latent = _softplus_inverse(
        background_offset.clamp_min(config.transform_epsilon)
    )
    analyzed_offset = config.echo_transform_scale_dbz * F.softplus(
        background_latent
        + (
            config.initial_increment_scale_dbz
            / config.echo_transform_scale_dbz
        )
        * field_control
        * frozen_outer_state.initial_support_mask
    )
    initial_dbz = floor_dbz + analyzed_offset
    echo_floor = initial_dbz.new_tensor(10.0 ** (floor_dbz / 10.0))
    initial_echo = torch.pow(10.0, initial_dbz / 10.0) - echo_floor
    displacement, growth = _decode_dynamics(
        dynamics_control,
        baseline,
        config,
        nowcast,
    )

    frames = [initial_echo]
    remap_cells = frozen_outer_state.analysis_remap_cells
    if remap_cells is None:
        remap_cells = tuple(
            freeze_remap_cell(step * displacement)
            for step in (1, 2)
        )
    for step in (1, 2):
        frame = advect(
            initial_echo,
            step * displacement,
            frozen_cell=remap_cells[step - 1],
        )
        frames.append(frame * torch.exp(step * growth))
    return AnalysisTrajectory(
        frames_linear=torch.stack(frames),
        displacement_yx=displacement,
        log_growth_per_step=growth,
    )


def observation_residual_dbz(
    control: Tensor,
    observations: AnalysisObservations,
    frozen_outer_state: FrozenOuterState,
) -> Tensor:
    """Return detected errors and one-sided censored errors in dBZ."""

    _validate_observations(observations)
    trajectory = analysis_trajectory(control, frozen_outer_state)
    prediction = _analysis_echo_to_dbz(
        trajectory.frames_linear,
        frozen_outer_state.nowcast_config,
    )
    detected_error = prediction - observations.dbz
    config = frozen_outer_state.analysis_config
    censored_error = config.censor_temperature_dbz * F.softplus(
        (
            prediction - config.detection_limit_dbz
        )
        / config.censor_temperature_dbz
    )
    return torch.where(
        observations.detected_mask,
        detected_error,
        torch.where(
            observations.censored_mask,
            censored_error,
            torch.zeros_like(prediction),
        ),
    )


def whitened_observation_residual(
    control: Tensor,
    observations: AnalysisObservations,
    frozen_outer_state: FrozenOuterState,
) -> Tensor:
    """Whiten the observation residual exactly once."""

    return (
        torch.sqrt(observations.quality_weight)
        *
        observation_residual_dbz(
            control,
            observations,
            frozen_outer_state,
        )
        / observations.std_dbz
    )


def freeze_irls_weights(
    control: Tensor,
    observations: AnalysisObservations,
    frozen_outer_state: FrozenOuterState,
) -> FrozenOuterState:
    """Recompute pseudo-Huber weights between, never inside, PCG solves."""

    frozen_outer_state = _freeze_analysis_remap_cells(
        control,
        frozen_outer_state,
    )
    residual = whitened_observation_residual(
        control,
        observations,
        frozen_outer_state,
    ).detach()
    delta = frozen_outer_state.analysis_config.pseudo_huber_delta
    sqrt_weight = torch.pow(1.0 + (residual / delta).square(), -0.25)
    sqrt_weight = torch.where(
        observations.valid_mask,
        sqrt_weight,
        torch.zeros_like(sqrt_weight),
    )
    return replace(
        frozen_outer_state,
        irls_sqrt_weight=sqrt_weight,
    )


def residual_vector(
    control: Tensor,
    observations: AnalysisObservations,
    frozen_outer_state: FrozenOuterState,
) -> Tensor:
    """Return once-whitened frozen-IRLS observations plus unit prior."""

    whitened = whitened_observation_residual(
        control,
        observations,
        frozen_outer_state,
    )
    weighted = whitened * frozen_outer_state.irls_sqrt_weight
    return torch.cat((weighted.reshape(-1), control))


def robust_objective(
    control: Tensor,
    observations: AnalysisObservations,
    frozen_outer_state: FrozenOuterState,
) -> Tensor:
    """Return the true pseudo-Huber objective used for step acceptance."""

    residual = whitened_observation_residual(
        control,
        observations,
        frozen_outer_state,
    )
    delta = frozen_outer_state.analysis_config.pseudo_huber_delta
    robust = delta**2 * (
        torch.sqrt(1.0 + (residual / delta).square()) - 1.0
    )
    robust = torch.where(
        observations.valid_mask,
        robust,
        torch.zeros_like(robust),
    )
    return robust.sum() + 0.5 * torch.dot(control, control)


def solve_analysis(
    observations: AnalysisObservations,
    frozen_outer_state: FrozenOuterState,
    *,
    control: Tensor | None = None,
) -> AnalysisResult:
    """Solve the P1 robust analysis with damped matrix-free LM-PCG."""

    _validate_observations(observations)
    control = (
        initial_control(observations)
        if control is None
        else control.detach().clone()
    )
    _validate_control(control, frozen_outer_state)
    frozen_outer_state = _freeze_analysis_remap_cells(
        control,
        frozen_outer_state,
    )
    initial_cost_tensor = robust_objective(
        control,
        observations,
        frozen_outer_state,
    )
    initial_cost = float(initial_cost_tensor.detach())
    if not bool(torch.any(observations.valid_mask)):
        return _fallback_result(
            observations,
            frozen_outer_state,
            control,
            initial_cost,
            "no_valid_observations",
        )
    if not math.isfinite(initial_cost):
        return _fallback_result(
            observations,
            frozen_outer_state,
            control,
            initial_cost,
            "nonfinite_initial_objective",
        )

    config = frozen_outer_state.analysis_config
    damping = config.initial_damping
    current_cost = initial_cost
    total_pcg_iterations = 0
    accepted_any = False
    converged = False
    reason = "maximum_outer_iterations"
    completed_iterations = 0

    for outer_iteration in range(1, config.maximum_outer_iterations + 1):
        completed_iterations = outer_iteration
        frozen_iteration = freeze_irls_weights(
            control,
            observations,
            frozen_outer_state,
        )
        residual_fn = lambda value: residual_vector(
            value,
            observations,
            frozen_iteration,
        )
        residual, pullback = torch.func.vjp(residual_fn, control)
        gradient = pullback(residual)[0]
        gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
        if not math.isfinite(gradient_norm):
            return _fallback_result(
                observations,
                frozen_outer_state,
                control,
                initial_cost,
                "nonfinite_gradient",
                completed_iterations,
                total_pcg_iterations,
            )
        if gradient_norm <= config.gradient_tolerance:
            converged = True
            reason = "gradient_tolerance"
            break

        accepted_this_outer = False
        linear_system_solved = False
        for _ in range(config.maximum_damping_retries + 1):
            operator = lambda vector: gauss_newton_hvp(
                residual_fn,
                control,
                vector,
                damping=damping,
            )
            try:
                candidate_linear = pcg(
                    operator,
                    -gradient,
                    rtol=config.pcg_relative_tolerance,
                    max_iterations=config.maximum_pcg_iterations,
                )
            except (ArithmeticError, RuntimeError, ValueError):
                candidate_linear = None
            if candidate_linear is None:
                damping = min(config.maximum_damping, 4.0 * damping)
                continue
            total_pcg_iterations += candidate_linear.iterations
            if not candidate_linear.converged or not bool(
                torch.all(torch.isfinite(candidate_linear.solution))
            ):
                damping = min(config.maximum_damping, 4.0 * damping)
                continue
            linear_system_solved = True

            raw_step = candidate_linear.solution
            hessian_raw_step = gauss_newton_hvp(
                residual_fn,
                control,
                raw_step,
            )
            directional_gradient = torch.dot(gradient, raw_step)
            directional_curvature = torch.dot(
                raw_step,
                hessian_raw_step,
            )
            for backtrack in range(12):
                scale = 0.5**backtrack
                step = scale * raw_step
                predicted_reduction = (
                    -scale * directional_gradient
                    - 0.5 * scale**2 * directional_curvature
                )
                predicted = float(predicted_reduction.detach())
                candidate = control + step
                candidate_frozen = _freeze_analysis_remap_cells(
                    candidate,
                    frozen_iteration,
                )
                candidate_cost = float(
                    robust_objective(
                        candidate,
                        observations,
                        candidate_frozen,
                    ).detach()
                )
                actual = current_cost - candidate_cost
                ratio = actual / predicted if predicted > 0 else -math.inf

                if not (
                    math.isfinite(candidate_cost)
                    and actual > 0
                    and ratio >= 0.1
                ):
                    continue

                control = candidate.detach()
                current_cost = candidate_cost
                accepted_any = True
                accepted_this_outer = True
                if ratio > 0.75:
                    damping = max(
                        config.minimum_damping,
                        0.5 * damping,
                    )
                elif ratio < 0.25:
                    damping = min(
                        config.maximum_damping,
                        2.0 * damping,
                    )
                relative_step = float(
                    torch.linalg.vector_norm(step).detach()
                ) / (
                    1.0
                    + float(torch.linalg.vector_norm(control).detach())
                )
                if relative_step <= config.step_tolerance:
                    converged = True
                    reason = "step_tolerance"
                break
            if accepted_this_outer:
                break
            damping = min(config.maximum_damping, 4.0 * damping)

        if not accepted_this_outer:
            return _fallback_result(
                observations,
                frozen_outer_state,
                control,
                initial_cost,
                (
                    "no_accepted_step"
                    if linear_system_solved
                    else "pcg_failed"
                ),
                completed_iterations,
                total_pcg_iterations,
            )
        if converged:
            break

    if not accepted_any and not converged:
        return _fallback_result(
            observations,
            frozen_outer_state,
            control,
            initial_cost,
            "no_accepted_step",
            completed_iterations,
            total_pcg_iterations,
        )

    return _analysis_result(
        control,
        frozen_outer_state,
        initial_cost,
        current_cost,
        completed_iterations,
        total_pcg_iterations,
        converged,
        False,
        reason,
    )


def variational_nowcast(
    frames_dbz: Tensor,
    *,
    nowcast_config: NowcastConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    observation_std_dbz: float | Tensor | None = None,
    quality_weight: float | Tensor | None = None,
    qc_mask: Tensor | None = None,
) -> tuple[Tensor, AnalysisResult]:
    """Analyze three frames, then forecast 18 leads from analyzed q(0)."""

    nowcast_config = nowcast_config or NowcastConfig()
    observations, frozen = prepare_analysis(
        frames_dbz,
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
        observation_std_dbz=observation_std_dbz,
        quality_weight=quality_weight,
        qc_mask=qc_mask,
    )
    result = solve_analysis(observations, frozen)
    return forecast_from_state(result.state, nowcast_config), result


def _analysis_result(
    control: Tensor,
    frozen: FrozenOuterState,
    initial_objective: float,
    final_objective: float,
    outer_iterations: int,
    pcg_iterations: int,
    converged: bool,
    used_fallback: bool,
    reason: str,
) -> AnalysisResult:
    frozen = _freeze_analysis_remap_cells(control, frozen)
    trajectory = analysis_trajectory(control, frozen)
    baseline = frozen.baseline_state
    state = RadarState(
        echo_amplitude=torch.sqrt(trajectory.frames_linear[-1]),
        displacement_yx=trajectory.displacement_yx,
        log_growth_per_step=trajectory.log_growth_per_step,
        pair_displacements_yx=baseline.pair_displacements_yx,
        pair_log_growth=baseline.pair_log_growth,
        provenance="p1_variational_analysis",
    )
    return AnalysisResult(
        control=control.detach(),
        state=_detach_state(state),
        analyzed_frames_linear=trajectory.frames_linear.detach(),
        initial_objective=initial_objective,
        final_objective=final_objective,
        outer_iterations=outer_iterations,
        pcg_iterations=pcg_iterations,
        converged=converged,
        used_fallback=used_fallback,
        reason=reason,
    )


def _fallback_result(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    control: Tensor,
    initial_objective: float,
    reason: str,
    outer_iterations: int = 0,
    pcg_iterations: int = 0,
) -> AnalysisResult:
    frames_linear = dbz_to_linear(
        observations.dbz,
        frozen.nowcast_config,
    )
    return AnalysisResult(
        control=torch.zeros_like(control),
        state=_detach_state(frozen.baseline_state),
        analyzed_frames_linear=frames_linear.detach(),
        initial_objective=initial_objective,
        final_objective=initial_objective,
        outer_iterations=outer_iterations,
        pcg_iterations=pcg_iterations,
        converged=False,
        used_fallback=True,
        reason=reason,
    )


def _bounded_update(
    background: Tensor,
    control: Tensor,
    scale: float,
    limit: float,
) -> Tensor:
    if limit == 0:
        return torch.zeros_like(background)
    ratio = (background / limit).clamp(-0.999999, 0.999999)
    latent = torch.atanh(ratio)
    return limit * torch.tanh(latent + (scale / limit) * control)


def _decode_dynamics(
    dynamics_control: Tensor,
    baseline: RadarState,
    config: AnalysisConfig,
    nowcast: NowcastConfig,
) -> tuple[Tensor, Tensor]:
    displacement = _bounded_update(
        baseline.displacement_yx,
        dynamics_control[:2],
        config.motion_increment_scale_px,
        nowcast.max_displacement_px,
    )
    growth = _bounded_update(
        baseline.log_growth_per_step,
        dynamics_control[2],
        config.growth_increment_scale,
        nowcast.max_log_growth_per_step,
    )
    return displacement, growth


def _freeze_analysis_remap_cells(
    control: Tensor,
    frozen: FrozenOuterState,
) -> FrozenOuterState:
    """Freeze the two analysis-window transport cells at an outer iterate."""

    height, width = frozen.initial_background_dbz.shape
    dynamics_control = control[height * width :]
    displacement, _ = _decode_dynamics(
        dynamics_control,
        frozen.baseline_state,
        frozen.analysis_config,
        frozen.nowcast_config,
    )
    cells = tuple(
        freeze_remap_cell(step * displacement)
        for step in (1, 2)
    )
    return replace(frozen, analysis_remap_cells=cells)


def _softplus_inverse(value: Tensor) -> Tensor:
    """Stable inverse of softplus for strictly positive values."""

    return value + torch.log(-torch.expm1(-value))


def _analysis_echo_to_dbz(echo: Tensor, config: NowcastConfig) -> Tensor:
    """Unclipped observation operator used only inside the analysis."""

    floor = echo.new_tensor(10.0 ** (config.min_dbz / 10.0))
    return 10.0 * torch.log10(echo.clamp_min(0.0) + floor)


def _observation_std(
    frames: Tensor,
    value: float | Tensor | None,
    config: AnalysisConfig,
) -> Tensor:
    source = config.observation_std_dbz if value is None else value
    std = torch.as_tensor(source, dtype=frames.dtype, device=frames.device)
    try:
        std = torch.broadcast_to(std, frames.shape)
    except RuntimeError as error:
        raise ValueError(
            "observation_std_dbz must broadcast to the frame shape"
        ) from error
    if not bool(torch.all(torch.isfinite(std))) or bool(
        torch.any(std < config.minimum_observation_std_dbz)
    ):
        raise ValueError(
            "observation_std_dbz must be finite and above the minimum"
        )
    return std.clone()


def _quality_weight(
    frames: Tensor,
    value: float | Tensor | None,
) -> Tensor:
    source = 1.0 if value is None else value
    weight = torch.as_tensor(
        source,
        dtype=frames.dtype,
        device=frames.device,
    )
    try:
        weight = torch.broadcast_to(weight, frames.shape)
    except RuntimeError as error:
        raise ValueError(
            "quality_weight must broadcast to the frame shape"
        ) from error
    if not bool(torch.all(torch.isfinite(weight))) or bool(
        torch.any((weight < 0) | (weight > 1))
    ):
        raise ValueError("quality_weight must be finite and between 0 and 1")
    return weight.clone()


def _validate_frames(frames: Tensor) -> None:
    if frames.ndim != 3 or frames.shape[0] != 3:
        raise ValueError("frames_dbz must have shape [3, height, width]")
    if frames.shape[1] < 2 or frames.shape[2] < 2:
        raise ValueError("frame height and width must both be at least 2")
    if not frames.is_floating_point():
        raise TypeError("frames_dbz must be a floating-point tensor")
    if frames.dtype not in (torch.float32, torch.float64):
        raise TypeError("P1 analysis requires float32 or float64 frames")


def _validate_observations(observations: AnalysisObservations) -> None:
    _validate_frames(observations.dbz)
    shape = observations.dbz.shape
    if observations.std_dbz.shape != shape:
        raise ValueError("std_dbz must have the observation shape")
    if (
        observations.std_dbz.dtype != observations.dbz.dtype
        or observations.std_dbz.device != observations.dbz.device
        or not bool(torch.all(torch.isfinite(observations.std_dbz)))
        or bool(torch.any(observations.std_dbz <= 0))
    ):
        raise ValueError("std_dbz must be finite, positive, and compatible")
    if observations.quality_weight.shape != shape:
        raise ValueError("quality_weight must have the observation shape")
    if (
        not observations.quality_weight.is_floating_point()
        or observations.quality_weight.dtype != observations.dbz.dtype
        or observations.quality_weight.device != observations.dbz.device
        or not bool(torch.all(torch.isfinite(observations.quality_weight)))
        or bool(
            torch.any(
                (observations.quality_weight < 0)
                | (observations.quality_weight > 1)
            )
        )
    ):
        raise ValueError("quality_weight must be compatible and in [0, 1]")
    for name in ("valid_mask", "detected_mask", "censored_mask"):
        value = getattr(observations, name)
        if (
            value.shape != shape
            or value.dtype != torch.bool
            or value.device != observations.dbz.device
        ):
            raise ValueError(f"{name} must be boolean with observation shape")
    if not torch.equal(
        observations.valid_mask,
        observations.detected_mask | observations.censored_mask,
    ):
        raise ValueError("detected and censored masks must partition validity")
    if bool(
        torch.any(
            observations.detected_mask & observations.censored_mask
        )
    ):
        raise ValueError("detected and censored masks cannot overlap")


def _validate_control(
    control: Tensor,
    frozen_outer_state: FrozenOuterState,
) -> None:
    height, width = frozen_outer_state.initial_background_dbz.shape
    expected = height * width + 3
    if (
        control.ndim != 1
        or control.numel() != expected
        or not control.is_floating_point()
    ):
        raise ValueError(f"control must be a floating vector of length {expected}")
    if control.device != frozen_outer_state.initial_background_dbz.device:
        raise ValueError("control and frozen state must use the same device")
    if control.dtype != frozen_outer_state.initial_background_dbz.dtype:
        raise ValueError("control and frozen state must use the same dtype")


def _detach_state(state: RadarState) -> RadarState:
    return RadarState(
        echo_amplitude=state.echo_amplitude.detach(),
        displacement_yx=state.displacement_yx.detach(),
        log_growth_per_step=state.log_growth_per_step.detach(),
        pair_displacements_yx=state.pair_displacements_yx.detach(),
        pair_log_growth=state.pair_log_growth.detach(),
        provenance=state.provenance,
    )
