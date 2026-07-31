from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .diagnostics import EchoPositivityError, PositivityAudit, validate_physical_echo
from .matrix_free import gauss_newton_hvp, pcg
from .nowcast import (
    ForecastMetadata,
    ForecastResult,
    ForecastRunContract,
    NowcastConfig,
    RadarState,
    estimate_prepared_state,
    forecast_from_state,
    merge_current_support,
    prepare_input,
)
from .physics import (
    RemapCell,
    advance,
    dbz_to_echo,
    echo_to_dbz,
    freeze_remap_cell,
    remap,
)


@dataclass(frozen=True)
class AnalysisConfig:
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
    minimum_control_reachability: float = 0.25
    causal_support_dilation_px: int = 2
    maximum_detected_error_std: float = 3.0
    maximum_unresolved_amplitude_fraction: float = 0.01
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
            "initial_increment_scale_dbz": self.initial_increment_scale_dbz,
            "motion_increment_scale_px": self.motion_increment_scale_px,
            "growth_increment_scale": self.growth_increment_scale,
            "minimum_control_reachability": (
                self.minimum_control_reachability
            ),
            "maximum_detected_error_std": self.maximum_detected_error_std,
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
        integer_limits = {
            "maximum_outer_iterations": self.maximum_outer_iterations,
            "maximum_pcg_iterations": self.maximum_pcg_iterations,
        }
        for name, value in integer_limits.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            type(self.maximum_damping_retries) is not int
            or self.maximum_damping_retries < 0
        ):
            raise ValueError("maximum_damping_retries cannot be negative")
        if (
            type(self.causal_support_dilation_px) is not int
            or self.causal_support_dilation_px < 0
        ):
            raise ValueError("causal_support_dilation_px cannot be negative")
        if self.minimum_damping > self.initial_damping:
            raise ValueError("minimum_damping cannot exceed initial_damping")
        if self.initial_damping > self.maximum_damping:
            raise ValueError("initial_damping cannot exceed maximum_damping")
        if self.minimum_control_reachability > 1.0:
            raise ValueError("minimum_control_reachability cannot exceed 1")
        if (
            not math.isfinite(self.maximum_unresolved_amplitude_fraction)
            or not 0.0
            <= self.maximum_unresolved_amplitude_fraction
            <= 1.0
        ):
            raise ValueError(
                "maximum_unresolved_amplitude_fraction must be in [0, 1]"
            )


@dataclass(frozen=True)
class AnalysisObservations:
    dbz: Tensor
    std_dbz: Tensor
    quality_weight: Tensor
    valid_mask: Tensor
    detected_mask: Tensor
    censored_mask: Tensor
    missing_mask: Tensor
    qc_rejected_mask: Tensor
    observed_clear_mask: Tensor


@dataclass(frozen=True)
class FrozenOuterState:
    initial_background_dbz: Tensor
    initial_support_mask: Tensor
    causal_only_mask: Tensor
    detected_masks: Tensor
    observed_mask: Tensor
    background_mask: Tensor
    background_age_minutes: float | None
    baseline_state: RadarState
    baseline_metadata: ForecastMetadata
    baseline_frames_dbz: Tensor
    irls_sqrt_weight: Tensor
    nowcast_config: NowcastConfig
    analysis_config: AnalysisConfig
    analysis_remap_cells: tuple[RemapCell, RemapCell]


@dataclass(frozen=True)
class AnalysisTrajectory:
    frames_linear: Tensor
    displacement_yx: Tensor
    log_growth_per_step: Tensor


@dataclass(frozen=True)
class AnalysisResult:
    control: Tensor
    state: RadarState
    metadata: ForecastMetadata
    analyzed_frames_linear: Tensor
    initial_objective: float
    final_objective: float
    outer_iterations: int
    pcg_iterations: int
    converged: bool
    used_fallback: bool
    reason: str
    audit: PositivityAudit
    degraded: bool = False
    minimum_reachability_margin: float | None = None
    unresolved_amplitude_fraction: float | None = None
    causal_control_cell_count: int = 0
    causal_seed_cell_count: int = 0
    causal_seed_prior_cost: float = 0.0


def prepare_analysis(
    frames_dbz: Tensor,
    *,
    nowcast_config: NowcastConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    observation_std_dbz: float | Tensor | None = None,
    quality_weight: float | Tensor | None = None,
    qc_mask: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
) -> tuple[AnalysisObservations, FrozenOuterState]:
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
    observed_dbz = torch.nan_to_num(
        frames_dbz,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    ).clamp(nowcast_config.min_dbz, nowcast_config.max_dbz)
    observed_dbz = torch.where(
        valid,
        observed_dbz,
        observed_dbz.new_full((), nowcast_config.min_dbz),
    )
    prepared = prepare_input(
        frames_dbz,
        nowcast_config,
        accepted_mask=valid,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
    )
    detected = valid & (
        observed_dbz >= analysis_config.detection_limit_dbz
    )
    censored = valid & ~detected
    observations = AnalysisObservations(
        dbz=observed_dbz.detach().clone(),
        std_dbz=std.detach().clone(),
        quality_weight=quality.detach().clone(),
        valid_mask=valid.detach().clone(),
        detected_mask=detected.detach().clone(),
        censored_mask=censored.detach().clone(),
        missing_mask=prepared.missing_mask.detach().clone(),
        qc_rejected_mask=prepared.qc_rejected_mask.detach().clone(),
        observed_clear_mask=censored.detach().clone(),
    )
    _validate_observations(observations)

    baseline_state, baseline_metadata = estimate_prepared_state(
        prepared,
        nowcast_config,
    )
    baseline_state = _detach_state(baseline_state)
    baseline_metadata = _detach_metadata(baseline_metadata)
    initial_support = _causal_initial_support(
        detected,
        prepared.observed_mask,
        prepared.background_mask,
        baseline_state.displacement_yx,
        analysis_config.minimum_control_reachability,
        analysis_config.causal_support_dilation_px,
    )
    causal_only = initial_support & ~detected[0]
    remap_cells = tuple(
        freeze_remap_cell(step * baseline_state.displacement_yx)
        for step in (1, 2)
    )
    baseline_frames_dbz = torch.where(
        prepared.observed_mask,
        prepared.frames_dbz,
        prepared.background_frames_dbz,
    )
    frozen = FrozenOuterState(
        initial_background_dbz=baseline_frames_dbz[0].detach().clone(),
        initial_support_mask=initial_support.detach().clone(),
        causal_only_mask=causal_only.detach().clone(),
        detected_masks=detected.detach().clone(),
        observed_mask=prepared.observed_mask.detach().clone(),
        background_mask=prepared.background_mask.detach().clone(),
        background_age_minutes=prepared.background_age_minutes,
        baseline_state=baseline_state,
        baseline_metadata=baseline_metadata,
        baseline_frames_dbz=baseline_frames_dbz.detach().clone(),
        irls_sqrt_weight=valid.to(dtype=frames_dbz.dtype).detach().clone(),
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
        analysis_remap_cells=remap_cells,
    )
    control = _warm_started_control(observations, frozen)
    return observations, freeze_irls_weights(
        control,
        observations,
        frozen,
    )


def _causal_initial_support(
    detected_mask: Tensor,
    observed_mask: Tensor,
    background_mask: Tensor,
    displacement_yx: Tensor,
    minimum_reachability: float,
    dilation_px: int,
) -> Tensor:
    initial_anchor = observed_mask[0] | background_mask[0]
    later_support = torch.zeros_like(detected_mask[0])
    for step in (1, 2):
        precursor = remap(
            detected_mask[step].to(dtype=displacement_yx.dtype),
            -step * displacement_yx,
        )
        later_support |= precursor >= minimum_reachability
    if dilation_px > 0:
        later_support = (
            F.max_pool2d(
                later_support[None, None].to(dtype=displacement_yx.dtype),
                kernel_size=2 * dilation_px + 1,
                stride=1,
                padding=dilation_px,
            )[0, 0]
            > 0
        )
    return (detected_mask[0] | later_support) & initial_anchor


def initial_control(observations: AnalysisObservations) -> Tensor:
    _validate_observations(observations)
    height, width = observations.dbz.shape[-2:]
    return observations.dbz.new_zeros(height * width + 3)


def _warm_started_control(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    control = initial_control(observations)
    seed_control = _precursor_seed_control(frozen)
    field_size = frozen.initial_background_dbz.numel()
    control[:field_size] = seed_control.flatten()
    return control


def _precursor_seed_control(frozen: FrozenOuterState) -> Tensor:
    seed_control = torch.zeros_like(frozen.initial_background_dbz)
    if not bool(torch.any(frozen.causal_only_mask)):
        return seed_control
    config = frozen.analysis_config
    floor_dbz = frozen.nowcast_config.min_dbz
    seed_dbz = max(
        config.detection_limit_dbz - config.censor_temperature_dbz,
        0.5 * (floor_dbz + config.detection_limit_dbz),
    )
    seed_mask = frozen.causal_only_mask & (
        frozen.initial_background_dbz < seed_dbz
    )
    if not bool(torch.any(seed_mask)):
        return seed_control

    background_offset = (
        frozen.initial_background_dbz - floor_dbz
    ) / config.echo_transform_scale_dbz
    background_latent = _softplus_inverse(
        background_offset.clamp_min(config.transform_epsilon)
    )
    seed_offset = seed_control.new_full(
        frozen.initial_background_dbz.shape,
        (seed_dbz - floor_dbz) / config.echo_transform_scale_dbz,
    )
    seed_latent = _softplus_inverse(
        seed_offset.clamp_min(config.transform_epsilon)
    )
    required_control = (
        (seed_latent - background_latent)
        * config.echo_transform_scale_dbz
        / config.initial_increment_scale_dbz
    )
    seed_control[seed_mask] = required_control[seed_mask]
    return seed_control


def _causal_seed_diagnostics(
    frozen: FrozenOuterState,
) -> tuple[int, int, float]:
    seed_control = _precursor_seed_control(frozen)
    return (
        int(torch.count_nonzero(frozen.causal_only_mask)),
        int(torch.count_nonzero(seed_control)),
        0.5 * float(torch.dot(seed_control.flatten(), seed_control.flatten())),
    )


def analysis_trajectory(
    control: Tensor,
    frozen: FrozenOuterState,
) -> AnalysisTrajectory:
    _validate_control(control, frozen)
    return _analysis_trajectory(
        control,
        _freeze_analysis_remap_cells(control, frozen),
    )


def _analysis_trajectory(
    control: Tensor,
    frozen: FrozenOuterState,
) -> AnalysisTrajectory:
    height, width = frozen.initial_background_dbz.shape
    field_control = control[: height * width].reshape(height, width)
    dynamics_control = control[height * width :]
    config = frozen.analysis_config
    nowcast = frozen.nowcast_config

    floor_dbz = nowcast.min_dbz
    background_offset = (
        frozen.initial_background_dbz - floor_dbz
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
        * frozen.initial_support_mask
    )
    initial_echo = dbz_to_echo(
        floor_dbz + analyzed_offset,
        min_dbz=nowcast.min_dbz,
    )
    displacement, growth = _decode_dynamics(
        dynamics_control,
        frozen.baseline_state,
        config,
        nowcast,
    )
    frames = [initial_echo]
    for step in (1, 2):
        frames.append(
            advance(
                initial_echo,
                step * displacement,
                step * growth,
                frozen.analysis_remap_cells[step - 1],
            )
        )
    return AnalysisTrajectory(
        frames_linear=torch.stack(frames),
        displacement_yx=displacement,
        log_growth_per_step=growth,
    )


def observation_residual_dbz(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    _validate_observations(observations)
    _validate_control(control, frozen)
    return _observation_residual(control, observations, frozen)


def _observation_residual(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    trajectory = _analysis_trajectory(control, frozen)
    prediction = echo_to_dbz(
        trajectory.frames_linear,
        min_dbz=frozen.nowcast_config.min_dbz,
    )
    return _observation_residual_from_prediction(
        prediction,
        observations,
        frozen.analysis_config,
    )


def _observation_residual_from_prediction(
    prediction: Tensor,
    observations: AnalysisObservations,
    config: AnalysisConfig,
) -> Tensor:
    detected_error = prediction - observations.dbz
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
    frozen: FrozenOuterState,
) -> Tensor:
    _validate_observations(observations)
    _validate_control(control, frozen)
    return _whitened_observation_residual(control, observations, frozen)


def _whitened_observation_residual(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    return (
        torch.sqrt(observations.quality_weight)
        * _observation_residual(control, observations, frozen)
        / observations.std_dbz
    )


def freeze_irls_weights(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> FrozenOuterState:
    frozen = _freeze_analysis_remap_cells(control, frozen)
    residual = _whitened_observation_residual(
        control,
        observations,
        frozen,
    ).detach()
    delta = frozen.analysis_config.pseudo_huber_delta
    sqrt_weight = torch.pow(1.0 + (residual / delta).square(), -0.25)
    return replace(
        frozen,
        irls_sqrt_weight=torch.where(
            observations.valid_mask,
            sqrt_weight,
            torch.zeros_like(sqrt_weight),
        ),
    )


def residual_vector(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    weighted = (
        _whitened_observation_residual(control, observations, frozen)
        * frozen.irls_sqrt_weight
    )
    return torch.cat((weighted.reshape(-1), control))


def robust_objective(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    return _robust_objective_from_residual(
        control,
        _whitened_observation_residual(control, observations, frozen),
        observations,
        frozen.analysis_config,
    )


def _robust_objective_from_residual(
    control: Tensor,
    residual: Tensor,
    observations: AnalysisObservations,
    config: AnalysisConfig,
) -> Tensor:
    delta = config.pseudo_huber_delta
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
    frozen: FrozenOuterState,
    *,
    control: Tensor | None = None,
) -> AnalysisResult:
    _validate_observations(observations)
    reference_control = initial_control(observations)
    reference_frozen = _freeze_analysis_remap_cells(
        reference_control,
        frozen,
    )
    try:
        reference_cost_tensor, reference_trajectory = _evaluate_control(
            reference_control,
            observations,
            reference_frozen,
        )
    except EchoPositivityError:
        return _fallback_result(
            frozen,
            reference_control,
            math.inf,
            "positivity_violation",
        )
    reference_cost = float(reference_cost_tensor.detach())

    control = (
        _warm_started_control(observations, frozen)
        if control is None
        else control.detach().clone()
    )
    _validate_control(control, frozen)
    if torch.equal(control, reference_control):
        frozen = reference_frozen
        current_cost_tensor = reference_cost_tensor
        current_trajectory = reference_trajectory
    else:
        frozen = _freeze_analysis_remap_cells(control, frozen)
        try:
            current_cost_tensor, current_trajectory = _evaluate_control(
                control,
                observations,
                frozen,
            )
        except EchoPositivityError:
            return _fallback_result(
                frozen,
                control,
                reference_cost,
                "positivity_violation",
            )
    current_cost = float(current_cost_tensor.detach())
    current_amplitude_fraction = _unresolved_amplitude_fraction(
        observations,
        frozen,
        current_trajectory,
    )
    if not bool(torch.any(observations.valid_mask)):
        return _fallback_result(
            frozen,
            control,
            reference_cost,
            "no_valid_observations",
        )
    if not bool(torch.any(frozen.initial_support_mask)):
        return _fallback_result(
            frozen,
            control,
            reference_cost,
            "no_initial_state_support",
        )
    if not math.isfinite(reference_cost):
        return _fallback_result(
            frozen,
            control,
            reference_cost,
            "nonfinite_reference_objective",
        )
    if not math.isfinite(current_cost):
        return _fallback_result(
            frozen,
            control,
            reference_cost,
            "nonfinite_initial_objective",
        )

    config = frozen.analysis_config
    field_size = frozen.initial_background_dbz.numel()
    damping = config.initial_damping
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
            frozen,
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
            return _failed_result(
                accepted_any,
                control,
                observations,
                frozen,
                reference_cost,
                current_cost,
                completed_iterations,
                total_pcg_iterations,
                "nonfinite_gradient",
            )
        if gradient_norm <= config.gradient_tolerance:
            converged = True
            reason = "gradient_tolerance"
            break

        accepted = False
        linear_system_solved = False
        for _ in range(config.maximum_damping_retries + 1):
            operator = lambda vector: gauss_newton_hvp(
                residual_fn,
                control,
                vector,
                damping=damping,
            )
            try:
                linear = pcg(
                    operator,
                    -gradient,
                    rtol=config.pcg_relative_tolerance,
                    max_iterations=config.maximum_pcg_iterations,
                )
            except (ArithmeticError, RuntimeError, ValueError):
                linear = None
            if linear is None:
                damping = min(config.maximum_damping, 4.0 * damping)
                continue
            total_pcg_iterations += linear.iterations
            if not linear.converged or not bool(
                torch.all(torch.isfinite(linear.solution))
            ):
                damping = min(config.maximum_damping, 4.0 * damping)
                continue
            linear_system_solved = True
            raw_step = linear.solution
            hessian_step = gauss_newton_hvp(
                residual_fn,
                control,
                raw_step,
            )
            directional_gradient = torch.dot(gradient, raw_step)
            directional_curvature = torch.dot(raw_step, hessian_step)
            for backtrack in range(12):
                scale = 0.5**backtrack
                step = scale * raw_step
                predicted = float(
                    (
                        -scale * directional_gradient
                        - 0.5 * scale**2 * directional_curvature
                    ).detach()
                )
                candidate = control + step
                candidate_displacement, _ = _decode_dynamics(
                    candidate[field_size:],
                    frozen_iteration.baseline_state,
                    config,
                    frozen_iteration.nowcast_config,
                )
                if not _analysis_window_is_representable(
                    frozen_iteration,
                    candidate_displacement,
                ):
                    continue
                candidate_frozen = _freeze_analysis_remap_cells(
                    candidate,
                    frozen_iteration,
                )
                try:
                    candidate_cost_tensor, candidate_trajectory = (
                        _evaluate_control(
                            candidate,
                            observations,
                            candidate_frozen,
                        )
                    )
                    candidate_cost = float(candidate_cost_tensor.detach())
                except EchoPositivityError:
                    continue
                candidate_amplitude_fraction = (
                    _unresolved_amplitude_fraction(
                        observations,
                        candidate_frozen,
                        candidate_trajectory,
                    )
                )
                if not _amplitude_trial_is_admissible(
                    current_amplitude_fraction,
                    candidate_amplitude_fraction,
                    config.maximum_unresolved_amplitude_fraction,
                ):
                    continue
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
                current_amplitude_fraction = candidate_amplitude_fraction
                accepted_any = True
                accepted = True
                if ratio > 0.75:
                    damping = max(config.minimum_damping, 0.5 * damping)
                elif ratio < 0.25:
                    damping = min(config.maximum_damping, 2.0 * damping)
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
            if accepted:
                break
            damping = min(config.maximum_damping, 4.0 * damping)

        if not accepted:
            failure_reason = (
                "no_accepted_step"
                if linear_system_solved
                else "pcg_failed"
            )
            return _failed_result(
                accepted_any,
                control,
                observations,
                frozen,
                reference_cost,
                current_cost,
                completed_iterations,
                total_pcg_iterations,
                failure_reason,
            )
        if converged:
            break

    if not accepted_any and not converged:
        return _fallback_result(
            frozen,
            control,
            reference_cost,
            "no_accepted_step",
            completed_iterations,
            total_pcg_iterations,
        )
    return _analysis_result(
        control,
        observations,
        frozen,
        reference_cost,
        current_cost,
        completed_iterations,
        total_pcg_iterations,
        converged,
        reason,
        degraded=not converged,
    )


def variational_nowcast(
    frames_dbz: Tensor,
    *,
    nowcast_config: NowcastConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    observation_std_dbz: float | Tensor | None = None,
    quality_weight: float | Tensor | None = None,
    qc_mask: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
    audit: bool = False,
) -> tuple[ForecastResult, AnalysisResult]:
    nowcast_config = nowcast_config or NowcastConfig()
    observations, frozen = prepare_analysis(
        frames_dbz,
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
        observation_std_dbz=observation_std_dbz,
        quality_weight=quality_weight,
        qc_mask=qc_mask,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
    )
    analysis = solve_analysis(observations, frozen)
    run = ForecastRunContract.from_inputs(
        nowcast_config,
        frames_dbz,
        observations.valid_mask[-1],
        background_frames_dbz,
    )
    forecast = forecast_from_state(
        analysis.state,
        analysis.metadata,
        nowcast_config,
        run=run,
        audit=audit,
    )
    return forecast, analysis


def _evaluate_control(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> tuple[Tensor, AnalysisTrajectory]:
    trajectory = _analysis_trajectory(control, frozen)
    frames, _ = validate_physical_echo(
        trajectory.frames_linear,
        name="analysis trial",
    )
    trajectory = replace(trajectory, frames_linear=frames)
    prediction = echo_to_dbz(
        trajectory.frames_linear,
        min_dbz=frozen.nowcast_config.min_dbz,
    )
    residual = (
        torch.sqrt(observations.quality_weight)
        * _observation_residual_from_prediction(
            prediction,
            observations,
            frozen.analysis_config,
        )
        / observations.std_dbz
    )
    return (
        _robust_objective_from_residual(
            control,
            residual,
            observations,
            frozen.analysis_config,
        ),
        trajectory,
    )


def _failed_result(
    accepted_any: bool,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    reference_cost: float,
    current_cost: float,
    outer_iterations: int,
    pcg_iterations: int,
    reason: str,
) -> AnalysisResult:
    if accepted_any:
        return _analysis_result(
            control,
            observations,
            frozen,
            reference_cost,
            current_cost,
            outer_iterations,
            pcg_iterations,
            False,
            reason,
            degraded=True,
        )
    return _fallback_result(
        frozen,
        control,
        reference_cost,
        reason,
        outer_iterations,
        pcg_iterations,
    )


def _analysis_result(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    reference_objective: float,
    final_objective: float,
    outer_iterations: int,
    pcg_iterations: int,
    converged: bool,
    reason: str,
    *,
    degraded: bool = False,
) -> AnalysisResult:
    frozen = _freeze_analysis_remap_cells(control, frozen)
    trajectory = _analysis_trajectory(control, frozen)
    reachability_margin = _analysis_window_reachability_margin(
        frozen,
        trajectory.displacement_yx,
    )
    if reachability_margin < 0:
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "unrepresentable_analysis_window",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=reachability_margin,
        )
    frames, audit = validate_physical_echo(
        trajectory.frames_linear,
        name="final analysis",
    )
    trajectory = replace(trajectory, frames_linear=frames)
    unresolved_amplitude_fraction = _unresolved_amplitude_fraction(
        observations,
        frozen,
        trajectory,
    )
    if (
        unresolved_amplitude_fraction
        > frozen.analysis_config.maximum_unresolved_amplitude_fraction
    ):
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "unresolved_growth_or_emergence",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=reachability_margin,
            unresolved_amplitude_fraction=unresolved_amplitude_fraction,
        )
    if not _objective_improves_reference(
        final_objective,
        reference_objective,
        control.dtype,
    ):
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "no_improvement_over_zero_control",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=reachability_margin,
            unresolved_amplitude_fraction=unresolved_amplitude_fraction,
        )
    state = RadarState(
        echo_linear=frames[-1],
        displacement_yx=trajectory.displacement_yx,
        log_growth_per_step=trajectory.log_growth_per_step,
    )
    initial_background_mask = torch.cat(
        (
            frozen.background_mask[:1],
            torch.zeros_like(frozen.background_mask[1:]),
        )
    )
    source_support, background_fraction = merge_current_support(
        frozen.observed_mask,
        initial_background_mask,
        trajectory.displacement_yx,
        frozen.nowcast_config,
    )
    background_used = background_fraction > frozen.nowcast_config.epsilon
    (
        causal_control_cell_count,
        causal_seed_cell_count,
        causal_seed_prior_cost,
    ) = _causal_seed_diagnostics(frozen)
    return AnalysisResult(
        control=control.detach(),
        state=_detach_state(state),
        metadata=replace(
            frozen.baseline_metadata,
            background_used=background_used,
            background_contribution_fraction=background_fraction,
            background_age_minutes=(
                frozen.background_age_minutes if background_used else None
            ),
            source_support=source_support.detach(),
            provenance="p1_variational_analysis",
        ),
        analyzed_frames_linear=frames.detach(),
        initial_objective=reference_objective,
        final_objective=final_objective,
        outer_iterations=outer_iterations,
        pcg_iterations=pcg_iterations,
        converged=converged,
        used_fallback=False,
        reason=reason,
        degraded=degraded,
        audit=audit,
        minimum_reachability_margin=reachability_margin,
        unresolved_amplitude_fraction=unresolved_amplitude_fraction,
        causal_control_cell_count=causal_control_cell_count,
        causal_seed_cell_count=causal_seed_cell_count,
        causal_seed_prior_cost=causal_seed_prior_cost,
    )


def _analysis_window_is_representable(
    frozen: FrozenOuterState,
    displacement_yx: Tensor,
) -> bool:
    return _analysis_window_reachability_margin(frozen, displacement_yx) >= 0


def _analysis_window_reachability_margin(
    frozen: FrozenOuterState,
    displacement_yx: Tensor,
) -> float:
    support = frozen.initial_support_mask.to(dtype=displacement_yx.dtype)
    threshold = frozen.analysis_config.minimum_control_reachability
    margins: list[Tensor] = []
    for step in (0, 1, 2):
        detected = frozen.detected_masks[step]
        if not bool(torch.any(detected)):
            continue
        reachable = support if step == 0 else remap(
            support,
            step * displacement_yx,
        )
        margins.append(torch.min(reachable[detected]) - threshold)
    if not margins:
        return 1.0 - threshold
    return float(torch.min(torch.stack(margins)).detach())


def _unresolved_amplitude_fraction(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    trajectory: AnalysisTrajectory,
) -> float:
    prediction_dbz = echo_to_dbz(
        trajectory.frames_linear,
        min_dbz=frozen.nowcast_config.min_dbz,
    )
    initial_detected = frozen.detected_masks[0].to(
        dtype=trajectory.displacement_yx.dtype
    )
    bad_weight = prediction_dbz.new_zeros(())
    total_weight = prediction_dbz.new_zeros(())
    amplitude_floor = (
        frozen.analysis_config.detection_limit_dbz
        - frozen.analysis_config.censor_temperature_dbz
    )

    for step in (1, 2):
        initial_reach = remap(
            initial_detected,
            step * trajectory.displacement_yx,
        )
        precursor_required = frozen.detected_masks[step] & (
            initial_reach
            < frozen.analysis_config.minimum_control_reachability
        )
        if not bool(torch.any(precursor_required)):
            continue

        local_prediction = F.max_pool2d(
            prediction_dbz[step][None, None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[0, 0]
        quality = observations.quality_weight[step]
        standardized_deficit = (
            torch.sqrt(quality)
            * (observations.dbz[step] - local_prediction)
            / observations.std_dbz[step]
        )
        unresolved = precursor_required & (
            (
                standardized_deficit
                > frozen.analysis_config.maximum_detected_error_std
            )
            | (local_prediction < amplitude_floor)
        )
        bad_weight = bad_weight + quality[unresolved].sum()
        total_weight = total_weight + quality[precursor_required].sum()

    if float(total_weight.detach()) <= 0:
        return 0.0
    return float((bad_weight / total_weight).detach())


def _amplitude_trial_is_admissible(
    current_fraction: float,
    candidate_fraction: float,
    maximum_fraction: float,
) -> bool:
    if candidate_fraction <= maximum_fraction:
        return True
    return (
        current_fraction > maximum_fraction
        and candidate_fraction < current_fraction
    )


def _objective_improves_reference(
    final_objective: float,
    reference_objective: float,
    dtype: torch.dtype,
) -> bool:
    tolerance = (
        32.0
        * torch.finfo(dtype).eps
        * max(1.0, abs(reference_objective))
    )
    return final_objective < reference_objective - tolerance


def _fallback_result(
    frozen: FrozenOuterState,
    control: Tensor,
    initial_objective: float,
    reason: str,
    outer_iterations: int = 0,
    pcg_iterations: int = 0,
    *,
    minimum_reachability_margin: float | None = None,
    unresolved_amplitude_fraction: float | None = None,
) -> AnalysisResult:
    frames = dbz_to_echo(
        frozen.baseline_frames_dbz,
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    )
    frames, audit = validate_physical_echo(
        frames,
        name="fallback analysis",
    )
    (
        causal_control_cell_count,
        causal_seed_cell_count,
        causal_seed_prior_cost,
    ) = _causal_seed_diagnostics(frozen)
    return AnalysisResult(
        control=torch.zeros_like(control),
        state=_detach_state(frozen.baseline_state),
        metadata=frozen.baseline_metadata,
        analyzed_frames_linear=frames.detach(),
        initial_objective=initial_objective,
        final_objective=initial_objective,
        outer_iterations=outer_iterations,
        pcg_iterations=pcg_iterations,
        converged=False,
        used_fallback=True,
        reason=reason,
        degraded=False,
        audit=audit,
        minimum_reachability_margin=minimum_reachability_margin,
        unresolved_amplitude_fraction=unresolved_amplitude_fraction,
        causal_control_cell_count=causal_control_cell_count,
        causal_seed_cell_count=causal_seed_cell_count,
        causal_seed_prior_cost=causal_seed_prior_cost,
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
    height, width = frozen.initial_background_dbz.shape
    displacement, _ = _decode_dynamics(
        control[height * width :],
        frozen.baseline_state,
        frozen.analysis_config,
        frozen.nowcast_config,
    )
    return replace(
        frozen,
        analysis_remap_cells=tuple(
            freeze_remap_cell(step * displacement)
            for step in (1, 2)
        ),
    )


def _softplus_inverse(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


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
    for name in (
        "valid_mask",
        "detected_mask",
        "censored_mask",
        "missing_mask",
        "qc_rejected_mask",
        "observed_clear_mask",
    ):
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
    if bool(
        torch.any(observations.missing_mask & observations.valid_mask)
    ):
        raise ValueError("missing observations cannot be valid")
    if bool(
        torch.any(observations.qc_rejected_mask & observations.valid_mask)
    ):
        raise ValueError("QC-rejected observations cannot be valid")


def _validate_control(
    control: Tensor,
    frozen: FrozenOuterState,
) -> None:
    height, width = frozen.initial_background_dbz.shape
    expected = height * width + 3
    if (
        control.ndim != 1
        or control.numel() != expected
        or not control.is_floating_point()
    ):
        raise ValueError(
            f"control must be a floating vector of length {expected}"
        )
    if control.device != frozen.initial_background_dbz.device:
        raise ValueError("control and frozen state must use the same device")
    if control.dtype != frozen.initial_background_dbz.dtype:
        raise ValueError("control and frozen state must use the same dtype")


def _detach_state(state: RadarState) -> RadarState:
    return RadarState(
        echo_linear=state.echo_linear.detach(),
        displacement_yx=state.displacement_yx.detach(),
        log_growth_per_step=state.log_growth_per_step.detach(),
    )


def _detach_metadata(metadata: ForecastMetadata) -> ForecastMetadata:
    return ForecastMetadata(
        data_status=metadata.data_status,
        coverage_by_frame=metadata.coverage_by_frame.detach(),
        background_used=metadata.background_used,
        background_contribution_fraction=(
            metadata.background_contribution_fraction
        ),
        background_age_minutes=metadata.background_age_minutes,
        source_support=metadata.source_support.detach(),
        motion_disagreement_px=metadata.motion_disagreement_px.detach(),
        growth_disagreement=metadata.growth_disagreement.detach(),
        tendency_pair_count=metadata.tendency_pair_count,
        tendency_source=metadata.tendency_source,
        provenance=metadata.provenance,
    )
