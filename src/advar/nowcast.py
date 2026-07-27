from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import torch
from torch import Tensor

from .diagnostics import (
    PositivityAudit,
    TransportAudit,
    audit_transport,
    validate_physical_echo,
)
from .physics import (
    FORECAST_INTEGRATOR_VERSION,
    RemapCell,
    dbz_to_echo,
    echo_to_dbz,
    freeze_remap_cell,
    remap,
    remap_core,
    react_core,
)


@dataclass(frozen=True)
class NowcastConfig:
    interval_minutes: int = 10
    horizon_minutes: int = 180
    min_dbz: float = -10.0
    max_dbz: float = 70.0
    echo_threshold_dbz: float = 5.0
    recent_weight: float = 2.0 / 3.0
    max_displacement_px: float = 20.0
    max_log_growth_per_step: float = math.log(1.35)
    growth_decay_minutes: float = 60.0
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if type(self.interval_minutes) is not int:
            raise TypeError("interval_minutes must be an integer")
        if type(self.horizon_minutes) is not int:
            raise TypeError("horizon_minutes must be an integer")
        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if self.horizon_minutes % self.interval_minutes:
            raise ValueError(
                "horizon_minutes must be divisible by interval_minutes"
            )
        numeric_values = (
            self.min_dbz,
            self.max_dbz,
            self.echo_threshold_dbz,
            self.recent_weight,
            self.max_displacement_px,
            self.max_log_growth_per_step,
            self.growth_decay_minutes,
            self.epsilon,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("all numeric configuration values must be finite")
        if self.min_dbz >= self.max_dbz:
            raise ValueError("min_dbz must be smaller than max_dbz")
        if not self.min_dbz <= self.echo_threshold_dbz <= self.max_dbz:
            raise ValueError("echo_threshold_dbz must be inside the dBZ range")
        if not 0.0 <= self.recent_weight <= 1.0:
            raise ValueError("recent_weight must be between 0 and 1")
        if self.max_displacement_px <= 0:
            raise ValueError("max_displacement_px must be positive")
        if self.max_log_growth_per_step < 0:
            raise ValueError("max_log_growth_per_step cannot be negative")
        if self.growth_decay_minutes <= 0:
            raise ValueError("growth_decay_minutes must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def forecast_steps(self) -> int:
        return self.horizon_minutes // self.interval_minutes


class DataStatus(str, Enum):
    OBSERVED = "OBSERVED"
    PARTIAL = "PARTIAL"
    STALE_BACKGROUND = "STALE_BACKGROUND"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class RadarState:
    echo_linear: Tensor
    displacement_yx: Tensor
    log_growth_per_step: Tensor


@dataclass(frozen=True)
class ForecastMetadata:
    data_status: DataStatus
    coverage_by_frame: Tensor
    background_used: bool
    background_age_minutes: float | None
    source_mask: Tensor | None
    motion_disagreement_px: Tensor
    growth_disagreement: Tensor
    provenance: str = "p0_fft_latest"


@dataclass(frozen=True)
class ForecastAudit:
    input_echo: PositivityAudit
    forecast_final: PositivityAudit
    transport: tuple[TransportAudit, ...]


@dataclass(frozen=True)
class ForecastResult:
    forecast_dbz: Tensor
    forecast_linear: Tensor
    valid_mask: Tensor | None
    state: RadarState
    metadata: ForecastMetadata
    audit: ForecastAudit | None = None


@dataclass(frozen=True)
class PreparedRadarInput:
    frames_dbz: Tensor
    observed_mask: Tensor
    missing_mask: Tensor
    qc_rejected_mask: Tensor
    observed_clear_mask: Tensor
    available_mask: Tensor
    data_status: DataStatus
    coverage_by_frame: Tensor
    background_used: bool
    background_age_minutes: float | None


def prepare_input(
    frames_dbz: Tensor,
    config: NowcastConfig,
    *,
    accepted_mask: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
) -> PreparedRadarInput:
    _validate_frames(frames_dbz)
    finite = torch.isfinite(frames_dbz)
    if accepted_mask is None:
        accepted = torch.ones_like(frames_dbz, dtype=torch.bool)
    else:
        if (
            accepted_mask.shape != frames_dbz.shape
            or accepted_mask.dtype != torch.bool
        ):
            raise ValueError(
                "accepted_mask must be boolean with the frame shape"
            )
        accepted = accepted_mask.to(device=frames_dbz.device)

    observed = finite & accepted
    missing = ~finite
    qc_rejected = finite & ~accepted
    clean_observations = torch.nan_to_num(
        frames_dbz,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    dense = torch.where(
        observed,
        clean_observations,
        clean_observations.new_full((), config.min_dbz),
    )
    available = observed.clone()
    background_used_mask = torch.zeros_like(observed)

    if background_frames_dbz is None:
        if background_age_minutes is not None:
            raise ValueError(
                "background_age_minutes requires background_frames_dbz"
            )
    else:
        if background_age_minutes is None:
            raise ValueError(
                "background_age_minutes is required with background_frames_dbz"
            )
        if (
            not math.isfinite(background_age_minutes)
            or background_age_minutes < 0
        ):
            raise ValueError(
                "background_age_minutes must be finite and non-negative"
            )

    if background_frames_dbz is not None:
        if (
            background_frames_dbz.shape != frames_dbz.shape
            or not background_frames_dbz.is_floating_point()
        ):
            raise ValueError(
                "background_frames_dbz must be floating with the frame shape"
            )
        background = background_frames_dbz.to(
            dtype=frames_dbz.dtype,
            device=frames_dbz.device,
        )
        background_finite = torch.isfinite(background)
        clean_background = torch.nan_to_num(
            background,
            nan=config.min_dbz,
            posinf=config.max_dbz,
            neginf=config.min_dbz,
        ).clamp(config.min_dbz, config.max_dbz)
        background_used_mask = ~available & background_finite
        dense = torch.where(background_used_mask, clean_background, dense)
        available = available | background_used_mask

    observed_count = int(observed.sum())
    observation_count = observed.numel()
    coverage_by_frame = observed.to(torch.float64).mean(dim=(1, 2))
    background_used = bool(torch.any(background_used_mask))
    if observed_count == 0:
        status = (
            DataStatus.STALE_BACKGROUND
            if bool(torch.any(available[-1]))
            else DataStatus.UNAVAILABLE
        )
    elif observed_count < observation_count:
        status = DataStatus.PARTIAL
    else:
        status = DataStatus.OBSERVED

    return PreparedRadarInput(
        frames_dbz=dense,
        observed_mask=observed,
        missing_mask=missing,
        qc_rejected_mask=qc_rejected,
        observed_clear_mask=observed
        & (clean_observations < config.echo_threshold_dbz),
        available_mask=available,
        data_status=status,
        coverage_by_frame=coverage_by_frame.detach(),
        background_used=background_used,
        background_age_minutes=(
            background_age_minutes if background_used else None
        ),
    )


def estimate_state(
    frames_dbz: Tensor,
    config: NowcastConfig,
    *,
    qc_mask: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
) -> tuple[RadarState, ForecastMetadata]:
    prepared = prepare_input(
        frames_dbz,
        config,
        accepted_mask=qc_mask,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
    )
    return estimate_prepared_state(prepared, config)


def estimate_prepared_state(
    prepared: PreparedRadarInput,
    config: NowcastConfig,
) -> tuple[RadarState, ForecastMetadata]:
    frames = prepared.frames_dbz
    linear = dbz_to_echo(
        frames,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    linear, _ = validate_physical_echo(linear, name="input echo conversion")
    displacement, growth, motion_disagreement, growth_disagreement = (
        _estimate_time_normalized_tendencies(prepared, linear, config)
    )
    current_echo, current_source_mask = _current_state_from_available_frame(
        prepared,
        linear,
        displacement,
        growth,
        config,
    )
    state = RadarState(
        echo_linear=current_echo,
        displacement_yx=displacement,
        log_growth_per_step=growth,
    )
    metadata = ForecastMetadata(
        data_status=prepared.data_status,
        coverage_by_frame=prepared.coverage_by_frame,
        background_used=prepared.background_used,
        background_age_minutes=prepared.background_age_minutes,
        source_mask=(
            None
            if bool(torch.all(current_source_mask))
            else current_source_mask.detach().clone()
        ),
        motion_disagreement_px=motion_disagreement.detach(),
        growth_disagreement=growth_disagreement.detach(),
    )
    return state, metadata


def _estimate_time_normalized_tendencies(
    prepared: PreparedRadarInput,
    linear: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    estimates = []
    for previous_index, current_index in ((0, 1), (1, 2)):
        estimate = _estimate_available_pair(
            prepared,
            linear,
            previous_index,
            current_index,
            config,
        )
        if estimate is not None:
            estimates.append(estimate)

    if not estimates:
        long_estimate = _estimate_available_pair(
            prepared,
            linear,
            0,
            2,
            config,
        )
        if long_estimate is not None:
            estimates.append(long_estimate)

    zero_motion = linear.new_zeros(2)
    zero_growth = linear.new_zeros(())
    if not estimates:
        return zero_motion, zero_growth, zero_growth, zero_growth
    if len(estimates) == 1:
        motion, growth = estimates[0]
        return motion, growth, zero_growth, zero_growth

    first_motion, first_growth = estimates[0]
    second_motion, second_growth = estimates[1]
    weight = config.recent_weight
    return (
        (1.0 - weight) * first_motion + weight * second_motion,
        (1.0 - weight) * first_growth + weight * second_growth,
        torch.linalg.vector_norm(second_motion - first_motion),
        torch.abs(second_growth - first_growth),
    )


def _estimate_available_pair(
    prepared: PreparedRadarInput,
    linear: Tensor,
    previous_index: int,
    current_index: int,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor] | None:
    common = (
        prepared.available_mask[previous_index]
        & prepared.available_mask[current_index]
    )
    if not bool(torch.any(common)):
        return None

    floor = prepared.frames_dbz.new_full((), config.min_dbz)
    previous_dbz = torch.where(
        common,
        prepared.frames_dbz[previous_index],
        floor,
    )
    current_dbz = torch.where(
        common,
        prepared.frames_dbz[current_index],
        floor,
    )
    previous_signal = (
        previous_dbz - config.echo_threshold_dbz
    ).clamp_min(0.0)
    current_signal = (
        current_dbz - config.echo_threshold_dbz
    ).clamp_min(0.0)
    if (
        float(torch.linalg.vector_norm(previous_signal)) <= config.epsilon
        or float(torch.linalg.vector_norm(current_signal)) <= config.epsilon
    ):
        return None

    step_span = current_index - previous_index
    total_motion = _phase_correlation_shift(
        previous_dbz,
        current_dbz,
        config,
        max_displacement_px=config.max_displacement_px * step_span,
    )
    previous_echo = torch.where(
        common,
        linear[previous_index],
        linear.new_zeros(()),
    )
    current_echo = torch.where(
        common,
        linear[current_index],
        linear.new_zeros(()),
    )
    total_growth = _log_aligned_growth(
        previous_echo,
        current_echo,
        total_motion,
        config,
        max_log_growth=config.max_log_growth_per_step * step_span,
    )
    return total_motion / step_span, total_growth / step_span


def _current_state_from_available_frame(
    prepared: PreparedRadarInput,
    linear: Tensor,
    displacement: Tensor,
    growth: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor]:
    latest_mask = prepared.available_mask[2]
    if bool(torch.all(latest_mask)):
        return linear[2], latest_mask

    retention = math.exp(
        -config.interval_minutes / config.growth_decay_minutes
    )
    current_echo = torch.zeros_like(linear[2])
    current_mask = torch.zeros_like(latest_mask)
    for source_index in range(3):
        source_mask = prepared.available_mask[source_index]
        if not bool(torch.any(source_mask)):
            continue

        steps = 2 - source_index
        candidate_echo = torch.where(
            source_mask,
            linear[source_index],
            linear.new_zeros(()),
        )
        candidate_mask = source_mask
        if steps:
            total_displacement = steps * displacement
            candidate_echo = remap(candidate_echo, total_displacement)
            growth_sum = sum(retention**power for power in range(steps))
            candidate_echo = react_core(candidate_echo, growth * growth_sum)
            candidate_mask = (
                remap(
                    source_mask.to(dtype=linear.dtype),
                    total_displacement,
                )
                > config.epsilon
            )

        current_echo = torch.where(
            candidate_mask,
            candidate_echo,
            current_echo,
        )
        current_mask = current_mask | candidate_mask

    return current_echo, current_mask


def forecast_linear_at_step(
    state: RadarState,
    step: int,
    config: NowcastConfig,
) -> Tensor:
    if not 1 <= step <= config.forecast_steps:
        raise ValueError("step must be inside the configured forecast horizon")
    displacement = step * state.displacement_yx
    return _forecast_linear_at_step_core(
        state,
        step,
        config,
        freeze_remap_cell(displacement),
    )


def _forecast_linear_at_step_core(
    state: RadarState,
    step: int,
    config: NowcastConfig,
    cell: RemapCell,
) -> Tensor:
    retention = math.exp(
        -config.interval_minutes / config.growth_decay_minutes
    )
    growth_sum = sum(retention**power for power in range(step))
    displacement = step * state.displacement_yx
    return react_core(
        remap_core(state.echo_linear, displacement, cell),
        state.log_growth_per_step * growth_sum,
    )


def forecast_linear_from_state(
    state: RadarState,
    config: NowcastConfig,
) -> Tensor:
    echo, _ = validate_physical_echo(
        state.echo_linear,
        name="forecast input state",
    )
    state = replace(state, echo_linear=echo)
    return torch.stack(
        [
            forecast_linear_at_step(state, step, config)
            for step in range(1, config.forecast_steps + 1)
        ]
    )


def forecast_from_state(
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
    *,
    audit: bool = False,
) -> ForecastResult:
    input_echo, input_audit = validate_physical_echo(
        state.echo_linear,
        name="forecast input state",
    )
    state = replace(state, echo_linear=input_echo)
    forecasts = []
    transport_audits = []
    for step in range(1, config.forecast_steps + 1):
        displacement = step * state.displacement_yx
        cell = freeze_remap_cell(displacement)
        moved = remap_core(state.echo_linear, displacement, cell)
        if audit:
            transport_audits.append(
                audit_transport(
                    state.echo_linear,
                    displacement,
                    cell=cell,
                    moved=moved,
                )
            )
        retention = math.exp(
            -config.interval_minutes / config.growth_decay_minutes
        )
        growth_sum = sum(retention**power for power in range(step))
        forecasts.append(
            react_core(
                moved,
                state.log_growth_per_step * growth_sum,
            )
        )

    forecast_linear, final_audit = validate_physical_echo(
        torch.stack(forecasts),
        name="final forecast",
    )
    forecast_dbz = echo_to_dbz(
        forecast_linear,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    valid_mask = _forecast_valid_mask(state, metadata, config)
    if valid_mask is not None:
        forecast_dbz = torch.where(
            valid_mask,
            forecast_dbz,
            forecast_dbz.new_full((), torch.nan),
        )
    return ForecastResult(
        forecast_dbz=forecast_dbz,
        forecast_linear=forecast_linear,
        valid_mask=valid_mask,
        state=state,
        metadata=metadata,
        audit=(
            ForecastAudit(
                input_echo=input_audit,
                forecast_final=final_audit,
                transport=tuple(transport_audits),
            )
            if audit
            else None
        ),
    )


def nowcast(
    frames_dbz: Tensor,
    config: NowcastConfig | None = None,
    *,
    qc_mask: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
    audit: bool = False,
) -> ForecastResult:
    config = config or NowcastConfig()
    state, metadata = estimate_state(
        frames_dbz,
        config,
        qc_mask=qc_mask,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
    )
    return forecast_from_state(
        state,
        metadata,
        config,
        audit=audit,
    )


def _forecast_valid_mask(
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
) -> Tensor | None:
    if metadata.data_status == DataStatus.UNAVAILABLE:
        return torch.zeros(
            (config.forecast_steps,) + state.echo_linear.shape,
            dtype=torch.bool,
            device=state.echo_linear.device,
        )
    if metadata.source_mask is None:
        return None
    source = metadata.source_mask.to(
        dtype=state.echo_linear.dtype,
        device=state.echo_linear.device,
    )
    return torch.stack(
        [
            remap(
                source,
                step * state.displacement_yx,
            )
            > config.epsilon
            for step in range(1, config.forecast_steps + 1)
        ]
    )


def _validate_frames(frames: Tensor) -> None:
    if frames.ndim != 3 or frames.shape[0] != 3:
        raise ValueError("frames_dbz must have shape [3, height, width]")
    if frames.shape[1] < 2 or frames.shape[2] < 2:
        raise ValueError("frame height and width must both be at least 2")
    if not frames.is_floating_point():
        raise TypeError("frames_dbz must be a floating-point tensor")


def _log_aligned_growth(
    previous: Tensor,
    current: Tensor,
    displacement_yx: Tensor,
    config: NowcastConfig,
    *,
    max_log_growth: float | None = None,
) -> Tensor:
    limit = (
        config.max_log_growth_per_step
        if max_log_growth is None
        else max_log_growth
    )
    aligned = remap(previous, displacement_yx)
    valid = _valid_advection_mask(previous.shape, displacement_yx)
    if int(valid.sum()) < 4:
        return previous.new_zeros(())

    previous_integrated_echo = aligned[valid].sum()
    current_integrated_echo = current[valid].sum()
    if float(previous_integrated_echo.detach()) <= config.epsilon:
        if float(current_integrated_echo.detach()) <= config.epsilon:
            return previous_integrated_echo.new_zeros(())
        return previous_integrated_echo.new_tensor(limit)
    growth = torch.log(
        (current_integrated_echo + config.epsilon)
        / (previous_integrated_echo + config.epsilon)
    )
    return growth.clamp(
        -limit,
        limit,
    )


def _valid_advection_mask(
    shape: torch.Size,
    displacement_yx: Tensor,
) -> Tensor:
    height, width = shape
    y = torch.arange(
        height,
        dtype=displacement_yx.dtype,
        device=displacement_yx.device,
    )
    x = torch.arange(
        width,
        dtype=displacement_yx.dtype,
        device=displacement_yx.device,
    )
    source_y = y[:, None] - displacement_yx[0]
    source_x = x[None, :] - displacement_yx[1]
    return (
        (source_y >= 0.0)
        & (source_y <= height - 1)
        & (source_x >= 0.0)
        & (source_x <= width - 1)
    )


def _phase_correlation_shift(
    previous_dbz: Tensor,
    current_dbz: Tensor,
    config: NowcastConfig,
    *,
    max_displacement_px: float | None = None,
) -> Tensor:
    previous = (previous_dbz - config.echo_threshold_dbz).clamp_min(0.0)
    current = (current_dbz - config.echo_threshold_dbz).clamp_min(0.0)

    energy = (
        torch.linalg.vector_norm(previous)
        * torch.linalg.vector_norm(current)
    )
    if float(energy.detach()) <= config.epsilon:
        return previous.new_zeros(2)

    height, width = previous.shape
    previous = previous - previous.mean()
    current = current - current.mean()
    centered_energy = (
        torch.linalg.vector_norm(previous)
        * torch.linalg.vector_norm(current)
    )
    if float(centered_energy.detach()) <= config.epsilon:
        return previous.new_zeros(2)

    padded_shape = (2 * height, 2 * width)
    cross_power = torch.fft.fft2(current, s=padded_shape) * torch.conj(
        torch.fft.fft2(previous, s=padded_shape)
    )
    cross_power = cross_power / cross_power.abs().clamp_min(config.epsilon)
    correlation = torch.fft.ifft2(cross_power).real

    peak_index = int(torch.argmax(correlation).item())
    correlation_height, correlation_width = correlation.shape
    peak_y, peak_x = divmod(peak_index, correlation_width)
    offset_y = _parabolic_peak_offset(correlation[:, peak_x], peak_y, config)
    offset_x = _parabolic_peak_offset(correlation[peak_y, :], peak_x, config)

    shift_y = peak_y + offset_y
    shift_x = peak_x + offset_x
    if shift_y > correlation_height / 2:
        shift_y -= correlation_height
    if shift_x > correlation_width / 2:
        shift_x -= correlation_width

    shift = correlation.new_tensor((shift_y, shift_x))
    limits = correlation.new_tensor(
        (
            min(
                (
                    config.max_displacement_px
                    if max_displacement_px is None
                    else max_displacement_px
                ),
                height - 1,
            ),
            min(
                (
                    config.max_displacement_px
                    if max_displacement_px is None
                    else max_displacement_px
                ),
                width - 1,
            ),
        )
    )
    return torch.maximum(torch.minimum(shift, limits), -limits)


def _parabolic_peak_offset(
    values: Tensor,
    peak: int,
    config: NowcastConfig,
) -> float:
    left = values[(peak - 1) % values.numel()]
    center = values[peak]
    right = values[(peak + 1) % values.numel()]
    denominator = left - 2.0 * center + right
    if abs(float(denominator.detach())) <= config.epsilon:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(offset.clamp(-0.5, 0.5).detach())
