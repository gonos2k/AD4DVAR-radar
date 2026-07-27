"""Simple three-frame radar echo nowcasting.

The analysis estimates one global translation and one global growth rate.
Each forecast lead directly applies a local conservative echo remap.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import Tensor


FORECAST_INTEGRATOR_VERSION = "local-conservative-remap-positive-v1"


@dataclass(frozen=True)
class NowcastConfig:
    """Configuration for three-frame, 10-minute radar nowcasting."""

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
            raise ValueError("horizon_minutes must be divisible by interval_minutes")
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
        """Number of output frames."""

        return self.horizon_minutes // self.interval_minutes


@dataclass(frozen=True)
class RemapCell:
    """Integer transport cell held fixed during one linearization."""

    y: int
    x: int

    def __post_init__(self) -> None:
        if type(self.y) is not int or type(self.x) is not int:
            raise TypeError("remap cell coordinates must be integers")


class EchoPositivityError(FloatingPointError):
    pass


class FrozenCellMismatchError(ValueError):
    pass


@dataclass(frozen=True)
class EchoPositivityDiagnostics:
    minimum_echo_linear: float
    negative_echo_count_before_roundoff_fix: int
    negative_echo_integral_before_fix: float
    roundoff_correction_count: int
    roundoff_correction_integral: float
    positivity_gate_passed: bool


@dataclass(frozen=True)
class TransportDiagnostics:
    positivity: EchoPositivityDiagnostics
    minimum_transport_weight: float
    maximum_transport_weight: float
    transport_weight_sum_error: float
    echo_integral_before_transport: float
    echo_integral_after_transport: float
    boundary_outflow_integral: float
    echo_budget_error: float


class ForecastStatus(str, Enum):
    """Operational meaning of the data used to initialize a forecast."""

    OBSERVED = "OBSERVED"
    PARTIAL_OBSERVATION = "PARTIAL_OBSERVATION"
    STALE_BACKGROUND = "STALE_BACKGROUND"
    UNAVAILABLE = "UNAVAILABLE"
    BASELINE_FALLBACK = "BASELINE_FALLBACK"


@dataclass(frozen=True)
class _PreparedRadarInput:
    """Dense calculation frames plus the original observation semantics."""

    frames_dbz: Tensor
    observed_mask: Tensor
    missing_mask: Tensor
    qc_rejected_mask: Tensor
    observed_clear_mask: Tensor
    available_mask: Tensor
    forecast_status: ForecastStatus
    data_coverage_fraction: float
    latest_data_coverage_fraction: float
    background_used: bool
    background_age_minutes: float | None


@dataclass(frozen=True)
class RadarState:
    """Low-dimensional state inferred from the three input frames."""

    echo_amplitude: Tensor
    displacement_yx: Tensor
    log_growth_per_step: Tensor
    pair_displacements_yx: Tensor
    pair_log_growth: Tensor
    provenance: str = "p0_fft_latest"
    forecast_status: ForecastStatus = ForecastStatus.OBSERVED
    data_coverage_fraction: float = 1.0
    latest_data_coverage_fraction: float = 1.0
    background_used: bool = False
    background_age_minutes: float | None = None
    forecast_source_mask: Tensor | None = None
    positivity_diagnostics: EchoPositivityDiagnostics | None = None

    @property
    def echo_linear(self) -> Tensor:
        """Latest non-negative linear echo amount."""

        return self.echo_amplitude.square()

    @property
    def motion_disagreement_px(self) -> Tensor:
        """Difference between the two independently estimated motions."""

        return torch.linalg.vector_norm(
            self.pair_displacements_yx[1] - self.pair_displacements_yx[0]
        )

    @property
    def growth_disagreement(self) -> Tensor:
        """Absolute difference between the two growth estimates."""

        return torch.abs(self.pair_log_growth[1] - self.pair_log_growth[0])


def dbz_to_linear(dbz: Tensor, config: NowcastConfig) -> Tensor:
    """Convert dBZ to a non-negative linear echo amount."""

    echo, _ = _dbz_to_linear_with_diagnostics(dbz, config)
    return echo


def _dbz_to_linear_with_diagnostics(
    dbz: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, EchoPositivityDiagnostics]:
    clean = torch.nan_to_num(
        dbz,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    floor = 10.0 ** (config.min_dbz / 10.0)
    return enforce_echo_positivity(
        torch.pow(10.0, clean / 10.0) - floor,
        name="dbz_to_linear",
    )


def linear_to_dbz(echo: Tensor, config: NowcastConfig) -> Tensor:
    """Convert the non-negative internal echo amount back to dBZ."""

    echo, _ = enforce_echo_positivity(echo, name="linear_to_dbz")
    floor = 10.0 ** (config.min_dbz / 10.0)
    dbz = 10.0 * torch.log10(echo + floor)
    return dbz.clamp(config.min_dbz, config.max_dbz)


def enforce_echo_positivity(
    echo: Tensor,
    *,
    name: str,
) -> tuple[Tensor, EchoPositivityDiagnostics]:
    """Reject physical negative echo and repair roundoff-sized negatives."""

    if not echo.is_floating_point():
        raise TypeError(f"{name}: echo must be floating point")
    if echo.numel() == 0:
        raise ValueError(f"{name}: echo must not be empty")
    detached = echo.detach()
    if not bool(torch.all(torch.isfinite(detached))):
        raise EchoPositivityError(f"{name}: non-finite physical echo")

    minimum = torch.amin(detached)
    tolerance_scale = torch.abs(minimum).clamp_min(1.0)
    tolerance = 32.0 * torch.finfo(echo.dtype).eps * tolerance_scale
    negative = detached < 0.0
    negative_count = int(torch.count_nonzero(negative))
    negative_integral = float(
        torch.where(negative, -detached, torch.zeros_like(detached)).sum()
    )
    if bool(minimum < -tolerance):
        raise EchoPositivityError(
            f"{name}: physical echo became negative "
            f"(minimum={float(minimum):.9g}, "
            f"tolerance={float(tolerance):.9g}, "
            f"count={negative_count})"
        )

    corrected = torch.where(echo < 0.0, torch.zeros_like(echo), echo)
    diagnostics = EchoPositivityDiagnostics(
        minimum_echo_linear=float(corrected.detach().min()),
        negative_echo_count_before_roundoff_fix=negative_count,
        negative_echo_integral_before_fix=negative_integral,
        roundoff_correction_count=negative_count,
        roundoff_correction_integral=negative_integral,
        positivity_gate_passed=True,
    )
    return corrected, diagnostics


def aggregate_echo_positivity_diagnostics(
    *items: EchoPositivityDiagnostics,
) -> EchoPositivityDiagnostics:
    if not items:
        raise ValueError("at least one positivity diagnostic is required")
    return EchoPositivityDiagnostics(
        minimum_echo_linear=min(item.minimum_echo_linear for item in items),
        negative_echo_count_before_roundoff_fix=sum(
            item.negative_echo_count_before_roundoff_fix for item in items
        ),
        negative_echo_integral_before_fix=sum(
            item.negative_echo_integral_before_fix for item in items
        ),
        roundoff_correction_count=sum(
            item.roundoff_correction_count for item in items
        ),
        roundoff_correction_integral=sum(
            item.roundoff_correction_integral for item in items
        ),
        positivity_gate_passed=all(
            item.positivity_gate_passed for item in items
        ),
    )


def freeze_remap_cell(displacement_yx: Tensor) -> RemapCell:
    """Return the integer transport cell for a fixed linearization point."""

    if displacement_yx.shape != (2,):
        raise ValueError("displacement_yx must have shape [2]")
    if not displacement_yx.is_floating_point():
        raise TypeError("displacement_yx must be floating point")
    detached = displacement_yx.detach()
    if not bool(torch.all(torch.isfinite(detached))):
        raise ValueError("displacement_yx must be finite")
    return RemapCell(
        y=math.floor(float(detached[0])),
        x=math.floor(float(detached[1])),
    )


def advect(
    echo: Tensor,
    displacement_yx: Tensor,
    *,
    frozen_cell: RemapCell | None = None,
) -> Tensor:
    """Move linear echo with a positive local conservative remap.

    ``frozen_cell`` is optional for ordinary forecasts. Matrix-free solvers
    pass the cell at their current outer iterate so every Krylov application
    uses the same piecewise-smooth transport operator.
    """

    moved, _ = _advect_with_diagnostics(
        echo,
        displacement_yx,
        frozen_cell,
    )
    return moved


def _advect_with_diagnostics(
    echo: Tensor,
    displacement_yx: Tensor,
    frozen_cell: RemapCell | None,
) -> tuple[Tensor, EchoPositivityDiagnostics]:
    if echo.ndim != 2:
        raise ValueError("echo must have shape [height, width]")
    if not echo.is_floating_point():
        raise TypeError("echo must be floating point")
    if displacement_yx.shape != (2,):
        raise ValueError("displacement_yx must have shape [2]")
    if not displacement_yx.is_floating_point():
        raise TypeError("displacement_yx must be floating point")
    height, width = echo.shape
    if height < 2 or width < 2:
        raise ValueError("echo height and width must both be at least 2")

    displacement = displacement_yx.to(dtype=echo.dtype, device=echo.device)
    detached_displacement = displacement.detach()
    if not bool(torch.all(torch.isfinite(detached_displacement))):
        raise ValueError("displacement_yx must be finite")
    echo, input_diagnostics = enforce_echo_positivity(
        echo,
        name="advect input",
    )
    moved = _local_conservative_remap(
        echo,
        displacement,
        frozen_cell,
    )
    moved, output_diagnostics = enforce_echo_positivity(
        moved,
        name="advect output",
    )
    return moved, aggregate_echo_positivity_diagnostics(
        input_diagnostics,
        output_diagnostics,
    )


def diagnose_transport(
    echo: Tensor,
    displacement_yx: Tensor,
    *,
    frozen_cell: RemapCell | None = None,
) -> TransportDiagnostics:
    """Audit one remap without changing the public advection result."""

    moved, positivity = _advect_with_diagnostics(
        echo,
        displacement_yx,
        frozen_cell,
    )
    echo, _ = enforce_echo_positivity(echo, name="transport diagnostic input")
    displacement = displacement_yx.to(dtype=echo.dtype, device=echo.device)
    _, _, fraction_y, fraction_x = _remap_cell_and_fraction(
        echo,
        displacement,
        frozen_cell,
    )
    weights = torch.stack(
        (
            (1.0 - fraction_y) * (1.0 - fraction_x),
            fraction_y * (1.0 - fraction_x),
            (1.0 - fraction_y) * fraction_x,
            fraction_y * fraction_x,
        )
    )
    before = echo.sum()
    after = moved.sum()
    height, width = echo.shape
    base_y, base_x, _, _ = _remap_cell_and_fraction(
        echo,
        displacement,
        frozen_cell,
    )
    source_y = torch.arange(height, device=echo.device)[:, None]
    source_x = torch.arange(width, device=echo.device)[None, :]
    outflow = echo.new_zeros(())
    weight_grid = (
        ((0, 0), weights[0]),
        ((1, 0), weights[1]),
        ((0, 1), weights[2]),
        ((1, 1), weights[3]),
    )
    for (offset_y, offset_x), weight in weight_grid:
        destination_y = source_y + base_y + offset_y
        destination_x = source_x + base_x + offset_x
        outside = (
            (destination_y < 0)
            | (destination_y >= height)
            | (destination_x < 0)
            | (destination_x >= width)
        )
        outflow = outflow + (echo * weight * outside).sum()
    budget_error = before - after - outflow
    return TransportDiagnostics(
        positivity=positivity,
        minimum_transport_weight=float(weights.detach().min()),
        maximum_transport_weight=float(weights.detach().max()),
        transport_weight_sum_error=float(
            torch.abs(weights.detach().sum() - 1.0)
        ),
        echo_integral_before_transport=float(before.detach()),
        echo_integral_after_transport=float(after.detach()),
        boundary_outflow_integral=float(outflow.detach()),
        echo_budget_error=float(torch.abs(budget_error.detach())),
    )


def _prepare_radar_input(
    frames_dbz: Tensor,
    config: NowcastConfig,
    *,
    accepted_mask: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
) -> _PreparedRadarInput:
    """Preserve missing/QC meaning while making dense calculation frames."""

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

    if background_age_minutes is not None:
        if background_frames_dbz is None:
            raise ValueError(
                "background_age_minutes requires background_frames_dbz"
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

    # A same-pixel observation from a neighboring input time is a transparent
    # persistence fill, not evidence that the target time was observed.
    source_orders = ((1, 2), (0, 2), (1, 0))
    for target, sources in enumerate(source_orders):
        for source in sources:
            fill = ~available[target] & observed[source]
            dense[target] = torch.where(
                fill,
                clean_observations[source],
                dense[target],
            )
            available[target] = available[target] | fill

    observed_count = int(observed.sum())
    observation_count = observed.numel()
    coverage = observed_count / observation_count
    latest_coverage = float(observed[-1].to(torch.float64).mean())
    background_used = bool(torch.any(background_used_mask))
    if observed_count == 0:
        status = (
            ForecastStatus.STALE_BACKGROUND
            if bool(torch.any(available[-1]))
            else ForecastStatus.UNAVAILABLE
        )
    elif observed_count < observation_count:
        status = ForecastStatus.PARTIAL_OBSERVATION
    else:
        status = ForecastStatus.OBSERVED

    return _PreparedRadarInput(
        frames_dbz=dense,
        observed_mask=observed,
        missing_mask=missing,
        qc_rejected_mask=qc_rejected,
        observed_clear_mask=observed
        & (clean_observations < config.echo_threshold_dbz),
        available_mask=available,
        forecast_status=status,
        data_coverage_fraction=coverage,
        latest_data_coverage_fraction=latest_coverage,
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
) -> RadarState:
    """Infer the latest echo, global motion, and global growth from 3 frames."""

    prepared = _prepare_radar_input(
        frames_dbz,
        config,
        accepted_mask=qc_mask,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
    )
    return _estimate_prepared_state(prepared, config)


def _estimate_prepared_state(
    prepared: _PreparedRadarInput,
    config: NowcastConfig,
) -> RadarState:
    frames = prepared.frames_dbz

    pair_motion = torch.stack(
        (
            _phase_correlation_shift(frames[0], frames[1], config),
            _phase_correlation_shift(frames[1], frames[2], config),
        )
    )

    linear, positivity = _dbz_to_linear_with_diagnostics(frames, config)
    pair_growth = torch.stack(
        (
            _log_aligned_growth(
                linear[0],
                linear[1],
                pair_motion[0],
                config,
            ),
            _log_aligned_growth(
                linear[1],
                linear[2],
                pair_motion[1],
                config,
            ),
        )
    )

    weight = config.recent_weight
    displacement = (1.0 - weight) * pair_motion[0] + weight * pair_motion[1]
    growth = (1.0 - weight) * pair_growth[0] + weight * pair_growth[1]
    return RadarState(
        echo_amplitude=torch.sqrt(linear[2]),
        displacement_yx=displacement,
        log_growth_per_step=growth,
        pair_displacements_yx=pair_motion,
        pair_log_growth=pair_growth,
        forecast_status=prepared.forecast_status,
        data_coverage_fraction=prepared.data_coverage_fraction,
        latest_data_coverage_fraction=(
            prepared.latest_data_coverage_fraction
        ),
        background_used=prepared.background_used,
        background_age_minutes=prepared.background_age_minutes,
        forecast_source_mask=(
            None
            if bool(torch.all(prepared.available_mask[-1]))
            else prepared.available_mask[-1].detach().clone()
        ),
        positivity_diagnostics=positivity,
    )


def forecast_linear_from_state(
    state: RadarState,
    config: NowcastConfig,
) -> Tensor:
    """Forecast non-negative linear echo before output clipping."""

    return torch.stack(
        [
            forecast_linear_at_step(state, step, config)
            for step in range(1, config.forecast_steps + 1)
        ]
    )


def forecast_linear_at_step(
    state: RadarState,
    step: int,
    config: NowcastConfig,
) -> Tensor:
    """Forecast one lead in linear echo space."""

    if not 1 <= step <= config.forecast_steps:
        raise ValueError("step must be inside the configured forecast horizon")
    retention = math.exp(-config.interval_minutes / config.growth_decay_minutes)
    growth_sum = sum(retention**power for power in range(step))
    echo = advect(
        state.echo_linear,
        step * state.displacement_yx,
    )
    forecast, _ = enforce_echo_positivity(
        echo
        * torch.exp(
            state.log_growth_per_step * growth_sum
        ),
        name=f"forecast lead {step}",
    )
    return forecast


def forecast_from_state(state: RadarState, config: NowcastConfig) -> Tensor:
    """Forecast output dBZ frames from an already estimated state."""

    forecast = linear_to_dbz(
        forecast_linear_from_state(state, config),
        config,
    )
    if state.forecast_status == ForecastStatus.UNAVAILABLE:
        return torch.full_like(forecast, torch.nan)
    if state.forecast_source_mask is None:
        return forecast

    source = state.forecast_source_mask.to(
        dtype=state.echo_amplitude.dtype,
        device=state.echo_amplitude.device,
    )
    available = torch.stack(
        [
            advect(source, step * state.displacement_yx)
            > config.epsilon
            for step in range(1, config.forecast_steps + 1)
        ]
    )
    return torch.where(
        available,
        forecast,
        forecast.new_full((), torch.nan),
    )


def nowcast(
    frames_dbz: Tensor,
    config: NowcastConfig | None = None,
    *,
    qc_mask: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
) -> tuple[Tensor, RadarState]:
    """Estimate state from 3 frames and return the next 3 hours of dBZ."""

    config = config or NowcastConfig()
    state = estimate_state(
        frames_dbz,
        config,
        qc_mask=qc_mask,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
    )
    return forecast_from_state(state, config), state


def _validate_frames(frames: Tensor) -> None:
    if frames.ndim != 3 or frames.shape[0] != 3:
        raise ValueError("frames_dbz must have shape [3, height, width]")
    if frames.shape[1] < 2 or frames.shape[2] < 2:
        raise ValueError("frame height and width must both be at least 2")
    if not frames.is_floating_point():
        raise TypeError("frames_dbz must be a floating-point tensor")


def _local_conservative_remap(
    echo: Tensor,
    displacement_yx: Tensor,
    frozen_cell: RemapCell | None,
) -> Tensor:
    """Deposit each source cell into its four local destination cells."""

    height, width = echo.shape
    base_y, base_x, fraction_y, fraction_x = _remap_cell_and_fraction(
        echo,
        displacement_yx,
        frozen_cell,
    )
    source_y = torch.arange(height, device=echo.device)[:, None]
    source_x = torch.arange(width, device=echo.device)[None, :]
    output = echo.new_zeros(height * width)

    y_options = ((0, 1.0 - fraction_y), (1, fraction_y))
    x_options = ((0, 1.0 - fraction_x), (1, fraction_x))
    for offset_y, weight_y in y_options:
        destination_y = source_y + base_y + offset_y
        valid_y = (destination_y >= 0) & (destination_y < height)
        for offset_x, weight_x in x_options:
            destination_x = source_x + base_x + offset_x
            valid = (
                valid_y
                & (destination_x >= 0)
                & (destination_x < width)
            )
            flat_index = (
                destination_y.clamp(0, height - 1) * width
                + destination_x.clamp(0, width - 1)
            )
            contribution = echo * weight_y * weight_x * valid
            output = output.scatter_add(
                0,
                flat_index.reshape(-1),
                contribution.reshape(-1),
            )
    return output.reshape(height, width)


def _remap_cell_and_fraction(
    echo: Tensor,
    displacement_yx: Tensor,
    frozen_cell: RemapCell | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dy, dx = displacement_yx
    if frozen_cell is None:
        base_y = torch.floor(dy).to(torch.int64)
        base_x = torch.floor(dx).to(torch.int64)
    else:
        base_y = torch.as_tensor(
            frozen_cell.y,
            dtype=torch.int64,
            device=echo.device,
        )
        base_x = torch.as_tensor(
            frozen_cell.x,
            dtype=torch.int64,
            device=echo.device,
        )

    fraction_y = dy - base_y.to(dtype=echo.dtype)
    fraction_x = dx - base_x.to(dtype=echo.dtype)
    if frozen_cell is not None:
        detached = displacement_yx.detach()
        scale = torch.amax(torch.abs(detached)).clamp_min(1.0)
        tolerance = 32.0 * torch.finfo(echo.dtype).eps * scale
        fractions = torch.stack((fraction_y.detach(), fraction_x.detach()))
        if bool(torch.any(fractions < -tolerance)) or bool(
            torch.any(fractions > 1.0 + tolerance)
        ):
            raise FrozenCellMismatchError(
                "frozen remap cell is inconsistent with displacement "
                f"(cell=({frozen_cell.y}, {frozen_cell.x}), "
                f"displacement=({float(detached[0]):.9g}, "
                f"{float(detached[1]):.9g}))"
            )
        fraction_y = fraction_y.clamp(0.0, 1.0)
        fraction_x = fraction_x.clamp(0.0, 1.0)
    return base_y, base_x, fraction_y, fraction_x


def _log_aligned_growth(
    previous: Tensor,
    current: Tensor,
    displacement_yx: Tensor,
    config: NowcastConfig,
) -> Tensor:
    aligned = advect(previous, displacement_yx)
    valid = _valid_advection_mask(previous.shape, displacement_yx)
    if int(valid.sum()) < 4:
        return previous.new_zeros(())

    previous_mass = aligned[valid].sum()
    current_mass = current[valid].sum()

    if float(previous_mass.detach()) <= config.epsilon:
        if float(current_mass.detach()) <= config.epsilon:
            return previous_mass.new_zeros(())
        return previous_mass.new_tensor(config.max_log_growth_per_step)

    growth = torch.log(
        (current_mass + config.epsilon) / (previous_mass + config.epsilon)
    )
    return growth.clamp(
        -config.max_log_growth_per_step,
        config.max_log_growth_per_step,
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
) -> Tensor:
    """Estimate the translation from ``previous`` to ``current``."""

    previous = (previous_dbz - config.echo_threshold_dbz).clamp_min(0.0)
    current = (current_dbz - config.echo_threshold_dbz).clamp_min(0.0)

    energy = torch.linalg.vector_norm(previous) * torch.linalg.vector_norm(current)
    if float(energy.detach()) <= config.epsilon:
        return previous.new_zeros(2)

    height, width = previous.shape
    previous = previous - previous.mean()
    current = current - current.mean()
    centered_energy = (
        torch.linalg.vector_norm(previous) * torch.linalg.vector_norm(current)
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
            min(config.max_displacement_px, (height - 1) / 2.0),
            min(config.max_displacement_px, (width - 1) / 2.0),
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
