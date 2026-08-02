from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ._digest import dataclass_digest, json_digest, tensor_digest
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


_MINIMUM_GRID_AXIS_SINE = 0.01
_MAXIMUM_GRID_AFFINE_CONDITION_NUMBER = 1000.0


@dataclass(frozen=True)
class NowcastConfig:
    interval_minutes: int = 10
    horizon_minutes: int = 180
    min_dbz: float = -10.0
    max_dbz: float = 70.0
    echo_threshold_dbz: float = 5.0
    recent_weight: float = 2.0 / 3.0
    pair_echo_dilation_px: int = 3
    pair_echo_dilation_m: float | None = None
    max_displacement_px: float = 20.0
    maximum_motion_speed_mps: float | None = None
    maximum_pair_motion_disagreement_px: float = 4.0
    maximum_pair_velocity_disagreement_mps: float = 10.0
    maximum_pair_growth_disagreement: float = math.log(1.10)
    minimum_pair_psr_advantage: float = 3.0
    long_pair_confidence_penalty: float = 0.5
    minimum_phase_correlation_psr: float = 8.0
    phase_correlation_sidelobe_radius_px: int = 2
    phase_correlation_sidelobe_radius_m: float | None = None
    max_log_growth_per_step: float = math.log(1.35)
    growth_decay_minutes: float = 60.0
    maximum_background_age_minutes: float = 60.0
    min_publish_support: float = 0.95
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if type(self.interval_minutes) is not int:
            raise TypeError("interval_minutes must be an integer")
        if type(self.horizon_minutes) is not int:
            raise TypeError("horizon_minutes must be an integer")
        if type(self.pair_echo_dilation_px) is not int:
            raise TypeError("pair_echo_dilation_px must be an integer")
        if type(self.phase_correlation_sidelobe_radius_px) is not int:
            raise TypeError(
                "phase_correlation_sidelobe_radius_px must be an integer"
            )
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
            self.maximum_pair_motion_disagreement_px,
            self.maximum_pair_velocity_disagreement_mps,
            self.maximum_pair_growth_disagreement,
            self.minimum_pair_psr_advantage,
            self.long_pair_confidence_penalty,
            self.minimum_phase_correlation_psr,
            self.max_log_growth_per_step,
            self.growth_decay_minutes,
            self.maximum_background_age_minutes,
            self.min_publish_support,
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
        if self.pair_echo_dilation_px < 0:
            raise ValueError("pair_echo_dilation_px cannot be negative")
        physical_radii = {
            "pair_echo_dilation_m": self.pair_echo_dilation_m,
            "phase_correlation_sidelobe_radius_m": (
                self.phase_correlation_sidelobe_radius_m
            ),
        }
        for name, value in physical_radii.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_displacement_px <= 0:
            raise ValueError("max_displacement_px must be positive")
        pair_limits = {
            "maximum_pair_motion_disagreement_px": (
                self.maximum_pair_motion_disagreement_px
            ),
            "maximum_pair_velocity_disagreement_mps": (
                self.maximum_pair_velocity_disagreement_mps
            ),
            "maximum_pair_growth_disagreement": (
                self.maximum_pair_growth_disagreement
            ),
        }
        for name, value in pair_limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_pair_psr_advantage <= 0:
            raise ValueError("minimum_pair_psr_advantage must be positive")
        if not 0.0 < self.long_pair_confidence_penalty <= 1.0:
            raise ValueError(
                "long_pair_confidence_penalty must be in (0, 1]"
            )
        if self.maximum_motion_speed_mps is not None and (
            isinstance(self.maximum_motion_speed_mps, bool)
            or not isinstance(self.maximum_motion_speed_mps, (int, float))
            or not math.isfinite(self.maximum_motion_speed_mps)
            or self.maximum_motion_speed_mps <= 0
        ):
            raise ValueError("maximum_motion_speed_mps must be positive")
        if self.minimum_phase_correlation_psr < 0:
            raise ValueError(
                "minimum_phase_correlation_psr cannot be negative"
            )
        if self.phase_correlation_sidelobe_radius_px < 0:
            raise ValueError(
                "phase_correlation_sidelobe_radius_px cannot be negative"
            )
        if self.max_log_growth_per_step < 0:
            raise ValueError("max_log_growth_per_step cannot be negative")
        if self.growth_decay_minutes <= 0:
            raise ValueError("growth_decay_minutes must be positive")
        if self.maximum_background_age_minutes <= 0:
            raise ValueError("maximum_background_age_minutes must be positive")
        if not 0.0 < self.min_publish_support <= 1.0:
            raise ValueError("min_publish_support must be in (0, 1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def forecast_steps(self) -> int:
        return self.horizon_minutes // self.interval_minutes

    @property
    def digest(self) -> str:
        return dataclass_digest(self)

def _parse_aware_time(value: str, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must contain ISO-8601 strings")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{name} must contain ISO-8601 strings") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include timezone offsets")
    return parsed.astimezone(timezone.utc)


def _canonical_time_tuple(
    name: str,
    values: tuple[str, str, str],
) -> tuple[str, str, str]:
    if not isinstance(values, tuple) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three times")
    canonical = tuple(
        _parse_aware_time(value, name).isoformat().replace("+00:00", "Z")
        for value in values
    )
    return canonical[0], canonical[1], canonical[2]


@dataclass(frozen=True)
class RadarGridTimeContract:
    valid_times: tuple[str, str, str]
    dx_m: float
    dy_m: float
    projection: str
    grid_hash: str
    background_valid_times: tuple[str, str, str] | None = None
    pixel_to_projected_matrix_m: (
        tuple[tuple[float, float], tuple[float, float]] | None
    ) = None

    def __post_init__(self) -> None:
        valid_times = _canonical_time_tuple("valid_times", self.valid_times)
        object.__setattr__(self, "valid_times", valid_times)
        if self.background_valid_times is not None:
            background_times = _canonical_time_tuple(
                "background_valid_times",
                self.background_valid_times,
            )
            object.__setattr__(
                self,
                "background_valid_times",
                background_times,
            )
        if (
            not isinstance(self.dx_m, (int, float))
            or isinstance(self.dx_m, bool)
            or not math.isfinite(self.dx_m)
            or self.dx_m <= 0
        ):
            raise ValueError("dx_m must be finite and positive")
        if (
            not isinstance(self.dy_m, (int, float))
            or isinstance(self.dy_m, bool)
            or not math.isfinite(self.dy_m)
            or self.dy_m <= 0
        ):
            raise ValueError("dy_m must be finite and positive")
        matrix = self.pixel_to_projected_matrix_m
        if matrix is None:
            matrix = ((float(self.dx_m), 0.0), (0.0, -float(self.dy_m)))
        if (
            not isinstance(matrix, tuple)
            or len(matrix) != 2
            or any(
                not isinstance(row, tuple) or len(row) != 2
                for row in matrix
            )
        ):
            raise ValueError(
                "pixel_to_projected_matrix_m must be a 2x2 tuple"
            )
        canonical_matrix = (
            (float(matrix[0][0]), float(matrix[0][1])),
            (float(matrix[1][0]), float(matrix[1][1])),
        )
        if not all(
            math.isfinite(value)
            for row in canonical_matrix
            for value in row
        ):
            raise ValueError("pixel_to_projected_matrix_m must be finite")
        determinant = (
            canonical_matrix[0][0] * canonical_matrix[1][1]
            - canonical_matrix[0][1] * canonical_matrix[1][0]
        )
        scale = max(abs(value) for row in canonical_matrix for value in row)
        if abs(determinant) <= math.ulp(max(scale * scale, 1.0)):
            raise ValueError("pixel_to_projected_matrix_m must be invertible")
        column_spacing = math.hypot(
            canonical_matrix[0][0], canonical_matrix[1][0]
        )
        row_spacing = math.hypot(
            canonical_matrix[0][1], canonical_matrix[1][1]
        )
        if not math.isclose(
            column_spacing,
            float(self.dx_m),
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ) or not math.isclose(
            row_spacing,
            float(self.dy_m),
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "pixel_to_projected_matrix_m must agree with dx_m and dy_m"
            )
        normalized_determinant = abs(determinant) / (
            column_spacing * row_spacing
        )
        frobenius_squared = sum(
            value * value for row in canonical_matrix for value in row
        )
        discriminant = max(
            frobenius_squared * frobenius_squared
            - 4.0 * determinant * determinant,
            0.0,
        )
        maximum_singular_value_squared = 0.5 * (
            frobenius_squared + math.sqrt(discriminant)
        )
        condition_number = maximum_singular_value_squared / abs(determinant)
        if (
            normalized_determinant < _MINIMUM_GRID_AXIS_SINE
            or condition_number > _MAXIMUM_GRID_AFFINE_CONDITION_NUMBER
        ):
            raise ValueError(
                "pixel_to_projected_matrix_m must be well-conditioned"
            )
        object.__setattr__(
            self,
            "pixel_to_projected_matrix_m",
            canonical_matrix,
        )
        if (
            not isinstance(self.projection, str)
            or not self.projection
            or self.projection.strip() != self.projection
        ):
            raise ValueError("projection must be a non-empty canonical string")
        _validate_sha256_digest("grid_hash", self.grid_hash)

    @property
    def digest(self) -> str:
        return dataclass_digest(self)

    @property
    def cell_area_m2(self) -> float:
        assert self.pixel_to_projected_matrix_m is not None
        (xx, xr), (yx, yr) = self.pixel_to_projected_matrix_m
        return abs(xx * yr - xr * yx)

    @property
    def grid_axes_are_orthogonal(self) -> bool:
        assert self.pixel_to_projected_matrix_m is not None
        (xx, xr), (yx, yr) = self.pixel_to_projected_matrix_m
        dot_product = xx * xr + yx * yr
        scale = float(self.dx_m) * float(self.dy_m)
        return math.isclose(dot_product, 0.0, abs_tol=1.0e-9 * scale)

    def projected_displacement_xy(
        self,
        displacement_yx: Tensor,
    ) -> Tensor:
        if displacement_yx.shape != (2,):
            raise ValueError("displacement_yx must have shape [2]")
        assert self.pixel_to_projected_matrix_m is not None
        matrix = displacement_yx.new_tensor(self.pixel_to_projected_matrix_m)
        column_row = torch.stack((displacement_yx[1], displacement_yx[0]))
        return matrix @ column_row

    def displacement_yx_from_projected_xy(
        self,
        projected_displacement_xy: Tensor,
    ) -> Tensor:
        if projected_displacement_xy.shape != (2,):
            raise ValueError("projected_displacement_xy must have shape [2]")
        assert self.pixel_to_projected_matrix_m is not None
        matrix = projected_displacement_xy.new_tensor(
            self.pixel_to_projected_matrix_m
        )
        column_row = torch.linalg.solve(matrix, projected_displacement_xy)
        return torch.stack((column_row[1], column_row[0]))

    def projected_velocity_xy(
        self,
        displacement_yx: Tensor,
        interval_minutes: int,
    ) -> Tensor:
        if type(interval_minutes) is not int or interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")
        return self.projected_displacement_xy(displacement_yx) / (
            interval_minutes * 60.0
        )

    def displacement_yx_from_projected_velocity(
        self,
        projected_velocity_xy: Tensor,
        interval_minutes: int,
    ) -> Tensor:
        if type(interval_minutes) is not int or interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")
        return self.displacement_yx_from_projected_xy(
            projected_velocity_xy * (interval_minutes * 60.0)
        )

    def maximum_displacement_yx(
        self,
        maximum_speed_mps: float,
        interval_minutes: int,
    ) -> tuple[float, float]:
        if not math.isfinite(maximum_speed_mps) or maximum_speed_mps <= 0:
            raise ValueError("maximum_speed_mps must be positive")
        if type(interval_minutes) is not int or interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")
        radius_m = maximum_speed_mps * interval_minutes * 60.0
        return self._maximum_index_displacement_yx(radius_m)

    def _maximum_index_displacement_yx(
        self,
        radius_m: float,
    ) -> tuple[float, float]:
        assert self.pixel_to_projected_matrix_m is not None
        (a, b), (c, d) = self.pixel_to_projected_matrix_m
        determinant = a * d - b * c
        inverse = (
            (d / determinant, -b / determinant),
            (-c / determinant, a / determinant),
        )
        maximum_col = radius_m * math.hypot(*inverse[0])
        maximum_row = radius_m * math.hypot(*inverse[1])
        return maximum_row, maximum_col

    def pixel_radius_yx(self, distance_m: float) -> tuple[int, int]:
        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("distance_m must be finite and nonnegative")
        maximum_row, maximum_col = self._maximum_index_displacement_yx(
            distance_m
        )
        tolerance = 64.0 * math.ulp(
            max(maximum_row, maximum_col, 1.0)
        )
        return (
            math.floor(maximum_row + tolerance),
            math.floor(maximum_col + tolerance),
        )

    def pixel_offsets_within_distance(
        self,
        distance_m: float,
        *,
        maximum_radius_yx: tuple[int, int],
    ) -> tuple[tuple[int, int], ...]:
        """Return integer row/column offsets inside a projected-distance ball."""

        if (
            not isinstance(maximum_radius_yx, tuple)
            or len(maximum_radius_yx) != 2
            or any(type(value) is not int or value < 0 for value in maximum_radius_yx)
        ):
            raise ValueError(
                "maximum_radius_yx must contain two nonnegative integers"
            )
        radius_y, radius_x = self.pixel_radius_yx(distance_m)
        if radius_y > maximum_radius_yx[0] or radius_x > maximum_radius_yx[1]:
            raise ValueError(
                "physical distance requires a grid radius larger than the "
                "analysis grid"
            )
        assert self.pixel_to_projected_matrix_m is not None
        (a, b), (c, d) = self.pixel_to_projected_matrix_m
        distance_tolerance = 64.0 * math.ulp(
            max(distance_m, float(self.dx_m), float(self.dy_m), 1.0)
        )
        maximum_distance = distance_m + distance_tolerance
        offsets = tuple(
            (row, column)
            for row in range(-radius_y, radius_y + 1)
            for column in range(-radius_x, radius_x + 1)
            if math.hypot(
                a * column + b * row,
                c * column + d * row,
            )
            <= maximum_distance
        )
        if (0, 0) not in offsets:
            raise RuntimeError("physical offset footprint must contain its origin")
        return offsets

    def validate_for(
        self,
        config: NowcastConfig,
        *,
        background_present: bool,
        background_age_minutes: float | None,
    ) -> None:
        if config.maximum_motion_speed_mps is not None:
            self.maximum_displacement_yx(
                config.maximum_motion_speed_mps,
                config.interval_minutes,
            )
        observation_times = tuple(
            _parse_aware_time(value, "valid_times")
            for value in self.valid_times
        )
        intervals = tuple(
            (later - earlier).total_seconds() / 60.0
            for earlier, later in zip(
                observation_times,
                observation_times[1:],
            )
        )
        if any(
            not math.isclose(
                interval,
                float(config.interval_minutes),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for interval in intervals
        ):
            raise ValueError(
                "valid_times must use NowcastConfig.interval_minutes"
            )
        if background_present != (self.background_valid_times is not None):
            raise ValueError(
                "background_valid_times must match background availability"
            )
        if self.background_valid_times is None:
            return
        background_times = tuple(
            _parse_aware_time(value, "background_valid_times")
            for value in self.background_valid_times
        )
        background_intervals = tuple(
            (later - earlier).total_seconds() / 60.0
            for earlier, later in zip(
                background_times,
                background_times[1:],
            )
        )
        if any(
            not math.isclose(
                interval,
                float(config.interval_minutes),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for interval in background_intervals
        ):
            raise ValueError(
                "background_valid_times must use NowcastConfig.interval_minutes"
            )
        ages = tuple(
            (observation - background).total_seconds() / 60.0
            for observation, background in zip(
                observation_times,
                background_times,
            )
        )
        if any(age < 0 for age in ages):
            raise ValueError("background valid times cannot be in the future")
        if any(age > config.maximum_background_age_minutes for age in ages):
            raise ValueError("background exceeds maximum_background_age_minutes")
        if background_age_minutes is None or not math.isclose(
            ages[-1],
            background_age_minutes,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "background_age_minutes must match the latest valid times"
            )


def motion_displacement_limits_yx(
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
    reference: Tensor,
) -> Tensor:
    if config.maximum_motion_speed_mps is None:
        return reference.new_full((2,), config.max_displacement_px)
    if grid_time_contract is None:
        raise ValueError(
            "maximum_motion_speed_mps requires a grid/time contract"
        )
    limits = grid_time_contract.maximum_displacement_yx(
        config.maximum_motion_speed_mps,
        config.interval_minutes,
    )
    return reference.new_tensor(limits)


class DataStatus(str, Enum):
    OBSERVED = "OBSERVED"
    PARTIAL = "PARTIAL"
    STALE_BACKGROUND = "STALE_BACKGROUND"
    UNAVAILABLE = "UNAVAILABLE"


class TendencySource(str, Enum):
    OBSERVATION = "OBSERVATION"
    BACKGROUND = "BACKGROUND"
    NONE = "NONE"


class TendencyPairSelection(str, Enum):
    NONE = "NONE"
    SINGLE = "SINGLE"
    LONG = "LONG"
    BLENDED = "BLENDED"
    EARLIER = "EARLIER"
    RECENT = "RECENT"
    PERSISTENCE = "PERSISTENCE"


@dataclass(frozen=True)
class _SourceTendencyEstimate:
    displacement_yx: Tensor
    log_growth_per_step: Tensor
    motion_disagreement_px: Tensor
    motion_disagreement_mps: Tensor
    growth_disagreement: Tensor
    minimum_phase_correlation_psr: Tensor
    tendency_pair_count: int
    motion_pair_count: int
    growth_pair_count: int
    motion_pair_selection: TendencyPairSelection
    growth_pair_selection: TendencyPairSelection
    motion_pair_conflict: bool
    growth_pair_conflict: bool

    @property
    def available(self) -> bool:
        return self.tendency_pair_count > 0


_FORECAST_INPUT_BUNDLE_VERSION = "forecast-input-bundle-v2"
_FORECAST_RUN_IDENTITY_VERSION = "forecast-run-identity-v4"


def _validate_background_age(
    config: NowcastConfig,
    *,
    background_present: bool,
    background_age_minutes: float | None,
) -> None:
    if not background_present:
        if background_age_minutes is not None:
            raise ValueError(
                "background_age_minutes requires background_frames_dbz"
            )
        return
    if background_age_minutes is None:
        raise ValueError(
            "background_age_minutes is required with background_frames_dbz"
        )
    if not math.isfinite(background_age_minutes) or background_age_minutes < 0:
        raise ValueError(
            "background_age_minutes must be finite and non-negative"
        )
    if background_age_minutes > config.maximum_background_age_minutes:
        raise ValueError("background exceeds maximum_background_age_minutes")


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
    background_contribution_fraction: float
    background_age_minutes: float | None
    source_support: Tensor
    motion_disagreement_px: Tensor
    motion_disagreement_mps: Tensor
    growth_disagreement: Tensor
    minimum_phase_correlation_psr: Tensor
    tendency_pair_count: int
    tendency_source: TendencySource
    provenance: str = "p0_support_merged"
    motion_pair_count: int = 0
    growth_pair_count: int = 0
    motion_pair_selection: TendencyPairSelection = TendencyPairSelection.NONE
    growth_pair_selection: TendencyPairSelection = TendencyPairSelection.NONE
    motion_pair_conflict: bool = False
    growth_pair_conflict: bool = False

    def __post_init__(self) -> None:
        if (
            self.tendency_pair_count in (1, 2)
            and self.motion_pair_count == 0
            and self.growth_pair_count == 0
            and self.motion_pair_selection is TendencyPairSelection.NONE
            and self.growth_pair_selection is TendencyPairSelection.NONE
        ):
            selection = (
                TendencyPairSelection.SINGLE
                if self.tendency_pair_count == 1
                else TendencyPairSelection.BLENDED
            )
            object.__setattr__(
                self,
                "motion_pair_count",
                self.tendency_pair_count,
            )
            object.__setattr__(
                self,
                "growth_pair_count",
                self.tendency_pair_count,
            )
            object.__setattr__(self, "motion_pair_selection", selection)
            object.__setattr__(self, "growth_pair_selection", selection)

    @property
    def background_state_support_fraction(self) -> float:
        return self.background_contribution_fraction

    @property
    def background_tendency_used(self) -> bool:
        return self.tendency_source is TendencySource.BACKGROUND


def state_metadata_digest(
    state: RadarState,
    metadata: ForecastMetadata,
) -> str:
    return json_digest(
        {
            "state": {
                "echo_linear": tensor_digest(state.echo_linear),
                "displacement_yx": tensor_digest(state.displacement_yx),
                "log_growth_per_step": tensor_digest(
                    state.log_growth_per_step
                ),
            },
            "metadata": {
                "data_status": metadata.data_status.value,
                "coverage_by_frame": tensor_digest(
                    metadata.coverage_by_frame
                ),
                "background_used": metadata.background_used,
                "background_contribution_fraction": (
                    metadata.background_contribution_fraction
                ),
                "background_state_support_fraction": (
                    metadata.background_state_support_fraction
                ),
                "background_tendency_used": metadata.background_tendency_used,
                "background_age_minutes": metadata.background_age_minutes,
                "source_support": tensor_digest(metadata.source_support),
                "motion_disagreement_px": tensor_digest(
                    metadata.motion_disagreement_px
                ),
                "motion_disagreement_mps": tensor_digest(
                    metadata.motion_disagreement_mps
                ),
                "growth_disagreement": tensor_digest(
                    metadata.growth_disagreement
                ),
                "minimum_phase_correlation_psr": tensor_digest(
                    metadata.minimum_phase_correlation_psr
                ),
                "tendency_pair_count": metadata.tendency_pair_count,
                "motion_pair_count": metadata.motion_pair_count,
                "growth_pair_count": metadata.growth_pair_count,
                "motion_pair_selection": metadata.motion_pair_selection.value,
                "growth_pair_selection": metadata.growth_pair_selection.value,
                "motion_pair_conflict": metadata.motion_pair_conflict,
                "growth_pair_conflict": metadata.growth_pair_conflict,
                "tendency_source": metadata.tendency_source.value,
                "provenance": metadata.provenance,
            },
        }
    )


@dataclass(frozen=True)
class ForecastAudit:
    input_echo: PositivityAudit
    forecast_final: PositivityAudit
    transport: tuple[TransportAudit, ...]


@dataclass(frozen=True)
class ForecastRunContract:
    config: NowcastConfig
    _latest_frame_dbz: Tensor
    _latest_observation_mask: Tensor
    _latest_background_dbz: Tensor | None
    latest_observation_mask_digest: str
    latest_frame_digest: str
    latest_background_digest: str | None
    input_bundle_digest: str
    background_age_minutes: float | None = None
    grid_time_contract: RadarGridTimeContract | None = None
    grid_time_contract_digest: str | None = None
    analysis_config_json: str | None = None
    analysis_config_digest: str | None = None
    analysis_input_digest: str | None = None
    forecast_integrator_version: str = FORECAST_INTEGRATOR_VERSION

    @classmethod
    def from_inputs(
        cls,
        config: NowcastConfig,
        frames_dbz: Tensor,
        observation_masks: Tensor,
        background_frames_dbz: Tensor | None,
        background_age_minutes: float | None = None,
        *,
        grid_time_contract: RadarGridTimeContract | None = None,
        analysis_config_json: str | None = None,
        analysis_config_digest: str | None = None,
        analysis_input_digest: str | None = None,
    ) -> ForecastRunContract:
        _validate_frames(frames_dbz)
        latest_frame = frames_dbz[-1]
        if (
            observation_masks.shape != frames_dbz.shape
            or observation_masks.dtype != torch.bool
        ):
            raise ValueError(
                "observation_masks must be boolean with the frame shape"
            )
        background_present = background_frames_dbz is not None
        _validate_background_age(
            config,
            background_present=background_present,
            background_age_minutes=background_age_minutes,
        )
        if background_frames_dbz is not None:
            if (
                background_frames_dbz.shape != frames_dbz.shape
                or not background_frames_dbz.is_floating_point()
            ):
                raise ValueError(
                    "background_frames_dbz must be floating with the frame shape"
                )
        if grid_time_contract is not None:
            grid_time_contract.validate_for(
                config,
                background_present=background_present,
                background_age_minutes=background_age_minutes,
            )
        _validate_analysis_lineage(
            analysis_config_json,
            analysis_config_digest,
            analysis_input_digest,
        )
        latest_background = (
            None
            if background_frames_dbz is None
            else tensor_digest(background_frames_dbz[-1])
        )
        accepted_mask = observation_masks[-1].detach().clone()
        accepted_frame = latest_frame.detach().clone()
        accepted_background = (
            None
            if background_frames_dbz is None
            else background_frames_dbz[-1].detach().clone()
        )
        return cls(
            config=config,
            _latest_frame_dbz=accepted_frame,
            _latest_observation_mask=accepted_mask,
            _latest_background_dbz=accepted_background,
            latest_observation_mask_digest=tensor_digest(accepted_mask),
            latest_frame_digest=tensor_digest(latest_frame),
            latest_background_digest=latest_background,
            input_bundle_digest=_forecast_input_bundle_digest(
                frames_dbz,
                observation_masks,
                background_frames_dbz,
                background_age_minutes,
                grid_time_contract,
                analysis_config_digest,
                analysis_input_digest,
            ),
            background_age_minutes=background_age_minutes,
            grid_time_contract=grid_time_contract,
            grid_time_contract_digest=(
                None
                if grid_time_contract is None
                else grid_time_contract.digest
            ),
            analysis_config_json=analysis_config_json,
            analysis_config_digest=analysis_config_digest,
            analysis_input_digest=analysis_input_digest,
        )

    @property
    def latest_frame_dbz(self) -> Tensor:
        self.validate_integrity()
        return self._latest_frame_dbz.clone()

    @property
    def latest_observation_mask(self) -> Tensor:
        self.validate_integrity()
        return self._latest_observation_mask.clone()

    @property
    def latest_background_dbz(self) -> Tensor | None:
        self.validate_integrity()
        if self._latest_background_dbz is None:
            return None
        return self._latest_background_dbz.clone()

    def validate_integrity(self) -> None:
        if self.forecast_integrator_version != FORECAST_INTEGRATOR_VERSION:
            raise ValueError(
                "forecast integrator version is incompatible with this runtime"
            )
        _validate_sha256_digest(
            "input_bundle_digest",
            self.input_bundle_digest,
        )
        _validate_sha256_digest(
            "latest_frame_digest",
            self.latest_frame_digest,
        )
        _validate_sha256_digest(
            "latest_observation_mask_digest",
            self.latest_observation_mask_digest,
        )
        if self.latest_background_digest is not None:
            _validate_sha256_digest(
                "latest_background_digest",
                self.latest_background_digest,
            )
        _validate_background_age(
            self.config,
            background_present=self.latest_background_digest is not None,
            background_age_minutes=self.background_age_minutes,
        )
        if (
            self._latest_frame_dbz.ndim != 2
            or not self._latest_frame_dbz.is_floating_point()
        ):
            raise ValueError("latest frame must be a floating 2-D tensor")
        if (
            self._latest_observation_mask.dtype != torch.bool
            or self._latest_observation_mask.shape
            != self._latest_frame_dbz.shape
        ):
            raise ValueError(
                "latest observation mask must match the latest frame"
            )
        if self._latest_background_dbz is not None and (
            not self._latest_background_dbz.is_floating_point()
            or self._latest_background_dbz.shape
            != self._latest_frame_dbz.shape
        ):
            raise ValueError("latest background must match the latest frame")
        if self.grid_time_contract is None:
            if self.grid_time_contract_digest is not None:
                raise ValueError(
                    "grid_time_contract_digest requires grid_time_contract"
                )
        else:
            if self.grid_time_contract_digest is None:
                raise ValueError(
                    "grid_time_contract requires grid_time_contract_digest"
                )
            _validate_sha256_digest(
                "grid_time_contract_digest",
                self.grid_time_contract_digest,
            )
            if self.grid_time_contract.digest != self.grid_time_contract_digest:
                raise ValueError("grid/time contract digest mismatch")
            self.grid_time_contract.validate_for(
                self.config,
                background_present=self.latest_background_digest is not None,
                background_age_minutes=self.background_age_minutes,
            )
        motion_displacement_limits_yx(
            self.config,
            self.grid_time_contract,
            self._latest_frame_dbz,
        )
        if (
            tensor_digest(self._latest_frame_dbz)
            != self.latest_frame_digest
        ):
            raise ValueError("latest frame disagrees with the forecast run")
        if (
            tensor_digest(self._latest_observation_mask)
            != self.latest_observation_mask_digest
        ):
            raise ValueError(
                "latest observation mask disagrees with the forecast run"
            )
        if self._latest_background_dbz is None:
            if self.latest_background_digest is not None:
                raise ValueError("latest background is missing from the run")
        elif (
            self.latest_background_digest is None
            or tensor_digest(self._latest_background_dbz)
            != self.latest_background_digest
        ):
            raise ValueError("latest background disagrees with the forecast run")
        _validate_analysis_lineage(
            self.analysis_config_json,
            self.analysis_config_digest,
            self.analysis_input_digest,
        )

    def validate_latest_frame(self, latest_frame_dbz: Tensor) -> None:
        if tensor_digest(latest_frame_dbz) != self.latest_frame_digest:
            raise ValueError("latest frame disagrees with the forecast run")

    def validate_latest_background(
        self,
        latest_background_dbz: Tensor | None,
    ) -> None:
        if latest_background_dbz is None:
            if self.latest_background_digest is None:
                return
            raise ValueError("latest background is required by the forecast run")
        if self.latest_background_digest is None:
            raise ValueError("forecast run did not use a background")
        if (
            tensor_digest(latest_background_dbz)
            != self.latest_background_digest
        ):
            raise ValueError("latest background disagrees with the forecast run")


def _forecast_input_bundle_digest(
    frames_dbz: Tensor,
    observation_masks: Tensor,
    background_frames_dbz: Tensor | None,
    background_age_minutes: float | None,
    grid_time_contract: RadarGridTimeContract | None,
    analysis_config_digest: str | None,
    analysis_input_digest: str | None,
) -> str:
    return json_digest(
        {
            "version": _FORECAST_INPUT_BUNDLE_VERSION,
            "frames_dbz": tensor_digest(frames_dbz),
            "observation_masks": tensor_digest(observation_masks),
            "background_frames_dbz": (
                None
                if background_frames_dbz is None
                else tensor_digest(background_frames_dbz)
            ),
            "background_age_minutes": background_age_minutes,
            "grid_time_contract_digest": (
                None
                if grid_time_contract is None
                else grid_time_contract.digest
            ),
            "analysis_config_digest": analysis_config_digest,
            "analysis_input_digest": analysis_input_digest,
        }
    )


def _validate_sha256_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_analysis_lineage(
    config_json: str | None,
    config_digest: str | None,
    input_digest: str | None,
) -> None:
    values = (config_json, config_digest, input_digest)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(
            "analysis config and input lineage must be provided together"
        )
    assert config_json is not None
    assert config_digest is not None
    assert input_digest is not None
    try:
        config_value = json.loads(config_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid analysis_config_json") from error
    _validate_sha256_digest("analysis_config_digest", config_digest)
    if json_digest(config_value) != config_digest:
        raise ValueError("analysis config digest mismatch")
    _validate_sha256_digest("analysis_input_digest", input_digest)


def _forecast_run_identity_digest(
    run: ForecastRunContract,
    state_digest: str,
    forecast_digest: str,
    valid_mask_digest: str,
) -> str:
    return json_digest(
        {
            "version": _FORECAST_RUN_IDENTITY_VERSION,
            "forecast_integrator_version": run.forecast_integrator_version,
            "nowcast_config_digest": run.config.digest,
            "input_bundle_digest": run.input_bundle_digest,
            "latest_frame_digest": run.latest_frame_digest,
            "latest_observation_mask_digest": (
                run.latest_observation_mask_digest
            ),
            "latest_background_digest": run.latest_background_digest,
            "background_age_minutes": run.background_age_minutes,
            "grid_time_contract_digest": run.grid_time_contract_digest,
            "analysis_config_digest": run.analysis_config_digest,
            "analysis_input_digest": run.analysis_input_digest,
            "state_metadata_digest": state_digest,
            "forecast_dbz_digest": forecast_digest,
            "valid_mask_digest": valid_mask_digest,
        }
    )


@dataclass(frozen=True)
class ForecastResult:
    forecast_dbz: Tensor
    valid_mask: Tensor
    forecast_dbz_digest: str
    valid_mask_digest: str
    state: RadarState
    metadata: ForecastMetadata
    run: ForecastRunContract
    state_metadata_digest: str
    forecast_run_digest: str
    audit: ForecastAudit | None = None

    @property
    def grid_velocity_mps_yx(self) -> Tensor | None:
        contract = self.run.grid_time_contract
        if contract is None:
            return None
        metres_per_pixel = self.state.displacement_yx.new_tensor(
            (contract.dy_m, contract.dx_m)
        )
        seconds_per_step = 60.0 * self.run.config.interval_minutes
        return self.state.displacement_yx * metres_per_pixel / seconds_per_step

    @property
    def displacement_mps_yx(self) -> Tensor | None:
        return self.grid_velocity_mps_yx

    @property
    def projected_velocity_mps_xy(self) -> Tensor | None:
        contract = self.run.grid_time_contract
        if contract is None:
            return None
        projected = contract.projected_displacement_xy(
            self.state.displacement_yx
        )
        seconds_per_step = 60.0 * self.run.config.interval_minutes
        return projected / seconds_per_step

    def validate_issuance(self) -> None:
        if tensor_digest(self.forecast_dbz) != self.forecast_dbz_digest:
            raise ValueError("forecast result disagrees with the issued forecast")
        if tensor_digest(self.valid_mask) != self.valid_mask_digest:
            raise ValueError(
                "forecast valid mask disagrees with the issued forecast"
            )
        if (
            state_metadata_digest(self.state, self.metadata)
            != self.state_metadata_digest
        ):
            raise ValueError(
                "forecast state or metadata disagrees with the issued forecast"
            )
        self.run.validate_integrity()
        if (
            _forecast_run_identity_digest(
                self.run,
                self.state_metadata_digest,
                self.forecast_dbz_digest,
                self.valid_mask_digest,
            )
            != self.forecast_run_digest
        ):
            raise ValueError("forecast run identity disagrees with the issued forecast")


@dataclass(frozen=True)
class PreparedRadarInput:
    frames_dbz: Tensor
    background_frames_dbz: Tensor
    observed_mask: Tensor
    background_mask: Tensor
    missing_mask: Tensor
    qc_rejected_mask: Tensor
    data_status: DataStatus
    coverage_by_frame: Tensor
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
    floor = clean_observations.new_full((), config.min_dbz)
    observation_frames = torch.where(
        observed,
        clean_observations,
        floor,
    )
    background_frames = torch.full_like(frames_dbz, config.min_dbz)
    background_mask = torch.zeros_like(observed)
    _validate_background_age(
        config,
        background_present=background_frames_dbz is not None,
        background_age_minutes=background_age_minutes,
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
        background_mask = torch.isfinite(background)
        clean_background = torch.nan_to_num(
            background,
            nan=config.min_dbz,
            posinf=config.max_dbz,
            neginf=config.min_dbz,
        ).clamp(config.min_dbz, config.max_dbz)
        background_frames = torch.where(
            background_mask,
            clean_background,
            floor,
        )
    observed_count = int(observed.sum())
    observation_count = observed.numel()
    coverage_by_frame = observed.to(torch.float64).mean(dim=(1, 2))
    if observed_count == 0:
        status = (
            DataStatus.STALE_BACKGROUND
            if bool(torch.any(background_mask[-1]))
            else DataStatus.UNAVAILABLE
        )
    elif observed_count < observation_count:
        status = DataStatus.PARTIAL
    else:
        status = DataStatus.OBSERVED

    return PreparedRadarInput(
        frames_dbz=observation_frames,
        background_frames_dbz=background_frames,
        observed_mask=observed,
        background_mask=background_mask,
        missing_mask=missing,
        qc_rejected_mask=qc_rejected,
        data_status=status,
        coverage_by_frame=coverage_by_frame.detach(),
        background_age_minutes=background_age_minutes,
    )


def estimate_state(
    frames_dbz: Tensor,
    config: NowcastConfig,
    *,
    qc_mask: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> tuple[RadarState, ForecastMetadata]:
    prepared = prepare_input(
        frames_dbz,
        config,
        accepted_mask=qc_mask,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
    )
    return estimate_prepared_state(
        prepared,
        config,
        grid_time_contract=grid_time_contract,
    )


def estimate_prepared_state(
    prepared: PreparedRadarInput,
    config: NowcastConfig,
    *,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> tuple[RadarState, ForecastMetadata]:
    if grid_time_contract is None and (
        config.pair_echo_dilation_m is not None
        or config.phase_correlation_sidelobe_radius_m is not None
    ):
        raise ValueError("physical pair settings require a grid/time contract")
    observation_linear = dbz_to_echo(
        prepared.frames_dbz,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    background_linear = dbz_to_echo(
        prepared.background_frames_dbz,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    observation_linear, _ = validate_physical_echo(
        observation_linear,
        name="observation echo conversion",
    )
    background_linear, _ = validate_physical_echo(
        background_linear,
        name="background echo conversion",
    )
    tendency, tendency_source = _estimate_time_normalized_tendencies(
        prepared,
        observation_linear,
        background_linear,
        config,
        grid_time_contract,
    )
    displacement = tendency.displacement_yx
    growth = tendency.log_growth_per_step
    (
        current_echo,
        current_source_support,
        background_contribution_fraction,
    ) = _merge_current_state(
        prepared,
        observation_linear,
        background_linear,
        displacement,
        growth,
        config,
    )
    state = RadarState(
        echo_linear=current_echo,
        displacement_yx=displacement,
        log_growth_per_step=growth,
    )
    background_tendency_used = tendency_source is TendencySource.BACKGROUND
    background_used = (
        background_contribution_fraction > config.epsilon
        or background_tendency_used
    )
    metadata = ForecastMetadata(
        data_status=prepared.data_status,
        coverage_by_frame=prepared.coverage_by_frame,
        background_used=background_used,
        background_contribution_fraction=background_contribution_fraction,
        background_age_minutes=(
            prepared.background_age_minutes if background_used else None
        ),
        source_support=current_source_support.detach().clone(),
        motion_disagreement_px=tendency.motion_disagreement_px.detach(),
        motion_disagreement_mps=tendency.motion_disagreement_mps.detach(),
        growth_disagreement=tendency.growth_disagreement.detach(),
        minimum_phase_correlation_psr=(
            tendency.minimum_phase_correlation_psr.detach()
        ),
        tendency_pair_count=tendency.tendency_pair_count,
        tendency_source=tendency_source,
        motion_pair_count=tendency.motion_pair_count,
        growth_pair_count=tendency.growth_pair_count,
        motion_pair_selection=tendency.motion_pair_selection,
        growth_pair_selection=tendency.growth_pair_selection,
        motion_pair_conflict=tendency.motion_pair_conflict,
        growth_pair_conflict=tendency.growth_pair_conflict,
    )
    return state, metadata


def _estimate_time_normalized_tendencies(
    prepared: PreparedRadarInput,
    observation_linear: Tensor,
    background_linear: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[_SourceTendencyEstimate, TendencySource]:
    observation_estimate = _estimate_source_tendencies(
        prepared.frames_dbz,
        prepared.observed_mask,
        observation_linear,
        config,
        grid_time_contract,
    )
    if observation_estimate.available:
        return observation_estimate, TendencySource.OBSERVATION
    background_estimate = _estimate_source_tendencies(
        prepared.background_frames_dbz,
        prepared.background_mask,
        background_linear,
        config,
        grid_time_contract,
    )
    if background_estimate.available:
        return background_estimate, TendencySource.BACKGROUND
    return observation_estimate, TendencySource.NONE


def _estimate_source_tendencies(
    frames_dbz: Tensor,
    masks: Tensor,
    linear: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> _SourceTendencyEstimate:
    adjacent_estimates: list[
        tuple[int, tuple[Tensor, Tensor, Tensor]]
    ] = []
    for pair_index, (previous_index, current_index) in enumerate(
        ((0, 1), (1, 2))
    ):
        estimate = _estimate_available_pair(
            frames_dbz,
            masks,
            linear,
            previous_index,
            current_index,
            config,
            grid_time_contract,
        )
        if estimate is not None:
            adjacent_estimates.append((pair_index, estimate))

    if len(adjacent_estimates) == 2:
        first_motion, first_growth, first_psr = adjacent_estimates[0][1]
        second_motion, second_growth, second_psr = adjacent_estimates[1][1]
        motion_disagreement = torch.linalg.vector_norm(
            second_motion - first_motion
        )
        motion_disagreement_mps = _motion_disagreement_mps(
            first_motion,
            second_motion,
            config,
            grid_time_contract,
        )
        growth_disagreement = torch.abs(second_growth - first_growth)
        motion_is_inconsistent = _motion_pairs_are_inconsistent(
            motion_disagreement,
            motion_disagreement_mps,
            config,
        )
        growth_is_inconsistent = float(growth_disagreement.detach()) >= (
            config.maximum_pair_growth_disagreement - config.epsilon
        )
        motion, motion_indices, motion_selection = _combine_pair_component(
            first_motion,
            second_motion,
            first_psr,
            second_psr,
            inconsistent=motion_is_inconsistent,
            config=config,
        )
        growth, growth_indices, growth_selection = _combine_pair_component(
            first_growth,
            second_growth,
            first_psr,
            second_psr,
            inconsistent=growth_is_inconsistent,
            config=config,
        )
        used_indices = tuple(sorted(set(motion_indices) | set(growth_indices)))
        minimum_psr = (
            torch.min(
                torch.stack(
                    tuple(
                        (first_psr, second_psr)[index]
                        for index in used_indices
                    )
                )
            )
            if used_indices
            else linear.new_full((), torch.nan)
        )
        return _SourceTendencyEstimate(
            displacement_yx=motion,
            log_growth_per_step=growth,
            motion_disagreement_px=motion_disagreement,
            motion_disagreement_mps=motion_disagreement_mps,
            growth_disagreement=growth_disagreement,
            minimum_phase_correlation_psr=minimum_psr,
            tendency_pair_count=len(used_indices),
            motion_pair_count=len(motion_indices),
            growth_pair_count=len(growth_indices),
            motion_pair_selection=motion_selection,
            growth_pair_selection=growth_selection,
            motion_pair_conflict=motion_is_inconsistent,
            growth_pair_conflict=growth_is_inconsistent,
        )

    zero_motion = linear.new_zeros(2)
    zero_growth = linear.new_zeros(())
    unavailable_psr = linear.new_full((), torch.nan)
    if adjacent_estimates:
        adjacent_pair_index, adjacent_estimate = adjacent_estimates[0]
        long_estimate = _estimate_available_pair(
            frames_dbz,
            masks,
            linear,
            0,
            2,
            config,
            grid_time_contract,
        )
        if long_estimate is None:
            motion, growth, psr = adjacent_estimate
            return _single_pair_tendency(motion, growth, psr)
        return _combine_single_adjacent_and_long(
            adjacent_estimate,
            long_estimate,
            adjacent_pair_index,
            masks,
            config,
            grid_time_contract,
        )

    long_estimate = _estimate_available_pair(
        frames_dbz,
        masks,
        linear,
        0,
        2,
        config,
        grid_time_contract,
    )
    if long_estimate is None:
        return _SourceTendencyEstimate(
            displacement_yx=zero_motion,
            log_growth_per_step=zero_growth,
            motion_disagreement_px=zero_growth,
            motion_disagreement_mps=unavailable_psr,
            growth_disagreement=zero_growth,
            minimum_phase_correlation_psr=unavailable_psr,
            tendency_pair_count=0,
            motion_pair_count=0,
            growth_pair_count=0,
            motion_pair_selection=TendencyPairSelection.NONE,
            growth_pair_selection=TendencyPairSelection.NONE,
            motion_pair_conflict=False,
            growth_pair_conflict=False,
        )

    motion, growth, psr = long_estimate
    return _single_pair_tendency(
        motion,
        growth,
        psr,
        selection=TendencyPairSelection.LONG,
    )


def _single_pair_tendency(
    motion: Tensor,
    growth: Tensor,
    psr: Tensor,
    *,
    selection: TendencyPairSelection = TendencyPairSelection.SINGLE,
) -> _SourceTendencyEstimate:
    zero = growth.new_zeros(())
    return _SourceTendencyEstimate(
        displacement_yx=motion,
        log_growth_per_step=growth,
        motion_disagreement_px=zero,
        motion_disagreement_mps=psr.new_full((), torch.nan),
        growth_disagreement=zero,
        minimum_phase_correlation_psr=psr,
        tendency_pair_count=1,
        motion_pair_count=1,
        growth_pair_count=1,
        motion_pair_selection=selection,
        growth_pair_selection=selection,
        motion_pair_conflict=False,
        growth_pair_conflict=False,
    )


def _combine_single_adjacent_and_long(
    adjacent: tuple[Tensor, Tensor, Tensor],
    long: tuple[Tensor, Tensor, Tensor],
    adjacent_pair_index: int,
    masks: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> _SourceTendencyEstimate:
    adjacent_motion, adjacent_growth, adjacent_psr = adjacent
    long_motion, long_growth, long_psr = long
    motion_disagreement = torch.linalg.vector_norm(
        long_motion - adjacent_motion
    )
    motion_disagreement_mps = _motion_disagreement_mps(
        adjacent_motion,
        long_motion,
        config,
        grid_time_contract,
    )
    growth_disagreement = torch.abs(long_growth - adjacent_growth)
    motion_is_inconsistent = _motion_pairs_are_inconsistent(
        motion_disagreement,
        motion_disagreement_mps,
        config,
    )
    growth_is_inconsistent = float(growth_disagreement.detach()) >= (
        config.maximum_pair_growth_disagreement - config.epsilon
    )
    adjacent_previous = adjacent_pair_index
    adjacent_current = adjacent_pair_index + 1
    adjacent_confidence = _pair_confidence(
        adjacent_psr,
        masks,
        adjacent_previous,
        adjacent_current,
        span_penalty=1.0,
    )
    long_confidence = _pair_confidence(
        long_psr,
        masks,
        0,
        2,
        span_penalty=config.long_pair_confidence_penalty,
    )
    motion, motion_adjacent, motion_long, motion_selection = (
        _select_single_adjacent_or_long_component(
            adjacent_motion,
            long_motion,
            adjacent_confidence,
            long_confidence,
            inconsistent=motion_is_inconsistent,
            config=config,
        )
    )
    growth, growth_adjacent, growth_long, growth_selection = (
        _select_single_adjacent_or_long_component(
            adjacent_growth,
            long_growth,
            adjacent_confidence,
            long_confidence,
            inconsistent=growth_is_inconsistent,
            config=config,
        )
    )
    adjacent_used = motion_adjacent or growth_adjacent
    long_used = motion_long or growth_long
    used_psrs = []
    if adjacent_used:
        used_psrs.append(adjacent_psr)
    if long_used:
        used_psrs.append(long_psr)
    minimum_psr = (
        torch.min(torch.stack(used_psrs))
        if used_psrs
        else adjacent_psr.new_full((), torch.nan)
    )
    return _SourceTendencyEstimate(
        displacement_yx=motion,
        log_growth_per_step=growth,
        motion_disagreement_px=motion_disagreement,
        motion_disagreement_mps=motion_disagreement_mps,
        growth_disagreement=growth_disagreement,
        minimum_phase_correlation_psr=minimum_psr,
        tendency_pair_count=int(adjacent_used) + int(long_used),
        motion_pair_count=int(motion_adjacent) + int(motion_long),
        growth_pair_count=int(growth_adjacent) + int(growth_long),
        motion_pair_selection=motion_selection,
        growth_pair_selection=growth_selection,
        motion_pair_conflict=motion_is_inconsistent,
        growth_pair_conflict=growth_is_inconsistent,
    )


def _pair_confidence(
    psr: Tensor,
    masks: Tensor,
    previous_index: int,
    current_index: int,
    *,
    span_penalty: float,
) -> Tensor:
    common_coverage = torch.mean(
        (masks[previous_index] & masks[current_index]).to(dtype=psr.dtype)
    )
    return psr * common_coverage * span_penalty


def _select_single_adjacent_or_long_component(
    adjacent: Tensor,
    long: Tensor,
    adjacent_confidence: Tensor,
    long_confidence: Tensor,
    *,
    inconsistent: bool,
    config: NowcastConfig,
) -> tuple[Tensor, bool, bool, TendencyPairSelection]:
    confidence_advantage = float(
        (long_confidence - adjacent_confidence).detach()
    )
    long_is_clearly_better = (
        confidence_advantage >= config.minimum_pair_psr_advantage
    )
    adjacent_is_clearly_better = (
        -confidence_advantage >= config.minimum_pair_psr_advantage
    )
    if inconsistent:
        if long_is_clearly_better:
            return long, False, True, TendencyPairSelection.LONG
        if adjacent_is_clearly_better:
            return adjacent, True, False, TendencyPairSelection.SINGLE
        return (
            torch.zeros_like(adjacent),
            False,
            False,
            TendencyPairSelection.PERSISTENCE,
        )
    if long_is_clearly_better:
        return long, False, True, TendencyPairSelection.LONG
    return adjacent, True, False, TendencyPairSelection.SINGLE


def _motion_disagreement_mps(
    first_motion: Tensor,
    second_motion: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> Tensor:
    if grid_time_contract is None:
        return first_motion.new_full((), torch.nan)
    projected = grid_time_contract.projected_displacement_xy(
        second_motion - first_motion
    )
    return torch.linalg.vector_norm(projected) / (
        config.interval_minutes * 60.0
    )


def _motion_pairs_are_inconsistent(
    motion_disagreement_px: Tensor,
    motion_disagreement_mps: Tensor,
    config: NowcastConfig,
) -> bool:
    if bool(torch.isfinite(motion_disagreement_mps)):
        disagreement = float(motion_disagreement_mps.detach())
        limit = config.maximum_pair_velocity_disagreement_mps
    else:
        disagreement = float(motion_disagreement_px.detach())
        limit = config.maximum_pair_motion_disagreement_px
    return disagreement >= limit - config.epsilon


def _combine_pair_component(
    first: Tensor,
    second: Tensor,
    first_psr: Tensor,
    second_psr: Tensor,
    *,
    inconsistent: bool,
    config: NowcastConfig,
) -> tuple[Tensor, tuple[int, ...], TendencyPairSelection]:
    if inconsistent:
        psr_difference = float((second_psr - first_psr).detach())
        if psr_difference >= config.minimum_pair_psr_advantage:
            return second, (1,), TendencyPairSelection.RECENT
        if -psr_difference >= config.minimum_pair_psr_advantage:
            return first, (0,), TendencyPairSelection.EARLIER
        return torch.zeros_like(first), (), TendencyPairSelection.PERSISTENCE

    first_weight = (1.0 - config.recent_weight) * first_psr
    second_weight = config.recent_weight * second_psr
    total_weight = (first_weight + second_weight).clamp_min(config.epsilon)
    combined = (first_weight * first + second_weight * second) / total_weight
    first_used = float(first_weight.detach()) > config.epsilon
    second_used = float(second_weight.detach()) > config.epsilon
    if first_used and second_used:
        return combined, (0, 1), TendencyPairSelection.BLENDED
    if first_used:
        return combined, (0,), TendencyPairSelection.EARLIER
    return combined, (1,), TendencyPairSelection.RECENT


def _estimate_available_pair(
    frames_dbz: Tensor,
    masks: Tensor,
    linear: Tensor,
    previous_index: int,
    current_index: int,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    previous_mask = masks[previous_index]
    current_mask = masks[current_index]
    common = previous_mask & current_mask
    if not _has_complete_echo_neighborhood(
        frames_dbz[previous_index],
        frames_dbz[current_index],
        common,
        config,
        grid_time_contract,
    ):
        return None

    floor = frames_dbz.new_full((), config.min_dbz)
    previous_dbz = torch.where(
        common,
        frames_dbz[previous_index],
        floor,
    )
    current_dbz = torch.where(
        common,
        frames_dbz[current_index],
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
    per_step_limits = motion_displacement_limits_yx(
        config,
        grid_time_contract,
        previous_dbz,
    )
    total_motion, phase_correlation_psr, search_interior = (
        _phase_correlation_shift_and_psr(
            previous_dbz,
            current_dbz,
            config,
            max_displacement_yx=per_step_limits * step_span,
            grid_time_contract=grid_time_contract,
        )
    )
    if not search_interior or float(phase_correlation_psr.detach()) < (
        config.minimum_phase_correlation_psr
    ):
        return None
    if (
        config.maximum_motion_speed_mps is not None
        and grid_time_contract is not None
    ):
        projected = grid_time_contract.projected_displacement_xy(total_motion)
        seconds = step_span * config.interval_minutes * 60.0
        speed = torch.linalg.vector_norm(projected) / seconds
        if float(speed.detach()) > (
            config.maximum_motion_speed_mps + config.epsilon
        ):
            return None
    previous_echo = torch.where(
        previous_mask,
        linear[previous_index],
        linear.new_zeros(()),
    )
    current_echo = torch.where(
        current_mask,
        linear[current_index],
        linear.new_zeros(()),
    )
    total_growth = _log_aligned_growth(
        previous_echo,
        current_echo,
        previous_mask,
        current_mask,
        total_motion,
        config,
        max_log_growth=config.max_log_growth_per_step * step_span,
    )
    return (
        total_motion / step_span,
        total_growth / step_span,
        phase_correlation_psr,
    )


def _has_complete_echo_neighborhood(
    previous_dbz: Tensor,
    current_dbz: Tensor,
    common: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> bool:
    active_echo = (
        (previous_dbz >= config.echo_threshold_dbz)
        | (current_dbz >= config.echo_threshold_dbz)
    )
    if not bool(torch.any(active_echo)):
        return False

    if config.pair_echo_dilation_m is not None:
        if grid_time_contract is None:
            raise ValueError(
                "pair_echo_dilation_m requires a grid/time contract"
            )
        offsets = grid_time_contract.pixel_offsets_within_distance(
            config.pair_echo_dilation_m,
            maximum_radius_yx=(active_echo.shape[0] - 1, active_echo.shape[1] - 1),
        )
        near_echo = _dilate_mask(active_echo, offsets)
    else:
        radius = config.pair_echo_dilation_px
        near_echo = (
            _dilate_mask(
                active_echo,
                tuple(
                    (row, column)
                    for row in range(-radius, radius + 1)
                    for column in range(-radius, radius + 1)
                ),
            )
            if radius
            else active_echo
        )

    return not bool(torch.any(near_echo & ~common))


def _dilate_mask(
    mask: Tensor,
    offsets_yx: tuple[tuple[int, int], ...],
) -> Tensor:
    height, width = mask.shape
    result = torch.zeros_like(mask)
    for offset_y, offset_x in offsets_yx:
        if abs(offset_y) >= height or abs(offset_x) >= width:
            continue
        source_y = slice(max(0, -offset_y), min(height, height - offset_y))
        source_x = slice(max(0, -offset_x), min(width, width - offset_x))
        target_y = slice(max(0, offset_y), min(height, height + offset_y))
        target_x = slice(max(0, offset_x), min(width, width + offset_x))
        result[target_y, target_x] |= mask[source_y, source_x]
    return result


def _merge_current_state(
    prepared: PreparedRadarInput,
    observation_linear: Tensor,
    background_linear: Tensor,
    displacement: Tensor,
    growth: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor, float]:
    observation_echo, observation_support = _merge_source_frames(
        observation_linear,
        prepared.observed_mask,
        displacement,
        growth,
        config,
    )
    background_echo, background_support = _merge_source_frames(
        background_linear,
        prepared.background_mask,
        displacement,
        growth,
        config,
    )
    observation_support = observation_support.clamp(0.0, 1.0)
    background_support = background_support.clamp(0.0, 1.0)
    (
        current_support,
        background_contribution,
        contribution_fraction,
    ) = _combine_source_supports(
        observation_support,
        background_support,
        config.epsilon,
    )
    numerator = (
        observation_support * observation_echo
        + background_contribution * background_echo
    )
    current_echo = torch.where(
        current_support > config.epsilon,
        numerator / current_support.clamp_min(config.epsilon),
        torch.zeros_like(numerator),
    )
    return current_echo, current_support, contribution_fraction


def merge_current_support(
    observed_masks: Tensor,
    background_masks: Tensor,
    displacement_yx: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, float]:
    observation_support = _merge_source_support(
        observed_masks,
        displacement_yx,
    ).clamp(0.0, 1.0)
    background_support = _merge_source_support(
        background_masks,
        displacement_yx,
    ).clamp(0.0, 1.0)
    current_support, _, contribution_fraction = _combine_source_supports(
        observation_support,
        background_support,
        config.epsilon,
    )
    return current_support, contribution_fraction


def _combine_source_supports(
    observation_support: Tensor,
    background_support: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor, float]:
    background_contribution = (
        (1.0 - observation_support) * background_support
    )
    current_support = observation_support + background_contribution
    contribution_fraction = float(
        background_contribution.sum()
        / current_support.sum().clamp_min(epsilon)
    )
    return current_support, background_contribution, contribution_fraction


def _merge_source_support(
    masks: Tensor,
    displacement_yx: Tensor,
) -> Tensor:
    support = torch.zeros_like(
        masks[2],
        dtype=displacement_yx.dtype,
        device=displacement_yx.device,
    )
    for source_index in range(3):
        candidate = masks[source_index].to(
            dtype=displacement_yx.dtype,
            device=displacement_yx.device,
        )
        steps = 2 - source_index
        if steps:
            candidate = remap(
                candidate,
                steps * displacement_yx,
            ).clamp(0.0, 1.0)
        support = candidate + (1.0 - candidate) * support
    return support


def _merge_source_frames(
    linear: Tensor,
    masks: Tensor,
    displacement: Tensor,
    growth: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor]:
    latest_mask = masks[2]
    if bool(torch.all(latest_mask)):
        return linear[2], latest_mask.to(dtype=linear.dtype)

    numerator = torch.zeros_like(linear[2])
    support = torch.zeros_like(linear[2])
    for source_index in range(3):
        source_mask = masks[source_index]
        if not bool(torch.any(source_mask)):
            continue

        steps = 2 - source_index
        candidate_support = source_mask.to(dtype=linear.dtype)
        candidate_value = linear[source_index] * candidate_support
        if steps:
            total_displacement = steps * displacement
            candidate_value = react_core(
                remap(candidate_value, total_displacement),
                steps * growth,
            )
            candidate_support = remap(
                candidate_support,
                total_displacement,
            ).clamp(0.0, 1.0)

        numerator = (
            candidate_value
            + (1.0 - candidate_support) * numerator
        )
        support = (
            candidate_support
            + (1.0 - candidate_support) * support
        )

    current_echo = torch.where(
        support > config.epsilon,
        numerator / support.clamp_min(config.epsilon),
        torch.zeros_like(numerator),
    )
    return current_echo, support


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
    run: ForecastRunContract,
    audit: bool = False,
) -> ForecastResult:
    if config != run.config:
        raise ValueError("forecast config must match the run contract")
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
    forecast_dbz = torch.where(
        valid_mask,
        forecast_dbz,
        forecast_dbz.new_full((), torch.nan),
    )
    forecast_dbz_digest = tensor_digest(forecast_dbz)
    valid_mask_digest = tensor_digest(valid_mask)
    state_digest = state_metadata_digest(state, metadata)
    return ForecastResult(
        forecast_dbz=forecast_dbz,
        valid_mask=valid_mask,
        forecast_dbz_digest=forecast_dbz_digest,
        valid_mask_digest=valid_mask_digest,
        state=state,
        metadata=metadata,
        run=run,
        state_metadata_digest=state_digest,
        forecast_run_digest=_forecast_run_identity_digest(
            run,
            state_digest,
            forecast_dbz_digest,
            valid_mask_digest,
        ),
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
    grid_time_contract: RadarGridTimeContract | None = None,
    audit: bool = False,
) -> ForecastResult:
    config = config or NowcastConfig()
    motion_displacement_limits_yx(config, grid_time_contract, frames_dbz)
    if grid_time_contract is not None:
        grid_time_contract.validate_for(
            config,
            background_present=background_frames_dbz is not None,
            background_age_minutes=background_age_minutes,
        )
    prepared = prepare_input(
        frames_dbz,
        config,
        accepted_mask=qc_mask,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
    )
    state, metadata = estimate_prepared_state(
        prepared,
        config,
        grid_time_contract=grid_time_contract,
    )
    run = ForecastRunContract.from_inputs(
        config,
        frames_dbz,
        prepared.observed_mask,
        background_frames_dbz,
        background_age_minutes,
        grid_time_contract=grid_time_contract,
    )
    return forecast_from_state(
        state,
        metadata,
        config,
        run=run,
        audit=audit,
    )


def _forecast_valid_mask(
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
) -> Tensor:
    if metadata.data_status == DataStatus.UNAVAILABLE:
        return torch.zeros(
            (config.forecast_steps,) + state.echo_linear.shape,
            dtype=torch.bool,
            device=state.echo_linear.device,
        )
    source = metadata.source_support.to(
        dtype=state.echo_linear.dtype,
        device=state.echo_linear.device,
    )
    return torch.stack(
        [
            remap(
                source,
                step * state.displacement_yx,
            )
            >= config.min_publish_support
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
    previous_mask: Tensor,
    current_mask: Tensor,
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
    moved_support = remap(
        previous_mask.to(dtype=previous.dtype),
        displacement_yx,
    )
    aligned = remap(previous, displacement_yx) / moved_support.clamp_min(
        config.epsilon
    )
    overlap = current_mask & (moved_support > config.epsilon)
    if int(overlap.sum()) < 4:
        return previous.new_zeros(())

    previous_integrated_echo = aligned[overlap].sum()
    current_integrated_echo = current[overlap].sum()
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


def _phase_correlation_shift(
    previous_dbz: Tensor,
    current_dbz: Tensor,
    config: NowcastConfig,
    *,
    max_displacement_yx: Tensor | None = None,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> Tensor:
    shift, _, search_interior = _phase_correlation_shift_and_psr(
        previous_dbz,
        current_dbz,
        config,
        max_displacement_yx=max_displacement_yx,
        grid_time_contract=grid_time_contract,
    )
    return shift if search_interior else torch.zeros_like(shift)


def _phase_correlation_shift_and_psr(
    previous_dbz: Tensor,
    current_dbz: Tensor,
    config: NowcastConfig,
    *,
    max_displacement_yx: Tensor | None = None,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> tuple[Tensor, Tensor, bool]:
    previous = (previous_dbz - config.echo_threshold_dbz).clamp_min(0.0)
    current = (current_dbz - config.echo_threshold_dbz).clamp_min(0.0)

    energy = (
        torch.linalg.vector_norm(previous)
        * torch.linalg.vector_norm(current)
    )
    if float(energy.detach()) <= config.epsilon:
        return previous.new_zeros(2), previous.new_zeros(()), False

    height, width = previous.shape
    previous = previous - previous.mean()
    current = current - current.mean()
    centered_energy = (
        torch.linalg.vector_norm(previous)
        * torch.linalg.vector_norm(current)
    )
    if float(centered_energy.detach()) <= config.epsilon:
        return previous.new_zeros(2), previous.new_zeros(()), False

    padded_shape = (2 * height, 2 * width)
    cross_power = torch.fft.fft2(current, s=padded_shape) * torch.conj(
        torch.fft.fft2(previous, s=padded_shape)
    )
    cross_power = cross_power / cross_power.abs().clamp_min(config.epsilon)
    correlation = torch.fft.ifft2(cross_power).real

    peak_index = int(torch.argmax(correlation).item())
    correlation_height, correlation_width = correlation.shape
    peak_y, peak_x = divmod(peak_index, correlation_width)
    psr = _peak_to_sidelobe_ratio(
        correlation,
        peak_y,
        peak_x,
        config,
        grid_time_contract,
    )
    offset_y = _parabolic_peak_offset(correlation[:, peak_x], peak_y, config)
    offset_x = _parabolic_peak_offset(correlation[peak_y, :], peak_x, config)

    shift_y = peak_y + offset_y
    shift_x = peak_x + offset_x
    if shift_y > correlation_height / 2:
        shift_y -= correlation_height
    if shift_x > correlation_width / 2:
        shift_x -= correlation_width

    shift = correlation.new_tensor((shift_y, shift_x))
    peak_shift_y = peak_y
    peak_shift_x = peak_x
    if peak_shift_y > correlation_height / 2:
        peak_shift_y -= correlation_height
    if peak_shift_x > correlation_width / 2:
        peak_shift_x -= correlation_width
    integer_peak_shift = correlation.new_tensor(
        (peak_shift_y, peak_shift_x)
    )
    requested_limits = (
        correlation.new_full((2,), config.max_displacement_px)
        if max_displacement_yx is None
        else max_displacement_yx.to(
            dtype=correlation.dtype,
            device=correlation.device,
        )
    )
    if requested_limits.shape != (2,) or not bool(
        torch.all(torch.isfinite(requested_limits) & (requested_limits > 0))
    ):
        raise ValueError("max_displacement_yx must be a positive finite [2]")
    limits = torch.minimum(
        requested_limits,
        correlation.new_tensor((height - 1, width - 1)),
    )
    inside_limits = bool(
        torch.all(torch.abs(shift) <= limits + config.epsilon)
    )
    interior_bin_limit = (limits - 0.5).clamp_min(config.epsilon)
    away_from_search_boundary = bool(
        torch.all(torch.abs(integer_peak_shift) < interior_bin_limit)
    )
    return shift, psr, inside_limits and away_from_search_boundary


def _peak_to_sidelobe_ratio(
    correlation: Tensor,
    peak_y: int,
    peak_x: int,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> Tensor:
    height, width = correlation.shape
    y_distance = torch.abs(
        torch.arange(height, device=correlation.device) - peak_y
    )
    x_distance = torch.abs(
        torch.arange(width, device=correlation.device) - peak_x
    )
    y_distance = torch.minimum(y_distance, height - y_distance)
    x_distance = torch.minimum(x_distance, width - x_distance)
    if config.phase_correlation_sidelobe_radius_m is None:
        radius = config.phase_correlation_sidelobe_radius_px
        sidelobe_mask = (y_distance[:, None] > radius) | (
            x_distance[None, :] > radius
        )
    else:
        if grid_time_contract is None:
            raise ValueError(
                "phase_correlation_sidelobe_radius_m requires a grid/time contract"
            )
        assert grid_time_contract.pixel_to_projected_matrix_m is not None
        (a, b), (c, d) = grid_time_contract.pixel_to_projected_matrix_m
        signed_y = torch.remainder(
            torch.arange(height, device=correlation.device)
            - peak_y
            + height // 2,
            height,
        ) - height // 2
        signed_x = torch.remainder(
            torch.arange(width, device=correlation.device)
            - peak_x
            + width // 2,
            width,
        ) - width // 2
        row = signed_y[:, None].to(dtype=correlation.dtype)
        column = signed_x[None, :].to(dtype=correlation.dtype)
        distance = torch.sqrt(
            (a * column + b * row).square()
            + (c * column + d * row).square()
        )
        sidelobe_mask = distance > config.phase_correlation_sidelobe_radius_m
    sidelobe = correlation[sidelobe_mask]
    if sidelobe.numel() < 2:
        return correlation.new_zeros(())
    sidelobe_mean = sidelobe.mean()
    sidelobe_std = sidelobe.std(correction=0)
    return (
        correlation[peak_y, peak_x] - sidelobe_mean
    ) / sidelobe_std.clamp_min(config.epsilon)


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
