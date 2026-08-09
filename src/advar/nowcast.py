from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math

import torch
from torch import Tensor

from ._digest import dataclass_digest, json_digest, tensor_digest
from .calibration import (
    OperationalCalibrationManifest,
    OperationalDataIdentity,
    algorithm_bundle_digest,
)
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


_OPERATIONAL_RUNTIME_PROFILE_VERSION = "operational-runtime-profile-v2"


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
    maximum_local_growth_log_error_per_step: float = math.log(1.50)
    minimum_pair_psr_advantage: float = 3.0
    minimum_pair_confidence_ratio: float = 1.5
    long_pair_confidence_penalty: float = 0.5
    minimum_phase_correlation_psr: float = 8.0
    phase_correlation_sidelobe_radius_px: int = 2
    phase_correlation_sidelobe_radius_m: float | None = None
    max_log_growth_per_step: float = math.log(1.35)
    minimum_growth_overlap_support: float = 4.0
    minimum_growth_overlap_area_km2: float | None = None
    growth_decay_minutes: float = 60.0
    maximum_background_age_minutes: float = 60.0
    min_publish_support: float = 0.95
    minimum_publish_verified_support: float | None = None
    minimum_publish_confidence: float | None = None
    minimum_publish_observation_verified_support: float | None = None
    maximum_publish_background_fraction: float | None = None
    forecast_velocity_uncertainty_mps: float = 1.0
    forecast_confidence_length_scale_m: float = 10_000.0
    forecast_log_growth_uncertainty_per_step: float = 0.05
    forecast_log_growth_confidence_scale: float = 1.0
    p1_motion_saturation_safe_margin_mps: float = 2.0
    p1_growth_saturation_safe_margin_per_step: float = math.log(1.05)
    p1_posterior_saturation_sigma_multiplier: float = 2.0
    p1_saturation_uncertainty_multiplier: float = 4.0
    single_pair_uncertainty_multiplier: float = 2.0
    persistence_uncertainty_multiplier: float = 4.0
    background_tendency_age_uncertainty_scale_minutes: float = 60.0
    epsilon: float = 1.0e-6
    support_presence_threshold: float = 1.0e-6
    contract_absolute_tolerance: float = 1.0e-6
    ratio_regularizer: float = 1.0e-12

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
            self.maximum_local_growth_log_error_per_step,
            self.minimum_pair_psr_advantage,
            self.minimum_pair_confidence_ratio,
            self.long_pair_confidence_penalty,
            self.minimum_phase_correlation_psr,
            self.max_log_growth_per_step,
            self.minimum_growth_overlap_support,
            self.growth_decay_minutes,
            self.maximum_background_age_minutes,
            self.min_publish_support,
            self.forecast_velocity_uncertainty_mps,
            self.forecast_confidence_length_scale_m,
            self.forecast_log_growth_uncertainty_per_step,
            self.forecast_log_growth_confidence_scale,
            self.p1_motion_saturation_safe_margin_mps,
            self.p1_growth_saturation_safe_margin_per_step,
            self.p1_posterior_saturation_sigma_multiplier,
            self.p1_saturation_uncertainty_multiplier,
            self.single_pair_uncertainty_multiplier,
            self.persistence_uncertainty_multiplier,
            self.background_tendency_age_uncertainty_scale_minutes,
            self.epsilon,
            self.support_presence_threshold,
            self.contract_absolute_tolerance,
            self.ratio_regularizer,
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
            "maximum_local_growth_log_error_per_step": (
                self.maximum_local_growth_log_error_per_step
            ),
        }
        for name, value in pair_limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_pair_psr_advantage <= 0:
            raise ValueError("minimum_pair_psr_advantage must be positive")
        if self.minimum_pair_confidence_ratio <= 1:
            raise ValueError("minimum_pair_confidence_ratio must exceed 1")
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
        if self.minimum_growth_overlap_support <= 0:
            raise ValueError("minimum_growth_overlap_support must be positive")
        if self.minimum_growth_overlap_area_km2 is not None and (
            isinstance(self.minimum_growth_overlap_area_km2, bool)
            or not isinstance(
                self.minimum_growth_overlap_area_km2,
                (int, float),
            )
            or not math.isfinite(self.minimum_growth_overlap_area_km2)
            or self.minimum_growth_overlap_area_km2 <= 0
        ):
            raise ValueError(
                "minimum_growth_overlap_area_km2 must be positive"
            )
        if self.growth_decay_minutes <= 0:
            raise ValueError("growth_decay_minutes must be positive")
        if self.maximum_background_age_minutes <= 0:
            raise ValueError("maximum_background_age_minutes must be positive")
        if not 0.0 < self.min_publish_support <= 1.0:
            raise ValueError("min_publish_support must be in (0, 1]")
        if isinstance(self.forecast_velocity_uncertainty_mps, bool) or (
            self.forecast_velocity_uncertainty_mps <= 0
        ):
            raise ValueError(
                "forecast_velocity_uncertainty_mps must be positive"
            )
        if isinstance(self.forecast_confidence_length_scale_m, bool) or (
            self.forecast_confidence_length_scale_m <= 0
        ):
            raise ValueError(
                "forecast_confidence_length_scale_m must be positive"
            )
        if isinstance(
            self.forecast_log_growth_uncertainty_per_step,
            bool,
        ) or self.forecast_log_growth_uncertainty_per_step <= 0:
            raise ValueError(
                "forecast_log_growth_uncertainty_per_step must be positive"
            )
        if isinstance(self.forecast_log_growth_confidence_scale, bool) or (
            self.forecast_log_growth_confidence_scale <= 0
        ):
            raise ValueError(
                "forecast_log_growth_confidence_scale must be positive"
            )
        p1_saturation_values = {
            "p1_motion_saturation_safe_margin_mps": (
                self.p1_motion_saturation_safe_margin_mps
            ),
            "p1_growth_saturation_safe_margin_per_step": (
                self.p1_growth_saturation_safe_margin_per_step
            ),
            "p1_posterior_saturation_sigma_multiplier": (
                self.p1_posterior_saturation_sigma_multiplier
            ),
            "p1_saturation_uncertainty_multiplier": (
                self.p1_saturation_uncertainty_multiplier
            ),
        }
        for name, value in p1_saturation_values.items():
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.single_pair_uncertainty_multiplier, bool) or (
            self.single_pair_uncertainty_multiplier < 1.0
        ):
            raise ValueError(
                "single_pair_uncertainty_multiplier must be at least 1"
            )
        if isinstance(self.persistence_uncertainty_multiplier, bool) or (
            self.persistence_uncertainty_multiplier
            < self.single_pair_uncertainty_multiplier
        ):
            raise ValueError(
                "persistence_uncertainty_multiplier cannot be smaller than "
                "single_pair_uncertainty_multiplier"
            )
        if isinstance(
            self.background_tendency_age_uncertainty_scale_minutes,
            bool,
        ) or self.background_tendency_age_uncertainty_scale_minutes <= 0:
            raise ValueError(
                "background_tendency_age_uncertainty_scale_minutes must be "
                "positive"
            )
        if self.minimum_publish_verified_support is not None and (
            isinstance(self.minimum_publish_verified_support, bool)
            or not isinstance(
                self.minimum_publish_verified_support,
                (int, float),
            )
            or not math.isfinite(self.minimum_publish_verified_support)
            or not 0.0 < self.minimum_publish_verified_support <= 1.0
        ):
            raise ValueError(
                "minimum_publish_verified_support must be in (0, 1]"
            )
        optional_unit_thresholds = {
            "minimum_publish_confidence": self.minimum_publish_confidence,
            "minimum_publish_observation_verified_support": (
                self.minimum_publish_observation_verified_support
            ),
            "maximum_publish_background_fraction": (
                self.maximum_publish_background_fraction
            ),
        }
        for name, value in optional_unit_thresholds.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")
        positive_publication_thresholds = {
            "minimum_publish_confidence": self.minimum_publish_confidence,
            "minimum_publish_observation_verified_support": (
                self.minimum_publish_observation_verified_support
            ),
        }
        for name, value in positive_publication_thresholds.items():
            if value == 0.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.support_presence_threshold <= 0:
            raise ValueError("support_presence_threshold must be positive")
        if self.contract_absolute_tolerance <= 0:
            raise ValueError("contract_absolute_tolerance must be positive")
        if self.ratio_regularizer <= 0:
            raise ValueError("ratio_regularizer must be positive")

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


class DynamicsSource(str, Enum):
    P0_RECONSTRUCTION = "P0_RECONSTRUCTION"
    P1_VARIATIONAL = "P1_VARIATIONAL"
    P0_FALLBACK = "P0_FALLBACK"


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
    source_displacement_yx: Tensor
    source_log_growth: Tensor
    source_usable: Tensor
    source_support_displacements_yx: Tensor
    motion_disagreement_px: Tensor
    motion_disagreement_mps: Tensor
    growth_disagreement: Tensor
    maximum_growth_saturation_excess: Tensor
    minimum_phase_correlation_psr: Tensor
    tendency_pair_count: int
    motion_pair_count: int
    growth_pair_count: int
    motion_pair_selection: TendencyPairSelection
    growth_pair_selection: TendencyPairSelection
    motion_pair_spans: tuple[tuple[int, int], ...]
    growth_pair_spans: tuple[tuple[int, int], ...]
    motion_pair_conflict: bool
    growth_pair_conflict: bool
    minimum_growth_overlap_support: Tensor
    minimum_growth_overlap_area_km2: Tensor
    reconstruction_pair_count: int
    reconstruction_selection: TendencyPairSelection
    reconstruction_minimum_psr: Tensor
    reconstruction_recent_psr: Tensor
    reconstruction_conflict: bool
    reconstruction_extrapolated: bool

    @property
    def future_available(self) -> bool:
        return self.tendency_pair_count > 0

    @property
    def reconstruction_available(self) -> bool:
        return bool(torch.any(self.source_usable))


@dataclass(frozen=True)
class _GrowthEvidence:
    value: Tensor
    raw_value: Tensor
    saturation_excess: Tensor
    available: bool
    overlap_support: Tensor
    overlap_area_km2: Tensor
    aligned_previous_integral: Tensor
    current_integral: Tensor
    alignment_log_error: Tensor


def _maximum_growth_saturation_excess(
    evidence: tuple[_GrowthEvidence, ...],
    reference: Tensor,
) -> Tensor:
    available = tuple(item for item in evidence if item.available)
    if not available:
        return reference.new_zeros(())
    return torch.max(
        torch.stack(tuple(item.saturation_excess for item in available))
    )


def _minimum_growth_evidence(
    evidence: tuple[_GrowthEvidence, ...],
    used_indices: tuple[int, ...],
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    if not used_indices:
        unavailable = reference.new_full((), torch.nan)
        return unavailable, unavailable.clone()
    used = tuple(evidence[index] for index in used_indices)
    minimum_support = torch.min(
        torch.stack(tuple(item.overlap_support for item in used))
    )
    areas = torch.stack(tuple(item.overlap_area_km2 for item in used))
    minimum_area = (
        torch.min(areas)
        if bool(torch.all(torch.isfinite(areas)))
        else reference.new_full((), torch.nan)
    )
    return minimum_support, minimum_area


_FORECAST_INPUT_BUNDLE_VERSION = "forecast-input-bundle-v5"
_FORECAST_RUN_IDENTITY_VERSION = "forecast-run-identity-v14"


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


def _validate_state_dynamics(
    state: RadarState,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> None:
    displacement = state.displacement_yx
    growth = state.log_growth_per_step
    if displacement.shape != (2,) or not bool(
        torch.all(torch.isfinite(displacement))
    ):
        raise ValueError("state displacement must be finite with shape [2]")
    if growth.ndim != 0 or not bool(torch.isfinite(growth)):
        raise ValueError("state log growth must be a finite scalar")
    if config.maximum_motion_speed_mps is None:
        motion_within_limit = bool(
            torch.all(
                torch.abs(displacement)
                <= config.max_displacement_px
                + config.contract_absolute_tolerance
            )
        )
    else:
        if grid_time_contract is None:
            raise ValueError(
                "physical state motion requires a grid/time contract"
            )
        speed = torch.linalg.vector_norm(
            grid_time_contract.projected_velocity_xy(
                displacement,
                config.interval_minutes,
            )
        )
        motion_within_limit = bool(
            speed
            <= config.maximum_motion_speed_mps
            + config.contract_absolute_tolerance
        )
    if not motion_within_limit:
        raise ValueError("state motion exceeds the configured limit")
    if bool(
        torch.abs(growth)
        > config.max_log_growth_per_step
        + config.contract_absolute_tolerance
    ):
        raise ValueError("state log growth exceeds the configured limit")


@dataclass(frozen=True)
class StatePathProvenance:
    mode: TendencyPairSelection = TendencyPairSelection.NONE
    pair_count: int = 0
    minimum_psr: float = math.nan
    conflict: bool = False
    extrapolated: bool = False
    age_minutes: float | None = None


@dataclass(frozen=True)
class ForecastMetadata:
    data_status: DataStatus
    coverage_by_frame: Tensor
    background_used: bool
    background_contribution_fraction: float
    background_age_minutes: float | None
    source_support: Tensor
    observation_source_support: Tensor
    background_source_support: Tensor
    path_verified_source_support: Tensor
    verified_source_support: Tensor
    local_motion_verified_support: Tensor
    local_growth_verified_support: Tensor
    local_dynamics_verified_support: Tensor
    observation_verified_source_support: Tensor
    background_verified_source_support: Tensor
    motion_disagreement_px: Tensor
    motion_disagreement_mps: Tensor
    growth_disagreement: Tensor
    maximum_growth_saturation_excess: Tensor
    posterior_velocity_uncertainty_mps: Tensor
    posterior_log_growth_uncertainty_per_step: Tensor
    p1_velocity_saturation_uncertainty_mps: Tensor
    p1_log_growth_saturation_uncertainty_per_step: Tensor
    minimum_phase_correlation_psr: Tensor
    tendency_pair_count: int
    tendency_source: TendencySource
    provenance: str = "p0_support_merged"
    dynamics_source: DynamicsSource = DynamicsSource.P0_RECONSTRUCTION
    motion_pair_count: int = 0
    growth_pair_count: int = 0
    motion_pair_selection: TendencyPairSelection = TendencyPairSelection.NONE
    growth_pair_selection: TendencyPairSelection = TendencyPairSelection.NONE
    motion_pair_conflict: bool = False
    growth_pair_conflict: bool = False
    state_path_source: TendencySource = TendencySource.NONE
    state_path_mode: TendencyPairSelection = TendencyPairSelection.NONE
    state_path_pair_count: int = 0
    state_path_minimum_psr: float = math.nan
    state_path_conflict: bool = False
    state_path_extrapolated: bool = False
    state_path_age_minutes: float | None = None
    observation_path: StatePathProvenance = StatePathProvenance()
    background_path: StatePathProvenance = StatePathProvenance()
    minimum_growth_overlap_support: float = math.nan
    minimum_growth_overlap_area_km2: float = math.nan

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
    def observation_state_support_fraction(self) -> float:
        if not bool(torch.any(self.source_support > 0)):
            return 0.0
        return 1.0 - self.background_contribution_fraction

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
                "observation_state_support_fraction": (
                    metadata.observation_state_support_fraction
                ),
                "background_tendency_used": metadata.background_tendency_used,
                "background_age_minutes": metadata.background_age_minutes,
                "source_support": tensor_digest(metadata.source_support),
                "observation_source_support": tensor_digest(
                    metadata.observation_source_support
                ),
                "background_source_support": tensor_digest(
                    metadata.background_source_support
                ),
                "path_verified_source_support": tensor_digest(
                    metadata.path_verified_source_support
                ),
                "verified_source_support": tensor_digest(
                    metadata.verified_source_support
                ),
                "local_motion_verified_support": tensor_digest(
                    metadata.local_motion_verified_support
                ),
                "local_growth_verified_support": tensor_digest(
                    metadata.local_growth_verified_support
                ),
                "local_dynamics_verified_support": tensor_digest(
                    metadata.local_dynamics_verified_support
                ),
                "observation_verified_source_support": tensor_digest(
                    metadata.observation_verified_source_support
                ),
                "background_verified_source_support": tensor_digest(
                    metadata.background_verified_source_support
                ),
                "motion_disagreement_px": tensor_digest(
                    metadata.motion_disagreement_px
                ),
                "motion_disagreement_mps": tensor_digest(
                    metadata.motion_disagreement_mps
                ),
                "growth_disagreement": tensor_digest(
                    metadata.growth_disagreement
                ),
                "maximum_growth_saturation_excess": tensor_digest(
                    metadata.maximum_growth_saturation_excess
                ),
                "posterior_velocity_uncertainty_mps": tensor_digest(
                    metadata.posterior_velocity_uncertainty_mps
                ),
                "posterior_log_growth_uncertainty_per_step": tensor_digest(
                    metadata.posterior_log_growth_uncertainty_per_step
                ),
                "p1_velocity_saturation_uncertainty_mps": tensor_digest(
                    metadata.p1_velocity_saturation_uncertainty_mps
                ),
                "p1_log_growth_saturation_uncertainty_per_step": (
                    tensor_digest(
                        metadata.p1_log_growth_saturation_uncertainty_per_step
                    )
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
                "dynamics_source": metadata.dynamics_source.value,
                "state_path_source": metadata.state_path_source.value,
                "state_path_mode": metadata.state_path_mode.value,
                "state_path_pair_count": metadata.state_path_pair_count,
                "state_path_minimum_psr": _finite_float_or_none(
                    metadata.state_path_minimum_psr
                ),
                "state_path_conflict": metadata.state_path_conflict,
                "state_path_extrapolated": (
                    metadata.state_path_extrapolated
                ),
                "state_path_age_minutes": metadata.state_path_age_minutes,
                "observation_path": _path_provenance_digest_values(
                    metadata.observation_path
                ),
                "background_path": _path_provenance_digest_values(
                    metadata.background_path
                ),
                "minimum_growth_overlap_support": _finite_float_or_none(
                    metadata.minimum_growth_overlap_support
                ),
                "minimum_growth_overlap_area_km2": _finite_float_or_none(
                    metadata.minimum_growth_overlap_area_km2
                ),
                "provenance": metadata.provenance,
            },
        }
    )


def _finite_float_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _path_provenance_digest_values(
    path: StatePathProvenance,
) -> dict[str, object]:
    return {
        "mode": path.mode.value,
        "pair_count": path.pair_count,
        "minimum_psr": _finite_float_or_none(path.minimum_psr),
        "conflict": path.conflict,
        "extrapolated": path.extrapolated,
        "age_minutes": path.age_minutes,
    }


@dataclass(frozen=True)
class ForecastAudit:
    input_echo: PositivityAudit
    forecast_final: PositivityAudit
    transport: tuple[TransportAudit, ...]


@dataclass(frozen=True)
class AnalysisInputIdentity:
    """One explicit identity for analysis data and its resolved input plan."""

    frames_digest: str
    fixed_context_digest: str
    full_data_digest: str
    input_plan_digest: str | None
    plan_resolution_digest: str | None
    contract: str = "analysis-input-identity-v1"
    identity_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "analysis-input-identity-v1":
            raise ValueError("unsupported analysis input identity")
        _validate_sha256_digest("frames_digest", self.frames_digest)
        _validate_sha256_digest("fixed_context_digest", self.fixed_context_digest)
        _validate_sha256_digest("full_data_digest", self.full_data_digest)
        if self.full_data_digest != _forecast_full_analysis_input_digest(
            input_frames_digest=self.frames_digest,
            fixed_input_context_digest=self.fixed_context_digest,
        ):
            raise ValueError("analysis full-data identity mismatch")
        if (self.input_plan_digest is None) != (
            self.plan_resolution_digest is None
        ):
            raise ValueError("analysis input plan identity is incomplete")
        if self.input_plan_digest is not None:
            assert self.plan_resolution_digest is not None
            _validate_sha256_digest("input_plan_digest", self.input_plan_digest)
            _validate_sha256_digest(
                "plan_resolution_digest",
                self.plan_resolution_digest,
            )
            expected = _forecast_input_plan_resolution_digest(
                input_plan_digest=self.input_plan_digest,
                full_analysis_input_digest=self.full_data_digest,
            )
            if self.plan_resolution_digest != expected:
                raise ValueError("analysis input plan resolution is not v2")
        object.__setattr__(
            self,
            "identity_digest",
            json_digest(
                {
                    key: value
                    for key, value in self.__dict__.items()
                    if key != "identity_digest"
                }
            ),
        )


@dataclass(frozen=True)
class ForecastRunContract:
    config: NowcastConfig
    _latest_frame_dbz: Tensor
    _latest_observation_mask: Tensor
    _latest_background_dbz: Tensor | None
    latest_observation_mask_digest: str
    latest_frame_digest: str
    latest_background_digest: str | None
    input_frames_digest: str
    observation_masks_digest: str | None
    observation_quality_weight_digest: str | None
    observation_std_dbz_digest: str | None
    background_frames_digest: str | None
    fixed_input_context_digest: str | None
    full_analysis_input_digest: str | None
    input_bundle_digest: str
    background_age_minutes: float | None = None
    grid_time_contract: RadarGridTimeContract | None = None
    grid_time_contract_digest: str | None = None
    analysis_config_json: str | None = None
    analysis_config_digest: str | None = None
    analysis_input_digest: str | None = None
    operational_calibration_manifest_json: str | None = None
    operational_calibration_manifest_digest: str | None = None
    operational_calibration_approval_digest: str | None = None
    operational_data_identity_json: str | None = None
    operational_data_identity_digest: str | None = None
    neural_prior_digest: str | None = None
    prior_application_digest: str | None = None
    prior_model_contract_digest: str | None = None
    prior_feature_schema_digest: str | None = None
    prior_training_manifest_digest: str | None = None
    prior_inference_evidence_digest: str | None = None
    prior_inference_algorithm_digest: str | None = None
    prior_numerical_runtime_digest: str | None = None
    prior_dependency: str | None = None
    prior_role: str | None = None
    prior_promotion_evidence_digest: str | None = None
    prior_regime_classification_evidence_digest: str | None = None
    prior_deployment_policy_digest: str | None = None
    prior_deployment_policy_trust_store_digest: str | None = None
    prior_deployment_selection_digest: str | None = None
    prior_deployment_fallback_reason: str | None = None
    prior_deployment_lineage_contract: str = (
        "neural-prior-deployment-lineage-v2"
    )
    prior_lineage_contract: str = "neural-prior-run-lineage-v2"
    input_plan_json: str | None = None
    input_plan_digest: str | None = None
    input_plan_resolution_digest: str | None = None
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
        observation_quality_weight: Tensor | None = None,
        observation_std_dbz: Tensor | None = None,
        grid_time_contract: RadarGridTimeContract | None = None,
        analysis_config_json: str | None = None,
        analysis_config_digest: str | None = None,
        analysis_input_digest: str | None = None,
        operational_calibration_manifest_json: str | None = None,
        operational_calibration_manifest_digest: str | None = None,
        operational_calibration_approval_digest: str | None = None,
        operational_data_identity_json: str | None = None,
        operational_data_identity_digest: str | None = None,
        neural_prior_digest: str | None = None,
        prior_application_digest: str | None = None,
        prior_model_contract_digest: str | None = None,
        prior_feature_schema_digest: str | None = None,
        prior_training_manifest_digest: str | None = None,
        prior_inference_evidence_digest: str | None = None,
        prior_inference_algorithm_digest: str | None = None,
        prior_numerical_runtime_digest: str | None = None,
        prior_dependency: str | None = None,
        prior_role: str | None = None,
        prior_promotion_evidence_digest: str | None = None,
        prior_regime_classification_evidence_digest: str | None = None,
        prior_deployment_policy_digest: str | None = None,
        prior_deployment_policy_trust_store_digest: str | None = None,
        prior_deployment_selection_digest: str | None = None,
        prior_deployment_fallback_reason: str | None = None,
        input_plan_json: str | None = None,
        input_plan_digest: str | None = None,
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
        accepted_quality_weight = (
            observation_masks.to(frames_dbz)
            if observation_quality_weight is None
            else observation_quality_weight
        )
        accepted_observation_std = (
            torch.full_like(frames_dbz, 2.0)
            if observation_std_dbz is None
            else observation_std_dbz
        )
        for name, value in (
            ("observation_quality_weight", accepted_quality_weight),
            ("observation_std_dbz", accepted_observation_std),
        ):
            if (
                value.shape != frames_dbz.shape
                or value.dtype != frames_dbz.dtype
                or not bool(torch.all(torch.isfinite(value)))
            ):
                raise ValueError(f"{name} must be finite and match the frames")
        if bool(
            torch.any(
                (accepted_quality_weight < 0.0)
                | (accepted_quality_weight > 1.0)
            )
        ) or bool(
            torch.any(
                accepted_quality_weight.masked_select(~observation_masks) != 0.0
            )
        ):
            raise ValueError("observation quality weights are invalid")
        if bool(torch.any(accepted_observation_std <= 0.0)):
            raise ValueError("observation standard deviations must be positive")
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
        _validate_operational_calibration_lineage(
            config,
            analysis_config_json,
            grid_time_contract,
            operational_calibration_manifest_json,
            operational_calibration_manifest_digest,
            operational_calibration_approval_digest,
            operational_data_identity_json,
            operational_data_identity_digest,
            FORECAST_INTEGRATOR_VERSION,
        )
        _validate_neural_prior_lineage(
            neural_prior_digest,
            prior_application_digest,
            prior_model_contract_digest,
            prior_feature_schema_digest,
            prior_training_manifest_digest,
            prior_inference_evidence_digest,
            prior_inference_algorithm_digest,
            prior_numerical_runtime_digest,
            prior_dependency,
            prior_role,
            "neural-prior-run-lineage-v2",
        )
        _validate_prior_deployment_lineage(
            analysis_config_json=analysis_config_json,
            prior_role=prior_role,
            promotion_evidence_digest=prior_promotion_evidence_digest,
            regime_classification_evidence_digest=(
                prior_regime_classification_evidence_digest
            ),
            deployment_policy_digest=prior_deployment_policy_digest,
            deployment_policy_trust_store_digest=(
                prior_deployment_policy_trust_store_digest
            ),
            deployment_selection_digest=prior_deployment_selection_digest,
            fallback_reason=prior_deployment_fallback_reason,
            contract="neural-prior-deployment-lineage-v2",
        )
        _validate_input_plan_lineage(input_plan_json, input_plan_digest)
        _validate_input_plan_resolution(
            input_plan_json,
            operational_data_identity_json,
            grid_time_contract,
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
        input_bundle_digest = _forecast_input_bundle_digest(
            frames_dbz,
            observation_masks,
            background_frames_dbz,
            background_age_minutes,
            grid_time_contract,
            operational_calibration_manifest_digest,
            operational_calibration_approval_digest,
            operational_data_identity_digest,
        )
        observation_masks_digest = tensor_digest(observation_masks)
        observation_quality_weight_digest = tensor_digest(
            accepted_quality_weight
        )
        observation_std_dbz_digest = tensor_digest(accepted_observation_std)
        background_frames_digest = (
            None
            if background_frames_dbz is None
            else tensor_digest(background_frames_dbz)
        )
        fixed_input_context_digest = _forecast_fixed_input_context_digest(
            observation_masks_digest=observation_masks_digest,
            observation_quality_weight_digest=(
                observation_quality_weight_digest
            ),
            observation_std_dbz_digest=observation_std_dbz_digest,
            background_frames_digest=background_frames_digest,
            background_age_minutes=background_age_minutes,
            grid_time_contract_digest=(
                None if grid_time_contract is None else grid_time_contract.digest
            ),
            operational_calibration_manifest_digest=(
                operational_calibration_manifest_digest
            ),
            operational_calibration_approval_digest=(
                operational_calibration_approval_digest
            ),
            operational_data_identity_digest=operational_data_identity_digest,
            input_plan_digest=input_plan_digest,
        )
        full_analysis_input_digest = _forecast_full_analysis_input_digest(
            input_frames_digest=tensor_digest(frames_dbz),
            fixed_input_context_digest=fixed_input_context_digest,
        )
        resolution_digest = (
            None
            if input_plan_digest is None
            else _forecast_input_plan_resolution_digest(
                input_plan_digest=input_plan_digest,
                full_analysis_input_digest=full_analysis_input_digest,
            )
        )
        return cls(
            config=config,
            _latest_frame_dbz=accepted_frame,
            _latest_observation_mask=accepted_mask,
            _latest_background_dbz=accepted_background,
            latest_observation_mask_digest=tensor_digest(accepted_mask),
            latest_frame_digest=tensor_digest(latest_frame),
            latest_background_digest=latest_background,
            input_frames_digest=tensor_digest(frames_dbz),
            observation_masks_digest=observation_masks_digest,
            observation_quality_weight_digest=(
                observation_quality_weight_digest
            ),
            observation_std_dbz_digest=observation_std_dbz_digest,
            background_frames_digest=background_frames_digest,
            fixed_input_context_digest=fixed_input_context_digest,
            full_analysis_input_digest=full_analysis_input_digest,
            input_bundle_digest=input_bundle_digest,
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
            operational_calibration_manifest_json=(
                operational_calibration_manifest_json
            ),
            operational_calibration_manifest_digest=(
                operational_calibration_manifest_digest
            ),
            operational_calibration_approval_digest=(
                operational_calibration_approval_digest
            ),
            operational_data_identity_json=operational_data_identity_json,
            operational_data_identity_digest=operational_data_identity_digest,
            neural_prior_digest=neural_prior_digest,
            prior_application_digest=prior_application_digest,
            prior_model_contract_digest=prior_model_contract_digest,
            prior_feature_schema_digest=prior_feature_schema_digest,
            prior_training_manifest_digest=prior_training_manifest_digest,
            prior_inference_evidence_digest=prior_inference_evidence_digest,
            prior_inference_algorithm_digest=prior_inference_algorithm_digest,
            prior_numerical_runtime_digest=prior_numerical_runtime_digest,
            prior_dependency=prior_dependency,
            prior_role=prior_role,
            prior_promotion_evidence_digest=prior_promotion_evidence_digest,
            prior_regime_classification_evidence_digest=(
                prior_regime_classification_evidence_digest
            ),
            prior_deployment_policy_digest=prior_deployment_policy_digest,
            prior_deployment_policy_trust_store_digest=(
                prior_deployment_policy_trust_store_digest
            ),
            prior_deployment_selection_digest=prior_deployment_selection_digest,
            prior_deployment_fallback_reason=prior_deployment_fallback_reason,
            input_plan_json=input_plan_json,
            input_plan_digest=input_plan_digest,
            input_plan_resolution_digest=resolution_digest,
        )

    @property
    def latest_frame_dbz(self) -> Tensor:
        self.validate_integrity()
        return self._latest_frame_dbz.clone()

    @property
    def analysis_input_identity(self) -> AnalysisInputIdentity | None:
        """Return the current v2 data/plan identity, or None for legacy runs."""

        if (
            self.fixed_input_context_digest is None
            or self.full_analysis_input_digest is None
        ):
            return None
        try:
            return AnalysisInputIdentity(
                frames_digest=self.input_frames_digest,
                fixed_context_digest=self.fixed_input_context_digest,
                full_data_digest=self.full_analysis_input_digest,
                input_plan_digest=self.input_plan_digest,
                plan_resolution_digest=self.input_plan_resolution_digest,
            )
        except ValueError:
            return None

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

    @property
    def operational_runtime_profile_digest(self) -> str | None:
        """Return the runtime settings address certified by the manifest."""

        if self.operational_calibration_manifest_json is None:
            return None
        grid = self.grid_time_contract
        if grid is None:
            raise ValueError(
                "operational calibration requires a grid/time contract"
            )
        config_value = None
        if self.analysis_config_json is not None:
            loaded = json.loads(self.analysis_config_json)
            if not isinstance(loaded, dict):
                raise ValueError("analysis_config_json must contain an object")
            config_value = loaded
        return operational_runtime_profile_digest(
            self.config,
            grid,
            analysis_config=config_value,
            forecast_integrator_version=self.forecast_integrator_version,
        )

    def validate_integrity(self) -> None:
        if self.forecast_integrator_version != FORECAST_INTEGRATOR_VERSION:
            raise ValueError(
                "forecast integrator version is incompatible with this runtime"
            )
        _validate_sha256_digest(
            "input_frames_digest",
            self.input_frames_digest,
        )
        _validate_sha256_digest(
            "input_bundle_digest",
            self.input_bundle_digest,
        )
        if self.observation_masks_digest is not None:
            _validate_sha256_digest(
                "observation_masks_digest",
                self.observation_masks_digest,
            )
        if self.observation_quality_weight_digest is not None:
            _validate_sha256_digest(
                "observation_quality_weight_digest",
                self.observation_quality_weight_digest,
            )
        if self.observation_std_dbz_digest is not None:
            _validate_sha256_digest(
                "observation_std_dbz_digest",
                self.observation_std_dbz_digest,
            )
        if self.background_frames_digest is not None:
            _validate_sha256_digest(
                "background_frames_digest",
                self.background_frames_digest,
            )
        if self.fixed_input_context_digest is not None:
            _validate_sha256_digest(
                "fixed_input_context_digest",
                self.fixed_input_context_digest,
            )
        if self.full_analysis_input_digest is not None:
            _validate_sha256_digest(
                "full_analysis_input_digest",
                self.full_analysis_input_digest,
            )
        full_context_values = (
            self.observation_masks_digest,
            self.observation_quality_weight_digest,
            self.observation_std_dbz_digest,
            self.fixed_input_context_digest,
            self.full_analysis_input_digest,
        )
        if any(value is None for value in full_context_values) and not all(
            value is None for value in full_context_values
        ):
            raise ValueError("full input-context digests must be recorded together")
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
        _validate_operational_calibration_lineage(
            self.config,
            self.analysis_config_json,
            self.grid_time_contract,
            self.operational_calibration_manifest_json,
            self.operational_calibration_manifest_digest,
            self.operational_calibration_approval_digest,
            self.operational_data_identity_json,
            self.operational_data_identity_digest,
            self.forecast_integrator_version,
        )
        _validate_neural_prior_lineage(
            self.neural_prior_digest,
            self.prior_application_digest,
            self.prior_model_contract_digest,
            self.prior_feature_schema_digest,
            self.prior_training_manifest_digest,
            self.prior_inference_evidence_digest,
            self.prior_inference_algorithm_digest,
            self.prior_numerical_runtime_digest,
            self.prior_dependency,
            self.prior_role,
            self.prior_lineage_contract,
        )
        _validate_prior_deployment_lineage(
            analysis_config_json=self.analysis_config_json,
            prior_role=self.prior_role,
            promotion_evidence_digest=self.prior_promotion_evidence_digest,
            regime_classification_evidence_digest=(
                self.prior_regime_classification_evidence_digest
            ),
            deployment_policy_digest=self.prior_deployment_policy_digest,
            deployment_policy_trust_store_digest=(
                self.prior_deployment_policy_trust_store_digest
            ),
            deployment_selection_digest=self.prior_deployment_selection_digest,
            fallback_reason=self.prior_deployment_fallback_reason,
            contract=self.prior_deployment_lineage_contract,
        )
        _validate_input_plan_lineage(self.input_plan_json, self.input_plan_digest)
        _validate_input_plan_resolution(
            self.input_plan_json,
            self.operational_data_identity_json,
            self.grid_time_contract,
        )
        expected_resolution_v2 = (
            None
            if self.input_plan_digest is None
            or self.full_analysis_input_digest is None
            else _forecast_input_plan_resolution_digest(
                input_plan_digest=self.input_plan_digest,
                full_analysis_input_digest=self.full_analysis_input_digest,
            )
        )
        legacy_resolution_v1 = (
            None
            if self.input_plan_digest is None
            else _legacy_forecast_input_plan_resolution_digest_v1(
                input_plan_digest=self.input_plan_digest,
                input_bundle_digest=self.input_bundle_digest,
            )
        )
        if self.input_plan_resolution_digest not in (
            expected_resolution_v2,
            legacy_resolution_v1,
        ):
            raise ValueError("input plan resolution digest mismatch")
        if self.observation_masks_digest is not None:
            assert self.observation_quality_weight_digest is not None
            assert self.observation_std_dbz_digest is not None
            expected_fixed_context = _forecast_fixed_input_context_digest(
                observation_masks_digest=self.observation_masks_digest,
                observation_quality_weight_digest=(
                    self.observation_quality_weight_digest
                ),
                observation_std_dbz_digest=self.observation_std_dbz_digest,
                background_frames_digest=self.background_frames_digest,
                background_age_minutes=self.background_age_minutes,
                grid_time_contract_digest=self.grid_time_contract_digest,
                operational_calibration_manifest_digest=(
                    self.operational_calibration_manifest_digest
                ),
                operational_calibration_approval_digest=(
                    self.operational_calibration_approval_digest
                ),
                operational_data_identity_digest=(
                    self.operational_data_identity_digest
                ),
                input_plan_digest=self.input_plan_digest,
            )
            if self.fixed_input_context_digest != expected_fixed_context:
                raise ValueError("fixed input context digest mismatch")
            expected_full_input = _forecast_full_analysis_input_digest(
                input_frames_digest=self.input_frames_digest,
                fixed_input_context_digest=expected_fixed_context,
            )
            if self.full_analysis_input_digest != expected_full_input:
                raise ValueError("full analysis input digest mismatch")

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
    operational_calibration_manifest_digest: str | None,
    operational_calibration_approval_digest: str | None,
    operational_data_identity_digest: str | None,
    *,
    grid_time_contract_digest: str | None = None,
) -> str:
    resolved_grid_digest = (
        grid_time_contract_digest
        if grid_time_contract is None
        else grid_time_contract.digest
    )
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
            "grid_time_contract_digest": resolved_grid_digest,
            "operational_calibration_manifest_digest": (
                operational_calibration_manifest_digest
            ),
            "operational_calibration_approval_digest": (
                operational_calibration_approval_digest
            ),
            "operational_data_identity_digest": (
                operational_data_identity_digest
            ),
        }
    )


def _forecast_full_analysis_input_digest(
    *,
    input_frames_digest: str,
    fixed_input_context_digest: str,
) -> str:
    """Address every data tensor and data-source contract for one analysis."""

    _validate_sha256_digest("input_frames_digest", input_frames_digest)
    _validate_sha256_digest(
        "fixed_input_context_digest",
        fixed_input_context_digest,
    )
    return json_digest(
        {
            "contract": "forecast-full-analysis-input-v1",
            "input_frames_digest": input_frames_digest,
            "fixed_input_context_digest": fixed_input_context_digest,
        }
    )


def _forecast_input_plan_resolution_digest(
    *,
    input_plan_digest: str,
    full_analysis_input_digest: str,
) -> str:
    """Resolve an input plan against the complete analysis data identity."""

    _validate_sha256_digest("input_plan_digest", input_plan_digest)
    _validate_sha256_digest(
        "full_analysis_input_digest",
        full_analysis_input_digest,
    )
    return json_digest(
        {
            "contract": "forecast-input-plan-resolution-v2",
            "input_plan_digest": input_plan_digest,
            "full_analysis_input_digest": full_analysis_input_digest,
        }
    )


def _legacy_forecast_input_plan_resolution_digest_v1(
    *,
    input_plan_digest: str,
    input_bundle_digest: str,
) -> str:
    """Reproduce a historical v1 resolution without promoting it to current."""

    return json_digest(
        {
            "contract": "forecast-input-plan-resolution-v1",
            "input_plan_digest": input_plan_digest,
            "input_bundle_digest": input_bundle_digest,
        }
    )


def _forecast_fixed_input_context_digest(
    *,
    observation_masks_digest: str,
    observation_quality_weight_digest: str,
    observation_std_dbz_digest: str,
    background_frames_digest: str | None,
    background_age_minutes: float | None,
    grid_time_contract_digest: str | None,
    operational_calibration_manifest_digest: str | None,
    operational_calibration_approval_digest: str | None,
    operational_data_identity_digest: str | None,
    input_plan_digest: str | None,
) -> str:
    """Address every non-radar input that a dBZ action must preserve."""

    return json_digest(
        {
            "contract": "forecast-fixed-input-context-v1",
            "observation_masks_digest": observation_masks_digest,
            "observation_quality_weight_digest": (
                observation_quality_weight_digest
            ),
            "observation_std_dbz_digest": observation_std_dbz_digest,
            "background_frames_digest": background_frames_digest,
            "background_age_minutes": background_age_minutes,
            "grid_time_contract_digest": grid_time_contract_digest,
            "operational_calibration_manifest_digest": (
                operational_calibration_manifest_digest
            ),
            "operational_calibration_approval_digest": (
                operational_calibration_approval_digest
            ),
            "operational_data_identity_digest": operational_data_identity_digest,
            "input_plan_digest": input_plan_digest,
        }
    )


def _validate_sha256_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_input_plan_lineage(
    input_plan_json: str | None,
    input_plan_digest: str | None,
) -> None:
    if input_plan_json is None and input_plan_digest is None:
        return
    if input_plan_json is None or input_plan_digest is None:
        raise ValueError("input plan JSON and digest must be provided together")
    _validate_sha256_digest("input_plan_digest", input_plan_digest)
    try:
        payload = json.loads(input_plan_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid input plan JSON") from error
    if not isinstance(payload, dict) or payload.get("contract") not in (
        "neural-prior-input-plan-v1",
        "neural-prior-input-plan-v2",
        "legacy-opaque-input-plan-v1",
    ):
        raise ValueError("unsupported input plan payload")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    expected = (
        payload.get("legacy_digest")
        if payload.get("contract") == "legacy-opaque-input-plan-v1"
        else json_digest(payload)
    )
    if canonical != input_plan_json or expected != input_plan_digest:
        raise ValueError("input plan payload digest mismatch")


def _validate_input_plan_resolution(
    input_plan_json: str | None,
    operational_data_identity_json: str | None,
    grid_time_contract: RadarGridTimeContract | None,
) -> None:
    """Bind a prospective selection plan to the actual run identity."""

    if input_plan_json is None:
        return
    payload = json.loads(input_plan_json)
    if payload["contract"] == "legacy-opaque-input-plan-v1":
        return
    if operational_data_identity_json is None or grid_time_contract is None:
        raise ValueError("input-plan runs require data and grid identity")
    identity = OperationalDataIdentity.from_json(operational_data_identity_json)
    expected = (
        ("radar_product_digest", identity.radar_product_digest),
        ("qc_pipeline_digest", identity.qc_pipeline_digest),
        (
            "background_cycle_rule_digest",
            identity.background_cycle_rule_digest,
        ),
        ("mask_policy_digest", identity.mask_policy_digest),
    )
    if any(value is None or payload[name] != value for name, value in expected):
        raise ValueError("input plan disagrees with operational data identity")
    valid_time = (
        payload["observation_valid_time"]
        if payload["contract"] == "neural-prior-input-plan-v2"
        else payload["issue_time"]
    )
    if (
        payload["grid_contract_digest"] != grid_time_contract.digest
        or tuple(payload["valid_times"]) != grid_time_contract.valid_times
        or valid_time != grid_time_contract.valid_times[-1]
    ):
        raise ValueError("input plan disagrees with the run grid or times")


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


def _validate_neural_prior_lineage(
    prior_digest: str | None,
    application_digest: str | None,
    model_contract_digest: str | None,
    feature_schema_digest: str | None,
    training_manifest_digest: str | None,
    inference_evidence_digest: str | None,
    inference_algorithm_digest: str | None,
    numerical_runtime_digest: str | None,
    dependency: str | None,
    role: str | None,
    contract: str,
) -> None:
    if contract not in (
        "neural-prior-run-lineage-v1-audit",
        "neural-prior-run-lineage-v2",
    ):
        raise ValueError("unsupported neural-prior run lineage")
    values = (
        prior_digest,
        application_digest,
        model_contract_digest,
        feature_schema_digest,
        training_manifest_digest,
        inference_evidence_digest,
        inference_algorithm_digest,
        numerical_runtime_digest,
        dependency,
        role,
    )
    if all(value is None for value in values):
        if contract != "neural-prior-run-lineage-v2":
            raise ValueError("legacy neural-prior lineage requires prior identity")
        return
    if contract == "neural-prior-run-lineage-v1-audit":
        legacy = (
            prior_digest,
            application_digest,
            model_contract_digest,
            feature_schema_digest,
            training_manifest_digest,
            role,
        )
        modern = (
            inference_evidence_digest,
            inference_algorithm_digest,
            numerical_runtime_digest,
            dependency,
        )
        if any(value is None for value in legacy) or any(
            value is not None for value in modern
        ):
            raise ValueError("legacy neural-prior lineage is malformed")
        for name, digest in (
            ("neural_prior_digest", prior_digest),
            ("prior_application_digest", application_digest),
            ("prior_model_contract_digest", model_contract_digest),
            ("prior_feature_schema_digest", feature_schema_digest),
            ("prior_training_manifest_digest", training_manifest_digest),
        ):
            assert digest is not None
            _validate_sha256_digest(name, digest)
        if role not in ("candidate", "parent"):
            raise ValueError("prior_role must be candidate or parent")
        return
    if any(value is None for value in values):
        raise ValueError("neural-prior run lineage must be complete")
    assert prior_digest is not None
    assert application_digest is not None
    assert model_contract_digest is not None
    assert feature_schema_digest is not None
    assert training_manifest_digest is not None
    assert inference_evidence_digest is not None
    assert inference_algorithm_digest is not None
    assert numerical_runtime_digest is not None
    assert dependency is not None
    assert role is not None
    for name, digest in (
        ("neural_prior_digest", prior_digest),
        ("prior_application_digest", application_digest),
        ("prior_model_contract_digest", model_contract_digest),
        ("prior_feature_schema_digest", feature_schema_digest),
        ("prior_training_manifest_digest", training_manifest_digest),
        ("prior_inference_evidence_digest", inference_evidence_digest),
        ("prior_inference_algorithm_digest", inference_algorithm_digest),
        ("prior_numerical_runtime_digest", numerical_runtime_digest),
    ):
        _validate_sha256_digest(name, digest)
    if dependency not in ("exogenous", "radar_dependent"):
        raise ValueError("unsupported prior_dependency")
    if role not in ("candidate", "parent"):
        raise ValueError("prior_role must be candidate or parent")


def _validate_prior_deployment_lineage(
    *,
    analysis_config_json: str | None,
    prior_role: str | None,
    promotion_evidence_digest: str | None,
    regime_classification_evidence_digest: str | None,
    deployment_policy_digest: str | None,
    deployment_policy_trust_store_digest: str | None,
    deployment_selection_digest: str | None,
    fallback_reason: str | None,
    contract: str,
) -> None:
    if contract not in {
        "neural-prior-deployment-lineage-v0-audit",
        "neural-prior-deployment-lineage-v1-audit",
        "neural-prior-deployment-lineage-v2",
    }:
        raise ValueError("unsupported neural-prior deployment lineage")
    values = (
        promotion_evidence_digest,
        regime_classification_evidence_digest,
        deployment_policy_digest,
        deployment_policy_trust_store_digest,
        deployment_selection_digest,
        fallback_reason,
    )
    if contract == "neural-prior-deployment-lineage-v0-audit":
        if any(value is not None for value in values):
            raise ValueError("legacy deployment lineage cannot claim evidence")
        return
    if contract == "neural-prior-deployment-lineage-v1-audit":
        legacy_values = (
            promotion_evidence_digest,
            regime_classification_evidence_digest,
            deployment_selection_digest,
            fallback_reason,
        )
        if all(value is None for value in legacy_values):
            return
        if any(value is None for value in legacy_values) or prior_role is None:
            raise ValueError("legacy deployment lineage is incomplete")
        for name, digest in (
            ("prior_promotion_evidence_digest", promotion_evidence_digest),
            (
                "prior_regime_classification_evidence_digest",
                regime_classification_evidence_digest,
            ),
            ("prior_deployment_selection_digest", deployment_selection_digest),
        ):
            assert digest is not None
            _validate_sha256_digest(name, digest)
        return
    if all(value is None for value in values):
        if analysis_config_json is not None and prior_role is not None:
            analysis = json.loads(analysis_config_json)
            if (
                isinstance(analysis, dict)
                and analysis.get("execution_mode") == "operational"
            ):
                raise ValueError(
                    "operational neural prior requires deployment lineage"
                )
        return
    if contract != "neural-prior-deployment-lineage-v2":
        raise ValueError("legacy deployment lineage is audit-only")
    if any(value is None for value in values) or prior_role is None:
        raise ValueError("neural-prior deployment lineage must be complete")
    assert promotion_evidence_digest is not None
    assert regime_classification_evidence_digest is not None
    assert deployment_policy_digest is not None
    assert deployment_policy_trust_store_digest is not None
    assert deployment_selection_digest is not None
    assert fallback_reason is not None
    for name, digest in (
        ("prior_promotion_evidence_digest", promotion_evidence_digest),
        (
            "prior_regime_classification_evidence_digest",
            regime_classification_evidence_digest,
        ),
        ("prior_deployment_policy_digest", deployment_policy_digest),
        (
            "prior_deployment_policy_trust_store_digest",
            deployment_policy_trust_store_digest,
        ),
        ("prior_deployment_selection_digest", deployment_selection_digest),
    ):
        _validate_sha256_digest(name, digest)
    if fallback_reason not in {
        "certified_candidate",
        "uncertified_regime",
        "uncertified_range_band",
        "low_regime_confidence",
        "ood_or_abstained",
        "promotion_ineligible",
        "no_certified_regime",
    } or (
        (prior_role == "candidate")
        != (fallback_reason == "certified_candidate")
    ):
        raise ValueError("neural-prior deployment role and fallback disagree")


def operational_runtime_profile_digest(
    config: NowcastConfig,
    grid: RadarGridTimeContract,
    *,
    analysis_config: Mapping[str, object] | None = None,
    forecast_integrator_version: str = FORECAST_INTEGRATOR_VERSION,
) -> str:
    config_value = None
    profile_kind = "p0"
    nowcast_value = asdict(config)
    if analysis_config is not None:
        config_value = dict(analysis_config)
        if config_value.get("execution_mode") != "operational":
            raise ValueError("operational P1 profile requires operational analysis")
        calibration_id = config_value.pop("operational_calibration_id", None)
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ValueError(
                "operational P1 profile requires a calibration identifier"
            )
        profile_kind = "p1"
    else:
        for name in (
            "p1_motion_saturation_safe_margin_mps",
            "p1_growth_saturation_safe_margin_per_step",
            "p1_posterior_saturation_sigma_multiplier",
            "p1_saturation_uncertainty_multiplier",
        ):
            nowcast_value.pop(name)
    return json_digest(
        {
            "version": _OPERATIONAL_RUNTIME_PROFILE_VERSION,
            "profile_kind": profile_kind,
            "forecast_integrator_version": forecast_integrator_version,
            "nowcast_config": nowcast_value,
            "analysis_config": config_value,
            "grid": {
                "dx_m": grid.dx_m,
                "dy_m": grid.dy_m,
                "projection": grid.projection,
                "grid_hash": grid.grid_hash,
                "pixel_to_projected_matrix_m": (
                    grid.pixel_to_projected_matrix_m
                ),
            },
        }
    )


def _validate_operational_calibration_lineage(
    config: NowcastConfig,
    analysis_config_json: str | None,
    grid: RadarGridTimeContract | None,
    manifest_json: str | None,
    manifest_digest: str | None,
    approval_digest: str | None,
    data_identity_json: str | None,
    data_identity_digest: str | None,
    forecast_integrator_version: str,
) -> None:
    lineage = (
        manifest_json,
        manifest_digest,
        approval_digest,
        data_identity_json,
        data_identity_digest,
    )
    config_value = None
    if analysis_config_json is not None:
        loaded = json.loads(analysis_config_json)
        if not isinstance(loaded, dict):
            raise ValueError("analysis_config_json must contain an object")
        config_value = loaded
        if loaded.get("execution_mode") != "operational":
            if any(value is not None for value in lineage):
                raise ValueError(
                    "calibration manifest requires operational analysis"
                )
            return
    if all(value is None for value in lineage):
        if config_value is not None:
            raise ValueError("operational analysis requires a calibration manifest")
        return
    if any(value is None for value in lineage):
        raise ValueError(
            "operational calibration manifest, approval, and data identity "
            "must be provided together"
        )
    assert manifest_json is not None
    assert manifest_digest is not None
    assert approval_digest is not None
    assert data_identity_json is not None
    assert data_identity_digest is not None
    if grid is None:
        raise ValueError("operational calibration requires a grid/time contract")
    manifest = OperationalCalibrationManifest.from_json(manifest_json)
    _validate_sha256_digest(
        "operational_calibration_manifest_digest",
        manifest_digest,
    )
    _validate_sha256_digest(
        "operational_calibration_approval_digest",
        approval_digest,
    )
    if manifest.json != manifest_json or manifest.digest != manifest_digest:
        raise ValueError("operational calibration manifest digest mismatch")
    if approval_digest != manifest.digest:
        raise ValueError("operational calibration manifest is not approved")
    _validate_sha256_digest(
        "operational_data_identity_digest",
        data_identity_digest,
    )
    data_identity = OperationalDataIdentity.from_json(data_identity_json)
    if data_identity.digest != data_identity_digest:
        raise ValueError("operational data identity digest mismatch")
    if data_identity != manifest.data_identity:
        raise ValueError("operational data identity is not calibrated")
    expected_kind = "p0" if config_value is None else "p1"
    if manifest.profile_kind != expected_kind:
        raise ValueError("calibration manifest profile kind mismatch")
    if config_value is not None and (
        manifest.calibration_id
        != config_value.get("operational_calibration_id")
    ):
        raise ValueError("calibration manifest identifier mismatch")
    profile_digest = operational_runtime_profile_digest(
        config,
        grid,
        analysis_config=config_value,
        forecast_integrator_version=forecast_integrator_version,
    )
    if manifest.expected_runtime_profile_digest != profile_digest:
        raise ValueError("calibration manifest runtime profile digest mismatch")
    if manifest.expected_algorithm_bundle_digest != algorithm_bundle_digest():
        raise ValueError("calibration manifest algorithm bundle digest mismatch")


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
            "input_frames_digest": run.input_frames_digest,
            "observation_masks_digest": run.observation_masks_digest,
            "observation_quality_weight_digest": (
                run.observation_quality_weight_digest
            ),
            "observation_std_dbz_digest": run.observation_std_dbz_digest,
            "background_frames_digest": run.background_frames_digest,
            "fixed_input_context_digest": run.fixed_input_context_digest,
            "full_analysis_input_digest": run.full_analysis_input_digest,
            "latest_frame_digest": run.latest_frame_digest,
            "latest_observation_mask_digest": (
                run.latest_observation_mask_digest
            ),
            "latest_background_digest": run.latest_background_digest,
            "background_age_minutes": run.background_age_minutes,
            "grid_time_contract_digest": run.grid_time_contract_digest,
            "analysis_config_digest": run.analysis_config_digest,
            "analysis_input_digest": run.analysis_input_digest,
            "operational_calibration_manifest_digest": (
                run.operational_calibration_manifest_digest
            ),
            "operational_calibration_approval_digest": (
                run.operational_calibration_approval_digest
            ),
            "operational_data_identity_digest": (
                run.operational_data_identity_digest
            ),
            "neural_prior_digest": run.neural_prior_digest,
            "prior_application_digest": run.prior_application_digest,
            "prior_model_contract_digest": run.prior_model_contract_digest,
            "prior_feature_schema_digest": run.prior_feature_schema_digest,
            "prior_training_manifest_digest": (
                run.prior_training_manifest_digest
            ),
            "prior_inference_evidence_digest": (
                run.prior_inference_evidence_digest
            ),
            "prior_inference_algorithm_digest": (
                run.prior_inference_algorithm_digest
            ),
            "prior_numerical_runtime_digest": (
                run.prior_numerical_runtime_digest
            ),
            "prior_dependency": run.prior_dependency,
            "prior_role": run.prior_role,
            "prior_promotion_evidence_digest": (
                run.prior_promotion_evidence_digest
            ),
            "prior_regime_classification_evidence_digest": (
                run.prior_regime_classification_evidence_digest
            ),
            "prior_deployment_policy_digest": (
                run.prior_deployment_policy_digest
            ),
            "prior_deployment_policy_trust_store_digest": (
                run.prior_deployment_policy_trust_store_digest
            ),
            "prior_deployment_selection_digest": (
                run.prior_deployment_selection_digest
            ),
            "prior_deployment_fallback_reason": (
                run.prior_deployment_fallback_reason
            ),
            "prior_deployment_lineage_contract": (
                run.prior_deployment_lineage_contract
            ),
            "prior_lineage_contract": run.prior_lineage_contract,
            "input_plan_digest": run.input_plan_digest,
            "input_plan_resolution_digest": run.input_plan_resolution_digest,
            "state_metadata_digest": state_digest,
            "forecast_dbz_digest": forecast_digest,
            "valid_mask_digest": valid_mask_digest,
        }
    )


def _validate_state_for_run(
    state: RadarState,
    run: ForecastRunContract,
) -> None:
    if (
        state.echo_linear.ndim != 2
        or state.echo_linear.shape != run._latest_frame_dbz.shape
    ):
        raise ValueError("state grid must match the run input grid")
    _validate_state_dynamics(
        state,
        run.config,
        run.grid_time_contract,
    )


@dataclass(frozen=True)
class ForecastEvidenceFields:
    source_support: Tensor
    path_verified_support: Tensor
    verified_support: Tensor
    local_motion_verified_support: Tensor
    local_growth_verified_support: Tensor
    local_dynamics_verified_support: Tensor
    observation_source_support: Tensor
    background_source_support: Tensor
    observation_verified_support: Tensor
    background_verified_support: Tensor
    velocity_uncertainty_mps: Tensor
    motion_evidence_uncertainty_multiplier: Tensor
    growth_evidence_uncertainty_multiplier: Tensor
    position_uncertainty_m: Tensor
    log_growth_uncertainty: Tensor
    confidence: Tensor


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
    evidence: ForecastEvidenceFields
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

    @property
    def forecast_verified_support(self) -> Tensor:
        return self.evidence.verified_support

    @property
    def forecast_local_dynamics_verified_support(self) -> Tensor:
        return self.evidence.local_dynamics_verified_support

    @property
    def forecast_local_motion_verified_support(self) -> Tensor:
        return self.evidence.local_motion_verified_support

    @property
    def forecast_local_growth_verified_support(self) -> Tensor:
        return self.evidence.local_growth_verified_support

    @property
    def forecast_observation_verified_support(self) -> Tensor:
        return self.evidence.observation_verified_support

    @property
    def forecast_background_verified_support(self) -> Tensor:
        return self.evidence.background_verified_support

    @property
    def forecast_velocity_uncertainty_mps(self) -> Tensor:
        return self.evidence.velocity_uncertainty_mps

    @property
    def motion_evidence_uncertainty_multiplier(self) -> Tensor:
        return self.evidence.motion_evidence_uncertainty_multiplier

    @property
    def growth_evidence_uncertainty_multiplier(self) -> Tensor:
        return self.evidence.growth_evidence_uncertainty_multiplier

    @property
    def forecast_position_uncertainty_m(self) -> Tensor:
        return self.evidence.position_uncertainty_m

    @property
    def forecast_log_growth_uncertainty(self) -> Tensor:
        return self.evidence.log_growth_uncertainty

    @property
    def forecast_confidence(self) -> Tensor:
        return self.evidence.confidence

    @property
    def radar_anchored_valid_mask(self) -> Tensor:
        return self.radar_state_anchored_valid_mask

    @property
    def radar_state_anchored_valid_mask(self) -> Tensor:
        threshold = (
            self.run.config.minimum_publish_observation_verified_support
        )
        if threshold is None:
            threshold = self.run.config.support_presence_threshold
        return self.valid_mask & (
            self.forecast_observation_verified_support >= threshold
        )

    @property
    def radar_dynamics_anchored_valid_mask(self) -> Tensor:
        if self.metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL:
            if not _p1_posterior_is_available(self.metadata):
                return torch.zeros_like(self.valid_mask)
            threshold = self.run.config.minimum_publish_verified_support
            if threshold is None:
                threshold = self.run.config.support_presence_threshold
            return self.radar_state_anchored_valid_mask & (
                self.forecast_local_dynamics_verified_support >= threshold
            )
        if (
            self.metadata.tendency_source is not TendencySource.OBSERVATION
            or self.metadata.motion_pair_count == 0
            or self.metadata.growth_pair_count == 0
        ):
            return torch.zeros_like(self.valid_mask)
        threshold = self.run.config.minimum_publish_verified_support
        if threshold is None:
            threshold = self.run.config.support_presence_threshold
        return self.radar_state_anchored_valid_mask & (
            self.forecast_local_dynamics_verified_support >= threshold
        )

    @property
    def background_dynamics_mask(self) -> Tensor:
        if self.metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL:
            return torch.zeros_like(self.valid_mask)
        if self.metadata.tendency_source is not TendencySource.BACKGROUND:
            return torch.zeros_like(self.valid_mask)
        threshold = self.run.config.minimum_publish_verified_support
        if threshold is None:
            threshold = self.run.config.support_presence_threshold
        return self.valid_mask & (
            self.forecast_local_dynamics_verified_support >= threshold
        )

    @property
    def background_fallback_mask(self) -> Tensor:
        return self.valid_mask & ~self.radar_anchored_valid_mask

    @property
    def forecast_path_verified_support(self) -> Tensor:
        return self.evidence.path_verified_support

    @property
    def forecast_source_support(self) -> Tensor:
        return self.evidence.source_support

    @property
    def forecast_observation_source_support(self) -> Tensor:
        return self.evidence.observation_source_support

    @property
    def forecast_background_source_support(self) -> Tensor:
        return self.evidence.background_source_support

    def validate_issuance(self) -> None:
        if tensor_digest(self.forecast_dbz) != self.forecast_dbz_digest:
            raise ValueError("forecast result disagrees with the issued forecast")
        if tensor_digest(self.valid_mask) != self.valid_mask_digest:
            raise ValueError(
                "forecast valid mask disagrees with the issued forecast"
            )
        self.run.validate_integrity()
        _validate_forecast_contract(self)
        if (
            state_metadata_digest(self.state, self.metadata)
            != self.state_metadata_digest
        ):
            raise ValueError(
                "forecast state or metadata disagrees with the issued forecast"
            )
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


def _validate_forecast_contract(result: ForecastResult) -> None:
    forecast = result.forecast_dbz
    valid = result.valid_mask
    state = result.state
    metadata = result.metadata
    config = result.run.config
    floating = (torch.float32, torch.float64)
    float_tensors = (
        forecast,
        state.echo_linear,
        state.displacement_yx,
        state.log_growth_per_step,
        metadata.coverage_by_frame,
        metadata.source_support,
        metadata.observation_source_support,
        metadata.background_source_support,
        metadata.path_verified_source_support,
        metadata.verified_source_support,
        metadata.local_motion_verified_support,
        metadata.local_growth_verified_support,
        metadata.local_dynamics_verified_support,
        metadata.observation_verified_source_support,
        metadata.background_verified_source_support,
        metadata.motion_disagreement_px,
        metadata.motion_disagreement_mps,
        metadata.growth_disagreement,
    )
    if any(value.dtype not in floating for value in float_tensors):
        raise ValueError("forecast run tensors must use float32 or float64")
    state_tensors = (
        forecast,
        state.echo_linear,
        state.displacement_yx,
        state.log_growth_per_step,
        metadata.source_support,
        metadata.observation_source_support,
        metadata.background_source_support,
        metadata.path_verified_source_support,
        metadata.verified_source_support,
        metadata.local_motion_verified_support,
        metadata.local_growth_verified_support,
        metadata.local_dynamics_verified_support,
        metadata.observation_verified_source_support,
        metadata.background_verified_source_support,
    )
    if len({value.dtype for value in state_tensors}) != 1:
        raise ValueError("forecast run state tensors must share one dtype")
    if forecast.ndim != 3 or forecast.shape[0] != config.forecast_steps:
        raise ValueError("forecast_dbz has the wrong lead shape")
    if valid.dtype != torch.bool or valid.shape != forecast.shape:
        raise ValueError("valid_mask must be boolean with the forecast shape")
    _validate_state_for_run(state, result.run)
    if forecast.shape[1:] != state.echo_linear.shape:
        raise ValueError("forecast and state grids disagree")
    if metadata.coverage_by_frame.shape != (3,):
        raise ValueError("coverage_by_frame must have shape [3]")
    if metadata.source_support.shape != state.echo_linear.shape:
        raise ValueError("source_support must match the state grid")
    if metadata.observation_source_support.shape != state.echo_linear.shape:
        raise ValueError("observation_source_support must match the state grid")
    if metadata.background_source_support.shape != state.echo_linear.shape:
        raise ValueError("background_source_support must match the state grid")
    if metadata.path_verified_source_support.shape != state.echo_linear.shape:
        raise ValueError(
            "path_verified_source_support must match the state grid"
        )
    if metadata.verified_source_support.shape != state.echo_linear.shape:
        raise ValueError("verified_source_support must match the state grid")
    if metadata.local_motion_verified_support.shape != state.echo_linear.shape:
        raise ValueError(
            "local_motion_verified_support must match the state grid"
        )
    if metadata.local_growth_verified_support.shape != state.echo_linear.shape:
        raise ValueError(
            "local_growth_verified_support must match the state grid"
        )
    if metadata.local_dynamics_verified_support.shape != state.echo_linear.shape:
        raise ValueError(
            "local_dynamics_verified_support must match the state grid"
        )
    if (
        metadata.observation_verified_source_support.shape
        != state.echo_linear.shape
    ):
        raise ValueError(
            "observation_verified_source_support must match the state grid"
        )
    if (
        metadata.background_verified_source_support.shape
        != state.echo_linear.shape
    ):
        raise ValueError(
            "background_verified_source_support must match the state grid"
        )
    if metadata.motion_disagreement_px.ndim != 0:
        raise ValueError("motion_disagreement_px must be scalar")
    if metadata.motion_disagreement_mps.ndim != 0:
        raise ValueError("motion_disagreement_mps must be scalar")
    if metadata.growth_disagreement.ndim != 0:
        raise ValueError("growth_disagreement must be scalar")
    if metadata.maximum_growth_saturation_excess.ndim != 0:
        raise ValueError("maximum_growth_saturation_excess must be scalar")
    if metadata.posterior_velocity_uncertainty_mps.ndim != 0:
        raise ValueError("posterior_velocity_uncertainty_mps must be scalar")
    if metadata.posterior_log_growth_uncertainty_per_step.ndim != 0:
        raise ValueError(
            "posterior_log_growth_uncertainty_per_step must be scalar"
        )
    if metadata.p1_velocity_saturation_uncertainty_mps.ndim != 0:
        raise ValueError(
            "p1_velocity_saturation_uncertainty_mps must be scalar"
        )
    if metadata.p1_log_growth_saturation_uncertainty_per_step.ndim != 0:
        raise ValueError(
            "p1_log_growth_saturation_uncertainty_per_step must be scalar"
        )
    selection_counts = {
        TendencyPairSelection.NONE: 0,
        TendencyPairSelection.PERSISTENCE: 0,
        TendencyPairSelection.SINGLE: 1,
        TendencyPairSelection.LONG: 1,
        TendencyPairSelection.EARLIER: 1,
        TendencyPairSelection.RECENT: 1,
        TendencyPairSelection.BLENDED: 2,
    }
    if metadata.motion_pair_count != selection_counts[
        metadata.motion_pair_selection
    ]:
        raise ValueError("motion pair count and selection disagree")
    if metadata.growth_pair_count != selection_counts[
        metadata.growth_pair_selection
    ]:
        raise ValueError("growth pair count and selection disagree")
    if metadata.state_path_pair_count != selection_counts[
        metadata.state_path_mode
    ]:
        raise ValueError("state path pair count and selection disagree")
    _validate_state_path_provenance(
        "observation path",
        metadata.observation_path,
        selection_counts,
    )
    _validate_state_path_provenance(
        "background path",
        metadata.background_path,
        selection_counts,
    )
    if (
        metadata.motion_pair_selection is TendencyPairSelection.PERSISTENCE
        and not metadata.motion_pair_conflict
    ) or (
        metadata.motion_pair_selection is TendencyPairSelection.BLENDED
        and metadata.motion_pair_conflict
    ):
        raise ValueError("motion pair conflict provenance is inconsistent")
    if (
        metadata.growth_pair_selection is TendencyPairSelection.PERSISTENCE
        and not metadata.growth_pair_conflict
    ) or (
        metadata.growth_pair_selection is TendencyPairSelection.BLENDED
        and metadata.growth_pair_conflict
    ):
        raise ValueError("growth pair conflict provenance is inconsistent")
    selections = (
        metadata.motion_pair_selection,
        metadata.growth_pair_selection,
    )
    pair_sources: dict[TendencyPairSelection, frozenset[str]] = {
        TendencyPairSelection.NONE: frozenset(),
        TendencyPairSelection.PERSISTENCE: frozenset(),
        TendencyPairSelection.SINGLE: frozenset(("single_adjacent",)),
        TendencyPairSelection.LONG: frozenset(("long",)),
        TendencyPairSelection.EARLIER: frozenset(("earlier",)),
        TendencyPairSelection.RECENT: frozenset(("recent",)),
        TendencyPairSelection.BLENDED: frozenset(("earlier", "recent")),
    }
    used_sources = pair_sources[selections[0]] | pair_sources[selections[1]]
    if metadata.tendency_pair_count != len(used_sources):
        raise ValueError("tendency_pair_count is inconsistent")
    minimum_psr = float(metadata.minimum_phase_correlation_psr)
    if metadata.tendency_pair_count == 0:
        if not math.isnan(minimum_psr):
            raise ValueError("unused tendency pairs must have NaN PSR")
    elif not math.isfinite(minimum_psr):
        raise ValueError("used tendency pairs must have finite PSR")
    state_path_psr = metadata.state_path_minimum_psr
    if metadata.state_path_pair_count == 0:
        if not math.isnan(state_path_psr):
            raise ValueError("unused state paths must have NaN PSR")
    elif not math.isfinite(state_path_psr):
        raise ValueError("used state paths must have finite PSR")
    if metadata.state_path_age_minutes is not None and (
        not math.isfinite(metadata.state_path_age_minutes)
        or metadata.state_path_age_minutes < 0
    ):
        raise ValueError("state path age must be finite and nonnegative")
    expected_provenance = {
        DynamicsSource.P0_RECONSTRUCTION: "p0_support_merged",
        DynamicsSource.P0_FALLBACK: "p0_support_merged",
        DynamicsSource.P1_VARIATIONAL: "p1_variational_analysis",
    }[metadata.dynamics_source]
    if metadata.provenance != expected_provenance:
        raise ValueError("dynamics source and provenance disagree")
    if metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL and (
        metadata.state_path_source is not TendencySource.NONE
        or metadata.state_path_mode is not TendencyPairSelection.NONE
        or metadata.state_path_pair_count != 0
        or not math.isnan(metadata.state_path_minimum_psr)
        or metadata.state_path_conflict
        or metadata.state_path_extrapolated
        or metadata.state_path_age_minutes is not None
        or not _path_is_empty(metadata.observation_path)
        or not _path_is_empty(metadata.background_path)
        or not math.isnan(metadata.minimum_growth_overlap_support)
        or not math.isnan(metadata.minimum_growth_overlap_area_km2)
    ):
        raise ValueError("P1 metadata cannot retain P0 path evidence")
    p1_uncertainty_values = (
        metadata.posterior_velocity_uncertainty_mps,
        metadata.posterior_log_growth_uncertainty_per_step,
        metadata.p1_velocity_saturation_uncertainty_mps,
        metadata.p1_log_growth_saturation_uncertainty_per_step,
    )
    if any(bool(torch.isinf(value)) for value in p1_uncertainty_values):
        raise ValueError("P1 uncertainties cannot be infinite")
    p1_uncertainty_available = tuple(
        bool(torch.isfinite(value)) for value in p1_uncertainty_values
    )
    if metadata.dynamics_source is not DynamicsSource.P1_VARIATIONAL:
        if not all(
            bool(torch.isnan(value)) for value in p1_uncertainty_values
        ):
            raise ValueError("P0 metadata cannot contain P1 uncertainty")
    else:
        if len(set(p1_uncertainty_available)) != 1:
            raise ValueError("P1 uncertainties must be jointly available")
        if all(p1_uncertainty_available) and any(
            float(value) < 0.0 for value in p1_uncertainty_values
        ):
            raise ValueError("P1 uncertainties cannot be negative")
    background_fraction = metadata.background_contribution_fraction
    if (
        not math.isfinite(background_fraction)
        or background_fraction < 0
        or background_fraction > 1
    ):
        raise ValueError("background contribution fraction must be in [0, 1]")
    expected_background_fraction = float(
        metadata.background_source_support.sum()
        / metadata.source_support.sum().clamp_min(
            config.ratio_regularizer
        )
    )
    if not math.isclose(
        background_fraction,
        expected_background_fraction,
        rel_tol=0.0,
        abs_tol=config.contract_absolute_tolerance,
    ):
        raise ValueError("background contribution fraction mismatch")
    if metadata.dynamics_source is not DynamicsSource.P1_VARIATIONAL:
        observation_path_used = _path_has_contribution(
            metadata.observation_path
        )
        background_path_used = _path_has_contribution(
            metadata.background_path
        )
        actual_observation_used = bool(
            torch.any(
                metadata.observation_source_support
                > config.support_presence_threshold
            )
        )
        actual_background_used = bool(
            torch.any(
                metadata.background_source_support
                > config.support_presence_threshold
            )
        )
        if (
            observation_path_used != actual_observation_used
            or background_path_used != actual_background_used
        ):
            raise ValueError("source path and actual contribution disagree")
        if observation_path_used:
            expected_path_source = TendencySource.OBSERVATION
            expected_path = metadata.observation_path
        elif background_path_used:
            expected_path_source = TendencySource.BACKGROUND
            expected_path = metadata.background_path
        else:
            expected_path_source = TendencySource.NONE
            expected_path = StatePathProvenance()
        if (
            metadata.state_path_source is not expected_path_source
            or not _aggregate_path_matches(metadata, expected_path)
        ):
            raise ValueError("aggregate state path provenance mismatch")
    background_state_used = bool(
        torch.any(
            metadata.background_source_support
            > config.support_presence_threshold
        )
    )
    expected_background_used = (
        background_state_used or metadata.background_tendency_used
    )
    if metadata.background_used != expected_background_used:
        raise ValueError("background usage provenance mismatch")
    background_age = metadata.background_age_minutes
    if metadata.background_used != (
        background_age is not None
    ):
        raise ValueError("background age provenance mismatch")
    if metadata.background_used:
        run_background_age = result.run.background_age_minutes
        if (
            background_age is None
            or run_background_age is None
            or not math.isclose(
                background_age,
                run_background_age,
                rel_tol=0.0,
                abs_tol=config.contract_absolute_tolerance,
            )
        ):
            raise ValueError("background age disagrees with the forecast run")
    growth_support = metadata.minimum_growth_overlap_support
    growth_area = metadata.minimum_growth_overlap_area_km2
    if metadata.dynamics_source is not DynamicsSource.P1_VARIATIONAL:
        if metadata.growth_pair_count == 0:
            if not math.isnan(growth_support) or not math.isnan(growth_area):
                raise ValueError("unused growth pairs cannot retain evidence")
        else:
            if (
                not math.isfinite(growth_support)
                or growth_support + config.contract_absolute_tolerance
                < config.minimum_growth_overlap_support
            ):
                raise ValueError("used growth pairs require valid evidence")
            if result.run.grid_time_contract is None:
                if not math.isnan(growth_area):
                    raise ValueError(
                        "growth overlap area requires a grid/time contract"
                    )
            elif (
                not math.isfinite(growth_area)
                or growth_area <= 0
                or (
                    config.minimum_growth_overlap_area_km2 is not None
                    and growth_area + config.contract_absolute_tolerance
                    < config.minimum_growth_overlap_area_km2
                )
            ):
                raise ValueError("used growth pairs require valid area evidence")
    if not torch.equal(torch.isfinite(forecast), valid):
        raise ValueError("valid_mask must match finite forecast values")
    finite_tensors = (
        state.echo_linear,
        state.displacement_yx,
        state.log_growth_per_step,
        metadata.coverage_by_frame,
        metadata.source_support,
        metadata.observation_source_support,
        metadata.background_source_support,
        metadata.path_verified_source_support,
        metadata.verified_source_support,
        metadata.local_motion_verified_support,
        metadata.local_growth_verified_support,
        metadata.local_dynamics_verified_support,
        metadata.observation_verified_source_support,
        metadata.background_verified_source_support,
        metadata.motion_disagreement_px,
        metadata.growth_disagreement,
        metadata.maximum_growth_saturation_excess,
    )
    if not all(
        bool(torch.all(torch.isfinite(value))) for value in finite_tensors
    ):
        raise ValueError("forecast run state and metadata must be finite")
    if float(metadata.maximum_growth_saturation_excess) < 0.0:
        raise ValueError("maximum_growth_saturation_excess cannot be negative")
    if math.isinf(float(metadata.motion_disagreement_mps)):
        raise ValueError("motion_disagreement_mps cannot be infinite")
    if bool(torch.any(state.echo_linear < 0)):
        raise ValueError("state_echo_linear cannot be negative")
    if not bool(
        torch.all(
            (metadata.coverage_by_frame >= 0)
            & (metadata.coverage_by_frame <= 1)
        )
    ):
        raise ValueError("coverage_by_frame must be in [0, 1]")
    if not bool(
        torch.all(
            (metadata.source_support >= 0)
            & (metadata.source_support <= 1)
        )
    ):
        raise ValueError("source_support must be in [0, 1]")
    if not bool(
        torch.all(
            (metadata.observation_source_support >= 0)
            & (metadata.background_source_support >= 0)
            & (
                metadata.observation_source_support
                + metadata.background_source_support
                <= 1.0 + config.contract_absolute_tolerance
            )
        )
    ):
        raise ValueError("source-specific support must be in [0, 1]")
    combined_source_support = (
        metadata.observation_source_support
        + metadata.background_source_support
    ).clamp(0.0, 1.0)
    if not bool(
        torch.allclose(
            metadata.source_support,
            combined_source_support,
            rtol=0.0,
            atol=config.contract_absolute_tolerance,
        )
    ):
        raise ValueError("source support and actual contributions disagree")
    if not bool(
        torch.all(
            (metadata.path_verified_source_support >= 0)
            & (
                metadata.path_verified_source_support
                <= metadata.source_support
                + config.contract_absolute_tolerance
            )
            & (metadata.verified_source_support >= 0)
            & (
                metadata.verified_source_support
                <= metadata.path_verified_source_support
            )
            & (metadata.local_dynamics_verified_support >= 0)
            & (metadata.local_motion_verified_support >= 0)
            & (
                metadata.local_motion_verified_support
                <= metadata.verified_source_support
                + config.contract_absolute_tolerance
            )
            & (metadata.local_growth_verified_support >= 0)
            & (
                metadata.local_growth_verified_support
                <= metadata.verified_source_support
                + config.contract_absolute_tolerance
            )
            & (
                metadata.local_dynamics_verified_support
                <= metadata.verified_source_support
                + config.contract_absolute_tolerance
            )
            & (metadata.observation_verified_source_support >= 0)
            & (metadata.background_verified_source_support >= 0)
            & (
                metadata.observation_verified_source_support
                + metadata.background_verified_source_support
                <= metadata.verified_source_support
                + config.contract_absolute_tolerance
            )
        )
    ):
        raise ValueError(
            "verified_source_support and evidence channels must be nested "
            "inside source_support"
        )
    expected_local_dynamics = torch.minimum(
        metadata.local_motion_verified_support,
        metadata.local_growth_verified_support,
    )
    if not torch.allclose(
        metadata.local_dynamics_verified_support,
        expected_local_dynamics,
        rtol=0.0,
        atol=config.contract_absolute_tolerance,
    ):
        raise ValueError(
            "local dynamics support must equal motion/growth intersection"
        )
    source_verified = (
        metadata.observation_verified_source_support
        + metadata.background_verified_source_support
    ).clamp(0.0, 1.0)
    if not bool(
        torch.allclose(
            metadata.verified_source_support,
            source_verified,
            rtol=0.0,
            atol=config.contract_absolute_tolerance,
        )
    ):
        raise ValueError(
            "verified_source_support must equal source evidence channels"
        )
    if metadata.background_age_minutes is not None and (
        not math.isfinite(metadata.background_age_minutes)
        or metadata.background_age_minutes < 0
    ):
        raise ValueError("background age must be finite and nonnegative")
    with torch.no_grad():
        _, expected_forecast, expected_valid, _, _, expected_evidence = (
            _forecast_fields_from_state(
                state,
                metadata,
                config,
                audit=False,
            )
        )
    if bool(torch.any(valid & ~expected_valid)):
        raise ValueError("valid mask does not close against the issued state")
    expected_issued_forecast = torch.where(
        valid,
        expected_forecast,
        expected_forecast.new_full((), torch.nan),
    )
    if not bool(
        torch.allclose(
            forecast,
            expected_issued_forecast,
            rtol=0.0,
            atol=config.contract_absolute_tolerance,
            equal_nan=True,
        )
    ):
        raise ValueError("forecast does not close against the issued state")
    for descriptor in fields(ForecastEvidenceFields):
        actual_value = getattr(result.evidence, descriptor.name)
        expected_value = getattr(expected_evidence, descriptor.name)
        if not torch.equal(actual_value, expected_value):
            raise ValueError(
                f"forecast evidence field {descriptor.name} does not close "
                "against the issued state"
            )


def _validate_state_path_provenance(
    name: str,
    path: StatePathProvenance,
    selection_counts: dict[TendencyPairSelection, int],
) -> None:
    if path.pair_count != selection_counts[path.mode]:
        raise ValueError(f"{name} pair count and mode disagree")
    if path.pair_count == 0:
        if not math.isnan(path.minimum_psr):
            raise ValueError(f"unused {name} must have NaN PSR")
    elif not math.isfinite(path.minimum_psr):
        raise ValueError(f"used {name} must have finite PSR")
    if path.age_minutes is not None and (
        not math.isfinite(path.age_minutes) or path.age_minutes < 0
    ):
        raise ValueError(f"{name} age must be finite and nonnegative")


def _path_is_empty(path: StatePathProvenance) -> bool:
    return (
        path.mode is TendencyPairSelection.NONE
        and path.pair_count == 0
        and math.isnan(path.minimum_psr)
        and not path.conflict
        and not path.extrapolated
        and path.age_minutes is None
    )


def _path_has_contribution(path: StatePathProvenance) -> bool:
    return path.age_minutes is not None


def _aggregate_path_matches(
    metadata: ForecastMetadata,
    path: StatePathProvenance,
) -> bool:
    psr_matches = (
        math.isnan(metadata.state_path_minimum_psr)
        and math.isnan(path.minimum_psr)
    ) or metadata.state_path_minimum_psr == path.minimum_psr
    return (
        metadata.state_path_mode is path.mode
        and metadata.state_path_pair_count == path.pair_count
        and psr_matches
        and metadata.state_path_conflict == path.conflict
        and metadata.state_path_extrapolated == path.extrapolated
        and metadata.state_path_age_minutes == path.age_minutes
    )


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
        or config.minimum_growth_overlap_area_km2 is not None
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
    (
        tendency,
        tendency_source,
        observation_paths,
        background_paths,
    ) = _estimate_time_normalized_tendencies(
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
        current_path_verified_source_support,
        current_verified_source_support,
        local_motion_verified_support,
        local_growth_verified_support,
        local_dynamics_verified_support,
        observation_verified_source_support,
        background_verified_source_support,
        observation_source_support,
        background_source_support,
        background_contribution_fraction,
        observation_contributors,
        background_contributors,
    ) = _merge_current_state(
        prepared,
        observation_linear,
        background_linear,
        observation_paths,
        background_paths,
        tendency_source,
        config,
        grid_time_contract,
    )
    state = RadarState(
        echo_linear=current_echo,
        displacement_yx=displacement,
        log_growth_per_step=growth,
    )
    background_tendency_used = tendency_source is TendencySource.BACKGROUND
    background_used = (
        bool(
            torch.any(
                background_source_support
                > config.support_presence_threshold
            )
        )
        or background_tendency_used
    )
    observation_path = _source_path_provenance(
        observation_paths,
        observation_contributors,
        config.interval_minutes,
    )
    background_path = _source_path_provenance(
        background_paths,
        background_contributors,
        config.interval_minutes,
        base_age_minutes=prepared.background_age_minutes or 0.0,
    )
    if bool(torch.any(observation_contributors)):
        state_path_source = TendencySource.OBSERVATION
        state_path = observation_path
    elif bool(torch.any(background_contributors)):
        state_path_source = TendencySource.BACKGROUND
        state_path = background_path
    else:
        state_path_source = TendencySource.NONE
        state_path = StatePathProvenance()
    if state_path_source is TendencySource.NONE:
        state_path_mode = TendencyPairSelection.NONE
        state_path_pair_count = 0
        state_path_minimum_psr = math.nan
        state_path_conflict = False
        state_path_extrapolated = False
        state_path_age_minutes = None
    else:
        state_path_mode = state_path.mode
        state_path_pair_count = state_path.pair_count
        state_path_minimum_psr = state_path.minimum_psr
        state_path_conflict = state_path.conflict
        state_path_extrapolated = state_path.extrapolated
        state_path_age_minutes = state_path.age_minutes
    metadata = ForecastMetadata(
        data_status=prepared.data_status,
        coverage_by_frame=prepared.coverage_by_frame,
        background_used=background_used,
        background_contribution_fraction=background_contribution_fraction,
        background_age_minutes=(
            prepared.background_age_minutes if background_used else None
        ),
        source_support=current_source_support.detach().clone(),
        observation_source_support=(
            observation_source_support.detach().clone()
        ),
        background_source_support=(
            background_source_support.detach().clone()
        ),
        path_verified_source_support=(
            current_path_verified_source_support.detach().clone()
        ),
        verified_source_support=(
            current_verified_source_support.detach().clone()
        ),
        local_motion_verified_support=(
            local_motion_verified_support.detach().clone()
        ),
        local_growth_verified_support=(
            local_growth_verified_support.detach().clone()
        ),
        local_dynamics_verified_support=(
            local_dynamics_verified_support.detach().clone()
        ),
        observation_verified_source_support=(
            observation_verified_source_support.detach().clone()
        ),
        background_verified_source_support=(
            background_verified_source_support.detach().clone()
        ),
        motion_disagreement_px=tendency.motion_disagreement_px.detach(),
        motion_disagreement_mps=tendency.motion_disagreement_mps.detach(),
        growth_disagreement=tendency.growth_disagreement.detach(),
        maximum_growth_saturation_excess=(
            tendency.maximum_growth_saturation_excess.detach()
        ),
        posterior_velocity_uncertainty_mps=(
            tendency.displacement_yx.new_full((), torch.nan)
        ),
        posterior_log_growth_uncertainty_per_step=(
            tendency.log_growth_per_step.new_full((), torch.nan)
        ),
        p1_velocity_saturation_uncertainty_mps=(
            tendency.displacement_yx.new_full((), torch.nan)
        ),
        p1_log_growth_saturation_uncertainty_per_step=(
            tendency.log_growth_per_step.new_full((), torch.nan)
        ),
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
        state_path_source=state_path_source,
        state_path_mode=state_path_mode,
        state_path_pair_count=state_path_pair_count,
        state_path_minimum_psr=state_path_minimum_psr,
        state_path_conflict=state_path_conflict,
        state_path_extrapolated=state_path_extrapolated,
        state_path_age_minutes=state_path_age_minutes,
        observation_path=observation_path,
        background_path=background_path,
        minimum_growth_overlap_support=float(
            tendency.minimum_growth_overlap_support.detach()
        ),
        minimum_growth_overlap_area_km2=float(
            tendency.minimum_growth_overlap_area_km2.detach()
        ),
    )
    return state, metadata


def _estimate_time_normalized_tendencies(
    prepared: PreparedRadarInput,
    observation_linear: Tensor,
    background_linear: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[
    _SourceTendencyEstimate,
    TendencySource,
    _SourceTendencyEstimate,
    _SourceTendencyEstimate,
]:
    observation_estimate = _estimate_source_tendencies(
        prepared.frames_dbz,
        prepared.observed_mask,
        observation_linear,
        config,
        grid_time_contract,
    )
    background_estimate = _estimate_source_tendencies(
        prepared.background_frames_dbz,
        prepared.background_mask,
        background_linear,
        config,
        grid_time_contract,
    )
    if observation_estimate.future_available:
        future = observation_estimate
        source = TendencySource.OBSERVATION
    elif background_estimate.future_available:
        future = background_estimate
        source = TendencySource.BACKGROUND
    else:
        future = observation_estimate
        source = TendencySource.NONE
    return future, source, observation_estimate, background_estimate


def _estimate_source_tendencies(
    frames_dbz: Tensor,
    masks: Tensor,
    linear: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> _SourceTendencyEstimate:
    adjacent_estimates: list[
        tuple[int, tuple[Tensor, _GrowthEvidence, Tensor]]
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
        source_paths = _adjacent_source_paths(
            adjacent_estimates[0][1],
            adjacent_estimates[1][1],
        )
        first_motion, _, first_psr = adjacent_estimates[0][1]
        second_motion, _, second_psr = adjacent_estimates[1][1]
        motion_disagreement = torch.linalg.vector_norm(
            second_motion - first_motion
        )
        motion_disagreement_mps = _motion_disagreement_mps(
            first_motion,
            second_motion,
            config,
            grid_time_contract,
        )
        motion_is_inconsistent = _motion_pairs_are_inconsistent(
            motion_disagreement,
            motion_disagreement_mps,
            config,
        )
        motion, motion_indices, motion_selection = _combine_pair_component(
            first_motion,
            second_motion,
            first_psr,
            second_psr,
            inconsistent=motion_is_inconsistent,
            config=config,
        )
        first_growth_evidence = _growth_evidence_aligned_with_motion(
            linear,
            masks,
            0,
            1,
            motion,
            config,
            grid_time_contract,
        )
        second_growth_evidence = _growth_evidence_aligned_with_motion(
            linear,
            masks,
            1,
            2,
            motion,
            config,
            grid_time_contract,
        )
        (
            growth,
            growth_indices,
            growth_selection,
            growth_disagreement,
            growth_is_inconsistent,
        ) = _combine_adjacent_growth_evidence(
            first_growth_evidence,
            second_growth_evidence,
            first_psr,
            second_psr,
            motion_selection,
            config,
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
        minimum_growth_support, minimum_growth_area = (
            _minimum_growth_evidence(
                (first_growth_evidence, second_growth_evidence),
                growth_indices,
                linear,
            )
        )
        return _SourceTendencyEstimate(
            displacement_yx=motion,
            log_growth_per_step=growth,
            source_displacement_yx=source_paths[0],
            source_log_growth=source_paths[1],
            source_usable=source_paths[2],
            source_support_displacements_yx=source_paths[3],
            motion_disagreement_px=motion_disagreement,
            motion_disagreement_mps=motion_disagreement_mps,
            growth_disagreement=growth_disagreement,
            maximum_growth_saturation_excess=(
                _maximum_growth_saturation_excess(
                    (first_growth_evidence, second_growth_evidence),
                    linear,
                )
            ),
            minimum_phase_correlation_psr=minimum_psr,
            tendency_pair_count=len(used_indices),
            motion_pair_count=len(motion_indices),
            growth_pair_count=len(growth_indices),
            motion_pair_selection=motion_selection,
            growth_pair_selection=growth_selection,
            motion_pair_spans=tuple(
                ((0, 1), (1, 2))[index] for index in motion_indices
            ),
            growth_pair_spans=tuple(
                ((0, 1), (1, 2))[index] for index in growth_indices
            ),
            motion_pair_conflict=motion_is_inconsistent,
            growth_pair_conflict=growth_is_inconsistent,
            minimum_growth_overlap_support=minimum_growth_support,
            minimum_growth_overlap_area_km2=minimum_growth_area,
            reconstruction_pair_count=2,
            reconstruction_selection=TendencyPairSelection.BLENDED,
            reconstruction_minimum_psr=torch.min(
                torch.stack((first_psr, second_psr))
            ),
            reconstruction_recent_psr=second_psr,
            reconstruction_conflict=motion_is_inconsistent,
            reconstruction_extrapolated=False,
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
            selection = (
                TendencyPairSelection.EARLIER
                if adjacent_pair_index == 0
                else TendencyPairSelection.RECENT
            )
            return _single_pair_tendency(
                motion,
                growth,
                psr,
                selection=selection,
                source_pair_index=adjacent_pair_index,
            )
        return _combine_single_adjacent_and_long(
            adjacent_estimate,
            long_estimate,
            adjacent_pair_index,
            linear,
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
        source_paths = (
            _uniform_source_paths(zero_motion, zero_growth)
            if bool(torch.any(masks[-1]))
            else _latest_only_source_paths(zero_motion, zero_growth)
        )
        return _SourceTendencyEstimate(
            displacement_yx=zero_motion,
            log_growth_per_step=zero_growth,
            source_displacement_yx=source_paths[0],
            source_log_growth=source_paths[1],
            source_usable=source_paths[2],
            source_support_displacements_yx=source_paths[3],
            motion_disagreement_px=zero_growth,
            motion_disagreement_mps=unavailable_psr,
            growth_disagreement=zero_growth,
            maximum_growth_saturation_excess=zero_growth,
            minimum_phase_correlation_psr=unavailable_psr,
            tendency_pair_count=0,
            motion_pair_count=0,
            growth_pair_count=0,
            motion_pair_selection=TendencyPairSelection.NONE,
            growth_pair_selection=TendencyPairSelection.NONE,
            motion_pair_spans=(),
            growth_pair_spans=(),
            motion_pair_conflict=False,
            growth_pair_conflict=False,
            minimum_growth_overlap_support=unavailable_psr,
            minimum_growth_overlap_area_km2=unavailable_psr,
            reconstruction_pair_count=0,
            reconstruction_selection=(
                TendencyPairSelection.PERSISTENCE
                if bool(torch.any(masks[-1]))
                else TendencyPairSelection.NONE
            ),
            reconstruction_minimum_psr=unavailable_psr,
            reconstruction_recent_psr=unavailable_psr,
            reconstruction_conflict=False,
            reconstruction_extrapolated=bool(torch.any(masks[-1])),
        )

    motion, growth, psr = long_estimate
    return _single_pair_tendency(
        motion,
        growth,
        psr,
        selection=TendencyPairSelection.LONG,
    )


def _latest_only_source_paths(
    motion: Tensor,
    growth: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    zero_motion = torch.zeros_like(motion)
    zero_growth = torch.zeros_like(growth)
    return (
        torch.stack((zero_motion, zero_motion, zero_motion)),
        torch.stack((zero_growth, zero_growth, zero_growth)),
        torch.tensor(
            (False, False, True),
            dtype=torch.bool,
            device=motion.device,
        ),
        motion.new_zeros((3, 2, 2)),
    )


def _uniform_source_paths(
    motion: Tensor,
    growth: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    zero_motion = torch.zeros_like(motion)
    zero_growth = torch.zeros_like(growth)
    return (
        torch.stack((2.0 * motion, motion, zero_motion)),
        torch.stack((2.0 * growth, growth, zero_growth)),
        torch.ones(3, dtype=torch.bool, device=motion.device),
        torch.stack(
            (
                torch.stack((motion, motion)),
                torch.stack((motion, zero_motion)),
                torch.stack((zero_motion, zero_motion)),
            )
        ),
    )


def _adjacent_source_paths(
    earlier: tuple[Tensor, _GrowthEvidence, Tensor],
    recent: tuple[Tensor, _GrowthEvidence, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    earlier_motion, earlier_growth_evidence, _ = earlier
    recent_motion, recent_growth_evidence, _ = recent
    earlier_growth = earlier_growth_evidence.value
    recent_growth = recent_growth_evidence.value
    zero_motion = torch.zeros_like(earlier_motion)
    zero_growth = torch.zeros_like(earlier_growth)
    return (
        torch.stack(
            (
                earlier_motion + recent_motion,
                recent_motion,
                zero_motion,
            )
        ),
        torch.stack(
            (
                earlier_growth + recent_growth,
                recent_growth,
                zero_growth,
            )
        ),
        torch.ones(3, dtype=torch.bool, device=earlier_motion.device),
        torch.stack(
            (
                torch.stack((earlier_motion, recent_motion)),
                torch.stack((recent_motion, zero_motion)),
                torch.stack((zero_motion, zero_motion)),
            )
        ),
    )


def _single_adjacent_source_paths(
    motion: Tensor,
    growth: Tensor,
    pair_index: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if pair_index == 0:
        return _latest_only_source_paths(motion, growth)
    if pair_index != 1:
        raise ValueError("adjacent pair index must be 0 or 1")
    zero_motion = torch.zeros_like(motion)
    zero_growth = torch.zeros_like(growth)
    return (
        torch.stack((zero_motion, motion, zero_motion)),
        torch.stack((zero_growth, growth, zero_growth)),
        torch.tensor(
            (False, True, True),
            dtype=torch.bool,
            device=motion.device,
        ),
        torch.stack(
            (
                torch.stack((zero_motion, zero_motion)),
                torch.stack((motion, zero_motion)),
                torch.stack((zero_motion, zero_motion)),
            )
        ),
    )


def _long_source_paths(
    motion: Tensor,
    growth: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    zero_motion = torch.zeros_like(motion)
    zero_growth = torch.zeros_like(growth)
    return (
        torch.stack((2.0 * motion, zero_motion, zero_motion)),
        torch.stack((2.0 * growth, zero_growth, zero_growth)),
        torch.tensor(
            (True, False, True),
            dtype=torch.bool,
            device=motion.device,
        ),
        torch.stack(
            (
                torch.stack((motion, motion)),
                torch.stack((zero_motion, zero_motion)),
                torch.stack((zero_motion, zero_motion)),
            )
        ),
    )


def _single_pair_tendency(
    motion: Tensor,
    growth_evidence: _GrowthEvidence,
    psr: Tensor,
    *,
    selection: TendencyPairSelection = TendencyPairSelection.SINGLE,
    source_pair_index: int | None = None,
) -> _SourceTendencyEstimate:
    growth = growth_evidence.value
    zero = growth.new_zeros(())
    if selection is TendencyPairSelection.LONG:
        source_paths = _long_source_paths(motion, growth)
        reconstruction_pair_count = 1
        reconstruction_selection = TendencyPairSelection.LONG
        reconstruction_extrapolated = False
        reconstruction_recent_psr = psr.new_full((), torch.nan)
    elif source_pair_index is not None:
        source_paths = _single_adjacent_source_paths(
            motion,
            growth,
            source_pair_index,
        )
        reconstruction_pair_count = int(source_pair_index == 1)
        reconstruction_selection = (
            TendencyPairSelection.RECENT
            if source_pair_index == 1
            else TendencyPairSelection.NONE
        )
        reconstruction_extrapolated = False
        reconstruction_recent_psr = (
            psr
            if source_pair_index == 1
            else psr.new_full((), torch.nan)
        )
    else:
        source_paths = _uniform_source_paths(motion, growth)
        reconstruction_pair_count = 1
        reconstruction_selection = selection
        reconstruction_extrapolated = True
        reconstruction_recent_psr = psr
    unavailable = psr.new_full((), torch.nan)
    if selection is TendencyPairSelection.LONG:
        pair_span = (0, 2)
    elif source_pair_index is not None:
        pair_span = (source_pair_index, source_pair_index + 1)
    else:
        pair_span = (1, 2)
    return _SourceTendencyEstimate(
        displacement_yx=motion,
        log_growth_per_step=growth,
        source_displacement_yx=source_paths[0],
        source_log_growth=source_paths[1],
        source_usable=source_paths[2],
        source_support_displacements_yx=source_paths[3],
        motion_disagreement_px=zero,
        motion_disagreement_mps=psr.new_full((), torch.nan),
        growth_disagreement=zero,
        maximum_growth_saturation_excess=(
            growth_evidence.saturation_excess
            if growth_evidence.available
            else zero
        ),
        minimum_phase_correlation_psr=psr,
        tendency_pair_count=1,
        motion_pair_count=1,
        growth_pair_count=int(growth_evidence.available),
        motion_pair_selection=selection,
        growth_pair_selection=(
            selection
            if growth_evidence.available
            else TendencyPairSelection.NONE
        ),
        motion_pair_spans=(pair_span,),
        growth_pair_spans=(pair_span,) if growth_evidence.available else (),
        motion_pair_conflict=False,
        growth_pair_conflict=False,
        minimum_growth_overlap_support=(
            growth_evidence.overlap_support
            if growth_evidence.available
            else unavailable
        ),
        minimum_growth_overlap_area_km2=(
            growth_evidence.overlap_area_km2
            if growth_evidence.available
            else unavailable
        ),
        reconstruction_pair_count=reconstruction_pair_count,
        reconstruction_selection=reconstruction_selection,
        reconstruction_minimum_psr=(
            psr if reconstruction_pair_count else unavailable
        ),
        reconstruction_recent_psr=reconstruction_recent_psr,
        reconstruction_conflict=False,
        reconstruction_extrapolated=reconstruction_extrapolated,
    )


def _combine_single_adjacent_and_long(
    adjacent: tuple[Tensor, _GrowthEvidence, Tensor],
    long: tuple[Tensor, _GrowthEvidence, Tensor],
    adjacent_pair_index: int,
    linear: Tensor,
    masks: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> _SourceTendencyEstimate:
    adjacent_motion, adjacent_source_growth, adjacent_psr = adjacent
    long_motion, long_source_growth, long_psr = long
    motion_disagreement = torch.linalg.vector_norm(
        long_motion - adjacent_motion
    )
    motion_disagreement_mps = _motion_disagreement_mps(
        adjacent_motion,
        long_motion,
        config,
        grid_time_contract,
    )
    motion_is_inconsistent = _motion_pairs_are_inconsistent(
        motion_disagreement,
        motion_disagreement_mps,
        config,
    )
    adjacent_previous = adjacent_pair_index
    adjacent_current = adjacent_pair_index + 1
    adjacent_confidence = _pair_confidence(
        adjacent_psr,
        span_penalty=1.0,
    )
    long_confidence = _pair_confidence(
        long_psr,
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
    adjacent_growth_evidence = _growth_evidence_aligned_with_motion(
        linear,
        masks,
        adjacent_previous,
        adjacent_current,
        motion,
        config,
        grid_time_contract,
    )
    long_growth_evidence = _growth_evidence_aligned_with_motion(
        linear,
        masks,
        0,
        2,
        motion,
        config,
        grid_time_contract,
    )
    (
        growth,
        growth_adjacent,
        growth_long,
        growth_selection,
        growth_disagreement,
        growth_is_inconsistent,
    ) = _select_adjacent_or_long_growth_evidence(
        adjacent_growth_evidence,
        long_growth_evidence,
        adjacent_psr,
        long_psr,
        motion_selection,
        config,
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
    growth_indices = tuple(
        index
        for index, used in enumerate((growth_adjacent, growth_long))
        if used
    )
    minimum_growth_support, minimum_growth_area = _minimum_growth_evidence(
        (adjacent_growth_evidence, long_growth_evidence),
        growth_indices,
        linear,
    )
    adjacent_span = (adjacent_pair_index, adjacent_pair_index + 1)
    motion_pair_spans = tuple(
        span
        for span, used in (
            (adjacent_span, motion_adjacent),
            ((0, 2), motion_long),
        )
        if used
    )
    growth_pair_spans = tuple(
        span
        for span, used in (
            (adjacent_span, growth_adjacent),
            ((0, 2), growth_long),
        )
        if used
    )
    if motion_selection is TendencyPairSelection.LONG:
        source_paths = _long_source_paths(
            long_motion,
            long_source_growth.value,
        )
        reconstruction_pair_count = 1
        reconstruction_selection = TendencyPairSelection.LONG
        reconstruction_minimum_psr = long_psr
        reconstruction_recent_psr = long_psr.new_full((), torch.nan)
    elif motion_selection is TendencyPairSelection.SINGLE:
        source_paths = _single_adjacent_source_paths(
            adjacent_motion,
            adjacent_source_growth.value,
            adjacent_pair_index,
        )
        reconstruction_pair_count = int(adjacent_pair_index == 1)
        reconstruction_selection = (
            TendencyPairSelection.RECENT
            if adjacent_pair_index == 1
            else TendencyPairSelection.NONE
        )
        reconstruction_minimum_psr = (
            adjacent_psr
            if reconstruction_pair_count
            else adjacent_psr.new_full((), torch.nan)
        )
        reconstruction_recent_psr = (
            adjacent_psr
            if adjacent_pair_index == 1
            else adjacent_psr.new_full((), torch.nan)
        )
    else:
        source_paths = _latest_only_source_paths(motion, growth)
        reconstruction_pair_count = 0
        reconstruction_selection = TendencyPairSelection.NONE
        reconstruction_minimum_psr = adjacent_psr.new_full((), torch.nan)
        reconstruction_recent_psr = adjacent_psr.new_full((), torch.nan)
    return _SourceTendencyEstimate(
        displacement_yx=motion,
        log_growth_per_step=growth,
        source_displacement_yx=source_paths[0],
        source_log_growth=source_paths[1],
        source_usable=source_paths[2],
        source_support_displacements_yx=source_paths[3],
        motion_disagreement_px=motion_disagreement,
        motion_disagreement_mps=motion_disagreement_mps,
        growth_disagreement=growth_disagreement,
        maximum_growth_saturation_excess=(
            _maximum_growth_saturation_excess(
                (adjacent_growth_evidence, long_growth_evidence),
                linear,
            )
        ),
        minimum_phase_correlation_psr=minimum_psr,
        tendency_pair_count=int(adjacent_used) + int(long_used),
        motion_pair_count=int(motion_adjacent) + int(motion_long),
        growth_pair_count=int(growth_adjacent) + int(growth_long),
        motion_pair_selection=motion_selection,
        growth_pair_selection=growth_selection,
        motion_pair_spans=motion_pair_spans,
        growth_pair_spans=growth_pair_spans,
        motion_pair_conflict=motion_is_inconsistent,
        growth_pair_conflict=growth_is_inconsistent,
        minimum_growth_overlap_support=minimum_growth_support,
        minimum_growth_overlap_area_km2=minimum_growth_area,
        reconstruction_pair_count=reconstruction_pair_count,
        reconstruction_selection=reconstruction_selection,
        reconstruction_minimum_psr=reconstruction_minimum_psr,
        reconstruction_recent_psr=reconstruction_recent_psr,
        reconstruction_conflict=motion_is_inconsistent,
        reconstruction_extrapolated=False,
    )


def _pair_confidence(
    psr: Tensor,
    *,
    span_penalty: float,
) -> Tensor:
    return psr * span_penalty


def _select_single_adjacent_or_long_component(
    adjacent: Tensor,
    long: Tensor,
    adjacent_confidence: Tensor,
    long_confidence: Tensor,
    *,
    inconsistent: bool,
    config: NowcastConfig,
) -> tuple[Tensor, bool, bool, TendencyPairSelection]:
    long_is_clearly_better = _confidence_is_clearly_greater(
        long_confidence,
        adjacent_confidence,
        config,
    )
    adjacent_is_clearly_better = _confidence_is_clearly_greater(
        adjacent_confidence,
        long_confidence,
        config,
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


def _confidence_is_clearly_greater(
    candidate: Tensor,
    reference: Tensor,
    config: NowcastConfig,
) -> bool:
    candidate_value = float(candidate.detach())
    reference_value = float(reference.detach())
    return (
        candidate_value
        > reference_value + config.contract_absolute_tolerance
        and candidate_value
        >= reference_value * config.minimum_pair_confidence_ratio
    )


def _select_adjacent_or_long_growth_evidence(
    adjacent: _GrowthEvidence,
    long: _GrowthEvidence,
    adjacent_psr: Tensor,
    long_psr: Tensor,
    motion_selection: TendencyPairSelection,
    config: NowcastConfig,
) -> tuple[Tensor, bool, bool, TendencyPairSelection, Tensor, bool]:
    disagreement = (
        torch.abs(long.value - adjacent.value)
        if adjacent.available and long.available
        else torch.zeros_like(adjacent.value)
    )
    if motion_selection is TendencyPairSelection.PERSISTENCE:
        return (
            torch.zeros_like(adjacent.value),
            False,
            False,
            TendencyPairSelection.PERSISTENCE,
            disagreement,
            True,
        )
    if adjacent.available and long.available:
        adjacent_confidence = _growth_evidence_confidence(
            adjacent,
            adjacent_psr,
            span_penalty=1.0,
            config=config,
        )
        long_confidence = _growth_evidence_confidence(
            long,
            long_psr,
            span_penalty=config.long_pair_confidence_penalty,
            config=config,
        )
        inconsistent = (
            float(disagreement.detach())
            >= config.maximum_pair_growth_disagreement
            - config.contract_absolute_tolerance
        )
        value, uses_adjacent, uses_long, selection = (
            _select_single_adjacent_or_long_component(
                adjacent.value,
                long.value,
                adjacent_confidence,
                long_confidence,
                inconsistent=inconsistent,
                config=config,
            )
        )
        return (
            value,
            uses_adjacent,
            uses_long,
            selection,
            disagreement,
            inconsistent,
        )
    if adjacent.available:
        return (
            adjacent.value,
            True,
            False,
            TendencyPairSelection.SINGLE,
            disagreement,
            False,
        )
    if long.available:
        return (
            long.value,
            False,
            True,
            TendencyPairSelection.LONG,
            disagreement,
            False,
        )
    return (
        torch.zeros_like(adjacent.value),
        False,
        False,
        TendencyPairSelection.NONE,
        disagreement,
        False,
    )


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
    return disagreement >= limit - config.contract_absolute_tolerance


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
    first_used = (
        float(first_weight.detach()) > config.contract_absolute_tolerance
    )
    second_used = (
        float(second_weight.detach()) > config.contract_absolute_tolerance
    )
    if first_used and second_used:
        return combined, (0, 1), TendencyPairSelection.BLENDED
    if first_used:
        return combined, (0,), TendencyPairSelection.EARLIER
    return combined, (1,), TendencyPairSelection.RECENT


def _combine_adjacent_growth_evidence(
    first: _GrowthEvidence,
    second: _GrowthEvidence,
    first_psr: Tensor,
    second_psr: Tensor,
    motion_selection: TendencyPairSelection,
    config: NowcastConfig,
) -> tuple[
    Tensor,
    tuple[int, ...],
    TendencyPairSelection,
    Tensor,
    bool,
]:
    disagreement = (
        torch.abs(second.value - first.value)
        if first.available and second.available
        else torch.zeros_like(first.value)
    )
    if motion_selection is TendencyPairSelection.PERSISTENCE:
        return (
            torch.zeros_like(first.value),
            (),
            TendencyPairSelection.PERSISTENCE,
            disagreement,
            True,
        )
    if first.available and second.available:
        first_confidence = _growth_evidence_confidence(
            first,
            first_psr,
            span_penalty=1.0,
            config=config,
        )
        second_confidence = _growth_evidence_confidence(
            second,
            second_psr,
            span_penalty=1.0,
            config=config,
        )
        inconsistent = (
            float(disagreement.detach())
            >= config.maximum_pair_growth_disagreement
            - config.contract_absolute_tolerance
        )
        if inconsistent:
            second_is_better = _confidence_is_clearly_greater(
                second_confidence,
                first_confidence,
                config,
            )
            first_is_better = _confidence_is_clearly_greater(
                first_confidence,
                second_confidence,
                config,
            )
            if second_is_better:
                value = second.value
                indices = (1,)
                selection = TendencyPairSelection.RECENT
            elif first_is_better:
                value = first.value
                indices = (0,)
                selection = TendencyPairSelection.EARLIER
            else:
                value = torch.zeros_like(first.value)
                indices = ()
                selection = TendencyPairSelection.PERSISTENCE
        else:
            first_weight = (1.0 - config.recent_weight) * first_confidence
            second_weight = config.recent_weight * second_confidence
            total_weight = (first_weight + second_weight).clamp_min(
                config.epsilon
            )
            value = (
                first_weight * first.value + second_weight * second.value
            ) / total_weight
            indices = (0, 1)
            selection = TendencyPairSelection.BLENDED
        return value, indices, selection, disagreement, inconsistent
    if first.available:
        return (
            first.value,
            (0,),
            TendencyPairSelection.EARLIER,
            disagreement,
            False,
        )
    if second.available:
        return (
            second.value,
            (1,),
            TendencyPairSelection.RECENT,
            disagreement,
            False,
        )
    return (
        torch.zeros_like(first.value),
        (),
        TendencyPairSelection.NONE,
        disagreement,
        False,
    )


def _growth_evidence_confidence(
    evidence: _GrowthEvidence,
    psr: Tensor,
    *,
    span_penalty: float,
    config: NowcastConfig,
) -> Tensor:
    if not evidence.available:
        return torch.zeros_like(psr)
    extent = (
        evidence.overlap_area_km2
        if bool(torch.isfinite(evidence.overlap_area_km2))
        else evidence.overlap_support
    ).clamp_min(config.epsilon)
    support = evidence.overlap_support.clamp_min(config.epsilon)
    previous_mean = evidence.aligned_previous_integral / support
    current_mean = evidence.current_integral / support
    signal = 0.5 * (
        torch.log1p(previous_mean.clamp_min(0.0))
        + torch.log1p(current_mean.clamp_min(0.0))
    )
    alignment_quality = 1.0 / (1.0 + evidence.alignment_log_error)
    return (
        psr.clamp_min(0.0)
        * span_penalty
        * torch.sqrt(extent)
        * signal
        * alignment_quality
    )


def _growth_evidence_aligned_with_motion(
    linear: Tensor,
    masks: Tensor,
    previous_index: int,
    current_index: int,
    motion_per_step: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> _GrowthEvidence:
    previous_mask = masks[previous_index]
    current_mask = masks[current_index]
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
    step_span = current_index - previous_index
    evidence = _aligned_growth_evidence(
        previous_echo,
        current_echo,
        previous_mask,
        current_mask,
        step_span * motion_per_step,
        config,
        max_log_growth=config.max_log_growth_per_step * step_span,
        grid_time_contract=grid_time_contract,
    )
    return replace(
        evidence,
        value=evidence.value / step_span,
        raw_value=evidence.raw_value / step_span,
        saturation_excess=evidence.saturation_excess / step_span,
    )


def _estimate_available_pair(
    frames_dbz: Tensor,
    masks: Tensor,
    linear: Tensor,
    previous_index: int,
    current_index: int,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[Tensor, _GrowthEvidence, Tensor] | None:
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
        float(torch.linalg.vector_norm(previous_signal).detach())
        <= config.epsilon
        or float(torch.linalg.vector_norm(current_signal).detach())
        <= config.epsilon
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
            config.maximum_motion_speed_mps
            + config.contract_absolute_tolerance
        ):
            return None
    per_step_motion = total_motion / step_span
    growth = _growth_evidence_aligned_with_motion(
        linear,
        masks,
        previous_index,
        current_index,
        per_step_motion,
        config,
        grid_time_contract,
    )
    return (
        per_step_motion,
        growth,
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

    offsets = _pair_echo_offsets(
        active_echo.shape,
        config,
        grid_time_contract,
    )
    near_echo = _dilate_mask(active_echo, offsets)

    return not bool(torch.any(near_echo & ~common))


def _dilate_mask(
    mask: Tensor,
    offsets_yx: tuple[tuple[int, int], ...],
) -> Tensor:
    result = torch.zeros_like(mask)
    for offset_y, offset_x in offsets_yx:
        slices = _offset_overlap_slices(mask.shape, offset_y, offset_x)
        if slices is None:
            continue
        source_y, source_x, target_y, target_x = slices
        result[target_y, target_x] |= mask[source_y, source_x]
    return result


def _offset_overlap_slices(
    shape: torch.Size,
    offset_y: int,
    offset_x: int,
) -> tuple[slice, slice, slice, slice] | None:
    height, width = shape
    if abs(offset_y) >= height or abs(offset_x) >= width:
        return None
    source_y = slice(max(0, -offset_y), min(height, height - offset_y))
    source_x = slice(max(0, -offset_x), min(width, width - offset_x))
    target_y = slice(max(0, offset_y), min(height, height + offset_y))
    target_x = slice(max(0, offset_x), min(width, width + offset_x))
    return source_y, source_x, target_y, target_x


def _merge_current_state(
    prepared: PreparedRadarInput,
    observation_linear: Tensor,
    background_linear: Tensor,
    observation_paths: _SourceTendencyEstimate,
    background_paths: _SourceTendencyEstimate,
    tendency_source: TendencySource,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    float,
    Tensor,
    Tensor,
]:
    observation_path_masks = _source_path_verification_masks(
        observation_linear,
        prepared.observed_mask,
        observation_paths,
        config,
        grid_time_contract,
    )
    background_path_masks = _source_path_verification_masks(
        background_linear,
        prepared.background_mask,
        background_paths,
        config,
        grid_time_contract,
    )
    (
        observation_echo,
        observation_support,
        observation_path_verified_support,
        observation_verified_support,
        observation_contribution_by_source,
    ) = (
        _merge_source_frames(
            observation_linear,
            prepared.observed_mask,
            observation_paths.source_displacement_yx,
            observation_paths.source_log_growth,
            observation_paths.source_usable,
            config,
            source_support_displacements_yx=(
                observation_paths.source_support_displacements_yx
            ),
            source_path_verified=observation_path_masks,
        )
    )
    (
        background_echo,
        background_support,
        background_path_verified_support,
        background_verified_support,
        background_contribution_by_source,
    ) = (
        _merge_source_frames(
            background_linear,
            prepared.background_mask,
            background_paths.source_displacement_yx,
            background_paths.source_log_growth,
            background_paths.source_usable,
            config,
            source_support_displacements_yx=(
                background_paths.source_support_displacements_yx
            ),
            source_path_verified=background_path_masks,
        )
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
        config.ratio_regularizer,
    )
    numerator = (
        observation_support * observation_echo
        + background_contribution * background_echo
    )
    current_echo = torch.where(
        current_support > config.support_presence_threshold,
        numerator / current_support.clamp_min(config.epsilon),
        torch.zeros_like(numerator),
    )
    actual_background_path_verified_support = (
        (1.0 - observation_support) * background_path_verified_support
    )
    actual_background_verified_support = (
        (1.0 - observation_support) * background_verified_support
    )
    actual_observation_contribution_by_source = (
        observation_contribution_by_source
    )
    actual_background_contribution_by_source = (
        (1.0 - observation_support)[None]
        * background_contribution_by_source
    )
    observation_source_support = (
        actual_observation_contribution_by_source.sum(dim=0)
    )
    background_source_support = (
        actual_background_contribution_by_source.sum(dim=0)
    )
    observation_contributors = (
        actual_observation_contribution_by_source.flatten(1).amax(dim=1)
        > config.support_presence_threshold
    )
    background_contributors = (
        actual_background_contribution_by_source.flatten(1).amax(dim=1)
        > config.support_presence_threshold
    )
    current_path_verified_support = (
        observation_path_verified_support
        + actual_background_path_verified_support
    ).clamp(0.0, 1.0)
    current_verified_support = (
        observation_verified_support + actual_background_verified_support
    ).clamp(0.0, 1.0)
    if tendency_source is TendencySource.OBSERVATION:
        selected_linear = observation_linear
        selected_masks = prepared.observed_mask
        selected_tendency = observation_paths
    elif tendency_source is TendencySource.BACKGROUND:
        selected_linear = background_linear
        selected_masks = prepared.background_mask
        selected_tendency = background_paths
    else:
        selected_linear = observation_linear
        selected_masks = torch.zeros_like(prepared.observed_mask)
        selected_tendency = observation_paths
    (
        local_motion_verified_support,
        local_growth_verified_support,
    ) = _selected_component_local_evidence(
        selected_linear,
        selected_masks,
        selected_tendency,
        current_echo,
        current_verified_support,
        config,
        grid_time_contract,
    )
    local_dynamics_verified_support = torch.minimum(
        local_motion_verified_support,
        local_growth_verified_support,
    )
    return (
        current_echo,
        current_support,
        current_path_verified_support,
        current_verified_support,
        local_motion_verified_support,
        local_growth_verified_support,
        local_dynamics_verified_support,
        observation_verified_support,
        actual_background_verified_support,
        observation_source_support,
        background_source_support,
        contribution_fraction,
        observation_contributors,
        background_contributors,
    )


def _selected_component_local_evidence(
    linear: Tensor,
    masks: Tensor,
    tendency: _SourceTendencyEstimate,
    current_echo: Tensor,
    state_verified_support: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[Tensor, Tensor]:
    return _local_component_evidence_from_pair_spans(
        linear,
        masks,
        tendency.displacement_yx,
        tendency.log_growth_per_step,
        tendency.motion_pair_spans,
        tendency.growth_pair_spans,
        current_echo,
        state_verified_support,
        config,
        grid_time_contract,
    )


def _local_component_evidence_from_pair_spans(
    linear: Tensor,
    masks: Tensor,
    displacement_yx: Tensor,
    log_growth_per_step: Tensor,
    motion_pair_spans: tuple[tuple[int, int], ...],
    growth_pair_spans: tuple[tuple[int, int], ...],
    current_echo: Tensor,
    state_verified_support: Tensor,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[Tensor, Tensor]:
    echo_threshold = dbz_to_echo(
        current_echo.new_tensor(config.echo_threshold_dbz),
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    detected = (
        state_verified_support > config.support_presence_threshold
    ) & (current_echo >= echo_threshold)
    offsets = _pair_echo_offsets(
        current_echo.shape,
        config,
        grid_time_contract,
    )
    motion_matches = _pair_interval_local_matches(
        linear,
        masks,
        displacement_yx,
        log_growth_per_step,
        motion_pair_spans,
        current_echo,
        detected,
        offsets,
        config,
        grid_time_contract,
        check_growth=False,
    )
    growth_matches = _pair_interval_local_matches(
        linear,
        masks,
        displacement_yx,
        log_growth_per_step,
        growth_pair_spans,
        current_echo,
        detected,
        offsets,
        config,
        grid_time_contract,
        check_growth=True,
    )

    def support_from_matches(matches: Tensor) -> Tensor:
        return torch.where(
            detected,
            state_verified_support * matches.to(dtype=current_echo.dtype),
            state_verified_support,
        ).clamp(0.0, 1.0)

    return (
        support_from_matches(motion_matches),
        support_from_matches(growth_matches),
    )


def _pair_interval_local_matches(
    linear: Tensor,
    masks: Tensor,
    displacement_yx: Tensor,
    log_growth_per_step: Tensor,
    pair_spans: tuple[tuple[int, int], ...],
    current_echo: Tensor,
    current_detected: Tensor,
    offsets: tuple[tuple[int, int], ...],
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
    *,
    check_growth: bool,
) -> Tensor:
    if not pair_spans:
        return torch.zeros_like(current_echo)
    matches = current_detected.to(dtype=current_echo.dtype)
    echo_threshold = dbz_to_echo(
        current_echo.new_tensor(config.echo_threshold_dbz),
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    for previous_index, current_index in pair_spans:
        if not (0 <= previous_index < current_index <= 2):
            raise ValueError("pair spans must be ordered analysis-frame indices")
        step_span = current_index - previous_index
        motion = displacement_yx
        candidate_value, candidate_support = _transport_source_candidate(
            linear[previous_index],
            masks[previous_index],
            step_span * motion,
            motion.expand(step_span, 2),
            log_growth_per_step.new_zeros(()),
            config,
        )
        candidate_echo = torch.where(
            candidate_support > config.support_presence_threshold,
            candidate_value / candidate_support.clamp_min(config.epsilon),
            torch.zeros_like(candidate_value),
        )
        candidate_detected = (
            candidate_support > config.support_presence_threshold
        ) & (candidate_echo >= echo_threshold)
        pair_current_echo = linear[current_index]
        pair_current_detected = masks[current_index] & (
            pair_current_echo >= echo_threshold
        )
        pair_matches_at_endpoint = _exclusive_local_current_matches(
            candidate_echo,
            candidate_support,
            pair_current_echo,
            candidate_detected,
            pair_current_detected,
            offsets,
            grid_time_contract,
            config,
            expected_growth_per_step=(
                log_growth_per_step if check_growth else None
            ),
            step_span=step_span,
        )
        pair_matches_at_current = pair_matches_at_endpoint.to(
            dtype=current_echo.dtype
        )
        for _ in range(2 - current_index):
            pair_matches_at_current = remap(
                pair_matches_at_current,
                motion,
            ).clamp(0.0, 1.0)
        pair_matches_at_current = torch.where(
            current_detected,
            pair_matches_at_current,
            torch.zeros_like(pair_matches_at_current),
        )
        matches = torch.minimum(matches, pair_matches_at_current)
    return matches


def _exclusive_local_current_matches(
    candidate_echo: Tensor,
    candidate_support: Tensor,
    current_echo: Tensor,
    candidate_detected: Tensor,
    current_detected: Tensor,
    offsets: tuple[tuple[int, int], ...],
    grid_time_contract: RadarGridTimeContract | None,
    config: NowcastConfig,
    *,
    expected_growth_per_step: Tensor | None,
    step_span: int,
) -> Tensor:
    shape = candidate_echo.shape
    candidate_ids = torch.arange(
        candidate_echo.numel(),
        dtype=torch.int64,
        device=candidate_echo.device,
    ).reshape(shape)
    best_candidate = torch.full(
        shape,
        -1,
        dtype=torch.int64,
        device=candidate_echo.device,
    )
    best_distance = torch.full_like(candidate_echo, torch.inf)
    best_error = torch.full_like(candidate_echo, torch.inf)
    best_support = torch.full_like(candidate_echo, -torch.inf)
    ordered_offsets = sorted(
        offsets,
        key=lambda offset: (
            _offset_distance_squared(offset, grid_time_contract),
            offset,
        ),
    )
    for offset_y, offset_x in ordered_offsets:
        slices = _offset_overlap_slices(shape, offset_y, offset_x)
        if slices is None:
            continue
        source_y, source_x, target_y, target_x = slices
        candidate = candidate_detected[source_y, source_x]
        current = current_detected[target_y, target_x]
        if expected_growth_per_step is None:
            error = torch.zeros_like(candidate_echo[source_y, source_x])
        else:
            error = torch.abs(
                torch.log(
                    (current_echo[target_y, target_x] + config.ratio_regularizer)
                    / (
                        candidate_echo[source_y, source_x]
                        + config.ratio_regularizer
                    )
                )
                / step_span
                - expected_growth_per_step
            )
        eligible = candidate & current
        if expected_growth_per_step is not None:
            eligible &= (
                error
                <= config.maximum_local_growth_log_error_per_step
                + config.contract_absolute_tolerance
            )
        distance = _offset_distance_squared(
            (offset_y, offset_x),
            grid_time_contract,
        )
        old_distance = best_distance[target_y, target_x]
        old_error = best_error[target_y, target_x]
        old_support = best_support[target_y, target_x]
        support = candidate_support[source_y, source_x]
        better = eligible & (
            (distance < old_distance)
            | (
                (distance == old_distance)
                & (
                    (error < old_error)
                    | ((error == old_error) & (support > old_support))
                )
            )
        )
        best_distance[target_y, target_x] = torch.where(
            better,
            old_distance.new_tensor(distance),
            old_distance,
        )
        best_error[target_y, target_x] = torch.where(
            better,
            error,
            old_error,
        )
        best_support[target_y, target_x] = torch.where(
            better,
            support,
            old_support,
        )
        best_candidate[target_y, target_x] = torch.where(
            better,
            candidate_ids[source_y, source_x],
            best_candidate[target_y, target_x],
        )
    assigned = best_candidate >= 0
    if not bool(torch.any(assigned)):
        return assigned
    counts = torch.bincount(
        best_candidate[assigned],
        minlength=candidate_echo.numel(),
    )
    candidate_count = counts[best_candidate.clamp_min(0)]
    return assigned & (candidate_count == 1)


def _offset_distance_squared(
    offset_yx: tuple[int, int],
    grid_time_contract: RadarGridTimeContract | None,
) -> float:
    offset_y, offset_x = offset_yx
    if grid_time_contract is None:
        return float(offset_y * offset_y + offset_x * offset_x)
    projected = grid_time_contract.projected_displacement_xy(
        torch.tensor((offset_y, offset_x), dtype=torch.float64)
    )
    return float(torch.dot(projected, projected))


def _source_path_verification_masks(
    linear: Tensor,
    masks: Tensor,
    paths: _SourceTendencyEstimate,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> Tensor:
    path_verified = torch.zeros_like(masks)
    path_verified[2] = masks[2]
    echo_threshold = dbz_to_echo(
        linear.new_tensor(config.echo_threshold_dbz),
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    current_detected = masks[2] & (linear[2] >= echo_threshold)
    offsets = _pair_echo_offsets(
        current_detected.shape,
        config,
        grid_time_contract,
    )
    nearby_current = _dilate_mask(current_detected, offsets)
    for source_index in (1, 0):
        if not bool(paths.source_usable[source_index]):
            continue
        candidate_value, candidate_support = _transport_source_candidate(
            linear[source_index],
            masks[source_index],
            paths.source_displacement_yx[source_index],
            paths.source_support_displacements_yx[source_index],
            paths.source_log_growth[source_index],
            config,
        )
        candidate_echo = torch.where(
            candidate_support > config.support_presence_threshold,
            candidate_value / candidate_support.clamp_min(config.epsilon),
            torch.zeros_like(candidate_value),
        )
        candidate_detected = (
            candidate_support > config.support_presence_threshold
        ) & (candidate_echo >= echo_threshold)
        path_verified[source_index] = candidate_detected & nearby_current
    return path_verified


def _pair_echo_offsets(
    shape: torch.Size,
    config: NowcastConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[tuple[int, int], ...]:
    if config.pair_echo_dilation_m is not None:
        if grid_time_contract is None:
            raise ValueError(
                "pair_echo_dilation_m requires a grid/time contract"
            )
        return grid_time_contract.pixel_offsets_within_distance(
            config.pair_echo_dilation_m,
            maximum_radius_yx=(shape[0] - 1, shape[1] - 1),
        )
    radius = config.pair_echo_dilation_px
    return tuple(
        (row, column)
        for row in range(-radius, radius + 1)
        for column in range(-radius, radius + 1)
    )


def _transport_source_candidate(
    linear: Tensor,
    mask: Tensor,
    displacement_yx: Tensor,
    support_displacements_yx: Tensor,
    log_growth: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor]:
    source_support = mask.to(dtype=linear.dtype)
    endpoint_support = remap(source_support, displacement_yx).clamp(0.0, 1.0)
    endpoint_numerator = react_core(
        remap(linear * source_support, displacement_yx),
        log_growth,
    )
    path_support = source_support
    for segment in support_displacements_yx:
        path_support = remap(path_support, segment).clamp(0.0, 1.0)
    candidate_support = torch.where(
        endpoint_support > config.support_presence_threshold,
        path_support,
        torch.zeros_like(path_support),
    )
    candidate_value = torch.where(
        endpoint_support > config.support_presence_threshold,
        endpoint_numerator
        / endpoint_support.clamp_min(config.epsilon)
        * candidate_support,
        torch.zeros_like(endpoint_numerator),
    )
    return candidate_value, candidate_support


def _actual_state_path_provenance(
    paths: _SourceTendencyEstimate,
    contributing_sources: Tensor,
) -> tuple[TendencyPairSelection, int, float, bool, bool]:
    if not bool(torch.any(contributing_sources[:2])):
        return TendencyPairSelection.NONE, 0, math.nan, False, False
    selection = paths.reconstruction_selection
    pair_count = paths.reconstruction_pair_count
    minimum_psr = float(paths.reconstruction_minimum_psr.detach())
    conflict = paths.reconstruction_conflict
    if (
        selection is TendencyPairSelection.BLENDED
        and not bool(contributing_sources[0])
        and bool(contributing_sources[1])
    ):
        selection = TendencyPairSelection.RECENT
        pair_count = 1
        minimum_psr = float(paths.reconstruction_recent_psr.detach())
        conflict = False
    return (
        selection,
        pair_count,
        minimum_psr,
        conflict,
        paths.reconstruction_extrapolated,
    )


def _source_path_provenance(
    paths: _SourceTendencyEstimate,
    contributing_sources: Tensor,
    interval_minutes: int,
    *,
    base_age_minutes: float = 0.0,
) -> StatePathProvenance:
    if not bool(torch.any(contributing_sources)):
        return StatePathProvenance()
    mode, pair_count, minimum_psr, conflict, extrapolated = (
        _actual_state_path_provenance(paths, contributing_sources)
    )
    return StatePathProvenance(
        mode=mode,
        pair_count=pair_count,
        minimum_psr=minimum_psr,
        conflict=conflict,
        extrapolated=extrapolated,
        age_minutes=_state_path_age_minutes(
            contributing_sources,
            interval_minutes,
            base_age_minutes=base_age_minutes,
        ),
    )


def _state_path_age_minutes(
    contributing_sources: Tensor,
    interval_minutes: int,
    *,
    base_age_minutes: float = 0.0,
) -> float | None:
    indices = torch.nonzero(contributing_sources, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    oldest_index = int(indices.min())
    return base_age_minutes + float((2 - oldest_index) * interval_minutes)


def merge_current_support(
    observed_masks: Tensor,
    background_masks: Tensor,
    displacement_yx: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor, float]:
    observation_support = _merge_source_support(
        observed_masks,
        displacement_yx,
    ).clamp(0.0, 1.0)
    background_support = _merge_source_support(
        background_masks,
        displacement_yx,
    ).clamp(0.0, 1.0)
    current_support, background_contribution, contribution_fraction = (
        _combine_source_supports(
            observation_support,
            background_support,
            config.ratio_regularizer,
        )
    )
    return current_support, background_contribution, contribution_fraction


def _combine_source_supports(
    observation_support: Tensor,
    background_support: Tensor,
    ratio_regularizer: float,
) -> tuple[Tensor, Tensor, float]:
    background_contribution = (
        (1.0 - observation_support) * background_support
    )
    current_support = observation_support + background_contribution
    contribution_fraction = float(
        background_contribution.sum()
        / current_support.sum().clamp_min(ratio_regularizer)
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
    source_displacement_yx: Tensor,
    source_log_growth: Tensor,
    source_usable: Tensor,
    config: NowcastConfig,
    *,
    source_support_displacements_yx: Tensor | None = None,
    source_path_verified: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    expected_shape = masks.shape
    default_verification = source_usable[:, None, None].expand(expected_shape)
    if source_path_verified is None:
        source_path_verified = default_verification
    if (
        source_path_verified.shape != expected_shape
        or source_path_verified.dtype != torch.bool
    ):
        raise ValueError(
            "source path verification must be boolean with the source shape"
        )
    latest_mask = masks[2]
    if bool(torch.all(latest_mask)):
        latest_support = latest_mask.to(dtype=linear.dtype)
        latest_contributions = torch.zeros_like(
            masks,
            dtype=linear.dtype,
        )
        latest_contributions[2] = latest_support
        return (
            linear[2],
            latest_support,
            latest_support
            * source_path_verified[2].to(dtype=linear.dtype),
            latest_support,
            latest_contributions,
        )

    numerator = torch.zeros_like(linear[2])
    support = torch.zeros_like(linear[2])
    path_verified_support = torch.zeros_like(linear[2])
    verified_support = torch.zeros_like(linear[2])
    contributions = [torch.zeros_like(linear[2]) for _ in range(3)]
    for source_index in range(3):
        if not bool(source_usable[source_index]):
            continue
        source_mask = masks[source_index]
        if not bool(torch.any(source_mask)):
            continue

        candidate_support = source_mask.to(dtype=linear.dtype)
        candidate_value = linear[source_index] * candidate_support
        if source_index < 2:
            support_displacements = (
                source_support_displacements_yx[source_index]
                if source_support_displacements_yx is not None
                else torch.stack(
                    (
                        source_displacement_yx[source_index],
                        source_displacement_yx.new_zeros(2),
                    )
                )
            )
            candidate_value, candidate_support = (
                _transport_source_candidate(
                    linear[source_index],
                    source_mask,
                    source_displacement_yx[source_index],
                    support_displacements,
                    source_log_growth[source_index],
                    config,
                )
            )

        remaining = 1.0 - candidate_support
        contributions = [remaining * value for value in contributions]
        contributions[source_index] = candidate_support
        numerator = candidate_value + remaining * numerator
        support = candidate_support + remaining * support
        path_verified_support = (
            candidate_support
            * source_path_verified[source_index].to(dtype=linear.dtype)
            + remaining * path_verified_support
        )
        direct_state_support = (
            candidate_support
            if source_index == 2
            else torch.zeros_like(candidate_support)
        )
        verified_support = direct_state_support + remaining * verified_support

    current_echo = torch.where(
        support > config.support_presence_threshold,
        numerator / support.clamp_min(config.epsilon),
        torch.zeros_like(numerator),
    )
    return (
        current_echo,
        support,
        path_verified_support,
        verified_support,
        torch.stack(contributions),
    )


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


def _forecast_fields_from_state(
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
    *,
    audit: bool,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    PositivityAudit,
    tuple[TransportAudit, ...],
    ForecastEvidenceFields,
]:
    current_evidence = _current_evidence_supports(state, metadata)
    current_fields = torch.cat((state.echo_linear[None], current_evidence))
    forecasts = []
    evidence_by_lead = []
    transport_audits = []
    for step in range(1, config.forecast_steps + 1):
        displacement = step * state.displacement_yx
        cell = freeze_remap_cell(displacement)
        moved_fields = remap_core(current_fields, displacement, cell)
        moved = moved_fields[0]
        evidence_by_lead.append(moved_fields[1:].clamp(0.0, 1.0))
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
    evidence = _forecast_evidence_fields_from_supports(
        torch.stack(evidence_by_lead),
        state,
        metadata,
        config,
    )
    valid_mask = _forecast_valid_mask(metadata, config, evidence)
    forecast_dbz = torch.where(
        valid_mask,
        forecast_dbz,
        forecast_dbz.new_full((), torch.nan),
    )
    return (
        forecast_linear,
        forecast_dbz,
        valid_mask,
        final_audit,
        tuple(transport_audits),
        evidence,
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
    run.validate_integrity()
    _validate_state_for_run(state, run)
    input_echo, input_audit = validate_physical_echo(
        state.echo_linear,
        name="forecast input state",
    )
    state = replace(state, echo_linear=input_echo)
    (
        _,
        forecast_dbz,
        valid_mask,
        final_audit,
        transport_audits,
        evidence,
    ) = _forecast_fields_from_state(
        state,
        metadata,
        config,
        audit=audit,
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
        evidence=evidence,
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
    operational_calibration_manifest: (
        OperationalCalibrationManifest | None
    ) = None,
    operational_calibration_approval_digest: str | None = None,
    operational_data_identity: OperationalDataIdentity | None = None,
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
        operational_calibration_manifest_json=(
            None
            if operational_calibration_manifest is None
            else operational_calibration_manifest.json
        ),
        operational_calibration_manifest_digest=(
            None
            if operational_calibration_manifest is None
            else operational_calibration_manifest.digest
        ),
        operational_calibration_approval_digest=(
            operational_calibration_approval_digest
        ),
        operational_data_identity_json=(
            None
            if operational_data_identity is None
            else operational_data_identity.json
        ),
        operational_data_identity_digest=(
            None
            if operational_data_identity is None
            else operational_data_identity.digest
        ),
    )
    return forecast_from_state(
        state,
        metadata,
        config,
        run=run,
        audit=audit,
    )


def _forecast_valid_mask(
    metadata: ForecastMetadata,
    config: NowcastConfig,
    evidence: ForecastEvidenceFields,
) -> Tensor:
    if metadata.data_status == DataStatus.UNAVAILABLE:
        return torch.zeros_like(evidence.source_support, dtype=torch.bool)
    valid = evidence.source_support >= config.min_publish_support
    if config.minimum_publish_verified_support is not None:
        valid &= (
            evidence.verified_support
            >= config.minimum_publish_verified_support
        )
    if config.minimum_publish_confidence is not None:
        valid &= evidence.confidence >= (
            config.minimum_publish_confidence
        )
    if config.minimum_publish_observation_verified_support is not None:
        valid &= evidence.observation_verified_support >= (
            config.minimum_publish_observation_verified_support
        )
    if (
        config.maximum_publish_background_fraction is not None
        and metadata.background_contribution_fraction
        > config.maximum_publish_background_fraction
        + config.contract_absolute_tolerance
    ):
        valid &= False
    return valid


def _forecast_velocity_uncertainty_mps(
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
) -> Tensor:
    posterior = metadata.posterior_velocity_uncertainty_mps
    if (
        metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL
        and bool(torch.isfinite(posterior))
    ):
        posterior = posterior.to(
            dtype=state.echo_linear.dtype,
            device=state.echo_linear.device,
        )
        model_error = posterior.new_tensor(
            config.forecast_velocity_uncertainty_mps
        )
        saturation = metadata.p1_velocity_saturation_uncertainty_mps.to(
            dtype=posterior.dtype,
            device=posterior.device,
        )
        return torch.sqrt(
            posterior.square()
            + model_error.square()
            + saturation.square()
        )
    uncertainty = state.echo_linear.new_tensor(
        config.forecast_velocity_uncertainty_mps
    )
    disagreement = metadata.motion_disagreement_mps
    if bool(torch.isfinite(disagreement)):
        uncertainty = torch.maximum(
            uncertainty,
            0.5 * disagreement.clamp_min(0.0),
        )
    return uncertainty * _dynamics_evidence_uncertainty_multiplier(
        state,
        metadata,
        config,
        pair_count=metadata.motion_pair_count,
    )


def _p1_posterior_is_available(metadata: ForecastMetadata) -> bool:
    return all(
        bool(torch.isfinite(value))
        for value in (
            metadata.posterior_velocity_uncertainty_mps,
            metadata.posterior_log_growth_uncertainty_per_step,
            metadata.p1_velocity_saturation_uncertainty_mps,
            metadata.p1_log_growth_saturation_uncertainty_per_step,
        )
    )


def _dynamics_evidence_uncertainty_multiplier(
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
    *,
    pair_count: int,
) -> Tensor:
    if metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL:
        return state.echo_linear.new_ones(())
    if pair_count == 0:
        multiplier = config.persistence_uncertainty_multiplier
    elif pair_count == 1:
        multiplier = config.single_pair_uncertainty_multiplier
    else:
        multiplier = 1.0

    psr = float(metadata.minimum_phase_correlation_psr)
    if pair_count > 0 and math.isfinite(psr) and psr > 0.0:
        reference_psr = 2.0 * config.minimum_phase_correlation_psr
        multiplier *= max(1.0, reference_psr / psr)

    if metadata.tendency_source is TendencySource.BACKGROUND:
        age = metadata.background_age_minutes
        if age is None:
            raise ValueError("background tendency requires background age")
        multiplier *= 1.0 + (
            age / config.background_tendency_age_uncertainty_scale_minutes
        )
    return state.echo_linear.new_tensor(multiplier)


def _current_evidence_supports(
    state: RadarState,
    metadata: ForecastMetadata,
) -> Tensor:
    return torch.stack(
        (
            metadata.source_support,
            metadata.path_verified_source_support,
            metadata.verified_source_support,
            metadata.local_motion_verified_support,
            metadata.local_growth_verified_support,
            metadata.local_dynamics_verified_support,
            metadata.observation_source_support,
            metadata.background_source_support,
            metadata.observation_verified_source_support,
            metadata.background_verified_source_support,
        )
    ).to(dtype=state.echo_linear.dtype, device=state.echo_linear.device)


def forecast_evidence_fields(
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
) -> ForecastEvidenceFields:
    current_support = _current_evidence_supports(state, metadata)
    supports_by_lead = []
    for step in range(1, config.forecast_steps + 1):
        displacement = step * state.displacement_yx
        supports_by_lead.append(
            remap_core(
                current_support,
                displacement,
                freeze_remap_cell(displacement),
            ).clamp(0.0, 1.0)
        )
    return _forecast_evidence_fields_from_supports(
        torch.stack(supports_by_lead),
        state,
        metadata,
        config,
    )


def _forecast_evidence_fields_from_supports(
    supports_by_lead: Tensor,
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
) -> ForecastEvidenceFields:
    (
        source_support,
        path_verified_support,
        verified_support,
        local_motion_verified_support,
        local_growth_verified_support,
        local_dynamics_verified_support,
        observation_source_support,
        background_source_support,
        observation_verified_support,
        background_verified_support,
    ) = supports_by_lead.unbind(dim=1)

    velocity_uncertainty = _forecast_velocity_uncertainty_mps(
        state,
        metadata,
        config,
    )
    motion_multiplier = _dynamics_evidence_uncertainty_multiplier(
        state,
        metadata,
        config,
        pair_count=metadata.motion_pair_count,
    )
    growth_multiplier = _dynamics_evidence_uncertainty_multiplier(
        state,
        metadata,
        config,
        pair_count=metadata.growth_pair_count,
    )
    lead_seconds = torch.arange(
        1,
        config.forecast_steps + 1,
        dtype=state.echo_linear.dtype,
        device=state.echo_linear.device,
    ) * (config.interval_minutes * 60.0)
    position_uncertainty = lead_seconds * velocity_uncertainty
    growth_uncertainty = _forecast_log_growth_uncertainty(
        state,
        metadata,
        config,
    )
    if (
        metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL
        and not _p1_posterior_is_available(metadata)
    ):
        confidence = torch.zeros_like(local_dynamics_verified_support)
    else:
        decay = torch.exp(
            -0.5
            * (
                (
                    position_uncertainty
                    / config.forecast_confidence_length_scale_m
                ).square()
                + (
                    growth_uncertainty
                    / config.forecast_log_growth_confidence_scale
                ).square()
            )
        )
        confidence = (
            local_dynamics_verified_support * decay[:, None, None]
        ).clamp(0.0, 1.0)

    return ForecastEvidenceFields(
        source_support=source_support,
        path_verified_support=path_verified_support,
        verified_support=verified_support,
        local_motion_verified_support=local_motion_verified_support,
        local_growth_verified_support=local_growth_verified_support,
        local_dynamics_verified_support=local_dynamics_verified_support,
        observation_source_support=observation_source_support,
        background_source_support=background_source_support,
        observation_verified_support=observation_verified_support,
        background_verified_support=background_verified_support,
        velocity_uncertainty_mps=velocity_uncertainty,
        motion_evidence_uncertainty_multiplier=motion_multiplier,
        growth_evidence_uncertainty_multiplier=growth_multiplier,
        position_uncertainty_m=position_uncertainty,
        log_growth_uncertainty=growth_uncertainty,
        confidence=confidence,
    )


def _forecast_log_growth_uncertainty(
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
) -> Tensor:
    posterior = metadata.posterior_log_growth_uncertainty_per_step
    if (
        metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL
        and bool(torch.isfinite(posterior))
    ):
        posterior = posterior.to(
            dtype=state.echo_linear.dtype,
            device=state.echo_linear.device,
        )
        model_error = posterior.new_tensor(
            config.forecast_log_growth_uncertainty_per_step
        )
        saturation = (
            metadata.p1_log_growth_saturation_uncertainty_per_step.to(
                dtype=posterior.dtype,
                device=posterior.device,
            )
        )
        uncertainty_per_step = torch.sqrt(
            posterior.square()
            + model_error.square()
            + saturation.square()
        )
    else:
        uncertainty_per_step = state.echo_linear.new_tensor(
            config.forecast_log_growth_uncertainty_per_step
        )
        disagreement = metadata.growth_disagreement
        if bool(torch.isfinite(disagreement)):
            uncertainty_per_step = torch.maximum(
                uncertainty_per_step,
                0.5 * disagreement.clamp_min(0.0),
            )
        uncertainty_per_step *= _dynamics_evidence_uncertainty_multiplier(
            state,
            metadata,
            config,
            pair_count=metadata.growth_pair_count,
        )
        uncertainty_per_step = torch.maximum(
            uncertainty_per_step,
            metadata.maximum_growth_saturation_excess,
        )
    retention = math.exp(
        -config.interval_minutes / config.growth_decay_minutes
    )
    powers = torch.arange(
        config.forecast_steps,
        dtype=state.echo_linear.dtype,
        device=state.echo_linear.device,
    )
    growth_sum = torch.cumsum(retention**powers, dim=0)
    return growth_sum * uncertainty_per_step


def _validate_frames(frames: Tensor) -> None:
    if frames.ndim != 3 or frames.shape[0] != 3:
        raise ValueError("frames_dbz must have shape [3, height, width]")
    if frames.shape[1] < 2 or frames.shape[2] < 2:
        raise ValueError("frame height and width must both be at least 2")
    if not frames.is_floating_point():
        raise TypeError("frames_dbz must be a floating-point tensor")


def _aligned_growth_evidence(
    previous: Tensor,
    current: Tensor,
    previous_mask: Tensor,
    current_mask: Tensor,
    displacement_yx: Tensor,
    config: NowcastConfig,
    *,
    max_log_growth: float | None = None,
    grid_time_contract: RadarGridTimeContract | None,
) -> _GrowthEvidence:
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
    echo_threshold = dbz_to_echo(
        previous.new_tensor(config.echo_threshold_dbz),
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    common_support = current_mask & (
        moved_support > config.support_presence_threshold
    )
    echo_relevant = common_support & (
        (aligned >= echo_threshold) | (current >= echo_threshold)
    )
    overlap_weight = moved_support * echo_relevant.to(dtype=previous.dtype)
    overlap_support = overlap_weight.sum()
    overlap_area_km2 = (
        overlap_support * (grid_time_contract.cell_area_m2 / 1.0e6)
        if grid_time_contract is not None
        else previous.new_full((), torch.nan)
    )
    previous_integrated_echo = (overlap_weight * aligned).sum()
    current_integrated_echo = (overlap_weight * current).sum()
    raw_growth = torch.log(
        (current_integrated_echo + config.ratio_regularizer)
        / (previous_integrated_echo + config.ratio_regularizer)
    )
    point_log_ratio = torch.log(
        (current + config.ratio_regularizer)
        / (aligned + config.ratio_regularizer)
    )
    alignment_log_error = (
        overlap_weight * torch.abs(point_log_ratio - raw_growth)
    ).sum() / overlap_support.clamp_min(config.epsilon)
    enough_support = (
        float(overlap_support.detach())
        + config.contract_absolute_tolerance
        >= config.minimum_growth_overlap_support
    )
    enough_area = (
        config.minimum_growth_overlap_area_km2 is None
        or (
            grid_time_contract is not None
            and float(overlap_area_km2.detach())
            + config.contract_absolute_tolerance
            >= config.minimum_growth_overlap_area_km2
        )
    )
    available = (
        enough_support
        and enough_area
        and float(previous_integrated_echo.detach())
        > config.support_presence_threshold
    )
    if not available:
        return _GrowthEvidence(
            value=previous.new_zeros(()),
            raw_value=raw_growth,
            saturation_excess=previous.new_zeros(()),
            available=False,
            overlap_support=overlap_support,
            overlap_area_km2=overlap_area_km2,
            aligned_previous_integral=previous_integrated_echo,
            current_integral=current_integrated_echo,
            alignment_log_error=alignment_log_error,
        )
    return _GrowthEvidence(
        value=raw_growth.clamp(-limit, limit),
        raw_value=raw_growth,
        saturation_excess=(torch.abs(raw_growth) - limit).clamp_min(0.0),
        available=True,
        overlap_support=overlap_support,
        overlap_area_km2=overlap_area_km2,
        aligned_previous_integral=previous_integrated_echo,
        current_integral=current_integrated_echo,
        alignment_log_error=alignment_log_error,
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
    shift, psr, search_interior, _ = _phase_correlation_details(
        previous_dbz,
        current_dbz,
        config,
        max_displacement_yx=max_displacement_yx,
        grid_time_contract=grid_time_contract,
    )
    return shift, psr, search_interior


def _phase_correlation_details(
    previous_dbz: Tensor,
    current_dbz: Tensor,
    config: NowcastConfig,
    *,
    max_displacement_yx: Tensor | None = None,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> tuple[Tensor, Tensor, bool, tuple[int, int]]:
    previous = (previous_dbz - config.echo_threshold_dbz).clamp_min(0.0)
    current = (current_dbz - config.echo_threshold_dbz).clamp_min(0.0)

    energy = (
        torch.linalg.vector_norm(previous)
        * torch.linalg.vector_norm(current)
    )
    if float(energy.detach()) <= config.epsilon:
        return previous.new_zeros(2), previous.new_zeros(()), False, (0, 0)

    height, width = previous.shape
    previous = previous - previous.mean()
    current = current - current.mean()
    centered_energy = (
        torch.linalg.vector_norm(previous)
        * torch.linalg.vector_norm(current)
    )
    if float(centered_energy.detach()) <= config.epsilon:
        return previous.new_zeros(2), previous.new_zeros(()), False, (0, 0)

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

    base_shift_y = (
        peak_y - correlation_height
        if peak_y > correlation_height / 2
        else peak_y
    )
    base_shift_x = (
        peak_x - correlation_width
        if peak_x > correlation_width / 2
        else peak_x
    )
    shift = torch.stack(
        (
            correlation.new_tensor(float(base_shift_y)) + offset_y,
            correlation.new_tensor(float(base_shift_x)) + offset_x,
        )
    )
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
        torch.all(
            torch.abs(shift)
            <= limits + config.contract_absolute_tolerance
        )
    )
    interior_bin_limit = (limits - 0.5).clamp_min(
        config.contract_absolute_tolerance
    )
    away_from_search_boundary = bool(
        torch.all(torch.abs(integer_peak_shift) < interior_bin_limit)
    )
    return (
        shift,
        psr,
        inside_limits and away_from_search_boundary,
        (int(peak_shift_y), int(peak_shift_x)),
    )


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
) -> Tensor:
    left = values[(peak - 1) % values.numel()]
    center = values[peak]
    right = values[(peak + 1) % values.numel()]
    denominator = left - 2.0 * center + right
    if abs(float(denominator.detach())) <= config.epsilon:
        return denominator.new_zeros(())
    offset = 0.5 * (left - right) / denominator
    return offset.clamp(-0.5, 0.5)
