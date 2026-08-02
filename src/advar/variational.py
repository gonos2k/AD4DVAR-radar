from __future__ import annotations

from collections.abc import Callable
from collections import deque
from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ._digest import json_digest, tensor_digest
from .diagnostics import EchoPositivityError, PositivityAudit, validate_physical_echo
from .matrix_free import pcg
from .nowcast import (
    ForecastMetadata,
    ForecastResult,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    estimate_prepared_state,
    forecast_from_state,
    merge_current_support,
    motion_displacement_limits_yx,
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


AmplitudeInformationPolicy = Literal[
    "research_degraded",
    "operational_fallback",
]
AmplitudeConfidencePolicy = Literal[
    "research_degraded",
    "operational_fallback",
]
AnalysisExecutionMode = Literal["research", "operational"]


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
    causal_support_uncertainty_m: float | None = None
    amplitude_displacement_tolerance_px: int = 1
    amplitude_displacement_tolerance_m: float | None = None
    maximum_latest_detected_error_std: float = 3.0
    maximum_unresolved_amplitude_fraction: float = 0.01
    minimum_amplitude_total_quality_weight: float = 0.01
    minimum_amplitude_effective_pixel_count: float = 1.0
    amplitude_information_policy: AmplitudeInformationPolicy = (
        "research_degraded"
    )
    minimum_integrated_echo_ratio_for_confidence: float = 0.5
    minimum_soft_echo_area_ratio_for_confidence: float = 0.5
    maximum_established_excess_growth_fraction_for_confidence: float = 0.01
    field_smoothness_weight: float = 0.01
    maximum_outer_iterations: int = 4
    maximum_pcg_iterations: int = 40
    maximum_damping_retries: int = 2
    pcg_relative_tolerance: float = 1.0e-5
    gradient_tolerance: float = 1.0e-5
    step_tolerance: float = 1.0e-4
    initial_damping: float = 1.0e-2
    minimum_damping: float = 1.0e-6
    maximum_damping: float = 1.0e6
    execution_mode: AnalysisExecutionMode = "research"
    operational_calibration_id: str | None = None
    amplitude_confidence_policy: AmplitudeConfidencePolicy = (
        "research_degraded"
    )
    maximum_integrated_echo_ratio_for_confidence: float = 2.0
    maximum_soft_echo_area_ratio_for_confidence: float = 2.0
    motion_increment_scale_mps: float | None = None

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
            "minimum_amplitude_total_quality_weight": (
                self.minimum_amplitude_total_quality_weight
            ),
            "minimum_amplitude_effective_pixel_count": (
                self.minimum_amplitude_effective_pixel_count
            ),
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
        if (
            type(self.amplitude_displacement_tolerance_px) is not int
            or self.amplitude_displacement_tolerance_px < 0
        ):
            raise ValueError(
                "amplitude_displacement_tolerance_px cannot be negative"
            )
        physical_distances = {
            "causal_support_uncertainty_m": self.causal_support_uncertainty_m,
            "amplitude_displacement_tolerance_m": (
                self.amplitude_displacement_tolerance_m
            ),
        }
        for name, value in physical_distances.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
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
        if self.amplitude_information_policy not in (
            "research_degraded",
            "operational_fallback",
        ):
            raise ValueError(
                "amplitude_information_policy must be research_degraded "
                "or operational_fallback"
            )
        if self.amplitude_confidence_policy not in (
            "research_degraded",
            "operational_fallback",
        ):
            raise ValueError(
                "amplitude_confidence_policy must be research_degraded "
                "or operational_fallback"
            )
        if self.execution_mode not in ("research", "operational"):
            raise ValueError("execution_mode must be research or operational")
        if self.operational_calibration_id is not None and (
            not isinstance(self.operational_calibration_id, str)
            or not self.operational_calibration_id
            or self.operational_calibration_id.strip()
            != self.operational_calibration_id
        ):
            raise ValueError(
                "operational_calibration_id must be a nonempty canonical string"
            )
        if self.execution_mode == "operational" and (
            self.amplitude_information_policy != "operational_fallback"
            or self.amplitude_confidence_policy != "operational_fallback"
            or self.operational_calibration_id is None
            or self.motion_increment_scale_mps is None
        ):
            raise ValueError(
                "operational execution requires fallback amplitude policies "
                "and calibrated physical control settings"
            )
        if (
            self.execution_mode == "research"
            and self.operational_calibration_id is not None
        ):
            raise ValueError(
                "operational_calibration_id requires operational execution"
            )
        confidence_fractions = {
            "minimum_integrated_echo_ratio_for_confidence": (
                self.minimum_integrated_echo_ratio_for_confidence
            ),
            "minimum_soft_echo_area_ratio_for_confidence": (
                self.minimum_soft_echo_area_ratio_for_confidence
            ),
            "maximum_established_excess_growth_fraction_for_confidence": (
                self.maximum_established_excess_growth_fraction_for_confidence
            ),
        }
        for name, value in confidence_fractions.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        confidence_upper_ratios = {
            "maximum_integrated_echo_ratio_for_confidence": (
                self.maximum_integrated_echo_ratio_for_confidence
            ),
            "maximum_soft_echo_area_ratio_for_confidence": (
                self.maximum_soft_echo_area_ratio_for_confidence
            ),
        }
        for name, value in confidence_upper_ratios.items():
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be finite and at least 1")
        if (
            self.minimum_integrated_echo_ratio_for_confidence
            > self.maximum_integrated_echo_ratio_for_confidence
        ):
            raise ValueError(
                "minimum integrated echo ratio cannot exceed its maximum"
            )
        if (
            self.minimum_soft_echo_area_ratio_for_confidence
            > self.maximum_soft_echo_area_ratio_for_confidence
        ):
            raise ValueError(
                "minimum soft echo area ratio cannot exceed its maximum"
            )
        if (
            not math.isfinite(self.field_smoothness_weight)
            or self.field_smoothness_weight < 0.0
        ):
            raise ValueError("field_smoothness_weight cannot be negative")
        if self.motion_increment_scale_mps is not None and (
            not math.isfinite(self.motion_increment_scale_mps)
            or self.motion_increment_scale_mps <= 0.0
        ):
            raise ValueError("motion_increment_scale_mps must be positive")

    @property
    def maximum_detected_error_std(self) -> float:
        return self.maximum_latest_detected_error_std


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


@dataclass(frozen=True)
class FrozenOuterState:
    initial_background_dbz: Tensor
    initial_support_mask: Tensor
    active_field_index: Tensor
    causal_only_mask: Tensor
    causal_seed_mask: Tensor
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
    grid_time_contract: RadarGridTimeContract | None
    motion_limits_yx: Tensor
    amplitude_displacement_offsets_yx: tuple[tuple[int, int], ...]
    analysis_remap_cells: tuple[RemapCell, RemapCell]
    smooth_edge_left_index: Tensor
    smooth_edge_right_index: Tensor
    smooth_edge_physical_weight: Tensor

    @property
    def amplitude_displacement_tolerance_yx(self) -> tuple[int, int]:
        return (
            max(abs(row) for row, _ in self.amplitude_displacement_offsets_yx),
            max(
                abs(column)
                for _, column in self.amplitude_displacement_offsets_yx
            ),
        )


@dataclass(frozen=True)
class AnalysisTrajectory:
    frames_linear: Tensor
    displacement_yx: Tensor
    log_growth_per_step: Tensor


@dataclass(frozen=True)
class _AmplitudeDiagnostics:
    unresolved_fraction_by_time: Tensor
    unresolved_pixel_fraction_by_time: Tensor
    violation_score_by_time: Tensor
    integrated_echo_ratio_by_time: Tensor
    displacement_tolerant_soft_echo_area_ratio_by_time: Tensor
    effective_pixel_count_by_time: Tensor
    bad_quality_weight_by_time: Tensor
    total_quality_weight_by_time: Tensor
    information_sufficient_by_time: Tensor
    established_echo_excess_growth_fraction_by_time: Tensor
    maximum_growth_envelope_ratio_by_time: Tensor
    precursor_object_count_by_time: Tensor
    insufficient_object_count_by_time: Tensor
    maximum_object_unresolved_fraction_by_time: Tensor
    minimum_object_integrated_echo_ratio_by_time: Tensor
    maximum_object_integrated_echo_ratio_by_time: Tensor
    minimum_object_soft_echo_area_ratio_by_time: Tensor
    maximum_object_soft_echo_area_ratio_by_time: Tensor

    def _gated(self, values: Tensor) -> Tensor:
        return torch.where(
            self.information_sufficient_by_time,
            values,
            torch.zeros_like(values),
        )

    @property
    def maximum_unresolved_fraction(self) -> Tensor:
        return torch.max(self.unresolved_fraction_by_time)

    @property
    def maximum_gated_unresolved_fraction(self) -> Tensor:
        return torch.max(self._gated(self.unresolved_fraction_by_time))

    @property
    def maximum_violation_score(self) -> Tensor:
        return torch.max(self.violation_score_by_time)

    @property
    def maximum_gated_violation_score(self) -> Tensor:
        return torch.max(self._gated(self.violation_score_by_time))

    @property
    def total_gated_violation_score(self) -> Tensor:
        return self._gated(self.violation_score_by_time).sum()

    @property
    def has_insufficient_information(self) -> bool:
        return bool(
            torch.any(~self.information_sufficient_by_time)
            or torch.any(self.insufficient_object_count_by_time > 0)
        )

    def degrades_confidence(self, config: AnalysisConfig) -> bool:
        available_echo_ratio = ~torch.isnan(
            self.integrated_echo_ratio_by_time
        )
        echo_ratio_low = available_echo_ratio & (
            self.integrated_echo_ratio_by_time
            < config.minimum_integrated_echo_ratio_for_confidence
        )
        echo_ratio_high = available_echo_ratio & (
            self.integrated_echo_ratio_by_time
            > config.maximum_integrated_echo_ratio_for_confidence
        )
        available_area_ratio = ~torch.isnan(
            self.displacement_tolerant_soft_echo_area_ratio_by_time
        )
        area_ratio_low = available_area_ratio & (
            self.displacement_tolerant_soft_echo_area_ratio_by_time
            < config.minimum_soft_echo_area_ratio_for_confidence
        )
        area_ratio_high = available_area_ratio & (
            self.displacement_tolerant_soft_echo_area_ratio_by_time
            > config.maximum_soft_echo_area_ratio_for_confidence
        )
        available_excess = ~torch.isnan(
            self.established_echo_excess_growth_fraction_by_time
        )
        excess_high = available_excess & (
            self.established_echo_excess_growth_fraction_by_time
            > config.maximum_established_excess_growth_fraction_for_confidence
        )
        object_unresolved_high = (
            self.maximum_object_unresolved_fraction_by_time
            > config.maximum_unresolved_amplitude_fraction
        )
        object_echo_low = (
            ~torch.isnan(self.minimum_object_integrated_echo_ratio_by_time)
            & (
                self.minimum_object_integrated_echo_ratio_by_time
                < config.minimum_integrated_echo_ratio_for_confidence
            )
        )
        object_echo_high = (
            ~torch.isnan(self.maximum_object_integrated_echo_ratio_by_time)
            & (
                self.maximum_object_integrated_echo_ratio_by_time
                > config.maximum_integrated_echo_ratio_for_confidence
            )
        )
        object_area_low = (
            ~torch.isnan(self.minimum_object_soft_echo_area_ratio_by_time)
            & (
                self.minimum_object_soft_echo_area_ratio_by_time
                < config.minimum_soft_echo_area_ratio_for_confidence
            )
        )
        object_area_high = (
            ~torch.isnan(self.maximum_object_soft_echo_area_ratio_by_time)
            & (
                self.maximum_object_soft_echo_area_ratio_by_time
                > config.maximum_soft_echo_area_ratio_for_confidence
            )
        )
        return bool(
            torch.any(
                echo_ratio_low
                | echo_ratio_high
                | area_ratio_low
                | area_ratio_high
                | excess_high
                | object_unresolved_high
                | object_echo_low
                | object_echo_high
                | object_area_low
                | object_area_high
            )
        )


@dataclass(frozen=True)
class _IdentifiabilityDiagnostics:
    dynamics_data_gram_eigenvalues: tuple[float, float, float]
    dynamics_data_information_trace: float
    dynamics_data_numerical_rank: int
    dynamics_data_effective_dimension: float
    dynamics_data_to_prior_ratio_by_mode: tuple[float, float, float]
    regularized_dynamics_hessian_eigenvalues: tuple[float, float, float]
    regularized_dynamics_hessian_condition_number: float
    field_growth_jacobian_cosine: float | None
    field_motion_jacobian_cosine_by_control: tuple[
        float | None,
        float | None,
    ]


AmplitudeDiagnosticsSource = Literal[
    "unavailable",
    "returned_analysis",
    "rejected_candidate",
]


@dataclass(frozen=True)
class AnalysisResult:
    control: Tensor
    active_field_index: Tensor
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
    unresolved_amplitude_fraction_by_time: tuple[float, float] | None = None
    unresolved_pixel_fraction_by_time: tuple[float, float] | None = None
    amplitude_violation_score: float | None = None
    amplitude_violation_score_by_time: tuple[float, float] | None = None
    integrated_echo_ratio_by_time: tuple[float, float] | None = None
    displacement_tolerant_soft_echo_area_ratio_by_time: (
        tuple[float, float] | None
    ) = None
    effective_precursor_pixel_count_by_time: (
        tuple[float, float] | None
    ) = None
    bad_quality_weight_by_time: tuple[float, float] | None = None
    total_quality_weight_by_time: tuple[float, float] | None = None
    amplitude_information_sufficient_by_time: (
        tuple[bool, bool] | None
    ) = None
    insufficient_amplitude_information: bool = False
    established_echo_excess_growth_fraction: float | None = None
    established_echo_excess_growth_fraction_by_time: (
        tuple[float, float] | None
    ) = None
    maximum_growth_envelope_ratio: float | None = None
    maximum_growth_envelope_ratio_by_time: tuple[float, float] | None = None
    amplitude_diagnostics_source: AmplitudeDiagnosticsSource = "unavailable"
    relative_objective_reduction: float | None = None
    causal_control_cell_count: int = 0
    causal_seed_cell_count: int = 0
    causal_seed_prior_cost: float = 0.0
    dynamics_data_gram_eigenvalues: tuple[float, float, float] | None = None
    dynamics_data_information_trace: float | None = None
    regularized_dynamics_hessian_eigenvalues: (
        tuple[float, float, float] | None
    ) = None
    regularized_dynamics_hessian_condition_number: float | None = None
    field_smoothness_prior_cost: float = 0.0
    motion_saturation_margin_yx: tuple[float, float] | None = None
    motion_speed_saturation_margin_mps: float | None = None
    growth_saturation_margin: float | None = None
    field_growth_jacobian_cosine: float | None = None
    field_motion_jacobian_cosine_by_control: (
        tuple[float | None, float | None] | None
    ) = None
    amplitude_confidence_failed: bool = False
    precursor_object_count_by_time: tuple[int, int] | None = None
    insufficient_amplitude_object_count_by_time: tuple[int, int] | None = None
    maximum_object_unresolved_fraction_by_time: (
        tuple[float, float] | None
    ) = None
    minimum_object_integrated_echo_ratio_by_time: (
        tuple[float, float] | None
    ) = None
    maximum_object_integrated_echo_ratio_by_time: (
        tuple[float, float] | None
    ) = None
    minimum_object_soft_echo_area_ratio_by_time: (
        tuple[float, float] | None
    ) = None
    maximum_object_soft_echo_area_ratio_by_time: (
        tuple[float, float] | None
    ) = None
    dynamics_data_numerical_rank: int | None = None
    dynamics_data_effective_dimension: float | None = None
    dynamics_data_to_prior_ratio_by_mode: (
        tuple[float, float, float] | None
    ) = None
    motion_control_coordinate_system: str = "grid_yx_px"
    field_smoothness_coordinate_system: str = "index_graph"


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
    grid_time_contract: RadarGridTimeContract | None = None,
) -> tuple[AnalysisObservations, FrozenOuterState]:
    nowcast_config = nowcast_config or NowcastConfig()
    analysis_config = analysis_config or AnalysisConfig()
    if analysis_config.motion_increment_scale_mps is not None:
        if grid_time_contract is None:
            raise ValueError(
                "motion_increment_scale_mps requires a grid/time contract"
            )
        if nowcast_config.maximum_motion_speed_mps is None:
            raise ValueError(
                "motion_increment_scale_mps requires a physical motion limit"
            )
    if (
        grid_time_contract is not None
        and analysis_config.field_smoothness_weight > 0.0
        and not grid_time_contract.grid_axes_are_orthogonal
    ):
        raise ValueError(
            "field smoothness requires orthogonal projected grid axes"
        )
    if analysis_config.execution_mode == "operational":
        if grid_time_contract is None:
            raise ValueError(
                "operational analysis requires a grid/time contract"
            )
        if nowcast_config.maximum_motion_speed_mps is None:
            raise ValueError(
                "operational analysis requires a physical motion limit"
            )
        if (
            nowcast_config.pair_echo_dilation_m is None
            or nowcast_config.phase_correlation_sidelobe_radius_m is None
        ):
            raise ValueError(
                "operational analysis requires physical pair confidence settings"
            )
        if (
            analysis_config.causal_support_uncertainty_m is None
            or analysis_config.amplitude_displacement_tolerance_m is None
        ):
            raise ValueError(
                "operational analysis requires physical causal and amplitude "
                "distance settings"
            )
    _validate_frames(frames_dbz)
    motion_limits = motion_displacement_limits_yx(
        nowcast_config,
        grid_time_contract,
        frames_dbz,
    )
    maximum_radius_yx = (frames_dbz.shape[1] - 1, frames_dbz.shape[2] - 1)
    if analysis_config.causal_support_uncertainty_m is not None:
        if grid_time_contract is None:
            raise ValueError(
                "causal_support_uncertainty_m requires a grid/time contract"
            )
        causal_dilation_offsets = grid_time_contract.pixel_offsets_within_distance(
            analysis_config.causal_support_uncertainty_m,
            maximum_radius_yx=maximum_radius_yx,
        )
    else:
        causal_dilation_offsets = _rectangular_offsets_yx(
            analysis_config.causal_support_dilation_px,
            analysis_config.causal_support_dilation_px,
        )
    if analysis_config.amplitude_displacement_tolerance_m is not None:
        if grid_time_contract is None:
            raise ValueError(
                "amplitude_displacement_tolerance_m requires a grid/time contract"
            )
        amplitude_tolerance_offsets = (
            grid_time_contract.pixel_offsets_within_distance(
                analysis_config.amplitude_displacement_tolerance_m,
                maximum_radius_yx=maximum_radius_yx,
            )
        )
    else:
        amplitude_tolerance_offsets = _rectangular_offsets_yx(
            analysis_config.amplitude_displacement_tolerance_px,
            analysis_config.amplitude_displacement_tolerance_px,
        )
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
    )
    _validate_observations(observations)

    baseline_state, baseline_metadata = estimate_prepared_state(
        prepared,
        nowcast_config,
        grid_time_contract=grid_time_contract,
    )
    baseline_state = _detach_state(baseline_state)
    baseline_metadata = _detach_metadata(baseline_metadata)
    initial_support, causal_seed = _causal_control_and_seed_support(
        detected,
        prepared.observed_mask,
        prepared.background_mask,
        baseline_state.displacement_yx,
        analysis_config.minimum_control_reachability,
        causal_dilation_offsets,
    )
    causal_only = initial_support & ~detected[0]
    active_field_index = torch.nonzero(
        initial_support.flatten(),
        as_tuple=False,
    ).flatten()
    (
        smooth_edge_left_index,
        smooth_edge_right_index,
        smooth_edge_physical_weight,
    ) = _active_smoothness_graph(
        initial_support,
        active_field_index,
        frames_dbz,
        grid_time_contract,
    )
    remap_cells = (
        freeze_remap_cell(baseline_state.displacement_yx),
        freeze_remap_cell(2 * baseline_state.displacement_yx),
    )
    baseline_frames_dbz = torch.where(
        prepared.observed_mask,
        prepared.frames_dbz,
        prepared.background_frames_dbz,
    )
    frozen = FrozenOuterState(
        initial_background_dbz=baseline_frames_dbz[0].detach().clone(),
        initial_support_mask=initial_support.detach().clone(),
        active_field_index=active_field_index.detach().clone(),
        causal_only_mask=causal_only.detach().clone(),
        causal_seed_mask=causal_seed.detach().clone(),
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
        grid_time_contract=grid_time_contract,
        motion_limits_yx=motion_limits.detach().clone(),
        amplitude_displacement_offsets_yx=amplitude_tolerance_offsets,
        analysis_remap_cells=remap_cells,
        smooth_edge_left_index=smooth_edge_left_index,
        smooth_edge_right_index=smooth_edge_right_index,
        smooth_edge_physical_weight=smooth_edge_physical_weight,
    )
    control = _warm_started_control(observations, frozen)
    return observations, freeze_irls_weights(
        control,
        observations,
        frozen,
    )


def _causal_control_and_seed_support(
    detected_mask: Tensor,
    observed_mask: Tensor,
    background_mask: Tensor,
    displacement_yx: Tensor,
    minimum_reachability: float,
    dilation_offsets_yx: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor]:
    initial_anchor = observed_mask[0] | background_mask[0]
    precursor_core = torch.zeros_like(detected_mask[0])
    for step in (1, 2):
        precursor = remap(
            detected_mask[step].to(dtype=displacement_yx.dtype),
            -step * displacement_yx,
        )
        precursor_core |= precursor >= minimum_reachability
    control_envelope = _footprint_maximum(
        precursor_core.to(dtype=displacement_yx.dtype),
        dilation_offsets_yx,
    ) > 0
    control_support = (
        detected_mask[0] | control_envelope
    ) & initial_anchor
    seed_support = (
        precursor_core & initial_anchor & ~detected_mask[0]
    )
    return control_support, seed_support


def _active_smoothness_graph(
    active_mask: Tensor,
    active_field_index: Tensor,
    reference: Tensor,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[Tensor, Tensor, Tensor]:
    height, width = active_mask.shape
    active_lookup = torch.full(
        (height * width,),
        -1,
        dtype=torch.long,
        device=active_mask.device,
    )
    active_lookup[active_field_index] = torch.arange(
        active_field_index.numel(),
        dtype=torch.long,
        device=active_mask.device,
    )
    global_index = torch.arange(
        height * width,
        dtype=torch.long,
        device=active_mask.device,
    ).reshape(height, width)
    vertical = active_mask[1:] & active_mask[:-1]
    horizontal = active_mask[:, 1:] & active_mask[:, :-1]
    left_global = torch.cat(
        (global_index[:-1][vertical], global_index[:, :-1][horizontal])
    )
    right_global = torch.cat(
        (global_index[1:][vertical], global_index[:, 1:][horizontal])
    )
    left = active_lookup[left_global]
    right = active_lookup[right_global]
    vertical_count = int(torch.count_nonzero(vertical))
    if grid_time_contract is None:
        physical_weight = reference.new_ones(left.numel())
    else:
        assert grid_time_contract.pixel_to_projected_matrix_m is not None
        (xx, xr), (yx, yr) = (
            grid_time_contract.pixel_to_projected_matrix_m
        )
        cell_area = grid_time_contract.cell_area_m2
        column_length_squared = xx * xx + yx * yx
        row_length_squared = xr * xr + yr * yr
        vertical_weight = column_length_squared / cell_area
        horizontal_weight = row_length_squared / cell_area
        physical_weight = torch.cat(
            (
                reference.new_full((vertical_count,), vertical_weight),
                reference.new_full(
                    (left.numel() - vertical_count,),
                    horizontal_weight,
                ),
            )
        )
    return left, right, physical_weight


def initial_control(frozen: FrozenOuterState) -> Tensor:
    return frozen.initial_background_dbz.new_zeros(
        frozen.active_field_index.numel() + 3
    )


def _warm_started_control(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    _validate_observations(observations)
    control = initial_control(frozen)
    seed_control = _precursor_seed_control(frozen)
    field_size = frozen.active_field_index.numel()
    control[:field_size] = seed_control.flatten()[frozen.active_field_index]
    return control


def _precursor_seed_control(frozen: FrozenOuterState) -> Tensor:
    seed_control = torch.zeros_like(frozen.initial_background_dbz)
    if not bool(torch.any(frozen.causal_seed_mask)):
        return seed_control
    config = frozen.analysis_config
    floor_dbz = frozen.nowcast_config.min_dbz
    seed_dbz = max(
        config.detection_limit_dbz - config.censor_temperature_dbz,
        0.5 * (floor_dbz + config.detection_limit_dbz),
    )
    seed_mask = frozen.causal_seed_mask & (
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
    field_size = frozen.active_field_index.numel()
    field_control = torch.zeros_like(
        frozen.initial_background_dbz,
    ).flatten().scatter(
        0,
        frozen.active_field_index,
        control[:field_size],
    ).reshape(height, width)
    dynamics_control = control[field_size:]
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
        frozen.motion_limits_yx,
        frozen.grid_time_contract,
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
    return torch.cat(
        (
            weighted.reshape(-1),
            control,
            _field_smoothness_residual(control, frozen),
        )
    )


def _field_smoothness_residual(
    control: Tensor,
    frozen: FrozenOuterState,
) -> Tensor:
    field_size = frozen.active_field_index.numel()
    weight = frozen.analysis_config.field_smoothness_weight
    if (
        field_size == 0
        or weight == 0.0
        or frozen.smooth_edge_left_index.numel() == 0
    ):
        return control.new_zeros(0)
    field = control[:field_size]
    differences = (
        field[frozen.smooth_edge_right_index]
        - field[frozen.smooth_edge_left_index]
    )
    return torch.sqrt(
        weight * frozen.smooth_edge_physical_weight
    ) * differences


def _field_smoothness_prior_cost(
    control: Tensor,
    frozen: FrozenOuterState,
) -> Tensor:
    residual = _field_smoothness_residual(control, frozen)
    return 0.5 * torch.dot(residual, residual)


def _scaled_dot(left: Tensor, right: Tensor) -> float:
    left_scale = float(torch.amax(torch.abs(left)).detach())
    right_scale = float(torch.amax(torch.abs(right)).detach())
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    normalized = torch.dot(left / left_scale, right / right_scale)
    return float(normalized.detach()) * left_scale * right_scale


def _absolute_jacobian_cosine(
    left: Tensor,
    right: Tensor,
) -> float | None:
    left_scale = float(torch.amax(torch.abs(left)).detach())
    right_scale = float(torch.amax(torch.abs(right)).detach())
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    left_scaled = left / left_scale
    right_scaled = right / right_scale
    denominator = float(
        (
            torch.linalg.vector_norm(left_scaled)
            * torch.linalg.vector_norm(right_scaled)
        ).detach()
    )
    if denominator == 0.0 or not math.isfinite(denominator):
        return None
    cosine = (
        abs(float(torch.dot(left_scaled, right_scaled).detach()))
        / denominator
    )
    if not math.isfinite(cosine):
        return None
    return min(1.0, cosine)


def _field_identifiability_directions(
    control: Tensor,
    frozen: FrozenOuterState,
    trajectory: AnalysisTrajectory,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    field_size = frozen.active_field_index.numel()
    if field_size == 0:
        return None, None, None

    height, width = frozen.initial_background_dbz.shape
    dense_field_control = torch.zeros_like(
        frozen.initial_background_dbz
    ).flatten().scatter(
        0,
        frozen.active_field_index,
        control[:field_size],
    ).reshape(height, width)
    config = frozen.analysis_config
    background_offset = (
        frozen.initial_background_dbz - frozen.nowcast_config.min_dbz
    ) / config.echo_transform_scale_dbz
    background_latent = _softplus_inverse(
        background_offset.clamp_min(config.transform_epsilon)
    )
    transform_derivative = config.initial_increment_scale_dbz * torch.sigmoid(
        background_latent
        + (
            config.initial_increment_scale_dbz
            / config.echo_transform_scale_dbz
        )
        * dense_field_control
    )
    transform_derivative = transform_derivative.clamp_min(
        config.initial_increment_scale_dbz * config.transform_epsilon
    )
    initial_dbz = echo_to_dbz(
        trajectory.frames_linear[0],
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    )

    gradient_y = torch.zeros_like(initial_dbz)
    if height > 1:
        gradient_y[0] = initial_dbz[1] - initial_dbz[0]
        gradient_y[-1] = initial_dbz[-1] - initial_dbz[-2]
    if height > 2:
        gradient_y[1:-1] = 0.5 * (initial_dbz[2:] - initial_dbz[:-2])

    gradient_x = torch.zeros_like(initial_dbz)
    if width > 1:
        gradient_x[:, 0] = initial_dbz[:, 1] - initial_dbz[:, 0]
        gradient_x[:, -1] = initial_dbz[:, -1] - initial_dbz[:, -2]
    if width > 2:
        gradient_x[:, 1:-1] = 0.5 * (
            initial_dbz[:, 2:] - initial_dbz[:, :-2]
        )

    def pack(field_values: Tensor) -> Tensor | None:
        active_values = field_values.flatten()[frozen.active_field_index]
        norm = float(torch.linalg.vector_norm(active_values).detach())
        if norm == 0.0 or not math.isfinite(norm):
            return None
        direction = torch.zeros_like(control)
        direction[:field_size] = active_values / norm
        return direction

    return (
        pack(torch.ones_like(initial_dbz) / transform_derivative),
        pack(-gradient_y / transform_derivative),
        pack(-gradient_x / transform_derivative),
    )


def _identifiability_diagnostics(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    trajectory: AnalysisTrajectory,
) -> _IdentifiabilityDiagnostics | None:
    frozen = freeze_irls_weights(control, observations, frozen)
    field_size = frozen.active_field_index.numel()

    residual_fn: Callable[[Tensor], Tensor] = lambda value: (
        (
            _whitened_observation_residual(value, observations, frozen)
            * frozen.irls_sqrt_weight
        ).reshape(-1)
    )

    def jacobian_vector(direction: Tensor) -> Tensor:
        result = torch.func.jvp(
            residual_fn,
            (control,),
            (direction,),
        )
        return cast(Tensor, result[1]).detach()

    dynamics_columns: list[Tensor] = []
    for dynamics_index in range(3):
        direction = torch.zeros_like(control)
        direction[field_size + dynamics_index] = 1.0
        column = jacobian_vector(direction)
        if not bool(torch.all(torch.isfinite(column))):
            return None
        dynamics_columns.append(column)

    gram_values = [
        [
            _scaled_dot(dynamics_columns[row], dynamics_columns[column])
            for column in range(3)
        ]
        for row in range(3)
    ]
    gram = torch.tensor(gram_values, dtype=torch.float64)
    gram = 0.5 * (gram + gram.mT)
    if not bool(torch.all(torch.isfinite(gram))):
        return None
    raw_data_eigenvalues = torch.linalg.eigvalsh(gram)
    if not bool(torch.all(torch.isfinite(raw_data_eigenvalues))):
        return None
    gram_scale = max(
        1.0,
        float(torch.amax(torch.abs(gram)).detach()),
    )
    negative_tolerance = 64.0 * torch.finfo(torch.float64).eps * gram_scale
    if float(raw_data_eigenvalues[0]) < -negative_tolerance:
        return None
    data_eigenvalues = torch.clamp_min(raw_data_eigenvalues, 0.0)
    data_information_trace = float(torch.sum(data_eigenvalues))
    maximum_data_eigenvalue = float(data_eigenvalues[-1])
    if maximum_data_eigenvalue == 0.0:
        data_numerical_rank = 0
    else:
        rank_tolerance = (
            3.0
            * torch.finfo(torch.float64).eps
            * maximum_data_eigenvalue
        )
        data_numerical_rank = int(
            torch.count_nonzero(data_eigenvalues > rank_tolerance)
        )
    data_to_prior_ratio = data_eigenvalues / (1.0 + data_eigenvalues)
    data_effective_dimension = float(torch.sum(data_to_prior_ratio))

    regularized_hessian = gram + torch.eye(3, dtype=torch.float64)
    regularized_eigenvalues = torch.linalg.eigvalsh(regularized_hessian)
    if not bool(torch.all(torch.isfinite(regularized_eigenvalues))):
        return None
    minimum_eigenvalue = float(regularized_eigenvalues[0])
    maximum_eigenvalue = float(regularized_eigenvalues[-1])
    if minimum_eigenvalue <= 0.0:
        return None

    field_scale, field_shift_y, field_shift_x = (
        _field_identifiability_directions(control, frozen, trajectory)
    )
    motion_field_directions = _motion_field_identifiability_directions(
        control,
        frozen,
        field_shift_y,
        field_shift_x,
    )

    def cosine(
        field_direction: Tensor | None,
        dynamics_column: Tensor,
    ) -> float | None:
        if field_direction is None:
            return None
        field_column = jacobian_vector(field_direction)
        if not bool(torch.all(torch.isfinite(field_column))):
            return None
        return _absolute_jacobian_cosine(field_column, dynamics_column)

    return _IdentifiabilityDiagnostics(
        dynamics_data_gram_eigenvalues=(
            float(data_eigenvalues[0]),
            float(data_eigenvalues[1]),
            float(data_eigenvalues[2]),
        ),
        dynamics_data_information_trace=data_information_trace,
        dynamics_data_numerical_rank=data_numerical_rank,
        dynamics_data_effective_dimension=data_effective_dimension,
        dynamics_data_to_prior_ratio_by_mode=(
            float(data_to_prior_ratio[0]),
            float(data_to_prior_ratio[1]),
            float(data_to_prior_ratio[2]),
        ),
        regularized_dynamics_hessian_eigenvalues=(
            float(regularized_eigenvalues[0]),
            float(regularized_eigenvalues[1]),
            float(regularized_eigenvalues[2]),
        ),
        regularized_dynamics_hessian_condition_number=(
            maximum_eigenvalue / minimum_eigenvalue
        ),
        field_growth_jacobian_cosine=cosine(
            field_scale,
            dynamics_columns[2],
        ),
        field_motion_jacobian_cosine_by_control=(
            cosine(motion_field_directions[0], dynamics_columns[0]),
            cosine(motion_field_directions[1], dynamics_columns[1]),
        ),
    )


def _motion_field_identifiability_directions(
    control: Tensor,
    frozen: FrozenOuterState,
    field_shift_y: Tensor | None,
    field_shift_x: Tensor | None,
) -> tuple[Tensor | None, Tensor | None]:
    if field_shift_y is None or field_shift_x is None:
        return None, None
    field_size = frozen.active_field_index.numel()
    dynamics_control = control[field_size:]

    def displacement(value: Tensor) -> Tensor:
        return _decode_dynamics(
            value,
            frozen.baseline_state,
            frozen.analysis_config,
            frozen.nowcast_config,
            frozen.motion_limits_yx,
            frozen.grid_time_contract,
        )[0]

    directions: list[Tensor] = []
    for index in range(2):
        basis = torch.zeros_like(dynamics_control)
        basis[index] = 1.0
        tangent = cast(
            Tensor,
            torch.func.jvp(
                displacement,
                (dynamics_control,),
                (basis,),
            )[1],
        )
        directions.append(
            tangent[0] * field_shift_y + tangent[1] * field_shift_x
        )
    return directions[0], directions[1]


def robust_objective(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    return _robust_objective_from_residual(
        control,
        _whitened_observation_residual(control, observations, frozen),
        observations,
        frozen,
    )


def _robust_objective_from_residual(
    control: Tensor,
    residual: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    config = frozen.analysis_config
    delta = config.pseudo_huber_delta
    robust = delta**2 * (
        torch.sqrt(1.0 + (residual / delta).square()) - 1.0
    )
    robust = torch.where(
        observations.valid_mask,
        robust,
        torch.zeros_like(robust),
    )
    return (
        robust.sum()
        + 0.5 * torch.dot(control, control)
        + _field_smoothness_prior_cost(control, frozen)
    )


def solve_analysis(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    control: Tensor | None = None,
) -> AnalysisResult:
    _validate_observations(observations)
    reference_control = initial_control(frozen)
    _validate_control(reference_control, frozen)
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

    if control is None:
        control = _warm_started_control(observations, frozen)
    else:
        control = control.detach().clone()
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
    current_amplitude = _amplitude_diagnostics(
        observations,
        frozen,
        current_trajectory,
        include_spatial_diagnostics=False,
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
    if (
        current_amplitude.has_insufficient_information
        and config.amplitude_information_policy == "operational_fallback"
    ):
        return _fallback_result(
            frozen,
            control,
            reference_cost,
            "insufficient_amplitude_information",
            amplitude_diagnostics=current_amplitude,
            amplitude_diagnostics_source="rejected_candidate",
        )
    field_size = frozen.active_field_index.numel()
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
        linearization_point: Tensor = control
        residual_fn: Callable[[Tensor], Tensor] = lambda value: (
            residual_vector(
                value,
                observations,
                frozen_iteration,
            )
        )
        vjp_result = torch.func.vjp(residual_fn, linearization_point)
        residual = cast(Tensor, vjp_result[0])
        pullback = cast(
            Callable[[Tensor], tuple[Tensor]],
            vjp_result[1],
        )

        def normal_product(vector: Tensor) -> Tensor:
            jvp_result = torch.func.jvp(
                residual_fn,
                (linearization_point,),
                (vector,),
            )
            jacobian_vector = cast(Tensor, jvp_result[1])
            return pullback(jacobian_vector)[0]

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
            operator: Callable[[Tensor], Tensor] = lambda vector: (
                normal_product(vector) + damping * vector
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
            hessian_step = normal_product(raw_step)
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
                    frozen_iteration.motion_limits_yx,
                    frozen_iteration.grid_time_contract,
                )
                if (
                    config.motion_increment_scale_mps is None
                    and not _motion_is_admissible(
                        candidate_displacement,
                        frozen_iteration,
                    )
                ):
                    continue
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
                candidate_amplitude = _amplitude_diagnostics(
                    observations,
                    candidate_frozen,
                    candidate_trajectory,
                    include_spatial_diagnostics=False,
                )
                if not _amplitude_trial_is_admissible(
                    current_amplitude,
                    candidate_amplitude,
                    config.maximum_unresolved_amplitude_fraction,
                    control.dtype,
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
                current_amplitude = candidate_amplitude
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
    grid_time_contract: RadarGridTimeContract | None = None,
    audit: bool = False,
) -> tuple[ForecastResult, AnalysisResult]:
    nowcast_config = nowcast_config or NowcastConfig()
    if grid_time_contract is not None:
        grid_time_contract.validate_for(
            nowcast_config,
            background_present=background_frames_dbz is not None,
            background_age_minutes=background_age_minutes,
        )
    observations, frozen = prepare_analysis(
        frames_dbz,
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
        observation_std_dbz=observation_std_dbz,
        quality_weight=quality_weight,
        qc_mask=qc_mask,
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
        grid_time_contract=grid_time_contract,
    )
    analysis = solve_analysis(observations, frozen)
    (
        analysis_config_json,
        analysis_config_digest,
        analysis_input_digest,
    ) = _analysis_input_lineage(observations, frozen.analysis_config)
    run = ForecastRunContract.from_inputs(
        nowcast_config,
        frames_dbz,
        observations.valid_mask,
        background_frames_dbz,
        background_age_minutes,
        grid_time_contract=grid_time_contract,
        analysis_config_json=analysis_config_json,
        analysis_config_digest=analysis_config_digest,
        analysis_input_digest=analysis_input_digest,
    )
    forecast = forecast_from_state(
        analysis.state,
        analysis.metadata,
        nowcast_config,
        run=run,
        audit=audit,
    )
    return forecast, analysis


def _analysis_input_lineage(
    observations: AnalysisObservations,
    config: AnalysisConfig,
) -> tuple[str, str, str]:
    config_value = asdict(config)
    config_json = json.dumps(
        config_value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    config_digest = json_digest(config_value)
    input_digest = json_digest(
        {
            "version": "p1-analysis-input-v1",
            "analysis_config_digest": config_digest,
            "observation_std_dbz": tensor_digest(observations.std_dbz),
            "quality_weight": tensor_digest(observations.quality_weight),
        }
    )
    return config_json, config_digest, input_digest


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
            frozen,
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
    amplitude = _amplitude_diagnostics(
        observations,
        frozen,
        trajectory,
    )
    if (
        amplitude.has_insufficient_information
        and frozen.analysis_config.amplitude_information_policy
        == "operational_fallback"
    ):
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "insufficient_amplitude_information",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=reachability_margin,
            amplitude_diagnostics=amplitude,
            amplitude_diagnostics_source="rejected_candidate",
        )
    if (
        float(amplitude.maximum_gated_unresolved_fraction.detach())
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
            amplitude_diagnostics=amplitude,
            amplitude_diagnostics_source="rejected_candidate",
        )
    amplitude_confidence_failed = amplitude.degrades_confidence(
        frozen.analysis_config
    )
    if (
        amplitude_confidence_failed
        and frozen.analysis_config.amplitude_confidence_policy
        == "operational_fallback"
    ):
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "amplitude_confidence_failure",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=reachability_margin,
            amplitude_diagnostics=amplitude,
            amplitude_diagnostics_source="rejected_candidate",
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
            amplitude_diagnostics=amplitude,
            amplitude_diagnostics_source="rejected_candidate",
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
    background_used = (
        background_fraction > frozen.nowcast_config.epsilon
        or frozen.baseline_metadata.background_tendency_used
    )
    (
        causal_control_cell_count,
        causal_seed_cell_count,
        causal_seed_prior_cost,
    ) = _causal_seed_diagnostics(frozen)
    identifiability = _identifiability_diagnostics(
        control,
        observations,
        frozen,
        trajectory,
    )
    field_smoothness_prior_cost = float(
        _field_smoothness_prior_cost(control, frozen).detach()
    )
    motion_saturation_margin = (
        frozen.motion_limits_yx
        - torch.abs(trajectory.displacement_yx)
    )
    motion_speed_saturation_margin = _motion_speed_saturation_margin(
        trajectory.displacement_yx,
        frozen,
    )
    growth_saturation_margin = (
        frozen.nowcast_config.max_log_growth_per_step
        - torch.abs(trajectory.log_growth_per_step)
    )
    return AnalysisResult(
        control=control.detach(),
        active_field_index=frozen.active_field_index.detach().clone(),
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
        degraded=(
            degraded
            or amplitude.has_insufficient_information
            or amplitude_confidence_failed
        ),
        audit=audit,
        minimum_reachability_margin=reachability_margin,
        unresolved_amplitude_fraction=float(
            amplitude.maximum_unresolved_fraction.detach()
        ),
        unresolved_amplitude_fraction_by_time=(
            _materialize_pair(amplitude.unresolved_fraction_by_time)
        ),
        unresolved_pixel_fraction_by_time=(
            _materialize_pair(amplitude.unresolved_pixel_fraction_by_time)
        ),
        amplitude_violation_score=float(
            amplitude.maximum_violation_score.detach()
        ),
        amplitude_violation_score_by_time=_materialize_pair(
            amplitude.violation_score_by_time
        ),
        integrated_echo_ratio_by_time=_materialize_pair(
            amplitude.integrated_echo_ratio_by_time
        ),
        displacement_tolerant_soft_echo_area_ratio_by_time=_materialize_pair(
            amplitude.displacement_tolerant_soft_echo_area_ratio_by_time
        ),
        effective_precursor_pixel_count_by_time=_materialize_pair(
            amplitude.effective_pixel_count_by_time
        ),
        bad_quality_weight_by_time=_materialize_pair(
            amplitude.bad_quality_weight_by_time
        ),
        total_quality_weight_by_time=_materialize_pair(
            amplitude.total_quality_weight_by_time
        ),
        amplitude_information_sufficient_by_time=_materialize_bool_pair(
            amplitude.information_sufficient_by_time
        ),
        insufficient_amplitude_information=(
            amplitude.has_insufficient_information
        ),
        amplitude_confidence_failed=amplitude_confidence_failed,
        precursor_object_count_by_time=_materialize_int_pair(
            amplitude.precursor_object_count_by_time
        ),
        insufficient_amplitude_object_count_by_time=_materialize_int_pair(
            amplitude.insufficient_object_count_by_time
        ),
        maximum_object_unresolved_fraction_by_time=_materialize_pair(
            amplitude.maximum_object_unresolved_fraction_by_time
        ),
        minimum_object_integrated_echo_ratio_by_time=_materialize_pair(
            amplitude.minimum_object_integrated_echo_ratio_by_time
        ),
        maximum_object_integrated_echo_ratio_by_time=_materialize_pair(
            amplitude.maximum_object_integrated_echo_ratio_by_time
        ),
        minimum_object_soft_echo_area_ratio_by_time=_materialize_pair(
            amplitude.minimum_object_soft_echo_area_ratio_by_time
        ),
        maximum_object_soft_echo_area_ratio_by_time=_materialize_pair(
            amplitude.maximum_object_soft_echo_area_ratio_by_time
        ),
        established_echo_excess_growth_fraction=(
            _materialize_finite_max(
                amplitude.established_echo_excess_growth_fraction_by_time
            )
        ),
        established_echo_excess_growth_fraction_by_time=_materialize_pair(
            amplitude.established_echo_excess_growth_fraction_by_time
        ),
        maximum_growth_envelope_ratio=_materialize_finite_max(
            amplitude.maximum_growth_envelope_ratio_by_time
        ),
        maximum_growth_envelope_ratio_by_time=_materialize_pair(
            amplitude.maximum_growth_envelope_ratio_by_time
        ),
        amplitude_diagnostics_source="returned_analysis",
        relative_objective_reduction=_relative_objective_reduction(
            reference_objective,
            final_objective,
        ),
        causal_control_cell_count=causal_control_cell_count,
        causal_seed_cell_count=causal_seed_cell_count,
        causal_seed_prior_cost=causal_seed_prior_cost,
        dynamics_data_gram_eigenvalues=(
            None
            if identifiability is None
            else identifiability.dynamics_data_gram_eigenvalues
        ),
        dynamics_data_information_trace=(
            None
            if identifiability is None
            else identifiability.dynamics_data_information_trace
        ),
        dynamics_data_numerical_rank=(
            None
            if identifiability is None
            else identifiability.dynamics_data_numerical_rank
        ),
        dynamics_data_effective_dimension=(
            None
            if identifiability is None
            else identifiability.dynamics_data_effective_dimension
        ),
        dynamics_data_to_prior_ratio_by_mode=(
            None
            if identifiability is None
            else identifiability.dynamics_data_to_prior_ratio_by_mode
        ),
        regularized_dynamics_hessian_eigenvalues=(
            None
            if identifiability is None
            else identifiability.regularized_dynamics_hessian_eigenvalues
        ),
        regularized_dynamics_hessian_condition_number=(
            None
            if identifiability is None
            else identifiability.regularized_dynamics_hessian_condition_number
        ),
        field_smoothness_prior_cost=field_smoothness_prior_cost,
        motion_saturation_margin_yx=(
            float(motion_saturation_margin[0]),
            float(motion_saturation_margin[1]),
        ),
        motion_speed_saturation_margin_mps=(
            motion_speed_saturation_margin
        ),
        growth_saturation_margin=float(growth_saturation_margin),
        field_growth_jacobian_cosine=(
            None
            if identifiability is None
            else identifiability.field_growth_jacobian_cosine
        ),
        field_motion_jacobian_cosine_by_control=(
            None
            if identifiability is None
            else identifiability.field_motion_jacobian_cosine_by_control
        ),
        motion_control_coordinate_system=(
            "projected_xy_mps_radial_ball"
            if frozen.analysis_config.motion_increment_scale_mps is not None
            else "grid_yx_px"
        ),
        field_smoothness_coordinate_system=(
            "projected_orthogonal_graph"
            if frozen.grid_time_contract is not None
            else "index_graph"
        ),
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
    maximum = _amplitude_diagnostics(
        observations,
        frozen,
        trajectory,
    ).maximum_unresolved_fraction
    return float(maximum.detach())


def _rectangular_offsets_yx(
    radius_y: int,
    radius_x: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (row, column)
        for row in range(-radius_y, radius_y + 1)
        for column in range(-radius_x, radius_x + 1)
    )


def _footprint_maximum(
    value: Tensor,
    offsets_yx: tuple[tuple[int, int], ...],
) -> Tensor:
    if value.ndim != 2:
        raise ValueError("footprint maximum requires a two-dimensional tensor")
    if not offsets_yx or (0, 0) not in offsets_yx:
        raise ValueError("offset footprint must contain its origin")
    radius_y = max(abs(row) for row, _ in offsets_yx)
    radius_x = max(abs(column) for _, column in offsets_yx)
    padded = F.pad(
        value,
        (radius_x, radius_x, radius_y, radius_y),
        value=-math.inf,
    )
    height, width = value.shape
    shifted = tuple(
        padded[
            radius_y + row : radius_y + row + height,
            radius_x + column : radius_x + column + width,
        ]
        for row, column in offsets_yx
    )
    result = shifted[0]
    for candidate in shifted[1:]:
        result = torch.maximum(result, candidate)
    return result


def _amplitude_diagnostics(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    trajectory: AnalysisTrajectory,
    *,
    include_spatial_diagnostics: bool = True,
) -> _AmplitudeDiagnostics:
    prediction_dbz = echo_to_dbz(
        trajectory.frames_linear,
        min_dbz=frozen.nowcast_config.min_dbz,
    )
    (
        established_excess_growth_fractions,
        maximum_growth_envelope_ratios,
    ) = _established_growth_envelope_diagnostics(
        observations,
        frozen,
        trajectory,
        enabled=include_spatial_diagnostics,
    )
    initial_detected = frozen.detected_masks[0].to(
        dtype=trajectory.displacement_yx.dtype
    )
    amplitude_floor = (
        frozen.analysis_config.detection_limit_dbz
        - frozen.analysis_config.censor_temperature_dbz
    )
    unresolved_fractions: list[Tensor] = []
    unresolved_pixel_fractions: list[Tensor] = []
    violation_scores: list[Tensor] = []
    integrated_echo_ratios: list[Tensor] = []
    soft_echo_area_ratios: list[Tensor] = []
    effective_pixel_counts: list[Tensor] = []
    bad_quality_weights: list[Tensor] = []
    total_quality_weights: list[Tensor] = []
    information_sufficient: list[Tensor] = []
    precursor_object_counts: list[Tensor] = []
    insufficient_object_counts: list[Tensor] = []
    maximum_object_unresolved_fractions: list[Tensor] = []
    minimum_object_integrated_echo_ratios: list[Tensor] = []
    maximum_object_integrated_echo_ratios: list[Tensor] = []
    minimum_object_soft_echo_area_ratios: list[Tensor] = []
    maximum_object_soft_echo_area_ratios: list[Tensor] = []
    zero = prediction_dbz.new_zeros(())
    nan = prediction_dbz.new_full((), math.nan)

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
            unresolved_fractions.append(zero)
            unresolved_pixel_fractions.append(zero)
            violation_scores.append(zero)
            integrated_echo_ratios.append(nan)
            soft_echo_area_ratios.append(nan)
            effective_pixel_counts.append(zero)
            bad_quality_weights.append(zero)
            total_quality_weights.append(zero)
            information_sufficient.append(
                torch.ones_like(zero, dtype=torch.bool)
            )
            precursor_object_counts.append(zero.to(dtype=torch.long))
            insufficient_object_counts.append(zero.to(dtype=torch.long))
            maximum_object_unresolved_fractions.append(zero)
            minimum_object_integrated_echo_ratios.append(nan)
            maximum_object_integrated_echo_ratios.append(nan)
            minimum_object_soft_echo_area_ratios.append(nan)
            maximum_object_soft_echo_area_ratios.append(nan)
            continue

        local_prediction = _footprint_maximum(
            prediction_dbz[step],
            frozen.amplitude_displacement_offsets_yx,
        )
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
        selected_quality = quality[precursor_required]
        bad_weight = quality[unresolved].sum()
        total_weight = selected_quality.sum()
        relative_quality = selected_quality / selected_quality.max()
        effective_pixel_count = relative_quality.sum().square() / (
            relative_quality.square().sum()
        )
        sufficient = (
            total_weight
            >= frozen.analysis_config.minimum_amplitude_total_quality_weight
        ) & (
            effective_pixel_count
            >= frozen.analysis_config.minimum_amplitude_effective_pixel_count
        )
        unresolved_fraction = bad_weight / total_weight
        unresolved_pixel_fraction = (
            unresolved[precursor_required].to(dtype=prediction_dbz.dtype).mean()
        )

        standardized_excess = torch.clamp_min(
            standardized_deficit
            - frozen.analysis_config.maximum_detected_error_std,
            0.0,
        )
        floor_excess = torch.clamp_min(
            torch.sqrt(quality)
            * (amplitude_floor - local_prediction)
            / observations.std_dbz[step],
            0.0,
        )
        violation = (
            standardized_excess[precursor_required].square()
            + floor_excess[precursor_required].square()
        ).sum() / effective_pixel_count

        integrated_echo_ratio = nan
        soft_echo_area_ratio = nan
        if include_spatial_diagnostics:
            expanded_region = (
                _footprint_maximum(
                    precursor_required.to(dtype=prediction_dbz.dtype),
                    frozen.amplitude_displacement_offsets_yx,
                )
                > 0
            )
            observed_echo = dbz_to_echo(
                observations.dbz[step],
                min_dbz=frozen.nowcast_config.min_dbz,
                max_dbz=frozen.nowcast_config.max_dbz,
            )
            observed_echo_integral = observed_echo[precursor_required].sum()
            predicted_echo_integral = trajectory.frames_linear[step][
                expanded_region
            ].sum()
            integrated_echo_ratio = (
                predicted_echo_integral / observed_echo_integral
            )

            temperature = frozen.analysis_config.censor_temperature_dbz
            observed_soft_area = torch.sigmoid(
                (
                    observations.dbz[step]
                    - frozen.analysis_config.detection_limit_dbz
                )
                / temperature
            )[precursor_required].sum()
            predicted_soft_area = torch.sigmoid(
                (
                    prediction_dbz[step]
                    - frozen.analysis_config.detection_limit_dbz
                )
                / temperature
            )[expanded_region].sum()
            soft_echo_area_ratio = (
                predicted_soft_area / observed_soft_area
            )

        (
            object_count,
            insufficient_object_count,
            maximum_object_unresolved_fraction,
            minimum_object_integrated_echo_ratio,
            maximum_object_integrated_echo_ratio,
            minimum_object_soft_echo_area_ratio,
            maximum_object_soft_echo_area_ratio,
        ) = _precursor_object_diagnostics(
            precursor_required,
            unresolved,
            quality,
            observations.dbz[step],
            prediction_dbz[step],
            trajectory.frames_linear[step],
            frozen,
            enabled=include_spatial_diagnostics,
        )

        unresolved_fractions.append(unresolved_fraction)
        unresolved_pixel_fractions.append(unresolved_pixel_fraction)
        violation_scores.append(violation)
        integrated_echo_ratios.append(integrated_echo_ratio)
        soft_echo_area_ratios.append(soft_echo_area_ratio)
        effective_pixel_counts.append(effective_pixel_count)
        bad_quality_weights.append(bad_weight)
        total_quality_weights.append(total_weight)
        information_sufficient.append(sufficient)
        precursor_object_counts.append(object_count)
        insufficient_object_counts.append(insufficient_object_count)
        maximum_object_unresolved_fractions.append(
            maximum_object_unresolved_fraction
        )
        minimum_object_integrated_echo_ratios.append(
            minimum_object_integrated_echo_ratio
        )
        maximum_object_integrated_echo_ratios.append(
            maximum_object_integrated_echo_ratio
        )
        minimum_object_soft_echo_area_ratios.append(
            minimum_object_soft_echo_area_ratio
        )
        maximum_object_soft_echo_area_ratios.append(
            maximum_object_soft_echo_area_ratio
        )

    return _AmplitudeDiagnostics(
        unresolved_fraction_by_time=torch.stack(unresolved_fractions),
        unresolved_pixel_fraction_by_time=torch.stack(
            unresolved_pixel_fractions
        ),
        violation_score_by_time=torch.stack(violation_scores),
        integrated_echo_ratio_by_time=torch.stack(integrated_echo_ratios),
        displacement_tolerant_soft_echo_area_ratio_by_time=torch.stack(
            soft_echo_area_ratios
        ),
        effective_pixel_count_by_time=torch.stack(effective_pixel_counts),
        bad_quality_weight_by_time=torch.stack(bad_quality_weights),
        total_quality_weight_by_time=torch.stack(total_quality_weights),
        information_sufficient_by_time=torch.stack(information_sufficient),
        established_echo_excess_growth_fraction_by_time=(
            established_excess_growth_fractions
        ),
        maximum_growth_envelope_ratio_by_time=(
            maximum_growth_envelope_ratios
        ),
        precursor_object_count_by_time=torch.stack(precursor_object_counts),
        insufficient_object_count_by_time=torch.stack(
            insufficient_object_counts
        ),
        maximum_object_unresolved_fraction_by_time=torch.stack(
            maximum_object_unresolved_fractions
        ),
        minimum_object_integrated_echo_ratio_by_time=torch.stack(
            minimum_object_integrated_echo_ratios
        ),
        maximum_object_integrated_echo_ratio_by_time=torch.stack(
            maximum_object_integrated_echo_ratios
        ),
        minimum_object_soft_echo_area_ratio_by_time=torch.stack(
            minimum_object_soft_echo_area_ratios
        ),
        maximum_object_soft_echo_area_ratio_by_time=torch.stack(
            maximum_object_soft_echo_area_ratios
        ),
    )


def _precursor_object_diagnostics(
    precursor_required: Tensor,
    unresolved: Tensor,
    quality: Tensor,
    observed_dbz: Tensor,
    prediction_dbz: Tensor,
    prediction_echo: Tensor,
    frozen: FrozenOuterState,
    *,
    enabled: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    zero = prediction_dbz.new_zeros(())
    nan = prediction_dbz.new_full((), math.nan)
    if not enabled:
        count = zero.to(dtype=torch.long)
        return count, count.clone(), zero, nan, nan, nan, nan

    components = _connected_component_flat_indices(precursor_required)
    object_count = zero.new_tensor(len(components), dtype=torch.long)
    insufficient_count = zero.to(dtype=torch.long)
    unresolved_fractions: list[Tensor] = []
    integrated_echo_ratios: list[Tensor] = []
    soft_echo_area_ratios: list[Tensor] = []
    flat_quality = quality.flatten()
    flat_unresolved = unresolved.flatten()
    flat_observed_dbz = observed_dbz.flatten()
    observed_echo = dbz_to_echo(
        observed_dbz,
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    ).flatten()
    temperature = frozen.analysis_config.censor_temperature_dbz

    for indices in components:
        selected_quality = flat_quality[indices]
        total_weight = selected_quality.sum()
        maximum_quality = selected_quality.max().clamp_min(
            frozen.analysis_config.transform_epsilon
        )
        relative_quality = selected_quality / maximum_quality
        effective_count = relative_quality.sum().square() / (
            relative_quality.square().sum().clamp_min(
                frozen.analysis_config.transform_epsilon
            )
        )
        sufficient = bool(
            (
                total_weight
                >= frozen.analysis_config.minimum_amplitude_total_quality_weight
            )
            & (
                effective_count
                >= frozen.analysis_config.minimum_amplitude_effective_pixel_count
            )
        )
        if not sufficient:
            insufficient_count = insufficient_count + 1
            continue

        unresolved_fractions.append(
            selected_quality[flat_unresolved[indices]].sum() / total_weight
        )
        object_mask = torch.zeros_like(precursor_required)
        object_mask.flatten()[indices] = True
        expanded = _footprint_maximum(
            object_mask.to(dtype=prediction_dbz.dtype),
            frozen.amplitude_displacement_offsets_yx,
        ) > 0
        integrated_echo_ratios.append(
            prediction_echo[expanded].sum() / observed_echo[indices].sum()
        )
        observed_soft_area = torch.sigmoid(
            (
                flat_observed_dbz[indices]
                - frozen.analysis_config.detection_limit_dbz
            )
            / temperature
        ).sum()
        predicted_soft_area = torch.sigmoid(
            (
                prediction_dbz
                - frozen.analysis_config.detection_limit_dbz
            )
            / temperature
        )[expanded].sum()
        soft_echo_area_ratios.append(predicted_soft_area / observed_soft_area)

    if not unresolved_fractions:
        return object_count, insufficient_count, zero, nan, nan, nan, nan
    unresolved_values = torch.stack(unresolved_fractions)
    echo_values = torch.stack(integrated_echo_ratios)
    area_values = torch.stack(soft_echo_area_ratios)
    return (
        object_count,
        insufficient_count,
        torch.max(unresolved_values),
        torch.min(echo_values),
        torch.max(echo_values),
        torch.min(area_values),
        torch.max(area_values),
    )


def _connected_component_flat_indices(mask: Tensor) -> tuple[Tensor, ...]:
    _, width = mask.shape
    remaining = {
        (int(row), int(column))
        for row, column in torch.nonzero(
            mask.detach().cpu(),
            as_tuple=False,
        ).tolist()
    }
    components: list[Tensor] = []
    while remaining:
        pending = deque((remaining.pop(),))
        flat_indices: list[int] = []
        while pending:
            row, column = pending.popleft()
            flat_indices.append(row * width + column)
            for delta_row in (-1, 0, 1):
                for delta_column in (-1, 0, 1):
                    neighbor = (row + delta_row, column + delta_column)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
        components.append(
            torch.tensor(
                flat_indices,
                dtype=torch.long,
                device=mask.device,
            )
        )
    return tuple(components)


def _established_growth_envelope_diagnostics(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    trajectory: AnalysisTrajectory,
    *,
    enabled: bool,
) -> tuple[Tensor, Tensor]:
    unavailable = trajectory.frames_linear.new_full((2,), math.nan)
    if not enabled or not bool(torch.any(frozen.detected_masks[0])):
        return unavailable, unavailable.clone()

    nowcast = frozen.nowcast_config
    analysis = frozen.analysis_config
    initial_quality = observations.quality_weight[0].clamp_min(
        analysis.transform_epsilon
    )
    initial_upper_dbz = (
        observations.dbz[0]
        + analysis.maximum_detected_error_std
        * observations.std_dbz[0]
        / torch.sqrt(initial_quality)
    ).clamp_max(nowcast.max_dbz)
    initial_upper_dbz = torch.where(
        frozen.detected_masks[0],
        initial_upper_dbz,
        initial_upper_dbz.new_full((), nowcast.min_dbz),
    )
    initial_upper_echo = dbz_to_echo(
        initial_upper_dbz,
        min_dbz=nowcast.min_dbz,
        max_dbz=nowcast.max_dbz,
    )
    initial_detected = frozen.detected_masks[0].to(
        dtype=trajectory.displacement_yx.dtype
    )
    excess_fractions: list[Tensor] = []
    maximum_ratios: list[Tensor] = []
    nan = trajectory.frames_linear.new_full((), math.nan)

    for index, step in enumerate((1, 2)):
        initial_reach = remap(
            initial_detected,
            step * trajectory.displacement_yx,
        )
        established = frozen.detected_masks[step] & (
            initial_reach >= analysis.minimum_control_reachability
        )
        if not bool(torch.any(established)):
            excess_fractions.append(nan)
            maximum_ratios.append(nan)
            continue

        envelope_echo = advance(
            initial_upper_echo,
            step * trajectory.displacement_yx,
            step * nowcast.max_log_growth_per_step,
            frozen.analysis_remap_cells[index],
        )
        local_envelope_echo = _footprint_maximum(
            envelope_echo,
            frozen.amplitude_displacement_offsets_yx,
        )
        local_envelope_dbz = echo_to_dbz(
            local_envelope_echo,
            min_dbz=nowcast.min_dbz,
            max_dbz=nowcast.max_dbz,
        )
        quality = observations.quality_weight[step]
        standardized_excess = (
            torch.sqrt(quality)
            * (observations.dbz[step] - local_envelope_dbz)
            / observations.std_dbz[step]
        )
        excess = established & (
            standardized_excess > analysis.maximum_detected_error_std
        )
        excess_fractions.append(
            quality[excess].sum() / quality[established].sum()
        )
        observed_echo = dbz_to_echo(
            observations.dbz[step],
            min_dbz=nowcast.min_dbz,
            max_dbz=nowcast.max_dbz,
        )
        maximum_ratios.append(
            torch.max(
                observed_echo[established]
                / local_envelope_echo[established].clamp_min(nowcast.epsilon)
            )
        )

    return torch.stack(excess_fractions), torch.stack(maximum_ratios)


def _amplitude_trial_is_admissible(
    current: _AmplitudeDiagnostics,
    candidate: _AmplitudeDiagnostics,
    maximum_fraction: float,
    dtype: torch.dtype,
) -> bool:
    metrics = torch.stack(
        (
            current.maximum_gated_unresolved_fraction,
            candidate.maximum_gated_unresolved_fraction,
            current.maximum_gated_violation_score,
            candidate.maximum_gated_violation_score,
            current.total_gated_violation_score,
            candidate.total_gated_violation_score,
        )
    ).detach().cpu()
    current_fraction = float(metrics[0])
    candidate_fraction = float(metrics[1])
    current_maximum = float(metrics[2])
    candidate_maximum = float(metrics[3])
    current_total = float(metrics[4])
    candidate_total = float(metrics[5])
    if candidate_fraction <= maximum_fraction:
        return True
    if current_fraction <= maximum_fraction:
        return False
    info = torch.finfo(dtype)
    maximum_tolerance = (
        32.0
        * info.eps
        * max(abs(current_maximum), abs(candidate_maximum), info.tiny)
    )
    if candidate_maximum < current_maximum - maximum_tolerance:
        return True
    if candidate_maximum > current_maximum + maximum_tolerance:
        return False
    total_tolerance = (
        32.0
        * info.eps
        * max(abs(current_total), abs(candidate_total), info.tiny)
    )
    return candidate_total < current_total - total_tolerance


def _materialize_pair(values: Tensor) -> tuple[float, float]:
    values = values.detach().cpu()
    return float(values[0]), float(values[1])


def _materialize_bool_pair(values: Tensor) -> tuple[bool, bool]:
    values = values.detach().cpu()
    return bool(values[0]), bool(values[1])


def _materialize_int_pair(values: Tensor) -> tuple[int, int]:
    values = values.detach().cpu()
    return int(values[0]), int(values[1])


def _materialize_finite_max(values: Tensor) -> float | None:
    values = values.detach().cpu()
    finite = values[torch.isfinite(values)]
    return None if finite.numel() == 0 else float(torch.max(finite))


def _relative_objective_reduction(
    reference_objective: float,
    final_objective: float,
) -> float | None:
    if not (
        math.isfinite(reference_objective)
        and math.isfinite(final_objective)
    ):
        return None
    return (reference_objective - final_objective) / max(
        abs(reference_objective),
        torch.finfo(torch.float64).eps,
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
    amplitude_diagnostics: _AmplitudeDiagnostics | None = None,
    amplitude_diagnostics_source: AmplitudeDiagnosticsSource = "unavailable",
) -> AnalysisResult:
    if (
        amplitude_diagnostics is None
        and amplitude_diagnostics_source != "unavailable"
    ):
        raise ValueError(
            "amplitude diagnostics source requires amplitude diagnostics"
        )
    if (
        amplitude_diagnostics is not None
        and amplitude_diagnostics_source == "unavailable"
    ):
        raise ValueError(
            "stored amplitude diagnostics require an explicit source"
        )
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
        active_field_index=frozen.active_field_index.detach().clone(),
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
        unresolved_amplitude_fraction=(
            None
            if amplitude_diagnostics is None
            else float(
                amplitude_diagnostics.maximum_unresolved_fraction.detach()
            )
        ),
        unresolved_amplitude_fraction_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.unresolved_fraction_by_time
            )
        ),
        unresolved_pixel_fraction_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.unresolved_pixel_fraction_by_time
            )
        ),
        amplitude_violation_score=(
            None
            if amplitude_diagnostics is None
            else float(amplitude_diagnostics.maximum_violation_score.detach())
        ),
        amplitude_violation_score_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.violation_score_by_time
            )
        ),
        integrated_echo_ratio_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.integrated_echo_ratio_by_time
            )
        ),
        displacement_tolerant_soft_echo_area_ratio_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics
                .displacement_tolerant_soft_echo_area_ratio_by_time
            )
        ),
        effective_precursor_pixel_count_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.effective_pixel_count_by_time
            )
        ),
        bad_quality_weight_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.bad_quality_weight_by_time
            )
        ),
        total_quality_weight_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.total_quality_weight_by_time
            )
        ),
        amplitude_information_sufficient_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_bool_pair(
                amplitude_diagnostics.information_sufficient_by_time
            )
        ),
        insufficient_amplitude_information=(
            False
            if amplitude_diagnostics is None
            else amplitude_diagnostics.has_insufficient_information
        ),
        amplitude_confidence_failed=(
            False
            if amplitude_diagnostics is None
            else amplitude_diagnostics.degrades_confidence(
                frozen.analysis_config
            )
        ),
        precursor_object_count_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_int_pair(
                amplitude_diagnostics.precursor_object_count_by_time
            )
        ),
        insufficient_amplitude_object_count_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_int_pair(
                amplitude_diagnostics.insufficient_object_count_by_time
            )
        ),
        maximum_object_unresolved_fraction_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics
                .maximum_object_unresolved_fraction_by_time
            )
        ),
        minimum_object_integrated_echo_ratio_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics
                .minimum_object_integrated_echo_ratio_by_time
            )
        ),
        maximum_object_integrated_echo_ratio_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics
                .maximum_object_integrated_echo_ratio_by_time
            )
        ),
        minimum_object_soft_echo_area_ratio_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics
                .minimum_object_soft_echo_area_ratio_by_time
            )
        ),
        maximum_object_soft_echo_area_ratio_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics
                .maximum_object_soft_echo_area_ratio_by_time
            )
        ),
        established_echo_excess_growth_fraction=(
            None
            if amplitude_diagnostics is None
            else _materialize_finite_max(
                amplitude_diagnostics
                .established_echo_excess_growth_fraction_by_time
            )
        ),
        established_echo_excess_growth_fraction_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics
                .established_echo_excess_growth_fraction_by_time
            )
        ),
        maximum_growth_envelope_ratio=(
            None
            if amplitude_diagnostics is None
            else _materialize_finite_max(
                amplitude_diagnostics.maximum_growth_envelope_ratio_by_time
            )
        ),
        maximum_growth_envelope_ratio_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.maximum_growth_envelope_ratio_by_time
            )
        ),
        amplitude_diagnostics_source=amplitude_diagnostics_source,
        relative_objective_reduction=(
            None
            if not math.isfinite(initial_objective)
            else 0.0
        ),
        causal_control_cell_count=causal_control_cell_count,
        causal_seed_cell_count=causal_seed_cell_count,
        causal_seed_prior_cost=causal_seed_prior_cost,
        motion_control_coordinate_system=(
            "projected_xy_mps_radial_ball"
            if frozen.analysis_config.motion_increment_scale_mps is not None
            else "grid_yx_px"
        ),
        field_smoothness_coordinate_system=(
            "projected_orthogonal_graph"
            if frozen.grid_time_contract is not None
            else "index_graph"
        ),
    )


def _motion_speed_saturation_margin(
    displacement_yx: Tensor,
    frozen: FrozenOuterState,
) -> float | None:
    maximum_speed = frozen.nowcast_config.maximum_motion_speed_mps
    contract = frozen.grid_time_contract
    if maximum_speed is None or contract is None:
        return None
    projected = contract.projected_displacement_xy(displacement_yx)
    speed = torch.linalg.vector_norm(projected) / (
        frozen.nowcast_config.interval_minutes * 60.0
    )
    return maximum_speed - float(speed.detach())


def _motion_is_admissible(
    displacement_yx: Tensor,
    frozen: FrozenOuterState,
) -> bool:
    margin = _motion_speed_saturation_margin(displacement_yx, frozen)
    return margin is None or margin >= -frozen.nowcast_config.epsilon


def _bounded_update(
    background: Tensor,
    control: Tensor,
    scale: float,
    limit: float | Tensor,
) -> Tensor:
    limit_tensor = torch.as_tensor(
        limit,
        dtype=background.dtype,
        device=background.device,
    )
    if bool(torch.any(limit_tensor < 0)):
        raise ValueError("bounded update limit cannot be negative")
    if bool(torch.all(limit_tensor == 0)):
        return torch.zeros_like(background)
    safe_limit = limit_tensor.clamp_min(torch.finfo(background.dtype).tiny)
    inside = torch.abs(background) < safe_limit
    unit = background.new_tensor(1.0)
    interior_limit = torch.nextafter(unit, background.new_zeros(()))
    ratio = (background / safe_limit).clamp(
        -interior_limit,
        interior_limit,
    )
    latent = torch.atanh(ratio)
    updated = limit_tensor * torch.tanh(
        latent + (scale / safe_limit) * control
    )
    projected = (background + scale * control).clamp(
        -limit_tensor,
        limit_tensor,
    )
    return torch.where(inside, updated, projected)


def _bounded_vector_update(
    background: Tensor,
    control: Tensor,
    scale: float,
    limit: float,
) -> Tensor:
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("bounded vector update limit must be positive")
    limit_tensor = background.new_tensor(limit)
    tiny = torch.finfo(background.dtype).tiny
    background_norm = torch.linalg.vector_norm(background)
    background_direction = background / background_norm.clamp_min(tiny)
    inside = background_norm < limit_tensor
    unit = limit_tensor.new_tensor(1.0)
    interior_limit = torch.nextafter(unit, limit_tensor.new_zeros(()))
    background_ratio = (background_norm / limit_tensor).clamp(
        max=interior_limit
    )
    latent = (
        torch.atanh(background_ratio) * background_direction
        + (scale / limit_tensor) * control
    )
    latent_norm = torch.linalg.vector_norm(latent)
    safe_norm = latent_norm.clamp_min(tiny)
    radial_factor = torch.where(
        latent_norm > math.sqrt(torch.finfo(background.dtype).eps),
        torch.tanh(latent_norm) / safe_norm,
        1.0 - latent_norm.square() / 3.0,
    )
    updated = limit_tensor * radial_factor * latent
    candidate = background + scale * control
    candidate_norm = torch.linalg.vector_norm(candidate)
    projected = candidate * torch.clamp(
        limit_tensor / candidate_norm.clamp_min(tiny),
        max=1.0,
    )
    return torch.where(inside, updated, projected)


def _decode_dynamics(
    dynamics_control: Tensor,
    baseline: RadarState,
    config: AnalysisConfig,
    nowcast: NowcastConfig,
    motion_limits_yx: Tensor,
    grid_time_contract: RadarGridTimeContract | None,
) -> tuple[Tensor, Tensor]:
    if config.motion_increment_scale_mps is None:
        displacement = _bounded_update(
            baseline.displacement_yx,
            dynamics_control[:2],
            config.motion_increment_scale_px,
            motion_limits_yx,
        )
    else:
        if (
            grid_time_contract is None
            or nowcast.maximum_motion_speed_mps is None
        ):
            raise ValueError(
                "physical motion control requires grid/time and speed limits"
            )
        baseline_velocity = grid_time_contract.projected_velocity_xy(
            baseline.displacement_yx,
            nowcast.interval_minutes,
        )
        projected_velocity = _bounded_vector_update(
            baseline_velocity,
            dynamics_control[:2],
            config.motion_increment_scale_mps,
            nowcast.maximum_motion_speed_mps,
        )
        displacement = (
            grid_time_contract.displacement_yx_from_projected_velocity(
                projected_velocity,
                nowcast.interval_minutes,
            )
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
    field_size = frozen.active_field_index.numel()
    displacement, _ = _decode_dynamics(
        control[field_size:],
        frozen.baseline_state,
        frozen.analysis_config,
        frozen.nowcast_config,
        frozen.motion_limits_yx,
        frozen.grid_time_contract,
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
    active_index = frozen.active_field_index
    expected = active_index.numel() + 3
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
    expected_index = torch.nonzero(
        frozen.initial_support_mask.flatten(),
        as_tuple=False,
    ).flatten()
    if (
        active_index.ndim != 1
        or active_index.dtype != torch.long
        or active_index.device != frozen.initial_background_dbz.device
        or not torch.equal(active_index, expected_index)
    ):
        raise ValueError(
            "active_field_index must enumerate initial support in flat order"
        )


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
        motion_disagreement_mps=metadata.motion_disagreement_mps.detach(),
        growth_disagreement=metadata.growth_disagreement.detach(),
        minimum_phase_correlation_psr=(
            metadata.minimum_phase_correlation_psr.detach()
        ),
        tendency_pair_count=metadata.tendency_pair_count,
        tendency_source=metadata.tendency_source,
        provenance=metadata.provenance,
        motion_pair_count=metadata.motion_pair_count,
        growth_pair_count=metadata.growth_pair_count,
        motion_pair_selection=metadata.motion_pair_selection,
        growth_pair_selection=metadata.growth_pair_selection,
        motion_pair_conflict=metadata.motion_pair_conflict,
        growth_pair_conflict=metadata.growth_pair_conflict,
    )
