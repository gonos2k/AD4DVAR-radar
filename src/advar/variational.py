from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from collections import deque
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import io
import json
import math
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._digest import json_digest, tensor_digest
from .calibration import (
    OperationalCalibrationManifest,
    OperationalDataIdentity,
    algorithm_bundle_digest,
)
from .diagnostics import EchoPositivityError, PositivityAudit, validate_physical_echo
from .matrix_free import pcg
from .nowcast import (
    DynamicsSource,
    ForecastMetadata,
    ForecastResult,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    StatePathProvenance,
    TendencyPairSelection,
    TendencySource,
    _local_component_evidence_from_pair_spans,
    estimate_prepared_state,
    forecast_from_state,
    merge_current_support,
    motion_displacement_limits_yx,
    prepare_input,
    state_metadata_digest,
)
from .physics import (
    RemapCell,
    advance,
    dbz_to_echo,
    echo_to_dbz,
    freeze_remap_cell,
    remap,
)
from ._runtime import numerical_runtime_identity_digest


AmplitudeInformationPolicy = Literal[
    "research_degraded",
    "operational_fallback",
]
AmplitudeConfidencePolicy = Literal[
    "research_degraded",
    "operational_fallback",
]
AnalysisExecutionMode = Literal["research", "operational"]
ObservationCommonBiasScope = Literal["per_frame", "all_times"]
CensoredBackgroundPolicy = Literal[
    "detection_limit",
    "floor",
    "external_background",
]
MAXIMUM_OBSERVATION_COMMON_BIAS_MODE_COUNT = 64
P1_LINEARIZATION_CONTRACT = "p1-final-frozen-irls-gn-v14"
P1_LINEARIZATION_DIGEST_CONTRACT = "p1-linearization-digest-v12"
_WHITENER_APPLY_COUNTER: ContextVar[list[int] | None] = ContextVar(
    "advar_whitener_apply_counter",
    default=None,
)
_WHITENER_TOTAL_OPERATION_LIMIT: ContextVar[tuple[int, int] | None] = (
    ContextVar(
        "advar_whitener_total_operation_limit",
        default=None,
    )
)


@contextmanager
def _count_observation_whitener_applies(
    *,
    operations_per_apply: int = 0,
    maximum_total_operations: int | None = None,
) -> Generator[list[int], None, None]:
    """Count and optionally limit low-rank whitener applications."""

    if operations_per_apply < 0:
        raise ValueError("whitener operations per apply cannot be negative")
    if maximum_total_operations is not None and maximum_total_operations <= 0:
        raise ValueError("whitener total operation limit must be positive")

    counter = [0]
    counter_token = _WHITENER_APPLY_COUNTER.set(counter)
    limit_token = _WHITENER_TOTAL_OPERATION_LIMIT.set(
        None
        if maximum_total_operations is None or operations_per_apply == 0
        else (operations_per_apply, maximum_total_operations)
    )
    try:
        yield counter
    finally:
        _WHITENER_TOTAL_OPERATION_LIMIT.reset(limit_token)
        _WHITENER_APPLY_COUNTER.reset(counter_token)


def _observation_whitener_operations_per_apply(
    observations: AnalysisObservations,
) -> int:
    weights = observations.common_bias_mode_weights
    if weights is None:
        return 0
    frame_multiplier = observations.dbz.shape[0] if weights.ndim == 3 else 1
    return 2 * frame_multiplier * weights.numel()


@dataclass(frozen=True)
class AnalysisConfig:
    detection_limit_dbz: float = 5.0
    censor_temperature_dbz: float = 1.0
    censored_background_policy: CensoredBackgroundPolicy = "floor"
    observation_std_dbz: float = 2.0
    minimum_observation_std_dbz: float = 0.1
    observation_common_bias_std_dbz: float = 0.0
    observation_common_bias_scope: ObservationCommonBiasScope = "per_frame"
    observation_common_bias_tile_size_px: int = 0
    observation_common_bias_group_map_digest: str | None = None
    observation_common_bias_mode_weights_digest: str | None = None
    maximum_common_bias_mode_weight_bytes: int = 2 * 1024**3
    maximum_common_bias_whitener_apply_operations: int = 256 * 1024**2
    maximum_common_bias_gram_multiply_adds: int = 8 * 1024**3
    maximum_frozen_whitener_bytes: int = 512 * 1024**2
    maximum_linearization_bytes: int = 8 * 1024**3
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
    minimum_local_verification_precision: float = 0.01
    maximum_local_analysis_verification_error_dbz: float = 6.0
    maximum_unresolved_amplitude_fraction: float = 0.01
    minimum_amplitude_total_quality_weight: float = 0.01
    minimum_amplitude_effective_pixel_count: float = 1.0
    amplitude_information_policy: AmplitudeInformationPolicy = (
        "research_degraded"
    )
    minimum_integrated_echo_ratio_for_confidence: float = 0.5
    minimum_soft_echo_area_ratio_for_confidence: float = 0.5
    maximum_established_excess_growth_fraction_for_confidence: float = 0.01
    minimum_object_count_ratio_for_confidence: float = 0.75
    field_smoothness_weight: float = 0.01
    maximum_outer_iterations: int = 4
    maximum_pcg_iterations: int = 40
    maximum_damping_retries: int = 2
    pcg_relative_tolerance: float = 1.0e-5
    gradient_tolerance: float = 1.0e-5
    step_tolerance: float = 1.0e-4
    final_linearization_relative_stationarity_tolerance: float = 2.0e-4
    final_robust_relative_stationarity_tolerance: float = 2.0e-4
    final_field_gradient_max_tolerance: float = 1.0e-2
    final_irls_relative_weight_tolerance: float = 1.0e-4
    maximum_final_linearization_polish_iterations: int = 4
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
            "minimum_local_verification_precision": (
                self.minimum_local_verification_precision
            ),
            "maximum_local_analysis_verification_error_dbz": (
                self.maximum_local_analysis_verification_error_dbz
            ),
            "minimum_amplitude_total_quality_weight": (
                self.minimum_amplitude_total_quality_weight
            ),
            "minimum_amplitude_effective_pixel_count": (
                self.minimum_amplitude_effective_pixel_count
            ),
            "pcg_relative_tolerance": self.pcg_relative_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
            "step_tolerance": self.step_tolerance,
            "final_linearization_relative_stationarity_tolerance": (
                self.final_linearization_relative_stationarity_tolerance
            ),
            "final_robust_relative_stationarity_tolerance": (
                self.final_robust_relative_stationarity_tolerance
            ),
            "final_field_gradient_max_tolerance": (
                self.final_field_gradient_max_tolerance
            ),
            "final_irls_relative_weight_tolerance": (
                self.final_irls_relative_weight_tolerance
            ),
            "initial_damping": self.initial_damping,
            "minimum_damping": self.minimum_damping,
            "maximum_damping": self.maximum_damping,
        }
        if not math.isfinite(self.detection_limit_dbz):
            raise ValueError("detection_limit_dbz must be finite")
        if self.censored_background_policy not in (
            "detection_limit",
            "floor",
            "external_background",
        ):
            raise ValueError(
                "censored_background_policy must be detection_limit, floor, "
                "or external_background"
            )
        if (
            not isinstance(self.observation_common_bias_std_dbz, (int, float))
            or isinstance(self.observation_common_bias_std_dbz, bool)
            or not math.isfinite(self.observation_common_bias_std_dbz)
            or self.observation_common_bias_std_dbz < 0.0
        ):
            raise ValueError(
                "observation_common_bias_std_dbz must be finite and "
                "nonnegative"
            )
        if self.observation_common_bias_scope not in (
            "per_frame",
            "all_times",
        ):
            raise ValueError(
                "observation_common_bias_scope must be per_frame or all_times"
            )
        if (
            type(self.observation_common_bias_tile_size_px) is not int
            or self.observation_common_bias_tile_size_px < 0
        ):
            raise ValueError(
                "observation_common_bias_tile_size_px must be a "
                "nonnegative integer"
            )
        if self.observation_common_bias_group_map_digest is not None and (
            not isinstance(
                self.observation_common_bias_group_map_digest,
                str,
            )
            or len(self.observation_common_bias_group_map_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.observation_common_bias_group_map_digest
            )
        ):
            raise ValueError(
                "observation_common_bias_group_map_digest must be a "
                "lowercase SHA-256 digest"
            )
        if self.observation_common_bias_mode_weights_digest is not None and (
            not isinstance(
                self.observation_common_bias_mode_weights_digest,
                str,
            )
            or len(self.observation_common_bias_mode_weights_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.observation_common_bias_mode_weights_digest
            )
        ):
            raise ValueError(
                "observation_common_bias_mode_weights_digest must be a "
                "lowercase SHA-256 digest"
            )
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        integer_limits = {
            "maximum_outer_iterations": self.maximum_outer_iterations,
            "maximum_pcg_iterations": self.maximum_pcg_iterations,
            "maximum_common_bias_mode_weight_bytes": (
                self.maximum_common_bias_mode_weight_bytes
            ),
            "maximum_common_bias_whitener_apply_operations": (
                self.maximum_common_bias_whitener_apply_operations
            ),
            "maximum_common_bias_gram_multiply_adds": (
                self.maximum_common_bias_gram_multiply_adds
            ),
            "maximum_frozen_whitener_bytes": (
                self.maximum_frozen_whitener_bytes
            ),
            "maximum_linearization_bytes": self.maximum_linearization_bytes,
        }
        for name, value in integer_limits.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            type(self.maximum_final_linearization_polish_iterations) is not int
            or self.maximum_final_linearization_polish_iterations < 0
        ):
            raise ValueError(
                "maximum_final_linearization_polish_iterations must be a "
                "nonnegative integer"
            )
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
            "minimum_object_count_ratio_for_confidence": (
                self.minimum_object_count_ratio_for_confidence
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
class CommonBiasResourceEstimate:
    """Allocation-free cost estimate for overlapping bias modes."""

    mode_shape: tuple[int, ...]
    frame_shape: tuple[int, int, int]
    dtype: str
    mode_count: int
    retained_mode_bytes: int
    whitener_operations_per_apply: int
    gram_multiply_adds: int
    correction_bytes: int
    frozen_whitener_bytes: int
    within_budget: bool
    rejection_reasons: tuple[str, ...]


def estimate_common_bias_resources(
    mode_shape: tuple[int, ...],
    frame_shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    temporal_scope: ObservationCommonBiasScope,
    config: AnalysisConfig | None = None,
) -> CommonBiasResourceEstimate:
    """Estimate dense overlapping-mode cost before allocating its Tensor."""

    config = config or AnalysisConfig()
    if (
        len(frame_shape) != 3
        or frame_shape[0] != 3
        or any(type(value) is not int or value <= 0 for value in frame_shape)
    ):
        raise ValueError("frame_shape must be positive [3,H,W]")
    if any(type(value) is not int or value <= 0 for value in mode_shape):
        raise ValueError("mode_shape dimensions must be positive integers")
    if len(mode_shape) == 3:
        mode_count, height, width = mode_shape
        frame_multiplier = frame_shape[0]
    elif len(mode_shape) == 4 and mode_shape[0] == frame_shape[0]:
        _, mode_count, height, width = mode_shape
        frame_multiplier = 1
    else:
        raise ValueError("mode_shape must be [K,H,W] or [3,K,H,W]")
    if (height, width) != frame_shape[1:]:
        raise ValueError("mode and frame spatial shapes must agree")
    item_sizes = {torch.float32: 4, torch.float64: 8}
    if dtype not in item_sizes:
        raise TypeError("common-bias resource dtype must be float32 or float64")
    if temporal_scope not in ("per_frame", "all_times"):
        raise ValueError("unsupported common-bias temporal scope")
    item_size = item_sizes[dtype]
    mode_elements = math.prod(mode_shape)
    retained_bytes = mode_elements * item_size
    operations_per_apply = 2 * frame_multiplier * mode_elements
    gram_operations = (
        frame_shape[0] * mode_count * mode_count * height * width
    )
    correction_count = (
        frame_shape[0] if temporal_scope == "per_frame" else 1
    )
    correction_bytes = correction_count * mode_count * mode_count * item_size
    frozen_bytes = (
        math.prod(frame_shape) * item_size + correction_bytes
    )
    reasons: list[str] = []
    if not 1 <= mode_count <= MAXIMUM_OBSERVATION_COMMON_BIAS_MODE_COUNT:
        reasons.append("mode_count")
    if retained_bytes > config.maximum_common_bias_mode_weight_bytes:
        reasons.append("retained_mode_bytes")
    if (
        operations_per_apply
        > config.maximum_common_bias_whitener_apply_operations
    ):
        reasons.append("whitener_operations_per_apply")
    if gram_operations > config.maximum_common_bias_gram_multiply_adds:
        reasons.append("gram_multiply_adds")
    if frozen_bytes > config.maximum_frozen_whitener_bytes:
        reasons.append("frozen_whitener_bytes")
    return CommonBiasResourceEstimate(
        mode_shape=mode_shape,
        frame_shape=frame_shape,
        dtype=str(dtype),
        mode_count=mode_count,
        retained_mode_bytes=retained_bytes,
        whitener_operations_per_apply=operations_per_apply,
        gram_multiply_adds=gram_operations,
        correction_bytes=correction_bytes,
        frozen_whitener_bytes=frozen_bytes,
        within_budget=not reasons,
        rejection_reasons=tuple(reasons),
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
    common_bias_group_index: Tensor | None = None
    common_bias_mode_weights: Tensor | None = None


PriorDependency = Literal["exogenous", "radar_dependent"]
PriorSupportPolicy = Literal["causal_clip", "expand_control"]


def _require_prior_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _module_state_digest(model: nn.Module) -> str:
    state = model.state_dict()
    if not state:
        raise ValueError("neural-prior model must have retained state")
    return json_digest(
        {
            "contract": "neural-prior-model-state-v1",
            "state": {
                name: tensor_digest(value.detach())
                for name, value in sorted(state.items())
            },
        }
    )


class _FeatureGraph(nn.Module):
    exclusion_mask: Tensor

    def __init__(
        self,
        feature_extractor: Callable[[Tensor], Tensor],
        exclusion_mask: Tensor,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.register_buffer("exclusion_mask", exclusion_mask)
        self.exclusion_mask = exclusion_mask

    def forward(self, frames: Tensor) -> Tensor:
        retained = torch.where(
            cast(Tensor, self.exclusion_mask),
            torch.zeros((), dtype=frames.dtype, device=frames.device),
            frames,
        )
        return self.feature_extractor(retained)


class _PriorPipeline(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Callable[[Tensor], Tensor],
        exclusion_mask: Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.feature_graph = _FeatureGraph(feature_extractor, exclusion_mask)

    def forward(self, frames: Tensor) -> tuple[Tensor, ...]:
        features = self.feature_graph(frames)
        raw = self.model(features)
        if isinstance(raw, tuple):
            return (features, *raw)
        return features, raw


def _export_graph(module: nn.Module, argument: Tensor) -> tuple[nn.Module, str]:
    program = torch.export.export(module, (argument,))
    target = io.BytesIO()
    torch.export.save(program, target)
    digest = json_digest(
        {
            "contract": "torch-exported-graph-v1",
            "bytes": target.getvalue().hex(),
        }
    )
    return program.module(), digest


@dataclass(frozen=True)
class NeuralPriorInferenceEvidence:
    """Reproducible evidence for one concrete prior inference."""

    neural_prior_digest: str
    input_bundle_digest: str
    input_frames_digest: str
    feature_tensor_digest: str
    feature_extractor_digest: str
    feature_extractor_code_digest: str
    model_code_digest: str
    exported_graph_digest: str
    model_artifact_digest: str
    model_contract_digest: str
    feature_schema_digest: str
    training_manifest_digest: str
    inference_algorithm_digest: str
    numerical_runtime_digest: str
    output_background_digest: str
    output_std_digest: str
    output_valid_mask_digest: str
    output_valid_probability_digest: str
    output_support_probability_digest: str
    prior_output_valid_time: str | None
    feature_source_valid_times: tuple[str, ...]
    feature_source_identity_digests: tuple[str, ...]
    feature_exclusion_mask_digest: str
    feature_exclusion_contract_digest: str
    artifact_derivative_defect: float
    run_local_derivative_defect: float
    validity_contract: Literal["probabilistic", "exogenous_static"]
    uncertainty_contract: Literal["model_spatial", "constant_research"]
    execution_contract_digest: str
    dependency: PriorDependency
    contract: str = "neural-prior-inference-evidence-v4"
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-inference-evidence-v4":
            raise ValueError("unsupported neural-prior inference evidence")
        for name in (
            "neural_prior_digest",
            "input_bundle_digest",
            "input_frames_digest",
            "feature_tensor_digest",
            "feature_extractor_digest",
            "feature_extractor_code_digest",
            "model_code_digest",
            "exported_graph_digest",
            "model_artifact_digest",
            "model_contract_digest",
            "feature_schema_digest",
            "training_manifest_digest",
            "inference_algorithm_digest",
            "numerical_runtime_digest",
            "output_background_digest",
            "output_std_digest",
            "output_valid_mask_digest",
            "output_valid_probability_digest",
            "output_support_probability_digest",
            "feature_exclusion_mask_digest",
            "feature_exclusion_contract_digest",
            "execution_contract_digest",
        ):
            _require_prior_digest(name, getattr(self, name))
        if self.dependency not in ("exogenous", "radar_dependent"):
            raise ValueError("unsupported neural-prior dependency")
        if self.uncertainty_contract not in ("model_spatial", "constant_research"):
            raise ValueError("unsupported neural-prior uncertainty contract")
        if self.validity_contract not in ("probabilistic", "exogenous_static"):
            raise ValueError("unsupported neural-prior validity contract")
        for defect in (
            self.artifact_derivative_defect,
            self.run_local_derivative_defect,
        ):
            if not math.isfinite(defect) or defect < 0.0:
                raise ValueError("neural-prior derivative defect must be nonnegative")
        if (self.prior_output_valid_time is None) != (
            not self.feature_source_valid_times
        ):
            raise ValueError("neural-prior feature times are incomplete")
        if len(self.feature_source_identity_digests) != len(
            self.feature_source_valid_times
        ):
            raise ValueError("neural-prior feature source identities are incomplete")
        for digest in self.feature_source_identity_digests:
            _require_prior_digest("feature source identity digest", digest)
        object.__setattr__(self, "evidence_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "evidence_digest"
        }

    def validate_integrity(self) -> None:
        if self.evidence_digest != json_digest(self.payload):
            raise ValueError("neural-prior inference evidence digest mismatch")


class NeuralPriorInferenceRunner:
    """Small deterministic adapter around one evaluated PyTorch prior."""

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Callable[[Tensor], Tensor],
        *,
        example_frames: Tensor,
        feature_extractor_digest: str | None = None,
        model_contract_digest: str,
        feature_schema_digest: str,
        training_manifest_digest: str,
        inference_algorithm_digest: str | None = None,
        numerical_runtime_digest: str | None = None,
        dependency: PriorDependency,
        prior_std_dbz: float = 1.0,
        allow_constant_uncertainty: bool = False,
        derivative_probe_count: int = 4,
        run_derivative_probe_count: int = 2,
        maximum_derivative_defect: float = 1.0e-4,
        support_policy: PriorSupportPolicy = "causal_clip",
        maximum_added_area_km2: float = 0.0,
        maximum_added_echo_integral: float = 0.0,
        feature_exclusion_mask: Tensor | None = None,
        feature_exclusion_contract_digest: str | None = None,
    ) -> None:
        if not isinstance(model, nn.Module) or model.training:
            raise ValueError("neural-prior model must be an eval-mode nn.Module")
        if not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        for name, value in (
            ("model_contract_digest", model_contract_digest),
            ("feature_schema_digest", feature_schema_digest),
            ("training_manifest_digest", training_manifest_digest),
        ):
            _require_prior_digest(name, value)
        if not example_frames.is_floating_point() or example_frames.ndim != 3:
            raise ValueError("neural-prior example frames must be floating [T,H,W]")
        canonical_frames = torch.zeros_like(example_frames)
        retained_exclusion_mask = (
            torch.zeros_like(example_frames, dtype=torch.bool)
            if feature_exclusion_mask is None
            else feature_exclusion_mask.detach().clone()
        )
        if (
            retained_exclusion_mask.shape != example_frames.shape
            or retained_exclusion_mask.dtype is not torch.bool
        ):
            raise ValueError("neural-prior feature exclusion mask is invalid")
        actual_exclusion_contract_digest = json_digest(
            {
                "contract": "neural-prior-feature-exclusion-v1",
                "mask_digest": tensor_digest(retained_exclusion_mask),
                "replacement": "zero",
            }
        )
        if feature_exclusion_contract_digest is not None and (
            feature_exclusion_contract_digest
            != actual_exclusion_contract_digest
        ):
            raise ValueError("declared feature exclusion is not the executed mask")
        with torch.no_grad():
            example_features = feature_extractor(
                torch.where(
                    retained_exclusion_mask,
                    torch.zeros_like(canonical_frames),
                    canonical_frames,
                )
            )
        if (
            not isinstance(example_features, Tensor)
            or not example_features.is_floating_point()
        ):
            raise TypeError("neural-prior features must be floating Tensor data")
        _, feature_graph_digest = _export_graph(
            _FeatureGraph(feature_extractor, retained_exclusion_mask),
            canonical_frames,
        )
        _, model_graph_digest = _export_graph(model, example_features)
        exported_pipeline, exported_graph_digest = _export_graph(
            _PriorPipeline(
                model,
                feature_extractor,
                retained_exclusion_mask,
            ).eval(),
            canonical_frames,
        )
        actual_feature_digest = feature_graph_digest
        actual_algorithm_digest = json_digest(
            {
                "contract": "advar-neural-prior-inference-algorithm-v4",
                "export": "torch.export",
                "derivatives": "rademacher-jvp-vjp-finite-difference",
                "output": "mean-log-std-valid-probability-support",
            }
        )
        for name, claimed, actual in (
            (
                "feature_extractor_digest",
                feature_extractor_digest,
                actual_feature_digest,
            ),
            (
                "inference_algorithm_digest",
                inference_algorithm_digest,
                actual_algorithm_digest,
            ),
        ):
            if claimed is not None and claimed != actual:
                raise ValueError(f"declared {name} is not the executed artifact")
        actual_runtime_digest = numerical_runtime_identity_digest(
            example_frames.device
        )
        if numerical_runtime_digest is not None and (
            numerical_runtime_digest != actual_runtime_digest
        ):
            raise ValueError("declared neural-prior runtime is not the actual runtime")
        if dependency not in ("exogenous", "radar_dependent"):
            raise ValueError("unsupported neural-prior dependency")
        if not math.isfinite(prior_std_dbz) or prior_std_dbz <= 0.0:
            raise ValueError("neural-prior uncertainty must be positive")
        if type(allow_constant_uncertainty) is not bool:
            raise TypeError("allow_constant_uncertainty must be bool")
        if type(derivative_probe_count) is not int or derivative_probe_count < 3:
            raise ValueError("neural-prior derivative probes must be at least three")
        if (
            type(run_derivative_probe_count) is not int
            or run_derivative_probe_count < 1
        ):
            raise ValueError("run-local derivative probes must be positive")
        if not math.isfinite(maximum_derivative_defect) or not (
            0.0 < maximum_derivative_defect < 1.0
        ):
            raise ValueError("maximum neural-prior derivative defect is invalid")
        if support_policy not in ("causal_clip", "expand_control"):
            raise ValueError("unsupported neural-prior support policy")
        for name, value in (
            ("maximum_added_area_km2", maximum_added_area_km2),
            ("maximum_added_echo_integral", maximum_added_echo_integral),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.model = model
        self.feature_extractor = feature_extractor
        self.feature_extractor_digest = actual_feature_digest
        self.model_contract_digest = model_contract_digest
        self.feature_schema_digest = feature_schema_digest
        self.training_manifest_digest = training_manifest_digest
        self.inference_algorithm_digest = actual_algorithm_digest
        self.numerical_runtime_digest = actual_runtime_digest
        self.dependency: PriorDependency = dependency
        self.prior_std_dbz = prior_std_dbz
        self.allow_constant_uncertainty = allow_constant_uncertainty
        self.derivative_probe_count = derivative_probe_count
        self.run_derivative_probe_count = run_derivative_probe_count
        self.maximum_derivative_defect = maximum_derivative_defect
        self.support_policy: PriorSupportPolicy = support_policy
        self.maximum_added_area_km2 = maximum_added_area_km2
        self.maximum_added_echo_integral = maximum_added_echo_integral
        self._feature_exclusion_mask = retained_exclusion_mask
        self.feature_exclusion_contract_digest = (
            actual_exclusion_contract_digest
        )
        self._model_state_digest = _module_state_digest(model)
        self._model_code_digest = model_graph_digest
        self._feature_extractor_code_digest = feature_graph_digest
        self._exported_pipeline_digest = exported_graph_digest
        self._exported_pipeline = exported_pipeline
        self._example_shape = tuple(example_frames.shape)
        self._example_dtype = example_frames.dtype
        self.execution_contract_digest = json_digest(
            {
                "contract": "neural-prior-execution-contract-v5",
                "model_state_digest": self._model_state_digest,
                "model_code_digest": self._model_code_digest,
                "feature_extractor_digest": actual_feature_digest,
                "feature_extractor_code_digest": (
                    self._feature_extractor_code_digest
                ),
                "exported_graph_digest": exported_graph_digest,
                "model_contract_digest": model_contract_digest,
                "feature_schema_digest": feature_schema_digest,
                "training_manifest_digest": training_manifest_digest,
                "inference_algorithm_digest": actual_algorithm_digest,
                "numerical_runtime_digest": actual_runtime_digest,
                "dependency": dependency,
                "prior_std_dbz": prior_std_dbz,
                "allow_constant_uncertainty": allow_constant_uncertainty,
                "derivative_probe_count": derivative_probe_count,
                "run_derivative_probe_count": run_derivative_probe_count,
                "maximum_derivative_defect": maximum_derivative_defect,
                "support_policy": support_policy,
                "maximum_added_area_km2": maximum_added_area_km2,
                "maximum_added_echo_integral": maximum_added_echo_integral,
                "feature_exclusion_contract_digest": (
                    actual_exclusion_contract_digest
                ),
            }
        )
        self.neural_prior_digest = self.execution_contract_digest
        self._certified_derivative_defect = self._validate_derivatives(
            canonical_frames,
            probe_count=derivative_probe_count,
        )

    @property
    def feature_exclusion_mask(self) -> Tensor:
        """Return the exact mask applied before feature extraction."""

        return self._feature_exclusion_mask.detach().clone()

    def _validate_state(self) -> None:
        if _module_state_digest(self.model) != self._model_state_digest:
            raise ValueError("neural-prior model state changed after approval")

    def _validate_input_contract(self, frames_dbz: Tensor) -> None:
        if (
            tuple(frames_dbz.shape) != self._example_shape
            or frames_dbz.dtype != self._example_dtype
        ):
            raise ValueError("neural-prior input shape or dtype changed")

    def _derivative_outputs(self, frames_dbz: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_input_contract(frames_dbz)
        raw = self._exported_pipeline(frames_dbz)
        if isinstance(raw, tuple) and len(raw) == 5:
            return cast(Tensor, raw[1]), torch.log(cast(Tensor, raw[2]))
        if isinstance(raw, tuple) and len(raw) == 2 and self.allow_constant_uncertainty:
            mean = cast(Tensor, raw[1])
            return mean, torch.full_like(mean, math.log(self.prior_std_dbz))
        else:
            raise RuntimeError("exported neural-prior pipeline output is invalid")

    def _output(
        self, frames_dbz: Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Literal["probabilistic", "exogenous_static"],
    ]:
        self._validate_state()
        self._validate_input_contract(frames_dbz)
        raw = self._exported_pipeline(frames_dbz)
        if isinstance(raw, tuple) and len(raw) == 5:
            features, output, std, valid, support = raw
        elif isinstance(raw, tuple) and len(raw) == 2 and self.allow_constant_uncertainty:
            features, output = raw
            std = torch.full_like(output, self.prior_std_dbz)
            valid = torch.ones_like(output, dtype=torch.bool)
            support = torch.ones_like(output)
        else:
            raise ValueError(
                "neural-prior model must return mean/std/valid/support output"
            )
        if not isinstance(features, Tensor) or not features.is_floating_point():
            raise TypeError("neural-prior features must be floating Tensor data")
        if (
            not isinstance(output, Tensor)
            or output.ndim != 2
            or not output.is_floating_point()
            or not bool(torch.all(torch.isfinite(output)))
        ):
            raise ValueError("neural-prior output must be a finite 2-D Tensor")
        if isinstance(valid, Tensor) and valid.dtype is torch.bool:
            if self.dependency != "exogenous" and not self.allow_constant_uncertainty:
                raise ValueError(
                    "radar-dependent prior validity must be a probability"
                )
            valid_mask = valid
            valid_probability = valid.to(output)
            validity_contract: Literal[
                "probabilistic", "exogenous_static"
            ] = "exogenous_static"
        elif isinstance(valid, Tensor) and valid.is_floating_point():
            if not bool(torch.all(torch.isfinite(valid))) or bool(
                torch.any((valid < 0.0) | (valid > 1.0))
            ):
                raise ValueError("neural-prior validity probability is invalid")
            valid_probability = valid
            valid_mask = valid >= 0.5
            validity_contract = "probabilistic"
        else:
            raise ValueError("neural-prior validity output is invalid")
        if (
            not isinstance(std, Tensor)
            or std.shape != output.shape
            or not bool(torch.all(torch.isfinite(std) & (std > 0.0)))
            or valid_mask.shape != output.shape
            or not isinstance(support, Tensor)
            or support.shape != output.shape
            or not bool(torch.all(torch.isfinite(support)))
            or bool(torch.any((support < 0.0) | (support > 1.0)))
        ):
            raise ValueError("neural-prior uncertainty output is invalid")
        self._validate_state()
        return (
            features,
            output,
            std,
            valid_mask,
            valid_probability,
            support,
            validity_contract,
        )

    def _validate_derivatives(
        self,
        frames_dbz: Tensor,
        *,
        probe_count: int,
    ) -> float:
        if self.dependency != "radar_dependent":
            return 0.0
        mean, log_std = self._derivative_outputs(frames_dbz)
        generator = torch.Generator(device="cpu").manual_seed(0)
        defects: list[float] = []
        step = 1.0e-3 if frames_dbz.dtype == torch.float32 else 1.0e-5
        epsilon = torch.finfo(frames_dbz.dtype).eps
        for _ in range(probe_count):
            tangent = (
                torch.randint(
                    0,
                    2,
                    frames_dbz.shape,
                    generator=generator,
                    dtype=torch.int8,
                ).to(frames_dbz) * 2.0 - 1.0
            )
            mean_cotangent = (
                torch.randint(
                    0,
                    2,
                    mean.shape,
                    generator=generator,
                    dtype=torch.int8,
                ).to(mean) * 2.0 - 1.0
            )
            log_std_cotangent = (
                torch.randint(
                    0,
                    2,
                    log_std.shape,
                    generator=generator,
                    dtype=torch.int8,
                ).to(log_std) * 2.0 - 1.0
            )
            mean_forward, log_std_forward = self.jvp_components(
                frames_dbz,
                tangent,
            )
            reverse = self.vjp_components(
                frames_dbz,
                mean_cotangent,
                log_std_cotangent,
            )
            left = torch.sum(mean_forward * mean_cotangent) + torch.sum(
                log_std_forward * log_std_cotangent
            )
            right = torch.sum(tangent * reverse)
            dual = torch.abs(left - right) / (
                torch.abs(left) + torch.abs(right) + epsilon
            )
            plus = self._derivative_outputs(frames_dbz + step * tangent)
            minus = self._derivative_outputs(frames_dbz - step * tangent)
            finite_mean = (plus[0] - minus[0]) / (2.0 * step)
            finite_log_std = (plus[1] - minus[1]) / (2.0 * step)
            forward_flat = torch.cat(
                (mean_forward.flatten(), log_std_forward.flatten())
            )
            finite_flat = torch.cat((finite_mean.flatten(), finite_log_std.flatten()))
            fd = torch.linalg.vector_norm(forward_flat - finite_flat) / (
                torch.linalg.vector_norm(forward_flat)
                + torch.linalg.vector_norm(finite_flat)
                + epsilon
            )
            defects.extend((float(dual.detach()), float(fd.detach())))
        defect = max(defects)
        if defect > self.maximum_derivative_defect:
            raise ValueError("neural-prior derivative defect exceeds its contract")
        return defect

    def infer(
        self,
        frames_dbz: Tensor,
        *,
        input_run: ForecastRunContract,
        role: Literal["candidate", "parent"],
    ) -> NeuralPriorApplication:
        """Run the model now and bind its output to the exact input bundle."""

        input_run.validate_integrity()
        if self._certified_derivative_defect > self.maximum_derivative_defect:
            raise ValueError("neural-prior derivative defect exceeds its contract")
        if tensor_digest(frames_dbz) != input_run.input_frames_digest:
            raise ValueError("neural-prior frames disagree with the input run")
        features, output, std, valid, valid_probability, support, validity = (
            self._output(frames_dbz)
        )
        run_local_defect = self._validate_derivatives(
            frames_dbz,
            probe_count=self.run_derivative_probe_count,
        )
        grid = input_run.grid_time_contract
        feature_source_valid_times = () if grid is None else tuple(grid.valid_times)
        source_identity = input_run.input_bundle_digest
        if input_run.operational_data_identity_json is not None:
            retained_source_identity = OperationalDataIdentity.from_json(
                input_run.operational_data_identity_json
            ).radar_product_digest
            if retained_source_identity is not None:
                source_identity = retained_source_identity
        feature_source_identity_digests = tuple(
            source_identity for _ in feature_source_valid_times
        )
        evidence = NeuralPriorInferenceEvidence(
            neural_prior_digest=self.neural_prior_digest,
            input_bundle_digest=input_run.input_bundle_digest,
            input_frames_digest=tensor_digest(frames_dbz),
            feature_tensor_digest=tensor_digest(features),
            feature_extractor_digest=self.feature_extractor_digest,
            feature_extractor_code_digest=self._feature_extractor_code_digest,
            model_code_digest=self._model_code_digest,
            exported_graph_digest=self._exported_pipeline_digest,
            model_artifact_digest=self._model_state_digest,
            model_contract_digest=self.model_contract_digest,
            feature_schema_digest=self.feature_schema_digest,
            training_manifest_digest=self.training_manifest_digest,
            inference_algorithm_digest=self.inference_algorithm_digest,
            numerical_runtime_digest=self.numerical_runtime_digest,
            output_background_digest=tensor_digest(output),
            output_std_digest=tensor_digest(std),
            output_valid_mask_digest=tensor_digest(valid),
            output_valid_probability_digest=tensor_digest(valid_probability),
            output_support_probability_digest=tensor_digest(support),
            prior_output_valid_time=(
                None
                if not feature_source_valid_times
                else feature_source_valid_times[0]
            ),
            feature_source_valid_times=feature_source_valid_times,
            feature_source_identity_digests=feature_source_identity_digests,
            feature_exclusion_mask_digest=tensor_digest(
                self._feature_exclusion_mask
            ),
            feature_exclusion_contract_digest=(
                self.feature_exclusion_contract_digest
            ),
            artifact_derivative_defect=self._certified_derivative_defect,
            run_local_derivative_defect=run_local_defect,
            validity_contract=validity,
            uncertainty_contract=(
                "constant_research"
                if self.allow_constant_uncertainty
                else "model_spatial"
            ),
            execution_contract_digest=self.execution_contract_digest,
            dependency=self.dependency,
        )
        return _new_neural_prior_application(
            initial_background_dbz=output,
            valid_mask=valid,
            valid_probability=valid_probability,
            std_dbz=std,
            support_probability=support,
            inference_evidence=evidence,
            role=role,
            support_policy=self.support_policy,
            maximum_added_area_km2=self.maximum_added_area_km2,
            maximum_added_echo_integral=self.maximum_added_echo_integral,
        )

    def reproduce(
        self,
        application: NeuralPriorApplication,
        frames_dbz: Tensor,
    ) -> None:
        """Independently rerun one retained application."""

        application.validate_integrity()
        features, output, std, valid, valid_probability, support, validity = (
            self._output(frames_dbz)
        )
        evidence = application.inference_evidence
        if (
            self.neural_prior_digest != evidence.neural_prior_digest
            or evidence.execution_contract_digest != self.execution_contract_digest
            or tensor_digest(frames_dbz) != evidence.input_frames_digest
            or tensor_digest(features) != evidence.feature_tensor_digest
            or tensor_digest(output) != evidence.output_background_digest
            or tensor_digest(std) != evidence.output_std_digest
            or tensor_digest(valid) != evidence.output_valid_mask_digest
            or tensor_digest(valid_probability)
            != evidence.output_valid_probability_digest
            or tensor_digest(support)
            != evidence.output_support_probability_digest
            or self._exported_pipeline_digest != evidence.exported_graph_digest
            or not torch.equal(output, application.initial_background_dbz)
            or validity != evidence.validity_contract
        ):
            raise ValueError("neural-prior inference cannot be reproduced")

    def validate_retained_output(
        self,
        frames_dbz: Tensor,
        raw_background_dbz: Tensor,
        *,
        execution_contract_digest: str,
    ) -> None:
        """Reproduce the raw prior output retained by a restart artifact."""

        if execution_contract_digest != self.execution_contract_digest:
            raise ValueError("neural-prior execution contract mismatch")
        _, output, _, _, _, _, _ = self._output(frames_dbz)
        if not torch.equal(output, raw_background_dbz):
            raise ValueError("retained neural-prior output cannot be reproduced")

    def jvp(self, frames_dbz: Tensor, tangent: Tensor) -> Tensor:
        return self.jvp_components(frames_dbz, tangent)[0]

    def jvp_components(
        self,
        frames_dbz: Tensor,
        tangent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.dependency != "radar_dependent":
            zero = frames_dbz.new_zeros(frames_dbz.shape[-2:])
            return zero, zero.clone()
        self._validate_state()
        return cast(
            tuple[Tensor, Tensor],
            torch.func.jvp(
                self._derivative_outputs,
                (frames_dbz,),
                (tangent,),
            )[1],
        )

    def vjp(self, frames_dbz: Tensor, cotangent: Tensor) -> Tensor:
        return self.vjp_components(
            frames_dbz,
            cotangent,
            torch.zeros_like(cotangent),
        )

    def vjp_components(
        self,
        frames_dbz: Tensor,
        mean_cotangent: Tensor,
        log_std_cotangent: Tensor,
    ) -> Tensor:
        if self.dependency != "radar_dependent":
            return torch.zeros_like(frames_dbz)
        self._validate_state()
        _, pullback = cast(
            tuple[
                tuple[Tensor, Tensor],
                Callable[[tuple[Tensor, Tensor]], tuple[Tensor]],
            ],
            torch.func.vjp(
                self._derivative_outputs,
                frames_dbz,
            ),
        )
        return cast(Tensor, pullback((mean_cotangent, log_std_cotangent))[0])

    def validate_adjoint_direction(
        self,
        frames_dbz: Tensor,
        mean_cotangent: Tensor,
        log_std_cotangent: Tensor | None = None,
    ) -> float:
        """Check the exact FSO cotangent against an independent JVP."""

        if self.dependency != "radar_dependent":
            return 0.0
        self._validate_input_contract(frames_dbz)
        generator = torch.Generator(device="cpu").manual_seed(1)
        tangent = (
            torch.randint(
                0,
                2,
                frames_dbz.shape,
                generator=generator,
                dtype=torch.int8,
            ).to(frames_dbz) * 2.0 - 1.0
        )
        mean_forward, log_std_forward = self.jvp_components(frames_dbz, tangent)
        retained_log_std_cotangent = (
            torch.zeros_like(mean_cotangent)
            if log_std_cotangent is None
            else log_std_cotangent
        )
        reverse = self.vjp_components(
            frames_dbz,
            mean_cotangent,
            retained_log_std_cotangent,
        )
        left = torch.sum(mean_forward * mean_cotangent) + torch.sum(
            log_std_forward * retained_log_std_cotangent
        )
        right = torch.sum(tangent * reverse)
        epsilon = torch.finfo(frames_dbz.dtype).eps
        defect = torch.abs(left - right) / (
            torch.abs(left) + torch.abs(right) + epsilon
        )
        value = float(defect.detach())
        if value > self.maximum_derivative_defect:
            raise ValueError("neural-prior adjoint-direction defect is too large")
        return value


@dataclass(frozen=True, init=False)
class NeuralPriorApplication:
    """One model-produced prior output that is actually consumed by P1."""

    initial_background_dbz: Tensor
    valid_mask: Tensor
    valid_probability: Tensor
    std_dbz: Tensor
    support_probability: Tensor
    inference_evidence: NeuralPriorInferenceEvidence
    role: Literal["candidate", "parent"]
    support_policy: PriorSupportPolicy
    maximum_added_area_km2: float
    maximum_added_echo_integral: float
    contract: str = "neural-prior-application-v4"
    application_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use NeuralPriorInferenceRunner.infer")

    @property
    def neural_prior_digest(self) -> str:
        return self.inference_evidence.neural_prior_digest

    @property
    def model_contract_digest(self) -> str:
        return self.inference_evidence.model_contract_digest

    @property
    def feature_schema_digest(self) -> str:
        return self.inference_evidence.feature_schema_digest

    @property
    def training_manifest_digest(self) -> str:
        return self.inference_evidence.training_manifest_digest

    @property
    def dependency(self) -> PriorDependency:
        return self.inference_evidence.dependency

    def validate_integrity(self) -> None:
        self.inference_evidence.validate_integrity()
        if self.application_digest != _neural_prior_application_digest(self):
            raise ValueError("neural-prior application digest mismatch")


def _new_neural_prior_application(**values: object) -> NeuralPriorApplication:
    result = object.__new__(NeuralPriorApplication)
    object.__setattr__(result, "contract", "neural-prior-application-v4")
    for name, value in values.items():
        object.__setattr__(result, name, value)
    background = result.initial_background_dbz.detach().clone()
    valid = result.valid_mask.detach().clone()
    valid_probability = result.valid_probability.detach().clone()
    std = result.std_dbz.detach().clone()
    support = result.support_probability.detach().clone()
    if (
        background.ndim != 2
        or not background.is_floating_point()
        or not bool(torch.all(torch.isfinite(background)))
        or valid.shape != background.shape
        or valid.dtype != torch.bool
        or valid_probability.shape != background.shape
        or not valid_probability.is_floating_point()
        or not bool(torch.all(torch.isfinite(valid_probability)))
        or bool(torch.any((valid_probability < 0.0) | (valid_probability > 1.0)))
        or not torch.equal(valid, valid_probability >= 0.5)
        or std.shape != background.shape
        or not std.is_floating_point()
        or not bool(torch.all(torch.isfinite(std) & (std > 0.0)))
        or support.shape != background.shape
        or not support.is_floating_point()
        or bool(torch.any(~torch.isfinite(support)))
        or bool(torch.any((support < 0.0) | (support > 1.0)))
    ):
        raise ValueError("neural-prior value, validity, or uncertainty is invalid")
    if result.role not in ("candidate", "parent"):
        raise ValueError("neural-prior role must be candidate or parent")
    if result.support_policy not in ("causal_clip", "expand_control"):
        raise ValueError("unsupported neural-prior support policy")
    for name in ("maximum_added_area_km2", "maximum_added_echo_integral"):
        value = getattr(result, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    result.inference_evidence.validate_integrity()
    if tensor_digest(background) != result.inference_evidence.output_background_digest:
        raise ValueError("neural-prior output disagrees with inference evidence")
    if (
        tensor_digest(std) != result.inference_evidence.output_std_digest
        or tensor_digest(valid) != result.inference_evidence.output_valid_mask_digest
        or tensor_digest(valid_probability)
        != result.inference_evidence.output_valid_probability_digest
        or tensor_digest(support)
        != result.inference_evidence.output_support_probability_digest
    ):
        raise ValueError("neural-prior uncertainty disagrees with inference evidence")
    object.__setattr__(result, "initial_background_dbz", background)
    object.__setattr__(result, "valid_mask", valid)
    object.__setattr__(result, "valid_probability", valid_probability)
    object.__setattr__(result, "std_dbz", std)
    object.__setattr__(result, "support_probability", support)
    object.__setattr__(
        result, "application_digest", _neural_prior_application_digest(result)
    )
    return result


def _neural_prior_application_digest(value: NeuralPriorApplication) -> str:
    return json_digest(
        {
            "contract": value.contract,
            "initial_background_dbz": tensor_digest(value.initial_background_dbz),
            "valid_mask": tensor_digest(value.valid_mask),
            "valid_probability": tensor_digest(value.valid_probability),
            "std_dbz": tensor_digest(value.std_dbz),
            "support_probability": tensor_digest(value.support_probability),
            "inference_evidence_digest": value.inference_evidence.evidence_digest,
            "role": value.role,
            "support_policy": value.support_policy,
            "maximum_added_area_km2": value.maximum_added_area_km2,
            "maximum_added_echo_integral": value.maximum_added_echo_integral,
        }
    )


@dataclass(frozen=True)
class FrozenObservationWhitener:
    """Precomputed constants for the frozen observation-error transform."""

    mode: Tensor | None
    overlapping_correction: Tensor | None
    per_frame: bool


@dataclass(frozen=True)
class FrozenOuterState:
    input_frames_dbz: Tensor
    background_frames_dbz: Tensor | None
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
    observation_whitener: FrozenObservationWhitener
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
    observation_derived_initial_background: bool = True
    neural_prior_std_dbz: Tensor | None = None
    neural_prior_valid_mask: Tensor | None = None
    neural_prior_dependency: PriorDependency | None = None
    neural_prior_application_digest: str | None = None
    neural_prior_raw_background_dbz: Tensor | None = None
    neural_prior_execution_contract_digest: str | None = None
    neural_prior_role: Literal["candidate", "parent"] | None = None

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
    minimum_object_count_ratio_by_time: Tensor

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
        object_count_low = (
            ~torch.isnan(self.minimum_object_count_ratio_by_time)
            & (
                self.minimum_object_count_ratio_by_time
                < config.minimum_object_count_ratio_for_confidence
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
                | object_count_low
            )
        )


def _amplitude_confidence_margin(
    diagnostics: _AmplitudeDiagnostics,
    config: AnalysisConfig,
) -> float | None:
    """Return the smallest signed slack across amplitude-confidence gates."""

    if diagnostics.has_insufficient_information:
        return None
    margins: list[Tensor] = []

    def lower(values: Tensor, threshold: float) -> None:
        available = torch.isfinite(values)
        if bool(torch.any(available)):
            margins.append(torch.min(values[available] - threshold))

    def upper(values: Tensor, threshold: float) -> None:
        available = torch.isfinite(values)
        if bool(torch.any(available)):
            margins.append(torch.min(threshold - values[available]))

    lower(
        diagnostics.integrated_echo_ratio_by_time,
        config.minimum_integrated_echo_ratio_for_confidence,
    )
    upper(
        diagnostics.integrated_echo_ratio_by_time,
        config.maximum_integrated_echo_ratio_for_confidence,
    )
    lower(
        diagnostics.displacement_tolerant_soft_echo_area_ratio_by_time,
        config.minimum_soft_echo_area_ratio_for_confidence,
    )
    upper(
        diagnostics.displacement_tolerant_soft_echo_area_ratio_by_time,
        config.maximum_soft_echo_area_ratio_for_confidence,
    )
    upper(
        diagnostics.established_echo_excess_growth_fraction_by_time,
        config.maximum_established_excess_growth_fraction_for_confidence,
    )
    upper(
        diagnostics.maximum_object_unresolved_fraction_by_time,
        config.maximum_unresolved_amplitude_fraction,
    )
    lower(
        diagnostics.minimum_object_integrated_echo_ratio_by_time,
        config.minimum_integrated_echo_ratio_for_confidence,
    )
    upper(
        diagnostics.maximum_object_integrated_echo_ratio_by_time,
        config.maximum_integrated_echo_ratio_for_confidence,
    )
    lower(
        diagnostics.minimum_object_soft_echo_area_ratio_by_time,
        config.minimum_soft_echo_area_ratio_for_confidence,
    )
    upper(
        diagnostics.maximum_object_soft_echo_area_ratio_by_time,
        config.maximum_soft_echo_area_ratio_for_confidence,
    )
    lower(
        diagnostics.minimum_object_count_ratio_by_time,
        config.minimum_object_count_ratio_for_confidence,
    )
    if not margins:
        return None
    return float(torch.min(torch.stack(margins)).detach())


@dataclass(frozen=True)
class _IdentifiabilityDiagnostics:
    dynamics_data_gram_eigenvalues: tuple[float, float, float]
    dynamics_data_information_trace: float
    dynamics_data_numerical_rank: int
    dynamics_data_effective_dimension: float
    dynamics_data_to_prior_ratio_by_mode: tuple[float, float, float]
    field_conditioned_dynamics_data_gram_eigenvalues: (
        tuple[float, float, float] | None
    )
    field_conditioned_dynamics_data_information_trace: float | None
    field_conditioned_dynamics_data_effective_dimension: float | None
    field_conditioning_maximum_relative_residual: float | None
    field_conditioned_dynamics_posterior_covariance: Tensor | None
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
class AnalysisFeasibilityMargins:
    """Signed distances from hard or operational P1 boundaries.

    Positive values are interior to the accepted-analysis contract.  The
    amplitude-confidence margin is dimensionless because it combines only
    dimensionless ratio and fraction gates.  Motion saturation is retained
    both as a coordinate-independent fraction of the configured bound and,
    when a physical grid contract exists, in m s-1.
    """

    reachability_support: float
    unresolved_amplitude_fraction: float
    amplitude_confidence: float | None
    motion_saturation_fraction: float
    motion_speed_saturation_mps: float | None
    growth_saturation_per_step: float


@dataclass(frozen=True)
class AnalysisLinearization:
    """Accepted P1 frozen-model state needed by an implicit adjoint."""

    observations: AnalysisObservations
    frozen: FrozenOuterState
    residual_norm: float
    gradient_norm: float
    field_gradient_rms: float
    field_gradient_max: float
    dynamics_gradient_max: float
    relative_stationarity: float
    robust_gradient_norm: float
    robust_field_gradient_rms: float
    robust_field_gradient_max: float
    robust_dynamics_gradient_max: float
    robust_relative_stationarity: float
    irls_relative_weight_change: float
    polish_iterations: int
    feasibility_margins: AnalysisFeasibilityMargins
    forecast_run_digest: str | None = None
    control_digest: str = ""
    algorithm_bundle_digest: str = ""
    numerical_runtime_digest: str = ""
    linearization_digest: str = ""
    contract: str = P1_LINEARIZATION_CONTRACT


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
    minimum_object_count_ratio_by_time: tuple[float, float] | None = None
    dynamics_data_numerical_rank: int | None = None
    dynamics_data_effective_dimension: float | None = None
    dynamics_data_to_prior_ratio_by_mode: (
        tuple[float, float, float] | None
    ) = None
    field_conditioned_dynamics_data_gram_eigenvalues: (
        tuple[float, float, float] | None
    ) = None
    field_conditioned_dynamics_data_information_trace: float | None = None
    field_conditioned_dynamics_data_effective_dimension: float | None = None
    field_conditioning_maximum_relative_residual: float | None = None
    motion_control_coordinate_system: str = "grid_yx_px"
    field_smoothness_coordinate_system: str = "index_graph"
    linearization_residual_norm: float | None = None
    linearization_gradient_norm: float | None = None
    linearization_field_gradient_rms: float | None = None
    linearization_field_gradient_max: float | None = None
    linearization_dynamics_gradient_max: float | None = None
    linearization_relative_stationarity: float | None = None
    robust_gradient_norm: float | None = None
    robust_field_gradient_rms: float | None = None
    robust_field_gradient_max: float | None = None
    robust_dynamics_gradient_max: float | None = None
    robust_relative_stationarity: float | None = None
    irls_relative_weight_change: float | None = None
    linearization_polish_iterations: int = 0
    linearization: AnalysisLinearization | None = None
    final_linearization_stationary: bool = False
    final_robust_stationary: bool = False
    final_irls_fixed_point: bool = False
    p1_forecast_eligible: bool = False
    posterior_eligible: bool = False
    fso_eligible: bool = False
    outer_converged: bool = False


@dataclass(frozen=True)
class P1LinearizationState:
    """Minimum accepted-analysis state needed for delayed P1 FSO."""

    control: Tensor
    active_field_index: Tensor
    state: RadarState
    linearization_residual_norm: float
    linearization_gradient_norm: float
    linearization_field_gradient_rms: float
    linearization_field_gradient_max: float
    linearization_dynamics_gradient_max: float
    linearization_relative_stationarity: float
    robust_gradient_norm: float
    robust_field_gradient_rms: float
    robust_field_gradient_max: float
    robust_dynamics_gradient_max: float
    robust_relative_stationarity: float
    irls_relative_weight_change: float
    linearization_polish_iterations: int
    linearization: AnalysisLinearization
    converged: bool = True
    used_fallback: bool = False
    degraded: bool = False
    final_linearization_stationary: bool = True
    final_robust_stationary: bool = True
    final_irls_fixed_point: bool = True
    p1_forecast_eligible: bool = True
    posterior_eligible: bool = True
    fso_eligible: bool = True
    outer_converged: bool = True


def _canonical_common_bias_group_index(
    group_index: Tensor,
    *,
    frame_shape: tuple[int, int, int],
    temporal_scope: ObservationCommonBiasScope,
    device: torch.device,
) -> Tensor:
    if group_index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("common-bias group map must use an integer dtype")
    if group_index.ndim == 2 and tuple(group_index.shape) == frame_shape[1:]:
        raw = group_index.to(device=device, dtype=torch.long).unsqueeze(0).expand(
            frame_shape
        )
    elif group_index.ndim == 3 and tuple(group_index.shape) == frame_shape:
        raw = group_index.to(device=device, dtype=torch.long)
    else:
        raise ValueError(
            "common-bias group map must have shape [H, W] or [3, H, W]"
        )
    if bool(torch.any(raw < -1)):
        raise ValueError("common-bias group labels must be -1 or nonnegative")
    canonical = torch.full_like(raw, -1)

    def compact(values: Tensor, *, offset: int) -> tuple[Tensor, int]:
        active = values >= 0
        if not bool(torch.any(active)):
            return torch.full_like(values, -1), offset
        labels = values[active]
        unique_labels, inverse = torch.unique(
            labels,
            sorted=True,
            return_inverse=True,
        )
        compacted = torch.full_like(values, -1)
        compacted[active] = inverse + offset
        return compacted, offset + unique_labels.numel()

    if temporal_scope == "per_frame":
        offset = 0
        frames: list[Tensor] = []
        for frame in raw:
            compacted, offset = compact(frame, offset=offset)
            frames.append(compacted)
        canonical = torch.stack(frames)
    else:
        canonical, _ = compact(raw, offset=0)
    return canonical.detach().clone()


def observation_common_bias_group_map_digest(
    group_index: Tensor,
    *,
    temporal_scope: ObservationCommonBiasScope = "per_frame",
) -> str:
    """Digest the canonical spatial partition used by common-bias modes."""

    if temporal_scope not in ("per_frame", "all_times"):
        raise ValueError("temporal_scope must be per_frame or all_times")
    if group_index.ndim == 2:
        frame_shape = (3, group_index.shape[0], group_index.shape[1])
    elif group_index.ndim == 3 and group_index.shape[0] == 3:
        frame_shape = (
            group_index.shape[0],
            group_index.shape[1],
            group_index.shape[2],
        )
    else:
        raise ValueError(
            "common-bias group map must have shape [H, W] or [3, H, W]"
        )
    canonical = _canonical_common_bias_group_index(
        group_index,
        frame_shape=frame_shape,
        temporal_scope=temporal_scope,
        device=group_index.device,
    )
    return tensor_digest(canonical)


def _canonical_common_bias_mode_weights(
    mode_weights: Tensor,
    *,
    frame_shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if not mode_weights.is_floating_point() or mode_weights.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise TypeError(
            "common-bias mode weights must use float32 or float64"
        )
    if (
        mode_weights.ndim == 3
        and tuple(mode_weights.shape[-2:]) == frame_shape[1:]
    ):
        raw = mode_weights.to(device=device, dtype=dtype)
    elif (
        mode_weights.ndim == 4
        and mode_weights.shape[0] == frame_shape[0]
        and tuple(mode_weights.shape[-2:]) == frame_shape[1:]
    ):
        raw = mode_weights.to(device=device, dtype=dtype)
    else:
        raise ValueError(
            "common-bias mode weights must have shape [K,H,W] or "
            "[3,K,H,W]"
        )
    by_frame = _common_bias_mode_weights_by_frame(raw)
    mode_count = raw.shape[-3]
    if not (1 <= mode_count <= MAXIMUM_OBSERVATION_COMMON_BIAS_MODE_COUNT):
        raise ValueError(
            "common-bias mode count must be between 1 and "
            f"{MAXIMUM_OBSERVATION_COMMON_BIAS_MODE_COUNT}"
        )
    if (
        not bool(torch.all(torch.isfinite(raw)))
        or bool(torch.any(raw < 0.0))
        or bool(torch.any(raw > 1.0))
    ):
        raise ValueError("common-bias mode weights must be finite and in [0,1]")
    if bool(
        torch.any(torch.sum(by_frame.square(), dim=1) > 1.0 + 1.0e-6)
    ):
        raise ValueError(
            "common-bias mode squared weights must sum to at most one"
        )
    if bool(torch.any(torch.sum(by_frame, dim=(0, 2, 3)) <= 0.0)):
        raise ValueError("every common-bias mode must have positive support")
    return raw.detach().clone()


def observation_common_bias_mode_weights_digest(
    mode_weights: Tensor,
) -> str:
    """Digest canonical overlapping common-bias basis weights."""

    if mode_weights.ndim == 3:
        frame_shape = (3, mode_weights.shape[-2], mode_weights.shape[-1])
    elif mode_weights.ndim == 4 and mode_weights.shape[0] == 3:
        frame_shape = (
            mode_weights.shape[0],
            mode_weights.shape[-2],
            mode_weights.shape[-1],
        )
    else:
        raise ValueError(
            "common-bias mode weights must have shape [K,H,W] or "
            "[3,K,H,W]"
        )
    canonical = _canonical_common_bias_mode_weights(
        mode_weights,
        frame_shape=frame_shape,
        dtype=torch.float64,
        device=mode_weights.device,
    )
    return tensor_digest(canonical)


def prepare_analysis(
    frames_dbz: Tensor,
    *,
    nowcast_config: NowcastConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    observation_std_dbz: float | Tensor | None = None,
    quality_weight: float | Tensor | None = None,
    qc_mask: Tensor | None = None,
    observation_common_bias_group_index: Tensor | None = None,
    observation_common_bias_mode_weights: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
    grid_time_contract: RadarGridTimeContract | None = None,
    neural_prior: NeuralPriorApplication | None = None,
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
            nowcast_config.p1_motion_saturation_safe_margin_mps
            > nowcast_config.maximum_motion_speed_mps
        ):
            raise ValueError(
                "operational P1 motion saturation margin cannot exceed "
                "the physical motion limit"
            )
        if (
            nowcast_config.p1_growth_saturation_safe_margin_per_step
            > nowcast_config.max_log_growth_per_step
        ):
            raise ValueError(
                "operational P1 growth saturation margin cannot exceed "
                "the log-growth limit"
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
    if (
        observation_common_bias_group_index is not None
        and observation_common_bias_mode_weights is not None
    ):
        raise ValueError(
            "common-bias group map and overlapping modes are mutually "
            "exclusive"
        )
    common_bias_group_index = None
    common_bias_mode_weights = None
    if observation_common_bias_group_index is not None:
        if analysis_config.observation_common_bias_std_dbz <= 0.0:
            raise ValueError(
                "common-bias group map requires positive common-bias std"
            )
        if analysis_config.observation_common_bias_tile_size_px > 0:
            raise ValueError(
                "common-bias group map and tile-size modes are mutually "
                "exclusive"
            )
        common_bias_group_index = _canonical_common_bias_group_index(
            observation_common_bias_group_index,
            frame_shape=(
                frames_dbz.shape[0],
                frames_dbz.shape[1],
                frames_dbz.shape[2],
            ),
            temporal_scope=analysis_config.observation_common_bias_scope,
            device=frames_dbz.device,
        )
        if not bool(torch.any(common_bias_group_index >= 0)):
            raise ValueError(
                "common-bias group map must contain at least one group"
            )
        group_digest = tensor_digest(common_bias_group_index)
        expected_digest = (
            analysis_config.observation_common_bias_group_map_digest
        )
        if expected_digest is not None and expected_digest != group_digest:
            raise ValueError("common-bias group map digest mismatch")
        if (
            analysis_config.execution_mode == "operational"
            and expected_digest is None
        ):
            raise ValueError(
                "operational common-bias group map requires its digest in "
                "AnalysisConfig"
            )
        if expected_digest is None:
            analysis_config = replace(
                analysis_config,
                observation_common_bias_group_map_digest=group_digest,
            )
    elif analysis_config.observation_common_bias_group_map_digest is not None:
        raise ValueError(
            "common-bias group map digest requires a group map input"
        )
    if observation_common_bias_mode_weights is not None:
        if analysis_config.observation_common_bias_std_dbz <= 0.0:
            raise ValueError(
                "common-bias mode weights require positive common-bias std"
            )
        if analysis_config.observation_common_bias_tile_size_px > 0:
            raise ValueError(
                "common-bias mode weights and tile-size modes are mutually "
                "exclusive"
            )
        resource_estimate = estimate_common_bias_resources(
            tuple(observation_common_bias_mode_weights.shape),
            (
                frames_dbz.shape[0],
                frames_dbz.shape[1],
                frames_dbz.shape[2],
            ),
            dtype=frames_dbz.dtype,
            temporal_scope=analysis_config.observation_common_bias_scope,
            config=analysis_config,
        )
        if "retained_mode_bytes" in resource_estimate.rejection_reasons:
            raise ValueError(
                "common-bias mode weights exceed their retained byte budget"
            )
        if (
            "whitener_operations_per_apply"
            in resource_estimate.rejection_reasons
        ):
            raise ValueError(
                "common-bias whitener exceeds its apply-operation budget"
            )
        if "gram_multiply_adds" in resource_estimate.rejection_reasons:
            raise ValueError(
                "common-bias Gram construction exceeds its operation budget"
            )
        if "frozen_whitener_bytes" in resource_estimate.rejection_reasons:
            raise ValueError(
                "frozen observation whitener exceeds its byte budget"
            )
        common_bias_mode_weights = _canonical_common_bias_mode_weights(
            observation_common_bias_mode_weights,
            frame_shape=(
                frames_dbz.shape[0],
                frames_dbz.shape[1],
                frames_dbz.shape[2],
            ),
            dtype=frames_dbz.dtype,
            device=frames_dbz.device,
        )
        mode_digest = observation_common_bias_mode_weights_digest(
            common_bias_mode_weights
        )
        expected_mode_digest = (
            analysis_config.observation_common_bias_mode_weights_digest
        )
        if (
            expected_mode_digest is not None
            and expected_mode_digest != mode_digest
        ):
            raise ValueError("common-bias mode weights digest mismatch")
        if (
            analysis_config.execution_mode == "operational"
            and expected_mode_digest is None
        ):
            raise ValueError(
                "operational common-bias mode weights require their digest "
                "in AnalysisConfig"
            )
        if expected_mode_digest is None:
            analysis_config = replace(
                analysis_config,
                observation_common_bias_mode_weights_digest=mode_digest,
            )
    elif (
        analysis_config.observation_common_bias_mode_weights_digest is not None
    ):
        raise ValueError(
            "common-bias mode weights digest requires a mode input"
        )
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
    if (
        common_bias_group_index is not None
        and not bool(torch.any((common_bias_group_index >= 0) & valid))
    ):
        raise ValueError(
            "common-bias group map must cover at least one valid observation"
        )
    if common_bias_mode_weights is not None and not bool(
        torch.any(
            (common_bias_mode_weights > 0.0)
            & valid.unsqueeze(1)
        )
    ):
        raise ValueError(
            "common-bias mode weights must cover a valid observation"
        )
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
    if analysis_config.censored_background_policy == "external_background":
        if bool(torch.any(censored & ~prepared.background_mask)):
            raise ValueError(
                "external_background censored policy requires background "
                "coverage at every censored observation"
            )
        detection_limit = prepared.frames_dbz.new_full(
            (),
            analysis_config.detection_limit_dbz,
        )
        below_detection = torch.nextafter(
            detection_limit,
            detection_limit.new_full((), -math.inf),
        )
        censor_fill = torch.minimum(
            prepared.background_frames_dbz,
            below_detection,
        )
    elif analysis_config.censored_background_policy == "floor":
        censor_fill = prepared.frames_dbz.new_full(
            (),
            nowcast_config.min_dbz,
        )
    else:
        detection_limit = prepared.frames_dbz.new_full(
            (),
            analysis_config.detection_limit_dbz,
        )
        censor_fill = torch.nextafter(
            detection_limit,
            detection_limit.new_full((), -math.inf),
        )
    canonical_observations = torch.where(
        censored,
        censor_fill,
        prepared.frames_dbz,
    )
    observations = AnalysisObservations(
        dbz=observed_dbz.detach().clone(),
        std_dbz=std.detach().clone(),
        quality_weight=quality.detach().clone(),
        valid_mask=valid.detach().clone(),
        detected_mask=detected.detach().clone(),
        censored_mask=censored.detach().clone(),
        missing_mask=prepared.missing_mask.detach().clone(),
        qc_rejected_mask=prepared.qc_rejected_mask.detach().clone(),
        common_bias_group_index=common_bias_group_index,
        common_bias_mode_weights=common_bias_mode_weights,
    )
    _validate_observations(observations)
    _validate_observation_common_bias_contract(
        observations,
        analysis_config,
    )

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
    baseline_frames_dbz = torch.where(
        prepared.observed_mask,
        prepared.frames_dbz,
        prepared.background_frames_dbz,
    )
    initial_background_dbz = torch.where(
        prepared.observed_mask[0],
        canonical_observations[0],
        prepared.background_frames_dbz[0],
    )
    prior_std_dbz = None
    prior_valid_mask = None
    if neural_prior is not None:
        neural_prior.validate_integrity()
        if (
            neural_prior.initial_background_dbz.shape
            != initial_background_dbz.shape
            or neural_prior.initial_background_dbz.dtype != frames_dbz.dtype
            or neural_prior.initial_background_dbz.device != frames_dbz.device
        ):
            raise ValueError(
                "neural-prior background must match the radar grid, dtype, and device"
            )
        prior_background = neural_prior.initial_background_dbz.clamp(
            nowcast_config.min_dbz,
            nowcast_config.max_dbz,
        )
        prior_valid_mask = neural_prior.valid_mask.to(frames_dbz.device) & (
            neural_prior.support_probability.to(frames_dbz) >= 0.5
        )
        prior_std_dbz = neural_prior.std_dbz.to(frames_dbz)
        prior_background = torch.where(
            prior_valid_mask,
            prior_background,
            initial_background_dbz,
        )
        prior_echo = dbz_to_echo(
            prior_background,
            min_dbz=nowcast_config.min_dbz,
            max_dbz=nowcast_config.max_dbz,
        )
        prior_support = prior_valid_mask & (
            prior_echo > analysis_config.transform_epsilon
        )
        added_support = prior_support & ~initial_support
        if neural_prior.support_policy == "causal_clip":
            prior_background = torch.where(
                initial_support,
                prior_background,
                prior_background.new_full((), nowcast_config.min_dbz),
            )
            prior_valid_mask = prior_valid_mask & initial_support
        else:
            if grid_time_contract is None and bool(torch.any(added_support)):
                raise ValueError(
                    "expanded neural-prior support requires a grid contract"
                )
            added_area = (
                0.0
                if grid_time_contract is None
                else int(torch.count_nonzero(added_support))
                * grid_time_contract.cell_area_m2
                / 1.0e6
            )
            added_echo = float(
                torch.sum(
                    prior_echo.masked_select(added_support)
                ).detach()
                * (
                    1.0
                    if grid_time_contract is None
                    else grid_time_contract.cell_area_m2 / 1.0e6
                )
            )
            if added_area > neural_prior.maximum_added_area_km2:
                raise ValueError("neural-prior added area exceeds its budget")
            if added_echo > neural_prior.maximum_added_echo_integral:
                raise ValueError("neural-prior added echo exceeds its budget")
            initial_support = initial_support | prior_support
        initial_background_dbz = prior_background
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
    observation_whitener = _freeze_observation_whitener(
        observations,
        analysis_config,
    )
    whitener_bytes = _retained_tensor_bytes(observation_whitener)
    if whitener_bytes > analysis_config.maximum_frozen_whitener_bytes:
        raise ValueError("frozen observation whitener exceeds its byte budget")
    frozen = FrozenOuterState(
        input_frames_dbz=frames_dbz.detach().clone(),
        background_frames_dbz=(
            None
            if background_frames_dbz is None
            else background_frames_dbz.detach().clone()
        ),
        initial_background_dbz=initial_background_dbz.detach().clone(),
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
        observation_whitener=observation_whitener,
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
        observation_derived_initial_background=(neural_prior is None),
        neural_prior_std_dbz=(
            None if prior_std_dbz is None else prior_std_dbz.detach().clone()
        ),
        neural_prior_valid_mask=(
            None if prior_valid_mask is None else prior_valid_mask.detach().clone()
        ),
        neural_prior_dependency=(
            None if neural_prior is None else neural_prior.dependency
        ),
        neural_prior_application_digest=(
            None if neural_prior is None else neural_prior.application_digest
        ),
        neural_prior_raw_background_dbz=(
            None
            if neural_prior is None
            else neural_prior.initial_background_dbz.detach().clone()
        ),
        neural_prior_execution_contract_digest=(
            None
            if neural_prior is None
            else neural_prior.inference_evidence.execution_contract_digest
        ),
        neural_prior_role=(None if neural_prior is None else neural_prior.role),
    )
    linearization_bytes = _retained_tensor_bytes((observations, frozen))
    if linearization_bytes > analysis_config.maximum_linearization_bytes:
        raise ValueError("P1 linearization exceeds its retained byte budget")
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

    analyzed_dbz = _initial_analysis_dbz(field_control, frozen)
    initial_echo = dbz_to_echo(
        analyzed_dbz,
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


def _initial_analysis_dbz(
    field_control: Tensor,
    frozen: FrozenOuterState,
) -> Tensor:
    config = frozen.analysis_config
    floor_dbz = frozen.nowcast_config.min_dbz
    background_offset = (
        frozen.initial_background_dbz - floor_dbz
    ) / config.echo_transform_scale_dbz
    background_latent = _softplus_inverse(
        background_offset.clamp_min(config.transform_epsilon)
    )
    return floor_dbz + config.echo_transform_scale_dbz * F.softplus(
        background_latent
        + (
            config.initial_increment_scale_dbz
            / config.echo_transform_scale_dbz
        )
        * field_control
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
    _validate_observation_common_bias_contract(
        observations,
        frozen.analysis_config,
    )
    _validate_control(control, frozen)
    return _whitened_observation_residual(control, observations, frozen)


def _whitened_observation_residual(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    standardized = (
        torch.sqrt(observations.quality_weight)
        * _observation_residual(control, observations, frozen)
        / observations.std_dbz
    )
    return _apply_observation_error_whitener(
        standardized,
        observations,
        frozen.analysis_config,
        whitener=frozen.observation_whitener,
    )


def _freeze_observation_whitener(
    observations: AnalysisObservations,
    config: AnalysisConfig,
) -> FrozenObservationWhitener:
    """Build constants reused by every frozen residual evaluation."""

    per_frame = config.observation_common_bias_scope == "per_frame"
    bias_std = config.observation_common_bias_std_dbz
    if bias_std == 0.0:
        return FrozenObservationWhitener(None, None, per_frame)
    mode = (
        torch.sqrt(observations.quality_weight)
        / observations.std_dbz
    ) * observations.valid_mask.to(dtype=observations.dbz.dtype)
    mode_weights = observations.common_bias_mode_weights
    if mode_weights is None:
        return FrozenObservationWhitener(
            mode.detach(),
            None,
            per_frame,
        )
    weights = _common_bias_mode_weights_by_frame(mode_weights)
    gram = bias_std**2 * torch.einsum(
        "tkhw,tlhw,thw->tkl" if per_frame else "tkhw,tlhw,thw->kl",
        weights,
        weights,
        mode.square(),
    )
    correction = _low_rank_inverse_sqrt_correction_from_gram(gram)
    return FrozenObservationWhitener(
        mode.detach(),
        correction.detach(),
        per_frame,
    )


def _validate_frozen_observation_whitener(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> None:
    expected = _freeze_observation_whitener(
        observations,
        frozen.analysis_config,
    )
    actual = frozen.observation_whitener
    if actual.per_frame != expected.per_frame:
        raise ValueError("frozen observation whitener scope mismatch")
    for name in ("mode", "overlapping_correction"):
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if (actual_value is None) != (expected_value is None) or (
            actual_value is not None
            and expected_value is not None
            and not torch.equal(actual_value, expected_value)
        ):
            raise ValueError("frozen observation whitener content mismatch")


def _apply_observation_error_whitener(
    values: Tensor,
    observations: AnalysisObservations,
    config: AnalysisConfig,
    *,
    whitener: FrozenObservationWhitener | None = None,
) -> Tensor:
    """Apply the symmetric inverse square root of a low-rank bias covariance.

    After diagonal standardization, an additive dBZ bias with standard
    deviation ``sigma_b`` gives ``C = I + sigma_b**2 a a.T``, where
    ``a = sqrt(quality) / std``.  The rank-one inverse square root is applied
    independently per frame or once across all three frames without forming
    ``C``.  A positive tile size partitions the spatial domain into
    independent rank-one blocks, including ragged blocks at the lower and
    right edges.
    """

    if values.shape != observations.dbz.shape:
        raise ValueError("observation whitener input must match observations")
    bias_std = config.observation_common_bias_std_dbz
    if bias_std == 0.0:
        return values
    whitener = whitener or _freeze_observation_whitener(observations, config)
    mode = whitener.mode
    if mode is None or mode.shape != values.shape:
        raise ValueError("frozen observation whitener is incompatible")
    if observations.common_bias_mode_weights is not None:
        correction = whitener.overlapping_correction
        if correction is None:
            raise ValueError("frozen overlapping whitener is incomplete")
        counter = _WHITENER_APPLY_COUNTER.get()
        if counter is not None:
            counter[0] += 1
            limit = _WHITENER_TOTAL_OPERATION_LIMIT.get()
            if limit is not None and counter[0] * limit[0] > limit[1]:
                raise ValueError(
                    "common-bias whitener total operation budget exhausted"
                )
        return _apply_compact_low_rank_whitener(
            values,
            observations.common_bias_mode_weights,
            mode,
            correction,
            bias_std=bias_std,
            per_frame=whitener.per_frame,
        )
    if observations.common_bias_group_index is not None:
        return _apply_grouped_observation_error_whitener(
            values,
            mode,
            group_index=observations.common_bias_group_index,
            bias_std=bias_std,
        )
    tile_size = config.observation_common_bias_tile_size_px
    if tile_size > 0:
        return _apply_tiled_observation_error_whitener(
            values,
            mode,
            bias_std=bias_std,
            tile_size=tile_size,
            temporal_scope=config.observation_common_bias_scope,
        )
    dimensions = (
        (-2, -1)
        if config.observation_common_bias_scope == "per_frame"
        else (-3, -2, -1)
    )
    mode_norm_squared = torch.sum(
        mode.square(),
        dim=dimensions,
        keepdim=True,
    )
    projection = torch.sum(
        mode * values,
        dim=dimensions,
        keepdim=True,
    )
    positive = mode_norm_squared > 0.0
    inverse_sqrt_eigenvalue = torch.rsqrt(
        1.0 + bias_std**2 * mode_norm_squared
    )
    coefficient = torch.where(
        positive,
        (1.0 - inverse_sqrt_eigenvalue)
        / mode_norm_squared.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(mode_norm_squared),
    )
    return values - coefficient * mode * projection


def _low_rank_inverse_sqrt_correction(
    basis: Tensor,
) -> Tensor:
    return _low_rank_inverse_sqrt_correction_from_gram(basis @ basis.mT)


def _low_rank_inverse_sqrt_correction_from_gram(gram: Tensor) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    correction_eigenvalues = torch.where(
        eigenvalues > torch.finfo(gram.dtype).eps,
        (torch.rsqrt(1.0 + eigenvalues) - 1.0) / eigenvalues,
        eigenvalues.new_full((), -0.5),
    )
    return eigenvectors @ (
        correction_eigenvalues.unsqueeze(-1) * eigenvectors.mT
    )


def _apply_compact_low_rank_whitener(
    values: Tensor,
    mode_weights: Tensor,
    mode: Tensor,
    correction: Tensor,
    *,
    bias_std: float,
    per_frame: bool,
) -> Tensor:
    weights = _common_bias_mode_weights_by_frame(mode_weights)
    weighted_values = bias_std * mode * values
    if per_frame:
        projection = torch.einsum(
            "tkhw,thw->tk",
            weights,
            weighted_values,
        )
        coefficients = torch.einsum("tkl,tl->tk", correction, projection)
        adjustment = bias_std * mode * torch.einsum(
            "tkhw,tk->thw",
            weights,
            coefficients,
        )
        return values + adjustment
    projection = torch.einsum(
        "tkhw,thw->k",
        weights,
        weighted_values,
    )
    coefficients = correction @ projection
    adjustment = bias_std * mode * torch.einsum(
        "tkhw,k->thw",
        weights,
        coefficients,
    )
    return values + adjustment


def _common_bias_mode_weights_by_frame(mode_weights: Tensor) -> Tensor:
    return mode_weights.unsqueeze(0) if mode_weights.ndim == 3 else mode_weights


def _apply_grouped_observation_error_whitener(
    values: Tensor,
    mode: Tensor,
    *,
    group_index: Tensor,
    bias_std: float,
) -> Tensor:
    flat_group = group_index.flatten()
    active = flat_group >= 0
    if not bool(torch.any(active)):
        return values
    group_count = int(torch.max(flat_group[active]).detach()) + 1
    safe_group = torch.where(active, flat_group, torch.zeros_like(flat_group))
    flat_mode = mode.flatten()
    flat_values = values.flatten()
    active_float = active.to(dtype=values.dtype)
    mode_norm_squared = values.new_zeros((group_count,)).scatter_add(
        0,
        safe_group,
        flat_mode.square() * active_float,
    )
    projection = values.new_zeros((group_count,)).scatter_add(
        0,
        safe_group,
        flat_mode * flat_values * active_float,
    )
    inverse_sqrt_eigenvalue = torch.rsqrt(
        1.0 + bias_std**2 * mode_norm_squared
    )
    coefficient = torch.where(
        mode_norm_squared > 0.0,
        (1.0 - inverse_sqrt_eigenvalue)
        / mode_norm_squared.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(mode_norm_squared),
    )
    adjustment = (
        coefficient[safe_group]
        * flat_mode
        * projection[safe_group]
        * active_float
    )
    return (flat_values - adjustment).reshape_as(values)


def _apply_tiled_observation_error_whitener(
    values: Tensor,
    mode: Tensor,
    *,
    bias_std: float,
    tile_size: int,
    temporal_scope: ObservationCommonBiasScope,
) -> Tensor:
    frame_count, height, width = values.shape
    padded_height = ((height + tile_size - 1) // tile_size) * tile_size
    padded_width = ((width + tile_size - 1) // tile_size) * tile_size
    padding = (0, padded_width - width, 0, padded_height - height)

    def blocks(tensor: Tensor) -> Tensor:
        return F.pad(tensor, padding).reshape(
            frame_count,
            padded_height // tile_size,
            tile_size,
            padded_width // tile_size,
            tile_size,
        ).permute(0, 1, 3, 2, 4)

    value_blocks = blocks(values)
    mode_blocks = blocks(mode)
    dimensions = (
        (-2, -1)
        if temporal_scope == "per_frame"
        else (0, -2, -1)
    )
    mode_norm_squared = torch.sum(
        mode_blocks.square(),
        dim=dimensions,
        keepdim=True,
    )
    projection = torch.sum(
        mode_blocks * value_blocks,
        dim=dimensions,
        keepdim=True,
    )
    inverse_sqrt_eigenvalue = torch.rsqrt(
        1.0 + bias_std**2 * mode_norm_squared
    )
    coefficient = torch.where(
        mode_norm_squared > 0.0,
        (1.0 - inverse_sqrt_eigenvalue)
        / mode_norm_squared.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(mode_norm_squared),
    )
    adjusted_blocks = (
        value_blocks - coefficient * mode_blocks * projection
    )
    adjusted = adjusted_blocks.permute(0, 1, 3, 2, 4).reshape(
        frame_count,
        padded_height,
        padded_width,
    )
    return adjusted[:, :height, :width]


def _observation_marginal_precision(
    observations: AnalysisObservations,
    config: AnalysisConfig,
) -> Tensor:
    quality = observations.quality_weight
    return quality / _observation_effective_std_dbz(
        observations,
        config,
    ).square()


def _observation_effective_std_dbz(
    observations: AnalysisObservations,
    config: AnalysisConfig,
) -> Tensor:
    variance_factor = torch.ones_like(observations.quality_weight)
    if observations.common_bias_group_index is not None:
        variance_factor = (
            observations.common_bias_group_index >= 0
        ).to(dtype=observations.quality_weight.dtype)
    elif observations.common_bias_mode_weights is not None:
        variance_factor = torch.sum(
            _common_bias_mode_weights_by_frame(
                observations.common_bias_mode_weights
            ).square(),
            dim=1,
        )
    return torch.sqrt(
        observations.std_dbz.square()
        + observations.quality_weight
        * config.observation_common_bias_std_dbz**2
        * variance_factor
    )


def freeze_irls_weights(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> FrozenOuterState:
    _validate_observation_common_bias_contract(
        observations,
        frozen.analysis_config,
    )
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
            _control_prior_residual(control, frozen),
            _field_smoothness_residual(control, frozen),
        )
    )


def _control_prior_residual(
    control: Tensor,
    frozen: FrozenOuterState,
) -> Tensor:
    """Scale neural-prior field increments by retained prior uncertainty."""

    prior_std = frozen.neural_prior_std_dbz
    prior_valid = frozen.neural_prior_valid_mask
    if prior_std is None or prior_valid is None:
        return control
    field_size = frozen.active_field_index.numel()
    residual = control.clone()
    flat_std = prior_std.flatten()[frozen.active_field_index]
    flat_valid = prior_valid.flatten()[frozen.active_field_index]
    height, width = frozen.initial_background_dbz.shape
    field_control = torch.zeros_like(frozen.initial_background_dbz).flatten().scatter(
        0,
        frozen.active_field_index,
        control[:field_size],
    ).reshape(height, width)
    analyzed = _initial_analysis_dbz(field_control, frozen).flatten()[
        frozen.active_field_index
    ]
    background = frozen.initial_background_dbz.flatten()[
        frozen.active_field_index
    ]
    standardized = (analyzed - background) / flat_std.clamp_min(
        frozen.analysis_config.transform_epsilon
    )
    residual[:field_size] = torch.where(
        flat_valid,
        standardized,
        control[:field_size],
    )
    return residual


@dataclass(frozen=True)
class _LinearizationStationarity:
    residual_norm: float
    gradient_norm: float
    field_gradient_rms: float
    field_gradient_max: float
    dynamics_gradient_max: float
    relative_stationarity: float


@dataclass(frozen=True)
class _FinalLinearizationPolish:
    control: Tensor
    frozen: FrozenOuterState
    exact_objective: float
    stationarity: _LinearizationStationarity
    robust_stationarity: _LinearizationStationarity
    relative_weight_change: float
    iterations: int
    pcg_iterations: int


def _block_stationarity(
    gradient: Tensor,
    field_size: int,
) -> tuple[float, float, float, float]:
    """Return field RMS/max and the standardized dynamics maximum."""

    field = gradient[:field_size]
    field_rms = (
        0.0
        if field_size == 0
        else float(torch.sqrt(torch.mean(field.square())).detach())
    )
    field_max = (
        0.0
        if field_size == 0
        else float(torch.amax(torch.abs(field)).detach())
    )
    dynamics = gradient[field_size:]
    dynamics_max = (
        0.0
        if dynamics.numel() == 0
        else float(torch.amax(torch.abs(dynamics)).detach())
    )
    return field_rms, field_max, dynamics_max, max(field_rms, dynamics_max)


def _stationarity_is_acceptable(
    stationarity: _LinearizationStationarity,
    *,
    block_tolerance: float,
    field_max_tolerance: float,
) -> bool:
    return (
        math.isfinite(stationarity.relative_stationarity)
        and stationarity.relative_stationarity <= block_tolerance
        and math.isfinite(stationarity.field_gradient_max)
        and stationarity.field_gradient_max <= field_max_tolerance
    )


def _linearization_stationarity(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> _LinearizationStationarity:
    residual_fn: Callable[[Tensor], Tensor] = lambda value: residual_vector(
        value,
        observations,
        frozen,
    )
    vjp_result = torch.func.vjp(residual_fn, control)
    residual = cast(Tensor, vjp_result[0])
    pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        vjp_result[1],
    )
    gradient = pullback(residual)[0]
    residual_norm = float(torch.linalg.vector_norm(residual).detach())
    gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
    field_rms, field_max, dynamics_max, relative_stationarity = (
        _block_stationarity(gradient, frozen.active_field_index.numel())
    )
    return _LinearizationStationarity(
        residual_norm=residual_norm,
        gradient_norm=gradient_norm,
        field_gradient_rms=field_rms,
        field_gradient_max=field_max,
        dynamics_gradient_max=dynamics_max,
        relative_stationarity=relative_stationarity,
    )


def _robust_stationarity(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> _LinearizationStationarity:
    objective = robust_objective(control, observations, frozen)
    gradient = torch.func.grad(robust_objective, argnums=0)(
        control,
        observations,
        frozen,
    )
    gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
    objective_value = float(objective.detach())
    field_rms, field_max, dynamics_max, relative_stationarity = (
        _block_stationarity(gradient, frozen.active_field_index.numel())
    )
    return _LinearizationStationarity(
        residual_norm=objective_value,
        gradient_norm=gradient_norm,
        field_gradient_rms=field_rms,
        field_gradient_max=field_max,
        dynamics_gradient_max=dynamics_max,
        relative_stationarity=relative_stationarity,
    )


def _relative_irls_weight_change(
    previous: FrozenOuterState,
    current: FrozenOuterState,
) -> float:
    difference = torch.linalg.vector_norm(
        current.irls_sqrt_weight - previous.irls_sqrt_weight
    )
    scale = 1.0 + torch.linalg.vector_norm(previous.irls_sqrt_weight)
    return float((difference / scale).detach())


def _polish_final_linearization(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    exact_objective: float,
) -> _FinalLinearizationPolish:
    config = frozen.analysis_config
    previous_frozen = frozen
    frozen = freeze_irls_weights(control, observations, frozen)
    relative_weight_change = _relative_irls_weight_change(
        previous_frozen,
        frozen,
    )
    control = control.detach()
    damping = config.initial_damping
    polish_iterations = 0
    total_pcg_iterations = 0
    stationarity: _LinearizationStationarity | None = None
    robust_stationarity: _LinearizationStationarity | None = None

    for _ in range(config.maximum_final_linearization_polish_iterations):
        residual_fn: Callable[[Tensor], Tensor] = lambda value: residual_vector(
            value,
            observations,
            frozen,
        )
        vjp_result = torch.func.vjp(residual_fn, control)
        residual = cast(Tensor, vjp_result[0])
        pullback = cast(
            Callable[[Tensor], tuple[Tensor]],
            vjp_result[1],
        )
        gradient = pullback(residual)[0]
        residual_norm = float(torch.linalg.vector_norm(residual).detach())
        gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
        field_rms, field_max, dynamics_max, relative_stationarity = (
            _block_stationarity(gradient, frozen.active_field_index.numel())
        )
        stationarity = _LinearizationStationarity(
            residual_norm=residual_norm,
            gradient_norm=gradient_norm,
            field_gradient_rms=field_rms,
            field_gradient_max=field_max,
            dynamics_gradient_max=dynamics_max,
            relative_stationarity=relative_stationarity,
        )
        frozen_stationary = _stationarity_is_acceptable(
            stationarity,
            block_tolerance=(
                config.final_linearization_relative_stationarity_tolerance
            ),
            field_max_tolerance=config.final_field_gradient_max_tolerance,
        )
        if frozen_stationary:
            refreshed = freeze_irls_weights(control, observations, frozen)
            relative_weight_change = _relative_irls_weight_change(
                frozen,
                refreshed,
            )
            frozen = refreshed
            robust_stationarity = _robust_stationarity(
                control,
                observations,
                frozen,
            )
            if (
                _stationarity_is_acceptable(
                    robust_stationarity,
                    block_tolerance=(
                        config.final_robust_relative_stationarity_tolerance
                    ),
                    field_max_tolerance=(
                        config.final_field_gradient_max_tolerance
                    ),
                )
                and relative_weight_change
                <= config.final_irls_relative_weight_tolerance
            ):
                stationarity = _linearization_stationarity(
                    control,
                    observations,
                    frozen,
                )
                break
            stationarity = None
            continue

        def normal_product(vector: Tensor) -> Tensor:
            jacobian_vector = cast(
                Tensor,
                torch.func.jvp(
                    residual_fn,
                    (control,),
                    (vector,),
                )[1],
            )
            return pullback(jacobian_vector)[0]

        current_frozen_objective = 0.5 * float(torch.dot(residual, residual))
        current_trajectory = _analysis_trajectory(control, frozen)
        current_amplitude = _amplitude_diagnostics(
            observations,
            frozen,
            current_trajectory,
            include_spatial_diagnostics=False,
        )
        accepted = False
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

            for backtrack in range(24):
                candidate = control + (0.5**backtrack) * linear.solution
                same_remap_branch = _analysis_remap_cells_match(
                    candidate,
                    frozen,
                )
                candidate_frozen = (
                    frozen
                    if same_remap_branch
                    else freeze_irls_weights(candidate, observations, frozen)
                )
                try:
                    candidate_residual = residual_vector(
                        candidate,
                        observations,
                        candidate_frozen,
                    )
                    candidate_frozen_objective = 0.5 * float(
                        torch.dot(candidate_residual, candidate_residual).detach()
                    )
                    candidate_exact_tensor, candidate_trajectory = (
                        _evaluate_control(
                            candidate,
                            observations,
                            candidate_frozen,
                        )
                    )
                    candidate_exact = float(candidate_exact_tensor.detach())
                except (EchoPositivityError, RuntimeError, ValueError):
                    continue
                if not (
                    math.isfinite(candidate_frozen_objective)
                    and (
                        not same_remap_branch
                        or candidate_frozen_objective
                        < current_frozen_objective
                    )
                    and math.isfinite(candidate_exact)
                    and candidate_exact
                    <= exact_objective
                    + _objective_comparison_tolerance(
                        exact_objective,
                        candidate_exact,
                        control.dtype,
                    )
                ):
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
                if not _analysis_window_is_representable(
                    candidate_frozen,
                    candidate_trajectory.displacement_yx,
                ):
                    continue
                if (
                    config.motion_increment_scale_mps is None
                    and not _motion_is_admissible(
                        candidate_trajectory.displacement_yx,
                        candidate_frozen,
                    )
                ):
                    continue
                control = candidate.detach()
                refreshed = freeze_irls_weights(
                    control,
                    observations,
                    candidate_frozen,
                )
                relative_weight_change = _relative_irls_weight_change(
                    frozen,
                    refreshed,
                )
                frozen = refreshed
                exact_objective = candidate_exact
                damping = max(config.minimum_damping, 0.5 * damping)
                polish_iterations += 1
                stationarity = None
                robust_stationarity = None
                accepted = True
                break
            if accepted:
                break
            damping = min(config.maximum_damping, 4.0 * damping)
        if not accepted:
            break

    retained_frozen = freeze_irls_weights(control, observations, frozen)
    relative_weight_change = _relative_irls_weight_change(
        frozen,
        retained_frozen,
    )
    frozen = retained_frozen
    stationarity = _linearization_stationarity(control, observations, frozen)
    robust_stationarity = _robust_stationarity(
        control,
        observations,
        frozen,
    )
    if not _analysis_remap_cells_match(control, frozen):
        raise RuntimeError("final linearization changed its frozen remap cell")
    return _FinalLinearizationPolish(
        control=control,
        frozen=frozen,
        exact_objective=exact_objective,
        stationarity=stationarity,
        robust_stationarity=robust_stationarity,
        relative_weight_change=relative_weight_change,
        iterations=polish_iterations,
        pcg_iterations=total_pcg_iterations,
    )


def _objective_comparison_tolerance(
    left: float,
    right: float,
    dtype: torch.dtype,
) -> float:
    return 32.0 * torch.finfo(dtype).eps * max(1.0, abs(left), abs(right))


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


def _field_conditioned_dynamics_gram(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    residual_fn: Callable[[Tensor], Tensor],
    dynamics_columns: list[Tensor],
    dynamics_gram: Tensor,
) -> tuple[Tensor | None, float | None]:
    field_size = frozen.active_field_index.numel()
    if field_size == 0:
        return dynamics_gram.clone(), 0.0

    observation_vjp = torch.func.vjp(residual_fn, control)
    observation_pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        observation_vjp[1],
    )
    full_residual_fn: Callable[[Tensor], Tensor] = lambda value: residual_vector(
        value,
        observations,
        frozen,
    )
    full_vjp = torch.func.vjp(full_residual_fn, control)
    full_pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        full_vjp[1],
    )

    def field_normal_product(field_direction: Tensor) -> Tensor:
        direction = torch.zeros_like(control)
        direction[:field_size] = field_direction
        full_jacobian_vector = cast(
            Tensor,
            torch.func.jvp(
                full_residual_fn,
                (control,),
                (direction,),
            )[1],
        )
        return cast(Tensor, full_pullback(full_jacobian_vector)[0])[
            :field_size
        ].detach()

    field_rhs = [
        cast(Tensor, observation_pullback(column)[0])[:field_size].detach()
        for column in dynamics_columns
    ]
    solutions: list[Tensor] = []
    relative_residuals: list[float] = []
    for rhs in field_rhs:
        try:
            solution = pcg(
                field_normal_product,
                rhs,
                rtol=frozen.analysis_config.pcg_relative_tolerance,
                max_iterations=frozen.analysis_config.maximum_pcg_iterations,
            )
        except (RuntimeError, ValueError):
            return None, None
        if not solution.converged:
            return None, max(
                (*relative_residuals, solution.relative_residual)
            )
        solutions.append(solution.solution)
        relative_residuals.append(solution.relative_residual)

    correction_values = [
        [_scaled_dot(left, right) for right in solutions]
        for left in field_rhs
    ]
    correction = torch.tensor(correction_values, dtype=torch.float64)
    conditioned = dynamics_gram - correction
    conditioned = 0.5 * (conditioned + conditioned.mT)
    if not bool(torch.all(torch.isfinite(conditioned))):
        return None, max(relative_residuals)
    return conditioned, max(relative_residuals)


def _identifiability_diagnostics(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    trajectory: AnalysisTrajectory,
    *,
    include_field_conditioned: bool = True,
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

    conditioned_gram: Tensor | None = None
    field_conditioning_residual: float | None = None
    if include_field_conditioned:
        conditioned_gram, field_conditioning_residual = (
            _field_conditioned_dynamics_gram(
                control,
                observations,
                frozen,
                residual_fn,
                dynamics_columns,
                gram,
            )
        )
    conditioned_eigenvalue_tuple: tuple[float, float, float] | None = None
    conditioned_information_trace: float | None = None
    conditioned_effective_dimension: float | None = None
    conditioned_posterior_covariance: Tensor | None = None
    if conditioned_gram is not None:
        conditioned_scale = max(
            1.0,
            float(torch.amax(torch.abs(conditioned_gram)).detach()),
        )
        conditioned_negative_tolerance = (
            256.0 * torch.finfo(torch.float64).eps * conditioned_scale
        )
        raw_conditioned_eigenvalues = torch.linalg.eigvalsh(conditioned_gram)
        if bool(torch.all(torch.isfinite(raw_conditioned_eigenvalues))) and (
            float(raw_conditioned_eigenvalues[0])
            >= -conditioned_negative_tolerance
        ):
            conditioned_eigenvalues = torch.clamp_min(
                raw_conditioned_eigenvalues,
                0.0,
            )
            conditioned_eigenvalue_tuple = (
                float(conditioned_eigenvalues[0]),
                float(conditioned_eigenvalues[1]),
                float(conditioned_eigenvalues[2]),
            )
            conditioned_information_trace = float(
                torch.sum(conditioned_eigenvalues)
            )
            conditioned_effective_dimension = float(
                torch.sum(
                    conditioned_eigenvalues
                    / (1.0 + conditioned_eigenvalues)
                )
            )
            conditioned_regularized = conditioned_gram + torch.eye(
                3,
                dtype=conditioned_gram.dtype,
                device=conditioned_gram.device,
            )
            cholesky, info = torch.linalg.cholesky_ex(
                conditioned_regularized
            )
            if int(info) == 0:
                conditioned_posterior_covariance = torch.cholesky_inverse(
                    cholesky
                )

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
        field_conditioned_dynamics_data_gram_eigenvalues=(
            conditioned_eigenvalue_tuple
        ),
        field_conditioned_dynamics_data_information_trace=(
            conditioned_information_trace
        ),
        field_conditioned_dynamics_data_effective_dimension=(
            conditioned_effective_dimension
        ),
        field_conditioning_maximum_relative_residual=(
            field_conditioning_residual
        ),
        field_conditioned_dynamics_posterior_covariance=(
            conditioned_posterior_covariance
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


def _posterior_physical_dynamics_uncertainty(
    control: Tensor,
    frozen: FrozenOuterState,
    diagnostics: _IdentifiabilityDiagnostics | None,
) -> tuple[Tensor, Tensor]:
    unavailable = control.new_full((), torch.nan)
    grid_time_contract = frozen.grid_time_contract
    if diagnostics is None or grid_time_contract is None:
        return unavailable, unavailable.clone()
    covariance = (
        diagnostics.field_conditioned_dynamics_posterior_covariance
    )
    if covariance is None:
        return unavailable, unavailable.clone()

    field_size = frozen.active_field_index.numel()
    dynamics_control = control[field_size:]

    def physical_dynamics(value: Tensor) -> Tensor:
        displacement, growth = _decode_dynamics(
            value,
            frozen.baseline_state,
            frozen.analysis_config,
            frozen.nowcast_config,
            frozen.motion_limits_yx,
            grid_time_contract,
        )
        projected_velocity = grid_time_contract.projected_velocity_xy(
            displacement,
            frozen.nowcast_config.interval_minutes,
        )
        return torch.cat((projected_velocity, growth.reshape(1)))

    decode_jacobian = cast(
        Tensor,
        torch.func.jacrev(physical_dynamics)(dynamics_control),
    ).detach()
    covariance = covariance.to(
        dtype=decode_jacobian.dtype,
        device=decode_jacobian.device,
    )
    physical_covariance = decode_jacobian @ covariance @ decode_jacobian.mT
    physical_covariance = 0.5 * (
        physical_covariance + physical_covariance.mT
    )
    if not bool(torch.all(torch.isfinite(physical_covariance))):
        return unavailable, unavailable.clone()
    velocity_variance = torch.linalg.eigvalsh(physical_covariance[:2, :2])[-1]
    growth_variance = physical_covariance[2, 2]
    covariance_scale = max(
        1.0,
        float(torch.amax(torch.abs(physical_covariance))),
    )
    negative_tolerance = (
        256.0 * torch.finfo(control.dtype).eps * covariance_scale
    )
    if (
        float(velocity_variance) < -negative_tolerance
        or float(growth_variance) < -negative_tolerance
    ):
        return unavailable, unavailable.clone()
    return (
        torch.sqrt(velocity_variance.clamp_min(0.0)),
        torch.sqrt(growth_variance.clamp_min(0.0)),
    )


def _posterior_saturation_is_safe(
    motion_margin_mps: float | None,
    growth_margin_per_step: Tensor,
    velocity_uncertainty_mps: Tensor,
    growth_uncertainty_per_step: Tensor,
    config: NowcastConfig,
) -> bool:
    """Require the posterior dynamics mass to remain inside safe limits."""

    if motion_margin_mps is None or not bool(
        torch.isfinite(velocity_uncertainty_mps)
        & torch.isfinite(growth_uncertainty_per_step)
    ):
        return False
    multiplier = config.p1_posterior_saturation_sigma_multiplier
    tolerance = config.contract_absolute_tolerance
    return (
        motion_margin_mps + tolerance
        >= config.p1_motion_saturation_safe_margin_mps
        + multiplier * float(velocity_uncertainty_mps)
        and float(growth_margin_per_step) + tolerance
        >= config.p1_growth_saturation_safe_margin_per_step
        + multiplier * float(growth_uncertainty_per_step)
    )


def _p1_saturation_uncertainty(
    posterior_velocity_uncertainty_mps: Tensor,
    posterior_log_growth_uncertainty_per_step: Tensor,
    motion_speed_margin_mps: float | None,
    growth_margin_per_step: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor]:
    unavailable = posterior_velocity_uncertainty_mps.new_full((), torch.nan)
    if (
        motion_speed_margin_mps is None
        or not bool(torch.isfinite(posterior_velocity_uncertainty_mps))
        or not bool(
            torch.isfinite(posterior_log_growth_uncertainty_per_step)
        )
    ):
        return unavailable, unavailable.clone()

    motion_exposure = 1.0 - (
        posterior_velocity_uncertainty_mps.new_tensor(
            motion_speed_margin_mps
        )
        / config.p1_motion_saturation_safe_margin_mps
    )
    growth_exposure = 1.0 - (
        growth_margin_per_step
        / config.p1_growth_saturation_safe_margin_per_step
    )
    multiplier = config.p1_saturation_uncertainty_multiplier
    return (
        posterior_velocity_uncertainty_mps.new_tensor(
            config.forecast_velocity_uncertainty_mps * multiplier
        )
        * motion_exposure.clamp(0.0, 1.0),
        posterior_log_growth_uncertainty_per_step.new_tensor(
            config.forecast_log_growth_uncertainty_per_step * multiplier
        )
        * growth_exposure.clamp(0.0, 1.0),
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
        + 0.5
        * torch.dot(
            _control_prior_residual(control, frozen),
            _control_prior_residual(control, frozen),
        )
        + _field_smoothness_prior_cost(control, frozen)
    )


def solve_analysis(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    control: Tensor | None = None,
) -> AnalysisResult:
    _validate_observations(observations)
    _validate_observation_common_bias_contract(
        observations,
        frozen.analysis_config,
    )
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
        degraded=False,
        polish_final_linearization=True,
    )


def variational_nowcast(
    frames_dbz: Tensor,
    *,
    nowcast_config: NowcastConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    observation_std_dbz: float | Tensor | None = None,
    quality_weight: float | Tensor | None = None,
    qc_mask: Tensor | None = None,
    observation_common_bias_group_index: Tensor | None = None,
    observation_common_bias_mode_weights: Tensor | None = None,
    background_frames_dbz: Tensor | None = None,
    background_age_minutes: float | None = None,
    grid_time_contract: RadarGridTimeContract | None = None,
    operational_calibration_manifest: (
        OperationalCalibrationManifest | None
    ) = None,
    operational_calibration_approval_digest: str | None = None,
    operational_data_identity: OperationalDataIdentity | None = None,
    neural_prior: NeuralPriorApplication | None = None,
    input_plan_json: str | None = None,
    input_plan_digest: str | None = None,
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
        observation_common_bias_group_index=(
            observation_common_bias_group_index
        ),
        observation_common_bias_mode_weights=(
            observation_common_bias_mode_weights
        ),
        background_frames_dbz=background_frames_dbz,
        background_age_minutes=background_age_minutes,
        grid_time_contract=grid_time_contract,
        neural_prior=neural_prior,
    )
    analysis = solve_analysis(observations, frozen)
    (
        analysis_config_json,
        analysis_config_digest,
        analysis_input_digest,
    ) = _analysis_input_lineage(
        observations,
        frozen.analysis_config,
        neural_prior_application_digest=(
            None if neural_prior is None else neural_prior.application_digest
        ),
    )
    run = ForecastRunContract.from_inputs(
        nowcast_config,
        frames_dbz,
        observations.valid_mask,
        background_frames_dbz,
        background_age_minutes,
        observation_quality_weight=(
            observations.quality_weight * observations.valid_mask
        ),
        observation_std_dbz=observations.std_dbz,
        grid_time_contract=grid_time_contract,
        analysis_config_json=analysis_config_json,
        analysis_config_digest=analysis_config_digest,
        analysis_input_digest=analysis_input_digest,
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
        neural_prior_digest=(
            None if neural_prior is None else neural_prior.neural_prior_digest
        ),
        prior_application_digest=(
            None if neural_prior is None else neural_prior.application_digest
        ),
        prior_model_contract_digest=(
            None if neural_prior is None else neural_prior.model_contract_digest
        ),
        prior_feature_schema_digest=(
            None if neural_prior is None else neural_prior.feature_schema_digest
        ),
        prior_training_manifest_digest=(
            None if neural_prior is None else neural_prior.training_manifest_digest
        ),
        prior_inference_evidence_digest=(
            None
            if neural_prior is None
            else neural_prior.inference_evidence.evidence_digest
        ),
        prior_inference_algorithm_digest=(
            None
            if neural_prior is None
            else neural_prior.inference_evidence.inference_algorithm_digest
        ),
        prior_numerical_runtime_digest=(
            None
            if neural_prior is None
            else neural_prior.inference_evidence.numerical_runtime_digest
        ),
        prior_dependency=(None if neural_prior is None else neural_prior.dependency),
        prior_role=None if neural_prior is None else neural_prior.role,
        input_plan_json=input_plan_json,
        input_plan_digest=input_plan_digest,
    )
    if neural_prior is not None:
        evidence = neural_prior.inference_evidence
        if evidence.input_bundle_digest != run.input_bundle_digest:
            raise ValueError("neural-prior inference used a different input bundle")
        if evidence.input_frames_digest != tensor_digest(frames_dbz):
            raise ValueError("neural-prior inference used different radar frames")
    forecast = forecast_from_state(
        analysis.state,
        analysis.metadata,
        nowcast_config,
        run=run,
        audit=audit,
    )
    if analysis.linearization is not None:
        bound_linearization = replace(
            analysis.linearization,
            forecast_run_digest=forecast.forecast_run_digest,
        )
        analysis = replace(
            analysis,
            linearization=_content_address_linearization(
                analysis.control,
                bound_linearization,
            ),
        )
    return forecast, analysis


def _analysis_input_lineage(
    observations: AnalysisObservations,
    config: AnalysisConfig,
    *,
    neural_prior_application_digest: str | None = None,
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
            "version": "p1-analysis-input-v2",
            "analysis_config_digest": config_digest,
            "observation_std_dbz": tensor_digest(observations.std_dbz),
            "quality_weight": tensor_digest(observations.quality_weight),
            "neural_prior_application_digest": (
                neural_prior_application_digest
            ),
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
    standardized = (
        torch.sqrt(observations.quality_weight)
        * _observation_residual_from_prediction(
            prediction,
            observations,
            frozen.analysis_config,
        )
        / observations.std_dbz
    )
    residual = _apply_observation_error_whitener(
        standardized,
        observations,
        frozen.analysis_config,
        whitener=frozen.observation_whitener,
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
            polish_final_linearization=False,
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
    polish_final_linearization: bool = False,
) -> AnalysisResult:
    outer_converged = converged
    frozen = _freeze_analysis_remap_cells(control, frozen)
    initial_trajectory = _analysis_trajectory(control, frozen)
    initial_reachability_margin = _analysis_window_reachability_margin(
        frozen,
        initial_trajectory.displacement_yx,
    )
    if initial_reachability_margin < 0:
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "unrepresentable_analysis_window",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=initial_reachability_margin,
        )
    if polish_final_linearization and not degraded:
        polish = _polish_final_linearization(
            control,
            observations,
            frozen,
            final_objective,
        )
        control = polish.control
        frozen = polish.frozen
        final_objective = polish.exact_objective
        pcg_iterations += polish.pcg_iterations
        stationarity = polish.stationarity
        robust_stationarity = polish.robust_stationarity
        irls_relative_weight_change = polish.relative_weight_change
        polish_iterations = polish.iterations
    else:
        previous_frozen = frozen
        frozen = freeze_irls_weights(control, observations, frozen)
        irls_relative_weight_change = _relative_irls_weight_change(
            previous_frozen,
            frozen,
        )
        stationarity = _linearization_stationarity(
            control,
            observations,
            frozen,
        )
        robust_stationarity = _robust_stationarity(
            control,
            observations,
            frozen,
        )
        polish_iterations = 0
    final_linearization_stationary = _stationarity_is_acceptable(
        stationarity,
        block_tolerance=(
            frozen.analysis_config
            .final_linearization_relative_stationarity_tolerance
        ),
        field_max_tolerance=(
            frozen.analysis_config.final_field_gradient_max_tolerance
        ),
    )
    final_robust_stationary = _stationarity_is_acceptable(
        robust_stationarity,
        block_tolerance=(
            frozen.analysis_config.final_robust_relative_stationarity_tolerance
        ),
        field_max_tolerance=(
            frozen.analysis_config.final_field_gradient_max_tolerance
        ),
    )
    final_irls_fixed_point = (
        math.isfinite(irls_relative_weight_change)
        and irls_relative_weight_change
        <= frozen.analysis_config.final_irls_relative_weight_tolerance
    )
    converged = (
        final_linearization_stationary
        and final_robust_stationary
        and final_irls_fixed_point
    )
    if converged and not outer_converged:
        reason = "final_robust_irls_fixed_point"
    if frozen.analysis_config.execution_mode == "operational" and degraded:
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "degraded_operational_analysis",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=initial_reachability_margin,
        )
    if frozen.analysis_config.execution_mode == "operational" and not converged:
        failure_reason = "final_irls_not_fixed_point"
        if not final_linearization_stationary:
            failure_reason = "final_linearization_not_stationary"
        elif not final_robust_stationary:
            failure_reason = "final_robust_not_stationary"
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            failure_reason,
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=initial_reachability_margin,
        )
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
    analysis_degraded = (
        degraded
        or not converged
        or amplitude.has_insufficient_information
        or amplitude_confidence_failed
    )
    if (
        frozen.analysis_config.execution_mode == "operational"
        and analysis_degraded
    ):
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "degraded_operational_analysis",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=reachability_margin,
            amplitude_diagnostics=amplitude,
            amplitude_diagnostics_source="rejected_candidate",
        )
    motion_speed_saturation_margin = _motion_speed_saturation_margin(
        trajectory.displacement_yx,
        frozen,
    )
    motion_saturation_margin_mps = _motion_saturation_margin_mps(
        trajectory.displacement_yx,
        frozen,
    )
    growth_saturation_margin = (
        frozen.nowcast_config.max_log_growth_per_step
        - torch.abs(trajectory.log_growth_per_step)
    )
    if frozen.analysis_config.execution_mode == "operational" and (
        motion_speed_saturation_margin is None
        or motion_speed_saturation_margin
        + frozen.nowcast_config.contract_absolute_tolerance
        < frozen.nowcast_config.p1_motion_saturation_safe_margin_mps
        or float(growth_saturation_margin)
        + frozen.nowcast_config.contract_absolute_tolerance
        < frozen.nowcast_config.p1_growth_saturation_safe_margin_per_step
    ):
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "dynamics_saturation_margin",
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
    (
        source_support,
        background_source_support,
        background_fraction,
    ) = merge_current_support(
        frozen.observed_mask,
        initial_background_mask,
        trajectory.displacement_yx,
        frozen.nowcast_config,
    )
    observation_source_support = (
        source_support - background_source_support
    ).clamp(0.0, 1.0)
    background_used = (
        bool(
            torch.any(
                background_source_support
                > frozen.nowcast_config.support_presence_threshold
            )
        )
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
        include_field_conditioned=not analysis_degraded,
    )
    (
        posterior_velocity_uncertainty_mps,
        posterior_log_growth_uncertainty_per_step,
    ) = _posterior_physical_dynamics_uncertainty(
        control,
        frozen,
        identifiability,
    )
    if (
        frozen.analysis_config.execution_mode == "operational"
        and not _posterior_saturation_is_safe(
            motion_saturation_margin_mps,
            growth_saturation_margin,
            posterior_velocity_uncertainty_mps,
            posterior_log_growth_uncertainty_per_step,
            frozen.nowcast_config,
        )
    ):
        return _fallback_result(
            frozen,
            control,
            reference_objective,
            "posterior_dynamics_saturation_margin",
            outer_iterations,
            pcg_iterations,
            minimum_reachability_margin=reachability_margin,
            amplitude_diagnostics=amplitude,
            amplitude_diagnostics_source="rejected_candidate",
        )
    (
        p1_velocity_saturation_uncertainty_mps,
        p1_log_growth_saturation_uncertainty_per_step,
    ) = _p1_saturation_uncertainty(
        posterior_velocity_uncertainty_mps,
        posterior_log_growth_uncertainty_per_step,
        motion_saturation_margin_mps,
        growth_saturation_margin,
        frozen.nowcast_config,
    )
    field_smoothness_prior_cost = float(
        _field_smoothness_prior_cost(control, frozen).detach()
    )
    motion_saturation_margin = (
        frozen.motion_limits_yx
        - torch.abs(trajectory.displacement_yx)
    )
    analysis_verified_support = torch.zeros_like(source_support)
    analysis_motion_verified_support = torch.zeros_like(source_support)
    analysis_growth_verified_support = torch.zeros_like(source_support)
    analysis_dynamics_verified_support = torch.zeros_like(source_support)
    if not analysis_degraded:
        (
            analysis_verified_support,
            analysis_motion_verified_support,
            analysis_growth_verified_support,
            analysis_dynamics_verified_support,
        ) = _local_analysis_evidence_supports(
            trajectory,
            observations,
            source_support,
            frozen,
        )
    feasibility_margins = _analysis_feasibility_margins(
        frozen,
        trajectory,
        amplitude,
        reachability_margin=reachability_margin,
        motion_speed_saturation_margin_mps=(
            motion_speed_saturation_margin
        ),
        growth_saturation_margin=growth_saturation_margin,
    )
    retained_linearization = None
    p1_forecast_eligible = converged and not analysis_degraded
    posterior_eligible = p1_forecast_eligible
    fso_eligible = p1_forecast_eligible
    if fso_eligible:
        retained_observations = _clone_analysis_observations(observations)
        retained_frozen = _clone_frozen_outer_state(frozen)
        retained_linearization = _content_address_linearization(
            control,
            AnalysisLinearization(
                observations=retained_observations,
                frozen=retained_frozen,
                residual_norm=stationarity.residual_norm,
                gradient_norm=stationarity.gradient_norm,
                field_gradient_rms=stationarity.field_gradient_rms,
                field_gradient_max=stationarity.field_gradient_max,
                dynamics_gradient_max=stationarity.dynamics_gradient_max,
                relative_stationarity=stationarity.relative_stationarity,
                robust_gradient_norm=robust_stationarity.gradient_norm,
                robust_field_gradient_rms=(
                    robust_stationarity.field_gradient_rms
                ),
                robust_field_gradient_max=(
                    robust_stationarity.field_gradient_max
                ),
                robust_dynamics_gradient_max=(
                    robust_stationarity.dynamics_gradient_max
                ),
                robust_relative_stationarity=(
                    robust_stationarity.relative_stationarity
                ),
                irls_relative_weight_change=irls_relative_weight_change,
                polish_iterations=polish_iterations,
                feasibility_margins=feasibility_margins,
                algorithm_bundle_digest=algorithm_bundle_digest(),
                numerical_runtime_digest=numerical_runtime_identity_digest(
                    control.device
                ),
            ),
        )
    return AnalysisResult(
        control=_clone_tensor(control),
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
            observation_source_support=observation_source_support.detach(),
            background_source_support=background_source_support.detach(),
            path_verified_source_support=analysis_verified_support,
            verified_source_support=analysis_verified_support,
            local_motion_verified_support=analysis_motion_verified_support,
            local_growth_verified_support=analysis_growth_verified_support,
            local_dynamics_verified_support=analysis_dynamics_verified_support,
            observation_verified_source_support=analysis_verified_support,
            background_verified_source_support=torch.zeros_like(
                source_support
            ),
            provenance="p1_variational_analysis",
            dynamics_source=DynamicsSource.P1_VARIATIONAL,
            state_path_source=TendencySource.NONE,
            state_path_mode=TendencyPairSelection.NONE,
            state_path_pair_count=0,
            state_path_minimum_psr=math.nan,
            state_path_conflict=False,
            state_path_extrapolated=False,
            state_path_age_minutes=None,
            observation_path=StatePathProvenance(),
            background_path=StatePathProvenance(),
            minimum_growth_overlap_support=math.nan,
            minimum_growth_overlap_area_km2=math.nan,
            posterior_velocity_uncertainty_mps=(
                posterior_velocity_uncertainty_mps.detach()
            ),
            posterior_log_growth_uncertainty_per_step=(
                posterior_log_growth_uncertainty_per_step.detach()
            ),
            p1_velocity_saturation_uncertainty_mps=(
                p1_velocity_saturation_uncertainty_mps.detach()
            ),
            p1_log_growth_saturation_uncertainty_per_step=(
                p1_log_growth_saturation_uncertainty_per_step.detach()
            ),
        ),
        analyzed_frames_linear=frames.detach(),
        initial_objective=reference_objective,
        final_objective=final_objective,
        outer_iterations=outer_iterations,
        pcg_iterations=pcg_iterations,
        converged=converged,
        used_fallback=False,
        reason=reason,
        degraded=analysis_degraded,
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
        minimum_object_count_ratio_by_time=_materialize_pair(
            amplitude.minimum_object_count_ratio_by_time
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
        field_conditioned_dynamics_data_gram_eigenvalues=(
            None
            if identifiability is None
            else identifiability
            .field_conditioned_dynamics_data_gram_eigenvalues
        ),
        field_conditioned_dynamics_data_information_trace=(
            None
            if identifiability is None
            else identifiability
            .field_conditioned_dynamics_data_information_trace
        ),
        field_conditioned_dynamics_data_effective_dimension=(
            None
            if identifiability is None
            else identifiability
            .field_conditioned_dynamics_data_effective_dimension
        ),
        field_conditioning_maximum_relative_residual=(
            None
            if identifiability is None
            else identifiability.field_conditioning_maximum_relative_residual
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
        linearization_residual_norm=stationarity.residual_norm,
        linearization_gradient_norm=stationarity.gradient_norm,
        linearization_field_gradient_rms=stationarity.field_gradient_rms,
        linearization_field_gradient_max=stationarity.field_gradient_max,
        linearization_dynamics_gradient_max=(
            stationarity.dynamics_gradient_max
        ),
        linearization_relative_stationarity=(
            stationarity.relative_stationarity
        ),
        robust_gradient_norm=robust_stationarity.gradient_norm,
        robust_field_gradient_rms=robust_stationarity.field_gradient_rms,
        robust_field_gradient_max=robust_stationarity.field_gradient_max,
        robust_dynamics_gradient_max=(
            robust_stationarity.dynamics_gradient_max
        ),
        robust_relative_stationarity=(
            robust_stationarity.relative_stationarity
        ),
        irls_relative_weight_change=irls_relative_weight_change,
        linearization_polish_iterations=polish_iterations,
        linearization=retained_linearization,
        final_linearization_stationary=final_linearization_stationary,
        final_robust_stationary=final_robust_stationary,
        final_irls_fixed_point=final_irls_fixed_point,
        p1_forecast_eligible=p1_forecast_eligible,
        posterior_eligible=posterior_eligible,
        fso_eligible=fso_eligible,
        outer_converged=outer_converged,
    )


def _local_analysis_verified_support(
    trajectory: AnalysisTrajectory,
    observations: AnalysisObservations,
    source_support: Tensor,
    frozen: FrozenOuterState,
) -> Tensor:
    prediction_dbz = echo_to_dbz(
        trajectory.frames_linear[-1],
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    )
    absolute_error = torch.abs(observations.dbz[-1] - prediction_dbz)
    precision = _observation_marginal_precision(
        observations,
        frozen.analysis_config,
    )[-1]
    standardized_error = torch.sqrt(precision) * absolute_error
    detected_fit = observations.detected_mask[-1] & (
        standardized_error
        <= frozen.analysis_config.maximum_latest_detected_error_std
    ) & (
        absolute_error
        <= frozen.analysis_config.maximum_local_analysis_verification_error_dbz
    )
    censored_fit = observations.censored_mask[-1] & (
        prediction_dbz < frozen.analysis_config.detection_limit_dbz
    )
    has_information = (
        precision
        >= frozen.analysis_config.minimum_local_verification_precision
    )
    local_fit = (
        observations.valid_mask[-1]
        & has_information
        & (detected_fit | censored_fit)
    )
    return source_support.detach() * local_fit.to(dtype=source_support.dtype)


def _local_analysis_evidence_supports(
    trajectory: AnalysisTrajectory,
    observations: AnalysisObservations,
    source_support: Tensor,
    frozen: FrozenOuterState,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state_verified = _local_analysis_verified_support(
        trajectory,
        observations,
        source_support,
        frozen,
    )
    precision = _observation_marginal_precision(
        observations,
        frozen.analysis_config,
    )
    interval_detected = (
        observations.valid_mask
        & observations.detected_mask
        & (
            precision
            >= frozen.analysis_config.minimum_local_verification_precision
        )
    )
    observation_linear = dbz_to_echo(
        observations.dbz,
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    )
    motion_verified, growth_verified = (
        _local_component_evidence_from_pair_spans(
            observation_linear,
            interval_detected,
            trajectory.displacement_yx,
            trajectory.log_growth_per_step,
            ((0, 1), (1, 2)),
            ((0, 1), (1, 2)),
            trajectory.frames_linear[-1],
            state_verified,
            frozen.nowcast_config,
            frozen.grid_time_contract,
        )
    )
    dynamics_verified = torch.minimum(motion_verified, growth_verified)
    return (
        state_verified,
        motion_verified,
        growth_verified,
        dynamics_verified,
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
    minimum_object_count_ratios: list[Tensor] = []
    zero = prediction_dbz.new_zeros(())
    nan = prediction_dbz.new_full((), math.nan)
    effective_std = _observation_effective_std_dbz(
        observations,
        frozen.analysis_config,
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
        precursor_attribution_region = (
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
            minimum_object_count_ratios.append(nan)
            continue

        local_prediction = _footprint_maximum(
            prediction_dbz[step],
            frozen.amplitude_displacement_offsets_yx,
        )
        quality = observations.quality_weight[step]
        standardized_deficit = (
            torch.sqrt(quality)
            * (observations.dbz[step] - local_prediction)
            / effective_std[step]
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
            / effective_std[step],
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
            expanded_region &= precursor_attribution_region
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
            minimum_object_count_ratio,
        ) = _precursor_object_diagnostics(
            precursor_required,
            unresolved,
            quality,
            observations.dbz[step],
            prediction_dbz[step],
            trajectory.frames_linear[step],
            precursor_attribution_region,
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
        minimum_object_count_ratios.append(minimum_object_count_ratio)

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
        minimum_object_count_ratio_by_time=torch.stack(
            minimum_object_count_ratios
        ),
    )


def _precursor_object_diagnostics(
    precursor_required: Tensor,
    unresolved: Tensor,
    quality: Tensor,
    observed_dbz: Tensor,
    prediction_dbz: Tensor,
    prediction_echo: Tensor,
    prediction_attribution_region: Tensor,
    frozen: FrozenOuterState,
    *,
    enabled: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    zero = prediction_dbz.new_zeros(())
    nan = prediction_dbz.new_full((), math.nan)
    if not enabled:
        count = zero.to(dtype=torch.long)
        return count, count.clone(), zero, nan, nan, nan, nan, nan

    components = _connected_component_flat_indices(precursor_required)
    object_count = zero.new_tensor(len(components), dtype=torch.long)
    insufficient_count = zero.to(dtype=torch.long)
    unresolved_fractions: list[Tensor] = []
    eligible_components: list[Tensor] = []
    integrated_echo_ratios: list[Tensor] = []
    soft_echo_area_ratios: list[Tensor] = []
    object_count_ratios: list[Tensor] = []
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
        eligible_components.append(indices)

    groups = _overlapping_footprint_groups(
        eligible_components,
        precursor_required,
        prediction_dbz,
        frozen.amplitude_displacement_offsets_yx,
    )
    for indices, expanded, observed_object_count in groups:
        expanded &= prediction_attribution_region
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
        predicted_objects = _connected_component_flat_indices(
            expanded
            & (
                prediction_dbz
                >= frozen.analysis_config.detection_limit_dbz
            )
        )
        object_count_ratios.append(
            prediction_dbz.new_tensor(
                len(predicted_objects) / observed_object_count
            )
        )

    if not unresolved_fractions:
        return (
            object_count,
            insufficient_count,
            zero,
            nan,
            nan,
            nan,
            nan,
            nan,
        )
    unresolved_values = torch.stack(unresolved_fractions)
    echo_values = torch.stack(integrated_echo_ratios)
    area_values = torch.stack(soft_echo_area_ratios)
    count_values = torch.stack(object_count_ratios)
    return (
        object_count,
        insufficient_count,
        torch.max(unresolved_values),
        torch.min(echo_values),
        torch.max(echo_values),
        torch.min(area_values),
        torch.max(area_values),
        torch.min(count_values),
    )


def _overlapping_footprint_groups(
    components: list[Tensor],
    mask_template: Tensor,
    value_template: Tensor,
    offsets_yx: tuple[tuple[int, int], ...],
) -> tuple[tuple[Tensor, Tensor, int], ...]:
    groups: list[tuple[list[Tensor], Tensor]] = []
    for indices in components:
        object_mask = torch.zeros_like(mask_template)
        object_mask.flatten()[indices] = True
        expanded = _footprint_maximum(
            object_mask.to(dtype=value_template.dtype),
            offsets_yx,
        ) > 0
        overlaps = [
            index
            for index, (_, group_mask) in enumerate(groups)
            if bool(torch.any(expanded & group_mask))
        ]
        if not overlaps:
            groups.append(([indices], expanded))
            continue

        merged_indices = [indices]
        merged_mask = expanded
        for index in reversed(overlaps):
            group_indices, group_mask = groups.pop(index)
            merged_indices.extend(group_indices)
            merged_mask |= group_mask
        groups.append((merged_indices, merged_mask))

    return tuple(
        (torch.cat(indices), expanded, len(indices))
        for indices, expanded in groups
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
    effective_std = _observation_effective_std_dbz(
        observations,
        analysis,
    )
    initial_upper_dbz = (
        observations.dbz[0]
        + analysis.maximum_detected_error_std
        * effective_std[0]
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
            / effective_std[step]
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
        metadata=replace(
            frozen.baseline_metadata,
            dynamics_source=DynamicsSource.P0_FALLBACK,
        ),
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
        minimum_object_count_ratio_by_time=(
            None
            if amplitude_diagnostics is None
            else _materialize_pair(
                amplitude_diagnostics.minimum_object_count_ratio_by_time
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


def _motion_saturation_margin_mps(
    displacement_yx: Tensor,
    frozen: FrozenOuterState,
) -> float | None:
    speed_margin = _motion_speed_saturation_margin(displacement_yx, frozen)
    if speed_margin is not None:
        return speed_margin
    contract = frozen.grid_time_contract
    if contract is None:
        return None
    margin_yx = frozen.motion_limits_yx - torch.abs(displacement_yx)
    zero = margin_yx.new_zeros(())
    axis_margins = torch.stack(
        (
            torch.stack((margin_yx[0], zero)),
            torch.stack((zero, margin_yx[1])),
        )
    )
    projected = torch.stack(
        tuple(
            contract.projected_displacement_xy(value)
            for value in axis_margins
        )
    )
    seconds_per_step = frozen.nowcast_config.interval_minutes * 60.0
    minimum_margin = torch.min(
        torch.linalg.vector_norm(projected, dim=1)
    )
    return float(minimum_margin.detach()) / seconds_per_step


def _motion_saturation_margin_fraction(
    displacement_yx: Tensor,
    frozen: FrozenOuterState,
) -> float:
    limits = frozen.motion_limits_yx
    active = limits > torch.finfo(limits.dtype).eps
    if not bool(torch.any(active)):
        return 1.0
    margin = (limits[active] - torch.abs(displacement_yx[active])) / limits[
        active
    ]
    return float(torch.min(margin).detach())


def _analysis_feasibility_margins(
    frozen: FrozenOuterState,
    trajectory: AnalysisTrajectory,
    amplitude: _AmplitudeDiagnostics,
    *,
    reachability_margin: float,
    motion_speed_saturation_margin_mps: float | None,
    growth_saturation_margin: Tensor,
) -> AnalysisFeasibilityMargins:
    """Materialize signed interior margins used by delayed P1 FSO."""

    unresolved_margin = (
        frozen.analysis_config.maximum_unresolved_amplitude_fraction
        - float(amplitude.maximum_gated_unresolved_fraction.detach())
    )
    return AnalysisFeasibilityMargins(
        reachability_support=reachability_margin,
        unresolved_amplitude_fraction=unresolved_margin,
        amplitude_confidence=_amplitude_confidence_margin(
            amplitude,
            frozen.analysis_config,
        ),
        motion_saturation_fraction=_motion_saturation_margin_fraction(
            trajectory.displacement_yx,
            frozen,
        ),
        motion_speed_saturation_mps=motion_speed_saturation_margin_mps,
        growth_saturation_per_step=float(growth_saturation_margin.detach()),
    )


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
            freeze_remap_cell(step * displacement) for step in (1, 2)
        ),
    )


def _analysis_remap_cells_match(
    control: Tensor,
    frozen: FrozenOuterState,
) -> bool:
    """Return whether ``control`` stays on the retained remap branch."""

    field_size = frozen.active_field_index.numel()
    displacement, _ = _decode_dynamics(
        control[field_size:],
        frozen.baseline_state,
        frozen.analysis_config,
        frozen.nowcast_config,
        frozen.motion_limits_yx,
        frozen.grid_time_contract,
    )
    cells = tuple(
        freeze_remap_cell(step * displacement) for step in (1, 2)
    )
    return cells == frozen.analysis_remap_cells


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
    group_index = observations.common_bias_group_index
    if group_index is not None and (
        group_index.shape != shape
        or group_index.dtype != torch.long
        or group_index.device != observations.dbz.device
        or bool(torch.any(group_index < -1))
    ):
        raise ValueError(
            "common_bias_group_index must be a compatible canonical index"
        )
    if group_index is not None:
        active_groups = group_index[group_index >= 0]
        if active_groups.numel() > 0 and not torch.equal(
            torch.unique(active_groups, sorted=True),
            torch.arange(
                int(torch.max(active_groups).detach()) + 1,
                dtype=torch.long,
                device=group_index.device,
            ),
        ):
            raise ValueError("common_bias_group_index must be compact")
    mode_weights = observations.common_bias_mode_weights
    if mode_weights is not None:
        by_frame = _common_bias_mode_weights_by_frame(mode_weights)
        shape_valid = (
            mode_weights.ndim == 3
            or (mode_weights.ndim == 4 and mode_weights.shape[0] == shape[0])
        )
        if (
            not shape_valid
            or tuple(mode_weights.shape[-2:]) != shape[-2:]
            or not mode_weights.is_floating_point()
            or mode_weights.dtype != observations.dbz.dtype
            or mode_weights.device != observations.dbz.device
            or not (
                1
                <= mode_weights.shape[-3]
                <= MAXIMUM_OBSERVATION_COMMON_BIAS_MODE_COUNT
            )
            or not bool(torch.all(torch.isfinite(mode_weights)))
            or bool(torch.any(mode_weights < 0.0))
            or bool(torch.any(mode_weights > 1.0))
            or bool(
                torch.any(
                    torch.sum(by_frame.square(), dim=1) > 1.0 + 1.0e-6
                )
            )
            or bool(torch.any(torch.sum(by_frame, dim=(0, 2, 3)) <= 0.0))
        ):
            raise ValueError(
                "common_bias_mode_weights must be canonical and compatible"
            )


def _validate_observation_common_bias_contract(
    observations: AnalysisObservations,
    config: AnalysisConfig,
) -> None:
    group_index = observations.common_bias_group_index
    mode_weights = observations.common_bias_mode_weights
    expected_digest = config.observation_common_bias_group_map_digest
    expected_mode_digest = config.observation_common_bias_mode_weights_digest
    if group_index is not None and mode_weights is not None:
        raise ValueError(
            "common-bias group observations and overlapping modes are "
            "mutually exclusive"
        )
    if mode_weights is not None:
        if config.observation_common_bias_std_dbz <= 0.0:
            raise ValueError(
                "common-bias mode observations require positive common-bias "
                "std"
            )
        if config.observation_common_bias_tile_size_px > 0:
            raise ValueError(
                "common-bias mode observations and tile modes are mutually "
                "exclusive"
            )
        if expected_digest is not None:
            raise ValueError(
                "common-bias mode observations cannot use a group-map digest"
            )
        if expected_mode_digest is None:
            raise ValueError("common-bias mode observations require a digest")
        if (
            observation_common_bias_mode_weights_digest(mode_weights)
            != expected_mode_digest
        ):
            raise ValueError("common-bias mode observation digest mismatch")
        return
    if expected_mode_digest is not None:
        raise ValueError(
            "common-bias mode weights digest requires mode observations"
        )
    if group_index is None:
        if expected_digest is not None:
            raise ValueError(
                "common-bias group map digest requires group observations"
            )
        return
    if config.observation_common_bias_std_dbz <= 0.0:
        raise ValueError(
            "common-bias group observations require positive common-bias std"
        )
    if config.observation_common_bias_tile_size_px > 0:
        raise ValueError(
            "common-bias group observations and tile modes are mutually "
            "exclusive"
        )
    if expected_digest is None:
        raise ValueError("common-bias group observations require a digest")
    if tensor_digest(group_index) != expected_digest:
        raise ValueError("common-bias group observation digest mismatch")


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
    prior_values = (
        frozen.neural_prior_std_dbz,
        frozen.neural_prior_valid_mask,
        frozen.neural_prior_dependency,
        frozen.neural_prior_application_digest,
        frozen.neural_prior_raw_background_dbz,
        frozen.neural_prior_execution_contract_digest,
        frozen.neural_prior_role,
    )
    if frozen.observation_derived_initial_background:
        if any(value is not None for value in prior_values):
            raise ValueError("observation-derived state cannot retain a prior")
        return
    (
        std,
        valid,
        dependency,
        application_digest,
        raw_background,
        execution_digest,
        role,
    ) = prior_values
    if (
        not isinstance(std, Tensor)
        or std.shape != frozen.initial_background_dbz.shape
        or std.dtype != frozen.initial_background_dbz.dtype
        or std.device != frozen.initial_background_dbz.device
        or not bool(torch.all(torch.isfinite(std) & (std > 0.0)))
        or not isinstance(valid, Tensor)
        or valid.shape != frozen.initial_background_dbz.shape
        or valid.dtype != torch.bool
        or valid.device != frozen.initial_background_dbz.device
        or dependency not in ("exogenous", "radar_dependent")
        or not isinstance(application_digest, str)
        or not isinstance(raw_background, Tensor)
        or raw_background.shape != frozen.initial_background_dbz.shape
        or raw_background.dtype != frozen.initial_background_dbz.dtype
        or raw_background.device != frozen.initial_background_dbz.device
        or not bool(torch.all(torch.isfinite(raw_background)))
        or not isinstance(execution_digest, str)
        or role not in ("candidate", "parent")
    ):
        raise ValueError("retained neural-prior state is incomplete")
    _require_prior_digest("neural_prior_application_digest", application_digest)
    _require_prior_digest("neural_prior_execution_contract_digest", execution_digest)


def _clone_tensor(value: Tensor) -> Tensor:
    return value.detach().clone()


def _retained_tensor_bytes(value: object) -> int:
    """Count Tensor payload retained by one nested result contract."""

    seen: set[int] = set()

    def count(item: object) -> int:
        if isinstance(item, Tensor):
            identity = id(item)
            if identity in seen:
                return 0
            seen.add(identity)
            return item.numel() * item.element_size()
        if is_dataclass(item) and not isinstance(item, type):
            return sum(count(getattr(item, field.name)) for field in fields(item))
        if isinstance(item, dict):
            return sum(count(entry) for entry in item.values())
        if isinstance(item, (tuple, list)):
            return sum(count(entry) for entry in item)
        return 0

    return count(value)


def _clone_optional_tensor(value: Tensor | None) -> Tensor | None:
    return None if value is None else _clone_tensor(value)


def _clone_analysis_observations(
    observations: AnalysisObservations,
) -> AnalysisObservations:
    return AnalysisObservations(
        dbz=_clone_tensor(observations.dbz),
        std_dbz=_clone_tensor(observations.std_dbz),
        quality_weight=_clone_tensor(observations.quality_weight),
        valid_mask=_clone_tensor(observations.valid_mask),
        detected_mask=_clone_tensor(observations.detected_mask),
        censored_mask=_clone_tensor(observations.censored_mask),
        missing_mask=_clone_tensor(observations.missing_mask),
        qc_rejected_mask=_clone_tensor(observations.qc_rejected_mask),
        common_bias_group_index=(
            None
            if observations.common_bias_group_index is None
            else _clone_tensor(observations.common_bias_group_index)
        ),
        common_bias_mode_weights=(
            None
            if observations.common_bias_mode_weights is None
            else _clone_tensor(observations.common_bias_mode_weights)
        ),
    )


def _detach_state(state: RadarState) -> RadarState:
    return RadarState(
        echo_linear=_clone_tensor(state.echo_linear),
        displacement_yx=_clone_tensor(state.displacement_yx),
        log_growth_per_step=_clone_tensor(state.log_growth_per_step),
    )


def _detach_metadata(metadata: ForecastMetadata) -> ForecastMetadata:
    return ForecastMetadata(
        data_status=metadata.data_status,
        coverage_by_frame=_clone_tensor(metadata.coverage_by_frame),
        background_used=metadata.background_used,
        background_contribution_fraction=(
            metadata.background_contribution_fraction
        ),
        background_age_minutes=metadata.background_age_minutes,
        source_support=_clone_tensor(metadata.source_support),
        observation_source_support=(
            _clone_tensor(metadata.observation_source_support)
        ),
        background_source_support=_clone_tensor(
            metadata.background_source_support
        ),
        path_verified_source_support=(
            _clone_tensor(metadata.path_verified_source_support)
        ),
        verified_source_support=_clone_tensor(
            metadata.verified_source_support
        ),
        local_motion_verified_support=(
            _clone_tensor(metadata.local_motion_verified_support)
        ),
        local_growth_verified_support=(
            _clone_tensor(metadata.local_growth_verified_support)
        ),
        local_dynamics_verified_support=(
            _clone_tensor(metadata.local_dynamics_verified_support)
        ),
        observation_verified_source_support=(
            _clone_tensor(metadata.observation_verified_source_support)
        ),
        background_verified_source_support=(
            _clone_tensor(metadata.background_verified_source_support)
        ),
        motion_disagreement_px=_clone_tensor(
            metadata.motion_disagreement_px
        ),
        motion_disagreement_mps=_clone_tensor(
            metadata.motion_disagreement_mps
        ),
        growth_disagreement=_clone_tensor(metadata.growth_disagreement),
        maximum_growth_saturation_excess=(
            _clone_tensor(metadata.maximum_growth_saturation_excess)
        ),
        posterior_velocity_uncertainty_mps=(
            _clone_tensor(metadata.posterior_velocity_uncertainty_mps)
        ),
        posterior_log_growth_uncertainty_per_step=(
            _clone_tensor(
                metadata.posterior_log_growth_uncertainty_per_step
            )
        ),
        p1_velocity_saturation_uncertainty_mps=(
            _clone_tensor(
                metadata.p1_velocity_saturation_uncertainty_mps
            )
        ),
        p1_log_growth_saturation_uncertainty_per_step=(
            _clone_tensor(
                metadata.p1_log_growth_saturation_uncertainty_per_step
            )
        ),
        minimum_phase_correlation_psr=(
            _clone_tensor(metadata.minimum_phase_correlation_psr)
        ),
        tendency_pair_count=metadata.tendency_pair_count,
        tendency_source=metadata.tendency_source,
        provenance=metadata.provenance,
        dynamics_source=metadata.dynamics_source,
        motion_pair_count=metadata.motion_pair_count,
        growth_pair_count=metadata.growth_pair_count,
        motion_pair_selection=metadata.motion_pair_selection,
        growth_pair_selection=metadata.growth_pair_selection,
        motion_pair_conflict=metadata.motion_pair_conflict,
        growth_pair_conflict=metadata.growth_pair_conflict,
        state_path_source=metadata.state_path_source,
        state_path_mode=metadata.state_path_mode,
        state_path_pair_count=metadata.state_path_pair_count,
        state_path_minimum_psr=metadata.state_path_minimum_psr,
        state_path_conflict=metadata.state_path_conflict,
        state_path_extrapolated=metadata.state_path_extrapolated,
        state_path_age_minutes=metadata.state_path_age_minutes,
        observation_path=metadata.observation_path,
        background_path=metadata.background_path,
        minimum_growth_overlap_support=(
            metadata.minimum_growth_overlap_support
        ),
        minimum_growth_overlap_area_km2=(
            metadata.minimum_growth_overlap_area_km2
        ),
    )


def _clone_frozen_outer_state(frozen: FrozenOuterState) -> FrozenOuterState:
    """Detach the retained adjoint model from all caller-owned storage."""

    return FrozenOuterState(
        input_frames_dbz=_clone_tensor(frozen.input_frames_dbz),
        background_frames_dbz=_clone_optional_tensor(
            frozen.background_frames_dbz
        ),
        initial_background_dbz=_clone_tensor(frozen.initial_background_dbz),
        initial_support_mask=_clone_tensor(frozen.initial_support_mask),
        active_field_index=_clone_tensor(frozen.active_field_index),
        causal_only_mask=_clone_tensor(frozen.causal_only_mask),
        causal_seed_mask=_clone_tensor(frozen.causal_seed_mask),
        detected_masks=_clone_tensor(frozen.detected_masks),
        observed_mask=_clone_tensor(frozen.observed_mask),
        background_mask=_clone_tensor(frozen.background_mask),
        background_age_minutes=frozen.background_age_minutes,
        baseline_state=_detach_state(frozen.baseline_state),
        baseline_metadata=_detach_metadata(frozen.baseline_metadata),
        baseline_frames_dbz=_clone_tensor(frozen.baseline_frames_dbz),
        observation_whitener=FrozenObservationWhitener(
            mode=_clone_optional_tensor(frozen.observation_whitener.mode),
            overlapping_correction=_clone_optional_tensor(
                frozen.observation_whitener.overlapping_correction
            ),
            per_frame=frozen.observation_whitener.per_frame,
        ),
        irls_sqrt_weight=_clone_tensor(frozen.irls_sqrt_weight),
        nowcast_config=frozen.nowcast_config,
        analysis_config=frozen.analysis_config,
        grid_time_contract=frozen.grid_time_contract,
        motion_limits_yx=_clone_tensor(frozen.motion_limits_yx),
        amplitude_displacement_offsets_yx=(
            frozen.amplitude_displacement_offsets_yx
        ),
        analysis_remap_cells=frozen.analysis_remap_cells,
        smooth_edge_left_index=_clone_tensor(
            frozen.smooth_edge_left_index
        ),
        smooth_edge_right_index=_clone_tensor(
            frozen.smooth_edge_right_index
        ),
        smooth_edge_physical_weight=_clone_tensor(
            frozen.smooth_edge_physical_weight
        ),
        observation_derived_initial_background=(
            frozen.observation_derived_initial_background
        ),
        neural_prior_std_dbz=_clone_optional_tensor(frozen.neural_prior_std_dbz),
        neural_prior_valid_mask=_clone_optional_tensor(frozen.neural_prior_valid_mask),
        neural_prior_dependency=frozen.neural_prior_dependency,
        neural_prior_application_digest=(frozen.neural_prior_application_digest),
        neural_prior_raw_background_dbz=_clone_optional_tensor(
            frozen.neural_prior_raw_background_dbz
        ),
        neural_prior_execution_contract_digest=(
            frozen.neural_prior_execution_contract_digest
        ),
        neural_prior_role=frozen.neural_prior_role,
    )


def _analysis_observations_digest_values(
    observations: AnalysisObservations,
) -> dict[str, str | None]:
    values: dict[str, str | None] = {
        name: tensor_digest(getattr(observations, name))
        for name in (
            "dbz",
            "std_dbz",
            "quality_weight",
            "valid_mask",
            "detected_mask",
            "censored_mask",
            "missing_mask",
            "qc_rejected_mask",
        )
    }
    values["common_bias_group_index"] = (
        None
        if observations.common_bias_group_index is None
        else tensor_digest(observations.common_bias_group_index)
    )
    values["common_bias_mode_weights"] = (
        None
        if observations.common_bias_mode_weights is None
        else tensor_digest(observations.common_bias_mode_weights)
    )
    return values


def _optional_tensor_digest(value: Tensor | None) -> str | None:
    return None if value is None else tensor_digest(value)


def _frozen_outer_state_digest_values(
    frozen: FrozenOuterState,
) -> dict[str, object]:
    grid_digest = (
        None
        if frozen.grid_time_contract is None
        else json_digest(asdict(frozen.grid_time_contract))
    )
    values: dict[str, object] = {
        "input_frames_dbz": tensor_digest(frozen.input_frames_dbz),
        "background_frames_dbz": _optional_tensor_digest(
            frozen.background_frames_dbz
        ),
        "initial_background_dbz": tensor_digest(
            frozen.initial_background_dbz
        ),
        "initial_support_mask": tensor_digest(frozen.initial_support_mask),
        "active_field_index": tensor_digest(frozen.active_field_index),
        "causal_only_mask": tensor_digest(frozen.causal_only_mask),
        "causal_seed_mask": tensor_digest(frozen.causal_seed_mask),
        "detected_masks": tensor_digest(frozen.detected_masks),
        "observed_mask": tensor_digest(frozen.observed_mask),
        "background_mask": tensor_digest(frozen.background_mask),
        "background_age_minutes": frozen.background_age_minutes,
        "baseline_state_metadata": state_metadata_digest(
            frozen.baseline_state,
            frozen.baseline_metadata,
        ),
        "baseline_frames_dbz": tensor_digest(frozen.baseline_frames_dbz),
        "observation_whitener": {
            "mode": _optional_tensor_digest(
                frozen.observation_whitener.mode
            ),
            "overlapping_correction": _optional_tensor_digest(
                frozen.observation_whitener.overlapping_correction
            ),
            "per_frame": frozen.observation_whitener.per_frame,
        },
        "irls_sqrt_weight": tensor_digest(frozen.irls_sqrt_weight),
        "nowcast_config": frozen.nowcast_config.digest,
        "analysis_config": json_digest(asdict(frozen.analysis_config)),
        "grid_time_contract": grid_digest,
        "motion_limits_yx": tensor_digest(frozen.motion_limits_yx),
        "amplitude_displacement_offsets_yx": [
            list(offset)
            for offset in frozen.amplitude_displacement_offsets_yx
        ],
        "analysis_remap_cells": [
            {"y": cell.y, "x": cell.x}
            for cell in frozen.analysis_remap_cells
        ],
        "smooth_edge_left_index": tensor_digest(
            frozen.smooth_edge_left_index
        ),
        "smooth_edge_right_index": tensor_digest(
            frozen.smooth_edge_right_index
        ),
        "smooth_edge_physical_weight": tensor_digest(
            frozen.smooth_edge_physical_weight
        ),
    }
    if not frozen.observation_derived_initial_background:
        values["observation_derived_initial_background"] = False
        values["neural_prior_std_dbz"] = _optional_tensor_digest(
            frozen.neural_prior_std_dbz
        )
        values["neural_prior_valid_mask"] = _optional_tensor_digest(
            frozen.neural_prior_valid_mask
        )
        values["neural_prior_dependency"] = frozen.neural_prior_dependency
        values["neural_prior_application_digest"] = (
            frozen.neural_prior_application_digest
        )
        values["neural_prior_raw_background_dbz"] = _optional_tensor_digest(
            frozen.neural_prior_raw_background_dbz
        )
        values["neural_prior_execution_contract_digest"] = (
            frozen.neural_prior_execution_contract_digest
        )
        values["neural_prior_role"] = frozen.neural_prior_role
    return values


def analysis_linearization_digest(
    control: Tensor,
    linearization: AnalysisLinearization,
) -> str:
    """Content-address every value used by the delayed P1 adjoint."""

    return json_digest(
        {
            "version": P1_LINEARIZATION_DIGEST_CONTRACT,
            "contract": linearization.contract,
            "forecast_run_digest": linearization.forecast_run_digest,
            "control": tensor_digest(control),
            "observations": _analysis_observations_digest_values(
                linearization.observations
            ),
            "frozen": _frozen_outer_state_digest_values(
                linearization.frozen
            ),
            "stationarity": {
                "residual_norm": linearization.residual_norm,
                "gradient_norm": linearization.gradient_norm,
                "field_gradient_rms": linearization.field_gradient_rms,
                "field_gradient_max": linearization.field_gradient_max,
                "dynamics_gradient_max": (
                    linearization.dynamics_gradient_max
                ),
                "relative_stationarity": (
                    linearization.relative_stationarity
                ),
                "robust_gradient_norm": (
                    linearization.robust_gradient_norm
                ),
                "robust_field_gradient_rms": (
                    linearization.robust_field_gradient_rms
                ),
                "robust_field_gradient_max": (
                    linearization.robust_field_gradient_max
                ),
                "robust_dynamics_gradient_max": (
                    linearization.robust_dynamics_gradient_max
                ),
                "robust_relative_stationarity": (
                    linearization.robust_relative_stationarity
                ),
                "irls_relative_weight_change": (
                    linearization.irls_relative_weight_change
                ),
                "polish_iterations": linearization.polish_iterations,
            },
            "feasibility_margins": asdict(
                linearization.feasibility_margins
            ),
            "algorithm_bundle_digest": (
                linearization.algorithm_bundle_digest
            ),
            "numerical_runtime_digest": (
                linearization.numerical_runtime_digest
            ),
        }
    )


def _content_address_linearization(
    control: Tensor,
    linearization: AnalysisLinearization,
) -> AnalysisLinearization:
    addressed = replace(
        linearization,
        control_digest=tensor_digest(control),
        linearization_digest="",
    )
    return replace(
        addressed,
        linearization_digest=analysis_linearization_digest(
            control,
            addressed,
        ),
    )


def validate_analysis_linearization_content(
    control: Tensor,
    linearization: AnalysisLinearization,
    *,
    require_current_environment: bool = True,
) -> None:
    """Validate immutable contents and, optionally, replay environment."""

    if linearization.contract != P1_LINEARIZATION_CONTRACT:
        raise ValueError("unsupported P1 linearization contract")
    feasibility_values = (
        linearization.feasibility_margins.reachability_support,
        linearization.feasibility_margins.unresolved_amplitude_fraction,
        linearization.feasibility_margins.motion_saturation_fraction,
        linearization.feasibility_margins.growth_saturation_per_step,
    )
    optional_feasibility_values = (
        linearization.feasibility_margins.amplitude_confidence,
        linearization.feasibility_margins.motion_speed_saturation_mps,
    )
    if not all(math.isfinite(value) for value in feasibility_values) or any(
        value is not None and not math.isfinite(value)
        for value in optional_feasibility_values
    ):
        raise ValueError("P1 linearization feasibility margins must be finite")
    _validate_observations(linearization.observations)
    retained_input = linearization.frozen.input_frames_dbz
    if (
        retained_input.shape != linearization.observations.dbz.shape
        or retained_input.dtype != linearization.observations.dbz.dtype
        or retained_input.device != linearization.observations.dbz.device
    ):
        raise ValueError("retained P1 input frames are incompatible")
    retained_background = linearization.frozen.background_frames_dbz
    if retained_background is not None and (
        retained_background.shape != retained_input.shape
        or retained_background.dtype != retained_input.dtype
        or retained_background.device != retained_input.device
    ):
        raise ValueError("retained P1 background frames are incompatible")
    _validate_observation_common_bias_contract(
        linearization.observations,
        linearization.frozen.analysis_config,
    )
    _validate_frozen_observation_whitener(
        linearization.observations,
        linearization.frozen,
    )
    if tensor_digest(control) != linearization.control_digest:
        raise ValueError("P1 linearization control digest mismatch")
    if (
        analysis_linearization_digest(control, linearization)
        != linearization.linearization_digest
    ):
        raise ValueError("P1 linearization content digest mismatch")
    if require_current_environment:
        if linearization.algorithm_bundle_digest != algorithm_bundle_digest():
            raise ValueError("P1 linearization algorithm bundle mismatch")
        if (
            linearization.numerical_runtime_digest
            != numerical_runtime_identity_digest(control.device)
        ):
            raise ValueError("P1 linearization numerical runtime mismatch")
