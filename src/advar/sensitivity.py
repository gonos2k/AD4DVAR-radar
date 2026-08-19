"""Forecast sensitivities that can be stored as conditional experience."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import IntEnum
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ._digest import dataclass_digest, json_digest, tensor_digest
from .calibration import OperationalCalibrationManifest, OperationalDataIdentity
from .matrix_free import pcg
from .nowcast import (
    DataStatus,
    DynamicsSource,
    ForecastMetadata,
    ForecastResult,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    TendencyPairSelection,
    TendencySource,
    _estimate_available_pair,
    _estimate_source_tendencies,
    _forecast_linear_at_step_core,
    _phase_correlation_details,
    forecast_from_state,
    forecast_linear_at_step,
    motion_displacement_limits_yx,
)
from .physics import RemapCell, dbz_to_echo, echo_to_dbz, freeze_remap_cell
from .variational import (
    P1_LINEARIZATION_CONTRACT,
    AnalysisFeasibilityMargins,
    AnalysisLinearization,
    AnalysisObservations,
    AnalysisResult,
    BoundNeuralPriorInput,
    FrozenOuterState,
    NeuralPriorApplication,
    NeuralPriorInferenceRunner,
    P1LinearizationState,
    _apply_observation_error_whitener,
    _analysis_trajectory,
    _analysis_input_lineage,
    _count_observation_whitener_applies,
    _linearization_stationarity,
    _relative_irls_weight_change,
    _observation_whitener_operations_per_apply,
    _robust_stationarity,
    _stationarity_is_acceptable,
    freeze_irls_weights,
    prepare_analysis,
    residual_vector,
    solve_analysis,
    validate_analysis_linearization_content,
    variational_nowcast,
)


SUPPORTED_METRICS = (
    "log_echo_mse",
    "soft_fss_error_35",
    "centroid_error",
    "centroid_error_m2",
)
DEFAULT_METRICS = (
    "log_echo_mse",
    "soft_fss_error_35",
    "centroid_error",
)

FSOMetricDomain = Literal[
    "issued",
    "radar_dynamics_anchored",
    "confidence_weighted",
]
PerturbationSemantics = Literal[
    "augmented_parameter",
    "physical_radar_value",
]
BaselineDynamicsBranchStatus = Literal[
    "not_applicable",
    "unknown",
    "certified",
    "invalid",
]
CandidateRankingObjective = Literal[
    "absolute_influence",
    "expected_error_reduction",
    "two_sided_diagnostic",
]
LearningSelectionMode = Literal["direct", "ranked_top_k"]
FirstOrderMetricDomain = Literal[
    "frozen_metric_domain",
    "resolved_issuance_domain",
]
TileShape = tuple[int, int]
LEARNING_POLICY_TRUST_STORE_CONTRACT = "advar-learning-policy-trust-store-v1"
MAXIMUM_LEARNING_POLICY_TRUST_STORE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _LearningPolicyTrustStore:
    approved_policy_digests: frozenset[str]
    content_digest: str


def _canonical_verification_time(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("verification valid times must be ISO-8601 strings")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            "verification valid times must be ISO-8601 strings"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("verification valid times must include timezones")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _load_learning_policy_trust_store(
    path: str | Path,
) -> _LearningPolicyTrustStore:
    """Read approved policy digests from a root-owned immutable JSON file."""

    trust_store = Path(path)
    if not trust_store.is_absolute():
        raise ValueError("learning policy trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(trust_store, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("learning policy trust store must be a file")
        if metadata.st_uid != 0 or metadata.st_mode & 0o022:
            raise ValueError(
                "learning policy trust store must be root-owned and not "
                "group/world-writable"
            )
        if metadata.st_size > MAXIMUM_LEARNING_POLICY_TRUST_STORE_BYTES:
            raise ValueError("learning policy trust store is too large")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            content = stream.read(MAXIMUM_LEARNING_POLICY_TRUST_STORE_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(content) > MAXIMUM_LEARNING_POLICY_TRUST_STORE_BYTES:
        raise ValueError("learning policy trust store is too large")
    document = json.loads(content)
    if not isinstance(document, dict) or set(document) != {
        "contract",
        "approved_policy_digests",
    }:
        raise ValueError("invalid learning policy trust store")
    if document["contract"] != LEARNING_POLICY_TRUST_STORE_CONTRACT:
        raise ValueError("unsupported learning policy trust store")
    raw_digests = document["approved_policy_digests"]
    if not isinstance(raw_digests, list) or any(
        not isinstance(digest, str) for digest in raw_digests
    ):
        raise ValueError("approved policy digests must be a list")
    digests = frozenset(raw_digests)
    if len(digests) != len(raw_digests):
        raise ValueError("approved policy digests must be unique")
    for digest in digests:
        _require_sha256("approved_policy_digest", digest)
    return _LearningPolicyTrustStore(
        approved_policy_digests=digests,
        content_digest=json_digest(
            {
                "contract": LEARNING_POLICY_TRUST_STORE_CONTRACT,
                "approved_policy_digests": sorted(digests),
            }
        ),
    )


@dataclass(frozen=True)
class VerificationObservationErrorPlan:
    """Pre-registered policy for deriving verification observation errors."""

    radar_source_kind: Literal["single_site", "mosaic"]
    source_registry_digest: str
    calibration_registry_digest: str
    range_elevation_validity_algorithm_digest: str
    beam_blockage_algorithm_digest: str
    attenuation_qc_digest: str
    censoring_rule_digest: str
    spatial_correlation_block_algorithm_digest: str
    quality_weight_interpretation_digest: str
    quality_weight_algorithm_digest: str
    observation_std_algorithm_digest: str
    observation_error_model_digest: str
    source_assignment_algorithm_digest: str
    minimum_detectable_echo_dbz: float
    observation_error_reference_std_dbz: float
    spatial_correlation_role: Literal["diagnostic_only"] = "diagnostic_only"
    contract: str = "verification-observation-error-plan-v1"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "verification-observation-error-plan-v1"
            or self.radar_source_kind not in {"single_site", "mosaic"}
            or not math.isfinite(self.minimum_detectable_echo_dbz)
            or not math.isfinite(self.observation_error_reference_std_dbz)
            or self.observation_error_reference_std_dbz <= 0.0
            or self.spatial_correlation_role != "diagnostic_only"
        ):
            raise ValueError("verification observation-error plan is invalid")
        for name in (
            "source_registry_digest",
            "calibration_registry_digest",
            "range_elevation_validity_algorithm_digest",
            "beam_blockage_algorithm_digest",
            "attenuation_qc_digest",
            "censoring_rule_digest",
            "spatial_correlation_block_algorithm_digest",
            "quality_weight_interpretation_digest",
            "quality_weight_algorithm_digest",
            "observation_std_algorithm_digest",
            "observation_error_model_digest",
            "source_assignment_algorithm_digest",
        ):
            _require_sha256(name, getattr(self, name))
        object.__setattr__(self, "plan_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "plan_digest"
        }


class VerificationCellState(IntEnum):
    """Per-cell observation semantics retained for scientific verification."""

    OBSERVED_CLEAR = 0
    OBSERVED_ECHO = 1
    SOURCE_MISSING = 2
    QC_INVALID = 3
    BEAM_BLOCKED = 4
    BELOW_DETECTION_CENSORED = 5
    MOSAIC_SOURCE_UNASSIGNED = 6


def _validate_verification_cell_states(
    *,
    frames_dbz: Tensor,
    valid_mask: Tensor,
    quality_weight: Tensor,
    observation_std_dbz: Tensor,
    observation_state_code: Tensor,
    minimum_detectable_echo_dbz: float,
    radar_source_kind: Literal["single_site", "mosaic"],
    source_radar_index_map: Tensor | None,
) -> None:
    if (
        observation_state_code.dtype is not torch.uint8
        or observation_state_code.shape != frames_dbz.shape
        or observation_state_code.device != frames_dbz.device
    ):
        raise ValueError("verification observation-state tensor is invalid")
    observed_clear = observation_state_code == VerificationCellState.OBSERVED_CLEAR
    observed_echo = observation_state_code == VerificationCellState.OBSERVED_ECHO
    censored = (
        observation_state_code
        == VerificationCellState.BELOW_DETECTION_CENSORED
    )
    recognized = torch.zeros_like(valid_mask)
    for state in VerificationCellState:
        recognized |= observation_state_code == state
    expected_valid = observed_clear | observed_echo | censored
    invalid = ~expected_valid
    mosaic_unassigned = (
        observation_state_code
        == VerificationCellState.MOSAIC_SOURCE_UNASSIGNED
    )
    if radar_source_kind == "single_site":
        source_semantics_valid = (
            source_radar_index_map is None
            and not bool(torch.any(mosaic_unassigned))
        )
    else:
        source_semantics_valid = (
            source_radar_index_map is not None
            and source_radar_index_map.dtype is torch.int64
            and source_radar_index_map.shape == frames_dbz.shape
            and source_radar_index_map.device == frames_dbz.device
            and bool(torch.all(source_radar_index_map >= -1))
            and bool(
                torch.equal(
                    source_radar_index_map == -1,
                    mosaic_unassigned,
                )
            )
            and bool(
                torch.all(
                    source_radar_index_map.masked_select(expected_valid) >= 0
                )
            )
        )
    if (
        not source_semantics_valid
        or not bool(torch.all(recognized))
        or not bool(torch.equal(valid_mask, expected_valid))
        or bool(torch.any(observed_clear & (frames_dbz >= minimum_detectable_echo_dbz)))
        or bool(torch.any(observed_echo & (frames_dbz < minimum_detectable_echo_dbz)))
        or bool(torch.any(censored & (frames_dbz > minimum_detectable_echo_dbz)))
        or not bool(torch.all(quality_weight.masked_select(invalid) == 0.0))
        or not bool(torch.all(observation_std_dbz.masked_select(invalid) == 0.0))
        or not bool(torch.all(observation_std_dbz.masked_select(expected_valid) > 0.0))
    ):
        raise ValueError("verification observation-state semantics are invalid")


@dataclass(frozen=True)
class VerificationObservationErrorContract:
    """Source-aware error semantics and exact weighting tensors for scoring."""

    radar_source_kind: Literal["single_site", "mosaic"]
    source_calibration_epochs: tuple[tuple[str, str], ...]
    range_elevation_validity_domain_digest: str
    beam_blockage_visibility_mask_digest: str
    attenuation_qc_digest: str
    censoring_rule_digest: str
    spatial_correlation_block_digest: str
    quality_weight_interpretation_digest: str
    observation_error_model_digest: str
    minimum_detectable_echo_dbz: float
    observation_error_reference_std_dbz: float
    valid_mask_digest: str
    quality_weight_digest: str
    observation_std_dbz_digest: str
    observation_state_code_digest: str
    observation_error_plan_digest: str
    source_radar_index_map_digest: str | None = None
    spatial_correlation_role: Literal["diagnostic_only"] = "diagnostic_only"
    missing_data_taxonomy: tuple[str, ...] = (
        "observed_clear",
        "observed_echo",
        "source_missing",
        "qc_invalid",
        "beam_blocked",
        "below_detection_censored",
        "mosaic_source_unassigned",
    )
    metric_weight_rule: str = "quality-times-normalized-inverse-variance-v1"
    contract: str = "verification-observation-error-contract-v3"
    contract_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "verification-observation-error-contract-v3"
            or self.radar_source_kind not in {"single_site", "mosaic"}
            or self.metric_weight_rule
            != "quality-times-normalized-inverse-variance-v1"
            or self.spatial_correlation_role != "diagnostic_only"
            or self.missing_data_taxonomy
            != (
                "observed_clear",
                "observed_echo",
                "source_missing",
                "qc_invalid",
                "beam_blocked",
                "below_detection_censored",
                "mosaic_source_unassigned",
            )
            or not math.isfinite(self.minimum_detectable_echo_dbz)
            or not math.isfinite(self.observation_error_reference_std_dbz)
            or self.observation_error_reference_std_dbz <= 0.0
            or not self.source_calibration_epochs
            or tuple(sorted(self.source_calibration_epochs))
            != self.source_calibration_epochs
            or len({item[0] for item in self.source_calibration_epochs})
            != len(self.source_calibration_epochs)
        ):
            raise ValueError("verification observation-error contract is invalid")
        for name in (
            "range_elevation_validity_domain_digest",
            "beam_blockage_visibility_mask_digest",
            "attenuation_qc_digest",
            "censoring_rule_digest",
            "spatial_correlation_block_digest",
            "quality_weight_interpretation_digest",
            "observation_error_model_digest",
            "valid_mask_digest",
            "quality_weight_digest",
            "observation_std_dbz_digest",
            "observation_state_code_digest",
            "observation_error_plan_digest",
        ):
            _require_sha256(name, getattr(self, name))
        for source_digest, calibration_epoch_digest in self.source_calibration_epochs:
            _require_sha256("verification source radar", source_digest)
            _require_sha256("radar calibration epoch", calibration_epoch_digest)
        if self.radar_source_kind == "single_site":
            if len(self.source_calibration_epochs) != 1 or (
                self.source_radar_index_map_digest is not None
            ):
                raise ValueError("single-site observation-error source is invalid")
        elif self.source_radar_index_map_digest is None:
            raise ValueError("mosaic observation-error source map is required")
        else:
            _require_sha256(
                "verification source radar index map",
                self.source_radar_index_map_digest,
            )
        object.__setattr__(self, "contract_digest", json_digest(self.payload))

    @classmethod
    def from_tensors(
        cls,
        *,
        plan: VerificationObservationErrorPlan,
        valid_mask: Tensor,
        quality_weight: Tensor,
        observation_std_dbz: Tensor,
        frames_dbz: Tensor,
        observation_state_code: Tensor,
        source_radar_index_map: Tensor | None,
        source_calibration_epochs: tuple[tuple[str, str], ...],
        range_elevation_validity_domain_digest: str,
        beam_blockage_visibility_mask_digest: str,
        spatial_correlation_block_digest: str,
    ) -> VerificationObservationErrorContract:
        if type(plan) is not VerificationObservationErrorPlan:
            raise ValueError("verification observation-error plan is invalid")
        _validate_verification_cell_states(
            frames_dbz=frames_dbz,
            valid_mask=valid_mask,
            quality_weight=quality_weight,
            observation_std_dbz=observation_std_dbz,
            observation_state_code=observation_state_code,
            minimum_detectable_echo_dbz=plan.minimum_detectable_echo_dbz,
            radar_source_kind=plan.radar_source_kind,
            source_radar_index_map=source_radar_index_map,
        )
        return cls(
            radar_source_kind=plan.radar_source_kind,
            source_calibration_epochs=tuple(sorted(source_calibration_epochs)),
            range_elevation_validity_domain_digest=(
                range_elevation_validity_domain_digest
            ),
            beam_blockage_visibility_mask_digest=(
                beam_blockage_visibility_mask_digest
            ),
            attenuation_qc_digest=plan.attenuation_qc_digest,
            censoring_rule_digest=plan.censoring_rule_digest,
            spatial_correlation_block_digest=spatial_correlation_block_digest,
            quality_weight_interpretation_digest=(
                plan.quality_weight_interpretation_digest
            ),
            observation_error_model_digest=plan.observation_error_model_digest,
            minimum_detectable_echo_dbz=plan.minimum_detectable_echo_dbz,
            observation_error_reference_std_dbz=(
                plan.observation_error_reference_std_dbz
            ),
            valid_mask_digest=tensor_digest(valid_mask),
            quality_weight_digest=tensor_digest(quality_weight),
            observation_std_dbz_digest=tensor_digest(observation_std_dbz),
            observation_state_code_digest=tensor_digest(
                observation_state_code
            ),
            observation_error_plan_digest=plan.plan_digest,
            spatial_correlation_role=plan.spatial_correlation_role,
            source_radar_index_map_digest=(
                None
                if source_radar_index_map is None
                else tensor_digest(source_radar_index_map)
            ),
        )

    def validate_against_plan(
        self,
        plan: VerificationObservationErrorPlan,
    ) -> None:
        if (
            type(plan) is not VerificationObservationErrorPlan
            or self.contract_digest != json_digest(self.payload)
            or self.observation_error_plan_digest != plan.plan_digest
            or self.radar_source_kind != plan.radar_source_kind
            or self.attenuation_qc_digest != plan.attenuation_qc_digest
            or self.censoring_rule_digest != plan.censoring_rule_digest
            or self.quality_weight_interpretation_digest
            != plan.quality_weight_interpretation_digest
            or self.observation_error_model_digest
            != plan.observation_error_model_digest
            or self.minimum_detectable_echo_dbz
            != plan.minimum_detectable_echo_dbz
            or self.observation_error_reference_std_dbz
            != plan.observation_error_reference_std_dbz
            or self.spatial_correlation_role != plan.spatial_correlation_role
        ):
            raise ValueError(
                "verification contract disagrees with its observation-error plan"
            )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "radar_source_kind": self.radar_source_kind,
            "source_calibration_epochs": [
                list(item) for item in self.source_calibration_epochs
            ],
            "range_elevation_validity_domain_digest": (
                self.range_elevation_validity_domain_digest
            ),
            "beam_blockage_visibility_mask_digest": (
                self.beam_blockage_visibility_mask_digest
            ),
            "attenuation_qc_digest": self.attenuation_qc_digest,
            "censoring_rule_digest": self.censoring_rule_digest,
            "spatial_correlation_block_digest": (
                self.spatial_correlation_block_digest
            ),
            "spatial_correlation_role": self.spatial_correlation_role,
            "quality_weight_interpretation_digest": (
                self.quality_weight_interpretation_digest
            ),
            "observation_error_model_digest": self.observation_error_model_digest,
            "minimum_detectable_echo_dbz": self.minimum_detectable_echo_dbz,
            "observation_error_reference_std_dbz": (
                self.observation_error_reference_std_dbz
            ),
            "valid_mask_digest": self.valid_mask_digest,
            "quality_weight_digest": self.quality_weight_digest,
            "observation_std_dbz_digest": self.observation_std_dbz_digest,
            "observation_state_code_digest": (
                self.observation_state_code_digest
            ),
            "observation_error_plan_digest": (
                self.observation_error_plan_digest
            ),
            "source_radar_index_map_digest": self.source_radar_index_map_digest,
            "missing_data_taxonomy": list(self.missing_data_taxonomy),
            "metric_weight_rule": self.metric_weight_rule,
        }


@dataclass(frozen=True)
class VerificationBundle:
    """Content-addressed future radar/QC bundle for delayed verification."""

    frames_dbz: Tensor
    valid_mask: Tensor
    valid_times: tuple[str, ...]
    grid_contract_digest: str
    radar_product_digest: str
    qc_pipeline_digest: str
    mask_policy_digest: str | None = None
    censor_policy_digest: str | None = None
    reflectivity_resolution_dbz: float | None = None
    quantization_origin_dbz: float | None = None
    threshold_bin_convention: str | None = None
    floor_representation_contract_digest: str | None = None
    quality_weight: Tensor | None = None
    observation_std_dbz: Tensor | None = None
    observation_state_code: Tensor | None = None
    source_radar_index_map: Tensor | None = None
    observation_error_contract: VerificationObservationErrorContract | None = None
    contract: str = "radar-verification-bundle-v1"
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract not in {
            "radar-verification-bundle-v1",
            "radar-verification-bundle-v2",
            "radar-verification-bundle-v3",
            "radar-verification-bundle-v4",
            "radar-verification-bundle-v5",
            "radar-verification-bundle-v6",
        }:
            raise ValueError("unsupported verification bundle contract")
        if self.frames_dbz.ndim != 3 or not self.frames_dbz.is_floating_point():
            raise ValueError(
                "verification frames must be floating with shape [lead,H,W]"
            )
        if (
            self.valid_mask.dtype != torch.bool
            or self.valid_mask.shape != self.frames_dbz.shape
            or self.valid_mask.device != self.frames_dbz.device
        ):
            raise ValueError(
                "verification valid_mask must be boolean and match frames"
            )
        if bool(torch.any(self.valid_mask & ~torch.isfinite(self.frames_dbz))):
            raise ValueError("valid verification cells must contain finite dBZ")
        if (
            not isinstance(self.valid_times, tuple)
            or len(self.valid_times) != self.frames_dbz.shape[0]
        ):
            raise ValueError(
                "verification valid_times must match the lead dimension"
            )
        canonical_times = tuple(
            _canonical_verification_time(value) for value in self.valid_times
        )
        parsed_times = tuple(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in canonical_times
        )
        if any(
            later <= earlier
            for earlier, later in zip(parsed_times, parsed_times[1:])
        ):
            raise ValueError("verification valid_times must be increasing")
        for name, value in (
            ("grid_contract_digest", self.grid_contract_digest),
            ("radar_product_digest", self.radar_product_digest),
            ("qc_pipeline_digest", self.qc_pipeline_digest),
        ):
            _require_sha256(name, value)
        measurement_values = (
            self.mask_policy_digest,
            self.censor_policy_digest,
            self.reflectivity_resolution_dbz,
            self.quantization_origin_dbz,
            self.threshold_bin_convention,
            self.floor_representation_contract_digest,
        )
        if self.contract == "radar-verification-bundle-v1":
            if any(value is not None for value in measurement_values):
                raise ValueError(
                    "legacy verification bundles cannot claim v2 measurement lineage"
                )
        else:
            for name in (
                "mask_policy_digest",
                "censor_policy_digest",
                "floor_representation_contract_digest",
            ):
                value = getattr(self, name)
                if value is None:
                    raise ValueError("v2 verification measurement lineage is incomplete")
                _require_sha256(name, value)
            if (
                self.reflectivity_resolution_dbz is None
                or not math.isfinite(self.reflectivity_resolution_dbz)
                or self.reflectivity_resolution_dbz <= 0.0
                or self.quantization_origin_dbz is None
                or not math.isfinite(self.quantization_origin_dbz)
                or self.threshold_bin_convention
                != (
                    "threshold_edge_centered_bins"
                    if self.contract == "radar-verification-bundle-v2"
                    else "nearest_rounding_threshold_censor"
                )
            ):
                raise ValueError("v2 verification quantization contract is invalid")
        error_values = (
            self.quality_weight,
            self.observation_std_dbz,
            self.observation_state_code,
            self.observation_error_contract,
        )
        if self.contract != "radar-verification-bundle-v6":
            if any(value is not None for value in error_values) or (
                self.source_radar_index_map is not None
            ):
                raise ValueError("legacy verification cannot claim observation error")
        else:
            if any(value is None for value in error_values):
                raise ValueError("verification observation-error lineage is incomplete")
            quality = cast(Tensor, self.quality_weight)
            observation_std = cast(Tensor, self.observation_std_dbz)
            observation_state = cast(Tensor, self.observation_state_code)
            source_radar_index_map = self.source_radar_index_map
            error_contract = cast(
                VerificationObservationErrorContract,
                self.observation_error_contract,
            )
            if (
                type(error_contract) is not VerificationObservationErrorContract
                or quality.shape != self.frames_dbz.shape
                or observation_std.shape != self.frames_dbz.shape
                or not quality.is_floating_point()
                or not observation_std.is_floating_point()
                or quality.dtype != self.frames_dbz.dtype
                or observation_std.dtype != self.frames_dbz.dtype
                or quality.device != self.frames_dbz.device
                or observation_std.device != self.frames_dbz.device
                or not bool(torch.all(torch.isfinite(quality)))
                or not bool(torch.all(torch.isfinite(observation_std)))
                or not bool(torch.all((quality >= 0.0) & (quality <= 1.0)))
                or not bool(torch.all(quality.masked_select(~self.valid_mask) == 0.0))
                or not bool(torch.all(observation_std.masked_select(self.valid_mask) > 0.0))
                or not bool(torch.all(observation_std.masked_select(~self.valid_mask) == 0.0))
                or error_contract.valid_mask_digest != tensor_digest(self.valid_mask)
                or error_contract.quality_weight_digest != tensor_digest(quality)
                or error_contract.observation_std_dbz_digest
                != tensor_digest(observation_std)
                or error_contract.observation_state_code_digest
                != tensor_digest(observation_state)
                or error_contract.source_radar_index_map_digest
                != (
                    None
                    if source_radar_index_map is None
                    else tensor_digest(source_radar_index_map)
                )
                or error_contract.attenuation_qc_digest != self.qc_pipeline_digest
                or error_contract.censoring_rule_digest != self.censor_policy_digest
            ):
                raise ValueError("verification observation-error tensors are invalid")
            _validate_verification_cell_states(
                frames_dbz=self.frames_dbz,
                valid_mask=self.valid_mask,
                quality_weight=quality,
                observation_std_dbz=observation_std,
                observation_state_code=observation_state,
                minimum_detectable_echo_dbz=(
                    error_contract.minimum_detectable_echo_dbz
                ),
                radar_source_kind=error_contract.radar_source_kind,
                source_radar_index_map=source_radar_index_map,
            )
        frames = self.frames_dbz.detach().clone()
        valid = self.valid_mask.detach().clone()
        object.__setattr__(self, "frames_dbz", frames)
        object.__setattr__(self, "valid_mask", valid)
        if self.quality_weight is not None:
            object.__setattr__(self, "quality_weight", self.quality_weight.detach().clone())
        if self.observation_std_dbz is not None:
            object.__setattr__(
                self,
                "observation_std_dbz",
                self.observation_std_dbz.detach().clone(),
            )
        if self.observation_state_code is not None:
            object.__setattr__(
                self,
                "observation_state_code",
                self.observation_state_code.detach().clone(),
            )
        if self.source_radar_index_map is not None:
            object.__setattr__(
                self,
                "source_radar_index_map",
                self.source_radar_index_map.detach().clone(),
            )
        object.__setattr__(self, "valid_times", canonical_times)
        object.__setattr__(
            self,
            "content_digest",
            _verification_content_digest(
                self.contract,
                frames,
                valid,
                canonical_times,
                self.grid_contract_digest,
                self.radar_product_digest,
                self.qc_pipeline_digest,
                self.mask_policy_digest,
                self.censor_policy_digest,
                self.reflectivity_resolution_dbz,
                self.quantization_origin_dbz,
                self.threshold_bin_convention,
                self.floor_representation_contract_digest,
                self.quality_weight,
                self.observation_std_dbz,
                self.observation_state_code,
                self.source_radar_index_map,
                self.observation_error_contract,
            ),
        )

    def validate_integrity(self) -> None:
        expected = _verification_content_digest(
            self.contract,
            self.frames_dbz,
            self.valid_mask,
            self.valid_times,
            self.grid_contract_digest,
            self.radar_product_digest,
            self.qc_pipeline_digest,
            self.mask_policy_digest,
            self.censor_policy_digest,
            self.reflectivity_resolution_dbz,
            self.quantization_origin_dbz,
            self.threshold_bin_convention,
            self.floor_representation_contract_digest,
            self.quality_weight,
            self.observation_std_dbz,
            self.observation_state_code,
            self.source_radar_index_map,
            self.observation_error_contract,
        )
        if expected != self.content_digest:
            raise ValueError("verification bundle content digest mismatch")

    @property
    def metric_weight(self) -> Tensor:
        self.validate_integrity()
        if self.contract != "radar-verification-bundle-v6":
            return self.valid_mask.to(self.frames_dbz)
        quality = cast(Tensor, self.quality_weight)
        observation_std = cast(Tensor, self.observation_std_dbz)
        error_contract = cast(
            VerificationObservationErrorContract,
            self.observation_error_contract,
        )
        observation_state = cast(Tensor, self.observation_state_code)
        reference = observation_std.new_tensor(
            error_contract.observation_error_reference_std_dbz
        )
        inverse_variance = torch.where(
            self.valid_mask,
            (reference / observation_std.clamp_min(torch.finfo(observation_std.dtype).tiny)).square().clamp(max=1.0),
            torch.zeros_like(observation_std),
        )
        point_observation = observation_state != (
            VerificationCellState.BELOW_DETECTION_CENSORED
        )
        return (quality * inverse_variance * point_observation).detach().clone()


VerificationInput = Tensor | VerificationBundle

CONTEXT_FEATURE_NAMES_V13 = (
    "motion_dy",
    "motion_dx",
    "motion_speed",
    "log_growth",
    "motion_disagreement",
    "growth_disagreement",
    "motion_pair_conflict",
    "growth_pair_conflict",
    "tendency_pair_count",
    "tendency_source_observation",
    "tendency_source_background",
    "state_path_pair_count",
    "state_path_source_observation",
    "state_path_source_background",
    "state_path_conflict",
    "state_path_extrapolated",
    "state_path_age_available",
    "state_path_age_minutes",
    "state_path_psr_available",
    "log1p_state_path_minimum_psr",
    "growth_overlap_support_available",
    "log1p_minimum_growth_overlap_support",
    "growth_overlap_area_available",
    "log1p_minimum_growth_overlap_area_km2",
    "current_state_support_fraction",
    "background_contribution_fraction",
    "latest_observation_coverage",
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_integrated_echo",
    *tuple(
        f"motion_pair_selection_{selection.value.lower()}"
        for selection in TendencyPairSelection
    ),
    *tuple(
        f"growth_pair_selection_{selection.value.lower()}"
        for selection in TendencyPairSelection
    ),
    *tuple(
        f"state_path_mode_{selection.value.lower()}"
        for selection in TendencyPairSelection
    ),
    "phase_correlation_psr_available",
    "log1p_minimum_phase_correlation_psr",
    "projected_velocity_available",
    "projected_velocity_x_mps",
    "projected_velocity_y_mps",
    "projected_speed_mps",
    "motion_disagreement_mps_available",
    "motion_disagreement_mps",
    "area_weighted_echo_available",
    "log1p_linear_reflectivity_integral_km2",
    "grid_spacing_available",
    "grid_column_spacing_m",
    "grid_row_spacing_m",
)
CONTEXT_FEATURE_NAMES = (
    *CONTEXT_FEATURE_NAMES_V13,
    "observation_path_pair_count",
    "observation_path_conflict",
    "observation_path_extrapolated",
    "observation_path_age_available",
    "observation_path_age_minutes",
    "observation_path_psr_available",
    "log1p_observation_path_minimum_psr",
    "background_path_pair_count",
    "background_path_conflict",
    "background_path_extrapolated",
    "background_path_age_available",
    "background_path_age_minutes",
    "background_path_psr_available",
    "log1p_background_path_minimum_psr",
)


@dataclass(frozen=True)
class SensitivityConfig:
    """Fixed metric and compression choices for one sensitivity contract."""

    metric_names: tuple[str, ...] = DEFAULT_METRICS
    metric_domain: FSOMetricDomain = "issued"
    require_verification_lineage: bool = False
    required_verification_radar_product_digest: str | None = None
    required_verification_qc_pipeline_digest: str | None = None
    full_map_lead_minutes: tuple[int, ...] = (30, 60, 120, 180)
    tile_size: int = 16
    tile_size_m: float | None = None
    soft_fss_temperature_dbz: float = 2.0
    soft_fss_window: int = 9
    soft_fss_window_m: float | None = None
    minimum_fss_truth_mass: float = 0.5
    active_margin_dbz: float = 0.1
    linearity_delta: tuple[float, float, float] = (0.05, -0.04, 0.005)
    pair_conflict_trust_penalty: float = 0.5
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if not isinstance(self.metric_names, tuple):
            raise TypeError("metric_names must be a tuple")
        unknown = set(self.metric_names) - set(SUPPORTED_METRICS)
        if unknown:
            raise ValueError(f"unsupported metrics: {sorted(unknown)}")
        if not self.metric_names:
            raise ValueError("at least one metric is required")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names must be unique")
        if self.metric_domain not in (
            "issued",
            "radar_dynamics_anchored",
            "confidence_weighted",
        ):
            raise ValueError("unsupported FSO metric domain")
        if type(self.require_verification_lineage) is not bool:
            raise TypeError("require_verification_lineage must be Boolean")
        required_lineage = (
            self.required_verification_radar_product_digest,
            self.required_verification_qc_pipeline_digest,
        )
        if (required_lineage[0] is None) != (required_lineage[1] is None):
            raise ValueError(
                "verification radar product and QC digests must be paired"
            )
        if required_lineage[0] is not None:
            if not self.require_verification_lineage:
                raise ValueError(
                    "approved verification identities require lineage"
                )
            _require_sha256(
                "required_verification_radar_product_digest",
                required_lineage[0],
            )
            _require_sha256(
                "required_verification_qc_pipeline_digest",
                cast(str, required_lineage[1]),
            )
        if not isinstance(self.full_map_lead_minutes, tuple):
            raise TypeError("full_map_lead_minutes must be a tuple")
        if len(set(self.full_map_lead_minutes)) != len(
            self.full_map_lead_minutes
        ):
            raise ValueError("full_map_lead_minutes must be unique")
        if any(
            type(minutes) is not int or minutes <= 0
            for minutes in self.full_map_lead_minutes
        ):
            raise ValueError("full-map leads must be positive integers")
        if type(self.tile_size) is not int or self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        for name, value in (
            ("tile_size_m", self.tile_size_m),
            ("soft_fss_window_m", self.soft_fss_window_m),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(self.soft_fss_temperature_dbz)
            or self.soft_fss_temperature_dbz <= 0
        ):
            raise ValueError("soft_fss_temperature_dbz must be positive")
        if (
            type(self.soft_fss_window) is not int
            or self.soft_fss_window <= 0
            or self.soft_fss_window % 2 == 0
        ):
            raise ValueError("soft_fss_window must be a positive odd integer")
        if (
            not math.isfinite(self.minimum_fss_truth_mass)
            or self.minimum_fss_truth_mass <= 0
        ):
            raise ValueError("minimum_fss_truth_mass must be positive")
        if (
            not math.isfinite(self.active_margin_dbz)
            or self.active_margin_dbz <= 0
        ):
            raise ValueError("active_margin_dbz must be positive")
        if len(self.linearity_delta) != 3 or not all(
            math.isfinite(value) for value in self.linearity_delta
        ):
            raise ValueError("linearity_delta must contain three finite values")
        if (
            not math.isfinite(self.pair_conflict_trust_penalty)
            or not 0.0 < self.pair_conflict_trust_penalty <= 1.0
        ):
            raise ValueError("pair_conflict_trust_penalty must be in (0, 1]")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def digest(self) -> str:
        return dataclass_digest(self)

    @classmethod
    def for_automated_learning(
        cls,
        *,
        radar_product_digest: str,
        qc_pipeline_digest: str,
    ) -> SensitivityConfig:
        """Return the fail-closed metric policy for automated learning."""

        return cls(
            metric_names=(
                "log_echo_mse",
                "soft_fss_error_35",
                "centroid_error_m2",
            ),
            metric_domain="radar_dynamics_anchored",
            require_verification_lineage=True,
            required_verification_radar_product_digest=(
                radar_product_digest
            ),
            required_verification_qc_pipeline_digest=qc_pipeline_digest,
            tile_size_m=16_000.0,
            soft_fss_window_m=9_000.0,
        )


VariationalPreconditioner = Literal[
    "none",
    "prior_smoothness_diagonal",
]


@dataclass(frozen=True)
class VariationalAdjointConfig:
    """Execution and local-validity budget for delayed P1 adjoints."""

    lead_minutes: tuple[int, ...] | None = None
    pcg_relative_tolerance: float | None = None
    maximum_pcg_iterations: int | None = None
    maximum_normal_products: int = 10_000
    maximum_whitener_total_operations: int = 100_000_000_000
    maximum_materialized_output_bytes: int = 2 * 1024**3
    warm_start_by_metric: bool = True
    preconditioner: VariationalPreconditioner = (
        "prior_smoothness_diagonal"
    )
    minimum_detection_margin_dbz: float = 1.0e-3
    minimum_remap_fraction_margin: float = 1.0e-4
    minimum_output_cap_margin_dbz: float = 1.0e-3
    minimum_publication_margin: float = 1.0e-4
    minimum_neural_prior_valid_margin: float = 1.0e-3
    minimum_neural_prior_support_margin: float = 1.0e-3
    require_active_set_margin: bool = False
    minimum_reachability_margin: float = 1.0e-3
    minimum_unresolved_amplitude_fraction_margin: float = 1.0e-4
    minimum_amplitude_confidence_margin: float = 1.0e-3
    minimum_motion_saturation_margin_fraction: float = 1.0e-3
    minimum_motion_speed_saturation_margin_mps: float = 0.0
    minimum_growth_saturation_margin_per_step: float = 1.0e-4
    require_feasibility_margin: bool = False
    gauss_newton_probe_count: int = 4
    gauss_newton_probe_seed: int = 0
    maximum_gauss_newton_relative_curvature_defect: float = 0.25
    require_gauss_newton_reliability: bool = False
    maximum_detected_delta_dbz: float = 0.5
    maximum_censor_delta_dbz: float = 0.5
    maximum_observation_weight_delta: float = 0.1
    maximum_background_delta_dbz: float = 0.5
    maximum_perturbed_pixel_count: int = 4096
    maximum_perturbed_fraction: float = 0.05
    maximum_perturbed_area_km2: float | None = None
    maximum_whitened_perturbation_l2: float = 8.0
    perturbation_tile_size: int = 16
    perturbation_tile_size_m: float | None = None
    maximum_per_tile_whitened_norm: float = 4.0
    maximum_observation_weight_l2: float = 1.0
    minimum_observation_multiplier: float = 0.5
    require_baseline_dynamics_branch_validity: bool = False

    def __post_init__(self) -> None:
        if self.lead_minutes is not None:
            if not isinstance(self.lead_minutes, tuple):
                raise TypeError("adjoint lead_minutes must be a tuple")
            if not self.lead_minutes:
                raise ValueError("adjoint lead_minutes cannot be empty")
            if len(set(self.lead_minutes)) != len(self.lead_minutes):
                raise ValueError("adjoint lead_minutes must be unique")
            if any(
                type(value) is not int or value <= 0
                for value in self.lead_minutes
            ):
                raise ValueError(
                    "adjoint lead_minutes must contain positive integers"
                )
            if tuple(sorted(self.lead_minutes)) != self.lead_minutes:
                raise ValueError("adjoint lead_minutes must be increasing")
        if self.pcg_relative_tolerance is not None and (
            isinstance(self.pcg_relative_tolerance, bool)
            or not math.isfinite(self.pcg_relative_tolerance)
            or self.pcg_relative_tolerance <= 0.0
        ):
            raise ValueError("adjoint PCG tolerance must be positive")
        if self.maximum_pcg_iterations is not None and (
            type(self.maximum_pcg_iterations) is not int
            or self.maximum_pcg_iterations <= 0
        ):
            raise ValueError("adjoint PCG iterations must be positive")
        for name, value in (
            ("maximum_normal_products", self.maximum_normal_products),
            (
                "maximum_whitener_total_operations",
                self.maximum_whitener_total_operations,
            ),
            (
                "maximum_materialized_output_bytes",
                self.maximum_materialized_output_bytes,
            ),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warm_start_by_metric) is not bool:
            raise TypeError("warm_start_by_metric must be Boolean")
        if self.preconditioner not in (
            "none",
            "prior_smoothness_diagonal",
        ):
            raise ValueError("unsupported variational preconditioner")
        margins = (
            self.minimum_detection_margin_dbz,
            self.minimum_remap_fraction_margin,
            self.minimum_output_cap_margin_dbz,
            self.minimum_publication_margin,
            self.minimum_neural_prior_valid_margin,
            self.minimum_neural_prior_support_margin,
            self.minimum_reachability_margin,
            self.minimum_unresolved_amplitude_fraction_margin,
            self.minimum_amplitude_confidence_margin,
            self.minimum_motion_saturation_margin_fraction,
            self.minimum_motion_speed_saturation_margin_mps,
            self.minimum_growth_saturation_margin_per_step,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
            for value in margins
        ):
            raise ValueError("active-set margins must be finite and nonnegative")
        if type(self.require_active_set_margin) is not bool:
            raise TypeError("require_active_set_margin must be Boolean")
        if type(self.require_feasibility_margin) is not bool:
            raise TypeError("require_feasibility_margin must be Boolean")
        if (
            type(self.gauss_newton_probe_count) is not int
            or self.gauss_newton_probe_count <= 0
        ):
            raise ValueError("gauss_newton_probe_count must be positive")
        if (
            type(self.gauss_newton_probe_seed) is not int
            or self.gauss_newton_probe_seed < 0
        ):
            raise ValueError("gauss_newton_probe_seed cannot be negative")
        if (
            isinstance(
                self.maximum_gauss_newton_relative_curvature_defect,
                bool,
            )
            or not math.isfinite(
                self.maximum_gauss_newton_relative_curvature_defect
            )
            or self.maximum_gauss_newton_relative_curvature_defect < 0.0
        ):
            raise ValueError(
                "maximum Gauss-Newton curvature defect must be nonnegative"
            )
        if type(self.require_gauss_newton_reliability) is not bool:
            raise TypeError("require_gauss_newton_reliability must be Boolean")
        if type(self.require_baseline_dynamics_branch_validity) is not bool:
            raise TypeError(
                "require_baseline_dynamics_branch_validity must be Boolean"
            )
        perturbation_limits = (
            self.maximum_detected_delta_dbz,
            self.maximum_censor_delta_dbz,
            self.maximum_observation_weight_delta,
            self.maximum_background_delta_dbz,
            self.maximum_whitened_perturbation_l2,
            self.maximum_per_tile_whitened_norm,
            self.maximum_observation_weight_l2,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0.0
            for value in perturbation_limits
        ):
            raise ValueError("local FSOI perturbation limits must be positive")
        if (
            type(self.maximum_perturbed_pixel_count) is not int
            or self.maximum_perturbed_pixel_count <= 0
        ):
            raise ValueError(
                "maximum_perturbed_pixel_count must be a positive integer"
            )
        if (
            type(self.perturbation_tile_size) is not int
            or self.perturbation_tile_size <= 0
        ):
            raise ValueError(
                "perturbation_tile_size must be a positive integer"
            )
        for name, value in (
            ("maximum_perturbed_area_km2", self.maximum_perturbed_area_km2),
            ("perturbation_tile_size_m", self.perturbation_tile_size_m),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(self.maximum_perturbed_fraction)
            or not 0.0 < self.maximum_perturbed_fraction <= 1.0
        ):
            raise ValueError("maximum_perturbed_fraction must be in (0, 1]")
        if (
            not math.isfinite(self.minimum_observation_multiplier)
            or not 0.0 < self.minimum_observation_multiplier <= 1.0
        ):
            raise ValueError(
                "minimum_observation_multiplier must be in (0, 1]"
            )

    @property
    def digest(self) -> str:
        return dataclass_digest(self)

    @classmethod
    def for_automated_learning(cls) -> VariationalAdjointConfig:
        """Return local-validity gates required by automated learning."""

        return cls(
            lead_minutes=(30, 60, 120, 180),
            require_active_set_margin=True,
            require_feasibility_margin=True,
            require_gauss_newton_reliability=True,
            require_baseline_dynamics_branch_validity=True,
            maximum_perturbed_area_km2=256.0,
            perturbation_tile_size_m=16_000.0,
        )


@dataclass(frozen=True)
class DirectSensitivity:
    maps: Tensor
    norm: Tensor
    tile_norm: Tensor
    whitened_tile_norm: Tensor | None = None
    impact: Tensor | None = None
    tile_impact: Tensor | None = None
    reward: Tensor | None = None


@dataclass(frozen=True)
class VariationalSensitivityChannel:
    """One frozen-model observation-parameter sensitivity channel."""

    maps: Tensor
    norm_by_time: Tensor
    tile_norm_by_time: Tensor


@dataclass(frozen=True)
class VariationalObservationSensitivity:
    """Separated frozen-structure P1 observation sensitivities.

    ``detected_dbz`` is the observation-residual path retained by the
    original frozen-GN contract. ``initial_background_dbz`` is the direct
    and implicit path through the accepted first-frame values used to build
    the P1 initial background. ``baseline_dynamics_dbz`` is the direct and
    implicit path through continuous P0 displacement/growth, with pair,
    integer FFT peak, and every other discrete selection frozen.
    ``frozen_structure_input_dbz`` is the sum of these three dBZ paths.
    """

    detected_dbz: VariationalSensitivityChannel
    censor_threshold_dbz: VariationalSensitivityChannel
    observation_weight: VariationalSensitivityChannel
    initial_background_dbz: VariationalSensitivityChannel
    baseline_dynamics_dbz: VariationalSensitivityChannel
    frozen_structure_input_dbz: VariationalSensitivityChannel
    baseline_branch_trusted_frozen_structure_input_dbz: (
        VariationalSensitivityChannel | None
    )

    @property
    def trusted_frozen_structure_input_dbz(
        self,
    ) -> VariationalSensitivityChannel | None:
        """Compatibility alias for the narrower branch-trust name."""

        return self.baseline_branch_trusted_frozen_structure_input_dbz


@dataclass(frozen=True)
class VariationalActiveSetMargins:
    """Distances to discrete or piecewise-smooth FSO contract boundaries."""

    detection_classification_dbz: float | None
    analysis_remap_fraction: float
    forecast_remap_fraction: float
    output_cap_dbz: float | None
    publication_support: float
    publication_confidence: float | None
    neural_prior_valid_probability: float | None
    neural_prior_support_probability: float | None
    low_local_validity: bool


@dataclass(frozen=True)
class VariationalFeasibilityMargins:
    """Accepted P1 interiority relative to hard feasibility boundaries."""

    reachability_support: float
    unresolved_amplitude_fraction: float
    amplitude_confidence: float | None
    motion_saturation_fraction: float
    motion_speed_saturation_mps: float | None
    growth_saturation_per_step: float
    low_interior_validity: bool


def _variational_feasibility_margins(
    margins: AnalysisFeasibilityMargins,
    config: VariationalAdjointConfig,
) -> VariationalFeasibilityMargins:
    low_interior_validity = (
        margins.reachability_support
        < config.minimum_reachability_margin
        or margins.unresolved_amplitude_fraction
        < config.minimum_unresolved_amplitude_fraction_margin
        or margins.amplitude_confidence is None
        or margins.amplitude_confidence
        < config.minimum_amplitude_confidence_margin
        or margins.motion_saturation_fraction
        < config.minimum_motion_saturation_margin_fraction
        or (
            config.minimum_motion_speed_saturation_margin_mps > 0.0
            and (
                margins.motion_speed_saturation_mps is None
                or margins.motion_speed_saturation_mps
                < config.minimum_motion_speed_saturation_margin_mps
            )
        )
        or margins.growth_saturation_per_step
        < config.minimum_growth_saturation_margin_per_step
    )
    return VariationalFeasibilityMargins(
        reachability_support=margins.reachability_support,
        unresolved_amplitude_fraction=(
            margins.unresolved_amplitude_fraction
        ),
        amplitude_confidence=margins.amplitude_confidence,
        motion_saturation_fraction=margins.motion_saturation_fraction,
        motion_speed_saturation_mps=margins.motion_speed_saturation_mps,
        growth_saturation_per_step=margins.growth_saturation_per_step,
        low_interior_validity=low_interior_validity,
    )


@dataclass(frozen=True)
class VariationalGaussNewtonDiagnostics:
    """Random-probe defect of exact frozen curvature from GN curvature."""

    relative_curvature_defect: Tensor
    maximum_relative_curvature_defect: float
    reliable: bool
    normal_products: int
    exact_hessian_products: int


@dataclass(frozen=True)
class VariationalFSO:
    """Digest-bound P1 FSO under one frozen final IRLS/GN model."""

    contract: str
    forecast_run_digest: str
    analysis_input_digest: str
    sensitivity_config_digest: str
    adjoint_config_digest: str
    linearization_contract: str
    linearization_digest: str
    verification_contract: str
    verification_bundle_digest: str
    verification_lineage_complete: bool
    verification_valid_times: tuple[str, ...] | None
    verification_grid_contract_digest: str | None
    verification_radar_product_digest: str | None
    verification_qc_pipeline_digest: str | None
    metric_contract_digest: str
    algorithm_bundle_digest: str
    numerical_runtime_digest: str
    variational_fso_digest: str
    sensitivity_scope: str
    baseline_dynamics_frozen: bool
    baseline_pair_selection_frozen: bool
    baseline_dynamics_branch_status: BaselineDynamicsBranchStatus
    metric_names: tuple[str, ...]
    metric_domain: FSOMetricDomain
    metric_domain_digest: str
    lead_minutes: tuple[int, ...]
    full_map_lead_minutes: tuple[int, ...]
    tile_size: int
    tile_shape_yx: TileShape
    forecast_scores: Tensor
    metric_available: Tensor
    metric_domain_weight_sum: Tensor
    metric_domain_weight_fraction: Tensor
    forecast_cap_active_mask: Tensor
    observation: VariationalObservationSensitivity
    adjoint_iterations: Tensor
    adjoint_relative_residual: Tensor
    adjoint_true_residual_norm: Tensor
    adjoint_normal_products: Tensor
    adjoint_warm_started: Tensor
    total_normal_products: int
    whitener_operations_per_apply: int
    observed_whitener_apply_count: int
    materialized_output_bytes: int
    neural_prior_adjoint_direction_maximum_defect: float
    active_set_margins: VariationalActiveSetMargins
    feasibility_margins: VariationalFeasibilityMargins
    gauss_newton_diagnostics: VariationalGaussNewtonDiagnostics


@dataclass(frozen=True)
class VariationalObservationPerturbation:
    """Explicit first-order perturbation applied to the P1 observation model.

    ``detected_dbz`` perturbs detected reflectivity values,
    ``censor_threshold_dbz`` perturbs the censor threshold for censored events,
    ``observation_weight`` perturbs the unit objective multiplier for each
    valid observation. Optional ``initial_background_dbz`` independently
    perturbs the accepted first-frame values used by the P1 initial
    background, while optional ``baseline_dynamics_dbz`` perturbs the input
    dBZ values that generated continuous P0 motion/growth under the retained
    pair/peak selection. All channels are local perturbations; full observation
    removal requires a separate re-solve rather than this first-order contract.
    """

    detected_dbz: Tensor
    censor_threshold_dbz: Tensor
    observation_weight: Tensor
    initial_background_dbz: Tensor | None = None
    baseline_dynamics_dbz: Tensor | None = None
    physical_radar_dbz_delta: Tensor | None = None
    perturbation_semantics: PerturbationSemantics = "augmented_parameter"
    contract: str = "p1-observation-perturbation-v7"

    @classmethod
    def from_radar_dbz_delta(
        cls,
        delta_dbz: Tensor,
        linearization: AnalysisLinearization,
        *,
        neural_prior_runner: NeuralPriorInferenceRunner | None = None,
        neural_prior_application: NeuralPriorApplication | None = None,
    ) -> VariationalObservationPerturbation:
        """Map one physical radar-value change through retained input paths."""

        observations = linearization.observations
        frozen = linearization.frozen
        _validate_perturbation_tensor(
            "physical_radar_dbz_delta",
            delta_dbz,
            observations,
            observations.detected_mask,
            active_domain="detected observations",
        )
        active_delta = torch.where(
            observations.detected_mask,
            delta_dbz,
            torch.zeros_like(delta_dbz),
        )
        _ = _physical_radar_input_margins(
            active_delta,
            observations,
            frozen,
        )
        detected, background, dynamics = _physical_radar_channels(
            active_delta,
            observations,
            frozen,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        return cls(
            detected_dbz=detected,
            censor_threshold_dbz=torch.zeros_like(active_delta),
            observation_weight=torch.zeros_like(active_delta),
            initial_background_dbz=background,
            baseline_dynamics_dbz=dynamics,
            physical_radar_dbz_delta=active_delta,
            perturbation_semantics="physical_radar_value",
        )

    @classmethod
    def from_censor_threshold_delta(
        cls,
        delta_dbz: Tensor,
        linearization: AnalysisLinearization,
    ) -> VariationalObservationPerturbation:
        """Perturb the threshold that defines retained censored events."""

        observations = linearization.observations
        _validate_perturbation_tensor(
            "censor_threshold_dbz",
            delta_dbz,
            observations,
            observations.censored_mask,
            active_domain="censored observations",
        )
        zeros = torch.zeros_like(delta_dbz)
        return cls(
            detected_dbz=zeros,
            censor_threshold_dbz=delta_dbz,
            observation_weight=zeros,
        )

    @classmethod
    def from_censored_event_weight_delta(
        cls,
        delta_weight: Tensor,
        linearization: AnalysisLinearization,
    ) -> VariationalObservationPerturbation:
        """Perturb inclusion weight only for retained censored events."""

        observations = linearization.observations
        _validate_perturbation_tensor(
            "observation_weight",
            delta_weight,
            observations,
            observations.censored_mask,
            active_domain="censored observations",
        )
        zeros = torch.zeros_like(delta_weight)
        return cls(
            detected_dbz=zeros,
            censor_threshold_dbz=zeros,
            observation_weight=delta_weight,
        )

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "detected_dbz": tensor_digest(self.detected_dbz),
                "censor_threshold_dbz": tensor_digest(
                    self.censor_threshold_dbz
                ),
                "observation_weight": tensor_digest(
                    self.observation_weight
                ),
                "initial_background_dbz": (
                    None
                    if self.initial_background_dbz is None
                    else tensor_digest(self.initial_background_dbz)
                ),
                "baseline_dynamics_dbz": (
                    None
                    if self.baseline_dynamics_dbz is None
                    else tensor_digest(self.baseline_dynamics_dbz)
                ),
                "physical_radar_dbz_delta": (
                    None
                    if self.physical_radar_dbz_delta is None
                    else tensor_digest(self.physical_radar_dbz_delta)
                ),
                "perturbation_semantics": self.perturbation_semantics,
            }
        )


@dataclass(frozen=True)
class VariationalPerturbationDiagnostics:
    perturbed_pixel_count: int
    perturbed_fraction: float
    perturbed_area_km2: float | None
    whitened_l2: float
    maximum_per_tile_whitened_norm: float
    observation_weight_l2: float
    minimum_input_floor_margin_dbz: float | None
    minimum_input_ceiling_margin_dbz: float | None
    directional_classification_valid: bool
    baseline_dynamics_branch_status: BaselineDynamicsBranchStatus
    baseline_dynamics_branch_signature_digest: str | None


def _physical_radar_channels(
    delta_dbz: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    detected = torch.where(
        observations.detected_mask,
        delta_dbz,
        torch.zeros_like(delta_dbz),
    )
    background = torch.zeros_like(delta_dbz)
    if frozen.observation_derived_initial_background:
        background[0] = torch.where(
            observations.detected_mask[0] & frozen.observed_mask[0],
            delta_dbz[0],
            torch.zeros_like(delta_dbz[0]),
        )
    elif frozen.neural_prior_dependency == "radar_dependent":
        if neural_prior_runner is None:
            raise ValueError("radar-dependent prior perturbation requires a runner")
        _validate_retained_prior_runner(
            frozen,
            neural_prior_runner,
            neural_prior_application,
        )
        background[0] = torch.where(
            _neural_prior_derivative_mask(frozen),
            neural_prior_runner.jvp(
                _require_bound_neural_prior_input(frozen),
                delta_dbz,
            ),
            torch.zeros_like(background[0]),
        )
    dynamics = torch.zeros_like(delta_dbz)
    if frozen.baseline_metadata.tendency_source is TendencySource.OBSERVATION:
        dynamics = torch.where(
            observations.detected_mask & frozen.observed_mask,
            delta_dbz,
            dynamics,
        )
    return detected, background, dynamics


def _validate_retained_prior_runner(
    frozen: FrozenOuterState,
    runner: NeuralPriorInferenceRunner,
    application: NeuralPriorApplication | None,
) -> None:
    raw = frozen.neural_prior_raw_background_dbz
    execution_digest = frozen.neural_prior_execution_contract_digest
    if raw is None or execution_digest is None:
        raise ValueError("neural-prior restart state is incomplete")
    if application is not None:
        if application.application_digest != frozen.neural_prior_application_digest:
            raise ValueError("neural-prior perturbation application mismatch")
        runner.reproduce(
            application,
            _require_bound_neural_prior_input(frozen),
        )
    runner.validate_retained_output(
        _require_bound_neural_prior_input(frozen),
        raw,
        execution_contract_digest=execution_digest,
    )


def _neural_prior_derivative_mask(
    frozen: FrozenOuterState,
) -> Tensor:
    """Return the retained interior branch of the consumed prior output."""

    valid = frozen.neural_prior_valid_mask
    raw = frozen.neural_prior_raw_background_dbz
    if valid is None or raw is None:
        raise ValueError("neural-prior derivative requires retained prior state")
    return valid


def _require_bound_neural_prior_input(
    frozen: FrozenOuterState,
) -> BoundNeuralPriorInput:
    bound_input = frozen.neural_prior_bound_input
    if bound_input is None:
        raise ValueError("neural-prior bound input is missing from restart state")
    return bound_input


def _neural_prior_support_margin(
    frozen: FrozenOuterState,
    application: NeuralPriorApplication | None,
) -> float | None:
    """Distance from retained soft support to its contracted hard branch."""

    if frozen.neural_prior_dependency is None:
        return None
    if application is None or (
        application.application_digest != frozen.neural_prior_application_digest
    ):
        return None
    selected = application.support_probability.masked_select(
        application.valid_mask
    )
    if selected.numel() == 0:
        return None
    return float(
        torch.amin(
            torch.abs(
                selected
                - application.state_contract.support_decision_probability
            )
        ).detach()
    )


def _neural_prior_valid_margin(
    frozen: FrozenOuterState,
    application: NeuralPriorApplication | None,
) -> float | None:
    """Distance from probabilistic validity to its retained hard branch."""

    if frozen.neural_prior_dependency is None:
        return None
    if application is None or (
        application.application_digest != frozen.neural_prior_application_digest
    ):
        return None
    if application.inference_evidence.validity_contract == "exogenous_static":
        return math.inf
    selected = application.valid_probability
    if selected.numel() == 0:
        return None
    return float(
        torch.amin(
            torch.abs(
                selected
                - application.state_contract.valid_decision_probability
            )
        ).detach()
    )


def _physical_radar_input_margins(
    delta_dbz: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> tuple[float | None, float | None]:
    """Fail closed when a physical perturbation crosses input clipping."""

    changed_mask = delta_dbz != 0
    if not bool(torch.any(changed_mask)):
        return None, None
    changed = (observations.dbz + delta_dbz).masked_select(changed_mask)
    floor_margin = float(
        torch.amin(changed - frozen.nowcast_config.min_dbz).detach()
    )
    ceiling_margin = float(
        torch.amin(frozen.nowcast_config.max_dbz - changed).detach()
    )
    if floor_margin < 0.0 or ceiling_margin < 0.0:
        raise ValueError("physical radar perturbation crosses input clamp")
    return floor_margin, ceiling_margin


@dataclass(frozen=True)
class VariationalImpactChannel:
    """Signed first-order metric change from one perturbation channel."""

    maps: Tensor
    sum_by_time: Tensor
    tile_sum_by_time: Tensor


@dataclass(frozen=True)
class VariationalObservationImpact:
    """Component and total signed P1 observation-impact estimates."""

    detected_dbz: VariationalImpactChannel
    censor_threshold_dbz: VariationalImpactChannel
    observation_weight: VariationalImpactChannel
    initial_background_dbz: VariationalImpactChannel
    baseline_dynamics_dbz: VariationalImpactChannel
    total: VariationalImpactChannel
    baseline_branch_trusted_total: VariationalImpactChannel | None

    @property
    def trusted_total(self) -> VariationalImpactChannel | None:
        """Compatibility alias; this certifies only the baseline branch."""

        return self.baseline_branch_trusted_total


@dataclass(frozen=True)
class VariationalFSOI:
    """Explicit-perturbation first-order impact derived from P1 FSO."""

    contract: str
    fso: VariationalFSO
    perturbation: VariationalObservationPerturbation
    perturbation_contract: str
    perturbation_digest: str
    perturbation_diagnostics: VariationalPerturbationDiagnostics
    baseline_dynamics_branch_status: BaselineDynamicsBranchStatus
    observation: VariationalObservationImpact
    variational_fsoi_digest: str


@dataclass(frozen=True)
class ObservationRemovalConfig:
    """Budget for an explicit, nonlocal observation-denial experiment."""

    maximum_removed_observation_count: int = 4096
    maximum_removed_fraction: float = 0.05
    maximum_removed_area_km2: float | None = 256.0
    maximum_whitener_total_operations: int = 100_000_000_000
    contract: str = "p1-observation-removal-config-v1"

    def __post_init__(self) -> None:
        if self.contract != "p1-observation-removal-config-v1":
            raise ValueError("unsupported observation-removal config")
        if (
            type(self.maximum_removed_observation_count) is not int
            or self.maximum_removed_observation_count <= 0
        ):
            raise ValueError("maximum removed observation count must be positive")
        if (
            isinstance(self.maximum_removed_fraction, bool)
            or not math.isfinite(self.maximum_removed_fraction)
            or not 0.0 < self.maximum_removed_fraction <= 1.0
        ):
            raise ValueError("maximum removed fraction must be in (0, 1]")
        if self.maximum_removed_area_km2 is not None and (
            isinstance(self.maximum_removed_area_km2, bool)
            or not math.isfinite(self.maximum_removed_area_km2)
            or self.maximum_removed_area_km2 <= 0.0
        ):
            raise ValueError("maximum removed area must be positive")
        if (
            type(self.maximum_whitener_total_operations) is not int
            or self.maximum_whitener_total_operations <= 0
        ):
            raise ValueError("removal whitener operation budget must be positive")

    @property
    def digest(self) -> str:
        return dataclass_digest(self)


@dataclass(frozen=True)
class ObservationRemovalRequest:
    """A set of accepted observations removed from the full P1 input."""

    removal_mask: Tensor
    linearization_digest: str
    contract: str = "p1-observation-removal-request-v1"
    request_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-observation-removal-request-v1":
            raise ValueError("unsupported observation-removal request")
        if (
            not isinstance(self.removal_mask, Tensor)
            or self.removal_mask.dtype is not torch.bool
            or self.removal_mask.ndim != 3
        ):
            raise TypeError("removal mask must be a Boolean [time, y, x] Tensor")
        if not bool(torch.any(self.removal_mask)):
            raise ValueError("removal mask must select at least one observation")
        _require_sha256("linearization_digest", self.linearization_digest)
        owned = self.removal_mask.detach().clone()
        object.__setattr__(self, "removal_mask", owned)
        object.__setattr__(
            self,
            "request_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "removal_mask": tensor_digest(owned),
                    "linearization_digest": self.linearization_digest,
                }
            ),
        )


@dataclass(frozen=True)
class ObservationRemovalImpact:
    """Resolved forecast-error change after a full observation denial."""

    request: ObservationRemovalRequest
    nominal_scores: Tensor
    removed_scores: Tensor
    metric_change: Tensor
    metric_available: Tensor
    lead_minutes: tuple[int, ...]
    metric_names: tuple[str, ...]
    metric_domain: FSOMetricDomain
    nominal_forecast_digest: str
    removed_forecast_digest: str
    removed_linearization_digest: str
    verification_bundle_digest: str
    sensitivity_config_digest: str
    removal_config_digest: str
    removed_observation_count: int
    removed_fraction: float
    removed_area_km2: float | None
    whitener_operations_per_apply: int
    observed_whitener_apply_count: int
    observed_whitener_total_operations: int
    contract: str = "p1-resolved-observation-removal-impact-v1"
    observation_removal_impact_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-resolved-observation-removal-impact-v1":
            raise ValueError("unsupported observation-removal impact")
        for name in (
            "nominal_scores",
            "removed_scores",
            "metric_change",
            "metric_available",
        ):
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            object.__setattr__(self, name, value.detach().clone())
        expected_shape = (len(self.lead_minutes), len(self.metric_names))
        if any(
            getattr(self, name).shape != expected_shape
            for name in (
                "nominal_scores",
                "removed_scores",
                "metric_change",
                "metric_available",
            )
        ):
            raise ValueError("observation-removal metric shapes disagree")
        if self.metric_available.dtype is not torch.bool:
            raise TypeError("observation-removal availability must be Boolean")
        available = self.metric_available
        if not torch.allclose(
            self.metric_change[available],
            (self.removed_scores - self.nominal_scores)[available],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("observation-removal metric change is inconsistent")
        for name in (
            "nominal_forecast_digest",
            "removed_forecast_digest",
            "removed_linearization_digest",
            "verification_bundle_digest",
            "sensitivity_config_digest",
            "removal_config_digest",
        ):
            _require_sha256(name, getattr(self, name))
        if type(self.removed_observation_count) is not int or (
            self.removed_observation_count <= 0
        ):
            raise ValueError("removed observation count must be positive")
        if not 0.0 < self.removed_fraction <= 1.0:
            raise ValueError("removed observation fraction must be in (0, 1]")
        if self.removed_area_km2 is not None and self.removed_area_km2 <= 0.0:
            raise ValueError("removed observation area must be positive")
        for name in (
            "whitener_operations_per_apply",
            "observed_whitener_apply_count",
            "observed_whitener_total_operations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.observed_whitener_total_operations != (
            self.whitener_operations_per_apply
            * self.observed_whitener_apply_count
        ):
            raise ValueError("observation-removal whitener accounting mismatch")
        object.__setattr__(
            self,
            "observation_removal_impact_digest",
            _observation_removal_impact_digest(self),
        )


def _observation_removal_impact_digest(
    impact: ObservationRemovalImpact,
) -> str:
    return json_digest(
        {
            "contract": impact.contract,
            "request_digest": impact.request.request_digest,
            "nominal_scores": tensor_digest(impact.nominal_scores),
            "removed_scores": tensor_digest(impact.removed_scores),
            "metric_change": tensor_digest(impact.metric_change),
            "metric_available": tensor_digest(impact.metric_available),
            "lead_minutes": list(impact.lead_minutes),
            "metric_names": list(impact.metric_names),
            "metric_domain": impact.metric_domain,
            "nominal_forecast_digest": impact.nominal_forecast_digest,
            "removed_forecast_digest": impact.removed_forecast_digest,
            "removed_linearization_digest": (
                impact.removed_linearization_digest
            ),
            "verification_bundle_digest": impact.verification_bundle_digest,
            "sensitivity_config_digest": impact.sensitivity_config_digest,
            "removal_config_digest": impact.removal_config_digest,
            "removed_observation_count": impact.removed_observation_count,
            "removed_fraction": impact.removed_fraction,
            "removed_area_km2": impact.removed_area_km2,
            "whitener_operations_per_apply": (
                impact.whitener_operations_per_apply
            ),
            "observed_whitener_apply_count": (
                impact.observed_whitener_apply_count
            ),
            "observed_whitener_total_operations": (
                impact.observed_whitener_total_operations
            ),
        }
    )


@dataclass(frozen=True)
class MetricTaylorThreshold:
    """Dimensionally correct Taylor limits for one forecast metric."""

    metric_name: str
    maximum_absolute_error: float
    material_impact_threshold: float
    ranking_scale: float | None = None
    ranking_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.metric_name not in SUPPORTED_METRICS:
            raise ValueError("unsupported Taylor metric")
        if (
            isinstance(self.maximum_absolute_error, bool)
            or not math.isfinite(self.maximum_absolute_error)
            or self.maximum_absolute_error < 0.0
        ):
            raise ValueError("metric Taylor absolute error must be nonnegative")
        if (
            isinstance(self.material_impact_threshold, bool)
            or not math.isfinite(self.material_impact_threshold)
            or self.material_impact_threshold <= 0.0
        ):
            raise ValueError("metric material impact threshold must be positive")
        if self.ranking_scale is not None and (
            isinstance(self.ranking_scale, bool)
            or not math.isfinite(self.ranking_scale)
            or self.ranking_scale <= 0.0
        ):
            raise ValueError("metric ranking scale must be positive")
        if (
            isinstance(self.ranking_weight, bool)
            or not math.isfinite(self.ranking_weight)
            or self.ranking_weight < 0.0
        ):
            raise ValueError("metric ranking weight must be nonnegative")

    @property
    def effective_ranking_scale(self) -> float:
        return (
            self.material_impact_threshold
            if self.ranking_scale is None
            else self.ranking_scale
        )


DEFAULT_METRIC_TAYLOR_THRESHOLDS = (
    MetricTaylorThreshold("log_echo_mse", 1.0e-6, 1.0e-6),
    MetricTaylorThreshold("soft_fss_error_35", 1.0e-6, 1.0e-6),
    MetricTaylorThreshold("centroid_error_m2", 1.0, 1.0),
)


@dataclass(frozen=True)
class AutomatedLearningPolicy:
    """One externally approved bundle for automated FSOI learning."""

    sensitivity_config: SensitivityConfig
    adjoint_config: VariationalAdjointConfig
    algorithm_bundle_digest: str
    numerical_runtime_digest: str
    metric_taylor_thresholds: tuple[
        MetricTaylorThreshold, ...
    ] = DEFAULT_METRIC_TAYLOR_THRESHOLDS
    maximum_linearity_relative_error: float = 0.1
    ranking_objective: CandidateRankingObjective = "expected_error_reduction"
    ranking_lead_weights: tuple[float, ...] = ()
    maximum_candidate_count: int = 10_000
    maximum_learning_candidates_to_validate: int = 32
    maximum_total_robust_resolves: int = 64
    maximum_candidate_bytes: int = 64 * 1024**2
    maximum_candidate_nonzeros: int = 1_000_000
    maximum_candidate_scoring_operations: int = 1_000_000_000
    maximum_candidate_ranking_wall_seconds: float = 300.0
    maximum_learning_pcg_iterations: int = 100_000
    maximum_learning_wall_seconds: float = 3_600.0
    maximum_whitener_total_operations: int = 100_000_000_000
    contract: str = "p1-automated-learning-policy-v7"

    def __post_init__(self) -> None:
        if self.contract != "p1-automated-learning-policy-v7":
            raise ValueError("unsupported automated-learning policy")
        _require_sha256(
            "algorithm_bundle_digest",
            self.algorithm_bundle_digest,
        )
        _require_sha256(
            "numerical_runtime_digest",
            self.numerical_runtime_digest,
        )
        if (
            isinstance(self.maximum_linearity_relative_error, bool)
            or not math.isfinite(self.maximum_linearity_relative_error)
            or not 0.0 < self.maximum_linearity_relative_error < 1.0
        ):
            raise ValueError(
                "maximum_linearity_relative_error must be in (0, 1)"
            )
        sensitivity = self.sensitivity_config
        if (
            sensitivity.metric_domain != "radar_dynamics_anchored"
            or not sensitivity.require_verification_lineage
            or sensitivity.required_verification_radar_product_digest is None
            or sensitivity.required_verification_qc_pipeline_digest is None
            or sensitivity.tile_size_m is None
            or (
                "soft_fss_error_35" in sensitivity.metric_names
                and sensitivity.soft_fss_window_m is None
            )
            or "centroid_error" in sensitivity.metric_names
        ):
            raise ValueError(
                "automated learning requires physical metrics and approved "
                "verification lineage"
            )
        adjoint = self.adjoint_config
        if not all(
            (
                adjoint.require_active_set_margin,
                adjoint.require_feasibility_margin,
                adjoint.require_gauss_newton_reliability,
                adjoint.require_baseline_dynamics_branch_validity,
                adjoint.maximum_perturbed_area_km2 is not None,
                adjoint.perturbation_tile_size_m is not None,
            )
        ):
            raise ValueError(
                "automated learning requires every local-validity gate"
            )
        if adjoint.lead_minutes != sensitivity.full_map_lead_minutes:
            raise ValueError(
                "automated learning requires a full map for every adjoint lead"
            )
        adjoint_leads = adjoint.lead_minutes
        if adjoint_leads is None:
            raise ValueError("automated learning requires explicit adjoint leads")
        if self.ranking_objective not in (
            "absolute_influence",
            "expected_error_reduction",
            "two_sided_diagnostic",
        ):
            raise ValueError("unsupported candidate ranking objective")
        if not isinstance(self.ranking_lead_weights, tuple):
            raise TypeError("ranking_lead_weights must be a tuple")
        if self.ranking_lead_weights and len(self.ranking_lead_weights) != len(
            adjoint_leads
        ):
            raise ValueError("ranking lead weights must match adjoint leads")
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
            for value in self.ranking_lead_weights
        ):
            raise ValueError("ranking lead weights must be nonnegative")
        if self.ranking_lead_weights and not any(self.ranking_lead_weights):
            raise ValueError("at least one ranking lead weight must be positive")
        for name, value in (
            ("maximum_candidate_count", self.maximum_candidate_count),
            (
                "maximum_learning_candidates_to_validate",
                self.maximum_learning_candidates_to_validate,
            ),
            ("maximum_total_robust_resolves", self.maximum_total_robust_resolves),
            ("maximum_candidate_bytes", self.maximum_candidate_bytes),
            ("maximum_candidate_nonzeros", self.maximum_candidate_nonzeros),
            (
                "maximum_candidate_scoring_operations",
                self.maximum_candidate_scoring_operations,
            ),
            (
                "maximum_learning_pcg_iterations",
                self.maximum_learning_pcg_iterations,
            ),
            (
                "maximum_whitener_total_operations",
                self.maximum_whitener_total_operations,
            ),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_learning_candidates_to_validate > self.maximum_candidate_count:
            raise ValueError(
                "learning validation count cannot exceed candidate count"
            )
        required_resolves = 2 * self.maximum_learning_candidates_to_validate
        if self.maximum_total_robust_resolves < required_resolves:
            raise ValueError(
                "robust-resolve budget must cover full/half candidate checks"
            )
        if (
            self.adjoint_config.maximum_whitener_total_operations
            > self.maximum_whitener_total_operations
        ):
            raise ValueError(
                "adjoint whitener budget cannot exceed the learning budget"
            )
        if (
            isinstance(self.maximum_learning_wall_seconds, bool)
            or not math.isfinite(self.maximum_learning_wall_seconds)
            or self.maximum_learning_wall_seconds <= 0.0
        ):
            raise ValueError("maximum_learning_wall_seconds must be positive")
        if (
            isinstance(self.maximum_candidate_ranking_wall_seconds, bool)
            or not math.isfinite(self.maximum_candidate_ranking_wall_seconds)
            or self.maximum_candidate_ranking_wall_seconds <= 0.0
        ):
            raise ValueError(
                "maximum_candidate_ranking_wall_seconds must be positive"
            )
        if not isinstance(self.metric_taylor_thresholds, tuple):
            raise TypeError("metric_taylor_thresholds must be a tuple")
        names = tuple(
            threshold.metric_name
            for threshold in self.metric_taylor_thresholds
        )
        if len(set(names)) != len(names):
            raise ValueError("metric Taylor thresholds must be unique")
        missing = set(sensitivity.metric_names) - set(names)
        if missing:
            raise ValueError(
                f"missing metric Taylor thresholds: {sorted(missing)}"
            )
        if not any(
            self.threshold_for(name).ranking_weight > 0.0
            for name in sensitivity.metric_names
        ):
            raise ValueError("at least one ranking metric weight must be positive")
        object.__setattr__(
            self,
            "metric_taylor_thresholds",
            tuple(
                sorted(
                    self.metric_taylor_thresholds,
                    key=lambda threshold: threshold.metric_name,
                )
            ),
        )

    def threshold_for(self, metric_name: str) -> MetricTaylorThreshold:
        for threshold in self.metric_taylor_thresholds:
            if threshold.metric_name == metric_name:
                return threshold
        raise ValueError(f"missing Taylor threshold for {metric_name}")

    @property
    def ranking_adjoint_config(self) -> VariationalAdjointConfig:
        """Use strict numerics while deferring candidate-specific branch checks."""

        return replace(
            self.adjoint_config,
            require_baseline_dynamics_branch_validity=False,
        )

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "sensitivity_config_digest": self.sensitivity_config.digest,
                "adjoint_config_digest": self.adjoint_config.digest,
                "algorithm_bundle_digest": self.algorithm_bundle_digest,
                "numerical_runtime_digest": self.numerical_runtime_digest,
                "metric_taylor_thresholds": [
                    {
                        "metric_name": threshold.metric_name,
                        "maximum_absolute_error": (
                            threshold.maximum_absolute_error
                        ),
                        "material_impact_threshold": (
                            threshold.material_impact_threshold
                        ),
                        "ranking_scale": threshold.effective_ranking_scale,
                        "ranking_weight": threshold.ranking_weight,
                    }
                    for threshold in self.metric_taylor_thresholds
                ],
                "maximum_linearity_relative_error": (
                    self.maximum_linearity_relative_error
                ),
                "ranking_objective": self.ranking_objective,
                "ranking_lead_weights": list(self.resolved_ranking_lead_weights),
                "maximum_candidate_count": self.maximum_candidate_count,
                "maximum_learning_candidates_to_validate": (
                    self.maximum_learning_candidates_to_validate
                ),
                "maximum_total_robust_resolves": (
                    self.maximum_total_robust_resolves
                ),
                "maximum_candidate_bytes": self.maximum_candidate_bytes,
                "maximum_candidate_nonzeros": self.maximum_candidate_nonzeros,
                "maximum_candidate_scoring_operations": (
                    self.maximum_candidate_scoring_operations
                ),
                "maximum_candidate_ranking_wall_seconds": (
                    self.maximum_candidate_ranking_wall_seconds
                ),
                "maximum_learning_pcg_iterations": (
                    self.maximum_learning_pcg_iterations
                ),
                "maximum_learning_wall_seconds": self.maximum_learning_wall_seconds,
                "maximum_whitener_total_operations": (
                    self.maximum_whitener_total_operations
                ),
                "perturbation_semantics": "physical_radar_value",
            }
        )

    @property
    def resolved_ranking_lead_weights(self) -> tuple[float, ...]:
        if self.ranking_lead_weights:
            return self.ranking_lead_weights
        lead_minutes = self.adjoint_config.lead_minutes
        if lead_minutes is None:
            raise RuntimeError("automated learning policy lacks adjoint leads")
        return (1.0,) * len(lead_minutes)


@dataclass(frozen=True)
class SparseRadarPerturbation:
    """Sparse physical radar-value delta used by candidate ranking."""

    flat_indices: Tensor
    delta_values: Tensor
    shape: tuple[int, int, int]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.flat_indices.ndim != 1
            or self.flat_indices.dtype != torch.int64
        ):
            raise ValueError("sparse candidate indices must be int64 and 1-D")
        if (
            self.delta_values.ndim != 1
            or not self.delta_values.is_floating_point()
            or self.delta_values.shape != self.flat_indices.shape
        ):
            raise ValueError("sparse candidate values must be floating and 1-D")
        if len(self.shape) != 3 or any(
            type(value) is not int or value <= 0 for value in self.shape
        ):
            raise ValueError("sparse candidate shape must be positive [3,H,W]")
        if self.shape[0] != 3:
            raise ValueError("sparse radar candidates require three input times")
        indices = self.flat_indices.detach().clone()
        values = self.delta_values.detach().clone()
        if indices.numel() == 0:
            raise ValueError("sparse candidate cannot be empty")
        if not bool(torch.all(torch.isfinite(values))) or bool(
            torch.any(values == 0)
        ):
            raise ValueError("sparse candidate values must be finite and nonzero")
        size = math.prod(self.shape)
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= size)):
            raise ValueError("sparse candidate index is outside its shape")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("sparse candidate indices must be unique")
        order = torch.argsort(indices)
        indices = indices[order]
        values = values[order]
        object.__setattr__(self, "flat_indices", indices)
        object.__setattr__(self, "delta_values", values)
        object.__setattr__(
            self,
            "digest",
            _sparse_radar_perturbation_digest(self),
        )

    @classmethod
    def from_dense(cls, delta_dbz: Tensor) -> SparseRadarPerturbation:
        if delta_dbz.ndim != 3 or not delta_dbz.is_floating_point():
            raise ValueError("dense radar candidate must be floating [3,H,W]")
        flat = delta_dbz.reshape(-1)
        indices = torch.nonzero(flat != 0, as_tuple=False).flatten()
        return cls(
            indices.to(torch.int64),
            flat[indices],
            (delta_dbz.shape[0], delta_dbz.shape[1], delta_dbz.shape[2]),
        )

    @property
    def nonzero_count(self) -> int:
        return self.flat_indices.numel()

    @property
    def retained_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.flat_indices, self.delta_values)
        )

    def materialize(self, reference: Tensor) -> Tensor:
        if self.digest != _sparse_radar_perturbation_digest(self):
            raise ValueError("sparse candidate digest mismatch")
        if tuple(reference.shape) != self.shape:
            raise ValueError("sparse candidate shape mismatch")
        result = reference.new_zeros(self.shape).reshape(-1)
        result.index_copy_(
            0,
            self.flat_indices.to(reference.device),
            self.delta_values.to(dtype=reference.dtype, device=reference.device),
        )
        return result.reshape(self.shape)


def _sparse_radar_perturbation_digest(
    perturbation: SparseRadarPerturbation,
) -> str:
    return json_digest(
        {
            "contract": "sparse-radar-perturbation-v1",
            "shape": list(perturbation.shape),
            "flat_indices": tensor_digest(perturbation.flat_indices),
            "delta_values": tensor_digest(perturbation.delta_values),
        }
    )


@dataclass(frozen=True)
class VariationalCandidatePrecheck:
    candidate_id: str
    perturbation_digest: str
    admissible: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        _require_sha256("candidate perturbation digest", self.perturbation_digest)
        if self.admissible == (self.rejection_reason is not None):
            raise ValueError("candidate precheck status and reason disagree")


@dataclass(frozen=True)
class VariationalCandidateScore:
    """One admissible candidate's dimensionless frozen-domain score."""

    candidate_id: str
    perturbation: SparseRadarPerturbation
    predicted_metric_change: Tensor
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a nonempty string")
        if not isinstance(self.predicted_metric_change, Tensor):
            raise TypeError("candidate prediction must be a Tensor")
        object.__setattr__(
            self,
            "predicted_metric_change",
            self.predicted_metric_change.detach().clone(),
        )
        if not math.isfinite(self.score) or self.score < 0.0:
            raise ValueError("candidate score must be finite and nonnegative")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("candidate rank must be a positive integer")


@dataclass(frozen=True)
class VariationalCandidateRanking:
    """Content-bound ranking produced from one shared FSO solve."""

    fso: VariationalFSO
    scores: tuple[VariationalCandidateScore, ...]
    prechecks: tuple[VariationalCandidatePrecheck, ...]
    policy_digest: str
    ranking_objective: CandidateRankingObjective
    candidate_count: int
    scoring_operations: int
    whitener_operations_per_apply: int
    observed_whitener_apply_count: int
    contract: str = "p1-variational-candidate-ranking-v2"
    ranking_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-variational-candidate-ranking-v2":
            raise ValueError("unsupported candidate ranking")
        if tuple(score.rank for score in self.scores) != tuple(
            range(1, len(self.scores) + 1)
        ):
            raise ValueError("candidate ranks must be contiguous")
        identifiers = tuple(score.candidate_id for score in self.scores)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate identifiers must be unique")
        if self.candidate_count != len(self.prechecks):
            raise ValueError("candidate count and prechecks disagree")
        precheck_by_id = {value.candidate_id: value for value in self.prechecks}
        if len(precheck_by_id) != len(self.prechecks):
            raise ValueError("candidate precheck identifiers must be unique")
        if any(
            score.candidate_id not in precheck_by_id
            or not precheck_by_id[score.candidate_id].admissible
            or precheck_by_id[score.candidate_id].perturbation_digest
            != score.perturbation.digest
            for score in self.scores
        ):
            raise ValueError("ranked candidates must pass their bound precheck")
        if type(self.scoring_operations) is not int or self.scoring_operations < 0:
            raise ValueError("candidate scoring operations must be nonnegative")
        if self.whitener_operations_per_apply < 0 or (
            self.observed_whitener_apply_count < 0
        ):
            raise ValueError("candidate whitener telemetry must be nonnegative")
        _require_sha256("candidate ranking policy digest", self.policy_digest)
        object.__setattr__(
            self,
            "ranking_digest",
            _variational_candidate_ranking_digest(self),
        )


def _variational_candidate_ranking_digest(
    ranking: VariationalCandidateRanking,
) -> str:
    return json_digest(
        {
            "contract": ranking.contract,
            "fso_digest": ranking.fso.variational_fso_digest,
            "policy_digest": ranking.policy_digest,
            "ranking_objective": ranking.ranking_objective,
            "candidate_count": ranking.candidate_count,
            "scoring_operations": ranking.scoring_operations,
            "whitener_operations_per_apply": (
                ranking.whitener_operations_per_apply
            ),
            "observed_whitener_apply_count": (
                ranking.observed_whitener_apply_count
            ),
            "prechecks": [dataclass_digest(value) for value in ranking.prechecks],
            "scores": [
                {
                    "candidate_id": score.candidate_id,
                    "perturbation_digest": _sparse_radar_perturbation_digest(
                        score.perturbation
                    ),
                    "prediction": tensor_digest(
                        score.predicted_metric_change
                    ),
                    "score": score.score,
                    "rank": score.rank,
                }
                for score in ranking.scores
            ],
        }
    )


@dataclass(frozen=True)
class RankedLearningOutcome:
    """One selected result with its complete candidate-ranking lineage."""

    candidate_id: str
    candidate_rank: int
    candidate_score: float
    ranking_digest: str
    result: VariationalLearningImpact

    def __post_init__(self) -> None:
        if not self.candidate_id or self.candidate_rank <= 0:
            raise ValueError("ranked learning outcome identity is invalid")
        if not math.isfinite(self.candidate_score) or self.candidate_score < 0.0:
            raise ValueError("ranked learning outcome score is invalid")
        _require_sha256("ranking_digest", self.ranking_digest)
        evidence = self.result.approval_evidence
        if evidence is not None and (
            evidence.selection_mode != "ranked_top_k"
            or evidence.candidate_id != self.candidate_id
            or evidence.candidate_rank != self.candidate_rank
            or evidence.candidate_score != self.candidate_score
            or evidence.ranking_digest != self.ranking_digest
        ):
            raise ValueError("ranked outcome and approval evidence disagree")


@dataclass(frozen=True)
class LearningEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    policy_digest: str

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise TypeError("learning eligibility must be Boolean")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason
            for reason in self.reasons
        ):
            raise ValueError("learning rejection reasons must be strings")
        if self.eligible == bool(self.reasons):
            raise ValueError("learning eligibility and reasons disagree")
        _require_sha256("learning policy digest", self.policy_digest)


@dataclass(frozen=True)
class FirstOrderValidation:
    """Full/half-step checks on one frozen-domain Taylor prediction."""

    source_fsoi_digest: str
    nominal_forecast_digest: str
    nominal_input_bundle_digest: str
    nominal_full_analysis_input_digest: str
    full_step_prediction: Tensor
    full_step_resolved_metric_change: Tensor
    full_step_absolute_error: Tensor
    half_step_prediction: Tensor
    half_step_resolved_metric_change: Tensor
    half_step_absolute_error: Tensor
    metric_available: Tensor
    full_step_resolved_analysis_converged: bool
    half_step_resolved_analysis_converged: bool
    active_branch_valid: bool
    full_step_valid: bool
    half_step_valid: bool
    sign_consistent_for_material_impacts: bool
    material_metric_count: int
    maximum_material_impact: float
    aggregate_material_impact_norm: float
    first_order_valid: bool
    full_step_analysis_digest: str | None
    half_step_analysis_digest: str | None
    full_step_forecast_digest: str | None
    half_step_forecast_digest: str | None
    full_step_input_bundle_digest: str | None = None
    half_step_input_bundle_digest: str | None = None
    full_step_pcg_iterations: int = 0
    half_step_pcg_iterations: int = 0
    observed_whitener_apply_count: int = 0
    frozen_domain_state_effect: Tensor | None = None
    issuance_policy_effect: Tensor | None = None
    end_to_end_issuance_effect: Tensor | None = None
    coverage_before: Tensor | None = None
    coverage_after: Tensor | None = None
    newly_issued_fraction: Tensor | None = None
    withdrawn_fraction: Tensor | None = None
    background_fallback_before: Tensor | None = None
    background_fallback_after: Tensor | None = None
    metric_domain_contract: FirstOrderMetricDomain = "frozen_metric_domain"
    contract: str = "p1-first-order-validation-v5"
    validation_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-first-order-validation-v5":
            raise ValueError("unsupported first-order validation contract")
        if self.metric_domain_contract not in (
            "frozen_metric_domain",
            "resolved_issuance_domain",
        ):
            raise ValueError("unsupported first-order metric domain")
        for name in (
            "source_fsoi_digest",
            "nominal_forecast_digest",
            "nominal_input_bundle_digest",
            "nominal_full_analysis_input_digest",
        ):
            _require_sha256(name, getattr(self, name))
        tensor_names = (
            "full_step_prediction",
            "full_step_resolved_metric_change",
            "full_step_absolute_error",
            "half_step_prediction",
            "half_step_resolved_metric_change",
            "half_step_absolute_error",
            "metric_available",
        )
        for name in tensor_names:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            object.__setattr__(self, name, value.detach().clone())
        issuance_names = (
            "frozen_domain_state_effect",
            "issuance_policy_effect",
            "end_to_end_issuance_effect",
            "coverage_before",
            "coverage_after",
            "newly_issued_fraction",
            "withdrawn_fraction",
            "background_fallback_before",
            "background_fallback_after",
        )
        issuance = tuple(getattr(self, name) for name in issuance_names)
        if self.metric_domain_contract == "resolved_issuance_domain":
            if any(value is None for value in issuance):
                raise ValueError("resolved issuance decomposition is incomplete")
            for name, value in zip(issuance_names, issuance, strict=True):
                assert value is not None
                object.__setattr__(self, name, value.detach().clone())
            state = self.frozen_domain_state_effect
            policy = self.issuance_policy_effect
            total = self.end_to_end_issuance_effect
            assert state is not None and policy is not None and total is not None
            if state.shape != self.full_step_prediction.shape or (
                policy.shape != state.shape or total.shape != state.shape
            ):
                raise ValueError("resolved issuance metric shapes disagree")
            if not torch.allclose(
                state + policy, total, rtol=0.0, atol=0.0, equal_nan=True
            ):
                raise ValueError("resolved issuance effects do not close")
            if not torch.allclose(
                total,
                self.full_step_resolved_metric_change,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise ValueError("end-to-end effect and resolved change disagree")
            for name in issuance_names[3:]:
                value = getattr(self, name)
                assert value is not None
                if value.shape != (self.full_step_prediction.shape[0],) or bool(
                    torch.any((value < 0.0) | (value > 1.0))
                ):
                    raise ValueError("resolved issuance coverage is invalid")
        elif any(value is not None for value in issuance):
            raise ValueError("frozen-domain validation cannot carry issuance effects")
        if type(self.material_metric_count) is not int or (
            self.material_metric_count < 0
        ):
            raise ValueError("material_metric_count must be nonnegative")
        for name in (
            "maximum_material_impact",
            "aggregate_material_impact_norm",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.material_metric_count == 0 and (
            self.maximum_material_impact != 0.0
            or self.aggregate_material_impact_norm != 0.0
        ):
            raise ValueError("empty material signal must have zero magnitude")
        if (
            self.aggregate_material_impact_norm
            < self.maximum_material_impact
        ):
            raise ValueError("material impact norm cannot be below its maximum")
        if self.first_order_valid and self.material_metric_count == 0:
            raise ValueError("first-order validity requires material impact")
        for name in ("full_step_pcg_iterations", "half_step_pcg_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            type(self.observed_whitener_apply_count) is not int
            or self.observed_whitener_apply_count < 0
        ):
            raise ValueError("observed whitener apply count must be nonnegative")
        for name in (
            "full_step_analysis_digest",
            "half_step_analysis_digest",
            "full_step_forecast_digest",
            "half_step_forecast_digest",
            "full_step_input_bundle_digest",
            "half_step_input_bundle_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(name, value)
        input_digests = (
            self.full_step_input_bundle_digest,
            self.half_step_input_bundle_digest,
        )
        if self.metric_domain_contract == "resolved_issuance_domain":
            if any(value is None for value in input_digests):
                raise ValueError("resolved issuance input lineage is incomplete")
        elif any(value is not None for value in input_digests):
            raise ValueError("frozen validation cannot carry resolved inputs")
        object.__setattr__(
            self,
            "validation_digest",
            first_order_validation_digest(self),
        )

    @property
    def total_resolved_pcg_iterations(self) -> int:
        return self.full_step_pcg_iterations + self.half_step_pcg_iterations


@dataclass(frozen=True)
class LearningApprovalEvidence:
    """Immutable identities that justify one eligible learning impact."""

    policy_digest: str
    trust_store_digest: str
    fsoi_digest: str
    full_step_analysis_digest: str
    half_step_analysis_digest: str
    full_step_forecast_digest: str
    half_step_forecast_digest: str
    first_order_validation_digest: str
    learning_impact_digest: str
    approved_action_digest: str | None = None
    nominal_input_bundle_digest: str | None = None
    nominal_full_analysis_input_digest: str | None = None
    selection_mode: LearningSelectionMode = "direct"
    candidate_id: str | None = None
    candidate_rank: int | None = None
    candidate_score: float | None = None
    candidate_perturbation_digest: str | None = None
    ranking_digest: str | None = None
    ranking_policy_digest: str | None = None
    ranking_objective: CandidateRankingObjective | None = None
    whitener_operations_per_apply: int = 0
    observed_whitener_apply_count: int = 0
    observed_whitener_total_operations: int = 0
    contract: str = "p1-learning-approval-evidence-v4"

    def __post_init__(self) -> None:
        if self.contract not in (
            "p1-learning-approval-evidence-v1",
            "p1-learning-approval-evidence-v2",
            "p1-learning-approval-evidence-v3",
            "p1-learning-approval-evidence-v4",
        ):
            raise ValueError("unsupported learning approval evidence")
        for name, value in (
            ("policy_digest", self.policy_digest),
            ("trust_store_digest", self.trust_store_digest),
            ("fsoi_digest", self.fsoi_digest),
            ("full_step_analysis_digest", self.full_step_analysis_digest),
            ("half_step_analysis_digest", self.half_step_analysis_digest),
            ("full_step_forecast_digest", self.full_step_forecast_digest),
            ("half_step_forecast_digest", self.half_step_forecast_digest),
            (
                "first_order_validation_digest",
                self.first_order_validation_digest,
            ),
            ("learning_impact_digest", self.learning_impact_digest),
        ):
            _require_sha256(name, value)
        action_values = (
            self.approved_action_digest,
            self.nominal_input_bundle_digest,
            self.nominal_full_analysis_input_digest,
        )
        if self.contract in (
            "p1-learning-approval-evidence-v3",
            "p1-learning-approval-evidence-v4",
        ):
            if self.contract == "p1-learning-approval-evidence-v3":
                action_values = action_values[:2]
            if any(value is None for value in action_values):
                raise ValueError("learning action lineage is incomplete")
            _require_sha256(
                "approved_action_digest",
                cast(str, self.approved_action_digest),
            )
            _require_sha256(
                "nominal_input_bundle_digest",
                cast(str, self.nominal_input_bundle_digest),
            )
            if self.contract == "p1-learning-approval-evidence-v4":
                _require_sha256(
                    "nominal_full_analysis_input_digest",
                    cast(str, self.nominal_full_analysis_input_digest),
                )
        elif any(value is not None for value in action_values):
            raise ValueError("legacy learning evidence cannot carry action lineage")
        ranked_values = (
            self.candidate_id,
            self.candidate_rank,
            self.candidate_score,
            self.candidate_perturbation_digest,
            self.ranking_digest,
            self.ranking_policy_digest,
            self.ranking_objective,
        )
        if self.contract == "p1-learning-approval-evidence-v1":
            if self.selection_mode != "direct" or any(
                value is not None for value in ranked_values
            ):
                raise ValueError("legacy learning evidence cannot carry ranking")
            if any(
                (
                    self.whitener_operations_per_apply,
                    self.observed_whitener_apply_count,
                    self.observed_whitener_total_operations,
                )
            ):
                raise ValueError("legacy learning evidence cannot carry telemetry")
        elif self.selection_mode == "direct":
            if any(value is not None for value in ranked_values):
                raise ValueError("direct learning evidence cannot carry ranking")
        elif self.selection_mode == "ranked_top_k":
            if any(value is None for value in ranked_values):
                raise ValueError("ranked learning evidence is incomplete")
            if not isinstance(self.candidate_id, str) or not self.candidate_id:
                raise ValueError("ranked candidate_id must be nonempty")
            if type(self.candidate_rank) is not int or self.candidate_rank <= 0:
                raise ValueError("ranked candidate_rank must be positive")
            if (
                isinstance(self.candidate_score, bool)
                or not math.isfinite(cast(float, self.candidate_score))
                or cast(float, self.candidate_score) < 0.0
            ):
                raise ValueError("ranked candidate_score must be nonnegative")
            _require_sha256(
                "candidate_perturbation_digest",
                cast(str, self.candidate_perturbation_digest),
            )
            _require_sha256("ranking_digest", cast(str, self.ranking_digest))
            _require_sha256(
                "ranking_policy_digest",
                cast(str, self.ranking_policy_digest),
            )
        else:
            raise ValueError("unsupported learning selection mode")
        for name in (
            "whitener_operations_per_apply",
            "observed_whitener_apply_count",
            "observed_whitener_total_operations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.observed_whitener_total_operations != (
            self.whitener_operations_per_apply
            * self.observed_whitener_apply_count
        ):
            raise ValueError("learning whitener operation accounting mismatch")

    @property
    def digest(self) -> str:
        if self.contract == "p1-learning-approval-evidence-v1":
            return json_digest(
                {
                    "policy_digest": self.policy_digest,
                    "trust_store_digest": self.trust_store_digest,
                    "fsoi_digest": self.fsoi_digest,
                    "full_step_analysis_digest": (
                        self.full_step_analysis_digest
                    ),
                    "half_step_analysis_digest": (
                        self.half_step_analysis_digest
                    ),
                    "full_step_forecast_digest": (
                        self.full_step_forecast_digest
                    ),
                    "half_step_forecast_digest": (
                        self.half_step_forecast_digest
                    ),
                    "first_order_validation_digest": (
                        self.first_order_validation_digest
                    ),
                    "learning_impact_digest": self.learning_impact_digest,
                    "contract": self.contract,
                }
            )
        return dataclass_digest(self)


@dataclass(frozen=True)
class VariationalLearningImpact:
    eligibility: LearningEligibility
    fsoi: VariationalFSOI | None
    first_order_validation: FirstOrderValidation | None
    frozen_domain_learning_impact: VariationalImpactChannel | None
    approval_evidence: LearningApprovalEvidence | None = None
    contract: str = "p1-variational-learning-impact-v2"
    learning_result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-variational-learning-impact-v2":
            raise ValueError("unsupported variational learning impact")
        if self.frozen_domain_learning_impact is not None:
            object.__setattr__(
                self,
                "frozen_domain_learning_impact",
                _clone_variational_impact_channel(
                    self.frozen_domain_learning_impact
                ),
            )
        complete = (
            self.fsoi is not None
            and self.first_order_validation is not None
            and self.first_order_validation.first_order_valid
            and self.frozen_domain_learning_impact is not None
            and self.approval_evidence is not None
        )
        if self.eligibility.eligible != complete:
            raise ValueError("learning eligibility and impact disagree")
        if self.first_order_validation is not None and self.fsoi is None:
            raise ValueError("first-order validation requires FSOI")
        if not self.eligibility.eligible and self.approval_evidence is not None:
            raise ValueError("rejected learning impact cannot carry approval")
        object.__setattr__(
            self,
            "learning_result_digest",
            variational_learning_impact_digest(self),
        )
        validate_variational_learning_impact(self)


@dataclass
class _VariationalChannelAccumulator:
    maps: Tensor
    by_time: Tensor
    tile_by_time: Tensor


@dataclass(frozen=True)
class _VariationalAdjointSensitivity:
    detected_dbz: Tensor
    censor_threshold_dbz: Tensor
    observation_weight: Tensor


@dataclass(frozen=True)
class _VariationalAdjointSolve:
    sensitivity: _VariationalAdjointSensitivity
    solution: Tensor
    iterations: int
    relative_residual: float
    true_residual_norm: float
    normal_products: int
    warm_started: bool


@dataclass(frozen=True)
class _FrozenBaselineDynamicsPath:
    """Reusable VJP for observation-derived baseline motion and growth."""

    active_mask: Tensor
    nominal_dynamics: Tensor
    observation_pullback: Callable[[Tensor], tuple[Tensor]]


@dataclass(frozen=True)
class P0TendencyBranchSignature:
    pair_spans: tuple[tuple[int, int], ...]
    motion_pair_spans: tuple[tuple[int, int], ...]
    growth_pair_spans: tuple[tuple[int, int], ...]
    integer_peak_yx_by_pair: tuple[tuple[int, int], ...]
    peak_is_search_interior_by_pair: tuple[bool, ...]
    pair_available_by_span: tuple[bool, ...]
    growth_evidence_available_by_span: tuple[bool, ...]
    motion_remap_cells: tuple[RemapCell, RemapCell]
    motion_selection: TendencyPairSelection
    growth_selection: TendencyPairSelection
    motion_conflict: bool
    growth_conflict: bool


@dataclass
class _NormalProductBudget:
    maximum: int
    used: int = 0

    def apply(
        self,
        operator: Callable[[Tensor], Tensor],
        value: Tensor,
    ) -> Tensor:
        if self.used >= self.maximum:
            raise ValueError("P1 FSO normal-product budget exhausted")
        self.used += 1
        return operator(value)


def _metric_tile_shape(
    config: SensitivityConfig,
    grid: RadarGridTimeContract | None,
) -> TileShape:
    if config.tile_size_m is None:
        return config.tile_size, config.tile_size
    if grid is None:
        raise ValueError("physical sensitivity settings require a grid contract")
    assert grid.pixel_to_projected_matrix_m is not None
    (a, b), (c, d) = grid.pixel_to_projected_matrix_m
    row_spacing = math.hypot(b, d)
    column_spacing = math.hypot(a, c)
    return (
        max(1, math.floor(config.tile_size_m / row_spacing + 0.5)),
        max(1, math.floor(config.tile_size_m / column_spacing + 0.5)),
    )


def _perturbation_tile_size(
    config: VariationalAdjointConfig,
    grid: RadarGridTimeContract | None,
) -> TileShape:
    if config.perturbation_tile_size_m is None:
        return config.perturbation_tile_size, config.perturbation_tile_size
    if grid is None:
        raise ValueError("physical perturbation tiles require a grid contract")
    assert grid.pixel_to_projected_matrix_m is not None
    (a, b), (c, d) = grid.pixel_to_projected_matrix_m
    return (
        max(
            1,
            math.floor(
                config.perturbation_tile_size_m / math.hypot(b, d) + 0.5
            ),
        ),
        max(
            1,
            math.floor(
                config.perturbation_tile_size_m / math.hypot(a, c) + 0.5
            ),
        ),
    )


def _metric_domain_weight(
    result: ForecastResult,
    verification_finite: Tensor,
    forecast_index: int,
    domain: FSOMetricDomain,
) -> Tensor:
    """Freeze the spatial domain/weight used by one forecast metric."""

    if verification_finite.dtype is not torch.bool:
        raise TypeError("verification_finite must be Boolean")
    if verification_finite.shape != result.valid_mask[forecast_index].shape:
        raise ValueError("verification domain must match one forecast lead")
    if domain == "radar_dynamics_anchored":
        eligible = result.radar_dynamics_anchored_valid_mask[forecast_index]
        weight = eligible.to(result.state.echo_linear)
    elif domain == "confidence_weighted":
        eligible = result.valid_mask[forecast_index]
        weight = result.forecast_confidence[forecast_index]
    elif domain == "issued":
        eligible = result.valid_mask[forecast_index]
        weight = eligible.to(result.state.echo_linear)
    else:
        raise ValueError("unsupported FSO metric domain")
    return torch.where(
        verification_finite & eligible,
        weight,
        torch.zeros_like(weight),
    ).detach()


def _metric_domain_digest(
    domain: FSOMetricDomain,
    lead_minutes: tuple[int, ...],
    weights: tuple[Tensor, ...],
) -> str:
    return json_digest(
        {
            "version": "p1-fso-metric-domain-v1",
            "domain": domain,
            "lead_minutes": list(lead_minutes),
            "weight_digests": [tensor_digest(weight) for weight in weights],
        }
    )


def _deterministic_unit_probe(
    reference: Tensor,
    *,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bits = torch.randint(
        0,
        2,
        reference.shape,
        generator=generator,
        dtype=torch.int8,
        device="cpu",
    )
    probe = (2 * bits - 1).to(
        dtype=reference.dtype,
        device=reference.device,
    )
    return probe / torch.linalg.vector_norm(probe)


def _gauss_newton_curvature_diagnostics(
    control: Tensor,
    residual_fn: Callable[[Tensor], Tensor],
    normal_product: Callable[[Tensor], Tensor],
    budget: _NormalProductBudget,
    config: VariationalAdjointConfig,
) -> VariationalGaussNewtonDiagnostics:
    """Compare exact frozen-objective and GN curvature on fixed probes."""

    def objective(value: Tensor) -> Tensor:
        residual = residual_fn(value)
        return 0.5 * torch.dot(residual, residual)

    objective_gradient = torch.func.grad(objective)
    defects = control.new_empty(config.gauss_newton_probe_count)
    products_before = budget.used
    for index in range(config.gauss_newton_probe_count):
        probe = _deterministic_unit_probe(
            control,
            seed=config.gauss_newton_probe_seed + index,
        )
        exact_product = cast(
            Tensor,
            torch.func.jvp(
                objective_gradient,
                (control,),
                (probe,),
            )[1],
        )
        gn_product = budget.apply(normal_product, probe)
        if not bool(
            torch.all(torch.isfinite(exact_product))
            & torch.all(torch.isfinite(gn_product))
        ):
            raise ValueError("P1 frozen curvature probe is not finite")
        denominator = torch.linalg.vector_norm(gn_product)
        defects[index] = (
            torch.linalg.vector_norm(exact_product - gn_product)
            / denominator
        )
    maximum = float(torch.amax(defects).detach())
    reliable = (
        math.isfinite(maximum)
        and maximum
        <= config.maximum_gauss_newton_relative_curvature_defect
    )
    diagnostics = VariationalGaussNewtonDiagnostics(
        relative_curvature_defect=defects.detach(),
        maximum_relative_curvature_defect=maximum,
        reliable=reliable,
        normal_products=budget.used - products_before,
        exact_hessian_products=config.gauss_newton_probe_count,
    )
    if config.require_gauss_newton_reliability and not reliable:
        raise ValueError("P1 Gauss-Newton curvature approximation is unreliable")
    return diagnostics


def _adjoint_lead_indices(
    config: VariationalAdjointConfig,
    all_lead_minutes: tuple[int, ...],
) -> tuple[int, ...]:
    if config.lead_minutes is None:
        return tuple(range(len(all_lead_minutes)))
    positions = {minutes: index for index, minutes in enumerate(all_lead_minutes)}
    missing = set(config.lead_minutes) - set(all_lead_minutes)
    if missing:
        raise ValueError(f"adjoint leads are outside the forecast: {sorted(missing)}")
    return tuple(positions[minutes] for minutes in config.lead_minutes)


def _prior_smoothness_diagonal_preconditioner(
    control: Tensor,
    frozen: FrozenOuterState,
) -> Callable[[Tensor], Tensor]:
    diagonal = torch.ones_like(control)
    field_size = frozen.active_field_index.numel()
    if field_size > 0 and frozen.smooth_edge_left_index.numel() > 0:
        edge_weight = (
            frozen.analysis_config.field_smoothness_weight
            * frozen.smooth_edge_physical_weight
        ).to(dtype=control.dtype, device=control.device)
        field_diagonal = diagonal[:field_size]
        field_diagonal.scatter_add_(
            0,
            frozen.smooth_edge_left_index,
            edge_weight,
        )
        field_diagonal.scatter_add_(
            0,
            frozen.smooth_edge_right_index,
            edge_weight,
        )

    def apply(value: Tensor) -> Tensor:
        return value / diagonal

    return apply


def _variational_preconditioner(
    control: Tensor,
    frozen: FrozenOuterState,
    config: VariationalAdjointConfig,
) -> Callable[[Tensor], Tensor] | None:
    if config.preconditioner == "none":
        return None
    return _prior_smoothness_diagonal_preconditioner(control, frozen)


def _variational_materialized_output_bytes(
    reference: Tensor,
    *,
    selected_count: int,
    lead_count: int,
    metric_count: int,
    height: int,
    width: int,
    tile_rows: int,
    tile_columns: int,
    include_impact: bool,
    gauss_newton_probe_count: int,
) -> int:
    channel_elements = (
        selected_count * metric_count * 3 * height * width
        + lead_count * metric_count * 3
        + lead_count * metric_count * 3 * tile_rows * tile_columns
    )
    # Six sensitivity channels are always materialized. Explicit FSOI adds
    # six signed impact channels (five parameters plus their total).
    channel_count = 12 if include_impact else 6
    float_elements = (
        channel_count * channel_elements
        + 3 * lead_count * metric_count
        + 2 * lead_count
        + gauss_newton_probe_count
    )
    bool_elements = (
        selected_count * height * width
        + 2 * lead_count * metric_count
    )
    int64_elements = 2 * lead_count * metric_count
    return (
        float_elements * reference.element_size()
        + bool_elements
        + int64_elements * 8
    )


def _minimum_masked_value(values: Tensor, mask: Tensor) -> float | None:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        return None
    result = float(torch.amin(selected).detach())
    return result if math.isfinite(result) else None


def _remap_fraction_margin(
    displacement_yx: Tensor,
    cell: RemapCell,
) -> float:
    cell_tensor = displacement_yx.new_tensor((cell.y, cell.x))
    fraction = displacement_yx - cell_tensor
    margin = torch.minimum(fraction, 1.0 - fraction)
    return max(0.0, float(torch.amin(margin).detach()))


def _analysis_remap_margin(
    displacement_yx: Tensor,
    cells: tuple[RemapCell, RemapCell],
) -> float:
    return min(
        _remap_fraction_margin((index + 1) * displacement_yx, cell)
        for index, cell in enumerate(cells)
    )


def _publication_margins(
    result: ForecastResult,
    forecast_indices: tuple[int, ...],
) -> tuple[float, float | None]:
    config = result.run.config
    support_margins = [
        torch.abs(
            result.forecast_source_support[list(forecast_indices)]
            - config.min_publish_support
        )
    ]
    if config.minimum_publish_verified_support is not None:
        support_margins.append(
            torch.abs(
                result.forecast_verified_support[list(forecast_indices)]
                - config.minimum_publish_verified_support
            )
        )
    if config.minimum_publish_observation_verified_support is not None:
        support_margins.append(
            torch.abs(
                result.forecast_observation_verified_support[
                    list(forecast_indices)
                ]
                - config.minimum_publish_observation_verified_support
            )
        )
    support_margin = min(
        float(torch.amin(values).detach()) for values in support_margins
    )
    if config.maximum_publish_background_fraction is not None:
        support_margin = min(
            support_margin,
            abs(
                result.metadata.background_contribution_fraction
                - config.maximum_publish_background_fraction
            ),
        )
    confidence_margin = None
    if config.minimum_publish_confidence is not None:
        confidence_margin = float(
            torch.amin(
                torch.abs(
                    result.forecast_confidence[list(forecast_indices)]
                    - config.minimum_publish_confidence
                )
            ).detach()
        )
    return support_margin, confidence_margin


def _variational_channel_digest_values(
    channel: VariationalSensitivityChannel,
) -> dict[str, str]:
    return {
        "maps": tensor_digest(channel.maps),
        "norm_by_time": tensor_digest(channel.norm_by_time),
        "tile_norm_by_time": tensor_digest(channel.tile_norm_by_time),
    }


def _variational_impact_digest_values(
    channel: VariationalImpactChannel,
) -> dict[str, str]:
    return {
        "maps": tensor_digest(channel.maps),
        "sum_by_time": tensor_digest(channel.sum_by_time),
        "tile_sum_by_time": tensor_digest(channel.tile_sum_by_time),
    }


def _clone_variational_impact_channel(
    channel: VariationalImpactChannel,
) -> VariationalImpactChannel:
    return VariationalImpactChannel(
        maps=channel.maps.detach().clone(),
        sum_by_time=channel.sum_by_time.detach().clone(),
        tile_sum_by_time=channel.tile_sum_by_time.detach().clone(),
    )


def _variational_impact_digest(channel: VariationalImpactChannel) -> str:
    return json_digest(
        {
            "contract": "p1-frozen-domain-learning-impact-v1",
            **_variational_impact_digest_values(channel),
        }
    )


def first_order_validation_digest(
    validation: FirstOrderValidation,
) -> str:
    """Content digest for full/half-step Taylor validation evidence."""

    return json_digest(
        {
            "contract": validation.contract,
            "source_fsoi_digest": validation.source_fsoi_digest,
            "nominal_forecast_digest": validation.nominal_forecast_digest,
            "nominal_input_bundle_digest": (
                validation.nominal_input_bundle_digest
            ),
            "nominal_full_analysis_input_digest": (
                validation.nominal_full_analysis_input_digest
            ),
            "metric_domain_contract": validation.metric_domain_contract,
            "full_step_prediction": tensor_digest(
                validation.full_step_prediction
            ),
            "full_step_resolved_metric_change": tensor_digest(
                validation.full_step_resolved_metric_change
            ),
            "full_step_absolute_error": tensor_digest(
                validation.full_step_absolute_error
            ),
            "half_step_prediction": tensor_digest(
                validation.half_step_prediction
            ),
            "half_step_resolved_metric_change": tensor_digest(
                validation.half_step_resolved_metric_change
            ),
            "half_step_absolute_error": tensor_digest(
                validation.half_step_absolute_error
            ),
            "metric_available": tensor_digest(validation.metric_available),
            "full_step_resolved_analysis_converged": (
                validation.full_step_resolved_analysis_converged
            ),
            "half_step_resolved_analysis_converged": (
                validation.half_step_resolved_analysis_converged
            ),
            "active_branch_valid": validation.active_branch_valid,
            "full_step_valid": validation.full_step_valid,
            "half_step_valid": validation.half_step_valid,
            "sign_consistent_for_material_impacts": (
                validation.sign_consistent_for_material_impacts
            ),
            "material_metric_count": validation.material_metric_count,
            "maximum_material_impact": validation.maximum_material_impact,
            "aggregate_material_impact_norm": (
                validation.aggregate_material_impact_norm
            ),
            "first_order_valid": validation.first_order_valid,
            "full_step_analysis_digest": (
                validation.full_step_analysis_digest
            ),
            "half_step_analysis_digest": (
                validation.half_step_analysis_digest
            ),
            "full_step_forecast_digest": (
                validation.full_step_forecast_digest
            ),
            "half_step_forecast_digest": (
                validation.half_step_forecast_digest
            ),
            "full_step_input_bundle_digest": (
                validation.full_step_input_bundle_digest
            ),
            "half_step_input_bundle_digest": (
                validation.half_step_input_bundle_digest
            ),
            "full_step_pcg_iterations": validation.full_step_pcg_iterations,
            "half_step_pcg_iterations": validation.half_step_pcg_iterations,
            "observed_whitener_apply_count": (
                validation.observed_whitener_apply_count
            ),
            "frozen_domain_state_effect": _optional_tensor_digest(
                validation.frozen_domain_state_effect
            ),
            "issuance_policy_effect": _optional_tensor_digest(
                validation.issuance_policy_effect
            ),
            "end_to_end_issuance_effect": _optional_tensor_digest(
                validation.end_to_end_issuance_effect
            ),
            "coverage_before": _optional_tensor_digest(
                validation.coverage_before
            ),
            "coverage_after": _optional_tensor_digest(validation.coverage_after),
            "newly_issued_fraction": _optional_tensor_digest(
                validation.newly_issued_fraction
            ),
            "withdrawn_fraction": _optional_tensor_digest(
                validation.withdrawn_fraction
            ),
            "background_fallback_before": _optional_tensor_digest(
                validation.background_fallback_before
            ),
            "background_fallback_after": _optional_tensor_digest(
                validation.background_fallback_after
            ),
        }
    )


def _optional_tensor_digest(value: Tensor | None) -> str | None:
    return None if value is None else tensor_digest(value)


def variational_learning_impact_digest(
    learning: VariationalLearningImpact,
) -> str:
    """Content digest for the final eligible or rejected learning result."""

    impact = learning.frozen_domain_learning_impact
    return json_digest(
        {
            "contract": learning.contract,
            "eligibility": {
                "eligible": learning.eligibility.eligible,
                "reasons": list(learning.eligibility.reasons),
                "policy_digest": learning.eligibility.policy_digest,
            },
            "fsoi_digest": (
                None
                if learning.fsoi is None
                else learning.fsoi.variational_fsoi_digest
            ),
            "first_order_validation_digest": (
                None
                if learning.first_order_validation is None
                else learning.first_order_validation.validation_digest
            ),
            "frozen_domain_learning_impact_digest": (
                None if impact is None else _variational_impact_digest(impact)
            ),
            "approval_evidence_digest": (
                None
                if learning.approval_evidence is None
                else learning.approval_evidence.digest
            ),
        }
    )


@dataclass(frozen=True)
class _ResolvedVerification:
    frames_dbz: Tensor
    valid_mask: Tensor
    metric_weight: Tensor
    contract: str
    content_digest: str
    lineage_complete: bool
    valid_times: tuple[str, ...] | None
    grid_contract_digest: str | None
    radar_product_digest: str | None
    qc_pipeline_digest: str | None


def _verification_content_digest(
    contract: str,
    frames_dbz: Tensor,
    valid_mask: Tensor,
    valid_times: tuple[str, ...] | None,
    grid_contract_digest: str | None,
    radar_product_digest: str | None,
    qc_pipeline_digest: str | None,
    mask_policy_digest: str | None = None,
    censor_policy_digest: str | None = None,
    reflectivity_resolution_dbz: float | None = None,
    quantization_origin_dbz: float | None = None,
    threshold_bin_convention: str | None = None,
    floor_representation_contract_digest: str | None = None,
    quality_weight: Tensor | None = None,
    observation_std_dbz: Tensor | None = None,
    observation_state_code: Tensor | None = None,
    source_radar_index_map: Tensor | None = None,
    observation_error_contract: VerificationObservationErrorContract | None = None,
) -> str:
    payload: dict[str, object] = {
        "version": "verification-bundle-content-v2",
        "contract": contract,
        "frames_dbz": tensor_digest(frames_dbz),
        "valid_mask": tensor_digest(valid_mask),
        "valid_times": None if valid_times is None else list(valid_times),
        "grid_contract_digest": grid_contract_digest,
        "radar_product_digest": radar_product_digest,
        "qc_pipeline_digest": qc_pipeline_digest,
    }
    if contract in {
        "radar-verification-bundle-v2",
        "radar-verification-bundle-v3",
        "radar-verification-bundle-v4",
        "radar-verification-bundle-v5",
        "radar-verification-bundle-v6",
    }:
        payload.update(
            {
                "version": (
                    "verification-bundle-content-v3"
                    if contract == "radar-verification-bundle-v2"
                    else (
                        "verification-bundle-content-v4"
                        if contract == "radar-verification-bundle-v3"
                        else (
                            "verification-bundle-content-v5"
                            if contract == "radar-verification-bundle-v4"
                            else (
                                "verification-bundle-content-v6"
                                if contract == "radar-verification-bundle-v5"
                                else "verification-bundle-content-v7"
                            )
                        )
                    )
                ),
                "mask_policy_digest": mask_policy_digest,
                "censor_policy_digest": censor_policy_digest,
                "reflectivity_resolution_dbz": reflectivity_resolution_dbz,
                "quantization_origin_dbz": quantization_origin_dbz,
                "threshold_bin_convention": threshold_bin_convention,
                "floor_representation_contract_digest": (
                    floor_representation_contract_digest
                ),
            }
        )
    if contract in {
        "radar-verification-bundle-v4",
        "radar-verification-bundle-v5",
        "radar-verification-bundle-v6",
    }:
        payload.update(
            {
                "quality_weight": (
                    None if quality_weight is None else tensor_digest(quality_weight)
                ),
                "observation_std_dbz": (
                    None
                    if observation_std_dbz is None
                    else tensor_digest(observation_std_dbz)
                ),
                "observation_error_contract": (
                    None
                    if observation_error_contract is None
                    else observation_error_contract.payload
                    | {"contract_digest": observation_error_contract.contract_digest}
                ),
            }
        )
    if contract == "radar-verification-bundle-v6":
        payload["observation_state_code"] = (
            None
            if observation_state_code is None
            else tensor_digest(observation_state_code)
        )
        payload["source_radar_index_map"] = (
            None
            if source_radar_index_map is None
            else tensor_digest(source_radar_index_map)
        )
    return json_digest(payload)


def _resolve_verification(
    verification: VerificationInput,
    result: ForecastResult,
    sensitivity_config: SensitivityConfig,
) -> _ResolvedVerification:
    if isinstance(verification, VerificationBundle):
        verification.validate_integrity()
        resolved = _ResolvedVerification(
            frames_dbz=verification.frames_dbz,
            valid_mask=verification.valid_mask,
            metric_weight=verification.metric_weight,
            contract=verification.contract,
            content_digest=verification.content_digest,
            lineage_complete=True,
            valid_times=verification.valid_times,
            grid_contract_digest=verification.grid_contract_digest,
            radar_product_digest=verification.radar_product_digest,
            qc_pipeline_digest=verification.qc_pipeline_digest,
        )
    else:
        valid = torch.isfinite(verification)
        contract = "legacy-verification-tensor-v1"
        resolved = _ResolvedVerification(
            frames_dbz=verification,
            valid_mask=valid,
            metric_weight=valid.to(verification),
            contract=contract,
            content_digest=_verification_content_digest(
                contract,
                verification,
                valid,
                None,
                None,
                None,
                None,
            ),
            lineage_complete=False,
            valid_times=None,
            grid_contract_digest=None,
            radar_product_digest=None,
            qc_pipeline_digest=None,
        )
    if sensitivity_config.require_verification_lineage and not (
        resolved.lineage_complete
    ):
        raise ValueError(
            "complete verification lineage requires VerificationBundle"
        )
    if not resolved.lineage_complete:
        return resolved
    grid = result.run.grid_time_contract
    if grid is None or result.run.grid_time_contract_digest is None:
        raise ValueError(
            "verification lineage requires a forecast grid/time contract"
        )
    if resolved.grid_contract_digest != result.run.grid_time_contract_digest:
        raise ValueError("verification and forecast grid contracts disagree")
    issue_time = datetime.fromisoformat(
        grid.valid_times[-1].replace("Z", "+00:00")
    )
    expected_times = tuple(
        (issue_time + timedelta(minutes=minutes))
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
        for minutes in range(
            result.run.config.interval_minutes,
            result.run.config.horizon_minutes + 1,
            result.run.config.interval_minutes,
        )
    )
    if resolved.valid_times != expected_times:
        raise ValueError(
            "verification valid times do not match forecast issue and leads"
        )
    required_product = (
        sensitivity_config.required_verification_radar_product_digest
    )
    required_qc = sensitivity_config.required_verification_qc_pipeline_digest
    if required_product is not None and (
        resolved.radar_product_digest != required_product
        or resolved.qc_pipeline_digest != required_qc
    ):
        raise ValueError("verification product or QC identity is not approved")
    return resolved


def _validate_verification_lineage_fields(
    *,
    contract: str,
    content_digest: str,
    lineage_complete: bool,
    valid_times: tuple[str, ...] | None,
    grid_contract_digest: str | None,
    radar_product_digest: str | None,
    qc_pipeline_digest: str | None,
) -> None:
    _require_sha256("verification_bundle_digest", content_digest)
    if type(lineage_complete) is not bool:
        raise TypeError("verification_lineage_complete must be Boolean")
    lineage_values = (
        grid_contract_digest,
        radar_product_digest,
        qc_pipeline_digest,
    )
    if not lineage_complete:
        if contract != "legacy-verification-tensor-v1":
            raise ValueError("incomplete verification must use legacy contract")
        if valid_times is not None or any(
            value is not None for value in lineage_values
        ):
            raise ValueError("incomplete verification cannot claim lineage")
        return
    if contract not in {
        "radar-verification-bundle-v1",
        "radar-verification-bundle-v2",
        "radar-verification-bundle-v3",
        "radar-verification-bundle-v4",
    }:
        raise ValueError("complete verification has the wrong contract")
    if valid_times is None or not valid_times:
        raise ValueError("complete verification requires valid times")
    canonical_times = tuple(
        _canonical_verification_time(value) for value in valid_times
    )
    parsed_times = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in canonical_times
    )
    if canonical_times != valid_times or any(
        later <= earlier
        for earlier, later in zip(parsed_times, parsed_times[1:])
    ):
        raise ValueError(
            "verification valid times must be canonical UTC and increasing"
        )
    for name, value in zip(
        (
            "verification_grid_contract_digest",
            "verification_radar_product_digest",
            "verification_qc_pipeline_digest",
        ),
        lineage_values,
    ):
        if value is None:
            raise ValueError(f"{name} is required")
        _require_sha256(name, value)


def _metric_contract_digest(config: SensitivityConfig) -> str:
    return json_digest(
        {
            "version": "p1-forecast-metric-contract-v4",
            "metric_names": list(config.metric_names),
            "metric_domain": config.metric_domain,
            "sensitivity_config_digest": config.digest,
        }
    )


def variational_fso_digest(fso: VariationalFSO) -> str:
    """Content digest for a complete frozen-model FSO result."""

    trusted_input = (
        fso.observation.baseline_branch_trusted_frozen_structure_input_dbz
    )
    return json_digest(
        {
            "version": "p1-variational-fso-digest-v14",
            "contract": fso.contract,
            "forecast_run_digest": fso.forecast_run_digest,
            "analysis_input_digest": fso.analysis_input_digest,
            "sensitivity_config_digest": fso.sensitivity_config_digest,
            "adjoint_config_digest": fso.adjoint_config_digest,
            "linearization_contract": fso.linearization_contract,
            "linearization_digest": fso.linearization_digest,
            "verification_contract": fso.verification_contract,
            "verification_bundle_digest": fso.verification_bundle_digest,
            "verification_lineage_complete": (
                fso.verification_lineage_complete
            ),
            "verification_valid_times": (
                None
                if fso.verification_valid_times is None
                else list(fso.verification_valid_times)
            ),
            "verification_grid_contract_digest": (
                fso.verification_grid_contract_digest
            ),
            "verification_radar_product_digest": (
                fso.verification_radar_product_digest
            ),
            "verification_qc_pipeline_digest": (
                fso.verification_qc_pipeline_digest
            ),
            "metric_contract_digest": fso.metric_contract_digest,
            "algorithm_bundle_digest": fso.algorithm_bundle_digest,
            "numerical_runtime_digest": fso.numerical_runtime_digest,
            "sensitivity_scope": fso.sensitivity_scope,
            "baseline_dynamics_frozen": fso.baseline_dynamics_frozen,
            "baseline_pair_selection_frozen": (
                fso.baseline_pair_selection_frozen
            ),
            "baseline_dynamics_branch_status": (
                fso.baseline_dynamics_branch_status
            ),
            "metric_names": list(fso.metric_names),
            "metric_domain": fso.metric_domain,
            "metric_domain_digest": fso.metric_domain_digest,
            "lead_minutes": list(fso.lead_minutes),
            "full_map_lead_minutes": list(fso.full_map_lead_minutes),
            "tile_size": fso.tile_size,
            "tile_shape_yx": list(fso.tile_shape_yx),
            "forecast_scores": tensor_digest(fso.forecast_scores),
            "metric_available": tensor_digest(fso.metric_available),
            "metric_domain_weight_sum": tensor_digest(
                fso.metric_domain_weight_sum
            ),
            "metric_domain_weight_fraction": tensor_digest(
                fso.metric_domain_weight_fraction
            ),
            "forecast_cap_active_mask": tensor_digest(
                fso.forecast_cap_active_mask
            ),
            "observation": {
                "detected_dbz": _variational_channel_digest_values(
                    fso.observation.detected_dbz
                ),
                "censor_threshold_dbz": (
                    _variational_channel_digest_values(
                        fso.observation.censor_threshold_dbz
                    )
                ),
                "observation_weight": _variational_channel_digest_values(
                    fso.observation.observation_weight
                ),
                "initial_background_dbz": (
                    _variational_channel_digest_values(
                        fso.observation.initial_background_dbz
                    )
                ),
                "baseline_dynamics_dbz": (
                    _variational_channel_digest_values(
                        fso.observation.baseline_dynamics_dbz
                    )
                ),
                "frozen_structure_input_dbz": (
                    _variational_channel_digest_values(
                        fso.observation.frozen_structure_input_dbz
                    )
                ),
                "baseline_branch_trusted_frozen_structure_input_dbz": (
                    None
                    if trusted_input is None
                    else _variational_channel_digest_values(trusted_input)
                ),
            },
            "adjoint_iterations": tensor_digest(fso.adjoint_iterations),
            "adjoint_relative_residual": tensor_digest(
                fso.adjoint_relative_residual
            ),
            "adjoint_true_residual_norm": tensor_digest(
                fso.adjoint_true_residual_norm
            ),
            "adjoint_normal_products": tensor_digest(
                fso.adjoint_normal_products
            ),
            "adjoint_warm_started": tensor_digest(
                fso.adjoint_warm_started
            ),
            "total_normal_products": fso.total_normal_products,
            "whitener_operations_per_apply": (
                fso.whitener_operations_per_apply
            ),
            "observed_whitener_apply_count": (
                fso.observed_whitener_apply_count
            ),
            "materialized_output_bytes": fso.materialized_output_bytes,
            "neural_prior_adjoint_direction_maximum_defect": (
                fso.neural_prior_adjoint_direction_maximum_defect
            ),
            "active_set_margins": {
                "detection_classification_dbz": (
                    fso.active_set_margins.detection_classification_dbz
                ),
                "analysis_remap_fraction": (
                    fso.active_set_margins.analysis_remap_fraction
                ),
                "forecast_remap_fraction": (
                    fso.active_set_margins.forecast_remap_fraction
                ),
                "output_cap_dbz": fso.active_set_margins.output_cap_dbz,
                "publication_support": (
                    fso.active_set_margins.publication_support
                ),
                "publication_confidence": (
                    fso.active_set_margins.publication_confidence
                ),
                "neural_prior_valid_probability": (
                    fso.active_set_margins.neural_prior_valid_probability
                ),
                "neural_prior_support_probability": (
                    fso.active_set_margins.neural_prior_support_probability
                ),
                "low_local_validity": (
                    fso.active_set_margins.low_local_validity
                ),
            },
            "feasibility_margins": {
                "reachability_support": (
                    fso.feasibility_margins.reachability_support
                ),
                "unresolved_amplitude_fraction": (
                    fso.feasibility_margins
                    .unresolved_amplitude_fraction
                ),
                "amplitude_confidence": (
                    fso.feasibility_margins.amplitude_confidence
                ),
                "motion_saturation_fraction": (
                    fso.feasibility_margins.motion_saturation_fraction
                ),
                "motion_speed_saturation_mps": (
                    fso.feasibility_margins
                    .motion_speed_saturation_mps
                ),
                "growth_saturation_per_step": (
                    fso.feasibility_margins.growth_saturation_per_step
                ),
                "low_interior_validity": (
                    fso.feasibility_margins.low_interior_validity
                ),
            },
            "gauss_newton_diagnostics": {
                "relative_curvature_defect": tensor_digest(
                    fso.gauss_newton_diagnostics.relative_curvature_defect
                ),
                "maximum_relative_curvature_defect": (
                    fso.gauss_newton_diagnostics
                    .maximum_relative_curvature_defect
                ),
                "reliable": fso.gauss_newton_diagnostics.reliable,
                "normal_products": (
                    fso.gauss_newton_diagnostics.normal_products
                ),
                "exact_hessian_products": (
                    fso.gauss_newton_diagnostics.exact_hessian_products
                ),
            },
        }
    )


def variational_fsoi_digest(fsoi: VariationalFSOI) -> str:
    """Content digest for one explicit first-order impact product."""

    return json_digest(
        {
            "version": "p1-variational-fsoi-digest-v12",
            "contract": fsoi.contract,
            "variational_fso_digest": fsoi.fso.variational_fso_digest,
            "perturbation_contract": fsoi.perturbation_contract,
            "perturbation_digest": fsoi.perturbation_digest,
            "perturbation_diagnostics": {
                "perturbed_pixel_count": (
                    fsoi.perturbation_diagnostics.perturbed_pixel_count
                ),
                "perturbed_fraction": (
                    fsoi.perturbation_diagnostics.perturbed_fraction
                ),
                "perturbed_area_km2": (
                    fsoi.perturbation_diagnostics.perturbed_area_km2
                ),
                "whitened_l2": fsoi.perturbation_diagnostics.whitened_l2,
                "maximum_per_tile_whitened_norm": (
                    fsoi.perturbation_diagnostics
                    .maximum_per_tile_whitened_norm
                ),
                "observation_weight_l2": (
                    fsoi.perturbation_diagnostics.observation_weight_l2
                ),
                "minimum_input_floor_margin_dbz": (
                    fsoi.perturbation_diagnostics
                    .minimum_input_floor_margin_dbz
                ),
                "minimum_input_ceiling_margin_dbz": (
                    fsoi.perturbation_diagnostics
                    .minimum_input_ceiling_margin_dbz
                ),
                "directional_classification_valid": (
                    fsoi.perturbation_diagnostics
                    .directional_classification_valid
                ),
                "baseline_dynamics_branch_status": (
                    fsoi.perturbation_diagnostics
                    .baseline_dynamics_branch_status
                ),
                "baseline_dynamics_branch_signature_digest": (
                    fsoi.perturbation_diagnostics
                    .baseline_dynamics_branch_signature_digest
                ),
            },
            "baseline_dynamics_branch_status": (
                fsoi.baseline_dynamics_branch_status
            ),
            "observation": {
                "detected_dbz": _variational_impact_digest_values(
                    fsoi.observation.detected_dbz
                ),
                "censor_threshold_dbz": _variational_impact_digest_values(
                    fsoi.observation.censor_threshold_dbz
                ),
                "observation_weight": _variational_impact_digest_values(
                    fsoi.observation.observation_weight
                ),
                "initial_background_dbz": (
                    _variational_impact_digest_values(
                        fsoi.observation.initial_background_dbz
                    )
                ),
                "baseline_dynamics_dbz": (
                    _variational_impact_digest_values(
                        fsoi.observation.baseline_dynamics_dbz
                    )
                ),
                "total": _variational_impact_digest_values(
                    fsoi.observation.total
                ),
                "baseline_branch_trusted_total": (
                    None
                    if fsoi.observation.baseline_branch_trusted_total is None
                    else _variational_impact_digest_values(
                        fsoi.observation.baseline_branch_trusted_total
                    )
                ),
            },
        }
    )


def validate_variational_fso(fso: VariationalFSO) -> None:
    """Reject any mutation of a content-addressed FSO result."""

    if fso.contract != "p1-variational-fso-v17":
        raise ValueError("unsupported P1 FSO contract")
    if (
        fso.sensitivity_scope
        != "residual_plus_input_dependent_initial_state_and_baseline_with_frozen_selection"
        or fso.baseline_dynamics_frozen is not False
        or fso.baseline_pair_selection_frozen is not True
    ):
        raise ValueError("unsupported P1 FSO sensitivity scope")
    trusted_branch = fso.baseline_dynamics_branch_status in (
        "not_applicable",
        "certified",
    )
    trusted_input = (
        fso.observation.baseline_branch_trusted_frozen_structure_input_dbz
    )
    if fso.baseline_dynamics_branch_status not in (
        "not_applicable",
        "unknown",
        "certified",
        "invalid",
    ) or (
        (trusted_input is not None) != trusted_branch
    ):
        raise ValueError("invalid P1 FSO baseline branch trust contract")
    _validate_verification_lineage_fields(
        contract=fso.verification_contract,
        content_digest=fso.verification_bundle_digest,
        lineage_complete=fso.verification_lineage_complete,
        valid_times=fso.verification_valid_times,
        grid_contract_digest=fso.verification_grid_contract_digest,
        radar_product_digest=fso.verification_radar_product_digest,
        qc_pipeline_digest=fso.verification_qc_pipeline_digest,
    )
    if variational_fso_digest(fso) != fso.variational_fso_digest:
        raise ValueError("P1 FSO result digest mismatch")


def validate_variational_fsoi(fsoi: VariationalFSOI) -> None:
    """Reject any mutation or cross-binding in a P1 impact result."""

    if fsoi.contract != "p1-linearized-observation-impact-v13":
        raise ValueError("unsupported P1 FSOI contract")
    validate_variational_fso(fsoi.fso)
    if fsoi.perturbation.digest != fsoi.perturbation_digest:
        raise ValueError("P1 FSOI perturbation digest mismatch")
    if (
        fsoi.baseline_dynamics_branch_status
        != fsoi.perturbation_diagnostics.baseline_dynamics_branch_status
    ):
        raise ValueError("P1 FSOI baseline branch status mismatch")
    expected_trusted = fsoi.baseline_dynamics_branch_status in (
        "not_applicable",
        "certified",
    )
    if (
        fsoi.observation.baseline_branch_trusted_total is not None
    ) != expected_trusted:
        raise ValueError("invalid P1 FSOI trusted-total contract")
    if variational_fsoi_digest(fsoi) != fsoi.variational_fsoi_digest:
        raise ValueError("P1 FSOI result digest mismatch")


def validate_variational_learning_impact(
    learning: VariationalLearningImpact,
    *,
    expected_trust_store_digest: str | None = None,
) -> None:
    """Reject mutation or cross-binding in a final learning decision."""

    if learning.contract != "p1-variational-learning-impact-v2":
        raise ValueError("unsupported variational learning impact")
    if expected_trust_store_digest is not None:
        _require_sha256(
            "expected_trust_store_digest",
            expected_trust_store_digest,
        )
    validation = learning.first_order_validation
    if learning.fsoi is not None:
        validate_variational_fsoi(learning.fsoi)
    if learning.fsoi is not None and validation is not None and (
        validation.source_fsoi_digest
        != learning.fsoi.variational_fsoi_digest
        or validation.nominal_forecast_digest
        != learning.fsoi.fso.forecast_run_digest
    ):
        raise ValueError("first-order validation lineage mismatch")
    if validation is not None and (
        first_order_validation_digest(validation)
        != validation.validation_digest
    ):
        raise ValueError("first-order validation digest mismatch")
    if not learning.eligibility.eligible:
        if learning.approval_evidence is not None:
            raise ValueError("rejected learning impact carries approval")
    else:
        fsoi = learning.fsoi
        impact = learning.frozen_domain_learning_impact
        evidence = learning.approval_evidence
        if fsoi is None or validation is None or impact is None or evidence is None:
            raise ValueError("eligible learning impact is incomplete")
        if not validation.first_order_valid:
            raise ValueError("eligible learning impact failed validation")
        expected = {
            "policy_digest": learning.eligibility.policy_digest,
            "fsoi_digest": fsoi.variational_fsoi_digest,
            "full_step_analysis_digest": validation.full_step_analysis_digest,
            "half_step_analysis_digest": validation.half_step_analysis_digest,
            "full_step_forecast_digest": validation.full_step_forecast_digest,
            "half_step_forecast_digest": validation.half_step_forecast_digest,
            "first_order_validation_digest": validation.validation_digest,
            "learning_impact_digest": _variational_impact_digest(impact),
            "approved_action_digest": fsoi.perturbation_digest,
            "nominal_input_bundle_digest": (
                validation.nominal_input_bundle_digest
            ),
        }
        if any(
            value is None or getattr(evidence, name) != value
            for name, value in expected.items()
        ):
            raise ValueError("learning approval evidence mismatch")
        if (
            expected_trust_store_digest is not None
            and evidence.trust_store_digest != expected_trust_store_digest
        ):
            raise ValueError("learning trust-store digest mismatch")
    if (
        variational_learning_impact_digest(learning)
        != learning.learning_result_digest
    ):
        raise ValueError("variational learning result digest mismatch")


def _new_variational_channel_accumulator(
    reference: Tensor,
    *,
    selected_count: int,
    lead_count: int,
    metric_count: int,
    height: int,
    width: int,
    tile_rows: int,
    tile_columns: int,
) -> _VariationalChannelAccumulator:
    return _VariationalChannelAccumulator(
        maps=reference.new_full(
            (selected_count, metric_count, 3, height, width),
            float("nan"),
        ),
        by_time=reference.new_full(
            (lead_count, metric_count, 3),
            float("nan"),
        ),
        tile_by_time=reference.new_full(
            (
                lead_count,
                metric_count,
                3,
                tile_rows,
                tile_columns,
            ),
            float("nan"),
        ),
    )


def _record_variational_channel(
    accumulator: _VariationalChannelAccumulator,
    values: Tensor,
    *,
    lead_index: int,
    metric_index: int,
    selected_index: int | None,
    tile_size: TileShape,
    signed_sum: bool,
) -> None:
    if signed_sum:
        accumulator.by_time[lead_index, metric_index] = values.reshape(
            3,
            -1,
        ).sum(dim=1)
        tile_function = _tile_sum
    else:
        accumulator.by_time[lead_index, metric_index] = (
            torch.linalg.vector_norm(values.reshape(3, -1), dim=1)
        )
        tile_function = _tile_l2
    accumulator.tile_by_time[lead_index, metric_index] = torch.stack(
        tuple(tile_function(values[index], tile_size) for index in range(3))
    )
    if selected_index is not None:
        accumulator.maps[selected_index, metric_index] = values


def _sensitivity_channel(
    accumulator: _VariationalChannelAccumulator,
) -> VariationalSensitivityChannel:
    return VariationalSensitivityChannel(
        maps=accumulator.maps,
        norm_by_time=accumulator.by_time,
        tile_norm_by_time=accumulator.tile_by_time,
    )


def _impact_channel(
    accumulator: _VariationalChannelAccumulator,
) -> VariationalImpactChannel:
    return VariationalImpactChannel(
        maps=accumulator.maps,
        sum_by_time=accumulator.by_time,
        tile_sum_by_time=accumulator.tile_by_time,
    )


@dataclass(frozen=True)
class SensitivitySnapshot:
    forecast_run_digest: str
    nowcast_config_digest: str
    sensitivity_config_digest: str
    grid_time_contract_digest: str | None
    verification_contract: str
    verification_bundle_digest: str
    verification_lineage_complete: bool
    verification_valid_times: tuple[str, ...] | None
    verification_grid_contract_digest: str | None
    verification_radar_product_digest: str | None
    verification_qc_pipeline_digest: str | None
    metric_names: tuple[str, ...]
    lead_minutes: tuple[int, ...]
    full_map_lead_minutes: tuple[int, ...]
    tile_size: int
    tile_shape_yx: TileShape
    context_feature_names: tuple[str, ...]
    context_features: Tensor
    analysis_control: Tensor
    forecast_scores: Tensor
    metric_available: Tensor
    control_sensitivity: Tensor
    forecast_sensitivity: Tensor
    forecast_cap_active_mask: Tensor
    forecast_confidence: Tensor
    path_evidence_by_metric: Tensor
    observation_source_fraction_by_metric: Tensor
    observation_verified_evidence_by_metric: Tensor
    background_verified_evidence_by_metric: Tensor
    direct: DirectSensitivity
    latest_sensitivity_mask: Tensor
    observation_std_dbz: Tensor | None
    observation_innovation_dbz: Tensor | None
    observation_innovation_mask: Tensor | None
    baseline_scores: Tensor | None
    reward_epsilon: float
    trust_components: dict[str, float]
    trust_score: float

    @property
    def impact_available(self) -> bool:
        return self.direct.impact is not None

    @property
    def reward_available(self) -> bool:
        return self.direct.reward is not None

    @property
    def whitened_tile_norm_available(self) -> bool:
        return self.direct.whitened_tile_norm is not None

    @property
    def observation_evidence_by_metric(self) -> Tensor:
        """Return the legacy observation-source fraction diagnostic."""

        return self.observation_source_fraction_by_metric


def compute_sensitivity_snapshot(
    latest_frame_dbz: Tensor,
    result: ForecastResult,
    verification_frames_dbz: VerificationInput,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    latest_background_dbz: Tensor | None = None,
    observation_std_dbz: float | Tensor | None = None,
    baseline_scores: Tensor | None = None,
) -> SensitivitySnapshot:
    """Compute M0 forecast/control/direct-observation sensitivities.

    The direct observation sensitivity is with respect to latest-frame dBZ
    inside a frozen active set. The FFT motion analysis is intentionally
    excluded: its discrete peak selection has no valid local derivative.
    Baseline-normalized reward remains disabled until baseline run, metric,
    verification, and valid-domain lineage can be verified together.
    """

    sensitivity_config = sensitivity_config or SensitivityConfig()
    if baseline_scores is not None:
        raise ValueError(
            "baseline_scores require a verified lineage contract; "
            "normalized reward is disabled until that contract exists"
        )
    nowcast_config = result.run.config
    result.validate_issuance()
    verification_bundle = _resolve_verification(
        verification_frames_dbz,
        result,
        sensitivity_config,
    )
    verification_frames = verification_bundle.frames_dbz
    result.run.validate_latest_frame(latest_frame_dbz)
    result.run.validate_latest_background(latest_background_dbz)
    latest_observation_mask = result.run.latest_observation_mask
    state = result.state
    metadata = result.metadata
    if metadata.data_status is DataStatus.UNAVAILABLE:
        raise ValueError("sensitivity is undefined for an unissued forecast")
    if metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL:
        raise ValueError("M0 direct sensitivity requires a P0 state")
    if (
        2 * sensitivity_config.active_margin_dbz
        >= nowcast_config.max_dbz - nowcast_config.min_dbz
    ):
        raise ValueError("active_margin_dbz leaves no differentiable range")
    _validate_inputs(
        latest_frame_dbz,
        verification_frames,
        state,
        nowcast_config,
        latest_background_dbz,
    )

    height, width = state.echo_linear.shape
    grid_time_contract = result.run.grid_time_contract
    tile_shape_yx = _metric_tile_shape(
        sensitivity_config,
        grid_time_contract,
    )
    lead_minutes = tuple(
        range(
            nowcast_config.interval_minutes,
            nowcast_config.horizon_minutes + 1,
            nowcast_config.interval_minutes,
        )
    )
    full_map_indices = _full_map_indices(
        sensitivity_config.full_map_lead_minutes,
        lead_minutes,
    )
    metric_count = len(sensitivity_config.metric_names)
    lead_count = len(lead_minutes)
    tile_rows = math.ceil(height / tile_shape_yx[0])
    tile_columns = math.ceil(width / tile_shape_yx[1])

    clean_verification = torch.nan_to_num(
        verification_frames,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    )
    verification_finite = verification_bundle.valid_mask
    issued_valid = torch.isfinite(result.forecast_dbz)
    if issued_valid.shape != verification_finite.shape:
        raise ValueError("issued forecast must match verification shape")
    if result.valid_mask.shape != verification_finite.shape:
        raise ValueError("forecast valid_mask must match verification shape")
    if not torch.equal(result.valid_mask, issued_valid):
        raise ValueError("forecast valid_mask must match issued finite values")
    verification_valid = verification_finite & issued_valid
    truth_linear = dbz_to_echo(
        clean_verification,
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    issued_echo = dbz_to_echo(
        torch.nan_to_num(
            result.forecast_dbz,
            nan=nowcast_config.min_dbz,
            posinf=nowcast_config.max_dbz,
            neginf=nowcast_config.min_dbz,
        ),
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    control = torch.cat(
        (state.displacement_yx, state.log_growth_per_step.reshape(1))
    )
    echo = state.echo_linear
    clean_latest, latest_active = _frozen_observation(
        latest_frame_dbz,
        latest_observation_mask,
        nowcast_config,
        sensitivity_config,
    )
    if not bool(torch.any(latest_active)):
        raise ValueError(
            "M0 direct sensitivity requires a valid latest observation"
        )
    observation_std, whitening_available = _observation_std(
        observation_std_dbz,
        latest_frame_dbz,
        sensitivity_config.epsilon,
    )

    score_shape = (lead_count, metric_count)
    forecast_scores = echo.new_full(score_shape, float("nan"))
    metric_available = torch.zeros(
        score_shape,
        dtype=torch.bool,
        device=echo.device,
    )
    control_sensitivity = echo.new_full(
        (lead_count, metric_count, 3),
        float("nan"),
    )
    direct_norm = echo.new_zeros((lead_count, metric_count))
    tile_direct_norm = echo.new_zeros(
        (lead_count, metric_count, tile_rows, tile_columns)
    )
    tile_shape = (lead_count, metric_count, tile_rows, tile_columns)
    if whitening_available:
        tile_whitened_norm = echo.new_zeros(tile_shape)
    else:
        tile_whitened_norm = None
    selected_count = len(full_map_indices)
    forecast_maps = echo.new_full(
        (selected_count, metric_count, height, width),
        float("nan"),
    )
    direct_maps = echo.new_zeros(
        (selected_count, metric_count, height, width)
    )
    selected_cap_masks = torch.zeros(
        (selected_count, height, width),
        dtype=torch.bool,
        device=echo.device,
    )
    all_cap_masks = torch.zeros(
        (lead_count, height, width),
        dtype=torch.bool,
        device=echo.device,
    )
    forecast_confidence = result.forecast_confidence
    forecast_source_support = result.forecast_source_support
    forecast_observation_support = result.forecast_observation_source_support
    forecast_verified_support = result.forecast_verified_support
    confidence_decay = torch.where(
        forecast_verified_support > 0,
        forecast_confidence
        / forecast_verified_support.clamp_min(
            torch.finfo(forecast_verified_support.dtype).tiny
        ),
        0.0,
    )
    observation_verified_confidence = (
        result.forecast_observation_verified_support * confidence_decay
    )
    background_verified_confidence = (
        result.forecast_background_verified_support * confidence_decay
    )
    if not torch.allclose(
        observation_verified_confidence + background_verified_confidence,
        forecast_confidence,
        rtol=1.0e-5,
        atol=nowcast_config.contract_absolute_tolerance,
    ):
        raise ValueError("forecast evidence channels do not close")
    path_evidence_by_metric = echo.new_full(score_shape, float("nan"))
    observation_source_fraction_by_metric = echo.new_full(
        score_shape,
        float("nan"),
    )
    observation_verified_evidence_by_metric = echo.new_full(
        score_shape,
        float("nan"),
    )
    background_verified_evidence_by_metric = echo.new_full(
        score_shape,
        float("nan"),
    )
    innovation, innovation_mask = _dbz_innovation(
        latest_frame_dbz,
        latest_background_dbz,
        latest_observation_mask,
        nowcast_config,
    )
    impact_input_available = (
        innovation is not None
        and innovation_mask is not None
        and bool(torch.any(innovation_mask & latest_active))
    )
    if not impact_input_available:
        innovation = None
        innovation_mask = None
        tile_impact = None
        observation_impact = None
    else:
        tile_impact = echo.new_zeros(
            (lead_count, metric_count, tile_rows, tile_columns)
        )
        observation_impact = echo.new_zeros((lead_count, metric_count))
    selected_position = {
        index: position for position, index in enumerate(full_map_indices)
    }

    for lead_index in range(lead_count):
        truth = truth_linear[lead_index]
        valid = _metric_domain_weight(
            result,
            verification_finite[lead_index],
            lead_index,
            sensitivity_config.metric_domain,
        )
        lead_cell = freeze_remap_cell(
            (lead_index + 1) * state.displacement_yx
        )
        latent_prediction = _forecast_linear_at_step_core(
            state,
            lead_index + 1,
            nowcast_config,
            lead_cell,
        )
        prediction, cap_active = _freeze_output_cap(
            latent_prediction,
            nowcast_config,
        )
        nominal_valid = issued_valid[lead_index]
        if not torch.allclose(
            prediction[nominal_valid],
            issued_echo[lead_index][nominal_valid],
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError(
                "sensitivity model disagrees with the issued forecast"
            )
        all_cap_masks[lead_index] = cap_active
        if lead_index in selected_position:
            selected_cap_masks[selected_position[lead_index]] = cap_active

        for metric_index, metric_name in enumerate(
            sensitivity_config.metric_names
        ):
            if not _metric_has_support(
                metric_name,
                prediction,
                truth,
                valid,
                nowcast_config,
                sensitivity_config,
            ):
                direct_norm[lead_index, metric_index] = float("nan")
                tile_direct_norm[lead_index, metric_index] = float("nan")
                if tile_whitened_norm is not None:
                    tile_whitened_norm[lead_index, metric_index] = float("nan")
                if lead_index in selected_position:
                    position = selected_position[lead_index]
                    direct_maps[position, metric_index] = float("nan")
                if observation_impact is not None and tile_impact is not None:
                    observation_impact[lead_index, metric_index] = float("nan")
                    tile_impact[lead_index, metric_index] = float("nan")
                continue

            metric_available[lead_index, metric_index] = True
            metric = lambda forecast: forecast_metric(
                metric_name,
                forecast,
                truth,
                valid,
                nowcast_config,
                sensitivity_config,
                grid_time_contract,
            )
            score = metric(prediction)

            def score_from_state(
                candidate_control: Tensor,
                candidate_latest_dbz: Tensor,
            ) -> Tensor:
                candidate_echo = _active_dbz_to_echo(
                    candidate_latest_dbz,
                    clean_latest,
                    echo,
                    latest_active,
                    nowcast_config,
                )
                candidate_state = _state_from_control(
                    state,
                    candidate_control,
                    candidate_echo,
                )
                candidate = _forecast_linear_at_step_core(
                    candidate_state,
                    lead_index + 1,
                    nowcast_config,
                    lead_cell,
                )
                return metric(_apply_output_cap(candidate, cap_active, nowcast_config))

            control_gradient, direct_gradient = torch.func.grad(
                score_from_state,
                argnums=(0, 1),
            )(control, clean_latest)
            forecast_gradient = torch.func.grad(metric)(prediction)
            whitened_gradient = direct_gradient * observation_std
            evidence_weight = torch.abs(forecast_gradient.detach())
            evidence = _metric_evidence_ratios(
                evidence_weight,
                forecast_source_support[lead_index],
                forecast_confidence[lead_index],
                forecast_observation_support[lead_index],
                observation_verified_confidence[lead_index],
                background_verified_confidence[lead_index],
                sensitivity_config.epsilon,
            )
            if evidence is not None:
                (
                    path_evidence_by_metric[lead_index, metric_index],
                    observation_source_fraction_by_metric[
                        lead_index, metric_index
                    ],
                    observation_verified_evidence_by_metric[
                        lead_index, metric_index
                    ],
                    background_verified_evidence_by_metric[
                        lead_index, metric_index
                    ],
                ) = evidence

            forecast_scores[lead_index, metric_index] = score.detach()
            control_sensitivity[lead_index, metric_index] = (
                control_gradient.detach()
            )
            direct_norm[lead_index, metric_index] = torch.linalg.vector_norm(
                direct_gradient.detach()
            )
            tile_direct_norm[lead_index, metric_index] = _tile_l2(
                direct_gradient.detach(),
                tile_shape_yx,
            )
            if tile_whitened_norm is not None:
                tile_whitened_norm[lead_index, metric_index] = _tile_l2(
                    whitened_gradient.detach(),
                    tile_shape_yx,
                )

            if lead_index in selected_position:
                position = selected_position[lead_index]
                forecast_maps[position, metric_index] = forecast_gradient.detach()
                direct_maps[position, metric_index] = direct_gradient.detach()

            if observation_impact is not None and tile_impact is not None:
                if innovation is None or innovation_mask is None:
                    raise RuntimeError(
                        "impact storage requires an observation innovation"
                    )
                contribution = torch.where(
                    innovation_mask,
                    direct_gradient.detach() * innovation,
                    torch.zeros_like(direct_gradient),
                )
                tiles = _tile_sum(
                    contribution,
                    tile_shape_yx,
                )
                tile_impact[lead_index, metric_index] = tiles
                observation_impact[lead_index, metric_index] = tiles.sum()

    has_metric_support = bool(torch.any(metric_available))
    if not has_metric_support:
        observation_impact = None
        tile_impact = None

    trust_components = _trust_components(
        state,
        metadata,
        control,
        echo,
        truth_linear,
        verification_valid,
        control_sensitivity,
        metric_available,
        all_cap_masks,
        observation_verified_evidence_by_metric,
        nowcast_config,
        sensitivity_config,
        grid_time_contract,
    )
    trust_score = math.prod(trust_components.values())

    return SensitivitySnapshot(
        forecast_run_digest=result.forecast_run_digest,
        nowcast_config_digest=nowcast_config.digest,
        sensitivity_config_digest=sensitivity_config.digest,
        grid_time_contract_digest=result.run.grid_time_contract_digest,
        verification_contract=verification_bundle.contract,
        verification_bundle_digest=verification_bundle.content_digest,
        verification_lineage_complete=(
            verification_bundle.lineage_complete
        ),
        verification_valid_times=verification_bundle.valid_times,
        verification_grid_contract_digest=(
            verification_bundle.grid_contract_digest
        ),
        verification_radar_product_digest=(
            verification_bundle.radar_product_digest
        ),
        verification_qc_pipeline_digest=(
            verification_bundle.qc_pipeline_digest
        ),
        metric_names=sensitivity_config.metric_names,
        lead_minutes=lead_minutes,
        full_map_lead_minutes=sensitivity_config.full_map_lead_minutes,
        tile_size=max(tile_shape_yx),
        tile_shape_yx=tile_shape_yx,
        context_feature_names=CONTEXT_FEATURE_NAMES,
        context_features=extract_context_features(
            latest_frame_dbz,
            state,
            metadata,
            nowcast_config,
            latest_observation_mask=latest_observation_mask,
            grid_time_contract=result.run.grid_time_contract,
        ),
        analysis_control=control.detach(),
        forecast_scores=forecast_scores,
        metric_available=metric_available,
        control_sensitivity=control_sensitivity,
        forecast_sensitivity=forecast_maps,
        forecast_cap_active_mask=selected_cap_masks,
        forecast_confidence=forecast_confidence.detach(),
        path_evidence_by_metric=path_evidence_by_metric.detach(),
        observation_source_fraction_by_metric=(
            observation_source_fraction_by_metric.detach()
        ),
        observation_verified_evidence_by_metric=(
            observation_verified_evidence_by_metric.detach()
        ),
        background_verified_evidence_by_metric=(
            background_verified_evidence_by_metric.detach()
        ),
        direct=DirectSensitivity(
            maps=direct_maps,
            norm=direct_norm,
            tile_norm=tile_direct_norm,
            whitened_tile_norm=tile_whitened_norm,
            impact=observation_impact,
            tile_impact=tile_impact,
            reward=None,
        ),
        latest_sensitivity_mask=latest_active,
        observation_std_dbz=(
            observation_std.detach()
            if whitening_available
            else None
        ),
        observation_innovation_dbz=(
            innovation
            if innovation is not None
            else None
        ),
        observation_innovation_mask=(
            innovation_mask
            if innovation_mask is not None
            else None
        ),
        baseline_scores=None,
        reward_epsilon=sensitivity_config.epsilon,
        trust_components=trust_components,
        trust_score=trust_score,
    )


def compute_sensitivity_snapshot_from_run(
    result: ForecastResult,
    verification_frames_dbz: VerificationInput,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    observation_std_dbz: float | Tensor | None = None,
    baseline_scores: Tensor | None = None,
) -> SensitivitySnapshot:
    """Compute delayed M0 using the exact inputs embedded in ``result``."""

    return compute_sensitivity_snapshot(
        result.run.latest_frame_dbz,
        result,
        verification_frames_dbz,
        sensitivity_config=sensitivity_config,
        latest_background_dbz=result.run.latest_background_dbz,
        observation_std_dbz=observation_std_dbz,
        baseline_scores=baseline_scores,
    )


def compute_variational_fso(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    adjoint_config: VariationalAdjointConfig | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalFSO:
    """Compute frozen-final P1 forecast sensitivity to observations."""

    resolved_adjoint = adjoint_config or VariationalAdjointConfig()
    operations_per_apply = _analysis_whitener_operations_per_apply(analysis)
    with _count_observation_whitener_applies(
        operations_per_apply=operations_per_apply,
        maximum_total_operations=(
            resolved_adjoint.maximum_whitener_total_operations
        ),
    ) as counter:
        fso, _, _ = _compute_variational_products(
            result,
            analysis,
            verification_frames_dbz,
            sensitivity_config=sensitivity_config,
            adjoint_config=resolved_adjoint,
            observation_perturbation=None,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
    return _bind_fso_whitener_telemetry(fso, analysis, counter[0])


def compute_variational_fsoi(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    observation_perturbation: VariationalObservationPerturbation,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    adjoint_config: VariationalAdjointConfig | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalFSOI:
    """Compute signed first-order impact for an explicit perturbation."""

    resolved_adjoint = adjoint_config or VariationalAdjointConfig()
    operations_per_apply = _analysis_whitener_operations_per_apply(analysis)
    with _count_observation_whitener_applies(
        operations_per_apply=operations_per_apply,
        maximum_total_operations=(
            resolved_adjoint.maximum_whitener_total_operations
        ),
    ) as counter:
        fso, observation_impact, perturbation_diagnostics = (
            _compute_variational_products(
                result,
                analysis,
                verification_frames_dbz,
                sensitivity_config=sensitivity_config,
                adjoint_config=resolved_adjoint,
                observation_perturbation=observation_perturbation,
                neural_prior_runner=neural_prior_runner,
                neural_prior_application=neural_prior_application,
            )
        )
    fso = _bind_fso_whitener_telemetry(fso, analysis, counter[0])
    if observation_impact is None:
        raise RuntimeError("variational FSOI impact was not materialized")
    if perturbation_diagnostics is None:
        raise RuntimeError("variational perturbation was not validated")
    fsoi = VariationalFSOI(
        contract="p1-linearized-observation-impact-v13",
        fso=fso,
        perturbation=observation_perturbation,
        perturbation_contract=observation_perturbation.contract,
        perturbation_digest=observation_perturbation.digest,
        perturbation_diagnostics=perturbation_diagnostics,
        baseline_dynamics_branch_status=(
            perturbation_diagnostics.baseline_dynamics_branch_status
        ),
        observation=observation_impact,
        variational_fsoi_digest="",
    )
    return replace(
        fsoi,
        variational_fsoi_digest=variational_fsoi_digest(fsoi),
    )


def compute_variational_observation_removal_impact(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    removal_mask: Tensor,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    removal_config: ObservationRemovalConfig | None = None,
) -> ObservationRemovalImpact:
    """Rebuild P0/P1 and forecast after removing accepted observations.

    This is a nonlinear denial experiment, not a first-order FSOI. The
    analysis background, active support, common-bias whitener, robust optimum,
    posterior, confidence, and issuance domain are all recomputed.
    """

    result.validate_issuance()
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("observation removal requires a P1 linearization")
    validate_analysis_linearization_content(analysis.control, linearization)
    request = ObservationRemovalRequest(
        removal_mask=removal_mask,
        linearization_digest=linearization.linearization_digest,
    )
    observations = linearization.observations
    frozen = linearization.frozen
    mask = request.removal_mask.to(observations.valid_mask.device)
    if mask.shape != observations.valid_mask.shape:
        raise ValueError("removal mask must match the observation grid")
    if bool(torch.any(mask & ~observations.valid_mask)):
        raise ValueError("only accepted observations can be removed")
    removal = removal_config or ObservationRemovalConfig()
    removed_count = int(torch.count_nonzero(mask))
    valid_count = int(torch.count_nonzero(observations.valid_mask))
    removed_fraction = removed_count / max(1, valid_count)
    if removed_count > removal.maximum_removed_observation_count:
        raise ValueError("observation removal exceeds its count budget")
    if removed_fraction > removal.maximum_removed_fraction:
        raise ValueError("observation removal exceeds its fraction budget")
    grid = frozen.grid_time_contract
    union_count = int(torch.count_nonzero(torch.any(mask, dim=0)))
    removed_area_km2 = (
        None if grid is None else union_count * grid.cell_area_m2 / 1.0e6
    )
    if removal.maximum_removed_area_km2 is not None:
        if removed_area_km2 is None:
            raise ValueError("physical removal budget requires a grid contract")
        if removed_area_km2 > removal.maximum_removed_area_km2:
            raise ValueError("observation removal exceeds its area budget")

    original_qc = ~observations.qc_rejected_mask
    changed_qc = original_qc & ~mask
    manifest = _operational_manifest_from_run(result)
    identity = _operational_identity_from_run(result)
    operations_per_apply = _observation_whitener_operations_per_apply(
        observations
    )
    with _count_observation_whitener_applies(
        operations_per_apply=operations_per_apply,
        maximum_total_operations=removal.maximum_whitener_total_operations,
    ) as counter:
        removed_forecast, removed_analysis = variational_nowcast(
            frozen.input_frames_dbz,
            nowcast_config=frozen.nowcast_config,
            analysis_config=frozen.analysis_config,
            observation_std_dbz=observations.std_dbz,
            quality_weight=observations.quality_weight,
            qc_mask=changed_qc,
            observation_common_bias_group_index=(
                observations.common_bias_group_index
            ),
            observation_common_bias_mode_weights=(
                observations.common_bias_mode_weights
            ),
            background_frames_dbz=frozen.background_frames_dbz,
            background_age_minutes=frozen.background_age_minutes,
            grid_time_contract=grid,
            operational_calibration_manifest=manifest,
            operational_calibration_approval_digest=(
                result.run.operational_calibration_approval_digest
            ),
            operational_data_identity=identity,
        )
    removed_linearization = removed_analysis.linearization
    if (
        not removed_analysis.converged
        or removed_analysis.used_fallback
        or removed_analysis.degraded
        or not removed_analysis.p1_forecast_eligible
        or removed_linearization is None
    ):
        raise RuntimeError(
            "observation-removal analysis did not produce an eligible P1"
        )
    removed_forecast.validate_issuance()
    config = sensitivity_config or SensitivityConfig()
    leads = config.full_map_lead_minutes
    if not leads:
        raise ValueError("observation removal requires at least one lead")
    interval = result.run.config.interval_minutes
    if any(
        minutes % interval != 0
        or minutes > result.run.config.horizon_minutes
        for minutes in leads
    ):
        raise ValueError("observation-removal leads must be issued forecast leads")
    verification = _resolve_verification(
        verification_frames_dbz,
        result,
        config,
    )
    _ = _resolve_verification(
        verification_frames_dbz,
        removed_forecast,
        config,
    )
    nominal_scores, nominal_available = _resolved_forecast_scores(
        result,
        analysis.state,
        verification,
        leads,
        config,
    )
    removed_scores, removed_available = _resolved_forecast_scores(
        removed_forecast,
        removed_analysis.state,
        verification,
        leads,
        config,
    )
    available = nominal_available & removed_available
    metric_change = torch.where(
        available,
        removed_scores - nominal_scores,
        torch.full_like(nominal_scores, float("nan")),
    )
    return ObservationRemovalImpact(
        request=request,
        nominal_scores=nominal_scores,
        removed_scores=removed_scores,
        metric_change=metric_change,
        metric_available=available,
        lead_minutes=leads,
        metric_names=config.metric_names,
        metric_domain=config.metric_domain,
        nominal_forecast_digest=_forecast_result_content_digest(result),
        removed_forecast_digest=_forecast_result_content_digest(
            removed_forecast
        ),
        removed_linearization_digest=(
            removed_linearization.linearization_digest
        ),
        verification_bundle_digest=verification.content_digest,
        sensitivity_config_digest=config.digest,
        removal_config_digest=removal.digest,
        removed_observation_count=removed_count,
        removed_fraction=removed_fraction,
        removed_area_km2=removed_area_km2,
        whitener_operations_per_apply=operations_per_apply,
        observed_whitener_apply_count=counter[0],
        observed_whitener_total_operations=operations_per_apply * counter[0],
    )


def validate_observation_removal_impact(
    impact: ObservationRemovalImpact,
) -> None:
    """Validate a resolved denial result before durable use."""

    expected_request = json_digest(
        {
            "contract": impact.request.contract,
            "removal_mask": tensor_digest(impact.request.removal_mask),
            "linearization_digest": impact.request.linearization_digest,
        }
    )
    if impact.request.request_digest != expected_request:
        raise ValueError("observation-removal request digest mismatch")
    if (
        impact.observation_removal_impact_digest
        != _observation_removal_impact_digest(impact)
    ):
        raise ValueError("observation-removal impact digest mismatch")


def _forecast_result_content_digest(result: ForecastResult) -> str:
    return json_digest(
        {
            "contract": "forecast-result-content-v1",
            "forecast_run_digest": result.forecast_run_digest,
            "forecast_dbz_digest": result.forecast_dbz_digest,
            "valid_mask_digest": result.valid_mask_digest,
            "state_metadata_digest": result.state_metadata_digest,
        }
    )


def _operational_manifest_from_run(
    result: ForecastResult,
) -> OperationalCalibrationManifest | None:
    value = result.run.operational_calibration_manifest_json
    return None if value is None else OperationalCalibrationManifest.from_json(value)


def _operational_identity_from_run(
    result: ForecastResult,
) -> OperationalDataIdentity | None:
    value = result.run.operational_data_identity_json
    return None if value is None else OperationalDataIdentity.from_json(value)


def _resolved_forecast_scores(
    result: ForecastResult,
    state: RadarState,
    verification: _ResolvedVerification,
    leads: tuple[int, ...],
    config: SensitivityConfig,
    *,
    domain_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    nowcast = result.run.config
    clean_truth = torch.nan_to_num(
        verification.frames_dbz,
        nan=nowcast.min_dbz,
        posinf=nowcast.max_dbz,
        neginf=nowcast.min_dbz,
    ).clamp(nowcast.min_dbz, nowcast.max_dbz)
    truth_linear = dbz_to_echo(
        clean_truth,
        min_dbz=nowcast.min_dbz,
        max_dbz=nowcast.max_dbz,
    )
    finite_truth = verification.valid_mask & torch.isfinite(
        verification.frames_dbz
    )
    scores = state.echo_linear.new_full(
        (len(leads), len(config.metric_names)),
        float("nan"),
    )
    available = torch.zeros_like(scores, dtype=torch.bool)
    for lead_index, minutes in enumerate(leads):
        step = minutes // nowcast.interval_minutes
        forecast_index = step - 1
        forecast, _ = _freeze_output_cap(
            forecast_linear_at_step(state, step, nowcast),
            nowcast,
        )
        weight = (
            _metric_domain_weight(
                result,
                finite_truth[forecast_index],
                forecast_index,
                config.metric_domain,
            )
            if domain_weights is None
            else domain_weights[lead_index]
        )
        if domain_weights is None:
            weight = weight * verification.metric_weight[forecast_index].to(weight)
        for metric_index, name in enumerate(config.metric_names):
            if not _metric_has_support(
                name,
                forecast,
                truth_linear[forecast_index],
                weight,
                nowcast,
                config,
            ):
                continue
            scores[lead_index, metric_index] = forecast_metric(
                name,
                forecast,
                truth_linear[forecast_index],
                weight,
                nowcast,
                config,
                result.run.grid_time_contract,
            )
            available[lead_index, metric_index] = True
    return scores.detach(), available.detach()


def _resolved_forecast_domain_weights(
    result: ForecastResult,
    verification: _ResolvedVerification,
    leads: tuple[int, ...],
    config: SensitivityConfig,
) -> Tensor:
    """Return one frozen metric-domain weight per requested lead."""

    finite = verification.valid_mask & torch.isfinite(verification.frames_dbz)
    weights = []
    for minutes in leads:
        forecast_index = minutes // result.run.config.interval_minutes - 1
        weights.append(
            _metric_domain_weight(
                result,
                finite[forecast_index],
                forecast_index,
                config.metric_domain,
            )
            * verification.metric_weight[forecast_index].to(result.state.echo_linear)
        )
    return torch.stack(weights).detach()


def validate_variational_fsoi_issuance_impact(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    fsoi: VariationalFSOI,
    *,
    policy: AutomatedLearningPolicy,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> FirstOrderValidation:
    """Re-solve a physical FSOI on the changed issuance domain.

    This is a research diagnostic. Automated learning remains bound to the
    smoother ``frozen_metric_domain`` contract.
    """

    validate_variational_fsoi(fsoi)
    result.validate_issuance()
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("issuance validation requires a linearization")
    if fsoi.perturbation.perturbation_semantics != "physical_radar_value":
        raise ValueError("issuance validation requires a physical perturbation")
    if fsoi.fso.forecast_run_digest != result.forecast_run_digest:
        raise ValueError("issuance validation forecast mismatch")
    if fsoi.fso.linearization_digest != linearization.linearization_digest:
        raise ValueError("issuance validation linearization mismatch")
    if fsoi.fso.sensitivity_config_digest != policy.sensitivity_config.digest:
        raise ValueError("issuance validation sensitivity policy mismatch")
    if fsoi.fso.adjoint_config_digest != policy.adjoint_config.digest:
        raise ValueError("issuance validation adjoint policy mismatch")
    return _validate_first_order_learning_impact(
        result,
        analysis,
        verification_frames_dbz,
        fsoi,
        policy,
        metric_domain_contract="resolved_issuance_domain",
        maximum_whitener_total_operations=(
            policy.maximum_whitener_total_operations
        ),
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )


def _bind_fso_whitener_telemetry(
    fso: VariationalFSO,
    analysis: AnalysisResult | P1LinearizationState,
    apply_count: int,
) -> VariationalFSO:
    linearization = analysis.linearization
    if linearization is None:
        raise RuntimeError("P1 FSO lacks a linearization")
    updated = replace(
        fso,
        whitener_operations_per_apply=(
            _observation_whitener_operations_per_apply(
                linearization.observations
            )
        ),
        observed_whitener_apply_count=apply_count,
        variational_fso_digest="",
    )
    return replace(
        updated,
        variational_fso_digest=variational_fso_digest(updated),
    )


def _analysis_whitener_operations_per_apply(
    analysis: AnalysisResult | P1LinearizationState,
) -> int:
    linearization = analysis.linearization
    if linearization is None:
        return 0
    return _observation_whitener_operations_per_apply(
        linearization.observations
    )


def score_candidate_perturbations(
    fso: VariationalFSO,
    analysis: AnalysisResult | P1LinearizationState,
    candidates: Iterable[
        tuple[
            str,
            SparseRadarPerturbation | VariationalObservationPerturbation,
        ]
    ],
    *,
    policy: AutomatedLearningPolicy,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalCandidateRanking:
    """Stream, precheck, and rank physical-radar candidates with one FSO."""

    validate_variational_fso(fso)
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("candidate ranking requires a linearization")
    if fso.linearization_digest != linearization.linearization_digest:
        raise ValueError("candidate FSO linearization mismatch")
    if fso.sensitivity_config_digest != policy.sensitivity_config.digest:
        raise ValueError("candidate FSO sensitivity policy mismatch")
    if fso.adjoint_config_digest != policy.ranking_adjoint_config.digest:
        raise ValueError("candidate FSO adjoint policy mismatch")
    if fso.lead_minutes != fso.full_map_lead_minutes:
        raise ValueError("candidate ranking requires a full map for every lead")
    sensitivity = fso.observation.frozen_structure_input_dbz
    maps = sensitivity.maps
    if maps.shape[:2] != fso.forecast_scores.shape:
        raise ValueError("candidate FSO map coverage is incomplete")
    scales = maps.new_tensor(
        tuple(
            policy.threshold_for(name).effective_ranking_scale
            for name in fso.metric_names
        )
    )
    metric_weights = maps.new_tensor(
        tuple(
            policy.threshold_for(name).ranking_weight
            for name in fso.metric_names
        )
    )
    lead_weights = maps.new_tensor(policy.resolved_ranking_lead_weights)
    weighted_scale = lead_weights[:, None] * metric_weights[None, :]
    flat_maps = maps.reshape(*maps.shape[:2], -1)
    top: list[tuple[str, SparseRadarPerturbation, Tensor, float]] = []
    prechecks: list[VariationalCandidatePrecheck] = []
    identifiers: set[str] = set()
    candidate_count = 0
    scoring_operations = 0
    whitener_apply_count = fso.observed_whitener_apply_count
    if (
        fso.whitener_operations_per_apply * whitener_apply_count
        > policy.maximum_whitener_total_operations
    ):
        raise ValueError("common-bias total operation budget exhausted")
    started = time.monotonic()
    for candidate_id, candidate in candidates:
        candidate_count += 1
        if time.monotonic() - started > (
            policy.maximum_candidate_ranking_wall_seconds
        ):
            raise ValueError("candidate ranking wall-time budget exhausted")
        if candidate_count > policy.maximum_candidate_count:
            raise ValueError("learning candidate count exceeds its policy budget")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("learning candidate identifiers must be nonempty")
        if candidate_id in identifiers:
            raise ValueError("learning candidate identifiers must be unique")
        identifiers.add(candidate_id)
        sparse = _sparse_candidate(candidate)
        reason = _candidate_precheck_reason(
            sparse,
            linearization,
            policy,
        )
        if reason is not None:
            prechecks.append(
                VariationalCandidatePrecheck(
                    candidate_id=candidate_id,
                    perturbation_digest=sparse.digest,
                    admissible=False,
                    rejection_reason=reason,
                )
            )
            continue
        operations = sparse.nonzero_count * maps.shape[0] * maps.shape[1]
        scoring_operations += operations
        if scoring_operations > policy.maximum_candidate_scoring_operations:
            raise ValueError("candidate scoring operation budget exhausted")
        indices = sparse.flat_indices.to(maps.device)
        values = sparse.delta_values.to(dtype=maps.dtype, device=maps.device)
        prediction = (
            flat_maps.index_select(-1, indices) * values[None, None, :]
        ).sum(dim=-1)
        prediction = torch.where(
            fso.metric_available,
            prediction,
            torch.full_like(prediction, float("nan")),
        )
        score = _candidate_ranking_score(
            prediction,
            fso.metric_available,
            scales,
            weighted_scale,
            policy.ranking_objective,
        )
        if score == 0.0:
            prechecks.append(
                VariationalCandidatePrecheck(
                    candidate_id=candidate_id,
                    perturbation_digest=sparse.digest,
                    admissible=True,
                    rejection_reason=None,
                )
            )
            continue
        would_enter = len(top) < policy.maximum_learning_candidates_to_validate
        if not would_enter:
            worst = top[-1]
            would_enter = (-score, candidate_id) < (-worst[3], worst[0])
        if would_enter:
            reason, precheck_applies = _candidate_full_precheck_reason(
                sparse,
                linearization,
                policy,
                neural_prior_runner=neural_prior_runner,
                neural_prior_application=neural_prior_application,
            )
            whitener_apply_count += precheck_applies
            if (
                fso.whitener_operations_per_apply * whitener_apply_count
                > policy.maximum_whitener_total_operations
            ):
                raise ValueError("common-bias total operation budget exhausted")
        prechecks.append(
            VariationalCandidatePrecheck(
                candidate_id=candidate_id,
                perturbation_digest=sparse.digest,
                admissible=reason is None,
                rejection_reason=reason,
            )
        )
        if reason is not None or not would_enter:
            continue
        top.append((candidate_id, sparse, prediction.detach(), score))
        top.sort(key=lambda item: (-item[3], item[0]))
        del top[policy.maximum_learning_candidates_to_validate :]
    if candidate_count == 0:
        raise ValueError("at least one learning candidate is required")
    return VariationalCandidateRanking(
        fso=fso,
        prechecks=tuple(prechecks),
        policy_digest=policy.digest,
        ranking_objective=policy.ranking_objective,
        candidate_count=candidate_count,
        scoring_operations=scoring_operations,
        whitener_operations_per_apply=fso.whitener_operations_per_apply,
        observed_whitener_apply_count=whitener_apply_count,
        scores=tuple(
            VariationalCandidateScore(
                candidate_id=candidate_id,
                perturbation=perturbation,
                predicted_metric_change=prediction,
                score=score,
                rank=rank,
            )
            for rank, (candidate_id, perturbation, prediction, score) in (
                enumerate(top, start=1)
            )
        ),
    )


def _sparse_candidate(
    candidate: SparseRadarPerturbation | VariationalObservationPerturbation,
) -> SparseRadarPerturbation:
    if isinstance(candidate, SparseRadarPerturbation):
        return candidate
    delta = candidate.physical_radar_dbz_delta
    if candidate.perturbation_semantics != "physical_radar_value" or delta is None:
        raise ValueError("candidate must be a physical radar perturbation")
    return SparseRadarPerturbation.from_dense(delta)


def _candidate_precheck_reason(
    candidate: SparseRadarPerturbation,
    linearization: AnalysisLinearization,
    policy: AutomatedLearningPolicy,
) -> str | None:
    if candidate.retained_bytes > policy.maximum_candidate_bytes:
        return "candidate_byte_budget_exceeded"
    if candidate.nonzero_count > policy.maximum_candidate_nonzeros:
        return "candidate_nonzero_budget_exceeded"
    observations = linearization.observations
    frozen = linearization.frozen
    if candidate.shape != tuple(observations.dbz.shape):
        return "candidate_perturbation_shape_mismatch"
    indices = candidate.flat_indices.to(observations.dbz.device)
    values = candidate.delta_values.to(
        dtype=observations.dbz.dtype,
        device=observations.dbz.device,
    )
    detected = observations.detected_mask.reshape(-1).index_select(0, indices)
    if not bool(torch.all(detected)):
        return "physical_radar_delta_outside_detected_observations"
    config = policy.adjoint_config
    if bool(torch.any(torch.abs(values) > config.maximum_detected_delta_dbz)):
        return "detected_dbz_exceeds_local_perturbation_limit"
    nominal = observations.dbz.reshape(-1).index_select(0, indices)
    changed = nominal + values
    if bool(torch.any(changed < frozen.nowcast_config.min_dbz)) or bool(
        torch.any(changed > frozen.nowcast_config.max_dbz)
    ):
        return "physical_radar_perturbation_crosses_input_clamp"
    if bool(
        torch.any(
            changed
            < frozen.analysis_config.detection_limit_dbz
            + config.minimum_detection_margin_dbz
        )
    ):
        return "observation_perturbation_crosses_classification_branch"
    count = candidate.nonzero_count
    valid_count = max(1, int(torch.count_nonzero(observations.valid_mask)))
    if count > config.maximum_perturbed_pixel_count:
        return "observation_perturbation_exceeds_pixel_budget"
    if count / valid_count > config.maximum_perturbed_fraction:
        return "observation_perturbation_exceeds_area_fraction"
    grid = frozen.grid_time_contract
    if config.maximum_perturbed_area_km2 is not None:
        if grid is None:
            return "physical_perturbation_area_requires_grid_contract"
        area_km2 = count * grid.cell_area_m2 / 1.0e6
        if area_km2 > config.maximum_perturbed_area_km2:
            return "observation_perturbation_exceeds_physical_area_budget"
    if observations.common_bias_mode_weights is None:
        quality = observations.quality_weight.reshape(-1).index_select(0, indices)
        std = observations.std_dbz.reshape(-1).index_select(0, indices)
        energy = quality * (values / std).square()
        if math.sqrt(float(torch.sum(energy).detach())) > (
            config.maximum_whitened_perturbation_l2
        ):
            return "observation_perturbation_exceeds_whitened_trust_radius"
        if _sparse_maximum_tile_norm(
            indices,
            energy,
            observations.dbz.shape,
            _perturbation_tile_size(config, grid),
        ) > config.maximum_per_tile_whitened_norm:
            return "observation_perturbation_exceeds_tile_trust_radius"
    return None


def _candidate_full_precheck_reason(
    candidate: SparseRadarPerturbation,
    linearization: AnalysisLinearization,
    policy: AutomatedLearningPolicy,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None,
    neural_prior_application: NeuralPriorApplication | None,
) -> tuple[str | None, int]:
    counter = [0]
    try:
        dense = candidate.materialize(linearization.observations.dbz)
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            dense,
            linearization,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        with _count_observation_whitener_applies() as counter:
            _validate_variational_observation_perturbation(
                perturbation,
                linearization.observations,
                linearization.frozen,
                policy.adjoint_config,
                neural_prior_runner=neural_prior_runner,
                neural_prior_application=neural_prior_application,
            )
    except (TypeError, ValueError) as error:
        return str(error), counter[0]
    return None, counter[0]


def _sparse_maximum_tile_norm(
    flat_indices: Tensor,
    energy: Tensor,
    shape: torch.Size,
    tile_shape: TileShape,
) -> float:
    _, height, width = shape
    tile_height, tile_width = tile_shape
    tile_columns = math.ceil(width / tile_width)
    spatial = flat_indices % (height * width)
    frame = torch.div(flat_indices, height * width, rounding_mode="floor")
    row = torch.div(spatial, width, rounding_mode="floor")
    column = spatial % width
    tile = (
        frame * math.ceil(height / tile_height) * tile_columns
        + torch.div(row, tile_height, rounding_mode="floor") * tile_columns
        + torch.div(column, tile_width, rounding_mode="floor")
    )
    totals = energy.new_zeros(int(torch.amax(tile).detach()) + 1)
    totals.scatter_add_(0, tile, energy)
    return math.sqrt(float(torch.amax(totals).detach()))


def _candidate_ranking_score(
    prediction: Tensor,
    available: Tensor,
    scales: Tensor,
    weights: Tensor,
    objective: CandidateRankingObjective,
) -> float:
    normalized = prediction / scales[None, :]
    normalized = torch.where(available, normalized, 0.0)
    if objective == "expected_error_reduction":
        normalized = torch.clamp(-normalized, min=0.0)
    elif objective == "two_sided_diagnostic":
        benefit = torch.clamp(-normalized, min=0.0)
        harm = torch.clamp(normalized, min=0.0)
        benefit_norm = torch.sum(weights * benefit.square())
        harm_norm = torch.sum(weights * harm.square())
        return math.sqrt(float(torch.maximum(benefit_norm, harm_norm).detach()))
    elif objective != "absolute_influence":
        raise ValueError("unsupported candidate ranking objective")
    value = torch.where(available, weights * normalized.square(), 0.0).sum()
    return math.sqrt(float(value.detach()))


def validate_top_k_learning_impacts(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    ranking: VariationalCandidateRanking,
    *,
    policy: AutomatedLearningPolicy,
    policy_trust_store_path: str | Path,
    maximum_candidates_to_validate: int | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> tuple[RankedLearningOutcome, ...]:
    """Run full/half robust re-solves only for the highest-ranked candidates."""

    validate_variational_fso(ranking.fso)
    if ranking.ranking_digest != _variational_candidate_ranking_digest(ranking):
        raise ValueError("variational candidate ranking digest mismatch")
    if ranking.policy_digest != policy.digest:
        raise ValueError("candidate ranking policy mismatch")
    limit = (
        policy.maximum_learning_candidates_to_validate
        if maximum_candidates_to_validate is None
        else maximum_candidates_to_validate
    )
    if type(limit) is not int or limit <= 0:
        raise ValueError("maximum_candidates_to_validate must be positive")
    if limit > policy.maximum_learning_candidates_to_validate:
        raise ValueError("learning candidate count exceeds its policy budget")
    resolve_limit = policy.maximum_total_robust_resolves
    if 2 * limit > resolve_limit:
        raise ValueError("full/half robust resolves exceed their policy budget")
    trust_store = _load_learning_policy_trust_store(policy_trust_store_path)
    rejection = _learning_context_rejection(
        result,
        analysis,
        ranking.fso,
        verification_frames_dbz,
        policy,
        trust_store,
    )
    if rejection is not None:
        return tuple(
            RankedLearningOutcome(
                candidate_id=scored.candidate_id,
                candidate_rank=scored.rank,
                candidate_score=scored.score,
                ranking_digest=ranking.ranking_digest,
                result=_rejected_learning_impact(policy, rejection),
            )
            for scored in ranking.scores[:limit]
        )
    linearization = analysis.linearization
    if linearization is None:
        raise RuntimeError("approved learning context lacks a linearization")
    outcomes: list[RankedLearningOutcome] = []
    started = time.monotonic()
    robust_resolves = 0
    total_pcg_iterations = 0
    for scored in ranking.scores[:limit]:
        dense = scored.perturbation.materialize(linearization.observations.dbz)
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            dense,
            linearization,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        with _count_observation_whitener_applies() as fsoi_counter:
            fsoi = _variational_fsoi_from_precomputed_fso(
                ranking.fso,
                linearization,
                perturbation,
                policy.adjoint_config,
                neural_prior_runner=neural_prior_runner,
                neural_prior_application=neural_prior_application,
            )
        robust_resolves += 2
        result_for_candidate = _learning_impact_from_fsoi(
            result,
            analysis,
            verification_frames_dbz,
            fsoi,
            policy,
            trust_store.content_digest,
            selection=_LearningSelection(
                mode="ranked_top_k",
                candidate_id=scored.candidate_id,
                candidate_rank=scored.rank,
                candidate_score=scored.score,
                candidate_perturbation_digest=scored.perturbation.digest,
                ranking_digest=ranking.ranking_digest,
                ranking_policy_digest=ranking.policy_digest,
                ranking_objective=ranking.ranking_objective,
                observed_whitener_apply_count=(
                    ranking.observed_whitener_apply_count + fsoi_counter[0]
                ),
            ),
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        validation = result_for_candidate.first_order_validation
        if validation is not None:
            total_pcg_iterations += validation.total_resolved_pcg_iterations
        if total_pcg_iterations > policy.maximum_learning_pcg_iterations:
            raise ValueError("learning PCG iteration budget exhausted")
        if time.monotonic() - started > policy.maximum_learning_wall_seconds:
            raise ValueError("learning wall-time budget exhausted")
        outcomes.append(
            RankedLearningOutcome(
                candidate_id=scored.candidate_id,
                candidate_rank=scored.rank,
                candidate_score=scored.score,
                ranking_digest=ranking.ranking_digest,
                result=result_for_candidate,
            )
        )
    if robust_resolves > resolve_limit:
        raise RuntimeError("learning robust-resolve accounting failed")
    return tuple(outcomes)


def compute_variational_fsoi_for_learning(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    observation_perturbation: VariationalObservationPerturbation,
    *,
    policy: AutomatedLearningPolicy,
    policy_trust_store_path: str | Path,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalLearningImpact:
    """Compute frozen-domain FSOI under a root-owned learning policy."""

    trust_store = _load_learning_policy_trust_store(
        policy_trust_store_path
    )
    if policy.digest not in trust_store.approved_policy_digests:
        return _rejected_learning_impact(policy, "unapproved_learning_policy")
    if (
        observation_perturbation.perturbation_semantics
        != "physical_radar_value"
    ):
        return _rejected_learning_impact(
            policy,
            "physical_radar_perturbation_required",
        )
    linearization = analysis.linearization
    if linearization is None:
        return _rejected_learning_impact(policy, "linearization_required")
    if linearization.algorithm_bundle_digest != policy.algorithm_bundle_digest:
        return _rejected_learning_impact(
            policy,
            "algorithm_bundle_not_approved",
        )
    if linearization.numerical_runtime_digest != policy.numerical_runtime_digest:
        return _rejected_learning_impact(
            policy,
            "numerical_runtime_not_approved",
        )
    try:
        fsoi = compute_variational_fsoi(
            result,
            analysis,
            verification_frames_dbz,
            observation_perturbation,
            sensitivity_config=policy.sensitivity_config,
            adjoint_config=policy.adjoint_config,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
    except ValueError as error:
        return _rejected_learning_impact(policy, str(error))
    return _learning_impact_from_fsoi(
        result,
        analysis,
        verification_frames_dbz,
        fsoi,
        policy,
        trust_store.content_digest,
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )


@dataclass(frozen=True)
class _LearningSelection:
    mode: LearningSelectionMode
    candidate_id: str | None = None
    candidate_rank: int | None = None
    candidate_score: float | None = None
    candidate_perturbation_digest: str | None = None
    ranking_digest: str | None = None
    ranking_policy_digest: str | None = None
    ranking_objective: CandidateRankingObjective | None = None
    observed_whitener_apply_count: int = 0


def _learning_impact_from_fsoi(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    fsoi: VariationalFSOI,
    policy: AutomatedLearningPolicy,
    trust_store_digest: str,
    *,
    selection: _LearningSelection | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalLearningImpact:
    selection = selection or _LearningSelection(mode="direct")
    impact = fsoi.observation.baseline_branch_trusted_total
    if impact is None:
        return _rejected_learning_impact(
            policy,
            "baseline_dynamics_branch_not_certified",
            fsoi=fsoi,
        )
    base_whitener_applies = (
        selection.observed_whitener_apply_count
        if selection.mode == "ranked_top_k"
        else fsoi.fso.observed_whitener_apply_count
    )
    base_whitener_operations = (
        fsoi.fso.whitener_operations_per_apply * base_whitener_applies
    )
    remaining_whitener_operations = (
        policy.maximum_whitener_total_operations - base_whitener_operations
    )
    if remaining_whitener_operations <= 0:
        return _rejected_learning_impact(
            policy,
            "common_bias_total_operation_budget_exhausted",
            fsoi=fsoi,
        )
    validation_started = time.monotonic()
    try:
        validation = _validate_first_order_learning_impact(
            result,
            analysis,
            verification_frames_dbz,
            fsoi,
            policy,
            maximum_whitener_total_operations=(
                remaining_whitener_operations
            ),
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
    except ValueError as error:
        if str(error) != (
            "common-bias whitener total operation budget exhausted"
        ):
            raise
        return _rejected_learning_impact(
            policy,
            "common_bias_total_operation_budget_exhausted",
            fsoi=fsoi,
        )
    if (
        validation.source_fsoi_digest != fsoi.variational_fsoi_digest
        or validation.nominal_forecast_digest != result.forecast_run_digest
        or validation.nominal_input_bundle_digest
        != result.run.input_bundle_digest
        or validation.nominal_full_analysis_input_digest
        != result.run.full_analysis_input_digest
    ):
        raise ValueError("first-order validation lineage mismatch")
    if (
        validation.total_resolved_pcg_iterations
        > policy.maximum_learning_pcg_iterations
    ):
        return _rejected_learning_impact(
            policy,
            "learning_pcg_iteration_budget_exhausted",
            fsoi=fsoi,
            first_order_validation=validation,
        )
    if (
        time.monotonic() - validation_started
        > policy.maximum_learning_wall_seconds
    ):
        return _rejected_learning_impact(
            policy,
            "learning_wall_time_budget_exhausted",
            fsoi=fsoi,
            first_order_validation=validation,
        )
    total_whitener_applies = (
        base_whitener_applies + validation.observed_whitener_apply_count
    )
    total_whitener_operations = (
        fsoi.fso.whitener_operations_per_apply * total_whitener_applies
    )
    if total_whitener_operations > policy.maximum_whitener_total_operations:
        return _rejected_learning_impact(
            policy,
            "common_bias_total_operation_budget_exhausted",
            fsoi=fsoi,
            first_order_validation=validation,
        )
    if not validation.first_order_valid:
        no_material_signal = validation.material_metric_count == 0 and all(
            (
                validation.full_step_resolved_analysis_converged,
                validation.half_step_resolved_analysis_converged,
                validation.active_branch_valid,
                validation.full_step_valid,
                validation.half_step_valid,
                validation.sign_consistent_for_material_impacts,
            )
        )
        reason = (
            "no_material_learning_signal"
            if no_material_signal
            else "first_order_validation_failed"
        )
        return _rejected_learning_impact(
            policy,
            reason,
            fsoi=fsoi,
            first_order_validation=validation,
        )
    owned_impact = _clone_variational_impact_channel(impact)
    analysis_digests = (
        validation.full_step_analysis_digest,
        validation.half_step_analysis_digest,
    )
    forecast_digests = (
        validation.full_step_forecast_digest,
        validation.half_step_forecast_digest,
    )
    if any(value is None for value in (*analysis_digests, *forecast_digests)):
        raise RuntimeError("eligible learning validation lacks resolved digests")
    evidence = LearningApprovalEvidence(
        policy_digest=policy.digest,
        trust_store_digest=trust_store_digest,
        fsoi_digest=fsoi.variational_fsoi_digest,
        full_step_analysis_digest=cast(str, analysis_digests[0]),
        half_step_analysis_digest=cast(str, analysis_digests[1]),
        full_step_forecast_digest=cast(str, forecast_digests[0]),
        half_step_forecast_digest=cast(str, forecast_digests[1]),
        first_order_validation_digest=validation.validation_digest,
        learning_impact_digest=_variational_impact_digest(owned_impact),
        approved_action_digest=fsoi.perturbation_digest,
        nominal_input_bundle_digest=validation.nominal_input_bundle_digest,
        nominal_full_analysis_input_digest=(
            validation.nominal_full_analysis_input_digest
        ),
        selection_mode=selection.mode,
        candidate_id=selection.candidate_id,
        candidate_rank=selection.candidate_rank,
        candidate_score=selection.candidate_score,
        candidate_perturbation_digest=(
            selection.candidate_perturbation_digest
        ),
        ranking_digest=selection.ranking_digest,
        ranking_policy_digest=selection.ranking_policy_digest,
        ranking_objective=selection.ranking_objective,
        whitener_operations_per_apply=(
            fsoi.fso.whitener_operations_per_apply
        ),
        observed_whitener_apply_count=total_whitener_applies,
        observed_whitener_total_operations=total_whitener_operations,
    )
    return VariationalLearningImpact(
        eligibility=LearningEligibility(
            eligible=True,
            reasons=(),
            policy_digest=policy.digest,
        ),
        fsoi=fsoi,
        first_order_validation=validation,
        frozen_domain_learning_impact=owned_impact,
        approval_evidence=evidence,
    )


def _learning_context_rejection(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    fso: VariationalFSO,
    verification_input: VerificationInput,
    policy: AutomatedLearningPolicy,
    trust_store: _LearningPolicyTrustStore,
) -> str | None:
    if policy.digest not in trust_store.approved_policy_digests:
        return "unapproved_learning_policy"
    linearization = analysis.linearization
    if linearization is None:
        return "linearization_required"
    if fso.forecast_run_digest != result.forecast_run_digest:
        return "candidate_FSO_forecast_mismatch"
    if fso.linearization_digest != linearization.linearization_digest:
        return "candidate_FSO_linearization_mismatch"
    if fso.sensitivity_config_digest != policy.sensitivity_config.digest:
        return "candidate_FSO_sensitivity_policy_mismatch"
    if fso.adjoint_config_digest != policy.ranking_adjoint_config.digest:
        return "candidate_FSO_adjoint_policy_mismatch"
    if linearization.algorithm_bundle_digest != policy.algorithm_bundle_digest:
        return "algorithm_bundle_not_approved"
    if linearization.numerical_runtime_digest != policy.numerical_runtime_digest:
        return "numerical_runtime_not_approved"
    try:
        _validate_variational_fso_lineage(result, analysis, linearization)
        verification = _resolve_verification(
            verification_input,
            result,
            policy.sensitivity_config,
        )
    except ValueError as error:
        return str(error)
    if verification.content_digest != fso.verification_bundle_digest:
        return "candidate_FSO_verification_mismatch"
    return None


def _variational_fsoi_from_precomputed_fso(
    fso: VariationalFSO,
    linearization: AnalysisLinearization,
    perturbation: VariationalObservationPerturbation,
    adjoint_config: VariationalAdjointConfig,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalFSOI:
    """Materialize one impact from retained maps without solving another adjoint."""

    if fso.linearization_digest != linearization.linearization_digest:
        raise ValueError("precomputed FSO linearization mismatch")
    ranking_config = replace(
        adjoint_config,
        require_baseline_dynamics_branch_validity=False,
    )
    if fso.adjoint_config_digest != ranking_config.digest:
        raise ValueError("precomputed FSO adjoint config mismatch")
    if fso.lead_minutes != fso.full_map_lead_minutes:
        raise ValueError("precomputed FSO does not retain every lead map")
    diagnostics = _validate_variational_observation_perturbation(
        perturbation,
        linearization.observations,
        linearization.frozen,
        adjoint_config,
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )
    if (
        adjoint_config.require_baseline_dynamics_branch_validity
        and diagnostics.baseline_dynamics_branch_status
        not in ("not_applicable", "certified")
    ):
        raise ValueError("P1 FSOI baseline dynamics branch is not certified")
    observations = linearization.observations
    component_pairs = (
        (fso.observation.detected_dbz, perturbation.detected_dbz),
        (
            fso.observation.censor_threshold_dbz,
            perturbation.censor_threshold_dbz,
        ),
        (fso.observation.observation_weight, perturbation.observation_weight),
        (
            fso.observation.initial_background_dbz,
            _initial_background_perturbation(perturbation, observations),
        ),
        (
            fso.observation.baseline_dynamics_dbz,
            _baseline_dynamics_perturbation(perturbation, observations),
        ),
    )
    components = tuple(
        _impact_from_precomputed_sensitivity(channel, delta, fso.tile_shape_yx)
        for channel, delta in component_pairs
    )
    total = _sum_impact_channels(components)
    trusted = diagnostics.baseline_dynamics_branch_status in (
        "not_applicable",
        "certified",
    )
    observation = VariationalObservationImpact(
        detected_dbz=components[0],
        censor_threshold_dbz=components[1],
        observation_weight=components[2],
        initial_background_dbz=components[3],
        baseline_dynamics_dbz=components[4],
        total=total,
        baseline_branch_trusted_total=total if trusted else None,
    )
    fsoi = VariationalFSOI(
        contract="p1-linearized-observation-impact-v13",
        fso=fso,
        perturbation=perturbation,
        perturbation_contract=perturbation.contract,
        perturbation_digest=perturbation.digest,
        perturbation_diagnostics=diagnostics,
        baseline_dynamics_branch_status=(
            diagnostics.baseline_dynamics_branch_status
        ),
        observation=observation,
        variational_fsoi_digest="",
    )
    return replace(
        fsoi,
        variational_fsoi_digest=variational_fsoi_digest(fsoi),
    )


def _impact_from_precomputed_sensitivity(
    sensitivity: VariationalSensitivityChannel,
    delta: Tensor,
    tile_size: TileShape,
) -> VariationalImpactChannel:
    maps = sensitivity.maps * delta
    if maps.ndim != 5 or maps.shape[2] != 3:
        raise ValueError("precomputed sensitivity maps have an invalid shape")
    lead_count, metric_count, _, height, width = maps.shape
    accumulator = _new_variational_channel_accumulator(
        maps,
        selected_count=lead_count,
        lead_count=lead_count,
        metric_count=metric_count,
        height=height,
        width=width,
        tile_rows=math.ceil(height / tile_size[0]),
        tile_columns=math.ceil(width / tile_size[1]),
    )
    for lead in range(lead_count):
        for metric in range(metric_count):
            _record_variational_channel(
                accumulator,
                maps[lead, metric],
                lead_index=lead,
                metric_index=metric,
                selected_index=lead,
                tile_size=tile_size,
                signed_sum=True,
            )
    return _impact_channel(accumulator)


def _sum_impact_channels(
    channels: tuple[VariationalImpactChannel, ...],
) -> VariationalImpactChannel:
    if not channels:
        raise ValueError("at least one impact channel is required")
    return VariationalImpactChannel(
        maps=sum(
            (channel.maps for channel in channels),
            torch.zeros_like(channels[0].maps),
        ),
        sum_by_time=sum(
            (channel.sum_by_time for channel in channels),
            torch.zeros_like(channels[0].sum_by_time),
        ),
        tile_sum_by_time=sum(
            (channel.tile_sum_by_time for channel in channels),
            torch.zeros_like(channels[0].tile_sum_by_time),
        ),
    )


def _rejected_learning_impact(
    policy: AutomatedLearningPolicy,
    reason: str,
    *,
    fsoi: VariationalFSOI | None = None,
    first_order_validation: FirstOrderValidation | None = None,
) -> VariationalLearningImpact:
    return VariationalLearningImpact(
        eligibility=LearningEligibility(
            eligible=False,
            reasons=(reason,),
            policy_digest=policy.digest,
        ),
        fsoi=fsoi,
        first_order_validation=first_order_validation,
        frozen_domain_learning_impact=None,
    )


def _validate_first_order_learning_impact(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_input: VerificationInput,
    fsoi: VariationalFSOI,
    policy: AutomatedLearningPolicy,
    *,
    metric_domain_contract: FirstOrderMetricDomain = "frozen_metric_domain",
    maximum_whitener_total_operations: int | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> FirstOrderValidation:
    """Re-solve full and half perturbations on one explicit metric domain."""

    full_prediction = (
        fsoi.observation.total.sum_by_time.sum(dim=-1).detach()
    )
    half_prediction = 0.5 * full_prediction
    operations_per_apply = _analysis_whitener_operations_per_apply(analysis)
    with _count_observation_whitener_applies(
        operations_per_apply=operations_per_apply,
        maximum_total_operations=maximum_whitener_total_operations,
    ) as whitener_counter:
        full = _resolve_learning_step(
            result,
            analysis,
            verification_input,
            fsoi,
            policy,
            scale=1.0,
            metric_domain_contract=metric_domain_contract,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        half = _resolve_learning_step(
            result,
            analysis,
            verification_input,
            fsoi,
            policy,
            scale=0.5,
            metric_domain_contract=metric_domain_contract,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
    available = fsoi.fso.metric_available
    full_error = torch.abs(full.metric_change - full_prediction)
    half_error = torch.abs(half.metric_change - half_prediction)
    full_step_valid = _taylor_step_is_valid(
        full_prediction,
        full.metric_change,
        available,
        policy,
    )
    half_step_valid = _taylor_step_is_valid(
        half_prediction,
        half.metric_change,
        available,
        policy,
    )
    sign_consistent = _material_impact_signs_are_consistent(
        (
            (full_prediction, full.metric_change),
            (half_prediction, half.metric_change),
        ),
        available,
        fsoi.fso.metric_names,
        policy,
    )
    material_count, maximum_material, aggregate_material = (
        _material_impact_summary(
            (
                (full_prediction, full.metric_change),
                (half_prediction, half.metric_change),
            ),
            available,
            fsoi.fso.metric_names,
            policy,
        )
    )
    branch_valid = full.active_branch_valid and half.active_branch_valid
    first_order_valid = (
        full.analysis_converged
        and half.analysis_converged
        and branch_valid
        and full_step_valid
        and half_step_valid
        and sign_consistent
        and material_count > 0
    )
    if result.run.full_analysis_input_digest is None:
        raise ValueError("learning validation requires full input identity")
    return FirstOrderValidation(
        source_fsoi_digest=fsoi.variational_fsoi_digest,
        nominal_forecast_digest=result.forecast_run_digest,
        nominal_input_bundle_digest=result.run.input_bundle_digest,
        nominal_full_analysis_input_digest=(
            result.run.full_analysis_input_digest
        ),
        full_step_prediction=full_prediction,
        full_step_resolved_metric_change=full.metric_change,
        full_step_absolute_error=full_error,
        half_step_prediction=half_prediction,
        half_step_resolved_metric_change=half.metric_change,
        half_step_absolute_error=half_error,
        metric_available=available.detach().clone(),
        full_step_resolved_analysis_converged=full.analysis_converged,
        half_step_resolved_analysis_converged=half.analysis_converged,
        active_branch_valid=branch_valid,
        full_step_valid=full_step_valid,
        half_step_valid=half_step_valid,
        sign_consistent_for_material_impacts=sign_consistent,
        material_metric_count=material_count,
        maximum_material_impact=maximum_material,
        aggregate_material_impact_norm=aggregate_material,
        first_order_valid=first_order_valid,
        full_step_analysis_digest=full.analysis_digest,
        half_step_analysis_digest=half.analysis_digest,
        full_step_forecast_digest=full.forecast_digest,
        half_step_forecast_digest=half.forecast_digest,
        full_step_input_bundle_digest=full.input_bundle_digest,
        half_step_input_bundle_digest=half.input_bundle_digest,
        full_step_pcg_iterations=full.pcg_iterations,
        half_step_pcg_iterations=half.pcg_iterations,
        observed_whitener_apply_count=whitener_counter[0],
        frozen_domain_state_effect=full.frozen_domain_state_effect,
        issuance_policy_effect=full.issuance_policy_effect,
        end_to_end_issuance_effect=full.end_to_end_issuance_effect,
        coverage_before=full.coverage_before,
        coverage_after=full.coverage_after,
        newly_issued_fraction=full.newly_issued_fraction,
        withdrawn_fraction=full.withdrawn_fraction,
        background_fallback_before=full.background_fallback_before,
        background_fallback_after=full.background_fallback_after,
        metric_domain_contract=metric_domain_contract,
    )


@dataclass(frozen=True)
class _ResolvedLearningStep:
    metric_change: Tensor
    frozen_domain_state_effect: Tensor | None
    issuance_policy_effect: Tensor | None
    end_to_end_issuance_effect: Tensor | None
    coverage_before: Tensor | None
    coverage_after: Tensor | None
    newly_issued_fraction: Tensor | None
    withdrawn_fraction: Tensor | None
    background_fallback_before: Tensor | None
    background_fallback_after: Tensor | None
    analysis_converged: bool
    active_branch_valid: bool
    analysis_digest: str | None
    forecast_digest: str | None
    input_bundle_digest: str | None
    pcg_iterations: int


def _resolve_learning_step(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_input: VerificationInput,
    fsoi: VariationalFSOI,
    policy: AutomatedLearningPolicy,
    *,
    scale: float,
    metric_domain_contract: FirstOrderMetricDomain,
    neural_prior_runner: NeuralPriorInferenceRunner | None,
    neural_prior_application: NeuralPriorApplication | None,
) -> _ResolvedLearningStep:
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("first-order validation requires a linearization")
    perturbation = fsoi.perturbation
    physical_delta = perturbation.physical_radar_dbz_delta
    if physical_delta is None:
        raise ValueError("first-order validation requires a physical delta")
    delta = scale * physical_delta
    observations = linearization.observations
    frozen = linearization.frozen
    changed_frames = frozen.input_frames_dbz + delta
    changed_prior: NeuralPriorApplication | None = None
    if frozen.neural_prior_dependency is not None:
        if neural_prior_runner is None:
            raise ValueError("neural-prior re-solve requires an inference runner")
        _validate_retained_prior_runner(
            frozen,
            neural_prior_runner,
            neural_prior_application,
        )
        preliminary_run = ForecastRunContract.from_inputs(
            result.run.config,
            changed_frames,
            observations.valid_mask,
            frozen.background_frames_dbz,
            frozen.background_age_minutes,
            observation_quality_weight=(
                observations.quality_weight * observations.valid_mask
            ),
            observation_std_dbz=observations.std_dbz,
            grid_time_contract=frozen.grid_time_contract,
            analysis_config_json=result.run.analysis_config_json,
            analysis_config_digest=result.run.analysis_config_digest,
            analysis_input_digest=result.run.analysis_input_digest,
            operational_calibration_manifest_json=(
                result.run.operational_calibration_manifest_json
            ),
            operational_calibration_manifest_digest=(
                result.run.operational_calibration_manifest_digest
            ),
            operational_calibration_approval_digest=(
                result.run.operational_calibration_approval_digest
            ),
            operational_data_identity_json=(result.run.operational_data_identity_json),
            operational_data_identity_digest=(
                result.run.operational_data_identity_digest
            ),
            input_plan_json=result.run.input_plan_json,
            input_plan_digest=result.run.input_plan_digest,
        )
        changed_prior = neural_prior_runner.infer(
            changed_frames,
            input_run=preliminary_run,
            role=cast(Literal["candidate", "parent"], frozen.neural_prior_role),
        )
        retained_raw = frozen.neural_prior_raw_background_dbz
        assert retained_raw is not None
        if frozen.neural_prior_dependency == "exogenous" and not torch.equal(
            changed_prior.initial_background_dbz,
            retained_raw,
        ):
            raise ValueError("exogenous neural prior changed with the radar input")
        changed_observations, changed_frozen = prepare_analysis(
            changed_frames,
            nowcast_config=result.run.config,
            analysis_config=frozen.analysis_config,
            observation_std_dbz=observations.std_dbz,
            quality_weight=observations.quality_weight,
            qc_mask=observations.valid_mask,
            observation_common_bias_group_index=(observations.common_bias_group_index),
            observation_common_bias_mode_weights=(
                observations.common_bias_mode_weights
            ),
            background_frames_dbz=frozen.background_frames_dbz,
            background_age_minutes=frozen.background_age_minutes,
            grid_time_contract=frozen.grid_time_contract,
            neural_prior=changed_prior,
        )
    else:
        changed_observations = replace(observations, dbz=observations.dbz + delta)
        background_delta = (
            scale * _initial_background_perturbation(perturbation, observations)[0]
        )
        baseline_dynamics = torch.cat(
            (
                frozen.baseline_state.displacement_yx,
                frozen.baseline_state.log_growth_per_step.reshape(1),
            )
        )
        if frozen.baseline_metadata.tendency_source is TendencySource.OBSERVATION:
            baseline_dynamics = _baseline_dynamics_from_observation(
                changed_observations.dbz,
                frozen,
            )
        changed_frozen = replace(
            frozen,
            initial_background_dbz=(frozen.initial_background_dbz + background_delta),
            baseline_state=RadarState(
                echo_linear=frozen.baseline_state.echo_linear,
                displacement_yx=baseline_dynamics[:2],
                log_growth_per_step=baseline_dynamics[2],
            ),
            baseline_frames_dbz=frozen.baseline_frames_dbz + delta,
        )
    if not torch.equal(
        changed_frozen.active_field_index,
        frozen.active_field_index,
    ):
        metric_change = torch.full_like(fsoi.fso.forecast_scores, float("nan"))
        return _ResolvedLearningStep(
            metric_change=metric_change,
            frozen_domain_state_effect=None,
            issuance_policy_effect=None,
            end_to_end_issuance_effect=None,
            coverage_before=None,
            coverage_after=None,
            newly_issued_fraction=None,
            withdrawn_fraction=None,
            background_fallback_before=None,
            background_fallback_after=None,
            analysis_converged=False,
            active_branch_valid=False,
            analysis_digest=None,
            forecast_digest=None,
            input_bundle_digest=None,
            pcg_iterations=0,
        )
    resolved = solve_analysis(
        changed_observations,
        changed_frozen,
        control=analysis.control,
    )
    resolved_linearization = resolved.linearization
    converged = (
        resolved.converged
        and not resolved.used_fallback
        and not resolved.degraded
        and resolved.p1_forecast_eligible
        and resolved_linearization is not None
    )
    metric_change = torch.full_like(
        fsoi.fso.forecast_scores,
        float("nan"),
    )
    resolved_domain = metric_domain_contract == "resolved_issuance_domain"
    state_effect = torch.full_like(metric_change, float("nan")) if resolved_domain else None
    policy_effect = torch.full_like(metric_change, float("nan")) if resolved_domain else None
    total_effect = torch.full_like(metric_change, float("nan")) if resolved_domain else None
    coverage_before = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    coverage_after = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    newly_issued = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    withdrawn = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    fallback_before = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    fallback_after = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    branch_valid = converged and resolved_linearization is not None
    if resolved_linearization is not None:
        branch_valid = (
            branch_valid
            and resolved_linearization.frozen.analysis_remap_cells
            == frozen.analysis_remap_cells
        )
        if frozen.neural_prior_dependency is not None:
            resolved_prior_valid = (
                resolved_linearization.frozen.neural_prior_valid_mask
            )
            retained_prior_valid = frozen.neural_prior_valid_mask
            branch_valid = (
                branch_valid
                and resolved_prior_valid is not None
                and retained_prior_valid is not None
                and torch.equal(resolved_prior_valid, retained_prior_valid)
            )
    if not converged:
        return _ResolvedLearningStep(
            metric_change=metric_change,
            frozen_domain_state_effect=state_effect,
            issuance_policy_effect=policy_effect,
            end_to_end_issuance_effect=total_effect,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            newly_issued_fraction=newly_issued,
            withdrawn_fraction=withdrawn,
            background_fallback_before=fallback_before,
            background_fallback_after=fallback_after,
            analysis_converged=False,
            active_branch_valid=branch_valid,
            analysis_digest=None,
            forecast_digest=None,
            input_bundle_digest=None,
            pcg_iterations=resolved.pcg_iterations,
        )
    if resolved_linearization is None:
        raise RuntimeError("converged learning re-solve lacks linearization")
    resolved_forecast = None
    resolved_run = None
    if metric_domain_contract == "resolved_issuance_domain":
        resolved_analysis_config_json = result.run.analysis_config_json
        resolved_analysis_config_digest = result.run.analysis_config_digest
        resolved_analysis_input_digest = result.run.analysis_input_digest
        if changed_prior is not None:
            (
                resolved_analysis_config_json,
                resolved_analysis_config_digest,
                resolved_analysis_input_digest,
            ) = _analysis_input_lineage(
                changed_observations,
                changed_frozen.analysis_config,
                neural_prior_application_digest=(changed_prior.application_digest),
            )
        resolved_run = ForecastRunContract.from_inputs(
            result.run.config,
            changed_frames,
            changed_observations.valid_mask,
            frozen.background_frames_dbz,
            frozen.background_age_minutes,
            observation_quality_weight=(
                changed_observations.quality_weight
                * changed_observations.valid_mask
            ),
            observation_std_dbz=changed_observations.std_dbz,
            grid_time_contract=frozen.grid_time_contract,
            analysis_config_json=resolved_analysis_config_json,
            analysis_config_digest=resolved_analysis_config_digest,
            analysis_input_digest=resolved_analysis_input_digest,
            operational_calibration_manifest_json=(
                result.run.operational_calibration_manifest_json
            ),
            operational_calibration_manifest_digest=(
                result.run.operational_calibration_manifest_digest
            ),
            operational_calibration_approval_digest=(
                result.run.operational_calibration_approval_digest
            ),
            operational_data_identity_json=(
                result.run.operational_data_identity_json
            ),
            operational_data_identity_digest=(
                result.run.operational_data_identity_digest
            ),
            neural_prior_digest=(
                None if changed_prior is None else changed_prior.neural_prior_digest
            ),
            prior_application_digest=(
                None if changed_prior is None else changed_prior.application_digest
            ),
            prior_model_contract_digest=(
                None if changed_prior is None else changed_prior.model_contract_digest
            ),
            prior_feature_schema_digest=(
                None if changed_prior is None else changed_prior.feature_schema_digest
            ),
            prior_training_manifest_digest=(
                None
                if changed_prior is None
                else changed_prior.training_manifest_digest
            ),
            prior_inference_evidence_digest=(
                None
                if changed_prior is None
                else changed_prior.inference_evidence.evidence_digest
            ),
            prior_inference_algorithm_digest=(
                None
                if changed_prior is None
                else changed_prior.inference_evidence.inference_algorithm_digest
            ),
            prior_numerical_runtime_digest=(
                None
                if changed_prior is None
                else changed_prior.inference_evidence.numerical_runtime_digest
            ),
            prior_dependency=(
                None if changed_prior is None else changed_prior.dependency
            ),
            prior_role=None if changed_prior is None else changed_prior.role,
            input_plan_json=result.run.input_plan_json,
            input_plan_digest=result.run.input_plan_digest,
        )
        resolved_forecast = forecast_from_state(
            resolved.state,
            resolved.metadata,
            result.run.config,
            run=resolved_run,
        )
        resolved_forecast.validate_issuance()

    verification = _resolve_verification(
        verification_input,
        result,
        policy.sensitivity_config,
    )
    config = result.run.config
    clean_truth = torch.nan_to_num(
        verification.frames_dbz,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    truth_linear = dbz_to_echo(
        clean_truth,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    finite_truth = verification.valid_mask & torch.isfinite(
        verification.frames_dbz
    )
    available = fsoi.fso.metric_available
    changed_forecasts: list[dict[str, str | int]] = []
    for lead_index, minutes in enumerate(fsoi.fso.lead_minutes):
        step = minutes // config.interval_minutes
        forecast_index = step - 1
        if freeze_remap_cell(step * resolved.state.displacement_yx) != (
            freeze_remap_cell(step * analysis.state.displacement_yx)
        ):
            branch_valid = False
        nominal_linear = forecast_linear_at_step(analysis.state, step, config)
        changed_linear = forecast_linear_at_step(resolved.state, step, config)
        nominal_capped, nominal_cap_active = _freeze_output_cap(
            nominal_linear,
            config,
        )
        changed_capped, changed_cap_active = _freeze_output_cap(
            changed_linear,
            config,
        )
        if not torch.equal(nominal_cap_active, changed_cap_active):
            branch_valid = False
        changed_forecasts.append(
            {
                "lead_minutes": minutes,
                "forecast_digest": tensor_digest(changed_capped),
            }
        )
        nominal_weight = _metric_domain_weight(
            result,
            finite_truth[forecast_index],
            forecast_index,
            policy.sensitivity_config.metric_domain,
        )
        changed_weight = (
            nominal_weight
            if not resolved_domain
            else _metric_domain_weight(
                cast(ForecastResult, resolved_forecast),
                finite_truth[forecast_index],
                forecast_index,
                policy.sensitivity_config.metric_domain,
            )
        )
        if resolved_domain:
            assert coverage_before is not None
            assert coverage_after is not None
            assert newly_issued is not None
            assert withdrawn is not None
            assert fallback_before is not None
            assert fallback_after is not None
            valid_count = torch.count_nonzero(finite_truth[forecast_index]).clamp_min(1)
            nominal_support = nominal_weight > 0
            changed_support = changed_weight > 0
            coverage_before[lead_index] = torch.count_nonzero(nominal_support).to(metric_change) / valid_count
            coverage_after[lead_index] = torch.count_nonzero(changed_support).to(metric_change) / valid_count
            newly_issued[lead_index] = torch.count_nonzero(changed_support & ~nominal_support).to(metric_change) / valid_count
            withdrawn[lead_index] = torch.count_nonzero(nominal_support & ~changed_support).to(metric_change) / valid_count
            nominal_fallback = result.background_fallback_mask[forecast_index]
            changed_fallback = cast(
                ForecastResult, resolved_forecast
            ).background_fallback_mask[forecast_index]
            fallback_before[lead_index] = torch.count_nonzero(
                nominal_fallback & finite_truth[forecast_index]
            ).to(metric_change) / valid_count
            fallback_after[lead_index] = torch.count_nonzero(
                changed_fallback & finite_truth[forecast_index]
            ).to(metric_change) / valid_count
        for metric_index, metric_name in enumerate(fsoi.fso.metric_names):
            if not bool(available[lead_index, metric_index]):
                continue
            nominal_score = forecast_metric(
                metric_name,
                nominal_capped,
                truth_linear[forecast_index],
                nominal_weight,
                config,
                policy.sensitivity_config,
                frozen.grid_time_contract,
            )
            if not torch.allclose(
                nominal_score,
                fsoi.fso.forecast_scores[lead_index, metric_index],
                rtol=0.0,
                atol=config.contract_absolute_tolerance,
            ):
                raise ValueError(
                    "first-order validation does not reproduce the nominal metric"
                )
            changed_nominal_domain_score = forecast_metric(
                metric_name,
                changed_capped,
                truth_linear[forecast_index],
                nominal_weight,
                config,
                policy.sensitivity_config,
                frozen.grid_time_contract,
            )
            if resolved_domain:
                changed_resolved_domain_score = forecast_metric(
                    metric_name,
                    changed_capped,
                    truth_linear[forecast_index],
                    changed_weight,
                    config,
                    policy.sensitivity_config,
                    frozen.grid_time_contract,
                )
                assert state_effect is not None
                assert policy_effect is not None
                assert total_effect is not None
                state_effect[lead_index, metric_index] = (
                    changed_nominal_domain_score - nominal_score
                ).detach()
                policy_effect[lead_index, metric_index] = (
                    changed_resolved_domain_score - changed_nominal_domain_score
                ).detach()
                total_effect[lead_index, metric_index] = (
                    changed_resolved_domain_score - nominal_score
                ).detach()
                metric_change[lead_index, metric_index] = total_effect[
                    lead_index, metric_index
                ]
            else:
                metric_change[lead_index, metric_index] = (
                    changed_nominal_domain_score - nominal_score
                ).detach()
    return _ResolvedLearningStep(
        metric_change=metric_change,
        frozen_domain_state_effect=state_effect,
        issuance_policy_effect=policy_effect,
        end_to_end_issuance_effect=total_effect,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        newly_issued_fraction=newly_issued,
        withdrawn_fraction=withdrawn,
        background_fallback_before=fallback_before,
        background_fallback_after=fallback_after,
        analysis_converged=converged,
        active_branch_valid=branch_valid,
        analysis_digest=resolved_linearization.linearization_digest,
        forecast_digest=(
            cast(ForecastResult, resolved_forecast).forecast_run_digest
            if metric_domain_contract == "resolved_issuance_domain"
            else json_digest(
                {
                    "contract": "p1-resolved-learning-forecast-v1",
                    "forecasts": changed_forecasts,
                }
            )
        ),
        input_bundle_digest=(
            None if resolved_run is None else resolved_run.input_bundle_digest
        ),
        pcg_iterations=resolved.pcg_iterations,
    )


def _taylor_step_is_valid(
    prediction: Tensor,
    actual: Tensor,
    available: Tensor,
    policy: AutomatedLearningPolicy,
) -> bool:
    error = torch.abs(actual - prediction)
    scale = torch.maximum(torch.abs(actual), torch.abs(prediction))
    absolute_error = prediction.new_tensor(
        tuple(
            policy.threshold_for(name).maximum_absolute_error
            for name in policy.sensitivity_config.metric_names
        )
    )
    tolerance = (
        absolute_error
        + policy.maximum_linearity_relative_error * scale
    )
    selected = error.masked_select(available)
    selected_tolerance = tolerance.masked_select(available)
    return (
        selected.numel() > 0
        and bool(torch.all(torch.isfinite(selected)))
        and bool(torch.all(selected <= selected_tolerance))
    )


def _material_impact_signs_are_consistent(
    steps: tuple[tuple[Tensor, Tensor], ...],
    available: Tensor,
    metric_names: tuple[str, ...],
    policy: AutomatedLearningPolicy,
) -> bool:
    materiality = available.new_tensor(
        tuple(
            policy.threshold_for(name).material_impact_threshold
            for name in metric_names
        ),
        dtype=steps[0][0].dtype,
    )
    for prediction, actual in steps:
        material = available & (
            torch.maximum(torch.abs(prediction), torch.abs(actual))
            >= materiality
        )
        if bool(torch.any(material & (prediction * actual <= 0.0))):
            return False
    return True


def _material_impact_summary(
    steps: tuple[tuple[Tensor, Tensor], ...],
    available: Tensor,
    metric_names: tuple[str, ...],
    policy: AutomatedLearningPolicy,
) -> tuple[int, float, float]:
    magnitudes = torch.stack(
        tuple(
            torch.maximum(torch.abs(prediction), torch.abs(actual))
            for prediction, actual in steps
        )
    ).amax(dim=0)
    thresholds = magnitudes.new_tensor(
        tuple(
            policy.threshold_for(name).material_impact_threshold
            for name in metric_names
        )
    )
    selected = magnitudes.masked_select(
        available & (magnitudes >= thresholds)
    )
    if selected.numel() == 0:
        return 0, 0.0, 0.0
    return (
        int(selected.numel()),
        float(torch.amax(selected).detach()),
        float(torch.linalg.vector_norm(selected).detach()),
    )


def _compute_variational_products(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    *,
    sensitivity_config: SensitivityConfig | None,
    adjoint_config: VariationalAdjointConfig | None,
    observation_perturbation: VariationalObservationPerturbation | None,
    neural_prior_runner: NeuralPriorInferenceRunner | None,
    neural_prior_application: NeuralPriorApplication | None,
) -> tuple[
    VariationalFSO,
    VariationalObservationImpact | None,
    VariationalPerturbationDiagnostics | None,
]:
    """Compute frozen-final P1 forecast sensitivity to observations.

    The final IRLS weights, remap cells, active controls, and observation
    classification are held fixed. Each forecast-metric adjoint uses the same
    matrix-free Gauss--Newton normal operator as the accepted analysis.
    """

    sensitivity_config = sensitivity_config or SensitivityConfig()
    adjoint_config = adjoint_config or VariationalAdjointConfig()
    if observation_perturbation is not None:
        adjoint_config = replace(
            adjoint_config,
            require_active_set_margin=True,
            require_feasibility_margin=True,
            require_gauss_newton_reliability=True,
        )
    result.validate_issuance()
    verification_bundle = _resolve_verification(
        verification_frames_dbz,
        result,
        sensitivity_config,
    )
    verification_frames = verification_bundle.frames_dbz
    if result.metadata.dynamics_source is not DynamicsSource.P1_VARIATIONAL:
        raise ValueError("variational FSO requires an accepted P1 forecast")
    if (
        analysis.used_fallback
        or not analysis.converged
        or analysis.degraded
        or not analysis.final_linearization_stationary
        or not analysis.final_robust_stationary
        or not analysis.final_irls_fixed_point
        or not analysis.p1_forecast_eligible
        or not analysis.posterior_eligible
        or not analysis.fso_eligible
    ):
        raise ValueError("variational FSO requires a converged P1 analysis")
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("P1 analysis does not retain a final linearization")
    if linearization.contract != P1_LINEARIZATION_CONTRACT:
        raise ValueError("unsupported P1 linearization contract")
    _validate_variational_fso_lineage(result, analysis, linearization)
    feasibility_margins = _variational_feasibility_margins(
        linearization.feasibility_margins,
        adjoint_config,
    )
    if (
        adjoint_config.require_feasibility_margin
        and feasibility_margins.low_interior_validity
    ):
        raise ValueError(
            "P1 FSO feasibility margin is below its requirement"
        )

    nowcast_config = result.run.config
    observations = linearization.observations
    frozen = linearization.frozen
    control = analysis.control
    validated_prior_input = None
    if frozen.neural_prior_dependency == "radar_dependent":
        if neural_prior_runner is None:
            raise ValueError("radar-dependent prior FSO requires an inference runner")
        _validate_retained_prior_runner(
            frozen,
            neural_prior_runner,
            neural_prior_application,
        )
        validated_prior_input = neural_prior_runner.validated_bound_input(
            _require_bound_neural_prior_input(frozen)
        )
    perturbation_diagnostics = None
    if observation_perturbation is not None:
        perturbation_diagnostics = _validate_variational_observation_perturbation(
            observation_perturbation,
            observations,
            frozen,
            adjoint_config,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        if (
            adjoint_config.require_baseline_dynamics_branch_validity
            and perturbation_diagnostics.baseline_dynamics_branch_status
            not in ("not_applicable", "certified")
        ):
            raise ValueError(
                "P1 FSOI baseline dynamics branch is not certified"
            )
    _validate_inputs(
        observations.dbz[-1],
        verification_frames,
        analysis.state,
        nowcast_config,
        None,
    )
    height, width = analysis.state.echo_linear.shape
    all_lead_minutes = tuple(
        range(
            nowcast_config.interval_minutes,
            nowcast_config.horizon_minutes + 1,
            nowcast_config.interval_minutes,
        )
    )
    forecast_indices = _adjoint_lead_indices(
        adjoint_config,
        all_lead_minutes,
    )
    lead_minutes = tuple(
        all_lead_minutes[index] for index in forecast_indices
    )
    full_map_indices = _full_map_indices(
        sensitivity_config.full_map_lead_minutes,
        lead_minutes,
    )
    selected_position = {
        index: position for position, index in enumerate(full_map_indices)
    }
    lead_count = len(lead_minutes)
    metric_count = len(sensitivity_config.metric_names)
    tile_shape_yx = _metric_tile_shape(
        sensitivity_config,
        frozen.grid_time_contract,
    )
    tile_rows = math.ceil(height / tile_shape_yx[0])
    tile_columns = math.ceil(width / tile_shape_yx[1])
    selected_count = len(full_map_indices)
    materialized_output_bytes = _variational_materialized_output_bytes(
        control,
        selected_count=selected_count,
        lead_count=lead_count,
        metric_count=metric_count,
        height=height,
        width=width,
        tile_rows=tile_rows,
        tile_columns=tile_columns,
        include_impact=observation_perturbation is not None,
        gauss_newton_probe_count=adjoint_config.gauss_newton_probe_count,
    )
    if (
        materialized_output_bytes
        > adjoint_config.maximum_materialized_output_bytes
    ):
        raise ValueError(
            "P1 FSO materialized output exceeds its byte budget"
        )
    score_shape = (lead_count, metric_count)
    forecast_scores = control.new_full(score_shape, float("nan"))
    metric_available = torch.zeros(
        score_shape,
        dtype=torch.bool,
        device=control.device,
    )
    channel_shape = {
        "selected_count": selected_count,
        "lead_count": lead_count,
        "metric_count": metric_count,
        "height": height,
        "width": width,
        "tile_rows": tile_rows,
        "tile_columns": tile_columns,
    }
    detected_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    censor_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    weight_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    initial_background_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    baseline_dynamics_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    frozen_structure_input_sensitivity = (
        _new_variational_channel_accumulator(
            control,
            **channel_shape,
        )
    )
    impact_accumulators: tuple[
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
    ] | None = None
    if observation_perturbation is not None:
        impact_accumulators = (
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
        )
    selected_cap_masks = torch.zeros(
        (selected_count, height, width),
        dtype=torch.bool,
        device=control.device,
    )
    adjoint_iterations = torch.zeros(
        score_shape,
        dtype=torch.int64,
        device=control.device,
    )
    adjoint_relative_residual = control.new_full(
        score_shape,
        float("nan"),
    )
    adjoint_true_residual_norm = control.new_full(
        score_shape,
        float("nan"),
    )
    adjoint_normal_products = torch.zeros(
        score_shape,
        dtype=torch.int64,
        device=control.device,
    )
    adjoint_warm_started = torch.zeros(
        score_shape,
        dtype=torch.bool,
        device=control.device,
    )
    neural_prior_adjoint_direction_maximum_defect = 0.0

    clean_verification = torch.nan_to_num(
        verification_frames,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    )
    verification_finite = verification_bundle.valid_mask
    if verification_finite.shape != result.valid_mask.shape:
        raise ValueError("verification frames must match the forecast shape")
    metric_domain_weights = tuple(
        _metric_domain_weight(
            result,
            verification_finite[forecast_index],
            forecast_index,
            sensitivity_config.metric_domain,
        )
        for forecast_index in forecast_indices
    )
    metric_domain_weight_sum = torch.stack(
        tuple(weight.sum() for weight in metric_domain_weights)
    )
    metric_domain_weight_fraction = metric_domain_weight_sum / float(
        height * width
    )
    metric_domain_digest = _metric_domain_digest(
        sensitivity_config.metric_domain,
        lead_minutes,
        metric_domain_weights,
    )
    truth_linear = dbz_to_echo(
        clean_verification,
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    issued_echo = dbz_to_echo(
        torch.nan_to_num(
            result.forecast_dbz,
            nan=nowcast_config.min_dbz,
            posinf=nowcast_config.max_dbz,
            neginf=nowcast_config.min_dbz,
        ),
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )

    residual_fn, normal_product = _variational_normal_operator(
        control,
        observations,
        frozen,
    )
    baseline_dynamics_path = _prepare_frozen_baseline_dynamics_path(
        observations,
        frozen,
    )
    baseline_dynamics_branch_status: BaselineDynamicsBranchStatus = (
        "not_applicable" if baseline_dynamics_path is None else "unknown"
    )
    baseline_dynamics_trusted = baseline_dynamics_branch_status in (
        "not_applicable",
        "certified",
    )
    if (
        adjoint_config.require_baseline_dynamics_branch_validity
        and observation_perturbation is None
        and not baseline_dynamics_trusted
    ):
        raise ValueError(
            "P1 FSO baseline dynamics branch margins are unavailable"
        )
    normal_product_budget = _NormalProductBudget(
        maximum=adjoint_config.maximum_normal_products
    )
    gauss_newton_diagnostics = _gauss_newton_curvature_diagnostics(
        control,
        residual_fn,
        normal_product,
        normal_product_budget,
        adjoint_config,
    )
    preconditioner = _variational_preconditioner(
        control,
        frozen,
        adjoint_config,
    )
    warm_solutions: dict[int, Tensor] = {}

    observation_count = observations.dbz.numel()
    base_observation_scale = (
        torch.sqrt(observations.quality_weight)
        * frozen.irls_sqrt_weight
        / observations.std_dbz
    )
    detected_observation_scale = torch.where(
        observations.detected_mask,
        base_observation_scale,
        torch.zeros_like(observations.dbz),
    )
    final_trajectory = _analysis_trajectory(control, frozen)
    analyzed_dbz = echo_to_dbz(
        final_trajectory.frames_linear,
        min_dbz=nowcast_config.min_dbz,
    )
    censor_response = torch.sigmoid(
        (
            analyzed_dbz
            - frozen.analysis_config.detection_limit_dbz
        )
        / frozen.analysis_config.censor_temperature_dbz
    )
    censor_error = (
        frozen.analysis_config.censor_temperature_dbz
        * F.softplus(
            (
                analyzed_dbz
                - frozen.analysis_config.detection_limit_dbz
            )
            / frozen.analysis_config.censor_temperature_dbz
        )
    )
    censor_cross_scale = base_observation_scale * (
        censor_response
        + (
            censor_error
            / frozen.analysis_config.censor_temperature_dbz
        )
        * (1.0 - censor_response)
    )
    censor_observation_scale = torch.where(
        observations.censored_mask,
        censor_cross_scale,
        torch.zeros_like(observations.dbz),
    )
    weighted_observation_residual = residual_fn(control)[
        :observation_count
    ].reshape_as(observations.dbz).detach()
    final_state = RadarState(
        echo_linear=final_trajectory.frames_linear[-1],
        displacement_yx=final_trajectory.displacement_yx,
        log_growth_per_step=final_trajectory.log_growth_per_step,
    )
    detection_margin = _minimum_masked_value(
        torch.abs(
            observations.dbz
            - frozen.analysis_config.detection_limit_dbz
        ),
        observations.valid_mask,
    )
    analysis_remap_margin = _analysis_remap_margin(
        final_trajectory.displacement_yx,
        frozen.analysis_remap_cells,
    )
    publication_support_margin, publication_confidence_margin = (
        _publication_margins(result, forecast_indices)
    )
    prior_support_margin = _neural_prior_support_margin(
        frozen,
        neural_prior_application,
    )
    prior_valid_margin = _neural_prior_valid_margin(
        frozen,
        neural_prior_application,
    )
    forecast_remap_margin = math.inf
    output_cap_margin: float | None = None

    for lead_index, forecast_index in enumerate(forecast_indices):
        truth = truth_linear[forecast_index]
        valid = metric_domain_weights[lead_index]
        forecast_step = forecast_index + 1
        lead_cell = freeze_remap_cell(
            forecast_step * analysis.state.displacement_yx
        )
        forecast_remap_margin = min(
            forecast_remap_margin,
            _remap_fraction_margin(
                forecast_step * analysis.state.displacement_yx,
                lead_cell,
            ),
        )
        latent_prediction = _forecast_linear_at_step_core(
            final_state,
            forecast_step,
            nowcast_config,
            lead_cell,
        )
        prediction, cap_active = _freeze_output_cap(
            latent_prediction,
            nowcast_config,
        )
        nominal_valid = result.valid_mask[forecast_index]
        latent_dbz = echo_to_dbz(
            latent_prediction,
            min_dbz=nowcast_config.min_dbz,
        )
        lead_cap_margin = _minimum_masked_value(
            torch.abs(latent_dbz - nowcast_config.max_dbz),
            nominal_valid,
        )
        if lead_cap_margin is not None:
            output_cap_margin = (
                lead_cap_margin
                if output_cap_margin is None
                else min(output_cap_margin, lead_cap_margin)
            )
        if not torch.allclose(
            prediction[nominal_valid],
            issued_echo[forecast_index][nominal_valid],
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError(
                "P1 FSO model disagrees with the issued forecast"
            )
        if lead_index in selected_position:
            selected_cap_masks[selected_position[lead_index]] = cap_active

        for metric_index, metric_name in enumerate(
            sensitivity_config.metric_names
        ):

            score_from_control: Callable[[Tensor], Tensor] = (
                lambda candidate_control: _variational_forecast_score(
                    candidate_control,
                    frozen,
                    forecast_step,
                    lead_cell,
                    cap_active,
                    metric_name,
                    truth,
                    valid,
                    nowcast_config,
                    sensitivity_config,
                )
            )

            if not _metric_has_support(
                metric_name,
                prediction,
                truth,
                valid,
                nowcast_config,
                sensitivity_config,
            ):
                continue

            metric_available[lead_index, metric_index] = True
            score = score_from_control(control)
            rhs = cast(
                Tensor,
                torch.func.grad(score_from_control)(control),
            ).detach()
            initial = (
                warm_solutions.get(metric_index)
                if adjoint_config.warm_start_by_metric
                else None
            )
            adjoint_solve = _variational_observation_adjoint(
                rhs,
                control,
                observations,
                frozen,
                residual_fn,
                normal_product,
                detected_observation_scale,
                censor_observation_scale,
                weighted_observation_residual,
                observation_count,
                adjoint_config=adjoint_config,
                preconditioner=preconditioner,
                initial=initial,
                budget=normal_product_budget,
            )
            if adjoint_config.warm_start_by_metric:
                warm_solutions[metric_index] = adjoint_solve.solution
            observation_sensitivity = adjoint_solve.sensitivity
            background_sensitivity = (
                _frozen_initial_background_observation_sensitivity(
                    adjoint_solve.solution,
                    control,
                    observations,
                    frozen,
                    forecast_step=forecast_step,
                    lead_cell=lead_cell,
                    cap_active=cap_active,
                    metric_name=metric_name,
                    truth=truth,
                    valid=valid,
                    nowcast_config=nowcast_config,
                    sensitivity_config=sensitivity_config,
                )
            )
            prior_input_sensitivity = background_sensitivity
            if frozen.neural_prior_dependency == "exogenous":
                prior_input_sensitivity = torch.zeros_like(background_sensitivity)
            elif frozen.neural_prior_dependency == "radar_dependent":
                assert neural_prior_runner is not None
                assert validated_prior_input is not None
                prior_cotangent = torch.where(
                    _neural_prior_derivative_mask(frozen),
                    background_sensitivity[0],
                    torch.zeros_like(background_sensitivity[0]),
                )
                prior_log_std_cotangent = (
                    _frozen_neural_prior_log_std_sensitivity(
                        adjoint_solve.solution,
                        control,
                        observations,
                        frozen,
                    )
                )
                with neural_prior_runner.derivative_session(
                    validated_prior_input
                ) as derivative_input:
                    neural_prior_adjoint_direction_maximum_defect = max(
                        neural_prior_adjoint_direction_maximum_defect,
                        neural_prior_runner.validate_adjoint_direction(
                            derivative_input,
                            prior_cotangent,
                            prior_log_std_cotangent,
                        ),
                    )
                    prior_input_sensitivity = neural_prior_runner.vjp_components(
                        derivative_input,
                        prior_cotangent,
                        prior_log_std_cotangent,
                    )
            dynamics_sensitivity = (
                _frozen_baseline_dynamics_observation_sensitivity(
                    baseline_dynamics_path,
                    adjoint_solve.solution,
                    control,
                    observations,
                    frozen,
                    forecast_step=forecast_step,
                    lead_cell=lead_cell,
                    cap_active=cap_active,
                    metric_name=metric_name,
                    truth=truth,
                    valid=valid,
                    nowcast_config=nowcast_config,
                    sensitivity_config=sensitivity_config,
                )
            )
            frozen_structure_input_sensitivity_values = (
                observation_sensitivity.detected_dbz
                + prior_input_sensitivity
                + dynamics_sensitivity
            )

            forecast_scores[lead_index, metric_index] = score.detach()
            adjoint_iterations[lead_index, metric_index] = (
                adjoint_solve.iterations
            )
            adjoint_relative_residual[lead_index, metric_index] = (
                adjoint_solve.relative_residual
            )
            adjoint_true_residual_norm[
                lead_index,
                metric_index,
            ] = adjoint_solve.true_residual_norm
            adjoint_normal_products[lead_index, metric_index] = (
                adjoint_solve.normal_products
            )
            adjoint_warm_started[lead_index, metric_index] = (
                adjoint_solve.warm_started
            )
            selected_index = selected_position.get(lead_index)
            sensitivity_channels = (
                (detected_sensitivity, observation_sensitivity.detected_dbz),
                (
                    censor_sensitivity,
                    observation_sensitivity.censor_threshold_dbz,
                ),
                (
                    weight_sensitivity,
                    observation_sensitivity.observation_weight,
                ),
                (
                    initial_background_sensitivity,
                    background_sensitivity,
                ),
                (
                    baseline_dynamics_sensitivity,
                    dynamics_sensitivity,
                ),
                (
                    frozen_structure_input_sensitivity,
                    frozen_structure_input_sensitivity_values,
                ),
            )
            for accumulator, values in sensitivity_channels:
                _record_variational_channel(
                    accumulator,
                    values,
                    lead_index=lead_index,
                    metric_index=metric_index,
                    selected_index=selected_index,
                    tile_size=tile_shape_yx,
                    signed_sum=False,
                )

            if (
                observation_perturbation is not None
                and impact_accumulators is not None
            ):
                background_impact = (
                    prior_input_sensitivity
                    * observation_perturbation.physical_radar_dbz_delta
                    if observation_perturbation.perturbation_semantics
                    == "physical_radar_value"
                    and observation_perturbation.physical_radar_dbz_delta
                    is not None
                    else background_sensitivity
                    * _initial_background_perturbation(
                        observation_perturbation,
                        observations,
                    )
                )
                component_impacts = (
                    observation_sensitivity.detected_dbz
                    * observation_perturbation.detected_dbz,
                    observation_sensitivity.censor_threshold_dbz
                    * observation_perturbation.censor_threshold_dbz,
                    observation_sensitivity.observation_weight
                    * observation_perturbation.observation_weight,
                    background_impact,
                    dynamics_sensitivity
                    * _baseline_dynamics_perturbation(
                        observation_perturbation,
                        observations,
                    ),
                )
                total_impact = (
                    component_impacts[0]
                    + component_impacts[1]
                    + component_impacts[2]
                    + component_impacts[3]
                    + component_impacts[4]
                )
                for accumulator, values in zip(
                    impact_accumulators,
                    (*component_impacts, total_impact),
                    strict=True,
                ):
                    _record_variational_channel(
                        accumulator,
                        values,
                        lead_index=lead_index,
                        metric_index=metric_index,
                        selected_index=selected_index,
                        tile_size=tile_shape_yx,
                        signed_sum=True,
                    )

    low_local_validity = (
        detection_margin is None
        or detection_margin < adjoint_config.minimum_detection_margin_dbz
        or analysis_remap_margin
        < adjoint_config.minimum_remap_fraction_margin
        or forecast_remap_margin
        < adjoint_config.minimum_remap_fraction_margin
        or output_cap_margin is None
        or output_cap_margin < adjoint_config.minimum_output_cap_margin_dbz
        or publication_support_margin
        < adjoint_config.minimum_publication_margin
        or (
            publication_confidence_margin is not None
            and publication_confidence_margin
            < adjoint_config.minimum_publication_margin
        )
        or (
            frozen.neural_prior_dependency is not None
            and (
                prior_valid_margin is None
                or prior_valid_margin
                < adjoint_config.minimum_neural_prior_valid_margin
                or prior_support_margin is None
                or prior_support_margin
                < adjoint_config.minimum_neural_prior_support_margin
            )
        )
    )
    active_set_margins = VariationalActiveSetMargins(
        detection_classification_dbz=detection_margin,
        analysis_remap_fraction=analysis_remap_margin,
        forecast_remap_fraction=forecast_remap_margin,
        output_cap_dbz=output_cap_margin,
        publication_support=publication_support_margin,
        publication_confidence=publication_confidence_margin,
        neural_prior_valid_probability=prior_valid_margin,
        neural_prior_support_probability=prior_support_margin,
        low_local_validity=low_local_validity,
    )
    if adjoint_config.require_active_set_margin and low_local_validity:
        raise ValueError("P1 FSO active-set margin is below its requirement")

    frozen_structure_channel = _sensitivity_channel(
        frozen_structure_input_sensitivity
    )
    observation_sensitivity = VariationalObservationSensitivity(
        detected_dbz=_sensitivity_channel(detected_sensitivity),
        censor_threshold_dbz=_sensitivity_channel(censor_sensitivity),
        observation_weight=_sensitivity_channel(weight_sensitivity),
        initial_background_dbz=_sensitivity_channel(
            initial_background_sensitivity
        ),
        baseline_dynamics_dbz=_sensitivity_channel(
            baseline_dynamics_sensitivity
        ),
        frozen_structure_input_dbz=frozen_structure_channel,
        baseline_branch_trusted_frozen_structure_input_dbz=(
            frozen_structure_channel if baseline_dynamics_trusted else None
        ),
    )
    fso = VariationalFSO(
        contract="p1-variational-fso-v17",
        forecast_run_digest=result.forecast_run_digest,
        analysis_input_digest=cast(str, result.run.analysis_input_digest),
        sensitivity_config_digest=sensitivity_config.digest,
        adjoint_config_digest=adjoint_config.digest,
        linearization_contract=linearization.contract,
        linearization_digest=linearization.linearization_digest,
        verification_contract=verification_bundle.contract,
        verification_bundle_digest=verification_bundle.content_digest,
        verification_lineage_complete=(
            verification_bundle.lineage_complete
        ),
        verification_valid_times=verification_bundle.valid_times,
        verification_grid_contract_digest=(
            verification_bundle.grid_contract_digest
        ),
        verification_radar_product_digest=(
            verification_bundle.radar_product_digest
        ),
        verification_qc_pipeline_digest=(
            verification_bundle.qc_pipeline_digest
        ),
        metric_contract_digest=_metric_contract_digest(sensitivity_config),
        algorithm_bundle_digest=linearization.algorithm_bundle_digest,
        numerical_runtime_digest=linearization.numerical_runtime_digest,
        variational_fso_digest="",
        sensitivity_scope=(
            "residual_plus_input_dependent_initial_state_and_baseline_with_frozen_selection"
        ),
        baseline_dynamics_frozen=False,
        baseline_pair_selection_frozen=True,
        baseline_dynamics_branch_status=baseline_dynamics_branch_status,
        metric_names=sensitivity_config.metric_names,
        metric_domain=sensitivity_config.metric_domain,
        metric_domain_digest=metric_domain_digest,
        lead_minutes=lead_minutes,
        full_map_lead_minutes=sensitivity_config.full_map_lead_minutes,
        tile_size=max(tile_shape_yx),
        tile_shape_yx=tile_shape_yx,
        forecast_scores=forecast_scores,
        metric_available=metric_available,
        metric_domain_weight_sum=metric_domain_weight_sum,
        metric_domain_weight_fraction=metric_domain_weight_fraction,
        forecast_cap_active_mask=selected_cap_masks,
        observation=observation_sensitivity,
        adjoint_iterations=adjoint_iterations,
        adjoint_relative_residual=adjoint_relative_residual,
        adjoint_true_residual_norm=adjoint_true_residual_norm,
        adjoint_normal_products=adjoint_normal_products,
        adjoint_warm_started=adjoint_warm_started,
        total_normal_products=normal_product_budget.used,
        whitener_operations_per_apply=0,
        observed_whitener_apply_count=0,
        materialized_output_bytes=materialized_output_bytes,
        neural_prior_adjoint_direction_maximum_defect=(
            neural_prior_adjoint_direction_maximum_defect
        ),
        active_set_margins=active_set_margins,
        feasibility_margins=feasibility_margins,
        gauss_newton_diagnostics=gauss_newton_diagnostics,
    )
    fso = replace(fso, variational_fso_digest=variational_fso_digest(fso))
    if validated_prior_input is not None:
        validated_prior_input.validate_completion()
    if impact_accumulators is None:
        return fso, None, None
    total_impact = _impact_channel(impact_accumulators[5])
    if perturbation_diagnostics is None:
        raise RuntimeError("variational perturbation diagnostics are missing")
    impact_branch_trusted = (
        perturbation_diagnostics.baseline_dynamics_branch_status
        in ("not_applicable", "certified")
    )
    return (
        fso,
        VariationalObservationImpact(
            detected_dbz=_impact_channel(impact_accumulators[0]),
            censor_threshold_dbz=_impact_channel(impact_accumulators[1]),
            observation_weight=_impact_channel(impact_accumulators[2]),
            initial_background_dbz=_impact_channel(
                impact_accumulators[3]
            ),
            baseline_dynamics_dbz=_impact_channel(
                impact_accumulators[4]
            ),
            total=total_impact,
            baseline_branch_trusted_total=(
                total_impact if impact_branch_trusted else None
            ),
        ),
        perturbation_diagnostics,
    )


def _variational_state(
    control: Tensor,
    frozen: FrozenOuterState,
) -> RadarState:
    trajectory = _analysis_trajectory(control, frozen)
    return RadarState(
        echo_linear=trajectory.frames_linear[-1],
        displacement_yx=trajectory.displacement_yx,
        log_growth_per_step=trajectory.log_growth_per_step,
    )


def _variational_forecast_score(
    control: Tensor,
    frozen: FrozenOuterState,
    step: int,
    lead_cell: RemapCell,
    cap_active: Tensor,
    metric_name: str,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> Tensor:
    latent = _forecast_linear_at_step_core(
        _variational_state(control, frozen),
        step,
        nowcast_config,
        lead_cell,
    )
    return forecast_metric(
        metric_name,
        _apply_output_cap(latent, cap_active, nowcast_config),
        truth,
        valid,
        nowcast_config,
        sensitivity_config,
        frozen.grid_time_contract,
    )


def _variational_normal_operator(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    residual_fn: Callable[[Tensor], Tensor] = lambda value: residual_vector(
        value,
        observations,
        frozen,
    )
    residual_vjp = torch.func.vjp(residual_fn, control)
    pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        residual_vjp[1],
    )

    def normal_product(direction: Tensor) -> Tensor:
        jacobian_direction = cast(
            Tensor,
            torch.func.jvp(residual_fn, (control,), (direction,))[1],
        )
        return pullback(jacobian_direction)[0]

    return residual_fn, normal_product


def _variational_observation_adjoint(
    rhs: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    residual_fn: Callable[[Tensor], Tensor],
    normal_product: Callable[[Tensor], Tensor],
    detected_observation_scale: Tensor,
    censor_observation_scale: Tensor,
    weighted_observation_residual: Tensor,
    observation_count: int,
    *,
    adjoint_config: VariationalAdjointConfig,
    preconditioner: Callable[[Tensor], Tensor] | None,
    initial: Tensor | None,
    budget: _NormalProductBudget,
) -> _VariationalAdjointSolve:
    products_before = budget.used

    def counted_normal_product(value: Tensor) -> Tensor:
        return budget.apply(normal_product, value)

    relative_tolerance = (
        frozen.analysis_config.pcg_relative_tolerance
        if adjoint_config.pcg_relative_tolerance is None
        else adjoint_config.pcg_relative_tolerance
    )
    maximum_iterations = (
        frozen.analysis_config.maximum_pcg_iterations
        if adjoint_config.maximum_pcg_iterations is None
        else adjoint_config.maximum_pcg_iterations
    )
    try:
        adjoint = pcg(
            counted_normal_product,
            rhs,
            preconditioner=preconditioner,
            initial=initial,
            rtol=relative_tolerance,
            max_iterations=maximum_iterations,
        )
    except (ArithmeticError, RuntimeError, ValueError) as error:
        if str(error) == "P1 FSO normal-product budget exhausted":
            raise ValueError(
                "P1 FSO normal-product budget exhausted"
            ) from error
        raise ValueError("P1 FSO adjoint solve failed") from error
    if not adjoint.converged or not bool(
        torch.all(torch.isfinite(adjoint.solution))
    ):
        raise ValueError("P1 FSO adjoint solve did not converge")
    rhs_norm = float(torch.linalg.vector_norm(rhs.detach()).cpu())
    true_residual_norm = adjoint.relative_residual * rhs_norm

    jacobian_adjoint = cast(
        Tensor,
        torch.func.jvp(
            residual_fn,
            (control,),
            (adjoint.solution,),
        )[1],
    )
    prediction_response = jacobian_adjoint[:observation_count].reshape_as(
        observations.dbz
    )
    # Detected dBZ changes only the residual offset. The censored threshold
    # also changes the observation Jacobian, so its cross scale includes the
    # (dJ/dL).T r contribution assembled above. The objective-weight channel
    # differentiates alpha * 0.5 * r_i**2 at alpha=1.
    if frozen.analysis_config.observation_common_bias_std_dbz > 0.0:
        sensitivity = _correlated_observation_parameter_sensitivity(
            adjoint.solution,
            control,
            observations,
            frozen,
        )
    else:
        sensitivity = _VariationalAdjointSensitivity(
            detected_dbz=(
                detected_observation_scale * prediction_response
            ).detach(),
            censor_threshold_dbz=(
                censor_observation_scale * prediction_response
            ).detach(),
            observation_weight=(
                -weighted_observation_residual * prediction_response
            ).detach(),
        )
    return _VariationalAdjointSolve(
        sensitivity=sensitivity,
        solution=adjoint.solution.detach(),
        iterations=adjoint.iterations,
        relative_residual=adjoint.relative_residual,
        true_residual_norm=true_residual_norm,
        normal_products=budget.used - products_before,
        warm_started=initial is not None,
    )


def _frozen_initial_background_observation_sensitivity(
    adjoint_solution: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    forecast_step: int,
    lead_cell: RemapCell,
    cap_active: Tensor,
    metric_name: str,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> Tensor:
    """Differentiate accepted first-frame values through the P1 background.

    The active field, P0-derived baseline dynamics, remap cells, observation
    classes, and every other frozen structure remain fixed. The result has the
    observation shape and is nonzero only where the first frame supplied the
    P1 initial background.
    """

    initial_background = frozen.initial_background_dbz.detach()

    def frozen_with_background(candidate: Tensor) -> FrozenOuterState:
        return replace(frozen, initial_background_dbz=candidate)

    def stationarity_from_background(candidate: Tensor) -> Tensor:
        def objective(candidate_control: Tensor) -> Tensor:
            residual = residual_vector(
                candidate_control,
                observations,
                frozen_with_background(candidate),
            )
            return 0.5 * torch.dot(residual, residual)

        return cast(Tensor, torch.func.grad(objective)(control))

    stationarity_pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        torch.func.vjp(
            stationarity_from_background,
            initial_background,
        )[1],
    )
    implicit = -stationarity_pullback(adjoint_solution)[0]

    def score_from_background(candidate: Tensor) -> Tensor:
        return _variational_forecast_score(
            control,
            frozen_with_background(candidate),
            forecast_step,
            lead_cell,
            cap_active,
            metric_name,
            truth,
            valid,
            nowcast_config,
            sensitivity_config,
        )

    direct = cast(
        Tensor,
        torch.func.grad(score_from_background)(initial_background),
    )
    accepted_first_frame = (
        observations.valid_mask[0] & frozen.observed_mask[0]
    )
    if not frozen.observation_derived_initial_background:
        if frozen.neural_prior_valid_mask is None:
            raise ValueError("neural-prior background validity is missing")
        accepted_first_frame = frozen.neural_prior_valid_mask
    first_frame = torch.where(
        accepted_first_frame,
        direct + implicit,
        torch.zeros_like(initial_background),
    )
    return torch.cat(
        (
            first_frame.unsqueeze(0),
            torch.zeros_like(observations.dbz[1:]),
        ),
        dim=0,
    ).detach()


def _frozen_neural_prior_log_std_sensitivity(
    adjoint_solution: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    """Differentiate P1 stationarity through the spatial prior precision."""

    prior_std = frozen.neural_prior_std_dbz
    prior_valid = frozen.neural_prior_valid_mask
    if prior_std is None or prior_valid is None:
        raise ValueError("neural-prior uncertainty state is missing")
    log_std = torch.log(prior_std.detach())

    def stationarity(candidate_log_std: Tensor) -> Tensor:
        candidate_frozen = replace(
            frozen,
            neural_prior_std_dbz=torch.exp(candidate_log_std),
        )

        def objective(candidate_control: Tensor) -> Tensor:
            residual = residual_vector(
                candidate_control,
                observations,
                candidate_frozen,
            )
            return 0.5 * torch.dot(residual, residual)

        return cast(Tensor, torch.func.grad(objective)(control))

    pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        torch.func.vjp(stationarity, log_std)[1],
    )
    sensitivity = -pullback(adjoint_solution)[0]
    return torch.where(
        prior_valid,
        sensitivity,
        torch.zeros_like(sensitivity),
    ).detach()


def _prepare_frozen_baseline_dynamics_path(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> _FrozenBaselineDynamicsPath | None:
    """Freeze P0 pair/peak branches and retain their continuous VJP."""

    if (
        frozen.baseline_metadata.tendency_source
        is not TendencySource.OBSERVATION
    ):
        return None
    observation_dbz = observations.dbz.detach()

    def dynamics_from_observation(candidate: Tensor) -> Tensor:
        return _baseline_dynamics_from_observation(candidate, frozen)

    nominal_dynamics, pullback = cast(
        tuple[Tensor, Callable[[Tensor], tuple[Tensor]]],
        torch.func.vjp(
            dynamics_from_observation,
            observation_dbz,
        ),
    )
    expected = torch.cat(
        (
            frozen.baseline_state.displacement_yx,
            frozen.baseline_state.log_growth_per_step.reshape(1),
        )
    )
    tolerance = frozen.nowcast_config.contract_absolute_tolerance
    if not torch.allclose(
        nominal_dynamics,
        expected,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "frozen P0 dynamics path does not reproduce the baseline state"
        )
    active = observations.detected_mask & frozen.observed_mask
    return _FrozenBaselineDynamicsPath(
        active_mask=active,
        nominal_dynamics=nominal_dynamics.detach(),
        observation_pullback=pullback,
    )


def _baseline_dynamics_from_observation(
    frames_dbz: Tensor,
    frozen: FrozenOuterState,
) -> Tensor:
    floor = frames_dbz.new_full((), frozen.nowcast_config.min_dbz)
    clean = torch.where(frozen.observed_mask, frames_dbz, floor)
    linear = dbz_to_echo(
        clean,
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    )
    estimate = _estimate_source_tendencies(
        clean,
        frozen.observed_mask,
        linear,
        frozen.nowcast_config,
        frozen.grid_time_contract,
    )
    return torch.cat(
        (
            estimate.displacement_yx,
            estimate.log_growth_per_step.reshape(1),
        )
    )


def _frozen_baseline_dynamics_observation_sensitivity(
    path: _FrozenBaselineDynamicsPath | None,
    adjoint_solution: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    forecast_step: int,
    lead_cell: RemapCell,
    cap_active: Tensor,
    metric_name: str,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> Tensor:
    """Differentiate continuous P0 dynamics with pair/peak selection fixed."""

    if path is None:
        return torch.zeros_like(observations.dbz)

    def frozen_with_dynamics(candidate: Tensor) -> FrozenOuterState:
        state = RadarState(
            echo_linear=frozen.baseline_state.echo_linear,
            displacement_yx=candidate[:2],
            log_growth_per_step=candidate[2],
        )
        return replace(frozen, baseline_state=state)

    def stationarity_from_dynamics(candidate: Tensor) -> Tensor:
        def objective(candidate_control: Tensor) -> Tensor:
            residual = residual_vector(
                candidate_control,
                observations,
                frozen_with_dynamics(candidate),
            )
            return 0.5 * torch.dot(residual, residual)

        return cast(Tensor, torch.func.grad(objective)(control))

    stationarity_pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        torch.func.vjp(
            stationarity_from_dynamics,
            path.nominal_dynamics,
        )[1],
    )
    implicit = -stationarity_pullback(adjoint_solution)[0]

    def score_from_dynamics(candidate: Tensor) -> Tensor:
        return _variational_forecast_score(
            control,
            frozen_with_dynamics(candidate),
            forecast_step,
            lead_cell,
            cap_active,
            metric_name,
            truth,
            valid,
            nowcast_config,
            sensitivity_config,
        )

    direct = cast(
        Tensor,
        torch.func.grad(score_from_dynamics)(path.nominal_dynamics),
    )
    observation_gradient = path.observation_pullback(direct + implicit)[0]
    return torch.where(
        path.active_mask,
        observation_gradient,
        torch.zeros_like(observation_gradient),
    ).detach()


def _correlated_observation_parameter_sensitivity(
    adjoint_solution: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> _VariationalAdjointSensitivity:
    """Differentiate frozen stationarity through a non-diagonal whitener."""

    detected_values = observations.dbz.detach()
    censor_threshold = observations.dbz.new_full(
        observations.dbz.shape,
        frozen.analysis_config.detection_limit_dbz,
    )
    observation_multiplier = torch.ones_like(observations.dbz)

    def stationarity(
        candidate_detected: Tensor,
        candidate_threshold: Tensor,
        candidate_multiplier: Tensor,
    ) -> Tensor:
        def observation_objective(candidate_control: Tensor) -> Tensor:
            trajectory = _analysis_trajectory(candidate_control, frozen)
            prediction = echo_to_dbz(
                trajectory.frames_linear,
                min_dbz=frozen.nowcast_config.min_dbz,
            )
            detected_error = prediction - candidate_detected
            censored_error = (
                frozen.analysis_config.censor_temperature_dbz
                * F.softplus(
                    (
                        prediction - candidate_threshold
                    )
                    / frozen.analysis_config.censor_temperature_dbz
                )
            )
            error = torch.where(
                observations.detected_mask,
                detected_error,
                torch.where(
                    observations.censored_mask,
                    censored_error,
                    torch.zeros_like(prediction),
                ),
            )
            standardized = (
                torch.sqrt(observations.quality_weight)
                * torch.sqrt(candidate_multiplier)
                * error
                / observations.std_dbz
            )
            whitened = _apply_observation_error_whitener(
                standardized,
                observations,
                frozen.analysis_config,
                whitener=frozen.observation_whitener,
            )
            residual = frozen.irls_sqrt_weight * whitened
            return 0.5 * torch.dot(residual.flatten(), residual.flatten())

        return cast(
            Tensor,
            torch.func.grad(observation_objective)(control),
        )

    vjp_result = torch.func.vjp(
        stationarity,
        detected_values,
        censor_threshold,
        observation_multiplier,
    )
    pullback = cast(
        Callable[[Tensor], tuple[Tensor, Tensor, Tensor]],
        vjp_result[1],
    )
    detected_gradient, censor_gradient, weight_gradient = pullback(
        adjoint_solution
    )
    return _VariationalAdjointSensitivity(
        detected_dbz=torch.where(
            observations.detected_mask,
            -detected_gradient,
            torch.zeros_like(detected_gradient),
        ).detach(),
        censor_threshold_dbz=torch.where(
            observations.censored_mask,
            -censor_gradient,
            torch.zeros_like(censor_gradient),
        ).detach(),
        observation_weight=torch.where(
            observations.valid_mask,
            -weight_gradient,
            torch.zeros_like(weight_gradient),
        ).detach(),
    )


def _validate_variational_observation_perturbation(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    config: VariationalAdjointConfig,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalPerturbationDiagnostics:
    if perturbation.contract != "p1-observation-perturbation-v7":
        raise ValueError("unsupported P1 observation perturbation contract")
    if perturbation.perturbation_semantics not in (
        "augmented_parameter",
        "physical_radar_value",
    ):
        raise ValueError("unsupported observation perturbation semantics")
    channels = (
        (
            "detected_dbz",
            perturbation.detected_dbz,
            observations.detected_mask,
        ),
        (
            "censor_threshold_dbz",
            perturbation.censor_threshold_dbz,
            observations.censored_mask,
        ),
        (
            "observation_weight",
            perturbation.observation_weight,
            observations.valid_mask,
        ),
    )
    for name, values, active_mask in channels:
        _validate_perturbation_tensor(
            name,
            values,
            observations,
            active_mask,
        )
    _require_local_perturbation(
        "detected_dbz",
        perturbation.detected_dbz,
        config.maximum_detected_delta_dbz,
    )
    _require_local_perturbation(
        "censor_threshold_dbz",
        perturbation.censor_threshold_dbz,
        config.maximum_censor_delta_dbz,
    )
    _require_local_perturbation(
        "observation_weight",
        perturbation.observation_weight,
        config.maximum_observation_weight_delta,
    )
    if perturbation.initial_background_dbz is not None:
        values = perturbation.initial_background_dbz
        active = torch.zeros_like(observations.valid_mask)
        if frozen.neural_prior_dependency == "radar_dependent":
            if frozen.neural_prior_valid_mask is None:
                raise ValueError("neural-prior perturbation lacks a valid mask")
            active[0] = frozen.neural_prior_valid_mask
        else:
            active[0] = observations.valid_mask[0] & frozen.observed_mask[0]
        _validate_perturbation_tensor(
            "initial_background_dbz",
            values,
            observations,
            active,
            active_domain="accepted first-frame observations",
        )
    if perturbation.baseline_dynamics_dbz is not None:
        values = perturbation.baseline_dynamics_dbz
        active = observations.valid_mask & frozen.observed_mask
        _validate_perturbation_tensor(
            "baseline_dynamics_dbz",
            values,
            observations,
            active,
            active_domain="accepted observations",
        )
    for name, values in (
        ("initial_background_dbz", perturbation.initial_background_dbz),
        ("baseline_dynamics_dbz", perturbation.baseline_dynamics_dbz),
    ):
        if values is not None:
            _require_local_perturbation(
                name,
                values,
                config.maximum_background_delta_dbz,
            )
    physical_delta = _validate_physical_radar_semantics(
        perturbation,
        observations,
        frozen,
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )
    _validate_directional_classification(
        perturbation,
        observations,
        frozen,
        config,
        physical_delta,
    )
    return _perturbation_diagnostics(
        perturbation,
        observations,
        frozen,
        config,
        physical_delta,
    )


def _validate_perturbation_tensor(
    name: str,
    values: Tensor,
    observations: AnalysisObservations,
    active_mask: Tensor,
    *,
    active_domain: str = "its active mask",
) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} perturbation must be a Tensor")
    if values.shape != observations.dbz.shape:
        raise ValueError(f"{name} perturbation shape mismatch")
    if values.dtype != observations.dbz.dtype:
        raise ValueError(f"{name} perturbation dtype mismatch")
    if values.device != observations.dbz.device:
        raise ValueError(f"{name} perturbation device mismatch")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError(f"{name} perturbation must be finite")
    if bool(torch.any(values.masked_select(~active_mask) != 0)):
        raise ValueError(
            f"{name} perturbation must be zero outside {active_domain}"
        )


def _validate_physical_radar_semantics(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> Tensor | None:
    physical_delta = perturbation.physical_radar_dbz_delta
    if perturbation.perturbation_semantics == "augmented_parameter":
        if physical_delta is not None:
            raise ValueError(
                "augmented parameter perturbation cannot carry a physical "
                "dBZ delta"
            )
        return None
    if physical_delta is None:
        raise ValueError(
            "physical radar perturbation requires its source dBZ delta"
        )
    _validate_perturbation_tensor(
        "physical_radar_dbz_delta",
        physical_delta,
        observations,
        observations.detected_mask,
        active_domain="detected observations",
    )
    detected, background, dynamics = _physical_radar_channels(
        physical_delta,
        observations,
        frozen,
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )
    expected = (
        detected,
        torch.zeros_like(physical_delta),
        torch.zeros_like(physical_delta),
        background,
        dynamics,
    )
    actual = (
        perturbation.detected_dbz,
        perturbation.censor_threshold_dbz,
        perturbation.observation_weight,
        perturbation.initial_background_dbz,
        perturbation.baseline_dynamics_dbz,
    )
    if any(
        value is None or not torch.equal(value, canonical)
        for value, canonical in zip(actual, expected, strict=True)
    ):
        raise ValueError(
            "physical radar perturbation channels are inconsistent"
        )
    _ = _physical_radar_input_margins(
        physical_delta,
        observations,
        frozen,
    )
    return physical_delta


def _validate_directional_classification(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    config: VariationalAdjointConfig,
    physical_delta: Tensor | None,
) -> None:
    delta_dbz = (
        perturbation.detected_dbz
        if physical_delta is None
        else physical_delta
    )
    changed_dbz = observations.dbz + delta_dbz
    changed_limit = (
        frozen.analysis_config.detection_limit_dbz
        + perturbation.censor_threshold_dbz
    )
    margin = config.minimum_detection_margin_dbz
    changed_classification = (delta_dbz != 0) | (
        perturbation.censor_threshold_dbz != 0
    )
    changed_detected = observations.detected_mask & changed_classification
    changed_censored = observations.censored_mask & changed_classification
    detected_valid = torch.all(
        changed_dbz.masked_select(changed_detected)
        >= changed_limit.masked_select(changed_detected) + margin
    )
    censored_valid = torch.all(
        changed_dbz.masked_select(changed_censored)
        <= changed_limit.masked_select(changed_censored) - margin
    )
    if not bool(detected_valid & censored_valid):
        raise ValueError(
            "observation perturbation crosses the detected/censored branch"
        )
    if bool(
        torch.any(
            1.0 + perturbation.observation_weight
            < config.minimum_observation_multiplier
        )
    ):
        raise ValueError(
            "observation perturbation crosses the weight-multiplier branch"
        )


def _perturbation_diagnostics(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    config: VariationalAdjointConfig,
    physical_delta: Tensor | None,
) -> VariationalPerturbationDiagnostics:
    floor_margin, ceiling_margin = (
        (None, None)
        if physical_delta is None
        else _physical_radar_input_margins(
            physical_delta,
            observations,
            frozen,
        )
    )
    baseline_delta = _baseline_dynamics_perturbation(
        perturbation,
        observations,
    )
    baseline_branch_status, baseline_signature_digest = (
        _baseline_dynamics_branch_certification(
            observations,
            frozen,
            baseline_delta,
        )
    )
    if baseline_branch_status == "invalid":
        raise ValueError(
            "observation perturbation crosses the frozen P0 tendency branch"
        )

    dbz_channels = (
        (physical_delta,)
        if physical_delta is not None
        else (
            perturbation.detected_dbz,
            perturbation.censor_threshold_dbz,
            perturbation.initial_background_dbz,
            perturbation.baseline_dynamics_dbz,
        )
    )
    active = perturbation.observation_weight != 0
    whitened_energy = torch.zeros_like(observations.dbz)
    for values in dbz_channels:
        if values is None:
            continue
        active |= values != 0
        standardized = (
            torch.sqrt(observations.quality_weight)
            * values
            / observations.std_dbz
        )
        whitened = _apply_observation_error_whitener(
            standardized,
            observations,
            frozen.analysis_config,
            whitener=frozen.observation_whitener,
        )
        whitened_energy += whitened.square()

    pixel_count = int(torch.count_nonzero(active).detach())
    valid_count = max(1, int(torch.count_nonzero(observations.valid_mask)))
    fraction = pixel_count / valid_count
    grid = frozen.grid_time_contract
    area_km2 = (
        None
        if grid is None
        else pixel_count * grid.cell_area_m2 / 1.0e6
    )
    whitened_l2 = math.sqrt(float(torch.sum(whitened_energy).detach()))
    tile_norm = _maximum_tile_norm(
        whitened_energy,
        _perturbation_tile_size(config, grid),
    )
    weight_l2 = float(
        torch.linalg.vector_norm(perturbation.observation_weight).detach()
    )
    limits = (
        (pixel_count, config.maximum_perturbed_pixel_count, "pixel budget"),
        (fraction, config.maximum_perturbed_fraction, "area fraction"),
        (
            whitened_l2,
            config.maximum_whitened_perturbation_l2,
            "whitened trust radius",
        ),
        (
            tile_norm,
            config.maximum_per_tile_whitened_norm,
            "per-tile trust radius",
        ),
        (
            weight_l2,
            config.maximum_observation_weight_l2,
            "weight trust radius",
        ),
    )
    for value, limit, name in limits:
        if value > limit:
            raise ValueError(f"observation perturbation exceeds its {name}")
    if config.maximum_perturbed_area_km2 is not None:
        if area_km2 is None:
            raise ValueError(
                "physical perturbation area requires a grid contract"
            )
        if area_km2 > config.maximum_perturbed_area_km2:
            raise ValueError(
                "observation perturbation exceeds its physical area budget"
            )
    return VariationalPerturbationDiagnostics(
        perturbed_pixel_count=pixel_count,
        perturbed_fraction=fraction,
        perturbed_area_km2=area_km2,
        whitened_l2=whitened_l2,
        maximum_per_tile_whitened_norm=tile_norm,
        observation_weight_l2=weight_l2,
        minimum_input_floor_margin_dbz=floor_margin,
        minimum_input_ceiling_margin_dbz=ceiling_margin,
        directional_classification_valid=True,
        baseline_dynamics_branch_status=baseline_branch_status,
        baseline_dynamics_branch_signature_digest=(
            baseline_signature_digest
        ),
    )


def _maximum_tile_norm(energy: Tensor, tile_size: TileShape) -> float:
    maximum = 0.0
    tile_height, tile_width = tile_size
    for frame in energy:
        for row in range(0, frame.shape[0], tile_height):
            for column in range(0, frame.shape[1], tile_width):
                tile = frame[
                    row : row + tile_height,
                    column : column + tile_width,
                ]
                maximum = max(
                    maximum,
                    math.sqrt(float(torch.sum(tile).detach())),
                )
    return maximum


def _baseline_branch_is_stable(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    delta_dbz: Tensor,
) -> bool:
    if (
        frozen.baseline_metadata.tendency_source
        is not TendencySource.OBSERVATION
    ):
        return not bool(torch.any(delta_dbz != 0))
    status, _ = _baseline_dynamics_branch_certification(
        observations,
        frozen,
        delta_dbz,
    )
    return status == "certified"


def _baseline_dynamics_branch_certification(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    delta_dbz: Tensor,
) -> tuple[BaselineDynamicsBranchStatus, str | None]:
    if (
        frozen.baseline_metadata.tendency_source
        is not TendencySource.OBSERVATION
    ):
        return "not_applicable", None
    nominal = _p0_tendency_branch_signature(observations.dbz, frozen)
    signature_digest = dataclass_digest(nominal)
    if not bool(torch.any(delta_dbz != 0)):
        return "certified", signature_digest
    for scale in (0.5, 1.0):
        changed = observations.dbz + scale * delta_dbz
        if _p0_tendency_branch_signature(changed, frozen) != nominal:
            return "invalid", signature_digest
    return "certified", signature_digest


def _p0_tendency_branch_signature(
    frames_dbz: Tensor,
    frozen: FrozenOuterState,
) -> P0TendencyBranchSignature:
    floor = frames_dbz.new_full((), frozen.nowcast_config.min_dbz)
    clean = torch.where(frozen.observed_mask, frames_dbz, floor)
    linear = dbz_to_echo(
        clean,
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    )
    estimate = _estimate_source_tendencies(
        clean,
        frozen.observed_mask,
        linear,
        frozen.nowcast_config,
        frozen.grid_time_contract,
    )
    pair_spans = ((0, 1), (1, 2), (0, 2))
    peaks: list[tuple[int, int]] = []
    interiors: list[bool] = []
    available: list[bool] = []
    growth_available: list[bool] = []
    for previous, current in pair_spans:
        common = frozen.observed_mask[previous] & frozen.observed_mask[current]
        previous_dbz = torch.where(common, clean[previous], floor)
        current_dbz = torch.where(common, clean[current], floor)
        step_span = current - previous
        limits = motion_displacement_limits_yx(
            frozen.nowcast_config,
            frozen.grid_time_contract,
            previous_dbz,
        )
        _, _, interior, peak = _phase_correlation_details(
            previous_dbz,
            current_dbz,
            frozen.nowcast_config,
            max_displacement_yx=limits * step_span,
            grid_time_contract=frozen.grid_time_contract,
        )
        pair = _estimate_available_pair(
            clean,
            frozen.observed_mask,
            linear,
            previous,
            current,
            frozen.nowcast_config,
            frozen.grid_time_contract,
        )
        peaks.append(peak)
        interiors.append(interior)
        available.append(pair is not None)
        growth_available.append(pair is not None and pair[1].available)
    return P0TendencyBranchSignature(
        pair_spans=pair_spans,
        motion_pair_spans=estimate.motion_pair_spans,
        growth_pair_spans=estimate.growth_pair_spans,
        integer_peak_yx_by_pair=tuple(peaks),
        peak_is_search_interior_by_pair=tuple(interiors),
        pair_available_by_span=tuple(available),
        growth_evidence_available_by_span=tuple(growth_available),
        motion_remap_cells=(
            freeze_remap_cell(estimate.displacement_yx),
            freeze_remap_cell(2.0 * estimate.displacement_yx),
        ),
        motion_selection=estimate.motion_pair_selection,
        growth_selection=estimate.growth_pair_selection,
        motion_conflict=estimate.motion_pair_conflict,
        growth_conflict=estimate.growth_pair_conflict,
    )


def _require_local_perturbation(
    name: str,
    values: Tensor,
    maximum_absolute_value: float,
) -> None:
    if bool(torch.any(torch.abs(values) > maximum_absolute_value)):
        raise ValueError(
            f"{name} perturbation exceeds the local first-order limit"
        )


def _initial_background_perturbation(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
) -> Tensor:
    values = perturbation.initial_background_dbz
    return torch.zeros_like(observations.dbz) if values is None else values


def _baseline_dynamics_perturbation(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
) -> Tensor:
    values = perturbation.baseline_dynamics_dbz
    return torch.zeros_like(observations.dbz) if values is None else values


def _validate_variational_fso_lineage(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    linearization: AnalysisLinearization,
) -> None:
    frozen = linearization.frozen
    observations = linearization.observations
    validate_analysis_linearization_content(
        analysis.control,
        linearization,
    )
    if linearization.forecast_run_digest != result.forecast_run_digest:
        raise ValueError("P1 linearization forecast run mismatch")
    if frozen.nowcast_config != result.run.config:
        raise ValueError("P1 linearization nowcast config mismatch")
    config_digest = dataclass_digest(frozen.analysis_config)
    if config_digest != result.run.analysis_config_digest:
        raise ValueError("P1 linearization analysis config mismatch")
    if (
        frozen.neural_prior_application_digest != result.run.prior_application_digest
        or frozen.neural_prior_dependency != result.run.prior_dependency
    ):
        raise ValueError("P1 linearization neural-prior lineage mismatch")
    input_digest = json_digest(
        {
            "version": "p1-analysis-input-v2",
            "analysis_config_digest": config_digest,
            "observation_std_dbz": tensor_digest(observations.std_dbz),
            "quality_weight": tensor_digest(observations.quality_weight),
            "neural_prior_application_digest": (
                result.run.prior_application_digest
            ),
        }
    )
    if input_digest != result.run.analysis_input_digest:
        raise ValueError("P1 linearization input lineage mismatch")
    if not torch.equal(
        analysis.active_field_index,
        frozen.active_field_index,
    ):
        raise ValueError("P1 linearization active controls mismatch")
    trajectory = _analysis_trajectory(analysis.control, frozen)
    state_values = (
        (analysis.state.echo_linear, trajectory.frames_linear[-1]),
        (analysis.state.displacement_yx, trajectory.displacement_yx),
        (
            analysis.state.log_growth_per_step,
            trajectory.log_growth_per_step,
        ),
    )
    if any(
        not torch.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=result.run.config.contract_absolute_tolerance,
        )
        for actual, expected in state_values
    ):
        raise ValueError("P1 linearization does not reproduce the analysis")
    stationarity = _linearization_stationarity(
        analysis.control,
        observations,
        frozen,
    )
    stored_stationarity = (
        ("residual norm", linearization.residual_norm, stationarity.residual_norm),
        ("gradient norm", linearization.gradient_norm, stationarity.gradient_norm),
        (
            "field gradient RMS",
            linearization.field_gradient_rms,
            stationarity.field_gradient_rms,
        ),
        (
            "field gradient maximum",
            linearization.field_gradient_max,
            stationarity.field_gradient_max,
        ),
        (
            "dynamics gradient maximum",
            linearization.dynamics_gradient_max,
            stationarity.dynamics_gradient_max,
        ),
        (
            "relative stationarity",
            linearization.relative_stationarity,
            stationarity.relative_stationarity,
        ),
    )
    tolerance = 64.0 * torch.finfo(analysis.control.dtype).eps
    for name, stored, actual in stored_stationarity:
        if not (
            math.isfinite(stored)
            and math.isfinite(actual)
            and math.isclose(
                stored,
                actual,
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
        ):
            raise ValueError(f"P1 linearization {name} mismatch")
    analysis_stationarity = (
        analysis.linearization_residual_norm,
        analysis.linearization_gradient_norm,
        analysis.linearization_field_gradient_rms,
        analysis.linearization_field_gradient_max,
        analysis.linearization_dynamics_gradient_max,
        analysis.linearization_relative_stationarity,
        analysis.robust_gradient_norm,
        analysis.robust_field_gradient_rms,
        analysis.robust_field_gradient_max,
        analysis.robust_dynamics_gradient_max,
        analysis.robust_relative_stationarity,
        analysis.irls_relative_weight_change,
        analysis.linearization_polish_iterations,
    )
    retained_stationarity = (
        linearization.residual_norm,
        linearization.gradient_norm,
        linearization.field_gradient_rms,
        linearization.field_gradient_max,
        linearization.dynamics_gradient_max,
        linearization.relative_stationarity,
        linearization.robust_gradient_norm,
        linearization.robust_field_gradient_rms,
        linearization.robust_field_gradient_max,
        linearization.robust_dynamics_gradient_max,
        linearization.robust_relative_stationarity,
        linearization.irls_relative_weight_change,
        linearization.polish_iterations,
    )
    if analysis_stationarity != retained_stationarity:
        raise ValueError("P1 analysis linearization diagnostics mismatch")
    if not _stationarity_is_acceptable(
        stationarity,
        block_tolerance=(
            frozen.analysis_config
            .final_linearization_relative_stationarity_tolerance
        ),
        field_max_tolerance=(
            frozen.analysis_config.final_field_gradient_max_tolerance
        ),
    ):
        raise ValueError("P1 final linearization is not stationary")
    robust = _robust_stationarity(
        analysis.control,
        observations,
        frozen,
    )
    refreshed = freeze_irls_weights(
        analysis.control,
        observations,
        frozen,
    )
    weight_change = _relative_irls_weight_change(frozen, refreshed)
    robust_diagnostics = (
        (
            "robust gradient norm",
            linearization.robust_gradient_norm,
            robust.gradient_norm,
        ),
        (
            "robust field gradient RMS",
            linearization.robust_field_gradient_rms,
            robust.field_gradient_rms,
        ),
        (
            "robust field gradient maximum",
            linearization.robust_field_gradient_max,
            robust.field_gradient_max,
        ),
        (
            "robust dynamics gradient maximum",
            linearization.robust_dynamics_gradient_max,
            robust.dynamics_gradient_max,
        ),
        (
            "robust relative stationarity",
            linearization.robust_relative_stationarity,
            robust.relative_stationarity,
        ),
        (
            "IRLS relative weight change",
            linearization.irls_relative_weight_change,
            weight_change,
        ),
    )
    for name, stored, actual in robust_diagnostics:
        if not math.isclose(
            stored,
            actual,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError(f"P1 linearization {name} mismatch")
    if (
        not _stationarity_is_acceptable(
            robust,
            block_tolerance=(
                frozen.analysis_config
                .final_robust_relative_stationarity_tolerance
            ),
            field_max_tolerance=(
                frozen.analysis_config.final_field_gradient_max_tolerance
            ),
        )
        or weight_change
        > frozen.analysis_config.final_irls_relative_weight_tolerance
    ):
        raise ValueError("P1 linearization is not a robust IRLS fixed point")


def forecast_metric(
    name: str,
    forecast_linear: Tensor,
    truth_linear: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> Tensor:
    """Evaluate one differentiable forecast metric."""

    if name == "log_echo_mse":
        floor = 10.0 ** (nowcast_config.min_dbz / 10.0)
        difference = torch.log(forecast_linear + floor) - torch.log(
            truth_linear + floor
        )
        return _masked_mean(difference.square(), valid)
    if name == "soft_fss_error_35":
        return _soft_fss_error(
            forecast_linear,
            truth_linear,
            valid,
            nowcast_config,
            sensitivity_config,
            grid_time_contract,
        )
    if name == "centroid_error":
        forecast_center = _soft_centroid(forecast_linear, valid)
        truth_center = _soft_centroid(truth_linear, valid)
        return torch.sum((forecast_center - truth_center).square())
    if name == "centroid_error_m2":
        if grid_time_contract is None:
            raise ValueError("centroid_error_m2 requires a grid contract")
        forecast_center = _soft_projected_centroid(
            forecast_linear,
            valid,
            grid_time_contract,
        )
        truth_center = _soft_projected_centroid(
            truth_linear,
            valid,
            grid_time_contract,
        )
        return torch.sum((forecast_center - truth_center).square())
    raise ValueError(f"unsupported metric: {name}")


def extract_context_features(
    latest_frame_dbz: Tensor,
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
    *,
    latest_observation_mask: Tensor,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> Tensor:
    """Extract a small auditable context vector for later retrieval."""

    latest_valid = (
        torch.isfinite(latest_frame_dbz) & latest_observation_mask
    )
    latest = torch.nan_to_num(
        latest_frame_dbz,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    active = latest_valid & (latest >= config.echo_threshold_dbz)
    strong = latest_valid & (latest >= 35.0)
    valid_values = latest[latest_valid]
    active_values = latest[active]
    if active_values.numel():
        q90 = torch.quantile(active_values, 0.9)
    else:
        q90 = latest.new_tensor(config.min_dbz)
    if valid_values.numel():
        latest_mean = valid_values.mean()
        latest_max = valid_values.max()
    else:
        latest_mean = latest.new_tensor(config.min_dbz)
        latest_max = latest.new_tensor(config.min_dbz)

    border_width = max(1, min(latest.shape) // 16)
    border = torch.zeros_like(active)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    active_count = active.sum().clamp_min(1)
    boundary_fraction = (active & border).sum() / active_count

    linear = dbz_to_echo(
        latest,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    center = torch.nan_to_num(
        _soft_centroid(linear, latest_valid),
        nan=0.0,
    )
    motion = state.displacement_yx
    valid_count = latest_valid.sum().clamp_min(1)
    support_fraction = metadata.source_support.to(latest).mean()
    tendency_observation = latest.new_tensor(
        float(metadata.tendency_source is TendencySource.OBSERVATION)
    )
    tendency_background = latest.new_tensor(
        float(metadata.tendency_source is TendencySource.BACKGROUND)
    )
    state_path_observation = latest.new_tensor(
        float(metadata.state_path_source is TendencySource.OBSERVATION)
    )
    state_path_background = latest.new_tensor(
        float(metadata.state_path_source is TendencySource.BACKGROUND)
    )
    pair_selection_features = tuple(
        latest.new_tensor(
            float(metadata.motion_pair_selection is selection)
        )
        for selection in TendencyPairSelection
    ) + tuple(
        latest.new_tensor(
            float(metadata.growth_pair_selection is selection)
        )
        for selection in TendencyPairSelection
    )
    state_path_selection_features = tuple(
        latest.new_tensor(float(metadata.state_path_mode is selection))
        for selection in TendencyPairSelection
    )
    state_path_age_available = metadata.state_path_age_minutes is not None
    state_path_psr_available = math.isfinite(metadata.state_path_minimum_psr)
    growth_support_available = math.isfinite(
        metadata.minimum_growth_overlap_support
    )
    growth_area_available = math.isfinite(
        metadata.minimum_growth_overlap_area_km2
    )
    observation_path_age_available = (
        metadata.observation_path.age_minutes is not None
    )
    observation_path_psr_available = math.isfinite(
        metadata.observation_path.minimum_psr
    )
    background_path_age_available = (
        metadata.background_path.age_minutes is not None
    )
    background_path_psr_available = math.isfinite(
        metadata.background_path.minimum_psr
    )
    psr_available = metadata.tendency_pair_count > 0 and bool(
        torch.isfinite(metadata.minimum_phase_correlation_psr)
    )
    finite_minimum_psr = torch.nan_to_num(
        metadata.minimum_phase_correlation_psr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    disagreement_mps_available = bool(
        torch.isfinite(metadata.motion_disagreement_mps)
    )
    finite_disagreement_mps = torch.nan_to_num(
        metadata.motion_disagreement_mps,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    grid_available = grid_time_contract is not None
    if grid_time_contract is None:
        projected_velocity = latest.new_zeros(2)
        area_weighted_echo = latest.new_zeros(())
        grid_spacing = latest.new_zeros(2)
    else:
        projected_velocity = grid_time_contract.projected_velocity_xy(
            state.displacement_yx,
            config.interval_minutes,
        ).to(latest)
        area_weighted_echo = linear[latest_valid].sum() * (
            grid_time_contract.cell_area_m2 / 1.0e6
        )
        grid_spacing = latest.new_tensor(
            (grid_time_contract.dx_m, grid_time_contract.dy_m)
        )
    return torch.stack(
        (
            motion[0],
            motion[1],
            torch.linalg.vector_norm(motion),
            state.log_growth_per_step,
            metadata.motion_disagreement_px,
            metadata.growth_disagreement,
            latest.new_tensor(float(metadata.motion_pair_conflict)),
            latest.new_tensor(float(metadata.growth_pair_conflict)),
            latest.new_tensor(float(metadata.tendency_pair_count)),
            tendency_observation,
            tendency_background,
            latest.new_tensor(float(metadata.state_path_pair_count)),
            state_path_observation,
            state_path_background,
            latest.new_tensor(float(metadata.state_path_conflict)),
            latest.new_tensor(float(metadata.state_path_extrapolated)),
            latest.new_tensor(float(state_path_age_available)),
            latest.new_tensor(metadata.state_path_age_minutes or 0.0),
            latest.new_tensor(float(state_path_psr_available)),
            latest.new_tensor(
                math.log1p(metadata.state_path_minimum_psr)
                if state_path_psr_available
                else 0.0
            ),
            latest.new_tensor(float(growth_support_available)),
            latest.new_tensor(
                math.log1p(metadata.minimum_growth_overlap_support)
                if growth_support_available
                else 0.0
            ),
            latest.new_tensor(float(growth_area_available)),
            latest.new_tensor(
                math.log1p(metadata.minimum_growth_overlap_area_km2)
                if growth_area_available
                else 0.0
            ),
            support_fraction,
            latest.new_tensor(metadata.background_contribution_fraction),
            metadata.coverage_by_frame[-1].to(latest),
            latest_mean,
            latest_max,
            q90,
            active.sum().to(latest.dtype) / valid_count,
            strong.sum().to(latest.dtype) / valid_count,
            boundary_fraction.to(latest.dtype),
            center[0],
            center[1],
            torch.log1p(linear[latest_valid].sum()),
            *pair_selection_features,
            *state_path_selection_features,
            latest.new_tensor(float(psr_available)),
            torch.log1p(finite_minimum_psr).to(latest),
            latest.new_tensor(float(grid_available)),
            projected_velocity[0],
            projected_velocity[1],
            torch.linalg.vector_norm(projected_velocity),
            latest.new_tensor(float(disagreement_mps_available)),
            finite_disagreement_mps.to(latest),
            latest.new_tensor(float(grid_available)),
            torch.log1p(area_weighted_echo),
            latest.new_tensor(float(grid_available)),
            grid_spacing[0],
            grid_spacing[1],
            latest.new_tensor(float(metadata.observation_path.pair_count)),
            latest.new_tensor(float(metadata.observation_path.conflict)),
            latest.new_tensor(float(metadata.observation_path.extrapolated)),
            latest.new_tensor(float(observation_path_age_available)),
            latest.new_tensor(metadata.observation_path.age_minutes or 0.0),
            latest.new_tensor(float(observation_path_psr_available)),
            latest.new_tensor(
                math.log1p(metadata.observation_path.minimum_psr)
                if observation_path_psr_available
                else 0.0
            ),
            latest.new_tensor(float(metadata.background_path.pair_count)),
            latest.new_tensor(float(metadata.background_path.conflict)),
            latest.new_tensor(float(metadata.background_path.extrapolated)),
            latest.new_tensor(float(background_path_age_available)),
            latest.new_tensor(metadata.background_path.age_minutes or 0.0),
            latest.new_tensor(float(background_path_psr_available)),
            latest.new_tensor(
                math.log1p(metadata.background_path.minimum_psr)
                if background_path_psr_available
                else 0.0
            ),
        )
    ).detach()


def _state_from_control(
    template: RadarState,
    control: Tensor,
    echo: Tensor,
) -> RadarState:
    return RadarState(
        echo_linear=echo,
        displacement_yx=control[:2],
        log_growth_per_step=control[2],
    )


def _freeze_output_cap(
    forecast: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor]:
    """Apply the issued dBZ cap and freeze its nominal active set."""

    maximum = _maximum_linear_echo(forecast, config)
    active = (forecast < maximum).detach()
    return torch.where(active, forecast, maximum), active


def _apply_output_cap(
    forecast: Tensor,
    active: Tensor,
    config: NowcastConfig,
) -> Tensor:
    maximum = _maximum_linear_echo(forecast, config)
    return torch.where(active, forecast, maximum)


def _maximum_linear_echo(reference: Tensor, config: NowcastConfig) -> Tensor:
    floor = 10.0 ** (config.min_dbz / 10.0)
    maximum = 10.0 ** (config.max_dbz / 10.0) - floor
    return reference.new_tensor(maximum)


def _metric_has_support(
    name: str,
    forecast: Tensor,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> bool:
    """Decide metric support once, before any differentiation."""

    if not bool(torch.any(valid)):
        return False
    if name == "soft_fss_error_35":
        floor = 10.0 ** (nowcast_config.min_dbz / 10.0)
        truth_dbz = 10.0 * torch.log10(truth + floor)
        truth_event = torch.sigmoid(
            (truth_dbz - 35.0)
            / sensitivity_config.soft_fss_temperature_dbz
        )
        truth_mass = torch.sum(truth_event * valid.to(truth.dtype))
        return bool(truth_mass >= sensitivity_config.minimum_fss_truth_mass)
    if name in ("centroid_error", "centroid_error_m2"):
        forecast_mass = torch.sum(
            torch.log1p(forecast) * valid.to(forecast.dtype)
        )
        truth_mass = torch.sum(torch.log1p(truth) * valid.to(truth.dtype))
        return bool(
            (forecast_mass > sensitivity_config.epsilon)
            & (truth_mass > sensitivity_config.epsilon)
        )
    return True


def _soft_fss_error(
    forecast: Tensor,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> Tensor:
    floor = 10.0 ** (nowcast_config.min_dbz / 10.0)
    forecast_dbz = 10.0 * torch.log10(forecast + floor)
    truth_dbz = 10.0 * torch.log10(truth + floor)
    temperature = sensitivity_config.soft_fss_temperature_dbz
    forecast_event = torch.sigmoid((forecast_dbz - 35.0) / temperature)
    truth_event = torch.sigmoid((truth_dbz - 35.0) / temperature)

    valid_float = valid.to(forecast.dtype)
    local_valid = _soft_fss_average(
        valid_float,
        sensitivity_config,
        grid_time_contract,
    )
    denominator = local_valid.clamp_min(sensitivity_config.epsilon)
    forecast_fraction = _soft_fss_average(
        forecast_event * valid_float,
        sensitivity_config,
        grid_time_contract,
    ) / denominator
    truth_fraction = _soft_fss_average(
        truth_event * valid_float,
        sensitivity_config,
        grid_time_contract,
    ) / denominator
    numerator = _weighted_mean(
        (forecast_fraction - truth_fraction).square(),
        local_valid,
        sensitivity_config.epsilon,
    )
    reference = _weighted_mean(
        forecast_fraction.square() + truth_fraction.square(),
        local_valid,
        sensitivity_config.epsilon,
    )
    return numerator / (reference + sensitivity_config.epsilon)


def _soft_fss_average(
    values: Tensor,
    config: SensitivityConfig,
    grid: RadarGridTimeContract | None,
) -> Tensor:
    if config.soft_fss_window_m is None:
        window = config.soft_fss_window
        return F.avg_pool2d(
            values[None, None],
            window,
            stride=1,
            padding=window // 2,
        )[0, 0]
    if grid is None:
        raise ValueError("physical FSS requires a grid contract")
    return _affine_footprint_average(
        values,
        grid,
        0.5 * config.soft_fss_window_m,
    )


def _affine_footprint_average(
    values: Tensor,
    grid: RadarGridTimeContract,
    radius_m: float,
) -> Tensor:
    """Average over the exact projected-distance footprint of one grid."""

    radius_y, radius_x = grid.pixel_radius_yx(radius_m)
    offsets = grid.pixel_offsets_within_distance(
        radius_m,
        maximum_radius_yx=(radius_y, radius_x),
    )
    kernel = values.new_zeros((2 * radius_y + 1, 2 * radius_x + 1))
    for row, column in offsets:
        kernel[row + radius_y, column + radius_x] = 1.0
    kernel /= len(offsets)
    return F.conv2d(
        values[None, None],
        kernel[None, None],
        padding=(radius_y, radius_x),
    )[0, 0]


def _weighted_mean(values: Tensor, weights: Tensor, epsilon: float) -> Tensor:
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(epsilon)


def _soft_centroid(echo: Tensor, valid: Tensor) -> Tensor:
    height, width = echo.shape
    y = torch.linspace(-1.0, 1.0, height, dtype=echo.dtype, device=echo.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=echo.dtype, device=echo.device)
    weights = torch.log1p(echo) * valid.to(echo.dtype)
    total = weights.sum()
    safe_total = total.clamp_min(torch.finfo(echo.dtype).eps)
    center = torch.stack(
        (
            torch.sum(weights * y[:, None]) / safe_total,
            torch.sum(weights * x[None, :]) / safe_total,
        )
    )
    return torch.where(
        total > torch.finfo(echo.dtype).eps,
        center,
        torch.full_like(center, float("nan")),
    )


def _soft_projected_centroid(
    echo: Tensor,
    valid: Tensor,
    grid: RadarGridTimeContract,
) -> Tensor:
    """Return the echo centroid in projected metres; origin cancels in errors."""

    height, width = echo.shape
    row = torch.arange(height, dtype=echo.dtype, device=echo.device)
    column = torch.arange(width, dtype=echo.dtype, device=echo.device)
    weights = torch.log1p(echo) * valid.to(echo.dtype)
    total = weights.sum()
    safe_total = total.clamp_min(torch.finfo(echo.dtype).eps)
    center_column_row = torch.stack(
        (
            torch.sum(weights * column[None, :]) / safe_total,
            torch.sum(weights * row[:, None]) / safe_total,
        )
    )
    assert grid.pixel_to_projected_matrix_m is not None
    matrix = echo.new_tensor(grid.pixel_to_projected_matrix_m)
    center = matrix @ center_column_row
    return torch.where(
        total > torch.finfo(echo.dtype).eps,
        center,
        torch.full_like(center, float("nan")),
    )


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    weights = valid.to(values.dtype)
    count = weights.sum()
    mean = torch.sum(values * weights) / count.clamp_min(
        torch.finfo(values.dtype).tiny
    )
    return torch.where(
        count > 0,
        mean,
        torch.full_like(mean, float("nan")),
    )


def _tile_l2(values: Tensor, tile_size: TileShape) -> Tensor:
    tiles = _as_tiles(values, tile_size)
    return torch.sqrt(torch.sum(tiles.square(), dim=(-1, -2)))


def _tile_sum(values: Tensor, tile_size: TileShape) -> Tensor:
    return torch.sum(_as_tiles(values, tile_size), dim=(-1, -2))


def _as_tiles(values: Tensor, tile_size: TileShape) -> Tensor:
    height, width = values.shape
    tile_height, tile_width = tile_size
    tile_rows = math.ceil(height / tile_height)
    tile_columns = math.ceil(width / tile_width)
    padded = F.pad(
        values,
        (
            0,
            tile_columns * tile_width - width,
            0,
            tile_rows * tile_height - height,
        ),
    )
    return padded.reshape(
        tile_rows,
        tile_height,
        tile_columns,
        tile_width,
    ).permute(0, 2, 1, 3)


def _frozen_observation(
    latest_frame: Tensor,
    accepted: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> tuple[Tensor, Tensor]:
    finite = torch.isfinite(latest_frame)
    clean = torch.nan_to_num(
        latest_frame,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    ).clamp(nowcast_config.min_dbz, nowcast_config.max_dbz)

    margin = sensitivity_config.active_margin_dbz
    latest_active = (
        finite
        & accepted
        & (clean > nowcast_config.min_dbz + margin)
        & (clean < nowcast_config.max_dbz - margin)
    )
    return clean.detach(), latest_active.detach()


def _active_dbz_to_echo(
    candidate_dbz: Tensor,
    nominal_dbz: Tensor,
    nominal_echo: Tensor,
    active: Tensor,
    config: NowcastConfig,
) -> Tensor:
    """Apply dBZ perturbations only where the frozen active set permits."""

    safe_dbz = torch.where(
        active,
        candidate_dbz,
        torch.zeros_like(candidate_dbz),
    )
    nominal_safe_dbz = torch.where(
        active,
        nominal_dbz,
        torch.zeros_like(nominal_dbz),
    )
    candidate_echo = dbz_to_echo(
        safe_dbz,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    nominal_active_echo = dbz_to_echo(
        nominal_safe_dbz,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    ).detach()
    perturbation = torch.where(
        active,
        candidate_echo - nominal_active_echo,
        torch.zeros_like(candidate_echo),
    )
    return nominal_echo.detach() + perturbation


def _observation_std(
    value: float | Tensor | None,
    frames: Tensor,
    epsilon: float,
) -> tuple[Tensor, bool]:
    if value is None:
        return torch.ones_like(frames), False
    if isinstance(value, (int, float)):
        result = torch.full_like(frames, float(value))
    else:
        result = value.to(dtype=frames.dtype, device=frames.device)
        if result.ndim == 0:
            result = torch.full_like(frames, float(result))
        elif result.shape != frames.shape:
            raise ValueError("observation_std_dbz must match frames shape")
    if not bool(torch.all(torch.isfinite(result))) or bool(
        torch.any(result <= epsilon)
    ):
        raise ValueError("observation_std_dbz must be finite and positive")
    return result, True


def _dbz_innovation(
    latest_frame: Tensor,
    background: Tensor | None,
    accepted: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor | None, Tensor | None]:
    if background is None:
        return None, None
    valid = (
        torch.isfinite(latest_frame)
        & torch.isfinite(background)
        & accepted
    )
    clean_frame = torch.nan_to_num(
        latest_frame,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    clean_background = torch.nan_to_num(
        background,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    innovation = torch.where(
        valid,
        clean_frame - clean_background,
        torch.full_like(latest_frame, float("nan")),
    )
    return innovation.detach(), valid.detach()


def _full_map_indices(
    selected_minutes: tuple[int, ...],
    all_minutes: tuple[int, ...],
) -> tuple[int, ...]:
    unknown = set(selected_minutes) - set(all_minutes)
    if unknown:
        raise ValueError(f"full-map leads outside forecast horizon: {sorted(unknown)}")
    return tuple(all_minutes.index(value) for value in selected_minutes)


def _metric_evidence_ratios(
    sensitivity_weight: Tensor,
    source_support: Tensor,
    forecast_confidence: Tensor,
    observation_source_support: Tensor,
    observation_verified_confidence: Tensor,
    background_verified_confidence: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    denominator = (sensitivity_weight * source_support).sum()
    if float(denominator) <= epsilon:
        return None

    def weighted_fraction(evidence: Tensor) -> Tensor:
        return (sensitivity_weight * evidence).sum() / denominator

    return (
        weighted_fraction(forecast_confidence),
        weighted_fraction(observation_source_support),
        weighted_fraction(observation_verified_confidence),
        weighted_fraction(background_verified_confidence),
    )


def _trust_components(
    template: RadarState,
    metadata: ForecastMetadata,
    control: Tensor,
    echo: Tensor,
    truth: Tensor,
    valid: Tensor,
    gradients: Tensor,
    metric_available: Tensor,
    cap_masks: Tensor,
    observation_verified_evidence_by_metric: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> dict[str, float]:
    verification_quality = valid.to(echo.dtype).mean().clamp(0.0, 1.0)
    support_quality = metric_available.to(echo.dtype).mean()
    conflict_count = int(metadata.motion_pair_conflict) + int(
        metadata.growth_pair_conflict
    )
    pair_consistency_quality = (
        sensitivity_config.pair_conflict_trust_penalty**conflict_count
    )
    evidence_available = metric_available & torch.isfinite(
        observation_verified_evidence_by_metric
    )
    if bool(torch.any(evidence_available)):
        observation_verified_evidence_quality = float(
            observation_verified_evidence_by_metric[evidence_available]
            .mean()
            .clamp(0.0, 1.0)
        )
    else:
        observation_verified_evidence_quality = 0.0
    if not bool(torch.any(metric_available)):
        return {
            "linearity": 0.0,
            "verification": float(verification_quality),
            "metric_support": 0.0,
            "pair_consistency": pair_consistency_quality,
            "observation_verified_evidence": (
                observation_verified_evidence_quality
            ),
        }

    delta = control.new_tensor(sensitivity_config.linearity_delta)
    predicted_change = torch.sum(gradients[metric_available].mean(dim=0) * delta)

    def aggregate(candidate_control: Tensor) -> Tensor:
        candidate_state = _state_from_control(
            template,
            candidate_control,
            echo,
        )
        scores: list[Tensor] = []
        for lead_index in range(nowcast_config.forecast_steps):
            latent_forecast = forecast_linear_at_step(
                candidate_state,
                lead_index + 1,
                nowcast_config,
            )
            forecast = _apply_output_cap(
                latent_forecast,
                cap_masks[lead_index],
                nowcast_config,
            )
            for metric_index, name in enumerate(
                sensitivity_config.metric_names
            ):
                if not bool(metric_available[lead_index, metric_index]):
                    continue
                scores.append(
                    forecast_metric(
                        name,
                        forecast,
                        truth[lead_index],
                        valid[lead_index],
                        nowcast_config,
                        sensitivity_config,
                        grid_time_contract,
                    )
                )
        return torch.stack(scores).mean()

    actual_change = aggregate(control + delta) - aggregate(control)
    linearity_error = torch.abs(actual_change - predicted_change) / (
        torch.abs(actual_change)
        + torch.abs(predicted_change)
        + sensitivity_config.epsilon
    )
    linearity_quality = torch.nan_to_num(
        torch.exp(-linearity_error / 0.25),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    return {
        "linearity": float(linearity_quality.detach()),
        "verification": float(verification_quality.detach()),
        "metric_support": float(support_quality.detach()),
        "pair_consistency": pair_consistency_quality,
        "observation_verified_evidence": (
            observation_verified_evidence_quality
        ),
    }


def _validate_inputs(
    latest_frame: Tensor,
    verification: Tensor,
    state: RadarState,
    config: NowcastConfig,
    background: Tensor | None,
) -> None:
    if latest_frame.ndim != 2:
        raise ValueError("latest_frame_dbz must have shape [height, width]")
    expected = (config.forecast_steps, *latest_frame.shape)
    if tuple(verification.shape) != expected:
        raise ValueError(f"verification_frames_dbz must have shape {expected}")
    if tuple(state.echo_linear.shape) != tuple(latest_frame.shape):
        raise ValueError("state grid must match frame grid")
    if background is not None and background.shape != latest_frame.shape:
        raise ValueError(
            "latest_background_dbz must match latest_frame_dbz shape"
        )
    if (
        not latest_frame.is_floating_point()
        or not verification.is_floating_point()
    ):
        raise TypeError(
            "latest frame and verification must be floating-point tensors"
        )
    if background is not None and not background.is_floating_point():
        raise TypeError("latest_background_dbz must be floating-point")
    if state.displacement_yx.shape != (2,):
        raise ValueError("state displacement must have shape [2]")
    if state.log_growth_per_step.ndim != 0:
        raise ValueError("state log growth must be scalar")
    if latest_frame.device != verification.device:
        raise ValueError(
            "latest frame and verification must use the same device"
        )
    state_tensors = (
        state.echo_linear,
        state.displacement_yx,
        state.log_growth_per_step,
    )
    if any(tensor.device != latest_frame.device for tensor in state_tensors):
        raise ValueError("state and latest frame must use the same device")
    if any(not tensor.is_floating_point() for tensor in state_tensors):
        raise TypeError("state tensors must be floating-point")
    if background is not None and background.device != latest_frame.device:
        raise ValueError(
            "background and latest frame must use the same device"
        )
