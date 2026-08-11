"""Fail-closed holdout evidence for learned radar priors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
from typing import Any, cast, Literal

import torch
from torch import Tensor, nn
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ._digest import json_digest, tensor_digest
from ._runtime import numerical_runtime_identity_digest
from .calibration import OperationalDataIdentity
from .nowcast import (
    ForecastResult,
    ForecastRunContract,
    _forecast_input_plan_resolution_digest,
)
from .range_geometry import (
    RangeGeometryContract,
    RangePartitionEvidence,
    resolve_range_geometry,
)
from .sensitivity import (
    SensitivityConfig,
    VerificationBundle,
    _ResolvedVerification,
    _forecast_result_content_digest,
    _load_learning_policy_trust_store,
    _metric_domain_weight,
    _resolve_verification,
    _resolved_forecast_domain_weights,
    _resolved_forecast_scores,
)
from .variational import (
    _connected_component_flat_indices,
    _export_graph,
    _module_state_digest,
    _new_neural_prior_deployment_selection,
    NeuralPriorApplication,
    NeuralPriorDeploymentSelection,
    NeuralPriorInferenceRunner,
)


PromotionRejectionReason = Literal[
    "unapproved_promotion_policy",
    "unapproved_candidate_manifest",
    "unapproved_holdout_plan",
    "unapproved_metric_contract",
    "insufficient_holdout_cases",
    "insufficient_material_cases",
    "insufficient_material_case_fraction",
    "insufficient_independent_cases",
    "insufficient_distinct_storms",
    "insufficient_distinct_days",
    "insufficient_distinct_radars",
    "insufficient_distinct_regimes",
    "insufficient_distinct_range_regimes",
    "insufficient_material_clusters",
    "no_material_outcome",
    "insufficient_beneficial_fraction",
    "excessive_harmful_fraction",
    "insufficient_mean_improvement",
    "excessive_single_degradation",
    "excessive_end_to_end_degradation",
    "excessive_issuance_change",
    "unreliable_prior_uncertainty",
    "inferior_prior_uncertainty",
    "insufficient_prior_echo_cases",
    "insufficient_prior_clear_cases",
    "insufficient_uncertainty_clusters",
    "insufficient_echo_clusters",
    "insufficient_clear_clusters",
    "insufficient_component_samples",
    "insufficient_component_area",
    "insufficient_echo_objects",
    "insufficient_bootstrap_tail_resolution",
    "unreliable_state_head",
    "inferior_state_head",
    "insufficient_state_calibration",
    "unreliable_regime_classifier",
    "unreliable_range_classifier",
    "ambiguous_regime_classifier_branch",
]

_UncertaintyComponent = Literal[
    "intensity",
    "pit_residual",
    "support",
    "echo_miss",
    "object_miss",
    "clear",
    "underdispersion",
    "state_nll",
    "state_pit_residual",
    "state_underdispersion",
    "state_support",
    "state_echo_miss",
    "state_object_miss",
    "state_false_support",
    "state_valid",
]

_UNCERTAINTY_COMPONENT_NAMES: tuple[str, ...] = (
    "intensity",
    "pit_residual",
    "support",
    "echo_miss",
    "object_miss",
    "clear",
    "underdispersion",
    "state_nll",
    "state_pit_residual",
    "state_underdispersion",
    "state_support",
    "state_echo_miss",
    "state_object_miss",
    "state_false_support",
    "state_valid",
)
PriorComponentStatus = Literal["available", "not_applicable"]


def _require_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _quantized_lattice_aligned(
    value: float,
    resolution: float,
    origin: float,
) -> bool:
    index = round((value - origin) / resolution)
    reconstructed = origin + index * resolution
    tolerance = max(
        1.0e-7,
        math.ulp(1.0) * max(abs(value), abs(reconstructed), 1.0) * 32.0,
    )
    return abs(value - reconstructed) <= tolerance


@dataclass(frozen=True)
class PromotionMetricScale:
    """Dimensionally valid scale and materiality for one error metric."""

    metric_name: str
    scale: float
    material_change: float
    weight: float = 1.0
    maximum_normalized_degradation: float = 1.0
    maximum_end_to_end_normalized_degradation: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValueError("promotion metric name must be nonempty")
        for name, value in (
            ("scale", self.scale),
            ("material_change", self.material_change),
            ("weight", self.weight),
            (
                "maximum_normalized_degradation",
                self.maximum_normalized_degradation,
            ),
            (
                "maximum_end_to_end_normalized_degradation",
                self.maximum_end_to_end_normalized_degradation,
            ),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"promotion metric {name} must be positive")


@dataclass(frozen=True)
class NeuralPriorInputPlan:
    """Content-addressed future input-selection rules, never future data."""

    valid_times: tuple[str, ...]
    grid_contract_digest: str
    radar_product_digest: str
    qc_pipeline_digest: str
    background_cycle_rule_digest: str
    mask_policy_digest: str
    observation_valid_time: str
    input_available_time: str
    decision_deadline: str
    publication_time: str
    contract: str = "neural-prior-input-plan-v2"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "grid_contract_digest",
            "radar_product_digest",
            "qc_pipeline_digest",
            "background_cycle_rule_digest",
            "mask_policy_digest",
        ):
            _require_digest(name, getattr(self, name))
        times = tuple(_canonical_time(value) for value in self.valid_times)
        valid = _canonical_time(self.observation_valid_time)
        available = _canonical_time(self.input_available_time)
        deadline = _canonical_time(self.decision_deadline)
        publication = _canonical_time(self.publication_time)
        if not times or times[-1] != valid:
            raise ValueError("input plan must end at its observation valid time")
        if not valid <= available <= deadline < publication:
            raise ValueError("input plan latency window is invalid")
        object.__setattr__(self, "valid_times", times)
        object.__setattr__(self, "observation_valid_time", valid)
        object.__setattr__(self, "input_available_time", available)
        object.__setattr__(self, "decision_deadline", deadline)
        object.__setattr__(self, "publication_time", publication)
        object.__setattr__(self, "plan_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "valid_times": list(self.valid_times),
            "grid_contract_digest": self.grid_contract_digest,
            "radar_product_digest": self.radar_product_digest,
            "qc_pipeline_digest": self.qc_pipeline_digest,
            "background_cycle_rule_digest": self.background_cycle_rule_digest,
            "mask_policy_digest": self.mask_policy_digest,
            "observation_valid_time": self.observation_valid_time,
            "input_available_time": self.input_available_time,
            "decision_deadline": self.decision_deadline,
            "publication_time": self.publication_time,
        }

    @property
    def json(self) -> str:
        return json.dumps(self.payload, sort_keys=True, separators=(",", ":"))


PriorUncertaintyTargetKind = Literal[
    "independent_sensor",
    "withheld_radar",
    "leave_one_time_out",
    "withheld_target_mask",
]


@dataclass(frozen=True)
class PriorUncertaintyTargetPlan:
    """Pre-registered source contract for an independent uncertainty target."""

    plan_id: str
    target_kind: PriorUncertaintyTargetKind
    source_identity_digest: str
    qc_pipeline_digest: str
    mask_policy_digest: str
    censor_policy_digest: str
    floor_representation_contract_digest: str
    grid_contract_digest: str
    feature_exclusion_contract_digest: str
    independence_evidence_digest: str
    target_valid_time: str
    prior_probability_contract_digest: str
    support_threshold_dbz: float = 5.0
    reflectivity_resolution_dbz: float = 0.5
    quantization_origin_dbz: float = -10.0
    threshold_bin_convention: Literal["nearest_rounding_threshold_censor"] = (
        "nearest_rounding_threshold_censor"
    )
    contract: str = "prior-uncertainty-target-plan-v6"
    support_event_digest: str = field(init=False)
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "prior-uncertainty-target-plan-v6":
            raise ValueError("unsupported uncertainty target plan")
        if not self.plan_id or self.plan_id.strip() != self.plan_id:
            raise ValueError("uncertainty target plan ID must be canonical")
        if self.target_kind not in (
            "independent_sensor",
            "withheld_radar",
            "leave_one_time_out",
            "withheld_target_mask",
        ):
            raise ValueError("unsupported uncertainty target kind")
        for name in (
            "source_identity_digest",
            "qc_pipeline_digest",
            "mask_policy_digest",
            "censor_policy_digest",
            "floor_representation_contract_digest",
            "grid_contract_digest",
            "feature_exclusion_contract_digest",
            "independence_evidence_digest",
            "prior_probability_contract_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            not math.isfinite(self.support_threshold_dbz)
            or not math.isfinite(self.reflectivity_resolution_dbz)
            or self.reflectivity_resolution_dbz <= 0.0
            or not math.isfinite(self.quantization_origin_dbz)
            or not _quantized_lattice_aligned(
                self.support_threshold_dbz,
                self.reflectivity_resolution_dbz,
                self.quantization_origin_dbz,
            )
            or self.threshold_bin_convention
            != "nearest_rounding_threshold_censor"
        ):
            raise ValueError("uncertainty support threshold must be finite")
        support_event_digest = json_digest(
            {
                "contract": "radar-support-event-v2",
                "variable": "radar_reflectivity_dbz",
                "operator": ">=",
                "threshold_dbz": self.support_threshold_dbz,
                "support_product_digest": self.source_identity_digest,
                "qc_pipeline_digest": self.qc_pipeline_digest,
                "reflectivity_resolution_dbz": (
                    self.reflectivity_resolution_dbz
                ),
                "quantization_origin_dbz": self.quantization_origin_dbz,
                "threshold_bin_convention": self.threshold_bin_convention,
            }
        )
        object.__setattr__(
            self,
            "target_valid_time",
            _canonical_time(self.target_valid_time),
        )
        object.__setattr__(self, "support_event_digest", support_event_digest)
        object.__setattr__(self, "plan_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "plan_id": self.plan_id,
            "target_kind": self.target_kind,
            "source_identity_digest": self.source_identity_digest,
            "qc_pipeline_digest": self.qc_pipeline_digest,
            "mask_policy_digest": self.mask_policy_digest,
            "censor_policy_digest": self.censor_policy_digest,
            "floor_representation_contract_digest": (
                self.floor_representation_contract_digest
            ),
            "grid_contract_digest": self.grid_contract_digest,
            "feature_exclusion_contract_digest": (
                self.feature_exclusion_contract_digest
            ),
            "independence_evidence_digest": self.independence_evidence_digest,
            "target_valid_time": self.target_valid_time,
            "prior_probability_contract_digest": (
                self.prior_probability_contract_digest
            ),
            "support_threshold_dbz": self.support_threshold_dbz,
            "reflectivity_resolution_dbz": self.reflectivity_resolution_dbz,
            "quantization_origin_dbz": self.quantization_origin_dbz,
            "threshold_bin_convention": self.threshold_bin_convention,
            "support_event_digest": self.support_event_digest,
        }


@dataclass(frozen=True)
class NeuralPriorStateCalibrationPlan:
    """Pre-registered withheld target for the P1-consumed state head."""

    plan_id: str
    target_kind: PriorUncertaintyTargetKind
    source_identity_digest: str
    qc_pipeline_digest: str
    mask_policy_digest: str
    censor_policy_digest: str
    floor_representation_contract_digest: str
    grid_contract_digest: str
    feature_exclusion_contract_digest: str
    independence_evidence_digest: str
    target_valid_time: str
    state_contract_digest: str
    support_threshold_dbz: float
    reflectivity_resolution_dbz: float = 0.5
    quantization_origin_dbz: float = -10.0
    threshold_bin_convention: Literal["nearest_rounding_threshold_censor"] = (
        "nearest_rounding_threshold_censor"
    )
    contract: str = "neural-prior-state-calibration-plan-v3"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-state-calibration-plan-v3"
            or not self.plan_id
            or self.plan_id.strip() != self.plan_id
            or self.target_kind not in (
                "independent_sensor",
                "withheld_radar",
                "leave_one_time_out",
                "withheld_target_mask",
            )
        ):
            raise ValueError("invalid neural-prior state calibration plan")
        for name in (
            "source_identity_digest",
            "qc_pipeline_digest",
            "mask_policy_digest",
            "censor_policy_digest",
            "floor_representation_contract_digest",
            "grid_contract_digest",
            "feature_exclusion_contract_digest",
            "independence_evidence_digest",
            "state_contract_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            not math.isfinite(self.support_threshold_dbz)
            or not math.isfinite(self.reflectivity_resolution_dbz)
            or self.reflectivity_resolution_dbz <= 0.0
            or not math.isfinite(self.quantization_origin_dbz)
            or not _quantized_lattice_aligned(
                self.support_threshold_dbz,
                self.reflectivity_resolution_dbz,
                self.quantization_origin_dbz,
            )
            or self.threshold_bin_convention
            != "nearest_rounding_threshold_censor"
        ):
            raise ValueError("invalid state calibration measurement contract")
        object.__setattr__(
            self, "target_valid_time", _canonical_time(self.target_valid_time)
        )
        object.__setattr__(self, "plan_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "plan_digest"
        }


@dataclass(frozen=True)
class RangeBandContract:
    """Pre-registered physical range zones for one holdout grid."""

    case_id: str
    range_regime_labels: tuple[str, ...]
    range_band_mask_digests: tuple[str, ...]
    reference_active_range_regimes: tuple[str, ...]
    grid_contract_digest: str
    range_geometry_contract_digest: str
    contract: str = "neural-prior-range-band-contract-v2"
    contract_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-range-band-contract-v2"
            or not self.case_id
            or self.case_id.strip() != self.case_id
            or not self.range_regime_labels
            or len(set(self.range_regime_labels)) != len(self.range_regime_labels)
            or any(not value or value.strip() != value for value in self.range_regime_labels)
            or len(self.range_band_mask_digests) != len(self.range_regime_labels)
            or len(set(self.reference_active_range_regimes))
            != len(self.reference_active_range_regimes)
            or any(
                value not in self.range_regime_labels
                for value in self.reference_active_range_regimes
            )
        ):
            raise ValueError("range-band contract is invalid")
        _require_digest("range-band grid contract", self.grid_contract_digest)
        _require_digest(
            "range geometry contract", self.range_geometry_contract_digest
        )
        for digest in self.range_band_mask_digests:
            _require_digest("range-band mask digest", digest)
        object.__setattr__(self, "contract_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "case_id": self.case_id,
            "range_regime_labels": list(self.range_regime_labels),
            "range_band_mask_digests": list(self.range_band_mask_digests),
            "reference_active_range_regimes": list(
                self.reference_active_range_regimes
            ),
            "grid_contract_digest": self.grid_contract_digest,
            "range_geometry_contract_digest": (
                self.range_geometry_contract_digest
            ),
        }

    def mask_digest(self, range_regime: str) -> str:
        try:
            index = self.range_regime_labels.index(range_regime)
        except ValueError as error:
            raise ValueError("range regime is outside the physical contract") from error
        return self.range_band_mask_digests[index]


def _validate_complete_range_partition(
    range_band_masks: dict[str, Tensor],
) -> None:
    """Require each operational grid cell to belong to exactly one band."""

    masks = tuple(range_band_masks.values())
    if (
        not masks
        or any(mask.dtype is not torch.bool or mask.ndim != 2 for mask in masks)
        or any(
            mask.shape != masks[0].shape or mask.device != masks[0].device
            for mask in masks[1:]
        )
    ):
        raise ValueError("range-band masks are not a complete partition")
    membership = torch.stack(tuple(mask.to(torch.int8) for mask in masks)).sum(
        dim=0
    )
    if bool(torch.any(membership != 1)):
        raise ValueError("range-band masks are not a complete partition")


@dataclass(frozen=True)
class RegimeClassifierManifest:
    """Training lineage for one preregistered deployment classifier."""

    classifier_digest: str
    training_dataset_digest: str
    training_case_ids: tuple[str, ...]
    training_input_bundle_digests: tuple[str, ...]
    training_full_analysis_input_digests: tuple[str, ...]
    training_physical_event_digests: tuple[str, ...]
    training_storm_ids: tuple[str, ...]
    training_days: tuple[str, ...]
    training_radar_ids: tuple[str, ...]
    training_grid_contract_digests: tuple[str, ...]
    training_time_windows: tuple[tuple[str, str], ...]
    training_algorithm_digest: str
    numerical_runtime_digest: str
    reference_label_contract_digest: str
    signed_training_member_manifest_digest: str
    contract: str = "neural-prior-regime-classifier-manifest-v3"
    manifest_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-regime-classifier-manifest-v3":
            raise ValueError("unsupported regime-classifier manifest")
        for name in (
            "classifier_digest",
            "training_dataset_digest",
            "training_algorithm_digest",
            "numerical_runtime_digest",
            "reference_label_contract_digest",
            "signed_training_member_manifest_digest",
        ):
            _require_digest(name, getattr(self, name))
        for name, values in (
            ("training case", self.training_case_ids),
            ("training storm", self.training_storm_ids),
            ("training day", self.training_days),
            ("training radar", self.training_radar_ids),
        ):
            if (
                not values
                or len(set(values)) != len(values)
                or any(not value or value.strip() != value for value in values)
            ):
                raise ValueError(f"classifier {name} identities are invalid")
        for name, values in (
            ("training input", self.training_input_bundle_digests),
            (
                "training full-analysis input",
                self.training_full_analysis_input_digests,
            ),
            ("training grid contract", self.training_grid_contract_digests),
            ("training physical event", self.training_physical_event_digests),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"classifier {name} digests are invalid")
            for value in values:
                _require_digest(name, value)
        windows = tuple(
            (_canonical_time(start), _canonical_time(end))
            for start, end in self.training_time_windows
        )
        if not windows or any(start >= end for start, end in windows):
            raise ValueError("classifier training windows are invalid")
        object.__setattr__(self, "training_time_windows", windows)
        object.__setattr__(self, "manifest_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "classifier_digest": self.classifier_digest,
            "training_dataset_digest": self.training_dataset_digest,
            "training_case_ids": list(self.training_case_ids),
            "training_input_bundle_digests": list(
                self.training_input_bundle_digests
            ),
            "training_full_analysis_input_digests": list(
                self.training_full_analysis_input_digests
            ),
            "training_physical_event_digests": list(
                self.training_physical_event_digests
            ),
            "training_storm_ids": list(self.training_storm_ids),
            "training_days": list(self.training_days),
            "training_radar_ids": list(self.training_radar_ids),
            "training_grid_contract_digests": list(
                self.training_grid_contract_digests
            ),
            "training_time_windows": [list(value) for value in self.training_time_windows],
            "training_algorithm_digest": self.training_algorithm_digest,
            "numerical_runtime_digest": self.numerical_runtime_digest,
            "reference_label_contract_digest": self.reference_label_contract_digest,
            "signed_training_member_manifest_digest": (
                self.signed_training_member_manifest_digest
            ),
        }


@dataclass(frozen=True)
class RegimeReferencePlan:
    """Preregistered rule for producing a weather-regime label after issue."""

    case_id: str
    labeler_id: str
    labeler_public_key_hex: str
    source_contract_digest: str
    labeling_valid_time: str
    adjudication_policy_digest: str
    contract: str = "neural-prior-regime-reference-plan-v1"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-regime-reference-plan-v1"
            or not self.case_id
            or self.case_id.strip() != self.case_id
            or not self.labeler_id
            or self.labeler_id.strip() != self.labeler_id
        ):
            raise ValueError("regime-reference plan identity is invalid")
        if (
            len(self.labeler_public_key_hex) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.labeler_public_key_hex
            )
        ):
            raise ValueError("regime-reference labeler key is invalid")
        try:
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.labeler_public_key_hex)
            )
        except ValueError as error:
            raise ValueError("regime-reference labeler key is invalid") from error
        for name in ("source_contract_digest", "adjudication_policy_digest"):
            _require_digest(name, getattr(self, name))
        object.__setattr__(
            self,
            "labeling_valid_time",
            _canonical_time(self.labeling_valid_time),
        )
        object.__setattr__(self, "plan_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "plan_digest"
        }


@dataclass(frozen=True, init=False)
class RegimeReferenceEvidence:
    """Signed post-event weather label bound to one completed holdout case."""

    reference_plan_digest: str
    full_analysis_input_digest: str
    verification_bundle_digest: str
    observed_regime: str
    observed_storm_id: str
    labeler_id: str
    labeled_at: str
    labeler_signature: str
    contract: str = "neural-prior-regime-reference-evidence-v1"
    evidence_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use RegimeReferenceEvidence.from_plan")

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "evidence_digest"
        }

    @classmethod
    def from_plan(
        cls,
        plan: RegimeReferencePlan,
        *,
        full_analysis_input_digest: str,
        verification_bundle_digest: str,
        observed_regime: str,
        observed_storm_id: str,
        labeled_at: str,
        labeler_private_key: Ed25519PrivateKey,
    ) -> RegimeReferenceEvidence:
        labeled = _canonical_time(labeled_at)
        if labeled < plan.labeling_valid_time:
            raise ValueError("regime reference was labeled before its valid time")
        for name, value in (
            ("full_analysis_input_digest", full_analysis_input_digest),
            ("verification_bundle_digest", verification_bundle_digest),
        ):
            _require_digest(name, value)
        for name, value in (
            ("observed_regime", observed_regime),
            ("observed_storm_id", observed_storm_id),
        ):
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be canonical")
        values: dict[str, object] = {
            "reference_plan_digest": plan.plan_digest,
            "full_analysis_input_digest": full_analysis_input_digest,
            "verification_bundle_digest": verification_bundle_digest,
            "observed_regime": observed_regime,
            "observed_storm_id": observed_storm_id,
            "labeler_id": plan.labeler_id,
            "labeled_at": labeled,
            "labeler_signature": "",
            "contract": "neural-prior-regime-reference-evidence-v1",
        }
        signature = labeler_private_key.sign(
            json_digest(values).encode("ascii")
        ).hex()
        return _new_regime_reference_evidence(
            **{**values, "labeler_signature": signature}
        )


def _new_regime_reference_evidence(
    **values: object,
) -> RegimeReferenceEvidence:
    result = object.__new__(RegimeReferenceEvidence)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, "evidence_digest", json_digest(result.payload))
    return result


def validate_regime_reference_evidence(
    evidence: RegimeReferenceEvidence,
    plan: RegimeReferencePlan,
) -> None:
    if (
        evidence.contract != "neural-prior-regime-reference-evidence-v1"
        or evidence.reference_plan_digest != plan.plan_digest
        or evidence.labeler_id != plan.labeler_id
        or evidence.evidence_digest != json_digest(evidence.payload)
        or _canonical_time(evidence.labeled_at) < plan.labeling_valid_time
        or len(evidence.labeler_signature) != 128
    ):
        raise ValueError("regime-reference evidence is invalid")
    unsigned = dict(evidence.payload)
    unsigned["labeler_signature"] = ""
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(plan.labeler_public_key_hex)
        ).verify(
            bytes.fromhex(evidence.labeler_signature),
            json_digest(unsigned).encode("ascii"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("regime-reference signature mismatch") from error


def regime_reference_public_key_hex(
    private_key: Ed25519PrivateKey,
) -> str:
    """Return the canonical raw public key committed by a reference plan."""

    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()


@dataclass(frozen=True)
class PhysicalEventCatalogPlan:
    """Outcome-blind commitment for grouping all holdout cases into events."""

    holdout_case_ids: tuple[str, ...]
    association_algorithm_digest: str
    spatial_membership_rule_digest: str
    adjudication_policy_digest: str
    adjudicator_id: str
    adjudicator_public_key_hex: str
    catalog_completion_deadline: str
    spatial_reference_digest: str
    motion_association_rule_digest: str
    scheduler_id: str
    scheduler_public_key_hex: str
    scheduler_trust_store_digest: str
    maximum_association_time_gap_minutes: float = 30.0
    minimum_association_spatial_iou: float = 0.1
    maximum_association_centroid_speed_mps: float = 40.0
    association_motion_buffer_m: float = 5_000.0
    contract: str = "physical-event-catalog-plan-v3"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "physical-event-catalog-plan-v3":
            raise ValueError("unsupported physical event-catalog plan")
        if (
            not self.holdout_case_ids
            or len(set(self.holdout_case_ids)) != len(self.holdout_case_ids)
            or any(not value or value.strip() != value for value in self.holdout_case_ids)
        ):
            raise ValueError("event-catalog plan cases must be nonempty and unique")
        for name in (
            "association_algorithm_digest",
            "spatial_membership_rule_digest",
            "adjudication_policy_digest",
            "spatial_reference_digest",
            "motion_association_rule_digest",
            "scheduler_trust_store_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            not math.isfinite(self.maximum_association_time_gap_minutes)
            or self.maximum_association_time_gap_minutes < 0.0
            or not math.isfinite(self.minimum_association_spatial_iou)
            or not 0.0 <= self.minimum_association_spatial_iou <= 1.0
            or not math.isfinite(self.maximum_association_centroid_speed_mps)
            or self.maximum_association_centroid_speed_mps <= 0.0
            or not math.isfinite(self.association_motion_buffer_m)
            or self.association_motion_buffer_m < 0.0
        ):
            raise ValueError("event association thresholds are invalid")
        if not self.adjudicator_id or self.adjudicator_id.strip() != self.adjudicator_id:
            raise ValueError("event-catalog adjudicator ID must be canonical")
        if len(self.adjudicator_public_key_hex) != 64:
            raise ValueError("event-catalog adjudicator public key is invalid")
        try:
            bytes.fromhex(self.adjudicator_public_key_hex)
        except ValueError as error:
            raise ValueError("event-catalog adjudicator public key is invalid") from error
        if (
            not self.scheduler_id
            or self.scheduler_id.strip() != self.scheduler_id
            or len(self.scheduler_public_key_hex) != 64
            or self.scheduler_id == self.adjudicator_id
            or self.scheduler_public_key_hex == self.adjudicator_public_key_hex
        ):
            raise ValueError("event-catalog scheduler authority is invalid")
        try:
            bytes.fromhex(self.scheduler_public_key_hex)
        except ValueError as error:
            raise ValueError("event-catalog scheduler authority is invalid") from error
        object.__setattr__(
            self,
            "catalog_completion_deadline",
            _canonical_time(self.catalog_completion_deadline),
        )
        object.__setattr__(self, "plan_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "plan_digest"
        }


@dataclass(frozen=True, init=False)
class TrustedProcessStartReceipt:
    """Scheduler-signed proof that a catalog existed before a trusted job start."""

    catalog_plan_digest: str
    catalog_result_digest: str
    process_kind: Literal["candidate_training", "candidate_scoring"]
    subject_digests: tuple[str, ...]
    process_algorithm_digest: str
    process_runtime_digest: str
    execution_contract_digest: str
    job_id: str
    launch_nonce: str
    scheduler_sequence_number: int
    previous_receipt_digest: str | None
    started_at: str
    scheduler_id: str
    scheduler_public_key_hex: str
    scheduler_trust_store_digest: str
    scheduler_signature: str
    contract: str = "trusted-process-start-receipt-v2"
    receipt_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use TrustedProcessStartReceipt.from_plan")

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: (list(value) if key == "subject_digests" else value)
            for key, value in self.__dict__.items()
            if key != "receipt_digest"
        }

    @classmethod
    def from_plan(
        cls,
        plan: PhysicalEventCatalogPlan,
        *,
        catalog_result_digest: str,
        process_kind: Literal["candidate_training", "candidate_scoring"],
        subject_digests: tuple[str, ...],
        process_algorithm_digest: str,
        process_runtime_digest: str,
        execution_contract_digest: str,
        job_id: str,
        launch_nonce: str,
        scheduler_sequence_number: int,
        previous_receipt_digest: str | None,
        started_at: str,
        scheduler_private_key: Ed25519PrivateKey,
    ) -> TrustedProcessStartReceipt:
        values: dict[str, object] = {
            "catalog_plan_digest": plan.plan_digest,
            "catalog_result_digest": catalog_result_digest,
            "process_kind": process_kind,
            "subject_digests": subject_digests,
            "process_algorithm_digest": process_algorithm_digest,
            "process_runtime_digest": process_runtime_digest,
            "execution_contract_digest": execution_contract_digest,
            "job_id": job_id,
            "launch_nonce": launch_nonce,
            "scheduler_sequence_number": scheduler_sequence_number,
            "previous_receipt_digest": previous_receipt_digest,
            "started_at": _canonical_time(started_at),
            "scheduler_id": plan.scheduler_id,
            "scheduler_public_key_hex": regime_reference_public_key_hex(
                scheduler_private_key
            ),
            "scheduler_trust_store_digest": plan.scheduler_trust_store_digest,
            "scheduler_signature": "",
            "contract": "trusted-process-start-receipt-v2",
        }
        unsigned = _trusted_process_start_payload(values)
        values["scheduler_signature"] = scheduler_private_key.sign(
            json_digest(unsigned).encode("ascii")
        ).hex()
        receipt = _new_trusted_process_start_receipt(**values)
        validate_trusted_process_start_receipt(receipt, plan)
        return receipt


def _trusted_process_start_payload(values: dict[str, object]) -> dict[str, object]:
    subjects = values.get("subject_digests")
    if not isinstance(subjects, tuple) or not all(
        isinstance(value, str) for value in subjects
    ):
        raise ValueError("trusted process subjects are invalid")
    payload = {
        key: value for key, value in values.items() if key != "receipt_digest"
    }
    payload["subject_digests"] = list(subjects)
    return payload


def _new_trusted_process_start_receipt(
    **values: object,
) -> TrustedProcessStartReceipt:
    receipt = object.__new__(TrustedProcessStartReceipt)
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    object.__setattr__(receipt, "receipt_digest", json_digest(receipt.payload))
    return receipt


def validate_trusted_process_start_receipt(
    receipt: TrustedProcessStartReceipt,
    plan: PhysicalEventCatalogPlan,
    *,
    catalog_result: PhysicalEventCatalogResult | None = None,
) -> None:
    _validate_trusted_process_start_receipt_integrity(receipt)
    if (
        receipt.catalog_plan_digest != plan.plan_digest
        or receipt.scheduler_id != plan.scheduler_id
        or receipt.scheduler_public_key_hex != plan.scheduler_public_key_hex
        or receipt.scheduler_trust_store_digest
        != plan.scheduler_trust_store_digest
    ):
        raise ValueError("trusted process-start receipt authority is invalid")
    if catalog_result is not None and (
        receipt.catalog_result_digest != catalog_result.result_digest
        or _canonical_time(catalog_result.cataloged_at)
        >= _canonical_time(receipt.started_at)
    ):
        raise ValueError("trusted process start does not follow its event catalog")


def _validate_trusted_process_start_receipt_integrity(
    receipt: TrustedProcessStartReceipt,
) -> None:
    try:
        if (
            receipt.contract != "trusted-process-start-receipt-v2"
            or receipt.process_kind not in ("candidate_training", "candidate_scoring")
            or not receipt.subject_digests
            or len(set(receipt.subject_digests)) != len(receipt.subject_digests)
            or not receipt.job_id
            or receipt.job_id.strip() != receipt.job_id
            or type(receipt.scheduler_sequence_number) is not int
            or receipt.scheduler_sequence_number <= 0
            or receipt.receipt_digest != json_digest(receipt.payload)
            or len(receipt.scheduler_signature) != 128
        ):
            raise ValueError("trusted process-start receipt is invalid")
        _canonical_time(receipt.started_at)
        _require_digest("start receipt catalog result", receipt.catalog_result_digest)
        _require_digest("start receipt algorithm", receipt.process_algorithm_digest)
        _require_digest("start receipt runtime", receipt.process_runtime_digest)
        _require_digest(
            "start receipt execution contract",
            receipt.execution_contract_digest,
        )
        _require_digest("start receipt launch nonce", receipt.launch_nonce)
        _require_digest(
            "start receipt scheduler trust store",
            receipt.scheduler_trust_store_digest,
        )
        if receipt.previous_receipt_digest is not None:
            _require_digest(
                "start receipt predecessor",
                receipt.previous_receipt_digest,
            )
        for digest in receipt.subject_digests:
            _require_digest("start receipt subject", digest)
        unsigned = dict(receipt.payload)
        unsigned["scheduler_signature"] = ""
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(receipt.scheduler_public_key_hex)
        ).verify(
            bytes.fromhex(receipt.scheduler_signature),
            json_digest(unsigned).encode("ascii"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("trusted process-start receipt is invalid") from error


@dataclass(frozen=True, init=False)
class TrustedProcessCompletionReceipt:
    """Scheduler-signed result bound to one ledger-recorded process start."""

    start_receipt_digest: str
    catalog_plan_digest: str
    catalog_result_digest: str
    process_kind: Literal["candidate_training", "candidate_scoring"]
    subject_digests: tuple[str, ...]
    process_algorithm_digest: str
    process_runtime_digest: str
    execution_contract_digest: str
    job_id: str
    launch_nonce: str
    scheduler_sequence_number: int
    started_at: str
    completed_at: str
    output_artifact_digest: str
    process_log_digest: str
    scheduler_id: str
    scheduler_public_key_hex: str
    scheduler_trust_store_digest: str
    scheduler_signature: str
    contract: str = "trusted-process-completion-receipt-v1"
    receipt_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use TrustedProcessCompletionReceipt.from_start")

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: (list(value) if key == "subject_digests" else value)
            for key, value in self.__dict__.items()
            if key != "receipt_digest"
        }

    @classmethod
    def from_start(
        cls,
        start: TrustedProcessStartReceipt,
        *,
        completed_at: str,
        output_artifact_digest: str,
        process_log_digest: str,
        scheduler_private_key: Ed25519PrivateKey,
    ) -> TrustedProcessCompletionReceipt:
        values: dict[str, object] = {
            "start_receipt_digest": start.receipt_digest,
            "catalog_plan_digest": start.catalog_plan_digest,
            "catalog_result_digest": start.catalog_result_digest,
            "process_kind": start.process_kind,
            "subject_digests": start.subject_digests,
            "process_algorithm_digest": start.process_algorithm_digest,
            "process_runtime_digest": start.process_runtime_digest,
            "execution_contract_digest": start.execution_contract_digest,
            "job_id": start.job_id,
            "launch_nonce": start.launch_nonce,
            "scheduler_sequence_number": start.scheduler_sequence_number,
            "started_at": start.started_at,
            "completed_at": _canonical_time(completed_at),
            "output_artifact_digest": output_artifact_digest,
            "process_log_digest": process_log_digest,
            "scheduler_id": start.scheduler_id,
            "scheduler_public_key_hex": regime_reference_public_key_hex(
                scheduler_private_key
            ),
            "scheduler_trust_store_digest": start.scheduler_trust_store_digest,
            "scheduler_signature": "",
            "contract": "trusted-process-completion-receipt-v1",
        }
        unsigned = _trusted_process_completion_payload(values)
        values["scheduler_signature"] = scheduler_private_key.sign(
            json_digest(unsigned).encode("ascii")
        ).hex()
        receipt = _new_trusted_process_completion_receipt(**values)
        validate_trusted_process_completion_receipt(receipt, start)
        return receipt


def _trusted_process_completion_payload(
    values: dict[str, object],
) -> dict[str, object]:
    subjects = values.get("subject_digests")
    if not isinstance(subjects, tuple) or not all(
        isinstance(value, str) for value in subjects
    ):
        raise ValueError("trusted process completion subjects are invalid")
    payload = {
        key: value for key, value in values.items() if key != "receipt_digest"
    }
    payload["subject_digests"] = list(subjects)
    return payload


def _new_trusted_process_completion_receipt(
    **values: object,
) -> TrustedProcessCompletionReceipt:
    receipt = object.__new__(TrustedProcessCompletionReceipt)
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    object.__setattr__(receipt, "receipt_digest", json_digest(receipt.payload))
    return receipt


def validate_trusted_process_completion_receipt(
    receipt: TrustedProcessCompletionReceipt,
    start: TrustedProcessStartReceipt,
) -> None:
    try:
        if (
            receipt.contract != "trusted-process-completion-receipt-v1"
            or receipt.start_receipt_digest != start.receipt_digest
            or receipt.catalog_plan_digest != start.catalog_plan_digest
            or receipt.catalog_result_digest != start.catalog_result_digest
            or receipt.process_kind != start.process_kind
            or receipt.subject_digests != start.subject_digests
            or receipt.process_algorithm_digest != start.process_algorithm_digest
            or receipt.process_runtime_digest != start.process_runtime_digest
            or receipt.execution_contract_digest != start.execution_contract_digest
            or receipt.job_id != start.job_id
            or receipt.launch_nonce != start.launch_nonce
            or receipt.scheduler_sequence_number
            != start.scheduler_sequence_number
            or receipt.started_at != start.started_at
            or receipt.scheduler_id != start.scheduler_id
            or receipt.scheduler_public_key_hex
            != start.scheduler_public_key_hex
            or receipt.scheduler_trust_store_digest
            != start.scheduler_trust_store_digest
            or _canonical_time(receipt.completed_at)
            <= _canonical_time(start.started_at)
            or receipt.receipt_digest != json_digest(receipt.payload)
            or len(receipt.scheduler_signature) != 128
        ):
            raise ValueError("trusted process-completion receipt is invalid")
        for name, value in (
            ("completion output", receipt.output_artifact_digest),
            ("completion log", receipt.process_log_digest),
        ):
            _require_digest(name, value)
        unsigned = dict(receipt.payload)
        unsigned["scheduler_signature"] = ""
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(receipt.scheduler_public_key_hex)
        ).verify(
            bytes.fromhex(receipt.scheduler_signature),
            json_digest(unsigned).encode("ascii"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("trusted process-completion receipt is invalid") from error


@dataclass(frozen=True)
class PhysicalEventTrackArtifact:
    """Replayable, timestamped object track in one declared projected CRS."""

    timestamps: tuple[str, ...]
    centroid_xy_m: tuple[tuple[float, float], ...]
    object_mask_digests: tuple[str, ...]
    source_radar_ids: tuple[str, ...]
    association_edge_digests: tuple[str, ...]
    spatial_reference_digest: str
    contract: str = "physical-event-track-artifact-v1"
    artifact_digest: str = field(init=False)

    def __post_init__(self) -> None:
        timestamps = tuple(_canonical_time(value) for value in self.timestamps)
        parsed = tuple(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in timestamps
        )
        if (
            self.contract != "physical-event-track-artifact-v1"
            or len(timestamps) < 2
            or len(set(timestamps)) != len(timestamps)
            or any(first >= second for first, second in zip(parsed, parsed[1:]))
            or len(self.centroid_xy_m) != len(timestamps)
            or len(self.object_mask_digests) != len(timestamps)
            or len(self.source_radar_ids) != len(timestamps)
            or len(self.association_edge_digests) != len(timestamps) - 1
            or any(
                len(point) != 2
                or any(not math.isfinite(value) for value in point)
                for point in self.centroid_xy_m
            )
            or any(not value or value.strip() != value for value in self.source_radar_ids)
        ):
            raise ValueError("physical event track artifact is invalid")
        _require_digest("track spatial reference", self.spatial_reference_digest)
        for value in self.object_mask_digests:
            _require_digest("track object mask", value)
        for value in self.association_edge_digests:
            _require_digest("track association edge", value)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "artifact_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "timestamps": list(self.timestamps),
            "centroid_xy_m": [list(point) for point in self.centroid_xy_m],
            "object_mask_digests": list(self.object_mask_digests),
            "source_radar_ids": list(self.source_radar_ids),
            "association_edge_digests": list(self.association_edge_digests),
            "spatial_reference_digest": self.spatial_reference_digest,
        }

    @property
    def mean_velocity_xy_mps(self) -> tuple[float, float]:
        start = datetime.fromisoformat(self.timestamps[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.timestamps[-1].replace("Z", "+00:00"))
        seconds = (end - start).total_seconds()
        return (
            (self.centroid_xy_m[-1][0] - self.centroid_xy_m[0][0]) / seconds,
            (self.centroid_xy_m[-1][1] - self.centroid_xy_m[0][1]) / seconds,
        )


def validate_physical_event_track_artifact(
    artifact: PhysicalEventTrackArtifact,
) -> None:
    """Rehash a replayable track instead of trusting its retained digest."""

    try:
        timestamps = tuple(_canonical_time(value) for value in artifact.timestamps)
        parsed = tuple(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in timestamps
        )
        if (
            artifact.contract != "physical-event-track-artifact-v1"
            or artifact.artifact_digest != json_digest(artifact.payload)
            or timestamps != artifact.timestamps
            or len(timestamps) < 2
            or any(first >= second for first, second in zip(parsed, parsed[1:]))
            or len(artifact.centroid_xy_m) != len(timestamps)
            or len(artifact.object_mask_digests) != len(timestamps)
            or len(artifact.source_radar_ids) != len(timestamps)
            or len(artifact.association_edge_digests) != len(timestamps) - 1
            or any(
                len(point) != 2
                or any(not math.isfinite(value) for value in point)
                for point in artifact.centroid_xy_m
            )
            or any(
                not value or value.strip() != value
                for value in artifact.source_radar_ids
            )
        ):
            raise ValueError("physical event track artifact is invalid")
        _require_digest("track spatial reference", artifact.spatial_reference_digest)
        for value in artifact.object_mask_digests:
            _require_digest("track object mask", value)
        for value in artifact.association_edge_digests:
            _require_digest("track association edge", value)
    except (AttributeError, ValueError) as error:
        raise ValueError("physical event track artifact is invalid") from error


@dataclass(frozen=True, init=False)
class PhysicalEventCatalogEvidence:
    """Signed, immutable mapping from holdout cases to one physical event."""

    event_id: str
    member_case_ids: tuple[str, ...]
    member_full_analysis_input_digests: tuple[str, ...]
    start_time: str
    end_time: str
    spatial_envelope_xy_m: tuple[float, float, float, float]
    start_centroid_xy_m: tuple[float, float]
    end_centroid_xy_m: tuple[float, float]
    mean_velocity_xy_mps: tuple[float, float]
    object_track_artifact: PhysicalEventTrackArtifact
    object_track_artifact_digest: str
    participating_radar_ids: tuple[str, ...]
    association_algorithm_digest: str
    adjudication_policy_digest: str
    adjudicator_id: str
    adjudicator_public_key_hex: str
    adjudicator_signature: str
    physical_event_identity_digest: str
    contract: str = "physical-event-catalog-evidence-v4"
    event_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use PhysicalEventCatalogEvidence.from_members")

    @property
    def payload(self) -> dict[str, object]:
        result = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("event_digest", "object_track_artifact")
        }
        result["object_track_artifact"] = (
            self.object_track_artifact.payload
            | {"artifact_digest": self.object_track_artifact.artifact_digest}
        )
        return result

    @classmethod
    def from_members(
        cls,
        *,
        event_id: str,
        member_case_ids: tuple[str, ...],
        member_full_analysis_input_digests: tuple[str, ...],
        start_time: str,
        end_time: str,
        spatial_envelope_xy_m: tuple[float, float, float, float],
        object_track_artifact: PhysicalEventTrackArtifact,
        participating_radar_ids: tuple[str, ...],
        association_algorithm_digest: str,
        adjudication_policy_digest: str,
        adjudicator_id: str,
        adjudicator_private_key: Ed25519PrivateKey,
    ) -> PhysicalEventCatalogEvidence:
        values: dict[str, object] = {
            "event_id": event_id,
            "member_case_ids": member_case_ids,
            "member_full_analysis_input_digests": (
                member_full_analysis_input_digests
            ),
            "start_time": _canonical_time(start_time),
            "end_time": _canonical_time(end_time),
            "spatial_envelope_xy_m": spatial_envelope_xy_m,
            "start_centroid_xy_m": object_track_artifact.centroid_xy_m[0],
            "end_centroid_xy_m": object_track_artifact.centroid_xy_m[-1],
            "mean_velocity_xy_mps": object_track_artifact.mean_velocity_xy_mps,
            "object_track_artifact": object_track_artifact,
            "object_track_artifact_digest": object_track_artifact.artifact_digest,
            "participating_radar_ids": participating_radar_ids,
            "association_algorithm_digest": association_algorithm_digest,
            "adjudication_policy_digest": adjudication_policy_digest,
            "adjudicator_id": adjudicator_id,
            "adjudicator_public_key_hex": regime_reference_public_key_hex(
                adjudicator_private_key
            ),
            "adjudicator_signature": "",
            "physical_event_identity_digest": "",
            "contract": "physical-event-catalog-evidence-v4",
        }
        values["physical_event_identity_digest"] = (
            _physical_event_identity_digest(values)
        )
        unsigned = {
            key: value
            for key, value in values.items()
            if key != "object_track_artifact"
        }
        unsigned["object_track_artifact"] = (
            object_track_artifact.payload
            | {"artifact_digest": object_track_artifact.artifact_digest}
        )
        signature = adjudicator_private_key.sign(
            json_digest(unsigned).encode("ascii")
        ).hex()
        return _new_physical_event_catalog_evidence(
            **{**values, "adjudicator_signature": signature}
        )


def _new_physical_event_catalog_evidence(
    **values: object,
) -> PhysicalEventCatalogEvidence:
    result = object.__new__(PhysicalEventCatalogEvidence)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, "event_digest", json_digest(result.payload))
    validate_physical_event_catalog(result)
    return result


def _physical_event_identity_digest(values: dict[str, object]) -> str:
    return json_digest(
        {
            "contract": "physical-event-identity-v2",
            "start_time": values["start_time"],
            "end_time": values["end_time"],
            "spatial_envelope_xy_m": values["spatial_envelope_xy_m"],
            "start_centroid_xy_m": values["start_centroid_xy_m"],
            "end_centroid_xy_m": values["end_centroid_xy_m"],
            "mean_velocity_xy_mps": values["mean_velocity_xy_mps"],
            "object_track_artifact_digest": values[
                "object_track_artifact_digest"
            ],
            "participating_radar_ids": values["participating_radar_ids"],
            "association_algorithm_digest": values[
                "association_algorithm_digest"
            ],
            "adjudication_policy_digest": values["adjudication_policy_digest"],
        }
    )


def validate_physical_event_catalog(
    evidence: PhysicalEventCatalogEvidence,
) -> None:
    try:
        validate_physical_event_track_artifact(evidence.object_track_artifact)
        start = _canonical_time(evidence.start_time)
        end = _canonical_time(evidence.end_time)
        minimum_x, minimum_y, maximum_x, maximum_y = (
            evidence.spatial_envelope_xy_m
        )
        motion_values = (
            *evidence.start_centroid_xy_m,
            *evidence.end_centroid_xy_m,
            *evidence.mean_velocity_xy_mps,
        )
        if (
            evidence.contract != "physical-event-catalog-evidence-v4"
            or not evidence.event_id
            or evidence.event_id.strip() != evidence.event_id
            or not evidence.adjudicator_id
            or evidence.adjudicator_id.strip() != evidence.adjudicator_id
            or not evidence.member_case_ids
            or len(set(evidence.member_case_ids)) != len(evidence.member_case_ids)
            or any(not value or value.strip() != value for value in evidence.member_case_ids)
            or len(evidence.member_case_ids)
            != len(evidence.member_full_analysis_input_digests)
            or not evidence.participating_radar_ids
            or len(set(evidence.participating_radar_ids))
            != len(evidence.participating_radar_ids)
            or any(
                not value or value.strip() != value
                for value in evidence.participating_radar_ids
            )
            or start >= end
            or any(
                not math.isfinite(value)
                for value in evidence.spatial_envelope_xy_m
            )
            or minimum_x >= maximum_x
            or minimum_y >= maximum_y
            or len(evidence.start_centroid_xy_m) != 2
            or len(evidence.end_centroid_xy_m) != 2
            or len(evidence.mean_velocity_xy_mps) != 2
            or any(not math.isfinite(value) for value in motion_values)
            or evidence.start_time != evidence.object_track_artifact.timestamps[0]
            or evidence.end_time != evidence.object_track_artifact.timestamps[-1]
            or evidence.start_centroid_xy_m
            != evidence.object_track_artifact.centroid_xy_m[0]
            or evidence.end_centroid_xy_m
            != evidence.object_track_artifact.centroid_xy_m[-1]
            or any(
                not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
                for actual, expected in zip(
                    evidence.mean_velocity_xy_mps,
                    evidence.object_track_artifact.mean_velocity_xy_mps,
                    strict=True,
                )
            )
            or evidence.object_track_artifact_digest
            != evidence.object_track_artifact.artifact_digest
            or set(evidence.participating_radar_ids)
            != set(evidence.object_track_artifact.source_radar_ids)
            or not (
                minimum_x
                <= evidence.start_centroid_xy_m[0]
                <= maximum_x
                and minimum_y
                <= evidence.start_centroid_xy_m[1]
                <= maximum_y
                and minimum_x
                <= evidence.end_centroid_xy_m[0]
                <= maximum_x
                and minimum_y
                <= evidence.end_centroid_xy_m[1]
                <= maximum_y
            )
            or any(
                not (
                    minimum_x <= point[0] <= maximum_x
                    and minimum_y <= point[1] <= maximum_y
                )
                for point in evidence.object_track_artifact.centroid_xy_m
            )
            or len(evidence.adjudicator_public_key_hex) != 64
            or len(evidence.adjudicator_signature) != 128
            or evidence.physical_event_identity_digest
            != _physical_event_identity_digest(evidence.payload)
            or evidence.event_digest != json_digest(evidence.payload)
        ):
            raise ValueError("physical event-catalog evidence is invalid")
        for name, value in (
            ("association algorithm", evidence.association_algorithm_digest),
            ("adjudication policy", evidence.adjudication_policy_digest),
            ("object track artifact", evidence.object_track_artifact_digest),
        ):
            _require_digest(name, value)
        for value in evidence.member_full_analysis_input_digests:
            _require_digest("event member input", value)
        unsigned = dict(evidence.payload)
        unsigned["adjudicator_signature"] = ""
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(evidence.adjudicator_public_key_hex)
        ).verify(
            bytes.fromhex(evidence.adjudicator_signature),
            json_digest(unsigned).encode("ascii"),
        )
    except (AttributeError, InvalidSignature, ValueError) as error:
        raise ValueError("physical event-catalog evidence is invalid") from error


@dataclass(frozen=True)
class PhysicalEventCaseSpatialEvidence:
    """Outcome-blind proof that one case belongs inside one physical event."""

    case_id: str
    full_analysis_input_digest: str
    physical_event_identity_digest: str
    observed_spatial_envelope_xy_m: tuple[float, float, float, float]
    event_spatial_envelope_xy_m: tuple[float, float, float, float]
    spatial_membership_rule_digest: str
    source_object_evidence_digest: str
    input_available_time: str
    spatial_reference_digest: str
    contract: str = "physical-event-case-spatial-evidence-v2"
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("case input", self.full_analysis_input_digest),
            ("physical event identity", self.physical_event_identity_digest),
            ("spatial membership rule", self.spatial_membership_rule_digest),
            ("source object evidence", self.source_object_evidence_digest),
            ("spatial reference", self.spatial_reference_digest),
        ):
            _require_digest(name, value)
        observed = self.observed_spatial_envelope_xy_m
        event = self.event_spatial_envelope_xy_m
        if (
            self.contract != "physical-event-case-spatial-evidence-v2"
            or not self.case_id
            or self.case_id.strip() != self.case_id
            or any(not math.isfinite(value) for value in (*observed, *event))
            or observed[0] >= observed[2]
            or observed[1] >= observed[3]
            or event[0] >= event[2]
            or event[1] >= event[3]
            or observed[0] < event[0]
            or observed[1] < event[1]
            or observed[2] > event[2]
            or observed[3] > event[3]
        ):
            raise ValueError(
                "physical event case spatial envelope is invalid or outside its event"
            )
        object.__setattr__(
            self,
            "input_available_time",
            _canonical_time(self.input_available_time),
        )
        object.__setattr__(self, "evidence_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "full_analysis_input_digest": self.full_analysis_input_digest,
            "physical_event_identity_digest": (
                self.physical_event_identity_digest
            ),
            "observed_spatial_envelope_xy_m": list(
                self.observed_spatial_envelope_xy_m
            ),
            "event_spatial_envelope_xy_m": list(
                self.event_spatial_envelope_xy_m
            ),
            "spatial_membership_rule_digest": (
                self.spatial_membership_rule_digest
            ),
            "source_object_evidence_digest": self.source_object_evidence_digest,
            "input_available_time": self.input_available_time,
            "spatial_reference_digest": self.spatial_reference_digest,
            "contract": self.contract,
        }


@dataclass(frozen=True, init=False)
class PhysicalEventCatalogResult:
    """Signed, candidate-neutral result covering one entire holdout plan."""

    catalog_plan_digest: str
    event_evidences: tuple[PhysicalEventCatalogEvidence, ...]
    case_spatial_membership_evidences: tuple[
        PhysicalEventCaseSpatialEvidence, ...
    ]
    cataloged_at: str
    adjudicator_id: str
    adjudicator_public_key_hex: str
    adjudicator_signature: str
    association_graph_digest: str
    contract: str = "physical-event-catalog-result-v3"
    result_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use PhysicalEventCatalogResult.from_plan")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "catalog_plan_digest": self.catalog_plan_digest,
            "event_evidences": [
                item.payload | {"event_digest": item.event_digest}
                for item in self.event_evidences
            ],
            "case_spatial_membership_evidences": [
                item.payload | {"evidence_digest": item.evidence_digest}
                for item in self.case_spatial_membership_evidences
            ],
            "cataloged_at": self.cataloged_at,
            "adjudicator_id": self.adjudicator_id,
            "adjudicator_public_key_hex": self.adjudicator_public_key_hex,
            "adjudicator_signature": self.adjudicator_signature,
            "association_graph_digest": self.association_graph_digest,
            "contract": self.contract,
        }

    @classmethod
    def from_plan(
        cls,
        plan: PhysicalEventCatalogPlan,
        *,
        event_evidences: tuple[PhysicalEventCatalogEvidence, ...],
        case_spatial_membership_evidences: tuple[
            PhysicalEventCaseSpatialEvidence, ...
        ],
        cataloged_at: str,
        adjudicator_private_key: Ed25519PrivateKey,
    ) -> PhysicalEventCatalogResult:
        values: dict[str, object] = {
            "catalog_plan_digest": plan.plan_digest,
            "event_evidences": event_evidences,
            "case_spatial_membership_evidences": case_spatial_membership_evidences,
            "cataloged_at": _canonical_time(cataloged_at),
            "adjudicator_id": plan.adjudicator_id,
            "adjudicator_public_key_hex": regime_reference_public_key_hex(
                adjudicator_private_key
            ),
            "adjudicator_signature": "",
            "association_graph_digest": "",
            "contract": "physical-event-catalog-result-v3",
        }
        values["association_graph_digest"] = _event_association_graph_digest(
            plan.plan_digest,
            event_evidences,
        )
        unsigned = _physical_event_catalog_result_payload(values)
        signature = adjudicator_private_key.sign(
            json_digest(unsigned).encode("ascii")
        ).hex()
        result = _new_physical_event_catalog_result(
            **{**values, "adjudicator_signature": signature}
        )
        validate_physical_event_catalog_result(result, plan)
        return result


def _physical_event_catalog_result_payload(
    values: dict[str, object],
) -> dict[str, object]:
    events = cast(tuple[PhysicalEventCatalogEvidence, ...], values["event_evidences"])
    memberships = cast(
        tuple[PhysicalEventCaseSpatialEvidence, ...],
        values["case_spatial_membership_evidences"],
    )
    return {
        "catalog_plan_digest": values["catalog_plan_digest"],
        "event_evidences": [
            item.payload | {"event_digest": item.event_digest} for item in events
        ],
        "case_spatial_membership_evidences": [
            item.payload | {"evidence_digest": item.evidence_digest}
            for item in memberships
        ],
        "cataloged_at": values["cataloged_at"],
        "adjudicator_id": values["adjudicator_id"],
        "adjudicator_public_key_hex": values["adjudicator_public_key_hex"],
        "adjudicator_signature": values["adjudicator_signature"],
        "association_graph_digest": values["association_graph_digest"],
        "contract": values["contract"],
    }


def _new_physical_event_catalog_result(
    **values: object,
) -> PhysicalEventCatalogResult:
    result = object.__new__(PhysicalEventCatalogResult)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, "result_digest", json_digest(result.payload))
    unsigned = dict(result.payload)
    unsigned["adjudicator_signature"] = ""
    try:
        if (
            result.contract != "physical-event-catalog-result-v3"
            or len(result.adjudicator_public_key_hex) != 64
            or len(result.adjudicator_signature) != 128
        ):
            raise ValueError("physical event-catalog result is invalid")
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(result.adjudicator_public_key_hex)
        ).verify(
            bytes.fromhex(result.adjudicator_signature),
            json_digest(unsigned).encode("ascii"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("physical event-catalog result signature mismatch") from error
    return result


def _event_association_graph_digest(
    plan_digest: str,
    events: tuple[PhysicalEventCatalogEvidence, ...],
) -> str:
    """Address the candidate-neutral event components used for inference."""

    return json_digest(
        {
            "contract": "physical-event-association-graph-v1",
            "catalog_plan_digest": plan_digest,
            "components": [
                {
                    "physical_event_identity_digest": (
                        event.physical_event_identity_digest
                    ),
                    "member_case_ids": sorted(event.member_case_ids),
                    "member_full_analysis_input_digests": sorted(
                        event.member_full_analysis_input_digests
                    ),
                }
                for event in sorted(
                    events,
                    key=lambda item: item.physical_event_identity_digest,
                )
            ],
        }
    )


def _event_spatial_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_x = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_y = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_x * intersection_y
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _track_centroid_at(
    track: PhysicalEventTrackArtifact,
    when: datetime,
) -> tuple[float, float]:
    times = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in track.timestamps
    )
    if when <= times[0]:
        return track.centroid_xy_m[0]
    if when >= times[-1]:
        return track.centroid_xy_m[-1]
    for index, (start, end) in enumerate(zip(times, times[1:])):
        if start <= when <= end:
            fraction = (when - start).total_seconds() / (end - start).total_seconds()
            first = track.centroid_xy_m[index]
            second = track.centroid_xy_m[index + 1]
            return (
                first[0] + fraction * (second[0] - first[0]),
                first[1] + fraction * (second[1] - first[1]),
            )
    raise ValueError("event track does not bracket the requested time")


def _overlap_track_distance(
    first: PhysicalEventTrackArtifact,
    second: PhysicalEventTrackArtifact,
) -> float | None:
    first_times = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in first.timestamps
    )
    second_times = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in second.timestamps
    )
    overlap_start = max(first_times[0], second_times[0])
    overlap_end = min(first_times[-1], second_times[-1])
    if overlap_start > overlap_end:
        return None
    sample_times = tuple(
        sorted(
            {
                overlap_start,
                overlap_end,
                *(value for value in first_times if overlap_start <= value <= overlap_end),
                *(value for value in second_times if overlap_start <= value <= overlap_end),
            }
        )
    )
    return min(
        math.dist(
            _track_centroid_at(first, value),
            _track_centroid_at(second, value),
        )
        for value in sample_times
    )


def _events_associate(
    first: PhysicalEventCatalogEvidence,
    second: PhysicalEventCatalogEvidence,
    plan: PhysicalEventCatalogPlan,
) -> bool:
    first_start = datetime.fromisoformat(first.start_time.replace("Z", "+00:00"))
    first_end = datetime.fromisoformat(first.end_time.replace("Z", "+00:00"))
    second_start = datetime.fromisoformat(second.start_time.replace("Z", "+00:00"))
    second_end = datetime.fromisoformat(second.end_time.replace("Z", "+00:00"))
    gap_seconds = max(
        0.0,
        (second_start - first_end).total_seconds(),
        (first_start - second_end).total_seconds(),
    )
    overlap_distance = _overlap_track_distance(
        first.object_track_artifact,
        second.object_track_artifact,
    )
    if first_end <= second_start:
        earlier = first
        later = second
    elif second_end <= first_start:
        earlier = second
        later = first
    else:
        earlier = first
        later = second
    predicted_x = (
        earlier.end_centroid_xy_m[0]
        + earlier.mean_velocity_xy_mps[0] * gap_seconds
    )
    predicted_y = (
        earlier.end_centroid_xy_m[1]
        + earlier.mean_velocity_xy_mps[1] * gap_seconds
    )
    centroid_distance = math.hypot(
        later.start_centroid_xy_m[0] - predicted_x,
        later.start_centroid_xy_m[1] - predicted_y,
    )
    maximum_motion_distance = plan.association_motion_buffer_m
    return (
        gap_seconds <= plan.maximum_association_time_gap_minutes * 60.0
        and (
            _event_spatial_iou(
                first.spatial_envelope_xy_m,
                second.spatial_envelope_xy_m,
            )
            >= plan.minimum_association_spatial_iou
            or (
                overlap_distance is not None
                and overlap_distance <= plan.association_motion_buffer_m
            )
            or centroid_distance <= maximum_motion_distance
        )
    )


def validate_physical_event_catalog_result(
    result: PhysicalEventCatalogResult,
    plan: PhysicalEventCatalogPlan,
    *,
    candidate_scoring_started_at: str | None = None,
) -> None:
    try:
        cataloged_at = _canonical_time(result.cataloged_at)
        if (
            result.contract != "physical-event-catalog-result-v3"
            or result.catalog_plan_digest != plan.plan_digest
            or result.adjudicator_id != plan.adjudicator_id
            or result.adjudicator_public_key_hex != plan.adjudicator_public_key_hex
            or result.result_digest != json_digest(result.payload)
            or result.association_graph_digest
            != _event_association_graph_digest(plan.plan_digest, result.event_evidences)
            or cataloged_at > plan.catalog_completion_deadline
            or len(result.adjudicator_signature) != 128
        ):
            raise ValueError("physical event-catalog result is invalid")
        events = result.event_evidences
        if not events or len({item.event_digest for item in events}) != len(events):
            raise ValueError("physical event-catalog result is incomplete")
        if any(
            _events_associate(first, second, plan)
            for index, first in enumerate(events)
            for second in events[index + 1 :]
        ):
            raise ValueError(
                "physical event association graph has split connected components"
            )
        member_cases: list[str] = []
        for event in events:
            validate_physical_event_catalog(event)
            if (
                event.association_algorithm_digest
                != plan.association_algorithm_digest
                or event.adjudication_policy_digest
                != plan.adjudication_policy_digest
                or event.object_track_artifact.spatial_reference_digest
                != plan.spatial_reference_digest
                or event.adjudicator_id != plan.adjudicator_id
                or event.adjudicator_public_key_hex != plan.adjudicator_public_key_hex
                or math.hypot(*event.mean_velocity_xy_mps)
                > plan.maximum_association_centroid_speed_mps
            ):
                raise ValueError("physical event-catalog result disagrees with its plan")
            member_cases.extend(event.member_case_ids)
        if (
            len(member_cases) != len(set(member_cases))
            or set(member_cases) != set(plan.holdout_case_ids)
        ):
            raise ValueError("physical event-catalog result membership is incomplete")
        event_by_case = {
            case_id: event
            for event in events
            for case_id in event.member_case_ids
        }
        input_by_case = {
            case_id: input_digest
            for event in events
            for case_id, input_digest in zip(
                event.member_case_ids,
                event.member_full_analysis_input_digests,
                strict=True,
            )
        }
        spatial = result.case_spatial_membership_evidences
        if (
            len(spatial) != len(plan.holdout_case_ids)
            or len({item.case_id for item in spatial}) != len(spatial)
            or {item.case_id for item in spatial} != set(plan.holdout_case_ids)
        ):
            raise ValueError("physical event spatial-membership evidence is incomplete")
        for item in spatial:
            event = event_by_case[item.case_id]
            if (
                item.evidence_digest != json_digest(item.payload)
                or item.full_analysis_input_digest != input_by_case[item.case_id]
                or item.physical_event_identity_digest
                != event.physical_event_identity_digest
                or item.event_spatial_envelope_xy_m
                != event.spatial_envelope_xy_m
                or item.spatial_membership_rule_digest
                != plan.spatial_membership_rule_digest
                or item.spatial_reference_digest != plan.spatial_reference_digest
            ):
                raise ValueError(
                    "physical event spatial-membership evidence disagrees with catalog"
                )
        if max(_canonical_time(item.input_available_time) for item in spatial) > cataloged_at:
            raise ValueError(
                "physical event catalog predates member input availability"
            )
        if candidate_scoring_started_at is not None and (
            cataloged_at >= _canonical_time(candidate_scoring_started_at)
        ):
            raise ValueError(
                "physical event catalog must be fixed before candidate scoring"
            )
        unsigned = dict(result.payload)
        unsigned["adjudicator_signature"] = ""
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(plan.adjudicator_public_key_hex)
        ).verify(
            bytes.fromhex(result.adjudicator_signature),
            json_digest(unsigned).encode("ascii"),
        )
    except InvalidSignature as error:
        raise ValueError("physical event-catalog result signature mismatch") from error


@dataclass(frozen=True)
class NeuralPriorHoldoutPlanCase:
    """One case committed before forecasts and verification are available."""

    case_id: str
    storm_id: str
    day: str
    radar_id: str
    regime: str
    range_regime: str
    input_plan_digest: str
    verification_plan_digest: str
    metric_contract_digest: str
    uncertainty_target_plan_digest: str
    state_calibration_target_plan_digest: str
    range_band_contract_digest: str
    reference_active_range_regimes: tuple[str, ...]
    regime_reference_plan_digest: str
    issue_time: str

    def __post_init__(self) -> None:
        _validate_holdout_case_identity(self)
        for name in (
            "input_plan_digest",
            "verification_plan_digest",
            "metric_contract_digest",
            "uncertainty_target_plan_digest",
            "state_calibration_target_plan_digest",
            "range_band_contract_digest",
            "regime_reference_plan_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            len(set(self.reference_active_range_regimes))
            != len(self.reference_active_range_regimes)
            or any(not value for value in self.reference_active_range_regimes)
        ):
            raise ValueError("reference active range regimes are invalid")
        object.__setattr__(self, "issue_time", _canonical_time(self.issue_time))


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanCase:
    """Read-only v1 case retained for audit, never for promotion."""

    case_id: str
    storm_id: str
    day: str
    radar_id: str
    regime: str
    range_regime: str
    input_bundle_digest: str
    verification_plan_digest: str
    metric_contract_digest: str
    issue_time: str


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanAudit:
    plan_id: str
    parent_prior_digest: str
    candidate_family_digests: tuple[str, ...]
    cases: tuple[LegacyNeuralPriorHoldoutPlanCase, ...]
    registered_at: str
    contract: str = "neural-prior-holdout-plan-v1"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "contract": self.contract,
            "plan_id": self.plan_id,
            "parent_prior_digest": self.parent_prior_digest,
            "candidate_family_digests": list(self.candidate_family_digests),
            "cases": [item.__dict__ for item in self.cases],
            "registered_at": self.registered_at,
        }
        object.__setattr__(self, "plan_digest", json_digest(payload))


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV2Case:
    """Read-only v2 case retained before uncertainty targets were planned."""

    case_id: str
    storm_id: str
    day: str
    radar_id: str
    regime: str
    range_regime: str
    input_plan_digest: str
    verification_plan_digest: str
    metric_contract_digest: str
    issue_time: str


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV3Case:
    """Read-only v3/v4/v5 case before state calibration was planned."""

    case_id: str
    storm_id: str
    day: str
    radar_id: str
    regime: str
    range_regime: str
    input_plan_digest: str
    verification_plan_digest: str
    metric_contract_digest: str
    uncertainty_target_plan_digest: str
    issue_time: str


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV2Audit:
    """Read-only v2 holdout retained for audit, never for promotion."""

    plan_id: str
    parent_prior_digest: str
    candidate_family_digests: tuple[str, ...]
    cases: tuple[LegacyNeuralPriorHoldoutPlanV2Case, ...]
    input_plans: tuple[NeuralPriorInputPlan, ...]
    registered_at: str
    mode: Literal["prospective", "sealed_historical"] = "prospective"
    sealed_historical_dataset_digest: str | None = None
    candidate_training_started_at: str | None = None
    contract: str = "neural-prior-holdout-plan-v2"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "contract": self.contract,
            "plan_id": self.plan_id,
            "parent_prior_digest": self.parent_prior_digest,
            "candidate_family_digests": list(self.candidate_family_digests),
            "cases": [item.__dict__ for item in self.cases],
            "input_plans": [item.payload for item in self.input_plans],
            "registered_at": self.registered_at,
            "mode": self.mode,
            "sealed_historical_dataset_digest": (
                self.sealed_historical_dataset_digest
            ),
            "candidate_training_started_at": self.candidate_training_started_at,
        }
        object.__setattr__(self, "plan_digest", json_digest(payload))


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV3Audit:
    """Read-only v3 holdout retained before target source attestation."""

    plan_id: str
    parent_prior_digest: str
    candidate_family_digests: tuple[str, ...]
    cases: tuple[LegacyNeuralPriorHoldoutPlanV3Case, ...]
    input_plans: tuple[NeuralPriorInputPlan, ...]
    uncertainty_target_plans: tuple[dict[str, object], ...]
    registered_at: str
    mode: Literal["prospective", "sealed_historical"] = "prospective"
    sealed_historical_dataset_digest: str | None = None
    candidate_training_started_at: str | None = None
    contract: str = "neural-prior-holdout-plan-v3"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "contract": self.contract,
            "plan_id": self.plan_id,
            "parent_prior_digest": self.parent_prior_digest,
            "candidate_family_digests": list(self.candidate_family_digests),
            "cases": [item.__dict__ for item in self.cases],
            "input_plans": [item.payload for item in self.input_plans],
            "uncertainty_target_plans": list(self.uncertainty_target_plans),
            "registered_at": self.registered_at,
            "mode": self.mode,
            "sealed_historical_dataset_digest": (
                self.sealed_historical_dataset_digest
            ),
            "candidate_training_started_at": self.candidate_training_started_at,
        }
        object.__setattr__(self, "plan_digest", json_digest(payload))


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV4Audit:
    """Read-only v4 holdout retained before probability semantics."""

    plan_id: str
    parent_prior_digest: str
    candidate_family_digests: tuple[str, ...]
    cases: tuple[LegacyNeuralPriorHoldoutPlanV3Case, ...]
    input_plans: tuple[NeuralPriorInputPlan, ...]
    uncertainty_target_plans: tuple[dict[str, object], ...]
    registered_at: str
    mode: Literal["prospective", "sealed_historical"] = "prospective"
    sealed_historical_dataset_digest: str | None = None
    candidate_training_started_at: str | None = None
    contract: str = "neural-prior-holdout-plan-v4"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "contract": self.contract,
            "plan_id": self.plan_id,
            "parent_prior_digest": self.parent_prior_digest,
            "candidate_family_digests": list(self.candidate_family_digests),
            "cases": [item.__dict__ for item in self.cases],
            "input_plans": [item.payload for item in self.input_plans],
            "uncertainty_target_plans": list(self.uncertainty_target_plans),
            "registered_at": self.registered_at,
            "mode": self.mode,
            "sealed_historical_dataset_digest": (
                self.sealed_historical_dataset_digest
            ),
            "candidate_training_started_at": self.candidate_training_started_at,
        }
        object.__setattr__(self, "plan_digest", json_digest(payload))


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV5Audit:
    """Raw v5 plan retained before state-calibration targets were required."""

    plan_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-holdout-plan-audit-v5"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy holdout plan digest", self.plan_digest)
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("invalid legacy holdout plan payload") from error
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-holdout-plan-v5"
            or payload.get("plan_digest") != self.plan_digest
        ):
            raise ValueError("invalid legacy v5 holdout plan")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if canonical != self.payload_json:
            raise ValueError("legacy v5 holdout plan is not canonical")
        original = dict(payload)
        original.pop("plan_digest")
        for collection in ("input_plans", "uncertainty_target_plans"):
            entries = original.get(collection)
            if not isinstance(entries, list):
                raise ValueError("legacy v5 holdout payload is incomplete")
            original[collection] = [
                {key: value for key, value in entry.items() if key != "plan_digest"}
                for entry in entries
            ]
        if json_digest(original) != self.plan_digest:
            raise ValueError("legacy v5 holdout plan digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "plan_digest": self.plan_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV6Audit:
    """Raw v6 plan retained before target measurement attestation."""

    plan_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-holdout-plan-audit-v6"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy holdout plan digest", self.plan_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-holdout-plan-v6"
            or payload.get("plan_digest") != self.plan_digest
        ):
            raise ValueError("invalid legacy v6 holdout plan")
        if json.dumps(payload, sort_keys=True, separators=(",", ":")) != (
            self.payload_json
        ):
            raise ValueError("legacy v6 holdout plan is not canonical")
        original = dict(payload)
        original.pop("plan_digest")
        for collection in (
            "input_plans",
            "uncertainty_target_plans",
            "state_calibration_target_plans",
        ):
            entries = original.get(collection)
            if not isinstance(entries, list):
                raise ValueError("legacy v6 holdout payload is incomplete")
            original[collection] = [
                {key: value for key, value in entry.items() if key != "plan_digest"}
                for entry in entries
            ]
        if json_digest(original) != self.plan_digest:
            raise ValueError("legacy v6 holdout plan digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "plan_digest": self.plan_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV7Audit:
    """Raw v7 plan retained before range/classifier preregistration."""

    plan_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-holdout-plan-audit-v7"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy holdout plan digest", self.plan_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-holdout-plan-v7"
            or payload.get("plan_digest") != self.plan_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v7 holdout plan")
        original = dict(payload)
        original.pop("plan_digest")
        for collection in (
            "input_plans",
            "uncertainty_target_plans",
            "state_calibration_target_plans",
        ):
            entries = original.get(collection)
            if not isinstance(entries, list):
                raise ValueError("legacy v7 holdout payload is incomplete")
            original[collection] = [
                {key: value for key, value in entry.items() if key != "plan_digest"}
                for entry in entries
            ]
        if json_digest(original) != self.plan_digest:
            raise ValueError("legacy v7 holdout plan digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "plan_digest": self.plan_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV8Audit:
    """Raw v8 plan retained before post-event regime references."""

    plan_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-holdout-plan-audit-v8"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy holdout plan digest", self.plan_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-holdout-plan-v8"
            or payload.get("plan_digest") != self.plan_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v8 holdout plan")
        original = dict(payload)
        original.pop("plan_digest")
        for collection, computed_fields in (
            ("input_plans", {"plan_digest"}),
            (
                "uncertainty_target_plans",
                {"plan_digest", "support_event_digest"},
            ),
            ("state_calibration_target_plans", {"plan_digest"}),
            ("range_band_contracts", {"contract_digest"}),
            ("regime_classifier_manifests", {"manifest_digest"}),
        ):
            entries = original.get(collection)
            if not isinstance(entries, list):
                raise ValueError("legacy v8 holdout payload is incomplete")
            original[collection] = [
                {
                    key: value
                    for key, value in entry.items()
                    if key not in computed_fields
                }
                for entry in entries
            ]
        if json_digest(original) != self.plan_digest:
            raise ValueError("legacy v8 holdout plan digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "plan_digest": self.plan_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV9Audit:
    """Raw v9 plan retained before physical range geometry binding."""

    plan_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-holdout-plan-audit-v9"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy holdout plan digest", self.plan_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-holdout-plan-v9"
            or payload.get("plan_digest") != self.plan_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v9 holdout plan")
        original = dict(payload)
        original.pop("plan_digest")
        if json_digest(original) != self.plan_digest:
            raise ValueError("legacy v9 holdout plan digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "plan_digest": self.plan_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV10Audit:
    """Raw v10 plan retained before candidate-neutral event catalogs."""

    plan_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-holdout-plan-audit-v10"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy holdout plan digest", self.plan_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-holdout-plan-v10"
            or payload.get("plan_digest") != self.plan_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v10 holdout plan")
        original = dict(payload)
        original.pop("plan_digest")
        if json_digest(original) != self.plan_digest:
            raise ValueError("legacy v10 holdout plan digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "plan_digest": self.plan_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV11Audit:
    """Raw v11 plan retained before trusted chronology and event graphs."""

    plan_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-holdout-plan-audit-v11"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy holdout plan digest", self.plan_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-holdout-plan-v11"
            or payload.get("plan_digest") != self.plan_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v11 holdout plan")
        original = dict(payload)
        original.pop("plan_digest")
        if json_digest(original) != self.plan_digest:
            raise ValueError("legacy v11 holdout plan digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "plan_digest": self.plan_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorHoldoutPlanV12Audit:
    """Raw v12 plan retained before trusted execution completion."""

    plan_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-holdout-plan-audit-v12"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy holdout plan digest", self.plan_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-holdout-plan-v12"
            or payload.get("plan_digest") != self.plan_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v12 holdout plan")
        original = dict(payload)
        original.pop("plan_digest")
        if json_digest(original) != self.plan_digest:
            raise ValueError("legacy v12 holdout plan digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "plan_digest": self.plan_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class NeuralPriorHoldoutPlan:
    """Root-approved holdout commitment created before any evaluated issue."""

    plan_id: str
    parent_prior_digest: str
    candidate_family_digests: tuple[str, ...]
    cases: tuple[NeuralPriorHoldoutPlanCase, ...]
    input_plans: tuple[NeuralPriorInputPlan, ...]
    uncertainty_target_plans: tuple[PriorUncertaintyTargetPlan, ...]
    state_calibration_target_plans: tuple[
        NeuralPriorStateCalibrationPlan, ...
    ]
    range_band_contracts: tuple[RangeBandContract, ...]
    range_geometry_contracts: tuple[RangeGeometryContract, ...]
    regime_reference_plans: tuple[RegimeReferencePlan, ...]
    regime_classifier_manifests: tuple[RegimeClassifierManifest, ...]
    reference_label_contract_digest: str
    physical_event_catalog_plan: PhysicalEventCatalogPlan
    scoring_algorithm_digest: str
    scoring_runtime_digest: str
    metric_engine_digest: str
    verification_resolver_digest: str
    registered_at: str
    mode: Literal["prospective", "sealed_historical"] = "prospective"
    sealed_historical_dataset_digest: str | None = None
    candidate_training_started_at: str | None = None
    contract: str = "neural-prior-holdout-plan-v13"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-holdout-plan-v13":
            raise ValueError("unsupported neural-prior holdout plan")
        if not self.plan_id or self.plan_id.strip() != self.plan_id:
            raise ValueError("holdout plan ID must be canonical")
        _require_digest("parent_prior_digest", self.parent_prior_digest)
        if not self.candidate_family_digests or len(
            set(self.candidate_family_digests)
        ) != len(self.candidate_family_digests):
            raise ValueError("candidate family must be nonempty and unique")
        for digest in self.candidate_family_digests:
            _require_digest("candidate family digest", digest)
        if self.parent_prior_digest in self.candidate_family_digests:
            raise ValueError("candidate family cannot contain the parent prior")
        case_ids = tuple(item.case_id for item in self.cases)
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("holdout plan cases must be nonempty and unique")
        input_plans = tuple(item.input_plan_digest for item in self.cases)
        if len(set(input_plans)) != len(input_plans):
            raise ValueError("holdout cases must use distinct input plans")
        retained_plans = {item.plan_digest: item for item in self.input_plans}
        if set(retained_plans) != set(input_plans):
            raise ValueError("holdout input-plan payloads are incomplete")
        target_plans = tuple(
            item.uncertainty_target_plan_digest for item in self.cases
        )
        retained_targets = {
            item.plan_digest: item for item in self.uncertainty_target_plans
        }
        if set(retained_targets) != set(target_plans):
            raise ValueError("holdout uncertainty-target plans are incomplete")
        state_target_plans = tuple(
            item.state_calibration_target_plan_digest for item in self.cases
        )
        retained_state_targets = {
            item.plan_digest: item for item in self.state_calibration_target_plans
        }
        if set(retained_state_targets) != set(state_target_plans):
            raise ValueError("holdout state-calibration plans are incomplete")
        retained_range_contracts = {
            item.contract_digest: item for item in self.range_band_contracts
        }
        expected_range_contracts = {
            item.range_band_contract_digest for item in self.cases
        }
        if (
            len(retained_range_contracts) != len(self.range_band_contracts)
            or set(retained_range_contracts) != expected_range_contracts
        ):
            raise ValueError("holdout range-band contracts are incomplete")
        for case in self.cases:
            range_contract = retained_range_contracts[
                case.range_band_contract_digest
            ]
            if (
                range_contract.case_id != case.case_id
                or range_contract.reference_active_range_regimes
                != case.reference_active_range_regimes
                or range_contract.grid_contract_digest
                != next(
                    item.grid_contract_digest
                    for item in self.input_plans
                    if item.plan_digest == case.input_plan_digest
                )
            ):
                raise ValueError("holdout range-band contract disagrees with its case")
        retained_geometries = {
            item.contract_digest: item for item in self.range_geometry_contracts
        }
        expected_geometries = {
            item.range_geometry_contract_digest
            for item in self.range_band_contracts
        }
        if (
            not retained_geometries
            or len(retained_geometries) != len(self.range_geometry_contracts)
            or set(retained_geometries) != expected_geometries
            or any(
                retained_geometries[item.range_geometry_contract_digest].grid_contract_digest
                != item.grid_contract_digest
                or retained_geometries[item.range_geometry_contract_digest].range_regime_labels
                != item.range_regime_labels
                for item in self.range_band_contracts
            )
        ):
            raise ValueError("holdout range geometry contracts are incomplete")
        reference_plans = {
            item.plan_digest: item for item in self.regime_reference_plans
        }
        if (
            len(reference_plans) != len(self.regime_reference_plans)
            or set(reference_plans)
            != {item.regime_reference_plan_digest for item in self.cases}
            or any(
                reference_plans[item.regime_reference_plan_digest].case_id
                != item.case_id
                for item in self.cases
            )
        ):
            raise ValueError("holdout regime-reference plans are incomplete")
        if any(
            reference_plans[item.regime_reference_plan_digest].labeling_valid_time
            <= item.issue_time
            for item in self.cases
        ):
            raise ValueError("regime reference must be produced after issue")
        _require_digest(
            "reference label contract", self.reference_label_contract_digest
        )
        for name in (
            "scoring_algorithm_digest",
            "scoring_runtime_digest",
            "metric_engine_digest",
            "verification_resolver_digest",
        ):
            _require_digest(name, getattr(self, name))
        if set(self.physical_event_catalog_plan.holdout_case_ids) != set(case_ids):
            raise ValueError("physical event-catalog plan cases are incomplete")
        if any(
            item.labeler_id != self.physical_event_catalog_plan.adjudicator_id
            or item.labeler_public_key_hex
            != self.physical_event_catalog_plan.adjudicator_public_key_hex
            or item.adjudication_policy_digest
            != self.physical_event_catalog_plan.adjudication_policy_digest
            for item in self.regime_reference_plans
        ):
            raise ValueError("physical event-catalog plan authority is untrusted")
        if any(
            item.source_contract_digest != self.reference_label_contract_digest
            for item in self.regime_reference_plans
        ):
            raise ValueError("regime-reference source contract is not approved")
        if (
            not self.regime_classifier_manifests
            or len(
                {item.manifest_digest for item in self.regime_classifier_manifests}
            )
            != len(self.regime_classifier_manifests)
            or len(
                {item.classifier_digest for item in self.regime_classifier_manifests}
            )
            != len(self.regime_classifier_manifests)
            or any(
                item.reference_label_contract_digest
                != self.reference_label_contract_digest
                for item in self.regime_classifier_manifests
            )
        ):
            raise ValueError("holdout classifier family is invalid")
        holdout_case_ids = {item.case_id for item in self.cases}
        holdout_storm_ids = {item.storm_id for item in self.cases}
        holdout_days = {item.day for item in self.cases}
        holdout_issues = tuple(item.issue_time for item in self.cases)
        for classifier in self.regime_classifier_manifests:
            if (
                set(classifier.training_case_ids) & holdout_case_ids
                or set(classifier.training_storm_ids) & holdout_storm_ids
                or set(classifier.training_days) & holdout_days
                or any(
                    start <= issue <= end
                    for start, end in classifier.training_time_windows
                    for issue in holdout_issues
                )
            ):
                raise ValueError("classifier training overlaps the holdout")
        registered = _canonical_time(self.registered_at)
        if self.mode == "prospective":
            if self.sealed_historical_dataset_digest is not None or (
                self.candidate_training_started_at is not None
            ):
                raise ValueError("prospective holdout cannot use a sealed dataset")
            if any(registered >= item.issue_time for item in self.cases):
                raise ValueError("holdout plan must precede every issue time")
            if any(
                item.regime != "pending" or item.storm_id != "pending"
                for item in self.cases
            ):
                raise ValueError(
                    "prospective holdout cannot preregister weather truth or storm identity"
                )
        elif self.mode == "sealed_historical":
            if self.sealed_historical_dataset_digest is None or (
                self.candidate_training_started_at is None
            ):
                raise ValueError("historical holdout requires a sealed dataset")
            _require_digest(
                "sealed_historical_dataset_digest",
                self.sealed_historical_dataset_digest,
            )
            started = _canonical_time(self.candidate_training_started_at)
            if registered >= started:
                raise ValueError("historical holdout must be sealed before training")
            object.__setattr__(self, "candidate_training_started_at", started)
        else:
            raise ValueError("unsupported holdout plan mode")
        object.__setattr__(self, "registered_at", registered)
        object.__setattr__(self, "plan_digest", json_digest(_holdout_plan_payload(self)))

    def case(self, case_id: str) -> NeuralPriorHoldoutPlanCase:
        matches = tuple(item for item in self.cases if item.case_id == case_id)
        if len(matches) != 1:
            raise ValueError("case is not in the holdout plan")
        return matches[0]

    @property
    def holdout_dataset_digest(self) -> str:
        return _holdout_dataset_digest(self.cases)

    @property
    def scoring_execution_contract_digest(self) -> str:
        return json_digest(
            {
                "contract": "neural-prior-scoring-execution-contract-v1",
                "holdout_plan_digest": self.plan_digest,
                "parent_prior_digest": self.parent_prior_digest,
                "candidate_family_digests": sorted(
                    self.candidate_family_digests
                ),
                "metric_contract_digests": sorted(
                    {item.metric_contract_digest for item in self.cases}
                ),
                "verification_plan_digests": sorted(
                    {item.verification_plan_digest for item in self.cases}
                ),
                "scoring_algorithm_digest": self.scoring_algorithm_digest,
                "scoring_runtime_digest": self.scoring_runtime_digest,
                "metric_engine_digest": self.metric_engine_digest,
                "verification_resolver_digest": (
                    self.verification_resolver_digest
                ),
            }
        )


@dataclass(frozen=True)
class NeuralPriorHoldoutPlanPolicy:
    """Root-approved limits for pre-registering holdout plans."""

    approved_plan_digests: tuple[str, ...]
    approved_metric_contract_digests: tuple[str, ...]
    maximum_candidate_family_size: int
    contract: str = "neural-prior-holdout-plan-policy-v1"

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-holdout-plan-policy-v1":
            raise ValueError("unsupported holdout plan policy")
        if not self.approved_plan_digests or not self.approved_metric_contract_digests:
            raise ValueError("holdout policy approvals must be nonempty")
        for digest in self.approved_plan_digests + self.approved_metric_contract_digests:
            _require_digest("holdout policy digest", digest)
        if (
            type(self.maximum_candidate_family_size) is not int
            or self.maximum_candidate_family_size <= 0
        ):
            raise ValueError("candidate family limit must be positive")

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "approved_plan_digests": sorted(self.approved_plan_digests),
                "approved_metric_contract_digests": sorted(
                    self.approved_metric_contract_digests
                ),
                "maximum_candidate_family_size": (
                    self.maximum_candidate_family_size
                ),
            }
        )


@dataclass(frozen=True)
class NeuralPriorHoldoutCase:
    """One completed paired forecast committed by a holdout plan."""

    case_id: str
    planned_storm_id: str
    storm_id: str
    physical_event_digest: str
    day: str
    radar_id: str
    planned_regime: str
    regime: str
    range_regime: str
    reference_active_range_regimes: tuple[str, ...]
    range_band_contract_digest: str
    regime_reference_plan_digest: str
    regime_reference_evidence_digest: str
    input_plan_digest: str
    input_plan_resolution_digest: str
    input_bundle_digest: str
    full_analysis_input_digest: str
    fixed_input_context_digest: str
    observation_quality_weight_digest: str
    observation_std_dbz_digest: str
    verification_plan_digest: str
    verification_bundle_digest: str
    metric_contract_digest: str
    uncertainty_target_plan_digest: str
    uncertainty_target_digest: str
    prior_probability_contract_digest: str
    state_calibration_target_plan_digest: str
    state_calibration_target_digest: str
    prior_state_contract_digest: str
    issue_time: str
    candidate_forecast_digest: str
    parent_forecast_digest: str
    candidate_prior_application_digest: str
    parent_prior_application_digest: str
    candidate_inference_evidence_digest: str
    parent_inference_evidence_digest: str

    def __post_init__(self) -> None:
        _validate_holdout_case_identity(self)
        for name in (
            "input_plan_digest",
            "input_plan_resolution_digest",
            "input_bundle_digest",
            "full_analysis_input_digest",
            "fixed_input_context_digest",
            "observation_quality_weight_digest",
            "observation_std_dbz_digest",
            "verification_plan_digest",
            "verification_bundle_digest",
            "metric_contract_digest",
            "uncertainty_target_plan_digest",
            "uncertainty_target_digest",
            "prior_probability_contract_digest",
            "state_calibration_target_plan_digest",
            "state_calibration_target_digest",
            "prior_state_contract_digest",
            "range_band_contract_digest",
            "regime_reference_plan_digest",
            "regime_reference_evidence_digest",
            "physical_event_digest",
            "candidate_forecast_digest",
            "parent_forecast_digest",
            "candidate_prior_application_digest",
            "parent_prior_application_digest",
            "candidate_inference_evidence_digest",
            "parent_inference_evidence_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            len(set(self.reference_active_range_regimes))
            != len(self.reference_active_range_regimes)
            or any(not value for value in self.reference_active_range_regimes)
        ):
            raise ValueError("completed reference range set is invalid")
        if (
            not self.planned_regime
            or self.planned_regime.strip() != self.planned_regime
            or not self.planned_storm_id
            or self.planned_storm_id.strip() != self.planned_storm_id
        ):
            raise ValueError("completed planned event identity is invalid")
        object.__setattr__(self, "issue_time", _canonical_time(self.issue_time))
        if self.candidate_forecast_digest == self.parent_forecast_digest:
            raise ValueError("candidate and parent holdout forecasts must differ")
        if self.candidate_prior_application_digest == (
            self.parent_prior_application_digest
        ):
            raise ValueError("candidate and parent prior applications must differ")

    def plan_case(self) -> NeuralPriorHoldoutPlanCase:
        return NeuralPriorHoldoutPlanCase(
            case_id=self.case_id,
            storm_id=self.planned_storm_id,
            day=self.day,
            radar_id=self.radar_id,
            regime=self.planned_regime,
            range_regime=self.range_regime,
            reference_active_range_regimes=self.reference_active_range_regimes,
            range_band_contract_digest=self.range_band_contract_digest,
            regime_reference_plan_digest=self.regime_reference_plan_digest,
            input_plan_digest=self.input_plan_digest,
            verification_plan_digest=self.verification_plan_digest,
            metric_contract_digest=self.metric_contract_digest,
            uncertainty_target_plan_digest=self.uncertainty_target_plan_digest,
            state_calibration_target_plan_digest=(
                self.state_calibration_target_plan_digest
            ),
            issue_time=self.issue_time,
        )


def _validate_classifier_holdout_independence(
    classifier: RegimeClassifierManifest,
    cases: tuple[NeuralPriorHoldoutCase, ...],
) -> None:
    """Reject renamed or re-manifested holdout inputs used in training."""

    holdout_bundle_digests = {item.input_bundle_digest for item in cases}
    holdout_full_digests = {item.full_analysis_input_digest for item in cases}
    if (
        set(classifier.training_input_bundle_digests) & holdout_bundle_digests
        or set(classifier.training_full_analysis_input_digests)
        & holdout_full_digests
    ):
        raise ValueError("classifier training inputs overlap the holdout")
    if set(classifier.training_physical_event_digests) & {
        item.physical_event_digest for item in cases
    }:
        raise ValueError("classifier training physical events overlap the holdout")


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV2:
    """Digest-verified v2 manifest retained only for historical audit."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v2"
    contains_full_input_identity: bool = field(init=False)
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("invalid legacy candidate manifest payload") from error
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v2"
            or payload.get("manifest_digest") != self.manifest_digest
        ):
            raise ValueError("invalid legacy candidate manifest")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if canonical != self.payload_json:
            raise ValueError("legacy candidate manifest is not canonical")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy candidate manifest digest mismatch")
        cases = payload.get("holdout_cases")
        if not isinstance(cases, list) or any(
            not isinstance(item, dict) for item in cases
        ):
            raise ValueError("legacy candidate manifest cases are invalid")
        required = {
            "full_analysis_input_digest",
            "fixed_input_context_digest",
            "observation_quality_weight_digest",
            "observation_std_dbz_digest",
        }
        object.__setattr__(
            self,
            "contains_full_input_identity",
            bool(cases) and all(required <= set(item) for item in cases),
        )
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV3:
    """Digest-verified v3 manifest retained before probability semantics."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v3"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("invalid legacy candidate manifest payload") from error
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v3"
            or payload.get("manifest_digest") != self.manifest_digest
        ):
            raise ValueError("invalid legacy candidate manifest")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if canonical != self.payload_json:
            raise ValueError("legacy candidate manifest is not canonical")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV4:
    """Digest-verified v4 manifest retained before state calibration."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v4"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("invalid legacy candidate manifest payload") from error
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v4"
            or payload.get("manifest_digest") != self.manifest_digest
        ):
            raise ValueError("invalid legacy candidate manifest")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if canonical != self.payload_json:
            raise ValueError("legacy candidate manifest is not canonical")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV5:
    """Digest-verified v5 manifest retained before range-band lineage."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v5"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("invalid legacy candidate manifest payload") from error
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v5"
            or payload.get("manifest_digest") != self.manifest_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v5 candidate manifest")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy v5 candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV6:
    """Digest-verified v6 manifest retained before regime-label evidence."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v6"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v6"
            or payload.get("manifest_digest") != self.manifest_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v6 candidate manifest")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy v6 candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV7:
    """Digest-verified v7 manifest retained before physical event catalogs."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v7"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v7"
            or payload.get("manifest_digest") != self.manifest_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v7 candidate manifest")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy v7 candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV8:
    """Digest-verified v8 manifest retained before neutral catalog results."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v8"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v8"
            or payload.get("manifest_digest") != self.manifest_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v8 candidate manifest")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy v8 candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV9:
    """Digest-verified v9 manifest retained before trusted start receipts."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v9"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v9"
            or payload.get("manifest_digest") != self.manifest_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v9 candidate manifest")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy v9 candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV10:
    """Digest-verified v10 manifest retained before completion receipts."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v10"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v10"
            or payload.get("manifest_digest") != self.manifest_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v10 candidate manifest")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy v10 candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorCandidateManifestAuditV11:
    """Digest-verified v11 manifest retained before scoring-cycle removal."""

    manifest_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-candidate-manifest-audit-v11"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("legacy candidate manifest digest", self.manifest_digest)
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "neural-prior-candidate-manifest-v11"
            or payload.get("manifest_digest") != self.manifest_digest
            or json.dumps(payload, sort_keys=True, separators=(",", ":"))
            != self.payload_json
        ):
            raise ValueError("invalid legacy v11 candidate manifest")
        original = dict(payload)
        original.pop("manifest_digest")
        if json_digest(original) != self.manifest_digest:
            raise ValueError("legacy v11 candidate manifest digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "manifest_digest": self.manifest_digest,
                    "payload": payload,
                }
            ),
        )


def _candidate_training_execution_contract_digest(
    *,
    training_dataset_digest: str,
    candidate_training_manifest_digest: str,
    model_contract_digest: str,
    feature_schema_digest: str,
    algorithm_bundle_digest: str,
    numerical_runtime_digest: str,
) -> str:
    """Address the exact contract authorized for one training process."""

    return json_digest(
        {
            "contract": "neural-prior-training-execution-contract-v1",
            "training_dataset_digest": training_dataset_digest,
            "candidate_training_manifest_digest": candidate_training_manifest_digest,
            "model_contract_digest": model_contract_digest,
            "feature_schema_digest": feature_schema_digest,
            "algorithm_bundle_digest": algorithm_bundle_digest,
            "numerical_runtime_digest": numerical_runtime_digest,
        }
    )


def _validate_holdout_case_identity(value: object) -> None:
    for name in ("case_id", "storm_id", "day", "radar_id", "regime", "range_regime"):
        item = getattr(value, name)
        if not isinstance(item, str) or not item or item.strip() != item:
            raise ValueError(f"{name} must be nonempty and canonical")
    try:
        parsed = date.fromisoformat(getattr(value, "day"))
    except ValueError as error:
        raise ValueError("holdout day must be ISO-8601") from error
    if parsed.isoformat() != getattr(value, "day"):
        raise ValueError("holdout day must be a canonical date")


def _canonical_time(value: str) -> str:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (AttributeError, ValueError) as error:
        raise ValueError("holdout time must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("holdout time must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _holdout_plan_payload(plan: NeuralPriorHoldoutPlan) -> dict[str, object]:
    return {
        "contract": plan.contract,
        "plan_id": plan.plan_id,
        "parent_prior_digest": plan.parent_prior_digest,
        "candidate_family_digests": list(plan.candidate_family_digests),
        "cases": [item.__dict__ for item in plan.cases],
        "input_plans": [item.payload for item in plan.input_plans],
        "uncertainty_target_plans": [
            item.payload for item in plan.uncertainty_target_plans
        ],
        "state_calibration_target_plans": [
            item.payload for item in plan.state_calibration_target_plans
        ],
        "range_band_contracts": [
            item.payload for item in plan.range_band_contracts
        ],
        "range_geometry_contracts": [
            item.payload for item in plan.range_geometry_contracts
        ],
        "regime_reference_plans": [
            item.payload | {"plan_digest": item.plan_digest}
            for item in plan.regime_reference_plans
        ],
        "regime_classifier_manifests": [
            item.payload for item in plan.regime_classifier_manifests
        ],
        "reference_label_contract_digest": (
            plan.reference_label_contract_digest
        ),
        "physical_event_catalog_plan": plan.physical_event_catalog_plan.payload,
        "scoring_algorithm_digest": plan.scoring_algorithm_digest,
        "scoring_runtime_digest": plan.scoring_runtime_digest,
        "metric_engine_digest": plan.metric_engine_digest,
        "verification_resolver_digest": plan.verification_resolver_digest,
        "registered_at": plan.registered_at,
        "mode": plan.mode,
        "sealed_historical_dataset_digest": (plan.sealed_historical_dataset_digest),
        "candidate_training_started_at": plan.candidate_training_started_at,
    }


def _holdout_dataset_digest(
    cases: tuple[NeuralPriorHoldoutPlanCase, ...],
) -> str:
    return json_digest(
        {
            "contract": "neural-prior-holdout-dataset-v4",
            "cases": [
                {
                    "case_id": item.case_id,
                    "input_plan_digest": item.input_plan_digest,
                    "verification_plan_digest": item.verification_plan_digest,
                    "metric_contract_digest": item.metric_contract_digest,
                    "uncertainty_target_plan_digest": (
                        item.uncertainty_target_plan_digest
                    ),
                    "state_calibration_target_plan_digest": (
                        item.state_calibration_target_plan_digest
                    ),
                    "range_band_contract_digest": item.range_band_contract_digest,
                    "regime_reference_plan_digest": (
                        item.regime_reference_plan_digest
                    ),
                    "reference_active_range_regimes": list(
                        item.reference_active_range_regimes
                    ),
                    "issue_time": item.issue_time,
                }
                for item in cases
            ],
        }
    )


def verification_plan_digest(
    *,
    valid_times: tuple[str, ...],
    grid_contract_digest: str,
    radar_product_digest: str,
    qc_pipeline_digest: str,
) -> str:
    """Address verification identity without depending on future frames."""

    for name, value in (
        ("grid_contract_digest", grid_contract_digest),
        ("radar_product_digest", radar_product_digest),
        ("qc_pipeline_digest", qc_pipeline_digest),
    ):
        _require_digest(name, value)
    canonical_times = tuple(_canonical_time(value) for value in valid_times)
    if not canonical_times or len(set(canonical_times)) != len(canonical_times):
        raise ValueError("verification plan times must be nonempty and unique")
    return json_digest(
        {
            "contract": "neural-prior-verification-plan-v1",
            "valid_times": list(canonical_times),
            "grid_contract_digest": grid_contract_digest,
            "radar_product_digest": radar_product_digest,
            "qc_pipeline_digest": qc_pipeline_digest,
        }
    )


def input_plan_digest(
    *,
    valid_times: tuple[str, ...],
    grid_contract_digest: str,
    radar_product_digest: str,
    qc_pipeline_digest: str,
    background_cycle_rule_digest: str,
    mask_policy_digest: str,
    observation_valid_time: str,
    input_available_time: str,
    decision_deadline: str,
    publication_time: str,
) -> str:
    """Identify future input selection rules without claiming future content."""

    return NeuralPriorInputPlan(
        valid_times=valid_times,
        grid_contract_digest=grid_contract_digest,
        radar_product_digest=radar_product_digest,
        qc_pipeline_digest=qc_pipeline_digest,
        background_cycle_rule_digest=background_cycle_rule_digest,
        mask_policy_digest=mask_policy_digest,
        observation_valid_time=observation_valid_time,
        input_available_time=input_available_time,
        decision_deadline=decision_deadline,
        publication_time=publication_time,
    ).plan_digest


@dataclass(frozen=True)
class NeuralPriorCandidateManifest:
    """Immutable pre-scoring lineage for exactly one prior candidate.

    Scoring completion deliberately lives outside this manifest.  Evaluations
    are produced from this precommitted lineage, collected into one canonical
    :class:`HoldoutScoringArtifact`, and only then sealed by a completion
    receipt.  Keeping the completion out of this object removes the former
    manifest/evaluation digest cycle.
    """

    candidate_prior_digest: str
    parent_prior_digest: str
    training_learning_approval_digests: tuple[str, ...]
    training_intervention_digests: tuple[str, ...]
    training_dataset_digest: str
    candidate_training_manifest_digest: str
    parent_training_manifest_digest: str
    model_contract_digest: str
    feature_schema_digest: str
    algorithm_bundle_digest: str
    numerical_runtime_digest: str
    holdout_dataset_digest: str
    holdout_plan_digest: str
    training_case_ids: tuple[str, ...]
    training_input_bundle_digests: tuple[str, ...]
    training_full_analysis_input_digests: tuple[str, ...]
    training_physical_event_digests: tuple[str, ...]
    training_physical_event_catalog_plan: PhysicalEventCatalogPlan
    training_physical_event_catalog_result: PhysicalEventCatalogResult
    candidate_training_started_at: str
    training_storm_ids: tuple[str, ...]
    training_days: tuple[str, ...]
    training_radars: tuple[str, ...]
    training_regimes: tuple[str, ...]
    training_time_windows: tuple[tuple[str, str], ...]
    regime_reference_evidences: tuple[RegimeReferenceEvidence, ...]
    physical_event_catalog_evidences: tuple[PhysicalEventCatalogEvidence, ...]
    physical_event_catalog_result: PhysicalEventCatalogResult
    candidate_scoring_started_at: str
    holdout_cases: tuple[NeuralPriorHoldoutCase, ...]
    candidate_training_start_receipt: TrustedProcessStartReceipt
    candidate_training_completion_receipt: TrustedProcessCompletionReceipt
    candidate_scoring_start_receipt: TrustedProcessStartReceipt
    contract: str = "neural-prior-candidate-manifest-v12"
    manifest_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-candidate-manifest-v12":
            raise ValueError("unsupported neural-prior candidate manifest")
        for name in (
            "candidate_prior_digest",
            "parent_prior_digest",
            "training_dataset_digest",
            "candidate_training_manifest_digest",
            "parent_training_manifest_digest",
            "model_contract_digest",
            "feature_schema_digest",
            "algorithm_bundle_digest",
            "numerical_runtime_digest",
            "holdout_dataset_digest",
            "holdout_plan_digest",
        ):
            _require_digest(name, getattr(self, name))
        if self.candidate_prior_digest == self.parent_prior_digest:
            raise ValueError("candidate and parent prior digests must differ")
        validate_trusted_process_start_receipt(
            self.candidate_training_start_receipt,
            self.training_physical_event_catalog_plan,
            catalog_result=self.training_physical_event_catalog_result,
        )
        _validate_trusted_process_start_receipt_integrity(
            self.candidate_scoring_start_receipt
        )
        validate_trusted_process_completion_receipt(
            self.candidate_training_completion_receipt,
            self.candidate_training_start_receipt,
        )
        training_execution_contract = _candidate_training_execution_contract_digest(
            training_dataset_digest=self.training_dataset_digest,
            candidate_training_manifest_digest=(
                self.candidate_training_manifest_digest
            ),
            model_contract_digest=self.model_contract_digest,
            feature_schema_digest=self.feature_schema_digest,
            algorithm_bundle_digest=self.algorithm_bundle_digest,
            numerical_runtime_digest=self.numerical_runtime_digest,
        )
        if (
            self.candidate_training_start_receipt.process_kind
            != "candidate_training"
            or self.candidate_scoring_start_receipt.process_kind
            != "candidate_scoring"
            or self.candidate_training_start_receipt.started_at
            != _canonical_time(self.candidate_training_started_at)
            or self.candidate_scoring_start_receipt.started_at
            != _canonical_time(self.candidate_scoring_started_at)
            or self.candidate_scoring_start_receipt.catalog_plan_digest
            != self.physical_event_catalog_result.catalog_plan_digest
            or self.candidate_scoring_start_receipt.catalog_result_digest
            != self.physical_event_catalog_result.result_digest
            or set(self.candidate_training_start_receipt.subject_digests)
            != {
                self.training_dataset_digest,
                self.candidate_training_manifest_digest,
            }
            or self.candidate_training_start_receipt.process_algorithm_digest
            != self.algorithm_bundle_digest
            or self.candidate_training_start_receipt.process_runtime_digest
            != self.numerical_runtime_digest
            or self.candidate_training_start_receipt.execution_contract_digest
            != training_execution_contract
            or self.candidate_training_completion_receipt.output_artifact_digest
            != self.candidate_prior_digest
        ):
            raise ValueError("candidate process-start receipt lineage disagrees")
        for digest in (
            self.training_learning_approval_digests
            + self.training_intervention_digests
            + self.training_input_bundle_digests
            + self.training_full_analysis_input_digests
            + self.training_physical_event_digests
        ):
            _require_digest("training evidence digest", digest)
        if not self.training_learning_approval_digests:
            raise ValueError("candidate manifest requires training approvals")
        if not self.training_intervention_digests:
            raise ValueError("candidate manifest requires realized training actions")
        if not self.training_input_bundle_digests or len(
            set(self.training_input_bundle_digests)
        ) != len(self.training_input_bundle_digests):
            raise ValueError("candidate manifest requires unique training inputs")
        if not self.training_physical_event_digests or len(
            set(self.training_physical_event_digests)
        ) != len(self.training_physical_event_digests):
            raise ValueError("candidate manifest requires unique training events")
        if not self.training_full_analysis_input_digests or len(
            set(self.training_full_analysis_input_digests)
        ) != len(self.training_full_analysis_input_digests):
            raise ValueError("candidate manifest requires unique full training inputs")
        holdout_ids = tuple(item.case_id for item in self.holdout_cases)
        if not holdout_ids or len(set(holdout_ids)) != len(holdout_ids):
            raise ValueError("holdout cases must be nonempty and unique")
        evidence = {
            item.evidence_digest: item for item in self.regime_reference_evidences
        }
        if (
            len(evidence) != len(self.regime_reference_evidences)
            or set(evidence)
            != {item.regime_reference_evidence_digest for item in self.holdout_cases}
            or any(
                evidence[item.regime_reference_evidence_digest].observed_regime
                != item.regime
                or evidence[
                    item.regime_reference_evidence_digest
                ].observed_storm_id
                != item.storm_id
                or evidence[
                    item.regime_reference_evidence_digest
                ].full_analysis_input_digest
                != item.full_analysis_input_digest
                or evidence[
                    item.regime_reference_evidence_digest
                ].verification_bundle_digest
                != item.verification_bundle_digest
                for item in self.holdout_cases
            )
        ):
            raise ValueError("candidate regime-reference evidence is incomplete")
        catalogs = {
            item.physical_event_identity_digest: item
            for item in self.physical_event_catalog_evidences
        }
        if (
            not catalogs
            or len(catalogs) != len(self.physical_event_catalog_evidences)
            or set(catalogs)
            != {item.physical_event_digest for item in self.holdout_cases}
        ):
            raise ValueError("candidate physical event-catalog evidence is incomplete")
        case_membership: dict[str, tuple[str, str]] = {}
        for catalog in self.physical_event_catalog_evidences:
            validate_physical_event_catalog(catalog)
            for case_id, input_digest in zip(
                catalog.member_case_ids,
                catalog.member_full_analysis_input_digests,
                strict=True,
            ):
                if case_id in case_membership:
                    raise ValueError("holdout case belongs to multiple physical events")
                case_membership[case_id] = (
                    catalog.physical_event_identity_digest,
                    input_digest,
                )
        if set(case_membership) != set(holdout_ids) or any(
            case_membership[item.case_id]
            != (item.physical_event_digest, item.full_analysis_input_digest)
            for item in self.holdout_cases
        ):
            raise ValueError("physical event-catalog membership disagrees")
        for item in self.holdout_cases:
            catalog = catalogs[item.physical_event_digest]
            issue_time = _canonical_time(item.issue_time)
            if (
                issue_time < _canonical_time(catalog.start_time)
                or issue_time >= _canonical_time(catalog.end_time)
                or item.radar_id not in catalog.participating_radar_ids
            ):
                raise ValueError(
                    "holdout case is outside physical event envelope"
                )
        if tuple(self.physical_event_catalog_result.event_evidences) != tuple(
            self.physical_event_catalog_evidences
        ):
            raise ValueError("candidate event catalogs disagree with catalog result")
        scoring_started = _canonical_time(self.candidate_scoring_started_at)
        if _canonical_time(self.physical_event_catalog_result.cataloged_at) >= scoring_started:
            raise ValueError("physical event catalog must be fixed before candidate scoring")
        object.__setattr__(self, "candidate_scoring_started_at", scoring_started)
        if len(set(self.training_case_ids)) != len(self.training_case_ids):
            raise ValueError("training case IDs must be unique")
        if any(
            not isinstance(value, str) or not value or value.strip() != value
            for value in self.training_case_ids
        ):
            raise ValueError("training case IDs must be canonical")
        if set(self.training_case_ids) & set(holdout_ids):
            raise ValueError("training and promotion holdout cases must be disjoint")
        if set(self.training_input_bundle_digests) & {
            item.input_bundle_digest for item in self.holdout_cases
        }:
            raise ValueError("training and holdout inputs must be disjoint")
        identities = (
            self.training_storm_ids,
            self.training_days,
            self.training_radars,
            self.training_regimes,
        )
        if any(
            not values
            or len(set(values)) != len(values)
            or any(not value or value.strip() != value for value in values)
            for values in identities
        ):
            raise ValueError("training event identities must be nonempty and unique")
        holdout_storms = {item.storm_id for item in self.holdout_cases}
        holdout_days = {item.day for item in self.holdout_cases}
        if set(self.training_storm_ids) & holdout_storms:
            raise ValueError("training and holdout storms must be disjoint")
        if set(self.training_days) & holdout_days:
            raise ValueError("training and holdout days must be disjoint")
        if set(self.training_physical_event_digests) & {
            item.physical_event_digest for item in self.holdout_cases
        }:
            raise ValueError(
                "training and holdout physical events must be disjoint"
            )
        if set(self.training_physical_event_catalog_plan.holdout_case_ids) != set(
            self.training_case_ids
        ):
            raise ValueError("candidate training event-catalog cases disagree")
        validate_physical_event_catalog_result(
            self.training_physical_event_catalog_result,
            self.training_physical_event_catalog_plan,
            candidate_scoring_started_at=self.candidate_training_started_at,
        )
        training_catalog_events = {
            item.physical_event_identity_digest
            for item in self.training_physical_event_catalog_result.event_evidences
        }
        training_catalog_inputs = {
            digest
            for item in self.training_physical_event_catalog_result.event_evidences
            for digest in item.member_full_analysis_input_digests
        }
        if (
            training_catalog_events != set(self.training_physical_event_digests)
            or training_catalog_inputs
            != set(self.training_full_analysis_input_digests)
        ):
            raise ValueError("candidate training physical-event lineage disagrees")
        training_plan = self.training_physical_event_catalog_plan
        if (
            any(
                event.association_algorithm_digest
                != training_plan.association_algorithm_digest
                for event in self.physical_event_catalog_result.event_evidences
            )
            or any(
                membership.spatial_membership_rule_digest
                != training_plan.spatial_membership_rule_digest
                or membership.spatial_reference_digest
                != training_plan.spatial_reference_digest
                for membership in (
                    self.physical_event_catalog_result
                    .case_spatial_membership_evidences
                )
            )
        ):
            raise ValueError(
                "training and holdout event association algorithm contracts differ"
            )
        if any(
            _events_associate(
                training_event,
                holdout_event,
                training_plan,
            )
            for training_event in (
                self.training_physical_event_catalog_result.event_evidences
            )
            for holdout_event in self.physical_event_catalog_result.event_evidences
        ):
            raise ValueError(
                "candidate training overlaps a holdout association component"
            )
        object.__setattr__(
            self,
            "candidate_training_started_at",
            _canonical_time(self.candidate_training_started_at),
        )
        if not self.training_time_windows:
            raise ValueError("training time windows must be nonempty")
        windows = tuple(
            (_canonical_time(start), _canonical_time(end))
            for start, end in self.training_time_windows
        )
        if any(start >= end for start, end in windows):
            raise ValueError("training time windows must have positive duration")
        holdout_issues = tuple(item.issue_time for item in self.holdout_cases)
        if any(
            start <= issue <= end for start, end in windows for issue in holdout_issues
        ):
            raise ValueError("training windows overlap holdout issues")
        object.__setattr__(self, "training_time_windows", windows)
        planned_cases = tuple(item.plan_case() for item in self.holdout_cases)
        if self.holdout_dataset_digest != _holdout_dataset_digest(planned_cases):
            raise ValueError("holdout dataset digest does not match its cases")
        object.__setattr__(self, "manifest_digest", json_digest(
            _candidate_manifest_payload(self)
        ))

    def holdout_case(self, case_id: str) -> NeuralPriorHoldoutCase:
        matches = tuple(
            item for item in self.holdout_cases if item.case_id == case_id
        )
        if len(matches) != 1:
            raise ValueError("case is not in the candidate holdout manifest")
        return matches[0]


def _candidate_manifest_payload(
    manifest: NeuralPriorCandidateManifest,
) -> dict[str, object]:
    return {
        "contract": manifest.contract,
        "candidate_prior_digest": manifest.candidate_prior_digest,
        "parent_prior_digest": manifest.parent_prior_digest,
        "training_learning_approval_digests": list(
            manifest.training_learning_approval_digests
        ),
        "training_intervention_digests": list(
            manifest.training_intervention_digests
        ),
        "training_input_bundle_digests": list(
            manifest.training_input_bundle_digests
        ),
        "training_physical_event_digests": list(
            manifest.training_physical_event_digests
        ),
        "training_full_analysis_input_digests": list(
            manifest.training_full_analysis_input_digests
        ),
        "training_physical_event_catalog_plan": (
            manifest.training_physical_event_catalog_plan.payload
            | {"plan_digest": manifest.training_physical_event_catalog_plan.plan_digest}
        ),
        "training_physical_event_catalog_result": (
            manifest.training_physical_event_catalog_result.payload
            | {
                "result_digest": (
                    manifest.training_physical_event_catalog_result.result_digest
                )
            }
        ),
        "candidate_training_started_at": manifest.candidate_training_started_at,
        "candidate_training_start_receipt": (
            manifest.candidate_training_start_receipt.payload
            | {
                "receipt_digest": (
                    manifest.candidate_training_start_receipt.receipt_digest
                )
            }
        ),
        "candidate_training_completion_receipt": (
            manifest.candidate_training_completion_receipt.payload
            | {
                "receipt_digest": (
                    manifest.candidate_training_completion_receipt.receipt_digest
                )
            }
        ),
        "training_dataset_digest": manifest.training_dataset_digest,
        "candidate_training_manifest_digest": (
            manifest.candidate_training_manifest_digest
        ),
        "parent_training_manifest_digest": (
            manifest.parent_training_manifest_digest
        ),
        "model_contract_digest": manifest.model_contract_digest,
        "feature_schema_digest": manifest.feature_schema_digest,
        "algorithm_bundle_digest": manifest.algorithm_bundle_digest,
        "numerical_runtime_digest": manifest.numerical_runtime_digest,
        "holdout_dataset_digest": manifest.holdout_dataset_digest,
        "holdout_plan_digest": manifest.holdout_plan_digest,
        "training_case_ids": list(manifest.training_case_ids),
        "training_storm_ids": list(manifest.training_storm_ids),
        "training_days": list(manifest.training_days),
        "training_radars": list(manifest.training_radars),
        "training_regimes": list(manifest.training_regimes),
        "training_time_windows": [
            list(value) for value in manifest.training_time_windows
        ],
        "regime_reference_evidences": [
            item.payload | {"evidence_digest": item.evidence_digest}
            for item in manifest.regime_reference_evidences
        ],
        "physical_event_catalog_evidences": [
            item.payload | {"event_digest": item.event_digest}
            for item in manifest.physical_event_catalog_evidences
        ],
        "physical_event_catalog_result": (
            manifest.physical_event_catalog_result.payload
            | {"result_digest": manifest.physical_event_catalog_result.result_digest}
        ),
        "candidate_scoring_started_at": manifest.candidate_scoring_started_at,
        "candidate_scoring_start_receipt": (
            manifest.candidate_scoring_start_receipt.payload
            | {
                "receipt_digest": (
                    manifest.candidate_scoring_start_receipt.receipt_digest
                )
            }
        ),
        "holdout_cases": [item.__dict__ for item in manifest.holdout_cases],
    }


def validate_neural_prior_holdout_plan(plan: NeuralPriorHoldoutPlan) -> None:
    if plan.plan_digest != json_digest(_holdout_plan_payload(plan)):
        raise ValueError("neural-prior holdout plan digest mismatch")


def validate_neural_prior_candidate_manifest(
    manifest: NeuralPriorCandidateManifest,
) -> None:
    if manifest.manifest_digest != json_digest(_candidate_manifest_payload(manifest)):
        raise ValueError("neural-prior candidate manifest digest mismatch")


def _validate_physical_event_catalogs_against_plan(
    manifest: NeuralPriorCandidateManifest,
    plan: NeuralPriorHoldoutPlan,
) -> None:
    """Bind signed event membership to preregistered adjudicator authority."""

    cases = {item.case_id: item for item in manifest.holdout_cases}
    input_plans = {item.plan_digest: item for item in plan.input_plans}
    reference_plans = {
        item.plan_digest: item for item in plan.regime_reference_plans
    }
    for catalog in manifest.physical_event_catalog_evidences:
        validate_physical_event_catalog(catalog)
        if (
            catalog.association_algorithm_digest
            != plan.physical_event_catalog_plan.association_algorithm_digest
        ):
            raise ValueError(
                "physical event-catalog association algorithm was not preregistered"
            )
        for case_id in catalog.member_case_ids:
            case = cases.get(case_id)
            reference = (
                None
                if case is None
                else reference_plans.get(case.regime_reference_plan_digest)
            )
            if (
                reference is None
                or catalog.adjudicator_id != reference.labeler_id
                or catalog.adjudicator_public_key_hex
                != reference.labeler_public_key_hex
                or catalog.adjudication_policy_digest
                != reference.adjudication_policy_digest
            ):
                raise ValueError("physical event-catalog adjudicator is untrusted")
    validate_physical_event_catalog_result(
        manifest.physical_event_catalog_result,
        plan.physical_event_catalog_plan,
    )
    for membership in (
        manifest.physical_event_catalog_result.case_spatial_membership_evidences
    ):
        case = cases[membership.case_id]
        input_plan = input_plans.get(case.input_plan_digest)
        if (
            input_plan is None
            or membership.input_available_time != input_plan.input_available_time
        ):
            raise ValueError(
                "physical event member input availability disagrees with its plan"
            )
    training_plan = manifest.training_physical_event_catalog_plan
    holdout_plan = plan.physical_event_catalog_plan
    if (
        training_plan.association_algorithm_digest
        != holdout_plan.association_algorithm_digest
        or training_plan.spatial_membership_rule_digest
        != holdout_plan.spatial_membership_rule_digest
        or training_plan.spatial_reference_digest
        != holdout_plan.spatial_reference_digest
        or training_plan.maximum_association_time_gap_minutes
        != holdout_plan.maximum_association_time_gap_minutes
        or training_plan.minimum_association_spatial_iou
        != holdout_plan.minimum_association_spatial_iou
        or training_plan.motion_association_rule_digest
        != holdout_plan.motion_association_rule_digest
        or training_plan.maximum_association_centroid_speed_mps
        != holdout_plan.maximum_association_centroid_speed_mps
        or training_plan.association_motion_buffer_m
        != holdout_plan.association_motion_buffer_m
    ):
        raise ValueError(
            "training and holdout event association contracts differ"
        )
    if any(
        _events_associate(training_event, holdout_event, holdout_plan)
        for training_event in (
            manifest.training_physical_event_catalog_result.event_evidences
        )
        for holdout_event in manifest.physical_event_catalog_result.event_evidences
    ):
        raise ValueError(
            "candidate training overlaps a holdout association component"
        )
    validate_trusted_process_start_receipt(
        manifest.candidate_scoring_start_receipt,
        plan.physical_event_catalog_plan,
        catalog_result=manifest.physical_event_catalog_result,
    )
    if (
        set(manifest.candidate_scoring_start_receipt.subject_digests)
        != set(plan.candidate_family_digests)
        or manifest.candidate_scoring_start_receipt.process_algorithm_digest
        != plan.scoring_algorithm_digest
        or manifest.candidate_scoring_start_receipt.process_runtime_digest
        != plan.scoring_runtime_digest
        or manifest.candidate_scoring_start_receipt.execution_contract_digest
        != plan.scoring_execution_contract_digest
    ):
        raise ValueError("candidate scoring receipt family disagrees with holdout plan")


@dataclass(frozen=True, init=False)
class PriorUncertaintyTarget:
    """Independent, withheld target used only for uncertainty calibration."""

    _target_dbz: Tensor
    _valid_mask: Tensor
    _echo_support: Tensor
    target_plan_digest: str
    source_digest: str
    independence_evidence_digest: str
    source_verification_bundle_digest: str
    support_event_digest: str
    target_digest: str

    def __init__(self) -> None:
        raise TypeError("use PriorUncertaintyTarget.from_verification_bundle")

    @classmethod
    def from_verification_bundle(
        cls,
        *,
        plan: PriorUncertaintyTargetPlan,
        verification: VerificationBundle,
    ) -> PriorUncertaintyTarget:
        verification.validate_integrity()
        if (
            plan.contract != "prior-uncertainty-target-plan-v6"
            or verification.contract != "radar-verification-bundle-v3"
            or verification.mask_policy_digest != plan.mask_policy_digest
            or verification.censor_policy_digest != plan.censor_policy_digest
            or verification.floor_representation_contract_digest
            != plan.floor_representation_contract_digest
            or verification.reflectivity_resolution_dbz
            != plan.reflectivity_resolution_dbz
            or verification.quantization_origin_dbz
            != plan.quantization_origin_dbz
            or verification.threshold_bin_convention
            != plan.threshold_bin_convention
            or plan.source_identity_digest != verification.radar_product_digest
            or plan.qc_pipeline_digest != verification.qc_pipeline_digest
            or plan.grid_contract_digest != verification.grid_contract_digest
        ):
            raise ValueError("uncertainty target source disagrees with its plan")
        matches = tuple(
            index
            for index, valid_time in enumerate(verification.valid_times)
            if valid_time == plan.target_valid_time
        )
        if len(matches) != 1:
            raise ValueError("uncertainty target valid time is not in its source")
        index = matches[0]
        target_dbz = verification.frames_dbz[index]
        valid_mask = verification.valid_mask[index]
        echo_support = valid_mask & (target_dbz >= plan.support_threshold_dbz)
        target_plan_digest = plan.plan_digest
        source_digest = verification.content_digest
        independence_evidence_digest = plan.independence_evidence_digest
        source_verification_bundle_digest = verification.content_digest
        if (
            target_dbz.ndim != 2
            or not target_dbz.is_floating_point()
            or valid_mask.shape != target_dbz.shape
            or valid_mask.dtype != torch.bool
            or echo_support.shape != target_dbz.shape
            or echo_support.dtype != torch.bool
            or not bool(torch.any(valid_mask & torch.isfinite(target_dbz)))
        ):
            raise ValueError("prior uncertainty target tensors are invalid")
        target = target_dbz.detach().clone()
        valid = valid_mask.detach().clone()
        support = echo_support.detach().clone()
        target_digest = json_digest(
            {
                "contract": "prior-uncertainty-target-v4",
                "target_dbz": tensor_digest(target),
                "valid_mask": tensor_digest(valid),
                "echo_support": tensor_digest(support),
                "target_plan_digest": target_plan_digest,
                "source_digest": source_digest,
                "independence_evidence_digest": independence_evidence_digest,
                "source_verification_bundle_digest": (
                    source_verification_bundle_digest
                ),
                "support_threshold_dbz": plan.support_threshold_dbz,
                "support_event_digest": plan.support_event_digest,
                "prior_probability_contract_digest": (
                    plan.prior_probability_contract_digest
                ),
            }
        )
        result = object.__new__(cls)
        for name, value in (
            ("_target_dbz", target),
            ("_valid_mask", valid),
            ("_echo_support", support),
            ("target_plan_digest", target_plan_digest),
            ("source_digest", source_digest),
            ("independence_evidence_digest", independence_evidence_digest),
            (
                "source_verification_bundle_digest",
                source_verification_bundle_digest,
            ),
            ("support_event_digest", plan.support_event_digest),
            ("target_digest", target_digest),
        ):
            object.__setattr__(result, name, value)
        return result


@dataclass(frozen=True, init=False)
class NeuralPriorStateCalibrationTarget:
    """Withheld state-product target used to calibrate the P1 state head."""

    _target_dbz: Tensor
    _valid_mask: Tensor
    _echo_support: Tensor
    target_plan_digest: str
    source_verification_bundle_digest: str
    target_digest: str

    def __init__(self) -> None:
        raise TypeError(
            "use NeuralPriorStateCalibrationTarget.from_verification_bundle"
        )

    @classmethod
    def from_verification_bundle(
        cls,
        *,
        plan: NeuralPriorStateCalibrationPlan,
        verification: VerificationBundle,
    ) -> NeuralPriorStateCalibrationTarget:
        verification.validate_integrity()
        if (
            verification.contract != "radar-verification-bundle-v3"
            or verification.mask_policy_digest != plan.mask_policy_digest
            or verification.censor_policy_digest != plan.censor_policy_digest
            or verification.floor_representation_contract_digest
            != plan.floor_representation_contract_digest
            or verification.reflectivity_resolution_dbz
            != plan.reflectivity_resolution_dbz
            or verification.quantization_origin_dbz
            != plan.quantization_origin_dbz
            or verification.threshold_bin_convention
            != plan.threshold_bin_convention
            or plan.source_identity_digest != verification.radar_product_digest
            or plan.qc_pipeline_digest != verification.qc_pipeline_digest
            or plan.grid_contract_digest != verification.grid_contract_digest
        ):
            raise ValueError("state calibration source disagrees with its plan")
        matches = tuple(
            index
            for index, valid_time in enumerate(verification.valid_times)
            if valid_time == plan.target_valid_time
        )
        if len(matches) != 1:
            raise ValueError("state calibration time is not in its source")
        index = matches[0]
        target = verification.frames_dbz[index].detach().clone()
        valid = verification.valid_mask[index].detach().clone()
        support = valid & (target >= plan.support_threshold_dbz)
        if (
            target.ndim != 2
            or not target.is_floating_point()
            or valid.shape != target.shape
            or valid.dtype is not torch.bool
            or not bool(torch.any(valid & torch.isfinite(target)))
        ):
            raise ValueError("state calibration target tensors are invalid")
        target_digest = json_digest(
            {
                "contract": "neural-prior-state-calibration-target-v2",
                "target_dbz": tensor_digest(target),
                "valid_mask": tensor_digest(valid),
                "echo_support": tensor_digest(support),
                "target_plan_digest": plan.plan_digest,
                "source_verification_bundle_digest": verification.content_digest,
            }
        )
        result = object.__new__(cls)
        for name, value in (
            ("_target_dbz", target),
            ("_valid_mask", valid),
            ("_echo_support", support),
            ("target_plan_digest", plan.plan_digest),
            ("source_verification_bundle_digest", verification.content_digest),
            ("target_digest", target_digest),
        ):
            object.__setattr__(result, name, value)
        return result


@dataclass(frozen=True)
class RangeBandEvaluation:
    """Paired skill and uncertainty restricted to one physical range mask."""

    range_regime: str
    range_band_mask_digest: str
    range_geometry_contract_digest: str
    metric_change: Tensor
    end_to_end_metric_change: Tensor
    metric_available: Tensor
    candidate_uncertainty_component_scores: tuple[tuple[str, float], ...]
    parent_uncertainty_component_scores: tuple[tuple[str, float], ...]
    uncertainty_component_differences: tuple[tuple[str, float], ...]
    uncertainty_component_sample_counts: tuple[tuple[str, int], ...]
    evaluated_area_km2: float
    metric_valid_area_km2_by_lead: tuple[float, ...]
    metric_valid_area_km2: Tensor
    issuance_domain_digest: str
    issuance_domain_cell_count_by_lead: tuple[int, ...]
    issuance_domain_area_km2_by_lead: tuple[float, ...]
    parent_issued_count_by_lead: tuple[int, ...]
    candidate_issued_count_by_lead: tuple[int, ...]
    withdrawn_count_by_lead: tuple[int, ...]
    newly_issued_count_by_lead: tuple[int, ...]
    parent_fallback_count_by_lead: tuple[int, ...]
    candidate_fallback_count_by_lead: tuple[int, ...]
    parent_confidence_weighted_issued_area_by_lead: tuple[float, ...]
    candidate_confidence_weighted_issued_area_by_lead: tuple[float, ...]
    withdrawn_fraction_by_lead: Tensor
    newly_issued_fraction_by_lead: Tensor
    background_fallback_increase_by_lead: Tensor
    confidence_weighted_coverage_change_by_lead: Tensor
    probability_valid_area_km2: float
    state_valid_area_km2: float
    echo_pixel_count: int
    clear_pixel_count: int
    echo_object_count: int
    state_echo_pixel_count: int
    state_clear_pixel_count: int
    state_echo_object_count: int
    contract: str = "neural-prior-range-band-evaluation-v6"
    evaluation_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-range-band-evaluation-v6"
            or not self.range_regime
            or self.range_regime.strip() != self.range_regime
        ):
            raise ValueError("range-band evaluation identity is invalid")
        _require_digest("range-band evaluation mask", self.range_band_mask_digest)
        _require_digest(
            "range-band evaluation geometry", self.range_geometry_contract_digest
        )
        _require_digest(
            "range-band operational issuance domain", self.issuance_domain_digest
        )
        change = self.metric_change.detach().clone()
        end_to_end = self.end_to_end_metric_change.detach().clone()
        available = self.metric_available.detach().clone()
        metric_valid_area = self.metric_valid_area_km2.detach().clone()
        withdrawn = self.withdrawn_fraction_by_lead.detach().clone()
        newly_issued = self.newly_issued_fraction_by_lead.detach().clone()
        fallback_increase = self.background_fallback_increase_by_lead.detach().clone()
        confidence_change = (
            self.confidence_weighted_coverage_change_by_lead.detach().clone()
        )
        if (
            change.shape != end_to_end.shape
            or change.shape != available.shape
            or metric_valid_area.shape != available.shape
            or not metric_valid_area.is_floating_point()
            or not bool(torch.all(torch.isfinite(metric_valid_area)))
            or bool(torch.any(metric_valid_area < 0.0))
            or bool(torch.any(metric_valid_area[~available] != 0.0))
            or available.dtype is not torch.bool
            or not change.is_floating_point()
            or not end_to_end.is_floating_point()
            or not bool(torch.any(available))
            or not bool(torch.all(torch.isfinite(change[available])))
            or not bool(torch.all(torch.isfinite(end_to_end[available])))
            or any(
                item.shape != (change.shape[0],)
                or not item.is_floating_point()
                or not bool(torch.all(torch.isfinite(item)))
                for item in (
                    withdrawn,
                    newly_issued,
                    fallback_increase,
                    confidence_change,
                )
            )
            or bool(torch.any((withdrawn < 0.0) | (withdrawn > 1.0)))
            or bool(torch.any((newly_issued < 0.0) | (newly_issued > 1.0)))
            or bool(torch.any(torch.abs(fallback_increase) > 1.0))
            or bool(torch.any(torch.abs(confidence_change) > 1.0))
        ):
            raise ValueError("range-band metric evidence is invalid")
        candidate_components = tuple(
            name for name, _ in self.candidate_uncertainty_component_scores
        )
        parent_components = tuple(
            name for name, _ in self.parent_uncertainty_component_scores
        )
        components = tuple(name for name, _ in self.uncertainty_component_differences)
        sample_components = tuple(
            name for name, _ in self.uncertainty_component_sample_counts
        )
        valid_areas = (
            self.metric_valid_area_km2_by_lead
            + (
                self.probability_valid_area_km2,
                self.state_valid_area_km2,
            )
        )
        physical_counts = (
            self.echo_pixel_count,
            self.clear_pixel_count,
            self.echo_object_count,
            self.state_echo_pixel_count,
            self.state_clear_pixel_count,
            self.state_echo_object_count,
        )
        issuance_lengths = (
            self.issuance_domain_cell_count_by_lead,
            self.issuance_domain_area_km2_by_lead,
            self.parent_issued_count_by_lead,
            self.candidate_issued_count_by_lead,
            self.withdrawn_count_by_lead,
            self.newly_issued_count_by_lead,
            self.parent_fallback_count_by_lead,
            self.candidate_fallback_count_by_lead,
            self.parent_confidence_weighted_issued_area_by_lead,
            self.candidate_confidence_weighted_issued_area_by_lead,
        )
        component_counts = dict(self.uncertainty_component_sample_counts)
        candidate_scores = dict(self.candidate_uncertainty_component_scores)
        parent_scores = dict(self.parent_uncertainty_component_scores)
        differences = dict(self.uncertainty_component_differences)
        expected_component_counts = {
            "intensity": self.echo_pixel_count,
            "pit_residual": self.echo_pixel_count,
            "support": self.echo_pixel_count + self.clear_pixel_count,
            "echo_miss": self.echo_pixel_count,
            "object_miss": self.echo_object_count,
            "clear": self.clear_pixel_count,
            "underdispersion": self.echo_pixel_count,
            "state_echo_miss": self.state_echo_pixel_count,
            "state_nll": self.state_echo_pixel_count + self.state_clear_pixel_count,
            "state_pit_residual": (
                self.state_echo_pixel_count + self.state_clear_pixel_count
            ),
            "state_underdispersion": (
                self.state_echo_pixel_count + self.state_clear_pixel_count
            ),
            "state_support": (
                self.state_echo_pixel_count + self.state_clear_pixel_count
            ),
            "state_object_miss": self.state_echo_object_count,
            "state_false_support": self.state_clear_pixel_count,
            "state_valid": self.state_echo_pixel_count + self.state_clear_pixel_count,
        }
        if (
            not components
            or len(set(components)) != len(components)
            or candidate_components != components
            or parent_components != components
            or set(components) != set(sample_components)
            or any(
                not math.isfinite(value)
                for _, value in (
                    self.candidate_uncertainty_component_scores
                    + self.parent_uncertainty_component_scores
                )
            )
            or any(
                name not in _UNCERTAINTY_COMPONENT_NAMES
                or not math.isfinite(value)
                for name, value in self.uncertainty_component_differences
            )
            or any(
                not math.isclose(
                    candidate_scores[name] - parent_scores[name],
                    differences[name],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                for name in components
            )
            or any(
                type(count) is not int or count <= 0
                for _, count in self.uncertainty_component_sample_counts
            )
            or not math.isfinite(self.evaluated_area_km2)
            or self.evaluated_area_km2 <= 0.0
            or len(self.metric_valid_area_km2_by_lead) != change.shape[0]
            or any(not math.isfinite(value) or value < 0.0 for value in valid_areas)
            or any(type(count) is not int or count < 0 for count in physical_counts)
            or any(len(values) != change.shape[0] for values in issuance_lengths)
            or any(
                type(value) is not int or value <= 0
                for value in self.issuance_domain_cell_count_by_lead
            )
            or any(
                type(value) is not int or value < 0
                for values in (
                    self.parent_issued_count_by_lead,
                    self.candidate_issued_count_by_lead,
                    self.withdrawn_count_by_lead,
                    self.newly_issued_count_by_lead,
                    self.parent_fallback_count_by_lead,
                    self.candidate_fallback_count_by_lead,
                )
                for value in values
            )
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.issuance_domain_area_km2_by_lead
            )
            or any(
                not math.isfinite(value) or value < 0.0
                for values in (
                    self.parent_confidence_weighted_issued_area_by_lead,
                    self.candidate_confidence_weighted_issued_area_by_lead,
                )
                for value in values
            )
            or any(
                not math.isclose(
                    float(withdrawn[index]),
                    self.withdrawn_count_by_lead[index]
                    / max(1, self.parent_issued_count_by_lead[index]),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    float(newly_issued[index]),
                    self.newly_issued_count_by_lead[index]
                    / self.issuance_domain_cell_count_by_lead[index],
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    float(fallback_increase[index]),
                    (
                        self.candidate_fallback_count_by_lead[index]
                        - self.parent_fallback_count_by_lead[index]
                    )
                    / self.issuance_domain_cell_count_by_lead[index],
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    float(confidence_change[index]),
                    (
                        self.candidate_confidence_weighted_issued_area_by_lead[index]
                        - self.parent_confidence_weighted_issued_area_by_lead[index]
                    )
                    / self.issuance_domain_area_km2_by_lead[index],
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
                for index in range(change.shape[0])
            )
            or any(
                component in component_counts
                and component_counts[component] != expected_count
                for component, expected_count in expected_component_counts.items()
            )
        ):
            raise ValueError("range-band uncertainty evidence is invalid")
        object.__setattr__(self, "metric_change", change)
        object.__setattr__(self, "end_to_end_metric_change", end_to_end)
        object.__setattr__(self, "metric_available", available)
        object.__setattr__(self, "metric_valid_area_km2", metric_valid_area)
        object.__setattr__(self, "withdrawn_fraction_by_lead", withdrawn)
        object.__setattr__(self, "newly_issued_fraction_by_lead", newly_issued)
        object.__setattr__(
            self,
            "background_fallback_increase_by_lead",
            fallback_increase,
        )
        object.__setattr__(
            self,
            "confidence_weighted_coverage_change_by_lead",
            confidence_change,
        )
        object.__setattr__(self, "evaluation_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "range_regime": self.range_regime,
            "range_band_mask_digest": self.range_band_mask_digest,
            "range_geometry_contract_digest": self.range_geometry_contract_digest,
            "metric_change": tensor_digest(self.metric_change),
            "end_to_end_metric_change": tensor_digest(
                self.end_to_end_metric_change
            ),
            "metric_available": tensor_digest(self.metric_available),
            "metric_valid_area_km2": tensor_digest(self.metric_valid_area_km2),
            "issuance_domain_digest": self.issuance_domain_digest,
            "issuance_domain_cell_count_by_lead": list(
                self.issuance_domain_cell_count_by_lead
            ),
            "issuance_domain_area_km2_by_lead": list(
                self.issuance_domain_area_km2_by_lead
            ),
            "parent_issued_count_by_lead": list(self.parent_issued_count_by_lead),
            "candidate_issued_count_by_lead": list(
                self.candidate_issued_count_by_lead
            ),
            "withdrawn_count_by_lead": list(self.withdrawn_count_by_lead),
            "newly_issued_count_by_lead": list(self.newly_issued_count_by_lead),
            "parent_fallback_count_by_lead": list(
                self.parent_fallback_count_by_lead
            ),
            "candidate_fallback_count_by_lead": list(
                self.candidate_fallback_count_by_lead
            ),
            "parent_confidence_weighted_issued_area_by_lead": list(
                self.parent_confidence_weighted_issued_area_by_lead
            ),
            "candidate_confidence_weighted_issued_area_by_lead": list(
                self.candidate_confidence_weighted_issued_area_by_lead
            ),
            "withdrawn_fraction_by_lead": tensor_digest(
                self.withdrawn_fraction_by_lead
            ),
            "newly_issued_fraction_by_lead": tensor_digest(
                self.newly_issued_fraction_by_lead
            ),
            "background_fallback_increase_by_lead": tensor_digest(
                self.background_fallback_increase_by_lead
            ),
            "confidence_weighted_coverage_change_by_lead": tensor_digest(
                self.confidence_weighted_coverage_change_by_lead
            ),
            "candidate_uncertainty_component_scores": [
                [name, value]
                for name, value in self.candidate_uncertainty_component_scores
            ],
            "parent_uncertainty_component_scores": [
                [name, value]
                for name, value in self.parent_uncertainty_component_scores
            ],
            "uncertainty_component_differences": [
                [name, value]
                for name, value in self.uncertainty_component_differences
            ],
            "uncertainty_component_sample_counts": [
                [name, count]
                for name, count in self.uncertainty_component_sample_counts
            ],
            "evaluated_area_km2": self.evaluated_area_km2,
            "metric_valid_area_km2_by_lead": list(
                self.metric_valid_area_km2_by_lead
            ),
            "probability_valid_area_km2": self.probability_valid_area_km2,
            "state_valid_area_km2": self.state_valid_area_km2,
            "echo_pixel_count": self.echo_pixel_count,
            "clear_pixel_count": self.clear_pixel_count,
            "echo_object_count": self.echo_object_count,
            "state_echo_pixel_count": self.state_echo_pixel_count,
            "state_clear_pixel_count": self.state_clear_pixel_count,
            "state_echo_object_count": self.state_echo_object_count,
        }

    def component_difference(self, component: str) -> float | None:
        return dict(self.uncertainty_component_differences).get(component)

    def candidate_component_score(self, component: str) -> float | None:
        return dict(self.candidate_uncertainty_component_scores).get(component)

    def parent_component_score(self, component: str) -> float | None:
        return dict(self.parent_uncertainty_component_scores).get(component)


@dataclass(frozen=True, init=False)
class PriorHoldoutEvaluation:
    """Paired prior holdout result over the full preregistered population."""

    holdout_plan_digest: str
    candidate_manifest_digest: str
    candidate_prior_digest: str
    parent_prior_digest: str
    case_id: str
    storm_id: str
    physical_event_digest: str
    day: str
    radar_id: str
    regime: str
    range_regime: str
    reference_active_range_regimes: tuple[str, ...]
    range_band_contract_digest: str
    range_band_evaluations: tuple[RangeBandEvaluation, ...]
    regime_classifier_digest: str
    regime_classifier_manifest_digest: str
    regime_classification_evidence_digest: str
    classified_regime: str
    classified_range_regimes: tuple[str, ...]
    classifier_regime_confidence: float
    classifier_range_confidence: float
    classifier_regime_entropy: float
    classifier_is_ood: bool
    classifier_reference_agreement: bool
    classifier_weather_reference_agreement: bool
    classifier_range_set_precision: float
    classifier_range_set_recall: float
    classifier_range_exact_set_match: bool
    classifier_false_active_band_fraction: float
    classifier_reference_range_is_ood: bool
    classifier_numerical_runtime_digest: str
    classifier_input_dtype: str
    classifier_input_device: str
    classifier_weather_top1_top2_gap: float
    classifier_minimum_range_presence_margin: float
    candidate_forecast_digest: str
    parent_forecast_digest: str
    candidate_prior_application_digest: str
    parent_prior_application_digest: str
    candidate_inference_evidence_digest: str
    parent_inference_evidence_digest: str
    metric_change: Tensor
    candidate_issuance_effect: Tensor
    parent_issuance_effect: Tensor
    end_to_end_metric_change: Tensor
    metric_available: Tensor
    lead_minutes: tuple[int, ...]
    metric_names: tuple[str, ...]
    verification_digest: str
    metric_contract_digest: str
    coverage_candidate: Tensor
    coverage_parent: Tensor
    coverage_common: Tensor
    newly_issued_fraction: Tensor
    withdrawn_fraction: Tensor
    prior_conditional_pit_residual_mean_abs: float | None
    prior_conditional_underdispersion_fraction: float | None
    prior_echo_intensity_nll: float | None
    prior_support_brier_score: float
    prior_echo_support_miss_score: float | None
    prior_echo_object_miss_score: float | None
    prior_clear_sky_false_echo_score: float | None
    parent_prior_conditional_pit_residual_mean_abs: float | None
    parent_prior_conditional_underdispersion_fraction: float | None
    parent_prior_echo_intensity_nll: float | None
    parent_prior_support_brier_score: float
    parent_prior_echo_support_miss_score: float | None
    parent_prior_echo_object_miss_score: float | None
    parent_prior_clear_sky_false_echo_score: float | None
    prior_echo_intensity_status: PriorComponentStatus
    prior_clear_sky_status: PriorComponentStatus
    prior_candidate_valid_fraction: float
    prior_parent_valid_fraction: float
    prior_candidate_valid_area_km2: float
    prior_abstention_increase_vs_parent: float
    prior_uncertainty_target_digest: str
    prior_uncertainty_sample_count: int
    prior_echo_intensity_sample_count: int
    prior_clear_sky_sample_count: int
    prior_echo_area_km2: float
    prior_clear_sky_area_km2: float
    prior_echo_object_count: int
    state_candidate_gaussian_nll: float
    state_parent_gaussian_nll: float
    state_candidate_pit_residual_mean_abs: float
    state_parent_pit_residual_mean_abs: float
    state_candidate_underdispersion_fraction: float
    state_parent_underdispersion_fraction: float
    state_candidate_support_brier_score: float
    state_parent_support_brier_score: float
    state_candidate_echo_support_miss_score: float | None
    state_parent_echo_support_miss_score: float | None
    state_candidate_echo_object_miss_score: float | None
    state_parent_echo_object_miss_score: float | None
    state_candidate_false_support_score: float | None
    state_parent_false_support_score: float | None
    state_candidate_valid_brier_score: float
    state_parent_valid_brier_score: float
    state_calibration_target_digest: str
    state_calibration_sample_count: int
    state_calibration_echo_sample_count: int
    state_calibration_clear_sample_count: int
    state_calibration_echo_object_count: int
    issue_time: str
    verification_valid_times: tuple[str, ...]
    contract: str = "prior-holdout-evaluation-v16"
    evaluation_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use PriorHoldoutEvaluation.from_forecasts")

    def __post_init__(self) -> None:
        if self.contract != "prior-holdout-evaluation-v16":
            raise ValueError("unsupported prior holdout evaluation")
        for name in (
            "holdout_plan_digest",
            "candidate_manifest_digest",
            "candidate_prior_digest",
            "parent_prior_digest",
            "candidate_forecast_digest",
            "parent_forecast_digest",
            "candidate_prior_application_digest",
            "parent_prior_application_digest",
            "candidate_inference_evidence_digest",
            "parent_inference_evidence_digest",
            "verification_digest",
            "metric_contract_digest",
            "prior_uncertainty_target_digest",
            "state_calibration_target_digest",
            "regime_classifier_digest",
            "regime_classifier_manifest_digest",
            "regime_classification_evidence_digest",
            "range_band_contract_digest",
            "classifier_numerical_runtime_digest",
            "physical_event_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            not self.classified_regime
            or (not self.classified_range_regimes and not self.classifier_is_ood)
            or len(set(self.classified_range_regimes))
            != len(self.classified_range_regimes)
            or any(not value for value in self.classified_range_regimes)
            or not 0.0 <= self.classifier_regime_confidence <= 1.0
            or not 0.0 <= self.classifier_range_confidence <= 1.0
            or not math.isfinite(self.classifier_regime_entropy)
            or self.classifier_regime_entropy < 0.0
            or len(set(self.reference_active_range_regimes))
            != len(self.reference_active_range_regimes)
            or any(not value for value in self.reference_active_range_regimes)
            or len(self.range_band_evaluations)
            != len(self.reference_active_range_regimes)
            or tuple(item.range_regime for item in self.range_band_evaluations)
            != self.reference_active_range_regimes
            or any(
                item.evaluation_digest != json_digest(item.payload)
                for item in self.range_band_evaluations
            )
            or not 0.0 <= self.classifier_range_set_precision <= 1.0
            or not 0.0 <= self.classifier_range_set_recall <= 1.0
            or not 0.0 <= self.classifier_false_active_band_fraction <= 1.0
            or not self.classifier_input_dtype
            or not self.classifier_input_device
            or not math.isfinite(self.classifier_weather_top1_top2_gap)
            or self.classifier_weather_top1_top2_gap < 0.0
            or not math.isfinite(self.classifier_minimum_range_presence_margin)
            or self.classifier_minimum_range_presence_margin < 0.0
        ):
            raise ValueError("holdout regime-classifier evidence is invalid")
        expected = (len(self.lead_minutes), len(self.metric_names))
        change = self.metric_change.detach().clone()
        candidate_policy = self.candidate_issuance_effect.detach().clone()
        parent_policy = self.parent_issuance_effect.detach().clone()
        end_to_end = self.end_to_end_metric_change.detach().clone()
        available = self.metric_available.detach().clone()
        candidate_coverage = self.coverage_candidate.detach().clone()
        parent_coverage = self.coverage_parent.detach().clone()
        common_coverage = self.coverage_common.detach().clone()
        newly_issued = self.newly_issued_fraction.detach().clone()
        withdrawn = self.withdrawn_fraction.detach().clone()
        if (
            change.shape != expected
            or candidate_policy.shape != expected
            or parent_policy.shape != expected
            or end_to_end.shape != expected
            or available.shape != expected
            or available.dtype is not torch.bool
            or candidate_coverage.shape != (len(self.lead_minutes),)
            or parent_coverage.shape != (len(self.lead_minutes),)
            or common_coverage.shape != (len(self.lead_minutes),)
            or newly_issued.shape != (len(self.lead_minutes),)
            or withdrawn.shape != (len(self.lead_minutes),)
            or any(
                not value.is_floating_point()
                for value in (change, candidate_policy, parent_policy, end_to_end)
            )
        ):
            raise ValueError("realized evaluation shapes disagree")
        if not bool(torch.any(available)) or any(
            not bool(torch.all(torch.isfinite(value[available])))
            for value in (change, candidate_policy, parent_policy, end_to_end)
        ):
            raise ValueError("realized evaluation must contain finite metrics")
        for coverage in (
            candidate_coverage,
            parent_coverage,
            common_coverage,
            newly_issued,
            withdrawn,
        ):
            if not bool(torch.all(torch.isfinite(coverage))) or bool(
                torch.any((coverage < 0.0) | (coverage > 1.0))
            ):
                raise ValueError("realized evaluation coverage must be in [0,1]")
        if self.candidate_prior_digest == self.parent_prior_digest:
            raise ValueError("candidate and parent priors must differ")
        echo_available = self.prior_echo_intensity_status == "available"
        clear_available = self.prior_clear_sky_status == "available"
        if self.prior_echo_intensity_status not in (
            "available",
            "not_applicable",
        ) or self.prior_clear_sky_status not in (
            "available",
            "not_applicable",
        ):
            raise ValueError("prior component status is invalid")
        echo_values = (
            self.prior_conditional_pit_residual_mean_abs,
            self.prior_conditional_underdispersion_fraction,
            self.prior_echo_intensity_nll,
            self.prior_echo_support_miss_score,
            self.prior_echo_object_miss_score,
            self.parent_prior_conditional_pit_residual_mean_abs,
            self.parent_prior_conditional_underdispersion_fraction,
            self.parent_prior_echo_intensity_nll,
            self.parent_prior_echo_support_miss_score,
            self.parent_prior_echo_object_miss_score,
        )
        clear_values = (
            self.prior_clear_sky_false_echo_score,
            self.parent_prior_clear_sky_false_echo_score,
        )
        if (
            (echo_available and any(value is None for value in echo_values))
            or (
                not echo_available
                and any(value is not None for value in echo_values)
            )
            or (clear_available and any(value is None for value in clear_values))
            or (
                not clear_available
                and any(value is not None for value in clear_values)
            )
            or echo_available != (self.prior_echo_intensity_sample_count > 0)
            or clear_available != (self.prior_clear_sky_sample_count > 0)
            or echo_available != (self.prior_echo_area_km2 > 0.0)
            or clear_available != (self.prior_clear_sky_area_km2 > 0.0)
            or echo_available != (self.prior_echo_object_count > 0)
        ):
            raise ValueError("prior component applicability is inconsistent")
        if (
            not math.isfinite(self.prior_support_brier_score)
            or not 0.0 <= self.prior_support_brier_score <= 1.0
            or not math.isfinite(self.parent_prior_support_brier_score)
            or not 0.0 <= self.parent_prior_support_brier_score <= 1.0
            or type(self.prior_uncertainty_sample_count) is not int
            or self.prior_uncertainty_sample_count <= 0
            or type(self.prior_echo_intensity_sample_count) is not int
            or self.prior_echo_intensity_sample_count < 0
            or type(self.prior_clear_sky_sample_count) is not int
            or self.prior_clear_sky_sample_count < 0
            or not math.isfinite(self.prior_echo_area_km2)
            or self.prior_echo_area_km2 < 0.0
            or not math.isfinite(self.prior_clear_sky_area_km2)
            or self.prior_clear_sky_area_km2 < 0.0
            or type(self.prior_echo_object_count) is not int
            or self.prior_echo_object_count < 0
            or self.prior_echo_intensity_sample_count
            + self.prior_clear_sky_sample_count
            != self.prior_uncertainty_sample_count
            or not 0.0 <= self.prior_candidate_valid_fraction <= 1.0
            or not 0.0 <= self.prior_parent_valid_fraction <= 1.0
            or not math.isfinite(self.prior_candidate_valid_area_km2)
            or self.prior_candidate_valid_area_km2 < 0.0
            or not math.isfinite(self.prior_abstention_increase_vs_parent)
            or not -1.0 <= self.prior_abstention_increase_vs_parent <= 1.0
        ):
            raise ValueError("prior uncertainty reliability evidence is invalid")
        if echo_available:
            echo_floats = tuple(
                value for value in echo_values if value is not None
            )
            if (
                any(not math.isfinite(value) for value in echo_floats)
                or echo_floats[0] < 0.0
                or not 0.0 <= echo_floats[1] <= 1.0
                or not 0.0 <= echo_floats[3] <= 1.0
                or not 0.0 <= echo_floats[4] <= 1.0
                or echo_floats[5] < 0.0
                or not 0.0 <= echo_floats[6] <= 1.0
                or not 0.0 <= echo_floats[8] <= 1.0
                or not 0.0 <= echo_floats[9] <= 1.0
            ):
                raise ValueError("prior echo-intensity evidence is invalid")
        if clear_available:
            clear_floats = tuple(
                value for value in clear_values if value is not None
            )
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in clear_floats
            ):
                raise ValueError("prior clear-sky evidence is invalid")
        state_scores = (
            self.state_candidate_gaussian_nll,
            self.state_parent_gaussian_nll,
            self.state_candidate_pit_residual_mean_abs,
            self.state_parent_pit_residual_mean_abs,
            self.state_candidate_underdispersion_fraction,
            self.state_parent_underdispersion_fraction,
            self.state_candidate_support_brier_score,
            self.state_parent_support_brier_score,
            self.state_candidate_valid_brier_score,
            self.state_parent_valid_brier_score,
        )
        optional_state_scores = (
            self.state_candidate_echo_support_miss_score,
            self.state_parent_echo_support_miss_score,
            self.state_candidate_echo_object_miss_score,
            self.state_parent_echo_object_miss_score,
            self.state_candidate_false_support_score,
            self.state_parent_false_support_score,
        )
        if (
            any(not math.isfinite(value) for value in state_scores)
            or any(
                value is not None and not math.isfinite(value)
                for value in optional_state_scores
            )
            or any(
                not 0.0 <= value <= 1.0
                for value in state_scores[4:]
            )
            or any(
                value is not None and not 0.0 <= value <= 1.0
                for value in optional_state_scores
            )
            or self.state_candidate_pit_residual_mean_abs < 0.0
            or self.state_parent_pit_residual_mean_abs < 0.0
            or type(self.state_calibration_sample_count) is not int
            or self.state_calibration_sample_count <= 0
            or self.state_calibration_echo_sample_count < 0
            or self.state_calibration_clear_sample_count < 0
            or self.state_calibration_echo_object_count < 0
            or self.state_calibration_echo_sample_count
            + self.state_calibration_clear_sample_count
            != self.state_calibration_sample_count
            or (self.state_calibration_echo_sample_count > 0)
            != (
                self.state_candidate_echo_support_miss_score is not None
                and self.state_parent_echo_support_miss_score is not None
                and self.state_candidate_echo_object_miss_score is not None
                and self.state_parent_echo_object_miss_score is not None
                and self.state_calibration_echo_object_count > 0
            )
            or (self.state_calibration_clear_sample_count > 0)
            != (
                self.state_candidate_false_support_score is not None
                and self.state_parent_false_support_score is not None
            )
        ):
            raise ValueError("state-head calibration evidence is invalid")
        issue = datetime.fromisoformat(self.issue_time.replace("Z", "+00:00"))
        valid = tuple(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in self.verification_valid_times
        )
        if any(issue >= value for value in valid):
            raise ValueError("holdout issue and verification times disagree")
        object.__setattr__(self, "metric_change", change)
        object.__setattr__(self, "candidate_issuance_effect", candidate_policy)
        object.__setattr__(self, "parent_issuance_effect", parent_policy)
        object.__setattr__(self, "end_to_end_metric_change", end_to_end)
        object.__setattr__(self, "metric_available", available)
        object.__setattr__(self, "coverage_candidate", candidate_coverage)
        object.__setattr__(self, "coverage_parent", parent_coverage)
        object.__setattr__(self, "coverage_common", common_coverage)
        object.__setattr__(self, "newly_issued_fraction", newly_issued)
        object.__setattr__(self, "withdrawn_fraction", withdrawn)
        object.__setattr__(self, "evaluation_digest", _evaluation_digest(self))

    @classmethod
    def from_forecasts(
        cls,
        manifest: NeuralPriorCandidateManifest,
        plan: NeuralPriorHoldoutPlan,
        *,
        case_id: str,
        candidate_forecast: ForecastResult,
        parent_forecast: ForecastResult,
        verification: VerificationBundle,
        metric_config: SensitivityConfig,
        candidate_prior_application: NeuralPriorApplication,
        parent_prior_application: NeuralPriorApplication,
        candidate_prior_runner: NeuralPriorInferenceRunner,
        parent_prior_runner: NeuralPriorInferenceRunner,
        input_frames_dbz: Tensor,
        uncertainty_target: PriorUncertaintyTarget,
        state_calibration_target: NeuralPriorStateCalibrationTarget,
        regime_classifier: NeuralPriorRegimeClassifier,
        regime_classifier_manifest: RegimeClassifierManifest,
        range_grid_x_m: Tensor,
        range_grid_y_m: Tensor,
    ) -> PriorHoldoutEvaluation:
        """Evaluate every planned prior case without intervention selection."""

        validate_neural_prior_holdout_plan(plan)
        validate_neural_prior_candidate_manifest(manifest)
        regime_classification_evidence = regime_classifier.classify(
            input_frames_dbz,
            input_run=candidate_forecast.run,
        )
        regime_classification_evidence.validate_integrity()
        if (
            regime_classification_evidence.classifier_digest
            != regime_classifier.classifier_digest
            or regime_classifier_manifest.classifier_digest
            != regime_classifier.classifier_digest
            or regime_classifier_manifest.numerical_runtime_digest
            != regime_classifier.numerical_runtime_digest
            or regime_classifier_manifest not in plan.regime_classifier_manifests
        ):
            raise ValueError("holdout regime-classifier evidence is untrusted")
        candidate_forecast.validate_issuance()
        parent_forecast.validate_issuance()
        if manifest.holdout_plan_digest != plan.plan_digest:
            raise ValueError("candidate manifest and holdout plan disagree")
        if manifest.candidate_prior_digest not in plan.candidate_family_digests or (
            manifest.parent_prior_digest != plan.parent_prior_digest
        ):
            raise ValueError("candidate priors are outside the holdout plan")
        case = manifest.holdout_case(case_id)
        planned_case = plan.case(case_id)
        reference_plan = next(
            item
            for item in plan.regime_reference_plans
            if item.plan_digest == planned_case.regime_reference_plan_digest
        )
        reference_evidence = next(
            item
            for item in manifest.regime_reference_evidences
            if item.evidence_digest == case.regime_reference_evidence_digest
        )
        validate_regime_reference_evidence(reference_evidence, reference_plan)
        if (
            reference_evidence.observed_regime != case.regime
            or reference_evidence.observed_storm_id != case.storm_id
            or reference_evidence.full_analysis_input_digest
            != case.full_analysis_input_digest
            or reference_evidence.verification_bundle_digest
            != case.verification_bundle_digest
        ):
            raise ValueError("regime-reference evidence disagrees with its case")
        range_contract = next(
            item
            for item in plan.range_band_contracts
            if item.contract_digest == planned_case.range_band_contract_digest
        )
        range_geometry = next(
            item
            for item in plan.range_geometry_contracts
            if item.contract_digest
            == range_contract.range_geometry_contract_digest
        )
        range_partition = resolve_range_geometry(
            range_geometry,
            grid_x_m=range_grid_x_m,
            grid_y_m=range_grid_y_m,
        )
        range_band_masks = {
            label: range_partition.mask(label)
            for label in range_partition.range_regime_labels
        }
        if (
            set(range_band_masks) != set(range_contract.range_regime_labels)
            or range_partition.range_geometry_contract_digest
            != range_contract.range_geometry_contract_digest
            or range_partition.range_band_mask_digests
            != range_contract.range_band_mask_digests
            or any(
                mask.dtype is not torch.bool
                or mask.ndim != 2
                or mask.shape != input_frames_dbz.shape[-2:]
                or mask.device != input_frames_dbz.device
                or tensor_digest(mask) != range_contract.mask_digest(label)
                for label, mask in range_band_masks.items()
            )
            or tuple(
                label
                for label in range_contract.range_regime_labels
                if bool(torch.any(range_band_masks[label]))
            )
            != range_contract.reference_active_range_regimes
        ):
            raise ValueError("holdout range-band masks disagree with their plan")
        _validate_complete_range_partition(range_band_masks)
        if (
            regime_classification_evidence.full_analysis_input_digest
            != candidate_forecast.run.full_analysis_input_digest
            or regime_classification_evidence.input_frames_digest
            != tensor_digest(input_frames_dbz)
        ):
            raise ValueError("holdout regime classification used different input")
        input_plan = next(
            item for item in plan.input_plans
            if item.plan_digest == planned_case.input_plan_digest
        )
        if case.plan_case() != planned_case:
            raise ValueError("completed holdout case disagrees with its plan")
        candidate_digest = _forecast_result_content_digest(candidate_forecast)
        parent_digest = _forecast_result_content_digest(parent_forecast)
        if (
            candidate_digest != case.candidate_forecast_digest
            or parent_digest != case.parent_forecast_digest
        ):
            raise ValueError("holdout forecast does not match the candidate manifest")
        if (
            candidate_forecast.run.grid_time_contract_digest
            != parent_forecast.run.grid_time_contract_digest
            or candidate_forecast.run.grid_time_contract is None
        ):
            raise ValueError("candidate and parent holdout grids disagree")
        if (
            candidate_forecast.run.input_bundle_digest
            != parent_forecast.run.input_bundle_digest
            or candidate_forecast.run.input_bundle_digest
            != case.input_bundle_digest
            or candidate_forecast.run.full_analysis_input_digest
            != parent_forecast.run.full_analysis_input_digest
            or candidate_forecast.run.full_analysis_input_digest
            != case.full_analysis_input_digest
            or candidate_forecast.run.fixed_input_context_digest
            != parent_forecast.run.fixed_input_context_digest
            or candidate_forecast.run.fixed_input_context_digest
            != case.fixed_input_context_digest
            or candidate_forecast.run.observation_quality_weight_digest
            != parent_forecast.run.observation_quality_weight_digest
            or candidate_forecast.run.observation_quality_weight_digest
            != case.observation_quality_weight_digest
            or candidate_forecast.run.observation_std_dbz_digest
            != parent_forecast.run.observation_std_dbz_digest
            or candidate_forecast.run.observation_std_dbz_digest
            != case.observation_std_dbz_digest
        ):
            raise ValueError("candidate and parent holdout inputs disagree")
        if (
            candidate_forecast.run.input_plan_digest != case.input_plan_digest
            or parent_forecast.run.input_plan_digest != case.input_plan_digest
            or candidate_forecast.run.input_plan_resolution_digest
            != case.input_plan_resolution_digest
            or parent_forecast.run.input_plan_resolution_digest
            != case.input_plan_resolution_digest
        ):
            raise ValueError("holdout forecast input plan disagrees")
        for run in (candidate_forecast.run, parent_forecast.run):
            grid = run.grid_time_contract
            if grid is None:
                raise ValueError("holdout forecast grid contract is missing")
            expected_resolution = _forecast_input_plan_resolution_digest(
                input_plan_digest=input_plan.plan_digest,
                full_analysis_input_digest=case.full_analysis_input_digest,
            )
            if (
                run.input_plan_json != input_plan.json
                or run.input_plan_resolution_digest != expected_resolution
                or run.grid_time_contract_digest != input_plan.grid_contract_digest
                or tuple(grid.valid_times) != input_plan.valid_times
                or run.operational_data_identity_json is None
            ):
                raise ValueError("holdout input plan was not resolved by the run")
            identity = OperationalDataIdentity.from_json(
                run.operational_data_identity_json
            )
            if (
                identity.radar_product_digest
                != input_plan.radar_product_digest
                or identity.qc_pipeline_digest != input_plan.qc_pipeline_digest
                or identity.background_cycle_rule_digest
                != input_plan.background_cycle_rule_digest
                or identity.mask_policy_digest != input_plan.mask_policy_digest
            ):
                raise ValueError("holdout input identity disagrees with its plan")
        if (
            candidate_forecast.run.neural_prior_digest
            != manifest.candidate_prior_digest
            or candidate_forecast.run.prior_role != "candidate"
            or parent_forecast.run.neural_prior_digest
            != manifest.parent_prior_digest
            or parent_forecast.run.prior_role != "parent"
        ):
            raise ValueError("holdout forecast prior lineage disagrees")
        if (
            candidate_forecast.run.prior_application_digest is None
            or parent_forecast.run.prior_application_digest is None
            or candidate_forecast.run.prior_application_digest
            == parent_forecast.run.prior_application_digest
        ):
            raise ValueError("holdout forecasts must consume distinct prior outputs")
        candidate_prior_runner.reproduce(
            candidate_prior_application,
            input_frames_dbz,
        )
        parent_prior_runner.reproduce(
            parent_prior_application,
            input_frames_dbz,
        )
        for run, application in (
            (candidate_forecast.run, candidate_prior_application),
            (parent_forecast.run, parent_prior_application),
        ):
            evidence = application.inference_evidence
            if (
                application.application_digest
                not in (
                    case.candidate_prior_application_digest,
                    case.parent_prior_application_digest,
                )
                or run.prior_application_digest != application.application_digest
                or run.prior_inference_evidence_digest != evidence.evidence_digest
                or run.prior_inference_algorithm_digest
                != evidence.inference_algorithm_digest
                or run.prior_numerical_runtime_digest
                != evidence.numerical_runtime_digest
                or run.prior_dependency != evidence.dependency
                or run.neural_prior_digest != evidence.neural_prior_digest
                or run.prior_model_contract_digest != evidence.model_contract_digest
                or run.prior_feature_schema_digest != evidence.feature_schema_digest
                or run.prior_training_manifest_digest
                != evidence.training_manifest_digest
                or evidence.input_bundle_digest != case.input_bundle_digest
                or evidence.full_analysis_input_digest
                != case.full_analysis_input_digest
                or evidence.input_frames_digest != tensor_digest(input_frames_dbz)
                or evidence.execution_contract_digest != run.neural_prior_digest
                or evidence.uncertainty_contract != "model_spatial"
            ):
                raise ValueError("holdout prior inference evidence disagrees")
        if (
            candidate_prior_application.application_digest
            != case.candidate_prior_application_digest
            or parent_prior_application.application_digest
            != case.parent_prior_application_digest
            or candidate_prior_application.inference_evidence.evidence_digest
            != case.candidate_inference_evidence_digest
            or parent_prior_application.inference_evidence.evidence_digest
            != case.parent_inference_evidence_digest
        ):
            raise ValueError("holdout prior inference is not manifested")
        if (
            candidate_prior_runner.inference_algorithm_digest
            != parent_prior_runner.inference_algorithm_digest
            or candidate_prior_runner.numerical_runtime_digest
            != parent_prior_runner.numerical_runtime_digest
            or candidate_prior_runner.numerical_runtime_digest
            != manifest.numerical_runtime_digest
        ):
            raise ValueError("holdout inference runtime is not manifested")
        if (
            uncertainty_target.target_plan_digest
            != case.uncertainty_target_plan_digest
            or uncertainty_target.target_digest != case.uncertainty_target_digest
        ):
            raise ValueError("prior uncertainty target is not independent and planned")
        if (
            state_calibration_target.target_plan_digest
            != case.state_calibration_target_plan_digest
            or state_calibration_target.target_digest
            != case.state_calibration_target_digest
        ):
            raise ValueError("state calibration target is not independent and planned")
        candidate_evidence = candidate_prior_application.inference_evidence
        parent_evidence = parent_prior_application.inference_evidence
        target_plan = next(
            item for item in plan.uncertainty_target_plans
            if item.plan_digest == case.uncertainty_target_plan_digest
        )
        state_target_plan = next(
            item
            for item in plan.state_calibration_target_plans
            if item.plan_digest == case.state_calibration_target_plan_digest
        )
        candidate_probability = candidate_prior_runner.probability_contract
        parent_probability = parent_prior_runner.probability_contract
        candidate_state = candidate_prior_runner.state_contract
        parent_state = parent_prior_runner.state_contract
        if (
            candidate_state.contract_digest != parent_state.contract_digest
            or candidate_state.state_product_digest
            != input_plan.radar_product_digest
            or candidate_evidence.state_contract_digest
            != candidate_state.contract_digest
            or parent_evidence.state_contract_digest
            != candidate_state.contract_digest
            or candidate_probability.contract_digest
            != parent_probability.contract_digest
            or candidate_probability.contract_digest
            != target_plan.prior_probability_contract_digest
            or candidate_probability.contract_digest
            != case.prior_probability_contract_digest
            or candidate_probability.support_event_digest
            != target_plan.support_event_digest
            or candidate_probability.support_event_digest
            != uncertainty_target.support_event_digest
            or candidate_probability.reflectivity_resolution_dbz
            != target_plan.reflectivity_resolution_dbz
            or candidate_probability.quantization_origin_dbz
            != target_plan.quantization_origin_dbz
            or candidate_probability.threshold_bin_convention
            != target_plan.threshold_bin_convention
            or candidate_evidence.probability_contract_digest
            != candidate_probability.contract_digest
            or parent_evidence.probability_contract_digest
            != candidate_probability.contract_digest
            or candidate_evidence.support_event_digest
            != target_plan.support_event_digest
            or parent_evidence.support_event_digest
            != target_plan.support_event_digest
        ):
            raise ValueError("prior probability event disagrees with its target")
        if (
            candidate_state.contract_digest != case.prior_state_contract_digest
            or candidate_state.contract_digest
            != state_target_plan.state_contract_digest
            or candidate_state.state_product_digest
            != state_target_plan.source_identity_digest
            or candidate_state.state_qc_pipeline_digest
            != state_target_plan.qc_pipeline_digest
            or candidate_state.state_mask_policy_digest
            != state_target_plan.mask_policy_digest
            or candidate_state.state_censor_policy_digest
            != state_target_plan.censor_policy_digest
            or candidate_state.support_threshold_dbz
            != state_target_plan.support_threshold_dbz
        ):
            raise ValueError("prior state contract disagrees with its target")
        if (
            candidate_evidence.prior_output_valid_time
            != target_plan.target_valid_time
            or parent_evidence.prior_output_valid_time
            != candidate_evidence.prior_output_valid_time
            or parent_evidence.feature_source_valid_times
            != candidate_evidence.feature_source_valid_times
            or parent_evidence.feature_source_identity_digests
            != candidate_evidence.feature_source_identity_digests
            or candidate_evidence.feature_exclusion_contract_digest
            != parent_evidence.feature_exclusion_contract_digest
            or candidate_evidence.feature_exclusion_contract_digest
            != target_plan.feature_exclusion_contract_digest
        ):
            raise ValueError("prior uncertainty target time or exclusion disagrees")
        if (
            candidate_evidence.prior_output_valid_time
            != state_target_plan.target_valid_time
            or candidate_evidence.feature_exclusion_contract_digest
            != state_target_plan.feature_exclusion_contract_digest
        ):
            raise ValueError("state calibration target time or exclusion disagrees")
        target_source_seen = target_plan.source_identity_digest in (
            candidate_evidence.feature_source_identity_digests
        )
        candidate_exclusion = candidate_prior_runner.feature_exclusion_mask
        parent_exclusion = parent_prior_runner.feature_exclusion_mask
        if (
            not torch.equal(candidate_exclusion, parent_exclusion)
            or tensor_digest(candidate_exclusion)
            != candidate_evidence.feature_exclusion_mask_digest
            or tensor_digest(parent_exclusion)
            != parent_evidence.feature_exclusion_mask_digest
        ):
            raise ValueError("holdout feature exclusion execution disagrees")
        if target_source_seen:
            if target_plan.target_kind not in (
                "leave_one_time_out",
                "withheld_target_mask",
            ):
                raise ValueError(
                    "prior uncertainty target was visible to the features"
                )
            target_mask = uncertainty_target._valid_mask.to(
                candidate_exclusion.device
            )
            matching_source_times = tuple(
                index
                for index, (valid_time, source_digest) in enumerate(
                    zip(
                        candidate_evidence.feature_source_valid_times,
                        candidate_evidence.feature_source_identity_digests,
                        strict=True,
                    )
                )
                if valid_time == target_plan.target_valid_time
                and source_digest == target_plan.source_identity_digest
            )
            if not matching_source_times or any(
                bool(torch.any(target_mask & ~candidate_exclusion[index]))
                for index in matching_source_times
            ):
                raise ValueError(
                    "prior uncertainty target was visible to the features"
                )
        state_target_source_seen = state_target_plan.source_identity_digest in (
            candidate_evidence.feature_source_identity_digests
        )
        if state_target_source_seen:
            if state_target_plan.target_kind not in (
                "leave_one_time_out",
                "withheld_target_mask",
            ):
                raise ValueError("state calibration target was visible to features")
            state_target_mask = state_calibration_target._valid_mask.to(
                candidate_exclusion.device
            )
            matching_state_times = tuple(
                index
                for index, (valid_time, source_digest) in enumerate(
                    zip(
                        candidate_evidence.feature_source_valid_times,
                        candidate_evidence.feature_source_identity_digests,
                        strict=True,
                    )
                )
                if valid_time == state_target_plan.target_valid_time
                and source_digest == state_target_plan.source_identity_digest
            )
            if not matching_state_times or any(
                bool(torch.any(state_target_mask & ~candidate_exclusion[index]))
                for index in matching_state_times
            ):
                raise ValueError("state calibration target was visible to features")
        prior_reference = uncertainty_target._target_dbz.to(input_frames_dbz)
        prior_valid = (
            uncertainty_target._valid_mask.to(input_frames_dbz.device)
            & torch.isfinite(prior_reference)
            & torch.isfinite(candidate_prior_application.truncated_location_dbz)
            & torch.isfinite(candidate_prior_application.truncated_scale_dbz)
            & (candidate_prior_application.truncated_scale_dbz > 0.0)
            & torch.isfinite(parent_prior_application.truncated_location_dbz)
            & torch.isfinite(parent_prior_application.truncated_scale_dbz)
            & (parent_prior_application.truncated_scale_dbz > 0.0)
        )
        prior_sample_count = int(torch.count_nonzero(prior_valid))
        if prior_sample_count == 0:
            raise ValueError("holdout has no valid prior uncertainty samples")
        support_target = uncertainty_target._echo_support.to(
            candidate_prior_application.event_probability.device
        )
        candidate_scores = _prior_uncertainty_scores(
            candidate_prior_application,
            prior_reference,
            support_target,
            prior_valid,
            support_threshold_dbz=target_plan.support_threshold_dbz,
            reflectivity_resolution_dbz=(
                target_plan.reflectivity_resolution_dbz
            ),
            quantization_origin_dbz=target_plan.quantization_origin_dbz,
            threshold_bin_convention=target_plan.threshold_bin_convention,
        )
        parent_scores = _prior_uncertainty_scores(
            parent_prior_application,
            prior_reference,
            support_target,
            prior_valid,
            support_threshold_dbz=target_plan.support_threshold_dbz,
            reflectivity_resolution_dbz=(
                target_plan.reflectivity_resolution_dbz
            ),
            quantization_origin_dbz=target_plan.quantization_origin_dbz,
            threshold_bin_convention=target_plan.threshold_bin_convention,
        )
        if (
            parent_scores.echo_sample_count
            != candidate_scores.echo_sample_count
            or parent_scores.clear_sample_count
            != candidate_scores.clear_sample_count
        ):
            raise ValueError("candidate and parent hurdle-score domains disagree")
        candidate_valid_fraction = float(
            torch.mean(
                candidate_prior_application.valid_mask.masked_select(prior_valid)
                .to(candidate_prior_application.std_dbz)
            ).detach()
        )
        parent_valid_fraction = float(
            torch.mean(
                parent_prior_application.valid_mask.masked_select(prior_valid)
                .to(parent_prior_application.std_dbz)
            ).detach()
        )
        grid = candidate_forecast.run.grid_time_contract
        if grid is None:
            raise ValueError("holdout prior coverage requires a physical grid")
        echo_area_km2 = (
            candidate_scores.echo_sample_count * grid.cell_area_m2 / 1.0e6
        )
        clear_area_km2 = (
            candidate_scores.clear_sample_count * grid.cell_area_m2 / 1.0e6
        )
        echo_objects = _connected_component_flat_indices(
            prior_valid & support_target.to(prior_valid)
        )
        echo_object_count = len(echo_objects)

        def object_miss_score(application: NeuralPriorApplication) -> float | None:
            if not echo_objects:
                return None
            flat_probability = application.event_probability.flatten()
            object_scores = torch.stack(
                tuple(
                    (1.0 - torch.mean(flat_probability[index])).square()
                    for index in echo_objects
                )
            )
            return float(torch.mean(object_scores).detach())

        candidate_object_miss = object_miss_score(candidate_prior_application)
        parent_object_miss = object_miss_score(parent_prior_application)
        state_reference = state_calibration_target._target_dbz.to(input_frames_dbz)
        state_support_target = state_calibration_target._echo_support.to(
            input_frames_dbz.device
        )
        state_valid = (
            state_calibration_target._valid_mask.to(input_frames_dbz.device)
            & torch.isfinite(state_reference)
            & torch.isfinite(candidate_prior_application.state_background_dbz)
            & torch.isfinite(candidate_prior_application.state_std_dbz)
            & torch.isfinite(parent_prior_application.state_background_dbz)
            & torch.isfinite(parent_prior_application.state_std_dbz)
        )
        if not bool(torch.any(state_valid)):
            raise ValueError("holdout has no valid state calibration samples")
        candidate_state_scores = _state_calibration_scores(
            candidate_prior_application,
            state_reference,
            state_support_target,
            state_valid,
            plan=state_target_plan,
        )
        parent_state_scores = _state_calibration_scores(
            parent_prior_application,
            state_reference,
            state_support_target,
            state_valid,
            plan=state_target_plan,
        )
        state_echo_objects = _connected_component_flat_indices(
            state_valid & state_support_target.to(state_valid)
        )

        def state_object_miss_score(
            application: NeuralPriorApplication,
        ) -> float | None:
            if not state_echo_objects:
                return None
            flat_probability = application.state_support_probability.flatten()
            values = torch.stack(
                tuple(
                    (1.0 - torch.mean(flat_probability[index])).square()
                    for index in state_echo_objects
                )
            )
            return float(torch.mean(values).detach())

        candidate_state_object_miss = state_object_miss_score(
            candidate_prior_application
        )
        parent_state_object_miss = state_object_miss_score(
            parent_prior_application
        )
        candidate_valid_area_km2 = (
            float(
                torch.count_nonzero(
                    candidate_prior_application.valid_mask & prior_valid
                )
            )
            * grid.cell_area_m2
            / 1.0e6
        )
        abstention_increase = parent_valid_fraction - candidate_valid_fraction
        for run, training_digest in (
            (candidate_forecast.run, manifest.candidate_training_manifest_digest),
            (parent_forecast.run, manifest.parent_training_manifest_digest),
        ):
            if (
                run.prior_model_contract_digest != manifest.model_contract_digest
                or run.prior_feature_schema_digest != manifest.feature_schema_digest
                or run.prior_training_manifest_digest != training_digest
            ):
                raise ValueError("holdout forecast prior contract disagrees")
        run_identities = (
            "analysis_config_digest",
            "operational_calibration_manifest_digest",
            "operational_data_identity_digest",
        )
        if candidate_forecast.run.config.digest != parent_forecast.run.config.digest or any(
            getattr(candidate_forecast.run, name) != getattr(parent_forecast.run, name)
            for name in run_identities
        ):
            raise ValueError("candidate and parent holdout run profiles disagree")
        resolved_candidate = _resolve_verification(
            verification, candidate_forecast, metric_config
        )
        resolved_parent = _resolve_verification(
            verification, parent_forecast, metric_config
        )
        if resolved_candidate.content_digest != resolved_parent.content_digest:
            raise ValueError("candidate and parent verification disagree")
        if resolved_candidate.content_digest != case.verification_bundle_digest:
            raise ValueError("verification content disagrees with the holdout result")
        if (
            resolved_candidate.valid_times is None
            or resolved_candidate.grid_contract_digest is None
            or resolved_candidate.radar_product_digest is None
            or resolved_candidate.qc_pipeline_digest is None
        ):
            raise ValueError("holdout verification lineage must be complete")
        if verification_plan_digest(
            valid_times=resolved_candidate.valid_times,
            grid_contract_digest=resolved_candidate.grid_contract_digest,
            radar_product_digest=resolved_candidate.radar_product_digest,
            qc_pipeline_digest=resolved_candidate.qc_pipeline_digest,
        ) != case.verification_plan_digest:
            raise ValueError("verification identity is not pre-registered")
        if metric_config.digest != case.metric_contract_digest:
            raise ValueError("metric contract is not registered by the holdout plan")
        leads = metric_config.full_map_lead_minutes
        candidate_weights = _resolved_forecast_domain_weights(
            candidate_forecast, resolved_candidate, leads, metric_config
        )
        parent_weights = _resolved_forecast_domain_weights(
            parent_forecast, resolved_parent, leads, metric_config
        )
        # The parent domain is the fixed paired reference.  Using the
        # intersection would let a candidate hide difficult withdrawn cells.
        common_weights = parent_weights
        candidate_common, candidate_available = _resolved_forecast_scores(
            candidate_forecast,
            candidate_forecast.state,
            resolved_candidate,
            leads,
            metric_config,
            domain_weights=common_weights,
        )
        parent_common, parent_available = _resolved_forecast_scores(
            parent_forecast,
            parent_forecast.state,
            resolved_parent,
            leads,
            metric_config,
            domain_weights=common_weights,
        )
        if not torch.equal(candidate_available, parent_available):
            raise ValueError("candidate and parent metric availability disagree")
        available = candidate_available
        candidate_native, candidate_native_available = _resolved_forecast_scores(
            candidate_forecast,
            candidate_forecast.state,
            resolved_candidate,
            leads,
            metric_config,
            domain_weights=candidate_weights,
        )
        parent_native, parent_native_available = _resolved_forecast_scores(
            parent_forecast,
            parent_forecast.state,
            resolved_parent,
            leads,
            metric_config,
            domain_weights=parent_weights,
        )
        if not torch.equal(candidate_native_available, parent_native_available) or (
            not torch.equal(candidate_native_available, available)
        ):
            raise ValueError("native and common metric availability disagree")
        change = candidate_common - parent_common
        candidate_policy = candidate_native - candidate_common
        parent_policy = parent_native - parent_common
        end_to_end = candidate_native - parent_native
        candidate_coverage = _forecast_coverage(
            candidate_forecast, resolved_candidate, leads, metric_config
        )
        parent_coverage = _forecast_coverage(
            parent_forecast, resolved_parent, leads, metric_config
        )
        finite = resolved_candidate.valid_mask & torch.isfinite(
            resolved_candidate.frames_dbz
        )
        denominators = torch.stack(
            [
                torch.count_nonzero(
                    finite[minutes // candidate_forecast.run.config.interval_minutes - 1]
                ).clamp_min(1)
                for minutes in leads
            ]
        ).to(common_weights)
        common_coverage = torch.count_nonzero(
            common_weights > 0, dim=(-2, -1)
        ).to(common_weights) / denominators
        candidate_support = candidate_weights > 0
        parent_support = parent_weights > 0
        newly_issued = torch.count_nonzero(
            candidate_support & ~parent_support, dim=(-2, -1)
        ).to(common_weights) / denominators
        withdrawn = torch.count_nonzero(
            parent_support & ~candidate_support, dim=(-2, -1)
        ).to(common_weights) / denominators
        range_band_evaluations: list[RangeBandEvaluation] = []
        for range_regime in range_contract.reference_active_range_regimes:
            range_mask = range_band_masks[range_regime]
            lead_mask = range_mask.unsqueeze(0).expand_as(common_weights)
            band_common_weights = common_weights * lead_mask.to(common_weights)
            band_candidate_common, band_candidate_available = (
                _resolved_forecast_scores(
                    candidate_forecast,
                    candidate_forecast.state,
                    resolved_candidate,
                    leads,
                    metric_config,
                    domain_weights=band_common_weights,
                )
            )
            band_parent_common, band_parent_available = _resolved_forecast_scores(
                parent_forecast,
                parent_forecast.state,
                resolved_parent,
                leads,
                metric_config,
                domain_weights=band_common_weights,
            )
            band_candidate_native, band_candidate_native_available = (
                _resolved_forecast_scores(
                    candidate_forecast,
                    candidate_forecast.state,
                    resolved_candidate,
                    leads,
                    metric_config,
                    domain_weights=candidate_weights
                    * lead_mask.to(candidate_weights),
                )
            )
            band_parent_native, band_parent_native_available = (
                _resolved_forecast_scores(
                    parent_forecast,
                    parent_forecast.state,
                    resolved_parent,
                    leads,
                    metric_config,
                    domain_weights=parent_weights * lead_mask.to(parent_weights),
                )
            )
            if not (
                torch.equal(band_candidate_available, band_parent_available)
                and torch.equal(
                    band_candidate_available, band_candidate_native_available
                )
                and torch.equal(
                    band_candidate_available, band_parent_native_available
                )
                and bool(torch.any(band_candidate_available))
            ):
                raise ValueError("range-band paired metric domain is unavailable")
            band_prior_valid = prior_valid & range_mask.to(prior_valid.device)
            band_state_valid = state_valid & range_mask.to(state_valid.device)
            if not bool(torch.any(band_prior_valid)) or not bool(
                torch.any(band_state_valid)
            ):
                raise ValueError("range-band calibration domain is empty")
            band_candidate_prior = _prior_uncertainty_scores(
                candidate_prior_application,
                prior_reference,
                support_target,
                band_prior_valid,
                support_threshold_dbz=target_plan.support_threshold_dbz,
                reflectivity_resolution_dbz=target_plan.reflectivity_resolution_dbz,
                quantization_origin_dbz=target_plan.quantization_origin_dbz,
                threshold_bin_convention=target_plan.threshold_bin_convention,
            )
            band_parent_prior = _prior_uncertainty_scores(
                parent_prior_application,
                prior_reference,
                support_target,
                band_prior_valid,
                support_threshold_dbz=target_plan.support_threshold_dbz,
                reflectivity_resolution_dbz=target_plan.reflectivity_resolution_dbz,
                quantization_origin_dbz=target_plan.quantization_origin_dbz,
                threshold_bin_convention=target_plan.threshold_bin_convention,
            )
            band_candidate_state = _state_calibration_scores(
                candidate_prior_application,
                state_reference,
                state_support_target,
                band_state_valid,
                plan=state_target_plan,
            )
            band_parent_state = _state_calibration_scores(
                parent_prior_application,
                state_reference,
                state_support_target,
                band_state_valid,
                plan=state_target_plan,
            )

            def masked_object_miss(
                probability: Tensor,
                mask: Tensor,
                target: Tensor,
            ) -> tuple[float | None, int]:
                objects = _connected_component_flat_indices(mask & target.to(mask))
                if not objects:
                    return None, 0
                flat = probability.flatten()
                return (
                    float(
                        torch.mean(
                            torch.stack(
                                tuple(
                                    (1.0 - torch.mean(flat[index])).square()
                                    for index in objects
                                )
                            )
                        ).detach()
                    ),
                    len(objects),
                )

            candidate_band_object_miss, candidate_band_object_count = masked_object_miss(
                candidate_prior_application.event_probability,
                band_prior_valid,
                support_target,
            )
            parent_band_object_miss, parent_band_object_count = masked_object_miss(
                parent_prior_application.event_probability,
                band_prior_valid,
                support_target,
            )
            (
                candidate_band_state_object_miss,
                candidate_band_state_object_count,
            ) = masked_object_miss(
                candidate_prior_application.state_support_probability,
                band_state_valid,
                state_support_target,
            )
            (
                parent_band_state_object_miss,
                parent_band_state_object_count,
            ) = masked_object_miss(
                parent_prior_application.state_support_probability,
                band_state_valid,
                state_support_target,
            )
            (
                candidate_component_scores,
                parent_component_scores,
                component_differences,
                component_counts,
            ) = (
                _range_uncertainty_components(
                    band_candidate_prior,
                    band_parent_prior,
                    band_candidate_state,
                    band_parent_state,
                    candidate_object_miss=candidate_band_object_miss,
                    parent_object_miss=parent_band_object_miss,
                    object_count=candidate_band_object_count,
                    candidate_state_object_miss=(
                        candidate_band_state_object_miss
                    ),
                    parent_state_object_miss=parent_band_state_object_miss,
                    state_object_count=candidate_band_state_object_count,
                )
            )
            if (
                candidate_band_object_count != parent_band_object_count
                or candidate_band_state_object_count
                != parent_band_state_object_count
            ):
                raise ValueError("range-band object domains disagree")
            cell_area_km2 = grid.cell_area_m2 / 1.0e6
            forecast_indices = tuple(
                minutes // candidate_forecast.run.config.interval_minutes - 1
                for minutes in leads
            )
            operational_domain = lead_mask.to(candidate_forecast.valid_mask.device)
            band_domain_counts = torch.count_nonzero(
                operational_domain,
                dim=(-2, -1),
            )
            band_domain_denominators = band_domain_counts.to(common_weights)
            band_parent_issued = torch.stack(
                tuple(
                    parent_forecast.valid_mask[index]
                    & range_mask.to(parent_forecast.valid_mask.device)
                    for index in forecast_indices
                )
            )
            band_candidate_issued = torch.stack(
                tuple(
                    candidate_forecast.valid_mask[index]
                    & range_mask.to(candidate_forecast.valid_mask.device)
                    for index in forecast_indices
                )
            )
            band_parent_issued_counts = torch.count_nonzero(
                band_parent_issued,
                dim=(-2, -1),
            )
            band_candidate_issued_counts = torch.count_nonzero(
                band_candidate_issued,
                dim=(-2, -1),
            )
            band_withdrawn_counts = torch.count_nonzero(
                band_parent_issued & ~band_candidate_issued,
                dim=(-2, -1),
            )
            band_newly_issued_counts = torch.count_nonzero(
                band_candidate_issued & ~band_parent_issued,
                dim=(-2, -1),
            )
            band_withdrawn = band_withdrawn_counts.to(common_weights) / (
                band_parent_issued_counts.clamp_min(1).to(common_weights)
            )
            band_newly_issued = band_newly_issued_counts.to(common_weights) / (
                band_domain_denominators
            )
            candidate_fallback_mask = torch.stack(
                tuple(
                    candidate_forecast.background_fallback_mask[index]
                    & range_mask.to(candidate_forecast.valid_mask.device)
                    for index in forecast_indices
                )
            )
            parent_fallback_mask = torch.stack(
                tuple(
                    parent_forecast.background_fallback_mask[index]
                    & range_mask.to(parent_forecast.valid_mask.device)
                    for index in forecast_indices
                )
            )
            candidate_band_fallback_counts = torch.count_nonzero(
                candidate_fallback_mask,
                dim=(-2, -1),
            )
            parent_band_fallback_counts = torch.count_nonzero(
                parent_fallback_mask,
                dim=(-2, -1),
            )
            candidate_band_fallback = (
                candidate_band_fallback_counts.to(common_weights)
                / band_domain_denominators
            )
            parent_band_fallback = (
                parent_band_fallback_counts.to(common_weights)
                / band_domain_denominators
            )
            candidate_confidence = candidate_forecast.forecast_confidence[
                list(forecast_indices)
            ]
            parent_confidence = parent_forecast.forecast_confidence[
                list(forecast_indices)
            ]
            candidate_confidence_area = torch.sum(
                candidate_confidence * band_candidate_issued.to(candidate_confidence),
                dim=(-2, -1),
            ) * cell_area_km2
            parent_confidence_area = torch.sum(
                parent_confidence * band_parent_issued.to(parent_confidence),
                dim=(-2, -1),
            ) * cell_area_km2
            band_domain_area = band_domain_denominators * cell_area_km2
            candidate_band_confidence = candidate_confidence_area / band_domain_area
            parent_band_confidence = parent_confidence_area / band_domain_area
            issuance_domain_digest = json_digest(
                {
                    "contract": "range-band-operational-issuance-domain-v1",
                    "range_band_mask_digest": range_contract.mask_digest(
                        range_regime
                    ),
                    "range_geometry_contract_digest": (
                        range_contract.range_geometry_contract_digest
                    ),
                    "forecast_frame_shape": list(
                        candidate_forecast.valid_mask.shape[-2:]
                    ),
                    "lead_minutes": list(leads),
                }
            )
            range_band_evaluations.append(
                RangeBandEvaluation(
                    range_regime=range_regime,
                    range_band_mask_digest=range_contract.mask_digest(range_regime),
                    range_geometry_contract_digest=(
                        range_contract.range_geometry_contract_digest
                    ),
                    metric_change=band_candidate_common - band_parent_common,
                    end_to_end_metric_change=(
                        band_candidate_native - band_parent_native
                    ),
                    metric_available=band_candidate_available,
                    candidate_uncertainty_component_scores=(
                        candidate_component_scores
                    ),
                    parent_uncertainty_component_scores=(
                        parent_component_scores
                    ),
                    uncertainty_component_differences=component_differences,
                    uncertainty_component_sample_counts=component_counts,
                    evaluated_area_km2=(
                        float(torch.count_nonzero(range_mask))
                        * cell_area_km2
                    ),
                    metric_valid_area_km2_by_lead=tuple(
                        float(torch.count_nonzero(weights > 0)) * cell_area_km2
                        for weights in band_common_weights
                    ),
                    metric_valid_area_km2=torch.stack(
                        tuple(
                            band_candidate_available[index].to(torch.float64)
                            * (
                                float(torch.count_nonzero(weights > 0))
                                * cell_area_km2
                            )
                            for index, weights in enumerate(band_common_weights)
                        )
                    ),
                    issuance_domain_digest=issuance_domain_digest,
                    issuance_domain_cell_count_by_lead=tuple(
                        int(value) for value in band_domain_counts
                    ),
                    issuance_domain_area_km2_by_lead=tuple(
                        float(value) for value in band_domain_area
                    ),
                    parent_issued_count_by_lead=tuple(
                        int(value) for value in band_parent_issued_counts
                    ),
                    candidate_issued_count_by_lead=tuple(
                        int(value) for value in band_candidate_issued_counts
                    ),
                    withdrawn_count_by_lead=tuple(
                        int(value) for value in band_withdrawn_counts
                    ),
                    newly_issued_count_by_lead=tuple(
                        int(value) for value in band_newly_issued_counts
                    ),
                    parent_fallback_count_by_lead=tuple(
                        int(value) for value in parent_band_fallback_counts
                    ),
                    candidate_fallback_count_by_lead=tuple(
                        int(value) for value in candidate_band_fallback_counts
                    ),
                    parent_confidence_weighted_issued_area_by_lead=tuple(
                        float(value) for value in parent_confidence_area
                    ),
                    candidate_confidence_weighted_issued_area_by_lead=tuple(
                        float(value) for value in candidate_confidence_area
                    ),
                    withdrawn_fraction_by_lead=band_withdrawn,
                    newly_issued_fraction_by_lead=band_newly_issued,
                    background_fallback_increase_by_lead=(
                        candidate_band_fallback - parent_band_fallback
                    ),
                    confidence_weighted_coverage_change_by_lead=(
                        candidate_band_confidence - parent_band_confidence
                    ),
                    probability_valid_area_km2=(
                        float(torch.count_nonzero(band_prior_valid))
                        * cell_area_km2
                    ),
                    state_valid_area_km2=(
                        float(torch.count_nonzero(band_state_valid))
                        * cell_area_km2
                    ),
                    echo_pixel_count=band_candidate_prior.echo_sample_count,
                    clear_pixel_count=band_candidate_prior.clear_sample_count,
                    echo_object_count=candidate_band_object_count,
                    state_echo_pixel_count=(
                        band_candidate_state.echo_sample_count
                    ),
                    state_clear_pixel_count=(
                        band_candidate_state.clear_sample_count
                    ),
                    state_echo_object_count=(
                        candidate_band_state_object_count
                    ),
                )
            )
        issue_time = candidate_forecast.run.grid_time_contract.valid_times[-1]
        parent_issue = parent_forecast.run.grid_time_contract
        if parent_issue is None or parent_issue.valid_times[-1] != issue_time:
            raise ValueError("candidate and parent issue times disagree")
        if issue_time != case.issue_time:
            raise ValueError("holdout issue time is not pre-registered")
        reference_ranges = set(range_contract.reference_active_range_regimes)
        classified_ranges = set(
            regime_classification_evidence.active_range_regimes
        )
        range_intersection = reference_ranges & classified_ranges
        range_precision = (
            len(range_intersection) / len(classified_ranges)
            if classified_ranges
            else (1.0 if not reference_ranges else 0.0)
        )
        range_recall = (
            len(range_intersection) / len(reference_ranges)
            if reference_ranges
            else (1.0 if not classified_ranges else 0.0)
        )
        false_active_fraction = (
            len(classified_ranges - reference_ranges) / len(classified_ranges)
            if classified_ranges
            else 0.0
        )
        exact_range_set = classified_ranges == reference_ranges
        weather_agreement = (
            regime_classification_evidence.regime == case.regime
        )
        return _new_prior_holdout_evaluation(
            holdout_plan_digest=plan.plan_digest,
            candidate_manifest_digest=manifest.manifest_digest,
            candidate_prior_digest=manifest.candidate_prior_digest,
            parent_prior_digest=manifest.parent_prior_digest,
            case_id=case.case_id,
            storm_id=case.storm_id,
            physical_event_digest=case.physical_event_digest,
            day=case.day,
            radar_id=case.radar_id,
            regime=case.regime,
            range_regime=case.range_regime,
            reference_active_range_regimes=(
                range_contract.reference_active_range_regimes
            ),
            range_band_contract_digest=range_contract.contract_digest,
            range_band_evaluations=tuple(range_band_evaluations),
            regime_classifier_digest=(
                regime_classification_evidence.classifier_digest
            ),
            regime_classifier_manifest_digest=(
                regime_classifier_manifest.manifest_digest
            ),
            regime_classification_evidence_digest=(
                regime_classification_evidence.evidence_digest
            ),
            classified_regime=regime_classification_evidence.regime,
            classified_range_regimes=(
                regime_classification_evidence.active_range_regimes
            ),
            classifier_regime_confidence=(
                regime_classification_evidence.regime_confidence
            ),
            classifier_range_confidence=(
                regime_classification_evidence.range_regime_confidence
            ),
            classifier_regime_entropy=(
                regime_classification_evidence.regime_entropy
            ),
            classifier_is_ood=regime_classification_evidence.is_ood,
            classifier_reference_agreement=(
                not regime_classification_evidence.is_ood
                and weather_agreement
                and exact_range_set
            ),
            classifier_weather_reference_agreement=weather_agreement,
            classifier_range_set_precision=range_precision,
            classifier_range_set_recall=range_recall,
            classifier_range_exact_set_match=exact_range_set,
            classifier_false_active_band_fraction=false_active_fraction,
            classifier_reference_range_is_ood=not reference_ranges,
            classifier_numerical_runtime_digest=(
                regime_classification_evidence.numerical_runtime_digest
            ),
            classifier_input_dtype=regime_classification_evidence.input_dtype,
            classifier_input_device=regime_classification_evidence.input_device,
            classifier_weather_top1_top2_gap=(
                regime_classification_evidence.weather_top1_top2_gap
            ),
            classifier_minimum_range_presence_margin=(
                regime_classification_evidence.minimum_range_presence_margin
            ),
            candidate_forecast_digest=candidate_digest,
            parent_forecast_digest=parent_digest,
            candidate_prior_application_digest=(
                candidate_prior_application.application_digest
            ),
            parent_prior_application_digest=(
                parent_prior_application.application_digest
            ),
            candidate_inference_evidence_digest=(
                candidate_prior_application.inference_evidence.evidence_digest
            ),
            parent_inference_evidence_digest=(
                parent_prior_application.inference_evidence.evidence_digest
            ),
            metric_change=change,
            candidate_issuance_effect=candidate_policy,
            parent_issuance_effect=parent_policy,
            end_to_end_metric_change=end_to_end,
            metric_available=available,
            lead_minutes=leads,
            metric_names=metric_config.metric_names,
            verification_digest=resolved_candidate.content_digest,
            metric_contract_digest=metric_config.digest,
            coverage_candidate=candidate_coverage,
            coverage_parent=parent_coverage,
            coverage_common=common_coverage,
            newly_issued_fraction=newly_issued,
            withdrawn_fraction=withdrawn,
            prior_conditional_pit_residual_mean_abs=(
                candidate_scores.conditional_pit_residual_mean_abs
            ),
            prior_conditional_underdispersion_fraction=(
                candidate_scores.conditional_underdispersion_fraction
            ),
            prior_echo_intensity_nll=candidate_scores.echo_intensity_nll,
            prior_support_brier_score=candidate_scores.support_brier_score,
            prior_echo_support_miss_score=(
                candidate_scores.echo_support_miss_score
            ),
            prior_echo_object_miss_score=candidate_object_miss,
            prior_clear_sky_false_echo_score=(
                candidate_scores.clear_sky_false_echo_score
            ),
            parent_prior_conditional_pit_residual_mean_abs=(
                parent_scores.conditional_pit_residual_mean_abs
            ),
            parent_prior_conditional_underdispersion_fraction=(
                parent_scores.conditional_underdispersion_fraction
            ),
            parent_prior_echo_intensity_nll=parent_scores.echo_intensity_nll,
            parent_prior_support_brier_score=parent_scores.support_brier_score,
            parent_prior_echo_support_miss_score=(
                parent_scores.echo_support_miss_score
            ),
            parent_prior_echo_object_miss_score=parent_object_miss,
            parent_prior_clear_sky_false_echo_score=(
                parent_scores.clear_sky_false_echo_score
            ),
            prior_echo_intensity_status=(
                "available"
                if candidate_scores.echo_sample_count
                else "not_applicable"
            ),
            prior_clear_sky_status=(
                "available"
                if candidate_scores.clear_sample_count
                else "not_applicable"
            ),
            prior_candidate_valid_fraction=candidate_valid_fraction,
            prior_parent_valid_fraction=parent_valid_fraction,
            prior_candidate_valid_area_km2=candidate_valid_area_km2,
            prior_abstention_increase_vs_parent=abstention_increase,
            prior_uncertainty_target_digest=uncertainty_target.target_digest,
            prior_uncertainty_sample_count=prior_sample_count,
            prior_echo_intensity_sample_count=(
                candidate_scores.echo_sample_count
            ),
            prior_clear_sky_sample_count=candidate_scores.clear_sample_count,
            prior_echo_area_km2=echo_area_km2,
            prior_clear_sky_area_km2=clear_area_km2,
            prior_echo_object_count=echo_object_count,
            state_candidate_gaussian_nll=candidate_state_scores.gaussian_nll,
            state_parent_gaussian_nll=parent_state_scores.gaussian_nll,
            state_candidate_pit_residual_mean_abs=(
                candidate_state_scores.pit_residual_mean_abs
            ),
            state_parent_pit_residual_mean_abs=(
                parent_state_scores.pit_residual_mean_abs
            ),
            state_candidate_underdispersion_fraction=(
                candidate_state_scores.underdispersion_fraction
            ),
            state_parent_underdispersion_fraction=(
                parent_state_scores.underdispersion_fraction
            ),
            state_candidate_support_brier_score=(
                candidate_state_scores.support_brier_score
            ),
            state_parent_support_brier_score=(
                parent_state_scores.support_brier_score
            ),
            state_candidate_echo_support_miss_score=(
                candidate_state_scores.echo_support_miss_score
            ),
            state_parent_echo_support_miss_score=(
                parent_state_scores.echo_support_miss_score
            ),
            state_candidate_echo_object_miss_score=(
                candidate_state_object_miss
            ),
            state_parent_echo_object_miss_score=parent_state_object_miss,
            state_candidate_false_support_score=(
                candidate_state_scores.false_support_score
            ),
            state_parent_false_support_score=(
                parent_state_scores.false_support_score
            ),
            state_candidate_valid_brier_score=(
                candidate_state_scores.valid_brier_score
            ),
            state_parent_valid_brier_score=(
                parent_state_scores.valid_brier_score
            ),
            state_calibration_target_digest=(
                state_calibration_target.target_digest
            ),
            state_calibration_sample_count=candidate_state_scores.sample_count,
            state_calibration_echo_sample_count=(
                candidate_state_scores.echo_sample_count
            ),
            state_calibration_clear_sample_count=(
                candidate_state_scores.clear_sample_count
            ),
            state_calibration_echo_object_count=len(state_echo_objects),
            issue_time=issue_time,
            verification_valid_times=verification.valid_times,
        )


@dataclass(frozen=True)
class _PriorUncertaintyScores:
    conditional_pit_residual_mean_abs: float | None
    conditional_underdispersion_fraction: float | None
    echo_intensity_nll: float | None
    support_brier_score: float
    echo_support_miss_score: float | None
    clear_sky_false_echo_score: float | None
    echo_sample_count: int
    clear_sample_count: int


def _quantized_bin_bounds(
    reference_dbz: Tensor,
    *,
    reflectivity_resolution_dbz: float,
    quantization_origin_dbz: float,
    support_threshold_dbz: float,
    threshold_bin_convention: str,
) -> tuple[Tensor, Tensor]:
    """Return disjoint latent intervals for quantized, threshold-censored dBZ."""

    if (
        not reference_dbz.is_floating_point()
        or not bool(torch.all(torch.isfinite(reference_dbz)))
        or not math.isfinite(reflectivity_resolution_dbz)
        or reflectivity_resolution_dbz <= 0.0
        or not math.isfinite(quantization_origin_dbz)
        or not math.isfinite(support_threshold_dbz)
        or threshold_bin_convention != "nearest_rounding_threshold_censor"
    ):
        raise ValueError("quantized dBZ bin contract is invalid")
    reference = reference_dbz.to(torch.float64)
    width = torch.as_tensor(
        reflectivity_resolution_dbz,
        dtype=torch.float64,
        device=reference.device,
    )
    threshold = torch.as_tensor(
        support_threshold_dbz,
        dtype=torch.float64,
        device=reference.device,
    )
    origin = torch.as_tensor(
        quantization_origin_dbz,
        dtype=torch.float64,
        device=reference.device,
    )
    threshold_index = torch.round((threshold - origin) / width)
    threshold_lattice_value = origin + threshold_index * width
    threshold_tolerance = max(
        1.0e-7,
        math.ulp(1.0)
        * max(abs(float(threshold)), abs(float(threshold_lattice_value)), 1.0)
        * 32.0,
    )
    if abs(float(threshold - threshold_lattice_value)) > threshold_tolerance:
        raise ValueError("support threshold is off its declared dBZ lattice")
    lattice_index = torch.round((reference - origin) / width)
    lattice_value = origin + lattice_index * width
    source_epsilon = torch.finfo(reference_dbz.dtype).eps
    tolerance = torch.maximum(
        torch.full_like(reference, 1.0e-7),
        source_epsilon
        * torch.maximum(reference.abs(), lattice_value.abs()).clamp_min(1.0)
        * 32.0,
    )
    if bool(torch.any(torch.abs(reference - lattice_value) > tolerance)):
        raise ValueError("quantized dBZ value is off its declared lattice")
    is_threshold_bin = torch.abs(reference - threshold) <= tolerance
    bin_lower = torch.where(
        is_threshold_bin,
        threshold,
        lattice_value - 0.5 * width,
    )
    bin_upper = torch.where(
        is_threshold_bin,
        threshold + 0.5 * width,
        lattice_value + 0.5 * width,
    )
    return bin_lower, bin_upper


def _quantized_gaussian_diagnostics(
    location_dbz: Tensor,
    scale_dbz: Tensor,
    reference_dbz: Tensor,
    *,
    reflectivity_resolution_dbz: float,
    quantization_origin_dbz: float,
    support_threshold_dbz: float,
    threshold_bin_convention: str,
) -> tuple[Tensor, Tensor]:
    """Gaussian interval NLL and midpoint PIT for quantized state dBZ."""

    location = location_dbz.to(torch.float64)
    scale = scale_dbz.to(torch.float64)
    reference = reference_dbz.to(torch.float64)
    if (
        not bool(torch.all(torch.isfinite(location)))
        or not bool(torch.all(torch.isfinite(scale) & (scale > 0.0)))
        or not bool(torch.all(torch.isfinite(reference)))
    ):
        raise ValueError("quantized Gaussian inputs are invalid")
    bin_lower, bin_upper = _quantized_bin_bounds(
        reference_dbz,
        reflectivity_resolution_dbz=reflectivity_resolution_dbz,
        quantization_origin_dbz=quantization_origin_dbz,
        support_threshold_dbz=support_threshold_dbz,
        threshold_bin_convention=threshold_bin_convention,
    )
    lower_z = (bin_lower - location) / scale
    upper_z = (bin_upper - location) / scale
    log_mass = _standard_normal_log_interval_mass(lower_z, upper_z)
    nll = -log_mass
    midpoint = 0.5 * (
        torch.special.ndtr(lower_z) + torch.special.ndtr(upper_z)
    )
    epsilon = torch.finfo(torch.float64).eps
    pit = torch.special.ndtri(midpoint.clamp(epsilon, 1.0 - epsilon))
    if not bool(torch.all(torch.isfinite(nll))) or not bool(
        torch.all(torch.isfinite(pit))
    ):
        raise ValueError("quantized Gaussian score is not finite")
    return nll, pit


def _logdiffexp(log_larger: Tensor, log_smaller: Tensor) -> Tensor:
    """Stable log(exp(a) - exp(b)) for a >= b."""

    ratio = torch.minimum(
        log_smaller - log_larger,
        torch.zeros((), dtype=log_larger.dtype, device=log_larger.device),
    )
    return log_larger + torch.log(-torch.expm1(ratio))


def _standard_normal_log_interval_mass(lower: Tensor, upper: Tensor) -> Tensor:
    """Stable normal interval mass in either CDF tail."""

    if lower.shape != upper.shape or bool(torch.any(upper <= lower)):
        raise ValueError("normal interval bounds are invalid")
    cdf_mass = _logdiffexp(
        torch.special.log_ndtr(upper),
        torch.special.log_ndtr(lower),
    )
    survival_mass = _logdiffexp(
        torch.special.log_ndtr(-lower),
        torch.special.log_ndtr(-upper),
    )
    return torch.where(lower >= 0.0, survival_mass, cdf_mass)


@dataclass(frozen=True)
class _StateCalibrationScores:
    gaussian_nll: float
    pit_residual_mean_abs: float
    underdispersion_fraction: float
    support_brier_score: float
    echo_support_miss_score: float | None
    false_support_score: float | None
    valid_brier_score: float
    sample_count: int
    echo_sample_count: int
    clear_sample_count: int


def _state_calibration_scores(
    application: NeuralPriorApplication,
    reference_dbz: Tensor,
    support_target: Tensor,
    evaluation_mask: Tensor,
    *,
    plan: NeuralPriorStateCalibrationPlan,
) -> _StateCalibrationScores:
    state_reference = reference_dbz.masked_select(evaluation_mask)
    state_location = application.state_background_dbz.masked_select(evaluation_mask)
    state_scale = application.state_std_dbz.masked_select(evaluation_mask)
    nll, pit = _quantized_gaussian_diagnostics(
        state_location,
        state_scale,
        state_reference,
        reflectivity_resolution_dbz=plan.reflectivity_resolution_dbz,
        quantization_origin_dbz=plan.quantization_origin_dbz,
        support_threshold_dbz=plan.support_threshold_dbz,
        threshold_bin_convention=plan.threshold_bin_convention,
    )
    absolute = torch.abs(pit)
    target = support_target.to(application.state_support_probability)
    support = application.state_support_probability.masked_select(evaluation_mask)
    target_values = target.masked_select(evaluation_mask)
    echo_mask = evaluation_mask & support_target.to(evaluation_mask)
    clear_mask = evaluation_mask & ~support_target.to(evaluation_mask)
    echo_count = int(torch.count_nonzero(echo_mask))
    clear_count = int(torch.count_nonzero(clear_mask))
    valid_probability = application.state_valid_probability.masked_select(
        evaluation_mask
    )
    return _StateCalibrationScores(
        gaussian_nll=float(torch.mean(nll).detach()),
        pit_residual_mean_abs=float(torch.mean(absolute).detach()),
        underdispersion_fraction=float(
            torch.mean((absolute > 2.0).to(absolute)).detach()
        ),
        support_brier_score=float(
            torch.mean((support - target_values).square()).detach()
        ),
        echo_support_miss_score=(
            None
            if echo_count == 0
            else float(
                torch.mean(
                    (
                        1.0
                        - application.state_support_probability.masked_select(
                            echo_mask
                        )
                    ).square()
                ).detach()
            )
        ),
        false_support_score=(
            None
            if clear_count == 0
            else float(
                torch.mean(
                    application.state_support_probability.masked_select(
                        clear_mask
                    ).square()
                ).detach()
            )
        ),
        valid_brier_score=float(
            torch.mean((1.0 - valid_probability).square()).detach()
        ),
        sample_count=int(torch.count_nonzero(evaluation_mask)),
        echo_sample_count=echo_count,
        clear_sample_count=clear_count,
    )


def _truncated_gaussian_diagnostics(
    location_dbz: Tensor,
    scale_dbz: Tensor,
    reference_dbz: Tensor,
    *,
    support_threshold_dbz: float,
    reflectivity_resolution_dbz: float = 0.5,
    quantization_origin_dbz: float = -10.0,
    threshold_bin_convention: str = "nearest_rounding_threshold_censor",
) -> tuple[Tensor, Tensor]:
    """Interval NLL and midpoint PIT for quantized lower-truncated dBZ."""

    location = location_dbz.to(torch.float64)
    scale = scale_dbz.to(torch.float64)
    reference = reference_dbz.to(torch.float64)
    if (
        not bool(torch.all(torch.isfinite(location)))
        or not bool(torch.all(torch.isfinite(scale) & (scale > 0.0)))
        or not bool(torch.all(torch.isfinite(reference)))
        or bool(torch.any(reference < support_threshold_dbz))
    ):
        raise ValueError("truncated-Gaussian inputs violate their support")
    threshold = torch.as_tensor(
        support_threshold_dbz,
        dtype=torch.float64,
        device=reference.device,
    )
    bin_lower, bin_upper = _quantized_bin_bounds(
        reference_dbz,
        reflectivity_resolution_dbz=reflectivity_resolution_dbz,
        quantization_origin_dbz=quantization_origin_dbz,
        support_threshold_dbz=support_threshold_dbz,
        threshold_bin_convention=threshold_bin_convention,
    )
    bin_lower = torch.maximum(bin_lower, threshold)
    if bool(torch.any(bin_upper <= bin_lower)):
        raise ValueError("quantized truncated-Gaussian bin is empty")
    truncation = (threshold - location) / scale
    standardized_lower = (bin_lower - location) / scale
    standardized_upper = (bin_upper - location) / scale
    log_survival_truncation = torch.special.log_ndtr(-truncation)
    log_survival_lower = torch.special.log_ndtr(-standardized_lower)
    log_survival_upper = torch.special.log_ndtr(-standardized_upper)
    log_ratio = torch.minimum(
        log_survival_upper - log_survival_lower,
        torch.zeros((), dtype=torch.float64, device=location.device),
    )
    log_interval_mass = log_survival_lower + torch.log(-torch.expm1(log_ratio))
    nll = -(log_interval_mass - log_survival_truncation)
    lower_tail = torch.exp(
        torch.minimum(
            log_survival_lower - log_survival_truncation,
            torch.zeros((), dtype=torch.float64, device=location.device),
        )
    )
    upper_tail = torch.exp(
        torch.minimum(
            log_survival_upper - log_survival_truncation,
            torch.zeros((), dtype=torch.float64, device=location.device),
        )
    )
    conditional_cdf = 1.0 - 0.5 * (lower_tail + upper_tail)
    epsilon = torch.finfo(torch.float64).eps
    conditional_cdf = conditional_cdf.clamp(epsilon, 1.0 - epsilon)
    pit_residual = torch.special.ndtri(conditional_cdf)
    if not bool(torch.all(torch.isfinite(nll))) or not bool(
        torch.all(torch.isfinite(pit_residual))
    ):
        raise ValueError("truncated-Gaussian score is not finite")
    return nll, pit_residual


def _prior_uncertainty_scores(
    application: NeuralPriorApplication,
    reference_dbz: Tensor,
    support_target: Tensor,
    evaluation_mask: Tensor,
    *,
    support_threshold_dbz: float,
    reflectivity_resolution_dbz: float = 0.5,
    quantization_origin_dbz: float = -10.0,
    threshold_bin_convention: str = "nearest_rounding_threshold_censor",
) -> _PriorUncertaintyScores:
    """Score support everywhere and truncated intensity only on echoes."""

    echo_mask = evaluation_mask & support_target.to(
        device=evaluation_mask.device,
        dtype=torch.bool,
    )
    clear_mask = evaluation_mask & ~support_target.to(
        device=evaluation_mask.device,
        dtype=torch.bool,
    )
    echo_count = int(torch.count_nonzero(echo_mask))
    clear_count = int(torch.count_nonzero(clear_mask))
    mean_absolute: float | None = None
    underdispersion: float | None = None
    intensity_nll: float | None = None
    echo_support_miss: float | None = None
    if echo_count:
        location = application.truncated_location_dbz.masked_select(echo_mask)
        reference = reference_dbz.masked_select(echo_mask)
        scale = application.truncated_scale_dbz.masked_select(echo_mask)
        nll, pit_residual = _truncated_gaussian_diagnostics(
            location,
            scale,
            reference,
            support_threshold_dbz=support_threshold_dbz,
            reflectivity_resolution_dbz=reflectivity_resolution_dbz,
            quantization_origin_dbz=quantization_origin_dbz,
            threshold_bin_convention=threshold_bin_convention,
        )
        absolute = torch.abs(pit_residual)
        mean_absolute = float(torch.mean(absolute).detach())
        underdispersion = float(
            torch.mean((absolute > 2.0).to(absolute)).detach()
        )
        intensity_nll = float(torch.mean(nll).detach())
        echo_probability = application.event_probability.masked_select(
            echo_mask
        )
        echo_support_miss = float(
            torch.mean((1.0 - echo_probability).square()).detach()
        )
    support = application.event_probability.masked_select(evaluation_mask)
    target = support_target.to(application.event_probability).masked_select(
        evaluation_mask
    )
    brier = float(torch.mean((support - target).square()).detach())
    clear_probability = application.event_probability.masked_select(clear_mask)
    clear_false_echo = None if clear_count == 0 else float(
        torch.mean(clear_probability.square()).detach()
    )
    return _PriorUncertaintyScores(
        conditional_pit_residual_mean_abs=mean_absolute,
        conditional_underdispersion_fraction=underdispersion,
        echo_intensity_nll=intensity_nll,
        support_brier_score=brier,
        echo_support_miss_score=echo_support_miss,
        clear_sky_false_echo_score=clear_false_echo,
        echo_sample_count=echo_count,
        clear_sample_count=clear_count,
    )


def _range_uncertainty_components(
    candidate_prior: _PriorUncertaintyScores,
    parent_prior: _PriorUncertaintyScores,
    candidate_state: _StateCalibrationScores,
    parent_state: _StateCalibrationScores,
    *,
    candidate_object_miss: float | None,
    parent_object_miss: float | None,
    object_count: int,
    candidate_state_object_miss: float | None,
    parent_state_object_miss: float | None,
    state_object_count: int,
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    tuple[tuple[str, int], ...],
]:
    """Return absolute paired scores, deltas, and physical sample sizes."""

    candidate_values: list[tuple[str, float]] = []
    parent_values: list[tuple[str, float]] = []
    differences: list[tuple[str, float]] = []
    counts: list[tuple[str, int]] = []

    def add(name: str, candidate: float | None, parent: float | None, count: int) -> None:
        if candidate is None or parent is None or count <= 0:
            return
        candidate_values.append((name, candidate))
        parent_values.append((name, parent))
        differences.append((name, candidate - parent))
        counts.append((name, count))

    total_prior = candidate_prior.echo_sample_count + candidate_prior.clear_sample_count
    total_state = candidate_state.sample_count
    add("intensity", candidate_prior.echo_intensity_nll, parent_prior.echo_intensity_nll, candidate_prior.echo_sample_count)
    add(
        "pit_residual",
        candidate_prior.conditional_pit_residual_mean_abs,
        parent_prior.conditional_pit_residual_mean_abs,
        candidate_prior.echo_sample_count,
    )
    add("support", candidate_prior.support_brier_score, parent_prior.support_brier_score, total_prior)
    add("echo_miss", candidate_prior.echo_support_miss_score, parent_prior.echo_support_miss_score, candidate_prior.echo_sample_count)
    add("object_miss", candidate_object_miss, parent_object_miss, object_count)
    add("clear", candidate_prior.clear_sky_false_echo_score, parent_prior.clear_sky_false_echo_score, candidate_prior.clear_sample_count)
    add("underdispersion", candidate_prior.conditional_underdispersion_fraction, parent_prior.conditional_underdispersion_fraction, candidate_prior.echo_sample_count)
    add("state_nll", candidate_state.gaussian_nll, parent_state.gaussian_nll, total_state)
    add(
        "state_pit_residual",
        candidate_state.pit_residual_mean_abs,
        parent_state.pit_residual_mean_abs,
        total_state,
    )
    add("state_underdispersion", candidate_state.underdispersion_fraction, parent_state.underdispersion_fraction, total_state)
    add("state_support", candidate_state.support_brier_score, parent_state.support_brier_score, total_state)
    add("state_echo_miss", candidate_state.echo_support_miss_score, parent_state.echo_support_miss_score, candidate_state.echo_sample_count)
    add(
        "state_object_miss",
        candidate_state_object_miss,
        parent_state_object_miss,
        state_object_count,
    )
    add("state_false_support", candidate_state.false_support_score, parent_state.false_support_score, candidate_state.clear_sample_count)
    add("state_valid", candidate_state.valid_brier_score, parent_state.valid_brier_score, total_state)
    return (
        tuple(candidate_values),
        tuple(parent_values),
        tuple(differences),
        tuple(counts),
    )


def _new_prior_holdout_evaluation(**values: object) -> PriorHoldoutEvaluation:
    """Internal constructor used only after forecast-derived values exist."""

    result = object.__new__(PriorHoldoutEvaluation)
    object.__setattr__(
        result,
        "contract",
        "prior-holdout-evaluation-v16",
    )
    for name, value in values.items():
        object.__setattr__(result, name, value)
    PriorHoldoutEvaluation.__post_init__(result)
    return result


def _forecast_coverage(
    result: ForecastResult,
    verification: _ResolvedVerification,
    leads: tuple[int, ...],
    config: SensitivityConfig,
) -> Tensor:
    finite = verification.valid_mask & torch.isfinite(verification.frames_dbz)
    values = result.state.echo_linear.new_zeros(len(leads))
    for index, minutes in enumerate(leads):
        frame = minutes // result.run.config.interval_minutes - 1
        weight = _metric_domain_weight(result, finite[frame], frame, config.metric_domain)
        denominator = torch.count_nonzero(finite[frame]).clamp_min(1)
        values[index] = torch.count_nonzero(weight > 0).to(values) / denominator
    return values


def _evaluation_digest(value: PriorHoldoutEvaluation) -> str:
    return json_digest(
        {
            "contract": value.contract,
            "holdout_plan_digest": value.holdout_plan_digest,
            "candidate_manifest_digest": value.candidate_manifest_digest,
            "candidate_prior_digest": value.candidate_prior_digest,
            "parent_prior_digest": value.parent_prior_digest,
            "case_id": value.case_id,
            "storm_id": value.storm_id,
            "physical_event_digest": value.physical_event_digest,
            "day": value.day,
            "radar_id": value.radar_id,
            "regime": value.regime,
            "range_regime": value.range_regime,
            "reference_active_range_regimes": list(
                value.reference_active_range_regimes
            ),
            "range_band_contract_digest": value.range_band_contract_digest,
            "range_band_evaluations": [
                item.payload | {"evaluation_digest": item.evaluation_digest}
                for item in value.range_band_evaluations
            ],
            "regime_classifier_digest": value.regime_classifier_digest,
            "regime_classifier_manifest_digest": (
                value.regime_classifier_manifest_digest
            ),
            "regime_classification_evidence_digest": (
                value.regime_classification_evidence_digest
            ),
            "classified_regime": value.classified_regime,
            "classified_range_regimes": list(value.classified_range_regimes),
            "classifier_regime_confidence": value.classifier_regime_confidence,
            "classifier_range_confidence": value.classifier_range_confidence,
            "classifier_regime_entropy": value.classifier_regime_entropy,
            "classifier_is_ood": value.classifier_is_ood,
            "classifier_reference_agreement": (
                value.classifier_reference_agreement
            ),
            "classifier_weather_reference_agreement": (
                value.classifier_weather_reference_agreement
            ),
            "classifier_range_set_precision": value.classifier_range_set_precision,
            "classifier_range_set_recall": value.classifier_range_set_recall,
            "classifier_range_exact_set_match": (
                value.classifier_range_exact_set_match
            ),
            "classifier_false_active_band_fraction": (
                value.classifier_false_active_band_fraction
            ),
            "classifier_reference_range_is_ood": (
                value.classifier_reference_range_is_ood
            ),
            "classifier_numerical_runtime_digest": (
                value.classifier_numerical_runtime_digest
            ),
            "classifier_input_dtype": value.classifier_input_dtype,
            "classifier_input_device": value.classifier_input_device,
            "classifier_weather_top1_top2_gap": (
                value.classifier_weather_top1_top2_gap
            ),
            "classifier_minimum_range_presence_margin": (
                value.classifier_minimum_range_presence_margin
            ),
            "candidate_forecast_digest": value.candidate_forecast_digest,
            "parent_forecast_digest": value.parent_forecast_digest,
            "candidate_prior_application_digest": (
                value.candidate_prior_application_digest
            ),
            "parent_prior_application_digest": (
                value.parent_prior_application_digest
            ),
            "candidate_inference_evidence_digest": (
                value.candidate_inference_evidence_digest
            ),
            "parent_inference_evidence_digest": (
                value.parent_inference_evidence_digest
            ),
            "metric_change": tensor_digest(value.metric_change),
            "candidate_issuance_effect": tensor_digest(
                value.candidate_issuance_effect
            ),
            "parent_issuance_effect": tensor_digest(value.parent_issuance_effect),
            "end_to_end_metric_change": tensor_digest(value.end_to_end_metric_change),
            "metric_available": tensor_digest(value.metric_available),
            "lead_minutes": list(value.lead_minutes),
            "metric_names": list(value.metric_names),
            "verification_digest": value.verification_digest,
            "metric_contract_digest": value.metric_contract_digest,
            "coverage_candidate": tensor_digest(value.coverage_candidate),
            "coverage_parent": tensor_digest(value.coverage_parent),
            "coverage_common": tensor_digest(value.coverage_common),
            "newly_issued_fraction": tensor_digest(value.newly_issued_fraction),
            "withdrawn_fraction": tensor_digest(value.withdrawn_fraction),
            "prior_conditional_pit_residual_mean_abs": (
                value.prior_conditional_pit_residual_mean_abs
            ),
            "prior_conditional_underdispersion_fraction": (
                value.prior_conditional_underdispersion_fraction
            ),
            "prior_echo_intensity_nll": value.prior_echo_intensity_nll,
            "prior_support_brier_score": value.prior_support_brier_score,
            "prior_echo_support_miss_score": (
                value.prior_echo_support_miss_score
            ),
            "prior_echo_object_miss_score": (
                value.prior_echo_object_miss_score
            ),
            "prior_clear_sky_false_echo_score": (
                value.prior_clear_sky_false_echo_score
            ),
            "parent_prior_conditional_underdispersion_fraction": (
                value.parent_prior_conditional_underdispersion_fraction
            ),
            "parent_prior_conditional_pit_residual_mean_abs": (
                value.parent_prior_conditional_pit_residual_mean_abs
            ),
            "parent_prior_echo_intensity_nll": (
                value.parent_prior_echo_intensity_nll
            ),
            "parent_prior_support_brier_score": (
                value.parent_prior_support_brier_score
            ),
            "parent_prior_echo_support_miss_score": (
                value.parent_prior_echo_support_miss_score
            ),
            "parent_prior_echo_object_miss_score": (
                value.parent_prior_echo_object_miss_score
            ),
            "parent_prior_clear_sky_false_echo_score": (
                value.parent_prior_clear_sky_false_echo_score
            ),
            "prior_echo_intensity_status": value.prior_echo_intensity_status,
            "prior_clear_sky_status": value.prior_clear_sky_status,
            "prior_candidate_valid_fraction": (
                value.prior_candidate_valid_fraction
            ),
            "prior_parent_valid_fraction": value.prior_parent_valid_fraction,
            "prior_candidate_valid_area_km2": (
                value.prior_candidate_valid_area_km2
            ),
            "prior_abstention_increase_vs_parent": (
                value.prior_abstention_increase_vs_parent
            ),
            "prior_uncertainty_target_digest": (
                value.prior_uncertainty_target_digest
            ),
            "prior_uncertainty_sample_count": value.prior_uncertainty_sample_count,
            "prior_echo_intensity_sample_count": (
                value.prior_echo_intensity_sample_count
            ),
            "prior_clear_sky_sample_count": value.prior_clear_sky_sample_count,
            "prior_echo_area_km2": value.prior_echo_area_km2,
            "prior_clear_sky_area_km2": value.prior_clear_sky_area_km2,
            "prior_echo_object_count": value.prior_echo_object_count,
            "state_candidate_gaussian_nll": value.state_candidate_gaussian_nll,
            "state_parent_gaussian_nll": value.state_parent_gaussian_nll,
            "state_candidate_pit_residual_mean_abs": (
                value.state_candidate_pit_residual_mean_abs
            ),
            "state_parent_pit_residual_mean_abs": (
                value.state_parent_pit_residual_mean_abs
            ),
            "state_candidate_underdispersion_fraction": (
                value.state_candidate_underdispersion_fraction
            ),
            "state_parent_underdispersion_fraction": (
                value.state_parent_underdispersion_fraction
            ),
            "state_candidate_support_brier_score": (
                value.state_candidate_support_brier_score
            ),
            "state_parent_support_brier_score": (
                value.state_parent_support_brier_score
            ),
            "state_candidate_echo_support_miss_score": (
                value.state_candidate_echo_support_miss_score
            ),
            "state_parent_echo_support_miss_score": (
                value.state_parent_echo_support_miss_score
            ),
            "state_candidate_echo_object_miss_score": (
                value.state_candidate_echo_object_miss_score
            ),
            "state_parent_echo_object_miss_score": (
                value.state_parent_echo_object_miss_score
            ),
            "state_candidate_false_support_score": (
                value.state_candidate_false_support_score
            ),
            "state_parent_false_support_score": (
                value.state_parent_false_support_score
            ),
            "state_candidate_valid_brier_score": (
                value.state_candidate_valid_brier_score
            ),
            "state_parent_valid_brier_score": (
                value.state_parent_valid_brier_score
            ),
            "state_calibration_target_digest": (
                value.state_calibration_target_digest
            ),
            "state_calibration_sample_count": value.state_calibration_sample_count,
            "state_calibration_echo_sample_count": (
                value.state_calibration_echo_sample_count
            ),
            "state_calibration_clear_sample_count": (
                value.state_calibration_clear_sample_count
            ),
            "state_calibration_echo_object_count": (
                value.state_calibration_echo_object_count
            ),
            "issue_time": value.issue_time,
            "verification_valid_times": list(value.verification_valid_times),
        }
    )


@dataclass(frozen=True)
class ProcessLogArtifact:
    """Canonical process log whose bytes are retained for durable audit."""

    process_kind: Literal["candidate_training", "candidate_scoring"]
    start_receipt_digest: str
    entries: tuple[str, ...]
    contract: str = "trusted-process-log-artifact-v1"
    artifact_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("process-log start receipt", self.start_receipt_digest)
        if (
            self.contract != "trusted-process-log-artifact-v1"
            or self.process_kind not in ("candidate_training", "candidate_scoring")
            or not self.entries
            or any(not item or item.strip() != item for item in self.entries)
        ):
            raise ValueError("trusted process-log artifact is invalid")
        object.__setattr__(self, "artifact_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "process_kind": self.process_kind,
            "start_receipt_digest": self.start_receipt_digest,
            "entries": list(self.entries),
        }


def validate_process_log_artifact(artifact: ProcessLogArtifact) -> None:
    """Rehash the retained log payload before trusting a completion receipt."""

    try:
        if (
            artifact.contract != "trusted-process-log-artifact-v1"
            or artifact.process_kind
            not in ("candidate_training", "candidate_scoring")
            or artifact.artifact_digest != json_digest(artifact.payload)
            or not artifact.entries
            or any(not item or item.strip() != item for item in artifact.entries)
        ):
            raise ValueError("trusted process-log artifact is invalid")
        _require_digest("process-log start receipt", artifact.start_receipt_digest)
    except (AttributeError, ValueError) as error:
        raise ValueError("trusted process-log artifact is invalid") from error


@dataclass(frozen=True, init=False)
class HoldoutScoringArtifact:
    """Canonical ordered output of one preregistered holdout scoring job."""

    holdout_plan_digest: str
    candidate_manifest_digest: str
    scoring_start_receipt_digest: str
    scoring_algorithm_digest: str
    scoring_runtime_digest: str
    scoring_execution_contract_digest: str
    ordered_case_ids: tuple[str, ...]
    ordered_evaluation_digests: tuple[str, ...]
    candidate_forecast_digests: tuple[str, ...]
    parent_forecast_digests: tuple[str, ...]
    verification_digests: tuple[str, ...]
    metric_contract_digests: tuple[str, ...]
    contract: str = "neural-prior-holdout-scoring-artifact-v1"
    artifact_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use HoldoutScoringArtifact.from_evaluations")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "holdout_plan_digest": self.holdout_plan_digest,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "scoring_start_receipt_digest": self.scoring_start_receipt_digest,
            "scoring_algorithm_digest": self.scoring_algorithm_digest,
            "scoring_runtime_digest": self.scoring_runtime_digest,
            "scoring_execution_contract_digest": (
                self.scoring_execution_contract_digest
            ),
            "ordered_case_ids": list(self.ordered_case_ids),
            "ordered_evaluation_digests": list(
                self.ordered_evaluation_digests
            ),
            "candidate_forecast_digests": list(
                self.candidate_forecast_digests
            ),
            "parent_forecast_digests": list(self.parent_forecast_digests),
            "verification_digests": list(self.verification_digests),
            "metric_contract_digests": list(self.metric_contract_digests),
        }

    @classmethod
    def from_evaluations(
        cls,
        manifest: NeuralPriorCandidateManifest,
        plan: NeuralPriorHoldoutPlan,
        evaluations: tuple[PriorHoldoutEvaluation, ...],
    ) -> HoldoutScoringArtifact:
        ordered = tuple(sorted(evaluations, key=lambda item: item.case_id))
        values: dict[str, object] = {
            "holdout_plan_digest": plan.plan_digest,
            "candidate_manifest_digest": manifest.manifest_digest,
            "scoring_start_receipt_digest": (
                manifest.candidate_scoring_start_receipt.receipt_digest
            ),
            "scoring_algorithm_digest": plan.scoring_algorithm_digest,
            "scoring_runtime_digest": plan.scoring_runtime_digest,
            "scoring_execution_contract_digest": (
                plan.scoring_execution_contract_digest
            ),
            "ordered_case_ids": tuple(item.case_id for item in ordered),
            "ordered_evaluation_digests": tuple(
                item.evaluation_digest for item in ordered
            ),
            "candidate_forecast_digests": tuple(
                item.candidate_forecast_digest for item in ordered
            ),
            "parent_forecast_digests": tuple(
                item.parent_forecast_digest for item in ordered
            ),
            "verification_digests": tuple(
                item.verification_digest for item in ordered
            ),
            "metric_contract_digests": tuple(
                item.metric_contract_digest for item in ordered
            ),
            "contract": "neural-prior-holdout-scoring-artifact-v1",
        }
        artifact = _new_holdout_scoring_artifact(**values)
        validate_holdout_scoring_artifact(artifact, manifest, plan, evaluations)
        return artifact


def _new_holdout_scoring_artifact(**values: object) -> HoldoutScoringArtifact:
    artifact = object.__new__(HoldoutScoringArtifact)
    for name, value in values.items():
        object.__setattr__(artifact, name, value)
    object.__setattr__(artifact, "artifact_digest", json_digest(artifact.payload))
    return artifact


def validate_holdout_scoring_artifact(
    artifact: HoldoutScoringArtifact,
    manifest: NeuralPriorCandidateManifest,
    plan: NeuralPriorHoldoutPlan,
    evaluations: tuple[PriorHoldoutEvaluation, ...],
) -> None:
    ordered = tuple(sorted(evaluations, key=lambda item: item.case_id))
    start = manifest.candidate_scoring_start_receipt
    if (
        artifact.contract != "neural-prior-holdout-scoring-artifact-v1"
        or artifact.artifact_digest != json_digest(artifact.payload)
        or artifact.holdout_plan_digest != plan.plan_digest
        or artifact.candidate_manifest_digest != manifest.manifest_digest
        or artifact.scoring_start_receipt_digest
        != manifest.candidate_scoring_start_receipt.receipt_digest
        or artifact.scoring_algorithm_digest != plan.scoring_algorithm_digest
        or artifact.scoring_runtime_digest != plan.scoring_runtime_digest
        or artifact.scoring_execution_contract_digest
        != plan.scoring_execution_contract_digest
        or start.process_kind != "candidate_scoring"
        or set(start.subject_digests) != set(plan.candidate_family_digests)
        or start.process_algorithm_digest != artifact.scoring_algorithm_digest
        or start.process_runtime_digest != artifact.scoring_runtime_digest
        or start.execution_contract_digest
        != artifact.scoring_execution_contract_digest
        or start.catalog_plan_digest
        != plan.physical_event_catalog_plan.plan_digest
        or start.catalog_result_digest
        != manifest.physical_event_catalog_result.result_digest
        or artifact.ordered_case_ids != tuple(item.case_id for item in ordered)
        or artifact.ordered_evaluation_digests
        != tuple(item.evaluation_digest for item in ordered)
        or artifact.candidate_forecast_digests
        != tuple(item.candidate_forecast_digest for item in ordered)
        or artifact.parent_forecast_digests
        != tuple(item.parent_forecast_digest for item in ordered)
        or artifact.verification_digests
        != tuple(item.verification_digest for item in ordered)
        or artifact.metric_contract_digests
        != tuple(item.metric_contract_digest for item in ordered)
        or len(set(artifact.ordered_case_ids)) != len(artifact.ordered_case_ids)
        or set(artifact.ordered_case_ids) != {item.case_id for item in plan.cases}
    ):
        raise ValueError("holdout scoring artifact disagrees with typed evaluations")
    for name in (
        "holdout_plan_digest",
        "candidate_manifest_digest",
        "scoring_start_receipt_digest",
        "scoring_algorithm_digest",
        "scoring_runtime_digest",
        "scoring_execution_contract_digest",
    ):
        _require_digest(f"scoring artifact {name}", getattr(artifact, name))


@dataclass(frozen=True)
class RangeMetricRequirement:
    """Minimum evidence for one operational weather/band/metric/lead cell."""

    weather_regime: str
    range_regime: str
    metric_name: str
    lead_minutes: int
    minimum_cases: int
    minimum_physical_events: int
    minimum_valid_area_km2: float
    maximum_mean_normalized_degradation: float
    maximum_harmful_fraction_upper_bound: float
    maximum_absolute_normalized_change: float = 2.0
    maximum_end_to_end_mean_normalized_degradation: float = 0.0
    maximum_end_to_end_harmful_fraction_upper_bound: float = 1.0
    contract: str = "range-metric-requirement-v5"

    def __post_init__(self) -> None:
        if (
            self.contract != "range-metric-requirement-v5"
            or any(
                not value or value.strip() != value
                for value in (
                    self.weather_regime,
                    self.range_regime,
                    self.metric_name,
                )
            )
            or type(self.lead_minutes) is not int
            or self.lead_minutes <= 0
            or type(self.minimum_cases) is not int
            or self.minimum_cases <= 0
            or type(self.minimum_physical_events) is not int
            or self.minimum_physical_events <= 0
            or not math.isfinite(self.minimum_valid_area_km2)
            or self.minimum_valid_area_km2 < 0.0
            or not math.isfinite(self.maximum_mean_normalized_degradation)
            or self.maximum_mean_normalized_degradation < 0.0
            or not math.isfinite(self.maximum_harmful_fraction_upper_bound)
            or not 0.0 <= self.maximum_harmful_fraction_upper_bound <= 1.0
            or not math.isfinite(self.maximum_absolute_normalized_change)
            or self.maximum_absolute_normalized_change <= 0.0
            or not math.isfinite(
                self.maximum_end_to_end_mean_normalized_degradation
            )
            or self.maximum_end_to_end_mean_normalized_degradation < 0.0
            or not math.isfinite(
                self.maximum_end_to_end_harmful_fraction_upper_bound
            )
            or not 0.0
            <= self.maximum_end_to_end_harmful_fraction_upper_bound
            <= 1.0
        ):
            raise ValueError("range metric requirement is invalid")

    @property
    def payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RangeIssuanceRequirement:
    """Verification-independent safety gate for one weather/band/lead cell."""

    weather_regime: str
    range_regime: str
    lead_minutes: int
    minimum_cases: int
    minimum_physical_events: int
    minimum_operational_area_km2: float
    maximum_withdrawn_fraction: float
    maximum_newly_issued_fraction: float
    maximum_background_fallback_increase: float
    maximum_confidence_weighted_coverage_loss: float
    contract: str = "range-issuance-requirement-v1"

    def __post_init__(self) -> None:
        if (
            self.contract != "range-issuance-requirement-v1"
            or any(
                not value or value.strip() != value
                for value in (self.weather_regime, self.range_regime)
            )
            or type(self.lead_minutes) is not int
            or self.lead_minutes <= 0
            or type(self.minimum_cases) is not int
            or self.minimum_cases <= 0
            or type(self.minimum_physical_events) is not int
            or self.minimum_physical_events <= 0
            or not math.isfinite(self.minimum_operational_area_km2)
            or self.minimum_operational_area_km2 <= 0.0
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in (
                    self.maximum_withdrawn_fraction,
                    self.maximum_newly_issued_fraction,
                    self.maximum_background_fallback_increase,
                    self.maximum_confidence_weighted_coverage_loss,
                )
            )
        ):
            raise ValueError("range issuance requirement is invalid")

    @property
    def payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class NeuralPriorPromotionPolicy:
    """Root-approved cluster-aware limits for promoting one prior."""

    metric_scales: tuple[PromotionMetricScale, ...]
    approved_candidate_manifest_digests: tuple[str, ...]
    approved_holdout_plan_digests: tuple[str, ...]
    approved_metric_contract_digests: tuple[str, ...]
    approved_physical_event_catalog_result_digest: str
    deployment_regime_classifier_digest: str
    deployment_regime_classifier_manifest_digest: str
    required_range_metrics: tuple[RangeMetricRequirement, ...]
    required_range_issuance: tuple[RangeIssuanceRequirement, ...]
    minimum_holdout_cases: int = 20
    minimum_material_cases: int = 20
    minimum_material_case_fraction: float = 0.8
    minimum_independent_cases: int = 20
    minimum_distinct_storms: int = 5
    minimum_distinct_days: int = 5
    minimum_distinct_radars: int = 1
    minimum_distinct_regimes: int = 2
    minimum_distinct_range_regimes: int = 2
    minimum_material_clusters: int = 5
    minimum_beneficial_fraction: float = 0.6
    maximum_harmful_fraction: float = 0.1
    minimum_mean_normalized_improvement: float = 0.05
    maximum_single_normalized_degradation: float = 1.0
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    maximum_coverage_loss: float = 0.05
    maximum_newly_issued_fraction: float = 0.05
    maximum_withdrawn_fraction: float = 0.05
    maximum_prior_conditional_pit_residual_mean_abs: float = 2.0
    maximum_prior_conditional_underdispersion_fraction: float = 0.1
    maximum_prior_echo_intensity_nll: float = 4.0
    maximum_prior_support_brier_score: float = 0.25
    maximum_prior_echo_support_miss_score: float = 0.25
    maximum_prior_echo_object_miss_score: float = 0.25
    maximum_prior_clear_sky_false_echo_score: float = 0.25
    minimum_prior_uncertainty_samples_per_case: int = 1
    minimum_prior_echo_pixels_per_case: int = 16
    minimum_prior_clear_pixels_per_case: int = 16
    minimum_prior_echo_area_km2_per_case: float = 1.0
    minimum_prior_clear_area_km2_per_case: float = 1.0
    minimum_prior_echo_objects_per_case: int = 1
    minimum_prior_echo_cases: int = 5
    minimum_prior_clear_cases: int = 5
    minimum_prior_echo_clusters: int = 5
    minimum_prior_clear_clusters: int = 5
    minimum_uncertainty_cases_per_regime: int = 2
    minimum_echo_cases_per_regime: int = 2
    minimum_clear_cases_per_regime: int = 2
    minimum_uncertainty_clusters_per_regime: int = 5
    minimum_echo_clusters_per_regime: int = 5
    minimum_clear_clusters_per_regime: int = 5
    minimum_prior_valid_fraction: float = 0.5
    minimum_prior_valid_area_km2: float = 1.0
    maximum_abstention_increase_vs_parent: float = 0.05
    prior_abstention_penalty_weight: float = 1.0
    maximum_prior_echo_intensity_nll_increase: float = 0.0
    maximum_prior_support_brier_increase: float = 0.0
    maximum_prior_echo_support_miss_increase: float = 0.0
    maximum_prior_echo_object_miss_increase: float = 0.0
    maximum_prior_clear_sky_false_echo_increase: float = 0.0
    maximum_prior_conditional_pit_residual_increase: float = 0.0
    maximum_prior_conditional_underdispersion_increase: float = 0.0
    maximum_state_pit_residual_mean_abs: float = 2.0
    maximum_state_underdispersion_fraction: float = 0.1
    maximum_state_gaussian_nll: float = 6.0
    maximum_state_support_brier_score: float = 0.25
    maximum_state_echo_support_miss_score: float = 0.25
    maximum_state_echo_object_miss_score: float = 0.25
    maximum_state_false_support_score: float = 0.25
    maximum_state_valid_brier_score: float = 0.25
    minimum_state_calibration_samples_per_case: int = 16
    minimum_state_calibration_cases_per_regime: int = 5
    minimum_state_calibration_clusters_per_regime: int = 5
    maximum_state_gaussian_nll_increase: float = 0.0
    maximum_state_pit_residual_increase: float = 0.0
    maximum_state_underdispersion_increase: float = 0.0
    maximum_state_support_brier_increase: float = 0.0
    maximum_state_echo_support_miss_increase: float = 0.0
    maximum_state_echo_object_miss_increase: float = 0.0
    maximum_state_false_support_increase: float = 0.0
    maximum_state_valid_brier_increase: float = 0.0
    minimum_regime_classifier_accuracy: float = 0.9
    minimum_regime_classifier_recall: float = 0.8
    minimum_regime_classifier_accuracy_lower_bound: float = 0.9
    minimum_regime_classifier_recall_lower_bound: float = 0.8
    maximum_regime_classifier_calibration_error: float = 0.1
    maximum_regime_classifier_false_routing_fraction: float = 0.05
    maximum_regime_classifier_false_routing_upper_bound: float = 0.05
    minimum_regime_classifier_clusters: int = 5
    minimum_regime_classifier_ood_cases: int = 1
    minimum_regime_classifier_ood_abstention_fraction: float = 0.9
    minimum_range_set_precision: float = 0.95
    minimum_range_set_recall: float = 0.95
    minimum_range_set_precision_lower_bound: float = 0.95
    minimum_range_set_recall_lower_bound: float = 0.95
    minimum_range_exact_set_accuracy: float = 0.9
    maximum_false_active_band_fraction: float = 0.05
    maximum_false_active_band_upper_bound: float = 0.05
    minimum_weather_top1_top2_gap: float = 0.05
    minimum_range_presence_margin: float = 0.05
    minimum_range_band_cases: int = 2
    minimum_range_band_clusters: int = 5
    minimum_range_band_area_km2: float = 1.0
    minimum_range_metric_valid_area_km2: float = 1.0
    minimum_range_probability_valid_area_km2: float = 1.0
    minimum_range_state_valid_area_km2: float = 1.0
    minimum_range_component_samples: int = 16
    minimum_range_echo_objects: int = 1
    minimum_range_state_echo_objects: int = 1
    minimum_range_classifier_ood_cases: int = 1
    minimum_range_classifier_ood_abstention_fraction: float = 0.9
    require_all_registered_regimes_certified: bool = False
    minimum_bootstrap_tail_replicates: int = 20
    maximum_exact_sign_clusters: int = 16
    minimum_deployment_metric_cell_events: int = 5
    minimum_continuous_metric_cell_events: int = 10
    contract: str = "neural-prior-promotion-policy-v22"

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-promotion-policy-v22":
            raise ValueError("unsupported neural-prior promotion policy")
        if not self.metric_scales or len({x.metric_name for x in self.metric_scales}) != len(self.metric_scales):
            raise ValueError("promotion metric scales must be unique")
        requirement_keys = tuple(
            (
                item.weather_regime,
                item.range_regime,
                item.metric_name,
                item.lead_minutes,
            )
            for item in self.required_range_metrics
        )
        if (
            not self.required_range_metrics
            or len(set(requirement_keys)) != len(requirement_keys)
            or any(
                item.metric_name not in {scale.metric_name for scale in self.metric_scales}
                for item in self.required_range_metrics
            )
        ):
            raise ValueError("required range metrics are invalid")
        issuance_keys = tuple(
            (item.weather_regime, item.range_regime, item.lead_minutes)
            for item in self.required_range_issuance
        )
        if (
            not self.required_range_issuance
            or len(set(issuance_keys)) != len(issuance_keys)
        ):
            raise ValueError("required range issuance cells are invalid")
        if not self.approved_candidate_manifest_digests:
            raise ValueError("promotion policy must approve candidate manifests")
        for digest in self.approved_candidate_manifest_digests:
            _require_digest("approved candidate manifest digest", digest)
        if not self.approved_holdout_plan_digests:
            raise ValueError("promotion policy must approve holdout plans")
        if not self.approved_metric_contract_digests:
            raise ValueError("promotion policy must approve metric contracts")
        for digest in (
            self.approved_holdout_plan_digests
            + self.approved_metric_contract_digests
        ):
            _require_digest("approved promotion contract digest", digest)
        _require_digest(
            "approved physical event-catalog result",
            self.approved_physical_event_catalog_result_digest,
        )
        _require_digest(
            "deployment regime classifier digest",
            self.deployment_regime_classifier_digest,
        )
        _require_digest(
            "deployment regime classifier manifest digest",
            self.deployment_regime_classifier_manifest_digest,
        )
        integer_limits = (
            self.minimum_holdout_cases,
            self.minimum_material_cases,
            self.minimum_independent_cases,
            self.minimum_distinct_storms,
            self.minimum_distinct_days,
            self.minimum_distinct_radars,
            self.minimum_distinct_regimes,
            self.minimum_distinct_range_regimes,
            self.minimum_material_clusters,
            self.minimum_prior_uncertainty_samples_per_case,
            self.minimum_prior_echo_pixels_per_case,
            self.minimum_prior_clear_pixels_per_case,
            self.minimum_prior_echo_objects_per_case,
            self.minimum_prior_echo_cases,
            self.minimum_prior_clear_cases,
            self.minimum_prior_echo_clusters,
            self.minimum_prior_clear_clusters,
            self.minimum_uncertainty_cases_per_regime,
            self.minimum_echo_cases_per_regime,
            self.minimum_clear_cases_per_regime,
            self.minimum_uncertainty_clusters_per_regime,
            self.minimum_echo_clusters_per_regime,
            self.minimum_clear_clusters_per_regime,
            self.minimum_bootstrap_tail_replicates,
            self.maximum_exact_sign_clusters,
            self.bootstrap_samples,
            self.minimum_state_calibration_samples_per_case,
            self.minimum_state_calibration_cases_per_regime,
            self.minimum_state_calibration_clusters_per_regime,
            self.minimum_range_band_cases,
            self.minimum_range_band_clusters,
            self.minimum_range_component_samples,
            self.minimum_range_echo_objects,
            self.minimum_range_state_echo_objects,
            self.minimum_regime_classifier_clusters,
            self.minimum_deployment_metric_cell_events,
            self.minimum_continuous_metric_cell_events,
        )
        if any(type(value) is not int or value <= 0 for value in integer_limits):
            raise ValueError("promotion count limits must be positive integers")
        probabilities = (
            self.minimum_material_case_fraction,
            self.minimum_beneficial_fraction,
            self.maximum_harmful_fraction,
            self.confidence_level,
            self.maximum_coverage_loss,
            self.maximum_newly_issued_fraction,
            self.maximum_withdrawn_fraction,
            self.maximum_prior_conditional_underdispersion_fraction,
            self.maximum_prior_support_brier_score,
            self.maximum_prior_echo_support_miss_score,
            self.maximum_prior_echo_object_miss_score,
            self.maximum_prior_clear_sky_false_echo_score,
            self.minimum_prior_valid_fraction,
            self.maximum_abstention_increase_vs_parent,
            self.maximum_state_underdispersion_fraction,
            self.maximum_state_support_brier_score,
            self.maximum_state_echo_support_miss_score,
            self.maximum_state_echo_object_miss_score,
            self.maximum_state_false_support_score,
            self.maximum_state_valid_brier_score,
            self.minimum_regime_classifier_accuracy,
            self.minimum_regime_classifier_recall,
            self.minimum_regime_classifier_accuracy_lower_bound,
            self.minimum_regime_classifier_recall_lower_bound,
            self.maximum_regime_classifier_calibration_error,
            self.maximum_regime_classifier_false_routing_fraction,
            self.maximum_regime_classifier_false_routing_upper_bound,
            self.minimum_regime_classifier_ood_abstention_fraction,
            self.minimum_range_set_precision,
            self.minimum_range_set_recall,
            self.minimum_range_set_precision_lower_bound,
            self.minimum_range_set_recall_lower_bound,
            self.minimum_range_exact_set_accuracy,
            self.maximum_false_active_band_fraction,
            self.maximum_false_active_band_upper_bound,
            self.minimum_range_classifier_ood_abstention_fraction,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("promotion fractions must be inside [0,1]")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("promotion confidence level must be inside (0,1)")
        if (
            type(self.minimum_regime_classifier_ood_cases) is not int
            or self.minimum_regime_classifier_ood_cases < 0
            or type(self.minimum_range_classifier_ood_cases) is not int
            or self.minimum_range_classifier_ood_cases < 0
            or type(self.require_all_registered_regimes_certified) is not bool
        ):
            raise ValueError("regime-classifier policy limits are invalid")
        for name, value in (
            (
                "minimum_mean_normalized_improvement",
                self.minimum_mean_normalized_improvement,
            ),
            (
                "maximum_single_normalized_degradation",
                self.maximum_single_normalized_degradation,
            ),
            (
                "maximum_prior_conditional_pit_residual_mean_abs",
                self.maximum_prior_conditional_pit_residual_mean_abs,
            ),
            (
                "maximum_prior_echo_intensity_nll",
                self.maximum_prior_echo_intensity_nll,
            ),
            ("minimum_prior_valid_area_km2", self.minimum_prior_valid_area_km2),
            (
                "minimum_prior_echo_area_km2_per_case",
                self.minimum_prior_echo_area_km2_per_case,
            ),
            (
                "minimum_prior_clear_area_km2_per_case",
                self.minimum_prior_clear_area_km2_per_case,
            ),
            (
                "prior_abstention_penalty_weight",
                self.prior_abstention_penalty_weight,
            ),
            (
                "maximum_prior_echo_intensity_nll_increase",
                self.maximum_prior_echo_intensity_nll_increase,
            ),
            (
                "maximum_prior_support_brier_increase",
                self.maximum_prior_support_brier_increase,
            ),
            (
                "maximum_prior_echo_support_miss_increase",
                self.maximum_prior_echo_support_miss_increase,
            ),
            (
                "maximum_prior_echo_object_miss_increase",
                self.maximum_prior_echo_object_miss_increase,
            ),
            (
                "maximum_prior_clear_sky_false_echo_increase",
                self.maximum_prior_clear_sky_false_echo_increase,
            ),
            (
                "maximum_prior_conditional_underdispersion_increase",
                self.maximum_prior_conditional_underdispersion_increase,
            ),
            (
                "maximum_prior_conditional_pit_residual_increase",
                self.maximum_prior_conditional_pit_residual_increase,
            ),
            ("maximum_state_pit_residual_mean_abs", self.maximum_state_pit_residual_mean_abs),
            ("maximum_state_gaussian_nll", self.maximum_state_gaussian_nll),
            ("maximum_state_gaussian_nll_increase", self.maximum_state_gaussian_nll_increase),
            (
                "maximum_state_pit_residual_increase",
                self.maximum_state_pit_residual_increase,
            ),
            ("maximum_state_underdispersion_increase", self.maximum_state_underdispersion_increase),
            ("maximum_state_support_brier_increase", self.maximum_state_support_brier_increase),
            ("maximum_state_echo_support_miss_increase", self.maximum_state_echo_support_miss_increase),
            ("maximum_state_echo_object_miss_increase", self.maximum_state_echo_object_miss_increase),
            ("maximum_state_false_support_increase", self.maximum_state_false_support_increase),
            ("maximum_state_valid_brier_increase", self.maximum_state_valid_brier_increase),
            ("minimum_weather_top1_top2_gap", self.minimum_weather_top1_top2_gap),
            ("minimum_range_presence_margin", self.minimum_range_presence_margin),
            ("minimum_range_band_area_km2", self.minimum_range_band_area_km2),
            (
                "minimum_range_metric_valid_area_km2",
                self.minimum_range_metric_valid_area_km2,
            ),
            (
                "minimum_range_probability_valid_area_km2",
                self.minimum_range_probability_valid_area_km2,
            ),
            (
                "minimum_range_state_valid_area_km2",
                self.minimum_range_state_valid_area_km2,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def digest(self) -> str:
        return json_digest({
            "contract": self.contract,
            "metric_scales": [item.__dict__ for item in self.metric_scales],
            "approved_candidate_manifest_digests": sorted(
                self.approved_candidate_manifest_digests
            ),
            "approved_holdout_plan_digests": sorted(
                self.approved_holdout_plan_digests
            ),
            "approved_metric_contract_digests": sorted(
                self.approved_metric_contract_digests
            ),
            "approved_physical_event_catalog_result_digest": (
                self.approved_physical_event_catalog_result_digest
            ),
            "deployment_regime_classifier_digest": (
                self.deployment_regime_classifier_digest
            ),
            "deployment_regime_classifier_manifest_digest": (
                self.deployment_regime_classifier_manifest_digest
            ),
            "required_range_metrics": [
                item.payload for item in self.required_range_metrics
            ],
            "minimum_holdout_cases": self.minimum_holdout_cases,
            "minimum_material_cases": self.minimum_material_cases,
            "minimum_material_case_fraction": self.minimum_material_case_fraction,
            "minimum_independent_cases": self.minimum_independent_cases,
            "minimum_distinct_storms": self.minimum_distinct_storms,
            "minimum_distinct_days": self.minimum_distinct_days,
            "minimum_distinct_radars": self.minimum_distinct_radars,
            "minimum_distinct_regimes": self.minimum_distinct_regimes,
            "minimum_distinct_range_regimes": (
                self.minimum_distinct_range_regimes
            ),
            "minimum_material_clusters": self.minimum_material_clusters,
            "minimum_beneficial_fraction": self.minimum_beneficial_fraction,
            "maximum_harmful_fraction": self.maximum_harmful_fraction,
            "minimum_mean_normalized_improvement": self.minimum_mean_normalized_improvement,
            "maximum_single_normalized_degradation": self.maximum_single_normalized_degradation,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "maximum_coverage_loss": self.maximum_coverage_loss,
            "maximum_newly_issued_fraction": self.maximum_newly_issued_fraction,
            "maximum_withdrawn_fraction": self.maximum_withdrawn_fraction,
            "maximum_prior_conditional_pit_residual_mean_abs": (
                self.maximum_prior_conditional_pit_residual_mean_abs
            ),
            "maximum_prior_conditional_underdispersion_fraction": (
                self.maximum_prior_conditional_underdispersion_fraction
            ),
            "maximum_prior_echo_intensity_nll": (
                self.maximum_prior_echo_intensity_nll
            ),
            "maximum_prior_support_brier_score": (
                self.maximum_prior_support_brier_score
            ),
            "maximum_prior_echo_support_miss_score": (
                self.maximum_prior_echo_support_miss_score
            ),
            "maximum_prior_echo_object_miss_score": (
                self.maximum_prior_echo_object_miss_score
            ),
            "maximum_prior_clear_sky_false_echo_score": (
                self.maximum_prior_clear_sky_false_echo_score
            ),
            "minimum_prior_uncertainty_samples_per_case": (
                self.minimum_prior_uncertainty_samples_per_case
            ),
            "minimum_prior_echo_pixels_per_case": (
                self.minimum_prior_echo_pixels_per_case
            ),
            "minimum_prior_clear_pixels_per_case": (
                self.minimum_prior_clear_pixels_per_case
            ),
            "minimum_prior_echo_area_km2_per_case": (
                self.minimum_prior_echo_area_km2_per_case
            ),
            "minimum_prior_clear_area_km2_per_case": (
                self.minimum_prior_clear_area_km2_per_case
            ),
            "minimum_prior_echo_objects_per_case": (
                self.minimum_prior_echo_objects_per_case
            ),
            "minimum_prior_echo_cases": self.minimum_prior_echo_cases,
            "minimum_prior_clear_cases": self.minimum_prior_clear_cases,
            "minimum_prior_echo_clusters": self.minimum_prior_echo_clusters,
            "minimum_prior_clear_clusters": self.minimum_prior_clear_clusters,
            "minimum_uncertainty_cases_per_regime": (
                self.minimum_uncertainty_cases_per_regime
            ),
            "minimum_echo_cases_per_regime": self.minimum_echo_cases_per_regime,
            "minimum_clear_cases_per_regime": self.minimum_clear_cases_per_regime,
            "minimum_uncertainty_clusters_per_regime": (
                self.minimum_uncertainty_clusters_per_regime
            ),
            "minimum_echo_clusters_per_regime": (
                self.minimum_echo_clusters_per_regime
            ),
            "minimum_clear_clusters_per_regime": (
                self.minimum_clear_clusters_per_regime
            ),
            "minimum_prior_valid_fraction": self.minimum_prior_valid_fraction,
            "minimum_prior_valid_area_km2": self.minimum_prior_valid_area_km2,
            "maximum_abstention_increase_vs_parent": (
                self.maximum_abstention_increase_vs_parent
            ),
            "prior_abstention_penalty_weight": (
                self.prior_abstention_penalty_weight
            ),
            "maximum_prior_echo_intensity_nll_increase": (
                self.maximum_prior_echo_intensity_nll_increase
            ),
            "maximum_prior_support_brier_increase": (
                self.maximum_prior_support_brier_increase
            ),
            "maximum_prior_echo_support_miss_increase": (
                self.maximum_prior_echo_support_miss_increase
            ),
            "maximum_prior_echo_object_miss_increase": (
                self.maximum_prior_echo_object_miss_increase
            ),
            "maximum_prior_clear_sky_false_echo_increase": (
                self.maximum_prior_clear_sky_false_echo_increase
            ),
            "maximum_prior_conditional_underdispersion_increase": (
                self.maximum_prior_conditional_underdispersion_increase
            ),
            "maximum_prior_conditional_pit_residual_increase": (
                self.maximum_prior_conditional_pit_residual_increase
            ),
            "maximum_state_pit_residual_mean_abs": self.maximum_state_pit_residual_mean_abs,
            "maximum_state_underdispersion_fraction": self.maximum_state_underdispersion_fraction,
            "maximum_state_gaussian_nll": self.maximum_state_gaussian_nll,
            "maximum_state_support_brier_score": self.maximum_state_support_brier_score,
            "maximum_state_echo_support_miss_score": self.maximum_state_echo_support_miss_score,
            "maximum_state_echo_object_miss_score": self.maximum_state_echo_object_miss_score,
            "maximum_state_false_support_score": self.maximum_state_false_support_score,
            "maximum_state_valid_brier_score": self.maximum_state_valid_brier_score,
            "minimum_state_calibration_samples_per_case": self.minimum_state_calibration_samples_per_case,
            "minimum_state_calibration_cases_per_regime": self.minimum_state_calibration_cases_per_regime,
            "minimum_state_calibration_clusters_per_regime": self.minimum_state_calibration_clusters_per_regime,
            "maximum_state_gaussian_nll_increase": self.maximum_state_gaussian_nll_increase,
            "maximum_state_pit_residual_increase": (
                self.maximum_state_pit_residual_increase
            ),
            "maximum_state_underdispersion_increase": self.maximum_state_underdispersion_increase,
            "maximum_state_support_brier_increase": self.maximum_state_support_brier_increase,
            "maximum_state_echo_support_miss_increase": self.maximum_state_echo_support_miss_increase,
            "maximum_state_echo_object_miss_increase": self.maximum_state_echo_object_miss_increase,
            "maximum_state_false_support_increase": self.maximum_state_false_support_increase,
            "maximum_state_valid_brier_increase": self.maximum_state_valid_brier_increase,
            "minimum_regime_classifier_accuracy": self.minimum_regime_classifier_accuracy,
            "minimum_regime_classifier_recall": self.minimum_regime_classifier_recall,
            "minimum_regime_classifier_accuracy_lower_bound": (
                self.minimum_regime_classifier_accuracy_lower_bound
            ),
            "minimum_regime_classifier_recall_lower_bound": (
                self.minimum_regime_classifier_recall_lower_bound
            ),
            "maximum_regime_classifier_calibration_error": self.maximum_regime_classifier_calibration_error,
            "maximum_regime_classifier_false_routing_fraction": self.maximum_regime_classifier_false_routing_fraction,
            "maximum_regime_classifier_false_routing_upper_bound": (
                self.maximum_regime_classifier_false_routing_upper_bound
            ),
            "minimum_regime_classifier_clusters": (
                self.minimum_regime_classifier_clusters
            ),
            "minimum_regime_classifier_ood_cases": self.minimum_regime_classifier_ood_cases,
            "minimum_regime_classifier_ood_abstention_fraction": self.minimum_regime_classifier_ood_abstention_fraction,
            "minimum_range_set_precision": self.minimum_range_set_precision,
            "minimum_range_set_recall": self.minimum_range_set_recall,
            "minimum_range_set_precision_lower_bound": (
                self.minimum_range_set_precision_lower_bound
            ),
            "minimum_range_set_recall_lower_bound": (
                self.minimum_range_set_recall_lower_bound
            ),
            "minimum_range_exact_set_accuracy": self.minimum_range_exact_set_accuracy,
            "maximum_false_active_band_fraction": self.maximum_false_active_band_fraction,
            "maximum_false_active_band_upper_bound": (
                self.maximum_false_active_band_upper_bound
            ),
            "minimum_weather_top1_top2_gap": self.minimum_weather_top1_top2_gap,
            "minimum_range_presence_margin": self.minimum_range_presence_margin,
            "minimum_range_band_cases": self.minimum_range_band_cases,
            "minimum_range_band_clusters": self.minimum_range_band_clusters,
            "minimum_range_band_area_km2": self.minimum_range_band_area_km2,
            "minimum_range_metric_valid_area_km2": (
                self.minimum_range_metric_valid_area_km2
            ),
            "minimum_range_probability_valid_area_km2": (
                self.minimum_range_probability_valid_area_km2
            ),
            "minimum_range_state_valid_area_km2": (
                self.minimum_range_state_valid_area_km2
            ),
            "minimum_range_component_samples": (
                self.minimum_range_component_samples
            ),
            "minimum_range_echo_objects": self.minimum_range_echo_objects,
            "minimum_range_state_echo_objects": (
                self.minimum_range_state_echo_objects
            ),
            "minimum_range_classifier_ood_cases": self.minimum_range_classifier_ood_cases,
            "minimum_range_classifier_ood_abstention_fraction": self.minimum_range_classifier_ood_abstention_fraction,
            "require_all_registered_regimes_certified": self.require_all_registered_regimes_certified,
            "minimum_bootstrap_tail_replicates": (
                self.minimum_bootstrap_tail_replicates
            ),
            "maximum_exact_sign_clusters": self.maximum_exact_sign_clusters,
            "minimum_deployment_metric_cell_events": (
                self.minimum_deployment_metric_cell_events
            ),
            "minimum_continuous_metric_cell_events": (
                self.minimum_continuous_metric_cell_events
            ),
            "required_range_issuance": [
                item.payload for item in self.required_range_issuance
            ],
        })


@dataclass(frozen=True)
class PromotionSampleSizePreflight:
    """Best-case independent-event requirement before holdout scoring starts."""

    family_size: int
    available_physical_events: int
    minimum_structural_events: int
    minimum_perfect_success_events: int
    minimum_zero_failure_events: int
    required_physical_events: int
    metric_cell_event_counts: tuple[
        tuple[str, str, str, int, int, int], ...
    ]
    issuance_cell_event_counts: tuple[
        tuple[str, str, int, int, int], ...
    ]
    cell_feasible: bool
    feasible: bool
    contract: str = "neural-prior-promotion-sample-size-preflight-v2"
    preflight_digest: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.family_size,
            self.available_physical_events,
            self.minimum_structural_events,
            self.minimum_perfect_success_events,
            self.minimum_zero_failure_events,
            self.required_physical_events,
        )
        if (
            self.contract
            != "neural-prior-promotion-sample-size-preflight-v2"
            or any(type(value) is not int or value < 0 for value in values)
            or self.family_size == 0
            or self.required_physical_events
            != max(
                self.minimum_structural_events,
                self.minimum_perfect_success_events,
                self.minimum_zero_failure_events,
            )
            or len({item[:4] for item in self.metric_cell_event_counts})
            != len(self.metric_cell_event_counts)
            or len({item[:3] for item in self.issuance_cell_event_counts})
            != len(self.issuance_cell_event_counts)
            or any(
                len(item) != 6
                or not all(item[index] for index in range(3))
                or type(item[3]) is not int
                or item[3] <= 0
                or type(item[4]) is not int
                or item[4] < 0
                or type(item[5]) is not int
                or item[5] <= 0
                for item in self.metric_cell_event_counts
            )
            or any(
                len(item) != 5
                or not item[0]
                or not item[1]
                or type(item[2]) is not int
                or item[2] <= 0
                or type(item[3]) is not int
                or item[3] < 0
                or type(item[4]) is not int
                or item[4] <= 0
                for item in self.issuance_cell_event_counts
            )
            or self.cell_feasible
            != all(
                item[-2] >= item[-1]
                for item in (
                    *self.metric_cell_event_counts,
                    *self.issuance_cell_event_counts,
                )
            )
            or self.feasible
            != (
                self.available_physical_events >= self.required_physical_events
                and self.cell_feasible
            )
        ):
            raise ValueError("promotion sample-size preflight is invalid")
        object.__setattr__(
            self,
            "preflight_digest",
            json_digest(self.payload),
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "preflight_digest"
        }


def _legacy_promotion_audit_digest(
    promotion_evidence_digest: str,
    payload_json: str,
    *,
    original_contract: str,
    audit_contract: str,
) -> str:
    _require_digest("legacy promotion evidence digest", promotion_evidence_digest)
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid legacy promotion evidence payload") from error
    if not isinstance(payload, dict) or payload.get("contract") != original_contract:
        raise ValueError("invalid legacy promotion evidence")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if canonical != payload_json:
        raise ValueError("legacy promotion evidence is not canonical")
    if json_digest(payload) != promotion_evidence_digest:
        raise ValueError("legacy promotion evidence digest mismatch")
    return json_digest(
        {
            "contract": audit_contract,
            "promotion_evidence_digest": promotion_evidence_digest,
            "payload": payload,
        }
    )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV3:
    """Original v3 decision; auditable but never reusable for promotion."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v3"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v3",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV4:
    """Original v4 decision; auditable but never reusable for promotion."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v4"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v4",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV5:
    """Original v5 decision retained before simultaneous inference."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v5"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v5",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV6:
    """Original v6 decision retained before prior-state score closure."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v6"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v6",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV7:
    """Original v7 decision retained before state-head calibration."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v7"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v7",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV8:
    """Original v8 decision retained before classifier holdout closure."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v8"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v8",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV9:
    """Original v9 decision retained before range-set certification."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v9"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v9",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV10:
    """Original v10 decision retained before band/classifier inference closure."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v10"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v10",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV11:
    """Original v11 decision retained before event-level band inference."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v11"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v11",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV12:
    """Original v12 decision retained before physical geometry eligibility."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v12"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v12",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV13:
    """Original v13 decision retained before metric-cell inference."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v13"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v13",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV14:
    """Original v14 decision retained before end-to-end metric-cell inference."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v14"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v14",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV15:
    """Original v15 decision retained before local issuance and preflight."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v15"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v15",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class LegacyNeuralPriorPromotionEvidenceAuditV16:
    """Original v16 decision retained before sealed scoring artifacts."""

    promotion_evidence_digest: str
    payload_json: str
    contract: str = "legacy-neural-prior-promotion-evidence-audit-v16"
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_digest",
            _legacy_promotion_audit_digest(
                self.promotion_evidence_digest,
                self.payload_json,
                original_contract="neural-prior-promotion-evidence-v16",
                audit_contract=self.contract,
            ),
        )


@dataclass(frozen=True)
class NeuralPriorPromotionEvidence:
    candidate_prior_digest: str
    parent_prior_digest: str
    candidate_manifest_digest: str
    policy_digest: str
    trust_store_digest: str
    scoring_artifact_digest: str
    scoring_process_log_digest: str
    scoring_completion_receipt_digest: str
    evaluation_digests: tuple[str, ...]
    holdout_case_count: int
    material_case_count: int
    distinct_case_count: int
    distinct_storm_count: int
    distinct_day_count: int
    distinct_radar_count: int
    distinct_regime_count: int
    distinct_range_regime_count: int
    beneficial_fraction: float
    beneficial_fraction_lower_bound: float
    harmful_fraction: float
    harmful_fraction_upper_bound: float
    mean_normalized_improvement: float
    mean_improvement_lower_bound: float
    maximum_normalized_degradation: float
    prior_echo_intensity_nll_increase_upper_bound: float
    prior_support_brier_increase_upper_bound: float
    prior_echo_support_miss_increase_upper_bound: float
    prior_echo_object_miss_increase_upper_bound: float
    prior_clear_sky_false_echo_increase_upper_bound: float
    prior_conditional_underdispersion_increase_upper_bound: float
    state_gaussian_nll_increase_upper_bound: float
    state_underdispersion_increase_upper_bound: float
    state_support_brier_increase_upper_bound: float
    state_echo_support_miss_increase_upper_bound: float
    state_echo_object_miss_increase_upper_bound: float
    state_false_support_increase_upper_bound: float
    state_valid_brier_increase_upper_bound: float
    deployment_regime_classifier_digest: str
    deployment_regime_classifier_manifest_digest: str
    classifier_family_size: int
    regime_classifier_evidence_digests: tuple[str, ...]
    regime_classifier_accuracy: float
    regime_classifier_accuracy_lower_bound: float
    minimum_regime_classifier_recall: float
    minimum_regime_classifier_recall_lower_bound: float
    regime_classifier_calibration_error: float
    regime_classifier_false_routing_fraction: float
    regime_classifier_false_routing_upper_bound: float
    regime_classifier_cluster_count: int
    regime_classifier_ood_case_count: int
    regime_classifier_ood_abstention_fraction: float
    regime_classifier_validated: bool
    range_set_precision: float
    range_set_precision_lower_bound: float
    range_set_recall: float
    range_set_recall_lower_bound: float
    range_exact_set_accuracy: float
    range_false_active_band_fraction: float
    range_false_active_band_upper_bound: float
    range_classifier_ood_case_count: int
    range_classifier_ood_abstention_fraction: float
    minimum_classifier_weather_margin: float
    minimum_classifier_range_margin: float
    prior_echo_component_status: PriorComponentStatus
    prior_clear_sky_component_status: PriorComponentStatus
    prior_echo_case_count: int
    prior_clear_sky_case_count: int
    prior_echo_cluster_count: int
    prior_clear_sky_cluster_count: int
    simultaneous_inference_test_count: int
    simultaneous_inference_method: Literal[
        "exact_sign_enumeration", "studentized_multiplier"
    ]
    simultaneous_inference_effective_replicates: int
    simultaneous_inference_critical_quantile: float
    simultaneous_inference_monte_carlo_standard_error: float
    simultaneous_inference_tail_replicates: float
    cluster_bootstrap_tail_replicates: float
    rate_inference_method: Literal[
        "event-fractional-empirical-bernstein-v1"
    ]
    range_band_skill_bounds: tuple[
        tuple[str, str, float, float, float], ...
    ]
    range_band_skill_inference_diagnostics: tuple[
        tuple[str, str, str, int, int, float, float, float, int, int], ...
    ]
    range_metric_cell_bounds: tuple[
        tuple[str, str, str, int, float, float, int, int], ...
    ]
    range_metric_end_to_end_cell_bounds: tuple[
        tuple[
            str,
            str,
            str,
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            int,
            int,
        ], ...
    ]
    range_issuance_cell_bounds: tuple[
        tuple[str, str, int, float, float, float, float, int, int], ...
    ]
    metric_cell_test_count: int
    metric_cell_inference_method: Literal[
        "joint-common-end-to-end-operational-issuance-bounded-event-inference-v2"
    ]
    metric_cell_effective_replicates: int
    metric_cell_tail_replicates: float
    metric_cell_critical_quantile: float
    metric_cell_monte_carlo_standard_error: float
    sample_size_preflight_digest: str
    sample_size_available_physical_events: int
    sample_size_required_physical_events: int
    sample_size_cell_feasible: bool
    sample_size_preflight_feasible: bool
    certified_applicability_regime_groups: tuple[tuple[str, str], ...]
    certified_range_geometry_contract_digests: tuple[str, ...]
    requires_parent_fallback_outside_certified_applicability: bool
    state_calibration_eligible: bool
    deployment_eligible: bool
    eligible: bool
    rejection_reasons: tuple[PromotionRejectionReason, ...]
    contract: str = "neural-prior-promotion-evidence-v17"
    promotion_evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-promotion-evidence-v17":
            raise ValueError("unsupported neural-prior promotion evidence")
        for name in (
            "candidate_prior_digest",
            "parent_prior_digest",
            "candidate_manifest_digest",
            "policy_digest",
            "trust_store_digest",
            "scoring_artifact_digest",
            "scoring_process_log_digest",
            "scoring_completion_receipt_digest",
            "deployment_regime_classifier_digest",
            "deployment_regime_classifier_manifest_digest",
            "sample_size_preflight_digest",
        ):
            _require_digest(name, getattr(self, name))
        for digest in self.evaluation_digests:
            _require_digest("promotion member digest", digest)
        for digest in self.regime_classifier_evidence_digests:
            _require_digest("regime classifier member digest", digest)
        for digest in self.certified_range_geometry_contract_digests:
            _require_digest("certified range geometry", digest)
        if len(self.regime_classifier_evidence_digests) != self.holdout_case_count:
            raise ValueError("regime classifier evidence count disagrees")
        if self.holdout_case_count != len(self.evaluation_digests):
            raise ValueError("promotion evidence member counts disagree")
        counts = (
            self.material_case_count,
            self.distinct_case_count,
            self.distinct_storm_count,
            self.distinct_day_count,
            self.distinct_radar_count,
            self.distinct_regime_count,
            self.distinct_range_regime_count,
            self.prior_echo_case_count,
            self.prior_clear_sky_case_count,
            self.prior_echo_cluster_count,
            self.prior_clear_sky_cluster_count,
        )
        if any(type(value) is not int or value < 0 for value in counts) or any(
            value > self.holdout_case_count for value in counts
        ):
            raise ValueError("promotion evidence counts are invalid")
        if (
            type(self.simultaneous_inference_test_count) is not int
            or self.simultaneous_inference_test_count <= 0
            or self.prior_echo_component_status
            not in ("available", "not_applicable")
            or self.prior_clear_sky_component_status
            not in ("available", "not_applicable")
            or (self.prior_echo_case_count > 0)
            != (self.prior_echo_component_status == "available")
            or (self.prior_clear_sky_case_count > 0)
            != (self.prior_clear_sky_component_status == "available")
        ):
            raise ValueError("promotion component evidence is invalid")
        fractions = (
            self.beneficial_fraction,
            self.beneficial_fraction_lower_bound,
            self.harmful_fraction,
            self.harmful_fraction_upper_bound,
            self.regime_classifier_accuracy,
            self.regime_classifier_accuracy_lower_bound,
            self.minimum_regime_classifier_recall,
            self.minimum_regime_classifier_recall_lower_bound,
            self.regime_classifier_calibration_error,
            self.regime_classifier_false_routing_fraction,
            self.regime_classifier_false_routing_upper_bound,
            self.regime_classifier_ood_abstention_fraction,
            self.range_set_precision,
            self.range_set_precision_lower_bound,
            self.range_set_recall,
            self.range_set_recall_lower_bound,
            self.range_exact_set_accuracy,
            self.range_false_active_band_fraction,
            self.range_false_active_band_upper_bound,
            self.range_classifier_ood_abstention_fraction,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("promotion evidence fractions are invalid")
        if not all(
            math.isfinite(value)
            for value in (
                self.mean_normalized_improvement,
                self.mean_improvement_lower_bound,
            )
        ):
            raise ValueError("promotion mean summaries are invalid")
        if not math.isfinite(self.maximum_normalized_degradation) or (
            self.maximum_normalized_degradation < 0.0
        ):
            raise ValueError("promotion maximum degradation is invalid")
        if any(
            not math.isfinite(value)
            for value in (
                self.prior_echo_intensity_nll_increase_upper_bound,
                self.prior_support_brier_increase_upper_bound,
                self.prior_echo_support_miss_increase_upper_bound,
                self.prior_echo_object_miss_increase_upper_bound,
                self.prior_clear_sky_false_echo_increase_upper_bound,
                self.prior_conditional_underdispersion_increase_upper_bound,
                self.state_gaussian_nll_increase_upper_bound,
                self.state_underdispersion_increase_upper_bound,
                self.state_support_brier_increase_upper_bound,
                self.state_echo_support_miss_increase_upper_bound,
                self.state_echo_object_miss_increase_upper_bound,
                self.state_false_support_increase_upper_bound,
                self.state_valid_brier_increase_upper_bound,
            )
        ):
            raise ValueError("promotion uncertainty bounds are invalid")
        if (
            self.simultaneous_inference_method
            not in ("exact_sign_enumeration", "studentized_multiplier")
            or type(self.simultaneous_inference_effective_replicates) is not int
            or self.simultaneous_inference_effective_replicates <= 0
            or not math.isfinite(self.simultaneous_inference_critical_quantile)
            or not math.isfinite(
                self.simultaneous_inference_monte_carlo_standard_error
            )
            or self.simultaneous_inference_monte_carlo_standard_error < 0.0
            or not math.isfinite(self.simultaneous_inference_tail_replicates)
            or self.simultaneous_inference_tail_replicates < 0.0
            or not math.isfinite(self.cluster_bootstrap_tail_replicates)
            or self.cluster_bootstrap_tail_replicates < 0.0
            or self.rate_inference_method
            != "event-fractional-empirical-bernstein-v1"
            or self.requires_parent_fallback_outside_certified_applicability
            is not True
            or len(set(self.certified_applicability_regime_groups))
            != len(self.certified_applicability_regime_groups)
            or len(set(self.certified_range_geometry_contract_digests))
            != len(self.certified_range_geometry_contract_digests)
            or len({item[:2] for item in self.range_band_skill_bounds})
            != len(self.range_band_skill_bounds)
            or len({item[:4] for item in self.range_metric_cell_bounds})
            != len(self.range_metric_cell_bounds)
            or len(
                {item[:4] for item in self.range_metric_end_to_end_cell_bounds}
            )
            != len(self.range_metric_end_to_end_cell_bounds)
            or {item[:4] for item in self.range_metric_end_to_end_cell_bounds}
            != {item[:4] for item in self.range_metric_cell_bounds}
            or len({item[:3] for item in self.range_issuance_cell_bounds})
            != len(self.range_issuance_cell_bounds)
            or len(self.range_band_skill_inference_diagnostics)
            != len(self.range_band_skill_bounds)
            or {
                item[:2] for item in self.range_band_skill_inference_diagnostics
            }
            != {item[:2] for item in self.range_band_skill_bounds}
            or any(
                len(item) != 5
                or not item[0]
                or not item[1]
                or not all(math.isfinite(value) for value in item[2:])
                for item in self.range_band_skill_bounds
            )
            or any(
                len(item) != 10
                or not item[0]
                or not item[1]
                or item[2] != "cluster_bootstrap"
                or type(item[3]) is not int
                or item[3] <= 0
                or type(item[4]) is not int
                or item[4] <= 0
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in item[5:8]
                )
                or type(item[8]) is not int
                or item[8] < 0
                or type(item[9]) is not int
                or item[9] < 0
                for item in self.range_band_skill_inference_diagnostics
            )
            or any(
                len(item) != 8
                or not all(item[index] for index in range(3))
                or type(item[3]) is not int
                or item[3] <= 0
                or not math.isfinite(item[4])
                or not math.isfinite(item[5])
                or not 0.0 <= item[5] <= 1.0
                or type(item[6]) is not int
                or item[6] <= 0
                or type(item[7]) is not int
                or item[7] <= 0
                for item in self.range_metric_cell_bounds
            )
            or any(
                len(item) != 12
                or not all(item[index] for index in range(3))
                or type(item[3]) is not int
                or item[3] <= 0
                or any(not math.isfinite(value) for value in item[4:10])
                or not 0.0 <= item[5] <= 1.0
                or not 0.0 <= item[6] <= 1.0
                or not 0.0 <= item[7] <= 1.0
                or not 0.0 <= item[8] <= 1.0
                or not 0.0 <= item[9] <= 1.0
                or type(item[10]) is not int
                or item[10] <= 0
                or type(item[11]) is not int
                or item[11] <= 0
                for item in self.range_metric_end_to_end_cell_bounds
            )
            or any(
                len(item) != 9
                or not item[0]
                or not item[1]
                or type(item[2]) is not int
                or item[2] <= 0
                or any(
                    not math.isfinite(value) or not 0.0 <= value <= 1.0
                    for value in item[3:7]
                )
                or type(item[7]) is not int
                or item[7] <= 0
                or type(item[8]) is not int
                or item[8] <= 0
                for item in self.range_issuance_cell_bounds
            )
            or any(
                not regime or not range_regime
                for regime, range_regime in self.certified_applicability_regime_groups
            )
        ):
            raise ValueError("promotion simultaneous-inference evidence is invalid")
        if (
            type(self.metric_cell_test_count) is not int
            or self.metric_cell_test_count <= 0
            or self.metric_cell_inference_method
            != "joint-common-end-to-end-operational-issuance-bounded-event-inference-v2"
            or type(self.metric_cell_effective_replicates) is not int
            or self.metric_cell_effective_replicates <= 0
            or not math.isfinite(self.metric_cell_tail_replicates)
            or self.metric_cell_tail_replicates < 0.0
            or not math.isfinite(self.metric_cell_critical_quantile)
            or not math.isfinite(self.metric_cell_monte_carlo_standard_error)
            or self.metric_cell_monte_carlo_standard_error < 0.0
            or type(self.sample_size_available_physical_events) is not int
            or self.sample_size_available_physical_events < 0
            or type(self.sample_size_required_physical_events) is not int
            or self.sample_size_required_physical_events < 0
            or self.sample_size_preflight_feasible
            != (
                self.sample_size_available_physical_events
                >= self.sample_size_required_physical_events
                and self.sample_size_cell_feasible
            )
        ):
            raise ValueError("promotion metric-cell or preflight evidence is invalid")
        if self.eligible != (not self.rejection_reasons):
            raise ValueError("promotion eligibility and reasons disagree")
        state_reasons = {
            "unreliable_state_head",
            "inferior_state_head",
            "insufficient_state_calibration",
        }
        if self.state_calibration_eligible != (
            not any(reason in state_reasons for reason in self.rejection_reasons)
        ) or self.deployment_eligible != (
            self.eligible
            and self.regime_classifier_validated
            and bool(self.certified_applicability_regime_groups)
            and bool(self.certified_range_geometry_contract_digests)
            and self.sample_size_preflight_feasible
        ):
            raise ValueError("state or deployment eligibility is inconsistent")
        if (
            type(self.regime_classifier_ood_case_count) is not int
            or self.regime_classifier_ood_case_count < 0
            or type(self.range_classifier_ood_case_count) is not int
            or self.range_classifier_ood_case_count < 0
            or type(self.classifier_family_size) is not int
            or self.classifier_family_size <= 0
            or type(self.regime_classifier_cluster_count) is not int
            or self.regime_classifier_cluster_count <= 0
            or type(self.regime_classifier_validated) is not bool
            or not math.isfinite(self.minimum_classifier_weather_margin)
            or self.minimum_classifier_weather_margin < 0.0
            or not math.isfinite(self.minimum_classifier_range_margin)
            or self.minimum_classifier_range_margin < 0.0
        ):
            raise ValueError("regime classifier promotion evidence is invalid")
        object.__setattr__(self, "promotion_evidence_digest", json_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "promotion_evidence_digest"
        }


def _paired_metric_score(
    metric_names: tuple[str, ...],
    metric_change: Tensor,
    end_to_end_metric_change: Tensor,
    metric_available: Tensor,
    policy: NeuralPriorPromotionPolicy,
) -> tuple[float | None, float, bool, bool]:
    scales = {item.metric_name: item for item in policy.metric_scales}
    values: list[Tensor] = []
    weights: list[Tensor] = []
    maximum_degradation = 0.0
    metric_limit_exceeded = False
    end_to_end_limit_exceeded = False
    for index, name in enumerate(metric_names):
        if name not in scales:
            raise ValueError("promotion policy lacks an evaluation metric scale")
        item = scales[name]
        end_to_end = (
            end_to_end_metric_change[:, index].masked_select(
                metric_available[:, index]
            )
            / item.scale
        )
        if bool(torch.any(end_to_end > item.maximum_end_to_end_normalized_degradation)):
            end_to_end_limit_exceeded = True
        selected = metric_available[:, index] & (
            torch.abs(metric_change[:, index]) >= item.material_change
        )
        if not bool(torch.any(selected)):
            continue
        normalized = metric_change[:, index].masked_select(selected) / item.scale
        metric_degradation = float(torch.amax(torch.clamp(normalized, min=0)))
        maximum_degradation = max(maximum_degradation, metric_degradation)
        if metric_degradation > item.maximum_normalized_degradation:
            metric_limit_exceeded = True
        values.append(-normalized)
        weights.append(torch.full_like(normalized, item.weight))
    if not values:
        return (
            None,
            maximum_degradation,
            metric_limit_exceeded,
            end_to_end_limit_exceeded,
        )
    value = torch.cat(values)
    weight = torch.cat(weights)
    return (
        float((torch.sum(value * weight) / torch.sum(weight)).detach()),
        maximum_degradation,
        metric_limit_exceeded,
        end_to_end_limit_exceeded,
    )


def _holdout_score(
    evaluation: PriorHoldoutEvaluation,
    policy: NeuralPriorPromotionPolicy,
) -> tuple[float | None, float, bool, bool]:
    return _paired_metric_score(
        evaluation.metric_names,
        evaluation.metric_change,
        evaluation.end_to_end_metric_change,
        evaluation.metric_available,
        policy,
    )


def _cluster_bounds(
    scores: list[float],
    clusters: list[str],
    policy: NeuralPriorPromotionPolicy,
    *,
    candidate_family_size: int,
) -> tuple[float, float, float]:
    """Bound event rates with Wilson and continuous skill with resampling."""

    grouped: dict[str, list[float]] = {}
    for score, cluster in zip(scores, clusters, strict=True):
        grouped.setdefault(cluster, []).append(score)
    keys = sorted(grouped)
    cluster_scores = {
        key: sum(values) / len(values) for key, values in grouped.items()
    }
    lower_beneficial, _ = _event_fractional_rate_interval(
        [float(value > 0.0) for value in scores],
        clusters,
        policy,
        family_size=candidate_family_size,
    )
    _, upper_harmful = _event_fractional_rate_interval(
        [float(value < 0.0) for value in scores],
        clusters,
        policy,
        family_size=candidate_family_size,
    )
    generator = random.Random(0)
    means: list[float] = []
    for _ in range(policy.bootstrap_samples):
        sample = [generator.choice(keys) for _ in keys]
        values = [cluster_scores[key] for key in sample]
        means.append(sum(values) / len(values))
    alpha = (1.0 - policy.confidence_level) / (
        2.0 * candidate_family_size
    )
    def quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, int(probability * len(ordered))))]
    return lower_beneficial, upper_harmful, quantile(means, alpha)


def _bootstrap_tail_diagnostics(
    policy: NeuralPriorPromotionPolicy,
    *,
    family_size: int,
    enforce: bool = True,
) -> tuple[int, float, float, float]:
    """Validate finite-sample resolution for one bootstrap test family."""

    if family_size <= 0:
        raise ValueError("bootstrap family size must be positive")
    alpha = (1.0 - policy.confidence_level) / (2.0 * family_size)
    effective_replicates = policy.bootstrap_samples
    tail_replicates = effective_replicates * alpha
    critical_quantile = 1.0 - alpha
    monte_carlo_standard_error = math.sqrt(
        alpha * (1.0 - alpha) / effective_replicates
    )
    if (
        enforce
        and tail_replicates < policy.minimum_bootstrap_tail_replicates
    ):
        raise ValueError("insufficient bootstrap tail resolution")
    return (
        effective_replicates,
        tail_replicates,
        critical_quantile,
        monte_carlo_standard_error,
    )


def _physical_event_cluster(evaluation: PriorHoldoutEvaluation) -> str:
    """Return the signed meteorological event identity for outer resampling."""

    if not evaluation.storm_id or evaluation.storm_id.strip() != evaluation.storm_id:
        raise ValueError("physical event identity is invalid")
    _require_digest("physical event", evaluation.physical_event_digest)
    return evaluation.physical_event_digest


def _event_binary_rate_interval(
    values: Sequence[float],
    clusters: Sequence[object],
    policy: NeuralPriorPromotionPolicy,
    *,
    family_size: int,
) -> tuple[float, float]:
    """Wilson interval for one binary outcome per independent event."""

    if (
        not values
        or len(values) != len(clusters)
        or family_size <= 0
        or len(set(clusters)) != len(clusters)
        or any(value not in (0.0, 1.0) for value in values)
    ):
        raise ValueError("binary event rate evidence is invalid")
    alpha = (1.0 - policy.confidence_level) / (2.0 * family_size)
    z_score = NormalDist().inv_cdf(1.0 - alpha)
    event_count = len(values)
    rate = sum(values) / event_count
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / event_count
    center = (rate + z_squared / (2.0 * event_count)) / denominator
    half_width = (
        z_score
        * math.sqrt(
            rate * (1.0 - rate) / event_count
            + z_squared / (4.0 * event_count * event_count)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _event_fractional_rate_interval(
    values: Sequence[float],
    clusters: Sequence[object],
    policy: NeuralPriorPromotionPolicy,
    *,
    family_size: int,
) -> tuple[float, float]:
    """Empirical-Bernstein bounds for bounded within-event rates."""

    if (
        not values
        or len(values) != len(clusters)
        or family_size <= 0
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
    ):
        raise ValueError("fractional event rate evidence is invalid")
    grouped: dict[object, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        grouped.setdefault(cluster, []).append(value)
    event_rates = tuple(
        sum(items) / len(items)
        for _, items in sorted(grouped.items(), key=lambda item: repr(item[0]))
    )
    alpha = (1.0 - policy.confidence_level) / (2.0 * family_size)
    event_count = len(event_rates)
    rate = sum(event_rates) / event_count
    if event_count == 1:
        radius = 1.0
    else:
        variance = sum((value - rate) ** 2 for value in event_rates) / (
            event_count - 1
        )
        log_term = math.log(2.0 / alpha)
        radius = math.sqrt(
            2.0 * variance * log_term / event_count
        ) + 7.0 * log_term / (3.0 * (event_count - 1))
    return max(0.0, rate - radius), min(1.0, rate + radius)


def promotion_sample_size_preflight(
    plan: NeuralPriorHoldoutPlan,
    policy: NeuralPriorPromotionPolicy,
    *,
    available_physical_events: int,
    metric_cell_event_counts: tuple[
        tuple[str, str, str, int, int, int], ...
    ] | None = None,
    issuance_cell_event_counts: tuple[
        tuple[str, str, int, int, int], ...
    ] | None = None,
) -> PromotionSampleSizePreflight:
    """Reject holdouts that cannot pass rate gates even with perfect outcomes."""

    validate_neural_prior_holdout_plan(plan)
    if type(available_physical_events) is not int or available_physical_events < 0:
        raise ValueError("available physical-event count must be nonnegative")
    base_family_size = len(plan.candidate_family_digests) * len(
        plan.regime_classifier_manifests
    )
    family_size = base_family_size * max(
        1,
        4 * len(policy.required_range_metrics)
        + 4 * len(policy.required_range_issuance),
    )
    alpha = (1.0 - policy.confidence_level) / (2.0 * family_size)
    finite_sample_constant = 7.0 * math.log(2.0 / alpha) / 3.0

    success_threshold = max(
        policy.minimum_regime_classifier_accuracy_lower_bound,
        policy.minimum_regime_classifier_recall_lower_bound,
        policy.minimum_range_set_precision_lower_bound,
        policy.minimum_range_set_recall_lower_bound,
    )
    failure_thresholds = (
        policy.maximum_regime_classifier_false_routing_upper_bound,
        policy.maximum_false_active_band_upper_bound,
        policy.maximum_harmful_fraction,
        *(item.maximum_harmful_fraction_upper_bound for item in policy.required_range_metrics),
        *(
            item.maximum_end_to_end_harmful_fraction_upper_bound
            for item in policy.required_range_metrics
        ),
        *(
            item.maximum_withdrawn_fraction
            for item in policy.required_range_issuance
        ),
        *(
            item.maximum_newly_issued_fraction
            for item in policy.required_range_issuance
        ),
        *(
            item.maximum_background_fallback_increase
            for item in policy.required_range_issuance
        ),
        *(
            item.maximum_confidence_weighted_coverage_loss
            for item in policy.required_range_issuance
        ),
    )
    if any(value <= 0.0 for value in failure_thresholds):
        raise ValueError(
            "finite-sample rate policy requires an unattainable exact bound"
        )
    success_margin = 1.0 - success_threshold
    failure_margin = min(failure_thresholds)

    def required_events(margin: float) -> int:
        if margin <= 0.0:
            raise ValueError(
                "finite-sample rate policy requires an unattainable exact bound"
            )
        return max(2, math.ceil(finite_sample_constant / margin) + 1)

    minimum_perfect_success_events = required_events(success_margin)
    minimum_zero_failure_events = required_events(failure_margin)
    minimum_structural_events = max(
        policy.minimum_material_clusters,
        policy.minimum_regime_classifier_clusters,
        policy.minimum_range_band_clusters,
        policy.minimum_deployment_metric_cell_events,
        policy.minimum_continuous_metric_cell_events,
        *(item.minimum_physical_events for item in policy.required_range_metrics),
        *(item.minimum_physical_events for item in policy.required_range_issuance),
    )
    required_physical_events = max(
        minimum_structural_events,
        minimum_perfect_success_events,
        minimum_zero_failure_events,
    )
    if metric_cell_event_counts is None:
        metric_cell_event_counts = tuple(
            (
                item.weather_regime,
                item.range_regime,
                item.metric_name,
                item.lead_minutes,
                available_physical_events,
                max(
                    item.minimum_physical_events,
                    policy.minimum_deployment_metric_cell_events,
                    policy.minimum_continuous_metric_cell_events,
                ),
            )
            for item in policy.required_range_metrics
        )
    if issuance_cell_event_counts is None:
        issuance_cell_event_counts = tuple(
            (
                item.weather_regime,
                item.range_regime,
                item.lead_minutes,
                available_physical_events,
                item.minimum_physical_events,
            )
            for item in policy.required_range_issuance
        )
    cell_feasible = all(
        item[-2] >= item[-1]
        for item in (*metric_cell_event_counts, *issuance_cell_event_counts)
    )
    return PromotionSampleSizePreflight(
        family_size=family_size,
        available_physical_events=available_physical_events,
        minimum_structural_events=minimum_structural_events,
        minimum_perfect_success_events=minimum_perfect_success_events,
        minimum_zero_failure_events=minimum_zero_failure_events,
        required_physical_events=required_physical_events,
        metric_cell_event_counts=metric_cell_event_counts,
        issuance_cell_event_counts=issuance_cell_event_counts,
        cell_feasible=cell_feasible,
        feasible=(
            available_physical_events >= required_physical_events
            and cell_feasible
        ),
    )


def _bounded_event_mean_upper_bound(
    values: list[float],
    clusters: list[str],
    policy: NeuralPriorPromotionPolicy,
    *,
    family_size: int,
    absolute_bound: float,
) -> float:
    """Finite-sample empirical-Bernstein UCB for bounded event means."""

    if (
        not values
        or len(values) != len(clusters)
        or family_size <= 0
        or any(not math.isfinite(value) for value in values)
        or not math.isfinite(absolute_bound)
        or absolute_bound <= 0.0
        or any(abs(value) > absolute_bound for value in values)
    ):
        raise ValueError("bounded event mean evidence is invalid")
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        grouped.setdefault(cluster, []).append(value)
    event_means = tuple(
        sum(items) / len(items) for _, items in sorted(grouped.items())
    )
    alpha = (1.0 - policy.confidence_level) / (2.0 * family_size)
    event_count = len(event_means)
    observed = sum(event_means) / event_count
    if event_count == 1:
        return absolute_bound
    variance = sum((value - observed) ** 2 for value in event_means) / (
        event_count - 1
    )
    log_term = math.log(3.0 / alpha)
    radius = math.sqrt(2.0 * variance * log_term / event_count) + (
        6.0 * absolute_bound * log_term / event_count
    )
    return min(absolute_bound, observed + radius)


@dataclass(frozen=True)
class _UncertaintyComparison:
    component: _UncertaintyComponent
    group: tuple[str, str] | None
    values: tuple[float, ...]
    clusters: tuple[str, ...]


def _cluster_means(
    comparison: _UncertaintyComparison,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(
        comparison.values,
        comparison.clusters,
        strict=True,
    ):
        grouped.setdefault(cluster, []).append(value)
    return {
        key: sum(items) / len(items)
        for key, items in grouped.items()
    }


@dataclass(frozen=True)
class _SimultaneousInferenceResult:
    bounds: dict[str, float]
    comparison_bounds: dict[tuple[str, tuple[str, str] | None], float]
    test_count: int
    method: Literal["exact_sign_enumeration", "studentized_multiplier"]
    effective_replicates: int
    critical_quantile: float
    monte_carlo_standard_error: float
    tail_replicates: float


def _simultaneous_uncertainty_upper_bounds(
    comparisons: tuple[_UncertaintyComparison, ...],
    policy: NeuralPriorPromotionPolicy,
    *,
    candidate_family_size: int,
) -> _SimultaneousInferenceResult:
    """Studentized shared-cluster max-stat UCBs across all score families."""

    if not comparisons:
        raise ValueError("simultaneous inference requires comparisons")
    retained = tuple((item, _cluster_means(item)) for item in comparisons)
    if any(not values for _, values in retained):
        raise ValueError("simultaneous inference comparison has no clusters")
    observed = tuple(
        sum(values.values()) / len(values)
        for _, values in retained
    )
    standard_errors = tuple(
        math.sqrt(
            sum((value - mean) ** 2 for value in values.values())
            / (len(values) * (len(values) - 1))
        )
        if len(values) > 1
        else 0.0
        for (_, values), mean in zip(retained, observed, strict=True)
    )
    cluster_universe = tuple(
        sorted({key for _, values in retained for key in values})
    )
    if len(cluster_universe) <= policy.maximum_exact_sign_clusters:
        multiplier_rows = product((-1.0, 1.0), repeat=len(cluster_universe))
        method: Literal[
            "exact_sign_enumeration", "studentized_multiplier"
        ] = "exact_sign_enumeration"
        effective_replicates = 2 ** len(cluster_universe)
    else:
        generator = random.Random(1)
        multiplier_rows = (
            tuple(
                -1.0 if generator.randrange(2) == 0 else 1.0
                for _ in cluster_universe
            )
            for _ in range(policy.bootstrap_samples)
        )
        method = "studentized_multiplier"
        effective_replicates = policy.bootstrap_samples
    maximum_statistics: list[float] = []
    for row in multiplier_rows:
        multipliers = dict(zip(cluster_universe, row, strict=True))
        studentized: list[float] = []
        for index, (_, cluster_values) in enumerate(retained):
            centered = sum(
                multipliers[key] * (value - observed[index])
                for key, value in cluster_values.items()
            ) / len(cluster_values)
            standard_error = standard_errors[index]
            studentized.append(
                centered / standard_error if standard_error > 0.0 else 0.0
            )
        maximum_statistics.append(max(studentized))
    alpha = (1.0 - policy.confidence_level) / (2.0 * candidate_family_size)
    ordered = sorted(maximum_statistics)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil((1.0 - alpha) * len(ordered)) - 1),
    )
    critical = ordered[index]
    bounds: dict[str, float] = {}
    comparison_bounds: dict[
        tuple[str, tuple[str, str] | None], float
    ] = {}
    for (comparison, _), mean, standard_error in zip(
        retained,
        observed,
        standard_errors,
        strict=True,
    ):
        bound = mean + critical * standard_error
        comparison_bounds[(comparison.component, comparison.group)] = bound
        bounds[comparison.component] = max(
            bounds.get(comparison.component, -math.inf),
            bound,
        )
    tail_replicates = effective_replicates * alpha
    monte_carlo_error = (
        0.0
        if method == "exact_sign_enumeration"
        else math.sqrt(alpha * (1.0 - alpha) / effective_replicates)
    )
    return _SimultaneousInferenceResult(
        bounds=bounds,
        comparison_bounds=comparison_bounds,
        test_count=candidate_family_size * len(comparisons),
        method=method,
        effective_replicates=effective_replicates,
        critical_quantile=1.0 - alpha,
        monte_carlo_standard_error=monte_carlo_error,
        tail_replicates=tail_replicates,
    )


def compute_neural_prior_promotion(
    manifest: NeuralPriorCandidateManifest,
    plan: NeuralPriorHoldoutPlan,
    evaluations: tuple[PriorHoldoutEvaluation, ...],
    *,
    policy: NeuralPriorPromotionPolicy,
    policy_trust_store_path: str | Path,
    scoring_artifact: HoldoutScoringArtifact | None = None,
    scoring_process_log: ProcessLogArtifact | None = None,
    scoring_completion_receipt: TrustedProcessCompletionReceipt | None = None,
) -> NeuralPriorPromotionEvidence:
    """Evaluate a manifested candidate on independent, forecast-derived cases."""

    if any(type(item) is not PriorHoldoutEvaluation for item in evaluations):
        raise ValueError(
            "legacy promotion evaluations are audit-only and cannot be reused"
        )
    if (
        scoring_artifact is None
        or scoring_process_log is None
        or scoring_completion_receipt is None
    ):
        raise ValueError("promotion requires sealed canonical scoring artifacts")
    validate_neural_prior_holdout_plan(plan)
    validate_neural_prior_candidate_manifest(manifest)
    validate_holdout_scoring_artifact(
        scoring_artifact,
        manifest,
        plan,
        evaluations,
    )
    validate_trusted_process_completion_receipt(
        scoring_completion_receipt,
        manifest.candidate_scoring_start_receipt,
    )
    validate_process_log_artifact(scoring_process_log)
    if (
        scoring_process_log.process_kind != "candidate_scoring"
        or scoring_process_log.start_receipt_digest
        != manifest.candidate_scoring_start_receipt.receipt_digest
        or scoring_process_log.artifact_digest
        != scoring_completion_receipt.process_log_digest
        or scoring_artifact.artifact_digest
        != scoring_completion_receipt.output_artifact_digest
    ):
        raise ValueError("scoring completion does not seal its canonical artifacts")
    _validate_physical_event_catalogs_against_plan(manifest, plan)
    if (
        manifest.training_physical_event_catalog_plan.association_algorithm_digest
        != plan.physical_event_catalog_plan.association_algorithm_digest
        or manifest.training_physical_event_catalog_plan.adjudication_policy_digest
        != plan.physical_event_catalog_plan.adjudication_policy_digest
    ):
        raise ValueError("candidate training event catalog is incompatible")
    trust = _load_learning_policy_trust_store(policy_trust_store_path)
    reasons: list[PromotionRejectionReason] = []
    classifier_manifests = {
        item.manifest_digest: item for item in plan.regime_classifier_manifests
    }
    selected_classifier_manifest = classifier_manifests.get(
        policy.deployment_regime_classifier_manifest_digest
    )
    if (
        selected_classifier_manifest is None
        or selected_classifier_manifest.classifier_digest
        != policy.deployment_regime_classifier_digest
    ):
        raise ValueError("deployment classifier was not preregistered")
    _validate_classifier_holdout_independence(
        selected_classifier_manifest,
        manifest.holdout_cases,
    )
    family_size = len(plan.candidate_family_digests) * len(
        plan.regime_classifier_manifests
    )

    def classified_groups(
        evaluation: PriorHoldoutEvaluation,
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (evaluation.classified_regime, item.range_regime)
            for item in evaluation.range_band_evaluations
        )
    if policy.digest not in trust.approved_policy_digests:
        reasons.append("unapproved_promotion_policy")
    if manifest.manifest_digest not in policy.approved_candidate_manifest_digests:
        reasons.append("unapproved_candidate_manifest")
    if (
        manifest.physical_event_catalog_result.result_digest
        != policy.approved_physical_event_catalog_result_digest
    ):
        raise ValueError("promotion used an unapproved event catalog result")
    if manifest.holdout_plan_digest != plan.plan_digest or (
        plan.plan_digest not in policy.approved_holdout_plan_digests
    ):
        reasons.append("unapproved_holdout_plan")
    reference_plans = {
        item.plan_digest: item for item in plan.regime_reference_plans
    }
    reference_evidences = {
        item.evidence_digest: item
        for item in manifest.regime_reference_evidences
    }
    for case in manifest.holdout_cases:
        reference_plan = reference_plans.get(case.regime_reference_plan_digest)
        reference_evidence = reference_evidences.get(
            case.regime_reference_evidence_digest
        )
        if reference_plan is None or reference_evidence is None:
            raise ValueError("regime-reference lineage is incomplete")
        validate_regime_reference_evidence(reference_evidence, reference_plan)
    planned_case_ids = {item.case_id for item in plan.cases}
    manifested_case_ids = {item.case_id for item in manifest.holdout_cases}
    evaluated_case_ids = {item.case_id for item in evaluations}
    if manifested_case_ids != planned_case_ids:
        raise ValueError("candidate manifest must complete every planned case")
    if evaluated_case_ids != planned_case_ids or len(evaluations) != len(plan.cases):
        raise ValueError("promotion requires one outcome for every planned case")
    if len(evaluations) < policy.minimum_holdout_cases:
        reasons.append("insufficient_holdout_cases")
    scores: list[float] = []
    clusters: list[str] = []
    material_evaluations: list[PriorHoldoutEvaluation] = []
    maximum_degradation = 0.0
    uncertainty_records: dict[
        _UncertaintyComponent,
        list[tuple[PriorHoldoutEvaluation, float, str]],
    ] = {
        "intensity": [],
        "pit_residual": [],
        "support": [],
        "echo_miss": [],
        "object_miss": [],
        "clear": [],
        "underdispersion": [],
        "state_nll": [],
        "state_pit_residual": [],
        "state_underdispersion": [],
        "state_support": [],
        "state_echo_miss": [],
        "state_object_miss": [],
        "state_false_support": [],
        "state_valid": [],
    }
    for evaluation in evaluations:
        if evaluation.evaluation_digest != _evaluation_digest(evaluation):
            raise ValueError("prior holdout evaluation digest mismatch")
        if (
            evaluation.candidate_manifest_digest != manifest.manifest_digest
            or evaluation.holdout_plan_digest != plan.plan_digest
            or evaluation.candidate_prior_digest != manifest.candidate_prior_digest
            or evaluation.parent_prior_digest != manifest.parent_prior_digest
        ):
            raise ValueError("evaluation does not belong to the candidate manifest")
        if (
            evaluation.regime_classifier_digest
            != policy.deployment_regime_classifier_digest
            or evaluation.regime_classifier_manifest_digest
            != policy.deployment_regime_classifier_manifest_digest
            or evaluation.classifier_numerical_runtime_digest
            != selected_classifier_manifest.numerical_runtime_digest
        ):
            raise ValueError("holdout used a different deployment classifier")
        manifested_case = manifest.holdout_case(evaluation.case_id)
        planned_range_contract = next(
            item
            for item in plan.range_band_contracts
            if item.contract_digest == manifested_case.range_band_contract_digest
        )
        if (
            evaluation.physical_event_digest
            != manifested_case.physical_event_digest
            or any(
                band.range_geometry_contract_digest
                != planned_range_contract.range_geometry_contract_digest
                for band in evaluation.range_band_evaluations
            )
        ):
            raise ValueError("holdout physical applicability lineage disagrees")
        uncertainty_cluster = _physical_event_cluster(evaluation)
        uncertainty_records["support"].append(
            (
                evaluation,
                evaluation.prior_support_brier_score
                - evaluation.parent_prior_support_brier_score,
                uncertainty_cluster,
            )
        )
        uncertainty_records["state_nll"].append(
            (
                evaluation,
                evaluation.state_candidate_gaussian_nll
                - evaluation.state_parent_gaussian_nll,
                uncertainty_cluster,
            )
        )
        uncertainty_records["state_pit_residual"].append(
            (
                evaluation,
                evaluation.state_candidate_pit_residual_mean_abs
                - evaluation.state_parent_pit_residual_mean_abs,
                uncertainty_cluster,
            )
        )
        uncertainty_records["state_underdispersion"].append(
            (
                evaluation,
                evaluation.state_candidate_underdispersion_fraction
                - evaluation.state_parent_underdispersion_fraction,
                uncertainty_cluster,
            )
        )
        uncertainty_records["state_support"].append(
            (
                evaluation,
                evaluation.state_candidate_support_brier_score
                - evaluation.state_parent_support_brier_score,
                uncertainty_cluster,
            )
        )
        uncertainty_records["state_valid"].append(
            (
                evaluation,
                evaluation.state_candidate_valid_brier_score
                - evaluation.state_parent_valid_brier_score,
                uncertainty_cluster,
            )
        )
        if evaluation.state_candidate_echo_support_miss_score is not None:
            assert evaluation.state_parent_echo_support_miss_score is not None
            assert evaluation.state_candidate_echo_object_miss_score is not None
            assert evaluation.state_parent_echo_object_miss_score is not None
            uncertainty_records["state_echo_miss"].append(
                (
                    evaluation,
                    evaluation.state_candidate_echo_support_miss_score
                    - evaluation.state_parent_echo_support_miss_score,
                    uncertainty_cluster,
                )
            )
            uncertainty_records["state_object_miss"].append(
                (
                    evaluation,
                    evaluation.state_candidate_echo_object_miss_score
                    - evaluation.state_parent_echo_object_miss_score,
                    uncertainty_cluster,
                )
            )
        if evaluation.state_candidate_false_support_score is not None:
            assert evaluation.state_parent_false_support_score is not None
            uncertainty_records["state_false_support"].append(
                (
                    evaluation,
                    evaluation.state_candidate_false_support_score
                    - evaluation.state_parent_false_support_score,
                    uncertainty_cluster,
                )
            )
        if evaluation.prior_echo_intensity_status == "available":
            assert evaluation.prior_conditional_pit_residual_mean_abs is not None
            assert (
                evaluation.parent_prior_conditional_pit_residual_mean_abs
                is not None
            )
            uncertainty_records["pit_residual"].append(
                (
                    evaluation,
                    evaluation.prior_conditional_pit_residual_mean_abs
                    - evaluation.parent_prior_conditional_pit_residual_mean_abs,
                    uncertainty_cluster,
                )
            )
            assert evaluation.prior_echo_intensity_nll is not None
            assert evaluation.parent_prior_echo_intensity_nll is not None
            assert evaluation.prior_echo_support_miss_score is not None
            assert evaluation.parent_prior_echo_support_miss_score is not None
            assert evaluation.prior_echo_object_miss_score is not None
            assert evaluation.parent_prior_echo_object_miss_score is not None
            assert evaluation.prior_conditional_underdispersion_fraction is not None
            assert evaluation.parent_prior_conditional_underdispersion_fraction is not None
            uncertainty_records["intensity"].append(
                (
                    evaluation,
                    evaluation.prior_echo_intensity_nll
                    - evaluation.parent_prior_echo_intensity_nll,
                    uncertainty_cluster,
                )
            )
            uncertainty_records["underdispersion"].append(
                (
                    evaluation,
                    evaluation.prior_conditional_underdispersion_fraction
                    - evaluation.parent_prior_conditional_underdispersion_fraction,
                    uncertainty_cluster,
                )
            )
            uncertainty_records["echo_miss"].append(
                (
                    evaluation,
                    evaluation.prior_echo_support_miss_score
                    - evaluation.parent_prior_echo_support_miss_score,
                    uncertainty_cluster,
                )
            )
            uncertainty_records["object_miss"].append(
                (
                    evaluation,
                    evaluation.prior_echo_object_miss_score
                    - evaluation.parent_prior_echo_object_miss_score,
                    uncertainty_cluster,
                )
            )
        if evaluation.prior_clear_sky_status == "available":
            assert evaluation.prior_clear_sky_false_echo_score is not None
            assert evaluation.parent_prior_clear_sky_false_echo_score is not None
            uncertainty_records["clear"].append(
                (
                    evaluation,
                    evaluation.prior_clear_sky_false_echo_score
                    - evaluation.parent_prior_clear_sky_false_echo_score,
                    uncertainty_cluster,
                )
            )
        if evaluation.metric_contract_digest not in (
            policy.approved_metric_contract_digests
        ):
            reasons.append("unapproved_metric_contract")
        if (
            evaluation.state_calibration_sample_count
            < policy.minimum_state_calibration_samples_per_case
        ):
            reasons.append("insufficient_state_calibration")
        state_absolute_failed = (
            evaluation.state_candidate_gaussian_nll
            > policy.maximum_state_gaussian_nll
            or evaluation.state_candidate_pit_residual_mean_abs
            > policy.maximum_state_pit_residual_mean_abs
            or evaluation.state_candidate_underdispersion_fraction
            > policy.maximum_state_underdispersion_fraction
            or evaluation.state_candidate_support_brier_score
            > policy.maximum_state_support_brier_score
            or evaluation.state_candidate_valid_brier_score
            > policy.maximum_state_valid_brier_score
            or (
                evaluation.state_candidate_echo_support_miss_score is not None
                and evaluation.state_candidate_echo_support_miss_score
                > policy.maximum_state_echo_support_miss_score
            )
            or (
                evaluation.state_candidate_echo_object_miss_score is not None
                and evaluation.state_candidate_echo_object_miss_score
                > policy.maximum_state_echo_object_miss_score
            )
            or (
                evaluation.state_candidate_false_support_score is not None
                and evaluation.state_candidate_false_support_score
                > policy.maximum_state_false_support_score
            )
        )
        if state_absolute_failed:
            reasons.append("unreliable_state_head")
        if (
            evaluation.prior_uncertainty_sample_count
            < policy.minimum_prior_uncertainty_samples_per_case
            or evaluation.prior_support_brier_score
            > policy.maximum_prior_support_brier_score
        ):
            reasons.append("unreliable_prior_uncertainty")
        if evaluation.prior_echo_intensity_status == "available":
            assert evaluation.prior_conditional_pit_residual_mean_abs is not None
            assert evaluation.prior_conditional_underdispersion_fraction is not None
            assert evaluation.prior_echo_intensity_nll is not None
            assert evaluation.prior_echo_support_miss_score is not None
            assert evaluation.prior_echo_object_miss_score is not None
            if (
                evaluation.prior_conditional_pit_residual_mean_abs
                > policy.maximum_prior_conditional_pit_residual_mean_abs
                or evaluation.prior_conditional_underdispersion_fraction
                > policy.maximum_prior_conditional_underdispersion_fraction
                or evaluation.prior_echo_support_miss_score
                > policy.maximum_prior_echo_support_miss_score
                or evaluation.prior_echo_object_miss_score
                > policy.maximum_prior_echo_object_miss_score
                or (
                    evaluation.prior_echo_intensity_nll
                    + policy.prior_abstention_penalty_weight
                    * (1.0 - evaluation.prior_candidate_valid_fraction)
                )
                > policy.maximum_prior_echo_intensity_nll
            ):
                reasons.append("unreliable_prior_uncertainty")
            if (
                evaluation.prior_echo_intensity_sample_count
                < policy.minimum_prior_echo_pixels_per_case
            ):
                reasons.append("insufficient_component_samples")
            if (
                evaluation.prior_echo_area_km2
                < policy.minimum_prior_echo_area_km2_per_case
            ):
                reasons.append("insufficient_component_area")
            if (
                evaluation.prior_echo_object_count
                < policy.minimum_prior_echo_objects_per_case
            ):
                reasons.append("insufficient_echo_objects")
        if evaluation.prior_clear_sky_status == "available":
            assert evaluation.prior_clear_sky_false_echo_score is not None
            if evaluation.prior_clear_sky_false_echo_score > (
                policy.maximum_prior_clear_sky_false_echo_score
            ):
                reasons.append("unreliable_prior_uncertainty")
            if (
                evaluation.prior_clear_sky_sample_count
                < policy.minimum_prior_clear_pixels_per_case
            ):
                reasons.append("insufficient_component_samples")
            if (
                evaluation.prior_clear_sky_area_km2
                < policy.minimum_prior_clear_area_km2_per_case
            ):
                reasons.append("insufficient_component_area")
        if (
            evaluation.prior_candidate_valid_fraction
            < policy.minimum_prior_valid_fraction
            or evaluation.prior_candidate_valid_area_km2
            < policy.minimum_prior_valid_area_km2
            or evaluation.prior_abstention_increase_vs_parent
            > policy.maximum_abstention_increase_vs_parent
        ):
            reasons.append("unreliable_prior_uncertainty")
        if bool(
            torch.any(
                evaluation.coverage_parent
                - evaluation.coverage_candidate
                > policy.maximum_coverage_loss
            )
        ):
            reasons.append("excessive_single_degradation")
        if (
            bool(
                torch.any(
                    evaluation.newly_issued_fraction
                    > policy.maximum_newly_issued_fraction
                )
            )
            or bool(
                torch.any(
                    evaluation.withdrawn_fraction
                    > policy.maximum_withdrawn_fraction
                )
            )
        ):
            reasons.append("excessive_issuance_change")
        (
            score,
            degradation,
            metric_limit_exceeded,
            end_to_end_limit_exceeded,
        ) = _holdout_score(evaluation, policy)
        if metric_limit_exceeded:
            reasons.append("excessive_single_degradation")
        if end_to_end_limit_exceeded:
            reasons.append("excessive_end_to_end_degradation")
        maximum_degradation = max(maximum_degradation, degradation)
        if score is not None:
            scores.append(score)
            clusters.append(_physical_event_cluster(evaluation))
            material_evaluations.append(evaluation)
    material_count = len(scores)
    if material_count == 0:
        reasons.append("no_material_outcome")
    if material_count < policy.minimum_material_cases:
        reasons.append("insufficient_material_cases")
    if material_count / max(1, len(evaluations)) < policy.minimum_material_case_fraction:
        reasons.append("insufficient_material_case_fraction")
    cases = {item.case_id for item in material_evaluations}
    storms = {item.storm_id for item in material_evaluations}
    days = {item.day for item in material_evaluations}
    radars = {item.radar_id for item in material_evaluations}
    regimes = {item.classified_regime for item in material_evaluations}
    range_regimes = {
        range_regime
        for item in material_evaluations
        for range_regime in item.classified_range_regimes
    }
    material_clusters = set(clusters)
    if len(cases) < policy.minimum_independent_cases:
        reasons.append("insufficient_independent_cases")
    if len(storms) < policy.minimum_distinct_storms:
        reasons.append("insufficient_distinct_storms")
    if len(days) < policy.minimum_distinct_days:
        reasons.append("insufficient_distinct_days")
    if len(radars) < policy.minimum_distinct_radars:
        reasons.append("insufficient_distinct_radars")
    if len(regimes) < policy.minimum_distinct_regimes:
        reasons.append("insufficient_distinct_regimes")
    if len(range_regimes) < policy.minimum_distinct_range_regimes:
        reasons.append("insufficient_distinct_range_regimes")
    if len(material_clusters) < policy.minimum_material_clusters:
        reasons.append("insufficient_material_clusters")
    if scores:
        beneficial = sum(value > 0 for value in scores) / len(scores)
        harmful = sum(value < 0 for value in scores) / len(scores)
        mean = sum(scores) / len(scores)
        lower_beneficial, upper_harmful, lower_mean = _cluster_bounds(
            scores,
            clusters,
            policy,
            candidate_family_size=family_size,
        )
    else:
        beneficial = harmful = mean = lower_beneficial = upper_harmful = lower_mean = 0.0
    cluster_bootstrap_tail_replicates = policy.bootstrap_samples * (
        (1.0 - policy.confidence_level) / (2.0 * family_size)
    )
    if (
        cluster_bootstrap_tail_replicates
        < policy.minimum_bootstrap_tail_replicates
    ):
        reasons.append("insufficient_bootstrap_tail_resolution")
    if lower_beneficial < policy.minimum_beneficial_fraction:
        reasons.append("insufficient_beneficial_fraction")
    if upper_harmful > policy.maximum_harmful_fraction:
        reasons.append("excessive_harmful_fraction")
    if lower_mean < policy.minimum_mean_normalized_improvement:
        reasons.append("insufficient_mean_improvement")
    if maximum_degradation > policy.maximum_single_normalized_degradation:
        reasons.append("excessive_single_degradation")
    echo_records = uncertainty_records["intensity"]
    clear_records = uncertainty_records["clear"]
    state_echo_records = uncertainty_records["state_echo_miss"]
    state_clear_records = uncertainty_records["state_false_support"]
    echo_clusters = {item[2] for item in echo_records}
    clear_clusters = {item[2] for item in clear_records}
    if len(echo_records) < policy.minimum_prior_echo_cases:
        reasons.append("insufficient_prior_echo_cases")
    if len(clear_records) < policy.minimum_prior_clear_cases:
        reasons.append("insufficient_prior_clear_cases")
    if len(echo_clusters) < policy.minimum_prior_echo_clusters:
        reasons.append("insufficient_echo_clusters")
    if len(clear_clusters) < policy.minimum_prior_clear_clusters:
        reasons.append("insufficient_clear_clusters")
    for records in (state_echo_records, state_clear_records):
        if len(records) < policy.minimum_state_calibration_cases_per_regime or len(
            {item[2] for item in records}
        ) < policy.minimum_state_calibration_clusters_per_regime:
            reasons.append("insufficient_state_calibration")
    classifier_correct = tuple(item.classifier_reference_agreement for item in evaluations)
    known_indices = tuple(
        index
        for index, item in enumerate(evaluations)
        if item.regime != "unknown"
    )
    classifier_accuracy = (
        sum(classifier_correct[index] for index in known_indices)
        / len(known_indices)
        if known_indices
        else 0.0
    )
    reference_regimes = sorted(
        {item.regime for item in evaluations if item.regime != "unknown"}
    )
    recalls = tuple(
        sum(
            item.classifier_reference_agreement
            for item in evaluations
            if item.regime == regime
        )
        / max(1, sum(item.regime == regime for item in evaluations))
        for regime in reference_regimes
    )
    minimum_classifier_recall = min(recalls, default=0.0)
    classifier_confidences = tuple(
        min(
            evaluations[index].classifier_regime_confidence,
            evaluations[index].classifier_range_confidence,
        )
        for index in known_indices
    )
    calibration_terms: list[float] = []
    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        members = tuple(
            index
            for index, confidence in zip(
                known_indices,
                classifier_confidences,
                strict=True,
            )
            if lower <= confidence <= upper
            and (bin_index == 9 or confidence < upper)
        )
        if members:
            calibration_terms.append(
                len(members)
                / len(known_indices)
                * abs(
                    sum(classifier_correct[index] for index in members)
                    / len(members)
                    - sum(
                        min(
                            evaluations[index].classifier_regime_confidence,
                            evaluations[index].classifier_range_confidence,
                        )
                        for index in members
                    )
                    / len(members)
                )
            )
    classifier_calibration_error = sum(calibration_terms)
    classifier_false_routing_fraction = sum(
        not correct and not item.classifier_is_ood
        for correct, item in zip(classifier_correct, evaluations, strict=True)
    ) / len(evaluations)
    ood_items = tuple(item for item in evaluations if item.regime == "unknown")
    ood_abstention_fraction = (
        1.0
        if not ood_items
        else sum(item.classifier_is_ood for item in ood_items) / len(ood_items)
    )
    range_known_items = tuple(
        item for item in evaluations if not item.classifier_reference_range_is_ood
    )
    total_range_reference = sum(
        len(item.reference_active_range_regimes) for item in range_known_items
    )
    total_range_predicted = sum(
        len(item.classified_range_regimes) for item in range_known_items
    )
    total_range_intersection = sum(
        len(
            set(item.reference_active_range_regimes)
            & set(item.classified_range_regimes)
        )
        for item in range_known_items
    )
    range_set_precision = (
        total_range_intersection / total_range_predicted
        if total_range_predicted
        else 0.0
    )
    range_set_recall = (
        total_range_intersection / total_range_reference
        if total_range_reference
        else 0.0
    )
    range_exact_set_accuracy = (
        sum(item.classifier_range_exact_set_match for item in range_known_items)
        / len(range_known_items)
        if range_known_items
        else 0.0
    )
    range_false_active_band_fraction = (
        sum(
            len(
                set(item.classified_range_regimes)
                - set(item.reference_active_range_regimes)
            )
            for item in range_known_items
        )
        / max(1, total_range_predicted)
    )
    range_ood_items = tuple(
        item for item in evaluations if item.classifier_reference_range_is_ood
    )
    range_ood_abstention_fraction = (
        1.0
        if not range_ood_items
        else sum(item.classifier_is_ood for item in range_ood_items)
        / len(range_ood_items)
    )
    minimum_weather_margin = min(
        (item.classifier_weather_top1_top2_gap for item in evaluations),
        default=0.0,
    )
    minimum_range_margin = min(
        (item.classifier_minimum_range_presence_margin for item in evaluations),
        default=0.0,
    )
    classifier_clusters = [
        _physical_event_cluster(item) for item in evaluations
    ]
    known_clusters = [classifier_clusters[index] for index in known_indices]
    (
        classifier_accuracy_lower_bound,
        _,
    ) = _event_fractional_rate_interval(
        [float(classifier_correct[index]) for index in known_indices],
        known_clusters,
        policy,
        family_size=family_size,
    ) if known_indices else (0.0, 1.0)
    recall_lower_bounds = tuple(
        _event_fractional_rate_interval(
            [
                float(item.classifier_reference_agreement)
                for item in evaluations
                if item.regime == regime
            ],
            [
                _physical_event_cluster(item)
                for item in evaluations
                if item.regime == regime
            ],
            policy,
            family_size=family_size * max(1, len(reference_regimes)),
        )[0]
        for regime in reference_regimes
    )
    minimum_classifier_recall_lower_bound = min(
        recall_lower_bounds,
        default=0.0,
    )
    _, classifier_false_routing_upper_bound = _event_fractional_rate_interval(
        [
            float(not correct and not item.classifier_is_ood)
            for correct, item in zip(classifier_correct, evaluations, strict=True)
        ],
        classifier_clusters,
        policy,
        family_size=family_size,
    )
    range_precision_lower_bound, _ = _event_fractional_rate_interval(
        [item.classifier_range_set_precision for item in range_known_items],
        [
            _physical_event_cluster(item)
            for item in range_known_items
        ],
        policy,
        family_size=family_size,
    ) if range_known_items else (0.0, 1.0)
    range_recall_lower_bound, _ = _event_fractional_rate_interval(
        [item.classifier_range_set_recall for item in range_known_items],
        [
            _physical_event_cluster(item)
            for item in range_known_items
        ],
        policy,
        family_size=family_size,
    ) if range_known_items else (0.0, 1.0)
    _, false_active_band_upper_bound = _event_fractional_rate_interval(
        [
            item.classifier_false_active_band_fraction
            for item in range_known_items
        ],
        [
            _physical_event_cluster(item)
            for item in range_known_items
        ],
        policy,
        family_size=family_size,
    ) if range_known_items else (0.0, 1.0)
    classifier_validated = (
        classifier_accuracy >= policy.minimum_regime_classifier_accuracy
        and minimum_classifier_recall >= policy.minimum_regime_classifier_recall
        and classifier_accuracy_lower_bound
        >= policy.minimum_regime_classifier_accuracy_lower_bound
        and minimum_classifier_recall_lower_bound
        >= policy.minimum_regime_classifier_recall_lower_bound
        and classifier_calibration_error
        <= policy.maximum_regime_classifier_calibration_error
        and classifier_false_routing_fraction
        <= policy.maximum_regime_classifier_false_routing_fraction
        and classifier_false_routing_upper_bound
        <= policy.maximum_regime_classifier_false_routing_upper_bound
        and len(set(classifier_clusters))
        >= policy.minimum_regime_classifier_clusters
        and len(ood_items) >= policy.minimum_regime_classifier_ood_cases
        and ood_abstention_fraction
        >= policy.minimum_regime_classifier_ood_abstention_fraction
        and range_set_precision >= policy.minimum_range_set_precision
        and range_set_recall >= policy.minimum_range_set_recall
        and range_precision_lower_bound
        >= policy.minimum_range_set_precision_lower_bound
        and range_recall_lower_bound
        >= policy.minimum_range_set_recall_lower_bound
        and range_exact_set_accuracy >= policy.minimum_range_exact_set_accuracy
        and range_false_active_band_fraction
        <= policy.maximum_false_active_band_fraction
        and false_active_band_upper_bound
        <= policy.maximum_false_active_band_upper_bound
        and len(range_ood_items) >= policy.minimum_range_classifier_ood_cases
        and range_ood_abstention_fraction
        >= policy.minimum_range_classifier_ood_abstention_fraction
        and minimum_weather_margin >= policy.minimum_weather_top1_top2_gap
        and minimum_range_margin >= policy.minimum_range_presence_margin
    )
    if not classifier_validated:
        reasons.append("unreliable_regime_classifier")
    if (
        range_set_precision < policy.minimum_range_set_precision
        or range_set_recall < policy.minimum_range_set_recall
        or range_exact_set_accuracy < policy.minimum_range_exact_set_accuracy
        or range_false_active_band_fraction
        > policy.maximum_false_active_band_fraction
    ):
        reasons.append("unreliable_range_classifier")
    if (
        minimum_weather_margin < policy.minimum_weather_top1_top2_gap
        or minimum_range_margin < policy.minimum_range_presence_margin
    ):
        reasons.append("ambiguous_regime_classifier_branch")
    range_band_records = tuple(
        (
            item,
            band,
            _physical_event_cluster(item),
        )
        for item in evaluations
        if not item.classifier_is_ood
        and item.classifier_weather_reference_agreement
        and item.classifier_range_exact_set_match
        for band in item.range_band_evaluations
    )
    groups = sorted(
        {
            (item.classified_regime, band.range_regime)
            for item, band, _ in range_band_records
        }
    )
    band_skill_family_size = family_size * max(1, len(groups))
    band_bootstrap_diagnostics = _bootstrap_tail_diagnostics(
        policy,
        family_size=band_skill_family_size,
        enforce=False,
    )
    band_bootstrap_tail_ok = (
        band_bootstrap_diagnostics[1]
        >= policy.minimum_bootstrap_tail_replicates
    )
    absolute_component_limits = {
        "intensity": policy.maximum_prior_echo_intensity_nll,
        "pit_residual": (
            policy.maximum_prior_conditional_pit_residual_mean_abs
        ),
        "support": policy.maximum_prior_support_brier_score,
        "echo_miss": policy.maximum_prior_echo_support_miss_score,
        "object_miss": policy.maximum_prior_echo_object_miss_score,
        "clear": policy.maximum_prior_clear_sky_false_echo_score,
        "underdispersion": (
            policy.maximum_prior_conditional_underdispersion_fraction
        ),
        "state_nll": policy.maximum_state_gaussian_nll,
        "state_pit_residual": policy.maximum_state_pit_residual_mean_abs,
        "state_underdispersion": policy.maximum_state_underdispersion_fraction,
        "state_support": policy.maximum_state_support_brier_score,
        "state_echo_miss": policy.maximum_state_echo_support_miss_score,
        "state_object_miss": policy.maximum_state_echo_object_miss_score,
        "state_false_support": policy.maximum_state_false_support_score,
        "state_valid": policy.maximum_state_valid_brier_score,
    }
    certified_groups: list[tuple[str, str]] = []
    range_band_skill_bounds: list[
        tuple[str, str, float, float, float]
    ] = []
    range_band_skill_inference_diagnostics: list[
        tuple[str, str, str, int, int, float, float, float, int, int]
    ] = []
    range_metric_cell_bounds: list[
        tuple[str, str, str, int, float, float, int, int]
    ] = []
    range_metric_end_to_end_cell_bounds: list[
        tuple[
            str,
            str,
            str,
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            int,
            int,
        ]
    ] = []
    range_issuance_cell_bounds: list[
        tuple[str, str, int, float, float, float, float, int, int]
    ] = []
    metric_cell_event_count_map = {
        (
            item.weather_regime,
            item.range_regime,
            item.metric_name,
            item.lead_minutes,
        ): (
            0,
            max(
                item.minimum_physical_events,
                policy.minimum_deployment_metric_cell_events,
                policy.minimum_continuous_metric_cell_events,
            ),
        )
        for item in policy.required_range_metrics
    }
    issuance_cell_event_count_map = {
        (item.weather_regime, item.range_regime, item.lead_minutes): (
            0,
            item.minimum_physical_events,
        )
        for item in policy.required_range_issuance
    }
    metric_cell_test_count = max(
        1,
        4 * len(policy.required_range_metrics)
        + 4 * len(policy.required_range_issuance),
    )
    metric_cell_family_size = family_size * metric_cell_test_count
    metric_cell_bootstrap_diagnostics = _bootstrap_tail_diagnostics(
        policy,
        family_size=metric_cell_family_size,
        enforce=False,
    )
    metric_cell_tail_ok = (
        metric_cell_bootstrap_diagnostics[1]
        >= policy.minimum_bootstrap_tail_replicates
    )
    for group in groups:
        band_group = tuple(
            (item, band, cluster)
            for item, band, cluster in range_band_records
            if (item.classified_regime, band.range_regime) == group
        )
        band_scores = tuple(
            _paired_metric_score(
                item.metric_names,
                band.metric_change,
                band.end_to_end_metric_change,
                band.metric_available,
                policy,
            )
            for item, band, _ in band_group
        )
        retained_band_pairs = tuple(
            (score[0], cluster)
            for score, (_, _, cluster) in zip(
                band_scores,
                band_group,
                strict=True,
            )
            if score[0] is not None
        )
        retained_band_scores = tuple(
            score for score, _ in retained_band_pairs
        )
        band_clusters = tuple(cluster for _, cluster in retained_band_pairs)
        (
            band_beneficial_lower_bound,
            band_harmful_upper_bound,
            band_mean_lower_bound,
        ) = _cluster_bounds(
            list(retained_band_scores),
            list(band_clusters),
            policy,
            candidate_family_size=band_skill_family_size,
        ) if retained_band_scores else (0.0, 1.0, 0.0)
        range_band_skill_bounds.append(
            (
                group[0],
                group[1],
                band_beneficial_lower_bound,
                band_harmful_upper_bound,
                band_mean_lower_bound,
            )
        )
        range_band_skill_inference_diagnostics.append(
            (
                group[0],
                group[1],
                "cluster_bootstrap",
                band_skill_family_size,
                band_bootstrap_diagnostics[0],
                band_bootstrap_diagnostics[1],
                band_bootstrap_diagnostics[2],
                band_bootstrap_diagnostics[3],
                len(set(band_clusters)),
                len(retained_band_scores),
            )
        )
        band_skill_ok = (
            band_bootstrap_tail_ok
            and len(band_group) >= policy.minimum_range_band_cases
            and len({cluster for _, _, cluster in band_group})
            >= policy.minimum_range_band_clusters
            and all(
                band.evaluated_area_km2 >= policy.minimum_range_band_area_km2
                for _, band, _ in band_group
            )
            and all(
                all(
                    area >= policy.minimum_range_metric_valid_area_km2
                    for area in band.metric_valid_area_km2_by_lead
                )
                and band.probability_valid_area_km2
                >= policy.minimum_range_probability_valid_area_km2
                and band.state_valid_area_km2
                >= policy.minimum_range_state_valid_area_km2
                for _, band, _ in band_group
            )
            and len(retained_band_scores) >= policy.minimum_range_band_cases
            and band_beneficial_lower_bound
            >= policy.minimum_beneficial_fraction
            and band_harmful_upper_bound <= policy.maximum_harmful_fraction
            and band_mean_lower_bound
            >= policy.minimum_mean_normalized_improvement
            and all(
                not metric_harm and not end_to_end_harm
                for _, _, metric_harm, end_to_end_harm in band_scores
            )
        )
        group_issuance_requirements = tuple(
            item
            for item in policy.required_range_issuance
            if (item.weather_regime, item.range_regime) == group
        )
        band_issuance_ok = bool(group_issuance_requirements)
        issuance_bounds_by_lead: dict[int, tuple[float, float, float, float]] = {}
        for requirement in group_issuance_requirements:
            withdrawn_values: list[float] = []
            newly_issued_values: list[float] = []
            fallback_increase_values: list[float] = []
            confidence_loss_values: list[float] = []
            issuance_clusters: list[str] = []
            for evaluation, band, cluster in band_group:
                try:
                    lead_index = evaluation.lead_minutes.index(
                        requirement.lead_minutes
                    )
                except ValueError:
                    continue
                if (
                    band.issuance_domain_area_km2_by_lead[lead_index]
                    < requirement.minimum_operational_area_km2
                ):
                    continue
                withdrawn_values.append(
                    float(band.withdrawn_fraction_by_lead[lead_index])
                )
                newly_issued_values.append(
                    float(band.newly_issued_fraction_by_lead[lead_index])
                )
                fallback_increase_values.append(
                    max(
                        0.0,
                        float(
                            band.background_fallback_increase_by_lead[
                                lead_index
                            ]
                        ),
                    )
                )
                confidence_loss_values.append(
                    max(
                        0.0,
                        -float(
                            band.confidence_weighted_coverage_change_by_lead[
                                lead_index
                            ]
                        ),
                    )
                )
                issuance_clusters.append(cluster)
            issuance_cell_event_count_map[
                (
                    requirement.weather_regime,
                    requirement.range_regime,
                    requirement.lead_minutes,
                )
            ] = (
                len(set(issuance_clusters)),
                requirement.minimum_physical_events,
            )
            if (
                len(withdrawn_values) < requirement.minimum_cases
                or len(set(issuance_clusters))
                < requirement.minimum_physical_events
            ):
                band_issuance_ok = False
                continue
            _, withdrawn_upper = _event_fractional_rate_interval(
                withdrawn_values,
                issuance_clusters,
                policy,
                family_size=metric_cell_family_size,
            )
            _, newly_issued_upper = _event_fractional_rate_interval(
                newly_issued_values,
                issuance_clusters,
                policy,
                family_size=metric_cell_family_size,
            )
            _, fallback_increase_upper = _event_fractional_rate_interval(
                fallback_increase_values,
                issuance_clusters,
                policy,
                family_size=metric_cell_family_size,
            )
            _, confidence_loss_upper = _event_fractional_rate_interval(
                confidence_loss_values,
                issuance_clusters,
                policy,
                family_size=metric_cell_family_size,
            )
            issuance_bounds_by_lead[requirement.lead_minutes] = (
                withdrawn_upper,
                newly_issued_upper,
                fallback_increase_upper,
                confidence_loss_upper,
            )
            range_issuance_cell_bounds.append(
                (
                    requirement.weather_regime,
                    requirement.range_regime,
                    requirement.lead_minutes,
                    withdrawn_upper,
                    newly_issued_upper,
                    fallback_increase_upper,
                    confidence_loss_upper,
                    len(withdrawn_values),
                    len(set(issuance_clusters)),
                )
            )
            if (
                withdrawn_upper > requirement.maximum_withdrawn_fraction
                or newly_issued_upper
                > requirement.maximum_newly_issued_fraction
                or fallback_increase_upper
                > requirement.maximum_background_fallback_increase
                or confidence_loss_upper
                > requirement.maximum_confidence_weighted_coverage_loss
            ):
                band_issuance_ok = False
        group_metric_requirements = tuple(
            item
            for item in policy.required_range_metrics
            if (item.weather_regime, item.range_regime) == group
        )
        band_metric_completeness_ok = bool(group_metric_requirements)
        for requirement in group_metric_requirements:
            qualifying = 0
            cell_values: list[float] = []
            end_to_end_cell_values: list[float] = []
            cell_clusters: list[str] = []
            scale = next(
                item.scale
                for item in policy.metric_scales
                if item.metric_name == requirement.metric_name
            )
            for evaluation, band, cluster in band_group:
                try:
                    lead_index = evaluation.lead_minutes.index(
                        requirement.lead_minutes
                    )
                    metric_index = evaluation.metric_names.index(
                        requirement.metric_name
                    )
                except ValueError:
                    continue
                if (
                    bool(band.metric_available[lead_index, metric_index])
                    and float(
                        band.metric_valid_area_km2[lead_index, metric_index]
                    )
                    >= requirement.minimum_valid_area_km2
                ):
                    qualifying += 1
                    cell_values.append(
                        float(band.metric_change[lead_index, metric_index]) / scale
                    )
                    end_to_end_cell_values.append(
                        float(
                            band.end_to_end_metric_change[
                                lead_index,
                                metric_index,
                            ]
                        )
                        / scale
                    )
                    cell_clusters.append(cluster)
            metric_cell_event_count_map[
                (
                    requirement.weather_regime,
                    requirement.range_regime,
                    requirement.metric_name,
                    requirement.lead_minutes,
                )
            ] = (
                len(set(cell_clusters)),
                max(
                    requirement.minimum_physical_events,
                    policy.minimum_deployment_metric_cell_events,
                    policy.minimum_continuous_metric_cell_events,
                ),
            )
            if (
                qualifying < requirement.minimum_cases
                or len(set(cell_clusters))
                < max(
                    requirement.minimum_physical_events,
                    policy.minimum_deployment_metric_cell_events,
                    policy.minimum_continuous_metric_cell_events,
                )
                or not metric_cell_tail_ok
            ):
                band_metric_completeness_ok = False
                continue
            mean_upper = _bounded_event_mean_upper_bound(
                cell_values,
                cell_clusters,
                policy,
                family_size=metric_cell_family_size,
                absolute_bound=requirement.maximum_absolute_normalized_change,
            )
            _, harmful_upper = _event_fractional_rate_interval(
                [float(value > 0.0) for value in cell_values],
                cell_clusters,
                policy,
                family_size=metric_cell_family_size,
            )
            end_to_end_mean_upper = _bounded_event_mean_upper_bound(
                end_to_end_cell_values,
                cell_clusters,
                policy,
                family_size=metric_cell_family_size,
                absolute_bound=requirement.maximum_absolute_normalized_change,
            )
            _, end_to_end_harmful_upper = _event_fractional_rate_interval(
                [float(value > 0.0) for value in end_to_end_cell_values],
                cell_clusters,
                policy,
                family_size=metric_cell_family_size,
            )
            issuance_bounds = issuance_bounds_by_lead.get(
                requirement.lead_minutes,
                (1.0, 1.0, 1.0, 1.0),
            )
            (
                withdrawn_upper,
                newly_issued_upper,
                fallback_increase_upper,
                confidence_loss_upper,
            ) = issuance_bounds
            range_metric_cell_bounds.append(
                (
                    requirement.weather_regime,
                    requirement.range_regime,
                    requirement.metric_name,
                    requirement.lead_minutes,
                    mean_upper,
                    harmful_upper,
                    qualifying,
                    len(set(cell_clusters)),
                )
            )
            range_metric_end_to_end_cell_bounds.append(
                (
                    requirement.weather_regime,
                    requirement.range_regime,
                    requirement.metric_name,
                    requirement.lead_minutes,
                    end_to_end_mean_upper,
                    end_to_end_harmful_upper,
                    withdrawn_upper,
                    newly_issued_upper,
                    fallback_increase_upper,
                    confidence_loss_upper,
                    qualifying,
                    len(set(cell_clusters)),
                )
            )
            if (
                mean_upper > requirement.maximum_mean_normalized_degradation
                or harmful_upper
                > requirement.maximum_harmful_fraction_upper_bound
                or end_to_end_mean_upper
                > requirement.maximum_end_to_end_mean_normalized_degradation
                or end_to_end_harmful_upper
                > requirement.maximum_end_to_end_harmful_fraction_upper_bound
            ):
                band_metric_completeness_ok = False
        def band_component_records(
            component: str,
        ) -> tuple[tuple[float, int, str], ...]:
            values: list[tuple[float, int, str]] = []
            for _, band, cluster in band_group:
                difference = band.component_difference(component)
                count = dict(band.uncertainty_component_sample_counts).get(
                    component
                )
                if difference is not None and count is not None:
                    values.append((difference, count, cluster))
            return tuple(values)

        band_component_samples_ok = all(
            all(
                count
                >= (
                    policy.minimum_range_echo_objects
                    if component == "object_miss"
                    else policy.minimum_range_state_echo_objects
                    if component == "state_object_miss"
                    else policy.minimum_range_component_samples
                )
                for component, count in band.uncertainty_component_sample_counts
            )
            for _, band, _ in band_group
        )
        band_absolute_calibration_ok = all(
            all(
                score <= absolute_component_limits[component]
                for component, score in (
                    band.candidate_uncertainty_component_scores
                )
            )
            for _, band, _ in band_group
        )

        state_group = band_component_records("state_nll")
        state_cases_ok = (
            len(state_group)
            >= policy.minimum_state_calibration_cases_per_regime
        )
        state_clusters_ok = len({item[2] for item in state_group}) >= (
            policy.minimum_state_calibration_clusters_per_regime
        )
        if not state_cases_ok or not state_clusters_ok:
            if policy.require_all_registered_regimes_certified:
                reasons.append("insufficient_state_calibration")
        state_echo_group = band_component_records("state_echo_miss")
        state_clear_group = band_component_records("state_false_support")
        state_components_ok = all(
            len(component) >= policy.minimum_state_calibration_cases_per_regime
            and len({item[2] for item in component})
            >= policy.minimum_state_calibration_clusters_per_regime
            for component in (state_echo_group, state_clear_group)
        )
        if not state_components_ok:
            if policy.require_all_registered_regimes_certified:
                reasons.append("insufficient_state_calibration")
        support_group = band_component_records("support")
        support_cases_ok = (
            len(support_group) >= policy.minimum_uncertainty_cases_per_regime
        )
        support_clusters_ok = len({item[2] for item in support_group}) >= (
            policy.minimum_uncertainty_clusters_per_regime
        )
        if not support_cases_ok:
            if policy.require_all_registered_regimes_certified:
                reasons.append("insufficient_uncertainty_clusters")
        if not support_clusters_ok:
            if policy.require_all_registered_regimes_certified:
                reasons.append("insufficient_uncertainty_clusters")
        echo_group = band_component_records("intensity")
        echo_cases_ok = len(echo_group) >= policy.minimum_echo_cases_per_regime
        echo_clusters_ok = len({item[2] for item in echo_group}) >= (
            policy.minimum_echo_clusters_per_regime
        )
        if echo_group and not echo_cases_ok:
            if policy.require_all_registered_regimes_certified:
                reasons.append("insufficient_prior_echo_cases")
        if echo_group and not echo_clusters_ok:
            if policy.require_all_registered_regimes_certified:
                reasons.append("insufficient_echo_clusters")
        clear_group = band_component_records("clear")
        clear_cases_ok = len(clear_group) >= policy.minimum_clear_cases_per_regime
        clear_clusters_ok = len({item[2] for item in clear_group}) >= (
            policy.minimum_clear_clusters_per_regime
        )
        if clear_group and not clear_cases_ok:
            if policy.require_all_registered_regimes_certified:
                reasons.append("insufficient_prior_clear_cases")
        if clear_group and not clear_clusters_ok:
            if policy.require_all_registered_regimes_certified:
                reasons.append("insufficient_clear_clusters")
        if (
            band_skill_ok
            and band_metric_completeness_ok
            and band_issuance_ok
            and
            band_component_samples_ok
            and band_absolute_calibration_ok
            and
            state_cases_ok
            and state_clusters_ok
            and state_components_ok
            and support_cases_ok
            and support_clusters_ok
            and echo_cases_ok
            and echo_clusters_ok
            and clear_cases_ok
            and clear_clusters_ok
        ):
            certified_groups.append(group)
    comparisons: list[_UncertaintyComparison] = []
    for component, records in uncertainty_records.items():
        if not records:
            continue
        comparisons.append(
            _UncertaintyComparison(
                component=component,
                group=None,
                values=tuple(item[1] for item in records),
                clusters=tuple(item[2] for item in records),
            )
        )
        for group in groups:
            selected_values: list[tuple[float, str]] = []
            for item, band, cluster in range_band_records:
                if (item.classified_regime, band.range_regime) != group:
                    continue
                difference = band.component_difference(component)
                if difference is not None:
                    selected_values.append((difference, cluster))
            selected = tuple(selected_values)
            if not selected:
                continue
            comparisons.append(
                _UncertaintyComparison(
                    component=component,
                    group=group,
                    values=tuple(item[0] for item in selected),
                    clusters=tuple(item[1] for item in selected),
                )
            )
    simultaneous = _simultaneous_uncertainty_upper_bounds(
        tuple(comparisons),
        policy,
        candidate_family_size=family_size,
    )
    if (
        simultaneous.method == "studentized_multiplier"
        and simultaneous.tail_replicates
        < policy.minimum_bootstrap_tail_replicates
    ):
        reasons.append("insufficient_bootstrap_tail_resolution")
    comparison_bounds = simultaneous.comparison_bounds
    intensity_nll_upper = comparison_bounds.get(("intensity", None), 0.0)
    brier_upper = comparison_bounds[("support", None)]
    echo_miss_upper = comparison_bounds.get(("echo_miss", None), 0.0)
    object_miss_upper = comparison_bounds.get(("object_miss", None), 0.0)
    clear_sky_upper = comparison_bounds.get(("clear", None), 0.0)
    underdispersion_upper = comparison_bounds.get(("underdispersion", None), 0.0)
    state_nll_upper = comparison_bounds[("state_nll", None)]
    state_underdispersion_upper = comparison_bounds[("state_underdispersion", None)]
    state_support_upper = comparison_bounds[("state_support", None)]
    state_echo_miss_upper = comparison_bounds.get(("state_echo_miss", None), 0.0)
    state_object_miss_upper = comparison_bounds.get(("state_object_miss", None), 0.0)
    state_false_support_upper = comparison_bounds.get(("state_false_support", None), 0.0)
    state_valid_upper = comparison_bounds[("state_valid", None)]
    component_limits = {
        "intensity": policy.maximum_prior_echo_intensity_nll_increase,
        "pit_residual": (
            policy.maximum_prior_conditional_pit_residual_increase
        ),
        "support": policy.maximum_prior_support_brier_increase,
        "echo_miss": policy.maximum_prior_echo_support_miss_increase,
        "object_miss": policy.maximum_prior_echo_object_miss_increase,
        "clear": policy.maximum_prior_clear_sky_false_echo_increase,
        "underdispersion": policy.maximum_prior_conditional_underdispersion_increase,
        "state_nll": policy.maximum_state_gaussian_nll_increase,
        "state_pit_residual": policy.maximum_state_pit_residual_increase,
        "state_underdispersion": policy.maximum_state_underdispersion_increase,
        "state_support": policy.maximum_state_support_brier_increase,
        "state_echo_miss": policy.maximum_state_echo_support_miss_increase,
        "state_object_miss": policy.maximum_state_echo_object_miss_increase,
        "state_false_support": policy.maximum_state_false_support_increase,
        "state_valid": policy.maximum_state_valid_brier_increase,
    }
    certified_groups = [
        group
        for group in certified_groups
        if all(
            comparison_bounds.get((component, group), -math.inf) <= limit
            for component, limit in component_limits.items()
            if (component, group) in comparison_bounds
        )
    ]
    geometry_groups: dict[str, set[tuple[str, str]]] = {}
    for evaluation, band, _ in range_band_records:
        geometry_groups.setdefault(
            band.range_geometry_contract_digest,
            set(),
        ).add((evaluation.classified_regime, band.range_regime))
    certified_group_set = set(certified_groups)
    certified_range_geometries = tuple(
        digest
        for digest, geometry_evidence_groups in sorted(geometry_groups.items())
        if geometry_evidence_groups
        and geometry_evidence_groups <= certified_group_set
    )
    if (
        intensity_nll_upper
        > policy.maximum_prior_echo_intensity_nll_increase
        or brier_upper > policy.maximum_prior_support_brier_increase
        or echo_miss_upper > policy.maximum_prior_echo_support_miss_increase
        or object_miss_upper > policy.maximum_prior_echo_object_miss_increase
        or clear_sky_upper
        > policy.maximum_prior_clear_sky_false_echo_increase
        or underdispersion_upper
        > policy.maximum_prior_conditional_underdispersion_increase
    ):
        reasons.append("inferior_prior_uncertainty")
    if (
        state_nll_upper > policy.maximum_state_gaussian_nll_increase
        or state_underdispersion_upper
        > policy.maximum_state_underdispersion_increase
        or state_support_upper > policy.maximum_state_support_brier_increase
        or state_echo_miss_upper
        > policy.maximum_state_echo_support_miss_increase
        or state_object_miss_upper
        > policy.maximum_state_echo_object_miss_increase
        or state_false_support_upper
        > policy.maximum_state_false_support_increase
        or state_valid_upper > policy.maximum_state_valid_brier_increase
    ):
        reasons.append("inferior_state_head")
    unique = tuple(dict.fromkeys(reasons))
    sample_size_preflight = promotion_sample_size_preflight(
        plan,
        policy,
        available_physical_events=len(
            {_physical_event_cluster(item) for item in evaluations}
        ),
        metric_cell_event_counts=tuple(
            (*key, available, required)
            for key, (available, required) in sorted(
                metric_cell_event_count_map.items()
            )
        ),
        issuance_cell_event_counts=tuple(
            (*key, available, required)
            for key, (available, required) in sorted(
                issuance_cell_event_count_map.items()
            )
        ),
    )
    return NeuralPriorPromotionEvidence(
        candidate_prior_digest=manifest.candidate_prior_digest,
        parent_prior_digest=manifest.parent_prior_digest,
        candidate_manifest_digest=manifest.manifest_digest,
        policy_digest=policy.digest,
        trust_store_digest=trust.content_digest,
        scoring_artifact_digest=scoring_artifact.artifact_digest,
        scoring_process_log_digest=scoring_process_log.artifact_digest,
        scoring_completion_receipt_digest=(
            scoring_completion_receipt.receipt_digest
        ),
        evaluation_digests=tuple(item.evaluation_digest for item in evaluations),
        holdout_case_count=len(evaluations),
        material_case_count=material_count,
        distinct_case_count=len(cases),
        distinct_storm_count=len(storms),
        distinct_day_count=len(days),
        distinct_radar_count=len(radars),
        distinct_regime_count=len(regimes),
        distinct_range_regime_count=len(range_regimes),
        beneficial_fraction=beneficial,
        beneficial_fraction_lower_bound=lower_beneficial,
        harmful_fraction=harmful,
        harmful_fraction_upper_bound=upper_harmful,
        mean_normalized_improvement=mean,
        mean_improvement_lower_bound=lower_mean,
        maximum_normalized_degradation=maximum_degradation,
        prior_echo_intensity_nll_increase_upper_bound=intensity_nll_upper,
        prior_support_brier_increase_upper_bound=brier_upper,
        prior_echo_support_miss_increase_upper_bound=echo_miss_upper,
        prior_echo_object_miss_increase_upper_bound=object_miss_upper,
        prior_clear_sky_false_echo_increase_upper_bound=clear_sky_upper,
        prior_conditional_underdispersion_increase_upper_bound=(
            underdispersion_upper
        ),
        state_gaussian_nll_increase_upper_bound=state_nll_upper,
        state_underdispersion_increase_upper_bound=(
            state_underdispersion_upper
        ),
        state_support_brier_increase_upper_bound=state_support_upper,
        state_echo_support_miss_increase_upper_bound=state_echo_miss_upper,
        state_echo_object_miss_increase_upper_bound=state_object_miss_upper,
        state_false_support_increase_upper_bound=state_false_support_upper,
        state_valid_brier_increase_upper_bound=state_valid_upper,
        deployment_regime_classifier_digest=(
            policy.deployment_regime_classifier_digest
        ),
        deployment_regime_classifier_manifest_digest=(
            policy.deployment_regime_classifier_manifest_digest
        ),
        classifier_family_size=len(plan.regime_classifier_manifests),
        regime_classifier_evidence_digests=tuple(
            item.regime_classification_evidence_digest for item in evaluations
        ),
        regime_classifier_accuracy=classifier_accuracy,
        regime_classifier_accuracy_lower_bound=(
            classifier_accuracy_lower_bound
        ),
        minimum_regime_classifier_recall=minimum_classifier_recall,
        minimum_regime_classifier_recall_lower_bound=(
            minimum_classifier_recall_lower_bound
        ),
        regime_classifier_calibration_error=classifier_calibration_error,
        regime_classifier_false_routing_fraction=(
            classifier_false_routing_fraction
        ),
        regime_classifier_false_routing_upper_bound=(
            classifier_false_routing_upper_bound
        ),
        regime_classifier_cluster_count=len(set(classifier_clusters)),
        regime_classifier_ood_case_count=len(ood_items),
        regime_classifier_ood_abstention_fraction=ood_abstention_fraction,
        regime_classifier_validated=classifier_validated,
        range_set_precision=range_set_precision,
        range_set_precision_lower_bound=range_precision_lower_bound,
        range_set_recall=range_set_recall,
        range_set_recall_lower_bound=range_recall_lower_bound,
        range_exact_set_accuracy=range_exact_set_accuracy,
        range_false_active_band_fraction=range_false_active_band_fraction,
        range_false_active_band_upper_bound=false_active_band_upper_bound,
        range_classifier_ood_case_count=len(range_ood_items),
        range_classifier_ood_abstention_fraction=(
            range_ood_abstention_fraction
        ),
        minimum_classifier_weather_margin=minimum_weather_margin,
        minimum_classifier_range_margin=minimum_range_margin,
        prior_echo_component_status=(
            "available" if echo_records else "not_applicable"
        ),
        prior_clear_sky_component_status=(
            "available" if clear_records else "not_applicable"
        ),
        prior_echo_case_count=len(echo_records),
        prior_clear_sky_case_count=len(clear_records),
        prior_echo_cluster_count=len(echo_clusters),
        prior_clear_sky_cluster_count=len(clear_clusters),
        simultaneous_inference_test_count=simultaneous.test_count,
        simultaneous_inference_method=simultaneous.method,
        simultaneous_inference_effective_replicates=(
            simultaneous.effective_replicates
        ),
        simultaneous_inference_critical_quantile=(
            simultaneous.critical_quantile
        ),
        simultaneous_inference_monte_carlo_standard_error=(
            simultaneous.monte_carlo_standard_error
        ),
        simultaneous_inference_tail_replicates=simultaneous.tail_replicates,
        cluster_bootstrap_tail_replicates=cluster_bootstrap_tail_replicates,
        rate_inference_method="event-fractional-empirical-bernstein-v1",
        range_band_skill_bounds=tuple(range_band_skill_bounds),
        range_band_skill_inference_diagnostics=tuple(
            range_band_skill_inference_diagnostics
        ),
        range_metric_cell_bounds=tuple(range_metric_cell_bounds),
        range_metric_end_to_end_cell_bounds=tuple(
            range_metric_end_to_end_cell_bounds
        ),
        range_issuance_cell_bounds=tuple(range_issuance_cell_bounds),
        metric_cell_test_count=metric_cell_test_count * family_size,
        metric_cell_inference_method=(
            "joint-common-end-to-end-operational-issuance-bounded-event-inference-v2"
        ),
        metric_cell_effective_replicates=(
            metric_cell_bootstrap_diagnostics[0]
        ),
        metric_cell_tail_replicates=metric_cell_bootstrap_diagnostics[1],
        metric_cell_critical_quantile=metric_cell_bootstrap_diagnostics[2],
        metric_cell_monte_carlo_standard_error=(
            metric_cell_bootstrap_diagnostics[3]
        ),
        sample_size_preflight_digest=sample_size_preflight.preflight_digest,
        sample_size_available_physical_events=(
            sample_size_preflight.available_physical_events
        ),
        sample_size_required_physical_events=(
            sample_size_preflight.required_physical_events
        ),
        sample_size_cell_feasible=sample_size_preflight.cell_feasible,
        sample_size_preflight_feasible=sample_size_preflight.feasible,
        certified_applicability_regime_groups=tuple(certified_groups),
        certified_range_geometry_contract_digests=(
            certified_range_geometries
        ),
        requires_parent_fallback_outside_certified_applicability=True,
        state_calibration_eligible=not any(
            reason
            in {
                "unreliable_state_head",
                "inferior_state_head",
                "insufficient_state_calibration",
            }
            for reason in unique
        ),
        deployment_eligible=(
            not unique
            and classifier_validated
            and bool(certified_groups)
            and bool(certified_range_geometries)
            and sample_size_preflight.feasible
        ),
        eligible=not unique,
        rejection_reasons=unique,
    )


def validate_neural_prior_promotion(evidence: NeuralPriorPromotionEvidence) -> None:
    if evidence.promotion_evidence_digest != json_digest(evidence._payload()):
        raise ValueError("neural-prior promotion evidence digest mismatch")
    if not evidence.eligible:
        raise ValueError("neural-prior promotion evidence is not eligible")


def validate_neural_prior_promotion_applicability(
    evidence: NeuralPriorPromotionEvidence,
    *,
    regime: str,
    range_regime: str,
) -> None:
    """Fail closed to the parent prior outside statistically certified regimes."""

    validate_neural_prior_promotion(evidence)
    if (regime, range_regime) not in (
        evidence.certified_applicability_regime_groups
    ):
        raise ValueError(
            "neural prior is uncertified for this regime; use the parent prior"
        )


@dataclass(frozen=True, init=False)
class RegimeClassificationEvidence:
    """Classifier-derived regime labels bound to one full analysis input."""

    full_analysis_input_digest: str
    input_frames_digest: str
    classifier_digest: str
    regime: str
    range_regime: str
    active_range_regimes: tuple[str, ...]
    regime_confidence: float
    range_regime_confidence: float
    regime_labels: tuple[str, ...]
    range_regime_labels: tuple[str, ...]
    range_presence_probability_threshold: float
    regime_probabilities: tuple[float, ...]
    range_regime_probabilities: tuple[float, ...]
    regime_entropy: float
    is_ood: bool
    numerical_runtime_digest: str
    input_dtype: str
    input_device: str
    weather_top1_top2_gap: float
    minimum_range_presence_margin: float
    contract: str = "neural-prior-regime-classification-evidence-v3"
    evidence_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use NeuralPriorRegimeClassifier.classify")

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "evidence_digest"
        }

    def validate_integrity(self) -> None:
        if (
            self.contract != "neural-prior-regime-classification-evidence-v3"
            or not self.regime
            or not self.range_regime
            or len(self.regime_labels) != len(self.regime_probabilities)
            or len(self.range_regime_labels)
            != len(self.range_regime_probabilities)
            or len(set(self.regime_labels)) != len(self.regime_labels)
            or len(set(self.range_regime_labels))
            != len(self.range_regime_labels)
            or self.regime not in self.regime_labels
            or self.range_regime not in self.range_regime_labels
            or len(set(self.active_range_regimes))
            != len(self.active_range_regimes)
            or any(not value for value in self.active_range_regimes)
            or not 0.0 <= self.regime_confidence <= 1.0
            or not 0.0 <= self.range_regime_confidence <= 1.0
            or not self.regime_probabilities
            or not self.range_regime_probabilities
            or not 0.5 < self.range_presence_probability_threshold < 1.0
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in (
                    self.regime_probabilities
                    + self.range_regime_probabilities
                )
            )
            or not math.isfinite(self.regime_entropy)
            or self.regime_entropy < 0.0
            or self.is_ood != (
                self.regime == "unknown" or not self.active_range_regimes
            )
            or not self.input_dtype
            or not self.input_device
            or not math.isfinite(self.weather_top1_top2_gap)
            or self.weather_top1_top2_gap < 0.0
            or not math.isfinite(self.minimum_range_presence_margin)
            or self.minimum_range_presence_margin < 0.0
        ):
            raise ValueError("regime-classification evidence is invalid")
        regime_index = max(
            range(len(self.regime_probabilities)),
            key=self.regime_probabilities.__getitem__,
        )
        range_index = max(
            range(len(self.range_regime_probabilities)),
            key=self.range_regime_probabilities.__getitem__,
        )
        expected_active_ranges = tuple(
            label
            for label, probability in zip(
                self.range_regime_labels,
                self.range_regime_probabilities,
                strict=True,
            )
            if probability >= self.range_presence_probability_threshold
        )
        expected_range_confidence = (
            min(
                probability
                for probability in self.range_regime_probabilities
                if probability >= self.range_presence_probability_threshold
            )
            if expected_active_ranges
            else max(self.range_regime_probabilities)
        )
        ordered_weather = sorted(self.regime_probabilities, reverse=True)
        expected_weather_gap = ordered_weather[0] - (
            ordered_weather[1] if len(ordered_weather) > 1 else 0.0
        )
        expected_range_margin = min(
            abs(value - self.range_presence_probability_threshold)
            for value in self.range_regime_probabilities
        )
        if (
            not math.isclose(sum(self.regime_probabilities), 1.0, abs_tol=1e-12)
            or self.regime_labels[regime_index] != self.regime
            or self.range_regime_labels[range_index] != self.range_regime
            or expected_active_ranges != self.active_range_regimes
            or not math.isclose(
                self.regime_confidence,
                self.regime_probabilities[regime_index],
                abs_tol=1e-12,
            )
            or not math.isclose(
                self.range_regime_confidence,
                expected_range_confidence,
                abs_tol=1e-12,
            )
            or not math.isclose(
                self.weather_top1_top2_gap,
                expected_weather_gap,
                abs_tol=1e-12,
            )
            or not math.isclose(
                self.minimum_range_presence_margin,
                expected_range_margin,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("regime-classification evidence is inconsistent")
        for name in (
            "full_analysis_input_digest",
            "input_frames_digest",
            "classifier_digest",
            "numerical_runtime_digest",
        ):
            _require_digest(name, getattr(self, name))
        if self.evidence_digest != json_digest(self.payload):
            raise ValueError("regime-classification evidence digest mismatch")


class NeuralPriorRegimeClassifier:
    """Exported deterministic classifier used by the deployment selector."""

    def __init__(
        self,
        model: nn.Module,
        *,
        example_frames: Tensor,
        regime_labels: tuple[str, ...],
        range_regime_labels: tuple[str, ...],
        classifier_algorithm_digest: str,
        range_presence_probability_threshold: float = 0.8,
    ) -> None:
        if not isinstance(model, nn.Module) or model.training:
            raise ValueError("regime classifier must be an eval-mode module")
        for digest_name, digest in (
            ("classifier_algorithm_digest", classifier_algorithm_digest),
        ):
            _require_digest(digest_name, digest)
        for name, labels in (
            ("regime", regime_labels),
            ("range regime", range_regime_labels),
        ):
            if (
                not labels
                or len(set(labels)) != len(labels)
                or any(not item or item.strip() != item for item in labels)
            ):
                raise ValueError(f"{name} classifier labels are invalid")
        if "unknown" not in regime_labels:
            raise ValueError("regime classifier requires an unknown/OOD label")
        if (
            not math.isfinite(range_presence_probability_threshold)
            or not 0.5 < range_presence_probability_threshold < 1.0
        ):
            raise ValueError("range presence threshold must be inside (0.5,1)")
        if example_frames.ndim != 3 or not example_frames.is_floating_point():
            raise ValueError("regime classifier example must be [T,H,W]")
        exported, graph_digest = _export_graph(model, example_frames)
        output = exported(example_frames)
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or any(not isinstance(item, Tensor) for item in output)
            or output[0].shape != (len(regime_labels),)
            or output[1].shape != (len(range_regime_labels),)
        ):
            raise ValueError("regime classifier must return two logit vectors")
        self._model = model
        self._exported = exported
        self._graph_digest = graph_digest
        self._model_state_digest = _module_state_digest(model)
        self.regime_labels = regime_labels
        self.range_regime_labels = range_regime_labels
        self.classifier_algorithm_digest = classifier_algorithm_digest
        self.range_presence_probability_threshold = (
            range_presence_probability_threshold
        )
        self.numerical_runtime_digest = numerical_runtime_identity_digest(
            example_frames.device
        )
        self.input_dtype = str(example_frames.dtype)
        self.input_device = str(example_frames.device)
        self.classifier_digest = self._current_classifier_digest()

    def _current_classifier_digest(self) -> str:
        return json_digest(
            {
                "contract": "neural-prior-regime-classifier-v3",
                "graph_digest": self._graph_digest,
                "model_state_digest": _module_state_digest(self._model),
                "classifier_algorithm_digest": self.classifier_algorithm_digest,
                "regime_labels": list(self.regime_labels),
                "range_regime_labels": list(self.range_regime_labels),
                "range_presence_probability_threshold": (
                    self.range_presence_probability_threshold
                ),
                "numerical_runtime_digest": self.numerical_runtime_digest,
                "input_dtype": self.input_dtype,
                "input_device": self.input_device,
            }
        )

    def classify(
        self,
        frames_dbz: Tensor,
        *,
        input_run: ForecastRunContract,
    ) -> RegimeClassificationEvidence:
        input_run.validate_integrity()
        if (
            input_run.full_analysis_input_digest is None
            or tensor_digest(frames_dbz) != input_run.input_frames_digest
            or _module_state_digest(self._model) != self._model_state_digest
            or self._current_classifier_digest() != self.classifier_digest
            or numerical_runtime_identity_digest(frames_dbz.device)
            != self.numerical_runtime_digest
            or str(frames_dbz.dtype) != self.input_dtype
            or str(frames_dbz.device) != self.input_device
        ):
            raise ValueError("regime classifier input or artifact changed")
        output = self._exported(frames_dbz)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("regime classifier output is invalid")
        regime_logits, range_logits = output
        if (
            not bool(torch.all(torch.isfinite(regime_logits)))
            or not bool(torch.all(torch.isfinite(range_logits)))
        ):
            raise ValueError("regime classifier logits must be finite")
        regime_probability = torch.softmax(regime_logits.to(torch.float64), dim=0)
        range_probability = torch.sigmoid(range_logits.to(torch.float64))
        regime_index = int(torch.argmax(regime_probability))
        range_index = int(torch.argmax(range_probability))
        active_ranges = tuple(
            label
            for label, probability in zip(
                self.range_regime_labels,
                range_probability,
                strict=True,
            )
            if float(probability.detach())
            >= self.range_presence_probability_threshold
        )
        active_range_probabilities = tuple(
            float(probability.detach())
            for probability in range_probability
            if float(probability.detach())
            >= self.range_presence_probability_threshold
        )
        positive_regime = regime_probability.clamp_min(
            torch.finfo(torch.float64).tiny
        )
        entropy = float(
            torch.sum(-positive_regime * torch.log(positive_regime)).detach()
        )
        regime = self.regime_labels[regime_index]
        ordered_weather = torch.sort(regime_probability, descending=True).values
        weather_gap = float(
            (
                ordered_weather[0]
                - (ordered_weather[1] if len(ordered_weather) > 1 else 0.0)
            ).detach()
        )
        range_margin = float(
            torch.amin(
                torch.abs(
                    range_probability - self.range_presence_probability_threshold
                )
            ).detach()
        )
        result = object.__new__(RegimeClassificationEvidence)
        values: dict[str, object] = {
            "contract": "neural-prior-regime-classification-evidence-v3",
            "full_analysis_input_digest": input_run.full_analysis_input_digest,
            "input_frames_digest": tensor_digest(frames_dbz),
            "classifier_digest": self.classifier_digest,
            "regime": regime,
            "range_regime": self.range_regime_labels[range_index],
            "active_range_regimes": active_ranges,
            "regime_confidence": float(regime_probability[regime_index].detach()),
            "range_regime_confidence": float(
                min(active_range_probabilities)
                if active_range_probabilities
                else range_probability[range_index].detach()
            ),
            "regime_labels": self.regime_labels,
            "range_regime_labels": self.range_regime_labels,
            "range_presence_probability_threshold": (
                self.range_presence_probability_threshold
            ),
            "regime_probabilities": tuple(
                float(value.detach()) for value in regime_probability
            ),
            "range_regime_probabilities": tuple(
                float(value.detach()) for value in range_probability
            ),
            "regime_entropy": entropy,
            "is_ood": regime == "unknown" or not active_ranges,
            "numerical_runtime_digest": self.numerical_runtime_digest,
            "input_dtype": self.input_dtype,
            "input_device": self.input_device,
            "weather_top1_top2_gap": weather_gap,
            "minimum_range_presence_margin": range_margin,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        object.__setattr__(result, "evidence_digest", json_digest(result.payload))
        result.validate_integrity()
        return result


@dataclass(frozen=True)
class DeployedNeuralPriorPolicy:
    """Root-approved deployment selector for one promoted candidate family."""

    candidate_prior_digest: str
    parent_prior_digest: str
    promotion_evidence_digest: str
    regime_classifier_digest: str
    regime_classifier_manifest_digest: str
    range_geometry_contract_digest: str
    minimum_regime_confidence: float = 0.8
    minimum_weather_top1_top2_gap: float = 0.05
    minimum_deployment_confidence_margin: float = 0.05
    contract: str = "deployed-neural-prior-policy-v4"
    policy_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "candidate_prior_digest",
            "parent_prior_digest",
            "promotion_evidence_digest",
            "regime_classifier_digest",
            "regime_classifier_manifest_digest",
            "range_geometry_contract_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.contract != "deployed-neural-prior-policy-v4"
            or self.candidate_prior_digest == self.parent_prior_digest
            or not math.isfinite(self.minimum_regime_confidence)
            or not 0.0 < self.minimum_regime_confidence <= 1.0
            or any(
                not math.isfinite(value) or value < 0.0 or value > 1.0
                for value in (
                    self.minimum_weather_top1_top2_gap,
                    self.minimum_deployment_confidence_margin,
                )
            )
        ):
            raise ValueError("deployed neural-prior policy is invalid")
        object.__setattr__(
            self,
            "policy_digest",
            json_digest(self.payload),
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "policy_digest"
        }

    def validate_integrity(self) -> None:
        if self.policy_digest != json_digest(self.payload):
            raise ValueError("neural-prior deployment policy digest mismatch")


def _select_deployed_prior(
    candidate_runner: NeuralPriorInferenceRunner,
    parent_runner: NeuralPriorInferenceRunner,
    promotion_evidence: NeuralPriorPromotionEvidence,
    regime_evidence: RegimeClassificationEvidence,
    range_partition_evidence: RangePartitionEvidence,
    policy: DeployedNeuralPriorPolicy,
    *,
    range_geometry_contract: RangeGeometryContract,
    operational_grid_contract_digest: str,
    operational_frame_shape: tuple[int, int],
    operational_radar_source_kind: str | None = None,
    operational_radar_site_digest: str | None = None,
    operational_radar_site_location_digest: str | None = None,
    policy_trust_store_path: str | Path,
) -> tuple[NeuralPriorInferenceRunner, NeuralPriorDeploymentSelection]:
    """Select the candidate only for classifier-attested certified regimes."""

    regime_evidence.validate_integrity()
    range_partition_evidence.validate_integrity()
    range_geometry_contract.validate_integrity()
    if (
        range_partition_evidence.grid_contract_digest
        != operational_grid_contract_digest
        or not range_partition_evidence.masks
        or range_partition_evidence.masks[0].shape
        != operational_frame_shape
        or range_geometry_contract.grid_contract_digest
        != operational_grid_contract_digest
        or range_geometry_contract.contract_digest
        != range_partition_evidence.range_geometry_contract_digest
        or range_geometry_contract.range_regime_labels
        != range_partition_evidence.range_regime_labels
    ):
        raise ValueError("range partition disagrees with operational grid")
    policy.validate_integrity()
    trust = _load_learning_policy_trust_store(policy_trust_store_path)
    if policy.policy_digest not in trust.approved_policy_digests:
        raise ValueError("unapproved neural-prior deployment policy")
    if promotion_evidence.promotion_evidence_digest != json_digest(
        promotion_evidence._payload()
    ):
        raise ValueError("neural-prior promotion evidence digest mismatch")
    if (
        policy.candidate_prior_digest != candidate_runner.neural_prior_digest
        or policy.parent_prior_digest != parent_runner.neural_prior_digest
        or policy.promotion_evidence_digest
        != promotion_evidence.promotion_evidence_digest
        or policy.regime_classifier_digest != regime_evidence.classifier_digest
        or policy.regime_classifier_digest
        != promotion_evidence.deployment_regime_classifier_digest
        or policy.regime_classifier_manifest_digest
        != promotion_evidence.deployment_regime_classifier_manifest_digest
        or promotion_evidence.candidate_prior_digest
        != candidate_runner.neural_prior_digest
        or promotion_evidence.parent_prior_digest
        != parent_runner.neural_prior_digest
        or policy.range_geometry_contract_digest
        != range_partition_evidence.range_geometry_contract_digest
    ):
        raise ValueError("deployment policy lineage disagrees")
    confidence_ok = regime_evidence.regime_confidence >= policy.minimum_regime_confidence
    deployment_confidence_margin = (
        regime_evidence.regime_confidence - policy.minimum_regime_confidence
    )
    branch_stable = (
        regime_evidence.weather_top1_top2_gap
        >= policy.minimum_weather_top1_top2_gap
        and deployment_confidence_margin
        >= policy.minimum_deployment_confidence_margin
    )
    active_groups = tuple(
        (regime_evidence.regime, range_regime)
        for range_regime in range_partition_evidence.active_range_regimes
    )
    certified = set(promotion_evidence.certified_applicability_regime_groups)
    if not promotion_evidence.deployment_eligible:
        selected = parent_runner
        role: Literal["candidate", "parent"] = "parent"
        reason = "promotion_ineligible"
    elif not promotion_evidence.certified_applicability_regime_groups:
        selected = parent_runner
        role = "parent"
        reason = "no_certified_regime"
    elif regime_evidence.is_ood:
        selected = parent_runner
        role = "parent"
        reason = "ood_or_abstained"
    elif not branch_stable:
        selected = parent_runner
        role = "parent"
        reason = "ambiguous_classifier_branch"
    elif not confidence_ok:
        selected = parent_runner
        role = "parent"
        reason = "low_regime_confidence"
    elif policy.range_geometry_contract_digest not in set(
        promotion_evidence.certified_range_geometry_contract_digests
    ):
        selected = parent_runner
        role = "parent"
        reason = "uncertified_range_geometry"
    elif not any(group[0] == regime_evidence.regime for group in certified):
        selected = parent_runner
        role = "parent"
        reason = "uncertified_regime"
    elif not active_groups or any(group not in certified for group in active_groups):
        selected = parent_runner
        role = "parent"
        reason = "uncertified_range_band"
    else:
        selected = candidate_runner
        role = "candidate"
        reason = "certified_candidate"
    deployment_decision_payload = {
        "contract": "neural-prior-deployment-decision-artifact-v4",
        "full_analysis_input_digest": regime_evidence.full_analysis_input_digest,
        "operational_grid_contract_digest": operational_grid_contract_digest,
        "operational_frame_shape": list(operational_frame_shape),
        "operational_radar_source_kind": operational_radar_source_kind,
        "operational_radar_site_digest": operational_radar_site_digest,
        "operational_radar_site_location_digest": (
            operational_radar_site_location_digest
        ),
        "regime_classification_evidence": (
            regime_evidence.payload
            | {"evidence_digest": regime_evidence.evidence_digest}
        ),
        "deployment_policy": policy.payload | {"policy_digest": policy.policy_digest},
        "range_partition_evidence": (
            range_partition_evidence.payload
            | {"evidence_digest": range_partition_evidence.evidence_digest}
        ),
        "range_geometry_contract": (
            range_geometry_contract.payload
            | {"contract_digest": range_geometry_contract.contract_digest}
        ),
        "promotion_selection_evidence": {
            "promotion_evidence_digest": (
                promotion_evidence.promotion_evidence_digest
            ),
            "candidate_prior_digest": promotion_evidence.candidate_prior_digest,
            "parent_prior_digest": promotion_evidence.parent_prior_digest,
            "deployment_eligible": promotion_evidence.deployment_eligible,
            "deployment_regime_classifier_digest": (
                promotion_evidence.deployment_regime_classifier_digest
            ),
            "deployment_regime_classifier_manifest_digest": (
                promotion_evidence.deployment_regime_classifier_manifest_digest
            ),
            "certified_applicability_regime_groups": [
                list(value)
                for value in (
                    promotion_evidence.certified_applicability_regime_groups
                )
            ],
            "certified_range_geometry_contract_digests": list(
                promotion_evidence.certified_range_geometry_contract_digests
            ),
        },
        "policy_trust_store": {
            "contract": "advar-learning-policy-trust-store-v1",
            "approved_policy_digests": sorted(trust.approved_policy_digests),
            "content_digest": trust.content_digest,
        },
        "selection": {
            "selected_prior_digest": selected.neural_prior_digest,
            "selected_role": role,
            "fallback_reason": reason,
            "deployment_confidence_margin": deployment_confidence_margin,
        },
    }
    deployment_decision_json = json.dumps(
        deployment_decision_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    selection = _new_neural_prior_deployment_selection(
        selected_prior_digest=selected.neural_prior_digest,
        selected_role=role,
        full_analysis_input_digest=regime_evidence.full_analysis_input_digest,
        promotion_evidence_digest=promotion_evidence.promotion_evidence_digest,
        regime_classification_evidence_digest=regime_evidence.evidence_digest,
        deployment_policy_digest=policy.policy_digest,
        deployment_policy_trust_store_digest=trust.content_digest,
        range_geometry_contract_digest=policy.range_geometry_contract_digest,
        range_partition_evidence_digest=range_partition_evidence.evidence_digest,
        classifier_numerical_runtime_digest=(
            regime_evidence.numerical_runtime_digest
        ),
        classifier_input_dtype=regime_evidence.input_dtype,
        classifier_input_device=regime_evidence.input_device,
        weather_top1_top2_gap=regime_evidence.weather_top1_top2_gap,
        minimum_range_presence_margin=(
            regime_evidence.minimum_range_presence_margin
        ),
        deployment_confidence_margin=deployment_confidence_margin,
        deployment_decision_artifact_json=deployment_decision_json,
        deployment_decision_artifact_digest=json_digest(
            deployment_decision_payload
        ),
        fallback_reason=reason,
    )
    if (
        validate_neural_prior_deployment_decision_artifact(
            selection.deployment_decision_artifact_json
        )
        != selection.deployment_decision_artifact_digest
    ):
        raise ValueError("neural-prior deployment artifact digest mismatch")
    return selected, selection


def validate_neural_prior_deployment_decision_artifact(
    artifact_json: str,
    *,
    expected_operational_grid_contract_digest: str | None = None,
    expected_operational_frame_shape: tuple[int, int] | None = None,
    expected_operational_radar_source_kind: str | None = None,
    expected_operational_radar_site_digest: str | None = None,
    expected_operational_radar_site_location_digest: str | None = None,
) -> str:
    """Replay a durable classifier/policy/certification deployment choice."""

    try:
        payload = json.loads(artifact_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid neural-prior deployment artifact") from error
    if (
        not isinstance(payload, dict)
        or payload.get("contract")
        != "neural-prior-deployment-decision-artifact-v4"
        or json.dumps(payload, sort_keys=True, separators=(",", ":"))
        != artifact_json
    ):
        raise ValueError("neural-prior deployment artifact is not canonical")
    regime = payload.get("regime_classification_evidence")
    policy = payload.get("deployment_policy")
    promotion = payload.get("promotion_selection_evidence")
    range_partition = payload.get("range_partition_evidence")
    range_geometry = payload.get("range_geometry_contract")
    trust = payload.get("policy_trust_store")
    selection = payload.get("selection")
    if not all(
        isinstance(value, dict)
        for value in (
            regime,
            policy,
            promotion,
            range_partition,
            range_geometry,
            trust,
            selection,
        )
    ):
        raise ValueError("neural-prior deployment artifact is incomplete")
    assert isinstance(regime, dict)
    assert isinstance(policy, dict)
    assert isinstance(promotion, dict)
    assert isinstance(range_partition, dict)
    assert isinstance(range_geometry, dict)
    assert isinstance(trust, dict)
    assert isinstance(selection, dict)
    regime_payload = dict(regime)
    regime_digest = regime_payload.pop("evidence_digest", None)
    policy_payload = dict(policy)
    policy_digest = policy_payload.pop("policy_digest", None)
    approved = trust.get("approved_policy_digests")
    range_partition_payload = dict(range_partition)
    range_partition_digest = range_partition_payload.pop("evidence_digest", None)
    range_geometry_payload = dict(range_geometry)
    range_geometry_digest = range_geometry_payload.pop("contract_digest", None)
    geometry_values = dict(range_geometry_payload)
    for name in ("range_regime_labels", "radial_distance_edges_m"):
        value = geometry_values.get(name)
        if not isinstance(value, list):
            raise ValueError("neural-prior deployment artifact is incomplete")
        geometry_values[name] = tuple(value)
    try:
        reconstructed_geometry = RangeGeometryContract(
            **cast(Any, geometry_values)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "neural-prior deployment artifact range geometry is invalid"
        ) from error
    if (
        not isinstance(regime_digest, str)
        or json_digest(regime_payload) != regime_digest
        or not isinstance(policy_digest, str)
        or json_digest(policy_payload) != policy_digest
        or not isinstance(approved, list)
        or policy_digest not in approved
        or json_digest(
            {
                "contract": trust.get("contract"),
                "approved_policy_digests": sorted(approved),
            }
        )
        != trust.get("content_digest")
        or policy.get("promotion_evidence_digest")
        != promotion.get("promotion_evidence_digest")
        or policy.get("regime_classifier_digest")
        != regime.get("classifier_digest")
        or promotion.get("deployment_regime_classifier_digest")
        != regime.get("classifier_digest")
        or policy.get("regime_classifier_manifest_digest")
        != promotion.get("deployment_regime_classifier_manifest_digest")
        or not isinstance(range_partition_digest, str)
        or json_digest(range_partition_payload) != range_partition_digest
        or not isinstance(range_geometry_digest, str)
        or json_digest(range_geometry_payload) != range_geometry_digest
        or reconstructed_geometry.contract_digest != range_geometry_digest
        or policy.get("range_geometry_contract_digest")
        != range_partition.get("range_geometry_contract_digest")
        or range_geometry_digest
        != range_partition.get("range_geometry_contract_digest")
        or range_geometry.get("grid_contract_digest")
        != range_partition.get("grid_contract_digest")
        or range_geometry.get("range_regime_labels")
        != range_partition.get("range_regime_labels")
        or payload.get("full_analysis_input_digest")
        != regime.get("full_analysis_input_digest")
    ):
        raise ValueError("neural-prior deployment artifact lineage disagrees")
    if (
        "operational_radar_source_kind" not in payload
        or "operational_radar_site_digest" not in payload
        or "operational_radar_site_location_digest" not in payload
        or
        payload.get("operational_grid_contract_digest")
        != range_partition.get("grid_contract_digest")
        or payload.get("operational_frame_shape")
        != range_partition.get("grid_shape")
    ):
        raise ValueError(
            "neural-prior deployment artifact disagrees with operational grid"
        )
    if (
        expected_operational_grid_contract_digest is not None
        and payload.get("operational_grid_contract_digest")
        != expected_operational_grid_contract_digest
    ) or (
        expected_operational_frame_shape is not None
        and payload.get("operational_frame_shape")
        != list(expected_operational_frame_shape)
    ) or (
        expected_operational_radar_source_kind is not None
        and payload.get("operational_radar_source_kind")
        != expected_operational_radar_source_kind
    ) or (
        expected_operational_radar_site_digest is not None
        and payload.get("operational_radar_site_digest")
        != expected_operational_radar_site_digest
    ) or (
        expected_operational_radar_site_location_digest is not None
        and payload.get("operational_radar_site_location_digest")
        != expected_operational_radar_site_location_digest
    ):
        raise ValueError(
            "neural-prior deployment artifact disagrees with current forecast run"
        )
    active_groups = tuple(
        (regime.get("regime"), range_regime)
        for range_regime in range_partition.get("active_range_regimes", [])
    )
    certified = {
        tuple(value)
        for value in promotion.get("certified_applicability_regime_groups", [])
        if isinstance(value, list) and len(value) == 2
    }
    confidence = float(regime.get("regime_confidence", -math.inf))
    deployment_margin = confidence - float(
        policy.get("minimum_regime_confidence", math.inf)
    )
    branch_stable = (
        float(regime.get("weather_top1_top2_gap", -math.inf))
        >= float(policy.get("minimum_weather_top1_top2_gap", math.inf))
        and deployment_margin
        >= float(policy.get("minimum_deployment_confidence_margin", math.inf))
    )
    if not promotion.get("deployment_eligible"):
        reason = "promotion_ineligible"
    elif not certified:
        reason = "no_certified_regime"
    elif regime.get("is_ood"):
        reason = "ood_or_abstained"
    elif not branch_stable:
        reason = "ambiguous_classifier_branch"
    elif confidence < float(policy.get("minimum_regime_confidence", math.inf)):
        reason = "low_regime_confidence"
    elif policy.get("range_geometry_contract_digest") not in set(
        promotion.get("certified_range_geometry_contract_digests", [])
    ):
        reason = "uncertified_range_geometry"
    elif not any(group[0] == regime.get("regime") for group in certified):
        reason = "uncertified_regime"
    elif not active_groups or any(group not in certified for group in active_groups):
        reason = "uncertified_range_band"
    else:
        reason = "certified_candidate"
    expected_role = "candidate" if reason == "certified_candidate" else "parent"
    expected_prior = (
        policy.get("candidate_prior_digest")
        if expected_role == "candidate"
        else policy.get("parent_prior_digest")
    )
    if (
        selection.get("fallback_reason") != reason
        or selection.get("selected_role") != expected_role
        or selection.get("selected_prior_digest") != expected_prior
        or not math.isclose(
            float(selection.get("deployment_confidence_margin", math.nan)),
            deployment_margin,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("neural-prior deployment selection replay failed")
    return json_digest(payload)


def infer_deployed_neural_prior(
    frames_dbz: Tensor,
    *,
    input_run: ForecastRunContract,
    candidate_runner: NeuralPriorInferenceRunner,
    parent_runner: NeuralPriorInferenceRunner,
    promotion_evidence: NeuralPriorPromotionEvidence,
    regime_classifier: NeuralPriorRegimeClassifier,
    range_geometry_contract: RangeGeometryContract,
    grid_x_m: Tensor,
    grid_y_m: Tensor,
    policy: DeployedNeuralPriorPolicy,
    policy_trust_store_path: str | Path,
) -> NeuralPriorApplication:
    """Classify, select, and infer without accepting caller-provided labels."""

    if (
        input_run.grid_time_contract_digest is None
        or range_geometry_contract.grid_contract_digest
        != input_run.grid_time_contract_digest
    ):
        raise ValueError("range geometry disagrees with operational grid")
    if (
        grid_x_m.shape != frames_dbz.shape[-2:]
        or grid_y_m.shape != frames_dbz.shape[-2:]
    ):
        raise ValueError("range coordinates disagree with radar frames")
    if input_run.operational_data_identity_json is None:
        raise ValueError("operational neural prior requires radar source identity")
    source_identity = OperationalDataIdentity.from_json(
        input_run.operational_data_identity_json
    )
    if source_identity.radar_source_kind != "single_site":
        raise ValueError(
            "single-radar range geometry cannot be used with a mosaic source"
        )
    if (
        source_identity.radar_site_digest != range_geometry_contract.radar_site_digest
        or source_identity.radar_site_location_digest
        != range_geometry_contract.radar_site_location_digest
    ):
        raise ValueError("range geometry disagrees with operational radar site")
    regime_evidence = regime_classifier.classify(frames_dbz, input_run=input_run)
    range_partition_evidence = resolve_range_geometry(
        range_geometry_contract,
        grid_x_m=grid_x_m,
        grid_y_m=grid_y_m,
    )
    runner, selection = _select_deployed_prior(
        candidate_runner,
        parent_runner,
        promotion_evidence,
        regime_evidence,
        range_partition_evidence,
        policy,
        range_geometry_contract=range_geometry_contract,
        operational_grid_contract_digest=input_run.grid_time_contract_digest,
        operational_frame_shape=(
            int(frames_dbz.shape[-2]),
            int(frames_dbz.shape[-1]),
        ),
        operational_radar_source_kind=source_identity.radar_source_kind,
        operational_radar_site_digest=source_identity.radar_site_digest,
        operational_radar_site_location_digest=(
            source_identity.radar_site_location_digest
        ),
        policy_trust_store_path=policy_trust_store_path,
    )
    return runner._infer_deployed(
        frames_dbz,
        input_run=input_run,
        deployment_selection=selection,
    )
