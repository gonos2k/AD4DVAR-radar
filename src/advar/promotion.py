"""Fail-closed holdout evidence for learned radar priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import random
from typing import Literal

import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest
from .calibration import OperationalDataIdentity
from .nowcast import ForecastResult, _forecast_input_plan_resolution_digest
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
    NeuralPriorApplication,
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
]
PriorComponentStatus = Literal["available", "not_applicable"]


def _require_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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
    grid_contract_digest: str
    feature_exclusion_contract_digest: str
    independence_evidence_digest: str
    target_valid_time: str
    prior_probability_contract_digest: str
    support_threshold_dbz: float = 5.0
    contract: str = "prior-uncertainty-target-plan-v3"
    support_event_digest: str = field(init=False)
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "prior-uncertainty-target-plan-v3":
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
            "grid_contract_digest",
            "feature_exclusion_contract_digest",
            "independence_evidence_digest",
            "prior_probability_contract_digest",
        ):
            _require_digest(name, getattr(self, name))
        if not math.isfinite(self.support_threshold_dbz):
            raise ValueError("uncertainty support threshold must be finite")
        support_event_digest = json_digest(
            {
                "contract": "radar-support-event-v1",
                "variable": "radar_reflectivity_dbz",
                "operator": ">=",
                "threshold_dbz": self.support_threshold_dbz,
                "support_product_digest": self.source_identity_digest,
                "qc_pipeline_digest": self.qc_pipeline_digest,
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
            "support_event_digest": self.support_event_digest,
        }


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
    issue_time: str

    def __post_init__(self) -> None:
        _validate_holdout_case_identity(self)
        for name in (
            "input_plan_digest",
            "verification_plan_digest",
            "metric_contract_digest",
            "uncertainty_target_plan_digest",
        ):
            _require_digest(name, getattr(self, name))
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
    cases: tuple[NeuralPriorHoldoutPlanCase, ...]
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
    cases: tuple[NeuralPriorHoldoutPlanCase, ...]
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
class NeuralPriorHoldoutPlan:
    """Root-approved holdout commitment created before any evaluated issue."""

    plan_id: str
    parent_prior_digest: str
    candidate_family_digests: tuple[str, ...]
    cases: tuple[NeuralPriorHoldoutPlanCase, ...]
    input_plans: tuple[NeuralPriorInputPlan, ...]
    uncertainty_target_plans: tuple[PriorUncertaintyTargetPlan, ...]
    registered_at: str
    mode: Literal["prospective", "sealed_historical"] = "prospective"
    sealed_historical_dataset_digest: str | None = None
    candidate_training_started_at: str | None = None
    contract: str = "neural-prior-holdout-plan-v5"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-holdout-plan-v5":
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
        registered = _canonical_time(self.registered_at)
        if self.mode == "prospective":
            if self.sealed_historical_dataset_digest is not None or (
                self.candidate_training_started_at is not None
            ):
                raise ValueError("prospective holdout cannot use a sealed dataset")
            if any(registered >= item.issue_time for item in self.cases):
                raise ValueError("holdout plan must precede every issue time")
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
    storm_id: str
    day: str
    radar_id: str
    regime: str
    range_regime: str
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
            "candidate_forecast_digest",
            "parent_forecast_digest",
            "candidate_prior_application_digest",
            "parent_prior_application_digest",
            "candidate_inference_evidence_digest",
            "parent_inference_evidence_digest",
        ):
            _require_digest(name, getattr(self, name))
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
            storm_id=self.storm_id,
            day=self.day,
            radar_id=self.radar_id,
            regime=self.regime,
            range_regime=self.range_regime,
            input_plan_digest=self.input_plan_digest,
            verification_plan_digest=self.verification_plan_digest,
            metric_contract_digest=self.metric_contract_digest,
            uncertainty_target_plan_digest=self.uncertainty_target_plan_digest,
            issue_time=self.issue_time,
        )


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
            "contract": "neural-prior-holdout-dataset-v2",
            "cases": [
                {
                    "case_id": item.case_id,
                    "input_plan_digest": item.input_plan_digest,
                    "verification_plan_digest": item.verification_plan_digest,
                    "metric_contract_digest": item.metric_contract_digest,
                    "uncertainty_target_plan_digest": (
                        item.uncertainty_target_plan_digest
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
    """Immutable training/holdout lineage for exactly one prior candidate."""

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
    training_storm_ids: tuple[str, ...]
    training_days: tuple[str, ...]
    training_radars: tuple[str, ...]
    training_regimes: tuple[str, ...]
    training_time_windows: tuple[tuple[str, str], ...]
    holdout_cases: tuple[NeuralPriorHoldoutCase, ...]
    contract: str = "neural-prior-candidate-manifest-v4"
    manifest_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-candidate-manifest-v4":
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
        for digest in (
            self.training_learning_approval_digests
            + self.training_intervention_digests
            + self.training_input_bundle_digests
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
        holdout_ids = tuple(item.case_id for item in self.holdout_cases)
        if not holdout_ids or len(set(holdout_ids)) != len(holdout_ids):
            raise ValueError("holdout cases must be nonempty and unique")
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
            plan.contract != "prior-uncertainty-target-plan-v3"
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
                "contract": "prior-uncertainty-target-v2",
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
class PriorHoldoutEvaluation:
    """Paired prior holdout result over the full preregistered population."""

    holdout_plan_digest: str
    candidate_manifest_digest: str
    candidate_prior_digest: str
    parent_prior_digest: str
    case_id: str
    storm_id: str
    day: str
    radar_id: str
    regime: str
    range_regime: str
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
    issue_time: str
    verification_valid_times: tuple[str, ...]
    contract: str = "prior-holdout-evaluation-v8"
    evaluation_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use PriorHoldoutEvaluation.from_forecasts")

    def __post_init__(self) -> None:
        if self.contract != "prior-holdout-evaluation-v8":
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
        ):
            _require_digest(name, getattr(self, name))
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
                or not 0.0 <= echo_floats[5] <= 1.0
                or not 0.0 <= echo_floats[7] <= 1.0
                or not 0.0 <= echo_floats[8] <= 1.0
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
    ) -> PriorHoldoutEvaluation:
        """Evaluate every planned prior case without intervention selection."""

        validate_neural_prior_holdout_plan(plan)
        validate_neural_prior_candidate_manifest(manifest)
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
        candidate_evidence = candidate_prior_application.inference_evidence
        parent_evidence = parent_prior_application.inference_evidence
        target_plan = next(
            item for item in plan.uncertainty_target_plans
            if item.plan_digest == case.uncertainty_target_plan_digest
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
        )
        parent_scores = _prior_uncertainty_scores(
            parent_prior_application,
            prior_reference,
            support_target,
            prior_valid,
            support_threshold_dbz=target_plan.support_threshold_dbz,
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
        issue_time = candidate_forecast.run.grid_time_contract.valid_times[-1]
        parent_issue = parent_forecast.run.grid_time_contract
        if parent_issue is None or parent_issue.valid_times[-1] != issue_time:
            raise ValueError("candidate and parent issue times disagree")
        if issue_time != case.issue_time:
            raise ValueError("holdout issue time is not pre-registered")
        return _new_prior_holdout_evaluation(
            holdout_plan_digest=plan.plan_digest,
            candidate_manifest_digest=manifest.manifest_digest,
            candidate_prior_digest=manifest.candidate_prior_digest,
            parent_prior_digest=manifest.parent_prior_digest,
            case_id=case.case_id,
            storm_id=case.storm_id,
            day=case.day,
            radar_id=case.radar_id,
            regime=case.regime,
            range_regime=case.range_regime,
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


def _truncated_gaussian_diagnostics(
    location_dbz: Tensor,
    scale_dbz: Tensor,
    reference_dbz: Tensor,
    *,
    support_threshold_dbz: float,
) -> tuple[Tensor, Tensor]:
    """Stable float64 NLL and conditional-PIT residual for a lower truncation."""

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
    lower = (support_threshold_dbz - location) / scale
    standardized = (reference - location) / scale
    log_survival_lower = torch.special.log_ndtr(-lower)
    log_survival_reference = torch.special.log_ndtr(-standardized)
    nll = (
        0.5 * standardized.square()
        + torch.log(scale)
        + 0.5 * math.log(2.0 * math.pi)
        + log_survival_lower
    )
    log_survival_ratio = torch.minimum(
        log_survival_reference - log_survival_lower,
        torch.zeros((), dtype=torch.float64, device=location.device),
    )
    conditional_cdf = -torch.expm1(log_survival_ratio)
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


def _new_prior_holdout_evaluation(**values: object) -> PriorHoldoutEvaluation:
    """Internal constructor used only after forecast-derived values exist."""

    result = object.__new__(PriorHoldoutEvaluation)
    object.__setattr__(
        result,
        "contract",
        "prior-holdout-evaluation-v8",
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
            "day": value.day,
            "radar_id": value.radar_id,
            "regime": value.regime,
            "range_regime": value.range_regime,
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
            "issue_time": value.issue_time,
            "verification_valid_times": list(value.verification_valid_times),
        }
    )


@dataclass(frozen=True)
class NeuralPriorPromotionPolicy:
    """Root-approved cluster-aware limits for promoting one prior."""

    metric_scales: tuple[PromotionMetricScale, ...]
    approved_candidate_manifest_digests: tuple[str, ...]
    approved_holdout_plan_digests: tuple[str, ...]
    approved_metric_contract_digests: tuple[str, ...]
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
    maximum_prior_conditional_underdispersion_increase: float = 0.0
    minimum_bootstrap_tail_replicates: int = 20
    maximum_exact_sign_clusters: int = 16
    contract: str = "neural-prior-promotion-policy-v12"

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-promotion-policy-v12":
            raise ValueError("unsupported neural-prior promotion policy")
        if not self.metric_scales or len({x.metric_name for x in self.metric_scales}) != len(self.metric_scales):
            raise ValueError("promotion metric scales must be unique")
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
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("promotion fractions must be inside [0,1]")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("promotion confidence level must be inside (0,1)")
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
            "minimum_bootstrap_tail_replicates": (
                self.minimum_bootstrap_tail_replicates
            ),
            "maximum_exact_sign_clusters": self.maximum_exact_sign_clusters,
        })


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
class NeuralPriorPromotionEvidence:
    candidate_prior_digest: str
    parent_prior_digest: str
    candidate_manifest_digest: str
    policy_digest: str
    trust_store_digest: str
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
    certified_applicability_regime_groups: tuple[tuple[str, str], ...]
    requires_parent_fallback_outside_certified_applicability: bool
    eligible: bool
    rejection_reasons: tuple[PromotionRejectionReason, ...]
    contract: str = "neural-prior-promotion-evidence-v7"
    promotion_evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-promotion-evidence-v7":
            raise ValueError("unsupported neural-prior promotion evidence")
        for name in ("candidate_prior_digest", "parent_prior_digest", "candidate_manifest_digest", "policy_digest", "trust_store_digest"):
            _require_digest(name, getattr(self, name))
        for digest in self.evaluation_digests:
            _require_digest("promotion member digest", digest)
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
            or self.requires_parent_fallback_outside_certified_applicability
            is not True
            or len(set(self.certified_applicability_regime_groups))
            != len(self.certified_applicability_regime_groups)
            or any(
                not regime or not range_regime
                for regime, range_regime in self.certified_applicability_regime_groups
            )
        ):
            raise ValueError("promotion simultaneous-inference evidence is invalid")
        if self.eligible != (not self.rejection_reasons):
            raise ValueError("promotion eligibility and reasons disagree")
        object.__setattr__(self, "promotion_evidence_digest", json_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "promotion_evidence_digest"
        }


def _holdout_score(
    evaluation: PriorHoldoutEvaluation,
    policy: NeuralPriorPromotionPolicy,
) -> tuple[float | None, float, bool, bool]:
    scales = {item.metric_name: item for item in policy.metric_scales}
    values: list[Tensor] = []
    weights: list[Tensor] = []
    maximum_degradation = 0.0
    metric_limit_exceeded = False
    end_to_end_limit_exceeded = False
    for index, name in enumerate(evaluation.metric_names):
        if name not in scales:
            raise ValueError("promotion policy lacks an evaluation metric scale")
        item = scales[name]
        end_to_end = (
            evaluation.end_to_end_metric_change[:, index].masked_select(
                evaluation.metric_available[:, index]
            )
            / item.scale
        )
        if bool(torch.any(end_to_end > item.maximum_end_to_end_normalized_degradation)):
            end_to_end_limit_exceeded = True
        selected = evaluation.metric_available[:, index] & (
            torch.abs(evaluation.metric_change[:, index]) >= item.material_change
        )
        if not bool(torch.any(selected)):
            continue
        normalized = evaluation.metric_change[:, index].masked_select(selected) / item.scale
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


def _cluster_bounds(
    scores: list[float],
    clusters: list[tuple[str, str, str]],
    policy: NeuralPriorPromotionPolicy,
    *,
    candidate_family_size: int,
) -> tuple[float, float, float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for score, cluster in zip(scores, clusters, strict=True):
        grouped.setdefault(cluster, []).append(score)
    keys = sorted(grouped)
    cluster_scores = {
        key: sum(values) / len(values) for key, values in grouped.items()
    }
    generator = random.Random(0)
    beneficial: list[float] = []
    harmful: list[float] = []
    means: list[float] = []
    for _ in range(policy.bootstrap_samples):
        sample = [generator.choice(keys) for _ in keys]
        values = [cluster_scores[key] for key in sample]
        beneficial.append(sum(value > 0 for value in values) / len(values))
        harmful.append(sum(value < 0 for value in values) / len(values))
        means.append(sum(values) / len(values))
    alpha = (1.0 - policy.confidence_level) / (
        2.0 * candidate_family_size
    )
    def quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, int(probability * len(ordered))))]
    return quantile(beneficial, alpha), quantile(harmful, 1.0 - alpha), quantile(means, alpha)


@dataclass(frozen=True)
class _UncertaintyComparison:
    component: Literal[
        "intensity",
        "support",
        "echo_miss",
        "object_miss",
        "clear",
        "underdispersion",
    ]
    group: tuple[str, str] | None
    values: tuple[float, ...]
    clusters: tuple[tuple[str, str, str], ...]


def _cluster_means(
    comparison: _UncertaintyComparison,
) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
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
    for (comparison, _), mean, standard_error in zip(
        retained,
        observed,
        standard_errors,
        strict=True,
    ):
        bounds[comparison.component] = max(
            bounds.get(comparison.component, -math.inf),
            mean + critical * standard_error,
        )
    tail_replicates = effective_replicates * alpha
    monte_carlo_error = (
        0.0
        if method == "exact_sign_enumeration"
        else math.sqrt(alpha * (1.0 - alpha) / effective_replicates)
    )
    return _SimultaneousInferenceResult(
        bounds=bounds,
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
) -> NeuralPriorPromotionEvidence:
    """Evaluate a manifested candidate on independent, forecast-derived cases."""

    if any(type(item) is not PriorHoldoutEvaluation for item in evaluations):
        raise ValueError(
            "legacy promotion evaluations are audit-only and cannot be reused"
        )
    validate_neural_prior_holdout_plan(plan)
    validate_neural_prior_candidate_manifest(manifest)
    trust = _load_learning_policy_trust_store(policy_trust_store_path)
    reasons: list[PromotionRejectionReason] = []
    if policy.digest not in trust.approved_policy_digests:
        reasons.append("unapproved_promotion_policy")
    if manifest.manifest_digest not in policy.approved_candidate_manifest_digests:
        reasons.append("unapproved_candidate_manifest")
    if manifest.holdout_plan_digest != plan.plan_digest or (
        plan.plan_digest not in policy.approved_holdout_plan_digests
    ):
        reasons.append("unapproved_holdout_plan")
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
    clusters: list[tuple[str, str, str]] = []
    material_evaluations: list[PriorHoldoutEvaluation] = []
    maximum_degradation = 0.0
    uncertainty_records: dict[
        Literal[
            "intensity",
            "support",
            "echo_miss",
            "object_miss",
            "clear",
            "underdispersion",
        ],
        list[tuple[PriorHoldoutEvaluation, float, tuple[str, str, str]]],
    ] = {
        "intensity": [],
        "support": [],
        "echo_miss": [],
        "object_miss": [],
        "clear": [],
        "underdispersion": [],
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
        manifest.holdout_case(evaluation.case_id)
        uncertainty_cluster = (
            evaluation.storm_id,
            evaluation.day,
            evaluation.radar_id,
        )
        uncertainty_records["support"].append(
            (
                evaluation,
                evaluation.prior_support_brier_score
                - evaluation.parent_prior_support_brier_score,
                uncertainty_cluster,
            )
        )
        if evaluation.prior_echo_intensity_status == "available":
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
        if bool(torch.any(evaluation.coverage_parent - evaluation.coverage_candidate > policy.maximum_coverage_loss)):
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
            clusters.append((evaluation.storm_id, evaluation.day, evaluation.radar_id))
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
    regimes = {item.regime for item in material_evaluations}
    range_regimes = {item.range_regime for item in material_evaluations}
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
            candidate_family_size=len(plan.candidate_family_digests),
        )
    else:
        beneficial = harmful = mean = lower_beneficial = upper_harmful = lower_mean = 0.0
    family_size = len(plan.candidate_family_digests)
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
    groups = sorted({(item.regime, item.range_regime) for item in evaluations})
    certified_groups: list[tuple[str, str]] = []
    for group in groups:
        support_group = [
            item
            for item in uncertainty_records["support"]
            if (item[0].regime, item[0].range_regime) == group
        ]
        support_cases_ok = (
            len(support_group) >= policy.minimum_uncertainty_cases_per_regime
        )
        support_clusters_ok = len({item[2] for item in support_group}) >= (
            policy.minimum_uncertainty_clusters_per_regime
        )
        if not support_cases_ok:
            reasons.append("insufficient_uncertainty_clusters")
        if not support_clusters_ok:
            reasons.append("insufficient_uncertainty_clusters")
        echo_group = [
            item
            for item in echo_records
            if (item[0].regime, item[0].range_regime) == group
        ]
        echo_cases_ok = len(echo_group) >= policy.minimum_echo_cases_per_regime
        echo_clusters_ok = len({item[2] for item in echo_group}) >= (
            policy.minimum_echo_clusters_per_regime
        )
        if echo_group and not echo_cases_ok:
            reasons.append("insufficient_prior_echo_cases")
        if echo_group and not echo_clusters_ok:
            reasons.append("insufficient_echo_clusters")
        clear_group = [
            item
            for item in clear_records
            if (item[0].regime, item[0].range_regime) == group
        ]
        clear_cases_ok = len(clear_group) >= policy.minimum_clear_cases_per_regime
        clear_clusters_ok = len({item[2] for item in clear_group}) >= (
            policy.minimum_clear_clusters_per_regime
        )
        if clear_group and not clear_cases_ok:
            reasons.append("insufficient_prior_clear_cases")
        if clear_group and not clear_clusters_ok:
            reasons.append("insufficient_clear_clusters")
        if (
            support_cases_ok
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
        component_groups = sorted(
            {(item[0].regime, item[0].range_regime) for item in records}
        )
        for group in component_groups:
            selected = tuple(
                item
                for item in records
                if (item[0].regime, item[0].range_regime) == group
            )
            comparisons.append(
                _UncertaintyComparison(
                    component=component,
                    group=group,
                    values=tuple(item[1] for item in selected),
                    clusters=tuple(item[2] for item in selected),
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
    simultaneous_bounds = simultaneous.bounds
    intensity_nll_upper = simultaneous_bounds.get("intensity", 0.0)
    brier_upper = simultaneous_bounds["support"]
    echo_miss_upper = simultaneous_bounds.get("echo_miss", 0.0)
    object_miss_upper = simultaneous_bounds.get("object_miss", 0.0)
    clear_sky_upper = simultaneous_bounds.get("clear", 0.0)
    underdispersion_upper = simultaneous_bounds.get("underdispersion", 0.0)
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
    unique = tuple(dict.fromkeys(reasons))
    return NeuralPriorPromotionEvidence(
        candidate_prior_digest=manifest.candidate_prior_digest,
        parent_prior_digest=manifest.parent_prior_digest,
        candidate_manifest_digest=manifest.manifest_digest,
        policy_digest=policy.digest,
        trust_store_digest=trust.content_digest,
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
        certified_applicability_regime_groups=tuple(certified_groups),
        requires_parent_fallback_outside_certified_applicability=True,
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
