"""Fail-closed holdout evidence for learned radar priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import math
from pathlib import Path
import random
from typing import Literal

import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest
from .intervention import (
    ObservationInterventionType,
    RealizedObservationIntervention,
    validate_realized_observation_intervention,
)
from .nowcast import ForecastResult
from .sensitivity import (
    LearningApprovalEvidence,
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


PromotionRejectionReason = Literal[
    "unapproved_promotion_policy",
    "unapproved_candidate_manifest",
    "unapproved_holdout_plan",
    "unapproved_metric_contract",
    "insufficient_realized_interventions",
    "insufficient_material_interventions",
    "insufficient_material_intervention_fraction",
    "insufficient_independent_cases",
    "insufficient_distinct_storms",
    "insufficient_distinct_days",
    "insufficient_distinct_radars",
    "insufficient_distinct_regimes",
    "insufficient_distinct_range_regimes",
    "insufficient_material_clusters",
    "unapproved_learning_policy",
    "unsupported_intervention_type",
    "no_material_outcome",
    "insufficient_beneficial_fraction",
    "excessive_harmful_fraction",
    "insufficient_mean_improvement",
    "excessive_single_degradation",
    "excessive_issuance_change",
]


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
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"promotion metric {name} must be positive")


@dataclass(frozen=True)
class NeuralPriorHoldoutPlanCase:
    """One case committed before forecasts and verification are available."""

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

    def __post_init__(self) -> None:
        _validate_holdout_case_identity(self)
        for name in (
            "input_bundle_digest",
            "verification_plan_digest",
            "metric_contract_digest",
        ):
            _require_digest(name, getattr(self, name))
        object.__setattr__(self, "issue_time", _canonical_time(self.issue_time))


@dataclass(frozen=True)
class NeuralPriorHoldoutPlan:
    """Root-approved holdout commitment created before any evaluated issue."""

    plan_id: str
    parent_prior_digest: str
    candidate_family_digests: tuple[str, ...]
    cases: tuple[NeuralPriorHoldoutPlanCase, ...]
    registered_at: str
    contract: str = "neural-prior-holdout-plan-v1"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-holdout-plan-v1":
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
        input_digests = tuple(item.input_bundle_digest for item in self.cases)
        if len(set(input_digests)) != len(input_digests):
            raise ValueError("holdout cases must use distinct inputs")
        registered = _canonical_time(self.registered_at)
        if any(registered >= item.issue_time for item in self.cases):
            raise ValueError("holdout plan must precede every issue time")
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
    input_bundle_digest: str
    verification_plan_digest: str
    verification_bundle_digest: str
    metric_contract_digest: str
    issue_time: str
    candidate_forecast_digest: str
    parent_forecast_digest: str

    def __post_init__(self) -> None:
        _validate_holdout_case_identity(self)
        for name in (
            "input_bundle_digest",
            "verification_plan_digest",
            "verification_bundle_digest",
            "metric_contract_digest",
            "candidate_forecast_digest",
            "parent_forecast_digest",
        ):
            _require_digest(name, getattr(self, name))
        object.__setattr__(self, "issue_time", _canonical_time(self.issue_time))
        if self.candidate_forecast_digest == self.parent_forecast_digest:
            raise ValueError("candidate and parent holdout forecasts must differ")

    def plan_case(self) -> NeuralPriorHoldoutPlanCase:
        return NeuralPriorHoldoutPlanCase(
            case_id=self.case_id,
            storm_id=self.storm_id,
            day=self.day,
            radar_id=self.radar_id,
            regime=self.regime,
            range_regime=self.range_regime,
            input_bundle_digest=self.input_bundle_digest,
            verification_plan_digest=self.verification_plan_digest,
            metric_contract_digest=self.metric_contract_digest,
            issue_time=self.issue_time,
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
        "registered_at": plan.registered_at,
    }


def _holdout_dataset_digest(
    cases: tuple[NeuralPriorHoldoutPlanCase, ...],
) -> str:
    return json_digest(
        {
            "contract": "neural-prior-holdout-dataset-v1",
            "cases": [
                {
                    "case_id": item.case_id,
                    "input_bundle_digest": item.input_bundle_digest,
                    "verification_plan_digest": item.verification_plan_digest,
                    "metric_contract_digest": item.metric_contract_digest,
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
    holdout_cases: tuple[NeuralPriorHoldoutCase, ...]
    contract: str = "neural-prior-candidate-manifest-v1"
    manifest_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-candidate-manifest-v1":
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
class RealizedInterventionEvaluation:
    """Forecast-derived candidate-minus-parent change on one holdout case."""

    intervention_digest: str
    intervention_type: ObservationInterventionType
    learning_result_digest: str
    learning_approval_evidence_digest: str
    learning_policy_digest: str
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
    metric_change: Tensor
    candidate_issuance_effect: Tensor
    parent_issuance_effect: Tensor
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
    issue_time: str
    applied_time: str
    verification_valid_times: tuple[str, ...]
    contract: str = "realized-intervention-evaluation-v2"
    evaluation_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use RealizedInterventionEvaluation.from_forecasts")

    def __post_init__(self) -> None:
        if self.contract != "realized-intervention-evaluation-v2":
            raise ValueError("unsupported realized intervention evaluation")
        for name in (
            "intervention_digest",
            "learning_result_digest",
            "learning_approval_evidence_digest",
            "learning_policy_digest",
            "holdout_plan_digest",
            "candidate_manifest_digest",
            "candidate_prior_digest",
            "parent_prior_digest",
            "candidate_forecast_digest",
            "parent_forecast_digest",
            "verification_digest",
            "metric_contract_digest",
        ):
            _require_digest(name, getattr(self, name))
        expected = (len(self.lead_minutes), len(self.metric_names))
        change = self.metric_change.detach().clone()
        candidate_policy = self.candidate_issuance_effect.detach().clone()
        parent_policy = self.parent_issuance_effect.detach().clone()
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
            or available.shape != expected
            or available.dtype is not torch.bool
            or candidate_coverage.shape != (len(self.lead_minutes),)
            or parent_coverage.shape != (len(self.lead_minutes),)
            or common_coverage.shape != (len(self.lead_minutes),)
            or newly_issued.shape != (len(self.lead_minutes),)
            or withdrawn.shape != (len(self.lead_minutes),)
            or not change.is_floating_point()
        ):
            raise ValueError("realized evaluation shapes disagree")
        if not bool(torch.any(available)) or not bool(
            torch.all(torch.isfinite(change[available]))
        ):
            raise ValueError("realized evaluation must contain finite metrics")
        for coverage in (
            candidate_coverage,
            parent_coverage,
            common_coverage,
            newly_issued,
            withdrawn,
        ):
            if bool(torch.any((coverage < 0.0) | (coverage > 1.0))):
                raise ValueError("realized evaluation coverage must be in [0,1]")
        if self.candidate_prior_digest == self.parent_prior_digest:
            raise ValueError("candidate and parent priors must differ")
        applied = datetime.fromisoformat(self.applied_time.replace("Z", "+00:00"))
        issue = datetime.fromisoformat(self.issue_time.replace("Z", "+00:00"))
        valid = tuple(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in self.verification_valid_times
        )
        if applied > issue or any(issue >= value for value in valid):
            raise ValueError("intervention, issue, and verification times disagree")
        object.__setattr__(self, "metric_change", change)
        object.__setattr__(self, "candidate_issuance_effect", candidate_policy)
        object.__setattr__(self, "parent_issuance_effect", parent_policy)
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
        intervention: RealizedObservationIntervention,
        learning_evidence: LearningApprovalEvidence,
        manifest: NeuralPriorCandidateManifest,
        plan: NeuralPriorHoldoutPlan,
        *,
        case_id: str,
        candidate_forecast: ForecastResult,
        parent_forecast: ForecastResult,
        verification: VerificationBundle,
        metric_config: SensitivityConfig,
    ) -> RealizedInterventionEvaluation:
        """Recompute holdout change; no caller-provided outcome is accepted."""

        validate_realized_observation_intervention(intervention)
        validate_neural_prior_holdout_plan(plan)
        validate_neural_prior_candidate_manifest(manifest)
        if intervention.contract != "realized-observation-intervention-v3":
            raise ValueError("holdout evaluation requires causal v3 intervention")
        candidate_forecast.validate_issuance()
        parent_forecast.validate_issuance()
        if intervention.learning_approval_evidence_digest != learning_evidence.digest:
            raise ValueError("intervention and learning approval disagree")
        if manifest.holdout_plan_digest != plan.plan_digest:
            raise ValueError("candidate manifest and holdout plan disagree")
        if manifest.candidate_prior_digest not in plan.candidate_family_digests or (
            manifest.parent_prior_digest != plan.parent_prior_digest
        ):
            raise ValueError("candidate priors are outside the holdout plan")
        case = manifest.holdout_case(case_id)
        planned_case = plan.case(case_id)
        if case.plan_case() != planned_case:
            raise ValueError("completed holdout case disagrees with its plan")
        if (
            intervention.case_id != case.case_id
            or intervention.radar_id != case.radar_id
            or intervention.issue_time != case.issue_time
            or intervention.input_bundle_after_digest != case.input_bundle_digest
        ):
            raise ValueError("intervention does not belong to the holdout case")
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
        ):
            raise ValueError("candidate and parent holdout inputs disagree")
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
        return _new_realized_evaluation(
            intervention_digest=intervention.intervention_digest,
            intervention_type=intervention.intervention_type,
            learning_result_digest=intervention.learning_result_digest,
            learning_approval_evidence_digest=learning_evidence.digest,
            learning_policy_digest=learning_evidence.policy_digest,
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
            metric_change=change,
            candidate_issuance_effect=candidate_policy,
            parent_issuance_effect=parent_policy,
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
            issue_time=issue_time,
            applied_time=intervention.applied_time,
            verification_valid_times=verification.valid_times,
        )


def _new_realized_evaluation(**values: object) -> RealizedInterventionEvaluation:
    """Internal constructor used only after forecast-derived values exist."""

    result = object.__new__(RealizedInterventionEvaluation)
    object.__setattr__(
        result,
        "contract",
        "realized-intervention-evaluation-v2",
    )
    for name, value in values.items():
        object.__setattr__(result, name, value)
    RealizedInterventionEvaluation.__post_init__(result)
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


def _evaluation_digest(value: RealizedInterventionEvaluation) -> str:
    return json_digest(
        {
            "contract": value.contract,
            "intervention_digest": value.intervention_digest,
            "intervention_type": value.intervention_type,
            "learning_result_digest": value.learning_result_digest,
            "learning_approval_evidence_digest": value.learning_approval_evidence_digest,
            "learning_policy_digest": value.learning_policy_digest,
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
            "metric_change": tensor_digest(value.metric_change),
            "candidate_issuance_effect": tensor_digest(
                value.candidate_issuance_effect
            ),
            "parent_issuance_effect": tensor_digest(value.parent_issuance_effect),
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
            "issue_time": value.issue_time,
            "applied_time": value.applied_time,
            "verification_valid_times": list(value.verification_valid_times),
        }
    )


@dataclass(frozen=True)
class NeuralPriorPromotionPolicy:
    """Root-approved cluster-aware limits for promoting one prior."""

    metric_scales: tuple[PromotionMetricScale, ...]
    approved_learning_policy_digests: tuple[str, ...]
    approved_candidate_manifest_digests: tuple[str, ...]
    approved_holdout_plan_digests: tuple[str, ...]
    approved_metric_contract_digests: tuple[str, ...]
    allowed_intervention_types: tuple[ObservationInterventionType, ...]
    minimum_realized_interventions: int = 20
    minimum_material_interventions: int = 20
    minimum_material_intervention_fraction: float = 0.8
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
    contract: str = "neural-prior-promotion-policy-v3"

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-promotion-policy-v3":
            raise ValueError("unsupported neural-prior promotion policy")
        if not self.metric_scales or len({x.metric_name for x in self.metric_scales}) != len(self.metric_scales):
            raise ValueError("promotion metric scales must be unique")
        for digest in self.approved_learning_policy_digests:
            _require_digest("approved learning policy digest", digest)
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
        if not self.allowed_intervention_types or len(
            set(self.allowed_intervention_types)
        ) != len(self.allowed_intervention_types):
            raise ValueError("promotion intervention types must be unique")
        integer_limits = (
            self.minimum_realized_interventions,
            self.minimum_material_interventions,
            self.minimum_independent_cases,
            self.minimum_distinct_storms,
            self.minimum_distinct_days,
            self.minimum_distinct_radars,
            self.minimum_distinct_regimes,
            self.minimum_distinct_range_regimes,
            self.minimum_material_clusters,
            self.bootstrap_samples,
        )
        if any(type(value) is not int or value <= 0 for value in integer_limits):
            raise ValueError("promotion count limits must be positive integers")
        probabilities = (
            self.minimum_material_intervention_fraction,
            self.minimum_beneficial_fraction,
            self.maximum_harmful_fraction,
            self.confidence_level,
            self.maximum_coverage_loss,
            self.maximum_newly_issued_fraction,
            self.maximum_withdrawn_fraction,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("promotion fractions must be inside [0,1]")
        for name, value in (
            (
                "minimum_mean_normalized_improvement",
                self.minimum_mean_normalized_improvement,
            ),
            (
                "maximum_single_normalized_degradation",
                self.maximum_single_normalized_degradation,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def digest(self) -> str:
        return json_digest({
            "contract": self.contract,
            "metric_scales": [item.__dict__ for item in self.metric_scales],
            "approved_learning_policy_digests": sorted(self.approved_learning_policy_digests),
            "approved_candidate_manifest_digests": sorted(
                self.approved_candidate_manifest_digests
            ),
            "approved_holdout_plan_digests": sorted(
                self.approved_holdout_plan_digests
            ),
            "approved_metric_contract_digests": sorted(
                self.approved_metric_contract_digests
            ),
            "allowed_intervention_types": sorted(self.allowed_intervention_types),
            "minimum_realized_interventions": self.minimum_realized_interventions,
            "minimum_material_interventions": self.minimum_material_interventions,
            "minimum_material_intervention_fraction": self.minimum_material_intervention_fraction,
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
        })


@dataclass(frozen=True)
class NeuralPriorPromotionEvidence:
    candidate_prior_digest: str
    parent_prior_digest: str
    candidate_manifest_digest: str
    policy_digest: str
    trust_store_digest: str
    evaluation_digests: tuple[str, ...]
    intervention_digests: tuple[str, ...]
    realized_intervention_count: int
    material_intervention_count: int
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
    eligible: bool
    rejection_reasons: tuple[PromotionRejectionReason, ...]
    contract: str = "neural-prior-promotion-evidence-v2"
    promotion_evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-promotion-evidence-v2":
            raise ValueError("unsupported neural-prior promotion evidence")
        for name in ("candidate_prior_digest", "parent_prior_digest", "candidate_manifest_digest", "policy_digest", "trust_store_digest"):
            _require_digest(name, getattr(self, name))
        for digest in self.evaluation_digests + self.intervention_digests:
            _require_digest("promotion member digest", digest)
        if len(self.evaluation_digests) != len(self.intervention_digests) or (
            self.realized_intervention_count != len(self.evaluation_digests)
        ):
            raise ValueError("promotion evidence member counts disagree")
        counts = (
            self.material_intervention_count,
            self.distinct_case_count,
            self.distinct_storm_count,
            self.distinct_day_count,
            self.distinct_radar_count,
            self.distinct_regime_count,
            self.distinct_range_regime_count,
        )
        if any(type(value) is not int or value < 0 for value in counts) or any(
            value > self.realized_intervention_count for value in counts
        ):
            raise ValueError("promotion evidence counts are invalid")
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
        if self.eligible != (not self.rejection_reasons):
            raise ValueError("promotion eligibility and reasons disagree")
        object.__setattr__(self, "promotion_evidence_digest", json_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "promotion_evidence_digest"
        }


def _intervention_score(
    evaluation: RealizedInterventionEvaluation,
    policy: NeuralPriorPromotionPolicy,
) -> tuple[float | None, float, bool]:
    scales = {item.metric_name: item for item in policy.metric_scales}
    values: list[Tensor] = []
    weights: list[Tensor] = []
    maximum_degradation = 0.0
    metric_limit_exceeded = False
    for index, name in enumerate(evaluation.metric_names):
        if name not in scales:
            raise ValueError("promotion policy lacks an evaluation metric scale")
        item = scales[name]
        selected = evaluation.metric_available[:, index] & (
            torch.abs(evaluation.metric_change[:, index]) >= item.material_change
        )
        if not bool(torch.any(selected)):
            continue
        normalized = evaluation.metric_change[:, index].masked_select(selected) / item.scale
        maximum_degradation = max(maximum_degradation, float(torch.amax(torch.clamp(normalized, min=0))))
        if maximum_degradation > item.maximum_normalized_degradation:
            metric_limit_exceeded = True
        values.append(-normalized)
        weights.append(torch.full_like(normalized, item.weight))
    if not values:
        return None, maximum_degradation, metric_limit_exceeded
    value = torch.cat(values)
    weight = torch.cat(weights)
    return (
        float(torch.sum(value * weight) / torch.sum(weight)),
        maximum_degradation,
        metric_limit_exceeded,
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


def compute_neural_prior_promotion(
    manifest: NeuralPriorCandidateManifest,
    plan: NeuralPriorHoldoutPlan,
    evaluations: tuple[RealizedInterventionEvaluation, ...],
    *,
    policy: NeuralPriorPromotionPolicy,
    policy_trust_store_path: str | Path,
) -> NeuralPriorPromotionEvidence:
    """Evaluate a manifested candidate on independent, forecast-derived cases."""

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
    if len(evaluations) < policy.minimum_realized_interventions:
        reasons.append("insufficient_realized_interventions")
    if len({item.intervention_digest for item in evaluations}) != len(evaluations):
        raise ValueError("promotion interventions must be unique")
    if set(manifest.training_intervention_digests) & {
        item.intervention_digest for item in evaluations
    }:
        raise ValueError("training and promotion interventions must be disjoint")
    scores: list[float] = []
    clusters: list[tuple[str, str, str]] = []
    material_evaluations: list[RealizedInterventionEvaluation] = []
    maximum_degradation = 0.0
    for evaluation in evaluations:
        if evaluation.evaluation_digest != _evaluation_digest(evaluation):
            raise ValueError("realized intervention evaluation digest mismatch")
        if (
            evaluation.candidate_manifest_digest != manifest.manifest_digest
            or evaluation.holdout_plan_digest != plan.plan_digest
            or evaluation.candidate_prior_digest != manifest.candidate_prior_digest
            or evaluation.parent_prior_digest != manifest.parent_prior_digest
        ):
            raise ValueError("evaluation does not belong to the candidate manifest")
        manifest.holdout_case(evaluation.case_id)
        if evaluation.metric_contract_digest not in (
            policy.approved_metric_contract_digests
        ):
            reasons.append("unapproved_metric_contract")
        if evaluation.learning_policy_digest not in policy.approved_learning_policy_digests:
            reasons.append("unapproved_learning_policy")
        if evaluation.intervention_type not in policy.allowed_intervention_types:
            reasons.append("unsupported_intervention_type")
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
        score, degradation, metric_limit_exceeded = _intervention_score(
            evaluation, policy
        )
        if metric_limit_exceeded:
            reasons.append("excessive_single_degradation")
        maximum_degradation = max(maximum_degradation, degradation)
        if score is not None:
            scores.append(score)
            clusters.append((evaluation.storm_id, evaluation.day, evaluation.radar_id))
            material_evaluations.append(evaluation)
    material_count = len(scores)
    if material_count == 0:
        reasons.append("no_material_outcome")
    if material_count < policy.minimum_material_interventions:
        reasons.append("insufficient_material_interventions")
    if material_count / max(1, len(evaluations)) < policy.minimum_material_intervention_fraction:
        reasons.append("insufficient_material_intervention_fraction")
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
    if lower_beneficial < policy.minimum_beneficial_fraction:
        reasons.append("insufficient_beneficial_fraction")
    if upper_harmful > policy.maximum_harmful_fraction:
        reasons.append("excessive_harmful_fraction")
    if lower_mean < policy.minimum_mean_normalized_improvement:
        reasons.append("insufficient_mean_improvement")
    if maximum_degradation > policy.maximum_single_normalized_degradation:
        reasons.append("excessive_single_degradation")
    unique = tuple(dict.fromkeys(reasons))
    return NeuralPriorPromotionEvidence(
        candidate_prior_digest=manifest.candidate_prior_digest,
        parent_prior_digest=manifest.parent_prior_digest,
        candidate_manifest_digest=manifest.manifest_digest,
        policy_digest=policy.digest,
        trust_store_digest=trust.content_digest,
        evaluation_digests=tuple(item.evaluation_digest for item in evaluations),
        intervention_digests=tuple(item.intervention_digest for item in evaluations),
        realized_intervention_count=len(evaluations),
        material_intervention_count=material_count,
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
        eligible=not unique,
        rejection_reasons=unique,
    )


def validate_neural_prior_promotion(evidence: NeuralPriorPromotionEvidence) -> None:
    if evidence.promotion_evidence_digest != json_digest(evidence._payload()):
        raise ValueError("neural-prior promotion evidence digest mismatch")
    if not evidence.eligible:
        raise ValueError("neural-prior promotion evidence is not eligible")
