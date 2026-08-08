"""Fail-closed promotion evidence for learned radar priors."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest
from .intervention import (
    ObservationInterventionType,
    RealizedObservationIntervention,
    validate_realized_observation_intervention,
)
from .sensitivity import (
    LearningApprovalEvidence,
    _load_learning_policy_trust_store,
)


PromotionRejectionReason = Literal[
    "unapproved_promotion_policy",
    "insufficient_realized_interventions",
    "unapproved_learning_policy",
    "unsupported_intervention_type",
    "no_material_outcome",
    "insufficient_beneficial_fraction",
    "excessive_harmful_fraction",
    "insufficient_mean_improvement",
    "excessive_single_degradation",
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
    """Dimensionally valid scale used to compare realized metric changes."""

    metric_name: str
    scale: float
    material_change: float

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValueError("promotion metric name must be nonempty")
        for name, value in (
            ("scale", self.scale),
            ("material_change", self.material_change),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"promotion metric {name} must be positive")


@dataclass(frozen=True)
class RealizedInterventionEvaluation:
    """Observed forecast-metric change for one realized intervention."""

    intervention_digest: str
    intervention_type: ObservationInterventionType
    learning_result_digest: str
    learning_approval_evidence_digest: str
    learning_policy_digest: str
    metric_change: Tensor
    metric_available: Tensor
    lead_minutes: tuple[int, ...]
    metric_names: tuple[str, ...]
    verification_digest: str
    contract: str = "realized-intervention-evaluation-v1"
    evaluation_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "realized-intervention-evaluation-v1":
            raise ValueError("unsupported realized intervention evaluation")
        for name in (
            "intervention_digest",
            "learning_result_digest",
            "learning_approval_evidence_digest",
            "learning_policy_digest",
            "verification_digest",
        ):
            _require_digest(name, getattr(self, name))
        if self.intervention_type not in (
            "realized_sensor_correction",
            "realized_qc_intervention",
            "operator_override",
        ):
            raise ValueError("unsupported realized intervention type")
        change = self.metric_change.detach().clone()
        available = self.metric_available.detach().clone()
        expected = (len(self.lead_minutes), len(self.metric_names))
        if (
            not change.is_floating_point()
            or change.shape != expected
            or available.dtype is not torch.bool
            or available.shape != expected
            or change.device != available.device
        ):
            raise ValueError("realized metric evaluation shapes disagree")
        if not bool(torch.any(available)) or not bool(
            torch.all(torch.isfinite(change[available]))
        ):
            raise ValueError("realized metric evaluation must be finite")
        if not self.lead_minutes or tuple(sorted(set(self.lead_minutes))) != (
            self.lead_minutes
        ):
            raise ValueError("realized evaluation leads must be sorted and unique")
        if not self.metric_names or len(set(self.metric_names)) != len(
            self.metric_names
        ):
            raise ValueError("realized evaluation metrics must be unique")
        object.__setattr__(self, "metric_change", change)
        object.__setattr__(self, "metric_available", available)
        object.__setattr__(
            self,
            "evaluation_digest",
            _realized_evaluation_digest(self),
        )

    @classmethod
    def from_evidence(
        cls,
        intervention: RealizedObservationIntervention,
        learning_evidence: LearningApprovalEvidence,
        *,
        metric_change: Tensor,
        metric_available: Tensor,
        lead_minutes: tuple[int, ...],
        metric_names: tuple[str, ...],
        verification_digest: str,
    ) -> RealizedInterventionEvaluation:
        validate_realized_observation_intervention(intervention)
        if (
            intervention.learning_approval_evidence_digest
            != learning_evidence.digest
        ):
            raise ValueError("intervention and learning approval disagree")
        if intervention.observed_outcome_digest != tensor_digest(metric_change):
            raise ValueError("realized metric change is not the observed outcome")
        return cls(
            intervention_digest=intervention.intervention_digest,
            intervention_type=intervention.intervention_type,
            learning_result_digest=intervention.learning_result_digest,
            learning_approval_evidence_digest=learning_evidence.digest,
            learning_policy_digest=learning_evidence.policy_digest,
            metric_change=metric_change,
            metric_available=metric_available,
            lead_minutes=lead_minutes,
            metric_names=metric_names,
            verification_digest=verification_digest,
        )


def _realized_evaluation_digest(
    evaluation: RealizedInterventionEvaluation,
) -> str:
    return json_digest(
        {
            "contract": evaluation.contract,
            "intervention_digest": evaluation.intervention_digest,
            "intervention_type": evaluation.intervention_type,
            "learning_result_digest": evaluation.learning_result_digest,
            "learning_approval_evidence_digest": (
                evaluation.learning_approval_evidence_digest
            ),
            "learning_policy_digest": evaluation.learning_policy_digest,
            "metric_change": tensor_digest(evaluation.metric_change),
            "metric_available": tensor_digest(evaluation.metric_available),
            "lead_minutes": list(evaluation.lead_minutes),
            "metric_names": list(evaluation.metric_names),
            "verification_digest": evaluation.verification_digest,
        }
    )


@dataclass(frozen=True)
class NeuralPriorPromotionPolicy:
    """Root-approved limits for promoting one learned prior artifact."""

    metric_scales: tuple[PromotionMetricScale, ...]
    approved_learning_policy_digests: tuple[str, ...]
    allowed_intervention_types: tuple[ObservationInterventionType, ...]
    minimum_realized_interventions: int = 20
    minimum_beneficial_fraction: float = 0.6
    maximum_harmful_fraction: float = 0.1
    minimum_mean_normalized_improvement: float = 0.05
    maximum_single_normalized_degradation: float = 1.0
    contract: str = "neural-prior-promotion-policy-v1"

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-promotion-policy-v1":
            raise ValueError("unsupported neural-prior promotion policy")
        if not self.metric_scales or len(
            {item.metric_name for item in self.metric_scales}
        ) != len(self.metric_scales):
            raise ValueError("promotion metric scales must be nonempty and unique")
        if not self.approved_learning_policy_digests:
            raise ValueError("promotion policy must approve learning policies")
        for digest in self.approved_learning_policy_digests:
            _require_digest("approved learning policy digest", digest)
        if not self.allowed_intervention_types or len(
            set(self.allowed_intervention_types)
        ) != len(self.allowed_intervention_types):
            raise ValueError("promotion intervention types must be unique")
        if (
            type(self.minimum_realized_interventions) is not int
            or self.minimum_realized_interventions <= 0
        ):
            raise ValueError("minimum realized interventions must be positive")
        probabilities = (
            self.minimum_beneficial_fraction,
            self.maximum_harmful_fraction,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in probabilities
        ):
            raise ValueError("promotion fractions must be inside [0, 1]")
        for name, value in (
            (
                "minimum mean normalized improvement",
                self.minimum_mean_normalized_improvement,
            ),
            (
                "maximum single normalized degradation",
                self.maximum_single_normalized_degradation,
            ),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "metric_scales": [
                    {
                        "metric_name": item.metric_name,
                        "scale": item.scale,
                        "material_change": item.material_change,
                    }
                    for item in self.metric_scales
                ],
                "approved_learning_policy_digests": sorted(
                    self.approved_learning_policy_digests
                ),
                "allowed_intervention_types": sorted(
                    self.allowed_intervention_types
                ),
                "minimum_realized_interventions": (
                    self.minimum_realized_interventions
                ),
                "minimum_beneficial_fraction": self.minimum_beneficial_fraction,
                "maximum_harmful_fraction": self.maximum_harmful_fraction,
                "minimum_mean_normalized_improvement": (
                    self.minimum_mean_normalized_improvement
                ),
                "maximum_single_normalized_degradation": (
                    self.maximum_single_normalized_degradation
                ),
            }
        )


@dataclass(frozen=True)
class NeuralPriorPromotionEvidence:
    """Content-addressed, fail-closed prior promotion decision."""

    candidate_prior_digest: str
    parent_prior_digest: str
    policy_digest: str
    trust_store_digest: str
    evaluation_digests: tuple[str, ...]
    intervention_digests: tuple[str, ...]
    realized_intervention_count: int
    material_outcome_count: int
    beneficial_fraction: float
    harmful_fraction: float
    mean_normalized_improvement: float
    maximum_normalized_degradation: float
    eligible: bool
    rejection_reasons: tuple[PromotionRejectionReason, ...]
    contract: str = "neural-prior-promotion-evidence-v1"
    promotion_evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "neural-prior-promotion-evidence-v1":
            raise ValueError("unsupported neural-prior promotion evidence")
        for name in (
            "candidate_prior_digest",
            "parent_prior_digest",
            "policy_digest",
            "trust_store_digest",
        ):
            _require_digest(name, getattr(self, name))
        for digest in self.evaluation_digests + self.intervention_digests:
            _require_digest("promotion member digest", digest)
        if len(self.evaluation_digests) != len(self.intervention_digests):
            raise ValueError("promotion evaluation and intervention counts disagree")
        if self.realized_intervention_count != len(self.intervention_digests):
            raise ValueError("promotion realized-intervention count disagrees")
        if (
            type(self.material_outcome_count) is not int
            or self.material_outcome_count < 0
        ):
            raise ValueError("promotion material-outcome count is invalid")
        for name, value in (
            ("beneficial_fraction", self.beneficial_fraction),
            ("harmful_fraction", self.harmful_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"promotion {name} must be inside [0, 1]")
        if not math.isfinite(self.mean_normalized_improvement) or (
            not math.isfinite(self.maximum_normalized_degradation)
            or self.maximum_normalized_degradation < 0.0
        ):
            raise ValueError("promotion normalized summaries are invalid")
        if self.eligible != (not self.rejection_reasons):
            raise ValueError("promotion eligibility and reasons disagree")
        object.__setattr__(
            self,
            "promotion_evidence_digest",
            _promotion_evidence_digest(self),
        )


def _promotion_evidence_digest(
    evidence: NeuralPriorPromotionEvidence,
) -> str:
    return json_digest(
        {
            "contract": evidence.contract,
            "candidate_prior_digest": evidence.candidate_prior_digest,
            "parent_prior_digest": evidence.parent_prior_digest,
            "policy_digest": evidence.policy_digest,
            "trust_store_digest": evidence.trust_store_digest,
            "evaluation_digests": list(evidence.evaluation_digests),
            "intervention_digests": list(evidence.intervention_digests),
            "realized_intervention_count": evidence.realized_intervention_count,
            "material_outcome_count": evidence.material_outcome_count,
            "beneficial_fraction": evidence.beneficial_fraction,
            "harmful_fraction": evidence.harmful_fraction,
            "mean_normalized_improvement": (
                evidence.mean_normalized_improvement
            ),
            "maximum_normalized_degradation": (
                evidence.maximum_normalized_degradation
            ),
            "eligible": evidence.eligible,
            "rejection_reasons": list(evidence.rejection_reasons),
        }
    )


def compute_neural_prior_promotion(
    candidate_prior_digest: str,
    parent_prior_digest: str,
    evaluations: tuple[RealizedInterventionEvaluation, ...],
    *,
    policy: NeuralPriorPromotionPolicy,
    policy_trust_store_path: str | Path,
) -> NeuralPriorPromotionEvidence:
    """Evaluate one candidate prior using only realized, approved outcomes."""

    _require_digest("candidate_prior_digest", candidate_prior_digest)
    _require_digest("parent_prior_digest", parent_prior_digest)
    if candidate_prior_digest == parent_prior_digest:
        raise ValueError("candidate and parent prior digests must differ")
    trust = _load_learning_policy_trust_store(policy_trust_store_path)
    reasons: list[PromotionRejectionReason] = []
    if policy.digest not in trust.approved_policy_digests:
        reasons.append("unapproved_promotion_policy")
    if len(evaluations) < policy.minimum_realized_interventions:
        reasons.append("insufficient_realized_interventions")
    if len({item.intervention_digest for item in evaluations}) != len(
        evaluations
    ):
        raise ValueError("promotion interventions must be unique")

    normalized_changes: list[Tensor] = []
    for evaluation in evaluations:
        if evaluation.evaluation_digest != _realized_evaluation_digest(
            evaluation
        ):
            raise ValueError("realized intervention evaluation digest mismatch")
        if evaluation.learning_policy_digest not in (
            policy.approved_learning_policy_digests
        ):
            reasons.append("unapproved_learning_policy")
        if evaluation.intervention_type not in policy.allowed_intervention_types:
            reasons.append("unsupported_intervention_type")
        scale_by_name = {item.metric_name: item for item in policy.metric_scales}
        if any(name not in scale_by_name for name in evaluation.metric_names):
            raise ValueError("promotion policy lacks an evaluation metric scale")
        scale = evaluation.metric_change.new_tensor(
            tuple(scale_by_name[name].scale for name in evaluation.metric_names)
        )
        material = evaluation.metric_change.new_tensor(
            tuple(
                scale_by_name[name].material_change
                for name in evaluation.metric_names
            )
        )
        selected = evaluation.metric_available & (
            torch.abs(evaluation.metric_change) >= material
        )
        if bool(torch.any(selected)):
            normalized_changes.append(
                (evaluation.metric_change / scale).masked_select(selected)
            )

    if normalized_changes:
        changes = torch.cat(normalized_changes)
        improvement = -changes
        material_count = int(changes.numel())
        beneficial_fraction = float(torch.mean((changes < 0).to(changes.dtype)))
        harmful_fraction = float(torch.mean((changes > 0).to(changes.dtype)))
        mean_improvement = float(torch.mean(improvement))
        maximum_degradation = float(torch.amax(torch.clamp(changes, min=0.0)))
    else:
        material_count = 0
        beneficial_fraction = 0.0
        harmful_fraction = 0.0
        mean_improvement = 0.0
        maximum_degradation = 0.0
        reasons.append("no_material_outcome")
    if beneficial_fraction < policy.minimum_beneficial_fraction:
        reasons.append("insufficient_beneficial_fraction")
    if harmful_fraction > policy.maximum_harmful_fraction:
        reasons.append("excessive_harmful_fraction")
    if mean_improvement < policy.minimum_mean_normalized_improvement:
        reasons.append("insufficient_mean_improvement")
    if maximum_degradation > policy.maximum_single_normalized_degradation:
        reasons.append("excessive_single_degradation")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return NeuralPriorPromotionEvidence(
        candidate_prior_digest=candidate_prior_digest,
        parent_prior_digest=parent_prior_digest,
        policy_digest=policy.digest,
        trust_store_digest=trust.content_digest,
        evaluation_digests=tuple(item.evaluation_digest for item in evaluations),
        intervention_digests=tuple(
            item.intervention_digest for item in evaluations
        ),
        realized_intervention_count=len(evaluations),
        material_outcome_count=material_count,
        beneficial_fraction=beneficial_fraction,
        harmful_fraction=harmful_fraction,
        mean_normalized_improvement=mean_improvement,
        maximum_normalized_degradation=maximum_degradation,
        eligible=not unique_reasons,
        rejection_reasons=unique_reasons,
    )


def validate_neural_prior_promotion(
    evidence: NeuralPriorPromotionEvidence,
) -> None:
    """Validate promotion evidence immediately before activating a prior."""

    if evidence.promotion_evidence_digest != _promotion_evidence_digest(evidence):
        raise ValueError("neural-prior promotion evidence digest mismatch")
    if not evidence.eligible:
        raise ValueError("neural-prior promotion evidence is not eligible")
