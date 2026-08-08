"""Provenance for realized observation interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Literal

import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest
from .sensitivity import (
    VariationalLearningImpact,
    _load_learning_policy_trust_store,
    validate_variational_learning_impact,
)
from .variational import AnalysisResult, P1LinearizationState


ObservationInterventionType = Literal[
    "realized_sensor_correction",
    "realized_qc_intervention",
    "operator_override",
]


@dataclass(frozen=True)
class InterventionMetricGuardrail:
    """Dimensionless benefit and harm limits for one forecast metric."""

    metric_name: str
    scale: float
    maximum_predicted_harm: float
    maximum_resolved_harm: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValueError("intervention metric name must be nonempty")
        for name, value in (
            ("scale", self.scale),
            ("maximum_predicted_harm", self.maximum_predicted_harm),
            ("maximum_resolved_harm", self.maximum_resolved_harm),
            ("weight", self.weight),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"intervention {name} must be positive")


@dataclass(frozen=True)
class InterventionExecutionPolicy:
    """Safety policy applied before an approved counterfactual is executed."""

    metric_guardrails: tuple[InterventionMetricGuardrail, ...]
    allowed_intervention_types: tuple[ObservationInterventionType, ...]
    minimum_predicted_normalized_benefit: float = 0.0
    minimum_resolved_normalized_benefit: float = 0.0
    approval_class: Literal["operator", "automation"] = "operator"
    contract: str = "observation-intervention-execution-policy-v1"

    def __post_init__(self) -> None:
        if self.contract != "observation-intervention-execution-policy-v1":
            raise ValueError("unsupported intervention execution policy")
        if not self.metric_guardrails or len(
            {item.metric_name for item in self.metric_guardrails}
        ) != len(self.metric_guardrails):
            raise ValueError("intervention metric guardrails must be unique")
        if not self.allowed_intervention_types or len(
            set(self.allowed_intervention_types)
        ) != len(self.allowed_intervention_types):
            raise ValueError("intervention types must be unique")
        for name, value in (
            (
                "minimum_predicted_normalized_benefit",
                self.minimum_predicted_normalized_benefit,
            ),
            (
                "minimum_resolved_normalized_benefit",
                self.minimum_resolved_normalized_benefit,
            ),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.approval_class not in ("operator", "automation"):
            raise ValueError("unsupported intervention approval class")

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "metric_guardrails": [
                    {
                        "metric_name": item.metric_name,
                        "scale": item.scale,
                        "maximum_predicted_harm": item.maximum_predicted_harm,
                        "maximum_resolved_harm": item.maximum_resolved_harm,
                        "weight": item.weight,
                    }
                    for item in self.metric_guardrails
                ],
                "allowed_intervention_types": sorted(
                    self.allowed_intervention_types
                ),
                "minimum_predicted_normalized_benefit": (
                    self.minimum_predicted_normalized_benefit
                ),
                "minimum_resolved_normalized_benefit": (
                    self.minimum_resolved_normalized_benefit
                ),
                "approval_class": self.approval_class,
            }
        )


def _require_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_time(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("intervention time must be an ISO-8601 string")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("intervention time must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("intervention time must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RealizedObservationIntervention:
    """A learning-approved counterfactual that was actually applied."""

    intervention_id: str
    intervention_type: ObservationInterventionType
    action_digest: str
    applied_time: str
    actual_input_before_digest: str
    actual_input_after_digest: str
    outcome_resolution_contract_digest: str
    execution_policy_digest: str
    execution_trust_store_digest: str
    predicted_normalized_benefit: float
    resolved_normalized_benefit: float
    learning_result_digest: str
    learning_approval_evidence_digest: str
    counterfactual_perturbation_digest: str
    linearization_digest: str
    contract: str = "realized-observation-intervention-v2"
    intervention_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "realized-observation-intervention-v2":
            raise ValueError("unsupported realized observation intervention")
        if (
            not isinstance(self.intervention_id, str)
            or not self.intervention_id
            or self.intervention_id.strip() != self.intervention_id
        ):
            raise ValueError("intervention_id must be nonempty and canonical")
        if self.intervention_type not in (
            "realized_sensor_correction",
            "realized_qc_intervention",
            "operator_override",
        ):
            raise ValueError("unsupported observation intervention type")
        object.__setattr__(self, "applied_time", _canonical_time(self.applied_time))
        for name in (
            "action_digest",
            "actual_input_before_digest",
            "actual_input_after_digest",
            "outcome_resolution_contract_digest",
            "execution_policy_digest",
            "execution_trust_store_digest",
            "learning_result_digest",
            "learning_approval_evidence_digest",
            "counterfactual_perturbation_digest",
            "linearization_digest",
        ):
            _require_digest(name, getattr(self, name))
        if self.actual_input_before_digest == self.actual_input_after_digest:
            raise ValueError("a realized intervention must change its input")
        for name in (
            "predicted_normalized_benefit",
            "resolved_normalized_benefit",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(
            self,
            "intervention_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "intervention_id": self.intervention_id,
                    "intervention_type": self.intervention_type,
                    "action_digest": self.action_digest,
                    "applied_time": self.applied_time,
                    "actual_input_before_digest": (
                        self.actual_input_before_digest
                    ),
                    "actual_input_after_digest": self.actual_input_after_digest,
                    "outcome_resolution_contract_digest": (
                        self.outcome_resolution_contract_digest
                    ),
                    "execution_policy_digest": self.execution_policy_digest,
                    "execution_trust_store_digest": (
                        self.execution_trust_store_digest
                    ),
                    "predicted_normalized_benefit": (
                        self.predicted_normalized_benefit
                    ),
                    "resolved_normalized_benefit": (
                        self.resolved_normalized_benefit
                    ),
                    "learning_result_digest": self.learning_result_digest,
                    "learning_approval_evidence_digest": (
                        self.learning_approval_evidence_digest
                    ),
                    "counterfactual_perturbation_digest": (
                        self.counterfactual_perturbation_digest
                    ),
                    "linearization_digest": self.linearization_digest,
                }
            ),
        )

    @classmethod
    def from_learning_result(
        cls,
        learning: VariationalLearningImpact,
        analysis: AnalysisResult | P1LinearizationState,
        *,
        intervention_id: str,
        intervention_type: ObservationInterventionType,
        action_digest: str,
        applied_time: str,
        actual_input_before: Tensor,
        actual_input_after: Tensor,
        execution_policy: InterventionExecutionPolicy,
        execution_policy_trust_store_path: str,
    ) -> RealizedObservationIntervention:
        """Bind immutable real inputs to one approved physical perturbation."""

        validate_variational_learning_impact(learning)
        evidence = learning.approval_evidence
        fsoi = learning.fsoi
        linearization = analysis.linearization
        if not learning.eligibility.eligible or evidence is None or fsoi is None:
            raise ValueError("realized intervention requires eligible learning")
        validation = learning.first_order_validation
        if validation is None:
            raise ValueError("realized intervention requires resolved validation")
        if linearization is None:
            raise ValueError("realized intervention requires a linearization")
        if fsoi.fso.linearization_digest != linearization.linearization_digest:
            raise ValueError("learning and intervention linearizations disagree")
        if intervention_type not in execution_policy.allowed_intervention_types:
            raise ValueError("intervention type is not approved for execution")
        trust = _load_learning_policy_trust_store(
            execution_policy_trust_store_path
        )
        if execution_policy.digest not in trust.approved_policy_digests:
            raise ValueError("intervention execution policy is not approved")
        if not isinstance(actual_input_before, Tensor) or not isinstance(
            actual_input_after,
            Tensor,
        ):
            raise TypeError("realized intervention inputs must be Tensors")
        retained = linearization.frozen.input_frames_dbz
        if actual_input_before.shape != retained.shape or (
            tensor_digest(actual_input_before) != tensor_digest(retained)
        ):
            raise ValueError("realized intervention before-input mismatch")
        delta = fsoi.perturbation.physical_radar_dbz_delta
        if delta is None:
            raise ValueError("realized intervention requires a physical delta")
        if actual_input_after.shape != actual_input_before.shape:
            raise ValueError("realized intervention input shapes disagree")
        expected = actual_input_before + delta.to(actual_input_before)
        finite = torch.isfinite(expected) & torch.isfinite(actual_input_after)
        same_nonfinite = torch.equal(
            torch.isfinite(expected),
            torch.isfinite(actual_input_after),
        )
        tolerance = 8.0 * torch.finfo(actual_input_before.dtype).eps
        if not same_nonfinite or not torch.allclose(
            expected[finite],
            actual_input_after[finite],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "realized input change disagrees with the approved perturbation"
            )
        metric_names = fsoi.fso.metric_names
        guardrails = {
            item.metric_name: item for item in execution_policy.metric_guardrails
        }
        if any(name not in guardrails for name in metric_names):
            raise ValueError("execution policy lacks a forecast metric")
        predicted = fsoi.observation.total.sum_by_time.sum(dim=-1)
        resolved = validation.full_step_resolved_metric_change
        available = validation.metric_available
        predicted_benefit = _normalized_benefit(
            predicted, available, metric_names, guardrails
        )
        resolved_benefit = _normalized_benefit(
            resolved, available, metric_names, guardrails
        )
        for metric_index, metric_name in enumerate(metric_names):
            selected = available[:, metric_index]
            if not bool(torch.any(selected)):
                continue
            guardrail = guardrails[metric_name]
            predicted_harm = torch.clamp(
                predicted[:, metric_index].masked_select(selected), min=0.0
            )
            resolved_harm = torch.clamp(
                resolved[:, metric_index].masked_select(selected), min=0.0
            )
            if float(torch.amax(predicted_harm)) > (
                guardrail.maximum_predicted_harm
            ):
                raise ValueError("predicted intervention harm exceeds policy")
            if float(torch.amax(resolved_harm)) > guardrail.maximum_resolved_harm:
                raise ValueError("resolved intervention harm exceeds policy")
        if execution_policy.approval_class == "automation":
            if predicted_benefit < (
                execution_policy.minimum_predicted_normalized_benefit
            ):
                raise ValueError("predicted intervention benefit is insufficient")
            if resolved_benefit < (
                execution_policy.minimum_resolved_normalized_benefit
            ):
                raise ValueError("resolved intervention benefit is insufficient")
        return cls(
            intervention_id=intervention_id,
            intervention_type=intervention_type,
            action_digest=action_digest,
            applied_time=applied_time,
            actual_input_before_digest=tensor_digest(actual_input_before),
            actual_input_after_digest=tensor_digest(actual_input_after),
            outcome_resolution_contract_digest=json_digest(
                {"contract": "resolved-intervention-outcome-from-forecasts-v1"}
            ),
            execution_policy_digest=execution_policy.digest,
            execution_trust_store_digest=trust.content_digest,
            predicted_normalized_benefit=predicted_benefit,
            resolved_normalized_benefit=resolved_benefit,
            learning_result_digest=learning.learning_result_digest,
            learning_approval_evidence_digest=evidence.digest,
            counterfactual_perturbation_digest=fsoi.perturbation_digest,
            linearization_digest=linearization.linearization_digest,
        )


def validate_realized_observation_intervention(
    intervention: RealizedObservationIntervention,
) -> None:
    """Validate immutable realized-action evidence before promotion."""

    reconstructed = RealizedObservationIntervention(
        intervention_id=intervention.intervention_id,
        intervention_type=intervention.intervention_type,
        action_digest=intervention.action_digest,
        applied_time=intervention.applied_time,
        actual_input_before_digest=intervention.actual_input_before_digest,
        actual_input_after_digest=intervention.actual_input_after_digest,
        outcome_resolution_contract_digest=(
            intervention.outcome_resolution_contract_digest
        ),
        execution_policy_digest=intervention.execution_policy_digest,
        execution_trust_store_digest=intervention.execution_trust_store_digest,
        predicted_normalized_benefit=(
            intervention.predicted_normalized_benefit
        ),
        resolved_normalized_benefit=intervention.resolved_normalized_benefit,
        learning_result_digest=intervention.learning_result_digest,
        learning_approval_evidence_digest=(
            intervention.learning_approval_evidence_digest
        ),
        counterfactual_perturbation_digest=(
            intervention.counterfactual_perturbation_digest
        ),
        linearization_digest=intervention.linearization_digest,
        contract=intervention.contract,
    )
    if reconstructed.intervention_digest != intervention.intervention_digest:
        raise ValueError("realized intervention digest mismatch")


def _normalized_benefit(
    change: Tensor,
    available: Tensor,
    metric_names: tuple[str, ...],
    guardrails: dict[str, InterventionMetricGuardrail],
) -> float:
    values: list[Tensor] = []
    weights: list[Tensor] = []
    for metric_index, metric_name in enumerate(metric_names):
        selected = available[:, metric_index]
        if not bool(torch.any(selected)):
            continue
        guardrail = guardrails[metric_name]
        metric_change = change[:, metric_index].masked_select(selected)
        values.append(-metric_change / guardrail.scale)
        weights.append(
            torch.full_like(metric_change, guardrail.weight)
        )
    if not values:
        raise ValueError("intervention execution requires an available metric")
    value = torch.cat(values)
    weight = torch.cat(weights)
    return float(torch.sum(value * weight) / torch.sum(weight))
