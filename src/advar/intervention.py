"""Provenance for realized observation interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest
from .sensitivity import (
    VariationalLearningImpact,
    validate_variational_learning_impact,
)
from .variational import AnalysisResult, P1LinearizationState


ObservationInterventionType = Literal[
    "realized_sensor_correction",
    "realized_qc_intervention",
    "operator_override",
]


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
    observed_outcome_digest: str
    learning_result_digest: str
    learning_approval_evidence_digest: str
    counterfactual_perturbation_digest: str
    linearization_digest: str
    contract: str = "realized-observation-intervention-v1"
    intervention_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "realized-observation-intervention-v1":
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
            "observed_outcome_digest",
            "learning_result_digest",
            "learning_approval_evidence_digest",
            "counterfactual_perturbation_digest",
            "linearization_digest",
        ):
            _require_digest(name, getattr(self, name))
        if self.actual_input_before_digest == self.actual_input_after_digest:
            raise ValueError("a realized intervention must change its input")
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
                    "observed_outcome_digest": self.observed_outcome_digest,
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
        observed_outcome: Tensor,
    ) -> RealizedObservationIntervention:
        """Bind immutable real inputs to one approved physical perturbation."""

        validate_variational_learning_impact(learning)
        evidence = learning.approval_evidence
        fsoi = learning.fsoi
        linearization = analysis.linearization
        if not learning.eligibility.eligible or evidence is None or fsoi is None:
            raise ValueError("realized intervention requires eligible learning")
        if linearization is None:
            raise ValueError("realized intervention requires a linearization")
        if fsoi.fso.linearization_digest != linearization.linearization_digest:
            raise ValueError("learning and intervention linearizations disagree")
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
        if not isinstance(observed_outcome, Tensor) or (
            not observed_outcome.is_floating_point()
            or not bool(torch.all(torch.isfinite(observed_outcome)))
        ):
            raise ValueError("observed outcome must be a finite floating Tensor")
        return cls(
            intervention_id=intervention_id,
            intervention_type=intervention_type,
            action_digest=action_digest,
            applied_time=applied_time,
            actual_input_before_digest=tensor_digest(actual_input_before),
            actual_input_after_digest=tensor_digest(actual_input_after),
            observed_outcome_digest=tensor_digest(observed_outcome),
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
        observed_outcome_digest=intervention.observed_outcome_digest,
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
