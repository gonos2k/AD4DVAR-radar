"""Provenance for realized observation interventions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
import io
import json
import math
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
import torch
from torch import Tensor
from torch import nn

from ._digest import json_digest, tensor_digest
from .action_contracts import (
    action_input_canonicalization_digest,
    canonicalize_action_frames,
)
from .sensitivity import (
    FirstOrderValidation,
    VariationalLearningImpact,
    _load_learning_policy_trust_store,
    first_order_validation_digest,
    validate_variational_learning_impact,
)
from .nowcast import (
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    ForecastResult,
    ForecastRunContract,
    RadarGridTimeContract,
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
    minimum_coverage_retention: float = 0.95
    maximum_withdrawn_fraction: float = 0.05
    maximum_background_fallback_increase: float = 0.05
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
        for name, value in (
            ("minimum_coverage_retention", self.minimum_coverage_retention),
            ("maximum_withdrawn_fraction", self.maximum_withdrawn_fraction),
            (
                "maximum_background_fallback_increase",
                self.maximum_background_fallback_increase,
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be inside [0, 1]")
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
                "minimum_coverage_retention": self.minimum_coverage_retention,
                "maximum_withdrawn_fraction": self.maximum_withdrawn_fraction,
                "maximum_background_fallback_increase": (
                    self.maximum_background_fallback_increase
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


def _canonical_datetime(value: str) -> datetime:
    """Parse one canonical UTC instant for ordering, never serialization."""

    return datetime.fromisoformat(
        _canonical_time(value).replace("Z", "+00:00")
    )


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
    case_id: str = ""
    radar_id: str = ""
    issue_time: str = ""
    input_bundle_before_digest: str = ""
    input_bundle_after_digest: str = ""
    resolved_issuance_validation_digest: str = ""
    contract: str = "realized-observation-intervention-v2"
    intervention_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract not in (
            "realized-observation-intervention-v1",
            "realized-observation-intervention-v2",
            "realized-observation-intervention-v3",
        ):
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
        if self.contract == "realized-observation-intervention-v3":
            if not self.case_id or self.case_id.strip() != self.case_id:
                raise ValueError("intervention case_id must be canonical")
            if not self.radar_id or self.radar_id.strip() != self.radar_id:
                raise ValueError("intervention radar_id must be canonical")
            object.__setattr__(self, "issue_time", _canonical_time(self.issue_time))
            applied = datetime.fromisoformat(self.applied_time.replace("Z", "+00:00"))
            issue = datetime.fromisoformat(self.issue_time.replace("Z", "+00:00"))
            if applied > issue:
                raise ValueError("intervention must precede its issue time")
            for name in (
                "input_bundle_before_digest",
                "input_bundle_after_digest",
                "resolved_issuance_validation_digest",
            ):
                _require_digest(name, getattr(self, name))
            if self.input_bundle_before_digest == self.input_bundle_after_digest:
                raise ValueError("intervention input bundles must differ")
        if self.actual_input_before_digest == self.actual_input_after_digest:
            raise ValueError("a realized intervention must change its input")
        for name in (
            "predicted_normalized_benefit",
            "resolved_normalized_benefit",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        payload: dict[str, object] = {
            "contract": self.contract,
            "intervention_id": self.intervention_id,
            "intervention_type": self.intervention_type,
            "action_digest": self.action_digest,
            "applied_time": self.applied_time,
            "actual_input_before_digest": self.actual_input_before_digest,
            "actual_input_after_digest": self.actual_input_after_digest,
            "learning_result_digest": self.learning_result_digest,
            "learning_approval_evidence_digest": (
                self.learning_approval_evidence_digest
            ),
            "counterfactual_perturbation_digest": (
                self.counterfactual_perturbation_digest
            ),
            "linearization_digest": self.linearization_digest,
        }
        if self.contract == "realized-observation-intervention-v1":
            payload["observed_outcome_digest"] = (
                self.outcome_resolution_contract_digest
            )
        else:
            payload.update(
                {
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
                }
            )
            if self.contract == "realized-observation-intervention-v3":
                payload.update(
                    {
                        "case_id": self.case_id,
                        "radar_id": self.radar_id,
                        "issue_time": self.issue_time,
                        "input_bundle_before_digest": (
                            self.input_bundle_before_digest
                        ),
                        "input_bundle_after_digest": self.input_bundle_after_digest,
                        "resolved_issuance_validation_digest": (
                            self.resolved_issuance_validation_digest
                        ),
                    }
                )
        object.__setattr__(self, "intervention_digest", json_digest(payload))

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
        nominal_forecast: ForecastResult,
        case_id: str,
        radar_id: str,
        resolved_issuance_validation: FirstOrderValidation,
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
        if resolved_issuance_validation.metric_domain_contract != (
            "resolved_issuance_domain"
        ):
            raise ValueError("intervention requires resolved-issuance validation")
        if resolved_issuance_validation.validation_digest != (
            first_order_validation_digest(resolved_issuance_validation)
        ):
            raise ValueError("resolved-issuance validation digest mismatch")
        nominal_forecast.validate_issuance()
        if (
            resolved_issuance_validation.source_fsoi_digest
            != fsoi.variational_fsoi_digest
            or resolved_issuance_validation.nominal_forecast_digest
            != nominal_forecast.forecast_run_digest
            or resolved_issuance_validation.nominal_input_bundle_digest
            != nominal_forecast.run.input_bundle_digest
        ):
            raise ValueError("resolved-issuance validation lineage mismatch")
        grid = nominal_forecast.run.grid_time_contract
        if grid is None:
            raise ValueError("realized intervention requires issue-time lineage")
        issue_time = grid.valid_times[-1]
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
        if (
            tensor_digest(actual_input_before[-1])
            != nominal_forecast.run.latest_frame_digest
        ):
            raise ValueError("realized inputs disagree with forecast runs")
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
        expected_run = ForecastRunContract.from_inputs(
            nominal_forecast.run.config,
            actual_input_after,
            linearization.observations.valid_mask,
            linearization.frozen.background_frames_dbz,
            linearization.frozen.background_age_minutes,
            grid_time_contract=linearization.frozen.grid_time_contract,
            analysis_config_json=nominal_forecast.run.analysis_config_json,
            analysis_config_digest=nominal_forecast.run.analysis_config_digest,
            analysis_input_digest=nominal_forecast.run.analysis_input_digest,
            operational_calibration_manifest_json=(
                nominal_forecast.run.operational_calibration_manifest_json
            ),
            operational_calibration_manifest_digest=(
                nominal_forecast.run.operational_calibration_manifest_digest
            ),
            operational_calibration_approval_digest=(
                nominal_forecast.run.operational_calibration_approval_digest
            ),
            operational_data_identity_json=(
                nominal_forecast.run.operational_data_identity_json
            ),
            operational_data_identity_digest=(
                nominal_forecast.run.operational_data_identity_digest
            ),
            neural_prior_digest=nominal_forecast.run.neural_prior_digest,
            prior_application_digest=(
                nominal_forecast.run.prior_application_digest
            ),
            prior_model_contract_digest=(
                nominal_forecast.run.prior_model_contract_digest
            ),
            prior_feature_schema_digest=(
                nominal_forecast.run.prior_feature_schema_digest
            ),
            prior_training_manifest_digest=(
                nominal_forecast.run.prior_training_manifest_digest
            ),
            prior_inference_evidence_digest=(
                nominal_forecast.run.prior_inference_evidence_digest
            ),
            prior_inference_algorithm_digest=(
                nominal_forecast.run.prior_inference_algorithm_digest
            ),
            prior_numerical_runtime_digest=(
                nominal_forecast.run.prior_numerical_runtime_digest
            ),
            prior_dependency=nominal_forecast.run.prior_dependency,
            prior_role=nominal_forecast.run.prior_role,
            input_plan_json=nominal_forecast.run.input_plan_json,
            input_plan_digest=nominal_forecast.run.input_plan_digest,
        )
        if resolved_issuance_validation.full_step_input_bundle_digest != (
            expected_run.input_bundle_digest
        ):
            raise ValueError("resolved-issuance perturbation lineage mismatch")
        metric_names = fsoi.fso.metric_names
        guardrails = {
            item.metric_name: item for item in execution_policy.metric_guardrails
        }
        if any(name not in guardrails for name in metric_names):
            raise ValueError("execution policy lacks a forecast metric")
        predicted = fsoi.observation.total.sum_by_time.sum(dim=-1)
        resolved = validation.full_step_resolved_metric_change
        available = validation.metric_available
        if execution_policy.approval_class == "automation":
            if not (
                resolved_issuance_validation.first_order_valid
                and resolved_issuance_validation.full_step_resolved_analysis_converged
                and resolved_issuance_validation.half_step_resolved_analysis_converged
                and resolved_issuance_validation.active_branch_valid
            ):
                raise ValueError(
                    "automation requires a valid resolved-issuance re-solve"
                )
            resolved = (
                resolved_issuance_validation.full_step_resolved_metric_change
            )
            available = resolved_issuance_validation.metric_available
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
            ) / guardrail.scale
            resolved_harm = torch.clamp(
                resolved[:, metric_index].masked_select(selected), min=0.0
            ) / guardrail.scale
            if float(torch.amax(predicted_harm)) > (
                guardrail.maximum_predicted_harm
            ):
                raise ValueError("predicted intervention harm exceeds policy")
            if float(torch.amax(resolved_harm)) > guardrail.maximum_resolved_harm:
                raise ValueError("resolved intervention harm exceeds policy")
        if execution_policy.approval_class == "automation":
            before = resolved_issuance_validation.coverage_before
            after = resolved_issuance_validation.coverage_after
            withdrawn = resolved_issuance_validation.withdrawn_fraction
            fallback_before = (
                resolved_issuance_validation.background_fallback_before
            )
            fallback_after = resolved_issuance_validation.background_fallback_after
            assert before is not None and after is not None
            assert withdrawn is not None
            assert fallback_before is not None and fallback_after is not None
            retention = after / before.clamp_min(torch.finfo(after.dtype).eps)
            if float(torch.amin(retention)) < (
                execution_policy.minimum_coverage_retention
            ):
                raise ValueError("resolved issuance loses excessive coverage")
            if float(torch.amax(withdrawn)) > (
                execution_policy.maximum_withdrawn_fraction
            ):
                raise ValueError("resolved issuance withdraws excessive coverage")
            fallback_increase = torch.clamp(
                fallback_after - fallback_before, min=0.0
            )
            if float(torch.amax(fallback_increase)) > (
                execution_policy.maximum_background_fallback_increase
            ):
                raise ValueError("resolved issuance increases background fallback")
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
            case_id=case_id,
            radar_id=radar_id,
            issue_time=issue_time,
            input_bundle_before_digest=nominal_forecast.run.input_bundle_digest,
            input_bundle_after_digest=(
                resolved_issuance_validation.full_step_input_bundle_digest
                or ""
            ),
            resolved_issuance_validation_digest=(
                resolved_issuance_validation.validation_digest
            ),
            contract="realized-observation-intervention-v3",
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
        case_id=intervention.case_id,
        radar_id=intervention.radar_id,
        issue_time=intervention.issue_time,
        input_bundle_before_digest=intervention.input_bundle_before_digest,
        input_bundle_after_digest=intervention.input_bundle_after_digest,
        resolved_issuance_validation_digest=(
            intervention.resolved_issuance_validation_digest
        ),
        contract=intervention.contract,
    )
    if reconstructed.intervention_digest != intervention.intervention_digest:
        raise ValueError("realized intervention digest mismatch")


@dataclass(frozen=True)
class RetrospectiveCounterfactualReplay:
    """A sealed historical replay that can never count as a realized action."""

    learning_result_digest: str
    perturbation_digest: str
    nominal_forecast_digest: str
    replayed_at: str
    contract: str = "retrospective-counterfactual-replay-v1"
    replay_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "learning_result_digest",
            "perturbation_digest",
            "nominal_forecast_digest",
        ):
            _require_digest(name, getattr(self, name))
        object.__setattr__(self, "replayed_at", _canonical_time(self.replayed_at))
        object.__setattr__(
            self,
            "replay_digest",
            _retrospective_replay_digest(self),
        )


def _retrospective_replay_digest(
    replay: RetrospectiveCounterfactualReplay,
) -> str:
    return json_digest(
        {
            "contract": replay.contract,
            "learning_result_digest": replay.learning_result_digest,
            "perturbation_digest": replay.perturbation_digest,
            "nominal_forecast_digest": replay.nominal_forecast_digest,
            "replayed_at": replay.replayed_at,
        }
    )


def validate_retrospective_counterfactual_replay(
    replay: RetrospectiveCounterfactualReplay,
) -> None:
    if replay.replay_digest != _retrospective_replay_digest(replay):
        raise ValueError("retrospective replay digest mismatch")


_DBZ_ACTION_CONTRACT = "radar-dbz-correction-action-v2"
_QC_ACTION_CONTRACT = "radar-qc-mask-action-v1"
_OVERRIDE_ACTION_CONTRACT = "radar-operator-override-action-v1"
_ACTION_CONTRACT_BY_TYPE: dict[ObservationInterventionType, str] = {
    "realized_sensor_correction": _DBZ_ACTION_CONTRACT,
    "realized_qc_intervention": _QC_ACTION_CONTRACT,
    "operator_override": _OVERRIDE_ACTION_CONTRACT,
}
_INTERVENTION_CONTEXT_SCHEMA_DIGEST = json_digest(
    {
        "contract": "intervention-generator-context-schema-v2",
        "channels": (
            "canonical_dbz",
            "observation_valid",
            "finite_input",
            "quality_weight",
            "observation_std_dbz",
            "background_dbz",
            "background_present",
            "applicability_mask",
        ),
    }
)


def _action_contract_digest(contract: str) -> str:
    return json_digest({"contract": contract})


def _same_tensor_with_nan(left: Tensor, right: Tensor) -> bool:
    """Exact equality with equal NaN locations treated as equal."""

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_nan = torch.isnan(left)
    if not torch.equal(left_nan, torch.isnan(right)):
        return False
    return torch.equal(left.masked_select(~left_nan), right.masked_select(~left_nan))


@dataclass(frozen=True, init=False)
class InterventionInputContext:
    """Canonical live inputs visible to and bound by an action generator."""

    _frames_dbz: Tensor
    _observation_masks: Tensor
    _quality_weight: Tensor
    _observation_std_dbz: Tensor
    _background_frames_dbz: Tensor | None
    _applicability_mask: Tensor
    radar_id: str
    input_bundle_digest: str
    input_plan_digest: str
    input_plan_resolution_digest: str
    analysis_input_identity_digest: str
    context_schema_digest: str
    applicability_region_digest: str
    applicability_mask_digest: str
    canonicalization_contract_digest: str
    min_dbz: float
    max_dbz: float
    missing_fill_dbz: float
    context_digest: str

    def __init__(self) -> None:
        raise TypeError("use InterventionInputContext.from_inputs")

    @classmethod
    def from_inputs(
        cls,
        *,
        frames_dbz: Tensor,
        observation_masks: Tensor,
        quality_weight: Tensor,
        observation_std_dbz: Tensor,
        background_frames_dbz: Tensor | None,
        radar_id: str,
        applicability_mask: Tensor,
        run: ForecastRunContract,
    ) -> InterventionInputContext:
        run.validate_integrity()
        shape = frames_dbz.shape
        if not radar_id or radar_id.strip() != radar_id:
            raise ValueError("intervention radar ID must be canonical")
        if frames_dbz.ndim != 3 or not frames_dbz.is_floating_point():
            raise ValueError("intervention frames must be floating [T,H,W]")
        if observation_masks.shape != shape or observation_masks.dtype != torch.bool:
            raise ValueError("intervention observation masks must match frames")
        for name, value in (
            ("quality_weight", quality_weight),
            ("observation_std_dbz", observation_std_dbz),
        ):
            if value.shape != shape or value.dtype != frames_dbz.dtype:
                raise ValueError(f"{name} must match intervention frames")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{name} must be finite")
        if bool(torch.any((quality_weight < 0.0) | (quality_weight > 1.0))):
            raise ValueError("quality_weight must be inside [0, 1]")
        if bool(torch.any(observation_masks & ~torch.isfinite(frames_dbz))):
            raise ValueError("non-finite radar pixels must be observation-invalid")
        if bool(torch.any(quality_weight.masked_select(~observation_masks) != 0.0)):
            raise ValueError("invalid observations must have zero quality weight")
        if bool(torch.any(observation_std_dbz <= 0.0)):
            raise ValueError("observation_std_dbz must be positive")
        canonical_quality_weight = torch.where(
            observation_masks,
            quality_weight,
            torch.zeros_like(quality_weight),
        )
        canonical_observation_std_dbz = torch.where(
            observation_masks,
            observation_std_dbz,
            torch.ones_like(observation_std_dbz),
        )
        if background_frames_dbz is not None and (
            background_frames_dbz.shape != shape
            or background_frames_dbz.dtype != frames_dbz.dtype
        ):
            raise ValueError("background frames must match intervention frames")
        if applicability_mask.shape != shape or applicability_mask.dtype != torch.bool:
            raise ValueError("intervention applicability mask must match frames")
        if tensor_digest(frames_dbz) != run.input_frames_digest:
            raise ValueError("intervention frames disagree with the input run")
        if run.observation_masks_digest is None or (
            tensor_digest(observation_masks) != run.observation_masks_digest
        ):
            raise ValueError("intervention masks disagree with the input run")
        if run.observation_quality_weight_digest is None or (
            tensor_digest(canonical_quality_weight)
            != run.observation_quality_weight_digest
        ):
            raise ValueError("intervention quality weights disagree with the input run")
        if run.observation_std_dbz_digest is None or (
            tensor_digest(canonical_observation_std_dbz)
            != run.observation_std_dbz_digest
        ):
            raise ValueError(
                "intervention observation errors disagree with the input run"
            )
        background_digest = (
            None
            if background_frames_dbz is None
            else tensor_digest(background_frames_dbz)
        )
        if background_digest != run.background_frames_digest:
            raise ValueError("intervention background disagrees with the input run")
        analysis_identity = run.analysis_input_identity
        if analysis_identity is None:
            raise ValueError(
                "prospective intervention requires a v2 resolved input plan"
            )
        assert run.input_plan_digest is not None
        assert run.input_plan_resolution_digest is not None
        canonicalization_contract_digest = action_input_canonicalization_digest(
            minimum_dbz=run.config.min_dbz,
            maximum_dbz=run.config.max_dbz,
            missing_fill_dbz=run.config.min_dbz,
        )
        applicability_mask_digest = tensor_digest(applicability_mask)
        applicability_region_digest = json_digest(
            {
                "contract": "intervention-applicability-region-v1",
                "radar_id": radar_id,
                "grid_time_contract_digest": run.grid_time_contract_digest,
                "mask_digest": applicability_mask_digest,
            }
        )
        context_digest = json_digest(
            {
                "contract": "intervention-input-context-v4",
                "input_bundle_digest": run.input_bundle_digest,
                "input_plan_digest": run.input_plan_digest,
                "input_plan_resolution_digest": run.input_plan_resolution_digest,
                "analysis_input_identity_digest": (
                    analysis_identity.identity_digest
                ),
                "frames_digest": tensor_digest(frames_dbz),
                "observation_masks_digest": tensor_digest(observation_masks),
                "quality_weight_digest": tensor_digest(
                    canonical_quality_weight
                ),
                "observation_std_dbz_digest": tensor_digest(
                    canonical_observation_std_dbz
                ),
                "background_frames_digest": background_digest,
                "radar_id": radar_id,
                "context_schema_digest": _INTERVENTION_CONTEXT_SCHEMA_DIGEST,
                "applicability_region_digest": applicability_region_digest,
                "applicability_mask_digest": applicability_mask_digest,
                "canonicalization_contract_digest": (
                    canonicalization_contract_digest
                ),
                "minimum_dbz": run.config.min_dbz,
                "maximum_dbz": run.config.max_dbz,
                "missing_fill_dbz": run.config.min_dbz,
            }
        )
        result = object.__new__(cls)
        for name, value in (
            ("_frames_dbz", frames_dbz.detach().clone()),
            ("_observation_masks", observation_masks.detach().clone()),
            (
                "_quality_weight",
                canonical_quality_weight.detach().clone(),
            ),
            (
                "_observation_std_dbz",
                canonical_observation_std_dbz.detach().clone(),
            ),
            (
                "_background_frames_dbz",
                None
                if background_frames_dbz is None
                else background_frames_dbz.detach().clone(),
            ),
            ("_applicability_mask", applicability_mask.detach().clone()),
            ("radar_id", radar_id),
            ("input_bundle_digest", run.input_bundle_digest),
            ("input_plan_digest", run.input_plan_digest),
            ("input_plan_resolution_digest", run.input_plan_resolution_digest),
            (
                "analysis_input_identity_digest",
                analysis_identity.identity_digest,
            ),
            ("context_schema_digest", _INTERVENTION_CONTEXT_SCHEMA_DIGEST),
            ("applicability_region_digest", applicability_region_digest),
            ("applicability_mask_digest", applicability_mask_digest),
            (
                "canonicalization_contract_digest",
                canonicalization_contract_digest,
            ),
            ("min_dbz", run.config.min_dbz),
            ("max_dbz", run.config.max_dbz),
            ("missing_fill_dbz", run.config.min_dbz),
            ("context_digest", context_digest),
        ):
            object.__setattr__(result, name, value)
        return result

    @property
    def frames_dbz(self) -> Tensor:
        return self._frames_dbz.clone()

    @property
    def observation_masks(self) -> Tensor:
        return self._observation_masks.clone()

    @property
    def quality_weight(self) -> Tensor:
        return self._quality_weight.clone()

    @property
    def observation_std_dbz(self) -> Tensor:
        return self._observation_std_dbz.clone()

    @property
    def background_frames_dbz(self) -> Tensor | None:
        if self._background_frames_dbz is None:
            return None
        return self._background_frames_dbz.clone()

    @property
    def applicability_mask(self) -> Tensor:
        return self._applicability_mask.clone()

    def generator_tensor(self) -> Tensor:
        finite_mask = torch.isfinite(self._frames_dbz)
        frames = canonicalize_action_frames(
            self._frames_dbz,
            self._observation_masks,
            minimum_dbz=self.min_dbz,
            maximum_dbz=self.max_dbz,
            missing_fill_dbz=self.missing_fill_dbz,
        )
        finite = finite_mask.to(frames)
        background = (
            torch.zeros_like(frames)
            if self._background_frames_dbz is None
            else canonicalize_action_frames(
                self._background_frames_dbz,
                torch.ones_like(self._background_frames_dbz, dtype=torch.bool),
                minimum_dbz=self.min_dbz,
                maximum_dbz=self.max_dbz,
                missing_fill_dbz=self.missing_fill_dbz,
            )
        )
        background_present = torch.full_like(
            frames,
            float(self._background_frames_dbz is not None),
        )
        return torch.stack(
            (
                frames,
                self._observation_masks.to(frames),
                finite,
                self._quality_weight,
                self._observation_std_dbz,
                background,
                background_present,
                self._applicability_mask.to(frames),
            )
        )


@dataclass(frozen=True)
class DbzCorrectionAction:
    delta_dbz: Tensor
    contract: str = _DBZ_ACTION_CONTRACT

    @property
    def payload_digest(self) -> str:
        return json_digest(
            {"contract": self.contract, "delta_dbz": tensor_digest(self.delta_dbz)}
        )


@dataclass(frozen=True)
class QcMaskAction:
    valid_mask_after: Tensor
    quality_weight_after: Tensor
    qc_reason: str
    contract: str = _QC_ACTION_CONTRACT

    @property
    def payload_digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "valid_mask_after": tensor_digest(self.valid_mask_after),
                "quality_weight_after": tensor_digest(self.quality_weight_after),
                "qc_reason": self.qc_reason,
            }
        )


@dataclass(frozen=True)
class OperatorOverrideAction:
    replacement_dbz: Tensor
    override_mask: Tensor
    override_reason: str
    contract: str = _OVERRIDE_ACTION_CONTRACT

    @property
    def payload_digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "replacement_dbz": tensor_digest(self.replacement_dbz),
                "override_mask": tensor_digest(self.override_mask),
                "override_reason": self.override_reason,
            }
        )


InterventionAction = DbzCorrectionAction | QcMaskAction | OperatorOverrideAction


@dataclass(frozen=True, init=False)
class InterventionActionGenerator:
    """Deterministic exported graph over a complete intervention context."""

    _artifact: bytes
    _shape: tuple[int, ...]
    _dtype: torch.dtype
    intervention_type: ObservationInterventionType
    action_reason: str | None
    generator_digest: str

    def __init__(self) -> None:
        raise TypeError("use InterventionActionGenerator.from_model")

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        example_context: InterventionInputContext,
        *,
        intervention_type: ObservationInterventionType,
        action_reason: str | None = None,
    ) -> InterventionActionGenerator:
        if not isinstance(model, nn.Module) or model.training:
            raise ValueError("action generator must be an eval-mode nn.Module")
        example_tensor = example_context.generator_tensor()
        canonical_context = torch.zeros_like(example_tensor)
        program = torch.export.export(model, (canonical_context,))
        stream = io.BytesIO()
        torch.export.save(program, stream)
        result = object.__new__(cls)
        artifact = stream.getvalue()
        shape = tuple(example_tensor.shape)
        dtype = example_tensor.dtype
        if intervention_type not in _ACTION_CONTRACT_BY_TYPE:
            raise ValueError("unsupported action-generator intervention type")
        if intervention_type != "realized_sensor_correction" and (
            not action_reason or action_reason.strip() != action_reason
        ):
            raise ValueError("QC and override generators require a canonical reason")
        object.__setattr__(result, "_artifact", artifact)
        object.__setattr__(result, "_shape", shape)
        object.__setattr__(result, "_dtype", dtype)
        object.__setattr__(result, "intervention_type", intervention_type)
        object.__setattr__(result, "action_reason", action_reason)
        object.__setattr__(result, "generator_digest", json_digest(
            {
                "contract": "exported-intervention-action-generator-v2",
                "artifact": artifact.hex(),
                "shape": list(shape),
                "dtype": str(dtype),
                "intervention_type": intervention_type,
                "action_reason": action_reason,
            }
        ))
        # Construction validates the exported graph even when the canonical
        # zero input is outside the policy's live applicability region.
        result._run(canonical_context, require_applicable=False)
        return result

    def generate(self, context: InterventionInputContext) -> InterventionAction:
        expected_digest = json_digest(
            {
                "contract": "exported-intervention-action-generator-v2",
                "artifact": self._artifact.hex(),
                "shape": list(self._shape),
                "dtype": str(self._dtype),
                "intervention_type": self.intervention_type,
                "action_reason": self.action_reason,
            }
        )
        if expected_digest != self.generator_digest:
            raise ValueError("action generator artifact digest mismatch")
        return self._run(context.generator_tensor(), require_applicable=True)

    @property
    def artifact_bytes(self) -> bytes:
        """Return an immutable copy for durable trusted replay."""

        return bytes(self._artifact)

    @classmethod
    def from_artifact(
        cls,
        *,
        artifact: bytes,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        intervention_type: ObservationInterventionType,
        action_reason: str | None,
        generator_digest: str,
    ) -> InterventionActionGenerator:
        """Rebuild and rehash an exported generator retained by the ledger."""

        result = object.__new__(cls)
        object.__setattr__(result, "_artifact", bytes(artifact))
        object.__setattr__(result, "_shape", tuple(shape))
        object.__setattr__(result, "_dtype", dtype)
        object.__setattr__(result, "intervention_type", intervention_type)
        object.__setattr__(result, "action_reason", action_reason)
        object.__setattr__(result, "generator_digest", generator_digest)
        expected = json_digest(
            {
                "contract": "exported-intervention-action-generator-v2",
                "artifact": bytes(artifact).hex(),
                "shape": list(shape),
                "dtype": str(dtype),
                "intervention_type": intervention_type,
                "action_reason": action_reason,
            }
        )
        if expected != generator_digest:
            raise ValueError("durable action generator artifact changed")
        return result

    def replay(self, context_tensor: Tensor) -> InterventionAction:
        """Run a previously retained canonical context during audit."""

        expected = InterventionActionGenerator.from_artifact(
            artifact=self._artifact,
            shape=self._shape,
            dtype=self._dtype,
            intervention_type=self.intervention_type,
            action_reason=self.action_reason,
            generator_digest=self.generator_digest,
        )
        return expected._run(context_tensor, require_applicable=True)

    def _run(
        self,
        context_tensor: Tensor,
        *,
        require_applicable: bool,
    ) -> InterventionAction:
        if tuple(context_tensor.shape) != self._shape or context_tensor.dtype != self._dtype:
            raise ValueError("action generator input contract changed")
        program = torch.export.load(io.BytesIO(self._artifact))
        module = program.module()
        first = module(context_tensor)
        second = module(context_tensor)
        if not isinstance(first, tuple) or not isinstance(second, tuple):
            raise ValueError("action generator output must be a tuple")
        if len(first) != len(second) or any(
            not isinstance(value, Tensor) for value in first + second
        ) or any(not torch.equal(a, b) for a, b in zip(first, second, strict=True)):
            raise ValueError("action generator output is invalid or nondeterministic")
        applicable = first[-1]
        if applicable.numel() != 1 or applicable.dtype is not torch.bool:
            raise ValueError("action applicability must be one boolean")
        if require_applicable and not bool(applicable.item()):
            raise ValueError("current input is outside the action policy applicability")
        frame_shape = context_tensor.shape[1:]
        if self.intervention_type == "realized_sensor_correction":
            if len(first) != 2:
                raise ValueError("dBZ generator must return delta and applicability")
            delta = first[0]
            if delta.shape != frame_shape or not delta.is_floating_point() or not bool(
                torch.all(torch.isfinite(delta))
            ):
                raise ValueError("dBZ action delta is invalid")
            return DbzCorrectionAction(delta.detach().clone())
        if len(first) != 3:
            raise ValueError("QC/override generator must return two fields and applicability")
        primary, secondary, _ = first
        if self.intervention_type == "realized_qc_intervention":
            if primary.shape != frame_shape or primary.dtype is not torch.bool:
                raise ValueError("QC action mask is invalid")
            if secondary.shape != frame_shape or not secondary.is_floating_point() or not bool(
                torch.all(torch.isfinite(secondary))
            ):
                raise ValueError("QC action quality weights are invalid")
            return QcMaskAction(
                primary.detach().clone(),
                secondary.detach().clone(),
                self.action_reason or "",
            )
        if primary.shape != frame_shape or not primary.is_floating_point() or not bool(
            torch.all(torch.isfinite(primary))
        ) or secondary.shape != frame_shape or secondary.dtype is not torch.bool:
            raise ValueError("operator override output is invalid")
        return OperatorOverrideAction(
            primary.detach().clone(),
            secondary.detach().clone(),
            self.action_reason or "",
        )


@dataclass(frozen=True)
class ActionSafetyDiagnostics:
    changed_pixel_count: int
    changed_fraction: float
    changed_area_km2: float
    global_diagonal_standardized_l2: float
    maximum_tile_diagonal_standardized_l2: float
    global_quality_precision_scale_l2: float
    maximum_tile_quality_precision_scale_l2: float
    minimum_input_floor_margin_dbz: float
    minimum_input_ceiling_margin_dbz: float
    changed_invalid_pixel_count: int
    diagnostics_digest: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            key: value
            for key, value in self.__dict__.items()
            if key != "diagnostics_digest"
        }
        payload = {"contract": "action-safety-diagnostics-v3", **values}
        object.__setattr__(
            self,
            "diagnostics_digest",
            json_digest(payload),
        )

    @property
    def json(self) -> str:
        values = {
            key: value
            for key, value in self.__dict__.items()
            if key != "diagnostics_digest"
        }
        return json.dumps(
            {"contract": "action-safety-diagnostics-v3", **values},
            sort_keys=True,
            separators=(",", ":"),
        )


def _maximum_tile_l2(value: Tensor, tile_size: int) -> float:
    maximum = 0.0
    for row in range(0, value.shape[-2], tile_size):
        for column in range(0, value.shape[-1], tile_size):
            tile = value[..., row : row + tile_size, column : column + tile_size]
            maximum = max(maximum, float(torch.linalg.vector_norm(tile)))
    return maximum


def _action_changed_mask(
    action: InterventionAction,
    context: InterventionInputContext,
) -> Tensor:
    if isinstance(action, DbzCorrectionAction):
        return action.delta_dbz != 0.0
    if isinstance(action, QcMaskAction):
        return (action.valid_mask_after != context._observation_masks) | (
            action.quality_weight_after != context._quality_weight
        )
    return action.override_mask


def _validate_action_safety(
    action: InterventionAction,
    context: InterventionInputContext,
    run: ForecastRunContract,
    policy: ReusableInterventionPolicyEvidence,
) -> ActionSafetyDiagnostics:
    if _run_uses_correlated_observation_error(run):
        raise ValueError(
            "prospective action requires diagonal observation error"
        )
    grid = run.grid_time_contract
    if grid is None:
        raise ValueError("prospective action safety requires a physical grid")
    return _compute_action_safety(
        action,
        context,
        policy,
        minimum_dbz=run.config.min_dbz,
        maximum_dbz=run.config.max_dbz,
        grid=grid,
    )


def _compute_action_safety(
    action: InterventionAction,
    context: InterventionInputContext,
    policy: ReusableInterventionPolicyEvidence,
    *,
    minimum_dbz: float,
    maximum_dbz: float,
    grid: RadarGridTimeContract | None = None,
    cell_area_m2: float | None = None,
    metric_domain_evidence_digest: str | None = None,
) -> ActionSafetyDiagnostics:
    if (grid is None) == (cell_area_m2 is None):
        raise ValueError("action safety requires exactly one grid-area authority")
    if grid is not None:
        resolved_cell_area_m2 = grid.cell_area_value_m2.nominal
    else:
        assert cell_area_m2 is not None
        resolved_cell_area_m2 = float(cell_area_m2)
        if (
            metric_domain_evidence_digest is not None
            and metric_domain_evidence_digest
            != CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
        ):
            raise ValueError("action safety metric-domain evidence is not current")
    expected_contract = _ACTION_CONTRACT_BY_TYPE[action_type(action)]
    if action.contract != expected_contract:
        raise ValueError("intervention type and action contract disagree")
    changed = _action_changed_mask(action, context)
    if bool(torch.any(changed & ~context._applicability_mask)):
        raise ValueError("generated action leaves its applicability region")
    count = int(torch.count_nonzero(changed))
    changed_fraction = count / changed.numel()
    union = torch.any(changed, dim=0)
    changed_invalid = int(
        torch.count_nonzero(changed & ~context._observation_masks)
    )
    union_count = int(torch.count_nonzero(union))
    area_km2 = union_count * resolved_cell_area_m2 / 1.0e6
    delta = torch.zeros_like(context._frames_dbz)
    quality_precision_delta = torch.zeros_like(context._quality_weight)
    if isinstance(action, DbzCorrectionAction):
        delta = action.delta_dbz.to(delta)
    elif isinstance(action, QcMaskAction):
        if bool(torch.any(action.valid_mask_after & ~context._observation_masks)):
            raise ValueError("QC action cannot create observations from invalid pixels")
        if bool(torch.any((action.quality_weight_after < 0.0) | (action.quality_weight_after > 1.0))):
            raise ValueError("QC action quality weights must be inside [0, 1]")
        if bool(torch.any(action.quality_weight_after.masked_select(~action.valid_mask_after) != 0.0)):
            raise ValueError("QC-rejected observations must have zero quality weight")
        if bool(torch.any(action.quality_weight_after > context._quality_weight)):
            raise ValueError("prospective QC may only reject or deweight observations")
        quality_precision_delta = (
            torch.sqrt(action.quality_weight_after.to(context._quality_weight))
            - torch.sqrt(context._quality_weight)
        ) / context._observation_std_dbz
    else:
        delta = torch.where(
            action.override_mask,
            action.replacement_dbz.to(delta) - context._frames_dbz,
            torch.zeros_like(delta),
        )
    if changed_invalid and not isinstance(action, QcMaskAction):
        raise ValueError("dBZ actions cannot change invalid observations")
    finite = torch.isfinite(context._frames_dbz)
    if bool(torch.any((delta != 0.0) & ~finite)):
        raise ValueError("dBZ actions must be zero at non-finite observations")
    changed_dbz = context._frames_dbz + delta
    changed_values = changed_dbz.masked_select((delta != 0.0) & finite)
    if changed_values.numel() and bool(
        torch.any(
            (changed_values < minimum_dbz)
            | (changed_values > maximum_dbz)
        )
    ):
        raise ValueError("action crosses the physical radar input clamp")
    whitened = (
        delta
        * torch.sqrt(context._quality_weight)
        / context._observation_std_dbz
    )
    global_norm = float(torch.linalg.vector_norm(whitened))
    tile_norm = _maximum_tile_l2(whitened, policy.perturbation_tile_size)
    quality_precision_norm = float(
        torch.linalg.vector_norm(quality_precision_delta)
    )
    quality_precision_tile_norm = _maximum_tile_l2(
        quality_precision_delta,
        policy.perturbation_tile_size,
    )
    if delta.numel():
        maximum_delta = float(torch.amax(torch.abs(delta)))
    else:
        maximum_delta = 0.0
    limits = (
        (maximum_delta, policy.maximum_absolute_delta_dbz, "per-pixel dBZ"),
        (count, policy.maximum_changed_pixel_count, "changed-pixel"),
        (changed_fraction, policy.maximum_changed_fraction, "changed-fraction"),
        (
            global_norm,
            policy.maximum_global_diagonal_standardized_l2,
            "global diagonal-standardized",
        ),
        (
            tile_norm,
            policy.maximum_tile_diagonal_standardized_l2,
            "tile diagonal-standardized",
        ),
        (
            quality_precision_norm,
            policy.maximum_global_quality_precision_scale_l2,
            "global QC precision-scale",
        ),
        (
            quality_precision_tile_norm,
            policy.maximum_tile_quality_precision_scale_l2,
            "tile QC precision-scale",
        ),
    )
    for actual, maximum, name in limits:
        if actual > maximum:
            raise ValueError(f"generated action exceeds its {name} safety limit")
    if grid is None and metric_domain_evidence_digest is None:
        if area_km2 > policy.maximum_changed_area_km2:
            raise ValueError(
                "generated action exceeds its changed-area safety limit"
            )
    elif grid is None:
        try:
            CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.validate_projected_area_maximum(
                area_km2,
                policy.maximum_changed_area_km2,
            )
        except ValueError as error:
            raise ValueError(
                "generated action exceeds or is uncertain against its "
                "changed-area safety limit"
            ) from error
    else:
        try:
            if grid.cell_count_area_maximum_status(
                union_count,
                policy.maximum_changed_area_km2,
            ) != "passes":
                raise ValueError("cell-derived changed area is not certified")
        except ValueError as error:
            raise ValueError(
                "generated action exceeds or is uncertain against its "
                "changed-area safety limit"
            ) from error
    finite_frames = changed_dbz.masked_select(finite)
    floor_margin = (
        float(torch.amin(finite_frames - minimum_dbz))
        if finite_frames.numel()
        else float("inf")
    )
    ceiling_margin = (
        float(torch.amin(maximum_dbz - finite_frames))
        if finite_frames.numel()
        else float("inf")
    )
    return ActionSafetyDiagnostics(
        changed_pixel_count=count,
        changed_fraction=changed_fraction,
        changed_area_km2=area_km2,
        global_diagonal_standardized_l2=global_norm,
        maximum_tile_diagonal_standardized_l2=tile_norm,
        global_quality_precision_scale_l2=quality_precision_norm,
        maximum_tile_quality_precision_scale_l2=(
            quality_precision_tile_norm
        ),
        minimum_input_floor_margin_dbz=floor_margin,
        minimum_input_ceiling_margin_dbz=ceiling_margin,
        changed_invalid_pixel_count=changed_invalid,
    )


def _run_uses_correlated_observation_error(run: ForecastRunContract) -> bool:
    if run.analysis_config_json is None:
        return False
    try:
        config = json.loads(run.analysis_config_json)
    except json.JSONDecodeError as error:
        raise ValueError("prospective action analysis config is invalid") from error
    if not isinstance(config, dict):
        raise ValueError("prospective action analysis config is invalid")
    bias_std = config.get("observation_common_bias_std_dbz", 0.0)
    return bool(
        isinstance(bias_std, (int, float))
        and not isinstance(bias_std, bool)
        and bias_std > 0.0
    )


def action_type(action: InterventionAction) -> ObservationInterventionType:
    if isinstance(action, DbzCorrectionAction):
        return "realized_sensor_correction"
    if isinstance(action, QcMaskAction):
        return "realized_qc_intervention"
    return "operator_override"


def validate_action_tensor_replay(
    action: InterventionAction,
    *,
    before_frames: Tensor,
    before_masks: Tensor,
    before_quality_weight: Tensor,
    after_frames: Tensor,
    after_masks: Tensor,
    after_quality_weight: Tensor,
) -> None:
    """Verify the typed action's exact tensor transition, including NaNs."""

    expected_frames = before_frames
    expected_masks = before_masks
    expected_quality = before_quality_weight
    if isinstance(action, DbzCorrectionAction):
        expected_frames = before_frames + action.delta_dbz.to(before_frames)
    elif isinstance(action, QcMaskAction):
        expected_masks = action.valid_mask_after
        expected_quality = action.quality_weight_after.to(before_quality_weight)
    else:
        expected_frames = torch.where(
            action.override_mask,
            action.replacement_dbz.to(before_frames),
            before_frames,
        )
    if not _same_tensor_with_nan(expected_frames, after_frames):
        raise ValueError("receipt after-input is not the approved action result")
    if not torch.equal(expected_masks, after_masks) or not torch.equal(
        expected_quality,
        after_quality_weight,
    ):
        raise ValueError("receipt after-QC state is not the approved action result")


@dataclass(frozen=True)
class ReusableInterventionPolicyEvidence:
    """Root-approved generator contract reusable across live input bundles."""

    policy_id: str
    action_generator_digest: str
    context_schema_digest: str
    applicability_region_digest: str
    execution_policy_digest: str
    allowed_intervention_types: tuple[ObservationInterventionType, ...]
    maximum_absolute_delta_dbz: float
    validation_evidence_digests: tuple[str, ...]
    maximum_changed_pixel_count: int = 4096
    maximum_changed_fraction: float = 0.05
    maximum_changed_area_km2: float = 256.0
    maximum_global_diagonal_standardized_l2: float = 8.0
    maximum_tile_diagonal_standardized_l2: float = 4.0
    maximum_global_quality_precision_scale_l2: float = 1.0
    maximum_tile_quality_precision_scale_l2: float = 1.0
    perturbation_tile_size: int = 16
    observation_error_contract: Literal["diagonal_only"] = "diagonal_only"
    execution_authority: Literal["operator_reviewed_only"] = (
        "operator_reviewed_only"
    )
    contract: str = "reusable-intervention-policy-evidence-v4"
    policy_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "reusable-intervention-policy-evidence-v4":
            raise ValueError("unsupported reusable intervention policy")
        if not self.policy_id or self.policy_id.strip() != self.policy_id:
            raise ValueError("reusable intervention policy ID must be canonical")
        for name in (
            "action_generator_digest",
            "context_schema_digest",
            "applicability_region_digest",
            "execution_policy_digest",
        ):
            _require_digest(name, getattr(self, name))
        if not self.validation_evidence_digests:
            raise ValueError("reusable intervention policy requires validation")
        for digest in self.validation_evidence_digests:
            _require_digest("validation evidence digest", digest)
        if len(self.allowed_intervention_types) != 1:
            raise ValueError("one reusable policy must have exactly one action type")
        if self.allowed_intervention_types[0] not in _ACTION_CONTRACT_BY_TYPE:
            raise ValueError("unsupported reusable intervention type")
        if self.observation_error_contract != "diagonal_only":
            raise ValueError("prospective action policy requires diagonal R")
        if self.execution_authority != "operator_reviewed_only":
            raise ValueError(
                "automatic action requires a current-case benefit contract"
            )
        if not math.isfinite(self.maximum_absolute_delta_dbz) or (
            self.maximum_absolute_delta_dbz <= 0.0
        ):
            raise ValueError("reusable intervention delta limit must be positive")
        if type(self.maximum_changed_pixel_count) is not int or (
            self.maximum_changed_pixel_count <= 0
        ):
            raise ValueError("changed-pixel limit must be positive")
        if type(self.perturbation_tile_size) is not int or self.perturbation_tile_size <= 0:
            raise ValueError("action tile size must be positive")
        for name in (
            "maximum_changed_fraction",
            "maximum_changed_area_km2",
            "maximum_global_diagonal_standardized_l2",
            "maximum_tile_diagonal_standardized_l2",
            "maximum_global_quality_precision_scale_l2",
            "maximum_tile_quality_precision_scale_l2",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_changed_fraction > 1.0:
            raise ValueError("changed-fraction limit must not exceed one")
        object.__setattr__(
            self,
            "policy_digest",
            json_digest(
                {
                    key: value
                    for key, value in self.__dict__.items()
                    if key != "policy_digest"
                }
            ),
        )

    def validate_integrity(self) -> None:
        expected = json_digest(
            {
                key: value
                for key, value in self.__dict__.items()
                if key != "policy_digest"
            }
        )
        if self.policy_digest != expected:
            raise ValueError("reusable intervention policy digest mismatch")


@dataclass(frozen=True, init=False)
class ProspectiveInterventionDecision:
    """A current-input action committed before its publication deadline."""

    decision_id: str
    case_id: str
    radar_id: str
    intervention_type: ObservationInterventionType
    action_policy_digest: str
    action_generator_digest: str
    action_context_digest: str
    intervention_input_context_digest: str
    actual_input_before_fixed_context_digest: str
    applicability_mask_digest: str
    action_payload_digest: str
    action_application_contract_digest: str
    action_safety_diagnostics_digest: str
    action_safety_diagnostics_json: str
    action_digest: str
    input_plan_digest: str
    input_plan_resolution_digest: str
    actual_input_before_frames_digest: str
    actual_input_before_bundle_digest: str
    actual_input_before_full_analysis_input_digest: str
    decision_basis_digest: str
    decision_policy_digest: str
    decision_trust_store_digest: str
    decided_at: str
    observation_valid_time: str
    input_available_time: str
    decision_deadline: str
    publication_time: str
    contract: str = "prospective-intervention-decision-v5"
    decision_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use ProspectiveInterventionDecision.from_policy")

    def __post_init__(self) -> None:
        if self.contract != "prospective-intervention-decision-v5":
            raise ValueError("unsupported prospective intervention decision")
        for name in ("decision_id", "case_id", "radar_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty and canonical")
        if self.intervention_type not in (
            "realized_sensor_correction",
            "realized_qc_intervention",
            "operator_override",
        ):
            raise ValueError("unsupported prospective intervention type")
        for name in (
            "action_policy_digest",
            "action_generator_digest",
            "action_context_digest",
            "intervention_input_context_digest",
            "actual_input_before_fixed_context_digest",
            "applicability_mask_digest",
            "action_payload_digest",
            "action_application_contract_digest",
            "action_safety_diagnostics_digest",
            "action_digest",
            "input_plan_digest",
            "input_plan_resolution_digest",
            "actual_input_before_frames_digest",
            "actual_input_before_bundle_digest",
            "actual_input_before_full_analysis_input_digest",
            "decision_basis_digest",
            "decision_policy_digest",
            "decision_trust_store_digest",
        ):
            _require_digest(name, getattr(self, name))
        decided = _canonical_time(self.decided_at)
        valid = _canonical_time(self.observation_valid_time)
        available = _canonical_time(self.input_available_time)
        deadline = _canonical_time(self.decision_deadline)
        publication = _canonical_time(self.publication_time)
        if not (
            _canonical_datetime(valid)
            <= _canonical_datetime(available)
            <= _canonical_datetime(decided)
            <= _canonical_datetime(deadline)
            < _canonical_datetime(publication)
        ):
            raise ValueError("prospective decision time order is invalid")
        expected_contract_digest = _action_contract_digest(
            _ACTION_CONTRACT_BY_TYPE[self.intervention_type]
        )
        if self.action_application_contract_digest != expected_contract_digest:
            raise ValueError("unsupported action application contract")
        try:
            diagnostics = json.loads(self.action_safety_diagnostics_json)
        except json.JSONDecodeError as error:
            raise ValueError("action safety diagnostics are invalid JSON") from error
        canonical_diagnostics = json.dumps(
            diagnostics,
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            not isinstance(diagnostics, dict)
            or diagnostics.get("contract") != "action-safety-diagnostics-v3"
            or canonical_diagnostics != self.action_safety_diagnostics_json
            or json_digest(diagnostics) != self.action_safety_diagnostics_digest
        ):
            raise ValueError("action safety diagnostics do not match their digest")
        expected_action = json_digest(
            {
                "contract": "generated-radar-action-v2",
                "action_policy_digest": self.action_policy_digest,
                "action_generator_digest": self.action_generator_digest,
                "action_context_digest": self.action_context_digest,
                "action_payload_digest": self.action_payload_digest,
                "action_application_contract_digest": (
                    self.action_application_contract_digest
                ),
                "action_safety_diagnostics_digest": (
                    self.action_safety_diagnostics_digest
                ),
            }
        )
        if self.action_digest != expected_action:
            raise ValueError("prospective action was not generated by its policy")
        object.__setattr__(self, "decided_at", decided)
        object.__setattr__(self, "observation_valid_time", valid)
        object.__setattr__(self, "input_available_time", available)
        object.__setattr__(self, "decision_deadline", deadline)
        object.__setattr__(self, "publication_time", publication)
        object.__setattr__(
            self,
            "decision_digest",
            json_digest(
                {
                    key: value
                    for key, value in self.__dict__.items()
                    if key != "decision_digest"
                }
            ),
        )

    @classmethod
    def from_policy(
        cls,
        policy: ReusableInterventionPolicyEvidence,
        *,
        action_generator: InterventionActionGenerator,
        decision_id: str,
        case_id: str,
        radar_id: str,
        intervention_type: ObservationInterventionType,
        actual_input_context: InterventionInputContext,
        actual_input_before_run: ForecastRunContract,
        input_plan_digest: str,
        decision_basis_digest: str,
        decision_policy_digest: str,
        decision_trust_store_digest: str,
        decided_at: str,
        observation_valid_time: str,
        input_available_time: str,
        decision_deadline: str,
        publication_time: str,
    ) -> ProspectiveInterventionDecision:
        """Generate and bind one bounded action to the current input context."""

        policy.validate_integrity()
        actual_input_before_run.validate_integrity()
        if actual_input_context.input_bundle_digest != actual_input_before_run.input_bundle_digest:
            raise ValueError("decision context disagrees with the current input run")
        if actual_input_context.radar_id != radar_id:
            raise ValueError("decision radar disagrees with the action context")
        if actual_input_before_run.input_plan_digest != input_plan_digest:
            raise ValueError("decision input plan disagrees with the current run")
        if actual_input_before_run.input_plan_resolution_digest != (
            actual_input_context.input_plan_resolution_digest
        ):
            raise ValueError("decision input-plan resolution disagrees with the run")
        if (
            actual_input_before_run.fixed_input_context_digest is None
            or actual_input_before_run.full_analysis_input_digest is None
        ):
            raise ValueError("prospective decision requires complete input identity")
        if actual_input_before_run.input_plan_json is None:
            raise ValueError("prospective decision requires input-plan JSON")
        try:
            plan = json.loads(actual_input_before_run.input_plan_json)
        except json.JSONDecodeError as error:
            raise ValueError("prospective input plan is invalid JSON") from error
        if not isinstance(plan, dict) or plan.get("contract") != "neural-prior-input-plan-v2":
            raise ValueError("prospective decision requires a causal v2 input plan")
        plan_times = {
            name: _canonical_time(str(plan.get(name, "")))
            for name in (
                "observation_valid_time",
                "input_available_time",
                "decision_deadline",
                "publication_time",
            )
        }
        supplied_times = {
            "observation_valid_time": _canonical_time(observation_valid_time),
            "input_available_time": _canonical_time(input_available_time),
            "decision_deadline": _canonical_time(decision_deadline),
            "publication_time": _canonical_time(publication_time),
        }
        if supplied_times != plan_times:
            raise ValueError("decision times disagree with the resolved input plan")
        if intervention_type not in policy.allowed_intervention_types:
            raise ValueError("intervention type is outside the reusable policy")
        if policy.context_schema_digest != actual_input_context.context_schema_digest:
            raise ValueError("action context schema is outside the reusable policy")
        if (
            policy.applicability_region_digest
            != actual_input_context.applicability_region_digest
        ):
            raise ValueError("action context is outside the approved region")
        if action_generator.generator_digest != policy.action_generator_digest:
            raise ValueError("action generator is outside the reusable policy")
        if action_generator.intervention_type != intervention_type:
            raise ValueError("action generator type disagrees with the intervention")
        action = action_generator.generate(actual_input_context)
        diagnostics = _validate_action_safety(
            action,
            actual_input_context,
            actual_input_before_run,
            policy,
        )
        context_digest = json_digest(
            {
                "contract": "intervention-action-context-v2",
                "case_id": case_id,
                "radar_id": radar_id,
                "input_plan_digest": input_plan_digest,
                "input_plan_resolution_digest": (
                    actual_input_context.input_plan_resolution_digest
                ),
                "input_bundle_digest": actual_input_before_run.input_bundle_digest,
                "full_analysis_input_digest": (
                    actual_input_before_run.full_analysis_input_digest
                ),
                "intervention_input_context_digest": (
                    actual_input_context.context_digest
                ),
                "context_schema_digest": policy.context_schema_digest,
                "applicability_region_digest": policy.applicability_region_digest,
            }
        )
        payload_digest = action.payload_digest
        application_digest = _action_contract_digest(action.contract)
        action_digest = json_digest(
            {
                "contract": "generated-radar-action-v2",
                "action_policy_digest": policy.policy_digest,
                "action_generator_digest": policy.action_generator_digest,
                "action_context_digest": context_digest,
                "action_payload_digest": payload_digest,
                "action_application_contract_digest": (
                    application_digest
                ),
                "action_safety_diagnostics_digest": diagnostics.diagnostics_digest,
            }
        )
        return _new_prospective_decision(
            decision_id=decision_id,
            case_id=case_id,
            radar_id=radar_id,
            intervention_type=intervention_type,
            action_policy_digest=policy.policy_digest,
            action_generator_digest=policy.action_generator_digest,
            action_context_digest=context_digest,
            intervention_input_context_digest=actual_input_context.context_digest,
            actual_input_before_fixed_context_digest=(
                actual_input_before_run.fixed_input_context_digest
            ),
            applicability_mask_digest=actual_input_context.applicability_mask_digest,
            action_payload_digest=payload_digest,
            action_application_contract_digest=application_digest,
            action_safety_diagnostics_digest=diagnostics.diagnostics_digest,
            action_safety_diagnostics_json=diagnostics.json,
            action_digest=action_digest,
            input_plan_digest=input_plan_digest,
            input_plan_resolution_digest=(
                actual_input_context.input_plan_resolution_digest
            ),
            actual_input_before_frames_digest=tensor_digest(
                actual_input_context._frames_dbz
            ),
            actual_input_before_bundle_digest=(
                actual_input_before_run.input_bundle_digest
            ),
            actual_input_before_full_analysis_input_digest=(
                actual_input_before_run.full_analysis_input_digest
            ),
            decision_basis_digest=decision_basis_digest,
            decision_policy_digest=decision_policy_digest,
            decision_trust_store_digest=decision_trust_store_digest,
            decided_at=decided_at,
            observation_valid_time=plan_times["observation_valid_time"],
            input_available_time=plan_times["input_available_time"],
            decision_deadline=plan_times["decision_deadline"],
            publication_time=plan_times["publication_time"],
        )


@dataclass(frozen=True, init=False)
class OperatorActionApproval:
    """Operator-signed approval of one exact prospective decision."""

    decision_digest: str
    action_digest: str
    full_analysis_input_digest: str
    safety_diagnostics_digest: str
    operator_key_id: str
    operator_role: str
    operator_trust_store_digest: str
    reviewed_at: str
    expires_at: str
    operator_comment_digest: str
    operator_signature: str
    contract: str = "operator-action-approval-v1"
    approval_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use OperatorActionApproval.from_decision")

    def __post_init__(self) -> None:
        if self.contract != "operator-action-approval-v1":
            raise ValueError("unsupported operator action approval")
        for name in (
            "decision_digest",
            "action_digest",
            "full_analysis_input_digest",
            "safety_diagnostics_digest",
            "operator_trust_store_digest",
            "operator_comment_digest",
        ):
            _require_digest(name, getattr(self, name))
        for name in ("operator_key_id", "operator_role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty and canonical")
        reviewed = _canonical_time(self.reviewed_at)
        expires = _canonical_time(self.expires_at)
        if _canonical_datetime(reviewed) >= _canonical_datetime(expires):
            raise ValueError("operator approval must expire after review")
        if (
            not isinstance(self.operator_signature, str)
            or len(self.operator_signature) != 128
            or any(character not in "0123456789abcdef" for character in self.operator_signature)
        ):
            raise ValueError("operator signature must be lowercase Ed25519")
        object.__setattr__(self, "reviewed_at", reviewed)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(
            self,
            "approval_digest",
            json_digest(
                {
                    key: value
                    for key, value in self.__dict__.items()
                    if key != "approval_digest"
                }
            ),
        )

    @classmethod
    def from_decision(
        cls,
        decision: ProspectiveInterventionDecision,
        *,
        operator_key_id: str,
        operator_role: str,
        operator_trust_store_digest: str,
        operator_private_key: Ed25519PrivateKey,
        reviewed_at: str,
        expires_at: str,
        operator_comment_digest: str,
    ) -> OperatorActionApproval:
        reviewed = _canonical_time(reviewed_at)
        expires = _canonical_time(expires_at)
        decided = _canonical_time(decision.decided_at)
        deadline = _canonical_time(decision.decision_deadline)
        if not (
            _canonical_datetime(decided)
            <= _canonical_datetime(reviewed)
            < _canonical_datetime(expires)
            <= _canonical_datetime(deadline)
        ):
            raise ValueError("operator approval is outside the decision window")
        values = {
            "decision_digest": decision.decision_digest,
            "action_digest": decision.action_digest,
            "full_analysis_input_digest": (
                decision.actual_input_before_full_analysis_input_digest
            ),
            "safety_diagnostics_digest": (
                decision.action_safety_diagnostics_digest
            ),
            "operator_key_id": operator_key_id,
            "operator_role": operator_role,
            "operator_trust_store_digest": operator_trust_store_digest,
            "reviewed_at": reviewed,
            "expires_at": expires,
            "operator_comment_digest": operator_comment_digest,
            "operator_signature": "",
            "contract": "operator-action-approval-v1",
        }
        signature = operator_private_key.sign(
            json_digest(values).encode("ascii")
        ).hex()
        return _new_operator_action_approval(
            **{**values, "operator_signature": signature}
        )


def _new_operator_action_approval(**values: object) -> OperatorActionApproval:
    expected = {
        item.name
        for item in fields(OperatorActionApproval)
        if item.name != "approval_digest"
    }
    if set(values) != expected:
        raise ValueError("operator action approval fields are incomplete")
    result = object.__new__(OperatorActionApproval)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    OperatorActionApproval.__post_init__(result)
    return result


def verify_operator_action_approval_signature(
    approval: OperatorActionApproval,
    operator_public_key: Ed25519PublicKey,
) -> None:
    """Verify one approval without access to an operator private key."""

    values = {
        key: value
        for key, value in approval.__dict__.items()
        if key != "approval_digest"
    }
    values["operator_signature"] = ""
    try:
        operator_public_key.verify(
            bytes.fromhex(approval.operator_signature),
            json_digest(values).encode("ascii"),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("operator action approval signature mismatch") from error


@dataclass(frozen=True, init=False)
class RealizedInterventionReceipt:
    """Executor-signed proof of one exact before/action/after transition."""

    decision_digest: str
    decision_id: str
    case_id: str
    radar_id: str
    intervention_type: ObservationInterventionType
    action_digest: str
    input_plan_digest: str
    input_plan_resolution_digest: str
    actual_input_before_frames_digest: str
    actual_input_after_frames_digest: str
    actual_input_before_masks_digest: str
    actual_input_after_masks_digest: str
    actual_quality_weight_before_digest: str
    actual_quality_weight_after_digest: str
    actual_input_before_bundle_digest: str
    actual_input_bundle_digest: str
    fixed_input_context_before_digest: str
    fixed_input_context_after_digest: str
    full_analysis_input_before_digest: str
    full_analysis_input_after_digest: str
    action_payload_digest: str
    action_application_contract_digest: str
    action_safety_diagnostics_digest: str
    action_artifact_digest: str
    executor_key_id: str
    executor_trust_store_digest: str
    executor_signature: str
    applied_time: str
    receipt_time: str
    observation_valid_time: str
    input_available_time: str
    publication_time: str
    executor_sequence_number: int
    contract: str = "realized-intervention-receipt-v5"
    receipt_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use RealizedInterventionReceipt.from_decision")

    def __post_init__(self) -> None:
        if self.contract != "realized-intervention-receipt-v5":
            raise ValueError("unsupported realized intervention receipt")
        for name in ("decision_id", "case_id", "radar_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty and canonical")
        for name in (
            "decision_digest",
            "action_digest",
            "input_plan_digest",
            "input_plan_resolution_digest",
            "actual_input_before_frames_digest",
            "actual_input_after_frames_digest",
            "actual_input_before_masks_digest",
            "actual_input_after_masks_digest",
            "actual_quality_weight_before_digest",
            "actual_quality_weight_after_digest",
            "actual_input_before_bundle_digest",
            "actual_input_bundle_digest",
            "fixed_input_context_before_digest",
            "fixed_input_context_after_digest",
            "full_analysis_input_before_digest",
            "full_analysis_input_after_digest",
            "action_payload_digest",
            "action_application_contract_digest",
            "action_safety_diagnostics_digest",
            "action_artifact_digest",
            "executor_trust_store_digest",
        ):
            _require_digest(name, getattr(self, name))
        if not self.executor_key_id or self.executor_key_id.strip() != self.executor_key_id:
            raise ValueError("executor key ID must be canonical")
        if len(self.executor_signature) != 128 or any(
            value not in "0123456789abcdef" for value in self.executor_signature
        ):
            raise ValueError("executor signature must be lowercase Ed25519")
        if (
            self.actual_input_before_frames_digest
            == self.actual_input_after_frames_digest
            and self.actual_input_before_masks_digest
            == self.actual_input_after_masks_digest
            and self.actual_quality_weight_before_digest
            == self.actual_quality_weight_after_digest
        ):
            raise ValueError("realized intervention receipt must change input")
        if self.full_analysis_input_before_digest == (
            self.full_analysis_input_after_digest
        ):
            raise ValueError("realized intervention receipt must change its full input")
        if type(self.executor_sequence_number) is not int or (
            self.executor_sequence_number < 0
        ):
            raise ValueError("executor sequence number must be nonnegative")
        applied = _canonical_time(self.applied_time)
        received = _canonical_time(self.receipt_time)
        valid = _canonical_time(self.observation_valid_time)
        available = _canonical_time(self.input_available_time)
        publication = _canonical_time(self.publication_time)
        if not (
            _canonical_datetime(valid)
            <= _canonical_datetime(available)
            <= _canonical_datetime(applied)
            <= _canonical_datetime(received)
            < _canonical_datetime(publication)
        ):
            raise ValueError("receipt time order is invalid")
        object.__setattr__(self, "applied_time", applied)
        object.__setattr__(self, "receipt_time", received)
        object.__setattr__(self, "observation_valid_time", valid)
        object.__setattr__(self, "input_available_time", available)
        object.__setattr__(self, "publication_time", publication)
        object.__setattr__(
            self,
            "receipt_digest",
            json_digest(
                {
                    key: value
                    for key, value in self.__dict__.items()
                    if key != "receipt_digest"
                }
            ),
        )

    @classmethod
    def from_decision(
        cls,
        decision: ProspectiveInterventionDecision,
        *,
        actual_input_before_context: InterventionInputContext,
        actual_input_before_run: ForecastRunContract,
        actual_input_after_context: InterventionInputContext,
        actual_input_after_run: ForecastRunContract,
        action_policy: ReusableInterventionPolicyEvidence,
        action_generator: InterventionActionGenerator,
        executor_key_id: str,
        executor_trust_store_digest: str,
        executor_private_key: Ed25519PrivateKey,
        executor_sequence_number: int,
        applied_time: str,
        receipt_time: str,
    ) -> RealizedInterventionReceipt:
        transition = validate_intervention_action_transition(
            decision,
            action_policy=action_policy,
            action_generator=action_generator,
            actual_input_before_context=actual_input_before_context,
            actual_input_before_run=actual_input_before_run,
            actual_input_after_context=actual_input_after_context,
            actual_input_after_run=actual_input_after_run,
        )
        applied = _canonical_time(applied_time)
        received = _canonical_time(receipt_time)
        before_fixed_context = actual_input_before_run.fixed_input_context_digest
        after_fixed_context = actual_input_after_run.fixed_input_context_digest
        before_full_input = actual_input_before_run.full_analysis_input_digest
        after_full_input = actual_input_after_run.full_analysis_input_digest
        if (
            before_fixed_context is None
            or after_fixed_context is None
            or before_full_input is None
            or after_full_input is None
        ):
            raise ValueError("prospective receipt requires complete input identity")
        values: dict[str, str | int] = dict(
            decision_digest=decision.decision_digest,
            decision_id=decision.decision_id,
            case_id=decision.case_id,
            radar_id=decision.radar_id,
            intervention_type=decision.intervention_type,
            action_digest=decision.action_digest,
            input_plan_digest=decision.input_plan_digest,
            input_plan_resolution_digest=decision.input_plan_resolution_digest,
            actual_input_before_frames_digest=transition.before_frames_digest,
            actual_input_after_frames_digest=transition.after_frames_digest,
            actual_input_before_masks_digest=transition.before_masks_digest,
            actual_input_after_masks_digest=transition.after_masks_digest,
            actual_quality_weight_before_digest=(
                transition.before_quality_weight_digest
            ),
            actual_quality_weight_after_digest=transition.after_quality_weight_digest,
            actual_input_before_bundle_digest=(
                actual_input_before_run.input_bundle_digest
            ),
            actual_input_bundle_digest=actual_input_after_run.input_bundle_digest,
            fixed_input_context_before_digest=before_fixed_context,
            fixed_input_context_after_digest=after_fixed_context,
            full_analysis_input_before_digest=before_full_input,
            full_analysis_input_after_digest=after_full_input,
            action_payload_digest=decision.action_payload_digest,
            action_application_contract_digest=(
                decision.action_application_contract_digest
            ),
            action_safety_diagnostics_digest=(
                decision.action_safety_diagnostics_digest
            ),
            action_artifact_digest=transition.action_artifact_digest,
            executor_key_id=executor_key_id,
            executor_trust_store_digest=executor_trust_store_digest,
            executor_signature="",
            applied_time=applied,
            receipt_time=received,
            observation_valid_time=decision.observation_valid_time,
            input_available_time=decision.input_available_time,
            publication_time=decision.publication_time,
            executor_sequence_number=executor_sequence_number,
            contract="realized-intervention-receipt-v5",
        )
        signature = executor_private_key.sign(
            json_digest(values).encode("ascii")
        ).hex()
        return _new_realized_intervention_receipt(
            decision_digest=decision.decision_digest,
            decision_id=decision.decision_id,
            case_id=decision.case_id,
            radar_id=decision.radar_id,
            intervention_type=decision.intervention_type,
            action_digest=decision.action_digest,
            input_plan_digest=decision.input_plan_digest,
            input_plan_resolution_digest=decision.input_plan_resolution_digest,
            actual_input_before_frames_digest=transition.before_frames_digest,
            actual_input_after_frames_digest=transition.after_frames_digest,
            actual_input_before_masks_digest=transition.before_masks_digest,
            actual_input_after_masks_digest=transition.after_masks_digest,
            actual_quality_weight_before_digest=(
                transition.before_quality_weight_digest
            ),
            actual_quality_weight_after_digest=transition.after_quality_weight_digest,
            actual_input_before_bundle_digest=(
                actual_input_before_run.input_bundle_digest
            ),
            actual_input_bundle_digest=actual_input_after_run.input_bundle_digest,
            fixed_input_context_before_digest=before_fixed_context,
            fixed_input_context_after_digest=after_fixed_context,
            full_analysis_input_before_digest=before_full_input,
            full_analysis_input_after_digest=after_full_input,
            action_payload_digest=decision.action_payload_digest,
            action_application_contract_digest=(
                decision.action_application_contract_digest
            ),
            action_safety_diagnostics_digest=(
                decision.action_safety_diagnostics_digest
            ),
            action_artifact_digest=transition.action_artifact_digest,
            executor_key_id=executor_key_id,
            executor_trust_store_digest=executor_trust_store_digest,
            executor_signature=signature,
            applied_time=applied,
            receipt_time=received,
            observation_valid_time=decision.observation_valid_time,
            input_available_time=decision.input_available_time,
            publication_time=decision.publication_time,
            executor_sequence_number=executor_sequence_number,
        )


def _new_prospective_decision(
    **values: object,
) -> ProspectiveInterventionDecision:
    values.setdefault("contract", "prospective-intervention-decision-v5")
    expected = {
        item.name
        for item in fields(ProspectiveInterventionDecision)
        if item.name != "decision_digest"
    }
    if set(values) != expected:
        raise ValueError("prospective decision fields are incomplete")
    result = object.__new__(ProspectiveInterventionDecision)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    ProspectiveInterventionDecision.__post_init__(result)
    return result


@dataclass(frozen=True)
class InterventionActionTransition:
    before_frames_digest: str
    after_frames_digest: str
    before_masks_digest: str
    after_masks_digest: str
    before_quality_weight_digest: str
    after_quality_weight_digest: str
    action_safety_diagnostics_digest: str
    action_artifact_digest: str


def _intervention_action_artifact_digest(
    *,
    generator_digest: str,
    before_context_digest: str,
    after_context_digest: str,
    action_payload_digest: str,
    before_frames_digest: str,
    after_frames_digest: str,
    before_masks_digest: str,
    after_masks_digest: str,
    before_quality_weight_digest: str,
    after_quality_weight_digest: str,
    grid_time_contract: RadarGridTimeContract | None,
) -> str:
    """Bind a current action transition to its exact physical grid evidence."""

    payload: dict[str, object] = {
        "contract": "intervention-action-artifact-v1",
        "generator_digest": generator_digest,
        "before_context_digest": before_context_digest,
        "after_context_digest": after_context_digest,
        "action_payload_digest": action_payload_digest,
        "before_frames_digest": before_frames_digest,
        "after_frames_digest": after_frames_digest,
        "before_masks_digest": before_masks_digest,
        "after_masks_digest": after_masks_digest,
        "before_quality_weight_digest": before_quality_weight_digest,
        "after_quality_weight_digest": after_quality_weight_digest,
    }
    if (
        grid_time_contract is not None
        and grid_time_contract.spatial_grid_contract
        == "radar-spatial-grid-identity-v6"
    ):
        grid_time_contract.validate_current_metric_domain_evidence()
        payload.update(
            {
                "contract": "intervention-action-artifact-v3",
                "grid_time_contract_digest": grid_time_contract.digest,
                "metric_domain_evidence_digest": (
                    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
                ),
            }
        )
    return json_digest(payload)


def validate_intervention_action_transition(
    decision: ProspectiveInterventionDecision,
    *,
    action_policy: ReusableInterventionPolicyEvidence,
    action_generator: InterventionActionGenerator,
    actual_input_before_context: InterventionInputContext,
    actual_input_before_run: ForecastRunContract,
    actual_input_after_context: InterventionInputContext,
    actual_input_after_run: ForecastRunContract,
) -> InterventionActionTransition:
    """Recompute the exact before/action/after transition without a private key."""

    action_policy.validate_integrity()
    if action_policy.policy_digest != decision.action_policy_digest:
        raise ValueError("receipt action policy disagrees with its decision")
    expected_decision = json_digest(
        {
            key: value
            for key, value in decision.__dict__.items()
            if key != "decision_digest"
        }
    )
    if decision.decision_digest != expected_decision:
        raise ValueError("prospective decision digest mismatch")
    actual_input_before_run.validate_integrity()
    actual_input_after_run.validate_integrity()
    if (
        actual_input_before_context.input_bundle_digest
        != actual_input_before_run.input_bundle_digest
        or actual_input_after_context.input_bundle_digest
        != actual_input_after_run.input_bundle_digest
    ):
        raise ValueError("receipt context disagrees with its input run")
    if (
        actual_input_before_context.context_digest
        != decision.intervention_input_context_digest
        or actual_input_before_run.fixed_input_context_digest
        != decision.actual_input_before_fixed_context_digest
        or actual_input_before_context.applicability_mask_digest
        != decision.applicability_mask_digest
    ):
        raise ValueError("receipt before-context disagrees with its decision")
    if (
        actual_input_before_run.full_analysis_input_digest
        != decision.actual_input_before_full_analysis_input_digest
        or actual_input_before_run.full_analysis_input_digest is None
        or actual_input_after_run.full_analysis_input_digest is None
    ):
        raise ValueError("receipt full input identity disagrees with its decision")
    if (
        actual_input_before_context.radar_id
        != actual_input_after_context.radar_id
        or actual_input_before_context.context_schema_digest
        != actual_input_after_context.context_schema_digest
        or actual_input_before_context.applicability_region_digest
        != actual_input_after_context.applicability_region_digest
        or not torch.equal(
            actual_input_before_context._applicability_mask,
            actual_input_after_context._applicability_mask,
        )
    ):
        raise ValueError("receipt changed its radar applicability context")
    before_frames = actual_input_before_context._frames_dbz
    after_frames = actual_input_after_context._frames_dbz
    before_masks = actual_input_before_context._observation_masks
    after_masks = actual_input_after_context._observation_masks
    before_quality = actual_input_before_context._quality_weight
    after_quality = actual_input_after_context._quality_weight
    before_digest = tensor_digest(before_frames)
    after_digest = tensor_digest(after_frames)
    if (
        before_digest != decision.actual_input_before_frames_digest
        or before_digest != actual_input_before_run.input_frames_digest
        or actual_input_before_run.input_bundle_digest
        != decision.actual_input_before_bundle_digest
    ):
        raise ValueError("receipt before-input disagrees with its decision")
    if after_digest != actual_input_after_run.input_frames_digest:
        raise ValueError("receipt frames disagree with the actual input run")
    if (
        actual_input_before_run.input_plan_digest != decision.input_plan_digest
        or actual_input_after_run.input_plan_digest != decision.input_plan_digest
        or actual_input_before_run.input_plan_resolution_digest
        != decision.input_plan_resolution_digest
    ):
        raise ValueError("receipt runs disagree with the decision input plan")
    unchanged_run_fields = (
        "analysis_config_digest",
        "source_available_mask_digest",
        "background_frames_digest",
        "background_age_minutes",
        "grid_time_contract_digest",
        "operational_calibration_manifest_digest",
        "operational_calibration_approval_digest",
        "operational_data_identity_digest",
    )
    if (
        actual_input_before_run.config.digest
        != actual_input_after_run.config.digest
        or any(
            getattr(actual_input_before_run, name)
            != getattr(actual_input_after_run, name)
            for name in unchanged_run_fields
        )
    ):
        raise ValueError("receipt changed non-radar input state")
    if action_generator.generator_digest != decision.action_generator_digest:
        raise ValueError("receipt action generator disagrees with its decision")
    action = action_generator.generate(actual_input_before_context)
    if action.payload_digest != decision.action_payload_digest:
        raise ValueError("receipt action payload disagrees with its decision")
    if action_type(action) != decision.intervention_type:
        raise ValueError("receipt action type disagrees with its decision")
    diagnostics = _validate_action_safety(
        action,
        actual_input_before_context,
        actual_input_before_run,
        action_policy,
    )
    if diagnostics.diagnostics_digest != decision.action_safety_diagnostics_digest:
        raise ValueError("receipt action safety disagrees with its decision")
    validate_action_tensor_replay(
        action,
        before_frames=before_frames,
        before_masks=before_masks,
        before_quality_weight=before_quality,
        after_frames=after_frames,
        after_masks=after_masks,
        after_quality_weight=after_quality,
    )
    if isinstance(action, QcMaskAction):
        expected_std = torch.where(
            after_masks,
            actual_input_before_context._observation_std_dbz,
            torch.ones_like(actual_input_before_context._observation_std_dbz),
        )
        if not torch.equal(
            actual_input_after_context._observation_std_dbz, expected_std
        ):
            raise ValueError("QC receipt changed observation standard deviation")
        if (
            tensor_digest(before_masks) == tensor_digest(after_masks)
            and tensor_digest(before_quality) == tensor_digest(after_quality)
        ):
            raise ValueError("QC receipt did not change mask or quality")
        if actual_input_before_run.fixed_input_context_digest == (
            actual_input_after_run.fixed_input_context_digest
        ):
            raise ValueError("QC receipt did not change its fixed input context")
    else:
        if actual_input_before_run.observation_std_dbz_digest != (
            actual_input_after_run.observation_std_dbz_digest
        ):
            raise ValueError("receipt changed non-radar input state")
        if before_digest == after_digest or (
            actual_input_before_run.input_bundle_digest
            == actual_input_after_run.input_bundle_digest
        ):
            raise ValueError("dBZ receipt did not change its radar input bundle")
        if actual_input_before_run.fixed_input_context_digest != (
            actual_input_after_run.fixed_input_context_digest
        ):
            raise ValueError("receipt changed non-radar input state")
    if actual_input_before_run.full_analysis_input_digest == (
        actual_input_after_run.full_analysis_input_digest
    ):
        raise ValueError("receipt did not change its full analysis input")
    action_artifact_digest = _intervention_action_artifact_digest(
        generator_digest=action_generator.generator_digest,
        before_context_digest=actual_input_before_context.context_digest,
        after_context_digest=actual_input_after_context.context_digest,
        action_payload_digest=action.payload_digest,
        before_frames_digest=before_digest,
        after_frames_digest=after_digest,
        before_masks_digest=tensor_digest(before_masks),
        after_masks_digest=tensor_digest(after_masks),
        before_quality_weight_digest=tensor_digest(before_quality),
        after_quality_weight_digest=tensor_digest(after_quality),
        grid_time_contract=actual_input_before_run.grid_time_contract,
    )
    return InterventionActionTransition(
        before_frames_digest=before_digest,
        after_frames_digest=after_digest,
        before_masks_digest=tensor_digest(before_masks),
        after_masks_digest=tensor_digest(after_masks),
        before_quality_weight_digest=tensor_digest(before_quality),
        after_quality_weight_digest=tensor_digest(after_quality),
        action_safety_diagnostics_digest=diagnostics.diagnostics_digest,
        action_artifact_digest=action_artifact_digest,
    )


def _new_realized_intervention_receipt(
    **values: object,
) -> RealizedInterventionReceipt:
    values.setdefault("contract", "realized-intervention-receipt-v5")
    expected = {
        item.name
        for item in fields(RealizedInterventionReceipt)
        if item.name != "receipt_digest"
    }
    if set(values) != expected:
        raise ValueError("realized receipt fields are incomplete")
    result = object.__new__(RealizedInterventionReceipt)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    RealizedInterventionReceipt.__post_init__(result)
    return result


def verify_intervention_receipt_signature(
    receipt: RealizedInterventionReceipt,
    executor_public_key: Ed25519PublicKey,
) -> None:
    values = {
        key: value
        for key, value in receipt.__dict__.items()
        if key not in ("receipt_digest", "executor_signature")
    }
    values["executor_signature"] = ""
    try:
        executor_public_key.verify(
            bytes.fromhex(receipt.executor_signature),
            json_digest(values).encode("ascii"),
        )
    except InvalidSignature as error:
        raise ValueError("realized receipt executor signature mismatch")


def validate_prospective_intervention(
    decision: ProspectiveInterventionDecision,
    receipt: RealizedInterventionReceipt,
) -> None:
    expected_decision = json_digest(
        {
            key: value
            for key, value in decision.__dict__.items()
            if key != "decision_digest"
        }
    )
    expected_receipt = json_digest(
        {
            key: value
            for key, value in receipt.__dict__.items()
            if key != "receipt_digest"
        }
    )
    if decision.decision_digest != expected_decision:
        raise ValueError("prospective decision digest mismatch")
    if receipt.receipt_digest != expected_receipt:
        raise ValueError("realized intervention receipt digest mismatch")
    pairs = (
        (receipt.decision_digest, decision.decision_digest),
        (receipt.decision_id, decision.decision_id),
        (receipt.case_id, decision.case_id),
        (receipt.radar_id, decision.radar_id),
        (receipt.intervention_type, decision.intervention_type),
        (receipt.action_digest, decision.action_digest),
        (receipt.action_payload_digest, decision.action_payload_digest),
        (
            receipt.action_application_contract_digest,
            decision.action_application_contract_digest,
        ),
        (receipt.input_plan_digest, decision.input_plan_digest),
        (
            receipt.input_plan_resolution_digest,
            decision.input_plan_resolution_digest,
        ),
        (
            receipt.actual_input_before_frames_digest,
            decision.actual_input_before_frames_digest,
        ),
        (
            receipt.actual_input_before_bundle_digest,
            decision.actual_input_before_bundle_digest,
        ),
        (
            receipt.full_analysis_input_before_digest,
            decision.actual_input_before_full_analysis_input_digest,
        ),
        (
            receipt.action_safety_diagnostics_digest,
            decision.action_safety_diagnostics_digest,
        ),
        (receipt.observation_valid_time, decision.observation_valid_time),
        (receipt.input_available_time, decision.input_available_time),
        (receipt.publication_time, decision.publication_time),
    )
    if any(actual != expected for actual, expected in pairs):
        raise ValueError("realized receipt disagrees with its decision")


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
