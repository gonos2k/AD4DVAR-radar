from __future__ import annotations

from unittest.mock import patch
import unittest

import torch

import advar.promotion as promotion_module
from advar import (
    LearningApprovalEvidence,
    NeuralPriorPromotionPolicy,
    PromotionMetricScale,
    RealizedInterventionEvaluation,
    RealizedObservationIntervention,
    compute_neural_prior_promotion,
    validate_neural_prior_promotion,
)
from advar._digest import tensor_digest
from advar.sensitivity import _LearningPolicyTrustStore


class NeuralPriorPromotionTests(unittest.TestCase):
    def learning_evidence(self) -> LearningApprovalEvidence:
        return LearningApprovalEvidence(
            policy_digest="1" * 64,
            trust_store_digest="2" * 64,
            fsoi_digest="3" * 64,
            full_step_analysis_digest="4" * 64,
            half_step_analysis_digest="5" * 64,
            full_step_forecast_digest="6" * 64,
            half_step_forecast_digest="7" * 64,
            first_order_validation_digest="8" * 64,
            learning_impact_digest="9" * 64,
            contract="p1-learning-approval-evidence-v1",
        )

    def evaluation(
        self,
        identifier: str,
        change_value: float,
    ) -> RealizedInterventionEvaluation:
        learning = self.learning_evidence()
        change = torch.tensor([[change_value]], dtype=torch.float64)
        intervention = RealizedObservationIntervention(
            intervention_id=identifier,
            intervention_type="realized_qc_intervention",
            action_digest="a" * 64,
            applied_time="2026-08-08T00:00:00Z",
            actual_input_before_digest="b" * 64,
            actual_input_after_digest="c" * 64,
            observed_outcome_digest=tensor_digest(change),
            learning_result_digest=("d" if identifier.endswith("1") else "e")
            * 64,
            learning_approval_evidence_digest=learning.digest,
            counterfactual_perturbation_digest="f" * 64,
            linearization_digest="0" * 64,
        )
        return RealizedInterventionEvaluation.from_evidence(
            intervention,
            learning,
            metric_change=change,
            metric_available=torch.ones_like(change, dtype=torch.bool),
            lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            verification_digest="a" * 64,
        )

    def policy(self) -> NeuralPriorPromotionPolicy:
        return NeuralPriorPromotionPolicy(
            metric_scales=(
                PromotionMetricScale("log_echo_mse", 1.0, 0.01),
            ),
            approved_learning_policy_digests=("1" * 64,),
            allowed_intervention_types=("realized_qc_intervention",),
            minimum_realized_interventions=2,
            minimum_beneficial_fraction=1.0,
            maximum_harmful_fraction=0.0,
            minimum_mean_normalized_improvement=0.1,
        )

    def test_promotes_only_realized_beneficial_outcomes(self) -> None:
        policy = self.policy()
        evaluations = (
            self.evaluation("qc-1", -0.2),
            self.evaluation("qc-2", -0.3),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=_LearningPolicyTrustStore(
                approved_policy_digests=frozenset((policy.digest,)),
                content_digest="b" * 64,
            ),
        ):
            result = compute_neural_prior_promotion(
                "c" * 64,
                "d" * 64,
                evaluations,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

        self.assertTrue(result.eligible, result.rejection_reasons)
        self.assertEqual(result.realized_intervention_count, 2)
        self.assertEqual(result.beneficial_fraction, 1.0)
        self.assertAlmostEqual(result.mean_normalized_improvement, 0.25)
        validate_neural_prior_promotion(result)

    def test_rejects_harmful_or_unapproved_evidence(self) -> None:
        policy = self.policy()
        evaluations = (
            self.evaluation("qc-1", -0.2),
            self.evaluation("qc-2", 0.3),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=_LearningPolicyTrustStore(
                approved_policy_digests=frozenset(),
                content_digest="b" * 64,
            ),
        ):
            result = compute_neural_prior_promotion(
                "c" * 64,
                "d" * 64,
                evaluations,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

        self.assertFalse(result.eligible)
        self.assertIn("unapproved_promotion_policy", result.rejection_reasons)
        self.assertIn("insufficient_beneficial_fraction", result.rejection_reasons)
        self.assertIn("excessive_harmful_fraction", result.rejection_reasons)
        with self.assertRaisesRegex(ValueError, "not eligible"):
            validate_neural_prior_promotion(result)

    def test_detects_mutated_realized_outcome(self) -> None:
        evaluation = self.evaluation("qc-1", -0.2)
        second = self.evaluation("qc-2", -0.2)
        evaluation.metric_change[0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "evaluation digest mismatch"):
            with patch.object(
                promotion_module,
                "_load_learning_policy_trust_store",
                return_value=_LearningPolicyTrustStore(
                    approved_policy_digests=frozenset(),
                    content_digest="b" * 64,
                ),
            ):
                compute_neural_prior_promotion(
                    "c" * 64,
                    "d" * 64,
                    (evaluation, second),
                    policy=self.policy(),
                    policy_trust_store_path=(
                        "/etc/advar/learning-policies.json"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
