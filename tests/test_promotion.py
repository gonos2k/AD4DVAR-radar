from __future__ import annotations

from unittest.mock import patch
import unittest
from types import SimpleNamespace

import torch

import advar.promotion as promotion_module
from advar import (
    LearningApprovalEvidence,
    NeuralPriorCandidateManifest,
    NeuralPriorHoldoutCase,
    NeuralPriorPromotionPolicy,
    PromotionMetricScale,
    RealizedObservationIntervention,
    compute_neural_prior_promotion,
    validate_neural_prior_promotion,
)
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
        )

    def manifest(self) -> NeuralPriorCandidateManifest:
        return NeuralPriorCandidateManifest(
            candidate_prior_digest="c" * 64,
            parent_prior_digest="d" * 64,
            training_learning_approval_digests=("a" * 64,),
            training_intervention_digests=("b" * 64,),
            training_dataset_digest="1" * 64,
            model_contract_digest="2" * 64,
            algorithm_bundle_digest="3" * 64,
            numerical_runtime_digest="4" * 64,
            holdout_dataset_digest="5" * 64,
            training_case_ids=("training-case",),
            holdout_cases=(
                NeuralPriorHoldoutCase(
                    "case-1", "storm-1", "2026-08-08", "radar-1",
                    "convective", "near_range",
                    "6" * 64, "7" * 64,
                ),
                NeuralPriorHoldoutCase(
                    "case-2", "storm-2", "2026-08-09", "radar-1",
                    "stratiform", "far_range",
                    "8" * 64, "9" * 64,
                ),
            ),
        )

    def evaluation(self, index: int, change: float):
        learning = self.learning_evidence()
        manifest = self.manifest()
        case = manifest.holdout_cases[index - 1]
        intervention = RealizedObservationIntervention(
            intervention_id=f"qc-{index}",
            intervention_type="realized_qc_intervention",
            action_digest="a" * 64,
            applied_time=f"2026-08-0{7 + index}T00:00:00Z",
            actual_input_before_digest="b" * 64,
            actual_input_after_digest="c" * 64,
            outcome_resolution_contract_digest="d" * 64,
            execution_policy_digest="e" * 64,
            execution_trust_store_digest="9" * 64,
            predicted_normalized_benefit=0.2,
            resolved_normalized_benefit=0.2,
            learning_result_digest=("d" if index == 1 else "e") * 64,
            learning_approval_evidence_digest=learning.digest,
            counterfactual_perturbation_digest="f" * 64,
            linearization_digest="0" * 64,
        )
        return promotion_module._new_realized_evaluation(
            intervention_digest=intervention.intervention_digest,
            intervention_type=intervention.intervention_type,
            learning_result_digest=intervention.learning_result_digest,
            learning_approval_evidence_digest=learning.digest,
            learning_policy_digest=learning.policy_digest,
            candidate_manifest_digest=manifest.manifest_digest,
            candidate_prior_digest=manifest.candidate_prior_digest,
            parent_prior_digest=manifest.parent_prior_digest,
            case_id=case.case_id,
            storm_id=case.storm_id,
            day=case.day,
            radar_id=case.radar_id,
            regime=case.regime,
            range_regime=case.range_regime,
            candidate_forecast_digest=case.candidate_forecast_digest,
            parent_forecast_digest=case.parent_forecast_digest,
            metric_change=torch.tensor([[change]], dtype=torch.float64),
            metric_available=torch.tensor([[True]]),
            lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            verification_digest="a" * 64,
            metric_contract_digest="b" * 64,
            coverage_candidate=torch.tensor([1.0], dtype=torch.float64),
            coverage_parent=torch.tensor([1.0], dtype=torch.float64),
            issue_time=f"2026-08-{8 + index:02d}T00:00:00Z",
            applied_time=intervention.applied_time,
            verification_valid_times=(
                f"2026-08-{8 + index:02d}T01:00:00Z",
            ),
        )

    def policy(self) -> NeuralPriorPromotionPolicy:
        return NeuralPriorPromotionPolicy(
            metric_scales=(PromotionMetricScale("log_echo_mse", 1.0, 0.01),),
            approved_learning_policy_digests=("1" * 64,),
            approved_candidate_manifest_digests=(self.manifest().manifest_digest,),
            allowed_intervention_types=("realized_qc_intervention",),
            minimum_realized_interventions=2,
            minimum_material_interventions=2,
            minimum_material_intervention_fraction=1.0,
            minimum_independent_cases=2,
            minimum_distinct_storms=2,
            minimum_distinct_days=2,
            minimum_distinct_radars=1,
            minimum_distinct_regimes=2,
            minimum_distinct_range_regimes=2,
            minimum_beneficial_fraction=1.0,
            maximum_harmful_fraction=0.0,
            minimum_mean_normalized_improvement=0.1,
            bootstrap_samples=32,
        )

    def compute(self, evaluations, *, approved: bool = True):
        policy = self.policy()
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=_LearningPolicyTrustStore(
                approved_policy_digests=(
                    frozenset((policy.digest,)) if approved else frozenset()
                ),
                content_digest="b" * 64,
            ),
        ):
            return compute_neural_prior_promotion(
                self.manifest(), evaluations, policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_promotes_only_independent_material_holdout_cases(self) -> None:
        result = self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        self.assertTrue(result.eligible, result.rejection_reasons)
        self.assertEqual(result.material_intervention_count, 2)
        self.assertEqual(result.beneficial_fraction_lower_bound, 1.0)
        validate_neural_prior_promotion(result)

    def test_one_material_case_cannot_represent_twenty_interventions(self) -> None:
        policy = self.policy()
        policy = NeuralPriorPromotionPolicy(
            **{
                **policy.__dict__,
                "minimum_realized_interventions": 2,
                "minimum_material_interventions": 2,
                "minimum_material_intervention_fraction": 1.0,
            }
        )
        result = self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.001)))
        self.assertFalse(result.eligible)
        self.assertIn("insufficient_material_interventions", result.rejection_reasons)

    def test_repeated_cycles_do_not_count_as_independent_storms(self) -> None:
        first = self.evaluation(1, -0.2)
        second = self.evaluation(2, -0.2)
        values = {
            key: value
            for key, value in second.__dict__.items()
            if key not in ("evaluation_digest", "storm_id", "day")
        }
        repeated = promotion_module._new_realized_evaluation(
            **values,
            storm_id=first.storm_id,
            day=first.day,
        )
        result = self.compute((first, repeated))
        self.assertFalse(result.eligible)
        self.assertIn("insufficient_distinct_storms", result.rejection_reasons)
        self.assertIn("insufficient_distinct_days", result.rejection_reasons)

    def test_training_and_holdout_cases_must_be_disjoint(self) -> None:
        values = {
            key: value
            for key, value in self.manifest().__dict__.items()
            if key not in ("manifest_digest", "training_case_ids")
        }
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            NeuralPriorCandidateManifest(
                **values,
                training_case_ids=("case-1",),
            )

    def test_rejects_unrelated_candidate_manifest(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        second = self.evaluation(2, -0.2)
        changed = NeuralPriorCandidateManifest(
            **{
                **{key: value for key, value in self.manifest().__dict__.items() if key != "manifest_digest"},
                "candidate_prior_digest": "f" * 64,
            }
        )
        with self.assertRaisesRegex(ValueError, "candidate manifest"):
            with patch.object(
                promotion_module,
                "_load_learning_policy_trust_store",
                return_value=_LearningPolicyTrustStore(
                    frozenset((self.policy().digest,)), "b" * 64
                ),
            ):
                compute_neural_prior_promotion(
                    changed, (evaluation, second), policy=self.policy(),
                    policy_trust_store_path="/etc/advar/learning-policies.json",
                )

    def test_detects_mutated_resolved_metric(self) -> None:
        first = self.evaluation(1, -0.2)
        first.metric_change[0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "evaluation digest mismatch"):
            self.compute((first, self.evaluation(2, -0.2)))

    def test_direct_evaluation_construction_is_disabled(self) -> None:
        with self.assertRaises(TypeError):
            promotion_module.RealizedInterventionEvaluation()  # type: ignore[call-arg]

    def test_forecast_factory_recomputes_candidate_minus_parent(self) -> None:
        manifest = self.manifest()
        learning = self.learning_evidence()
        intervention = RealizedObservationIntervention(
            intervention_id="qc-factory",
            intervention_type="realized_qc_intervention",
            action_digest="a" * 64,
            applied_time="2026-08-08T00:00:00Z",
            actual_input_before_digest="b" * 64,
            actual_input_after_digest="c" * 64,
            outcome_resolution_contract_digest="d" * 64,
            execution_policy_digest="e" * 64,
            execution_trust_store_digest="9" * 64,
            predicted_normalized_benefit=0.2,
            resolved_normalized_benefit=0.2,
            learning_result_digest="f" * 64,
            learning_approval_evidence_digest=learning.digest,
            counterfactual_perturbation_digest="0" * 64,
            linearization_digest="1" * 64,
        )
        grid = SimpleNamespace(valid_times=("2026-08-08T00:00:00Z",))
        run = SimpleNamespace(
            grid_time_contract_digest="2" * 64,
            grid_time_contract=grid,
        )
        candidate = SimpleNamespace(run=run, state=object(), validate_issuance=lambda: None)
        parent = SimpleNamespace(run=run, state=object(), validate_issuance=lambda: None)
        verification = SimpleNamespace(valid_times=("2026-08-08T01:00:00Z",))
        config = SimpleNamespace(
            full_map_lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            digest="3" * 64,
        )
        resolved = SimpleNamespace(content_digest="4" * 64)
        with (
            patch.object(
                promotion_module,
                "_forecast_result_content_digest",
                side_effect=("6" * 64, "7" * 64),
            ),
            patch.object(
                promotion_module,
                "_resolve_verification",
                return_value=resolved,
            ),
            patch.object(
                promotion_module,
                "_resolved_forecast_scores",
                side_effect=(
                    (torch.tensor([[0.8]]), torch.tensor([[True]])),
                    (torch.tensor([[1.0]]), torch.tensor([[True]])),
                ),
            ),
            patch.object(
                promotion_module,
                "_forecast_coverage",
                side_effect=(torch.tensor([0.9]), torch.tensor([1.0])),
            ),
        ):
            evaluation = promotion_module.RealizedInterventionEvaluation.from_forecasts(
                intervention,
                learning,
                manifest,
                case_id="case-1",
                candidate_forecast=candidate,
                parent_forecast=parent,
                verification=verification,
                metric_config=config,
            )
        self.assertAlmostEqual(float(evaluation.metric_change[0, 0]), -0.2)
        self.assertEqual(evaluation.candidate_manifest_digest, manifest.manifest_digest)


if __name__ == "__main__":
    unittest.main()
