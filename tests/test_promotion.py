from __future__ import annotations

from unittest.mock import patch
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

import advar.promotion as promotion_module
from advar._digest import json_digest
from advar import (
    LearningApprovalEvidence,
    NeuralPriorCandidateManifest,
    NeuralPriorHoldoutCase,
    NeuralPriorHoldoutPlan,
    NeuralPriorHoldoutPlanCase,
    NeuralPriorPromotionPolicy,
    PromotionMetricScale,
    RealizedObservationIntervention,
    compute_neural_prior_promotion,
    validate_neural_prior_promotion,
    validate_neural_prior_candidate_manifest,
    validate_neural_prior_holdout_plan,
    verification_plan_digest,
)
from advar.sensitivity import _LearningPolicyTrustStore


class NeuralPriorPromotionTests(unittest.TestCase):
    def verification_plan(self, valid_time: str) -> str:
        return verification_plan_digest(
            valid_times=(valid_time,),
            grid_contract_digest="2" * 64,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
        )

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

    def plan(self) -> NeuralPriorHoldoutPlan:
        return NeuralPriorHoldoutPlan(
            plan_id="holdout-plan",
            parent_prior_digest="d" * 64,
            candidate_family_digests=("c" * 64,),
            cases=(
                NeuralPriorHoldoutPlanCase(
                    "case-1", "storm-1", "2026-08-08", "radar-1",
                    "convective", "near_range", "e" * 64,
                    self.verification_plan("2026-08-09T01:00:00Z"),
                    "b" * 64, "2026-08-09T00:00:00Z",
                ),
                NeuralPriorHoldoutPlanCase(
                    "case-2", "storm-2", "2026-08-09", "radar-1",
                    "stratiform", "far_range", "f" * 64,
                    self.verification_plan("2026-08-10T01:00:00Z"),
                    "b" * 64, "2026-08-10T00:00:00Z",
                ),
            ),
            registered_at="2026-08-07T00:00:00Z",
        )

    def manifest(self) -> NeuralPriorCandidateManifest:
        plan = self.plan()
        return NeuralPriorCandidateManifest(
            candidate_prior_digest="c" * 64,
            parent_prior_digest="d" * 64,
            training_learning_approval_digests=("a" * 64,),
            training_intervention_digests=("b" * 64,),
            training_dataset_digest="1" * 64,
            candidate_training_manifest_digest="2" * 64,
            parent_training_manifest_digest="3" * 64,
            model_contract_digest="2" * 64,
            feature_schema_digest="4" * 64,
            algorithm_bundle_digest="3" * 64,
            numerical_runtime_digest="4" * 64,
            holdout_dataset_digest=plan.holdout_dataset_digest,
            holdout_plan_digest=plan.plan_digest,
            training_case_ids=("training-case",),
            training_input_bundle_digests=("1" * 64,),
            holdout_cases=(
                NeuralPriorHoldoutCase(
                    "case-1", "storm-1", "2026-08-08", "radar-1",
                    "convective", "near_range",
                    "e" * 64,
                    self.verification_plan("2026-08-09T01:00:00Z"),
                    "a" * 64, "b" * 64,
                    "2026-08-09T00:00:00Z", "6" * 64, "7" * 64,
                ),
                NeuralPriorHoldoutCase(
                    "case-2", "storm-2", "2026-08-09", "radar-1",
                    "stratiform", "far_range",
                    "f" * 64,
                    self.verification_plan("2026-08-10T01:00:00Z"),
                    "a" * 64, "b" * 64,
                    "2026-08-10T00:00:00Z", "8" * 64, "9" * 64,
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
            holdout_plan_digest=manifest.holdout_plan_digest,
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
            candidate_issuance_effect=torch.zeros((1, 1), dtype=torch.float64),
            parent_issuance_effect=torch.zeros((1, 1), dtype=torch.float64),
            metric_available=torch.tensor([[True]]),
            lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            verification_digest="a" * 64,
            metric_contract_digest="b" * 64,
            coverage_candidate=torch.tensor([1.0], dtype=torch.float64),
            coverage_parent=torch.tensor([1.0], dtype=torch.float64),
            coverage_common=torch.tensor([1.0], dtype=torch.float64),
            newly_issued_fraction=torch.tensor([0.0], dtype=torch.float64),
            withdrawn_fraction=torch.tensor([0.0], dtype=torch.float64),
            issue_time=case.issue_time,
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
            approved_holdout_plan_digests=(self.plan().plan_digest,),
            approved_metric_contract_digests=("b" * 64,),
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
            minimum_material_clusters=2,
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
                self.manifest(), self.plan(), evaluations, policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def forecast_evaluation_inputs(self):
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
            case_id="case-1",
            radar_id="radar-1",
            issue_time="2026-08-09T00:00:00Z",
            input_bundle_before_digest="f" * 64,
            input_bundle_after_digest="e" * 64,
            resolved_issuance_validation_digest="1" * 64,
            contract="realized-observation-intervention-v3",
        )
        grid = SimpleNamespace(valid_times=("2026-08-09T00:00:00Z",))
        common_run = dict(
            grid_time_contract_digest="2" * 64,
            grid_time_contract=grid,
            input_bundle_digest="e" * 64,
            config=SimpleNamespace(digest="3" * 64, interval_minutes=10),
            analysis_config_digest="4" * 64,
            operational_calibration_manifest_digest="5" * 64,
            operational_data_identity_digest="6" * 64,
            prior_model_contract_digest=manifest.model_contract_digest,
            prior_feature_schema_digest=manifest.feature_schema_digest,
        )
        candidate_run = SimpleNamespace(
            **common_run,
            neural_prior_digest=manifest.candidate_prior_digest,
            prior_application_digest="7" * 64,
            prior_role="candidate",
            prior_training_manifest_digest=(
                manifest.candidate_training_manifest_digest
            ),
        )
        parent_run = SimpleNamespace(
            **common_run,
            neural_prior_digest=manifest.parent_prior_digest,
            prior_application_digest="8" * 64,
            prior_role="parent",
            prior_training_manifest_digest=manifest.parent_training_manifest_digest,
        )
        candidate = SimpleNamespace(
            run=candidate_run, state=object(), validate_issuance=lambda: None
        )
        parent = SimpleNamespace(
            run=parent_run, state=object(), validate_issuance=lambda: None
        )
        verification = SimpleNamespace(valid_times=("2026-08-09T01:00:00Z",))
        config = SimpleNamespace(
            full_map_lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            digest="b" * 64,
        )
        resolved = SimpleNamespace(
            content_digest="a" * 64,
            valid_times=("2026-08-09T01:00:00Z",),
            grid_contract_digest="2" * 64,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            valid_mask=torch.ones((6, 1, 1), dtype=torch.bool),
            frames_dbz=torch.ones((6, 1, 1)),
        )
        return (
            intervention,
            learning,
            manifest,
            candidate,
            parent,
            verification,
            config,
            resolved,
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

    def test_training_and_holdout_inputs_must_be_disjoint(self) -> None:
        manifest = self.manifest()
        values = {
            key: value
            for key, value in manifest.__dict__.items()
            if key not in ("manifest_digest", "training_input_bundle_digests")
        }
        with self.assertRaisesRegex(ValueError, "inputs must be disjoint"):
            NeuralPriorCandidateManifest(
                **values,
                training_input_bundle_digests=(
                    manifest.holdout_cases[0].input_bundle_digest,
                ),
            )

    def test_stale_plan_and_manifest_digests_are_rejected(self) -> None:
        plan = self.plan()
        object.__setattr__(plan, "cases", tuple(reversed(plan.cases)))
        with self.assertRaisesRegex(ValueError, "plan digest mismatch"):
            validate_neural_prior_holdout_plan(plan)

        manifest = self.manifest()
        object.__setattr__(manifest, "training_case_ids", ("mutated",))
        with self.assertRaisesRegex(ValueError, "manifest digest mismatch"):
            validate_neural_prior_candidate_manifest(manifest)

    def test_holdout_cases_cannot_relabel_the_same_input(self) -> None:
        plan = self.plan()
        duplicated = replace(
            plan.cases[1],
            input_bundle_digest=plan.cases[0].input_bundle_digest,
        )
        with self.assertRaisesRegex(ValueError, "distinct inputs"):
            NeuralPriorHoldoutPlan(
                plan_id=plan.plan_id,
                parent_prior_digest=plan.parent_prior_digest,
                candidate_family_digests=plan.candidate_family_digests,
                cases=(plan.cases[0], duplicated),
                registered_at=plan.registered_at,
            )

    def test_metric_harm_is_not_removed_from_material_statistics(self) -> None:
        policy = NeuralPriorPromotionPolicy(
            **{
                **self.policy().__dict__,
                "metric_scales": (
                    PromotionMetricScale(
                        "log_echo_mse",
                        1.0,
                        0.01,
                        maximum_normalized_degradation=0.1,
                    ),
                ),
            }
        )
        score, _, exceeded = promotion_module._intervention_score(
            self.evaluation(1, 0.2), policy
        )
        self.assertIsNotNone(score)
        self.assertTrue(exceeded)

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
                    changed,
                    self.plan(),
                    (evaluation, second),
                    policy=self.policy(),
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

    def test_legacy_intervention_remains_auditable_but_not_promotable(self) -> None:
        legacy = RealizedObservationIntervention(
            intervention_id="legacy-qc",
            intervention_type="realized_qc_intervention",
            action_digest="a" * 64,
            applied_time="2026-08-08T00:00:00Z",
            actual_input_before_digest="b" * 64,
            actual_input_after_digest="c" * 64,
            outcome_resolution_contract_digest="d" * 64,
            execution_policy_digest="0" * 64,
            execution_trust_store_digest="0" * 64,
            predicted_normalized_benefit=0.0,
            resolved_normalized_benefit=0.0,
            learning_result_digest="e" * 64,
            learning_approval_evidence_digest="f" * 64,
            counterfactual_perturbation_digest="1" * 64,
            linearization_digest="2" * 64,
            contract="realized-observation-intervention-v1",
        )
        self.assertEqual(
            legacy.intervention_digest,
            json_digest(
                {
                    "contract": legacy.contract,
                    "intervention_id": legacy.intervention_id,
                    "intervention_type": legacy.intervention_type,
                    "action_digest": legacy.action_digest,
                    "applied_time": legacy.applied_time,
                    "actual_input_before_digest": legacy.actual_input_before_digest,
                    "actual_input_after_digest": legacy.actual_input_after_digest,
                    "observed_outcome_digest": "d" * 64,
                    "learning_result_digest": legacy.learning_result_digest,
                    "learning_approval_evidence_digest": (
                        legacy.learning_approval_evidence_digest
                    ),
                    "counterfactual_perturbation_digest": (
                        legacy.counterfactual_perturbation_digest
                    ),
                    "linearization_digest": legacy.linearization_digest,
                }
            ),
        )

    def test_forecast_factory_recomputes_candidate_minus_parent(self) -> None:
        (
            intervention,
            learning,
            manifest,
            candidate,
            parent,
            verification,
            config,
            resolved,
        ) = self.forecast_evaluation_inputs()
        candidate_weights = torch.full((1, 1, 1), 0.5)
        parent_weights = torch.ones((1, 1, 1))
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
                "_resolved_forecast_domain_weights",
                side_effect=(candidate_weights, parent_weights),
            ),
            patch.object(
                promotion_module,
                "_resolved_forecast_scores",
                side_effect=(
                    (torch.tensor([[0.8]]), torch.tensor([[True]])),
                    (torch.tensor([[1.0]]), torch.tensor([[True]])),
                    (torch.tensor([[0.75]]), torch.tensor([[True]])),
                    (torch.tensor([[1.0]]), torch.tensor([[True]])),
                ),
            ) as score_mock,
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
                self.plan(),
                case_id="case-1",
                candidate_forecast=candidate,
                parent_forecast=parent,
                verification=verification,
                metric_config=config,
            )
        self.assertAlmostEqual(float(evaluation.metric_change[0, 0]), -0.2)
        self.assertAlmostEqual(
            float(evaluation.candidate_issuance_effect[0, 0]), -0.05
        )
        self.assertAlmostEqual(
            float(evaluation.parent_issuance_effect[0, 0]), 0.0
        )
        torch.testing.assert_close(
            score_mock.call_args_list[0].kwargs["domain_weights"],
            parent_weights,
        )
        torch.testing.assert_close(
            score_mock.call_args_list[2].kwargs["domain_weights"],
            candidate_weights,
        )
        self.assertEqual(evaluation.candidate_manifest_digest, manifest.manifest_digest)

    def test_forecast_factory_rejects_different_holdout_inputs(self) -> None:
        values = list(self.forecast_evaluation_inputs())
        parent = values[4]
        run_values = {**parent.run.__dict__, "input_bundle_digest": "f" * 64}
        values[4] = SimpleNamespace(
            run=SimpleNamespace(**run_values),
            state=parent.state,
            validate_issuance=parent.validate_issuance,
        )
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=("6" * 64, "7" * 64),
        ), self.assertRaisesRegex(ValueError, "holdout inputs disagree"):
            promotion_module.RealizedInterventionEvaluation.from_forecasts(
                values[0], values[1], values[2], self.plan(),
                case_id="case-1", candidate_forecast=values[3],
                parent_forecast=values[4], verification=values[5],
                metric_config=values[6],
            )

    def test_forecast_factory_rejects_prior_lineage_mismatch(self) -> None:
        values = list(self.forecast_evaluation_inputs())
        candidate = values[3]
        run_values = {**candidate.run.__dict__, "neural_prior_digest": "f" * 64}
        values[3] = SimpleNamespace(
            run=SimpleNamespace(**run_values),
            state=candidate.state,
            validate_issuance=candidate.validate_issuance,
        )
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=("6" * 64, "7" * 64),
        ), self.assertRaisesRegex(ValueError, "prior lineage disagrees"):
            promotion_module.RealizedInterventionEvaluation.from_forecasts(
                values[0], values[1], values[2], self.plan(),
                case_id="case-1", candidate_forecast=values[3],
                parent_forecast=values[4], verification=values[5],
                metric_config=values[6],
            )

    def test_forecast_factory_requires_registered_verification_and_metric(self) -> None:
        values = self.forecast_evaluation_inputs()
        wrong_verification = SimpleNamespace(
            **{**values[7].__dict__, "content_digest": "f" * 64}
        )
        with (
            patch.object(
                promotion_module,
                "_forecast_result_content_digest",
                side_effect=("6" * 64, "7" * 64),
            ),
            patch.object(
                promotion_module,
                "_resolve_verification",
                return_value=wrong_verification,
            ),
            self.assertRaisesRegex(ValueError, "verification content"),
        ):
            promotion_module.RealizedInterventionEvaluation.from_forecasts(
                values[0], values[1], values[2], self.plan(),
                case_id="case-1", candidate_forecast=values[3],
                parent_forecast=values[4], verification=values[5],
                metric_config=values[6],
            )

        wrong_config = SimpleNamespace(
            **{**values[6].__dict__, "digest": "f" * 64}
        )
        with (
            patch.object(
                promotion_module,
                "_forecast_result_content_digest",
                side_effect=("6" * 64, "7" * 64),
            ),
            patch.object(
                promotion_module,
                "_resolve_verification",
                return_value=values[7],
            ),
            self.assertRaisesRegex(ValueError, "metric contract"),
        ):
            promotion_module.RealizedInterventionEvaluation.from_forecasts(
                values[0], values[1], values[2], self.plan(),
                case_id="case-1", candidate_forecast=values[3],
                parent_forecast=values[4], verification=values[5],
                metric_config=wrong_config,
            )


if __name__ == "__main__":
    unittest.main()
