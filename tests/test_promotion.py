from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import advar.promotion as promotion_module
import advar.ledger as ledger_module
from advar.nowcast import _validate_input_plan_resolution
from advar import (
    NeuralPriorCandidateManifest,
    NeuralPriorHoldoutCase,
    NeuralPriorHoldoutPlan,
    NeuralPriorHoldoutPlanCase,
    NeuralPriorPromotionPolicy,
    PromotionMetricScale,
    ProspectiveInterventionDecision,
    RealizedInterventionReceipt,
    RealizedObservationIntervention,
    compute_neural_prior_promotion,
    validate_neural_prior_candidate_manifest,
    validate_neural_prior_promotion,
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

    def plan(self) -> NeuralPriorHoldoutPlan:
        input_plans = tuple(
            promotion_module.NeuralPriorInputPlan(
                valid_times=(issue,),
                grid_contract_digest="2" * 64,
                radar_product_digest="a" * 64,
                qc_pipeline_digest="9" * 64,
                background_cycle_rule_digest=("1" if index == 1 else "2") * 64,
                mask_policy_digest="3" * 64,
                issue_time=issue,
            )
            for index, issue in enumerate(
                ("2026-08-09T00:00:00Z", "2026-08-10T00:00:00Z"),
                start=1,
            )
        )
        return NeuralPriorHoldoutPlan(
            plan_id="holdout-plan",
            parent_prior_digest="d" * 64,
            candidate_family_digests=("c" * 64,),
            cases=(
                NeuralPriorHoldoutPlanCase(
                    "case-1",
                    "storm-1",
                    "2026-08-08",
                    "radar-1",
                    "convective",
                    "near_range",
                    input_plans[0].plan_digest,
                    self.verification_plan("2026-08-09T01:00:00Z"),
                    "b" * 64,
                    "2026-08-09T00:00:00Z",
                ),
                NeuralPriorHoldoutPlanCase(
                    "case-2",
                    "storm-2",
                    "2026-08-09",
                    "radar-1",
                    "stratiform",
                    "far_range",
                    input_plans[1].plan_digest,
                    self.verification_plan("2026-08-10T01:00:00Z"),
                    "b" * 64,
                    "2026-08-10T00:00:00Z",
                ),
            ),
            input_plans=input_plans,
            registered_at="2026-08-07T00:00:00Z",
        )

    def completed_case(self, index: int) -> NeuralPriorHoldoutCase:
        planned = self.plan().cases[index - 1]
        return NeuralPriorHoldoutCase(
            case_id=planned.case_id,
            storm_id=planned.storm_id,
            day=planned.day,
            radar_id=planned.radar_id,
            regime=planned.regime,
            range_regime=planned.range_regime,
            input_plan_digest=planned.input_plan_digest,
            input_bundle_digest=("e" if index == 1 else "f") * 64,
            verification_plan_digest=planned.verification_plan_digest,
            verification_bundle_digest="a" * 64,
            metric_contract_digest=planned.metric_contract_digest,
            issue_time=planned.issue_time,
            candidate_forecast_digest=("6" if index == 1 else "8") * 64,
            parent_forecast_digest=("7" if index == 1 else "9") * 64,
            candidate_prior_application_digest=("3" if index == 1 else "4") * 64,
            parent_prior_application_digest=("5" if index == 1 else "6") * 64,
            candidate_inference_evidence_digest=("7" if index == 1 else "8") * 64,
            parent_inference_evidence_digest=("9" if index == 1 else "0") * 64,
        )

    def test_input_plan_must_match_actual_operational_identity(self) -> None:
        plan = self.plan().input_plans[0]
        identity = promotion_module.OperationalDataIdentity(
            radar_class="test",
            qc_pipeline_digest=plan.qc_pipeline_digest,
            observation_error_model_digest="4" * 64,
            background_model_digest="5" * 64,
            radar_product_digest=plan.radar_product_digest,
            background_cycle_rule_digest=plan.background_cycle_rule_digest,
            mask_policy_digest=plan.mask_policy_digest,
        )
        grid = SimpleNamespace(
            digest=plan.grid_contract_digest,
            valid_times=plan.valid_times,
        )
        _validate_input_plan_resolution(plan.json, identity.json, grid)
        for field in (
            "radar_product_digest",
            "background_cycle_rule_digest",
            "mask_policy_digest",
        ):
            changed = replace(identity, **{field: "f" * 64})
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "operational data identity"):
                    _validate_input_plan_resolution(
                        plan.json,
                        changed.json,
                        grid,
                    )

    def manifest(self) -> NeuralPriorCandidateManifest:
        plan = self.plan()
        return NeuralPriorCandidateManifest(
            candidate_prior_digest="c" * 64,
            parent_prior_digest="d" * 64,
            training_learning_approval_digests=("a" * 64,),
            training_intervention_digests=("f" * 64,),
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
            training_input_bundle_digests=("0" * 64,),
            training_storm_ids=("training-storm",),
            training_days=("2026-07-01",),
            training_radars=("radar-1",),
            training_regimes=("convective",),
            training_time_windows=(
                (
                    "2026-07-01T00:00:00Z",
                    "2026-07-01T01:00:00Z",
                ),
            ),
            holdout_cases=(self.completed_case(1), self.completed_case(2)),
        )

    def evaluation(
        self,
        index: int,
        change: float,
        *,
        end_to_end: float | None = None,
    ) -> promotion_module.RealizedInterventionEvaluation:
        manifest = self.manifest()
        case = manifest.holdout_cases[index - 1]
        return promotion_module._new_realized_evaluation(
            intervention_digest=("a" if index == 1 else "b") * 64,
            intervention_type="realized_qc_intervention",
            learning_result_digest=("d" if index == 1 else "e") * 64,
            learning_approval_evidence_digest=("1" if index == 1 else "2") * 64,
            learning_policy_digest="1" * 64,
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
            candidate_prior_application_digest=(
                case.candidate_prior_application_digest
            ),
            parent_prior_application_digest=case.parent_prior_application_digest,
            candidate_inference_evidence_digest=(
                case.candidate_inference_evidence_digest
            ),
            parent_inference_evidence_digest=(case.parent_inference_evidence_digest),
            metric_change=torch.tensor([[change]], dtype=torch.float64),
            candidate_issuance_effect=torch.zeros((1, 1), dtype=torch.float64),
            parent_issuance_effect=torch.zeros((1, 1), dtype=torch.float64),
            end_to_end_metric_change=torch.tensor(
                [[change if end_to_end is None else end_to_end]],
                dtype=torch.float64,
            ),
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
            applied_time=f"2026-08-{7 + index:02d}T00:00:00Z",
            verification_valid_times=(f"2026-08-{8 + index:02d}T01:00:00Z",),
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

    def compute(self, evaluations):
        policy = self.policy()
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=_LearningPolicyTrustStore(
                approved_policy_digests=frozenset((policy.digest,)),
                content_digest="b" * 64,
            ),
        ):
            return compute_neural_prior_promotion(
                self.manifest(),
                self.plan(),
                evaluations,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_promotes_only_independent_material_holdout_cases(self) -> None:
        result = self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        self.assertTrue(result.eligible)
        validate_neural_prior_promotion(result)

    def test_end_to_end_harm_blocks_promotion(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2, end_to_end=2.0),
                self.evaluation(2, -0.3),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("excessive_end_to_end_degradation", result.rejection_reasons)

    def test_end_to_end_harm_is_checked_when_common_skill_is_immaterial(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.001, end_to_end=2.0),
                self.evaluation(2, -0.3),
            )
        )
        self.assertIn("excessive_end_to_end_degradation", result.rejection_reasons)

    def test_training_and_holdout_storms_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "storms must be disjoint"):
            replace(self.manifest(), training_storm_ids=("storm-1",))

    def test_training_and_holdout_inputs_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "inputs must be disjoint"):
            replace(
                self.manifest(),
                training_input_bundle_digests=(
                    self.manifest().holdout_cases[0].input_bundle_digest,
                ),
            )

    def test_mutated_metric_is_detected(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        evaluation.metric_change[0, 0] = -10.0
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.compute((evaluation, self.evaluation(2, -0.3)))

    def test_promotion_audit_payload_recomputes_tensor_digests(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        payload = ledger_module._evaluation_audit_payload(evaluation)
        decoded = ledger_module._decode_evaluation_audit_payloads([payload])
        self.assertEqual(decoded[0].evaluation_digest, evaluation.evaluation_digest)

        tensor_payload = payload["metric_change"]
        assert isinstance(tensor_payload, dict)
        values = tensor_payload["values"]
        assert isinstance(values, list)
        values[0][0] = 99.0
        with self.assertRaisesRegex(ValueError, "tensor digest mismatch"):
            ledger_module._decode_evaluation_audit_payloads([payload])

    def test_legacy_promotion_tensor_lists_load_as_audit_only(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        payload = ledger_module._evaluation_audit_payload(evaluation)
        for name, value in tuple(payload.items()):
            if isinstance(value, dict) and value.get("kind") == "tensor":
                payload[name] = value["values"]

        decoded = ledger_module._decode_evaluation_audit_payloads([payload])

        self.assertIsInstance(
            decoded[0],
            ledger_module.LegacyPromotionEvaluationAudit,
        )
        self.assertEqual(decoded[0].evaluation_digest, evaluation.evaluation_digest)

    def test_direct_evaluation_construction_is_disabled(self) -> None:
        with self.assertRaisesRegex(TypeError, "from_forecasts"):
            promotion_module.RealizedInterventionEvaluation()

    def test_legacy_intervention_is_not_a_prospective_receipt(self) -> None:
        legacy = RealizedObservationIntervention(
            intervention_id="legacy",
            intervention_type="realized_qc_intervention",
            action_digest="a" * 64,
            applied_time="2026-08-08T00:00:00Z",
            actual_input_before_digest="b" * 64,
            actual_input_after_digest="c" * 64,
            outcome_resolution_contract_digest="d" * 64,
            execution_policy_digest="e" * 64,
            execution_trust_store_digest="f" * 64,
            predicted_normalized_benefit=0.0,
            resolved_normalized_benefit=0.0,
            learning_result_digest="1" * 64,
            learning_approval_evidence_digest="2" * 64,
            counterfactual_perturbation_digest="3" * 64,
            linearization_digest="4" * 64,
        )
        self.assertNotIsInstance(legacy, RealizedInterventionReceipt)

    def test_prospective_receipt_rejects_backdated_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "precede its issue"):
            ProspectiveInterventionDecision(
                decision_id="decision-1",
                case_id="case-1",
                radar_id="radar-1",
                intervention_type="realized_qc_intervention",
                action_digest="a" * 64,
                input_plan_digest="b" * 64,
                actual_input_before_digest="c" * 64,
                decision_basis_digest="d" * 64,
                decision_policy_digest="e" * 64,
                decision_trust_store_digest="f" * 64,
                decided_at="2026-08-09T00:01:00Z",
                issue_time="2026-08-09T00:00:00Z",
            )

    def test_candidate_manifest_digest_detects_lineage_mutation(self) -> None:
        manifest = self.manifest()
        object.__setattr__(manifest, "training_storm_ids", ("changed",))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_neural_prior_candidate_manifest(manifest)

    def test_holdout_factory_uses_common_domain_and_inference_evidence(self) -> None:
        manifest = self.manifest()
        case = manifest.holdout_cases[0]
        plan = self.plan()
        planned_input = next(
            item for item in plan.input_plans
            if item.plan_digest == case.input_plan_digest
        )
        decision = ProspectiveInterventionDecision(
            decision_id="decision-1",
            case_id=case.case_id,
            radar_id=case.radar_id,
            intervention_type="realized_qc_intervention",
            action_digest="a" * 64,
            input_plan_digest=case.input_plan_digest,
            actual_input_before_digest="b" * 64,
            decision_basis_digest="d" * 64,
            decision_policy_digest="1" * 64,
            decision_trust_store_digest="f" * 64,
            decided_at="2026-08-08T23:00:00Z",
            issue_time=case.issue_time,
        )
        receipt = RealizedInterventionReceipt(
            decision_digest=decision.decision_digest,
            decision_id=decision.decision_id,
            case_id=decision.case_id,
            radar_id=decision.radar_id,
            intervention_type=decision.intervention_type,
            action_digest=decision.action_digest,
            input_plan_digest=decision.input_plan_digest,
            actual_input_before_digest=decision.actual_input_before_digest,
            actual_input_after_digest="a" * 64,
            actual_input_bundle_digest=case.input_bundle_digest,
            executor_key_id="executor-1",
            executor_trust_store_digest="f" * 64,
            executor_signature="e" * 64,
            applied_time="2026-08-08T23:30:00Z",
            receipt_time="2026-08-08T23:31:00Z",
            issue_time=decision.issue_time,
        )
        grid = SimpleNamespace(valid_times=planned_input.valid_times)
        data_identity = promotion_module.OperationalDataIdentity(
            radar_class="test",
            qc_pipeline_digest=planned_input.qc_pipeline_digest,
            observation_error_model_digest="4" * 64,
            background_model_digest="5" * 64,
            radar_product_digest=planned_input.radar_product_digest,
            background_cycle_rule_digest=(
                planned_input.background_cycle_rule_digest
            ),
            mask_policy_digest=planned_input.mask_policy_digest,
        )
        common = dict(
            grid_time_contract_digest="2" * 64,
            grid_time_contract=grid,
            input_bundle_digest=case.input_bundle_digest,
            input_plan_digest=case.input_plan_digest,
            config=SimpleNamespace(digest="3" * 64, interval_minutes=10),
            analysis_config_digest="4" * 64,
            operational_calibration_manifest_digest="5" * 64,
            operational_data_identity_json=data_identity.json,
            operational_data_identity_digest=data_identity.digest,
            prior_model_contract_digest=manifest.model_contract_digest,
            prior_feature_schema_digest=manifest.feature_schema_digest,
            prior_inference_algorithm_digest="8" * 64,
            prior_numerical_runtime_digest="9" * 64,
            prior_dependency="radar_dependent",
            input_plan_json=planned_input.json,
            input_plan_resolution_digest=promotion_module.json_digest(
                {
                    "contract": "forecast-input-plan-resolution-v1",
                    "input_plan_digest": case.input_plan_digest,
                    "input_bundle_digest": case.input_bundle_digest,
                }
            ),
        )
        candidate_app = SimpleNamespace(
            application_digest=case.candidate_prior_application_digest,
            inference_evidence=SimpleNamespace(
                evidence_digest=case.candidate_inference_evidence_digest,
                inference_algorithm_digest="8" * 64,
                numerical_runtime_digest="9" * 64,
                dependency="radar_dependent",
                input_bundle_digest=case.input_bundle_digest,
                input_frames_digest=promotion_module.tensor_digest(torch.zeros(3, 2, 2)),
                execution_contract_digest=manifest.candidate_prior_digest,
                neural_prior_digest=manifest.candidate_prior_digest,
                model_contract_digest=manifest.model_contract_digest,
                feature_schema_digest=manifest.feature_schema_digest,
                training_manifest_digest=(manifest.candidate_training_manifest_digest),
            ),
        )
        parent_app = SimpleNamespace(
            application_digest=case.parent_prior_application_digest,
            inference_evidence=SimpleNamespace(
                evidence_digest=case.parent_inference_evidence_digest,
                inference_algorithm_digest="8" * 64,
                numerical_runtime_digest="9" * 64,
                dependency="radar_dependent",
                input_bundle_digest=case.input_bundle_digest,
                input_frames_digest=promotion_module.tensor_digest(torch.zeros(3, 2, 2)),
                execution_contract_digest=manifest.parent_prior_digest,
                neural_prior_digest=manifest.parent_prior_digest,
                model_contract_digest=manifest.model_contract_digest,
                feature_schema_digest=manifest.feature_schema_digest,
                training_manifest_digest=manifest.parent_training_manifest_digest,
            ),
        )
        candidate_run = SimpleNamespace(
            **common,
            neural_prior_digest=manifest.candidate_prior_digest,
            prior_application_digest=candidate_app.application_digest,
            prior_inference_evidence_digest=(
                candidate_app.inference_evidence.evidence_digest
            ),
            prior_role="candidate",
            prior_training_manifest_digest=(
                manifest.candidate_training_manifest_digest
            ),
        )
        parent_run = SimpleNamespace(
            **common,
            neural_prior_digest=manifest.parent_prior_digest,
            prior_application_digest=parent_app.application_digest,
            prior_inference_evidence_digest=(
                parent_app.inference_evidence.evidence_digest
            ),
            prior_role="parent",
            prior_training_manifest_digest=manifest.parent_training_manifest_digest,
        )
        candidate = SimpleNamespace(
            run=candidate_run,
            state=object(),
            validate_issuance=Mock(),
        )
        parent = SimpleNamespace(
            run=parent_run,
            state=object(),
            validate_issuance=Mock(),
        )
        resolved = SimpleNamespace(
            content_digest=case.verification_bundle_digest,
            valid_times=("2026-08-09T01:00:00Z",),
            grid_contract_digest="2" * 64,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            valid_mask=torch.ones((6, 1, 1), dtype=torch.bool),
            frames_dbz=torch.zeros((6, 1, 1)),
        )
        verification = SimpleNamespace(valid_times=resolved.valid_times)
        config = SimpleNamespace(
            digest=case.metric_contract_digest,
            full_map_lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            metric_domain="issued",
        )
        candidate_runner = SimpleNamespace(
            reproduce=Mock(),
            inference_algorithm_digest="8" * 64,
            numerical_runtime_digest="9" * 64,
        )
        parent_runner = SimpleNamespace(
            reproduce=Mock(),
            inference_algorithm_digest="8" * 64,
            numerical_runtime_digest="9" * 64,
        )
        candidate_weights = torch.full((1, 1, 1), 0.5)
        parent_weights = torch.ones((1, 1, 1))
        with (
            patch.object(
                promotion_module,
                "_forecast_result_content_digest",
                side_effect=(
                    case.candidate_forecast_digest,
                    case.parent_forecast_digest,
                ),
            ),
            patch.object(
                promotion_module, "_resolve_verification", return_value=resolved
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
            ),
            patch.object(
                promotion_module,
                "_forecast_coverage",
                side_effect=(torch.tensor([0.9]), torch.tensor([1.0])),
            ),
        ):
            evaluation = promotion_module.RealizedInterventionEvaluation.from_forecasts(
                decision,
                receipt,
                manifest,
            plan,
                case_id=case.case_id,
                candidate_forecast=candidate,
                parent_forecast=parent,
                verification=verification,
                metric_config=config,
                candidate_prior_application=candidate_app,
                parent_prior_application=parent_app,
                candidate_prior_runner=candidate_runner,
                parent_prior_runner=parent_runner,
                input_frames_dbz=torch.zeros((3, 2, 2)),
            )
        self.assertAlmostEqual(float(evaluation.metric_change[0, 0]), -0.2)
        self.assertAlmostEqual(float(evaluation.end_to_end_metric_change[0, 0]), -0.25)
        candidate_runner.reproduce.assert_called_once()
        parent_runner.reproduce.assert_called_once()


if __name__ == "__main__":
    unittest.main()
