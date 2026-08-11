from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import advar.promotion as promotion_module
import advar.ledger as ledger_module
from advar.nowcast import _validate_input_plan_resolution
from advar import (
    EpisodeLedger,
    DeployedNeuralPriorPolicy,
    ForecastRunContract,
    NowcastConfig,
    NeuralPriorCandidateManifest,
    NeuralPriorHoldoutCase,
    NeuralPriorHoldoutPlan,
    NeuralPriorHoldoutPlanCase,
    NeuralPriorPromotionPolicy,
    NeuralPriorProbabilityContract,
    NeuralPriorRegimeClassifier,
    NeuralPriorStateContract,
    NeuralPriorStateCalibrationPlan,
    NeuralPriorStateCalibrationTarget,
    PriorUncertaintyTarget,
    PriorUncertaintyTargetPlan,
    PromotionMetricScale,
    ProspectiveInterventionDecision,
    RealizedInterventionReceipt,
    RealizedObservationIntervention,
    RadarGridTimeContract,
    VerificationBundle,
    compute_neural_prior_promotion,
    validate_neural_prior_candidate_manifest,
    validate_neural_prior_promotion,
    validate_neural_prior_promotion_applicability,
    verification_plan_digest,
    neural_prior_state_censor_policy_digest,
)
from advar.sensitivity import _LearningPolicyTrustStore


class _FixedRegimeClassifier(nn.Module):
    def __init__(self, regime_logits: tuple[float, ...], range_logits: tuple[float, ...]):
        super().__init__()
        self.register_buffer("regime_logits", torch.tensor(regime_logits))
        self.register_buffer("range_logits", torch.tensor(range_logits))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        retained = frames.sum() * 0.0
        return self.regime_logits + retained, self.range_logits + retained


class NeuralPriorPromotionTests(unittest.TestCase):
    @staticmethod
    def regime_labeler_key() -> Ed25519PrivateKey:
        return Ed25519PrivateKey.from_private_bytes(b"\x01" * 32)

    def state_contract(self) -> NeuralPriorStateContract:
        return NeuralPriorStateContract(
            state_product_digest="a" * 64,
            state_qc_pipeline_digest="9" * 64,
            state_mask_policy_digest="3" * 64,
            state_censor_policy_digest=neural_prior_state_censor_policy_digest(
                detection_limit_dbz=5.0,
                censor_temperature_dbz=1.0,
                censored_background_policy="floor",
                minimum_dbz=-10.0,
                maximum_dbz=70.0,
            ),
            support_threshold_dbz=5.0,
            minimum_state_dbz=-10.0,
            maximum_state_dbz=70.0,
            minimum_state_std_dbz=0.1,
            maximum_state_std_dbz=20.0,
        )

    def probability_contract(self) -> NeuralPriorProbabilityContract:
        return NeuralPriorProbabilityContract(
            support_threshold_dbz=5.0,
            support_product_digest="6" * 64,
            qc_pipeline_digest="9" * 64,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
        )

    def test_physical_applicability_contract_versions_are_new_generations(
        self,
    ) -> None:
        plan = self.plan()
        manifest = self.manifest()
        evaluation = self.evaluation(1, -1.0)
        policy = self.policy()

        self.assertEqual(plan.contract, "neural-prior-holdout-plan-v12")
        self.assertTrue(
            all(
                item.contract == "neural-prior-range-band-contract-v2"
                for item in plan.range_band_contracts
            )
        )
        self.assertEqual(
            manifest.contract,
            "neural-prior-candidate-manifest-v10",
        )
        self.assertEqual(
            evaluation.contract,
            "prior-holdout-evaluation-v14",
        )
        self.assertTrue(
            all(
                band.contract == "neural-prior-range-band-evaluation-v4"
                for band in evaluation.range_band_evaluations
            )
        )
        self.assertEqual(policy.contract, "neural-prior-promotion-policy-v20")

    def test_immediately_previous_contracts_remain_audit_only(self) -> None:
        plan_payload = promotion_module._holdout_plan_payload(self.plan())
        plan_payload["contract"] = "neural-prior-holdout-plan-v10"
        plan_payload.pop("physical_event_catalog_plan")
        plan_digest = promotion_module.json_digest(plan_payload)
        stored_plan = plan_payload | {"plan_digest": plan_digest}
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest, plan_id, plan_json, policy_digest, "
                    "trust_store_digest, registered_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan_digest,
                        "holdout-plan",
                        json.dumps(stored_plan, sort_keys=True),
                        "6" * 64,
                        "7" * 64,
                        "2026-08-07T00:00:00Z",
                        "2026-08-07T00:00:00Z",
                    ),
                )
            loaded_plan = ledger.load_neural_prior_holdout_plan(plan_digest)
        self.assertIsInstance(
            loaded_plan,
            promotion_module.LegacyNeuralPriorHoldoutPlanV10Audit,
        )

        manifest_payload = promotion_module._candidate_manifest_payload(
            self.manifest()
        )
        manifest_payload["contract"] = "neural-prior-candidate-manifest-v8"
        for name in (
            "physical_event_catalog_result",
            "candidate_scoring_started_at",
            "training_physical_event_catalog_plan",
            "training_physical_event_catalog_result",
            "candidate_training_started_at",
        ):
            manifest_payload.pop(name)
        manifest_digest = promotion_module.json_digest(manifest_payload)
        loaded_manifest = ledger_module._decode_candidate_manifest(
            json.dumps(
                manifest_payload | {"manifest_digest": manifest_digest},
                sort_keys=True,
            ),
            expected_digest=manifest_digest,
        )
        self.assertIsInstance(
            loaded_manifest,
            promotion_module.LegacyNeuralPriorCandidateManifestAuditV8,
        )

        promotion_payload = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )._payload()
        promotion_payload["contract"] = "neural-prior-promotion-evidence-v13"
        promotion_payload.pop("range_metric_cell_bounds")
        promotion_payload.pop("rate_inference_method")
        promotion_digest = promotion_module.json_digest(promotion_payload)
        promotion_audit = (
            promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV13(
                promotion_evidence_digest=promotion_digest,
                payload_json=json.dumps(
                    promotion_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        self.assertEqual(
            promotion_audit.promotion_evidence_digest,
            promotion_digest,
        )
        self.assertFalse(hasattr(promotion_audit, "eligible"))

    def test_pr102_contracts_remain_audit_only(self) -> None:
        plan_payload = promotion_module._holdout_plan_payload(self.plan())
        plan_payload["contract"] = "neural-prior-holdout-plan-v11"
        catalog_plan = plan_payload["physical_event_catalog_plan"]
        catalog_plan["contract"] = "physical-event-catalog-plan-v1"
        for name in (
            "spatial_reference_digest",
            "maximum_association_time_gap_minutes",
            "minimum_association_spatial_iou",
            "scheduler_id",
            "scheduler_public_key_hex",
        ):
            catalog_plan.pop(name)
        plan_digest = promotion_module.json_digest(plan_payload)
        stored_plan = plan_payload | {"plan_digest": plan_digest}
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest, plan_id, plan_json, policy_digest, "
                    "trust_store_digest, registered_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan_digest,
                        "holdout-plan",
                        json.dumps(
                            stored_plan,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "6" * 64,
                        "7" * 64,
                        "2026-08-07T00:00:00Z",
                        "2026-08-07T00:00:00Z",
                    ),
                )
            loaded_plan = ledger.load_neural_prior_holdout_plan(plan_digest)
        self.assertIsInstance(
            loaded_plan,
            promotion_module.LegacyNeuralPriorHoldoutPlanV11Audit,
        )

        manifest_payload = promotion_module._candidate_manifest_payload(
            self.manifest()
        )
        manifest_payload["contract"] = "neural-prior-candidate-manifest-v9"
        manifest_payload.pop("candidate_training_start_receipt")
        manifest_payload.pop("candidate_scoring_start_receipt")
        manifest_digest = promotion_module.json_digest(manifest_payload)
        loaded_manifest = ledger_module._decode_candidate_manifest(
            json.dumps(
                manifest_payload | {"manifest_digest": manifest_digest},
                sort_keys=True,
                separators=(",", ":"),
            ),
            expected_digest=manifest_digest,
        )
        self.assertIsInstance(
            loaded_manifest,
            promotion_module.LegacyNeuralPriorCandidateManifestAuditV9,
        )

        promotion_payload = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )._payload()
        promotion_payload["contract"] = "neural-prior-promotion-evidence-v14"
        promotion_payload.pop("range_metric_end_to_end_cell_bounds")
        promotion_digest = promotion_module.json_digest(promotion_payload)
        promotion_audit = (
            promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV14(
                promotion_evidence_digest=promotion_digest,
                payload_json=json.dumps(
                    promotion_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        self.assertEqual(
            promotion_audit.promotion_evidence_digest,
            promotion_digest,
        )
        self.assertFalse(hasattr(promotion_audit, "eligible"))

    def test_previous_physical_contracts_load_as_audit_only(self) -> None:
        plan_payload = promotion_module._holdout_plan_payload(self.plan())
        plan_payload["contract"] = "neural-prior-holdout-plan-v9"
        plan_payload.pop("range_geometry_contracts")
        for range_contract in plan_payload["range_band_contracts"]:
            range_contract["contract"] = "neural-prior-range-band-contract-v1"
            range_contract.pop("range_geometry_contract_digest")
        for classifier in plan_payload["regime_classifier_manifests"]:
            classifier["contract"] = "neural-prior-regime-classifier-manifest-v2"
            classifier.pop("training_physical_event_digests")
        plan_digest = promotion_module.json_digest(plan_payload)
        stored_plan = plan_payload | {"plan_digest": plan_digest}

        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest, plan_id, plan_json, policy_digest, "
                    "trust_store_digest, registered_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan_digest,
                        "holdout-plan",
                        json.dumps(
                            stored_plan,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "6" * 64,
                        "7" * 64,
                        "2026-08-07T00:00:00Z",
                        "2026-08-07T00:00:00Z",
                    ),
                )
            loaded_plan = ledger.load_neural_prior_holdout_plan(plan_digest)

        self.assertIsInstance(
            loaded_plan,
            promotion_module.LegacyNeuralPriorHoldoutPlanV9Audit,
        )

        manifest_payload = promotion_module._candidate_manifest_payload(
            self.manifest()
        )
        manifest_payload["contract"] = "neural-prior-candidate-manifest-v7"
        manifest_payload.pop("physical_event_catalog_evidences")
        for case in manifest_payload["holdout_cases"]:
            case.pop("physical_event_digest")
        manifest_digest = promotion_module.json_digest(manifest_payload)
        stored_manifest = manifest_payload | {
            "manifest_digest": manifest_digest
        }
        loaded_manifest = ledger_module._decode_candidate_manifest(
            json.dumps(stored_manifest, sort_keys=True, separators=(",", ":")),
            expected_digest=manifest_digest,
        )
        self.assertIsInstance(
            loaded_manifest,
            promotion_module.LegacyNeuralPriorCandidateManifestAuditV7,
        )

    def deployment_range_geometry(
        self,
        *,
        labels: tuple[str, str],
        include_second_band: bool,
    ):
        grid_x_m = torch.tensor(
            [[0.0, 20_000.0], [40_000.0, 80_000.0]]
            if include_second_band
            else [[0.0, 20_000.0], [10_000.0, 25_000.0]]
        )
        grid_y_m = torch.zeros_like(grid_x_m)
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="a" * 64,
            radar_site_location_digest="a" * 64,
            grid_contract_digest="2" * 64,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=labels,
            radial_distance_edges_m=(0.0, 30_000.0, 100_000.0),
            horizontal_range_rule_digest="b" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(grid_x_m),
            grid_y_m_digest=promotion_module.tensor_digest(grid_y_m),
        )
        return geometry, promotion_module.resolve_range_geometry(
            geometry,
            grid_x_m=grid_x_m,
            grid_y_m=grid_y_m,
        )

    def verification_plan(self, valid_time: str) -> str:
        return verification_plan_digest(
            valid_times=(valid_time,),
            grid_contract_digest="2" * 64,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
        )

    def holdout_range_geometry(self, index: int):
        grid_x_m = torch.zeros((2, 2))
        grid_y_m = torch.zeros_like(grid_x_m)
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest=("a" if index == 1 else "b") * 64,
            radar_site_location_digest=("a" if index == 1 else "b") * 64,
            grid_contract_digest="2" * 64,
            radar_x_m=0.0 if index == 1 else 50_000.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range", "far_range"),
            radial_distance_edges_m=(0.0, 30_000.0, 100_000.0),
            horizontal_range_rule_digest="c" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(grid_x_m),
            grid_y_m_digest=promotion_module.tensor_digest(grid_y_m),
        )
        return geometry, promotion_module.resolve_range_geometry(
            geometry,
            grid_x_m=grid_x_m,
            grid_y_m=grid_y_m,
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
                observation_valid_time=issue,
                input_available_time=issue,
                decision_deadline=(
                    datetime.fromisoformat(issue.replace("Z", "+00:00"))
                    + timedelta(minutes=2)
                ).isoformat(),
                publication_time=(
                    datetime.fromisoformat(issue.replace("Z", "+00:00"))
                    + timedelta(minutes=5)
                ).isoformat(),
            )
            for index, issue in enumerate(
                ("2026-08-09T00:00:00Z", "2026-08-10T00:00:00Z"),
                start=1,
            )
        )
        target_plans = tuple(
            PriorUncertaintyTargetPlan(
                plan_id=f"uncertainty-{index}",
                target_kind="independent_sensor",
                source_identity_digest="6" * 64,
                qc_pipeline_digest="9" * 64,
                mask_policy_digest="3" * 64,
                censor_policy_digest=self.state_contract().state_censor_policy_digest,
                floor_representation_contract_digest="e" * 64,
                grid_contract_digest="2" * 64,
                feature_exclusion_contract_digest="5" * 64,
                independence_evidence_digest="8" * 64,
                target_valid_time=valid_time,
                prior_probability_contract_digest=(
                    self.probability_contract().contract_digest
                ),
            )
            for index, valid_time in enumerate(
                ("2026-08-09T00:00:00Z", "2026-08-10T00:00:00Z"),
                start=1,
            )
        )
        state_target_plans = tuple(
            NeuralPriorStateCalibrationPlan(
                plan_id=f"state-calibration-{index}",
                target_kind="withheld_target_mask",
                source_identity_digest="a" * 64,
                qc_pipeline_digest="9" * 64,
                mask_policy_digest="3" * 64,
                censor_policy_digest=self.state_contract().state_censor_policy_digest,
                floor_representation_contract_digest="e" * 64,
                grid_contract_digest="2" * 64,
                feature_exclusion_contract_digest="5" * 64,
                independence_evidence_digest="8" * 64,
                target_valid_time=valid_time,
                state_contract_digest=self.state_contract().contract_digest,
                support_threshold_dbz=5.0,
            )
            for index, valid_time in enumerate(
                ("2026-08-09T00:00:00Z", "2026-08-10T00:00:00Z"),
                start=1,
            )
        )
        range_labels = ("near_range", "far_range")
        range_geometries = tuple(
            self.holdout_range_geometry(index)[0] for index in (1, 2)
        )
        range_partitions = tuple(
            self.holdout_range_geometry(index)[1] for index in (1, 2)
        )
        range_contracts = tuple(
            promotion_module.RangeBandContract(
                case_id=f"case-{index}",
                range_regime_labels=range_labels,
                range_band_mask_digests=range_partitions[index - 1].range_band_mask_digests,
                reference_active_range_regimes=(
                    "near_range" if index == 1 else "far_range",
                ),
                grid_contract_digest="2" * 64,
                range_geometry_contract_digest=(
                    range_geometries[index - 1].contract_digest
                ),
            )
            for index in (1, 2)
        )
        classifier_manifest = promotion_module.RegimeClassifierManifest(
            classifier_digest="e" * 64,
            training_dataset_digest="4" * 64,
            training_case_ids=("classifier-training-case",),
            training_input_bundle_digests=("9" * 64,),
            training_full_analysis_input_digests=("8" * 64,),
            training_physical_event_digests=("4" * 64,),
            training_storm_ids=("classifier-training-storm",),
            training_days=("2026-06-01",),
            training_radar_ids=("classifier-radar",),
            training_grid_contract_digests=("6" * 64,),
            training_time_windows=((
                "2026-06-01T00:00:00Z",
                "2026-06-01T01:00:00Z",
            ),),
            training_algorithm_digest="5" * 64,
            numerical_runtime_digest=(
                promotion_module.numerical_runtime_identity_digest("cpu")
            ),
            reference_label_contract_digest="7" * 64,
            signed_training_member_manifest_digest="5" * 64,
        )
        reference_plans = tuple(
            promotion_module.RegimeReferencePlan(
                case_id=f"case-{index}",
                labeler_id="independent-weather-labeler",
                labeler_public_key_hex=(
                    promotion_module.regime_reference_public_key_hex(
                        self.regime_labeler_key()
                    )
                ),
                source_contract_digest="7" * 64,
                labeling_valid_time=valid_time,
                adjudication_policy_digest="6" * 64,
            )
            for index, valid_time in enumerate(
                ("2026-08-09T01:00:00Z", "2026-08-10T01:00:00Z"),
                start=1,
            )
        )
        return NeuralPriorHoldoutPlan(
            plan_id="holdout-plan",
            parent_prior_digest="d" * 64,
            candidate_family_digests=("c" * 64,),
            cases=(
                NeuralPriorHoldoutPlanCase(
                    case_id="case-1",
                    storm_id="pending",
                    day="2026-08-08",
                    radar_id="radar-1",
                    regime="pending",
                    range_regime="near_range",
                    input_plan_digest=input_plans[0].plan_digest,
                    verification_plan_digest=self.verification_plan(
                        "2026-08-09T01:00:00Z"
                    ),
                    metric_contract_digest="b" * 64,
                    uncertainty_target_plan_digest=target_plans[0].plan_digest,
                    state_calibration_target_plan_digest=(
                        state_target_plans[0].plan_digest
                    ),
                    range_band_contract_digest=range_contracts[0].contract_digest,
                    reference_active_range_regimes=("near_range",),
                    regime_reference_plan_digest=reference_plans[0].plan_digest,
                    issue_time="2026-08-09T00:00:00Z",
                ),
                NeuralPriorHoldoutPlanCase(
                    case_id="case-2",
                    storm_id="pending",
                    day="2026-08-09",
                    radar_id="radar-1",
                    regime="pending",
                    range_regime="far_range",
                    input_plan_digest=input_plans[1].plan_digest,
                    verification_plan_digest=self.verification_plan(
                        "2026-08-10T01:00:00Z"
                    ),
                    metric_contract_digest="b" * 64,
                    uncertainty_target_plan_digest=target_plans[1].plan_digest,
                    state_calibration_target_plan_digest=(
                        state_target_plans[1].plan_digest
                    ),
                    range_band_contract_digest=range_contracts[1].contract_digest,
                    reference_active_range_regimes=("far_range",),
                    regime_reference_plan_digest=reference_plans[1].plan_digest,
                    issue_time="2026-08-10T00:00:00Z",
                ),
            ),
            input_plans=input_plans,
            uncertainty_target_plans=target_plans,
            state_calibration_target_plans=state_target_plans,
            range_band_contracts=range_contracts,
            range_geometry_contracts=range_geometries,
            regime_reference_plans=reference_plans,
            regime_classifier_manifests=(classifier_manifest,),
            reference_label_contract_digest="7" * 64,
            physical_event_catalog_plan=self.event_catalog_plan(),
            registered_at="2026-08-07T00:00:00Z",
        )

    def completed_case(self, index: int) -> NeuralPriorHoldoutCase:
        planned = self.plan().cases[index - 1]
        uncertainty_target = self.uncertainty_target(index)
        state_target = self.state_target(index)
        full_digest = ("1" if index == 1 else "2") * 64
        return NeuralPriorHoldoutCase(
            case_id=planned.case_id,
            planned_storm_id=planned.storm_id,
            storm_id=f"storm-{index}",
            physical_event_digest=(
                self.event_catalog(index).physical_event_identity_digest
            ),
            day=planned.day,
            radar_id=planned.radar_id,
            planned_regime=planned.regime,
            regime="convective" if index == 1 else "stratiform",
            range_regime=planned.range_regime,
            input_plan_digest=planned.input_plan_digest,
            input_plan_resolution_digest=(
                promotion_module._forecast_input_plan_resolution_digest(
                    input_plan_digest=planned.input_plan_digest,
                    full_analysis_input_digest=full_digest,
                )
            ),
            input_bundle_digest=("e" if index == 1 else "f") * 64,
            full_analysis_input_digest=full_digest,
            fixed_input_context_digest=("a" if index == 1 else "b") * 64,
            observation_quality_weight_digest=(
                "c" if index == 1 else "d"
            ) * 64,
            observation_std_dbz_digest=("4" if index == 1 else "5") * 64,
            verification_plan_digest=planned.verification_plan_digest,
            verification_bundle_digest="a" * 64,
            metric_contract_digest=planned.metric_contract_digest,
            uncertainty_target_plan_digest=(
                planned.uncertainty_target_plan_digest
            ),
            uncertainty_target_digest=uncertainty_target.target_digest,
            state_calibration_target_plan_digest=(
                planned.state_calibration_target_plan_digest
            ),
            state_calibration_target_digest=state_target.target_digest,
            prior_state_contract_digest=self.state_contract().contract_digest,
            issue_time=planned.issue_time,
            candidate_forecast_digest=("6" if index == 1 else "8") * 64,
            parent_forecast_digest=("7" if index == 1 else "9") * 64,
            candidate_prior_application_digest=("3" if index == 1 else "4") * 64,
            parent_prior_application_digest=("5" if index == 1 else "6") * 64,
            candidate_inference_evidence_digest=("7" if index == 1 else "8") * 64,
            parent_inference_evidence_digest=("9" if index == 1 else "0") * 64,
            prior_probability_contract_digest=(
                self.probability_contract().contract_digest
            ),
            range_band_contract_digest=planned.range_band_contract_digest,
            regime_reference_plan_digest=planned.regime_reference_plan_digest,
            regime_reference_evidence_digest=(
                self.reference_evidence(index).evidence_digest
            ),
            reference_active_range_regimes=(
                planned.reference_active_range_regimes
            ),
        )

    def event_catalog(self, index: int):
        issue = "2026-08-09T00:00:00Z" if index == 1 else "2026-08-10T00:00:00Z"
        end = "2026-08-09T02:00:00Z" if index == 1 else "2026-08-10T02:00:00Z"
        return promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id=f"physical-event-{index}",
            member_case_ids=(f"case-{index}",),
            member_full_analysis_input_digests=(
                ("1" if index == 1 else "2") * 64,
            ),
            start_time=issue,
            end_time=end,
            spatial_envelope_xy_m=(0.0, 0.0, 100_000.0, 100_000.0),
            participating_radar_ids=("radar-1",),
            association_algorithm_digest="3" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_private_key=self.regime_labeler_key(),
        )

    def event_spatial_evidence(
        self,
        event,
        *,
        case_id: str,
        full_analysis_input_digest: str,
        source_object_evidence_digest: str,
    ):
        input_available_time = (
            "2026-07-01T01:00:00Z"
            if case_id == "training-case"
            else "2026-08-09T00:00:00Z"
            if case_id == "case-1"
            else "2026-08-10T00:00:00Z"
        )
        return promotion_module.PhysicalEventCaseSpatialEvidence(
            case_id=case_id,
            full_analysis_input_digest=full_analysis_input_digest,
            physical_event_identity_digest=(
                event.physical_event_identity_digest
            ),
            observed_spatial_envelope_xy_m=(
                10_000.0,
                10_000.0,
                90_000.0,
                90_000.0,
            ),
            event_spatial_envelope_xy_m=event.spatial_envelope_xy_m,
            spatial_membership_rule_digest="4" * 64,
            source_object_evidence_digest=source_object_evidence_digest,
            input_available_time=input_available_time,
            spatial_reference_digest="7" * 64,
        )

    def training_event_catalog(self):
        return promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="training-physical-event",
            member_case_ids=("training-case",),
            member_full_analysis_input_digests=("8" * 64,),
            start_time="2026-07-01T00:00:00Z",
            end_time="2026-07-01T02:00:00Z",
            spatial_envelope_xy_m=(0.0, 0.0, 100_000.0, 100_000.0),
            participating_radar_ids=("radar-1",),
            association_algorithm_digest="3" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_private_key=self.regime_labeler_key(),
        )

    def training_event_catalog_plan(self):
        return promotion_module.PhysicalEventCatalogPlan(
            holdout_case_ids=("training-case",),
            association_algorithm_digest="3" * 64,
            spatial_membership_rule_digest="4" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.regime_labeler_key()
                )
            ),
            catalog_completion_deadline="2026-07-01T03:00:00Z",
            spatial_reference_digest="7" * 64,
        )

    def training_event_catalog_result(self):
        event = self.training_event_catalog()
        return promotion_module.PhysicalEventCatalogResult.from_plan(
            self.training_event_catalog_plan(),
            event_evidences=(event,),
            case_spatial_membership_evidences=(
                self.event_spatial_evidence(
                    event,
                    case_id="training-case",
                    full_analysis_input_digest="8" * 64,
                    source_object_evidence_digest="c" * 64,
                ),
            ),
            cataloged_at="2026-07-01T02:30:00Z",
            adjudicator_private_key=self.regime_labeler_key(),
        )

    def event_catalog_plan(self):
        return promotion_module.PhysicalEventCatalogPlan(
            holdout_case_ids=("case-1", "case-2"),
            association_algorithm_digest="3" * 64,
            spatial_membership_rule_digest="4" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.regime_labeler_key()
                )
            ),
            catalog_completion_deadline="2026-08-11T00:00:00Z",
            spatial_reference_digest="7" * 64,
        )

    def event_catalog_result(self):
        first = self.event_catalog(1)
        second = self.event_catalog(2)
        return promotion_module.PhysicalEventCatalogResult.from_plan(
            self.event_catalog_plan(),
            event_evidences=(first, second),
            case_spatial_membership_evidences=(
                self.event_spatial_evidence(
                    first,
                    case_id="case-1",
                    full_analysis_input_digest="1" * 64,
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    second,
                    case_id="case-2",
                    full_analysis_input_digest="2" * 64,
                    source_object_evidence_digest="b" * 64,
                ),
            ),
            cataloged_at="2026-08-10T03:00:00Z",
            adjudicator_private_key=self.regime_labeler_key(),
        )

    def training_start_receipt(self):
        return promotion_module.TrustedProcessStartReceipt.from_plan(
            self.training_event_catalog_plan(),
            catalog_result_digest=self.training_event_catalog_result().result_digest,
            process_kind="candidate_training",
            subject_digests=("1" * 64, "2" * 64),
            process_algorithm_digest="3" * 64,
            job_id="candidate-training-job",
            started_at="2026-07-02T00:00:00Z",
            scheduler_private_key=self.regime_labeler_key(),
        )

    def scoring_start_receipt(self):
        return self.scoring_start_receipt_for(
            self.event_catalog_plan(),
            self.event_catalog_result(),
        )

    def scoring_start_receipt_for(
        self,
        plan,
        result,
        *,
        private_key=None,
        subject_digests=("c" * 64,),
    ):
        return promotion_module.TrustedProcessStartReceipt.from_plan(
            plan,
            catalog_result_digest=result.result_digest,
            process_kind="candidate_scoring",
            subject_digests=subject_digests,
            process_algorithm_digest="9" * 64,
            job_id="candidate-scoring-job",
            started_at="2026-08-12T01:00:00Z",
            scheduler_private_key=(
                self.regime_labeler_key()
                if private_key is None
                else private_key
            ),
        )

    def reference_evidence(self, index: int):
        plan = self.plan()
        case = plan.cases[index - 1]
        reference_plan = plan.regime_reference_plans[index - 1]
        return promotion_module.RegimeReferenceEvidence.from_plan(
            reference_plan,
            full_analysis_input_digest=("1" if index == 1 else "2") * 64,
            verification_bundle_digest="a" * 64,
            observed_regime="convective" if index == 1 else "stratiform",
            observed_storm_id=f"storm-{index}",
            labeled_at=reference_plan.labeling_valid_time,
            labeler_private_key=self.regime_labeler_key(),
        )

    def state_target(self, index: int) -> NeuralPriorStateCalibrationTarget:
        plan = self.plan()
        target_plan = plan.state_calibration_target_plans[index - 1]
        verification = VerificationBundle(
            frames_dbz=torch.tensor([[[10.0, 1.0], [10.0, 1.0]]]),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest=target_plan.source_identity_digest,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
            mask_policy_digest=target_plan.mask_policy_digest,
            censor_policy_digest=target_plan.censor_policy_digest,
            reflectivity_resolution_dbz=(
                target_plan.reflectivity_resolution_dbz
            ),
            quantization_origin_dbz=target_plan.quantization_origin_dbz,
            threshold_bin_convention=target_plan.threshold_bin_convention,
            floor_representation_contract_digest=(
                target_plan.floor_representation_contract_digest
            ),
            contract="radar-verification-bundle-v3",
        )
        return NeuralPriorStateCalibrationTarget.from_verification_bundle(
            plan=target_plan,
            verification=verification,
        )

    def uncertainty_target(self, index: int) -> PriorUncertaintyTarget:
        plan = self.plan()
        target_plan = plan.uncertainty_target_plans[index - 1]
        verification = VerificationBundle(
            frames_dbz=torch.tensor([[[10.0, 1.0], [10.0, 1.0]]]),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest=target_plan.source_identity_digest,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
            mask_policy_digest=target_plan.mask_policy_digest,
            censor_policy_digest=target_plan.censor_policy_digest,
            reflectivity_resolution_dbz=(
                target_plan.reflectivity_resolution_dbz
            ),
            quantization_origin_dbz=target_plan.quantization_origin_dbz,
            threshold_bin_convention=target_plan.threshold_bin_convention,
            floor_representation_contract_digest=(
                target_plan.floor_representation_contract_digest
            ),
            contract="radar-verification-bundle-v3",
        )
        return PriorUncertaintyTarget.from_verification_bundle(
            plan=target_plan,
            verification=verification,
        )

    def test_uncertainty_target_requires_its_planned_verification_source(self) -> None:
        target_plan = self.plan().uncertainty_target_plans[0]
        wrong_source = VerificationBundle(
            frames_dbz=torch.ones((1, 2, 2)),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest="f" * 64,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            PriorUncertaintyTarget.from_verification_bundle(
                plan=target_plan,
                verification=wrong_source,
            )
        legacy_measurement_source = VerificationBundle(
            frames_dbz=torch.ones((1, 2, 2)),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest=target_plan.source_identity_digest,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            PriorUncertaintyTarget.from_verification_bundle(
                plan=target_plan,
                verification=legacy_measurement_source,
            )
        self.assertFalse(hasattr(PriorUncertaintyTarget, "from_tensors"))

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
            training_full_analysis_input_digests=("8" * 64,),
            training_physical_event_digests=(
                self.training_event_catalog().physical_event_identity_digest,
            ),
            training_physical_event_catalog_plan=(
                self.training_event_catalog_plan()
            ),
            training_physical_event_catalog_result=(
                self.training_event_catalog_result()
            ),
            candidate_training_started_at="2026-07-02T00:00:00Z",
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
            regime_reference_evidences=(
                self.reference_evidence(1),
                self.reference_evidence(2),
            ),
            physical_event_catalog_evidences=(
                self.event_catalog(1),
                self.event_catalog(2),
            ),
            physical_event_catalog_result=self.event_catalog_result(),
            candidate_scoring_started_at="2026-08-12T01:00:00Z",
            holdout_cases=(self.completed_case(1), self.completed_case(2)),
            candidate_training_start_receipt=self.training_start_receipt(),
            candidate_scoring_start_receipt=self.scoring_start_receipt(),
        )

    def evaluation(
        self,
        index: int,
        change: float,
        *,
        end_to_end: float | None = None,
        candidate_issuance: float = 0.0,
        prior_residual_mean_abs: float = 0.5,
        prior_underdispersion_fraction: float = 0.0,
        prior_sample_count: int = 16,
        prior_candidate_valid_fraction: float = 1.0,
        prior_parent_valid_fraction: float = 1.0,
        prior_candidate_valid_area_km2: float = 4.0,
        prior_echo_intensity_nll: float = 0.5,
        parent_prior_echo_intensity_nll: float = 0.5,
        prior_support_brier_score: float = 0.05,
        parent_prior_support_brier_score: float = 0.05,
        prior_echo_support_miss_score: float = 0.05,
        parent_prior_echo_support_miss_score: float = 0.05,
        prior_echo_object_miss_score: float = 0.05,
        parent_prior_echo_object_miss_score: float = 0.05,
        prior_clear_sky_false_echo_score: float = 0.05,
        parent_prior_clear_sky_false_echo_score: float = 0.05,
        parent_prior_underdispersion_fraction: float | None = None,
        state_candidate_gaussian_nll: float = 0.5,
        state_parent_gaussian_nll: float = 0.5,
        state_candidate_support_brier_score: float = 0.05,
        state_parent_support_brier_score: float = 0.05,
        state_candidate_false_support_score: float = 0.05,
        state_parent_false_support_score: float = 0.05,
        echo_available: bool = True,
        clear_available: bool = True,
        regime_classifier_digest: str = "e" * 64,
        classified_regime: str | None = None,
        classified_range_regimes: tuple[str, ...] | None = None,
        classifier_regime_confidence: float = 1.0,
        classifier_range_confidence: float = 1.0,
        classifier_regime_entropy: float = 0.0,
        classifier_is_ood: bool = False,
        classifier_reference_agreement: bool = True,
        range_change: float | None = None,
        range_end_to_end_change: float | None = None,
        secondary_range_change: float | None = None,
        range_component_sample_count: int = 8,
        range_valid_area_km2: float = 1.0,
        range_object_count: int = 1,
        range_candidate_support_brier_score: float | None = None,
        range_parent_support_brier_score: float | None = None,
        physical_event_digest: str | None = None,
    ) -> promotion_module.PriorHoldoutEvaluation:
        manifest = self.manifest()
        case = manifest.holdout_cases[index - 1]
        plan = self.plan()
        classifier_manifest = plan.regime_classifier_manifests[0]
        reference_ranges = case.reference_active_range_regimes
        predicted_ranges = (
            (case.range_regime,)
            if classified_range_regimes is None
            else classified_range_regimes
        )
        reference_set = set(reference_ranges)
        predicted_set = set(predicted_ranges)
        intersection = len(reference_set & predicted_set)
        range_precision = (
            1.0 if not predicted_set and not reference_set else
            intersection / len(predicted_set) if predicted_set else 0.0
        )
        range_recall = (
            1.0 if not reference_set else intersection / len(reference_set)
        )
        false_active_fraction = (
            len(predicted_set - reference_set) / len(predicted_set)
            if predicted_set
            else 0.0
        )
        echo_count = (
            prior_sample_count
            if echo_available and not clear_available
            else prior_sample_count // 2
            if echo_available
            else 0
        )
        clear_count = (
            prior_sample_count
            if clear_available and not echo_available
            else prior_sample_count - prior_sample_count // 2
            if clear_available
            else 0
        )
        range_contract = next(
            item
            for item in plan.range_band_contracts
            if item.contract_digest == case.range_band_contract_digest
        )
        range_component_names = tuple(
            name
            for name in promotion_module._UNCERTAINTY_COMPONENT_NAMES
            if (
                echo_available
                or name not in (
                    "intensity",
                    "pit_residual",
                    "echo_miss",
                    "object_miss",
                    "underdispersion",
                )
            )
            and (clear_available or name != "clear")
        )
        candidate_component_values = {
            "intensity": prior_echo_intensity_nll,
            "pit_residual": prior_residual_mean_abs,
            "support": prior_support_brier_score,
            "echo_miss": prior_echo_support_miss_score,
            "object_miss": prior_echo_object_miss_score,
            "clear": prior_clear_sky_false_echo_score,
            "underdispersion": prior_underdispersion_fraction,
            "state_nll": state_candidate_gaussian_nll,
            "state_pit_residual": 0.5,
            "state_underdispersion": 0.0,
            "state_support": state_candidate_support_brier_score,
            "state_echo_miss": 0.05,
            "state_object_miss": 0.05,
            "state_false_support": state_candidate_false_support_score,
            "state_valid": 0.05,
        }
        parent_component_values = {
            "intensity": parent_prior_echo_intensity_nll,
            "pit_residual": prior_residual_mean_abs,
            "support": parent_prior_support_brier_score,
            "echo_miss": parent_prior_echo_support_miss_score,
            "object_miss": parent_prior_echo_object_miss_score,
            "clear": parent_prior_clear_sky_false_echo_score,
            "underdispersion": (
                prior_underdispersion_fraction
                if parent_prior_underdispersion_fraction is None
                else parent_prior_underdispersion_fraction
            ),
            "state_nll": state_parent_gaussian_nll,
            "state_pit_residual": 0.5,
            "state_underdispersion": 0.0,
            "state_support": state_parent_support_brier_score,
            "state_echo_miss": 0.05,
            "state_object_miss": 0.05,
            "state_false_support": state_parent_false_support_score,
            "state_valid": 0.05,
        }
        if range_candidate_support_brier_score is not None:
            candidate_component_values["support"] = (
                range_candidate_support_brier_score
            )
        if range_parent_support_brier_score is not None:
            parent_component_values["support"] = (
                range_parent_support_brier_score
            )
        range_candidate_components = tuple(
            (name, candidate_component_values[name])
            for name in range_component_names
        )
        range_parent_components = tuple(
            (name, parent_component_values[name])
            for name in range_component_names
        )
        range_components = tuple(
            (
                name,
                candidate_component_values[name]
                - parent_component_values[name],
            )
            for name in range_component_names
        )
        range_component_counts = tuple(
            (
                name,
                range_object_count
                if name in ("object_miss", "state_object_miss")
                else range_component_sample_count
                * ((1 if echo_available else 0) + (1 if clear_available else 0))
                if name == "support"
                else 2 * range_component_sample_count
                if name
                in (
                    "state_nll",
                    "state_pit_residual",
                    "state_underdispersion",
                    "state_support",
                    "state_valid",
                )
                else range_component_sample_count,
            )
            for name in range_component_names
        )
        band_change = change if range_change is None else range_change
        metric_names = (
            ("log_echo_mse", "soft_fss_error_35")
            if secondary_range_change is not None
            else ("log_echo_mse",)
        )
        band_metric_change = torch.tensor(
            [[band_change]]
            if secondary_range_change is None
            else [[band_change, secondary_range_change]],
            dtype=torch.float64,
        )
        band_end_to_end_metric_change = (
            band_metric_change.clone()
            if range_end_to_end_change is None
            else torch.full_like(
                band_metric_change,
                range_end_to_end_change,
            )
        )
        range_evaluations = tuple(
            promotion_module.RangeBandEvaluation(
                range_regime=range_regime,
                range_band_mask_digest=range_contract.mask_digest(range_regime),
                range_geometry_contract_digest=(
                    range_contract.range_geometry_contract_digest
                ),
                metric_change=band_metric_change,
                end_to_end_metric_change=band_end_to_end_metric_change,
                metric_available=torch.ones_like(
                    band_metric_change, dtype=torch.bool
                ),
                candidate_uncertainty_component_scores=(
                    range_candidate_components
                ),
                parent_uncertainty_component_scores=range_parent_components,
                uncertainty_component_differences=range_components,
                uncertainty_component_sample_counts=range_component_counts,
                evaluated_area_km2=1000.0,
                metric_valid_area_km2_by_lead=(range_valid_area_km2,),
                metric_valid_area_km2=torch.full_like(
                    band_metric_change, range_valid_area_km2
                ),
                probability_valid_area_km2=range_valid_area_km2,
                state_valid_area_km2=range_valid_area_km2,
                echo_pixel_count=(
                    range_component_sample_count if echo_available else 0
                ),
                clear_pixel_count=(
                    range_component_sample_count if clear_available else 0
                ),
                echo_object_count=(
                    range_object_count if echo_available else 0
                ),
                state_echo_pixel_count=range_component_sample_count,
                state_clear_pixel_count=range_component_sample_count,
                state_echo_object_count=range_object_count,
            )
            for range_regime in reference_ranges
        )
        return promotion_module._new_prior_holdout_evaluation(
            holdout_plan_digest=manifest.holdout_plan_digest,
            candidate_manifest_digest=manifest.manifest_digest,
            candidate_prior_digest=manifest.candidate_prior_digest,
            parent_prior_digest=manifest.parent_prior_digest,
            case_id=case.case_id,
            storm_id=case.storm_id,
            physical_event_digest=(
                case.physical_event_digest
                if physical_event_digest is None
                else physical_event_digest
            ),
            day=case.day,
            radar_id=case.radar_id,
            regime=case.regime,
            range_regime=case.range_regime,
            reference_active_range_regimes=reference_ranges,
            range_band_contract_digest=case.range_band_contract_digest,
            range_band_evaluations=range_evaluations,
            regime_classifier_digest=regime_classifier_digest,
            regime_classifier_manifest_digest=classifier_manifest.manifest_digest,
            regime_classification_evidence_digest=(
                ("e" if index == 1 else "f") * 64
            ),
            classified_regime=(
                case.regime if classified_regime is None else classified_regime
            ),
            classified_range_regimes=(
                predicted_ranges
            ),
            classifier_regime_confidence=classifier_regime_confidence,
            classifier_range_confidence=classifier_range_confidence,
            classifier_regime_entropy=classifier_regime_entropy,
            classifier_is_ood=classifier_is_ood,
            classifier_reference_agreement=classifier_reference_agreement,
            classifier_weather_reference_agreement=(
                (case.regime if classified_regime is None else classified_regime)
                == case.regime
            ),
            classifier_range_set_precision=range_precision,
            classifier_range_set_recall=range_recall,
            classifier_range_exact_set_match=(predicted_set == reference_set),
            classifier_false_active_band_fraction=false_active_fraction,
            classifier_reference_range_is_ood=not reference_ranges,
            classifier_numerical_runtime_digest=(
                classifier_manifest.numerical_runtime_digest
            ),
            classifier_input_dtype=str(torch.float32),
            classifier_input_device="cpu",
            classifier_weather_top1_top2_gap=1.0,
            classifier_minimum_range_presence_margin=0.5,
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
            metric_change=torch.tensor(
                [[change]]
                if secondary_range_change is None
                else [[change, secondary_range_change]],
                dtype=torch.float64,
            ),
            candidate_issuance_effect=torch.tensor(
                [[candidate_issuance] * len(metric_names)], dtype=torch.float64
            ),
            parent_issuance_effect=torch.zeros(
                (1, len(metric_names)), dtype=torch.float64
            ),
            end_to_end_metric_change=torch.tensor(
                [[change if end_to_end is None else end_to_end] * len(metric_names)],
                dtype=torch.float64,
            ),
            metric_available=torch.ones((1, len(metric_names)), dtype=torch.bool),
            lead_minutes=(60,),
            metric_names=metric_names,
            verification_digest="a" * 64,
            metric_contract_digest="b" * 64,
            coverage_candidate=torch.tensor([1.0], dtype=torch.float64),
            coverage_parent=torch.tensor([1.0], dtype=torch.float64),
            coverage_common=torch.tensor([1.0], dtype=torch.float64),
            newly_issued_fraction=torch.tensor([0.0], dtype=torch.float64),
            withdrawn_fraction=torch.tensor([0.0], dtype=torch.float64),
            prior_conditional_pit_residual_mean_abs=(
                prior_residual_mean_abs if echo_available else None
            ),
            prior_conditional_underdispersion_fraction=(
                prior_underdispersion_fraction if echo_available else None
            ),
            prior_echo_intensity_nll=(
                prior_echo_intensity_nll if echo_available else None
            ),
            prior_support_brier_score=prior_support_brier_score,
            prior_echo_support_miss_score=(
                prior_echo_support_miss_score if echo_available else None
            ),
            prior_echo_object_miss_score=(
                prior_echo_object_miss_score if echo_available else None
            ),
            prior_clear_sky_false_echo_score=(
                prior_clear_sky_false_echo_score if clear_available else None
            ),
            parent_prior_conditional_pit_residual_mean_abs=(
                prior_residual_mean_abs if echo_available else None
            ),
            parent_prior_conditional_underdispersion_fraction=(
                None
                if not echo_available
                else prior_underdispersion_fraction
                if parent_prior_underdispersion_fraction is None
                else parent_prior_underdispersion_fraction
            ),
            parent_prior_echo_intensity_nll=(
                parent_prior_echo_intensity_nll if echo_available else None
            ),
            parent_prior_support_brier_score=parent_prior_support_brier_score,
            parent_prior_echo_support_miss_score=(
                parent_prior_echo_support_miss_score if echo_available else None
            ),
            parent_prior_echo_object_miss_score=(
                parent_prior_echo_object_miss_score if echo_available else None
            ),
            parent_prior_clear_sky_false_echo_score=(
                parent_prior_clear_sky_false_echo_score
                if clear_available
                else None
            ),
            prior_echo_intensity_status=(
                "available" if echo_available else "not_applicable"
            ),
            prior_clear_sky_status=(
                "available" if clear_available else "not_applicable"
            ),
            prior_candidate_valid_fraction=prior_candidate_valid_fraction,
            prior_parent_valid_fraction=prior_parent_valid_fraction,
            prior_candidate_valid_area_km2=prior_candidate_valid_area_km2,
            prior_abstention_increase_vs_parent=(
                prior_parent_valid_fraction - prior_candidate_valid_fraction
            ),
            prior_uncertainty_target_digest=case.uncertainty_target_digest,
            prior_uncertainty_sample_count=prior_sample_count,
            prior_echo_intensity_sample_count=echo_count,
            prior_clear_sky_sample_count=clear_count,
            prior_echo_area_km2=echo_count * 0.25,
            prior_clear_sky_area_km2=clear_count * 0.25,
            prior_echo_object_count=1 if echo_available else 0,
            state_candidate_gaussian_nll=state_candidate_gaussian_nll,
            state_parent_gaussian_nll=state_parent_gaussian_nll,
            state_candidate_pit_residual_mean_abs=0.5,
            state_parent_pit_residual_mean_abs=0.5,
            state_candidate_underdispersion_fraction=0.0,
            state_parent_underdispersion_fraction=0.0,
            state_candidate_support_brier_score=(
                state_candidate_support_brier_score
            ),
            state_parent_support_brier_score=state_parent_support_brier_score,
            state_candidate_echo_support_miss_score=0.05,
            state_parent_echo_support_miss_score=0.05,
            state_candidate_echo_object_miss_score=0.05,
            state_parent_echo_object_miss_score=0.05,
            state_candidate_false_support_score=(
                state_candidate_false_support_score
            ),
            state_parent_false_support_score=state_parent_false_support_score,
            state_candidate_valid_brier_score=0.05,
            state_parent_valid_brier_score=0.05,
            state_calibration_target_digest=case.state_calibration_target_digest,
            state_calibration_sample_count=16,
            state_calibration_echo_sample_count=8,
            state_calibration_clear_sample_count=8,
            state_calibration_echo_object_count=1,
            issue_time=case.issue_time,
            verification_valid_times=(f"2026-08-{8 + index:02d}T01:00:00Z",),
        )

    def policy(self) -> NeuralPriorPromotionPolicy:
        return NeuralPriorPromotionPolicy(
            metric_scales=(PromotionMetricScale("log_echo_mse", 1.0, 0.01),),
            approved_candidate_manifest_digests=(self.manifest().manifest_digest,),
            approved_holdout_plan_digests=(self.plan().plan_digest,),
            approved_metric_contract_digests=("b" * 64,),
            approved_physical_event_catalog_result_digest=(
                self.event_catalog_result().result_digest
            ),
            deployment_regime_classifier_digest="e" * 64,
            deployment_regime_classifier_manifest_digest=(
                self.plan().regime_classifier_manifests[0].manifest_digest
            ),
            required_range_metrics=(
                promotion_module.RangeMetricRequirement(
                    weather_regime="convective",
                    range_regime="near_range",
                    metric_name="log_echo_mse",
                    lead_minutes=60,
                    minimum_cases=1,
                    minimum_physical_events=1,
                    minimum_valid_area_km2=0.0,
                    maximum_mean_normalized_degradation=0.0,
                    maximum_harmful_fraction_upper_bound=1.0,
                ),
                promotion_module.RangeMetricRequirement(
                    weather_regime="stratiform",
                    range_regime="far_range",
                    metric_name="log_echo_mse",
                    lead_minutes=60,
                    minimum_cases=1,
                    minimum_physical_events=1,
                    minimum_valid_area_km2=0.0,
                    maximum_mean_normalized_degradation=0.0,
                    maximum_harmful_fraction_upper_bound=1.0,
                ),
            ),
            minimum_holdout_cases=2,
            minimum_material_cases=2,
            minimum_material_case_fraction=1.0,
            minimum_independent_cases=2,
            minimum_distinct_storms=2,
            minimum_distinct_days=2,
            minimum_distinct_radars=1,
            minimum_distinct_regimes=2,
            minimum_distinct_range_regimes=2,
            minimum_material_clusters=2,
            minimum_prior_echo_cases=2,
            minimum_prior_clear_cases=2,
            minimum_prior_echo_clusters=2,
            minimum_prior_clear_clusters=2,
            minimum_uncertainty_cases_per_regime=1,
            minimum_echo_cases_per_regime=1,
            minimum_clear_cases_per_regime=1,
            minimum_uncertainty_clusters_per_regime=1,
            minimum_echo_clusters_per_regime=1,
            minimum_clear_clusters_per_regime=1,
            minimum_prior_echo_pixels_per_case=1,
            minimum_prior_clear_pixels_per_case=1,
            minimum_prior_echo_area_km2_per_case=0.0,
            minimum_prior_clear_area_km2_per_case=0.0,
            minimum_prior_echo_objects_per_case=1,
            minimum_bootstrap_tail_replicates=1,
            minimum_state_calibration_samples_per_case=1,
            minimum_state_calibration_cases_per_regime=1,
            minimum_state_calibration_clusters_per_regime=1,
            minimum_regime_classifier_ood_cases=0,
            minimum_regime_classifier_accuracy_lower_bound=0.0,
            minimum_regime_classifier_recall_lower_bound=0.0,
            maximum_regime_classifier_false_routing_upper_bound=1.0,
            minimum_regime_classifier_clusters=1,
            minimum_range_classifier_ood_cases=0,
            minimum_range_set_precision_lower_bound=0.0,
            minimum_range_set_recall_lower_bound=0.0,
            maximum_false_active_band_upper_bound=1.0,
            minimum_range_band_cases=1,
            minimum_range_band_clusters=1,
            minimum_range_band_area_km2=0.0,
            minimum_range_metric_valid_area_km2=0.0,
            minimum_range_probability_valid_area_km2=0.0,
            minimum_range_state_valid_area_km2=0.0,
            minimum_range_component_samples=1,
            minimum_range_echo_objects=1,
            minimum_range_state_echo_objects=1,
            minimum_weather_top1_top2_gap=0.0,
            minimum_range_presence_margin=0.0,
            minimum_beneficial_fraction=0.0,
            maximum_harmful_fraction=1.0,
            minimum_mean_normalized_improvement=0.1,
            bootstrap_samples=256,
            minimum_deployment_metric_cell_events=1,
        )

    def compute(self, evaluations):
        policy = self.policy()
        return self.compute_with_policy(evaluations, policy)

    def compute_with_policy(self, evaluations, policy):
        plan = self.plan()
        manifest = self.manifest()
        default_classifier = plan.regime_classifier_manifests[0]
        if (
            policy.deployment_regime_classifier_digest
            != default_classifier.classifier_digest
        ):
            classifier_manifest = replace(
                default_classifier,
                classifier_digest=policy.deployment_regime_classifier_digest,
            )
            plan = replace(
                plan,
                regime_classifier_manifests=(classifier_manifest,),
            )
            manifest = replace(
                manifest,
                holdout_plan_digest=plan.plan_digest,
                holdout_dataset_digest=plan.holdout_dataset_digest,
            )
            policy = replace(
                policy,
                approved_candidate_manifest_digests=(manifest.manifest_digest,),
                approved_holdout_plan_digests=(plan.plan_digest,),
                deployment_regime_classifier_manifest_digest=(
                    classifier_manifest.manifest_digest
                ),
            )
            evaluations = tuple(
                promotion_module._new_prior_holdout_evaluation(
                    **{
                        key: value
                        for key, value in evaluation.__dict__.items()
                        if key not in ("contract", "evaluation_digest")
                    }
                    | {
                        "holdout_plan_digest": plan.plan_digest,
                        "candidate_manifest_digest": manifest.manifest_digest,
                        "regime_classifier_manifest_digest": (
                            classifier_manifest.manifest_digest
                        ),
                    }
                )
                for evaluation in evaluations
            )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=_LearningPolicyTrustStore(
                approved_policy_digests=frozenset((policy.digest,)),
                content_digest="b" * 64,
            ),
        ):
            return compute_neural_prior_promotion(
                manifest,
                plan,
                evaluations,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_promotes_only_independent_material_holdout_cases(self) -> None:
        result = self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        self.assertTrue(result.eligible)
        validate_neural_prior_promotion(result)

    def test_current_promotion_round_trips_durable_v15_evidence(self) -> None:
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        evidence = self.compute(evaluations)
        manifest = self.manifest()
        plan = self.plan()
        policy = self.policy()
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest, plan_id, plan_json, policy_digest, "
                    "trust_store_digest, registered_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan.plan_digest,
                        plan.plan_id,
                        json.dumps(asdict(plan), sort_keys=True),
                        "6" * 64,
                        "7" * 64,
                        plan.registered_at,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                approval_schema = connection.execute(
                    "PRAGMA table_info(variational_learning_approvals)"
                ).fetchall()
                approval_columns = [str(row[1]) for row in approval_schema]
                approval_overrides: dict[str, object] = {
                    "learning_result_digest": "8" * 64,
                    "approval_evidence_digest": "a" * 64,
                    "created_at": "2026-07-01T00:00:00+00:00",
                }
                approval_values = [
                    approval_overrides.get(
                        str(row[1]),
                        0 if str(row[2]).upper() == "INTEGER" else 0.0
                        if str(row[2]).upper() == "REAL"
                        else "",
                    )
                    for row in approval_schema
                ]
                connection.execute(
                    f"INSERT INTO variational_learning_approvals "
                    f"({','.join(approval_columns)}) VALUES "
                    f"({','.join('?' for _ in approval_columns)})",
                    approval_values,
                )
                connection.execute(
                    "INSERT INTO prospective_intervention_decisions "
                    "(decision_digest, decision_id, decision_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        "e" * 64,
                        "training-decision",
                        json.dumps(
                            {"decision_basis_digest": "a" * 64},
                            sort_keys=True,
                        ),
                        "2026-07-01T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO realized_intervention_receipts "
                    "(receipt_digest, decision_digest, receipt_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        "f" * 64,
                        "e" * 64,
                        json.dumps(
                            {
                                "actual_input_bundle_digest": "0" * 64,
                                "decision_digest": "e" * 64,
                            },
                            sort_keys=True,
                        ),
                        "2026-07-01T00:00:00+00:00",
                    ),
                )
            ledger.append_physical_event_catalog_result(
                plan,
                manifest.physical_event_catalog_result,
            )
            assert manifest.candidate_scoring_start_receipt is not None
            with patch.object(
                ledger_module,
                "datetime",
                wraps=datetime,
            ) as trusted_datetime:
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T02:00:00+00:00"
                )
                ledger.append_trusted_process_start_receipt(
                    plan,
                    manifest.physical_event_catalog_result,
                    manifest.candidate_scoring_start_receipt,
                )
            trust = _LearningPolicyTrustStore(
                approved_policy_digests=frozenset((policy.digest,)),
                content_digest="b" * 64,
            )
            with patch.object(
                promotion_module,
                "_load_learning_policy_trust_store",
                return_value=trust,
            ):
                stored = ledger.append_neural_prior_promotion(
                    evidence,
                    manifest,
                    plan,
                    evaluations,
                    policy=policy,
                    policy_trust_store_path="/etc/advar/learning-policies.json",
                )
            loaded = ledger.load_neural_prior_promotion(stored)
            self.assertEqual(loaded.promotion_evidence_digest, stored)
            self.assertEqual(loaded.contract, "neural-prior-promotion-evidence-v15")

            legacy_payload = evidence._payload()
            legacy_payload["contract"] = "neural-prior-promotion-evidence-v12"
            legacy_payload.pop("certified_range_geometry_contract_digests")
            legacy_digest = promotion_module.json_digest(legacy_payload)
            with sqlite3.connect(ledger.index_path) as connection:
                connection.row_factory = sqlite3.Row
                current_row = connection.execute(
                    "SELECT * FROM neural_prior_promotions "
                    "WHERE promotion_evidence_digest = ?",
                    (stored,),
                ).fetchone()
                assert current_row is not None
                legacy_row = dict(current_row)
                legacy_row["promotion_evidence_digest"] = legacy_digest
                legacy_row["evidence_contract"] = (
                    "neural-prior-promotion-evidence-v12"
                )
                legacy_row["evidence_payload_json"] = json.dumps(
                    legacy_payload,
                    sort_keys=True,
                )
            legacy_ledger = EpisodeLedger(Path(directory) / "legacy")
            columns = tuple(legacy_row)
            with sqlite3.connect(legacy_ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_promotions "
                    f"({','.join(columns)}) VALUES "
                    f"({','.join('?' for _ in columns)})",
                    tuple(legacy_row[name] for name in columns),
                )
            legacy = legacy_ledger.load_neural_prior_promotion(legacy_digest)
            self.assertIsInstance(
                legacy,
                promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV12,
            )

    def test_promotion_requires_every_preregistered_case(self) -> None:
        with self.assertRaisesRegex(ValueError, "every planned case"):
            self.compute((self.evaluation(1, -0.2),))

    def test_prior_holdout_evidence_has_no_intervention_selection_fields(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        self.assertFalse(hasattr(evaluation, "intervention_digest"))
        self.assertFalse(hasattr(evaluation, "population_contract"))

    def test_end_to_end_harm_blocks_promotion(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2, end_to_end=2.0),
                self.evaluation(2, -0.3),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("excessive_end_to_end_degradation", result.rejection_reasons)

    def test_nonfinite_issuance_effect_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite metrics"):
            self.evaluation(1, -0.2, candidate_issuance=float("nan"))

    def test_unreliable_prior_uncertainty_blocks_promotion(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_underdispersion_fraction=0.5,
                ),
                self.evaluation(2, -0.3),
            )
        )
        self.assertIn("unreliable_prior_uncertainty", result.rejection_reasons)

    def test_prior_uncertainty_must_not_regress_against_parent(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_echo_intensity_nll=3.9,
                    parent_prior_echo_intensity_nll=1.0,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    prior_echo_intensity_nll=3.9,
                    parent_prior_echo_intensity_nll=1.0,
                ),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("inferior_prior_uncertainty", result.rejection_reasons)

    def test_probability_skill_cannot_hide_unreliable_state_uncertainty(
        self,
    ) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    state_candidate_gaussian_nll=100.0,
                ),
                self.evaluation(2, -0.3),
            )
        )

        self.assertFalse(result.eligible)
        self.assertFalse(result.state_calibration_eligible)
        self.assertIn("unreliable_state_head", result.rejection_reasons)

    def test_state_uncertainty_regression_against_parent_blocks_promotion(
        self,
    ) -> None:
        result = self.compute(
            tuple(
                self.evaluation(
                    index,
                    -0.1 * index,
                    state_candidate_gaussian_nll=2.0,
                    state_parent_gaussian_nll=0.5,
                )
                for index in (1, 2)
            )
        )

        self.assertFalse(result.eligible)
        self.assertIn("inferior_state_head", result.rejection_reasons)

    def test_state_false_support_is_a_direct_promotion_guard(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    state_candidate_false_support_score=1.0,
                    state_parent_false_support_score=0.0,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    state_candidate_false_support_score=1.0,
                    state_parent_false_support_score=0.0,
                ),
            )
        )

        self.assertFalse(result.eligible)
        self.assertIn("unreliable_state_head", result.rejection_reasons)
        self.assertIn("inferior_state_head", result.rejection_reasons)

    def test_clear_sky_gain_cannot_hide_echo_intensity_regression(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_echo_intensity_nll=1.5,
                    parent_prior_echo_intensity_nll=0.5,
                    prior_clear_sky_false_echo_score=0.0,
                    parent_prior_clear_sky_false_echo_score=0.2,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    prior_echo_intensity_nll=1.5,
                    parent_prior_echo_intensity_nll=0.5,
                    prior_clear_sky_false_echo_score=0.0,
                    parent_prior_clear_sky_false_echo_score=0.2,
                ),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("inferior_prior_uncertainty", result.rejection_reasons)

    def test_clear_sky_hurdle_score_is_floor_representation_invariant(self) -> None:
        application = SimpleNamespace(
            truncated_location_dbz=torch.zeros((1, 3)),
            truncated_scale_dbz=torch.ones((1, 3)),
            event_probability=torch.full((1, 3), 0.2),
        )
        mask = torch.ones((1, 3), dtype=torch.bool)
        support = torch.zeros((1, 3), dtype=torch.bool)
        scores = [
            promotion_module._prior_uncertainty_scores(
                application,
                torch.full((1, 3), floor),
                support,
                mask,
                support_threshold_dbz=5.0,
            )
            for floor in (-10.0, 0.0, 4.9)
        ]
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scores[1], scores[2])
        self.assertIsNone(scores[0].echo_intensity_nll)
        self.assertEqual(scores[0].echo_sample_count, 0)
        self.assertEqual(scores[0].clear_sample_count, 3)

    def test_pure_clear_and_pure_echo_cases_use_component_applicability(self) -> None:
        policy = replace(
            self.policy(),
            minimum_prior_echo_cases=1,
            minimum_prior_clear_cases=1,
            minimum_prior_echo_clusters=1,
            minimum_prior_clear_clusters=1,
        )
        result = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    echo_available=False,
                    clear_available=True,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    echo_available=True,
                    clear_available=False,
                ),
            ),
            policy,
        )

        self.assertTrue(result.eligible)
        self.assertEqual(result.prior_echo_case_count, 1)
        self.assertEqual(result.prior_clear_sky_case_count, 1)

    def test_one_cluster_regime_cannot_support_promotion(self) -> None:
        policy = replace(
            self.policy(),
            minimum_uncertainty_clusters_per_regime=2,
        )
        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )

        self.assertTrue(result.eligible)
        self.assertFalse(result.deployment_eligible)
        self.assertEqual(result.certified_applicability_regime_groups, ())

    def test_simultaneous_uncertainty_count_covers_family_components_groups(
        self,
    ) -> None:
        components = (
            "intensity",
            "support",
            "echo_miss",
            "object_miss",
            "clear",
            "underdispersion",
        )
        groups = (None,) + tuple((f"regime-{i}", "range") for i in range(6))
        comparisons = tuple(
            promotion_module._UncertaintyComparison(
                component=component,
                group=group,
                values=(0.0, 0.0),
                clusters=(("storm-1", "day-1", "radar-1"),
                          ("storm-2", "day-2", "radar-1")),
            )
            for component in components
            for group in groups
        )

        result = promotion_module._simultaneous_uncertainty_upper_bounds(
            comparisons,
            self.policy(),
            candidate_family_size=10,
        )

        self.assertEqual(result.test_count, 10 * 6 * 7)
        self.assertEqual(result.method, "exact_sign_enumeration")
        self.assertEqual(result.effective_replicates, 4)

    def test_truncated_tail_score_is_stable_and_differentiable(self) -> None:
        for lower in (0.0, 5.0, 8.0, 12.0, 20.0):
            location = torch.tensor([5.0 - lower], dtype=torch.float32, requires_grad=True)
            scale = torch.ones(1, dtype=torch.float32, requires_grad=True)
            reference = torch.tensor([5.0], dtype=torch.float32)
            nll, _ = promotion_module._truncated_gaussian_diagnostics(
                location,
                scale,
                reference,
                support_threshold_dbz=5.0,
            )
            log_lower = torch.special.log_ndtr(
                torch.tensor(-lower, dtype=torch.float64)
            )
            log_upper = torch.special.log_ndtr(
                torch.tensor(-(lower + 0.25), dtype=torch.float64)
            )
            expected = -torch.log(-torch.expm1(log_upper - log_lower))
            torch.testing.assert_close(nll[0], expected)
            nll.sum().backward()
            self.assertTrue(torch.isfinite(location.grad).all())
            self.assertTrue(torch.isfinite(scale.grad).all())

    def test_echo_miss_is_not_hidden_by_clear_sky_prevalence(self) -> None:
        application = SimpleNamespace(
            truncated_location_dbz=torch.full((1000,), 5.0),
            truncated_scale_dbz=torch.ones(1000),
            event_probability=torch.zeros(1000),
        )
        support = torch.zeros(1000, dtype=torch.bool)
        support[0] = True
        scores = promotion_module._prior_uncertainty_scores(
            application,
            torch.where(support, 5.0, -10.0),
            support,
            torch.ones(1000, dtype=torch.bool),
            support_threshold_dbz=5.0,
        )
        self.assertAlmostEqual(scores.support_brier_score, 0.001)
        self.assertEqual(scores.echo_support_miss_score, 1.0)
        self.assertEqual(scores.clear_sky_false_echo_score, 0.0)

    def test_conditional_pit_replaces_raw_truncated_z_score(self) -> None:
        lower = 8.0
        log_lower_survival = torch.special.log_ndtr(
            torch.tensor(-lower, dtype=torch.float64)
        )
        left, right = lower, lower + 4.0
        for _ in range(80):
            midpoint = (left + right) / 2.0
            log_ratio = float(
                torch.special.log_ndtr(
                    torch.tensor(-midpoint, dtype=torch.float64)
                )
                - log_lower_survival
            )
            conditional_cdf = -torch.expm1(
                torch.tensor(log_ratio, dtype=torch.float64)
            ).item()
            if conditional_cdf < 0.5:
                left = midpoint
            else:
                right = midpoint
        reference = torch.tensor([(left + right) / 2.0], dtype=torch.float64)
        _, pit = promotion_module._truncated_gaussian_diagnostics(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            reference,
            support_threshold_dbz=8.0,
            reflectivity_resolution_dbz=1.0e-6,
            quantization_origin_dbz=float(reference[0]),
        )
        self.assertGreater(float(reference[0]), 8.0)
        self.assertAlmostEqual(float(pit[0]), 0.0, places=6)

    def test_quantized_threshold_uses_interval_midpoint_pit(self) -> None:
        threshold = 5.0
        width = 0.5
        location = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        reference = torch.tensor(
            [threshold, threshold + width, threshold + 2.0 * width],
            dtype=torch.float32,
        )

        nll, pit = promotion_module._truncated_gaussian_diagnostics(
            location,
            scale,
            reference,
            support_threshold_dbz=threshold,
            reflectivity_resolution_dbz=width,
            quantization_origin_dbz=-10.0,
        )

        self.assertTrue(torch.all(torch.isfinite(nll)))
        self.assertTrue(torch.all(torch.isfinite(pit)))
        self.assertGreater(float(pit[0]), -2.0)
        self.assertNotAlmostEqual(float(pit[0]), -8.126, places=2)
        with self.assertRaisesRegex(ValueError, "off its declared lattice"):
            promotion_module._truncated_gaussian_diagnostics(
                torch.tensor([5.0]),
                torch.tensor([1.0]),
                torch.tensor([5.3]),
                support_threshold_dbz=threshold,
                reflectivity_resolution_dbz=width,
                quantization_origin_dbz=-10.0,
            )
        with self.assertRaisesRegex(ValueError, "threshold.*lattice"):
            promotion_module._truncated_gaussian_diagnostics(
                torch.tensor([5.0]),
                torch.tensor([1.0]),
                torch.tensor([5.5]),
                support_threshold_dbz=5.1,
                reflectivity_resolution_dbz=width,
                quantization_origin_dbz=-10.0,
            )

    def test_quantized_threshold_bins_form_a_disjoint_partition(self) -> None:
        reference = torch.tensor([5.0, 5.5], dtype=torch.float32)

        lower, upper = promotion_module._quantized_bin_bounds(
            reference,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
            support_threshold_dbz=5.0,
            threshold_bin_convention="nearest_rounding_threshold_censor",
        )

        self.assertEqual(float(lower[0]), 5.0)
        self.assertEqual(float(upper[0]), float(lower[1]))
        self.assertEqual(float(upper[1]), 5.75)

    def test_float32_value_on_decimal_lattice_is_accepted(self) -> None:
        nll, pit = promotion_module._quantized_gaussian_diagnostics(
            torch.tensor([35.0], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([35.1], dtype=torch.float32),
            reflectivity_resolution_dbz=0.1,
            quantization_origin_dbz=-10.0,
            support_threshold_dbz=5.0,
            threshold_bin_convention="nearest_rounding_threshold_censor",
        )

        self.assertTrue(torch.all(torch.isfinite(nll)))
        self.assertTrue(torch.all(torch.isfinite(pit)))

    def test_range_masks_must_cover_the_complete_operational_grid(self) -> None:
        incomplete = {
            "near_range": torch.tensor(
                [[True, False], [False, False]], dtype=torch.bool
            ),
            "far_range": torch.zeros((2, 2), dtype=torch.bool),
        }

        with self.assertRaisesRegex(ValueError, "complete partition"):
            promotion_module._validate_complete_range_partition(incomplete)

    def test_range_geometry_resolves_a_complete_physical_partition(self) -> None:
        grid_x_m = torch.tensor([[0.0, 20_000.0], [40_000.0, 80_000.0]])
        grid_y_m = torch.zeros_like(grid_x_m)
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="a" * 64,
            radar_site_location_digest="a" * 64,
            grid_contract_digest="2" * 64,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range", "far_range"),
            radial_distance_edges_m=(0.0, 30_000.0, 100_000.0),
            horizontal_range_rule_digest="b" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(grid_x_m),
            grid_y_m_digest=promotion_module.tensor_digest(grid_y_m),
        )

        partition = promotion_module.resolve_range_geometry(
            geometry,
            grid_x_m=grid_x_m,
            grid_y_m=grid_y_m,
        )

        self.assertEqual(partition.active_range_regimes, ("near_range", "far_range"))
        self.assertTrue(
            torch.equal(
                partition.masks[0],
                torch.tensor([[True, True], [False, False]]),
            )
        )
        self.assertTrue(
            torch.equal(
                partition.masks[1],
                torch.tensor([[False, False], [True, True]]),
            )
        )
        object.__setattr__(
            partition,
            "active_range_regimes",
            ("near_range",),
        )
        with self.assertRaisesRegex(ValueError, "range partition evidence"):
            partition.validate_integrity()

    def test_range_geometry_contract_is_explicitly_horizontal_only(self) -> None:
        coordinates = torch.zeros((2, 2))
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="a" * 64,
            radar_site_location_digest="a" * 64,
            grid_contract_digest="2" * 64,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range",),
            radial_distance_edges_m=(0.0, 100_000.0),
            horizontal_range_rule_digest="b" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(coordinates),
            grid_y_m_digest=promotion_module.tensor_digest(coordinates),
        )

        self.assertEqual(
            geometry.contract,
            "radar-horizontal-range-geometry-contract-v3",
        )
        self.assertEqual(
            geometry.resolver_algorithm,
            "projected-horizontal-euclidean-range-v3",
        )

    def test_operational_range_geometry_must_match_current_run_grid(self) -> None:
        frames = torch.zeros((3, 2, 2))
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-09T00:00:00Z",
                "2026-08-09T00:10:00Z",
                "2026-08-09T00:20:00Z",
            ),
            dx_m=1_000.0,
            dy_m=1_000.0,
            projection="EPSG:3857",
            grid_hash="1" * 64,
        )
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
            grid_time_contract=grid,
        )
        x = torch.tensor([[0.0, 1_000.0], [0.0, 1_000.0]])
        y = torch.tensor([[0.0, 0.0], [1_000.0, 1_000.0]])
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="a" * 64,
            radar_site_location_digest="a" * 64,
            grid_contract_digest="2" * 64,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range",),
            radial_distance_edges_m=(0.0, 10_000.0),
            horizontal_range_rule_digest="b" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(x),
            grid_y_m_digest=promotion_module.tensor_digest(y),
        )
        selected = Mock()
        selected._infer_deployed.return_value = object()

        with patch.object(
            promotion_module,
            "_select_deployed_prior",
            return_value=(selected, Mock()),
        ), self.assertRaisesRegex(ValueError, "operational grid"):
            promotion_module.infer_deployed_neural_prior(
                frames,
                input_run=run,
                candidate_runner=Mock(),
                parent_runner=Mock(),
                promotion_evidence=Mock(),
                regime_classifier=Mock(classify=Mock(return_value=Mock())),
                range_geometry_contract=geometry,
                grid_x_m=x,
                grid_y_m=y,
                policy=Mock(),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

    def test_operational_range_coordinates_must_match_frame_shape(self) -> None:
        frames = torch.zeros((3, 4, 4))
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-09T00:00:00Z",
                "2026-08-09T00:10:00Z",
                "2026-08-09T00:20:00Z",
            ),
            dx_m=1_000.0,
            dy_m=1_000.0,
            projection="EPSG:3857",
            grid_hash="1" * 64,
        )
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
            grid_time_contract=grid,
        )
        x = torch.tensor([[0.0, 1_000.0], [0.0, 1_000.0]])
        y = torch.tensor([[0.0, 0.0], [1_000.0, 1_000.0]])
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="a" * 64,
            radar_site_location_digest="a" * 64,
            grid_contract_digest=grid.digest,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range",),
            radial_distance_edges_m=(0.0, 10_000.0),
            horizontal_range_rule_digest="b" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(x),
            grid_y_m_digest=promotion_module.tensor_digest(y),
        )
        selected = Mock()
        selected._infer_deployed.return_value = object()

        with patch.object(
            promotion_module,
            "_select_deployed_prior",
            return_value=(selected, Mock()),
        ), self.assertRaisesRegex(ValueError, "radar frames"):
            promotion_module.infer_deployed_neural_prior(
                frames,
                input_run=run,
                candidate_runner=Mock(),
                parent_runner=Mock(),
                promotion_evidence=Mock(),
                regime_classifier=Mock(classify=Mock(return_value=Mock())),
                range_geometry_contract=geometry,
                grid_x_m=x,
                grid_y_m=y,
                policy=Mock(),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

    def test_selector_rechecks_operational_partition_grid(self) -> None:
        partition = Mock(
            grid_contract_digest="2" * 64,
            masks=(torch.zeros((2, 2), dtype=torch.bool),),
        )

        with self.assertRaisesRegex(ValueError, "operational grid"):
            promotion_module._select_deployed_prior(
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                partition,
                Mock(),
                range_geometry_contract=Mock(),
                operational_grid_contract_digest="1" * 64,
                operational_frame_shape=(2, 2),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

    def test_cluster_mean_cannot_hide_case_level_harm_frequency(self) -> None:
        scores = ([-0.1] * 9 + [1.0]) * 2
        event_ids = ["event-1"] * 10 + ["event-2"] * 10

        _, harmful_upper, mean_lower = promotion_module._cluster_bounds(
            scores,
            event_ids,
            self.policy(),
            candidate_family_size=1,
        )

        self.assertGreater(mean_lower, 0.0)
        self.assertGreaterEqual(harmful_upper, 0.9)

    def test_same_physical_storm_across_days_and_radars_is_one_cluster(self) -> None:
        first = SimpleNamespace(
            storm_id="mcs-2026-08-10-01",
            physical_event_digest="7" * 64,
            day="2026-08-10",
            radar_id="radar-a",
        )
        second = SimpleNamespace(
            storm_id="mcs-2026-08-10-01",
            physical_event_digest="7" * 64,
            day="2026-08-11",
            radar_id="radar-c",
        )

        self.assertEqual(
            promotion_module._physical_event_cluster(first),
            promotion_module._physical_event_cluster(second),
        )

    def test_band_family_requires_its_own_bootstrap_tail_resolution(self) -> None:
        policy = replace(
            self.policy(),
            bootstrap_samples=1_000,
            minimum_bootstrap_tail_replicates=20,
        )

        with self.assertRaisesRegex(ValueError, "tail resolution"):
            promotion_module._bootstrap_tail_diagnostics(
                policy,
                family_size=10,
            )

    def test_five_perfect_events_do_not_produce_certain_rate_bounds(self) -> None:
        policy = self.policy()
        event_ids = [f"event-{index}" for index in range(5)]

        perfect_lower, _ = promotion_module._event_fractional_rate_interval(
            [1.0] * 5,
            event_ids,
            policy,
            family_size=1,
        )
        _, zero_failure_upper = promotion_module._event_fractional_rate_interval(
            [0.0] * 5,
            event_ids,
            policy,
            family_size=1,
        )

        self.assertLess(perfect_lower, 0.9)
        self.assertGreater(zero_failure_upper, 0.0)

    def test_binary_event_interval_rejects_fractional_event_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary event"):
            promotion_module._event_binary_rate_interval(
                [0.2, 0.8],
                ["event-1", "event-2"],
                self.policy(),
                family_size=1,
            )

    def test_skill_rate_bounds_retain_finite_event_uncertainty(self) -> None:
        policy = self.policy()
        event_ids = [f"event-{index}" for index in range(5)]

        beneficial_lower, harmful_upper, _ = promotion_module._cluster_bounds(
            [1.0] * 5,
            event_ids,
            policy,
            candidate_family_size=1,
        )

        self.assertLess(beneficial_lower, 0.9)
        self.assertGreater(harmful_upper, 0.0)

    def test_physical_event_digest_controls_outer_cluster(self) -> None:
        shared_event = "7" * 64
        first = self.evaluation(1, -0.2, physical_event_digest=shared_event)
        second = self.evaluation(2, -0.3, physical_event_digest=shared_event)

        self.assertNotEqual(first.storm_id, second.storm_id)
        self.assertEqual(
            promotion_module._physical_event_cluster(first),
            promotion_module._physical_event_cluster(second),
        )

    def test_physical_event_catalog_signature_binds_case_membership(self) -> None:
        catalog = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="mcs-2026-08-09",
            member_case_ids=("case-1", "case-2"),
            member_full_analysis_input_digests=("1" * 64, "2" * 64),
            start_time="2026-08-09T00:00:00Z",
            end_time="2026-08-10T00:00:00Z",
            spatial_envelope_xy_m=(-10_000.0, -20_000.0, 80_000.0, 90_000.0),
            participating_radar_ids=("radar-1", "radar-2"),
            association_algorithm_digest="3" * 64,
            adjudication_policy_digest="4" * 64,
            adjudicator_id="independent-event-adjudicator",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        changed = object.__new__(promotion_module.PhysicalEventCatalogEvidence)
        for name, value in catalog.__dict__.items():
            object.__setattr__(
                changed,
                name,
                ("case-1",) if name == "member_case_ids" else value,
            )

        promotion_module.validate_physical_event_catalog(catalog)
        with self.assertRaisesRegex(ValueError, "event-catalog"):
            promotion_module.validate_physical_event_catalog(changed)

    def test_physical_event_catalog_must_enclose_member_issue_times(self) -> None:
        manifest = self.manifest()
        first_case = manifest.holdout_cases[0]
        catalog = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="late-event",
            member_case_ids=(first_case.case_id,),
            member_full_analysis_input_digests=(
                first_case.full_analysis_input_digest,
            ),
            start_time="2026-08-09T03:00:00Z",
            end_time="2026-08-09T04:00:00Z",
            spatial_envelope_xy_m=(0.0, 0.0, 100_000.0, 100_000.0),
            participating_radar_ids=(first_case.radar_id,),
            association_algorithm_digest="5" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        replaced_case = replace(
            first_case,
            physical_event_digest=catalog.physical_event_identity_digest,
        )

        with self.assertRaisesRegex(ValueError, "outside physical event envelope"):
            replace(
                manifest,
                holdout_cases=(replaced_case, manifest.holdout_cases[1]),
                physical_event_catalog_evidences=(
                    catalog,
                    manifest.physical_event_catalog_evidences[1],
                ),
            )

    def test_event_catalog_adjudicator_must_be_preregistered(self) -> None:
        other_key = Ed25519PrivateKey.from_private_bytes(b"\x02" * 32)
        original = self.manifest()
        first_case = original.holdout_cases[0]
        untrusted = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="untrusted-event",
            member_case_ids=(first_case.case_id,),
            member_full_analysis_input_digests=(
                first_case.full_analysis_input_digest,
            ),
            start_time="2026-08-09T00:00:00Z",
            end_time="2026-08-09T02:00:00Z",
            spatial_envelope_xy_m=(0.0, 0.0, 100_000.0, 100_000.0),
            participating_radar_ids=(first_case.radar_id,),
            association_algorithm_digest="3" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="untrusted-adjudicator",
            adjudicator_private_key=other_key,
        )
        changed_case = replace(
            first_case,
            physical_event_digest=untrusted.physical_event_identity_digest,
        )
        second_case = original.holdout_cases[1]
        untrusted_second = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="untrusted-event-2",
            member_case_ids=(second_case.case_id,),
            member_full_analysis_input_digests=(
                second_case.full_analysis_input_digest,
            ),
            start_time="2026-08-10T00:00:00Z",
            end_time="2026-08-10T02:00:00Z",
            spatial_envelope_xy_m=(0.0, 0.0, 100_000.0, 100_000.0),
            participating_radar_ids=(second_case.radar_id,),
            association_algorithm_digest="3" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="untrusted-adjudicator",
            adjudicator_private_key=other_key,
        )
        changed_second_case = replace(
            second_case,
            physical_event_digest=untrusted_second.physical_event_identity_digest,
        )
        untrusted_plan = replace(
            self.event_catalog_plan(),
            adjudicator_id="untrusted-adjudicator",
            adjudicator_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(other_key)
            ),
            scheduler_id="untrusted-adjudicator",
            scheduler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(other_key)
            ),
        )
        untrusted_result = promotion_module.PhysicalEventCatalogResult.from_plan(
            untrusted_plan,
            event_evidences=(untrusted, untrusted_second),
            case_spatial_membership_evidences=(
                self.event_spatial_evidence(
                    untrusted,
                    case_id="case-1",
                    full_analysis_input_digest="1" * 64,
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    untrusted_second,
                    case_id="case-2",
                    full_analysis_input_digest="2" * 64,
                    source_object_evidence_digest="b" * 64,
                ),
            ),
            cataloged_at="2026-08-10T03:00:00Z",
            adjudicator_private_key=other_key,
        )
        changed_manifest = replace(
            original,
            physical_event_catalog_evidences=(
                untrusted,
                untrusted_second,
            ),
            physical_event_catalog_result=untrusted_result,
            holdout_cases=(changed_case, changed_second_case),
            candidate_scoring_start_receipt=self.scoring_start_receipt_for(
                untrusted_plan,
                untrusted_result,
                private_key=other_key,
            ),
        )
        policy = replace(
            self.policy(),
            approved_candidate_manifest_digests=(changed_manifest.manifest_digest,),
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((policy.digest,)),
            content_digest="b" * 64,
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "event-catalog adjudicator"):
            compute_neural_prior_promotion(
                changed_manifest,
                self.plan(),
                (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_event_association_algorithm_must_be_preregistered(self) -> None:
        original = self.manifest()
        first_case = original.holdout_cases[0]
        changed_catalog = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="changed-association-event",
            member_case_ids=(first_case.case_id,),
            member_full_analysis_input_digests=(
                first_case.full_analysis_input_digest,
            ),
            start_time="2026-08-09T00:00:00Z",
            end_time="2026-08-09T02:00:00Z",
            spatial_envelope_xy_m=(0.0, 0.0, 100_000.0, 100_000.0),
            participating_radar_ids=(first_case.radar_id,),
            association_algorithm_digest="9" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        changed_case = replace(
            first_case,
            physical_event_digest=changed_catalog.physical_event_identity_digest,
        )
        changed_catalog_plan = replace(
            self.event_catalog_plan(),
            association_algorithm_digest="9" * 64,
        )
        second_case = original.holdout_cases[1]
        changed_second_catalog = (
            promotion_module.PhysicalEventCatalogEvidence.from_members(
                event_id="changed-association-event-2",
                member_case_ids=(second_case.case_id,),
                member_full_analysis_input_digests=(
                    second_case.full_analysis_input_digest,
                ),
                start_time="2026-08-10T00:00:00Z",
                end_time="2026-08-10T02:00:00Z",
                spatial_envelope_xy_m=(0.0, 0.0, 100_000.0, 100_000.0),
                participating_radar_ids=(second_case.radar_id,),
                association_algorithm_digest="9" * 64,
                adjudication_policy_digest="6" * 64,
                adjudicator_id="independent-weather-labeler",
                adjudicator_private_key=self.regime_labeler_key(),
            )
        )
        changed_second_case = replace(
            second_case,
            physical_event_digest=(
                changed_second_catalog.physical_event_identity_digest
            ),
        )
        changed_result = promotion_module.PhysicalEventCatalogResult.from_plan(
            changed_catalog_plan,
            event_evidences=(
                changed_catalog,
                changed_second_catalog,
            ),
            case_spatial_membership_evidences=(
                self.event_spatial_evidence(
                    changed_catalog,
                    case_id="case-1",
                    full_analysis_input_digest="1" * 64,
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    changed_second_catalog,
                    case_id="case-2",
                    full_analysis_input_digest="2" * 64,
                    source_object_evidence_digest="b" * 64,
                ),
            ),
            cataloged_at="2026-08-10T03:00:00Z",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        with self.assertRaisesRegex(ValueError, "association algorithm"):
            replace(
                original,
                holdout_cases=(changed_case, changed_second_case),
                physical_event_catalog_evidences=(
                    changed_catalog,
                    changed_second_catalog,
                ),
                physical_event_catalog_result=changed_result,
                candidate_scoring_start_receipt=self.scoring_start_receipt_for(
                    changed_catalog_plan,
                    changed_result,
                ),
            )

    def test_physical_event_catalog_result_is_fixed_before_candidate_scoring(
        self,
    ) -> None:
        catalog_plan = promotion_module.PhysicalEventCatalogPlan(
            holdout_case_ids=("case-1", "case-2"),
            association_algorithm_digest="3" * 64,
            spatial_membership_rule_digest="4" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.regime_labeler_key()
                )
            ),
            catalog_completion_deadline="2026-08-11T00:00:00Z",
            spatial_reference_digest="7" * 64,
        )
        first = self.event_catalog(1)
        second = self.event_catalog(2)
        result = promotion_module.PhysicalEventCatalogResult.from_plan(
            catalog_plan,
            event_evidences=(first, second),
            case_spatial_membership_evidences=(
                self.event_spatial_evidence(
                    first,
                    case_id="case-1",
                    full_analysis_input_digest="1" * 64,
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    second,
                    case_id="case-2",
                    full_analysis_input_digest="2" * 64,
                    source_object_evidence_digest="b" * 64,
                ),
            ),
            cataloged_at="2026-08-10T03:00:00Z",
            adjudicator_private_key=self.regime_labeler_key(),
        )

        promotion_module.validate_physical_event_catalog_result(
            result,
            catalog_plan,
            candidate_scoring_started_at="2026-08-11T01:00:00Z",
        )
        with self.assertRaisesRegex(ValueError, "before candidate scoring"):
            promotion_module.validate_physical_event_catalog_result(
                result,
                catalog_plan,
                candidate_scoring_started_at="2026-08-10T02:00:00Z",
            )

    def test_candidate_manifest_binds_the_candidate_neutral_event_catalog(
        self,
    ) -> None:
        result = self.event_catalog_result()
        manifest = replace(
            self.manifest(),
            physical_event_catalog_result=result,
        )

        self.assertEqual(
            manifest.physical_event_catalog_result.result_digest,
            result.result_digest,
        )

    def test_promotion_policy_rejects_a_different_event_catalog_result(
        self,
    ) -> None:
        policy = replace(
            self.policy(),
            approved_physical_event_catalog_result_digest="f" * 64,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((policy.digest,)),
            content_digest="b" * 64,
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "event catalog result"):
            compute_neural_prior_promotion(
                self.manifest(),
                self.plan(),
                (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_holdout_plan_accepts_only_one_registered_event_catalog_result(
        self,
    ) -> None:
        plan = self.plan()
        result = self.event_catalog_result()
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest, plan_id, plan_json, policy_digest, "
                    "trust_store_digest, registered_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan.plan_digest,
                        plan.plan_id,
                        json.dumps(asdict(plan), sort_keys=True),
                        "6" * 64,
                        "7" * 64,
                        plan.registered_at,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
            stored = ledger.append_physical_event_catalog_result(
                plan,
                result,
            )
            self.assertEqual(stored, result.result_digest)

            with self.assertRaisesRegex(
                (FileExistsError, ValueError),
                "catalog|registered",
            ):
                ledger.append_physical_event_catalog_result(
                    plan,
                    result,
                )

    def test_candidate_training_events_must_be_disjoint_from_holdout_events(
        self,
    ) -> None:
        manifest = self.manifest()
        with self.assertRaisesRegex(ValueError, "physical events must be disjoint"):
            replace(
                manifest,
                training_physical_event_digests=(
                    manifest.holdout_cases[0].physical_event_digest,
                ),
            )

    def test_boundary_shift_cannot_hide_training_event_overlap(self) -> None:
        manifest = self.manifest()
        training_plan = replace(
            self.training_event_catalog_plan(),
            catalog_completion_deadline="2026-08-10T00:00:00Z",
        )
        holdout_event = self.event_catalog(1)
        shifted = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="shifted-training-copy",
            member_case_ids=("training-case",),
            member_full_analysis_input_digests=("8" * 64,),
            start_time="2026-08-09T00:05:00Z",
            end_time="2026-08-09T02:05:00Z",
            spatial_envelope_xy_m=(-1_000.0, 0.0, 101_000.0, 100_000.0),
            participating_radar_ids=holdout_event.participating_radar_ids,
            association_algorithm_digest=training_plan.association_algorithm_digest,
            adjudication_policy_digest=training_plan.adjudication_policy_digest,
            adjudicator_id=training_plan.adjudicator_id,
            adjudicator_private_key=self.regime_labeler_key(),
        )
        membership = promotion_module.PhysicalEventCaseSpatialEvidence(
            case_id="training-case",
            full_analysis_input_digest="8" * 64,
            physical_event_identity_digest=(
                shifted.physical_event_identity_digest
            ),
            observed_spatial_envelope_xy_m=(
                10_000.0,
                10_000.0,
                90_000.0,
                90_000.0,
            ),
            event_spatial_envelope_xy_m=shifted.spatial_envelope_xy_m,
            spatial_membership_rule_digest=(
                training_plan.spatial_membership_rule_digest
            ),
            source_object_evidence_digest="c" * 64,
            input_available_time="2026-08-09T00:00:00Z",
            spatial_reference_digest="7" * 64,
        )
        training_result = promotion_module.PhysicalEventCatalogResult.from_plan(
            training_plan,
            event_evidences=(shifted,),
            case_spatial_membership_evidences=(membership,),
            cataloged_at="2026-08-09T03:00:00Z",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        training_receipt = promotion_module.TrustedProcessStartReceipt.from_plan(
            training_plan,
            catalog_result_digest=training_result.result_digest,
            process_kind="candidate_training",
            subject_digests=("1" * 64, "2" * 64),
            process_algorithm_digest="3" * 64,
            job_id="shifted-training-job",
            started_at="2026-08-10T01:00:00Z",
            scheduler_private_key=self.regime_labeler_key(),
        )

        with self.assertRaisesRegex(ValueError, "association component"):
            replace(
                manifest,
                training_physical_event_digests=(
                    shifted.physical_event_identity_digest,
                ),
                training_physical_event_catalog_plan=training_plan,
                training_physical_event_catalog_result=training_result,
                candidate_training_started_at=training_receipt.started_at,
                candidate_training_start_receipt=training_receipt,
            )

    def test_physical_event_identity_is_not_a_free_form_label(self) -> None:
        original = self.event_catalog(1)
        renamed = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="renamed-training-copy",
            member_case_ids=("other-case",),
            member_full_analysis_input_digests=("9" * 64,),
            start_time=original.start_time,
            end_time=original.end_time,
            spatial_envelope_xy_m=original.spatial_envelope_xy_m,
            participating_radar_ids=original.participating_radar_ids,
            association_algorithm_digest=original.association_algorithm_digest,
            adjudication_policy_digest=original.adjudication_policy_digest,
            adjudicator_id=original.adjudicator_id,
            adjudicator_private_key=self.regime_labeler_key(),
        )

        self.assertEqual(
            renamed.physical_event_identity_digest,
            original.physical_event_identity_digest,
        )

    def test_candidate_training_event_lineage_is_a_signed_catalog_result(
        self,
    ) -> None:
        manifest = self.manifest()

        self.assertIsInstance(
            manifest.training_physical_event_catalog_plan,
            promotion_module.PhysicalEventCatalogPlan,
        )
        self.assertIsInstance(
            manifest.training_physical_event_catalog_result,
            promotion_module.PhysicalEventCatalogResult,
        )
        promotion_module.validate_physical_event_catalog_result(
            manifest.training_physical_event_catalog_result,
            manifest.training_physical_event_catalog_plan,
            candidate_scoring_started_at=manifest.candidate_training_started_at,
        )

    def test_event_case_spatial_membership_must_fit_the_event_envelope(self) -> None:
        event = self.event_catalog(1)
        with self.assertRaisesRegex(ValueError, "spatial envelope"):
            promotion_module.PhysicalEventCaseSpatialEvidence(
                case_id="case-1",
                full_analysis_input_digest="1" * 64,
                physical_event_identity_digest=(
                    event.physical_event_identity_digest
                ),
                observed_spatial_envelope_xy_m=(
                    200_000.0,
                    200_000.0,
                    210_000.0,
                    210_000.0,
                ),
                event_spatial_envelope_xy_m=event.spatial_envelope_xy_m,
                spatial_membership_rule_digest="4" * 64,
                source_object_evidence_digest="d" * 64,
                input_available_time="2026-08-09T00:00:00Z",
                spatial_reference_digest="7" * 64,
            )

    def test_required_metric_cell_must_be_non_inferior(self) -> None:
        requirement = promotion_module.RangeMetricRequirement(
            weather_regime="convective",
            range_regime="near_range",
            metric_name="soft_fss_error_35",
            lead_minutes=60,
            minimum_cases=1,
            minimum_physical_events=1,
            minimum_valid_area_km2=0.0,
            maximum_mean_normalized_degradation=0.0,
            maximum_harmful_fraction_upper_bound=0.5,
        )
        policy = replace(
            self.policy(),
            metric_scales=(
                PromotionMetricScale("log_echo_mse", 1.0, 0.01),
                PromotionMetricScale("soft_fss_error_35", 1.0, 0.01),
            ),
            required_range_metrics=(
                requirement,
                self.policy().required_range_metrics[1],
            ),
        )

        result = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -1.0,
                    range_change=-1.0,
                    secondary_range_change=0.05,
                ),
                self.evaluation(2, -0.3),
            ),
            policy,
        )

        self.assertNotIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )

    def test_required_metric_cell_requires_independent_events(self) -> None:
        requirement = promotion_module.RangeMetricRequirement(
            weather_regime="convective",
            range_regime="near_range",
            metric_name="log_echo_mse",
            lead_minutes=60,
            minimum_cases=1,
            minimum_physical_events=3,
            minimum_valid_area_km2=0.0,
            maximum_mean_normalized_degradation=1.0,
            maximum_harmful_fraction_upper_bound=1.0,
        )
        policy = replace(
            self.policy(),
            required_range_metrics=(requirement,),
        )

        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )

        self.assertNotIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )

    def test_sample_size_preflight_rejects_an_impossible_small_holdout(self) -> None:
        policy = replace(
            self.policy(),
            minimum_regime_classifier_accuracy_lower_bound=0.9,
            minimum_regime_classifier_recall_lower_bound=0.8,
            minimum_range_set_precision_lower_bound=0.95,
            minimum_range_set_recall_lower_bound=0.95,
            maximum_regime_classifier_false_routing_upper_bound=0.05,
            maximum_false_active_band_upper_bound=0.05,
            maximum_harmful_fraction=0.1,
            minimum_deployment_metric_cell_events=5,
        )
        small = promotion_module.promotion_sample_size_preflight(
            self.plan(),
            policy,
            available_physical_events=30,
        )
        sufficient = promotion_module.promotion_sample_size_preflight(
            self.plan(),
            policy,
            available_physical_events=small.required_physical_events,
        )

        self.assertFalse(small.feasible)
        self.assertGreater(small.required_physical_events, 30)
        self.assertTrue(sufficient.feasible)
        self.assertRegex(small.preflight_digest, r"^[0-9a-f]{64}$")

    def test_metric_cell_event_minimum_is_bound_into_policy_digest(self) -> None:
        policy = self.policy()

        self.assertNotEqual(
            policy.digest,
            replace(
                policy,
                minimum_deployment_metric_cell_events=(
                    policy.minimum_deployment_metric_cell_events + 1
                ),
            ).digest,
        )

    def test_band_skill_evidence_preserves_tail_and_event_diagnostics(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2),
                self.evaluation(2, -0.3),
            )
        )

        self.assertEqual(len(result.range_band_skill_inference_diagnostics), 2)
        for diagnostic in result.range_band_skill_inference_diagnostics:
            self.assertEqual(diagnostic[2], "cluster_bootstrap")
            self.assertEqual(diagnostic[3], 2)
            self.assertEqual(diagnostic[4], 256)
            self.assertGreaterEqual(diagnostic[5], 1.0)
            self.assertEqual(diagnostic[8], 1)
            self.assertEqual(diagnostic[9], 1)

    def test_missing_required_band_metric_prevents_certification(self) -> None:
        requirement = promotion_module.RangeMetricRequirement(
            weather_regime="convective",
            range_regime="near_range",
            metric_name="soft_fss_error_35",
            lead_minutes=60,
            minimum_cases=1,
            minimum_physical_events=1,
            minimum_valid_area_km2=1.0,
            maximum_mean_normalized_degradation=0.0,
            maximum_harmful_fraction_upper_bound=1.0,
        )
        policy = replace(
            self.policy(),
            metric_scales=(
                PromotionMetricScale("log_echo_mse", 1.0, 0.01),
                PromotionMetricScale("soft_fss_error_35", 1.0, 0.01),
            ),
            required_range_metrics=(requirement,),
        )

        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )

        self.assertNotIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )

    def test_range_object_component_uses_object_count_not_pixel_count(
        self,
    ) -> None:
        prior = SimpleNamespace(
            conditional_pit_residual_mean_abs=0.5,
            echo_intensity_nll=0.5,
            support_brier_score=0.1,
            echo_support_miss_score=0.1,
            clear_sky_false_echo_score=0.1,
            conditional_underdispersion_fraction=0.0,
            echo_sample_count=100,
            clear_sample_count=50,
        )
        state = SimpleNamespace(
            gaussian_nll=0.5,
            pit_residual_mean_abs=0.5,
            underdispersion_fraction=0.0,
            support_brier_score=0.1,
            echo_support_miss_score=0.1,
            false_support_score=0.1,
            valid_brier_score=0.1,
            sample_count=150,
            echo_sample_count=100,
            clear_sample_count=50,
        )

        _, _, _, counts = promotion_module._range_uncertainty_components(
            prior,
            prior,
            state,
            state,
            candidate_object_miss=0.1,
            parent_object_miss=0.1,
            object_count=3,
            candidate_state_object_miss=0.1,
            parent_state_object_miss=0.1,
            state_object_count=4,
        )

        self.assertEqual(dict(counts)["object_miss"], 3)
        self.assertEqual(dict(counts)["state_object_miss"], 4)

    def test_quantized_gaussian_upper_tail_has_finite_large_nll(self) -> None:
        nll, pit = promotion_module._quantized_gaussian_diagnostics(
            torch.tensor([-10.0], dtype=torch.float32),
            torch.tensor([0.1], dtype=torch.float32),
            torch.tensor([5.5], dtype=torch.float32),
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
            support_threshold_dbz=5.0,
            threshold_bin_convention="nearest_rounding_threshold_censor",
        )
        self.assertTrue(torch.all(torch.isfinite(nll)))
        self.assertTrue(torch.all(torch.isfinite(pit)))
        self.assertGreater(float(nll[0]), 1_000.0)

    def test_state_target_requires_v2_measurement_attestation(self) -> None:
        target_plan = self.plan().state_calibration_target_plans[0]
        wrong = VerificationBundle(
            frames_dbz=torch.tensor([[[10.0, 1.0], [10.0, 1.0]]]),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest=target_plan.source_identity_digest,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
            mask_policy_digest="f" * 64,
            censor_policy_digest=target_plan.censor_policy_digest,
            reflectivity_resolution_dbz=target_plan.reflectivity_resolution_dbz,
            quantization_origin_dbz=target_plan.quantization_origin_dbz,
            threshold_bin_convention=target_plan.threshold_bin_convention,
            floor_representation_contract_digest=(
                target_plan.floor_representation_contract_digest
            ),
            contract="radar-verification-bundle-v3",
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            NeuralPriorStateCalibrationTarget.from_verification_bundle(
                plan=target_plan,
                verification=wrong,
            )
        wrong_censor = replace(
            wrong,
            mask_policy_digest=target_plan.mask_policy_digest,
            censor_policy_digest="e" * 64,
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            NeuralPriorStateCalibrationTarget.from_verification_bundle(
                plan=target_plan,
                verification=wrong_censor,
            )

    def test_component_geometry_rejects_single_pixel_evidence(self) -> None:
        policy = replace(
            self.policy(),
            minimum_prior_echo_pixels_per_case=2,
            minimum_prior_echo_area_km2_per_case=1.0,
        )
        result = self.compute_with_policy(
            (
                self.evaluation(1, -0.2, prior_sample_count=2),
                self.evaluation(2, -0.3, prior_sample_count=2),
            ),
            policy,
        )
        self.assertIn("insufficient_component_samples", result.rejection_reasons)
        self.assertIn("insufficient_component_area", result.rejection_reasons)

    def test_missed_echo_objects_fail_the_probability_guard(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_echo_object_miss_score=1.0,
                    parent_prior_echo_object_miss_score=0.0,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    prior_echo_object_miss_score=1.0,
                    parent_prior_echo_object_miss_score=0.0,
                ),
            )
        )
        self.assertIn("unreliable_prior_uncertainty", result.rejection_reasons)
        self.assertIn("inferior_prior_uncertainty", result.rejection_reasons)

    def test_bootstrap_tail_resolution_fails_closed(self) -> None:
        policy = replace(
            self.policy(),
            bootstrap_samples=10,
            minimum_bootstrap_tail_replicates=1,
        )
        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )
        self.assertIn(
            "insufficient_bootstrap_tail_resolution",
            result.rejection_reasons,
        )

    def test_family_ten_with_one_thousand_bootstraps_is_rejected(self) -> None:
        plan = replace(
            self.plan(),
            candidate_family_digests=("c" * 64,)
            + tuple(character * 64 for character in "012345678"),
        )
        manifest = replace(
            self.manifest(),
            holdout_plan_digest=plan.plan_digest,
            candidate_scoring_start_receipt=self.scoring_start_receipt_for(
                plan.physical_event_catalog_plan,
                self.event_catalog_result(),
                subject_digests=plan.candidate_family_digests,
            ),
        )
        rebound: list[promotion_module.PriorHoldoutEvaluation] = []
        for evaluation in (
            self.evaluation(1, -0.2),
            self.evaluation(2, -0.3),
        ):
            values = {
                name: value
                for name, value in evaluation.__dict__.items()
                if name not in {"contract", "evaluation_digest"}
            }
            values["holdout_plan_digest"] = plan.plan_digest
            values["candidate_manifest_digest"] = manifest.manifest_digest
            rebound.append(
                promotion_module._new_prior_holdout_evaluation(**values)
            )
        policy = replace(
            self.policy(),
            approved_candidate_manifest_digests=(manifest.manifest_digest,),
            approved_holdout_plan_digests=(plan.plan_digest,),
            bootstrap_samples=1000,
            minimum_bootstrap_tail_replicates=20,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((policy.digest,)),
            content_digest="b" * 64,
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            result = compute_neural_prior_promotion(
                manifest,
                plan,
                tuple(rebound),
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )
        self.assertAlmostEqual(result.cluster_bootstrap_tail_replicates, 2.5)
        self.assertIn(
            "insufficient_bootstrap_tail_resolution",
            result.rejection_reasons,
        )

    def test_uncertified_regime_requires_parent_fallback(self) -> None:
        evidence = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        validate_neural_prior_promotion_applicability(
            evidence,
            regime="convective",
            range_regime="near_range",
        )
        with self.assertRaisesRegex(ValueError, "parent prior"):
            validate_neural_prior_promotion_applicability(
                evidence,
                regime="unseen",
                range_regime="far_range",
            )

    def test_classifier_attested_uncertified_regime_selects_parent(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((0.0, 12.0, -12.0), (0.0, 12.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unseen", "unknown"),
            range_regime_labels=("near_range", "unseen_range"),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        geometry, partition = self.holdout_range_geometry(1)
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=geometry.contract_digest,
        )

        classified = classifier.classify(frames, input_run=run)
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

        self.assertIs(selected, parent)
        self.assertEqual(selection.selected_role, "parent")
        self.assertEqual(selection.fallback_reason, "uncertified_regime")
        self.assertEqual(
            selection.full_analysis_input_digest,
            run.full_analysis_input_digest,
        )

    def test_regime_classifier_rehashes_mutable_execution_contract(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0), (12.0,)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unknown"),
            range_regime_labels=("near_range",),
            classifier_algorithm_digest="1" * 64,
        )
        classifier.classify(frames, input_run=run)
        classifier.regime_labels = ("stratiform", "unknown")

        with self.assertRaisesRegex(ValueError, "artifact changed"):
            classifier.classify(frames, input_run=run)

    def test_classifier_attested_certified_regime_selects_candidate(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0, -12.0), (12.0, 0.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unseen", "unknown"),
            range_regime_labels=("near_range", "unseen_range"),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        geometry, partition = self.holdout_range_geometry(1)
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=geometry.contract_digest,
        )

        classified = classifier.classify(frames, input_run=run)
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

        self.assertIs(selected, candidate)
        self.assertEqual(selection.selected_role, "candidate")
        self.assertEqual(selection.fallback_reason, "certified_candidate")
        deployment_artifact = json.loads(
            selection.deployment_decision_artifact_json
        )
        self.assertEqual(
            deployment_artifact["operational_grid_contract_digest"],
            partition.grid_contract_digest,
        )
        self.assertEqual(
            deployment_artifact["operational_frame_shape"],
            list(partition.masks[0].shape),
        )
        self.assertEqual(
            deployment_artifact["range_geometry_contract"],
            json.loads(
                json.dumps(
                    geometry.payload
                    | {"contract_digest": geometry.contract_digest}
                )
            ),
        )
        incomplete_artifact = dict(deployment_artifact)
        incomplete_artifact.pop("range_geometry_contract")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            promotion_module.validate_neural_prior_deployment_decision_artifact(
                json.dumps(
                    incomplete_artifact,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        with self.assertRaisesRegex(ValueError, "current forecast run"):
            promotion_module.validate_neural_prior_deployment_decision_artifact(
                selection.deployment_decision_artifact_json,
                expected_operational_grid_contract_digest="3" * 64,
                expected_operational_frame_shape=tuple(partition.masks[0].shape),
            )
        deployment_artifact["operational_grid_contract_digest"] = "3" * 64
        changed_artifact_json = json.dumps(
            deployment_artifact,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.assertRaisesRegex(ValueError, "operational grid"):
            promotion_module.validate_neural_prior_deployment_decision_artifact(
                changed_artifact_json
            )
        changed_shape_artifact = json.loads(
            selection.deployment_decision_artifact_json
        )
        changed_shape_artifact["operational_frame_shape"] = [4, 4]
        with self.assertRaisesRegex(ValueError, "operational grid"):
            promotion_module.validate_neural_prior_deployment_decision_artifact(
                json.dumps(
                    changed_shape_artifact,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

        unapproved = replace(deployment_policy, minimum_regime_confidence=0.01)
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "unapproved"):
            promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                partition,
                unapproved,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )
        changed_trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((unapproved.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [unapproved.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=changed_trust,
        ):
            _, changed_selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                partition,
                unapproved,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )
        self.assertNotEqual(
            changed_selection.selection_digest,
            selection.selection_digest,
        )
        self.assertEqual(
            changed_selection.deployment_policy_digest,
            unapproved.policy_digest,
        )
        tampered = replace(deployment_policy, minimum_regime_confidence=0.01)
        object.__setattr__(
            tampered,
            "policy_digest",
            deployment_policy.policy_digest,
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "policy digest mismatch"):
            promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                partition,
                tampered,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

        external_geometry, external_partition = self.deployment_range_geometry(
            labels=classifier.range_regime_labels,
            include_second_band=False,
        )
        external_policy = replace(
            deployment_policy,
            range_geometry_contract_digest=external_geometry.contract_digest,
        )
        external_trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((external_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [external_policy.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=external_trust,
        ):
            external_selected, external_selection = (
                promotion_module._select_deployed_prior(
                    candidate,
                    parent,
                    evidence,
                    classified,
                    external_partition,
                    external_policy,
                    range_geometry_contract=external_geometry,
                    operational_grid_contract_digest=(
                        external_partition.grid_contract_digest
                    ),
                    operational_frame_shape=tuple(
                        external_partition.masks[0].shape
                    ),
                    policy_trust_store_path=(
                        "/etc/advar/deployment-policies.json"
                    ),
                )
            )
        self.assertIs(external_selected, parent)
        self.assertEqual(
            external_selection.fallback_reason,
            "uncertified_range_geometry",
        )

    def test_all_active_range_bands_must_be_certified(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0, -12.0), (12.0, 12.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "stratiform", "unknown"),
            range_regime_labels=("near_range", "far_range"),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        geometry, partition = self.deployment_range_geometry(
            labels=classifier.range_regime_labels,
            include_second_band=True,
        )
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=geometry.contract_digest,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classifier.classify(frames, input_run=run),
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )
        self.assertIs(selected, parent)
        self.assertEqual(selection.fallback_reason, "uncertified_range_geometry")

    def test_current_physical_range_partition_controls_deployment(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0, -12.0), (12.0, 0.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unseen", "unknown"),
            range_regime_labels=("near_range", "far_range"),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        grid_x_m = torch.tensor([[0.0, 20_000.0], [40_000.0, 80_000.0]])
        grid_y_m = torch.zeros_like(grid_x_m)
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="a" * 64,
            radar_site_location_digest="a" * 64,
            grid_contract_digest="2" * 64,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range", "far_range"),
            radial_distance_edges_m=(0.0, 30_000.0, 100_000.0),
            horizontal_range_rule_digest="b" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(grid_x_m),
            grid_y_m_digest=promotion_module.tensor_digest(grid_y_m),
        )
        partition = promotion_module.resolve_range_geometry(
            geometry,
            grid_x_m=grid_x_m,
            grid_y_m=grid_y_m,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=geometry.contract_digest,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classifier.classify(frames, input_run=run),
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

        self.assertIs(selected, parent)
        self.assertEqual(selection.fallback_reason, "uncertified_range_geometry")
        self.assertEqual(
            selection.range_partition_evidence_digest,
            partition.evidence_digest,
        )

    def test_regime_classifier_evidence_is_bound_to_current_input(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0, -12.0), (12.0, 0.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "stratiform", "unknown"),
            range_regime_labels=("near_range", "far_range"),
            classifier_algorithm_digest="1" * 64,
        )

        with self.assertRaisesRegex(ValueError, "input or artifact changed"):
            classifier.classify(frames + 1.0, input_run=run)

    def test_classifier_holdout_rejects_constant_false_routing(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2),
                self.evaluation(
                    2,
                    -0.3,
                    classified_regime="convective",
                    classified_range_regimes=("near_range",),
                    classifier_reference_agreement=False,
                ),
            )
        )
        self.assertFalse(result.regime_classifier_validated)
        self.assertIn(
            "unreliable_regime_classifier",
            result.rejection_reasons,
        )

    def test_all_active_range_classifier_fails_set_precision(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    classified_range_regimes=("near_range", "far_range"),
                ),
                self.evaluation(
                    2,
                    -0.3,
                    classified_range_regimes=("near_range", "far_range"),
                ),
            )
        )

        self.assertFalse(result.regime_classifier_validated)
        self.assertAlmostEqual(result.range_set_precision, 0.5)
        self.assertEqual(result.range_exact_set_accuracy, 0.0)
        self.assertIn("unreliable_range_classifier", result.rejection_reasons)

    def test_range_band_harm_is_not_hidden_by_whole_domain_skill(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2, range_change=-0.2),
                self.evaluation(2, -0.3, range_change=1.1),
            )
        )

        self.assertTrue(result.eligible)
        self.assertIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )
        self.assertNotIn(
            ("stratiform", "far_range"),
            result.certified_applicability_regime_groups,
        )
        far_bounds = next(
            item
            for item in result.range_band_skill_bounds
            if item[:2] == ("stratiform", "far_range")
        )
        self.assertLess(far_bounds[4], 0.0)

    def test_consistent_subthreshold_band_harm_fails_certification(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -1.0, range_change=-1.0),
                self.evaluation(2, -1.0, range_change=0.2),
            )
        )

        self.assertTrue(result.eligible)
        self.assertIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )
        self.assertNotIn(
            ("stratiform", "far_range"),
            result.certified_applicability_regime_groups,
        )

    def test_range_band_preserves_absolute_candidate_and_parent_scores(self) -> None:
        band = self.evaluation(
            1,
            -0.2,
            prior_support_brier_score=0.9,
            parent_prior_support_brier_score=0.8,
        ).range_band_evaluations[0]

        self.assertEqual(
            dict(band.candidate_uncertainty_component_scores)["support"],
            0.9,
        )
        self.assertEqual(
            dict(band.parent_uncertainty_component_scores)["support"],
            0.8,
        )
        self.assertEqual(
            dict(band.candidate_uncertainty_component_scores)["pit_residual"],
            0.5,
        )
        self.assertEqual(
            dict(band.candidate_uncertainty_component_scores)[
                "state_pit_residual"
            ],
            0.5,
        )
        self.assertAlmostEqual(band.component_difference("support"), 0.1)

    def test_absolute_bad_band_is_not_certified_when_parent_is_equally_bad(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2),
                self.evaluation(
                    2,
                    -0.3,
                    range_candidate_support_brier_score=0.9,
                    range_parent_support_brier_score=0.9,
                ),
            )
        )

        self.assertTrue(result.eligible)
        self.assertNotIn(
            ("stratiform", "far_range"),
            result.certified_applicability_regime_groups,
        )

    def test_large_mask_cannot_hide_tiny_valid_band_sample(self) -> None:
        policy = replace(
            self.policy(),
            minimum_range_metric_valid_area_km2=0.5,
            minimum_range_probability_valid_area_km2=0.5,
            minimum_range_state_valid_area_km2=0.5,
            minimum_range_component_samples=4,
            minimum_range_echo_objects=1,
            minimum_range_state_echo_objects=1,
        )
        result = self.compute_with_policy(
            (
                self.evaluation(1, -0.2),
                self.evaluation(
                    2,
                    -0.3,
                    range_component_sample_count=1,
                    range_valid_area_km2=0.25,
                ),
            ),
            policy,
        )

        self.assertTrue(result.eligible)
        self.assertNotIn(
            ("stratiform", "far_range"),
            result.certified_applicability_regime_groups,
        )

    def test_classifier_training_input_cannot_reappear_under_new_case_name(
        self,
    ) -> None:
        completed = self.completed_case(1)
        classifier = replace(
            self.plan().regime_classifier_manifests[0],
            training_case_ids=("renamed-training-case",),
            training_storm_ids=("renamed-training-storm",),
            training_input_bundle_digests=(completed.input_bundle_digest,),
        )

        with self.assertRaisesRegex(ValueError, "classifier training inputs"):
            promotion_module._validate_classifier_holdout_independence(
                classifier,
                (completed,),
            )

    def test_classifier_training_event_cannot_overlap_holdout_cycle(self) -> None:
        completed = self.completed_case(1)
        classifier = replace(
            self.plan().regime_classifier_manifests[0],
            training_physical_event_digests=(completed.physical_event_digest,),
        )

        with self.assertRaisesRegex(ValueError, "physical events overlap"):
            promotion_module._validate_classifier_holdout_independence(
                classifier,
                (completed,),
            )

    def test_prospective_plan_cannot_store_future_weather_truth(self) -> None:
        plan = self.plan()
        cases = (
            replace(plan.cases[0], regime="convective"),
            plan.cases[1],
        )

        with self.assertRaisesRegex(ValueError, "weather truth"):
            replace(plan, cases=cases)

        storm_cases = (
            replace(plan.cases[0], storm_id="future-storm"),
            plan.cases[1],
        )
        with self.assertRaisesRegex(ValueError, "storm identity"):
            replace(plan, cases=storm_cases)

    def test_regime_reference_signature_binds_observed_label(self) -> None:
        plan = self.plan().regime_reference_plans[0]
        evidence = self.reference_evidence(1)
        changed = object.__new__(promotion_module.RegimeReferenceEvidence)
        for name, value in evidence.__dict__.items():
            object.__setattr__(
                changed,
                name,
                "stratiform" if name == "observed_regime" else value,
            )

        with self.assertRaisesRegex(ValueError, "evidence|signature"):
            promotion_module.validate_regime_reference_evidence(changed, plan)

    def test_classifier_eighteen_of_twenty_does_not_certify_at_point_threshold(
        self,
    ) -> None:
        values = [1.0] * 18 + [0.0] * 2
        clusters = [
            (f"storm-{index}", f"2026-07-{index + 1:02d}", "radar-1")
            for index in range(20)
        ]

        lower, _ = promotion_module._event_fractional_rate_interval(
            values,
            clusters,
            self.policy(),
            family_size=1,
        )

        self.assertLess(lower, 0.9)

    def test_classifier_training_storm_cannot_overlap_holdout(self) -> None:
        plan = self.plan()
        overlapping = replace(
            plan.regime_classifier_manifests[0],
            training_storm_ids=(plan.cases[0].storm_id,),
        )

        with self.assertRaisesRegex(ValueError, "classifier training overlaps"):
            replace(plan, regime_classifier_manifests=(overlapping,))

    def test_promotion_rejects_classifier_outside_preregistered_family(self) -> None:
        plan = self.plan()
        manifest = self.manifest()
        policy = replace(
            self.policy(),
            deployment_regime_classifier_digest="f" * 64,
            deployment_regime_classifier_manifest_digest="0" * 64,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((policy.digest,)),
            content_digest="b" * 64,
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "not preregistered"):
            compute_neural_prior_promotion(
                manifest,
                plan,
                (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_ambiguous_current_weather_branch_falls_back_to_parent(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((0.01, 0.0), (12.0,)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unknown"),
            range_regime_labels=("near_range",),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        grid_x_m = torch.zeros((2, 2))
        grid_y_m = torch.zeros_like(grid_x_m)
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="a" * 64,
            radar_site_location_digest="a" * 64,
            grid_contract_digest="2" * 64,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range",),
            radial_distance_edges_m=(0.0, 100_000.0),
            horizontal_range_rule_digest="b" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(grid_x_m),
            grid_y_m_digest=promotion_module.tensor_digest(grid_y_m),
        )
        partition = promotion_module.resolve_range_geometry(
            geometry,
            grid_x_m=grid_x_m,
            grid_y_m=grid_y_m,
        )
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=geometry.contract_digest,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classifier.classify(frames, input_run=run),
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

        self.assertIs(selected, parent)
        self.assertEqual(selection.fallback_reason, "ambiguous_classifier_branch")

    def test_deployment_requires_preregistered_ood_validation(self) -> None:
        policy = replace(self.policy(), minimum_regime_classifier_ood_cases=1)
        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )
        self.assertFalse(result.deployment_eligible)
        self.assertIn("unreliable_regime_classifier", result.rejection_reasons)

    def test_partial_regime_certification_only_falls_back_for_missing_group(
        self,
    ) -> None:
        policy = replace(
            self.policy(),
            minimum_prior_echo_cases=1,
            minimum_prior_echo_clusters=1,
        )
        result = self.compute_with_policy(
            (
                self.evaluation(1, -0.2),
                self.evaluation(2, -0.3, echo_available=False),
            ),
            policy,
        )
        self.assertTrue(result.eligible)
        self.assertIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )
        self.assertNotIn(
            ("stratiform", "far_range"),
            result.certified_applicability_regime_groups,
        )

    def test_regime_specific_uncertainty_regression_is_not_averaged_away(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_support_brier_score=0.2,
                    parent_prior_support_brier_score=0.1,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    prior_support_brier_score=0.0,
                    parent_prior_support_brier_score=0.1,
                ),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("inferior_prior_uncertainty", result.rejection_reasons)

    def test_self_selected_one_percent_validity_blocks_promotion(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_candidate_valid_fraction=0.01,
                    prior_candidate_valid_area_km2=0.04,
                ),
                self.evaluation(2, -0.3),
            )
        )

        self.assertFalse(result.eligible)
        self.assertIn("unreliable_prior_uncertainty", result.rejection_reasons)

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
        self.assertFalse(decoded[0].content_digest_verified)
        self.assertFalse(decoded[0].statistical_reuse_permitted)
        with self.assertRaisesRegex(ValueError, "audit-only"):
            self.compute(decoded)

    def test_v10_promotion_evidence_remains_audit_only(self) -> None:
        current = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        payload = current._payload()
        for name in (
            "regime_classifier_accuracy_lower_bound",
            "minimum_regime_classifier_recall_lower_bound",
            "regime_classifier_false_routing_upper_bound",
            "regime_classifier_cluster_count",
            "range_set_precision_lower_bound",
            "range_set_recall_lower_bound",
            "range_false_active_band_upper_bound",
            "range_band_skill_bounds",
        ):
            payload.pop(name)
        payload["contract"] = "neural-prior-promotion-evidence-v10"
        digest = promotion_module.json_digest(payload)

        audit = promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV10(
            promotion_evidence_digest=digest,
            payload_json=json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        self.assertEqual(audit.promotion_evidence_digest, digest)
        self.assertFalse(hasattr(audit, "eligible"))

    def test_v11_promotion_evidence_remains_audit_only(self) -> None:
        current = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        payload = current._payload()
        payload.pop("range_band_skill_inference_diagnostics")
        payload["contract"] = "neural-prior-promotion-evidence-v11"
        digest = promotion_module.json_digest(payload)

        audit = promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV11(
            promotion_evidence_digest=digest,
            payload_json=json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        self.assertEqual(audit.promotion_evidence_digest, digest)
        self.assertFalse(hasattr(audit, "eligible"))

    def test_v6_hurdle_evaluation_remains_content_verified_audit_only(
        self,
    ) -> None:
        payload = ledger_module._evaluation_audit_payload(
            self.evaluation(1, -0.2)
        )
        payload["contract"] = "prior-holdout-evaluation-v6"
        payload.pop("prior_echo_intensity_status")
        payload.pop("prior_clear_sky_status")
        normalized = dict(payload)
        normalized.pop("evaluation_digest")
        for name, value in tuple(normalized.items()):
            if isinstance(value, dict) and value.get("kind") == "tensor":
                normalized[name] = value["digest"]
        payload["evaluation_digest"] = promotion_module.json_digest(normalized)

        decoded = ledger_module._decode_evaluation_audit_payloads([payload])

        self.assertIsInstance(
            decoded[0],
            ledger_module.LegacyPromotionEvaluationAudit,
        )
        self.assertTrue(decoded[0].content_digest_verified)
        self.assertFalse(decoded[0].statistical_reuse_permitted)

    def test_direct_evaluation_construction_is_disabled(self) -> None:
        with self.assertRaisesRegex(TypeError, "from_forecasts"):
            promotion_module.PriorHoldoutEvaluation()

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

    def test_prospective_decision_direct_construction_is_disabled(self) -> None:
        with self.assertRaisesRegex(TypeError, "from_policy"):
            ProspectiveInterventionDecision()
        with self.assertRaisesRegex(TypeError, "from_decision"):
            RealizedInterventionReceipt()

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
        grid = SimpleNamespace(
            valid_times=planned_input.valid_times,
            cell_area_m2=1_000_000.0,
        )
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
            full_analysis_input_digest=case.full_analysis_input_digest,
            fixed_input_context_digest=case.fixed_input_context_digest,
            observation_quality_weight_digest=(
                case.observation_quality_weight_digest
            ),
            observation_std_dbz_digest=case.observation_std_dbz_digest,
            input_plan_digest=case.input_plan_digest,
            config=SimpleNamespace(digest="3" * 64, interval_minutes=10),
            analysis_config_digest="4" * 64,
            operational_calibration_manifest_digest="5" * 64,
            operational_data_identity_json=data_identity.json,
            operational_data_identity_digest=data_identity.digest,
            prior_model_contract_digest=manifest.model_contract_digest,
            prior_feature_schema_digest=manifest.feature_schema_digest,
            prior_inference_algorithm_digest="8" * 64,
            prior_numerical_runtime_digest="4" * 64,
            prior_dependency="radar_dependent",
            input_plan_json=planned_input.json,
            input_plan_resolution_digest=case.input_plan_resolution_digest,
        )
        candidate_app = SimpleNamespace(
            application_digest=case.candidate_prior_application_digest,
            initial_background_dbz=torch.zeros((2, 2)),
            state_background_dbz=torch.zeros((2, 2)),
            std_dbz=torch.ones((2, 2)),
            state_std_dbz=torch.ones((2, 2)),
            valid_mask=torch.tensor(
                [[True, False], [False, False]], dtype=torch.bool
            ),
            support_probability=torch.zeros((2, 2)),
            state_support_probability=torch.zeros((2, 2)),
            state_valid_probability=torch.tensor(
                [[1.0, 0.0], [0.0, 0.0]]
            ),
            truncated_location_dbz=torch.zeros((2, 2)),
            truncated_scale_dbz=torch.ones((2, 2)),
            event_probability=torch.zeros((2, 2)),
            inference_evidence=SimpleNamespace(
                evidence_digest=case.candidate_inference_evidence_digest,
                inference_algorithm_digest="8" * 64,
                numerical_runtime_digest="4" * 64,
                dependency="radar_dependent",
                input_bundle_digest=case.input_bundle_digest,
                full_analysis_input_digest=case.full_analysis_input_digest,
                input_frames_digest=promotion_module.tensor_digest(torch.zeros(3, 2, 2)),
                execution_contract_digest=manifest.candidate_prior_digest,
                neural_prior_digest=manifest.candidate_prior_digest,
                model_contract_digest=manifest.model_contract_digest,
                feature_schema_digest=manifest.feature_schema_digest,
                training_manifest_digest=(manifest.candidate_training_manifest_digest),
                uncertainty_contract="model_spatial",
                probability_contract_digest=(
                    self.probability_contract().contract_digest
                ),
                support_event_digest=(
                    self.probability_contract().support_event_digest
                ),
                state_contract_digest=self.state_contract().contract_digest,
                prior_output_valid_time="2026-08-09T00:00:00Z",
                feature_source_valid_times=planned_input.valid_times,
                feature_source_identity_digests=("a" * 64,),
                feature_exclusion_contract_digest="5" * 64,
                feature_exclusion_mask_digest=promotion_module.tensor_digest(
                    torch.ones((3, 2, 2), dtype=torch.bool)
                ),
            ),
        )
        parent_app = SimpleNamespace(
            application_digest=case.parent_prior_application_digest,
            initial_background_dbz=torch.zeros((2, 2)),
            state_background_dbz=torch.zeros((2, 2)),
            std_dbz=torch.ones((2, 2)),
            state_std_dbz=torch.ones((2, 2)),
            valid_mask=torch.ones((2, 2), dtype=torch.bool),
            support_probability=torch.zeros((2, 2)),
            state_support_probability=torch.zeros((2, 2)),
            state_valid_probability=torch.ones((2, 2)),
            truncated_location_dbz=torch.zeros((2, 2)),
            truncated_scale_dbz=torch.ones((2, 2)),
            event_probability=torch.zeros((2, 2)),
            inference_evidence=SimpleNamespace(
                evidence_digest=case.parent_inference_evidence_digest,
                inference_algorithm_digest="8" * 64,
                numerical_runtime_digest="4" * 64,
                dependency="radar_dependent",
                input_bundle_digest=case.input_bundle_digest,
                full_analysis_input_digest=case.full_analysis_input_digest,
                input_frames_digest=promotion_module.tensor_digest(torch.zeros(3, 2, 2)),
                execution_contract_digest=manifest.parent_prior_digest,
                neural_prior_digest=manifest.parent_prior_digest,
                model_contract_digest=manifest.model_contract_digest,
                feature_schema_digest=manifest.feature_schema_digest,
                training_manifest_digest=manifest.parent_training_manifest_digest,
                uncertainty_contract="model_spatial",
                probability_contract_digest=(
                    self.probability_contract().contract_digest
                ),
                support_event_digest=(
                    self.probability_contract().support_event_digest
                ),
                state_contract_digest=self.state_contract().contract_digest,
                prior_output_valid_time="2026-08-09T00:00:00Z",
                feature_source_valid_times=planned_input.valid_times,
                feature_source_identity_digests=("a" * 64,),
                feature_exclusion_contract_digest="5" * 64,
                feature_exclusion_mask_digest=promotion_module.tensor_digest(
                    torch.ones((3, 2, 2), dtype=torch.bool)
                ),
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
            valid_mask=torch.ones((6, 2, 2), dtype=torch.bool),
            frames_dbz=torch.zeros((6, 2, 2)),
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
            numerical_runtime_digest="4" * 64,
            feature_exclusion_mask=torch.ones((3, 2, 2), dtype=torch.bool),
            state_contract=self.state_contract(),
            probability_contract=self.probability_contract(),
        )
        parent_runner = SimpleNamespace(
            reproduce=Mock(),
            inference_algorithm_digest="8" * 64,
            numerical_runtime_digest="4" * 64,
            feature_exclusion_mask=torch.ones((3, 2, 2), dtype=torch.bool),
            state_contract=self.state_contract(),
            probability_contract=self.probability_contract(),
        )
        regime_evidence = SimpleNamespace(
            validate_integrity=Mock(),
            full_analysis_input_digest=case.full_analysis_input_digest,
            input_frames_digest=promotion_module.tensor_digest(
                torch.zeros((3, 2, 2))
            ),
            classifier_digest="e" * 64,
            evidence_digest="f" * 64,
            numerical_runtime_digest=(
                plan.regime_classifier_manifests[0].numerical_runtime_digest
            ),
            input_dtype=str(torch.float32),
            input_device="cpu",
            regime=case.regime,
            active_range_regimes=(case.range_regime,),
            regime_confidence=1.0,
            range_regime_confidence=1.0,
            regime_entropy=0.0,
            is_ood=False,
            weather_top1_top2_gap=1.0,
            minimum_range_presence_margin=0.5,
        )
        regime_classifier = SimpleNamespace(
            classifier_digest=regime_evidence.classifier_digest,
            numerical_runtime_digest=regime_evidence.numerical_runtime_digest,
            classify=Mock(return_value=regime_evidence),
        )
        candidate_weights = torch.full((1, 2, 2), 0.5)
        parent_weights = torch.ones((1, 2, 2))
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
            evaluation = promotion_module.PriorHoldoutEvaluation.from_forecasts(
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
                uncertainty_target=self.uncertainty_target(1),
                state_calibration_target=self.state_target(1),
                regime_classifier=regime_classifier,
                regime_classifier_manifest=(
                    plan.regime_classifier_manifests[0]
                ),
                range_grid_x_m=torch.zeros((2, 2)),
                range_grid_y_m=torch.zeros((2, 2)),
            )
        self.assertAlmostEqual(float(evaluation.metric_change[0, 0]), -0.2)
        self.assertAlmostEqual(float(evaluation.end_to_end_metric_change[0, 0]), -0.25)
        self.assertEqual(evaluation.prior_uncertainty_sample_count, 4)
        self.assertAlmostEqual(evaluation.prior_candidate_valid_fraction, 0.25)
        self.assertAlmostEqual(evaluation.prior_candidate_valid_area_km2, 1.0)
        self.assertAlmostEqual(evaluation.prior_abstention_increase_vs_parent, 0.75)
        candidate_runner.reproduce.assert_called_once()
        parent_runner.reproduce.assert_called_once()

        candidate_runner.probability_contract = NeuralPriorProbabilityContract(
            support_threshold_dbz=35.0,
            support_product_digest="6" * 64,
            qc_pipeline_digest="9" * 64,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
        )
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ), self.assertRaisesRegex(ValueError, "probability event"):
            promotion_module.PriorHoldoutEvaluation.from_forecasts(
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
                uncertainty_target=self.uncertainty_target(1),
                state_calibration_target=self.state_target(1),
                regime_classifier=regime_classifier,
                regime_classifier_manifest=(
                    plan.regime_classifier_manifests[0]
                ),
                range_grid_x_m=torch.zeros((2, 2)),
                range_grid_y_m=torch.zeros((2, 2)),
            )
        candidate_runner.probability_contract = self.probability_contract()

        parent_run.observation_quality_weight_digest = "0" * 64
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ), self.assertRaisesRegex(ValueError, "holdout inputs disagree"):
            promotion_module.PriorHoldoutEvaluation.from_forecasts(
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
                uncertainty_target=self.uncertainty_target(1),
                state_calibration_target=self.state_target(1),
                regime_classifier=regime_classifier,
                regime_classifier_manifest=plan.regime_classifier_manifests[0],
                range_grid_x_m=torch.zeros((2, 2)),
                range_grid_y_m=torch.zeros((2, 2)),
            )
        parent_run.observation_quality_weight_digest = (
            case.observation_quality_weight_digest
        )
        parent_run.observation_std_dbz_digest = "0" * 64
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ), self.assertRaisesRegex(ValueError, "holdout inputs disagree"):
            promotion_module.PriorHoldoutEvaluation.from_forecasts(
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
                uncertainty_target=self.uncertainty_target(1),
                state_calibration_target=self.state_target(1),
                regime_classifier=regime_classifier,
                regime_classifier_manifest=plan.regime_classifier_manifests[0],
                range_grid_x_m=torch.zeros((2, 2)),
                range_grid_y_m=torch.zeros((2, 2)),
            )
        parent_run.observation_std_dbz_digest = case.observation_std_dbz_digest

        candidate_app.inference_evidence.prior_output_valid_time = (
            "2026-08-09T01:00:00Z"
        )
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "target time"):
                promotion_module.PriorHoldoutEvaluation.from_forecasts(
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
                    uncertainty_target=self.uncertainty_target(1),
                    state_calibration_target=self.state_target(1),
                    regime_classifier=regime_classifier,
                    regime_classifier_manifest=(
                        plan.regime_classifier_manifests[0]
                    ),
                    range_grid_x_m=torch.zeros((2, 2)),
                    range_grid_y_m=torch.zeros((2, 2)),
                )

        candidate_app.inference_evidence.prior_output_valid_time = (
            "2026-08-09T00:00:00Z"
        )
        candidate_app.inference_evidence.feature_source_identity_digests = (
            "6" * 64,
        )
        parent_app.inference_evidence.feature_source_identity_digests = (
            "6" * 64,
        )
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "visible to the features"):
                promotion_module.PriorHoldoutEvaluation.from_forecasts(
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
                    uncertainty_target=self.uncertainty_target(1),
                    state_calibration_target=self.state_target(1),
                    regime_classifier=regime_classifier,
                    regime_classifier_manifest=(
                        plan.regime_classifier_manifests[0]
                    ),
                    range_grid_x_m=torch.zeros((2, 2)),
                    range_grid_y_m=torch.zeros((2, 2)),
                )

    def test_operational_range_geometry_must_match_current_radar_site(self) -> None:
        identity = promotion_module.OperationalDataIdentity(
            radar_class="single-site",
            qc_pipeline_digest="1" * 64,
            observation_error_model_digest="2" * 64,
            background_model_digest="3" * 64,
            radar_site_digest="a" * 64,
            radar_site_location_digest="b" * 64,
            radar_source_contract_digest="c" * 64,
        )
        frames = torch.zeros((3, 2, 2))
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-09T00:00:00Z",
                "2026-08-09T00:10:00Z",
                "2026-08-09T00:20:00Z",
            ),
            dx_m=1_000.0,
            dy_m=1_000.0,
            projection="EPSG:3857",
            grid_hash="4" * 64,
        )
        run = replace(ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
            grid_time_contract=grid,
        ),
            operational_data_identity_json=identity.json,
            operational_data_identity_digest=identity.digest,
        )
        coordinates = torch.zeros((2, 2))
        geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="d" * 64,
            radar_site_location_digest="e" * 64,
            grid_contract_digest=grid.digest,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range",),
            radial_distance_edges_m=(0.0, 100_000.0),
            horizontal_range_rule_digest="e" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(coordinates),
            grid_y_m_digest=promotion_module.tensor_digest(coordinates),
        )

        with self.assertRaisesRegex(ValueError, "radar site"):
            promotion_module.infer_deployed_neural_prior(
                frames,
                input_run=run,
                candidate_runner=Mock(),
                parent_runner=Mock(),
                promotion_evidence=Mock(),
                regime_classifier=Mock(),
                range_geometry_contract=geometry,
                grid_x_m=coordinates,
                grid_y_m=coordinates,
                policy=Mock(),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

    def test_event_catalog_append_rejects_future_catalog_time(self) -> None:
        plan = self.plan()
        future_result = promotion_module.PhysicalEventCatalogResult.from_plan(
            plan.physical_event_catalog_plan,
            event_evidences=(self.event_catalog(1), self.event_catalog(2)),
            case_spatial_membership_evidences=(
                self.event_spatial_evidence(
                    self.event_catalog(1),
                    case_id="case-1",
                    full_analysis_input_digest="1" * 64,
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    self.event_catalog(2),
                    case_id="case-2",
                    full_analysis_input_digest="2" * 64,
                    source_object_evidence_digest="b" * 64,
                ),
            ),
            cataloged_at="2026-08-10T23:00:00Z",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest, plan_id, plan_json, policy_digest, "
                    "trust_store_digest, registered_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan.plan_digest,
                        plan.plan_id,
                        json.dumps(asdict(plan), sort_keys=True),
                        "6" * 64,
                        "7" * 64,
                        plan.registered_at,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
            with patch.object(
                ledger_module,
                "datetime",
                wraps=datetime,
            ) as trusted_datetime:
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-10T22:00:00+00:00"
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "future|trusted ledger time",
                ):
                    ledger.append_physical_event_catalog_result(plan, future_result)

    def test_event_member_input_time_must_match_preregistered_plan(self) -> None:
        plan = self.plan()
        original = self.event_catalog_result()
        changed_membership = replace(
            original.case_spatial_membership_evidences[0],
            input_available_time="2026-08-09T00:01:00Z",
        )
        changed_result = promotion_module.PhysicalEventCatalogResult.from_plan(
            plan.physical_event_catalog_plan,
            event_evidences=original.event_evidences,
            case_spatial_membership_evidences=(
                changed_membership,
                original.case_spatial_membership_evidences[1],
            ),
            cataloged_at=original.cataloged_at,
            adjudicator_private_key=self.regime_labeler_key(),
        )
        changed_manifest = replace(
            self.manifest(),
            physical_event_catalog_result=changed_result,
            candidate_scoring_start_receipt=self.scoring_start_receipt_for(
                plan.physical_event_catalog_plan,
                changed_result,
            ),
        )

        with self.assertRaisesRegex(ValueError, "input availability"):
            promotion_module._validate_physical_event_catalogs_against_plan(
                changed_manifest,
                plan,
            )

    def test_split_events_with_an_association_edge_are_rejected(self) -> None:
        first = self.event_catalog(1)
        split_second = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="split-copy",
            member_case_ids=("case-2",),
            member_full_analysis_input_digests=("2" * 64,),
            start_time="2026-08-09T01:00:00Z",
            end_time="2026-08-09T03:00:00Z",
            spatial_envelope_xy_m=(1_000.0, 1_000.0, 99_000.0, 99_000.0),
            participating_radar_ids=("radar-1",),
            association_algorithm_digest="3" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        with self.assertRaisesRegex(ValueError, "association graph|connected component"):
            promotion_module.PhysicalEventCatalogResult.from_plan(
                self.event_catalog_plan(),
                event_evidences=(first, split_second),
                case_spatial_membership_evidences=(
                    self.event_spatial_evidence(
                        first,
                        case_id="case-1",
                        full_analysis_input_digest="1" * 64,
                        source_object_evidence_digest="a" * 64,
                    ),
                    self.event_spatial_evidence(
                        split_second,
                        case_id="case-2",
                        full_analysis_input_digest="2" * 64,
                        source_object_evidence_digest="b" * 64,
                    ),
                ),
                cataloged_at="2026-08-10T03:00:00Z",
                adjudicator_private_key=self.regime_labeler_key(),
            )

    def test_required_metric_cell_checks_end_to_end_non_inferiority(self) -> None:
        requirement = promotion_module.RangeMetricRequirement(
            weather_regime="convective",
            range_regime="near_range",
            metric_name="log_echo_mse",
            lead_minutes=60,
            minimum_cases=1,
            minimum_physical_events=1,
            minimum_valid_area_km2=0.0,
            maximum_mean_normalized_degradation=1.0,
            maximum_harmful_fraction_upper_bound=1.0,
            maximum_end_to_end_mean_normalized_degradation=0.0,
            maximum_end_to_end_harmful_fraction_upper_bound=0.5,
        )
        policy = replace(
            self.policy(),
            required_range_metrics=(
                requirement,
                self.policy().required_range_metrics[1],
            ),
            bootstrap_samples=512,
        )

        result = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    range_change=0.0,
                    range_end_to_end_change=0.2,
                ),
                self.evaluation(2, -0.3),
            ),
            policy,
        )

        self.assertNotIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )
        self.assertEqual(len(result.range_metric_end_to_end_cell_bounds), 2)

    def test_scoring_start_requires_registered_catalog_and_trusted_time(self) -> None:
        plan = self.plan()
        result = self.event_catalog_result()
        receipt = self.scoring_start_receipt()
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest, plan_id, plan_json, policy_digest, "
                    "trust_store_digest, registered_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan.plan_digest,
                        plan.plan_id,
                        json.dumps(asdict(plan), sort_keys=True),
                        "6" * 64,
                        "7" * 64,
                        plan.registered_at,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
            with self.assertRaisesRegex(ValueError, "registered event catalog"):
                ledger.append_trusted_process_start_receipt(plan, result, receipt)

            ledger.append_physical_event_catalog_result(plan, result)
            with patch.object(
                ledger_module,
                "datetime",
                wraps=datetime,
            ) as trusted_datetime:
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-11T00:00:00+00:00"
                )
                with self.assertRaisesRegex(ValueError, "future start"):
                    ledger.append_trusted_process_start_receipt(
                        plan,
                        result,
                        receipt,
                    )

                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T02:00:00+00:00"
                )
                stored = ledger.append_trusted_process_start_receipt(
                    plan,
                    result,
                    receipt,
                )
            self.assertEqual(stored, receipt.receipt_digest)

    def test_trusted_process_start_receipt_rejects_signature_tampering(self) -> None:
        receipt = self.scoring_start_receipt()
        object.__setattr__(receipt, "scheduler_signature", "0" * 128)

        with self.assertRaisesRegex(ValueError, "receipt is invalid"):
            promotion_module.validate_trusted_process_start_receipt(
                receipt,
                self.event_catalog_plan(),
                catalog_result=self.event_catalog_result(),
            )


if __name__ == "__main__":
    unittest.main()
