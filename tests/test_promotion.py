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
    algorithm_bundle_digest,
    compute_neural_prior_promotion,
    validate_neural_prior_candidate_manifest,
    validate_neural_prior_promotion,
    validate_neural_prior_promotion_applicability,
    verification_plan_digest,
    neural_prior_state_censor_policy_digest,
    numerical_runtime_manifest,
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

    @staticmethod
    def scheduler_key() -> Ed25519PrivateKey:
        return Ed25519PrivateKey.from_private_bytes(b"\x02" * 32)

    def scheduler_trust_store(self, plan=None):
        catalog_plan = self.event_catalog_plan() if plan is None else plan
        return ledger_module._SchedulerTrustStore(
            keys={
                catalog_plan.scheduler_id: self.scheduler_key().public_key(),
            },
            content_digest=catalog_plan.scheduler_trust_store_digest,
        )

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

        self.assertEqual(plan.contract, "neural-prior-holdout-plan-v16")
        self.assertTrue(
            all(
                item.contract == "neural-prior-range-band-contract-v2"
                for item in plan.range_band_contracts
            )
        )
        self.assertEqual(
            manifest.contract,
            "neural-prior-candidate-manifest-v12",
        )
        self.assertEqual(
            evaluation.contract,
            "prior-holdout-evaluation-v20",
        )
        self.assertTrue(
            all(
                band.contract == "neural-prior-range-band-evaluation-v6"
                for band in evaluation.range_band_evaluations
            )
        )
        self.assertEqual(policy.contract, "neural-prior-promotion-policy-v25")

    def test_v15_promotion_remains_audit_only(self) -> None:
        current = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        payload = current._payload()
        payload["contract"] = "neural-prior-promotion-evidence-v15"
        payload["range_metric_end_to_end_cell_bounds"] = tuple(
            item[:8] + item[10:]
            for item in current.range_metric_end_to_end_cell_bounds
        )
        for name in (
            "metric_cell_test_count",
            "metric_cell_inference_method",
            "metric_cell_effective_replicates",
            "metric_cell_tail_replicates",
            "metric_cell_critical_quantile",
            "metric_cell_monte_carlo_standard_error",
            "sample_size_preflight_digest",
            "sample_size_available_physical_events",
            "sample_size_required_physical_events",
            "sample_size_automatic_inference",
            "sample_size_preflight_feasible",
        ):
            payload.pop(name)
        digest = promotion_module.json_digest(payload)
        audit = promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV15(
            promotion_evidence_digest=digest,
            payload_json=json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        self.assertRegex(audit.audit_digest, r"^[0-9a-f]{64}$")
        self.assertFalse(hasattr(audit, "deployment_eligible"))

    def test_v18_promotion_remains_audit_only(self) -> None:
        current = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        payload = current._payload()
        payload["contract"] = "neural-prior-promotion-evidence-v18"
        payload.pop("sample_size_automatic_inference")
        digest = promotion_module.json_digest(payload)
        audit = promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV18(
            promotion_evidence_digest=digest,
            payload_json=json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        self.assertRegex(audit.audit_digest, r"^[0-9a-f]{64}$")
        self.assertFalse(hasattr(audit, "deployment_eligible"))

    def test_pr108_contracts_remain_audit_only(self) -> None:
        plan_payload = promotion_module._holdout_plan_payload(self.plan())
        plan_payload["contract"] = "neural-prior-holdout-plan-v15"
        plan_payload.pop("promotion_experiment_family")
        plan_digest = promotion_module.json_digest(plan_payload)
        plan_payload["plan_digest"] = plan_digest
        plan_audit = promotion_module.LegacyNeuralPriorHoldoutPlanV15Audit(
            plan_digest=plan_digest,
            payload_json=json.dumps(
                plan_payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        current = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        evidence_payload = current._payload()
        evidence_payload["contract"] = "neural-prior-promotion-evidence-v19"
        for name in (
            "promotion_experiment_family_digest",
            "promotion_experiment_family_size",
            "holdout_mode",
        ):
            evidence_payload.pop(name)
        evidence_digest = promotion_module.json_digest(evidence_payload)
        evidence_audit = (
            promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV19(
                promotion_evidence_digest=evidence_digest,
                payload_json=json.dumps(
                    evidence_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )

        self.assertRegex(plan_audit.audit_digest, r"^[0-9a-f]{64}$")
        self.assertRegex(evidence_audit.audit_digest, r"^[0-9a-f]{64}$")
        self.assertFalse(hasattr(evidence_audit, "deployment_eligible"))

    def test_pr105_contracts_remain_audit_only(self) -> None:
        plan_payload = promotion_module._holdout_plan_payload(self.plan())
        plan_payload["contract"] = "neural-prior-holdout-plan-v13"
        plan_payload.pop("operational_issuance_domain_plans")
        cases = plan_payload["cases"]
        assert isinstance(cases, list)
        for case in cases:
            assert isinstance(case, dict)
            case.pop("operational_issuance_domain_plan_digest")
        plan_digest = promotion_module.json_digest(plan_payload)
        plan_payload["plan_digest"] = plan_digest
        plan_audit = promotion_module.LegacyNeuralPriorHoldoutPlanV13Audit(
            plan_digest=plan_digest,
            payload_json=json.dumps(
                plan_payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        current = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        promotion_payload = current._payload()
        promotion_payload["contract"] = "neural-prior-promotion-evidence-v17"
        promotion_payload.pop("scoring_input_artifact_digest")
        promotion_digest = promotion_module.json_digest(promotion_payload)
        promotion_audit = (
            promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV17(
                promotion_evidence_digest=promotion_digest,
                payload_json=json.dumps(
                    promotion_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )

        self.assertRegex(plan_audit.audit_digest, r"^[0-9a-f]{64}$")
        self.assertFalse(hasattr(plan_audit, "cases"))
        self.assertEqual(
            promotion_audit.promotion_evidence_digest,
            promotion_digest,
        )
        self.assertFalse(hasattr(promotion_audit, "eligible"))

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

    def classifier_manifest(self):
        return promotion_module.RegimeClassifierManifest(
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
        issuance_plans = tuple(
            promotion_module.OperationalIssuanceDomainPlan(
                case_id=f"case-{index}",
                grid_contract_digest="2" * 64,
                radar_source_contract_digest="d" * 64,
                lead_minutes=(60,),
                publication_policy_digest="1" * 64,
                source_coverage_policy_digest="2" * 64,
                permanent_exclusion_policy_digest="3" * 64,
                publication_eligible_mask_digest=(
                    promotion_module.tensor_digest(
                        torch.ones((1, 2, 2), dtype=torch.bool)
                    )
                ),
                source_coverage_mask_digest=(
                    promotion_module.tensor_digest(
                        torch.ones((1, 2, 2), dtype=torch.bool)
                    )
                ),
                permanent_exclusion_mask_digest=(
                    promotion_module.tensor_digest(
                        torch.zeros((1, 2, 2), dtype=torch.bool)
                    )
                ),
            )
            for index in (1, 2)
        )
        classifier_manifest = self.classifier_manifest()
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
        cases = (
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
                    operational_issuance_domain_plan_digest=(
                        issuance_plans[0].plan_digest
                    ),
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
                    operational_issuance_domain_plan_digest=(
                        issuance_plans[1].plan_digest
                    ),
                    issue_time="2026-08-10T00:00:00Z",
                ),
            )
        decision_rule_digest = self.decision_rule().rule_digest
        experiment_family = promotion_module.PromotionExperimentFamily(
            holdout_cohort_digest=promotion_module._holdout_dataset_digest(cases),
            parent_prior_digest="d" * 64,
            trials=(
                promotion_module.PromotionExperimentTrial(
                    candidate_prior_digest="c" * 64,
                    promotion_decision_rule_digest=decision_rule_digest,
                    classifier_manifest_digests=(
                        classifier_manifest.manifest_digest,
                    ),
                ),
            ),
            winner_selection_rule_digest="f" * 64,
        )
        return NeuralPriorHoldoutPlan(
            plan_id="holdout-plan",
            parent_prior_digest="d" * 64,
            candidate_family_digests=("c" * 64,),
            cases=cases,
            input_plans=input_plans,
            uncertainty_target_plans=target_plans,
            state_calibration_target_plans=state_target_plans,
            range_band_contracts=range_contracts,
            range_geometry_contracts=range_geometries,
            operational_issuance_domain_plans=issuance_plans,
            regime_reference_plans=reference_plans,
            regime_classifier_manifests=(classifier_manifest,),
            promotion_experiment_family=experiment_family,
            promotion_decision_rule_digest=decision_rule_digest,
            reference_label_contract_digest="7" * 64,
            physical_event_catalog_plan=self.event_catalog_plan(),
            scoring_algorithm_digest="9" * 64,
            scoring_runtime_digest=(
                promotion_module.numerical_runtime_identity_digest("cpu")
            ),
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
            verification_resolver_digest="6" * 64,
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
            operational_issuance_domain_plan_digest=(
                planned.operational_issuance_domain_plan_digest
            ),
            operational_issuance_domain_artifact_digest=(
                self.issuance_domain(index).artifact_digest
            ),
            reference_active_range_regimes=(
                planned.reference_active_range_regimes
            ),
        )

    def issuance_domain(
        self,
        index: int,
        *,
        publication_mask: torch.Tensor | None = None,
    ):
        plan = self.plan().operational_issuance_domain_plans[index - 1]
        publication = (
            torch.ones((1, 2, 2), dtype=torch.bool)
            if publication_mask is None
            else publication_mask
        )
        return promotion_module.OperationalIssuanceDomainArtifact.from_masks(
            plan,
            publication_eligible_mask=publication,
            source_coverage_mask=torch.ones_like(publication),
            permanent_exclusion_mask=torch.zeros_like(publication),
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
            object_track_artifact=self.event_track(index),
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
            source_object_evidence_digest=(
                event.object_track_artifact.object_mask_digests[0]
            ),
            track_artifact_digest=event.object_track_artifact.artifact_digest,
            track_sample_index=0,
            track_sample_time=event.object_track_artifact.timestamps[0],
            track_object_mask_digest=(
                event.object_track_artifact.object_mask_digests[0]
            ),
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
            object_track_artifact=self.training_event_track(),
            participating_radar_ids=("radar-1",),
            association_algorithm_digest="3" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_private_key=self.regime_labeler_key(),
        )

    def event_track(self, index: int):
        day = 8 + index
        return promotion_module.PhysicalEventTrackArtifact(
            timestamps=(
                f"2026-08-{day:02d}T00:00:00Z",
                f"2026-08-{day:02d}T02:00:00Z",
            ),
            centroid_xy_m=((50_000.0, 50_000.0), (50_000.0, 50_000.0)),
            object_mask_digests=(("d" if index == 1 else "e") * 64,) * 2,
            source_radar_ids=("radar-1", "radar-1"),
            association_edge_digests=(("1" if index == 1 else "2") * 64,),
            spatial_reference_digest="7" * 64,
        )

    def training_event_track(self):
        return promotion_module.PhysicalEventTrackArtifact(
            timestamps=(
                "2026-07-01T00:00:00Z",
                "2026-07-01T02:00:00Z",
            ),
            centroid_xy_m=((50_000.0, 50_000.0), (50_000.0, 50_000.0)),
            object_mask_digests=("f" * 64, "e" * 64),
            source_radar_ids=("radar-1", "radar-1"),
            association_edge_digests=("d" * 64,),
            spatial_reference_digest="7" * 64,
        )

    def track_artifact(
        self,
        *,
        start_time,
        end_time,
        start_centroid,
        end_centroid,
        artifact_seed,
        radar_ids,
    ):
        return promotion_module.PhysicalEventTrackArtifact(
            timestamps=(start_time, end_time),
            centroid_xy_m=(start_centroid, end_centroid),
            object_mask_digests=(artifact_seed * 64,) * 2,
            source_radar_ids=(radar_ids[0], radar_ids[-1]),
            association_edge_digests=(artifact_seed * 64,),
            spatial_reference_digest="7" * 64,
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
            motion_association_rule_digest="8" * 64,
            scheduler_id="trusted-training-scheduler",
            scheduler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.scheduler_key()
                )
            ),
            scheduler_trust_store_digest="5" * 64,
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
            motion_association_rule_digest="8" * 64,
            scheduler_id="trusted-training-scheduler",
            scheduler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.scheduler_key()
                )
            ),
            scheduler_trust_store_digest="5" * 64,
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
            process_runtime_digest="4" * 64,
            execution_contract_digest=(
                promotion_module._candidate_training_execution_contract_digest(
                    training_dataset_digest="1" * 64,
                    candidate_training_manifest_digest="2" * 64,
                    model_contract_digest="2" * 64,
                    feature_schema_digest="4" * 64,
                    algorithm_bundle_digest="3" * 64,
                    numerical_runtime_digest="4" * 64,
                )
            ),
            job_id="candidate-training-job",
            launch_nonce="a" * 64,
            scheduler_sequence_number=1,
            previous_receipt_digest=None,
            started_at="2026-07-02T00:00:00Z",
            scheduler_private_key=self.scheduler_key(),
        )

    def scoring_start_receipt(self):
        return self.scoring_start_receipt_for(
            self.event_catalog_plan(),
            self.event_catalog_result(),
            subject_digests=(self.scoring_input_artifact().artifact_digest,),
        )

    def scoring_start_receipt_for(
        self,
        plan,
        result,
        *,
        private_key=None,
        subject_digests=("c" * 64,),
        execution_contract_digest=None,
    ):
        return promotion_module.TrustedProcessStartReceipt.from_plan(
            plan,
            catalog_result_digest=result.result_digest,
            process_kind="candidate_scoring",
            subject_digests=subject_digests,
            process_algorithm_digest="9" * 64,
            process_runtime_digest=self.plan().scoring_runtime_digest,
            execution_contract_digest=(
                self.plan().scoring_execution_contract_digest
                if execution_contract_digest is None
                else execution_contract_digest
            ),
            job_id="candidate-scoring-job",
            launch_nonce="b" * 64,
            scheduler_sequence_number=2,
            previous_receipt_digest=self.training_start_receipt().receipt_digest,
            started_at="2026-08-12T01:00:00Z",
            scheduler_private_key=(
                self.scheduler_key()
                if private_key is None
                else private_key
            ),
        )

    def training_completion_receipt(self):
        process_log = self.training_process_log()
        return promotion_module.TrustedProcessCompletionReceipt.from_start(
            self.training_start_receipt(),
            completed_at="2026-07-03T00:00:00Z",
            output_artifact_digest="c" * 64,
            process_log_digest=process_log.artifact_digest,
            scheduler_private_key=self.scheduler_key(),
        )

    def training_process_log(self):
        return promotion_module.ProcessLogArtifact(
            process_kind="candidate_training",
            start_receipt_digest=self.training_start_receipt().receipt_digest,
            entries=("candidate training completed",),
        )

    def scoring_process_log(self, manifest=None):
        retained = self.manifest() if manifest is None else manifest
        return promotion_module.ProcessLogArtifact(
            process_kind="candidate_scoring",
            start_receipt_digest=(
                retained.candidate_scoring_start_receipt.receipt_digest
            ),
            entries=("holdout scoring completed",),
        )

    def decision_rule(self):
        return promotion_module.PromotionDecisionRule.from_policy(
            self.policy(for_decision_rule=True)
        )

    def scoring_input_artifact(self, *, plan=None, cases=None, policy=None):
        retained_plan = self.plan() if plan is None else plan
        retained_cases = (
            (self.completed_case(1), self.completed_case(2))
            if cases is None
            else tuple(cases)
        )
        return promotion_module.HoldoutScoringInputArtifact.from_cases(
            retained_plan,
            candidate_prior_digest="c" * 64,
            parent_prior_digest="d" * 64,
            candidate_training_manifest_digest="2" * 64,
            parent_training_manifest_digest="3" * 64,
            holdout_cases=retained_cases,
        )

    def scoring_replay_tensors(self, evaluations):
        result = {}
        integer_roles = {
            "source_radar_index_map",
            "range_band_index",
            "event_object_labels",
        }
        boolean_roles = {
            "input_qc_valid_mask",
            "outage_mask",
            "candidate_publication_mask",
            "parent_publication_mask",
            "verification_valid_mask",
            "operational_issuance_mask",
        }
        for evaluation in evaluations:
            for role in ledger_module.SCORING_REPLAY_REQUIRED_TENSOR_ROLES:
                if role in integer_roles:
                    value = torch.zeros((1, 2, 2), dtype=torch.int64)
                elif role in boolean_roles:
                    value = torch.ones((1, 2, 2), dtype=torch.bool)
                else:
                    value = torch.ones((1, 2, 2), dtype=torch.float32)
                result[(evaluation.case_id, role)] = value
        return result

    def scoring_artifact(
        self,
        evaluations,
        *,
        manifest=None,
        plan=None,
        policy=None,
        replay_bundle_digest=None,
    ):
        retained_manifest = self.manifest() if manifest is None else manifest
        retained_plan = self.plan() if plan is None else plan
        return promotion_module.HoldoutScoringArtifact.from_evaluations(
            retained_manifest,
            retained_plan,
            self.scoring_input_artifact(
                plan=retained_plan,
                cases=retained_manifest.holdout_cases,
                policy=policy,
            ),
            tuple(evaluations),
            scoring_replay_bundle_digest=(
                "b" * 64
                if replay_bundle_digest is None
                else replay_bundle_digest
            ),
        )

    def scoring_completion_receipt(self, evaluations, *, manifest=None, plan=None):
        retained = self.manifest() if manifest is None else manifest
        artifact = self.scoring_artifact(
            evaluations,
            manifest=retained,
            plan=plan,
        )
        process_log = self.scoring_process_log(retained)
        return promotion_module.TrustedProcessCompletionReceipt.from_start(
            retained.candidate_scoring_start_receipt,
            completed_at="2026-08-12T02:00:00Z",
            output_artifact_digest=artifact.artifact_digest,
            process_log_digest=process_log.artifact_digest,
            scheduler_private_key=self.scheduler_key(),
        )

    def sealed_scoring(self, evaluations, *, manifest=None, plan=None):
        retained_manifest = self.manifest() if manifest is None else manifest
        retained_plan = self.plan() if plan is None else plan
        artifact = self.scoring_artifact(
            evaluations,
            manifest=retained_manifest,
            plan=retained_plan,
        )
        process_log = self.scoring_process_log(retained_manifest)
        completion = promotion_module.TrustedProcessCompletionReceipt.from_start(
            retained_manifest.candidate_scoring_start_receipt,
            completed_at="2026-08-12T02:00:00Z",
            output_artifact_digest=artifact.artifact_digest,
            process_log_digest=process_log.artifact_digest,
            scheduler_private_key=self.scheduler_key(),
        )
        return artifact, process_log, completion

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
            candidate_training_completion_receipt=(
                self.training_completion_receipt()
            ),
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
        band_withdrawn_fraction: float = 0.0,
        band_newly_issued_fraction: float = 0.0,
        band_fallback_increase: float = 0.0,
        band_confidence_change: float = 0.0,
        band_metric_available: tuple[bool, ...] | None = None,
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
        band_available = torch.tensor(
            [
                [True] * len(metric_names)
                if band_metric_available is None
                else list(band_metric_available)
            ],
            dtype=torch.bool,
        )
        if band_available.shape != band_metric_change.shape:
            raise ValueError("band metric-availability fixture shape mismatch")
        issuance_domain_cells = 10_000
        issuance_domain_area = 1000.0
        parent_issued = 5_000
        withdrawn_count = round(band_withdrawn_fraction * parent_issued)
        newly_issued_count = round(
            band_newly_issued_fraction * issuance_domain_cells
        )
        parent_fallback = 5_000
        candidate_fallback = parent_fallback + round(
            band_fallback_increase * issuance_domain_cells
        )
        parent_confidence_area = 500.0
        candidate_confidence_area = parent_confidence_area + (
            band_confidence_change * issuance_domain_area
        )
        realized_withdrawn_fraction = withdrawn_count / parent_issued
        realized_newly_issued_fraction = (
            newly_issued_count / issuance_domain_cells
        )
        realized_fallback_increase = (
            candidate_fallback - parent_fallback
        ) / issuance_domain_cells
        realized_confidence_change = (
            candidate_confidence_area - parent_confidence_area
        ) / issuance_domain_area
        range_evaluations = tuple(
            promotion_module.RangeBandEvaluation(
                range_regime=range_regime,
                range_band_mask_digest=range_contract.mask_digest(range_regime),
                range_geometry_contract_digest=(
                    range_contract.range_geometry_contract_digest
                ),
                metric_change=band_metric_change,
                end_to_end_metric_change=band_end_to_end_metric_change,
                metric_available=band_available,
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
                ).masked_fill(~band_available, 0.0),
                issuance_domain_digest=(
                    ("9" if range_regime == "near_range" else "8") * 64
                ),
                issuance_domain_cell_count_by_lead=(issuance_domain_cells,),
                issuance_domain_area_km2_by_lead=(issuance_domain_area,),
                parent_issued_count_by_lead=(parent_issued,),
                candidate_issued_count_by_lead=(
                    parent_issued - withdrawn_count + newly_issued_count,
                ),
                withdrawn_count_by_lead=(withdrawn_count,),
                newly_issued_count_by_lead=(newly_issued_count,),
                parent_fallback_count_by_lead=(parent_fallback,),
                candidate_fallback_count_by_lead=(candidate_fallback,),
                parent_confidence_weighted_issued_area_by_lead=(
                    parent_confidence_area,
                ),
                candidate_confidence_weighted_issued_area_by_lead=(
                    candidate_confidence_area,
                ),
                withdrawn_fraction_by_lead=torch.tensor(
                    [realized_withdrawn_fraction], dtype=torch.float64
                ),
                newly_issued_fraction_by_lead=torch.tensor(
                    [realized_newly_issued_fraction], dtype=torch.float64
                ),
                background_fallback_increase_by_lead=torch.tensor(
                    [realized_fallback_increase], dtype=torch.float64
                ),
                confidence_weighted_coverage_change_by_lead=torch.tensor(
                    [realized_confidence_change], dtype=torch.float64
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
        metric_supports = {
            "log_echo_mse": self.metric_support(),
            "soft_fss_error_35": (
                promotion_module.MetricSupportContract.for_metric(
                    "soft_fss_error_35",
                    nowcast_config_digest="a" * 64,
                    spatial_grid_digest="2" * 64,
                    metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
                )
            ),
        }
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
            metric_support_contract_digests=tuple(
                metric_supports[name].contract_digest for name in metric_names
            ),
            nowcast_config_digest="a" * 64,
            grid_contract_digest="2" * 64,
            spatial_grid_digest="2" * 64,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
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

    def metric_support(self):
        return promotion_module.MetricSupportContract.for_metric(
            "log_echo_mse",
            minimum_dbz=0.0,
            maximum_dbz=6.0,
            nowcast_config_digest="a" * 64,
            spatial_grid_digest="2" * 64,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
        )

    def policy(self, *, for_decision_rule: bool = False) -> NeuralPriorPromotionPolicy:
        metric_support = self.metric_support()
        return NeuralPriorPromotionPolicy(
            metric_scales=(PromotionMetricScale("log_echo_mse", 1.0, 0.01),),
            metric_support_contracts=(metric_support,),
            approved_candidate_manifest_digests=(
                ("0" * 64,)
                if for_decision_rule
                else (self.manifest().manifest_digest,)
            ),
            approved_holdout_plan_digests=(
                ("1" * 64,) if for_decision_rule else (self.plan().plan_digest,)
            ),
            approved_metric_contract_digests=("b" * 64,),
            approved_physical_event_catalog_result_digest=(
                "2" * 64
                if for_decision_rule
                else self.event_catalog_result().result_digest
            ),
            deployment_regime_classifier_digest="e" * 64,
            deployment_regime_classifier_manifest_digest=(
                self.classifier_manifest().manifest_digest
                if for_decision_rule
                else self.plan().regime_classifier_manifests[0].manifest_digest
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
                    maximum_mean_normalized_degradation=2.0,
                    maximum_harmful_fraction_upper_bound=1.0,
                    metric_support_contract_digests=(metric_support.contract_digest,),
                    maximum_end_to_end_mean_normalized_degradation=2.0,
                ),
                promotion_module.RangeMetricRequirement(
                    weather_regime="stratiform",
                    range_regime="far_range",
                    metric_name="log_echo_mse",
                    lead_minutes=60,
                    minimum_cases=1,
                    minimum_physical_events=1,
                    minimum_valid_area_km2=0.0,
                    maximum_mean_normalized_degradation=2.0,
                    maximum_harmful_fraction_upper_bound=1.0,
                    metric_support_contract_digests=(metric_support.contract_digest,),
                    maximum_end_to_end_mean_normalized_degradation=2.0,
                ),
            ),
            required_range_issuance=(
                promotion_module.RangeIssuanceRequirement(
                    weather_regime="convective",
                    range_regime="near_range",
                    lead_minutes=60,
                    minimum_cases=1,
                    minimum_physical_events=1,
                    minimum_operational_area_km2=0.1,
                    maximum_withdrawn_fraction=1.0,
                    maximum_newly_issued_fraction=1.0,
                    maximum_background_fallback_increase=1.0,
                    maximum_confidence_weighted_coverage_loss=1.0,
                ),
                promotion_module.RangeIssuanceRequirement(
                    weather_regime="stratiform",
                    range_regime="far_range",
                    lead_minutes=60,
                    minimum_cases=1,
                    minimum_physical_events=1,
                    minimum_operational_area_km2=0.1,
                    maximum_withdrawn_fraction=1.0,
                    maximum_newly_issued_fraction=1.0,
                    maximum_background_fallback_increase=1.0,
                    maximum_confidence_weighted_coverage_loss=1.0,
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
            bootstrap_samples=1024,
            minimum_deployment_metric_cell_events=1,
            minimum_continuous_metric_cell_events=1,
            allow_shadow_small_sample_bootstrap=True,
        )

    def compute(self, evaluations):
        policy = self.policy()
        return self.compute_with_policy(evaluations, policy)

    @staticmethod
    def deployment_ready(evidence):
        """Make small unit fixtures deployment-feasible without weakening production."""

        return replace(
            evidence,
            sample_size_preflight_digest="f" * 64,
            sample_size_required_physical_events=(
                evidence.sample_size_available_physical_events
            ),
            sample_size_cell_feasible=True,
            sample_size_automatic_inference=True,
            sample_size_preflight_feasible=True,
            deployment_eligible=(
                evidence.eligible
                and evidence.regime_classifier_validated
                and bool(evidence.certified_applicability_regime_groups)
                and bool(evidence.certified_range_geometry_contract_digests)
            ),
        )

    def compute_with_policy(self, evaluations, policy):
        plan = self.plan()
        manifest = self.manifest()
        if any(
            not isinstance(item, promotion_module.PriorHoldoutEvaluation)
            for item in evaluations
        ):
            return compute_neural_prior_promotion(
                manifest,
                plan,
                evaluations,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )
        default_classifier = plan.regime_classifier_manifests[0]
        decision_rule = promotion_module.PromotionDecisionRule.from_policy(policy)
        if (
            decision_rule.rule_digest == plan.promotion_decision_rule_digest
            and policy.deployment_regime_classifier_digest
            == default_classifier.classifier_digest
            and policy.deployment_regime_classifier_manifest_digest
            == default_classifier.manifest_digest
        ):
            scoring_input = self.scoring_input_artifact(
                plan=plan,
                cases=manifest.holdout_cases,
            )
            scoring_artifact, scoring_log, scoring_completion = self.sealed_scoring(
                evaluations,
                manifest=manifest,
                plan=plan,
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
                    scoring_input_artifact=scoring_input,
                    scoring_artifact=scoring_artifact,
                    scoring_process_log=scoring_log,
                    scoring_completion_receipt=scoring_completion,
                )
        classifier_manifest = (
            default_classifier
            if policy.deployment_regime_classifier_digest
            == default_classifier.classifier_digest
            else replace(
                default_classifier,
                classifier_digest=policy.deployment_regime_classifier_digest,
            )
        )
        policy = replace(
            policy,
            deployment_regime_classifier_manifest_digest=(
                classifier_manifest.manifest_digest
            ),
        )
        decision_rule = promotion_module.PromotionDecisionRule.from_policy(policy)
        experiment_trial = promotion_module.PromotionExperimentTrial(
            candidate_prior_digest=plan.candidate_family_digests[0],
            promotion_decision_rule_digest=decision_rule.rule_digest,
            classifier_manifest_digests=(classifier_manifest.manifest_digest,),
        )
        plan = replace(
            plan,
            regime_classifier_manifests=(classifier_manifest,),
            promotion_experiment_family=replace(
                plan.promotion_experiment_family,
                trials=(experiment_trial,),
            ),
            promotion_decision_rule_digest=decision_rule.rule_digest,
        )
        manifest = replace(
            manifest,
            holdout_plan_digest=plan.plan_digest,
            holdout_dataset_digest=plan.holdout_dataset_digest,
        )
        scoring_input = self.scoring_input_artifact(
            plan=plan,
            cases=manifest.holdout_cases,
        )
        scoring_start = promotion_module.TrustedProcessStartReceipt.from_plan(
            plan.physical_event_catalog_plan,
            catalog_result_digest=self.event_catalog_result().result_digest,
            process_kind="candidate_scoring",
            subject_digests=(scoring_input.artifact_digest,),
            process_algorithm_digest=plan.scoring_algorithm_digest,
            process_runtime_digest=plan.scoring_runtime_digest,
            execution_contract_digest=plan.scoring_execution_contract_digest,
            job_id="candidate-scoring-job",
            launch_nonce="b" * 64,
            scheduler_sequence_number=2,
            previous_receipt_digest=self.training_start_receipt().receipt_digest,
            started_at="2026-08-12T01:00:00Z",
            scheduler_private_key=self.scheduler_key(),
        )
        manifest = replace(
            manifest,
            candidate_scoring_start_receipt=scoring_start,
        )
        policy = replace(
            policy,
            approved_candidate_manifest_digests=(manifest.manifest_digest,),
            approved_holdout_plan_digests=(plan.plan_digest,),
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
        scoring_artifact = self.scoring_artifact(
            evaluations,
            manifest=manifest,
            plan=plan,
            policy=policy,
        )
        scoring_process_log = self.scoring_process_log(manifest)
        scoring_completion = promotion_module.TrustedProcessCompletionReceipt.from_start(
            manifest.candidate_scoring_start_receipt,
            completed_at="2026-08-12T02:00:00Z",
            output_artifact_digest=scoring_artifact.artifact_digest,
            process_log_digest=scoring_process_log.artifact_digest,
            scheduler_private_key=self.scheduler_key(),
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
                scoring_input_artifact=scoring_input,
                scoring_artifact=scoring_artifact,
                scoring_process_log=scoring_process_log,
                scoring_completion_receipt=scoring_completion,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_promotes_only_independent_material_holdout_cases(self) -> None:
        result = self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        self.assertTrue(result.eligible)
        validate_neural_prior_promotion(result)

    def test_scoring_replay_reproduces_evaluation_digest(self) -> None:
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        manifest = self.manifest()
        plan = self.plan()
        policy = self.policy()
        scoring_process_log = self.scoring_process_log(manifest)
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
                decision_rule = self.decision_rule()
                connection.execute(
                    "INSERT INTO neural_prior_promotion_rule_definitions "
                    "(rule_digest, payload_json, trust_store_digest, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        decision_rule.rule_digest,
                        json.dumps(
                            decision_rule.payload
                            | {"rule_digest": decision_rule.rule_digest},
                            sort_keys=True,
                        ),
                        "7" * 64,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plan_rule_bindings "
                    "(holdout_plan_digest, rule_digest, bound_at) "
                    "VALUES (?, ?, ?)",
                    (
                        plan.plan_digest,
                        decision_rule.rule_digest,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                experiment_family = plan.promotion_experiment_family
                matching_trial = next(
                    item
                    for item in experiment_family.trials
                    if item.candidate_prior_digest
                    == plan.candidate_family_digests[0]
                    and item.promotion_decision_rule_digest
                    == plan.promotion_decision_rule_digest
                )
                connection.execute(
                    "INSERT INTO neural_prior_promotion_experiment_families "
                    "(family_digest, holdout_cohort_digest, payload_json, "
                    "trust_store_digest, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        experiment_family.family_digest,
                        experiment_family.holdout_cohort_digest,
                        json.dumps(
                            experiment_family.payload
                            | {"family_digest": experiment_family.family_digest},
                            sort_keys=True,
                        ),
                        "7" * 64,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plan_experiment_bindings "
                    "(holdout_plan_digest, family_digest, trial_digest, bound_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        plan.plan_digest,
                        experiment_family.family_digest,
                        matching_trial.trial_digest,
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
            self.assertEqual(
                ledger.load_neural_prior_holdout_plan(plan.plan_digest),
                plan,
            )
            scheduler_trust = self.scheduler_trust_store(
                manifest.training_physical_event_catalog_plan
            )
            with (
                patch.object(
                    ledger_module,
                    "_load_scheduler_trust_store",
                    return_value=scheduler_trust,
                ),
                patch.object(
                    ledger_module,
                    "datetime",
                    wraps=datetime,
                ) as trusted_datetime,
            ):
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-07-01T03:00:00+00:00"
                )
                ledger.append_physical_event_catalog_result(
                    manifest.training_physical_event_catalog_plan,
                    manifest.training_physical_event_catalog_result,
                )
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-07-02T01:00:00+00:00"
                )
                ledger.append_trusted_process_start_receipt(
                    manifest.training_physical_event_catalog_plan,
                    manifest.training_physical_event_catalog_result,
                    manifest.candidate_training_start_receipt,
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-07-03T01:00:00+00:00"
                )
                ledger.append_trusted_process_completion_receipt(
                    manifest.candidate_training_start_receipt,
                    manifest.candidate_training_completion_receipt,
                    process_log_artifact=self.training_process_log(),
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-10T04:00:00+00:00"
                )
                ledger.append_physical_event_catalog_result(
                    plan,
                    manifest.physical_event_catalog_result,
                )
                scoring_input = self.scoring_input_artifact(
                    plan=plan,
                    cases=manifest.holdout_cases,
                )
                ledger.append_neural_prior_holdout_scoring_input_artifact(
                    plan,
                    manifest.physical_event_catalog_result,
                    scoring_input,
                )
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T01:30:00+00:00"
                )
                ledger.append_trusted_process_start_receipt(
                    plan,
                    manifest.physical_event_catalog_result,
                    manifest.candidate_scoring_start_receipt,
                    scoring_input_artifact=scoring_input,
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )
                replay_manifest = (
                    ledger.append_neural_prior_scoring_replay_bundle(
                        scoring_input,
                        evaluations,
                        self.scoring_replay_tensors(evaluations),
                        algorithm_source_manifest_digest=(
                            algorithm_bundle_digest()
                        ),
                        runtime_manifest=numerical_runtime_manifest("cpu"),
                    )
                )
                replayed = ledger.load_neural_prior_scoring_replay_bundle(
                    replay_manifest.bundle_digest
                )
                self.assertEqual(
                    tuple(
                        item.evaluation_digest for item in replayed.evaluations
                    ),
                    tuple(item.evaluation_digest for item in evaluations),
                )
                replay_archive = (
                    ledger.scoring_replays_dir
                    / replay_manifest.bundle_digest
                    / "replay_arrays.npz"
                )
                original_archive = replay_archive.read_bytes()
                replay_archive.write_bytes(
                    original_archive[:-1]
                    + bytes((original_archive[-1] ^ 0x01,))
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "scoring replay bundle member checksum mismatch",
                ):
                    ledger.load_neural_prior_scoring_replay_bundle(
                        replay_manifest.bundle_digest
                    )
                replay_archive.write_bytes(original_archive)
                scoring_artifact = self.scoring_artifact(
                    evaluations,
                    manifest=manifest,
                    plan=plan,
                    replay_bundle_digest=replay_manifest.bundle_digest,
                )
                scoring_completion = (
                    promotion_module.TrustedProcessCompletionReceipt.from_start(
                        manifest.candidate_scoring_start_receipt,
                        completed_at="2026-08-12T02:00:00Z",
                        output_artifact_digest=scoring_artifact.artifact_digest,
                        process_log_digest=scoring_process_log.artifact_digest,
                        scheduler_private_key=self.scheduler_key(),
                    )
                )
                with patch.object(
                    promotion_module,
                    "_load_learning_policy_trust_store",
                    return_value=_LearningPolicyTrustStore(
                        approved_policy_digests=frozenset((policy.digest,)),
                        content_digest="b" * 64,
                    ),
                ):
                    evidence = compute_neural_prior_promotion(
                        manifest,
                        plan,
                        evaluations,
                        scoring_input_artifact=scoring_input,
                        scoring_artifact=scoring_artifact,
                        scoring_process_log=scoring_process_log,
                        scoring_completion_receipt=scoring_completion,
                        policy=policy,
                        policy_trust_store_path=(
                            "/etc/advar/learning-policies.json"
                        ),
                    )
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T03:00:00+00:00"
                )
                ledger.append_trusted_process_completion_receipt(
                    manifest.candidate_scoring_start_receipt,
                    scoring_completion,
                    process_log_artifact=scoring_process_log,
                    scoring_artifact=scoring_artifact,
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
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
                    scoring_input_artifact=scoring_input,
                    scoring_artifact=scoring_artifact,
                    scoring_process_log=scoring_process_log,
                    scoring_completion_receipt=scoring_completion,
                    policy=policy,
                    policy_trust_store_path="/etc/advar/learning-policies.json",
                )
            loaded = ledger.load_neural_prior_promotion(stored)
            self.assertEqual(loaded.promotion_evidence_digest, stored)
            self.assertEqual(loaded.contract, "neural-prior-promotion-evidence-v21")

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
        with self.assertRaisesRegex(ValueError, "scoring artifact"):
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

    def test_deployment_sized_constant_event_scores_keep_finite_uncertainty(
        self,
    ) -> None:
        scores = [0.06] * 10
        event_ids = [f"event-{index}" for index in range(10)]

        _, _, mean_lower = promotion_module._cluster_bounds(
            scores,
            event_ids,
            self.policy(),
            candidate_family_size=1,
        )

        self.assertLess(mean_lower, 0.06)

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

    def test_cross_radar_motion_components_cannot_be_split(self) -> None:
        plan = replace(
            self.event_catalog_plan(),
            holdout_case_ids=("handoff-a", "handoff-b"),
        )
        first = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="handoff-a",
            member_case_ids=("handoff-a",),
            member_full_analysis_input_digests=("1" * 64,),
            start_time="2026-08-10T00:00:00Z",
            end_time="2026-08-10T00:10:00Z",
            spatial_envelope_xy_m=(0.0, 0.0, 15_000.0, 10_000.0),
            object_track_artifact=self.track_artifact(
                start_time="2026-08-10T00:00:00Z",
                end_time="2026-08-10T00:10:00Z",
                start_centroid=(2_500.0, 5_000.0),
                end_centroid=(10_000.0, 5_000.0),
                artifact_seed="a",
                radar_ids=("radar-a",),
            ),
            participating_radar_ids=("radar-a",),
            association_algorithm_digest=plan.association_algorithm_digest,
            adjudication_policy_digest=plan.adjudication_policy_digest,
            adjudicator_id=plan.adjudicator_id,
            adjudicator_private_key=self.regime_labeler_key(),
        )
        second = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="handoff-b",
            member_case_ids=("handoff-b",),
            member_full_analysis_input_digests=("2" * 64,),
            start_time="2026-08-10T00:30:00Z",
            end_time="2026-08-10T00:40:00Z",
            spatial_envelope_xy_m=(20_000.0, 0.0, 35_000.0, 10_000.0),
            object_track_artifact=self.track_artifact(
                start_time="2026-08-10T00:30:00Z",
                end_time="2026-08-10T00:40:00Z",
                start_centroid=(25_000.0, 5_000.0),
                end_centroid=(32_500.0, 5_000.0),
                artifact_seed="b",
                radar_ids=("radar-b",),
            ),
            participating_radar_ids=("radar-b",),
            association_algorithm_digest=plan.association_algorithm_digest,
            adjudication_policy_digest=plan.adjudication_policy_digest,
            adjudicator_id=plan.adjudicator_id,
            adjudicator_private_key=self.regime_labeler_key(),
        )

        with self.assertRaisesRegex(ValueError, "split connected components"):
            promotion_module.PhysicalEventCatalogResult.from_plan(
                plan,
                event_evidences=(first, second),
                case_spatial_membership_evidences=(
                    promotion_module.PhysicalEventCaseSpatialEvidence(
                        case_id="handoff-a",
                        full_analysis_input_digest="1" * 64,
                        physical_event_identity_digest=(
                            first.physical_event_identity_digest
                        ),
                        observed_spatial_envelope_xy_m=(
                            1_000.0,
                            1_000.0,
                            9_000.0,
                            9_000.0,
                        ),
                        event_spatial_envelope_xy_m=first.spatial_envelope_xy_m,
                        spatial_membership_rule_digest=(
                            plan.spatial_membership_rule_digest
                        ),
                        source_object_evidence_digest=(
                            first.object_track_artifact.object_mask_digests[0]
                        ),
                        track_artifact_digest=first.object_track_artifact.artifact_digest,
                        track_sample_index=0,
                        track_sample_time=first.object_track_artifact.timestamps[0],
                        track_object_mask_digest=(
                            first.object_track_artifact.object_mask_digests[0]
                        ),
                        input_available_time="2026-08-10T02:30:00Z",
                        spatial_reference_digest=plan.spatial_reference_digest,
                    ),
                    promotion_module.PhysicalEventCaseSpatialEvidence(
                        case_id="handoff-b",
                        full_analysis_input_digest="2" * 64,
                        physical_event_identity_digest=(
                            second.physical_event_identity_digest
                        ),
                        observed_spatial_envelope_xy_m=(
                            21_000.0,
                            1_000.0,
                            34_000.0,
                            9_000.0,
                        ),
                        event_spatial_envelope_xy_m=second.spatial_envelope_xy_m,
                        spatial_membership_rule_digest=(
                            plan.spatial_membership_rule_digest
                        ),
                        source_object_evidence_digest=(
                            second.object_track_artifact.object_mask_digests[0]
                        ),
                        track_artifact_digest=second.object_track_artifact.artifact_digest,
                        track_sample_index=0,
                        track_sample_time=second.object_track_artifact.timestamps[0],
                        track_object_mask_digest=(
                            second.object_track_artifact.object_mask_digests[0]
                        ),
                        input_available_time="2026-08-10T02:30:00Z",
                        spatial_reference_digest=plan.spatial_reference_digest,
                    ),
                ),
                cataloged_at="2026-08-10T03:00:00Z",
                adjudicator_private_key=self.regime_labeler_key(),
            )

    def test_scheduler_authority_must_be_independent_from_adjudicator(self) -> None:
        plan = self.event_catalog_plan()

        with self.assertRaisesRegex(ValueError, "scheduler authority"):
            replace(
                plan,
                scheduler_id=plan.adjudicator_id,
                scheduler_public_key_hex=plan.adjudicator_public_key_hex,
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
            object_track_artifact=self.track_artifact(
                start_time="2026-08-09T00:00:00Z",
                end_time="2026-08-10T00:00:00Z",
                start_centroid=(20_000.0, 20_000.0),
                end_centroid=(30_000.0, 30_000.0),
                artifact_seed="5",
                radar_ids=("radar-1", "radar-2"),
            ),
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
            object_track_artifact=self.track_artifact(
                start_time="2026-08-09T03:00:00Z",
                end_time="2026-08-09T04:00:00Z",
                start_centroid=(50_000.0, 50_000.0),
                end_centroid=(50_000.0, 50_000.0),
                artifact_seed="5",
                radar_ids=(first_case.radar_id,),
            ),
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
        other_key = Ed25519PrivateKey.from_private_bytes(b"\x03" * 32)
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
            object_track_artifact=self.event_track(1),
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
            object_track_artifact=self.event_track(2),
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
        changed_scoring_start = self.scoring_start_receipt_for(
            untrusted_plan,
            untrusted_result,
            private_key=self.scheduler_key(),
        )
        changed_manifest = replace(
            original,
            physical_event_catalog_evidences=(
                untrusted,
                untrusted_second,
            ),
            physical_event_catalog_result=untrusted_result,
            holdout_cases=(changed_case, changed_second_case),
            candidate_scoring_start_receipt=changed_scoring_start,
        )
        with self.assertRaisesRegex(ValueError, "event-catalog adjudicator"):
            promotion_module._validate_physical_event_catalogs_against_plan(
                changed_manifest,
                self.plan(),
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
            object_track_artifact=self.event_track(1),
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
                object_track_artifact=self.event_track(2),
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
        changed_scoring_start = self.scoring_start_receipt_for(
            changed_catalog_plan,
            changed_result,
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
                candidate_scoring_start_receipt=changed_scoring_start,
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
            motion_association_rule_digest="8" * 64,
            scheduler_id="trusted-training-scheduler",
            scheduler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.scheduler_key()
                )
            ),
            scheduler_trust_store_digest="5" * 64,
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
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        scoring_artifact, scoring_log, scoring_completion = self.sealed_scoring(
            evaluations
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "event catalog result"):
            compute_neural_prior_promotion(
                self.manifest(),
                self.plan(),
                evaluations,
                scoring_input_artifact=self.scoring_input_artifact(),
                scoring_artifact=scoring_artifact,
                scoring_process_log=scoring_log,
                scoring_completion_receipt=scoring_completion,
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
            object_track_artifact=self.track_artifact(
                start_time="2026-08-09T00:05:00Z",
                end_time="2026-08-09T02:05:00Z",
                start_centroid=(50_000.0, 50_000.0),
                end_centroid=(50_000.0, 50_000.0),
                artifact_seed="5",
                radar_ids=holdout_event.participating_radar_ids,
            ),
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
            source_object_evidence_digest=(
                shifted.object_track_artifact.object_mask_digests[0]
            ),
            track_artifact_digest=shifted.object_track_artifact.artifact_digest,
            track_sample_index=0,
            track_sample_time=shifted.object_track_artifact.timestamps[0],
            track_object_mask_digest=(
                shifted.object_track_artifact.object_mask_digests[0]
            ),
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
            process_runtime_digest="4" * 64,
            execution_contract_digest=(
                promotion_module._candidate_training_execution_contract_digest(
                    training_dataset_digest="1" * 64,
                    candidate_training_manifest_digest="2" * 64,
                    model_contract_digest="2" * 64,
                    feature_schema_digest="4" * 64,
                    algorithm_bundle_digest="3" * 64,
                    numerical_runtime_digest="4" * 64,
                )
            ),
            job_id="shifted-training-job",
            launch_nonce="c" * 64,
            scheduler_sequence_number=1,
            previous_receipt_digest=None,
            started_at="2026-08-10T01:00:00Z",
            scheduler_private_key=self.scheduler_key(),
        )
        training_completion = (
            promotion_module.TrustedProcessCompletionReceipt.from_start(
                training_receipt,
                completed_at="2026-08-10T02:00:00Z",
                output_artifact_digest="c" * 64,
                process_log_digest="d" * 64,
                scheduler_private_key=self.scheduler_key(),
            )
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
                candidate_training_completion_receipt=training_completion,
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
            object_track_artifact=original.object_track_artifact,
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
                track_artifact_digest=event.object_track_artifact.artifact_digest,
                track_sample_index=0,
                track_sample_time=event.object_track_artifact.timestamps[0],
                track_object_mask_digest="d" * 64,
                input_available_time="2026-08-09T00:00:00Z",
                spatial_reference_digest="7" * 64,
            )

    def test_case_object_evidence_must_match_the_track_sample(self) -> None:
        event = self.event_catalog(1)
        forged = replace(
            self.event_spatial_evidence(
                event,
                case_id="case-1",
                full_analysis_input_digest="1" * 64,
                source_object_evidence_digest="d" * 64,
            ),
            source_object_evidence_digest="f" * 64,
            track_object_mask_digest="f" * 64,
        )

        with self.assertRaisesRegex(ValueError, "spatial-membership evidence"):
            promotion_module.PhysicalEventCatalogResult.from_plan(
                replace(
                    self.event_catalog_plan(),
                    holdout_case_ids=("case-1",),
                ),
                event_evidences=(event,),
                case_spatial_membership_evidences=(forged,),
                cataloged_at="2026-08-10T03:00:00Z",
                adjudicator_private_key=self.regime_labeler_key(),
            )

    def test_metric_bound_is_derived_from_the_metric_support(self) -> None:
        fss = promotion_module.MetricSupportContract.for_metric(
            "soft_fss_error_35",
            nowcast_config_digest="a" * 64,
            spatial_grid_digest="2" * 64,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
        )
        echo = promotion_module.MetricSupportContract.for_metric(
            "log_echo_mse",
            minimum_dbz=-10.0,
            maximum_dbz=70.0,
            nowcast_config_digest="a" * 64,
            spatial_grid_digest="2" * 64,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
        )

        self.assertEqual((fss.lower_bound, fss.upper_bound), (0.0, 1.0))
        self.assertAlmostEqual(
            echo.upper_bound,
            (8.0 * promotion_module.math.log(10.0)) ** 2,
        )
        with self.assertRaises(TypeError):
            promotion_module.MetricSupportContract()

    def test_centroid_support_uses_the_full_affine_grid_diameter(self) -> None:
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
            pixel_to_projected_matrix_m=(
                (1_000.0, -500.0),
                (0.0, 500.0 * promotion_module.math.sqrt(3.0)),
            ),
        )
        frames = torch.zeros((3, 101, 101), dtype=torch.float64)
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
            grid_time_contract=grid,
        )

        support = promotion_module.MetricSupportContract.from_run(
            "centroid_error_m2",
            run,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
            grid_shape=(101, 101),
        )

        self.assertAlmostEqual(
            promotion_module.math.sqrt(support.upper_bound),
            173_205.0808,
            places=3,
        )

    def test_metric_support_reuses_spatial_identity_across_valid_times(self) -> None:
        first_grid = RadarGridTimeContract(
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
        second_grid = replace(
            first_grid,
            valid_times=(
                "2026-08-09T01:00:00Z",
                "2026-08-09T01:10:00Z",
                "2026-08-09T01:20:00Z",
            ),
        )
        frames = torch.zeros((3, 4, 5), dtype=torch.float64)
        first_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
            grid_time_contract=first_grid,
        )
        second_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
            grid_time_contract=second_grid,
        )
        engine = promotion_module.scoring_metric_engine_identity_digest()
        first_support = promotion_module.MetricSupportContract.from_run(
            "centroid_error_m2",
            first_run,
            metric_engine_digest=engine,
            grid_shape=(4, 5),
        )
        second_support = promotion_module.MetricSupportContract.from_run(
            "centroid_error_m2",
            second_run,
            metric_engine_digest=engine,
            grid_shape=(4, 5),
        )

        self.assertNotEqual(first_grid.digest, second_grid.digest)
        self.assertEqual(
            first_grid.spatial_grid_digest,
            second_grid.spatial_grid_digest,
        )
        self.assertEqual(first_support.contract_digest, second_support.contract_digest)

    def test_metric_support_rejects_an_uninstalled_metric_engine(self) -> None:
        first = promotion_module.MetricSupportContract.for_metric(
            "soft_fss_error_35",
            nowcast_config_digest="a" * 64,
            spatial_grid_digest="2" * 64,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
        )
        self.assertEqual(
            first.metric_engine_digest,
            promotion_module.scoring_metric_engine_identity_digest(),
        )
        with self.assertRaisesRegex(ValueError, "installed implementation"):
            promotion_module.MetricSupportContract.for_metric(
                "soft_fss_error_35",
                nowcast_config_digest="a" * 64,
                spatial_grid_digest="2" * 64,
                metric_engine_digest="8" * 64,
            )

    def test_metric_support_must_match_the_forecast_run_and_grid(self) -> None:
        original = self.evaluation(1, -0.2)
        forged = promotion_module._new_prior_holdout_evaluation(
            **{
                key: value
                for key, value in original.__dict__.items()
                if key not in ("contract", "evaluation_digest")
            }
            | {"nowcast_config_digest": "f" * 64}
        )
        with self.assertRaisesRegex(
            ValueError,
            "disagrees with forecast run, grid, or engine",
        ):
            self.compute_with_policy(
                (forged, self.evaluation(2, -0.3)),
                self.policy(),
            )

        forged_engine = promotion_module._new_prior_holdout_evaluation(
            **{
                key: value
                for key, value in original.__dict__.items()
                if key not in ("contract", "evaluation_digest")
            }
            | {"metric_engine_digest": "f" * 64}
        )
        with self.assertRaisesRegex(ValueError, "run, grid, or engine"):
            self.compute_with_policy(
                (forged_engine, self.evaluation(2, -0.3)),
                self.policy(),
            )

    def test_scoring_input_seals_completed_reference_truth(self) -> None:
        plan = self.plan()
        manifest = self.manifest()
        artifact = self.scoring_input_artifact(
            plan=plan,
            cases=manifest.holdout_cases,
        )
        changed_cases = (
            replace(
                manifest.holdout_cases[0],
                regime_reference_evidence_digest="f" * 64,
            ),
            manifest.holdout_cases[1],
        )

        with self.assertRaisesRegex(ValueError, "forecast set"):
            promotion_module.validate_holdout_scoring_input_artifact(
                artifact,
                plan,
                candidate_prior_digest=manifest.candidate_prior_digest,
                parent_prior_digest=manifest.parent_prior_digest,
                candidate_training_manifest_digest=(
                    manifest.candidate_training_manifest_digest
                ),
                parent_training_manifest_digest=(
                    manifest.parent_training_manifest_digest
                ),
                holdout_cases=changed_cases,
            )

    def test_automatic_small_sample_aggregate_does_not_use_point_bootstrap(
        self,
    ) -> None:
        policy = replace(
            self.policy(),
            allow_shadow_small_sample_bootstrap=False,
        )
        _, _, lower = promotion_module._cluster_bounds(
            [0.06] * 8,
            [f"event-{index}" for index in range(8)],
            policy,
            candidate_family_size=1,
            absolute_bound=1.0,
        )

        self.assertLess(lower, 0.06)
        self.assertLess(lower, policy.minimum_mean_normalized_improvement)

    def test_automatic_uncertainty_never_treats_zero_variance_as_certainty(
        self,
    ) -> None:
        policy = replace(
            self.policy(),
            allow_shadow_small_sample_bootstrap=False,
        )
        clusters = tuple(f"event-{index}" for index in range(5))
        result = promotion_module._simultaneous_uncertainty_upper_bounds(
            (
                promotion_module._UncertaintyComparison(
                    component="support",
                    group=None,
                    values=(-0.01,) * 5,
                    clusters=clusters,
                ),
                promotion_module._UncertaintyComparison(
                    component="state_nll",
                    group=None,
                    values=(-0.01,) * 5,
                    clusters=clusters,
                ),
            ),
            policy,
            candidate_family_size=1,
        )

        self.assertEqual(result.method, "support_bounded_hybrid")
        self.assertGreater(result.comparison_bounds[("support", None)], -0.01)
        self.assertGreater(result.comparison_bounds[("state_nll", None)], -0.01)
        self.assertFalse(result.unresolved_unbounded_zero_variance)

    def test_experiment_family_counts_trials_across_separate_plans(self) -> None:
        plan = self.plan()
        original_trial = plan.promotion_experiment_family.trials[0]
        additional = tuple(
            promotion_module.PromotionExperimentTrial(
                candidate_prior_digest=f"{index:064x}",
                promotion_decision_rule_digest=f"{index + 100:064x}",
                classifier_manifest_digests=(f"{index + 200:064x}",),
            )
            for index in range(1, 20)
        )
        family = replace(
            plan.promotion_experiment_family,
            trials=(original_trial, *additional),
        )
        expanded_plan = replace(plan, promotion_experiment_family=family)
        policy = replace(
            self.policy(),
            allow_shadow_small_sample_bootstrap=False,
        )
        clusters = [f"event-{index}" for index in range(1_000)]
        _, _, single_lower = promotion_module._cluster_bounds(
            [0.2] * len(clusters),
            clusters,
            policy,
            candidate_family_size=1,
            absolute_bound=1.0,
        )
        _, _, family_lower = promotion_module._cluster_bounds(
            [0.2] * len(clusters),
            clusters,
            policy,
            candidate_family_size=family.total_family_size,
            absolute_bound=1.0,
        )

        self.assertEqual(expanded_plan.promotion_experiment_family.total_family_size, 20)
        self.assertLess(family_lower, single_lower)

    def test_ledger_rejects_a_second_family_for_the_same_holdout_cohort(
        self,
    ) -> None:
        plan = self.plan()
        second_family = replace(
            plan.promotion_experiment_family,
            winner_selection_rule_digest="e" * 64,
        )
        second_plan = replace(
            plan,
            plan_id="holdout-plan-second-family",
            promotion_experiment_family=second_family,
        )
        first_policy = promotion_module.NeuralPriorHoldoutPlanPolicy(
            approved_plan_digests=(plan.plan_digest,),
            approved_metric_contract_digests=tuple(
                sorted({item.metric_contract_digest for item in plan.cases})
            ),
            maximum_candidate_family_size=1,
        )
        second_policy = replace(
            first_policy,
            approved_plan_digests=(second_plan.plan_digest,),
        )
        decision_rule = self.decision_rule()
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset(
                (
                    first_policy.digest,
                    second_policy.digest,
                    decision_rule.rule_digest,
                    plan.promotion_experiment_family.family_digest,
                    second_family.family_digest,
                )
            ),
            content_digest="b" * 64,
        )
        scheduler_trust = self.scheduler_trust_store(
            plan.physical_event_catalog_plan
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with (
                patch.object(
                    ledger_module,
                    "_load_learning_policy_trust_store",
                    return_value=trust,
                ),
                patch.object(
                    ledger_module,
                    "_load_scheduler_trust_store",
                    return_value=scheduler_trust,
                ),
                patch.object(ledger_module, "datetime", wraps=datetime) as clock,
            ):
                clock.now.return_value = datetime.fromisoformat(
                    "2026-08-08T00:00:00+00:00"
                )
                ledger.append_neural_prior_holdout_plan(
                    plan,
                    promotion_decision_rule=decision_rule,
                    policy=first_policy,
                    policy_trust_store_path="/etc/advar/learning-policies.json",
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )
                with self.assertRaisesRegex(ValueError, "holdout cohort"):
                    ledger.append_neural_prior_holdout_plan(
                        second_plan,
                        promotion_decision_rule=decision_rule,
                        policy=second_policy,
                        policy_trust_store_path=(
                            "/etc/advar/learning-policies.json"
                        ),
                        scheduler_trust_store_path="/etc/advar/schedulers.json",
                    )

    def test_historical_evidence_cannot_be_marked_deployment_eligible(self) -> None:
        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        self.assertTrue(evidence.deployment_eligible)

        with self.assertRaisesRegex(ValueError, "deployment eligibility"):
            replace(evidence, holdout_mode="sealed_historical")

    def test_required_metric_cell_must_be_non_inferior(self) -> None:
        support = promotion_module.MetricSupportContract.for_metric(
            "soft_fss_error_35",
            nowcast_config_digest="a" * 64,
            spatial_grid_digest="2" * 64,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
        )
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
            metric_support_contract_digests=(support.contract_digest,),
        )
        policy = replace(
            self.policy(),
            metric_scales=(
                PromotionMetricScale("log_echo_mse", 1.0, 0.01),
                PromotionMetricScale("soft_fss_error_35", 1.0, 0.01),
            ),
            metric_support_contracts=(
                self.policy().metric_support_contracts[0],
                support,
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
            metric_support_contract_digests=(
                self.policy().metric_support_contracts[0].contract_digest,
            ),
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
            allow_shadow_small_sample_bootstrap=False,
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

    def test_promotion_evidence_makes_preflight_a_deployment_gate(self) -> None:
        result = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )

        self.assertTrue(result.eligible)
        self.assertFalse(result.sample_size_preflight_feasible)
        self.assertFalse(result.deployment_eligible)
        self.assertLess(
            result.sample_size_available_physical_events,
            result.sample_size_required_physical_events,
        )

    def test_metric_cell_family_includes_local_issuance_bounds(self) -> None:
        policy = self.policy()
        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )

        expected_tests = (
            4 * len(policy.required_range_metrics)
            + 4 * len(policy.required_range_issuance)
        )
        self.assertEqual(result.metric_cell_test_count, expected_tests)
        self.assertAlmostEqual(
            result.metric_cell_tail_replicates,
            policy.bootstrap_samples
            * (1.0 - policy.confidence_level)
            / (2.0 * expected_tests),
        )

    def test_band_local_withdrawal_blocks_only_the_affected_group(self) -> None:
        near_requirement = replace(
            self.policy().required_range_issuance[0],
            maximum_withdrawn_fraction=0.1,
        )
        policy = replace(
            self.policy(),
            required_range_issuance=(
                near_requirement,
                self.policy().required_range_issuance[1],
            ),
        )
        result = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    band_withdrawn_fraction=1.0,
                ),
                self.evaluation(2, -0.3),
            ),
            policy,
        )

        self.assertNotIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )
        near_bounds = next(
            item
            for item in result.range_issuance_cell_bounds
            if item[:3] == ("convective", "near_range", 60)
        )
        self.assertGreater(near_bounds[3], 0.1)

    def test_issuance_gate_includes_cases_with_unavailable_required_metric(
        self,
    ) -> None:
        near_requirement = replace(
            self.policy().required_range_issuance[0],
            maximum_withdrawn_fraction=0.1,
        )
        policy = replace(
            self.policy(),
            metric_scales=(
                PromotionMetricScale("log_echo_mse", 1.0, 0.01),
                PromotionMetricScale("soft_fss_error_35", 1.0, 0.01),
            ),
            metric_support_contracts=(
                self.metric_support(),
                promotion_module.MetricSupportContract.for_metric(
                    "soft_fss_error_35",
                    nowcast_config_digest="a" * 64,
                    spatial_grid_digest="2" * 64,
                    metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
                ),
            ),
            required_range_issuance=(
                near_requirement,
                self.policy().required_range_issuance[1],
            ),
        )
        result = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    secondary_range_change=-0.1,
                    band_metric_available=(False, True),
                    band_withdrawn_fraction=1.0,
                ),
                self.evaluation(2, -0.3),
            ),
            policy,
        )

        near_issuance = next(
            item
            for item in result.range_issuance_cell_bounds
            if item[:3] == ("convective", "near_range", 60)
        )
        self.assertGreater(near_issuance[3], 0.1)
        self.assertNotIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )

    def test_scoring_completion_rejects_an_unbacked_output_digest(self) -> None:
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        manifest = self.manifest()
        artifact = self.scoring_artifact(evaluations, manifest=manifest)
        process_log = self.scoring_process_log(manifest)
        bad_completion = promotion_module.TrustedProcessCompletionReceipt.from_start(
            manifest.candidate_scoring_start_receipt,
            completed_at="2026-08-12T02:00:00Z",
            output_artifact_digest="b" * 64,
            process_log_digest=process_log.artifact_digest,
            scheduler_private_key=self.scheduler_key(),
        )

        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with self.assertRaisesRegex(ValueError, "canonical scoring artifact"):
                ledger.append_trusted_process_completion_receipt(
                    manifest.candidate_scoring_start_receipt,
                    bad_completion,
                    process_log_artifact=process_log,
                    scoring_artifact=artifact,
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )

    def test_scoring_artifact_detects_a_changed_evaluation(self) -> None:
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        manifest = self.manifest()
        artifact = self.scoring_artifact(evaluations, manifest=manifest)
        changed_values = {
            name: value
            for name, value in evaluations[0].__dict__.items()
            if name not in {"contract", "evaluation_digest"}
        }
        changed_values["metric_change"] = torch.tensor(
            [[0.75]], dtype=torch.float64
        )
        changed = promotion_module._new_prior_holdout_evaluation(
            **changed_values
        )

        with self.assertRaisesRegex(ValueError, "typed evaluations"):
            promotion_module.validate_holdout_scoring_artifact(
                artifact,
                manifest,
                self.plan(),
                self.scoring_input_artifact(),
                (changed, evaluations[1]),
            )

    def test_scoring_input_seals_the_exact_forecast_realization(self) -> None:
        original_cases = (self.completed_case(1), self.completed_case(2))
        artifact = self.scoring_input_artifact(cases=original_cases)
        changed_cases = (
            replace(
                original_cases[0],
                candidate_forecast_digest="f" * 64,
            ),
            original_cases[1],
        )

        with self.assertRaisesRegex(ValueError, "forecast set"):
            promotion_module.validate_holdout_scoring_input_artifact(
                artifact,
                self.plan(),
                candidate_prior_digest="c" * 64,
                parent_prior_digest="d" * 64,
                candidate_training_manifest_digest="2" * 64,
                parent_training_manifest_digest="3" * 64,
                holdout_cases=changed_cases,
            )

    def test_piecewise_linear_tracks_detect_mid_segment_crossing(self) -> None:
        first = self.track_artifact(
            start_time="2026-08-10T00:00:00Z",
            end_time="2026-08-10T00:10:00Z",
            start_centroid=(-10_000.0, 0.0),
            end_centroid=(10_000.0, 0.0),
            artifact_seed="a",
            radar_ids=("radar-a",),
        )
        second = self.track_artifact(
            start_time="2026-08-10T00:00:00Z",
            end_time="2026-08-10T00:10:00Z",
            start_centroid=(0.0, -10_000.0),
            end_centroid=(0.0, 10_000.0),
            artifact_seed="b",
            radar_ids=("radar-b",),
        )

        self.assertAlmostEqual(
            promotion_module._overlap_track_distance(first, second),
            0.0,
        )

    def test_track_rejects_hidden_extreme_segment_speed(self) -> None:
        with self.assertRaisesRegex(ValueError, "segment speed"):
            promotion_module.PhysicalEventTrackArtifact(
                timestamps=(
                    "2026-08-10T00:00:00Z",
                    "2026-08-10T01:00:00Z",
                    "2026-08-10T02:00:00Z",
                ),
                centroid_xy_m=(
                    (0.0, 0.0),
                    (500_000.0, 0.0),
                    (0.0, 0.0),
                ),
                object_mask_digests=("d" * 64,) * 3,
                source_radar_ids=("radar-a",) * 3,
                association_edge_digests=("1" * 64, "2" * 64),
                spatial_reference_digest="7" * 64,
            )

    def test_disjoint_track_extrapolation_uses_terminal_motion(self) -> None:
        track = promotion_module.PhysicalEventTrackArtifact(
            timestamps=(
                "2026-08-10T00:00:00Z",
                "2026-08-10T00:10:00Z",
                "2026-08-10T00:20:00Z",
            ),
            centroid_xy_m=((0.0, 0.0), (10_000.0, 0.0), (10_000.0, 10_000.0)),
            object_mask_digests=("d" * 64,) * 3,
            source_radar_ids=("radar-a",) * 3,
            association_edge_digests=("1" * 64, "2" * 64),
            spatial_reference_digest="7" * 64,
        )

        self.assertAlmostEqual(track.terminal_velocity_xy_mps[0], 0.0)
        self.assertAlmostEqual(
            track.terminal_velocity_xy_mps[1],
            10_000.0 / 600.0,
        )

    def test_operational_domain_uses_publication_cells_as_denominator(self) -> None:
        publication = torch.zeros((1, 10, 10), dtype=torch.bool)
        publication[:, :2, :5] = True
        source = torch.ones_like(publication)
        exclusion = torch.zeros_like(publication)
        plan = replace(
            self.plan().operational_issuance_domain_plans[0],
            publication_eligible_mask_digest=(
                promotion_module.tensor_digest(publication)
            ),
            source_coverage_mask_digest=promotion_module.tensor_digest(source),
            permanent_exclusion_mask_digest=(
                promotion_module.tensor_digest(exclusion)
            ),
        )
        artifact = promotion_module.OperationalIssuanceDomainArtifact.from_masks(
            plan,
            publication_eligible_mask=publication,
            source_coverage_mask=source,
            permanent_exclusion_mask=exclusion,
        )

        self.assertEqual(artifact.eligible_cell_counts, (10,))
        self.assertEqual(int(torch.count_nonzero(artifact.eligible_mask)), 10)

        object.__setattr__(
            artifact,
            "_eligible_mask",
            torch.ones_like(publication),
        )
        with self.assertRaisesRegex(ValueError, "issuance-domain artifact"):
            promotion_module.validate_operational_issuance_domain_artifact(
                artifact
            )

    def test_operational_masks_must_match_the_holdout_plan(self) -> None:
        publication = torch.ones((1, 2, 2), dtype=torch.bool)
        publication[0, 1, 1] = False

        with self.assertRaisesRegex(ValueError, "not preregistered"):
            promotion_module.OperationalIssuanceDomainArtifact.from_masks(
                self.plan().operational_issuance_domain_plans[0],
                publication_eligible_mask=publication,
                source_coverage_mask=torch.ones_like(publication),
                permanent_exclusion_mask=torch.zeros_like(publication),
            )

    def test_dynamic_source_coverage_resolves_after_input_availability(self) -> None:
        plan = self.plan().operational_issuance_domain_plans[0]
        nominal = torch.ones((1, 2, 2), dtype=torch.bool)
        source_index = torch.tensor([[[0, 1], [-1, 1]]], dtype=torch.int64)
        outage = torch.tensor([[[False, True], [False, False]]])
        qc_valid = torch.ones_like(nominal)
        resolved = promotion_module.ResolvedSourceCoverageArtifact.from_observations(
            plan,
            nominal_source_coverage_mask=nominal,
            source_radar_index_map=source_index,
            outage_mask=outage,
            dynamic_qc_valid_mask=qc_valid,
            input_bundle_digest="e" * 64,
            full_analysis_input_digest="1" * 64,
            input_available_at="2026-08-09T00:00:00Z",
            resolved_at="2026-08-09T00:01:00Z",
        )
        domain = promotion_module.OperationalIssuanceDomainArtifact.from_masks(
            plan,
            publication_eligible_mask=nominal,
            source_coverage_mask=nominal,
            permanent_exclusion_mask=torch.zeros_like(nominal),
            resolved_source_coverage=resolved,
        )

        self.assertEqual(resolved.resolved_cell_counts, (2,))
        self.assertEqual(domain.eligible_cell_counts, (2,))
        self.assertEqual(
            domain.resolved_source_coverage_artifact_digest,
            resolved.artifact_digest,
        )
        self.assertEqual(domain.resolved_input_bundle_digest, "e" * 64)

    def test_dynamic_source_coverage_cannot_predate_input(self) -> None:
        plan = self.plan().operational_issuance_domain_plans[0]
        mask = torch.ones((1, 2, 2), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "before input availability"):
            promotion_module.ResolvedSourceCoverageArtifact.from_observations(
                plan,
                nominal_source_coverage_mask=mask,
                source_radar_index_map=torch.zeros_like(mask, dtype=torch.int64),
                outage_mask=torch.zeros_like(mask),
                dynamic_qc_valid_mask=mask,
                input_bundle_digest="e" * 64,
                full_analysis_input_digest="1" * 64,
                input_available_at="2026-08-09T00:01:00Z",
                resolved_at="2026-08-09T00:00:00Z",
            )

    def test_track_payload_tampering_is_detected_by_rehash(self) -> None:
        event = self.event_catalog(1)
        object.__setattr__(
            event.object_track_artifact,
            "centroid_xy_m",
            ((50_000.0, 50_000.0), (60_000.0, 50_000.0)),
        )

        with self.assertRaisesRegex(ValueError, "event-catalog"):
            promotion_module.validate_physical_event_catalog(event)

    def test_event_envelope_must_contain_the_entire_replayable_track(self) -> None:
        track = promotion_module.PhysicalEventTrackArtifact(
            timestamps=(
                "2026-08-09T00:00:00Z",
                "2026-08-09T01:00:00Z",
                "2026-08-09T02:00:00Z",
            ),
            centroid_xy_m=(
                (50_000.0, 50_000.0),
                (150_000.0, 50_000.0),
                (50_000.0, 50_000.0),
            ),
            object_mask_digests=("d" * 64,) * 3,
            source_radar_ids=("radar-1",) * 3,
            association_edge_digests=("1" * 64, "2" * 64),
            spatial_reference_digest="7" * 64,
        )

        with self.assertRaisesRegex(ValueError, "event-catalog"):
            promotion_module.PhysicalEventCatalogEvidence.from_members(
                event_id="escaped-track",
                member_case_ids=("case-1",),
                member_full_analysis_input_digests=("1" * 64,),
                start_time="2026-08-09T00:00:00Z",
                end_time="2026-08-09T02:00:00Z",
                spatial_envelope_xy_m=(0.0, 0.0, 100_000.0, 100_000.0),
                object_track_artifact=track,
                participating_radar_ids=("radar-1",),
                association_algorithm_digest="3" * 64,
                adjudication_policy_digest="6" * 64,
                adjudicator_id="independent-weather-labeler",
                adjudicator_private_key=self.regime_labeler_key(),
            )

    def test_process_log_payload_tampering_is_detected_by_rehash(self) -> None:
        process_log = self.scoring_process_log()
        object.__setattr__(process_log, "entries", ("rewritten after sealing",))

        with self.assertRaisesRegex(ValueError, "process-log artifact"):
            promotion_module.validate_process_log_artifact(process_log)

    def test_identical_small_metric_sample_retains_population_uncertainty(
        self,
    ) -> None:
        bound = promotion_module._bounded_event_mean_upper_bound(
            [-0.01] * 5,
            [f"event-{index}" for index in range(5)],
            self.policy(),
            family_size=1,
            absolute_bound=2.0,
        )

        self.assertGreater(bound, -0.01)

    def test_sample_preflight_fails_a_sparse_metric_cell(self) -> None:
        preflight = promotion_module.promotion_sample_size_preflight(
            self.plan(),
            self.policy(),
            available_physical_events=200,
            metric_cell_event_counts=(
                ("convective", "near_range", "log_echo_mse", 60, 5, 10),
            ),
            issuance_cell_event_counts=(
                ("convective", "near_range", 60, 200, 1),
            ),
        )

        self.assertFalse(preflight.cell_feasible)
        self.assertFalse(preflight.feasible)

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
            self.assertEqual(diagnostic[4], 1024)
            self.assertGreaterEqual(diagnostic[5], 1.0)
            self.assertEqual(diagnostic[8], 1)
            self.assertEqual(diagnostic[9], 1)

    def test_missing_required_band_metric_prevents_certification(self) -> None:
        support = promotion_module.MetricSupportContract.for_metric(
            "soft_fss_error_35",
            nowcast_config_digest="a" * 64,
            spatial_grid_digest="2" * 64,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
        )
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
            metric_support_contract_digests=(support.contract_digest,),
        )
        policy = replace(
            self.policy(),
            metric_scales=(
                PromotionMetricScale("log_echo_mse", 1.0, 0.01),
                PromotionMetricScale("soft_fss_error_35", 1.0, 0.01),
            ),
            metric_support_contracts=(
                self.policy().metric_support_contracts[0],
                support,
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

    def test_uncertainty_scoring_clips_before_event_aggregation(self) -> None:
        application = SimpleNamespace(
            truncated_location_dbz=torch.tensor([-10.0]),
            truncated_scale_dbz=torch.tensor([0.05]),
            event_probability=torch.tensor([1.0]),
        )
        scores = promotion_module._prior_uncertainty_scores(
            application,
            torch.tensor([5.5]),
            torch.tensor([True]),
            torch.tensor([True]),
            support_threshold_dbz=5.0,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
        )

        self.assertEqual(
            scores.echo_intensity_nll,
            promotion_module.UncertaintyScoreSupportContract().maximum_nll_score,
        )

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
        with self.assertRaisesRegex(ValueError, "exactly one candidate"):
            replace(
                self.plan(),
                candidate_family_digests=("c" * 64,)
                + tuple(character * 64 for character in "012345678"),
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
        evidence = self.deployment_ready(evidence)
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
        evidence = self.deployment_ready(evidence)
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
        evidence = self.deployment_ready(evidence)
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
        evidence = self.deployment_ready(evidence)
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
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        scoring_artifact, scoring_log, scoring_completion = self.sealed_scoring(
            evaluations,
            manifest=manifest,
            plan=plan,
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(
            ValueError,
            "pre-outcome holdout rule",
        ):
            compute_neural_prior_promotion(
                manifest,
                plan,
                evaluations,
                scoring_input_artifact=self.scoring_input_artifact(
                    plan=plan,
                    cases=manifest.holdout_cases,
                ),
                scoring_artifact=scoring_artifact,
                scoring_process_log=scoring_log,
                scoring_completion_receipt=scoring_completion,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_promotion_rejects_post_scoring_threshold_changes(self) -> None:
        plan = self.plan()
        manifest = self.manifest()
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        scoring_artifact, scoring_log, scoring_completion = self.sealed_scoring(
            evaluations,
            manifest=manifest,
            plan=plan,
        )
        changed = replace(
            self.policy(),
            maximum_harmful_fraction=0.99,
        )
        with (
            patch.object(
                promotion_module,
                "_load_learning_policy_trust_store",
                return_value=_LearningPolicyTrustStore(
                    approved_policy_digests=frozenset((changed.digest,)),
                    content_digest="b" * 64,
                ),
            ),
            self.assertRaisesRegex(
                ValueError,
                "pre-outcome holdout rule",
            ),
        ):
            compute_neural_prior_promotion(
                manifest,
                plan,
                evaluations,
                scoring_input_artifact=self.scoring_input_artifact(),
                scoring_artifact=scoring_artifact,
                scoring_process_log=scoring_log,
                scoring_completion_receipt=scoring_completion,
                policy=changed,
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
        evidence = self.deployment_ready(evidence)
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

    def test_v19_evaluation_loads_as_audit_only(self) -> None:
        payload = ledger_module._evaluation_audit_payload(
            self.evaluation(1, -0.2)
        )
        payload["contract"] = "prior-holdout-evaluation-v19"
        normalized = dict(payload)
        normalized.pop("evaluation_digest")
        for name, value in tuple(normalized.items()):
            if isinstance(value, dict) and value.get("kind") == "tensor":
                normalized[name] = value["digest"]
        payload["evaluation_digest"] = ledger_module._json_digest(normalized)

        decoded = ledger_module._decode_evaluation_audit_payloads([payload])

        self.assertIsInstance(
            decoded[0],
            ledger_module.LegacyPromotionEvaluationAudit,
        )
        self.assertFalse(decoded[0].statistical_reuse_permitted)

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
            digest="2" * 64,
            spatial_grid_digest="2" * 64,
            projected_displacement_xy=lambda value: value * 1000.0,
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
            radar_source_kind="single_site",
            radar_site_digest="a" * 64,
            radar_site_location_digest="a" * 64,
            radar_source_contract_digest="d" * 64,
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
            config=SimpleNamespace(
                digest="3" * 64,
                interval_minutes=10,
                min_dbz=-10.0,
                max_dbz=70.0,
            ),
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
            valid_mask=torch.ones((6, 2, 2), dtype=torch.bool),
            background_fallback_mask=torch.zeros(
                (6, 2, 2), dtype=torch.bool
            ),
            forecast_confidence=torch.ones((6, 2, 2)),
        )
        parent = SimpleNamespace(
            run=parent_run,
            state=object(),
            validate_issuance=Mock(),
            valid_mask=torch.tensor(
                [[[True, False], [False, False]]] * 6,
                dtype=torch.bool,
            ),
            background_fallback_mask=torch.zeros(
                (6, 2, 2), dtype=torch.bool
            ),
            forecast_confidence=torch.ones((6, 2, 2)),
        )
        resolved = SimpleNamespace(
            content_digest=case.verification_bundle_digest,
            valid_times=("2026-08-09T01:00:00Z",),
            grid_contract_digest="2" * 64,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            valid_mask=torch.tensor(
                [[[True, False], [False, False]]] * 6,
                dtype=torch.bool,
            ),
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
        candidate_weights = torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0]]]
        )
        parent_weights = candidate_weights.clone()
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
                operational_issuance_domain=self.issuance_domain(1),
            )
        self.assertAlmostEqual(float(evaluation.metric_change[0, 0]), -0.2)
        self.assertAlmostEqual(float(evaluation.end_to_end_metric_change[0, 0]), -0.25)
        self.assertEqual(evaluation.prior_uncertainty_sample_count, 4)
        self.assertAlmostEqual(evaluation.prior_candidate_valid_fraction, 0.25)
        self.assertAlmostEqual(evaluation.prior_candidate_valid_area_km2, 1.0)
        self.assertAlmostEqual(evaluation.prior_abstention_increase_vs_parent, 0.75)
        band = evaluation.range_band_evaluations[0]
        self.assertEqual(band.issuance_domain_cell_count_by_lead, (4,))
        self.assertEqual(band.newly_issued_count_by_lead, (3,))
        self.assertAlmostEqual(float(band.newly_issued_fraction_by_lead[0]), 0.75)
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
                operational_issuance_domain=self.issuance_domain(1),
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
                operational_issuance_domain=self.issuance_domain(1),
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
                operational_issuance_domain=self.issuance_domain(1),
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
                    operational_issuance_domain=self.issuance_domain(1),
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
                    operational_issuance_domain=self.issuance_domain(1),
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
                decision_rule = self.decision_rule()
                connection.execute(
                    "INSERT INTO neural_prior_promotion_rule_definitions "
                    "(rule_digest, payload_json, trust_store_digest, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        decision_rule.rule_digest,
                        json.dumps(
                            decision_rule.payload
                            | {"rule_digest": decision_rule.rule_digest},
                            sort_keys=True,
                        ),
                        "7" * 64,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plan_rule_bindings "
                    "(holdout_plan_digest, rule_digest, bound_at) "
                    "VALUES (?, ?, ?)",
                    (
                        plan.plan_digest,
                        decision_rule.rule_digest,
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
        changed_scoring_start = self.scoring_start_receipt_for(
            plan.physical_event_catalog_plan,
            changed_result,
        )
        changed_manifest = replace(
            self.manifest(),
            physical_event_catalog_result=changed_result,
            candidate_scoring_start_receipt=changed_scoring_start,
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
            object_track_artifact=self.track_artifact(
                start_time="2026-08-09T01:00:00Z",
                end_time="2026-08-09T03:00:00Z",
                start_centroid=(50_000.0, 50_000.0),
                end_centroid=(50_000.0, 50_000.0),
                artifact_seed="5",
                radar_ids=("radar-1",),
            ),
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
            metric_support_contract_digests=(
                self.policy().metric_support_contracts[0].contract_digest,
            ),
            maximum_end_to_end_mean_normalized_degradation=0.0,
            maximum_end_to_end_harmful_fraction_upper_bound=0.5,
        )
        policy = replace(
            self.policy(),
            required_range_metrics=(
                requirement,
                self.policy().required_range_metrics[1],
            ),
            bootstrap_samples=1024,
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
        decision_rule = self.decision_rule()
        result = self.event_catalog_result()
        scoring_input = self.scoring_input_artifact(plan=plan)
        receipt = promotion_module.TrustedProcessStartReceipt.from_plan(
            plan.physical_event_catalog_plan,
            catalog_result_digest=result.result_digest,
            process_kind="candidate_scoring",
            subject_digests=(scoring_input.artifact_digest,),
            process_algorithm_digest=plan.scoring_algorithm_digest,
            process_runtime_digest=plan.scoring_runtime_digest,
            execution_contract_digest=plan.scoring_execution_contract_digest,
            job_id="standalone-scoring-job",
            launch_nonce="f" * 64,
            scheduler_sequence_number=1,
            previous_receipt_digest=None,
            started_at="2026-08-12T01:00:00Z",
            scheduler_private_key=self.scheduler_key(),
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
                connection.execute(
                    "INSERT INTO neural_prior_promotion_rule_definitions "
                    "(rule_digest, payload_json, trust_store_digest, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        decision_rule.rule_digest,
                        json.dumps(
                            decision_rule.payload
                            | {"rule_digest": decision_rule.rule_digest},
                            sort_keys=True,
                        ),
                        "7" * 64,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plan_rule_bindings "
                    "(holdout_plan_digest, rule_digest, bound_at) "
                    "VALUES (?, ?, ?)",
                    (
                        plan.plan_digest,
                        decision_rule.rule_digest,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
            trust = self.scheduler_trust_store(plan.physical_event_catalog_plan)
            with patch.object(
                ledger_module,
                "_load_scheduler_trust_store",
                return_value=trust,
            ):
                with self.assertRaisesRegex(ValueError, "registered event catalog"):
                    ledger.append_trusted_process_start_receipt(
                        plan,
                        result,
                        receipt,
                        scoring_input_artifact=scoring_input,
                        scheduler_trust_store_path="/etc/advar/schedulers.json",
                    )

            with patch.object(
                ledger_module,
                "datetime",
                wraps=datetime,
            ) as catalog_datetime:
                catalog_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T00:10:00+00:00"
                )
                ledger.append_physical_event_catalog_result(plan, result)
            with patch.object(
                ledger_module,
                "_load_scheduler_trust_store",
                return_value=trust,
            ), self.assertRaisesRegex(ValueError, "ledger input artifact"):
                ledger.append_trusted_process_start_receipt(
                    plan,
                    result,
                    receipt,
                    scoring_input_artifact=scoring_input,
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )
            with (
                patch.object(
                    ledger_module,
                    "datetime",
                    wraps=datetime,
                ) as input_datetime,
            ):
                input_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T00:30:00+00:00"
                )
                ledger.append_neural_prior_holdout_scoring_input_artifact(
                    plan,
                    result,
                    scoring_input,
                )
            backdated = promotion_module.TrustedProcessStartReceipt.from_plan(
                plan.physical_event_catalog_plan,
                catalog_result_digest=result.result_digest,
                process_kind="candidate_scoring",
                subject_digests=(scoring_input.artifact_digest,),
                process_algorithm_digest=plan.scoring_algorithm_digest,
                process_runtime_digest=plan.scoring_runtime_digest,
                execution_contract_digest=plan.scoring_execution_contract_digest,
                job_id="backdated-scoring-job",
                launch_nonce="e" * 64,
                scheduler_sequence_number=1,
                previous_receipt_digest=None,
                started_at="2026-08-12T00:20:00Z",
                scheduler_private_key=self.scheduler_key(),
            )
            with (
                patch.object(
                    ledger_module,
                    "_load_scheduler_trust_store",
                    return_value=trust,
                ),
                patch.object(
                    ledger_module,
                    "datetime",
                    wraps=datetime,
                ) as backdated_datetime,
                self.assertRaisesRegex(
                    ValueError,
                    "before scoring input ledger append",
                ),
            ):
                backdated_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T02:00:00+00:00"
                )
                ledger.append_trusted_process_start_receipt(
                    plan,
                    result,
                    backdated,
                    scoring_input_artifact=scoring_input,
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )
            with (
                patch.object(
                    ledger_module,
                    "_load_scheduler_trust_store",
                    return_value=trust,
                ),
                patch.object(
                    ledger_module,
                    "datetime",
                    wraps=datetime,
                ) as trusted_datetime,
            ):
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-11T00:00:00+00:00"
                )
                with self.assertRaisesRegex(ValueError, "future start"):
                    ledger.append_trusted_process_start_receipt(
                        plan,
                        result,
                        receipt,
                        scoring_input_artifact=scoring_input,
                        scheduler_trust_store_path="/etc/advar/schedulers.json",
                    )

                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T02:00:00+00:00"
                )
                stored = ledger.append_trusted_process_start_receipt(
                    plan,
                    result,
                    receipt,
                    scoring_input_artifact=scoring_input,
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )
            self.assertEqual(stored, receipt.receipt_digest)

    def test_training_completion_requires_a_ledger_backed_start(self) -> None:
        start = self.training_start_receipt()
        completion = self.training_completion_receipt()
        trust = self.scheduler_trust_store(
            self.training_event_catalog_plan()
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with patch.object(
                ledger_module,
                "_load_scheduler_trust_store",
                return_value=trust,
            ), self.assertRaisesRegex(ValueError, "ledger start row"):
                ledger.append_trusted_process_completion_receipt(
                    start,
                    completion,
                    process_log_artifact=self.training_process_log(),
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )

    def test_scoring_algorithm_must_match_the_preregistered_plan(self) -> None:
        plan = self.plan()
        result = self.event_catalog_result()
        bad_start = promotion_module.TrustedProcessStartReceipt.from_plan(
            plan.physical_event_catalog_plan,
            catalog_result_digest=result.result_digest,
            process_kind="candidate_scoring",
            subject_digests=plan.candidate_family_digests,
            process_algorithm_digest="0" * 64,
            process_runtime_digest=plan.scoring_runtime_digest,
            execution_contract_digest=plan.scoring_execution_contract_digest,
            job_id="wrong-scoring-engine",
            launch_nonce="9" * 64,
            scheduler_sequence_number=2,
            previous_receipt_digest=self.training_start_receipt().receipt_digest,
            started_at="2026-08-12T01:00:00Z",
            scheduler_private_key=self.scheduler_key(),
        )
        bad_manifest = replace(
            self.manifest(),
            candidate_scoring_start_receipt=bad_start,
        )
        with self.assertRaisesRegex(ValueError, "scoring artifact disagrees"):
            self.scoring_artifact(
                (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
                manifest=bad_manifest,
                plan=plan,
            )

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
