from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
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
from advar.nowcast import (
    DataStatus,
    ForecastMetadata,
    RadarState,
    StatePathProvenance,
    TendencySource,
    _validate_input_plan_resolution,
    forecast_from_state as forecast_result_from_state,
)
from advar.physics import dbz_to_echo
from advar import (
    AnalysisConfig,
    CalibrationMetric,
    CalibrationRegime,
    EpisodeLedger,
    DeployedNeuralPriorPolicy,
    ForecastRunContract,
    NowcastConfig,
    OperationalCalibrationManifest,
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
    operational_runtime_profile_digest,
    load_forecast_run,
    save_forecast_run,
    variational_nowcast,
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


class _ReplayPrior(nn.Module):
    """Small deterministic seven-head product prior used by replay tests."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.register_buffer("offset", torch.tensor(offset))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = value + self.offset
        std = torch.ones_like(value)
        valid_probability = torch.ones_like(value)
        support_probability = torch.sigmoid(state - 5.0)
        event_probability = torch.sigmoid(value)
        return (
            state,
            std,
            valid_probability,
            support_probability,
            event_probability,
            state,
            std,
        )


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

    def classifier_subset_counts(
        self,
        count: int,
        *,
        overrides: dict[str, int] | None = None,
        policy: NeuralPriorPromotionPolicy | None = None,
    ) -> tuple[tuple[str, int], ...]:
        weather_strata, range_strata = promotion_module._registered_classifier_strata(
            self.plan(), self.policy() if policy is None else policy
        )
        names = (
            "known_weather",
            "known_range",
            "weather_ood",
            "range_ood",
            "brier_valid",
            *(f"known_weather:{regime}" for regime in weather_strata),
            *("known_range:" + ",".join(regimes) for regimes in range_strata),
        )
        retained = {} if overrides is None else overrides
        return tuple((name, retained.get(name, count)) for name in names)

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

        self.assertEqual(plan.contract, "neural-prior-holdout-plan-v19")
        self.assertTrue(
            all(
                item.contract == "neural-prior-range-band-contract-v3"
                for item in plan.range_band_contracts
            )
        )
        self.assertEqual(
            manifest.contract,
            "neural-prior-candidate-manifest-v13",
        )
        self.assertEqual(
            evaluation.contract,
            "prior-holdout-evaluation-v22",
        )
        self.assertTrue(
            all(
                band.contract == "neural-prior-range-band-evaluation-v7"
                for band in evaluation.range_band_evaluations
            )
        )
        self.assertEqual(policy.contract, "neural-prior-promotion-policy-v28")

    def test_semantic_replay_generation_reaches_promotion_and_deployment(
        self,
    ) -> None:
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        scoring = self.scoring_artifact(evaluations)
        evidence = self.compute(evaluations)
        deployment = DeployedNeuralPriorPolicy(
            candidate_prior_digest=evidence.candidate_prior_digest,
            parent_prior_digest=evidence.parent_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest="8" * 64,
            promotion_deployment_authority_trust_store_digest="7" * 64,
            regime_classifier_digest=(
                evidence.deployment_regime_classifier_digest
            ),
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=(
                evidence.certified_range_geometry_contract_digests[0]
            ),
        )

        self.assertEqual(
            scoring.contract,
            "neural-prior-holdout-scoring-artifact-v7",
        )
        self.assertEqual(evidence.contract, "neural-prior-promotion-evidence-v27")
        self.assertEqual(deployment.contract, "deployed-neural-prior-policy-v13")
        self.assertEqual(
            evidence.semantic_replay_generation_digest,
            promotion_module.SEMANTIC_SCORING_REPLAY_GENERATION_DIGEST,
        )
        self.assertEqual(
            deployment.semantic_replay_generation_digest,
            evidence.semantic_replay_generation_digest,
        )

    def test_deployment_certificate_binds_the_full_promotion_preimage(
        self,
    ) -> None:
        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        certificate, trust = self.deployment_certificate(evidence)

        promotion_module._validate_ledgered_promotion_deployment_certificate(
            certificate,
            authority_trust_store=trust,
            promotion_evidence=evidence,
        )
        minimal_evidence = {
            "contract": "neural-prior-promotion-evidence-v27",
            "deployment_eligible": True,
        }
        forged_values = dict(certificate.payload)
        forged_values.pop("authority_signature_hex")
        forged_values["promotion_evidence_payload_json"] = json.dumps(
            minimal_evidence,
            sort_keys=True,
            separators=(",", ":"),
        )
        forged_values["promotion_evidence_digest"] = promotion_module.json_digest(
            minimal_evidence
        )
        forged_values["ledger_chain_head_digest"] = (
            promotion_module._deployment_certificate_chain_head(
                ledger_instance_digest=certificate.ledger_instance_digest,
                sequence_number=certificate.sequence_number,
                promotion_evidence_digest=forged_values[
                    "promotion_evidence_digest"
                ],
                scoring_replay_bundle_digest=(
                    certificate.scoring_replay_bundle_digest
                ),
                scoring_artifact_digest=certificate.scoring_artifact_digest,
                scoring_completion_receipt_digest=(
                    certificate.scoring_completion_receipt_digest
                ),
                previous_certificate_digest=(
                    certificate.previous_certificate_digest
                ),
            )
        )
        authority_key = Ed25519PrivateKey.from_private_bytes(b"\x03" * 32)
        forged_values["authority_signature_hex"] = authority_key.sign(
            json.dumps(
                forged_values,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hex()
        forged = promotion_module._ledgered_promotion_deployment_certificate_from_payload(
            forged_values
        )
        with self.assertRaisesRegex(
            ValueError, "ledger receipt|complete current evidence"
        ):
            promotion_module._validate_ledgered_promotion_deployment_certificate(
                forged,
                authority_trust_store=trust,
            )
        object.__setattr__(certificate, "promotion_evidence_payload_json", "{}")
        with self.assertRaisesRegex(ValueError, "certificate integrity"):
            promotion_module._validate_ledgered_promotion_deployment_certificate(
                certificate,
                authority_trust_store=trust,
                promotion_evidence=evidence,
            )

        payload = evidence._payload()
        payload["contract"] = "neural-prior-promotion-evidence-v22"
        for name in (
            "scoring_replay_contract",
            "scoring_replay_method",
            "semantic_replay_generation_digest",
        ):
            payload.pop(name)
        digest = promotion_module.json_digest(payload)
        legacy = promotion_module.LegacyNeuralPriorPromotionEvidenceAuditV22(
            promotion_evidence_digest=digest,
            payload_json=json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        with self.assertRaisesRegex(TypeError, "current"):
            validate_neural_prior_promotion(legacy)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "replay generation"):
            replace(
                evidence,
                scoring_replay_method="builtin-semantic-scoring-recomputation-v2",
            )

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

    @staticmethod
    def input_grid(index: int) -> RadarGridTimeContract:
        issue = datetime.fromisoformat(
            f"2026-08-{8 + index:02d}T00:00:00+00:00"
        )
        return RadarGridTimeContract(
            valid_times=tuple(
                (issue - timedelta(minutes=offset)).isoformat()
                for offset in (20, 10, 0)
            ),
            dx_m=1_000.0,
            dy_m=1_000.0,
            projection="EPSG:3857",
            grid_hash=("1" if index == 1 else "2") * 64,
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
            grid_contract_digest=self.input_grid(index).digest,
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
        training_registry_receipt = self.training_raw_registry_receipt()
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
            training_raw_volume_identity_digests=("b" * 64,),
            training_sampling_unit_digests=("c" * 64,),
            training_raw_registry_receipt_digest=(
                training_registry_receipt.receipt_digest
            ),
            training_raw_registry_receipt_payload_json=json.dumps(
                training_registry_receipt.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
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

    def training_raw_registry_receipt(self):
        return promotion_module.TrainingRawRegistryReceipt.issue(
            raw_volume_identity_digests=("b" * 64,),
            sampling_unit_digests=("c" * 64,),
            registry_id="test-global-sampling-registry",
            authority_id="test-sampling-authority",
            authority_private_key=Ed25519PrivateKey.from_private_bytes(
                b"\x22" * 32
            ),
            committed_at="2026-06-30T00:00:00Z",
        )

    def plan(
        self,
        *,
        first_labeling_valid_time: str = "2026-08-09T01:00:00Z",
    ) -> NeuralPriorHoldoutPlan:
        input_plans = tuple(
            promotion_module.NeuralPriorInputPlan(
                valid_times=self.input_grid(index).valid_times,
                grid_contract_digest=self.input_grid(index).digest,
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
        raw_ingestor_key = Ed25519PrivateKey.from_private_bytes(b"\x21" * 32)
        processor_key = Ed25519PrivateKey.from_private_bytes(b"\x23" * 32)
        raw_slots = tuple(
            promotion_module.RawObservationSlotPlan(
                radar_site_digest="a" * 64,
                acquisition_valid_time=valid_time,
                scan_strategy_rule_digest="5" * 64,
                source_selection_rule_digest="d" * 64,
                canonical_geodetic_footprint_digest="6" * 64,
            )
            for input_plan in input_plans
            for valid_time in input_plan.valid_times
        )
        sampling_units = tuple(
            promotion_module.MeteorologicalSamplingUnit(
                raw_observation_slot_digests=tuple(
                    item.slot_digest
                    for item in raw_slots
                    if item.acquisition_valid_time in input_plan.valid_times
                ),
                canonical_geodetic_footprint_digest="6" * 64,
            )
            for input_plan in input_plans
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
                grid_contract_digest=self.input_grid(index).digest,
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
                grid_contract_digest=self.input_grid(index).digest,
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
                grid_contract_digest=self.input_grid(index).digest,
                range_geometry_contract_digest=(
                    range_geometries[index - 1].contract_digest
                ),
            )
            for index in (1, 2)
        )
        issuance_plans = tuple(
            promotion_module.OperationalIssuanceDomainPlan(
                case_id=f"case-{index}",
                grid_contract_digest=self.input_grid(index).digest,
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
                (first_labeling_valid_time, "2026-08-10T01:00:00Z"),
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
                    meteorological_sampling_unit_digest=(
                        sampling_units[0].sampling_unit_digest
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
                    meteorological_sampling_unit_digest=(
                        sampling_units[1].sampling_unit_digest
                    ),
                    issue_time="2026-08-10T00:00:00Z",
                ),
            )
        decision_rule_digest = self.decision_rule().rule_digest
        holdout_cohort_digest = promotion_module._holdout_dataset_digest(cases)
        trials = (
            promotion_module.PromotionExperimentTrial(
                candidate_prior_digest="c" * 64,
                promotion_decision_rule_digest=decision_rule_digest,
                classifier_manifest_digests=(
                    classifier_manifest.manifest_digest,
                ),
            ),
        )
        sampling_registry_key = Ed25519PrivateKey.from_private_bytes(b"\x22" * 32)
        training_registry_receipt = self.training_raw_registry_receipt()
        reservation = promotion_module.GlobalSamplingReservationReceipt.issue(
            experiment_scope_digest=(
                promotion_module._promotion_experiment_scope_digest(
                    holdout_cohort_digest=holdout_cohort_digest,
                    parent_prior_digest="d" * 64,
                    trials=trials,
                    winner_selection_rule_digest="f" * 64,
                )
            ),
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in raw_slots
            ),
            registry_id="test-global-sampling-registry",
            authority_id="test-sampling-authority",
            authority_private_key=sampling_registry_key,
            reserved_at="2026-08-06T00:00:00Z",
            registry_sequence_number=(
                training_registry_receipt.registry_sequence_number + 1
            ),
            previous_registry_root_digest=(
                training_registry_receipt.committed_registry_root_digest
            ),
        )
        experiment_family = promotion_module.PromotionExperimentFamily(
            holdout_cohort_digest=holdout_cohort_digest,
            meteorological_sampling_unit_digests=tuple(
                item.meteorological_sampling_unit_digest for item in cases
            ),
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in raw_slots
            ),
            global_sampling_reservation=reservation,
            parent_prior_digest="d" * 64,
            trials=trials,
            winner_selection_rule_digest="f" * 64,
        )
        return NeuralPriorHoldoutPlan(
            plan_id="holdout-plan",
            parent_prior_digest="d" * 64,
            candidate_family_digests=("c" * 64,),
            cases=cases,
            input_plans=input_plans,
            raw_observation_slot_plans=raw_slots,
            meteorological_sampling_units=sampling_units,
            raw_ingestor_trust_store=promotion_module.RawIngestorTrustStore(
                authorities=((
                    "test-raw-ingestor",
                    raw_ingestor_key.public_key().public_bytes_raw().hex(),
                ),),
            ),
            analysis_processor_id="test-analysis-processor",
            analysis_processor_public_key_hex=(
                processor_key.public_key().public_bytes_raw().hex()
            ),
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

    def resolved_raw_context(self, plan=None):
        plan = self.plan() if plan is None else plan
        raw_ingestor_key = Ed25519PrivateKey.from_private_bytes(b"\x21" * 32)
        input_plan_by_time = {
            valid_time: (case_index, input_plan, frame_index)
            for case_index, input_plan in enumerate(plan.input_plans, start=1)
            for frame_index, valid_time in enumerate(input_plan.valid_times)
        }
        receipts = tuple(
            promotion_module.ResolvedRawObservationReceipt.from_ingestor(
                slot=slot,
                raw_grid_volume=(
                    promotion_module.CanonicalRawGridVolumeArtifact.from_tensors(
                        reflectivity_dbz=torch.full(
                            (2, 2),
                            float(input_plan_by_time[slot.acquisition_valid_time][0])
                            + (-1.0, -0.5, 0.0)[
                                input_plan_by_time[slot.acquisition_valid_time][2]
                            ],
                        ),
                        qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
                        quality_weight=torch.ones((2, 2)),
                        observation_std_dbz=torch.full((2, 2), 2.0),
                        radar_site_digest=slot.radar_site_digest,
                        acquisition_valid_time=slot.acquisition_valid_time,
                        canonical_scan_identity_digest=(
                            slot.scan_strategy_rule_digest
                        ),
                        radar_product_digest=(
                            input_plan_by_time[slot.acquisition_valid_time][1]
                            .radar_product_digest
                        ),
                        grid_contract_digest=(
                            input_plan_by_time[slot.acquisition_valid_time][1]
                            .grid_contract_digest
                        ),
                    )
                ),
                raw_ingestor_id="test-raw-ingestor",
                raw_ingestor_private_key=raw_ingestor_key,
                received_at=(
                    datetime.fromisoformat(
                        slot.acquisition_valid_time.replace("Z", "+00:00")
                    )
                    + timedelta(seconds=30)
                ).isoformat(),
            )
            for index, slot in enumerate(
                plan.raw_observation_slot_plans,
                start=1,
            )
        )
        by_slot = {item.slot_plan_digest: item for item in receipts}
        sampling_by_digest = {
            item.sampling_unit_digest: item
            for item in plan.meteorological_sampling_units
        }
        registry_key = Ed25519PrivateKey.from_private_bytes(b"\x22" * 32)
        reservation = plan.promotion_experiment_family.global_sampling_reservation
        previous_root = reservation.committed_registry_root_digest
        next_sequence = reservation.registry_sequence_number + 1
        resolutions = {}
        for planned_case in sorted(plan.cases, key=lambda item: item.issue_time):
            sampling = sampling_by_digest[
                planned_case.meteorological_sampling_unit_digest
            ]
            case_receipts = tuple(
                by_slot[digest]
                for digest in sampling.raw_observation_slot_digests
            )
            resolved_at = (
                max(
                    promotion_module._canonical_datetime(
                        item.raw_volume_attestation.received_at
                    )
                    for item in case_receipts
                )
                + timedelta(seconds=10)
            ).isoformat()
            resolution = promotion_module.GlobalRawVolumeResolutionReceipt.issue(
                reservation=reservation,
                slot_identity_bindings=tuple(
                    (
                        item.slot_plan_digest,
                        item.raw_volume_identity.identity_digest,
                    )
                    for item in case_receipts
                ),
                authority_private_key=registry_key,
                resolved_at=resolved_at,
                registry_sequence_number=next_sequence,
                previous_registry_root_digest=previous_root,
            )
            resolutions[planned_case.case_id] = resolution
            next_sequence += 1
            previous_root = resolution.committed_registry_root_digest
        return receipts, resolutions

    def analysis_input_context(self, index: int, plan=None):
        retained_plan = self.plan() if plan is None else plan
        planned = retained_plan.cases[index - 1]
        input_plan = next(
            item
            for item in retained_plan.input_plans
            if item.plan_digest == planned.input_plan_digest
        )
        sampling_unit = next(
            item
            for item in retained_plan.meteorological_sampling_units
            if item.sampling_unit_digest
            == planned.meteorological_sampling_unit_digest
        )
        receipts, resolutions = self.resolved_raw_context(retained_plan)
        resolution = resolutions[planned.case_id]
        by_slot = {item.slot_plan_digest: item for item in receipts}
        case_receipts = tuple(
            by_slot[digest]
            for digest in sampling_unit.raw_observation_slot_digests
        )
        spatial = torch.full((2, 2), float(index))
        input_frames = torch.stack((spatial - 1.0, spatial - 0.5, spatial))
        input_valid = torch.ones_like(input_frames, dtype=torch.bool)
        input_quality = torch.ones_like(input_frames)
        observation_std = torch.full_like(input_frames, 2.0)
        config = NowcastConfig()
        grid = self.input_grid(index)
        data_identity = promotion_module.OperationalDataIdentity(
            radar_class="single-site-test",
            qc_pipeline_digest=input_plan.qc_pipeline_digest,
            observation_error_model_digest="2" * 64,
            background_model_digest="3" * 64,
            radar_product_digest=input_plan.radar_product_digest,
            background_cycle_rule_digest=(
                input_plan.background_cycle_rule_digest
            ),
            mask_policy_digest=input_plan.mask_policy_digest,
        )
        calibration = OperationalCalibrationManifest(
            calibration_id=f"replay-case-{index}",
            profile_kind="p0",
            expected_runtime_profile_digest=operational_runtime_profile_digest(
                config,
                grid,
            ),
            expected_algorithm_bundle_digest=algorithm_bundle_digest(),
            calibration_dataset_digest="5" * 64,
            validation_dataset_digest="6" * 64,
            data_identity=data_identity,
            training_period=(
                "2025-01-01T00:00:00Z",
                "2025-07-01T00:00:00Z",
            ),
            validation_period=(
                "2025-07-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
            validation_case_count=20,
            validation_regimes=(CalibrationRegime("convective", 20),),
            validation_metrics=(
                CalibrationMetric(
                    name="csi_35",
                    definition_digest="7" * 64,
                    direction="maximize",
                    acceptance_threshold=0.4,
                    value=0.5,
                ),
            ),
        )
        run_kwargs = {
            "observation_quality_weight": input_quality,
            "observation_std_dbz": observation_std,
            "grid_time_contract": grid,
            "operational_calibration_manifest_json": calibration.json,
            "operational_calibration_manifest_digest": calibration.digest,
            "operational_calibration_approval_digest": calibration.digest,
            "operational_data_identity_json": data_identity.json,
            "operational_data_identity_digest": data_identity.digest,
            "input_plan_json": input_plan.json,
            "input_plan_digest": input_plan.plan_digest,
        }
        base_run = ForecastRunContract.from_inputs(
            config,
            input_frames,
            input_valid,
            None,
            **run_kwargs,
        )
        derivation = promotion_module.AnalysisInputDerivationArtifact.from_products(
            case_id=planned.case_id,
            input_plan=input_plan,
            resolved_raw_observations=case_receipts,
            global_resolution_receipt=resolution,
            run=base_run,
            resolved_source_coverage=None,
            background_frames_dbz=None,
            processed_at=(
                promotion_module._canonical_datetime(resolution.resolved_at)
                + timedelta(seconds=10)
            ).isoformat(),
            processor_id=retained_plan.analysis_processor_id,
            processor_private_key=Ed25519PrivateKey.from_private_bytes(
                b"\x23" * 32
            ),
        )
        derivation_json = json.dumps(
            derivation.payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        run_kwargs.update(
            analysis_input_derivation_artifact_json=derivation_json,
            analysis_input_derivation_artifact_digest=derivation.artifact_digest,
        )
        base_run = ForecastRunContract.from_inputs(
            config,
            input_frames,
            input_valid,
            None,
            **run_kwargs,
        )
        return (
            input_frames,
            input_valid,
            input_quality,
            config,
            run_kwargs,
            base_run,
            derivation,
            case_receipts,
            resolution,
        )

    def completed_case(self, index: int) -> NeuralPriorHoldoutCase:
        plan = self.plan()
        planned = plan.cases[index - 1]
        (
            _,
            _,
            _,
            _,
            _,
            base_run,
            derivation,
            case_receipts,
            resolution,
        ) = self.analysis_input_context(index, plan)
        uncertainty_target = self.uncertainty_target(index)
        state_target = self.state_target(index)
        assert base_run.full_analysis_input_digest is not None
        full_digest = base_run.full_analysis_input_digest
        return NeuralPriorHoldoutCase(
            case_id=planned.case_id,
            planned_storm_id=planned.storm_id,
            storm_id=f"storm-{index}",
            physical_event_digest=(
                self.event_catalog(index).physical_event_identity_digest
            ),
            meteorological_sampling_unit_digest=(
                planned.meteorological_sampling_unit_digest
            ),
            day=planned.day,
            radar_id=planned.radar_id,
            planned_regime=planned.regime,
            regime="convective" if index == 1 else "stratiform",
            range_regime=planned.range_regime,
            dynamic_range_source_resolution=False,
            input_plan_digest=planned.input_plan_digest,
            input_plan_resolution_digest=(
                promotion_module._forecast_input_plan_resolution_digest(
                    input_plan_digest=planned.input_plan_digest,
                    full_analysis_input_digest=full_digest,
                )
            ),
            input_bundle_digest=base_run.input_bundle_digest,
            full_analysis_input_digest=full_digest,
            analysis_input_derivation_artifact_digest=derivation.artifact_digest,
            resolved_raw_volume_identity_digests=tuple(
                sorted(
                    item.raw_volume_identity.identity_digest
                    for item in case_receipts
                )
            ),
            global_raw_resolution_receipt_digest=resolution.receipt_digest,
            fixed_input_context_digest=base_run.fixed_input_context_digest,
            observation_quality_weight_digest=(
                base_run.observation_quality_weight_digest
            ),
            observation_std_dbz_digest=base_run.observation_std_dbz_digest,
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
        base_run = self.analysis_input_context(index, self.plan())[5]
        assert base_run.full_analysis_input_digest is not None
        return promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id=f"physical-event-{index}",
            member_case_ids=(f"case-{index}",),
            member_full_analysis_input_digests=(
                base_run.full_analysis_input_digest,
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
                    full_analysis_input_digest=(
                        first.member_full_analysis_input_digests[0]
                    ),
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    second,
                    case_id="case-2",
                    full_analysis_input_digest=(
                        second.member_full_analysis_input_digests[0]
                    ),
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
                    training_raw_registry_receipt_digest=(
                        self.training_raw_registry_receipt().receipt_digest
                    ),
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

    def mosaic_issuance_context(self):
        holdout_plan = self.plan()
        input_plan = holdout_plan.input_plans[0]
        registry = promotion_module.SourceRadarRegistry(
            radar_site_digests=("a" * 64, "b" * 64),
            source_selection_policy_digest="2" * 64,
        )
        issuance_plan = replace(
            holdout_plan.operational_issuance_domain_plans[0],
            radar_source_kind="mosaic",
            source_radar_registry_digest=registry.registry_digest,
            source_radar_count=2,
            data_ingestor_id="trusted-radar-ingestor",
            data_ingestor_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.scheduler_key()
                )
            ),
        )
        return issuance_plan, input_plan, registry

    def scoring_replay_cases(self, evaluations, *, manifest=None, plan=None):
        retained_manifest = self.manifest() if manifest is None else manifest
        retained_plan = self.plan() if plan is None else plan
        cases = []
        for index, evaluation in enumerate(evaluations, start=1):
            (
                input_frames,
                input_valid,
                input_quality,
                config,
                run_kwargs,
                base_run,
                analysis_input_derivation,
                resolved_raw_observations,
                global_raw_resolution,
            ) = self.analysis_input_context(index, retained_plan)
            spatial = input_frames[-1]

            def prior_runner(offset: float, training_digest: str):
                return promotion_module.NeuralPriorInferenceRunner(
                    _ReplayPrior(offset).eval(),
                    lambda value: value[-1],
                    example_frames=input_frames,
                    model_contract_digest=retained_manifest.model_contract_digest,
                    feature_schema_digest=retained_manifest.feature_schema_digest,
                    training_manifest_digest=training_digest,
                    state_contract=self.state_contract(),
                    probability_contract=self.probability_contract(),
                    dependency="radar_dependent",
                    allow_constant_uncertainty=False,
                )

            candidate_runner = prior_runner(
                0.1,
                retained_manifest.candidate_training_manifest_digest,
            )
            parent_runner = prior_runner(
                0.0,
                retained_manifest.parent_training_manifest_digest,
            )
            candidate_application = candidate_runner.infer(
                input_frames,
                input_run=base_run,
                role="candidate",
            )
            parent_application = parent_runner.infer(
                input_frames,
                input_run=base_run,
                role="parent",
            )

            def forecast_result(runner, application, role):
                evidence = application.inference_evidence
                run = ForecastRunContract.from_inputs(
                    config,
                    input_frames,
                    input_valid,
                    None,
                    **run_kwargs,
                    neural_prior_digest=runner.neural_prior_digest,
                    prior_application_digest=application.application_digest,
                    prior_model_contract_digest=evidence.model_contract_digest,
                    prior_feature_schema_digest=evidence.feature_schema_digest,
                    prior_training_manifest_digest=evidence.training_manifest_digest,
                    prior_inference_evidence_digest=evidence.evidence_digest,
                    prior_inference_algorithm_digest=(
                        evidence.inference_algorithm_digest
                    ),
                    prior_numerical_runtime_digest=evidence.numerical_runtime_digest,
                    prior_dependency=evidence.dependency,
                    prior_role=role,
                )
                state = RadarState(
                    echo_linear=dbz_to_echo(
                        spatial,
                        min_dbz=config.min_dbz,
                        max_dbz=config.max_dbz,
                    ),
                    displacement_yx=torch.zeros(2),
                    log_growth_per_step=torch.zeros(()),
                )
                metadata = ForecastMetadata(
                    data_status=DataStatus.OBSERVED,
                    coverage_by_frame=torch.ones(3),
                    background_used=False,
                    background_contribution_fraction=0.0,
                    background_age_minutes=None,
                    source_support=torch.ones_like(spatial),
                    observation_source_support=torch.ones_like(spatial),
                    background_source_support=torch.zeros_like(spatial),
                    path_verified_source_support=torch.ones_like(spatial),
                    verified_source_support=torch.ones_like(spatial),
                    local_motion_verified_support=torch.ones_like(spatial),
                    local_growth_verified_support=torch.ones_like(spatial),
                    local_dynamics_verified_support=torch.ones_like(spatial),
                    observation_verified_source_support=torch.ones_like(spatial),
                    background_verified_source_support=torch.zeros_like(spatial),
                    motion_disagreement_px=torch.zeros(()),
                    motion_disagreement_mps=torch.full((), torch.nan),
                    growth_disagreement=torch.zeros(()),
                    maximum_growth_saturation_excess=torch.zeros(()),
                    posterior_velocity_uncertainty_mps=torch.full((), torch.nan),
                    posterior_log_growth_uncertainty_per_step=torch.full(
                        (), torch.nan
                    ),
                    p1_velocity_saturation_uncertainty_mps=torch.full(
                        (), torch.nan
                    ),
                    p1_log_growth_saturation_uncertainty_per_step=torch.full(
                        (), torch.nan
                    ),
                    minimum_phase_correlation_psr=torch.tensor(10.0),
                    tendency_pair_count=2,
                    tendency_source=TendencySource.OBSERVATION,
                    state_path_source=TendencySource.OBSERVATION,
                    state_path_age_minutes=0.0,
                    observation_path=StatePathProvenance(
                        age_minutes=0.0
                    ),
                    minimum_growth_overlap_support=float(spatial.numel()),
                    minimum_growth_overlap_area_km2=float(spatial.numel()),
                )
                return forecast_result_from_state(
                    state,
                    metadata,
                    config,
                    run=run,
                )

            candidate_forecast = forecast_result(
                candidate_runner,
                candidate_application,
                "candidate",
            )
            parent_forecast = forecast_result(
                parent_runner,
                parent_application,
                "parent",
            )
            classifier = NeuralPriorRegimeClassifier(
                _FixedRegimeClassifier(
                    (2.0, 0.0, -1.0),
                    (1.0, -1.0),
                ).eval(),
                example_frames=input_frames,
                regime_labels=("convective", "stratiform", "unknown"),
                range_regime_labels=("near_range", "far_range"),
                classifier_algorithm_digest="5" * 64,
            )
            verification = VerificationBundle(
                frames_dbz=spatial.unsqueeze(0),
                valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
                valid_times=(f"2026-08-{8 + index:02d}T01:00:00Z",),
                grid_contract_digest="2" * 64,
                radar_product_digest="a" * 64,
                qc_pipeline_digest="9" * 64,
            )
            cases.append(
                promotion_module.ScoringReplayCaseArtifact.from_products(
                    manifest=retained_manifest,
                    plan=retained_plan,
                    case_id=evaluation.case_id,
                    candidate_forecast=candidate_forecast,
                    parent_forecast=parent_forecast,
                    verification=verification,
                    metric_config=promotion_module.SensitivityConfig(
                        metric_names=("log_echo_mse",),
                        full_map_lead_minutes=(60,),
                    ),
                    candidate_prior_application=candidate_application,
                    parent_prior_application=parent_application,
                    candidate_prior_runner=candidate_runner,
                    parent_prior_runner=parent_runner,
                    input_frames_dbz=input_frames,
                    input_qc_valid_mask=input_valid,
                    input_quality_weight=input_quality,
                    background_frames_dbz=None,
                    uncertainty_target=self.uncertainty_target(index),
                    state_calibration_target=self.state_target(index),
                    regime_classifier=classifier,
                    regime_classifier_manifest=self.classifier_manifest(),
                    range_grid_x_m=torch.tensor(
                        [[0.0, 1_000.0], [0.0, 1_000.0]]
                    ),
                    range_grid_y_m=torch.tensor(
                        [[0.0, 0.0], [1_000.0, 1_000.0]]
                    ),
                    operational_issuance_domain=self.issuance_domain(index),
                    analysis_input_derivation=analysis_input_derivation,
                    resolved_raw_observations=resolved_raw_observations,
                    global_raw_resolution_receipt=global_raw_resolution,
                )
            )
        return tuple(cases)

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

    def test_semantic_replay_role_schema_rejects_wrong_dtype(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        tensors = self.scoring_replay_cases((evaluation,))[0].replay_tensors()
        tensors["verification_valid_mask"] = tensors[
            "verification_valid_mask"
        ].to(torch.float32)
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            ledger_module._validate_scoring_replay_case_tensors(
                tensors,
                dynamic_source=False,
                background_present=False,
            )

    def test_semantic_replay_requires_one_device_across_every_role(self) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]
        tensors = case.replay_tensors()
        tensors["verification_valid_mask"] = torch.empty(
            tensors["verification_valid_mask"].shape,
            dtype=torch.bool,
            device="meta",
        )

        with self.assertRaisesRegex(ValueError, "multiple tensor devices"):
            ledger_module._semantic_replay_execution_device(
                (case,),
                {case.case_id: tensors},
            )

    def test_scoring_backend_certification_is_device_specific(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "CPU scoring cannot claim MPS certification",
        ):
            ledger_module._validate_scoring_backend_certification(
                torch.device("cpu"),
                Mock(),
                Mock(),
            )
        with self.assertRaisesRegex(
            ValueError,
            "automatic promotion scoring requires the certified CPU backend",
        ):
            ledger_module._validate_scoring_backend_certification(
                torch.device("mps"),
                None,
                None,
            )

    def test_full_product_semantic_replay_reaches_promotion_without_scorer_patch(
        self,
    ) -> None:
        """Exercise the product scorer, not a prepared evaluation callback."""

        case_id = "semantic-product-case"
        issue_time = "2026-08-09T00:20:00Z"
        verification_time = "2026-08-09T01:20:00Z"
        forecast_valid_times = tuple(
            f"2026-08-09T{hour:02d}:{minute:02d}:00Z"
            for hour, minute in (
                (0, 30),
                (0, 40),
                (0, 50),
                (1, 0),
                (1, 10),
                (1, 20),
            )
        )
        input_frames = torch.tensor(
            [
                [[8.0, 1.0], [8.0, 1.0]],
                [[9.0, 1.0], [9.0, 1.0]],
                [[10.0, 1.0], [10.0, 1.0]],
            ]
        )
        input_valid = torch.ones_like(input_frames, dtype=torch.bool)
        input_quality = torch.ones_like(input_frames)
        observation_std = torch.full_like(input_frames, 2.0)
        config = NowcastConfig(horizon_minutes=60)
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-09T00:00:00Z",
                "2026-08-09T00:10:00Z",
                issue_time,
            ),
            dx_m=1_000.0,
            dy_m=1_000.0,
            projection="EPSG:3857",
            grid_hash="4" * 64,
        )
        grid_x = torch.tensor([[0.0, 1_000.0], [0.0, 1_000.0]])
        grid_y = torch.tensor([[0.0, 0.0], [1_000.0, 1_000.0]])
        source_map = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
        source_registry = promotion_module.SourceRadarRegistry(
            radar_site_digests=("a" * 64, "e" * 64),
            source_selection_policy_digest="2" * 64,
        )
        location_registry = promotion_module.RadarSiteLocationRegistry(
            projection_digest="7" * 64,
            radar_site_digests=source_registry.radar_site_digests,
            radar_site_location_digests=("b" * 64, "f" * 64),
            radar_projected_xy_m=((0.0, 0.0), (1_000.0, 0.0)),
        )
        effective_range = torch.tensor(
            [[0.0, 0.0], [1_000.0, 1_000.0]],
            dtype=torch.float64,
        )
        input_plan = promotion_module.NeuralPriorInputPlan(
            valid_times=grid.valid_times,
            grid_contract_digest=grid.digest,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            background_cycle_rule_digest="1" * 64,
            mask_policy_digest="3" * 64,
            observation_valid_time=issue_time,
            input_available_time=issue_time,
            decision_deadline="2026-08-09T00:22:00Z",
            publication_time="2026-08-09T00:25:00Z",
        )
        semantic_raw_key = Ed25519PrivateKey.from_private_bytes(b"\x31" * 32)
        semantic_processor_key = Ed25519PrivateKey.from_private_bytes(
            b"\x33" * 32
        )
        semantic_sampling_registry_key = Ed25519PrivateKey.from_private_bytes(
            b"\x32" * 32
        )
        semantic_training_registry_receipt = (
            promotion_module.TrainingRawRegistryReceipt.issue(
                raw_volume_identity_digests=("b" * 64,),
                sampling_unit_digests=("c" * 64,),
                registry_id="semantic-global-sampling-registry",
                authority_id="semantic-sampling-authority",
                authority_private_key=semantic_sampling_registry_key,
                committed_at="2026-06-29T00:00:00Z",
            )
        )
        raw_slots = tuple(
            promotion_module.RawObservationSlotPlan(
                radar_site_digest=radar_site_digest,
                acquisition_valid_time=valid_time,
                scan_strategy_rule_digest="5" * 64,
                source_selection_rule_digest=(
                    source_registry.source_selection_policy_digest
                ),
                canonical_geodetic_footprint_digest="6" * 64,
            )
            for index, (valid_time, radar_site_digest) in enumerate(
                (
                    (valid_time, radar_site_digest)
                    for valid_time in grid.valid_times
                    for radar_site_digest in source_registry.radar_site_digests
                ),
                start=1,
            )
        )
        sampling_unit = promotion_module.MeteorologicalSamplingUnit(
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in raw_slots
            ),
            canonical_geodetic_footprint_digest="6" * 64,
        )
        data_identity = promotion_module.OperationalDataIdentity(
            radar_class="mosaic",
            qc_pipeline_digest="9" * 64,
            observation_error_model_digest="2" * 64,
            background_model_digest="3" * 64,
            radar_product_digest="a" * 64,
            background_cycle_rule_digest="1" * 64,
            mask_policy_digest="3" * 64,
            radar_source_kind="mosaic",
            radar_source_contract_digest=source_registry.registry_digest,
            source_radar_index_map_digest=promotion_module.tensor_digest(
                source_map
            ),
            effective_horizontal_range_map_digest=(
                promotion_module.tensor_digest(effective_range)
            ),
            source_selection_policy_digest=(
                source_registry.source_selection_policy_digest
            ),
            outage_mask_digest=promotion_module.tensor_digest(
                torch.zeros((2, 2), dtype=torch.bool)
            ),
            dynamic_qc_valid_mask_digest=promotion_module.tensor_digest(
                torch.ones((2, 2), dtype=torch.bool)
            ),
        )
        calibration = OperationalCalibrationManifest(
            calibration_id="semantic-replay-p0",
            profile_kind="p0",
            expected_runtime_profile_digest=operational_runtime_profile_digest(
                config,
                grid,
            ),
            expected_algorithm_bundle_digest=algorithm_bundle_digest(),
            calibration_dataset_digest="5" * 64,
            validation_dataset_digest="6" * 64,
            data_identity=data_identity,
            training_period=(
                "2025-01-01T00:00:00Z",
                "2025-07-01T00:00:00Z",
            ),
            validation_period=(
                "2025-07-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
            validation_case_count=20,
            validation_regimes=(CalibrationRegime("convective", 20),),
            validation_metrics=(
                CalibrationMetric(
                    name="csi_35",
                    definition_digest="7" * 64,
                    direction="maximize",
                    acceptance_threshold=0.4,
                    value=0.5,
                ),
            ),
        )
        run_kwargs = {
            "observation_quality_weight": input_quality,
            "observation_std_dbz": observation_std,
            "grid_time_contract": grid,
            "operational_calibration_manifest_json": calibration.json,
            "operational_calibration_manifest_digest": calibration.digest,
            "operational_calibration_approval_digest": calibration.digest,
            "operational_data_identity_json": data_identity.json,
            "operational_data_identity_digest": data_identity.digest,
            "input_plan_json": input_plan.json,
            "input_plan_digest": input_plan.plan_digest,
        }
        base_run = ForecastRunContract.from_inputs(
            config,
            input_frames,
            input_valid,
            None,
            **run_kwargs,
        )

        def prior_runner(offset: float, training_digest: str):
            return promotion_module.NeuralPriorInferenceRunner(
                _ReplayPrior(offset).eval(),
                lambda value: value[-1],
                example_frames=input_frames,
                model_contract_digest="2" * 64,
                feature_schema_digest="4" * 64,
                training_manifest_digest=training_digest,
                state_contract=self.state_contract(),
                probability_contract=self.probability_contract(),
                dependency="radar_dependent",
                allow_constant_uncertainty=False,
                feature_exclusion_mask=torch.ones_like(
                    input_frames, dtype=torch.bool
                ),
            )

        candidate_runner = prior_runner(0.0, "2" * 64)
        parent_runner = prior_runner(0.1, "3" * 64)
        candidate_application = candidate_runner.infer(
            input_frames,
            input_run=base_run,
            role="candidate",
        )
        parent_application = parent_runner.infer(
            input_frames,
            input_run=base_run,
            role="parent",
        )

        def product_forecast(runner, application, role):
            evidence = application.inference_evidence
            run = ForecastRunContract.from_inputs(
                config,
                input_frames,
                input_valid,
                None,
                **run_kwargs,
                neural_prior_digest=runner.neural_prior_digest,
                prior_application_digest=application.application_digest,
                prior_model_contract_digest=evidence.model_contract_digest,
                prior_feature_schema_digest=evidence.feature_schema_digest,
                prior_training_manifest_digest=evidence.training_manifest_digest,
                prior_inference_evidence_digest=evidence.evidence_digest,
                prior_inference_algorithm_digest=(
                    evidence.inference_algorithm_digest
                ),
                prior_numerical_runtime_digest=evidence.numerical_runtime_digest,
                prior_dependency=evidence.dependency,
                prior_role=role,
            )
            state = RadarState(
                echo_linear=dbz_to_echo(
                    input_frames[-1]
                    + (0.0 if role == "candidate" else 2.0),
                    min_dbz=config.min_dbz,
                    max_dbz=config.max_dbz,
                ),
                displacement_yx=torch.zeros(2),
                log_growth_per_step=torch.zeros(()),
            )
            metadata = ForecastMetadata(
                data_status=DataStatus.OBSERVED,
                coverage_by_frame=torch.ones(3),
                background_used=False,
                background_contribution_fraction=0.0,
                background_age_minutes=None,
                source_support=torch.ones((2, 2)),
                observation_source_support=torch.ones((2, 2)),
                background_source_support=torch.zeros((2, 2)),
                path_verified_source_support=torch.ones((2, 2)),
                verified_source_support=torch.ones((2, 2)),
                local_motion_verified_support=torch.ones((2, 2)),
                local_growth_verified_support=torch.ones((2, 2)),
                local_dynamics_verified_support=torch.ones((2, 2)),
                observation_verified_source_support=torch.ones((2, 2)),
                background_verified_source_support=torch.zeros((2, 2)),
                motion_disagreement_px=torch.zeros(()),
                motion_disagreement_mps=torch.full((), torch.nan),
                growth_disagreement=torch.zeros(()),
                maximum_growth_saturation_excess=torch.zeros(()),
                posterior_velocity_uncertainty_mps=torch.full((), torch.nan),
                posterior_log_growth_uncertainty_per_step=torch.full(
                    (), torch.nan
                ),
                p1_velocity_saturation_uncertainty_mps=torch.full(
                    (), torch.nan
                ),
                p1_log_growth_saturation_uncertainty_per_step=torch.full(
                    (), torch.nan
                ),
                minimum_phase_correlation_psr=torch.tensor(10.0),
                tendency_pair_count=2,
                tendency_source=TendencySource.OBSERVATION,
                state_path_source=TendencySource.OBSERVATION,
                state_path_age_minutes=0.0,
                observation_path=StatePathProvenance(age_minutes=0.0),
                minimum_growth_overlap_support=4.0,
                minimum_growth_overlap_area_km2=4.0,
            )
            return forecast_result_from_state(state, metadata, config, run=run)

        candidate_forecast = product_forecast(
            candidate_runner,
            candidate_application,
            "candidate",
        )
        parent_forecast = product_forecast(
            parent_runner,
            parent_application,
            "parent",
        )
        metric_config = promotion_module.SensitivityConfig(
            metric_names=("log_echo_mse",),
            full_map_lead_minutes=(60,),
        )
        metric_support = promotion_module.MetricSupportContract.from_run(
            "log_echo_mse",
            candidate_forecast.run,
            metric_engine_digest=(
                promotion_module.scoring_metric_engine_identity_digest()
            ),
            grid_shape=(2, 2),
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier(
                (12.0, 0.0, -12.0),
                (12.0, -12.0),
            ).eval(),
            example_frames=input_frames,
            regime_labels=("convective", "stratiform", "unknown"),
            range_regime_labels=("near_range", "far_range"),
            classifier_algorithm_digest="5" * 64,
        )
        classifier_manifest = promotion_module.RegimeClassifierManifest(
            classifier_digest=classifier.classifier_digest,
            training_dataset_digest="4" * 64,
            training_case_ids=("classifier-training-case",),
            training_input_bundle_digests=("9" * 64,),
            training_full_analysis_input_digests=("8" * 64,),
            training_physical_event_digests=("4" * 64,),
            training_storm_ids=("classifier-training-storm",),
            training_days=("2026-06-01",),
            training_radar_ids=("classifier-radar",),
            training_grid_contract_digests=("6" * 64,),
            training_raw_volume_identity_digests=("b" * 64,),
            training_sampling_unit_digests=("c" * 64,),
            training_raw_registry_receipt_digest=(
                semantic_training_registry_receipt.receipt_digest
            ),
            training_raw_registry_receipt_payload_json=json.dumps(
                semantic_training_registry_receipt.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
            training_time_windows=((
                "2026-06-01T00:00:00Z",
                "2026-06-01T01:00:00Z",
            ),),
            training_algorithm_digest="5" * 64,
            numerical_runtime_digest=classifier.numerical_runtime_digest,
            reference_label_contract_digest="7" * 64,
            signed_training_member_manifest_digest="5" * 64,
        )
        target_time = grid.valid_times[0]
        target_plan = PriorUncertaintyTargetPlan(
            plan_id="semantic-uncertainty",
            target_kind="independent_sensor",
            source_identity_digest="6" * 64,
            qc_pipeline_digest="9" * 64,
            mask_policy_digest="3" * 64,
            censor_policy_digest=self.state_contract().state_censor_policy_digest,
            floor_representation_contract_digest="e" * 64,
            grid_contract_digest=grid.digest,
            feature_exclusion_contract_digest=(
                candidate_runner.feature_exclusion_contract_digest
            ),
            independence_evidence_digest="8" * 64,
            target_valid_time=target_time,
            prior_probability_contract_digest=(
                self.probability_contract().contract_digest
            ),
        )
        state_target_plan = NeuralPriorStateCalibrationPlan(
            plan_id="semantic-state",
            target_kind="withheld_target_mask",
            source_identity_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            mask_policy_digest="3" * 64,
            censor_policy_digest=self.state_contract().state_censor_policy_digest,
            floor_representation_contract_digest="e" * 64,
            grid_contract_digest=grid.digest,
            feature_exclusion_contract_digest=(
                candidate_runner.feature_exclusion_contract_digest
            ),
            independence_evidence_digest="8" * 64,
            target_valid_time=target_time,
            state_contract_digest=self.state_contract().contract_digest,
            support_threshold_dbz=5.0,
        )
        target_verification = VerificationBundle(
            frames_dbz=torch.tensor([[[10.0, 1.0], [10.0, 1.0]]]),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_time,),
            grid_contract_digest=grid.digest,
            radar_product_digest="6" * 64,
            qc_pipeline_digest="9" * 64,
            mask_policy_digest="3" * 64,
            censor_policy_digest=self.state_contract().state_censor_policy_digest,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
            threshold_bin_convention="nearest_rounding_threshold_censor",
            floor_representation_contract_digest="e" * 64,
            contract="radar-verification-bundle-v3",
        )
        state_verification = replace(
            target_verification,
            radar_product_digest="a" * 64,
        )
        uncertainty_target = PriorUncertaintyTarget.from_verification_bundle(
            plan=target_plan,
            verification=target_verification,
        )
        state_target = NeuralPriorStateCalibrationTarget.from_verification_bundle(
            plan=state_target_plan,
            verification=state_verification,
        )
        verification = VerificationBundle(
            frames_dbz=input_frames[-1].repeat((6, 1, 1)),
            valid_mask=torch.ones((6, 2, 2), dtype=torch.bool),
            valid_times=forecast_valid_times,
            grid_contract_digest=grid.digest,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            mask_policy_digest="3" * 64,
            censor_policy_digest=self.state_contract().state_censor_policy_digest,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
            threshold_bin_convention="nearest_rounding_threshold_censor",
            floor_representation_contract_digest="e" * 64,
            contract="radar-verification-bundle-v3",
        )
        range_geometry = promotion_module.MosaicRangeGeometryContract.from_registry(
            source_registry,
            location_registry,
            grid_contract_digest=grid.digest,
            projection_digest=location_registry.projection_digest,
            range_regime_labels=("near_range", "far_range"),
            radial_distance_edges_m=(0.0, 30_000.0, 100_000.0),
            horizontal_range_rule_digest="c" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(grid_x),
            grid_y_m_digest=promotion_module.tensor_digest(grid_y),
        )
        range_partition, resolved_effective_range = (
            promotion_module.resolve_mosaic_range_geometry(
                range_geometry,
                grid_x_m=grid_x,
                grid_y_m=grid_y,
                source_radar_index_map=source_map,
            )
        )
        self.assertTrue(torch.equal(resolved_effective_range, effective_range))
        range_contract = promotion_module.RangeBandContract(
            case_id=case_id,
            range_regime_labels=("near_range", "far_range"),
            range_band_mask_digests=(),
            reference_active_range_regimes=(),
            grid_contract_digest=grid.digest,
            range_geometry_contract_digest=range_geometry.contract_digest,
            dynamic_source_resolution=True,
            registered_active_range_regime_sets=(
                ("far_range",),
                ("near_range",),
                ("far_range", "near_range"),
            ),
        )
        publication = torch.ones((1, 2, 2), dtype=torch.bool)
        issuance_plan = promotion_module.OperationalIssuanceDomainPlan(
            case_id=case_id,
            grid_contract_digest=grid.digest,
            radar_source_contract_digest=source_registry.registry_digest,
            lead_minutes=(60,),
            publication_policy_digest="1" * 64,
            source_coverage_policy_digest=(
                source_registry.source_selection_policy_digest
            ),
            permanent_exclusion_policy_digest="3" * 64,
            publication_eligible_mask_digest=(
                promotion_module.tensor_digest(publication)
            ),
            source_coverage_mask_digest=(
                promotion_module.tensor_digest(publication)
            ),
            permanent_exclusion_mask_digest=(
                promotion_module.tensor_digest(torch.zeros_like(publication))
            ),
            radar_source_kind="mosaic",
            source_radar_registry_digest=source_registry.registry_digest,
            source_radar_count=len(source_registry.radar_site_digests),
            data_ingestor_id="trusted-radar-ingestor",
            data_ingestor_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.scheduler_key()
                )
            ),
        )
        resolved_coverage = (
            promotion_module.ResolvedSourceCoverageArtifact.from_observations(
                issuance_plan,
                input_plan,
                source_registry,
                nominal_source_coverage_mask=publication,
                source_radar_index_map=source_map,
                outage_mask=torch.zeros((2, 2), dtype=torch.bool),
                dynamic_qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
                input_bundle_digest=base_run.input_bundle_digest,
                full_analysis_input_digest=base_run.full_analysis_input_digest,
                resolved_at="2026-08-09T00:21:00Z",
                data_ingestor_id="trusted-radar-ingestor",
                data_ingestor_private_key=self.scheduler_key(),
            )
        )
        issuance_domain = (
            promotion_module.OperationalIssuanceDomainArtifact.from_masks(
                issuance_plan,
                publication_eligible_mask=publication,
                source_coverage_mask=publication,
                permanent_exclusion_mask=torch.zeros_like(publication),
                resolved_source_coverage=resolved_coverage,
            )
        )
        reference_plan = promotion_module.RegimeReferencePlan(
            case_id=case_id,
            labeler_id="independent-weather-labeler",
            labeler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.regime_labeler_key()
                )
            ),
            source_contract_digest="7" * 64,
            labeling_valid_time=verification_time,
            adjudication_policy_digest="6" * 64,
        )
        catalog_plan = promotion_module.PhysicalEventCatalogPlan(
            holdout_case_ids=(case_id,),
            association_algorithm_digest="3" * 64,
            spatial_membership_rule_digest="4" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.regime_labeler_key()
                )
            ),
            catalog_completion_deadline="2026-08-09T03:00:00Z",
            spatial_reference_digest="7" * 64,
            motion_association_rule_digest="8" * 64,
            scheduler_id="trusted-scheduler",
            scheduler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    self.scheduler_key()
                )
            ),
            scheduler_trust_store_digest="8" * 64,
        )
        required_metric = promotion_module.RangeMetricRequirement(
            weather_regime="convective",
            range_regime="near_range",
            metric_name="log_echo_mse",
            lead_minutes=60,
            minimum_cases=1,
            minimum_physical_events=1,
            minimum_valid_area_km2=1.0,
            maximum_mean_normalized_degradation=1_000_000.0,
            maximum_harmful_fraction_upper_bound=1.0,
            metric_support_contract_digests=(metric_support.contract_digest,),
            maximum_end_to_end_mean_normalized_degradation=1_000_000.0,
        )
        required_issuance = promotion_module.RangeIssuanceRequirement(
            weather_regime="convective",
            range_regime="near_range",
            lead_minutes=60,
            minimum_cases=1,
            minimum_physical_events=1,
            minimum_operational_area_km2=1.0,
            maximum_withdrawn_fraction=1.0,
            maximum_newly_issued_fraction=1.0,
            maximum_background_fallback_increase=1.0,
            maximum_confidence_weighted_coverage_loss=1.0,
        )
        preregistered_policy = replace(
            self.policy(for_decision_rule=True),
            minimum_holdout_cases=1,
            minimum_material_cases=1,
            minimum_independent_cases=1,
            minimum_distinct_storms=1,
            minimum_distinct_days=1,
            minimum_distinct_radars=1,
            minimum_distinct_regimes=1,
            minimum_distinct_range_regimes=1,
            minimum_material_clusters=1,
            minimum_prior_echo_cases=1,
            minimum_prior_clear_cases=1,
            minimum_prior_echo_clusters=1,
            minimum_prior_clear_clusters=1,
            minimum_uncertainty_cases_per_regime=1,
            minimum_echo_cases_per_regime=1,
            minimum_clear_cases_per_regime=1,
            minimum_uncertainty_clusters_per_regime=1,
            minimum_echo_clusters_per_regime=1,
            minimum_clear_clusters_per_regime=1,
            minimum_state_calibration_cases_per_regime=1,
            minimum_state_calibration_clusters_per_regime=1,
            minimum_range_band_cases=1,
            minimum_range_band_clusters=1,
            minimum_range_component_samples=1,
            minimum_deployment_metric_cell_events=1,
            minimum_continuous_metric_cell_events=1,
            minimum_bootstrap_tail_replicates=1,
            maximum_prior_conditional_pit_residual_mean_abs=8.0,
            maximum_prior_conditional_underdispersion_fraction=1.0,
            maximum_prior_echo_intensity_nll=1_000.0,
            maximum_prior_support_brier_score=1.0,
            maximum_prior_echo_support_miss_score=1.0,
            maximum_prior_echo_object_miss_score=1.0,
            maximum_prior_clear_sky_false_echo_score=1.0,
            maximum_prior_echo_intensity_nll_increase=1_000.0,
            maximum_prior_support_brier_increase=1.0,
            maximum_prior_echo_support_miss_increase=1.0,
            maximum_prior_echo_object_miss_increase=1.0,
            maximum_prior_clear_sky_false_echo_increase=1.0,
            maximum_prior_conditional_pit_residual_increase=8.0,
            maximum_prior_conditional_underdispersion_increase=1.0,
            maximum_state_pit_residual_mean_abs=8.0,
            maximum_state_underdispersion_fraction=1.0,
            maximum_state_gaussian_nll=1_000.0,
            maximum_state_support_brier_score=1.0,
            maximum_state_echo_support_miss_score=1.0,
            maximum_state_echo_object_miss_score=1.0,
            maximum_state_false_support_score=1.0,
            maximum_state_valid_brier_score=1.0,
            maximum_state_gaussian_nll_increase=1_000.0,
            maximum_state_pit_residual_increase=8.0,
            maximum_state_underdispersion_increase=1.0,
            maximum_state_support_brier_increase=1.0,
            maximum_state_echo_support_miss_increase=1.0,
            maximum_state_echo_object_miss_increase=1.0,
            maximum_state_false_support_increase=1.0,
            maximum_state_valid_brier_increase=1.0,
            metric_scales=(
                PromotionMetricScale(
                    metric_name="log_echo_mse",
                    scale=1.0,
                    material_change=0.01,
                ),
            ),
            metric_support_contracts=(metric_support,),
            approved_metric_contract_digests=(metric_config.digest,),
            deployment_regime_classifier_digest=classifier.classifier_digest,
            deployment_regime_classifier_manifest_digest=(
                classifier_manifest.manifest_digest
            ),
            required_range_metrics=(required_metric,),
            required_range_issuance=(required_issuance,),
        )
        decision_rule = promotion_module.PromotionDecisionRule.from_policy(
            preregistered_policy
        )
        plan_case = NeuralPriorHoldoutPlanCase(
            case_id=case_id,
            storm_id="pending",
            day="2026-08-09",
            radar_id="radar-1",
            regime="pending",
            range_regime="near_range",
            input_plan_digest=input_plan.plan_digest,
            verification_plan_digest=verification_plan_digest(
                valid_times=forecast_valid_times,
                grid_contract_digest=grid.digest,
                radar_product_digest="a" * 64,
                qc_pipeline_digest="9" * 64,
            ),
            metric_contract_digest=metric_config.digest,
            uncertainty_target_plan_digest=target_plan.plan_digest,
            state_calibration_target_plan_digest=state_target_plan.plan_digest,
            range_band_contract_digest=range_contract.contract_digest,
            reference_active_range_regimes=(),
            regime_reference_plan_digest=reference_plan.plan_digest,
            operational_issuance_domain_plan_digest=issuance_plan.plan_digest,
            meteorological_sampling_unit_digest=(
                sampling_unit.sampling_unit_digest
            ),
            issue_time=issue_time,
        )
        holdout_cohort_digest = promotion_module._holdout_dataset_digest(
            (plan_case,)
        )
        trials = (
            promotion_module.PromotionExperimentTrial(
                candidate_prior_digest=candidate_runner.neural_prior_digest,
                promotion_decision_rule_digest=decision_rule.rule_digest,
                classifier_manifest_digests=(
                    classifier_manifest.manifest_digest,
                ),
            ),
        )
        reservation = promotion_module.GlobalSamplingReservationReceipt.issue(
            experiment_scope_digest=(
                promotion_module._promotion_experiment_scope_digest(
                    holdout_cohort_digest=holdout_cohort_digest,
                    parent_prior_digest=parent_runner.neural_prior_digest,
                    trials=trials,
                    winner_selection_rule_digest="f" * 64,
                )
            ),
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in raw_slots
            ),
            registry_id="semantic-global-sampling-registry",
            authority_id="semantic-sampling-authority",
            authority_private_key=semantic_sampling_registry_key,
            reserved_at="2026-06-30T00:00:00Z",
            registry_sequence_number=(
                semantic_training_registry_receipt.registry_sequence_number + 1
            ),
            previous_registry_root_digest=(
                semantic_training_registry_receipt.committed_registry_root_digest
            ),
        )
        resolved_raw_observations = tuple(
            promotion_module.ResolvedRawObservationReceipt.from_ingestor(
                slot=slot,
                raw_grid_volume=(
                    promotion_module.CanonicalRawGridVolumeArtifact.from_tensors(
                        reflectivity_dbz=input_frames[
                            grid.valid_times.index(slot.acquisition_valid_time)
                        ],
                        qc_valid_mask=input_valid[
                            grid.valid_times.index(slot.acquisition_valid_time)
                        ],
                        quality_weight=input_quality[
                            grid.valid_times.index(slot.acquisition_valid_time)
                        ],
                        observation_std_dbz=observation_std[
                            grid.valid_times.index(slot.acquisition_valid_time)
                        ],
                        radar_site_digest=slot.radar_site_digest,
                        acquisition_valid_time=slot.acquisition_valid_time,
                        canonical_scan_identity_digest=(
                            slot.scan_strategy_rule_digest
                        ),
                        radar_product_digest=input_plan.radar_product_digest,
                        grid_contract_digest=input_plan.grid_contract_digest,
                    )
                ),
                raw_ingestor_id="semantic-raw-ingestor",
                raw_ingestor_private_key=semantic_raw_key,
                received_at=(
                    datetime.fromisoformat(
                        slot.acquisition_valid_time.replace("Z", "+00:00")
                    )
                    + timedelta(seconds=30)
                ).isoformat(),
            )
            for index, slot in enumerate(raw_slots, start=1)
        )
        global_raw_resolution = (
            promotion_module.GlobalRawVolumeResolutionReceipt.issue(
                reservation=reservation,
                slot_identity_bindings=tuple(
                    (
                        item.slot_plan_digest,
                        item.raw_volume_identity.identity_digest,
                    )
                    for item in resolved_raw_observations
                ),
                authority_private_key=semantic_sampling_registry_key,
                resolved_at="2026-08-09T00:21:00Z",
            )
        )
        experiment_family = promotion_module.PromotionExperimentFamily(
            holdout_cohort_digest=holdout_cohort_digest,
            meteorological_sampling_unit_digests=(
                plan_case.meteorological_sampling_unit_digest,
            ),
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in raw_slots
            ),
            global_sampling_reservation=reservation,
            parent_prior_digest=parent_runner.neural_prior_digest,
            trials=trials,
            winner_selection_rule_digest="f" * 64,
        )
        plan = NeuralPriorHoldoutPlan(
            plan_id="semantic-product-plan",
            parent_prior_digest=parent_runner.neural_prior_digest,
            candidate_family_digests=(candidate_runner.neural_prior_digest,),
            cases=(plan_case,),
            input_plans=(input_plan,),
            raw_observation_slot_plans=raw_slots,
            meteorological_sampling_units=(sampling_unit,),
            raw_ingestor_trust_store=promotion_module.RawIngestorTrustStore(
                authorities=((
                    "semantic-raw-ingestor",
                    semantic_raw_key.public_key().public_bytes_raw().hex(),
                ),),
            ),
            analysis_processor_id="semantic-analysis-processor",
            analysis_processor_public_key_hex=(
                semantic_processor_key.public_key().public_bytes_raw().hex()
            ),
            uncertainty_target_plans=(target_plan,),
            state_calibration_target_plans=(state_target_plan,),
            range_band_contracts=(range_contract,),
            range_geometry_contracts=(range_geometry,),
            operational_issuance_domain_plans=(issuance_plan,),
            regime_reference_plans=(reference_plan,),
            regime_classifier_manifests=(classifier_manifest,),
            promotion_experiment_family=experiment_family,
            promotion_decision_rule_digest=decision_rule.rule_digest,
            reference_label_contract_digest="7" * 64,
            physical_event_catalog_plan=catalog_plan,
            scoring_algorithm_digest="9" * 64,
            scoring_runtime_digest=candidate_runner.numerical_runtime_digest,
            metric_engine_digest=(
                promotion_module.scoring_metric_engine_identity_digest()
            ),
            verification_resolver_digest="a" * 64,
            registered_at="2026-07-01T00:00:00Z",
        )
        analysis_input_derivation = (
            promotion_module.AnalysisInputDerivationArtifact.from_products(
                case_id=case_id,
                input_plan=input_plan,
                resolved_raw_observations=resolved_raw_observations,
                global_resolution_receipt=global_raw_resolution,
                run=base_run,
                resolved_source_coverage=resolved_coverage,
                background_frames_dbz=None,
                processed_at="2026-08-09T00:21:30Z",
                processor_id=plan.analysis_processor_id,
                processor_private_key=semantic_processor_key,
            )
        )
        derivation_json = json.dumps(
            analysis_input_derivation.payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        run_kwargs.update(
            analysis_input_derivation_artifact_json=derivation_json,
            analysis_input_derivation_artifact_digest=(
                analysis_input_derivation.artifact_digest
            ),
        )
        base_run = ForecastRunContract.from_inputs(
            config,
            input_frames,
            input_valid,
            None,
            **run_kwargs,
        )
        candidate_application = candidate_runner.infer(
            input_frames,
            input_run=base_run,
            role="candidate",
        )
        parent_application = parent_runner.infer(
            input_frames,
            input_run=base_run,
            role="parent",
        )
        candidate_forecast = product_forecast(
            candidate_runner,
            candidate_application,
            "candidate",
        )
        parent_forecast = product_forecast(
            parent_runner,
            parent_application,
            "parent",
        )
        reference_evidence = promotion_module.RegimeReferenceEvidence.from_plan(
            reference_plan,
            full_analysis_input_digest=base_run.full_analysis_input_digest,
            verification_bundle_digest=verification.content_digest,
            observed_regime="convective",
            observed_storm_id="semantic-storm",
            labeled_at=verification_time,
            labeler_private_key=self.regime_labeler_key(),
        )
        track = promotion_module.PhysicalEventTrackArtifact(
            timestamps=(issue_time, "2026-08-09T00:30:00Z"),
            centroid_xy_m=((500.0, 500.0), (600.0, 500.0)),
            object_mask_digests=("a" * 64, "b" * 64),
            source_radar_ids=("radar-1", "radar-1"),
            association_edge_digests=("c" * 64,),
            spatial_reference_digest="7" * 64,
        )
        event = promotion_module.PhysicalEventCatalogEvidence.from_members(
            event_id="semantic-event",
            member_case_ids=(case_id,),
            member_full_analysis_input_digests=(
                base_run.full_analysis_input_digest,
            ),
            start_time=issue_time,
            end_time="2026-08-09T00:30:00Z",
            spatial_envelope_xy_m=(0.0, 0.0, 1_000.0, 1_000.0),
            object_track_artifact=track,
            participating_radar_ids=("radar-1",),
            association_algorithm_digest="3" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="independent-weather-labeler",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        spatial_evidence = promotion_module.PhysicalEventCaseSpatialEvidence(
            case_id=case_id,
            full_analysis_input_digest=base_run.full_analysis_input_digest,
            physical_event_identity_digest=event.physical_event_identity_digest,
            observed_spatial_envelope_xy_m=(0.0, 0.0, 1_000.0, 1_000.0),
            event_spatial_envelope_xy_m=(0.0, 0.0, 1_000.0, 1_000.0),
            spatial_membership_rule_digest="4" * 64,
            source_object_evidence_digest="a" * 64,
            track_artifact_digest=track.artifact_digest,
            track_sample_index=0,
            track_sample_time=issue_time,
            track_object_mask_digest="a" * 64,
            input_available_time=issue_time,
            spatial_reference_digest="7" * 64,
        )
        catalog_result = promotion_module.PhysicalEventCatalogResult.from_plan(
            catalog_plan,
            event_evidences=(event,),
            case_spatial_membership_evidences=(spatial_evidence,),
            cataloged_at="2026-08-09T02:00:00Z",
            adjudicator_private_key=self.regime_labeler_key(),
        )
        completed_case = NeuralPriorHoldoutCase(
            case_id=case_id,
            planned_storm_id="pending",
            storm_id="semantic-storm",
            physical_event_digest=event.physical_event_identity_digest,
            meteorological_sampling_unit_digest=(
                plan_case.meteorological_sampling_unit_digest
            ),
            day="2026-08-09",
            radar_id="radar-1",
            planned_regime="pending",
            regime="convective",
            range_regime="near_range",
            dynamic_range_source_resolution=True,
            reference_active_range_regimes=("near_range",),
            range_band_contract_digest=range_contract.contract_digest,
            regime_reference_plan_digest=reference_plan.plan_digest,
            regime_reference_evidence_digest=reference_evidence.evidence_digest,
            operational_issuance_domain_plan_digest=issuance_plan.plan_digest,
            operational_issuance_domain_artifact_digest=(
                issuance_domain.artifact_digest
            ),
            input_plan_digest=input_plan.plan_digest,
            input_plan_resolution_digest=(
                candidate_forecast.run.input_plan_resolution_digest
            ),
            input_bundle_digest=base_run.input_bundle_digest,
            full_analysis_input_digest=base_run.full_analysis_input_digest,
            analysis_input_derivation_artifact_digest=(
                analysis_input_derivation.artifact_digest
            ),
            resolved_raw_volume_identity_digests=tuple(
                sorted(
                    item.raw_volume_identity.identity_digest
                    for item in resolved_raw_observations
                )
            ),
            global_raw_resolution_receipt_digest=(
                global_raw_resolution.receipt_digest
            ),
            fixed_input_context_digest=base_run.fixed_input_context_digest,
            observation_quality_weight_digest=(
                base_run.observation_quality_weight_digest
            ),
            observation_std_dbz_digest=base_run.observation_std_dbz_digest,
            verification_plan_digest=plan_case.verification_plan_digest,
            verification_bundle_digest=verification.content_digest,
            metric_contract_digest=metric_config.digest,
            uncertainty_target_plan_digest=target_plan.plan_digest,
            uncertainty_target_digest=uncertainty_target.target_digest,
            prior_probability_contract_digest=(
                self.probability_contract().contract_digest
            ),
            state_calibration_target_plan_digest=state_target_plan.plan_digest,
            state_calibration_target_digest=state_target.target_digest,
            prior_state_contract_digest=self.state_contract().contract_digest,
            issue_time=issue_time,
            candidate_forecast_digest=(
                promotion_module._forecast_result_content_digest(
                    candidate_forecast
                )
            ),
            parent_forecast_digest=(
                promotion_module._forecast_result_content_digest(parent_forecast)
            ),
            candidate_prior_application_digest=(
                candidate_application.application_digest
            ),
            parent_prior_application_digest=parent_application.application_digest,
            candidate_inference_evidence_digest=(
                candidate_application.inference_evidence.evidence_digest
            ),
            parent_inference_evidence_digest=(
                parent_application.inference_evidence.evidence_digest
            ),
        )
        scoring_input = promotion_module.HoldoutScoringInputArtifact.from_cases(
            plan,
            candidate_prior_digest=candidate_runner.neural_prior_digest,
            parent_prior_digest=parent_runner.neural_prior_digest,
            candidate_training_manifest_digest="2" * 64,
            parent_training_manifest_digest="3" * 64,
            holdout_cases=(completed_case,),
        )
        training_plan = self.training_event_catalog_plan()
        training_result = self.training_event_catalog_result()
        training_execution = (
            promotion_module._candidate_training_execution_contract_digest(
                training_dataset_digest="1" * 64,
                candidate_training_manifest_digest="2" * 64,
                model_contract_digest="2" * 64,
                feature_schema_digest="4" * 64,
                algorithm_bundle_digest="3" * 64,
                numerical_runtime_digest=candidate_runner.numerical_runtime_digest,
                training_raw_registry_receipt_digest=(
                    semantic_training_registry_receipt.receipt_digest
                ),
            )
        )
        training_start = promotion_module.TrustedProcessStartReceipt.from_plan(
            training_plan,
            catalog_result_digest=training_result.result_digest,
            process_kind="candidate_training",
            subject_digests=("1" * 64, "2" * 64),
            process_algorithm_digest="3" * 64,
            process_runtime_digest=candidate_runner.numerical_runtime_digest,
            execution_contract_digest=training_execution,
            job_id="semantic-training-job",
            launch_nonce="a" * 64,
            scheduler_sequence_number=1,
            previous_receipt_digest=None,
            started_at="2026-07-02T00:00:00Z",
            scheduler_private_key=self.scheduler_key(),
        )
        training_log = promotion_module.ProcessLogArtifact(
            process_kind="candidate_training",
            start_receipt_digest=training_start.receipt_digest,
            entries=("semantic candidate training completed",),
        )
        training_completion = (
            promotion_module.TrustedProcessCompletionReceipt.from_start(
                training_start,
                completed_at="2026-07-03T00:00:00Z",
                output_artifact_digest=candidate_runner.neural_prior_digest,
                process_log_digest=training_log.artifact_digest,
                scheduler_private_key=self.scheduler_key(),
            )
        )
        scoring_start = promotion_module.TrustedProcessStartReceipt.from_plan(
            catalog_plan,
            catalog_result_digest=catalog_result.result_digest,
            process_kind="candidate_scoring",
            subject_digests=(scoring_input.artifact_digest,),
            process_algorithm_digest=plan.scoring_algorithm_digest,
            process_runtime_digest=plan.scoring_runtime_digest,
            execution_contract_digest=plan.scoring_execution_contract_digest,
            job_id="semantic-scoring-job",
            launch_nonce="b" * 64,
            scheduler_sequence_number=2,
            previous_receipt_digest=training_start.receipt_digest,
            started_at="2026-08-09T02:30:00Z",
            scheduler_private_key=self.scheduler_key(),
        )
        manifest = NeuralPriorCandidateManifest(
            candidate_prior_digest=candidate_runner.neural_prior_digest,
            parent_prior_digest=parent_runner.neural_prior_digest,
            training_learning_approval_digests=("a" * 64,),
            training_intervention_digests=("f" * 64,),
            training_dataset_digest="1" * 64,
            candidate_training_manifest_digest="2" * 64,
            parent_training_manifest_digest="3" * 64,
            model_contract_digest="2" * 64,
            feature_schema_digest="4" * 64,
            algorithm_bundle_digest="3" * 64,
            numerical_runtime_digest=candidate_runner.numerical_runtime_digest,
            holdout_dataset_digest=plan.holdout_dataset_digest,
            holdout_plan_digest=plan.plan_digest,
            training_case_ids=("training-case",),
            training_input_bundle_digests=("0" * 64,),
            training_full_analysis_input_digests=("8" * 64,),
            training_raw_volume_identity_digests=("b" * 64,),
            training_sampling_unit_digests=("c" * 64,),
            training_raw_registry_receipt_digest=(
                semantic_training_registry_receipt.receipt_digest
            ),
            training_raw_registry_receipt_payload_json=json.dumps(
                semantic_training_registry_receipt.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
            training_physical_event_digests=(
                self.training_event_catalog().physical_event_identity_digest,
            ),
            training_physical_event_catalog_plan=training_plan,
            training_physical_event_catalog_result=training_result,
            candidate_training_started_at="2026-07-02T00:00:00Z",
            training_storm_ids=("training-storm",),
            training_days=("2026-07-01",),
            training_radars=("radar-1",),
            training_regimes=("convective",),
            training_time_windows=((
                "2026-07-01T00:00:00Z",
                "2026-07-01T01:00:00Z",
            ),),
            regime_reference_evidences=(reference_evidence,),
            physical_event_catalog_evidences=(event,),
            physical_event_catalog_result=catalog_result,
            candidate_scoring_started_at="2026-08-09T02:30:00Z",
            holdout_cases=(completed_case,),
            candidate_training_start_receipt=training_start,
            candidate_training_completion_receipt=training_completion,
            candidate_scoring_start_receipt=scoring_start,
        )
        replay_case = promotion_module.ScoringReplayCaseArtifact.from_products(
            manifest=manifest,
            plan=plan,
            case_id=case_id,
            candidate_forecast=candidate_forecast,
            parent_forecast=parent_forecast,
            verification=verification,
            metric_config=metric_config,
            candidate_prior_application=candidate_application,
            parent_prior_application=parent_application,
            candidate_prior_runner=candidate_runner,
            parent_prior_runner=parent_runner,
            input_frames_dbz=input_frames,
            input_qc_valid_mask=input_valid,
            input_quality_weight=input_quality,
            background_frames_dbz=None,
            uncertainty_target=uncertainty_target,
            state_calibration_target=state_target,
            regime_classifier=classifier,
            regime_classifier_manifest=classifier_manifest,
            range_grid_x_m=grid_x,
            range_grid_y_m=grid_y,
            operational_issuance_domain=issuance_domain,
            analysis_input_derivation=analysis_input_derivation,
            resolved_raw_observations=resolved_raw_observations,
            global_raw_resolution_receipt=global_raw_resolution,
            resolved_source_coverage=resolved_coverage,
        )
        self.assertIn(
            "effective_horizontal_range_m",
            replay_case.replay_tensors(),
        )
        evaluation = replay_case.recompute_evaluation()
        self.assertEqual(evaluation.case_id, case_id)
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with ledger._connect() as connection:
                family = plan.promotion_experiment_family
                connection.execute(
                    "INSERT INTO neural_prior_promotion_experiment_families "
                    "(family_digest,holdout_cohort_digest,payload_json,"
                    "trust_store_digest,created_at) VALUES (?,?,?,?,?)",
                    (
                        family.family_digest,
                        family.holdout_cohort_digest,
                        json.dumps(
                            family.payload | {"family_digest": family.family_digest},
                            sort_keys=True,
                        ),
                        "b" * 64,
                        "2026-07-01T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest,plan_id,plan_json,policy_digest,"
                    "trust_store_digest,registered_at,created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        plan.plan_digest,
                        plan.plan_id,
                        json.dumps(asdict(plan), sort_keys=True),
                        "a" * 64,
                        "b" * 64,
                        plan.registered_at,
                        "2026-07-01T00:00:00+00:00",
                    ),
                )
                training_receipt_json = json.dumps(
                    semantic_training_registry_receipt.payload
                    | {
                        "receipt_digest": (
                            semantic_training_registry_receipt.receipt_digest
                        )
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    "INSERT INTO training_raw_registry_entries "
                    "(receipt_digest,registry_id,registry_sequence_number,"
                    "previous_registry_root_digest,committed_registry_root_digest,"
                    "payload_json,created_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        semantic_training_registry_receipt.receipt_digest,
                        semantic_training_registry_receipt.registry_id,
                        semantic_training_registry_receipt.registry_sequence_number,
                        semantic_training_registry_receipt
                        .previous_registry_root_digest,
                        semantic_training_registry_receipt
                        .committed_registry_root_digest,
                        training_receipt_json,
                        "2026-07-01T00:00:00+00:00",
                    ),
                )
                reservation = family.global_sampling_reservation
                connection.execute(
                    "INSERT INTO global_sampling_registry_entries "
                    "(registry_id,registry_sequence_number,"
                    "previous_registry_root_digest,committed_registry_root_digest,"
                    "receipt_digest,entry_kind,family_digest,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        reservation.registry_id,
                        reservation.registry_sequence_number,
                        reservation.previous_registry_root_digest,
                        reservation.committed_registry_root_digest,
                        reservation.receipt_digest,
                        "slot_reservation",
                        family.family_digest,
                        "2026-07-01T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO neural_prior_holdout_scoring_input_artifacts "
                    "(artifact_digest, holdout_plan_digest, payload_json, "
                    "created_at) VALUES (?, ?, ?, ?)",
                    (
                        scoring_input.artifact_digest,
                        plan.plan_digest,
                        json.dumps(
                            scoring_input.payload
                            | {"artifact_digest": scoring_input.artifact_digest},
                            sort_keys=True,
                        ),
                        "2026-08-09T02:20:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO trusted_process_start_receipts_v2 "
                    "(receipt_digest, catalog_plan_digest, "
                    "catalog_result_digest, process_kind, scheduler_id, "
                    "scheduler_sequence_number, job_id, launch_nonce, "
                    "previous_receipt_digest, receipt_json, started_at, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        scoring_start.receipt_digest,
                        catalog_plan.plan_digest,
                        catalog_result.result_digest,
                        "candidate_scoring",
                        scoring_start.scheduler_id,
                        scoring_start.scheduler_sequence_number,
                        scoring_start.job_id,
                        scoring_start.launch_nonce,
                        scoring_start.previous_receipt_digest,
                        json.dumps(
                            {
                                "subject_digests": [
                                    scoring_input.artifact_digest
                                ]
                            },
                            sort_keys=True,
                        ),
                        scoring_start.started_at,
                        "2026-08-09T02:30:00+00:00",
                    ),
                )
            with patch.object(
                ledger_module,
                "datetime",
                wraps=datetime,
            ) as trusted_datetime:
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-09T00:21:30+00:00"
                )
                ledger.append_analysis_input_provenance(
                    plan,
                    case_id=case_id,
                    run=replay_case.candidate_forecast.run,
                    resolved_raw_observations=resolved_raw_observations,
                    global_resolution=global_raw_resolution,
                    derivation=analysis_input_derivation,
                    resolved_source_coverage=resolved_coverage,
                )
            replay_manifest = ledger.append_neural_prior_scoring_replay_bundle(
                scoring_input,
                (replay_case,),
                algorithm_source_manifest_digest=algorithm_bundle_digest(),
            )
            replayed = ledger.load_neural_prior_scoring_replay_bundle(
                replay_manifest.bundle_digest,
                cases=(replay_case,),
            )
            self.assertTrue(replayed.semantic_replay_verified)
            self.assertEqual(
                replayed.evaluations[0].evaluation_digest,
                evaluation.evaluation_digest,
            )
        scoring_artifact = promotion_module.HoldoutScoringArtifact.from_evaluations(
            manifest,
            plan,
            scoring_input,
            (evaluation,),
            scoring_replay_bundle_digest=replay_manifest.bundle_digest,
        )
        scoring_log = promotion_module.ProcessLogArtifact(
            process_kind="candidate_scoring",
            start_receipt_digest=scoring_start.receipt_digest,
            entries=("semantic holdout scoring completed",),
        )
        scoring_completion = (
            promotion_module.TrustedProcessCompletionReceipt.from_start(
                scoring_start,
                completed_at="2026-08-09T02:40:00Z",
                output_artifact_digest=scoring_artifact.artifact_digest,
                process_log_digest=scoring_log.artifact_digest,
                scheduler_private_key=self.scheduler_key(),
            )
        )
        policy = replace(
            preregistered_policy,
            approved_candidate_manifest_digests=(manifest.manifest_digest,),
            approved_holdout_plan_digests=(plan.plan_digest,),
            approved_physical_event_catalog_result_digest=(
                catalog_result.result_digest
            ),
        )
        self.assertEqual(
            promotion_module.PromotionDecisionRule.from_policy(policy).rule_digest,
            decision_rule.rule_digest,
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
                (evaluation,),
                scoring_input_artifact=scoring_input,
                scoring_artifact=scoring_artifact,
                scoring_process_log=scoring_log,
                scoring_completion_receipt=scoring_completion,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )
        self.assertTrue(evidence.eligible, evidence.rejection_reasons)
        evidence = self.deployment_ready(evidence)
        self.assertTrue(
            evidence.deployment_eligible,
            (
                evidence.rejection_reasons,
                evidence.regime_classifier_validated,
                evidence.certified_applicability_regime_groups,
                evidence.certified_range_geometry_contract_digests,
                evidence.sample_size_preflight_feasible,
            ),
        )
        self.assertEqual(
            evidence.scoring_artifact_digest,
            scoring_artifact.artifact_digest,
        )
        self.assertEqual(
            evidence.semantic_replay_generation_digest,
            promotion_module.SEMANTIC_SCORING_REPLAY_GENERATION_DIGEST,
        )

    @staticmethod
    def replay_case_product_kwargs(case):
        return {
            name: getattr(case, name)
            for name in (
                "manifest",
                "plan",
                "case_id",
                "candidate_forecast",
                "parent_forecast",
                "verification",
                "metric_config",
                "candidate_prior_application",
                "parent_prior_application",
                "candidate_prior_runner",
                "parent_prior_runner",
                "input_frames_dbz",
                "input_qc_valid_mask",
                "input_quality_weight",
                "background_frames_dbz",
                "uncertainty_target",
                "state_calibration_target",
                "regime_classifier",
                "regime_classifier_manifest",
                "range_grid_x_m",
                "range_grid_y_m",
                "operational_issuance_domain",
                "analysis_input_derivation",
                "resolved_raw_observations",
                "global_raw_resolution_receipt",
                "resolved_source_coverage",
            )
        }

    def test_semantic_replay_requires_factory_and_exact_product_types(self) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]
        with self.assertRaisesRegex(TypeError, "from_products"):
            promotion_module.ScoringReplayCaseArtifact()
        attacks = {
            "candidate_forecast": SimpleNamespace(
                **case.candidate_forecast.__dict__
            ),
            "candidate_prior_application": SimpleNamespace(
                **case.candidate_prior_application.__dict__
            ),
            "candidate_prior_runner": SimpleNamespace(),
            "regime_classifier": SimpleNamespace(
                classifier_digest=case.regime_classifier.classifier_digest
            ),
        }
        for name, replacement in attacks.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                TypeError,
                "exact product type",
            ):
                promotion_module.ScoringReplayCaseArtifact.from_products(
                    **(
                        self.replay_case_product_kwargs(case)
                        | {name: replacement}
                    )
                )

    def test_semantic_replay_detects_forecast_tensor_after_rehash_attempt(
        self,
    ) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]
        case.candidate_forecast.forecast_dbz[0, 0, 0] += 1.0
        object.__setattr__(
            case.candidate_forecast,
            "forecast_dbz_digest",
            promotion_module.tensor_digest(
                case.candidate_forecast.forecast_dbz
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "run identity|snapshot changed|issued state",
        ):
            case.validate_integrity()

    def test_semantic_replay_requires_time_resolved_quality_weight(self) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]

        self.assertIsInstance(
            case.candidate_forecast.run,
            ForecastRunContract,
        )
        self.assertEqual(
            case.input_quality_weight.shape,
            case.input_frames_dbz.shape,
        )
        self.assertEqual(
            promotion_module.tensor_digest(case.input_quality_weight),
            case.candidate_forecast.run.observation_quality_weight_digest,
        )
        with self.assertRaisesRegex(
            ValueError, "semantic scoring replay case is invalid"
        ):
            object.__setattr__(
                case,
                "input_quality_weight",
                case.input_quality_weight[-1],
            )
            case.validate_integrity()

        tensors = case.replay_tensors()
        tensors["input_quality_weight"] = tensors[
            "input_quality_weight"
        ][-1]
        with self.assertRaisesRegex(ValueError, "input tensor shape"):
            ledger_module._validate_scoring_replay_case_tensors(
                tensors,
                dynamic_source=False,
                background_present=False,
            )

    def test_semantic_replay_requires_exact_background_time_shape(self) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]

        with self.assertRaisesRegex(
            ValueError, "semantic scoring replay case is invalid"
        ):
            object.__setattr__(
                case,
                "background_frames_dbz",
                torch.ones((2, 2, 2)),
            )
            case.validate_integrity()

        tensors = case.replay_tensors()
        tensors["background_frames_dbz"] = torch.ones((2, 2, 2))
        with self.assertRaisesRegex(ValueError, "background tensor shape"):
            ledger_module._validate_scoring_replay_case_tensors(
                tensors,
                dynamic_source=False,
                background_present=True,
            )

    def test_semantic_replay_raw_input_must_match_forecast_run(self) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]
        object.__setattr__(
            case.candidate_forecast.run,
            "input_frames_digest",
            "0" * 64,
        )
        with self.assertRaisesRegex(
            ValueError,
            "raw inputs disagree with forecast run|forecast run|input digest mismatch",
        ):
            case.validate_integrity()

    def test_pr110_snapshot_bundle_remains_audit_loadable(self) -> None:
        case_ids = ("case-1", "case-2")
        tensor = torch.ones((1, 2, 2))
        records = tuple(
            ledger_module.ScoringReplayTensorRecord(
                case_id=case_id,
                role=role,
                archive_member=f"{case_id}__{role}",
                dtype="float32",
                shape=(1, 2, 2),
                tensor_digest=promotion_module.tensor_digest(tensor),
            )
            for case_id in case_ids
            for role in sorted(
                ledger_module.LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V1
            )
        )
        legacy = ledger_module.LegacyScoringReplayBundleManifestAuditV1(
            scoring_input_artifact_digest="1" * 64,
            ordered_case_ids=case_ids,
            ordered_evaluation_digests=("2" * 64, "3" * 64),
            algorithm_source_manifest_digest="4" * 64,
            runtime_compatibility_digest="5" * 64,
            runtime_exact_digest="6" * 64,
            tensor_records=records,
            tensor_archive_sha256="7" * 64,
            evaluation_payload_sha256="8" * 64,
        )
        decoded = ledger_module._decode_scoring_replay_bundle_manifest(
            json.dumps(
                legacy.payload | {"bundle_digest": legacy.bundle_digest},
                sort_keys=True,
                separators=(",", ":"),
            ),
            expected_digest=legacy.bundle_digest,
        )
        self.assertIsInstance(
            decoded,
            ledger_module.LegacyScoringReplayBundleManifestAuditV1,
        )

    def test_pr111_semantic_bundle_remains_audit_loadable(self) -> None:
        case_id = "case-1"
        records = tuple(
            ledger_module.ScoringReplayTensorRecord(
                case_id=case_id,
                role=role,
                archive_member=f"case_000000__{role}",
                dtype="float32",
                shape=(1, 2, 2),
                tensor_digest="1" * 64,
            )
            for role in sorted(
                ledger_module.LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6
            )
        )
        legacy = ledger_module.LegacyScoringReplayBundleManifestAuditV2(
            scoring_input_artifact_digest="2" * 64,
            ordered_case_ids=(case_id,),
            ordered_evaluation_digests=("3" * 64,),
            semantic_case_digests=("4" * 64,),
            dynamic_source_case_ids=(),
            background_case_ids=(),
            algorithm_source_manifest_digest="5" * 64,
            runtime_compatibility_digest="6" * 64,
            runtime_exact_digest="7" * 64,
            tensor_records=records,
            tensor_archive_sha256="8" * 64,
            evaluation_payload_sha256="9" * 64,
        )

        decoded = ledger_module._decode_scoring_replay_bundle_manifest(
            json.dumps(
                legacy.payload | {"bundle_digest": legacy.bundle_digest},
                sort_keys=True,
                separators=(",", ":"),
            ),
            expected_digest=legacy.bundle_digest,
        )

        self.assertIsInstance(
            decoded,
            ledger_module.LegacyScoringReplayBundleManifestAuditV2,
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
        base_run = self.analysis_input_context(index, plan)[5]
        assert base_run.full_analysis_input_digest is not None
        return promotion_module.RegimeReferenceEvidence.from_plan(
            reference_plan,
            full_analysis_input_digest=base_run.full_analysis_input_digest,
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
        training_registry_receipt = self.training_raw_registry_receipt()
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
            training_raw_volume_identity_digests=("b" * 64,),
            training_sampling_unit_digests=("c" * 64,),
            training_raw_registry_receipt_digest=(
                training_registry_receipt.receipt_digest
            ),
            training_raw_registry_receipt_payload_json=json.dumps(
                training_registry_receipt.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
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
            classifier_regime_labels=(
                "convective",
                "stratiform",
                "unknown",
            ),
            classifier_range_regime_labels=("near_range", "far_range"),
            classifier_range_probability_contract=(
                "conditionally-independent-bernoulli-range-heads-v1"
            ),
            classifier_regime_probabilities=tuple(
                0.9
                if label
                == (case.regime if classified_regime is None else classified_regime)
                else 0.05
                for label in ("convective", "stratiform", "unknown")
            ),
            classifier_range_regime_probabilities=tuple(
                0.9 if label in predicted_set else 0.1
                for label in ("near_range", "far_range")
            ),
            classifier_range_presence_probability_threshold=0.8,
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
            minimum_regime_classifier_ood_abstention_lower_bound=0.0,
            maximum_regime_classifier_brier_score_upper_bound=1.0,
            maximum_weather_multiclass_brier_score_upper_bound=1.0,
            maximum_range_multilabel_brier_score_upper_bound=1.0,
            maximum_weather_ood_brier_score_upper_bound=1.0,
            maximum_range_ood_brier_score_upper_bound=1.0,
            minimum_regime_classifier_accuracy_lower_bound=0.0,
            minimum_regime_classifier_recall_lower_bound=0.0,
            maximum_regime_classifier_false_routing_upper_bound=1.0,
            minimum_regime_classifier_clusters=1,
            minimum_range_classifier_ood_cases=0,
            minimum_range_classifier_ood_abstention_lower_bound=0.0,
            minimum_range_set_precision_lower_bound=0.0,
            minimum_range_set_recall_lower_bound=0.0,
            minimum_range_exact_set_accuracy_lower_bound=0.0,
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

    def deployment_certificate(self, evidence):
        retained_roots = getattr(self, "_operational_ledger_roots", [])
        ledger_root = tempfile.TemporaryDirectory()
        retained_roots.append(ledger_root)
        self._operational_ledger_roots = retained_roots
        operational_ledger = EpisodeLedger(ledger_root.name)
        with sqlite3.connect(operational_ledger.index_path) as connection:
            ledger_instance_digest = connection.execute(
                "SELECT ledger_instance_digest FROM "
                "deployment_certificate_chain_head WHERE singleton = 1"
            ).fetchone()[0]
        ledger_key = Ed25519PrivateKey.from_private_bytes(b"\x03" * 32)
        promotion_key = Ed25519PrivateKey.from_private_bytes(b"\x04" * 32)
        operational_key = Ed25519PrivateKey.from_private_bytes(b"\x05" * 32)
        trust = promotion_module._PromotionDeploymentAuthorityTrustStore(
            keys={
                "test-ledger": ledger_key.public_key(),
                "test-promotion": promotion_key.public_key(),
                "test-operational": operational_key.public_key(),
            },
            content_digest="7" * 64,
            roles={
                "test-ledger": frozenset({"ledger_issuance"}),
                "test-promotion": frozenset({"promotion_certificate"}),
                "test-operational": frozenset({"operational_decision"}),
            },
            not_before={
                name: "2026-01-01T00:00:00+00:00"
                for name in ("test-ledger", "test-promotion", "test-operational")
            },
            not_after={
                name: "2027-01-01T00:00:00+00:00"
                for name in ("test-ledger", "test-promotion", "test-operational")
            },
            revoked_at={
                name: None
                for name in ("test-ledger", "test-promotion", "test-operational")
            },
            ledger_instance_digests={
                "test-ledger": frozenset({ledger_instance_digest}),
                "test-promotion": frozenset(),
                "test-operational": frozenset(),
            },
            ledger_instance_index_paths={
                ledger_instance_digest: operational_ledger.index_path,
            },
        )
        ledger_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-ledger",
            ledger_key,
            fixed_signing_time="2026-08-09T00:00:30Z",
        )
        promotion_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-promotion",
            promotion_key,
            fixed_signing_time="2026-08-09T00:00:40Z",
        )
        receipt = promotion_module._issue_ledger_issuance_receipt(
            ledger_instance_digest=ledger_instance_digest,
            sequence_number=1,
            previous_certificate_digest=(
                promotion_module.PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST
            ),
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            scoring_replay_bundle_digest=evidence.scoring_replay_bundle_digest,
            scoring_replay_archive_sha256="a" * 64,
            scoring_evaluation_payload_sha256="b" * 64,
            scoring_artifact_digest=evidence.scoring_artifact_digest,
            scoring_completion_receipt_digest=(
                evidence.scoring_completion_receipt_digest
            ),
            scoring_completion_completed_at="2026-08-09T00:00:00Z",
            issued_at=ledger_signer.signing_time(),
            signer=ledger_signer,
            authority_trust_store=trust,
        )
        certificate = (
            promotion_module._issue_ledgered_promotion_deployment_certificate(
                evidence,
                issued_at=promotion_signer.signing_time(),
                ledger_issuance_receipt=receipt,
                signer=promotion_signer,
                authority_trust_store=trust,
            )
        )
        with sqlite3.connect(operational_ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_promotion_deployment_certificates_v3 "
                "(certificate_digest,ledger_instance_digest,sequence_number,"
                "promotion_evidence_digest,previous_certificate_digest,"
                "ledger_chain_head_digest,payload_json,issued_at,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    certificate.certificate_digest,
                    certificate.ledger_instance_digest,
                    certificate.sequence_number,
                    certificate.promotion_evidence_digest,
                    certificate.previous_certificate_digest,
                    certificate.ledger_chain_head_digest,
                    json.dumps(
                        certificate.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    certificate.issued_at,
                    certificate.issued_at,
                ),
            )
        self._latest_operational_ledger = operational_ledger
        self._latest_deployment_authority_trust = trust
        ledger_trust_patcher = patch.object(
            ledger_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=trust,
        )
        ledger_trust_patcher.start()
        self.addCleanup(ledger_trust_patcher.stop)
        return certificate, trust

    @staticmethod
    def operational_decision_signer():
        return promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-operational",
            Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
        )

    @staticmethod
    def operational_ledger_signer():
        return promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-ledger",
            Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
        )

    def operational_decision_client(self):
        return self._latest_operational_ledger.committed_operational_decision_client(
            ledger_signer=self.operational_ledger_signer(),
            operational_signer=self.operational_decision_signer(),
            authority_trust_store_path="/etc/advar/deployment-authorities.json",
        )

    @staticmethod
    def live_operational_input_plan(plan):
        now = datetime.now().astimezone()
        observation = now - timedelta(minutes=2)
        available = now - timedelta(minutes=1)
        return replace(
            plan,
            valid_times=(observation.isoformat(),),
            observation_valid_time=observation.isoformat(),
            input_available_time=available.isoformat(),
            decision_deadline=(now + timedelta(minutes=5)).isoformat(),
            publication_time=(now + timedelta(minutes=10)).isoformat(),
        )

    @staticmethod
    def validate_deployment_artifact(
        artifact_json,
        *,
        certificate_trust,
        policy_trust,
        **kwargs,
    ):
        with patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=certificate_trust,
        ), patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=policy_trust,
        ):
            return promotion_module.validate_neural_prior_deployment_decision_artifact(
                artifact_json,
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                deployment_policy_trust_store_path=(
                    "/etc/advar/deployment-policies.json"
                ),
                **kwargs,
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
        family = plan.promotion_experiment_family
        updated_reservation = promotion_module.GlobalSamplingReservationReceipt.issue(
            experiment_scope_digest=(
                promotion_module._promotion_experiment_scope_digest(
                    holdout_cohort_digest=family.holdout_cohort_digest,
                    parent_prior_digest=family.parent_prior_digest,
                    trials=(experiment_trial,),
                    winner_selection_rule_digest=family.winner_selection_rule_digest,
                )
            ),
            raw_observation_slot_digests=family.raw_observation_slot_digests,
            registry_id=family.global_sampling_reservation.registry_id,
            authority_id=family.global_sampling_reservation.authority_id,
            authority_private_key=Ed25519PrivateKey.from_private_bytes(
                b"\x22" * 32
            ),
            reserved_at=family.global_sampling_reservation.reserved_at,
            registry_sequence_number=(
                family.global_sampling_reservation.registry_sequence_number
            ),
            previous_registry_root_digest=(
                family.global_sampling_reservation.previous_registry_root_digest
            ),
        )
        plan = replace(
            plan,
            regime_classifier_manifests=(classifier_manifest,),
            promotion_experiment_family=replace(
                family,
                trials=(experiment_trial,),
                global_sampling_reservation=updated_reservation,
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
        evaluation_by_case = {item.case_id: item for item in evaluations}
        recompute_patch = patch.object(
            promotion_module.PriorHoldoutEvaluation,
            "from_forecasts",
            side_effect=lambda *args, **kwargs: evaluation_by_case[
                kwargs["case_id"]
            ],
        )
        recompute_patch.start()
        self.addCleanup(recompute_patch.stop)
        replay_cases = self.scoring_replay_cases(
            evaluations,
            manifest=manifest,
            plan=plan,
        )
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
                reservation = experiment_family.global_sampling_reservation
                connection.execute(
                    "INSERT INTO global_sampling_registry_entries "
                    "(registry_id,registry_sequence_number,"
                    "previous_registry_root_digest,committed_registry_root_digest,"
                    "receipt_digest,entry_kind,family_digest,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        reservation.registry_id,
                        reservation.registry_sequence_number,
                        reservation.previous_registry_root_digest,
                        reservation.committed_registry_root_digest,
                        reservation.receipt_digest,
                        "slot_reservation",
                        experiment_family.family_digest,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                training_receipt = self.training_raw_registry_receipt()
                connection.execute(
                    "INSERT INTO training_raw_registry_entries "
                    "(receipt_digest,registry_id,registry_sequence_number,"
                    "previous_registry_root_digest,committed_registry_root_digest,"
                    "payload_json,created_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        training_receipt.receipt_digest,
                        training_receipt.registry_id,
                        training_receipt.registry_sequence_number,
                        training_receipt.previous_registry_root_digest,
                        training_receipt.committed_registry_root_digest,
                        json.dumps(
                            training_receipt.payload
                            | {"receipt_digest": training_receipt.receipt_digest},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                connection.executemany(
                    "INSERT INTO promotion_sampling_unit_reservations "
                    "(sampling_unit_digest,family_digest,reserved_at) "
                    "VALUES (?,?,?)",
                    tuple(
                        (
                            digest,
                            experiment_family.family_digest,
                            "2026-08-07T00:00:00+00:00",
                        )
                        for digest in (
                            experiment_family.meteorological_sampling_unit_digests
                        )
                    ),
                )
                connection.executemany(
                    "INSERT INTO promotion_raw_observation_slot_reservations "
                    "(raw_observation_slot_digest,family_digest,global_receipt_digest,"
                    "reserved_at) VALUES (?,?,?,?)",
                    tuple(
                        (
                            digest,
                            experiment_family.family_digest,
                            experiment_family.global_sampling_reservation.receipt_digest,
                            "2026-08-07T00:00:00+00:00",
                        )
                        for digest in experiment_family.raw_observation_slot_digests
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
                for replay_case in sorted(
                    replay_cases,
                    key=lambda item: item.analysis_input_derivation.processed_at,
                ):
                    trusted_datetime.now.return_value = (
                        promotion_module._canonical_datetime(
                            replay_case.analysis_input_derivation.processed_at
                        )
                        + timedelta(seconds=1)
                    )
                    ledger.append_analysis_input_provenance(
                        plan,
                        case_id=replay_case.case_id,
                        run=replay_case.candidate_forecast.run,
                        resolved_raw_observations=(
                            replay_case.resolved_raw_observations
                        ),
                        global_resolution=(
                            replay_case.global_raw_resolution_receipt
                        ),
                        derivation=replay_case.analysis_input_derivation,
                        resolved_source_coverage=(
                            replay_case.resolved_source_coverage
                        ),
                        background_frames_dbz=replay_case.background_frames_dbz,
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
                        replay_cases,
                        algorithm_source_manifest_digest=(
                            algorithm_bundle_digest()
                        ),
                    )
                )
                replayed = ledger.load_neural_prior_scoring_replay_bundle(
                    replay_manifest.bundle_digest,
                    cases=replay_cases,
                )
                self.assertTrue(replayed.semantic_replay_verified)
                self.assertEqual(
                    tuple(
                        item.evaluation_digest for item in replayed.evaluations
                    ),
                    tuple(item.evaluation_digest for item in evaluations),
                )
                original_candidate_forecast = (
                    replay_cases[0].candidate_forecast.forecast_dbz.clone()
                )
                replay_cases[0].candidate_forecast.forecast_dbz[0, 0, 0] += 1.0
                with self.assertRaisesRegex(
                    ValueError, "forecast result disagrees with the issued forecast"
                ):
                    ledger.load_neural_prior_scoring_replay_bundle(
                        replay_manifest.bundle_digest,
                        cases=replay_cases,
                    )
                replay_cases[0].candidate_forecast.forecast_dbz.copy_(
                    original_candidate_forecast
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
                evidence = self.deployment_ready(evidence)
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T03:00:00+00:00"
                )
                ledger.append_trusted_process_completion_receipt(
                    manifest.candidate_scoring_start_receipt,
                    scoring_completion,
                    process_log_artifact=scoring_process_log,
                    scoring_artifact=scoring_artifact,
                    scoring_replay_cases=replay_cases,
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
            ), patch.object(
                ledger_module,
                "compute_neural_prior_promotion",
                return_value=evidence,
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
                    scoring_replay_cases=replay_cases,
                    policy=policy,
                    policy_trust_store_path="/etc/advar/learning-policies.json",
                )
            loaded = ledger.load_neural_prior_promotion(stored)
            self.assertEqual(loaded.promotion_evidence_digest, stored)
            self.assertEqual(loaded.contract, "neural-prior-promotion-evidence-v27")
            self.assertTrue(loaded.deployment_eligible)
            ledger_authority_key = Ed25519PrivateKey.from_private_bytes(
                b"\x03" * 32
            )
            promotion_authority_key = Ed25519PrivateKey.from_private_bytes(
                b"\x04" * 32
            )
            operational_authority_key = Ed25519PrivateKey.from_private_bytes(
                b"\x05" * 32
            )
            with sqlite3.connect(ledger.index_path) as connection:
                ledger_instance_digest = connection.execute(
                    "SELECT ledger_instance_digest FROM "
                    "deployment_certificate_chain_head WHERE singleton = 1"
                ).fetchone()[0]
            authority_trust = (
                promotion_module._PromotionDeploymentAuthorityTrustStore(
                    keys={
                        "test-ledger": ledger_authority_key.public_key(),
                        "test-promotion": promotion_authority_key.public_key(),
                        "test-operational": (
                            operational_authority_key.public_key()
                        ),
                    },
                    content_digest="7" * 64,
                    roles={
                        "test-ledger": frozenset({"ledger_issuance"}),
                        "test-promotion": frozenset(
                            {"promotion_certificate"}
                        ),
                        "test-operational": frozenset(
                            {"operational_decision"}
                        ),
                    },
                    not_before={
                        "test-ledger": "2026-01-01T00:00:00+00:00",
                        "test-promotion": "2026-01-01T00:00:00+00:00",
                        "test-operational": "2026-01-01T00:00:00+00:00",
                    },
                    not_after={
                        "test-ledger": "2027-01-01T00:00:00+00:00",
                        "test-promotion": "2027-01-01T00:00:00+00:00",
                        "test-operational": "2027-01-01T00:00:00+00:00",
                    },
                    revoked_at={
                        "test-ledger": None,
                        "test-promotion": None,
                        "test-operational": None,
                    },
                    ledger_instance_digests={
                        "test-ledger": frozenset({ledger_instance_digest}),
                        "test-promotion": frozenset(),
                        "test-operational": frozenset(),
                    },
                    ledger_instance_index_paths={
                        ledger_instance_digest: ledger.index_path,
                    },
                )
            )
            ledger_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
                "test-ledger",
                ledger_authority_key,
                fixed_signing_time="2026-08-12T02:30:00Z",
            )
            promotion_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
                "test-promotion",
                promotion_authority_key,
                fixed_signing_time="2026-08-12T02:31:00Z",
            )
            with patch.object(
                ledger_module,
                "_load_promotion_deployment_authority_trust_store",
                return_value=authority_trust,
            ):
                for invalid_times in (
                    (
                        "2026-08-12T01:59:00Z",
                        "2026-08-12T02:01:00Z",
                    ),
                    (
                        "2099-08-12T03:01:00Z",
                        "2099-08-12T03:02:00Z",
                    ),
                ):
                    with self.subTest(invalid_times=invalid_times), self.assertRaisesRegex(
                        ValueError,
                        "issuance chronology",
                    ):
                        ledger.issue_neural_prior_promotion_deployment_certificate(
                            stored,
                            scoring_replay_cases=replay_cases,
                            ledger_signer=(
                                promotion_module.Ed25519DeploymentAuthoritySigner(
                                    "test-ledger",
                                    ledger_authority_key,
                                    fixed_signing_time=invalid_times[0],
                                )
                            ),
                            deployment_signer=(
                                promotion_module.Ed25519DeploymentAuthoritySigner(
                                    "test-promotion",
                                    promotion_authority_key,
                                    fixed_signing_time=invalid_times[1],
                                )
                            ),
                            authority_trust_store_path=(
                                "/etc/advar/deployment-authorities.json"
                            ),
                        )

                def issue_certificate():
                    try:
                        return ledger.issue_neural_prior_promotion_deployment_certificate(
                            stored,
                            scoring_replay_cases=replay_cases,
                            ledger_signer=ledger_signer,
                            deployment_signer=promotion_signer,
                            authority_trust_store_path=(
                                "/etc/advar/deployment-authorities.json"
                            ),
                        )
                    except FileExistsError as error:
                        return error

                with ThreadPoolExecutor(max_workers=2) as executor:
                    issuance_results = tuple(
                        executor.map(lambda _: issue_certificate(), range(2))
                    )
                issued = tuple(
                    item
                    for item in issuance_results
                    if isinstance(
                        item,
                        promotion_module.LedgeredPromotionDeploymentCertificate,
                    )
                )
                self.assertEqual(len(issued), 1, issuance_results)
                self.assertEqual(
                    sum(isinstance(item, FileExistsError) for item in issuance_results),
                    1,
                )
                certificate = issued[0]
                reloaded_certificate = (
                    ledger.load_neural_prior_promotion_deployment_certificate(
                        certificate.certificate_digest,
                        authority_trust_store_path=(
                            "/etc/advar/deployment-authorities.json"
                        ),
                    )
                )
            self.assertEqual(
                certificate.contract,
                "ledgered-promotion-deployment-certificate-v4",
            )
            self.assertEqual(certificate.sequence_number, 1)
            self.assertEqual(
                certificate.previous_certificate_digest,
                ledger_module._PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST,
            )
            self.assertEqual(
                reloaded_certificate.certificate_digest,
                certificate.certificate_digest,
            )
            deployment_policy = DeployedNeuralPriorPolicy(
                candidate_prior_digest=evidence.candidate_prior_digest,
                parent_prior_digest=evidence.parent_prior_digest,
                promotion_evidence_digest=evidence.promotion_evidence_digest,
                promotion_deployment_certificate_digest=(
                    certificate.certificate_digest
                ),
                promotion_deployment_authority_trust_store_digest=(
                    certificate.authority_trust_store_digest
                ),
                regime_classifier_digest=(
                    evidence.deployment_regime_classifier_digest
                ),
                regime_classifier_manifest_digest=(
                    evidence.deployment_regime_classifier_manifest_digest
                ),
                range_geometry_contract_digest=(
                    evidence.certified_range_geometry_contract_digests[0]
                ),
            )
            operational_signer = (
                promotion_module.Ed25519DeploymentAuthoritySigner(
                    "test-operational",
                    operational_authority_key,
                    fixed_signing_time="2026-08-12T02:37:00Z",
                )
            )
            operational_ledger_signer = (
                promotion_module.Ed25519DeploymentAuthoritySigner(
                    "test-ledger",
                    ledger_authority_key,
                    fixed_signing_time="2026-08-12T02:36:00Z",
                )
            )
            operational_input_plan = plan.input_plans[0]
            operational_cycle_id = promotion_module.json_digest(
                {
                    "contract": "advar-operational-cycle-v1",
                    "input_plan_digest": operational_input_plan.plan_digest,
                    "full_analysis_input_digest": (
                        manifest.holdout_cases[0].full_analysis_input_digest
                    ),
                }
            )
            operational_decision = {
                "promotion_deployment_certificate": certificate.payload
                | {"certificate_digest": certificate.certificate_digest},
                "full_analysis_input_digest": (
                    manifest.holdout_cases[0].full_analysis_input_digest
                ),
                "input_plan_digest": operational_input_plan.plan_digest,
                "observation_valid_time": "2026-08-12T02:31:00Z",
                "input_available_time": "2026-08-12T02:32:00Z",
                "decision_deadline": "2026-08-12T02:40:00Z",
                "publication_time": "2026-08-12T02:45:00Z",
                "operational_cycle_id": operational_cycle_id,
                "selection": {
                    "selected_prior_digest": evidence.candidate_prior_digest,
                    "selected_role": "candidate",
                    "fallback_reason": "certified_candidate",
                },
            }
            with (
                patch.object(
                    ledger_module,
                    "_load_promotion_deployment_authority_trust_store",
                    return_value=authority_trust,
                ),
                patch.object(
                    ledger_module,
                    "datetime",
                    wraps=datetime,
                ) as decision_datetime,
            ):
                decision_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-12T02:35:00+00:00"
                )
                operational_certificate = (
                    ledger.issue_operational_deployment_decision(
                        operational_decision,
                        promotion_deployment_certificate=certificate,
                        promotion_evidence=evidence,
                        policy=deployment_policy,
                        policy_trust_store_digest="b" * 64,
                        ledger_signer=operational_ledger_signer,
                        operational_signer=operational_signer,
                        authority_trust_store_path=(
                            "/etc/advar/deployment-authorities.json"
                        ),
                    )
                )
                with self.assertRaisesRegex(ValueError, "separate keys"):
                    ledger.issue_operational_deployment_decision(
                        operational_decision,
                        promotion_deployment_certificate=certificate,
                        promotion_evidence=evidence,
                        policy=deployment_policy,
                        policy_trust_store_digest="b" * 64,
                        ledger_signer=ledger_signer,
                        operational_signer=ledger_signer,
                        authority_trust_store_path=(
                            "/etc/advar/deployment-authorities.json"
                        ),
                    )
            self.assertEqual(
                operational_certificate.contract,
                "operational-deployment-decision-certificate-v4",
            )
            self.assertEqual(operational_certificate.ledger_sequence_number, 1)

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
                promotion_deployment_certificate=Mock(),
                regime_classifier=Mock(classify=Mock(return_value=Mock())),
                range_geometry_contract=geometry,
                grid_x_m=x,
                grid_y_m=y,
                policy=Mock(),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_decision_client=Mock(),
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
                promotion_deployment_certificate=Mock(),
                regime_classifier=Mock(classify=Mock(return_value=Mock())),
                range_geometry_contract=geometry,
                grid_x_m=x,
                grid_y_m=y,
                policy=Mock(),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_decision_client=Mock(),
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
                Mock(),
                partition,
                Mock(),
                range_geometry_contract=Mock(),
                operational_grid_contract_digest="1" * 64,
                operational_frame_shape=(2, 2),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_decision_client=Mock(),
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
                    full_analysis_input_digest=(
                        untrusted.member_full_analysis_input_digests[0]
                    ),
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    untrusted_second,
                    case_id="case-2",
                    full_analysis_input_digest=(
                        untrusted_second.member_full_analysis_input_digests[0]
                    ),
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
                    full_analysis_input_digest=(
                        changed_catalog.member_full_analysis_input_digests[0]
                    ),
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    changed_second_catalog,
                    case_id="case-2",
                    full_analysis_input_digest=(
                        changed_second_catalog.member_full_analysis_input_digests[0]
                    ),
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
                    full_analysis_input_digest=(
                        first.member_full_analysis_input_digests[0]
                    ),
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    second,
                    case_id="case-2",
                    full_analysis_input_digest=(
                        second.member_full_analysis_input_digests[0]
                    ),
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
                    training_raw_registry_receipt_digest=(
                        self.training_raw_registry_receipt().receipt_digest
                    ),
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
                full_analysis_input_digest=(
                    event.member_full_analysis_input_digests[0]
                ),
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
                full_analysis_input_digest=(
                    event.member_full_analysis_input_digests[0]
                ),
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
        expanded_trials = (original_trial, *additional)
        reservation = promotion_module.GlobalSamplingReservationReceipt.issue(
            experiment_scope_digest=(
                promotion_module._promotion_experiment_scope_digest(
                    holdout_cohort_digest=(
                        plan.promotion_experiment_family.holdout_cohort_digest
                    ),
                    parent_prior_digest=(
                        plan.promotion_experiment_family.parent_prior_digest
                    ),
                    trials=expanded_trials,
                    winner_selection_rule_digest=(
                        plan.promotion_experiment_family
                        .winner_selection_rule_digest
                    ),
                )
            ),
            raw_observation_slot_digests=(
                plan.promotion_experiment_family.raw_observation_slot_digests
            ),
            registry_id=(
                plan.promotion_experiment_family
                .global_sampling_reservation.registry_id
            ),
            authority_id=(
                plan.promotion_experiment_family
                .global_sampling_reservation.authority_id
            ),
            authority_private_key=Ed25519PrivateKey.from_private_bytes(
                b"\x22" * 32
            ),
            reserved_at=(
                plan.promotion_experiment_family.global_sampling_reservation
                .reserved_at
            ),
            registry_sequence_number=(
                plan.promotion_experiment_family.global_sampling_reservation
                .registry_sequence_number
            ),
            previous_registry_root_digest=(
                plan.promotion_experiment_family.global_sampling_reservation
                .previous_registry_root_digest
            ),
        )
        family = replace(
            plan.promotion_experiment_family,
            trials=expanded_trials,
            global_sampling_reservation=reservation,
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

    def test_ledger_rejects_repackaged_processing_for_the_same_weather_sample(
        self,
    ) -> None:
        plan = self.plan()
        repackaged_input = replace(
            plan.input_plans[0],
            qc_pipeline_digest="9" * 64,
            mask_policy_digest="8" * 64,
        )
        # QC, projection, and grid changes define a new processing cohort, not
        # a new meteorological sample.  The three prospective raw slots for
        # each case therefore retain their original sampling-unit identity.
        regridded_sampling_units = plan.meteorological_sampling_units
        repackaged_cases = tuple(
            replace(
                item,
                input_plan_digest=(
                    repackaged_input.plan_digest
                    if index == 0
                    else item.input_plan_digest
                ),
                meteorological_sampling_unit_digest=(
                    regridded_sampling_units[index].sampling_unit_digest
                ),
            )
            for index, item in enumerate(plan.cases)
        )
        second_cohort_digest = promotion_module._holdout_dataset_digest(
            repackaged_cases
        )
        second_registry_key = Ed25519PrivateKey.from_private_bytes(b"\x22" * 32)
        second_reservation = promotion_module.GlobalSamplingReservationReceipt.issue(
            experiment_scope_digest=(
                promotion_module._promotion_experiment_scope_digest(
                    holdout_cohort_digest=second_cohort_digest,
                    parent_prior_digest=plan.promotion_experiment_family.parent_prior_digest,
                    trials=plan.promotion_experiment_family.trials,
                    winner_selection_rule_digest="e" * 64,
                )
            ),
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in plan.raw_observation_slot_plans
            ),
            registry_id="test-global-sampling-registry",
            authority_id="test-sampling-authority",
            authority_private_key=second_registry_key,
            reserved_at="2026-08-06T00:00:00Z",
            registry_sequence_number=(
                plan.promotion_experiment_family.global_sampling_reservation
                .registry_sequence_number
            ),
            previous_registry_root_digest=(
                plan.promotion_experiment_family.global_sampling_reservation
                .previous_registry_root_digest
            ),
        )
        second_family = replace(
            plan.promotion_experiment_family,
            holdout_cohort_digest=second_cohort_digest,
            meteorological_sampling_unit_digests=tuple(
                item.sampling_unit_digest
                for item in regridded_sampling_units
            ),
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in plan.raw_observation_slot_plans
            ),
            winner_selection_rule_digest="e" * 64,
            global_sampling_reservation=second_reservation,
        )
        second_plan = replace(
            plan,
            plan_id="holdout-plan-second-family",
            cases=repackaged_cases,
            input_plans=(repackaged_input, *plan.input_plans[1:]),
            raw_observation_slot_plans=plan.raw_observation_slot_plans,
            meteorological_sampling_units=regridded_sampling_units,
            promotion_experiment_family=second_family,
        )
        self.assertNotEqual(
            plan.promotion_experiment_family.holdout_cohort_digest,
            second_family.holdout_cohort_digest,
        )
        self.assertEqual(
            set(plan.promotion_experiment_family.meteorological_sampling_unit_digests),
            set(second_family.meteorological_sampling_unit_digests),
        )
        self.assertEqual(
            set(plan.promotion_experiment_family.raw_observation_slot_digests)
            & set(second_family.raw_observation_slot_digests),
            set(plan.promotion_experiment_family.raw_observation_slot_digests),
        )
        first_policy = promotion_module.NeuralPriorHoldoutPlanPolicy(
            approved_plan_digests=(plan.plan_digest,),
            approved_metric_contract_digests=tuple(
                sorted({item.metric_contract_digest for item in plan.cases})
            ),
            maximum_candidate_family_size=1,
            sampling_registry_id=(
                plan.promotion_experiment_family.global_sampling_reservation.registry_id
            ),
            sampling_registry_authority_id=(
                plan.promotion_experiment_family.global_sampling_reservation.authority_id
            ),
            sampling_registry_authority_public_key_hex=(
                plan.promotion_experiment_family.global_sampling_reservation.authority_public_key_hex
            ),
            approved_sampling_registry_root_digests=(
                plan.promotion_experiment_family.global_sampling_reservation.committed_registry_root_digest,
                second_family.global_sampling_reservation.committed_registry_root_digest,
            ),
            raw_ingestor_trust_store_digest=(
                plan.raw_ingestor_trust_store.content_digest
            ),
            analysis_processor_id=plan.analysis_processor_id,
            analysis_processor_public_key_hex=(
                plan.analysis_processor_public_key_hex
            ),
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
                with self.assertRaisesRegex(
                    ValueError, "sampling unit|observation slot"
                ):
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
            classifier_subset_event_counts=self.classifier_subset_counts(30),
        )
        sufficient = promotion_module.promotion_sample_size_preflight(
            self.plan(),
            policy,
            available_physical_events=small.required_physical_events,
            classifier_subset_event_counts=self.classifier_subset_counts(
                small.required_physical_events
            ),
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

    def test_sample_size_preflight_is_subset_aware(self) -> None:
        result = promotion_module.promotion_sample_size_preflight(
            self.plan(),
            self.policy(),
            available_physical_events=10_000,
            classifier_subset_event_counts=self.classifier_subset_counts(
                10_000,
                overrides={"weather_ood": 0},
            ),
        )

        self.assertFalse(result.classifier_subset_feasible)
        self.assertFalse(result.feasible)
        weather_ood = next(
            item
            for item in result.classifier_subset_event_counts
            if item[0] == "weather_ood"
        )
        self.assertEqual(weather_ood[1], 0)
        self.assertGreater(weather_ood[2], 0)

        sparse_regime = promotion_module.promotion_sample_size_preflight(
            self.plan(),
            self.policy(),
            available_physical_events=10_000,
            classifier_subset_event_counts=self.classifier_subset_counts(
                10_000,
                overrides={"known_weather:convective": 0},
            ),
        )
        self.assertFalse(sparse_regime.classifier_subset_feasible)
        convective = next(
            item
            for item in sparse_regime.classifier_subset_event_counts
            if item[0] == "known_weather:convective"
        )
        self.assertEqual(convective[1], 0)
        self.assertGreater(convective[2], 0)

    def test_brier_preflight_inverts_the_actual_upper_bound(self) -> None:
        policy = replace(
            self.policy(),
            maximum_regime_classifier_brier_score_upper_bound=0.1,
            maximum_weather_multiclass_brier_score_upper_bound=0.1,
            maximum_range_multilabel_brier_score_upper_bound=0.1,
            maximum_weather_ood_brier_score_upper_bound=0.1,
            maximum_range_ood_brier_score_upper_bound=0.1,
        )
        family_size = promotion_module._classifier_simultaneous_family_size(
            self.plan(), policy
        )
        required = promotion_module._required_bounded_mean_events(
            threshold=0.1,
            absolute_bound=1.0,
            confidence_level=policy.confidence_level,
            family_size=family_size,
        )
        at_boundary = promotion_module._bounded_event_mean_upper_bound(
            [0.0] * required,
            [f"event-{index}" for index in range(required)],
            policy,
            family_size=family_size,
            absolute_bound=1.0,
        )
        below_boundary = promotion_module._bounded_event_mean_upper_bound(
            [0.0] * (required - 1),
            [f"event-{index}" for index in range(required - 1)],
            policy,
            family_size=family_size,
            absolute_bound=1.0,
        )
        self.assertLessEqual(at_boundary, 0.1)
        self.assertGreater(below_boundary, 0.1)

        preflight = promotion_module.promotion_sample_size_preflight(
            self.plan(),
            policy,
            available_physical_events=required,
            classifier_subset_event_counts=self.classifier_subset_counts(
                required,
                policy=policy,
            ),
        )
        brier = next(
            item
            for item in preflight.classifier_subset_event_counts
            if item[0] == "brier_valid"
        )
        self.assertEqual(brier[2], required)
        weather, ranges = promotion_module._registered_classifier_strata(
            self.plan(), policy
        )
        self.assertEqual(
            preflight.classifier_family_size,
            self.plan().promotion_experiment_family.total_family_size
            * len(promotion_module._CLASSIFIER_SIMULTANEOUS_ENDPOINTS)
            * len(weather)
            * len(ranges),
        )

    def test_ece_is_diagnostic_not_a_dead_policy_gate(self) -> None:
        policy = self.policy()
        evidence = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        self.assertFalse(
            hasattr(policy, "maximum_regime_classifier_calibration_error")
        )
        self.assertGreaterEqual(
            evidence.diagnostic_case_weighted_regime_classifier_ece,
            0.0,
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
        plan, input_plan, registry = self.mosaic_issuance_context()
        nominal = torch.ones((1, 2, 2), dtype=torch.bool)
        source_index = torch.tensor([[0, 1], [-1, 1]], dtype=torch.int64)
        outage = torch.tensor([[False, True], [False, False]])
        qc_valid = torch.ones((2, 2), dtype=torch.bool)
        resolved = promotion_module.ResolvedSourceCoverageArtifact.from_observations(
            plan,
            input_plan,
            registry,
            nominal_source_coverage_mask=nominal,
            source_radar_index_map=source_index,
            outage_mask=outage,
            dynamic_qc_valid_mask=qc_valid,
            input_bundle_digest="e" * 64,
            full_analysis_input_digest="1" * 64,
            resolved_at="2026-08-09T00:01:00Z",
            data_ingestor_id="trusted-radar-ingestor",
            data_ingestor_private_key=self.scheduler_key(),
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
        plan, input_plan, registry = self.mosaic_issuance_context()
        mask = torch.ones((1, 2, 2), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "outside its issue window"):
            promotion_module.ResolvedSourceCoverageArtifact.from_observations(
                plan,
                input_plan,
                registry,
                nominal_source_coverage_mask=mask,
                source_radar_index_map=torch.zeros((2, 2), dtype=torch.int64),
                outage_mask=torch.zeros((2, 2), dtype=torch.bool),
                dynamic_qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
                input_bundle_digest="e" * 64,
                full_analysis_input_digest="1" * 64,
                resolved_at="2026-08-08T23:59:00Z",
                data_ingestor_id="trusted-radar-ingestor",
                data_ingestor_private_key=self.scheduler_key(),
            )

    def test_dynamic_source_coverage_rejects_unregistered_radar_index(self) -> None:
        plan, input_plan, registry = self.mosaic_issuance_context()
        with self.assertRaisesRegex(ValueError, "registered radar set"):
            promotion_module.ResolvedSourceCoverageArtifact.from_observations(
                plan,
                input_plan,
                registry,
                nominal_source_coverage_mask=torch.ones(
                    (1, 2, 2), dtype=torch.bool
                ),
                source_radar_index_map=torch.tensor(
                    [[0, 999], [1, -1]], dtype=torch.int64
                ),
                outage_mask=torch.zeros((2, 2), dtype=torch.bool),
                dynamic_qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
                input_bundle_digest="e" * 64,
                full_analysis_input_digest="1" * 64,
                resolved_at="2026-08-09T00:01:00Z",
                data_ingestor_id="trusted-radar-ingestor",
                data_ingestor_private_key=self.scheduler_key(),
            )

    def test_mosaic_issuance_domain_requires_dynamic_resolution(self) -> None:
        plan, _, _ = self.mosaic_issuance_context()
        mask = torch.ones((1, 2, 2), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "resolved source coverage"):
            promotion_module.OperationalIssuanceDomainArtifact.from_masks(
                plan,
                publication_eligible_mask=mask,
                source_coverage_mask=mask,
                permanent_exclusion_mask=torch.zeros_like(mask),
            )

    def test_dynamic_source_coverage_must_be_ledgered_before_deadline(self) -> None:
        plan, input_plan, registry = self.mosaic_issuance_context()
        nominal = torch.ones((1, 2, 2), dtype=torch.bool)
        resolved = promotion_module.ResolvedSourceCoverageArtifact.from_observations(
            plan,
            input_plan,
            registry,
            nominal_source_coverage_mask=nominal,
            source_radar_index_map=torch.tensor(
                [[0, 1], [0, 1]], dtype=torch.int64
            ),
            outage_mask=torch.zeros((2, 2), dtype=torch.bool),
            dynamic_qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            input_bundle_digest="e" * 64,
            full_analysis_input_digest="1" * 64,
            resolved_at="2026-08-09T00:01:00Z",
            data_ingestor_id="trusted-radar-ingestor",
            data_ingestor_private_key=self.scheduler_key(),
        )
        domain = promotion_module.OperationalIssuanceDomainArtifact.from_masks(
            plan,
            publication_eligible_mask=nominal,
            source_coverage_mask=nominal,
            permanent_exclusion_mask=torch.zeros_like(nominal),
            resolved_source_coverage=resolved,
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with patch.object(
                ledger_module, "datetime", wraps=datetime
            ) as trusted_datetime:
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-09T00:01:30+00:00"
                )
                self.assertEqual(
                    ledger.append_resolved_source_coverage_artifact(
                        plan, input_plan, resolved, domain
                    ),
                    resolved.artifact_digest,
                )
        with tempfile.TemporaryDirectory() as directory:
            late_ledger = EpisodeLedger(Path(directory))
            with patch.object(
                ledger_module, "datetime", wraps=datetime
            ) as trusted_datetime:
                trusted_datetime.now.return_value = datetime.fromisoformat(
                    "2026-08-09T00:03:00+00:00"
                )
                with self.assertRaisesRegex(ValueError, "pre-issue"):
                    late_ledger.append_resolved_source_coverage_artifact(
                        plan, input_plan, resolved, domain
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
            classifier_subset_event_counts=self.classifier_subset_counts(200),
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
        deployment_certificate, deployment_certificate_trust = (
            self.deployment_certificate(evidence)
        )
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=(
                deployment_certificate.certificate_digest
            ),
            promotion_deployment_authority_trust_store_digest=(
                deployment_certificate_trust.content_digest
            ),
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
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                deployment_certificate,
                classified,
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
            )

            class ForwardingRecorder:
                def issue_operational_deployment_decision(self, *args, **kwargs):
                    raise AssertionError("an uncommitted recorder was called")

            with self.assertRaisesRegex(TypeError, "EpisodeLedger-bound"):
                promotion_module._select_deployed_prior(
                    candidate,
                    parent,
                    evidence,
                    deployment_certificate,
                    classified,
                    partition,
                    deployment_policy,
                    range_geometry_contract=geometry,
                    operational_grid_contract_digest=(
                        partition.grid_contract_digest
                    ),
                    operational_frame_shape=tuple(partition.masks[0].shape),
                    policy_trust_store_path=(
                        "/etc/advar/deployment-policies.json"
                    ),
                    deployment_certificate_trust_store_path=(
                        "/etc/advar/deployment-authorities.json"
                    ),
                    operational_input_plan=self.live_operational_input_plan(
                        self.plan().input_plans[0]
                    ),
                    operational_decision_client=ForwardingRecorder(),
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
        deployment_certificate, deployment_certificate_trust = (
            self.deployment_certificate(evidence)
        )
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=(
                deployment_certificate.certificate_digest
            ),
            promotion_deployment_authority_trust_store_digest=(
                deployment_certificate_trust.content_digest
            ),
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
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                deployment_certificate,
                classified,
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
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
            self.validate_deployment_artifact(
                json.dumps(
                    incomplete_artifact,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                certificate_trust=deployment_certificate_trust,
                policy_trust=trust,
            )
        with self.assertRaisesRegex(ValueError, "current forecast run"):
            self.validate_deployment_artifact(
                selection.deployment_decision_artifact_json,
                certificate_trust=deployment_certificate_trust,
                policy_trust=trust,
                expected_operational_grid_contract_digest="3" * 64,
                expected_operational_frame_shape=tuple(partition.masks[0].shape),
            )
        deployment_artifact["operational_grid_contract_digest"] = "3" * 64
        changed_artifact_json = json.dumps(
            deployment_artifact,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.assertRaisesRegex(ValueError, "decision certificate"):
            self.validate_deployment_artifact(
                changed_artifact_json,
                certificate_trust=deployment_certificate_trust,
                policy_trust=trust,
            )
        changed_shape_artifact = json.loads(
            selection.deployment_decision_artifact_json
        )
        changed_shape_artifact["operational_frame_shape"] = [4, 4]
        with self.assertRaisesRegex(ValueError, "decision certificate"):
            self.validate_deployment_artifact(
                json.dumps(
                    changed_shape_artifact,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                certificate_trust=deployment_certificate_trust,
                policy_trust=trust,
            )

        unapproved = replace(deployment_policy, minimum_regime_confidence=0.01)
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ), self.assertRaisesRegex(ValueError, "unapproved"):
            promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                deployment_certificate,
                classified,
                partition,
                unapproved,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
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
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ):
            _, changed_selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                deployment_certificate,
                classified,
                partition,
                unapproved,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
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
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ), self.assertRaisesRegex(ValueError, "policy digest mismatch"):
            promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                deployment_certificate,
                classified,
                partition,
                tampered,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
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
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ):
            external_selected, external_selection = (
                promotion_module._select_deployed_prior(
                    candidate,
                    parent,
                    evidence,
                    deployment_certificate,
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
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
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
        deployment_certificate, deployment_certificate_trust = (
            self.deployment_certificate(evidence)
        )
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=(
                deployment_certificate.certificate_digest
            ),
            promotion_deployment_authority_trust_store_digest=(
                deployment_certificate_trust.content_digest
            ),
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
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                deployment_certificate,
                classifier.classify(frames, input_run=run),
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
            )
        self.assertIs(selected, parent)
        self.assertEqual(selection.fallback_reason, "uncertified_range_geometry")

    def test_uncertified_geometry_parent_fallback_round_trips_forecast(
        self,
    ) -> None:
        replay_case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]
        frames = replay_case.input_frames_dbz
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
        state_contract = replay_case.parent_prior_runner.state_contract
        identity = promotion_module.OperationalDataIdentity(
            radar_class="single-site",
            qc_pipeline_digest=state_contract.state_qc_pipeline_digest,
            observation_error_model_digest="2" * 64,
            background_model_digest="3" * 64,
            radar_product_digest=state_contract.state_product_digest,
            background_cycle_rule_digest="8" * 64,
            mask_policy_digest=state_contract.state_mask_policy_digest,
            radar_site_digest="a" * 64,
            radar_site_location_digest="b" * 64,
            radar_source_contract_digest="c" * 64,
        )
        nowcast_config = NowcastConfig(
            minimum_publish_verified_support=0.01,
            minimum_publish_confidence=0.01,
            minimum_publish_observation_verified_support=0.01,
            maximum_publish_background_fraction=1.0,
            maximum_motion_speed_mps=100.0,
            minimum_phase_correlation_psr=0.0,
            pair_echo_dilation_m=1_000.0,
            phase_correlation_sidelobe_radius_m=1_000.0,
            maximum_pair_velocity_disagreement_mps=10.0,
            maximum_pair_growth_disagreement=0.0953,
            maximum_local_growth_log_error_per_step=0.4055,
            p1_motion_saturation_safe_margin_mps=2.0,
            p1_growth_saturation_safe_margin_per_step=0.04879,
            p1_posterior_saturation_sigma_multiplier=2.0,
            p1_saturation_uncertainty_multiplier=4.0,
            minimum_pair_psr_advantage=3.0,
            minimum_pair_confidence_ratio=1.5,
            long_pair_confidence_penalty=0.5,
            minimum_growth_overlap_support=1.0,
            minimum_growth_overlap_area_km2=1.0,
        )
        calibration_id = "range-fallback-calibration-v1"
        analysis_config = AnalysisConfig(
            execution_mode="operational",
            operational_calibration_id=calibration_id,
            motion_increment_scale_mps=2.0,
            causal_support_uncertainty_m=1_000.0,
            amplitude_displacement_tolerance_m=1_000.0,
            maximum_latest_detected_error_std=10.0,
            minimum_local_verification_precision=0.01,
            maximum_local_analysis_verification_error_dbz=70.0,
            maximum_unresolved_amplitude_fraction=1.0,
            minimum_amplitude_total_quality_weight=0.001,
            minimum_amplitude_effective_pixel_count=1.0,
            amplitude_information_policy="operational_fallback",
            minimum_integrated_echo_ratio_for_confidence=0.01,
            maximum_integrated_echo_ratio_for_confidence=100.0,
            minimum_soft_echo_area_ratio_for_confidence=0.01,
            maximum_soft_echo_area_ratio_for_confidence=100.0,
            maximum_established_excess_growth_fraction_for_confidence=1.0,
            minimum_object_count_ratio_for_confidence=0.01,
            amplitude_confidence_policy="operational_fallback",
            observation_common_bias_std_dbz=0.0,
            observation_common_bias_scope="per_frame",
            observation_common_bias_tile_size_px=0,
        )
        calibration = OperationalCalibrationManifest(
            calibration_id=calibration_id,
            profile_kind="p1",
            expected_runtime_profile_digest=operational_runtime_profile_digest(
                nowcast_config,
                grid,
                analysis_config=asdict(analysis_config),
            ),
            expected_algorithm_bundle_digest=algorithm_bundle_digest(),
            calibration_dataset_digest="5" * 64,
            validation_dataset_digest="6" * 64,
            data_identity=identity,
            training_period=(
                "2025-01-01T00:00:00Z",
                "2025-07-01T00:00:00Z",
            ),
            validation_period=(
                "2025-07-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
            validation_case_count=20,
            validation_regimes=(
                CalibrationRegime("convective", 10),
                CalibrationRegime("stratiform", 10),
            ),
            validation_metrics=(
                CalibrationMetric(
                    name="csi_35",
                    definition_digest="7" * 64,
                    direction="maximize",
                    acceptance_threshold=0.4,
                    value=0.5,
                ),
            ),
        )
        operational_input_plan = promotion_module.NeuralPriorInputPlan(
            valid_times=grid.valid_times,
            grid_contract_digest=grid.digest,
            radar_product_digest=identity.radar_product_digest or "",
            qc_pipeline_digest=identity.qc_pipeline_digest,
            background_cycle_rule_digest=(
                identity.background_cycle_rule_digest or ""
            ),
            mask_policy_digest=identity.mask_policy_digest or "",
            observation_valid_time=grid.valid_times[-1],
            input_available_time="2026-08-09T00:20:00Z",
            decision_deadline="2026-08-09T00:22:00Z",
            publication_time="2026-08-09T00:25:00Z",
        )
        baseline, _ = variational_nowcast(
            frames,
            nowcast_config=nowcast_config,
            analysis_config=analysis_config,
            grid_time_contract=grid,
            operational_calibration_manifest=calibration,
            operational_calibration_approval_digest=calibration.digest,
            operational_data_identity=identity,
            input_plan_json=operational_input_plan.json,
            input_plan_digest=operational_input_plan.plan_digest,
        )
        input_run = baseline.run
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier(
                (12.0, 0.0, -12.0),
                (12.0, 0.0),
            ).eval(),
            example_frames=frames,
            regime_labels=("convective", "stratiform", "unknown"),
            range_regime_labels=("near_range", "far_range"),
            classifier_algorithm_digest="5" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.deployment_ready(
            self.compute_with_policy(
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
        )
        evidence = replace(
            evidence,
            candidate_prior_digest=(
                replay_case.candidate_prior_runner.neural_prior_digest
            ),
            parent_prior_digest=(
                replay_case.parent_prior_runner.neural_prior_digest
            ),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        grid_x_m = torch.tensor([[0.0, 1_000.0], [0.0, 1_000.0]])
        grid_y_m = torch.tensor([[0.0, 0.0], [1_000.0, 1_000.0]])
        uncertified_geometry = promotion_module.RangeGeometryContract(
            radar_site_digest=identity.radar_site_digest or "",
            radar_site_location_digest=identity.radar_site_location_digest or "",
            grid_contract_digest=grid.digest,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range", "far_range"),
            radial_distance_edges_m=(0.0, 30_000.0, 100_000.0),
            horizontal_range_rule_digest="d" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(grid_x_m),
            grid_y_m_digest=promotion_module.tensor_digest(grid_y_m),
        )
        deployment_certificate, deployment_certificate_trust = (
            self.deployment_certificate(evidence)
        )
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=evidence.candidate_prior_digest,
            parent_prior_digest=evidence.parent_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=(
                deployment_certificate.certificate_digest
            ),
            promotion_deployment_authority_trust_store_digest=(
                deployment_certificate_trust.content_digest
            ),
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=uncertified_geometry.contract_digest,
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
        operational_client = (
            self._latest_operational_ledger.committed_operational_decision_client(
                ledger_signer=(
                    promotion_module.Ed25519DeploymentAuthoritySigner(
                        "test-ledger",
                        Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
                        fixed_signing_time="2026-08-09T00:21:20Z",
                    )
                ),
                operational_signer=(
                    promotion_module.Ed25519DeploymentAuthoritySigner(
                        "test-operational",
                        Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
                        fixed_signing_time="2026-08-09T00:21:30Z",
                    )
                ),
                authority_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
            )
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ), patch.object(
            ledger_module, "datetime", wraps=datetime
        ) as operational_clock:
            operational_clock.now.return_value = datetime.fromisoformat(
                "2026-08-09T00:21:10+00:00"
            )
            application = promotion_module.infer_deployed_neural_prior(
                frames,
                input_run=input_run,
                candidate_runner=replay_case.candidate_prior_runner,
                parent_runner=replay_case.parent_prior_runner,
                promotion_evidence=evidence,
                promotion_deployment_certificate=deployment_certificate,
                regime_classifier=classifier,
                range_geometry_contract=uncertified_geometry,
                grid_x_m=grid_x_m,
                grid_y_m=grid_y_m,
                policy=deployment_policy,
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_decision_client=operational_client,
            )
        assert application.deployment_selection is not None
        self.assertEqual(application.role, "parent")
        self.assertEqual(
            application.deployment_selection.fallback_reason,
            "uncertified_range_geometry",
        )
        forecast, _ = variational_nowcast(
            frames,
            nowcast_config=nowcast_config,
            analysis_config=analysis_config,
            grid_time_contract=grid,
            operational_calibration_manifest=calibration,
            operational_calibration_approval_digest=calibration.digest,
            operational_data_identity=identity,
            input_plan_json=operational_input_plan.json,
            input_plan_digest=operational_input_plan.plan_digest,
            neural_prior=application,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent-fallback.npz"
            save_forecast_run(forecast, path)
            with patch.object(
                promotion_module,
                "_load_promotion_deployment_authority_trust_store",
                return_value=deployment_certificate_trust,
            ), patch.object(
                promotion_module,
                "_load_learning_policy_trust_store",
                return_value=trust,
            ):
                loaded = load_forecast_run(
                    path,
                    deployment_certificate_trust_store_path=(
                        "/etc/advar/deployment-authorities.json"
                    ),
                    deployment_policy_trust_store_path=(
                        "/etc/advar/deployment-policies.json"
                    ),
                )
        self.assertEqual(loaded.run.prior_role, "parent")
        self.assertEqual(
            loaded.run.prior_deployment_fallback_reason,
            "uncertified_range_geometry",
        )
        self.assertEqual(
            loaded.run.prior_deployment_lineage_contract,
            "neural-prior-deployment-lineage-v12",
        )

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
        deployment_certificate, deployment_certificate_trust = (
            self.deployment_certificate(evidence)
        )
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=(
                deployment_certificate.certificate_digest
            ),
            promotion_deployment_authority_trust_store_digest=(
                deployment_certificate_trust.content_digest
            ),
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
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                deployment_certificate,
                classifier.classify(frames, input_run=run),
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
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
        deployment_certificate, deployment_certificate_trust = (
            self.deployment_certificate(evidence)
        )
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=(
                deployment_certificate.certificate_digest
            ),
            promotion_deployment_authority_trust_store_digest=(
                deployment_certificate_trust.content_digest
            ),
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
        ), patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=deployment_certificate_trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                deployment_certificate,
                classifier.classify(frames, input_run=run),
                partition,
                deployment_policy,
                range_geometry_contract=geometry,
                operational_grid_contract_digest=partition.grid_contract_digest,
                operational_frame_shape=tuple(partition.masks[0].shape),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_input_plan=self.live_operational_input_plan(
                    self.plan().input_plans[0]
                ),
                operational_decision_client=self.operational_decision_client(),
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
            digest=planned_input.grid_contract_digest,
            spatial_grid_digest=planned_input.grid_contract_digest,
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
            grid_time_contract_digest=planned_input.grid_contract_digest,
            grid_time_contract=grid,
            input_bundle_digest=case.input_bundle_digest,
            input_frames_digest=promotion_module.tensor_digest(
                torch.zeros((3, 2, 2))
            ),
            observation_masks_digest=promotion_module.tensor_digest(
                torch.ones((3, 2, 2), dtype=torch.bool)
            ),
            background_frames_digest=None,
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
            state_valid_mask=torch.tensor(
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
                feature_source_identity_digests=("a" * 64,) * 3,
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
            state_valid_mask=torch.ones((2, 2), dtype=torch.bool),
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
                feature_source_identity_digests=("a" * 64,) * 3,
                feature_exclusion_contract_digest="5" * 64,
                feature_exclusion_mask_digest=promotion_module.tensor_digest(
                    torch.ones((3, 2, 2), dtype=torch.bool)
                ),
            ),
        )
        def typed_run(
            *,
            role: str,
            prior_digest: str,
            application_digest: str,
            evidence_digest: str,
            training_manifest_digest: str,
        ) -> ForecastRunContract:
            frames = torch.zeros((3, 2, 2))
            masks = torch.ones_like(frames, dtype=torch.bool)
            run = ForecastRunContract.from_inputs(
                NowcastConfig(),
                frames,
                masks,
                None,
                observation_quality_weight=torch.ones_like(frames),
                neural_prior_digest=prior_digest,
                prior_application_digest=application_digest,
                prior_model_contract_digest=manifest.model_contract_digest,
                prior_feature_schema_digest=manifest.feature_schema_digest,
                prior_training_manifest_digest=training_manifest_digest,
                prior_inference_evidence_digest=evidence_digest,
                prior_inference_algorithm_digest="8" * 64,
                prior_numerical_runtime_digest="4" * 64,
                prior_dependency="radar_dependent",
                prior_role=role,
            )
            # The surrounding test exercises preregistered holdout lineage.
            # Retain its canonical fixture digests on a real product run object.
            for name, value in common.items():
                object.__setattr__(run, name, value)
            return run

        candidate_run = typed_run(
            role="candidate",
            prior_digest=manifest.candidate_prior_digest,
            application_digest=candidate_app.application_digest,
            evidence_digest=candidate_app.inference_evidence.evidence_digest,
            training_manifest_digest=(
                manifest.candidate_training_manifest_digest
            ),
        )
        parent_run = typed_run(
            role="parent",
            prior_digest=manifest.parent_prior_digest,
            application_digest=parent_app.application_digest,
            evidence_digest=parent_app.inference_evidence.evidence_digest,
            training_manifest_digest=manifest.parent_training_manifest_digest,
        )
        candidate = SimpleNamespace(
            run=candidate_run,
            state=SimpleNamespace(
                echo_linear=torch.ones((2, 2)),
                displacement_yx=torch.zeros(2),
                log_growth_per_step=torch.tensor(0.0),
            ),
            validate_issuance=Mock(),
            forecast_dbz=torch.zeros((6, 2, 2)),
            forecast_run_digest="a" * 64,
            forecast_dbz_digest="b" * 64,
            valid_mask_digest="c" * 64,
            state_metadata_digest="d" * 64,
            valid_mask=torch.ones((6, 2, 2), dtype=torch.bool),
            background_fallback_mask=torch.zeros(
                (6, 2, 2), dtype=torch.bool
            ),
            forecast_confidence=torch.ones((6, 2, 2)),
        )
        parent = SimpleNamespace(
            run=parent_run,
            state=SimpleNamespace(
                echo_linear=torch.ones((2, 2)),
                displacement_yx=torch.zeros(2),
                log_growth_per_step=torch.tensor(0.0),
            ),
            validate_issuance=Mock(),
            forecast_dbz=torch.zeros((6, 2, 2)),
            forecast_run_digest="e" * 64,
            forecast_dbz_digest="f" * 64,
            valid_mask_digest="1" * 64,
            state_metadata_digest="2" * 64,
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
        verification = SimpleNamespace(
            valid_times=resolved.valid_times,
            frames_dbz=resolved.frames_dbz,
            valid_mask=resolved.valid_mask,
            content_digest=resolved.content_digest,
        )
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
            regime_labels=("convective", "stratiform", "unknown"),
            range_regime_labels=("near_range", "far_range"),
            range_probability_contract=(
                "conditionally-independent-bernoulli-range-heads-v1"
            ),
            range_presence_probability_threshold=0.8,
            regime_probabilities=(1.0, 0.0, 0.0),
            range_regime_probabilities=(1.0, 0.0),
            regime_entropy=0.0,
            is_ood=False,
            weather_top1_top2_gap=1.0,
            minimum_range_presence_margin=0.5,
        )
        regime_classifier = SimpleNamespace(
            classifier_digest=regime_evidence.classifier_digest,
            numerical_runtime_digest=regime_evidence.numerical_runtime_digest,
            classify=Mock(return_value=regime_evidence),
            classification_logits=Mock(
                return_value=(
                    torch.tensor((2.0, 0.0, -1.0)),
                    torch.tensor((1.0, -1.0)),
                )
            ),
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

        object.__setattr__(
            parent_run,
            "observation_quality_weight_digest",
            "0" * 64,
        )
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
        object.__setattr__(
            parent_run,
            "observation_quality_weight_digest",
            case.observation_quality_weight_digest,
        )
        object.__setattr__(
            parent_run,
            "observation_std_dbz_digest",
            "0" * 64,
        )
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
        object.__setattr__(
            parent_run,
            "observation_std_dbz_digest",
            case.observation_std_dbz_digest,
        )

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
        input_plan = promotion_module.NeuralPriorInputPlan(
            valid_times=grid.valid_times,
            grid_contract_digest=grid.digest,
            radar_product_digest="6" * 64,
            qc_pipeline_digest=identity.qc_pipeline_digest,
            background_cycle_rule_digest="7" * 64,
            mask_policy_digest="8" * 64,
            observation_valid_time=grid.valid_times[-1],
            input_available_time="2026-08-09T00:20:00Z",
            decision_deadline="2026-08-09T00:22:00Z",
            publication_time="2026-08-09T00:25:00Z",
        )
        run = replace(
            run,
            input_plan_json=input_plan.json,
            input_plan_digest=input_plan.plan_digest,
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
                promotion_deployment_certificate=Mock(),
                regime_classifier=Mock(),
                range_geometry_contract=geometry,
                grid_x_m=coordinates,
                grid_y_m=coordinates,
                policy=Mock(),
                policy_trust_store_path="/etc/advar/deployment-policies.json",
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                operational_decision_client=Mock(),
            )

    def test_cpu_only_scoring_generation_has_a_stable_backend_contract(self) -> None:
        self.assertEqual(
            promotion_module.SEMANTIC_SCORING_REPLAY_CONTRACT,
            "neural-prior-scoring-replay-bundle-v7",
        )
        self.assertEqual(
            promotion_module.SEMANTIC_SCORING_REPLAY_METHOD,
            "builtin-semantic-scoring-recomputation-v7",
        )
        self.assertEqual(
            promotion_module.SEMANTIC_SCORING_REPLAY_GENERATION_PAYLOAD,
            {
                "contract": "neural-prior-semantic-scoring-generation-v5",
                "replay_contract": "neural-prior-scoring-replay-bundle-v7",
                "replay_method": "builtin-semantic-scoring-recomputation-v7",
                "case_contract": "neural-prior-semantic-scoring-case-v6",
                "product_type_policy": "exact-shipped-product-types-v1",
                "forecast_integrity": "forecast-result-raw-content-validation-v1",
                "prior_integrity": "runner-reproduced-prior-application-v1",
                "classifier_integrity": "exported-classifier-reexecution-v1",
                "snapshot_policy": "single-frozen-tensor-snapshot-v1",
                "backend_policy": "single-device-cpu-only-scoring-v1",
            },
        )
        evidence = self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        with self.assertRaisesRegex(ValueError, "CPU-scoring"):
            replace(
                evidence,
                scoring_backend_certification_policy_digest="a" * 64,
                scoring_backend_certification_evidence_digest="b" * 64,
            )

    def test_sampling_unit_is_independent_of_processing_contracts(self) -> None:
        input_plan = self.plan().input_plans[0]
        repackaged = replace(
            input_plan,
            qc_pipeline_digest="9" * 64,
            mask_policy_digest="8" * 64,
            radar_product_digest="7" * 64,
        )

        def sampling_digest(_plan):
            return promotion_module.meteorological_sampling_unit_digest(
                raw_observation_slot_digests=(
                    self.plan().raw_observation_slot_plans[0].slot_digest,
                ),
                canonical_geodetic_footprint_digest="6" * 64,
            )

        self.assertNotEqual(input_plan.plan_digest, repackaged.plan_digest)
        self.assertEqual(sampling_digest(input_plan), sampling_digest(repackaged))

    def test_overlapping_windows_share_raw_observation_identity(self) -> None:
        raw_slots = tuple(
            promotion_module.RawObservationSlotPlan(
                radar_site_digest="a" * 64,
                acquisition_valid_time=f"2026-08-09T00:{index:02d}:00Z",
                scan_strategy_rule_digest="5" * 64,
                source_selection_rule_digest="6" * 64,
                canonical_geodetic_footprint_digest="6" * 64,
            )
            for index in (0, 10, 20, 30)
        )
        first = promotion_module.MeteorologicalSamplingUnit(
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in raw_slots[:3]
            ),
            canonical_geodetic_footprint_digest="6" * 64,
        )
        second = promotion_module.MeteorologicalSamplingUnit(
            raw_observation_slot_digests=tuple(
                item.slot_digest for item in raw_slots[1:]
            ),
            canonical_geodetic_footprint_digest="6" * 64,
        )
        self.assertNotEqual(first.sampling_unit_digest, second.sampling_unit_digest)
        self.assertEqual(
            set(first.raw_observation_slot_digests)
            & set(second.raw_observation_slot_digests),
            {raw_slots[1].slot_digest, raw_slots[2].slot_digest},
        )

    def test_prospective_plan_precedes_every_raw_slot(self) -> None:
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "raw observation slots"):
            replace(
                plan,
                registered_at=(
                    plan.raw_observation_slot_plans[0].acquisition_valid_time
                ),
            )

        slot = plan.raw_observation_slot_plans[0]
        raw_grid_volume = promotion_module.CanonicalRawGridVolumeArtifact.from_tensors(
            reflectivity_dbz=torch.zeros((2, 2)),
            qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            quality_weight=torch.ones((2, 2)),
            observation_std_dbz=torch.full((2, 2), 2.0),
            radar_site_digest=slot.radar_site_digest,
            acquisition_valid_time=slot.acquisition_valid_time,
            canonical_scan_identity_digest=slot.scan_strategy_rule_digest,
            radar_product_digest=plan.input_plans[0].radar_product_digest,
            grid_contract_digest=plan.input_plans[0].grid_contract_digest,
        )
        with self.assertRaisesRegex(ValueError, "resolved raw observation"):
            promotion_module.ResolvedRawObservationReceipt.from_ingestor(
                slot=slot,
                raw_grid_volume=raw_grid_volume,
                raw_ingestor_id="too-early-ingestor",
                raw_ingestor_private_key=Ed25519PrivateKey.from_private_bytes(
                    b"\x24" * 32
                ),
                received_at=(
                    promotion_module._canonical_datetime(
                        slot.acquisition_valid_time
                    )
                    - timedelta(microseconds=1)
                ).isoformat(),
            )

    def test_raw_identity_survives_ingestor_key_rotation(self) -> None:
        plan = self.plan()
        slot = plan.raw_observation_slot_plans[0]
        raw_grid_volume = promotion_module.CanonicalRawGridVolumeArtifact.from_tensors(
            reflectivity_dbz=torch.zeros((2, 2)),
            qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            quality_weight=torch.ones((2, 2)),
            observation_std_dbz=torch.full((2, 2), 2.0),
            radar_site_digest=slot.radar_site_digest,
            acquisition_valid_time=slot.acquisition_valid_time,
            canonical_scan_identity_digest=slot.scan_strategy_rule_digest,
            radar_product_digest=plan.input_plans[0].radar_product_digest,
            grid_contract_digest=plan.input_plans[0].grid_contract_digest,
        )
        received_at = (
            promotion_module._canonical_datetime(slot.acquisition_valid_time)
            + timedelta(seconds=1)
        ).isoformat()
        first = promotion_module.ResolvedRawObservationReceipt.from_ingestor(
            slot=slot,
            raw_grid_volume=raw_grid_volume,
            raw_ingestor_id="ingestor-before-rotation",
            raw_ingestor_private_key=Ed25519PrivateKey.from_private_bytes(
                b"\x24" * 32
            ),
            received_at=received_at,
        )
        second = promotion_module.ResolvedRawObservationReceipt.from_ingestor(
            slot=slot,
            raw_grid_volume=raw_grid_volume,
            raw_ingestor_id="ingestor-after-rotation",
            raw_ingestor_private_key=Ed25519PrivateKey.from_private_bytes(
                b"\x25" * 32
            ),
            received_at=received_at,
        )
        self.assertEqual(
            first.raw_volume_identity.identity_digest,
            second.raw_volume_identity.identity_digest,
        )
        self.assertNotEqual(
            first.raw_volume_attestation.attestation_digest,
            second.raw_volume_attestation.attestation_digest,
        )
        self.assertEqual(
            first.raw_volume_identity.radar_product_digest,
            raw_grid_volume.radar_product_digest,
        )
        self.assertEqual(
            first.raw_volume_identity.grid_contract_digest,
            raw_grid_volume.grid_contract_digest,
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            values = (
                first.raw_volume_identity.identity_digest,
                "a" * 64,
                "b" * 64,
                received_at,
            )
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO promotion_raw_volume_identity_reservations "
                    "(raw_volume_identity_digest,family_digest,"
                    "global_resolution_receipt_digest,reserved_at) "
                    "VALUES (?,?,?,?)",
                    values,
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT INTO promotion_raw_volume_identity_reservations "
                        "(raw_volume_identity_digest,family_digest,"
                        "global_resolution_receipt_digest,reserved_at) "
                        "VALUES (?,?,?,?)",
                        (
                            second.raw_volume_identity.identity_digest,
                            "c" * 64,
                            "d" * 64,
                            received_at,
                        ),
                    )

    def test_background_lineage_uses_model_times_not_observation_times(
        self,
    ) -> None:
        observation_times = (
            "2026-08-09T00:00:00+00:00",
            "2026-08-09T00:10:00+00:00",
            "2026-08-09T00:20:00+00:00",
        )
        background_times = (
            "2026-08-08T23:50:00+00:00",
            "2026-08-09T00:00:00+00:00",
            "2026-08-09T00:10:00+00:00",
        )
        grid = RadarGridTimeContract(
            valid_times=observation_times,
            background_valid_times=background_times,
            dx_m=1_000.0,
            dy_m=1_000.0,
            projection="EPSG:3857",
            grid_hash="1" * 64,
        )
        input_plan = promotion_module.NeuralPriorInputPlan(
            valid_times=observation_times,
            grid_contract_digest=grid.digest,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            background_cycle_rule_digest="b" * 64,
            mask_policy_digest="3" * 64,
            observation_valid_time=observation_times[-1],
            input_available_time="2026-08-09T00:20:01+00:00",
            decision_deadline="2026-08-09T00:21:00+00:00",
            publication_time="2026-08-09T00:22:00+00:00",
        )
        data_identity = promotion_module.OperationalDataIdentity(
            radar_class="test-radar",
            qc_pipeline_digest=input_plan.qc_pipeline_digest,
            observation_error_model_digest="c" * 64,
            background_model_digest="d" * 64,
            radar_product_digest=input_plan.radar_product_digest,
            background_cycle_rule_digest=(
                input_plan.background_cycle_rule_digest
            ),
            mask_policy_digest=input_plan.mask_policy_digest,
        )
        frames = torch.zeros((3, 2, 2))
        masks = torch.ones_like(frames, dtype=torch.bool)
        quality = torch.ones_like(frames)
        standard_deviation = torch.full_like(frames, 2.0)
        background = torch.stack(
            tuple(torch.full((2, 2), float(index)) for index in range(3))
        )
        calibration = OperationalCalibrationManifest(
            calibration_id="background-lineage-test",
            profile_kind="p0",
            expected_runtime_profile_digest=operational_runtime_profile_digest(
                NowcastConfig(),
                grid,
            ),
            expected_algorithm_bundle_digest=algorithm_bundle_digest(),
            calibration_dataset_digest="5" * 64,
            validation_dataset_digest="6" * 64,
            data_identity=data_identity,
            training_period=(
                "2025-01-01T00:00:00Z",
                "2025-07-01T00:00:00Z",
            ),
            validation_period=(
                "2025-07-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
            validation_case_count=20,
            validation_regimes=(CalibrationRegime("convective", 20),),
            validation_metrics=(
                CalibrationMetric(
                    name="csi_35",
                    definition_digest="7" * 64,
                    direction="maximize",
                    acceptance_threshold=0.4,
                    value=0.5,
                ),
            ),
        )
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            masks,
            background,
            background_age_minutes=10.0,
            observation_quality_weight=quality,
            observation_std_dbz=standard_deviation,
            grid_time_contract=grid,
            operational_calibration_manifest_json=calibration.json,
            operational_calibration_manifest_digest=calibration.digest,
            operational_calibration_approval_digest=calibration.digest,
            operational_data_identity_json=data_identity.json,
            operational_data_identity_digest=data_identity.digest,
            input_plan_json=input_plan.json,
            input_plan_digest=input_plan.plan_digest,
        )
        retained_times, source_digest, frame_digests = (
            promotion_module._background_input_identity_digests(
                input_plan=input_plan,
                run=run,
                background_frames_dbz=background,
            )
        )
        self.assertEqual(retained_times, tuple(grid.background_valid_times or ()))
        self.assertNotEqual(retained_times, input_plan.valid_times)
        self.assertIsNotNone(source_digest)
        self.assertEqual(len(frame_digests), 3)

    def test_training_raw_and_sampling_units_cannot_overlap_holdout(self) -> None:
        manifest = self.manifest()
        holdout_raw = manifest.holdout_cases[0].resolved_raw_volume_identity_digests[0]
        holdout_sampling = (
            manifest.holdout_cases[0].meteorological_sampling_unit_digest
        )
        raw_overlap_receipt = promotion_module.TrainingRawRegistryReceipt.issue(
            raw_volume_identity_digests=(holdout_raw,),
            sampling_unit_digests=manifest.training_sampling_unit_digests,
            registry_id="test-global-sampling-registry",
            authority_id="test-sampling-authority",
            authority_private_key=Ed25519PrivateKey.from_private_bytes(
                b"\x22" * 32
            ),
            committed_at="2026-06-30T00:00:00Z",
        )

        def manifest_with_training_receipt(
            receipt,
            *,
            raw_digests,
            sampling_digests,
        ):
            training_start = promotion_module.TrustedProcessStartReceipt.from_plan(
                manifest.training_physical_event_catalog_plan,
                catalog_result_digest=(
                    manifest.training_physical_event_catalog_result.result_digest
                ),
                process_kind="candidate_training",
                subject_digests=(
                    manifest.training_dataset_digest,
                    manifest.candidate_training_manifest_digest,
                ),
                process_algorithm_digest=manifest.algorithm_bundle_digest,
                process_runtime_digest=manifest.numerical_runtime_digest,
                execution_contract_digest=(
                    promotion_module._candidate_training_execution_contract_digest(
                        training_dataset_digest=manifest.training_dataset_digest,
                        candidate_training_manifest_digest=(
                            manifest.candidate_training_manifest_digest
                        ),
                        model_contract_digest=manifest.model_contract_digest,
                        feature_schema_digest=manifest.feature_schema_digest,
                        algorithm_bundle_digest=manifest.algorithm_bundle_digest,
                        numerical_runtime_digest=manifest.numerical_runtime_digest,
                        training_raw_registry_receipt_digest=receipt.receipt_digest,
                    )
                ),
                job_id=manifest.candidate_training_start_receipt.job_id,
                launch_nonce=manifest.candidate_training_start_receipt.launch_nonce,
                scheduler_sequence_number=(
                    manifest.candidate_training_start_receipt
                    .scheduler_sequence_number
                ),
                previous_receipt_digest=(
                    manifest.candidate_training_start_receipt
                    .previous_receipt_digest
                ),
                started_at=manifest.candidate_training_started_at,
                scheduler_private_key=self.scheduler_key(),
            )
            training_completion = (
                promotion_module.TrustedProcessCompletionReceipt.from_start(
                    training_start,
                    completed_at=(
                        manifest.candidate_training_completion_receipt.completed_at
                    ),
                    output_artifact_digest=manifest.candidate_prior_digest,
                    process_log_digest=(
                        manifest.candidate_training_completion_receipt
                        .process_log_digest
                    ),
                    scheduler_private_key=self.scheduler_key(),
                )
            )
            return replace(
                manifest,
                training_raw_volume_identity_digests=raw_digests,
                training_sampling_unit_digests=sampling_digests,
                training_raw_registry_receipt_digest=receipt.receipt_digest,
                training_raw_registry_receipt_payload_json=json.dumps(
                    receipt.payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                candidate_training_start_receipt=training_start,
                candidate_training_completion_receipt=training_completion,
            )

        with self.assertRaisesRegex(ValueError, "raw volumes"):
            manifest_with_training_receipt(
                raw_overlap_receipt,
                raw_digests=(holdout_raw,),
                sampling_digests=manifest.training_sampling_unit_digests,
            )
        sampling_overlap_receipt = (
            promotion_module.TrainingRawRegistryReceipt.issue(
                raw_volume_identity_digests=("b" * 64,),
                sampling_unit_digests=(holdout_sampling,),
                registry_id="test-global-sampling-registry",
                authority_id="test-sampling-authority",
                authority_private_key=Ed25519PrivateKey.from_private_bytes(
                    b"\x22" * 32
                ),
                committed_at="2026-06-30T00:00:00Z",
            )
        )
        with self.assertRaisesRegex(ValueError, "sampling units"):
            manifest_with_training_receipt(
                sampling_overlap_receipt,
                raw_digests=manifest.training_raw_volume_identity_digests,
                sampling_digests=(holdout_sampling,),
            )

        classifier = self.plan().regime_classifier_manifests[0]
        with self.assertRaisesRegex(ValueError, "raw volumes"):
            promotion_module._validate_classifier_holdout_independence(
                replace(
                    classifier,
                    training_raw_volume_identity_digests=(holdout_raw,),
                    training_raw_registry_receipt_digest=(
                        raw_overlap_receipt.receipt_digest
                    ),
                    training_raw_registry_receipt_payload_json=json.dumps(
                        raw_overlap_receipt.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
                manifest.holdout_cases,
            )
        with self.assertRaisesRegex(ValueError, "sampling units"):
            promotion_module._validate_classifier_holdout_independence(
                replace(
                    classifier,
                    training_sampling_unit_digests=(holdout_sampling,),
                    training_raw_registry_receipt_digest=(
                        sampling_overlap_receipt.receipt_digest
                    ),
                    training_raw_registry_receipt_payload_json=json.dumps(
                        sampling_overlap_receipt.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
                manifest.holdout_cases,
            )

    def test_training_raw_receipt_cannot_be_rewritten_after_reservation(self) -> None:
        plan = self.plan()
        classifier = plan.regime_classifier_manifests[0]
        rewritten = promotion_module.TrainingRawRegistryReceipt.issue(
            raw_volume_identity_digests=("d" * 64,),
            sampling_unit_digests=("e" * 64,),
            registry_id="test-global-sampling-registry",
            authority_id="test-sampling-authority",
            authority_private_key=Ed25519PrivateKey.from_private_bytes(
                b"\x22" * 32
            ),
            committed_at="2026-06-30T00:00:00Z",
        )
        rewritten_classifier = replace(
            classifier,
            training_raw_volume_identity_digests=("d" * 64,),
            training_sampling_unit_digests=("e" * 64,),
            training_raw_registry_receipt_digest=rewritten.receipt_digest,
            training_raw_registry_receipt_payload_json=json.dumps(
                rewritten.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "training overlaps"):
            replace(
                plan,
                regime_classifier_manifests=(rewritten_classifier,),
            )

    def test_scored_input_cannot_be_rebound_to_different_raw_volumes(
        self,
    ) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]
        plan = case.plan
        raw_ingestor_key = Ed25519PrivateKey.from_private_bytes(b"\x21" * 32)
        replacement_receipts = tuple(
            promotion_module.ResolvedRawObservationReceipt.from_ingestor(
                slot=slot,
                raw_grid_volume=(
                    promotion_module.CanonicalRawGridVolumeArtifact.from_tensors(
                        reflectivity_dbz=torch.full((2, 2), float(index)),
                        qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
                        quality_weight=torch.ones((2, 2)),
                        observation_std_dbz=torch.full((2, 2), 2.0),
                        radar_site_digest=slot.radar_site_digest,
                        acquisition_valid_time=slot.acquisition_valid_time,
                        canonical_scan_identity_digest=(
                            slot.scan_strategy_rule_digest
                        ),
                        radar_product_digest=next(
                            item.radar_product_digest
                            for item in plan.input_plans
                            if slot.acquisition_valid_time in item.valid_times
                        ),
                        grid_contract_digest=next(
                            item.grid_contract_digest
                            for item in plan.input_plans
                            if slot.acquisition_valid_time in item.valid_times
                        ),
                    )
                ),
                raw_ingestor_id="test-raw-ingestor",
                raw_ingestor_private_key=raw_ingestor_key,
                received_at=(
                    promotion_module._canonical_datetime(
                        slot.acquisition_valid_time
                    )
                    + timedelta(seconds=30)
                ).isoformat(),
            )
            for index, slot in enumerate(
                plan.raw_observation_slot_plans,
                start=10,
            )
        )
        required_slots = {
            receipt.slot_plan_digest
            for receipt in case.resolved_raw_observations
        }
        replacement_case_receipts = tuple(
            receipt
            for receipt in replacement_receipts
            if receipt.slot_plan_digest in required_slots
        )
        replacement_resolution = (
            promotion_module.GlobalRawVolumeResolutionReceipt.issue(
                reservation=(
                    plan.promotion_experiment_family.global_sampling_reservation
                ),
                slot_identity_bindings=tuple(
                    (
                        receipt.slot_plan_digest,
                        receipt.raw_volume_identity.identity_digest,
                    )
                    for receipt in replacement_case_receipts
                ),
                authority_private_key=Ed25519PrivateKey.from_private_bytes(
                    b"\x22" * 32
                ),
                resolved_at=case.global_raw_resolution_receipt.resolved_at,
            )
        )
        input_plan = next(
            item
            for item in plan.input_plans
            if item.plan_digest == plan.case(case.case_id).input_plan_digest
        )
        original = case.analysis_input_derivation
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            promotion_module.AnalysisInputDerivationArtifact.from_products(
                case_id=case.case_id,
                input_plan=input_plan,
                resolved_raw_observations=replacement_case_receipts,
                global_resolution_receipt=replacement_resolution,
                run=case.candidate_forecast.run,
                resolved_source_coverage=case.resolved_source_coverage,
                background_frames_dbz=case.background_frames_dbz,
                processed_at=original.processed_at,
                processor_id=original.processor_id,
                processor_private_key=Ed25519PrivateKey.from_private_bytes(
                    b"\x23" * 32
                ),
            )

    def test_raw_resolution_after_case_deadline_is_rejected(self) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]
        input_plan = next(
            item
            for item in case.plan.input_plans
            if item.plan_digest == case.analysis_input_derivation.input_plan_digest
        )
        late_time = (
            promotion_module._canonical_datetime(input_plan.decision_deadline)
            + timedelta(microseconds=1)
        )
        late_resolution = (
            promotion_module.GlobalRawVolumeResolutionReceipt.issue(
                reservation=(
                    case.plan.promotion_experiment_family.global_sampling_reservation
                ),
                slot_identity_bindings=tuple(
                    (
                        item.slot_plan_digest,
                        item.raw_volume_identity.identity_digest,
                    )
                    for item in case.resolved_raw_observations
                ),
                authority_private_key=Ed25519PrivateKey.from_private_bytes(
                    b"\x22" * 32
                ),
                resolved_at=late_time.isoformat(),
            )
        )
        with self.assertRaisesRegex(ValueError, "decision deadline"):
            promotion_module.AnalysisInputDerivationArtifact.from_products(
                case_id=case.case_id,
                input_plan=input_plan,
                resolved_raw_observations=case.resolved_raw_observations,
                global_resolution_receipt=late_resolution,
                run=case.candidate_forecast.run,
                resolved_source_coverage=case.resolved_source_coverage,
                background_frames_dbz=case.background_frames_dbz,
                processed_at=(late_time + timedelta(microseconds=1)).isoformat(),
                processor_id=case.analysis_input_derivation.processor_id,
                processor_private_key=Ed25519PrivateKey.from_private_bytes(
                    b"\x23" * 32
                ),
            )

    def test_backdated_raw_payload_cannot_be_committed_after_deadline(self) -> None:
        case = self.scoring_replay_cases((self.evaluation(1, -0.2),))[0]
        plan = case.plan
        input_plan = next(
            item
            for item in plan.input_plans
            if item.plan_digest == case.analysis_input_derivation.input_plan_digest
        )
        family = plan.promotion_experiment_family
        reservation = family.global_sampling_reservation
        after_deadline = (
            promotion_module._canonical_datetime(input_plan.decision_deadline)
            + timedelta(microseconds=1)
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_promotion_experiment_families "
                    "(family_digest,holdout_cohort_digest,payload_json,"
                    "trust_store_digest,created_at) VALUES (?,?,?,?,?)",
                    (
                        family.family_digest,
                        family.holdout_cohort_digest,
                        json.dumps(
                            family.payload | {"family_digest": family.family_digest},
                            sort_keys=True,
                        ),
                        "7" * 64,
                        plan.registered_at,
                    ),
                )
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest,plan_id,plan_json,policy_digest,"
                    "trust_store_digest,registered_at,created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        plan.plan_digest,
                        plan.plan_id,
                        json.dumps(asdict(plan), sort_keys=True),
                        "6" * 64,
                        "7" * 64,
                        plan.registered_at,
                        plan.registered_at,
                    ),
                )
                connection.execute(
                    "INSERT INTO global_sampling_registry_entries "
                    "(registry_id,registry_sequence_number,"
                    "previous_registry_root_digest,committed_registry_root_digest,"
                    "receipt_digest,entry_kind,family_digest,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        reservation.registry_id,
                        reservation.registry_sequence_number,
                        reservation.previous_registry_root_digest,
                        reservation.committed_registry_root_digest,
                        reservation.receipt_digest,
                        "slot_reservation",
                        family.family_digest,
                        plan.registered_at,
                    ),
                )
            with patch.object(
                ledger_module,
                "datetime",
                wraps=datetime,
            ) as trusted_datetime:
                trusted_datetime.now.return_value = after_deadline
                with self.assertRaisesRegex(ValueError, "durable deadline"):
                    ledger.append_analysis_input_provenance(
                        plan,
                        case_id=case.case_id,
                        run=case.candidate_forecast.run,
                        resolved_raw_observations=case.resolved_raw_observations,
                        global_resolution=case.global_raw_resolution_receipt,
                        derivation=case.analysis_input_derivation,
                        resolved_source_coverage=case.resolved_source_coverage,
                        background_frames_dbz=case.background_frames_dbz,
                    )

    def test_operational_client_cannot_be_constructed_or_faked(self) -> None:
        with self.assertRaisesRegex(TypeError, "created by EpisodeLedger"):
            promotion_module._EpisodeLedgerOperationalDecisionClient()

        class ForwardingRecorder:
            def issue_operational_deployment_decision(self, *args, **kwargs):
                raise AssertionError("an unbound recorder must never be called")

        self.assertIsNot(
            type(ForwardingRecorder()),
            promotion_module._EpisodeLedgerOperationalDecisionClient,
        )
        forged = object.__new__(
            promotion_module._EpisodeLedgerOperationalDecisionClient
        )
        object.__setattr__(forged, "_issuer", Mock())
        object.__setattr__(forged, "_ledger", Mock())
        object.__setattr__(
            forged,
            "_authority_trust_store_path",
            "/etc/advar/deployment-authorities.json",
        )
        self.assertFalse(
            hasattr(
                promotion_module,
                "_bind_episode_ledger_operational_decision_client",
            )
        )
        with self.assertRaisesRegex(TypeError, "not bound to an EpisodeLedger"):
            forged._validate_committed_decision(
                Mock(),
                {},
            )
        uninitialized_ledger = object.__new__(EpisodeLedger)
        with self.assertRaises(AttributeError):
            uninitialized_ledger._connect = Mock()  # type: ignore[method-assign]
        object.__setattr__(forged, "_ledger", uninitialized_ledger)
        with self.assertRaisesRegex(TypeError, "not normally initialized"):
            forged._validate_committed_decision(
                Mock(),
                {},
            )
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as attacker_directory,
        ):
            approved_ledger = EpisodeLedger(directory)
            attacker_ledger = EpisodeLedger(attacker_directory)
            with sqlite3.connect(approved_ledger.index_path) as connection:
                approved_instance_digest = connection.execute(
                    "SELECT ledger_instance_digest FROM "
                    "deployment_certificate_chain_head WHERE singleton = 1"
                ).fetchone()[0]
            trust = promotion_module._PromotionDeploymentAuthorityTrustStore(
                keys={},
                content_digest="7" * 64,
                roles={},
                not_before={},
                not_after={},
                revoked_at={},
                ledger_instance_digests={},
                ledger_instance_index_paths={
                    approved_instance_digest: approved_ledger.index_path,
                },
            )
            object.__setattr__(forged, "_ledger", approved_ledger)
            forged_certificate = Mock()
            forged_certificate.certificate_digest = "a" * 64
            forged_certificate.ledger_instance_digest = approved_instance_digest
            forged_certificate.payload = {}
            with patch.object(
                promotion_module,
                "_load_promotion_deployment_authority_trust_store",
                return_value=trust,
            ):
                with self.assertRaisesRegex(ValueError, "committed publication"):
                    forged._validate_committed_decision(
                        forged_certificate,
                        {},
                    )

                injected_ledger = object.__new__(EpisodeLedger)
                for attribute in (
                    "root",
                    "episodes_dir",
                    "interventions_dir",
                    "scoring_replays_dir",
                    "analysis_input_provenance_dir",
                    "index_path",
                ):
                    object.__setattr__(
                        injected_ledger,
                        attribute,
                        object.__getattribute__(attacker_ledger, attribute),
                    )
                object.__setattr__(
                    injected_ledger,
                    "_initialization_state",
                    ledger_module._EpisodeLedgerInitializationState(
                        token=ledger_module._EPISODE_LEDGER_INITIALIZATION_TOKEN,
                        root=attacker_ledger.root,
                        index_path=attacker_ledger.index_path,
                    ),
                )
                object.__setattr__(forged, "_ledger", injected_ledger)
                with self.assertRaisesRegex(ValueError, "not root-approved"):
                    forged._validate_committed_decision(
                        forged_certificate,
                        {},
                    )

                class SpoofedPath:
                    def __fspath__(self):
                        return str(attacker_ledger.index_path)

                    def __eq__(self, other):
                        return True

                    def __ne__(self, other):
                        return False

                    def __truediv__(self, other):
                        return self

                    def expanduser(self):
                        return self

                    def resolve(self):
                        return self

                    def is_file(self):
                        return True

                spoofed_path = SpoofedPath()
                pathlike_ledger = object.__new__(EpisodeLedger)
                for attribute in (
                    "root",
                    "episodes_dir",
                    "interventions_dir",
                    "scoring_replays_dir",
                    "analysis_input_provenance_dir",
                    "index_path",
                ):
                    object.__setattr__(
                        pathlike_ledger,
                        attribute,
                        spoofed_path,
                    )
                object.__setattr__(
                    pathlike_ledger,
                    "_initialization_state",
                    ledger_module._EpisodeLedgerInitializationState(
                        token=ledger_module._EPISODE_LEDGER_INITIALIZATION_TOKEN,
                        root=spoofed_path,
                        index_path=spoofed_path,
                    ),
                )
                object.__setattr__(forged, "_ledger", pathlike_ledger)
                with self.assertRaisesRegex(TypeError, "initialization state"):
                    forged._validate_committed_decision(
                        forged_certificate,
                        {},
                    )

    def test_event_weighted_estimand_is_invariant_to_case_replication(self) -> None:
        first = "1" * 64
        second = "2" * 64
        baseline = promotion_module._event_weighted_mean(
            (0.2, 0.8),
            (first, second),
        )
        replicated = promotion_module._event_weighted_mean(
            (0.2, 0.2, 0.2, 0.8),
            (first, first, first, second),
        )
        self.assertAlmostEqual(baseline, 0.5)
        self.assertAlmostEqual(replicated, baseline)

    def test_classifier_safety_gates_use_event_level_finite_sample_bounds(
        self,
    ) -> None:
        evidence = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        self.assertEqual(
            evidence.primary_estimand_contract,
            "equal_weight_physical_event_v1",
        )
        self.assertLessEqual(
            evidence.regime_classifier_brier_score,
            evidence.regime_classifier_brier_score_upper_bound,
        )
        self.assertLessEqual(
            evidence.range_exact_set_accuracy_lower_bound,
            evidence.range_exact_set_accuracy,
        )
        self.assertLessEqual(
            evidence.regime_classifier_ood_abstention_fraction_lower_bound,
            evidence.regime_classifier_ood_abstention_fraction,
        )
        self.assertLessEqual(
            evidence.range_classifier_ood_abstention_fraction_lower_bound,
            evidence.range_classifier_ood_abstention_fraction,
        )
        for score, upper_bound in (
            (
                evidence.weather_multiclass_brier_score,
                evidence.weather_multiclass_brier_score_upper_bound,
            ),
            (
                evidence.range_multilabel_brier_score,
                evidence.range_multilabel_brier_score_upper_bound,
            ),
            (
                evidence.weather_ood_brier_score,
                evidence.weather_ood_brier_score_upper_bound,
            ),
            (
                evidence.range_ood_brier_score,
                evidence.range_ood_brier_score_upper_bound,
            ),
        ):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertLessEqual(score, upper_bound)

    def test_mosaic_range_uses_the_resolved_source_radar(self) -> None:
        grid_x = torch.tensor([[0.0, 9_000.0], [1_000.0, 10_000.0]])
        grid_y = torch.zeros_like(grid_x)
        source = torch.tensor([[0, 1], [-1, 1]], dtype=torch.int64)
        expected_range = torch.tensor(
            [[0.0, 1_000.0], [0.0, 0.0]], dtype=torch.float64
        )
        source_registry = promotion_module.SourceRadarRegistry(
            radar_site_digests=("4" * 64, "5" * 64),
            source_selection_policy_digest="2" * 64,
        )
        location_registry = promotion_module.RadarSiteLocationRegistry(
            projection_digest="3" * 64,
            radar_site_digests=("4" * 64, "5" * 64),
            radar_site_location_digests=("6" * 64, "7" * 64),
            radar_projected_xy_m=((0.0, 0.0), (10_000.0, 0.0)),
        )
        contract = promotion_module.MosaicRangeGeometryContract.from_registry(
            source_registry,
            location_registry,
            grid_contract_digest="3" * 64,
            projection_digest="3" * 64,
            range_regime_labels=("near_range", "far_range"),
            radial_distance_edges_m=(0.0, 500.0, 2_000.0),
            horizontal_range_rule_digest="8" * 64,
            grid_x_m_digest=promotion_module.tensor_digest(grid_x),
            grid_y_m_digest=promotion_module.tensor_digest(grid_y),
        )
        partition, effective_range = (
            promotion_module.resolve_mosaic_range_geometry(
                contract,
                grid_x_m=grid_x,
                grid_y_m=grid_y,
                source_radar_index_map=source,
            )
        )
        torch.testing.assert_close(effective_range, expected_range)
        self.assertTrue(bool(partition.masks[0][0, 0]))
        self.assertTrue(bool(partition.masks[1][0, 1]))
        self.assertFalse(bool(partition.valid_range_domain_mask[1, 0]))
        self.assertFalse(any(bool(mask[1, 0]) for mask in partition.masks))

        handed_off = source.clone()
        handed_off[1, 0] = 0
        handed_off_partition, handed_off_range = (
            promotion_module.resolve_mosaic_range_geometry(
                contract,
                grid_x_m=grid_x,
                grid_y_m=grid_y,
                source_radar_index_map=handed_off,
            )
        )
        self.assertTrue(bool(handed_off_partition.valid_range_domain_mask[1, 0]))
        self.assertEqual(float(handed_off_range[1, 0]), 1_000.0)
        self.assertNotEqual(
            partition.evidence_digest,
            handed_off_partition.evidence_digest,
        )

        without_far_band = promotion_module.restrict_range_partition_domain(
            partition,
            valid_range_domain_mask=(
                partition.valid_range_domain_mask & ~partition.masks[1]
            ),
        )
        self.assertNotIn("far_range", without_far_band.active_range_regimes)

        invalid = source.clone()
        invalid[0, 0] = 999
        with self.assertRaisesRegex(ValueError, "mosaic range geometry"):
            promotion_module.resolve_mosaic_range_geometry(
                contract,
                grid_x_m=grid_x,
                grid_y_m=grid_y,
                source_radar_index_map=invalid,
            )

        changing = torch.stack((source, source.clone()))
        with self.assertRaisesRegex(ValueError, "inputs disagree"):
            promotion_module.resolve_mosaic_range_geometry(
                contract,
                grid_x_m=grid_x,
                grid_y_m=grid_y,
                source_radar_index_map=changing,
            )

        with self.assertRaisesRegex(ValueError, "registries disagree"):
            replace(
                contract,
                radar_projected_xy_m=tuple(reversed(contract.radar_projected_xy_m)),
            )
        with self.assertRaisesRegex(ValueError, "registries disagree"):
            replace(contract, source_radar_registry_digest="f" * 64)
        with self.assertRaisesRegex(ValueError, "order disagrees"):
            promotion_module.MosaicRangeGeometryContract.from_registry(
                source_registry,
                replace(
                    location_registry,
                    radar_site_digests=tuple(
                        reversed(location_registry.radar_site_digests)
                    ),
                    radar_site_location_digests=tuple(
                        reversed(
                            location_registry.radar_site_location_digests
                        )
                    ),
                    radar_projected_xy_m=tuple(
                        reversed(location_registry.radar_projected_xy_m)
                    ),
                ),
                grid_contract_digest="3" * 64,
                projection_digest="3" * 64,
                range_regime_labels=("near_range", "far_range"),
                radial_distance_edges_m=(0.0, 500.0, 2_000.0),
                horizontal_range_rule_digest="8" * 64,
                grid_x_m_digest=promotion_module.tensor_digest(grid_x),
                grid_y_m_digest=promotion_module.tensor_digest(grid_y),
            )

    def test_certificate_authority_roles_cannot_share_one_key(self) -> None:
        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        key = Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)
        trust = promotion_module._PromotionDeploymentAuthorityTrustStore(
            keys={"combined": key.public_key()},
            content_digest="9" * 64,
            roles={
                "combined": frozenset(
                    {"ledger_issuance", "promotion_certificate"}
                )
            },
            not_before={"combined": "2026-01-01T00:00:00+00:00"},
            not_after={"combined": "2027-01-01T00:00:00+00:00"},
            revoked_at={"combined": None},
            ledger_instance_digests={"combined": frozenset({"6" * 64})},
        )
        signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "combined",
            key,
            fixed_signing_time="2026-08-09T00:01:00Z",
        )
        receipt = promotion_module._issue_ledger_issuance_receipt(
            ledger_instance_digest="6" * 64,
            sequence_number=1,
            previous_certificate_digest=(
                promotion_module.PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST
            ),
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            scoring_replay_bundle_digest=evidence.scoring_replay_bundle_digest,
            scoring_replay_archive_sha256="a" * 64,
            scoring_evaluation_payload_sha256="b" * 64,
            scoring_artifact_digest=evidence.scoring_artifact_digest,
            scoring_completion_receipt_digest=(
                evidence.scoring_completion_receipt_digest
            ),
            scoring_completion_completed_at="2026-08-09T00:00:00Z",
            issued_at=signer.signing_time(),
            signer=signer,
            authority_trust_store=trust,
        )
        with self.assertRaisesRegex(ValueError, "separate authorities"):
            promotion_module._issue_ledgered_promotion_deployment_certificate(
                evidence,
                issued_at=signer.signing_time(),
                ledger_issuance_receipt=receipt,
                signer=signer,
                authority_trust_store=trust,
            )

    def test_operational_decision_after_deadline_is_rejected(self) -> None:
        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        certificate, trust = self.deployment_certificate(evidence)
        policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=evidence.candidate_prior_digest,
            parent_prior_digest=evidence.parent_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=certificate.certificate_digest,
            promotion_deployment_authority_trust_store_digest=(
                certificate.authority_trust_store_digest
            ),
            regime_classifier_digest=evidence.deployment_regime_classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=(
                evidence.certified_range_geometry_contract_digests[0]
            ),
        )
        decision = {
            "promotion_deployment_certificate": certificate.payload
            | {"certificate_digest": certificate.certificate_digest},
            "input_plan_digest": "1" * 64,
            "selection": {
                "selected_prior_digest": evidence.candidate_prior_digest,
                "selected_role": "candidate",
                "fallback_reason": "certified_candidate",
            },
            "observation_valid_time": "2026-08-09T00:00:00Z",
            "input_available_time": "2026-08-09T00:01:00Z",
            "decision_deadline": "2026-08-09T00:02:00Z",
            "publication_time": "2026-08-09T00:05:00Z",
        }
        late_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-operational",
            Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
            fixed_signing_time="2026-08-09T00:03:00Z",
        )
        late_ledger_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-ledger",
            Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
            fixed_signing_time="2026-08-09T00:03:00Z",
        )
        late_entry, late_root = (
            promotion_module._operational_decision_commit_digests(
                decision,
                ledger_instance_digest=certificate.ledger_instance_digest,
                sequence_number=1,
                previous_operational_decision_digest=(
                    promotion_module.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
                ),
                accepted_at=late_signer.signing_time(),
            )
        )
        with self.assertRaisesRegex(ValueError, "before its deadline"):
            promotion_module._issue_operational_decision_ledger_receipt(
                decision,
                ledger_instance_digest=certificate.ledger_instance_digest,
                sequence_number=1,
                previous_operational_decision_digest=(
                    promotion_module.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
                ),
                accepted_at=late_signer.signing_time(),
                committed_at=late_signer.signing_time(),
                commit_entry_digest=late_entry,
                committed_chain_root_digest=late_root,
                signer=late_ledger_signer,
                authority_trust_store=trust,
            )

        before_promotion = decision | {
            "input_available_time": "2026-08-09T00:00:10Z",
        }
        premature_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-operational",
            Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
            fixed_signing_time="2026-08-09T00:00:20Z",
        )
        premature_ledger_signer = (
            promotion_module.Ed25519DeploymentAuthoritySigner(
                "test-ledger",
                Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
                fixed_signing_time="2026-08-09T00:00:15Z",
            )
        )
        premature_entry, premature_root = (
            promotion_module._operational_decision_commit_digests(
                before_promotion,
                ledger_instance_digest=certificate.ledger_instance_digest,
                sequence_number=1,
                previous_operational_decision_digest=(
                    promotion_module.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
                ),
                accepted_at="2026-08-09T00:00:15Z",
            )
        )
        premature_receipt = (
            promotion_module._issue_operational_decision_ledger_receipt(
                before_promotion,
                ledger_instance_digest=certificate.ledger_instance_digest,
                sequence_number=1,
                previous_operational_decision_digest=(
                    promotion_module.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
                ),
                accepted_at="2026-08-09T00:00:15Z",
                committed_at="2026-08-09T00:00:15Z",
                commit_entry_digest=premature_entry,
                committed_chain_root_digest=premature_root,
                signer=premature_ledger_signer,
                authority_trust_store=trust,
            )
        )
        with self.assertRaisesRegex(ValueError, "prepublication window"):
            promotion_module._issue_operational_deployment_decision_certificate(
                before_promotion,
                promotion_deployment_certificate=certificate,
                promotion_evidence=evidence,
                policy=policy,
                policy_trust_store_digest="c" * 64,
                ledger_receipt=premature_receipt,
                signer=premature_signer,
                authority_trust_store=trust,
            )

    def test_operational_commit_crossing_deadline_is_rejected(self) -> None:
        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        certificate, _ = self.deployment_certificate(evidence)
        policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=evidence.candidate_prior_digest,
            parent_prior_digest=evidence.parent_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=certificate.certificate_digest,
            promotion_deployment_authority_trust_store_digest=(
                certificate.authority_trust_store_digest
            ),
            regime_classifier_digest=evidence.deployment_regime_classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=(
                evidence.certified_range_geometry_contract_digests[0]
            ),
        )
        decision = {
            "promotion_deployment_certificate": certificate.payload
            | {"certificate_digest": certificate.certificate_digest},
            "input_plan_digest": "1" * 64,
            "decision_deadline": "2026-08-09T00:02:00Z",
        }
        before_deadline = datetime.fromisoformat(
            "2026-08-09T00:01:59+00:00"
        )
        after_deadline = datetime.fromisoformat(
            "2026-08-09T00:02:01+00:00"
        )
        with patch.object(
            ledger_module, "datetime", wraps=datetime
        ) as trusted_clock:
            trusted_clock.now.side_effect = (before_deadline, after_deadline)
            with self.assertRaisesRegex(ValueError, "committed after"):
                self._latest_operational_ledger.issue_operational_deployment_decision(
                    decision,
                    promotion_deployment_certificate=certificate,
                    promotion_evidence=evidence,
                    policy=policy,
                    policy_trust_store_digest="c" * 64,
                    ledger_signer=(
                        promotion_module.Ed25519DeploymentAuthoritySigner(
                            "test-ledger",
                            Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
                            fixed_signing_time="2026-08-09T00:01:59Z",
                        )
                    ),
                    operational_signer=(
                        promotion_module.Ed25519DeploymentAuthoritySigner(
                            "test-operational",
                            Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
                            fixed_signing_time="2026-08-09T00:01:59Z",
                        )
                    ),
                    authority_trust_store_path=(
                        "/etc/advar/deployment-authorities.json"
                    ),
                )

    def test_late_final_operational_row_is_recorded_unusable(self) -> None:
        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        certificate, _ = self.deployment_certificate(evidence)
        policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=evidence.candidate_prior_digest,
            parent_prior_digest=evidence.parent_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=certificate.certificate_digest,
            promotion_deployment_authority_trust_store_digest=(
                certificate.authority_trust_store_digest
            ),
            regime_classifier_digest=evidence.deployment_regime_classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=(
                evidence.certified_range_geometry_contract_digests[0]
            ),
        )
        before = datetime.now().astimezone()
        deadline = before + timedelta(seconds=10)
        after = deadline + timedelta(microseconds=1)
        input_plan_digest = "1" * 64
        decision = {
            "promotion_deployment_certificate": certificate.payload
            | {"certificate_digest": certificate.certificate_digest},
            "deployment_policy": policy.payload
            | {"policy_digest": policy.policy_digest},
            "policy_trust_store": {
                "approved_policy_digests": [policy.policy_digest],
                "content_digest": "c" * 64,
            },
            "full_analysis_input_digest": "2" * 64,
            "input_plan_digest": input_plan_digest,
            "observation_valid_time": (before - timedelta(minutes=2)).isoformat(),
            "input_available_time": (before - timedelta(minutes=1)).isoformat(),
            "decision_deadline": deadline.isoformat(),
            "publication_time": (deadline + timedelta(minutes=1)).isoformat(),
            "operational_cycle_id": "late-final-row",
            "selection": {
                "selected_prior_digest": evidence.candidate_prior_digest,
                "selected_role": "candidate",
                "fallback_reason": "certified_candidate",
            },
        }
        with patch.object(ledger_module, "datetime", wraps=datetime) as clock:
            clock.now.side_effect = [before] * 6 + [after]
            with self.assertRaisesRegex(ValueError, "recording committed after"):
                self._latest_operational_ledger.issue_operational_deployment_decision(
                    decision,
                    promotion_deployment_certificate=certificate,
                    promotion_evidence=evidence,
                    policy=policy,
                    policy_trust_store_digest="c" * 64,
                    ledger_signer=promotion_module.Ed25519DeploymentAuthoritySigner(
                        "test-ledger",
                        Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
                        fixed_signing_time=before.isoformat(),
                    ),
                    operational_signer=(
                        promotion_module.Ed25519DeploymentAuthoritySigner(
                            "test-operational",
                            Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
                            fixed_signing_time=before.isoformat(),
                        )
                    ),
                    authority_trust_store_path=(
                        "/etc/advar/deployment-authorities.json"
                    ),
                )
        with sqlite3.connect(
            self._latest_operational_ledger.index_path
        ) as connection:
            row = connection.execute(
                "SELECT usable,decision_row_committed_at FROM "
                "operational_decision_publications WHERE certificate_digest IN "
                "(SELECT certificate_digest FROM operational_deployment_decisions_v2 "
                "WHERE input_plan_digest = ?)",
                (input_plan_digest,),
            ).fetchone()
        self.assertEqual(row, (0, after.isoformat()))

    def test_concurrent_distinct_operational_decisions_form_linear_chain(
        self,
    ) -> None:
        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        certificate, _ = self.deployment_certificate(evidence)
        policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=evidence.candidate_prior_digest,
            parent_prior_digest=evidence.parent_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            promotion_deployment_certificate_digest=(
                certificate.certificate_digest
            ),
            promotion_deployment_authority_trust_store_digest=(
                certificate.authority_trust_store_digest
            ),
            regime_classifier_digest=(
                evidence.deployment_regime_classifier_digest
            ),
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            range_geometry_contract_digest=(
                evidence.certified_range_geometry_contract_digests[0]
            ),
        )
        now = datetime.now().astimezone()

        def decision(marker: str) -> dict[str, object]:
            input_plan_digest = promotion_module.json_digest(
                {"contract": "concurrent-input-plan-v1", "marker": marker}
            )
            full_input_digest = promotion_module.json_digest(
                {"contract": "concurrent-analysis-input-v1", "marker": marker}
            )
            return {
                "promotion_deployment_certificate": certificate.payload
                | {"certificate_digest": certificate.certificate_digest},
                "deployment_policy": policy.payload
                | {"policy_digest": policy.policy_digest},
                "policy_trust_store": {
                    "approved_policy_digests": [policy.policy_digest],
                    "content_digest": "c" * 64,
                },
                "full_analysis_input_digest": full_input_digest,
                "input_plan_digest": input_plan_digest,
                "observation_valid_time": (now - timedelta(minutes=2)).isoformat(),
                "input_available_time": (now - timedelta(minutes=1)).isoformat(),
                "decision_deadline": (now + timedelta(minutes=2)).isoformat(),
                "publication_time": (now + timedelta(minutes=5)).isoformat(),
                "operational_cycle_id": promotion_module.json_digest(
                    {
                        "contract": "advar-operational-cycle-v1",
                        "input_plan_digest": input_plan_digest,
                        "full_analysis_input_digest": full_input_digest,
                    }
                ),
                "selection": {
                    "selected_prior_digest": evidence.candidate_prior_digest,
                    "selected_role": "candidate",
                    "fallback_reason": "certified_candidate",
                },
            }

        def issue(marker: str):
            return self._latest_operational_ledger.issue_operational_deployment_decision(
                decision(marker),
                promotion_deployment_certificate=certificate,
                promotion_evidence=evidence,
                policy=policy,
                policy_trust_store_digest="c" * 64,
                ledger_signer=promotion_module.Ed25519DeploymentAuthoritySigner(
                    "test-ledger",
                    Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
                ),
                operational_signer=(
                    promotion_module.Ed25519DeploymentAuthoritySigner(
                        "test-operational",
                        Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
                    )
                ),
                authority_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            issued = tuple(executor.map(issue, ("A", "B")))
        ordered = tuple(sorted(issued, key=lambda item: item.ledger_sequence_number))
        self.assertEqual(
            tuple(item.ledger_sequence_number for item in ordered),
            (1, 2),
        )
        first_receipt = json.loads(
            ordered[0].operational_ledger_receipt_payload_json
        )
        self.assertEqual(
            ordered[0].previous_operational_decision_digest,
            promotion_module.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST,
        )
        self.assertEqual(
            ordered[1].previous_operational_decision_digest,
            first_receipt["committed_chain_root_digest"],
        )
        with sqlite3.connect(
            self._latest_operational_ledger.index_path
        ) as connection:
            head = connection.execute(
                "SELECT sequence_number,certificate_digest FROM "
                "operational_decision_chain_head WHERE singleton = 1"
            ).fetchone()
            committed_count = connection.execute(
                "SELECT COUNT(*) FROM operational_decision_commits"
            ).fetchone()[0]
        second_receipt = json.loads(
            ordered[1].operational_ledger_receipt_payload_json
        )
        self.assertEqual(
            head,
            (2, second_receipt["committed_chain_root_digest"]),
        )
        self.assertEqual(committed_count, 2)

    def test_fractional_second_chronology_uses_instants_not_strings(self) -> None:
        common = {
            "valid_times": ("2026-08-09T00:00:00Z",),
            "grid_contract_digest": "1" * 64,
            "radar_product_digest": "2" * 64,
            "qc_pipeline_digest": "3" * 64,
            "background_cycle_rule_digest": "4" * 64,
            "mask_policy_digest": "5" * 64,
            "observation_valid_time": "2026-08-09T09:00:00+09:00",
            "input_available_time": "2026-08-09T00:00:00Z",
            "publication_time": "2026-08-09T00:00:01Z",
        }
        equal_instant = promotion_module.NeuralPriorInputPlan(
            **common,
            decision_deadline="2026-08-09T00:00:00.500000Z",
        )
        self.assertEqual(
            equal_instant.observation_valid_time,
            "2026-08-09T00:00:00Z",
        )
        with self.assertRaisesRegex(ValueError, "latency window"):
            promotion_module.NeuralPriorInputPlan(
                **(
                    common
                    | {
                        "input_available_time": (
                            "2026-08-09T00:00:00.500000Z"
                        ),
                        "decision_deadline": "2026-08-09T00:00:00Z",
                    }
                ),
            )

        fractional_reference_plan = self.plan(
            first_labeling_valid_time="2026-08-09T00:00:00.500000Z"
        )
        self.assertEqual(
            fractional_reference_plan.regime_reference_plans[
                0
            ].labeling_valid_time,
            "2026-08-09T00:00:00.500000Z",
        )

        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        certificate, trust = self.deployment_certificate(evidence)
        still_valid = replace(
            trust,
            revoked_at=trust.revoked_at
            | {"test-promotion": "2026-08-09T00:00:40.500000Z"},
        )
        promotion_module._validate_ledgered_promotion_deployment_certificate(
            certificate,
            authority_trust_store=still_valid,
            promotion_evidence=evidence,
        )
        revoked_before_issue = replace(
            trust,
            revoked_at=trust.revoked_at
            | {"test-promotion": "2026-08-09T00:00:39.999999Z"},
        )
        with self.assertRaisesRegex(ValueError, "root-approved"):
            promotion_module._validate_ledgered_promotion_deployment_certificate(
                certificate,
                authority_trust_store=revoked_before_issue,
                promotion_evidence=evidence,
            )

    def test_revoked_authority_and_foreign_ledger_are_rejected(self) -> None:
        evidence = self.deployment_ready(
            self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        )
        certificate, trust = self.deployment_certificate(evidence)
        revoked = replace(
            trust,
            revoked_at=trust.revoked_at
            | {"test-promotion": "2026-08-09T00:00:00+00:00"},
        )
        with self.assertRaisesRegex(ValueError, "root-approved"):
            promotion_module._validate_ledgered_promotion_deployment_certificate(
                certificate,
                authority_trust_store=revoked,
                promotion_evidence=evidence,
            )

        ledger_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-ledger",
            Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
            fixed_signing_time="2026-08-09T00:01:00Z",
        )
        with self.assertRaisesRegex(ValueError, "ledger instance"):
            promotion_module._issue_ledger_issuance_receipt(
                ledger_instance_digest="f" * 64,
                sequence_number=1,
                previous_certificate_digest=(
                    promotion_module.PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST
                ),
                promotion_evidence_digest=evidence.promotion_evidence_digest,
                scoring_replay_bundle_digest=evidence.scoring_replay_bundle_digest,
                scoring_replay_archive_sha256="a" * 64,
                scoring_evaluation_payload_sha256="b" * 64,
                scoring_artifact_digest=evidence.scoring_artifact_digest,
                scoring_completion_receipt_digest=(
                    evidence.scoring_completion_receipt_digest
                ),
                scoring_completion_completed_at="2026-08-09T00:00:00Z",
                issued_at=ledger_signer.signing_time(),
                signer=ledger_signer,
                authority_trust_store=trust,
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
                    full_analysis_input_digest=(
                        self.event_catalog(1).member_full_analysis_input_digests[0]
                    ),
                    source_object_evidence_digest="a" * 64,
                ),
                self.event_spatial_evidence(
                    self.event_catalog(2),
                    case_id="case-2",
                    full_analysis_input_digest=(
                        self.event_catalog(2).member_full_analysis_input_digests[0]
                    ),
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
                        full_analysis_input_digest=(
                            first.member_full_analysis_input_digests[0]
                        ),
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
