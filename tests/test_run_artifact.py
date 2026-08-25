from collections import Counter
from dataclasses import asdict, replace
import importlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from typing import Any, cast
import zipfile

import numpy as np
import torch
from torch import nn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar import (  # noqa: E402
    AnalysisConfig,
    CalibrationMetric,
    CalibrationRegime,
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    DeployedNeuralPriorPolicy,
    DynamicsSource,
    ForecastRunContract,
    NeuralPriorApplication,
    NeuralPriorInferenceRunner,
    NeuralPriorProbabilityContract,
    NeuralPriorStateContract,
    NowcastConfig,
    OperationalDeploymentUnsupportedError,
    OperationalCalibrationManifest,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    RadarGridTimeContract,
    SensitivityConfig,
    TendencyPairSelection,
    compute_sensitivity_snapshot,
    compute_sensitivity_snapshot_from_run,
    load_forecast_run,
    nowcast,
    save_forecast_run,
    validate_neural_prior_deployment_decision_artifact,
    variational_nowcast,
    operational_runtime_profile_digest,
    algorithm_bundle_digest,
    radar_projected_crs_digest,
    radar_projected_crs_semantic_digest,
)
from advar.variational import (  # noqa: E402
    _new_neural_prior_deployment_selection,
    neural_prior_state_censor_policy_digest,
    prepare_analysis,
)
from advar._digest import json_digest, tensor_digest  # noqa: E402
from advar.sensitivity import _LearningPolicyTrustStore  # noqa: E402
from advar.physics import FORECAST_INTEGRATOR_VERSION  # noqa: E402
import advar.run_artifact as run_artifact  # noqa: E402
import advar.promotion as promotion_module  # noqa: E402
import advar.ledger as ledger_module  # noqa: E402
from advar.run_artifact import seal_forecast_run_arrays  # noqa: E402
import test_promotion as promotion_test_module  # noqa: E402

nowcast_module = importlib.import_module("advar.nowcast")


class ForecastRunArtifactTests(unittest.TestCase):
    _base_promotion_evidence: Any = None

    def setUp(self) -> None:
        target_trust_patch = patch.object(
            ledger_module,
            "_load_training_target_source_trust_store",
            return_value=self._training_target_source_trust(),
        )
        target_trust_patch.start()
        self.addCleanup(target_trust_patch.stop)
        runtime_closure_patch = patch.object(
            promotion_module,
            "validate_current_runtime_closure",
        )
        runtime_closure_patch.start()
        self.addCleanup(runtime_closure_patch.stop)

    class _Prior(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.offset

    @staticmethod
    def _deployment_certificate_artifact(
        *,
        promotion_evidence: Any,
    ) -> dict[str, object]:
        ledger_key = Ed25519PrivateKey.from_private_bytes(b"\x03" * 32)
        promotion_key = Ed25519PrivateKey.from_private_bytes(b"\x04" * 32)
        trust = ForecastRunArtifactTests._deployment_certificate_trust()
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
            ledger_instance_digest="6" * 64,
            sequence_number=1,
            previous_certificate_digest=(
                promotion_module.PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST
            ),
            promotion_evidence_digest=promotion_evidence.promotion_evidence_digest,
            scoring_replay_bundle_digest=(
                promotion_evidence.scoring_replay_bundle_digest
            ),
            scoring_replay_archive_sha256="a" * 64,
            scoring_evaluation_payload_sha256="b" * 64,
            scoring_artifact_digest=promotion_evidence.scoring_artifact_digest,
            scoring_completion_receipt_digest=(
                promotion_evidence.scoring_completion_receipt_digest
            ),
            scoring_completion_completed_at="2026-08-09T00:00:00Z",
            issued_at=ledger_signer.signing_time(),
            signer=ledger_signer,
            authority_trust_store=trust,
        )
        certificate = promotion_module._issue_ledgered_promotion_deployment_certificate(
            promotion_evidence,
            issued_at=promotion_signer.signing_time(),
            ledger_issuance_receipt=receipt,
            signer=promotion_signer,
            authority_trust_store=trust,
            raw_ingestor_trust_store_digest=(
                promotion_evidence.raw_ingestor_trust_store_digest
            ),
        )
        return certificate.payload | {
            "certificate_digest": certificate.certificate_digest
        }

    @classmethod
    def _promotion_evidence(
        cls,
        *,
        candidate_prior_digest: str,
        parent_prior_digest: str,
        classifier_digest: str,
        classifier_manifest_digest: str,
        range_geometry_digest: str,
    ) -> Any:
        if cls._base_promotion_evidence is None:
            fixture = promotion_test_module.NeuralPriorPromotionTests()
            cls._base_promotion_evidence = fixture.deployment_ready(
                fixture.compute(
                    (fixture.evaluation(1, -0.2), fixture.evaluation(2, -0.3))
                )
            )
        return replace(
            cls._base_promotion_evidence,
            candidate_prior_digest=candidate_prior_digest,
            parent_prior_digest=parent_prior_digest,
            deployment_regime_classifier_digest=classifier_digest,
            deployment_regime_classifier_manifest_digest=(
                classifier_manifest_digest
            ),
            raw_ingestor_trust_store_digest="9" * 64,
            certified_applicability_regime_groups=(("convective", "near_range"),),
            certified_range_geometry_contract_digests=(range_geometry_digest,),
            deployment_eligible=True,
        )

    @staticmethod
    def _deployment_certificate_trust():
        ledger_key = Ed25519PrivateKey.from_private_bytes(b"\x03" * 32)
        promotion_key = Ed25519PrivateKey.from_private_bytes(b"\x04" * 32)
        operational_key = Ed25519PrivateKey.from_private_bytes(b"\x05" * 32)
        runtime_activation_key = Ed25519PrivateKey.from_private_bytes(
            b"\x06" * 32
        )
        release_approval_key = Ed25519PrivateKey.from_private_bytes(b"\x07" * 32)
        return promotion_module._PromotionDeploymentAuthorityTrustStore(
            keys={
                "test-ledger": ledger_key.public_key(),
                "test-promotion": promotion_key.public_key(),
                "test-operational": operational_key.public_key(),
                "test-runtime-activation": (
                    runtime_activation_key.public_key()
                ),
                "test-release-approval": release_approval_key.public_key(),
            },
            content_digest="7" * 64,
            roles={
                "test-ledger": frozenset({"ledger_issuance"}),
                "test-promotion": frozenset({"promotion_certificate"}),
                "test-operational": frozenset({"operational_decision"}),
                "test-runtime-activation": frozenset({"runtime_activation"}),
                "test-release-approval": frozenset({"release_approval"}),
            },
            not_before={
                name: "2026-01-01T00:00:00+00:00"
                for name in (
                    "test-ledger",
                    "test-promotion",
                    "test-operational",
                    "test-runtime-activation",
                    "test-release-approval",
                )
            },
            not_after={
                name: "2032-01-01T00:00:00+00:00"
                for name in (
                    "test-ledger",
                    "test-promotion",
                    "test-operational",
                    "test-runtime-activation",
                    "test-release-approval",
                )
            },
            revoked_at={
                name: None
                for name in (
                    "test-ledger",
                    "test-promotion",
                    "test-operational",
                    "test-runtime-activation",
                    "test-release-approval",
                )
            },
            ledger_instance_digests={
                "test-ledger": frozenset({"6" * 64}),
                "test-promotion": frozenset(),
                "test-operational": frozenset(),
                "test-runtime-activation": frozenset(),
                "test-release-approval": frozenset(),
            },
            ledger_instance_index_paths={
                "6" * 64: Path("/approved/advar/index.sqlite")
            },
        )

    @staticmethod
    def _deployment_policy_trust(artifact_json: str):
        payload = json.loads(artifact_json)
        trust = payload["policy_trust_store"]
        return _LearningPolicyTrustStore(
            approved_policy_digests=frozenset(
                trust["approved_policy_digests"]
            ),
            content_digest=trust["content_digest"],
        )

    @staticmethod
    def _training_target_source_trust():
        return (
            promotion_test_module.NeuralPriorPromotionTests()
            .plan()
            .training_target_source_trust_store
        )

    @classmethod
    def _authorize_operational_decision(
        cls,
        artifact: dict[str, object],
        *,
        promotion_evidence: Any,
        policy: DeployedNeuralPriorPolicy,
        promotion_certificate_payload: dict[str, object],
    ) -> dict[str, object]:
        artifact = artifact | {
            "input_plan_digest": artifact.get("input_plan_digest", "1" * 64),
            "observation_valid_time": artifact.get(
                "observation_valid_time", "2026-08-09T00:00:00+00:00"
            ),
            "input_available_time": artifact.get(
                "input_available_time", "2026-08-09T00:00:00+00:00"
            ),
            "decision_deadline": artifact.get(
                "decision_deadline", "2026-08-09T00:02:00+00:00"
            ),
            "publication_time": artifact.get(
                "publication_time", "2026-08-09T00:05:00+00:00"
            ),
        }
        artifact["operational_cycle_id"] = json_digest(
            {
                "contract": "advar-operational-cycle-v2",
                "input_plan_digest": artifact["input_plan_digest"],
                "full_analysis_input_digest": artifact[
                    "full_analysis_input_digest"
                ],
                "analysis_input_derivation_artifact_digest": artifact[
                    "analysis_input_derivation_artifact_digest"
                ],
                "global_raw_resolution_receipt_digest": artifact[
                    "global_raw_resolution_receipt_digest"
                ],
                "resolved_raw_volume_identity_set_digest": artifact[
                    "resolved_raw_volume_identity_set_digest"
                ],
            }
        )
        certificate_values = dict(promotion_certificate_payload)
        certificate_values.pop("certificate_digest")
        promotion_certificate = (
            promotion_module._ledgered_promotion_deployment_certificate_from_payload(
                certificate_values
            )
        )
        key = Ed25519PrivateKey.from_private_bytes(b"\x05" * 32)
        signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-operational",
            key,
            fixed_signing_time=cast(str, artifact["input_available_time"]),
        )
        ledger_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-ledger",
            Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
            fixed_signing_time=cast(str, artifact["input_available_time"]),
        )
        accepted_at = ledger_signer.signing_time()
        runtime_signer = promotion_module.Ed25519DeploymentAuthoritySigner(
            "test-runtime-activation",
            Ed25519PrivateKey.from_private_bytes(b"\x06" * 32),
            fixed_signing_time=accepted_at,
        )
        release_approval = (
            promotion_module._issue_deployment_bundle_release_approval(
                deployment_bundle_digest="a" * 64,
                bundle_manifest_digest="9" * 64,
                source_commit="a" * 40,
                repository="gonos2k/AD4DVAR-radar",
                source_ref="refs/tags/v0.93.0",
                platform="linux-x86_64-cpu",
                runtime_mode="deployable",
                expires_at="2031-01-01T00:00:00Z",
                signer=promotion_module.Ed25519DeploymentAuthoritySigner(
                    "test-release-approval",
                    Ed25519PrivateKey.from_private_bytes(b"\x07" * 32),
                    fixed_signing_time=accepted_at,
                ),
                authority_trust_store=cls._deployment_certificate_trust(),
            )
        )
        runtime_activation_receipt = (
            promotion_module._issue_deployment_runtime_activation_receipt(
                release_approval=release_approval,
                runtime_tree_digest="b" * 64,
                interpreter_closure_digest="c" * 64,
                installation_attestation_sha256="d" * 64,
                deployment_instance_digest="e" * 64,
                host_identity_digest="f" * 64,
                runtime_mode="deployable",
                activation_sequence_number=1,
                previous_activation_receipt_digest=(
                    promotion_module.DEPLOYMENT_RUNTIME_ACTIVATION_GENESIS_DIGEST
                ),
                expires_at="2031-01-01T00:00:00Z",
                signer=runtime_signer,
                authority_trust_store=cls._deployment_certificate_trust(),
            )
        )
        artifact["deployment_bundle_release_approval"] = (
            release_approval.payload
            | {"approval_digest": release_approval.approval_digest}
        )
        artifact["deployment_runtime_activation_receipt"] = (
            runtime_activation_receipt.payload
            | {"receipt_digest": runtime_activation_receipt.receipt_digest}
        )
        commit_entry_digest, committed_chain_root_digest = (
            promotion_module._operational_decision_commit_digests(
                artifact,
                ledger_instance_digest=(
                    promotion_certificate.ledger_instance_digest
                ),
                sequence_number=1,
                previous_operational_decision_digest=(
                    promotion_module.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
                ),
                accepted_at=accepted_at,
            )
        )
        ledger_receipt = promotion_module._issue_operational_decision_ledger_receipt(
            artifact,
            ledger_instance_digest=promotion_certificate.ledger_instance_digest,
            sequence_number=1,
            previous_operational_decision_digest=(
                promotion_module.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
            ),
            accepted_at=accepted_at,
            committed_at=accepted_at,
            commit_entry_digest=commit_entry_digest,
            committed_chain_root_digest=committed_chain_root_digest,
            signer=ledger_signer,
            authority_trust_store=cls._deployment_certificate_trust(),
        )
        decision_certificate = (
            promotion_module._issue_operational_deployment_decision_certificate(
                artifact,
                deployment_bundle_release_approval=release_approval,
                promotion_deployment_certificate=promotion_certificate,
                promotion_evidence=promotion_evidence,
                policy=policy,
                policy_trust_store_digest=str(
                    cast(dict[str, object], artifact["policy_trust_store"])[
                        "content_digest"
                    ]
                ),
                regime_evidence=None,
                range_partition_evidence=None,
                range_geometry_contract=None,
                ledger_receipt=ledger_receipt,
                deployment_runtime_activation_receipt=(
                    runtime_activation_receipt
                ),
                signer=signer,
                authority_trust_store=cls._deployment_certificate_trust(),
            )
        )
        publication_receipt = (
            promotion_module._issue_operational_decision_publication_receipt(
                decision_certificate,
                decision_row_committed_at=accepted_at,
                signer=ledger_signer,
                authority_trust_store=cls._deployment_certificate_trust(),
            )
        )
        activation_receipt = (
            promotion_module._issue_operational_decision_activation_receipt(
                decision_certificate,
                publication_receipt,
                publication_payload_committed_at=accepted_at,
                activation_authorized_at=accepted_at,
                publication_guard_interval_seconds=0.05,
                committed_chain_root_digest=committed_chain_root_digest,
                signer=ledger_signer,
                authority_trust_store=cls._deployment_certificate_trust(),
            )
        )
        commit_authorization = (
            promotion_module._issue_operational_decision_commit_authorization_receipt(
                decision_certificate,
                activation_receipt,
                terminal_commit_authorized_at=accepted_at,
                signer=ledger_signer,
                authority_trust_store=cls._deployment_certificate_trust(),
            )
        )
        return artifact | {
            "operational_decision_certificate": (
                decision_certificate.payload
                | {"certificate_digest": decision_certificate.certificate_digest}
            ),
            "operational_decision_publication_receipt": (
                publication_receipt.payload
                | {"receipt_digest": publication_receipt.receipt_digest}
            ),
            "operational_decision_activation_receipt": (
                activation_receipt.payload
                | {"receipt_digest": activation_receipt.receipt_digest}
            ),
            "operational_decision_commit_authorization_receipt": (
                commit_authorization.payload
                | {"receipt_digest": commit_authorization.receipt_digest}
            ),
        }

    def _state_contract(self) -> NeuralPriorStateContract:
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

    def _probability_contract(self) -> NeuralPriorProbabilityContract:
        return NeuralPriorProbabilityContract(
            support_threshold_dbz=5.0,
            support_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
        )

    def _current_provenance_run(
        self,
        frames: torch.Tensor,
    ) -> ForecastRunContract:
        nowcast_config = self._operational_nowcast_config()
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-09T00:00:00Z",
                "2026-08-09T00:10:00Z",
                "2026-08-09T00:20:00Z",
            ),
            dx_m=1_000.0,
            dy_m=1_000.0,
            projection="EPSG:5179",
            grid_hash="d" * 64,
        )
        input_plan = promotion_module.NeuralPriorInputPlan(
            valid_times=grid.valid_times,
            grid_contract_digest=grid.digest,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            background_cycle_rule_digest="b" * 64,
            mask_policy_digest="3" * 64,
            observation_valid_time=grid.valid_times[-1],
            input_available_time="2026-08-09T00:20:30Z",
            decision_deadline="2026-08-09T00:22:00Z",
            publication_time="2026-08-09T00:25:00Z",
        )
        source_identity = promotion_module.OperationalDataIdentity(
            radar_class="single-site-grid-product",
            qc_pipeline_digest=input_plan.qc_pipeline_digest,
            observation_error_model_digest="6" * 64,
            background_model_digest="8" * 64,
            radar_product_digest=input_plan.radar_product_digest,
            background_cycle_rule_digest=input_plan.background_cycle_rule_digest,
            mask_policy_digest=input_plan.mask_policy_digest,
        )
        analysis_config = self._operational_analysis_config()
        analysis_config_payload = asdict(analysis_config)
        analysis_config_json = json.dumps(
            analysis_config_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        calibration = OperationalCalibrationManifest(
            calibration_id="run-artifact-current-provenance",
            profile_kind="p1",
            expected_runtime_profile_digest=operational_runtime_profile_digest(
                nowcast_config,
                grid,
                analysis_config=asdict(analysis_config),
            ),
            expected_algorithm_bundle_digest=algorithm_bundle_digest(),
            calibration_dataset_digest="c" * 64,
            validation_dataset_digest="d" * 64,
            data_identity=source_identity,
            training_period=(
                "2025-01-01T00:00:00Z",
                "2025-07-01T00:00:00Z",
            ),
            validation_period=(
                "2025-07-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
            validation_case_count=1,
            validation_regimes=(CalibrationRegime("convective", 1),),
            validation_metrics=(
                CalibrationMetric(
                    name="csi_35",
                    definition_digest="e" * 64,
                    direction="maximize",
                    acceptance_threshold=0.4,
                    value=0.5,
                ),
            ),
        )
        run = ForecastRunContract.from_inputs(
            nowcast_config,
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
            observation_quality_weight=torch.ones_like(frames),
            observation_std_dbz=torch.full_like(frames, 2.0),
            source_available_mask=torch.ones_like(frames, dtype=torch.bool),
            grid_time_contract=grid,
            analysis_config_json=analysis_config_json,
            analysis_config_digest=json_digest(analysis_config_payload),
            analysis_input_digest="f" * 64,
            operational_calibration_manifest_json=calibration.json,
            operational_calibration_manifest_digest=calibration.digest,
            operational_calibration_approval_digest=calibration.digest,
            operational_data_identity_json=source_identity.json,
            operational_data_identity_digest=source_identity.digest,
            input_plan_json=input_plan.json,
            input_plan_digest=input_plan.plan_digest,
        )
        processor_key = Ed25519PrivateKey.from_private_bytes(b"\x23" * 32)
        unsigned = {
            "contract": "analysis-input-derivation-artifact-v5",
            "case_id": "run-artifact-current-provenance",
            "input_plan_digest": input_plan.plan_digest,
            "resolved_raw_observation_receipt_digests": ["2" * 64],
            "canonical_raw_volume_identity_digests": ["3" * 64],
            "global_raw_resolution_receipt_digest": "4" * 64,
            "decoder_version_digest": "5" * 64,
            "qc_algorithm_digest": "6" * 64,
            "qc_policy_digest": input_plan.mask_policy_digest,
            "source_selection_evidence_digest": "8" * 64,
            "regrid_algorithm_digest": "9" * 64,
            "grid_contract_digest": run.grid_time_contract_digest,
            "background_cycle_rule_digest": input_plan.background_cycle_rule_digest,
            "background_valid_times": [],
            "background_source_identity_digest": None,
            "background_input_identity_digests": [],
            "input_frames_digest": run.input_frames_digest,
            "observation_masks_digest": run.observation_masks_digest,
            "observation_quality_weight_digest": (
                run.observation_quality_weight_digest
            ),
            "observation_std_dbz_digest": run.observation_std_dbz_digest,
            "source_available_mask_digest": (
                run.source_available_mask_digest
            ),
            "learned_model_input_features_digest": (
                run.learned_model_input_features_digest
            ),
            "background_frames_digest": None,
            "input_bundle_digest": run.input_bundle_digest,
            "full_analysis_input_digest": run.full_analysis_input_digest,
            "processed_at": "2026-08-09T00:20:30Z",
            "processor_id": "test-analysis-processor",
            "processor_public_key_hex": (
                processor_key.public_key().public_bytes_raw().hex()
            ),
        }
        artifact = promotion_module.AnalysisInputDerivationArtifact(
            **cast(Any, unsigned),
            processor_signature_hex=processor_key.sign(
                json_digest(unsigned).encode("ascii")
            ).hex(),
        )
        result = replace(
            run,
            analysis_input_derivation_artifact_json=json.dumps(
                artifact.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
            analysis_input_derivation_artifact_digest=artifact.artifact_digest,
        )
        result.validate_integrity()
        return result

    @staticmethod
    def _operational_nowcast_config() -> NowcastConfig:
        return NowcastConfig(
            maximum_motion_speed_mps=30.0,
            pair_echo_dilation_m=1_000.0,
            phase_correlation_sidelobe_radius_m=1_000.0,
        )

    @staticmethod
    def _operational_analysis_config() -> AnalysisConfig:
        return AnalysisConfig(
            execution_mode="operational",
            operational_calibration_id="run-artifact-current-provenance",
            motion_increment_scale_mps=2.0,
            causal_support_uncertainty_m=1_000.0,
            amplitude_displacement_tolerance_m=1_000.0,
            amplitude_information_policy="operational_fallback",
            amplitude_confidence_policy="operational_fallback",
        )

    @staticmethod
    def _variational_from_current_run(
        frames: torch.Tensor,
        prior: NeuralPriorApplication | None,
        input_run: ForecastRunContract,
    ):
        if input_run.operational_calibration_manifest_json is None:
            return variational_nowcast(frames, neural_prior=prior)
        calibration = OperationalCalibrationManifest.from_json(
            cast(str, input_run.operational_calibration_manifest_json)
        )
        return variational_nowcast(
            frames,
            neural_prior=prior,
            nowcast_config=input_run.config,
            analysis_config=ForecastRunArtifactTests._operational_analysis_config(),
            grid_time_contract=input_run.grid_time_contract,
            operational_calibration_manifest=(
                calibration
            ),
            operational_calibration_approval_digest=(
                input_run.operational_calibration_approval_digest
            ),
            operational_data_identity=(
                promotion_module.OperationalDataIdentity.from_json(
                    cast(str, input_run.operational_data_identity_json)
                )
            ),
            input_plan_json=input_run.input_plan_json,
            input_plan_digest=input_run.input_plan_digest,
            analysis_input_derivation_artifact_json=(
                input_run.analysis_input_derivation_artifact_json
            ),
            analysis_input_derivation_artifact_digest=(
                input_run.analysis_input_derivation_artifact_digest
            ),
        )

    @staticmethod
    def _legacy_deployment_fixture_from_current_run(
        frames: torch.Tensor,
        prior: NeuralPriorApplication,
        input_run: ForecastRunContract,
    ):
        """Manufacture predecessor bytes; never authorize current action."""

        with patch.object(
            nowcast_module,
            "_validate_prior_deployment_lineage",
        ):
            result, analysis = (
                ForecastRunArtifactTests._variational_from_current_run(
                    frames,
                    prior,
                    input_run,
                )
            )
        audit_run = replace(
            result.run,
            prior_deployment_lineage_contract=(
                "neural-prior-deployment-lineage-v19-audit"
            ),
        )
        audit_result = replace(
            result,
            run=audit_run,
            forecast_run_digest=nowcast_module._forecast_run_identity_digest(
                audit_run,
                result.state_metadata_digest,
                result.forecast_dbz_digest,
                result.valid_mask_digest,
            ),
        )
        return audit_result, analysis

    def _range_geometry_artifact(
        self,
        frames: torch.Tensor,
        *,
        grid_contract_digest: str = "d" * 64,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "radar_site_digest": "e" * 64,
            "radar_site_location_digest": "f" * 64,
            "grid_contract_digest": grid_contract_digest,
            "radar_x_m": 0.0,
            "radar_y_m": 0.0,
            "range_regime_labels": ["near_range"],
            "radial_distance_edges_m": [0.0, 100_000.0],
            "horizontal_range_rule_digest": "4" * 64,
            "grid_x_m_digest": tensor_digest(torch.zeros(frames.shape[1:])),
            "grid_y_m_digest": tensor_digest(torch.zeros(frames.shape[1:])),
            "resolver_algorithm": "projected-horizontal-euclidean-range-v3",
            "contract": "radar-horizontal-range-geometry-contract-v3",
        }
        return payload | {"contract_digest": json_digest(payload)}

    def _deployment_selection(
        self,
        runner: NeuralPriorInferenceRunner,
        input_run: ForecastRunContract,
        frames: torch.Tensor,
    ) -> Any:
        assert input_run.full_analysis_input_digest is not None
        classifier_digest = "6" * 64
        classifier_manifest_digest = "a" * 64
        assert input_run.grid_time_contract_digest is not None
        range_geometry = self._range_geometry_artifact(
            frames,
            grid_contract_digest=input_run.grid_time_contract_digest,
        )
        range_geometry_digest = str(range_geometry["contract_digest"])
        range_partition = {
            "contract": "radar-range-partition-evidence-v4",
            "range_geometry_contract_digest": range_geometry_digest,
            "grid_contract_digest": input_run.grid_time_contract_digest,
            "range_regime_labels": ["near_range"],
            "range_band_mask_digests": [
                tensor_digest(torch.ones(frames.shape[1:], dtype=torch.bool))
            ],
            "valid_range_domain_mask_digest": tensor_digest(
                torch.ones(frames.shape[1:], dtype=torch.bool)
            ),
            "active_range_regimes": ["near_range"],
            "grid_shape": list(frames.shape[1:]),
        }
        range_partition_digest = json_digest(range_partition)
        regime = {
            "contract": "neural-prior-regime-classification-evidence-v4",
            "full_analysis_input_digest": input_run.full_analysis_input_digest,
            "input_frames_digest": tensor_digest(frames),
            "classifier_digest": classifier_digest,
            "regime": "convective",
            "range_regime": "near_range",
            "active_range_regimes": ["near_range"],
            "regime_confidence": 1.0,
            "range_regime_confidence": 1.0,
            "regime_labels": ["convective", "unknown"],
            "range_regime_labels": ["near_range"],
            "range_probability_contract": (
                "conditionally-independent-bernoulli-range-heads-v1"
            ),
            "range_presence_probability_threshold": 0.8,
            "regime_probabilities": [1.0, 0.0],
            "range_regime_probabilities": [1.0],
            "regime_entropy": 0.0,
            "is_ood": False,
            "numerical_runtime_digest": "9" * 64,
            "input_dtype": str(frames.dtype),
            "input_device": str(frames.device),
            "weather_top1_top2_gap": 1.0,
            "minimum_range_presence_margin": 0.2,
        }
        regime_digest = json_digest(regime)
        promotion_evidence = self._promotion_evidence(
            candidate_prior_digest="f" * 64,
            parent_prior_digest=runner.neural_prior_digest,
            classifier_digest=classifier_digest,
            classifier_manifest_digest=classifier_manifest_digest,
            range_geometry_digest=range_geometry_digest,
        )
        certificate = self._deployment_certificate_artifact(
            promotion_evidence=promotion_evidence,
        )
        promotion_digest = str(certificate["promotion_evidence_digest"])
        policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest="f" * 64,
            parent_prior_digest=runner.neural_prior_digest,
            promotion_evidence_digest=promotion_digest,
            promotion_deployment_certificate_digest=str(
                certificate["certificate_digest"]
            ),
            promotion_deployment_authority_trust_store_digest=str(
                certificate["authority_trust_store_digest"]
            ),
            regime_classifier_digest=classifier_digest,
            regime_classifier_manifest_digest=classifier_manifest_digest,
            range_geometry_contract_digest=range_geometry_digest,
        )
        trust = {
            "contract": "advar-learning-policy-trust-store-v1",
            "approved_policy_digests": [policy.policy_digest],
        }
        artifact = {
            "contract": "neural-prior-deployment-decision-artifact-v19",
            "routing_semantic_replay_verified": False,
            "full_analysis_input_digest": input_run.full_analysis_input_digest,
            "analysis_input_derivation_artifact_digest": (
                input_run.analysis_input_derivation_artifact_digest
            ),
            "global_raw_resolution_receipt_digest": "4" * 64,
            "resolved_raw_volume_identity_set_digest": json_digest(
                {
                    "contract": "resolved-raw-volume-identity-set-v1",
                    "identity_digests": ["3" * 64],
                }
            ),
            "analysis_processor_trust_store_digest": "7" * 64,
            "raw_ingestor_trust_store_digest": (
                promotion_evidence.raw_ingestor_trust_store_digest
            ),
            "training_target_source_trust_store_digest": (
                promotion_evidence.training_target_source_trust_store_digest
            ),
            "input_plan_digest": input_run.input_plan_digest,
            "operational_grid_contract_digest": (
                input_run.grid_time_contract_digest
            ),
            "operational_frame_shape": list(frames.shape[1:]),
            "operational_radar_source_kind": None,
            "operational_radar_site_digest": None,
            "operational_radar_site_location_digest": None,
            "regime_classification_evidence": regime
            | {"evidence_digest": regime_digest},
            "deployment_policy": policy.payload
            | {"policy_digest": policy.policy_digest},
            "range_partition_evidence": range_partition
            | {"evidence_digest": range_partition_digest},
            "range_geometry_contract": range_geometry,
            "promotion_deployment_certificate": certificate,
            "policy_trust_store": trust
            | {"content_digest": json_digest(trust)},
            "selection": {
                "selected_prior_digest": runner.neural_prior_digest,
                "selected_role": "parent",
                "fallback_reason": "unverified_routing_evidence",
                "deployment_confidence_margin": 0.0,
            },
        }
        retained_input_plan = json.loads(cast(str, input_run.input_plan_json))
        for name in (
            "observation_valid_time",
            "input_available_time",
            "decision_deadline",
            "publication_time",
        ):
            artifact[name] = retained_input_plan[name]
        artifact["analysis_input_provenance_commitment_digest"] = json_digest(
            {
                "contract": "operational-analysis-input-provenance-commitment-v2",
                "analysis_input_derivation_artifact_digest": artifact[
                    "analysis_input_derivation_artifact_digest"
                ],
                "global_raw_resolution_receipt_digest": artifact[
                    "global_raw_resolution_receipt_digest"
                ],
                "resolved_raw_volume_identity_set_digest": artifact[
                    "resolved_raw_volume_identity_set_digest"
                ],
                "analysis_processor_trust_store_digest": artifact[
                    "analysis_processor_trust_store_digest"
                ],
                "raw_ingestor_trust_store_digest": artifact[
                    "raw_ingestor_trust_store_digest"
                ],
            }
        )
        artifact = self._authorize_operational_decision(
            artifact,
            promotion_evidence=promotion_evidence,
            policy=policy,
            promotion_certificate_payload=certificate,
        )
        artifact_json = json.dumps(
            artifact, sort_keys=True, separators=(",", ":")
        )
        return _new_neural_prior_deployment_selection(
            selected_prior_digest=runner.neural_prior_digest,
            selected_role="parent",
            full_analysis_input_digest=input_run.full_analysis_input_digest,
            promotion_evidence_digest=promotion_digest,
            promotion_deployment_certificate_digest=str(
                certificate["certificate_digest"]
            ),
            regime_classification_evidence_digest=regime_digest,
            deployment_policy_digest=policy.policy_digest,
            deployment_policy_trust_store_digest=json_digest(trust),
            range_geometry_contract_digest=range_geometry_digest,
            range_partition_evidence_digest=range_partition_digest,
            classifier_numerical_runtime_digest="9" * 64,
            classifier_input_dtype=str(frames.dtype),
            classifier_input_device=str(frames.device),
            weather_top1_top2_gap=1.0,
            minimum_range_presence_margin=0.2,
            deployment_confidence_margin=0.0,
            deployment_decision_artifact_json=artifact_json,
            deployment_decision_artifact_digest=json_digest(artifact),
            fallback_reason="unverified_routing_evidence",
        )

    def test_deployment_replay_binds_the_operational_radar_site(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        input_run = self._current_provenance_run(frames)
        selection = self._deployment_selection(runner, input_run, frames)

        with patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=self._deployment_certificate_trust(),
        ), patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=self._deployment_policy_trust(
                selection.deployment_decision_artifact_json
            ),
        ), patch.object(
            ledger_module,
            "_load_raw_ingestor_trust_store",
            return_value=SimpleNamespace(content_digest="9" * 64),
        ), self.assertRaisesRegex(ValueError, "current forecast run"):
            validate_neural_prior_deployment_decision_artifact(
                selection.deployment_decision_artifact_json,
                expected_operational_radar_source_kind="single_site",
                expected_operational_radar_site_digest="e" * 64,
                expected_operational_radar_site_location_digest="f" * 64,
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                deployment_policy_trust_store_path=(
                    "/etc/advar/deployment-policies.json"
                ),
                raw_ingestor_trust_store_path=(
                    "/etc/advar/raw-ingestors.json"
                ),
                training_target_source_trust_store_path=(
                    "/etc/advar/training-target-sources.json"
                ),
            )

    def test_current_deployment_requires_terminal_activation_receipt(self) -> None:
        frames = self.frames()
        input_run = self._current_provenance_run(frames)
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        selection = self._deployment_selection(runner, input_run, frames)
        artifact = json.loads(selection.deployment_decision_artifact_json)
        artifact.pop("operational_decision_activation_receipt")

        with patch.object(
            ledger_module,
            "_load_raw_ingestor_trust_store",
            return_value=SimpleNamespace(content_digest="9" * 64),
        ), self.assertRaisesRegex(ValueError, "incomplete"):
            validate_neural_prior_deployment_decision_artifact(
                json.dumps(artifact, sort_keys=True, separators=(",", ":")),
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                deployment_policy_trust_store_path=(
                    "/etc/advar/deployment-policies.json"
                ),
                raw_ingestor_trust_store_path=(
                    "/etc/advar/raw-ingestors.json"
                ),
                training_target_source_trust_store_path=(
                    "/etc/advar/training-target-sources.json"
                ),
            )

        artifact = json.loads(selection.deployment_decision_artifact_json)
        artifact.pop("operational_decision_commit_authorization_receipt")
        with patch.object(
            ledger_module,
            "_load_raw_ingestor_trust_store",
            return_value=SimpleNamespace(content_digest="9" * 64),
        ), self.assertRaisesRegex(ValueError, "incomplete"):
            validate_neural_prior_deployment_decision_artifact(
                json.dumps(artifact, sort_keys=True, separators=(",", ":")),
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                deployment_policy_trust_store_path=(
                    "/etc/advar/deployment-policies.json"
                ),
                raw_ingestor_trust_store_path=(
                    "/etc/advar/raw-ingestors.json"
                ),
                training_target_source_trust_store_path=(
                    "/etc/advar/training-target-sources.json"
                ),
            )

    def test_current_deployment_binds_exact_forecast_run_lineage(self) -> None:
        frames = self.frames()
        input_run = self._current_provenance_run(frames)
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        selection = self._deployment_selection(runner, input_run, frames)
        mismatches = {
            "expected_full_analysis_input_digest": "0" * 64,
            "expected_analysis_input_derivation_artifact_digest": "1" * 64,
            "expected_promotion_evidence_digest": "2" * 64,
            "expected_regime_classification_evidence_digest": "3" * 64,
            "expected_deployment_policy_digest": "4" * 64,
            "expected_deployment_selection_digest": "5" * 64,
            "expected_selected_prior_digest": "6" * 64,
        }
        for name, value in mismatches.items():
            with self.subTest(name=name), patch.object(
                promotion_module,
                "_load_promotion_deployment_authority_trust_store",
                return_value=self._deployment_certificate_trust(),
            ), patch.object(
                promotion_module,
                "_load_learning_policy_trust_store",
                return_value=self._deployment_policy_trust(
                    selection.deployment_decision_artifact_json
                ),
            ), patch.object(
                ledger_module,
                "_load_raw_ingestor_trust_store",
                return_value=SimpleNamespace(content_digest="9" * 64),
            ), self.assertRaisesRegex(ValueError, "forecast run"):
                validate_neural_prior_deployment_decision_artifact(
                    selection.deployment_decision_artifact_json,
                    **{name: value},
                    deployment_certificate_trust_store_path=(
                        "/etc/advar/deployment-authorities.json"
                    ),
                    deployment_policy_trust_store_path=(
                        "/etc/advar/deployment-policies.json"
                    ),
                    raw_ingestor_trust_store_path=(
                        "/etc/advar/raw-ingestors.json"
                    ),
                    training_target_source_trust_store_path=(
                        "/etc/advar/training-target-sources.json"
                    ),
                )

    def test_signed_promotion_and_external_policy_anchor_durable_decision(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        input_run = self._current_provenance_run(frames)
        selection = self._deployment_selection(runner, input_run, frames)
        original = json.loads(selection.deployment_decision_artifact_json)
        original_policy_trust = self._deployment_policy_trust(
            selection.deployment_decision_artifact_json
        )

        for field, value in (
            ("candidate_prior_digest", "1" * 64),
            ("parent_prior_digest", "2" * 64),
            ("regime_classifier_digest", "3" * 64),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(original))
                policy_payload = changed["deployment_policy"]
                policy_payload[field] = value
                unsigned_policy = dict(policy_payload)
                unsigned_policy.pop("policy_digest")
                policy_digest = json_digest(unsigned_policy)
                policy_payload["policy_digest"] = policy_digest
                changed["policy_trust_store"]["approved_policy_digests"] = [
                    policy_digest
                ]
                changed["policy_trust_store"]["content_digest"] = json_digest(
                    {
                        "contract": "advar-learning-policy-trust-store-v1",
                        "approved_policy_digests": [policy_digest],
                    }
                )
                changed_json = json.dumps(
                    changed,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                changed_trust = self._deployment_policy_trust(changed_json)
                with patch.object(
                    promotion_module,
                    "_load_promotion_deployment_authority_trust_store",
                    return_value=self._deployment_certificate_trust(),
                ), patch.object(
                    promotion_module,
                    "_load_learning_policy_trust_store",
                    return_value=changed_trust,
                ), patch.object(
                    ledger_module,
                    "_load_raw_ingestor_trust_store",
                    return_value=SimpleNamespace(content_digest="9" * 64),
                ), self.assertRaisesRegex(ValueError, "certificate|lineage"):
                    validate_neural_prior_deployment_decision_artifact(
                        changed_json,
                        deployment_certificate_trust_store_path=(
                            "/etc/advar/deployment-authorities.json"
                        ),
                        deployment_policy_trust_store_path=(
                            "/etc/advar/deployment-policies.json"
                        ),
                        raw_ingestor_trust_store_path=(
                            "/etc/advar/raw-ingestors.json"
                        ),
                        training_target_source_trust_store_path=(
                            "/etc/advar/training-target-sources.json"
                        ),
                    )

        relaxed = json.loads(json.dumps(original))
        relaxed_policy = relaxed["deployment_policy"]
        relaxed_policy["minimum_regime_confidence"] = 0.01
        unsigned_relaxed_policy = dict(relaxed_policy)
        unsigned_relaxed_policy.pop("policy_digest")
        relaxed_digest = json_digest(unsigned_relaxed_policy)
        relaxed_policy["policy_digest"] = relaxed_digest
        relaxed["policy_trust_store"]["approved_policy_digests"] = [
            relaxed_digest
        ]
        relaxed["policy_trust_store"]["content_digest"] = json_digest(
            {
                "contract": "advar-learning-policy-trust-store-v1",
                "approved_policy_digests": [relaxed_digest],
            }
        )
        with patch.object(
            promotion_module,
            "_load_promotion_deployment_authority_trust_store",
            return_value=self._deployment_certificate_trust(),
        ), patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=original_policy_trust,
        ), patch.object(
            ledger_module,
            "_load_raw_ingestor_trust_store",
            return_value=SimpleNamespace(content_digest="9" * 64),
        ), self.assertRaisesRegex(ValueError, "certificate|lineage"):
            validate_neural_prior_deployment_decision_artifact(
                json.dumps(relaxed, sort_keys=True, separators=(",", ":")),
                deployment_certificate_trust_store_path=(
                    "/etc/advar/deployment-authorities.json"
                ),
                deployment_policy_trust_store_path=(
                    "/etc/advar/deployment-policies.json"
                ),
                raw_ingestor_trust_store_path=(
                    "/etc/advar/raw-ingestors.json"
                ),
                training_target_source_trust_store_path=(
                    "/etc/advar/training-target-sources.json"
                ),
            )

    def _save_arrays(self, path: Path, arrays: dict[str, Any]) -> None:
        np.savez_compressed(path, **seal_forecast_run_arrays(arrays))

    def test_tensor_digest_accepts_scalar_tensors(self) -> None:
        value = torch.tensor(0.125, dtype=torch.float64)

        self.assertEqual(tensor_digest(value), tensor_digest(value.clone()))
        self.assertNotEqual(
            tensor_digest(value),
            tensor_digest(value.to(dtype=torch.float32)),
        )

    def test_artifact_digest_is_independent_of_numpy_memory_layout(self) -> None:
        contiguous = np.arange(24, dtype=np.float64).reshape(4, 6)
        fortran = np.asfortranarray(contiguous)

        self.assertEqual(
            run_artifact._forecast_run_artifact_digest(
                {"value": contiguous}
            ),
            run_artifact._forecast_run_artifact_digest({"value": fortran}),
        )

    def frames(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        coordinates = torch.arange(8, dtype=dtype)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        echo = -10.0 + 40.0 * torch.exp(
            -((y - 3.5).square() + (x - 3.5).square()) / 4.0
        )
        return torch.stack((echo, echo, echo))

    def test_v62_derivation_round_trip(self) -> None:
        frames = self.frames()
        input_run = self._current_provenance_run(frames)
        result, _ = self._variational_from_current_run(
            frames, None, input_run
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(
            loaded.run.analysis_input_derivation_artifact_json,
            input_run.analysis_input_derivation_artifact_json,
        )
        self.assertEqual(
            loaded.run.analysis_input_derivation_artifact_digest,
            input_run.analysis_input_derivation_artifact_digest,
        )
        self.assertEqual(loaded.forecast_run_digest, result.forecast_run_digest)

        incomplete = json.loads(
            input_run.analysis_input_derivation_artifact_json
        )
        incomplete.pop("qc_policy_digest")
        with self.assertRaisesRegex(ValueError, "payload digest mismatch"):
            replace(
                input_run,
                analysis_input_derivation_artifact_json=json.dumps(
                    incomplete,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                analysis_input_derivation_artifact_digest=json_digest(
                    incomplete
                ),
            ).validate_integrity()

    def test_restart_round_trip_preserves_m0_snapshot(self) -> None:
        frames = self.frames()
        config = NowcastConfig(
            growth_decay_minutes=120.0,
            max_dbz=60.0,
            min_publish_support=0.7,
        )
        result = nowcast(frames, config)
        sensitivity_config = SensitivityConfig(
            full_map_lead_minutes=(30,),
            tile_size=4,
            soft_fss_window=3,
        )
        in_memory = compute_sensitivity_snapshot(
            frames[-1],
            result,
            result.forecast_dbz.clone(),
            sensitivity_config=sensitivity_config,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        loaded.validate_issuance()
        self.assertEqual(loaded.run.config, config)
        self.assertEqual(
            loaded.state_metadata_digest,
            result.state_metadata_digest,
        )
        self.assertEqual(
            loaded.run.input_bundle_digest,
            result.run.input_bundle_digest,
        )
        self.assertEqual(
            loaded.forecast_run_digest,
            result.forecast_run_digest,
        )
        self.assertIsNone(loaded.run.analysis_config_json)
        self.assertEqual(loaded.forecast_dbz_digest, result.forecast_dbz_digest)
        self.assertEqual(loaded.valid_mask_digest, result.valid_mask_digest)
        self.assertIsNone(loaded.audit)
        torch.testing.assert_close(
            loaded.forecast_dbz, result.forecast_dbz, equal_nan=True
        )
        torch.testing.assert_close(
            loaded.state.echo_linear,
            result.state.echo_linear,
        )
        torch.testing.assert_close(
            loaded.metadata.source_support,
            result.metadata.source_support,
        )
        torch.testing.assert_close(
            loaded.metadata.path_verified_source_support,
            result.metadata.path_verified_source_support,
        )
        torch.testing.assert_close(
            loaded.metadata.verified_source_support,
            result.metadata.verified_source_support,
        )
        torch.testing.assert_close(
            loaded.metadata.local_motion_verified_support,
            result.metadata.local_motion_verified_support,
        )
        torch.testing.assert_close(
            loaded.metadata.local_growth_verified_support,
            result.metadata.local_growth_verified_support,
        )
        torch.testing.assert_close(
            loaded.metadata.local_dynamics_verified_support,
            result.metadata.local_dynamics_verified_support,
        )
        torch.testing.assert_close(
            loaded.metadata.observation_verified_source_support,
            result.metadata.observation_verified_source_support,
        )
        torch.testing.assert_close(
            loaded.metadata.background_verified_source_support,
            result.metadata.background_verified_source_support,
        )
        torch.testing.assert_close(
            loaded.forecast_path_verified_support,
            result.forecast_path_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_verified_support,
            result.forecast_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_local_motion_verified_support,
            result.forecast_local_motion_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_local_growth_verified_support,
            result.forecast_local_growth_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_local_dynamics_verified_support,
            result.forecast_local_dynamics_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_observation_verified_support,
            result.forecast_observation_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_background_verified_support,
            result.forecast_background_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_velocity_uncertainty_mps,
            result.forecast_velocity_uncertainty_mps,
        )
        torch.testing.assert_close(
            loaded.forecast_position_uncertainty_m,
            result.forecast_position_uncertainty_m,
        )
        torch.testing.assert_close(
            loaded.forecast_log_growth_uncertainty,
            result.forecast_log_growth_uncertainty,
        )
        torch.testing.assert_close(
            loaded.metadata.maximum_growth_saturation_excess,
            result.metadata.maximum_growth_saturation_excess,
        )
        torch.testing.assert_close(
            loaded.metadata.posterior_velocity_uncertainty_mps,
            result.metadata.posterior_velocity_uncertainty_mps,
            equal_nan=True,
        )
        torch.testing.assert_close(
            loaded.metadata.posterior_log_growth_uncertainty_per_step,
            result.metadata.posterior_log_growth_uncertainty_per_step,
            equal_nan=True,
        )
        torch.testing.assert_close(
            loaded.metadata.p1_velocity_saturation_uncertainty_mps,
            result.metadata.p1_velocity_saturation_uncertainty_mps,
            equal_nan=True,
        )
        torch.testing.assert_close(
            loaded.metadata.p1_log_growth_saturation_uncertainty_per_step,
            result.metadata.p1_log_growth_saturation_uncertainty_per_step,
            equal_nan=True,
        )
        torch.testing.assert_close(
            loaded.forecast_confidence,
            result.forecast_confidence,
        )
        torch.testing.assert_close(
            loaded.radar_anchored_valid_mask,
            result.radar_anchored_valid_mask,
        )
        torch.testing.assert_close(
            loaded.background_fallback_mask,
            result.background_fallback_mask,
        )
        for loaded_path, expected_path in (
            (
                loaded.metadata.observation_path,
                result.metadata.observation_path,
            ),
            (
                loaded.metadata.background_path,
                result.metadata.background_path,
            ),
        ):
            self.assertEqual(loaded_path.mode, expected_path.mode)
            self.assertEqual(loaded_path.pair_count, expected_path.pair_count)
            self.assertTrue(math.isnan(loaded_path.minimum_psr))
            self.assertEqual(loaded_path.conflict, expected_path.conflict)
            self.assertEqual(
                loaded_path.extrapolated,
                expected_path.extrapolated,
            )
            self.assertEqual(
                loaded_path.age_minutes,
                expected_path.age_minutes,
            )
        self.assertEqual(
            loaded.metadata.background_tendency_used,
            result.metadata.background_tendency_used,
        )
        self.assertEqual(
            loaded.metadata.background_state_support_fraction,
            result.metadata.background_state_support_fraction,
        )
        self.assertEqual(
            loaded.metadata.dynamics_source,
            DynamicsSource.P0_RECONSTRUCTION,
        )
        torch.testing.assert_close(
            loaded.metadata.minimum_phase_correlation_psr,
            result.metadata.minimum_phase_correlation_psr,
        )
        self.assertEqual(
            loaded.metadata.motion_pair_count,
            result.metadata.motion_pair_count,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_count,
            result.metadata.growth_pair_count,
        )
        self.assertEqual(
            loaded.metadata.motion_pair_selection,
            result.metadata.motion_pair_selection,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_selection,
            result.metadata.growth_pair_selection,
        )
        self.assertEqual(
            loaded.metadata.motion_pair_conflict,
            result.metadata.motion_pair_conflict,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_conflict,
            result.metadata.growth_pair_conflict,
        )
        for name in (
            "dynamics_source",
            "state_path_source",
            "state_path_mode",
            "state_path_pair_count",
            "state_path_minimum_psr",
            "state_path_conflict",
            "state_path_extrapolated",
            "state_path_age_minutes",
            "minimum_growth_overlap_support",
            "minimum_growth_overlap_area_km2",
        ):
            with self.subTest(name=name):
                loaded_value = getattr(loaded.metadata, name)
                expected = getattr(result.metadata, name)
                if isinstance(expected, float) and math.isnan(expected):
                    self.assertTrue(math.isnan(loaded_value))
                else:
                    self.assertEqual(loaded_value, expected)
        torch.testing.assert_close(
            loaded.run.latest_observation_mask,
            result.run.latest_observation_mask,
        )
        self.assertEqual(
            loaded.run.forecast_integrator_version,
            FORECAST_INTEGRATOR_VERSION,
        )

        reloaded = compute_sensitivity_snapshot_from_run(
            loaded,
            loaded.forecast_dbz.clone(),
            sensitivity_config=sensitivity_config,
        )
        torch.testing.assert_close(
            reloaded.context_features,
            in_memory.context_features,
        )
        torch.testing.assert_close(
            reloaded.forecast_scores,
            in_memory.forecast_scores,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.control_sensitivity,
            in_memory.control_sensitivity,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.forecast_sensitivity,
            in_memory.forecast_sensitivity,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.direct.maps,
            in_memory.direct.maps,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.direct.tile_norm,
            in_memory.direct.tile_norm,
            equal_nan=True,
        )
        self.assertEqual(reloaded.trust_score, in_memory.trust_score)

    def test_round_trip_preserves_long_pair_provenance(self) -> None:
        frames = self.frames()
        frames[1] = torch.nan
        result = nowcast(frames)

        self.assertEqual(
            result.metadata.motion_pair_selection,
            TendencyPairSelection.LONG,
        )
        self.assertEqual(
            result.metadata.growth_pair_selection,
            TendencyPairSelection.LONG,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "long-pair-run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(
            loaded.metadata.motion_pair_selection,
            TendencyPairSelection.LONG,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_selection,
            TendencyPairSelection.LONG,
        )
        self.assertEqual(loaded.metadata.tendency_pair_count, 1)
        self.assertFalse(loaded.metadata.motion_pair_conflict)
        self.assertFalse(loaded.metadata.growth_pair_conflict)

    def test_round_trip_preserves_motion_conditioned_growth_conflict(
        self,
    ) -> None:
        coordinates = torch.arange(64, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        first = -10.0 + 40.0 * torch.exp(
            -((y - 31.5).square() + (x - 31.5).square()) / 32.0
        )
        result = nowcast(
            torch.stack(
                (first, torch.roll(first, shifts=10, dims=1), first)
            )
        )

        self.assertTrue(result.metadata.motion_pair_conflict)
        self.assertTrue(result.metadata.growth_pair_conflict)
        self.assertEqual(
            result.metadata.motion_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertEqual(
            result.metadata.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pair-conflict-run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertTrue(loaded.metadata.motion_pair_conflict)
        self.assertTrue(loaded.metadata.growth_pair_conflict)
        self.assertEqual(
            loaded.metadata.motion_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        loaded.validate_issuance()
        snapshot = compute_sensitivity_snapshot_from_run(
            loaded,
            loaded.forecast_dbz.clone(),
            sensitivity_config=SensitivityConfig(
                metric_names=("log_echo_mse",),
                full_map_lead_minutes=(10,),
                tile_size=16,
            ),
        )
        context = dict(
            zip(snapshot.context_feature_names, snapshot.context_features)
        )
        self.assertEqual(float(context["motion_pair_conflict"]), 1.0)
        self.assertEqual(float(context["growth_pair_conflict"]), 1.0)
        self.assertEqual(
            float(context["motion_pair_selection_persistence"]),
            1.0,
        )
        self.assertEqual(
            float(context["growth_pair_selection_persistence"]),
            1.0,
        )
        self.assertEqual(float(context["phase_correlation_psr_available"]), 0.0)
        self.assertEqual(
            float(context["log1p_minimum_phase_correlation_psr"]),
            0.0,
        )

    def test_loader_materializes_each_archive_member_once(self) -> None:
        result = nowcast(self.frames())
        reads: Counter[str] = Counter()
        original_getitem = np.lib.npyio.NpzFile.__getitem__

        def counted_getitem(
            archive: np.lib.npyio.NpzFile,
            name: str,
        ) -> np.ndarray:
            reads[name] += 1
            return original_getitem(archive, name)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "single-materialization-run.npz"
            save_forecast_run(result, path)
            with patch.object(
                np.lib.npyio.NpzFile,
                "__getitem__",
                counted_getitem,
            ):
                loaded = load_forecast_run(path)

        loaded.validate_issuance()
        self.assertEqual(set(reads), run_artifact._CORE_ARRAY_NAMES)
        self.assertEqual(set(reads.values()), {1})

    def test_restart_m0_uses_embedded_latest_background(self) -> None:
        frames = self.frames()
        background = frames - 2.0
        result = nowcast(
            frames,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )
        expected = compute_sensitivity_snapshot(
            frames[-1],
            result,
            result.forecast_dbz.clone(),
            latest_background_dbz=background[-1],
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        torch.testing.assert_close(loaded.run.latest_frame_dbz, frames[-1])
        loaded_background = loaded.run.latest_background_dbz
        self.assertIsNotNone(loaded_background)
        assert loaded_background is not None
        torch.testing.assert_close(
            loaded_background,
            background[-1],
        )
        actual = compute_sensitivity_snapshot_from_run(
            loaded,
            loaded.forecast_dbz.clone(),
        )
        torch.testing.assert_close(
            actual.control_sensitivity,
            expected.control_sensitivity,
            equal_nan=True,
        )
        torch.testing.assert_close(
            actual.direct.maps,
            expected.direct.maps,
            equal_nan=True,
        )

    def test_round_trip_preserves_grid_time_contract(self) -> None:
        frames = self.frames()
        background = frames - 1.0
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=500.0,
            dy_m=750.0,
            projection="EPSG:5179",
            grid_hash="c" * 64,
            background_valid_times=(
                "2026-07-30T23:50:00Z",
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
            ),
        )
        result = nowcast(
            frames,
            background_frames_dbz=background,
            background_age_minutes=10.0,
            grid_time_contract=contract,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.grid_time_contract, contract)
        self.assertEqual(
            loaded.run.grid_time_contract_digest,
            contract.digest,
        )
        self.assertEqual(loaded.run.background_age_minutes, 10.0)
        self.assertEqual(loaded.forecast_run_digest, result.forecast_run_digest)
        torch.testing.assert_close(
            loaded.displacement_mps_yx,
            result.displacement_mps_yx,
        )
        torch.testing.assert_close(
            loaded.projected_velocity_mps_xy,
            result.projected_velocity_mps_xy,
        )
        torch.testing.assert_close(
            loaded.metadata.motion_disagreement_mps,
            result.metadata.motion_disagreement_mps,
            equal_nan=True,
        )
        snapshot = compute_sensitivity_snapshot_from_run(
            loaded,
            loaded.forecast_dbz.clone(),
            sensitivity_config=SensitivityConfig(
                metric_names=("log_echo_mse",),
                full_map_lead_minutes=(10,),
                tile_size=4,
            ),
        )
        context = dict(
            zip(snapshot.context_feature_names, snapshot.context_features)
        )
        self.assertEqual(snapshot.grid_time_contract_digest, contract.digest)
        self.assertEqual(context["projected_velocity_available"], 1.0)
        self.assertEqual(
            context["motion_disagreement_mps_available"], 1.0
        )
        torch.testing.assert_close(
            context["motion_disagreement_mps"],
            loaded.metadata.motion_disagreement_mps,
        )
        self.assertEqual(context["area_weighted_echo_available"], 1.0)
        self.assertGreater(
            context["log1p_linear_reflectivity_integral_km2"], 0.0
        )
        torch.testing.assert_close(
            torch.stack(
                (
                    context["projected_velocity_x_mps"],
                    context["projected_velocity_y_mps"],
                )
            ),
            loaded.projected_velocity_mps_xy,
        )
        torch.testing.assert_close(
            context["projected_speed_mps"],
            torch.linalg.vector_norm(loaded.projected_velocity_mps_xy),
        )

    def test_round_trip_preserves_scientific_projected_grid_identity(
        self,
    ) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=500.0,
            dy_m=750.0,
            projection="EPSG:5179",
            grid_hash="c" * 64,
            pixel_to_projected_matrix_m=((0.0, -750.0), (500.0, 0.0)),
            spatial_grid_contract="radar-spatial-grid-identity-v6",
            grid_shape_yx=tuple(self.frames().shape[-2:]),
            projected_crs_digest=radar_projected_crs_semantic_digest(
                "EPSG:5179"
            ),
            metric_domain_digest=CURRENT_RADAR_METRIC_DOMAIN.digest,
            metric_domain_evidence_digest=(
                CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
            ),
            cell_center_origin_xy_m=(1_000_000.0, 2_000_000.0),
            grid_coordinate_dtype=RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
            cell_center_convention=(
                RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION
            ),
        )
        result = nowcast(self.frames(), grid_time_contract=contract)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.grid_time_contract, contract)
        self.assertEqual(
            loaded.run.grid_time_contract.spatial_grid_digest,
            contract.spatial_grid_digest,
        )
        self.assertEqual(loaded.forecast_run_digest, result.forecast_run_digest)

    def test_load_rejects_tampered_grid_time_contract(self) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="c" * 64,
        )
        result = nowcast(self.frames(), grid_time_contract=contract)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            contract_value = arrays["grid_time_contract_json"].item()
            arrays["grid_time_contract_json"] = np.asarray(
                contract_value.replace(
                    '"projection":"EPSG:5179"',
                    '"projection":"EPSG:3857"',
                )
            )
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "grid/time contract digest mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_tampered_physical_displacement(self) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=500.0,
            projection="EPSG:5179",
            grid_hash="d" * 64,
        )
        result = nowcast(self.frames(), grid_time_contract=contract)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["displacement_mps_yx"] += 1.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "displacement_mps_yx disagrees",
            ):
                load_forecast_run(path)

    def test_input_bundle_digest_covers_all_three_input_times(self) -> None:
        frames = self.frames()
        accepted = torch.ones_like(frames, dtype=torch.bool)
        background = frames - 1.0
        base = nowcast(
            frames,
            qc_mask=accepted,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        earlier_frame = frames.clone()
        earlier_frame[0, 2, 2] += 0.25
        earlier_mask = accepted.clone()
        earlier_mask[0, 2, 2] = False
        earlier_background = background.clone()
        earlier_background[0, 2, 2] += 0.25
        variants = (
            nowcast(
                earlier_frame,
                qc_mask=accepted,
                background_frames_dbz=background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=earlier_mask,
                background_frames_dbz=background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=accepted,
                background_frames_dbz=earlier_background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=accepted,
                background_frames_dbz=background,
                background_age_minutes=20.0,
            ),
        )

        for variant in variants:
            self.assertNotEqual(
                variant.run.input_bundle_digest,
                base.run.input_bundle_digest,
            )
            self.assertNotEqual(
                variant.forecast_run_digest,
                base.forecast_run_digest,
            )

    def test_neural_prior_lineage_round_trips_with_the_run(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        prior = runner.infer(
            frames,
            input_run=input_run,
            role="candidate",
        )
        result, _ = self._variational_from_current_run(
            frames, prior, input_run
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.neural_prior_digest, prior.neural_prior_digest)
        self.assertEqual(loaded.run.prior_application_digest, prior.application_digest)
        self.assertEqual(loaded.run.prior_model_contract_digest, "2" * 64)
        self.assertEqual(loaded.run.prior_feature_schema_digest, "3" * 64)
        self.assertEqual(loaded.run.prior_training_manifest_digest, "4" * 64)
        self.assertEqual(
            loaded.run.prior_inference_evidence_digest,
            prior.inference_evidence.evidence_digest,
        )
        self.assertEqual(loaded.run.prior_dependency, "radar_dependent")
        self.assertEqual(loaded.run.prior_role, "candidate")
        self.assertEqual(loaded.forecast_run_digest, result.forecast_run_digest)
        _, frozen = prepare_analysis(frames, neural_prior=prior)
        active_prior = (
            prior.state_valid_mask
            & (prior.state_support_probability >= 0.5)
            & frozen.initial_support_mask
        )
        torch.testing.assert_close(
            frozen.initial_background_dbz.masked_select(active_prior),
            prior.state_background_dbz.masked_select(active_prior),
        )

    def test_current_run_rejects_deployment_selection_lineage(self) -> None:
        frames = self.frames()
        input_run = self._current_provenance_run(frames)
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0, -1],
            example_frames=frames,
            example_qc_valid_mask=torch.ones_like(frames, dtype=torch.bool),
            example_quality_weight=torch.ones_like(frames),
            example_observation_std_dbz=torch.full_like(frames, 2.0),
            example_source_available_mask=torch.ones_like(
                frames, dtype=torch.bool
            ),
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        selection = self._deployment_selection(runner, input_run, frames)
        prior = runner._infer_deployed(
            frames,
            input_run=input_run,
            deployment_selection=selection,
            qc_valid_mask=torch.ones_like(frames, dtype=torch.bool),
            quality_weight=torch.ones_like(frames),
            observation_std_dbz=torch.full_like(frames, 2.0),
            source_available_mask=torch.ones_like(frames, dtype=torch.bool),
        )

        with self.assertRaisesRegex(
            OperationalDeploymentUnsupportedError,
            "cannot claim operational neural-prior deployment lineage",
        ):
            self._variational_from_current_run(frames, prior, input_run)

    def test_v49_candidate_run_loads_as_deployment_audit(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        prior = runner.infer(frames, input_run=input_run, role="candidate")
        result, _ = self._legacy_deployment_fixture_from_current_run(
            frames, prior, input_run
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-v49.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name
                    not in {
                        "prior_promotion_evidence_digest",
                        "prior_regime_classification_evidence_digest",
                        "prior_deployment_selection_digest",
                        "prior_deployment_fallback_reason",
                        "prior_deployment_lineage_contract",
                    }
                }
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v49"
            )
            self._save_arrays(path, arrays)

            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.prior_role, "candidate")
        self.assertEqual(
            loaded.run.prior_deployment_lineage_contract,
            "neural-prior-deployment-lineage-v0-audit",
        )
        self.assertIsNone(loaded.run.prior_deployment_selection_digest)

    def test_v50_deployment_lineage_loads_as_policy_audit(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0, -1],
            example_frames=frames,
            example_qc_valid_mask=torch.ones_like(frames, dtype=torch.bool),
            example_quality_weight=torch.ones_like(frames),
            example_observation_std_dbz=torch.full_like(frames, 2.0),
            example_source_available_mask=torch.ones_like(
                frames, dtype=torch.bool
            ),
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        input_run = self._current_provenance_run(frames)
        selection = self._deployment_selection(runner, input_run, frames)
        prior = runner._infer_deployed(
            frames,
            input_run=input_run,
            deployment_selection=selection,
            qc_valid_mask=torch.ones_like(frames, dtype=torch.bool),
            quality_weight=torch.ones_like(frames),
            observation_std_dbz=torch.full_like(frames, 2.0),
            source_available_mask=torch.ones_like(frames, dtype=torch.bool),
        )
        result, _ = self._legacy_deployment_fixture_from_current_run(
            frames, prior, input_run
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-v50.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name
                    not in {
                        "prior_deployment_policy_digest",
                        "prior_deployment_policy_trust_store_digest",
                        "prior_deployment_decision_artifact_json",
                        "prior_deployment_decision_artifact_digest",
                    }
                }
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v50"
            )
            self._save_arrays(path, arrays)

            loaded = load_forecast_run(path)

        self.assertEqual(
            loaded.run.prior_deployment_lineage_contract,
            "neural-prior-deployment-lineage-v1-audit",
        )
        self.assertIsNone(loaded.run.prior_deployment_policy_digest)
        self.assertIsNone(
            loaded.run.prior_deployment_policy_trust_store_digest
        )

    def test_v51_deployment_lineage_loads_as_decision_audit(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0, -1],
            example_frames=frames,
            example_qc_valid_mask=torch.ones_like(frames, dtype=torch.bool),
            example_quality_weight=torch.ones_like(frames),
            example_observation_std_dbz=torch.full_like(frames, 2.0),
            example_source_available_mask=torch.ones_like(
                frames, dtype=torch.bool
            ),
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        input_run = self._current_provenance_run(frames)
        selection = self._deployment_selection(runner, input_run, frames)
        prior = runner._infer_deployed(
            frames,
            input_run=input_run,
            deployment_selection=selection,
            qc_valid_mask=torch.ones_like(frames, dtype=torch.bool),
            quality_weight=torch.ones_like(frames),
            observation_std_dbz=torch.full_like(frames, 2.0),
            source_available_mask=torch.ones_like(frames, dtype=torch.bool),
        )
        result, _ = self._legacy_deployment_fixture_from_current_run(
            frames, prior, input_run
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-v51.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name
                    not in {
                        "prior_deployment_decision_artifact_json",
                        "prior_deployment_decision_artifact_digest",
                    }
                }
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v51"
            )
            self._save_arrays(path, arrays)

            loaded = load_forecast_run(path)

        self.assertEqual(
            loaded.run.prior_deployment_lineage_contract,
            "neural-prior-deployment-lineage-v2-audit",
        )
        self.assertIsNotNone(loaded.run.prior_deployment_policy_digest)
        self.assertIsNone(
            loaded.run.prior_deployment_decision_artifact_digest
        )

    def test_v53_cannot_omit_deployment_lineage_contract(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "prior_deployment_lineage_contract"
                }
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "lacks deployment lineage contract",
            ):
                load_forecast_run(path)

    def test_v52_deployment_geometry_loads_as_audit_only(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0, -1],
            example_frames=frames,
            example_qc_valid_mask=torch.ones_like(frames, dtype=torch.bool),
            example_quality_weight=torch.ones_like(frames),
            example_observation_std_dbz=torch.full_like(frames, 2.0),
            example_source_available_mask=torch.ones_like(
                frames, dtype=torch.bool
            ),
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        input_run = self._current_provenance_run(frames)
        selection = self._deployment_selection(runner, input_run, frames)
        prior = runner._infer_deployed(
            frames,
            input_run=input_run,
            deployment_selection=selection,
            qc_valid_mask=torch.ones_like(frames, dtype=torch.bool),
            quality_weight=torch.ones_like(frames),
            observation_std_dbz=torch.full_like(frames, 2.0),
            source_available_mask=torch.ones_like(frames, dtype=torch.bool),
        )
        result, _ = self._legacy_deployment_fixture_from_current_run(
            frames, prior, input_run
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-v52.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            v69_arrays = dict(arrays)
            v69_arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v69"
            )
            v69_path = Path(temporary) / "legacy-v69.npz"
            self._save_arrays(v69_path, v69_arrays)
            loaded_v69 = load_forecast_run(v69_path)
            v66_arrays = dict(arrays)
            v66_arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v66"
            )
            v66_path = Path(temporary) / "legacy-v66.npz"
            self._save_arrays(v66_path, v66_arrays)
            loaded_v66 = load_forecast_run(v66_path)
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v52"
            )
            self._save_arrays(path, arrays)

            loaded = load_forecast_run(path)

        self.assertEqual(
            loaded.run.prior_deployment_lineage_contract,
            "neural-prior-deployment-lineage-v3-audit",
        )
        self.assertEqual(
            loaded_v69.run.prior_deployment_lineage_contract,
            "neural-prior-deployment-lineage-v19-audit",
        )
        self.assertEqual(
            loaded_v66.run.prior_deployment_lineage_contract,
            "neural-prior-deployment-lineage-v17-audit",
        )

    def test_neural_prior_lineage_is_all_or_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "neural-prior run lineage"):
            ForecastRunContract.from_inputs(
                NowcastConfig(),
                self.frames(),
                torch.ones_like(self.frames(), dtype=torch.bool),
                None,
                neural_prior_digest="1" * 64,
            )

    def test_v42_artifact_migrates_without_prior_lineage(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-v42.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name not in {
                        "neural_prior_digest",
                        "prior_application_digest",
                        "prior_model_contract_digest",
                        "prior_feature_schema_digest",
                        "prior_training_manifest_digest",
                        "prior_role",
                    }
                }
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v42"
            )
            self._save_arrays(path, arrays)

            loaded = load_forecast_run(path)

        self.assertIsNone(loaded.run.neural_prior_digest)
        torch.testing.assert_close(loaded.forecast_dbz, result.forecast_dbz)

    def test_v44_artifact_preserves_existing_prior_lineage(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            state_contract=self._state_contract(),
            probability_contract=self._probability_contract(),
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        prior = runner.infer(frames, input_run=input_run, role="candidate")
        result, _ = variational_nowcast(frames, neural_prior=prior)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-v44.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name not in {
                        "prior_inference_evidence_digest",
                        "prior_inference_algorithm_digest",
                        "prior_numerical_runtime_digest",
                        "prior_dependency",
                    }
                }
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v44"
            )
            self._save_arrays(path, arrays)

            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.neural_prior_digest, prior.neural_prior_digest)
        self.assertEqual(
            loaded.run.prior_lineage_contract,
            "neural-prior-run-lineage-v1-audit",
        )
        self.assertIsNone(loaded.run.prior_inference_evidence_digest)

    def test_load_rejects_tampered_state(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["state_echo_linear"][0, 0] += 1.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast does not close against the issued state",
            ):
                load_forecast_run(path)

    def test_load_rejects_resigned_forecast_verified_support(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "forecast_run_artifact_digest"
                }
            arrays["forecast_verified_support"][0, 0, 0] = 0.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast verified support mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_resigned_local_dynamics_support(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "forecast_run_artifact_digest"
                }
            arrays["forecast_local_dynamics_verified_support"][0, 0, 0] = 0.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast local dynamics verified support mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_resigned_local_motion_support(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "forecast_run_artifact_digest"
                }
            arrays["forecast_local_motion_verified_support"][0, 0, 0] = 0.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast local motion verified support mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_resigned_forecast_confidence(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "forecast_run_artifact_digest"
                }
            arrays["forecast_confidence"][0, 0, 0] = 0.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast confidence mismatch",
            ):
                load_forecast_run(path)

    def test_save_rejects_mutated_metadata(self) -> None:
        result = nowcast(self.frames())
        result.metadata.source_support[0, 0] = 0.0
        result.metadata.path_verified_source_support[0, 0] = 0.0
        result.metadata.verified_source_support[0, 0] = 0.0
        result.metadata.observation_verified_source_support[0, 0] = 0.0

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            with self.assertRaisesRegex(
                ValueError,
                "source support and actual contributions",
            ):
                save_forecast_run(result, path)
            self.assertFalse(path.exists())

    def test_load_rejects_tampered_phase_correlation_psr(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["minimum_phase_correlation_psr"] += 1.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "state or metadata",
            ):
                load_forecast_run(path)

    def test_load_rejects_pair_selection_count_mismatch(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["motion_pair_selection"] = np.asarray("PERSISTENCE")
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "motion pair count and selection disagree",
            ):
                load_forecast_run(path)

    def test_load_rejects_incorrect_pair_union_count(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["motion_pair_count"] = np.asarray(1)
            arrays["growth_pair_count"] = np.asarray(1)
            arrays["motion_pair_selection"] = np.asarray("EARLIER")
            arrays["growth_pair_selection"] = np.asarray("RECENT")
            arrays["tendency_pair_count"] = np.asarray(1)
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "tendency_pair_count is inconsistent",
            ):
                load_forecast_run(path)

    def test_load_rejects_malformed_phase_correlation_psr_schema(self) -> None:
        result = nowcast(self.frames())
        malformed_values = (
            np.asarray(1, dtype=np.int64),
            np.asarray([1.0], dtype=np.float64),
            np.asarray(np.inf, dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                original: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }

            for malformed in malformed_values:
                with self.subTest(value=malformed):
                    arrays = dict(original)
                    arrays["minimum_phase_correlation_psr"] = malformed
                    self._save_arrays(path, arrays)

                    with self.assertRaisesRegex(
                        ValueError,
                        "minimum_phase_correlation_psr must be",
                    ):
                        load_forecast_run(path)

    def test_load_rejects_tampered_background_tendency_provenance(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["background_tendency_used"] = np.asarray(True)
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "background tendency provenance mismatch",
            ):
                load_forecast_run(path)

    def test_load_uses_central_metadata_semantics(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                original: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }

            for name, value, message in (
                (
                    "background_used",
                    np.asarray(True),
                    "background usage provenance",
                ),
                (
                    "minimum_growth_overlap_support",
                    np.asarray(np.nan),
                    "used growth pairs",
                ),
            ):
                with self.subTest(name=name):
                    arrays = dict(original)
                    arrays[name] = value
                    self._save_arrays(path, arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        load_forecast_run(path)

    def test_save_rejects_mutated_analysis_lineage(self) -> None:
        result = nowcast(self.frames())
        changed_run = replace(
            result.run,
            analysis_config_json="{}",
            analysis_config_digest="0" * 64,
            analysis_input_digest="1" * 64,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            with self.assertRaisesRegex(
                ValueError,
                "analysis config digest mismatch",
            ):
                save_forecast_run(
                    replace(result, run=changed_run),
                    path,
                )
            self.assertFalse(path.exists())

    def test_save_rejects_incompatible_forecast_integrator(self) -> None:
        result = nowcast(self.frames())
        changed_run = replace(
            result.run,
            forecast_integrator_version="incompatible-integrator",
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            with self.assertRaisesRegex(
                ValueError,
                "forecast integrator version",
            ):
                save_forecast_run(replace(result, run=changed_run), path)
            self.assertFalse(path.exists())

    def test_load_rejects_config_json_that_disagrees_with_digest(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["nowcast_config_json"] = np.asarray(
                '{"interval_minutes":5,"horizon_minutes":180,'
                '"min_dbz":-10.0,"max_dbz":70.0,'
                '"echo_threshold_dbz":5.0,"recent_weight":0.6666666666666666,'
                '"pair_echo_dilation_px":3,"max_displacement_px":20.0,'
                '"max_log_growth_per_step":0.30010459245033816,'
                '"growth_decay_minutes":60.0,"min_publish_support":0.95,'
                '"epsilon":1e-06}'
            )
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "config digest mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_tampered_input_bundle_identity(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["input_bundle_digest"] = np.asarray("0" * 64)
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(ValueError, "run identity"):
                load_forecast_run(path)

    def test_load_rejects_unknown_archive_member(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["unexpected_payload"] = np.asarray(1)
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(ValueError, "unknown members"):
                load_forecast_run(path)

    def test_load_applies_archive_resource_limits_before_materialization(
        self,
    ) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                member_count = len(archive.files)

            cases = (
                ({"maximum_member_count": member_count - 1}, "too many"),
                ({"maximum_member_bytes": 1}, "member is too large"),
                (
                    {"maximum_total_expanded_bytes": 1},
                    "expands beyond",
                ),
            )
            for limits, message in cases:
                with self.subTest(limits=limits):
                    with self.assertRaisesRegex(ValueError, message):
                        load_forecast_run(path, **limits)

    def test_load_rejects_declared_array_size_before_materialization(
        self,
    ) -> None:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {
                "descr": np.dtype(np.float64).str,
                "fortran_order": False,
                "shape": (1_000_000_000,),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hostile.npz"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("forecast_dbz.npy", header.getvalue())

            with self.assertRaisesRegex(
                ValueError,
                "declares too much array data",
            ):
                load_forecast_run(path, maximum_member_bytes=1024)

    def test_load_rejects_object_array_header_before_materialization(
        self,
    ) -> None:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {
                "descr": np.dtype(object).str,
                "fortran_order": False,
                "shape": (1,),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "object.npz"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("forecast_dbz.npy", header.getvalue())

            with self.assertRaisesRegex(ValueError, "non-object dtypes"):
                load_forecast_run(path)

    def test_load_uses_the_same_file_after_resource_preflight(self) -> None:
        original_result = nowcast(self.frames())
        replacement_frames = self.frames() + 1.0
        replacement_result = nowcast(replacement_frames)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "run.npz"
            replacement = directory / "replacement.npz"
            save_forecast_run(original_result, path)
            save_forecast_run(replacement_result, replacement)
            real_preflight = run_artifact._preflight_archive

            def preflight_then_replace(
                source: Any,
                **limits: Any,
            ) -> None:
                real_preflight(source, **limits)
                replacement.replace(path)

            with patch(
                "advar.run_artifact._preflight_archive",
                side_effect=preflight_then_replace,
            ):
                loaded = load_forecast_run(path)

        self.assertEqual(
            loaded.forecast_run_digest,
            original_result.forecast_run_digest,
        )


if __name__ == "__main__":
    unittest.main()
