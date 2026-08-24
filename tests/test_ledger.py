from concurrent.futures import ThreadPoolExecutor
import base64
from copy import copy
from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import advar.ledger as ledger_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar._digest import json_digest, tensor_digest  # noqa: E402
from advar.ledger import (  # noqa: E402
    EpisodeLedger,
    LegacyProspectiveInterventionDecisionAudit,
    LegacyRealizedInterventionReceiptAudit,
    ModelContract,
    SensitivityEpisode,
)
from advar.intervention import (  # noqa: E402
    InterventionActionGenerator,
    InterventionInputContext,
    OperatorActionApproval,
    OperatorOverrideAction,
    QcMaskAction,
    ProspectiveInterventionDecision,
    ReusableInterventionPolicyEvidence,
    RealizedInterventionReceipt,
    RealizedObservationIntervention,
    _validate_action_safety,
    RetrospectiveCounterfactualReplay,
)
from advar.promotion import (  # noqa: E402
    LegacyNeuralPriorCandidateManifestAuditV3,
    LegacyNeuralPriorCandidateManifestAuditV4,
    LegacyNeuralPriorHoldoutPlanAudit,
    LegacyNeuralPriorHoldoutPlanCase,
    LegacyNeuralPriorHoldoutPlanV2Audit,
    LegacyNeuralPriorHoldoutPlanV2Case,
    LegacyNeuralPriorHoldoutPlanV3Case,
    LegacyNeuralPriorHoldoutPlanV3Audit,
    LegacyNeuralPriorHoldoutPlanV4Audit,
    LegacyNeuralPriorHoldoutPlanV5Audit,
    LegacyNeuralPriorHoldoutPlanV6Audit,
    LegacyNeuralPriorPromotionEvidenceAuditV3,
    LegacyNeuralPriorPromotionEvidenceAuditV5,
    LegacyNeuralPriorPromotionEvidenceAuditV6,
    LegacyNeuralPriorPromotionEvidenceAuditV7,
    LegacyNeuralPriorPromotionEvidenceAuditV8,
    NeuralPriorPromotionEvidence,
    NeuralPriorCandidateManifest,
    NeuralPriorHoldoutCase,
    NeuralPriorHoldoutPlan,
    NeuralPriorHoldoutPlanCase,
    NeuralPriorHoldoutPlanPolicy,
    NeuralPriorInputPlan,
    NeuralPriorStateCalibrationPlan,
    PriorUncertaintyTargetPlan,
    NeuralPriorPromotionPolicy,
    PromotionMetricScale,
    compute_neural_prior_promotion,
)
import advar.promotion as promotion_module  # noqa: E402
from advar.nowcast import (  # noqa: E402
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    ForecastRunContract,
    NowcastConfig,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    RadarGridTimeContract,
    nowcast,
    operational_runtime_profile_digest,
    radar_projected_crs_semantic_digest,
)
from advar.calibration import (  # noqa: E402
    CalibrationMetric,
    CalibrationRegime,
    OperationalCalibrationManifest,
    OperationalDataIdentity,
    algorithm_bundle_digest,
)
from advar.sensitivity import (  # noqa: E402
    LearningApprovalEvidence,
    MosaicObservationSourceRegistry,
    OBSERVATION_CENSOR_STATE_ALGORITHM_V1_DIGEST,
    OBSERVATION_DETECTION_LIMIT_ALGORITHM_V1_DIGEST,
    OBSERVATION_DETECTION_LIMIT_ALGORITHM_V2_DIGEST,
    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V4_DIGEST,
    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V5_DIGEST,
    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V6_DIGEST,
    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V7_DIGEST,
    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V8_DIGEST,
    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V10_DIGEST,
    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V11_DIGEST,
    OBSERVATION_MASK_DERIVATION_ALGORITHM_V3_DIGEST,
    OBSERVATION_MASK_DERIVATION_ALGORITHM_V4_DIGEST,
    OBSERVATION_MASK_DERIVATION_ALGORITHM_V5_DIGEST,
    OBSERVATION_MASK_DERIVATION_ALGORITHM_V6_DIGEST,
    OBSERVATION_MASK_DERIVATION_ALGORITHM_V7_DIGEST,
    OBSERVATION_MASK_DERIVATION_ALGORITHM_V9_DIGEST,
    OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST,
    OBSERVATION_REPORT_KIND_ALGORITHM_V1_DIGEST,
    OBSERVATION_SOURCE_SELECTION_ALGORITHM_V1_DIGEST,
    OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V1_DIGEST,
    OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V3_DIGEST,
    OBSERVATION_TEMPORAL_ERROR_ALGORITHM_V1_DIGEST,
    OBSERVATION_TEMPORAL_QUALITY_DECAY_ALGORITHM_V1_DIGEST,
    ObservationRadarSource,
    RadarObservationGeometryContract,
    SensitivityConfig,
    VerificationObservationErrorPlan,
    compute_sensitivity_snapshot,
)


SCHEMA_ONE_CONTEXT_FEATURE_NAMES = (
    "motion_dy",
    "motion_dx",
    "motion_speed",
    "log_growth",
    "motion_disagreement",
    "growth_disagreement",
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_echo_mass",
)
SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES = (
    "motion_dy",
    "motion_dx",
    "motion_speed",
    "log_growth",
    "motion_disagreement",
    "growth_disagreement",
    "tendency_pair_count",
    "tendency_source_observation",
    "tendency_source_background",
    "current_state_support_fraction",
    "background_contribution_fraction",
    "latest_observation_coverage",
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_integrated_echo",
)
SCHEMA_FIVE_CONTEXT_FEATURE_NAMES = (
    "motion_dy",
    "motion_dx",
    "motion_speed",
    "log_growth",
    "motion_disagreement",
    "growth_disagreement",
    "motion_pair_conflict",
    "growth_pair_conflict",
    "tendency_pair_count",
    "tendency_source_observation",
    "tendency_source_background",
    "current_state_support_fraction",
    "background_contribution_fraction",
    "latest_observation_coverage",
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_integrated_echo",
)
SCHEMA_SIX_CONTEXT_FEATURE_NAMES = (
    *SCHEMA_FIVE_CONTEXT_FEATURE_NAMES,
    "motion_pair_selection_none",
    "motion_pair_selection_single",
    "motion_pair_selection_long",
    "motion_pair_selection_blended",
    "motion_pair_selection_earlier",
    "motion_pair_selection_recent",
    "motion_pair_selection_persistence",
    "growth_pair_selection_none",
    "growth_pair_selection_single",
    "growth_pair_selection_long",
    "growth_pair_selection_blended",
    "growth_pair_selection_earlier",
    "growth_pair_selection_recent",
    "growth_pair_selection_persistence",
)
SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES = (
    *SCHEMA_SIX_CONTEXT_FEATURE_NAMES,
    "phase_correlation_psr_available",
    "log1p_minimum_phase_correlation_psr",
)
SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES = (
    *SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES,
    "projected_velocity_available",
    "projected_velocity_x_mps",
    "projected_velocity_y_mps",
    "projected_speed_mps",
)
SCHEMA_NINE_CONTEXT_FEATURE_NAMES = (
    *SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES,
    "motion_disagreement_mps_available",
    "motion_disagreement_mps",
)
SCHEMA_TEN_CONTEXT_FEATURE_NAMES = (
    *SCHEMA_NINE_CONTEXT_FEATURE_NAMES,
    "area_weighted_echo_available",
    "log1p_linear_reflectivity_integral_km2",
)

VERIFICATION_LINEAGE_MANIFEST_FIELDS = (
    "verification_contract",
    "verification_bundle_digest",
    "verification_lineage_complete",
    "verification_valid_times",
    "verification_grid_contract_digest",
    "verification_radar_product_digest",
    "verification_qc_pipeline_digest",
)


def _drop_verification_lineage(manifest: dict[str, object]) -> None:
    for name in VERIFICATION_LINEAGE_MANIFEST_FIELDS:
        manifest.pop(name)
    manifest.pop("tile_shape_yx", None)


def _contract(snapshot=None) -> ModelContract:
    return ModelContract(
        model_commit="model-v1",
        residual_contract_version="residual-v1",
        forecast_metric_version="issued-domain-metrics-v2",
        observation_contract_version="direct-latest-dbz-active-set-v2",
        forecast_integrator_version="integrator-v1",
        grid_geometry_version="grid-v1",
        radar_qc_version="qc-v1",
        nowcast_config_digest=(
            "a" * 64 if snapshot is None else snapshot.nowcast_config_digest
        ),
        sensitivity_config_digest=(
            "b" * 64 if snapshot is None else snapshot.sensitivity_config_digest
        ),
        grid_time_contract_digest=(
            None if snapshot is None else snapshot.grid_time_contract_digest
        ),
    )


def _computed_snapshot():
    config = NowcastConfig()
    frames = torch.full((3, 2, 2), 20.0, dtype=torch.float64)
    background = frames - 0.5
    result = nowcast(
        frames,
        config,
        background_frames_dbz=background,
        background_age_minutes=10.0,
    )
    verification = frames.new_full((config.forecast_steps, 2, 2), 20.0)
    return compute_sensitivity_snapshot(
        frames[-1],
        result,
        verification,
        latest_background_dbz=background[-1],
        sensitivity_config=SensitivityConfig(
            metric_names=("log_echo_mse",),
        ),
    )


class ModelContractTests(unittest.TestCase):
    def test_digest_changes_when_any_contract_field_changes(self) -> None:
        contract = _contract()
        self.assertRegex(contract.digest, r"^[0-9a-f]{64}$")
        self.assertEqual(contract.digest, _contract().digest)

        for field in fields(contract):
            with self.subTest(field=field.name):
                value = (
                    "c" * 64
                    if field.name == "grid_time_contract_digest"
                    else f"{getattr(contract, field.name)}-changed"
                )
                changed = replace(
                    contract,
                    **{field.name: value},
                )
                self.assertNotEqual(contract.digest, changed.digest)

    def test_episode_rejects_config_digest_mismatch(self) -> None:
        snapshot = _computed_snapshot()
        contract = replace(
            _contract(snapshot),
            nowcast_config_digest="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "config digests"):
            SensitivityEpisode(
                episode_id="mismatched-config",
                issue_time="2026-07-26T05:00:00+00:00",
                radar_id="KTLX",
                contract=contract,
                snapshot=snapshot,
            )

    def test_episode_rejects_grid_digest_mismatch(self) -> None:
        snapshot = replace(
            _computed_snapshot(),
            grid_time_contract_digest="c" * 64,
        )
        with self.assertRaisesRegex(ValueError, "grid digest"):
            SensitivityEpisode(
                episode_id="mismatched-grid",
                issue_time="2026-07-26T05:00:00+00:00",
                radar_id="KTLX",
                contract=_contract(snapshot),
                snapshot=replace(snapshot, grid_time_contract_digest="d" * 64),
            )


class _AddOneAction(torch.nn.Module):
    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones_like(context[0]), torch.tensor(True, device=context.device)


class _AddTwoAction(torch.nn.Module):
    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.ones_like(context[0]) * 2.0,
            torch.tensor(True, device=context.device),
        )


class _NonpositiveOnlyAction(torch.nn.Module):
    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones_like(context[0]), torch.mean(context[0]) <= 0.0


class _RejectAllQcAction(torch.nn.Module):
    def forward(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros_like(context[0], dtype=torch.bool),
            torch.zeros_like(context[0]),
            torch.tensor(True, device=context.device),
        )


class _DeweightOnlyQcAction(torch.nn.Module):
    def forward(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            context[1].to(torch.bool),
            context[2] * 0.5,
            torch.tensor(True, device=context.device),
        )


class _UpweightQcAction(torch.nn.Module):
    def forward(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            context[1].to(torch.bool),
            torch.ones_like(context[2]),
            torch.tensor(True, device=context.device),
        )


class _ValidOnlyAddAction(torch.nn.Module):
    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return context[1], torch.tensor(True, device=context.device)


class _SinglePixelOverrideAction(torch.nn.Module):
    def forward(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        replacement = context[0].clone()
        replacement[0, 0, 0] = 1.0
        override = torch.zeros_like(context[0], dtype=torch.bool)
        override[0, 0, 0] = True
        return replacement, override, torch.tensor(True, device=context.device)


def _prospective_run_and_context(
    frames: torch.Tensor,
    masks: torch.Tensor,
    *,
    valid_times: tuple[str, str, str] = (
        "2026-08-08T00:00:00Z",
        "2026-08-08T00:10:00Z",
        "2026-08-08T00:20:00Z",
    ),
    input_available_time: str = "2026-08-08T00:21:00Z",
    decision_deadline: str = "2099-08-08T00:30:00Z",
    publication_time: str = "2099-08-08T01:00:00Z",
    quality_weight: torch.Tensor | None = None,
    observation_std_dbz: torch.Tensor | None = None,
    applicability_mask: torch.Tensor | None = None,
    analysis_config_json: str | None = None,
) -> tuple[ForecastRunContract, InterventionInputContext, str, str]:
    config = NowcastConfig()
    grid = RadarGridTimeContract(
        valid_times=valid_times,
        dx_m=1000.0,
        dy_m=1000.0,
        projection="EPSG:5179",
        grid_hash="1" * 64,
        spatial_grid_contract="radar-spatial-grid-identity-v5",
        grid_shape_yx=(int(frames.shape[-2]), int(frames.shape[-1])),
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
    identity = OperationalDataIdentity(
        radar_class="test-radar",
        qc_pipeline_digest="2" * 64,
        observation_error_model_digest="3" * 64,
        background_model_digest="4" * 64,
        radar_product_digest="5" * 64,
        background_cycle_rule_digest="6" * 64,
        mask_policy_digest="7" * 64,
    )
    manifest = OperationalCalibrationManifest(
        calibration_id="prospective-test",
        profile_kind="p0",
        expected_runtime_profile_digest=operational_runtime_profile_digest(
            config,
            grid,
        ),
        expected_algorithm_bundle_digest=algorithm_bundle_digest(),
        calibration_dataset_digest="8" * 64,
        validation_dataset_digest="9" * 64,
        data_identity=identity,
        training_period=("2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        validation_period=("2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        validation_case_count=1,
        validation_regimes=(CalibrationRegime("all", 1),),
        validation_metrics=(
            CalibrationMetric("skill", "a" * 64, "maximize", 0.0, 1.0),
        ),
    )
    input_plan = NeuralPriorInputPlan(
        valid_times=grid.valid_times,
        grid_contract_digest=grid.digest,
        radar_product_digest=identity.radar_product_digest or "",
        qc_pipeline_digest=identity.qc_pipeline_digest,
        background_cycle_rule_digest=identity.background_cycle_rule_digest or "",
        mask_policy_digest=identity.mask_policy_digest or "",
        observation_valid_time=grid.valid_times[-1],
        input_available_time=input_available_time,
        decision_deadline=decision_deadline,
        publication_time=publication_time,
    )
    retained_quality = masks.to(frames) if quality_weight is None else quality_weight
    retained_std = (
        torch.full_like(frames, 2.0)
        if observation_std_dbz is None
        else observation_std_dbz
    )
    retained_applicability = (
        torch.ones_like(masks)
        if applicability_mask is None
        else applicability_mask
    )
    run = ForecastRunContract.from_inputs(
        config,
        frames,
        masks,
        None,
        observation_quality_weight=retained_quality,
        observation_std_dbz=retained_std,
        grid_time_contract=grid,
        analysis_config_json=analysis_config_json,
        operational_calibration_manifest_json=manifest.json,
        operational_calibration_manifest_digest=manifest.digest,
        operational_calibration_approval_digest=manifest.digest,
        operational_data_identity_json=identity.json,
        operational_data_identity_digest=identity.digest,
        input_plan_json=input_plan.json,
        input_plan_digest=input_plan.plan_digest,
    )
    context = InterventionInputContext.from_inputs(
        frames_dbz=frames,
        observation_masks=masks,
        quality_weight=retained_quality,
        observation_std_dbz=retained_std,
        background_frames_dbz=None,
        radar_id="radar-1",
        applicability_mask=retained_applicability,
        run=run,
    )
    return run, context, input_plan.plan_digest, input_plan.json


def _operator_approval(
    decision: ProspectiveInterventionDecision,
) -> tuple[OperatorActionApproval, SimpleNamespace, Ed25519PrivateKey]:
    private_key = Ed25519PrivateKey.generate()
    trust = SimpleNamespace(
        keys={"operator-1": private_key.public_key()},
        roles={"operator-1": frozenset(("duty-meteorologist",))},
        content_digest="1" * 64,
    )
    approval = OperatorActionApproval.from_decision(
        decision,
        operator_key_id="operator-1",
        operator_role="duty-meteorologist",
        operator_trust_store_digest=trust.content_digest,
        operator_private_key=private_key,
        reviewed_at=decision.decided_at,
        expires_at=decision.decision_deadline,
        operator_comment_digest="2" * 64,
    )
    return approval, trust, private_key


class EpisodeLedgerTests(unittest.TestCase):
    def test_pr140_generations_load_as_audit_only(self) -> None:
        plan_payload = {
            "contract": "neural-prior-holdout-plan-v31",
            "plan_id": "pr140-audit-plan",
        }
        plan_digest = json_digest(plan_payload)
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
                        "pr140-audit-plan",
                        json.dumps(
                            stored_plan,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "1" * 64,
                        "2" * 64,
                        "2026-08-23T00:00:00Z",
                        "2026-08-23T00:00:00Z",
                    ),
                )
            loaded = ledger.load_neural_prior_holdout_plan(plan_digest)
        self.assertIsInstance(
            loaded,
            promotion_module.LegacyNeuralPriorHoldoutPlanV31Audit,
        )
        self.assertFalse(hasattr(loaded, "cases"))

        shard_digest = "3" * 64
        tensor_records = tuple(
            ledger_module.ScoringReplayTensorRecord(
                case_id="case-a",
                role=role,
                archive_member="tensor",
                dtype="float32",
                shape=(1,),
                tensor_digest="4" * 64,
                archive_sha256=shard_digest,
            )
            for role in sorted(
                ledger_module.SCORING_REPLAY_REQUIRED_TENSOR_ROLES
            )
        )
        legacy_manifest = (
            ledger_module.LegacyScoringReplayBundleManifestAuditV21(
                scoring_input_artifact_digest="5" * 64,
                ordered_case_ids=("case-a",),
                ordered_evaluation_digests=("6" * 64,),
                semantic_case_digests=("7" * 64,),
                dynamic_source_case_ids=(),
                background_case_ids=(),
                algorithm_source_manifest_digest="8" * 64,
                runtime_compatibility_digest="9" * 64,
                runtime_exact_digest="a" * 64,
                scoring_backend_certification_policy_digest=None,
                scoring_backend_certification_evidence_digest=None,
                tensor_records=tensor_records,
                tensor_archive_sha256="b" * 64,
                evaluation_payload_sha256="c" * 64,
                raw_provenance_payload_sha256="d" * 64,
                verification_provenance_payload_sha256="e" * 64,
                raw_ingestor_trust_store_digest="f" * 64,
                tensor_shard_sha256s=(shard_digest,),
            )
        )
        decoded = ledger_module._decode_scoring_replay_bundle_manifest(
            json.dumps(
                legacy_manifest.payload
                | {"bundle_digest": legacy_manifest.bundle_digest},
                sort_keys=True,
                separators=(",", ":"),
            ),
            expected_digest=legacy_manifest.bundle_digest,
        )
        self.assertIs(
            type(decoded),
            ledger_module.LegacyScoringReplayBundleManifestAuditV21,
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = _computed_snapshot()
        cls.contract = _contract(cls.snapshot)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.ledger = EpisodeLedger(self.root)

    def test_executor_public_key_store_must_not_be_world_writable(self) -> None:
        path = self.root / "executor.json"
        path.write_text(
            json.dumps(
                {
                    "contract": "advar-executor-trust-store-v2",
                    "public_keys": {"executor": (b"x" * 32).hex()},
                }
            ),
            encoding="utf-8",
        )
        metadata = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o666,
            st_uid=0,
        )
        with patch("advar.ledger.os.fstat", return_value=metadata):
            with self.assertRaisesRegex(ValueError, "root-owned and non-writable"):
                ledger_module._load_executor_trust_store(path)

    def test_operator_public_key_store_must_not_be_world_writable(self) -> None:
        path = self.root / "operators.json"
        path.write_text(
            json.dumps(
                {
                    "contract": "advar-operator-trust-store-v1",
                    "operators": {
                        "operator": {
                            "public_key": (b"x" * 32).hex(),
                            "roles": ["duty-meteorologist"],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        metadata = SimpleNamespace(st_mode=stat.S_IFREG | 0o666, st_uid=0)
        with patch("advar.ledger.os.fstat", return_value=metadata):
            with self.assertRaisesRegex(ValueError, "root-owned and non-writable"):
                ledger_module._load_operator_trust_store(path)

    def test_holdout_plan_rechecks_clock_before_commit(self) -> None:
        issue = "2030-01-01T00:00:00Z"
        metric_grid = RadarGridTimeContract(
            valid_times=(
                "2029-12-31T23:40:00Z",
                "2029-12-31T23:50:00Z",
                issue,
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="1" * 64,
            spatial_grid_contract="radar-spatial-grid-identity-v5",
            grid_shape_yx=(1, 2),
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
        input_plan = NeuralPriorInputPlan(
            valid_times=(issue,),
            grid_contract_digest=metric_grid.digest,
            radar_product_digest="2" * 64,
            qc_pipeline_digest="3" * 64,
            background_cycle_rule_digest="4" * 64,
            mask_policy_digest="5" * 64,
            observation_valid_time=issue,
            input_available_time=issue,
            decision_deadline="2030-01-01T00:02:00Z",
            publication_time="2030-01-01T00:05:00Z",
        )
        verification_source_key = Ed25519PrivateKey.from_private_bytes(
            b"\x2c" * 32
        )
        observation_geometry = (
            RadarObservationGeometryContract.from_grid_time_contract(
                metric_grid
            )
        )
        observation_source_registry = MosaicObservationSourceRegistry(
            radar_source_kind="single_site",
            ordered_sources=(
                ObservationRadarSource(
                    radar_site_digest="2" * 64,
                    calibration_epoch_digest="2" * 64,
                    quality_weight=1.0,
                    observation_std_dbz=2.0,
                    projected_x_m=1_000_000.0,
                    projected_y_m=2_000_000.0,
                    radar_altitude_m=100.0,
                    representative_scan_elevation_deg=1.0,
                    contract="observation-radar-source-v4",
                ),
            ),
            projected_crs_digest=(
                observation_geometry.projected_crs_digest
            ),
            metric_domain_digest=CURRENT_RADAR_METRIC_DOMAIN.digest,
            geometry_model="projected-horizontal-representative-tilt-v1",
            radar_altitude_role="provenance_only",
            contract="mosaic-observation-source-registry-v6",
        )
        observation_error_plan = VerificationObservationErrorPlan(
            radar_source_kind="single_site",
            source_registry_digest=(
                observation_source_registry.source_registry_digest
            ),
            calibration_registry_digest=(
                observation_source_registry.calibration_registry_digest
            ),
            range_elevation_validity_algorithm_digest=(
                OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST
            ),
            beam_blockage_algorithm_digest=(
                OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST
            ),
            attenuation_qc_digest="3" * 64,
            censoring_rule_digest="f" * 64,
            spatial_correlation_block_algorithm_digest="7" * 64,
            quality_weight_interpretation_digest="8" * 64,
            quality_weight_algorithm_digest=(
                OBSERVATION_ERROR_DERIVATION_ALGORITHM_V11_DIGEST
            ),
            observation_std_algorithm_digest=(
                OBSERVATION_ERROR_DERIVATION_ALGORITHM_V11_DIGEST
            ),
            observation_error_model_digest="b" * 64,
            source_assignment_algorithm_digest=(
                OBSERVATION_SOURCE_SELECTION_ALGORITHM_V1_DIGEST
            ),
            minimum_detectable_echo_dbz=-10.0,
            observation_error_reference_std_dbz=2.0,
            derivation_algorithm_digest=(
                OBSERVATION_ERROR_DERIVATION_ALGORITHM_V11_DIGEST
            ),
            mask_derivation_algorithm_digest=(
                OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST
            ),
            maximum_range_km=300.0,
            minimum_elevation_deg=-1.0,
            maximum_elevation_deg=90.0,
            maximum_beam_blockage_fraction=0.5,
            minimum_attenuation_qc_score=0.5,
            verification_source_authority_id="clock-verification-source",
            verification_source_authority_public_key_hex=(
                verification_source_key.public_key().public_bytes_raw().hex()
            ),
            maximum_acquisition_age_seconds=300.0,
            temporal_quality_decay_scale_seconds=120.0,
            temporal_quality_decay_power=2.0,
            temporal_error_growth_dbz_per_second=0.01,
            temporal_quality_decay_algorithm_digest=(
                OBSERVATION_TEMPORAL_QUALITY_DECAY_ALGORITHM_V1_DIGEST
            ),
            temporal_error_algorithm_digest=(
                OBSERVATION_TEMPORAL_ERROR_ALGORITHM_V1_DIGEST
            ),
            detection_limit_derivation_algorithm_digest=(
                OBSERVATION_DETECTION_LIMIT_ALGORITHM_V2_DIGEST
            ),
            censor_state_derivation_algorithm_digest=(
                OBSERVATION_REPORT_KIND_ALGORITHM_V1_DIGEST
            ),
            geometry_contract_digest=observation_geometry.geometry_digest,
            acquisition_timestamp_reference="volume_end",
            spatial_metric_reference_speed_mps=20.0,
            spatial_metric_maximum_displacement_fraction_cells=1.0,
            spatial_age_gate_algorithm_digest=(
                OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V3_DIGEST
            ),
            contract="verification-observation-error-plan-v12",
        )
        target_plan = PriorUncertaintyTargetPlan(
            plan_id="uncertainty-clock",
            target_kind="withheld_radar",
            source_identity_digest="b" * 64,
            qc_pipeline_digest="3" * 64,
            mask_policy_digest="5" * 64,
            censor_policy_digest="f" * 64,
            floor_representation_contract_digest="a" * 64,
            grid_contract_digest=metric_grid.digest,
            feature_exclusion_contract_digest="c" * 64,
            independence_evidence_digest="d" * 64,
            verification_observation_error_plan_digest=(
                observation_error_plan.plan_digest
            ),
            target_valid_time="2030-01-01T01:00:00Z",
            prior_probability_contract_digest="e" * 64,
        )
        state_target_plan = NeuralPriorStateCalibrationPlan(
            plan_id="state-clock",
            target_kind="withheld_radar",
            source_identity_digest="b" * 64,
            qc_pipeline_digest="3" * 64,
            mask_policy_digest="5" * 64,
            censor_policy_digest="f" * 64,
            floor_representation_contract_digest="a" * 64,
            grid_contract_digest=metric_grid.digest,
            feature_exclusion_contract_digest="c" * 64,
            independence_evidence_digest="d" * 64,
            verification_observation_error_plan_digest=(
                observation_error_plan.plan_digest
            ),
            target_valid_time=issue,
            state_contract_digest="0" * 64,
            support_threshold_dbz=5.0,
        )
        range_grid = torch.zeros((2, 2), dtype=torch.float64)
        range_geometry = promotion_module.RangeGeometryContract(
            radar_site_digest="7" * 64,
            radar_site_location_digest="7" * 64,
            grid_contract_digest=metric_grid.digest,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near_range",),
            radial_distance_edges_m=(0.0, 100_000.0),
            horizontal_range_rule_digest="8" * 64,
            grid_x_m_digest=tensor_digest(range_grid),
            grid_y_m_digest=tensor_digest(range_grid),
        )
        range_contract = promotion_module.RangeBandContract(
            case_id="case-clock",
            range_regime_labels=("near_range",),
            range_band_mask_digests=(
                tensor_digest(torch.ones((2, 2), dtype=torch.bool)),
            ),
            reference_active_range_regimes=("near_range",),
            grid_contract_digest=metric_grid.digest,
            range_geometry_contract_digest=range_geometry.contract_digest,
        )
        issuance_domain_plan = promotion_module.OperationalIssuanceDomainPlan(
            case_id="case-clock",
            grid_contract_digest=metric_grid.digest,
            radar_source_contract_digest="2" * 64,
            lead_minutes=(60,),
            publication_policy_digest="3" * 64,
            source_coverage_policy_digest="4" * 64,
            permanent_exclusion_policy_digest="5" * 64,
            publication_eligible_mask_digest=tensor_digest(
                torch.ones((1, 2, 2), dtype=torch.bool)
            ),
            source_coverage_mask_digest=tensor_digest(
                torch.ones((1, 2, 2), dtype=torch.bool)
            ),
            permanent_exclusion_mask_digest=tensor_digest(
                torch.zeros((1, 2, 2), dtype=torch.bool)
            ),
        )
        training_registry_key = (
            promotion_module.Ed25519PrivateKey.from_private_bytes(b"\x24" * 32)
        )
        training_registry_receipt = (
            promotion_module.TrainingRawRegistryReceipt.issue(
                raw_volume_identity_digests=("6" * 64,),
                sampling_unit_digests=("7" * 64,),
                registry_id="clock-global-sampling-registry",
                authority_id="clock-sampling-authority",
                authority_private_key=training_registry_key,
                committed_at="2028-12-31T00:00:00Z",
            )
        )
        processor_key = promotion_module.Ed25519PrivateKey.from_private_bytes(
            b"\x25" * 32
        )
        feature_tensor = torch.arange(20, dtype=torch.float64).reshape(5, 2, 2)
        target_tensor = torch.arange(4, dtype=torch.float64).reshape(2, 2)
        member_unsigned = {
            "contract": "analysis-input-derivation-artifact-v5",
            "case_id": "classifier-training-case",
            "input_plan_digest": "9" * 64,
            "resolved_raw_observation_receipt_digests": ("5" * 64,),
            "canonical_raw_volume_identity_digests": ("6" * 64,),
            "global_raw_resolution_receipt_digest": (
                training_registry_receipt.receipt_digest
            ),
            "decoder_version_digest": "8" * 64,
            "qc_algorithm_digest": "a" * 64,
            "qc_policy_digest": "b" * 64,
            "source_selection_evidence_digest": "c" * 64,
            "regrid_algorithm_digest": "5" * 64,
            "grid_contract_digest": "3" * 64,
            "background_cycle_rule_digest": "d" * 64,
            "background_valid_times": (),
            "background_source_identity_digest": None,
            "background_input_identity_digests": (),
            "input_frames_digest": "e" * 64,
            "observation_masks_digest": "f" * 64,
            "observation_quality_weight_digest": "0" * 64,
            "observation_std_dbz_digest": "1" * 64,
            "source_available_mask_digest": "2" * 64,
            "learned_model_input_features_digest": tensor_digest(
                feature_tensor
            ),
            "background_frames_digest": None,
            "input_bundle_digest": "1" * 64,
            "full_analysis_input_digest": "2" * 64,
            "processed_at": "2028-12-30T00:00:00Z",
            "processor_id": "clock-analysis-processor",
            "processor_public_key_hex": (
                processor_key.public_key().public_bytes_raw().hex()
            ),
        }
        member_derivation = promotion_module.AnalysisInputDerivationArtifact(
            **member_unsigned,
            processor_signature_hex=processor_key.sign(
                promotion_module.json_digest(member_unsigned).encode("ascii")
            ).hex(),
        )
        signed_member_manifest_digest = promotion_module.json_digest(
            {
                "contract": "signed-training-member-manifest-v1",
                "members": [
                    {
                        "analysis_input_derivation_artifact_digest": (
                            member_derivation.artifact_digest
                        ),
                        "processor_id": member_derivation.processor_id,
                        "processor_public_key_hex": (
                            member_derivation.processor_public_key_hex
                        ),
                    }
                ],
            }
        )
        target_source_key = (
            promotion_module.Ed25519PrivateKey.from_private_bytes(b"\x28" * 32)
        )
        target_source_trust_store = (
            promotion_module.TrainingTargetSourceTrustStore.issue(
                authorities=((
                    "test-target-source-authority",
                    target_source_key.public_key().public_bytes_raw().hex(),
                    1,
                    "2025-01-01T00:00:00Z",
                    "2035-01-01T00:00:00Z",
                    None,
                    ("a" * 64,),
                    ("b" * 64,),
                ),),
                root_authority_id="test-target-source-root",
                root_private_key=promotion_module.Ed25519PrivateKey.from_private_bytes(
                    b"\x2b" * 32
                ),
            )
        )
        event_key = promotion_module.Ed25519PrivateKey.from_private_bytes(
            b"\x29" * 32
        )
        event_scheduler_key = (
            promotion_module.Ed25519PrivateKey.from_private_bytes(b"\x2a" * 32)
        )
        training_event_plan = promotion_module.PhysicalEventCatalogPlan(
            holdout_case_ids=(member_derivation.case_id,),
            association_algorithm_digest="3" * 64,
            spatial_membership_rule_digest="4" * 64,
            adjudication_policy_digest="5" * 64,
            adjudicator_id="training-event-adjudicator",
            adjudicator_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(event_key)
            ),
            catalog_completion_deadline="2029-01-01T01:00:00Z",
            spatial_reference_digest="7" * 64,
            motion_association_rule_digest="8" * 64,
            scheduler_id="training-event-scheduler",
            scheduler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(
                    event_scheduler_key
                )
            ),
            scheduler_trust_store_digest="9" * 64,
        )
        training_event_track = promotion_module.PhysicalEventTrackArtifact(
            timestamps=(
                "2029-01-01T00:00:00Z",
                "2029-01-01T00:50:00Z",
            ),
            centroid_xy_m=((0.0, 0.0), (0.0, 0.0)),
            object_mask_digests=("a" * 64, "b" * 64),
            source_radar_ids=("classifier-radar", "classifier-radar"),
            association_edge_digests=("c" * 64,),
            spatial_reference_digest="7" * 64,
        )
        training_event = (
            promotion_module.PhysicalEventCatalogEvidence.from_members(
                event_id="classifier-training-event",
                member_case_ids=(member_derivation.case_id,),
                member_full_analysis_input_digests=(
                    member_derivation.full_analysis_input_digest,
                ),
                start_time="2029-01-01T00:00:00Z",
                end_time="2029-01-01T00:50:00Z",
                spatial_envelope_xy_m=(0.0, 0.0, 10_000.0, 10_000.0),
                object_track_artifact=training_event_track,
                participating_radar_ids=("classifier-radar",),
                association_algorithm_digest=(
                    training_event_plan.association_algorithm_digest
                ),
                adjudication_policy_digest=(
                    training_event_plan.adjudication_policy_digest
                ),
                adjudicator_id=training_event_plan.adjudicator_id,
                adjudicator_private_key=event_key,
            )
        )
        training_event_spatial = (
            promotion_module.PhysicalEventCaseSpatialEvidence(
                case_id=member_derivation.case_id,
                full_analysis_input_digest=(
                    member_derivation.full_analysis_input_digest
                ),
                physical_event_identity_digest=(
                    training_event.physical_event_identity_digest
                ),
                observed_spatial_envelope_xy_m=(0.0, 0.0, 5_000.0, 5_000.0),
                event_spatial_envelope_xy_m=(
                    training_event.spatial_envelope_xy_m
                ),
                spatial_membership_rule_digest=(
                    training_event_plan.spatial_membership_rule_digest
                ),
                source_object_evidence_digest=(
                    training_event_track.object_mask_digests[0]
                ),
                track_artifact_digest=training_event_track.artifact_digest,
                track_sample_index=0,
                track_sample_time=training_event_track.timestamps[0],
                track_object_mask_digest=(
                    training_event_track.object_mask_digests[0]
                ),
                input_available_time="2029-01-01T00:10:00Z",
                spatial_reference_digest=(
                    training_event_plan.spatial_reference_digest
                ),
            )
        )
        training_event_result = (
            promotion_module.PhysicalEventCatalogResult.from_plan(
                training_event_plan,
                event_evidences=(training_event,),
                case_spatial_membership_evidences=(training_event_spatial,),
                cataloged_at="2029-01-01T00:20:00Z",
                adjudicator_private_key=event_key,
            )
        )
        target_source_receipt = (
            promotion_module.TrainingTargetSourceReceipt.issue(
                target_source_identity_digest="4" * 64,
                target_source_valid_time="2029-01-01T00:10:00Z",
                physical_event_digest=(
                    training_event.physical_event_identity_digest
                ),
                source_object_digest=tensor_digest(target_tensor),
                observed_at="2029-01-01T00:15:00Z",
                source_contract_digest="a" * 64,
                radar_product_scope_digest="b" * 64,
                trust_store=target_source_trust_store,
                authority_id="test-target-source-authority",
                authority_key_epoch=1,
                authority_private_key=target_source_key,
            )
        )
        target_derivation = (
            promotion_module.TrainingTargetDerivationArtifact.issue(
                case_id=member_derivation.case_id,
                target_source_receipt=target_source_receipt,
                target_qc_policy_digest="5" * 64,
                target_censor_policy_digest="6" * 64,
                target_algorithm_digest="6" * 64,
                target_schema_digest="7" * 64,
                target_tensor=target_tensor,
                generated_at="2029-01-01T00:20:00Z",
                training_cutoff_time="2029-01-01T00:30:00Z",
                processor_id=member_derivation.processor_id,
                processor_private_key=processor_key,
            )
        )
        target_derivation_json = json.dumps(
            target_derivation.payload
            | {"artifact_digest": target_derivation.artifact_digest},
            sort_keys=True,
            separators=(",", ":"),
        )
        archive_root = self.root / "training-shards"
        feature_shard = promotion_module.write_training_tensor_archive_shard(
            {"feature_00000": feature_tensor},
            directory=archive_root,
            shard_id="features-00000",
        )
        target_shard = promotion_module.write_training_tensor_archive_shard(
            {
                "target_00000": target_tensor,
                "target_valid_mask_00000": torch.ones_like(
                    target_tensor, dtype=torch.bool
                ),
                "target_quality_00000": torch.ones_like(target_tensor),
            },
            directory=archive_root,
            shard_id="targets-00000",
        )
        normalization_tensor = (
            promotion_module.recompute_training_normalization_statistics(
                (feature_tensor,),
                channel_definitions=(
                    "dbz",
                    "qc_valid",
                    "quality",
                    "observation_std",
                    "source_available",
                ),
            )
        )
        normalization_shard = (
            promotion_module.write_training_tensor_archive_shard(
                {"normalization": normalization_tensor},
                directory=archive_root,
                shard_id="normalization",
            )
        )
        training_member = promotion_module.TrainingDatasetMember(
                    case_id=member_derivation.case_id,
                    analysis_derivation_artifact_digest=(
                        member_derivation.artifact_digest
                    ),
                    feature_archive_member="feature_00000",
                    feature_tensor_digest=(
                        member_derivation.learned_model_input_features_digest
                    ),
                    target_archive_member="target_00000",
                    target_tensor_digest=tensor_digest(target_tensor),
                    target_valid_mask_archive_member=(
                        "target_valid_mask_00000"
                    ),
                    target_quality_archive_member="target_quality_00000",
                    target_derivation_artifact_json=target_derivation_json,
                    sample_weight=1.0,
                    split="train",
                    augmentation_seed=0,
                )
        normalization_derivation = (
            promotion_module.NormalizationDerivationArtifact.from_training_dataset(
                members=(training_member,),
                normalization_shard=normalization_shard,
                channel_definitions=(
                    "dbz",
                    "qc_valid",
                    "quality",
                    "observation_std",
                    "source_available",
                ),
                normalization_algorithm_digest=(
                    promotion_module.TRAINING_NORMALIZATION_ALGORITHM_DIGEST
                ),
                mask_weight_policy_digest=(
                    promotion_module
                    .TRAINING_NORMALIZATION_MASK_WEIGHT_POLICY_DIGEST
                ),
            )
        )
        training_feature_dataset = (
            promotion_module.TrainingFeatureDatasetArtifact(
                members=(training_member,),
                normalization_statistics_digest=tensor_digest(
                    normalization_tensor
                ),
                feature_algorithm_digest="f" * 64,
                feature_schema_digest="0" * 64,
                target_algorithm_digest="6" * 64,
                target_schema_digest="7" * 64,
                feature_archive_shards=(feature_shard,),
                target_archive_shards=(target_shard,),
                normalization_statistics_shard=normalization_shard,
                normalization_derivation_artifact_json=(
                    normalization_derivation.json
                ),
                normalization_derivation_artifact_digest=(
                    normalization_derivation.artifact_digest
                ),
            )
        )
        training_feature_dataset_json = json.dumps(
            training_feature_dataset.payload
            | {"dataset_digest": training_feature_dataset.dataset_digest},
            sort_keys=True,
            separators=(",", ":"),
        )
        training_derivation = promotion_module.TrainingDatasetDerivationArtifact(
            training_raw_registry_receipt_digest=(
                training_registry_receipt.receipt_digest
            ),
            raw_volume_identity_digests=("6" * 64,),
            sampling_unit_digests=("7" * 64,),
            training_input_bundle_digests=("1" * 64,),
            training_full_analysis_input_digests=("2" * 64,),
            training_grid_contract_digests=("3" * 64,),
            training_member_analysis_derivation_artifact_jsons=(
                json.dumps(
                    member_derivation.payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
            decoder_algorithm_digest="8" * 64,
            qc_algorithm_digest="a" * 64,
            regrid_algorithm_digest="5" * 64,
            feature_algorithm_digest="f" * 64,
            feature_schema_digest="0" * 64,
            signed_training_member_manifest_digest=(
                signed_member_manifest_digest
            ),
            training_feature_dataset_artifact_digest=(
                training_feature_dataset.dataset_digest
            ),
            training_feature_dataset_artifact_json=(
                training_feature_dataset_json
            ),
            training_physical_event_catalog_plan_digest=(
                training_event_plan.plan_digest
            ),
            training_physical_event_catalog_plan_json=(
                promotion_module._physical_event_catalog_plan_json(
                    training_event_plan
                )
            ),
            training_physical_event_catalog_result_digest=(
                training_event_result.result_digest
            ),
            training_physical_event_catalog_result_json=(
                promotion_module._physical_event_catalog_result_json(
                    training_event_result
                )
            ),
            training_target_source_trust_store_digest=(
                target_source_trust_store.content_digest
            ),
            training_target_source_trust_store_json=json.dumps(
                target_source_trust_store.payload
                | {"content_digest": target_source_trust_store.content_digest},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        classifier_manifest = promotion_module.RegimeClassifierManifest(
            classifier_digest="b" * 64,
            training_dataset_digest=training_derivation.training_dataset_digest,
            training_case_ids=("classifier-training-case",),
            training_input_bundle_digests=("1" * 64,),
            training_full_analysis_input_digests=("2" * 64,),
            training_physical_event_digests=(
                training_event.physical_event_identity_digest,
            ),
            training_storm_ids=("classifier-training-storm",),
            training_days=("2029-01-01",),
            training_radar_ids=("classifier-radar",),
            training_grid_contract_digests=("3" * 64,),
            training_raw_volume_identity_digests=("6" * 64,),
            training_sampling_unit_digests=("7" * 64,),
            training_raw_registry_receipt_digest=(
                training_registry_receipt.receipt_digest
            ),
            training_raw_registry_receipt_payload_json=json.dumps(
                training_registry_receipt.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
            training_dataset_derivation_artifact_digest=(
                training_derivation.artifact_digest
            ),
            training_dataset_derivation_artifact_json=json.dumps(
                training_derivation.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
            training_time_windows=((
                "2029-01-01T00:00:00Z",
                "2029-01-01T01:00:00Z",
            ),),
            training_algorithm_digest="d" * 64,
            numerical_runtime_digest=(
                promotion_module.numerical_runtime_identity_digest("cpu")
            ),
            reference_label_contract_digest="e" * 64,
            signed_training_member_manifest_digest=(
                signed_member_manifest_digest
            ),
        )
        labeler_key = promotion_module.Ed25519PrivateKey.from_private_bytes(
            b"\x02" * 32
        )
        scheduler_key = promotion_module.Ed25519PrivateKey.from_private_bytes(
            b"\x03" * 32
        )
        reference_plan = promotion_module.RegimeReferencePlan(
            case_id="case-clock",
            labeler_id="clock-labeler",
            labeler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(labeler_key)
            ),
            source_contract_digest="e" * 64,
            labeling_valid_time="2030-01-01T01:00:00Z",
            adjudication_policy_digest="6" * 64,
        )
        event_catalog_plan = promotion_module.PhysicalEventCatalogPlan(
            holdout_case_ids=("case-clock",),
            association_algorithm_digest="a" * 64,
            spatial_membership_rule_digest="b" * 64,
            adjudication_policy_digest="6" * 64,
            adjudicator_id="clock-labeler",
            adjudicator_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(labeler_key)
            ),
            catalog_completion_deadline="2030-01-01T02:00:00Z",
            spatial_reference_digest="c" * 64,
            motion_association_rule_digest="d" * 64,
            scheduler_id="clock-scheduler",
            scheduler_public_key_hex=(
                promotion_module.regime_reference_public_key_hex(scheduler_key)
            ),
            scheduler_trust_store_digest="f" * 64,
        )
        decision_rule = object.__new__(promotion_module.PromotionDecisionRule)
        object.__setattr__(decision_rule, "decision_payload_json", "{}")
        object.__setattr__(
            decision_rule,
            "contract",
            "neural-prior-promotion-decision-rule-v1",
        )
        object.__setattr__(
            decision_rule,
            "rule_digest",
            json_digest(decision_rule.payload),
        )
        raw_ingestor_key = (
            promotion_module.Ed25519PrivateKey.from_private_bytes(b"\x23" * 32)
        )
        processor_key = (
            promotion_module.Ed25519PrivateKey.from_private_bytes(b"\x25" * 32)
        )
        raw_slot = promotion_module.RawObservationSlotPlan(
            radar_site_digest=range_geometry.radar_site_digest,
            acquisition_valid_time=input_plan.observation_valid_time,
            scan_strategy_rule_digest="5" * 64,
            source_selection_rule_digest="6" * 64,
            canonical_geodetic_footprint_digest="6" * 64,
        )
        sampling_unit = promotion_module.MeteorologicalSamplingUnit(
            raw_observation_slot_digests=(raw_slot.slot_digest,),
            canonical_geodetic_footprint_digest="6" * 64,
        )
        cases = (
                NeuralPriorHoldoutPlanCase(
                    case_id="case-clock",
                    storm_id="pending",
                    day="2029-12-31",
                    radar_id="radar-clock",
                    regime="pending",
                    range_regime="near_range",
                    input_plan_digest=input_plan.plan_digest,
                    verification_plan_digest="8" * 64,
                    metric_contract_digest="9" * 64,
                    uncertainty_target_plan_digest=target_plan.plan_digest,
                    state_calibration_target_plan_digest=(
                        state_target_plan.plan_digest
                    ),
                    range_band_contract_digest=range_contract.contract_digest,
                    reference_active_range_regimes=("near_range",),
                    regime_reference_plan_digest=reference_plan.plan_digest,
                    operational_issuance_domain_plan_digest=(
                        issuance_domain_plan.plan_digest
                    ),
                    meteorological_sampling_unit_digest=(
                        sampling_unit.sampling_unit_digest
                    ),
                    issue_time=issue,
                ),
            )
        holdout_cohort_digest = promotion_module._holdout_dataset_digest(cases)
        trials = (
            promotion_module.PromotionExperimentTrial(
                candidate_prior_digest="7" * 64,
                promotion_decision_rule_digest=decision_rule.rule_digest,
                classifier_manifest_digests=(
                    classifier_manifest.manifest_digest,
                ),
            ),
        )
        registry_key = promotion_module.Ed25519PrivateKey.from_private_bytes(
            b"\x24" * 32
        )
        reservation = promotion_module.GlobalSamplingReservationReceipt.issue(
            experiment_scope_digest=(
                promotion_module._promotion_experiment_scope_digest(
                    holdout_cohort_digest=holdout_cohort_digest,
                    parent_prior_digest="6" * 64,
                    trials=trials,
                    winner_selection_rule_digest="5" * 64,
                )
            ),
            raw_observation_slot_digests=(raw_slot.slot_digest,),
            registry_id="clock-global-sampling-registry",
            authority_id="clock-sampling-authority",
            authority_private_key=registry_key,
            reserved_at="2029-01-01T00:00:00Z",
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
            raw_observation_slot_digests=(raw_slot.slot_digest,),
            global_sampling_reservation=reservation,
            parent_prior_digest="6" * 64,
            trials=trials,
            winner_selection_rule_digest="5" * 64,
        )
        plan = NeuralPriorHoldoutPlan(
            plan_id="clock-plan",
            parent_prior_digest="6" * 64,
            candidate_family_digests=("7" * 64,),
            cases=cases,
            input_plans=(input_plan,),
            raw_observation_slot_plans=(raw_slot,),
            meteorological_sampling_units=(sampling_unit,),
            raw_ingestor_trust_store=promotion_module.RawIngestorTrustStore(
                authorities=((
                    "clock-raw-ingestor",
                    raw_ingestor_key.public_key().public_bytes_raw().hex(),
                    "2026-01-01T00:00:00Z",
                    "2027-01-01T00:00:00Z",
                    None,
                ),),
            ),
            training_target_source_trust_store=target_source_trust_store,
            analysis_processor_id="clock-analysis-processor",
            analysis_processor_public_key_hex=(
                processor_key.public_key().public_bytes_raw().hex()
            ),
            training_target_source_authority_id=(
                "test-target-source-authority"
            ),
            training_target_source_authority_public_key_hex=(
                target_source_key
                .public_key()
                .public_bytes_raw()
                .hex()
            ),
            uncertainty_target_plans=(target_plan,),
            state_calibration_target_plans=(state_target_plan,),
            verification_observation_error_plans=(observation_error_plan,),
            range_band_contracts=(range_contract,),
            range_geometry_contracts=(range_geometry,),
            operational_issuance_domain_plans=(issuance_domain_plan,),
            regime_reference_plans=(reference_plan,),
            physical_event_catalog_plan=event_catalog_plan,
            regime_classifier_manifests=(classifier_manifest,),
            promotion_experiment_family=experiment_family,
            promotion_decision_rule_digest=decision_rule.rule_digest,
            reference_label_contract_digest="e" * 64,
            scoring_algorithm_digest="1" * 64,
            scoring_runtime_digest="2" * 64,
            metric_engine_digest=promotion_module.scoring_metric_engine_identity_digest(),
            verification_resolver_digest="4" * 64,
            registered_at="2029-01-01T00:00:00Z",
        )
        policy = NeuralPriorHoldoutPlanPolicy(
            approved_plan_digests=(plan.plan_digest,),
            approved_metric_contract_digests=("9" * 64,),
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
            ),
            raw_ingestor_trust_store_digest=(
                plan.raw_ingestor_trust_store.content_digest
            ),
            training_target_source_trust_store_digest=(
                plan.training_target_source_trust_store.content_digest
            ),
            analysis_processor_id=plan.analysis_processor_id,
            analysis_processor_public_key_hex=(
                plan.analysis_processor_public_key_hex
            ),
            training_target_source_authority_id=(
                plan.training_target_source_authority_id
            ),
            training_target_source_authority_public_key_hex=(
                plan.training_target_source_authority_public_key_hex
            ),
        )
        trust = SimpleNamespace(
            approved_policy_digests=frozenset(
                (
                    policy.digest,
                    decision_rule.rule_digest,
                    experiment_family.family_digest,
                )
            ),
            content_digest="a" * 64,
        )
        scheduler_trust = SimpleNamespace(
            keys={"clock-scheduler": scheduler_key.public_key()},
            content_digest="f" * 64,
        )

        class _CrossingClock(datetime):
            calls = 0

            @classmethod
            def now(cls, tz=None):  # type: ignore[no-untyped-def]
                cls.calls += 1
                value = (
                    "2029-12-31T23:59:59+00:00"
                    if cls.calls == 1
                    else "2030-01-01T00:00:01+00:00"
                )
                return datetime.fromisoformat(value)

        with patch(
            "advar.ledger._load_learning_policy_trust_store",
            return_value=trust,
        ), patch(
            "advar.ledger._load_scheduler_trust_store",
            return_value=scheduler_trust,
        ), patch("advar.ledger.datetime", _CrossingClock):
            with self.assertRaisesRegex(ValueError, "crossed its forecast issue"):
                self.ledger.append_neural_prior_holdout_plan(
                    plan,
                    promotion_decision_rule=decision_rule,
                    policy=policy,
                    policy_trust_store_path="/etc/advar/policies.json",
                    scheduler_trust_store_path="/etc/advar/schedulers.json",
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def episode(self, episode_id: str = "episode-001") -> SensitivityEpisode:
        return SensitivityEpisode(
            episode_id=episode_id,
            issue_time="2026-07-26T05:00:00+00:00",
            radar_id="KTLX",
            contract=self.contract,
            snapshot=self.snapshot,
            action_features=(1.25, -0.5),
        )

    def test_append_load_verify_and_reopen_computed_snapshot(self) -> None:
        episode = self.episode()
        target = self.ledger.append(episode)

        self.assertEqual(
            target,
            self.ledger.episodes_dir / episode.episode_id,
        )
        self.ledger.verify(episode.episode_id)
        loaded = self.ledger.load(episode.episode_id)
        self.assertEqual(
            loaded.arrays["direct_observation_sensitivity_norm"].shape,
            (18, 1),
        )
        np.testing.assert_array_equal(
            loaded.arrays["forecast_scores"],
            self.snapshot.forecast_scores.numpy(),
        )
        np.testing.assert_array_equal(
            loaded.arrays["action_features"],
            np.asarray(episode.action_features, dtype=np.float64),
        )

        manifest = loaded.manifest
        self.assertEqual(manifest["schema_version"], 18)
        self.assertEqual(manifest["tile_shape_yx"], [16, 16])
        self.assertEqual(
            manifest["verification_contract"],
            "legacy-verification-tensor-v1",
        )
        self.assertIs(manifest["verification_lineage_complete"], False)
        self.assertIsNone(manifest["verification_valid_times"])
        self.assertIs(manifest["baseline_lineage_available"], False)
        self.assertIs(manifest["reward_available"], False)
        self.assertNotIn("baseline_scores", loaded.arrays)
        self.assertNotIn("direct_normalized_reward", loaded.arrays)
        self.assertEqual(
            manifest["context_feature_names"][6:8],
            ["motion_pair_conflict", "growth_pair_conflict"],
        )
        self.assertEqual(
            loaded.arrays["context_features"].shape,
            (len(manifest["context_feature_names"]),),
        )
        self.assertEqual(manifest["contract_hash"], self.contract.digest)
        self.assertEqual(
            manifest["forecast_run_digest"],
            self.snapshot.forecast_run_digest,
        )
        self.assertEqual(
            episode.forecast_run_digest,
            self.snapshot.forecast_run_digest,
        )
        self.assertEqual(
            manifest["trust_components"]["pair_consistency"],
            1.0,
        )
        self.assertEqual(
            manifest["trust_components"]["observation_verified_evidence"],
            self.snapshot.trust_components["observation_verified_evidence"],
        )
        for name in (
            "forecast_confidence",
            "path_evidence_by_metric",
            "observation_source_fraction_by_metric",
            "observation_verified_evidence_by_metric",
            "background_verified_evidence_by_metric",
        ):
            np.testing.assert_array_equal(
                loaded.arrays[name],
                getattr(self.snapshot, name).numpy(),
            )
        self.assertIs(
            manifest["indirect_observation_sensitivity_available"],
            False,
        )
        self.assertIs(
            manifest["total_observation_sensitivity_available"],
            False,
        )
        self.assertEqual(
            manifest["sensitivity_scope"],
            {
                "input_0_minutes": ("partial_direct_latest_dbz_fixed_control"),
                "indirect_analysis_path": (
                    "unavailable_implicit_variational_fso_not_implemented"
                ),
                "total_observation_sensitivity": "unavailable",
            },
        )

        impacts = self.ledger.list_impacts(episode.episode_id)
        self.assertEqual(len(impacts), 18)
        self.assertEqual(
            {
                (
                    row["lead_minutes"],
                    row["metric_name"],
                    row["input_offset_minutes"],
                )
                for row in impacts
            },
            {(lead, "log_echo_mse", 0) for lead in range(10, 181, 10)},
        )
        self.assertEqual(
            {
                (row["input_offset_minutes"], row["direct_path_status"])
                for row in impacts
            },
            {
                (0, "partial_direct_latest_dbz_fixed_control"),
            },
        )

        reopened = EpisodeLedger(self.root)
        reopened.verify(episode.episode_id)
        reloaded = reopened.load(episode.episode_id)
        self.assertEqual(reloaded.manifest, manifest)
        np.testing.assert_array_equal(
            reloaded.arrays["direct_observation_sensitivity_norm"],
            loaded.arrays["direct_observation_sensitivity_norm"],
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 42)

    def test_unavailable_optional_arrays_are_omitted(self) -> None:
        direct = replace(
            self.snapshot.direct,
            whitened_tile_norm=None,
            impact=None,
            tile_impact=None,
            reward=None,
        )
        snapshot = replace(
            self.snapshot,
            direct=direct,
            observation_std_dbz=None,
            observation_innovation_dbz=None,
            observation_innovation_mask=None,
            baseline_scores=None,
        )
        episode = replace(
            self.episode("without-optional-arrays"),
            snapshot=snapshot,
        )

        self.ledger.append(episode)
        loaded = self.ledger.load(episode.episode_id)

        for name in (
            "tile_whitened_direct_sensitivity_norm",
            "direct_observation_impact",
            "tile_direct_observation_impact",
            "direct_normalized_reward",
            "observation_std_dbz",
            "observation_innovation_dbz",
            "observation_innovation_mask",
            "baseline_scores",
        ):
            self.assertNotIn(name, loaded.arrays)
        self.assertFalse(loaded.manifest["impact_available"])
        self.assertFalse(loaded.manifest["reward_available"])

    def test_legacy_learning_approval_remains_loadable(self) -> None:
        evidence = LearningApprovalEvidence(
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
        learning_result_digest = "a" * 64
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                """
                INSERT INTO variational_learning_approvals (
                    learning_result_digest, approval_evidence_digest,
                    evidence_contract, policy_digest, trust_store_digest,
                    fsoi_digest, full_step_analysis_digest,
                    half_step_analysis_digest, full_step_forecast_digest,
                    half_step_forecast_digest,
                    first_order_validation_digest, learning_impact_digest,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    learning_result_digest,
                    evidence.digest,
                    evidence.contract,
                    evidence.policy_digest,
                    evidence.trust_store_digest,
                    evidence.fsoi_digest,
                    evidence.full_step_analysis_digest,
                    evidence.half_step_analysis_digest,
                    evidence.full_step_forecast_digest,
                    evidence.half_step_forecast_digest,
                    evidence.first_order_validation_digest,
                    evidence.learning_impact_digest,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        loaded = self.ledger.load_variational_learning_approval(learning_result_digest)
        self.assertEqual(loaded, evidence)

        intervention = RealizedObservationIntervention(
            intervention_id="radar-qc-20260808-001",
            intervention_type="realized_qc_intervention",
            action_digest="b" * 64,
            applied_time="2026-08-08T01:02:03+09:00",
            actual_input_before_digest="c" * 64,
            actual_input_after_digest="d" * 64,
            outcome_resolution_contract_digest="3" * 64,
            execution_policy_digest="4" * 64,
            execution_trust_store_digest="5" * 64,
            predicted_normalized_benefit=0.2,
            resolved_normalized_benefit=0.2,
            learning_result_digest=learning_result_digest,
            learning_approval_evidence_digest=evidence.digest,
            counterfactual_perturbation_digest="f" * 64,
            linearization_digest="0" * 64,
            case_id="case-1",
            radar_id="radar-1",
            issue_time="2027-08-08T00:00:00Z",
            input_bundle_before_digest="1" * 64,
            input_bundle_after_digest="2" * 64,
            resolved_issuance_validation_digest="3" * 64,
            contract="realized-observation-intervention-v3",
        )
        digest = self.ledger.append_realized_observation_intervention(intervention)
        self.assertEqual(
            self.ledger.load_realized_observation_intervention(digest),
            intervention,
        )
        legacy_intervention = replace(
            intervention,
            intervention_id="legacy-radar-qc-20260808-001",
            outcome_resolution_contract_digest="6" * 64,
            execution_policy_digest="0" * 64,
            execution_trust_store_digest="0" * 64,
            predicted_normalized_benefit=0.0,
            resolved_normalized_benefit=0.0,
            contract="realized-observation-intervention-v1",
        )
        legacy_digest = self.ledger.append_realized_observation_intervention(
            legacy_intervention
        )
        self.assertEqual(
            self.ledger.load_realized_observation_intervention(legacy_digest),
            legacy_intervention,
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            receipt_count = connection.execute(
                "SELECT COUNT(*) FROM realized_intervention_receipts"
            ).fetchone()[0]
        self.assertEqual(receipt_count, 0)

    def test_prospective_decision_and_executor_receipt_round_trip(self) -> None:
        before = torch.zeros((3, 2, 2), dtype=torch.float64)
        delta = torch.ones_like(before)
        after = before + delta
        masks = torch.ones_like(before, dtype=torch.bool)
        before_run, before_context, plan_digest, _ = _prospective_run_and_context(
            before,
            masks,
        )
        after_run, after_context, _, _ = _prospective_run_and_context(after, masks)
        action_generator = InterventionActionGenerator.from_model(
            _AddOneAction().eval(),
            before_context,
            intervention_type="realized_sensor_correction",
        )
        action_policy = ReusableInterventionPolicyEvidence(
            policy_id="policy-1",
            action_generator_digest=action_generator.generator_digest,
            context_schema_digest=before_context.context_schema_digest,
            applicability_region_digest=(
                before_context.applicability_region_digest
            ),
            execution_policy_digest="e" * 64,
            allowed_intervention_types=("realized_sensor_correction",),
            maximum_absolute_delta_dbz=2.0,
            maximum_changed_fraction=1.0,
            validation_evidence_digests=("d" * 64,),
        )
        contextual_generator = InterventionActionGenerator.from_model(
            _NonpositiveOnlyAction().eval(),
            before_context,
            intervention_type="realized_sensor_correction",
        )
        contextual_policy = replace(
            action_policy,
            action_generator_digest=contextual_generator.generator_digest,
        )
        positive = torch.ones_like(before)
        positive_run, positive_context, _, _ = _prospective_run_and_context(
            positive,
            torch.ones_like(positive, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "outside the action policy"):
            ProspectiveInterventionDecision.from_policy(
                contextual_policy,
                action_generator=contextual_generator,
                decision_id="inapplicable-context",
                case_id="case-1",
                radar_id="radar-1",
                intervention_type="realized_sensor_correction",
                actual_input_context=positive_context,
                actual_input_before_run=positive_run,
                input_plan_digest=plan_digest,
                decision_basis_digest="d" * 64,
                decision_policy_digest="e" * 64,
                decision_trust_store_digest="f" * 64,
                decided_at="2026-08-08T00:22:00Z",
                observation_valid_time="2026-08-08T00:20:00Z",
                input_available_time="2026-08-08T00:21:00Z",
                decision_deadline="2099-08-08T00:30:00Z",
                publication_time="2099-08-08T01:00:00Z",
            )
        with self.assertRaisesRegex(ValueError, "outside the reusable policy"):
            ProspectiveInterventionDecision.from_policy(
                action_policy,
                action_generator=InterventionActionGenerator.from_model(
                    _AddTwoAction().eval(),
                    before_context,
                    intervention_type="realized_sensor_correction",
                ),
                decision_id="wrong-generator",
                case_id="case-1",
                radar_id="radar-1",
                intervention_type="realized_sensor_correction",
                actual_input_context=before_context,
                actual_input_before_run=before_run,
                input_plan_digest=plan_digest,
                decision_basis_digest="d" * 64,
                decision_policy_digest="e" * 64,
                decision_trust_store_digest="f" * 64,
                decided_at="2026-08-08T00:22:00Z",
                observation_valid_time="2026-08-08T00:20:00Z",
                input_available_time="2026-08-08T00:21:00Z",
                decision_deadline="2099-08-08T00:30:00Z",
                publication_time="2099-08-08T01:00:00Z",
            )
        decision = ProspectiveInterventionDecision.from_policy(
            action_policy,
            action_generator=action_generator,
            decision_id="decision-1",
            case_id="case-1",
            radar_id="radar-1",
            intervention_type="realized_sensor_correction",
            actual_input_context=before_context,
            actual_input_before_run=before_run,
            input_plan_digest=plan_digest,
            decision_basis_digest="d" * 64,
            decision_policy_digest="e" * 64,
            decision_trust_store_digest="f" * 64,
            decided_at="2026-08-08T00:22:00Z",
            observation_valid_time="2026-08-08T00:20:00Z",
            input_available_time="2026-08-08T00:21:00Z",
            decision_deadline="2099-08-08T00:30:00Z",
            publication_time="2099-08-08T01:00:00Z",
        )
        safety = json.loads(decision.action_safety_diagnostics_json)
        self.assertEqual(safety["changed_pixel_count"], delta.numel())
        self.assertEqual(
            json_digest(safety),
            decision.action_safety_diagnostics_digest,
        )
        executor_private = Ed25519PrivateKey.generate()
        executor_public = executor_private.public_key()
        trust = SimpleNamespace(
            content_digest="f" * 64,
            approved_policy_digests=frozenset(
                ("e" * 64, action_policy.policy_digest)
            ),
        )
        executor_trust = SimpleNamespace(
            keys={"executor-1": executor_public},
            content_digest="0" * 64,
        )
        operator_approval, operator_trust, operator_private = _operator_approval(
            decision
        )
        with patch(
            "advar.ledger._load_learning_policy_trust_store", return_value=trust
        ), patch(
            "advar.ledger._load_executor_trust_store", return_value=executor_trust
        ), patch(
            "advar.ledger._load_operator_trust_store", return_value=operator_trust
        ):
            forged = copy(decision)
            object.__setattr__(forged, "action_payload_digest", "9" * 64)
            object.__setattr__(
                forged,
                "action_digest",
                json_digest(
                    {
                        "contract": "generated-radar-action-v2",
                        "action_policy_digest": forged.action_policy_digest,
                        "action_generator_digest": forged.action_generator_digest,
                        "action_context_digest": forged.action_context_digest,
                        "action_payload_digest": forged.action_payload_digest,
                        "action_application_contract_digest": (
                            forged.action_application_contract_digest
                        ),
                        "action_safety_diagnostics_digest": (
                            forged.action_safety_diagnostics_digest
                        ),
                    }
                ),
            )
            object.__setattr__(
                forged,
                "decision_digest",
                json_digest(
                    {
                        key: value
                        for key, value in forged.__dict__.items()
                        if key != "decision_digest"
                    }
                ),
            )
            with self.assertRaisesRegex(ValueError, "policy output"):
                self.ledger.append_prospective_intervention_decision(
                    forged,
                    operator_approval=operator_approval,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    actual_input_before_context=before_context,
                    actual_input_before_run=before_run,
                    trust_store_path="/etc/advar/policies.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            other_decision = copy(decision)
            object.__setattr__(other_decision, "decision_digest", "9" * 64)
            other_approval = OperatorActionApproval.from_decision(
                other_decision,
                operator_key_id="operator-1",
                operator_role="duty-meteorologist",
                operator_trust_store_digest=operator_trust.content_digest,
                operator_private_key=operator_private,
                reviewed_at=decision.decided_at,
                expires_at=decision.decision_deadline,
                operator_comment_digest="2" * 64,
            )
            with self.assertRaisesRegex(ValueError, "another decision"):
                self.ledger.append_prospective_intervention_decision(
                    decision,
                    operator_approval=other_approval,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    actual_input_before_context=before_context,
                    actual_input_before_run=before_run,
                    trust_store_path="/etc/advar/policies.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            forged_approval = copy(operator_approval)
            object.__setattr__(forged_approval, "operator_signature", "0" * 128)
            object.__setattr__(
                forged_approval,
                "approval_digest",
                json_digest(
                    {
                        key: value
                        for key, value in forged_approval.__dict__.items()
                        if key != "approval_digest"
                    }
                ),
            )
            with self.assertRaisesRegex(ValueError, "operator.*signature"):
                self.ledger.append_prospective_intervention_decision(
                    decision,
                    operator_approval=forged_approval,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    actual_input_before_context=before_context,
                    actual_input_before_run=before_run,
                    trust_store_path="/etc/advar/policies.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            self.ledger.append_prospective_intervention_decision(
                decision,
                operator_approval=operator_approval,
                action_policy=action_policy,
                action_generator=action_generator,
                actual_input_before_context=before_context,
                actual_input_before_run=before_run,
                trust_store_path="/etc/advar/policies.json",
                operator_trust_store_path="/etc/advar/operators.json",
            )
            applied = datetime.now(timezone.utc).isoformat()
            receipt = RealizedInterventionReceipt.from_decision(
                decision,
                actual_input_before_context=before_context,
                actual_input_before_run=before_run,
                actual_input_after_context=after_context,
                actual_input_after_run=after_run,
                action_policy=action_policy,
                action_generator=action_generator,
                executor_key_id="executor-1",
                executor_trust_store_digest=executor_trust.content_digest,
                executor_private_key=executor_private,
                executor_sequence_number=1,
                applied_time=applied,
                receipt_time=applied,
            )
            changed_quality = torch.full_like(before, 0.25)
            changed_std = torch.full_like(before, 0.5)
            changed_before_run, changed_before_context, _, _ = (
                _prospective_run_and_context(
                    before,
                    masks,
                    quality_weight=changed_quality,
                    observation_std_dbz=changed_std,
                )
            )
            changed_after_run, changed_after_context, _, _ = (
                _prospective_run_and_context(
                    after,
                    masks,
                    quality_weight=changed_quality,
                    observation_std_dbz=changed_std,
                )
            )
            with self.assertRaisesRegex(ValueError, "before-context"):
                RealizedInterventionReceipt.from_decision(
                    decision,
                    actual_input_before_context=changed_before_context,
                    actual_input_before_run=changed_before_run,
                    actual_input_after_context=changed_after_context,
                    actual_input_after_run=changed_after_run,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    executor_key_id="executor-1",
                    executor_trust_store_digest=executor_trust.content_digest,
                    executor_private_key=executor_private,
                    executor_sequence_number=2,
                    applied_time=applied,
                    receipt_time=applied,
                )
            changed_applicability = torch.ones_like(masks)
            changed_applicability[0, 0, 0] = False
            changed_applicability_context = InterventionInputContext.from_inputs(
                frames_dbz=before,
                observation_masks=masks,
                quality_weight=masks.to(before),
                observation_std_dbz=torch.full_like(before, 2.0),
                background_frames_dbz=None,
                radar_id="radar-1",
                applicability_mask=changed_applicability,
                run=before_run,
            )
            with self.assertRaisesRegex(ValueError, "before-context"):
                RealizedInterventionReceipt.from_decision(
                    decision,
                    actual_input_before_context=changed_applicability_context,
                    actual_input_before_run=before_run,
                    actual_input_after_context=after_context,
                    actual_input_after_run=after_run,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    executor_key_id="executor-1",
                    executor_trust_store_digest=executor_trust.content_digest,
                    executor_private_key=executor_private,
                    executor_sequence_number=2,
                    applied_time=applied,
                    receipt_time=applied,
                )
            with self.assertRaisesRegex(ValueError, "approved action result"):
                RealizedInterventionReceipt.from_decision(
                    decision,
                    actual_input_before_context=before_context,
                    actual_input_before_run=before_run,
                    actual_input_after_context=before_context,
                    actual_input_after_run=before_run,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    executor_key_id="executor-1",
                    executor_trust_store_digest=executor_trust.content_digest,
                    executor_private_key=executor_private,
                    executor_sequence_number=2,
                    applied_time=applied,
                    receipt_time=applied,
                )
            other_run = replace(
                after_run,
                input_plan_digest="9" * 64,
            )
            with self.assertRaisesRegex(ValueError, "input plan"):
                RealizedInterventionReceipt.from_decision(
                    decision,
                    actual_input_before_context=before_context,
                    actual_input_before_run=before_run,
                    actual_input_after_context=after_context,
                    actual_input_after_run=other_run,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    executor_key_id="executor-1",
                    executor_trust_store_digest=executor_trust.content_digest,
                    executor_private_key=executor_private,
                    executor_sequence_number=2,
                    applied_time=applied,
                    receipt_time=applied,
                )
            changed_mask_run, changed_mask_context, _, _ = (
                _prospective_run_and_context(
                after,
                torch.zeros_like(after, dtype=torch.bool),
                )
            )
            with self.assertRaisesRegex(ValueError, "after-QC|non-radar input"):
                RealizedInterventionReceipt.from_decision(
                    decision,
                    actual_input_before_context=before_context,
                    actual_input_before_run=before_run,
                    actual_input_after_context=changed_mask_context,
                    actual_input_after_run=changed_mask_run,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    executor_key_id="executor-1",
                    executor_trust_store_digest=executor_trust.content_digest,
                    executor_private_key=executor_private,
                    executor_sequence_number=2,
                    applied_time=applied,
                    receipt_time=applied,
                )
            with patch.object(
                ledger_module,
                "_MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES",
                1,
            ), self.assertRaisesRegex(ValueError, "expanded-byte budget"):
                self.ledger.append_realized_intervention_receipt(
                    decision,
                    receipt,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    actual_input_before_context=before_context,
                    actual_input_before_run=before_run,
                    actual_input_after_context=after_context,
                    actual_input_after_run=after_run,
                    trust_store_path="/etc/advar/policies.json",
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            digest = self.ledger.append_realized_intervention_receipt(
                decision,
                receipt,
                action_policy=action_policy,
                action_generator=action_generator,
                actual_input_before_context=before_context,
                actual_input_before_run=before_run,
                actual_input_after_context=after_context,
                actual_input_after_run=after_run,
                trust_store_path="/etc/advar/policies.json",
                executor_trust_store_path="/etc/advar/executors.json",
                operator_trust_store_path="/etc/advar/operators.json",
            )

            tampered = copy(receipt)
            object.__setattr__(
                tampered,
                "executor_signature",
                "0" * 128,
            )
            object.__setattr__(
                tampered,
                "receipt_digest",
                json_digest(
                    {
                        key: value
                        for key, value in tampered.__dict__.items()
                        if key != "receipt_digest"
                    }
                ),
            )
            with self.assertRaisesRegex(ValueError, "signature"):
                self.ledger.append_realized_intervention_receipt(
                    decision,
                    tampered,
                    action_policy=action_policy,
                    action_generator=action_generator,
                    actual_input_before_context=before_context,
                    actual_input_before_run=before_run,
                    actual_input_after_context=after_context,
                    actual_input_after_run=after_run,
                    trust_store_path="/etc/advar/policies.json",
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )

        with patch(
            "advar.ledger._load_executor_trust_store",
            return_value=executor_trust,
        ), patch(
            "advar.ledger._load_operator_trust_store",
            return_value=operator_trust,
        ):
            self.assertEqual(
                self.ledger.load_prospective_intervention(
                    digest,
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                ),
                (decision, receipt, operator_approval),
            )
            expired_approval = OperatorActionApproval.from_decision(
                decision,
                operator_key_id="operator-1",
                operator_role="duty-meteorologist",
                operator_trust_store_digest=operator_trust.content_digest,
                operator_private_key=operator_private,
                reviewed_at=decision.decided_at,
                expires_at=receipt.applied_time,
                operator_comment_digest="2" * 64,
            )
            with sqlite3.connect(self.ledger.index_path) as connection:
                retained_triggers = tuple(
                    connection.execute(
                        "SELECT name, sql FROM sqlite_master "
                        "WHERE type = 'trigger' AND "
                        "tbl_name = 'prospective_intervention_decisions'"
                    )
                )
                for name, _ in retained_triggers:
                    connection.execute(f'DROP TRIGGER "{name}"')
                connection.execute(
                    "UPDATE prospective_intervention_decisions SET "
                    "operator_approval_digest = ?, operator_approval_json = ? "
                    "WHERE decision_digest = ?",
                    (
                        expired_approval.approval_digest,
                        json.dumps(asdict(expired_approval), sort_keys=True),
                        decision.decision_digest,
                    ),
                )
            with self.assertRaisesRegex(ValueError, "time order"):
                self.ledger.load_prospective_intervention(
                    digest,
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            with sqlite3.connect(self.ledger.index_path) as connection:
                connection.execute(
                    "UPDATE prospective_intervention_decisions SET "
                    "operator_approval_digest = ?, operator_approval_json = ? "
                    "WHERE decision_digest = ?",
                    (
                        operator_approval.approval_digest,
                        json.dumps(asdict(operator_approval), sort_keys=True),
                        decision.decision_digest,
                    ),
                )
                for _, sql in retained_triggers:
                    assert isinstance(sql, str)
                    connection.execute(sql)
            artifact_dir = self.ledger.interventions_dir / digest
            manifest_path = artifact_dir / "manifest.json"
            checksums_path = artifact_dir / "checksums.json"
            original_manifest = manifest_path.read_bytes()
            original_checksums = checksums_path.read_bytes()
            manifest = json.loads(original_manifest)
            self.assertEqual(
                manifest["contract"],
                "durable-intervention-action-artifact-v6",
            )
            self.assertEqual(
                manifest["metric_domain_evidence_digest"],
                CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest,
            )
            manifest["metric_domain_evidence_digest"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            checksums = json.loads(original_checksums)
            checksums["manifest.json"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            checksums_path.write_text(
                json.dumps(checksums, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "evidence is not current"):
                self.ledger.load_prospective_intervention(
                    digest,
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            manifest = json.loads(original_manifest)
            manifest["before_data_identity_digest"] = "8" * 64
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            checksums = json.loads(original_checksums)
            checksums["manifest.json"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            checksums_path.write_text(
                json.dumps(checksums, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "input bundle|fixed input context",
            ):
                self.ledger.load_prospective_intervention(
                    digest,
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            manifest_path.write_bytes(original_manifest)
            checksums_path.write_bytes(original_checksums)
            with patch.object(
                ledger_module,
                "_MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES",
                1,
            ), self.assertRaisesRegex(ValueError, "archive is too large"):
                self.ledger.load_prospective_intervention(
                    digest,
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            unexpected = artifact_dir / "unexpected.bin"
            unexpected.write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "artifact members"):
                self.ledger.load_prospective_intervention(
                    digest,
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )
            unexpected.unlink()
            artifact = self.ledger.interventions_dir / digest / "generator.pt2"
            artifact.write_bytes(artifact.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.ledger.load_prospective_intervention(
                    digest,
                    executor_trust_store_path="/etc/advar/executors.json",
                    operator_trust_store_path="/etc/advar/operators.json",
                )

    def test_typed_actions_plan_times_and_global_radius_fail_closed(self) -> None:
        frames = torch.zeros((3, 2, 2), dtype=torch.float64)
        masks = torch.ones_like(frames, dtype=torch.bool)
        run, context, plan_digest, _ = _prospective_run_and_context(frames, masks)
        dbz_generator = InterventionActionGenerator.from_model(
            _AddOneAction().eval(),
            context,
            intervention_type="realized_sensor_correction",
        )
        strict_policy = ReusableInterventionPolicyEvidence(
            policy_id="strict-radius",
            action_generator_digest=dbz_generator.generator_digest,
            context_schema_digest=context.context_schema_digest,
            applicability_region_digest=context.applicability_region_digest,
            execution_policy_digest="e" * 64,
            allowed_intervention_types=("realized_sensor_correction",),
            maximum_absolute_delta_dbz=2.0,
            validation_evidence_digests=("d" * 64,),
        )
        common = dict(
            action_generator=dbz_generator,
            decision_id="strict",
            case_id="case-1",
            radar_id="radar-1",
            intervention_type="realized_sensor_correction",
            actual_input_context=context,
            actual_input_before_run=run,
            input_plan_digest=plan_digest,
            decision_basis_digest="d" * 64,
            decision_policy_digest="e" * 64,
            decision_trust_store_digest="f" * 64,
            decided_at="2026-08-08T00:22:00Z",
            observation_valid_time="2026-08-08T00:20:00Z",
            input_available_time="2026-08-08T00:21:00Z",
            decision_deadline="2099-08-08T00:30:00Z",
            publication_time="2099-08-08T01:00:00Z",
        )
        with self.assertRaisesRegex(ValueError, "changed-fraction"):
            ProspectiveInterventionDecision.from_policy(strict_policy, **common)
        with self.assertRaisesRegex(ValueError, "observation errors"):
            InterventionInputContext.from_inputs(
                frames_dbz=frames,
                observation_masks=masks,
                quality_weight=masks.to(frames),
                observation_std_dbz=torch.full_like(frames, 100.0),
                background_frames_dbz=None,
                radar_id="radar-1",
                applicability_mask=torch.ones_like(masks),
                run=run,
            )
        restricted_mask = torch.zeros_like(masks)
        restricted_mask[0, 0, 0] = True
        restricted_context = InterventionInputContext.from_inputs(
            frames_dbz=frames,
            observation_masks=masks,
            quality_weight=masks.to(frames),
            observation_std_dbz=torch.full_like(frames, 2.0),
            background_frames_dbz=None,
            radar_id="radar-1",
            applicability_mask=restricted_mask,
            run=run,
        )
        restricted_policy = replace(
            strict_policy,
            applicability_region_digest=(
                restricted_context.applicability_region_digest
            ),
            maximum_changed_fraction=1.0,
        )
        with self.assertRaisesRegex(ValueError, "applicability region"):
            ProspectiveInterventionDecision.from_policy(
                restricted_policy,
                **{**common, "actual_input_context": restricted_context},
            )
        with self.assertRaisesRegex(ValueError, "decision radar"):
            ProspectiveInterventionDecision.from_policy(
                replace(strict_policy, maximum_changed_fraction=1.0),
                **{**common, "radar_id": "radar-2"},
            )
        with self.assertRaisesRegex(ValueError, "current-case benefit"):
            replace(strict_policy, execution_authority="automatic")
        relaxed = replace(strict_policy, maximum_changed_fraction=1.0)
        correlated_run = replace(
            run,
            analysis_config_json=json.dumps(
                {
                    "observation_common_bias_std_dbz": 1.0,
                }
            ),
        )
        with self.assertRaisesRegex(ValueError, "diagonal observation error"):
            _validate_action_safety(
                dbz_generator.generate(context),
                context,
                correlated_run,
                relaxed,
            )
        with self.assertRaisesRegex(ValueError, "resolved input plan"):
            ProspectiveInterventionDecision.from_policy(
                relaxed,
                **{**common, "publication_time": "2099-08-08T02:00:00Z"},
            )
        qc_generator = InterventionActionGenerator.from_model(
            _RejectAllQcAction().eval(),
            context,
            intervention_type="realized_qc_intervention",
            action_reason="clutter",
        )
        with self.assertRaisesRegex(ValueError, "generator type"):
            ProspectiveInterventionDecision.from_policy(
                replace(relaxed, action_generator_digest=qc_generator.generator_digest),
                **{**common, "action_generator": qc_generator},
            )
        qc_policy = replace(
            relaxed,
            action_generator_digest=qc_generator.generator_digest,
            allowed_intervention_types=("realized_qc_intervention",),
            maximum_global_quality_precision_scale_l2=4.0,
            maximum_tile_quality_precision_scale_l2=4.0,
        )
        qc_action = qc_generator.generate(context)
        self.assertIsInstance(qc_action, QcMaskAction)
        qc_decision = ProspectiveInterventionDecision.from_policy(
            qc_policy,
            **{
                **common,
                "action_generator": qc_generator,
                "intervention_type": "realized_qc_intervention",
            },
        )
        self.assertNotEqual(
            qc_decision.action_application_contract_digest,
            json_digest({"contract": "radar-dbz-correction-action-v2"}),
        )
        rejected_masks = torch.zeros_like(masks)
        qc_after_run, qc_after_context, _, _ = _prospective_run_and_context(
            frames,
            rejected_masks,
        )
        receipt_time = datetime.now(timezone.utc).isoformat()
        qc_receipt = RealizedInterventionReceipt.from_decision(
            qc_decision,
            actual_input_before_context=context,
            actual_input_before_run=run,
                actual_input_after_context=qc_after_context,
                actual_input_after_run=qc_after_run,
                action_policy=qc_policy,
                action_generator=qc_generator,
            executor_key_id="executor-qc",
            executor_trust_store_digest="0" * 64,
            executor_private_key=Ed25519PrivateKey.generate(),
            executor_sequence_number=1,
            applied_time=receipt_time,
            receipt_time=receipt_time,
        )
        self.assertEqual(
            qc_receipt.actual_input_before_frames_digest,
            qc_receipt.actual_input_after_frames_digest,
        )
        self.assertNotEqual(
            qc_receipt.actual_input_before_masks_digest,
            qc_receipt.actual_input_after_masks_digest,
        )

        deweight_generator = InterventionActionGenerator.from_model(
            _DeweightOnlyQcAction().eval(),
            context,
            intervention_type="realized_qc_intervention",
            action_reason="confidence deweighting",
        )
        deweight_policy = replace(
            qc_policy,
            action_generator_digest=deweight_generator.generator_digest,
        )
        deweight_decision = ProspectiveInterventionDecision.from_policy(
            deweight_policy,
            **{
                **common,
                "decision_id": "quality-only",
                "action_generator": deweight_generator,
                "intervention_type": "realized_qc_intervention",
            },
        )
        deweighted_quality = torch.full_like(frames, 0.5)
        deweighted_run, deweighted_context, _, _ = _prospective_run_and_context(
            frames,
            masks,
            quality_weight=deweighted_quality,
        )
        deweight_receipt = RealizedInterventionReceipt.from_decision(
            deweight_decision,
            actual_input_before_context=context,
            actual_input_before_run=run,
            actual_input_after_context=deweighted_context,
            actual_input_after_run=deweighted_run,
            action_policy=deweight_policy,
            action_generator=deweight_generator,
            executor_key_id="executor-quality",
            executor_trust_store_digest="0" * 64,
            executor_private_key=Ed25519PrivateKey.generate(),
            executor_sequence_number=1,
            applied_time=receipt_time,
            receipt_time=receipt_time,
        )
        self.assertEqual(
            deweight_receipt.actual_input_before_bundle_digest,
            deweight_receipt.actual_input_bundle_digest,
        )
        self.assertNotEqual(
            deweight_receipt.full_analysis_input_before_digest,
            deweight_receipt.full_analysis_input_after_digest,
        )
        self.assertNotEqual(
            run.input_plan_resolution_digest,
            deweighted_run.input_plan_resolution_digest,
        )

        low_quality = torch.full_like(frames, 0.01)
        low_std = torch.full_like(frames, 0.1)
        low_run, low_context, low_plan, _ = _prospective_run_and_context(
            frames,
            masks,
            quality_weight=low_quality,
            observation_std_dbz=low_std,
        )
        reject_low_generator = InterventionActionGenerator.from_model(
            _RejectAllQcAction().eval(),
            low_context,
            intervention_type="realized_qc_intervention",
            action_reason="low-confidence rejection",
        )
        reject_low_policy = replace(
            qc_policy,
            action_generator_digest=reject_low_generator.generator_digest,
            maximum_global_quality_precision_scale_l2=1.0,
            maximum_tile_quality_precision_scale_l2=1.0,
        )
        with self.assertRaisesRegex(ValueError, "precision-scale"):
            ProspectiveInterventionDecision.from_policy(
                reject_low_policy,
                **{
                    **common,
                    "decision_id": "precision-radius",
                    "action_generator": reject_low_generator,
                    "actual_input_context": low_context,
                    "actual_input_before_run": low_run,
                    "input_plan_digest": low_plan,
                    "intervention_type": "realized_qc_intervention",
                },
            )

        quarter_quality = torch.full_like(frames, 0.25)
        quarter_run, quarter_context, quarter_plan, _ = (
            _prospective_run_and_context(
                frames,
                masks,
                quality_weight=quarter_quality,
            )
        )
        upweight_generator = InterventionActionGenerator.from_model(
            _UpweightQcAction().eval(),
            quarter_context,
            intervention_type="realized_qc_intervention",
            action_reason="untrusted restoration",
        )
        with self.assertRaisesRegex(ValueError, "reject or deweight"):
            ProspectiveInterventionDecision.from_policy(
                replace(
                    qc_policy,
                    action_generator_digest=upweight_generator.generator_digest,
                ),
                **{
                    **common,
                    "decision_id": "qc-upweight",
                    "action_generator": upweight_generator,
                    "actual_input_context": quarter_context,
                    "actual_input_before_run": quarter_run,
                    "input_plan_digest": quarter_plan,
                    "intervention_type": "realized_qc_intervention",
                },
            )
        override_generator = InterventionActionGenerator.from_model(
            _SinglePixelOverrideAction().eval(),
            context,
            intervention_type="operator_override",
            action_reason="operator-confirmed artifact",
        )
        override_action = override_generator.generate(context)
        self.assertIsInstance(override_action, OperatorOverrideAction)
        override_policy = replace(
            relaxed,
            action_generator_digest=override_generator.generator_digest,
            allowed_intervention_types=("operator_override",),
        )
        override_decision = ProspectiveInterventionDecision.from_policy(
            override_policy,
            **{
                **common,
                "action_generator": override_generator,
                "intervention_type": "operator_override",
            },
        )
        self.assertNotEqual(
            override_decision.action_application_contract_digest,
            qc_decision.action_application_contract_digest,
        )

        maximum = torch.full_like(frames, NowcastConfig().max_dbz)
        maximum_run, maximum_context, maximum_plan, _ = (
            _prospective_run_and_context(maximum, masks)
        )
        with self.assertRaisesRegex(ValueError, "input clamp"):
            ProspectiveInterventionDecision.from_policy(
                relaxed,
                **{
                    **common,
                    "actual_input_context": maximum_context,
                    "actual_input_before_run": maximum_run,
                    "input_plan_digest": maximum_plan,
                },
            )

        missing = frames.clone()
        missing[0, 0, 0] = float("nan")
        missing_masks = masks.clone()
        missing_masks[0, 0, 0] = False
        missing_run, _, missing_plan, _ = _prospective_run_and_context(
            missing,
            missing_masks,
        )
        missing_context = InterventionInputContext.from_inputs(
            frames_dbz=missing,
            observation_masks=missing_masks,
            quality_weight=missing_masks.to(missing),
            observation_std_dbz=torch.full_like(missing, 2.0),
            background_frames_dbz=None,
            radar_id="radar-1",
            applicability_mask=torch.ones_like(missing_masks),
            run=missing_run,
        )
        for nonfinite in (float("inf"), float("-inf")):
            alternative = frames.clone()
            alternative[0, 0, 0] = nonfinite
            alternative_run, alternative_context, _, _ = (
                _prospective_run_and_context(alternative, missing_masks)
            )
            self.assertTrue(
                torch.equal(
                    missing_context.generator_tensor(),
                    alternative_context.generator_tensor(),
                )
            )
            self.assertEqual(
                missing_context.canonicalization_contract_digest,
                alternative_context.canonicalization_contract_digest,
            )
        with self.assertRaisesRegex(ValueError, "invalid observations"):
            ProspectiveInterventionDecision.from_policy(
                relaxed,
                **{
                    **common,
                    "actual_input_context": missing_context,
                    "actual_input_before_run": missing_run,
                    "input_plan_digest": missing_plan,
                },
            )
        valid_only_generator = InterventionActionGenerator.from_model(
            _ValidOnlyAddAction().eval(),
            missing_context,
            intervention_type="realized_sensor_correction",
        )
        valid_only_policy = replace(
            relaxed,
            action_generator_digest=valid_only_generator.generator_digest,
        )
        valid_only_decision = ProspectiveInterventionDecision.from_policy(
            valid_only_policy,
            **{
                **common,
                "decision_id": "nan-safe",
                "action_generator": valid_only_generator,
                "actual_input_context": missing_context,
                "actual_input_before_run": missing_run,
                "input_plan_digest": missing_plan,
            },
        )
        missing_after = missing + missing_masks.to(missing)
        missing_after_run, missing_after_context, _, _ = (
            _prospective_run_and_context(missing_after, missing_masks)
        )
        RealizedInterventionReceipt.from_decision(
            valid_only_decision,
            actual_input_before_context=missing_context,
            actual_input_before_run=missing_run,
            actual_input_after_context=missing_after_context,
            actual_input_after_run=missing_after_run,
            action_policy=valid_only_policy,
            action_generator=valid_only_generator,
            executor_key_id="executor-nan",
            executor_trust_store_digest="0" * 64,
            executor_private_key=Ed25519PrivateKey.generate(),
            executor_sequence_number=1,
            applied_time=datetime.now(timezone.utc).isoformat(),
            receipt_time=datetime.now(timezone.utc).isoformat(),
        )

    def test_retrospective_replay_is_a_separate_audit_record(self) -> None:
        replay = RetrospectiveCounterfactualReplay(
            learning_result_digest="1" * 64,
            perturbation_digest="2" * 64,
            nominal_forecast_digest="3" * 64,
            replayed_at="2026-08-08T00:00:00Z",
        )
        self.ledger.append_retrospective_counterfactual_replay(replay)
        with sqlite3.connect(self.ledger.index_path) as connection:
            replays = connection.execute(
                "SELECT COUNT(*) FROM retrospective_counterfactual_replays"
            ).fetchone()[0]
            receipts = connection.execute(
                "SELECT COUNT(*) FROM realized_intervention_receipts"
            ).fetchone()[0]
        self.assertEqual((replays, receipts), (1, 0))
        self.assertEqual(
            self.ledger.load_retrospective_counterfactual_replay(
                replay.replay_digest
            ),
            replay,
        )

    def test_legacy_holdout_plan_loads_as_read_only_audit(self) -> None:
        audit = LegacyNeuralPriorHoldoutPlanAudit(
            plan_id="legacy-plan",
            parent_prior_digest="1" * 64,
            candidate_family_digests=("2" * 64,),
            cases=(
                LegacyNeuralPriorHoldoutPlanCase(
                    case_id="case-1",
                    storm_id="storm-1",
                    day="2026-08-08",
                    radar_id="radar-1",
                    regime="convective",
                    range_regime="near",
                    input_bundle_digest="3" * 64,
                    verification_plan_digest="4" * 64,
                    metric_contract_digest="5" * 64,
                    issue_time="2026-08-08T00:00:00Z",
                ),
            ),
            registered_at="2026-08-07T00:00:00Z",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_holdout_plans "
                "(plan_digest, plan_id, plan_json, policy_digest, "
                "trust_store_digest, registered_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    audit.plan_digest,
                    audit.plan_id,
                    json.dumps(asdict(audit), sort_keys=True),
                    "6" * 64,
                    "7" * 64,
                    audit.registered_at,
                    audit.registered_at,
                ),
            )

        loaded = self.ledger.load_neural_prior_holdout_plan(audit.plan_digest)

        self.assertIsInstance(loaded, LegacyNeuralPriorHoldoutPlanAudit)
        self.assertEqual(loaded, audit)

        input_plan = NeuralPriorInputPlan(
            valid_times=("2026-08-08T00:00:00Z",),
            grid_contract_digest="8" * 64,
            radar_product_digest="9" * 64,
            qc_pipeline_digest="a" * 64,
            background_cycle_rule_digest="b" * 64,
            mask_policy_digest="c" * 64,
            observation_valid_time="2026-08-08T00:00:00Z",
            input_available_time="2026-08-08T00:01:00Z",
            decision_deadline="2026-08-08T00:02:00Z",
            publication_time="2026-08-08T00:05:00Z",
        )
        audit_v2 = LegacyNeuralPriorHoldoutPlanV2Audit(
            plan_id="legacy-v2-plan",
            parent_prior_digest="d" * 64,
            candidate_family_digests=("e" * 64,),
            cases=(
                LegacyNeuralPriorHoldoutPlanV2Case(
                    case_id="case-v2",
                    storm_id="storm-v2",
                    day="2026-08-08",
                    radar_id="radar-v2",
                    regime="convective",
                    range_regime="near",
                    input_plan_digest=input_plan.plan_digest,
                    verification_plan_digest="f" * 64,
                    metric_contract_digest="0" * 64,
                    issue_time="2026-08-08T00:00:00Z",
                ),
            ),
            input_plans=(input_plan,),
            registered_at="2026-08-07T00:00:00Z",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_holdout_plans "
                "(plan_digest, plan_id, plan_json, policy_digest, "
                "trust_store_digest, registered_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    audit_v2.plan_digest,
                    audit_v2.plan_id,
                    json.dumps(asdict(audit_v2), sort_keys=True),
                    "1" * 64,
                    "2" * 64,
                    audit_v2.registered_at,
                    audit_v2.registered_at,
                ),
            )
        loaded_v2 = self.ledger.load_neural_prior_holdout_plan(
            audit_v2.plan_digest
        )
        self.assertIsInstance(loaded_v2, LegacyNeuralPriorHoldoutPlanV2Audit)
        self.assertEqual(loaded_v2, audit_v2)

        target_v1: dict[str, object] = {
            "contract": "prior-uncertainty-target-plan-v1",
            "plan_id": "legacy-target-v1",
            "target_kind": "independent_sensor",
            "source_identity_digest": "3" * 64,
            "qc_pipeline_digest": "4" * 64,
            "feature_exclusion_contract_digest": "5" * 64,
            "independence_evidence_digest": "6" * 64,
            "target_valid_time": "2026-08-08T01:00:00Z",
        }
        audit_v3 = LegacyNeuralPriorHoldoutPlanV3Audit(
            plan_id="legacy-v3-plan",
            parent_prior_digest="7" * 64,
            candidate_family_digests=("8" * 64,),
            cases=(
                LegacyNeuralPriorHoldoutPlanV3Case(
                    case_id="case-v3",
                    storm_id="storm-v3",
                    day="2026-08-08",
                    radar_id="radar-v3",
                    regime="convective",
                    range_regime="near",
                    input_plan_digest=input_plan.plan_digest,
                    verification_plan_digest="9" * 64,
                    metric_contract_digest="a" * 64,
                    uncertainty_target_plan_digest=json_digest(target_v1),
                    issue_time="2026-08-08T00:00:00Z",
                ),
            ),
            input_plans=(input_plan,),
            uncertainty_target_plans=(target_v1,),
            registered_at="2026-08-07T00:00:00Z",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_holdout_plans "
                "(plan_digest, plan_id, plan_json, policy_digest, "
                "trust_store_digest, registered_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    audit_v3.plan_digest,
                    audit_v3.plan_id,
                    json.dumps(asdict(audit_v3), sort_keys=True),
                    "b" * 64,
                    "c" * 64,
                    audit_v3.registered_at,
                    audit_v3.registered_at,
                ),
            )
        loaded_v3 = self.ledger.load_neural_prior_holdout_plan(
            audit_v3.plan_digest
        )
        self.assertIsInstance(loaded_v3, LegacyNeuralPriorHoldoutPlanV3Audit)
        self.assertEqual(loaded_v3, audit_v3)

        target_v2: dict[str, object] = {
            "contract": "prior-uncertainty-target-plan-v2",
            "plan_id": "legacy-target-v2",
            "target_kind": "independent_sensor",
            "source_identity_digest": "d" * 64,
            "qc_pipeline_digest": "e" * 64,
            "grid_contract_digest": "8" * 64,
            "feature_exclusion_contract_digest": "f" * 64,
            "independence_evidence_digest": "0" * 64,
            "target_valid_time": "2026-08-08T01:00:00Z",
            "support_threshold_dbz": 5.0,
        }
        target_v2_digest = json_digest(target_v2)
        audit_v4 = LegacyNeuralPriorHoldoutPlanV4Audit(
            plan_id="legacy-v4-plan",
            parent_prior_digest="1" * 64,
            candidate_family_digests=("2" * 64,),
            cases=(
                LegacyNeuralPriorHoldoutPlanV3Case(
                    case_id="case-v4",
                    storm_id="storm-v4",
                    day="2026-08-08",
                    radar_id="radar-v4",
                    regime="clear",
                    range_regime="far",
                    input_plan_digest=input_plan.plan_digest,
                    verification_plan_digest="3" * 64,
                    metric_contract_digest="4" * 64,
                    uncertainty_target_plan_digest=target_v2_digest,
                    issue_time="2026-08-08T00:00:00Z",
                ),
            ),
            input_plans=(input_plan,),
            uncertainty_target_plans=(target_v2,),
            registered_at="2026-08-07T00:00:00Z",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_holdout_plans "
                "(plan_digest, plan_id, plan_json, policy_digest, "
                "trust_store_digest, registered_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    audit_v4.plan_digest,
                    audit_v4.plan_id,
                    json.dumps(asdict(audit_v4), sort_keys=True),
                    "5" * 64,
                    "6" * 64,
                    audit_v4.registered_at,
                    audit_v4.registered_at,
                ),
            )
        loaded_v4 = self.ledger.load_neural_prior_holdout_plan(
            audit_v4.plan_digest
        )
        self.assertIsInstance(loaded_v4, LegacyNeuralPriorHoldoutPlanV4Audit)
        self.assertEqual(loaded_v4, audit_v4)

        target_v4 = PriorUncertaintyTargetPlan(
            plan_id="legacy-target-v4",
            target_kind="independent_sensor",
            source_identity_digest="7" * 64,
            qc_pipeline_digest="a" * 64,
            mask_policy_digest="b" * 64,
            censor_policy_digest="c" * 64,
            floor_representation_contract_digest="d" * 64,
            grid_contract_digest="8" * 64,
            feature_exclusion_contract_digest="9" * 64,
            independence_evidence_digest="0" * 64,
            verification_observation_error_plan_digest="1" * 64,
            target_valid_time="2026-08-08T01:00:00Z",
            prior_probability_contract_digest="f" * 64,
        )
        legacy_case_v5 = LegacyNeuralPriorHoldoutPlanV3Case(
            case_id="case-v5",
            storm_id="storm-v5",
            day="2026-08-08",
            radar_id="radar-v5",
            regime="mixed",
            range_regime="middle",
            input_plan_digest=input_plan.plan_digest,
            verification_plan_digest="1" * 64,
            metric_contract_digest="2" * 64,
            uncertainty_target_plan_digest=target_v4.plan_digest,
            issue_time="2026-08-08T00:00:00Z",
        )
        normalized_v5: dict[str, object] = {
            "contract": "neural-prior-holdout-plan-v5",
            "plan_id": "legacy-v5-plan",
            "parent_prior_digest": "3" * 64,
            "candidate_family_digests": ["4" * 64],
            "cases": [asdict(legacy_case_v5)],
            "input_plans": [input_plan.payload],
            "uncertainty_target_plans": [target_v4.payload],
            "registered_at": "2026-08-07T00:00:00Z",
            "mode": "prospective",
            "sealed_historical_dataset_digest": None,
            "candidate_training_started_at": None,
        }
        v5_digest = json_digest(normalized_v5)
        stored_v5 = dict(normalized_v5)
        stored_v5["plan_digest"] = v5_digest
        stored_v5["input_plans"] = [asdict(input_plan)]
        stored_v5["uncertainty_target_plans"] = [asdict(target_v4)]
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_holdout_plans "
                "(plan_digest, plan_id, plan_json, policy_digest, "
                "trust_store_digest, registered_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    v5_digest,
                    "legacy-v5-plan",
                    json.dumps(stored_v5, sort_keys=True),
                    "5" * 64,
                    "6" * 64,
                    "2026-08-07T00:00:00Z",
                    "2026-08-07T00:00:00Z",
                ),
            )
        loaded_v5 = self.ledger.load_neural_prior_holdout_plan(v5_digest)
        self.assertIsInstance(loaded_v5, LegacyNeuralPriorHoldoutPlanV5Audit)
        self.assertEqual(loaded_v5.plan_digest, v5_digest)

        normalized_v6: dict[str, object] = {
            "contract": "neural-prior-holdout-plan-v6",
            "plan_id": "legacy-v6-plan",
            "parent_prior_digest": "7" * 64,
            "candidate_family_digests": ["8" * 64],
            "cases": [],
            "input_plans": [],
            "uncertainty_target_plans": [],
            "state_calibration_target_plans": [],
            "registered_at": "2026-08-07T00:00:00Z",
            "mode": "prospective",
            "sealed_historical_dataset_digest": None,
            "candidate_training_started_at": None,
        }
        v6_digest = json_digest(normalized_v6)
        stored_v6 = {**normalized_v6, "plan_digest": v6_digest}
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_holdout_plans "
                "(plan_digest, plan_id, plan_json, policy_digest, "
                "trust_store_digest, registered_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    v6_digest,
                    "legacy-v6-plan",
                    json.dumps(stored_v6, sort_keys=True, separators=(",", ":")),
                    "9" * 64,
                    "a" * 64,
                    "2026-08-07T00:00:00Z",
                    "2026-08-07T00:00:00Z",
                ),
            )
        loaded_v6 = self.ledger.load_neural_prior_holdout_plan(v6_digest)
        self.assertIsInstance(loaded_v6, LegacyNeuralPriorHoldoutPlanV6Audit)
        self.assertEqual(loaded_v6.plan_digest, v6_digest)

    def test_v29_holdout_plan_loads_as_byte_audit_only(self) -> None:
        original = {
            "contract": "neural-prior-holdout-plan-v29",
            "plan_id": "legacy-v29-plan",
            "scientific_generation": "shared-affine-v1",
        }
        plan_digest = json_digest(original)
        payload = original | {"plan_digest": plan_digest}
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_holdout_plans "
                "(plan_digest, plan_id, plan_json, policy_digest, "
                "trust_store_digest, registered_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    plan_digest,
                    "legacy-v29-plan",
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    "6" * 64,
                    "7" * 64,
                    "2026-08-22T00:00:00Z",
                    "2026-08-22T00:00:00Z",
                ),
            )

        loaded = self.ledger.load_neural_prior_holdout_plan(plan_digest)

        self.assertIs(
            type(loaded),
            promotion_module.LegacyNeuralPriorHoldoutPlanV29Audit,
        )
        self.assertEqual(loaded.plan_digest, plan_digest)

    def test_v30_holdout_plan_loads_as_byte_audit_only(self) -> None:
        original = {
            "contract": "neural-prior-holdout-plan-v30",
            "plan_id": "legacy-v30-plan",
            "scientific_generation": "pre-metric-crs-closure-v1",
        }
        plan_digest = json_digest(original)
        payload = original | {"plan_digest": plan_digest}
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO neural_prior_holdout_plans "
                "(plan_digest, plan_id, plan_json, policy_digest, "
                "trust_store_digest, registered_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    plan_digest,
                    "legacy-v30-plan",
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    "8" * 64,
                    "9" * 64,
                    "2026-08-23T00:00:00Z",
                    "2026-08-23T00:00:00Z",
                ),
            )

        loaded = self.ledger.load_neural_prior_holdout_plan(plan_digest)

        self.assertIs(
            type(loaded),
            promotion_module.LegacyNeuralPriorHoldoutPlanV30Audit,
        )
        self.assertEqual(loaded.plan_digest, plan_digest)

    def test_legacy_v2_prospective_receipt_loads_as_signed_audit(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        executor_trust = SimpleNamespace(
            keys={"executor-v2": private_key.public_key()},
            content_digest="0" * 64,
        )
        decision_values: dict[str, object] = {
            "decision_id": "legacy-decision",
            "case_id": "legacy-case",
            "radar_id": "legacy-radar",
            "intervention_type": "realized_sensor_correction",
            "action_policy_digest": "1" * 64,
            "action_generator_digest": "2" * 64,
            "action_context_digest": "3" * 64,
            "action_payload_digest": "4" * 64,
            "action_application_contract_digest": "5" * 64,
            "action_digest": "6" * 64,
            "input_plan_digest": "7" * 64,
            "actual_input_before_frames_digest": "8" * 64,
            "actual_input_before_bundle_digest": "9" * 64,
            "decision_basis_digest": "a" * 64,
            "decision_policy_digest": "b" * 64,
            "decision_trust_store_digest": "c" * 64,
            "decided_at": "2026-08-08T00:02:00Z",
            "observation_valid_time": "2026-08-08T00:00:00Z",
            "input_available_time": "2026-08-08T00:01:00Z",
            "decision_deadline": "2026-08-08T00:03:00Z",
            "publication_time": "2026-08-08T00:05:00Z",
            "contract": "prospective-intervention-decision-v2",
        }
        decision_digest = json_digest(decision_values)
        decision_values["decision_digest"] = decision_digest
        receipt_values: dict[str, object] = {
            "decision_digest": decision_digest,
            "decision_id": "legacy-decision",
            "case_id": "legacy-case",
            "radar_id": "legacy-radar",
            "intervention_type": "realized_sensor_correction",
            "action_digest": "6" * 64,
            "input_plan_digest": "7" * 64,
            "actual_input_before_frames_digest": "8" * 64,
            "actual_input_after_frames_digest": "d" * 64,
            "actual_input_before_bundle_digest": "9" * 64,
            "actual_input_bundle_digest": "e" * 64,
            "action_payload_digest": "4" * 64,
            "action_application_contract_digest": "5" * 64,
            "executor_key_id": "executor-v2",
            "executor_trust_store_digest": executor_trust.content_digest,
            "executor_signature": "",
            "applied_time": "2026-08-08T00:03:00Z",
            "receipt_time": "2026-08-08T00:03:30Z",
            "observation_valid_time": "2026-08-08T00:00:00Z",
            "input_available_time": "2026-08-08T00:01:00Z",
            "publication_time": "2026-08-08T00:05:00Z",
            "executor_sequence_number": 1,
            "contract": "realized-intervention-receipt-v2",
        }
        receipt_values["executor_signature"] = private_key.sign(
            json_digest(receipt_values).encode("ascii")
        ).hex()
        receipt_digest = json_digest(receipt_values)
        receipt_values["receipt_digest"] = receipt_digest
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO prospective_intervention_decisions "
                "(decision_digest, decision_id, decision_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    decision_digest,
                    "legacy-decision",
                    json.dumps(decision_values, sort_keys=True),
                    "2026-08-08T00:02:00+00:00",
                ),
            )
            connection.execute(
                "INSERT INTO realized_intervention_receipts "
                "(receipt_digest, decision_digest, executor_key_id, "
                "executor_sequence_number, receipt_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    receipt_digest,
                    decision_digest,
                    "executor-v2",
                    1,
                    json.dumps(receipt_values, sort_keys=True),
                    "2026-08-08T00:03:30+00:00",
                ),
            )
        with patch(
            "advar.ledger._load_executor_trust_store",
            return_value=executor_trust,
        ):
            loaded = self.ledger.load_prospective_intervention(
                receipt_digest,
                executor_trust_store_path="/etc/advar/executors.json",
            )
        self.assertIsInstance(
            loaded[0],
            LegacyProspectiveInterventionDecisionAudit,
        )
        self.assertIsInstance(
            loaded[1],
            LegacyRealizedInterventionReceiptAudit,
        )

    def test_v3_promotion_and_pre_full_input_manifest_load_audit_only(self) -> None:
        candidate_digest = "1" * 64
        parent_digest = "2" * 64
        manifest_payload: dict[str, object] = {
            "contract": "neural-prior-candidate-manifest-v2",
            "candidate_prior_digest": candidate_digest,
            "parent_prior_digest": parent_digest,
            "holdout_cases": [{"case_id": "legacy-case"}],
        }
        manifest_digest = json_digest(manifest_payload)
        manifest_payload["manifest_digest"] = manifest_digest
        promotion_payload: dict[str, object] = {
            "candidate_prior_digest": candidate_digest,
            "parent_prior_digest": parent_digest,
            "candidate_manifest_digest": manifest_digest,
            "policy_digest": "3" * 64,
            "trust_store_digest": "4" * 64,
            "evaluation_digests": (),
            "holdout_case_count": 0,
            "material_case_count": 0,
            "distinct_case_count": 0,
            "distinct_storm_count": 0,
            "distinct_day_count": 0,
            "distinct_radar_count": 0,
            "distinct_regime_count": 0,
            "distinct_range_regime_count": 0,
            "beneficial_fraction": 0.0,
            "beneficial_fraction_lower_bound": 0.0,
            "harmful_fraction": 0.0,
            "harmful_fraction_upper_bound": 0.0,
            "mean_normalized_improvement": 0.0,
            "mean_improvement_lower_bound": 0.0,
            "maximum_normalized_degradation": 0.0,
            "eligible": False,
            "rejection_reasons": ("no_material_outcome",),
            "contract": "neural-prior-promotion-evidence-v3",
        }
        evidence_digest = json_digest(promotion_payload)
        overrides: dict[str, object] = {
            "promotion_evidence_digest": evidence_digest,
            "candidate_prior_digest": candidate_digest,
            "parent_prior_digest": parent_digest,
            "candidate_manifest_digest": manifest_digest,
            "candidate_manifest_json": json.dumps(
                manifest_payload,
                sort_keys=True,
            ),
            "holdout_plan_digest": "5" * 64,
            "policy_digest": promotion_payload["policy_digest"],
            "trust_store_digest": promotion_payload["trust_store_digest"],
            "evaluation_digests_json": "[]",
            "evaluation_payloads_json": "[]",
            "intervention_digests_json": "[]",
            "rejection_reasons_json": json.dumps(["no_material_outcome"]),
            "evidence_contract": promotion_payload["contract"],
            "created_at": "2026-08-01T00:00:00+00:00",
        }
        with sqlite3.connect(self.ledger.index_path) as connection:
            schema = connection.execute(
                "PRAGMA table_info(neural_prior_promotions)"
            ).fetchall()
            columns = [str(row[1]) for row in schema]
            values: list[object] = []
            for row in schema:
                name = str(row[1])
                if name in overrides:
                    values.append(overrides[name])
                elif str(row[2]).upper() == "INTEGER":
                    values.append(0)
                elif str(row[2]).upper() == "REAL":
                    values.append(0.0)
                else:
                    values.append("")
            placeholders = ",".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO neural_prior_promotions "
                f"({','.join(columns)}) VALUES ({placeholders})",
                values,
            )
        loaded = self.ledger.load_neural_prior_promotion(evidence_digest)
        self.assertIsInstance(loaded, LegacyNeuralPriorPromotionEvidenceAuditV3)
        self.assertNotIsInstance(loaded, NeuralPriorPromotionEvidence)
        self.assertFalse(
            hasattr(loaded, "prior_echo_intensity_nll_increase_upper_bound")
        )

    def test_v5_promotion_and_pre_probability_manifest_load_audit_only(
        self,
    ) -> None:
        candidate_digest = "1" * 64
        parent_digest = "2" * 64
        manifest_payload: dict[str, object] = {
            "contract": "neural-prior-candidate-manifest-v3",
            "candidate_prior_digest": candidate_digest,
            "parent_prior_digest": parent_digest,
            "holdout_cases": [{"case_id": "legacy-v5-case"}],
        }
        manifest_digest = json_digest(manifest_payload)
        manifest_payload["manifest_digest"] = manifest_digest
        promotion_payload: dict[str, object] = {
            "candidate_prior_digest": candidate_digest,
            "parent_prior_digest": parent_digest,
            "candidate_manifest_digest": manifest_digest,
            "policy_digest": "3" * 64,
            "trust_store_digest": "4" * 64,
            "evaluation_digests": (),
            "holdout_case_count": 0,
            "material_case_count": 0,
            "distinct_case_count": 0,
            "distinct_storm_count": 0,
            "distinct_day_count": 0,
            "distinct_radar_count": 0,
            "distinct_regime_count": 0,
            "distinct_range_regime_count": 0,
            "beneficial_fraction": 0.0,
            "beneficial_fraction_lower_bound": 0.0,
            "harmful_fraction": 0.0,
            "harmful_fraction_upper_bound": 0.0,
            "mean_normalized_improvement": 0.0,
            "mean_improvement_lower_bound": 0.0,
            "maximum_normalized_degradation": 0.0,
            "prior_echo_intensity_nll_increase_upper_bound": 0.0,
            "prior_support_brier_increase_upper_bound": 0.0,
            "prior_clear_sky_false_echo_increase_upper_bound": 0.0,
            "prior_underdispersion_increase_upper_bound": 0.0,
            "eligible": False,
            "rejection_reasons": ("no_material_outcome",),
            "contract": "neural-prior-promotion-evidence-v5",
        }
        evidence_digest = json_digest(promotion_payload)
        overrides: dict[str, object] = {
            "promotion_evidence_digest": evidence_digest,
            "candidate_prior_digest": candidate_digest,
            "parent_prior_digest": parent_digest,
            "candidate_manifest_digest": manifest_digest,
            "candidate_manifest_json": json.dumps(
                manifest_payload,
                sort_keys=True,
            ),
            "holdout_plan_digest": "5" * 64,
            "policy_digest": promotion_payload["policy_digest"],
            "trust_store_digest": promotion_payload["trust_store_digest"],
            "evaluation_digests_json": "[]",
            "evaluation_payloads_json": "[]",
            "intervention_digests_json": "[]",
            "rejection_reasons_json": json.dumps(["no_material_outcome"]),
            "evidence_contract": promotion_payload["contract"],
            "created_at": "2026-08-01T00:00:00+00:00",
        }
        for name in (
            "prior_echo_intensity_nll_increase_upper_bound",
            "prior_support_brier_increase_upper_bound",
            "prior_clear_sky_false_echo_increase_upper_bound",
            "prior_underdispersion_increase_upper_bound",
        ):
            overrides[name] = promotion_payload[name]
        with sqlite3.connect(self.ledger.index_path) as connection:
            schema = connection.execute(
                "PRAGMA table_info(neural_prior_promotions)"
            ).fetchall()
            columns = [str(row[1]) for row in schema]
            values = [
                overrides.get(
                    str(row[1]),
                    0 if str(row[2]).upper() == "INTEGER" else 0.0
                    if str(row[2]).upper() == "REAL"
                    else "",
                )
                for row in schema
            ]
            connection.execute(
                f"INSERT INTO neural_prior_promotions "
                f"({','.join(columns)}) VALUES "
                f"({','.join('?' for _ in columns)})",
                values,
            )

        loaded = self.ledger.load_neural_prior_promotion(evidence_digest)

        self.assertIsInstance(loaded, LegacyNeuralPriorPromotionEvidenceAuditV5)
        decoded_manifest = ledger_module._decode_candidate_manifest(
            json.dumps(manifest_payload, sort_keys=True),
            expected_digest=manifest_digest,
        )
        self.assertIsInstance(
            decoded_manifest,
            LegacyNeuralPriorCandidateManifestAuditV3,
        )

    def test_v6_promotion_and_v7_evaluation_load_audit_only(self) -> None:
        manifest_payload: dict[str, object] = {
            "contract": "neural-prior-candidate-manifest-v3",
            "candidate_prior_digest": "1" * 64,
            "parent_prior_digest": "2" * 64,
            "holdout_cases": [{"case_id": "legacy-v6-case"}],
        }
        manifest_digest = json_digest(manifest_payload)
        manifest_payload["manifest_digest"] = manifest_digest
        evaluation_payload = {
            "contract": "prior-holdout-evaluation-v7",
            "evaluation_digest": "e" * 64,
        }
        payload: dict[str, object] = {
            "candidate_prior_digest": "1" * 64,
            "parent_prior_digest": "2" * 64,
            "candidate_manifest_digest": manifest_digest,
            "policy_digest": "3" * 64,
            "trust_store_digest": "4" * 64,
            "evaluation_digests": ("e" * 64,),
            "holdout_case_count": 1,
            "material_case_count": 0,
            "distinct_case_count": 0,
            "distinct_storm_count": 0,
            "distinct_day_count": 0,
            "distinct_radar_count": 0,
            "distinct_regime_count": 0,
            "distinct_range_regime_count": 0,
            "beneficial_fraction": 0.0,
            "beneficial_fraction_lower_bound": 0.0,
            "harmful_fraction": 0.0,
            "harmful_fraction_upper_bound": 0.0,
            "mean_normalized_improvement": 0.0,
            "mean_improvement_lower_bound": 0.0,
            "maximum_normalized_degradation": 0.0,
            "prior_echo_intensity_nll_increase_upper_bound": 0.0,
            "prior_support_brier_increase_upper_bound": 0.0,
            "prior_clear_sky_false_echo_increase_upper_bound": 0.0,
            "prior_underdispersion_increase_upper_bound": 0.0,
            "prior_echo_component_status": "not_applicable",
            "prior_clear_sky_component_status": "not_applicable",
            "prior_echo_case_count": 0,
            "prior_clear_sky_case_count": 0,
            "prior_echo_cluster_count": 0,
            "prior_clear_sky_cluster_count": 0,
            "simultaneous_inference_test_count": 1,
            "eligible": False,
            "rejection_reasons": ("no_material_outcome",),
            "contract": "neural-prior-promotion-evidence-v6",
        }
        evidence_digest = json_digest(payload)
        overrides: dict[str, object] = {
            "promotion_evidence_digest": evidence_digest,
            "candidate_manifest_json": json.dumps(manifest_payload, sort_keys=True),
            "holdout_plan_digest": "5" * 64,
            "evaluation_digests_json": json.dumps(["e" * 64]),
            "evaluation_payloads_json": json.dumps([evaluation_payload]),
            "intervention_digests_json": "[]",
            "rejection_reasons_json": json.dumps(["no_material_outcome"]),
            "evidence_contract": payload["contract"],
            "created_at": "2026-08-01T00:00:00+00:00",
        }
        column_mapping = {
            "realized_intervention_count": "holdout_case_count",
            "material_outcome_count": "material_case_count",
        }
        with sqlite3.connect(self.ledger.index_path) as connection:
            schema = connection.execute(
                "PRAGMA table_info(neural_prior_promotions)"
            ).fetchall()
            columns = [str(row[1]) for row in schema]
            values: list[object] = []
            for row in schema:
                name = str(row[1])
                source = column_mapping.get(name, name)
                if name in overrides:
                    values.append(overrides[name])
                elif source in payload:
                    values.append(payload[source])
                elif str(row[2]).upper() == "INTEGER":
                    values.append(0)
                elif str(row[2]).upper() == "REAL":
                    values.append(0.0)
                else:
                    values.append("")
            connection.execute(
                f"INSERT INTO neural_prior_promotions "
                f"({','.join(columns)}) VALUES "
                f"({','.join('?' for _ in columns)})",
                values,
            )

        loaded = self.ledger.load_neural_prior_promotion(evidence_digest)
        evaluations = self.ledger.load_neural_prior_promotion_evaluations(
            evidence_digest
        )
        self.assertIsInstance(loaded, LegacyNeuralPriorPromotionEvidenceAuditV6)
        self.assertIsInstance(evaluations[0], ledger_module.LegacyPromotionEvaluationAudit)
        self.assertFalse(evaluations[0].statistical_reuse_permitted)

    def test_v7_promotion_and_v4_manifest_load_audit_only(self) -> None:
        manifest_payload: dict[str, object] = {
            "contract": "neural-prior-candidate-manifest-v4",
            "candidate_prior_digest": "1" * 64,
            "parent_prior_digest": "2" * 64,
            "holdout_cases": [{"case_id": "legacy-v7-case"}],
        }
        manifest_digest = json_digest(manifest_payload)
        manifest_payload["manifest_digest"] = manifest_digest
        payload: dict[str, object] = {
            "candidate_prior_digest": "1" * 64,
            "parent_prior_digest": "2" * 64,
            "candidate_manifest_digest": manifest_digest,
            "policy_digest": "3" * 64,
            "trust_store_digest": "4" * 64,
            "evaluation_digests": (),
            "holdout_case_count": 0,
            "material_case_count": 0,
            "distinct_case_count": 0,
            "distinct_storm_count": 0,
            "distinct_day_count": 0,
            "distinct_radar_count": 0,
            "distinct_regime_count": 0,
            "distinct_range_regime_count": 0,
            "beneficial_fraction": 0.0,
            "beneficial_fraction_lower_bound": 0.0,
            "harmful_fraction": 0.0,
            "harmful_fraction_upper_bound": 0.0,
            "mean_normalized_improvement": 0.0,
            "mean_improvement_lower_bound": 0.0,
            "maximum_normalized_degradation": 0.0,
            "prior_echo_intensity_nll_increase_upper_bound": 0.0,
            "prior_support_brier_increase_upper_bound": 0.0,
            "prior_echo_support_miss_increase_upper_bound": 0.0,
            "prior_echo_object_miss_increase_upper_bound": 0.0,
            "prior_clear_sky_false_echo_increase_upper_bound": 0.0,
            "prior_conditional_underdispersion_increase_upper_bound": 0.0,
            "prior_echo_component_status": "not_applicable",
            "prior_clear_sky_component_status": "not_applicable",
            "prior_echo_case_count": 0,
            "prior_clear_sky_case_count": 0,
            "prior_echo_cluster_count": 0,
            "prior_clear_sky_cluster_count": 0,
            "simultaneous_inference_test_count": 1,
            "simultaneous_inference_method": "exact_sign_enumeration",
            "simultaneous_inference_effective_replicates": 1,
            "simultaneous_inference_critical_quantile": 0.95,
            "simultaneous_inference_monte_carlo_standard_error": 0.0,
            "simultaneous_inference_tail_replicates": 0.05,
            "cluster_bootstrap_tail_replicates": 0.05,
            "certified_applicability_regime_groups": (),
            "requires_parent_fallback_outside_certified_applicability": True,
            "eligible": False,
            "rejection_reasons": ("no_material_outcome",),
            "contract": "neural-prior-promotion-evidence-v7",
        }
        evidence_digest = json_digest(payload)
        overrides: dict[str, object] = {
            "promotion_evidence_digest": evidence_digest,
            "candidate_manifest_json": json.dumps(
                manifest_payload,
                sort_keys=True,
            ),
            "holdout_plan_digest": "5" * 64,
            "evaluation_digests_json": "[]",
            "evaluation_payloads_json": "[]",
            "intervention_digests_json": "[]",
            "rejection_reasons_json": json.dumps(["no_material_outcome"]),
            "evidence_contract": payload["contract"],
            "evidence_payload_json": json.dumps(payload, sort_keys=True),
            "created_at": "2026-08-01T00:00:00+00:00",
        }
        column_mapping = {
            "realized_intervention_count": "holdout_case_count",
            "material_outcome_count": "material_case_count",
        }
        with sqlite3.connect(self.ledger.index_path) as connection:
            schema = connection.execute(
                "PRAGMA table_info(neural_prior_promotions)"
            ).fetchall()
            columns = [str(row[1]) for row in schema]
            values: list[object] = []
            for row in schema:
                name = str(row[1])
                source = column_mapping.get(name, name)
                if name in overrides:
                    values.append(overrides[name])
                elif source in payload:
                    values.append(payload[source])
                elif str(row[2]).upper() == "INTEGER":
                    values.append(0)
                elif str(row[2]).upper() == "REAL":
                    values.append(0.0)
                else:
                    values.append("")
            connection.execute(
                f"INSERT INTO neural_prior_promotions "
                f"({','.join(columns)}) VALUES "
                f"({','.join('?' for _ in columns)})",
                values,
            )

        loaded = self.ledger.load_neural_prior_promotion(evidence_digest)
        decoded_manifest = ledger_module._decode_candidate_manifest(
            json.dumps(manifest_payload, sort_keys=True),
            expected_digest=manifest_digest,
        )

        self.assertIsInstance(
            loaded,
            LegacyNeuralPriorPromotionEvidenceAuditV7,
        )
        self.assertIsInstance(
            decoded_manifest,
            LegacyNeuralPriorCandidateManifestAuditV4,
        )

    def test_v8_promotion_loads_as_pre_classifier_audit(self) -> None:
        manifest_payload: dict[str, object] = {
            "contract": "neural-prior-candidate-manifest-v4",
            "candidate_prior_digest": "1" * 64,
            "parent_prior_digest": "2" * 64,
            "holdout_cases": [{"case_id": "legacy-v8-case"}],
        }
        manifest_digest = json_digest(manifest_payload)
        manifest_payload["manifest_digest"] = manifest_digest
        payload: dict[str, object] = {
            "candidate_prior_digest": "1" * 64,
            "parent_prior_digest": "2" * 64,
            "candidate_manifest_digest": manifest_digest,
            "policy_digest": "3" * 64,
            "trust_store_digest": "4" * 64,
            "evaluation_digests": (),
            "holdout_case_count": 0,
            "material_case_count": 0,
            "distinct_case_count": 0,
            "distinct_storm_count": 0,
            "distinct_day_count": 0,
            "distinct_radar_count": 0,
            "distinct_regime_count": 0,
            "distinct_range_regime_count": 0,
            "beneficial_fraction": 0.0,
            "beneficial_fraction_lower_bound": 0.0,
            "harmful_fraction": 0.0,
            "harmful_fraction_upper_bound": 0.0,
            "mean_normalized_improvement": 0.0,
            "mean_improvement_lower_bound": 0.0,
            "maximum_normalized_degradation": 0.0,
            "prior_echo_intensity_nll_increase_upper_bound": 0.0,
            "prior_support_brier_increase_upper_bound": 0.0,
            "prior_echo_support_miss_increase_upper_bound": 0.0,
            "prior_echo_object_miss_increase_upper_bound": 0.0,
            "prior_clear_sky_false_echo_increase_upper_bound": 0.0,
            "prior_conditional_underdispersion_increase_upper_bound": 0.0,
            "state_gaussian_nll_increase_upper_bound": 0.0,
            "state_underdispersion_increase_upper_bound": 0.0,
            "state_support_brier_increase_upper_bound": 0.0,
            "state_echo_support_miss_increase_upper_bound": 0.0,
            "state_echo_object_miss_increase_upper_bound": 0.0,
            "state_false_support_increase_upper_bound": 0.0,
            "state_valid_brier_increase_upper_bound": 0.0,
            "deployment_regime_classifier_digest": "5" * 64,
            "prior_echo_component_status": "not_applicable",
            "prior_clear_sky_component_status": "not_applicable",
            "prior_echo_case_count": 0,
            "prior_clear_sky_case_count": 0,
            "prior_echo_cluster_count": 0,
            "prior_clear_sky_cluster_count": 0,
            "simultaneous_inference_test_count": 1,
            "simultaneous_inference_method": "exact_sign_enumeration",
            "simultaneous_inference_effective_replicates": 1,
            "simultaneous_inference_critical_quantile": 0.95,
            "simultaneous_inference_monte_carlo_standard_error": 0.0,
            "simultaneous_inference_tail_replicates": 0.05,
            "cluster_bootstrap_tail_replicates": 0.05,
            "certified_applicability_regime_groups": (),
            "requires_parent_fallback_outside_certified_applicability": True,
            "state_calibration_eligible": False,
            "deployment_eligible": False,
            "eligible": False,
            "rejection_reasons": ("no_material_outcome",),
            "contract": "neural-prior-promotion-evidence-v8",
        }
        evidence_digest = json_digest(payload)
        overrides: dict[str, object] = {
            "promotion_evidence_digest": evidence_digest,
            "candidate_manifest_json": json.dumps(
                manifest_payload,
                sort_keys=True,
            ),
            "holdout_plan_digest": "6" * 64,
            "evaluation_digests_json": "[]",
            "evaluation_payloads_json": "[]",
            "intervention_digests_json": "[]",
            "rejection_reasons_json": json.dumps(["no_material_outcome"]),
            "evidence_contract": payload["contract"],
            "evidence_payload_json": json.dumps(payload, sort_keys=True),
            "created_at": "2026-08-01T00:00:00+00:00",
        }
        column_mapping = {
            "realized_intervention_count": "holdout_case_count",
            "material_outcome_count": "material_case_count",
        }
        with sqlite3.connect(self.ledger.index_path) as connection:
            schema = connection.execute(
                "PRAGMA table_info(neural_prior_promotions)"
            ).fetchall()
            columns = [str(row[1]) for row in schema]
            values: list[object] = []
            for row in schema:
                name = str(row[1])
                source = column_mapping.get(name, name)
                if name in overrides:
                    values.append(overrides[name])
                elif source in payload:
                    values.append(payload[source])
                elif str(row[2]).upper() == "INTEGER":
                    values.append(0)
                elif str(row[2]).upper() == "REAL":
                    values.append(0.0)
                else:
                    values.append("")
            connection.execute(
                f"INSERT INTO neural_prior_promotions "
                f"({','.join(columns)}) VALUES "
                f"({','.join('?' for _ in columns)})",
                values,
            )

        loaded = self.ledger.load_neural_prior_promotion(evidence_digest)

        self.assertIsInstance(
            loaded,
            LegacyNeuralPriorPromotionEvidenceAuditV8,
        )
        self.assertNotIsInstance(loaded, NeuralPriorPromotionEvidence)

    def test_backdated_decision_cannot_be_recorded_after_issue(self) -> None:
        frames = torch.zeros((3, 2, 2), dtype=torch.float64)
        run, context, plan_digest, _ = _prospective_run_and_context(
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            valid_times=(
                "2020-08-07T23:50:00Z",
                "2020-08-08T00:00:00Z",
                "2020-08-08T00:10:00Z",
            ),
            input_available_time="2020-08-08T00:10:00Z",
            decision_deadline="2020-08-08T00:30:00Z",
            publication_time="2020-08-08T01:00:00Z",
        )
        generator = InterventionActionGenerator.from_model(
            _AddOneAction().eval(),
            context,
            intervention_type="realized_sensor_correction",
        )
        action_policy = ReusableInterventionPolicyEvidence(
            policy_id="backdated-policy",
            action_generator_digest=generator.generator_digest,
            context_schema_digest=context.context_schema_digest,
            applicability_region_digest=context.applicability_region_digest,
            execution_policy_digest="e" * 64,
            allowed_intervention_types=("realized_sensor_correction",),
            maximum_absolute_delta_dbz=1.0,
            maximum_changed_fraction=1.0,
            validation_evidence_digests=("d" * 64,),
        )
        decision = ProspectiveInterventionDecision.from_policy(
            action_policy,
            action_generator=generator,
            decision_id="backdated",
            case_id="case-1",
            radar_id="radar-1",
            intervention_type="realized_sensor_correction",
            actual_input_context=context,
            actual_input_before_run=run,
            input_plan_digest=plan_digest,
            decision_basis_digest="d" * 64,
            decision_policy_digest="e" * 64,
            decision_trust_store_digest="f" * 64,
            decided_at="2020-08-08T00:11:00Z",
            observation_valid_time="2020-08-08T00:10:00Z",
            input_available_time="2020-08-08T00:10:00Z",
            decision_deadline="2020-08-08T00:30:00Z",
            publication_time="2020-08-08T01:00:00Z",
        )
        trust = SimpleNamespace(
            content_digest="f" * 64,
            approved_policy_digests=frozenset(
                ("e" * 64, action_policy.policy_digest)
            ),
        )
        operator_approval, operator_trust, _ = _operator_approval(decision)
        with (
            patch(
                "advar.ledger._load_learning_policy_trust_store",
                return_value=trust,
            ),
            patch(
                "advar.ledger._load_operator_trust_store",
                return_value=operator_trust,
            ),
            self.assertRaisesRegex(ValueError, "decision deadline"),
        ):
            self.ledger.append_prospective_intervention_decision(
                decision,
                operator_approval=operator_approval,
                action_policy=action_policy,
                action_generator=generator,
                actual_input_before_context=context,
                actual_input_before_run=run,
                trust_store_path="/etc/advar/policies.json",
                operator_trust_store_path="/etc/advar/operators.json",
            )

    def test_complete_verification_lineage_round_trips(self) -> None:
        snapshot = replace(
            self.snapshot,
            grid_time_contract_digest="1" * 64,
            verification_contract="radar-verification-bundle-v1",
            verification_bundle_digest="2" * 64,
            verification_lineage_complete=True,
            verification_valid_times=tuple(
                (datetime(2026, 7, 26, tzinfo=timezone.utc) + timedelta(minutes=lead))
                .isoformat()
                .replace("+00:00", "Z")
                for lead in range(10, 181, 10)
            ),
            verification_grid_contract_digest="1" * 64,
            verification_radar_product_digest="3" * 64,
            verification_qc_pipeline_digest="4" * 64,
        )
        episode = SensitivityEpisode(
            episode_id="complete-verification-lineage",
            issue_time="2026-07-26T00:00:00+00:00",
            radar_id="KTLX",
            contract=_contract(snapshot),
            snapshot=snapshot,
        )
        self.ledger.append(episode)
        loaded = self.ledger.load(episode.episode_id)

        self.assertTrue(loaded.manifest["verification_lineage_complete"])
        self.assertEqual(
            loaded.manifest["verification_bundle_digest"],
            "2" * 64,
        )
        self.assertEqual(
            loaded.manifest["verification_grid_contract_digest"],
            "1" * 64,
        )

        with self.assertRaisesRegex(ValueError, "bundle digest"):
            self.ledger.append(
                SensitivityEpisode(
                    episode_id="invalid-verification-lineage",
                    issue_time="2026-07-26T00:00:00+00:00",
                    radar_id="KTLX",
                    contract=_contract(snapshot),
                    snapshot=replace(
                        snapshot,
                        verification_bundle_digest="bad",
                    ),
                )
            )

    def test_v2_layout_cannot_claim_schema_one(self) -> None:
        episode = self.episode("v2-as-schema-one")
        target = self.ledger.append(episode)
        manifest_path = target / "manifest.json"
        checksums_path = target / "checksums.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["schema_version"] = 1
        _drop_verification_lineage(manifest)
        manifest.pop("forecast_run_digest")
        manifest["contract"].pop("nowcast_config_digest")
        manifest["contract"].pop("sensitivity_config_digest")
        manifest["contract"].pop("grid_time_contract_digest")
        manifest["context_feature_names"] = list(SCHEMA_ONE_CONTEXT_FEATURE_NAMES)
        manifest["arrays"]["context_features"]["shape"] = [
            len(SCHEMA_ONE_CONTEXT_FEATURE_NAMES)
        ]
        interval = manifest["lead_minutes"][0]
        manifest["sensitivity_scope"] = {
            f"input_{-2 * interval}_minutes": "no_direct_forecast_path",
            f"input_{-interval}_minutes": "no_direct_forecast_path",
            "input_0_minutes": "partial_direct_latest_dbz_fixed_control",
            "indirect_analysis_path": (
                "unavailable_implicit_variational_fso_not_implemented"
            ),
            "total_observation_sensitivity": "unavailable",
        }
        contract_value = {
            "contract_schema_version": 1,
            **manifest["contract"],
        }
        manifest["contract_hash"] = hashlib.sha256(
            json.dumps(
                contract_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        checksums = json.loads(checksums_path.read_text("utf-8"))
        checksums["manifest.json"] = manifest_hash
        checksums_path.write_text(
            json.dumps(
                checksums,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute("DROP TRIGGER episodes_no_update")
            connection.execute(
                """
                UPDATE episodes
                SET contract_hash = ?, manifest_sha256 = ?
                WHERE episode_id = ?
                """,
                (manifest["contract_hash"], manifest_hash, episode.episode_id),
            )

        reopened = EpisodeLedger(self.root)
        with self.assertRaisesRegex(ValueError, "episode schema"):
            reopened.verify(episode.episode_id)

    def test_schema_one_episode_remains_verifiable(self) -> None:
        episode = self.episode("schema-one")
        target = self.ledger.append(episode)
        arrays_path = target / "sensitivity_arrays.npz"
        manifest_path = target / "manifest.json"
        checksums_path = target / "checksums.json"
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}

        context = arrays["context_features"]
        arrays["context_features"] = np.concatenate((context[:6], context[14:23]))

        def latest_axis(name: str, *, fill: float = 0.0) -> np.ndarray:
            source = arrays[name]
            shape = (*source.shape[:2], 3, *source.shape[2:])
            expanded = np.full(shape, fill, dtype=source.dtype)
            expanded[:, :, 2] = source
            return expanded

        arrays["direct_observation_sensitivity"] = latest_axis(
            "direct_observation_sensitivity"
        )
        arrays["direct_observation_sensitivity_norm"] = latest_axis(
            "direct_observation_sensitivity_norm"
        )
        arrays["tile_direct_sensitivity_norm"] = latest_axis(
            "tile_direct_sensitivity_norm"
        )
        arrays["baseline_scores"] = np.ones_like(arrays["forecast_scores"])
        arrays["direct_normalized_reward"] = (
            -arrays["direct_observation_impact"] / arrays["baseline_scores"]
        )
        arrays["direct_observation_impact"] = latest_axis("direct_observation_impact")
        arrays["tile_direct_observation_impact"] = latest_axis(
            "tile_direct_observation_impact"
        )
        arrays["direct_normalized_reward"] = latest_axis("direct_normalized_reward")

        height, width = arrays["latest_sensitivity_mask"].shape
        tile_shape = arrays["tile_direct_sensitivity_norm"].shape
        lead_count, metric_count = tile_shape[:2]
        tile_rows, tile_columns = tile_shape[-2:]
        arrays["tile_whitened_direct_sensitivity_norm"] = np.full(
            (lead_count, metric_count, 3, tile_rows, tile_columns),
            np.nan,
        )
        arrays["observation_std_dbz"] = np.full((3, height, width), np.nan)
        innovation = np.full((3, height, width), np.nan)
        innovation[2] = arrays["observation_innovation_dbz"]
        arrays["observation_innovation_dbz"] = innovation
        innovation_mask = np.zeros((3, height, width), dtype=np.bool_)
        innovation_mask[2] = arrays["observation_innovation_mask"]
        arrays["observation_innovation_mask"] = innovation_mask
        np.savez_compressed(arrays_path, **arrays)

        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["schema_version"] = 1
        _drop_verification_lineage(manifest)
        manifest["reward_available"] = True
        manifest.pop("baseline_lineage_available")
        manifest.pop("forecast_run_digest")
        manifest["contract"].pop("nowcast_config_digest")
        manifest["contract"].pop("sensitivity_config_digest")
        manifest["contract"].pop("grid_time_contract_digest")
        manifest["context_feature_names"] = list(SCHEMA_ONE_CONTEXT_FEATURE_NAMES)
        interval = manifest["lead_minutes"][0]
        manifest["sensitivity_scope"] = {
            f"input_{-2 * interval}_minutes": "no_direct_forecast_path",
            f"input_{-interval}_minutes": "no_direct_forecast_path",
            "input_0_minutes": "partial_direct_latest_dbz_fixed_control",
            "indirect_analysis_path": (
                "unavailable_implicit_variational_fso_not_implemented"
            ),
            "total_observation_sensitivity": "unavailable",
        }
        manifest["contract_hash"] = hashlib.sha256(
            json.dumps(
                {"contract_schema_version": 1, **manifest["contract"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest["arrays"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        }
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        arrays_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        checksums_path.write_text(
            json.dumps(
                {
                    "manifest.json": manifest_hash,
                    "sensitivity_arrays.npz": arrays_hash,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute("DROP TRIGGER episodes_no_update")
            connection.execute(
                """
                UPDATE episodes
                SET contract_hash = ?, manifest_sha256 = ?, arrays_sha256 = ?
                WHERE episode_id = ?
                """,
                (
                    manifest["contract_hash"],
                    manifest_hash,
                    arrays_hash,
                    episode.episode_id,
                ),
            )

        reopened = EpisodeLedger(self.root)
        reopened.verify(episode.episode_id)
        loaded = reopened.load(episode.episode_id)
        self.assertEqual(loaded.manifest["schema_version"], 1)
        self.assertEqual(
            loaded.arrays["direct_observation_sensitivity_norm"].shape,
            (18, 1, 3),
        )
        self.assertEqual(
            loaded.manifest["context_feature_names"],
            list(SCHEMA_ONE_CONTEXT_FEATURE_NAMES),
        )
        self.assertEqual(loaded.arrays["context_features"].shape, (15,))

    def test_schema_three_episode_without_run_identity_remains_verifiable(
        self,
    ) -> None:
        episode = self.episode("schema-three")
        target = self.ledger.append(episode)
        arrays_path = target / "sensitivity_arrays.npz"
        manifest_path = target / "manifest.json"
        checksums_path = target / "checksums.json"
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        context = arrays["context_features"]
        arrays["context_features"] = np.concatenate((context[:6], context[8:23]))
        np.savez_compressed(arrays_path, **arrays)
        arrays_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()

        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["schema_version"] = 3
        _drop_verification_lineage(manifest)
        manifest.pop("forecast_run_digest")
        manifest["contract"].pop("grid_time_contract_digest")
        manifest["context_feature_names"] = list(
            SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES
        )
        manifest["arrays"]["context_features"]["shape"] = [
            len(SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES)
        ]
        legacy_contract_hash = hashlib.sha256(
            json.dumps(
                {"contract_schema_version": 3, **manifest["contract"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest["contract_hash"] = legacy_contract_hash
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        checksums = json.loads(checksums_path.read_text("utf-8"))
        checksums["manifest.json"] = manifest_hash
        checksums["sensitivity_arrays.npz"] = arrays_hash
        checksums_path.write_text(
            json.dumps(
                checksums,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute("DROP TRIGGER episodes_no_update")
            connection.execute(
                """
                UPDATE episodes
                SET contract_hash = ?, manifest_sha256 = ?, arrays_sha256 = ?
                WHERE episode_id = ?
                """,
                (
                    legacy_contract_hash,
                    manifest_hash,
                    arrays_hash,
                    episode.episode_id,
                ),
            )

        reopened = EpisodeLedger(self.root)
        reopened.verify(episode.episode_id)
        loaded = reopened.load(episode.episode_id)
        self.assertEqual(loaded.manifest["schema_version"], 3)
        self.assertNotIn("forecast_run_digest", loaded.manifest)
        self.assertEqual(
            loaded.arrays["context_features"].shape,
            (len(SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES),),
        )

        manifest["forecast_run_digest"] = None
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        checksums["manifest.json"] = manifest_hash
        checksums_path.write_text(
            json.dumps(
                checksums,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute("DROP TRIGGER episodes_no_update")
            connection.execute(
                """
                UPDATE episodes
                SET manifest_sha256 = ?
                WHERE episode_id = ?
                """,
                (manifest_hash, episode.episode_id),
            )
        with self.assertRaisesRegex(
            ValueError,
            "forecast run provenance does not match",
        ):
            reopened.verify(episode.episode_id)

    def test_schema_four_context_contract_remains_verifiable(self) -> None:
        episode = self.episode("schema-four")
        target = self.ledger.append(episode)
        arrays_path = target / "sensitivity_arrays.npz"
        manifest_path = target / "manifest.json"
        checksums_path = target / "checksums.json"
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        context = arrays["context_features"]
        arrays["context_features"] = np.concatenate((context[:6], context[8:23]))
        np.savez_compressed(arrays_path, **arrays)
        arrays_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()

        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["schema_version"] = 4
        _drop_verification_lineage(manifest)
        manifest["contract"].pop("grid_time_contract_digest")
        manifest["context_feature_names"] = list(
            SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES
        )
        manifest["arrays"]["context_features"]["shape"] = [
            len(SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES)
        ]
        legacy_contract_hash = hashlib.sha256(
            json.dumps(
                {"contract_schema_version": 3, **manifest["contract"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest["contract_hash"] = legacy_contract_hash
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        checksums = json.loads(checksums_path.read_text("utf-8"))
        checksums["manifest.json"] = manifest_hash
        checksums["sensitivity_arrays.npz"] = arrays_hash
        checksums_path.write_text(
            json.dumps(
                checksums,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute("DROP TRIGGER episodes_no_update")
            connection.execute(
                """
                UPDATE episodes
                SET contract_hash = ?, manifest_sha256 = ?, arrays_sha256 = ?
                WHERE episode_id = ?
                """,
                (
                    legacy_contract_hash,
                    manifest_hash,
                    arrays_hash,
                    episode.episode_id,
                ),
            )

        reopened = EpisodeLedger(self.root)
        reopened.verify(episode.episode_id)
        loaded = reopened.load(episode.episode_id)
        self.assertEqual(loaded.manifest["schema_version"], 4)
        self.assertEqual(
            loaded.manifest["context_feature_names"],
            list(SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES),
        )
        self.assertEqual(
            loaded.arrays["context_features"].shape,
            (len(SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES),),
        )

    def test_schema_five_conflict_context_remains_verifiable(self) -> None:
        episode = self.episode("schema-five")
        target = self.ledger.append(episode)
        arrays_path = target / "sensitivity_arrays.npz"
        manifest_path = target / "manifest.json"
        checksums_path = target / "checksums.json"
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["context_features"] = arrays["context_features"][:23]
        np.savez_compressed(arrays_path, **arrays)
        arrays_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()

        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["schema_version"] = 5
        _drop_verification_lineage(manifest)
        manifest["contract"].pop("grid_time_contract_digest")
        manifest["context_feature_names"] = list(SCHEMA_FIVE_CONTEXT_FEATURE_NAMES)
        manifest["arrays"]["context_features"]["shape"] = [
            len(SCHEMA_FIVE_CONTEXT_FEATURE_NAMES)
        ]
        legacy_contract_hash = hashlib.sha256(
            json.dumps(
                {"contract_schema_version": 4, **manifest["contract"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest["contract_hash"] = legacy_contract_hash
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        checksums = json.loads(checksums_path.read_text("utf-8"))
        checksums["manifest.json"] = manifest_hash
        checksums["sensitivity_arrays.npz"] = arrays_hash
        checksums_path.write_text(
            json.dumps(
                checksums,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute("DROP TRIGGER episodes_no_update")
            connection.execute(
                """
                UPDATE episodes
                SET contract_hash = ?, manifest_sha256 = ?, arrays_sha256 = ?
                WHERE episode_id = ?
                """,
                (
                    legacy_contract_hash,
                    manifest_hash,
                    arrays_hash,
                    episode.episode_id,
                ),
            )

        reopened = EpisodeLedger(self.root)
        reopened.verify(episode.episode_id)
        loaded = reopened.load(episode.episode_id)
        self.assertEqual(loaded.manifest["schema_version"], 5)
        self.assertEqual(
            loaded.manifest["context_feature_names"],
            list(SCHEMA_FIVE_CONTEXT_FEATURE_NAMES),
        )
        self.assertEqual(
            loaded.arrays["context_features"].shape,
            (len(SCHEMA_FIVE_CONTEXT_FEATURE_NAMES),),
        )

    def test_extended_context_schemas_remain_verifiable(self) -> None:
        cases = (
            (6, 5, SCHEMA_SIX_CONTEXT_FEATURE_NAMES),
            (7, 6, SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES),
            (8, 7, SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES),
            (9, 8, SCHEMA_NINE_CONTEXT_FEATURE_NAMES),
            (10, 9, SCHEMA_TEN_CONTEXT_FEATURE_NAMES),
            (11, 10, SCHEMA_TEN_CONTEXT_FEATURE_NAMES),
        )
        for schema_version, contract_version, context_names in cases:
            with self.subTest(schema_version=schema_version):
                episode = self.episode(f"schema-{schema_version}")
                target = self.ledger.append(episode)
                arrays_path = target / "sensitivity_arrays.npz"
                manifest_path = target / "manifest.json"
                checksums_path = target / "checksums.json"
                with np.load(arrays_path, allow_pickle=False) as archive:
                    arrays = {name: archive[name].copy() for name in archive.files}
                arrays["context_features"] = arrays["context_features"][
                    : len(context_names)
                ]
                np.savez_compressed(arrays_path, **arrays)
                arrays_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()

                manifest = json.loads(manifest_path.read_text("utf-8"))
                manifest["schema_version"] = schema_version
                _drop_verification_lineage(manifest)
                if schema_version < 11:
                    manifest["contract"].pop("grid_time_contract_digest")
                manifest["context_feature_names"] = list(context_names)
                manifest["arrays"]["context_features"]["shape"] = [len(context_names)]
                legacy_contract_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "contract_schema_version": contract_version,
                            **manifest["contract"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                manifest["contract_hash"] = legacy_contract_hash
                manifest_text = json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                manifest_path.write_text(manifest_text, encoding="utf-8")
                manifest_hash = hashlib.sha256(
                    manifest_text.encode("utf-8")
                ).hexdigest()
                checksums = json.loads(checksums_path.read_text("utf-8"))
                checksums["manifest.json"] = manifest_hash
                checksums["sensitivity_arrays.npz"] = arrays_hash
                checksums_path.write_text(
                    json.dumps(
                        checksums,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
                with sqlite3.connect(self.ledger.index_path) as connection:
                    connection.execute("DROP TRIGGER episodes_no_update")
                    connection.execute(
                        """
                        UPDATE episodes
                        SET contract_hash = ?, manifest_sha256 = ?, arrays_sha256 = ?
                        WHERE episode_id = ?
                        """,
                        (
                            legacy_contract_hash,
                            manifest_hash,
                            arrays_hash,
                            episode.episode_id,
                        ),
                    )

                reopened = EpisodeLedger(self.root)
                reopened.verify(episode.episode_id)
                loaded = reopened.load(episode.episode_id)
                self.assertEqual(loaded.manifest["schema_version"], schema_version)
                self.assertEqual(
                    loaded.manifest["context_feature_names"],
                    list(context_names),
                )
                self.assertEqual(
                    loaded.arrays["context_features"].shape,
                    (len(context_names),),
                )

    def test_schema_fifteen_evidence_contract_remains_verifiable(
        self,
    ) -> None:
        episode = self.episode("schema-15-evidence")
        target = self.ledger.append(episode)
        arrays_path = target / "sensitivity_arrays.npz"
        manifest_path = target / "manifest.json"
        checksums_path = target / "checksums.json"

        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["observation_evidence_by_metric"] = arrays.pop(
            "observation_source_fraction_by_metric"
        )
        arrays.pop("observation_verified_evidence_by_metric")
        arrays.pop("background_verified_evidence_by_metric")
        np.savez_compressed(arrays_path, **arrays)
        arrays_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()

        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["schema_version"] = 15
        _drop_verification_lineage(manifest)
        manifest["units"]["observation_evidence_by_metric"] = manifest["units"].pop(
            "observation_source_fraction_by_metric"
        )
        manifest["units"].pop("observation_verified_evidence_by_metric")
        manifest["units"].pop("background_verified_evidence_by_metric")
        joint_trust = manifest["trust_components"].pop("observation_verified_evidence")
        manifest["trust_components"].update(
            path_evidence=1.0,
            observation_evidence=joint_trust,
        )
        manifest["arrays"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        }
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        checksums = json.loads(checksums_path.read_text("utf-8"))
        checksums["manifest.json"] = manifest_hash
        checksums["sensitivity_arrays.npz"] = arrays_hash
        checksums_path.write_text(
            json.dumps(
                checksums,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute("DROP TRIGGER episodes_no_update")
            connection.execute(
                """
                UPDATE episodes
                SET manifest_sha256 = ?, arrays_sha256 = ?
                WHERE episode_id = ?
                """,
                (manifest_hash, arrays_hash, episode.episode_id),
            )

        reopened = EpisodeLedger(self.root)
        reopened.verify(episode.episode_id)
        loaded = reopened.load(episode.episode_id)
        self.assertEqual(loaded.manifest["schema_version"], 15)
        self.assertIn("observation_evidence_by_metric", loaded.arrays)
        self.assertNotIn(
            "observation_verified_evidence_by_metric",
            loaded.arrays,
        )

    def test_duplicate_episode_id_is_rejected(self) -> None:
        episode = self.episode()
        self.ledger.append(episode)

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.ledger.append(episode)

        self.assertEqual(len(self.ledger.list_episodes()), 1)
        self.assertEqual(
            {path.name for path in self.ledger.episodes_dir.iterdir()},
            {episode.episode_id},
        )
        self.ledger.verify(episode.episode_id)

    def test_invalid_and_traversal_episode_ids_are_rejected(self) -> None:
        for episode_id in ("", ".", "..", "../outside", "nested/outside"):
            with self.subTest(episode_id=episode_id):
                with self.assertRaisesRegex(ValueError, "episode_id"):
                    self.episode(episode_id)
                with self.assertRaisesRegex(ValueError, "episode_id"):
                    self.ledger.load(episode_id)

        self.assertEqual(list(self.ledger.episodes_dir.iterdir()), [])

    def test_file_corruption_is_detected(self) -> None:
        episode = self.episode()
        target = self.ledger.append(episode)
        manifest_path = target / "manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

        with self.assertRaisesRegex(
            ValueError,
            r"checksum mismatch: .*/manifest\.json",
        ):
            self.ledger.verify(episode.episode_id)
        with self.assertRaises(ValueError):
            self.ledger.load(episode.episode_id)

    def test_forecast_run_digest_is_validated_independently_of_contract(
        self,
    ) -> None:
        episode = self.episode("invalid-run-provenance")
        target = self.ledger.append(episode)
        manifest_path = target / "manifest.json"
        checksums_path = target / "checksums.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        original_contract_hash = manifest["contract_hash"]
        manifest["forecast_run_digest"] = "not-a-sha256"
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        checksums = json.loads(checksums_path.read_text("utf-8"))
        checksums["manifest.json"] = manifest_hash
        checksums_path.write_text(
            json.dumps(
                checksums,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with sqlite3.connect(self.ledger.index_path) as connection:
            connection.execute("DROP TRIGGER episodes_no_update")
            connection.execute(
                """
                UPDATE episodes
                SET manifest_sha256 = ?
                WHERE episode_id = ?
                """,
                (manifest_hash, episode.episode_id),
            )

        self.assertEqual(manifest["contract_hash"], original_contract_hash)
        with self.assertRaisesRegex(
            ValueError,
            "forecast_run_digest is invalid",
        ):
            EpisodeLedger(self.root).verify(episode.episode_id)

    def test_forecast_run_identity_does_not_change_contract_compatibility(
        self,
    ) -> None:
        changed_snapshot = replace(
            self.snapshot,
            forecast_run_digest="f" * 64,
        )
        changed_episode = replace(
            self.episode("different-run"),
            snapshot=changed_snapshot,
        )

        self.assertEqual(
            changed_episode.contract.digest,
            self.episode().contract.digest,
        )
        self.assertNotEqual(
            changed_episode.forecast_run_digest,
            self.episode().forecast_run_digest,
        )

    def test_invalid_snapshot_forecast_run_digest_is_rejected(self) -> None:
        for index, invalid_digest in enumerate(("invalid", None)):
            with self.subTest(invalid_digest=invalid_digest):
                malformed = replace(
                    self.snapshot,
                    forecast_run_digest=invalid_digest,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "forecast_run_digest must be a SHA-256 digest",
                ):
                    replace(
                        self.episode(f"bad-run-digest-{index}"),
                        snapshot=malformed,
                    )

    def test_untracked_episode_file_is_detected(self) -> None:
        episode = self.episode("extra-file")
        target = self.ledger.append(episode)
        (target / "untracked.bin").write_bytes(b"not part of the contract")

        with self.assertRaisesRegex(ValueError, "file set"):
            self.ledger.verify(episode.episode_id)

    def test_sqlite_rows_cannot_be_updated_or_deleted(self) -> None:
        episode = self.episode()
        self.ledger.append(episode)
        statements = (
            (
                "episodes",
                "UPDATE episodes SET radar_id = radar_id WHERE episode_id = ?",
            ),
            ("episodes", "DELETE FROM episodes WHERE episode_id = ?"),
            (
                "episode_impacts",
                "UPDATE episode_impacts SET forecast_score = forecast_score "
                "WHERE episode_id = ?",
            ),
            (
                "episode_impacts",
                "DELETE FROM episode_impacts WHERE episode_id = ?",
            ),
        )

        for table, statement in statements:
            with self.subTest(statement=statement):
                with sqlite3.connect(self.ledger.index_path) as connection:
                    with self.assertRaisesRegex(
                        sqlite3.IntegrityError,
                        rf"{table} rows are immutable",
                    ):
                        connection.execute(statement, (episode.episode_id,))

        self.ledger.verify(episode.episode_id)
        self.assertEqual(len(self.ledger.list_impacts(episode.episode_id)), 18)

        with sqlite3.connect(self.ledger.index_path) as connection:
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "episode_impacts rows are immutable",
            ):
                connection.execute(
                    """
                    INSERT INTO episode_impacts VALUES (
                        ?, 999, 'log_echo_mse', 0, 'injected',
                        0.0, 0.0, 0.0, 0.0
                    )
                    """,
                    (episode.episode_id,),
                )

    def test_malformed_or_contradictory_snapshot_is_rejected(self) -> None:
        invalid_confidence = self.snapshot.forecast_confidence.clone()
        invalid_confidence[0, 0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "forecast_confidence"):
            self.ledger.append(
                replace(
                    self.episode("bad-confidence"),
                    snapshot=replace(
                        self.snapshot,
                        forecast_confidence=invalid_confidence,
                    ),
                )
            )

        invalid_evidence = self.snapshot.path_evidence_by_metric.clone()
        invalid_evidence[0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "metric evidence"):
            self.ledger.append(
                replace(
                    self.episode("bad-evidence"),
                    snapshot=replace(
                        self.snapshot,
                        path_evidence_by_metric=invalid_evidence,
                    ),
                )
            )

        invalid_joint_path = self.snapshot.path_evidence_by_metric.clone()
        invalid_joint_observation = (
            self.snapshot.observation_verified_evidence_by_metric.clone()
        )
        invalid_joint_background = (
            self.snapshot.background_verified_evidence_by_metric.clone()
        )
        invalid_joint_path[0, 0] = 0.5
        invalid_joint_observation[0, 0] = 0.4
        invalid_joint_background[0, 0] = 0.2
        with self.assertRaisesRegex(ValueError, "channels do not close"):
            self.ledger.append(
                replace(
                    self.episode("bad-joint-evidence"),
                    snapshot=replace(
                        self.snapshot,
                        path_evidence_by_metric=invalid_joint_path,
                        observation_verified_evidence_by_metric=(
                            invalid_joint_observation
                        ),
                        background_verified_evidence_by_metric=(
                            invalid_joint_background
                        ),
                    ),
                )
            )

        malformed = replace(
            self.snapshot,
            forecast_sensitivity=torch.zeros(1),
        )
        with self.assertRaisesRegex(ValueError, "forecast_sensitivity"):
            self.ledger.append(replace(self.episode("bad-shape"), snapshot=malformed))

        direct = self.snapshot.direct.maps.clone()
        direct[0, 0, 0, 0] = 1.0
        contradictory = replace(
            self.snapshot,
            direct=replace(self.snapshot.direct, maps=direct),
        )
        with self.assertRaisesRegex(ValueError, "map and norm disagree"):
            self.ledger.append(
                replace(self.episode("bad-direct"), snapshot=contradictory)
            )

        invalid_baseline = replace(
            self.snapshot,
            baseline_scores=torch.ones_like(self.snapshot.forecast_scores),
        )
        with self.assertRaisesRegex(ValueError, "baseline lineage contract"):
            self.ledger.append(
                replace(
                    self.episode("bad-baseline"),
                    snapshot=invalid_baseline,
                )
            )

        reward = torch.zeros_like(self.snapshot.forecast_scores)
        invalid_reward = replace(
            self.snapshot,
            direct=replace(self.snapshot.direct, reward=reward),
        )
        with self.assertRaisesRegex(
            ValueError,
            "baseline lineage contract",
        ):
            self.ledger.append(
                replace(
                    self.episode("bad-reward"),
                    snapshot=invalid_reward,
                )
            )

    def test_append_uses_one_owned_tensor_snapshot(self) -> None:
        scores = self.snapshot.forecast_scores.clone()
        original_scores = scores.clone()
        episode = replace(
            self.episode("owned-copy"),
            snapshot=replace(self.snapshot, forecast_scores=scores),
        )
        original_save = np.savez_compressed

        def save_then_mutate(*args, **kwargs):
            result = original_save(*args, **kwargs)
            scores.fill_(999.0)
            return result

        with patch(
            "advar.ledger.np.savez_compressed",
            side_effect=save_then_mutate,
        ):
            self.ledger.append(episode)

        loaded = self.ledger.load(episode.episode_id)
        np.testing.assert_array_equal(
            loaded.arrays["forecast_scores"],
            original_scores.numpy(),
        )
        indexed_scores = {
            row["forecast_score"]
            for row in self.ledger.list_impacts(episode.episode_id)
        }
        self.assertEqual(indexed_scores, {float(original_scores[0, 0])})

    def test_same_id_concurrent_append_commits_once(self) -> None:
        episode = self.episode()

        def append():
            try:
                return self.ledger.append(episode)
            except FileExistsError as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: append(), range(2)))

        self.assertEqual(sum(isinstance(result, Path) for result in results), 1)
        self.assertEqual(
            sum(isinstance(result, FileExistsError) for result in results),
            1,
        )
        self.assertEqual(len(self.ledger.list_episodes()), 1)
        self.ledger.verify(episode.episode_id)

    def test_distinct_certificate_requests_form_one_linear_chain(self) -> None:
        """Exercise the chain-head lock without a same-promotion UNIQUE shortcut."""

        def append_distinct(marker: str) -> tuple[int, str, str]:
            promotion_digest = ledger_module._json_digest(
                {"contract": "concurrent-promotion-fixture-v1", "marker": marker}
            )
            with sqlite3.connect(self.ledger.index_path, timeout=10.0) as connection:
                connection.execute("BEGIN IMMEDIATE")
                head = connection.execute(
                    "SELECT ledger_instance_digest,sequence_number,"
                    "certificate_digest FROM deployment_certificate_chain_head "
                    "WHERE singleton = 1"
                ).fetchone()
                assert head is not None
                sequence = int(head[1]) + 1
                previous = str(head[2])
                certificate_digest = ledger_module._json_digest(
                    {
                        "contract": "concurrent-certificate-fixture-v1",
                        "ledger_instance_digest": head[0],
                        "sequence_number": sequence,
                        "previous_certificate_digest": previous,
                        "promotion_evidence_digest": promotion_digest,
                    }
                )
                chain_digest = ledger_module._json_digest(
                    {
                        "contract": "concurrent-chain-fixture-v1",
                        "certificate_digest": certificate_digest,
                        "previous_certificate_digest": previous,
                    }
                )
                connection.execute(
                    "INSERT INTO neural_prior_promotion_deployment_certificates_v3 "
                    "(certificate_digest,ledger_instance_digest,sequence_number,"
                    "promotion_evidence_digest,previous_certificate_digest,"
                    "ledger_chain_head_digest,payload_json,issued_at,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        certificate_digest,
                        head[0],
                        sequence,
                        promotion_digest,
                        previous,
                        chain_digest,
                        "{}",
                        "2026-08-14T00:00:00Z",
                        "2026-08-14T00:00:00Z",
                    ),
                )
                updated = connection.execute(
                    "UPDATE deployment_certificate_chain_head SET "
                    "sequence_number=?,certificate_digest=?,"
                    "ledger_chain_head_digest=?,updated_at=? "
                    "WHERE singleton=1 AND sequence_number=? AND "
                    "certificate_digest=?",
                    (
                        sequence,
                        certificate_digest,
                        chain_digest,
                        "2026-08-14T00:00:00Z",
                        head[1],
                        previous,
                    ),
                )
                if updated.rowcount != 1:
                    raise AssertionError("certificate chain forked")
            return sequence, certificate_digest, previous

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(append_distinct, ("A", "B")))
        ordered = tuple(sorted(results))
        self.assertEqual(tuple(item[0] for item in ordered), (1, 2))
        self.assertEqual(
            ordered[0][2],
            ledger_module._PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST,
        )
        self.assertEqual(ordered[1][2], ordered[0][1])

    def test_previous_scalar_schema_is_migrated(self) -> None:
        root = self.root / "upgrade"
        EpisodeLedger(root)
        with sqlite3.connect(root / "index.sqlite") as connection:
            for trigger in (
                "episode_impacts_no_update",
                "episode_impacts_no_delete",
                "episode_impacts_no_late_insert",
            ):
                connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            connection.execute("DROP TABLE episode_impacts")
            connection.execute(
                """
                CREATE TABLE episode_impacts (
                    episode_id TEXT NOT NULL,
                    lead_minutes INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    input_offset_minutes INTEGER NOT NULL,
                    direct_path_status TEXT NOT NULL,
                    forecast_score REAL NOT NULL,
                    direct_sensitivity_norm REAL NOT NULL,
                    direct_impact REAL,
                    direct_normalized_reward REAL,
                    PRIMARY KEY (
                        episode_id, lead_minutes, metric_name,
                        input_offset_minutes
                    ),
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                )
                """
            )
            connection.execute("PRAGMA user_version = 1")

        upgraded = EpisodeLedger(root)
        episode = self.episode("after-upgrade")
        upgraded.append(episode)
        upgraded.verify(episode.episode_id)
        self.assertEqual(len(upgraded.list_impacts(episode.episode_id)), 18)
        with sqlite3.connect(root / "index.sqlite") as connection:
            columns = {
                row[1]: row
                for row in connection.execute("PRAGMA table_info(episode_impacts)")
            }
            schema = connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = 'episode_impacts'
                """
            ).fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(columns["forecast_score"][3], 0)
        self.assertEqual(columns["direct_sensitivity_norm"][3], 0)
        self.assertIn("DEFERRABLE INITIALLY DEFERRED", schema)
        self.assertEqual(version, 42)

    def test_operational_raw_resolution_history_is_append_only(self) -> None:
        ledger = EpisodeLedger(self.root / "operational-raw-history")
        authority = Ed25519PrivateKey.generate()
        slot = "1" * 64

        def entry(
            *,
            identity: str,
            kind: str,
            previous: str,
            transition: str,
            ordinal: int,
        ):
            return promotion_module.OperationalRawResolutionHistoryEntry.issue(
                provenance_plan_digest=f"{ordinal:x}" * 64,
                slot_digest=slot,
                resolution_identity_digest=identity,
                resolution_kind=kind,
                previous_entry_digest=previous,
                transition=transition,
                reason=f"audited transition {ordinal}",
                issued_at=f"2026-08-16T00:00:0{ordinal}Z",
                authority_id="analysis-processor",
                authority_private_key=authority,
            )

        invalid_first = entry(
            identity="2" * 64,
            kind="missing",
            previous=promotion_module.OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST,
            transition="correction",
            ordinal=1,
        )
        with sqlite3.connect(ledger.index_path) as connection:
            with self.assertRaisesRegex(ValueError, "equivocated"):
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=invalid_first,
                    raw_resolution_receipt_digest="9" * 64,
                    recorded_at="2026-08-16T00:00:01Z",
                )

        mutable = entry(
            identity="2" * 64,
            kind="missing",
            previous=promotion_module.OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST,
            transition="original",
            ordinal=1,
        )
        object.__setattr__(mutable, "reason", "mutated after signing")
        with sqlite3.connect(ledger.index_path) as connection:
            with self.assertRaisesRegex(ValueError, "invalid|digest|unsigned"):
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=mutable,
                    raw_resolution_receipt_digest="8" * 64,
                    recorded_at="2026-08-16T00:00:01Z",
                    expected_authority_id="analysis-processor",
                    expected_authority_public_key_hex=(
                        authority.public_key().public_bytes_raw().hex()
                    ),
                )

        alternate_authority = Ed25519PrivateKey.generate()
        alternate = promotion_module.OperationalRawResolutionHistoryEntry.issue(
            provenance_plan_digest="1" * 64,
            slot_digest=slot,
            resolution_identity_digest="2" * 64,
            resolution_kind="missing",
            previous_entry_digest=(
                promotion_module.OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST
            ),
            transition="original",
            reason="alternate approved processor",
            issued_at="2026-08-16T00:00:01Z",
            authority_id="alternate-analysis-processor",
            authority_private_key=alternate_authority,
        )
        with sqlite3.connect(ledger.index_path) as connection:
            with self.assertRaisesRegex(ValueError, "disagrees with its plan"):
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=alternate,
                    raw_resolution_receipt_digest="7" * 64,
                    recorded_at="2026-08-16T00:00:01Z",
                    expected_authority_id="analysis-processor",
                    expected_authority_public_key_hex=(
                        authority.public_key().public_bytes_raw().hex()
                    ),
                )

        transitions = (
            ("2" * 64, "missing", "original"),
            ("3" * 64, "resolved", "correction"),
            ("3" * 64, "resolved", "reuse"),
            ("4" * 64, "resolved", "supersession"),
            ("5" * 64, "missing", "cancellation"),
        )
        previous = promotion_module.OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST
        expected = []
        for ordinal, (identity, kind, transition) in enumerate(
            transitions, start=1
        ):
            current = entry(
                identity=identity,
                kind=kind,
                previous=previous,
                transition=transition,
                ordinal=ordinal,
            )
            with sqlite3.connect(ledger.index_path) as connection:
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=current,
                    raw_resolution_receipt_digest=f"{ordinal + 9:x}" * 64,
                    recorded_at=f"2026-08-16T00:01:0{ordinal}Z",
                    expected_authority_id="analysis-processor",
                    expected_authority_public_key_hex=(
                        authority.public_key().public_bytes_raw().hex()
                    ),
                )
            expected.append(current)
            previous = current.entry_digest

        retained = ledger.load_operational_raw_resolution_history(
            slot,
            expected_authority_id="analysis-processor",
            expected_authority_public_key_hex=(
                authority.public_key().public_bytes_raw().hex()
            ),
        )
        self.assertEqual(retained, tuple(expected))

        poisoned_slot = "c" * 64
        attacker_key = Ed25519PrivateKey.generate()
        poisoned = promotion_module.OperationalRawResolutionHistoryEntry.issue(
            provenance_plan_digest="1" * 64,
            slot_digest=poisoned_slot,
            resolution_identity_digest="7" * 64,
            resolution_kind="resolved",
            previous_entry_digest=(
                promotion_module.OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST
            ),
            transition="original",
            reason="unauthorized predecessor",
            issued_at="2026-08-16T00:02:00Z",
            authority_id="attacker",
            authority_private_key=attacker_key,
        )
        with sqlite3.connect(ledger.index_path) as connection:
            connection.execute(
                "INSERT INTO operational_raw_resolution_history "
                "(slot_digest,sequence_number,entry_digest,"
                "previous_entry_digest,provenance_plan_digest,"
                "resolution_identity_digest,resolution_kind,transition,"
                "entry_json,raw_resolution_receipt_digest,recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    poisoned.slot_digest,
                    1,
                    poisoned.entry_digest,
                    poisoned.previous_entry_digest,
                    poisoned.provenance_plan_digest,
                    poisoned.resolution_identity_digest,
                    poisoned.resolution_kind,
                    poisoned.transition,
                    json.dumps(
                        poisoned.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "8" * 64,
                    "2026-08-16T00:02:01Z",
                ),
            )
        authorized_successor = promotion_module.OperationalRawResolutionHistoryEntry.issue(
            provenance_plan_digest="1" * 64,
            slot_digest=poisoned_slot,
            resolution_identity_digest="8" * 64,
            resolution_kind="resolved",
            previous_entry_digest=poisoned.entry_digest,
            transition="supersession",
            reason="must not extend poisoned predecessor",
            issued_at="2026-08-16T00:02:02Z",
            authority_id="analysis-processor",
            authority_private_key=authority,
        )
        with sqlite3.connect(ledger.index_path) as connection:
            with self.assertRaisesRegex(ValueError, "row changed"):
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=authorized_successor,
                    raw_resolution_receipt_digest="9" * 64,
                    recorded_at="2026-08-16T00:02:03Z",
                    expected_authority_id="analysis-processor",
                    expected_authority_public_key_hex=(
                        authority.public_key().public_bytes_raw().hex()
                    ),
                )
        forged_previous = entry(
            identity="6" * 64,
            kind="resolved",
            previous="f" * 64,
            transition="correction",
            ordinal=6,
        )
        with sqlite3.connect(ledger.index_path) as connection:
            with self.assertRaisesRegex(ValueError, "equivocated"):
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=forged_previous,
                    raw_resolution_receipt_digest="f" * 64,
                    recorded_at="2026-08-16T00:01:06Z",
                    expected_authority_id="analysis-processor",
                    expected_authority_public_key_hex=(
                        authority.public_key().public_bytes_raw().hex()
                    ),
                )

        backdated = entry(
            identity="6" * 64,
            kind="resolved",
            previous=previous,
            transition="correction",
            ordinal=1,
        )
        with sqlite3.connect(ledger.index_path) as connection:
            with self.assertRaisesRegex(ValueError, "chronology regressed"):
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=backdated,
                    raw_resolution_receipt_digest="e" * 64,
                    recorded_at="2026-08-16T00:01:07Z",
                    expected_authority_id="analysis-processor",
                    expected_authority_public_key_hex=(
                        authority.public_key().public_bytes_raw().hex()
                    ),
                )

    def test_analysis_provenance_sql_requires_prepared_immutable_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            committed_at = "2026-08-16T00:00:00+00:00"
            values = (
                "1" * 64,
                "operational",
                "2" * 64,
                "sql-state-case",
                "3" * 64,
                "4" * 64,
                "{}",
                "5" * 64,
                "6" * 64,
                str(Path(directory) / "artifact"),
                "7" * 64,
                None,
                committed_at,
                1,
                "active",
                committed_at,
                committed_at,
                None,
            )
            statement = (
                "INSERT INTO analysis_input_provenance_commits "
                "(artifact_digest,provenance_kind,provenance_plan_digest,"
                "case_id,input_plan_digest,raw_resolution_receipt_digest,"
                "payload_json,arrays_sha256,metadata_sha256,path,"
                "raw_ingestor_trust_store_digest,raw_trust_validated_at,"
                "committed_at,usable,status,payload_committed_at,"
                "activated_at,expired_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?)"
            )
            with sqlite3.connect(ledger.index_path) as connection:
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "inserted prepared",
                ):
                    connection.execute(statement, values)
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "inserted prepared",
                ):
                    connection.execute(
                        "INSERT INTO neural_prior_analysis_input_provenance "
                        "(artifact_digest,holdout_plan_digest,case_id,"
                        "input_plan_digest,global_resolution_receipt_digest,"
                        "payload_json,arrays_sha256,metadata_sha256,path,"
                        "raw_ingestor_trust_store_digest,"
                        "raw_trust_validated_at,committed_at,usable,status,"
                        "payload_committed_at,activated_at,expired_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            "8" * 64,
                            "9" * 64,
                            "holdout-direct-active",
                            "a" * 64,
                            "b" * 64,
                            "{}",
                            "c" * 64,
                            "d" * 64,
                            str(Path(directory) / "holdout-artifact"),
                            "e" * 64,
                            committed_at,
                            committed_at,
                            1,
                            "active",
                            committed_at,
                            committed_at,
                            None,
                        ),
                    )
                prepared = list(values)
                prepared[13] = 0
                prepared[14] = "prepared"
                prepared[16] = None
                connection.execute(statement, tuple(prepared))
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "state transition",
                ):
                    connection.execute(
                        "UPDATE analysis_input_provenance_commits "
                        "SET case_id = ? WHERE artifact_digest = ?",
                        ("swapped-case", "1" * 64),
                    )


if __name__ == "__main__":
    unittest.main()
