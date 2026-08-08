from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.ledger import (  # noqa: E402
    EpisodeLedger,
    ModelContract,
    SensitivityEpisode,
)
from advar.intervention import RealizedObservationIntervention  # noqa: E402
from advar.nowcast import (  # noqa: E402
    NowcastConfig,
    nowcast,
)
from advar.sensitivity import (  # noqa: E402
    LearningApprovalEvidence,
    SensitivityConfig,
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
            "a" * 64
            if snapshot is None
            else snapshot.nowcast_config_digest
        ),
        sensitivity_config_digest=(
            "b" * 64
            if snapshot is None
            else snapshot.sensitivity_config_digest
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


class EpisodeLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = _computed_snapshot()
        cls.contract = _contract(cls.snapshot)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.ledger = EpisodeLedger(self.root)

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
            self.snapshot.trust_components[
                "observation_verified_evidence"
            ],
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
                "input_0_minutes": (
                    "partial_direct_latest_dbz_fixed_control"
                ),
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
            {
                (lead, "log_echo_mse", 0)
                for lead in range(10, 181, 10)
            },
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
        self.assertEqual(version, 6)

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

        loaded = self.ledger.load_variational_learning_approval(
            learning_result_digest
        )
        self.assertEqual(loaded, evidence)

        intervention = RealizedObservationIntervention(
            intervention_id="radar-qc-20260808-001",
            intervention_type="realized_qc_intervention",
            action_digest="b" * 64,
            applied_time="2026-08-08T01:02:03+09:00",
            actual_input_before_digest="c" * 64,
            actual_input_after_digest="d" * 64,
            observed_outcome_digest="e" * 64,
            learning_result_digest=learning_result_digest,
            learning_approval_evidence_digest=evidence.digest,
            counterfactual_perturbation_digest="f" * 64,
            linearization_digest="0" * 64,
        )
        digest = self.ledger.append_realized_observation_intervention(
            intervention
        )
        self.assertEqual(
            self.ledger.load_realized_observation_intervention(digest),
            intervention,
        )

    def test_complete_verification_lineage_round_trips(self) -> None:
        snapshot = replace(
            self.snapshot,
            grid_time_contract_digest="1" * 64,
            verification_contract="radar-verification-bundle-v1",
            verification_bundle_digest="2" * 64,
            verification_lineage_complete=True,
            verification_valid_times=tuple(
                (
                    datetime(2026, 7, 26, tzinfo=timezone.utc)
                    + timedelta(minutes=lead)
                )
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
        manifest["context_feature_names"] = list(
            SCHEMA_ONE_CONTEXT_FEATURE_NAMES
        )
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
        arrays["context_features"] = np.concatenate(
            (context[:6], context[14:23])
        )

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
        arrays["direct_normalized_reward"] = -arrays[
            "direct_observation_impact"
        ] / arrays["baseline_scores"]
        arrays["direct_observation_impact"] = latest_axis(
            "direct_observation_impact"
        )
        arrays["tile_direct_observation_impact"] = latest_axis(
            "tile_direct_observation_impact"
        )
        arrays["direct_normalized_reward"] = latest_axis(
            "direct_normalized_reward"
        )

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
        manifest["context_feature_names"] = list(
            SCHEMA_ONE_CONTEXT_FEATURE_NAMES
        )
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
        arrays["context_features"] = np.concatenate(
            (context[:6], context[8:23])
        )
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
        arrays["context_features"] = np.concatenate(
            (context[:6], context[8:23])
        )
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
        manifest["context_feature_names"] = list(
            SCHEMA_FIVE_CONTEXT_FEATURE_NAMES
        )
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
                    arrays = {
                        name: archive[name].copy() for name in archive.files
                    }
                arrays["context_features"] = arrays["context_features"][
                    : len(context_names)
                ]
                np.savez_compressed(arrays_path, **arrays)
                arrays_hash = hashlib.sha256(
                    arrays_path.read_bytes()
                ).hexdigest()

                manifest = json.loads(manifest_path.read_text("utf-8"))
                manifest["schema_version"] = schema_version
                _drop_verification_lineage(manifest)
                if schema_version < 11:
                    manifest["contract"].pop("grid_time_contract_digest")
                manifest["context_feature_names"] = list(context_names)
                manifest["arrays"]["context_features"]["shape"] = [
                    len(context_names)
                ]
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
                self.assertEqual(
                    loaded.manifest["schema_version"], schema_version
                )
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
        manifest["units"]["observation_evidence_by_metric"] = (
            manifest["units"].pop(
                "observation_source_fraction_by_metric"
            )
        )
        manifest["units"].pop("observation_verified_evidence_by_metric")
        manifest["units"].pop("background_verified_evidence_by_metric")
        joint_trust = manifest["trust_components"].pop(
            "observation_verified_evidence"
        )
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
                "UPDATE episodes SET radar_id = radar_id "
                "WHERE episode_id = ?",
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
            self.ledger.append(
                replace(self.episode("bad-shape"), snapshot=malformed)
            )

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
                for row in connection.execute(
                    "PRAGMA table_info(episode_impacts)"
                )
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
        self.assertEqual(version, 6)


if __name__ == "__main__":
    unittest.main()
