"""Focused A5 regressions for ledger storage contracts."""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ledger import _computed_snapshot, _contract  # noqa: E402

import advar.ledger as ledger_module  # noqa: E402
from advar.ledger import EpisodeLedger, SensitivityEpisode  # noqa: E402


class A5LedgerStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = _computed_snapshot()

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.ledger = EpisodeLedger(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def episode(self, episode_id: str, snapshot=None) -> SensitivityEpisode:
        retained = self.snapshot if snapshot is None else snapshot
        return SensitivityEpisode(
            episode_id=episode_id,
            issue_time="2026-07-26T05:00:00+00:00",
            radar_id="KTLX",
            contract=_contract(retained),
            snapshot=retained,
            action_features=(1.25, -0.5),
        )

    def test_empty_active_support_is_rejected_but_normal_missingness_is_valid(
        self,
    ) -> None:
        missing = replace(
            self.snapshot,
            direct=replace(
                self.snapshot.direct,
                impact=None,
                tile_impact=None,
                whitened_tile_norm=None,
            ),
            latest_sensitivity_mask=torch.zeros_like(
                self.snapshot.latest_sensitivity_mask
            ),
            observation_std_dbz=None,
            observation_innovation_dbz=None,
            observation_innovation_mask=None,
        )
        with self.assertRaisesRegex(ValueError, "active support"):
            self.ledger.append(self.episode("empty-support", missing))

        valid = replace(
            missing,
            latest_sensitivity_mask=torch.ones_like(
                self.snapshot.latest_sensitivity_mask
            ),
        )
        target = self.ledger.append(self.episode("missing-optional", valid))
        self.assertTrue(target.is_dir())
        self.assertFalse(valid.impact_available)

    def test_direct_maps_must_be_zero_outside_retained_support(self) -> None:
        mask = torch.ones_like(self.snapshot.latest_sensitivity_mask)
        mask[0, 0] = False
        maps = self.snapshot.direct.maps.clone()
        maps[0, 0, 0, 0] = 1.0
        norms = self.snapshot.direct.norm.clone()
        norms[2, 0] = 1.0  # full-map position 0 is the 30-minute lead.
        tile_norm = self.snapshot.direct.tile_norm.clone()
        tile_norm[2, 0, 0, 0] = 1.0
        direct = replace(
            self.snapshot.direct,
            maps=maps,
            norm=norms,
            tile_norm=tile_norm,
        )
        malformed = replace(self.snapshot, latest_sensitivity_mask=mask, direct=direct)
        with self.assertRaisesRegex(ValueError, "outside active support"):
            self.ledger.append(self.episode("outside-support", malformed))

    def test_retained_tile_l2_norms_are_nonnegative_and_map_consistent(self) -> None:
        maps = self.snapshot.direct.maps.clone()
        maps[0, 0, 0, 0] = 1.0
        norms = self.snapshot.direct.norm.clone()
        norms[2, 0] = 1.0
        tile_norm = torch.zeros(
            (*norms.shape, 2, 2),
            dtype=norms.dtype,
        )
        tile_norm[2, 0, 0, 1] = 1.0
        direct = replace(
            self.snapshot.direct,
            maps=maps,
            norm=norms,
            tile_norm=tile_norm,
            impact=None,
            tile_impact=None,
            whitened_tile_norm=None,
        )
        malformed = replace(
            self.snapshot,
            tile_size=1,
            tile_shape_yx=(1, 1),
            direct=direct,
            observation_std_dbz=None,
            observation_innovation_dbz=None,
            observation_innovation_mask=None,
        )
        with self.assertRaisesRegex(ValueError, "retained direct maps"):
            self.ledger.append(self.episode("mismatched-tile-norm", malformed))

        negative_tile_norm = tile_norm.clone()
        negative_tile_norm[2, 0, 0, 1] = -1.0
        negative = replace(
            malformed,
            direct=replace(malformed.direct, tile_norm=negative_tile_norm),
        )
        with self.assertRaisesRegex(ValueError, "tile norms must be nonnegative"):
            self.ledger.append(self.episode("negative-tile-norm", negative))

    def test_direct_impact_is_rederived_from_retained_map_and_innovation(self) -> None:
        maps = self.snapshot.direct.maps.clone()
        maps[0, 0, 0, 0] = 1.0
        norms = self.snapshot.direct.norm.clone()
        norms[2, 0] = 1.0
        tile_norm = self.snapshot.direct.tile_norm.clone()
        tile_norm[2, 0, 0, 0] = 1.0
        impact = self.snapshot.direct.impact.clone()
        tile_impact = self.snapshot.direct.tile_impact.clone()
        impact[2, 0] = 123.0
        tile_impact[2, 0, 0, 0] = 123.0
        direct = replace(
            self.snapshot.direct,
            maps=maps,
            norm=norms,
            tile_norm=tile_norm,
            impact=impact,
            tile_impact=tile_impact,
        )
        malformed = replace(self.snapshot, direct=direct)
        with self.assertRaisesRegex(ValueError, "sensitivity and innovation"):
            self.ledger.append(self.episode("mismatched-impact", malformed))

        impact[2, 0] = 0.5
        tile_impact[2, 0, 0, 0] = 0.5
        valid = replace(
            self.snapshot,
            direct=replace(
                direct,
                impact=impact,
                tile_impact=tile_impact,
            ),
        )
        self.ledger.append(self.episode("consistent-impact", valid))

    def test_episode_append_rolls_back_scalar_rows_and_directory(self) -> None:
        episode = self.episode("rollback")
        with patch.object(
            EpisodeLedger,
            "_insert_episode",
            side_effect=RuntimeError("simulated index failure"),
        ), self.assertRaisesRegex(RuntimeError, "index failure"):
            self.ledger.append(episode)

        with sqlite3.connect(self.ledger.index_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM episodes WHERE episode_id = ?",
                    (episode.episode_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM episode_impacts WHERE episode_id = ?",
                    (episode.episode_id,),
                ).fetchone()[0],
                0,
            )
        self.assertFalse((self.ledger.episodes_dir / episode.episode_id).exists())

    def test_holdout_plan_clock_abort_rolls_back_all_plan_rows(self) -> None:
        """The public deadline check must leave no partially registered plan."""

        from test_ledger import EpisodeLedgerTests

        fixture = EpisodeLedgerTests("runTest")
        fixture.setUp()
        try:
            fixture.test_holdout_plan_rechecks_clock_before_commit()
            with sqlite3.connect(fixture.ledger.index_path) as connection:
                for table in (
                    "neural_prior_holdout_plans",
                    "neural_prior_holdout_plan_experiment_bindings",
                    "neural_prior_holdout_plan_rule_bindings",
                    "neural_prior_promotion_experiment_families",
                    "promotion_sampling_unit_reservations",
                    "promotion_raw_observation_slot_reservations",
                    "training_raw_registry_entries",
                    "global_sampling_registry_entries",
                ):
                    self.assertEqual(
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
                        0,
                        table,
                    )
        finally:
            fixture.tearDown()

    def test_scalar_impact_migration_preserves_existing_rows(self) -> None:
        legacy_snapshot = replace(
            self.snapshot,
            direct=replace(
                self.snapshot.direct,
                impact=None,
                tile_impact=None,
                whitened_tile_norm=None,
            ),
            observation_std_dbz=None,
            observation_innovation_dbz=None,
            observation_innovation_mask=None,
        )
        historical = self.episode("before-upgrade", legacy_snapshot)
        self.ledger.append(historical)
        before = self.ledger.list_impacts(historical.episode_id)

        with sqlite3.connect(self.ledger.index_path) as connection:
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
            connection.executemany(
                """
                INSERT INTO episode_impacts (
                    episode_id, lead_minutes, metric_name,
                    input_offset_minutes, direct_path_status,
                    forecast_score, direct_sensitivity_norm,
                    direct_impact, direct_normalized_reward
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        historical.episode_id,
                        row["lead_minutes"],
                        row["metric_name"],
                        row["input_offset_minutes"],
                        row["direct_path_status"],
                        row["forecast_score"],
                        row["direct_sensitivity_norm"],
                        row["direct_impact"],
                        row["direct_normalized_reward"],
                    )
                    for row in before
                ],
            )
            connection.execute("PRAGMA user_version = 1")

        upgraded = EpisodeLedger(self.root)
        self.assertEqual(upgraded.list_impacts(historical.episode_id), before)
        upgraded.verify(historical.episode_id)
        upgraded.append(self.episode("after-upgrade", legacy_snapshot))

    def test_standalone_evaluation_loader_binds_declared_digest_list(self) -> None:
        evaluation_digest = "e" * 64
        evaluation_payload = {
            "contract": "prior-holdout-evaluation-v7",
            "evaluation_digest": evaluation_digest,
        }

        def insert_row(evidence_digest: str, declared: str) -> None:
            with sqlite3.connect(self.ledger.index_path) as connection:
                schema = connection.execute(
                    "PRAGMA table_info(neural_prior_promotions)"
                ).fetchall()
                columns = [str(row[1]) for row in schema]
                values: list[object] = []
                for row in schema:
                    name = str(row[1])
                    if name == "promotion_evidence_digest":
                        values.append(evidence_digest)
                    elif name == "candidate_prior_digest":
                        values.append("c" * 63 + evidence_digest[0])
                    elif name == "evaluation_digests_json":
                        values.append(json.dumps([declared]))
                    elif name == "evaluation_payloads_json":
                        values.append(json.dumps([evaluation_payload]))
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

        matching_digest = "a" * 64
        mismatched_digest = "b" * 64
        insert_row(matching_digest, evaluation_digest)
        insert_row(mismatched_digest, "0" * 64)
        loaded = self.ledger.load_neural_prior_promotion_evaluations(
            matching_digest
        )
        self.assertEqual(
            tuple(item.evaluation_digest for item in loaded),
            (evaluation_digest,),
        )
        with self.assertRaisesRegex(ValueError, "evaluation audit mismatch"):
            self.ledger.load_neural_prior_promotion_evaluations(mismatched_digest)

    def test_public_scoring_replay_retries_after_process_death_orphan(self) -> None:
        """A crash after rename leaves an orphan that the public retry removes."""

        from test_promotion import NeuralPriorPromotionTests

        fixture = NeuralPriorPromotionTests("runTest")
        original_append = EpisodeLedger.append_neural_prior_scoring_replay_bundle
        original_publish = ledger_module._publish_durable_directory
        state = {"crashed": False, "target": None}

        def publish_then_die(**kwargs):
            original_publish(**kwargs)
            state["target"] = kwargs["target"]
            if not state["crashed"]:
                state["crashed"] = True
                raise KeyboardInterrupt("simulated process death")

        def retry_append(ledger, *args, **kwargs):
            try:
                return original_append(ledger, *args, **kwargs)
            except KeyboardInterrupt:
                target = state["target"]
                assert isinstance(target, Path)
                self.assertTrue(target.is_dir())
                with sqlite3.connect(ledger.index_path) as connection:
                    self.assertIsNone(
                        connection.execute(
                            "SELECT 1 FROM neural_prior_scoring_replay_bundles "
                            "WHERE bundle_digest = ?",
                            (target.name,),
                        ).fetchone()
                    )
                return original_append(ledger, *args, **kwargs)

        with patch.object(
            ledger_module,
            "_publish_durable_directory",
            side_effect=publish_then_die,
        ), patch.object(
            EpisodeLedger,
            "append_neural_prior_scoring_replay_bundle",
            new=retry_append,
        ):
            fixture.test_full_product_semantic_replay_reaches_promotion_without_scorer_patch()

        self.assertTrue(state["crashed"])


if __name__ == "__main__":
    unittest.main()
