from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
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
from advar.nowcast import (  # noqa: E402
    NowcastConfig,
    nowcast,
)
from advar.sensitivity import (  # noqa: E402
    SensitivityConfig,
    compute_sensitivity_snapshot,
)


def _contract() -> ModelContract:
    return ModelContract(
        model_commit="model-v1",
        residual_contract_version="residual-v1",
        forecast_metric_version="metric-v1",
        observation_contract_version="observation-v1",
        forecast_integrator_version="integrator-v1",
        grid_geometry_version="grid-v1",
        radar_qc_version="qc-v1",
    )


def _computed_snapshot():
    config = NowcastConfig()
    frames = torch.full((3, 2, 2), 20.0, dtype=torch.float64)
    result = nowcast(frames, config)
    verification = frames.new_full((config.forecast_steps, 2, 2), 20.0)
    return compute_sensitivity_snapshot(
        frames,
        result,
        verification,
        nowcast_config=config,
        background_frames_dbz=frames - 0.5,
        baseline_scores=torch.ones(
            config.forecast_steps,
            1,
            dtype=frames.dtype,
        ),
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
                changed = replace(
                    contract,
                    **{field.name: f"{getattr(contract, field.name)}-changed"},
                )
                self.assertNotEqual(contract.digest, changed.digest)


class EpisodeLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = _computed_snapshot()
        cls.contract = _contract()

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
            (18, 1, 3),
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
        self.assertEqual(manifest["contract_hash"], self.contract.digest)
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
                "input_-20_minutes": "no_direct_forecast_path",
                "input_-10_minutes": "no_direct_forecast_path",
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
        self.assertEqual(len(impacts), 54)
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
                (lead, "log_echo_mse", offset)
                for lead in range(10, 181, 10)
                for offset in (-20, -10, 0)
            },
        )
        self.assertEqual(
            {
                (row["input_offset_minutes"], row["direct_path_status"])
                for row in impacts
            },
            {
                (-20, "no_direct_forecast_path"),
                (-10, "no_direct_forecast_path"),
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
        self.assertEqual(len(self.ledger.list_impacts(episode.episode_id)), 54)

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
        malformed = replace(
            self.snapshot,
            forecast_sensitivity=torch.zeros(1),
        )
        with self.assertRaisesRegex(ValueError, "forecast_sensitivity"):
            self.ledger.append(
                replace(self.episode("bad-shape"), snapshot=malformed)
            )

        direct = self.snapshot.direct_observation_sensitivity.clone()
        direct[0, 0, 0, 0, 0] = 1.0
        contradictory = replace(
            self.snapshot,
            direct_observation_sensitivity=direct,
        )
        with self.assertRaisesRegex(ValueError, "old-frame direct maps"):
            self.ledger.append(
                replace(self.episode("bad-direct"), snapshot=contradictory)
            )

        invalid_baseline = replace(
            self.snapshot,
            baseline_scores=torch.full_like(
                self.snapshot.baseline_scores,
                float("nan"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "reward availability"):
            self.ledger.append(
                replace(
                    self.episode("bad-baseline"),
                    snapshot=invalid_baseline,
                )
            )

        reward = self.snapshot.direct_normalized_reward.clone()
        reward[:, :, 2] = 123.0
        invalid_reward = replace(
            self.snapshot,
            direct_normalized_reward=reward,
        )
        with self.assertRaisesRegex(
            ValueError,
            "disagrees with impact and baseline",
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
        self.assertEqual(len(upgraded.list_impacts(episode.episode_id)), 54)
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
        self.assertEqual(version, 2)


if __name__ == "__main__":
    unittest.main()
