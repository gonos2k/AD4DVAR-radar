"""Append-only, checksum-verified storage for sensitivity episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .sensitivity import CONTEXT_FEATURE_NAMES, SensitivitySnapshot


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_EPISODE_FILES = {"manifest.json", "sensitivity_arrays.npz"}
_INDEX_SCHEMA_VERSION = 3
_SCHEMA_ONE_CONTEXT_FEATURE_NAMES = (
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


@dataclass(frozen=True)
class ModelContract:
    """Versions that determine whether stored gradients are compatible."""

    model_commit: str
    residual_contract_version: str
    forecast_metric_version: str
    observation_contract_version: str
    forecast_integrator_version: str
    grid_geometry_version: str
    radar_qc_version: str
    nowcast_config_digest: str
    sensitivity_config_digest: str

    def __post_init__(self) -> None:
        if not all(asdict(self).values()):
            raise ValueError("all model contract fields must be non-empty")

    @property
    def digest(self) -> str:
        return _model_contract_digest(self, schema_version=3)


@dataclass(frozen=True)
class SensitivityEpisode:
    """One finalized forecast/verification/sensitivity experience."""

    episode_id: str
    issue_time: str
    radar_id: str
    contract: ModelContract
    snapshot: SensitivitySnapshot
    action_features: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _validate_episode_id(self.episode_id)
        issue_time = datetime.fromisoformat(self.issue_time)
        if issue_time.tzinfo is None:
            raise ValueError("issue_time must include a timezone")
        if not self.radar_id:
            raise ValueError("radar_id must be non-empty")
        if not all(math.isfinite(value) for value in self.action_features):
            raise ValueError("action_features must be finite")
        if (
            self.contract.nowcast_config_digest
            != self.snapshot.nowcast_config_digest
            or self.contract.sensitivity_config_digest
            != self.snapshot.sensitivity_config_digest
        ):
            raise ValueError(
                "model contract config digests must match the sensitivity snapshot"
            )


@dataclass(frozen=True)
class LoadedEpisode:
    """Verified data loaded without Python object deserialization."""

    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


class EpisodeLedger:
    """Immutable M0 storage backed by SQLite, JSON, and NPZ."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.episodes_dir = self.root / "episodes"
        self.index_path = self.root / "index.sqlite"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_index()

    def append(self, episode: SensitivityEpisode) -> Path:
        """Append one episode; the SQLite commit makes it visible."""

        episode = _owned_episode_copy(episode)
        _validate_m0_snapshot(episode.snapshot)
        target = self.episodes_dir / episode.episode_id
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{episode.episode_id}.",
                dir=self.episodes_dir,
            )
        )
        published = False
        connection: sqlite3.Connection | None = None
        try:
            arrays = _snapshot_arrays(episode)
            arrays_path = temporary / "sensitivity_arrays.npz"
            np.savez_compressed(arrays_path, **arrays)

            manifest = _episode_manifest(episode, arrays)
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(_json_text(manifest), encoding="utf-8")

            checksums = {
                "manifest.json": _file_digest(manifest_path),
                "sensitivity_arrays.npz": _file_digest(arrays_path),
            }
            (temporary / "checksums.json").write_text(
                _json_text(checksums),
                encoding="utf-8",
            )
            for name in (*_EPISODE_FILES, "checksums.json"):
                _fsync_file(temporary / name)
            _fsync_directory(temporary)

            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            if _row_exists(connection, episode.episode_id):
                raise FileExistsError(
                    f"episode already exists: {episode.episode_id}"
                )
            if target.exists():
                raise FileExistsError(
                    f"uncommitted episode directory exists: {episode.episode_id}"
                )

            os.rename(temporary, target)
            published = True
            _fsync_directory(self.episodes_dir)
            self._insert_scalar_impacts(connection, episode)
            self._insert_episode(connection, episode, target, checksums)
            connection.commit()
        except Exception:
            if connection is not None:
                connection.rollback()
            if temporary.exists():
                shutil.rmtree(temporary)
            if published and target.exists() and not self._is_indexed(
                episode.episode_id
            ):
                shutil.rmtree(target)
            raise
        finally:
            if connection is not None:
                connection.close()
        return target

    def load(self, episode_id: str, *, verify: bool = True) -> LoadedEpisode:
        """Load a committed episode and optionally verify its integrity."""

        target, _ = self._indexed_target(episode_id)
        if verify:
            self.verify(episode_id)

        manifest = json.loads((target / "manifest.json").read_text("utf-8"))
        with np.load(target / "sensitivity_arrays.npz", allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        return LoadedEpisode(manifest=manifest, arrays=arrays)

    def verify(self, episode_id: str) -> None:
        """Raise ``ValueError`` if a committed episode was modified."""

        target, row = self._indexed_target(episode_id)
        required = _EPISODE_FILES | {"checksums.json"}
        if {path.name for path in target.iterdir()} != required:
            raise ValueError(f"episode file set is invalid: {episode_id}")
        if not all(
            (target / name).is_file()
            and not (target / name).is_symlink()
            for name in required
        ):
            raise ValueError(f"episode files are incomplete: {episode_id}")

        checksums = json.loads((target / "checksums.json").read_text("utf-8"))
        if set(checksums) != _EPISODE_FILES:
            raise ValueError("checksums.json has an invalid file set")

        manifest_hash = _file_digest(target / "manifest.json")
        arrays_hash = _file_digest(target / "sensitivity_arrays.npz")
        if manifest_hash != checksums["manifest.json"]:
            raise ValueError(f"checksum mismatch: {episode_id}/manifest.json")
        if arrays_hash != checksums["sensitivity_arrays.npz"]:
            raise ValueError(
                f"checksum mismatch: {episode_id}/sensitivity_arrays.npz"
            )
        if manifest_hash != row["manifest_sha256"]:
            raise ValueError("manifest checksum disagrees with the index")
        if arrays_hash != row["arrays_sha256"]:
            raise ValueError("array checksum disagrees with the index")

        manifest = json.loads((target / "manifest.json").read_text("utf-8"))
        _verify_manifest(manifest, episode_id, row)
        _verify_array_schema(
            target / "sensitivity_arrays.npz",
            manifest["arrays"],
        )

    def list_episodes(
        self,
        *,
        contract_hash: str | None = None,
        minimum_trust: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return searchable committed metadata, newest first."""

        query = """
            SELECT episode_id, issue_time, radar_id, contract_hash,
                   trust_score, promotion_eligible, impact_available, path
            FROM episodes
            WHERE trust_score >= ?
        """
        parameters: list[Any] = [minimum_trust]
        if contract_hash is not None:
            query += " AND contract_hash = ?"
            parameters.append(contract_hash)
        query += " ORDER BY issue_time DESC"

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_impacts(self, episode_id: str) -> list[dict[str, Any]]:
        """Return scalar lead/metric/input summaries for one episode."""

        _validate_episode_id(episode_id)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            if not _row_exists(connection, episode_id):
                raise KeyError(f"unknown episode: {episode_id}")
            rows = connection.execute(
                """
                SELECT lead_minutes, metric_name, input_offset_minutes,
                       direct_path_status, forecast_score,
                       direct_sensitivity_norm, direct_impact,
                       direct_normalized_reward
                FROM episode_impacts
                WHERE episode_id = ?
                ORDER BY lead_minutes, metric_name, input_offset_minutes
                """,
                (episode_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _initialize_index(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    issue_time TEXT NOT NULL,
                    radar_id TEXT NOT NULL,
                    contract_hash TEXT NOT NULL,
                    model_commit TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    promotion_eligible INTEGER NOT NULL CHECK(
                        promotion_eligible = 0
                    ),
                    impact_available INTEGER NOT NULL,
                    indirect_observation_sensitivity_available INTEGER NOT NULL
                        DEFAULT 0 CHECK(
                            indirect_observation_sensitivity_available = 0
                        ),
                    manifest_sha256 TEXT NOT NULL,
                    arrays_sha256 TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            _ensure_episode_impacts_schema(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS episodes_contract_time
                ON episodes(contract_hash, issue_time)
                """
            )
            connection.execute(
                f"PRAGMA user_version = {_INDEX_SCHEMA_VERSION}"
            )
            for table in ("episodes", "episode_impacts"):
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} rows are immutable');
                    END
                    """
                )
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} rows are immutable');
                    END
                    """
                )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episode_impacts_no_late_insert
                BEFORE INSERT ON episode_impacts
                WHEN EXISTS (
                    SELECT 1 FROM episodes
                    WHERE episode_id = NEW.episode_id
                )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'episode_impacts rows are immutable'
                    );
                END
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _is_indexed(self, episode_id: str) -> bool:
        with self._connect() as connection:
            return _row_exists(connection, episode_id)

    def _indexed_target(
        self,
        episode_id: str,
    ) -> tuple[Path, sqlite3.Row]:
        _validate_episode_id(episode_id)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT contract_hash, manifest_sha256, arrays_sha256,
                       indirect_observation_sensitivity_available, path
                FROM episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown episode: {episode_id}")

        expected = Path("episodes") / episode_id
        relative = Path(row["path"])
        if relative != expected:
            raise ValueError("indexed episode path violates the storage contract")
        unresolved = self.root / relative
        if unresolved.is_symlink():
            raise ValueError("indexed episode directory cannot be a symlink")
        target = unresolved.resolve()
        if target.parent != self.episodes_dir or not target.is_dir():
            raise ValueError("indexed episode path is missing or unsafe")
        return target, row

    def _insert_episode(
        self,
        connection: sqlite3.Connection,
        episode: SensitivityEpisode,
        target: Path,
        checksums: dict[str, str],
    ) -> None:
        snapshot = episode.snapshot
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, issue_time, radar_id, contract_hash,
                model_commit, trust_score, promotion_eligible,
                impact_available,
                indirect_observation_sensitivity_available,
                manifest_sha256, arrays_sha256, path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.issue_time,
                episode.radar_id,
                episode.contract.digest,
                episode.contract.model_commit,
                snapshot.trust_score,
                0,
                int(snapshot.impact_available),
                0,
                checksums["manifest.json"],
                checksums["sensitivity_arrays.npz"],
                str(target.relative_to(self.root)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def _insert_scalar_impacts(
        self,
        connection: sqlite3.Connection,
        episode: SensitivityEpisode,
    ) -> None:
        snapshot = episode.snapshot
        rows: list[tuple[Any, ...]] = []
        for lead_index, lead in enumerate(snapshot.lead_minutes):
            for metric_index, metric in enumerate(snapshot.metric_names):
                rows.append(
                    (
                        episode.episode_id,
                        lead,
                        metric,
                        0,
                        "partial_direct_latest_dbz_fixed_control",
                        _sqlite_number(
                            snapshot.forecast_scores[lead_index, metric_index]
                        ),
                        _sqlite_number(
                            snapshot.direct.norm[lead_index, metric_index]
                        ),
                        _sqlite_number(
                            None
                            if snapshot.direct.impact is None
                            else snapshot.direct.impact[
                                lead_index, metric_index
                            ]
                        ),
                        _sqlite_number(
                            None
                            if snapshot.direct.reward is None
                            else snapshot.direct.reward[
                                lead_index, metric_index
                            ]
                        ),
                    )
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
            rows,
        )


def _ensure_episode_impacts_schema(
    connection: sqlite3.Connection,
) -> None:
    """Create or transactionally upgrade the scalar impact table."""

    row = connection.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type = 'table' AND name = 'episode_impacts'
        """
    ).fetchone()
    if row is None:
        _create_episode_impacts_table(connection)
        return

    columns = {
        value[1]: value
        for value in connection.execute(
            "PRAGMA table_info(episode_impacts)"
        ).fetchall()
    }
    nullable_scores = all(
        name in columns and columns[name][3] == 0
        for name in ("forecast_score", "direct_sensitivity_norm")
    )
    deferred_foreign_key = (
        "DEFERRABLE INITIALLY DEFERRED" in row[0].upper()
    )
    if nullable_scores and deferred_foreign_key:
        return

    for trigger in (
        "episode_impacts_no_update",
        "episode_impacts_no_delete",
        "episode_impacts_no_late_insert",
    ):
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    connection.execute(
        "ALTER TABLE episode_impacts RENAME TO episode_impacts_legacy"
    )
    _create_episode_impacts_table(connection)
    connection.execute(
        """
        INSERT INTO episode_impacts (
            episode_id, lead_minutes, metric_name,
            input_offset_minutes, direct_path_status,
            forecast_score, direct_sensitivity_norm,
            direct_impact, direct_normalized_reward
        )
        SELECT
            episode_id, lead_minutes, metric_name,
            input_offset_minutes, direct_path_status,
            forecast_score, direct_sensitivity_norm,
            direct_impact, direct_normalized_reward
        FROM episode_impacts_legacy
        """
    )
    connection.execute("DROP TABLE episode_impacts_legacy")


def _create_episode_impacts_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE episode_impacts (
            episode_id TEXT NOT NULL,
            lead_minutes INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            input_offset_minutes INTEGER NOT NULL,
            direct_path_status TEXT NOT NULL,
            forecast_score REAL,
            direct_sensitivity_norm REAL,
            direct_impact REAL,
            direct_normalized_reward REAL,
            PRIMARY KEY (
                episode_id, lead_minutes, metric_name,
                input_offset_minutes
            ),
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                DEFERRABLE INITIALLY DEFERRED
        )
        """
    )


def _owned_episode_copy(episode: SensitivityEpisode) -> SensitivityEpisode:
    """Take one private snapshot before creating any persistent artifact."""

    replacements: dict[str, Any] = {}
    for field in fields(episode.snapshot):
        value = getattr(episode.snapshot, field.name)
        if isinstance(value, Tensor):
            replacements[field.name] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            replacements[field.name] = dict(value)
    direct_replacements: dict[str, Tensor] = {}
    for field in fields(episode.snapshot.direct):
        value = getattr(episode.snapshot.direct, field.name)
        if isinstance(value, Tensor):
            direct_replacements[field.name] = value.detach().cpu().clone()
    replacements["direct"] = replace(
        episode.snapshot.direct,
        **direct_replacements,
    )
    owned_snapshot = replace(episode.snapshot, **replacements)
    return replace(
        episode,
        snapshot=owned_snapshot,
        action_features=tuple(episode.action_features),
    )


def _snapshot_arrays(episode: SensitivityEpisode) -> dict[str, np.ndarray]:
    snapshot = episode.snapshot
    tensor_arrays = {
        "context_features": snapshot.context_features,
        "analysis_control": snapshot.analysis_control,
        "forecast_scores": snapshot.forecast_scores,
        "metric_available": snapshot.metric_available,
        "control_sensitivity": snapshot.control_sensitivity,
        "forecast_sensitivity": snapshot.forecast_sensitivity,
        "forecast_cap_active_mask": snapshot.forecast_cap_active_mask,
        "direct_observation_sensitivity": snapshot.direct.maps,
        "direct_observation_sensitivity_norm": snapshot.direct.norm,
        "tile_direct_sensitivity_norm": snapshot.direct.tile_norm,
        "latest_sensitivity_mask": snapshot.latest_sensitivity_mask,
    }
    optional_arrays = {
        "tile_whitened_direct_sensitivity_norm": (
            snapshot.direct.whitened_tile_norm
        ),
        "direct_observation_impact": snapshot.direct.impact,
        "tile_direct_observation_impact": snapshot.direct.tile_impact,
        "direct_normalized_reward": snapshot.direct.reward,
        "observation_std_dbz": snapshot.observation_std_dbz,
        "observation_innovation_dbz": snapshot.observation_innovation_dbz,
        "observation_innovation_mask": snapshot.observation_innovation_mask,
        "baseline_scores": snapshot.baseline_scores,
    }
    tensor_arrays.update(
        {
            name: value
            for name, value in optional_arrays.items()
            if value is not None
        }
    )
    arrays = {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in tensor_arrays.items()
    }
    arrays["action_features"] = np.asarray(
        episode.action_features,
        dtype=np.float64,
    )
    return arrays


def _episode_manifest(
    episode: SensitivityEpisode,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    snapshot = episode.snapshot
    return {
        "schema_version": 3,
        "episode_id": episode.episode_id,
        "issue_time": episode.issue_time,
        "radar_id": episode.radar_id,
        "contract": asdict(episode.contract),
        "contract_hash": episode.contract.digest,
        "metric_names": list(snapshot.metric_names),
        "lead_minutes": list(snapshot.lead_minutes),
        "full_map_lead_minutes": list(snapshot.full_map_lead_minutes),
        "tile_size": snapshot.tile_size,
        "context_feature_names": list(snapshot.context_feature_names),
        "sensitivity_scope": _sensitivity_scope(),
        "units": {
            "forecast_sensitivity": "d_error/d_linear_echo",
            "control_sensitivity": (
                "d_error/d_[dy_px,dx_px,log_growth_per_step]"
            ),
            "direct_observation_sensitivity": "d_error/d_dbz",
            "direct_observation_impact": "error_change",
        },
        "whitened_tile_norm_available": (
            snapshot.whitened_tile_norm_available
        ),
        "indirect_observation_sensitivity_available": False,
        "total_observation_sensitivity_available": False,
        "impact_available": snapshot.impact_available,
        "reward_available": snapshot.reward_available,
        "reward_epsilon": snapshot.reward_epsilon,
        "trust_components": snapshot.trust_components,
        "trust_score": snapshot.trust_score,
        "promotion_eligible": False,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in arrays.items()
        },
    }


def _sensitivity_scope() -> dict[str, str]:
    return {
        "input_0_minutes": "partial_direct_latest_dbz_fixed_control",
        "indirect_analysis_path": (
            "unavailable_implicit_variational_fso_not_implemented"
        ),
        "total_observation_sensitivity": "unavailable",
    }


def _verify_manifest(
    manifest: dict[str, Any],
    episode_id: str,
    row: sqlite3.Row,
) -> None:
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2, 3):
        raise ValueError("unsupported episode schema")
    if manifest.get("episode_id") != episode_id:
        raise ValueError("manifest episode_id disagrees with the index")
    contract_values = manifest.get("contract")
    if not isinstance(contract_values, dict):
        raise ValueError("manifest contains an invalid contract")
    if schema_version < 3:
        current_fields = {field.name for field in fields(ModelContract)}
        legacy_fields = current_fields - {
            "nowcast_config_digest",
            "sensitivity_config_digest",
        }
        if (
            set(contract_values) != legacy_fields
            or not all(contract_values.values())
        ):
            raise ValueError("manifest contains an invalid contract")
        contract_hash = _json_digest(
            {
                "contract_schema_version": schema_version,
                **contract_values,
            }
        )
    else:
        try:
            contract = ModelContract(**contract_values)
        except (TypeError, ValueError) as error:
            raise ValueError("manifest contains an invalid contract") from error
        contract_hash = _model_contract_digest(
            contract,
            schema_version=schema_version,
        )
    if contract_hash != manifest.get("contract_hash"):
        raise ValueError("manifest contract hash is invalid")
    if contract_hash != row["contract_hash"]:
        raise ValueError("contract hash disagrees with the index")
    if manifest.get("indirect_observation_sensitivity_available") is not False:
        raise ValueError("M0 episodes cannot contain indirect sensitivity")
    if manifest.get("total_observation_sensitivity_available") is not False:
        raise ValueError("M0 episodes cannot contain total sensitivity")
    if manifest.get("promotion_eligible") is not False:
        raise ValueError("M0 episodes cannot be promoted automatically")
    reward_epsilon = manifest.get("reward_epsilon")
    if (
        not isinstance(reward_epsilon, (int, float))
        or isinstance(reward_epsilon, bool)
        or not math.isfinite(reward_epsilon)
        or reward_epsilon <= 0
    ):
        raise ValueError("manifest reward_epsilon must be positive")
    if row["indirect_observation_sensitivity_available"] != 0:
        raise ValueError("index contains an invalid indirect sensitivity flag")
    if not isinstance(manifest.get("arrays"), dict):
        raise ValueError("manifest array schema is missing")
    _verify_manifest_layout(manifest)


def _verify_manifest_layout(manifest: dict[str, Any]) -> None:
    arrays = manifest["arrays"]
    schema_version = manifest["schema_version"]
    core = {
        "context_features",
        "analysis_control",
        "forecast_scores",
        "metric_available",
        "control_sensitivity",
        "forecast_sensitivity",
        "forecast_cap_active_mask",
        "direct_observation_sensitivity",
        "direct_observation_sensitivity_norm",
        "tile_direct_sensitivity_norm",
        "latest_sensitivity_mask",
        "action_features",
    }
    optional = {
        "tile_whitened_direct_sensitivity_norm",
        "direct_observation_impact",
        "tile_direct_observation_impact",
        "direct_normalized_reward",
        "observation_std_dbz",
        "observation_innovation_dbz",
        "observation_innovation_mask",
        "baseline_scores",
    }
    array_names = set(arrays)
    if schema_version == 1:
        context_names = _SCHEMA_ONE_CONTEXT_FEATURE_NAMES
        leads = manifest.get("lead_minutes")
        if (
            not isinstance(leads, list)
            or not leads
            or type(leads[0]) is not int
            or leads[0] <= 0
        ):
            raise ValueError("schema 1 lead_minutes are invalid")
        if array_names != core | optional:
            raise ValueError("manifest arrays do not match episode schema 1")
        interval = leads[0]
        expected_scope = {
            f"input_{-2 * interval}_minutes": "no_direct_forecast_path",
            f"input_{-interval}_minutes": "no_direct_forecast_path",
            "input_0_minutes": "partial_direct_latest_dbz_fixed_control",
            "indirect_analysis_path": (
                "unavailable_implicit_variational_fso_not_implemented"
            ),
            "total_observation_sensitivity": "unavailable",
        }
    else:
        context_names = CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError("manifest arrays do not match episode schema 2")
        expected_scope = _sensitivity_scope()

    if manifest.get("context_feature_names") != list(context_names):
        raise ValueError("context features do not match the episode schema")
    if _declared_array_shape(arrays, "context_features") != (
        len(context_names),
    ):
        raise ValueError("context feature shape does not match the episode schema")
    ranks = {
        "direct_observation_sensitivity": (5, 4, 2),
        "direct_observation_sensitivity_norm": (3, 2, 2),
        "tile_direct_sensitivity_norm": (5, 4, 2),
        "tile_whitened_direct_sensitivity_norm": (5, 4, 2),
        "direct_observation_impact": (3, 2, 2),
        "tile_direct_observation_impact": (5, 4, 2),
        "direct_normalized_reward": (3, 2, 2),
        "observation_std_dbz": (3, 2, 0),
        "observation_innovation_dbz": (3, 2, 0),
        "observation_innovation_mask": (3, 2, 0),
    }
    for name, (v1_rank, v2_rank, input_axis) in ranks.items():
        if name not in arrays:
            continue
        shape = _declared_array_shape(arrays, name)
        expected_rank = v1_rank if schema_version == 1 else v2_rank
        if len(shape) != expected_rank:
            raise ValueError(
                f"{name} does not match episode schema {schema_version}"
            )
        if schema_version == 1 and shape[input_axis] != 3:
            raise ValueError(f"{name} does not preserve three input times")
    if manifest.get("sensitivity_scope") != expected_scope:
        raise ValueError("sensitivity_scope does not match the episode schema")


def _declared_array_shape(
    arrays: dict[str, Any],
    name: str,
) -> tuple[int, ...]:
    declaration = arrays.get(name)
    if not isinstance(declaration, dict):
        raise ValueError(f"manifest array declaration is missing: {name}")
    shape = declaration.get("shape")
    if not isinstance(shape, list) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError(f"manifest array shape is invalid: {name}")
    return tuple(shape)


def _verify_array_schema(
    arrays_path: Path,
    expected_schema: dict[str, Any],
) -> None:
    with np.load(arrays_path, allow_pickle=False) as data:
        if set(data.files) != set(expected_schema):
            raise ValueError("NPZ array names disagree with the manifest")
        for name in data.files:
            array = data[name]
            expected = expected_schema[name]
            if list(array.shape) != expected.get("shape"):
                raise ValueError(f"array shape mismatch: {name}")
            if str(array.dtype) != expected.get("dtype"):
                raise ValueError(f"array dtype mismatch: {name}")


def _validate_m0_snapshot(snapshot: SensitivitySnapshot) -> None:
    for name, digest in (
        ("nowcast_config_digest", snapshot.nowcast_config_digest),
        ("sensitivity_config_digest", snapshot.sensitivity_config_digest),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{name} must be a SHA-256 digest")
    if not math.isfinite(snapshot.trust_score):
        raise ValueError("trust_score must be finite")
    if snapshot.reward_available and not snapshot.impact_available:
        raise ValueError("reward availability requires impact availability")
    if (
        not isinstance(snapshot.reward_epsilon, (int, float))
        or isinstance(snapshot.reward_epsilon, bool)
        or not math.isfinite(snapshot.reward_epsilon)
        or snapshot.reward_epsilon <= 0
    ):
        raise ValueError("reward_epsilon must be positive")

    leads = snapshot.lead_minutes
    metrics = snapshot.metric_names
    selected_leads = snapshot.full_map_lead_minutes
    if not leads or any(
        type(value) is not int or value <= 0 for value in leads
    ):
        raise ValueError("lead_minutes must be positive integers")
    if tuple(sorted(set(leads))) != leads:
        raise ValueError("lead_minutes must be unique and increasing")
    interval = leads[0]
    if leads != tuple(interval * step for step in range(1, len(leads) + 1)):
        raise ValueError("lead_minutes must use one uniform interval")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metric_names must be non-empty and unique")
    if any(not isinstance(name, str) or not name for name in metrics):
        raise ValueError("metric names must be non-empty strings")
    if (
        not snapshot.context_feature_names
        or len(snapshot.context_feature_names)
        != len(set(snapshot.context_feature_names))
    ):
        raise ValueError("context feature names must be non-empty and unique")
    if (
        len(set(selected_leads)) != len(selected_leads)
        or not set(selected_leads).issubset(leads)
        or any(type(value) is not int for value in selected_leads)
    ):
        raise ValueError("full-map leads must be unique forecast leads")
    if type(snapshot.tile_size) is not int or snapshot.tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")

    _require_bool_tensor(
        "latest_sensitivity_mask",
        snapshot.latest_sensitivity_mask,
    )
    if snapshot.latest_sensitivity_mask.ndim != 2:
        raise ValueError("latest_sensitivity_mask must be two-dimensional")
    height, width = snapshot.latest_sensitivity_mask.shape
    lead_count = len(leads)
    metric_count = len(metrics)
    selected_count = len(selected_leads)
    tile_rows = math.ceil(height / snapshot.tile_size)
    tile_columns = math.ceil(width / snapshot.tile_size)

    core_tensors = {
        "context_features": (len(snapshot.context_feature_names),),
        "analysis_control": (3,),
        "forecast_scores": (lead_count, metric_count),
        "metric_available": (lead_count, metric_count),
        "control_sensitivity": (lead_count, metric_count, 3),
        "forecast_sensitivity": (
            selected_count,
            metric_count,
            height,
            width,
        ),
        "forecast_cap_active_mask": (selected_count, height, width),
        "direct_observation_sensitivity": (
            selected_count,
            metric_count,
            height,
            width,
        ),
        "direct_observation_sensitivity_norm": (
            lead_count,
            metric_count,
        ),
        "tile_direct_sensitivity_norm": (
            lead_count,
            metric_count,
            tile_rows,
            tile_columns,
        ),
        "latest_sensitivity_mask": (height, width),
    }
    core_values = {
        "context_features": snapshot.context_features,
        "analysis_control": snapshot.analysis_control,
        "forecast_scores": snapshot.forecast_scores,
        "metric_available": snapshot.metric_available,
        "control_sensitivity": snapshot.control_sensitivity,
        "forecast_sensitivity": snapshot.forecast_sensitivity,
        "forecast_cap_active_mask": snapshot.forecast_cap_active_mask,
        "direct_observation_sensitivity": snapshot.direct.maps,
        "direct_observation_sensitivity_norm": snapshot.direct.norm,
        "tile_direct_sensitivity_norm": snapshot.direct.tile_norm,
        "latest_sensitivity_mask": snapshot.latest_sensitivity_mask,
    }
    for name, expected in core_tensors.items():
        value = core_values[name]
        if not isinstance(value, Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")

    for name, value in core_values.items():
        if name in {
            "metric_available",
            "forecast_cap_active_mask",
            "latest_sensitivity_mask",
        }:
            _require_bool_tensor(name, value)
        else:
            _require_float_tensor(name, value)

    _require_finite("context_features", snapshot.context_features)
    _require_finite("analysis_control", snapshot.analysis_control)

    available = snapshot.metric_available
    if not bool(torch.all(torch.isfinite(snapshot.forecast_scores[available]))):
        raise ValueError("available forecast scores must be finite")
    if not bool(torch.all(torch.isnan(snapshot.forecast_scores[~available]))):
        raise ValueError("unavailable forecast scores must be NaN")
    if not bool(
        torch.all(torch.isfinite(snapshot.control_sensitivity[available]))
    ):
        raise ValueError("available control sensitivities must be finite")
    if not bool(
        torch.all(torch.isnan(snapshot.control_sensitivity[~available]))
    ):
        raise ValueError("unavailable control sensitivities must be NaN")

    latest_norm = snapshot.direct.norm
    latest_tile_norm = snapshot.direct.tile_norm
    if not bool(torch.all(torch.isfinite(latest_norm[available]))):
        raise ValueError("available latest-frame norms must be finite")
    if not bool(torch.all(torch.isnan(latest_norm[~available]))):
        raise ValueError("unavailable latest-frame norms must be NaN")
    combined_tile_norm = torch.sqrt(
        torch.sum(latest_tile_norm.square(), dim=(-1, -2))
    )
    if not torch.allclose(
        combined_tile_norm[available],
        latest_norm[available],
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("tile norms and whole-field norms disagree")
    if not bool(torch.all(torch.isnan(latest_tile_norm[~available]))):
        raise ValueError("unavailable latest-frame tile norms must be NaN")
    if snapshot.whitened_tile_norm_available:
        whitened_latest = snapshot.direct.whitened_tile_norm
        expected = (lead_count, metric_count, tile_rows, tile_columns)
        if tuple(whitened_latest.shape) != expected:
            raise ValueError(
                "tile_whitened_direct_sensitivity_norm "
                f"must have shape {expected}"
            )
        _require_float_tensor(
            "tile_whitened_direct_sensitivity_norm",
            whitened_latest,
        )
        if snapshot.observation_std_dbz is None:
            raise ValueError("whitened summaries require observation_std_dbz")
        if tuple(snapshot.observation_std_dbz.shape) != (height, width):
            raise ValueError("observation_std_dbz must match the latest frame")
        _require_float_tensor("observation_std_dbz", snapshot.observation_std_dbz)
        _require_finite("observation_std_dbz", snapshot.observation_std_dbz)
        if bool(torch.any(snapshot.observation_std_dbz <= 0)):
            raise ValueError("observation_std_dbz must be positive")
        if not bool(torch.all(torch.isfinite(whitened_latest[available]))):
            raise ValueError("available whitened tile norms must be finite")
        if not bool(torch.all(torch.isnan(whitened_latest[~available]))):
            raise ValueError("unavailable whitened tile norms must be NaN")
    elif snapshot.observation_std_dbz is not None:
        raise ValueError(
            "observation_std_dbz requires whitened sensitivity summaries"
        )
    for position, lead in enumerate(selected_leads):
        lead_index = leads.index(lead)
        for metric_index in range(metric_count):
            is_available = bool(available[lead_index, metric_index])
            forecast_map = snapshot.forecast_sensitivity[
                position, metric_index
            ]
            direct_map = snapshot.direct.maps[position, metric_index]
            if is_available:
                _require_finite("forecast sensitivity map", forecast_map)
                _require_finite("direct sensitivity map", direct_map)
                expected_norm = torch.linalg.vector_norm(direct_map)
                stored_norm = latest_norm[lead_index, metric_index]
                if not torch.allclose(
                    expected_norm,
                    stored_norm,
                    rtol=1.0e-5,
                    atol=1.0e-7,
                ):
                    raise ValueError("direct map and norm disagree")
            elif not bool(
                torch.all(torch.isnan(forecast_map))
                and torch.all(torch.isnan(direct_map))
            ):
                raise ValueError("unavailable sensitivity maps must be NaN")

    innovation_values = (
        snapshot.observation_innovation_dbz,
        snapshot.observation_innovation_mask,
    )
    if (innovation_values[0] is None) != (innovation_values[1] is None):
        raise ValueError("innovation value and mask availability must agree")
    if innovation_values[0] is not None and innovation_values[1] is not None:
        innovation, innovation_mask = innovation_values
        if tuple(innovation.shape) != (height, width):
            raise ValueError("observation_innovation_dbz must match latest frame")
        if tuple(innovation_mask.shape) != (height, width):
            raise ValueError("observation_innovation_mask must match latest frame")
        _require_float_tensor("observation_innovation_dbz", innovation)
        _require_bool_tensor("observation_innovation_mask", innovation_mask)
        if not bool(torch.all(torch.isfinite(innovation[innovation_mask]))):
            raise ValueError("valid innovations must be finite")
        if not bool(torch.all(torch.isnan(innovation[~innovation_mask]))):
            raise ValueError("invalid innovations must be NaN")

    if snapshot.impact_available:
        if snapshot.observation_innovation_mask is None or not bool(
            torch.any(
                snapshot.observation_innovation_mask
                & snapshot.latest_sensitivity_mask
            )
        ):
            raise ValueError("impact requires a valid latest-frame innovation")
        if not bool(torch.any(snapshot.metric_available)):
            raise ValueError("impact requires an available metric")
        latest_impact = snapshot.direct.impact
        latest_tile_impact = snapshot.direct.tile_impact
        if latest_tile_impact is None:
            raise ValueError("direct impact requires tile impact")
        if tuple(latest_impact.shape) != (lead_count, metric_count):
            raise ValueError("direct_observation_impact has invalid shape")
        expected = (lead_count, metric_count, tile_rows, tile_columns)
        if tuple(latest_tile_impact.shape) != expected:
            raise ValueError("tile_direct_observation_impact has invalid shape")
        _require_float_tensor("direct_observation_impact", latest_impact)
        _require_float_tensor(
            "tile_direct_observation_impact",
            latest_tile_impact,
        )
        if not bool(torch.all(torch.isfinite(latest_impact[available]))):
            raise ValueError("available latest-frame impacts must be finite")
        if not bool(torch.all(torch.isnan(latest_impact[~available]))):
            raise ValueError("unavailable latest-frame impacts must be NaN")
        if not torch.allclose(
            latest_tile_impact.sum(dim=(-1, -2))[available],
            latest_impact[available],
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError("tile impacts and whole-field impacts disagree")
    elif snapshot.direct.tile_impact is not None:
        raise ValueError("tile impact cannot exist without direct impact")
    baseline = snapshot.baseline_scores
    baseline_available = baseline is not None
    if baseline is not None:
        if tuple(baseline.shape) != (lead_count, metric_count):
            raise ValueError("baseline_scores has invalid shape")
        _require_float_tensor("baseline_scores", baseline)
        if not bool(
            torch.all(torch.isfinite(baseline)) and torch.all(baseline >= 0)
        ):
            raise ValueError("baseline_scores must be finite and non-negative")
    expected_reward_available = snapshot.impact_available and baseline_available
    if snapshot.reward_available != expected_reward_available:
        raise ValueError("reward availability disagrees with baseline_scores")

    if snapshot.reward_available:
        reward = snapshot.direct.reward
        if tuple(reward.shape) != (lead_count, metric_count):
            raise ValueError("direct_normalized_reward has invalid shape")
        _require_float_tensor("direct_normalized_reward", reward)
        if not bool(
            torch.all(torch.isfinite(reward[snapshot.metric_available]))
        ):
            raise ValueError("available rewards must be finite")
        expected_reward = -snapshot.direct.impact / (
            baseline + snapshot.reward_epsilon
        )
        if not torch.allclose(
            reward,
            expected_reward,
            rtol=1.0e-5,
            atol=1.0e-7,
            equal_nan=True,
        ):
            raise ValueError(
                "direct_normalized_reward disagrees with impact and baseline"
            )
    if any(
        not math.isfinite(value)
        for value in snapshot.trust_components.values()
    ):
        raise ValueError("trust components must be finite")
    expected_trust = math.prod(snapshot.trust_components.values())
    if not math.isclose(
        snapshot.trust_score,
        expected_trust,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("trust_score must equal the component product")


def _require_float_tensor(name: str, value: Any) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point tensor")


def _require_bool_tensor(name: str, value: Any) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")


def _row_exists(
    connection: sqlite3.Connection,
    episode_id: str,
) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        is not None
    )


def _sqlite_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _model_contract_digest(
    contract: ModelContract,
    *,
    schema_version: int,
) -> str:
    return _json_digest(
        {
            "contract_schema_version": schema_version,
            **asdict(contract),
        }
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_episode_id(episode_id: str) -> None:
    if (
        not episode_id
        or len(episode_id) > 128
        or episode_id in {".", ".."}
        or not _SAFE_ID.fullmatch(episode_id)
    ):
        raise ValueError(
            "episode_id must be 1-128 safe filename characters and not . or .."
        )
