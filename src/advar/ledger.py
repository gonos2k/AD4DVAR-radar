"""Append-only, checksum-verified storage for sensitivity episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import tempfile
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ._digest import tensor_digest
from .action_artifacts import (
    expanded_tensor_bytes,
    preflight_npz_archive,
    validate_artifact_directory,
)
from .action_contracts import canonicalize_action_frames
from .nowcast import (
    ForecastRunContract,
    _forecast_full_analysis_input_digest,
    _forecast_input_bundle_digest,
    _forecast_input_plan_resolution_digest,
)

from .sensitivity import (
    CONTEXT_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES_V13,
    LearningApprovalEvidence,
    SensitivitySnapshot,
    VariationalLearningImpact,
    _load_learning_policy_trust_store,
    validate_variational_learning_impact,
)
from .intervention import (
    InterventionInputContext,
    InterventionActionGenerator,
    OperatorActionApproval,
    ProspectiveInterventionDecision,
    ReusableInterventionPolicyEvidence,
    RealizedInterventionReceipt,
    RealizedObservationIntervention,
    RetrospectiveCounterfactualReplay,
    validate_prospective_intervention,
    validate_intervention_action_transition,
    validate_action_tensor_replay,
    validate_retrospective_counterfactual_replay,
    verify_intervention_receipt_signature,
    verify_operator_action_approval_signature,
    _compute_action_safety,
    _new_prospective_decision,
    _new_operator_action_approval,
    _new_realized_intervention_receipt,
)
from .promotion import (
    LegacyNeuralPriorCandidateManifestAuditV2,
    LegacyNeuralPriorCandidateManifestAuditV3,
    NeuralPriorCandidateManifest,
    LegacyNeuralPriorHoldoutPlanAudit,
    LegacyNeuralPriorHoldoutPlanCase,
    LegacyNeuralPriorHoldoutPlanV2Audit,
    LegacyNeuralPriorHoldoutPlanV2Case,
    LegacyNeuralPriorHoldoutPlanV3Audit,
    LegacyNeuralPriorHoldoutPlanV4Audit,
    NeuralPriorHoldoutCase,
    NeuralPriorHoldoutPlan,
    NeuralPriorHoldoutPlanCase,
    NeuralPriorInputPlan,
    PriorUncertaintyTargetPlan,
    NeuralPriorHoldoutPlanPolicy,
    NeuralPriorPromotionEvidence,
    LegacyNeuralPriorPromotionEvidenceAuditV3,
    LegacyNeuralPriorPromotionEvidenceAuditV4,
    LegacyNeuralPriorPromotionEvidenceAuditV5,
    NeuralPriorPromotionPolicy,
    PriorHoldoutEvaluation,
    compute_neural_prior_promotion,
    validate_neural_prior_promotion,
    validate_neural_prior_candidate_manifest,
    validate_neural_prior_holdout_plan,
    _new_prior_holdout_evaluation,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_EXECUTOR_TRUST_STORE_CONTRACT = "advar-executor-trust-store-v2"
_OPERATOR_TRUST_STORE_CONTRACT = "advar-operator-trust-store-v1"
_EPISODE_FILES = {"manifest.json", "sensitivity_arrays.npz"}
_INDEX_SCHEMA_VERSION = 14
_EPISODE_SCHEMA_VERSION = 18
_MODEL_CONTRACT_SCHEMA_VERSION = 11
_MAXIMUM_ACTION_ARTIFACT_MEMBERS = 12
_MAXIMUM_ACTION_ARTIFACT_FILE_BYTES = 2 * 1024**3
_MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES = 8 * 1024**3
_MAXIMUM_ACTION_GENERATOR_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class _ExecutorTrustStore:
    keys: dict[str, Ed25519PublicKey]
    content_digest: str


@dataclass(frozen=True)
class _OperatorTrustStore:
    keys: dict[str, Ed25519PublicKey]
    roles: dict[str, frozenset[str]]
    content_digest: str


@dataclass(frozen=True)
class LegacyPromotionEvaluationAudit:
    """Read-only payload retained before typed Tensor audit evidence existed."""

    evaluation_digest: str
    payload_json: str
    contract: str = "legacy-promotion-evaluation-audit-v1"
    content_digest_verified: bool = field(init=False)
    statistical_reuse_permitted: bool = field(init=False, default=False)
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.evaluation_digest) is None:
            raise ValueError("invalid legacy promotion evaluation digest")
        payload = json.loads(self.payload_json)
        if not isinstance(payload, dict):
            raise ValueError("invalid legacy promotion evaluation payload")
        if payload.get("evaluation_digest") != self.evaluation_digest:
            raise ValueError("legacy promotion evaluation digest mismatch")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if canonical != self.payload_json:
            raise ValueError("legacy promotion evaluation payload is not canonical")
        normalized = dict(payload)
        normalized.pop("evaluation_digest")
        tensor_payloads = [
            value
            for value in normalized.values()
            if isinstance(value, dict) and value.get("kind") == "tensor"
        ]
        verified = bool(tensor_payloads)
        if verified:
            for name, value in tuple(normalized.items()):
                if not isinstance(value, dict) or value.get("kind") != "tensor":
                    continue
                tensor = _decode_audit_tensor(name, value)
                normalized[name] = tensor_digest(tensor)
            if _json_digest(normalized) != self.evaluation_digest:
                raise ValueError("legacy promotion evaluation digest mismatch")
        object.__setattr__(self, "content_digest_verified", verified)
        object.__setattr__(self, "statistical_reuse_permitted", False)
        object.__setattr__(
            self,
            "audit_digest",
            _json_digest(
                {
                    "contract": self.contract,
                    "evaluation_digest": self.evaluation_digest,
                    "payload": payload,
                    "content_digest_verified": verified,
                    "statistical_reuse_permitted": False,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyProspectiveInterventionDecisionAudit:
    """Pre-operator-signature decision retained without execution reuse."""

    decision_digest: str
    payload_json: str
    audit_digest: str = ""

    def __post_init__(self) -> None:
        payload = json.loads(self.payload_json)
        if not isinstance(payload, dict) or (
            payload.get("contract") not in {
                "prospective-intervention-decision-v2",
                "prospective-intervention-decision-v3",
                "prospective-intervention-decision-v4",
                "prospective-intervention-decision-v5",
            }
            or payload.get("decision_digest") != self.decision_digest
        ):
            raise ValueError("invalid legacy prospective decision")
        retained = dict(payload)
        retained.pop("decision_digest")
        if _json_digest(retained) != self.decision_digest:
            raise ValueError("legacy prospective decision digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            _json_digest(
                {
                    "contract": "legacy-prospective-decision-audit-v1",
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyRealizedInterventionReceiptAudit:
    """Earlier signed receipt retained without current approval semantics."""

    receipt_digest: str
    decision_digest: str
    payload_json: str
    audit_digest: str = ""

    def __post_init__(self) -> None:
        payload = json.loads(self.payload_json)
        if not isinstance(payload, dict) or (
            payload.get("contract") not in {
                "realized-intervention-receipt-v2",
                "realized-intervention-receipt-v3",
                "realized-intervention-receipt-v4",
                "realized-intervention-receipt-v5",
            }
            or payload.get("receipt_digest") != self.receipt_digest
            or payload.get("decision_digest") != self.decision_digest
        ):
            raise ValueError("invalid legacy realized receipt")
        retained = dict(payload)
        retained.pop("receipt_digest")
        if _json_digest(retained) != self.receipt_digest:
            raise ValueError("legacy realized receipt digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            _json_digest(
                {
                    "contract": "legacy-realized-receipt-audit-v1",
                    "payload": payload,
                }
            ),
        )


def _load_executor_trust_store(path: str | Path) -> _ExecutorTrustStore:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError("executor trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError("executor trust store must be root-owned and non-writable")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            document = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict) or set(document) != {
        "contract",
        "public_keys",
    }:
        raise ValueError("invalid executor trust store")
    if document["contract"] != _EXECUTOR_TRUST_STORE_CONTRACT:
        raise ValueError("unsupported executor trust store")
    raw = document["public_keys"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("executor trust store requires keys")
    keys: dict[str, Ed25519PublicKey] = {}
    for key_id, public_hex in raw.items():
        if not isinstance(key_id, str) or not key_id or not isinstance(public_hex, str):
            raise ValueError("invalid executor trust-store key")
        try:
            public = bytes.fromhex(public_hex)
            key = Ed25519PublicKey.from_public_bytes(public)
        except ValueError as error:
            raise ValueError("invalid executor trust-store public key") from error
        if len(public) != 32:
            raise ValueError("executor public keys must contain 32 bytes")
        keys[key_id] = key
    return _ExecutorTrustStore(keys=keys, content_digest=_json_digest(document))


def _load_operator_trust_store(path: str | Path) -> _OperatorTrustStore:
    """Load a root-owned public-key store used only for action review."""

    source = Path(path)
    if not source.is_absolute():
        raise ValueError("operator trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError("operator trust store must be root-owned and non-writable")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            document = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict) or set(document) != {"contract", "operators"}:
        raise ValueError("invalid operator trust store")
    if document["contract"] != _OPERATOR_TRUST_STORE_CONTRACT:
        raise ValueError("unsupported operator trust store")
    raw = document["operators"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("operator trust store requires operators")
    keys: dict[str, Ed25519PublicKey] = {}
    roles: dict[str, frozenset[str]] = {}
    for key_id, record in raw.items():
        if (
            not isinstance(key_id, str)
            or not key_id
            or not isinstance(record, dict)
            or set(record) != {"public_key", "roles"}
            or not isinstance(record["public_key"], str)
            or not isinstance(record["roles"], list)
            or not record["roles"]
            or any(
                not isinstance(role, str) or not role or role.strip() != role
                for role in record["roles"]
            )
        ):
            raise ValueError("invalid operator trust-store entry")
        try:
            public = bytes.fromhex(record["public_key"])
            key = Ed25519PublicKey.from_public_bytes(public)
        except ValueError as error:
            raise ValueError("invalid operator trust-store public key") from error
        if len(public) != 32:
            raise ValueError("operator public keys must contain 32 bytes")
        keys[key_id] = key
        roles[key_id] = frozenset(record["roles"])
    return _OperatorTrustStore(
        keys=keys,
        roles=roles,
        content_digest=_json_digest(document),
    )


def _torch_dtype(value: str) -> torch.dtype:
    mapping = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError("unsupported durable action dtype") from error


def _intervention_context_tensor(
    frames: Tensor,
    masks: Tensor,
    quality: Tensor,
    std: Tensor,
    background: Tensor | None,
    applicability: Tensor,
    *,
    minimum_dbz: float,
    maximum_dbz: float,
    missing_fill_dbz: float,
) -> Tensor:
    finite_mask = torch.isfinite(frames)
    canonical_frames = canonicalize_action_frames(
        frames,
        masks,
        minimum_dbz=minimum_dbz,
        maximum_dbz=maximum_dbz,
        missing_fill_dbz=missing_fill_dbz,
    )
    finite = finite_mask.to(canonical_frames)
    canonical_background = (
        torch.zeros_like(canonical_frames)
        if background is None
        else canonicalize_action_frames(
            background,
            torch.ones_like(background, dtype=torch.bool),
            minimum_dbz=minimum_dbz,
            maximum_dbz=maximum_dbz,
            missing_fill_dbz=missing_fill_dbz,
        )
    )
    background_present = torch.full_like(
        canonical_frames,
        float(background is not None),
    )
    return torch.stack(
        (
            canonical_frames,
            masks.to(canonical_frames),
            finite,
            quality,
            std,
            canonical_background,
            background_present,
            applicability.to(canonical_frames),
        )
    )


def _artifact_intervention_context(
    manifest: dict[str, object],
    tensors: dict[str, Tensor],
    background: Tensor | None,
    *,
    prefix: str = "before",
) -> InterventionInputContext:
    """Rebuild the immutable action context from its durable tensor members."""

    result = object.__new__(InterventionInputContext)
    values: tuple[tuple[str, object], ...] = (
        ("_frames_dbz", tensors[f"{prefix}_frames"]),
        ("_observation_masks", tensors[f"{prefix}_masks"]),
        ("_quality_weight", tensors[f"{prefix}_quality"]),
        ("_observation_std_dbz", tensors[f"{prefix}_std"]),
        ("_background_frames_dbz", background),
        ("_applicability_mask", tensors[f"{prefix}_applicability"]),
        ("radar_id", manifest[f"{prefix}_radar_id"]),
        ("input_bundle_digest", manifest[f"{prefix}_input_bundle_digest"]),
        ("input_plan_digest", manifest["input_plan_digest"]),
        (
            "input_plan_resolution_digest",
            manifest[f"{prefix}_input_plan_resolution_digest"],
        ),
        (
            "analysis_input_identity_digest",
            manifest.get(f"{prefix}_analysis_input_identity_digest"),
        ),
        ("context_schema_digest", manifest[f"{prefix}_context_schema_digest"]),
        (
            "applicability_region_digest",
            manifest[f"{prefix}_applicability_region_digest"],
        ),
        (
            "applicability_mask_digest",
            manifest[f"{prefix}_applicability_mask_digest"],
        ),
        (
            "canonicalization_contract_digest",
            manifest[f"{prefix}_canonicalization_contract_digest"],
        ),
        ("min_dbz", manifest["minimum_dbz"]),
        ("max_dbz", manifest["maximum_dbz"]),
        ("missing_fill_dbz", manifest["missing_fill_dbz"]),
        ("context_digest", manifest[f"{prefix}_context_digest"]),
    )
    for name, value in values:
        object.__setattr__(result, name, value)
    return result
_TRUST_COMPONENTS_V13 = {
    "linearity",
    "verification",
    "metric_support",
    "pair_consistency",
    "path_evidence",
    "observation_evidence",
}
_TRUST_COMPONENTS_V16 = {
    "linearity",
    "verification",
    "metric_support",
    "pair_consistency",
    "observation_verified_evidence",
}
_MODEL_CONTRACT_SCHEMA_BY_EPISODE_SCHEMA = {
    3: 3,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
}
_LEGACY_MODEL_CONTRACT_FIELDS_V1_V2 = {
    "model_commit",
    "residual_contract_version",
    "forecast_metric_version",
    "observation_contract_version",
    "forecast_integrator_version",
    "grid_geometry_version",
    "radar_qc_version",
}
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
_SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES = (
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
_SCHEMA_FIVE_CONTEXT_FEATURE_NAMES = (
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
_SCHEMA_SIX_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_FIVE_CONTEXT_FEATURE_NAMES,
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
_SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_SIX_CONTEXT_FEATURE_NAMES,
    "phase_correlation_psr_available",
    "log1p_minimum_phase_correlation_psr",
)
_SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES,
    "projected_velocity_available",
    "projected_velocity_x_mps",
    "projected_velocity_y_mps",
    "projected_speed_mps",
)
_SCHEMA_NINE_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES,
    "motion_disagreement_mps_available",
    "motion_disagreement_mps",
)
_SCHEMA_TEN_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_NINE_CONTEXT_FEATURE_NAMES,
    "area_weighted_echo_available",
    "log1p_linear_reflectivity_integral_km2",
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
    grid_time_contract_digest: str | None = None

    def __post_init__(self) -> None:
        required = asdict(self)
        grid_digest = required.pop("grid_time_contract_digest")
        if not all(required.values()):
            raise ValueError("all model contract fields must be non-empty")
        if grid_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", grid_digest
        ) is None:
            raise ValueError(
                "grid_time_contract_digest must be a SHA-256 digest"
            )

    @property
    def digest(self) -> str:
        return _model_contract_digest(
            self,
            schema_version=_MODEL_CONTRACT_SCHEMA_VERSION,
        )


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
        forecast_run_digest = self.snapshot.forecast_run_digest
        if (
            not isinstance(forecast_run_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", forecast_run_digest) is None
        ):
            raise ValueError("forecast_run_digest must be a SHA-256 digest")
        if (
            self.contract.nowcast_config_digest
            != self.snapshot.nowcast_config_digest
            or self.contract.sensitivity_config_digest
            != self.snapshot.sensitivity_config_digest
        ):
            raise ValueError(
                "model contract config digests must match the sensitivity snapshot"
            )
        if (
            self.contract.grid_time_contract_digest
            != self.snapshot.grid_time_contract_digest
        ):
            raise ValueError(
                "model contract grid digest must match the sensitivity snapshot"
            )
        if self.snapshot.verification_lineage_complete:
            if (
                self.snapshot.verification_grid_contract_digest
                != self.snapshot.grid_time_contract_digest
            ):
                raise ValueError(
                    "verification grid digest must match the forecast grid"
                )
            expected_times = _expected_verification_times(
                self.issue_time,
                self.snapshot.lead_minutes,
            )
            if self.snapshot.verification_valid_times != expected_times:
                raise ValueError(
                    "verification valid times must match issue time and leads"
                )

    @property
    def forecast_run_digest(self) -> str:
        return self.snapshot.forecast_run_digest


@dataclass(frozen=True)
class LoadedEpisode:
    """Verified data loaded without Python object deserialization."""

    manifest: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


class EpisodeLedger:
    """Immutable M0 storage backed by SQLite, JSON, and NPZ."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.episodes_dir = self.root / "episodes"
        self.interventions_dir = self.root / "interventions"
        self.index_path = self.root / "index.sqlite"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.interventions_dir.mkdir(parents=True, exist_ok=True)
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
            np.savez_compressed(
                arrays_path,
                **cast(dict[str, Any], arrays),
            )

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

    def append_variational_learning_approval(
        self,
        learning: VariationalLearningImpact,
    ) -> str:
        """Append the digest evidence for one eligible P1 learning result."""

        validate_variational_learning_impact(learning)
        evidence = learning.approval_evidence
        if not learning.eligibility.eligible or evidence is None:
            raise ValueError("only eligible learning impacts can be recorded")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO variational_learning_approvals (
                    learning_result_digest, approval_evidence_digest,
                    evidence_contract, policy_digest, trust_store_digest,
                    fsoi_digest, full_step_analysis_digest,
                    half_step_analysis_digest, full_step_forecast_digest,
                    half_step_forecast_digest,
                    first_order_validation_digest,
                    learning_impact_digest, approved_action_digest,
                    nominal_input_bundle_digest,
                    nominal_full_analysis_input_digest,
                    selection_mode, candidate_id,
                    candidate_rank, candidate_score,
                    candidate_perturbation_digest, ranking_digest,
                    ranking_policy_digest, ranking_objective,
                    whitener_operations_per_apply,
                    observed_whitener_apply_count,
                    observed_whitener_total_operations, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    learning.learning_result_digest,
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
                    evidence.approved_action_digest,
                    evidence.nominal_input_bundle_digest,
                    evidence.nominal_full_analysis_input_digest,
                    evidence.selection_mode,
                    evidence.candidate_id,
                    evidence.candidate_rank,
                    evidence.candidate_score,
                    evidence.candidate_perturbation_digest,
                    evidence.ranking_digest,
                    evidence.ranking_policy_digest,
                    evidence.ranking_objective,
                    evidence.whitener_operations_per_apply,
                    evidence.observed_whitener_apply_count,
                    evidence.observed_whitener_total_operations,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return learning.learning_result_digest

    def load_variational_learning_approval(
        self,
        learning_result_digest: str,
    ) -> LearningApprovalEvidence:
        """Load and verify one immutable P1 learning approval record."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM variational_learning_approvals
                WHERE learning_result_digest = ?
                """,
                (learning_result_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown learning approval: {learning_result_digest}")
        evidence = LearningApprovalEvidence(
            policy_digest=row["policy_digest"],
            trust_store_digest=row["trust_store_digest"],
            fsoi_digest=row["fsoi_digest"],
            full_step_analysis_digest=row["full_step_analysis_digest"],
            half_step_analysis_digest=row["half_step_analysis_digest"],
            full_step_forecast_digest=row["full_step_forecast_digest"],
            half_step_forecast_digest=row["half_step_forecast_digest"],
            first_order_validation_digest=(
                row["first_order_validation_digest"]
            ),
            learning_impact_digest=row["learning_impact_digest"],
            approved_action_digest=row["approved_action_digest"],
            nominal_input_bundle_digest=row["nominal_input_bundle_digest"],
            nominal_full_analysis_input_digest=(
                row["nominal_full_analysis_input_digest"]
            ),
            selection_mode=row["selection_mode"],
            candidate_id=row["candidate_id"],
            candidate_rank=row["candidate_rank"],
            candidate_score=row["candidate_score"],
            candidate_perturbation_digest=(
                row["candidate_perturbation_digest"]
            ),
            ranking_digest=row["ranking_digest"],
            ranking_policy_digest=row["ranking_policy_digest"],
            ranking_objective=row["ranking_objective"],
            whitener_operations_per_apply=(
                row["whitener_operations_per_apply"]
            ),
            observed_whitener_apply_count=(
                row["observed_whitener_apply_count"]
            ),
            observed_whitener_total_operations=(
                row["observed_whitener_total_operations"]
            ),
            contract=row["evidence_contract"],
        )
        if evidence.digest != row["approval_evidence_digest"]:
            raise ValueError("learning approval evidence digest mismatch")
        return evidence

    def append_realized_observation_intervention(
        self,
        intervention: RealizedObservationIntervention,
    ) -> str:
        """Append one realized action linked to approved learning evidence."""

        if not isinstance(intervention, RealizedObservationIntervention):
            raise TypeError("intervention must be realized evidence")
        with self._connect() as connection:
            approved = connection.execute(
                """
                SELECT approval_evidence_digest
                FROM variational_learning_approvals
                WHERE learning_result_digest = ?
                """,
                (intervention.learning_result_digest,),
            ).fetchone()
            if approved is None:
                raise ValueError("intervention learning result is not recorded")
            if approved[0] != intervention.learning_approval_evidence_digest:
                raise ValueError("intervention learning approval mismatch")
            connection.execute(
                """
                INSERT INTO realized_observation_interventions (
                    intervention_digest, intervention_id, intervention_type,
                    action_digest, applied_time,
                    actual_input_before_digest, actual_input_after_digest,
                    observed_outcome_digest, execution_policy_digest,
                    execution_trust_store_digest,
                    predicted_normalized_benefit,
                    resolved_normalized_benefit, learning_result_digest,
                    learning_approval_evidence_digest,
                    counterfactual_perturbation_digest,
                    linearization_digest, case_id, radar_id, issue_time,
                    input_bundle_before_digest, input_bundle_after_digest,
                    resolved_issuance_validation_digest,
                    evidence_contract, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intervention.intervention_digest,
                    intervention.intervention_id,
                    intervention.intervention_type,
                    intervention.action_digest,
                    intervention.applied_time,
                    intervention.actual_input_before_digest,
                    intervention.actual_input_after_digest,
                    intervention.outcome_resolution_contract_digest,
                    intervention.execution_policy_digest,
                    intervention.execution_trust_store_digest,
                    intervention.predicted_normalized_benefit,
                    intervention.resolved_normalized_benefit,
                    intervention.learning_result_digest,
                    intervention.learning_approval_evidence_digest,
                    intervention.counterfactual_perturbation_digest,
                    intervention.linearization_digest,
                    intervention.case_id,
                    intervention.radar_id,
                    intervention.issue_time,
                    intervention.input_bundle_before_digest,
                    intervention.input_bundle_after_digest,
                    intervention.resolved_issuance_validation_digest,
                    intervention.contract,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return intervention.intervention_digest

    def append_retrospective_counterfactual_replay(
        self,
        replay: RetrospectiveCounterfactualReplay,
    ) -> str:
        """Retain a replay for audit without making it promotion evidence."""

        validate_retrospective_counterfactual_replay(replay)
        payload = asdict(replay)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO retrospective_counterfactual_replays "
                "(replay_digest, replay_json, created_at) VALUES (?, ?, ?)",
                (
                    replay.replay_digest,
                    json.dumps(payload, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return replay.replay_digest

    def load_retrospective_counterfactual_replay(
        self,
        replay_digest: str,
    ) -> RetrospectiveCounterfactualReplay:
        """Load and revalidate one historical replay audit."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT replay_json FROM retrospective_counterfactual_replays "
                "WHERE replay_digest = ?",
                (replay_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown retrospective replay: {replay_digest}")
        values = json.loads(row[0])
        if not isinstance(values, dict):
            raise ValueError("invalid retrospective replay payload")
        values.pop("replay_digest", None)
        replay = RetrospectiveCounterfactualReplay(**values)
        if replay.replay_digest != replay_digest:
            raise ValueError("retrospective replay ledger digest mismatch")
        validate_retrospective_counterfactual_replay(replay)
        return replay

    def _write_intervention_action_artifact(
        self,
        decision: ProspectiveInterventionDecision,
        receipt: RealizedInterventionReceipt,
        action_policy: ReusableInterventionPolicyEvidence,
        generator: InterventionActionGenerator,
        before: InterventionInputContext,
        before_run: ForecastRunContract,
        after: InterventionInputContext,
        after_run: ForecastRunContract,
    ) -> None:
        target = self.interventions_dir / receipt.receipt_digest
        if target.exists():
            self._replay_intervention_action_artifact(
                decision, receipt, action_policy
            )
            return
        generator_bytes = generator.artifact_bytes
        if len(generator_bytes) > _MAXIMUM_ACTION_GENERATOR_BYTES:
            raise ValueError("action generator artifact exceeds its byte budget")
        expanded_bytes = expanded_tensor_bytes((
            before.frames_dbz,
            before.observation_masks,
            before.quality_weight,
            before.observation_std_dbz,
            before.applicability_mask,
            after.frames_dbz,
            after.observation_masks,
            after.quality_weight,
            after.observation_std_dbz,
            after.applicability_mask,
            before.background_frames_dbz,
            after.background_frames_dbz,
        ))
        if expanded_bytes > _MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES:
            raise ValueError("action transition artifact exceeds its expanded-byte budget")
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{receipt.receipt_digest}.",
                dir=self.interventions_dir,
            )
        )
        try:
            before_background = before.background_frames_dbz
            after_background = after.background_frames_dbz
            np.savez_compressed(
                temporary / "transition.npz",
                before_frames=before.frames_dbz.detach().cpu().numpy(),
                before_masks=before.observation_masks.detach().cpu().numpy(),
                before_quality=before.quality_weight.detach().cpu().numpy(),
                before_std=before.observation_std_dbz.detach().cpu().numpy(),
                before_background=(
                    np.empty((0,), dtype=np.float32)
                    if before_background is None
                    else before_background.detach().cpu().numpy()
                ),
                before_applicability=before.applicability_mask.detach().cpu().numpy(),
                after_frames=after.frames_dbz.detach().cpu().numpy(),
                after_masks=after.observation_masks.detach().cpu().numpy(),
                after_quality=after.quality_weight.detach().cpu().numpy(),
                after_std=after.observation_std_dbz.detach().cpu().numpy(),
                after_background=(
                    np.empty((0,), dtype=np.float32)
                    if after_background is None
                    else after_background.detach().cpu().numpy()
                ),
                after_applicability=after.applicability_mask.detach().cpu().numpy(),
            )
            (temporary / "generator.pt2").write_bytes(generator_bytes)
            manifest: dict[str, object] = {
                "contract": "durable-intervention-action-artifact-v4",
                "receipt_digest": receipt.receipt_digest,
                "action_artifact_digest": receipt.action_artifact_digest,
                "action_payload_digest": receipt.action_payload_digest,
                "generator_digest": generator.generator_digest,
                "generator_shape": list(generator._shape),
                "generator_dtype": str(generator._dtype),
                "intervention_type": generator.intervention_type,
                "action_reason": generator.action_reason,
                "action_policy_digest": action_policy.policy_digest,
                "action_policy_json": asdict(action_policy),
                "action_safety_diagnostics_digest": (
                    receipt.action_safety_diagnostics_digest
                ),
                "before_context_digest": before.context_digest,
                "after_context_digest": after.context_digest,
                "before_radar_id": before.radar_id,
                "after_radar_id": after.radar_id,
                "before_context_schema_digest": before.context_schema_digest,
                "after_context_schema_digest": after.context_schema_digest,
                "before_applicability_region_digest": (
                    before.applicability_region_digest
                ),
                "after_applicability_region_digest": (
                    after.applicability_region_digest
                ),
                "before_applicability_mask_digest": tensor_digest(
                    before.applicability_mask
                ),
                "after_applicability_mask_digest": tensor_digest(
                    after.applicability_mask
                ),
                "before_canonicalization_contract_digest": (
                    before.canonicalization_contract_digest
                ),
                "after_canonicalization_contract_digest": (
                    after.canonicalization_contract_digest
                ),
                "minimum_dbz": before.min_dbz,
                "maximum_dbz": before.max_dbz,
                "missing_fill_dbz": before.missing_fill_dbz,
                "cell_area_m2": before_run.grid_time_contract.cell_area_m2
                if before_run.grid_time_contract is not None
                else None,
                "analysis_config_json": before_run.analysis_config_json,
                "analysis_config_digest": before_run.analysis_config_digest,
                "before_input_bundle_digest": before.input_bundle_digest,
                "after_input_bundle_digest": after.input_bundle_digest,
                "before_fixed_input_context_digest": (
                    before_run.fixed_input_context_digest
                ),
                "after_fixed_input_context_digest": (
                    after_run.fixed_input_context_digest
                ),
                "before_full_analysis_input_digest": (
                    before_run.full_analysis_input_digest
                ),
                "after_full_analysis_input_digest": (
                    after_run.full_analysis_input_digest
                ),
                "before_analysis_input_identity_digest": (
                    before.analysis_input_identity_digest
                ),
                "after_analysis_input_identity_digest": (
                    after.analysis_input_identity_digest
                ),
                "input_plan_digest": before.input_plan_digest,
                "before_input_plan_resolution_digest": (
                    before.input_plan_resolution_digest
                ),
                "after_input_plan_resolution_digest": (
                    after.input_plan_resolution_digest
                ),
                "before_background_present": before_background is not None,
                "after_background_present": after_background is not None,
            }
            for prefix, run in (("before", before_run), ("after", after_run)):
                manifest[f"{prefix}_background_age_minutes"] = (
                    run.background_age_minutes
                )
                manifest[f"{prefix}_grid_time_contract_digest"] = (
                    run.grid_time_contract_digest
                )
                manifest[f"{prefix}_calibration_manifest_digest"] = (
                    run.operational_calibration_manifest_digest
                )
                manifest[f"{prefix}_calibration_approval_digest"] = (
                    run.operational_calibration_approval_digest
                )
                manifest[f"{prefix}_data_identity_digest"] = (
                    run.operational_data_identity_digest
                )
            (temporary / "manifest.json").write_text(
                _json_text(manifest),
                encoding="utf-8",
            )
            checksums = {
                name: _file_digest(temporary / name)
                for name in ("manifest.json", "generator.pt2", "transition.npz")
            }
            (temporary / "checksums.json").write_text(
                _json_text(checksums),
                encoding="utf-8",
            )
            if any(
                (temporary / name).stat().st_size
                > _MAXIMUM_ACTION_ARTIFACT_FILE_BYTES
                for name in (*checksums, "checksums.json")
            ):
                raise ValueError("action artifact member exceeds its file-byte budget")
            for name in (*checksums, "checksums.json"):
                _fsync_file(temporary / name)
            _fsync_directory(temporary)
            os.rename(temporary, target)
            _fsync_directory(self.interventions_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _replay_intervention_action_artifact(
        self,
        decision: ProspectiveInterventionDecision,
        receipt: RealizedInterventionReceipt,
        action_policy: ReusableInterventionPolicyEvidence,
    ) -> None:
        source = self.interventions_dir / receipt.receipt_digest
        if not source.is_dir() or source.is_symlink():
            raise ValueError("durable intervention artifact is missing")
        validate_artifact_directory(
            source,
            expected_members=frozenset(
                {
                    "checksums.json",
                    "manifest.json",
                    "generator.pt2",
                    "transition.npz",
                }
            ),
            maximum_members=_MAXIMUM_ACTION_ARTIFACT_MEMBERS,
            maximum_file_bytes=_MAXIMUM_ACTION_ARTIFACT_FILE_BYTES,
        )
        checksum_path = source / "checksums.json"
        checksums = json.loads(checksum_path.read_text("utf-8"))
        expected_members = {"manifest.json", "generator.pt2", "transition.npz"}
        if not isinstance(checksums, dict) or set(checksums) != expected_members:
            raise ValueError("durable intervention artifact members are invalid")
        if any(
            not (source / name).is_file()
            or (source / name).is_symlink()
            or (source / name).stat().st_size
            > _MAXIMUM_ACTION_ARTIFACT_FILE_BYTES
            or _file_digest(source / name) != digest
            for name, digest in checksums.items()
        ):
            raise ValueError("durable intervention artifact checksum mismatch")
        manifest = json.loads((source / "manifest.json").read_text("utf-8"))
        if not isinstance(manifest, dict) or (
            manifest.get("contract")
            not in {
                "durable-intervention-action-artifact-v3",
                "durable-intervention-action-artifact-v4",
            }
            or manifest.get("receipt_digest") != receipt.receipt_digest
            or manifest.get("action_artifact_digest")
            != receipt.action_artifact_digest
            or manifest.get("before_input_bundle_digest")
            != receipt.actual_input_before_bundle_digest
            or manifest.get("after_input_bundle_digest")
            != receipt.actual_input_bundle_digest
            or manifest.get("action_policy_digest") != action_policy.policy_digest
            or manifest.get("action_safety_diagnostics_digest")
            != receipt.action_safety_diagnostics_digest
            or manifest.get("before_context_digest")
            != decision.intervention_input_context_digest
            or manifest.get("before_fixed_input_context_digest")
            != decision.actual_input_before_fixed_context_digest
            or manifest.get("before_full_analysis_input_digest")
            != decision.actual_input_before_full_analysis_input_digest
            or manifest.get("before_applicability_mask_digest")
            != decision.applicability_mask_digest
        ):
            raise ValueError("durable intervention artifact manifest mismatch")
        retained_policy_json = manifest.get("action_policy_json")
        if not isinstance(retained_policy_json, dict):
            raise ValueError("durable intervention policy is invalid")
        policy_payload = dict(cast(dict[str, object], retained_policy_json))
        policy_payload.pop("policy_digest", None)
        for name in ("allowed_intervention_types", "validation_evidence_digests"):
            policy_payload[name] = tuple(cast(list[object], policy_payload[name]))
        retained_policy = ReusableInterventionPolicyEvidence(
            **cast(Any, policy_payload)
        )
        if retained_policy.policy_digest != action_policy.policy_digest:
            raise ValueError("durable intervention policy changed")
        transition_path = source / "transition.npz"
        expected_arrays = frozenset(
            {
                "before_frames",
                "before_masks",
                "before_quality",
                "before_std",
                "before_background",
                "before_applicability",
                "after_frames",
                "after_masks",
                "after_quality",
                "after_std",
                "after_background",
                "after_applicability",
            }
        )
        preflight_npz_archive(
            transition_path,
            expected_members=expected_arrays,
            maximum_members=_MAXIMUM_ACTION_ARTIFACT_MEMBERS,
            maximum_expanded_bytes=_MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES,
        )
        with np.load(transition_path, allow_pickle=False) as arrays:
            if set(arrays.files) != expected_arrays:
                raise ValueError("durable intervention tensor members are invalid")
            tensors = {
                name: torch.from_numpy(arrays[name].copy())
                for name in arrays.files
            }
        before_background = (
            tensors["before_background"]
            if bool(manifest["before_background_present"])
            else None
        )
        after_background = (
            tensors["after_background"]
            if bool(manifest["after_background_present"])
            else None
        )
        before_applicability_digest = tensor_digest(
            tensors["before_applicability"]
        )
        after_applicability_digest = tensor_digest(
            tensors["after_applicability"]
        )
        if (
            before_applicability_digest
            != manifest.get("before_applicability_mask_digest")
            or after_applicability_digest
            != manifest.get("after_applicability_mask_digest")
            or before_applicability_digest != after_applicability_digest
        ):
            raise ValueError("durable intervention applicability changed")
        for prefix, background in (
            ("before", before_background),
            ("after", after_background),
        ):
            mask_digest = tensor_digest(tensors[f"{prefix}_masks"])
            quality_digest = tensor_digest(tensors[f"{prefix}_quality"])
            std_digest = tensor_digest(tensors[f"{prefix}_std"])
            background_digest = (
                None if background is None else tensor_digest(background)
            )
            expected_bundle = _forecast_input_bundle_digest(
                tensors[f"{prefix}_frames"],
                tensors[f"{prefix}_masks"],
                background,
                cast(float | None, manifest[f"{prefix}_background_age_minutes"]),
                None,
                cast(
                    str | None,
                    manifest[f"{prefix}_calibration_manifest_digest"],
                ),
                cast(
                    str | None,
                    manifest[f"{prefix}_calibration_approval_digest"],
                ),
                cast(str | None, manifest[f"{prefix}_data_identity_digest"]),
                grid_time_contract_digest=cast(
                    str | None,
                    manifest[f"{prefix}_grid_time_contract_digest"],
                ),
            )
            expected_fixed = _json_digest(
                {
                    "contract": "forecast-fixed-input-context-v1",
                    "observation_masks_digest": mask_digest,
                    "observation_quality_weight_digest": quality_digest,
                    "observation_std_dbz_digest": std_digest,
                    "background_frames_digest": background_digest,
                    "background_age_minutes": manifest[
                        f"{prefix}_background_age_minutes"
                    ],
                    "grid_time_contract_digest": manifest[
                        f"{prefix}_grid_time_contract_digest"
                    ],
                    "operational_calibration_manifest_digest": manifest[
                        f"{prefix}_calibration_manifest_digest"
                    ],
                    "operational_calibration_approval_digest": manifest[
                        f"{prefix}_calibration_approval_digest"
                    ],
                    "operational_data_identity_digest": manifest[
                        f"{prefix}_data_identity_digest"
                    ],
                    "input_plan_digest": manifest["input_plan_digest"],
                }
            )
            expected_full = _forecast_full_analysis_input_digest(
                input_frames_digest=tensor_digest(tensors[f"{prefix}_frames"]),
                fixed_input_context_digest=expected_fixed,
            )
            if manifest["contract"] == "durable-intervention-action-artifact-v4":
                expected_resolution = _forecast_input_plan_resolution_digest(
                    input_plan_digest=cast(str, manifest["input_plan_digest"]),
                    full_analysis_input_digest=expected_full,
                )
            else:
                expected_resolution = _json_digest(
                    {
                        "contract": "forecast-input-plan-resolution-v1",
                        "input_plan_digest": manifest["input_plan_digest"],
                        "input_bundle_digest": expected_bundle,
                    }
                )
            if (
                expected_bundle != manifest[f"{prefix}_input_bundle_digest"]
                or expected_resolution
                != manifest[f"{prefix}_input_plan_resolution_digest"]
            ):
                raise ValueError("durable intervention input bundle changed")
            expected_context = _json_digest(
                {
                    "contract": (
                        "intervention-input-context-v4"
                        if manifest["contract"]
                        == "durable-intervention-action-artifact-v4"
                        else "intervention-input-context-v3"
                    ),
                    "input_bundle_digest": manifest[
                        f"{prefix}_input_bundle_digest"
                    ],
                    "input_plan_digest": manifest["input_plan_digest"],
                    "input_plan_resolution_digest": manifest[
                        f"{prefix}_input_plan_resolution_digest"
                    ],
                    **(
                        {
                            "analysis_input_identity_digest": manifest[
                                f"{prefix}_analysis_input_identity_digest"
                            ]
                        }
                        if manifest["contract"]
                        == "durable-intervention-action-artifact-v4"
                        else {}
                    ),
                    "frames_digest": tensor_digest(tensors[f"{prefix}_frames"]),
                    "observation_masks_digest": mask_digest,
                    "quality_weight_digest": quality_digest,
                    "observation_std_dbz_digest": std_digest,
                    "background_frames_digest": background_digest,
                    "radar_id": manifest[f"{prefix}_radar_id"],
                    "context_schema_digest": manifest[
                        f"{prefix}_context_schema_digest"
                    ],
                    "applicability_region_digest": manifest[
                        f"{prefix}_applicability_region_digest"
                    ],
                    "applicability_mask_digest": manifest[
                        f"{prefix}_applicability_mask_digest"
                    ],
                    "canonicalization_contract_digest": manifest[
                        f"{prefix}_canonicalization_contract_digest"
                    ],
                    "minimum_dbz": manifest["minimum_dbz"],
                    "maximum_dbz": manifest["maximum_dbz"],
                    "missing_fill_dbz": manifest["missing_fill_dbz"],
                }
            )
            if expected_context != manifest[f"{prefix}_context_digest"]:
                raise ValueError("durable intervention context digest mismatch")
            retained_fixed = getattr(
                receipt,
                f"fixed_input_context_{prefix}_digest",
            )
            if (
                expected_fixed != manifest[f"{prefix}_fixed_input_context_digest"]
                or expected_fixed != retained_fixed
            ):
                raise ValueError("durable fixed input context changed")
            retained_full = getattr(
                receipt,
                f"full_analysis_input_{prefix}_digest",
            )
            if (
                expected_full != manifest[f"{prefix}_full_analysis_input_digest"]
                or expected_full != retained_full
            ):
                raise ValueError("durable full analysis input changed")
            if manifest["contract"] == "durable-intervention-action-artifact-v4":
                expected_identity = _json_digest(
                    {
                        "frames_digest": tensor_digest(
                            tensors[f"{prefix}_frames"]
                        ),
                        "fixed_context_digest": expected_fixed,
                        "full_data_digest": expected_full,
                        "input_plan_digest": manifest["input_plan_digest"],
                        "plan_resolution_digest": expected_resolution,
                        "contract": "analysis-input-identity-v1",
                    }
                )
                if expected_identity != manifest[
                    f"{prefix}_analysis_input_identity_digest"
                ]:
                    raise ValueError("durable analysis input identity changed")
        context_tensor = _intervention_context_tensor(
            tensors["before_frames"],
            tensors["before_masks"],
            tensors["before_quality"],
            tensors["before_std"],
            before_background,
            tensors["before_applicability"],
            minimum_dbz=float(manifest["minimum_dbz"]),
            maximum_dbz=float(manifest["maximum_dbz"]),
            missing_fill_dbz=float(manifest["missing_fill_dbz"]),
        )
        generator = InterventionActionGenerator.from_artifact(
            artifact=(source / "generator.pt2").read_bytes(),
            shape=tuple(int(value) for value in manifest["generator_shape"]),
            dtype=_torch_dtype(str(manifest["generator_dtype"])),
            intervention_type=cast(Any, manifest["intervention_type"]),
            action_reason=cast(str | None, manifest["action_reason"]),
            generator_digest=str(manifest["generator_digest"]),
        )
        action = generator.replay(context_tensor)
        if generator.intervention_type != receipt.intervention_type:
            raise ValueError("durable action type disagrees with its receipt")
        if action.payload_digest != receipt.action_payload_digest:
            raise ValueError("durable action payload changed")
        analysis_config_json = manifest.get("analysis_config_json")
        analysis_config_digest = manifest.get("analysis_config_digest")
        if (analysis_config_json is None) != (analysis_config_digest is None):
            raise ValueError("durable action analysis config is incomplete")
        if analysis_config_json is not None:
            if not isinstance(analysis_config_json, str) or not isinstance(
                analysis_config_digest, str
            ):
                raise ValueError("durable action analysis config is invalid")
            try:
                analysis_config = json.loads(analysis_config_json)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "durable action analysis config is invalid"
                ) from error
            if (
                not isinstance(analysis_config, dict)
                or _json_digest(analysis_config) != analysis_config_digest
                or float(
                    analysis_config.get("observation_common_bias_std_dbz", 0.0)
                )
                > 0.0
            ):
                raise ValueError(
                    "durable prospective action used correlated observation error"
                )
        diagnostics = _compute_action_safety(
            action,
            _artifact_intervention_context(manifest, tensors, before_background),
            retained_policy,
            minimum_dbz=float(manifest["minimum_dbz"]),
            maximum_dbz=float(manifest["maximum_dbz"]),
            cell_area_m2=float(manifest["cell_area_m2"]),
        )
        if diagnostics.diagnostics_digest != receipt.action_safety_diagnostics_digest:
            raise ValueError("durable action safety diagnostics changed")
        validate_action_tensor_replay(
            action,
            before_frames=tensors["before_frames"],
            before_masks=tensors["before_masks"],
            before_quality_weight=tensors["before_quality"],
            after_frames=tensors["after_frames"],
            after_masks=tensors["after_masks"],
            after_quality_weight=tensors["after_quality"],
        )
        retained_tensor_digests = (
            (tensor_digest(tensors["before_frames"]), receipt.actual_input_before_frames_digest),
            (tensor_digest(tensors["after_frames"]), receipt.actual_input_after_frames_digest),
            (tensor_digest(tensors["before_masks"]), receipt.actual_input_before_masks_digest),
            (tensor_digest(tensors["after_masks"]), receipt.actual_input_after_masks_digest),
            (
                tensor_digest(tensors["before_quality"]),
                receipt.actual_quality_weight_before_digest,
            ),
            (
                tensor_digest(tensors["after_quality"]),
                receipt.actual_quality_weight_after_digest,
            ),
        )
        if any(actual != expected for actual, expected in retained_tensor_digests):
            raise ValueError("durable intervention tensors disagree with receipt")
        replay_digest = _json_digest(
            {
                "contract": "intervention-action-artifact-v1",
                "generator_digest": generator.generator_digest,
                "before_context_digest": manifest["before_context_digest"],
                "after_context_digest": manifest["after_context_digest"],
                "action_payload_digest": action.payload_digest,
                "before_frames_digest": tensor_digest(tensors["before_frames"]),
                "after_frames_digest": tensor_digest(tensors["after_frames"]),
                "before_masks_digest": tensor_digest(tensors["before_masks"]),
                "after_masks_digest": tensor_digest(tensors["after_masks"]),
                "before_quality_weight_digest": tensor_digest(
                    tensors["before_quality"]
                ),
                "after_quality_weight_digest": tensor_digest(
                    tensors["after_quality"]
                ),
            }
        )
        if replay_digest != receipt.action_artifact_digest:
            raise ValueError("durable action transition digest mismatch")

    def append_prospective_intervention_decision(
        self,
        decision: ProspectiveInterventionDecision,
        *,
        operator_approval: OperatorActionApproval,
        action_policy: ReusableInterventionPolicyEvidence,
        action_generator: InterventionActionGenerator,
        actual_input_before_context: InterventionInputContext,
        actual_input_before_run: ForecastRunContract,
        trust_store_path: str | Path,
        operator_trust_store_path: str | Path,
    ) -> str:
        """Commit a policy-generated current-input action before its deadline."""

        action_policy.validate_integrity()
        reproduced = ProspectiveInterventionDecision.from_policy(
            action_policy,
            action_generator=action_generator,
            decision_id=decision.decision_id,
            case_id=decision.case_id,
            radar_id=decision.radar_id,
            intervention_type=decision.intervention_type,
            actual_input_context=actual_input_before_context,
            actual_input_before_run=actual_input_before_run,
            input_plan_digest=decision.input_plan_digest,
            decision_basis_digest=decision.decision_basis_digest,
            decision_policy_digest=decision.decision_policy_digest,
            decision_trust_store_digest=decision.decision_trust_store_digest,
            decided_at=decision.decided_at,
            observation_valid_time=decision.observation_valid_time,
            input_available_time=decision.input_available_time,
            decision_deadline=decision.decision_deadline,
            publication_time=decision.publication_time,
        )
        if reproduced.decision_digest != decision.decision_digest:
            raise ValueError("prospective decision is not the policy output")
        trust = _load_learning_policy_trust_store(trust_store_path)
        if trust.content_digest != decision.decision_trust_store_digest:
            raise ValueError("prospective decision trust-store mismatch")
        if decision.decision_policy_digest not in trust.approved_policy_digests:
            raise ValueError("prospective decision policy is not approved")
        if action_policy.policy_digest not in trust.approved_policy_digests:
            raise ValueError("reusable intervention policy is not approved")
        if (
            decision.action_policy_digest != action_policy.policy_digest
            or decision.action_generator_digest
            != action_policy.action_generator_digest
            or decision.decision_policy_digest
            != action_policy.execution_policy_digest
            or decision.decision_basis_digest
            not in action_policy.validation_evidence_digests
        ):
            raise ValueError("prospective decision disagrees with its action policy")
        decided = datetime.fromisoformat(decision.decided_at.replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(
            decision.decision_deadline.replace("Z", "+00:00")
        )
        publication = datetime.fromisoformat(
            decision.publication_time.replace("Z", "+00:00")
        )
        expected = _json_digest(
            {
                key: value
                for key, value in decision.__dict__.items()
                if key != "decision_digest"
            }
        )
        if expected != decision.decision_digest:
            raise ValueError("prospective decision digest mismatch")
        operator_trust = _load_operator_trust_store(operator_trust_store_path)
        if operator_approval.operator_trust_store_digest != (
            operator_trust.content_digest
        ):
            raise ValueError("operator approval trust-store mismatch")
        public_key = operator_trust.keys.get(operator_approval.operator_key_id)
        approved_roles = operator_trust.roles.get(
            operator_approval.operator_key_id,
            frozenset(),
        )
        if public_key is None or operator_approval.operator_role not in approved_roles:
            raise ValueError("operator is not approved for the declared role")
        if (
            operator_approval.decision_digest != decision.decision_digest
            or operator_approval.action_digest != decision.action_digest
            or operator_approval.full_analysis_input_digest
            != decision.actual_input_before_full_analysis_input_digest
            or operator_approval.safety_diagnostics_digest
            != decision.action_safety_diagnostics_digest
        ):
            raise ValueError("operator approval is bound to another decision")
        expected_approval_digest = _json_digest(
            {
                key: value
                for key, value in operator_approval.__dict__.items()
                if key != "approval_digest"
            }
        )
        if operator_approval.approval_digest != expected_approval_digest:
            raise ValueError("operator action approval digest mismatch")
        verify_operator_action_approval_signature(operator_approval, public_key)
        reviewed = datetime.fromisoformat(
            operator_approval.reviewed_at.replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(
            operator_approval.expires_at.replace("Z", "+00:00")
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            if (
                decided > reviewed
                or reviewed > now
                or now >= expires
                or expires > deadline
                or deadline >= publication
            ):
                raise ValueError("prospective decision missed its decision deadline")
            retained = connection.execute(
                "SELECT policy_json FROM reusable_intervention_policies "
                "WHERE policy_digest = ?",
                (action_policy.policy_digest,),
            ).fetchone()
            policy_json = json.dumps(asdict(action_policy), sort_keys=True)
            if retained is not None and retained[0] != policy_json:
                raise ValueError("recorded reusable intervention policy changed")
            if retained is None:
                connection.execute(
                    "INSERT INTO reusable_intervention_policies "
                    "(policy_digest, policy_json, created_at) VALUES (?, ?, ?)",
                    (action_policy.policy_digest, policy_json, now.isoformat()),
                )
            connection.execute(
                "INSERT INTO prospective_intervention_decisions "
                "(decision_digest, decision_id, decision_json, "
                "operator_approval_digest, operator_approval_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_digest,
                    decision.decision_id,
                    json.dumps(asdict(decision), sort_keys=True),
                    operator_approval.approval_digest,
                    json.dumps(asdict(operator_approval), sort_keys=True),
                    now.isoformat(),
                ),
            )
            if datetime.now(timezone.utc) > deadline:
                raise ValueError("prospective decision crossed its deadline")
        return decision.decision_digest

    def append_realized_intervention_receipt(
        self,
        decision: ProspectiveInterventionDecision,
        receipt: RealizedInterventionReceipt,
        *,
        action_policy: ReusableInterventionPolicyEvidence,
        action_generator: InterventionActionGenerator,
        actual_input_before_context: InterventionInputContext,
        actual_input_before_run: ForecastRunContract,
        actual_input_after_context: InterventionInputContext,
        actual_input_after_run: ForecastRunContract,
        trust_store_path: str | Path,
        executor_trust_store_path: str | Path,
        operator_trust_store_path: str | Path,
    ) -> str:
        """Record the executor receipt while the issue is still prospective."""

        validate_prospective_intervention(decision, receipt)
        transition = validate_intervention_action_transition(
            decision,
            action_policy=action_policy,
            action_generator=action_generator,
            actual_input_before_context=actual_input_before_context,
            actual_input_before_run=actual_input_before_run,
            actual_input_after_context=actual_input_after_context,
            actual_input_after_run=actual_input_after_run,
        )
        if (
            receipt.actual_input_before_frames_digest
            != transition.before_frames_digest
            or receipt.actual_input_after_frames_digest
            != transition.after_frames_digest
            or receipt.actual_input_before_bundle_digest
            != actual_input_before_run.input_bundle_digest
            or receipt.actual_input_bundle_digest
            != actual_input_after_run.input_bundle_digest
            or receipt.full_analysis_input_before_digest
            != actual_input_before_run.full_analysis_input_digest
            or receipt.full_analysis_input_after_digest
            != actual_input_after_run.full_analysis_input_digest
        ):
            raise ValueError("realized receipt transition evidence disagrees")
        trust = _load_learning_policy_trust_store(trust_store_path)
        if trust.content_digest != decision.decision_trust_store_digest:
            raise ValueError("realized receipt trust-store mismatch")
        executor_trust = _load_executor_trust_store(executor_trust_store_path)
        if receipt.executor_trust_store_digest != executor_trust.content_digest:
            raise ValueError("realized receipt executor trust-store mismatch")
        public_key = executor_trust.keys.get(receipt.executor_key_id)
        if public_key is None:
            raise ValueError("realized receipt executor is not approved")
        verify_intervention_receipt_signature(receipt, public_key)
        operator_trust = _load_operator_trust_store(operator_trust_store_path)
        publication = datetime.fromisoformat(
            receipt.publication_time.replace("Z", "+00:00")
        )
        received = datetime.fromisoformat(receipt.receipt_time.replace("Z", "+00:00"))
        applied = datetime.fromisoformat(receipt.applied_time.replace("Z", "+00:00"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            recorded = connection.execute(
                "SELECT decision_json, operator_approval_digest, "
                "operator_approval_json, created_at "
                "FROM prospective_intervention_decisions "
                "WHERE decision_digest = ?",
                (decision.decision_digest,),
            ).fetchone()
            if recorded is None:
                raise ValueError("realized receipt decision is not recorded")
            if json.loads(recorded[0]) != asdict(decision):
                raise ValueError("recorded prospective decision changed")
            if not recorded[1] or not recorded[2]:
                raise ValueError("realized receipt requires operator approval")
            approval_values = json.loads(recorded[2])
            retained_approval_digest = approval_values.pop(
                "approval_digest",
                None,
            )
            approval = _new_operator_action_approval(**approval_values)
            if (
                approval.approval_digest != retained_approval_digest
                or approval.approval_digest != recorded[1]
                or approval.decision_digest != decision.decision_digest
                or approval.operator_trust_store_digest
                != operator_trust.content_digest
            ):
                raise ValueError("recorded operator approval changed")
            operator_key = operator_trust.keys.get(approval.operator_key_id)
            if operator_key is None or approval.operator_role not in (
                operator_trust.roles.get(approval.operator_key_id, frozenset())
            ):
                raise ValueError("recorded operator is not approved")
            verify_operator_action_approval_signature(approval, operator_key)
            decision_created = datetime.fromisoformat(recorded[3])
            approval_expiry = datetime.fromisoformat(
                approval.expires_at.replace("Z", "+00:00")
            )
            if not (
                decision_created
                <= applied
                < approval_expiry
                and applied <= received <= now < publication
            ):
                raise ValueError("realized receipt violates trusted clock order")
            previous_sequence = connection.execute(
                "SELECT MAX(executor_sequence_number) "
                "FROM realized_intervention_receipts WHERE executor_key_id = ?",
                (receipt.executor_key_id,),
            ).fetchone()[0]
            if previous_sequence is not None and (
                receipt.executor_sequence_number <= previous_sequence
            ):
                raise ValueError("executor receipt sequence must increase")
            self._write_intervention_action_artifact(
                decision,
                receipt,
                action_policy,
                action_generator,
                actual_input_before_context,
                actual_input_before_run,
                actual_input_after_context,
                actual_input_after_run,
            )
            connection.execute(
                "INSERT INTO realized_intervention_receipts "
                "(receipt_digest, decision_digest, executor_key_id, "
                "executor_sequence_number, receipt_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    receipt.receipt_digest,
                    receipt.decision_digest,
                    receipt.executor_key_id,
                    receipt.executor_sequence_number,
                    json.dumps(asdict(receipt), sort_keys=True),
                    now.isoformat(),
                ),
            )
            if datetime.now(timezone.utc) >= publication:
                raise ValueError("realized receipt crossed publication time")
        return receipt.receipt_digest

    def load_prospective_intervention(
        self,
        receipt_digest: str,
        *,
        executor_trust_store_path: str | Path,
        operator_trust_store_path: str | Path | None = None,
    ) -> (
        tuple[
            ProspectiveInterventionDecision,
            RealizedInterventionReceipt,
            OperatorActionApproval,
        ]
        | tuple[
            LegacyProspectiveInterventionDecisionAudit,
            LegacyRealizedInterventionReceiptAudit,
        ]
    ):
        """Load and verify one causally ordered action and receipt."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT r.receipt_json, d.decision_json, "
                "d.operator_approval_digest, d.operator_approval_json "
                "FROM realized_intervention_receipts r "
                "JOIN prospective_intervention_decisions d "
                "ON d.decision_digest = r.decision_digest "
                "WHERE r.receipt_digest = ?",
                (receipt_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown realized receipt: {receipt_digest}")
        decision_values = json.loads(row["decision_json"])
        receipt_values = json.loads(row["receipt_json"])
        legacy_contracts = {
            (
                "prospective-intervention-decision-v2",
                "realized-intervention-receipt-v2",
            ),
            (
                "prospective-intervention-decision-v3",
                "realized-intervention-receipt-v3",
            ),
            (
                "prospective-intervention-decision-v4",
                "realized-intervention-receipt-v4",
            ),
        }
        approval_missing = not row["operator_approval_digest"] or not row[
            "operator_approval_json"
        ]
        if approval_missing or (
            decision_values.get("contract"),
            receipt_values.get("contract"),
        ) in legacy_contracts:
            decision_digest = decision_values.get("decision_digest")
            retained_receipt_digest = receipt_values.get("receipt_digest")
            if not isinstance(decision_digest, str) or (
                retained_receipt_digest != receipt_digest
                or receipt_values.get("decision_digest") != decision_digest
            ):
                raise ValueError("legacy prospective storage linkage is invalid")
            decision_audit = LegacyProspectiveInterventionDecisionAudit(
                decision_digest=decision_digest,
                payload_json=json.dumps(
                    decision_values,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            receipt_audit = LegacyRealizedInterventionReceiptAudit(
                receipt_digest=receipt_digest,
                decision_digest=decision_digest,
                payload_json=json.dumps(
                    receipt_values,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            executor_trust = _load_executor_trust_store(
                executor_trust_store_path
            )
            if receipt_values.get("executor_trust_store_digest") != (
                executor_trust.content_digest
            ):
                raise ValueError("legacy receipt executor trust-store mismatch")
            key_id = receipt_values.get("executor_key_id")
            public_key = executor_trust.keys.get(str(key_id))
            signature = receipt_values.get("executor_signature")
            if public_key is None or not isinstance(signature, str):
                raise ValueError("legacy receipt executor is not approved")
            signed_values = dict(receipt_values)
            signed_values.pop("receipt_digest")
            signed_values["executor_signature"] = ""
            try:
                public_key.verify(
                    bytes.fromhex(signature),
                    _json_digest(signed_values).encode("ascii"),
                )
            except (InvalidSignature, ValueError) as error:
                raise ValueError("legacy receipt executor signature mismatch") from error
            with self._connect() as connection:
                sequences = connection.execute(
                    "SELECT executor_sequence_number FROM "
                    "realized_intervention_receipts WHERE executor_key_id = ? "
                    "ORDER BY created_at, receipt_digest",
                    (key_id,),
                ).fetchall()
            sequence_values = [int(item[0]) for item in sequences]
            if sequence_values != sorted(set(sequence_values)):
                raise ValueError("legacy executor receipt sequence is invalid")
            return decision_audit, receipt_audit
        retained_decision_digest = decision_values.pop("decision_digest", None)
        retained_receipt_digest = receipt_values.pop("receipt_digest", None)
        decision = _new_prospective_decision(**decision_values)
        receipt = _new_realized_intervention_receipt(**receipt_values)
        if (
            decision.decision_digest != retained_decision_digest
            or receipt.receipt_digest != retained_receipt_digest
        ):
            raise ValueError("prospective intervention storage digest mismatch")
        validate_prospective_intervention(decision, receipt)
        executor_trust = _load_executor_trust_store(executor_trust_store_path)
        if receipt.executor_trust_store_digest != executor_trust.content_digest:
            raise ValueError("realized receipt executor trust-store mismatch")
        public_key = executor_trust.keys.get(receipt.executor_key_id)
        if public_key is None:
            raise ValueError("realized receipt executor is not approved")
        verify_intervention_receipt_signature(receipt, public_key)
        if operator_trust_store_path is None:
            raise ValueError("current prospective action requires operator trust store")
        try:
            approval_values = json.loads(row["operator_approval_json"])
        except json.JSONDecodeError as error:
            raise ValueError("stored operator approval is invalid") from error
        if not isinstance(approval_values, dict):
            raise ValueError("stored operator approval is invalid")
        retained_approval_digest = approval_values.pop("approval_digest", None)
        approval = _new_operator_action_approval(**approval_values)
        if (
            approval.approval_digest != retained_approval_digest
            or approval.approval_digest != row["operator_approval_digest"]
            or approval.decision_digest != decision.decision_digest
            or approval.action_digest != decision.action_digest
            or approval.full_analysis_input_digest
            != decision.actual_input_before_full_analysis_input_digest
            or approval.safety_diagnostics_digest
            != decision.action_safety_diagnostics_digest
        ):
            raise ValueError("stored operator approval changed")
        decided_at = datetime.fromisoformat(
            decision.decided_at.replace("Z", "+00:00")
        )
        reviewed_at = datetime.fromisoformat(
            approval.reviewed_at.replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            approval.expires_at.replace("Z", "+00:00")
        )
        deadline = datetime.fromisoformat(
            decision.decision_deadline.replace("Z", "+00:00")
        )
        applied_at = datetime.fromisoformat(
            receipt.applied_time.replace("Z", "+00:00")
        )
        if not (
            decided_at <= reviewed_at <= applied_at < expires_at <= deadline
        ):
            raise ValueError("stored operator approval time order is invalid")
        operator_trust = _load_operator_trust_store(operator_trust_store_path)
        if approval.operator_trust_store_digest != operator_trust.content_digest:
            raise ValueError("operator approval trust-store mismatch")
        operator_key = operator_trust.keys.get(approval.operator_key_id)
        if operator_key is None or approval.operator_role not in (
            operator_trust.roles.get(approval.operator_key_id, frozenset())
        ):
            raise ValueError("stored operator is not approved")
        verify_operator_action_approval_signature(approval, operator_key)
        with self._connect() as connection:
            sequences = connection.execute(
                "SELECT executor_sequence_number FROM "
                "realized_intervention_receipts WHERE executor_key_id = ? "
                "ORDER BY created_at, receipt_digest",
                (receipt.executor_key_id,),
            ).fetchall()
        values = [int(row[0]) for row in sequences]
        if values != sorted(set(values)):
            raise ValueError("executor receipt sequence ledger is invalid")
        with self._connect() as connection:
            policy_row = connection.execute(
                "SELECT policy_json FROM reusable_intervention_policies "
                "WHERE policy_digest = ?",
                (decision.action_policy_digest,),
            ).fetchone()
        if policy_row is None or not isinstance(policy_row[0], str):
            raise ValueError("prospective intervention policy is not retained")
        policy_values = json.loads(policy_row[0])
        if not isinstance(policy_values, dict):
            raise ValueError("prospective intervention policy is invalid")
        retained_policy_digest = policy_values.pop("policy_digest", None)
        policy_values["allowed_intervention_types"] = tuple(
            policy_values["allowed_intervention_types"]
        )
        policy_values["validation_evidence_digests"] = tuple(
            policy_values["validation_evidence_digests"]
        )
        action_policy = ReusableInterventionPolicyEvidence(
            **cast(Any, policy_values)
        )
        if (
            action_policy.policy_digest != retained_policy_digest
            or action_policy.policy_digest != decision.action_policy_digest
        ):
            raise ValueError("prospective intervention policy digest mismatch")
        self._replay_intervention_action_artifact(
            decision, receipt, action_policy
        )
        return decision, receipt, approval

    def load_realized_observation_intervention(
        self,
        intervention_digest: str,
    ) -> RealizedObservationIntervention:
        """Load and verify one immutable realized intervention."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM realized_observation_interventions
                WHERE intervention_digest = ?
                """,
                (intervention_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown realized intervention: {intervention_digest}")
        intervention = RealizedObservationIntervention(
            intervention_id=row["intervention_id"],
            intervention_type=row["intervention_type"],
            action_digest=row["action_digest"],
            applied_time=row["applied_time"],
            actual_input_before_digest=row["actual_input_before_digest"],
            actual_input_after_digest=row["actual_input_after_digest"],
            outcome_resolution_contract_digest=row["observed_outcome_digest"],
            execution_policy_digest=row["execution_policy_digest"],
            execution_trust_store_digest=row["execution_trust_store_digest"],
            predicted_normalized_benefit=row["predicted_normalized_benefit"],
            resolved_normalized_benefit=row["resolved_normalized_benefit"],
            learning_result_digest=row["learning_result_digest"],
            learning_approval_evidence_digest=(
                row["learning_approval_evidence_digest"]
            ),
            counterfactual_perturbation_digest=(
                row["counterfactual_perturbation_digest"]
            ),
            linearization_digest=row["linearization_digest"],
            case_id=row["case_id"],
            radar_id=row["radar_id"],
            issue_time=row["issue_time"],
            input_bundle_before_digest=row["input_bundle_before_digest"],
            input_bundle_after_digest=row["input_bundle_after_digest"],
            resolved_issuance_validation_digest=(
                row["resolved_issuance_validation_digest"]
            ),
            contract=row["evidence_contract"],
        )
        if intervention.intervention_digest != intervention_digest:
            raise ValueError("realized intervention digest mismatch")
        return intervention

    def append_neural_prior_holdout_plan(
        self,
        plan: NeuralPriorHoldoutPlan,
        *,
        policy: NeuralPriorHoldoutPlanPolicy,
        policy_trust_store_path: str | Path,
    ) -> str:
        """Pre-register one approved plan before its first forecast issue."""

        trust = _load_learning_policy_trust_store(policy_trust_store_path)
        validate_neural_prior_holdout_plan(plan)
        if policy.digest not in trust.approved_policy_digests:
            raise ValueError("holdout plan policy is not approved")
        if plan.plan_digest not in policy.approved_plan_digests:
            raise ValueError("holdout plan is not approved")
        if len(plan.candidate_family_digests) > policy.maximum_candidate_family_size:
            raise ValueError("holdout candidate family exceeds policy")
        approved_metrics = set(policy.approved_metric_contract_digests)
        if any(item.metric_contract_digest not in approved_metrics for item in plan.cases):
            raise ValueError("holdout metric contract is not approved")
        registered = datetime.fromisoformat(plan.registered_at.replace("Z", "+00:00"))
        issue_times = tuple(
            datetime.fromisoformat(item.issue_time.replace("Z", "+00:00"))
            for item in plan.cases
        )
        started = None
        if plan.mode == "sealed_historical":
            assert plan.candidate_training_started_at is not None
            started = datetime.fromisoformat(
                plan.candidate_training_started_at.replace("Z", "+00:00")
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            if registered > now:
                raise ValueError("holdout plan registration cannot be in the future")
            if plan.mode == "prospective" and any(
                now >= issue for issue in issue_times
            ):
                raise ValueError("holdout plan must be recorded before forecast issue")
            if started is not None and now >= started:
                raise ValueError(
                    "historical holdout must be recorded before candidate training"
                )
            connection.execute(
                """
                INSERT INTO neural_prior_holdout_plans (
                    plan_digest, plan_id, plan_json, policy_digest,
                    trust_store_digest, registered_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.plan_digest,
                    plan.plan_id,
                    json.dumps(asdict(plan), sort_keys=True),
                    policy.digest,
                    trust.content_digest,
                    plan.registered_at,
                    now.isoformat(),
                ),
            )
            final = datetime.now(timezone.utc)
            if plan.mode == "prospective" and any(
                final >= issue for issue in issue_times
            ):
                raise ValueError("holdout plan crossed its forecast issue time")
            if started is not None and final >= started:
                raise ValueError("holdout plan crossed candidate training time")
        return plan.plan_digest

    def load_neural_prior_holdout_plan(
        self,
        plan_digest: str,
    ) -> (
        NeuralPriorHoldoutPlan
        | LegacyNeuralPriorHoldoutPlanAudit
        | LegacyNeuralPriorHoldoutPlanV2Audit
        | LegacyNeuralPriorHoldoutPlanV3Audit
        | LegacyNeuralPriorHoldoutPlanV4Audit
    ):
        """Load and verify one immutable pre-registered holdout plan."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT plan_json FROM neural_prior_holdout_plans "
                "WHERE plan_digest = ?",
                (plan_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown neural-prior holdout plan: {plan_digest}")
        value = json.loads(row[0])
        value.pop("plan_digest", None)
        value["candidate_family_digests"] = tuple(
            value["candidate_family_digests"]
        )
        if value.get("contract") == "neural-prior-holdout-plan-v1":
            value["cases"] = tuple(
                LegacyNeuralPriorHoldoutPlanCase(**item) for item in value["cases"]
            )
            plan = LegacyNeuralPriorHoldoutPlanAudit(**value)
            if plan.plan_digest != plan_digest:
                raise ValueError("legacy neural-prior holdout digest mismatch")
            return plan
        if value.get("contract") == "neural-prior-holdout-plan-v2":
            value["cases"] = tuple(
                LegacyNeuralPriorHoldoutPlanV2Case(**item)
                for item in value["cases"]
            )
            value["input_plans"] = tuple(
                NeuralPriorInputPlan(
                    **{
                        key: entry
                        for key, entry in item.items()
                        if key != "plan_digest"
                    }
                )
                for item in value["input_plans"]
            )
            plan_v2 = LegacyNeuralPriorHoldoutPlanV2Audit(**value)
            if plan_v2.plan_digest != plan_digest:
                raise ValueError("legacy v2 neural-prior holdout digest mismatch")
            return plan_v2
        if value.get("contract") == "neural-prior-holdout-plan-v3":
            value["cases"] = tuple(
                NeuralPriorHoldoutPlanCase(**item) for item in value["cases"]
            )
            value["input_plans"] = tuple(
                NeuralPriorInputPlan(
                    **{
                        key: entry
                        for key, entry in item.items()
                        if key != "plan_digest"
                    }
                )
                for item in value["input_plans"]
            )
            value["uncertainty_target_plans"] = tuple(
                {
                    key: entry
                    for key, entry in item.items()
                    if key != "plan_digest"
                }
                for item in value["uncertainty_target_plans"]
            )
            plan_v3 = LegacyNeuralPriorHoldoutPlanV3Audit(**value)
            if plan_v3.plan_digest != plan_digest:
                raise ValueError("legacy v3 neural-prior holdout digest mismatch")
            return plan_v3
        if value.get("contract") == "neural-prior-holdout-plan-v4":
            value["cases"] = tuple(
                NeuralPriorHoldoutPlanCase(**item) for item in value["cases"]
            )
            value["input_plans"] = tuple(
                NeuralPriorInputPlan(
                    **{
                        key: entry
                        for key, entry in item.items()
                        if key != "plan_digest"
                    }
                )
                for item in value["input_plans"]
            )
            value["uncertainty_target_plans"] = tuple(
                {
                    key: entry
                    for key, entry in item.items()
                    if key != "plan_digest"
                }
                for item in value["uncertainty_target_plans"]
            )
            plan_v4 = LegacyNeuralPriorHoldoutPlanV4Audit(**value)
            if plan_v4.plan_digest != plan_digest:
                raise ValueError("legacy v4 neural-prior holdout digest mismatch")
            return plan_v4
        value["cases"] = tuple(
            NeuralPriorHoldoutPlanCase(**item) for item in value["cases"]
        )
        value["input_plans"] = tuple(
            NeuralPriorInputPlan(
                **{key: entry for key, entry in item.items() if key != "plan_digest"}
            )
            for item in value["input_plans"]
        )
        value["uncertainty_target_plans"] = tuple(
            PriorUncertaintyTargetPlan(
                **{
                    key: entry
                    for key, entry in item.items()
                    if key not in {"plan_digest", "support_event_digest"}
                }
            )
            for item in value["uncertainty_target_plans"]
        )
        plan = NeuralPriorHoldoutPlan(**value)
        if plan.plan_digest != plan_digest:
            raise ValueError("neural-prior holdout plan digest mismatch")
        return plan

    def append_neural_prior_promotion(
        self,
        evidence: NeuralPriorPromotionEvidence,
        manifest: NeuralPriorCandidateManifest,
        plan: NeuralPriorHoldoutPlan,
        evaluations: tuple[PriorHoldoutEvaluation, ...],
        *,
        policy: NeuralPriorPromotionPolicy,
        policy_trust_store_path: str | Path,
    ) -> str:
        """Append one promotion over every preregistered holdout case."""

        recomputed = compute_neural_prior_promotion(
            manifest,
            plan,
            evaluations,
            policy=policy,
            policy_trust_store_path=policy_trust_store_path,
        )
        validate_neural_prior_holdout_plan(plan)
        validate_neural_prior_candidate_manifest(manifest)
        if (
            recomputed.promotion_evidence_digest
            != evidence.promotion_evidence_digest
        ):
            raise ValueError("neural-prior promotion evidence is not reproducible")
        validate_neural_prior_promotion(evidence)
        with self._connect() as connection:
            plan_row = connection.execute(
                "SELECT plan_json, created_at FROM neural_prior_holdout_plans "
                "WHERE plan_digest = ?",
                (plan.plan_digest,),
            ).fetchone()
            if plan_row is None or plan_row[0] != json.dumps(
                asdict(plan), sort_keys=True
            ):
                raise ValueError("promotion holdout plan is not pre-registered")
            plan_created = datetime.fromisoformat(plan_row[1])
            if plan.mode == "prospective" and any(
                plan_created
                >= datetime.fromisoformat(item.issue_time.replace("Z", "+00:00"))
                for item in plan.cases
            ):
                raise ValueError("holdout plan was registered after forecast issue")
            receipt_rows = tuple(
                connection.execute(
                    "SELECT receipt_digest, receipt_json, created_at "
                    "FROM realized_intervention_receipts"
                )
            )
            recorded = {row[0]: json.loads(row[1]) for row in receipt_rows}
            recorded_approvals = {
                row[0]
                for row in connection.execute(
                    "SELECT approval_evidence_digest "
                    "FROM variational_learning_approvals"
                )
            }
            if set(manifest.training_learning_approval_digests) - recorded_approvals:
                raise ValueError("candidate training approvals are not recorded")
            training_interventions = set(manifest.training_intervention_digests)
            if training_interventions - set(recorded):
                raise ValueError("candidate training interventions are not recorded")
            for digest in training_interventions:
                if recorded[digest]["actual_input_bundle_digest"] not in (
                    manifest.training_input_bundle_digests
                ):
                    raise ValueError("candidate training input lineage is inconsistent")
                training_decision = connection.execute(
                    "SELECT decision_json FROM prospective_intervention_decisions "
                    "WHERE decision_digest = ?",
                    (recorded[digest]["decision_digest"],),
                ).fetchone()
                if training_decision is None:
                    raise ValueError("candidate training decision is not recorded")
                decision_payload = json.loads(training_decision[0])
                if decision_payload["decision_basis_digest"] not in (
                    manifest.training_learning_approval_digests
                ):
                    raise ValueError("training receipt is not linked to its approval")
            connection.execute(
                """
                INSERT INTO neural_prior_promotions (
                    promotion_evidence_digest, candidate_prior_digest,
                    parent_prior_digest, candidate_manifest_digest,
                    candidate_manifest_json, holdout_plan_digest,
                    policy_digest, trust_store_digest,
                    evaluation_digests_json, evaluation_payloads_json,
                    intervention_digests_json,
                    realized_intervention_count, material_outcome_count,
                    distinct_case_count, distinct_storm_count,
                    distinct_day_count, distinct_radar_count,
                    distinct_regime_count, distinct_range_regime_count,
                    beneficial_fraction, beneficial_fraction_lower_bound,
                    harmful_fraction, harmful_fraction_upper_bound,
                    mean_normalized_improvement, mean_improvement_lower_bound,
                    maximum_normalized_degradation,
                    prior_gaussian_nll_increase_upper_bound,
                    prior_support_brier_increase_upper_bound,
                    prior_underdispersion_increase_upper_bound,
                    prior_echo_intensity_nll_increase_upper_bound,
                    prior_clear_sky_false_echo_increase_upper_bound,
                    prior_echo_component_status,
                    prior_clear_sky_component_status,
                    prior_echo_case_count,
                    prior_clear_sky_case_count,
                    prior_echo_cluster_count,
                    prior_clear_sky_cluster_count,
                    simultaneous_inference_test_count,
                    eligible,
                    rejection_reasons_json, evidence_contract, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.promotion_evidence_digest,
                    evidence.candidate_prior_digest,
                    evidence.parent_prior_digest,
                    evidence.candidate_manifest_digest,
                    json.dumps(asdict(manifest), sort_keys=True),
                    plan.plan_digest,
                    evidence.policy_digest,
                    evidence.trust_store_digest,
                    json.dumps(list(evidence.evaluation_digests)),
                    json.dumps(
                        [_evaluation_audit_payload(item) for item in evaluations],
                        sort_keys=True,
                    ),
                    json.dumps([]),
                    evidence.holdout_case_count,
                    evidence.material_case_count,
                    evidence.distinct_case_count,
                    evidence.distinct_storm_count,
                    evidence.distinct_day_count,
                    evidence.distinct_radar_count,
                    evidence.distinct_regime_count,
                    evidence.distinct_range_regime_count,
                    evidence.beneficial_fraction,
                    evidence.beneficial_fraction_lower_bound,
                    evidence.harmful_fraction,
                    evidence.harmful_fraction_upper_bound,
                    evidence.mean_normalized_improvement,
                    evidence.mean_improvement_lower_bound,
                    evidence.maximum_normalized_degradation,
                    0.0,
                    evidence.prior_support_brier_increase_upper_bound,
                    evidence.prior_underdispersion_increase_upper_bound,
                    evidence.prior_echo_intensity_nll_increase_upper_bound,
                    evidence.prior_clear_sky_false_echo_increase_upper_bound,
                    evidence.prior_echo_component_status,
                    evidence.prior_clear_sky_component_status,
                    evidence.prior_echo_case_count,
                    evidence.prior_clear_sky_case_count,
                    evidence.prior_echo_cluster_count,
                    evidence.prior_clear_sky_cluster_count,
                    evidence.simultaneous_inference_test_count,
                    int(evidence.eligible),
                    json.dumps(list(evidence.rejection_reasons)),
                    evidence.contract,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return evidence.promotion_evidence_digest

    def load_neural_prior_promotion(
        self,
        promotion_evidence_digest: str,
    ) -> (
        NeuralPriorPromotionEvidence
        | LegacyNeuralPriorPromotionEvidenceAuditV3
        | LegacyNeuralPriorPromotionEvidenceAuditV4
        | LegacyNeuralPriorPromotionEvidenceAuditV5
    ):
        """Load and validate one immutable prior-promotion decision."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM neural_prior_promotions
                WHERE promotion_evidence_digest = ?
                """,
                (promotion_evidence_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(
                f"unknown neural-prior promotion: {promotion_evidence_digest}"
            )
        common_payload: dict[str, object] = {
            "candidate_prior_digest": row["candidate_prior_digest"],
            "parent_prior_digest": row["parent_prior_digest"],
            "candidate_manifest_digest": row["candidate_manifest_digest"],
            "policy_digest": row["policy_digest"],
            "trust_store_digest": row["trust_store_digest"],
            "evaluation_digests": tuple(
                json.loads(row["evaluation_digests_json"])
            ),
            "holdout_case_count": row["realized_intervention_count"],
            "material_case_count": row["material_outcome_count"],
            "distinct_case_count": row["distinct_case_count"],
            "distinct_storm_count": row["distinct_storm_count"],
            "distinct_day_count": row["distinct_day_count"],
            "distinct_radar_count": row["distinct_radar_count"],
            "distinct_regime_count": row["distinct_regime_count"],
            "distinct_range_regime_count": row["distinct_range_regime_count"],
            "beneficial_fraction": row["beneficial_fraction"],
            "beneficial_fraction_lower_bound": (
                row["beneficial_fraction_lower_bound"]
            ),
            "harmful_fraction": row["harmful_fraction"],
            "harmful_fraction_upper_bound": row["harmful_fraction_upper_bound"],
            "mean_normalized_improvement": row["mean_normalized_improvement"],
            "mean_improvement_lower_bound": row["mean_improvement_lower_bound"],
            "maximum_normalized_degradation": (
                row["maximum_normalized_degradation"]
            ),
            "eligible": bool(row["eligible"]),
            "rejection_reasons": tuple(
                json.loads(row["rejection_reasons_json"])
            ),
            "contract": row["evidence_contract"],
        }
        contract = row["evidence_contract"]
        if contract == "neural-prior-promotion-evidence-v3":
            evidence: (
                NeuralPriorPromotionEvidence
                | LegacyNeuralPriorPromotionEvidenceAuditV3
                | LegacyNeuralPriorPromotionEvidenceAuditV4
                | LegacyNeuralPriorPromotionEvidenceAuditV5
            ) = LegacyNeuralPriorPromotionEvidenceAuditV3(
                promotion_evidence_digest=promotion_evidence_digest,
                payload_json=json.dumps(
                    common_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        else:
            v4_payload = {
                **common_payload,
                "prior_gaussian_nll_increase_upper_bound": (
                    row["prior_gaussian_nll_increase_upper_bound"]
                ),
                "prior_support_brier_increase_upper_bound": (
                    row["prior_support_brier_increase_upper_bound"]
                ),
                "prior_underdispersion_increase_upper_bound": (
                    row["prior_underdispersion_increase_upper_bound"]
                ),
            }
            if contract == "neural-prior-promotion-evidence-v4":
                evidence = LegacyNeuralPriorPromotionEvidenceAuditV4(
                    promotion_evidence_digest=promotion_evidence_digest,
                    payload_json=json.dumps(
                        v4_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            else:
                v5_payload = {
                    **common_payload,
                    "prior_echo_intensity_nll_increase_upper_bound": (
                        row["prior_echo_intensity_nll_increase_upper_bound"]
                    ),
                    "prior_support_brier_increase_upper_bound": (
                        row["prior_support_brier_increase_upper_bound"]
                    ),
                    "prior_clear_sky_false_echo_increase_upper_bound": (
                        row["prior_clear_sky_false_echo_increase_upper_bound"]
                    ),
                    "prior_underdispersion_increase_upper_bound": (
                        row["prior_underdispersion_increase_upper_bound"]
                    ),
                }
                if contract == "neural-prior-promotion-evidence-v5":
                    evidence = LegacyNeuralPriorPromotionEvidenceAuditV5(
                        promotion_evidence_digest=promotion_evidence_digest,
                        payload_json=json.dumps(
                            v5_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                else:
                    if contract != "neural-prior-promotion-evidence-v6":
                        raise ValueError(
                            "unsupported neural-prior promotion evidence"
                        )
                    v6_payload = {
                        **v5_payload,
                        "prior_echo_component_status": (
                            row["prior_echo_component_status"]
                        ),
                        "prior_clear_sky_component_status": (
                            row["prior_clear_sky_component_status"]
                        ),
                        "prior_echo_case_count": row["prior_echo_case_count"],
                        "prior_clear_sky_case_count": (
                            row["prior_clear_sky_case_count"]
                        ),
                        "prior_echo_cluster_count": (
                            row["prior_echo_cluster_count"]
                        ),
                        "prior_clear_sky_cluster_count": (
                            row["prior_clear_sky_cluster_count"]
                        ),
                        "simultaneous_inference_test_count": (
                            row["simultaneous_inference_test_count"]
                        ),
                    }
                    v6_payload["contract"] = contract
                    evidence = NeuralPriorPromotionEvidence(
                        **cast(Any, v6_payload)
                    )
                if evidence.promotion_evidence_digest != promotion_evidence_digest:
                    raise ValueError("neural-prior promotion digest mismatch")
        payloads = json.loads(row["evaluation_payloads_json"])
        evaluations = _decode_evaluation_audit_payloads(payloads)
        if tuple(item.evaluation_digest for item in evaluations) != (
            tuple(cast(tuple[str, ...], common_payload["evaluation_digests"]))
        ):
            raise ValueError("neural-prior promotion evaluation audit mismatch")
        _decode_candidate_manifest(
            row["candidate_manifest_json"],
            expected_digest=row["candidate_manifest_digest"],
        )
        return evidence

    def load_neural_prior_promotion_evaluations(
        self,
        promotion_evidence_digest: str,
    ) -> tuple[
        PriorHoldoutEvaluation | LegacyPromotionEvaluationAudit,
        ...,
    ]:
        """Load typed metrics or an explicit read-only legacy audit."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT evaluation_payloads_json FROM neural_prior_promotions "
                "WHERE promotion_evidence_digest = ?",
                (promotion_evidence_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(
                f"unknown neural-prior promotion: {promotion_evidence_digest}"
            )
        return _decode_evaluation_audit_payloads(json.loads(row[0]))

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
                CREATE TABLE IF NOT EXISTS variational_learning_approvals (
                    learning_result_digest TEXT PRIMARY KEY,
                    approval_evidence_digest TEXT NOT NULL,
                    evidence_contract TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    fsoi_digest TEXT NOT NULL,
                    full_step_analysis_digest TEXT NOT NULL,
                    half_step_analysis_digest TEXT NOT NULL,
                    full_step_forecast_digest TEXT NOT NULL,
                    half_step_forecast_digest TEXT NOT NULL,
                    first_order_validation_digest TEXT NOT NULL,
                    learning_impact_digest TEXT NOT NULL,
                    approved_action_digest TEXT,
                    nominal_input_bundle_digest TEXT,
                    nominal_full_analysis_input_digest TEXT,
                    selection_mode TEXT NOT NULL DEFAULT 'direct',
                    candidate_id TEXT,
                    candidate_rank INTEGER,
                    candidate_score REAL,
                    candidate_perturbation_digest TEXT,
                    ranking_digest TEXT,
                    ranking_policy_digest TEXT,
                    ranking_objective TEXT,
                    whitener_operations_per_apply INTEGER NOT NULL DEFAULT 0,
                    observed_whitener_apply_count INTEGER NOT NULL DEFAULT 0,
                    observed_whitener_total_operations INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS realized_observation_interventions (
                    intervention_digest TEXT PRIMARY KEY,
                    intervention_id TEXT NOT NULL UNIQUE,
                    intervention_type TEXT NOT NULL,
                    action_digest TEXT NOT NULL,
                    applied_time TEXT NOT NULL,
                    actual_input_before_digest TEXT NOT NULL,
                    actual_input_after_digest TEXT NOT NULL,
                    observed_outcome_digest TEXT NOT NULL,
                    execution_policy_digest TEXT NOT NULL,
                    execution_trust_store_digest TEXT NOT NULL,
                    predicted_normalized_benefit REAL NOT NULL,
                    resolved_normalized_benefit REAL NOT NULL,
                    learning_result_digest TEXT NOT NULL,
                    learning_approval_evidence_digest TEXT NOT NULL,
                    counterfactual_perturbation_digest TEXT NOT NULL,
                    linearization_digest TEXT NOT NULL,
                    case_id TEXT NOT NULL DEFAULT '',
                    radar_id TEXT NOT NULL DEFAULT '',
                    issue_time TEXT NOT NULL DEFAULT '',
                    input_bundle_before_digest TEXT NOT NULL DEFAULT '',
                    input_bundle_after_digest TEXT NOT NULL DEFAULT '',
                    resolved_issuance_validation_digest TEXT NOT NULL DEFAULT '',
                    evidence_contract TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (learning_result_digest)
                        REFERENCES variational_learning_approvals(
                            learning_result_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrospective_counterfactual_replays (
                    replay_digest TEXT PRIMARY KEY,
                    replay_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reusable_intervention_policies (
                    policy_digest TEXT PRIMARY KEY,
                    policy_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prospective_intervention_decisions (
                    decision_digest TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL UNIQUE,
                    decision_json TEXT NOT NULL,
                    operator_approval_digest TEXT NOT NULL DEFAULT '',
                    operator_approval_json TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS realized_intervention_receipts (
                    receipt_digest TEXT PRIMARY KEY,
                    decision_digest TEXT NOT NULL UNIQUE,
                    executor_key_id TEXT NOT NULL DEFAULT '',
                    executor_sequence_number INTEGER NOT NULL DEFAULT 0,
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (decision_digest)
                        REFERENCES prospective_intervention_decisions(
                            decision_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_holdout_plans (
                    plan_digest TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL UNIQUE,
                    plan_json TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_promotions (
                    promotion_evidence_digest TEXT PRIMARY KEY,
                    candidate_prior_digest TEXT NOT NULL UNIQUE,
                    parent_prior_digest TEXT NOT NULL,
                    candidate_manifest_digest TEXT NOT NULL,
                    candidate_manifest_json TEXT NOT NULL,
                    holdout_plan_digest TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    evaluation_digests_json TEXT NOT NULL,
                    evaluation_payloads_json TEXT NOT NULL,
                    intervention_digests_json TEXT NOT NULL,
                    realized_intervention_count INTEGER NOT NULL,
                    material_outcome_count INTEGER NOT NULL,
                    distinct_case_count INTEGER NOT NULL,
                    distinct_storm_count INTEGER NOT NULL,
                    distinct_day_count INTEGER NOT NULL,
                    distinct_radar_count INTEGER NOT NULL,
                    distinct_regime_count INTEGER NOT NULL,
                    distinct_range_regime_count INTEGER NOT NULL,
                    beneficial_fraction REAL NOT NULL,
                    beneficial_fraction_lower_bound REAL NOT NULL,
                    harmful_fraction REAL NOT NULL,
                    harmful_fraction_upper_bound REAL NOT NULL,
                    mean_normalized_improvement REAL NOT NULL,
                    mean_improvement_lower_bound REAL NOT NULL,
                    maximum_normalized_degradation REAL NOT NULL,
                    prior_gaussian_nll_increase_upper_bound REAL NOT NULL,
                    prior_support_brier_increase_upper_bound REAL NOT NULL,
                    prior_underdispersion_increase_upper_bound REAL NOT NULL,
                    prior_echo_intensity_nll_increase_upper_bound REAL NOT NULL,
                    prior_clear_sky_false_echo_increase_upper_bound REAL NOT NULL,
                    prior_echo_component_status TEXT NOT NULL,
                    prior_clear_sky_component_status TEXT NOT NULL,
                    prior_echo_case_count INTEGER NOT NULL,
                    prior_clear_sky_case_count INTEGER NOT NULL,
                    prior_echo_cluster_count INTEGER NOT NULL,
                    prior_clear_sky_cluster_count INTEGER NOT NULL,
                    simultaneous_inference_test_count INTEGER NOT NULL,
                    eligible INTEGER NOT NULL,
                    rejection_reasons_json TEXT NOT NULL,
                    evidence_contract TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            _ensure_variational_learning_approval_schema(connection)
            _ensure_realized_intervention_schema(connection)
            _ensure_prospective_receipt_schema(connection)
            _ensure_prospective_decision_schema(connection)
            _ensure_neural_prior_promotion_schema(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS episodes_contract_time
                ON episodes(contract_hash, issue_time)
                """
            )
            connection.execute(
                f"PRAGMA user_version = {_INDEX_SCHEMA_VERSION}"
            )
            for table in (
                "episodes",
                "episode_impacts",
                "variational_learning_approvals",
                "realized_observation_interventions",
                "retrospective_counterfactual_replays",
                "reusable_intervention_policies",
                "prospective_intervention_decisions",
                "realized_intervention_receipts",
                "neural_prior_holdout_plans",
                "neural_prior_promotions",
            ):
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


def _ensure_variational_learning_approval_schema(
    connection: sqlite3.Connection,
) -> None:
    """Add ranked-selection lineage columns to an existing append-only index."""

    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(variational_learning_approvals)"
        ).fetchall()
    }
    definitions = {
        "selection_mode": "TEXT NOT NULL DEFAULT 'direct'",
        "candidate_id": "TEXT",
        "candidate_rank": "INTEGER",
        "candidate_score": "REAL",
        "candidate_perturbation_digest": "TEXT",
        "ranking_digest": "TEXT",
        "ranking_policy_digest": "TEXT",
        "ranking_objective": "TEXT",
        "approved_action_digest": "TEXT",
        "nominal_input_bundle_digest": "TEXT",
        "nominal_full_analysis_input_digest": "TEXT",
        "whitener_operations_per_apply": "INTEGER NOT NULL DEFAULT 0",
        "observed_whitener_apply_count": "INTEGER NOT NULL DEFAULT 0",
        "observed_whitener_total_operations": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                f"ALTER TABLE variational_learning_approvals "
                f"ADD COLUMN {name} {definition}"
            )


def _ensure_realized_intervention_schema(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(realized_observation_interventions)"
        ).fetchall()
    }
    definitions = {
        "execution_policy_digest": f"TEXT NOT NULL DEFAULT '{'0' * 64}'",
        "execution_trust_store_digest": f"TEXT NOT NULL DEFAULT '{'0' * 64}'",
        "predicted_normalized_benefit": "REAL NOT NULL DEFAULT 0",
        "resolved_normalized_benefit": "REAL NOT NULL DEFAULT 0",
        "case_id": "TEXT NOT NULL DEFAULT ''",
        "radar_id": "TEXT NOT NULL DEFAULT ''",
        "issue_time": "TEXT NOT NULL DEFAULT ''",
        "input_bundle_before_digest": "TEXT NOT NULL DEFAULT ''",
        "input_bundle_after_digest": "TEXT NOT NULL DEFAULT ''",
        "resolved_issuance_validation_digest": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                "ALTER TABLE realized_observation_interventions "
                f"ADD COLUMN {name} {definition}"
            )


def _ensure_prospective_receipt_schema(connection: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(realized_intervention_receipts)"
        ).fetchall()
    }
    definitions = {
        "executor_key_id": "TEXT NOT NULL DEFAULT ''",
        "executor_sequence_number": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                "ALTER TABLE realized_intervention_receipts "
                f"ADD COLUMN {name} {definition}"
            )


def _ensure_prospective_decision_schema(connection: sqlite3.Connection) -> None:
    """Add signed-review columns without fabricating evidence for old rows."""

    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(prospective_intervention_decisions)"
        ).fetchall()
    }
    definitions = {
        "operator_approval_digest": "TEXT NOT NULL DEFAULT ''",
        "operator_approval_json": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                "ALTER TABLE prospective_intervention_decisions "
                f"ADD COLUMN {name} {definition}"
            )


def _ensure_neural_prior_promotion_schema(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(neural_prior_promotions)"
        ).fetchall()
    }
    definitions = {
        "candidate_manifest_digest": f"TEXT NOT NULL DEFAULT '{'0' * 64}'",
        "candidate_manifest_json": "TEXT NOT NULL DEFAULT '{}'",
        "holdout_plan_digest": f"TEXT NOT NULL DEFAULT '{'0' * 64}'",
        "distinct_case_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_storm_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_day_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_radar_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_regime_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_range_regime_count": "INTEGER NOT NULL DEFAULT 0",
        "beneficial_fraction_lower_bound": "REAL NOT NULL DEFAULT 0",
        "harmful_fraction_upper_bound": "REAL NOT NULL DEFAULT 1",
        "mean_improvement_lower_bound": "REAL NOT NULL DEFAULT 0",
        "prior_gaussian_nll_increase_upper_bound": "REAL NOT NULL DEFAULT 0",
        "prior_support_brier_increase_upper_bound": "REAL NOT NULL DEFAULT 0",
        "prior_underdispersion_increase_upper_bound": "REAL NOT NULL DEFAULT 0",
        "prior_echo_intensity_nll_increase_upper_bound": (
            "REAL NOT NULL DEFAULT 0"
        ),
        "prior_clear_sky_false_echo_increase_upper_bound": (
            "REAL NOT NULL DEFAULT 0"
        ),
        "prior_echo_component_status": "TEXT NOT NULL DEFAULT ''",
        "prior_clear_sky_component_status": "TEXT NOT NULL DEFAULT ''",
        "prior_echo_case_count": "INTEGER NOT NULL DEFAULT 0",
        "prior_clear_sky_case_count": "INTEGER NOT NULL DEFAULT 0",
        "prior_echo_cluster_count": "INTEGER NOT NULL DEFAULT 0",
        "prior_clear_sky_cluster_count": "INTEGER NOT NULL DEFAULT 0",
        "simultaneous_inference_test_count": "INTEGER NOT NULL DEFAULT 0",
        "evaluation_payloads_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                "ALTER TABLE neural_prior_promotions "
                f"ADD COLUMN {name} {definition}"
            )


def _evaluation_audit_payload(
    evaluation: PriorHoldoutEvaluation,
) -> dict[str, object]:
    """Serialize the small paired-evaluation payload for later audit."""

    result: dict[str, object] = {}
    for name, value in evaluation.__dict__.items():
        if isinstance(value, Tensor):
            owned = value.detach().cpu()
            result[name] = {
                "kind": "tensor",
                "dtype": str(owned.dtype).removeprefix("torch."),
                "shape": list(owned.shape),
                "digest": tensor_digest(owned),
                "values": owned.tolist(),
            }
        elif isinstance(value, tuple):
            result[name] = list(value)
        elif hasattr(value, "value"):
            result[name] = value.value
        else:
            result[name] = value
    return result


def _decode_audit_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "dtype",
        "shape",
        "digest",
        "values",
    } or value["kind"] != "tensor":
        raise ValueError(f"invalid promotion audit tensor: {name}")
    dtype = getattr(torch, str(value["dtype"]), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"invalid promotion audit dtype: {name}")
    tensor = torch.tensor(value["values"], dtype=dtype)
    if list(tensor.shape) != value["shape"] or tensor_digest(tensor) != value["digest"]:
        raise ValueError(f"promotion audit tensor digest mismatch: {name}")
    return tensor


def _decode_evaluation_audit_payloads(
    value: object,
) -> tuple[PriorHoldoutEvaluation | LegacyPromotionEvaluationAudit, ...]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("invalid promotion evaluation audit payload")
    if value and (
        isinstance(value[0].get("metric_change"), list)
        or value[0].get("contract") != "prior-holdout-evaluation-v7"
    ):
        audits: list[LegacyPromotionEvaluationAudit] = []
        for raw in value:
            assert isinstance(raw, dict)
            digest = raw.get("evaluation_digest")
            if not isinstance(digest, str):
                raise ValueError("legacy promotion evaluation digest is missing")
            audits.append(
                LegacyPromotionEvaluationAudit(
                    evaluation_digest=digest,
                    payload_json=json.dumps(
                        raw,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
        return tuple(audits)
    evaluations: list[PriorHoldoutEvaluation] = []
    tensor_names = {
        "metric_change",
        "candidate_issuance_effect",
        "parent_issuance_effect",
        "end_to_end_metric_change",
        "metric_available",
        "coverage_candidate",
        "coverage_parent",
        "coverage_common",
        "newly_issued_fraction",
        "withdrawn_fraction",
    }
    tuple_names = {"lead_minutes", "metric_names", "verification_valid_times"}
    for raw in value:
        assert isinstance(raw, dict)
        values = dict(raw)
        stored_digest = values.pop("evaluation_digest", None)
        for name in tensor_names:
            values[name] = _decode_audit_tensor(name, values[name])
        for name in tuple_names:
            values[name] = tuple(values[name])
        values.pop("contract", None)
        evaluation = _new_prior_holdout_evaluation(**values)
        if evaluation.evaluation_digest != stored_digest:
            raise ValueError("promotion evaluation digest mismatch")
        evaluations.append(evaluation)
    return tuple(evaluations)


def _decode_candidate_manifest(
    text: str,
    *,
    expected_digest: str,
) -> (
    NeuralPriorCandidateManifest
    | LegacyNeuralPriorCandidateManifestAuditV2
    | LegacyNeuralPriorCandidateManifestAuditV3
):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid candidate manifest audit payload")
    values = dict(value)
    stored_digest = values.pop("manifest_digest", None)
    if values.get("contract") == "neural-prior-candidate-manifest-v2":
        audit = LegacyNeuralPriorCandidateManifestAuditV2(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit
    if values.get("contract") == "neural-prior-candidate-manifest-v3":
        audit_v3 = LegacyNeuralPriorCandidateManifestAuditV3(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v3.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v3
    values["holdout_cases"] = tuple(
        NeuralPriorHoldoutCase(**item) for item in values["holdout_cases"]
    )
    for name in (
        "training_learning_approval_digests",
        "training_intervention_digests",
        "training_case_ids",
        "training_input_bundle_digests",
        "training_storm_ids",
        "training_days",
        "training_radars",
        "training_regimes",
    ):
        values[name] = tuple(values[name])
    values["training_time_windows"] = tuple(
        tuple(item) for item in values["training_time_windows"]
    )
    manifest = NeuralPriorCandidateManifest(**values)
    if manifest.manifest_digest != stored_digest:
        raise ValueError("candidate manifest digest mismatch")
    if manifest.manifest_digest != expected_digest:
        raise ValueError("candidate manifest ledger digest mismatch")
    return manifest


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


def _snapshot_arrays(episode: SensitivityEpisode) -> dict[str, NDArray[Any]]:
    snapshot = episode.snapshot
    tensor_arrays = {
        "context_features": snapshot.context_features,
        "analysis_control": snapshot.analysis_control,
        "forecast_scores": snapshot.forecast_scores,
        "metric_available": snapshot.metric_available,
        "control_sensitivity": snapshot.control_sensitivity,
        "forecast_sensitivity": snapshot.forecast_sensitivity,
        "forecast_cap_active_mask": snapshot.forecast_cap_active_mask,
        "forecast_confidence": snapshot.forecast_confidence,
        "path_evidence_by_metric": snapshot.path_evidence_by_metric,
        "observation_source_fraction_by_metric": (
            snapshot.observation_source_fraction_by_metric
        ),
        "observation_verified_evidence_by_metric": (
            snapshot.observation_verified_evidence_by_metric
        ),
        "background_verified_evidence_by_metric": (
            snapshot.background_verified_evidence_by_metric
        ),
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
        "observation_std_dbz": snapshot.observation_std_dbz,
        "observation_innovation_dbz": snapshot.observation_innovation_dbz,
        "observation_innovation_mask": snapshot.observation_innovation_mask,
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
    arrays: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    snapshot = episode.snapshot
    return {
        "schema_version": _EPISODE_SCHEMA_VERSION,
        "episode_id": episode.episode_id,
        "issue_time": episode.issue_time,
        "radar_id": episode.radar_id,
        "forecast_run_digest": episode.forecast_run_digest,
        "verification_contract": snapshot.verification_contract,
        "verification_bundle_digest": snapshot.verification_bundle_digest,
        "verification_lineage_complete": (
            snapshot.verification_lineage_complete
        ),
        "verification_valid_times": (
            None
            if snapshot.verification_valid_times is None
            else list(snapshot.verification_valid_times)
        ),
        "verification_grid_contract_digest": (
            snapshot.verification_grid_contract_digest
        ),
        "verification_radar_product_digest": (
            snapshot.verification_radar_product_digest
        ),
        "verification_qc_pipeline_digest": (
            snapshot.verification_qc_pipeline_digest
        ),
        "contract": asdict(episode.contract),
        "contract_hash": episode.contract.digest,
        "metric_names": list(snapshot.metric_names),
        "lead_minutes": list(snapshot.lead_minutes),
        "full_map_lead_minutes": list(snapshot.full_map_lead_minutes),
        "tile_size": snapshot.tile_size,
        "tile_shape_yx": list(snapshot.tile_shape_yx),
        "context_feature_names": list(snapshot.context_feature_names),
        "sensitivity_scope": _sensitivity_scope(),
        "units": {
            "forecast_sensitivity": "d_error/d_linear_echo",
            "control_sensitivity": (
                "d_error/d_[dy_px,dx_px,log_growth_per_step]"
            ),
            "direct_observation_sensitivity": "d_error/d_dbz",
            "direct_observation_impact": "error_change",
            "forecast_confidence": "dimensionless_evidence_score",
            "path_evidence_by_metric": "dimensionless",
            "observation_source_fraction_by_metric": "dimensionless",
            "observation_verified_evidence_by_metric": "dimensionless",
            "background_verified_evidence_by_metric": "dimensionless",
        },
        "whitened_tile_norm_available": (
            snapshot.whitened_tile_norm_available
        ),
        "indirect_observation_sensitivity_available": False,
        "total_observation_sensitivity_available": False,
        "impact_available": snapshot.impact_available,
        "reward_available": snapshot.reward_available,
        "baseline_lineage_available": False,
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
    if (
        type(schema_version) is not int
        or not 1 <= schema_version <= _EPISODE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported episode schema")
    if manifest.get("episode_id") != episode_id:
        raise ValueError("manifest episode_id disagrees with the index")
    forecast_run_digest = manifest.get("forecast_run_digest")
    if schema_version >= 4:
        if (
            not isinstance(forecast_run_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", forecast_run_digest) is None
        ):
            raise ValueError("manifest forecast_run_digest is invalid")
    elif "forecast_run_digest" in manifest:
        raise ValueError(
            "forecast run provenance does not match the episode schema"
        )
    verification_keys = {
        "verification_contract",
        "verification_bundle_digest",
        "verification_lineage_complete",
        "verification_valid_times",
        "verification_grid_contract_digest",
        "verification_radar_product_digest",
        "verification_qc_pipeline_digest",
    }
    if schema_version >= 17:
        _validate_verification_manifest(manifest)
    elif verification_keys & set(manifest):
        raise ValueError(
            "verification provenance does not match the episode schema"
        )
    if schema_version >= 18:
        tile_shape = manifest.get("tile_shape_yx")
        if (
            not isinstance(tile_shape, list)
            or len(tile_shape) != 2
            or any(type(value) is not int or value <= 0 for value in tile_shape)
        ):
            raise ValueError("manifest physical tile shape is invalid")
    elif "tile_shape_yx" in manifest:
        raise ValueError("physical tile shape does not match episode schema")
    contract_values = manifest.get("contract")
    if not isinstance(contract_values, dict):
        raise ValueError("manifest contains an invalid contract")
    if schema_version < 3:
        if (
            set(contract_values) != _LEGACY_MODEL_CONTRACT_FIELDS_V1_V2
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
            schema_version=_MODEL_CONTRACT_SCHEMA_BY_EPISODE_SCHEMA.get(
                schema_version,
                _MODEL_CONTRACT_SCHEMA_VERSION,
            ),
        )
    if contract_hash != manifest.get("contract_hash"):
        raise ValueError("manifest contract hash is invalid")
    if contract_hash != row["contract_hash"]:
        raise ValueError("contract hash disagrees with the index")
    if schema_version >= 17 and manifest.get(
        "verification_lineage_complete"
    ):
        if manifest.get("verification_grid_contract_digest") != (
            contract_values.get("grid_time_contract_digest")
        ):
            raise ValueError(
                "manifest verification grid disagrees with the forecast grid"
            )
        expected_times = _expected_verification_times(
            manifest.get("issue_time"),
            manifest.get("lead_minutes"),
        )
        if tuple(manifest.get("verification_valid_times", ())) != (
            expected_times
        ):
            raise ValueError(
                "manifest verification times disagree with issue and leads"
            )
    if manifest.get("indirect_observation_sensitivity_available") is not False:
        raise ValueError("M0 episodes cannot contain indirect sensitivity")
    if manifest.get("total_observation_sensitivity_available") is not False:
        raise ValueError("M0 episodes cannot contain total sensitivity")
    if manifest.get("promotion_eligible") is not False:
        raise ValueError("M0 episodes cannot be promoted automatically")
    trust_components = manifest.get("trust_components")
    expected_trust_components = (
        _TRUST_COMPONENTS_V16
        if schema_version >= 16
        else _TRUST_COMPONENTS_V13
    )
    if schema_version >= 13 and (
        not isinstance(trust_components, dict)
        or set(trust_components) != expected_trust_components
    ):
        raise ValueError("manifest trust component contract is invalid")
    if schema_version >= 15 and (
        manifest.get("baseline_lineage_available") is not False
        or manifest.get("reward_available") is not False
    ):
        raise ValueError(
            "schema 15 and later require fail-closed baseline reward lineage"
        )
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


def _validate_verification_manifest(manifest: dict[str, Any]) -> None:
    digest = manifest.get("verification_bundle_digest")
    if not isinstance(digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", digest
    ) is None:
        raise ValueError("manifest verification bundle digest is invalid")
    complete = manifest.get("verification_lineage_complete")
    if type(complete) is not bool:
        raise ValueError("manifest verification lineage flag is invalid")
    valid_times = manifest.get("verification_valid_times")
    lineage_digests = tuple(
        manifest.get(name)
        for name in (
            "verification_grid_contract_digest",
            "verification_radar_product_digest",
            "verification_qc_pipeline_digest",
        )
    )
    if not complete:
        if manifest.get("verification_contract") != (
            "legacy-verification-tensor-v1"
        ):
            raise ValueError("manifest incomplete verification contract is invalid")
        if valid_times is not None or any(
            value is not None for value in lineage_digests
        ):
            raise ValueError("manifest incomplete verification claims lineage")
        return
    if manifest.get("verification_contract") != (
        "radar-verification-bundle-v1"
    ):
        raise ValueError("manifest complete verification contract is invalid")
    if (
        not isinstance(valid_times, list)
        or not valid_times
        or any(not isinstance(value, str) or not value for value in valid_times)
    ):
        raise ValueError("manifest verification valid times are invalid")
    parsed_times = tuple(
        _canonical_utc_datetime(value, "verification valid time")
        for value in valid_times
    )
    canonical_times = [
        value.isoformat().replace("+00:00", "Z") for value in parsed_times
    ]
    if valid_times != canonical_times or any(
        later <= earlier
        for earlier, later in zip(parsed_times, parsed_times[1:])
    ):
        raise ValueError(
            "manifest verification valid times must be canonical and increasing"
        )
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in lineage_digests
    ):
        raise ValueError("manifest verification lineage digest is invalid")


def _canonical_utc_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 string")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a timezone-aware ISO-8601 string"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _expected_verification_times(
    issue_time: object,
    lead_minutes: object,
) -> tuple[str, ...]:
    issue = _canonical_utc_datetime(issue_time, "issue_time")
    if (
        not isinstance(lead_minutes, (tuple, list))
        or not lead_minutes
        or any(type(value) is not int or value <= 0 for value in lead_minutes)
    ):
        raise ValueError("lead_minutes must contain positive integers")
    return tuple(
        (issue + timedelta(minutes=value))
        .isoformat()
        .replace("+00:00", "Z")
        for value in lead_minutes
    )


def _verify_manifest_layout(manifest: dict[str, Any]) -> None:
    arrays = manifest["arrays"]
    schema_version = manifest["schema_version"]
    legacy_core = {
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
    legacy_evidence_arrays = {
        "forecast_confidence",
        "path_evidence_by_metric",
        "observation_evidence_by_metric",
    }
    joint_evidence_arrays = {
        "forecast_confidence",
        "path_evidence_by_metric",
        "observation_source_fraction_by_metric",
        "observation_verified_evidence_by_metric",
        "background_verified_evidence_by_metric",
    }
    if schema_version >= 16:
        required_evidence_arrays = joint_evidence_arrays
    elif schema_version >= 14:
        required_evidence_arrays = legacy_evidence_arrays
    else:
        required_evidence_arrays = set()
    core = legacy_core | required_evidence_arrays
    legacy_optional = {
        "tile_whitened_direct_sensitivity_norm",
        "direct_observation_impact",
        "tile_direct_observation_impact",
        "direct_normalized_reward",
        "observation_std_dbz",
        "observation_innovation_dbz",
        "observation_innovation_mask",
        "baseline_scores",
    }
    optional = set(legacy_optional)
    if schema_version < 14:
        optional |= legacy_evidence_arrays | joint_evidence_arrays
    if schema_version >= 15:
        optional -= {"direct_normalized_reward", "baseline_scores"}
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
        if array_names not in (
            core | legacy_optional,
            core | legacy_optional | joint_evidence_arrays,
            core | optional,
        ):
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
    elif schema_version in (2, 3, 4):
        context_names = _SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                f"manifest arrays do not match episode schema {schema_version}"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 5:
        context_names = _SCHEMA_FIVE_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 5"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 6:
        context_names = _SCHEMA_SIX_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 6"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 7:
        context_names = _SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 7"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 8:
        context_names = _SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 8"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 9:
        context_names = _SCHEMA_NINE_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 9"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version in (10, 11):
        context_names = _SCHEMA_TEN_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 10"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version in (12, 13):
        context_names = CONTEXT_FEATURE_NAMES_V13
        if not core <= array_names <= core | optional:
            raise ValueError(
                f"manifest arrays do not match episode schema {schema_version}"
            )
        expected_scope = _sensitivity_scope()
    else:
        context_names = CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                f"manifest arrays do not match episode schema {schema_version}"
            )
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
        ("forecast_run_digest", snapshot.forecast_run_digest),
        ("nowcast_config_digest", snapshot.nowcast_config_digest),
        ("sensitivity_config_digest", snapshot.sensitivity_config_digest),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{name} must be a SHA-256 digest")
    grid_digest = snapshot.grid_time_contract_digest
    if grid_digest is not None and re.fullmatch(
        r"[0-9a-f]{64}", grid_digest
    ) is None:
        raise ValueError(
            "grid_time_contract_digest must be a SHA-256 digest"
        )
    verification_manifest = {
        "verification_contract": snapshot.verification_contract,
        "verification_bundle_digest": snapshot.verification_bundle_digest,
        "verification_lineage_complete": (
            snapshot.verification_lineage_complete
        ),
        "verification_valid_times": (
            None
            if snapshot.verification_valid_times is None
            else list(snapshot.verification_valid_times)
        ),
        "verification_grid_contract_digest": (
            snapshot.verification_grid_contract_digest
        ),
        "verification_radar_product_digest": (
            snapshot.verification_radar_product_digest
        ),
        "verification_qc_pipeline_digest": (
            snapshot.verification_qc_pipeline_digest
        ),
    }
    _validate_verification_manifest(verification_manifest)
    if not math.isfinite(snapshot.trust_score):
        raise ValueError("trust_score must be finite")
    if snapshot.baseline_scores is not None or snapshot.reward_available:
        raise ValueError(
            "normalized reward requires a verified baseline lineage contract"
        )
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
    if (
        len(snapshot.tile_shape_yx) != 2
        or any(type(value) is not int or value <= 0 for value in snapshot.tile_shape_yx)
        or snapshot.tile_size != max(snapshot.tile_shape_yx)
    ):
        raise ValueError("tile_shape_yx must contain two positive dimensions")

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
    tile_rows = math.ceil(height / snapshot.tile_shape_yx[0])
    tile_columns = math.ceil(width / snapshot.tile_shape_yx[1])

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
        "forecast_confidence": (lead_count, height, width),
        "path_evidence_by_metric": (lead_count, metric_count),
        "observation_source_fraction_by_metric": (
            lead_count,
            metric_count,
        ),
        "observation_verified_evidence_by_metric": (
            lead_count,
            metric_count,
        ),
        "background_verified_evidence_by_metric": (
            lead_count,
            metric_count,
        ),
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
        "forecast_confidence": snapshot.forecast_confidence,
        "path_evidence_by_metric": snapshot.path_evidence_by_metric,
        "observation_source_fraction_by_metric": (
            snapshot.observation_source_fraction_by_metric
        ),
        "observation_verified_evidence_by_metric": (
            snapshot.observation_verified_evidence_by_metric
        ),
        "background_verified_evidence_by_metric": (
            snapshot.background_verified_evidence_by_metric
        ),
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
    if not bool(
        torch.all(
            (snapshot.forecast_confidence >= 0)
            & (snapshot.forecast_confidence <= 1)
        )
    ):
        raise ValueError("forecast_confidence must be in [0, 1]")

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
    evidence_values = (
        snapshot.path_evidence_by_metric,
        snapshot.observation_source_fraction_by_metric,
        snapshot.observation_verified_evidence_by_metric,
        snapshot.background_verified_evidence_by_metric,
    )
    for value in evidence_values:
        finite = torch.isfinite(value)
        if not bool(torch.all((value[finite] >= 0) & (value[finite] <= 1))):
            raise ValueError("metric evidence must be in [0, 1]")
        if bool(torch.any(finite & ~available)):
            raise ValueError("unavailable metrics cannot retain evidence")
    joint_evidence_available = (
        torch.isfinite(snapshot.path_evidence_by_metric)
        & torch.isfinite(snapshot.observation_verified_evidence_by_metric)
        & torch.isfinite(snapshot.background_verified_evidence_by_metric)
    )
    if not torch.allclose(
        (
            snapshot.observation_verified_evidence_by_metric
            + snapshot.background_verified_evidence_by_metric
        )[joint_evidence_available],
        snapshot.path_evidence_by_metric[joint_evidence_available],
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("source evidence channels do not close")

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
        if whitened_latest is None:
            raise ValueError("whitened summaries require whitened tile norms")
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
        observation_std = snapshot.observation_std_dbz
        if observation_std is None:
            raise ValueError("whitened summaries require observation_std_dbz")
        if tuple(observation_std.shape) != (height, width):
            raise ValueError("observation_std_dbz must match the latest frame")
        _require_float_tensor("observation_std_dbz", observation_std)
        _require_finite("observation_std_dbz", observation_std)
        if bool(torch.any(observation_std <= 0)):
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

    innovation = snapshot.observation_innovation_dbz
    innovation_mask = snapshot.observation_innovation_mask
    if (innovation is None) != (innovation_mask is None):
        raise ValueError("innovation value and mask availability must agree")
    if innovation is not None and innovation_mask is not None:
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
        if latest_impact is None or latest_tile_impact is None:
            raise ValueError("direct impact requires impact and tile impact")
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
    if any(
        not math.isfinite(value)
        for value in snapshot.trust_components.values()
    ):
        raise ValueError("trust components must be finite")
    if set(snapshot.trust_components) != _TRUST_COMPONENTS_V16:
        raise ValueError("trust component contract is invalid")
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
    values = asdict(contract)
    if schema_version < 10:
        values.pop("grid_time_contract_digest")
    return _json_digest(
        {
            "contract_schema_version": schema_version,
            **values,
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
