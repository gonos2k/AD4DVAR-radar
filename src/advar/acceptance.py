"""Report-only acceptance evidence for mock-free operational radar cases.

The harness intentionally never mutates a ledger or authorizes deployment.
It verifies a content-addressed set of artifacts emitted by the public product
path and produces a deterministic readiness report for external review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Literal, cast

from ._digest import json_digest


REAL_CASE_ACCEPTANCE_SCENARIOS = (
    "single_site_three_frame",
    "delayed_frame",
    "complete_radar_outage",
    "mosaic_source_handoff",
    "duplicate_acquisition_repackaging",
    "target_source_delayed_or_corrupt",
    "authority_revocation",
    "provenance_commit_crash",
    "deployment_activation_crash",
    "offline_clean_install_and_restart",
)

REAL_CASE_ACCEPTANCE_STAGES = (
    "native_radar_artifact",
    "trusted_raw_resolution",
    "analysis_input_provenance_activation",
    "forecast_run_contract",
    "variational_analysis",
    "candidate_parent_forecast",
    "production_verification_input",
    "verification_observation_error",
    "target_derivation",
    "scoring_replay",
    "promotion_evidence",
    "deployment_bundle_approval",
    "operational_decision_activation",
)


def _canonical_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("acceptance time is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("acceptance time must include UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if value != canonical:
        raise ValueError("acceptance time must be canonical UTC")
    return canonical


def _require_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _safe_relative_path(value: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "." in candidate.parts
        or str(candidate) != value
    ):
        raise ValueError("acceptance artifact path must be canonical and relative")
    return candidate


def _read_single_snapshot(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o022:
            raise ValueError("acceptance artifact must be regular and non-writable")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError("acceptance artifact size is invalid")
        data = bytearray()
        while len(data) < before.st_size:
            block = os.read(descriptor, min(1024 * 1024, before.st_size - len(data)))
            if not block:
                break
            data.extend(block)
        after = os.fstat(descriptor)
        if (
            len(data) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("acceptance artifact changed during validation")
        return bytes(data)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class AcceptanceArtifactReference:
    stage: str
    relative_path: str
    file_sha256: str
    product_contract: str
    product_artifact_digest: str

    def __post_init__(self) -> None:
        if self.stage not in REAL_CASE_ACCEPTANCE_STAGES:
            raise ValueError("acceptance artifact stage is unsupported")
        _safe_relative_path(self.relative_path)
        _require_digest("acceptance artifact file", self.file_sha256)
        _require_digest("product artifact", self.product_artifact_digest)
        if not self.product_contract or self.product_contract.strip() != self.product_contract:
            raise ValueError("product artifact contract must be canonical")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "relative_path": self.relative_path,
            "file_sha256": self.file_sha256,
            "product_contract": self.product_contract,
            "product_artifact_digest": self.product_artifact_digest,
        }


@dataclass(frozen=True)
class RealCaseAcceptanceCase:
    case_id: str
    scenario: str
    physical_event_digest: str
    sample_size_preflight_digest: str
    artifacts: tuple[AcceptanceArtifactReference, ...]

    def __post_init__(self) -> None:
        if not self.case_id or self.case_id.strip() != self.case_id:
            raise ValueError("acceptance case ID must be canonical")
        if self.scenario not in REAL_CASE_ACCEPTANCE_SCENARIOS:
            raise ValueError("acceptance scenario is unsupported")
        _require_digest("acceptance physical event", self.physical_event_digest)
        _require_digest("sample-size preflight", self.sample_size_preflight_digest)
        stages = tuple(item.stage for item in self.artifacts)
        if stages != REAL_CASE_ACCEPTANCE_STAGES:
            raise ValueError("acceptance case must contain the exact product stage chain")
        if len({item.relative_path for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("acceptance case artifact paths must be unique")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "scenario": self.scenario,
            "physical_event_digest": self.physical_event_digest,
            "sample_size_preflight_digest": self.sample_size_preflight_digest,
            "artifacts": [item.payload for item in self.artifacts],
        }


@dataclass(frozen=True)
class RealCaseAcceptanceManifest:
    created_at: str
    sample_size_preflight_relative_path: str
    sample_size_preflight_file_sha256: str
    sample_size_preflight_digest: str
    required_independent_physical_event_count: int
    cases: tuple[RealCaseAcceptanceCase, ...]
    mode: Literal["REPORT_ONLY"] = "REPORT_ONLY"
    contract: str = "advar-real-case-acceptance-manifest-v1"
    manifest_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "advar-real-case-acceptance-manifest-v1" or self.mode != "REPORT_ONLY":
            raise ValueError("real-case acceptance is report-only")
        object.__setattr__(self, "created_at", _canonical_utc(self.created_at))
        _safe_relative_path(self.sample_size_preflight_relative_path)
        _require_digest(
            "sample-size preflight file",
            self.sample_size_preflight_file_sha256,
        )
        _require_digest("sample-size preflight", self.sample_size_preflight_digest)
        if self.required_independent_physical_event_count <= 0 or not self.cases:
            raise ValueError("real-case acceptance sample-size contract is invalid")
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise ValueError("acceptance case IDs must be unique")
        if len({item.physical_event_digest for item in self.cases}) != len(self.cases):
            raise ValueError("acceptance cases must use independent physical events")
        if any(
            item.sample_size_preflight_digest != self.sample_size_preflight_digest
            for item in self.cases
        ):
            raise ValueError("acceptance cases disagree on sample-size preflight")
        if any(
            artifact.relative_path == self.sample_size_preflight_relative_path
            for item in self.cases
            for artifact in item.artifacts
        ):
            raise ValueError("sample-size preflight path must be independent")
        object.__setattr__(self, "manifest_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "mode": self.mode,
            "created_at": self.created_at,
            "sample_size_preflight_relative_path": (
                self.sample_size_preflight_relative_path
            ),
            "sample_size_preflight_file_sha256": (
                self.sample_size_preflight_file_sha256
            ),
            "sample_size_preflight_digest": self.sample_size_preflight_digest,
            "required_independent_physical_event_count": (
                self.required_independent_physical_event_count
            ),
            "cases": [item.payload for item in self.cases],
        }

    @property
    def json(self) -> str:
        return json.dumps(
            self.payload | {"manifest_digest": self.manifest_digest},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> RealCaseAcceptanceManifest:
        try:
            values = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("acceptance manifest JSON is invalid") from error
        if not isinstance(values, dict):
            raise ValueError("acceptance manifest JSON is invalid")
        retained = dict(values)
        stored_digest = retained.pop("manifest_digest", None)
        raw_cases = retained.pop("cases", None)
        if not isinstance(raw_cases, list):
            raise ValueError("acceptance manifest cases are invalid")
        cases = []
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise ValueError("acceptance manifest case is invalid")
            case_values = dict(raw_case)
            raw_artifacts = case_values.pop("artifacts", None)
            if not isinstance(raw_artifacts, list):
                raise ValueError("acceptance manifest artifacts are invalid")
            artifacts = tuple(
                AcceptanceArtifactReference(**cast(Any, artifact))
                for artifact in raw_artifacts
            )
            cases.append(
                RealCaseAcceptanceCase(
                    artifacts=artifacts,
                    **cast(Any, case_values),
                )
            )
        manifest = cls(cases=tuple(cases), **cast(Any, retained))
        if stored_digest != manifest.manifest_digest or text != manifest.json:
            raise ValueError("acceptance manifest digest mismatch")
        return manifest


def _artifact_path(root: Path, relative_path: str) -> Path:
    relative = _safe_relative_path(relative_path)
    retained = root
    root_metadata = retained.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_mode & 0o022:
        raise ValueError("acceptance artifact root must be non-writable")
    for part in relative.parts[:-1]:
        retained = retained / part
        metadata = retained.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_mode & 0o022
        ):
            raise ValueError("acceptance artifact ancestry is unsafe")
    return retained / relative.parts[-1]


def _load_sample_size_preflight(
    manifest: RealCaseAcceptanceManifest,
    *,
    root: Path,
    maximum_artifact_bytes: int,
) -> tuple[int, str]:
    from .promotion import PromotionSampleSizePreflight

    path = _artifact_path(root, manifest.sample_size_preflight_relative_path)
    data = _read_single_snapshot(path, maximum_bytes=maximum_artifact_bytes)
    file_digest = sha256(data).hexdigest()
    if file_digest != manifest.sample_size_preflight_file_sha256:
        raise ValueError("sample-size preflight file digest mismatch")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sample-size preflight is not typed JSON") from error
    if not isinstance(value, dict):
        raise ValueError("sample-size preflight is not typed JSON")
    retained = dict(value)
    stored_digest = retained.pop("preflight_digest", None)
    for field_name in (
        "metric_cell_event_counts",
        "issuance_cell_event_counts",
        "classifier_subset_event_counts",
    ):
        rows = retained.get(field_name)
        if not isinstance(rows, list) or any(
            not isinstance(row, list) for row in rows
        ):
            raise ValueError("sample-size preflight rows are invalid")
        retained[field_name] = tuple(tuple(row) for row in rows)
    try:
        preflight = PromotionSampleSizePreflight(**cast(Any, retained))
    except (TypeError, ValueError) as error:
        raise ValueError("sample-size preflight is invalid") from error
    canonical = json.dumps(
        preflight.payload | {"preflight_digest": preflight.preflight_digest},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if (
        stored_digest != preflight.preflight_digest
        or stored_digest != manifest.sample_size_preflight_digest
        or data != canonical
        or preflight.required_physical_events
        != manifest.required_independent_physical_event_count
    ):
        raise ValueError("sample-size preflight identity mismatch")
    return preflight.required_physical_events, file_digest


def verify_real_case_acceptance(
    manifest: RealCaseAcceptanceManifest,
    *,
    artifact_root: Path,
    maximum_artifact_bytes: int = 1024 * 1024 * 1024,
) -> dict[str, object]:
    """Verify exact artifact bytes and emit a non-authorizing readiness report."""

    if not artifact_root.is_absolute() or artifact_root.is_symlink():
        raise ValueError("acceptance artifact root must be absolute and unsymlinked")
    root = artifact_root.resolve(strict=True)
    required_event_count, preflight_file_digest = _load_sample_size_preflight(
        manifest,
        root=root,
        maximum_artifact_bytes=maximum_artifact_bytes,
    )
    verified_files: list[dict[str, object]] = []
    for case in manifest.cases:
        for reference in case.artifacts:
            path = _artifact_path(root, reference.relative_path)
            data = _read_single_snapshot(path, maximum_bytes=maximum_artifact_bytes)
            digest = sha256(data).hexdigest()
            if digest != reference.file_sha256:
                raise ValueError("acceptance artifact file digest mismatch")
            try:
                product_value = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("acceptance product artifact is not typed JSON") from error
            digest_fields = (
                "artifact_digest",
                "receipt_digest",
                "bundle_digest",
                "certificate_digest",
                "run_digest",
                "evidence_digest",
            )
            retained_product_digests = tuple(
                product_value.get(name)
                for name in digest_fields
                if isinstance(product_value, dict) and name in product_value
            )
            if (
                not isinstance(product_value, dict)
                or product_value.get("contract") != reference.product_contract
                or retained_product_digests
                != (reference.product_artifact_digest,)
            ):
                raise ValueError("acceptance product artifact identity mismatch")
            verified_files.append(
                {
                    "case_id": case.case_id,
                    "stage": reference.stage,
                    "relative_path": reference.relative_path,
                    "file_sha256": digest,
                    "product_contract": reference.product_contract,
                    "product_artifact_digest": reference.product_artifact_digest,
                }
            )
    scenarios = tuple(sorted({item.scenario for item in manifest.cases}))
    event_count = len({item.physical_event_digest for item in manifest.cases})
    complete_matrix = set(scenarios) == set(REAL_CASE_ACCEPTANCE_SCENARIOS)
    sample_size_satisfied = (
        event_count >= required_event_count
    )
    report_payload: dict[str, object] = {
        "contract": "advar-real-case-acceptance-report-v1",
        "mode": "REPORT_ONLY",
        "manifest_digest": manifest.manifest_digest,
        "sample_size_preflight_digest": manifest.sample_size_preflight_digest,
        "sample_size_preflight_file_sha256": preflight_file_digest,
        "verified_at": manifest.created_at,
        "verified_files": verified_files,
        "covered_scenarios": list(scenarios),
        "independent_physical_event_count": event_count,
        "required_independent_physical_event_count": (
            required_event_count
        ),
        "acceptance_matrix_complete": complete_matrix,
        "sample_size_satisfied": sample_size_satisfied,
        "eligible_for_live_review": complete_matrix and sample_size_satisfied,
        "authorizes_deployment": False,
    }
    return report_payload | {"report_digest": json_digest(report_payload)}
