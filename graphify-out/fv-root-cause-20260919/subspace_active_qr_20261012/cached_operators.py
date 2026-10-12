"""Pinned offline access to the PR302 active-subspace operator receipt."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
CACHE_WORK = ROOT / "graphify-out/fv-root-cause-20260919/subspace_rank_fix_20261012"
PLAN = CACHE_WORK / "EXPANSION_PLAN.json"
MANIFEST = CACHE_WORK / "RECEIPT_MANIFEST.json"
SOURCE_TAR = CACHE_WORK / "source.tar.gz"
RECEIPTS_TAR = CACHE_WORK / "receipts.tar.gz"
STEP = CACHE_WORK / "expansion1/step.json"
CLOSED_ENDPOINT_SNAPSHOT = WORK / "CLOSED_ENDPOINT_SNAPSHOT.json"
PLAN_SHA256 = "466c4ed8c8eb594e6c4f3293a204a9692dd145448eb0c85fc79c110e108f0c03"
MANIFEST_SHA256 = "57c812eefa96a3b0edecafc2d394c5ca1e99bd9d39492d4e658e46286d4ff4ca"
SOURCE_TAR_SHA256 = "ac3dd6ece38bd23425adbca6d5c436529cc4054e4b5e8fedb709ef4f7723bbed"
RECEIPTS_TAR_SHA256 = "0ddb163057c4a4d7650c009f77401e604fb7aab0dcf7c2d7e3186656f91cdca0"
STEP_SHA256 = "00b097714e7b6e76c207ebe87c2fa49a96ce822b8c6cb1889ae91c714ca73e92"
RUN_SHA256 = "07a526e86657c5edc95f2a865ebc40cc8ed38696200e31625b7aa1cf69b3353c"
RESOURCE_SHA256 = "352c91df25a2e60bb836fe8b59cd9f126f5efe2eb28efda22032509ca98a1b28"
BASE_SHA256 = "7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a"
BASE_RADIUS = 1.8757177916958825e-5
ENDPOINT_SNAPSHOT_SHA256 = "0044aee455f4eec8b6b082583c0849f9fa70e0f1ca8a85aa29b641db1410ec54"
NUMERICAL_SEARCH_SOURCES = {
    "examples/weather_scenarios/fv_point_3h_subspace_model.py",
    "examples/weather_scenarios/fv_point_3h_netflux_continuation.py",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_pinned(path: Path, expected: str) -> bytes:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"cache input must be an in-repository regular file: {path}")
    data = path.read_bytes()
    if sha256(data) != expected:
        raise ValueError(f"pinned cache input changed: {path.name}")
    return data


def _object(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _tar_files(data: bytes, label: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.name in files:
                    raise ValueError(f"{label} contains a non-file or duplicate member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{label} contains an unreadable member")
                files[member.name] = stream.read()
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"{label} is unreadable") from error
    return files


def admit() -> dict[str, Any]:
    """Verify the frozen plan, complete source/archive pins, and receipt bundles.

    This intentionally does not call the archived PR302 plan loader: its source
    transition governs a live experiment, while this function only admits bytes.
    """
    plan = _object(_read_pinned(PLAN, PLAN_SHA256), PLAN.name)
    manifest = _object(_read_pinned(MANIFEST, MANIFEST_SHA256), MANIFEST.name)
    source_bytes = _read_pinned(SOURCE_TAR, SOURCE_TAR_SHA256)
    receipt_bytes = _read_pinned(RECEIPTS_TAR, RECEIPTS_TAR_SHA256)
    source_files = plan.get("source_files")
    archive_files = plan.get("archive_files")
    receipt_files = manifest.get("files")
    if (not isinstance(source_files, dict) or len(source_files) != 222
            or not isinstance(archive_files, dict) or len(archive_files) != 291
            or not isinstance(receipt_files, dict) or len(receipt_files) != 5
            or manifest.get("source_bundle_sha256") != SOURCE_TAR_SHA256
            or manifest.get("bundle_sha256") != RECEIPTS_TAR_SHA256):
        raise ValueError("PR302 manifest counts or bundle identities differ from pinned values")

    bundled_sources = _tar_files(source_bytes, "PR302 source bundle")
    if set(bundled_sources) != set(source_files):
        raise ValueError("PR302 source bundle member set differs from the pinned plan")
    for name, digest in source_files.items():
        if sha256(bundled_sources[name]) != digest:
            raise ValueError(f"PR302 source bundle member hash mismatch: {name}")
    transition = plan.get("source_transition")
    if not isinstance(transition, dict):
        raise ValueError("PR302 plan has no explicit numerical source transition")
    current_source_drift: dict[str, str] = {}
    for name, digest in source_files.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"current PR302 physical/source input differs from its pin: {name}")
        current_digest = sha256(path.read_bytes())
        if current_digest != digest:
            if name not in transition and name not in NUMERICAL_SEARCH_SOURCES:
                raise ValueError(f"current PR302 physical/source input differs from its pin: {name}")
            current_source_drift[name] = current_digest
    for name, digest in archive_files.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()) != digest:
            raise ValueError(f"current PR302 archived input differs from its pin: {name}")

    bundled_receipts = _tar_files(receipt_bytes, "PR302 receipt bundle")
    if set(bundled_receipts) != set(receipt_files):
        raise ValueError("PR302 receipt bundle member set differs from its manifest")
    for name, entry in receipt_files.items():
        data = bundled_receipts[name]
        if sha256(data) != entry.get("sha256") or len(data) != entry.get("bytes"):
            raise ValueError(f"PR302 receipt member hash/length mismatch: {name}")
    step_bytes = _read_pinned(STEP, STEP_SHA256)
    if step_bytes != bundled_receipts["expansion1/step.json"]:
        raise ValueError("current expansion1 step differs from its pinned receipt bytes")
    step = _object(step_bytes, "expansion1/step.json")
    run = _object(bundled_receipts["expansion1/step.run.json"], "step.run.json")
    resource = _object(bundled_receipts["expansion1/step.resource.json"], "step.resource.json")
    operator = _object(bundled_receipts["expansion1/operators/GN.json"], "GN operator")
    if (sha256(bundled_receipts["expansion1/step.run.json"]) != RUN_SHA256
            or sha256(bundled_receipts["expansion1/step.resource.json"]) != RESOURCE_SHA256
            or run.get("execution_status") != "completed"
            or run.get("child_sha256") != STEP_SHA256
            or run.get("resource") != resource
            or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or step.get("execution_status") != "completed"
            or step.get("base_control_sha256") != BASE_SHA256
            or step.get("current_control_sha256") != BASE_SHA256
            or step.get("last_confirmed_control_sha256") != BASE_SHA256
            or step.get("candidate_evaluations_started") != 0
            or step.get("candidate_evaluations_completed") != 0
            or step.get("candidate_committed") is not False
            or step.get("hvp_started") != 4 or step.get("hvp_completed") != 4
            or step.get("source_unchanged") is not True
            or step.get("runtime_unchanged") is not True
            or step.get("fixed_input_unchanged") is not True
            or step.get("source_before") != step.get("source_after")
            or step.get("runtime_before") != step.get("runtime_after")
            or step.get("input_before") != step.get("input_after")
            or operator.get("jacobian_rows_started") != 24
            or operator.get("jacobian_rows_completed") != 24
            or operator.get("dense_solves_started") != 1
            or operator.get("dense_solves_completed") != 1):
        raise ValueError("PR302 expansion receipt does not satisfy its closed-budget lineage")

    # This closure was captured while the pinned PR301 endpoint driver still
    # admitted its old archives. Reuse those closed values instead of invoking
    # its live loader after the search orchestration has changed.
    endpoint_snapshot = _object(
        _read_pinned(CLOSED_ENDPOINT_SNAPSHOT, ENDPOINT_SNAPSHOT_SHA256),
        CLOSED_ENDPOINT_SNAPSHOT.name)
    snapshot_base = endpoint_snapshot.get("base", {})
    if (endpoint_snapshot.get("status") != "offline_cached_current_model_audit_complete"
            or endpoint_snapshot.get("admission", {}).get("plan_sha256") != PLAN_SHA256
            or endpoint_snapshot.get("admission", {}).get("manifest_sha256") != MANIFEST_SHA256
            or endpoint_snapshot.get("admission", {}).get("source_bundle_sha256") != SOURCE_TAR_SHA256
            or endpoint_snapshot.get("admission", {}).get("receipt_bundle_sha256") != RECEIPTS_TAR_SHA256
            or endpoint_snapshot.get("admission", {}).get("step_sha256") != STEP_SHA256
            or snapshot_base.get("control_sha256") != BASE_SHA256
            or snapshot_base.get("control") != step.get("current_control")
            or snapshot_base.get("full_G") != step.get("current_full_G")
            or snapshot_base.get("J") != step.get("base_J")
            or snapshot_base.get("R") != step.get("base_R")
            or snapshot_base.get("theta") != step.get("current_theta")
            or snapshot_base.get("context") != step.get("input_after")
            or snapshot_base.get("repeat", {}).get("control") != step.get("current_control")
            or snapshot_base.get("repeat", {}).get("objective") != step.get("base_J")
            or snapshot_base.get("repeat", {}).get("F_squared") != step.get("base_R")):
        raise ValueError("saved closed 7839 endpoint does not match the pinned PR302 current model")
    rows = operator.get("jacobian_row_history")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("PR302 cached Jacobian row history is incomplete")
    for i, row in enumerate(rows):
        if (row.get("status") != "completed" or row.get("base_control_sha256") != BASE_SHA256
                or len(row.get("gradient", [])) != 26 or row.get("row") != i % 12
                or row.get("side") != (-1 if i < 12 else 1)):
            raise ValueError("PR302 cached Jacobian rows are not a 24×26 control-space matrix")
    dense = operator.get("dense_solve_audit", {})
    if (len(dense.get("B", [])) != 12 or any(len(row) != 26 for row in dense["B"])
            or len(dense.get("S", [])) != 12
            or any(len(row) != 12 for row in dense["S"])):
        raise ValueError("PR302 archived GN matrix dimensions changed")
    return {
        "plan": plan,
        "manifest": manifest,
        "step": step,
        "operator": operator,
        "run": run,
        "resource": resource,
        "endpoint_snapshot": endpoint_snapshot,
        "bundled_receipts": bundled_receipts,
        "pins": {
            "plan_sha256": PLAN_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "source_bundle_sha256": SOURCE_TAR_SHA256,
            "receipt_bundle_sha256": RECEIPTS_TAR_SHA256,
            "step_sha256": STEP_SHA256,
            "closed_endpoint_snapshot_sha256": ENDPOINT_SNAPSHOT_SHA256,
            "source_member_count": len(source_files),
            "archive_pin_count": len(archive_files),
            "receipt_member_count": len(receipt_files),
            "approved_numerical_source_drift": current_source_drift,
        },
    }
