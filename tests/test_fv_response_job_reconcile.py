"""Classification only for fixed response-job crash artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from threading import Event, Thread
from typing import Any
import uuid

import pytest

from examples.weather_scenarios import fv_response_job_lifecycle as jobs
from examples.weather_scenarios import fv_response_job_reconcile as reconcile
from examples.weather_scenarios import fv_process_response_worker as worker


FIXTURE = (jobs.EVIDENCE / "response_job_lifecycle_attempt1/complete")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _make_job(directory: Path, *, lifecycle: bool = False) -> None:
    directory.mkdir(parents=True)
    for name in ("published.json", "worker.raw.json", "worker.resource.json"):
        shutil.copyfile(FIXTURE / name, directory / name)
    raw_path = directory / "worker.raw.json"
    resource_path = directory / "worker.resource.json"
    published_path = directory / "published.json"
    child = _json(raw_path)
    child["source_before"] = worker._hashes()
    child["source_after"] = worker._hashes()
    _write(raw_path, child)
    resource = _json(resource_path)
    resource["command"] = [
        sys.executable,
        str((jobs.ROOT / "examples/weather_scenarios/fv_process_response_worker.py").resolve()),
        "--case", "fv4x5", "--output", str(raw_path.resolve()),
    ]
    _write(resource_path, resource)
    published = _json(published_path)
    published["worker_report_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    published["request_id"] = "request_1"
    _write(published_path, published)
    if lifecycle:
        shutil.copyfile(FIXTURE / "lifecycle.json", directory / "lifecycle.json")
        lifecycle_record = _json(directory / "lifecycle.json")
        lifecycle_record["request_id"] = "request_1"
        lifecycle_record["resource"] = resource
        lifecycle_record["source_before"] = jobs._sources()
        lifecycle_record["source_after"] = dict(lifecycle_record["source_before"])
        lifecycle_record["archive_before"] = jobs._archives()
        lifecycle_record["archive_after"] = dict(lifecycle_record["archive_before"])
        _write(directory / "lifecycle.json", lifecycle_record)


def _make_v2_job(directory: Path, *, lifecycle: bool = True) -> None:
    _make_job(directory, lifecycle=lifecycle)
    attempt_id = str(uuid.uuid4())
    raw_path = directory / "worker.raw.json"
    resource_path = directory / "worker.resource.json"
    manifest_path = directory / "attempt.json"
    raw = _json(raw_path)
    raw["attempt_id"] = attempt_id
    _write(raw_path, raw)

    command = [
        sys.executable,
        str((jobs.ROOT / "examples/weather_scenarios/fv_process_response_worker.py").resolve()),
        "--case", "fv4x5", "--attempt-id", attempt_id,
        "--output", str(raw_path.resolve()),
    ]
    identity, expected_total = jobs._identity()
    manifest = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "request_id": "request_1",
        "case_id": "fv4x5",
        "job_dir": str(directory.resolve()),
        "created_utc": "2026-09-27T00:00:00Z",
        "command": command,
        "source_sha256": jobs._sources(),
        "archive_sha256": jobs._archives(),
        "input_identity": identity,
        "expected_response_total": expected_total,
        "plan_sha256": jobs._sha(
            jobs.EVIDENCE / "FV_RESPONSE_JOB_ATTEMPT_BINDING_PLAN.md"),
        "resource_budget": {
            "wall_seconds": jobs.CHILD_WALL_SECONDS,
            "rss_limit_bytes": jobs.CHILD_RSS_BYTES,
            "sampling_seconds": 0.25,
            "scope": "sampled child RSS only; parent and between-sample spikes excluded",
        },
    }
    _write(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    resource = _json(resource_path)
    resource.update({
        "command": command,
        "attempt_id": attempt_id,
        "attempt_manifest_sha256": manifest_sha,
    })
    _write(resource_path, resource)

    published = _json(directory / "published.json")
    published.update({
        "publication_schema_version": 2,
        "attempt_id": attempt_id,
        "attempt_manifest_sha256": manifest_sha,
        "worker_report_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "resource_report_sha256": hashlib.sha256(resource_path.read_bytes()).hexdigest(),
        "job_dir": str(directory.resolve()),
        "source_sha256": jobs._sources(),
        "archive_sha256": jobs._archives(),
        "input_identity": identity,
    })
    _write(directory / "published.json", published)

    if lifecycle:
        lifecycle_record = _json(directory / "lifecycle.json")
        lifecycle_record["attempt_id"] = attempt_id
        lifecycle_record["resource"] = resource
        _write(directory / "lifecycle.json", lifecycle_record)


def _inspect(directory: Path):
    return reconcile.inspect_fixed_job_after_restart(directory, "request_1")


def test_missing_and_empty_job_report_no_observed_artifacts(tmp_path):
    missing = reconcile.inspect_fixed_job_after_restart(tmp_path / "missing", "request_1")
    empty_path = tmp_path / "empty"
    empty_path.mkdir()
    empty = reconcile.inspect_fixed_job_after_restart(empty_path, "request_1")
    assert (missing["state"], missing["reason_code"]) == ("no_job_artifacts", "job_directory_missing")
    assert (empty["state"], empty["reason_code"]) == ("no_job_artifacts", "empty_job_directory")


def test_raw_resource_crash_window_without_publication_needs_reconciliation(tmp_path):
    job_dir = tmp_path / "orphan"
    job_dir.mkdir()
    for name in ("worker.raw.json", "worker.resource.json"):
        shutil.copyfile(FIXTURE / name, job_dir / name)
    result = _inspect(job_dir)
    assert result["state"] == "needs_reconciliation"
    assert result["reason_code"] == "publication_missing"


def test_publication_without_lifecycle_needs_reconciliation(tmp_path):
    job_dir = tmp_path / "published"
    _make_job(job_dir)
    result = _inspect(job_dir)
    assert result["state"] == "needs_reconciliation"
    assert result["reason_code"] == "lifecycle_missing"


def test_consistent_terminal_lifecycle_allows_candidate_and_is_idempotent(tmp_path):
    job_dir = tmp_path / "published"
    _make_job(job_dir, lifecycle=True)
    before = {path.name: path.read_bytes() for path in job_dir.iterdir()}
    first = _inspect(job_dir)
    second = _inspect(job_dir)
    after = {path.name: path.read_bytes() for path in job_dir.iterdir()}
    assert first == second
    assert first["state"] == "publication_candidate"
    assert first["candidate_is_authoritative"] is False
    assert "mutable lifecycle self-report" in first["candidate_limit"]
    assert before == after


def test_fully_bound_v2_artifacts_allow_only_unauthenticated_candidate(tmp_path):
    job_dir = tmp_path / "v2"
    _make_v2_job(job_dir)
    result = _inspect(job_dir)
    assert result["state"] == "publication_candidate"
    assert result["candidate_is_authoritative"] is False


@pytest.mark.parametrize("marker", [
    "attempt.json", "attempt_id", "manifest_digest", "resource_digest", "schema2",
])
def test_any_v2_marker_disables_legacy_fallback(tmp_path, marker):
    job_dir = tmp_path / marker.replace(".", "-")
    _make_job(job_dir, lifecycle=True)
    if marker == "attempt.json":
        _write(job_dir / marker, {"schema_version": 1})
    else:
        published = _json(job_dir / "published.json")
        if marker == "attempt_id":
            published["attempt_id"] = str(uuid.uuid4())
        elif marker == "manifest_digest":
            published["attempt_manifest_sha256"] = "0" * 64
        elif marker == "resource_digest":
            published["resource_report_sha256"] = "0" * 64
        else:
            published["publication_schema_version"] = 2
        _write(job_dir / "published.json", published)
    assert _inspect(job_dir)["state"] == "needs_reconciliation"


def test_v2_marker_in_worker_log_disables_legacy_fallback(tmp_path):
    job_dir = tmp_path / "log-marker"
    _make_job(job_dir, lifecycle=True)
    (job_dir / "worker.log").write_text("worker argv included --attempt-id abc\n")
    assert _inspect(job_dir)["state"] == "needs_reconciliation"


@pytest.mark.parametrize("tamper", [
    "manifest", "raw_id", "resource_id", "resource_digest", "publication_digest",
    "lifecycle_id", "job_path", "source_drift", "input_drift", "command",
])
def test_v2_binding_tampering_and_missing_chain_fail_closed(tmp_path, tamper):
    job_dir = tmp_path / "tampered-v2"
    _make_v2_job(job_dir)
    if tamper == "manifest":
        (job_dir / "attempt.json").unlink()
    elif tamper in ("raw_id", "resource_id", "resource_digest"):
        path = job_dir / ("worker.raw.json" if tamper == "raw_id" else "worker.resource.json")
        record = _json(path)
        key = "attempt_id" if tamper != "resource_digest" else "attempt_manifest_sha256"
        record[key] = "0" * 64
        _write(path, record)
    elif tamper in ("publication_digest", "job_path", "source_drift", "input_drift"):
        path = job_dir / "published.json"
        record = _json(path)
        key = {
            "publication_digest": "resource_report_sha256",
            "job_path": "job_dir",
            "source_drift": "source_sha256",
            "input_drift": "input_identity",
        }[tamper]
        record[key] = "0" * 64 if tamper == "publication_digest" else {}
        _write(path, record)
    elif tamper == "lifecycle_id":
        record = _json(job_dir / "lifecycle.json")
        record["attempt_id"] = str(uuid.uuid4())
        _write(job_dir / "lifecycle.json", record)
    else:
        record = _json(job_dir / "attempt.json")
        if tamper == "command":
            record["command"][7] = str((tmp_path / "copied" / "worker.raw.json").resolve())
        else:
            record["source_sha256"]["examples/weather_scenarios/fv_response_job_lifecycle.py"] = "0" * 64
        _write(job_dir / "attempt.json", record)
    assert _inspect(job_dir)["state"] == "needs_reconciliation"


def test_v2_copied_job_path_is_unresolved(tmp_path):
    original = tmp_path / "original"
    _make_v2_job(original)
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    assert _inspect(copied)["state"] == "needs_reconciliation"


def test_v2_manifest_from_different_current_source_snapshot_is_unresolved(monkeypatch, tmp_path):
    job_dir = tmp_path / "source-drift"
    _make_v2_job(job_dir)
    current = jobs._sources()
    changed = dict(current)
    changed["examples/weather_scenarios/fv_response_job_lifecycle.py"] = "0" * 64
    monkeypatch.setattr(jobs, "_sources", lambda: changed)
    assert _inspect(job_dir)["state"] == "needs_reconciliation"


@pytest.mark.parametrize("artifact", ["worker.raw.json", "worker.resource.json"])
def test_artifact_swap_after_snapshot_never_yields_candidate(monkeypatch, tmp_path, artifact):
    job_dir = tmp_path / "snapshot-race"
    _make_v2_job(job_dir)
    stable_snapshot = jobs._stable_file_snapshot
    swapped = False

    def snapshot_then_swap(path: Path, *, sync: bool = False):
        nonlocal swapped
        result = stable_snapshot(path, sync=sync)
        if path.name == artifact and not swapped:
            swapped = True
            changed = _json(path)
            changed["snapshot_race"] = True
            _write(path, changed)
        return result

    monkeypatch.setattr(jobs, "_stable_file_snapshot", snapshot_then_swap)
    result = _inspect(job_dir)
    assert swapped
    assert result["state"] == "needs_reconciliation"


@pytest.mark.parametrize("name", ["worker.raw.json", "worker.resource.json"])
def test_missing_worker_record_is_not_recovered(tmp_path, name):
    job_dir = tmp_path / "missing-record"
    _make_job(job_dir)
    (job_dir / name).unlink()
    assert _inspect(job_dir)["state"] == "needs_reconciliation"


def test_partial_raw_and_cancelled_without_publication_need_reconciliation(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "worker.raw.json").write_text('{"status":"running"')
    assert _inspect(partial)["reason_code"] == "publication_missing"

    cancelled = tmp_path / "cancelled"
    cancelled.mkdir()
    shutil.copyfile(FIXTURE / "worker.raw.json", cancelled / "worker.raw.json")
    resource = _json(FIXTURE / "worker.resource.json")
    resource["resource_termination"] = "cancelled"
    _write(cancelled / "worker.resource.json", resource)
    assert _inspect(cancelled)["state"] == "needs_reconciliation"


@pytest.mark.parametrize("tamper", ["sha", "pid", "request", "case", "scope", "input",
                                    "response", "boolean_direct", "nan_direct",
                                    "time", "source", "archive"])
def test_contradictory_publication_or_worker_is_rejected(tmp_path, tamper):
    job_dir = tmp_path / "tampered"
    _make_job(job_dir)
    published_path = job_dir / "published.json"
    raw_path = job_dir / "worker.raw.json"
    if tamper == "sha":
        published = _json(published_path)
        published["worker_report_sha256"] = "0" * 64
        _write(published_path, published)
    elif tamper == "pid":
        published = _json(published_path)
        published["worker_pid"] += 1
        _write(published_path, published)
    elif tamper == "request":
        published = _json(published_path)
        published["request_id"] = "other"
        _write(published_path, published)
    elif tamper == "case":
        published = _json(published_path)
        published["case_id"] = "fv8x10"
        _write(published_path, published)
    elif tamper == "scope":
        published = _json(published_path)
        published["scope"] = "different response scope"
        _write(published_path, published)
    elif tamper == "input":
        published = _json(published_path)
        published["input_identity"]["problem"] = "0" * 64
        _write(published_path, published)
    elif tamper == "response":
        published = _json(published_path)
        published["response_total"] += 1.0
        _write(published_path, published)
    elif tamper in ("boolean_direct", "nan_direct"):
        published = _json(published_path)
        published["response_direct"] = (
            False if tamper == "boolean_direct" else float("nan"))
        _write(published_path, published)
    elif tamper == "time":
        published = _json(published_path)
        published["publication_decision_monotonic"] = published["worker_response_ended_monotonic"] - 1
        _write(published_path, published)
    else:
        child = _json(raw_path)
        child["source_after" if tamper == "source" else "archived_report_sha256"] = {}
        _write(raw_path, child)
        published = _json(published_path)
        published["worker_report_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        _write(published_path, published)
    assert _inspect(job_dir)["state"] == "needs_reconciliation"


def test_temporary_unexpected_symlink_and_malformed_records_are_rejected(tmp_path):
    job_dir = tmp_path / "job"
    _make_job(job_dir, lifecycle=True)
    (job_dir / "published.json.tmp").write_text("{}")
    assert _inspect(job_dir)["reason_code"] == "unexpected_or_temporary_artifact"
    (job_dir / "published.json.tmp").unlink()

    raw = job_dir / "worker.raw.json"
    backup = tmp_path / "worker.raw.saved"
    raw.rename(backup)
    raw.symlink_to(backup)
    assert _inspect(job_dir)["reason_code"] == "worker_record_not_regular"

    raw.unlink()
    raw.write_text("not json")
    assert _inspect(job_dir)["reason_code"] == "record_malformed"


@pytest.mark.parametrize("field,value", [
    ("execution_status", "cancelled"),
    ("physical_validation", "passed"),
    ("runner_error", "unexpected failure"),
    ("raw_child_read_error", "malformed raw report"),
])
def test_lifecycle_contradiction_forces_reconciliation(tmp_path, field, value):
    job_dir = tmp_path / "job"
    _make_job(job_dir, lifecycle=True)
    lifecycle = _json(job_dir / "lifecycle.json")
    lifecycle[field] = value
    _write(job_dir / "lifecycle.json", lifecycle)
    assert _inspect(job_dir)["reason_code"] == "lifecycle_mismatch"


def test_copied_legacy_relative_worker_command_is_unresolved(tmp_path):
    job_dir = tmp_path / "copied"
    job_dir.mkdir()
    for name in ("published.json", "worker.raw.json", "worker.resource.json", "lifecycle.json"):
        shutil.copyfile(FIXTURE / name, job_dir / name)
    result = reconcile.inspect_fixed_job_after_restart(job_dir, "complete")
    assert result["state"] == "needs_reconciliation"
    assert result["reason_code"] == "worker_command_mismatch"


def test_changed_resource_exit_is_unresolved_even_when_lifecycle_copy_agrees(tmp_path):
    job_dir = tmp_path / "job"
    _make_job(job_dir, lifecycle=True)
    resource = _json(job_dir / "worker.resource.json")
    resource["exit_code"] = 1
    _write(job_dir / "worker.resource.json", resource)
    lifecycle = _json(job_dir / "lifecycle.json")
    lifecycle["resource"] = resource
    _write(job_dir / "lifecycle.json", lifecycle)
    assert _inspect(job_dir)["reason_code"] == "worker_validation_failed"


def test_current_fixed_archive_drift_is_unresolved(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    _make_job(job_dir, lifecycle=True)
    monkeypatch.setattr(jobs, "_archives", lambda: {})
    assert _inspect(job_dir)["reason_code"] == "archived_reference_drift"


def test_lifecycle_source_snapshot_must_match_current_sources(tmp_path):
    job_dir = tmp_path / "job"
    _make_job(job_dir, lifecycle=True)
    lifecycle = _json(job_dir / "lifecycle.json")
    lifecycle["source_before"]["examples/weather_scenarios/fv_response_job_lifecycle.py"] = "0" * 64
    lifecycle["source_after"] = dict(lifecycle["source_before"])
    _write(job_dir / "lifecycle.json", lifecycle)
    assert _inspect(job_dir)["reason_code"] == "lifecycle_mismatch"


def test_two_inspectors_contend_on_shared_lock(tmp_path):
    job_dir = tmp_path / "job"
    _make_job(job_dir, lifecycle=True)
    entered, release = Event(), Event()
    outcomes = []

    def inspect_while_locked():
        with jobs._exclusive_job_launch(job_dir):
            entered.set()
            assert release.wait(5)

    first = Thread(target=inspect_while_locked)
    first.start()
    assert entered.wait(5)
    with pytest.raises(jobs.ResponseJobBusyError):
        _inspect(job_dir)
    release.set()
    first.join(5)
    assert not first.is_alive()
    outcomes.append(_inspect(job_dir))
    assert outcomes[0]["state"] == "publication_candidate"


def test_parent_path_aliases_use_the_same_lock_and_classification(tmp_path):
    parent = tmp_path / "real"
    parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    job_dir = parent / "job"
    _make_job(job_dir, lifecycle=True)
    with jobs._exclusive_job_launch(job_dir):
        with pytest.raises(jobs.ResponseJobBusyError):
            reconcile.inspect_fixed_job_after_restart(alias / "job", "request_1")
    assert reconcile.inspect_fixed_job_after_restart(alias / "job", "request_1")["state"] == "publication_candidate"


def test_invalid_request_id_does_not_inspect_or_create_job_artifacts(tmp_path):
    result = reconcile.inspect_fixed_job_after_restart(tmp_path / "not-created", "../bad")
    assert result["state"] == "needs_reconciliation"
    assert result["reason_code"] == "invalid_request_id"
    assert not (tmp_path / "not-created").exists()
