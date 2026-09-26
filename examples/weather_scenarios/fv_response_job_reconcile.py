"""Read-only classification of fixed ``fv4x5`` response-job artifacts."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from datetime import datetime
from typing import Any, cast
import uuid

from examples.weather_scenarios import fv_process_response_runner as previous
from examples.weather_scenarios import fv_process_response_worker as worker
from examples.weather_scenarios import fv_response_job_lifecycle as jobs
from examples.weather_scenarios import fv_concurrent_response_probe as shared


_PUBLICATION = "published.json"
_RAW = "worker.raw.json"
_RESOURCE = "worker.resource.json"
_LIFECYCLE = "lifecycle.json"
_MANIFEST = "attempt.json"
_ALLOWED_ENTRIES = {_MANIFEST, _PUBLICATION, _RAW, _RESOURCE, _LIFECYCLE, "worker.log"}
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_PUBLICATION_SCOPE = (
    "fixed archived fv4x5 conditional local response; "
    "no nonlinear reanalysis/physical skill"
)


def _result(state: str, reason_code: str, request_id: str) -> dict[str, Any]:
    return {"state": state, "reason_code": reason_code,
            "request_id": request_id, "case_id": "fv4x5"}


def _parse_json_snapshot(snapshot: tuple[bytes, str] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    try:
        value = json.loads(snapshot[0])
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _valid_command(resource: dict[str, Any], raw_path: Path) -> bool:
    command = resource.get("command")
    if not isinstance(command, list) or len(command) != 6:
        return False
    if not isinstance(command[5], str) or not Path(command[5]).is_absolute():
        return False
    try:
        executable = Path(command[0]).resolve(strict=True)
        worker_script = Path(command[1]).resolve(strict=True)
        output_path = Path(command[5]).resolve(strict=True)
        expected_executable = Path(sys.executable).resolve(strict=True)
        expected_worker = (jobs.ROOT / "examples/weather_scenarios"
                           / "fv_process_response_worker.py").resolve(strict=True)
        expected_raw = raw_path.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        executable == expected_executable
        and os.access(executable, os.X_OK)
        and worker_script == expected_worker
        and command[2:5] == ["--case", "fv4x5", "--output"]
        and output_path == expected_raw
    )


def _contains_v2_marker(value: object) -> bool:
    """Detect v2 provenance markers anywhere in any JSON artifact."""
    markers = {
        "attempt_id", "attempt_manifest_sha256", "attempt_manifest_digest",
        "manifest_sha256", "manifest_digest", "resource_report_sha256",
        "resource_report_digest", "resource_digest",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if key in markers:
                return True
            if (key == "schema" or key.endswith("schema_version")
                    or key == "publication_schema"):
                if (isinstance(item, (int, float)) and not isinstance(item, bool)
                        and item >= 2):
                    return True
            if _contains_v2_marker(item):
                return True
    elif isinstance(value, list):
        return any(_contains_v2_marker(item) for item in value)
    return False


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns)


def _stream_log_markers(path: Path) -> tuple[bool, tuple[int, int, int, int, int]]:
    """Scan the log without loading it all, and reject a changing read."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("worker log must be a regular file")
        marker_found = False
        overlap = b""
        bytes_read = 0
        while chunk := os.read(descriptor, 65536):
            bytes_read += len(chunk)
            content = overlap + chunk
            schema_marker = re.search(
                rb"(?:schema(?:_version)?|publication_schema_version)[\"']?\s*[:=]\s*(\d+)",
                content,
            )
            marker_found |= (
                b"--attempt-id" in content or b"attempt_id" in content
                or b"attempt_manifest_sha256" in content
                or b"manifest_sha256" in content
                or b"resource_report_sha256" in content
                or (schema_marker is not None and int(schema_marker.group(1)) >= 2)
            )
            overlap = content[-64:]
        after = os.fstat(descriptor)
        if (_stat_signature(before) != _stat_signature(after)
                or bytes_read != after.st_size):
            raise OSError("worker log changed while it was scanned")
        return marker_found, _stat_signature(after)
    finally:
        os.close(descriptor)


def _has_v2_marker(names: set[str], snapshots: dict[str, tuple[bytes, str]],
                   log_marker: bool) -> bool:
    if _MANIFEST in names or log_marker:
        return True
    return any(
        (record := _parse_json_snapshot(snapshots.get(name))) is not None
        and _contains_v2_marker(record)
        for name in snapshots
    )


def _snapshots_still_match(directory: Path, names: set[str],
                           snapshots: dict[str, tuple[bytes, str]],
                           log_marker: bool,
                           log_signature: tuple[int, int, int, int, int] | None) -> bool:
    try:
        if {entry.name for entry in directory.iterdir()} != names:
            return False
        for name, pinned in snapshots.items():
            if jobs._stable_file_snapshot(directory / name) != pinned:
                return False
        if "worker.log" in names:
            current_marker, current_signature = _stream_log_markers(directory / "worker.log")
            return current_marker == log_marker and current_signature == log_signature
    except (OSError, ValueError):
        return False
    return True


def _v2_field(record: dict[str, Any], *names: str) -> Any:
    """Read a deliberately small set of equivalent serialized field names."""
    for name in names:
        if name in record:
            return record[name]
    return None


def _valid_v2_manifest(manifest: dict[str, Any], directory: Path,
                       request_id: str, identity: dict[str, Any],
                       sources: dict[str, str], archives: dict[str, str],
                       expected_total: float,
                       raw_path: Path, manifest_sha256: str) -> str | None:
    schema = _v2_field(manifest, "schema_version", "schema")
    attempt_id = manifest.get("attempt_id")
    try:
        canonical_dir = directory.resolve(strict=True)
        expected_raw = raw_path.resolve(strict=True)
        uuid_value = uuid.UUID(attempt_id) if isinstance(attempt_id, str) else None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    command = manifest.get("command")
    if not isinstance(command, list) or len(command) != 8:
        return None
    created_utc = manifest.get("created_utc")
    if not isinstance(created_utc, str):
        return None
    try:
        datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    try:
        executable = Path(command[0]).resolve(strict=True)
        script = Path(command[1]).resolve(strict=True)
        output = Path(command[7]).resolve(strict=True)
        expected_executable = Path(sys.executable).resolve(strict=True)
        expected_script = (jobs.ROOT / "examples/weather_scenarios"
                           / "fv_process_response_worker.py").resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    valid = (
        schema == 1
        and uuid_value is not None and uuid_value.version == 4
        and str(uuid_value) == attempt_id
        and manifest.get("request_id") == request_id
        and manifest.get("case_id") == "fv4x5"
        and _v2_field(manifest, "job_dir", "canonical_job_dir") == str(canonical_dir)
        and all(isinstance(part, str) for part in command)
        and Path(command[0]).is_absolute() and Path(command[1]).is_absolute()
        and command[2:4] == ["--case", "fv4x5"]
        and command[4:6] == ["--attempt-id", attempt_id]
        and command[6] == "--output"
        and Path(command[7]).is_absolute() and output == expected_raw
        and executable == expected_executable and os.access(executable, os.X_OK)
        and script == expected_script
        and isinstance(manifest.get("resource_budget"), dict)
        and manifest["resource_budget"].get("wall_seconds") == jobs.CHILD_WALL_SECONDS
        and manifest["resource_budget"].get("rss_limit_bytes") == jobs.CHILD_RSS_BYTES
        and manifest["resource_budget"].get("sampling_seconds") == 0.25
        and manifest["resource_budget"].get("scope") == (
            "sampled child RSS only; parent and between-sample spikes excluded")
        and manifest.get("source_sha256") == sources
        and manifest.get("archive_sha256") == archives
        and manifest.get("input_identity") == identity
        and _finite(manifest.get("expected_response_total"))
        and manifest.get("expected_response_total") == expected_total
    )
    if not valid:
        return None
    try:
        plan_sha = jobs._sha(jobs.EVIDENCE / "FV_RESPONSE_JOB_ATTEMPT_BINDING_PLAN.md")
        if manifest.get("plan_sha256") != plan_sha:
            return None
    except OSError:
        return None
    return attempt_id if isinstance(attempt_id, str) and manifest_sha256 else None


def _valid_lifecycle(lifecycle: dict[str, Any], published: dict[str, Any],
                     resource: dict[str, Any], request_id: str,
                     identity: dict[str, Any], sources: dict[str, str],
                     archives: dict[str, str]) -> bool:
    lifecycle_resource = lifecycle.get("resource")
    publication_time = lifecycle.get("published_monotonic")
    decision_time = published.get("publication_decision_monotonic")
    child_time = published.get("child_completed_monotonic")
    return (
        lifecycle.get("request_id") == request_id
        and lifecycle.get("execution_status") == "completed"
        and lifecycle.get("numerical_status") == "eligible"
        and lifecycle.get("response_validation") == "archived_reference_matched"
        and lifecycle.get("physical_validation") == "not_performed"
        and lifecycle.get("response_published") is True
        and lifecycle.get("raw_child_read_error") is None
        and lifecycle.get("runner_error") is None
        and lifecycle.get("cancellation_requested_monotonic") is None
        and lifecycle_resource == resource
        and lifecycle.get("child_completed_monotonic") == child_time
        and _finite(publication_time) and _finite(decision_time)
        and cast(float, publication_time) >= cast(float, decision_time)
        and lifecycle.get("input_before") == lifecycle.get("input_after") == identity
        and lifecycle.get("source_before") == lifecycle.get("source_after") == sources
        and lifecycle.get("archive_before") == lifecycle.get("archive_after") == archives
    )


def inspect_fixed_job_after_restart(job_dir: Path,
                                    request_id: str) -> dict[str, Any]:
    """Classify fixed-job artifacts while holding the coordinator's job lock.

    This function never starts, signals, or adopts a worker and never writes
    inside ``job_dir``. The lifecycle module's persistent sibling lock may be
    created while acquiring the same lock used by normal coordinators.
    """
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        return _result("needs_reconciliation", "invalid_request_id", str(request_id))

    directory = Path(job_dir)
    with jobs._exclusive_job_launch(directory):
        if not directory.exists():
            return _result("no_job_artifacts", "job_directory_missing", request_id)
        try:
            if not directory.is_dir() or directory.is_symlink():
                return _result("needs_reconciliation", "invalid_job_directory", request_id)
            entries = list(directory.iterdir())
        except OSError:
            return _result("needs_reconciliation", "unreadable_job_directory", request_id)
        if not entries:
            return _result("no_job_artifacts", "empty_job_directory", request_id)

        names = {entry.name for entry in entries}
        if names - _ALLOWED_ENTRIES:
            return _result("needs_reconciliation", "unexpected_or_temporary_artifact", request_id)
        if _PUBLICATION not in names:
            return _result("needs_reconciliation", "publication_missing", request_id)
        if _LIFECYCLE not in names:
            return _result("needs_reconciliation", "lifecycle_missing", request_id)
        base_required = {_PUBLICATION, _RAW, _RESOURCE, _LIFECYCLE}
        if not base_required <= names:
            return _result("needs_reconciliation", "worker_record_missing", request_id)

        json_names = {name for name in names if name.endswith(".json")}
        for name in json_names:
            path = directory / name
            try:
                mode = path.lstat().st_mode
            except OSError:
                return _result("needs_reconciliation", "worker_record_missing", request_id)
            if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                return _result("needs_reconciliation", "worker_record_not_regular", request_id)
        if "worker.log" in names:
            try:
                if not stat.S_ISREG((directory / "worker.log").lstat().st_mode):
                    return _result("needs_reconciliation", "worker_log_not_regular", request_id)
            except OSError:
                return _result("needs_reconciliation", "worker_log_missing", request_id)

        snapshots: dict[str, tuple[bytes, str]] = {}
        try:
            for name in json_names:
                snapshots[name] = jobs._stable_file_snapshot(directory / name)
        except (OSError, ValueError):
            return _result("needs_reconciliation", "artifact_snapshot_unstable", request_id)
        try:
            log_marker, log_signature = (
                _stream_log_markers(directory / "worker.log")
                if "worker.log" in names else (False, None)
            )
        except (OSError, ValueError):
            return _result("needs_reconciliation", "worker_log_unstable", request_id)
        v2 = _has_v2_marker(names, snapshots, log_marker)
        if v2 and _MANIFEST not in names:
            return _result("needs_reconciliation", "attempt_manifest_missing", request_id)

        published = _parse_json_snapshot(snapshots.get(_PUBLICATION))
        child = _parse_json_snapshot(snapshots.get(_RAW))
        resource = _parse_json_snapshot(snapshots.get(_RESOURCE))
        lifecycle = _parse_json_snapshot(snapshots.get(_LIFECYCLE))
        manifest = _parse_json_snapshot(snapshots.get(_MANIFEST)) if v2 else None
        if (published is None or child is None or resource is None
                or lifecycle is None or (v2 and manifest is None)):
            return _result("needs_reconciliation", "record_malformed", request_id)

        try:
            identity, expected_total = jobs._identity()
            sources = jobs._sources()
            archives = jobs._archives()
        except (OSError, RuntimeError, ValueError):
            return _result("needs_reconciliation", "current_reference_unavailable", request_id)
        if archives != shared.ARCHIVED:
            return _result("needs_reconciliation", "archived_reference_drift", request_id)

        raw_path = directory / _RAW
        raw_sha256 = snapshots[_RAW][1]
        resource_sha256 = snapshots[_RESOURCE][1]
        manifest_sha = snapshots[_MANIFEST][1] if v2 else None
        attempt_id = None
        if v2:
            assert manifest is not None
            attempt_id = _valid_v2_manifest(
                manifest, directory, request_id, identity, sources, archives,
                expected_total, raw_path, cast(str, manifest_sha))
            if attempt_id is None or manifest_sha is None:
                return _result("needs_reconciliation", "attempt_manifest_mismatch", request_id)
            if (child.get("attempt_id") != attempt_id
                    or resource.get("attempt_id") != attempt_id
                    or resource.get("attempt_manifest_sha256") != manifest_sha
                    or resource.get("command") != manifest.get("command")):
                return _result("needs_reconciliation", "attempt_worker_binding_mismatch", request_id)
        elif not _valid_command(resource, raw_path):
            return _result("needs_reconciliation", "worker_command_mismatch", request_id)
        command = resource.get("command")
        assert isinstance(command, list)
        completed_time = published.get("child_completed_monotonic")
        if not _finite(completed_time):
            return _result("needs_reconciliation", "child_completion_time_invalid", request_id)
        if not jobs._valid_worker(child, resource, identity, expected_total,
                                  sources, command, cast(float, completed_time),
                                  expected_attempt_id=attempt_id if v2 else None,
                                  expected_manifest_sha256=manifest_sha if v2 else None):
            return _result("needs_reconciliation", "worker_validation_failed", request_id)

        response = child.get("response")
        if not isinstance(response, dict):
            return _result("needs_reconciliation", "worker_response_missing", request_id)
        if v2:
            canonical_dir = str(directory.resolve(strict=True))
            if not (
                published.get("publication_schema_version") == 2
                and published.get("attempt_id") == attempt_id
                and published.get("attempt_manifest_sha256") == manifest_sha
                and published.get("worker_report_sha256") == raw_sha256
                and published.get("resource_report_sha256") == resource_sha256
                and published.get("job_dir") == canonical_dir
                and published.get("source_sha256") == sources
                and published.get("archive_sha256") == archives
                and published.get("input_identity") == identity
            ):
                return _result("needs_reconciliation", "publication_attempt_binding_mismatch", request_id)
        worker_pid = child.get("pid")
        response_end = child.get("response_ended_monotonic")
        decision_time = published.get("publication_decision_monotonic")
        if not (
            published.get("request_id") == request_id
            and published.get("case_id") == "fv4x5"
            and published.get("scope") == _PUBLICATION_SCOPE
            and published.get("worker_report_sha256") == raw_sha256
            and published.get("input_identity") == identity
            and type(worker_pid) is int and published.get("worker_pid") == worker_pid
            and resource.get("child_pid") == worker_pid
            and all(_finite(published.get(name)) for name in (
                "response_total", "response_direct", "response_indirect",
                "true_adjoint_relative_residual",
            ))
            and published.get("response_total") == response.get("total")
            and published.get("response_direct") == response.get("direct")
            and published.get("response_indirect") == response.get("indirect")
            and published.get("true_adjoint_relative_residual")
                == response.get("true_adjoint_relative_residual")
            and published.get("worker_response_ended_monotonic") == response_end
            and _finite(published.get("worker_response_ended_monotonic"))
            and _finite(response_end) and _finite(completed_time)
            and _finite(decision_time)
            and cast(float, response_end) <= cast(float, completed_time)
            and cast(float, completed_time) <= cast(float, decision_time)
        ):
            return _result("needs_reconciliation", "publication_mismatch", request_id)

        if v2 and lifecycle.get("attempt_id") != attempt_id:
            return _result("needs_reconciliation", "lifecycle_attempt_binding_mismatch", request_id)
        if not _valid_lifecycle(lifecycle, published, resource, request_id,
                                identity, sources, archives):
            return _result("needs_reconciliation", "lifecycle_mismatch", request_id)
        if not _snapshots_still_match(directory, names, snapshots,
                                      log_marker, log_signature):
            return _result("needs_reconciliation", "artifact_snapshot_changed", request_id)

        result = _result("publication_candidate", "validated_publication_candidate", request_id)
        result["candidate_is_authoritative"] = False
        result["candidate_limit"] = (
            "The attempt manifest and artifact chain are mutable and unsigned; this "
            "candidate is not authenticated."
            if v2 else
            "The coordinator source snapshot is a mutable lifecycle self-report; artifacts "
            "lack an authenticated attempt manifest, boot ID, and resource-report digest."
        )
        return result
