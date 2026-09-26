"""Read-only classification of fixed ``fv4x5`` response-job artifacts."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, cast

from examples.weather_scenarios import fv_process_response_runner as previous
from examples.weather_scenarios import fv_process_response_worker as worker
from examples.weather_scenarios import fv_response_job_lifecycle as jobs
from examples.weather_scenarios import fv_concurrent_response_probe as shared


_PUBLICATION = "published.json"
_RAW = "worker.raw.json"
_RESOURCE = "worker.resource.json"
_LIFECYCLE = "lifecycle.json"
_ALLOWED_ENTRIES = {_PUBLICATION, _RAW, _RESOURCE, _LIFECYCLE, "worker.log"}
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_PUBLICATION_SCOPE = (
    "fixed archived fv4x5 conditional local response; "
    "no nonlinear reanalysis/physical skill"
)


def _result(state: str, reason_code: str, request_id: str) -> dict[str, Any]:
    return {"state": state, "reason_code": reason_code,
            "request_id": request_id, "case_id": "fv4x5"}


def _read_regular_json(path: Path) -> dict[str, Any] | None:
    """Read one regular file without following a final-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError):
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            value = json.load(stream)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
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
        required = {_PUBLICATION, _RAW, _RESOURCE, _LIFECYCLE}
        if not required <= names:
            return _result("needs_reconciliation", "worker_record_missing", request_id)

        paths = {name: directory / name for name in required}
        for path in paths.values():
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

        published = _read_regular_json(paths[_PUBLICATION])
        child = previous._read_child(paths[_RAW])[0]
        resource = _read_regular_json(paths[_RESOURCE])
        if published is None or child is None or resource is None:
            return _result("needs_reconciliation", "record_malformed", request_id)

        try:
            identity, expected_total = jobs._identity()
            sources = jobs._sources()
            archives = jobs._archives()
        except (OSError, RuntimeError, ValueError):
            return _result("needs_reconciliation", "current_reference_unavailable", request_id)
        if archives != shared.ARCHIVED:
            return _result("needs_reconciliation", "archived_reference_drift", request_id)

        raw_path = paths[_RAW]
        if not _valid_command(resource, raw_path):
            return _result("needs_reconciliation", "worker_command_mismatch", request_id)
        command = resource.get("command")
        assert isinstance(command, list)
        completed_time = published.get("child_completed_monotonic")
        if not _finite(completed_time):
            return _result("needs_reconciliation", "child_completion_time_invalid", request_id)
        if not jobs._valid_worker(child, resource, identity, expected_total,
                                  sources, command, cast(float, completed_time)):
            return _result("needs_reconciliation", "worker_validation_failed", request_id)

        response = child.get("response")
        if not isinstance(response, dict):
            return _result("needs_reconciliation", "worker_response_missing", request_id)
        try:
            raw_sha256 = jobs._sha(raw_path)
        except OSError:
            return _result("needs_reconciliation", "worker_report_unreadable", request_id)
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

        lifecycle = _read_regular_json(directory / _LIFECYCLE)
        if lifecycle is None:
            return _result("needs_reconciliation", "lifecycle_malformed", request_id)
        if not _valid_lifecycle(lifecycle, published, resource, request_id,
                                identity, sources, archives):
            return _result("needs_reconciliation", "lifecycle_mismatch", request_id)

        result = _result("publication_candidate", "validated_publication_candidate", request_id)
        result["candidate_is_authoritative"] = False
        result["candidate_limit"] = (
            "The coordinator source snapshot is a mutable lifecycle self-report; artifacts "
            "lack an authenticated attempt manifest, boot ID, and resource-report digest."
        )
        return result
