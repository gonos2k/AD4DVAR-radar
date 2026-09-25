"""Process-isolated, cancellable publication of one fixed FV response."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from threading import Lock
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_concurrent_response_probe as shared
from examples.weather_scenarios import fv_process_response_runner as previous
from examples.weather_scenarios import fv_process_response_worker as worker
from examples.weather_scenarios.fv86_resource_runner import run_guarded


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_RESPONSE_JOB_LIFECYCLE_PLAN.md"
PLAN_SHA256 = "b2cba4e1e092298b31a5b9208da98a17f59032f15e79670650408615fe8ca43e"
CHILD_WALL_SECONDS = 300
CHILD_RSS_BYTES = 768 * 1024**2
SOURCE_PATHS = tuple(dict.fromkeys((*worker.SOURCE_PATHS,
    "examples/weather_scenarios/fv_response_job_lifecycle.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _archives() -> dict[str, str]:
    return {name: _sha(EVIDENCE / name) for name in shared.ARCHIVED}


def _identity() -> tuple[dict[str, Any], float]:
    problem, control, parameters, direction, expected = shared._small_case()
    identity = {
        "problem": problem.identity["fixed_problem_sha256"],
        "control": shared._tensor_identity(control),
        "parameters": shared._tensor_identity(parameters),
        "verification": shared._tensor_identity(problem.verification),
        "direction": shared._tensor_identity(direction),
    }
    return identity, expected


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


class CancellationToken:
    """Serialize cancellation with the final atomic publication decision."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._cancelled = False
        self._cancelled_at: float | None = None
        self._published = False

    def cancel(self) -> bool:
        with self._lock:
            if self._published:
                return False
            self._cancelled = True
            if self._cancelled_at is None:
                self._cancelled_at = time.monotonic()
            return True

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def cancelled_at(self) -> float | None:
        with self._lock:
            return self._cancelled_at

    def publish_if_active(self, path: Path, data: dict[str, Any]) -> bool:
        with self._lock:
            if self._cancelled:
                return False
            data["publication_decision_monotonic"] = time.monotonic()
            _atomic_json(path, data)
            self._published = True
            return True


def _valid_worker(child: object, resource: dict[str, Any],
                  expected_identity: dict[str, Any], expected_total: float,
                  source: dict[str, str], command: list[str],
                  completed_monotonic: float) -> bool:
    if not isinstance(child, dict):
        return False
    response = child.get("response")
    if not isinstance(response, dict):
        return False
    monitor = response.get("pcg_monitor")
    if not isinstance(monitor, dict):
        return False
    intervals = monitor.get("hvp_intervals")
    events = monitor.get("hvp_event_times")
    if not (
        resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("command") == command
        and type(resource.get("child_pid")) is int and resource["child_pid"] > 0
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= CHILD_RSS_BYTES
        and resource.get("wall_limit_seconds") == CHILD_WALL_SECONDS
        and resource.get("rss_limit_bytes") == CHILD_RSS_BYTES
        and _finite(resource.get("elapsed_seconds"))
        and 0 <= resource["elapsed_seconds"] <= CHILD_WALL_SECONDS
        and child.get("case_id") == "fv4x5"
        and child.get("status") == "completed"
        and child.get("phase") == "finished"
        and child.get("source_unchanged") is True
        and child.get("inputs_unchanged") is True
        and child.get("source_before") == child.get("source_after") == worker._hashes()
        and child.get("archived_report_sha256") == shared.ARCHIVED
        and child.get("input_identity") == child.get("input_identity_after") == expected_identity
        and type(child.get("pid")) is int and child["pid"] == resource["child_pid"]
        and child.get("executable") == sys.executable
        and response.get("branch_euler_stages") == 54
        and response.get("stage_observer_events") == 54
        and response.get("branch_choices_sha256") == response.get("archived_branch_choices_sha256")
        and response.get("branch_face_signs_sha256") == response.get("archived_branch_face_signs_sha256")
        and all(_finite(response.get(key)) for key in (
            "gradient_max", "true_adjoint_relative_residual", "pcg_relative_residual",
            "direct", "indirect", "total", "relative_difference",
        ))
        and 0 <= response["gradient_max"] < 1e-10
        and 0 <= response["true_adjoint_relative_residual"] <= 1e-10
        and 0 <= response["pcg_relative_residual"] <= 1e-10
        and abs(response["direct"] + response["indirect"] - response["total"]) <= 1e-12
        and abs(response["total"] - expected_total) <= 1e-6 * abs(expected_total)
        and response["relative_difference"] <= 1e-6
        and monitor.get("converged") is True
        and monitor.get("hvp_calls") == response.get("hvp_count")
        and monitor.get("iterations") == response.get("pcg_iterations")
        and monitor.get("relative_residual") == response.get("pcg_relative_residual")
        and previous._valid_hvp_intervals(intervals)
        and isinstance(events, list) and len(events) == len(intervals)
        and all(_finite(value) for value in events)
        and all(start <= value <= stop for (start, stop), value in zip(intervals, events))
        and len(intervals) == monitor["hvp_calls"]
        and _finite(child.get("response_started_monotonic"))
        and _finite(child.get("response_ended_monotonic"))
        and child["response_started_monotonic"] < child["response_ended_monotonic"]
        and intervals[0][0] >= child["response_started_monotonic"]
        and intervals[-1][1] <= child["response_ended_monotonic"]
        and child["response_ended_monotonic"] <= completed_monotonic
    ):
        return False
    return source == _sources()


def run_job(directory: Path, *, request_id: str,
            token: CancellationToken | None = None) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
        raise ValueError("response request_id must be 1-64 safe characters")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("response job directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    token = token if token is not None else CancellationToken()
    started = time.monotonic()
    source_before = _sources()
    archive_before = _archives()
    identity_before, expected_total = _identity()
    if _sha(PLAN) != PLAN_SHA256 or archive_before != shared.ARCHIVED:
        raise ValueError("response job plan or archived input changed before launch")
    raw_path = directory / "worker.raw.json"
    command = [sys.executable,
               str(ROOT / "examples/weather_scenarios/fv_process_response_worker.py"),
               "--case", "fv4x5", "--output", str(raw_path)]
    error = None
    resource: dict[str, Any] | None = None
    if not token.is_cancelled():
        try:
            resource = run_guarded(
                command, wall_seconds=CHILD_WALL_SECONDS,
                rss_bytes=CHILD_RSS_BYTES,
                report_path=directory / "worker.resource.json",
                log_path=directory / "worker.log",
                cancel_requested=token.is_cancelled,
            )
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
    completed_monotonic = time.monotonic()
    child, read_error = previous._read_child(raw_path)
    source_after = _sources()
    archive_after = _archives()
    identity_after, _ = _identity()
    identities_ok = (source_before == source_after
                     and archive_before == archive_after == shared.ARCHIVED
                     and identity_before == identity_after
                     and _sha(PLAN) == PLAN_SHA256)
    numerically_eligible = bool(
        resource is not None and identities_ok
        and _valid_worker(child, resource, identity_before,
                          expected_total, source_before, command,
                          completed_monotonic)
    )
    publication_time = None
    published = False
    if numerically_eligible and isinstance(child, dict):
        response = child["response"]
        published = token.publish_if_active(directory / "published.json", {
            "request_id": request_id, "case_id": "fv4x5",
            "response_total": response["total"],
            "response_direct": response["direct"],
            "response_indirect": response["indirect"],
            "true_adjoint_relative_residual": response["true_adjoint_relative_residual"],
            "input_identity": identity_before,
            "worker_report_sha256": _sha(raw_path),
            "worker_pid": child["pid"],
            "worker_response_ended_monotonic": child["response_ended_monotonic"],
            "child_completed_monotonic": completed_monotonic,
            "scope": "fixed archived fv4x5 conditional local response; no nonlinear reanalysis/physical skill",
        })
        if published:
            publication_time = time.monotonic()
    cancellation = token.is_cancelled()
    termination = resource.get("resource_termination") if resource is not None else None
    status = ("completed" if published else "cancelled" if cancellation
              else "resource_limited" if termination in ("wall_time_limit", "rss_limit")
              else "failed")
    result: dict[str, Any] = {
        "request_id": request_id,
        "execution_status": status,
        "numerical_status": "eligible" if numerically_eligible else "not_verified",
        "response_validation": "archived_reference_matched" if published else "not_performed",
        "physical_validation": "not_performed",
        "response_published": published,
        "resource": resource,
        "raw_child_read_error": read_error,
        "runner_error": error,
        "cancellation_requested_monotonic": token.cancelled_at(),
        "source_before": source_before,
        "source_after": source_after,
        "archive_before": archive_before,
        "archive_after": archive_after,
        "input_before": identity_before,
        "input_after": identity_after,
        "child_completed_monotonic": completed_monotonic,
        "published_monotonic": publication_time,
        "elapsed_seconds": time.monotonic() - started,
        "memory_scope": "sampled child RSS only; parent and between-sample spikes excluded",
    }
    _atomic_json(directory / "lifecycle.json", result)
    return result


def run_pair(directory: Path) -> dict[str, Any]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("response pair directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    success_token, cancel_token = CancellationToken(), CancellationToken()
    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = pool.submit(run_job, directory / "complete",
                                request_id="complete", token=success_token)
        interrupted = pool.submit(run_job, directory / "cancel",
                                  request_id="cancel", token=cancel_token)
        raw = directory / "cancel/worker.raw.json"
        until = time.monotonic() + 30
        saw_running = False
        while not interrupted.done() and time.monotonic() < until:
            child, _ = previous._read_child(raw)
            if (isinstance(child, dict) and child.get("status") == "running"
                    and child.get("phase") == "response"
                    and type(child.get("pid")) is int and child["pid"] > 0):
                saw_running = True
                break
            time.sleep(0.01)
        cancellation_requested = cancel_token.cancel()
        first = completed.result(timeout=CHILD_WALL_SECONDS + 30)
        second = interrupted.result(timeout=CHILD_WALL_SECONDS + 30)
    first_resource = first.get("resource")
    second_resource = second.get("resource")
    distinct_pids = (isinstance(first_resource, dict) and isinstance(second_resource, dict)
                     and type(first_resource.get("child_pid")) is int
                     and type(second_resource.get("child_pid")) is int
                     and first_resource["child_pid"] != second_resource["child_pid"])
    cancelled_at = cancel_token.cancelled_at()
    cancelled_before_completion = (
        cancelled_at is not None
        and cancelled_at < second["child_completed_monotonic"]
    )
    valid = (saw_running and cancellation_requested
             and first["execution_status"] == "completed"
             and first["response_published"] is True
             and second["execution_status"] == "cancelled"
             and second["response_published"] is False
             and isinstance(second_resource, dict)
             and second_resource.get("resource_termination") == "cancelled"
             and (directory / "complete/published.json").is_file()
             and not (directory / "cancel/published.json").exists()
             and distinct_pids and cancelled_before_completion)
    result = {
        "execution_status": "completed" if valid else "failed",
        "scope": "two isolated fixed fv4x5 jobs; one published after validation, one cancelled without publication",
        "parseable_running_report_seen_before_cancellation": saw_running,
        "cancellation_requested_before_worker_finished": cancelled_before_completion,
        "distinct_child_pids": distinct_pids,
        "jobs": {"complete": first, "cancel": second},
        "elapsed_seconds": time.monotonic() - started,
        "memory_scope": "per-child sampled RSS peaks; no whole-system peak claim",
    }
    _atomic_json(directory / "job_pair.run.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_pair(arguments.directory)
    print(json.dumps({key: value for key, value in result.items() if key != "jobs"},
                     indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
