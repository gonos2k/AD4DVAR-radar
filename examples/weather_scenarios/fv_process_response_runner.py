"""Launch two independently guarded FV response interpreters concurrently."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from threading import Barrier
import time
from typing import Any, TypeGuard

from examples.weather_scenarios.fv86_resource_runner import run_guarded


ROOT = Path(__file__).resolve().parents[2]
CHILD_WALL_SECONDS = 300
CHILD_RSS_BYTES = 768 * 1024**2
CASES = ("fv4x5", "fv8x10")
ARCHIVED_REPORT_SHA256 = {
    "minmod_middle_time_bias_final.json": "a0f9568b59bce283b39d1527d3ab64f7034d41c3d41eb3e843ca0ffd7b8fce88",
    "minmod_parameter_vjp.json": "b207c838945baaedffdd3d53e6a41ad124950b9b4a96ea771f59953c0bd25456",
    "fv86_seed_a.json": "c8de13325e4d76a5e904a6ef457deed853aa762cc6d051afd7fb59cd6b9e3bd0",
    "fv86_reanalysis.json": "c80282301138ccae67ce49b86c1d1f2dcc88bc8a144b571eecbe23904ed5a302",
}
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
SOURCE_PATHS = (
    "src/advar/matrix_free.py", "src/advar/transport.py",
    "src/advar/local_response.py", "src/advar/fv_research_problem.py",
    "src/advar/variational.py", "src/advar/physics.py", "src/advar/nowcast.py",
    "examples/weather_scenarios/fv_minmod_inverse_probe.py",
    "examples/weather_scenarios/fv_sensitivity_probe.py",
    "examples/weather_scenarios/fv_scaled_research_case.py",
    "examples/weather_scenarios/fv_concurrent_response_probe.py",
    "examples/weather_scenarios/fv_concurrent_response_runner.py",
    "examples/weather_scenarios/fv86_resource_runner.py",
    "examples/weather_scenarios/fv_process_response_worker.py",
    "examples/weather_scenarios/fv_process_response_runner.py",
)


def _hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def _archive_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
            for name in ARCHIVED_REPORT_SHA256}


def _read_child(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "child report missing"
    try:
        child = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(child, dict):
        return None, "child report is not an object"
    return child, None


def _valid_hvp_intervals(intervals: object) -> TypeGuard[list[list[float]]]:
    return (isinstance(intervals, list) and len(intervals) >= 2
            and all(isinstance(pair, list) and len(pair) == 2
                    and all(type(value) in (int, float) and math.isfinite(value)
                            for value in pair)
                    and pair[0] < pair[1] for pair in intervals)
            and all(first[0] <= second[0] for first, second in zip(intervals, intervals[1:])))


def run(directory: Path) -> dict[str, Any]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("process-isolated run directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    source_before = _hashes()
    archive_before = _archive_hashes()
    if archive_before != ARCHIVED_REPORT_SHA256:
        raise ValueError("process-isolated archive baseline changed before launch")
    launch_barrier = Barrier(2)
    started = time.monotonic()

    def guarded(case_id: str) -> dict[str, Any]:
        report_path = directory / f"{case_id}.json"
        resource_path = directory / f"{case_id}.resource.json"
        log_path = directory / f"{case_id}.log"
        command = [
            sys.executable, "-m", "examples.weather_scenarios.fv_process_response_worker",
            "--case", case_id, "--output", str(report_path),
        ]
        launch_barrier.wait(timeout=30)
        try:
            resource = run_guarded(
                command, wall_seconds=CHILD_WALL_SECONDS,
                rss_bytes=CHILD_RSS_BYTES, report_path=resource_path,
                log_path=log_path,
            )
        except Exception as error:
            resource = {
                "exit_code": None, "resource_termination": "runner_error",
                "monitor_error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": None, "sampled_peak_rss_bytes": 0,
            }
        child, read_error = _read_child(report_path)
        return {"resource": resource, "child": child,
                "child_read_error": read_error,
                "child_report_sha256": (
                    hashlib.sha256(report_path.read_bytes()).hexdigest()
                    if report_path.exists() else None
                )}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {case_id: pool.submit(guarded, case_id) for case_id in CASES}
        outcomes = {case_id: future.result(timeout=CHILD_WALL_SECONDS + 30)
                    for case_id, future in futures.items()}
    source_after = _hashes()
    archive_after = _archive_hashes()
    child_pids = []
    valid = source_before == source_after and archive_before == archive_after
    for case_id in CASES:
        outcome = outcomes[case_id]
        resource = outcome["resource"]
        child = outcome["child"]
        response = child.get("response") if isinstance(child, dict) else None
        identity = child.get("input_identity") if isinstance(child, dict) else None
        monitor = response.get("pcg_monitor") if isinstance(response, dict) else None
        events = monitor.get("hvp_event_times") if isinstance(monitor, dict) else None
        intervals = monitor.get("hvp_intervals") if isinstance(monitor, dict) else None
        expected_command = [
            sys.executable, "-m", "examples.weather_scenarios.fv_process_response_worker",
            "--case", case_id, "--output", str(directory / f"{case_id}.json"),
        ]
        resource_ok = (
            resource.get("exit_code") == 0
            and resource.get("resource_termination") is None
            and resource.get("monitor_error") is None
            and resource.get("command") == expected_command
            and type(resource.get("child_pid")) is int and resource["child_pid"] > 0
            and type(resource.get("rss_samples")) is int and resource["rss_samples"] >= 1
            and resource.get("wall_limit_seconds") == CHILD_WALL_SECONDS
            and resource.get("rss_limit_bytes") == CHILD_RSS_BYTES
            and isinstance(resource.get("elapsed_seconds"), (int, float))
            and math.isfinite(resource["elapsed_seconds"])
            and resource["elapsed_seconds"] <= CHILD_WALL_SECONDS
            and type(resource.get("sampled_peak_rss_bytes")) is int
            and 0 < resource["sampled_peak_rss_bytes"] <= CHILD_RSS_BYTES
        )
        child_ok = (
            isinstance(child, dict)
            and child.get("case_id") == case_id
            and child.get("status") == "completed"
            and child.get("phase") == "finished"
            and child.get("source_unchanged") is True
            and child.get("inputs_unchanged") is True
            and child.get("source_before") == source_before
            and child.get("source_after") == source_before
            and child.get("archived_report_sha256") == ARCHIVED_REPORT_SHA256
            and child.get("executable") == sys.executable
            and child.get("python") == platform.python_version()
            and isinstance(child.get("torch"), str) and bool(child["torch"])
            and isinstance(identity, dict)
            and all(key in identity for key in (
                "problem", "control", "parameters", "verification", "direction"
            ))
            and child.get("input_identity_after") == identity
            and type(child.get("pid")) is int and child["pid"] > 0
            and child["pid"] != os.getpid()
            and child["pid"] == resource.get("child_pid")
            and isinstance(response, dict)
            and response.get("branch_euler_stages") == (54 if case_id == "fv4x5" else 108)
            and response.get("stage_observer_events") == response.get("branch_euler_stages")
            and response.get("branch_choices_sha256") == response.get("archived_branch_choices_sha256")
            and response.get("branch_face_signs_sha256") == response.get("archived_branch_face_signs_sha256")
            and all(type(response.get(key)) in (int, float)
                    and math.isfinite(response[key]) for key in (
                        "gradient_max", "true_adjoint_relative_residual",
                        "total", "archived_total", "relative_difference"
                    ))
            and response["gradient_max"] < 1e-10
            and response["true_adjoint_relative_residual"] <= 1e-10
            and response["relative_difference"] <= 1e-6
            and isinstance(monitor, dict)
            and monitor.get("converged") is True
            and monitor.get("hvp_calls") == response.get("hvp_count")
            and monitor.get("iterations") == response.get("pcg_iterations")
            and monitor.get("relative_residual") == response.get("pcg_relative_residual")
            and isinstance(events, list) and len(events) >= 2
            and all(type(event) in (int, float) and math.isfinite(event) for event in events)
            and events[0] < events[-1]
            and _valid_hvp_intervals(intervals)
            and len(intervals) == monitor["hvp_calls"]
            and len(events) == monitor["hvp_calls"]
            and all(start <= event <= stop for (start, stop), event in zip(intervals, events))
            and all(type(child.get(key)) in (int, float) and math.isfinite(child[key])
                    for key in ("response_started_monotonic", "response_ended_monotonic"))
            and child["response_started_monotonic"] < child["response_ended_monotonic"]
            and intervals[0][0] >= child["response_started_monotonic"]
            and intervals[-1][1] <= child["response_ended_monotonic"]
        )
        outcome["qualified"] = bool(resource_ok and child_ok)
        valid = valid and outcome["qualified"]
        if isinstance(child, dict):
            child_pids.append(child.get("pid"))
    distinct_pids = (len(child_pids) == 2 and all(type(pid) is int for pid in child_pids)
                     and len(set(child_pids)) == 2)
    valid = valid and distinct_pids
    if valid:
        torch_versions = {outcomes[name]["child"]["torch"] for name in CASES}
        valid = len(torch_versions) == 1

    overlap = None
    hvp_overlap = None
    if valid:
        children = [outcomes[name]["child"] for name in CASES]
        overlap = min(child["response_ended_monotonic"] for child in children) - max(
            child["response_started_monotonic"] for child in children
        )
        first_intervals = children[0]["response"]["pcg_monitor"]["hvp_intervals"]
        second_intervals = children[1]["response"]["pcg_monitor"]["hvp_intervals"]
        hvp_overlap = max(
            min(first[1], second[1]) - max(first[0], second[0])
            for first in first_intervals for second in second_intervals
        )
        valid = overlap > 0 and hvp_overlap > 0

    summed_sampled_peaks = sum(
        outcomes[name]["resource"].get("sampled_peak_rss_bytes", 0)
        for name in CASES
    )
    valid = valid and summed_sampled_peaks <= 2 * CHILD_RSS_BYTES
    result: dict[str, Any] = {
        "execution_status": "completed" if valid else "failed",
        "scope": "two OS child processes; no in-process concurrent forward AD",
        "parent_pid": os.getpid(), "source_before": source_before,
        "source_after": source_after,
        "archive_before": archive_before,
        "archive_after": archive_after,
        "cases": outcomes,
        "distinct_child_pids": distinct_pids,
        "response_overlap_seconds": overlap,
        "pcg_hvp_work_overlap_seconds": hvp_overlap,
        "summed_sampled_child_peak_rss_bytes": summed_sampled_peaks,
        "sampled_child_rss_limit_bytes_each": CHILD_RSS_BYTES,
        "parent_elapsed_seconds": time.monotonic() - started,
        "memory_scope": "sum of separate sampled child RSS peaks, excludes parent and missed between-sample spikes",
    }
    (directory / "process_response.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("cases", "source_before", "source_after")},
                     indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
