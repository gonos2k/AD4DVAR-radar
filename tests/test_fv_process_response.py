"""Fast preflight and parent-contract tests for process-isolated FV response."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import (
    fv_concurrent_response_probe as shared,
    fv_process_response_runner as runner,
    fv_process_response_worker as worker,
)


def test_worker_and_runner_bind_the_same_sources():
    assert worker.SOURCE_PATHS == runner.SOURCE_PATHS


def test_worker_attempt_id_is_provenance_only_and_legacy_reports_omit_it(
    tmp_path, monkeypatch
):
    attempt_id = "123e4567-e89b-42d3-a456-426614174000"
    problem = SimpleNamespace(
        identity={"fixed_problem_sha256": "problem"}, verification=object(),
        objective=object(), score=object(),
    )
    monkeypatch.setattr(
        shared, "_small_case", lambda: (problem, "control", "parameters", "direction", 0.08)
    )
    monkeypatch.setattr(shared, "_tensor_identity", lambda value: str(value))
    monkeypatch.setattr(worker, "_hashes", lambda: {"worker": "source"})
    response_calls = []

    def stop_before_response(*args, **kwargs):
        response_calls.append((args, kwargs))
        raise RuntimeError("test stop after running report")

    monkeypatch.setattr(worker, "compute_local_response", stop_before_response)
    writes: dict[Path, list[dict[str, Any]]] = {}
    original_write_text = Path.write_text

    def capture_report(path, contents, *args, **kwargs):
        result = original_write_text(path, contents, *args, **kwargs)
        if path in writes:
            writes[path].append(json.loads(contents))
        return result

    monkeypatch.setattr(Path, "write_text", capture_report)
    tagged_path = tmp_path / "tagged.json"
    legacy_path = tmp_path / "legacy.json"
    writes[tagged_path] = []
    writes[legacy_path] = []

    with pytest.raises(RuntimeError, match="test stop"):
        worker.run("fv4x5", tagged_path, attempt_id)
    with pytest.raises(RuntimeError, match="test stop"):
        worker.run("fv4x5", legacy_path)

    tagged_running, tagged_final = writes[tagged_path]
    legacy_running, legacy_final = writes[legacy_path]
    assert tagged_running["status"] == "running"
    assert tagged_final["status"] == "execution_error"
    assert tagged_running["attempt_id"] == tagged_final["attempt_id"] == attempt_id
    assert "attempt_id" not in legacy_running
    assert "attempt_id" not in legacy_final
    assert response_calls[0][0] == response_calls[1][0]
    assert response_calls[0][1].keys() == response_calls[1][1].keys()
    assert response_calls[0][1]["input_identity"] == response_calls[1][1]["input_identity"]


@pytest.mark.parametrize(
    "attempt_id",
    (
        "123E4567-E89B-42D3-A456-426614174000",
        "{123e4567-e89b-42d3-a456-426614174000}",
        "123e4567-e89b-12d3-a456-426614174000",
        "123e4567-e89b-42d3-0456-426614174000",
        "123e4567e89b42d3a456426614174000",
    ),
)
def test_worker_rejects_noncanonical_attempt_id_in_run(tmp_path, attempt_id):
    with pytest.raises(ValueError, match="canonical UUID4"):
        worker.run("fv4x5", tmp_path / "invalid-run.json", attempt_id)


def test_worker_rejects_invalid_attempt_id_in_cli(tmp_path):
    output_path = tmp_path / "invalid-cli.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.weather_scenarios.fv_process_response_worker",
            "--case",
            "fv4x5",
            "--attempt-id",
            "not-a-uuid",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "canonical UUID4" in result.stderr
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("case_factory", "archive_names", "direction_slice", "expected_stages"),
    [
        (
            shared._small_case,
            ("minmod_middle_time_bias_final.json", "minmod_parameter_vjp.json"),
            slice(20, 40),
            54,
        ),
        (
            shared._large_case,
            ("fv86_seed_a.json", "fv86_reanalysis.json"),
            slice(80, 160),
            108,
        ),
    ],
)
def test_case_archives_direction_branch_and_nominal_gradient_without_pcg(
    monkeypatch, case_factory, archive_names, direction_slice, expected_stages
):
    # Case construction and the stationarity/branch preflight must stay outside
    # the local-response PCG/HVP path.
    def unexpected_pcg(*args, **kwargs):
        raise AssertionError("preflight must not enter PCG")

    monkeypatch.setattr(shared.matrix_free, "pcg", unexpected_pcg)

    for name in archive_names:
        assert shared._hash(shared.EVIDENCE / name) == shared.ARCHIVED[name]

    problem, control, parameters, direction, _ = case_factory()

    assert torch.count_nonzero(direction).item() == direction_slice.stop - direction_slice.start
    expected_direction = torch.zeros_like(direction)
    expected_direction[direction_slice] = 1.0
    assert torch.equal(direction, expected_direction)

    branch, _ = problem.branch_check(control, parameters)
    assert branch["euler_stages"] == expected_stages
    assert branch["choices"] == problem.expected_branch["choices"]
    assert branch["face_signs"] == problem.expected_branch["face_signs"]

    gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    assert bool(torch.isfinite(gradient).all())
    assert gradient.abs().max().item() < 1e-10


def _child_report(case_id, source_hashes, pid, *, response_start, response_end,
                  hvp_times, source_after=None) -> dict[str, Any]:
    stages = 54 if case_id == "fv4x5" else 108
    return {
        "case_id": case_id,
        "status": "completed",
        "phase": "finished",
        "pid": pid,
        "executable": sys.executable,
        "python": platform.python_version(),
        "torch": "2.13.0",
        "source_before": source_hashes,
        "source_after": source_hashes if source_after is None else source_after,
        "source_unchanged": True,
        "inputs_unchanged": True,
        "archived_report_sha256": runner.ARCHIVED_REPORT_SHA256,
        "input_identity": {
            "problem": f"{case_id}-problem",
            **{name: {"shape": [1], "dtype": "torch.float64", "sha256": f"{case_id}-{name}"}
               for name in ("control", "parameters", "verification", "direction")},
        },
        "response_started_monotonic": response_start,
        "response_ended_monotonic": response_end,
        "response": {
            "branch_euler_stages": stages,
            "stage_observer_events": stages,
            "branch_choices_sha256": "choices",
            "archived_branch_choices_sha256": "choices",
            "branch_face_signs_sha256": "faces",
            "archived_branch_face_signs_sha256": "faces",
            "gradient_max": 0.0,
            "true_adjoint_residual": 0.0,
            "true_adjoint_relative_residual": 0.0,
            "pcg_relative_residual": 1e-12,
            "pcg_iterations": 4,
            "hvp_count": len(hvp_times),
            "direct": 0.1,
            "indirect": -0.02,
            "total": 0.08,
            "archived_total": 0.08,
            "relative_difference": 0.0,
            "pcg_monitor": {
                "hvp_calls": len(hvp_times),
                "hvp_event_times": hvp_times,
                "hvp_intervals": [[time - 0.1, time] for time in hvp_times],
                "converged": True,
                "iterations": 4,
                "relative_residual": 1e-12,
            },
        },
    }


def _install_mock_children(monkeypatch, *, failure=None):
    source_before = {name: f"before-{name}" for name in runner.SOURCE_PATHS}
    source_after = dict(source_before)
    if failure == "source_change":
        source_after[runner.SOURCE_PATHS[0]] = "changed"
    hashes = iter((source_before, source_after))
    monkeypatch.setattr(runner, "_hashes", lambda: next(hashes))
    monkeypatch.setattr(
        runner, "_archive_hashes", lambda: dict(runner.ARCHIVED_REPORT_SHA256)
    )

    commands = []
    commands_lock = threading.Lock()

    def fake_run_guarded(command, **kwargs):
        with commands_lock:
            commands.append(list(command))
        case_id = command[command.index("--case") + 1]
        output_path = Path(command[command.index("--output") + 1])
        start, end, hvp = (100.0, 110.0, [101.0, 109.0])
        if failure == "response_no_overlap" and case_id == "fv8x10":
            start, end, hvp = 120.0, 130.0, [121.0, 129.0]
        elif failure == "hvp_no_overlap" and case_id == "fv8x10":
            hvp = [104.0, 106.0]
        child = _child_report(
            case_id, source_before, os.getpid() + (100 if case_id == "fv4x5" else 101),
            response_start=start, response_end=end, hvp_times=hvp,
        )
        if failure == "missing_identity":
            child.pop("input_identity")
        elif failure == "missing_response_fields":
            child["response"].pop("relative_difference")
        else:
            child["input_identity_after"] = child["input_identity"]
        if failure == "bad_hvp_interval" and case_id == "fv8x10":
            child["response"]["pcg_monitor"]["hvp_intervals"][0] = [99.0, 101.0]
        if failure == "malformed_child":
            output_path.write_text("{")
        else:
            output_path.write_text(json.dumps(child))
        return {
            "exit_code": 1 if failure == "nonzero_exit" else 0,
            "resource_termination": "wall_time_limit" if failure == "resource_wall" else (
                "rss_limit" if failure == "resource_rss" else None
            ),
            "monitor_error": "ps failed" if failure == "monitor_error" else None,
            "command": list(command),
            "child_pid": child["pid"] + (1 if failure == "pid_mismatch" else 0),
            "rss_samples": 0 if failure == "zero_samples" else 2,
            "wall_limit_seconds": runner.CHILD_WALL_SECONDS,
            "rss_limit_bytes": runner.CHILD_RSS_BYTES,
            "elapsed_seconds": runner.CHILD_WALL_SECONDS + 0.01
            if failure == "over_wall_limit" else 1.0,
            "sampled_peak_rss_bytes": runner.CHILD_RSS_BYTES + 1
            if failure == "rss_over" else runner.CHILD_RSS_BYTES,
        }

    monkeypatch.setattr(runner, "run_guarded", fake_run_guarded)
    return source_before, commands


def test_runner_requires_two_complete_independent_children_and_both_overlaps(
    tmp_path, monkeypatch
):
    source_before, commands = _install_mock_children(monkeypatch)

    result = runner.run(tmp_path)

    assert result["execution_status"] == "completed"
    assert result["source_before"] == source_before
    assert result["source_after"] == source_before
    assert result["archive_before"] == result["archive_after"] == runner.ARCHIVED_REPORT_SHA256
    assert result["distinct_child_pids"] is True
    child_pids = [result["cases"][case_id]["child"]["pid"] for case_id in runner.CASES]
    assert len(set(child_pids)) == 2
    assert all(pid != result["parent_pid"] for pid in child_pids)
    assert result["response_overlap_seconds"] > 0
    assert result["pcg_hvp_work_overlap_seconds"] > 0
    assert len(commands) == 2
    command_cases = [command[command.index("--case") + 1] for command in commands]
    assert set(command_cases) == set(runner.CASES)
    assert all(command.count("--case") == 1 for command in commands)
    assert all("-m" in command for command in commands)

    for case_id in runner.CASES:
        outcome = result["cases"][case_id]
        child = outcome["child"]
        assert outcome["qualified"] is True
        assert child["source_before"] == child["source_after"] == source_before
        assert set(child["input_identity"]) == {
            "problem", "control", "parameters", "verification", "direction"
        }
        response = child["response"]
        assert response["branch_euler_stages"] == (54 if case_id == "fv4x5" else 108)
        assert response["stage_observer_events"] == response["branch_euler_stages"]
        assert response["branch_choices_sha256"] == response["archived_branch_choices_sha256"]
        assert response["branch_face_signs_sha256"] == response["archived_branch_face_signs_sha256"]
        assert response["pcg_monitor"]["hvp_calls"] >= 2
        assert response["pcg_monitor"]["hvp_event_times"]
        assert response["relative_difference"] <= 1e-6


@pytest.mark.parametrize(
    "failure",
    (
        "nonzero_exit",
        "malformed_child",
        "source_change",
        "missing_identity",
        "missing_response_fields",
        "response_no_overlap",
        "hvp_no_overlap",
        "over_wall_limit",
        "resource_wall",
        "resource_rss",
        "monitor_error",
        "zero_samples",
        "rss_over",
        "pid_mismatch",
        "bad_hvp_interval",
    ),
)
def test_runner_fails_closed_on_child_or_overlap_failures(tmp_path, monkeypatch, failure):
    _install_mock_children(monkeypatch, failure=failure)

    result = runner.run(tmp_path)

    assert result["execution_status"] == "failed"
