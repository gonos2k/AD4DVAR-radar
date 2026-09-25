"""Cheap guards for the bounded point-response no-solver launch."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_response_preflight as probe
from examples.weather_scenarios import fv_point_response_preflight_runner as runner


def test_fixed_point_inputs_keep_correlation_and_middle_time_direction():
    problem, warm, parameters, direction = probe.fixed_problem()
    assert warm.shape == (26,) and parameters.shape == direction.shape == (13,)
    assert problem.observation_status is None
    assert torch.equal(direction, torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float64
    ))
    correlation = problem.observation_correlation
    assert correlation is not None
    assert torch.linalg.eigvalsh(correlation).min() > 0
    assert correlation[0, 2] == correlation[2, 0] == 0.3
    assert correlation[1, 3] == correlation[3, 1] == -0.2


@pytest.mark.parametrize("mutation", [
    None, "exit_code", "resource_termination", "monitor_error", "rss_samples",
    "peak_rss", "child_pid", "child_status", "signature",
])
def test_outer_runner_requires_resource_and_complete_child(monkeypatch, tmp_path, mutation):
    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 120 and rss_bytes == 1024**3
        assert command[1:3] == ["-m", "examples.weather_scenarios.fv_point_response_preflight"]
        assert Path(command[-1]).name == "point_response_preflight.json"
        child: dict[str, Any] = {
            "status": "preflight_only", "numerical_solver_runs": 0,
            "source_unchanged": True, "response_validation": "not_performed",
            "pid": 123, "warm_branch": {"choices": [0] * 54, "face_signs": [1] * 54},
        }
        resource: dict[str, Any] = {
            "command": command, "child_pid": 123, "exit_code": 0,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 120, "rss_limit_bytes": 1024**3,
            "rss_samples": 2, "sampled_peak_rss_bytes": 123456,
        }
        if mutation == "exit_code":
            resource["exit_code"] = 1
        elif mutation == "resource_termination":
            resource["resource_termination"] = "rss_limit"
        elif mutation == "monitor_error":
            resource["monitor_error"] = "monitor failed"
        elif mutation == "rss_samples":
            resource["rss_samples"] = 0
        elif mutation == "peak_rss":
            resource["sampled_peak_rss_bytes"] = 1024**3 + 1
        elif mutation == "child_pid":
            child["pid"] = 456
        elif mutation == "child_status":
            child["status"] = "execution_error"
        elif mutation == "signature":
            child["warm_branch"]["face_signs"] = []
        Path(command[-1]).write_text(json.dumps(child))
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource

    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / "attempt")
    assert result["execution_status"] == ("completed" if mutation is None else "failed")
    assert (tmp_path / "attempt/point_response_preflight.run.json").is_file()
