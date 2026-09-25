"""Classify one guarded point-Newton attempt separately from its result."""

import json
from pathlib import Path
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_nominal_runner as runner


@pytest.mark.parametrize("status,mutation,expected", [
    ("nominal_stationarity", None, "completed"),
    ("refused", None, "completed"),
    ("refused", "resource_limit", "failed"),
    ("refused", "wrong_exit", "failed"),
    ("refused", "missing_identity", "failed"),
    ("refused", "wrong_pid", "failed"),
    ("refused", "wall_overrun", "failed"),
])
def test_guard_classifies_execution_and_numerical_status(
    monkeypatch, tmp_path, status, mutation, expected,
):
    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 240 and rss_bytes == 1024**3
        assert command[1:3] == ["-m", "examples.weather_scenarios.fv_point_nominal_probe"]
        child: dict[str, Any] = {
            "numerical_status": status, "response_validation": "not_performed",
            "source_unchanged": True, "input_unchanged": True,
            "plan_unchanged": True, "preflight_plan_unchanged": True,
            "preflight_unchanged": True,
            "pid": 123,
        }
        resource: dict[str, Any] = {
            "command": command, "child_pid": 123,
            "exit_code": 0 if status == "nominal_stationarity" else 2,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 240, "rss_limit_bytes": 1024**3,
            "elapsed_seconds": 10.0,
            "rss_samples": 2, "sampled_peak_rss_bytes": 200_000_000,
        }
        if mutation == "resource_limit":
            resource["resource_termination"] = "wall_limit"
        elif mutation == "wrong_exit":
            resource["exit_code"] = 0
        elif mutation == "missing_identity":
            child["source_unchanged"] = False
        elif mutation == "wrong_pid":
            child["pid"] = 456
        elif mutation == "wall_overrun":
            resource["elapsed_seconds"] = 240.01
        Path(command[command.index("--output") + 1]).write_text(json.dumps(child))
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource

    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / "attempt")
    assert result["execution_status"] == expected
    assert result["numerical_status"] == status
    assert result["response_validation"] == "not_performed"
