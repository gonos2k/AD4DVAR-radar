"""Outer resource and numerical classifications for a bounded basin run."""

import json
from pathlib import Path
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_basin_runner as runner


@pytest.mark.parametrize("status,mutation,expected", [
    ("nominal_stationarity", None, "completed"),
    ("basin_incomplete", None, "completed"),
    ("curvature_refused", None, "completed"),
    ("refinement_refused", None, "completed"),
    ("basin_refused", None, "completed"),
    ("exploration_terminal_ineligible", None, "completed"),
    ("basin_incomplete", "over_time", "failed"),
    ("basin_incomplete", "over_rss", "failed"),
    ("basin_incomplete", "wrong_pid", "failed"),
    ("basin_incomplete", "wrong_exit", "failed"),
    ("basin_incomplete", "source_drift", "failed"),
    ("nominal_stationarity", "missing_gradient", "failed"),
    ("nominal_stationarity", "bad_curvature", "failed"),
    ("nominal_stationarity", "bad_final_margin", "failed"),
])
def test_guard_requires_matching_resource_child_and_numerical_status(
    monkeypatch, tmp_path, status, mutation, expected,
):
    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 1200 and rss_bytes == 1024**3
        assert command[1:3] == ["-m", "examples.weather_scenarios.fv_point_basin_probe"]
        child: dict[str, Any] = {
            "pid": 123, "numerical_status": status,
            "response_validation": "not_performed",
            "source_unchanged": True, "input_unchanged": True,
            "plan_unchanged": True, "preflight_plan_unchanged": True,
            "preflight_unchanged": True,
            "attempt1_unchanged": True,
            "phase": (
                "finished" if status == "nominal_stationarity" else
                "terminal_eligibility" if status == "exploration_terminal_ineligible" else
                "curvature" if status == "curvature_refused" else
                "exact_refinement" if status == "refinement_refused" else
                "basin_search"
            ),
            "basin_status": (
                "step_budget" if status == "basin_incomplete" else "candidate"
            ),
        }
        if status == "nominal_stationarity":
            curvature = {"symmetry_relative": 0.0, "lambda_min": 1.0,
                         "lambda_max": 2.0, "lambda_ratio": 0.5,
                         "hvp_columns": 26}
            child.update(control=[0.0] * 26, objective=0.1, score=0.01,
                         gradient_max=1e-12, gradient_norm=2e-12,
                         refinement_iterations=2,
                         basin_terminal_branch={
                             "euler_stages": 54,
                             "minimum_scaled_slope_margin": 2e-4,
                             "minimum_scaled_face_flux_margin": 3e-4,
                             "signature_sha256": "a" * 64,
                         },
                         final_branch_signature_sha256="a" * 64,
                         final_branch={
                             "euler_stages": 54,
                             "minimum_scaled_slope_margin": 2e-4,
                             "minimum_scaled_face_flux_margin": 3e-4,
                             "signature_sha256": "a" * 64,
                         },
                         curvature=curvature, final_curvature=curvature)
        elif status != "basin_incomplete":
            child["refusal"] = "declared numerical refusal"
        resource: dict[str, Any] = {
            "command": command, "child_pid": 123,
            "exit_code": 0 if status == "nominal_stationarity" else 2,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 1200, "elapsed_seconds": 10.0,
            "rss_limit_bytes": 1024**3, "rss_samples": 3,
            "sampled_peak_rss_bytes": 300_000_000,
        }
        if mutation == "over_time":
            resource["elapsed_seconds"] = 1200.01
        elif mutation == "over_rss":
            resource["sampled_peak_rss_bytes"] = 1024**3 + 1
        elif mutation == "wrong_pid":
            child["pid"] = 456
        elif mutation == "wrong_exit":
            resource["exit_code"] = 0
        elif mutation == "source_drift":
            child["source_unchanged"] = False
        elif mutation == "missing_gradient":
            child.pop("gradient_max")
        elif mutation == "bad_curvature":
            child["final_curvature"]["lambda_min"] = -1.0
        elif mutation == "bad_final_margin":
            child["final_branch"]["minimum_scaled_face_flux_margin"] = float("nan")
        Path(command[-1]).write_text(json.dumps(child))
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource

    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / "attempt")
    assert result["execution_status"] == expected
    assert result["numerical_status"] == status
    assert result["response_validation"] == "not_performed"
