"""Pinned endpoint and guarded exact-Newton status checks without a FV solve."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_terminal_newton_probe as probe
from examples.weather_scenarios import fv_point_terminal_newton_runner as runner
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios import fv_point_basin_probe as basin_probe


def test_pinned_last_accepted_endpoint_matches_current_fixed_problem():
    assert hashlib.sha256(probe.ATTEMPT2.read_bytes()).hexdigest() == probe.ATTEMPT2_SHA256
    saved = json.loads(probe.ATTEMPT2.read_text())
    problem, _warm, parameters, _direction = preflight.fixed_problem()
    last = saved["accepted_events"][-1]
    control = torch.tensor(last["control"], dtype=torch.float64)
    assert saved["basin_status"] == "step_budget"
    assert saved["basin_accepted_steps"] == 100
    assert preflight._tensor_sha(control) == probe.CONTROL_SHA256
    branch, _ = basin_probe._strict_branch(problem, control, parameters)
    assert basin_probe._signature(branch) == json.loads(last["branch_key"])
    assert basin_probe._signature_sha(branch) == probe.BRANCH_SHA256
    probe._close("objective", float(problem.objective(control, parameters)),
                 last["objective"])
    gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    torch.testing.assert_close(gradient, torch.tensor(last["gradient"], dtype=torch.float64),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("status,mutation,expected", [
    ("nominal_stationarity", None, "completed"),
    ("seed_curvature_refused", None, "completed"),
    ("newton_refused", None, "completed"),
    ("seed_curvature_refused", "wrong_exit", "failed"),
    ("seed_curvature_refused", "wall_overrun", "failed"),
    ("nominal_stationarity", "missing_final_curvature", "failed"),
    ("nominal_stationarity", "branch_mismatch", "failed"),
    ("nominal_stationarity", "source_drift", "failed"),
    ("nominal_stationarity", "source_map_drift", "failed"),
    ("nominal_stationarity", "plan_hash_drift", "failed"),
])
def test_terminal_runner_requires_status_specific_evidence(
    monkeypatch, tmp_path, status, mutation, expected,
):
    provenance = {
        "source": {"fixed.py": "a" * 64},
        "input": {"problem": "b" * 64},
        "plan": probe.PLAN_SHA256,
        "preflight_plan": "c" * 64,
        "expected_preflight_plan": "c" * 64,
        "preflight": probe.PREFLIGHT_SHA256,
        "attempt2": probe.ATTEMPT2_SHA256,
    }
    monkeypatch.setattr(runner, "_current_provenance", lambda: provenance)

    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 600 and rss_bytes == 1024**3
        child: dict[str, Any] = {
            "pid": 123, "numerical_status": status,
            "response_validation": "not_performed",
            "phase": "finished" if status == "nominal_stationarity" else (
                "seed_curvature" if status == "seed_curvature_refused" else "exact_refinement"
            ),
            "source_unchanged": True, "input_unchanged": True,
            "seed_control_unchanged": True,
            "plan_unchanged": True, "preflight_plan_unchanged": True,
            "preflight_unchanged": True, "attempt2_unchanged": True,
            "seed_control_sha256": runner.EXPECTED_SEED_SHA256,
            "plan_sha256": provenance["plan"],
            "preflight_plan_sha256": provenance["preflight_plan"],
            "preflight_sha256": provenance["preflight"],
            "attempt2_sha256": provenance["attempt2"],
            "source_before": provenance["source"],
            "source_after": provenance["source"],
            "input_before": provenance["input"],
            "input_after": provenance["input"],
        }
        if status != "nominal_stationarity":
            child["refusal"] = "declared numerical refusal"
        else:
            curvature = {"symmetry_relative": 0.0, "lambda_min": 1.0,
                         "lambda_max": 2.0, "lambda_ratio": 0.5,
                         "hvp_columns": 26}
            branch = {"euler_stages": 54, "minimum_scaled_slope_margin": 2e-4,
                      "minimum_scaled_face_flux_margin": 3e-4,
                      "signature_sha256": runner.EXPECTED_BRANCH_SHA256}
            child.update(control=[0.0] * 26, objective=0.1, score=0.01,
                         gradient_max=1e-12, gradient_norm=2e-12,
                         refinement_iterations=2, seed_branch=branch,
                         final_branch=branch.copy(), seed_curvature=curvature,
                         final_curvature=curvature)
        resource: dict[str, Any] = {
            "command": command, "child_pid": 123,
            "exit_code": 0 if status == "nominal_stationarity" else 2,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 600, "elapsed_seconds": 10.0,
            "rss_limit_bytes": 1024**3, "rss_samples": 3,
            "sampled_peak_rss_bytes": 300_000_000,
        }
        if mutation == "wrong_exit":
            resource["exit_code"] = 0
        elif mutation == "wall_overrun":
            resource["elapsed_seconds"] = 600.01
        elif mutation == "missing_final_curvature":
            child.pop("final_curvature")
        elif mutation == "branch_mismatch":
            child["final_branch"]["signature_sha256"] = "b" * 64
        elif mutation == "source_drift":
            child["source_unchanged"] = False
        elif mutation == "source_map_drift":
            child["source_after"] = {"fixed.py": "d" * 64}
        elif mutation == "plan_hash_drift":
            child["plan_sha256"] = "d" * 64
        Path(command[-1]).write_text(json.dumps(child))
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource

    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / "attempt")
    assert result["execution_status"] == expected
    assert result["numerical_status"] == status
