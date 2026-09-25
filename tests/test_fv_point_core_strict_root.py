"""Pinned root-only input and status checks without running FV Newton."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_core_strict_root_probe as probe
from examples.weather_scenarios import fv_point_core_strict_root_runner as runner
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios import fv_point_terminal_newton_probe as previous
from examples.weather_scenarios import fv_point_basin_probe as basin_probe


def test_locked_seed_and_prior_report_still_match_current_problem():
    assert hashlib.sha256(probe.PLAN.read_bytes()).hexdigest() == probe.PLAN_SHA256
    assert hashlib.sha256(probe.PRIOR_REPORT.read_bytes()).hexdigest() == probe.PRIOR_SHA256
    prior = json.loads(probe.PRIOR_REPORT.read_text())
    problem, _warm, parameters, _direction = preflight.fixed_problem()
    control = torch.tensor(prior["seed_control"], dtype=torch.float64)
    assert preflight._tensor_sha(control) == previous.CONTROL_SHA256
    branch, _ = probe._core_branch(problem, control, parameters)
    assert basin_probe._signature_sha(branch) == previous.BRANCH_SHA256


@pytest.mark.parametrize("stages,slope,face,accepted", [
    (54, 5e-5, 5e-5, True),
    (53, 5e-5, 5e-5, False),
    (54, 0.0, 5e-5, False),
    (54, float("nan"), 5e-5, False),
    (54, 5e-5, float("nan"), False),
])
def test_core_gate_is_distinct_from_final_response_margin(
    monkeypatch, stages, slope, face, accepted,
):
    def fake(_problem, _control, _parameters):
        return {"euler_stages": stages, "minimum_scaled_slope_margin": slope,
                "choices": [], "face_signs": []}, "core trace", face

    monkeypatch.setattr(probe.preflight, "_branch", fake)
    control = torch.zeros(1, dtype=torch.float64)
    if accepted:
        branch, _ = probe._core_branch(None, control, control)
        assert branch["minimum_scaled_face_flux_margin"] == 5e-5
        assert branch["minimum_scaled_slope_margin"] <= 1e-4
    else:
        with pytest.raises(ValueError, match="resolvability refused"):
            probe._core_branch(None, control, control)


@pytest.mark.parametrize("status,mutation,expected", [
    ("core_strict_root_margin_qualified", None, "completed"),
    ("core_strict_root_low_margin", None, "completed"),
    ("seed_curvature_refused", None, "completed"),
    ("root_refused", None, "completed"),
    ("root_refused", "final_curvature_phase", "completed"),
    ("root_refused", "wrong_exit", "failed"),
    ("root_refused", "wall_overrun", "failed"),
    ("core_strict_root_margin_qualified", "missing_final_curvature", "failed"),
    ("core_strict_root_margin_qualified", "wrong_plan_hash", "failed"),
    ("core_strict_root_margin_qualified", "wrong_preflight_plan_hash", "failed"),
    ("core_strict_root_low_margin", "false_high_margin_flag", "failed"),
])
def test_runner_separates_execution_root_and_response_margin(
    monkeypatch, tmp_path, status, mutation, expected,
):
    provenance = {
        "source": {"fixed.py": "a" * 64}, "input": {"problem": "b" * 64},
        "plan": probe.PLAN_SHA256, "prior": probe.PRIOR_SHA256,
        "preflight_plan": "c" * 64,
        "expected_preflight_plan": "c" * 64,
        "preflight": previous.PREFLIGHT_SHA256,
        "attempt2": previous.ATTEMPT2_SHA256,
    }
    monkeypatch.setattr(runner, "_current_provenance", lambda: provenance)

    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 600 and rss_bytes == 1024**3
        root_found = status.startswith("core_strict_root_")
        high_margin = status == "core_strict_root_margin_qualified"
        child: dict[str, Any] = {
            "pid": 123, "numerical_status": status,
            "response_validation": "not_performed",
            "phase": "finished" if root_found else (
                "seed_curvature" if status == "seed_curvature_refused" else "exact_refinement"
            ),
            "source_unchanged": True, "input_unchanged": True,
            "seed_control_unchanged": True, "plan_unchanged": True,
            "preflight_plan_unchanged": True,
            "prior_unchanged": True, "preflight_unchanged": True,
            "attempt2_unchanged": True,
            "plan_sha256": provenance["plan"],
            "preflight_plan_sha256": provenance["preflight_plan"],
            "prior_report_sha256": provenance["prior"],
            "preflight_sha256": provenance["preflight"],
            "attempt2_sha256": provenance["attempt2"],
            "source_before": provenance["source"], "source_after": provenance["source"],
            "input_before": provenance["input"], "input_after": provenance["input"],
            "seed_control_sha256": previous.CONTROL_SHA256,
        }
        if root_found:
            curvature = {"symmetry_relative": 0.0, "lambda_min": 1.0,
                         "lambda_max": 2.0, "lambda_ratio": 0.5,
                         "hvp_columns": 26}
            seed_branch = {"euler_stages": 54,
                           "minimum_scaled_slope_margin": 2e-4,
                           "minimum_scaled_face_flux_margin": 3e-4,
                           "signature_sha256": previous.BRANCH_SHA256}
            final_branch = {**seed_branch,
                            "minimum_scaled_slope_margin": 2e-4 if high_margin else 5e-5}
            child.update(control=[0.0] * 26, objective=0.1, score=0.01,
                         gradient_max=1e-12, gradient_norm=2e-12,
                         refinement_iterations=2, seed_branch=seed_branch,
                         final_branch=final_branch,
                         seed_curvature=curvature, final_curvature=curvature,
                         response_margin_qualified=high_margin)
        else:
            child["refusal"] = "declared scientific refusal"
        resource: dict[str, Any] = {
            "command": command, "child_pid": 123,
            "exit_code": 0 if root_found else 2,
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
        elif mutation == "wrong_plan_hash":
            child["plan_sha256"] = "d" * 64
        elif mutation == "wrong_preflight_plan_hash":
            child["preflight_plan_sha256"] = "d" * 64
        elif mutation == "false_high_margin_flag":
            child["response_margin_qualified"] = True
        elif mutation == "final_curvature_phase":
            child["phase"] = "final_curvature"
        Path(command[-1]).write_text(json.dumps(child))
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource

    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / "attempt")
    assert result["execution_status"] == expected
    assert result["numerical_status"] == status
    assert result["response_validation"] == "not_performed"
