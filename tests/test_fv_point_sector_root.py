"""Pinned sector-root input and outer publication gates without FV Newton."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_sector_root_probe as probe
from examples.weather_scenarios import fv_point_sector_root_runner as runner
from examples.weather_scenarios import fv_point_response_preflight as preflight
from advar.local_refinement import RefinementCallbackError, RefinementNumericalRefusal


def test_pinned_prior_control_branch_and_plan_are_current():
    assert hashlib.sha256(probe.PLAN.read_bytes()).hexdigest() == probe.PLAN_SHA256
    assert hashlib.sha256(probe.PRIOR.read_bytes()).hexdigest() == probe.PRIOR_SHA256
    prior = json.loads(probe.PRIOR.read_text())
    problem, _warm, parameters, _direction = preflight.fixed_problem()
    control = torch.tensor(prior["seed_control"], dtype=torch.float64)
    assert preflight._tensor_sha(control) == probe.previous.previous.CONTROL_SHA256
    branch, _ = probe.previous._core_branch(problem, control, parameters)
    assert branch["euler_stages"] == 54
    assert probe.basin_probe._signature_sha(branch) == probe.previous.previous.BRANCH_SHA256


@pytest.mark.parametrize("status,mutation,expected", [
    ("sector_root_margin_qualified", None, "completed"),
    ("sector_root_low_margin", None, "completed"),
    ("seed_curvature_refused", None, "completed"),
    ("sector_refused", None, "completed"),
    ("sector_refused", "wrong_exit", "failed"),
    ("sector_refused", "wall_overrun", "failed"),
    ("sector_root_margin_qualified", "missing_final_curvature", "failed"),
    ("sector_root_margin_qualified", "missing_final_trace", "failed"),
    ("sector_root_margin_qualified", "wrong_final_control_hash", "failed"),
    ("sector_root_margin_qualified", "wrong_plan_hash", "failed"),
    ("sector_root_low_margin", "false_high_margin", "failed"),
])
def test_runner_requires_status_resource_and_provenance(
    monkeypatch, tmp_path, status, mutation, expected,
):
    provenance = {
        "source": {"fixed.py": "a" * 64}, "input": {"problem": "b" * 64},
        "plan": probe.PLAN_SHA256, "prior": probe.PRIOR_SHA256,
        "previous_plan": "c" * 64, "expected_previous_plan": "c" * 64,
        "preflight_plan": "d" * 64, "expected_preflight_plan": "d" * 64,
        "preflight": probe.previous.previous.PREFLIGHT_SHA256,
        "attempt2": probe.previous.previous.ATTEMPT2_SHA256,
    }
    monkeypatch.setattr(runner, "_current_provenance", lambda: provenance)

    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 600 and rss_bytes == 1024**3
        root_found = status.startswith("sector_root_")
        high_margin = status == "sector_root_margin_qualified"
        child: dict[str, Any] = {
            "pid": 123, "numerical_status": status,
            "response_validation": "not_performed",
            "phase": "finished" if root_found else (
                "seed_curvature" if status == "seed_curvature_refused" else "sector_refinement"
            ),
            "source_unchanged": True, "input_unchanged": True,
            "seed_control_unchanged": True, "plan_unchanged": True,
            "prior_unchanged": True, "previous_plan_unchanged": True,
            "preflight_plan_unchanged": True,
            "preflight_unchanged": True, "attempt2_unchanged": True,
            "plan_sha256": provenance["plan"],
            "prior_report_sha256": provenance["prior"],
            "previous_plan_sha256": provenance["previous_plan"],
            "preflight_plan_sha256": provenance["preflight_plan"],
            "preflight_sha256": provenance["preflight"],
            "attempt2_sha256": provenance["attempt2"],
            "source_before": provenance["source"], "source_after": provenance["source"],
            "input_before": provenance["input"], "input_after": provenance["input"],
            "seed_control_sha256": probe.previous.previous.CONTROL_SHA256,
        }
        if root_found:
            curvature = {"symmetry_relative": 0.0, "lambda_min": 1.0,
                         "lambda_max": 2.0, "lambda_ratio": 0.5,
                         "hvp_columns": 26}
            seed_branch = {"euler_stages": 54,
                           "minimum_scaled_slope_margin": 2e-4,
                           "minimum_scaled_face_flux_margin": 3e-4,
                           "signature_sha256": probe.previous.previous.BRANCH_SHA256}
            final_branch = {"euler_stages": 54,
                            "minimum_scaled_slope_margin": 2e-4 if high_margin else 5e-5,
                            "minimum_scaled_face_flux_margin": 3e-4,
                            "signature_sha256": "e" * 64}
            child.update(control=[0.0] * 26, objective=0.1, score=0.01,
                         gradient_max=1e-12, gradient_norm=2e-12,
                         refinement_iterations=2, seed_branch=seed_branch,
                         final_branch=final_branch,
                         seed_curvature=curvature, final_curvature=curvature,
                         response_margin_qualified=high_margin,
                         final_control_sha256=preflight._tensor_sha(
                             torch.zeros(26, dtype=torch.float64)
                         ),
                         branch_calls=[{
                             "status": "core_branch_admitted",
                             "control_sha256": preflight._tensor_sha(
                                 torch.zeros(26, dtype=torch.float64)
                             ),
                             "signature_sha256": final_branch["signature_sha256"],
                             "euler_stages": 54,
                             "minimum_scaled_slope_margin": final_branch[
                                 "minimum_scaled_slope_margin"
                             ],
                             "minimum_scaled_face_flux_margin": final_branch[
                                 "minimum_scaled_face_flux_margin"
                             ],
                         }])
        else:
            child["refusal"] = "declared numerical refusal"
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
        elif mutation == "missing_final_trace":
            child["branch_calls"] = []
        elif mutation == "wrong_final_control_hash":
            child["final_control_sha256"] = "f" * 64
        elif mutation == "wrong_plan_hash":
            child["plan_sha256"] = "f" * 64
        elif mutation == "false_high_margin":
            child["response_margin_qualified"] = True
        Path(command[-1]).write_text(json.dumps(child))
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource

    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / "attempt")
    assert result["execution_status"] == expected
    assert result["numerical_status"] == status
    assert result["response_validation"] == "not_performed"


@pytest.mark.parametrize("error_type,expected_status", [
    (RefinementNumericalRefusal, "sector_refused"),
    (RefinementCallbackError, "running"),
])
def test_probe_separates_numerical_refusal_from_callback_failure(
    monkeypatch, tmp_path, error_type, expected_status,
):
    monkeypatch.setattr(probe.basin_probe, "_hessian_audit",
                        lambda *_: {"lambda_min": 1.0})

    def fail(*_args, **_kwargs):
        raise error_type("injected sentinel")

    monkeypatch.setattr(probe, "refine_stationary", fail)
    output = tmp_path / "result.json"
    if error_type is RefinementCallbackError:
        with pytest.raises(RefinementCallbackError, match="injected sentinel"):
            probe.run(output)
    else:
        result = probe.run(output)
        assert result["numerical_status"] == "sector_refused"
    saved = json.loads(output.read_text())
    assert saved["numerical_status"] == expected_status
    assert saved["response_validation"] == "not_performed"
