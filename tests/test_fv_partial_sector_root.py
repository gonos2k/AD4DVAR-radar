"""Current-source two-hole root search: fixed inputs and fail-closed gates."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from advar.local_refinement import RefinementCallbackError, RefinementNumericalRefusal, RefinementTrial
from examples.weather_scenarios import fv_partial_sector_root_probe as probe
from examples.weather_scenarios import fv_partial_sector_root_runner as runner


def _trial(**changes):
    base = RefinementTrial(
        iteration=2, backtrack=0,
        current_objective=0.1, candidate_objective=0.11,
        current_gradient_norm=2.0, candidate_gradient_norm=1.0,
        current_gradient_max=1.0, candidate_gradient_max=1.1,
        current_branch={"choices": [-1], "face_signs": [-1]},
        candidate_branch={"choices": [1], "face_signs": [1]},
        step_scale=0.5, normalized_slope=-1.0, armijo_ratio=1.1,
    )
    return replace(base, **changes)


def test_current_source_preflight_preserves_frozen_tensors_but_versions_identity():
    assert probe._sha(probe.PLAN) == probe.PLAN_SHA256
    frozen = probe._preflight_identity()
    assert frozen["historical_problem_identity"]["fixed_problem_sha256"] == probe.HISTORICAL_PROBLEM_SHA256
    assert frozen["current_problem_identity"]["fixed_problem_sha256"] == probe.CURRENT_PROBLEM_SHA256
    assert probe.HISTORICAL_PROBLEM_SHA256 != probe.CURRENT_PROBLEM_SHA256
    assert frozen["fixed_input_fields"]["observation_counts"] == {
        "valid": 58, "missing": 2, "censored": 0, "qc_rejected": 0,
    }
    assert frozen["fixed_input_fields"]["warm_start_branch"]["euler_stages"] == 54
    assert len(probe._sources()) == len(probe.SOURCE_PATHS)


def test_changed_sector_gets_explicit_measured_merit_decision():
    records = []
    assert probe.merit_trial_acceptance(_trial(), records.append) == (
        True, "merit_switch_decrease")
    assert records[0]["accepted"] is True
    assert records[0]["candidate_objective"] > records[0]["current_objective"]
    assert records[0]["candidate_gradient_max"] > records[0]["current_gradient_max"]
    assert probe.merit_trial_acceptance(_trial(candidate_gradient_norm=3.0)) == (
        False, "merit_switch_refused")
    assert probe.merit_trial_acceptance(_trial(candidate_branch={
        "choices": [-1], "face_signs": [-1],
    })) is None


@pytest.mark.parametrize("status,mutation,expected", [
    ("root_margin_qualified", None, "completed"),
    ("root_margin_qualified", "valid_switch", "completed"),
    ("root_margin_qualified", "wrong_switch_merit", "failed"),
    ("root_low_margin", None, "completed"),
    ("root_refused", None, "completed"),
    ("seed_curvature_refused", None, "completed"),
    ("root_margin_qualified", "wrong_final_hash", "failed"),
    ("root_margin_qualified", "missing_final_trace", "failed"),
    ("root_margin_qualified", "missing_seed_trace", "failed"),
    ("root_margin_qualified", "wrong_seed_curvature_hash", "failed"),
    ("root_margin_qualified", "missing_trials", "failed"),
    ("root_margin_qualified", "wrong_accepted_hash", "failed"),
    ("root_margin_qualified", "wrong_true_residual", "failed"),
    ("root_margin_qualified", "wrong_pcg_iterations", "failed"),
    ("root_margin_qualified", "unproved_switch", "failed"),
    ("root_margin_qualified", "fake_armijo", "failed"),
    ("root_low_margin", "false_high_margin", "failed"),
    ("root_refused", "wrong_exit", "failed"),
    ("root_refused", "resource_limited", "failed"),
])
def test_parent_distinguishes_root_refusal_and_resource(
    monkeypatch, tmp_path, status, mutation, expected,
):
    source = {"fixed.py": "a" * 64}
    input_identity = {"current_problem_identity": {"fixed_problem_sha256": probe.CURRENT_PROBLEM_SHA256}}
    monkeypatch.setattr(probe, "_sources", lambda: source)
    monkeypatch.setattr(probe, "_preflight_identity", lambda: input_identity)
    control = [0.0] * 26
    control_sha = probe._tensor_sha(torch.zeros(26, dtype=torch.float64))
    curvature = {"hvp_columns": 26, "symmetry_relative": 0.0,
                 "lambda_min": 1.0, "lambda_max": 2.0, "lambda_ratio": 0.5}
    seed_branch = {"euler_stages": 54, "signature_sha256": "b" * 64,
                   "minimum_scaled_slope_margin": 2e-4,
                   "minimum_scaled_face_flux_margin": 3e-4}
    root = status in runner.ROOT_STATUSES
    high = status == "root_margin_qualified"

    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 600 and rss_bytes == 1024**3
        child: dict[str, Any] = {
            "pid": 123, "phase": "finished" if root else (
                "seed_curvature" if status == "seed_curvature_refused" else "sector_refinement"),
            "numerical_status": status, "response_validation": "not_performed",
            "physical_validation": "not_performed",
            "source_unchanged": True, "input_unchanged": True,
            "plan_unchanged": True, "warm_control_unchanged": True,
            "parameters_unchanged": True,
            "source_before": source, "source_after": source,
            "input_before": input_identity, "input_after": input_identity,
            "plan_sha256": probe.PLAN_SHA256,
            "gauss_newton": {"control": control, "control_sha256": control_sha},
            "seed_branch": seed_branch,
            "seed_gradient_norm": 1.0,
            "branch_calls": [{"status": "core_branch_admitted",
                              "control_sha256": control_sha, **seed_branch}],
        }
        if root:
            final = {**seed_branch,
                     "minimum_scaled_slope_margin": 2e-4 if high else 5e-5,
                     "signature_sha256": "b" * 64}
            child.update(
                control=control, final_control_sha256=control_sha,
                seed_curvature=curvature, final_curvature=curvature,
                seed_curvature_control_sha256=control_sha,
                final_curvature_control_sha256=control_sha,
                final_branch=final, final_gradient_max=1e-12,
                final_objective=0.01, final_score=0.02,
                response_margin_qualified=high,
                refinement_iterations=1,
                linear_solves=[{"phase": "sector_refinement", "converged": True,
                                "relative_residual": 1e-12, "iterations": 1}],
                trial_records=[{"iteration": 1, "accepted": True,
                                "backtrack": 0, "step_scale": 1.0,
                                "candidate_control": control,
                                "candidate_control_sha256": control_sha,
                                "branch": final, "linear_relative_residual": 1e-12,
                                "pcg_relative_residual": 1e-12,
                                "pcg_iterations": 1, "gradient_norm": 1e-12,
                                "armijo_ratio": 0.5}],
                policy_records=[],
            )
            child["branch_calls"].append({"status": "core_branch_admitted",
                                          "control_sha256": control_sha, **final})
        else:
            child["refusal"] = "declared numerical refusal"
        if mutation == "wrong_final_hash":
            child["final_control_sha256"] = "f" * 64
        elif mutation == "missing_final_trace":
            child["branch_calls"] = []
        elif mutation == "missing_seed_trace":
            child["branch_calls"].pop(0)
        elif mutation == "wrong_seed_curvature_hash":
            child["seed_curvature_control_sha256"] = "f" * 64
        elif mutation == "missing_trials":
            child["trial_records"] = []
        elif mutation == "wrong_accepted_hash":
            child["trial_records"][0]["candidate_control_sha256"] = "f" * 64
        elif mutation == "wrong_true_residual":
            child["trial_records"][0]["linear_relative_residual"] = 1e-3
        elif mutation == "wrong_pcg_iterations":
            child["trial_records"][0]["pcg_iterations"] = 2
        elif mutation in ("valid_switch", "wrong_switch_merit", "unproved_switch"):
            switched = "c" * 64
            child["final_branch"]["signature_sha256"] = switched
            child["branch_calls"][-1]["signature_sha256"] = switched
            child["trial_records"][0]["branch"]["signature_sha256"] = switched
            if mutation in ("valid_switch", "wrong_switch_merit"):
                trial = child["trial_records"][0]
                trial.update(acceptance_policy="trial_acceptance",
                             policy_reason="merit_switch_decrease")
                old_merit = 0.5
                new_merit = 0.5 * trial["gradient_norm"] ** 2
                floor = 128 * torch.finfo(torch.float64).eps * old_merit
                child["policy_records"] = [{
                    "iteration": 1, "backtrack": 0, "step_scale": 1.0,
                    "current_signature_sha256": "b" * 64,
                    "candidate_signature_sha256": switched,
                    "old_merit": old_merit, "new_merit": new_merit,
                    "delta": old_merit - new_merit, "floor": floor,
                    "accepted": True, "reason": "merit_switch_decrease",
                }]
                if mutation == "wrong_switch_merit":
                    child["policy_records"][0]["delta"] = 0.0
        elif mutation == "fake_armijo":
            child["trial_records"][0]["armijo_ratio"] = 1.1
        elif mutation == "false_high_margin":
            child["response_margin_qualified"] = True
        resource = {
            "command": command, "child_pid": 123,
            "exit_code": 0 if root else 2,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 600, "rss_limit_bytes": 1024**3,
            "rss_samples": 3, "sampled_peak_rss_bytes": 300_000_000,
            "elapsed_seconds": 10.0,
        }
        if mutation == "wrong_exit":
            resource["exit_code"] = 0
        elif mutation == "resource_limited":
            resource["resource_termination"] = "wall_time_limit"
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
    (RefinementNumericalRefusal, "root_refused"),
    (RefinementCallbackError, "running"),
])
def test_callback_failure_is_not_scientific_refusal(
    monkeypatch, tmp_path, error_type, expected_status,
):
    monkeypatch.setattr(probe.basin, "_hessian_audit", lambda *_: {
        "hvp_columns": 26, "symmetry_relative": 0.0,
        "lambda_min": 1.0, "lambda_max": 2.0, "lambda_ratio": 0.5,
    })
    def fake_gn(_observations, _contract, *, control):
        return SimpleNamespace(control=control.clone(), reason="test",
                               outer_iterations=0, pcg_iterations=0,
                               initial_objective=0.0, final_objective=0.0)
    monkeypatch.setattr(probe.variational, "solve_analysis", fake_gn)
    def fail(*_args, **_kwargs):
        raise error_type("injected sentinel")
    monkeypatch.setattr(probe, "refine_stationary", fail)
    output = tmp_path / "child.json"
    if error_type is RefinementCallbackError:
        with pytest.raises(RefinementCallbackError, match="injected sentinel"):
            probe.run(output)
    else:
        result = probe.run(output)
        assert result["numerical_status"] == "root_refused"
    saved = json.loads(output.read_text())
    assert saved["numerical_status"] == expected_status
    assert saved["response_validation"] == "not_performed"


@pytest.mark.parametrize("candidate_error,expected", [
    ("unexpected malformed observer", "execution_error"),
    ("minmod joint oracle left its strict smooth branch", "root_refused"),
])
def test_candidate_branch_value_error_has_explicit_classification(
    monkeypatch, tmp_path, candidate_error, expected,
):
    identity = {"current_problem_identity": {"fixed_problem_sha256": "test"}}
    monkeypatch.setattr(probe, "_preflight_identity", lambda: identity)
    monkeypatch.setattr(probe, "_sources", lambda: {"test.py": "a" * 64})
    problem = SimpleNamespace(
        identity=identity["current_problem_identity"],
        observations=object(), contract=lambda p: p,
        objective=lambda c, p: 0.5 * ((c - p) ** 2).sum(),
        score=lambda c, p: ((c - p) ** 2).sum(),
    )
    warm = torch.zeros(1, dtype=torch.float64)
    parameters = torch.ones(1, dtype=torch.float64)
    monkeypatch.setattr(probe, "_problem", lambda: (problem, warm, parameters))
    monkeypatch.setattr(probe.basin, "_hessian_audit", lambda *_: {
        "hvp_columns": 1, "symmetry_relative": 0.0,
        "lambda_min": 1.0, "lambda_max": 1.0, "lambda_ratio": 1.0,
    })
    monkeypatch.setattr(probe.variational, "solve_analysis", lambda *_, **__: SimpleNamespace(
        control=warm.clone(), reason="test", outer_iterations=0, pcg_iterations=0,
        initial_objective=0.5, final_objective=0.5,
    ))
    branch = {"euler_stages": 54, "minimum_scaled_slope_margin": 0.2,
              "choices": [0], "face_signs": [1]}
    def traced(_problem, control, _parameters):
        if bool(control.abs().max() > 0):
            raise ValueError(candidate_error)
        return branch, "fixed", 0.3
    monkeypatch.setattr(probe.preflight, "branch_with_face_margin", traced)
    output = tmp_path / "child.json"
    if expected == "execution_error":
        with pytest.raises(RefinementCallbackError, match="unexpected branch oracle ValueError"):
            probe.run(output)
        assert json.loads(output.read_text())["numerical_status"] == "running"
    else:
        assert probe.run(output)["numerical_status"] == expected


def test_nonfinite_candidate_is_recorded_without_invalid_json(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.basin, "_hessian_audit", lambda *_: {
        "hvp_columns": 26, "symmetry_relative": 0.0,
        "lambda_min": 1.0, "lambda_max": 2.0, "lambda_ratio": 0.5,
    })
    monkeypatch.setattr(probe.variational, "solve_analysis", lambda _, __, *, control: SimpleNamespace(
        control=control.clone(), reason="test", outer_iterations=0, pcg_iterations=0,
        initial_objective=0.0, final_objective=0.0,
    ))
    def nonfinite_trial(*_args, trial_observer, **_kwargs):
        trial_observer({"iteration": 1, "backtrack": 0, "accepted": False,
                        "rejection": "nonfinite_candidate",
                        "candidate_control": [float("inf")] * 26})
        raise RefinementNumericalRefusal("declared numerical refusal")
    monkeypatch.setattr(probe, "refine_stationary", nonfinite_trial)
    report = probe.run(tmp_path / "child.json")
    assert report["numerical_status"] == "root_refused"
    trial = json.loads((tmp_path / "child.json").read_text())["trial_records"][0]
    assert trial["candidate_nonfinite"] is True
    assert trial["candidate_control"] is None
    assert len(trial["candidate_control_sha256"]) == 64
