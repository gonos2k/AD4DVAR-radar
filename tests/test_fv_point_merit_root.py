"""Gradient-merit sector policy and parent publication gates without FV Newton."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from advar.local_refinement import RefinementTrial
from examples.weather_scenarios import fv_point_merit_root_probe as probe
from examples.weather_scenarios import fv_point_merit_root_runner as runner
from examples.weather_scenarios import fv_point_response_preflight as preflight


def _trial(**changes):
    baseline = RefinementTrial(
        iteration=1, backtrack=0,
        current_objective=1.0, candidate_objective=1.1,
        current_gradient_norm=2.0, candidate_gradient_norm=1.0,
        current_gradient_max=1.0, candidate_gradient_max=1.2,
        current_branch={"choices": [-1], "face_signs": [-1]},
        candidate_branch={"choices": [1], "face_signs": [1]},
        step_scale=0.5, normalized_slope=-1.0, armijo_ratio=1.1,
    )
    return replace(baseline, **changes)


def test_changed_sector_uses_measured_merit_not_objective_or_max_gradient():
    records = []
    assert probe.merit_trial_acceptance(_trial(), records.append) == (
        True, "merit_switch_decrease")
    assert records[0]["old_merit"] == 2.0
    assert records[0]["new_merit"] == 0.5
    assert records[0]["candidate_objective"] > records[0]["current_objective"]
    assert records[0]["candidate_gradient_max"] > records[0]["current_gradient_max"]
    assert records[0]["current_signature_sha256"] != records[0]["candidate_signature_sha256"]


def test_merit_rejects_increase_roundoff_and_overflow():
    for trial in (
        _trial(candidate_gradient_norm=3.0),
        _trial(candidate_gradient_norm=2.0 - 1e-15),
        _trial(candidate_gradient_norm=1e200),
    ):
        assert probe.merit_trial_acceptance(trial) == (False, "merit_switch_refused")
    assert probe.merit_trial_acceptance(_trial(
        candidate_branch={"choices": [-1], "face_signs": [-1]})) is None


def test_pinned_seed_and_plan_without_newton():
    assert hashlib.sha256(probe.PLAN.read_bytes()).hexdigest() == probe.PLAN_SHA256
    assert hashlib.sha256(probe.PRIOR.read_bytes()).hexdigest() == probe.PRIOR_SHA256
    prior = json.loads(probe.PRIOR.read_text())
    accepted = [v for v in prior["trial_records"] if v["accepted"]]
    assert len(accepted) == 6 and accepted[-1]["iteration"] == 6
    seed = torch.tensor(accepted[-1]["candidate_control"], dtype=torch.float64)
    assert preflight._tensor_sha(seed) == probe.SEED_SHA256
    assert accepted[-1]["branch_signature_sha256"] == probe.SEED_SIGNATURE_SHA256


@pytest.mark.parametrize("status,mutation,expected", [
    ("merit_refused", None, "completed"),
    ("merit_refused", "wrong_exit", "failed"),
    ("merit_refused", "wrong_plan", "failed"),
    ("merit_root_margin_qualified", None, "completed"),
    ("merit_root_low_margin", None, "completed"),
    ("merit_root_margin_qualified", "missing_final_trace", "failed"),
    ("merit_root_margin_qualified", "bad_control_hash", "failed"),
])
def test_parent_separates_execution_refusal_and_final_root(
    monkeypatch, tmp_path, status, mutation, expected,
):
    problem, warm, p, direction = preflight.fixed_problem()
    identity = preflight._input_identity(problem, warm, p, direction)
    source = probe._sources()
    prior = json.loads(probe.PRIOR.read_text())
    seed_control = [v for v in prior["trial_records"] if v["accepted"]][-1]["candidate_control"]
    curvature = {"symmetry_relative": 0.0, "lambda_min": 1.0,
                 "lambda_max": 2.0, "lambda_ratio": 0.5, "hvp_columns": 26}
    root = status.startswith("merit_root_")
    high = status == "merit_root_margin_qualified"

    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 600 and rss_bytes == 1024**3
        child = {
            "pid": 123, "phase": "finished" if root else "merit_refinement",
            "numerical_status": status, "response_validation": "not_performed",
            "source_unchanged": True, "input_unchanged": True,
            "plan_unchanged": True, "prior_unchanged": True,
            "seed_control_unchanged": True,
            "source_before": source, "source_after": source,
            "input_before": identity, "input_after": identity,
            "plan_sha256": probe.PLAN_SHA256,
            "prior_report_sha256": probe.PRIOR_SHA256,
            "seed_control_sha256": probe.SEED_SHA256,
            "seed_control": seed_control,
            "seed_curvature": curvature,
            "seed_branch": {"signature_sha256": probe.SEED_SIGNATURE_SHA256,
                            "euler_stages": 54,
                            "minimum_scaled_slope_margin": 7e-4,
                            "minimum_scaled_face_flux_margin": 2e-6},
        }
        if root:
            control = [0.0] * 26
            digest = preflight._tensor_sha(torch.tensor(control, dtype=torch.float64))
            branch = {"signature_sha256": "e" * 64, "euler_stages": 54,
                      "minimum_scaled_slope_margin": 2e-4 if high else 5e-5,
                      "minimum_scaled_face_flux_margin": 3e-4}
            child.update(control=control, final_control_sha256=digest,
                         final_branch=branch, final_curvature=curvature,
                         branch_calls=[{"status": "core_branch_admitted",
                                        "control_sha256": digest, **branch}],
                         response_margin_qualified=high,
                         gradient_max=1e-12, gradient_norm=2e-12,
                         objective=0.1, score=0.01, refinement_iterations=2)
        else:
            child["refusal"] = "declared numerical refusal"
        if mutation == "wrong_plan":
            child["plan_sha256"] = "f" * 64
        elif mutation == "missing_final_trace":
            child["branch_calls"] = []
        elif mutation == "bad_control_hash":
            child["final_control_sha256"] = "f" * 64
        resource = {"command": command, "child_pid": 123,
                    "exit_code": 0 if root else 2,
                    "resource_termination": None, "monitor_error": None,
                    "wall_limit_seconds": 600, "elapsed_seconds": 10.0,
                    "rss_limit_bytes": 1024**3, "rss_samples": 3,
                    "sampled_peak_rss_bytes": 300_000_000}
        if mutation == "wrong_exit":
            resource["exit_code"] = 1
        Path(command[-1]).write_text(json.dumps(child))
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource

    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / status)
    assert result["execution_status"] == expected
    assert result["numerical_status"] == status
    assert result["response_validation"] == "not_performed"


def test_unexpected_final_score_error_propagates_as_execution_failure(monkeypatch, tmp_path):
    problem, warm, p, direction = preflight.fixed_problem()
    prior = json.loads(probe.PRIOR.read_text())
    # Isolate callback classification from the historical prior-source gate;
    # current refiner source has legitimately changed since that archive.
    monkeypatch.setattr(probe.sector, "_sources", lambda: prior["source_before"])
    row = [v for v in prior["trial_records"] if v["accepted"]][-1]
    seed = torch.tensor(row["candidate_control"], dtype=torch.float64)
    gradient = torch.func.grad(problem.objective, argnums=0)(seed, p)
    branch, scope = probe.sector.previous._core_branch(problem, seed, p)
    real_grad = torch.func.grad

    def fake_grad(function, *, argnums=0):
        if function == problem.objective:
            return lambda control, _p: gradient if torch.equal(control, seed) else torch.zeros_like(control)
        return real_grad(function, argnums=argnums)

    monkeypatch.setattr(probe.preflight, "fixed_problem", lambda: (problem, warm, p, direction))
    monkeypatch.setattr(probe.torch.func, "grad", fake_grad)
    monkeypatch.setattr(probe.basin, "_hessian_audit", lambda *_: {
        "symmetry_relative": 0.0, "lambda_min": 1.0,
        "lambda_max": 2.0, "lambda_ratio": 0.5, "hvp_columns": 26})
    monkeypatch.setattr(probe.sector.previous, "_core_branch", lambda *_: (branch, scope))
    monkeypatch.setattr(probe, "refine_stationary", lambda *_args, **_kwargs:
                        SimpleNamespace(control=seed + 1e-9, iterations=1, hvp_count=0))

    def broken_score(_self, _control, _parameters):
        raise ValueError("unexpected score sentinel")

    monkeypatch.setattr(type(problem), "score", broken_score)
    output = tmp_path / "unexpected_score.json"
    with pytest.raises(ValueError, match="unexpected score sentinel"):
        probe.run(output)
    saved = json.loads(output.read_text())
    assert saved["phase"] == "final_curvature"
    assert saved["numerical_status"] == "running"
