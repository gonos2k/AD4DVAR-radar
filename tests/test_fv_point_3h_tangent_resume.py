from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_tangent_resume as resume
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def test_resume_base_loader_uses_the_last_accepted_trial_and_final_repeat():
    base = resume._load_current_base({"initial_theta": resume.BASE_THETA})

    assert base["child_sha256"] == "77726dcdf533bb5b8949b97d7bebf39e28820c00240cdeb16401170a01f9ebd1"
    assert tangent._tensor_sha(base["control"]) == resume.BASE_CONTROL_SHA
    assert base["theta"] == resume.BASE_THETA
    assert base["accepted"]["theta"] == resume.BASE_THETA
    assert base["objective"] == 0.0612512070514904
    assert base["accepted"]["F_squared"] == 0.005238386295728529
    assert set(base["accepted"]["side_gradients"]) == {"-1", "1"}
    assert base["resume_efficiency"]["residual_norm"] > 0
    assert math.isfinite(base["resume_efficiency"]["cos_squared"])


def test_saved_commit_chain_rejects_base_theta_or_hvp_label_drift():
    previous = json.loads(resume.PREVIOUS_PLAN.read_text())
    raw = json.loads(gzip.decompress(resume.STEP_ARCHIVE.read_bytes()))
    expected = resume._validate_iteration_chain(raw["iterations"], raw["hvp_history"],
        previous["base_control_sha256"], previous["initial_theta"])
    assert expected[0] == resume.BASE_CONTROL_SHA
    assert expected[1] == resume.BASE_THETA

    changed_base = copy.deepcopy(raw)
    changed_base["iterations"][1]["base_control_sha256"] = resume.BASE_CONTROL_SHA
    with pytest.raises(ValueError, match="preceding commit"):
        resume._validate_iteration_chain(changed_base["iterations"], changed_base["hvp_history"],
            previous["base_control_sha256"], previous["initial_theta"])

    changed_hvp = copy.deepcopy(raw)
    changed_hvp["hvp_history"][4]["theta"] += 0.01
    with pytest.raises(ValueError, match="fresh labeled side HVPs"):
        resume._validate_iteration_chain(changed_hvp["iterations"], changed_hvp["hvp_history"],
            previous["base_control_sha256"], previous["initial_theta"])


def test_normalized_quadratic_prediction_matches_cos_squared_at_optimal_alpha():
    residual = torch.tensor([3.0, 4.0], dtype=torch.float64)
    derivative = torch.tensor([-4.0, 0.0], dtype=torch.float64)
    optimal_alpha = -float(torch.dot(residual, derivative)) / float(torch.dot(derivative, derivative))
    accepted = {"alpha": optimal_alpha, "F_squared": 16.0}
    iteration = {"residual": residual.tolist(), "residual_direction": derivative.tolist(),
        "tangent_gradient": [1.0] * 26, "mixed_gradient": [2.0] * 26}

    optimal = resume._resume_efficiency(iteration, accepted)
    assert optimal["normalized_predicted_reduction"] is not None
    assert optimal["cos_squared"] is not None
    assert optimal["normalized_actual_reduction"] is not None
    assert optimal["normalized_predicted_reduction"] == pytest.approx(optimal["cos_squared"])
    assert optimal["normalized_actual_reduction"] == pytest.approx(9.0 / 25.0)

    clipped = resume._resume_efficiency(iteration, {**accepted, "alpha": optimal_alpha / 2})
    assert clipped["normalized_predicted_reduction"] is not None
    assert clipped["cos_squared"] is not None
    assert clipped["normalized_predicted_reduction"] < clipped["cos_squared"]


def test_resume_cli_routes_through_shared_guarded_runner_with_injected_loaders(
    tmp_path: Path, monkeypatch,
):
    captured = {}

    def run(plan_path, plan_sha, output, resource, log, **kwargs):
        captured.update(plan_path=plan_path, plan_sha=plan_sha, output=output,
            resource=resource, log=log, **kwargs)
        return {"parent": {"execution_status": "not-launched"}, "child": None}

    monkeypatch.setattr(tangent, "run", run)
    plan, output, resource, log = (tmp_path / name for name in
        ("plan.json", "step.json", "step.resource.json", "step.log"))
    result = resume.run(plan, "resume-plan-digest", output, resource, log)

    assert result["parent"]["execution_status"] == "not-launched"
    assert captured["plan_loader"] is resume._load_plan
    assert captured["child_script"] == Path(resume.__file__)
    assert captured["base_control_sha256"] == resume.BASE_CONTROL_SHA


def test_parent_runner_rejects_committed_theta_drift(tmp_path: Path, monkeypatch):
    control = torch.linspace(-0.1, 0.2, 26, dtype=torch.float64)
    current_sha = tangent._tensor_sha(control)
    base_sha = "a" * 64
    resource_result = {"exit_code": 0, "resource_termination": None,
        "monitor_error": None, "received_sigterm": False,
        "wall_limit_seconds": tangent.POLICY["outer_seconds"],
        "rss_limit_bytes": tangent.POLICY["rss_bytes"],
        "elapsed_seconds": 1.0, "sampled_peak_rss_bytes": 1024}
    plan = {"policy": tangent.POLICY, "base_control_sha256": base_sha,
        "initial_theta": resume.BASE_THETA}

    def run_case(directory: Path, theta: float):
        output, resource, log = (directory / name for name in
            ("step.json", "step.resource.json", "step.log"))
        committed = {"accepted": True, "base_control_sha256": "b" * 64,
            "committed_control": control.tolist(), "committed_theta": resume.BASE_THETA}
        history = [{"status": "completed", "base_control_sha256": "b" * 64, "side": side}
            for side in (-1, 1)]
        child = {"phase": "finished", "execution_status": "completed",
            "plan_sha256": "test-plan", "base_control_sha256": base_sha,
            "source_unchanged": True, "fixed_input_unchanged": True,
            "runtime_unchanged": True, "deadline_passed": True,
            "runtime": {"python": "mock"}, "runtime_after": {"python": "mock"},
            "source_before": {}, "source_after": {}, "active_candidate_committed": False,
            "candidate_committed": True, "accepted_iterations": 1,
            "optimizer_steps_applied": 1, "hvp_calls_started": 2,
            "hvp_calls_completed": 2, "hvp_history": history,
            "iterations": [committed], "current_control": control.tolist(),
            "current_control_sha256": current_sha, "current_theta": theta}

        def guarded(command, **_kwargs):
            output_index = command.index("--output") + 1
            tangent._write(Path(command[output_index]), child)
            return resource_result

        monkeypatch.setattr(tangent, "run_guarded_diagnostic", guarded)
        return tangent.run(tmp_path / "resume-plan.json", "test-plan", output, resource, log,
            plan_loader=lambda *_: plan, child_script=Path(resume.__file__),
            base_control_sha256=base_sha)

    correct = run_case(tmp_path / "correct", resume.BASE_THETA)
    assert correct["parent"]["execution_status"] == "completed"
    drift = run_case(tmp_path / "drift", resume.BASE_THETA + 0.01)
    assert drift["parent"]["execution_status"] == "failed"
    assert drift["parent"]["execution_failure_reason"] == "child closure refused"
