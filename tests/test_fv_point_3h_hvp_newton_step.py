"""Focused live-operator tests; no FV trajectory is executed."""
import pytest
import torch
import json
import sys
from types import SimpleNamespace

from examples.weather_scenarios import fv_point_3h_hvp_newton_step as probe


def test_newton_pcg_uses_live_operator_not_frozen_preconditioner():
    diagonal = torch.linspace(1.5, 3.5, 26, dtype=torch.float64)
    gradient = torch.linspace(-0.8, 0.9, 26, dtype=torch.float64)
    live = lambda value: diagonal * value
    frozen_preconditioner = lambda value: value / 2.0
    step, audit = probe.block_step.solve_newton_direction(live, gradient, frozen_preconditioner)
    torch.testing.assert_close(live(step), -gradient, rtol=5e-10, atol=1e-12)
    assert audit["true_relative_residual"] <= 1e-10
    assert not torch.allclose(frozen_preconditioner(gradient), live(gradient))


def test_negative_live_curvature_refuses_before_search_candidate():
    gradient = torch.ones(26, dtype=torch.float64)
    candidate_calls = []
    with pytest.raises(probe.block_step.StepRefusal, match="operator must be symmetric positive definite"):
        step, _ = probe.block_step.solve_newton_direction(
            lambda value: -value, gradient, lambda value: value)
        candidate_calls.append(step)
    assert candidate_calls == []


def test_live_hvp_cap_refuses_without_candidate():
    gradient = torch.ones(26, dtype=torch.float64)
    deadline = torch.inf
    counted, counts = probe.counted_hvp(lambda value: value, deadline, limit=1)
    candidate_calls = []
    with pytest.raises(probe.search.BudgetRefusal, match="live HVP cap 1 reached"):
        step, _ = probe.block_step.solve_newton_direction(
            counted, gradient, lambda value: value)
        candidate_calls.append(step)
    assert counts == {"started": 1, "completed": 1}
    assert candidate_calls == []


def test_nonobject_child_cannot_leave_completed_parent(tmp_path, monkeypatch):
    directory = tmp_path / "attempt"
    monkeypatch.setattr(probe, "load_base", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["step", "--plan-sha256", "unused", "--directory", str(directory)])

    def fake_guard(*args, **kwargs):
        (directory / "step.json").write_text("[]")
        return {"exit_code": 0, "elapsed_seconds": 1., "wall_limit_seconds": 300.,
                "rss_limit_bytes": 1024**3, "sampled_peak_rss_bytes": 0,
                "resource_termination": None, "monitor_error": None,
                "child_process_group_cleanup_sent": False, "child_process_group_cleanup_error": None}

    monkeypatch.setattr(probe, "run_guarded_diagnostic", fake_guard)
    probe.main()
    parent = json.loads((directory / "step.run.json").read_text())
    assert parent["execution_status"] == "failed"
    assert parent["child_read_error"]


def _toy_run_setup(tmp_path, monkeypatch):
    base = json.loads(probe.BASE.read_text())
    old = json.loads(probe.CURVATURE.read_text())
    trial = probe._accepted_trial(base)
    p = torch.tensor(base["parameters"], dtype=torch.float64)
    g = torch.tensor(trial["gradient"], dtype=torch.float64)
    identity = base["input_after"]
    problem = SimpleNamespace(objective=lambda x, pp: x.square().sum()/2)
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    plan = tmp_path/"plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(probe, "load_base", lambda *args:
        ({"source_files": {}, "archive_files": {}, "policy": {}}, base, old))
    monkeypatch.setattr(probe.curvature.seed, "_prepare_fixed_seed", lambda:
        (problem, object(), None, p, object(), {}))
    monkeypatch.setattr(probe.curvature.seed, "_input_identity", lambda *args: identity)
    monkeypatch.setattr(probe.curvature.tail, "_full_current_branch", lambda *args:
        (trial["branch"], trial["margins"]))
    monkeypatch.setattr(probe.curvature.tail, "_fresh_merit", lambda *args:
        (torch.tensor(trial["objective"], dtype=torch.float64), g,
         torch.tensor(trial["phi"], dtype=torch.float64)))
    return plan, tmp_path/"step.json", g, problem


@pytest.mark.parametrize("message,unexpected,inside_operator", [
    ("unexpected operator failure", True, False), ("PCG direction update is not finite", False, False),
    ("unexpected operator failure", True, True)])
def test_solver_error_preserves_counts_and_error_class(tmp_path, monkeypatch, message, unexpected, inside_operator):
    plan, output, _, _ = _toy_run_setup(tmp_path, monkeypatch)

    def broken(operator, gradient, preconditioner, deadline):
        operator(torch.ones_like(gradient))
        raise RuntimeError(message)

    if inside_operator:
        def failed_jvp(*args):
            raise RuntimeError(message)
        monkeypatch.setattr(probe.torch.func, "jvp", failed_jvp)

    monkeypatch.setattr(probe.block_step, "solve_newton_direction", broken)
    if unexpected:
        with pytest.raises(RuntimeError, match=message):
            probe.run(plan, probe.curvature.sha(plan), output)
    else:
        probe.run(plan, probe.curvature.sha(plan), output)
    record = json.loads(output.read_text())
    assert record["hvp_calls"] == 1
    assert record["hvp_calls_completed"] == (0 if inside_operator else 1)
    assert record["optimizer_steps_applied"] == 0
    assert record["numerical_status"] == ("programming_error" if unexpected else "step_refusal")
    if not inside_operator:
        assert record["last_live_product"]["product"] == [1.]*26


@pytest.mark.parametrize("expiry", ["model", "integrity", "model_error"])
def test_post_candidate_deadline_prevents_commit(tmp_path, monkeypatch, expiry):
    plan, output, g, problem = _toy_run_setup(tmp_path, monkeypatch)
    base = json.loads(probe.BASE.read_text())
    identity = base["input_after"]
    clock = {"now": 0.}
    monkeypatch.setattr(probe.search.time, "monotonic", lambda: clock["now"])
    signature = {"choices": [0]*3600, "face_signs": [1]*3600}
    problem.branch_check = lambda *args: (signature, "toy")
    problem.frozen = SimpleNamespace(fv_transport=SimpleNamespace(substeps_per_interval=90))
    problem.layout = {"observation_times_seconds": (0., 600., 1200.)}
    partition = probe.curvature._branch_partition(signature, (0., 600., 1200.), 90)
    monkeypatch.setattr(probe.block_step, "solve_newton_direction", lambda *args:
        (-g, {"true_residual": [0.]*26, "iterations": 1, "true_relative_residual": 0.}))

    def candidate(*args, **kwargs):
        c = args[0]-.01*g
        row = {"status": "accepted", "alpha": .01, "objective": .06, "gradient": (.9*g).tolist(),
               "branch": {"signature_sha256": partition["full_signature_sha256"]}}
        kwargs["on_trial"](row)
        return c, [row], "accepted"

    monkeypatch.setattr(probe.search, "bounded_original_j_search", candidate)
    monkeypatch.setattr(probe.diagnostics, "physical_state", lambda *args: {"toy": torch.tensor(1.)})
    original_model = probe.diagnostics.model_diagnostics

    def model(*args, **kwargs):
        result = original_model(*args, **kwargs)
        if expiry == "model_error":
            raise RuntimeError("unexpected diagnostic failure")
        if expiry == "model":
            clock["now"] = 241.
        return result

    def input_identity(problem, original, c, *args):
        control_sha = probe.curvature.tensor_sha(c)
        if expiry == "integrity" and control_sha != base["accepted_control_sha256"]:
            clock["now"] = 241.
        return {**identity, "control_sha256": control_sha}

    monkeypatch.setattr(probe.diagnostics, "model_diagnostics", model)
    monkeypatch.setattr(probe.curvature.seed, "_input_identity", input_identity)
    if expiry == "model_error":
        with pytest.raises(RuntimeError, match="unexpected diagnostic failure"):
            probe.run(plan, probe.curvature.sha(plan), output)
        result = json.loads(output.read_text())
    else:
        result = probe.run(plan, probe.curvature.sha(plan), output)
    assert result["numerical_status"] == ("programming_error" if expiry == "model_error" else "budget_refusal")
    assert result["optimizer_steps_applied"] == 0
    assert result["trials"][-1]["status"] == "candidate_not_committed"
    assert "accepted_control" not in result
