"""Cached linear algebra and search policy checks; no FV trajectory execution."""
import json
import time

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as probe


def strict_branch(control, parameters):
    saved = json.loads(probe.BASE_AUDIT.read_text())
    return {**saved["branch"], "signature_sha256": "new_branch"}, saved["margins"]


def search(objective, gradient_fn, *, branch_fn=strict_branch, deadline=None):
    c = torch.zeros(26, dtype=torch.float64)
    p = torch.zeros(13, dtype=torch.float64)
    g = torch.zeros_like(c)
    g[0] = 1
    return probe.bounded_original_j_search(
        c, p, objective(c, p), g, -g, objective, gradient_fn, branch_fn,
        "old_branch", deadline=deadline)


def test_cached_direction_retains_nonzero_field_and_cross_blocks():
    saved = json.loads(probe.BASE_AUDIT.read_text())
    h = torch.tensor(saved["fresh_hessian"]["hessian"], dtype=torch.float64)
    g = torch.tensor(saved["gradient"], dtype=torch.float64)
    s, record = probe.exact_shifted_direction(h, g)
    m = h.clone()
    m[20:, 20:] += 40 * torch.eye(6, dtype=torch.float64)
    expected = torch.linalg.solve(m, -g)
    torch.testing.assert_close(s, expected, rtol=1e-10, atol=1e-12)
    assert record["modified_relative_residual"] < 1e-10
    assert float(g @ s) < 0 and float(g @ (h @ s)) < 0
    assert s.norm().item() == pytest.approx(.461223701531, rel=1e-10)
    # Fixing the field would discard the eliminated nonzero gradient correction.
    wrong = torch.linalg.solve(m[20:, 20:], -g[20:])
    assert (s[20:] - wrong).norm() > .01


def test_all_plan_sources_are_audited_before_and_after():
    archive = json.loads(probe.ARCHIVE.read_text())
    plan = json.loads(probe.PLAN.read_text())
    paths = probe._source_paths(probe.PLAN, archive)
    assert set(plan["source_files"]) <= set(paths)
    assert {"examples/weather_scenarios/fv_diagnostic_guard.py",
            "examples/weather_scenarios/fv_point_3h_endpoint_diagnostics.py"} <= probe._required_source_files(archive)


def test_block_curvature_refusal_uses_driver_numerical_refusal():
    with pytest.raises(probe.StepRefusal, match="block qualification refused"):
        probe.exact_shifted_direction(-torch.eye(26, dtype=torch.float64),
                                      torch.ones(26, dtype=torch.float64))


def test_unexpected_block_error_is_not_hidden_as_numerical_refusal(monkeypatch):
    def broken(matrix):
        raise RuntimeError("unexpected program failure")

    monkeypatch.setattr(probe.block_step, "block_inverse_preconditioner", broken)
    with pytest.raises(RuntimeError, match="unexpected program failure"):
        probe.exact_shifted_direction(torch.eye(26, dtype=torch.float64),
                                      torch.ones(26, dtype=torch.float64))


def test_initial_block_refusal_leaves_numerical_record(tmp_path, monkeypatch):
    def refused(**kwargs):
        probe.exact_shifted_direction(-torch.eye(26, dtype=torch.float64),
                                      torch.ones(26, dtype=torch.float64))

    monkeypatch.setattr(probe, "run", refused)
    monkeypatch.setattr(probe.sys, "argv", ["step", "--child", "--plan-sha256", "unused",
                                            "--output-directory", str(tmp_path)])
    probe.main()
    report = json.loads((tmp_path / "step.json").read_text())
    assert report["numerical_status"] == "initial_step_refusal"
    assert report["optimizer_steps_applied"] == 0
    assert report["source_unchanged"] is None
    assert report["integrity_status"] == "not_verified"


def test_setup_deadline_stops_before_another_operation(tmp_path, monkeypatch):
    clock = iter([0., 241.])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(probe, "_load_cached", lambda *args: ({}, {}, {}, {}, torch.empty(26, 26)))

    def forbidden(*args):
        raise AssertionError("expired setup cannot continue to source/direction work")

    monkeypatch.setattr(probe, "_source_paths", forbidden)
    with pytest.raises(probe.BudgetRefusal, match="budget exhausted during setup"):
        probe.run(plan_path=tmp_path / "plan.json", plan_sha256="unused", output=tmp_path / "step.json")


def test_phi_ascent_does_not_override_original_j_armijo():
    def objective(c, p):
        return 1 + c[0] - c[0] ** 2 / 2

    accepted, trials, status = search(objective, torch.func.grad(objective))
    assert status == "one_original_J_step_accepted" and accepted is not None
    assert len(trials) == 1 and accepted.norm() == pytest.approx(.05)
    assert trials[0]["objective"] < 1
    assert trials[0]["phi"] > .5
    assert trials[0]["branch_signature_changed"] is True


def test_decrease_does_not_bypass_candidate_branch_gate():
    objective = lambda c, p: 1 + c[0]

    def refused(c, p):
        branch, margins = strict_branch(c, p)
        return {**branch, "status": "branch_refused"}, margins

    accepted, trials, status = search(objective, torch.func.grad(objective), branch_fn=refused)
    assert accepted is None and status == "original_J_armijo_grid_exhausted"
    assert len(trials) == 16
    assert all(t["armijo_passed"] and not t["strict_point_passed"] for t in trials)
    assert trials[-1]["alpha"] == .05 * 2 ** -15


def test_armijo_uses_actual_objective_even_with_small_step():
    objective = lambda c, p: 1 - c[0]
    accepted, trials, status = search(objective, torch.func.grad(objective))
    assert accepted is None and len(trials) == 16
    assert all(not t["armijo_passed"] for t in trials)


def test_deadline_before_evaluation_keeps_unaccepted_trial():
    def forbidden(c, p):
        raise AssertionError("expired budget must not call objective")

    c = torch.zeros(26, dtype=torch.float64)
    s = torch.ones_like(c)
    accepted, trials, status = probe.bounded_original_j_search(
        c, torch.zeros(13, dtype=torch.float64), torch.tensor(1., dtype=torch.float64),
        -s, s, forbidden, forbidden, strict_branch, "old", deadline=time.monotonic() - 1)
    assert accepted is None and status == "step_budget_refusal"
    assert trials[0]["status"] == "budget_refused_before_evaluation"


def test_deadline_after_evaluation_does_not_accept(monkeypatch):
    times = iter([0., 2.])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(times))
    objective = lambda c, p: 1 + c[0]
    accepted, trials, status = search(objective, torch.func.grad(objective), deadline=1.)
    assert accepted is None and status == "step_budget_refusal"
    assert trials[0]["armijo_passed"]
    assert trials[0]["status"] == "budget_refused_after_evaluation"


def test_nonfinite_candidate_cannot_be_accepted():
    calls = 0

    def objective(c, p):
        nonlocal calls
        calls += 1
        return torch.tensor(1. if calls == 1 else float("nan"), dtype=torch.float64)

    accepted, trials, _ = search(objective, lambda c, p: torch.ones_like(c))
    assert accepted is None and len(trials) == 16
    assert all(t["status"] == "nonfinite_objective" for t in trials)


@pytest.mark.parametrize("change,status", [
    ({}, "completed"), ({"exit_code": 1}, "failed"),
    ({"elapsed_seconds": 301.}, "resource_limited"),
    ({"elapsed_seconds": float("nan")}, "failed"),
    ({"resource_termination": "wall_time_limit"}, "resource_limited"),
    ({"child_process_group_cleanup_sent": True}, "failed"),
    ({"child_process_group_cleanup_error": "failure"}, "failed"),
    ({"received_sigterm": True}, "failed"),
    ({"resource_termination": "cancelled"}, "failed"),
    ({"resource_termination": "rss_monitor_unavailable"}, "failed"),
    ({"resource_termination": "unknown"}, "failed"),
    ({"monitor_error": "sample failure", "resource_termination": "rss_limit"}, "failed"),
])
def test_execution_resource_outcome_is_separate(change, status):
    resource = {"exit_code": 0, "elapsed_seconds": 1., "resource_termination": None,
                "monitor_error": None, "child_process_group_cleanup_sent": False,
                "child_process_group_cleanup_error": None}
    assert probe.execution_status({**resource, **change}) == status


@pytest.mark.parametrize("child", [None, "[]", "{broken", '{"phase":"running"}'])
def test_parent_keeps_failure_for_missing_or_invalid_child(tmp_path, monkeypatch, child):
    directory = tmp_path / "run"
    monkeypatch.setattr(probe, "_load_cached", lambda *args: None)
    monkeypatch.setattr(probe.sys, "argv", ["step", "--plan-sha256", "unused",
                                            "--output-directory", str(directory)])

    def fake_guard(*args, **kwargs):
        if child is not None:
            (directory / "step.json").write_text(child)
        return {"exit_code": 0, "elapsed_seconds": 1., "resource_termination": None,
                "monitor_error": None, "child_process_group_cleanup_sent": False,
                "child_process_group_cleanup_error": None}

    monkeypatch.setattr(probe, "run_guarded_diagnostic", fake_guard)
    probe.main()
    parent = json.loads((directory / "step.run.json").read_text())
    assert parent["execution_status"] == "failed"
    assert parent["resource"]["exit_code"] == 0
    if child == '{"phase":"running"}':
        assert parent["execution_failure_reason"] == "child_completion_or_integrity_not_verified"
