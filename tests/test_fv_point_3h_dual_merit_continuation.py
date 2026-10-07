from __future__ import annotations

import time
import json
from typing import Any

import pytest
import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation


def _problem(target: float = 0.0) -> tuple[Any, ...]:
    control = torch.zeros(26, dtype=torch.float64)
    control[0] = 0.15
    parameters = torch.zeros(13, dtype=torch.float64)
    hvp_points: list[Tensor] = []

    def objective(candidate, _parameters):
        shifted = candidate.clone()
        shifted[0] -= target
        return shifted @ shifted / 2

    def gradient(candidate, _parameters):
        result = candidate.clone()
        result[0] -= target
        return result

    def branch(candidate, _parameters):
        return ({"status": "passed_strict_branch", "euler_stages": 3600,
                 "choice_stage_count": 3600, "face_sign_stage_count": 3600,
                 "signature_sha256": "same"}, {"complete": True})

    def hvp(point, _parameters, vector):
        hvp_points.append(point.clone())
        return vector.clone()

    def preconditioner(vector):
        return vector.clone()

    return control, parameters, objective, gradient, branch, hvp, preconditioner, hvp_points


def test_repeats_fresh_pcg_directions_at_each_accepted_point(monkeypatch):
    inputs = _problem()
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10, max_iterations=2)

    assert record["execution_status"] == "completed"
    assert record["numerical_status"] == "iteration_limit"
    assert record["optimizer_steps_applied"] == 2
    assert len(record["iterations"]) == 2
    assert record["iterations"][0]["status"] == record["iterations"][1]["status"] == "accepted"
    assert record["hvp_calls"] == record["hvp_calls_completed"]
    assert record["pcg_iterations_completed"] == 2
    assert len(inputs[7]) > 2
    assert any(not torch.equal(point, inputs[0]) for point in inputs[7])
    assert inputs[0][0] == pytest.approx(0.15)
    assert (record["iterations"][0]["accepted_control_sha256"]
            == record["iterations"][1]["base_control_sha256"])


def test_stationarity_stop_is_pending_audit_and_never_eligible(monkeypatch):
    inputs = list(_problem())
    inputs[0][0] = 0.05
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10, max_iterations=3)

    assert record["numerical_status"] == "root_pending_audit"
    assert record["optimizer_steps_applied"] == 1
    assert record["root_pending_audit"] is True
    assert record["eligible_stationary_point"] is False
    assert record["full_root_claim"] is False


@pytest.mark.parametrize("constant", [0.0, 1.0, 0.063])
@pytest.mark.parametrize("initial_gradient", [1e-7, 1e-8])
def test_root_candidate_survives_cost_roundoff_floor(constant, initial_gradient, monkeypatch):
    control, parameters, _objective, _gradient, branch, hvp, preconditioner, _points = _problem()
    control.zero_()
    control[0] = initial_gradient

    def objective(candidate, _parameters):
        return torch.tensor(constant, dtype=torch.float64) + candidate @ candidate / 2

    def gradient(candidate, _parameters):
        return candidate.clone()

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    closure_calls = []
    record = continuation.run_iterations(control, parameters, objective, gradient, branch, hvp,
        preconditioner, time.monotonic() + 10, max_iterations=1,
        commit_candidate=lambda *_args: closure_calls.append("closed") or {"closed": True})

    assert record["numerical_status"] == "root_pending_audit"
    assert record["optimizer_steps_applied"] == 1
    assert record["iterations"][0]["accepted_gradient_inf"] == 0.0
    assert closure_calls == ["closed"]
    assert record["root_pending_audit"] is True
    assert record["eligible_stationary_point"] is False
    assert record["full_root_claim"] is False


def test_exact_root_can_pass_below_displacement_roundoff_floor(monkeypatch):
    control, parameters, _objective, _gradient, branch, _hvp, _preconditioner, _points = _problem()
    control.zero_()
    control[0] = 1e-22
    curvature = 1e15
    objective = lambda candidate, _parameters: torch.tensor(1.0, dtype=torch.float64) + curvature * (candidate @ candidate) / 2
    gradient = lambda candidate, _parameters: curvature * candidate.clone()
    hvp = lambda _point, _parameters, vector: curvature * vector.clone()
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(control, parameters, objective, gradient, branch, hvp,
        lambda vector: vector.clone(), time.monotonic() + 10, max_iterations=1,
        commit_candidate=lambda *_args: {"closed": True})

    iteration = record["iterations"][0]
    assert record["numerical_status"] == "root_pending_audit"
    assert iteration["displacement_l2"] < 128 * continuation.EPS
    assert iteration["displacement_below_roundoff"] is True
    assert iteration["accepted_gradient_inf"] <= continuation.ROOT_GRADIENT_INF
    assert record["eligible_stationary_point"] is False


def test_nonroot_below_displacement_floor_is_still_refused(monkeypatch):
    control, parameters, _objective, _gradient, branch, _hvp, _preconditioner, _points = _problem()
    control.zero_()
    control[0] = 4e-25
    curvature = 1e15
    objective = lambda candidate, _parameters: torch.tensor(1.0, dtype=torch.float64) + curvature * (candidate @ candidate) / 2
    gradient = lambda candidate, _parameters: curvature * candidate.clone()
    hvp = lambda _point, _parameters, vector: curvature * vector.clone()

    def branch_with_root_boundary(candidate, parameters):
        result, margins = branch(candidate, parameters)
        if candidate[0] < 1e-25:
            result = {**result, "status": "branch_boundary"}
        return result, margins

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    commits = []
    record = continuation.run_iterations(control, parameters, objective, gradient,
        branch_with_root_boundary, hvp, lambda vector: vector.clone(),
        time.monotonic() + 10, max_iterations=1,
        commit_candidate=lambda *_args: commits.append(True) or {"closed": True})

    assert record["numerical_status"] == "displacement_stagnation"
    assert record["optimizer_steps_applied"] == 0
    assert commits == []
    assert max(abs(value) for value in record["current_iteration"]["trials"][-1]["gradient"]) > continuation.ROOT_GRADIENT_INF


def test_nonroot_candidate_still_hits_numerically_zero_decrease_floor(monkeypatch):
    inputs = list(_problem())
    inputs[0].zero_()
    inputs[0][0] = 1e-7

    def objective(candidate, _parameters):
        return torch.tensor(1.0, dtype=torch.float64) + candidate @ candidate / 2

    def branch(candidate, parameters):
        result, margins = inputs[4](candidate, parameters)
        if candidate[0] <= 0:
            result = {**result, "status": "branch_boundary"}
        return result, margins

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], objective, inputs[3], branch,
        inputs[5], inputs[6], time.monotonic() + 10, max_iterations=1)

    assert record["numerical_status"] == "numerically_zero_decrease"
    assert record["optimizer_steps_applied"] == 0
    assert record["current_iteration"]["candidate_not_committed"] is True
    assert max(abs(value) for value in record["current_iteration"]["trials"][-1]["gradient"]) > continuation.ROOT_GRADIENT_INF
    assert record["eligible_stationary_point"] is False


@pytest.mark.parametrize("failure", ["gradient", "branch", "closure"])
def test_root_candidate_requires_re_evaluation_branch_and_closure(failure, monkeypatch):
    inputs = list(_problem())
    inputs[0].zero_()
    inputs[0][0] = 1e-7
    objective = lambda candidate, _parameters: candidate @ candidate / 2
    gradient = lambda candidate, _parameters: candidate.clone()
    branch = inputs[4]
    def commit(*_args) -> dict[str, bool]:
        if failure == "closure":
            raise RuntimeError("candidate integrity closure failed")
        return {"closed": True}
    if failure == "gradient":
        def gradient(candidate, _parameters):
            result = candidate.clone()
            if candidate[0] == 0:
                result[0] = 1e-5
            return result
    elif failure == "branch":
        def branch(candidate, parameters):
            result, margins = inputs[4](candidate, parameters)
            if not torch.equal(candidate, inputs[0]):
                result = {**result, "status": "branch_boundary"}
            return result, margins

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], objective, gradient, branch,
        inputs[5], inputs[6], time.monotonic() + 10, max_iterations=1,
        commit_candidate=commit)

    assert record.get("root_pending_audit") is not True
    assert record["eligible_stationary_point"] is False
    assert record["full_root_claim"] is False
    if failure == "gradient":
        assert record["optimizer_steps_applied"] == 1
        assert record["iterations"][0]["accepted_gradient_inf"] > continuation.ROOT_GRADIENT_INF
    else:
        assert record["optimizer_steps_applied"] == 0
    if failure == "closure":
        assert record["execution_status"] == "failed"
        assert record["current_iteration"]["trials"][-1]["status"] == "candidate_not_committed"
    elif failure == "branch":
        assert all(not row.get("strict_point_passed", False) or
                   max(abs(value) for value in row.get("gradient", [1.0])) > continuation.ROOT_GRADIENT_INF
                   for row in record["trials"])
    else:
        assert all(max(abs(value) for value in row.get("gradient", [1.0])) > continuation.ROOT_GRADIENT_INF
                   for row in record["trials"] if row.get("gradient") is not None)


@pytest.mark.parametrize("failure", ["objective", "gradient", "branch"])
def test_root_pending_requires_consistent_fresh_candidate_callbacks(failure, monkeypatch):
    inputs = list(_problem())
    inputs[0].zero_()
    inputs[0][0] = 1e-7
    objective_calls = 0
    gradient_calls = 0
    branch_calls = 0

    def objective(candidate, _parameters):
        nonlocal objective_calls
        objective_calls += 1
        value = candidate @ candidate / 2
        return value + (1e-3 if failure == "objective" and objective_calls >= 3 else 0.0)

    def gradient(candidate, _parameters):
        nonlocal gradient_calls
        gradient_calls += 1
        result = candidate.clone()
        if failure == "gradient" and gradient_calls >= 3:
            result[0] += 1e-5
        return result

    def branch(candidate, parameters):
        nonlocal branch_calls
        branch_calls += 1
        result, margins = inputs[4](candidate, parameters)
        if failure == "branch" and branch_calls >= 3:
            result = {**result, "signature_sha256": "changed"}
        return result, margins

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    commits = []
    record = continuation.run_iterations(inputs[0], inputs[1], objective, gradient, branch,
        inputs[5], inputs[6], time.monotonic() + 10, max_iterations=1,
        commit_candidate=lambda *_args: commits.append(True) or {"closed": True})

    assert record["numerical_status"] == "root_candidate_recheck_failed"
    assert record.get("root_pending_audit") is not True
    assert record["optimizer_steps_applied"] == 0
    assert commits == []
    assert record["eligible_stationary_point"] is False


def test_total_hvp_cap_keeps_prior_commit_and_records_failed_solve(monkeypatch):
    inputs = _problem()
    monkeypatch.setattr(continuation, "MAX_HVP", 3)
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10, max_iterations=3)

    assert record["numerical_status"] == "budget_refusal"
    assert record["optimizer_steps_applied"] == 1
    assert record["hvp_calls"] == 3
    assert record["hvp_calls_completed"] == 3
    assert record["current_iteration"]["pcg_iterations_failed_solve"] == "not_recorded"
    assert record["accepted_control"][0] == pytest.approx(0.10)


def test_nonpositive_curvature_refuses_without_j_only_fallback(monkeypatch):
    inputs = list(_problem())
    inputs[5] = lambda _point, _parameters, vector: -vector
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10)

    assert record["numerical_status"] in {"linear_solve_refusal", "curvature_refusal"}
    assert record["optimizer_steps_applied"] == 0
    assert record["trials"] == []
    assert record["hvp_calls"] > 0
    assert "fallback" not in record


@pytest.mark.parametrize("initial", [0.10, 0.15])
def test_later_commit_failure_discards_only_provisional_candidate(initial, monkeypatch):
    inputs = _problem()
    inputs[0][0] = initial
    calls = 0

    def commit(*_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("diagnostic closure failed")
        return {"closed": True}

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10,
        max_iterations=3, commit_candidate=commit)

    assert record["execution_status"] == "failed"
    assert record["optimizer_steps_applied"] == 1
    assert record["iterations"][0]["status"] == "accepted"
    assert record["current_iteration"]["trials"][-1]["status"] == "candidate_not_committed"
    assert record["accepted_control"][0] == pytest.approx(initial - 0.05)
    assert record["hvp_calls"] == record["hvp_calls_completed"]


def test_deadline_is_checked_after_slow_branch_callback(monkeypatch):
    inputs = list(_problem())
    normal_branch = inputs[4]

    def slow_branch(candidate, parameters):
        time.sleep(0.02)
        return normal_branch(candidate, parameters)

    inputs[4] = slow_branch
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 0.005)

    assert record["numerical_status"] == "budget_refusal"
    assert record["optimizer_steps_applied"] == 0
    assert "budget exhausted" in record["refusal"]


def test_parent_summary_ignores_nonobject_partial_child(tmp_path):
    child = tmp_path / "step.json"
    child.write_text("[]\n")
    parent = {"completed_iterations": 0, "hvp_calls": 0, "numerical_status": "not_reached"}

    continuation._copy_partial_child_counts(parent, child)

    assert parent == {"completed_iterations": 0, "hvp_calls": 0, "numerical_status": "not_reached"}


@pytest.mark.parametrize("changes,expected", [
    ({}, "completed"),
    ({"wall_limit_seconds": 300.0}, "failed"),
    ({"sampled_peak_rss_bytes": continuation.RSS_BYTES + 1}, "resource_limited"),
    ({"resource_termination": "wall_time_limit"}, "resource_limited"),
    ({"received_sigterm": True}, "failed"),
    ({"monitor_error": "sampling failed"}, "failed"),
])
def test_continuation_classifies_its_own_actual_resource_plan(changes, expected):
    path = continuation.EVIDENCE / "e0b_repeat_dual_20261006_attempt1/step.resource.json"
    resource = json.loads(path.read_text())
    historical = resource.copy()
    resource.update(changes)

    assert continuation.execution_status(resource) == expected
    assert historical["wall_limit_seconds"] == 780.0
    assert continuation.search.execution_status(historical) == "failed"


def test_linear_mode_policy_and_forcing_schedule_preserve_strict_default():
    assert continuation.policy_dict() == continuation.policy_dict("strict", 3)
    assert continuation.policy_dict() == continuation._policy()
    inexact = continuation.policy_dict("inexact", 1)
    assert inexact["max_iterations"] == 1
    assert inexact["linear_mode"] == "inexact"
    assert inexact["forcing_tolerance_min"] == continuation.PCG_RTOL
    assert inexact["forcing_tolerance_max"] == 1e-3
    assert continuation.forcing_tolerance(1.0, "strict") == continuation.PCG_RTOL
    assert continuation.forcing_tolerance(1.0, "inexact") == 1e-3
    assert continuation.forcing_tolerance(1e-12, "inexact") == continuation.PCG_RTOL
    assert continuation.forcing_tolerance(1e-2, "inexact") == 1e-3
    with pytest.raises(ValueError):
        continuation.forcing_tolerance(-1.0, "inexact")


def test_inexact_pcg_meets_declared_residual_and_reduces_work(monkeypatch):
    control = torch.zeros(26, dtype=torch.float64)
    control[0], control[1] = 0.15, 1.5e-7
    parameters = torch.zeros(13, dtype=torch.float64)
    diagonal = torch.ones(26, dtype=torch.float64)
    diagonal[1] = 1e6
    hessian = torch.diag(diagonal)
    objective = lambda c, _p: c @ hessian @ c / 2
    gradient = lambda c, _p: hessian @ c
    branch = _problem()[4]
    hvp_calls = []

    def hvp(_c, _p, vector):
        hvp_calls.append(True)
        return hessian @ vector

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    receipts = {}
    for mode in ("strict", "inexact"):
        counts = {"hvp_calls": 0, "hvp_calls_completed": 0,
                  "pcg_iterations_completed": 0, "pcg_solves_started": 0}
        receipts[mode] = continuation.run_iterations(control, parameters, objective, gradient,
            branch, hvp, lambda v: v.clone(), time.monotonic() + 10,
            max_iterations=1, linear_mode=mode, shared_counts=counts)

    strict = receipts["strict"]["iterations"][0]["solve"]
    inexact = receipts["inexact"]["iterations"][0]["solve"]
    assert receipts["strict"]["iterations"][0]["status"] == "accepted"
    assert receipts["inexact"]["iterations"][0]["status"] == "accepted"
    assert strict["rtol"] == continuation.PCG_RTOL
    assert inexact["rtol"] == 1e-3
    assert inexact["true_relative_residual"] <= inexact["rtol"]
    assert inexact["iterations"] < strict["iterations"]
    assert receipts["strict"]["accepted_control"][0] == pytest.approx(
        receipts["inexact"]["accepted_control"][0], abs=1e-10)
    assert receipts["strict"]["policy"] == continuation.policy_dict("strict", 1)
    assert receipts["inexact"]["policy"] == continuation.policy_dict("inexact", 1)


def test_independent_true_residual_gate_rejects_bad_hvp(monkeypatch):
    inputs = _problem()
    calls = 0

    def inconsistent_hvp(_point, _parameters, vector):
        nonlocal calls
        calls += 1
        return vector.clone() if calls <= 2 else vector.clone() + 1e-4

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3],
        inputs[4], inconsistent_hvp, inputs[6], time.monotonic() + 10,
        max_iterations=1)
    assert record["numerical_status"] == "linear_solve_refusal"
    assert record["optimizer_steps_applied"] == 0
    assert record["current_iteration"]["pcg_iterations_failed_solve"] == "not_recorded"
    assert "true-residual gate failed" in record["refusal"]


def test_shared_hvp_budget_and_absolute_deadline_are_not_reset_between_arms(monkeypatch):
    inputs = _problem()
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    counts = {"hvp_calls": 89, "hvp_calls_completed": 89,
              "pcg_iterations_completed": 0, "pcg_solves_started": 0}
    deadline = time.monotonic() + 10
    first = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3],
        inputs[4], inputs[5], inputs[6], deadline, max_iterations=1,
        shared_counts=counts)
    second = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3],
        inputs[4], inputs[5], inputs[6], deadline, max_iterations=1,
        shared_counts=counts)
    assert first["numerical_status"] == second["numerical_status"] == "budget_refusal"
    assert counts["hvp_calls"] == counts["hvp_calls_completed"] == 90
    assert first["hvp_calls"] == second["hvp_calls"] == 90

    expired = time.monotonic() - 1
    third = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3],
        inputs[4], inputs[5], inputs[6], expired, max_iterations=1,
        shared_counts=counts)
    assert third["numerical_status"] == "budget_refusal"
    assert third["hvp_calls"] == 90


def test_run_uses_passed_absolute_deadline_instead_of_starting_a_new_budget(tmp_path):
    output = tmp_path / "expired.json"
    counts = {"hvp_calls": 90, "hvp_calls_completed": 90,
              "pcg_iterations_completed": 0, "pcg_solves_started": 0}
    loader_calls = []
    def forbidden_loader(*_args):
        loader_calls.append(True)
        raise AssertionError("expired deadline must not load another arm")
    receipt = continuation.run(tmp_path / "unused-plan.json", "unused-sha", output,
        base_loader=forbidden_loader, max_iterations=1,
        linear_mode="inexact", shared_counts=counts,
        absolute_deadline=time.monotonic() - 1)
    assert receipt["numerical_status"] == "budget_refusal"
    assert receipt["policy"] == continuation.policy_dict("inexact", 1)
    assert receipt["hvp_calls"] == 90
    assert loader_calls == []
