from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing


def test_minimum_mixture_weight_is_differentiable_and_requires_resolved_interior_value():
    dtype = torch.float64
    gm = torch.tensor([-0.2, 1.0], dtype=dtype, requires_grad=True)
    gp = torch.tensor([0.2, 1.0], dtype=dtype, requires_grad=True)
    theta = tangent.minimum_mixture_weight(gm, gp)
    mixed = gm + theta * (gp - gm)
    torch.testing.assert_close(theta, torch.tensor(0.5, dtype=dtype))
    torch.testing.assert_close(torch.dot(mixed, gp - gm), torch.zeros((), dtype=dtype), atol=1e-15, rtol=0)
    theta_grads = torch.autograd.grad(theta, (gm, gp))
    assert all(torch.isfinite(gradient).all() for gradient in theta_grads)

    with pytest.raises(ValueError, match="unresolved"):
        tangent.minimum_mixture_weight(torch.tensor([0.0, 1.0], dtype=dtype),
            torch.tensor([1e-16, 1.0], dtype=dtype))
    with pytest.raises(ValueError, match="strictly inside"):
        tangent.minimum_mixture_weight(torch.tensor([0.0, 1.0], dtype=dtype),
            torch.tensor([1.0, 1.0], dtype=dtype))
    with pytest.raises(ValueError, match="strictly inside"):
        tangent.minimum_mixture_weight(torch.tensor([-1.0, 1.0], dtype=dtype),
            torch.tensor([0.0, 1.0], dtype=dtype))


def _two_branch_objectives(point: torch.Tensor):
    x, y = point.unbind()
    h = y + 0.5 * y.square()
    return h + (-0.2 + 0.4 * y) * x, h + (0.2 + 0.6 * y) * x


def test_full_theta_prime_residual_derivative_matches_autograd_jvp_and_alpha_cap():
    dtype = torch.float64
    point = torch.zeros(2, dtype=dtype, requires_grad=True)
    normal = torch.tensor([1.0, 0.0], dtype=dtype)
    chart = torch.tensor([[0.0], [1.0]], dtype=dtype)
    gm_fn = lambda value: torch.autograd.grad(_two_branch_objectives(value)[0], value,
        create_graph=True)[0]
    gp_fn = lambda value: torch.autograd.grad(_two_branch_objectives(value)[1], value,
        create_graph=True)[0]
    gm_graph, gp_graph = gm_fn(point), gp_fn(point)
    theta = tangent.minimum_mixture_weight(gm_graph, gp_graph)
    gm, gp = gm_graph.detach(), gp_graph.detach()
    direction = torch.tensor([0.0, -1.0], dtype=dtype)
    hm = torch.tensor([-0.4, -1.0], dtype=dtype)
    hp = torch.tensor([-0.6, -1.0], dtype=dtype)
    model = mixing.minimum_tangent_model(gm, gp, hm, hp, normal, chart,
        float(theta.detach()), 0.0, pivot=0, face_scale=0.84)
    assert model.delta_theta == pytest.approx(1.25)
    expected_direction = torch.tensor([0.0, -1.0, 0.0], dtype=dtype)

    def envelope_residual(value):
        minus, plus = _two_branch_objectives(value)
        gminus = torch.autograd.grad(minus, value, create_graph=True)[0]
        gplus = torch.autograd.grad(plus, value, create_graph=True)[0]
        theta_value = tangent.minimum_mixture_weight(gminus, gplus)
        gstar = (1.0 - theta_value) * gminus + theta_value * gplus
        return torch.cat((gstar, value[0:1] / 0.84))

    _, observed_direction = torch.autograd.functional.jvp(
        envelope_residual, point, direction, create_graph=False)
    torch.testing.assert_close(model.residual_direction, expected_direction, atol=2e-14, rtol=0)
    torch.testing.assert_close(model.residual_direction, observed_direction, atol=2e-12, rtol=0)
    full_cap = -model.merit_product / float(torch.dot(model.residual_direction, model.residual_direction))
    incomplete = torch.tensor([-0.5, -1.0, 0.0], dtype=dtype)
    incomplete_cap = -model.merit_product / float(torch.dot(incomplete, incomplete))
    assert full_cap == pytest.approx(1.0)
    assert incomplete_cap == pytest.approx(0.8)
    assert model.gates["envelope_slope_matches_residual_dot"] is True


def test_mixing_search_uses_actual_minimum_theta_after_evaluation_not_linear_prediction():
    dtype = torch.float64
    gm, gp = torch.tensor([-0.2, 1.0], dtype=dtype), torch.tensor([0.2, 1.0], dtype=dtype)
    normal, chart = torch.tensor([1.0, 0.0], dtype=dtype), torch.tensor([[0.0], [1.0]], dtype=dtype)
    hm, hp = torch.tensor([-0.4, -1.0], dtype=dtype), torch.tensor([-0.6, -1.0], dtype=dtype)
    model = mixing.minimum_tangent_model(gm, gp, hm, hp, normal, chart,
        0.5, 0.0, pivot=0, face_scale=0.84)
    model = replace(model, delta_theta=100.0)
    current = torch.zeros(2, dtype=dtype)
    evaluations = []

    def evaluate(candidate, linear_theta, alpha):
        evaluations.append((linear_theta, alpha))
        return {"theta": 0.5, "mixing_minimum_valid": True,
            "J_armijo_passed": True, "F_squared_armijo_passed": True,
            "face_audit_passed": True, "branch_pair_passed": True,
            "side_objectives_match_native": True, "side_gradients_finite": True}

    proposal, trials = tangent.search_candidates(model, current, evaluate,
        chart_candidate=lambda base, direction, alpha: base + alpha * direction,
        candidate_mixing_minimum=True)
    assert proposal is not None
    assert proposal["theta"] == 0.5
    assert proposal["linear_theta_prediction"] > 1.0
    assert evaluations and trials[0]["status"] == "accepted"

    refused, refused_trials = tangent.search_candidates(model, current,
        lambda *_args: {"theta": 0.0, "mixing_minimum_valid": False,
            "J_armijo_passed": True, "F_squared_armijo_passed": True,
            "face_audit_passed": True, "branch_pair_passed": True,
            "side_objectives_match_native": True, "side_gradients_finite": True},
        chart_candidate=lambda base, direction, alpha: base + alpha * direction,
        candidate_mixing_minimum=True)
    assert refused is None
    assert refused_trials
    assert all(item["status"] == "mixing_minimum_refused" for item in refused_trials)

    no_op_evaluations = []
    no_op, no_op_trials = tangent.search_candidates(model, current,
        lambda *_args: (no_op_evaluations.append(True) or {"theta": 0.5,
            "mixing_minimum_valid": True}),
        chart_candidate=lambda base, _direction, _alpha: base.clone(),
        candidate_mixing_minimum=True)
    assert no_op is None
    assert not no_op_evaluations
    assert no_op_trials
    assert all(item["status"] == "zero_control_movement_refused" for item in no_op_trials)


def test_mixing_final_closure_requires_its_own_theta_star_match():
    proposal: dict[str, Any] = {key: True for key in ("candidate_mixing_minimum", "theta", "control",
        "objective", "F_squared", "side_gradients")}
    proposal.update(candidate_mixing_minimum=True, theta=0.4, control=[0.0, 0.0],
        objective=1.0, F_squared=0.2,
        side_gradients={"-1": [0.0, 1.0], "1": [1.0, -1.0]})
    repeat: dict[str, Any] = {key: True for key in ("side_gradients_finite", "native_objective_matches_proposal",
        "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
        "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
        "fixed_input_unchanged", "runtime_unchanged", "deadline_passed",
        "mixing_minimum_valid", "minimum_theta_matches_proposal")}
    repeat.update(theta=0.4, control=[0.0, 0.0],
        side_gradients={"-1": [0.0, 1.0], "1": [1.0, -1.0]})
    assert tangent.fresh_final_closure(proposal, repeat)
    repeat["minimum_theta_matches_proposal"] = False
    assert not tangent.fresh_final_closure(proposal, repeat)


def test_pr270_base_loader_uses_terminal_selected_final_repeat_not_stale_top_level_scalars():
    base = mixing._load_current_base({})
    assert tangent._tensor_sha(base["control"]) == mixing.BASE_CONTROL_SHA
    assert base["theta"] == mixing.BASE_THETA
    assert base["objective"] == mixing.BASE_OBJECTIVE
    assert base["accepted"]["F_squared"] == mixing.BASE_CARRIED_F_SQUARED
    assert base["raw"]["current_native_objective"] != base["objective"]
    assert base["raw"]["accepted_F_squared"] != base["accepted"]["F_squared"]


def test_fake_full_child_records_two_hvps_zero_rows_and_keeps_carried_theta_until_commit(
        tmp_path, monkeypatch):
    base = mixing._load_current_base({})
    control = base["control"]
    accepted = base["accepted"]
    gm = torch.as_tensor(accepted["side_gradients"]["-1"], dtype=torch.float64)
    gp = torch.as_tensor(accepted["side_gradients"]["1"], dtype=torch.float64)
    jump = gp - gm
    flow_derivative = 1.0 - torch.tanh(control[20:25]).square()
    weights = jump[20:25] / flow_derivative
    weights = 0.84 * weights / weights.abs().max()
    hessian = torch.full((26,), 0.05, dtype=torch.float64)
    mixed_base = gm + float(tangent.minimum_mixture_weight(gm, gp)) * jump

    class ToyProblem:
        def __init__(self):
            self.frozen = SimpleNamespace()

        def objective(self, value, _parameters):
            delta = value - control
            return (value.new_tensor(base["objective"]) + torch.dot(mixed_base, delta)
                + 0.5 * torch.dot(hessian * delta, delta))

    problem = ToyProblem()
    parameters = torch.zeros(1, dtype=torch.float64)
    original = torch.zeros((4, 5), dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)
    runtime = base["raw"]["runtime_after"]
    source = {"fixture.py": "frozen"}
    plan = {"experiment_kind": "current_tangent_mixing_minimum", "policy": mixing.POLICY,
        "method": mixing.METHOD, "source_files": {}, "archive_files": {},
        "base_control_sha256": mixing.BASE_CONTROL_SHA, "initial_theta": mixing.BASE_THETA,
        "base_objective": mixing.BASE_OBJECTIVE,
        "base_carried_F_squared": mixing.BASE_CARRIED_F_SQUARED, "face": mixing.FACE}
    output = tmp_path / "synthetic-child.json"

    def observe(_probe, _problem, value, _parameters, face_weights):
        delta = value - control
        objective = problem.objective(value, parameters)
        gminus = gm + hessian * delta
        gplus = gp + hessian * delta
        q = geometry._face_value(value, face_weights)
        return {"native_j": objective,
            "side": {-1: (objective, gminus), 1: (objective, gplus)},
            "traces": {-1: accepted["branch_trace"]["-1"],
                       1: accepted["branch_trace"]["1"]},
            "pair": accepted["current_branch_pair_gate"], "q": q,
            "production_q": q, "face_bound": 1e-12,
            "face_ok": bool(abs(float(q)) <= 1e-12),
            "side_objectives_match_native": True,
            "side_gradients_finite": True}

    monkeypatch.setattr(mixing, "_load_plan", lambda *_args: plan)
    monkeypatch.setattr(mixing, "_load_current_base", lambda _plan: base)
    monkeypatch.setattr(tangent.seed, "_prepare_fixed_seed", lambda: (
        problem, original, None, parameters, truth, {}))
    monkeypatch.setattr(shared, "_input_identity", lambda *_args: base["raw"]["input_after"])
    monkeypatch.setattr(shared, "_runtime", lambda: runtime)
    monkeypatch.setattr(shared, "_source_hashes", lambda *_args: source)
    monkeypatch.setattr(tangent, "_observe", observe)
    monkeypatch.setattr(geometry, "_face_weights", lambda *_args, **_kwargs: weights)
    monkeypatch.setattr(geometry.model, "_fixed_input", lambda *_args: True)
    monkeypatch.setattr(shared.transport, "selected_face_extension", lambda *_args, **_kwargs: nullcontext())

    child = mixing._run_child_impl(tmp_path / "plan.json", "synthetic-plan", output)
    assert child["execution_status"] == "completed"
    assert child["numerical_status"] == "tangent_mixing_minimum_accepted"
    assert child["base_control_sha256"] == mixing.BASE_CONTROL_SHA
    assert child["jacobian_rows_started"] == child["jacobian_rows_completed"] == 0
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 2
    assert len(child["iterations"]) == 1
    item = child["iterations"][0]
    assert item["theta"] == mixing.BASE_THETA
    assert item["working_theta"] != item["theta"] or child["accepted_iterations"] == 1
    assert item["accepted"] is True
    assert item["final_repeat"]["mixing_minimum_valid"] is True
    assert item["final_repeat"]["minimum_theta_matches_proposal"] is True
    assert child["candidate_committed"] is True
    assert child["current_theta"] == item["committed_theta"]

    def boundary_refusal(_gm, _gp):
        raise ValueError("synthetic base mixing weight is on a boundary")

    monkeypatch.setattr(tangent, "minimum_mixture_weight", boundary_refusal)
    refused = mixing._run_child_impl(tmp_path / "plan-refusal.json", "synthetic-plan",
        tmp_path / "synthetic-refusal.json")
    assert refused["numerical_status"] == "tangent_mixing_minimum_refused"
    assert refused["accepted_iterations"] == 0
    assert refused["hvp_calls_started"] == refused["hvp_calls_completed"] == 0
    assert refused["current_control_sha256"] == mixing.BASE_CONTROL_SHA
    assert refused["current_theta"] == mixing.BASE_THETA
    assert refused["last_confirmed_control_sha256"] == mixing.BASE_CONTROL_SHA
    assert refused["last_confirmed_theta"] == mixing.BASE_THETA
