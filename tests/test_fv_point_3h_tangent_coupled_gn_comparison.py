from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import json
import torch
import pytest

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_comparison as coupled
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios import fv_point_3h_tangent_precondition_comparison as precondition
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_resume as resume


def test_coupled_woodbury_direction_matches_tiny_restricted_reference_and_counts_one_solve():
    dtype = torch.float64
    normal = torch.tensor([1.0, 0.0], dtype=dtype)
    tangent_residual = torch.tensor([0.0, 3.0], dtype=dtype)
    rows = torch.tensor([[3.0, 4.0], [2.0, -1.0]], dtype=dtype)
    curvature = torch.tensor([2.0, 0.5], dtype=dtype)
    chart = torch.tensor([[0.0], [1.0]], dtype=dtype)
    counts = {"started": 0, "completed": 0}
    direction, audit = coupled.coupled_woodbury_direction(normal, tangent_residual,
        rows, curvature, chart, pivot=0,
        on_solve_start=lambda: counts.__setitem__("started", counts["started"] + 1),
        on_solve_complete=lambda: counts.__setitem__("completed", counts["completed"] + 1))
    torch.testing.assert_close(direction, torch.tensor([0.0, -6.0 / 67.0], dtype=dtype),
        atol=2e-15, rtol=0)
    assert counts == {"started": 1, "completed": 1}
    assert audit["dense_solve_dimension"] == 2
    assert abs(float(torch.dot(normal, direction))) < 1e-15
    torch.testing.assert_close(torch.dot(tangent_residual, direction),
        torch.tensor(-18.0 / 67.0, dtype=dtype), atol=2e-15, rtol=0)

    no_rows, zero_audit = coupled.coupled_woodbury_direction(normal, tangent_residual,
        torch.zeros((2, 2), dtype=dtype), torch.zeros(2, dtype=dtype), chart, pivot=0)
    torch.testing.assert_close(no_rows, torch.tensor([0.0, -3.0], dtype=dtype))
    assert zero_audit["positive_definite_from_cholesky"] is True


def _side_objectives(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = value.unbind()
    h = y + 0.5 * y.square()
    return h + (-0.2 + 0.4 * y) * x, h + (0.2 + 0.6 * y) * x


def test_minimum_model_direction_override_keeps_hvp_theta_prime_and_df_on_same_path():
    dtype = torch.float64
    point = torch.zeros(2, dtype=dtype, requires_grad=True)
    normal = torch.tensor([1.0, 0.0], dtype=dtype)
    chart = torch.tensor([[0.0], [1.0]], dtype=dtype)
    minus, plus = _side_objectives(point)
    gm_graph = torch.autograd.grad(minus, point, create_graph=True)[0]
    gp_graph = torch.autograd.grad(plus, point, create_graph=True)[0]
    theta = tangent.minimum_mixture_weight(gm_graph, gp_graph)
    gm, gp = gm_graph.detach(), gp_graph.detach()
    direction = torch.tensor([0.0, -6.0 / 67.0], dtype=dtype)
    hminus = torch.tensor([-2.4 / 67.0, -6.0 / 67.0], dtype=dtype)
    hplus = torch.tensor([-3.6 / 67.0, -6.0 / 67.0], dtype=dtype)
    model = mixing.minimum_tangent_model(gm, gp, hminus, hplus, normal, chart,
        float(theta.detach()), 0.0, pivot=0, face_scale=0.84,
        direction_override=direction)
    torch.testing.assert_close(model.direction, direction, atol=0, rtol=0)
    torch.testing.assert_close(torch.tensor(model.side_products, dtype=dtype),
        torch.tensor([-6.0 / 67.0, -6.0 / 67.0], dtype=dtype), atol=2e-14, rtol=0)
    assert model.delta_theta == pytest.approx(7.5 / 67.0)
    torch.testing.assert_close(model.residual_direction,
        torch.tensor([0.0, -6.0 / 67.0, 0.0], dtype=dtype), atol=3e-14, rtol=0)
    assert model.merit_product == pytest.approx(-6.0 / 67.0)

    def envelope(value):
        jminus, jplus = _side_objectives(value)
        gminus = torch.autograd.grad(jminus, value, create_graph=True)[0]
        gplus = torch.autograd.grad(jplus, value, create_graph=True)[0]
        theta_star = tangent.minimum_mixture_weight(gminus, gplus)
        gstar = (1.0 - theta_star) * gminus + theta_star * gplus
        return torch.cat((gstar, value[0:1] / 0.84))

    _, df = torch.autograd.functional.jvp(envelope, point, direction)
    torch.testing.assert_close(model.residual_direction, df, atol=2e-12, rtol=0)


def _fake_coupled_child(tmp_path, monkeypatch):
    base = coupled._load_current_base({})
    control = base["control"]
    accepted = base["accepted"]
    gm0 = torch.as_tensor(accepted["side_gradients"]["-1"], dtype=torch.float64)
    gp0 = torch.as_tensor(accepted["side_gradients"]["1"], dtype=torch.float64)
    theta_star = float(tangent.minimum_mixture_weight(gm0, gp0))
    mixed0 = (1.0 - theta_star) * gm0 + theta_star * gp0
    jump0 = gp0 - gm0
    flow_derivative = 1.0 - torch.tanh(control[20:25]).square()
    weights = jump0[20:25] / flow_derivative
    weights = 0.84 * weights / weights.abs().max()
    normal0 = geometry._face_normal(control, weights)
    jump_scale = torch.dot(jump0, normal0) / torch.dot(normal0, normal0)
    hessian = torch.full((26,), 0.03, dtype=torch.float64)
    delta = 2.0

    transport = SimpleNamespace(psi_basis=torch.zeros((5, 5, 5), dtype=torch.float64),
        coefficient_limits=torch.ones(5, dtype=torch.float64))
    frozen = SimpleNamespace(fv_transport=transport, active_field_index=torch.arange(20),
        neural_prior_std_dbz=None, neural_prior_valid_mask=None, neural_prior_dependency=None,
        analysis_config=SimpleNamespace(field_smoothness_weight=0.0, pseudo_huber_delta=delta))

    class ToyProblem:
        layout = {"controls": 26, "parameters": 13, "state_shape": (4, 5),
            "observation_times": (0.0, 1.0, 2.0)}
        observation_dbz = torch.zeros((3, 4), dtype=torch.float64)
        observation_std_dbz = torch.ones((3, 4), dtype=torch.float64)
        quality_weight = torch.ones((3, 4), dtype=torch.float64)
        observation_correlation = None
        _correlation_whitener = None
        observation_status = None
        _detected_mask = torch.ones((3, 4), dtype=torch.bool)

        def __init__(self):
            self.frozen = frozen

        def contract(self, _parameters):
            return frozen

        def objective(self, value, _parameters):
            displacement = value - control
            return (value.new_tensor(base["objective"]) + torch.dot(mixed0, displacement)
                + 0.5 * torch.dot(hessian * displacement, displacement))

        def observation_residual(self, value, _parameters):
            objective = self.objective(value, _parameters)
            data_cost = objective - 0.5 * torch.dot(value, value)
            one_row = data_cost / 12.0
            ratio = 1.0 + one_row / delta**2
            magnitude = delta * torch.sqrt(ratio.square() - 1.0)
            return magnitude.expand(3, 4)

    problem = ToyProblem()
    parameters = torch.zeros(13, dtype=torch.float64)
    original = torch.zeros((4, 5), dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)
    runtime = base["raw"]["runtime_after"]
    source = {"fixture.py": "frozen"}
    plan = {"experiment_kind": "current_tangent_coupled_gn_comparison",
        "policy": coupled.POLICY, "method": coupled.METHOD,
        "source_files": {}, "archive_files": {},
        "base_control_sha256": coupled.BASE_CONTROL_SHA, "initial_theta": coupled.BASE_THETA,
        "base_objective": coupled.BASE_OBJECTIVE,
        "base_carried_F_squared": coupled.BASE_CARRIED_F_SQUARED, "face": coupled.FACE}
    output = tmp_path / "synthetic-coupled-child.json"

    def observe(_probe, _problem, value, _parameters, face_weights):
        displacement = value - control
        objective = problem.objective(value, parameters)
        normal = geometry._face_normal(value, face_weights)
        jump = jump_scale * normal
        mixture = mixed0 + hessian * displacement
        mixture = mixture - normal * (torch.dot(normal, mixture) / torch.dot(normal, normal))
        gminus = mixture - theta_star * jump
        gplus = mixture + (1.0 - theta_star) * jump
        q = geometry._face_value(value, face_weights)
        return {"native_j": objective,
            "side": {-1: (objective, gminus), 1: (objective, gplus)},
            "traces": {-1: accepted["branch_trace"]["-1"],
                1: accepted["branch_trace"]["1"]},
            "pair": accepted["current_branch_pair_gate"], "q": q,
            "production_q": q, "face_bound": 1e-12,
            "face_ok": bool(abs(float(q)) <= 1e-12),
            "side_objectives_match_native": True, "side_gradients_finite": True}

    monkeypatch.setattr(coupled, "_load_plan", lambda *_args: plan)
    monkeypatch.setattr(coupled, "_load_current_base", lambda _plan: base)
    monkeypatch.setattr(tangent.seed, "_prepare_fixed_seed", lambda: (
        problem, original, None, parameters, truth, {}))
    monkeypatch.setattr(shared, "_input_identity", lambda *_args: base["raw"]["input_after"])
    monkeypatch.setattr(shared, "_runtime", lambda: runtime)
    monkeypatch.setattr(shared, "_source_hashes", lambda *_args: source)
    monkeypatch.setattr(tangent, "_observe", observe)
    monkeypatch.setattr(geometry, "_face_weights", lambda *_args, **_kwargs: weights)
    monkeypatch.setattr(geometry.model, "_fixed_input", lambda *_args: True)
    monkeypatch.setattr(shared.transport, "selected_face_extension", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(precondition.v, "_control_prior_residual", lambda value, _frozen: value)
    monkeypatch.setattr(precondition.v, "_field_smoothness_prior_cost",
        lambda value, _frozen: value.new_zeros(()))
    return coupled._run_child_impl(tmp_path / "plan.json", "synthetic-plan", output), plan


def test_fake_full_child_records_two_arms_24_rows_four_hvps_and_one_12solve(tmp_path, monkeypatch):
    child, plan = _fake_coupled_child(tmp_path, monkeypatch)
    assert child["execution_status"] == "completed"
    assert child["numerical_status"] in (
        "tangent_coupled_gn_comparison_accepted", "tangent_coupled_gn_comparison_refused")
    assert child["jacobian_rows_started"] == child["jacobian_rows_completed"] == 24
    assert len(child["jacobian_row_history"]) == 24
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 4
    assert child["dense_solves_started"] == child["dense_solves_completed"] == 1
    assert len(child["iterations"]) == 1
    arms = child["iterations"][0]["model_comparisons"]
    assert {arm["name"] for arm in arms} == {"baseline_tangent", "robust_gn_coupled"}
    assert all(len(arm["trials"]) <= 16 for arm in arms)
    assert child["dense_solve_audit"]["S_dimension"] == 12
    assert child["dense_solve_audit"]["positive_definite_from_cholesky"] is True
    assert len(child["dense_solve_audit"]["S"]) == 12
    assert len(child["dense_solve_audit"]["B"]) == 12
    if child["accepted_iterations"]:
        assert child["iterations"][0]["final_repeat"]["mixing_minimum_valid"] is True
        assert child["iterations"][0]["final_repeat"]["minimum_theta_matches_proposal"] is True


def test_parent_closure_accepts_only_the_saved_coupled_arm_hvp_row_and_dense_receipts(
        tmp_path, monkeypatch):
    child, plan = _fake_coupled_child(tmp_path, monkeypatch)
    plan_path = tmp_path / "frozen-plan.json"
    plan_path.write_text("synthetic plan path; loader is pinned by the fixture")
    output = tmp_path / "parent-child.json"
    resource_path = tmp_path / "parent-resource.json"
    log_path = tmp_path / "parent.log"

    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        target = Path(command[command.index("--output") + 1])
        target.write_text(json.dumps(child))
        resource = {"exit_code": 0, "resource_termination": None,
            "monitor_error": None, "received_sigterm": False,
            "wall_limit_seconds": wall_seconds, "rss_limit_bytes": rss_bytes,
            "elapsed_seconds": 1.0, "sampled_peak_rss_bytes": 1_000_000,
            "child_process_group_cleanup_sent": False}
        report_path.write_text(json.dumps(resource))
        log_path.write_text("")
        return resource

    monkeypatch.setattr(tangent, "run_guarded_diagnostic", guarded)
    result = coupled.run(plan_path, "synthetic-plan", output, resource_path, log_path)
    assert result["parent"]["execution_status"] == "completed"
    assert result["child"]["source_unchanged"] is True
    assert result["child"]["jacobian_rows_completed"] == 24
    assert result["child"]["hvp_calls_completed"] == 4
    assert result["child"]["dense_solves_completed"] == 1

