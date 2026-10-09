from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry


def _model(*, q=0.0, theta=0.25):
    dtype = torch.float64
    gm = torch.tensor([-1.0, -1.0], dtype=dtype)
    gp = torch.tensor([1.0, -1.0], dtype=dtype)
    normal = torch.tensor([1.0, 0.0], dtype=dtype)
    z = torch.tensor([[0.0], [1.0]], dtype=dtype)
    hm = torch.tensor([0.2, 0.4], dtype=dtype)
    hp = torch.tensor([0.6, 0.8], dtype=dtype)
    return tangent.tangent_model(gm, gp, hm, hp, normal, z, theta, q=q)


def test_fresh_tangent_products_match_analytic_face_model():
    model = _model()
    torch.testing.assert_close(model.mixed_gradient, torch.tensor([-0.5, -1.0], dtype=torch.float64))
    torch.testing.assert_close(model.tangent_gradient, torch.tensor([0.0, -1.0], dtype=torch.float64))
    torch.testing.assert_close(model.direction, torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert model.delta_theta_numerator == pytest.approx(-0.2)
    assert model.delta_theta_denominator == pytest.approx(2.0)
    assert model.delta_theta == pytest.approx(0.1)
    torch.testing.assert_close(model.residual_direction,
        torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64))
    assert model.side_products == pytest.approx((-1.0, -1.0))
    assert model.merit_product == pytest.approx(-0.75)
    assert model.gates["both_side_gradients_descend"]
    assert model.gates["merit_descends"]


def test_26d_curved_face_hvps_match_autodiff_along_true_chart_tangent():
    dtype = torch.float64
    control = torch.linspace(-0.12, 0.18, 26, dtype=dtype)
    weights = torch.tensor([1.0, -0.7, 0.9, 0.5, -0.8], dtype=dtype)
    pivot = geometry._pivot(control, weights)
    point, z, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    normal = geometry._face_normal(point, weights)
    unit = normal / torch.linalg.vector_norm(normal)
    retained = [i for i in range(26) if i != pivot]
    coords = torch.linspace(0.03, 0.17, 25, dtype=dtype)
    tangent_gradient = z @ coords
    gm = tangent_gradient - 0.2 * normal
    gp = tangent_gradient + 0.8 * normal
    # The stored Q has a curved tanh chart; its analytic chart tangent is Z*d.
    mix, _, direction, _ = tangent.tangent_direction(gm, gp, normal, z, 0.35)
    torch.testing.assert_close(direction, -(z @ coords), atol=2e-15, rtol=2e-14)
    assert abs(float(torch.dot(normal, direction))) < 2e-15

    a = torch.diag(torch.linspace(0.5, 1.8, 26, dtype=dtype))
    b = torch.diag(torch.linspace(1.9, 0.4, 26, dtype=dtype))
    def objective(x, matrix, linear, face_coefficient):
        q = torch.dot(weights, torch.tanh(x[20:25]))
        dx = x - point
        return 0.5 * dx @ matrix @ dx + torch.dot(linear, dx) + face_coefficient * q

    hvps: list[torch.Tensor] = []
    for matrix, linear, coeff in ((a, tangent_gradient, -0.2),
                                  (b, tangent_gradient, 0.8)):
        grad_fn = torch.func.grad(lambda x: objective(x, matrix, linear, coeff))
        # Verify the endpoint gradient and the fresh HVP against independent AD.
        actual_gradient = grad_fn(point)
        torch.testing.assert_close(actual_gradient, gm if coeff < 0 else gp, atol=2e-14, rtol=2e-14)
        hvps.append(torch.func.jvp(grad_fn, (point,), (direction,))[1])
    model = tangent.tangent_model(gm, gp, hvps[0], hvps[1], normal, z, 0.35)
    assert model.direction.shape == (26,)
    assert model.gates["common_face_gradient_jump"]
    assert abs(float(torch.dot(normal, model.direction))) < 3e-15
    expected_num = torch.dot(unit, mix + (0.65 * hvps[0] + 0.35 * hvps[1]))
    expected_den = torch.dot(unit, gp - gm)
    assert model.delta_theta == pytest.approx(float(-expected_num / expected_den), rel=2e-14)


def test_jump_must_be_common_face_supported_and_normal_component_resolved():
    gm = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    n = torch.tensor([1.0, 0.0], dtype=torch.float64)
    z = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    h = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="common face"):
        tangent.tangent_model(gm, torch.tensor([1.0, -0.1], dtype=torch.float64),
            h, h, n, z, 0.5)
    with pytest.raises(ValueError, match="normal component"):
        tangent.tangent_model(gm, torch.tensor([-1.0, -1.0], dtype=torch.float64),
            h, h, n, z, 0.5)


def test_tangent_uses_the_actual_chart_pivot_instead_of_inferring_from_normal():
    dtype = torch.float64
    gm = torch.tensor([2.0, -1.0], dtype=dtype)
    normal = torch.tensor([1.0, 2.0], dtype=dtype)
    gp = gm + normal
    chart = torch.tensor([[-2.0], [1.0]], dtype=dtype)
    _, _, direction, _ = tangent.tangent_direction(gm, gp, normal, chart, 0.4, pivot=0)
    torch.testing.assert_close(direction, torch.tensor([-2.0, 1.0], dtype=dtype))


def test_actual_chart_radius_and_theta_domain_are_checked_in_search():
    model = _model()
    current = torch.zeros(2, dtype=torch.float64)
    seen = []

    def evaluate(candidate, theta, alpha):
        assert alpha == pytest.approx(float(candidate[1]))
        seen.append((candidate.clone(), theta))
        return {name: True for name in ("J_armijo_passed", "F_squared_armijo_passed",
            "face_audit_passed", "branch_pair_passed", "side_objectives_match_native",
            "side_gradients_finite")} | {"objective": 1.0, "F_squared": 0.2,
                "side_gradients": {"-1": [-1, -1], "1": [1, -1]}}

    proposal, trials = tangent.search_candidates(model, current, evaluate,
        chart_candidate=lambda base, direction, alpha: base + alpha * direction)
    assert proposal is not None and proposal["accepted"]
    assert float(torch.linalg.vector_norm(torch.tensor(proposal["control"], dtype=torch.float64))) <= 0.05
    assert all(t["status"] != "actual_chart_radius_refused" for t in trials)
    assert len(seen) == 1

    # Force theta outside its convex domain: every trial is refused before evaluation.
    bad = replace(model, delta_theta=100.0)
    seen.clear()
    proposal, trials = tangent.search_candidates(bad, current, evaluate,
        chart_candidate=lambda base, direction, alpha: base + alpha * direction)
    assert proposal is not None
    assert [trial["status"] for trial in trials[:3]] == ["theta_domain_refused"] * 3
    assert 0.0 <= proposal["theta"] <= 1.0
    assert len(seen) == 1


def test_final_closure_requires_each_p2_side_gradient_and_every_closure_fact():
    proposal: dict[str, Any] = {key: True for key in ("theta", "control", "objective", "F_squared", "side_gradients")}
    proposal.update(theta=1.0, control=[1.0, 2.0], objective=1.0, F_squared=0.2)
    proposal["side_gradients"] = {"-1": [1.0, 2.0], "1": [3.0, 4.0]}
    repeat: dict[str, Any] = {key: True for key in ("side_gradients_finite", "native_objective_matches_proposal",
        "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
        "branch_pair_passed", "trace_matches_proposal", "source_unchanged", "fixed_input_unchanged",
        "runtime_unchanged", "deadline_passed")}
    repeat.update(theta=1.0, control=[1.0, 2.0])
    repeat["side_gradients"] = {"-1": [1.0, 2.0], "1": [3.0, 4.0]}
    assert tangent.fresh_final_closure(proposal, repeat)
    repeat["side_gradients"] = {"-1": [1.0, 3.0], "1": [3.0, 3.0]}
    assert not tangent.fresh_final_closure(proposal, repeat)
    repeat["side_gradients"] = {"-1": [1.0, 2.0], "1": [3.0, 4.0]}
    repeat["trace_matches_proposal"] = False
    assert not tangent.fresh_final_closure(proposal, repeat)


def test_bounded_loop_recomputes_direction_and_two_hvps_after_each_commit():
    control = torch.zeros(2, dtype=torch.float64)
    counts = {"point": 0, "hvp": 0, "final": 0}
    durable_counts = []
    gm, gp = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, -1.0])
    normal, z = torch.tensor([1.0, 0.0]), torch.tensor([[0.0], [1.0]])

    def current_products(point, theta):
        counts["point"] += 1
        return gm.to(point), gp.to(point), normal.to(point), z.to(point), 0.0

    def hvp(point, direction, side):
        counts["hvp"] += 1
        return torch.tensor([0.2, 0.4] if side < 0 else [0.6, 0.8], dtype=point.dtype)

    def evaluate(candidate, theta, alpha):
        return {name: True for name in ("J_armijo_passed", "F_squared_armijo_passed",
            "face_audit_passed", "branch_pair_passed", "side_objectives_match_native",
            "side_gradients_finite")} | {"theta": theta, "control": candidate.tolist(),
                "objective": 1.0, "F_squared": 0.2,
                "side_gradients": {"-1": [-1.0, -1.0], "1": [1.0, -1.0]}}

    def final(proposal):
        counts["final"] += 1
        return {key: True for key in ("side_gradients_finite", "native_objective_matches_proposal",
            "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
            "branch_pair_passed", "trace_matches_proposal", "source_unchanged", "fixed_input_unchanged",
            "runtime_unchanged", "deadline_passed")} | {"theta": proposal["theta"],
                "control": proposal["control"], "side_gradients": {
                "-1": proposal["side_gradients"]["-1"],
                "1": proposal["side_gradients"]["1"]}}

    result = tangent.bounded_continuation(control, 0.25, current_products, hvp,
        lambda base, direction, alpha: base + alpha * direction, evaluate, final,
        committed_progress=lambda records: durable_counts.append(len(records)))
    assert result["accepted_iterations"] == 3
    assert result["candidate_type"] == "tangent_gradient_correction"
    assert counts == {"point": 3, "hvp": 6, "final": 3}
    assert durable_counts == [1, 2, 3]
    assert result["theta"] <= 1.0


def test_failed_fresh_final_repeat_preserves_last_confirmed_point_and_theta():
    control = torch.zeros(2, dtype=torch.float64)
    gm, gp = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, -1.0])
    n, z = torch.tensor([1.0, 0.0]), torch.tensor([[0.0], [1.0]])

    def products(point, theta):
        return gm, gp, n, z, 0.0

    def hvp(point, direction, side):
        return torch.tensor([0.2, 0.4] if side < 0 else [0.6, 0.8])

    def evaluate(candidate, theta, alpha):
        return {name: True for name in ("J_armijo_passed", "F_squared_armijo_passed",
            "face_audit_passed", "branch_pair_passed", "side_objectives_match_native",
            "side_gradients_finite")} | {"theta": theta, "control": candidate.tolist(),
                "objective": 1.0, "F_squared": 0.2,
                "side_gradients": {"-1": [-1.0, -1.0], "1": [1.0, -1.0]}}

    before = control.clone()
    result = tangent.bounded_continuation(control, 0.25, products, hvp,
        lambda base, direction, alpha: base + alpha * direction, evaluate,
        lambda _: {"side_gradients_finite": True,
            "theta": 0.25, "control": before.tolist(),
            "native_objective_matches_proposal": True,
            "side_objectives_match_native": True, "merit_matches_proposal": True,
            "face_audit_passed": True, "branch_pair_passed": True,
            "trace_matches_proposal": True, "source_unchanged": True,
            "fixed_input_unchanged": True, "runtime_unchanged": True,
            "deadline_passed": True,
            "side_gradients": {"-1": [-1.0, -1.0], "1": [1.0, -1.0]}})
    torch.testing.assert_close(result["control"], before)
    assert result["theta"] == 0.25
    assert result["accepted_iterations"] == 0


def test_second_iteration_exception_keeps_the_first_confirmed_commit():
    control = torch.zeros(2, dtype=torch.float64)
    gm, gp = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, -1.0])
    n, z = torch.tensor([1.0, 0.0]), torch.tensor([[0.0], [1.0]])
    points = 0

    def products(point, theta):
        nonlocal points
        points += 1
        if points == 2:
            raise TimeoutError("synthetic stop after one confirmed commit")
        return gm.to(point), gp.to(point), n.to(point), z.to(point), 0.0

    def hvp(point, direction, side):
        return torch.tensor([0.2, 0.4] if side < 0 else [0.6, 0.8], dtype=point.dtype)

    def evaluate(candidate, theta, alpha):
        return {name: True for name in ("J_armijo_passed", "F_squared_armijo_passed",
            "face_audit_passed", "branch_pair_passed", "side_objectives_match_native",
            "side_gradients_finite")} | {"theta": theta, "control": candidate.tolist(),
                "objective": 1.0, "F_squared": 0.2,
                "side_gradients": {"-1": [-1.0, -1.0], "1": [1.0, -1.0]}}

    def final(proposal):
        return {name: True for name in ("side_gradients_finite", "native_objective_matches_proposal",
            "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
            "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
            "fixed_input_unchanged", "runtime_unchanged", "deadline_passed")} | {
                "theta": proposal["theta"], "control": proposal["control"],
                "side_gradients": proposal["side_gradients"]}

    result = tangent.bounded_continuation(control, 0.25, products, hvp,
        lambda base, direction, alpha: base + alpha * direction, evaluate, final)
    assert result["accepted_iterations"] == 1
    assert result["theta"] != 0.25
    assert torch.linalg.vector_norm(result["control"]) > 0
    assert "TimeoutError" in result["iterations"][-1]["refusal"]


def test_child_exception_keeps_durable_two_commit_count(tmp_path, monkeypatch):
    control = torch.linspace(-0.1, 0.2, 26, dtype=torch.float64)
    digest = tangent._tensor_sha(control)
    output = tmp_path / 'step.json'

    def interrupted(plan_path, plan_sha, child_output):
        tangent._write(child_output, {
            'last_confirmed_control': control.tolist(),
            'last_confirmed_control_sha256': digest,
            'last_confirmed_theta': 0.4,
            'last_confirmed_iterations': 2,
        })
        raise TimeoutError('interrupted after two commits')

    monkeypatch.setattr(tangent, '_run_child_impl', interrupted)
    result = tangent._run_child(tmp_path / 'plan.json', 'fixed-plan', output)
    assert result['execution_status'] == 'failed'
    assert result['numerical_status'] == 'internal_deadline_refused'
    assert result['optimizer_steps_applied'] == result['accepted_iterations'] == 2
    assert result['current_control_sha256'] == digest
    assert result['current_control'] == control.tolist()
    assert result['current_theta'] == 0.4
