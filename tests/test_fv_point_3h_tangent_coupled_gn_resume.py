from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_comparison as coupled
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_resume as resume
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing


def _fake_resume(tmp_path, monkeypatch, *, refuse_after_first=False,
        parity_refusal_after_first=False, solve_failure_after_first=False, reject_final=False):
    base = resume._load_current_base({})
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

    class ToyProblem:
        def __init__(self):
            self.frozen = SimpleNamespace()

        def objective(self, value, _parameters):
            displacement = value - control
            return (value.new_tensor(base["objective"]) + torch.dot(mixed0, displacement)
                + 0.5 * torch.dot(hessian * displacement, displacement))

    problem = ToyProblem()
    parameters = torch.zeros(1, dtype=torch.float64)
    original = torch.zeros((4, 5), dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)
    runtime = base["raw"]["runtime_after"]
    source = {"fixture.py": "frozen"}
    plan = {"experiment_kind": "current_tangent_coupled_gn_resume", "policy": resume.POLICY,
        "method": resume.METHOD, "source_files": {}, "archive_files": {},
        "base_control_sha256": resume.BASE_CONTROL_SHA, "initial_theta": resume.BASE_THETA,
        "base_objective": resume.BASE_OBJECTIVE, "base_carried_F_squared": resume.BASE_CARRIED_F_SQUARED,
        "face": resume.FACE}

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
        return {"native_j": objective, "side": {-1: (objective, gminus), 1: (objective, gplus)},
            "traces": {-1: accepted["branch_trace"]["-1"], 1: accepted["branch_trace"]["1"]},
            "pair": accepted["current_branch_pair_gate"], "q": q, "production_q": q,
            "face_bound": 1e-12, "face_ok": bool(abs(float(q)) <= 1e-12),
            "side_objectives_match_native": True, "side_gradients_finite": True}

    original_factory = mixing._working_model

    def fake_coupled_factory(context):
        build_minimum = original_factory(context)
        calls = 0

        def build(control_now, reference_theta, *args):
            nonlocal calls
            calls += 1
            if refuse_after_first and calls == 2:
                raise ValueError("synthetic later-point coupled refusal")
            spec = build_minimum(control_now, reference_theta, *args)[0]
            minimum = context["record"]["base_mixing_minimum"]
            point_sha = tangent._tensor_sha(control_now)
            row_history = context["record"].setdefault("jacobian_row_history", [])
            context["record"]["jacobian_rows_started"] = len(row_history) + 24
            for side in (-1, 1):
                for row in range(12):
                    row_history.append({"base_control_sha256": point_sha,
                        "theta": minimum["working_theta_star"], "side": side, "row": row,
                        "gradient": [0.0] * 26, "residual_value": 0.0, "status": "completed"})
            context["record"]["jacobian_rows_completed"] = len(row_history)
            if parity_refusal_after_first and calls == 2:
                raise ValueError("synthetic later-point row-parity refusal")
            audit = {"B": [[0.0] * 26 for _ in range(12)], "S": torch.eye(12).tolist(),
                "dense_solves": 1, "dense_solve_dimension": 12, "S_dimension": 12,
                "positive_definite_from_cholesky": True, "solve_residual": 0.0,
                "solve_residual_budget": 1e-14, "tangent_descent_product": -1.0,
                "tangent_descent_budget": 1e-14}
            context["record"]["dense_solves_started"] = calls
            context["record"]["dense_solves_completed"] = calls
            context["record"]["dense_solve_audit"] = audit
            if solve_failure_after_first and calls == 2:
                raise ValueError("synthetic later-point GN solve closure failure")
            return [{**spec, "name": "robust_gn_coupled", "diagnostics": audit}]
        return build

    monkeypatch.setattr(resume, "_load_plan", lambda *_args: plan)
    monkeypatch.setattr(resume, "_load_current_base", lambda _plan: base)
    monkeypatch.setattr(coupled, "_direction_model_factory", fake_coupled_factory)
    monkeypatch.setattr(tangent.seed, "_prepare_fixed_seed", lambda: (
        problem, original, None, parameters, truth, {}))
    monkeypatch.setattr(shared, "_input_identity", lambda *_args: base["raw"]["input_after"])
    monkeypatch.setattr(shared, "_runtime", lambda: runtime)
    monkeypatch.setattr(shared, "_source_hashes", lambda *_args: source)
    monkeypatch.setattr(tangent, "_observe", observe)
    monkeypatch.setattr(geometry, "_face_weights", lambda *_args, **_kwargs: weights)
    monkeypatch.setattr(geometry.model, "_fixed_input", lambda *_args: not reject_final)
    monkeypatch.setattr(shared.transport, "selected_face_extension", lambda *_args, **_kwargs: nullcontext())
    output = tmp_path / "gn-resume.json"
    return resume._run_child_impl(tmp_path / "plan.json", "synthetic-plan", output), plan


def test_fake_coupled_gn_resume_closes_three_steps_and_renews_point_receipts(tmp_path, monkeypatch):
    child, plan = _fake_resume(tmp_path, monkeypatch)
    assert child["execution_status"] == "completed"
    assert child["numerical_status"] == "tangent_coupled_gn_resume_cap_reached"
    assert child["accepted_iterations"] == child["optimizer_steps_applied"] == 3
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 6
    assert child["jacobian_rows_started"] == child["jacobian_rows_completed"] == 72
    assert child["dense_solves_started"] == child["dense_solves_completed"] == 3
    assert len(child["coupled_gn_point_history"]) == len(child["coupled_gn_solve_history"]) == 3
    assert all(len([row for row in child["jacobian_row_history"]
        if row["base_control_sha256"] == point["base_control_sha256"]]) == 24
        for point in child["coupled_gn_point_history"])
    assert tangent._mixing_minimum_resume_chain_closed(child, plan,
        resume.BASE_CONTROL_SHA, child["hvp_history"], expected_model_name="robust_gn_coupled")
    broken = dict(child)
    broken["dense_solves_completed"] -= 1
    assert not tangent._mixing_minimum_resume_chain_closed(broken, plan,
        resume.BASE_CONTROL_SHA, child["hvp_history"], expected_model_name="robust_gn_coupled")


def test_later_coupled_gn_refusal_keeps_last_confirmed_step(tmp_path, monkeypatch):
    child, plan = _fake_resume(tmp_path, monkeypatch, refuse_after_first=True)
    assert child["execution_status"] == "completed"
    assert child["numerical_status"] == "tangent_coupled_gn_resume_stopped_after_commit"
    assert child["accepted_iterations"] == child["optimizer_steps_applied"] == 1
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 2
    assert child["jacobian_rows_started"] == child["jacobian_rows_completed"] == 24
    assert child["dense_solves_started"] == child["dense_solves_completed"] == 1
    item = child["iterations"][0]
    assert child["current_control_sha256"] == tangent._tensor_sha(
        torch.as_tensor(item["committed_control"], dtype=torch.float64))
    assert child["current_theta"] == item["committed_theta"]
    assert tangent._mixing_minimum_resume_chain_closed(child, plan,
        resume.BASE_CONTROL_SHA, child["hvp_history"], expected_model_name="robust_gn_coupled")


def test_row_parity_refusal_after_complete_batch_keeps_last_confirmed_step(tmp_path, monkeypatch):
    child, plan = _fake_resume(tmp_path, monkeypatch, parity_refusal_after_first=True)
    assert child["execution_status"] == "completed"
    assert child["numerical_status"] == "tangent_coupled_gn_resume_stopped_after_commit"
    assert child["accepted_iterations"] == child["optimizer_steps_applied"] == 1
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 2
    assert child["jacobian_rows_started"] == child["jacobian_rows_completed"] == 48
    assert child["dense_solves_started"] == child["dense_solves_completed"] == 1
    committed = child["iterations"][0]
    committed_sha = tangent._tensor_sha(torch.as_tensor(committed["committed_control"], dtype=torch.float64))
    terminal_rows = [row for row in child["jacobian_row_history"]
        if row["base_control_sha256"] == committed_sha]
    assert len(terminal_rows) == 24
    assert {(row["side"], row["row"]) for row in terminal_rows} == {
        (side, row) for side in (-1, 1) for row in range(12)}
    assert child["current_control_sha256"] == tangent._tensor_sha(
        torch.as_tensor(committed["committed_control"], dtype=torch.float64))
    assert child["current_theta"] == committed["committed_theta"]
    assert tangent._mixing_minimum_resume_chain_closed(child, plan,
        resume.BASE_CONTROL_SHA, child["hvp_history"], expected_model_name="robust_gn_coupled")


def test_final_input_closure_failure_preserves_confirmed_control(tmp_path, monkeypatch):
    child, _plan = _fake_resume(tmp_path, monkeypatch, reject_final=True)
    assert child["execution_status"] == "failed"
    assert child["numerical_status"] == "diagnostic_closure_failed"
    assert child["jacobian_rows_completed"] == 24
    assert child["hvp_calls_completed"] == 2
    assert child["dense_solves_completed"] == 1
    assert child["last_confirmed_control_sha256"] == resume.BASE_CONTROL_SHA
    assert child["current_control_sha256"] == resume.BASE_CONTROL_SHA
    assert child["accepted_iterations"] == child["optimizer_steps_applied"] == 0


def test_completed_solve_without_model_receipt_fails_closure_and_preserves_commit(tmp_path, monkeypatch):
    child, _plan = _fake_resume(tmp_path, monkeypatch, solve_failure_after_first=True)
    assert child["execution_status"] == "failed"
    assert child["numerical_status"] == "diagnostic_closure_failed"
    assert child["accepted_iterations"] == child["optimizer_steps_applied"] == 1
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 2
    assert child["jacobian_rows_started"] == child["jacobian_rows_completed"] == 48
    assert child["dense_solves_started"] == child["dense_solves_completed"] == 2
    committed = child["iterations"][0]
    committed_sha = tangent._tensor_sha(torch.as_tensor(committed["committed_control"], dtype=torch.float64))
    assert child["last_confirmed_control_sha256"] == committed_sha
    assert child["current_control_sha256"] == committed_sha
