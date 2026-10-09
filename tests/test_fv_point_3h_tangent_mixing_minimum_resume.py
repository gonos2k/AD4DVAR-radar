from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_resume as resume


def _synthetic_base():
    base = resume._load_current_base({})
    accepted = base["accepted"]
    control = base["control"]
    gm = torch.as_tensor(accepted["side_gradients"]["-1"], dtype=torch.float64)
    gp = torch.as_tensor(accepted["side_gradients"]["1"], dtype=torch.float64)
    theta_star = float(tangent.minimum_mixture_weight(gm, gp))
    jump = gp - gm
    flow_derivative = 1.0 - torch.tanh(control[20:25]).square()
    weights = jump[20:25] / flow_derivative
    weights = 0.84 * weights / weights.abs().max()
    normal = geometry._face_normal(control, weights)
    jump_scale = torch.dot(jump, normal) / torch.dot(normal, normal)
    mixed = gm + theta_star * jump
    return base, control, gm, gp, theta_star, mixed, jump_scale, weights


def _fake_child(tmp_path, monkeypatch, *, later_refusal=False):
    base, control, gm_base, gp_base, theta_star, mixed_base, jump_scale, weights = _synthetic_base()
    accepted = base["accepted"]
    hessian = torch.full((26,), 0.05, dtype=torch.float64)

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
    plan = {"experiment_kind": "current_tangent_mixing_minimum_resume",
        "policy": resume.POLICY, "method": resume.METHOD,
        "source_files": {}, "archive_files": {},
        "base_control_sha256": resume.BASE_CONTROL_SHA,
        "initial_theta": resume.BASE_THETA,
        "base_objective": resume.BASE_OBJECTIVE,
        "base_carried_F_squared": resume.BASE_CARRIED_F_SQUARED,
        "face": resume.FACE}

    def observe(_probe, _problem, value, _parameters, face_weights):
        delta = value - control
        objective = problem.objective(value, parameters)
        point_normal = geometry._face_normal(value, face_weights)
        jump = jump_scale * point_normal
        mixed = mixed_base + hessian * delta
        mixed = mixed - point_normal * (torch.dot(point_normal, mixed)
            / torch.dot(point_normal, point_normal))
        gminus = mixed - theta_star * jump
        gplus = mixed + (1.0 - theta_star) * jump
        q = geometry._face_value(value, face_weights)
        return {"native_j": objective,
            "side": {-1: (objective, gminus), 1: (objective, gplus)},
            "traces": {-1: accepted["branch_trace"]["-1"],
                1: accepted["branch_trace"]["1"]},
            "pair": accepted["current_branch_pair_gate"], "q": q,
            "production_q": q, "face_bound": 1e-12,
            "face_ok": bool(abs(float(q)) <= 1e-12),
            "side_objectives_match_native": True, "side_gradients_finite": True}

    monkeypatch.setattr(resume, "_load_plan", lambda *_args: plan)
    monkeypatch.setattr(resume, "_load_current_base", lambda _plan: base)
    monkeypatch.setattr(tangent.seed, "_prepare_fixed_seed", lambda: (
        problem, original, None, parameters, truth, {}))
    monkeypatch.setattr(shared, "_input_identity", lambda *_args: base["raw"]["input_after"])
    monkeypatch.setattr(shared, "_runtime", lambda: runtime)
    monkeypatch.setattr(shared, "_source_hashes", lambda *_args: source)
    monkeypatch.setattr(tangent, "_observe", observe)
    monkeypatch.setattr(geometry, "_face_weights", lambda *_args, **_kwargs: weights)
    monkeypatch.setattr(geometry.model, "_fixed_input", lambda *_args: True)
    monkeypatch.setattr(shared.transport, "selected_face_extension", lambda *_args, **_kwargs: nullcontext())
    if later_refusal:
        original_model_factory = mixing._working_model

        def one_then_refuse(context):
            actual = original_model_factory(context)
            calls = 0

            def build(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError("synthetic later-point theta refusal")
                return actual(*args)
            return build
        monkeypatch.setattr(mixing, "_working_model", one_then_refuse)
    output = tmp_path / ("resume-refusal.json" if later_refusal else "resume-3step.json")
    child = resume._run_child_impl(tmp_path / "plan.json", "synthetic-plan", output)
    return child, plan


def test_resume_loader_uses_terminal_pr271_final_repeat_as_new_base():
    base = resume._load_current_base({})
    assert tangent._tensor_sha(base["control"]) == resume.BASE_CONTROL_SHA
    assert base["theta"] == resume.BASE_THETA
    assert base["objective"] == resume.BASE_OBJECTIVE
    assert base["accepted"]["F_squared"] == resume.BASE_CARRIED_F_SQUARED
    assert base["accepted"] == base["raw"]["last_confirmed_closure"]
    assert base["raw"]["current_native_objective"] != resume.BASE_OBJECTIVE


def test_fake_resume_child_closes_three_step_six_hvp_chain_and_parent_validator(tmp_path, monkeypatch):
    child, plan = _fake_child(tmp_path, monkeypatch)
    assert child["execution_status"] == "completed"
    assert child["numerical_status"] == "tangent_mixing_minimum_resume_cap_reached"
    assert child["accepted_iterations"] == child["optimizer_steps_applied"] == 3
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 6
    assert child["jacobian_rows_started"] == child["jacobian_rows_completed"] == 0
    assert len(child["iterations"]) == 3
    assert len(child["mixing_minimum_point_history"]) == 3
    items = child["iterations"]
    assert items[0]["theta"] == resume.BASE_THETA
    assert items[1]["theta"] == items[0]["committed_theta"]
    assert items[2]["theta"] == items[1]["committed_theta"]
    assert all(item["accepted"] and item["candidate_mixing_minimum"] for item in items)
    assert all(len(item["model_comparisons"][0]["trials"]) <= 16 for item in items)
    assert tangent._mixing_minimum_resume_chain_closed(
        child, plan, resume.BASE_CONTROL_SHA, child["hvp_history"])

    broken = dict(child)
    broken["iterations"] = [dict(item) for item in items]
    broken["iterations"][1]["theta"] += 0.01
    assert not tangent._mixing_minimum_resume_chain_closed(
        broken, plan, resume.BASE_CONTROL_SHA, child["hvp_history"])


def test_later_resume_refusal_preserves_last_confirmed_candidate_and_theta(tmp_path, monkeypatch):
    child, plan = _fake_child(tmp_path, monkeypatch, later_refusal=True)
    assert child["execution_status"] == "completed"
    assert child["numerical_status"] == "tangent_mixing_minimum_resume_stopped_after_commit"
    assert child["accepted_iterations"] == child["optimizer_steps_applied"] == 1
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 2
    assert len(child["iterations"]) == 2
    committed = child["iterations"][0]
    refused = child["iterations"][1]
    assert committed["accepted"] is True and refused["accepted"] is False
    assert "synthetic later-point theta refusal" in refused["refusal"]
    assert child["current_control_sha256"] == tangent._tensor_sha(
        torch.as_tensor(committed["committed_control"], dtype=torch.float64))
    assert child["current_theta"] == committed["committed_theta"]
    assert child["last_confirmed_control_sha256"] == child["current_control_sha256"]
    assert child["last_confirmed_theta"] == child["current_theta"]
    assert tangent._mixing_minimum_resume_chain_closed(
        child, plan, resume.BASE_CONTROL_SHA, child["hvp_history"])


def test_resume_status_distinguishes_refusal_partial_progress_and_cap():
    assert tangent._mixing_minimum_resume_status(0, 3) == "tangent_mixing_minimum_resume_refused"
    assert tangent._mixing_minimum_resume_status(2, 3) == "tangent_mixing_minimum_resume_stopped_after_commit"
    assert tangent._mixing_minimum_resume_status(3, 3) == "tangent_mixing_minimum_resume_cap_reached"
