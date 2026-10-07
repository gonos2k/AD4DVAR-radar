from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_model_guided_continuation as guided
from examples.weather_scenarios import fv_point_3h_qy32_tangent_step as tangent


def test_model_alpha_is_full_space_radius_and_phi_model_bound():
    g = torch.zeros(26, dtype=torch.float64)
    g[0] = 0.2
    hd = -g
    alpha = guided.model_alpha(g, hd)
    assert alpha == pytest.approx(min(1.0, guided.RADIUS / 0.2, 1.0))
    assert alpha * float(torch.linalg.vector_norm(g)) <= guided.RADIUS
    with pytest.raises(guided.guard_policy.StepRefusal):
        guided.model_alpha(torch.zeros_like(g), hd)


def test_linearized_phi_is_exact_norm_of_linearized_gradient():
    g = torch.arange(26, dtype=torch.float64) / 20
    d, hd, alpha = -g, -0.5 * g, 0.1
    model = guided.first_order_model(1.0, 0.5, g, d, hd, alpha)
    expected = torch.dot(g + alpha * hd, g + alpha * hd) / 2
    assert model["predicted_Phi_from_linear_gradient"] == pytest.approx(float(expected))
    assert torch.allclose(torch.tensor(model["predicted_gradient"], dtype=torch.float64), g + alpha * hd)


def test_hvp_started_count_is_written_only_after_deadline_admission():
    control = torch.zeros(26, dtype=torch.float64)
    params = torch.zeros(13, dtype=torch.float64)
    direction = torch.ones_like(control)
    starts = 0
    completions = 0

    def on_start():
        nonlocal starts
        starts += 1

    def on_complete():
        nonlocal completions
        completions += 1

    with pytest.raises(TimeoutError):
        guided._current_hvp(lambda value, _p: value, control, params, direction,
                            time.monotonic() - 1, on_start, on_complete)
    assert starts == completions == 0
    product = guided._current_hvp(lambda value, _p: value, control, params, direction,
                                  time.monotonic() + 10, on_start, on_complete)
    assert torch.equal(product, direction)
    assert (starts, completions) == (1, 1)


def test_actual_phi_armijo_refuses_nonlinear_phi_increase_without_j_floor():
    base_j = 1.0e12
    result = guided.merit_acceptance(base_j, 0.5, base_j, 0.5001,
                                     0.01, -0.1, -0.2)
    assert result["J_armijo_passed"] is True  # floating-point J threshold rounds to its offset
    assert result["Phi_armijo_passed"] is False
    assert result["phi_decrease_resolved"] is False
    assert result["accepted"] is False


def test_iteration_commit_updates_state_only_after_closure():
    control0 = torch.zeros(26, dtype=torch.float64)
    control1 = control0.clone()
    control1[0] = 0.1
    measure0 = {"objective": 1.0, "phi": 1.0, "gradient": [1.0] * 26,
        "gradient_inf": 1.0, "gradient_norm": 5.0,
        "branch": {"signature_sha256": "a"}, "branch_partition": {}}
    measure1 = {**measure0, "objective": 0.9, "gradient": [0.5] * 26,
                "gradient_inf": 0.5, "gradient_norm": 2.5,
                "branch": {"signature_sha256": "b"}}
    record: dict[str, Any] = {"iterations": []}
    first = {"accepted_control": control1.tolist(), "accepted_control_sha256": "one"}
    assert guided._commit_iteration(record, control1, measure1, first, closure_ok=True)
    committed_state = dict(record["current_state"])
    second = {"accepted_control": control0.tolist(), "accepted_control_sha256": "two"}
    assert not guided._commit_iteration(record, control0, measure0, second, closure_ok=False)
    assert len(record["iterations"]) == 1
    assert record["current_state"] == committed_state


class _Quadratic:
    def __init__(self, scale: float = 1.0, offset: float = 1.0e12) -> None:
        self.scale = scale
        self.offset = offset

    def objective(self, control: torch.Tensor, _parameters: torch.Tensor) -> torch.Tensor:
        return control.new_tensor(self.offset) + self.scale * control.square().sum() / 2


def _blackbox_case(monkeypatch, *, fail_second_hvp: bool = False,
                   scale: float = 1.0, offset: float = 1.0e12,
                   initial: float = 0.2):
    control = torch.zeros(26, dtype=torch.float64)
    control[0] = initial
    parameters = torch.zeros(13, dtype=torch.float64)
    truth = torch.zeros(1, dtype=torch.float64)
    original = torch.zeros(1, dtype=torch.float64)
    problem = _Quadratic(scale, offset)
    parameter_sha = tangent._tensor_sha(parameters)
    truth_sha = "truth-fixed"
    base_identity = {"control_sha256": tangent._tensor_sha(control),
        "parameters_sha256": parameter_sha, "terminal_truth_sha256": truth_sha,
        "archived_input": {"profile": "same"}}

    def input_identity(_problem, _original, value, _params, _truth):
        return {**base_identity, "control_sha256": tangent._tensor_sha(value)}

    branch = {"status": "passed_strict_branch", "signature_sha256": "fixed-branch",
              "euler_stages": 3600, "choice_stage_count": 3600, "face_sign_stage_count": 3600}
    def measure(_problem, _params, value, _eta, gradient_fn, *, with_gradient, deadline):
        objective = problem.objective(value, _params)
        gradient = gradient_fn(value, _params)
        phi = torch.dot(gradient, gradient) / 2
        return {"objective": float(objective), "phi": float(phi), "gradient": gradient.tolist(),
            "gradient_inf": float(gradient.abs().max()), "gradient_norm": float(torch.linalg.vector_norm(gradient)),
            "branch": branch, "branch_partition": {"full_sha256": "fixed-branch"},
            "geometry": {}, "branch_margins": {"complete": True}, "branch_scope": "test",
            "branch_signature": {"choices": [], "face_signs": []}, "static_flux_signs": {}}

    fresh_gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    state = {"objective": float(problem.objective(control, parameters)),
             "phi": float(torch.dot(fresh_gradient, fresh_gradient) / 2),
             "gradient": fresh_gradient.tolist(), "gradient_inf": float(fresh_gradient.abs().max()),
             "branch": branch}
    trial = {"status": "accepted", "control_sha256": tangent._tensor_sha(control),
             "objective": state["objective"], "phi": state["phi"],
             "gradient": state["gradient"], "gradient_inf": state["gradient_inf"],
             "J_armijo_passed": True, "Phi_armijo_passed": True,
             "branch_signature_sha256": "fixed-branch", "branch": branch}
    raw = {"accepted_control": control.tolist(), "accepted_control_sha256": tangent._tensor_sha(control),
           "parameters_sha256": parameter_sha, "input_after": base_identity,
           "runtime": {"runtime": "fixed"}, "current_state": state,
           "accepted_objective": state["objective"], "accepted_phi": state["phi"],
           "accepted_gradient": state["gradient"], "trials": [trial]}
    monkeypatch.setattr(guided, "CONTROL_SHA", tangent._tensor_sha(control))
    monkeypatch.setattr(guided, "PARAMETERS_SHA", parameter_sha)
    monkeypatch.setattr(guided, "_load_plan", lambda *_args: {"source_files": {}, "archive_files": {}})
    monkeypatch.setattr(guided, "_load_base", lambda: raw)
    monkeypatch.setattr(guided.seed, "_prepare_fixed_seed", lambda: (
        problem, original, control, parameters, truth, base_identity))
    monkeypatch.setattr(guided.seed, "_input_identity", input_identity)
    monkeypatch.setattr(guided.qy, "production_qy32", lambda *_args: 0.0)
    monkeypatch.setattr(guided.qy, "_measure", measure)
    monkeypatch.setattr(guided, "_static_zero_faces", lambda *_args: [])
    monkeypatch.setattr(guided.guard_policy.blocks, "runtime_identity", lambda: {"runtime": "fixed"})
    plan_sha = "p" * 64
    monkeypatch.setattr(guided, "_sha", lambda path: plan_sha if Path(path) == guided.PLAN
                        else hashlib.sha256(Path(path).read_bytes()).hexdigest())
    if fail_second_hvp:
        live = guided._current_hvp
        count = 0
        def fail_after_one(gradient_fn, c, p, d, deadline, on_start, on_complete):
            nonlocal count
            count += 1
            if count == 2:
                raise TimeoutError("synthetic second-iteration budget")
            return live(gradient_fn, c, p, d, deadline, on_start, on_complete)
        monkeypatch.setattr(guided, "_current_hvp", fail_after_one)
    return control, raw, plan_sha


def test_continuation_uses_each_new_point_gradient_and_hvp(monkeypatch, tmp_path):
    control, _, plan_sha = _blackbox_case(monkeypatch)
    output = tmp_path / "step.json"
    result = guided._run_child(guided.PLAN, plan_sha, output)
    assert result["optimizer_steps_applied"] == 3
    assert result["hvp_calls"] == result["hvp_calls_completed"] == 3
    assert len(result["iterations"]) == 3
    for index, iteration in enumerate(result["iterations"]):
        expected_gradient = torch.tensor(
            _blackbox_initial_gradient(control) if index == 0
            else result["iterations"][index - 1]["accepted_gradient"], dtype=torch.float64)
        assert torch.allclose(torch.tensor(iteration["direction"], dtype=torch.float64), -expected_gradient)
        assert torch.allclose(torch.tensor(iteration["H_direction"], dtype=torch.float64), -expected_gradient)
    assert result["candidate_committed"] is True


def _blackbox_initial_gradient(control: torch.Tensor) -> list[float]:
    return control.tolist()  # Identity Hessian for the mock quadratic.


def test_second_iteration_budget_preserves_first_committed_point_and_hvp_count(
        monkeypatch, tmp_path):
    _, _, plan_sha = _blackbox_case(monkeypatch, fail_second_hvp=True)
    result = guided._run_child(guided.PLAN, plan_sha, tmp_path / "partial.json")
    assert result["numerical_status"] == "budget_refusal"
    assert result["optimizer_steps_applied"] == 1
    assert result["hvp_calls"] == result["hvp_calls_completed"] == 1
    assert result["candidate_committed"] is True
    assert result["current_control_sha256"] == result["iterations"][0]["accepted_control_sha256"]
    assert result["current_state"]["gradient"] == result["iterations"][0]["accepted_gradient"]
    assert result["active_iteration"]["hvp_calls_started"] == 0


def test_large_phi_increase_backtracks_to_smaller_candidate(monkeypatch, tmp_path):
    control, _, plan_sha = _blackbox_case(monkeypatch)
    actual_measure = guided.qy._measure
    monkeypatch.setattr(guided, "model_alpha", lambda *_args: 0.2)

    def nonlinear_measure(problem, params, candidate, eta, gradient_fn, *, with_gradient, deadline):
        result = actual_measure(problem, params, candidate, eta, gradient_fn,
                                with_gradient=with_gradient, deadline=deadline)
        if float(torch.linalg.vector_norm(candidate - control)) > 0.03:
            result["phi"] = 0.125  # The larger candidate increases actual Phi.
        return result

    monkeypatch.setattr(guided.qy, "_measure", nonlinear_measure)
    result = guided._run_child(guided.PLAN, plan_sha, tmp_path / "backtrack.json")
    trials = result["iterations"][0]["trials"]
    assert trials[0]["phi"] > result["iterations"][0]["base_phi"]
    assert trials[0]["Phi_armijo_passed"] is False
    assert trials[1]["status"] == "accepted"
    assert trials[1]["alpha"] == pytest.approx(trials[0]["alpha"] / 2)


def test_near_root_uses_gradient_threshold_without_an_objective_floor(monkeypatch, tmp_path):
    control, _, plan_sha = _blackbox_case(monkeypatch, scale=1e-8, offset=1.0, initial=1e-4)
    result = guided._run_child(guided.PLAN, plan_sha, tmp_path / "root-pending.json")
    assert result["numerical_status"] == "root_pending_audit"
    assert result["root_pending_audit"] is True
    assert result["full_root_claim"] is False
    assert result["hvp_calls"] == 0
    assert result["optimizer_steps_applied"] == 0
    assert result["current_control"] == control.tolist()
    assert result["current_state"]["objective"] == 1.0
