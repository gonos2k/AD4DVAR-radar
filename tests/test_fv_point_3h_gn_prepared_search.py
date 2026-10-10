from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def test_search_uses_24_dyadic_slots_and_stops_at_first_later_pass(monkeypatch):
    alphas = [0.04 / (2**index) for index in range(24)]
    seen_limits: list[int] = []
    monkeypatch.setattr(prepared.tangent, "candidate_alphas",
        lambda _model, *, radius, limit: seen_limits.append(limit) or alphas[:limit])
    control = torch.zeros(2, dtype=torch.float64)
    model = SimpleNamespace(direction=torch.tensor([1.0, 0.0]), theta=0.5)
    evaluated: list[int] = []

    def evaluate(candidate: torch.Tensor, theta_prediction: float, alpha: float) -> dict[str, Any]:
        index = round(torch.linalg.vector_norm(candidate).item() / alphas[0] * (2**17))
        evaluated.append(len(evaluated) + 1)
        passed = len(evaluated) == 18
        return {"mixing_minimum_valid": passed, "J_armijo_passed": passed,
            "F_squared_armijo_passed": passed, "branch_pair_passed": passed,
            "face_audit_passed": passed, "side_objectives_match_native": passed,
            "side_gradients_finite": passed, "finite": passed, "theta": 0.4,
            "objective": 1.0, "F_squared": 1.0, "J_armijo_bound": 1.1,
            "F_squared_armijo_bound": 1.1}

    accepted, trials, meta = prepared._search_candidates(model, control, evaluate,
        chart_candidate=lambda point, direction, alpha: point + alpha * direction)

    assert seen_limits == [24]
    assert accepted is not None
    assert len(trials) == len(evaluated) == 18
    assert accepted["index"] == meta["first_pass_candidate_index"] == 18
    assert meta["first_pass_count"] == meta["evaluated_count"] == 18
    assert meta["slot_cap"] == meta["computed_alpha_count"] == 24
    assert meta["computed_alpha0"] == alphas[0]
    assert meta["minimum_computed_alpha"] == alphas[-1]


def test_actual_chart_radius_and_zero_movement_are_refused_before_evaluation(monkeypatch):
    monkeypatch.setattr(prepared.tangent, "candidate_alphas",
        lambda _model, *, radius, limit: [0.1, 0.05])
    control = torch.zeros(2, dtype=torch.float64)
    model = SimpleNamespace(direction=torch.tensor([1.0, 0.0]), theta=0.5)
    calls: list[float] = []
    accepted, trials, _ = prepared._search_candidates(model, control,
        lambda _point, _theta, alpha: calls.append(alpha) or {"mixing_minimum_valid": False},
        chart_candidate=lambda point, direction, alpha: point + 2.0 * alpha * direction,
        radius=0.05)

    assert accepted is None
    assert calls == []
    assert [trial["status"] for trial in trials] == [
        "actual_chart_radius_refused", "actual_chart_radius_refused"]

    _, zero_trials, _ = prepared._search_candidates(model, control,
        lambda *_: pytest.fail("zero movement must be rejected before evaluation"),
        chart_candidate=lambda point, _direction, _alpha: point.clone())
    assert zero_trials[0]["status"] == "zero_control_movement_refused"


def test_unusable_model_returns_empty_completed_search_window(monkeypatch):
    monkeypatch.setattr(prepared.tangent, "candidate_alphas",
        lambda _model, *, radius, limit: [])
    model = SimpleNamespace(direction=torch.zeros(2), theta=0.5)
    accepted, trials, meta = prepared._search_candidates(model, torch.zeros(2),
        lambda *_: pytest.fail("an empty alpha window must not evaluate a candidate"),
        chart_candidate=lambda point, _direction, _alpha: point)
    assert accepted is None
    assert trials == []
    assert meta["evaluated_count"] == meta["computed_alpha_count"] == 0
    assert meta["computed_alpha0"] is None
    assert meta["minimum_computed_alpha"] is None


def test_original_armijo_coefficients_and_unclipped_theta_gate(monkeypatch):
    j_bound, f2_bound = prepared._armijo_bounds(2.0, 0.5, 0.25, -4.0, -3.0)
    assert j_bound == 2.0 - 1e-4
    assert f2_bound == 0.5 - 1.5e-4

    control = torch.zeros(2, dtype=torch.float64)
    model = SimpleNamespace(direction=torch.tensor([1.0, 0.0]), theta=0.5)
    monkeypatch.setattr(prepared.tangent, "candidate_alphas",
        lambda _model, *, radius, limit: [0.01])
    trial_theta_values: list[float] = []
    facts = {"mixing_minimum_valid": False, "J_armijo_passed": True,
        "F_squared_armijo_passed": True, "branch_pair_passed": True,
        "face_audit_passed": True, "side_objectives_match_native": True,
        "side_gradients_finite": True, "finite": True, "theta": 0.0}
    accepted, trials, _ = prepared._search_candidates(model, control,
        lambda _point, theta_prediction, _alpha: trial_theta_values.append(theta_prediction) or facts,
        chart_candidate=lambda point, direction, alpha: point + alpha * direction)
    assert accepted is None
    assert trials[0]["accepted"] is False
    assert trial_theta_values[0] == model.theta
    assert trials[0]["theta"] == 0.0


def test_saved_current_point_archive_is_admitted_without_fv_setup():
    identity = prepared._archive_identity()
    assert identity["raw_sha256"] == prepared.COMMIT_RAW_SHA
    assert identity["audit"]["P2_closed"] is True
    assert identity["audit"]["postcommit_readiness_complete"] is True
    assert identity["raw"]["postcommit_gn_readiness"]["direction_sha256"] == prepared.BASE_DIRECTION_SHA


def _proposal_and_repeat() -> tuple[dict[str, Any], dict[str, Any]]:
    minus = [0.0] * 26
    plus = [0.0] * 26
    minus[0], plus[0] = 1.0, -1.0
    proposal = {"control": [0.0] * 26, "control_sha256": tangent._tensor_sha(
        torch.zeros(26, dtype=torch.float64)), "theta": 0.5, "objective": 1.0,
        "F_squared": 0.25, "candidate_mixing_minimum": True, "alpha": 1e-6,
        "J_armijo_passed": True, "F_squared_armijo_passed": True,
        "face_audit_passed": True, "branch_pair_passed": True,
        "side_objectives_match_native": True, "side_gradients_finite": True,
        "side_gradients": {"-1": minus, "1": plus}, "branch_trace": {"-1": {}, "1": {}}}
    repeat = {"control": proposal["control"], "theta": 0.5, "objective": 1.0,
        "F_squared": 0.25, "side_gradients": {"-1": minus, "1": plus},
        **{key: True for key in ("mixing_minimum_valid", "minimum_theta_matches_proposal",
            "side_gradients_finite", "native_objective_matches_proposal",
            "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
            "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
            "fixed_input_unchanged", "runtime_unchanged", "deadline_passed")}}
    return proposal, repeat


def test_p2_side_cancellation_rejects_commit_and_atomic_failure_keeps_base(tmp_path, monkeypatch):
    proposal, repeat = _proposal_and_repeat()
    repeat["side_gradients"] = {"-1": [0.0] * 26, "1": [0.0] * 26}
    path = tmp_path / "step.json"
    base = {"current_control_sha256": prepared.BASE_CONTROL_SHA,
        "current_theta": prepared.BASE_THETA, "optimizer_steps_applied": 0,
        "candidate_committed": False}
    prepared.gni._atomic_write(path, base)
    with pytest.raises(ValueError, match="P2 final closure"):
        prepared._persist_closed_commit(path, base, proposal, repeat)
    assert json.loads(path.read_text()) == base

    proposal, repeat = _proposal_and_repeat()
    def fail_write(_path: Path, _value: dict[str, Any]) -> None:
        raise OSError("atomic replacement refused")
    monkeypatch.setattr(prepared.gni, "_atomic_write", fail_write)
    with pytest.raises(OSError, match="atomic replacement"):
        prepared._persist_closed_commit(path, base, proposal, repeat)
    assert json.loads(path.read_text()) == base


def test_persisted_proposal_keeps_selected_alpha_instead_of_fixed_predecessor_alpha(tmp_path):
    proposal, repeat = _proposal_and_repeat()
    path = tmp_path / "step.json"
    committed = prepared._persist_closed_commit(path, {"optimizer_steps_applied": 0}, proposal, repeat)
    assert committed["postcommit_candidate_alpha"] == proposal["alpha"]
    assert committed["postcommit_candidate_alpha"] != prepared.gni.ALPHA
    assert committed["current_control_sha256"] == proposal["control_sha256"]
    failed = prepared.gni._persist_readiness_failure(path, committed, RuntimeError("readiness stopped"))
    disk = json.loads(path.read_text())
    assert failed["candidate_committed"] is True
    assert failed["readiness_complete"] is False
    assert disk["last_confirmed_control_sha256"] == proposal["control_sha256"]
    assert disk["final_repeat"] == repeat
