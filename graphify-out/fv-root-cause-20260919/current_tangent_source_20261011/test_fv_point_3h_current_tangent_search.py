from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_current_tangent_search as search
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


DTYPE = torch.float64


def _repeat_and_proposal() -> tuple[dict[str, Any], dict[str, Any], Tensor, dict[str, Any]]:
    control = torch.zeros(26, dtype=DTYPE)
    minus = torch.zeros_like(control)
    plus = torch.zeros_like(control)
    minus[:2] = torch.tensor([-1.0, 0.5], dtype=DTYPE)
    plus[:2] = torch.tensor([1.0, 0.5], dtype=DTYPE)
    sides = {"-1": minus.tolist(), "1": plus.tolist()}
    weights: Tensor = torch.tensor([0.0, -0.08, -0.28, -0.14, -0.84], dtype=DTYPE)
    repeated_G = torch.cat(((minus + plus) * 0.5, torch.zeros(1, dtype=DTYPE)))
    repeated_R = float(torch.dot(repeated_G, repeated_G))
    repeat: dict[str, Any] = {
        "control": control.tolist(),
        "theta": 0.5,
        "objective": 1.0,
        "F_squared": repeated_R,
        "side_gradients": sides,
        "side_gradients_finite": True,
        "native_objective_matches_proposal": True,
        "side_objectives_match_native": True,
        "merit_matches_proposal": True,
        "face_audit_passed": True,
        "branch_pair_passed": True,
        "trace_matches_proposal": True,
        "source_unchanged": True,
        "fixed_input_unchanged": True,
        "runtime_unchanged": True,
        "deadline_passed": True,
    }
    full_g = search.repeat_full_G(repeat, repeated_G.tolist(), weights)
    assert full_g["passed"]
    repeat["F_squared"] = float(torch.dot(torch.tensor(full_g["G"], dtype=DTYPE),
                                          torch.tensor(full_g["G"], dtype=DTYPE)))
    proposal: dict[str, Any] = {
        "control": control.tolist(),
        "control_sha256": tangent._tensor_sha(control),
        "theta": 0.5,
        "objective": 1.0,
        "F_squared": repeat["F_squared"],
        "side_gradients": sides,
        "J_armijo_passed": True,
        "F_squared_armijo_passed": True,
        "J_armijo_bound": 1.1,
        "F_squared_armijo_bound": 0.3,
    }
    return repeat, proposal, weights, full_g


def test_candidate_selection_uses_only_accepted_R_then_J_then_declared_order():
    arms = [
        {"name": "rejected-low-R", "order": 0,
         "candidate": {"accepted": False, "F_squared": 0.01, "objective": 0.1}},
        {"name": "higher-R", "order": 1,
         "candidate": {"accepted": True, "F_squared": 0.3, "objective": 0.5}},
        {"name": "lower-R", "order": 2,
         "candidate": {"accepted": True, "F_squared": 0.2, "objective": 0.9}},
        {"name": "R-tie-lower-J", "order": 3,
         "candidate": {"accepted": True, "F_squared": 0.2, "objective": 0.8}},
    ]
    chosen = search.choose_candidate(arms)
    assert chosen is arms[3]
    assert search.choose_candidate([arms[0]]) is None


def _trace179(choose_left: bool = False, *, left_sign: int = -1,
              right_sign: int = -1, slope_sign: int = -1) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "choose_left": [[False] * 3 for _ in range(2)],
        "left_sign": [[-1] * 3 for _ in range(2)],
        "right_sign": [[-1] * 3 for _ in range(2)],
        "slope_sign": [[-1] * 3 for _ in range(2)],
    }
    choice["choose_left"][1][2] = choose_left
    choice["left_sign"][1][2] = left_sign
    choice["right_sign"][1][2] = right_sign
    choice["slope_sign"][1][2] = slope_sign
    choices = [[copy.deepcopy(choice), copy.deepcopy(choice)] for _ in range(360)]
    return {"choices": choices}


def test_current_event_trace_binding_accepts_matching_trace179_selector():
    values = torch.tensor([-0.4, -0.3, 1.0], dtype=DTYPE)
    assert search.event_trace_binding(values, _trace179(choose_left=False)) is True


@pytest.mark.parametrize("values,trace", [
    (torch.tensor([-0.4, -0.3, 1.0], dtype=DTYPE), _trace179(choose_left=True)),
    (torch.tensor([-0.3, -0.4, 1.0], dtype=DTYPE), _trace179(choose_left=False)),
    (torch.tensor([-0.35, -0.35, 1.0], dtype=DTYPE), _trace179(choose_left=False)),
    (torch.tensor([float("nan"), -0.3, 1.0], dtype=DTYPE), _trace179(choose_left=False)),
    (torch.tensor([-0.4, -0.3, 1.0], dtype=DTYPE),
     _trace179(choose_left=False, left_sign=1)),
])
def test_current_event_trace_binding_rejects_selector_sign_tie_or_nonfinite_mismatch(values, trace):
    assert search.event_trace_binding(values, trace) is False


def test_repeat_full_G_checks_components_not_only_equal_merit():
    repeat, _, weights, full_g = _repeat_and_proposal()
    matched = search.repeat_full_G(repeat, full_g["G"], weights)
    assert matched["passed"] is True
    assert matched["merit_match"] is True

    rotated = list(full_g["G"])
    rotated[0], rotated[1] = rotated[1], rotated[0]
    assert sum(value * value for value in rotated) == pytest.approx(repeat["F_squared"])
    rejected = search.repeat_full_G(repeat, rotated, weights)
    assert rejected["merit_match"] is True
    assert rejected["component_match"]["passed"] is False
    assert rejected["passed"] is False


def test_successful_commit_closes_P2_and_records_current_last_confirmed_and_full_G(tmp_path):
    repeat, proposal, _, full_g = _repeat_and_proposal()
    output = tmp_path / "step.json"
    prefix = {"current_control": [0.1] * 26, "current_control_sha256": "old-prefix",
        "last_confirmed_control": [0.1] * 26, "candidate_committed": False,
        "optimizer_steps_applied": 0, "future_score_observation_only": True}
    committed = search.persist_commit(output, prefix, proposal, repeat, full_g)

    assert committed["candidate_committed"] is True
    assert committed["optimizer_steps_applied"] == 1
    assert committed["postcommit_readiness_performed"] is False
    assert committed["current_control"] == repeat["control"]
    assert committed["last_confirmed_control"] == repeat["control"]
    assert committed["current_full_G"] == full_g["G"]
    assert committed["last_confirmed_full_G"] == full_g["G"]
    assert prefix["current_control_sha256"] == "old-prefix"
    assert json.loads(output.read_text())["current_control_sha256"] == tangent._tensor_sha(
        torch.tensor(repeat["control"], dtype=DTYPE))


@pytest.mark.parametrize("failure", [
    "repeat_J_above_bound",
    "repeat_R_above_bound",
    "repeat_J_mismatch",
    "repeat_R_mismatch",
    "full_G_component_mismatch",
    "full_G_merit_mismatch",
])
def test_failed_repeat_or_full_G_write_preserves_closed_prefix_and_input(tmp_path, monkeypatch, failure):
    repeat, proposal, weights, full_g = _repeat_and_proposal()
    output = tmp_path / "step.json"
    prefix = {"current_control": [0.1] * 26, "current_control_sha256": "closed-base",
        "last_confirmed_control": [0.1] * 26, "candidate_committed": False,
        "optimizer_steps_applied": 0, "input_sha256": "fixed-input"}
    tangent._write(output, prefix)
    before = output.read_bytes()
    original = copy.deepcopy(prefix)

    if failure == "repeat_J_above_bound":
        repeat["objective"] = proposal["J_armijo_bound"] + 1e-4
    elif failure == "repeat_R_above_bound":
        proposal["F_squared_armijo_bound"] = repeat["F_squared"] - 1e-4
    elif failure == "repeat_J_mismatch":
        repeat["native_objective_matches_proposal"] = False
    elif failure == "repeat_R_mismatch":
        repeat["merit_matches_proposal"] = False
    elif failure == "full_G_component_mismatch":
        rotated = list(full_g["G"])
        rotated[0], rotated[1] = rotated[1], rotated[0]
        full_g = search.repeat_full_G(repeat, rotated, weights)
    else:
        full_g = {**full_g, "merit_match": False, "passed": False}

    with pytest.raises(ValueError):
        search.persist_commit(output, prefix, proposal, repeat, full_g)
    assert output.read_bytes() == before
    assert prefix == original
    assert list(tmp_path.glob("step.json.*.tmp")) == []


def test_atomic_writer_failure_leaves_old_confirmed_prefix_and_inputs(tmp_path, monkeypatch):
    repeat, proposal, _, full_g = _repeat_and_proposal()
    output = tmp_path / "step.json"
    prefix = {"current_control": [0.1] * 26, "current_control_sha256": "closed-base",
        "last_confirmed_control": [0.1] * 26, "candidate_committed": False,
        "optimizer_steps_applied": 0, "input_sha256": "fixed-input"}
    tangent._write(output, prefix)
    before = output.read_bytes()
    original = copy.deepcopy(prefix)

    def fail_replace(_source, _target):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(tangent.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        search.persist_commit(output, prefix, proposal, repeat, full_g)
    assert output.read_bytes() == before
    assert prefix == original
    assert list(tmp_path.glob("step.json.*.tmp")) == []
