from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_event_direction_comparison as compare
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def test_euclidean_projection_removes_event_component_without_rescaling():
    direction = torch.zeros(26, dtype=torch.float64)
    event_gradient = torch.zeros_like(direction)
    direction[:2] = torch.tensor([3.0, 4.0])
    event_gradient[0] = 2.0

    corrected = compare._project_away(direction, event_gradient)

    assert torch.equal(corrected[:2], torch.tensor([0.0, 4.0], dtype=direction.dtype))
    assert torch.equal(corrected[2:], direction[2:])
    assert torch.dot(event_gradient, corrected) == 0
    assert torch.linalg.vector_norm(corrected) == 4.0
    assert torch.linalg.vector_norm(corrected) < torch.linalg.vector_norm(direction)

    with pytest.raises(ValueError, match="zero direction"):
        compare._project_away(event_gradient, event_gradient)
    with pytest.raises(ValueError, match="unresolved Euclidean norm"):
        compare._project_away(direction, torch.zeros_like(direction))


def test_common_event_normal_reprojects_both_saved_sides_and_checks_component_scales():
    normal = torch.zeros(26, dtype=torch.float64)
    normal[25] = 1.0
    tangent_component = torch.zeros_like(normal)
    tangent_component[3], tangent_component[8] = 1.25, -0.75
    minus = tangent_component.clone()
    plus = tangent_component.clone()
    minus[25] += 2.0
    plus[25] -= 3.0
    receipt = {"event_derivatives": {
        "-1": {"gradient": minus.tolist(), "tangent_gradient": tangent_component.tolist()},
        "1": {"gradient": plus.tolist(), "tangent_gradient": tangent_component.tolist()},
    }}

    common, audit = compare._common_event_tangent_gradient(receipt, normal)

    assert torch.equal(common, tangent_component)
    assert audit["side_projection_agreement"]["passed"] is True
    assert audit["averaging_rule"].startswith("arithmetic mean")


def test_original_j_and_residual_merit_bounds_use_the_active_arm_direction():
    base_j, base_r, alpha = 1.0, 1.0, 0.1
    baseline_bounds = compare._armijo_bounds(base_j, base_r, alpha, (-2.0, -1.0), -1.0)
    corrected_bounds = compare._armijo_bounds(base_j, base_r, alpha, (-4.0, -3.0), -3.0)
    actual_j, actual_r = 0.99998, 0.99996

    assert actual_j <= baseline_bounds[0]
    assert actual_r <= baseline_bounds[1]
    assert actual_j > corrected_bounds[0]
    assert actual_r > corrected_bounds[1]


def test_arm_selection_orders_actual_merit_then_cost_then_baseline_arm():
    candidate = lambda r, j: {"F_squared": r, "objective": j}
    results = {
        "prepared_gn": {"accepted": candidate(0.9, 0.8), "search": {"first_pass_count": 7}},
        "event_orthogonal": {"accepted": candidate(0.8, 1.2), "search": {"first_pass_count": 4}},
    }
    selected = compare._select_arm_candidate(results, torch.float64)
    assert selected is not None and selected["arm"] == "event_orthogonal"

    tied_merit = {
        "prepared_gn": {"accepted": candidate(1.0, 0.9)},
        "event_orthogonal": {"accepted": candidate(1.0 + 32 * torch.finfo(torch.float64).eps, 0.8)},
    }
    selected = compare._select_arm_candidate(tied_merit, torch.float64)
    assert selected is not None and selected["arm"] == "event_orthogonal"

    exact_tie = {
        "prepared_gn": {"accepted": candidate(1.0, 0.8)},
        "event_orthogonal": {"accepted": candidate(1.0, 0.8)},
    }
    selected = compare._select_arm_candidate(exact_tie, torch.float64)
    assert selected is not None and selected["arm"] == "prepared_gn"
    assert compare._select_arm_candidate({"prepared_gn": {"accepted": None}}, torch.float64) is None

    fallback = {
        "prepared_gn": {"supported": False, "accepted": None, "search": {}},
        "event_orthogonal": {"supported": True, "accepted": candidate(0.7, 0.9),
            "search": {"first_pass_count": 11}},
    }
    selected = compare._select_arm_candidate(fallback, torch.float64)
    assert selected is not None and selected["arm"] == "event_orthogonal"
    assert selected["comparison_complete"] is False
    assert selected["selection_basis"] == "single_supported_arm_fallback"
    assert compare._per_arm_first_pass_counts(fallback) == {"prepared_gn": 0, "event_orthogonal": 11}


def _closed_proposal_repeat() -> tuple[dict[str, Any], dict[str, Any]]:
    control = torch.linspace(-0.2, 0.3, 26, dtype=torch.float64)
    minus, plus = torch.zeros_like(control), torch.zeros_like(control)
    minus[0], plus[0] = 1.0, -1.0
    side_gradients = {"-1": minus.tolist(), "1": plus.tolist()}
    proposal = {"control": control.tolist(), "control_sha256": tangent._tensor_sha(control),
        "theta": 0.5, "objective": 1.0, "F_squared": 0.25,
        "candidate_mixing_minimum": True, "alpha": 6.25e-7,
        "side_gradients": side_gradients}
    repeat = {"control": control.tolist(), "theta": 0.5, "objective": 1.0,
        "F_squared": 0.25, "side_gradients": side_gradients,
        **{key: True for key in (
            "mixing_minimum_valid", "minimum_theta_matches_proposal", "side_gradients_finite",
            "native_objective_matches_proposal", "side_objectives_match_native", "merit_matches_proposal",
            "face_audit_passed", "branch_pair_passed", "trace_matches_proposal",
            "source_unchanged", "fixed_input_unchanged", "runtime_unchanged", "deadline_passed")}}
    return proposal, repeat


def test_selected_candidate_p2_failure_preserves_base_and_commit_records_one_step(tmp_path):
    proposal, repeat = _closed_proposal_repeat()
    path = tmp_path / "step.json"
    base = {"current_control_sha256": compare.BASE_SHA, "optimizer_steps_applied": 0,
        "candidate_committed": False}
    tangent._write(path, base)
    bad_repeat = {**repeat, "runtime_unchanged": False}
    with pytest.raises(ValueError, match="P2"):
        compare._persist_selected_commit(path, base, proposal, bad_repeat)
    assert json.loads(path.read_text()) == base

    committed = compare._persist_selected_commit(path, base, proposal, repeat)
    saved = json.loads(path.read_text())
    assert committed["candidate_committed"] is True
    assert committed["optimizer_steps_applied"] == committed["accepted_iterations"] == 1
    assert committed["postcommit_candidate_alpha"] == proposal["alpha"]
    assert committed["postcommit_readiness_performed"] is False
    assert saved["last_confirmed_control_sha256"] == proposal["control_sha256"]
    assert saved["final_repeat"] == repeat


def test_atomic_commit_write_failure_leaves_original_027_record(tmp_path, monkeypatch):
    proposal, repeat = _closed_proposal_repeat()
    path = tmp_path / "step.json"
    base = {"current_control_sha256": compare.BASE_SHA, "optimizer_steps_applied": 0,
        "candidate_committed": False}
    tangent._write(path, base)

    def fail(_path: Path, _value: dict[str, Any]) -> None:
        raise OSError("atomic replace failed")

    monkeypatch.setattr(compare.tangent, "_write", fail)
    with pytest.raises(OSError, match="atomic replace"):
        compare._persist_selected_commit(path, base, proposal, repeat)
    assert json.loads(path.read_text()) == base
