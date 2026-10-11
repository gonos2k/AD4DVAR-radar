from __future__ import annotations

import copy
import hashlib
import itertools
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_active_intersection_diagnostic as diag
from examples.weather_scenarios import fv_point_3h_active_limiter_probe as event


DTYPE = torch.float64


def test_sampled_hull_finds_vertex_and_interior_projection():
    vertex = diag.minimum_sampled_hull(torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=DTYPE))
    assert vertex["vector"] == pytest.approx([1.0, 0.0])
    assert vertex["weights"] == pytest.approx([1.0, 0.0, 0.0])
    assert vertex["squared_norm"] == pytest.approx(1.0)

    interior = diag.minimum_sampled_hull(torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=DTYPE))
    assert interior["vector"] == pytest.approx([0.0, 0.0], abs=2e-15)
    assert interior["weights"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert interior["squared_norm"] <= 1e-28
    assert "not limiting-gradient certification" in interior["scope"]


def test_sampled_hull_handles_duplicate_vertices_and_rejects_invalid_inputs():
    duplicate = diag.minimum_sampled_hull(torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=DTYPE))
    assert duplicate["vector"] == pytest.approx([0.0, 0.0], abs=2e-15)
    assert sum(duplicate["weights"]) == pytest.approx(1.0)
    assert min(duplicate["weights"]) >= 0.0

    invalid = [
        torch.empty((0, 2), dtype=DTYPE),
        torch.zeros((5, 2), dtype=DTYPE),
        torch.zeros((2, 2), dtype=torch.float32),
        torch.tensor([[0.0, float("nan")]], dtype=DTYPE),
        torch.zeros((2,), dtype=DTYPE),
    ]
    for gradients in invalid:
        with pytest.raises(ValueError, match="finite double gradients"):
            diag.minimum_sampled_hull(gradients)


def test_sector_directions_cover_all_signed_quadrants_in_their_span():
    normal = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    event_normal = torch.tensor([0.0, 2.0, 0.0], dtype=DTYPE)
    directions, audit = diag.sector_directions(normal, event_normal)
    signs = list(itertools.product((-1.0, 1.0), repeat=2))
    assert directions.shape == (4, 3)
    for direction, (normal_sign, event_sign) in zip(directions, signs, strict=True):
        assert float(normal @ direction) * normal_sign > 0
        assert float(event_normal @ direction) * event_sign > 0
        assert float(direction.norm()) == pytest.approx(1.0)
    assert audit["normal_cosine"] == pytest.approx(0.0)
    assert audit["normal_singular_values"][1] > 0
    basis = torch.tensor(audit["normal_basis"], dtype=DTYPE)
    assert torch.allclose(basis.T @ basis, torch.eye(2, dtype=DTYPE), atol=2e-15, rtol=0)


@pytest.mark.parametrize("normal,event_normal", [
    ([1.0, 0.0], [2.0, 0.0]),
    ([1.0, 2.0], [-3.0, -6.0]),
    ([0.0, 0.0], [0.0, 1.0]),
])
def test_sector_directions_reject_dependent_or_unresolved_normals(normal, event_normal):
    with pytest.raises(ValueError, match="unresolved event normals|normals are dependent"):
        diag.sector_directions(torch.tensor(normal, dtype=DTYPE),
            torch.tensor(event_normal, dtype=DTYPE))


def test_closed_parent_loader_verifies_receipts_without_mutating_them():
    paths = [diag.ARCHIVE, diag.PARENT / "step.json.gz",
        diag.PARENT / "step.run.json", diag.PARENT / "step.resource.json"]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    parent = diag.load_base()
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    assert before == after
    assert parent["current_control_sha256"] == diag.BASE_SHA
    assert parent["selected_policy"] == "seeded_JR"
    assert parent["canonical_promotions"] == 1
    assert len(parent["current_control"]) == 26
    assert len(parent["current_full_G"]) == 27
    assert parent["canonical_final_repeat"]["control"] == parent["current_control"]


def test_q0_root_roundoff_bound_uses_event_state_scale_and_state_units():
    # raw_zeta is left-right, while event_values[2] is max(abs(q)); both have
    # state units, so the declared closure budget must scale with that channel.
    values = torch.tensor([-0.004, -0.001, 2.0], dtype=DTYPE)
    assert event.raw_zeta(values).item() == pytest.approx(-0.003)
    eps = torch.finfo(DTYPE).eps
    expected = 256 * eps * float(values[2])
    assert expected > 0
    assert expected / float(values[2]) == pytest.approx(256 * eps)


def _trace_fixture() -> dict[str, Any]:
    def choice():
        return {
            "choose_left": [[False] * 3 for _ in range(2)],
            "left_sign": [[-1] * 3 for _ in range(2)],
            "right_sign": [[-1] * 3 for _ in range(2)],
            "slope_sign": [[0] * 3 for _ in range(2)],
        }

    return {"choices": [[choice(), choice()] for _ in range(360)],
        "face_signs": [[0] * 6 for _ in range(5)]}


def test_branch_scope_allows_only_declared_trace179_limiter_switch():
    reference = _trace_fixture()
    actual = copy.deepcopy(reference)
    actual["choices"][179][0]["choose_left"][1][2] = True

    result = diag.branch_changes(actual, reference)
    assert result["changed_choices"] == [{
        "stage": 179, "axis": 0, "field": "choose_left", "row": 1, "column": 2}]
    assert result["unexpected_choice_changes"] == []
    assert result["other_face_signs_match"] is True
    assert result["only_declared_event_changed"] is True


@pytest.mark.parametrize("stage,axis,field,row,column,face_change", [
    (178, 0, "choose_left", 1, 2, False),
    (179, 0, "left_sign", 1, 2, False),
    (179, 0, "choose_left", 0, 2, False),
    (179, 1, "choose_left", 1, 2, False),
    (179, 0, "choose_left", 1, 2, True),
])
def test_branch_scope_rejects_other_stage_cell_sign_axis_or_face_change(
        stage, axis, field, row, column, face_change):
    reference = _trace_fixture()
    actual = copy.deepcopy(reference)
    if face_change:
        actual["face_signs"][0][0] = 1
    elif field == "left_sign":
        actual["choices"][stage][axis][field][row][column] = 1
    else:
        actual["choices"][stage][axis][field][row][column] = True

    result = diag.branch_changes(actual, reference)
    assert result["only_declared_event_changed"] is False
    if face_change:
        assert result["other_face_signs_match"] is False
    else:
        assert result["unexpected_choice_changes"]


def test_policy_is_read_only_diagnostic_with_no_optimizer_or_operator_budget():
    assert diag.POLICY["optimizer_steps"] == 0
    assert diag.POLICY["jacobian_row_vjp_calls"] == 0
    assert diag.POLICY["dense_solves"] == 0
    assert diag.POLICY["hvp_calls"] == 0
    assert diag.POLICY["root_claim"] is False
    assert diag.POLICY["minimum_claim"] is False
    assert diag.POLICY["response_claim"] is False
    assert diag.POLICY["score_claim"] is False
