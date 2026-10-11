from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_current_gradient_continuation as continuation
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


DTYPE = torch.float64


def _closure() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    control = torch.zeros(26, dtype=DTYPE)
    minus = torch.zeros_like(control)
    plus = torch.zeros_like(control)
    minus[0], plus[0] = -0.5, 0.5
    minus[1] = plus[1] = 0.25
    sides = {"-1": minus.tolist(), "1": plus.tolist()}
    theta = 0.5
    full_g = torch.cat(((1 - theta) * minus + theta * plus, torch.zeros(1, dtype=DTYPE)))
    merit = float(torch.dot(full_g, full_g))
    repeat = {"control": control.tolist(), "theta": theta, "objective": 1.0,
        "F_squared": merit, "side_gradients": sides, "side_gradients_finite": True,
        "native_objective_matches_proposal": True, "side_objectives_match_native": True,
        "merit_matches_proposal": True, "face_audit_passed": True,
        "branch_pair_passed": True, "trace_matches_proposal": True,
        "source_unchanged": True, "fixed_input_unchanged": True,
        "runtime_unchanged": True, "deadline_passed": True}
    proposal = {"control": control.tolist(), "control_sha256": tangent._tensor_sha(control),
        "theta": theta, "objective": 1.0, "F_squared": merit,
        "side_gradients": sides, "J_armijo_passed": True,
        "F_squared_armijo_passed": True, "J_armijo_bound": 1.1,
        "F_squared_armijo_bound": merit + 1e-8}
    weights = torch.tensor([0.0, -0.08, -0.28, -0.14, -0.84], dtype=DTYPE)
    return repeat, proposal, {"G": full_g.tolist(), "weights": weights}


def test_archived_98ee_endpoint_load_is_receipt_only(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("archive admission must not prepare or evaluate FV")

    monkeypatch.setattr(continuation.seed, "_prepare_fixed_seed", forbidden)
    plan = {"archive_files": {path.relative_to(continuation.ROOT).as_posix():
        continuation._sha(path) for path in (continuation.ARCHIVE, continuation.BASE_GZIP,
            continuation.BASE_RUN, continuation.BASE_RESOURCE)}}
    base = continuation.load_current_base(plan)
    assert tangent._tensor_sha(base["control"]) == continuation.BASE_SHA
    assert base["raw"]["candidate_committed"] is True
    assert base["raw"]["optimizer_steps_applied"] == 1
    assert base["raw"]["current_full_G"] == base["raw"]["last_confirmed_full_G"]


def test_equal_residual_norm_does_not_hide_component_drift():
    repeat, _, facts = _closure()
    expected = torch.tensor(facts["G"], dtype=DTYPE)
    swapped = expected.clone()
    swapped[0], swapped[1] = expected[1], expected[0]
    assert torch.dot(expected, expected) == torch.dot(swapped, swapped)
    checked = continuation.fresh_full_G(repeat, swapped, facts["weights"])
    assert checked["merit_match"] is True
    assert checked["component_match"]["passed"] is False
    assert checked["passed"] is False


def test_failed_atomic_commit_preserves_confirmed_prefix(tmp_path, monkeypatch):
    repeat, proposal, facts = _closure()
    checked = continuation.fresh_full_G(repeat, torch.tensor(facts["G"], dtype=DTYPE), facts["weights"])
    output = tmp_path / "step.json"
    prefix = {"current_control": [0.1] * 26, "current_control_sha256": "old",
        "last_confirmed_control": [0.1] * 26, "optimizer_steps_applied": 0,
        "closed_steps": []}
    tangent._write(output, prefix)
    before, snapshot = output.read_bytes(), copy.deepcopy(prefix)

    def fail_replace(_source, _target):
        raise OSError("simulated atomic prefix write failure")

    monkeypatch.setattr(tangent.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic prefix write failure"):
        continuation.persist_commit(output, prefix, {"point_index": 0}, proposal, repeat, checked)
    assert output.read_bytes() == before
    assert prefix == snapshot
    assert list(tmp_path.glob("step.json.*.tmp")) == []


def test_progress_budget_and_small_residual_are_not_convergence_claims():
    assert continuation.POLICY["max_candidates"] == 72
    assert continuation.POLICY["hvp_calls_per_point"] == 2
    assert continuation.classify_no_candidate(residual_norm=1e-10, cap_hit=True) == \
        "residual_candidate_requires_qualification"
    assert continuation.classify_no_candidate(residual_norm=0.1, cap_hit=True) == "candidate_cap_reached"
    assert continuation.classify_no_candidate(residual_norm=0.1) == "search_exhausted"
    assert continuation.classify_no_candidate(residual_norm=0.1, budget_hit=True) == "budget_exhausted"
    assert continuation.classify_no_candidate(residual_norm=0.1, unsupported=True) == \
        "unsupported_current_direction"


def test_any_resolved_native_outward_descent_requires_face_release():
    normal = torch.tensor([2.0, 0.0], dtype=DTYPE)
    one_side_outward = continuation.native_face_release(
        torch.tensor([1.0, 0.0], dtype=DTYPE), torch.tensor([1.0, 0.0], dtype=DTYPE), normal)
    both_inward = continuation.native_face_release(
        torch.tensor([-1.0, 0.0], dtype=DTYPE), torch.tensor([1.0, 0.0], dtype=DTYPE), normal)
    assert one_side_outward["release_required"] is True
    assert both_inward["release_required"] is False


def test_previous_p2_requalification_checks_original_j_and_both_branch_traces():
    control = torch.zeros(26, dtype=DTYPE)
    minus = torch.zeros_like(control)
    plus = torch.zeros_like(control)
    minus[0], plus[0], minus[1], plus[1] = -0.5, 0.5, 0.25, 0.25
    mixed = (minus + plus) * 0.5
    full_g = torch.cat((mixed, torch.zeros(1, dtype=DTYPE)))
    previous = {"theta": 0.5, "objective": 1.0,
        "side_gradients": {"-1": minus.tolist(), "1": plus.tolist()},
        "branch_trace": {"-1": {"signature_sha256": "m"}, "1": {"signature_sha256": "p"}}}
    observed: dict[str, Any] = {"native_j": 1.0,
        "side": {-1: (torch.tensor(1.), minus), 1: (torch.tensor(1.), plus)},
        "traces": {-1: {"signature_sha256": "m"}, 1: {"signature_sha256": "p"}}}
    assert continuation.match_previous_P2(observed, 0.5, full_g, full_g, previous)["passed"] is True
    observed["native_j"] += 1e-4
    assert continuation.match_previous_P2(observed, 0.5, full_g, full_g, previous)["passed"] is False
    observed["native_j"] = 1.0
    changed_trace: dict[str, Any] = copy.deepcopy(observed)
    changed_trace["traces"][1]["signature_sha256"] = "changed"
    assert continuation.match_previous_P2(changed_trace, 0.5, full_g, full_g, previous)["passed"] is False
