from __future__ import annotations

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_gn_local_window_probe as probe


def test_frozen_plan_pins_the_completed_memory_attempt():
    plan = probe._load_plan(probe.PLAN, probe._sha(probe.PLAN))
    for name in ("step.json.gz", "step.run.json", "step.resource.json"):
        path = (probe.MEMORY_ATTEMPT / name).relative_to(probe.ROOT).as_posix()
        assert plan["archive_files"][path] == probe._sha(probe.MEMORY_ATTEMPT / name)


def test_relative_plan_is_normalized_before_source_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_load_plan", lambda *_: {})

    def capture_source_manifest(_plan, plan_path, _digest):
        assert plan_path == probe.PLAN.resolve()
        raise RuntimeError("stop before archived base or numerical setup")

    monkeypatch.setattr(probe.shared, "_source_hashes", capture_source_manifest)
    with pytest.raises(RuntimeError, match="stop before archived base"):
        probe._run_child_impl(probe.PLAN.relative_to(probe.ROOT), "unused", tmp_path / "unused.json")


def test_curve_summary_reports_linear_residual_error_and_does_not_select_a_step():
    dtype = torch.float64
    g0 = torch.tensor([1.0, -2.0], dtype=dtype)
    r = torch.tensor([0.5, 0.25], dtype=dtype)
    alpha = 0.125
    g = g0 + alpha * r + torch.tensor([0.0, 0.03125], dtype=dtype)
    summary = probe._finite_curve_metrics(g0, g, r, alpha,
        torch.tensor([3.0, 4.0], dtype=dtype), 0.5, 0.7, 0.8, -0.2, 5.0, 4.9, -1.0)

    assert summary["predicted_displacement_l2"] == pytest.approx(0.625)
    assert summary["actual_displacement_l2"] == pytest.approx(0.5)
    assert summary["E_l2"] == pytest.approx(0.03125)
    assert summary["E_over_h_l2"] == pytest.approx(0.25)
    assert summary["E_over_h2_l2"] == pytest.approx(2.0)
    assert summary["actual_F2_armijo_rhs"] == pytest.approx(0.799995)
    assert summary["actual_J_armijo_rhs"] == pytest.approx(4.9999875)
    assert "accepted" not in summary


def test_endpoint_closure_rejects_missing_evidence_without_model_setup():
    control = torch.zeros(26, dtype=torch.float64)
    closure = {
        "control": control.tolist(), "theta": probe.BASE_THETA,
        "objective": probe.BASE_J, "F_squared": probe.BASE_F2,
        "side_gradients": {"-1": [0.0] * 26, "1": [0.0] * 26},
        "branch_trace": {"-1": {}, "1": {}},
        **{key: True for key in (
            "side_gradients_finite", "native_objective_matches_proposal",
            "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
            "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
            "fixed_input_unchanged", "runtime_unchanged", "deadline_passed")},
    }
    probe._require_endpoint_closure(closure, control)
    del closure["branch_trace"]
    with pytest.raises(ValueError, match="complete accepted closure"):
        probe._require_endpoint_closure(closure, control)


def test_completed_memory_archive_reduces_to_the_correct_2ec_endpoint_without_fv_setup():
    base = probe._archive_identity()

    assert probe.tangent._tensor_sha(base["control"]) == probe.BASE_CONTROL_SHA
    assert base["theta"] == probe.BASE_THETA
    assert base["objective"] == probe.BASE_J
    assert base["child_sha256"] == "72224458e690cd6675f14479f0787bf7bf4aae878ef0fa7362ed3b0113553552"
    assert probe.tangent._tensor_sha(base["direction"]) == (
        "0240d2e18fb8dfd4012bd7a0c8d3dd75f13cc12c81c5c941db9afafb3c3d6e94")
    assert set(base) == {"control", "input_after", "runtime_after", "closure", "source_before",
        "source_after", "theta", "objective", "accepted", "direction", "stored_residual",
        "stored_residual_direction", "stored_hminus", "stored_hplus", "stored_hvp_history",
        "child_sha256"}


def test_scaled_component_comparison_handles_zero_components_and_rejects_mismatch():
    dtype = torch.float64
    zero = torch.zeros(3, dtype=dtype)
    exact = probe._scaled_error(zero, zero, zero)
    changed = probe._scaled_error(torch.tensor([1e-4, 0.0, 0.0], dtype=dtype), zero,
        torch.tensor([1e-4, 0.0, 0.0], dtype=dtype))

    assert exact["passed"] is True
    assert exact["scaled_linf"] == 0.0
    assert changed["passed"] is False
    with pytest.raises(ValueError, match="fresh HVP or residual model"):
        probe._require_model_matches({"minus": exact}, {"G": changed})


def test_trace_delta_keeps_selector_sign_and_margin_evidence():
    base = {"signature_sha256": "base", "choices": [[
        {"choose_left": [[False]], "slope_sign": [[1]]},
        {"choose_left": [[False]], "slope_sign": [[1]]}]],
        "face_signs": [{"x": [[-1]], "y": [[None]]}],
        "minimum_margins": {"active_limiter_gap_over_q": 1e-6}}
    current = {"signature_sha256": "candidate", "choices": [[
        {"choose_left": [[True]], "slope_sign": [[1]]},
        {"choose_left": [[False]], "slope_sign": [[1]]}]],
        "face_signs": [{"x": [[1]], "y": [[None]]}],
        "stage_count": 360, "minimum_margins": {"active_limiter_gap_over_q": 2e-7},
        "nonfinite_or_tie": False}
    result = probe._trace_delta(base, current)

    assert result["signature_sha256"] == "candidate"
    assert result["choices_changed"] is True
    assert result["face_signs_changed"] is True
    assert result["first_changed_selector"] == {
        "stage": 0, "axis": "x", "key": "choose_left", "row": 0, "column": 0}
    assert result["first_changed_face_sign"] == {"stage": 0, "axis": "x", "row": 0, "column": 0}
    assert result["minimum_margins"] == {"active_limiter_gap_over_q": 2e-7}
    assert result["nonfinite_or_tie"] is False
