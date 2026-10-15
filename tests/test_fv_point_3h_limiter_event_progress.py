import pytest
import torch
from typing import Any

from examples.weather_scenarios.fv_point_3h_limiter_event_progress import (
    Budget, Execution, active_stage, bisect_gap, close_claims, event_newton_model,
    positive_gap_active, supported_transition,
)


def test_lagrangian_curvature_includes_curved_event_constraint():
    # f=y, h=y-x²/2: along h=0, f=x²/2 has positive curvature at zero.
    g = torch.tensor([0., 1.], dtype=torch.float64)
    n = torch.tensor([0., 1.], dtype=torch.float64)
    hessian = torch.zeros(2, 2, dtype=torch.float64)
    event_hessian = torch.diag(torch.tensor([-1., 0.], dtype=torch.float64))
    step, model = event_newton_model(g, hessian, n, event_hessian)
    assert model["multiplier"] == -1
    assert model["lambda_min"] == 1
    assert model["shift"] == 0
    assert model["projected_gradient_inf"] == 0
    assert torch.equal(step, torch.zeros(2, dtype=torch.float64))


def test_event_step_is_tangent_and_descending():
    g = torch.tensor([1., 2.], dtype=torch.float64)
    n = torch.tensor([0., 1.], dtype=torch.float64)
    step, model = event_newton_model(g, torch.eye(2, dtype=torch.float64), n,
                                    torch.zeros(2, 2, dtype=torch.float64))
    assert abs(float(n @ step)) < 1e-14
    assert model["slope"] < 0


def test_same_control_replay_refuses_any_additional_active_tie():
    assert positive_gap_active(.4, .4, 40)
    assert not positive_gap_active(0, 0, 40)
    assert active_stage({"active_limiter_events": []}) is None
    event = {"stage": 48, "axis": "x", "cell": [1, 1], "left": .4, "right": .4, "q_scale": 40}
    assert active_stage({"active_limiter_events": [event]}) == 48
    with pytest.raises(ValueError, match="simultaneous"):
        active_stage({"active_limiter_events": [event, dict(event, stage=47)]})
    assert active_stage({"active_limiter_events": [dict(event, stage=46)]}) == 46


def test_dependent_zero_event_normal_refuses_before_model_solve():
    with pytest.raises(ValueError, match="normal"):
        event_newton_model(torch.ones(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64),
                           torch.zeros(2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64))


def test_bisection_uses_actual_gap_and_refuses_unbracketed_event():
    def evaluate(alpha):
        gap = .37 - alpha
        return {"events": {"48": {"left": 1., "right": 1. + gap, "q_scale": 1., "gap": gap}}}
    root, row, bracket = bisect_gap(evaluate, 48, 0., 1.)
    assert abs(root - .37) < 1e-13
    assert bracket[0] <= root <= bracket[1]
    assert positive_gap_active(row["events"]["48"]["left"], row["events"]["48"]["right"], 1.)
    with pytest.raises(ValueError, match="bracket"):
        bisect_gap(evaluate, 48, 0., .2)


def test_retraction_evaluates_native_value_on_curved_constraint():
    execution = object.__new__(Execution)
    reference: dict[str, Any] = {"exact_minmod_selectors_hex": bytes([1] * 4320).hex(), "flux_signs": [0] * 49}
    def value(z, purpose):
        h = float(z[1] - z[0]**2 / 2)
        return {**reference, "active_limiter_events": [], "native_J": float(z[1] + 2*z[0]**2), "purpose": purpose,
            "events": {"48": {"left": .4, "right": .4 + h, "gap": h,
                "q_scale": 1., "active": positive_gap_active(.4, .4+h, 1.)}}}
    execution.value = value
    point, row = execution.retract(torch.tensor([.01, 0.], dtype=torch.float64), 48,
                                   torch.tensor([0., 1.], dtype=torch.float64), reference)
    assert abs(float(point[1] - point[0]**2/2)) < 1e-14
    assert row["native_J"] == float(point[1] + 2*point[0]**2)


def test_search_cannot_spend_terminal_reserved_native_call():
    import time
    state = {"plan": {"budget": {"additional_wall_seconds": 900,
        "RSS_cap_bytes": 2**63, "direct_J_cap": 256}},
        "counts": {"direct_J_started": 254}}
    with pytest.raises(RuntimeError, match="budget"):
        Budget(state, time.monotonic()).take("direct_J")
    assert state["counts"]["direct_J_started"] == 254


def test_active_limiter_never_inherits_two_flux_face_P2():
    execution = object.__new__(Execution)
    terminal: dict[str, Any] = {"minimum_claim": False, "P2_complete": False,
        "branch": {"unique_nearzero_face_entities": ["y:2:1", "y:4:3"],
                   "unique_near_tie_slope_entities": ["x:1:1"]},
        "active_limiter_events": [{"stage": 48}]}
    execution.qualify_two_faces(torch.zeros(24, dtype=torch.float64), terminal, [])
    assert terminal["minimum_claim"] is False and terminal["P2_complete"] is False
    assert "exceeds" in terminal["unresolved_reason"]


def test_only_identified_selector_events_and_selected_flux_signs_can_change():
    trace = bytearray([1] * 4320)
    before = {"exact_minmod_selectors_hex": trace.hex(), "flux_signs": [0] * 49}
    trace[48*12] = 2
    after: dict[str, Any] = {"exact_minmod_selectors_hex": trace.hex(), "flux_signs": [0] * 49}
    after["flux_signs"][35] = 1
    assert supported_transition(before, after) == [
        {"stage": 48, "axis": "x", "cell": [1, 1], "before": 1, "after": 2}]
    trace[46*12] = 2
    after["exact_minmod_selectors_hex"] = trace.hex()
    assert len(supported_transition(before, after)) == 2


def test_source_closure_failure_downgrades_terminal_minimum_claim():
    state: dict[str, Any] = {"minimum_claim": True,
        "terminal": {"minimum_claim": True, "P2_complete": True}}
    close_claims(state, True, False)
    assert state["minimum_claim"] is False
    assert state["terminal"]["minimum_claim"] is False
    assert state["terminal"]["P2_complete"] is False


def test_retraction_refuses_intermediate_forbidden_event_before_it_can_revert():
    execution = object.__new__(Execution)
    reference: dict[str, Any] = {"exact_minmod_selectors_hex": bytes([1] * 4320).hex(), "flux_signs": [0] * 49}
    calls = []
    def value(z, purpose):
        calls.append(purpose)
        changed = bytearray.fromhex(reference["exact_minmod_selectors_hex"])
        changed[46*12] = 2
        return {**reference, "exact_minmod_selectors_hex": changed.hex()}
    execution.value = value
    with pytest.raises(ValueError, match="intervening limiter"):
        execution.retract(torch.zeros(2, dtype=torch.float64), 48,
                          torch.ones(2, dtype=torch.float64), reference)
    assert len(calls) == 1
