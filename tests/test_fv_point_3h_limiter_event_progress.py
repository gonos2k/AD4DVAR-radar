import pytest
import torch
from typing import Any

from examples.weather_scenarios.fv_point_3h_limiter_event_progress import (
    Budget, EventExecution, close_claims, event_newton_model,
)
from examples.weather_scenarios.fv_minmod_events import MinmodEvent


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


def test_actual_event_union_allows_two_events_and_rejects_more_than_two():
    execution = object.__new__(EventExecution)
    row: dict[str, Any] = {"branch": {"unique_nearzero_face_entities": ["y:2:1", "y:4:3"]},
        "active_limiter_events": [
            {"stage": 124, "axis": "y", "cell": [2, 3], "left": 2., "right": 0., "q_scale": 40.},
            {"stage": 125, "axis": "y", "cell": [2, 3], "left": 2., "right": 0., "q_scale": 40.},
        ]}
    assert [event.stage for event in execution.actual_events(row)] == [124, 125]
    row["active_limiter_events"].append(
        {"stage": 126, "axis": "y", "cell": [2, 3], "left": 2., "right": 0., "q_scale": 40.})
    with pytest.raises(ValueError, match="more than two simultaneous"):
        execution.actual_events(row)


def test_actual_event_union_rejects_an_unmodeled_flux_face():
    execution = object.__new__(EventExecution)
    row = {"branch": {"unique_nearzero_face_entities": ["y:2:1", "y:3:2", "y:4:3"]},
        "active_limiter_events": []}
    with pytest.raises(ValueError, match="additional current flux-face activation"):
        execution.actual_events(row)


def test_dependent_zero_event_normal_refuses_before_model_solve():
    with pytest.raises(ValueError, match="normal"):
        event_newton_model(torch.ones(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64),
                           torch.zeros(2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64))


def test_event_restoration_keeps_native_value_on_curved_constraint():
    execution = object.__new__(EventExecution)
    event = MinmodEvent(48, "x", 1, 1, "gap", 1)

    def value(z: torch.Tensor, purpose: str) -> dict[str, Any]:
        h = float(z[1] - z[0]**2 / 2)
        return {"native_J": float(z[1] + 2*z[0]**2), "gap": h}

    execution.value = value
    def operands(row: dict[str, Any], event: MinmodEvent) -> tuple[float, float, float]:
        return 0.4, 0.4 + row["gap"], 1.

    execution.operands = operands
    point, row = execution.restore_events(torch.tensor([.01, 0.], dtype=torch.float64),
        [event], torch.tensor([[0., 1.]], dtype=torch.float64))
    assert abs(float(point[1] - point[0]**2 / 2)) < 1e-14
    assert row["native_J"] == float(point[1] + 2*point[0]**2)


def test_search_cannot_spend_terminal_reserved_native_call():
    import time
    state = {"plan": {"budget": {"additional_wall_seconds": 900,
        "RSS_cap_bytes": 2**63, "direct_J_cap": 256}},
        "counts": {"direct_J_started": 254}}
    with pytest.raises(RuntimeError, match="budget"):
        Budget(state, time.monotonic()).take("direct_J")
    assert state["counts"]["direct_J_started"] == 254


def test_source_closure_failure_downgrades_terminal_minimum_claim():
    state: dict[str, Any] = {"minimum_claim": True,
        "terminal": {"minimum_claim": True, "P2_complete": True}}
    close_claims(state, True, False)
    assert state["minimum_claim"] is False
    assert state["terminal"]["minimum_claim"] is False
    assert state["terminal"]["P2_complete"] is False


def test_changed_trace_rejects_unmodeled_intervening_flux_face():
    execution = object.__new__(EventExecution)
    trace = bytes([1] * 4320).hex()
    before = {"exact_minmod_selectors_hex": trace, "flux_signs": [0] * 49}
    after_flux = [0] * 49
    after_flux[0] = 1
    after = {"exact_minmod_selectors_hex": trace, "flux_signs": after_flux}
    with pytest.raises(ValueError, match="additional flux-face transition"):
        execution.changed_trace(before, after)

    changed = bytearray.fromhex(trace)
    changed[124 * 12 + 6 + 5] = 2
    after_selector = {"exact_minmod_selectors_hex": changed.hex(), "flux_signs": [0] * 49}
    assert execution.changed_trace(before, after_selector)
