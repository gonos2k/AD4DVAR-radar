from __future__ import annotations

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_current_event_tangent as current


def _case():
    dtype = torch.float64
    control = torch.zeros(26, dtype=dtype)
    normal = torch.zeros_like(control)
    normal[25] = 1.0
    chart = torch.zeros((26, 25), dtype=dtype)
    chart[:25, :] = torch.eye(25, dtype=dtype)
    gminus, gplus = torch.zeros_like(control), torch.zeros_like(control)
    gminus[:2] = torch.tensor([1.0, -2.0], dtype=dtype)
    gplus[:2] = torch.tensor([-1.0, -2.0], dtype=dtype)
    event_minus, event_plus = torch.zeros_like(control), torch.zeros_like(control)
    event_minus[:2] = torch.tensor([1.0, 0.2], dtype=dtype)
    event_plus[:2] = event_minus[:2]
    event_minus[25], event_plus[25] = 3.0, -4.0
    return gminus, gplus, normal, event_minus, event_plus, control, chart


def test_current_tangent_projects_current_event_gradient_and_audits_both_sides():
    gm, gp, normal, em, ep, control, chart = _case()

    result = current.current_event_tangent(gm, gp, normal, em, ep,
        control, chart, 25)

    direction = result["direction"]
    audit = result["audit"]
    assert result["theta"] == pytest.approx(0.5)
    assert torch.dot(normal, direction) == pytest.approx(0.0, abs=1e-14)
    assert torch.dot(em, direction) == pytest.approx(0.0, abs=1e-14)
    assert torch.dot(ep, direction) == pytest.approx(0.0, abs=1e-14)
    assert audit["side_projection_agreement"]["passed"] is True
    assert max(audit["cost_products"]) < 0.0
    assert audit["face_decision"] == "retain_for_bounded_test"
    assert audit["scope"] == "first_order_cost_and_event_tangency_only_no_residual_merit_claim"


def test_event_side_disagreement_and_unresolved_projected_norm_are_rejected():
    gm, gp, normal, em, ep, control, chart = _case()
    stale_side = ep.clone()
    stale_side[3] = 1.0
    with pytest.raises(ValueError, match="disagree componentwise"):
        current.current_event_tangent(gm, gp, normal, em, stale_side,
            control, chart, 25)

    normal_event = 2.0 * normal
    with pytest.raises(ValueError, match="unresolved norm"):
        current.current_event_tangent(gm, gp, normal, normal_event,
            normal_event, control, chart, 25)


def test_invalid_dtype_nonfinite_and_invalid_chart_direction_are_rejected():
    gm, gp, normal, em, ep, control, chart = _case()
    with pytest.raises(ValueError, match="float64"):
        current.current_event_tangent(gm.float(), gp.float(), normal.float(),
            em.float(), ep.float(), control.float(), chart.float(), 25)
    bad = ep.clone()
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="finite float64"):
        current.current_event_tangent(gm, gp, normal, em, bad, control, chart, 25)

    wrong_chart = chart.clone()
    wrong_chart[0, 0] = 0.5
    with pytest.raises(ValueError, match="retained-coordinate chart"):
        current.current_event_tangent(gm, gp, normal, em, ep,
            control, wrong_chart, 25)


def test_negative_native_outward_slope_requires_face_release():
    gm, gp, normal, em, ep, control, chart = _case()
    gm[25] = 0.5

    result = current.current_event_tangent(gm, gp, normal, em, ep,
        control, chart, 25)

    assert result["audit"]["outward_native_cost_slopes"][0] < 0.0
    assert result["audit"]["face_decision"] == "native_face_release_required"


def test_cost_descent_does_not_imply_residual_merit_descent_for_spd_hessian():
    dtype = torch.float64
    gm = torch.zeros(26, dtype=dtype)
    gp = torch.zeros_like(gm)
    gm[:2] = torch.tensor([-1.0, -1.0], dtype=dtype)
    gp[:2] = torch.tensor([1.0, -1.0], dtype=dtype)
    normal = torch.zeros_like(gm)
    normal[25] = 1.0
    event = torch.zeros_like(gm)
    event[0] = 1.0
    control = torch.zeros_like(gm)
    chart = torch.zeros((26, 25), dtype=dtype)
    chart[:25, :] = torch.eye(25, dtype=dtype)

    result = current.current_event_tangent(gm, gp, normal, event, event,
        control, chart, 25)
    direction = result["direction"]
    hessian = torch.eye(26, dtype=dtype)
    point = direction.clone()
    cost_slopes = torch.stack((gm @ direction, gp @ direction))
    residual_merit_derivative = point @ hessian @ direction

    assert torch.all(cost_slopes < 0)
    assert torch.linalg.eigvalsh(hessian).min() > 0
    assert residual_merit_derivative > 0
