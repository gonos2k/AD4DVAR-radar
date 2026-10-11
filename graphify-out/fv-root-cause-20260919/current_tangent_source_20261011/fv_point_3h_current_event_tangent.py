"""Pure current-point tangent directions for a one-sided limiter event.

The returned cost slopes diagnose first-order behavior only. They do not
predict residual-merit change or certify a root or minimum.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_gn_local_window_probe as local
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def current_event_tangent(
    gminus: Tensor,
    gplus: Tensor,
    normal: Tensor,
    event_gradient_minus: Tensor,
    event_gradient_plus: Tensor,
    control: Tensor,
    chart_jacobian: Tensor,
    pivot: int,
) -> dict[str, Any]:
    """Build the strict-interior mixture descent and event-orthogonal tangent.

    All vectors are in the 26-dimensional control coordinates.  The event
    gradients are freshly supplied at the same current point, one per side.
    """
    vectors = (gminus, gplus, normal, event_gradient_minus,
        event_gradient_plus, control)
    if (any(value.shape != (26,) or value.dtype != torch.float64
            or not bool(torch.isfinite(value).all()) for value in vectors)
            or chart_jacobian.shape != (26, 25)
            or chart_jacobian.dtype != torch.float64
            or chart_jacobian.device != control.device
            or any(value.device != control.device for value in vectors)
            or not bool(torch.isfinite(chart_jacobian).all())
            or not 0 <= pivot < 26):
        raise ValueError("current tangent inputs must be finite float64 26-vectors and a 26x25 chart")

    eps = torch.finfo(control.dtype).eps
    n_norm = torch.linalg.vector_norm(normal)
    if not bool(torch.isfinite(n_norm) & (n_norm > torch.finfo(control.dtype).tiny)):
        raise ValueError("selected-face normal has unresolved norm")
    n_unit = normal / n_norm

    projected_sides: list[Tensor] = []
    for gradient in (event_gradient_minus, event_gradient_plus):
        gradient_norm = torch.linalg.vector_norm(gradient)
        projected = gradient - n_unit * torch.dot(n_unit, gradient)
        projected_norm = torch.linalg.vector_norm(projected)
        if (not bool(torch.isfinite(gradient_norm) & torch.isfinite(projected_norm))
                or not bool(gradient_norm > torch.finfo(control.dtype).tiny)
                or not bool(projected_norm > 128 * eps * gradient_norm)):
            raise ValueError("current event tangent gradient has unresolved norm")
        projected_sides.append(projected)

    common_scale = 0.5 * (projected_sides[0] + projected_sides[1])
    agreement = local._scaled_error(projected_sides[0], projected_sides[1], common_scale)
    if agreement["passed"] is not True:
        raise ValueError("current one-sided event tangent gradients disagree componentwise")
    event_tangent_gradient = common_scale

    theta = tangent.minimum_mixture_weight(gminus, gplus)
    mixed_gradient = (1 - theta) * gminus + theta * gplus
    baseline_direction = -(mixed_gradient - n_unit * torch.dot(n_unit, mixed_gradient))
    denominator = torch.dot(event_tangent_gradient, event_tangent_gradient)
    if (not bool(torch.isfinite(denominator))
            or not bool(denominator > torch.finfo(control.dtype).tiny)):
        raise ValueError("common current event tangent gradient has unresolved norm")
    direction = baseline_direction - event_tangent_gradient * (
        torch.dot(event_tangent_gradient, baseline_direction) / denominator)
    chart_error = tangent._validate_direction_override(
        control, normal, chart_jacobian, pivot, direction)

    cost_products = (torch.dot(gminus, direction), torch.dot(gplus, direction))
    baseline_cost_products = (torch.dot(gminus, baseline_direction),
        torch.dot(gplus, baseline_direction))
    event_products = (torch.dot(event_gradient_minus, direction),
        torch.dot(event_gradient_plus, direction))
    outward_slopes = (torch.dot(gminus, -n_unit), torch.dot(gplus, n_unit))
    outward_tolerances = tuple(128 * eps * torch.linalg.vector_norm(gradient)
        for gradient in (gminus, gplus))
    retains_face = all(bool(torch.isfinite(slope) & (slope >= -tolerance))
        for slope, tolerance in zip(outward_slopes, outward_tolerances))

    return {
        "theta": float(theta),
        "baseline_direction": baseline_direction,
        "direction": direction,
        "event_tangent_gradient": event_tangent_gradient,
        "audit": {
            "side_projection_agreement": agreement,
            "baseline_cost_products": tuple(float(value) for value in baseline_cost_products),
            "cost_products": tuple(float(value) for value in cost_products),
            "event_gradient_products": tuple(float(value) for value in event_products),
            "outward_native_cost_slopes": tuple(float(value) for value in outward_slopes),
            "outward_slope_tolerances": tuple(float(value) for value in outward_tolerances),
            "face_decision": "retain_for_bounded_test" if retains_face
                else "native_face_release_required",
            "direction_chart_error": chart_error,
            "scope": "first_order_cost_and_event_tangency_only_no_residual_merit_claim",
        },
    }
