from __future__ import annotations

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_residual_direction as residual


def _curved_common_face():
    dtype = torch.float64
    control = torch.linspace(-0.12, 0.18, 26, dtype=dtype)
    weights = torch.tensor([1.0, -0.7, 0.9, 0.5, -0.8], dtype=dtype)
    pivot = geometry._pivot(control, weights)
    point, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    normal = geometry._face_normal(point, weights)
    tangent_gradient = chart @ torch.linspace(0.03, 0.17, 25, dtype=dtype)
    unit = normal / torch.linalg.vector_norm(normal)

    def projector(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        n = geometry._face_normal(x, weights)
        return value - n * torch.dot(n, value) / torch.dot(n, n)

    def potential(x: torch.Tensor) -> torch.Tensor:
        dx = x - point
        matrix = torch.diag(torch.linspace(0.5, 1.8, 26, dtype=dtype))
        return 0.5 * dx @ matrix @ dx + torch.dot(tangent_gradient, dx)

    def common_gradient(x: torch.Tensor) -> torch.Tensor:
        return projector(x, torch.func.grad(potential)(x))

    mixed = common_gradient(point)
    gm, gp = mixed - 0.6 * unit, mixed + 0.6 * unit
    return point, weights, pivot, chart, normal, gm, gp, common_gradient


def test_curved_face_action_matches_independent_direct_autodiff_and_chart_jvp():
    point, weights, pivot, chart, normal, gm, gp, common_gradient = _curved_common_face()
    direction, diagnostics = residual.build_residual_direction(
        lambda vector: torch.func.jvp(common_gradient, (point,), (vector,))[1],
        normal, gm, gp, chart, pivot)
    raw = diagnostics["residual_direction_raw"]
    descent = diagnostics["descent_direction"]

    # Independently differentiate the control-space map, including the changing
    # curved-face projector, then apply the Euclidean projector at the base.
    direct = torch.func.jvp(common_gradient, (point,), (descent,))[1]
    unit = normal / torch.linalg.vector_norm(normal)
    expected_raw = direct - unit * torch.dot(unit, direct)
    torch.testing.assert_close(raw, expected_raw, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(torch.linalg.vector_norm(direction),
        torch.linalg.vector_norm(descent), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(chart @ direction[[i for i in range(26) if i != pivot]],
        direction, atol=2e-13, rtol=2e-13)

    retained = [i for i in range(26) if i != pivot]

    def chart_path(alpha: torch.Tensor) -> torch.Tensor:
        coordinates = point.clone()
        coordinates[retained] = point[retained] + alpha * descent[retained]
        return geometry._chart(coordinates, weights, pivot, 0.0)

    zero = torch.zeros((), dtype=point.dtype)
    chart_derivative = torch.func.jvp(
        lambda alpha: common_gradient(chart_path(alpha)),
        (zero,), (torch.ones_like(zero),))[1]
    torch.testing.assert_close(chart_derivative, direct, atol=4e-13, rtol=4e-13)
    assert diagnostics["finite"]
    assert diagnostics["both_side_gradients_descend"]


def test_projection_is_euclidean_and_normal_scale_does_not_change_direction():
    dtype = torch.float64
    gm = torch.tensor([-1.0, -1.0], dtype=dtype)
    gp = torch.tensor([1.0, -1.0], dtype=dtype)
    chart = torch.tensor([[0.0], [1.0]], dtype=dtype)
    normal = torch.tensor([3.0, 0.0], dtype=dtype)
    action = lambda vector: torch.tensor([2.0, 4.0], dtype=dtype) * vector
    first, _ = residual.build_residual_direction(action, normal, gm, gp, chart, 0)
    second, info = residual.build_residual_direction(action, 17.0 * normal,
        gm, gp, chart, 0)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(info["residual_direction_raw"],
        torch.tensor([0.0, 4.0], dtype=dtype))
    torch.testing.assert_close(info["normalization_target_norm"],
        torch.tensor(1.0, dtype=dtype))


def test_bad_common_face_jump_and_chart_are_rejected():
    dtype = torch.float64
    gm = torch.tensor([-1.0, -1.0], dtype=dtype)
    gp = torch.tensor([1.0, -0.1], dtype=dtype)
    normal = torch.tensor([1.0, 0.0], dtype=dtype)
    chart = torch.tensor([[0.0], [1.0]], dtype=dtype)
    with pytest.raises(ValueError, match="supported by the common face"):
        residual.build_residual_direction(lambda v: v, normal, gm, gp, chart, 0)

    with pytest.raises(ValueError, match="unresolved"):
        residual.build_residual_direction(lambda v: v, normal, gm, gm, chart, 0)

    # The requested retained-coordinate chart must represent the Euclidean
    # projected descent exactly; this one has the wrong tangent slope.
    with pytest.raises(ValueError, match="retained-coordinate chart"):
        residual.build_residual_direction(lambda v: v, normal,
            torch.tensor([-1.0, -1.0], dtype=dtype),
            torch.tensor([1.0, -1.0], dtype=dtype),
            torch.tensor([[0.0], [0.9]], dtype=dtype), 0)


def test_adverse_curvature_is_measured_and_marked_conditional():
    dtype = torch.float64
    gm = torch.tensor([-1.0, -1.0], dtype=dtype)
    gp = torch.tensor([1.0, -1.0], dtype=dtype)
    normal = torch.tensor([1.0, 0.0], dtype=dtype)
    chart = torch.tensor([[0.0], [1.0]], dtype=dtype)
    direction, info = residual.build_residual_direction(
        lambda vector: -vector, normal, gm, gp, chart, 0)
    assert info["finite"]
    assert not info["both_side_gradients_descend"]
    assert not info["merit_descends"]
    assert float(info["merit_product"]) > 0.0
    assert torch.isfinite(direction).all()
