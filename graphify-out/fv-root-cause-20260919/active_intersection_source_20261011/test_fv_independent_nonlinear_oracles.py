"""Independent FV checks against exact nonlinear characteristic cell means.

The smooth and discontinuous initial fields are integrated analytically over
cells and faces.  These unit-domain experiments are not forecast skill tests.
Echo integral checks concern the modeled echo proxy and do not measure hydrometeor mass.
"""

import math

import pytest
import torch

from advar.transport import face_volume_fluxes, finite_volume_step


DTYPE = torch.float64


def _smooth_cell_means(n, t, velocity):
    dx = 1.0 / n
    edges = torch.arange(n + 1, dtype=DTYPE) * dx
    u, v = velocity
    mean_sin_x = (torch.cos(2 * math.pi * (edges[:-1] - u * t)) -
                  torch.cos(2 * math.pi * (edges[1:] - u * t))) / (2 * math.pi * dx)
    mean_cos_y = (torch.sin(2 * math.pi * (edges[1:] - v * t)) -
                  torch.sin(2 * math.pi * (edges[:-1] - v * t))) / (2 * math.pi * dx)
    return 2.0 + 0.3 * mean_cos_y[:, None] * mean_sin_x[None, :]


def _smooth_face_means(n, t, velocity):
    dx = 1.0 / n
    edges = torch.arange(n + 1, dtype=DTYPE) * dx
    u, v = velocity
    sin_x = torch.sin(2 * math.pi * (edges - u * t))
    cos_y = torch.cos(2 * math.pi * (edges - v * t))
    mean_sin = (torch.cos(2 * math.pi * (edges[:-1] - u * t)) -
                torch.cos(2 * math.pi * (edges[1:] - u * t))) / (2 * math.pi * dx)
    mean_cos = (torch.sin(2 * math.pi * (edges[1:] - v * t)) -
                torch.sin(2 * math.pi * (edges[:-1] - v * t))) / (2 * math.pi * dx)
    return (2 + 0.3 * sin_x[0] * mean_cos, 2 + 0.3 * sin_x[-1] * mean_cos,
            2 + 0.3 * mean_sin * cos_y[0], 2 + 0.3 * mean_sin * cos_y[-1])


_RECTANGLE = (0.94, 1.26, 0.20, 0.70)


def _rectangle_cell_means(n, t, velocity):
    dx = 1.0 / n
    edges = torch.arange(n + 1, dtype=DTYPE) * dx
    u, v = velocity
    xa, xb, ya, yb = _RECTANGLE
    x_overlap = (torch.minimum(edges[1:], edges.new_tensor(xb + u * t)) -
                 torch.maximum(edges[:-1], edges.new_tensor(xa + u * t))).clamp_min(0)
    y_overlap = (torch.minimum(edges[1:], edges.new_tensor(yb + v * t)) -
                 torch.maximum(edges[:-1], edges.new_tensor(ya + v * t))).clamp_min(0)
    return (y_overlap[:, None] * x_overlap[None, :]) / (dx * dx)


def _rectangle_face_means(n, t, velocity):
    dx = 1.0 / n
    edges = torch.arange(n + 1, dtype=DTYPE) * dx
    u, v = velocity
    xa, xb, ya, yb = _RECTANGLE
    xl, xr, ybnd, yt = xa + u * t, xb + u * t, ya + v * t, yb + v * t
    y_overlap = (torch.minimum(edges[1:], edges.new_tensor(yt)) -
                 torch.maximum(edges[:-1], edges.new_tensor(ybnd))).clamp_min(0) / dx
    x_overlap = (torch.minimum(edges[1:], edges.new_tensor(xr)) -
                 torch.maximum(edges[:-1], edges.new_tensor(xl))).clamp_min(0) / dx
    left = y_overlap if xl <= 0 <= xr else torch.zeros_like(y_overlap)
    right = y_overlap if xl <= 1 <= xr else torch.zeros_like(y_overlap)
    bottom = x_overlap if ybnd <= 0 <= yt else torch.zeros_like(x_overlap)
    top = x_overlap if ybnd <= 1 <= yt else torch.zeros_like(x_overlap)
    return left, right, bottom, top


def _run_characteristics(n, lead, velocity, cell_means, face_means):
    dx = 1.0 / n
    vertices = torch.arange(n + 1, dtype=DTYPE) * dx
    yv, xv = torch.meshgrid(vertices, vertices, indexing="ij")
    u, v = velocity
    qx, qy = face_volume_fluxes(u * yv - v * xv)
    assert float((qx[:, 1:] - qx[:, :-1] + qy[1:] - qy[:-1]).abs().max()) < 1e-14
    echo = cell_means(n, 0.0, velocity)
    support = torch.ones_like(echo)
    dt_limit = 0.35 / (abs(u) / dx + abs(v) / dx)
    count = math.ceil(lead / dt_limit)
    dt = lead / count
    budget = 0.0
    for index in range(count):
        start = index * dt
        boundary = (face_means(n, start, velocity), face_means(n, start + dt, velocity))
        known = tuple(torch.ones(n, dtype=DTYPE) for _ in range(4))
        result = finite_volume_step(
            echo, support, qx, qy, dt_seconds=dt, spacing_yx=(dx, dx),
            boundary_echo=boundary, boundary_support=(known, known),
            reconstruction="minmod",
        )
        assert bool(torch.isfinite(result.echo).all())
        assert bool((result.echo >= 0).all())
        torch.testing.assert_close(result.support, torch.ones_like(support), atol=2e-14, rtol=0)
        budget += float(result.physical_inflow - result.physical_outflow + result.source_quadrature)
        echo, support = result.echo, result.support
    echo_integral_change = float((echo.sum() - cell_means(n, 0.0, velocity).sum()) * dx * dx)
    return echo, echo_integral_change, budget


@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("lead", [0.4, 0.8])
def test_smooth_positive_sine_cosine_characteristics_have_exact_cell_average_truth(n, lead):
    velocity = (0.12, -0.07)
    actual, echo_integral_change, budget = _run_characteristics(
        n, lead, velocity, _smooth_cell_means, _smooth_face_means,
    )
    truth = _smooth_cell_means(n, lead, velocity)
    relative_cell_mean_error = torch.linalg.vector_norm(actual - truth) / torch.linalg.vector_norm(truth)
    assert float(relative_cell_mean_error) < 0.025
    assert abs(echo_integral_change - budget) < 2e-12
    assert float(truth.min()) > 0.0


@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("lead", [0.4, 0.8])
def test_rectangle_characteristics_include_exact_inflow_and_cell_overlap_truth(n, lead):
    velocity = (-0.12, 0.02)
    actual, echo_integral_change, budget = _run_characteristics(
        n, lead, velocity, _rectangle_cell_means, _rectangle_face_means,
    )
    truth = _rectangle_cell_means(n, lead, velocity)
    relative_cell_mean_error = torch.linalg.vector_norm(actual - truth) / torch.linalg.vector_norm(truth)
    assert float(relative_cell_mean_error) < 0.40
    assert abs(echo_integral_change - budget) < 2e-12
    # The rectangle stays away from y-edges and left x-edge. Its right inflow
    # is exactly |u| * lead times the rectangle's vertical length.
    expected_inflow = 0.12 * lead * (0.70 - 0.20)
    assert echo_integral_change == pytest.approx(expected_inflow, abs=2e-6)
    # The rectangle begins outside the domain and enters through the right
    # face for u < 0; this confirms the analytic inflow trace is nonzero.
    assert float(_rectangle_face_means(n, lead, velocity)[1].sum()) > 0.0


@pytest.mark.parametrize("lead", [0.4, 0.8])
def test_rectangle_characteristic_error_improves_under_joint_grid_and_time_refinement(lead):
    velocity = (-0.12, 0.02)
    errors = []
    for n in (8, 16):
        actual, _, _ = _run_characteristics(
            n, lead, velocity, _rectangle_cell_means, _rectangle_face_means,
        )
        truth = _rectangle_cell_means(n, lead, velocity)
        errors.append(float(torch.linalg.vector_norm(actual - truth) / torch.linalg.vector_norm(truth)))
    # This checks only that refinement helps for this discontinuous case; it
    # makes no formal order-of-accuracy claim.
    assert errors[1] < 0.5 * errors[0]
