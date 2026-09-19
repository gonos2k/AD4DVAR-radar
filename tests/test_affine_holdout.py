"""Check the holdout oracle against closed affine characteristics."""

from pathlib import Path
import math
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "weather_scenarios"))
from affine_holdout import affine_cell_averages, affine_face_fluxes, _run_case


@pytest.mark.parametrize("mode", ["translation", "shear_with_translation", "strain"])
def test_affine_oracle_matches_closed_characteristic_cell_integrals(mode):
    if mode == "translation":
        matrix, velocity = [[0., 0.], [0., 0.]], [.2, -.1]
    elif mode == "shear_with_translation":
        matrix, velocity = [[0., .04], [0., 0.]], [.2, -.1]
    else:
        matrix, velocity = [[.03, 0.], [0., -.03]], [0., 0.]
    case = {"name": mode, "A_per_second": matrix, "b_mps": velocity,
            "growth_per_second": .02, "initial_center_xy_m": [-.5, .8],
            "initial_scale_xy_m": [1.8, 2.7], "initial_angle_degrees": 23.}
    size, side, elapsed = 8, 16., 2.5
    actual = affine_cell_averages(case, size, side, elapsed, order=8)
    centers = (np.arange(size) + .5 - size / 2) * side / size
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    nodes, weights = np.polynomial.legendre.leggauss(8)
    expected = np.zeros((size, size))
    angle = math.radians(23.)
    for ny, wy in zip(nodes, weights):
        for nx, wx in zip(nodes, weights):
            x, y = xx + nx * side / size / 2, yy + ny * side / size / 2
            if mode == "strain":
                dx, dy = np.exp(-.03 * elapsed) * x, np.exp(.03 * elapsed) * y
            else:
                shear = matrix[0][1]
                dx = x - shear * y * elapsed - velocity[0] * elapsed + .5 * shear * velocity[1] * elapsed**2
                dy = y - velocity[1] * elapsed
            dx, dy = dx + .5, dy - .8
            u = math.cos(angle) * dx + math.sin(angle) * dy
            v = -math.sin(angle) * dx + math.cos(angle) * dy
            radius2 = (u / 1.8)**2 + (v / 2.7)**2
            expected += wx * wy / 4 * np.exp(-radius2 / 2) * np.maximum(1 - radius2 / 9, 0)**4
    expected *= np.exp(.02 * elapsed)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-12, atol=2e-14)


def test_affine_streamfunction_flux_matches_integrated_normal_velocity():
    case = {"A_per_second": [[.03, -.02], [.07, -.03]], "b_mps": [.2, -.1]}
    size, side = 5, 20.
    qx, qy = affine_face_fluxes(case, size, side)
    spacing = side / size
    vertex = (torch.arange(size + 1, dtype=torch.float64) - size / 2) * spacing
    center = (vertex[1:] + vertex[:-1]) / 2
    expected_x = spacing * (.03 * vertex[None, :] - .02 * center[:, None] + .2)
    expected_y = spacing * (.07 * center[None, :] - .03 * vertex[:, None] - .1)
    torch.testing.assert_close(qx, expected_x, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(qy, expected_y, rtol=2e-14, atol=2e-14)


def test_trace_check_uses_the_flow_scale():
    case = {"name": "small divergent flow", "A_per_second": [[1e-15, 0.], [0., 0.]], "b_mps": [0., 0.]}
    with pytest.raises(ValueError, match="trace zero"):
        affine_face_fluxes(case, 4, 8.)


def test_lead_outflow_uses_a_common_integrating_factor_origin():
    case = {"name": "audit", "A_per_second": [[0., 0.], [0., 0.]], "b_mps": [1., 0.],
            "growth_per_second": 1., "initial_center_xy_m": [1.7, 0.],
            "initial_scale_xy_m": [.5, .6], "initial_angle_degrees": 0.}
    manifest = {"grid_size": 4, "domain_side_m": 8., "lead_interval_seconds": .4,
                "lead_count": 1, "speed_bound_mps": 3., "max_courant": .5,
                "thresholds": {"truth_anisotropy_minimum": .1}}
    record = _run_case(case, manifest, "minmod")
    assert record["substeps_per_lead"] > 1
    assert record["leads"][0]["transformed_outflow"] > 0.
    assert record["leads"][0]["lead_transformed_budget_residual"] < 1e-12
