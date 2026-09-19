"""Moment diagnostics distinguish rigid rotation from translation."""

import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from advar.physics import echo_to_dbz


_SPEC = spec_from_file_location(
    "motion_diagnostics",
    Path(__file__).parents[1] / "examples/weather_scenarios/motion_diagnostics.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_DIAGNOSTICS = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DIAGNOSTICS)


BASE = 100.0
MIN_DBZ = -20.0


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    size = 81
    coords = (torch.arange(size, dtype=torch.float64) + 0.5 - size / 2.0)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return x, y


def _profile(x: torch.Tensor, y: torch.Tensor, center: tuple[float, float], angle: float) -> torch.Tensor:
    cx, cy = center
    c, s = math.cos(angle), math.sin(angle)
    # Coordinates in the profile's principal frame.
    u = c * (x - cx) + s * (y - cy)
    v = -s * (x - cx) + c * (y - cy)
    return 180.0 * torch.exp(-0.5 * (u.square() / 8.0**2 + v.square() / 2.5**2))


def _dbz(q_excess: torch.Tensor) -> torch.Tensor:
    return echo_to_dbz(BASE + q_excess, min_dbz=MIN_DBZ)


def test_rigid_rotation_changes_polar_and_covariance_axis():
    x, y = _grid()
    angle = math.radians(32.0)
    center = (8.0, -5.0)
    reference = _profile(x, y, center, math.radians(18.0))
    c, s = math.cos(angle), math.sin(angle)
    # Pull the current grid back by R(-angle), which carries both the centre
    # and the anisotropic shape through a rigid rotation around the window
    # centre.
    x0 = c * x + s * y
    y0 = -s * x + c * y
    rotated = _profile(x0, y0, center, math.radians(18.0))
    result = _DIAGNOSTICS.motion_diagnostics(reference_grid=_dbz(reference), field=_dbz(rotated), spacing_m=1.0, background_echo=BASE, min_dbz=MIN_DBZ)

    assert result["status"] == "ok"
    assert result["polar_change_deg"] == pytest.approx(32.0, abs=0.15)
    assert result["axis_change_deg"] == pytest.approx(32.0, abs=0.15)
    assert result["radius_m"] == pytest.approx(result["reference_radius_m"], rel=2e-3)
    assert result["anisotropy"] == pytest.approx(result["reference_anisotropy"], rel=2e-3)


def test_translation_preserves_axis_but_centroid_angle_is_not_rotation_proof():
    x, y = _grid()
    center = (8.0, -5.0)
    reference = _profile(x, y, center, math.radians(18.0))
    translated = _profile(x, y, (center[0] + 7.0, center[1] + 3.0), math.radians(18.0))
    result = _DIAGNOSTICS.motion_diagnostics(_dbz(reference), _dbz(translated), 1.0, BASE, MIN_DBZ)

    assert result["status"] == "ok"
    assert result["axis_change_deg"] == pytest.approx(0.0, abs=0.15)
    # The centroid angle can change under translation, so it cannot identify
    # rotation by itself.
    assert abs(result["polar_change_deg"]) > 1.0
    assert result["radius_m"] != pytest.approx(result["reference_radius_m"], rel=1e-3)


def test_zero_missing_and_point_like_excess_are_explicitly_unavailable():
    size = 9
    zero = _dbz(torch.zeros((size, size), dtype=torch.float64))
    zero_result = _DIAGNOSTICS.motion_diagnostics(zero, zero, 1.0, BASE, MIN_DBZ)
    assert zero_result["status"] == "unavailable"
    assert zero_result["centroid_xy_m"] is None
    assert zero_result["axis_change_deg"] is None

    missing = zero.clone()
    missing[0, 0] = float("nan")
    missing_result = _DIAGNOSTICS.motion_diagnostics(zero, missing, 1.0, BASE, MIN_DBZ)
    assert missing_result["status"] == "unavailable"
    assert missing_result["reason"] == "incomplete_nonfinite_frame"
    assert missing_result["centroid_xy_m"] is None

    point_excess = torch.zeros((size, size), dtype=torch.float64)
    point_excess[size // 2, size // 2] = 30.0
    point_result = _DIAGNOSTICS.motion_diagnostics(
        _dbz(point_excess), _dbz(point_excess), 1.0, BASE, MIN_DBZ
    )
    assert point_result["status"] == "partial"
    assert point_result["reason"] == "degenerate_covariance_orientation"
    assert point_result["axis_angle_deg"] is None
    assert point_result["widths_m"] is None


@pytest.mark.parametrize("spacing", [1e-8, 1., 1e8])
def test_axis_identifiability_is_independent_of_coordinate_units(spacing):
    x, y = _grid()
    elongated = _dbz(_profile(x, y, (8., -5.), .3))
    result = _DIAGNOSTICS.motion_diagnostics(elongated, elongated, spacing, BASE, MIN_DBZ)
    assert result['status'] == 'ok'
    assert result['axis_change_deg'] == pytest.approx(0.)
    circle = _dbz(100. * torch.exp(-(x.square() + y.square()) / 20.))
    ambiguous = _DIAGNOSTICS.motion_diagnostics(circle, circle, spacing, BASE, MIN_DBZ)
    assert ambiguous['axis_angle_deg'] is None
