"""Independent characteristic oracle checks for the prescribed FV probe."""

import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


_SPEC = spec_from_file_location(
    "finite_volume_probe_oracle",
    Path(__file__).parents[1] / "examples/weather_scenarios/finite_volume_probe.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_ORACLE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ORACLE)


def _profile(y, x):
    y = torch.as_tensor(y, dtype=torch.float64)
    x = torch.as_tensor(x, dtype=torch.float64)
    radius2 = ((x + 5000) / 3000).square() + ((y - 2000) / 4500).square()
    return torch.exp(-radius2 / 2) * (1 - radius2 / 9).clamp_min(0).pow(4)


def test_pointwise_translation_uses_departure_coordinates():
    y = torch.tensor([[1000.0, 2200.0]], dtype=torch.float64)
    x = torch.tensor([[-4200.0, -2800.0]], dtype=torch.float64)
    elapsed, velocity = 120.0, (3.0, -2.0)
    expected = _profile(y - velocity[1] * elapsed, x - velocity[0] * elapsed)
    actual = _ORACLE.characteristic_echo(y, x, elapsed, velocity=velocity)
    torch.testing.assert_close(actual, expected)


def test_pointwise_rotation_has_inverse_characteristic_sign():
    angle = math.pi / 2
    # The profile peak (-5000, 2000) is carried to (x, y)=(-2000, -5000).
    y = torch.tensor([-5000.0, -2000.0], dtype=torch.float64)
    x = torch.tensor([-2000.0, -5000.0], dtype=torch.float64)
    actual = _ORACLE.characteristic_echo(y, x, 1.0, omega=angle)
    assert actual[0] > 0.999999
    assert actual[0] > actual[1]


def test_cell_averages_match_independent_midpoint_integration():
    size, spacing, elapsed = 4, 1000.0, 30.0
    velocity = (0.5, -0.25)
    actual = _ORACLE.cell_averages(size, spacing, elapsed, velocity, 0.0, 0.001, quadrature_order=12)
    centers = (torch.arange(size, dtype=torch.float64) + 0.5 - size / 2) * spacing
    offsets = (torch.arange(128, dtype=torch.float64) + 0.5) / 128 * spacing - spacing / 2
    # Build the selected cell means without calling the oracle's averaging code.
    expected = torch.empty_like(actual)
    for row in range(size):
        for col in range(size):
            yy = centers[row] + offsets[:, None]
            xx = centers[col] + offsets[None, :]
            expected[row, col] = _profile(yy - velocity[1] * elapsed, xx - velocity[0] * elapsed).mean()
    expected *= math.exp(0.001 * elapsed)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)


def test_growth_multiplies_echo_and_zero_time_is_identity():
    y = torch.tensor([1800.0], dtype=torch.float64)
    x = torch.tensor([-4300.0], dtype=torch.float64)
    elapsed, gamma = 40.0, -0.0125
    base = _ORACLE.characteristic_echo(y, x, elapsed)
    grown = _ORACLE.characteristic_echo(y, x, elapsed, growth_rate=gamma)
    torch.testing.assert_close(grown, base * math.exp(gamma * elapsed))
    moving = _ORACLE.characteristic_echo(y, x, 0.0, velocity=(99.0, -31.0), growth_rate=2.0)
    torch.testing.assert_close(moving, _ORACLE.characteristic_echo(y, x, 0.0))


def test_face_averages_match_independent_fine_midpoint_quadrature():
    size, spacing, elapsed = 8, 1000.0, 120.0
    velocity = (0.75, -0.25)
    faces = _ORACLE.face_averages(size, spacing, elapsed, velocity, 0.0, 0.001, quadrature_order=12)
    centers = (torch.arange(size, dtype=torch.float64) + 0.5 - size / 2) * spacing
    offsets = (torch.arange(4096, dtype=torch.float64) + 0.5) / 4096 * spacing - spacing / 2
    boundary = size * spacing / 2
    expected = (
        _profile(centers[None, :] + offsets[:, None] - velocity[1] * elapsed,
                 -boundary - velocity[0] * elapsed).mean(0) * math.exp(0.001 * elapsed),
        _profile(centers[None, :] + offsets[:, None] - velocity[1] * elapsed,
                 boundary - velocity[0] * elapsed).mean(0) * math.exp(0.001 * elapsed),
        _profile(-boundary - velocity[1] * elapsed,
                 centers[None, :] + offsets[:, None] - velocity[0] * elapsed).mean(0) * math.exp(0.001 * elapsed),
        _profile(boundary - velocity[1] * elapsed,
                 centers[None, :] + offsets[:, None] - velocity[0] * elapsed).mean(0) * math.exp(0.001 * elapsed),
    )
    for actual, reference in zip(faces, expected):
        torch.testing.assert_close(actual, reference, rtol=3e-6, atol=2e-8)


def test_oracle_rejects_combined_flow_and_invalid_quadrature():
    with pytest.raises(ValueError, match="simultaneous"):
        _ORACLE.cell_averages(4, 1.0, 1.0, (1.0, 0.0), 0.1, 0.0)
    with pytest.raises(ValueError):
        _ORACLE.face_averages(0, 1.0, 0.0, (0.0, 0.0), 0.0, 0.0)
    with pytest.raises(ValueError):
        _ORACLE.cell_averages(4, 1.0, 0.0, (0.0, 0.0), 0.0, 0.0, quadrature_order=0)
