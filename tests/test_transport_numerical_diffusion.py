"""Independent moment, Fourier, and smooth-grid checks of actual transport."""

import math

import pytest
import torch

from advar.nowcast import NowcastConfig, RadarState, forecast_linear_at_step
from advar.physics import remap


def _variance_x(echo):
    probability = echo.sum(dim=0) / echo.sum()
    x = torch.arange(echo.shape[1], dtype=echo.dtype)
    center = (probability * x).sum()
    return float((probability * (x - center).square()).sum())


@pytest.mark.parametrize("shift", [-8.3, -0.5, 0.0, 0.3, 0.5, 8.75])
def test_single_remap_variance_and_fourier_damping(shift):
    source = torch.zeros((3, 128), dtype=torch.float64)
    source[1, 48] = 1
    actual = remap(source, source.new_tensor((0, shift)))
    fraction = shift - math.floor(shift)
    assert float(actual.sum()) == pytest.approx(1, abs=1e-15)
    assert _variance_x(actual) == pytest.approx(fraction * (1 - fraction), abs=1e-14)
    frequency = 2 * math.pi * torch.fft.rfftfreq(128, dtype=source.dtype)
    expected_power = 1 - 4 * fraction * (1 - fraction) * torch.sin(frequency / 2).square()
    actual_power = torch.fft.rfft(actual.sum(dim=0)).abs().square()
    torch.testing.assert_close(actual_power, expected_power, rtol=1e-13, atol=1e-14)


def test_forecast_does_not_accumulate_repeated_interpolation_diffusion():
    source = torch.zeros((3, 128), dtype=torch.float64)
    source[1, 48] = 1
    displacement = source.new_tensor((0, 0.3))
    state = RadarState(source, displacement, source.new_zeros(()))
    config = NowcastConfig()
    repeated = source
    for lead in range(1, 19):
        direct = forecast_linear_at_step(state, lead, config)
        repeated = remap(repeated, displacement)
        fraction = lead * 0.3 % 1
        assert _variance_x(direct) == pytest.approx(fraction * (1 - fraction), abs=1e-13)
        assert _variance_x(repeated) == pytest.approx(lead * 0.3 * 0.7, abs=1e-13)
    assert _variance_x(direct) == pytest.approx(0.24)
    assert _variance_x(repeated) == pytest.approx(3.78)


def test_smooth_cell_average_error_decreases_under_grid_refinement():
    # Exact Gaussian cell averages; fixed physical displacement, no boundary outflow.
    errors = []
    for size in (64, 128, 256):
        dx = 4 / size
        edges = torch.linspace(-2, 2, size + 1, dtype=torch.float64)

        def average(center):
            integral = torch.erf((edges - center) / (math.sqrt(2) * 0.3))
            return math.sqrt(math.pi / 2) * 0.3 * torch.diff(integral) / dx

        source = average(-0.4).repeat(2, 1)
        actual = remap(source, source.new_tensor((0, (1 / 3) / dx)))[0]
        truth = average(-0.4 + 1 / 3)
        centers = (edges[1:] + edges[:-1]) / 2
        interior = centers.abs() < 1
        errors.append(float((actual[interior] - truth[interior]).square().mean().sqrt()))
    # This is fixed-time smooth translation convergence, not a general PDE order claim.
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
