from __future__ import annotations

import torch

from advar import transport


def _constant_step(amplitude: torch.Tensor):
    echo = amplitude * torch.ones((3, 3), dtype=amplitude.dtype)
    support = torch.ones_like(echo)
    qx = torch.full((3, 4), 0.11, dtype=echo.dtype)
    qy = torch.zeros((4, 3), dtype=echo.dtype)
    edges = (echo[:, 0], echo[:, -1], echo[0], echo[-1])
    known = tuple(torch.ones_like(edge) for edge in edges)
    return transport.finite_volume_step(
        echo, support, qx, qy, dt_seconds=1.0, spacing_yx=(1.0, 1.0),
        boundary_echo=(edges, edges), boundary_support=(known, known),
        reconstruction="minmod")


def test_constant_field_is_stationary_at_balanced_nonzero_flux():
    # Independent in/out updates shifted 0.3 upward by one ULP at CFL 0.11.
    amplitude = torch.tensor(0.3, dtype=torch.float64)
    result = _constant_step(amplitude)
    assert torch.equal(result.echo, amplitude.expand(3, 3))
    assert torch.equal(result.support, torch.ones_like(result.support))


def test_balanced_update_preserves_constant_amplitude_derivative():
    amplitude = torch.tensor(0.3, dtype=torch.float64)
    derivative = torch.func.grad(lambda x: _constant_step(x).echo.sum())(amplitude)
    assert torch.isfinite(derivative)
    assert torch.allclose(derivative, amplitude.new_tensor(9.0), rtol=0.0,
                          atol=128 * torch.finfo(amplitude.dtype).eps * 9.0)
