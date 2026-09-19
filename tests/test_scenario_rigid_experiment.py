"""Independent mass and coordinate oracles for the isolated rigid prototype."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/weather_scenarios'))
from rigid_experiment import (
    fit_rigid_transform_from_positions,
    forecast_rigid_positions,
    rigid_remap,
)


def independent_scatter(field, shift, angle, steps):
    height, width = field.shape
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    transform = np.eye(3)
    transform[:2, :2] = rotation
    transform[:2, 2] = shift[::-1]
    transform = np.linalg.matrix_power(transform, steps)
    out = np.zeros_like(field)
    for y in range(height):
        for x in range(width):
            px, py, _ = transform @ np.array([x-(width-1)/2, y-(height-1)/2, 1.])
            px, py = px+(width-1)/2, py+(height-1)/2
            for ty in (math.floor(py), math.floor(py)+1):
                for tx in (math.floor(px), math.floor(px)+1):
                    if 0 <= ty < height and 0 <= tx < width:
                        out[ty,tx] += field[y,x] * max(0, 1-abs(ty-py)) * max(0, 1-abs(tx-px))
    return out


@pytest.mark.parametrize('angle', [-.14, 0., .14])
@pytest.mark.parametrize('steps', [1, 6, 18])
def test_rigid_coordinates_mass_and_outflow(angle, steps):
    field = torch.arange(63, dtype=torch.float64).reshape(7,9)/63
    shift = torch.tensor([.31, -.17], dtype=field.dtype)
    actual = rigid_remap(field, shift, field.new_tensor(angle), steps=steps)
    expected = independent_scatter(field.numpy(), shift.numpy(), angle, steps)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert bool((actual >= 0).all())
    assert float(actual.sum()) <= float(field.sum()) + 1e-12


def test_rigid_preserves_interior_mass_and_identity():
    field = torch.zeros((15,15), dtype=torch.float64)
    field[5:10,5:10] = 2
    shift = field.new_tensor([.2, -.3])
    actual = rigid_remap(field, shift, field.new_tensor(.17))
    torch.testing.assert_close(actual.sum(), field.sum(), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(rigid_remap(field, shift*0, field.new_tensor(0)), field)


def test_rigid_derivatives_within_cells_and_connected_outflow():
    field = torch.linspace(.1, 1., 20, dtype=torch.float64).reshape(4,5).requires_grad_()
    shift = field.new_tensor([.37, -.23], requires_grad=True)
    angle = field.new_tensor(.027, requires_grad=True)
    assert torch.autograd.gradcheck(rigid_remap, (field, shift, angle))
    assert torch.autograd.gradgradcheck(rigid_remap, (field, shift, angle))
    outside = field.new_tensor([1e200, -1e200], requires_grad=True)
    result = rigid_remap(field, outside, angle)
    assert bool((result == 0).all())
    gradients = torch.autograd.grad(result.sum(), (field, outside, angle))
    assert all(bool(torch.isfinite(g).all() & (g == 0).all()) for g in gradients)


def independent_position(initial_yx, translation_yx, angle, steps):
    """Independent xy-coordinate oracle for T^steps, returned as yx."""
    position_xy = np.asarray(initial_yx, dtype=np.float64)[::-1].copy()
    translation_xy = np.asarray(translation_yx, dtype=np.float64)[::-1]
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
    )
    for _ in range(steps):
        position_xy = rotation @ position_xy + translation_xy
    return position_xy[::-1]


@pytest.mark.parametrize('angle', [-.12, .11])
def test_fit_signed_rotation_translation_and_eighteen_lead_positions(angle):
    initial = np.array([1.8, -2.3])
    translation = np.array([.37, -.21])
    observed = np.stack([
        independent_position(initial, translation, angle, step)
        for step in range(3)
    ])
    fitted_translation, fitted_angle = fit_rigid_transform_from_positions(
        torch.from_numpy(observed),
    )
    np.testing.assert_allclose(fitted_translation.numpy(), translation, rtol=0, atol=1e-14)
    np.testing.assert_allclose(fitted_angle.item(), angle, rtol=0, atol=1e-14)

    future = forecast_rigid_positions(
        torch.from_numpy(initial),
        fitted_translation,
        fitted_angle,
        lead_count=18,
    )
    expected = np.stack([
        independent_position(initial, translation, angle, step)
        for step in range(1, 19)
    ])
    np.testing.assert_allclose(future.detach().numpy(), expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize('angle', [-.16, .13])
def test_fit_pure_signed_rotation_without_translation(angle):
    initial = np.array([2.4, -1.7])
    translation = np.zeros(2)
    observed = np.stack([
        independent_position(initial, translation, angle, step)
        for step in range(3)
    ])
    fitted_translation, fitted_angle = fit_rigid_transform_from_positions(
        torch.from_numpy(observed),
    )
    np.testing.assert_allclose(fitted_translation.numpy(), translation, rtol=0, atol=1e-14)
    np.testing.assert_allclose(fitted_angle.item(), angle, rtol=0, atol=1e-14)


def test_constant_feature_rejects_unidentifiable_omega():
    positions = torch.zeros((3, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match='omega is unidentifiable'):
        fit_rigid_transform_from_positions(positions)
