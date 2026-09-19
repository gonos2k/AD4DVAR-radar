"""Independent Phase 1 boundary scenarios.

These cases combine dimensions and dtypes that are individually covered by
the baseline tests.  The expected transport values are produced by a small
NumPy destination-scatter oracle, while PCG and metric expectations use the
closed-form equations directly.
"""

import math

import numpy as np
import pytest
import torch

from advar.matrix_free import pcg
from advar.metrics import mae, rmse
from advar.physics import RemapCell, remap, remap_core


SEED = 20260908


def _scatter_oracle(
    source: np.ndarray,
    displacement_yx: tuple[float, float],
) -> np.ndarray:
    """Scatter bilinear mass into zero-padded destinations independently."""

    height, width = source.shape
    dy, dx = displacement_yx
    cell_y, cell_x = math.floor(dy), math.floor(dx)
    fraction_y, fraction_x = dy - cell_y, dx - cell_x
    expected = np.zeros_like(source)
    for source_y in range(height):
        for source_x in range(width):
            value = source[source_y, source_x]
            for offset_y, weight_y in ((0, 1.0 - fraction_y), (1, fraction_y)):
                for offset_x, weight_x in ((0, 1.0 - fraction_x), (1, fraction_x)):
                    destination_y = source_y + cell_y + offset_y
                    destination_x = source_x + cell_x + offset_x
                    if 0 <= destination_y < height and 0 <= destination_x < width:
                        expected[destination_y, destination_x] += (
                            value * weight_y * weight_x
                        )
    return expected


def _analytic_branch_jvp_oracle(
    source: np.ndarray,
    point: tuple[float, float],
    cell: RemapCell,
    tangent: tuple[float, float],
) -> np.ndarray:
    """Differentiate bilinear weights over four independent integer shifts."""

    four_integer_shifts = {
        (offset_y, offset_x): _scatter_oracle(
            source,
            (cell.y + offset_y, cell.x + offset_x),
        )
        for offset_y in (0, 1)
        for offset_x in (0, 1)
    }
    base = four_integer_shifts[0, 0]
    y_shift = four_integer_shifts[1, 0]
    x_shift = four_integer_shifts[0, 1]
    diagonal_shift = four_integer_shifts[1, 1]
    fraction_y = point[0] - cell.y
    fraction_x = point[1] - cell.x
    derivative_y = (1.0 - fraction_x) * (y_shift - base) + fraction_x * (
        diagonal_shift - x_shift
    )
    derivative_x = (1.0 - fraction_y) * (x_shift - base) + fraction_y * (
        diagonal_shift - y_shift
    )
    return tangent[0] * derivative_y + tangent[1] * derivative_x


@pytest.mark.parametrize("shape", ((2, 7), (7, 2)))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_skinny_grid_fractional_transport_matches_scatter_oracle(
    shape: tuple[int, int], dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(SEED + shape[0] + shape[1])
    source = torch.rand(shape, generator=generator, dtype=dtype) + 0.25
    displacement = torch.tensor((-0.375, 0.625), dtype=dtype)

    actual = remap(source, displacement).numpy()
    expected = _scatter_oracle(source.numpy(), tuple(displacement.tolist()))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)
    assert float(actual.sum()) <= float(source.sum()) + 5e-6


def test_skinny_grid_empty_overlap_accepts_massive_mixed_dtype_motion() -> None:
    """An empty overlap still has a finite, connected zero AD path."""

    echo = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    displacement = torch.tensor((1.0e308, -1.0e308), dtype=torch.float64)

    def function(field: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        return remap(field, motion)

    output, tangent = torch.func.jvp(
        function,
        (echo, displacement),
        (torch.ones_like(echo), torch.ones_like(displacement)),
    )
    _, pullback = torch.func.vjp(function, echo, displacement)
    echo_gradient, displacement_gradient = pullback(torch.ones_like(output))

    for value in (output, tangent, echo_gradient, displacement_gradient):
        torch.testing.assert_close(value, torch.zeros_like(value))
        assert bool(torch.isfinite(value).all())
    assert output.dtype is torch.float32
    assert displacement_gradient.dtype is torch.float64


@pytest.mark.parametrize("side", ("below", "above"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_near_integer_auto_branch_matches_frozen_ad_path(
    side: str, dtype: torch.dtype,
) -> None:
    field = torch.arange(20, dtype=dtype).reshape(4, 5)
    epsilon = 16.0 * torch.finfo(dtype).eps
    y = 1.0 - epsilon if side == "below" else 1.0 + epsilon
    point = torch.tensor((y, -0.25), dtype=dtype)
    tangent = torch.tensor((0.3, -0.4), dtype=dtype)
    cell = RemapCell(math.floor(float(y)), -1)

    expected, expected_tangent = torch.func.jvp(
        lambda motion: remap_core(field, motion, cell),
        (point,),
        (tangent,),
    )
    actual, actual_tangent = torch.func.jvp(
        lambda motion: remap(field, motion),
        (point,),
        (tangent,),
    )
    oracle_tangent = _analytic_branch_jvp_oracle(
        field.numpy(),
        tuple(point.tolist()),
        cell,
        tuple(tangent.tolist()),
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_tangent, expected_tangent, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        actual_tangent.numpy(),
        oracle_tangent,
        rtol=4e-6 if dtype is torch.float32 else 1e-12,
        atol=4e-6 if dtype is torch.float32 else 1e-12,
    )
    assert bool(torch.isfinite(actual_tangent).all())


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_pcg_diagonal_preconditioner_handles_dynamic_scale_residual(
    dtype: torch.dtype,
) -> None:
    exponent = 20 if dtype is torch.float32 else 200
    scale = 10.0**exponent
    diagonal = torch.tensor((1.0 / scale, 1.0, scale), dtype=dtype)
    rhs = torch.tensor((1.0, 2.0, 3.0), dtype=dtype)
    expected = rhs / diagonal

    result = pcg(
        lambda value: diagonal * value,
        rhs,
        preconditioner=lambda value: value / diagonal,
        rtol=1.0e-6,
    )
    assert result.converged
    assert result.iterations == 1
    assert result.relative_residual == 0.0
    torch.testing.assert_close(result.solution, expected, rtol=1e-6, atol=0.0)
    direct_residual = rhs - diagonal * result.solution
    torch.testing.assert_close(direct_residual, torch.zeros_like(rhs), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_extreme_scale_metrics_match_finite_scaled_oracle(dtype: torch.dtype) -> None:
    maximum = torch.finfo(dtype).max
    forecast = torch.tensor((maximum, 0.0, 0.0, 0.0), dtype=dtype)
    truth = torch.tensor((-maximum, 0.0, 0.0, 0.0), dtype=dtype)

    # The first difference is 2*maximum and cannot be represented.  The
    # finite mathematical scores are maximum/2 and maximum after averaging.
    torch.testing.assert_close(mae(forecast, truth), forecast.new_tensor(maximum / 2.0))
    torch.testing.assert_close(rmse(forecast, truth), forecast.new_tensor(maximum))
