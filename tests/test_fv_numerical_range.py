"""Numerical range regressions for the positive FV step."""

import math
from decimal import Decimal, localcontext

import pytest
import torch

from advar.transport import finite_volume_step


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
def test_stationary_minimum_subnormal_and_its_derivative_are_preserved(dtype, scheme):
    tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype))
    q = tiny.reshape(1, 1).requires_grad_()

    def advance(field):
        return finite_volume_step(
            field, torch.ones_like(field), field.new_zeros(1, 2),
            field.new_zeros(2, 1), dt_seconds=1., spacing_yx=(1., 1.),
            reconstruction=scheme,
        ).echo

    actual = advance(q)
    torch.testing.assert_close(actual, q, rtol=0, atol=0)
    gradient, = torch.autograd.grad(actual.sum(), q)
    torch.testing.assert_close(gradient, torch.ones_like(q), rtol=0, atol=0)
    _, tangent = torch.func.jvp(advance, (q,), (torch.ones_like(q),))
    torch.testing.assert_close(tangent, torch.ones_like(q), rtol=0, atol=0)


def _one_cell(boundary, growth, courant, scheme):
    q = boundary.new_zeros(1, 1)
    zero, one = boundary.new_zeros(1), boundary.new_ones(1)
    edges = tuple((value.reshape(1), zero, zero, zero) for value in boundary)
    return finite_volume_step(
        q, torch.ones_like(q), q.new_full((1, 2), courant), q.new_zeros(2, 1),
        dt_seconds=1., spacing_yx=(1., 1.), log_growth=growth,
        boundary_echo=edges, boundary_support=((one, one, one, one),) * 2,
        reconstruction=scheme,
    )


@pytest.mark.parametrize("dtype,growth", [(torch.float32, 80.), (torch.float64, 700.)])
@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
def test_tiny_boundary_is_amplified_before_transport_rounding(dtype, growth, scheme):
    tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype))
    boundary = tiny.repeat(2).requires_grad_()
    result = _one_cell(boundary, growth, .5, scheme)
    with localcontext() as context:
        context.prec = 100
        t = Decimal.from_float(tiny.item())
        expected = float(Decimal('0.125') * Decimal(growth).exp() * t + Decimal('.25') * t)
    torch.testing.assert_close(result.echo, result.echo.new_full((1, 1), expected),
                               rtol=8*torch.finfo(dtype).eps, atol=0)
    gradient, = torch.autograd.grad(result.echo.sum(), boundary)
    expected_gradient = torch.stack((torch.exp(boundary.new_tensor(growth)) * .125,
                                     boundary.new_tensor(.25)))
    torch.testing.assert_close(gradient, expected_gradient, rtol=8*torch.finfo(dtype).eps, atol=0)


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
def test_large_decaying_boundary_has_finite_transformed_budget(scheme):
    boundary = torch.tensor([0., 1e4], dtype=torch.float32, requires_grad=True)
    result = _one_cell(boundary, -80., .1, scheme)
    torch.testing.assert_close(result.echo, torch.tensor([[500.]]), rtol=1e-6, atol=0)
    expected_budget = torch.exp(torch.tensor(80.)) * 500.
    torch.testing.assert_close(result.transport_inflow, expected_budget, rtol=1e-6, atol=0)
    gradient, = torch.autograd.grad(result.echo.sum(), boundary)
    torch.testing.assert_close(gradient[1], torch.tensor(.05), rtol=1e-6, atol=0)


def test_unrepresentable_transformed_budget_is_explicitly_rejected():
    # The physical answer exists, but the returned transformed budget does not.
    with pytest.raises(FloatingPointError, match="transport_inflow"):
        _one_cell(torch.tensor([0., 1e100], dtype=torch.float64), -700., .1, "donorcell")


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
def test_large_growing_boundary_is_transported_before_scaling_when_needed(scheme):
    boundary = torch.tensor([1e38, 0.], dtype=torch.float32)
    result = _one_cell(boundary, 2., .01, scheme)
    # exp(2)*b0 overflows, but its CFL-weighted contribution is representable.
    expected = .5 * .01 * (1 - .01) * float(boundary[0]) * math.exp(2.)
    torch.testing.assert_close(result.echo, result.echo.new_full((1, 1), expected),
                               rtol=8*torch.finfo(boundary.dtype).eps, atol=0)


@pytest.mark.parametrize("dtype,growth", [(torch.float32, -80.), (torch.float64, -700.)])
def test_decay_transformed_inflow_retains_tiny_boundary(dtype, growth):
    tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype))
    result = _one_cell(torch.stack((torch.zeros_like(tiny), tiny)), growth, .5, "donorcell")
    with localcontext() as context:
        context.prec = 100
        expected = float(Decimal.from_float(tiny.item()) * Decimal(-growth).exp() / 4)
    torch.testing.assert_close(result.transport_inflow, tiny.new_tensor(expected),
                               rtol=8*torch.finfo(dtype).eps, atol=0)


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
@pytest.mark.parametrize("log_growth", [-1e-8, 0., 1e-8])
def test_growth_branch_has_the_analytic_first_and_second_derivatives(scheme, log_growth):
    boundary = torch.tensor([2., 3.], dtype=torch.float64)
    growth = boundary.new_tensor(log_growth, requires_grad=True)
    result = _one_cell(boundary, growth, .2, scheme)
    derivative, = torch.autograd.grad(result.echo.sum(), growth, create_graph=True)
    second, = torch.autograd.grad(derivative, growth)
    expected = .5 * .2 * .8 * boundary[0] * growth.exp()
    torch.testing.assert_close(derivative, expected, rtol=1e-13, atol=0)
    torch.testing.assert_close(second, expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
@pytest.mark.parametrize("growth", [-80., 80.])
def test_large_unused_boundary_does_not_erase_tiny_inflow(scheme, growth):
    q = torch.zeros(1, 1, dtype=torch.float32)
    tiny = torch.nextafter(q.new_tensor(0.), q.new_tensor(1.)).reshape(1)
    zero, one = q.new_zeros(1), q.new_ones(1)
    huge = q.new_full((1,), torch.finfo(q.dtype).max)
    mixed, empty = (tiny, huge, zero, zero), (zero, zero, zero, zero)
    result = finite_volume_step(
        q, torch.ones_like(q), q.new_full((1, 2), 1e-5), q.new_zeros(2, 1),
        dt_seconds=1., spacing_yx=(1., 1.), log_growth=growth,
        boundary_echo=(mixed, empty) if growth > 0 else (empty, mixed),
        boundary_support=((one, one, one, one),) * 2, reconstruction=scheme,
    )
    expected = .5 * 1e-5 * tiny.double() * math.exp(abs(growth))
    if growth > 0:
        expected = expected * (1-1e-5)
    actual = result.echo.flatten() if growth > 0 else result.transport_inflow.reshape(1)
    torch.testing.assert_close(actual, expected.float(),
                               rtol=8*torch.finfo(q.dtype).eps, atol=0)
