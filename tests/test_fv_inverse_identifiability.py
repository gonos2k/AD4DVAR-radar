"""Inverse-problem limits of the actual FV map, independent of solver success."""

import pytest
import torch

from advar.transport import finite_volume_trajectory


@pytest.mark.parametrize('replay', [False, True])
def test_zero_flow_selected_jacobian_is_not_a_directional_stationarity_test(replay):
    dtype = torch.float64
    q = torch.ones((1, 1), dtype=dtype)
    basis = torch.tensor([[[0., 0.], [1., 1.]]], dtype=dtype)
    empty = tuple(torch.zeros(1, dtype=dtype) for _ in range(4))
    known = tuple(torch.ones(1, dtype=dtype) for _ in range(4))

    def forecast(a):
        frames, _ = finite_volume_trajectory(
            q, q, a.reshape(1), a.new_zeros(()), psi_basis=basis,
            leads=1, substeps_per_interval=1, interval_seconds=.1,
            spacing_yx=(1., 1.), boundary_echo=((empty, empty),),
            boundary_support=((known, known),), reconstruction='donorcell',
            replay=replay,
        )
        return frames[-1, 0, 0]

    zero = torch.zeros((), dtype=dtype)
    value, selected = torch.func.jvp(forecast, (zero,), (torch.ones_like(zero),))
    torch.testing.assert_close(selected, zero, rtol=0, atol=0)
    objective = lambda a: .5 * (forecast(a) - .9).square() + .5*a.square()
    torch.testing.assert_close(torch.func.grad(objective)(zero), zero, rtol=0, atol=0)
    # Exact one-cell SSPRK2: q(a)=1-dt*|a|+dt²*a²/2.
    for sign in (-1., 1.):
        errors = []
        for h in (1e-3, 5e-4):
            a = zero.new_tensor(sign*h)
            expected = 1 - .1*h + .005*h*h
            torch.testing.assert_close(forecast(a), zero.new_tensor(expected), rtol=0, atol=2e-16)
            slope = (forecast(a)-value)/h
            errors.append(abs(float(slope)+.1))
            assert objective(a) < objective(zero)
        assert errors[1] < .51 * errors[0]


@pytest.mark.parametrize('scheme', ['donorcell', 'minmod'])
def test_constant_echo_with_matching_boundary_cannot_identify_rotation(scheme):
    dtype = torch.float64
    y, x = torch.meshgrid(torch.arange(4, dtype=dtype), torch.arange(5, dtype=dtype), indexing='ij')
    rotation = -.5 * ((x-2).square()+(y-1.5).square())
    basis = (rotation-rotation[0, 0]).unsqueeze(0)
    q = torch.full((3, 4), 2., dtype=dtype)
    known = (q.new_ones(3), q.new_ones(3), q.new_ones(4), q.new_ones(4))
    edges = tuple(2*edge for edge in known)

    def forecast(a):
        frames, _ = finite_volume_trajectory(
            q, torch.ones_like(q), a.reshape(1), a.new_zeros(()),
            psi_basis=basis, leads=2, substeps_per_interval=1,
            interval_seconds=.1, spacing_yx=(1., 1.),
            boundary_echo=((edges, edges),)*2, boundary_support=((known, known),)*2,
            reconstruction=scheme,
        )
        return frames

    for a in (-.2, .1, .3):
        point = q.new_tensor(a)
        frames, tangent = torch.func.jvp(forecast, (point,), (torch.ones_like(point),))
        torch.testing.assert_close(frames, q.expand(3, -1, -1), rtol=0, atol=2e-15)
        torch.testing.assert_close(tangent, torch.zeros_like(tangent), rtol=0, atol=2e-15)


def test_constant_echo_psi_null_does_not_certify_mixed_field_psi_derivatives():
    q = torch.ones((1, 1), dtype=torch.float64)
    basis = q.new_tensor([[[0., 0.], [1., 1.]]])
    edges = tuple(q.new_ones(1) for _ in range(4))

    def forecast(c, a):
        frames, _ = finite_volume_trajectory(
            q+c, q, a.reshape(1), a.new_zeros(()), psi_basis=basis,
            leads=1, substeps_per_interval=1, interval_seconds=.1,
            spacing_yx=(1., 1.), boundary_echo=((edges, edges),),
            boundary_support=((edges, edges),), reconstruction='donorcell',
        )
        return frames[-1, 0, 0]

    zero = q.new_zeros(())
    def field_derivative(a):
        return torch.func.jvp(lambda c: forecast(c, a), (zero,), (zero+1,))[1]

    # F(0,a)=1, but dF/dc=1-.1|a|+.005a² is not differentiable at a=0.
    for sign in (-1., 1.):
        a = q.new_tensor(sign*1e-4)
        torch.testing.assert_close(forecast(zero, a), zero+1, rtol=0, atol=0)
        slope = (field_derivative(a)-field_derivative(zero))/a.abs()
        torch.testing.assert_close(slope, zero-.1, rtol=0, atol=6e-7)
    selected_mixed = torch.func.jvp(field_derivative, (zero,), (zero+1,))[1]
    torch.testing.assert_close(selected_mixed, zero, rtol=0, atol=0)
