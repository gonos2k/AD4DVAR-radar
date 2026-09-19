"""The fixed CFL domain and the smooth coefficient chart are distinct checks."""

import itertools

import pytest
import torch

from advar.transport import bounded_fv_coefficients, face_volume_fluxes, finite_volume_step


def _basis(dtype):
    y, x = torch.meshgrid(torch.arange(4, dtype=dtype), torch.arange(5, dtype=dtype), indexing="ij")
    return torch.stack((y, -x, -.5*(x.square()+y.square())))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
def test_every_small_box_corner_and_interior_point_respects_actual_cfl(dtype, scheme):
    basis = _basis(dtype)
    limits = torch.tensor([.1, .12, .015], dtype=dtype)
    zero = torch.zeros(3, dtype=dtype)
    bounded_fv_coefficients(zero, psi_basis=basis, coefficient_limits=limits,
                            dt_seconds=.2, spacing_yx=(1.5, 2.), reconstruction=scheme)
    points = [limits * torch.tensor(signs, dtype=dtype)
              for signs in itertools.product((-1., 1.), repeat=3)]
    generator = torch.Generator().manual_seed(53)
    points += list(limits * (2*torch.rand(20, 3, dtype=dtype, generator=generator)-1))
    q = torch.ones(3, 4, dtype=dtype)
    edges = (q.new_ones(3), q.new_ones(3), q.new_ones(4), q.new_ones(4))
    for coefficients in points:
        psi = (basis * coefficients[:, None, None]).sum(0)
        qx, qy = face_volume_fluxes(psi)
        outgoing = qx[:, 1:].clamp_min(0) + (-qx[:, :-1]).clamp_min(0)
        outgoing = outgoing + qy[1:, :].clamp_min(0) + (-qy[:-1, :]).clamp_min(0)
        assert .2/3 * outgoing.max() < .5
        result = finite_volume_step(q, q, qx, qy, dt_seconds=.2, spacing_yx=(1.5, 2.),
                                     boundary_echo=(edges, edges), boundary_support=(edges, edges),
                                     reconstruction=scheme, max_courant=.5)
        torch.testing.assert_close(result.echo, q, rtol=16*torch.finfo(dtype).eps, atol=0)


def test_unsafe_box_is_rejected_even_at_zero_current_control():
    basis = _basis(torch.float64)
    # A safe current point does not justify an unsafe trial domain.
    with pytest.raises(ValueError, match="CFL domain"):
        bounded_fv_coefficients(torch.zeros(3, dtype=basis.dtype), psi_basis=basis,
                                coefficient_limits=torch.full((3,), 20., dtype=basis.dtype),
                                dt_seconds=1., spacing_yx=(1., 1.))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_near_cap_box_rejection_is_from_rounding_allowance(dtype):
    # One anchored streamfunction basis gives a divergence-free one-cell flow
    # whose exact box CFL is easy to compute.  Keep that analytic bound below
    # the 64*eps cap margin, while the helper's vertex-rounding allowance puts
    # the certified bound above it.
    eps = torch.finfo(dtype).eps
    basis = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=dtype)
    limits = torch.tensor([2.5 * (1.0 - 80.0 * eps)], dtype=dtype)
    psi = (basis * limits[:, None, None]).sum(0)
    qx, qy = face_volume_fluxes(psi)
    outgoing = (
        qx[:, 1:].clamp_min(0.0)
        + (-qx[:, :-1]).clamp_min(0.0)
        + qy[1:, :].clamp_min(0.0)
        + (-qy[:-1, :]).clamp_min(0.0)
    )
    actual_cfl = 0.2 * outgoing.max()
    cap_margin = 0.5 * (1.0 - 64.0 * eps)
    rounding_adjusted_bound = limits[0] * 0.2 * (1.0 + 32.0 * eps)
    assert bool(actual_cfl < 0.5)
    assert bool(actual_cfl <= cap_margin)
    assert bool(rounding_adjusted_bound > cap_margin)
    with pytest.raises(ValueError, match="CFL domain"):
        bounded_fv_coefficients(
            torch.zeros(1, dtype=dtype),
            psi_basis=basis,
            coefficient_limits=limits,
            dt_seconds=0.2,
            spacing_yx=(1.0, 1.0),
            reconstruction="minmod",
        )


def test_chart_preserves_first_and_second_derivatives():
    basis = _basis(torch.float64)
    limits = torch.tensor([.1, .12, .015], dtype=basis.dtype)
    def chart(control):
        return bounded_fv_coefficients(control, psi_basis=basis, coefficient_limits=limits,
                                       dt_seconds=.2, spacing_yx=(1.5, 2.))
    control = torch.tensor([0., -.2, .3], dtype=basis.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(chart, (control,))
    assert torch.autograd.gradgradcheck(chart, (control,))
    torch.testing.assert_close(torch.func.jacfwd(chart)(control), torch.diag(limits*(1-control.tanh().square())))
    extreme = chart(torch.tensor([1e300, -1e300, 0.], dtype=basis.dtype))
    torch.testing.assert_close(extreme, limits * limits.new_tensor([1., -1., 0.]), rtol=0, atol=0)
