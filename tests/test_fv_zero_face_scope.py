"""Distinguish structural and controlled zero FV faces in research profiles."""

from typing import cast

import pytest
import torch

from advar.transport import bounded_fv_coefficients, face_volume_fluxes
from examples.weather_scenarios import fv_minmod_inverse_probe as small
from examples.weather_scenarios import fv_scaled_research_case as large


def _flow_spec(profile: str):
    if profile == "4x5":
        _, frozen, _, _ = small.make_spatial_case()
    else:
        frozen = large.make_case().frozen
    spec = frozen.fv_transport
    assert spec is not None
    return spec


def _controlled_flux(control, basis, limits, spec):
    coefficients = bounded_fv_coefficients(
        control,
        psi_basis=basis,
        coefficient_limits=limits,
        dt_seconds=60.0 / spec.substeps_per_interval,
        spacing_yx=spec.spacing_yx,
        reconstruction="minmod",
    )
    return face_volume_fluxes(torch.einsum("k,kij->ij", coefficients, basis))


@pytest.mark.parametrize("profile", ["4x5", "8x10"])
def test_every_zero_face_is_flow_control_sensitive_in_supported_profiles(profile):
    spec = _flow_spec(profile)
    zero = spec.coefficient_limits.new_zeros(spec.coefficient_limits.shape)
    x_direction = torch.zeros_like(zero)
    y_direction = torch.zeros_like(zero)
    x_direction[0] = 1
    y_direction[1] = 1

    def flux(control):
        return _controlled_flux(control, spec.psi_basis, spec.coefficient_limits, spec)

    (qx, qy), (dqx, _) = cast(
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        torch.func.jvp(flux, (zero,), (x_direction,)),
    )
    _, (_, dqy) = cast(
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        torch.func.jvp(flux, (zero,), (y_direction,)),
    )
    assert torch.equal(qx, torch.zeros_like(qx))
    assert torch.equal(qy, torch.zeros_like(qy))
    assert bool((dqx.abs() > 0).all())
    assert bool((dqy.abs() > 0).all())


def test_restricted_streamfunction_has_structural_zero_faces():
    spec = _flow_spec("4x5")
    basis = spec.psi_basis[:1]
    limits = spec.coefficient_limits[:1]
    control = limits.new_tensor([0.4])

    def flux(value):
        return _controlled_flux(value, basis, limits, spec)

    (qx, qy), (dqx, dqy) = cast(
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        torch.func.jvp(flux, (control,), (torch.ones_like(control),)),
    )
    assert bool((qx.abs() > 0).all())
    assert bool((dqx.abs() > 0).all())
    assert torch.equal(qy, torch.zeros_like(qy))
    assert torch.equal(dqy, torch.zeros_like(dqy))
