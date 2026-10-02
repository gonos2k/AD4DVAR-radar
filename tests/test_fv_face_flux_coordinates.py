"""Local mathematical checks for the FV face-flux coordinate chart."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from advar.transport import bounded_fv_coefficients, face_volume_fluxes
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart


PR227_CONTROL = [
    0.0017664928222495286, 0.0008520370486646672, 0.0005095373226338965,
    0.0002955685804080073, 0.0008503571863323684, 0.001342524024508611,
    -0.0005384339356051018, -0.00042987647580803256, -0.0004891564587107851,
    0.0006995855765834549, 0.001479852238125385, -0.0007226371180486566,
    -0.0009867246816843247, -0.001088352616625441, -0.0009247899467105755,
    -8.015329707073345e-05, -0.000606676322981361, -0.0012911157382276654,
    -0.00191127224507232, -0.0026456180060602987, 0.030501987157785092,
    -0.01211979985433054, 0.04176354108639636, -0.05849581157867682,
    -0.0619650232075461, 0.022710409711402733,
]
PR227_CONTROL_SHA256 = "a78d1b8e0cd8f997572f5e2a913f7023dcd6333c124ac4b7c9414358f8484e26"


def _production_basis_fixture():
    y, x = torch.meshgrid(torch.arange(5, dtype=torch.float64),
                          torch.arange(6, dtype=torch.float64), indexing="ij")
    basis = torch.stack((y, x, x * y, 0.5 * (x.square() - y.square()), x.square() * y))
    limits = torch.tensor([0.11, 0.08, 0.07, 0.04, 0.03], dtype=torch.float64)
    return SimpleNamespace(psi_basis=basis, coefficient_limits=limits,
                           substeps_per_interval=9, spacing_yx=(10.0, 10.0))


def _physical_coefficients(original, chart, spec):
    flow = original[chart.field_count:chart.field_count + chart.flow_count]
    return bounded_fv_coefficients(
        flow,
        psi_basis=spec.psi_basis,
        coefficient_limits=spec.coefficient_limits,
        dt_seconds=60.0 / spec.substeps_per_interval,
        spacing_yx=spec.spacing_yx,
        reconstruction="minmod",
    )


def _chart(*, axis="y", face=(2, 0), pivot_index=1):
    spec = _production_basis_fixture()
    return FVFaceFluxCoordinateChart.from_basis(
        field_count=20,
        growth_count=1,
        coefficient_limits=spec.coefficient_limits,
        psi_basis=spec.psi_basis,
        axis=axis,
        face=face,
        pivot_index=pivot_index,
    ), spec


def test_pr227_full_control_roundtrips_and_eta_is_production_face_flux():
    chart, spec = _chart()
    original = torch.tensor(PR227_CONTROL, dtype=torch.float64)
    assert hashlib.sha256(original.numpy().tobytes()).hexdigest() == PR227_CONTROL_SHA256

    original_before = original.clone()
    coordinates = chart.to_face_coordinates(original)
    recovered = chart.from_face_coordinates(coordinates)
    coefficients = _physical_coefficients(original, chart, spec)
    _, qy = face_volume_fluxes(torch.einsum("k,kij->ij", coefficients, spec.psi_basis))
    eta_index = chart.field_count + chart.pivot_index

    torch.testing.assert_close(recovered, original, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(original, original_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        coordinates[eta_index], qy[2, 0] / chart.face_scale,
        rtol=2e-12, atol=2e-12,
    )
    assert coordinates.shape == original.shape
    assert chart.layout["face_flux_coordinate"] == eta_index
    assert chart.layout["control_count"] == original.numel()
    metric = chart.identity["metric"]
    assert isinstance(metric, str)
    assert metric.startswith("normalized signed face flux in real arithmetic")


def test_x_face_uses_production_orientation_and_chart_identity_includes_fixed_data():
    y_chart, spec = _chart()
    x_chart, _ = _chart(axis="x", face=(2, 2), pivot_index=2)
    original = torch.tensor(PR227_CONTROL, dtype=torch.float64)
    coordinates = x_chart.to_face_coordinates(original)
    coefficients = _physical_coefficients(original, x_chart, spec)
    qx, _ = face_volume_fluxes(torch.einsum("k,kij->ij", coefficients, spec.psi_basis))
    eta_index = x_chart.field_count + x_chart.pivot_index
    torch.testing.assert_close(coordinates[eta_index], qx[2, 2] / x_chart.face_scale,
                               rtol=2e-14, atol=0.0)
    assert y_chart != x_chart
    assert hash(x_chart) == hash(_chart(axis="x", face=(2, 2), pivot_index=2)[0])


def test_jvp_vjp_and_hessian_follow_the_full_coordinate_chain_rule():
    chart, _ = _chart()
    original = torch.tensor(PR227_CONTROL, dtype=torch.float64)
    coordinates = chart.to_face_coordinates(original)
    direction = torch.linspace(-0.3, 0.4, original.numel(), dtype=original.dtype)
    diagonal = torch.linspace(0.7, 1.3, original.numel(), dtype=original.dtype)

    def objective(control):
        return (0.5 * diagonal * control.square()).sum() + 0.07 * control.sin().sum()

    def in_chart(value):
        return objective(chart.from_face_coordinates(value))

    chart_jacobian = torch.func.jacrev(chart.to_face_coordinates)(original)
    expected_chart_jacobian = torch.eye(original.numel(), dtype=original.dtype)
    eta_index = chart.field_count + chart.pivot_index
    flow_start = chart.field_count
    flow_end = flow_start + chart.flow_count
    expected_chart_jacobian[eta_index] = 0.0
    expected_chart_jacobian[eta_index, flow_start:flow_end] = (
        chart.weights * chart.limits
        * (1 - torch.tanh(original[flow_start:flow_end]).square())
        / chart.face_scale
    )
    torch.testing.assert_close(chart_jacobian, expected_chart_jacobian,
                               rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(in_chart(coordinates), objective(original),
                               rtol=2e-13, atol=2e-13)

    inverse_jvp = torch.func.jvp(chart.from_face_coordinates, (coordinates,), (direction,))
    tangent, jvp = inverse_jvp[0], inverse_jvp[1]
    gradient = torch.func.grad(objective)(tangent)
    objective_jvp = torch.func.jvp(in_chart, (coordinates,), (direction,))
    actual_jvp = objective_jvp[1]
    expected_jvp = torch.dot(gradient, jvp)
    torch.testing.assert_close(actual_jvp, expected_jvp, rtol=2e-12, atol=2e-12)

    inverse_jacobian = torch.func.jacrev(chart.from_face_coordinates)(coordinates)
    actual_vjp = torch.func.vjp(in_chart, coordinates)[1](torch.ones((), dtype=coordinates.dtype))[0]
    expected_vjp = inverse_jacobian.mT @ gradient
    torch.testing.assert_close(actual_vjp, expected_vjp, rtol=2e-12, atol=2e-12)
    inverse_cotangent = torch.linspace(-0.2, 0.3, original.numel(), dtype=original.dtype)
    inverse_vjp = torch.func.vjp(chart.from_face_coordinates, coordinates)[1](inverse_cotangent)[0]
    torch.testing.assert_close(inverse_vjp, inverse_jacobian.mT @ inverse_cotangent,
                               rtol=2e-12, atol=2e-12)

    objective_hessian = torch.func.hessian(objective)(tangent)
    inverse_hessians = torch.stack([
        torch.func.hessian(lambda value: chart.from_face_coordinates(value)[index])(coordinates)
        for index in range(original.numel())
    ])
    expected_hessian = inverse_jacobian.mT @ objective_hessian @ inverse_jacobian
    expected_hessian = expected_hessian + torch.einsum("i,ijk->jk", gradient, inverse_hessians)
    actual_hessian = torch.func.hessian(in_chart)(coordinates)
    torch.testing.assert_close(actual_hessian, expected_hessian, rtol=3e-10, atol=3e-10)


def test_chart_refuses_noninvertible_coordinates_and_protects_fixed_inputs():
    chart, spec = _chart()
    source_limits = spec.coefficient_limits.clone()
    identity = chart.identity
    original = torch.tensor(PR227_CONTROL, dtype=torch.float64)
    coordinates = chart.to_face_coordinates(original)
    pivot = chart.field_count + chart.pivot_index

    spec.coefficient_limits[0] = 9.0
    chart_limits = chart.limits
    chart_limits[0] = 8.0
    chart_weights = chart.weights
    chart_weights[0] = 7.0
    internal_limits = chart._limits
    internal_limits[0] = 6.0
    assert torch.equal(chart.limits, source_limits)
    assert chart.identity == identity

    invalid = coordinates.clone()
    invalid[pivot] = 1e6
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        chart.from_face_coordinates(invalid)
    boundary = torch.zeros_like(coordinates)
    boundary[pivot] = chart.weights[chart.pivot_index].sign()
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        chart.from_face_coordinates(boundary)
    with pytest.raises(ValueError, match="full-control layout"):
        chart.to_face_coordinates(original[:-1])


def test_chart_rejects_saturated_forward_coordinates_and_unrepresentable_scale():
    chart, _ = _chart()
    saturated = torch.zeros(chart.control_count, dtype=torch.float64)
    saturated[chart.field_count + chart.pivot_index] = 20.0
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        chart.to_face_coordinates(saturated)

    for magnitude in (1e-308, 1e308):
        with pytest.raises(ValueError, match="normalization must be finite and positive"):
            FVFaceFluxCoordinateChart(
                field_count=0,
                growth_count=0,
                pivot_index=0,
                limits=torch.tensor([magnitude], dtype=torch.float64),
                weights=torch.tensor([magnitude], dtype=torch.float64),
            )

    with pytest.raises(ValueError, match="products must be finite"):
        FVFaceFluxCoordinateChart(
            field_count=0,
            growth_count=0,
            pivot_index=0,
            limits=torch.tensor([1.0, 1e308], dtype=torch.float64),
            weights=torch.tensor([1.0, 1e308], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="normalized face row"):
        FVFaceFluxCoordinateChart(
            field_count=0,
            growth_count=0,
            pivot_index=0,
            limits=torch.tensor([1e-308, 1.0], dtype=torch.float64),
            weights=torch.tensor([1.0, 2.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="absolute bound"):
        FVFaceFluxCoordinateChart(
            field_count=0,
            growth_count=0,
            pivot_index=0,
            limits=torch.tensor([1e-308, 1.0, 1.0], dtype=torch.float64),
            weights=torch.ones(3, dtype=torch.float64),
        )
    zero_nonpivot = FVFaceFluxCoordinateChart(
        field_count=0,
        growth_count=0,
        pivot_index=0,
        limits=torch.tensor([1.0, 1e308], dtype=torch.float64),
        weights=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert zero_nonpivot.flow_count == 2


def test_normalized_chart_avoids_overflow_from_finite_raw_face_terms():
    chart = FVFaceFluxCoordinateChart(
        field_count=0,
        growth_count=0,
        pivot_index=0,
        limits=torch.tensor([1e308, 1e308], dtype=torch.float64),
        weights=torch.ones(2, dtype=torch.float64),
    )
    original = torch.tensor([2.0, 2.0], dtype=torch.float64)
    coordinates = chart.to_face_coordinates(original)

    torch.testing.assert_close(coordinates, torch.tensor([2 * torch.tanh(original).mean(), 2.0], dtype=torch.float64),
                               rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(chart.from_face_coordinates(coordinates), original,
                               rtol=1e-15, atol=1e-15)

    cancellation_chart = FVFaceFluxCoordinateChart(
        field_count=0,
        growth_count=0,
        pivot_index=0,
        limits=torch.ones(2, dtype=torch.float64),
        weights=torch.tensor([1.0, 1e16], dtype=torch.float64),
    )
    cancellation = torch.tensor([0.1, 1.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        cancellation_chart.to_face_coordinates(cancellation)
    sub_epsilon_cancellation = torch.tensor([1e-20, 1.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        cancellation_chart.to_face_coordinates(sub_epsilon_cancellation)

    cancelling_others_chart = FVFaceFluxCoordinateChart(
        field_count=0,
        growth_count=0,
        pivot_index=0,
        limits=torch.ones(3, dtype=torch.float64),
        weights=torch.tensor([1.0, 1e16, -1e16], dtype=torch.float64),
    )
    cancelling_others = torch.tensor([1e-20, 1.0, 1.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        cancelling_others_chart.to_face_coordinates(cancelling_others)

    with pytest.raises(ValueError, match="face metadata"):
        FVFaceFluxCoordinateChart(
            field_count=0,
            growth_count=0,
            pivot_index=0,
            limits=torch.ones(1, dtype=torch.float64),
            weights=torch.ones(1, dtype=torch.float64),
            face=("z", 0, 0),
        )
