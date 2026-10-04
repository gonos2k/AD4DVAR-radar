"""Small algebra checks for chart-metric diagnostics; no FV execution."""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart
from examples.weather_scenarios.fv_chart_diagnostics import tangent_metric_diagnostics
from examples.weather_scenarios.fv_response_contract import compose_full_control


def _plane_chart(pivot: int, scale: float = 1.0) -> Tensor:
    chart = FVFaceFluxCoordinateChart(
        field_count=0, growth_count=0, pivot_index=pivot,
        limits=torch.ones(3, dtype=torch.float64),
        weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        face=("x", 3, 4),
    )
    free = [index for index in range(3) if index != pivot]

    def lift(tangent: Tensor) -> Tensor:
        coordinates = torch.zeros(3, dtype=torch.float64)
        coordinates[free] = tangent
        return chart.from_face_coordinates(coordinates)

    jacobian = torch.func.jacrev(lift)(torch.zeros(2, dtype=torch.float64))
    return jacobian @ torch.diag(torch.tensor([1.0, scale], dtype=torch.float64))


def _kkt_tangent_gradient(gradient: Tensor) -> Tensor:
    normal = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    return gradient - normal * torch.dot(normal, gradient) / torch.dot(normal, normal)


@pytest.mark.parametrize("pivot", [0, 1])
@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-10])
def test_metric_norm_matches_kkt_projection_across_scales_and_pivots(pivot: int, scale: float):
    jacobian = _plane_chart(pivot, scale)
    ambient_gradient = torch.tensor([1.5, -0.5, 4.0], dtype=torch.float64)
    coordinate_gradient = jacobian.mT @ ambient_gradient
    result = tangent_metric_diagnostics(jacobian, coordinate_gradient)
    expected = torch.linalg.vector_norm(_kkt_tangent_gradient(ambient_gradient))
    eps = torch.finfo(torch.float64).eps
    budget = 128 * eps * max(jacobian.shape) * result["condition_number"] * torch.linalg.vector_norm(coordinate_gradient)

    assert result["jacobian_rank"] == 2
    assert result["scope"] == "original-standardized-control Euclidean tangent metric"
    assert result["raw_tangent_gradient_inf_norm"] == pytest.approx(float(coordinate_gradient.abs().amax()))
    assert abs(float(result["tangent_gradient_norm"] - expected)) <= float(budget)


def test_generalized_curvature_is_pivot_invariant_at_a_constrained_stationary_point():
    ambient_hessian = torch.tensor(
        [[3.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 4.0]], dtype=torch.float64
    )
    normal = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ambient_gradient = 2.0 * normal  # Tangent-stationary on the linear face through zero.
    spectra = []
    for pivot in (0, 1):
        jacobian = _plane_chart(pivot)
        gradient = jacobian.mT @ ambient_gradient
        chart_hessian = jacobian.mT @ ambient_hessian @ jacobian
        result = tangent_metric_diagnostics(jacobian, gradient, hessian=chart_hessian)
        spectra.append(result["coordinate_normalized_hessian_eigenvalues"])
        assert result["tangent_gradient_norm"] == pytest.approx(0.0, abs=1e-14)
        assert result["hessian_scope"].endswith("no SPD or root certification")
    q, _ = torch.linalg.qr(_plane_chart(0), mode="reduced")
    expected = torch.linalg.eigvalsh(q.mT @ ambient_hessian @ q)
    assert torch.allclose(spectra[0], expected, atol=1e-13, rtol=1e-13)
    assert torch.allclose(spectra[1], expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-10])
def test_nonlinear_chart_metric_does_not_turn_coordinate_rescaling_into_stationarity(scale: float):
    base = torch.tensor([0.1], dtype=torch.float64)
    parameters = torch.empty(0, dtype=torch.float64)
    lift = lambda tangent, _p: torch.stack((scale * tangent[0], (scale * tangent[0]).square()))
    objective = compose_full_control(
        lambda control, _p: 0.252 * control[0], lift
    )
    tangent = base / scale  # All three charts represent the same c=(0.1,0.01).
    coordinate_gradient = torch.func.grad(lambda t: objective(t, parameters))(tangent)
    jacobian = torch.func.jacrev(lambda t: lift(t, parameters))(tangent)
    result = tangent_metric_diagnostics(jacobian, coordinate_gradient)

    assert coordinate_gradient[0] == pytest.approx(0.252 * scale, rel=1e-14)
    assert result["tangent_gradient_norm"] == pytest.approx(0.252 / (1.04**0.5), rel=1e-14)
    raw_gate_value = result["raw_tangent_gradient_inf_norm"]
    if scale == 1e-10:
        assert raw_gate_value < 1e-10
    else:
        assert raw_gate_value >= 1e-10
    ambient_gradient = torch.func.grad(lambda c: 0.252 * c[0])(lift(tangent, parameters))
    assert ambient_gradient.abs().amax() == pytest.approx(0.252)
    assert ambient_gradient.abs().amax() > 1e-10


def test_nonzero_pivot_curvature_uses_the_composed_nonlinear_chart_hessian():
    chart = FVFaceFluxCoordinateChart(
        field_count=0, growth_count=0, pivot_index=1,
        limits=torch.ones(2, dtype=torch.float64),
        weights=torch.tensor([1.0, 2.0], dtype=torch.float64),
        face=("x", 3, 4),
    )
    free_index = 0

    def lift(tangent: Tensor) -> Tensor:
        coordinates = torch.zeros(2, dtype=torch.float64)
        coordinates[free_index] = tangent[0]
        return chart.from_face_coordinates(coordinates)

    point = torch.tensor([0.2], dtype=torch.float64)
    control = lift(point)
    weights = chart.weights
    face_normal = weights * (1.0 - torch.tanh(control).square())
    parameters = torch.empty(0, dtype=torch.float64)
    objective = compose_full_control(
        lambda candidate, _p: torch.dot(face_normal, candidate),
        lambda tangent, _p: lift(tangent),
    )
    coordinate_gradient = torch.func.grad(lambda t: objective(t, parameters))(point)
    jacobian = torch.func.jacrev(lift)(point)
    hessian = torch.func.hessian(lambda t: objective(t, parameters))(point)
    result = tangent_metric_diagnostics(jacobian, coordinate_gradient, hessian=hessian)

    # The ambient objective is linear, so all nonzero restricted curvature is
    # the g·Gamma_tt term; Z^T H_c Z alone would incorrectly be zero.
    assert abs(float(coordinate_gradient[0])) < 1e-13
    assert abs(float(hessian[0, 0])) > 1e-3
    assert abs(float(result["coordinate_normalized_hessian_eigenvalues"][0])) > 1e-3


def test_uniformly_rescaled_full_rank_chart_is_not_rejected_by_absolute_units():
    jacobian = _plane_chart(0) * 1e-40
    ambient_gradient = torch.tensor([1.5, -0.5, 4.0], dtype=torch.float64)
    gradient = jacobian.mT @ ambient_gradient
    result = tangent_metric_diagnostics(jacobian, gradient)
    expected = torch.linalg.vector_norm(_kkt_tangent_gradient(ambient_gradient))
    assert result["jacobian_rank"] == 2
    assert result["condition_number"] == pytest.approx(torch.linalg.cond(_plane_chart(0)))
    assert result["tangent_gradient_norm"] == pytest.approx(float(expected), rel=1e-14)


def test_near_rank_loss_is_refused_using_a_relative_fp64_threshold():
    jacobian = torch.tensor([[1.0, 0.0], [0.0, 1e-20], [0.0, 0.0]], dtype=torch.float64)
    gradient = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="numerically rank deficient"):
        tangent_metric_diagnostics(jacobian, gradient)


def test_stable_tangent_norm_retains_small_nonzero_components():
    jacobian = torch.eye(2, dtype=torch.float64)
    gradient = torch.tensor([1e-200, 1e-200], dtype=torch.float64)
    result = tangent_metric_diagnostics(jacobian, gradient)
    assert result["tangent_gradient_norm"] > 0
    assert result["tangent_gradient_norm"] == pytest.approx(2**0.5 * 1e-200, rel=1e-14, abs=0)
    assert result["raw_tangent_gradient_inf_norm"] > 0
    assert result["raw_tangent_gradient_inf_norm"] == pytest.approx(1e-200, rel=1e-14, abs=0)


def test_finite_inputs_that_overflow_the_tangent_metric_are_refused():
    jacobian = torch.tensor([[1e-310]], dtype=torch.float64)
    coordinate_gradient = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="overflowed to a nonfinite result"):
        tangent_metric_diagnostics(jacobian, coordinate_gradient)


def test_hessian_is_symmetrized_for_spectrum_and_asymmetry_is_reported():
    jacobian = torch.eye(2, dtype=torch.float64)
    gradient = torch.zeros(2, dtype=torch.float64)
    hessian = torch.tensor(
        [[2.0, 1.0 + 4 * torch.finfo(torch.float64).eps], [1.0, 3.0]], dtype=torch.float64
    )
    result = tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)
    symmetric = 0.5 * (hessian + hessian.mT)
    assert torch.allclose(result["coordinate_normalized_hessian_eigenvalues"],
                          torch.linalg.eigvalsh(symmetric))
    assert result["hessian_input_asymmetry_relative"] > 0
    assert result["hessian_input_asymmetry_relative"] <= result[
        "hessian_input_asymmetry_relative_tolerance"
    ]
    assert result["hessian_input_asymmetry_relative_tolerance"] == pytest.approx(
        64 * torch.finfo(torch.float64).eps * 2
    )
    assert result["coordinate_normalized_hessian_pre_symmetry_asymmetry_relative"] == 0


@pytest.mark.parametrize("bad", [
    torch.tensor([[1.0, float("nan")], [0.0, 1.0]], dtype=torch.float64),
    torch.ones((1, 2), dtype=torch.float64),
])
def test_invalid_jacobian_is_refused(bad: Tensor):
    gradient = torch.ones(bad.shape[1], dtype=torch.float64)
    with pytest.raises(ValueError):
        tangent_metric_diagnostics(bad, gradient)


def test_nonfinite_optional_hessian_is_refused():
    jacobian = torch.eye(2, dtype=torch.float64)
    gradient = torch.zeros(2, dtype=torch.float64)
    hessian = torch.tensor([[1.0, 0.0], [0.0, float("inf")]], dtype=torch.float64)
    with pytest.raises(ValueError, match="hessian must be finite"):
        tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)


def test_large_but_representable_hessian_norm_is_retained():
    jacobian = torch.eye(2, dtype=torch.float64)
    gradient = torch.zeros(2, dtype=torch.float64)
    hessian = torch.eye(2, dtype=torch.float64) * 1e308
    assert bool(torch.isfinite(hessian).all())
    result = tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)
    assert torch.equal(result["coordinate_normalized_hessian_eigenvalues"], torch.diag(hessian))


@pytest.mark.parametrize("chart_scale", [1.0, 1e-150])
def test_tiny_hessian_with_material_relative_skew_is_refused_at_any_chart_scale(chart_scale: float):
    jacobian = torch.eye(2, dtype=torch.float64) * chart_scale
    gradient = torch.zeros(2, dtype=torch.float64)
    physical_hessian = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    hessian = physical_hessian * chart_scale**2
    assert bool(torch.isfinite(hessian).all())
    with pytest.raises(ValueError, match="asymmetry exceeds"):
        tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)


@pytest.mark.parametrize(("diagonal", "off_diagonal"), [
    ((1e6, 1e-8), 1e-320),
    ((2.0, 0.5), 5e-324),
])
def test_triangular_solve_skew_is_refused_before_symmetric_projection(
    diagonal: tuple[float, float], off_diagonal: float,
):
    jacobian = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    gradient = torch.zeros(2, dtype=torch.float64)
    hessian = torch.tensor([[0.0, off_diagonal], [off_diagonal, 0.0]], dtype=torch.float64)
    assert bool(torch.isfinite(jacobian).all() & torch.isfinite(hessian).all())
    with pytest.raises(ValueError, match="coordinate-normalized hessian asymmetry"):
        tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)


def test_true_curvature_eigenvalue_overflow_is_refused():
    jacobian = torch.eye(2, dtype=torch.float64)
    gradient = torch.zeros(2, dtype=torch.float64)
    hessian = torch.full((2, 2), 1e308, dtype=torch.float64)
    with pytest.raises(ValueError, match="nonfinite"):
        tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)


@pytest.mark.parametrize("small_curvature", [-1e-16, 1e-16])
def test_symmetrization_preserves_mixed_scale_diagonal_curvature(small_curvature: float):
    jacobian = torch.eye(2, dtype=torch.float64)
    gradient = torch.zeros(2, dtype=torch.float64)
    hessian = torch.diag(torch.tensor([1e308, small_curvature], dtype=torch.float64))
    result = tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)
    eigenvalues = result["coordinate_normalized_hessian_eigenvalues"]
    assert torch.sign(eigenvalues[0]) == torch.sign(torch.tensor(small_curvature))
    assert eigenvalues[0] == pytest.approx(small_curvature, rel=2e-15, abs=0)
    assert eigenvalues[1] == pytest.approx(1e308, rel=2e-15, abs=0)


def test_symmetrization_preserves_representable_subnormal_diagonal_entries():
    jacobian = torch.eye(2, dtype=torch.float64)
    gradient = torch.zeros(2, dtype=torch.float64)
    smallest_subnormal = torch.nextafter(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    hessian = torch.eye(2, dtype=torch.float64) * smallest_subnormal
    result = tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)
    expected = smallest_subnormal.expand(2).clone()
    assert torch.equal(result["coordinate_normalized_hessian_eigenvalues"], expected)


def test_finite_hessian_with_overflowing_coordinate_normalization_is_refused():
    jacobian = torch.tensor([[1e-310]], dtype=torch.float64)
    gradient = torch.zeros(1, dtype=torch.float64)
    hessian = torch.ones((1, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="nonfinite result|overflowed"):
        tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)


def test_materially_asymmetric_hessian_is_rejected_with_a_relative_tolerance():
    jacobian = torch.eye(2, dtype=torch.float64)
    gradient = torch.zeros(2, dtype=torch.float64)
    hessian = torch.tensor([[2.0, 1.2], [0.8, 3.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="asymmetry exceeds"):
        tangent_metric_diagnostics(jacobian, gradient, hessian=hessian)
