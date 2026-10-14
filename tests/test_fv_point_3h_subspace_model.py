from __future__ import annotations

import pytest
import torch

from examples.weather_scenarios.fv_point_3h_subspace_model import (
    residual_model_fidelity,
    solve_two_direction_subspace,
)


def test_unconstrained_two_column_minimum_and_autograd():
    G = torch.tensor([-2.0, -1.0], dtype=torch.float64, requires_grad=True)
    D = torch.eye(2, dtype=torch.float64)
    V = torch.eye(2, dtype=torch.float64)

    result = solve_two_direction_subspace(G, D, V, radius=3.0)

    torch.testing.assert_close(result.coefficients, torch.tensor([2.0, 1.0], dtype=G.dtype))
    torch.testing.assert_close(result.direction, torch.tensor([2.0, 1.0], dtype=G.dtype))
    torch.testing.assert_close(result.residual_direction, -G)
    torch.testing.assert_close(result.predicted_R, torch.zeros((), dtype=G.dtype), atol=1e-28, rtol=0)
    assert result.diagnostics["selected_active_set"] == "full_2d"
    grad = torch.autograd.grad(result.predicted_R + result.direction.sum(), G)[0]
    assert torch.isfinite(grad).all()


def test_radius_active_kkt_is_feasible_and_beats_axes():
    G = torch.tensor([-2.0, -1.0], dtype=torch.float64)
    D = torch.eye(2, dtype=G.dtype)
    V = torch.eye(2, dtype=G.dtype)

    result = solve_two_direction_subspace(G, D, V, radius=1.0)

    assert result.diagnostics["selected_active_set"] == "full_2d"
    torch.testing.assert_close(torch.linalg.vector_norm(result.direction), torch.tensor(1.0, dtype=G.dtype),
                               atol=1e-12, rtol=1e-12)
    assert float(result.predicted_R) < 2.0  # best one-axis model has R=2
    assert bool((result.coefficients >= 0.0).all())


def test_signed_quadrant_can_cancel_residual_when_nonnegative_cone_cannot():
    # This is a standalone Euclidean linear-residual example, not an FV run.
    # The Hessian scales the two model columns differently, so the exact
    # residual cancellation needs a negative coefficient on the second axis.
    dtype = torch.float64
    G = torch.tensor([1.0, 1.0], dtype=dtype)
    hessian = torch.diag(torch.tensor([1.0, 10.0], dtype=dtype))
    D = torch.tensor([[-1.0, -1.0], [-1.0, -10.0]], dtype=dtype)
    V = hessian @ D
    radius = 2.0

    cone = solve_two_direction_subspace(G, D, V, radius)
    torch.testing.assert_close(cone.coefficients,
        torch.tensor([11.0 / 101.0, 0.0], dtype=dtype), atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(cone.predicted_R,
        torch.tensor(8181.0 / 10201.0, dtype=dtype), atol=2e-13, rtol=2e-13)
    assert torch.linalg.vector_norm(cone.direction) < radius

    signs = torch.tensor([1.0, -1.0], dtype=dtype)
    signed = solve_two_direction_subspace(G, D * signs, V * signs, radius)
    original_coefficients = signed.coefficients * signs
    torch.testing.assert_close(original_coefficients,
        torch.tensor([1.1, -0.1], dtype=dtype), atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(G + V @ original_coefficients,
        torch.zeros_like(G), atol=2e-13, rtol=0.0)
    torch.testing.assert_close(signed.predicted_R,
        torch.zeros((), dtype=dtype), atol=2e-26, rtol=0.0)
    assert torch.linalg.vector_norm(D @ original_coefficients) < radius
    assert torch.dot(G, D @ original_coefficients) < 0.0


def test_radius_active_coefficients_have_implicit_kkt_gradient():
    D = torch.eye(2, dtype=torch.float64)
    V = torch.eye(2, dtype=torch.float64, requires_grad=True)
    G = torch.tensor([-2.0, -1.0], dtype=torch.float64, requires_grad=True)

    def coefficients(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return solve_two_direction_subspace(g, D, v, radius=1.0).coefficients

    assert torch.autograd.gradcheck(coefficients, (G, V), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_skew_control_and_residual_columns_have_radius_active_gradients():
    D = torch.tensor([[1.7, 0.2], [0.1, 1.3], [0.4, -0.3]], dtype=torch.float64, requires_grad=True)
    V = torch.tensor([[1.2, 0.3], [0.2, 1.4], [0.5, 0.1]], dtype=torch.float64, requires_grad=True)
    G = torch.tensor([-1.8, -1.2, -0.7], dtype=torch.float64, requires_grad=True)

    def coefficients(g: torch.Tensor, d: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        result = solve_two_direction_subspace(g, d, v, radius=0.22)
        assert result.diagnostics["selected_active_set"] == "full_2d"
        return result.coefficients

    assert torch.autograd.gradcheck(coefficients, (G, D, V), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_deterministic_40_case_dense_angle_oracle_and_finite_gradients():
    generator = torch.Generator().manual_seed(20261012)
    angles = torch.linspace(0.0, torch.pi / 2, 4097, dtype=torch.float64)
    unit_coefficients = torch.stack((angles.cos(), angles.sin()), dim=1)
    for _ in range(40):
        G = torch.randn(7, generator=generator, dtype=torch.float64).requires_grad_()
        D = torch.randn(9, 2, generator=generator, dtype=torch.float64)
        V = torch.randn(7, 2, generator=generator, dtype=torch.float64)
        radius = 0.05 + 0.4 * torch.rand((), generator=generator, dtype=torch.float64)
        result = solve_two_direction_subspace(G, D, V, float(radius))

        assert torch.isfinite(result.predicted_R)
        assert torch.isfinite(result.coefficients).all()
        assert torch.linalg.vector_norm(result.direction) <= radius * (1.0 + 1e-12)
        gradient = torch.autograd.grad(result.predicted_R, G)[0]
        assert torch.isfinite(gradient).all()

        scales = torch.linalg.vector_norm(D, dim=0)
        Dn, Vn = D / scales, V / scales
        movement_per_ray = torch.linalg.vector_norm(unit_coefficients @ Dn.T, dim=1)
        residual_per_ray = unit_coefficients @ Vn.T
        linear = residual_per_ray @ G
        curvature = residual_per_ray.square().sum(dim=1)
        maximum_step = float(radius) / movement_per_ray
        ray_step = torch.where(curvature > 0.0, -linear / curvature.clamp_min(torch.finfo(G.dtype).tiny), 0.0)
        ray_step = torch.minimum(torch.clamp_min(ray_step, 0.0), maximum_step)
        ray_residual = G + ray_step[:, None] * residual_per_ray
        oracle_R = ray_residual.square().sum(dim=1).min()
        oracle_tolerance = 2e-9 * torch.maximum(torch.ones_like(oracle_R), oracle_R)
        assert bool(result.predicted_R <= oracle_R + oracle_tolerance)


def test_column_normalization_preserves_physical_step_and_raw_coefficients():
    G = torch.tensor([-0.8, -0.6], dtype=torch.float64)
    D = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=G.dtype)
    V = D.clone()

    result = solve_two_direction_subspace(G, D, V, radius=1.0)

    torch.testing.assert_close(result.direction, torch.tensor([0.8, 0.6], dtype=G.dtype),
                               atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.coefficients, torch.tensor([0.4, 0.15], dtype=G.dtype),
                               atol=1e-12, rtol=1e-12)


def test_same_ray_dependence_falls_back_to_best_axis_and_flags_reduced_domain():
    G = torch.tensor([-1.0, -0.5], dtype=torch.float64, requires_grad=True)
    D = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=G.dtype)
    V = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=G.dtype)

    result = solve_two_direction_subspace(G, D, V, radius=0.5)

    assert result.diagnostics["reduced_domain"] is True
    assert result.diagnostics["full_kkt_status"] == "dependent_axis_fallback"
    assert result.diagnostics["selected_active_set"] in {"axis_0", "axis_1"}
    torch.testing.assert_close(torch.linalg.vector_norm(result.direction), torch.tensor(0.5, dtype=G.dtype),
                               atol=1e-12, rtol=1e-12)
    grad = torch.autograd.grad(result.predicted_R + result.direction.sum(), G)[0]
    assert torch.isfinite(grad).all()


def test_near_antiparallel_directions_are_refused_before_cancellation():
    G = torch.tensor([1.0, 0.0], dtype=torch.float64)
    D = torch.tensor([[1.0, -1.0], [0.0, 1e-12]], dtype=G.dtype)
    V = torch.tensor([[-1.0, 1.0], [0.0, 1e-12]], dtype=G.dtype)

    with pytest.raises(ValueError, match="antiparallel"):
        solve_two_direction_subspace(G, D, V, radius=0.1)


def test_fidelity_keeps_full_residual_and_step_change_errors_distinct():
    G = torch.tensor([1.0, 0.0], dtype=torch.float64)
    predicted = torch.tensor([0.0, 0.0], dtype=G.dtype)
    actual = torch.tensor([0.0, 10.0], dtype=G.dtype)

    fidelity = residual_model_fidelity(G, predicted, actual)

    torch.testing.assert_close(fidelity["full_vector_error"], torch.tensor(10.0, dtype=G.dtype))
    torch.testing.assert_close(fidelity["delta_vector_error"], torch.tensor(10.0, dtype=G.dtype))
    torch.testing.assert_close(fidelity["predicted_R"], torch.tensor(0.0, dtype=G.dtype))
    torch.testing.assert_close(fidelity["actual_R"], torch.tensor(100.0, dtype=G.dtype))
    assert fidelity["relative_R_error"] > 0.99


def test_actual_merit_gate_must_reject_a_model_that_predicts_gain():
    G = torch.tensor([1.0], dtype=torch.float64)
    predicted = torch.tensor([0.95], dtype=G.dtype)
    # A nonlinear remainder at the actual candidate overwhelms the linear model.
    actual = torch.tensor([1.2], dtype=G.dtype)

    fidelity = residual_model_fidelity(G, predicted, actual)

    assert fidelity["predicted_R"] < torch.dot(G, G)
    assert fidelity["actual_R"] > torch.dot(G, G)


def test_invalid_shapes_zero_columns_and_radius_are_refused():
    G = torch.tensor([1.0, 0.0], dtype=torch.float64)
    V = torch.eye(2, dtype=G.dtype)
    with pytest.raises(ValueError, match="exactly two"):
        solve_two_direction_subspace(G, torch.ones((2, 1), dtype=G.dtype), V, radius=1.0)
    with pytest.raises(ValueError, match="nonzero"):
        solve_two_direction_subspace(G, torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=G.dtype), V, radius=1.0)
    with pytest.raises(ValueError, match="positive"):
        solve_two_direction_subspace(G, torch.eye(2, dtype=G.dtype), V, radius=0.0)
