from __future__ import annotations

import math

import pytest
import torch

from examples.weather_scenarios.fv_point_3h_subspace_model import solve_two_direction_subspace


def test_rank_one_exact_fit_inside_radius_is_not_sent_to_singular_lambda_zero():
    G = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    D = torch.eye(2, dtype=G.dtype)
    V = torch.ones((2, 2), dtype=G.dtype)

    result = solve_two_direction_subspace(G, D, V, radius=0.8)

    torch.testing.assert_close(result.coefficients, torch.tensor([0.5, 0.5], dtype=G.dtype))
    torch.testing.assert_close(result.predicted_R, torch.zeros((), dtype=G.dtype), atol=1e-28, rtol=0)
    torch.testing.assert_close(result.direction.norm(), torch.tensor(math.sqrt(0.5), dtype=G.dtype))
    assert result.direction.norm() < 0.8


def test_rank_one_exact_fit_uses_control_metric_not_euclidean_pseudoinverse():
    # Two unit directions at -60 degrees; H is a smooth, constant physical Jacobian.
    D = torch.zeros((26, 2), dtype=torch.float64)
    D[0, 0] = 1.0
    D[0, 1] = -0.5
    D[1, 1] = math.sqrt(3.0) / 2.0
    H = torch.zeros((27, 26), dtype=D.dtype)
    H[0, 0] = 1.0
    H[0, 1] = 5.0 / math.sqrt(3.0)
    V = H @ D
    G = torch.zeros(27, dtype=D.dtype)
    G[0] = -1.0
    radius = 0.34

    result = solve_two_direction_subspace(G, D, V, radius=radius)
    euclidean_pseudoinverse = torch.linalg.pinv(V) @ -G
    expected = torch.tensor([2.0 / 7.0, 5.0 / 14.0], dtype=D.dtype)

    torch.testing.assert_close(result.coefficients, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(result.predicted_R, torch.zeros((), dtype=D.dtype), atol=1e-26, rtol=0)
    assert result.direction.norm() < radius
    assert torch.linalg.vector_norm(D @ euclidean_pseudoinverse) > radius


def test_smooth_quadratic_gradient_and_hessian_directions_have_global_zero_inside_radius():
    # J(x,y)=0.5(x+y-1)^2 at (0,0): g=(-1,-1), H=ones. The second
    # positive direction comes from a distinct SPD preconditioner K.
    x0 = torch.zeros(2, dtype=torch.float64)
    G = (x0.sum() - 1.0) * torch.ones(2, dtype=x0.dtype)
    H = torch.ones((2, 2), dtype=x0.dtype)
    K = torch.diag(torch.tensor([1.0, 2.0], dtype=x0.dtype))
    D = torch.stack((-G, -torch.linalg.solve(K, G)), dim=1)
    V = H @ D
    radius = 0.8

    result = solve_two_direction_subspace(G, D, V, radius=radius)
    x1 = x0 + result.direction
    exact_gradient = (x1.sum() - 1.0) * torch.ones_like(x1)

    torch.testing.assert_close(exact_gradient, torch.zeros_like(G), atol=2e-12, rtol=0)
    torch.testing.assert_close(result.predicted_R, torch.zeros((), dtype=G.dtype), atol=1e-26, rtol=0)
    assert result.direction.norm() < radius


def test_rank_one_metric_candidate_has_finite_gradients_on_fixed_rank_manifold():
    G = torch.tensor([-1.0, 0.0], dtype=torch.float64, requires_grad=True)
    D = torch.tensor([[1.0, -0.5], [0.0, math.sqrt(0.75)]], dtype=G.dtype, requires_grad=True)
    u_parameter = torch.tensor([1.0, 0.1], dtype=G.dtype, requires_grad=True)
    w = torch.tensor([1.0, 2.0], dtype=G.dtype, requires_grad=True)

    def coefficients(g: torch.Tensor, d: torch.Tensor, u: torch.Tensor,
                     direction_weights: torch.Tensor) -> torch.Tensor:
        unit_u = u / u.norm()
        V = unit_u[:, None] * direction_weights[None, :]
        return solve_two_direction_subspace(g, d, V, radius=0.34).coefficients

    assert torch.autograd.gradcheck(coefficients, (G, D, u_parameter, w),
                                    eps=1e-6, atol=3e-5, rtol=3e-4)


def test_rank_one_active_radius_has_finite_gradients_on_fixed_rank_manifold():
    G = torch.tensor([-2.0, 0.0], dtype=torch.float64, requires_grad=True)
    D = torch.eye(2, dtype=G.dtype, requires_grad=True)
    u_parameter = torch.tensor([1.0, 0.1], dtype=G.dtype, requires_grad=True)
    w = torch.tensor([1.0, 1.0], dtype=G.dtype, requires_grad=True)

    def coefficients(g: torch.Tensor, d: torch.Tensor, u: torch.Tensor,
                     direction_weights: torch.Tensor) -> torch.Tensor:
        unit_u = u / u.norm()
        V = unit_u[:, None] * direction_weights[None, :]
        result = solve_two_direction_subspace(g, d, V, radius=0.6)
        assert result.diagnostics["full_kkt_status"] == "rank_one_radius_active"
        return result.coefficients

    assert torch.autograd.gradcheck(coefficients, (G, D, u_parameter, w),
                                    eps=1e-6, atol=3e-5, rtol=3e-4)


def test_rank_one_exact_fit_padded_to_live_26_control_27_residual_shapes():
    G = torch.zeros(27, dtype=torch.float64)
    G[:2] = -1.0
    D = torch.eye(26, dtype=G.dtype)[:, :2]
    V = torch.zeros((27, 2), dtype=G.dtype)
    V[:2, :] = 1.0

    result = solve_two_direction_subspace(G, D, V, radius=0.8)

    torch.testing.assert_close(result.coefficients, torch.tensor([0.5, 0.5], dtype=G.dtype))
    assert result.direction.shape == (26,)
    assert result.residual_direction.shape == (27,)
    torch.testing.assert_close(result.predicted_R, torch.zeros((), dtype=G.dtype), atol=1e-28, rtol=0)


def test_rank_one_radius_active_minimum_norm_solution_hits_radius():
    G = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    D = torch.eye(2, dtype=G.dtype)
    V = torch.ones((2, 2), dtype=G.dtype)
    radius = 0.6

    result = solve_two_direction_subspace(G, D, V, radius=radius)

    expected = torch.full((2,), radius / math.sqrt(2.0), dtype=G.dtype)
    expected_R = 2.0 * (1.0 - math.sqrt(2.0) * radius) ** 2
    torch.testing.assert_close(result.coefficients, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(result.direction.norm(), torch.tensor(radius, dtype=G.dtype), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(result.predicted_R, torch.tensor(expected_R, dtype=G.dtype), atol=2e-12, rtol=2e-12)


def test_rank_one_metric_minimum_outside_positive_orthant_uses_axis_boundary():
    # The exact-fit affine line is beta0 + 0.1*beta1 = 1. Its unconstrained
    # control-metric minimum has beta1 < 0; the positive feasible endpoint is axis 0.
    correlation = 0.9
    D = torch.zeros((26, 2), dtype=torch.float64)
    D[0, 0] = 1.0
    D[0, 1] = correlation
    D[1, 1] = math.sqrt(1.0 - correlation**2)
    H = torch.zeros((27, 26), dtype=D.dtype)
    H[0, 0] = 1.0
    H[0, 1] = (0.1 - correlation) / math.sqrt(1.0 - correlation**2)
    V = H @ D
    G = torch.zeros(27, dtype=D.dtype)
    G[0] = -1.0

    result = solve_two_direction_subspace(G, D, V, radius=0.9)

    torch.testing.assert_close(result.coefficients, torch.tensor([0.9, 0.0], dtype=D.dtype), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(result.predicted_R, torch.tensor(0.01, dtype=D.dtype), atol=2e-12, rtol=2e-12)


def test_zero_residual_directions_choose_zero_and_keep_inactive_gradients_finite():
    G = torch.tensor([-1.0, 0.5], dtype=torch.float64, requires_grad=True)
    D = torch.eye(2, dtype=G.dtype, requires_grad=True)
    V = torch.zeros((2, 2), dtype=G.dtype, requires_grad=True)

    result = solve_two_direction_subspace(G, D, V, radius=0.8)

    torch.testing.assert_close(result.coefficients, torch.zeros(2, dtype=G.dtype))
    torch.testing.assert_close(result.predicted_R, torch.dot(G, G))
    gradients = torch.autograd.grad(result.predicted_R + result.direction.square().sum(), (G, D, V))
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


@pytest.mark.parametrize("delta", [5e-13, 2e-12])
def test_near_rank_transition_keeps_solution_and_inactive_branch_gradients_finite(delta: float):
    G = torch.zeros(27, dtype=torch.float64, requires_grad=True)
    D = torch.eye(26, dtype=G.dtype)[:, :2]
    V = torch.zeros((27, 2), dtype=G.dtype)
    V[0, :] = 1.0
    V[1, 1] = delta
    G_data = -(V @ torch.tensor([0.5, 0.5], dtype=G.dtype))
    G = G_data.requires_grad_()

    result = solve_two_direction_subspace(G, D, V, radius=0.8)
    gradients = torch.autograd.grad(result.predicted_R + result.direction.sum(), G)

    assert bool(torch.isfinite(result.coefficients).all())
    assert bool(torch.isfinite(result.predicted_R))
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert torch.linalg.vector_norm(result.direction) <= 0.8 * (1.0 + 1e-12)
