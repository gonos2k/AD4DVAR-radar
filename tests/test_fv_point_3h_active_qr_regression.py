from __future__ import annotations

from decimal import Decimal, localcontext
import math

import pytest
import torch

from examples.weather_scenarios.fv_point_3h_subspace_model import (
    solve_two_direction_subspace,
)


def _ill_conditioned_spd_case(
    perturbation: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    H = torch.tensor(
        [[1.0, -1.0], [-1.0, 1.0 + perturbation]], dtype=dtype
    )
    # The small-eigenvalue vector is proportional to (1, 1 - mu). Compute mu
    # from det(H)/lambda_max to avoid cancellation in the eigenvalue formula.
    mu = perturbation / (
        (2.0 + perturbation + math.sqrt(4.0 + perturbation**2)) / 2.0
    )
    u = torch.tensor([1.0, 1.0 - mu], dtype=dtype)
    u = u / torch.linalg.vector_norm(u)
    return H, -u, u


def _decimal_reference(
    perturbation: float = 1.0e-8,
) -> tuple[torch.Tensor, Decimal, Decimal]:
    """Return small eigenvector, eigenvalue, and multiplier at ||beta||=.8.

    Decimal.from_float preserves the actual binary64 matrix entry used by the
    solver, so this is an oracle for the represented input rather than for an
    idealized decimal 1e-8 perturbation.
    """
    with localcontext() as context:
        context.prec = 80
        one = Decimal(1)
        two = Decimal(2)
        h22 = Decimal.from_float(
            float(torch.tensor(1.0 + perturbation, dtype=torch.float64))
        )
        delta = h22 - one
        lambda_max = (two + delta + (Decimal(4) + delta * delta).sqrt()) / two
        eigenvalue = delta / lambda_max
        x = one
        y = one - eigenvalue
        norm = (x * x + y * y).sqrt()
        ux, uy = x / norm, y / norm
        radius = Decimal.from_float(0.8)
        multiplier = eigenvalue / radius - eigenvalue * eigenvalue
        u = torch.tensor([float(ux), float(uy)], dtype=torch.float64)
        return u, eigenvalue, multiplier


def _decimal_active_reference(
    perturbation: float, gradient: torch.Tensor, radius: float
) -> tuple[torch.Tensor, Decimal]:
    """Solve the represented 2D active KKT equations at 80 digit precision."""
    with localcontext() as context:
        context.prec = 80
        zero, one, two = Decimal(0), Decimal(1), Decimal(2)
        a = one
        b = Decimal(-1)
        d = Decimal.from_float(
            float(torch.tensor(1.0 + perturbation, dtype=torch.float64))
        )
        trace = a + d
        discriminant = ((a - d) ** 2 + 4 * b * b).sqrt()
        eigenvalues = ((trace - discriminant) / two,
                       (trace + discriminant) / two)
        eigenvectors = []
        for value in eigenvalues:
            x, y = one, -(a - value) / b
            norm = (x * x + y * y).sqrt()
            eigenvectors.append((x / norm, y / norm))
        g = tuple(Decimal.from_float(float(value)) for value in gradient)
        projections = [sum(v[i] * g[i] for i in range(2)) for v in eigenvectors]
        radius_decimal = Decimal.from_float(radius)

        def coefficients(multiplier: Decimal) -> tuple[Decimal, Decimal]:
            return tuple(
                -eigenvalues[k] * projections[k]
                / (eigenvalues[k] ** 2 + multiplier)
                for k in range(2)
            )

        def norm_squared(multiplier: Decimal) -> Decimal:
            values = coefficients(multiplier)
            return values[0] ** 2 + values[1] ** 2

        low, high = zero, one
        for _ in range(400):
            middle = (low + high) / two
            if norm_squared(middle) > radius_decimal**2:
                low = middle
            else:
                high = middle
        multiplier = (low + high) / two
        spectral_beta = coefficients(multiplier)
        beta = tuple(
            sum(eigenvectors[k][i] * spectral_beta[k] for k in range(2))
            for i in range(2)
        )
        return torch.tensor([float(value) for value in beta], dtype=torch.float64), multiplier


@pytest.mark.parametrize("perturbation", [1.0e-8, 1.0e-9])
def test_active_rank_two_spd_uses_stable_augmented_qr_solution_when_padded(
    perturbation: float,
):
    H, G2, _ = _ill_conditioned_spd_case(perturbation)
    u_reference, small_eigenvalue, multiplier = _decimal_reference(perturbation)
    torch.testing.assert_close(-G2, u_reference, rtol=0.0, atol=2.0e-16)

    # Match the live two-control/27-residual layout while keeping the second
    # residual direction and the 24 extra controls exactly inactive.
    G = torch.zeros(27, dtype=torch.float64)
    G[:2] = G2
    D = torch.eye(26, dtype=torch.float64)[:, :2]
    V = torch.zeros((27, 2), dtype=torch.float64)
    V[:2] = H
    radius = 0.8

    result = solve_two_direction_subspace(G, D, V, radius=radius)

    expected = radius * u_reference
    augmented_condition = 2.0 / math.sqrt(float(multiplier))
    coefficient_tolerance = (
        64.0 * torch.finfo(G.dtype).eps * augmented_condition * radius
    )
    torch.testing.assert_close(
        result.coefficients, expected, rtol=0.0, atol=coefficient_tolerance
    )
    torch.testing.assert_close(
        result.direction[:2], expected,
        rtol=0.0, atol=coefficient_tolerance,
    )
    torch.testing.assert_close(
        result.direction.norm(), G.new_tensor(radius),
        rtol=0.0, atol=32.0 * torch.finfo(G.dtype).eps,
    )
    expected_R = float((Decimal(1) - Decimal.from_float(radius) * small_eigenvalue) ** 2)
    torch.testing.assert_close(
        result.predicted_R, G.new_tensor(expected_R), rtol=0.0,
        atol=32.0 * torch.finfo(G.dtype).eps,
    )
    assert result.diagnostics["full_kkt_status"] == "full_2d_radius_active"
    assert float(multiplier) > 0.0


def test_active_rank_two_spd_coefficients_have_finite_implicit_derivatives():
    H, G2, _ = _ill_conditioned_spd_case()
    G = torch.zeros(27, dtype=torch.float64)
    G[:2] = G2
    D = torch.eye(26, dtype=torch.float64)[:, :2]
    V = torch.zeros((27, 2), dtype=torch.float64)
    V[:2] = H

    def active_coefficients(g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        padded_g = torch.zeros(27, dtype=g.dtype, device=g.device)
        padded_g[:2] = g
        padded_v = torch.zeros((27, 2), dtype=h.dtype, device=h.device)
        padded_v[:2] = h
        result = solve_two_direction_subspace(padded_g, D, padded_v, radius=0.8)
        assert result.diagnostics["full_kkt_status"] in {
            "full_2d_radius_active", "full_2d_radius_active_bracket_endpoint"
        }
        return result.coefficients

    grad_g, grad_h = torch.autograd.grad(
        active_coefficients(G2.requires_grad_(), H.requires_grad_()).sum(),
        (G2, H),
    )
    assert bool(torch.isfinite(grad_g).all())
    assert bool(torch.isfinite(grad_h).all())
    # Scale G while holding its small-eigenvector direction fixed. On the
    # active sphere the exact coefficient vector is invariant to this scale;
    # this checks the implicit multiplier correction without crossing the
    # extremely narrow orthant active-set boundary caused by perturbing H.
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    def scale_invariant_coefficients(value: torch.Tensor) -> torch.Tensor:
        return active_coefficients(value * G2, H).sum().reshape(1)

    assert torch.autograd.gradcheck(
        scale_invariant_coefficients, (scale,), eps=1.0e-6,
        atol=2.0e-7, rtol=2.0e-6,
    )


def test_original_scaled_gradient_active_case_matches_decimal_kkt_reference():
    perturbation = 1.0e-8
    H, _, _ = _ill_conditioned_spd_case(perturbation)
    G = torch.tensor([-1.0e-8, -1.0e-8], dtype=torch.float64)
    radius = 0.8
    expected, multiplier = _decimal_active_reference(perturbation, G, radius)

    result = solve_two_direction_subspace(G, torch.eye(2, dtype=G.dtype), H, radius)

    # Coefficient accuracy is condition scaled. The independent norm equation
    # remains a near-machine-precision target after active-radius closure.
    augmented_condition = 2.0 / math.sqrt(float(multiplier))
    coefficient_tolerance = (
        64.0 * torch.finfo(G.dtype).eps * augmented_condition * radius
    )
    torch.testing.assert_close(
        result.coefficients, expected, rtol=0.0, atol=coefficient_tolerance
    )
    torch.testing.assert_close(
        result.direction.norm(), G.new_tensor(radius), rtol=0.0,
        atol=32.0 * torch.finfo(G.dtype).eps,
    )
    assert result.diagnostics["full_kkt_status"] == "full_2d_radius_active"


def test_near_singular_rank_two_inactive_solution_keeps_finite_gradients():
    H, _, _ = _ill_conditioned_spd_case()
    beta_expected = torch.tensor([0.2, 0.1], dtype=torch.float64)
    G = -(H @ beta_expected)
    G.requires_grad_()
    D = torch.eye(2, dtype=torch.float64)
    V = H.clone().requires_grad_()

    result = solve_two_direction_subspace(G, D, V, radius=0.8)
    gradients = torch.autograd.grad(
        result.predicted_R + result.direction.square().sum(), (G, V)
    )

    assert result.diagnostics["full_kkt_status"] == "full_2d_interior"
    assert bool(torch.isfinite(result.coefficients).all())
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    torch.testing.assert_close(
        result.coefficients, beta_expected, rtol=2.0e-8, atol=2.0e-8
    )
