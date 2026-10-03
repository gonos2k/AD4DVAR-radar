"""Pure algebra contracts for the PR227 20+6 Schur diagnostic."""
import pytest
import torch

from examples.weather_scenarios.fv_partial_schur_probe import (
    SchurScientificRefusal,
    _assemble_hessian,
    _relative,
    schur_algebra,
)


def _quadratic() -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    a = torch.diag(torch.linspace(1.0, 2.0, 20, dtype=dtype))
    b = torch.arange(120, dtype=dtype).reshape(20, 6) / 2000.0
    c = torch.diag(torch.linspace(3.0, 5.0, 6, dtype=dtype))
    h = torch.cat((torch.cat((a, b), 1), torch.cat((b.T, c + b.T @ torch.linalg.solve(a, b)), 1)), 0)
    g = torch.linspace(-0.7, 0.9, 26, dtype=dtype)
    return h, g


def test_schur_step_matches_full_solve_and_keeps_field_gradient_in_rhs() -> None:
    h, g = _quadratic()
    result = schur_algebra(h, g)
    assert result["full_newton_relative_difference"] < 1e-12
    assert result["true_hx_plus_g_relative"] < 1e-12
    assert result["block_residual_relative"] < 1e-12
    assert result["hff_gradient_solve_relative"] < 1e-12
    assert result["hff_cross_solve_relative"] < 1e-12
    assert result["schur_solve_relative"] < 1e-12
    hf = h[:20, :20]
    expected_rhs = -g[20:] + h[20:, :20] @ torch.linalg.solve(hf, g[:20])
    torch.testing.assert_close(result["rhs"], expected_rhs)
    wrong_rhs_step = torch.linalg.solve(result["schur"], -g[20:])
    step = result["step"]
    assert isinstance(step, torch.Tensor)
    assert torch.linalg.vector_norm(wrong_rhs_step - step[20:]) > 1e-3


def test_one_hvp_per_control_column_assembles_quadratic_hessian() -> None:
    h, g = _quadratic()
    objective = lambda x: 0.5 * x @ h @ x + g @ x
    actual = _assemble_hessian(objective, torch.zeros(26, dtype=torch.float64))
    torch.testing.assert_close(actual, h, atol=1e-12, rtol=1e-12)


def test_relative_error_has_no_unit_floor_at_small_scales() -> None:
    a = torch.tensor([2e-20], dtype=torch.float64)
    b = torch.tensor([1e-20], dtype=torch.float64)
    assert _relative(a, b) == 1.0


def test_full_step_relative_difference_uses_the_small_step_norm() -> None:
    h, g = _quadratic()
    result = schur_algebra(h, g * 1e-20)
    expected = result["full_newton_absolute_difference"] / torch.linalg.vector_norm(result["direct_step"]).item()
    assert result["full_newton_relative_difference"] == expected
    assert result["full_newton_relative_difference"] > 1e-20


def test_indefinite_schur_is_reported_and_dense_diagnostic_solve_is_allowed() -> None:
    h, g = _quadratic()
    h[20:, 20:] -= 8.0 * torch.eye(6, dtype=torch.float64)
    result = schur_algebra(h, g)
    eigenvalues = result["schur_eigenvalues"]
    assert isinstance(eigenvalues, torch.Tensor)
    assert eigenvalues[0] < 0
    assert result["full_newton_relative_difference"] < 1e-12


def test_singular_schur_refuses_before_solving() -> None:
    h, g = torch.eye(26, dtype=torch.float64), torch.ones(26, dtype=torch.float64)
    h[20:, 20:] = 0
    with pytest.raises(SchurScientificRefusal, match="Schur complement"):
        schur_algebra(h, g)


def test_schur_symmetry_uses_its_own_scale_when_hff_is_large() -> None:
    h, g = torch.eye(26, dtype=torch.float64), torch.ones(26, dtype=torch.float64)
    h[:20, :20] *= 1e12
    h[20:, 20:] = torch.diag(torch.arange(2, 8, dtype=torch.float64))
    h[20, 21] = 1e-8
    with pytest.raises(ValueError, match="Schur complement symmetry"):
        schur_algebra(h, g)


@pytest.mark.parametrize("mutation", ["indefinite", "nonsymmetric", "nonfinite"])
def test_invalid_field_curvature_or_hessian_contract_refuses(mutation: str) -> None:
    h, g = _quadratic()
    if mutation == "indefinite":
        h[0, 0] = -1.0
    elif mutation == "nonsymmetric":
        h[0, 20] += 1e-3
    else:
        h[0, 0] = torch.inf
    expected = SchurScientificRefusal if mutation == "indefinite" else ValueError
    with pytest.raises(expected):
        schur_algebra(h, g)
