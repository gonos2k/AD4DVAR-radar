from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from examples.weather_scenarios.fv_point_3h_subspace_model import solve_two_direction_subspace

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "graphify-out/fv-root-cause-20260919/subspace_rank_fix_20261012/axis_radius_audit.py"
spec = importlib.util.spec_from_file_location("subspace_radius_audit", AUDIT_PATH)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def test_rank_one_lambda_zero_representative_is_minimum_control_norm_and_feasible() -> None:
    dtype = torch.float64
    G = torch.tensor([-1.0, 0.0, 0.0], dtype=dtype)
    D = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    V = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=dtype)

    result = audit.rank_deficient_lambda_zero(G, D, V, radius=0.8)

    torch.testing.assert_close(result["beta"], torch.tensor([0.5, 0.5], dtype=dtype))
    assert result["feasible_at_lambda_zero"] is True
    assert result["movement_norm"] == pytest.approx(2**-0.5)
    assert result["predicted_R"] == pytest.approx(0.0, abs=1e-28)
    assert result["equation_error_l2"] < 1e-14


def test_current_solver_uses_rank_one_lambda_zero_interior_without_singular_solve() -> None:
    dtype = torch.float64
    G = torch.tensor([-1.0, 0.0, 0.0], dtype=dtype, requires_grad=True)
    D = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    V = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=dtype)

    result = solve_two_direction_subspace(G, D, V, radius=0.8)

    assert result.diagnostics["full_kkt_status"] == "rank_one_interior"
    torch.testing.assert_close(result.coefficients, torch.tensor([0.5, 0.5], dtype=dtype))
    torch.testing.assert_close(result.predicted_R, torch.tensor(0.0, dtype=dtype), atol=1e-28, rtol=0.0)
    movement = result.diagnostics["movement_norm"]
    assert isinstance(movement, torch.Tensor) and bool(movement < 0.8)
    derivative = torch.autograd.grad(result.predicted_R + result.direction.sum(), G)[0]
    assert bool(torch.isfinite(derivative).all())


def test_current_solver_rank_one_radius_active_solution_uses_both_nonnegative_axes() -> None:
    dtype = torch.float64
    G = torch.tensor([-1.0, 0.0, 0.0], dtype=dtype)
    D = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    V = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=dtype)

    result = solve_two_direction_subspace(G, D, V, radius=0.5)

    assert result.diagnostics["full_kkt_status"] == "rank_one_radius_active"
    torch.testing.assert_close(result.coefficients,
        torch.tensor([2**-1.5, 2**-1.5], dtype=dtype), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.diagnostics["movement_norm"],
        torch.tensor(0.5, dtype=dtype), atol=1e-12, rtol=1e-12)


def test_axis0_kkt_interval_ends_at_inactive_multiplier_zero() -> None:
    dtype = torch.float64
    M = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=dtype)
    Q = torch.tensor([[1.0, 0.4], [0.4, 1.0]], dtype=dtype)
    c = torch.tensor([-1.0, -0.3], dtype=dtype)

    at = audit.axis0_kkt(M, Q, c, radius=0.2)
    interval = audit.axis0_activation(M, Q, c)

    assert at["axis0_kkt"] is True
    assert at["lambda"] == pytest.approx(3.0)
    assert at["mu1"] == pytest.approx(0.04)
    assert interval["critical_axis_activation_radius"] == pytest.approx(1.0 / 3.0)
    assert interval["axis_lambda_zero_radius"] == pytest.approx(0.5)
    assert interval["lambda_at_mu1_zero"] > 0.0
    assert interval["inactive_axis_activates_before_axis_lambda_zero"] is True

    tiny_scale = 1e-200
    tiny_at = audit.axis0_kkt(M * tiny_scale, Q, c * tiny_scale, radius=0.2)
    tiny_interval = audit.axis0_activation(M * tiny_scale, Q, c * tiny_scale)
    assert tiny_at["axis0_kkt"] is True
    assert tiny_at["lambda"] / tiny_scale == pytest.approx(3.0)
    assert tiny_at["mu1"] / tiny_scale == pytest.approx(0.04)
    assert tiny_interval["critical_axis_activation_radius"] == pytest.approx(1.0 / 3.0)


def test_archived_d19e_kkt_sweep_is_pinned_and_reaches_mixed_axis_above_critical_radius() -> None:
    result = audit.build_audit()
    rows = result["radius_sweep"]
    near = min(rows, key=lambda row: abs(row["radius"] - result["axis0_activation"]["critical_axis_activation_radius"]))
    expanded = next(row for row in rows if row["radius_factor"] == pytest.approx(1.5))

    assert result["source_member_count"] == 216
    assert result["archive_pin_count"] == 287
    assert result["receipt_member_count"] == 11
    assert result["base_control_sha256"].startswith("d19e882d")
    assert result["V_rank"] == 2
    assert result["axis0_current_radius_KKT"]["axis0_kkt"] is True
    assert result["axis0_current_radius_KKT"]["mu1"] > 0.0
    assert result["axis0_activation"]["inactive_axis_activates_before_axis_lambda_zero"] is True
    assert result["axis0_activation"]["critical_axis_activation_radius"] > result["current_radius"]
    assert near["radius_factor"] == pytest.approx(
        result["axis0_activation"]["critical_axis_activation_radius"] / result["current_radius"],
        abs=0.02)
    assert expanded["second_coefficient_positive"] is True
    assert expanded["model_gain_vs_gradient_axis"] > 0.0
    assert result["limited_new_point_experiment_worthwhile"] is True
