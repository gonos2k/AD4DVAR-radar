from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_qy32_diagnostic as qy
from examples.weather_scenarios import fv_point_3h_qy32_tangent_step as tangent


def _base() -> torch.Tensor:
    value = torch.zeros(26, dtype=torch.float64)
    value[21:25] = torch.tensor((0.15, -0.12, 0.08, 0.03), dtype=torch.float64)
    return value


def test_common_tangent_direction_solves_chart_metric_system():
    z = torch.zeros((26, 25), dtype=torch.float64)
    z[:25] = torch.diag(torch.linspace(0.7, 1.5, 25, dtype=torch.float64))
    q_t = torch.cos(torch.arange(25, dtype=torch.float64) * 0.21)
    dt, direction = tangent.tangent_direction(z, q_t)
    assert torch.allclose(z.T @ direction, -q_t, atol=2e-14, rtol=2e-14)
    assert torch.allclose(dt, torch.linalg.solve(z.T @ z, -q_t), atol=2e-14, rtol=2e-14)


def test_fixed_eta_chart_step_preserves_face_and_exposes_curved_displacement():
    base = _base()
    eta = float(qy.qy32(base))
    dt = torch.zeros(25, dtype=torch.float64)
    dt[21:24] = torch.tensor((0.4, -0.3, 0.2), dtype=torch.float64)
    z, _ = qy.chart_jacobian(base, eta)
    alpha = 0.2
    point = tangent.chart_candidate(base, eta, dt, alpha)
    assert float(qy.qy32(point)) == pytest.approx(eta, abs=2e-16)
    actual = float(torch.linalg.vector_norm(point - base))
    linearized = float(torch.linalg.vector_norm(alpha * (z @ dt)))
    assert abs(actual - linearized) > 1e-8


def test_chart_radius_start_uses_actual_nonlinear_control_norm():
    base = _base()
    eta = float(qy.qy32(base))
    dt = torch.zeros(25, dtype=torch.float64)
    dt[21:24] = torch.tensor((0.9, -0.5, 0.4), dtype=torch.float64)
    z, _ = qy.chart_jacobian(base, eta)
    alpha = tangent.bounded_start_alpha(base, eta, dt,
                                        float(torch.linalg.vector_norm(z @ dt)))
    point = tangent.chart_candidate(base, eta, dt, alpha)
    assert alpha > 0
    assert float(torch.linalg.vector_norm(point - base)) <= tangent.RADIUS


def test_static_domain_refusal_shrinks_before_any_objective_evaluation():
    base = torch.zeros(26, dtype=torch.float64)
    base[24] = torch.atanh(torch.tensor(0.999999, dtype=torch.float64))
    eta = float(qy.qy32(base))
    dt = torch.zeros(25, dtype=torch.float64)
    dt[21] = -1.0
    # The nonlinear boundary lies inside a nominal retained-coordinate move.
    with pytest.raises(ValueError, match="chart domain"):
        tangent.chart_candidate(base, eta, dt, 0.05)
    alpha = tangent.bounded_start_alpha(base, eta, dt, direction_norm=1.0)
    point = tangent.chart_candidate(base, eta, dt, alpha)
    assert 0 < alpha < 0.05
    assert torch.isfinite(point).all()
    assert float(torch.linalg.vector_norm(point - base)) <= tangent.RADIUS
    assert float(qy.qy32(point)) == pytest.approx(eta, abs=2e-16)


def test_direction_requires_resolved_descent_in_original_j_and_phi():
    g = torch.tensor((1.0, 0.5), dtype=torch.float64)
    d = torch.tensor((-1.0, 0.0), dtype=torch.float64)
    hd = torch.tensor((-1.0, -2.0), dtype=torch.float64)
    result = tangent.resolved_direction_checks(g, d, hd)
    assert result["g_dot_d"] == -1.0
    assert result["g_dot_Hd"] == -2.0
    with pytest.raises(tangent.guard_policy.StepRefusal, match="resolved descent"):
        tangent.resolved_direction_checks(g, torch.zeros_like(g), hd)


def test_phi_armijo_rejects_a_real_phi_increase_even_if_j_decreases():
    result = tangent.dual_armijo(1.0, 0.5, 0.9, 0.50001,
                                 0.1, -1.0, -0.2)
    assert result["J_armijo_passed"] is True
    assert result["Phi_armijo_passed"] is False
    assert result["accepted"] is False


def test_single_hvp_cap_and_failed_product_counts():
    point = torch.zeros(26, dtype=torch.float64)
    params = torch.zeros(13, dtype=torch.float64)
    direction = torch.ones_like(point)
    matrix = torch.eye(26, dtype=torch.float64)
    apply, calls = tangent.counted_hvp(
        lambda x, _p: matrix @ x, point, params, direction, time.monotonic() + 10)
    assert torch.equal(apply(), direction)
    assert calls == {"started": 1, "completed": 1}
    with pytest.raises(ValueError, match="cap"):
        apply()
    assert calls == {"started": 1, "completed": 1}

    def broken(_x, _p):
        raise RuntimeError("live HVP failed")

    failed_apply, failed_calls = tangent.counted_hvp(
        broken, point, params, direction, time.monotonic() + 10)
    with pytest.raises(RuntimeError, match="live HVP failed"):
        failed_apply()
    assert failed_calls == {"started": 1, "completed": 0}


def test_final_recheck_failure_marks_tentative_candidate_uncommitted():
    record: dict[str, Any] = {"optimizer_steps_applied": 0, "candidate_committed": False,
                             "trials": [{"status": "accepted", "accepted": True, "alpha": 0.01}]}
    tangent._mark_not_committed(record, "final endpoint recheck failed")
    assert record["optimizer_steps_applied"] == 0
    assert record["candidate_committed"] is False
    assert record["trials"][-1] == {"status": "candidate_not_committed", "accepted": False,
                                    "alpha": 0.01,
                                    "commit_refusal": "final endpoint recheck failed"}


def test_resolved_direction_refusal_is_completed_numerical_status_without_candidate():
    record: dict[str, Any] = {"trials": [], "optimizer_steps_applied": 0,
                             "candidate_committed": False}
    tangent._record_direction_refusal(record, "resolved descent failed", closed=True)
    assert record["phase"] == "finished"
    assert record["execution_status"] == "completed"
    assert record["numerical_status"] == "direction_refusal"
    assert record["optimizer_steps_applied"] == 0
    assert record["candidate_committed"] is False
    assert record["trials"] == []
