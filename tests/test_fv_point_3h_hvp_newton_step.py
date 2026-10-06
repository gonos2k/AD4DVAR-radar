"""Focused live-operator tests; no FV trajectory is executed."""
import pytest
import torch

from examples.weather_scenarios import fv_point_3h_hvp_newton_step as probe


def test_newton_pcg_uses_live_operator_not_frozen_preconditioner():
    diagonal = torch.linspace(1.5, 3.5, 26, dtype=torch.float64)
    gradient = torch.linspace(-0.8, 0.9, 26, dtype=torch.float64)
    live = lambda value: diagonal * value
    frozen_preconditioner = lambda value: value / 2.0
    step, audit = probe.block_step.solve_newton_direction(live, gradient, frozen_preconditioner)
    torch.testing.assert_close(live(step), -gradient, rtol=5e-10, atol=1e-12)
    assert audit["true_relative_residual"] <= 1e-10
    assert not torch.allclose(frozen_preconditioner(gradient), live(gradient))


def test_negative_live_curvature_refuses_before_search_candidate():
    gradient = torch.ones(26, dtype=torch.float64)
    candidate_calls = []
    with pytest.raises(probe.block_step.StepRefusal, match="operator must be symmetric positive definite"):
        step, _ = probe.block_step.solve_newton_direction(
            lambda value: -value, gradient, lambda value: value)
        candidate_calls.append(step)
    assert candidate_calls == []


def test_live_hvp_cap_refuses_without_candidate():
    gradient = torch.ones(26, dtype=torch.float64)
    deadline = torch.inf
    counted, counts = probe.counted_hvp(lambda value: value, deadline, limit=1)
    candidate_calls = []
    with pytest.raises(probe.search.BudgetRefusal, match="live HVP cap 1 reached"):
        step, _ = probe.block_step.solve_newton_direction(
            counted, gradient, lambda value: value)
        candidate_calls.append(step)
    assert counts == {"started": 1, "completed": 1}
    assert candidate_calls == []
