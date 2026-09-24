"""Analytic checks for the conditional matrix-free stationarity refiner."""
from __future__ import annotations

import math

import pytest
import torch

from advar import local_refinement as module
from advar.matrix_free import PCGResult


def _case(size: int = 3, *, initial: float = 0.0, coupling: float = 0.0):
    dtype = torch.float64
    diagonal = torch.linspace(1.5, 3.0, size, dtype=dtype)
    target = torch.linspace(-0.3, 0.4, size, dtype=dtype)
    p = torch.tensor([0.2, -0.1], dtype=dtype)
    control = torch.full((size,), initial, dtype=dtype)

    def objective(c, q):
        shift = target + 0.1 * q[0]
        delta = c - shift
        coupling_cost = 0.5 * coupling * (delta[1:] - delta[:-1]).square().sum()
        return (
            0.5 * (diagonal * delta.square()).sum()
            + 0.01 * delta.pow(4).sum()
            + coupling_cost
        )

    def branch_check(c, q):
        return "analytic", "fixed analytic branch"

    stationary = target + 0.1 * p[0]
    return objective, control, p, stationary, branch_check


@pytest.mark.parametrize("initial", [0.0, 0.5, -0.5])
def test_nonlinear_spd_refinement_matches_known_stationary_point_without_dense_solve(
    monkeypatch, initial
):
    objective, control, p, stationary, branch_check = _case(initial=initial)
    original_control = control.clone()
    original_p = p.clone()
    monkeypatch.setattr(torch.linalg, "solve", lambda *args, **kwargs: pytest.fail("dense solve used"))
    result = module.refine_stationary(objective, control, p, branch_check=branch_check)
    torch.testing.assert_close(result.control, stationary, rtol=1e-10, atol=1e-10)
    assert result.gradient_max < 1e-10
    assert result.iterations > 0
    assert result.hvp_count > 0
    assert any(record["accepted"] for record in result.history)
    torch.testing.assert_close(control, original_control)
    torch.testing.assert_close(p, original_p)


@pytest.mark.parametrize("coupling", [0.0, 4.0])
@pytest.mark.parametrize("initial", [0.0, 0.5, -0.5])
def test_sixty_four_control_case_agrees_with_analytic_reference(coupling, initial):
    objective, control, p, stationary, branch_check = _case(
        64, initial=initial, coupling=coupling
    )
    result = module.refine_stationary(objective, control, p, branch_check=branch_check)
    torch.testing.assert_close(result.control, stationary, rtol=1e-10, atol=1e-10)
    assert result.gradient_max < 1e-10
    assert result.hvp_count < 4 * control.numel() * result.iterations + result.iterations


def test_initial_stationary_control_returns_without_claiming_minimum():
    objective, _, p, stationary, branch_check = _case()
    result = module.refine_stationary(objective, stationary, p, branch_check=branch_check)
    assert result.iterations == 0
    assert result.hvp_count == 0
    assert result.history == []
    assert result.gradient_max < 1e-10


def test_gradient_at_exact_tolerance_is_refined_instead_of_accepted():
    dtype = torch.float64
    control = torch.zeros(2, dtype=dtype)
    p = torch.zeros(1, dtype=dtype)

    def objective(c, q):
        return 0.5 * c.square().sum() + 1.0e-10 * c.sum()

    result = module.refine_stationary(
        objective,
        control,
        p,
        branch_check=lambda c, q: ("quadratic", "fixed"),
    )
    assert result.iterations > 0
    assert result.gradient_max < 1e-10


def test_normalized_armijo_rejects_overshoot_before_accepting_backtrack():
    dtype = torch.float64
    control = torch.tensor([-2.0], dtype=dtype)
    p = torch.zeros(1, dtype=dtype)

    def objective(c, q):
        return torch.exp(c[0]) - c[0]

    result = module.refine_stationary(
        objective,
        control,
        p,
        branch_check=lambda c, q: ("exp", "fixed"),
    )
    assert result.gradient_max < 1e-10
    assert any(not record["accepted"] for record in result.history)
    assert any(
        record["accepted"] and record["step_scale"] < 1.0
        for record in result.history
    )
    first_rejection = next(record for record in result.history if not record["accepted"])
    assert first_rejection["norm_ratio"] > first_rejection["armijo_limit"]


@pytest.mark.parametrize("initial", [-7.0, -8.0])
def test_nonfinite_exponential_trials_are_rejected_until_a_finite_step(initial):
    dtype = torch.float64
    control = torch.tensor([initial], dtype=dtype)
    p = torch.zeros(1, dtype=dtype)

    def objective(c, q):
        return torch.exp(c[0]) - c[0]

    result = module.refine_stationary(
        objective, control, p,
        branch_check=lambda c, q: ("exp", "fixed"),
    )

    assert result.gradient_max < 1e-10
    assert any(record.get("rejection") == "nonfinite_candidate" for record in result.history)
    assert any(record["accepted"] for record in result.history)
    assert all(
        all(not isinstance(value, float) or math.isfinite(value)
            for value in record.values())
        for record in result.history
    )


@pytest.mark.parametrize(
    ("gradient", "message"),
    [
        (torch.tensor([float("inf"), 1.0], dtype=torch.float64), "gradient must be finite"),
        (torch.tensor([1.0e308, 1.0e308], dtype=torch.float64), "gradient norm must be finite"),
    ],
)
def test_nonfinite_candidate_gradient_is_rejected_as_a_candidate(gradient, message):
    finite_objective = torch.tensor(0.0, dtype=torch.float64)

    with pytest.raises(module._NonFiniteEvaluation, match=message):
        module._evaluate(
            lambda c, q: finite_objective,
            lambda c, q: gradient,
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )


def test_all_nonfinite_candidate_trials_end_with_a_finite_budget_refusal():
    dtype = torch.float64
    control = torch.zeros(1, dtype=dtype)
    p = torch.zeros(1, dtype=dtype)

    def objective(c, q):
        finite = 0.5 * c.square().sum() + c.sum()
        return torch.where(c[0] == 0.0, finite, c.new_tensor(float("inf")))

    with pytest.raises(RuntimeError, match="no finite candidate evaluation") as refusal:
        module.refine_stationary(
            objective, control, p,
            branch_check=lambda c, q: ("fixed", "fixed"),
            max_backtracks=3,
        )
    assert "nonfinite_candidate_rejections=3" in str(refusal.value)
    assert "branch_rejections=0" in str(refusal.value)


@pytest.mark.parametrize("error", [ValueError, RuntimeError, TypeError])
def test_candidate_objective_callback_errors_propagate(error):
    objective, control, p, _, branch_check = _case()
    candidate_seen = False

    def broken_objective(c, q):
        if candidate_seen:
            raise error("callback failure")
        return objective(c, q)

    def identify_candidate(c, q):
        nonlocal candidate_seen
        candidate_seen = not torch.equal(c, control)
        return branch_check(c, q)

    with pytest.raises(error, match="callback failure"):
        module.refine_stationary(
            broken_objective, control, p, branch_check=identify_candidate
        )
    assert candidate_seen


def test_malformed_candidate_objective_output_propagates():
    objective, control, p, _, branch_check = _case()
    candidate_seen = False

    def malformed_objective(c, q):
        if candidate_seen:
            return torch.stack((c.sum(), c.sum()))
        return objective(c, q)

    def identify_candidate(c, q):
        nonlocal candidate_seen
        candidate_seen = not torch.equal(c, control)
        return branch_check(c, q)

    with pytest.raises(ValueError, match="objective must return a scalar"):
        module.refine_stationary(
            malformed_objective, control, p, branch_check=identify_candidate
        )
    assert candidate_seen


def test_successful_pcg_flag_with_wrong_solution_is_rejected(monkeypatch):
    objective, control, p, _, branch_check = _case()
    monkeypatch.setattr(
        module,
        "pcg",
        lambda *args, **kwargs: PCGResult(
            solution=torch.zeros_like(control),
            converged=True,
            iterations=1,
            relative_residual=0.0,
        ),
    )
    with pytest.raises(RuntimeError, match="true residual"):
        module.refine_stationary(objective, control, p, branch_check=branch_check)


def test_negative_curvature_is_rejected_by_pcg():
    dtype = torch.float64
    control = torch.ones(2, dtype=dtype)
    p = torch.zeros(1, dtype=dtype)

    def objective(c, q):
        return -0.5 * c.square().sum()

    with pytest.raises(RuntimeError, match="positive definite"):
        module.refine_stationary(
            objective, control, p,
            branch_check=lambda c, q: ("branch", "fixed"),
        )


def test_branch_value_error_rejects_trials_and_reports_no_convergence():
    objective, control, p, _, _ = _case()
    calls = 0

    def branch_check(c, q):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("candidate crossed branch")
        return "initial", "fixed initial branch"

    with pytest.raises(RuntimeError, match="Armijo") as refusal:
        module.refine_stationary(
            objective, control, p, branch_check=branch_check, max_backtracks=3
        )
    assert calls == 4
    assert "branch_rejections=3" in str(refusal.value)
    assert "nonfinite_candidate_rejections=0" in str(refusal.value)


def test_iteration_budget_reports_no_convergence():
    objective, control, p, _, branch_check = _case()
    with pytest.raises(RuntimeError, match="iteration budget"):
        module.refine_stationary(
            objective, control, p, branch_check=branch_check, max_iterations=1
        )


def test_nonfinite_inputs_and_objective_are_rejected():
    objective, control, p, _, branch_check = _case()
    with pytest.raises(ValueError, match="finite"):
        module.refine_stationary(
            objective, control, torch.tensor([float("nan"), 0.0], dtype=torch.float64),
            branch_check=branch_check,
        )
    bad = lambda c, q: c.new_tensor(float("inf"))
    with pytest.raises(ValueError, match="objective"):
        module.refine_stationary(bad, control, p, branch_check=branch_check)
