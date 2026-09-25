"""Endpoint-measured sector-switch policy and additive Newton hook checks."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from advar import matrix_free
from advar import local_refinement as refinement
from advar.local_refinement import (
    RefinementCallbackError, RefinementTrial, refine_stationary,
)
from examples.weather_scenarios.fv_point_sector_policy import sector_trial_acceptance


def _trial(**changes):
    trial = RefinementTrial(
        iteration=2, backtrack=0,
        current_objective=0.8, candidate_objective=0.2,
        current_gradient_norm=2.0, candidate_gradient_norm=1.0,
        current_gradient_max=2.0, candidate_gradient_max=1.0,
        current_branch={"choices": [-1], "face_signs": [-1]},
        candidate_branch={"choices": [1], "face_signs": [1]},
        step_scale=1.0, normalized_slope=-1.0, armijo_ratio=1.1,
    )
    return replace(trial, **changes)


def test_switch_requires_actual_objective_merit_and_max_gradient_decrease():
    records = []
    assert sector_trial_acceptance(_trial(), records.append) == (
        True, "sector_switch_measured_decrease"
    )
    assert len(records) == 1 and records[0]["accepted"] is True
    assert all(value["passed"] for value in records[0]["comparisons"].values())
    for change, name in (
        ({"candidate_objective": 0.9}, "objective"),
        ({"candidate_gradient_norm": 3.0}, "gradient_merit"),
        ({"candidate_gradient_max": 3.0}, "gradient_max"),
    ):
        decision = sector_trial_acceptance(_trial(**change))
        assert decision is not None and decision[0] is False
        assert name in decision[1]


def test_switch_rejects_roundoff_scale_and_nonfinite_merit():
    tiny_improvement = 0.8 - 1e-15
    decision = sector_trial_acceptance(_trial(candidate_objective=tiny_improvement))
    assert decision is not None and decision[0] is False
    assert "objective" in decision[1]
    nonfinite = sector_trial_acceptance(_trial(current_gradient_norm=1e200))
    assert nonfinite is not None and nonfinite[0] is False
    assert "gradient_merit" in nonfinite[1]


def test_same_signature_delegates_to_legacy_armijo():
    trial = _trial(candidate_branch={"choices": [-1], "face_signs": [-1]})
    assert sector_trial_acceptance(trial) is None


def test_default_and_none_hook_have_identical_quadratic_result_and_history():
    p = torch.zeros(1, dtype=torch.float64)
    start = torch.tensor([0.4], dtype=torch.float64)

    def objective(control, _parameters):
        return 0.5 * (control[0] - 1).square()

    def branch(_control, _parameters):
        return {"choices": [1], "face_signs": [1]}, "smooth"

    legacy = refine_stationary(objective, start, p, branch_check=branch)
    observed = refine_stationary(objective, start, p, branch_check=branch,
                                trial_acceptance=lambda _trial: None)
    torch.testing.assert_close(legacy.control, observed.control, rtol=0, atol=0)
    assert legacy.gradient_max == observed.gradient_max
    assert legacy.iterations == observed.iterations
    assert legacy.history == observed.history


def test_accepted_switch_relinearizes_at_the_new_sector():
    p = torch.zeros(1, dtype=torch.float64)
    start = torch.tensor([-0.2], dtype=torch.float64)
    decisions = []
    hessian_values = []

    def objective(control, _parameters):
        x = control[0]
        return 0.5 * (x - 0.8).square() + 0.1 * torch.relu(-x).square()

    def branch(control, _parameters):
        if abs(float(control[0])) <= 1e-8:
            raise ValueError("unresolved sector")
        sector = -1 if float(control[0]) < 0 else 1
        return {"choices": [sector], "face_signs": [sector]}, "resolved sector"

    def monitor(original):
        def solve(operator, rhs, **kwargs):
            hessian_values.append(float(operator(torch.ones_like(rhs))[0]))
            return original(operator, rhs, **kwargs)
        return solve

    with matrix_free.observe_pcg_calls(monitor):
        result = refine_stationary(
            objective, start, p, branch_check=branch,
            trial_acceptance=lambda trial: sector_trial_acceptance(trial, decisions.append),
        )
    assert result.gradient_max < 1e-10
    torch.testing.assert_close(result.control, torch.tensor([0.8], dtype=start.dtype),
                               rtol=0, atol=1e-12)
    assert any(record["accepted"] for record in decisions)
    assert hessian_values[0] == pytest.approx(1.2)
    assert hessian_values[1] == pytest.approx(1.0)
    assert any(record.get("acceptance_policy") == "trial_acceptance"
               for record in result.history)


def test_invalid_hook_result_and_hook_exception_fail_closed():
    p = torch.zeros(1, dtype=torch.float64)
    start = torch.tensor([0.4], dtype=torch.float64)

    def objective(control, _parameters):
        return 0.5 * (control[0] - 1).square()

    def branch(_control, _parameters):
        return {"choices": [1], "face_signs": [1]}, "smooth"

    with pytest.raises(RefinementCallbackError, match="trial_acceptance must return"):
        refine_stationary(objective, start, p, branch_check=branch,
                          trial_acceptance=cast(Any, lambda _trial: (1, "bad")))

    def broken(_trial):
        raise RuntimeError("hook programmer error")

    with pytest.raises(RefinementCallbackError, match="hook programmer error"):
        refine_stationary(objective, start, p, branch_check=branch,
                          trial_acceptance=broken)

    def broken_observer(_record):
        raise ValueError("observer programmer error")

    with pytest.raises(RefinementCallbackError, match="observer programmer error"):
        refine_stationary(objective, start, p, branch_check=branch,
                          trial_observer=broken_observer)


def test_trial_observer_gets_a_copy_and_cannot_mutate_history():
    p = torch.zeros(1, dtype=torch.float64)
    start = torch.tensor([0.4], dtype=torch.float64)
    calls = []

    def objective(control, _parameters):
        return 0.5 * (control[0] - 1).square()

    def branch(_control, _parameters):
        return {"choices": [1], "face_signs": [1]}, "smooth"

    def observer(record):
        calls.append(record)
        record["candidate_control"][0] = 999.0
        record["branch"]["choices"][0] = 999

    result = refine_stationary(objective, start, p, branch_check=branch,
                               trial_observer=observer)
    assert calls and calls[0]["candidate_control"] == [999.0]
    assert result.history[0]["candidate_control"] != [999.0]
    assert result.history[0]["branch"]["choices"] == [1]


def test_switch_hook_uses_finite_endpoint_even_when_old_armijo_limit_is_zero(monkeypatch):
    monkeypatch.setattr(refinement, "_ARMIJO_CONSTANT", 1.0)
    p = torch.zeros(1, dtype=torch.float64)
    start = torch.tensor([-0.2], dtype=torch.float64)

    def objective(control, _parameters):
        x = control[0]
        return 0.5 * (x - 0.8).square() + 0.1 * torch.relu(-x).square()

    def branch(control, _parameters):
        sector = -1 if float(control[0]) < 0 else 1
        return {"choices": [sector], "face_signs": [sector]}, "strict point"

    switched_records = []
    with pytest.raises(RuntimeError, match="iteration budget"):
        refine_stationary(
            objective, start, p, branch_check=branch, max_iterations=1,
            trial_acceptance=sector_trial_acceptance,
            trial_observer=switched_records.append,
        )
    assert switched_records[0]["accepted"] is True
    assert switched_records[0]["armijo_ratio"] is None
    assert switched_records[0]["acceptance_policy"] == "trial_acceptance"

    legacy_records = []
    with pytest.raises(RuntimeError, match="Armijo step"):
        refine_stationary(
            objective, start, p, branch_check=branch, max_iterations=1,
            trial_observer=legacy_records.append,
        )
    assert legacy_records[0]["accepted"] is False
    assert legacy_records[0]["armijo_ratio"] == float("inf")
