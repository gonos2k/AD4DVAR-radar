from __future__ import annotations

import torch
import pytest
from typing import Any

from examples.weather_scenarios import fv_point_3h_dual_merit_step as step


def _inputs() -> tuple[Any, ...]:
    control = torch.zeros(26, dtype=torch.float64)
    direction = torch.zeros_like(control)
    direction[0] = -1.
    gradient = torch.zeros_like(control)
    gradient[0] = 1.
    hs = torch.zeros_like(control)
    hs[0] = -2.
    params = torch.zeros(13, dtype=torch.float64)
    base_j = torch.tensor(1., dtype=torch.float64)
    base_phi = torch.tensor(.5, dtype=torch.float64)

    def objective(candidate, _parameters):
        return 1. + candidate[0]

    def gradient_fn(candidate, _parameters):
        alpha = -candidate[0]
        result = torch.zeros_like(candidate)
        result[0] = 1. - 2. * alpha + 100. * alpha.square()
        return result

    def branch_fn(_candidate, _parameters):
        return ({"status": "passed_strict_branch", "euler_stages": 3600,
                 "choice_stage_count": 3600, "face_sign_stage_count": 3600,
                 "signature_sha256": "strict"}, {"complete": True})

    return control, direction, base_j, gradient, base_phi, hs, objective, gradient_fn, params, branch_fn


def test_dual_merit_rejects_j_only_decrease_then_accepts_same_direction(monkeypatch):
    monkeypatch.setattr(step.merit.seed_linear, "_valid_margins", lambda *_args, **_kwargs: True)
    args = _inputs()
    rows = []
    candidate, trials, status = step.dual_merit_search(
        *args, deadline=float("inf"), base_branch_signature="strict", on_trial=rows.append,
    )

    assert status == "one_dual_merit_step_accepted"
    assert len(trials) == 3
    assert trials[0]["status"] == "rejected"
    assert trials[0]["J_armijo_passed"] is True
    assert trials[0]["Phi_armijo_passed"] is False
    assert "actual Phi exceeds Phi Armijo threshold" in trials[0]["reasons"]
    assert trials[-1]["status"] == "accepted"
    assert trials[-1]["J_armijo_passed"] is True
    assert trials[-1]["Phi_armijo_passed"] is True
    assert trials[-1]["gradient_blocks"]["full_l2"] == trials[-1]["gradient_l2"]
    assert candidate is not None
    assert torch.equal(candidate, args[0] + trials[-1]["alpha"] * args[1])
    assert rows == trials
    assert all(row["new_hvp"] == 0 for row in trials)


def test_dual_merit_requires_candidate_branch_even_when_both_armijo_tests_pass(monkeypatch):
    monkeypatch.setattr(step.merit.seed_linear, "_valid_margins", lambda *_args, **_kwargs: True)
    args: list[Any] = list(_inputs())
    # A linear gradient makes Phi decrease from the first bounded trial onward.
    args[7] = lambda candidate, _parameters: torch.cat(
        ((1. + 2. * candidate[0]).reshape(1), torch.zeros(25, dtype=torch.float64)))

    def branch_fn(candidate, _parameters):
        passed = -float(candidate[0]) <= .03
        return ({"status": "passed_strict_branch" if passed else "branch_refused",
                 "euler_stages": 3600, "choice_stage_count": 3600,
                 "face_sign_stage_count": 3600, "signature_sha256": "strict"}, {"complete": passed})

    args[9] = branch_fn
    candidate, trials, status = step.dual_merit_search(
        *args, deadline=float("inf"), base_branch_signature="strict",
    )

    assert status == "one_dual_merit_step_accepted"
    assert trials[0]["J_armijo_passed"] is True
    assert trials[0]["Phi_armijo_passed"] is True
    assert trials[0]["strict_point_passed"] is False
    assert "candidate full strict branch or margins failed" in trials[0]["reasons"]
    assert trials[1]["strict_point_passed"] is True
    assert candidate is not None


def test_post_acceptance_failure_clears_provisional_commit():
    record: dict[str, Any] = {"optimizer_steps_applied": 0, "trials": [
        {"status": "accepted", "accepted": True, "alpha": .01}]}

    step._mark_candidate_not_committed(record, "physical diagnostic failed")

    assert record["trials"][-1] == {"status": "candidate_not_committed", "accepted": False,
                                    "alpha": .01, "commit_refusal": "physical diagnostic failed"}


@pytest.mark.parametrize("phase,index", [("objective", 6), ("gradient", 7), ("branch", 9)])
def test_callback_error_preserves_current_candidate(phase, index):
    args: list[Any] = list(_inputs())
    rows = []

    def broken(*args):
        raise RuntimeError("unexpected callback failure")

    args[index] = broken
    with pytest.raises(RuntimeError, match="unexpected callback failure"):
        step.dual_merit_search(*args, deadline=float("inf"), base_branch_signature="strict", on_trial=rows.append)
    assert len(rows) == 1
    assert rows[0]["status"] == "evaluation_error"
    assert rows[0]["failed_phase"] == phase


def test_malformed_branch_contract_keeps_candidate_receipt():
    args: list[Any] = list(_inputs())
    args[9] = lambda *args: (None, {})
    rows = []
    with pytest.raises(AttributeError):
        step.dual_merit_search(*args, deadline=float("inf"), base_branch_signature="strict", on_trial=rows.append)
    assert rows[0]["status"] == "evaluation_error"
    assert rows[0]["failed_phase"] == "branch"
