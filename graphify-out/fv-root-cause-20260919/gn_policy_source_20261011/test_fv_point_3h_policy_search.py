from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_policy_search as searcher


GATES = (
    "mixing_minimum_valid", "J_armijo_passed", "F_squared_armijo_passed",
    "branch_pair_passed", "face_audit_passed", "side_objectives_match_native",
    "side_gradients_finite", "finite",
)


def _model(*, merit=True):
    return SimpleNamespace(direction=torch.tensor([1., 0.], dtype=torch.float64), theta=.5,
        gates={"finite": True, "both_side_gradients_descend": True, "merit_descends": merit},
        merit_product=-.01 if merit else .01,
        residual_direction=torch.tensor([1., 0.], dtype=torch.float64),
        side_products=(-.02, -.01))


def _facts(*, J=True, R=True, value=1., rvalue=1.):
    return {**{key: True for key in GATES}, "J_armijo_passed": J,
        "F_squared_armijo_passed": R, "objective": value, "F_squared": rvalue}


def _run(policy, evaluator, *, merit=True, previous=0.01, reference_R=1.0, chart=None):
    progress = []
    result = searcher.search(_model(merit=merit), torch.zeros(2, dtype=torch.float64), evaluator,
        chart_candidate=chart or (lambda current, direction, alpha: current + alpha * direction),
        progress=lambda trials, meta: progress.append((json.dumps(trials, allow_nan=False), dict(meta))),
        policy=policy, previous_movement=previous, reference_R=reference_R)
    return result, progress


@pytest.mark.parametrize("bad", [[float("inf"), 0.], [float("nan"), 0.], [1e308, 1e308]])
def test_bad_chart_receipts_are_json_safe_and_search_continues(bad):
    calls = []
    def chart(current, direction, alpha):
        calls.append(alpha)
        return torch.tensor(bad, dtype=current.dtype) if len(calls) == 1 else current + alpha * direction
    result, _ = _run({"name": "dyadic_JR", "max_evaluations": 2, "max_backtracks": 2},
        lambda *_: _facts(), chart=chart)
    chosen, trials, meta = result
    assert chosen is not None and chosen["index"] == 2
    assert trials[0]["status"] == "nonfinite_chart_movement_refused"
    assert trials[0]["actual_path_norm"] is None
    assert meta["slots_started"] == meta["slots_completed"] == 2
    assert meta["evaluations_started"] == meta["evaluations_completed"] == 1


def test_seeded_step_reduces_candidate_cost_and_keeps_true_R_semantics():
    seeded_calls, dyadic_calls = [], []
    def evaluate(calls):
        def run(_point, _theta, alpha):
            calls.append(alpha)
            return _facts(R=alpha <= .0045, value=2., rvalue=3.)
        return run
    result, _ = _run({"name": "seeded_JR", "gamma": 4., "max_evaluations": 8},
        evaluate(seeded_calls), previous=.001)
    chosen, trials, meta = result
    assert chosen is not None and chosen["F_squared_armijo_passed"] is True
    assert meta["initial_alpha"] < meta["computed_model_alpha"]
    baseline, _ = _run({"name": "dyadic_JR", "max_evaluations": 8},
        evaluate(dyadic_calls), previous=.001)
    assert baseline[0] is not None
    assert len(seeded_calls) == 2 and len(dyadic_calls) == 3
    assert len(trials) == 2
    assert meta["status"] == "eligible" and chosen["policy_eligible"] is True


def test_seed_expansion_stops_at_first_refusal_and_never_exceeds_model_alpha():
    alphas = []
    def evaluate(_point, _theta, alpha):
        alphas.append(alpha)
        return _facts(R=len(alphas) < 3, value=3. - alpha, rvalue=2.)
    result, _ = _run({"name": "seeded_JR", "max_expansions": 3,
                      "max_evaluations": 8}, evaluate, previous=.001)
    chosen, trials, meta = result
    assert chosen is not None and len(trials) == 3
    assert all(alpha <= meta["computed_model_alpha"] for alpha in alphas)
    assert trials[2]["status"] == "gate_refused"
    assert meta["selected_index"] in (1, 2)
    assert meta["first_pass_candidate_index"] == 1
    assert meta["first_pass_count"] == 1


def test_bounded_cap_is_fixed_and_does_not_claim_F_squared_armijo():
    def evaluate(*_args):
        return _facts(R=False, value=.5, rvalue=1.005)
    result, _ = _run({"name": "bounded_J_Rcap", "Rcap_fraction": .01,
                      "max_evaluations": 4}, evaluate, merit=False,
        previous=.02, reference_R=1.0)
    chosen, trials, meta = result
    assert chosen is not None and chosen["R_cap_passed"] is True
    assert chosen["F_squared_armijo_passed"] is False
    assert chosen["R_cap"] == pytest.approx(1.01)
    assert meta["reference_R"] == 1.0


def test_R_cap_refuses_candidate_outside_same_reference_horizon():
    result, _ = _run({"name": "bounded_J_Rcap", "Rcap_fraction": .01,
                      "max_evaluations": 2},
        lambda *_: _facts(R=False, rvalue=1.02), merit=False,
        previous=.02, reference_R=1.0)
    chosen, trials, _ = result
    assert chosen is None
    assert trials[0]["R_cap_passed"] is False and trials[0]["accepted"] is False


def test_R_cap_is_required_even_when_R_armijo_passes():
    result, _ = _run({"name": "bounded_J_Rcap", "Rcap_fraction": .01,
                      "max_evaluations": 2},
        lambda *_: _facts(R=True, rvalue=1.02), merit=False,
        previous=.02, reference_R=1.0)
    chosen, trials, _ = result
    assert chosen is None
    assert trials[0]["F_squared_armijo_passed"] is True
    assert trials[0]["R_cap_passed"] is False


def test_bounded_policy_uses_gamma_seed_and_expands_to_fallback_proposal_cap():
    alphas = []
    result, _ = _run({"name": "bounded_J_Rcap", "gamma": 4., "max_expansions": 3,
                      "max_evaluations": 8},
        lambda *_args: (alphas.append(_args[-1]) or _facts(R=False, value=1., rvalue=1.)),
        merit=False, previous=.001, reference_R=1.0)
    chosen, trials, meta = result
    assert chosen is not None
    assert meta["initial_alpha"] == pytest.approx(4. * .001)
    assert alphas[:3] == pytest.approx([.004, .008, .016])
    assert max(alphas) <= .05


@pytest.mark.parametrize("budget", ["max_evaluations", "max_backtracks"])
def test_zero_search_budget_returns_without_starting_a_slot(budget):
    called = []
    result, _ = _run({"name": "seeded_JR", budget: 0},
        lambda *_: (called.append(True) or _facts()), previous=.001)
    chosen, trials, meta = result
    assert chosen is None and trials == [] and called == []
    assert meta["slots_started"] == meta["slots_completed"] == 0
    assert meta["evaluations_started"] == meta["evaluations_completed"] == 0


def test_failed_gate_is_never_returned_as_eligible():
    result, _ = _run({"name": "seeded_JR", "max_evaluations": 3},
        lambda *_: _facts(J=False), previous=.02)
    chosen, trials, meta = result
    assert chosen is None and not any(t["accepted"] for t in trials)
    assert meta["status"] == "gate_refused"


def test_timeout_receipt_preserves_started_completed_counts_and_reraises():
    progress = []
    def evaluate(*_args):
        raise TimeoutError("budget expired")
    with pytest.raises(TimeoutError, match="budget expired"):
        searcher.search(_model(), torch.zeros(2, dtype=torch.float64), evaluate,
            chart_candidate=lambda current, direction, alpha: current + alpha * direction,
            progress=lambda trials, meta: progress.append((trials, dict(meta))),
            policy={"name": "dyadic_JR", "max_evaluations": 2})
    trial, meta = progress[-1]
    assert trial[0]["status"] == "slot_interrupted"
    assert meta["slots_started"] == 1 and meta["slots_completed"] == 0
    assert meta["evaluations_started"] == 1 and meta["evaluations_completed"] == 0


def test_dyadic_baseline_stops_at_first_pass():
    calls = []
    result, _ = _run({"name": "dyadic_JR", "max_evaluations": 5},
        lambda *_: (calls.append(1) or _facts()))
    chosen, trials, _ = result
    assert chosen is not None and len(calls) == len(trials) == 1
