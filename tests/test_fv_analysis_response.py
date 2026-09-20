"""Tests for the bounded FV response orchestration wrapper."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fv_analysis_response", ROOT / "examples/weather_scenarios/fv_analysis_response.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.fixture()
def analytic_case():
    dtype = torch.float64
    A = torch.diag(torch.tensor([2.0, 3.0], dtype=dtype))
    B = torch.tensor([[0.5], [-0.25]], dtype=dtype)
    p = torch.tensor([0.8], dtype=dtype)
    stationary = torch.linalg.solve(A, B @ p)
    directions = {"p": torch.ones_like(p)}
    state = {"branch": "stable"}

    def objective(c, q):
        return 0.5 * c @ A @ c - c @ (B @ q)

    def score(c, q):
        residual = c.square().sum() + q.square().sum()
        return 0.5 * residual

    def branch_check(c, q):
        if state["branch"] != "stable":
            raise ValueError("branch changed")
        return state["branch"], "analytic fixed branch"

    identity = {"fixture": "analysis-response-test"}
    return locals()


def _call(case, control, *, refine=None):
    return MODULE.prepare_response(
        case["objective"], case["score"], control, case["p"], case["directions"],
        branch_check=case["branch_check"], input_identity=case["identity"], refine=refine,
    )


def test_eligible_control_skips_refinement_and_returns_response(analytic_case):
    case = analytic_case

    def unexpected(_, __):
        pytest.fail("refinement should not run for an eligible control")

    result = _call(case, case["stationary"], refine=unexpected)
    assert result["status"] == "eligible"
    assert result["response"] is not None
    assert result["refinement_used"] is False
    torch.testing.assert_close(result["control"], case["stationary"])
    assert result["before"]["gradient_max"] < 1e-10
    assert result["after"]["branch"] == "stable"


def test_nonstationary_without_refinement_is_ineligible_and_preserves_control(analytic_case):
    case = analytic_case
    original = case["stationary"] + torch.tensor([0.1, -0.1], dtype=torch.float64)
    result = _call(case, original)
    assert result["status"] == "ineligible"
    assert result["response"] is None
    assert result["refinement_used"] is False
    torch.testing.assert_close(result["control"], original)
    assert "not stationary" in result["error"]


@pytest.mark.parametrize("failure", ["branch", "nonfinite"])
def test_initial_assessment_failure_never_invokes_refinement(analytic_case, failure):
    case = analytic_case
    calls = []
    if failure == "branch":
        def branch_check(_, __):
            raise ValueError("initial branch refusal")
        objective = case["objective"]
    else:
        branch_check = case["branch_check"]

        def objective(_, q):
            return q.new_tensor(float("inf"))

    def refine(_, __):
        calls.append(True)
        return case["stationary"]

    result = MODULE.prepare_response(
        objective, case["score"], case["stationary"], case["p"], case["directions"],
        branch_check=branch_check, input_identity=case["identity"], refine=refine,
    )
    assert result["status"] == "ineligible"
    assert result["response"] is None
    assert calls == []
    assert result["refinement_used"] is False


def test_response_failure_never_returns_partial_response(analytic_case):
    case = analytic_case

    def bad_score(c, q):
        return c.new_tensor(float("inf"))

    result = MODULE.prepare_response(
        case["objective"], bad_score, case["stationary"], case["p"], case["directions"],
        branch_check=case["branch_check"], input_identity=case["identity"],
    )
    assert result["status"] == "ineligible"
    assert result["response"] is None
    assert result["control"] is not case["stationary"]
    torch.testing.assert_close(result["control"], case["stationary"])
    assert "local response failed" in result["error"]


def test_non_tensor_refiner_result_is_a_bounded_refusal(analytic_case):
    case = analytic_case
    original = case["stationary"] + torch.tensor([0.1, -0.1], dtype=torch.float64)

    def refine(_, __):
        return None

    result = _call(case, original, refine=refine)
    assert result["status"] == "ineligible"
    assert result["response"] is None
    torch.testing.assert_close(result["control"], original)
    assert "tensor control" in result["error"]


@pytest.mark.parametrize("mode", ["raises", "bad_control", "branch_change"])
def test_failed_refinement_or_branch_change_returns_no_response(analytic_case, mode):
    case = analytic_case
    original = case["stationary"] + torch.tensor([0.1, -0.1], dtype=torch.float64)

    if mode == "raises":
        def refine(_, __):
            raise ValueError("root refusal")
    elif mode == "bad_control":
        def refine(_, __):
            return original
    else:
        def refine(_, __):
            case["state"]["branch"] = "changed"
            return case["stationary"]

    result = _call(case, original, refine=refine)
    assert result["status"] == "ineligible"
    assert result["response"] is None
    assert result["refinement_used"] is True
    torch.testing.assert_close(result["control"], original)
    if mode == "branch_change":
        assert "branch changed" in result["error"]


def test_explicit_refinement_enables_actual_response(analytic_case):
    case = analytic_case
    original = case["stationary"] + torch.tensor([0.1, -0.1], dtype=torch.float64)
    parameters_before = case["p"].clone()
    calls = []

    def refine(control, parameters):
        calls.append(control.clone())
        torch.testing.assert_close(parameters, case["p"])
        return case["stationary"]

    result = _call(case, original, refine=refine)
    assert result["status"] == "eligible"
    assert result["response"] is not None
    assert result["refinement_used"] is True
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], original)
    torch.testing.assert_close(result["control"], case["stationary"])
    torch.testing.assert_close(case["p"], parameters_before)
    assert result["after"]["gradient_max"] < 1e-10
    assert result["response"].total_gradient.shape == case["p"].shape


@pytest.mark.parametrize("raises", [False, True])
def test_refiner_parameter_mutation_is_rejected_and_caller_parameter_is_preserved(analytic_case, raises):
    case = analytic_case
    original = case["stationary"] + torch.tensor([0.1, -0.1], dtype=torch.float64)
    parameters_before = case["p"].clone()

    def refine(_, parameters):
        parameters.add_(1.0)
        if raises:
            raise RuntimeError("refiner failed after mutation")
        return case["stationary"]

    result = _call(case, original, refine=refine)
    assert result["status"] == "ineligible"
    assert result["response"] is None
    torch.testing.assert_close(result["control"], original)
    torch.testing.assert_close(case["p"], parameters_before)
    if not raises:
        assert "mutated parameters" in result["error"]


@pytest.mark.parametrize("kind", ["shape", "dtype", "device"])
def test_refiner_candidate_contract_is_checked_before_post_assessment(analytic_case, kind):
    case = analytic_case
    original = case["stationary"] + torch.tensor([0.1, -0.1], dtype=torch.float64)

    def refine(_, __):
        if kind == "shape":
            return torch.zeros(3, dtype=torch.float64)
        if kind == "dtype":
            return case["stationary"].float()
        return torch.empty_like(case["stationary"], device="meta")

    result = _call(case, original, refine=refine)
    assert result["status"] == "ineligible"
    assert result["response"] is None
    torch.testing.assert_close(result["control"], original)
    assert "wrong" in result["error"]


@pytest.mark.parametrize("positive_curvature", [True, False])
def test_matrix_free_refiner_integrates_or_refuses_without_partial_response(analytic_case, positive_curvature):
    from advar.local_refinement import refine_stationary

    case = analytic_case
    original_objective = case["objective"]
    if not positive_curvature:
        case["objective"] = lambda c, p: -original_objective(c, p)
    control = case["stationary"] + 0.1
    saved_control, saved_p = control.clone(), case["p"].clone()

    def refine(c, p):
        return refine_stationary(
            case["objective"], c, p, branch_check=case["branch_check"]
        ).control

    result = _call(case, control, refine=refine)
    torch.testing.assert_close(control, saved_control, rtol=0, atol=0)
    torch.testing.assert_close(case["p"], saved_p, rtol=0, atol=0)
    if positive_curvature:
        assert result["status"] == "eligible"
        expected = case["p"] + case["stationary"] @ torch.linalg.solve(case["A"], case["B"])
        torch.testing.assert_close(result["response"].total_gradient, expected)
    else:
        assert result["status"] == "ineligible"
        assert result["response"] is None
        torch.testing.assert_close(result["control"], saved_control, rtol=0, atol=0)
        assert "positive definite" in result["error"]
