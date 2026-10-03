"""Paired face composition and native point-objective equivalence checks."""
import json
from pathlib import Path

import pytest
import torch

from advar.transport import face_volume_fluxes
from examples.weather_scenarios.fv_point_paired_face_binding import FVPointPairedFaceBinding
from examples.weather_scenarios.fv_point_response_preflight import fixed_problem


@pytest.fixture(scope="module")
def case():
    problem, _, parameters, _ = fixed_problem()
    evidence = (Path(__file__).resolve().parents[1]
                / "graphify-out/fv-root-cause-20260919/point_active_face_stationarity_attempt1/stationarity.json")
    control = torch.tensor(json.loads(evidence.read_text())["last_accepted_control"], dtype=torch.float64)
    binding = FVPointPairedFaceBinding(problem)
    tangent = binding.tangent_coordinates(control)
    return problem, control, parameters, binding, tangent


def _gamma_budget(scale: float, *, stages: int, operations: int) -> float:
    count = 2 * stages * 20 * operations + 2 * 12 * 8
    n_epsilon = count * torch.finfo(torch.float64).eps
    return n_epsilon / (1 - n_epsilon) * scale


def test_two_projected_modes_have_exact_zeros_rank_three_and_divergence_free(case):
    _, _, _, binding, tangent = case
    qx, qy = binding.face_fluxes(tangent)
    assert torch.equal(qx[3, 4], torch.zeros_like(qx[3, 4]))
    assert torch.equal(qy[3, 0], torch.zeros_like(qy[3, 0]))
    divergence = qx[:, 1:] - qx[:, :-1] + qy[1:, :] - qy[:-1, :]
    scale = (qx[:, 1:].abs() + qx[:, :-1].abs()
             + qy[1:, :].abs() + qy[:-1, :].abs())
    assert bool((divergence.abs() <= 256 * torch.finfo(qx.dtype).eps * scale).all())
    assert torch.linalg.matrix_rank(binding.transport.psi_basis.flatten(1)) == 3
    assert binding.control_count == 24
    assert binding.reduced_problem.layout["controls"] == 24
    assert binding.reduced_problem.layout["parameters"] == 13
    assert binding.problem.layout["controls"] == 26
    assert binding.support["projected_mode_rank"] == 3
    assert binding.support["full_root_claim"] is False
    with pytest.raises(ValueError, match="strict smooth branch"):
        binding.problem.branch_check(binding.lift(tangent), case[2])


def _public_calls(binding, tangent, parameters):
    return (
        lambda: binding.lift(tangent),
        lambda: binding.objective(tangent, parameters),
        lambda: binding.forecast(tangent, parameters),
        lambda: binding.score(tangent, parameters),
        lambda: binding.face_fluxes(tangent),
    )


def test_each_removed_face_pivot_domain_is_enforced_by_all_public_methods(case):
    _, _, parameters, binding, _ = case

    # The second chart cannot recover its omitted coefficient for this requested face value.
    second_invalid = torch.zeros(24, dtype=torch.float64)
    second_invalid[20] = 100.0
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.second_chart.from_face_coordinates(torch.cat((
            second_invalid[:20], second_invalid.new_zeros(1), second_invalid[20:],
        )))
    for evaluate in _public_calls(binding, second_invalid, parameters):
        with pytest.raises(RuntimeError, match="strict representable chart domain"):
            evaluate()

    # Keep the second chart in-domain while selecting surviving first-chart flows
    # that make the original q_x pivot unrepresentable.
    first_invalid = torch.zeros(24, dtype=torch.float64)
    first_invalid[20:23] = torch.tensor([-0.7, -0.9, 0.9], dtype=torch.float64)
    first_tangent = binding.second_chart.from_face_coordinates(torch.cat((
        first_invalid[:20], first_invalid.new_zeros(1), first_invalid[20:],
    )))
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.first_binding.lift(first_tangent)
    for evaluate in _public_calls(binding, first_invalid, parameters):
        with pytest.raises(RuntimeError, match="strict representable chart domain"):
            evaluate()

    # The inverse chart's open interval includes pivots too small to survive
    # the forward chart's cancellation guard; the composed lift must reject it.
    near_cancellation = torch.zeros(24, dtype=torch.float64)
    near_cancellation[20] = 0.1
    near_cancellation[22] = torch.atanh(
        -(2.625 / 1.125) * torch.tanh(near_cancellation[20])
    )
    first_tangent = binding.second_chart.from_face_coordinates(torch.cat((
        near_cancellation[:20], near_cancellation.new_zeros(1), near_cancellation[20:],
    )))
    binding.first_binding.lift(first_tangent)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.second_chart.to_face_coordinates(first_tangent)
    for evaluate in _public_calls(binding, near_cancellation, parameters):
        with pytest.raises(RuntimeError, match="strict representable chart domain"):
            evaluate()


def test_reduced_problem_and_restored_priors_match_original_j_score_gradient_and_hvp(case):
    problem, _, parameters, binding, tangent = case
    lift = binding.lift(tangent)
    original = lambda value: problem.objective(binding.lift(value), parameters)
    paired = lambda value: binding.objective(value, parameters)

    reduced_j = binding.reduced_problem.objective(tangent, parameters)
    restored_prior = 0.5 * (lift[20].square() + lift[21].square())
    prior_scale = 0.5 * (lift.square().sum() + tangent.square().sum())
    assert abs(float(paired(tangent) - reduced_j - restored_prior)) <= _gamma_budget(
        float(prior_scale), stages=1, operations=4,
    )

    original_j, paired_j = original(tangent), paired(tangent)
    j_scale = abs(float(original_j)) + abs(float(paired_j))
    assert abs(float(original_j - paired_j)) <= _gamma_budget(j_scale, stages=54, operations=128)
    original_score = problem.score(lift, parameters)
    paired_score = binding.score(tangent, parameters)
    score_scale = abs(float(original_score)) + abs(float(paired_score))
    assert abs(float(original_score - paired_score)) <= _gamma_budget(
        score_scale, stages=54, operations=128,
    )

    original_gradient = torch.func.grad(original)(tangent)
    paired_gradient = torch.func.grad(paired)(tangent)
    gradient_scale = (torch.linalg.vector_norm(original_gradient)
                      + torch.linalg.vector_norm(paired_gradient)
                      + 2 * torch.linalg.vector_norm(torch.func.grad(
                          lambda value: 0.5 * binding.lift(value).square().sum()
                      )(tangent)))
    assert torch.linalg.vector_norm(original_gradient - paired_gradient) <= _gamma_budget(
        float(gradient_scale), stages=54, operations=768,
    )

    direction = torch.linspace(-0.3, 0.4, 24, dtype=torch.float64)
    original_hvp = torch.func.jvp(torch.func.grad(original), (tangent,), (direction,))[1]
    paired_hvp = torch.func.jvp(torch.func.grad(paired), (tangent,), (direction,))[1]
    hvp_scale = torch.linalg.vector_norm(original_hvp) + torch.linalg.vector_norm(paired_hvp)
    assert torch.linalg.vector_norm(original_hvp - paired_hvp) <= _gamma_budget(
        float(hvp_scale), stages=54, operations=3072,
    )
    qx, _ = face_volume_fluxes(torch.einsum(
        "k,kij->ij", problem.frozen.fv_transport.coefficient_limits
        * torch.tanh(lift[20:25]), problem.frozen.fv_transport.psi_basis,
    ))
    assert abs(float(qx[3, 4])) <= 128 * torch.finfo(torch.float64).eps * float(qx.abs().max())
