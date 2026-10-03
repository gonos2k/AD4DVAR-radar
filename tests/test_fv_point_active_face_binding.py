"""Fixed point active-face binding checks; no optimization or response run."""
from dataclasses import replace

import pytest
import torch

from advar.transport import face_volume_fluxes
from examples.weather_scenarios.fv_point_active_face_binding import FVPointActiveFaceBinding
from examples.weather_scenarios.fv_point_response_preflight import fixed_problem


@pytest.fixture(scope="module")
def case():
    problem, warm, parameters, _ = fixed_problem()
    return problem, warm, parameters, FVPointActiveFaceBinding(problem)


def _budget(scale: float, *, stages: int, operations: int) -> float:
    count = 2 * stages * 20 * operations + 2 * 12 * 8
    n_epsilon = count * torch.finfo(torch.float64).eps
    return n_epsilon / (1 - n_epsilon) * scale


def test_projected_basis_has_structural_zero_rank_four_and_divergence_free(case):
    problem, warm, parameters, binding = case
    tangent = binding.tangent_coordinates(warm)
    qx, qy = binding.face_fluxes(tangent)
    assert torch.equal(qx[3, 4], torch.zeros_like(qx[3, 4]))
    divergence = qx[:, 1:] - qx[:, :-1] + qy[1:, :] - qy[:-1, :]
    scale = (qx[:, 1:].abs() + qx[:, :-1].abs()
             + qy[1:, :].abs() + qy[:-1, :].abs())
    assert bool((divergence.abs() <= 256 * torch.finfo(qx.dtype).eps * scale).all())
    assert torch.linalg.matrix_rank(binding.transport.psi_basis.flatten(1)) == 4
    assert binding.support["projected_mode_rank"] == 4
    assert torch.equal(binding.chart.weights, torch.tensor([1.0, 0.0, 4.0, -3.5, 16.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="strict smooth branch"):
        problem.branch_check(binding.lift(tangent), parameters)


def test_original_pivot_boundary_is_refused_by_all_public_evaluators(case):
    problem, _, parameters, binding = case
    tangent = torch.zeros(25, dtype=torch.float64)
    tangent[21:24] = 0.7
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.lift(tangent)
    for evaluate in (binding.objective, binding.forecast, binding.score):
        with pytest.raises(RuntimeError, match="strict representable chart domain"):
            evaluate(tangent, parameters)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.face_fluxes(tangent)
    assert problem.support["response_validation"] == "not_performed"
    assert binding.support["full_root_claim"] is False


def test_dependent_projected_modes_refuse_even_when_point_layout_is_unchanged(case):
    problem, _, _, _ = case
    spec = problem.frozen.fv_transport
    assert spec is not None
    basis = spec.psi_basis.clone()
    basis[4] = basis[2]
    dependent_spec = replace(spec, psi_basis=basis)
    dependent_problem = replace(problem, frozen=replace(problem.frozen, fv_transport=dependent_spec))
    with pytest.raises(ValueError, match="projected flow basis must have rank four"):
        FVPointActiveFaceBinding(dependent_problem)


def test_reduced_problem_plus_missing_pivot_prior_matches_original_primal_gradient_and_hvp(case):
    problem, warm, parameters, binding = case
    tangent = binding.tangent_coordinates(warm)
    lifted = binding.lift(tangent)
    pivot = binding.field_count + binding.pivot_index
    reduced = binding.reduced_problem.objective
    original_on_chart = lambda value: problem.objective(binding.lift(value), parameters)
    reduced_full_prior = lambda value: binding.objective(value, parameters)
    prior_difference = 0.5 * lifted[pivot].square()
    prior_scale = 0.5 * (float(lifted.square().sum()) + float(tangent.square().sum()))
    assert abs(float(binding.objective(tangent, parameters)
                     - reduced(tangent, parameters) - prior_difference)) <= _budget(prior_scale, stages=1, operations=4)

    original_j = original_on_chart(tangent)
    reduced_j = reduced_full_prior(tangent)
    j_scale = abs(float(original_j)) + abs(float(reduced_j))
    assert abs(float(original_j - reduced_j)) <= _budget(j_scale, stages=54, operations=128)

    original_score = problem.score(lifted, parameters)
    reduced_score = binding.score(tangent, parameters)
    score_scale = abs(float(original_score)) + abs(float(reduced_score))
    assert abs(float(original_score - reduced_score)) <= _budget(score_scale, stages=54, operations=128)

    original_gradient = torch.func.grad(original_on_chart)(tangent)
    reduced_gradient = torch.func.grad(reduced_full_prior)(tangent)
    prior = lambda value: 0.5 * binding.lift(value).square().sum()
    reduced_prior = lambda value: 0.5 * value.square().sum()
    gradient_scale = (float(torch.linalg.vector_norm(original_gradient))
                      + 2 * float(torch.linalg.vector_norm(torch.func.grad(prior)(tangent)))
                      + float(torch.linalg.vector_norm(reduced_gradient))
                      + 2 * float(torch.linalg.vector_norm(torch.func.grad(reduced_prior)(tangent))))
    assert float(torch.linalg.vector_norm(original_gradient - reduced_gradient)) <= _budget(
        gradient_scale, stages=54, operations=512,
    )

    direction = torch.linspace(-0.3, 0.4, 25, dtype=torch.float64)
    original_hvp = torch.func.jvp(torch.func.grad(original_on_chart), (tangent,), (direction,))[1]
    reduced_hvp = torch.func.jvp(torch.func.grad(reduced_full_prior), (tangent,), (direction,))[1]
    original_prior_hvp = torch.func.jvp(torch.func.grad(prior), (tangent,), (direction,))[1]
    reduced_prior_hvp = torch.func.jvp(torch.func.grad(reduced_prior), (tangent,), (direction,))[1]
    hvp_scale = (float(torch.linalg.vector_norm(original_hvp))
                 + 2 * float(torch.linalg.vector_norm(original_prior_hvp))
                 + float(torch.linalg.vector_norm(reduced_hvp))
                 + 2 * float(torch.linalg.vector_norm(reduced_prior_hvp)))
    assert float(torch.linalg.vector_norm(original_hvp - reduced_hvp)) <= _budget(
        hvp_scale, stages=54, operations=2048,
    )

    original_qx, _ = face_volume_fluxes(torch.einsum(
        "k,kij->ij", problem.frozen.fv_transport.coefficient_limits
        * torch.tanh(lifted[20:25]), problem.frozen.fv_transport.psi_basis,
    ))
    assert abs(float(original_qx[3, 4])) <= 128 * torch.finfo(torch.float64).eps * float(
        original_qx.abs().max()
    )
