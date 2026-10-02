"""Tests for the fixed q_y[2,0]=0 coordinate binding (no solver use)."""
import pytest
import torch

from advar import variational as v
from advar.transport import face_volume_fluxes
from examples.weather_scenarios.fv_active_face_binding import FVActiveFaceBinding
from tests.test_fv_research_partial_observation import _problem


def _case():
    problem, original, parameters = _problem()
    return problem, original, parameters, FVActiveFaceBinding(problem)


def _roundoff_budget(scale: float, *, stages: int, operations_per_cell_stage: int) -> float:
    """Two-path gamma_n comparison budget, not a forward-error certificate.

    Counts are per cell/stage plus eight scalar operations per observation
    residual; the 4x5 operator has 20 cells, 60 residual entries and 36
    analysis Euler stages (54 for score including one forecast interval).
    Gradient/HVP counts conservatively cover reverse/second-order work.
    """
    operations = 2 * stages * 20 * operations_per_cell_stage + 2 * 60 * 8
    n_epsilon = operations * torch.finfo(torch.float64).eps
    return (n_epsilon / (1 - n_epsilon)) * scale


def test_projected_modes_are_structurally_zero_face_rank_four_and_divergence_free():
    _, original, _, binding = _case()
    tangent = binding.tangent_coordinates(original)
    qx, qy = binding.face_fluxes(tangent)
    assert torch.equal(qy[2, 0], torch.zeros_like(qy[2, 0]))
    divergence = (qx[:, 1:] - qx[:, :-1]) + (qy[1:, :] - qy[:-1, :])
    scale = (qx[:, 1:].abs() + qx[:, :-1].abs()
             + qy[1:, :].abs() + qy[:-1, :].abs())
    assert bool((divergence.abs() <= 256 * torch.finfo(qx.dtype).eps * scale).all())
    assert binding.support["structural_face_zero"] is True


def test_lift_refuses_original_pivot_latent_boundary_without_clipping():
    problem, _, parameters, binding = _case()
    tangent = torch.zeros(binding.control_count, dtype=torch.float64)
    tangent[21:24] = 0.7  # Original flow modes 2, 3, 4 drive |alpha_p| above L_p.
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.lift(tangent)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.objective(tangent, parameters)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.forecast(tangent, parameters)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        binding.face_fluxes(tangent)


def test_objective_lift_keeps_full_prior_and_matches_full_problem_derivatives():
    problem, original, parameters, binding = _case()
    tangent = binding.tangent_coordinates(original)
    lifted = binding.lift(tangent)
    observations, original_frozen, reduced_frozen = binding._fixed_state(parameters)
    residual = v.whitened_observation_residual(tangent, observations, reduced_frozen)
    full_prior_objective = v._robust_objective_from_residual(
        lifted, residual, observations, original_frozen,
    )
    reduced_prior_objective = v._robust_objective_from_residual(
        tangent, residual, observations, reduced_frozen,
    )
    pivot = binding.field_count + binding.pivot_index
    prior_difference = 0.5 * lifted[pivot].square()
    prior_scale = 0.5 * (float(lifted.square().sum()) + float(tangent.square().sum()))
    torch.testing.assert_close(full_prior_objective - reduced_prior_objective,
                               prior_difference, rtol=0.0,
                               atol=_roundoff_budget(prior_scale, stages=1,
                                                     operations_per_cell_stage=4))
    assert float(prior_difference) > 0.0
    assert problem.frozen.analysis_config.field_smoothness_weight == 0.0

    original_objective = lambda value: problem.objective(binding.lift(value), parameters)
    reduced_objective = lambda value: binding.objective(value, parameters)
    original_value, reduced_value = original_objective(tangent), reduced_objective(tangent)
    objective_scale = abs(float(original_value)) + abs(float(reduced_value))
    assert abs(float(original_value - reduced_value)) <= _roundoff_budget(
        objective_scale, stages=36, operations_per_cell_stage=64,
    )

    original_score = problem.score(lifted, parameters)
    reduced_score = binding.score(tangent, parameters)
    score_scale = abs(float(original_score)) + abs(float(reduced_score))
    assert abs(float(original_score - reduced_score)) <= _roundoff_budget(
        score_scale, stages=54, operations_per_cell_stage=64,
    )

    parameter_direction = torch.zeros_like(parameters)
    parameter_direction[0] = 0.5
    parameter_direction[-1] = 0.2
    original_parameter_jvp = torch.func.jvp(
        lambda value: problem.objective(lifted, value), (parameters,), (parameter_direction,),
    )[1]
    reduced_parameter_jvp = torch.func.jvp(
        lambda value: binding.objective(tangent, value), (parameters,), (parameter_direction,),
    )[1]
    parameter_scale = abs(float(original_parameter_jvp)) + abs(float(reduced_parameter_jvp))
    parameter_budget = _roundoff_budget(parameter_scale, stages=36,
                                        operations_per_cell_stage=256)
    assert abs(float(original_parameter_jvp - reduced_parameter_jvp)) <= parameter_budget

    original_gradient = torch.func.grad(original_objective)(tangent)
    reduced_gradient = torch.func.grad(reduced_objective)(tangent)
    original_prior = lambda value: 0.5 * binding.lift(value).square().sum()
    reduced_prior = lambda value: 0.5 * value.square().sum()
    original_prior_gradient = torch.func.grad(original_prior)(tangent)
    reduced_prior_gradient = torch.func.grad(reduced_prior)(tangent)
    # ||g_total||+2||g_prior|| bounds ||g_data||+||g_prior|| without
    # allowing cancellation between the data and prior derivative terms.
    gradient_scale = (
        float(torch.linalg.vector_norm(original_gradient))
        + 2 * float(torch.linalg.vector_norm(original_prior_gradient))
        + float(torch.linalg.vector_norm(reduced_gradient))
        + 2 * float(torch.linalg.vector_norm(reduced_prior_gradient))
    )
    gradient_budget = _roundoff_budget(gradient_scale, stages=36,
                                       operations_per_cell_stage=256)
    assert float(torch.linalg.vector_norm(original_gradient - reduced_gradient)) <= gradient_budget

    direction = torch.linspace(-0.3, 0.4, tangent.numel(), dtype=tangent.dtype)
    original_hvp = torch.func.jvp(torch.func.grad(original_objective),
                                  (tangent,), (direction,))[1]
    reduced_hvp = torch.func.jvp(torch.func.grad(reduced_objective),
                                 (tangent,), (direction,))[1]
    original_prior_hvp = torch.func.jvp(torch.func.grad(original_prior),
                                        (tangent,), (direction,))[1]
    reduced_prior_hvp = torch.func.jvp(torch.func.grad(reduced_prior),
                                       (tangent,), (direction,))[1]
    hvp_scale = (
        float(torch.linalg.vector_norm(original_hvp))
        + 2 * float(torch.linalg.vector_norm(original_prior_hvp))
        + float(torch.linalg.vector_norm(reduced_hvp))
        + 2 * float(torch.linalg.vector_norm(reduced_prior_hvp))
    )
    hvp_budget = _roundoff_budget(hvp_scale, stages=36, operations_per_cell_stage=1024)
    assert float(torch.linalg.vector_norm(original_hvp - reduced_hvp)) <= hvp_budget
