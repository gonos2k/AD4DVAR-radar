"""Analytic checks for fixed point-observation correlation whitening."""
from dataclasses import replace

import pytest
import torch

from advar import variational as v
from advar.fv_point_sampler import point_dbz_bilinear
from advar.physics import echo_to_dbz
from examples.weather_scenarios.fv_point_research_case import make_case


@pytest.fixture(scope="module")
def case():
    return make_case()


def _correlation(rho: float = 0.4) -> torch.Tensor:
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 1] = rho
    correlation[1, 0] = rho
    return correlation


def _block_inverse_sqrt(rho: float) -> torch.Tensor:
    """Closed form symmetric C^{-1/2} for the leading 2x2 block."""
    along_sum = (1.0 + rho) ** -0.5
    along_difference = (1.0 - rho) ** -0.5
    diagonal = 0.5 * (along_sum + along_difference)
    off_diagonal = 0.5 * (along_sum - along_difference)
    return torch.tensor(
        [[diagonal, off_diagonal], [off_diagonal, diagonal]],
        dtype=torch.float64,
    )


def _diagonal_data_residual(problem, control, parameters):
    contract = problem.contract(parameters)
    predicted_echo = v.analysis_trajectory(control, contract).frames_linear
    predicted_dbz = echo_to_dbz(predicted_echo, min_dbz=contract.nowcast_config.min_dbz)
    sampled = point_dbz_bilinear(predicted_dbz, problem.observation_coordinates)
    return problem.quality_weight.sqrt() * (sampled - parameters[:-1].reshape_as(
        problem.observation_dbz
    )) / problem.observation_std_dbz


def _non_data_cost(problem, control, parameters):
    contract = problem.contract(parameters)
    prior = v._control_prior_residual(control, contract)
    return 0.5 * torch.dot(prior, prior) + v._field_smoothness_prior_cost(control, contract)


def test_correlated_objective_matches_analytic_two_point_whitening_and_data_gradient(case):
    problem, control, parameters = case
    rho = 0.4
    correlated = replace(problem, observation_correlation=_correlation(rho))
    residual = _diagonal_data_residual(correlated, control, parameters)
    whitened = residual.clone()
    block = _block_inverse_sqrt(rho)
    whitened[:, :2] = residual[:, :2] @ block.T
    delta = correlated.frozen.analysis_config.pseudo_huber_delta
    expected_objective = (
        v._pseudo_huber_cost(whitened, delta).sum()
        + _non_data_cost(correlated, control, parameters)
    )
    torch.testing.assert_close(
        correlated.objective(control, parameters), expected_objective,
        rtol=1e-12, atol=1e-12,
    )

    transformed_slope = whitened / torch.sqrt(1.0 + (whitened / delta).square())
    pulled_slope = transformed_slope.clone()
    pulled_slope[:, :2] = transformed_slope[:, :2] @ block
    scale = correlated.quality_weight.sqrt() / correlated.observation_std_dbz
    expected_gradient = -pulled_slope * scale
    actual_gradient = torch.func.grad(correlated.objective, argnums=1)(control, parameters)
    torch.testing.assert_close(
        actual_gradient[:-1].reshape_as(expected_gradient), expected_gradient,
        rtol=1e-11, atol=1e-12,
    )


def test_correlation_changes_fixed_problem_identity(case):
    problem, _, _ = case
    correlated = replace(problem, observation_correlation=_correlation(0.4))
    changed = replace(problem, observation_correlation=_correlation(0.2))
    assert correlated.identity != problem.identity
    assert changed.identity != correlated.identity


def test_correlation_cannot_change_after_whitener_is_frozen(case):
    problem, control, parameters = case
    matrix = _correlation(0.4)
    correlated = replace(problem, observation_correlation=matrix)
    matrix[0, 1] = 0.2
    matrix[1, 0] = 0.2
    with pytest.raises(ValueError, match="changed after problem construction"):
        correlated.objective(control, parameters)
    with pytest.raises(ValueError, match="changed after problem construction"):
        _ = correlated.identity


def test_cached_whitener_cannot_change_without_invalidating_objective(case):
    problem, control, parameters = case
    correlated = replace(problem, observation_correlation=_correlation(0.4))
    whitener = correlated._correlation_whitener
    assert whitener is not None
    whitener[0, 0] *= 2
    with pytest.raises(ValueError, match="whitener changed after problem construction"):
        correlated.objective(control, parameters)
    with pytest.raises(ValueError, match="whitener changed after problem construction"):
        _ = correlated.identity


def test_default_objective_retains_diagonal_pseudohuber_contract(case):
    problem, control, parameters = case
    residual = _diagonal_data_residual(problem, control, parameters)
    expected = (
        v._pseudo_huber_cost(residual, problem.frozen.analysis_config.pseudo_huber_delta).sum()
        + _non_data_cost(problem, control, parameters)
    )
    torch.testing.assert_close(problem.objective(control, parameters), expected, rtol=0, atol=0)
    identity = replace(problem, observation_correlation=torch.eye(4, dtype=torch.float64))
    torch.testing.assert_close(identity.objective(control, parameters), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.func.grad(identity.objective, argnums=1)(control, parameters),
        torch.func.grad(problem.objective, argnums=1)(control, parameters),
        rtol=0, atol=0,
    )


def test_point_reordering_with_correlation_permutation_preserves_robust_cost(case):
    problem, control, parameters = case
    correlation = _correlation(0.4)
    correlated = replace(problem, observation_correlation=correlation)
    ordering = torch.tensor([2, 1, 0, 3])
    rearranged = replace(
        problem,
        observation_coordinates=problem.observation_coordinates[ordering],
        observation_dbz=problem.observation_dbz[:, ordering],
        observation_std_dbz=problem.observation_std_dbz[:, ordering],
        quality_weight=problem.quality_weight[:, ordering],
        observation_correlation=correlation[ordering][:, ordering],
    )
    rearranged_parameters = torch.cat(
        (parameters[:-1].reshape(3, 4)[:, ordering].flatten(), parameters[-1:])
    )
    torch.testing.assert_close(
        rearranged.objective(control, rearranged_parameters),
        correlated.objective(control, parameters), rtol=1e-12, atol=1e-12,
    )
    original_gradient = torch.func.grad(correlated.objective, argnums=1)(control, parameters)
    rearranged_gradient = torch.func.grad(rearranged.objective, argnums=1)(control, rearranged_parameters)
    torch.testing.assert_close(
        rearranged_gradient[:-1].reshape(3, 4),
        original_gradient[:-1].reshape(3, 4)[:, ordering], rtol=1e-11, atol=1e-12,
    )


def test_roundoff_asymmetry_is_interpreted_by_a_symmetric_matrix(case):
    problem, control, parameters = case
    asymmetric = _correlation(0.4)
    asymmetric[0, 1] += 16 * torch.finfo(torch.float64).eps
    symmetric = 0.5 * (asymmetric + asymmetric.T)
    first = replace(problem, observation_correlation=asymmetric)
    second = replace(problem, observation_correlation=symmetric)
    torch.testing.assert_close(first.objective(control, parameters),
                               second.objective(control, parameters), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.func.grad(first.objective, argnums=1)(control, parameters),
        torch.func.grad(second.objective, argnums=1)(control, parameters),
        rtol=1e-12, atol=1e-12,
    )
    assert first.identity != second.identity  # Raw declared inputs remain distinct.


@pytest.mark.parametrize("mutation", [
    "shape", "dtype", "requires_grad", "asymmetric", "diagonal",
    "singular", "near_singular",
])
def test_problem_rejects_invalid_observation_correlation(case, mutation):
    problem, _, _ = case
    correlation = _correlation()
    if mutation == "shape":
        correlation = torch.eye(3, dtype=torch.float64)
    elif mutation == "dtype":
        correlation = correlation.float()
    elif mutation == "requires_grad":
        correlation.requires_grad_()
    elif mutation == "asymmetric":
        correlation[0, 1] += 0.1
    elif mutation == "diagonal":
        correlation[0, 0] = 0.9
    elif mutation == "singular":
        correlation = _correlation(1.0)
    elif mutation == "near_singular":
        correlation = _correlation(1.0 - 0.5 * torch.finfo(torch.float64).eps ** 0.5)

    with pytest.raises(ValueError):
        replace(problem, observation_correlation=correlation)
