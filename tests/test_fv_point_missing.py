"""Missing point-observation masks and principal-submatrix whitening."""
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


def _status(pattern: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    """Convert per-time detected indices to the fixed 0/1 status layout."""
    status = torch.ones((3, 4), dtype=torch.uint8)
    for time, detected in enumerate(pattern):
        status[time, list(detected)] = 0
    return status


def _masked_case(problem, control, parameters, status):
    values = problem.observation_dbz.clone()
    values[status == 1] = problem.frozen.nowcast_config.min_dbz
    masked = replace(problem, observation_dbz=values, observation_status=status)
    return masked, parameters.clone()


def _prediction_dbz(problem, control, parameters):
    contract = problem.contract(parameters)
    echo = v.analysis_trajectory(control, contract).frames_linear
    return point_dbz_bilinear(
        echo_to_dbz(echo, min_dbz=contract.nowcast_config.min_dbz),
        problem.observation_coordinates,
    )


def _expected_masked_objective(problem, control, parameters, correlation=None):
    """Reference objective using only each time's valid principal covariance."""
    contract = problem.contract(parameters)
    observed = parameters[:-1].reshape_as(problem.observation_dbz)
    predicted = _prediction_dbz(problem, control, parameters)
    mask = problem.observation_status == 0
    data_cost = predicted.new_zeros(())
    for time in range(3):
        valid = torch.nonzero(mask[time], as_tuple=False).flatten()
        residual = (
            problem.quality_weight[time, valid].sqrt()
            * (predicted[time, valid] - observed[time, valid])
            / problem.observation_std_dbz[time, valid]
        )
        if correlation is not None:
            selected = correlation[valid][:, valid]
            eigenvalues, eigenvectors = torch.linalg.eigh(selected)
            whitener = (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.T
            residual = whitener @ residual
        data_cost = data_cost + v._pseudo_huber_cost(
            residual, contract.analysis_config.pseudo_huber_delta
        ).sum()
    prior = v._control_prior_residual(control, contract)
    return (
        data_cost + 0.5 * torch.dot(prior, prior)
        + v._field_smoothness_prior_cost(control, contract)
    )


def _parameters_for(problem, original):
    values = problem.observation_dbz.flatten()
    return torch.cat((values, original[-1:]))


def test_single_valid_point_uses_singleton_correlation_and_ignores_missing_payload(case):
    problem, control, parameters = case
    status = _status(((0,), (0,), (0,)))
    masked, parameters = _masked_case(problem, control, parameters, status)
    parameters = _parameters_for(masked, parameters)
    parameters[:-1].reshape_as(status)[status == 1] = 1.0e6
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 1] = correlation[1, 0] = 0.4
    masked = replace(masked, observation_correlation=correlation)

    expected = _expected_masked_objective(masked, control, parameters, correlation)
    torch.testing.assert_close(masked.objective(control, parameters), expected, rtol=1e-12, atol=1e-12)

    # A full four-point whitener would mix the deliberately large missing payload
    # into each singleton residual. Principal-submatrix whitening cannot do so.
    full_residual = (
        masked.quality_weight.sqrt()
        * (_prediction_dbz(masked, control, parameters) - parameters[:-1].reshape_as(masked.observation_dbz))
        / masked.observation_std_dbz
    ) @ masked._correlation_whitener.T
    full_cost = (
        v._pseudo_huber_cost(full_residual, masked.frozen.analysis_config.pseudo_huber_delta).sum()
        + 0.5 * torch.dot(v._control_prior_residual(control, masked.contract(parameters)),
                          v._control_prior_residual(control, masked.contract(parameters)))
        + v._field_smoothness_prior_cost(control, masked.contract(parameters))
    )
    assert not torch.isclose(expected, full_cost, rtol=1e-8, atol=1e-8)


def test_time_varying_masks_match_per_time_principal_submatrices(case):
    problem, control, parameters = case
    status = _status(((0,), (1, 2), (0, 2, 3)))
    masked, parameters = _masked_case(problem, control, parameters, status)
    parameters = _parameters_for(masked, parameters)
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 1] = correlation[1, 0] = 0.35
    correlation[1, 2] = correlation[2, 1] = -0.2
    correlation[0, 3] = correlation[3, 0] = 0.15
    masked = replace(masked, observation_correlation=correlation)

    expected = _expected_masked_objective(masked, control, parameters, correlation)
    torch.testing.assert_close(masked.objective(control, parameters), expected, rtol=2e-12, atol=2e-12)


def test_missing_payload_has_zero_objective_and_mixed_derivatives_and_no_forecast_effect(case):
    problem, control, parameters = case
    status = _status(((0, 2), (1, 3), (0, 1, 3)))
    masked, parameters = _masked_case(problem, control, parameters, status)
    parameters = _parameters_for(masked, parameters)
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 1] = correlation[1, 0] = 0.35
    correlation[1, 2] = correlation[2, 1] = -0.2
    masked = replace(masked, observation_correlation=correlation)
    missing = status == 1
    changed = parameters.clone()
    changed[:-1].reshape_as(status)[missing] += 100.0

    torch.testing.assert_close(masked.objective(control, changed), masked.objective(control, parameters), rtol=0, atol=0)
    torch.testing.assert_close(masked.forecast(control, changed), masked.forecast(control, parameters), rtol=0, atol=0)
    torch.testing.assert_close(masked.score(control, changed), masked.score(control, parameters), rtol=0, atol=0)

    objective_gradient = torch.func.grad(masked.objective, argnums=1)(control, changed)
    torch.testing.assert_close(
        objective_gradient[:-1].reshape_as(status)[missing],
        torch.zeros_like(objective_gradient[:-1].reshape_as(status)[missing]),
        rtol=0, atol=0,
    )
    data_direction = torch.zeros_like(changed)
    data_direction[:-1].reshape_as(status)[missing] = 1.0
    forecast_tangent = torch.func.jvp(
        lambda p: masked.forecast(control, p), (changed,), (data_direction,)
    )[1]
    torch.testing.assert_close(forecast_tangent, torch.zeros_like(forecast_tangent), rtol=0, atol=0)
    control_gradient = torch.func.grad(masked.objective, argnums=0)
    mixed = torch.func.jvp(
        lambda p: control_gradient(control, p), (changed,), (data_direction,)
    )[1]
    torch.testing.assert_close(mixed, torch.zeros_like(mixed), rtol=0, atol=0)


def test_active_observation_range_is_checked_but_missing_payload_may_match(case):
    problem, control, parameters = case
    status = _status(((0, 2), (1, 3), (0, 1, 3)))
    masked, parameters = _masked_case(problem, control, parameters, status)
    parameters = _parameters_for(masked, parameters)
    missing = status == 1
    active = status == 0
    out_of_range_values = (
        masked.frozen.analysis_config.detection_limit_dbz - 1.0,
        masked.frozen.nowcast_config.max_dbz,
    )
    active_time, active_point = torch.nonzero(active, as_tuple=False)[0].tolist()

    for out_of_range in out_of_range_values:
        changed = parameters.clone()
        changed[:-1].reshape_as(status)[missing] = out_of_range
        torch.testing.assert_close(
            masked.objective(control, changed), masked.objective(control, parameters),
            rtol=0, atol=0,
        )

        invalid_active = parameters.clone()
        invalid_active[:-1].reshape_as(status)[active_time, active_point] = out_of_range
        with pytest.raises(ValueError, match="remain detected observations"):
            masked.objective(control, invalid_active)


@pytest.mark.parametrize("status_mutation", [
    "dtype", "shape", "quality_control", "censored",
])
def test_status_contract_rejects_invalid_layout_or_unsupported_codes(case, status_mutation):
    problem, _, _ = case
    status = torch.zeros((3, 4), dtype=torch.uint8)
    if status_mutation == "dtype":
        status = status.to(torch.int64)
    elif status_mutation == "shape":
        status = status[:, :-1]
    else:
        status[1, 0] = 2 if status_mutation == "quality_control" else 3
    with pytest.raises(ValueError):
        replace(problem, observation_status=status)


@pytest.mark.parametrize("status", [
    torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.uint8),
])
def test_every_frame_requires_at_least_one_detected_point(case, status):
    problem, _, _ = case
    with pytest.raises(ValueError, match="at least one detected"):
        replace(problem, observation_status=status)


def test_missing_stored_values_must_use_canonical_minimum_dbz(case):
    problem, _, _ = case
    status = _status(((0,), (0,), (0,)))
    values = problem.observation_dbz.clone()
    values[status == 1] = problem.frozen.nowcast_config.min_dbz + 1
    with pytest.raises(ValueError, match="canonical"):
        replace(problem, observation_status=status, observation_dbz=values)


def test_missing_mask_and_cached_subset_whiteners_are_bound_to_identity(case):
    problem, control, parameters = case
    status = _status(((0, 1), (1,), (2,)))
    masked, parameters = _masked_case(problem, control, parameters, status)
    parameters = _parameters_for(masked, parameters)
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 1] = correlation[1, 0] = 0.4
    masked = replace(masked, observation_correlation=correlation)
    assert masked.identity != problem.identity

    status[0, 2] = 0
    with pytest.raises(ValueError, match="status changed after problem construction"):
        masked.objective(control, parameters)

    status = _status(((0, 1), (1,), (2,)))
    masked, parameters = _masked_case(problem, control, parameters, status)
    masked = replace(masked, observation_correlation=correlation)
    assert masked._masked_whiteners is not None
    assert masked._masked_whiteners[0] is not None
    masked._masked_whiteners[0][0, 0] *= 2
    with pytest.raises(ValueError, match="whitener changed after construction"):
        masked.objective(control, parameters)


def test_point_permutation_carries_status_data_and_correlation(case):
    problem, control, parameters = case
    status = _status(((0, 1, 3), (0, 2, 3), (0, 1, 2)))
    masked, parameters = _masked_case(problem, control, parameters, status)
    parameters = _parameters_for(masked, parameters)
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 1] = correlation[1, 0] = 0.3
    correlation[1, 2] = correlation[2, 1] = -0.1
    masked = replace(masked, observation_correlation=correlation)

    ordering = torch.tensor([2, 0, 3, 1])
    rearranged = replace(
        masked,
        observation_coordinates=masked.observation_coordinates[ordering],
        observation_dbz=masked.observation_dbz[:, ordering],
        observation_status=masked.observation_status[:, ordering],
        observation_std_dbz=masked.observation_std_dbz[:, ordering],
        quality_weight=masked.quality_weight[:, ordering],
        observation_correlation=correlation[ordering][:, ordering],
    )
    rearranged_parameters = torch.cat((
        parameters[:-1].reshape(3, 4)[:, ordering].flatten(), parameters[-1:]
    ))
    torch.testing.assert_close(
        rearranged.objective(control, rearranged_parameters),
        masked.objective(control, parameters), rtol=2e-12, atol=2e-12,
    )


def test_all_detected_status_matches_legacy_none_contract(case):
    problem, control, parameters = case
    all_detected = replace(problem, observation_status=torch.zeros((3, 4), dtype=torch.uint8))
    assert all_detected.observation_status is not None
    torch.testing.assert_close(all_detected.objective(control, parameters), problem.objective(control, parameters), rtol=0, atol=0)
    torch.testing.assert_close(all_detected.forecast(control, parameters), problem.forecast(control, parameters), rtol=0, atol=0)
    torch.testing.assert_close(all_detected.score(control, parameters), problem.score(control, parameters), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.func.grad(all_detected.objective, argnums=1)(control, parameters),
        torch.func.grad(problem.objective, argnums=1)(control, parameters), rtol=0, atol=0,
    )

    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 1] = correlation[1, 0] = 0.4
    correlated_legacy = replace(problem, observation_correlation=correlation)
    correlated_status = replace(
        correlated_legacy,
        observation_status=torch.zeros((3, 4), dtype=torch.uint8),
    )
    torch.testing.assert_close(
        correlated_status.objective(control, parameters),
        correlated_legacy.objective(control, parameters), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        torch.func.grad(correlated_status.objective, argnums=1)(control, parameters),
        torch.func.grad(correlated_legacy.objective, argnums=1)(control, parameters),
        rtol=0, atol=0,
    )
