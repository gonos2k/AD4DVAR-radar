"""Explicit one-empty-time point-observation research profile."""

from dataclasses import fields, replace

import pytest
import torch

from advar import variational as v
from advar.fv_point_sampler import point_dbz_bilinear
from advar.physics import echo_to_dbz
from examples.weather_scenarios.fv_point_research_case import make_case


@pytest.fixture(scope="module")
def case():
    return make_case()


def _empty_case(problem, time_index: int):
    status = torch.zeros((3, 4), dtype=torch.uint8)
    status[time_index] = torch.tensor([1, 2, 1, 2], dtype=torch.uint8)
    status[(time_index + 1) % 3, 1] = 2
    values = problem.observation_dbz.clone()
    values[status != 0] = problem.frozen.nowcast_config.min_dbz
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 2] = correlation[2, 0] = 0.3
    correlation[1, 3] = correlation[3, 1] = -0.2
    return replace(problem, observation_dbz=values,
                   observation_status=status, observation_correlation=correlation,
                   empty_observation_time=time_index), status


def _two_time_objective(problem, control, parameters):
    contract = problem.contract(parameters)
    predicted_echo = v.analysis_trajectory(control, contract).frames_linear
    sampled = point_dbz_bilinear(
        echo_to_dbz(predicted_echo, min_dbz=contract.nowcast_config.min_dbz),
        problem.observation_coordinates,
    )
    observed = parameters[:-1].reshape_as(problem.observation_dbz)
    data_cost = control.new_zeros(())
    for time_index in range(3):
        valid = (problem.observation_status[time_index] == 0).nonzero().flatten()
        if valid.numel() == 0:
            continue
        residual = (problem.quality_weight[time_index, valid].sqrt()
                    * (sampled[time_index, valid] - observed[time_index, valid])
                    / problem.observation_std_dbz[time_index, valid])
        correlation = problem.observation_correlation[valid][:, valid]
        values, vectors = torch.linalg.eigh(correlation)
        residual = (vectors * values.rsqrt().unsqueeze(0)) @ vectors.T @ residual
        data_cost = data_cost + v._pseudo_huber_cost(
            residual, contract.analysis_config.pseudo_huber_delta
        ).sum()
    prior = v._control_prior_residual(control, contract)
    return (data_cost + 0.5 * torch.dot(prior, prior)
            + v._field_smoothness_prior_cost(control, contract))


@pytest.mark.parametrize("empty_time", [0, 1, 2])
def test_one_declared_empty_time_has_zero_likelihood_and_inactive_derivatives(case, empty_time):
    problem, control, parameters = case
    selected, status = _empty_case(problem, empty_time)
    assert selected.support["response_validation"] == "not_performed"
    assert selected.support["general_minmod_response_eligible"] is False
    assert selected.support["observation_masks"].count(str(empty_time)) >= 1
    torch.testing.assert_close(selected.objective(control, parameters),
                               _two_time_objective(selected, control, parameters),
                               rtol=1e-12, atol=1e-12)
    gradient = torch.func.grad(selected.objective, argnums=1)(control, parameters)
    assert bool(torch.isfinite(gradient).all())
    assert gradient[-1] != 0  # The exogenous state-grid background still depends on theta.
    assert torch.count_nonzero(gradient[:-1].reshape_as(status)[status != 0]) == 0
    direction = torch.zeros_like(parameters)
    direction[:-1].reshape_as(status)[status == 0] = 0.0
    direction[:-1].reshape_as(status)[status != 0] = 1.0
    mixed = torch.func.jvp(
        lambda p: torch.func.grad(selected.objective, argnums=0)(control, p),
        (parameters,), (direction,),
    )[1]
    assert torch.count_nonzero(mixed) == 0
    changed = parameters + 100.0 * direction
    torch.testing.assert_close(selected.objective(control, changed),
                               selected.objective(control, parameters), rtol=0, atol=0)
    torch.testing.assert_close(selected.forecast(control, changed),
                               selected.forecast(control, parameters), rtol=0, atol=0)
    torch.testing.assert_close(selected.score(control, changed),
                               selected.score(control, parameters), rtol=0, atol=0)
    assert selected.identity["fixed_problem_sha256"] != problem.identity["fixed_problem_sha256"]


def test_default_and_mismatched_empty_time_declarations_fail_closed(case):
    problem, _control, _parameters = case
    selected, status = _empty_case(problem, 1)
    with pytest.raises(ValueError, match="at least one detected"):
        replace(selected, empty_observation_time=None)
    for invalid in (0, 2, True, -1, 3):
        with pytest.raises(ValueError, match="empty observation time"):
            replace(selected, empty_observation_time=invalid)

    two_empty = status.clone()
    two_empty[0] = 1
    values = selected.observation_dbz.clone()
    values[0] = problem.frozen.nowcast_config.min_dbz
    with pytest.raises(ValueError, match="empty observation time"):
        replace(selected, observation_status=two_empty, observation_dbz=values)


def test_legacy_nonempty_identity_is_unchanged(case):
    problem, _control, _parameters = case
    assert problem.identity["fixed_problem_sha256"] == (
        "dd8ec41e5c1a115fe58f4862462c39614234bf31b5065fc634711f864db06f0f"
    )


def test_legacy_positional_expected_branch_slot_is_preserved(case):
    problem, _control, _parameters = case
    marker = {"choices": [], "face_signs": []}
    with_branch = replace(problem, expected_branch=marker)
    legacy_arguments = [
        getattr(with_branch, item.name) for item in fields(with_branch)
        if item.init and item.name != "empty_observation_time"
    ]
    reconstructed = type(problem)(*legacy_arguments)
    assert reconstructed.expected_branch is marker
    assert reconstructed.empty_observation_time is None


@pytest.mark.parametrize("inactive_status", [1, 2])
def test_entire_declared_empty_time_may_have_one_inactive_reason(case, inactive_status):
    problem, _control, _parameters = case
    status = torch.zeros((3, 4), dtype=torch.uint8)
    status[1] = inactive_status
    values = problem.observation_dbz.clone()
    values[1] = problem.frozen.nowcast_config.min_dbz
    selected = replace(problem, observation_status=status,
                       observation_dbz=values, empty_observation_time=1)
    assert selected.empty_observation_time == 1
    assert "declared empty observation time 1" in selected.support["observation_masks"]
