"""Fixed off-grid observation dependency and derivative contract."""
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from advar import variational as v
from advar.fv_point_sampler import point_dbz_bilinear
from advar.physics import echo_to_dbz
from examples.weather_scenarios import fv_point_research_case
from examples.weather_scenarios.fv_point_research_case import make_case


@pytest.fixture(scope="module")
def case():
    return make_case()


def test_point_problem_has_separate_state_observation_and_time_layout(case):
    problem, control, parameters = case
    assert problem.layout["state_shape"] == (4, 5)
    assert problem.layout["observation_shape"] == (3, 4)
    assert problem.layout["controls"] == 26
    assert problem.layout["parameters"] == 13
    assert problem.layout["observation_times_seconds"] == (0.0, 60.0, 120.0)
    assert problem.layout["forecast_time_seconds"] == 180.0
    assert problem.support["observation_operator"] == "point_dbz_bilinear"
    assert problem.support["general_minmod_response_eligible"] is False
    assert problem.branch_check(control, parameters)[0]["euler_stages"] == 54
    assert torch.isfinite(problem.objective(control, parameters))
    assert torch.isfinite(problem.score(control, parameters))


def test_point_values_do_not_redefine_the_fixed_state_grid_background(case):
    problem, control, parameters = case
    changed = parameters.clone()
    changed[:4] += 0.01
    assert torch.equal(
        problem.contract(parameters).initial_background_dbz,
        problem.contract(changed).initial_background_dbz,
    )
    assert torch.equal(problem.forecast(control, parameters), problem.forecast(control, changed))
    assert torch.equal(problem.score(control, parameters), problem.score(control, changed))
    assert not torch.equal(problem.objective(control, parameters), problem.objective(control, changed))
    assert torch.equal(
        torch.func.grad(problem.score, argnums=1)(control, parameters)[:-1],
        torch.zeros_like(parameters[:-1]),
    )


def test_point_objective_data_gradient_matches_diagonal_pseudohuber(case):
    problem, control, parameters = case
    contract = problem.contract(parameters)
    echo = v.analysis_trajectory(control, contract).frames_linear
    predicted = point_dbz_bilinear(
        echo_to_dbz(echo, min_dbz=contract.nowcast_config.min_dbz),
        problem.observation_coordinates,
    )
    residual = (
        problem.quality_weight.sqrt() * (predicted - problem.observation_dbz)
        / problem.observation_std_dbz
    )
    delta = contract.analysis_config.pseudo_huber_delta
    expected = -(
        problem.quality_weight.sqrt() / problem.observation_std_dbz
        * residual / torch.sqrt(1 + (residual / delta).square())
    )
    actual = torch.func.grad(problem.objective, argnums=1)(control, parameters)
    torch.testing.assert_close(actual[:-1].reshape_as(expected), expected, rtol=1e-11, atol=1e-12)


def test_point_objective_keeps_control_and_data_mixed_derivatives(case):
    problem, control, parameters = case
    control_direction = torch.sin(torch.arange(control.numel(), dtype=control.dtype))
    data_direction = torch.cos(torch.arange(parameters.numel(), dtype=parameters.dtype))
    data_direction[-1] = 0
    gradient = torch.func.grad(problem.objective, argnums=0)
    mixed = torch.func.jvp(
        lambda p: gradient(control, p), (parameters,), (data_direction,)
    )[1]
    _, pullback = cast(
        tuple[torch.Tensor, Any],
        torch.func.vjp(lambda p: gradient(control, p), parameters, has_aux=False),
    )
    transpose = pullback(control_direction)[0]
    assert float(mixed.norm()) > 0
    torch.testing.assert_close(
        torch.dot(control_direction, mixed),
        torch.dot(data_direction, transpose), rtol=1e-11, atol=1e-12,
    )
    step = 1e-4
    central = (
        problem.objective(control + step * control_direction, parameters)
        - problem.objective(control - step * control_direction, parameters)
    ) / (2 * step)
    torch.testing.assert_close(
        central, torch.dot(gradient(control, parameters), control_direction),
        rtol=2e-5, atol=1e-9,
    )


def test_point_objective_hvp_matches_branch_stable_central_gradients(case):
    problem, control, parameters = case
    direction = torch.sin(torch.arange(control.numel(), dtype=control.dtype))
    nominal, _ = problem.branch_check(control, parameters)
    gradient = torch.func.grad(problem.objective, argnums=0)
    hvp = torch.func.jvp(lambda value: gradient(value, parameters),
                         (control,), (direction,))[1]
    errors = []
    for step in (1e-4, 5e-5):
        plus = control + step * direction
        minus = control - step * direction
        for endpoint in (plus, minus):
            branch, _ = problem.branch_check(endpoint, parameters)
            assert branch["choices"] == nominal["choices"]
            assert branch["face_signs"] == nominal["face_signs"]
        central = (gradient(plus, parameters) - gradient(minus, parameters)) / (2 * step)
        errors.append(float((central - hvp).norm() / hvp.norm()))
    assert errors[1] < 0.3 * errors[0]
    assert errors[1] < 2e-7


def test_geometry_and_source_identity_are_bound(case):
    problem, _, _ = case
    assert problem.source_sha256 == hashlib.sha256(
        Path(fv_point_research_case.__file__).read_bytes()
    ).hexdigest()
    moved = replace(problem, observation_coordinates=problem.observation_coordinates + 0.01)
    other_source = replace(problem, source_sha256="0" * 64)
    assert moved.identity != problem.identity
    assert other_source.identity != problem.identity


def test_nominal_branch_can_be_explicitly_pinned_for_response_checks(case):
    problem, control, parameters = case
    nominal, _ = problem.branch_check(control, parameters)
    generator = torch.Generator().manual_seed(91)
    candidate = control + 1e-3 * torch.randn(
        control.shape, generator=generator, dtype=control.dtype
    )
    other, _ = problem.branch_check(candidate, parameters)
    assert other["choices"] != nominal["choices"] or other["face_signs"] != nominal["face_signs"]
    pinned = replace(problem, expected_branch=nominal)
    with pytest.raises(ValueError, match="branch identity mismatch"):
        pinned.branch_check(candidate, parameters)


@pytest.mark.parametrize("mutation", ["float32", "integer", "nonfinite", "shape"])
def test_point_objective_refuses_noncontract_parameter_vectors(case, mutation):
    problem, control, parameters = case
    if mutation == "float32":
        changed = parameters.float()
    elif mutation == "integer":
        changed = parameters.long()
    elif mutation == "nonfinite":
        changed = parameters.clone()
        changed[0] = torch.nan
    else:
        changed = parameters[:-1]
    with pytest.raises(ValueError, match="finite CPU FP64 layout"):
        problem.objective(control, changed)
    with pytest.raises(ValueError, match="finite CPU FP64 layout"):
        problem.forecast(control, changed)


def test_point_objective_refuses_values_outside_detected_contract(case):
    problem, control, parameters = case
    for bad in (
        problem.frozen.analysis_config.detection_limit_dbz - 1,
        problem.frozen.nowcast_config.max_dbz,
    ):
        changed = parameters.clone()
        changed[0] = bad
        with pytest.raises(ValueError, match="remain detected observations"):
            problem.objective(control, changed)
        with pytest.raises(ValueError, match="remain detected observations"):
            problem.forecast(control, changed)


@pytest.mark.parametrize("mutation", [
    "missing", "censored", "std", "minimum_std", "bias", "outside",
    "values_grad", "std_grad", "quality_grad", "background_grad",
    "verification_grad", "boundary_grad", "boundary_shape", "boundary_nonfinite",
])
def test_point_problem_refuses_unsupported_observation_contract(case, mutation):
    problem, _, _ = case
    with pytest.raises(ValueError):
        if mutation == "missing":
            values = problem.observation_dbz.clone()
            values[1, 0] = torch.nan
            replace(problem, observation_dbz=values)
        elif mutation == "censored":
            values = problem.observation_dbz.clone()
            values[1, 0] = problem.frozen.analysis_config.detection_limit_dbz - 1
            replace(problem, observation_dbz=values)
        elif mutation == "std":
            replace(problem, observation_std_dbz=problem.observation_std_dbz[:-1])
        elif mutation == "minimum_std":
            replace(problem, observation_std_dbz=torch.full_like(problem.observation_std_dbz, 1e-12))
        elif mutation == "bias":
            frozen = replace(problem.frozen, analysis_config=replace(
                problem.frozen.analysis_config, observation_common_bias_std_dbz=0.2,
            ))
            replace(problem, frozen=frozen)
        elif mutation == "values_grad":
            replace(problem, observation_dbz=problem.observation_dbz.clone().requires_grad_())
        elif mutation == "std_grad":
            replace(problem, observation_std_dbz=problem.observation_std_dbz.clone().requires_grad_())
        elif mutation == "quality_grad":
            replace(problem, quality_weight=problem.quality_weight.clone().requires_grad_())
        elif mutation == "background_grad":
            replace(problem, background_dbz=problem.background_dbz.clone().requires_grad_())
        elif mutation == "verification_grad":
            replace(problem, verification_dbz=problem.verification_dbz.clone().requires_grad_())
        elif mutation == "boundary_grad":
            first = problem.future_boundary_echo[0]
            edges = first[0]
            changed_edges = (edges[0].clone().requires_grad_(), edges[1], edges[2], edges[3])
            changed_schedule = ((changed_edges, first[1]),) + problem.future_boundary_echo[1:]
            replace(problem, future_boundary_echo=changed_schedule)
        elif mutation in ("boundary_shape", "boundary_nonfinite"):
            first = problem.future_boundary_echo[0]
            edges = first[0]
            changed_edge = (edges[0][:-1] if mutation == "boundary_shape"
                            else edges[0].clone().fill_(float("nan")))
            changed_edges = (changed_edge, edges[1], edges[2], edges[3])
            changed_schedule = ((changed_edges, first[1]),) + problem.future_boundary_echo[1:]
            replace(problem, future_boundary_echo=changed_schedule)
        else:
            positions = problem.observation_coordinates.clone()
            positions[0, 0] = 0
            replace(problem, observation_coordinates=positions)
