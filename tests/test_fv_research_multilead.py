"""Focused multi-interval checks for the fixed-support minmod research problem."""

from dataclasses import replace

import pytest
import torch

from advar.transport import BoundaryEdges, BoundarySchedule, finite_volume_trajectory
from examples.weather_scenarios import fv_multilead_research_case as case_builder


@pytest.fixture(scope="module")
def multilead_case():
    """Use the shared bounded inverse fixture rather than rebuilding it here."""
    return case_builder.make_case()


def test_multilead_layout_forecast_and_score_are_per_lead_mean(multilead_case):
    one, problem, control, parameters = multilead_case

    layout = problem.layout
    assert problem.leads == 2
    assert layout["forecast_times_seconds"] == (180.0, 240.0)
    assert layout["forecast_time_seconds"] == 240.0
    assert layout["euler_stages"] == 72

    forecasts = problem.forecast(control, parameters)
    assert forecasts.shape == (2, 4, 5)
    torch.testing.assert_close(forecasts[0], one.forecast(control, parameters), rtol=0, atol=0)
    per_lead_mse = (forecasts - problem.verification).square().flatten(1).mean(dim=1)
    expected = per_lead_mse.mean()
    torch.testing.assert_close(problem.score(control, parameters), expected, rtol=0, atol=0)

    branch, _ = problem.branch_check(control, parameters)
    assert branch["euler_stages"] == 72
    if problem.expected_branch is not None:
        assert branch["choices"] == problem.expected_branch["choices"]
        assert branch["face_signs"] == problem.expected_branch["face_signs"]


def test_multilead_score_control_jvp_matches_stable_central_differences(multilead_case):
    _, problem, control, parameters = multilead_case
    direction = torch.zeros_like(control)
    direction[0] = 1.0

    nominal, _ = problem.branch_check(control, parameters)
    derivative = torch.func.jvp(
        lambda value: problem.score(value, parameters),
        (control,),
        (direction,),
    )[1]

    for step in (1e-3, 5e-4):
        plus = control + step * direction
        minus = control - step * direction
        plus_branch, _ = problem.branch_check(plus, parameters)
        minus_branch, _ = problem.branch_check(minus, parameters)
        assert _branch_signature(plus_branch) == _branch_signature(nominal), (
            f"positive h={step:g} left the nominal branch"
        )
        assert _branch_signature(minus_branch) == _branch_signature(nominal), (
            f"negative h={step:g} left the nominal branch"
        )
        finite_difference = (problem.score(plus, parameters) - problem.score(minus, parameters)) / (2 * step)
        torch.testing.assert_close(derivative, finite_difference, rtol=2e-4, atol=1e-10)


def _branch_signature(branch):
    return branch["euler_stages"], branch["choices"], branch["face_signs"]


@pytest.mark.parametrize(
    "mutation",
    ["horizon", "verification", "future_schedule"],
)
def test_multilead_contract_rejects_horizon_shape_and_schedule_mismatches(
    multilead_case, mutation
):
    _, problem, _, _ = multilead_case
    with pytest.raises(ValueError):
        if mutation == "horizon":
            replace(problem, frozen=replace(
                problem.frozen,
                nowcast_config=replace(problem.frozen.nowcast_config, horizon_minutes=1),
            ))
        elif mutation == "verification":
            replace(problem, verification=problem.verification[0])
        else:
            replace(problem, future_boundary_echo=problem.future_boundary_echo[:-1])


@pytest.mark.parametrize("interval_growth", [-0.008, 0.0, 0.008])
def test_uniform_zero_face_trajectory_applies_signed_growth_once_per_interval(interval_growth):
    dtype = torch.float64
    initial = torch.full((4, 5), 3.0, dtype=dtype)
    basis = torch.arange(5, dtype=dtype)[:, None].expand(5, 6).unsqueeze(0)
    edges: BoundaryEdges = (initial[:, 0], initial[:, -1], initial[0], initial[-1])
    support_edges: BoundaryEdges = (
        torch.ones_like(edges[0]), torch.ones_like(edges[1]),
        torch.ones_like(edges[2]), torch.ones_like(edges[3]),
    )
    boundary: BoundarySchedule = ((edges, edges),) * 18
    support: BoundarySchedule = ((support_edges, support_edges),) * 18

    frames, _ = finite_volume_trajectory(
        initial,
        torch.ones_like(initial),
        initial.new_zeros(1),
        initial.new_tensor(interval_growth),
        psi_basis=basis,
        leads=2,
        substeps_per_interval=9,
        interval_seconds=60.0,
        spacing_yx=(10.0, 10.0),
        boundary_echo=boundary,
        boundary_support=support,
        reconstruction="minmod",
    )

    torch.testing.assert_close(frames[0], initial, rtol=0, atol=0)
    torch.testing.assert_close(
        frames[1], initial * torch.exp(initial.new_tensor(interval_growth)), rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        frames[2], initial * torch.exp(initial.new_tensor(2 * interval_growth)), rtol=0, atol=1e-12
    )
