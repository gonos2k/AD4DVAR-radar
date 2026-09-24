"""Analytic zero-flow checks for the bounded 18-lead FV research case."""

from dataclasses import replace

import pytest
import torch

from advar import variational as v
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.transport import bounded_fv_coefficients
from examples.weather_scenarios import fv_long_horizon_case as case_builder


@pytest.fixture(scope="module")
def long_horizon_case():
    return case_builder.make_case()


def test_long_horizon_times_layout_and_stage_count(long_horizon_case):
    problem, _, _ = long_horizon_case

    assert problem.leads == 18
    assert problem.layout["observation_times_seconds"] == (0.0, 600.0, 1200.0)
    assert problem.layout["forecast_times_seconds"] == tuple(
        range(1800, 12001, 600)
    )
    assert problem.layout["forecast_time_seconds"] == 12000.0
    assert problem.layout["substeps_per_interval"] == 90
    assert problem.layout["euler_stages"] == 3600
    assert problem.layout["observation_shape"][0] == 3
    assert problem.observations.detected_mask.all()
    assert problem.support["general_minmod_response_eligible"] is False


def test_long_horizon_forecast_matches_per_interval_analytic_echo_growth(
    long_horizon_case,
):
    problem, control, parameters = long_horizon_case

    # The last analysis frame is the known state at 1200 seconds. With zero
    # face fluxes, each ten-minute forecast interval applies exp(0.08) once.
    analyzed = v.analysis_trajectory(control, problem.contract(parameters))
    initial_echo = analyzed.frames_linear[-1]
    lead_numbers = torch.arange(1, problem.leads + 1, dtype=initial_echo.dtype)
    expected_echo = initial_echo.unsqueeze(0) * torch.exp(
        0.08 * lead_numbers
    ).reshape(-1, 1, 1)
    expected_dbz = echo_to_dbz(
        expected_echo, min_dbz=problem.frozen.nowcast_config.min_dbz
    )

    forecasts = problem.forecast(control, parameters)
    torch.testing.assert_close(forecasts, expected_dbz, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(forecasts, problem.verification, rtol=2e-12, atol=2e-12)


def test_long_horizon_scores_are_mean_of_per_lead_spatial_mse(long_horizon_case):
    problem, control, parameters = long_horizon_case

    forecasts = problem.forecast(control, parameters)
    per_lead_mse = (forecasts - problem.verification).square().flatten(1).mean(1)
    torch.testing.assert_close(
        problem.score(control, parameters), per_lead_mse.mean(), rtol=0, atol=0
    )

    # Unequal lead offsets distinguish averaging the 18 lead scores from a
    # score that accidentally selects one lead or weights leads by time.
    offsets = torch.arange(1, problem.leads + 1, dtype=forecasts.dtype).reshape(
        -1, 1, 1
    ) * 0.01
    perturbed = replace(problem, verification=problem.verification + offsets)
    perturbed_per_lead = (forecasts - perturbed.verification).square().flatten(1).mean(1)
    torch.testing.assert_close(
        perturbed.score(control, parameters),
        perturbed_per_lead.mean(),
        rtol=2e-15,
        atol=2e-18,
    )


def test_long_horizon_refuses_short_future_boundary_schedule(long_horizon_case):
    problem, _, _ = long_horizon_case

    with pytest.raises(ValueError, match="boundary schedule mismatch"):
        replace(problem, future_boundary_echo=problem.future_boundary_echo[:-1])


def test_ten_minute_interval_keeps_coefficient_box_cfl_with_ninety_substeps(
    long_horizon_case,
):
    problem, _, _ = long_horizon_case
    spec = problem.frozen.fv_transport
    assert spec is not None
    control = torch.zeros_like(spec.coefficient_limits)
    arguments = dict(
        psi_basis=spec.psi_basis,
        coefficient_limits=spec.coefficient_limits,
        spacing_yx=spec.spacing_yx,
        reconstruction=spec.reconstruction,
        max_courant=spec.max_courant,
    )
    with pytest.raises(ValueError, match="CFL domain"):
        bounded_fv_coefficients(control, dt_seconds=600.0 / 9, **arguments)
    torch.testing.assert_close(
        bounded_fv_coefficients(control, dt_seconds=600.0 / 90, **arguments),
        torch.zeros_like(control), rtol=0, atol=0,
    )


@pytest.mark.parametrize(
    "verification_times",
    [
        lambda times: tuple(reversed(times)),
        lambda times: tuple(time + 60.0 for time in times),
    ],
)
def test_long_horizon_refuses_verification_time_misalignment(
    long_horizon_case, verification_times
):
    problem, _, _ = long_horizon_case

    with pytest.raises(ValueError, match="verification times"):
        replace(
            problem,
            verification_times_seconds=verification_times(
                problem.verification_times_seconds
            ),
        )


def test_explicit_verification_times_are_bound_into_problem_identity(
    long_horizon_case,
):
    problem, _, _ = long_horizon_case

    implicit_times = replace(problem, verification_times_seconds=None)
    assert implicit_times.identity != problem.identity


def test_growth_control_is_integrated_once_per_ten_minute_interval(long_horizon_case):
    problem, control, parameters = long_horizon_case

    initial_echo = v.analysis_trajectory(
        control, problem.contract(parameters)
    ).frames_linear[-1]
    forecast_echo = dbz_to_echo(
        problem.forecast(control, parameters),
        min_dbz=problem.frozen.nowcast_config.min_dbz,
    )
    for lead in (1, 9, problem.leads):
        torch.testing.assert_close(
            forecast_echo[lead - 1],
            initial_echo * torch.exp(initial_echo.new_tensor(0.08 * lead)),
            rtol=2e-12,
            atol=2e-12,
        )


def test_bounded_nonzero_flow_long_horizon_forward_matches_synthetic_truth():
    problem, control, parameters = case_builder.make_flow_case()

    flow_start = problem.layout["field_controls"]
    flow_stop = flow_start + problem.layout["flow_controls"]
    assert torch.count_nonzero(control[flow_start:flow_stop]) > 0
    torch.testing.assert_close(
        problem.forecast(control, parameters), problem.verification,
        rtol=2e-12, atol=2e-12,
    )


def test_nonzero_flow_uses_future_boundary_traces_in_time_order():
    problem, control, parameters = case_builder.make_flow_case()
    substeps = problem.layout["substeps_per_interval"]
    varied = tuple(
        tuple(
            tuple(edge * (1.0 + 0.01 * (step // substeps)) for edge in edges)
            for edges in stages
        )
        for step, stages in enumerate(problem.future_boundary_echo)
    )
    forward = replace(problem, future_boundary_echo=varied)
    reversed_time = replace(problem, future_boundary_echo=tuple(reversed(varied)))
    with torch.no_grad():
        in_order = forward.forecast(control, parameters)
        reversed_output = reversed_time.forecast(control, parameters)
    assert torch.isfinite(in_order).all()
    assert torch.isfinite(reversed_output).all()
    assert float((in_order[-1] - reversed_output[-1]).abs().max()) > 1e-6
