"""Analytic zero-flow 10-minute/three-hour FV time-contract fixture.

This exercises regular lead and boundary scheduling at an exactly known
forward solution. Zero face flux makes its strict classical response
ineligible; it is not a GN, adjoint, or nonzero-flow forecast validation.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import Tensor

from advar import variational as v
from advar.fv_research_problem import FVResearchProblem
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.transport import BoundaryEdges, BoundarySchedule, finite_volume_trajectory
from examples.weather_scenarios import fv_minmod_inverse_probe as probe


INTERVAL_MINUTES = 10
FORECAST_LEADS = 18
LOG_GROWTH_PER_INTERVAL = 0.08


def make_case() -> tuple[FVResearchProblem, Tensor, Tensor]:
    """Return an exactly growing 4×5 state with fixed, known zero-flow edges."""
    seed_observations, seed_frozen, _, _ = probe.make_spatial_case()
    seed_transport = seed_frozen.fv_transport
    if seed_transport is None:
        raise ValueError("the 4x5 seed needs FV transport")
    config = replace(
        seed_frozen.nowcast_config,
        interval_minutes=INTERVAL_MINUTES,
        horizon_minutes=INTERVAL_MINUTES * FORECAST_LEADS,
    )
    initial_echo = dbz_to_echo(
        seed_observations.dbz[0], min_dbz=config.min_dbz,
    )
    edges: BoundaryEdges = (
        initial_echo[:, 0], initial_echo[:, -1], initial_echo[0], initial_echo[-1],
    )
    support_edges: BoundaryEdges = (
        torch.ones_like(edges[0]), torch.ones_like(edges[1]),
        torch.ones_like(edges[2]), torch.ones_like(edges[3]),
    )
    echo_step: BoundarySchedule = ((edges, edges),)
    support_step: BoundarySchedule = ((support_edges, support_edges),)
    # Preserve the seed case's 6 2/3-second substep and its coefficient-box
    # CFL bound when one observation interval grows from 1 to 10 minutes.
    substeps = seed_transport.substeps_per_interval * INTERVAL_MINUTES
    transport = replace(
        seed_transport,
        substeps_per_interval=substeps,
        boundary_echo=echo_step * (2 * substeps),
        boundary_support=support_step * (2 * substeps),
    )
    growth = initial_echo.new_tensor(LOG_GROWTH_PER_INTERVAL)
    observed_echo = torch.stack([
        initial_echo * torch.exp(growth * time_index)
        for time_index in range(3)
    ])
    observed_dbz = echo_to_dbz(observed_echo, min_dbz=config.min_dbz).detach()
    observations, frozen = v.prepare_analysis(
        observed_dbz,
        nowcast_config=config,
        analysis_config=seed_frozen.analysis_config,
        observation_std_dbz=0.1,
        fv_transport=transport,
    )
    verification = echo_to_dbz(
        torch.stack([
            initial_echo * torch.exp(growth * (2 + lead))
            for lead in range(1, FORECAST_LEADS + 1)
        ]),
        min_dbz=config.min_dbz,
    ).detach()
    pattern = torch.linspace(-0.2, 0.3, initial_echo.numel(), dtype=initial_echo.dtype).reshape_as(initial_echo)
    future_echo = echo_step * (FORECAST_LEADS * substeps)
    future_support = support_step * (FORECAST_LEADS * substeps)
    problem = FVResearchProblem(
        observations=observations,
        frozen=frozen,
        future_boundary_echo=future_echo,
        future_boundary_support=future_support,
        pattern=pattern,
        verification=verification,
        trace_branches=probe.inspect_branches,
        leads=FORECAST_LEADS,
        verification_times_seconds=tuple(
            (2 + lead) * INTERVAL_MINUTES * 60.0
            for lead in range(1, FORECAST_LEADS + 1)
        ),
    )
    control = torch.zeros(problem.layout["controls"], dtype=initial_echo.dtype)
    control[-1] = math.atanh(LOG_GROWTH_PER_INTERVAL / config.max_log_growth_per_step)
    parameters = torch.cat((observed_dbz.flatten(), control.new_zeros(1)))
    return problem, control, parameters


def make_flow_case() -> tuple[FVResearchProblem, Tensor, Tensor]:
    """Generate a matching nonzero-flow 3-hour truth and fixed-control input."""
    zero_problem, control, _ = make_case()
    spec = zero_problem.frozen.fv_transport
    if spec is None:
        raise ValueError("the long-horizon case needs FV transport")
    initial_echo = dbz_to_echo(
        zero_problem.observations.dbz[0],
        min_dbz=zero_problem.frozen.nowcast_config.min_dbz,
    )
    fractions = control.new_tensor([0.7, -0.6, 0.5, 0.4, -0.3])
    flow_coefficients = spec.coefficient_limits * fractions
    complete_echo = spec.boundary_echo + zero_problem.future_boundary_echo
    complete_support = spec.boundary_support + zero_problem.future_boundary_support
    with torch.no_grad():
        truth, _ = finite_volume_trajectory(
            initial_echo,
            torch.ones_like(initial_echo),
            flow_coefficients,
            control.new_tensor(LOG_GROWTH_PER_INTERVAL),
            psi_basis=spec.psi_basis,
            leads=2 + FORECAST_LEADS,
            substeps_per_interval=spec.substeps_per_interval,
            interval_seconds=INTERVAL_MINUTES * 60.0,
            spacing_yx=spec.spacing_yx,
            boundary_echo=complete_echo,
            boundary_support=complete_support,
            reconstruction=spec.reconstruction,
        )
        truth_dbz = echo_to_dbz(
            truth, min_dbz=zero_problem.frozen.nowcast_config.min_dbz,
        ).detach()
    observations, frozen = v.prepare_analysis(
        truth_dbz[:3],
        nowcast_config=zero_problem.frozen.nowcast_config,
        analysis_config=zero_problem.frozen.analysis_config,
        observation_std_dbz=0.1,
        fv_transport=spec,
    )
    problem = replace(
        zero_problem,
        observations=observations,
        frozen=frozen,
        verification=truth_dbz[3:],
    )
    controls = control.clone()
    controls[-6:-1] = torch.atanh(fractions)
    parameters = torch.cat((truth_dbz[:3].flatten(), controls.new_zeros(1)))
    return problem, controls, parameters
