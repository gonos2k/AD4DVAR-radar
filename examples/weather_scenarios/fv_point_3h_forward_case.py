"""Fixed off-grid point observations over the existing 3-hour FV trajectory."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

from advar.fv_point_research_problem import FVPointResearchProblem
from advar.fv_point_sampler import point_dbz_bilinear
from examples.weather_scenarios import fv_long_horizon_case as long_case
from examples.weather_scenarios import fv_point_response_preflight as point_preflight


ROOT = Path(__file__).resolve().parents[2]


def make_case() -> tuple[FVPointResearchProblem, Tensor, Tensor, Tensor]:
    """Return one fixed point profile and its same-operator terminal truth."""
    long_problem, control, _ = long_case.make_flow_case()
    point_template, _, _, _ = point_preflight.fixed_problem()
    values = point_dbz_bilinear(
        long_problem.observations.dbz, point_template.observation_coordinates,
    ).detach()
    frozen = long_problem.frozen
    background = frozen.initial_background_dbz.detach().clone()
    verification = long_problem.verification[-1].detach().clone()
    problem = FVPointResearchProblem(
        frozen=frozen,
        observation_coordinates=point_template.observation_coordinates,
        observation_dbz=values,
        observation_std_dbz=point_template.observation_std_dbz,
        quality_weight=point_template.quality_weight,
        background_dbz=background,
        background_pattern=point_template.background_pattern,
        verification_dbz=verification,
        future_boundary_echo=long_problem.future_boundary_echo,
        future_boundary_support=long_problem.future_boundary_support,
        trace_branches=point_template.trace_branches,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observation_correlation=point_template.observation_correlation,
    )
    parameters = torch.cat((values.flatten(), control.new_zeros(1)))
    return problem, control, parameters, verification
