"""One fixed missing-point profile over the centered-prior FV case."""
from __future__ import annotations

from dataclasses import replace

import torch

from examples.weather_scenarios.fv_point_centered_prior_case import (
    CenteredPointCase, make_case,
)


MISSING_PARAMETER_INDEX = 5


def make_missing_case() -> CenteredPointCase:
    full = make_case()
    status = torch.zeros_like(full.problem.observation_dbz, dtype=torch.uint8)
    status[1, 1] = 1
    observed = full.problem.observation_dbz.clone()
    observed[1, 1] = full.problem.frozen.nowcast_config.min_dbz
    problem = replace(full.problem, observation_dbz=observed,
                      observation_status=status)
    parameters = torch.cat((observed.flatten(), full.parameters[-1:].clone()))
    direction = full.direction.clone()
    direction[MISSING_PARAMETER_INDEX] = 0.0
    case = replace(full, problem=problem, parameters=parameters,
                   direction=direction)
    case.branch_check(case.control, case.parameters)
    return case
