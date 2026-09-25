"""One empty first point-observation time over a complete FV state."""
from __future__ import annotations

from dataclasses import replace

import torch

from examples.weather_scenarios.fv_point_centered_prior_case import (
    CenteredPointCase, make_case,
)


EMPTY_TIME = 0
INACTIVE_PARAMETERS = (0, 1, 2, 3)


def make_empty_time_case() -> CenteredPointCase:
    full = make_case()
    status = torch.zeros_like(full.problem.observation_dbz, dtype=torch.uint8)
    status[EMPTY_TIME, :] = 1
    observed = full.problem.observation_dbz.clone()
    observed[EMPTY_TIME, :] = full.problem.frozen.nowcast_config.min_dbz
    problem = replace(
        full.problem, observation_dbz=observed,
        observation_status=status, empty_observation_time=EMPTY_TIME,
    )
    parameters = torch.cat((observed.flatten(), full.parameters[-1:].clone()))
    case = replace(full, problem=problem, parameters=parameters)
    case.branch_check(case.control, case.parameters)
    return case
