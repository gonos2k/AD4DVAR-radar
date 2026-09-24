"""Reproduce one bounded fixed-control FV check with genuinely missing point rows.

This probe verifies the observation contract and local derivatives. It does
not solve a new analysis or certify an implicit stationary response.
"""
from __future__ import annotations

from dataclasses import replace
import json

import torch

from examples.weather_scenarios.fv_point_research_case import make_case


def run() -> dict[str, object]:
    problem, control, original_parameters = make_case()
    status = torch.zeros_like(problem.observation_dbz, dtype=torch.uint8)
    status[0, 1] = 1
    status[1, 0] = 1
    status[2, 2] = 1
    stored_values = problem.observation_dbz.clone()
    stored_values[status == 1] = problem.frozen.nowcast_config.min_dbz

    correlation = torch.eye(status.shape[1], dtype=torch.float64)
    correlation[0, 1] = correlation[1, 0] = 0.4
    correlation[1, 2] = correlation[2, 1] = -0.2
    masked = replace(
        problem,
        observation_dbz=stored_values,
        observation_status=status,
        observation_correlation=correlation,
    )
    parameters = torch.cat((stored_values.flatten(), original_parameters[-1:]))
    missing_direction = torch.zeros_like(parameters)
    missing_direction[:-1].reshape_as(status)[status == 1] = 1.0
    control_gradient = torch.func.grad(masked.objective, argnums=0)
    objective_gradient = torch.func.grad(masked.objective, argnums=1)(control, parameters)
    mixed_missing = torch.func.jvp(
        lambda p: control_gradient(control, p),
        (parameters,), (missing_direction,),
    )[1]
    forecast_missing = torch.func.jvp(
        lambda p: masked.forecast(control, p),
        (parameters,), (missing_direction,),
    )[1]

    detected_direction = torch.zeros_like(parameters)
    detected_direction[0] = 1.0
    step = 1e-4
    positive = masked.objective(control, parameters + step * detected_direction)
    negative = masked.objective(control, parameters - step * detected_direction)
    central_difference = (positive - negative) / (2 * step)
    derivative = torch.dot(objective_gradient, detected_direction)
    branch, _ = masked.branch_check(control, parameters)
    return {
        "scope": "fixed-control 4x5 FV point-observation contract; no GN, refinement or adjoint",
        "problem_identity": masked.identity,
        "layout": masked.layout,
        "detected_per_time": [int((status[t] == 0).sum()) for t in range(3)],
        "missing_parameter_gradient_max_abs": float(
            objective_gradient[:-1].reshape_as(status)[status == 1].abs().max()
        ),
        "missing_mixed_jvp_max_abs": float(mixed_missing.abs().max()),
        "missing_forecast_jvp_max_abs": float(forecast_missing.abs().max()),
        "detected_direction_gradient": float(derivative),
        "detected_direction_central_difference": float(central_difference),
        "detected_direction_absolute_difference": float(abs(derivative - central_difference)),
        "objective": float(masked.objective(control, parameters)),
        "score": float(masked.score(control, parameters)),
        "euler_stages": branch["euler_stages"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
