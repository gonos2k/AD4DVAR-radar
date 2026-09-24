"""One fixed four-point, off-grid dBZ observation case for FV research."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from advar import variational as v
from advar.fv_point_research_problem import FVPointResearchProblem
from advar.fv_point_sampler import point_dbz_bilinear
from advar.physics import echo_to_dbz
from advar.transport import BoundarySchedule
from examples.weather_scenarios import fv_minmod_inverse_probe as probe


EVIDENCE = Path(__file__).resolve().parents[2] / "graphify-out/fv-root-cause-20260919"


def make_case() -> tuple[FVPointResearchProblem, Tensor, Tensor]:
    """Return a fixed external-background problem, warm control and parameters.

    The synthetic point observations are generated once from the warm-control
    trajectory and then detached. They never define the state-grid background.
    """
    _, frozen, future_echo, future_support = probe.make_spatial_case()
    future_support = cast(BoundarySchedule, future_support)
    saved = json.loads((EVIDENCE / "minmod_middle_time_bias_final.json").read_text())
    control = torch.tensor(saved["nominal_control"], dtype=torch.float64)
    coordinates = control.new_tensor([
        [0.35, 0.65], [1.25, 2.40], [2.50, 3.10], [1.70, 1.45],
    ])
    background = frozen.initial_background_dbz.detach().clone()
    pattern = torch.linspace(-0.2, 0.3, 20, dtype=control.dtype).reshape(4, 5)
    theta = control.new_tensor(0.02)
    contract = replace(frozen, initial_background_dbz=background + theta * pattern)
    trajectory = v.analysis_trajectory(control, contract)
    model_dbz = echo_to_dbz(trajectory.frames_linear, min_dbz=frozen.nowcast_config.min_dbz)
    values = (
        point_dbz_bilinear(model_dbz, coordinates)
        + torch.linspace(-0.02, 0.02, 12, dtype=control.dtype).reshape(3, 4)
    ).detach()
    forecast = v.forecast_fv_analysis(
        control, contract, leads=1, boundary_start_interval=2,
        boundary_echo=future_echo, boundary_support=future_support,
    ).frames_linear[-1]
    verification = (
        echo_to_dbz(forecast, min_dbz=frozen.nowcast_config.min_dbz)
        + 0.1 * pattern
    ).detach()
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    problem = FVPointResearchProblem(
        frozen=frozen,
        observation_coordinates=coordinates,
        observation_dbz=values,
        observation_std_dbz=torch.full_like(values, 0.1),
        quality_weight=torch.linspace(0.5, 1.0, values.numel(), dtype=values.dtype).reshape_as(values),
        background_dbz=background,
        background_pattern=pattern,
        verification_dbz=verification,
        future_boundary_echo=future_echo,
        future_boundary_support=future_support,
        trace_branches=probe.inspect_branches,
        source_sha256=source_sha256,
    )
    parameters = torch.cat((values.flatten(), theta.reshape(1)))
    return problem, control, parameters
