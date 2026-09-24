"""Two-lead time variant of the existing 4x5 fully known research case."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from advar import transport as t, variational as v
from advar.fv_research_problem import FVResearchProblem
from advar.physics import dbz_to_echo, echo_to_dbz
from examples.weather_scenarios import fv_minmod_inverse_probe as probe

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'graphify-out/fv-root-cause-20260919'


def make_case() -> tuple[FVResearchProblem, FVResearchProblem, Tensor, Tensor]:
    """Return matched 1/2-lead problems, one archived control and parameters."""
    observations, frozen, future_echo, future_support = probe.make_spatial_case()
    spec = frozen.fv_transport
    if spec is None:
        raise ValueError('the 4x5 case requires FV transport')
    future_support = cast(t.BoundarySchedule, future_support)
    future_echo_two = future_echo * 2
    # The legacy fixture has four edges at each of the two RK stages, but its
    # unannotated tuple comprehension is inferred as variable-length tuples.
    future_support_two = future_support * 2
    next_config = replace(frozen.nowcast_config, horizon_minutes=2)
    observations_two, frozen_two = v.prepare_analysis(
        observations.dbz,
        nowcast_config=next_config,
        analysis_config=frozen.analysis_config,
        observation_std_dbz=observations.std_dbz,
        fv_transport=spec,
    )
    pattern = torch.linspace(-.2,.3,20,dtype=observations.dbz.dtype).reshape(4,5)
    parameters = torch.cat((observations.dbz.flatten(),observations.dbz.new_tensor([.02])))
    if not torch.equal(observations_two.dbz, observations.dbz):
        raise ValueError('two-lead preparation changed the fixed observations')
    nominal = json.loads((EVIDENCE/'minmod_middle_time_bias_final.json').read_text())
    control = torch.tensor(nominal['nominal_control'],dtype=parameters.dtype)
    coefficients = spec.coefficient_limits * parameters.new_tensor([.7,-.6,.5,.4,-.3])
    future_truth, _ = t.finite_volume_trajectory(
        dbz_to_echo(observations.dbz[-1],min_dbz=frozen.nowcast_config.min_dbz),
        torch.ones_like(observations.dbz[-1]),coefficients,parameters.new_tensor(.008),
        psi_basis=spec.psi_basis,leads=2,substeps_per_interval=spec.substeps_per_interval,
        interval_seconds=60.0*frozen.nowcast_config.interval_minutes,
        spacing_yx=spec.spacing_yx,boundary_echo=future_echo_two,
        boundary_support=future_support_two,reconstruction='minmod',
    )
    verification=(echo_to_dbz(future_truth[1:],min_dbz=frozen.nowcast_config.min_dbz)
                  +.1*pattern).detach()
    one=FVResearchProblem(observations,frozen,future_echo,future_support,pattern,
                          verification[0],probe.inspect_branches)
    two=FVResearchProblem(observations_two,frozen_two,future_echo_two,future_support_two,
                          pattern,verification,probe.inspect_branches,leads=2)
    return one,two,control,parameters
