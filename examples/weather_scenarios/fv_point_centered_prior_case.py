"""Construct one smooth, fixed-prior-mean point FV response fixture."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any

import torch
from torch import Tensor

from advar import variational as v
from advar.fv_point_research_problem import FVPointResearchProblem
from advar.fv_point_sampler import point_dbz_bilinear
from advar.physics import echo_to_dbz
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios import fv_point_core_strict_root_probe as strict


def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _branch_signature(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("choices", "face_signs")}


class CenteredBranchRefusal(ValueError):
    """A declared local minmod selector or margin disqualification."""


@dataclass(frozen=True)
class CenteredPointCase:
    problem: FVPointResearchProblem
    control: Tensor
    parameters: Tensor
    direction: Tensor
    dynamics_prior_mean: Tensor
    nominal_branch: dict[str, Any]

    def objective(self, control: Tensor, parameters: Tensor) -> Tensor:
        dynamics = control[-self.dynamics_prior_mean.numel():]
        centered = dynamics - self.dynamics_prior_mean
        return (self.problem.objective(control, parameters)
                - 0.5 * torch.dot(dynamics, dynamics)
                + 0.5 * torch.dot(centered, centered))

    def score(self, control: Tensor, parameters: Tensor) -> Tensor:
        return self.problem.score(control, parameters)

    def branch_check(self, control: Tensor, parameters: Tensor) -> tuple[dict[str, Any], str]:
        try:
            branch, scope = strict._core_branch(self.problem, control, parameters)
        except ValueError as error:
            if not str(error).startswith((
                "minmod joint oracle left its strict smooth branch",
                "core-strict stage or branch resolvability refused",
            )):
                raise
            raise CenteredBranchRefusal(str(error)) from error
        if (branch["euler_stages"] != 54
                or branch["minimum_scaled_slope_margin"] <= 1e-4
                or branch["minimum_scaled_face_flux_margin"] <= 1e-4
                or _branch_signature(branch) != _branch_signature(self.nominal_branch)):
            raise CenteredBranchRefusal("centered point endpoint left its qualified nominal branch")
        return branch, scope

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "problem": self.problem.identity,
            "objective": "point_objective_minus_half_dynamics_norm_plus_half_centered_norm_v1",
            "dynamics_prior_mean_sha256": tensor_sha(self.dynamics_prior_mean),
            "control_sha256": tensor_sha(self.control),
            "parameters_sha256": tensor_sha(self.parameters),
            "direction_sha256": tensor_sha(self.direction),
        }


def make_case() -> CenteredPointCase:
    base, warm, original_p, _ = preflight.fixed_problem()
    if (base.frozen.neural_prior_std_dbz is not None
            or base.frozen.neural_prior_valid_mask is not None):
        raise ValueError("centered point construction requires the unchanged identity control prior")
    control = torch.zeros_like(warm)
    control[-6:] = warm[-6:]
    mean = control[-6:].detach().clone()
    contract = base.contract(original_p)
    frames = v.analysis_trajectory(control, contract).frames_linear
    predicted = point_dbz_bilinear(
        echo_to_dbz(frames, min_dbz=base.frozen.nowcast_config.min_dbz),
        base.observation_coordinates,
    ).detach()
    verification = (base.forecast(control, original_p)
                    + 0.1 * base.background_pattern).detach()
    problem = replace(base, observation_dbz=predicted,
                      verification_dbz=verification)
    parameters = torch.cat((predicted.flatten(), original_p[-1:].clone()))
    direction = torch.zeros_like(parameters)
    direction[4:8] = 1.0
    branch, _ = strict._core_branch(problem, control, parameters)
    if (branch["euler_stages"] != 54
            or branch["minimum_scaled_slope_margin"] <= 1e-4
            or branch["minimum_scaled_face_flux_margin"] <= 1e-4):
        raise ValueError("centered point nominal branch is not response qualified")
    return CenteredPointCase(
        problem=problem, control=control, parameters=parameters,
        direction=direction, dynamics_prior_mean=mean,
        nominal_branch=branch,
    )
