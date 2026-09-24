"""Shared fixed-support FV research problem, separate from numerical policy.

This binds the existing collocated dBZ observation operator and frozen error
transform. It is not an arbitrary observation-operator or general minmod API.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
from typing import Any

import torch
from torch import Tensor

from . import variational as v
from .physics import echo_to_dbz
from .transport import BoundarySchedule


def _fingerprint(value: object) -> str:
    def encode(item: object) -> object:
        if isinstance(item, Tensor):
            return {'shape':list(item.shape),'dtype':str(item.dtype),
                    'sha256':hashlib.sha256(item.detach().contiguous().cpu().numpy().tobytes()).hexdigest()}
        if is_dataclass(item) and not isinstance(item, type):
            return {field.name:getattr(item,field.name) for field in fields(item)}
        raise TypeError(f'unsupported fixed input type: {type(item).__name__}')
    # Fixed diagnostic metadata may contain NaN; tensor bytes and model inputs
    # remain explicit. This digest is an identity, not a finiteness certificate.
    payload=json.dumps(value,default=encode,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FVResearchProblem:
    """Thin binding of an existing model/observation definition and fixed score.

    Inputs are fixed and callbacks must be deterministic/nonmutating. Numerical
    tolerances and solvers remain in their existing modules. The supplied tracer
    owns the physical branch policy and may impose additional restrictions.
    """

    observations: v.AnalysisObservations
    frozen: v.FrozenOuterState
    future_boundary_echo: BoundarySchedule
    future_boundary_support: BoundarySchedule
    pattern: Tensor
    verification: Tensor
    trace_branches: Callable[[Callable[[], Tensor]], dict[str, Any]]
    expected_branch: Mapping[str, Any] | None = None
    observation_operator: str = 'collocated_dbz'
    leads: int = 1

    def __post_init__(self) -> None:
        obs, frozen = self.observations, self.frozen
        spec=frozen.fv_transport
        shape=tuple(frozen.initial_background_dbz.shape)
        if (type(self.leads) is not int or self.leads <= 0
                or self.leads != frozen.nowcast_config.forecast_steps):
            raise ValueError('research forecast leads must match the configured horizon')
        if (self.observation_operator != 'collocated_dbz' or spec is None
                or spec.reconstruction != 'minmod'
                or obs.dbz.dtype != torch.float64 or obs.dbz.device.type != 'cpu'
                or obs.dbz.ndim != 3 or obs.dbz.shape[0] != 3
                or tuple(obs.dbz.shape[1:]) != shape or len(shape) != 2 or min(shape) < 3
                or not bool(obs.detected_mask.all()) or not bool(obs.valid_mask.all())
                or not bool(frozen.initial_support_mask.all())
                or frozen.neural_prior_dependency is not None or frozen.grid_time_contract is not None):
            raise ValueError('research minmod requires the CPU FP64 collocated full-support contract')
        for value, expected_shape in ((self.pattern,shape),
                                      (self.verification,shape if self.leads == 1 else (self.leads,*shape))):
            if (tuple(value.shape) != expected_shape or value.dtype != obs.dbz.dtype
                    or value.device != obs.dbz.device or not bool(torch.isfinite(value).all())):
                raise ValueError('research minmod pattern/verification mismatch')
        for echoes,supports,steps in (
            (spec.boundary_echo,spec.boundary_support,(obs.dbz.shape[0]-1)*spec.substeps_per_interval),
            (self.future_boundary_echo,self.future_boundary_support,self.leads*spec.substeps_per_interval),
        ):
            if len(echoes)!=steps or len(supports)!=steps:
                raise ValueError('research minmod boundary schedule mismatch')
            for stages in supports:
                if len(stages)!=2 or any(len(edges)!=4 for edges in stages):
                    raise ValueError('research minmod boundary stage layout mismatch')
                for edges in stages:
                    if not all(bool(torch.all(edge==1)) for edge in edges):
                        raise ValueError('research minmod requires fully known boundary support')

    @property
    def layout(self) -> dict[str, Any]:
        spec=self.frozen.fv_transport
        if spec is None:
            raise ValueError('FV transport is required')
        field_count=self.frozen.active_field_index.numel()
        flow_count=spec.coefficient_limits.numel()
        observation_count=self.observations.dbz.numel()
        observation_frames=self.observations.dbz.shape[0]
        interval=60.0*self.frozen.nowcast_config.interval_minutes
        forecast_times=tuple((observation_frames-1+lead)*interval for lead in range(1,self.leads+1))
        return {'state_shape':tuple(self.frozen.initial_background_dbz.shape),
                'observation_shape':tuple(self.observations.dbz.shape),
                'field_controls':field_count,'flow_controls':flow_count,'growth_controls':1,
                'controls':field_count+flow_count+1,
                'observation_values':observation_count,'background_parameters':1,
                'parameters':observation_count+1,'theta_index':observation_count,
                'observation_times_seconds':tuple(i*interval for i in range(observation_frames)),
                'forecast_time_seconds':forecast_times[-1],
                'forecast_times_seconds':forecast_times,
                'forecast_leads':self.leads,
                'substeps_per_interval':spec.substeps_per_interval,
                'euler_stages':(observation_frames-1+self.leads)*spec.substeps_per_interval*2}

    @property
    def support(self) -> dict[str, Any]:
        return {'observation_operator':self.observation_operator,
                'observation_errors':'existing FrozenObservationWhitener',
                'observation_masks':'all valid and detected',
                'state_and_boundary_support':'fully known',
                'time_schedule':('three regular observations; one subsequent forecast interval'
                                 if self.leads == 1 else
                                 f'three regular observations; {self.leads} subsequent forecast intervals'),
                'background_dependency':'first observation + theta * fixed pattern',
                'branch_policy':'caller tracer; pointwise check required',
                'response_validation':'not_performed',
                'general_minmod_response_eligible':False}

    @property
    def identity(self) -> dict[str, str]:
        return {'fixed_problem_sha256':_fingerprint({
                    'observations':self.observations,'frozen':self.frozen,
                    'future_echo':self.future_boundary_echo,'future_support':self.future_boundary_support,
                    'pattern':self.pattern,'verification':self.verification,
                    'layout':self.layout,'support':self.support}),
                'scope':'fixed-input identity; caller must also bind code, parameters, control and branch for cache reuse'}

    def contract(self, parameters: Tensor) -> v.FrozenOuterState:
        observed=parameters[:-1].reshape_as(self.observations.dbz)
        return replace(self.frozen,initial_background_dbz=observed[0]+parameters[-1]*self.pattern)

    def objective(self, control: Tensor, parameters: Tensor) -> Tensor:
        observations=replace(self.observations,dbz=parameters[:-1].reshape_as(self.observations.dbz))
        return v.robust_objective(control,observations,self.contract(parameters))

    def forecast(self, control: Tensor, parameters: Tensor) -> Tensor:
        frames=v.forecast_fv_analysis(
            control,self.contract(parameters),leads=self.leads,boundary_start_interval=self.observations.dbz.shape[0]-1,
            boundary_echo=self.future_boundary_echo,boundary_support=self.future_boundary_support,
        ).frames_linear
        return echo_to_dbz(frames[-1] if self.leads == 1 else frames[1:],
                           min_dbz=self.frozen.nowcast_config.min_dbz)

    def score(self, control: Tensor, parameters: Tensor) -> Tensor:
        return (self.forecast(control,parameters)-self.verification).square().mean()

    def branch_check(self, control: Tensor, parameters: Tensor) -> tuple[dict[str, Any], str]:
        layout=self.layout
        if control.shape != (layout['controls'],) or parameters.shape != (layout['parameters'],):
            raise ValueError('research minmod control/parameter layout mismatch')
        background=self.contract(parameters).initial_background_dbz
        cfg=self.frozen.nowcast_config
        ac=self.frozen.analysis_config
        offset=(background-cfg.min_dbz)/ac.echo_transform_scale_dbz
        margin=64*torch.finfo(background.dtype).eps*((background.abs()+abs(cfg.min_dbz))/ac.echo_transform_scale_dbz+ac.transform_epsilon)
        if not bool(((offset-ac.transform_epsilon>margin)&(background<cfg.max_dbz)).all()):
            raise ValueError('research minmod background is outside its smooth branch')
        if not bool(((parameters[:-1]>ac.detection_limit_dbz)&(parameters[:-1]<cfg.max_dbz)).all()):
            raise ValueError('research minmod requires fixed detected observations')
        branch=self.trace_branches(lambda:self.forecast(control,parameters))
        if branch['euler_stages']!=layout['euler_stages']:
            raise ValueError('research minmod RK stage count mismatch')
        if self.expected_branch is not None and (
            branch['choices']!=self.expected_branch['choices']
            or branch['face_signs']!=self.expected_branch['face_signs']
        ):
            raise ValueError('research minmod branch identity mismatch')
        return branch,'strict collocated full-support minmod; fixed errors/boundaries; caller branch policy; no finite-path/general eligibility'

    def functions(self):
        return self.objective,self.score,self.branch_check
