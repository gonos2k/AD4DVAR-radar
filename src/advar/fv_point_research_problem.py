"""Fixed off-grid dBZ-point observations for a bounded FV research problem.

The observation values live on fixed interior points, while the initial
background, state and FV boundaries remain on the model grid. This is not a
radar-footprint average, general covariance model, or operational interface.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from . import variational as v
from .fv_point_sampler import point_dbz_bilinear
from .fv_research_problem import _fingerprint
from .physics import echo_to_dbz
from .transport import BoundarySchedule


@dataclass(frozen=True)
class FVPointResearchProblem:
    """One-lead, all-detected, independent-error off-grid FV research binding.

    ``parameters`` are the three observation vectors followed by one
    background-pattern coefficient. Point observations never define the
    state-grid initial background. The caller supplies a strict branch tracer.
    """

    frozen: v.FrozenOuterState
    observation_coordinates: Tensor
    observation_dbz: Tensor
    observation_std_dbz: Tensor
    quality_weight: Tensor
    background_dbz: Tensor
    background_pattern: Tensor
    verification_dbz: Tensor
    future_boundary_echo: BoundarySchedule
    future_boundary_support: BoundarySchedule
    trace_branches: Callable[[Callable[[], Tensor]], dict[str, Any]]
    source_sha256: str
    expected_branch: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        frozen = self.frozen
        spec = frozen.fv_transport
        shape = tuple(frozen.initial_background_dbz.shape)
        if (self.observation_coordinates.ndim != 2
                or self.observation_coordinates.shape[1] != 2
                or self.observation_coordinates.shape[0] == 0
                or self.observation_coordinates.requires_grad):
            raise ValueError("point FV coordinates must be fixed [N, 2] geometry")
        count = self.observation_coordinates.shape[0]
        if (spec is None or spec.reconstruction != "minmod"
                or len(shape) != 2 or min(shape) < 3
                or frozen.nowcast_config.forecast_steps != 1
                or frozen.initial_background_dbz.requires_grad
                or not bool(frozen.initial_support_mask.all())
                or frozen.neural_prior_dependency is not None
                or frozen.grid_time_contract is not None
                or frozen.analysis_config.observation_common_bias_std_dbz != 0
                or frozen.observation_whitener.mode is not None):
            raise ValueError("point FV research requires one-lead full-support minmod with diagonal errors")
        if (self.observation_dbz.shape != (3, count)
                or self.observation_dbz.dtype != torch.float64
                or self.observation_dbz.device.type != "cpu"
                or self.observation_dbz.requires_grad
                or not bool(torch.isfinite(self.observation_dbz).all())
                or not bool((self.observation_dbz > frozen.analysis_config.detection_limit_dbz).all())
                or not bool((self.observation_dbz < frozen.nowcast_config.max_dbz).all())):
            raise ValueError("point observations must be three detected FP64 vectors")
        for name, value in (("std", self.observation_std_dbz),
                            ("quality", self.quality_weight)):
            if (value.shape != self.observation_dbz.shape
                    or value.dtype != torch.float64 or value.device.type != "cpu"
                    or value.requires_grad
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"point observation {name} has incompatible layout")
        if (not bool((self.observation_std_dbz >= frozen.analysis_config.minimum_observation_std_dbz).all())
                or not bool(((self.quality_weight > 0) & (self.quality_weight <= 1)).all())):
            raise ValueError("point observation std/quality are outside their fixed domain")
        for name, value in (("background", self.background_dbz),
                            ("pattern", self.background_pattern),
                            ("verification", self.verification_dbz)):
            if (tuple(value.shape) != shape or value.dtype != torch.float64
                    or value.requires_grad
                    or value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
                raise ValueError(f"point FV {name} must be a finite state-grid field")
        point_dbz_bilinear(self.background_dbz, self.observation_coordinates)
        if len(self.source_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.source_sha256):
            raise ValueError("point observation source identity must be a SHA-256 digest")
        for echoes, supports, steps in (
            (spec.boundary_echo, spec.boundary_support, 2 * spec.substeps_per_interval),
            (self.future_boundary_echo, self.future_boundary_support, spec.substeps_per_interval),
        ):
            v._clone_fv_boundary_schedule(
                echoes, name="point FV echo boundary", steps=steps,
                reference=self.background_dbz,
            )
            v._clone_fv_boundary_schedule(
                supports, name="point FV support boundary", steps=steps,
                reference=self.background_dbz,
            )
            if any(any(edge.requires_grad for stage in stages for edge in stage)
                   for stages in echoes + supports):
                raise ValueError("point FV boundaries must be fixed tensors")
            if any(not all(bool(torch.all(edge == 1)) for stage in stages for edge in stage)
                   for stages in supports):
                raise ValueError("point FV requires fully known boundaries")

    @property
    def layout(self) -> dict[str, Any]:
        spec = self.frozen.fv_transport
        if spec is None:
            raise ValueError("point FV transport is required")
        observations = self.observation_dbz.numel()
        controls = self.frozen.active_field_index.numel() + spec.coefficient_limits.numel() + 1
        interval = 60.0 * self.frozen.nowcast_config.interval_minutes
        return {
            "state_shape": tuple(self.frozen.initial_background_dbz.shape),
            "observation_shape": tuple(self.observation_dbz.shape),
            "controls": controls,
            "parameters": observations + 1,
            "theta_index": observations,
            "observation_times_seconds": (0.0, interval, 2 * interval),
            "forecast_time_seconds": 3 * interval,
            "euler_stages": 6 * spec.substeps_per_interval,
        }

    @property
    def support(self) -> dict[str, Any]:
        return {
            "observation_operator": "point_dbz_bilinear",
            "observation_units": "dBZ point value; interpolate dBZ after state echo-to-dBZ conversion",
            "coordinate_convention": "fixed (row, column) cell-center indices; strict interior 2x2 stencils",
            "observation_errors": "independent diagonal std/quality in point-observation space",
            "observation_masks": "all valid and detected; missing/censored/QC unsupported",
            "initial_background": "fixed exogenous state-grid field + theta * fixed pattern",
            "state_and_boundary_support": "fully known",
            "branch_policy": "pointwise strict minmod; caller pins expected_branch for endpoint comparison",
            "response_validation": "not_performed",
            "general_minmod_response_eligible": False,
        }

    @property
    def identity(self) -> dict[str, str]:
        return {
            "fixed_problem_sha256": _fingerprint({
                "frozen": self.frozen,
                "observation_coordinates": self.observation_coordinates,
                "observation_operator": "point_dbz_bilinear_v1",
                "observation_dbz": self.observation_dbz,
                "observation_std_dbz": self.observation_std_dbz,
                "quality_weight": self.quality_weight,
                "background_dbz": self.background_dbz,
                "background_pattern": self.background_pattern,
                "verification_dbz": self.verification_dbz,
                "future_boundary_echo": self.future_boundary_echo,
                "future_boundary_support": self.future_boundary_support,
                "source_sha256": self.source_sha256,
                "layout": self.layout,
                "support": self.support,
            }),
            "scope": "fixed point-observation identity; caller also binds code, control, parameters and branch",
        }

    def contract(self, parameters: Tensor) -> v.FrozenOuterState:
        if (not isinstance(parameters, Tensor)
                or parameters.shape != (self.layout["parameters"],)
                or parameters.dtype != torch.float64
                or parameters.device.type != "cpu"
                or not bool(torch.isfinite(parameters).all())):
            raise ValueError("point FV parameters must match the finite CPU FP64 layout")
        observed = parameters[:-1]
        if not bool(((observed > self.frozen.analysis_config.detection_limit_dbz)
                     & (observed < self.frozen.nowcast_config.max_dbz)).all()):
            raise ValueError("point FV parameters must remain detected observations")
        return replace(self.frozen,
                       initial_background_dbz=self.background_dbz + parameters[-1] * self.background_pattern)

    def objective(self, control: Tensor, parameters: Tensor) -> Tensor:
        contract = self.contract(parameters)
        observed = parameters[:-1].reshape_as(self.observation_dbz)
        predicted_echo = v.analysis_trajectory(control, contract).frames_linear
        predicted_dbz = echo_to_dbz(predicted_echo, min_dbz=contract.nowcast_config.min_dbz)
        sampled_dbz = point_dbz_bilinear(predicted_dbz, self.observation_coordinates)
        residual = self.quality_weight.sqrt() * (sampled_dbz - observed) / self.observation_std_dbz
        prior = v._control_prior_residual(control, contract)
        return (v._pseudo_huber_cost(residual, contract.analysis_config.pseudo_huber_delta).sum()
                + 0.5 * torch.dot(prior, prior)
                + v._field_smoothness_prior_cost(control, contract))

    def forecast(self, control: Tensor, parameters: Tensor) -> Tensor:
        frames = v.forecast_fv_analysis(
            control, self.contract(parameters), leads=1, boundary_start_interval=2,
            boundary_echo=self.future_boundary_echo,
            boundary_support=self.future_boundary_support,
        ).frames_linear
        return echo_to_dbz(frames[-1], min_dbz=self.frozen.nowcast_config.min_dbz)

    def score(self, control: Tensor, parameters: Tensor) -> Tensor:
        return (self.forecast(control, parameters) - self.verification_dbz).square().mean()

    def branch_check(self, control: Tensor, parameters: Tensor) -> tuple[dict[str, Any], str]:
        layout = self.layout
        if (control.shape != (layout["controls"],) or parameters.shape != (layout["parameters"],)
                or control.dtype != torch.float64 or parameters.dtype != torch.float64
                or control.device.type != "cpu" or parameters.device.type != "cpu"
                or not bool(torch.isfinite(control).all())
                or not bool(torch.isfinite(parameters).all())):
            raise ValueError("point FV control/parameter layout or finiteness mismatch")
        cfg, analysis = self.frozen.nowcast_config, self.frozen.analysis_config
        background = self.contract(parameters).initial_background_dbz
        offset = (background - cfg.min_dbz) / analysis.echo_transform_scale_dbz
        margin = 64 * torch.finfo(background.dtype).eps * (
            (background.abs() + abs(cfg.min_dbz)) / analysis.echo_transform_scale_dbz
            + analysis.transform_epsilon
        )
        if not bool(((offset - analysis.transform_epsilon > margin) & (background < cfg.max_dbz)).all()):
            raise ValueError("point FV background is outside its smooth transform branch")
        branch = self.trace_branches(lambda: self.forecast(control, parameters))
        if branch["euler_stages"] != layout["euler_stages"]:
            raise ValueError("point FV RK stage count mismatch")
        if self.expected_branch is not None and (
            branch["choices"] != self.expected_branch["choices"]
            or branch["face_signs"] != self.expected_branch["face_signs"]
        ):
            raise ValueError("point FV branch identity mismatch")
        return branch, "fixed off-grid dBZ points, independent errors, known state/boundary; pointwise branch only"

    def functions(self):
        return self.objective, self.score, self.branch_check
