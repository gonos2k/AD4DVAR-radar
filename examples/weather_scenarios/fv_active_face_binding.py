"""Fixed two-hole q_y[2,0]=0 chart binding; no stationarity/response claim."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from advar import variational as v
from advar.fv_research_problem import FVResearchProblem
from advar.physics import echo_to_dbz
from advar.transport import bounded_fv_coefficients, face_volume_fluxes
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart


class FVActiveFaceBinding:
    """Bind 25 tangent controls to the unchanged 26-control objective at qy[2,0]=0."""

    field_count = 20
    pivot_index = 1
    face_row = 2
    face_column = 0

    def __init__(self, problem: FVResearchProblem) -> None:
        original = problem.frozen
        spec = original.fv_transport
        missing = torch.nonzero(~problem.observations.valid_mask, as_tuple=False).tolist()
        if (spec is None or spec.coefficient_limits.numel() != 5
                or original.active_field_index.numel() != self.field_count
                or problem.observations.dbz.shape != (3, 4, 5)
                or missing != [[1, 1, 2], [2, 2, 3]] or problem.leads != 1
                or original.neural_prior_std_dbz is not None
                or original.grid_time_contract is not None):
            raise ValueError("active-face binding requires the fixed 20+5+1 two-hole profile")
        self.problem = problem
        self.chart = FVFaceFluxCoordinateChart.from_basis(
            field_count=self.field_count, growth_count=1,
            coefficient_limits=spec.coefficient_limits, psi_basis=spec.psi_basis,
            axis="y", face=(self.face_row, self.face_column), pivot_index=self.pivot_index,
        )
        weights = self.chart.weights
        keep = [index for index in range(5) if index != self.pivot_index]
        ratio = weights[keep] / weights[self.pivot_index]
        projected = spec.psi_basis[keep] - ratio[:, None, None] * spec.psi_basis[self.pivot_index]
        if not torch.equal(projected[:, self.face_row, self.face_column],
                           projected[:, self.face_row, self.face_column + 1]):
            raise ValueError("projected modes do not give a structural zero on the selected face")
        limits = spec.coefficient_limits[keep].clone()
        candidate = replace(spec, psi_basis=projected, coefficient_limits=limits)
        dt = problem.frozen.nowcast_config.interval_minutes * 60.0 / spec.substeps_per_interval
        bounded_fv_coefficients(
            limits.new_zeros(4), psi_basis=projected, coefficient_limits=limits,
            dt_seconds=dt, spacing_yx=spec.spacing_yx,
            reconstruction=spec.reconstruction, max_courant=spec.max_courant,
        )
        self.transport = v._freeze_fv_analysis_transport(
            candidate, reference=original.input_frames_dbz,
        )
        self.control_count = self.field_count + 4 + 1

    @property
    def support(self) -> dict[str, Any]:
        return {"face": ("y", self.face_row, self.face_column),
                "pivot_flow_index": self.pivot_index,
                "flow_controls": 4, "structural_face_zero": True,
                "branch_policy": "original strict oracle still rejects zero face; no active-face branch qualification",
                "full_root_claim": False, "response_validation": "not_performed"}

    def lift(self, tangent: Tensor) -> Tensor:
        """Lift [field, nonpivot latents, growth], refusing the original pivot boundary."""
        if (not isinstance(tangent, Tensor) or tangent.shape != (self.control_count,)
                or tangent.dtype != torch.float64 or tangent.device.type != "cpu"):
            raise ValueError("tangent control must be CPU FP64 length 25")
        pivot = self.field_count + self.pivot_index
        face_coordinates = torch.cat((tangent[:pivot], tangent.new_zeros(1), tangent[pivot:]))
        return self.chart.from_face_coordinates(face_coordinates)

    def tangent_coordinates(self, original_control: Tensor) -> Tensor:
        coordinates = self.chart.to_face_coordinates(original_control)
        pivot = self.field_count + self.pivot_index
        return torch.cat((coordinates[:pivot], coordinates[pivot + 1:]))

    def _fixed_state(self, parameters: Tensor):
        observations = replace(
            self.problem.observations,
            dbz=self.problem._observation_values(parameters),
        )
        original_frozen = self.problem.contract(parameters)
        reduced_frozen = replace(original_frozen, fv_transport=self.transport)
        return observations, original_frozen, reduced_frozen

    def objective(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        original_control = self.lift(tangent)
        observations, original_frozen, reduced_frozen = self._fixed_state(parameters)
        residual = v.whitened_observation_residual(tangent, observations, reduced_frozen)
        return v._robust_objective_from_residual(
            original_control, residual, observations, original_frozen,
        )

    def forecast(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        self.lift(tangent)
        _, _, reduced_frozen = self._fixed_state(parameters)
        trajectory = v.forecast_fv_analysis(
            tangent, reduced_frozen, leads=self.problem.leads, boundary_start_interval=2,
            boundary_echo=self.problem.future_boundary_echo,
            boundary_support=self.problem.future_boundary_support,
        )
        frames = trajectory.frames_linear
        selected = frames[-1] if self.problem.leads == 1 else frames[1:]
        return echo_to_dbz(selected, min_dbz=reduced_frozen.nowcast_config.min_dbz)

    def score(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        return (self.forecast(tangent, parameters) - self.problem.verification).square().mean()

    def face_fluxes(self, tangent: Tensor) -> tuple[Tensor, Tensor]:
        if tangent.shape != (self.control_count,):
            raise ValueError("tangent control must have length 25")
        self.lift(tangent)
        coefficients = bounded_fv_coefficients(
            tangent[self.field_count:self.field_count + 4],
            psi_basis=self.transport.psi_basis,
            coefficient_limits=self.transport.coefficient_limits,
            dt_seconds=(self.problem.frozen.nowcast_config.interval_minutes * 60.0
                        / self.transport.substeps_per_interval),
            spacing_yx=self.transport.spacing_yx,
            reconstruction=self.transport.reconstruction,
            max_courant=self.transport.max_courant,
        )
        psi = torch.einsum("k,kij->ij", coefficients, self.transport.psi_basis)
        return face_volume_fluxes(psi)
