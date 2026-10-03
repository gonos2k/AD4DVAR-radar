"""Fixed point-problem binding to the structural q_x[3,4]=0 face."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from advar import variational as v
from advar.fv_point_research_problem import FVPointResearchProblem
from advar.transport import bounded_fv_coefficients, face_volume_fluxes
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart


class FVPointActiveFaceBinding:
    """Bind 25 tangent controls to the unchanged 26-control point objective."""

    field_count = 20
    pivot_index = 0
    face_row = 3
    face_column = 4

    def __init__(self, problem: FVPointResearchProblem) -> None:
        original = problem.frozen
        spec = original.fv_transport
        correlation = problem.observation_correlation
        expected_correlation = torch.eye(4, dtype=torch.float64)
        expected_correlation[0, 2] = expected_correlation[2, 0] = 0.3
        expected_correlation[1, 3] = expected_correlation[3, 1] = -0.2
        if (spec is None or spec.coefficient_limits.shape != (5,)
                or original.initial_background_dbz.shape != (4, 5)
                or original.active_field_index.numel() != self.field_count
                or problem.layout["controls"] != 26 or problem.layout["parameters"] != 13
                or problem.layout["euler_stages"] != 54
                or problem.observation_coordinates.shape != (4, 2)
                or correlation is None or not torch.equal(correlation, expected_correlation)
                or problem.observation_status is not None or not bool(problem._detected_mask.all())
                or original.neural_prior_std_dbz is not None
                or original.neural_prior_dependency is not None
                or original.grid_time_contract is not None):
            raise ValueError("point active-face binding requires the original full-valid correlated 4x5 profile")
        self.problem = problem
        self.chart = FVFaceFluxCoordinateChart.from_basis(
            field_count=self.field_count, growth_count=1,
            coefficient_limits=spec.coefficient_limits, psi_basis=spec.psi_basis,
            axis="x", face=(self.face_row, self.face_column), pivot_index=self.pivot_index,
        )
        weights = self.chart.weights
        keep = [index for index in range(5) if index != self.pivot_index]
        ratio = weights[keep] / weights[self.pivot_index]
        projected = (spec.psi_basis[keep]
                     - ratio[:, None, None] * spec.psi_basis[self.pivot_index])
        if not torch.equal(projected[:, self.face_row, self.face_column],
                           projected[:, self.face_row + 1, self.face_column]):
            raise ValueError("projected modes do not structurally zero q_x[3,4]")
        projected_rank = int(torch.linalg.matrix_rank(projected.flatten(1)))
        if projected_rank != 4:
            raise ValueError("projected flow basis must have rank four")
        limits = spec.coefficient_limits[keep].clone()
        candidate = replace(spec, psi_basis=projected.clone(), coefficient_limits=limits)
        dt = original.nowcast_config.interval_minutes * 60.0 / spec.substeps_per_interval
        bounded_fv_coefficients(
            limits.new_zeros(4), psi_basis=candidate.psi_basis,
            coefficient_limits=limits, dt_seconds=dt, spacing_yx=spec.spacing_yx,
            reconstruction=spec.reconstruction, max_courant=spec.max_courant,
        )
        projected_transport = v._freeze_fv_analysis_transport(
            candidate, reference=original.input_frames_dbz,
        )
        self.transport = projected_transport
        self.reduced_problem = replace(problem, frozen=replace(original, fv_transport=projected_transport))
        self.control_count = self.field_count + 4 + 1

    @property
    def support(self) -> dict[str, Any]:
        return {"face": ("x", self.face_row, self.face_column),
                "pivot_flow_index": self.pivot_index,
                "flow_controls": 4, "projected_mode_rank": 4,
                "structural_face_zero": True,
                "branch_policy": "original strict point oracle rejects the selected zero face; no active-face qualification",
                "original_objective_prior": "all 26 zero-centered control terms retained",
                "full_root_claim": False, "response_validation": "not_performed"}

    def lift(self, tangent: Tensor) -> Tensor:
        if (not isinstance(tangent, Tensor) or tangent.shape != (self.control_count,)
                or tangent.dtype != torch.float64 or tangent.device.type != "cpu"):
            raise ValueError("tangent must be CPU FP64 length 25")
        pivot = self.field_count + self.pivot_index
        coordinates = torch.cat((tangent[:pivot], tangent.new_zeros(1), tangent[pivot:]))
        return self.chart.from_face_coordinates(coordinates)

    def tangent_coordinates(self, control: Tensor) -> Tensor:
        coordinates = self.chart.to_face_coordinates(control)
        pivot = self.field_count + self.pivot_index
        return torch.cat((coordinates[:pivot], coordinates[pivot + 1:]))

    def objective(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        original_control = self.lift(tangent)
        pivot = self.field_count + self.pivot_index
        return (self.reduced_problem.objective(tangent, parameters)
                + 0.5 * original_control[pivot].square())

    def forecast(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        self.lift(tangent)
        return self.reduced_problem.forecast(tangent, parameters)

    def score(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        self.lift(tangent)
        return self.reduced_problem.score(tangent, parameters)

    def face_fluxes(self, tangent: Tensor) -> tuple[Tensor, Tensor]:
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
