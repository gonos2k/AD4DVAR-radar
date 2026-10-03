"""Compose two exact FV face constraints over the fixed point-problem chart."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from advar import variational as v
from advar.fv_point_research_problem import FVPointResearchProblem
from advar.transport import bounded_fv_coefficients, face_volume_fluxes
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart
from examples.weather_scenarios.fv_point_active_face_binding import FVPointActiveFaceBinding


class FVPointPairedFaceBinding:
    """Bind 24 tangent controls to the unchanged 26-control point objective."""

    field_count = 20
    growth_count = 1
    face_row = 3
    face_column = 0
    pivot_index = 0

    def __init__(self, problem: FVPointResearchProblem) -> None:
        self.problem = problem
        self.first_binding = FVPointActiveFaceBinding(problem)
        first_transport = self.first_binding.transport
        self.second_chart = FVFaceFluxCoordinateChart.from_basis(
            field_count=self.field_count,
            growth_count=self.growth_count,
            coefficient_limits=first_transport.coefficient_limits,
            psi_basis=first_transport.psi_basis,
            axis="y",
            face=(self.face_row, self.face_column),
            pivot_index=self.pivot_index,
        )
        weights = self.second_chart.weights
        keep = [index for index in range(first_transport.coefficient_limits.numel())
                if index != self.pivot_index]
        projected = (first_transport.psi_basis[keep]
                     - (weights[keep] / weights[self.pivot_index])[:, None, None]
                     * first_transport.psi_basis[self.pivot_index])
        selected_qy = -(projected[:, self.face_row, self.face_column + 1]
                        - projected[:, self.face_row, self.face_column])
        if not torch.equal(selected_qy, torch.zeros_like(selected_qy)):
            raise ValueError("projected modes do not structurally zero q_y[3,0]")
        selected_qx = projected[:, 4, 4] - projected[:, 3, 4]
        if not torch.equal(selected_qx, torch.zeros_like(selected_qx)):
            raise ValueError("second projection broke the structural q_x[3,4] zero")
        projected_rank = int(torch.linalg.matrix_rank(projected.flatten(1)))
        if projected_rank != 3:
            raise ValueError("projected flow basis must have rank three")
        limits = first_transport.coefficient_limits[keep].clone()
        candidate = replace(first_transport, psi_basis=projected.clone(), coefficient_limits=limits)
        dt = (problem.frozen.nowcast_config.interval_minutes * 60.0
              / first_transport.substeps_per_interval)
        bounded_fv_coefficients(
            limits.new_zeros(3), psi_basis=candidate.psi_basis,
            coefficient_limits=limits, dt_seconds=dt, spacing_yx=candidate.spacing_yx,
            reconstruction=candidate.reconstruction, max_courant=candidate.max_courant,
        )
        self.transport = v._freeze_fv_analysis_transport(
            candidate, reference=problem.frozen.input_frames_dbz,
        )
        reduced_frozen = replace(self.first_binding.reduced_problem.frozen,
                                  fv_transport=self.transport)
        self.reduced_problem = replace(self.first_binding.reduced_problem, frozen=reduced_frozen)
        self.control_count = self.field_count + 3 + self.growth_count
        self._projected_mode_rank = projected_rank

    @property
    def support(self) -> dict[str, Any]:
        return {
            "faces": (("x", 3, 4), ("y", self.face_row, self.face_column)),
            "chart_pivot_flow_indices": (self.first_binding.pivot_index, self.pivot_index),
            "original_flow_indices": (self.first_binding.pivot_index,
                                      self.first_binding.pivot_index + 1),
            "flow_controls": 3,
            "projected_mode_rank": self._projected_mode_rank,
            "structural_face_zeros": (("x", 3, 4), ("y", self.face_row, self.face_column)),
            "branch_policy": "original strict point oracle; no active-face qualification",
            "original_objective_prior": "all 26 zero-centered control terms retained",
            "full_root_claim": False,
            "response_validation": "not_performed",
        }

    def lift(self, tangent: Tensor) -> Tensor:
        if (not isinstance(tangent, Tensor) or tangent.shape != (self.control_count,)
                or tangent.dtype != torch.float64 or tangent.device.type != "cpu"):
            raise ValueError("tangent must be CPU FP64 length 24")
        pivot = self.field_count + self.pivot_index
        first_tangent = self.second_chart.from_face_coordinates(
            torch.cat((tangent[:pivot], tangent.new_zeros(1), tangent[pivot:]))
        )
        full_control = self.first_binding.lift(first_tangent)
        # Inverse bounds alone do not ensure that a tiny recovered pivot can
        # survive the chart's stricter forward round-trip guard.
        self.second_chart.to_face_coordinates(first_tangent)
        self.first_binding.chart.to_face_coordinates(full_control)
        return full_control

    def tangent_coordinates(self, control: Tensor) -> Tensor:
        first_tangent = self.first_binding.tangent_coordinates(control)
        coordinates = self.second_chart.to_face_coordinates(first_tangent)
        pivot = self.field_count + self.pivot_index
        return torch.cat((coordinates[:pivot], coordinates[pivot + 1:]))

    def objective(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        full_control = self.lift(tangent)
        return (self.reduced_problem.objective(tangent, parameters)
                + 0.5 * (full_control[20].square() + full_control[21].square()))

    def forecast(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        self.lift(tangent)
        return self.reduced_problem.forecast(tangent, parameters)

    def score(self, tangent: Tensor, parameters: Tensor) -> Tensor:
        self.lift(tangent)
        return self.reduced_problem.score(tangent, parameters)

    def face_fluxes(self, tangent: Tensor) -> tuple[Tensor, Tensor]:
        self.lift(tangent)
        coefficients = bounded_fv_coefficients(
            tangent[self.field_count:self.field_count + 3],
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
