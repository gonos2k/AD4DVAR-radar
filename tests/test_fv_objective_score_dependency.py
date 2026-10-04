"""Dependency wiring for the real point objective/score methods; no FV run."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor

from advar import variational as v
from advar.fv_point_research_problem import FVPointResearchProblem
from advar.nowcast import ForecastMetadata, NowcastConfig, RadarState
from advar.physics import RemapCell


@dataclass(frozen=True)
class _UnusedBaselineMetadata:
    """Fingerprintable baseline placeholder; objective/score never read it."""

    fixture_id: str = "toy-dependency-wiring"


def _boundary_schedule(steps: int, echo_value: float, *, support: bool = False):
    edge_value = 1.0 if support else echo_value
    vertical = torch.full((3,), edge_value, dtype=torch.float64)
    horizontal = torch.full((3,), edge_value, dtype=torch.float64)
    edges = (vertical, vertical.clone(), horizontal, horizontal.clone())
    stage = (edges, edges)
    return tuple(stage for _ in range(steps))


def _toy_problem(monkeypatch) -> tuple[FVPointResearchProblem, Tensor, Tensor]:
    analysis_config = v.AnalysisConfig(pseudo_huber_delta=1.25)
    nowcast_config = NowcastConfig(interval_minutes=1, horizon_minutes=1)
    initial_background = torch.full((3, 3), 12.0, dtype=torch.float64)
    analysis_boundary_echo = _boundary_schedule(2, 0.0)
    analysis_boundary_support = _boundary_schedule(2, 1.0, support=True)
    transport = v.FVAnalysisTransport(
        psi_basis=torch.zeros((1, 4, 4), dtype=torch.float64),
        coefficient_limits=torch.ones(1, dtype=torch.float64),
        substeps_per_interval=1,
        spacing_yx=(1.0, 1.0),
        boundary_echo=analysis_boundary_echo,
        boundary_support=analysis_boundary_support,
    )
    frozen = v.FrozenOuterState(
        input_frames_dbz=torch.full((3, 3, 3), 12.0, dtype=torch.float64),
        background_frames_dbz=None,
        initial_background_dbz=initial_background,
        initial_support_mask=torch.ones((3, 3), dtype=torch.bool),
        active_field_index=torch.empty(0, dtype=torch.long),
        causal_only_mask=torch.zeros((3, 3), dtype=torch.bool),
        causal_seed_mask=torch.zeros((3, 3), dtype=torch.bool),
        detected_masks=torch.ones((3, 1), dtype=torch.bool),
        observed_mask=torch.ones((3, 3), dtype=torch.bool),
        background_mask=torch.ones((3, 3), dtype=torch.bool),
        background_age_minutes=None,
        baseline_state=RadarState(
            echo_linear=torch.ones((3, 3), dtype=torch.float64),
            displacement_yx=torch.zeros(2, dtype=torch.float64),
            log_growth_per_step=torch.zeros((), dtype=torch.float64),
        ),
        # Objective/score do not consume baseline forecast metadata.
        baseline_metadata=cast(ForecastMetadata, cast(object, _UnusedBaselineMetadata())),
        baseline_frames_dbz=initial_background,
        observation_whitener=v.FrozenObservationWhitener(None, None, True),
        irls_sqrt_weight=torch.ones((3, 1), dtype=torch.float64),
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
        grid_time_contract=None,
        motion_limits_yx=torch.ones(2, dtype=torch.float64),
        amplitude_displacement_offsets_yx=((0, 0),),
        analysis_remap_cells=(RemapCell(0, 0), RemapCell(1, 1)),
        smooth_edge_left_index=torch.empty(0, dtype=torch.long),
        smooth_edge_right_index=torch.empty(0, dtype=torch.long),
        smooth_edge_physical_weight=torch.empty(0, dtype=torch.float64),
        fv_transport=transport,
    )
    grid = torch.arange(9, dtype=torch.float64).reshape(3, 3)

    def analysis_stub(control: Tensor, contract: v.FrozenOuterState):
        # The analysis trajectory has three fixed observation frames; its
        # polynomial transport stub does not consume verification, future
        # boundaries, or the terminal forecast horizon.
        level = (2.0 + 0.2 * control[0] - 0.1 * control[1]
                 + 0.01 * contract.initial_background_dbz[0, 0])
        frames = torch.stack(tuple(level + 0.01 * grid + 0.02 * index for index in range(3)))
        return SimpleNamespace(frames_linear=frames)

    def forecast_stub(
        control: Tensor,
        contract: v.FrozenOuterState,
        *,
        leads: int,
        boundary_start_interval: int,
        boundary_echo,
        boundary_support,
    ):
        assert boundary_start_interval == 2
        assert len(boundary_echo) == len(boundary_support) == leads
        boundary_level = boundary_echo[-1][0][0].mean()
        level = (2.5 + 0.12 * control[0] + 0.08 * control[1]
                 + 0.01 * contract.initial_background_dbz[0, 0]
                 + 0.2 * boundary_level)
        frames = torch.stack(tuple(level + 0.01 * grid + 0.1 * index
                                   for index in range(leads)))
        return SimpleNamespace(frames_linear=frames)

    monkeypatch.setattr(v, "analysis_trajectory", analysis_stub)
    monkeypatch.setattr(v, "forecast_fv_analysis", forecast_stub)
    problem = FVPointResearchProblem(
        frozen=frozen,
        observation_coordinates=torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        observation_dbz=torch.full((3, 1), 12.0, dtype=torch.float64),
        observation_std_dbz=torch.full((3, 1), 0.5, dtype=torch.float64),
        quality_weight=torch.ones((3, 1), dtype=torch.float64),
        background_dbz=initial_background,
        background_pattern=0.1 * grid,
        verification_dbz=torch.full((3, 3), 11.0, dtype=torch.float64),
        future_boundary_echo=_boundary_schedule(1, 0.0),
        future_boundary_support=_boundary_schedule(1, 1.0, support=True),
        trace_branches=lambda _run: {"euler_stages": 0},
        source_sha256="a" * 64,
    )
    control = torch.tensor([0.1, -0.2], dtype=torch.float64)
    parameters = torch.tensor([11.8, 12.2, 12.5, 0.2], dtype=torch.float64)
    return problem, control, parameters


def _objective_metrics(problem, control: Tensor, parameters: Tensor):
    objective = problem.objective(control, parameters)
    gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    direction = torch.tensor([0.3, -0.4], dtype=torch.float64)
    hvp = torch.func.jvp(
        lambda value: torch.func.grad(problem.objective, argnums=0)(value, parameters),
        (control,),
        (direction,),
    )[1]
    return objective, gradient, hvp


@pytest.mark.parametrize("change", ["verification", "future_boundary", "leads"])
def test_actual_objective_is_independent_of_score_only_inputs(monkeypatch, change: str):
    problem, control, parameters = _toy_problem(monkeypatch)
    if change == "verification":
        changed = replace(problem, verification_dbz=problem.verification_dbz + 0.25)
    elif change == "future_boundary":
        changed = replace(problem, future_boundary_echo=_boundary_schedule(1, 0.75))
    else:
        longer_config = replace(problem.frozen.nowcast_config, horizon_minutes=2)
        longer_frozen = replace(problem.frozen, nowcast_config=longer_config)
        assert longer_frozen.analysis_config is problem.frozen.analysis_config
        assert longer_frozen.fv_transport is problem.frozen.fv_transport
        assert longer_frozen.nowcast_config.forecast_steps != problem.frozen.nowcast_config.forecast_steps
        changed = replace(
            problem,
            frozen=longer_frozen,
            future_boundary_echo=_boundary_schedule(2, 0.0),
            future_boundary_support=_boundary_schedule(2, 1.0, support=True),
        )

    assert changed.identity != problem.identity
    original_jgh = _objective_metrics(problem, control, parameters)
    changed_jgh = _objective_metrics(changed, control, parameters)
    assert all(torch.equal(left, right) for left, right in zip(original_jgh, changed_jgh))

    original_score = problem.score(control, parameters)
    changed_score = changed.score(control, parameters)
    original_score_gradient = torch.func.grad(problem.score, argnums=0)(control, parameters)
    changed_score_gradient = torch.func.grad(changed.score, argnums=0)(control, parameters)
    assert not torch.equal(original_score, changed_score)
    assert not torch.equal(original_score_gradient, changed_score_gradient)
