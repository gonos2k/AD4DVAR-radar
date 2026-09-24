"""Explicit 8x10 scaled minmod research fixture; not a general FV API."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import importlib.util
from typing import cast

import torch
from advar import transport as t, variational as v
from advar.nowcast import NowcastConfig
from advar.physics import echo_to_dbz


@dataclass(frozen=True)
class FVScaledResearchCase:
    """The fixed 86-control / 241-parameter conditional research problem."""

    observations: v.AnalysisObservations
    frozen: v.FrozenOuterState
    future_boundary_echo: t.BoundarySchedule
    future_boundary_support: t.BoundarySchedule
    parameters: torch.Tensor
    pattern: torch.Tensor
    verification: torch.Tensor
    truth_initial_echo: torch.Tensor
    definition: dict[str, object]


_HEIGHT = 8
_WIDTH = 10
_SUBSTEPS = 18
_LEADS = 3
_LIMITS = (0.11, 0.08, 0.07, 0.04, 0.03)
_TRUTH_MULTIPLIERS = (0.7, -0.6, 0.5, 0.4, -0.3)
_CONTROL_SIZE = _HEIGHT * _WIDTH + len(_LIMITS) + 1
_PARAMETER_SIZE = 3 * _HEIGHT * _WIDTH + 1
_RK_STAGE_COUNT = 2 * _LEADS * _SUBSTEPS


def _load_inverse_probe():
    path = Path(__file__).with_name("fv_minmod_inverse_probe.py")
    spec = importlib.util.spec_from_file_location("fv_scaled_inverse_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case() -> FVScaledResearchCase:
    """Build the approved half-increment grid and three-interval truth case."""
    dtype = torch.float64
    yv, xv = torch.meshgrid(
        torch.arange(_HEIGHT + 1, dtype=dtype) / 2,
        torch.arange(_WIDTH + 1, dtype=dtype) / 2,
        indexing="ij",
    )
    basis = torch.stack(
        (yv, xv, xv * yv, 0.5 * (xv.square() - yv.square()), xv.square() * yv)
    )
    cell_y, cell_x = yv[:-1, :-1], xv[:-1, :-1]
    initial_echo = 20 * (
        1.5
        + 0.04 * cell_y
        + 0.03 * cell_x
        + 0.01 * cell_x * cell_y
        + 0.001 * cell_x.square()
        + 0.002 * cell_y.square()
        + 0.0003 * cell_x * cell_y.square()
    )
    limits = initial_echo.new_tensor(_LIMITS)
    truth_coefficients = limits * initial_echo.new_tensor(_TRUTH_MULTIPLIERS)
    edges = (initial_echo[:, 0], initial_echo[:, -1], initial_echo[0], initial_echo[-1])
    ones = cast(t.BoundaryEdges, tuple(torch.ones_like(edge) for edge in edges))
    schedule = ((edges, edges),) * (_LEADS * _SUBSTEPS)
    support_schedule = ((ones, ones),) * (_LEADS * _SUBSTEPS)
    transport = v.FVAnalysisTransport(
        psi_basis=basis,
        coefficient_limits=limits,
        substeps_per_interval=_SUBSTEPS,
        spacing_yx=(5.0, 5.0),
        boundary_echo=schedule[: 2 * _SUBSTEPS],
        boundary_support=support_schedule[: 2 * _SUBSTEPS],
        reconstruction="minmod",
        replay=True,
    )

    # This invokes the production conservative coefficient-box CFL bound.
    t.bounded_fv_coefficients(
        initial_echo.new_zeros(5),
        psi_basis=basis,
        coefficient_limits=limits,
        dt_seconds=60.0 / _SUBSTEPS,
        spacing_yx=(5.0, 5.0),
        reconstruction="minmod",
    )
    weights = limits[:, None, None]
    fx_bound = ((basis[:, 1:, :] - basis[:, :-1, :]).abs() * weights).sum(0)
    fy_bound = ((basis[:, :, 1:] - basis[:, :, :-1]).abs() * weights).sum(0)
    vertex_bound = (basis.abs() * weights).sum(0)
    roundoff = 8 * 6 * torch.finfo(dtype).eps
    ex = roundoff * (vertex_bound[1:, :] + vertex_bound[:-1, :])
    ey = roundoff * (vertex_bound[:, 1:] + vertex_bound[:, :-1])
    outward_bound = 0.5 * (
        fx_bound[:, 1:] + fx_bound[:, :-1] + fy_bound[1:, :] + fy_bound[:-1, :]
    ) + ex[:, 1:] + ex[:, :-1] + ey[1:, :] + ey[:-1, :]
    coefficient_box_cfl = (60.0 / _SUBSTEPS) * (outward_bound / 25.0).amax()
    truth_frames, _ = t.finite_volume_trajectory(
        initial_echo,
        torch.ones_like(initial_echo),
        truth_coefficients,
        initial_echo.new_tensor(0.008),
        psi_basis=basis,
        leads=_LEADS,
        substeps_per_interval=_SUBSTEPS,
        interval_seconds=60.0,
        spacing_yx=(5.0, 5.0),
        boundary_echo=schedule,
        boundary_support=support_schedule,
        reconstruction="minmod",
    )
    observation_frames = echo_to_dbz(truth_frames[:3], min_dbz=-10.0)
    obs, frozen = v.prepare_analysis(
        observation_frames,
        nowcast_config=NowcastConfig(interval_minutes=1, horizon_minutes=1, min_dbz=-10.0),
        analysis_config=v.AnalysisConfig(field_smoothness_weight=0.0),
        observation_std_dbz=0.1,
        fv_transport=transport,
    )
    pattern = torch.linspace(-0.2, 0.3, _HEIGHT * _WIDTH, dtype=dtype).reshape(
        _HEIGHT, _WIDTH
    )
    parameters = torch.cat((obs.dbz.flatten(), obs.dbz.new_tensor([0.02])))
    verification = (
        echo_to_dbz(truth_frames[3], min_dbz=-10.0) + 0.1 * pattern
    ).detach()
    return FVScaledResearchCase(
        observations=obs,
        frozen=frozen,
        future_boundary_echo=schedule[2 * _SUBSTEPS :],
        future_boundary_support=support_schedule[2 * _SUBSTEPS :],
        parameters=parameters,
        pattern=pattern,
        verification=verification,
        truth_initial_echo=initial_echo,
        definition={
            "grid": [_HEIGHT, _WIDTH],
            "spacing_yx": [5.0, 5.0],
            "vertex_coordinate_increment": 0.5,
            "basis": ["Y", "X", "XY", "(X^2-Y^2)/2", "X^2Y"],
            "coefficient_limits": list(_LIMITS),
            "truth_multipliers": list(_TRUTH_MULTIPLIERS),
            "truth_growth_per_interval": 0.008,
            "substeps_per_interval": _SUBSTEPS,
            "observation_leads_seconds": [0, 60, 120],
            "verification_lead_seconds": 180,
            "analysis_boundary_stage_pairs": 2 * _SUBSTEPS,
            "future_boundary_stage_pairs": _SUBSTEPS,
            "total_boundary_stage_pairs": _LEADS * _SUBSTEPS,
            "rk_stages": _RK_STAGE_COUNT,
            "coefficient_box_cfl_upper_bound": float(coefficient_box_cfl),
            "controls": _CONTROL_SIZE,
            "parameters": _PARAMETER_SIZE,
            "observation_cost": "sum pseudo-Huber + half squared control-prior norm",
            "support": "full known support",
        },
    )


def actual_cfl(case: FVScaledResearchCase, control: torch.Tensor) -> float:
    """Return the controlled-flow Courant number; growth does not enter it."""
    if control.shape != (_CONTROL_SIZE,):
        raise ValueError("scaled minmod control layout mismatch")
    spec = case.frozen.fv_transport
    if spec is None:
        raise ValueError("scaled minmod case requires FV transport")
    coefficients = t.bounded_fv_coefficients(
        control[-len(_LIMITS) - 1 : -1],
        psi_basis=spec.psi_basis,
        coefficient_limits=spec.coefficient_limits,
        dt_seconds=60.0 / _SUBSTEPS,
        spacing_yx=spec.spacing_yx,
        reconstruction="minmod",
    )
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    qx, qy = t.face_volume_fluxes(psi)
    outward = (
        qx[:, 1:].clamp_min(0)
        - qx[:, :-1].clamp_max(0)
        + qy[1:, :].clamp_min(0)
        - qy[:-1, :].clamp_max(0)
    )
    cell_area = spec.spacing_yx[0] * spec.spacing_yx[1]
    return float((60.0 / _SUBSTEPS) * (outward / cell_area).amax())


def functions(case: FVScaledResearchCase, expected_branch=None):
    """Return summed objective, conditional score, and strict branch checker."""
    probe = _load_inverse_probe()
    obs, frozen = case.observations, case.frozen
    if (
        obs.dbz.shape != (3, _HEIGHT, _WIDTH)
        or obs.dbz.dtype != torch.float64
        or obs.dbz.device.type != 'cpu'
        or frozen.fv_transport is None
        or frozen.fv_transport.reconstruction != "minmod"
        or frozen.fv_transport.substeps_per_interval != _SUBSTEPS
        or case.parameters.shape != (_PARAMETER_SIZE,)
        or case.pattern.shape != (_HEIGHT, _WIDTH)
        or case.verification.shape != (_HEIGHT, _WIDTH)
        or case.pattern.dtype != obs.dbz.dtype
        or case.verification.dtype != obs.dbz.dtype
        or not bool(torch.isfinite(case.pattern).all())
        or not bool(torch.isfinite(case.verification).all())
        or not bool(obs.detected_mask.all())
        or not bool(obs.valid_mask.all())
        or not bool(frozen.initial_support_mask.all())
        or frozen.neural_prior_dependency is not None
        or frozen.grid_time_contract is not None
    ):
        raise ValueError("scaled minmod case is outside its fixed full-support contract")
    for schedule in (frozen.fv_transport.boundary_support, case.future_boundary_support):
        for stages in schedule:
            for edges in stages:
                if not all(bool(torch.all(edge == 1)) for edge in edges):
                    raise ValueError("scaled minmod requires fully known boundary support")

    def contract(parameters):
        observed = parameters[:-1].reshape_as(obs.dbz)
        return replace(frozen, initial_background_dbz=observed[0] + parameters[-1] * case.pattern)

    def forecast(control, parameters):
        return echo_to_dbz(
            v.forecast_fv_analysis(
                control,
                contract(parameters),
                leads=1,
                boundary_start_interval=2,
                boundary_echo=case.future_boundary_echo,
                boundary_support=case.future_boundary_support,
            ).frames_linear[-1],
            min_dbz=-10.0,
        )

    def objective(control, parameters):
        observations = replace(obs, dbz=parameters[:-1].reshape_as(obs.dbz))
        return v.robust_objective(control, observations, contract(parameters))

    def score(control, parameters):
        return (forecast(control, parameters) - case.verification).square().mean()

    def branch_check(control, parameters):
        if control.shape != (_CONTROL_SIZE,) or parameters.shape != (_PARAMETER_SIZE,):
            raise ValueError("scaled minmod control/parameter layout mismatch")
        background = contract(parameters).initial_background_dbz
        analysis = frozen.analysis_config
        nowcast = frozen.nowcast_config
        offset = (background - nowcast.min_dbz) / analysis.echo_transform_scale_dbz
        margin = 64 * torch.finfo(background.dtype).eps * (
            (background.abs() + abs(nowcast.min_dbz)) / analysis.echo_transform_scale_dbz
            + analysis.transform_epsilon
        )
        if not bool(
            ((offset - analysis.transform_epsilon > margin) & (background < nowcast.max_dbz)).all()
        ):
            raise ValueError("scaled minmod background is outside its smooth transform branch")
        if not bool(
            ((parameters[:-1] > analysis.detection_limit_dbz) & (parameters[:-1] < nowcast.max_dbz)).all()
        ):
            raise ValueError("scaled minmod requires fixed detected observations")
        branch = probe.inspect_branches(lambda: forecast(control, parameters))
        if branch["euler_stages"] != _RK_STAGE_COUNT:
            raise ValueError(f"scaled minmod expected {_RK_STAGE_COUNT} inspected RK stages")
        if expected_branch is not None and (
            branch["choices"] != expected_branch["choices"]
            or branch["face_signs"] != expected_branch["face_signs"]
        ):
            raise ValueError("scaled minmod branch identity mismatch")
        return branch, "strict full-support 8x10 minmod; fixed masks, precision, geometry, and boundaries"

    return objective, score, branch_check
