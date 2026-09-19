"""C4c FV continuation checks; no forecast or sensitivity certification."""

from typing import cast

import pytest
import torch

from advar.nowcast import NowcastConfig
from advar.physics import echo_to_dbz
from advar.transport import BoundarySchedule, finite_volume_trajectory
from advar.variational import (
    AnalysisConfig,
    AnalysisObservations,
    AnalysisTrajectory,
    FrozenOuterState,
    FVAnalysisTransport,
    analysis_trajectory,
    forecast_fv_analysis,
    initial_control,
    prepare_analysis,
)


def _schedule(
    *,
    height: int,
    width: int,
    steps: int,
    scale: torch.Tensor | None = None,
    start: int = 0,
) -> BoundarySchedule:
    """Build distinct positive traces for every future substep."""

    dtype = torch.float64
    one = torch.ones((), dtype=dtype) if scale is None else scale
    result = []
    for step in range(steps):
        base = one.new_tensor(0.18 + 0.035 * (step + start))
        edges = (
            torch.ones(height, dtype=dtype) * (base + 0.02 * one),
            torch.ones(height, dtype=dtype) * (base + 0.01 * one),
            torch.ones(width, dtype=dtype) * (base + 0.03 * one),
            torch.ones(width, dtype=dtype) * (base + 0.015 * one),
        )
        result.append((edges, edges))
    return cast(BoundarySchedule, tuple(result))


def _fixture(reconstruction: str = "donorcell") -> tuple[AnalysisObservations, FrozenOuterState, AnalysisTrajectory, FVAnalysisTransport]:
    dtype = torch.float64
    height, width = 3, 4
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )
    y_vertex, _ = torch.meshgrid(
        torch.arange(height + 1, dtype=dtype),
        torch.arange(width + 1, dtype=dtype),
        indexing="ij",
    )
    initial_echo = 4.0 + 0.15*x + 0.1*y + 0.04*x*y + 0.013*x.square() + 0.017*y.square()
    basis = y_vertex.unsqueeze(0)
    limits = torch.tensor([0.4], dtype=dtype)
    analysis_echo = _schedule(height=height, width=width, steps=4)
    analysis_support = tuple(
        (
            tuple(torch.ones_like(edge) for edge in analysis_echo[0][0]),
            tuple(torch.ones_like(edge) for edge in analysis_echo[0][0]),
        )
        for _ in range(4)
    )
    specification = FVAnalysisTransport(
        psi_basis=basis,
        coefficient_limits=limits,
        substeps_per_interval=2,
        spacing_yx=(10.0, 10.0),
        boundary_echo=analysis_echo,
        boundary_support=analysis_support,
        reconstruction=reconstruction,
        max_courant=0.5,
        replay=True,
    )
    target, _ = finite_volume_trajectory(
        initial_echo,
        torch.ones_like(initial_echo),
        initial_echo.new_tensor([0.012]),
        initial_echo.new_tensor(0.005),
        psi_basis=basis,
        leads=2,
        substeps_per_interval=2,
        interval_seconds=60.0,
        spacing_yx=(10.0, 10.0),
        boundary_echo=analysis_echo,
        boundary_support=analysis_support,
        reconstruction=reconstruction,
        max_courant=0.5,
        replay=True,
    )
    frames_dbz = echo_to_dbz(target, min_dbz=-10.0)
    observations, frozen = prepare_analysis(
        frames_dbz,
        nowcast_config=NowcastConfig(
            interval_minutes=1,
            horizon_minutes=1,
            min_dbz=-10.0,
        ),
        analysis_config=AnalysisConfig(field_smoothness_weight=0.0),
        observation_std_dbz=0.1,
        fv_transport=specification,
    )
    return observations, frozen, analysis_trajectory(initial_control(frozen), frozen), specification


def _future_schedules(
    frozen: FrozenOuterState,
    *,
    scale: torch.Tensor | None = None,
) -> tuple[BoundarySchedule, BoundarySchedule]:
    height, width = frozen.initial_background_dbz.shape
    echo = _schedule(height=height, width=width, steps=4, scale=scale, start=4)
    support = tuple(
        (
            tuple(torch.ones_like(edge) for edge in echo[0][0]),
            tuple(torch.ones_like(edge) for edge in echo[0][0]),
        )
        for _ in range(4)
    )
    return echo, support


def test_forecast_starts_at_analysis_end_and_matches_split_backend():
    _, frozen, analysis, _ = _fixture()
    control = initial_control(frozen)
    control[-2:] = control.new_tensor([0.02, 0.01])
    analysis = analysis_trajectory(control, frozen)
    future_echo, future_support = _future_schedules(frozen)
    result = forecast_fv_analysis(
        control,
        frozen,
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future_echo,
        boundary_support=future_support,
    )
    assert result.displacement_yx is None
    assert result.frames_linear.shape == (3, 3, 4)
    assert result.support_frames is not None
    torch.testing.assert_close(result.frames_linear[0], analysis.frames_linear[-1])
    torch.testing.assert_close(result.support_frames[0], analysis.support_frames[-1])

    psi = analysis.psi_coefficients
    growth = analysis.log_growth_per_step
    first_echo, first_support = finite_volume_trajectory(
        analysis.frames_linear[-1], analysis.support_frames[-1], psi, growth,
        psi_basis=frozen.fv_transport.psi_basis, leads=1,
        substeps_per_interval=2, interval_seconds=60.0,
        spacing_yx=frozen.fv_transport.spacing_yx,
        boundary_echo=future_echo[:2], boundary_support=future_support[:2],
        reconstruction=frozen.fv_transport.reconstruction,
        max_courant=frozen.fv_transport.max_courant, replay=True,
    )
    second_echo, second_support = finite_volume_trajectory(
        first_echo[-1], first_support[-1], psi, growth,
        psi_basis=frozen.fv_transport.psi_basis, leads=1,
        substeps_per_interval=2, interval_seconds=60.0,
        spacing_yx=frozen.fv_transport.spacing_yx,
        boundary_echo=future_echo[2:], boundary_support=future_support[2:],
        reconstruction=frozen.fv_transport.reconstruction,
        max_courant=frozen.fv_transport.max_courant, replay=True,
    )
    expected = torch.stack((analysis.frames_linear[-1], first_echo[-1], second_echo[-1]))
    expected_support = torch.stack((analysis.support_frames[-1], first_support[-1], second_support[-1]))
    torch.testing.assert_close(result.frames_linear, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.support_frames, expected_support, rtol=0, atol=0)


def test_future_time_varying_edges_are_consumed_and_differentiable():
    _, frozen, analysis, _ = _fixture()
    control = initial_control(frozen)
    control[-2:] = control.new_tensor([0.02, 0.01])
    scale = torch.ones((), dtype=control.dtype)

    def run_edges(value: torch.Tensor) -> torch.Tensor:
        echo, support = _future_schedules(frozen, scale=value)
        return forecast_fv_analysis(
            control,
            frozen,
            leads=2,
            boundary_start_interval=2,
            boundary_echo=echo,
            boundary_support=support,
        ).frames_linear

    low = run_edges(scale - 0.5)
    high = run_edges(scale + 0.5)
    assert not torch.equal(low, high)
    torch.testing.assert_close(low[0], high[0], rtol=0, atol=0)
    value, tangent = torch.func.jvp(run_edges, (scale,), (scale.new_tensor(0.1),))
    assert torch.isfinite(value).all()
    assert torch.isfinite(tangent).all()
    torch.testing.assert_close(tangent[0], torch.zeros_like(tangent[0]), rtol=0, atol=0)
    assert torch.linalg.vector_norm(tangent) > 0

    def run_control(value: torch.Tensor) -> torch.Tensor:
        echo, support = _future_schedules(frozen)
        return forecast_fv_analysis(
            value,
            frozen,
            leads=2,
            boundary_start_interval=2,
            boundary_echo=echo,
            boundary_support=support,
        ).frames_linear

    direction = torch.zeros_like(control)
    field_size = frozen.active_field_index.numel()
    direction[:field_size] = 0.01
    direction[-2:] = control.new_tensor([0.01, 0.02])
    value, tangent = torch.func.jvp(run_control, (control,), (direction,))
    _, pullback = torch.func.vjp(run_control, control)
    seed = torch.linspace(0.2, 1.0, value.numel(), dtype=control.dtype).reshape_as(value)
    torch.testing.assert_close(
        (seed * tangent).sum(),
        (direction * pullback(seed)[0]).sum(),
        rtol=1e-10,
        atol=1e-12,
    )
    assert torch.linalg.vector_norm(tangent) > 0


def test_forecast_rejects_a_boundary_start_other_than_analysis_interval():
    _, frozen, _, _ = _fixture()
    control = initial_control(frozen)
    future_echo, future_support = _future_schedules(frozen)
    with pytest.raises(ValueError, match="boundary_start_interval|start"):
        forecast_fv_analysis(
            control,
            frozen,
            leads=1,
            boundary_start_interval=1,
            boundary_echo=future_echo[:2],
            boundary_support=future_support[:2],
        )


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
def test_analysis_plus_forecast_matches_one_uninterrupted_trajectory_and_jvp(scheme):
    _, frozen, _, specification = _fixture(scheme)
    control = initial_control(frozen)
    control[-2:] = control.new_tensor([.02, .01])
    future, support = _future_schedules(frozen)

    def split(c):
        return forecast_fv_analysis(
            c, frozen, leads=2, boundary_start_interval=2,
            boundary_echo=future, boundary_support=support,
        ).frames_linear

    def whole(c):
        decoded = analysis_trajectory(c, frozen)
        frames, _ = finite_volume_trajectory(
            decoded.frames_linear[0], frozen.initial_support_mask.to(c),
            decoded.psi_coefficients, decoded.log_growth_per_step,
            psi_basis=specification.psi_basis, leads=4, substeps_per_interval=2,
            interval_seconds=60., spacing_yx=specification.spacing_yx,
            boundary_echo=specification.boundary_echo+future,
            boundary_support=specification.boundary_support+support,
            reconstruction=specification.reconstruction, replay=False,
        )
        return frames[2:]

    direction = .01*torch.cos(torch.arange(control.numel(), dtype=control.dtype))
    value, tangent = torch.func.jvp(split, (control,), (direction,))
    full_value, full_tangent = torch.func.jvp(whole, (control,), (direction,))
    torch.testing.assert_close(value, full_value, rtol=0, atol=0)
    torch.testing.assert_close(tangent, full_tangent, rtol=1e-11, atol=1e-13)
    h = 1e-4
    difference = (split(control+h*direction)-split(control-h*direction))/(2*h)
    torch.testing.assert_close(tangent, difference, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('start', [0, 1, 3, 2., True])
def test_wrong_boundary_time_is_rejected_before_analysis(monkeypatch, start):
    from advar import variational as implementation
    _, frozen, _, _ = _fixture()
    def unexpected(*args, **kwargs):
        raise AssertionError('analysis must not run for invalid boundary time')
    monkeypatch.setattr(implementation, 'analysis_trajectory', unexpected)
    with pytest.raises(ValueError, match='interval 2'):
        forecast_fv_analysis(
            initial_control(frozen), frozen, leads=1, boundary_start_interval=start,
            boundary_echo=(), boundary_support=(),
        )


def test_future_boundary_schedule_cannot_reuse_a_different_horizon():
    _, frozen, _, _ = _fixture()
    future, support = _future_schedules(frozen)
    with pytest.raises(ValueError, match='length'):
        forecast_fv_analysis(
            initial_control(frozen), frozen, leads=1, boundary_start_interval=2,
            boundary_echo=future, boundary_support=support,
        )
