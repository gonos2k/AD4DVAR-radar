from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import Tensor

from advar import transport
from examples.weather_scenarios import fv_point_3h_active_limiter_probe as active_probe
from examples.weather_scenarios import fv_point_3h_limiter_event_probe as base_probe


def _trajectory(control: Tensor, *, stages: int = 360,
                fail_after: int | None = None) -> Tensor:
    observer = transport._minmod_stage_observer.get()
    selected = transport._selected_face.get()
    if observer is None:
        raise AssertionError("active event probe did not install its observer")
    if selected is None or (selected.axis, selected.row, selected.column) != ("y", 4, 3):
        raise AssertionError("active event probe did not bind Qy[4, 3]")

    side = selected.side
    center = 10.0 + 0.1 * control[0]
    left_slope = -2.0 + control[1].square() + 0.1 * side * control[4]
    right_slope = -1.0 + control[2] * control[3]
    q = torch.full((4, 5), -0.5, dtype=control.dtype)
    q = q.clone()
    q[2, 3] = center
    q[2, 2] = center - left_slope
    q[2, 4] = center + right_slope
    zero = torch.zeros_like(q)
    for stage in range(stages):
        observer(q, zero, zero)
        if fail_after is not None and stage == fail_after:
            raise RuntimeError("synthetic trajectory failure")
    return q.sum()


def test_trace179_uses_ghost_offset_and_keeps_values_connected_for_ad():
    control = torch.zeros(26, dtype=torch.float64)
    control[:5] = torch.tensor([0.5, 0.2, 0.3, 0.4, 0.25], dtype=control.dtype)
    direction = torch.linspace(-0.4, 0.6, 26, dtype=control.dtype)
    trajectory = lambda point: _trajectory(point)

    values = active_probe.event_values(trajectory, control, side=1)
    left = -2.0 + control[1].square() + 0.1 * control[4]
    right = -1.0 + control[2] * control[3]
    # Interior (1, 2) maps to state-grid (2, 3) after its one-cell ghost border.
    assert torch.allclose(values, torch.stack((left, right, 10 + 0.1 * control[0] - left)),
                          rtol=0, atol=1e-14)
    assert active_probe.same_sign_orientation(values) == -1

    event_function = lambda point: active_probe.event_values(trajectory, point, side=1)
    returned, directional, zeta_jvp, zeta_gradient = active_probe.event_derivatives(
        event_function, control, direction)
    assert torch.allclose(returned, values, rtol=0, atol=1e-14)
    assert torch.allclose(zeta_jvp, directional[0] - directional[1], rtol=0, atol=1e-14)
    assert zeta_gradient.shape == (26,)
    assert torch.allclose(torch.dot(zeta_gradient, direction), zeta_jvp,
                          rtol=1e-11, atol=1e-12)
    assert torch.allclose(zeta_gradient[1], 2 * control[1])
    assert torch.allclose(zeta_gradient[4], torch.tensor(0.1, dtype=control.dtype))
    assert torch.count_nonzero(zeta_gradient[5:]) == 0
    negative_side_values = active_probe.event_values(trajectory, control, side=-1)
    assert torch.allclose(negative_side_values[0], left - 0.2 * control[4])


@pytest.mark.parametrize("side", [-1, 1])
def test_fixed_qy_face_side_is_bound_and_capture_requires_exactly_360_stages(side: int):
    control = torch.zeros(26, dtype=torch.float64)
    values = active_probe.event_values(lambda point: _trajectory(point), control, side=side)
    assert values.shape == (3,)
    assert transport._selected_face.get() is None
    assert transport._minmod_stage_observer.get() is None

    for stages in (359, 361):
        with pytest.raises(ValueError, match="exactly 360 stages"):
            active_probe.event_values(lambda point: _trajectory(point, stages=stages),
                                      control, side=side)
        assert transport._selected_face.get() is None
        assert transport._minmod_stage_observer.get() is None


def test_orientation_qualification_rejects_crossing_tie_and_nonfinite_values():
    assert active_probe.same_sign_orientation(torch.tensor([-2.0, -1.0, 3.0])) == -1
    assert active_probe.same_sign_orientation(torch.tensor([2.0, 1.0, 3.0])) == 1
    assert active_probe.same_sign_orientation(torch.tensor([-2.0, 1.0, 3.0])) is None
    assert active_probe.same_sign_orientation(torch.tensor([-1.0, -1.0, 3.0])) is None
    assert active_probe.same_sign_orientation(torch.tensor([float("nan"), -1.0, 3.0])) is None


def test_capture_restores_context_after_trajectory_exception():
    control = torch.zeros(26, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="synthetic trajectory failure"):
        active_probe.event_values(lambda point: _trajectory(point, fail_after=180),
                                  control, side=-1)
    assert transport._selected_face.get() is None
    assert transport._minmod_stage_observer.get() is None


def test_capture_rejects_nesting_and_restores_outer_observer():
    control = torch.zeros(26, dtype=torch.float64)
    seen: list[Tensor] = []
    with transport.observe_minmod_stages(lambda q, _qx, _qy: seen.append(q)):
        with pytest.raises(RuntimeError, match="cannot be nested"):
            active_probe.event_values(lambda point: _trajectory(point), control, side=1)
        assert transport._minmod_stage_observer.get() is not None
    assert transport._minmod_stage_observer.get() is None
    assert transport._selected_face.get() is None


def test_only_declared_trace_and_contract_are_supported():
    control = torch.zeros(26, dtype=torch.float64)
    invalid_events = (
        active_probe.LimiterEvent(stage_index=178),
        active_probe.LimiterEvent(interior_row=cast(Any, 1.0)),
        active_probe.LimiterEvent(stage_index=True),
        active_probe.LimiterEvent(orientation="y"),
        active_probe.LimiterEvent(interior_row=0),
        active_probe.LimiterEvent(expected_stage_count=359),
    )
    for event in invalid_events:
        with pytest.raises(ValueError, match="trace-179"):
            active_probe.event_values(lambda point: _trajectory(point), control,
                                      side=1, event=event)
    with pytest.raises(ValueError, match="trace-179"):
        active_probe.event_values(lambda point: _trajectory(point), control,
                                  side=1, expected_stages=359)
    with pytest.raises(ValueError, match=r"side must be -1 or \+1"):
        active_probe.event_values(lambda point: _trajectory(point), control,
                                  side=0)
    with pytest.raises(ValueError, match=r"CPU control\[26\]"):
        active_probe.event_values(lambda point: _trajectory(point), cast(Any, [0.0] * 26),
                                  side=1)
    assert transport._selected_face.get() is None
    assert transport._minmod_stage_observer.get() is None


def test_event_algebra_and_derivatives_are_reused_from_original_probe():
    assert active_probe.raw_zeta is base_probe.raw_zeta
    assert active_probe.event_derivatives is base_probe.event_derivatives
