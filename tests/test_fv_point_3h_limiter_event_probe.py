from __future__ import annotations

import pytest
import torch
from torch import Tensor

from advar import transport
from examples.weather_scenarios import fv_point_3h_limiter_event_probe as event_probe


def _trajectory(control: Tensor, *, stages: int = 360, fail_after: int | None = None) -> Tensor:
    observer = transport._minmod_stage_observer.get()
    if observer is None:
        raise AssertionError("event probe did not install its observer")
    selected = transport._selected_face.get()
    if selected is None or (selected.axis, selected.row, selected.column) != ("y", 4, 3):
        raise AssertionError("event probe did not bind the selected Qy face")
    background = control[5] * torch.ones((4, 5), dtype=control.dtype)
    center = 10.0 + 0.01 * control[0]
    left_slope = -2.0 + control[1].square()
    right_slope = -1.0 + control[2] * control[3]
    q = background.clone()
    q[2, 3] = center
    q[2, 2] = center - left_slope
    q[2, 4] = center + right_slope
    zero = torch.zeros_like(q)
    for stage in range(stages):
        observer(q, zero, zero)
        if fail_after is not None and stage == fail_after:
            raise RuntimeError("synthetic trajectory failure")
    return q.sum()


def test_event_extracts_exact_trace_coordinate_with_connected_derivatives():
    control = torch.zeros(26, dtype=torch.float64)
    control[0] = 0.5
    control[1:4] = torch.tensor([0.2, 0.3, 0.4], dtype=control.dtype)
    control[5] = 0.1
    direction = torch.linspace(-0.4, 0.6, 26, dtype=control.dtype)
    trajectory = lambda point: _trajectory(point)
    values = event_probe.event_values(trajectory, control, side=1)
    left = -2.0 + control[1].square()
    right = -1.0 + control[2] * control[3]
    assert torch.allclose(values[:2], torch.stack((left, right)), rtol=0, atol=1e-14)
    assert values[2] == (10.0 + 0.01 * control[0] - left)
    assert event_probe.same_sign_orientation(values) == -1

    function = lambda point: event_probe.event_values(trajectory, point, side=1)
    returned, directional, zeta_directional, zeta_gradient = event_probe.event_derivatives(
        function, control, direction)
    assert torch.allclose(returned, values, rtol=0, atol=1e-14)
    assert torch.allclose(zeta_directional, directional[0] - directional[1], rtol=0, atol=1e-14)
    assert zeta_gradient.shape == (26,)
    assert torch.allclose(torch.dot(zeta_gradient, direction), zeta_directional,
        rtol=1e-11, atol=1e-12)
    assert torch.count_nonzero(zeta_gradient[6:]) == 0


def test_event_capture_resets_both_contexts_after_trajectory_failure():
    control = torch.zeros(26, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="synthetic trajectory failure"):
        event_probe.event_values(lambda point: _trajectory(point, fail_after=140), control, side=-1)
    assert transport._minmod_stage_observer.get() is None
    assert transport._selected_face.get() is None


def test_event_capture_requires_one_target_and_exact_stage_count():
    control = torch.zeros(26, dtype=torch.float64)
    with pytest.raises(ValueError, match="exactly 360 stages"):
        event_probe.event_values(lambda point: _trajectory(point, stages=361), control, side=1)
    assert transport._minmod_stage_observer.get() is None
    assert transport._selected_face.get() is None

    with pytest.raises(ValueError, match="exactly 360 stages"):
        event_probe.event_values(lambda point: _trajectory(point, stages=359), control, side=1)
    assert transport._minmod_stage_observer.get() is None
    assert transport._selected_face.get() is None


def test_public_detached_observer_nesting_is_rejected_and_restored():
    control = torch.zeros(26, dtype=torch.float64)
    received: list[Tensor] = []
    with transport.observe_minmod_stages(lambda q, _qx, _qy: received.append(q)):
        with pytest.raises(RuntimeError, match="cannot be nested"):
            event_probe.event_values(lambda point: _trajectory(point), control, side=1)
        assert transport._minmod_stage_observer.get() is not None
    assert transport._minmod_stage_observer.get() is None


def test_same_sign_qualification_is_outside_event_derivative():
    same_sign = torch.tensor([-2.0, -1.0, 3.0], dtype=torch.float64)
    crossing = torch.tensor([-2.0, 1.0, 3.0], dtype=torch.float64)
    tie = torch.tensor([-1.0, -1.0, 3.0], dtype=torch.float64)
    nonfinite = torch.tensor([float("inf"), 1.0, 3.0], dtype=torch.float64)
    assert event_probe.same_sign_orientation(same_sign) == -1
    assert event_probe.same_sign_orientation(crossing) is None
    assert event_probe.same_sign_orientation(tie) is None
    assert event_probe.same_sign_orientation(nonfinite) is None
