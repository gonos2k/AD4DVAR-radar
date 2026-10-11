"""Graph-connected capture for the declared trace-179 limiter event.

This research probe records one fixed minmod stage during a replay-disabled
trajectory.  It reuses the scoped observer and derivative definitions from
``fv_point_3h_limiter_event_probe`` so both probes share the same contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable

import torch
from torch import Tensor

from advar import transport
from examples.weather_scenarios import fv_point_3h_limiter_event_probe as limiter_event


EXPECTED_STAGE_COUNT = 360
FACE_AXIS = limiter_event.FACE_AXIS
FACE_ROW = limiter_event.FACE_ROW
FACE_COLUMN = limiter_event.FACE_COLUMN


@dataclass(frozen=True)
class LimiterEvent:
    """The fixed trace-179 x-direction event at interior cell (1, 2)."""

    stage_index: int = 179
    orientation: str = "x"
    interior_row: int = 1
    interior_column: int = 2
    expected_stage_count: int = EXPECTED_STAGE_COUNT


TARGET_EVENT = LimiterEvent()
EVENT = TARGET_EVENT
Trajectory = Callable[[Tensor], Any]


def _validate_event(event: LimiterEvent, expected_stages: int) -> None:
    if not isinstance(event, LimiterEvent):
        raise ValueError("only the frozen trace-179 x-cell (1, 2) event is supported")
    indices = (event.stage_index, event.interior_row, event.interior_column,
               event.expected_stage_count, expected_stages)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in indices):
        raise ValueError("only the frozen trace-179 x-cell (1, 2) event is supported")
    if (event != TARGET_EVENT or expected_stages != EXPECTED_STAGE_COUNT
            or event.expected_stage_count != expected_stages):
        raise ValueError("only the frozen trace-179 x-cell (1, 2) event is supported")


def event_values(trajectory: Trajectory, control: Tensor, side: int,
                 event: LimiterEvent = EVENT,
                 expected_stages: int = EXPECTED_STAGE_COUNT) -> Tensor:
    """Return connected ``[left, right, max(abs(q))]`` at trace 179.

    ``trajectory`` must use the analysis discretization with FV tape replay
    disabled. Parameters are closed over by the callable. Checkpoint replay
    cannot report this observer side effect because it is not a custom
    autograd Function output.
    """
    _validate_event(event, expected_stages)
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (-1, 1):
        raise ValueError("selected face side must be -1 or +1")
    if (not isinstance(control, Tensor) or control.shape != (26,)
            or control.dtype not in (torch.float32, torch.float64)
            or control.device.type != "cpu"):
        raise ValueError("event inputs require CPU control[26] in float32 or float64")

    stage_count = 0
    target_count = 0
    captures: list[Tensor] = []
    # The observer state includes a one-cell ghost border around the interior.
    row = event.interior_row + 1
    column = event.interior_column + 1

    def observe(q: Tensor, _qx: Tensor, _qy: Tensor) -> None:
        nonlocal stage_count, target_count
        stage = stage_count
        stage_count += 1
        if stage != event.stage_index:
            return
        target_count += 1
        if q.ndim != 2 or q.shape[0] <= row or q.shape[1] <= column + 1:
            raise ValueError("target limiter trace has an incompatible state-grid shape")
        center = q[row, column]
        left = center - q[row, column - 1]
        right = q[row, column + 1] - center
        captures.append(torch.stack((left, right, q.abs().amax())))

    with transport.selected_face_extension(FACE_AXIS, FACE_ROW, FACE_COLUMN, side):
        with limiter_event._capture_connected_stage_observer(observe):
            trajectory(control)
    if stage_count != expected_stages or target_count != 1 or len(captures) != 1:
        raise ValueError("analysis trajectory did not emit exactly 360 stages and one target event")
    return captures[0]


# Keep event algebra and AD transforms identical to the original probe.
raw_zeta = limiter_event.raw_zeta
same_sign_orientation = limiter_event.same_sign_orientation
event_derivatives = limiter_event.event_derivatives
