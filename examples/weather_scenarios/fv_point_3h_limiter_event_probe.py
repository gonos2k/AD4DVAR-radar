"""Differentiable capture helpers for one frozen FV limiter event.

This is a research-only observer path. The public transport observer is meant
for detached diagnostics, so this module installs a private, read-only callback
while evaluating a replay-disabled diagnostic trajectory.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Iterator

import torch
from torch import Tensor

from advar import transport


EXPECTED_STAGE_COUNT = 360
FACE_AXIS = "y"
FACE_ROW = 4
FACE_COLUMN = 3


@dataclass(frozen=True)
class LimiterEvent:
    """The event at trace 124, x-direction interior cell (1, 2)."""

    stage_index: int = 124
    orientation: str = "x"
    interior_row: int = 1
    interior_column: int = 2
    expected_stage_count: int = EXPECTED_STAGE_COUNT


TARGET_EVENT = LimiterEvent()
EVENT = TARGET_EVENT
Trajectory = Callable[[Tensor], Any]


@contextmanager
def _capture_connected_stage_observer(
    observer: Callable[[Tensor, Tensor, Tensor], None],
) -> Iterator[None]:
    """Capture graph-connected stages without the public observer's copies.

    The transport callback is read-only and scoped to this event probe. Reject
    nesting so this private path cannot bypass another observer's contract.
    """
    if transport._minmod_stage_observer.get() is not None:
        raise RuntimeError("connected limiter-event capture cannot be nested")
    token = transport._minmod_stage_observer.set(observer)
    try:
        yield
    finally:
        transport._minmod_stage_observer.reset(token)


def _validate_event(event: LimiterEvent, expected_stages: int) -> None:
    if (event != TARGET_EVENT or expected_stages != EXPECTED_STAGE_COUNT
            or event.expected_stage_count != expected_stages
            or event.stage_index < 0 or event.stage_index >= expected_stages):
        raise ValueError("only the frozen trace-124 x-cell (1, 2) event is supported")


def event_values(trajectory: Trajectory, control: Tensor, side: int,
                 event: LimiterEvent = EVENT,
                 expected_stages: int = EXPECTED_STAGE_COUNT) -> Tensor:
    """Return graph-connected ``[left, right, max(abs(q))]`` at the event.

    ``trajectory`` must be the analysis trajectory using the same fixed
    discretization with FV tape replay disabled. Captured model parameters are
    closed over by that callable; the trajectory must not use checkpoint replay,
    whose observer side effect is not an output of its custom Function.
    """
    _validate_event(event, expected_stages)
    if isinstance(side, bool) or not isinstance(side, Integral) or side not in (-1, 1):
        raise ValueError("selected face side must be -1 or +1")
    if (control.shape != (26,) or control.dtype not in (torch.float32, torch.float64)
            or control.device.type != "cpu"):
        raise ValueError("event inputs require CPU control[26] in float32 or float64")

    stage_count = 0
    target_count = 0
    captures: list[Tensor] = []
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
        with _capture_connected_stage_observer(observe):
            trajectory(control)
    if stage_count != expected_stages or target_count != 1 or len(captures) != 1:
        raise ValueError("analysis trajectory did not emit exactly 360 stages and one target event")
    return captures[0]


def raw_zeta(values: Tensor) -> Tensor:
    """Return the un-oriented event function ``left - right``."""
    if values.shape != (3,):
        raise ValueError("event values must have shape [left, right, qabsmax]")
    return values[0] - values[1]


def same_sign_orientation(values: Tensor) -> int | None:
    """Qualify a same-sign limiter event outside AD and return its orientation.

    The orientation makes the base positive gap. For the prepared point this
    is ``-1`` because left and right are both negative and ``left-right < 0``.
    """
    if values.shape != (3,):
        raise ValueError("event values must have shape [left, right, qabsmax]")
    if not bool(torch.isfinite(values).all()):
        return None
    left, right = values[0], values[1]
    same_positive = bool((left > 0) & (right > 0))
    same_negative = bool((left < 0) & (right < 0))
    if not (same_positive or same_negative):
        return None
    zeta = left - right
    if bool(zeta == 0):
        return None
    return 1 if bool(zeta > 0) else -1


def event_derivatives(event_function: Callable[[Tensor], Tensor], control: Tensor,
                      direction: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return values, vector JVP, raw-zeta JVP, and full raw-zeta gradient."""
    if control.shape != (26,) or direction.shape != control.shape:
        raise ValueError("event derivative inputs must be matching full 26-vectors")
    jvp_result = torch.func.jvp(event_function, (control,), (direction,))
    values, directional = jvp_result[0], jvp_result[1]
    zeta_directional = raw_zeta(directional)
    gradient = torch.func.grad(lambda point: raw_zeta(event_function(point)))(control)
    return values, directional, zeta_directional, gradient
