"""The two observed minmod event forms, independent of stage/cell numbers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class MinmodEvent:
    stage: int
    axis: str
    row: int
    column: int
    kind: str
    other_sign: int

    @property
    def key(self) -> tuple[int, str, int, int]:
        return self.stage, self.axis, self.row, self.column

    def selector(self, side: int) -> int:
        if side not in (-1, 1):
            raise ValueError("event side must be -1 or +1")
        if self.kind == "gap":
            return 1 if side == self.other_sign else 2
        active = 1 if self.kind == "left_zero" else 2
        return active if side == self.other_sign else 0

    def value(self, left: Tensor, right: Tensor) -> Tensor:
        if self.kind == "gap":
            return right - left
        return left if self.kind == "left_zero" else right


def classify_event(stage: int, axis: str, row: int, column: int,
                   left: float, right: float, scale: float) -> MinmodEvent:
    tolerance = 128 * torch.finfo(torch.float64).eps * scale
    if abs(left) <= tolerance and abs(right) <= tolerance:
        raise ValueError("both operands unresolved: two event surfaces coincide")
    if abs(left) <= tolerance:
        return MinmodEvent(stage, axis, row, column, "left_zero", 1 if right > 0 else -1)
    if abs(right) <= tolerance:
        return MinmodEvent(stage, axis, row, column, "right_zero", 1 if left > 0 else -1)
    if ((left > 0 and right > 0) or (left < 0 and right < 0)) and abs(right-left) <= tolerance:
        return MinmodEvent(stage, axis, row, column, "gap", 1 if left > 0 else -1)
    raise ValueError("operands do not define either observed event form")


def classify_change(identity: tuple[int, str, int, int], old: int, new: int,
                    before: tuple[float, float], after: tuple[float, float]) -> MinmodEvent:
    if old in (1, 2, 3) and new in (1, 2, 3):
        sign = 1 if before[0] > 0 else -1
        if not all(sign*x > 0 for x in (*before, *after)):
            raise ValueError("left/right transition changed operand sign as well")
        return MinmodEvent(*identity, "gap", sign)
    active = new if old == 0 else old
    if 0 not in (old, new) or active not in (1, 2):
        raise ValueError("selector change has unresolved simultaneous event forms")
    index = active - 1
    if (before[index] > 0 and after[index] > 0) or (before[index] < 0 and after[index] < 0):
        raise ValueError("active/inactive transition does not bracket its selected operand")
    other = before[1-index] if old else after[1-index]
    if other == 0 or not ((before[1-index] > 0 and after[1-index] > 0)
                         or (before[1-index] < 0 and after[1-index] < 0)):
        raise ValueError("both operand signs changed: simultaneous event unresolved")
    return MinmodEvent(*identity, "left_zero" if active == 1 else "right_zero", 1 if other > 0 else -1)


def event_action(decisions: list[dict[str,Any]], hull_inf: float) -> str:
    if any(d["decision"]=="unresolved" for d in decisions):
        return "unresolved"
    if any(d["decision"]=="release" for d in decisions):
        return "release"
    if hull_inf<1e-10:
        return "stationary"
    return "maintain" if decisions else "smooth"
