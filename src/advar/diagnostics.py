from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .physics import (
    RemapCell,
    freeze_remap_cell,
    remap,
    remap_fractions,
    shift_zero,
    validate_remap_cell,
)


class EchoPositivityError(FloatingPointError):
    pass


@dataclass(frozen=True)
class PositivityAudit:
    minimum_before_fix: float
    corrected_count: int
    corrected_integral: float


@dataclass(frozen=True)
class TransportAudit:
    echo_integral_before: float
    echo_integral_after: float
    boundary_outflow_integral: float
    echo_budget_error: float


def validate_physical_echo(
    echo: Tensor,
    *,
    name: str,
) -> tuple[Tensor, PositivityAudit]:
    if not echo.is_floating_point():
        raise TypeError(f"{name}: echo must be floating point")
    if echo.numel() == 0:
        raise ValueError(f"{name}: echo must not be empty")
    detached = echo.detach()
    if not bool(torch.all(torch.isfinite(detached))):
        raise EchoPositivityError(f"{name}: non-finite physical echo")

    minimum = torch.amin(detached)
    tolerance = (
        32.0
        * torch.finfo(echo.dtype).eps
        * torch.abs(minimum).clamp_min(1.0)
    )
    negative = detached < 0.0
    corrected_count = int(torch.count_nonzero(negative))
    corrected_integral = float(
        torch.where(negative, -detached, torch.zeros_like(detached)).sum()
    )
    if bool(minimum < -tolerance):
        raise EchoPositivityError(
            f"{name}: physical echo became negative "
            f"(minimum={float(minimum):.9g}, "
            f"tolerance={float(tolerance):.9g}, "
            f"count={corrected_count})"
        )

    return (
        torch.where(echo < 0.0, torch.zeros_like(echo), echo),
        PositivityAudit(
            minimum_before_fix=float(minimum),
            corrected_count=corrected_count,
            corrected_integral=corrected_integral,
        ),
    )


def audit_transport(
    echo: Tensor,
    displacement_yx: Tensor,
    *,
    cell: RemapCell | None = None,
    moved: Tensor | None = None,
) -> TransportAudit:
    echo, _ = validate_physical_echo(echo, name="transport input")
    cell = freeze_remap_cell(displacement_yx) if cell is None else cell
    validate_remap_cell(displacement_yx, cell)
    moved = (
        remap(echo, displacement_yx, cell=cell)
        if moved is None
        else validate_physical_echo(moved, name="transport output")[0]
    )
    fraction_y, fraction_x = remap_fractions(
        echo,
        displacement_yx,
        cell,
    )
    branches = (
        ((cell.y, cell.x), (1.0 - fraction_y) * (1.0 - fraction_x)),
        ((cell.y + 1, cell.x), fraction_y * (1.0 - fraction_x)),
        ((cell.y, cell.x + 1), (1.0 - fraction_y) * fraction_x),
        ((cell.y + 1, cell.x + 1), fraction_y * fraction_x),
    )
    before = echo.sum()
    after = moved.sum()
    outflow = echo.new_zeros(())
    for (dy, dx), weight in branches:
        retained = shift_zero(echo, dy, dx).sum()
        outflow = outflow + weight * (before - retained)
    budget_error = before - after - outflow
    return TransportAudit(
        echo_integral_before=float(before.detach()),
        echo_integral_after=float(after.detach()),
        boundary_outflow_integral=float(outflow.detach()),
        echo_budget_error=float(torch.abs(budget_error.detach())),
    )
