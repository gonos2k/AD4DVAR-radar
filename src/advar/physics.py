from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor


FORECAST_INTEGRATOR_VERSION = "local-conservative-slice-remap-v2"


@dataclass(frozen=True)
class RemapCell:
    y: int
    x: int

    def __post_init__(self) -> None:
        if type(self.y) is not int or type(self.x) is not int:
            raise TypeError("remap cell coordinates must be integers")


class FrozenCellMismatchError(ValueError):
    pass


def dbz_to_echo(
    dbz: Tensor,
    *,
    min_dbz: float,
    max_dbz: float | None = None,
) -> Tensor:
    clean = torch.nan_to_num(
        dbz,
        nan=min_dbz,
        posinf=max_dbz,
        neginf=min_dbz,
    ).clamp_min(min_dbz)
    if max_dbz is not None:
        clean = clean.clamp_max(max_dbz)
    echo_floor = 10.0 ** (min_dbz / 10.0)
    exponent = (math.log(10.0) / 10.0) * (clean - min_dbz)
    return echo_floor * torch.expm1(exponent)


def echo_to_dbz(
    echo: Tensor,
    *,
    min_dbz: float,
    max_dbz: float | None = None,
) -> Tensor:
    echo_floor = 10.0 ** (min_dbz / 10.0)
    dbz = min_dbz + (10.0 / math.log(10.0)) * torch.log1p(
        echo / echo_floor
    )
    return dbz if max_dbz is None else dbz.clamp(max=max_dbz)


def freeze_remap_cell(displacement_yx: Tensor) -> RemapCell:
    _validate_displacement(displacement_yx)
    detached = displacement_yx.detach()
    return RemapCell(
        y=math.floor(float(detached[0])),
        x=math.floor(float(detached[1])),
    )


def remap(
    echo: Tensor,
    displacement_yx: Tensor,
    *,
    cell: RemapCell | None = None,
) -> Tensor:
    _validate_echo_shape(echo)
    _validate_displacement(displacement_yx)
    # Preserve the displacement device and dtype until after the frozen branch
    # and fractions are known. A CPU float64 shift is valid for an MPS float32
    # echo, and a giant finite shift must not overflow before the zero branch.
    displacement = displacement_yx
    cell = freeze_remap_cell(displacement) if cell is None else cell
    validate_remap_cell(displacement, cell)
    return remap_core(echo, displacement, cell)


def remap_core(
    echo: Tensor,
    displacement_yx: Tensor,
    cell: RemapCell,
) -> Tensor:
    fraction_y, fraction_x = remap_fractions(
        echo,
        displacement_yx,
        cell,
    )
    return (
        (1.0 - fraction_y)
        * (1.0 - fraction_x)
        * shift_zero(echo, cell.y, cell.x)
        + fraction_y
        * (1.0 - fraction_x)
        * shift_zero(echo, cell.y + 1, cell.x)
        + (1.0 - fraction_y)
        * fraction_x
        * shift_zero(echo, cell.y, cell.x + 1)
        + fraction_y
        * fraction_x
        * shift_zero(echo, cell.y + 1, cell.x + 1)
    )


def react_core(echo: Tensor, log_growth: Tensor | float) -> Tensor:
    growth = torch.as_tensor(
        log_growth,
        dtype=echo.dtype,
        device=echo.device,
    )
    return echo * torch.exp(growth)


def advance(
    echo: Tensor,
    displacement_yx: Tensor,
    log_growth: Tensor | float,
    cell: RemapCell,
) -> Tensor:
    return react_core(
        remap_core(echo, displacement_yx, cell),
        log_growth,
    )


def validate_remap_cell(
    displacement_yx: Tensor,
    cell: RemapCell,
) -> None:
    _validate_displacement(displacement_yx)
    detached = displacement_yx.detach()
    # Convert the frozen integer through Python float before constructing a
    # tensor.  Tensor scalar arithmetic otherwise attempts to represent large
    # Python integers in the tensor dtype and can overflow for finite motion.
    cell_tensor = torch.stack(
        (
            detached.new_zeros(()) + float(cell.y),
            detached.new_zeros(()) + float(cell.x),
        )
    )
    fractions = detached - cell_tensor
    tolerance = 32.0 * min(
        torch.finfo(detached.dtype).eps,
        torch.finfo(torch.float32).eps,
    )
    if bool(torch.any(fractions < -tolerance)) or bool(
        torch.any(fractions > 1.0 + tolerance)
    ):
        raise FrozenCellMismatchError(
            "frozen remap cell is inconsistent with displacement "
            f"(cell=({cell.y}, {cell.x}), "
            f"displacement=({float(detached[0]):.9g}, "
            f"{float(detached[1]):.9g}))"
        )


def remap_fractions(
    echo: Tensor,
    displacement_yx: Tensor,
    cell: RemapCell,
) -> tuple[Tensor, Tensor]:
    displacement = displacement_yx
    # Keep the displacement as the differentiable operand.  The cell is a
    # frozen branch coordinate; converting it through float avoids an
    # overflowing Python-int-to-tensor conversion for finite extreme shifts.
    # Scalar arithmetic also works under MPS function transforms, where
    # new_tensor with a Python sequence is not supported.
    cell_tensor = torch.stack(
        (
            displacement.new_zeros(()) + float(cell.y),
            displacement.new_zeros(()) + float(cell.x),
        )
    )
    fractions = (displacement - cell_tensor).clamp(0.0, 1.0)
    # MPS cannot widen to FP64, in forward or reverse mode. Move away from
    # MPS before widening, and narrow on the source before moving to MPS.
    if fractions.device.type == "mps":
        fractions = fractions.to(device=echo.device)
    fractions = fractions.to(dtype=echo.dtype).to(device=echo.device)
    fraction_y, fraction_x = fractions.unbind()
    return fraction_y, fraction_x


def shift_zero(echo: Tensor, dy: int, dx: int) -> Tensor:
    height, width = echo.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        # Keep zero derivatives without empty-slice padding, whose MPS VJP fails.
        return echo.clone().zero_()
    source = echo[
        ...,
        max(0, -dy) : min(height, height - dy),
        max(0, -dx) : min(width, width - dx),
    ]
    return F.pad(
        source,
        (
            max(dx, 0),
            max(-dx, 0),
            max(dy, 0),
            max(-dy, 0),
        ),
    )


def _validate_echo_shape(echo: Tensor) -> None:
    if echo.ndim != 2:
        raise ValueError("echo must have shape [height, width]")
    if not echo.is_floating_point():
        raise TypeError("echo must be floating point")
    if echo.shape[0] < 2 or echo.shape[1] < 2:
        raise ValueError("echo height and width must both be at least 2")


def _validate_displacement(displacement_yx: Tensor) -> None:
    if displacement_yx.shape != (2,):
        raise ValueError("displacement_yx must have shape [2]")
    if not displacement_yx.is_floating_point():
        raise TypeError("displacement_yx must be floating point")
    if not bool(torch.all(torch.isfinite(displacement_yx.detach()))):
        raise ValueError("displacement_yx must be finite")
