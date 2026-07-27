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
    displacement = displacement_yx.to(dtype=echo.dtype, device=echo.device)
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
    fractions = detached - detached.new_tensor((cell.y, cell.x))
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
    base = echo.new_tensor((cell.y, cell.x))
    fractions = (
        displacement_yx.to(dtype=echo.dtype, device=echo.device) - base
    ).clamp(0.0, 1.0)
    return fractions[0], fractions[1]


def shift_zero(echo: Tensor, dy: int, dx: int) -> Tensor:
    height, width = echo.shape
    padded = F.pad(
        echo,
        (
            max(dx, 0),
            max(-dx, 0),
            max(dy, 0),
            max(-dy, 0),
        ),
    )
    start_y = max(-dy, 0)
    start_x = max(-dx, 0)
    return padded[start_y : start_y + height, start_x : start_x + width]


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
