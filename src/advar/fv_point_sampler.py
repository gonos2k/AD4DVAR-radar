"""Frozen bilinear sampling of dBZ fields at state-grid index coordinates."""
from __future__ import annotations

import torch
from torch import Tensor


def point_dbz_bilinear(field: Tensor, coordinates: Tensor) -> Tensor:
    """Sample a dBZ field at fixed ``(row, column)`` cell-center indices.

    ``field`` may have any leading dimensions and must end in ``[H, W]``.
    ``coordinates`` has shape ``[N, 2]`` and contains row/column indices in
    the same cell-center coordinate system. Coordinates and values are CPU
    FP64. Every point must lie strictly inside the grid so its floor-based
    2x2 stencil is wholly in-domain. The returned ``[..., N]`` values are
    bilinear averages in dBZ; this operator does not convert to linear Z.

    Coordinates are fixed geometry and are not differentiated. Autograd is
    preserved with respect to ``field``.
    """
    if not isinstance(field, Tensor) or field.ndim < 2:
        raise ValueError("field must be a tensor with trailing [H, W] dimensions")
    if field.dtype != torch.float64 or field.device.type != "cpu":
        raise ValueError("field must use CPU float64")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("field must contain only finite values")
    if not isinstance(coordinates, Tensor) or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [N, 2] in (row, column) order")
    if coordinates.dtype != torch.float64 or coordinates.device.type != "cpu":
        raise ValueError("coordinates must use CPU float64")
    if coordinates.requires_grad:
        raise ValueError("coordinates are fixed geometry and cannot require gradients")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("coordinates must contain only finite values")

    height, width = field.shape[-2:]
    if height < 3 or width < 3:
        raise ValueError("field grid must be at least 3x3 for an interior 2x2 stencil")
    row, column = coordinates.unbind(dim=1)
    if not bool(((row > 0) & (row < height - 1) & (column > 0) & (column < width - 1)).all()):
        raise ValueError("every point must be strictly inside the grid's 2x2 stencil domain")

    row0 = row.floor().to(torch.long)
    column0 = column.floor().to(torch.long)
    row_fraction = row - row0
    column_fraction = column - column0
    w00 = (1 - row_fraction) * (1 - column_fraction)
    w01 = (1 - row_fraction) * column_fraction
    w10 = row_fraction * (1 - column_fraction)
    w11 = row_fraction * column_fraction

    return (
        field[..., row0, column0] * w00
        + field[..., row0, column0 + 1] * w01
        + field[..., row0 + 1, column0] * w10
        + field[..., row0 + 1, column0 + 1] * w11
    )
