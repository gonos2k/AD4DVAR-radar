"""Standalone minmod-MUSCL FV comparison kernel.

This file is deliberately outside the production transport path.  It supports
known-zero exterior traces and a stationary, discretely divergence-free flow.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Tuple

import torch
from torch import Tensor


def _tensor(name: str, value: object, *, dtype: torch.dtype | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have float32 or float64 dtype")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _minmod(left: Tensor, right: Tensor) -> Tensor:
    # Avoid left*right: finite inputs can overflow in that sign test.
    positive = (left > 0) & (right > 0)
    negative = (left < 0) & (right < 0)
    positive_value = torch.minimum(left, right)
    negative_value = torch.maximum(left, right)
    return torch.where(positive, positive_value, torch.where(negative, negative_value, torch.zeros_like(left)))


def _slopes(q: Tensor) -> Tuple[Tensor, Tensor]:
    height, width = q.shape
    if width <= 2:
        sx = torch.zeros_like(q)
    else:
        interior_x = _minmod(q[:, 1:-1] - q[:, :-2], q[:, 2:] - q[:, 1:-1])
        sx = torch.cat((torch.zeros_like(q[:, :1]), interior_x, torch.zeros_like(q[:, :1])), dim=1)
        if height > 2:
            sx = torch.cat((torch.zeros_like(q[:1, :]), sx[1:-1, :], torch.zeros_like(q[:1, :])), dim=0)
        else:
            sx = torch.zeros_like(q)
    if height <= 2:
        sy = torch.zeros_like(q)
    else:
        interior_y = _minmod(q[1:-1, :] - q[:-2, :], q[2:, :] - q[1:-1, :])
        sy = torch.cat((torch.zeros_like(q[:1, :]), interior_y, torch.zeros_like(q[:1, :])), dim=0)
        if width > 2:
            sy = torch.cat((torch.zeros_like(q[:, :1]), sy[:, 1:-1], torch.zeros_like(q[:, :1])), dim=1)
        else:
            sy = torch.zeros_like(q)
    return sx, sy


def _euler_muscl(q: Tensor, qx: Tensor, qy: Tensor, dt: Tensor, area: Tensor) -> Tuple[Tensor, Tensor]:
    sx, sy = _slopes(q)
    zero_x = torch.zeros_like(q[:, :1])
    zero_y = torch.zeros_like(q[:1, :])
    left_face = q - 0.5 * sx
    right_face = q + 0.5 * sx
    bottom_face = q - 0.5 * sy
    top_face = q + 0.5 * sy

    qx_plus, qx_minus = torch.maximum(qx, torch.zeros_like(qx)), torch.minimum(qx, torch.zeros_like(qx))
    qy_plus, qy_minus = torch.maximum(qy, torch.zeros_like(qy)), torch.minimum(qy, torch.zeros_like(qy))
    left_source = torch.cat((zero_x, right_face[:, :-1]), dim=1)
    right_source = torch.cat((left_face[:, 1:], zero_x), dim=1)
    bottom_source = torch.cat((zero_y, top_face[:-1, :]), dim=0)
    top_source = torch.cat((bottom_face[1:, :], zero_y), dim=0)
    incoming = (
        qx_plus[:, :-1] * left_source - qx_minus[:, 1:] * right_source
        + qy_plus[:-1, :] * bottom_source - qy_minus[1:, :] * top_source
    )
    outgoing = (
        qx_plus[:, 1:] * right_face - qx_minus[:, :-1] * left_face
        + qy_plus[1:, :] * top_face - qy_minus[:-1, :] * bottom_face
    )
    updated = q - dt * (outgoing / area) + dt * (incoming / area)
    boundary_outflow = (
        (-qx_minus[:, 0] * left_face[:, 0]).sum()
        + (qx_plus[:, -1] * right_face[:, -1]).sum()
        + (-qy_minus[0, :] * bottom_face[0, :]).sum()
        + (qy_plus[-1, :] * top_face[-1, :]).sum()
    )
    return updated, boundary_outflow


def _check_flow(qx: Tensor, qy: Tensor, height: int, width: int) -> None:
    if qx.shape != (height, width + 1):
        raise ValueError(f"qx must have shape {(height, width + 1)}")
    if qy.shape != (height + 1, width):
        raise ValueError(f"qy must have shape {(height + 1, width)}")
    divergence = qx[:, 1:] - qx[:, :-1] + qy[1:, :] - qy[:-1, :]
    scale = torch.abs(qx[:, 1:]) + torch.abs(qx[:, :-1]) + torch.abs(qy[1:, :]) + torch.abs(qy[:-1, :])
    if not bool(torch.isfinite(divergence).all()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("discrete divergence is not finite")
    tolerance = 256 * torch.finfo(qx.dtype).eps * scale
    if bool((torch.abs(divergence) > tolerance).any()):
        raise ValueError("qx and qy must satisfy discrete divergence zero")


def muscl_step(
    echo: Tensor,
    qx: Tensor,
    qy: Tensor,
    *,
    dt_seconds: float,
    spacing_yx: Tuple[float, float],
    log_growth: Tensor | float = 0,
) -> Tuple[Tensor, Tensor]:
    """Advance one constant-growth SSPRK2 minmod-MUSCL step.

    The exterior trace is zero.  The second result is transformed (r-space)
    boundary outflow integrated with the SSPRK2 weights.
    """
    echo = _tensor("echo", echo)
    if echo.ndim != 2 or not all(echo.shape):
        raise ValueError("echo must be a non-empty [H, W] grid")
    _check_nonnegative(echo, "echo")
    qx = _tensor("qx", qx, dtype=echo.dtype)
    qy = _tensor("qy", qy, dtype=echo.dtype)
    height, width = echo.shape
    _check_flow(qx, qy, height, width)

    if isinstance(dt_seconds, bool) or not isinstance(dt_seconds, Real):
        raise TypeError("dt_seconds must be a positive finite float")
    dt_value = float(dt_seconds)
    if not math.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("dt_seconds must be a positive finite float")
    if not isinstance(spacing_yx, (tuple, list)) or len(spacing_yx) != 2:
        raise ValueError("spacing_yx must be (dy, dx)")
    dy, dx = float(spacing_yx[0]), float(spacing_yx[1])
    if not math.isfinite(dy) or not math.isfinite(dx) or dy <= 0 or dx <= 0:
        raise ValueError("spacing_yx must contain finite positive floats")
    area = echo.new_tensor(dy * dx)
    if not bool(torch.isfinite(area)) or not bool(area > 0):
        raise ValueError("cell area is not representable in the input dtype")
    dt = echo.new_tensor(dt_value)
    if not bool(torch.isfinite(dt)) or not bool(dt > 0):
        raise ValueError("dt_seconds is not representable in the input dtype")

    qx_plus, qx_minus = torch.maximum(qx, torch.zeros_like(qx)), torch.minimum(qx, torch.zeros_like(qx))
    qy_plus, qy_minus = torch.maximum(qy, torch.zeros_like(qy)), torch.minimum(qy, torch.zeros_like(qy))
    outgoing_rate = (qx_plus[:, 1:] - qx_minus[:, :-1] + qy_plus[1:, :] - qy_minus[:-1, :]) / area
    if not bool(torch.isfinite(outgoing_rate).all()):
        raise FloatingPointError("outgoing CFL is not finite")
    cfl = dt * outgoing_rate.max()
    if not bool(torch.isfinite(cfl)) or cfl.detach().item() > 0.5:
        raise ValueError("outgoing CFL exceeds 0.5")

    if isinstance(log_growth, Tensor):
        growth_log = _tensor("log_growth", log_growth)
        if growth_log.ndim != 0:
            raise ValueError("log_growth tensor must be scalar")
        if growth_log.dtype != echo.dtype:
            growth_log = growth_log.to(dtype=echo.dtype)
    elif isinstance(log_growth, Real) and not isinstance(log_growth, bool):
        value = float(log_growth)
        if not math.isfinite(value):
            raise ValueError("log_growth must be finite")
        growth_log = echo.new_tensor(value)
    else:
        raise TypeError("log_growth must be a finite scalar tensor or float")
    growth = torch.exp(growth_log)
    inverse_growth = torch.exp(-growth_log)
    if not bool(torch.isfinite(growth)) or not bool(torch.isfinite(inverse_growth)):
        raise FloatingPointError("growth exponential is not finite")

    r1, out0 = _euler_muscl(echo, qx, qy, dt, area)
    _check_finite_nonnegative(r1, "stage-1 echo")
    r2, out1 = _euler_muscl(r1, qx, qy, dt, area)
    _check_finite_nonnegative(r2, "stage-2 echo")
    echo_new = growth * (0.5 * echo + 0.5 * r2)
    transformed_outflow = dt * (0.5 * out0 + 0.5 * out1)
    _check_finite_nonnegative(echo_new, "echo result")
    if not bool(torch.isfinite(transformed_outflow)) or not bool(transformed_outflow >= 0):
        raise FloatingPointError("transformed outflow is not finite and nonnegative")
    return echo_new, transformed_outflow


def _check_nonnegative(value: Tensor, name: str) -> None:
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be nonnegative")


def _check_finite_nonnegative(value: Tensor, name: str) -> None:
    _tensor(name, value)
    _check_nonnegative(value, name)
