"""Finite-window motion diagnostics for saved dBZ fields.

The diagnostic treats ``background_echo`` as a uniform q-space background.  A
cell contributes ``max(dbz_to_echo(dbz) - background_echo - excess_threshold,
0)``.  The threshold is in q units and only removes background/roundoff-sized
excess; it is not a detection or rotation decision threshold.

Coordinates are cell-centre coordinates relative to the geometric centre of
the supplied window.  The returned moments therefore describe what is visible
in this particular window.  A moving pattern that is clipped by the window
can change its moments even when the underlying motion is rigid.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from advar.physics import dbz_to_echo


BASE_ECHO = 100.0


def _none_result(reason: str, *, background_echo: float, excess_threshold: float) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "background_echo": background_echo,
        "excess_threshold": excess_threshold,
        "centroid_xy_m": None,
        "reference_centroid_xy_m": None,
        "radius_m": None,
        "reference_radius_m": None,
        "polar_change_deg": None,
        "axis_angle_deg": None,
        "reference_axis_angle_deg": None,
        "axis_change_deg": None,
        "anisotropy": None,
        "reference_anisotropy": None,
        "widths_m": None,
        "reference_widths_m": None,
    }


def _coordinates(
    shape: tuple[int, int],
    spacing_m: float | tuple[float, float],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    height, width = shape
    if isinstance(spacing_m, tuple):
        if len(spacing_m) != 2:
            raise ValueError("spacing_m tuple must be (dy, dx)")
        dy, dx = (float(spacing_m[0]), float(spacing_m[1]))
    else:
        dy = dx = float(spacing_m)
    if not math.isfinite(dy) or not math.isfinite(dx) or dy <= 0.0 or dx <= 0.0:
        raise ValueError("spacing_m must be positive and finite")
    y = (torch.arange(height, dtype=dtype, device=device) + 0.5 - height / 2.0) * dy
    x = (torch.arange(width, dtype=dtype, device=device) + 0.5 - width / 2.0) * dx
    return torch.meshgrid(y, x, indexing="ij")


def _moment(weights: Tensor, x: Tensor, y: Tensor) -> dict[str, Any] | None:
    total = weights.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        return None
    centroid_x = (weights * x).sum() / total
    centroid_y = (weights * y).sum() / total
    dx = x - centroid_x
    dy = y - centroid_y
    cxx = (weights * dx * dx).sum() / total
    cxy = (weights * dx * dy).sum() / total
    cyy = (weights * dy * dy).sum() / total
    covariance = torch.stack((torch.stack((cxx, cxy)), torch.stack((cxy, cyy))))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)
    trace = float(eigenvalues.sum())
    # A point has no covariance orientation.  An elongated line does have a
    # stable major axis, even though its minor width is zero.
    eigen_gap = float(eigenvalues[1] - eigenvalues[0])
    scale_tolerance = 64.0 * torch.finfo(weights.dtype).eps * trace
    if (
        not math.isfinite(trace)
        or trace <= scale_tolerance
        or eigen_gap <= scale_tolerance
    ):
        anisotropy = None
        widths = None
        axis_angle = None
    else:
        minor, major = float(eigenvalues[0]), float(eigenvalues[1])
        major_vector = eigenvectors[:, 1]
        axis_angle = math.degrees(math.atan2(float(major_vector[1]), float(major_vector[0])))
        # Orientation is an axis, hence differences are modulo 180 degrees.
        anisotropy = (major - minor) / (major + minor)
        widths = [math.sqrt(major), math.sqrt(minor)]
    radius = math.hypot(float(centroid_x), float(centroid_y))
    return {
        "centroid_xy_m": [float(centroid_x), float(centroid_y)],
        "radius_m": radius,
        "axis_angle_deg": axis_angle,
        "anisotropy": anisotropy,
        "widths_m": widths,
    }


def _wrap_axis_degrees(angle: float) -> float:
    return (angle + 90.0) % 180.0 - 90.0


def _wrap_polar_degrees(reference: list[float], current: list[float], radius_tolerance: float) -> float | None:
    reference_angle = math.atan2(reference[1], reference[0])
    current_angle = math.atan2(current[1], current[0])
    reference_radius = math.hypot(*reference)
    current_radius = math.hypot(*current)
    if reference_radius <= radius_tolerance or current_radius <= radius_tolerance:
        return None
    return math.degrees(math.atan2(math.sin(current_angle - reference_angle), math.cos(current_angle - reference_angle)))


def motion_diagnostics(
    reference_grid: Tensor,
    field: Tensor,
    spacing_m: float | tuple[float, float],
    background_echo: float = BASE_ECHO,
    min_dbz: float = -20.0,
    *,
    max_dbz: float | None = None,
    excess_threshold: float = 1.0e-10,
) -> dict[str, Any]:
    """Return comparable centroid, radial, and covariance-shape diagnostics.

    ``reference_grid`` and ``field`` are dBZ tensors with identical 2-D shape.
    Inputs containing NaN or infinity are reported as unavailable because a
    missing saved frame must not silently become clear sky.  The result has no
    success boolean: callers should compare the reported values with a known
    same-window oracle when a rotation claim is needed.
    """
    if not isinstance(reference_grid, Tensor) or not isinstance(field, Tensor):
        return _none_result("inputs_must_be_tensors", background_echo=float(background_echo), excess_threshold=float(excess_threshold))
    background_echo = float(background_echo)
    excess_threshold = float(excess_threshold)
    if not math.isfinite(background_echo) or not math.isfinite(excess_threshold) or excess_threshold < 0.0:
        return _none_result("invalid_background_or_threshold", background_echo=background_echo, excess_threshold=excess_threshold)
    if reference_grid.ndim != 2 or field.ndim != 2 or tuple(reference_grid.shape) != tuple(field.shape):
        return _none_result("incomplete_or_shape_mismatch", background_echo=background_echo, excess_threshold=excess_threshold)
    if not reference_grid.is_floating_point() or not field.is_floating_point():
        return _none_result("inputs_must_be_floating_point", background_echo=background_echo, excess_threshold=excess_threshold)
    if not bool(torch.isfinite(reference_grid).all()) or not bool(torch.isfinite(field).all()):
        return _none_result("incomplete_nonfinite_frame", background_echo=background_echo, excess_threshold=excess_threshold)
    try:
        y, x = _coordinates(
            tuple(reference_grid.shape),
            spacing_m,
            reference_grid.dtype,
            reference_grid.device,
        )
    except (TypeError, ValueError):
        return _none_result("invalid_spacing", background_echo=background_echo, excess_threshold=excess_threshold)
    reference_q = dbz_to_echo(reference_grid, min_dbz=float(min_dbz), max_dbz=max_dbz)
    field_q = dbz_to_echo(field, min_dbz=float(min_dbz), max_dbz=max_dbz)
    reference_weights = (reference_q - background_echo - excess_threshold).clamp_min(0.0)
    field_weights = (field_q - background_echo - excess_threshold).clamp_min(0.0)
    reference_moment = _moment(reference_weights, x, y)
    field_moment = _moment(field_weights, x, y)
    if reference_moment is None or field_moment is None:
        return _none_result("zero_nonnegative_excess", background_echo=background_echo, excess_threshold=excess_threshold)

    result: dict[str, Any] = {
        "status": "ok",
        "reason": None,
        "background_echo": background_echo,
        "excess_threshold": excess_threshold,
        **reference_moment,
    }
    result["reference_centroid_xy_m"] = reference_moment["centroid_xy_m"]
    result["reference_radius_m"] = reference_moment["radius_m"]
    result["reference_axis_angle_deg"] = reference_moment["axis_angle_deg"]
    result["reference_anisotropy"] = reference_moment["anisotropy"]
    result["reference_widths_m"] = reference_moment["widths_m"]
    result.update(field_moment)
    result["polar_change_deg"] = _wrap_polar_degrees(
        reference_moment["centroid_xy_m"], field_moment["centroid_xy_m"],
        64.0 * torch.finfo(reference_grid.dtype).eps * float(torch.maximum(x.abs().max(), y.abs().max()))
    )
    if reference_moment["axis_angle_deg"] is None or field_moment["axis_angle_deg"] is None:
        result["axis_change_deg"] = None
        result["status"] = "partial"
        result["reason"] = "degenerate_covariance_orientation"
    else:
        result["axis_change_deg"] = _wrap_axis_degrees(
            field_moment["axis_angle_deg"] - reference_moment["axis_angle_deg"]
        )
    if result["polar_change_deg"] is None:
        result["status"] = "partial"
        if result["reason"] is None:
            result["reason"] = "centroid_at_window_origin"
    return result
