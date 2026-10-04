"""Coordinate-aware diagnostics for a full-rank chart Jacobian."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def tangent_metric_diagnostics(
    jacobian: Tensor,
    gradient: Tensor,
    *,
    hessian: Tensor | None = None,
) -> dict[str, Any]:
    """Measure the ambient-Euclidean tangent gradient in chart coordinates.

    ``jacobian`` is ``d Gamma / d t`` with shape ``[ambient, tangent]`` and
    ``gradient`` is the tangent-coordinate gradient ``j_t``. If supplied,
    ``hessian`` is the Hessian of the already-composed objective in chart
    coordinates; callers must include the chart-curvature term from AD.

    A rank-deficient chart is refused. A returned Hessian spectrum is a
    coordinate-normalized tangent diagnostic, not an SPD or root certificate.
    """
    if (not isinstance(jacobian, Tensor) or jacobian.device.type != "cpu"
            or jacobian.dtype != torch.float64 or jacobian.ndim != 2):
        raise ValueError("jacobian must be a CPU FP64 matrix")
    ambient, tangent = jacobian.shape
    if ambient < tangent or tangent < 1:
        raise ValueError("jacobian must have ambient >= tangent > 0")
    if (not isinstance(gradient, Tensor) or gradient.device.type != "cpu"
            or gradient.dtype != torch.float64 or gradient.shape != (tangent,)):
        raise ValueError("gradient must be a CPU FP64 vector matching the tangent dimension")
    if not bool(torch.isfinite(jacobian).all() & torch.isfinite(gradient).all()):
        raise ValueError("jacobian and tangent gradient must be finite")
    if hessian is not None and (
        not isinstance(hessian, Tensor) or hessian.device.type != "cpu"
        or hessian.dtype != torch.float64 or hessian.shape != (tangent, tangent)
    ):
        raise ValueError("hessian must be a CPU FP64 matrix matching the tangent dimension")
    if hessian is not None and not bool(torch.isfinite(hessian).all()):
        raise ValueError("hessian must be finite")

    singular_values = torch.linalg.svdvals(jacobian)
    if not bool(torch.isfinite(singular_values).all()):
        raise ValueError("jacobian singular values are nonfinite")
    largest = singular_values[0]
    rank_tolerance = torch.finfo(jacobian.dtype).eps * max(ambient, tangent) * largest
    if not bool(torch.isfinite(rank_tolerance)):
        raise ValueError("jacobian rank threshold is nonfinite")
    numerical_rank = int((singular_values > rank_tolerance).sum())
    if numerical_rank != tangent:
        raise ValueError("jacobian is numerically rank deficient at the FP64 scale")

    q, r = torch.linalg.qr(jacobian, mode="reduced")
    if not bool(torch.isfinite(q).all() & torch.isfinite(r).all()):
        raise ValueError("QR factorization of jacobian is nonfinite")
    tangent_gradient = torch.linalg.solve_triangular(
        r.mT, gradient.unsqueeze(1), upper=False
    ).squeeze(1)
    tangent_norm = torch.linalg.vector_norm(tangent_gradient)
    condition = singular_values[0] / singular_values[-1]
    if not bool(torch.isfinite(tangent_gradient).all() & torch.isfinite(tangent_norm)
                & torch.isfinite(condition)):
        raise ValueError("tangent metric diagnostics overflowed to a nonfinite result")

    result: dict[str, Any] = {
        "scope": "original-standardized-control Euclidean tangent metric",
        "tangent_gradient_norm": tangent_norm,
        "raw_tangent_gradient_inf_norm": gradient.abs().amax(),
        "jacobian_rank": numerical_rank,
        "rank_tolerance": rank_tolerance,
        "singular_values": singular_values,
        "condition_number": condition,
        "hessian_scope": None,
        "hessian_asymmetry_relative": None,
        "coordinate_normalized_hessian_eigenvalues": None,
    }
    if hessian is not None:
        scale = torch.linalg.vector_norm(hessian)
        asymmetry = torch.linalg.vector_norm(hessian - hessian.mT)
        if not bool(torch.isfinite(scale) & torch.isfinite(asymmetry)):
            raise ValueError("hessian scale or asymmetry overflowed to a nonfinite result")
        asymmetry_tolerance = 64 * torch.finfo(hessian.dtype).eps * tangent * scale
        if not bool(torch.isfinite(asymmetry_tolerance)):
            raise ValueError("hessian asymmetry tolerance is nonfinite")
        if not bool(asymmetry <= asymmetry_tolerance):
            raise ValueError("hessian asymmetry exceeds the FP64 component-scale tolerance")
        asymmetry_relative = asymmetry / scale if bool(scale > 0) else scale
        symmetric = 0.5 * (hessian + hessian.mT)
        left = torch.linalg.solve_triangular(r.mT, symmetric, upper=False)
        normalized = torch.linalg.solve_triangular(r.mT, left.mT, upper=False).mT
        normalized = 0.5 * (normalized + normalized.mT)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("coordinate-normalized hessian overflowed to a nonfinite result")
        eigenvalues = torch.linalg.eigvalsh(normalized)
        if not bool(torch.isfinite(eigenvalues).all()):
            raise ValueError("coordinate-normalized hessian eigenvalues are nonfinite")
        result.update(
            hessian_scope=(
                "coordinate-normalized tangent Hessian; nonlinear-chart invariance "
                "only at stationarity; no SPD or root certification"
            ),
            hessian_asymmetry_relative=asymmetry_relative,
            hessian_asymmetry_tolerance=asymmetry_tolerance,
            coordinate_normalized_hessian_eigenvalues=eigenvalues,
        )
    return result
