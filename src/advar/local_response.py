"""Small matrix-free local stationary-response oracle for research adapters.

The caller owns the physical model's branch proof.  This module only solves the
conditional implicit derivative after a caller-supplied branch check succeeds.
It is intentionally independent of the public FV, learning, and FSOI APIs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .matrix_free import pcg, vjp


_STATIONARITY_TOLERANCE = 1.0e-10
_PCG_RTOL = 1.0e-10
_PCG_MAX_ITERATIONS = 104


@dataclass(frozen=True)
class LocalResponse:
    """Conditional local response at a supplied stationary control."""

    direct: dict[str, Tensor]
    indirect: dict[str, Tensor]
    total: dict[str, Tensor]
    mixed_gradients: dict[str, Tensor]
    score_control_gradient: Tensor
    adjoint: Tensor
    gradient_max: float
    true_adjoint_residual: float
    true_adjoint_relative_residual: float
    pcg_relative_residual: float
    pcg_iterations: int
    hvp_count: int
    branch_signature: Any
    scope: str
    input_identity: dict[str, Any]


def _require_vector(name: str, value: Tensor, *, dtype: torch.dtype | None = None) -> None:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional tensor")
    if dtype is not None and value.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _scalar(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must return a scalar tensor")
    if not bool(torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _branch_result(branch_check: Callable[[Tensor, Tensor], tuple[Any, str]], control: Tensor, p: Tensor) -> tuple[Any, str]:
    if not callable(branch_check):
        raise TypeError("branch_check is required")
    try:
        result = branch_check(control, p)
    except Exception as error:
        raise ValueError("branch check failed") from error
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("branch_check must return (signature, scope)")
    signature, scope = result
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("branch_check scope must be a nonempty string")
    return signature, f"conditional on caller-supplied branch_check: {scope}"


def compute_local_response(
    objective: Callable[[Tensor, Tensor], Tensor],
    score: Callable[[Tensor, Tensor], Tensor],
    control: Tensor,
    p: Tensor,
    directions: Mapping[str, Tensor],
    *,
    branch_check: Callable[[Tensor, Tensor], tuple[Any, str]],
    input_identity: Mapping[str, Any],
) -> LocalResponse:
    """Compute exact matrix-free stationary responses for fixed local branches.

    ``objective(control, p)`` is differentiated twice through HVPs.  The score
    may have a direct parameter derivative.  ``branch_check`` must inspect the
    actual adapter trajectory and return ``(signature, scope)``; this function
    does not infer branch validity from endpoint arithmetic.
    """
    if not callable(objective) or not callable(score):
        raise TypeError("objective and score must be callable")
    _require_vector("control", control)
    _require_vector("p", p, dtype=control.dtype)
    if control.dtype is not torch.float64 or p.dtype is not torch.float64:
        raise TypeError("local response requires float64 control and parameters")
    if control.device != p.device or control.device.type != "cpu":
        raise TypeError("local response requires CPU tensors")
    if not isinstance(input_identity, Mapping) or not input_identity:
        raise TypeError("input_identity must be an explicit nonempty mapping")
    if not isinstance(directions, Mapping) or not directions:
        raise ValueError("directions must be a nonempty mapping")

    branch_signature, scope = _branch_result(branch_check, control, p)
    _scalar("objective", objective(control, p))
    _scalar("score", score(control, p))
    gradient = torch.func.grad(objective, argnums=0)
    g = gradient(control, p)
    _require_vector("objective gradient", g, dtype=control.dtype)
    gradient_max = float(g.abs().max())
    if gradient_max >= _STATIONARITY_TOLERANCE:
        raise ValueError("control is not stationary at the fixed response gate")

    rhs, direct_gradient = torch.func.grad(score, argnums=(0, 1))(control, p)
    _require_vector("score control gradient", rhs, dtype=control.dtype)
    _require_vector("score parameter gradient", direct_gradient, dtype=p.dtype)

    products = 0

    def hessian_vector(direction: Tensor) -> Tensor:
        nonlocal products
        products += 1
        result = torch.func.jvp(lambda c: gradient(c, p), (control,), (direction,))[1]
        _require_vector("HVP", result, dtype=control.dtype)
        return result

    solve = pcg(
        hessian_vector,
        rhs,
        rtol=_PCG_RTOL,
        max_iterations=_PCG_MAX_ITERATIONS,
    )
    if not solve.converged:
        raise RuntimeError("matrix-free adjoint PCG did not converge")
    adjoint = solve.solution
    _require_vector("adjoint", adjoint, dtype=control.dtype)

    # Independently form H^T lambda through a VJP of the control gradient.  No
    # dense Hessian or implicit symmetry assumption is used for this residual.
    _, transpose_product = vjp(lambda c: gradient(c, p), control, adjoint)
    _require_vector("transposed Hessian product", transpose_product, dtype=control.dtype)
    rhs_norm = float(rhs.norm())
    true_residual = float((transpose_product - rhs).norm())
    if rhs_norm == 0.0:
        if true_residual != 0.0:
            raise RuntimeError("true adjoint residual is nonzero for zero rhs")
        true_relative = 0.0
    else:
        true_relative = true_residual / rhs_norm
    if not torch.isfinite(torch.tensor(true_relative)):
        raise RuntimeError("true adjoint residual is nonfinite")
    if true_relative > _PCG_RTOL:
        raise RuntimeError("true adjoint residual exceeds PCG tolerance")

    direct: dict[str, Tensor] = {}
    indirect: dict[str, Tensor] = {}
    total: dict[str, Tensor] = {}
    mixed_gradients: dict[str, Tensor] = {}
    for name, direction in directions.items():
        if not isinstance(name, str) or not name:
            raise TypeError("direction names must be nonempty strings")
        _require_vector(f"direction {name}", direction, dtype=p.dtype)
        if direction.shape != p.shape or direction.device != p.device:
            raise TypeError(f"direction {name} must match p")
        if not bool(torch.isfinite(direction).all()):
            raise ValueError(f"direction {name} must be finite")
        cross = torch.func.jvp(
            lambda q: gradient(control, q), (p,), (direction,)
        )[1]
        _require_vector(f"mixed gradient {name}", cross, dtype=control.dtype)
        direct_value = torch.dot(direct_gradient, direction)
        indirect_value = -torch.dot(adjoint, cross)
        total_value = direct_value + indirect_value
        for label, value in (("direct", direct_value), ("indirect", indirect_value), ("total", total_value)):
            _scalar(f"{label} response {name}", value)
        mixed_gradients[name] = cross
        direct[name], indirect[name], total[name] = direct_value, indirect_value, total_value

    return LocalResponse(
        direct=direct,
        indirect=indirect,
        total=total,
        mixed_gradients=mixed_gradients,
        score_control_gradient=rhs,
        adjoint=adjoint,
        gradient_max=gradient_max,
        true_adjoint_residual=true_residual,
        true_adjoint_relative_residual=true_relative,
        pcg_relative_residual=solve.relative_residual,
        pcg_iterations=solve.iterations,
        hvp_count=products,
        branch_signature=branch_signature,
        scope=scope,
        input_identity=dict(input_identity),
    )
