"""Conditional matrix-free stationarity refinement for research adapters.

The caller supplies the objective's fixed branch check.  This module performs
only a local inexact-Newton root solve; it makes no minimum, path, or global
positive-definiteness claim.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

import torch
from torch import Tensor

from .matrix_free import pcg


_STATIONARITY_TOLERANCE = 1.0e-10
_PCG_RELATIVE_TOLERANCE = 1.0e-10
_ARMIJO_CONSTANT = 1.0e-4


class _NonFiniteEvaluation(ValueError):
    """A numerically invalid objective evaluation at a candidate control."""


@dataclass(frozen=True)
class RefinementResult:
    """Result of a conditional local stationarity refinement."""

    control: Tensor
    gradient_max: float
    iterations: int
    hvp_count: int
    history: list[dict[str, Any]]


def _require_vector(
    name: str,
    value: Tensor,
    *,
    dtype: torch.dtype | None = None,
    require_finite: bool = True,
) -> None:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional tensor")
    if dtype is not None and value.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if require_finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _finite_scalar(name: str, value: Tensor) -> float:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must return a scalar tensor")
    if not bool(torch.isfinite(value)):
        raise _NonFiniteEvaluation(f"{name} must be finite")
    return float(value)


def _branch(
    branch_check: Callable[[Tensor, Tensor], tuple[Any, str]],
    control: Tensor,
    parameters: Tensor,
) -> tuple[Any, str]:
    if not callable(branch_check):
        raise TypeError("branch_check is required")
    result = branch_check(control, parameters)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("branch_check must return (signature, scope)")
    signature, scope = result
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("branch_check scope must be a nonempty string")
    return signature, scope


def _validate_option(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _evaluate(
    objective: Callable[[Tensor, Tensor], Tensor],
    gradient: Callable[[Tensor, Tensor], Tensor],
    control: Tensor,
    parameters: Tensor,
) -> tuple[float, Tensor, float, float]:
    objective_value = _finite_scalar("objective", objective(control, parameters))
    current_gradient = gradient(control, parameters)
    _require_vector(
        "objective gradient", current_gradient, dtype=control.dtype, require_finite=False
    )
    if current_gradient.shape != control.shape:
        raise ValueError("objective gradient shape must match control")
    if not bool(torch.isfinite(current_gradient).all()):
        raise _NonFiniteEvaluation("objective gradient must be finite")
    gradient_norm = float(torch.linalg.vector_norm(current_gradient))
    gradient_max = float(current_gradient.abs().max())
    if not math.isfinite(gradient_norm) or not math.isfinite(gradient_max):
        raise _NonFiniteEvaluation("objective gradient norm must be finite")
    return objective_value, current_gradient, gradient_norm, gradient_max


def refine_stationary(
    objective: Callable[[Tensor, Tensor], Tensor],
    control: Tensor,
    p: Tensor,
    *,
    branch_check: Callable[[Tensor, Tensor], tuple[Any, str]],
    max_iterations: int = 8,
    max_backtracks: int = 16,
    pcg_max_iterations: int | None = None,
) -> RefinementResult:
    """Refine a nearby stationary control with exact matrix-free Newton steps.

    Each Newton equation uses exact gradient-JVP products and PCG.  The
    objective is used only to validate finite evaluations; acceptance is based
    on the gradient merit ``||g||²/2`` and its measured Newton slope.  The
    objective and branch callback must be deterministic and must not mutate
    their tensor arguments; the copied parameters protect the caller tensor,
    but cannot isolate shared callback state.  A
    ``ValueError`` from the branch checker rejects one trial; a ``RuntimeError``
    from it propagates to the caller.  Detected nonfinite candidate controls,
    objectives, gradients, and gradient norms reject that trial and continue
    backtracking; malformed outputs and callback exceptions propagate.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    _require_vector("control", control)
    _require_vector("p", p)
    if control.dtype is not torch.float64 or p.dtype is not torch.float64:
        raise TypeError("local refinement requires float64 control and parameters")
    if control.device.type != "cpu" or p.device != control.device:
        raise TypeError("local refinement requires CPU control and parameters")
    max_iterations_value = _validate_option("max_iterations", max_iterations)
    max_backtracks_value = _validate_option("max_backtracks", max_backtracks)
    if pcg_max_iterations is None:
        pcg_limit = 4 * control.numel()
    else:
        pcg_limit = _validate_option("pcg_max_iterations", pcg_max_iterations)

    parameters = p.detach().clone()
    current = control.detach().clone()
    gradient = torch.func.grad(objective, argnums=0)
    _branch(branch_check, current, parameters)
    _, current_gradient, gradient_norm, gradient_max = _evaluate(
        objective, gradient, current, parameters
    )
    if gradient_max < _STATIONARITY_TOLERANCE:
        return RefinementResult(
            control=current,
            gradient_max=gradient_max,
            iterations=0,
            hvp_count=0,
            history=[],
        )

    history: list[dict[str, Any]] = []
    hvp_count = 0
    for iteration in range(1, max_iterations_value + 1):
        def hessian_vector(direction: Tensor) -> Tensor:
            nonlocal hvp_count
            hvp_count += 1
            result = torch.func.jvp(
                lambda c: gradient(c, parameters), (current,), (direction,)
            )[1]
            _require_vector("HVP", result, dtype=current.dtype)
            if result.shape != current.shape:
                raise ValueError("HVP shape must match control")
            return result

        solve = pcg(
            hessian_vector,
            -current_gradient,
            rtol=_PCG_RELATIVE_TOLERANCE,
            max_iterations=pcg_limit,
        )
        if not solve.converged:
            raise RuntimeError("matrix-free Newton PCG did not converge")

        step = solve.solution
        _require_vector("Newton step", step, dtype=current.dtype)
        hessian_step = hessian_vector(step)
        linear_residual = hessian_step + current_gradient
        linear_relative = float(torch.linalg.vector_norm(linear_residual)) / gradient_norm
        if not math.isfinite(linear_relative) or linear_relative > _PCG_RELATIVE_TOLERANCE:
            raise RuntimeError("matrix-free Newton PCG true residual exceeds tolerance")
        normalized_slope = float(
            torch.dot(current_gradient / gradient_norm, hessian_step / gradient_norm)
        )
        if not math.isfinite(normalized_slope):
            raise RuntimeError("Newton merit slope is nonfinite")
        if normalized_slope >= 0.0:
            raise RuntimeError("Newton step is not a descent direction for gradient merit")

        accepted = False
        last_trial_norm: float | None = None
        for backtrack in range(max_backtracks_value):
            scale = 0.5**backtrack
            candidate = current + scale * step
            if not bool(torch.isfinite(candidate).all()):
                history.append({
                    "iteration": iteration,
                    "backtrack": backtrack,
                    "step_scale": scale,
                    "accepted": False,
                    "rejection": "nonfinite_candidate",
                    "hvp_count": hvp_count,
                })
                continue
            try:
                branch_signature, _ = _branch(branch_check, candidate, parameters)
            except ValueError:
                history.append({
                    "iteration": iteration,
                    "backtrack": backtrack,
                    "step_scale": scale,
                    "accepted": False,
                    "rejection": "branch",
                    "hvp_count": hvp_count,
                })
                continue
            try:
                candidate_objective, candidate_gradient, candidate_norm, candidate_max = _evaluate(
                    objective, gradient, candidate, parameters
                )
            except _NonFiniteEvaluation:
                history.append({
                    "iteration": iteration,
                    "backtrack": backtrack,
                    "step_scale": scale,
                    "accepted": False,
                    "rejection": "nonfinite_candidate",
                    "hvp_count": hvp_count,
                })
                continue
            last_trial_norm = candidate_norm
            norm_ratio = candidate_norm / gradient_norm
            armijo_limit_squared = 1.0 + 2.0 * _ARMIJO_CONSTANT * scale * normalized_slope
            armijo_limit = math.sqrt(max(0.0, armijo_limit_squared))
            armijo_ratio = norm_ratio / armijo_limit if armijo_limit > 0.0 else math.inf
            finite_trial = (
                math.isfinite(candidate_objective)
                and math.isfinite(candidate_norm)
                and math.isfinite(candidate_max)
                and math.isfinite(norm_ratio)
                and math.isfinite(armijo_ratio)
            )
            record = {
                "iteration": iteration,
                "backtrack": backtrack,
                "step_scale": scale,
                "accepted": bool(finite_trial and armijo_ratio <= 1.0),
                "objective": candidate_objective,
                "gradient_norm": candidate_norm,
                "gradient_max": candidate_max,
                "normalized_slope": normalized_slope,
                "norm_ratio": norm_ratio,
                "armijo_limit": armijo_limit,
                "armijo_ratio": armijo_ratio,
                "linear_relative_residual": linear_relative,
                "pcg_relative_residual": solve.relative_residual,
                "pcg_iterations": solve.iterations,
                "hvp_count": hvp_count,
                "branch": branch_signature,
            }
            history.append(record)
            if record["accepted"]:
                current = candidate.detach().clone()
                current_gradient = candidate_gradient
                gradient_norm = candidate_norm
                gradient_max = candidate_max
                accepted = True
                break
        if not accepted:
            trials = [record for record in history if record["iteration"] == iteration]
            branch_rejections = sum(record.get("rejection") == "branch" for record in trials)
            nonfinite_rejections = sum(
                record.get("rejection") == "nonfinite_candidate" for record in trials
            )
            finite_armijo_rejections = len(trials) - branch_rejections - nonfinite_rejections
            trial_detail = (
                f"last_trial_gradient_norm={last_trial_norm}"
                if last_trial_norm is not None
                else "no finite candidate evaluation"
            )
            raise RuntimeError(
                "stationarity refinement failed to find an Armijo step: "
                f"iteration={iteration}; {trial_detail}; "
                f"branch_rejections={branch_rejections}; "
                f"nonfinite_candidate_rejections={nonfinite_rejections}; "
                f"finite_armijo_rejections={finite_armijo_rejections}"
            )
        if gradient_max < _STATIONARITY_TOLERANCE:
            return RefinementResult(
                control=current,
                gradient_max=gradient_max,
                iterations=iteration,
                hvp_count=hvp_count,
                history=history,
            )

    raise RuntimeError(
        "stationarity refinement iteration budget exhausted: "
        f"gradient_max={gradient_max}"
    )
