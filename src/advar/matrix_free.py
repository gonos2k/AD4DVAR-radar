"""Small matrix-free automatic-differentiation and linear-solve helpers."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import cast

import torch
from torch import Tensor


TensorFunction = Callable[[Tensor], Tensor]


def _check_real_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a real floating-point dtype")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")


def _check_finite_tensor(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _check_like(name: str, value: Tensor, reference: Tensor) -> None:
    _check_real_tensor(name, value)
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} has shape {tuple(value.shape)}, "
            f"expected {tuple(reference.shape)}"
        )
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must have the same dtype and device as the input")


def _nonnegative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _inner(left: Tensor, right: Tensor) -> Tensor:
    return torch.sum(left * right)


def _pcg_reduction_dtype(value: Tensor) -> torch.dtype:
    # MPS does not support float64. FP32 still prevents FP16/BF16 reductions
    # from overflowing for the intended control-vector sizes.
    return torch.float32 if value.device.type == "mps" else torch.float64


def _maximum_absolute(value: Tensor) -> float:
    return float(torch.amax(torch.abs(value.detach())).cpu())


@dataclass(frozen=True)
class _ScaledInnerProduct:
    sign: int
    log_absolute: float

    @property
    def is_positive_finite(self) -> bool:
        return self.sign > 0 and math.isfinite(self.log_absolute)


def _pcg_inner(left: Tensor, right: Tensor) -> _ScaledInnerProduct:
    """Return an inner product without reconstructing extreme magnitudes."""

    left_scale = _maximum_absolute(left)
    right_scale = _maximum_absolute(right)
    if left_scale == 0.0 or right_scale == 0.0:
        return _ScaledInnerProduct(sign=0, log_absolute=-math.inf)

    dtype = _pcg_reduction_dtype(left)
    normalized = torch.sum(
        (left / left_scale).to(dtype) * (right / right_scale).to(dtype)
    )
    normalized_value = float(normalized.detach().cpu())
    if normalized_value == 0.0:
        return _ScaledInnerProduct(sign=0, log_absolute=-math.inf)
    if not math.isfinite(normalized_value):
        return _ScaledInnerProduct(sign=0, log_absolute=math.nan)

    return _ScaledInnerProduct(
        sign=1 if normalized_value > 0.0 else -1,
        log_absolute=(
            math.log(abs(normalized_value))
            + math.log(left_scale)
            + math.log(right_scale)
        ),
    )


def _inner_ratio(
    numerator: _ScaledInnerProduct,
    denominator: _ScaledInnerProduct,
) -> float:
    sign = numerator.sign * denominator.sign
    if sign == 0:
        return 0.0
    try:
        magnitude = math.exp(
            numerator.log_absolute - denominator.log_absolute
        )
    except OverflowError:
        magnitude = math.inf
    return sign * magnitude


def _norm(value: Tensor) -> float:
    scale = _maximum_absolute(value)
    if scale == 0.0:
        return 0.0

    dtype = _pcg_reduction_dtype(value)
    normalized = torch.linalg.vector_norm((value / scale).to(dtype))
    return float(normalized.detach().cpu()) * scale


def jvp(
    function: TensorFunction,
    point: Tensor,
    direction: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``function(point)`` and the Jacobian-vector product."""

    result = torch.func.jvp(function, (point,), (direction,))
    return cast(Tensor, result[0]), cast(Tensor, result[1])


def vjp(
    function: TensorFunction,
    point: Tensor,
    cotangent: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``function(point)`` and the vector-Jacobian product."""

    result = torch.func.vjp(function, point)
    value = cast(Tensor, result[0])
    pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        result[1],
    )
    return value, pullback(cotangent)[0]


def hvp(
    scalar_function: TensorFunction,
    point: Tensor,
    direction: Tensor,
) -> Tensor:
    """Apply a scalar function's Hessian without constructing the matrix."""

    gradient = torch.func.grad(scalar_function)
    result = torch.func.jvp(gradient, (point,), (direction,))
    return cast(Tensor, result[1])


def gauss_newton_value_and_gradient(
    residual_fn: TensorFunction,
    point: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``0.5 * ||F(point)||²`` and ``J_F.T @ F(point)``."""

    _check_real_tensor("point", point)
    result = torch.func.vjp(residual_fn, point)
    residual = cast(Tensor, result[0])
    pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        result[1],
    )
    _check_real_tensor("residual_fn(point)", residual)

    value = _inner(0.5 * residual, residual)
    gradient = pullback(residual)[0]
    _check_like("gradient", gradient, point)
    return value, gradient


def gauss_newton_hvp(
    residual_fn: TensorFunction,
    point: Tensor,
    direction: Tensor,
    damping: Real | float = 0.0,
) -> Tensor:
    """Return ``J_F.T @ (J_F @ direction) + damping * direction``."""

    _check_real_tensor("point", point)
    _check_like("direction", direction, point)
    damping_value = _nonnegative_float("damping", damping)

    result = torch.func.vjp(residual_fn, point)
    residual = cast(Tensor, result[0])
    pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        result[1],
    )
    _check_real_tensor("residual_fn(point)", residual)
    jvp_result = torch.func.jvp(
        residual_fn,
        (point,),
        (direction,),
    )
    jacobian_vector = cast(Tensor, jvp_result[1])
    _check_like("J_F @ direction", jacobian_vector, residual)

    product = pullback(jacobian_vector)[0]
    _check_like("Gauss-Newton product", product, point)
    return product + damping_value * direction


@dataclass(frozen=True)
class PCGResult:
    """Result of a preconditioned conjugate-gradient solve."""

    solution: Tensor
    converged: bool
    iterations: int
    relative_residual: float


def pcg(
    operator: TensorFunction,
    rhs: Tensor,
    *,
    preconditioner: TensorFunction | None = None,
    initial: Tensor | None = None,
    rtol: Real | float = 1.0e-6,
    atol: Real | float = 0.0,
    max_iterations: int | None = None,
) -> PCGResult:
    """Solve ``operator(x) = rhs`` using matrix-free PCG.

    ``operator`` and ``preconditioner`` must preserve the shape, dtype, and
    device of ``rhs``. The operator is expected to be symmetric positive
    definite and the preconditioner positive definite. Reported convergence
    and ``relative_residual`` use a freshly recomputed ``rhs - operator(x)``;
    a drifted recursive residual restarts the Krylov recurrence.
    """

    _check_real_tensor("rhs", rhs)
    _check_finite_tensor("rhs", rhs)
    rtol_value = _nonnegative_float("rtol", rtol)
    atol_value = _nonnegative_float("atol", atol)

    if max_iterations is None:
        iteration_limit = rhs.numel()
    else:
        if isinstance(max_iterations, bool) or not isinstance(
            max_iterations, Integral
        ):
            raise TypeError("max_iterations must be an integer")
        iteration_limit = int(max_iterations)
        if iteration_limit <= 0:
            raise ValueError("max_iterations must be positive")

    if initial is not None:
        _check_like("initial", initial, rhs)
        _check_finite_tensor("initial", initial)

    rhs_norm = _norm(rhs)
    if not math.isfinite(rhs_norm):
        raise ValueError("rhs norm is too large to represent")
    if rhs_norm == 0.0:
        return PCGResult(
            solution=torch.zeros_like(rhs),
            converged=True,
            iterations=0,
            relative_residual=0.0,
        )

    def apply(
        function: TensorFunction,
        value: Tensor,
        output_name: str,
    ) -> Tensor:
        output = function(value)
        _check_like(output_name, output, rhs)
        _check_finite_tensor(output_name, output)
        return output

    def recompute_residual(candidate: Tensor) -> Tensor:
        return rhs - apply(operator, candidate, "operator output")

    if initial is None:
        solution = torch.zeros_like(rhs)
        residual = rhs.clone()
    else:
        solution = initial.clone()
        residual = recompute_residual(solution)

    tolerance = max(atol_value, rtol_value * rhs_norm)
    residual_norm = _norm(residual)
    if not math.isfinite(residual_norm):
        raise RuntimeError("residual norm is not finite")
    if residual_norm <= tolerance:
        return PCGResult(
            solution=solution,
            converged=True,
            iterations=0,
            relative_residual=residual_norm / rhs_norm,
        )

    if preconditioner is None:
        preconditioned = residual.clone()
    else:
        preconditioned = apply(
            preconditioner,
            residual,
            "preconditioner output",
        )

    residual_dot = _pcg_inner(residual, preconditioned)
    if not residual_dot.is_positive_finite:
        raise RuntimeError("preconditioner must be positive definite")
    search_direction = preconditioned.clone()

    for iteration in range(1, iteration_limit + 1):
        operator_direction = apply(
            operator,
            search_direction,
            "operator output",
        )
        curvature = _pcg_inner(search_direction, operator_direction)
        if not curvature.is_positive_finite:
            raise RuntimeError("operator must be symmetric positive definite")

        step = _inner_ratio(residual_dot, curvature)
        if not math.isfinite(step):
            raise RuntimeError("PCG step is not finite")
        solution = solution + step * search_direction
        residual = residual - step * operator_direction
        residual_norm = _norm(residual)
        if not math.isfinite(residual_norm):
            raise RuntimeError("residual norm is not finite")

        restarted = False
        if residual_norm <= tolerance or iteration == iteration_limit:
            residual = recompute_residual(solution)
            residual_norm = _norm(residual)
            if not math.isfinite(residual_norm):
                raise RuntimeError("true residual norm is not finite")
            if residual_norm <= tolerance:
                return PCGResult(
                    solution=solution,
                    converged=True,
                    iterations=iteration,
                    relative_residual=residual_norm / rhs_norm,
                )
            if iteration == iteration_limit:
                break
            restarted = True

        if preconditioner is None:
            preconditioned = residual.clone()
        else:
            preconditioned = apply(
                preconditioner,
                residual,
                "preconditioner output",
            )
        next_residual_dot = _pcg_inner(residual, preconditioned)
        if not next_residual_dot.is_positive_finite:
            raise RuntimeError("preconditioner must be positive definite")

        if restarted:
            search_direction = preconditioned.clone()
        else:
            beta = _inner_ratio(next_residual_dot, residual_dot)
            if not math.isfinite(beta):
                raise RuntimeError("PCG direction update is not finite")
            search_direction = preconditioned + beta * search_direction
        residual_dot = next_residual_dot

    return PCGResult(
        solution=solution,
        converged=False,
        iterations=iteration_limit,
        relative_residual=residual_norm / rhs_norm,
    )
