"""Bounded orchestration for a conditional FV analysis response.

This wrapper owns eligibility and optional root-refinement bookkeeping.  The
caller still owns the physical branch proof; the local response solver remains
the only numerical response implementation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import time
from typing import Any

import torch
from torch import Tensor

from advar.local_response import LocalResponse, compute_local_response


_STATIONARITY_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class _Snapshot:
    objective: float
    gradient_max: float
    branch: Any
    branch_scope: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "gradient_max": self.gradient_max,
            "branch": self.branch,
            "branch_scope": self.branch_scope,
        }


def _control_digest(control: Tensor) -> str:
    return hashlib.sha256(
        control.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _snapshot(
    objective: Callable[[Tensor, Tensor], Tensor],
    control: Tensor,
    p: Tensor,
    branch_check: Callable[[Tensor, Tensor], tuple[Any, str]],
) -> _Snapshot:
    value = objective(control, p)
    if not isinstance(value, Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
        raise ValueError("objective must return a finite scalar tensor")
    gradient = torch.func.grad(objective, argnums=0)(control, p)
    if not isinstance(gradient, Tensor) or gradient.shape != control.shape:
        raise ValueError("objective gradient shape does not match control")
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("objective gradient must be finite")
    checked = branch_check(control, p)
    if not isinstance(checked, tuple) or len(checked) != 2:
        raise ValueError("branch_check must return (signature, scope)")
    branch, scope = checked
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("branch_check scope must be a nonempty string")
    return _Snapshot(
        objective=float(value),
        gradient_max=float(gradient.abs().max()),
        branch=branch,
        branch_scope=scope,
    )


def _result(
    *,
    status: str,
    response: LocalResponse | None,
    original_control: Tensor,
    control: Tensor,
    before: dict[str, Any],
    after: dict[str, Any],
    refinement_used: bool,
    error: str | None,
    timings: dict[str, float],
) -> dict[str, Any]:
    return {
        "status": status,
        "response": response,
        "original_control": original_control,
        "control": control,
        "before": before,
        "after": after,
        "refinement_used": refinement_used,
        "error": error,
        "timings": timings,
    }


def prepare_response(
    objective: Callable[[Tensor, Tensor], Tensor],
    score: Callable[[Tensor, Tensor], Tensor],
    control: Tensor,
    p: Tensor,
    directions: Mapping[str, Tensor],
    *,
    branch_check: Callable[[Tensor, Tensor], tuple[Any, str]],
    input_identity: Mapping[str, Any],
    refine: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, Any]:
    """Assess, optionally refine, and compute a conditional local response.

    A refusal always returns the original control and ``response=None``.
    ``refine`` is deliberately explicit: this wrapper does not select or
    implement a physical optimizer.
    """
    started = time.monotonic()
    original_control = control.detach().clone()
    empty_snapshot = {
        "objective": None,
        "gradient_max": None,
        "branch": None,
        "branch_scope": None,
    }
    before: dict[str, Any] = dict(empty_snapshot)
    after: dict[str, Any] = dict(empty_snapshot)
    assessment_seconds = 0.0
    refinement_seconds = 0.0
    response_seconds = 0.0

    def refuse(reason: str, *, refinement_used: bool) -> dict[str, Any]:
        return _result(
            status="ineligible",
            response=None,
            original_control=original_control,
            control=original_control,
            before=before,
            after=after,
            refinement_used=refinement_used,
            error=reason,
            timings={
                "assessment_seconds": assessment_seconds,
                "refinement_seconds": refinement_seconds,
                "response_seconds": response_seconds,
                "total_seconds": time.monotonic() - started,
            },
        )

    before_started = time.monotonic()
    before_snapshot: _Snapshot | None = None
    try:
        before_snapshot = _snapshot(objective, control, p, branch_check)
        before = before_snapshot.as_dict()
        before_error: str | None = None
    except (ValueError, RuntimeError) as error:
        before_error = str(error)
    assessment_seconds += time.monotonic() - before_started

    if before_error is not None:
        return refuse(f"initial assessment failed: {before_error}", refinement_used=False)
    if before_snapshot is not None and before_snapshot.gradient_max < _STATIONARITY_TOLERANCE:
        candidate = control
        refinement_used = False
    elif refine is None:
        reason = before_error or (
            "control is not stationary: "
            f"gradient_max={before_snapshot.gradient_max if before_snapshot is not None else 'unknown'}"
        )
        after = dict(before)
        return refuse(reason, refinement_used=False)
    else:
        refinement_used = True
        refine_started = time.monotonic()
        try:
            candidate = refine(original_control.clone(), p)
            if not isinstance(candidate, Tensor):
                raise ValueError("refine must return a tensor control")
        except (ValueError, RuntimeError) as error:
            refinement_seconds += time.monotonic() - refine_started
            return refuse(f"refinement failed: {error}", refinement_used=True)
        refinement_seconds += time.monotonic() - refine_started

    after_started = time.monotonic()
    try:
        after_snapshot = _snapshot(objective, candidate, p, branch_check)
        after = after_snapshot.as_dict()
    except (ValueError, RuntimeError) as error:
        assessment_seconds += time.monotonic() - after_started
        return refuse(f"post-refinement assessment failed: {error}", refinement_used=refinement_used)
    assessment_seconds += time.monotonic() - after_started

    if after_snapshot.gradient_max >= _STATIONARITY_TOLERANCE:
        return refuse(
            "refined control is not stationary: "
            f"gradient_max={after_snapshot.gradient_max}",
            refinement_used=refinement_used,
        )
    identity = dict(input_identity)
    identity["control_sha256"] = _control_digest(candidate)
    response_started = time.monotonic()
    try:
        response = compute_local_response(
            objective,
            score,
            candidate,
            p,
            directions,
            branch_check=branch_check,
            input_identity=identity,
        )
    except (ValueError, RuntimeError) as error:
        response_seconds += time.monotonic() - response_started
        return refuse(f"local response failed: {error}", refinement_used=refinement_used)
    response_seconds += time.monotonic() - response_started
    return _result(
        status="eligible",
        response=response,
        original_control=original_control,
        control=candidate,
        before=before,
        after=after,
        refinement_used=refinement_used,
        error=None,
        timings={
            "assessment_seconds": assessment_seconds,
            "refinement_seconds": refinement_seconds,
            "response_seconds": response_seconds,
            "total_seconds": time.monotonic() - started,
        },
    )
