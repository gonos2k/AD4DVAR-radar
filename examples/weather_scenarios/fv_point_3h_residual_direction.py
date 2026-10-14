"""Build a face-tangent residual direction from a supplied derivative action.

The caller owns derivative evaluation. ``model(direction)`` must return the
fresh directional derivative ``DG(direction)`` at the current control. This
module only projects and audits that result; it does not run HVPs or model
evaluations.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def _finite_vector(value: Tensor, like: Tensor, label: str) -> None:
    if (not isinstance(value, Tensor) or value.ndim != 1 or value.shape != like.shape
            or value.dtype != like.dtype or value.device != like.device
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{label} must be a finite vector matching the control")


def _roundoff(dtype: torch.dtype, *scales: Tensor) -> Tensor:
    scale = torch.stack([value.abs() for value in scales]).amax().clamp_min(
        torch.finfo(dtype).tiny)
    return 128.0 * torch.finfo(dtype).eps * scale


def build_residual_direction(
    model: Callable[[Tensor], Tensor],
    normal: Tensor,
    gm: Tensor,
    gp: Tensor,
    chart: Tensor,
    pivot: int,
    *,
    normalize_to: Tensor | None = None,
    seed_direction: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Return normalized ``s=P DG[d]`` and diagnostics for a smooth common face.

    The common-face gradient is the strict-interior minimum-norm mixture
    ``g*=(1-theta)gm+theta*gp``. The Euclidean control-metric projector is
    ``P=I-n n.T/(n.T n)`` and the seed is ``d=-P g*``. A supplied ``model``
    action returns the *raw* ``DG[d]``; the raw residual direction is
    ``s_raw=P DG[d]``. The returned trial direction scales ``s_raw`` to
    ``||d||`` by default (or to ``||normalize_to||`` when given), preserving
    sign and direction. Diagnostics retain both raw and scaled tensors.

    ``seed_merit_product`` is ``g*·s_raw = g*·P DG[d]``, the derivative of the
    squared residual merit along the seed ``d``. It does not measure merit along
    the returned direction ``s``. The latter requires a fresh caller-owned
    action ``DG[s]`` and its product ``g*·DG[s]``. The old generic keys
    ``merit_product``, ``merit_descent_budget``, and ``merit_descends`` remain
    explicit seed-only aliases for the frozen ``direction_trial.py`` consumer.
    The caller must compute fresh actions whenever its point or requested
    direction changes.

    The mixture-weight derivative belongs inside ``model`` when ``G`` includes
    a control-dependent mixture. This helper does not assume either-side or
    merit descent: it reports those products from the supplied current data.
    """
    if not callable(model):
        raise ValueError("model must be a directional derivative action")
    _finite_vector(gm, gm, "minus gradient")
    _finite_vector(gp, gm, "plus gradient")
    _finite_vector(normal, gm, "face normal")
    if chart.ndim != 2 or chart.shape != (gm.numel(), gm.numel() - 1):
        raise ValueError("chart must span the retained-coordinate face tangent")
    if (chart.dtype != gm.dtype or chart.device != gm.device
            or not bool(torch.isfinite(chart).all())):
        raise ValueError("chart must be finite and match the control dtype/device")
    if not 0 <= pivot < gm.numel():
        raise ValueError("pivot is outside the control")
    if normalize_to is not None:
        _finite_vector(normalize_to, gm, "normalization reference")

    normal_squared = torch.dot(normal, normal)
    normal_norm = torch.sqrt(normal_squared)
    tiny = torch.finfo(gm.dtype).tiny
    if not bool(torch.isfinite(normal_norm) & (normal_norm > tiny)):
        raise ValueError("selected face normal is unresolved")
    unit_normal = normal / normal_norm
    jump = gp - gm
    jump_norm = torch.linalg.vector_norm(jump)
    normal_jump = torch.dot(unit_normal, jump)
    jump_budget = _roundoff(gm.dtype, torch.linalg.vector_norm(gm),
        torch.linalg.vector_norm(gp), jump_norm)
    if not bool(torch.isfinite(jump_norm) & (jump_norm > jump_budget)):
        raise ValueError("common-face gradient jump is unresolved")
    tangent_jump = jump - unit_normal * normal_jump
    common_face = torch.linalg.vector_norm(tangent_jump) <= jump_budget
    if not bool(common_face):
        raise ValueError("one-sided gradient jump is not supported by the common face")
    if not bool(normal_jump.abs() > jump_budget):
        raise ValueError("common-face jump has an unresolved normal component")

    theta = -torch.dot(gm, jump) / torch.dot(jump, jump)
    theta_budget = 128.0 * torch.finfo(gm.dtype).eps
    if not bool(torch.isfinite(theta) & (theta > theta_budget)
            & (theta < 1.0 - theta_budget)):
        raise ValueError("common-face minimum-mixture weight is not interior")
    mixed = gm + theta * jump
    mixed_normal = torch.dot(unit_normal, mixed)
    mixed_budget = _roundoff(gm.dtype, torch.linalg.vector_norm(mixed))
    if not bool(mixed_normal.abs() <= mixed_budget):
        raise ValueError("minimum-mixture gradient is not tangent to the common face")

    projected_gradient = mixed - unit_normal * mixed_normal
    descent = -projected_gradient
    if seed_direction is not None:
        _finite_vector(seed_direction, gm, "qualified seed direction")
        budget = _roundoff(gm.dtype, torch.linalg.vector_norm(descent),
            torch.linalg.vector_norm(seed_direction))
        if not bool(torch.linalg.vector_norm(seed_direction-descent) <= budget):
            raise ValueError("qualified seed differs from projected gradient descent")
        descent = seed_direction
    tangent_chart_error = tangent._validate_direction_override(
        gm, normal, chart, pivot, descent)
    action = model(descent)
    _finite_vector(action, gm, "model directional derivative")
    raw = action - unit_normal * torch.dot(unit_normal, action)
    raw_norm = torch.linalg.vector_norm(raw)
    target = torch.linalg.vector_norm(descent if normalize_to is None else normalize_to)
    if not bool(torch.isfinite(raw_norm) & (raw_norm > tiny)
            & torch.isfinite(target) & (target > tiny)):
        raise ValueError("residual direction or normalization scale is unresolved")
    direction = raw * (target / raw_norm)
    direction_chart_error = tangent._validate_direction_override(
        gm, normal, chart, pivot, direction)

    side_products = (torch.dot(gm, direction), torch.dot(gp, direction))
    side_budget = _roundoff(gm.dtype,
        torch.linalg.vector_norm(gm) * torch.linalg.vector_norm(direction),
        torch.linalg.vector_norm(gp) * torch.linalg.vector_norm(direction))
    seed_merit_product = torch.dot(mixed, raw)
    seed_merit_budget = _roundoff(gm.dtype,
        torch.linalg.vector_norm(mixed) * torch.linalg.vector_norm(raw))
    seed_merit_descends = bool(seed_merit_product < -seed_merit_budget)
    diagnostics: dict[str, Any] = {
        "theta": theta,
        "mixed_gradient": mixed,
        "descent_direction": descent,
        "model_action_raw": action,
        "residual_direction_raw": raw,
        "residual_direction_norm": raw_norm,
        "normalization_target_norm": target,
        "normalization_scale": target / raw_norm,
        "side_products": side_products,
        "side_descent_budget": side_budget,
        "both_side_gradients_descend": bool(max(side_products) < -side_budget),
        "seed_merit_product": seed_merit_product,
        "seed_merit_descent_budget": seed_merit_budget,
        "seed_merit_descends": seed_merit_descends,
        # Backward-compatible names consumed by the frozen direction_trial.
        "merit_product": seed_merit_product,
        "merit_descent_budget": seed_merit_budget,
        "merit_descends": seed_merit_descends,
        "normal_jump": normal_jump,
        "common_face_tangent_jump_norm": torch.linalg.vector_norm(tangent_jump),
        "common_face_jump_budget": jump_budget,
        "mixture_normal_component": mixed_normal,
        "mixture_tangent_budget": mixed_budget,
        "descent_chart_error": tangent_chart_error,
        "direction_chart_error": direction_chart_error,
        "finite": bool(all(torch.isfinite(value).all() for value in
            (theta, mixed, descent, action, raw, raw_norm, direction, seed_merit_product))),
    }
    return direction, diagnostics
