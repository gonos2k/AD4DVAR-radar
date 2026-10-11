"""Fresh same-control P2 closure for the proposed net-flux evaluator."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_current_gradient_continuation as previous
from examples.weather_scenarios import fv_point_3h_current_gradient_endpoint as endpoint
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_gn_finite_continuation as finite
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent

BASE_SHA = endpoint.BASE_SHA
FACE = {"axis": "y", "row": 4, "column": 3}
FACE_SCALE = previous.FACE_SCALE


def requalify_base(backend: Any, base: dict[str, Any], output: Path) -> dict[str, Any]:
    """Recompute a fresh current P2 closure without admitting the archived P2."""
    control = torch.as_tensor(base["control"], dtype=torch.float64)
    old_sources = base.get("raw", {}).get("source_after")
    source = backend.source_anchor
    old_transport = old_sources.get("src/advar/transport.py") if isinstance(old_sources, dict) else None
    new_transport = source.get("src/advar/transport.py")
    if (tangent._tensor_sha(control) != BASE_SHA or control.shape != (26,)
            or not bool(torch.isfinite(control).all())
            or not isinstance(old_transport, str) or not isinstance(new_transport, str)
            or new_transport == old_transport):
        raise ValueError("base is not the fixed 556a point under a changed transport source")
    backend._check(control)
    problem, parameters, weights = backend.problem, backend.parameters, backend.weights
    input_before = shared._input_identity(
        problem, backend.original, control, parameters, backend.truth)
    runtime_before = shared._runtime()
    if (input_before != base.get("input_after") or runtime_before != base.get("runtime_after")
            or input_before != backend.input_anchor or runtime_before != backend.runtime_anchor):
        raise ValueError("fixed 556a input/runtime differs from the archived endpoint")

    observed = tangent._observe(shared, problem, control, parameters, weights)
    if (observed["pair"].get("passed") is not True or observed.get("face_ok") is not True
            or observed.get("side_objectives_match_native") is not True
            or observed.get("side_gradients_finite") is not True
            or not math.isfinite(float(observed["native_j"]))):
        raise ValueError("fresh net-flux current pair, face, cost, or gradients are invalid")
    gm, gp = observed["side"][-1][1], observed["side"][1][1]
    if (gm.shape != control.shape or gp.shape != control.shape
            or gm.dtype != control.dtype or gp.dtype != control.dtype
            or not bool(torch.isfinite(gm).all() and torch.isfinite(gp).all())):
        raise ValueError("fresh net-flux side gradients are malformed")
    normal = geometry._face_normal(control, weights)
    outward = previous.native_face_release(gm, gp, normal)
    if outward["release_required"]:
        tangent._write(output, {"phase": "new_evaluator_current_point_requalification",
            "status": "native_face_release_required_no_Q0_commit", "control_sha256": BASE_SHA,
            "source_identity": source, "outward_derivatives": outward})
        raise ValueError("new evaluator native face release required")
    theta_t = tangent.minimum_mixture_weight(gm, gp)
    theta = float(theta_t)
    if not math.isfinite(theta) or not 0.0 < theta < 1.0:
        raise ValueError("fresh net-flux mixture is not strictly interior")
    pivot = geometry._pivot(control, weights)
    _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    mixed, _tangent_gradient, direction, support = tangent.tangent_direction(
        gm, gp, normal, chart, theta, pivot=pivot)
    chart_error = tangent._validate_direction_override(gm, normal, chart, pivot, direction)
    full_g = torch.cat((mixed, (observed["q"] / FACE_SCALE).reshape(1)))
    if full_g.shape != (27,) or not bool(torch.isfinite(full_g).all()):
        raise ValueError("fresh net-flux full-G is malformed")
    merit = float(torch.dot(full_g, full_g))
    if not math.isfinite(merit):
        raise ValueError("fresh net-flux full-G merit is nonfinite")

    # Reuse the established repeat format at alpha zero; this is a closure, not a trial.
    trial = {"control": control, "objective": float(observed["native_j"]),
        "F_squared": merit, "theta": theta, "alpha": 0.0,
        "J_armijo_bound": float(observed["native_j"]),
        "F_squared_armijo_bound": merit, "face_audit_passed": observed["face_ok"],
        "branch_pair_passed": observed["pair"]["passed"],
        "side_objectives_match_native": observed["side_objectives_match_native"],
        "side_gradients_finite": observed["side_gradients_finite"],
        "side_gradients": {str(s): observed["side"][s][1].tolist() for s in (-1, 1)},
        "branch_trace": {str(s): observed["traces"][s] for s in (-1, 1)}}
    proposal, repeat = backend.repeat({"direction": direction}, trial)
    repeat_theta = float(repeat.get("theta", float("nan")))
    if (repeat.get("control") != control.tolist()
            or repeat.get("mixing_minimum_valid") is not True
            or repeat.get("minimum_theta_matches_proposal") is not True
            or repeat.get("native_objective_matches_proposal") is not True
            or repeat.get("side_objectives_match_native") is not True
            or repeat.get("merit_matches_proposal") is not True
            or repeat.get("side_gradients_finite") is not True
            or repeat.get("face_audit_passed") is not True
            or repeat.get("branch_pair_passed") is not True
            or repeat.get("trace_matches_proposal") is not True
            or repeat.get("fixed_input_unchanged") is not True
            or repeat.get("runtime_unchanged") is not True
            or repeat.get("source_unchanged") is not True
            or repeat.get("deadline_passed") is not True
            or not math.isfinite(repeat_theta) or not 0.0 < repeat_theta < 1.0):
        raise ValueError("independent net-flux repeat failed cost, trace, input, or mixture closure")
    repeat_objective = float(repeat.get("objective", float("nan")))
    objective_budget = 128 * torch.finfo(control.dtype).eps * max(
        abs(float(observed["native_j"])), abs(repeat_objective), torch.finfo(control.dtype).tiny)
    if (not math.isfinite(repeat_objective)
            or abs(repeat_objective - float(observed["native_j"])) > objective_budget
            or abs(repeat_theta - theta) > 128 * torch.finfo(control.dtype).eps * max(
                abs(theta), abs(repeat_theta), torch.finfo(control.dtype).tiny)):
        raise ValueError("independent net-flux repeat cost or mixture differs from first observation")
    full_g_check = previous.fresh_full_G(repeat, full_g, weights)
    repeat_g = torch.as_tensor(full_g_check.get("G"), dtype=control.dtype)
    if (full_g_check.get("passed") is not True or repeat_g.shape != (27,)
            or not bool(torch.isfinite(repeat_g).all())):
        raise ValueError("independent net-flux repeat full-G differs componentwise")
    repeat_gradients = {s: torch.as_tensor(repeat["side_gradients"][str(s)], dtype=control.dtype)
                        for s in (-1, 1)}
    if not shared._gradient_pair_match({-1: gm, 1: gp}, repeat_gradients):
        raise ValueError("independent net-flux repeat gradients differ from the first observation")

    # Archived P2 values are reported only as version deltas, never used as gates.
    old_repeat = base["repeat"]
    old_gradients = {s: torch.as_tensor(old_repeat["side_gradients"][str(s)], dtype=control.dtype)
                     for s in (-1, 1)}
    old_full_g = torch.as_tensor(base["full_G"], dtype=control.dtype)
    result = {"phase": "new_evaluator_current_point_requalification",
        "status": "new_evaluator_current_P2_requalified", "operator":
            "q + dt * ((incoming - outgoing) / area)",
        "plan_sha256": backend.plan_sha, "control": control.tolist(),
        "control_sha256": BASE_SHA, "source_identity": source,
        "old_transport_source_sha256": old_sources["src/advar/transport.py"],
        "new_transport_source_sha256": source["src/advar/transport.py"],
        "input_before": input_before, "runtime_before": runtime_before,
        "first_observation": {"objective": float(observed["native_j"]), "theta": theta,
            "full_G": full_g.tolist(), "branch_pair": observed["pair"],
            "face": {"axis": FACE["axis"], "row": FACE["row"],
                "column": FACE["column"], "formula_q": float(observed["q"]),
                "production_q": float(observed["production_q"]),
                "bound": float(observed["face_bound"]), "normal": normal.tolist()},
            "face_ok": observed["face_ok"], "side_objectives_match_native": True,
            "side_gradients_finite": True,
            "outward_derivatives": outward,
            "branch_trace": {str(s): observed["traces"][s] for s in (-1, 1)},
            "common_face_support": support, "chart_error": chart_error},
        "repeat": repeat, "full_G_check": full_g_check,
        "new_repeat_matches_first_full_G": True,
        "archived_P2_deltas_only": {"objective": float(observed["native_j"])
                - float(old_repeat["objective"]), "theta": theta - float(old_repeat["theta"]),
            "side_gradient_linf": {str(s): float((repeat_gradients[s] - old_gradients[s]).abs().max())
                                   for s in (-1, 1)},
            "full_G_l2": float(torch.linalg.vector_norm(repeat_g - old_full_g))},
        "operator_sha256": source["src/advar/transport.py"]}
    backend._check(control)
    input_after = shared._input_identity(problem, backend.original, control, parameters, backend.truth)
    runtime_after = shared._runtime()
    if input_after != input_before or runtime_after != runtime_before:
        raise ValueError("fixed input/runtime changed during net-flux P2 requalification")
    result.update(input_after=input_after, runtime_after=runtime_after,
        source_unchanged=True, input_unchanged=True, runtime_unchanged=True)
    tangent._write(output, result)
    return result
