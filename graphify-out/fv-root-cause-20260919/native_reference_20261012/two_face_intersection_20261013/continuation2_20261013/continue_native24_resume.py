"""Cumulative native-J continuation from the saved eight-step 24D endpoint."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
BASE_SCRIPT = BASE_DIR / "continue_native24.py"
BASE_RESULT = BASE_DIR / "native24_qualification.json"
BASE_PLAN = BASE_DIR / "NATIVE24_PLAN_v6.json"
BASE_SNAPSHOT = BASE_DIR / "native24_launch_snapshot_v6.tar.gz"
GEOMETRY_RECEIPT = BASE_DIR / "native24_resume_geometry_preflight.json"
PLAN = HERE / "RESUME_PLAN_v3.json"
PLAN_SHA_PATH = HERE / "RESUME_PLAN_v3_SHA256.txt"
INPUT = BASE_RESULT
OUTPUT = HERE / "native24_resume_qualification.json"
SOURCE_SNAPSHOT = HERE / "resume_launch_snapshot_v3.tar.gz"
FACE_ENTITIES = ["y:2:1", "y:4:3"]
SIDES = ((-1, -1), (-1, 1), (1, -1), (1, 1))
STATIONARITY_TOL = 1.0e-10
ROUNDING_MULTIPLIER = 128.0
RSS_CAP_BYTES = 2 * 1024**3

_base_spec = importlib.util.spec_from_file_location("native24_base_helpers", BASE_SCRIPT)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError("frozen native24 helpers cannot be loaded")
base = importlib.util.module_from_spec(_base_spec)
sys.modules[_base_spec.name] = base
_base_spec.loader.exec_module(base)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return _sha_bytes(value.detach().contiguous().cpu().numpy().tobytes())


def _rss_bytes() -> int:
    value = int(base.resource.getrusage(base.resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_plan() -> tuple[dict[str, Any], str]:
    raw = PLAN.read_bytes()
    digest = _sha_bytes(raw)
    if PLAN_SHA_PATH.read_text().strip() != digest:
        raise ValueError("resume plan sidecar mismatch")
    plan = json.loads(raw)
    if plan.get("status") != "frozen_native24_resume":
        raise ValueError("native24 resume plan is not frozen")
    for name, expected in plan["source_files"].items():
        path = ROOT / name
        if not path.is_file() or _sha_bytes(path.read_bytes()) != expected:
            raise ValueError(f"resume source pin changed: {name}")
    for name, expected in plan["artifact_pins"].items():
        path = ROOT / name
        if not path.is_file() or _sha_bytes(path.read_bytes()) != expected:
            raise ValueError(f"resume artifact pin changed: {name}")
    if _sha_bytes(SOURCE_SNAPSHOT.read_bytes()) != plan["launch_snapshot_sha256"]:
        raise ValueError("resume launch snapshot hash mismatch")
    if _sha_bytes(INPUT.read_bytes()) != plan["input_sha256"]:
        raise ValueError("eight-step terminal input changed")
    return plan, digest


def _write(state: dict[str, Any]) -> None:
    base._write(state)


def _requalify_roundoff(data: dict[str, Any], lift: Any,
                        coordinates: Tensor) -> dict[str, Any]:
    """Scale side-agreement by the normal-gradient cancellation arithmetic."""
    full = torch.tensor([sector["full_gradient"] for sector in data["sectors"]],
                        dtype=torch.float64)
    tangent = torch.tensor([sector["tangent_gradient"] for sector in data["sectors"]],
                           dtype=torch.float64)
    jacobian = torch.func.jacrev(lift)(coordinates)
    normal = torch.tensor(data["normal_matrix"], dtype=torch.float64)
    full_scale = float(full.norm(dim=1).max())
    tangent_scale = float(tangent.norm(dim=1).max())
    chart_scale = float(torch.linalg.matrix_norm(jacobian))
    arithmetic_scale = max(chart_scale * full_scale, tangent_scale,
                           torch.finfo(torch.float64).tiny)
    agreement = float((tangent - tangent[:1]).abs().max())
    budget = ROUNDING_MULTIPLIER * torch.finfo(torch.float64).eps * arithmetic_scale
    hull_inf = max(abs(value) for value in data["full_clarke_hull"]["vector"])
    tangent_hull_inf = max(abs(value) for value in data["tangent_hull"]["vector"])
    data["tangent_gradient_agreement_inf"] = agreement
    data["tangent_gradient_agreement_budget"] = budget
    data["tangent_agreement_roundoff_scale"] = {
        "chart_jacobian_frobenius": chart_scale,
        "maximum_full_gradient_l2": full_scale,
        "maximum_tangent_gradient_l2": tangent_scale,
        "budget_formula": "128*eps*max(||J_chart||_F*max_s||g_s||_2, max_s||J_chart^T g_s||_2, tiny)",
        "normal_times_chart_jacobian_inf": float((normal @ jacobian).abs().max()),
    }
    data["tangent_stationary"] = tangent_hull_inf < STATIONARITY_TOL
    data["full_clarke_stationary"] = hull_inf < STATIONARITY_TOL
    data["first_order_pass"] = bool(
        data["branch_supported_exactly_two_faces"]
        and data["normal_rank_two"]
        and data["all_normal_signs_strict"]
        and data["tangent_stationary"]
        and data["full_clarke_stationary"]
        and agreement <= budget
    )
    return data


def _hessian_metrics(hessian: Tensor) -> dict[str, Any]:
    symmetric = (hessian + hessian.T) / 2
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    norm = float(torch.linalg.matrix_norm(symmetric))
    symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T)
                     / max(norm, torch.finfo(hessian.dtype).tiny))
    floor = max(base.qualifier.blocks.CURVATURE_REL_TOL * float(eigenvalues.abs().max()),
                ROUNDING_MULTIPLIER * torch.finfo(hessian.dtype).eps
                * max(norm, torch.finfo(hessian.dtype).tiny))
    pd = symmetry <= base.qualifier.blocks.SYMMETRY_TOL and float(eigenvalues[0]) > floor
    target = max(floor, 1.0e-3 * norm)
    shift = 0.0 if pd else max(0.0, target - float(eigenvalues[0])) + ROUNDING_MULTIPLIER * torch.finfo(hessian.dtype).eps * norm
    return {"symmetric": symmetric, "eigenvalues": eigenvalues,
            "lambda_min": float(eigenvalues[0]), "symmetry_relative": symmetry,
            "positive_floor": floor, "positive_definite": pd, "modified_shift": shift}


def _profile_hessian_at(state: dict[str, Any], profile: Any, control: Tensor,
                        coordinates: Tensor, lift: Any, parameters: Tensor,
                        current: dict[str, Any], deadline: float,
                        *, sides: tuple[int, int] = (1, 1)) -> tuple[Tensor, dict[str, Any]]:
    hessian, profile_j, profile_gradient = base._hessian_profile(
        profile, control, coordinates, lift, parameters, sides, state, deadline
    )
    sector_index = SIDES.index(sides)
    reference_gradient = torch.tensor(current["sectors"][sector_index]["tangent_gradient"],
                                      dtype=control.dtype)
    budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * max(1.0, abs(current["native_J"]))
    metric = _hessian_metrics(hessian)
    record = {
        "control_sha256": _tensor_sha(control),
        "sides": list(sides),
        "hessian": hessian.tolist(),
        "hessian_sha256": _tensor_sha(hessian),
        "eigenvalues": metric["eigenvalues"].tolist(),
        "lambda_min": metric["lambda_min"],
        "symmetry_relative": metric["symmetry_relative"],
        "positive_floor": metric["positive_floor"],
        "positive_definite": metric["positive_definite"],
        "modified_shift": metric["modified_shift"],
        "profile_J": profile_j,
        "profile_J_abs_gap": abs(profile_j - float(current["native_J"])),
        "profile_gradient_inf_gap": float((profile_gradient - reference_gradient).abs().max()),
        "profile_comparison_budget": budget,
    }
    record["profile_match"] = (record["profile_J_abs_gap"] <= budget
                               and record["profile_gradient_inf_gap"] <= budget)
    state.setdefault("fresh_hessians", []).append(record)
    _write(state)
    return metric["symmetric"] + metric["modified_shift"] * torch.eye(24, dtype=control.dtype), record


def _terminal_p2(state: dict[str, Any], profile: Any, problem: Any,
                 parameters: Tensor, lift: Any, coordinates: Tensor,
                 control: Tensor, point: dict[str, Any], deadline: float) -> bool:
    matrices: dict[str, Any] = {}
    passed = True
    for sides in SIDES:
        hessian, profile_j, profile_gradient = base._hessian_profile(
            profile, control, coordinates, lift, parameters, sides, state, deadline
        )
        metric = _hessian_metrics(hessian)
        index = SIDES.index(sides)
        reference = torch.tensor(point["sectors"][index]["tangent_gradient"], dtype=control.dtype)
        budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * max(1.0, abs(point["native_J"]))
        profile_j_gap = abs(profile_j - float(point["native_J"]))
        profile_g_gap = float((profile_gradient - reference).abs().max())
        match = profile_j_gap <= budget and profile_g_gap <= budget
        passed = passed and metric["positive_definite"] and match
        key = f"{sides[0]},{sides[1]}"
        matrices[key] = {
            "control_sha256": _tensor_sha(control),
            "hessian": hessian.tolist(),
            "hessian_sha256": _tensor_sha(hessian),
            "eigenvalues": metric["eigenvalues"].tolist(),
            "lambda_min": metric["lambda_min"],
            "symmetry_relative": metric["symmetry_relative"],
            "positive_floor": metric["positive_floor"],
            "positive_definite": metric["positive_definite"],
            "profile_J_abs_gap": profile_j_gap,
            "profile_gradient_inf_gap": profile_g_gap,
            "profile_comparison_budget": budget,
            "profile_match": match,
        }
        state["final_hessians"] = matrices
        state["phase"] = "terminal_four_quadrant_P2"
        _write(state)
    state["final_hessians"] = matrices
    state["final_point"] = point
    state["minimum_claim"] = passed
    state["response_claim"] = False
    state["status"] = ("qualified_original_native_J_two_face_minimum" if passed
                        else "two_face_stationary_P2_failed")
    state["phase"] = "terminal_same_point_qualification"
    _write(state)
    return passed


def run() -> dict[str, Any]:
    plan, plan_sha = _load_plan()
    if OUTPUT.exists():
        raise ValueError("resume output already exists; no implicit rerun")
    prior = json.loads(INPUT.read_text())
    point = prior["current_point"]
    control = torch.tensor(point["control"], dtype=torch.float64)
    coordinates = torch.tensor(point["coordinates"], dtype=torch.float64)
    if _tensor_sha(control) != plan["input_control_sha256"] or _tensor_sha(control) != point["control_sha256"]:
        raise ValueError("resume control differs from exact v6 terminal receipt")
    started = time.monotonic()
    base._RUN_STARTED = started
    base.OUTPUT = OUTPUT
    deadline = started + float(plan["budget"]["phase_wall_seconds"])
    case = base.qualifier.cycle.load_fixed_case()
    problem, original, truth, parameters = (case["problem"], case["original"], case["truth"], case["parameters"])
    if _tensor_sha(parameters) != plan["parameters_sha256"]:
        raise ValueError("fixed parameters differ from prior continuation")
    if problem.frozen.fv_transport.replay is not True:
        raise ValueError("authoritative native problem must remain replay=True")
    profile, profile_identity = base._profile_problem(problem)
    basis = problem.frozen.fv_transport.psi_basis
    limits = problem.frozen.fv_transport.coefficient_limits
    matrix, pivots, free = base._face_matrix(basis, limits)
    lift, roundtrip_coordinates = base._build_lift(control, matrix, pivots, free)
    if not torch.equal(roundtrip_coordinates, coordinates) or _tensor_sha(lift(coordinates)) != _tensor_sha(control):
        raise ValueError("saved terminal control/coordinates do not roundtrip the pinned two-face chart")
    state: dict[str, Any] = {
        "status": "in_progress",
        "phase": "resume_input_ready",
        "ledger": {
            "original_wall_cap_seconds": plan["budget"]["original_wall_cap_seconds"],
            "charged_before_phase_seconds": plan["budget"]["charged_seconds_before_phase"],
            "phase_wall_cap_seconds": plan["budget"]["phase_wall_seconds"],
            "remaining_at_phase_start_seconds": plan["budget"]["estimated_remaining_seconds_before_phase"],
            "reserved_after_phase_seconds": plan["budget"]["reserved_after_phase_seconds"],
            "native_J_calls_before_phase": plan["budget"]["native_J_calls_before"],
            "native_J_cumulative_cap": plan["budget"]["native_J_cumulative_cap"],
            "prior_hessian_rows": plan["budget"]["prior_hessian_rows"],
            "additional_hessian_row_cap": plan["budget"]["additional_hessian_row_cap"],
            "gradient_calls_before_phase": plan["budget"]["gradient_calls_before_phase"],
            "cumulative_gradient_cap": plan["budget"]["cumulative_gradient_cap"],
        },
        "phase_plan_sha256": plan_sha,
        "input_result_sha256": _sha_bytes(INPUT.read_bytes()),
        "input_control_sha256": _tensor_sha(control),
        "input_native_J": float(point["native_J"]),
        "input_point_first_order_pass": point["first_order_pass"],
        "base_plan_sha256": plan["base_plan_sha256"],
        "source_before": plan["effective_source_hashes"],
        "transport_extension_transition": plan["transport_extension_transition"],
        "native_J_calls_before_phase": int(plan["budget"]["native_J_calls_before"]),
        "native_J_calls_started": 0,
        "native_J_calls_completed": int(plan["budget"]["native_J_calls_before"]),
        "native_J_cumulative_cap": int(plan["budget"]["native_J_cumulative_cap"]),
        "gradient_calls_started": 0,
        "gradient_calls_completed": 0,
        "gradient_call_cap": int(plan["budget"]["additional_gradient_call_cap"]),
        "hessian_rows_started": 0,
        "hessian_rows_completed": 0,
        "hessian_row_cap": int(plan["budget"]["additional_hessian_row_cap"]),
        "profile_scalar_calls": 0,
        "profile_scalar_call_cap": 20,
        "fresh_hessian_cycle_cap": int(plan["budget"]["fresh_hessian_cycles"]),
        "chord_steps_per_hessian_cap": int(plan["budget"]["chord_steps_per_hessian"]),
        "basis": basis,
        "limits": limits,
        "profile_identity": profile_identity,
        "chart": {"pivot_flow_indices": pivots, "free_flow_indices": free,
                  "matrix": matrix.tolist(), "jacobian_shape": [26, 24]},
        "input_fixed_case_identity": base.qualifier.cycle.shared._input_identity(
            problem, original, control, parameters, truth),
        "iterations": [],
        "accepted_steps": [],
        "fresh_hessians": [],
        "current_control": control.tolist(),
        "current_control_sha256": _tensor_sha(control),
        "native_J": float(point["native_J"]),
        "memory_cap_bytes": RSS_CAP_BYTES,
    }
    _write(state)
    if _rss_bytes() > RSS_CAP_BYTES:
        state.update(status="RSS_cap_exhausted_preflight", phase="RSS_preflight_gate")
        _write(state)
        return state

    # Re-audit the supplied current point under the actual replay=True problem.
    current_control = control
    current_coordinates = coordinates
    current = base._audit_point(problem, parameters, current_control, lift,
                                current_coordinates, matrix, pivots, free,
                                profile, state, deadline, compute_hessian=False)
    current = _requalify_roundoff(current, lift, current_coordinates)
    state["current_point"] = current
    state["native_J"] = current["native_J"]
    state["current_control"] = current_control.tolist()
    state["current_control_sha256"] = _tensor_sha(current_control)
    _write(state)
    native_recheck_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * max(1.0, abs(float(point["native_J"])))
    if abs(float(current["native_J"]) - float(point["native_J"])) > native_recheck_budget:
        state.update(status="resume_native_J_recheck_mismatch", phase="native_J_recheck_gate")
        _write(state)
        return state
    if not current["branch_supported_exactly_two_faces"]:
        state.update(status="unsupported_current_two_face_branch", phase="initial_branch_gate")
        _write(state)
        return state
    if not current["normal_rank_two"] or not current["all_normal_signs_strict"]:
        state.update(status="unsupported_current_two_face_normal_gate", phase="initial_normal_gate")
        _write(state)
        return state

    final_root = False
    total_moves = 0
    current_metric: Tensor | None = None
    metric_source_sha: str | None = None
    for cycle in range(int(plan["budget"]["fresh_hessian_cycles"])):
        if current["first_order_pass"]:
            final_root = True
            break
        if _rss_bytes() > RSS_CAP_BYTES or time.monotonic() >= deadline:
            state.update(status="resume_resource_budget_exhausted", phase="cycle_resource_gate")
            _write(state)
            break
        if cycle == 0 or current_metric is None or current["control_sha256"] != metric_source_sha:
            raw_h, hrecord = _profile_hessian_at(
                state, profile, current_control, current_coordinates, lift,
                parameters, current, deadline, sides=(1, 1)
            )
            if not hrecord["profile_match"]:
                state.update(status="replay_profile_mismatch", phase="fresh_H_profile_gate")
                _write(state)
                break
            metric = _hessian_metrics(raw_h)
            current_metric = metric["symmetric"] + metric["modified_shift"] * torch.eye(24, dtype=control.dtype)
            metric_source_sha = current["control_sha256"]
        else:
            raw_h = current_metric
            metric = _hessian_metrics(raw_h)
        state["cycle_state"] = {"cycle": cycle, "metric_source_control_sha256": metric_source_sha,
                                "fresh_hessian": cycle == 0 or bool(state.get("fresh_hessian_added_this_cycle"))}
        _write(state)
        try:
            factor = torch.linalg.cholesky(current_metric)
        except RuntimeError:
            state.update(status="resume_modified_metric_not_spd", phase="metric_factor_gate")
            _write(state)
            break
        # One Newton step plus up to three chord corrections per refreshed H.
        for local_step in range(4):
            if current["first_order_pass"]:
                final_root = True
                break
            if total_moves >= int(plan["budget"]["total_move_cap"]):
                state.update(status="resume_move_cap_exhausted", phase="move_cap_gate")
                _write(state)
                break
            if time.monotonic() >= deadline or state["native_J_calls_completed"] >= state["native_J_cumulative_cap"]:
                state.update(status="resume_cumulative_budget_exhausted", phase="move_budget_gate")
                _write(state)
                break
            tangent_gradients = torch.tensor(
                [sector["tangent_gradient"] for sector in current["sectors"]], dtype=control.dtype
            )
            gradient = tangent_gradients.mean(dim=0)
            step = -torch.cholesky_solve(gradient[:, None], factor).squeeze(1)
            descent = float(torch.dot(gradient, step))
            move = {"cycle": cycle, "local_step": local_step,
                    "kind": "fresh_H_newton" if local_step == 0 else "fixed_H_chord",
                    "from_control_sha256": current["control_sha256"],
                    "native_J": current["native_J"],
                    "tangent_gradient_inf": float(gradient.abs().max()),
                    "metric_source_control_sha256": metric_source_sha,
                    "metric_lambda_min": metric["lambda_min"],
                    "metric_shift": metric["modified_shift"],
                    "descent_product": descent,
                    "step_norm": float(torch.linalg.vector_norm(step)),
                    "line_search": []}
            state["iterations"].append(move)
            _write(state)
            if not math.isfinite(descent) or descent >= 0:
                state.update(status="resume_non_descent_step", phase="descent_gate")
                _write(state)
                break
            accepted = False
            for backtrack in range(28):
                if time.monotonic() >= deadline or state["native_J_calls_completed"] >= state["native_J_cumulative_cap"]:
                    state.update(status="resume_native_J_or_wall_cap", phase="line_search_cap_gate")
                    _write(state)
                    break
                alpha = 0.5**backtrack
                trial_coordinates = current_coordinates + alpha * step
                try:
                    trial_control = lift(trial_coordinates)
                except (ValueError, RuntimeError) as error:
                    move["line_search"].append({"alpha": alpha, "status": "chart_domain_reject", "error": str(error)})
                    _write(state)
                    continue
                value = base._native_value(problem, trial_control, parameters, state, deadline)
                rhs = float(current["native_J"]) + 1e-4 * alpha * descent
                item = {"alpha": alpha, "control_sha256": _tensor_sha(trial_control),
                        "native_J": value, "armijo_rhs": rhs,
                        "status": "accepted" if value <= rhs else "Armijo_reject"}
                move["line_search"].append(item)
                _write(state)
                if value <= rhs:
                    accepted = True
                    current_control = trial_control
                    current_coordinates = trial_coordinates
                    total_moves += 1
                    state["accepted_steps"].append({
                        "cycle": cycle, "local_step": local_step, "kind": move["kind"],
                        "alpha": alpha, "native_J": value,
                        "control_sha256": _tensor_sha(current_control),
                        "step_norm": alpha * float(torch.linalg.vector_norm(step)),
                        "metric_source_control_sha256": metric_source_sha,
                    })
                    state["current_control"] = current_control.tolist()
                    state["current_control_sha256"] = _tensor_sha(current_control)
                    state["native_J"] = value
                    state["phase"] = "accepted_native24_resume_step"
                    _write(state)
                    current = base._audit_point(
                        problem, parameters, current_control, lift, current_coordinates,
                        matrix, pivots, free, profile, state, deadline,
                        compute_hessian=False,
                    )
                    current = _requalify_roundoff(current, lift, current_coordinates)
                    state["current_point"] = current
                    state["native_J"] = current["native_J"]
                    state["current_control"] = current_control.tolist()
                    state["current_control_sha256"] = current["control_sha256"]
                    _write(state)
                    if not current["branch_supported_exactly_two_faces"]:
                        state.update(status="unsupported_new_face_or_limiter_event", phase="accepted_point_branch_gate")
                        _write(state)
                        break
                    if not current["normal_rank_two"] or not current["all_normal_signs_strict"]:
                        state.update(status="unsupported_normal_release", phase="accepted_point_normal_gate")
                        _write(state)
                        break
                    if current["first_order_pass"]:
                        final_root = True
                    break
            if state.get("status") in ("resume_native_J_or_wall_cap", "unsupported_new_face_or_limiter_event", "unsupported_normal_release"):
                break
            if not accepted:
                state.update(status="resume_Armijo_grid_exhausted", phase="line_search_failed")
                _write(state)
                break
            if final_root:
                break
        if final_root or state.get("status") not in ("in_progress", "accepted_native24_resume_step"):
            break

    if final_root:
        passed = _terminal_p2(state, profile, problem, parameters, lift,
                              current_coordinates, current_control, current, deadline)
        state["status"] = ("qualified_original_native_J_two_face_minimum" if passed
                           else "two_face_stationary_P2_failed")
    elif state.get("status") in ("in_progress", "accepted_native24_resume_step"):
        state["status"] = "four_cycle_budget_exhausted_without_two_face_stationarity"
        state["phase"] = "terminal_nonstationary"
        state["terminal_point"] = current
        _write(state)

    state["elapsed_seconds"] = time.monotonic() - started
    state["peak_rss_bytes"] = _rss_bytes()
    state["native_J_calls_completed_total"] = state["native_J_calls_completed"]
    state["gradient_calls_completed_total"] = plan["budget"]["gradient_calls_before_phase"] + state["gradient_calls_completed"]
    state["hessian_rows_completed_total"] = plan["budget"]["prior_hessian_rows"] + state["hessian_rows_completed"]
    state["ledger"]["charged_at_phase_end_seconds"] = plan["budget"]["charged_seconds_before_phase"] + state["elapsed_seconds"]
    state["ledger"]["estimated_remaining_at_phase_end_seconds"] = 3600.0 - state["ledger"]["charged_at_phase_end_seconds"]
    state["ledger"]["cumulative_native_J_calls_at_phase_end"] = state["native_J_calls_completed"]
    state["source_after"] = plan["effective_source_hashes"]
    state["source_archive_closure"] = all(
        _sha_bytes((ROOT / name).read_bytes()) == digest
        for name, digest in plan["source_files"].items()
    ) and all(_sha_bytes((ROOT / name).read_bytes()) == digest
              for name, digest in plan["artifact_pins"].items())
    _write(state)
    return state


if __name__ == "__main__":
    try:
        result = run()
    except BaseException as error:
        if OUTPUT.exists():
            record = json.loads(OUTPUT.read_text())
            record.update(status="terminal_exception", terminal_exception_type=type(error).__name__,
                          terminal_exception=str(error))
            _write(record)
        raise
    print(json.dumps({key: result.get(key) for key in (
        "status", "phase", "native_J", "current_control_sha256", "accepted_steps",
        "native_J_calls_completed", "gradient_calls_completed", "hessian_rows_completed",
        "minimum_claim", "elapsed_seconds", "ledger",
    )}, indent=2, sort_keys=True))
