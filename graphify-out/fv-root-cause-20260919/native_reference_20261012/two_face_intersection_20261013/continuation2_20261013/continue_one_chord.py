"""Use one remaining fixed-H chord correction, then qualify only if eligible."""
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
BASE_DIR = HERE.parent.parent
RESUME_SCRIPT = HERE / "continue_native24_resume.py"
INPUT = HERE / "native24_resume_qualification.json"
PLAN = HERE / "CHORD_PLAN_v2.json"
PLAN_SHA_PATH = HERE / "CHORD_PLAN_v2_SHA256.txt"
OUTPUT = HERE / "native24_chord_extension_result.json"
SOURCE_SNAPSHOT = HERE / "chord_launch_snapshot_v2.tar.gz"
RSS_CAP_BYTES = 2 * 1024**3
STATIONARITY_TOL = 1.0e-10

_spec = importlib.util.spec_from_file_location("one_chord_resume_helpers", RESUME_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("frozen native24 resume helpers are unavailable")
resume = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = resume
_spec.loader.exec_module(resume)
base = resume.base
FACES, SIDES = base.FACES, resume.SIDES


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return _sha_bytes(value.detach().contiguous().cpu().numpy().tobytes())


def _rss_bytes() -> int:
    value = int(base.resource.getrusage(base.resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _write(state: dict[str, Any]) -> None:
    base._write(state)


def _load_plan() -> tuple[dict[str, Any], str]:
    raw = PLAN.read_bytes()
    digest = _sha_bytes(raw)
    if PLAN_SHA_PATH.read_text().strip() != digest:
        raise ValueError("one-chord plan sidecar mismatch")
    plan = json.loads(raw)
    if plan.get("status") != "frozen_native24_chord_extension":
        raise ValueError("one-chord continuation plan is not frozen")
    for name, expected in plan["source_files"].items():
        path = ROOT / name
        if not path.is_file() or _sha_bytes(path.read_bytes()) != expected:
            raise ValueError(f"chord source pin changed: {name}")
    for name, expected in plan["artifact_pins"].items():
        path = ROOT / name
        if not path.is_file() or _sha_bytes(path.read_bytes()) != expected:
            raise ValueError(f"chord artifact pin changed: {name}")
    if _sha_bytes(SOURCE_SNAPSHOT.read_bytes()) != plan["launch_snapshot_sha256"]:
        raise ValueError("chord launch snapshot mismatch")
    if _sha_bytes(INPUT.read_bytes()) != plan["input_sha256"]:
        raise ValueError("chord current-control checkpoint changed")
    return plan, digest


def run() -> dict[str, Any]:
    plan, plan_sha = _load_plan()
    if OUTPUT.exists():
        raise ValueError("chord output exists; no implicit repeat")
    prior = json.loads(INPUT.read_text())
    current = prior["terminal_point"]
    control = torch.tensor(current["control"], dtype=torch.float64)
    coordinates = torch.tensor(current["coordinates"], dtype=torch.float64)
    if _tensor_sha(control) != plan["input_control_sha256"] or _tensor_sha(control) != current["control_sha256"]:
        raise ValueError("chord input control does not match the exact terminal receipt")
    if current["first_order_pass"] or current["branch"]["unique_nearzero_face_entities"] != resume.FACE_ENTITIES:
        raise ValueError("chord input is not the recorded supported nonstationary two-face point")
    hessian_record = prior["fresh_hessians"][-1]
    metric_control_sha = hessian_record["control_sha256"]
    metric_raw = torch.tensor(hessian_record["hessian"], dtype=control.dtype)
    metric_record = resume._hessian_metrics(metric_raw)
    metric = metric_record["symmetric"] + metric_record["modified_shift"] * torch.eye(24, dtype=control.dtype)
    if not metric_record["positive_definite"] or hessian_record["profile_match"] is not True:
        raise ValueError("latest exact 24D Hessian is not a qualified SPD chord metric")
    start = time.monotonic()
    base._RUN_STARTED = start
    base.OUTPUT = OUTPUT
    deadline = start + float(plan["budget"]["phase_wall_seconds"])
    case = base.qualifier.cycle.load_fixed_case()
    problem, original, truth, parameters = (case["problem"], case["original"], case["truth"], case["parameters"])
    if _tensor_sha(parameters) != plan["parameters_sha256"]:
        raise ValueError("fixed parameters differ from the chord plan")
    profile, profile_identity = base._profile_problem(problem)
    basis = problem.frozen.fv_transport.psi_basis
    limits = problem.frozen.fv_transport.coefficient_limits
    matrix, pivots, free = base._face_matrix(basis, limits)
    lift, reconstructed = base._build_lift(control, matrix, pivots, free)
    if not torch.equal(reconstructed, coordinates) or _tensor_sha(lift(coordinates)) != _tensor_sha(control):
        raise ValueError("terminal control does not roundtrip the saved 24D chart coordinates")
    state: dict[str, Any] = {
        "status": "in_progress", "phase": "same_task_one_chord_input_ready",
        "ledger": {
            "original_wall_cap_seconds": plan["budget"]["original_wall_cap_seconds"],
            "charged_before_phase_seconds": plan["budget"]["charged_seconds_before_phase"],
            "phase_wall_cap_seconds": plan["budget"]["phase_wall_seconds"],
            "remaining_at_phase_start_seconds": plan["budget"]["estimated_remaining_seconds_before_phase"],
            "reserved_after_phase_seconds": plan["budget"]["reserved_after_phase_seconds"],
            "native_J_calls_before_phase": plan["budget"]["native_J_calls_before"],
            "native_J_cumulative_cap": plan["budget"]["native_J_cumulative_cap"],
            "gradient_calls_before_phase": plan["budget"]["gradient_calls_before_phase"],
            "additional_gradient_call_cap": plan["budget"]["additional_gradient_call_cap"],
            "hessian_rows_before_phase": plan["budget"]["hessian_rows_before_phase"],
            "additional_hessian_row_cap": plan["budget"]["additional_hessian_row_cap"],
        },
        "plan_sha256": plan_sha,
        "input_result_sha256": _sha_bytes(INPUT.read_bytes()),
        "input_control_sha256": _tensor_sha(control),
        "input_native_J": float(current["native_J"]),
        "metric_source_control_sha256": metric_control_sha,
        "metric_source_hessian_sha256": hessian_record["hessian_sha256"],
        "metric_source_age_control_norm": float(torch.linalg.vector_norm(
            torch.tensor(hessian_record["control"], dtype=control.dtype) - control
        )) if "control" in hessian_record else None,
        "metric_interpretation": "exact Hessian at its recorded earlier point; reused only as a fixed SPD descent metric for this one same-task chord",
        "profile_identity": profile_identity,
        "source_before": plan["effective_source_hashes"],
        "transport_extension_transition": plan["transport_extension_transition"],
        "native_J_calls_before_phase": plan["budget"]["native_J_calls_before"],
        "native_J_calls_started": 0,
        "native_J_calls_completed": plan["budget"]["native_J_calls_before"],
        "native_J_cumulative_cap": plan["budget"]["native_J_cumulative_cap"],
        "gradient_calls_started": 0,
        "gradient_calls_completed": 0,
        "gradient_call_cap": plan["budget"]["additional_gradient_call_cap"],
        "hessian_rows_started": 0,
        "hessian_rows_completed": 0,
        "hessian_row_cap": plan["budget"]["additional_hessian_row_cap"],
        "profile_scalar_calls": 0,
        "profile_scalar_call_cap": 4,
        "basis": basis,
        "limits": limits,
        "current_control": control.tolist(),
        "current_control_sha256": _tensor_sha(control),
        "native_J": float(current["native_J"]),
        "iterations": [],
        "accepted_steps": [],
        "memory_cap_bytes": RSS_CAP_BYTES,
    }
    _write(state)
    try:
        factor = torch.linalg.cholesky(metric)
    except RuntimeError as error:
        state.update(status="chord_metric_not_spd", phase="metric_factor_gate", error=str(error))
        _write(state)
        return state
    tangent_gradients = torch.tensor([s["tangent_gradient"] for s in current["sectors"]],
                                     dtype=control.dtype)
    gradient = tangent_gradients.mean(dim=0)
    step = -torch.cholesky_solve(gradient[:, None], factor).squeeze(1)
    descent = float(torch.dot(gradient, step))
    step_record = {"kind": "one_fixed_H_chord", "from_control_sha256": _tensor_sha(control),
                   "native_J": float(current["native_J"]),
                   "tangent_gradient_inf": float(gradient.abs().max()),
                   "descent_product": descent, "step_norm": float(torch.linalg.vector_norm(step)),
                   "metric_source_control_sha256": metric_control_sha, "line_search": []}
    state["iterations"].append(step_record)
    _write(state)
    if not math.isfinite(descent) or descent >= 0:
        state.update(status="chord_direction_not_descent", phase="descent_gate")
        _write(state)
        return state
    accepted = False
    for backtrack in range(27):  # Leave room for one fresh branch audit under the cumulative J cap.
        alpha = 0.5**backtrack
        trial_coordinates = coordinates + alpha * step
        try:
            trial_control = lift(trial_coordinates)
        except (ValueError, RuntimeError) as error:
            step_record["line_search"].append({"alpha": alpha, "status": "chart_domain_reject", "error": str(error)})
            _write(state)
            continue
        if state["native_J_calls_completed"] >= state["native_J_cumulative_cap"] - 1:
            state.update(status="native_J_budget_preserves_branch_audit", phase="line_search_cap_gate")
            _write(state)
            break
        value = base._native_value(problem, trial_control, parameters, state, deadline)
        rhs = float(current["native_J"]) + 1.0e-4 * alpha * descent
        item = {"alpha": alpha, "control_sha256": _tensor_sha(trial_control),
                "native_J": value, "armijo_rhs": rhs,
                "status": "accepted" if value <= rhs else "Armijo_reject"}
        step_record["line_search"].append(item)
        _write(state)
        if value <= rhs:
            accepted = True
            control, coordinates = trial_control, trial_coordinates
            state["accepted_steps"].append({"alpha": alpha, "native_J": value,
                "control_sha256": _tensor_sha(control), "step_norm": alpha * float(torch.linalg.vector_norm(step)),
                "metric_source_control_sha256": metric_control_sha})
            state["current_control"] = control.tolist()
            state["current_control_sha256"] = _tensor_sha(control)
            state["native_J"] = value
            _write(state)
            break
    if not accepted:
        state.update(status="one_chord_Armijo_exhausted", phase="line_search_failed")
        _write(state)
    else:
        point = base._audit_point(problem, parameters, control, lift, coordinates,
                                  matrix, pivots, free, profile, state, deadline,
                                  compute_hessian=False)
        point = resume._requalify_roundoff(point, lift, coordinates)
        state["current_point"] = point
        state["native_J"] = point["native_J"]
        state["current_control"] = control.tolist()
        state["current_control_sha256"] = _tensor_sha(control)
        if not point["branch_supported_exactly_two_faces"]:
            state.update(status="unsupported_new_face_or_limiter_event", phase="branch_gate")
        elif not point["normal_rank_two"] or not point["all_normal_signs_strict"]:
            state.update(status="unsupported_normal_release", phase="normal_gate")
        elif point["first_order_pass"]:
            passed = resume._terminal_p2(state, profile, problem, parameters,
                                         lift, coordinates, control, point, deadline)
            state["status"] = "qualified_original_native_J_two_face_minimum" if passed else "two_face_stationary_P2_failed"
        else:
            state.update(status="one_chord_nonstationary", phase="terminal_first_order_not_met")
            state["terminal_point"] = point
        _write(state)
    state["elapsed_seconds"] = time.monotonic() - start
    state["peak_rss_bytes"] = _rss_bytes()
    state["native_J_calls_completed_total"] = state["native_J_calls_completed"]
    state["gradient_calls_completed_total"] = plan["budget"]["gradient_calls_before_phase"] + state["gradient_calls_completed"]
    state["hessian_rows_completed_total"] = plan["budget"]["hessian_rows_before_phase"] + state["hessian_rows_completed"]
    state["ledger"]["charged_at_phase_end_seconds"] = plan["budget"]["charged_seconds_before_phase"] + state["elapsed_seconds"]
    state["ledger"]["estimated_remaining_at_phase_end_seconds"] = 3600.0 - state["ledger"]["charged_at_phase_end_seconds"]
    state["ledger"]["cumulative_native_J_calls_at_phase_end"] = state["native_J_calls_completed"]
    state["source_after"] = plan["effective_source_hashes"]
    state["source_archive_closure"] = all(_sha_bytes((ROOT / name).read_bytes()) == digest
        for name, digest in plan["source_files"].items()) and all(_sha_bytes((ROOT / name).read_bytes()) == digest
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
