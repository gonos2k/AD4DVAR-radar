"""Saved-array-only audit for up to three mixing-minimum iterations.

No model/project modules are imported. This script only reads already-saved
JSON arrays and receipts; it never evaluates FV or creates gradients/HVPs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

EVIDENCE = Path(__file__).resolve().parent
ATTEMPT = EVIDENCE / "tangent_mixing_minimum_resume_20261010_attempt1"
PLAN = EVIDENCE / "TANGENT_MIXING_MINIMUM_RESUME_PLAN_20261010.json"
EPS = 2.220446049250313e-16
EPS128 = 128.0 * EPS
FACE_SCALE = 0.84
J_C1 = 1e-4
F_C1 = 1e-4
CONTROL_SIZE = 26


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def vector_sha(values: list[float]) -> str:
    return sha_bytes(struct.pack(f"<{len(values)}d", *(float(x) for x in values)))


def dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("saved vectors have mismatched lengths")
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def norm(values: list[float]) -> float:
    return math.sqrt(max(0.0, dot(values, values)))


def difference(left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("saved vectors have mismatched lengths")
    return [float(a) - float(b) for a, b in zip(left, right)]


def close(a: float, b: float, factor: float = 128.0) -> bool:
    scale = max(abs(a), abs(b), float.fromhex("0x1.0p-1022"))
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= factor * EPS * scale


def gradient_summary(values: list[float]) -> dict[str, Any]:
    blocks = {"initial_field_20": values[:20], "flow_5": values[20:25], "growth_1": values[25:26]}
    return {"l2": norm(values), "linf": max((abs(float(x)) for x in values), default=0.0),
        "blocks_l2": {name: norm(block) for name, block in blocks.items()}}


def mixed_gradient(gradients: dict[str, list[float]], theta: float) -> list[float]:
    minus, plus = gradients["-1"], gradients["1"]
    if len(minus) != len(plus):
        raise ValueError("saved side gradients have mismatched lengths")
    return [(1.0 - theta) * float(a) + theta * float(b) for a, b in zip(minus, plus)]


def theta_minimum(gradients: dict[str, list[float]]) -> dict[str, Any]:
    minus, plus = gradients["-1"], gradients["1"]
    jump = difference(plus, minus)
    jump_norm = norm(jump)
    jump_budget = EPS128 * max(norm(minus), norm(plus), float.fromhex("0x1.0p-1022"))
    jump_squared = dot(jump, jump)
    theta = -dot(minus, jump) / jump_squared if jump_squared > 0.0 else math.nan
    valid = (math.isfinite(theta) and jump_norm > jump_budget
        and EPS128 < theta < 1.0 - EPS128)
    mixture = mixed_gradient(gradients, theta) if math.isfinite(theta) else []
    return {"theta_star": theta, "valid_strict_interior": valid,
        "boundary_distance": min(theta, 1.0 - theta) if math.isfinite(theta) else None,
        "jump_norm": jump_norm, "jump_budget": jump_budget, "jump_resolved": jump_norm > jump_budget,
        "stationarity_dot": dot(mixture, jump) if mixture else None,
        "minimum_gradient": gradient_summary(mixture) if mixture else None,
        "minimum_gradient_squared": dot(mixture, mixture) if mixture else None}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_child(path: Path) -> tuple[bytes, dict[str, Any]]:
    stored = path.read_bytes()
    data = gzip.decompress(stored) if path.suffix == ".gz" else stored
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("child receipt must be a JSON object")
    return data, value


def load_predecessor(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    producer_path = root / plan["producing_plan"]
    if hashlib.sha256(producer_path.read_bytes()).hexdigest() != plan["producing_plan_sha256"]:
        raise ValueError("predecessor plan digest mismatch")
    archive_path, run_path = root / plan["base_archive"], root / plan["base_run"]
    resource_path, manifest_path = root / plan["base_resource"], root / plan["base_archive_manifest"]
    for path, key in ((archive_path, "base_archive_sha256"), (run_path, "base_run_sha256"),
                      (resource_path, "base_resource_sha256"),
                      (manifest_path, "base_archive_manifest_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != plan[key]:
            raise ValueError(f"predecessor receipt digest mismatch: {path}")
    stored = archive_path.read_bytes()
    raw_bytes = gzip.decompress(stored) if archive_path.suffix == ".gz" else stored
    if sha_bytes(raw_bytes) != plan["base_child_sha256"]:
        raise ValueError("predecessor child digest mismatch")
    predecessor = json.loads(raw_bytes)
    manifest, parent, resource = read_json(manifest_path), read_json(run_path), read_json(resource_path)
    if (manifest.get("raw_sha256") != plan["base_child_sha256"]
            or manifest.get("gzip_sha256") != plan["base_archive_sha256"]
            or manifest.get("run_sha256") != plan["base_run_sha256"]
            or manifest.get("resource_sha256") != plan["base_resource_sha256"]
            or parent.get("execution_status") != "completed"
            or parent.get("child_sha256") != plan["base_child_sha256"]
            or parent.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None or resource.get("received_sigterm") is not False):
        raise ValueError("predecessor execution/archive receipt does not close")
    endpoint = predecessor.get("last_confirmed_closure")
    if not isinstance(endpoint, dict):
        raise ValueError("predecessor has no last confirmed closure")
    control = endpoint.get("control")
    if (not isinstance(control, list) or len(control) != CONTROL_SIZE
            or vector_sha([float(x) for x in control]) != plan["base_control_sha256"]
            or predecessor.get("current_control_sha256") != plan["base_control_sha256"]
            or predecessor.get("current_theta") != plan["initial_theta"]
            or endpoint.get("theta") != plan["initial_theta"]
            or endpoint.get("objective") != plan["base_objective"]):
        raise ValueError("predecessor endpoint does not match this resume plan")
    return predecessor


def static_qy(control: list[float], row: int, column: int) -> float:
    """Five-mode saved-control static flux diagnostic; no model code is imported."""
    amplitudes = (0.11, 0.08, 0.07, 0.04, 0.03)
    flow = [math.tanh(float(control[20 + i])) for i in range(5)]

    def psi(y: int, x: int) -> float:
        basis = (float(y), float(x), float(x * y),
            0.5 * float(x * x - y * y), float(x * x * y))
        return math.fsum(amplitudes[k] * flow[k] * basis[k] for k in range(5))

    return -(psi(row, column + 1) - psi(row, column))


def per_side_repeat(trial: dict[str, Any] | None,
                    repeat: dict[str, Any] | None) -> dict[str, Any]:
    sides: dict[str, Any] = {}
    if not isinstance(trial, dict) or not isinstance(repeat, dict):
        return {"passed": False, "sides": sides}
    expected, actual = trial.get("side_gradients", {}), repeat.get("side_gradients", {})
    for side in ("-1", "1"):
        left, right = expected.get(side, []), actual.get(side, [])
        error = max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=math.inf)
        scale = max(norm([float(x) for x in left]), norm([float(x) for x in right]),
            float.fromhex("0x1.0p-1022"))
        budget = EPS128 * scale
        sides[side] = {"max_abs_error": error, "budget": budget,
            "matched": len(left) == len(right) == CONTROL_SIZE and error <= budget}
    return {"passed": all(value["matched"] for value in sides.values()), "sides": sides}


def model_actual_delta(base_residual: list[float], residual_direction: list[float],
                       alpha: float, actual_residual: list[float]) -> dict[str, Any]:
    predicted_delta = [alpha * float(x) for x in residual_direction]
    actual_delta = difference(actual_residual, base_residual)
    error_vector = difference(actual_delta, predicted_delta)
    predicted_norm, actual_norm = norm(predicted_delta), norm(actual_delta)
    error_norm = norm(error_vector)
    cosine = (dot(predicted_delta, actual_delta) / (predicted_norm * actual_norm)
        if predicted_norm > 0.0 and actual_norm > 0.0 else None)
    delta_angle = (math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if cosine is not None and math.isfinite(cosine) else None)
    predicted_residual = [float(a) + float(b) for a, b in zip(base_residual, predicted_delta)]
    predicted_residual_norm, actual_residual_norm = norm(predicted_residual), norm(actual_residual)
    full_cosine = (dot(predicted_residual, actual_residual)
        / (predicted_residual_norm * actual_residual_norm)
        if predicted_residual_norm > 0.0 and actual_residual_norm > 0.0 else None)
    full_angle = (math.degrees(math.acos(max(-1.0, min(1.0, full_cosine))))
        if full_cosine is not None and math.isfinite(full_cosine) else None)
    base_f2, predicted_f2, actual_f2 = (dot(base_residual, base_residual),
        dot(predicted_residual, predicted_residual), dot(actual_residual, actual_residual))
    predicted_f2_drop = base_f2 - predicted_f2
    actual_f2_drop = base_f2 - actual_f2
    return {"model_delta_norm": predicted_norm, "actual_delta_norm": actual_norm,
        "model_to_actual_delta_norm_ratio": predicted_norm / actual_norm if actual_norm else None,
        "delta_vector_error_l2": error_norm,
        "delta_vector_relative_error": error_norm / actual_norm if actual_norm else None,
        "delta_vector_angle_degrees": delta_angle,
        "full_residual_angle_degrees": full_angle,
        "predicted_residual": predicted_residual,
        "actual_residual": actual_residual,
        "base_F_squared": base_f2, "predicted_F_squared": predicted_f2,
        "actual_F_squared": actual_f2,
        "predicted_F_squared_decrease": predicted_f2_drop,
        "predicted_F_squared_decrease_percent": 100.0 * predicted_f2_drop / base_f2 if base_f2 else None,
        "actual_F_squared_decrease": actual_f2_drop,
        "actual_F_squared_decrease_percent": 100.0 * actual_f2_drop / base_f2 if base_f2 else None,
        "actual_to_predicted_F_squared_decrease_ratio": (
            actual_f2_drop / predicted_f2_drop if predicted_f2_drop else None),
        "model_residual_relative_error": norm(difference(predicted_residual, actual_residual))
            / max(norm(actual_residual), float.fromhex("0x1.0p-1022"))}


def _step_summary(index: int, item: dict[str, Any], base_control: list[float],
        carried_theta: float, base_j: float, base_gradients: dict[str, list[float]],
        policy: dict[str, Any], hvp_history: list[dict[str, Any]]) -> tuple[
            dict[str, Any], list[float], float, float, dict[str, list[float]], bool]:
    arms = item.get("model_comparisons", [])
    arm = next((row for row in arms if row.get("name") == "candidate_mixing_minimum"), None)
    trials = arm.get("trials", []) if isinstance(arm, dict) else []
    accepted_trials = [trial for trial in trials if trial.get("accepted") is True]
    selected = accepted_trials[0] if len(accepted_trials) == 1 else None
    repeat = item.get("final_repeat") if isinstance(item.get("final_repeat"), dict) else None
    base_sha = vector_sha(base_control)
    working_theta = float(arm.get("working_theta", item.get("working_theta", math.nan))) if arm else math.nan
    base_min = theta_minimum(base_gradients)
    face_scale = float(policy.get("face_scale", FACE_SCALE))
    q_base = float(arm.get("residual", [0.0])[-1]) * face_scale if arm and arm.get("residual") else 0.0
    carried_gradient = mixed_gradient(base_gradients, carried_theta)
    working_gradient = mixed_gradient(base_gradients, working_theta) if math.isfinite(working_theta) else []
    carried_residual = carried_gradient + [q_base / face_scale]
    working_residual = working_gradient + [q_base / face_scale]
    carried_f2, working_f2 = dot(carried_residual, carried_residual), dot(working_residual, working_residual)

    direction = [float(x) for x in arm.get("direction", [])] if arm else []
    direction_sha = vector_sha(direction) if len(direction) == CONTROL_SIZE else None
    hvps = [row for row in hvp_history if row.get("base_control_sha256") == base_sha
        and row.get("direction_model") == "candidate_mixing_minimum"]
    hvp_sides = {str(side): [row for row in hvps if row.get("side") == side] for side in (-1, 1)}
    hvp_checks = {"count": len(hvps), "exact_two": len(hvps) == 2,
        "one_per_side": all(len(rows) == 1 for rows in hvp_sides.values()),
        "all_completed": all(row.get("status") == "completed" for row in hvps),
        "carried_theta_matches": all(row.get("theta") == carried_theta for row in hvps),
        "working_theta_matches": all(row.get("working_theta") == working_theta for row in hvps),
        "direction_hash_matches": all(row.get("direction_sha256") == direction_sha for row in hvps),
        "operator_scope_matches": all(row.get("operator") == "selected_face_extension"
            and row.get("scope") == "one current chart tangent" for row in hvps)}
    chain_base_matches = item.get("base_control_sha256") == base_sha and item.get("theta") == carried_theta
    accepted = item.get("accepted") is True and selected is not None and isinstance(repeat, dict)
    repeat_gradients = per_side_repeat(selected, repeat)
    candidate: dict[str, Any] | None = None

    if selected is not None:
        candidate_control = [float(x) for x in selected.get("control", [])]
        movement = difference(candidate_control, base_control) if candidate_control else []
        movement_l2 = norm(movement)
        candidate_grads = selected.get("side_gradients", {})
        candidate_min = (theta_minimum(candidate_grads)
            if all(side in candidate_grads for side in ("-1", "1")) else {})
        candidate_theta = selected.get("candidate_theta_star", selected.get("theta"))
        candidate_q = float(selected.get("face_value", 0.0))
        candidate_mix = (mixed_gradient(candidate_grads, float(candidate_theta))
            if isinstance(candidate_theta, (int, float))
                and all(side in candidate_grads for side in ("-1", "1")) else [])
        actual_residual = candidate_mix + [candidate_q / face_scale] if candidate_mix else []
        f2_recomputed = dot(actual_residual, actual_residual) if actual_residual else None
        j_slope = (max(dot(base_gradients["-1"], direction), dot(base_gradients["1"], direction))
            if len(direction) == CONTROL_SIZE else math.nan)
        alpha = float(selected.get("alpha", math.nan))
        j_bound = base_j + J_C1 * alpha * j_slope
        merit_slope = float(item.get("merit_product", math.nan))
        f_bound = working_f2 + 2.0 * F_C1 * alpha * merit_slope
        trial_j, trial_f2 = float(selected.get("objective", math.nan)), float(selected.get("F_squared", math.nan))
        working_summary = gradient_summary(working_gradient) if working_gradient else None
        candidate_summary = gradient_summary(candidate_mix) if candidate_mix else None
        closure_keys = ("side_gradients_finite", "native_objective_matches_proposal",
            "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
            "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
            "fixed_input_unchanged", "runtime_unchanged", "deadline_passed",
            "mixing_minimum_valid", "minimum_theta_matches_proposal")
        closure_gates = ({key: repeat.get(key) is True for key in closure_keys}
            if isinstance(repeat, dict) else {})
        candidate = {"accepted_trial_count": len(accepted_trials), "alpha": alpha,
            "theta_star": candidate_theta, "theta_star_recomputed": candidate_min.get("theta_star"),
            "theta_star_matches_saved": isinstance(candidate_theta, (int, float))
                and close(float(candidate_theta), candidate_min.get("theta_star", math.nan)),
            "theta_star_boundary_distance": candidate_min.get("boundary_distance"),
            "minimum_theta_valid": candidate_min.get("valid_strict_interior"),
            "jump_norm": candidate_min.get("jump_norm"), "jump_budget": candidate_min.get("jump_budget"),
            "jump_resolved": candidate_min.get("jump_resolved"),
            "minimum_F_squared_recomputed": f2_recomputed, "minimum_F_squared_saved": trial_f2,
            "minimum_F_squared_matches_saved": (isinstance(f2_recomputed, (int, float))
                and close(float(f2_recomputed), trial_f2)),
            "base_minimum_F_squared": working_f2,
            "minimum_F_squared_decrease": (working_f2 - trial_f2) if math.isfinite(trial_f2) else None,
            "minimum_F_squared_decrease_percent": (
                100.0 * (working_f2 - trial_f2) / working_f2 if working_f2 > 0.0 else None),
            "base_minimum_gradient_l2": working_summary["l2"] if working_summary else None,
            "base_minimum_gradient_linf": working_summary["linf"] if working_summary else None,
            "minimum_gradient_l2_decrease": (working_summary["l2"]
                - candidate_summary["l2"] if working_summary and candidate_summary else None),
            "minimum_gradient_l2_decrease_percent": (100.0 * (
                working_summary["l2"] - candidate_summary["l2"]) / working_summary["l2"]
                if working_summary and candidate_summary and working_summary["l2"] > 0.0 else None),
            "minimum_gradient_linf_decrease": (working_summary["linf"]
                - candidate_summary["linf"] if working_summary and candidate_summary else None),
            "minimum_gradient_linf_decrease_percent": (100.0 * (
                working_summary["linf"] - candidate_summary["linf"]) / working_summary["linf"]
                if working_summary and candidate_summary and working_summary["linf"] > 0.0 else None),
            "native_J": trial_j, "control_sha256": vector_sha(candidate_control)
                if len(candidate_control) == CONTROL_SIZE else None,
            "base_native_J": base_j,
            "native_J_decrease": base_j - trial_j,
            "minimum_F_squared_decrease_from_base_percent": (
                100.0 * (1.0 - trial_f2 / working_f2) if working_f2 > 0.0 else None),
            "native_J_decrease_from_base_percent": (
                100.0 * (1.0 - trial_j / base_j) if base_j != 0.0 else None),
            "movement_l2": movement_l2, "movement_linf": max(map(abs, movement), default=0.0),
            "movement_l2_saved": selected.get("actual_path_norm"),
            "movement_l2_matches_saved": close(movement_l2,
                float(selected.get("actual_path_norm", math.nan))),
            "movement_nonzero": movement_l2 > 0.0,
            "within_radius": math.isfinite(movement_l2) and movement_l2 > 0.0
                and movement_l2 <= float(policy.get("radius", math.nan)) * (1.0 + 64.0 * EPS),
            "J_armijo_saved": selected.get("J_armijo_passed"),
            "J_armijo_recomputed": trial_j <= j_bound,
            "J_armijo_bound_saved": selected.get("J_armijo_bound"),
            "J_armijo_bound_recomputed": j_bound,
            "F_squared_armijo_saved": selected.get("F_squared_armijo_passed"),
            "F_squared_armijo_recomputed": trial_f2 <= f_bound,
            "F_squared_armijo_bound_saved": selected.get("F_squared_armijo_bound"),
            "F_squared_armijo_bound_recomputed": f_bound,
            "face_Q32_base": static_qy(base_control, 3, 2),
            "face_Q43_base": static_qy(base_control, 4, 3),
            "face_Q32_candidate": static_qy(candidate_control, 3, 2),
            "face_Q43_candidate": static_qy(candidate_control, 4, 3),
            "candidate_gradient_summary": candidate_summary,
            "repeat_P2": repeat_gradients,
            "trial_repeat_control_equal": isinstance(repeat, dict)
                and selected.get("control") == repeat.get("control"),
            "trial_repeat_trace_signatures_match": (isinstance(repeat, dict)
                and all(selected.get("branch_trace", {}).get(side, {}).get("signature_sha256")
                    == repeat.get("branch_trace", {}).get(side, {}).get("signature_sha256")
                    for side in ("-1", "1"))),
            "closure_gates": closure_gates,
            "repeat_all_required_closure_gates": bool(closure_gates) and all(closure_gates.values())}
        if isinstance(repeat, dict):
            candidate.update(endpoint_J=repeat.get("objective"), endpoint_F_squared=repeat.get("F_squared"),
                repeat_theta_star=repeat.get("recomputed_mixing_theta_star"),
                repeat_minimum_theta_matches=repeat.get("minimum_theta_matches_proposal"),
                repeat_J_matches_trial=close(float(repeat.get("objective", math.nan)), trial_j),
                repeat_F_squared_matches_trial=close(float(repeat.get("F_squared", math.nan)), trial_f2))
        residual = [float(x) for x in arm.get("residual", [])]
        residual_direction = [float(x) for x in arm.get("residual_direction", [])]
        if len(actual_residual) == len(residual) == len(residual_direction) and math.isfinite(alpha):
            candidate["model_delta_vs_actual"] = model_actual_delta(
                residual, residual_direction, alpha, actual_residual)
        else:
            candidate["model_delta_vs_actual"] = {"available": False,
                "reason": "residual vector lengths or alpha mismatch"}
        candidate["commit_chain_matches"] = (accepted
            and item.get("committed_control") == selected.get("control")
            and item.get("committed_theta") == selected.get("theta")
            and repeat.get("control") == selected.get("control")
            and repeat.get("theta") == selected.get("theta"))
        candidate["unaccepted_base_preserved"] = (accepted or (
            item.get("committed_control", base_control) == base_control
            and item.get("committed_theta", carried_theta) == carried_theta))
    elif trials:
        candidate = {"accepted_trial_count": 0,
            "trial_statuses": [row.get("status") for row in trials],
            "zero_movement_refusals": sum(row.get("status") == "zero_control_movement_refused" for row in trials),
            "base_preserved_by_iteration": (item.get("committed_control", base_control) == base_control
                and item.get("committed_theta", carried_theta) == carried_theta)}

    diagnostics = arm.get("diagnostics", {}) if isinstance(arm, dict) else {}
    gates = arm.get("gates", {}) if isinstance(arm, dict) else {}
    residual = [float(x) for x in arm.get("residual", [])] if arm else []
    current = {"index": index, "base_control_sha256": item.get("base_control_sha256"),
        "expected_base_control_sha256": base_sha, "base_chain_matches": chain_base_matches,
        "carried_theta": item.get("theta"), "expected_carried_theta": carried_theta,
        "working_theta_star": working_theta, "base_theta_star_recomputed": base_min.get("theta_star"),
        "base_theta_star_matches_working": close(working_theta, base_min.get("theta_star", math.nan)),
        "base_theta_boundary_distance": base_min.get("boundary_distance"),
        "base_minimum_valid": base_min.get("valid_strict_interior"),
        "base_jump_norm": base_min.get("jump_norm"), "base_jump_budget": base_min.get("jump_budget"),
        "base_jump_resolved": base_min.get("jump_resolved"),
        "base_theta_stationarity_dot": base_min.get("stationarity_dot"),
        "base_carried_F_squared_recomputed": carried_f2,
        "base_minimum_F_squared_recomputed": working_f2,
        "base_minimum_F_squared_decrease_from_carried_percent": (
            100.0 * (1.0 - working_f2 / carried_f2) if carried_f2 > 0.0 else None),
        "base_minimum_F_squared_saved_residual": dot(residual, residual) if residual else None,
        "base_native_J": base_j,
        "base_minimum_F_squared_matches_saved_residual": (bool(residual)
            and close(working_f2, dot(residual, residual))),
        "base_minimum_gradient_summary": gradient_summary(working_gradient) if working_gradient else None,
        "direction_sha256": direction_sha, "direction_norm": norm(direction) if direction else None,
        "direction_normal_residual": diagnostics.get("direction_normal_residual"),
        "gradient_jump_tangent_norm": gates.get("gradient_jump_tangent_norm"),
        "gradient_jump_support_budget": gates.get("gradient_jump_support_budget"),
        "common_face_gradient_jump": gates.get("common_face_gradient_jump"),
        "base_face_Q32": static_qy(base_control, 3, 2), "base_face_Q43": static_qy(base_control, 4, 3),
        "carried_theta_reweighting_is_optimizer_step": False,
        "candidate_count": len(trials), "candidate": candidate,
        "hvp_provenance": hvp_checks, "model_merit_slope": item.get("merit_product"),
        "candidate_accepted": accepted, "iteration_final_closure_accepted": item.get("accepted") is True,
        "unaccepted_base_preserved": (accepted or (
            item.get("committed_control", base_control) == base_control
            and item.get("committed_theta", carried_theta) == carried_theta)),
        "refusal": item.get("refusal")}
    if accepted and selected is not None and isinstance(repeat, dict):
        next_control = [float(x) for x in repeat["control"]]
        next_theta = float(repeat["theta"])
        next_j = float(repeat["objective"])
        next_gradients = {side: [float(x) for x in repeat["side_gradients"][side]] for side in ("-1", "1")}
    else:
        next_control, next_theta, next_j, next_gradients = base_control, carried_theta, base_j, base_gradients
    current["terminal_control_sha256"] = vector_sha(next_control)
    current["terminal_theta"] = next_theta
    return current, next_control, next_theta, next_j, next_gradients, accepted


def collect(child_path: Path, run_path: Path, resource_path: Path, plan_path: Path) -> dict[str, Any]:
    root = EVIDENCE.parents[1]
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    raw_bytes, child = read_child(child_path)
    parent, resource = read_json(run_path), read_json(resource_path)
    predecessor = load_predecessor(plan, root)
    base_control = [float(x) for x in predecessor["last_confirmed_closure"]["control"]]
    base_gradients = {side: [float(x) for x in predecessor["last_confirmed_closure"]["side_gradients"][side]]
        for side in ("-1", "1")}
    base_sha, base_theta = vector_sha(base_control), float(plan["initial_theta"])
    base_j = float(predecessor["last_confirmed_closure"]["objective"])
    face_scale = float(plan.get("policy", {}).get("face_scale", FACE_SCALE))
    base_q = float(predecessor.get("final_face_audit", {}).get("value", 0.0))
    base_theta_info = theta_minimum(base_gradients)
    base_mix = mixed_gradient(base_gradients, float(base_theta_info["theta_star"]))
    base_min_residual = base_mix + [base_q / face_scale]
    base_min_f2 = dot(base_min_residual, base_min_residual)
    base_min_summary = gradient_summary(base_mix)

    source_receipt = {**plan.get("source_files", {}), **plan.get("archive_files", {}),
        plan_path.relative_to(root).as_posix(): sha_bytes(plan_bytes)}
    source_before, source_after = child.get("source_before", {}), child.get("source_after", {})
    source_receipt_matches = source_before == source_receipt and source_after == source_receipt
    iterations = child.get("iterations", [])
    if not isinstance(iterations, list):
        raise ValueError("child iteration chain must be a list")
    policy = plan.get("policy", {})
    max_steps = int(policy.get("max_accepted_iterations", 3))
    if len(iterations) > max_steps:
        raise ValueError("child exceeds its accepted-iteration cap")

    steps: list[dict[str, Any]] = []
    accepted_count = 0
    current_control, current_theta, current_j, current_gradients = (
        base_control, base_theta, base_j, base_gradients)
    refused_seen = False
    for index, item in enumerate(iterations):
        summary, next_control, next_theta, next_j, next_gradients, accepted = _step_summary(
            index, item, current_control, current_theta, current_j, current_gradients,
            policy, child.get("hvp_history", []))
        if refused_seen:
            summary["chain_after_refusal_invalid"] = True
        steps.append(summary)
        accepted_count += int(accepted)
        if accepted:
            current_control, current_theta, current_j, current_gradients = (
                next_control, next_theta, next_j, next_gradients)
        else:
            refused_seen = True

    expected_sha = vector_sha(current_control)
    terminal = child.get("current_control", child.get("last_confirmed_control", []))
    terminal_sha = vector_sha([float(x) for x in terminal]) if isinstance(terminal, list) else None
    terminal_theta = child.get("current_theta")
    terminal_q = float(child.get("final_face_audit", {}).get("value", 0.0))
    terminal_theta_info = theta_minimum(current_gradients)
    terminal_mix = mixed_gradient(current_gradients, float(terminal_theta_info["theta_star"]))
    terminal_residual = terminal_mix + [terminal_q / face_scale]
    terminal_min_f2 = dot(terminal_residual, terminal_residual)
    terminal_min_summary = gradient_summary(terminal_mix)
    history = child.get("hvp_history", [])
    side_counts = {str(side): sum(row.get("side") == side for row in history) for side in (-1, 1)}
    elapsed = resource.get("elapsed_seconds", math.inf)
    peak = resource.get("sampled_peak_rss_bytes", math.inf)
    resource_closed = (resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("received_sigterm") is False
        and resource.get("wall_limit_seconds") == policy.get("outer_seconds")
        and resource.get("rss_limit_bytes") == policy.get("rss_bytes")
        and elapsed <= policy.get("outer_seconds", -1)
        and peak < policy.get("rss_bytes", 0))
    hvp_total = int(child.get("hvp_calls_completed", -1))
    candidate_counts = [int(step.get("candidate_count", 0)) for step in steps]
    internal_elapsed = float(child.get("elapsed_seconds", math.inf))
    accounting = {"guarded_launches": policy.get("guarded_launches"),
        "accepted_iterations_saved": child.get("accepted_iterations"),
        "accepted_iterations_recomputed": accepted_count,
        "optimizer_steps_applied": child.get("optimizer_steps_applied"),
        "hvp_started": child.get("hvp_calls_started"), "hvp_completed": hvp_total,
        "hvp_within_budget": 0 <= hvp_total <= int(policy.get("hvp_calls", 0)),
        "hvp_started_completed_match": child.get("hvp_calls_started") == child.get("hvp_calls_completed"),
        "hvp_sides": side_counts, "jacobian_rows_started": child.get("jacobian_rows_started"),
        "jacobian_rows_completed": child.get("jacobian_rows_completed"),
        "jacobian_rows_within_budget": child.get("jacobian_rows_started") == 0
            and child.get("jacobian_rows_completed") == 0,
        "candidate_trials_by_iteration": candidate_counts,
        "candidate_trials_within_each_cap": all(count <= int(policy.get("max_candidates_per_iteration", 0))
            for count in candidate_counts),
        "candidate_trials_total": sum(candidate_counts),
        "candidate_slot_cap_per_iteration": policy.get("max_candidates_per_iteration"),
        "pcg_solves": policy.get("pcg_solves"), "dense_solves": policy.get("dense_solves"),
        "sampled_peak_rss_bytes": peak, "rss_limit_bytes": policy.get("rss_bytes"),
        "elapsed_seconds": elapsed, "internal_elapsed_seconds": internal_elapsed,
        "internal_seconds": policy.get("internal_seconds"),
        "internal_deadline_passed": internal_elapsed <= policy.get("internal_seconds", -1),
        "outer_seconds": policy.get("outer_seconds"),
        "accepted_iteration_count_matches": child.get("accepted_iterations") == accepted_count,
        "optimizer_step_count_matches": child.get("optimizer_steps_applied") == accepted_count}
    return {"scope": "Saved JSON arrays and receipts only; no FV, AD, gradient, or HVP generation",
        "plan_sha256": sha_bytes(plan_bytes), "raw_sha256": sha_bytes(raw_bytes),
        "predecessor_plan_sha256": plan.get("producing_plan_sha256"),
        "predecessor_child_sha256": plan.get("base_child_sha256"),
        "execution_status": child.get("execution_status"), "numerical_status": child.get("numerical_status"),
        "base": {"control_sha256": base_sha, "theta": base_theta, "native_J": base_j,
            "theta_star": base_theta_info,
            "F_squared_at_carried_theta": predecessor["last_confirmed_closure"].get("F_squared"),
            "F_squared_at_fresh_minimum": base_min_f2,
            "minimum_gradient_summary": base_min_summary,
            "face_Q": base_q,
            "per_side_gradients": {side: gradient_summary(base_gradients[side]) for side in ("-1", "1")},
            "face_Q32": static_qy(base_control, 3, 2), "face_Q43": static_qy(base_control, 4, 3)},
        "steps": steps,
        "terminal": {"expected_control_sha256": expected_sha, "saved_control_sha256": terminal_sha,
            "control_chain_matches": expected_sha == terminal_sha, "expected_theta": current_theta,
            "saved_theta": terminal_theta, "theta_chain_matches": terminal_theta == current_theta,
            "native_J": current_j, "face_Q": terminal_q,
            "theta_star": terminal_theta_info, "F_squared_at_fresh_minimum": terminal_min_f2,
            "minimum_gradient_summary": terminal_min_summary,
            "face_Q32": static_qy(current_control, 3, 2), "face_Q43": static_qy(current_control, 4, 3)},
        "total_decreases": {
            "native_J": {"absolute": base_j - current_j,
                "percent": 100.0 * (base_j - current_j) / base_j if base_j else None},
            "F_squared_at_fresh_minimum": {"absolute": base_min_f2 - terminal_min_f2,
                "percent": 100.0 * (base_min_f2 - terminal_min_f2) / base_min_f2 if base_min_f2 else None},
            "minimum_gradient_l2": {"absolute": base_min_summary["l2"] - terminal_min_summary["l2"],
                "percent": 100.0 * (base_min_summary["l2"] - terminal_min_summary["l2"])
                    / base_min_summary["l2"] if base_min_summary["l2"] else None},
            "minimum_gradient_linf": {"absolute": base_min_summary["linf"] - terminal_min_summary["linf"],
                "percent": 100.0 * (base_min_summary["linf"] - terminal_min_summary["linf"])
                    / base_min_summary["linf"] if base_min_summary["linf"] else None}},
        "source_receipt_matches_plan": source_receipt_matches,
        "parent_resource_matches_child_resource": parent.get("resource") == resource,
        "parent_child_digest_matches": parent.get("child_sha256") == sha_bytes(raw_bytes),
        "resource_closed": resource_closed, "source_before_after_equal": source_before == source_after,
        "accounting": accounting,
        "claims": {"full_smooth_root": child.get("full_smooth_root"),
            "minimum_claim": child.get("minimum_claim"), "response_claim": child.get("response_claim"),
            "score_claim": policy.get("score_claim")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, default=ATTEMPT / "step.json")
    parser.add_argument("--run", type=Path, default=ATTEMPT / "step.run.json")
    parser.add_argument("--resource", type=Path, default=ATTEMPT / "step.resource.json")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "MIXMIN_RESUME_RESULT_20261010.json")
    args = parser.parse_args()
    result = collect(args.child, args.run, args.resource, args.plan)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    keys = ("plan_sha256", "raw_sha256", "execution_status", "numerical_status",
        "base", "steps", "terminal", "source_receipt_matches_plan", "resource_closed", "accounting")
    print(json.dumps({key: result[key] for key in keys}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
