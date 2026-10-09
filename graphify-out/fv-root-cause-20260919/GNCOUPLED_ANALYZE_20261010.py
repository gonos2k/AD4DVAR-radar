"""Audit a saved coupled-GN comparison from JSON arrays only.

NumPy is used only for row-space linear algebra on already-saved arrays.
This script imports no FV/model code and never generates gradients or HVPs.
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

import numpy as np

EVIDENCE = Path(__file__).resolve().parent
ATTEMPT = EVIDENCE / "tangent_coupled_gn_comparison_20261010_attempt1"
PLAN = EVIDENCE / "TANGENT_COUPLED_GN_COMPARISON_PLAN_20261010.json"
FACE_SCALE = 0.84
EPS = np.finfo(np.float64).eps
EPS128 = 128.0 * EPS
J_C1 = F_C1 = 1e-4
N, M = 12, 26
LIMITS = np.array([0.11, 0.08, 0.07, 0.04, 0.03], dtype=np.float64)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def vsha(value: Any) -> str:
    array = np.asarray(value, dtype="<f8").reshape(-1)
    return sha_bytes(array.tobytes())


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object at {path}")
    return obj


def read_child(path: Path) -> tuple[bytes, dict[str, Any]]:
    stored = path.read_bytes()
    data = gzip.decompress(stored) if path.suffix == ".gz" else stored
    obj = json.loads(data)
    if not isinstance(obj, dict):
        raise ValueError("child must be a JSON object")
    return data, obj


def dot(a: Any, b: Any) -> float:
    return float(np.dot(np.asarray(a, dtype=np.float64).reshape(-1),
                        np.asarray(b, dtype=np.float64).reshape(-1)))


def norm(a: Any) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).reshape(-1)))


def rel_error(actual: Any, expected: Any) -> dict[str, float | bool]:
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_match": False, "max_abs_error": math.inf, "relative_l2_error": math.inf}
    error = a - b
    scale = max(norm(a), norm(b), np.finfo(np.float64).tiny)
    return {"shape_match": True, "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "relative_l2_error": norm(error) / scale}


def theta_minimum(gminus: Any, gplus: Any) -> dict[str, Any]:
    gm, gp = np.asarray(gminus, dtype=np.float64), np.asarray(gplus, dtype=np.float64)
    jump = gp - gm
    denominator = dot(jump, jump)
    theta = -dot(gm, jump) / denominator if denominator > 0.0 else math.nan
    mixed = gm + theta * jump if math.isfinite(theta) else np.full_like(gm, np.nan)
    scale = max(norm(gm), norm(gp), np.finfo(np.float64).tiny)
    budget = EPS128 * scale
    return {"theta_star": theta, "jump_norm": norm(jump), "jump_budget": budget,
        "jump_resolved": norm(jump) > budget,
        "boundary_distance": min(theta, 1.0-theta) if math.isfinite(theta) else None,
        "strict_interior": (math.isfinite(theta) and norm(jump) > budget
            and EPS128 < theta < 1.0-EPS128),
        "stationarity_dot": dot(mixed, jump),
        "minimum_gradient_l2": norm(mixed),
        "minimum_gradient_linf": float(np.max(np.abs(mixed)))}


def face_geometry(control: Any, face: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(control, dtype=np.float64)
    row, col, axis = int(face["row"]), int(face["column"]), str(face["axis"])

    def basis(y: int, xx: int) -> np.ndarray:
        return np.array([y, xx, y*xx, 0.5*(xx*xx-y*y), y*xx*xx], dtype=np.float64)

    if axis == "x":
        delta = basis(row+1, col) - basis(row, col)
    elif axis == "y":
        delta = -(basis(row, col+1) - basis(row, col))
    else:
        raise ValueError("face axis must be x or y")
    weights = delta * LIMITS
    flow = np.tanh(x[20:25])
    q = float(weights @ flow)
    normal = np.zeros(M, dtype=np.float64)
    normal[20:25] = weights * (1.0-flow*flow)
    return weights, normal, q


def static_qy(control: Any, row: int, col: int) -> float:
    x = np.asarray(control, dtype=np.float64)
    flow = np.tanh(x[20:25])
    def psi(y: int, xx: int) -> float:
        b = np.array([y, xx, y*xx, 0.5*(xx*xx-y*y), y*xx*xx], dtype=np.float64)
        return float((LIMITS*flow) @ b)
    return -(psi(row, col+1)-psi(row, col))


def load_predecessor(plan: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    producer_path = root / plan["producing_plan"]
    if sha_file(producer_path) != plan["producing_plan_sha256"]:
        raise ValueError("predecessor plan digest mismatch")
    archive, run = root / plan["base_archive"], root / plan["base_run"]
    resource, manifest = root / plan["base_resource"], root / plan["base_archive_manifest"]
    for path, key in ((archive, "base_archive_sha256"), (run, "base_run_sha256"),
                      (resource, "base_resource_sha256"),
                      (manifest, "base_archive_manifest_sha256")):
        if sha_file(path) != plan[key]:
            raise ValueError(f"predecessor receipt digest mismatch: {path}")
    stored = archive.read_bytes()
    raw_bytes = gzip.decompress(stored) if archive.suffix == ".gz" else stored
    if sha_bytes(raw_bytes) != plan["base_child_sha256"]:
        raise ValueError("predecessor raw child digest mismatch")
    prev, parent, resources, archive_meta = (json.loads(raw_bytes), read_json(run),
        read_json(resource), read_json(manifest))
    if (parent.get("execution_status") != "completed" or parent.get("child_sha256") != plan["base_child_sha256"]
            or parent.get("resource") != resources or resources.get("exit_code") != 0
            or resources.get("resource_termination") is not None
            or archive_meta.get("raw_sha256") != plan["base_child_sha256"]):
        raise ValueError("predecessor execution receipts do not close")
    endpoint = prev.get("last_confirmed_closure")
    if not isinstance(endpoint, dict) or vsha(endpoint.get("control", [])) != plan["base_control_sha256"]:
        raise ValueError("predecessor endpoint does not match planned base control")
    if endpoint.get("theta") != plan["initial_theta"] or endpoint.get("objective") != plan["base_objective"]:
        raise ValueError("predecessor endpoint theta/J differ from frozen base")
    return prev, endpoint


def build_rows(child: dict[str, Any], base_sha: str, theta: float) -> dict[str, Any]:
    history = child.get("jacobian_row_history", [])
    matrices = {side: np.full((N, M), np.nan) for side in (-1, 1)}
    residuals = {side: np.full(N, np.nan) for side in (-1, 1)}
    counts: dict[int, set[int]] = {-1: set(), 1: set()}
    for item in history:
        side, row = int(item["side"]), int(item["row"])
        if side not in (-1, 1) or not 0 <= row < N:
            raise ValueError("row receipt has an invalid side/index")
        if (item.get("status") != "completed" or item.get("base_control_sha256") != base_sha
                or item.get("theta") != theta or row in counts[side]):
            raise ValueError("row receipt status/base/theta/uniqueness check failed")
        grad = np.asarray(item["gradient"], dtype=np.float64)
        if grad.shape != (M,) or not np.isfinite(grad).all():
            raise ValueError("row gradient has invalid shape or nonfinite values")
        matrices[side][row] = grad
        residuals[side][row] = float(item["residual_value"])
        counts[side].add(row)
    complete = {str(s): len(counts[s]) == N and counts[s] == set(range(N)) for s in (-1, 1)}
    return {"A_minus": matrices[-1], "A_plus": matrices[1],
        "r_minus": residuals[-1], "r_plus": residuals[1],
        "count": len(history), "complete_by_side": complete,
        "finite": all(np.isfinite(matrices[s]).all() and np.isfinite(residuals[s]).all() for s in (-1,1))}


def component_metrics(error: np.ndarray) -> dict[str, Any]:
    e = np.asarray(error, dtype=np.float64).reshape(-1)
    total = float(e @ e)
    block = e[12:15]
    return {"c12_c15_error": block.tolist(),
        "c12_c15_squared_error_share_percent": 100.0*float(block@block)/total if total else None,
        "total_error_l2": norm(e)}


def arm_summary(arm: dict[str, Any], *, base_control: np.ndarray,
        base_j: float, base_gradients: dict[str, np.ndarray], theta_star: float,
        normal: np.ndarray, projector: np.ndarray, row_data: dict[str, Any],
        robust_d: np.ndarray, diagnostics: dict[str, Any], residual_q: float,
        face_scale: float) -> dict[str, Any]:
    name = arm["name"]
    direction = np.asarray(arm["direction"], dtype=np.float64)
    hminus = np.asarray(arm["hminus_direction"], dtype=np.float64)
    hplus = np.asarray(arm["hplus_direction"], dtype=np.float64)
    gm, gp = base_gradients["-1"], base_gradients["1"]
    jump = gp-gm
    gstar = (1.0-theta_star)*gm + theta_star*gp
    hmix = (1.0-theta_star)*hminus + theta_star*hplus
    theta_num = float(jump @ hmix + gstar @ (hplus-hminus))
    theta_den = float(jump @ jump)
    theta_prime = -theta_num/theta_den if theta_den > 0 else math.nan
    residual = np.concatenate((gstar, np.array([residual_q/face_scale])))
    dface = float(normal @ direction)/face_scale
    residual_direction = np.concatenate((hmix+jump*theta_prime, np.array([dface])))
    envelope_slope = float(gstar @ hmix + residual_q*(normal @ direction)/(face_scale**2))
    f_dot_df = float(residual @ residual_direction)
    df_saved = np.asarray(arm["residual_direction"], dtype=np.float64)
    r_saved = np.asarray(arm["residual"], dtype=np.float64)
    diagnostic = arm.get("diagnostics", {})
    candidates = arm.get("trials", [])
    accepted = [trial for trial in candidates if trial.get("accepted") is True]
    endpoint = accepted[0] if len(accepted) == 1 else None
    result = {"name": name, "direction_sha256": vsha(direction), "direction_norm": norm(direction),
        "tangent_normal_residual": float(normal @ direction),
        "base_theta_star": theta_star, "theta_prime_recomputed": theta_prime,
        "theta_prime_saved": arm.get("delta_theta"),
        "theta_prime_abs_error": abs(theta_prime-float(arm.get("delta_theta", math.nan))),
        "theta_prime_numerator": theta_num, "theta_prime_saved_numerator": arm.get("delta_theta_numerator"),
        "theta_prime_denominator": theta_den, "theta_prime_saved_denominator": arm.get("delta_theta_denominator"),
        "DF_recomputed": residual_direction.tolist(),
        "DF_saved_max_abs_error": float(np.max(np.abs(residual_direction-df_saved))),
        "residual_saved_max_abs_error": float(np.max(np.abs(residual-r_saved))),
        "F_dot_DF_recomputed": f_dot_df, "envelope_slope_recomputed": envelope_slope,
        "envelope_identity_error": abs(f_dot_df-envelope_slope),
        "merit_slope_saved": arm.get("merit_product"),
        "candidate_trials": len(candidates),
        "accepted_trial_count": len(accepted),
        "accepted_endpoint": None}

    if name == "robust_gn_coupled":
        # Reconstruct B/S/Woodbury solve from the independently saved row ledger.
        bcalc = np.sqrt(robust_d)[:, None] * (row_data["A_minus"] @ projector)
        scalc = np.eye(N) + bcalc @ bcalc.T
        tcalc = projector @ gstar
        rhs = bcalc @ tcalc
        zcalc = np.linalg.solve(scalc, rhs)
        duncalc = -tcalc + bcalc.T @ zcalc
        stored_b = np.asarray(diagnostic["B"], dtype=np.float64)
        stored_s = np.asarray(diagnostic["S"], dtype=np.float64)
        stored_t = np.asarray(diagnostic["t"], dtype=np.float64)
        stored_bt = np.asarray(diagnostic["B_t"], dtype=np.float64)
        stored_z = np.asarray(diagnostic["solve_vector"], dtype=np.float64)
        stored_du = np.asarray(diagnostic["uncorrected_direction"], dtype=np.float64)
        normal_norm = norm(normal)
        pivot = int(20+np.argmax(np.abs(normal[20:25])))
        retained = [i for i in range(M) if i != pivot]
        chart = np.zeros((M, M-1), dtype=np.float64)
        chart[retained, np.arange(M-1)] = 1.0
        for col, coord in enumerate(retained):
            if 20 <= coord < 25:
                chart[pivot, col] = -normal[coord]/normal[pivot]
        dchart = chart @ duncalc[retained]
        pminus = row_data["A_minus"] @ projector
        pplus = row_data["A_plus"] @ projector
        row_diff = pminus-pplus
        result["woodbury"] = {
            "B_relative_error": rel_error(stored_b, bcalc),
            "S_relative_error": rel_error(stored_s, scalc),
            "t_relative_error": rel_error(stored_t, tcalc),
            "B_t_relative_error": rel_error(stored_bt, rhs),
            "solve_vector_relative_error": rel_error(stored_z, zcalc),
            "uncorrected_direction_relative_error": rel_error(stored_du, duncalc),
            "direction_relative_error_after_chart": rel_error(direction, dchart),
            "B_recomputed": bcalc.tolist(), "S_recomputed": scalc.tolist(),
            "S_eigenvalue_minimum": float(np.linalg.eigvalsh(scalc).min()),
            "projected_row_parity_linf": float(np.max(np.abs(row_diff))),
            "projected_row_parity_l2": float(np.linalg.norm(row_diff)),
            "projected_row_error_saved": diagnostic.get("projected_row_errors"),
            "projected_row_budget_saved": diagnostic.get("projected_row_budgets"),
            "D_recomputed": robust_d.tolist(),
            "D_saved_max_abs_error": float(np.max(np.abs(
                robust_d-np.asarray(diagnostic["robust_curvature_D"], dtype=np.float64))))}

    if endpoint is not None:
        alpha = float(endpoint["alpha"])
        cand_control = np.asarray(endpoint["control"], dtype=np.float64)
        delta_control = cand_control-base_control
        cand_gm = np.asarray(endpoint["side_gradients"]["-1"], dtype=np.float64)
        cand_gp = np.asarray(endpoint["side_gradients"]["1"], dtype=np.float64)
        cand_min = theta_minimum(cand_gm, cand_gp)
        cand_theta = float(endpoint.get("candidate_theta_star", endpoint["theta"]))
        cand_q = float(endpoint.get("face_value", 0.0))
        cand_g = (1.0-cand_theta)*cand_gm + cand_theta*cand_gp
        actual_r = np.concatenate((cand_g, np.array([cand_q/face_scale])))
        predicted_r = residual+alpha*residual_direction
        pred_delta, actual_delta = predicted_r-residual, actual_r-residual
        err_delta = actual_delta-pred_delta
        pred_delta_norm, actual_delta_norm = norm(pred_delta), norm(actual_delta)
        cos_delta = float(pred_delta@actual_delta)/(pred_delta_norm*actual_delta_norm) \
            if pred_delta_norm and actual_delta_norm else math.nan
        cos_full = float(predicted_r@actual_r)/(norm(predicted_r)*norm(actual_r)) \
            if norm(predicted_r) and norm(actual_r) else math.nan
        pred_f2, base_f2, actual_f2 = float(predicted_r@predicted_r), float(residual@residual), float(actual_r@actual_r)
        pred_f2_drop, actual_f2_drop = base_f2-pred_f2, base_f2-actual_f2
        j_slope = max(float(gm @ direction), float(gp @ direction))
        j_bound = base_j + J_C1*alpha*j_slope
        f_bound = base_f2 + 2.0*F_C1*alpha*float(arm["merit_product"])
        static_q32 = static_qy(cand_control, 3, 2)
        static_q43 = static_qy(cand_control, 4, 3)
        result["accepted_endpoint"] = {
            "control_sha256": vsha(cand_control), "theta_star_saved": cand_theta,
            "theta_star_recomputed": cand_min["theta_star"],
            "theta_star_linear_prediction": endpoint.get("linear_theta_prediction"),
            "theta_star_linear_mismatch": (cand_theta-float(endpoint["linear_theta_prediction"])
                if isinstance(endpoint.get("linear_theta_prediction"), (int,float)) else None),
            "theta_boundary_distance": cand_min["boundary_distance"],
            "jump_norm": cand_min["jump_norm"], "jump_resolved": cand_min["jump_resolved"],
            "native_J": endpoint["objective"], "J_decrease_from_base": base_j-float(endpoint["objective"]),
            "J_decrease_percent": 100.0*(base_j-float(endpoint["objective"]))/base_j if base_j else None,
            "F_squared": float(endpoint["F_squared"]), "F_squared_recomputed": actual_f2,
            "F_squared_decrease_from_base": base_f2-actual_f2,
            "F_squared_decrease_percent": 100.0*actual_f2_drop/base_f2 if base_f2 else None,
            "predicted_F_squared": pred_f2,
            "predicted_F_squared_decrease_percent": 100.0*pred_f2_drop/base_f2 if base_f2 else None,
            "actual_to_predicted_F_squared_decrease_ratio": actual_f2_drop/pred_f2_drop
                if pred_f2_drop else None,
            "model_delta_norm": pred_delta_norm, "actual_delta_norm": actual_delta_norm,
            "model_to_actual_delta_norm_ratio": pred_delta_norm/actual_delta_norm if actual_delta_norm else None,
            "delta_vector_relative_error": norm(err_delta)/actual_delta_norm if actual_delta_norm else None,
            "delta_vector_angle_degrees": math.degrees(math.acos(max(-1,min(1,cos_delta))))
                if math.isfinite(cos_delta) else None,
            "full_residual_angle_degrees": math.degrees(math.acos(max(-1,min(1,cos_full))))
                if math.isfinite(cos_full) else None,
            "c12_c15_model_error": component_metrics(err_delta),
            "J_model_slope": j_slope, "J_Armijo_bound_recomputed": j_bound,
            "J_Armijo_passed_recomputed": float(endpoint["objective"]) <= j_bound,
            "F_squared_Armijo_bound_recomputed": f_bound,
            "F_squared_Armijo_passed_recomputed": actual_f2 <= f_bound,
            "movement_l2": norm(delta_control), "movement_linf": float(np.max(np.abs(delta_control))),
            "face_Q": cand_q, "static_Q32": static_q32, "static_Q43": static_q43,
            "minimum_gradient_l2": norm(cand_g), "minimum_gradient_linf": float(np.max(np.abs(cand_g))),
            "minimum_gradient_blocks": {"initial_field_20": norm(cand_g[:20]),
                "flow_5": norm(cand_g[20:25]), "growth_1": abs(float(cand_g[25]))},
            "theta_minimum_valid": endpoint.get("mixing_minimum_valid"),
            "face_passed": endpoint.get("face_audit_passed"),
            "branch_pair_passed": endpoint.get("branch_pair_passed"),
            "F_squared_Armijo_saved": endpoint.get("F_squared_armijo_passed"),
            "J_Armijo_saved": endpoint.get("J_armijo_passed")}
    return result


def select_winner(summaries: list[dict[str, Any]], dtype_epsilon: float = EPS) -> str | None:
    proposals = [summary for summary in summaries if summary.get("accepted_endpoint") is not None]
    if not proposals:
        return None
    winner = proposals[0]
    for item in proposals[1:]:
        a, b = item["accepted_endpoint"], winner["accepted_endpoint"]
        fa, fb = float(a["F_squared"]), float(b["F_squared"])
        fscale = max(abs(fa), abs(fb), np.finfo(np.float64).tiny)
        tie_f = abs(fa-fb) <= 128.0*dtype_epsilon*fscale
        if fa < fb and not tie_f:
            winner = item
        elif tie_f:
            ja, jb = float(a["native_J"]), float(b["native_J"])
            jscale = max(abs(ja), abs(jb), np.finfo(np.float64).tiny)
            tie_j = abs(ja-jb) <= 128.0*dtype_epsilon*jscale
            if (ja < jb and not tie_j) or (tie_j and item.get("name") == "baseline_tangent"):
                winner = item
    return str(winner["name"])


def collect(child_path: Path, run_path: Path, resource_path: Path, plan_path: Path) -> dict[str, Any]:
    root = EVIDENCE.parents[1]
    raw_bytes, child = read_child(child_path)
    parent, resource, plan = read_json(run_path), read_json(resource_path), read_json(plan_path)
    _, predecessor = load_predecessor(plan, root)
    control = np.asarray(predecessor["control"], dtype=np.float64)
    gradients = {side: np.asarray(predecessor["side_gradients"][side], dtype=np.float64)
        for side in ("-1", "1")}
    minimum = theta_minimum(gradients["-1"], gradients["1"])
    theta = float(minimum["theta_star"])
    _, normal, q = face_geometry(control, plan["face"])
    unit = normal / norm(normal)
    projector = np.eye(M) - np.outer(unit, unit)
    rows = build_rows(child, plan["base_control_sha256"], child["base_mixing_minimum"]["working_theta_star"])
    residual = rows["r_minus"]
    robust_d = (2.0 / np.hypot(2.0, residual))**3
    iteration = child["iterations"][0]
    arms = [arm_summary(arm, base_control=control,
        base_j=float(predecessor["objective"]), base_gradients=gradients,
        theta_star=theta, normal=normal, projector=projector, row_data=rows,
        robust_d=robust_d, diagnostics=arm.get("diagnostics", {}),
        residual_q=q, face_scale=FACE_SCALE) for arm in iteration["model_comparisons"]]
    winner = select_winner(arms)
    chosen = next(arm for arm in iteration["model_comparisons"] if arm["name"] == winner)
    proposal = next(trial for trial in chosen["trials"] if trial["accepted"] is True)
    repeat = iteration["final_repeat"]
    p2 = {}
    for side in ("-1", "1"):
        actual = np.asarray(repeat["side_gradients"][side])
        expected = np.asarray(proposal["side_gradients"][side])
        scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(expected))), np.finfo(float).tiny)
        error = float(np.max(np.abs(actual-expected)))
        p2[side] = {"max_error": error, "budget": EPS128*scale, "passed": error <= EPS128*scale}
    final_min = theta_minimum(repeat["side_gradients"]["-1"], repeat["side_gradients"]["1"])
    final_theta = float(final_min["theta_star"])
    final_gradient = ((1-final_theta)*np.asarray(repeat["side_gradients"]["-1"])
        + final_theta*np.asarray(repeat["side_gradients"]["1"]))
    _, _, final_q = face_geometry(repeat["control"], plan["face"])
    final_f2 = float(final_gradient@final_gradient + (final_q/FACE_SCALE)**2)
    source_expected = {**plan["source_files"], **plan["archive_files"],
        str(plan_path.resolve().relative_to(root)): sha_file(plan_path)}
    pin_errors = [name for name, digest in {**plan["source_files"], **plan["archive_files"]}.items()
        if sha_file(root/name) != digest]
    closure_names = ("side_gradients_finite", "native_objective_matches_proposal",
        "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
        "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
        "fixed_input_unchanged", "runtime_unchanged", "deadline_passed",
        "mixing_minimum_valid", "minimum_theta_matches_proposal")
    history = child["hvp_history"]
    hvp_checks = {}
    for arm in iteration["model_comparisons"]:
        entries = [entry for entry in history if entry["direction_model"] == arm["name"]]
        hvp_checks[arm["name"]] = len(entries) == 2 and {e["side"] for e in entries} == {-1, 1} and all(
            e["status"] == "completed" and e["base_control_sha256"] == plan["base_control_sha256"]
            and e["direction_sha256"] == vsha(arm["direction"])
            and e["theta"] == plan["initial_theta"] and e["working_theta"] == arm["working_theta"]
            for e in entries)
    checks = {"parent_completed": parent["execution_status"] == "completed",
        "raw_digest_matches_parent": sha_bytes(raw_bytes) == parent["child_sha256"],
        "resource_matches_parent": parent["resource"] == resource,
        "source_receipts_match": child["source_before"] == child["source_after"] == source_expected,
        "all_plan_pins_match": not pin_errors,
        "winner_matches_saved": winner == iteration["selected_direction_model"],
        "only_selected_final_control": repeat["control"] == proposal["control"] == child["current_control"],
        "final_control_digest": vsha(repeat["control"]) == child["current_control_sha256"],
        "final_min_theta_matches": abs(final_theta-repeat["theta"]) <= EPS128*max(abs(final_theta),abs(repeat["theta"])),
        "final_F_squared_matches": abs(final_f2-repeat["F_squared"]) <= EPS128*max(abs(final_f2),abs(repeat["F_squared"])),
        "per_side_gradient_matches": all(check["passed"] for check in p2.values()),
        "all_final_closure_flags": all(repeat.get(key) is True for key in closure_names),
        "HVP_provenance": all(hvp_checks.values()),
        "rows_complete": rows["count"] == 24 and rows["finite"] and all(rows["complete_by_side"].values()),
        "resource_closed": resource["exit_code"] == 0 and resource["resource_termination"] is None
            and resource["monitor_error"] is None and resource["elapsed_seconds"] < plan["policy"]["outer_seconds"]
            and resource["sampled_peak_rss_bytes"] < plan["policy"]["rss_bytes"]}
    if not all(checks.values()):
        raise ValueError(f"saved receipt verification failed: {[k for k,v in checks.items() if not v]}")
    return {"scope": "Saved arrays and linear algebra only; no production FV/gradient/HVP regeneration",
        "plan_sha256": sha_file(plan_path), "raw_sha256": sha_bytes(raw_bytes),
        "execution_status": parent["execution_status"], "numerical_status": parent["numerical_status"],
        "base": {"control_sha256": plan["base_control_sha256"], "native_J": predecessor["objective"],
            "theta_star": theta, "F_squared": child["base_mixing_minimum"]["F_squared"]},
        "arms": arms, "selected_direction": winner,
        "selected_endpoint": {"control_sha256": child["current_control_sha256"], "theta_star": final_theta,
            "native_J": repeat["objective"], "F_squared": final_f2, "gradient_norm": norm(final_gradient),
            "gradient_linf": float(np.max(np.abs(final_gradient))),
            "J_decrease_percent": 100*(1-repeat["objective"]/predecessor["objective"]),
            "F_squared_decrease_percent": 100*(1-final_f2/child["base_mixing_minimum"]["F_squared"])},
        "checks": checks, "P2": p2, "HVP_provenance_by_arm": hvp_checks,
        "final_closure_flags": {key: repeat.get(key) is True for key in closure_names},
        "accounting": {"guarded_launches": 1, "row_started": child["jacobian_rows_started"],
            "row_completed": child["jacobian_rows_completed"], "HVP_started": child["hvp_calls_started"],
            "HVP_completed": child["hvp_calls_completed"], "dense_started": child["dense_solves_started"],
            "dense_completed": child["dense_solves_completed"], "candidate_slots": sum(a["candidate_trials"] for a in arms),
            "optimizer_steps": child["optimizer_steps_applied"], "elapsed_seconds": resource["elapsed_seconds"],
            "sampled_peak_rss_bytes": resource["sampled_peak_rss_bytes"]},
        "claims": {"minimum": child["minimum_claim"], "smooth_root": child["full_smooth_root"],
            "response": child["response_claim"], "forecast_score": plan["policy"]["score_claim"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, default=ATTEMPT/"step.json")
    parser.add_argument("--run", type=Path, default=ATTEMPT/"step.run.json")
    parser.add_argument("--resource", type=Path, default=ATTEMPT/"step.resource.json")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=EVIDENCE/"GNCOUPLED_RESULT_20261010.json")
    args = parser.parse_args()
    result = collect(args.child, args.run, args.resource, args.plan)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=lambda value: value.item())+"\n")
    print(json.dumps({key:result[key] for key in ("selected_direction","selected_endpoint","checks","accounting")},indent=2, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
