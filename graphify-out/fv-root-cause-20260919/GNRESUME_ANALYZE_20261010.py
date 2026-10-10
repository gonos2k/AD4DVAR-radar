"""Audit a saved coupled-GN resume using receipts and saved arrays only.

This module reuses the one-point GN analyzer's NumPy helpers. It imports no FV,
seed, autodiff, production, or optimizer modules and never regenerates derivatives.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import GNCOUPLED_ANALYZE_20261010 as gn

EVIDENCE = Path(__file__).resolve().parent
PLAN = EVIDENCE / "TANGENT_COUPLED_GN_RESUME_PLAN_20261010.json"
ATTEMPT = EVIDENCE / "tangent_coupled_gn_resume_20261010_attempt1"
OUT = EVIDENCE / "GNRESUME_RESULT_20261010.json"
MODEL = "robust_gn_coupled"
CLOSURE_FLAGS = (
    "side_gradients_finite", "native_objective_matches_proposal", "side_objectives_match_native",
    "merit_matches_proposal", "face_audit_passed", "branch_pair_passed", "trace_matches_proposal",
    "source_unchanged", "fixed_input_unchanged", "runtime_unchanged", "deadline_passed",
    "mixing_minimum_valid", "minimum_theta_matches_proposal",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_child(path: Path) -> tuple[bytes, dict[str, Any]]:
    stored = path.read_bytes()
    raw = gzip.decompress(stored) if path.suffix == ".gz" else stored
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("resume child is not a JSON object")
    return raw, value


def near(a: float, b: float, multiplier: float = gn.EPS128) -> bool:
    return abs(a-b) <= multiplier * max(abs(a), abs(b), np.finfo(np.float64).tiny)


def load_pr273_base(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    """Close the selected one-step producer endpoint from its pinned raw archive."""
    producer_plan_path = root / plan["producing_plan"]
    producer_plan = read_json(producer_plan_path)
    if sha_file(producer_plan_path) != plan["producing_plan_sha256"]:
        raise ValueError("PR273 producer plan digest mismatch")
    archive = root / plan["base_archive"]
    run_path = root / plan["base_run"]
    resource_path = root / plan["base_resource"]
    manifest_path = root / plan["base_archive_manifest"]
    for path, key in ((archive, "base_archive_sha256"), (run_path, "base_run_sha256"),
                      (resource_path, "base_resource_sha256"),
                      (manifest_path, "base_archive_manifest_sha256")):
        if sha_file(path) != plan[key]:
            raise ValueError(f"PR273 receipt digest mismatch: {path}")
    raw_bytes, child = read_child(archive)
    run, resource, manifest = read_json(run_path), read_json(resource_path), read_json(manifest_path)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    if (raw_sha != plan["base_child_sha256"] or manifest.get("raw_sha256") != raw_sha
            or manifest.get("gzip_sha256") != sha_file(archive)
            or manifest.get("run_sha256") != sha_file(run_path)
            or manifest.get("resource_sha256") != sha_file(resource_path)
            or manifest.get("lossless_roundtrip") is not True
            or run.get("execution_status") != "completed"
            or run.get("child_sha256") != raw_sha or run.get("resource") != resource
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None):
        raise ValueError("PR273 producer run/archive/resource receipts do not close")
    expected_sources = {**producer_plan["source_files"], **producer_plan["archive_files"],
        str(producer_plan_path.resolve().relative_to(root.resolve())): plan["producing_plan_sha256"]}
    if (child.get("plan_sha256") != plan["producing_plan_sha256"]
            or child.get("source_before") != expected_sources or child.get("source_after") != expected_sources
            or child.get("phase") != "finished" or child.get("execution_status") != "completed"
            or child.get("numerical_status") != "tangent_coupled_gn_comparison_accepted"
            or child.get("accepted_iterations") != 1 or child.get("optimizer_steps_applied") != 1
            or child.get("current_control_sha256") != plan["base_control_sha256"]
            or child.get("current_theta") != plan["initial_theta"]):
        raise ValueError("PR273 selected endpoint identity/provenance is not closed")
    iterations = child.get("iterations", [])
    if len(iterations) != 1:
        raise ValueError("PR273 producer must contain exactly one comparison point")
    item = iterations[0]
    arms = [arm for arm in item.get("model_comparisons", []) if arm.get("name") == MODEL]
    accepted = [trial for arm in arms for trial in arm.get("trials", []) if trial.get("accepted") is True]
    endpoint = child.get("last_confirmed_closure")
    if (item.get("accepted") is not True or item.get("selected_direction_model") != MODEL
            or len(arms) != 1 or len(accepted) != 1 or not isinstance(endpoint, dict)
            or endpoint != item.get("final_repeat") or accepted[0].get("control") != endpoint.get("control")
            or gn.vsha(endpoint.get("control", [])) != plan["base_control_sha256"]
            or endpoint.get("theta") != plan["initial_theta"]
            or endpoint.get("objective") != plan["base_objective"]
            or endpoint.get("F_squared") != plan["base_carried_F_squared"]
            or not all(endpoint.get(flag) is True for flag in CLOSURE_FLAGS)):
        raise ValueError("PR273 selected robust-GN final repeat does not close")
    theta = gn.theta_minimum(endpoint["side_gradients"]["-1"], endpoint["side_gradients"]["1"])
    if not theta["strict_interior"] or not near(theta["theta_star"], float(endpoint["theta"])):
        raise ValueError("PR273 selected endpoint is not the saved strict-interior point minimum")
    control = np.asarray(endpoint["control"], dtype=np.float64)
    _, _, q = gn.face_geometry(control, plan["face"])
    gradient = ((1-theta["theta_star"]) * np.asarray(endpoint["side_gradients"]["-1"])
        + theta["theta_star"] * np.asarray(endpoint["side_gradients"]["1"]))
    f2 = float(gradient@gradient + (q/gn.FACE_SCALE)**2)
    if not near(f2, float(endpoint["F_squared"])):
        raise ValueError("PR273 selected endpoint F-squared does not recompute")
    return {"child": child, "endpoint": endpoint, "control": control,
        "control_sha256": gn.vsha(control), "theta": float(theta["theta_star"]),
        "objective": float(endpoint["objective"]), "F_squared": f2, "raw_sha256": raw_sha,
        "producer_receipt_hashes": {"plan_sha256": sha_file(producer_plan_path),
            "raw_sha256": raw_sha, "gzip_sha256": sha_file(archive), "run_sha256": sha_file(run_path),
            "resource_sha256": sha_file(resource_path), "archive_manifest_sha256": sha_file(manifest_path)}}


def _saved_source_checks(child: dict[str, Any], plan_path: Path, plan: dict[str, Any], root: Path) -> dict[str, Any]:
    expected = {**plan["source_files"], **plan["archive_files"],
        str(plan_path.resolve().relative_to(root.resolve())): sha_file(plan_path)}
    mismatches = [name for name, digest in {**plan["source_files"], **plan["archive_files"]}.items()
        if sha_file(root/name) != digest]
    after_present = isinstance(child.get("source_after"), dict)
    before_matches = child.get("source_before") == expected
    after_matches = child.get("source_after") == expected if after_present else None
    return {"source_before_matches": before_matches,
        "source_after_present": after_present, "source_after_matches": after_matches,
        "all_current_plan_pins_match": not mismatches, "pin_mismatches": mismatches}


def _rows_for_point(child: dict[str, Any], control_sha: str, theta: float) -> dict[str, Any]:
    history = child.get("jacobian_row_history", [])
    point_history = [row for row in history if row.get("base_control_sha256") == control_sha]
    filtered = {"jacobian_row_history": point_history}
    rows = gn.build_rows(filtered, control_sha, theta)
    rows["history"] = point_history
    return rows


def _endpoint_metrics(base_control: np.ndarray, base_j: float, arm: dict[str, Any],
        repeat: dict[str, Any], face: dict[str, Any], model_summary: dict[str, Any],
        radius: float) -> dict[str, Any]:
    control = np.asarray(repeat["control"], dtype=np.float64)
    gradients = {side: np.asarray(repeat["side_gradients"][side], dtype=np.float64)
        for side in ("-1", "1")}
    minimum = gn.theta_minimum(gradients["-1"], gradients["1"])
    theta = float(minimum["theta_star"])
    if not minimum["strict_interior"] or not near(theta, float(repeat["theta"])):
        raise ValueError("accepted repeat theta does not match recomputed strict point minimum")
    _, _, q = gn.face_geometry(control, face)
    gstar = (1-theta)*gradients["-1"] + theta*gradients["1"]
    f2 = float(gstar@gstar + (q/gn.FACE_SCALE)**2)
    if not near(f2, float(repeat["F_squared"])):
        raise ValueError("accepted repeat F-squared does not recompute")
    accepted = [trial for trial in arm.get("trials", []) if trial.get("accepted") is True]
    if len(accepted) != 1:
        raise ValueError("accepted GN arm must have exactly one accepted candidate")
    trial = accepted[0]
    p2: dict[str, Any] = {}
    for side in ("-1", "1"):
        actual = gradients[side]
        proposed = np.asarray(trial["side_gradients"][side], dtype=np.float64)
        scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(proposed))), np.finfo(float).tiny)
        error = float(np.max(np.abs(actual-proposed)))
        p2[side] = {"max_error": error, "budget": gn.EPS128*scale, "passed": error <= gn.EPS128*scale}
    first_alpha = float(arm["trials"][0]["alpha"]) if arm.get("trials") else None
    accepted_alpha = float(trial["alpha"])
    base_f2 = float(np.dot(np.asarray(arm["residual"], dtype=np.float64),
        np.asarray(arm["residual"], dtype=np.float64)))
    direction_norm = gn.norm(arm["direction"])
    derivative_norm = gn.norm(arm["residual_direction"])
    radius_cap = radius/direction_norm if direction_norm else math.inf
    merit_cap = (-float(arm["merit_product"])/(derivative_norm*derivative_norm)
        if derivative_norm else math.inf)
    initial_model_alpha = min(1.0, radius_cap, merit_cap)
    model_residual_base = np.asarray(arm["residual"], dtype=np.float64)
    model_chi_cos_squared = (float(arm["merit_product"])**2
        / (float(model_residual_base@model_residual_base) * derivative_norm**2)
        if derivative_norm and float(model_residual_base@model_residual_base) else None)
    trial_theta = float(trial.get("candidate_theta_star", trial["theta"]))
    trial_control = np.asarray(trial["control"], dtype=np.float64)
    _, _, trial_q = gn.face_geometry(trial_control, face)
    trial_g = ((1-trial_theta)*np.asarray(trial["side_gradients"]["-1"], dtype=np.float64)
        + trial_theta*np.asarray(trial["side_gradients"]["1"], dtype=np.float64))
    actual_residual = np.concatenate((trial_g, np.array([trial_q/gn.FACE_SCALE])))
    model_residual = (np.asarray(arm["residual"], dtype=np.float64)
        + accepted_alpha*np.asarray(arm["residual_direction"], dtype=np.float64))
    full_residual_error = model_residual-actual_residual
    model_metrics = model_summary.get("accepted_endpoint") or {}
    metrics = {"control_sha256": gn.vsha(control), "theta_star": theta,
        "theta_boundary_distance": minimum["boundary_distance"], "jump_norm": minimum["jump_norm"],
        "theta_stationarity_dot": minimum["stationarity_dot"], "native_J": float(repeat["objective"]),
        "F_squared": f2, "J_decrease_from_base": base_j-float(repeat["objective"]),
        "J_decrease_percent": 100*(base_j-float(repeat["objective"]))/base_j if base_j else None,
        "F_squared_decrease_from_base": base_f2-f2,
        "F_squared_decrease_percent": 100*(base_f2-f2)/base_f2 if base_f2 else None,
        "face_Q": q, "face_Q32": gn.static_qy(control, 3, 2), "face_Q43": gn.static_qy(control, 4, 3),
        "minimum_gradient_l2": gn.norm(gstar), "minimum_gradient_linf": float(np.max(np.abs(gstar))),
        "minimum_gradient_blocks": {"initial_field_20": gn.norm(gstar[:20]),
            "flow_5": gn.norm(gstar[20:25]), "growth_1": abs(float(gstar[25]))},
        "movement_l2": gn.norm(control-base_control), "movement_linf": float(np.max(np.abs(control-base_control))),
        "candidate_count": len(arm.get("trials", [])), "first_model_alpha": first_alpha,
        "initial_model_alpha_alpha0": initial_model_alpha,
        "initial_alpha_formula": "min(1, radius/||direction||, -merit_product/||DF||^2)",
        "initial_alpha_matches_first_candidate": near(initial_model_alpha, first_alpha)
            if first_alpha is not None else False,
        "model_chi_cos_squared": model_chi_cos_squared,
        "model_max_F_squared_reduction_percent": 100*model_chi_cos_squared
            if model_chi_cos_squared is not None else None,
        "accepted_alpha": accepted_alpha,
        "accepted_alpha_over_first_model_alpha": accepted_alpha/first_alpha if first_alpha else None,
        "accepted_alpha_over_initial_model_alpha": accepted_alpha/initial_model_alpha
            if initial_model_alpha else None,
        "candidate_halving_index": (round(math.log2(first_alpha/accepted_alpha))
            if first_alpha and accepted_alpha and first_alpha/accepted_alpha >= 1 else None),
        "predicted_vs_actual": {"model_chi_cos_squared": model_chi_cos_squared,
            "model_max_F_squared_reduction_percent": 100*model_chi_cos_squared
                if model_chi_cos_squared is not None else None,
            "predicted_F_squared": model_metrics.get("predicted_F_squared"),
            "predicted_F_squared_decrease_percent": model_metrics.get("predicted_F_squared_decrease_percent"),
            "actual_to_predicted_F_squared_decrease_ratio": model_metrics.get(
                "actual_to_predicted_F_squared_decrease_ratio"),
            "model_delta_norm": model_metrics.get("model_delta_norm"),
            "actual_delta_norm": model_metrics.get("actual_delta_norm"),
            "delta_vector_relative_error": model_metrics.get("delta_vector_relative_error"),
            "delta_vector_angle_degrees": model_metrics.get("delta_vector_angle_degrees"),
            "full_residual_angle_degrees": model_metrics.get("full_residual_angle_degrees"),
            "full_residual_error_l2": gn.norm(full_residual_error),
            "full_residual_relative_error_to_actual_endpoint": gn.norm(full_residual_error)/gn.norm(actual_residual)
                if gn.norm(actual_residual) else None,
            "c12_c15_model_error": model_metrics.get("c12_c15_model_error")}}
    # arm_summary is the authoritative shared reconstruction of theta-prime, DF, model residual,
    # predicted/actual merit, full-residual angle, delta error, and c12:c15 contribution.
    return metrics | {"P2": p2, "accepted_trial_control_matches_repeat": trial.get("control") == repeat.get("control"),
        "accepted_trial_J_matches_repeat": trial.get("objective") == repeat.get("objective"),
        "accepted_trial_theta_matches_repeat": trial.get("theta") == repeat.get("theta"),
        "closure_flags": {flag: repeat.get(flag) is True for flag in CLOSURE_FLAGS}}


def collect(child_path: Path, run_path: Path, resource_path: Path, plan_path: Path,
            archive_manifest_path: Path | None = None) -> dict[str, Any]:
    root = EVIDENCE.parents[1]
    raw_bytes, child = read_child(child_path)
    parent, resource, plan = gn.read_json(run_path), gn.read_json(resource_path), gn.read_json(plan_path)
    base = load_pr273_base(plan, root)
    row_history = child.get("jacobian_row_history", [])
    hvp_history = child.get("hvp_history", [])
    solve_history = child.get("coupled_gn_solve_history", [])
    point_history = child.get("coupled_gn_point_history", [])
    iterations = child.get("iterations", [])
    current_control = base["control"]
    current_sha, current_theta, current_j = base["control_sha256"], base["theta"], base["objective"]
    current_gradients = {side: base["endpoint"]["side_gradients"][side] for side in ("-1", "1")}
    points: list[dict[str, Any]] = []
    committed_prefix = 0
    prefix_open = True
    total_candidate_slots = 0
    for index, item in enumerate(iterations):
        point: dict[str, Any] = {"index": index, "base_control_sha256": item.get("base_control_sha256"),
            "carried_theta": item.get("theta"), "accepted": item.get("accepted") is True}
        if item.get("index") != index or item.get("base_control_sha256") != current_sha or item.get("theta") != current_theta:
            point.update(status="chain_mismatch", checks_passed=False)
            points.append(point)
            prefix_open = False
            break
        arms = item.get("model_comparisons", [])
        if len(arms) != 1 or arms[0].get("name") != MODEL:
            point.update(status="missing_or_non_GN_arm", checks_passed=False)
            points.append(point)
            prefix_open = False
            break
        arm = arms[0]
        total_candidate_slots += len(arm.get("trials", []))
        working_theta = float(arm.get("working_theta", math.nan))
        working_min = gn.theta_minimum(current_gradients["-1"], current_gradients["1"])
        _, _, current_q = gn.face_geometry(current_control, plan["face"])
        current_mixed = ((1-working_min["theta_star"])*np.asarray(current_gradients["-1"], dtype=np.float64)
            + working_min["theta_star"]*np.asarray(current_gradients["1"], dtype=np.float64))
        current_f2 = float(current_mixed@current_mixed + (current_q/gn.FACE_SCALE)**2)
        arm_rows = _rows_for_point(child, current_sha, working_theta)
        arm_hvps = [entry for entry in hvp_history if entry.get("base_control_sha256") == current_sha]
        arm_solves = [entry for entry in solve_history if entry.get("base_control_sha256") == current_sha]
        row_index_pairs = {(int(row.get("side", 0)), int(row.get("row", -1)))
            for row in arm_rows["history"] if row.get("status") == "completed"}
        rows_ok = (arm_rows["count"] == 24 and arm_rows["finite"]
            and all(arm_rows["complete_by_side"].values())
            and row_index_pairs == {(side, row) for side in (-1, 1) for row in range(12)}
            and all(row.get("theta") == working_theta for row in arm_rows["history"]))
        direction_sha = gn.vsha(arm.get("direction", []))
        hvps_ok = (len(arm_hvps) == 2 and {entry.get("side") for entry in arm_hvps} == {-1, 1}
            and all(entry.get("status") == "completed" and entry.get("direction_model") == MODEL
                and entry.get("direction_sha256") == direction_sha and entry.get("theta") == current_theta
                and entry.get("working_theta") == working_theta
                and entry.get("scope") == "one current chart tangent"
                for entry in arm_hvps))
        solve_ok = (len(arm_solves) == 1 and arm.get("diagnostics", {}).get("dense_solves") == 1
            and arm.get("diagnostics", {}).get("row_vjps_completed") == 24*(index+1))
        point_receipt = point_history[index] if index < len(point_history) else None
        point_history_ok = (isinstance(point_receipt, dict)
            and point_receipt.get("index") == index
            and point_receipt.get("base_control_sha256") == current_sha
            and point_receipt.get("reference_theta") == current_theta
            and point_receipt.get("working_theta_star") == working_theta
            and near(float(point_receipt.get("F_squared", math.nan)), current_f2)
            and near(float(point_receipt.get("Psi", math.nan)), 0.5*current_f2)
            and point_receipt.get("optimizer_step") is False)
        point.update({"working_theta_star": working_theta,
            "working_theta_recomputed": working_min["theta_star"],
            "working_theta_is_current_minimum": near(working_theta, working_min["theta_star"]),
            "base_minimum_F_squared_recomputed": current_f2,
            "base_minimum_F_squared_history_matches": point_history_ok,
            "row_count": arm_rows["count"], "rows_complete": rows_ok,
            "hvp_count": len(arm_hvps), "hvp_provenance": hvps_ok,
            "dense_solve_count": len(arm_solves), "dense_solve_receipt": solve_ok,
            "row_vjps_completed_cumulative": arm.get("diagnostics", {}).get("row_vjps_completed"),
            "point_history_matches": point_history_ok,
            "candidate_count": len(arm.get("trials", [])),
            "candidate_count_within_cap": len(arm.get("trials", [])) <= int(plan["policy"].get(
                "candidate_grid_per_direction", plan["policy"].get("max_candidates_per_iteration", 0)))})
        if rows_ok and hvps_ok and solve_ok:
            _, normal, q = gn.face_geometry(current_control, plan["face"])
            unit = normal/gn.norm(normal)
            projector = np.eye(gn.M)-np.outer(unit, unit)
            robust_d = (2.0/np.hypot(2.0, arm_rows["r_minus"]))**3
            summary = gn.arm_summary(arm, base_control=current_control, base_j=current_j,
                base_gradients={side: np.asarray(current_gradients[side], dtype=np.float64)
                    for side in ("-1", "1")}, theta_star=working_theta, normal=normal,
                projector=projector, row_data=arm_rows, robust_d=robust_d,
                diagnostics=arm.get("diagnostics", {}), residual_q=q, face_scale=gn.FACE_SCALE)
            projected_error = arm_rows["A_minus"]@projector-arm_rows["A_plus"]@projector
            projected_row_linf = np.max(np.abs(projected_error), axis=1)
            projected_budget = np.asarray(arm.get("diagnostics", {}).get("projected_row_budgets", []),
                dtype=np.float64)
            residual_error = np.abs(arm_rows["r_minus"]-arm_rows["r_plus"])
            residual_budget = float(arm.get("diagnostics", {}).get("residual_pair_max_budget", math.nan))
            w = summary.get("woodbury", {})
            rel_keys = ("B_relative_error", "S_relative_error", "t_relative_error",
                "B_t_relative_error", "solve_vector_relative_error",
                "uncorrected_direction_relative_error", "direction_relative_error_after_chart")
            woodbury_ok = (all(w.get(key, {}).get("shape_match") is True
                and w[key].get("relative_l2_error", math.inf) <= 4096*np.finfo(float).eps
                for key in rel_keys)
                and w.get("S_eigenvalue_minimum", -math.inf) > 1.0
                and w.get("D_saved_max_abs_error", math.inf) <= 512*np.finfo(float).eps
                    * max(1.0, float(np.max(np.abs(robust_d)))))
            parity_ok = (projected_budget.shape == projected_row_linf.shape
                and bool(np.isfinite(projected_budget).all())
                and bool((projected_row_linf <= projected_budget).all())
                and math.isfinite(residual_budget)
                and float(np.max(residual_error)) <= residual_budget)
            summary["saved_array_consistency"] = {"woodbury_solve_matches": woodbury_ok,
                "projected_row_parity": {"max_error": float(np.max(projected_row_linf)),
                    "within_saved_budget": bool(projected_budget.shape == projected_row_linf.shape
                        and (projected_row_linf <= projected_budget).all())},
                "residual_value_parity": {"max_error": float(np.max(residual_error)),
                    "budget": residual_budget,
                    "within_saved_budget": float(np.max(residual_error)) <= residual_budget},
                "parity_passed": parity_ok}
            model_endpoint = summary.get("accepted_endpoint") or {}
            predicted_keys = ("predicted_F_squared", "predicted_F_squared_decrease_percent",
                "actual_to_predicted_F_squared_decrease_ratio", "model_delta_norm", "actual_delta_norm",
                "delta_vector_relative_error", "delta_vector_angle_degrees", "full_residual_angle_degrees",
                "c12_c15_model_error")
            model_vs_actual = {key: model_endpoint.get(key) for key in predicted_keys}
            model_vs_actual.update({"first_model_alpha": (arm.get("trials") or [{}])[0].get("alpha"),
                "accepted_alpha": next((trial.get("alpha") for trial in arm.get("trials", [])
                    if trial.get("accepted") is True), None)})
            first_alpha = model_vs_actual["first_model_alpha"]
            model_vs_actual["accepted_alpha_over_first_model_alpha"] = (
                model_vs_actual["accepted_alpha"]/first_alpha if first_alpha else None)
            summary["predicted_vs_actual"] = model_vs_actual
            point["local_model"] = summary
        accepted = item.get("accepted") is True
        repeat = item.get("final_repeat")
        endpoint_checks: dict[str, Any] = {}
        if accepted and isinstance(repeat, dict):
            endpoint_checks = _endpoint_metrics(current_control, current_j, arm, repeat,
                plan["face"], point["local_model"], float(plan["policy"]["radius"]))
            endpoint_checks["final_repeat_equals_committed_control"] = repeat.get("control") == item.get("committed_control")
            endpoint_checks["committed_theta_matches_repeat"] = item.get("committed_theta") == repeat.get("theta")
            endpoint_checks["committed_control_sha256"] = gn.vsha(item.get("committed_control", []))
            endpoint_ok = (endpoint_checks["final_repeat_equals_committed_control"]
                and endpoint_checks["committed_theta_matches_repeat"]
                and endpoint_checks["accepted_trial_control_matches_repeat"]
                and endpoint_checks["accepted_trial_J_matches_repeat"]
                and endpoint_checks["accepted_trial_theta_matches_repeat"]
                and all(val["passed"] for val in endpoint_checks["P2"].values())
                and all(endpoint_checks["closure_flags"].values())
                and endpoint_checks["committed_control_sha256"] == gn.vsha(repeat["control"]))
            endpoint_checks["accepted_endpoint_closed"] = endpoint_ok
            point["endpoint"] = endpoint_checks
            if endpoint_ok and prefix_open:
                committed_prefix += 1
                current_control = np.asarray(repeat["control"], dtype=np.float64)
                current_sha = gn.vsha(current_control)
                current_theta = float(repeat["theta"])
                current_j = float(repeat["objective"])
                current_gradients = {side: repeat["side_gradients"][side] for side in ("-1", "1")}
            else:
                prefix_open = False
        else:
            endpoint_checks["final_repeat_present"] = isinstance(repeat, dict)
            point["endpoint"] = endpoint_checks
            prefix_open = False
        point["status"] = "accepted_and_closed" if accepted and endpoint_checks.get("accepted_endpoint_closed") else (
            "accepted_but_not_closed" if accepted else "modeled_unaccepted_or_incomplete")
        point["checks_passed"] = bool(rows_ok and hvps_ok and solve_ok and point_history_ok
            and point["working_theta_is_current_minimum"] and point["candidate_count_within_cap"]
            and ("saved_array_consistency" not in point.get("local_model", {})
                or (point["local_model"]["saved_array_consistency"]["woodbury_solve_matches"]
                    and point["local_model"]["saved_array_consistency"]["parity_passed"]))
            and (not accepted or endpoint_checks.get("accepted_endpoint_closed", False)))
        points.append(point)
        if not point["checks_passed"]:
            prefix_open = False
            if accepted:
                break
        if not accepted:
            break
    # Audit unmodeled partial tails through receipts and counters, without treating them as commits.
    modeled_hashes = {point.get("base_control_sha256") for point in points}
    tail_rows = [row for row in row_history if row.get("base_control_sha256") not in modeled_hashes]
    tail_hvps = [entry for entry in hvp_history if entry.get("base_control_sha256") not in modeled_hashes]
    tail_solves = [entry for entry in solve_history if entry.get("base_control_sha256") not in modeled_hashes]
    resource_closed = (resource.get("exit_code") == 0 and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None and resource.get("received_sigterm") is False
        and float(resource.get("elapsed_seconds", math.inf)) < float(plan["policy"]["outer_seconds"])
        and int(resource.get("sampled_peak_rss_bytes", plan["policy"]["rss_bytes"]))
            < int(plan["policy"]["rss_bytes"]))
    archive_info = {"raw_sha256": hashlib.sha256(raw_bytes).hexdigest(), "raw_bytes": len(raw_bytes),
        "child_path": str(child_path), "run_sha256": sha_file(run_path),
        "resource_sha256": sha_file(resource_path), "plan_sha256": sha_file(plan_path)}
    gzip_path = child_path if child_path.suffix == ".gz" else child_path.with_suffix(child_path.suffix+".gz")
    if gzip_path.exists():
        archive_info.update(gzip_sha256=sha_file(gzip_path), gzip_roundtrip=(gzip.decompress(gzip_path.read_bytes()) == raw_bytes))
    if archive_manifest_path and archive_manifest_path.exists():
        archive_info["archive_manifest_sha256"] = sha_file(archive_manifest_path)
        manifest = read_json(archive_manifest_path)
        archive_info["archive_manifest"] = manifest
        archive_info["manifest_receipts_match"] = (
            manifest.get("raw_sha256") == archive_info["raw_sha256"]
            and manifest.get("raw_bytes") == archive_info["raw_bytes"]
            and manifest.get("run_sha256") == archive_info["run_sha256"]
            and manifest.get("resource_sha256") == archive_info["resource_sha256"]
            and ("gzip_sha256" not in manifest or manifest.get("gzip_sha256") == archive_info.get("gzip_sha256")))
    source_checks = _saved_source_checks(child, plan_path, plan, root)
    global_identity_keys = ("input_before", "input_after", "runtime_before", "runtime_after")
    global_identity = {"status": "not_recorded" if not any(key in child for key in global_identity_keys)
            else "partial_or_recorded",
        "present": {key: key in child for key in global_identity_keys},
        "comparison_passed": None if not all(key in child for key in global_identity_keys) else
            child.get("input_before") == child.get("input_after")
            and child.get("runtime_before") == child.get("runtime_after")}
    parent_digest_ok = parent.get("child_sha256") == hashlib.sha256(raw_bytes).hexdigest()
    parent_resource_ok = parent.get("resource") == resource
    prefix_receipt_ok = (child.get("last_confirmed_control_sha256") == current_sha
        and child.get("last_confirmed_theta") == current_theta
        and child.get("last_confirmed_iterations") == committed_prefix
        and child.get("optimizer_steps_applied") == committed_prefix
        and child.get("accepted_iterations") == committed_prefix)
    totals = {"modeled_points": len(points), "accepted_confirmed_prefix": committed_prefix,
        "configured_step_cap": int(plan["policy"]["max_accepted_iterations"]),
        "row_started": int(child.get("jacobian_rows_started", 0)),
        "row_completed": int(child.get("jacobian_rows_completed", 0)),
        "row_receipts": len(row_history), "row_budget": int(plan["policy"]["jacobian_row_vjp_calls"]),
        "HVP_started": int(child.get("hvp_calls_started", 0)),
        "HVP_completed": int(child.get("hvp_calls_completed", 0)),
        "HVP_receipts": len(hvp_history), "HVP_budget": int(plan["policy"]["hvp_calls"]),
        "dense_started": int(child.get("dense_solves_started", 0)),
        "dense_completed": int(child.get("dense_solves_completed", 0)),
        "dense_receipts": len(solve_history), "dense_budget": int(plan["policy"]["dense_solves"]),
        "candidate_slots_total": total_candidate_slots,
        "unmodeled_tail_rows": len(tail_rows), "unmodeled_tail_hvps": len(tail_hvps),
        "unmodeled_tail_solves": len(tail_solves),
        "unmodeled_tail_row_status_counts": {status: sum(row.get("status") == status for row in tail_rows)
            for status in sorted({str(row.get("status")) for row in tail_rows})},
        "optimizer_steps_saved": child.get("optimizer_steps_applied"),
        "elapsed_seconds": resource.get("elapsed_seconds"),
        "sampled_peak_rss_bytes": resource.get("sampled_peak_rss_bytes"),
        "resource_closed": resource_closed}
    receipt_accounting = {
        "row_start_complete_receipt_counts_match": totals["row_started"] == totals["row_completed"] == totals["row_receipts"],
        "HVP_start_complete_receipt_counts_match": totals["HVP_started"] == totals["HVP_completed"] == totals["HVP_receipts"],
        "dense_start_complete_receipt_counts_match": totals["dense_started"] == totals["dense_completed"] == totals["dense_receipts"],
        "three_step_totals_when_cap_reached": (totals["row_completed"] == 72
            and totals["HVP_completed"] == 6 and totals["dense_completed"] == 3)
            if committed_prefix == int(plan["policy"]["max_accepted_iterations"]) else None,
        "per_point_checks_passed": all(point.get("checks_passed") is True for point in points),
        "confirmed_prefix_matches_child_checkpoint": prefix_receipt_ok,
    }
    return {"scope": "Saved arrays and receipts only; no production FV, gradient, HVP, or optimizer regeneration",
        "execution_status": parent.get("execution_status"), "numerical_status": parent.get("numerical_status"),
        "plan_sha256": sha_file(plan_path), "base": {"control_sha256": base["control_sha256"],
            "theta_star": base["theta"], "native_J": base["objective"], "F_squared": base["F_squared"],
            "producer_raw_sha256": base["raw_sha256"],
            "producer_receipt_hashes": base["producer_receipt_hashes"]},
        "points": points, "confirmed_prefix": {"accepted_steps": committed_prefix,
            "terminal_control_sha256": current_sha, "terminal_theta": current_theta,
            "terminal_native_J": current_j},
        "receipt_hashes": archive_info, "source_checks": source_checks,
        "global_input_runtime_identity": global_identity,
        "accounting": totals, "receipt_accounting_checks": receipt_accounting,
        "terminal_closed": (child.get("phase") == "finished"
            and child.get("execution_status") == "completed"
            and parent.get("execution_status") == "completed" and resource_closed),
        "parent_checks": {"raw_digest_matches_parent": parent_digest_ok,
            "resource_matches_parent": parent_resource_ok,
            "confirmed_prefix_matches_child_checkpoint": prefix_receipt_ok},
        "claims": {"minimum": child.get("minimum_claim"), "smooth_root": child.get("full_smooth_root"),
            "response": child.get("response_claim"), "forecast_score": plan["policy"].get("score_claim")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, default=ATTEMPT/"step.json")
    parser.add_argument("--run", type=Path, default=ATTEMPT/"step.run.json")
    parser.add_argument("--resource", type=Path, default=ATTEMPT/"step.resource.json")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--archive-manifest", type=Path, default=EVIDENCE/"GNRESUME_ARCHIVE_20261010.json")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = collect(args.child, args.run, args.resource, args.plan, args.archive_manifest)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=lambda value: value.item())+"\n")
    print(json.dumps({key: result[key] for key in (
        "execution_status", "numerical_status", "confirmed_prefix", "accounting", "source_checks", "parent_checks")},
        indent=2, sort_keys=True, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
