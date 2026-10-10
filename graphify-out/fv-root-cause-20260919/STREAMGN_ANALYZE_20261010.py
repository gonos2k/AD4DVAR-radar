"""Audit the streamed partial-GN continuation from saved receipts and arrays only.

This adapter reuses the NumPy-only GNRESUME analyzer and replaces its PR273 base
loader with the single closed endpoint from the interrupted PR274 archive. It
does not import production/FV/seed/autodiff/optimizer modules or rerun products.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import GNRESUME_ANALYZE_20261010 as resume

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]
PLAN = EVIDENCE / "TANGENT_COUPLED_GN_PARTIAL_RESUME_PLAN_20261010.json"
OLD_PLAN = EVIDENCE / "TANGENT_COUPLED_GN_RESUME_PLAN_20261010.json"
OLD_ATTEMPT = EVIDENCE / "tangent_coupled_gn_resume_20261010_attempt1"
OLD_GZIP = OLD_ATTEMPT / "step.json.gz"
OLD_RUN = OLD_ATTEMPT / "step.run.json"
OLD_RESOURCE = OLD_ATTEMPT / "step.resource.json"
OLD_MANIFEST = EVIDENCE / "GNRESUME_ARCHIVE_20261010.json"
ATTEMPT = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt2"
OUT = EVIDENCE / "STREAMGN_RESULT_20261010.json"
FAILED_PREFLIGHT = EVIDENCE / "STREAMGN_PREFLIGHT_ARCHIVE_20261010.json"
FAILED_PREFLIGHT_ATTEMPT = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt1"
MODEL = resume.MODEL
FLAGS = resume.CLOSURE_FLAGS
EXPECTED_BASE_SHA = "7cec4c62bc2ebe3a96106086408adc0ecdbc959fe1b53beafa46902cd41fb7dd"
EXPECTED_THETA = 0.4829895933920232
EXPECTED_PLAN_SHA = "529110d25ca28ac85ab3ce3fa2fd23a8b2a8ba0122c2f184ab5e26a3c20b4f17"
EXPECTED_ANCHOR_PROVENANCE = {
    "input": "derived_from_pr273_input_after_control_sha_and_saved_control_flow_diagnostics_per_fixed_input_contract",
    "runtime": "reused_from_pr273_runtime_after",
}


def _load_partial_prefix(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    """Close the saved P2 endpoint, then expose only its confirmed state."""
    old_plan_sha = resume.sha_file(OLD_PLAN)
    if (old_plan_sha != plan.get("predecessor_plan_sha256")
            or resume.sha_file(OLD_GZIP) != plan.get("predecessor_gzip_sha256")
            or resume.sha_file(OLD_RUN) != plan.get("predecessor_run_sha256")
            or resume.sha_file(OLD_RESOURCE) != plan.get("predecessor_resource_sha256")
            or resume.sha_file(OLD_MANIFEST) != plan.get("predecessor_manifest_sha256")):
        raise ValueError("interrupted PR274 predecessor receipt digest mismatch")

    old_plan = resume.read_json(OLD_PLAN)
    raw_bytes, child = resume.read_child(OLD_GZIP)
    parent = resume.read_json(OLD_RUN)
    resource = resume.read_json(OLD_RESOURCE)
    manifest = resume.read_json(OLD_MANIFEST)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    source_before = {**old_plan["source_files"], **old_plan["archive_files"],
        str(OLD_PLAN.resolve().relative_to(root.resolve())): old_plan_sha}
    if (raw_sha != plan.get("predecessor_raw_sha256")
            or manifest.get("raw_sha256") != raw_sha
            or manifest.get("gzip_sha256") != resume.sha_file(OLD_GZIP)
            or manifest.get("run_sha256") != resume.sha_file(OLD_RUN)
            or manifest.get("resource_sha256") != resume.sha_file(OLD_RESOURCE)
            or manifest.get("lossless_roundtrip") is not True
            or manifest.get("terminal_execution_closed") is not False
            or parent.get("execution_status") != "rss_limit"
            or parent.get("child_sha256") != raw_sha or parent.get("resource") != resource
            or resource.get("resource_termination") != "rss_limit"
            or resource.get("exit_code") != -15 or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("sampled_peak_rss_bytes", 0) < int(plan["policy"]["rss_bytes"])
            or child.get("plan_sha256") != old_plan_sha
            or child.get("plan_hashes", {}).get("source_files") != old_plan["source_files"]
            or child.get("plan_hashes", {}).get("archive_files") != old_plan["archive_files"]
            or child.get("source_before") != source_before or child.get("source_after") is not None
            or any(child.get(key) is not None for key in
                ("input_before", "input_after", "runtime_before", "runtime_after"))
            or child.get("phase") != "running" or child.get("execution_status") != "running"
            or child.get("accepted_iterations") != 1 or child.get("optimizer_steps_applied") != 1
            or child.get("jacobian_rows_started") != 26
            or child.get("jacobian_rows_completed") != 25
            or child.get("hvp_calls_started") != 2 or child.get("hvp_calls_completed") != 2
            or child.get("dense_solves_started") != 1 or child.get("dense_solves_completed") != 1
            or child.get("current_control_sha256") != EXPECTED_BASE_SHA
            or child.get("last_confirmed_control_sha256") != EXPECTED_BASE_SHA
            or child.get("current_theta") != EXPECTED_THETA
            or child.get("last_confirmed_theta") != EXPECTED_THETA):
        raise ValueError("interrupted PR274 archive does not match its saved one-step prefix")

    iterations = child.get("iterations", [])
    endpoint = child.get("last_confirmed_closure")
    if len(iterations) != 1 or not isinstance(endpoint, dict):
        raise ValueError("PR274 saved prefix must contain exactly one confirmed closure")
    step = iterations[0]
    arms = [arm for arm in step.get("model_comparisons", []) if arm.get("name") == MODEL]
    accepted = [trial for arm in arms for trial in arm.get("trials", [])
        if trial.get("accepted") is True]
    control = resume.np.asarray(endpoint.get("control", []), dtype=resume.np.float64)
    if (step.get("accepted") is not True or step.get("selected_direction_model") != MODEL
            or step.get("final_repeat") != endpoint
            or len(arms) != 1 or arms[0].get("accepted") is not True or len(accepted) != 1
            or endpoint.get("control") != step.get("committed_control")
            or endpoint.get("theta") != EXPECTED_THETA
            or endpoint.get("objective") != plan.get("base_objective")
            or endpoint.get("F_squared") != plan.get("base_carried_F_squared")
            or resume.gn.vsha(control) != EXPECTED_BASE_SHA
            or not all(endpoint.get(flag) is True for flag in FLAGS)):
        raise ValueError("PR274 saved endpoint does not close through its independent P2 repeat")

    first_sha = step.get("base_control_sha256")
    rows = child.get("jacobian_row_history", [])
    first_rows = [row for row in rows if row.get("base_control_sha256") == first_sha]
    tail_rows = [row for row in rows if row.get("base_control_sha256") == EXPECTED_BASE_SHA]
    completed_pairs = {(row.get("side"), row.get("row")) for row in first_rows
        if row.get("status") == "completed"}
    hvps = child.get("hvp_history", [])
    tail_hvps = [row for row in hvps if row.get("base_control_sha256") == EXPECTED_BASE_SHA]
    solves = child.get("coupled_gn_solve_history", [])
    tail_solves = [row for row in solves if row.get("base_control_sha256") == EXPECTED_BASE_SHA]
    if (len(first_rows) != 24 or completed_pairs != {(side, row) for side in (-1, 1) for row in range(12)}
            or any(row.get("status") != "completed" for row in first_rows)
            or len(tail_rows) != 2
            or sorted((row.get("side"), row.get("row"), row.get("status")) for row in tail_rows)
                != [(-1, 0, "completed"), (-1, 1, "started")]
            or any(row.get("theta") != EXPECTED_THETA for row in tail_rows)
            or len(hvps) != 2 or any(row.get("status") != "completed" for row in hvps)
            or tail_hvps or len(solves) != 1 or tail_solves
            or len(child.get("coupled_gn_point_history", [])) != 1
            or len(child.get("coupled_gn_solve_history", [])) != 1
            or child.get("comparison_progress", {}).get("base_control_sha256") != first_sha):
        raise ValueError("PR274 saved row/HVP/solve boundary does not close its confirmed prefix")

    trial = accepted[0]
    minimum = resume.gn.theta_minimum(endpoint["side_gradients"]["-1"],
        endpoint["side_gradients"]["1"])
    _, _, face_q = resume.gn.face_geometry(control, plan["face"])
    mixed = ((1.0-minimum["theta_star"])*resume.np.asarray(endpoint["side_gradients"]["-1"])
        + minimum["theta_star"]*resume.np.asarray(endpoint["side_gradients"]["1"]))
    f_squared = float(mixed@mixed + (face_q/resume.gn.FACE_SCALE)**2)
    if (not minimum["strict_interior"]
            or not resume.near(minimum["theta_star"], EXPECTED_THETA)
            or not resume.near(f_squared, float(plan["base_carried_F_squared"]))
            or trial.get("control") != endpoint["control"]
            or trial.get("theta") != endpoint["theta"]
            or trial.get("objective") != endpoint["objective"]
            or trial.get("F_squared") != endpoint["F_squared"]):
        raise ValueError("PR274 confirmed control, minimum theta, or merit failed saved-array closure")
    for side in ("-1", "1"):
        actual = resume.np.asarray(endpoint["side_gradients"][side], dtype=resume.np.float64)
        proposed = resume.np.asarray(trial["side_gradients"][side], dtype=resume.np.float64)
        budget = resume.gn.EPS128 * max(float(resume.np.max(resume.np.abs(actual))),
            float(resume.np.max(resume.np.abs(proposed))), resume.np.finfo(float).tiny)
        if float(resume.np.max(resume.np.abs(actual-proposed))) > budget:
            raise ValueError(f"PR274 saved side {side} P2 gradient failed")

    # Input/runtime are derived expectations from completed PR273 receipts,
    # not retroactively asserted closure fields on the interrupted PR274 child.
    old_base = resume.load_pr273_base(old_plan, root)
    predecessor_input = old_base["child"]["input_after"]
    expected_input = dict(predecessor_input)
    expected_input["control_sha256"] = EXPECTED_BASE_SHA
    expected_flow = resume.np.tanh(control[20:25])
    expected_input["shifted_flow_fractions"] = expected_flow.tolist()
    expected_runtime = old_base["child"]["runtime_after"]
    provenance = plan.get("anchor_provenance")
    changed_fields = {key for key in set(predecessor_input) | set(expected_input)
        if predecessor_input.get(key) != expected_input.get(key)}
    if (provenance != EXPECTED_ANCHOR_PROVENANCE
            or changed_fields != {"control_sha256", "shifted_flow_fractions"}
            or plan.get("base_control_sha256") != EXPECTED_BASE_SHA
            or plan.get("initial_theta") != EXPECTED_THETA):
        raise ValueError("partial-resume plan lacks the expected derived-anchor provenance")
    return {"child": child, "endpoint": endpoint, "control": control,
        "control_sha256": EXPECTED_BASE_SHA, "theta": EXPECTED_THETA,
        "objective": float(endpoint["objective"]), "F_squared": f_squared,
        "raw_sha256": raw_sha, "producer_receipt_hashes": {
            "plan_sha256": old_plan_sha, "raw_sha256": raw_sha,
            "gzip_sha256": resume.sha_file(OLD_GZIP), "run_sha256": resume.sha_file(OLD_RUN),
            "resource_sha256": resume.sha_file(OLD_RESOURCE),
            "archive_manifest_sha256": resume.sha_file(OLD_MANIFEST)},
        "anchor": {"input": expected_input, "runtime": expected_runtime,
            "provenance": provenance}, "partial_raw": child,
        "partial_archive_manifest": manifest, "partial_resource": resource,
        "changed_anchor_input_fields": sorted(changed_fields),
        "expected_flow_from_saved_control": expected_flow,
        "tail_counts": {"rows_started": len(tail_rows),
            "rows_completed": sum(row.get("status") == "completed" for row in tail_rows),
            "HVPs": len(tail_hvps), "dense_solves": len(tail_solves),
            "candidates": 0, "commits": 0},
        "predecessor_progress_was_duplicate": child.get("comparison_progress", {}).get(
            "model_comparisons") == step.get("model_comparisons")}


def _brief_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Project the saved analyzer result into the per-point review quantities."""
    brief: list[dict[str, Any]] = []
    for point in result.get("points", []):
        endpoint = point.get("endpoint", {})
        model = point.get("local_model", {})
        actual = endpoint.get("predicted_vs_actual", {})
        brief.append({"index": point.get("index"),
            "base_control_sha256": point.get("base_control_sha256"),
            "rows": point.get("row_count"), "rows_complete": point.get("rows_complete"),
            "HVPs": point.get("hvp_count"), "HVP_provenance": point.get("hvp_provenance"),
            "dense_solves": point.get("dense_solve_count"),
            "dense_solve_receipt": point.get("dense_solve_receipt"),
            "working_theta_star": point.get("working_theta_star"),
            "endpoint_theta_star": endpoint.get("theta_star"),
            "native_J": endpoint.get("native_J"), "F_squared": endpoint.get("F_squared"),
            "minimum_gradient_l2": endpoint.get("minimum_gradient_l2"),
            "minimum_gradient_linf": endpoint.get("minimum_gradient_linf"),
            "minimum_gradient_blocks": endpoint.get("minimum_gradient_blocks"),
            "model_chi_cos_squared": actual.get("model_chi_cos_squared"),
            "first_model_alpha": endpoint.get("first_model_alpha"),
            "initial_model_alpha": endpoint.get("initial_model_alpha_alpha0"),
            "accepted_alpha": endpoint.get("accepted_alpha"),
            "accepted_alpha_over_initial": endpoint.get("accepted_alpha_over_initial_model_alpha"),
            "full_residual_relative_error_to_actual_endpoint": actual.get(
                "full_residual_relative_error_to_actual_endpoint"),
            "full_residual_angle_degrees": actual.get("full_residual_angle_degrees"),
            "delta_vector_relative_error": actual.get("delta_vector_relative_error"),
            "delta_vector_angle_degrees": actual.get("delta_vector_angle_degrees"),
            "c12_c15_model_error": actual.get("c12_c15_model_error"),
            "accepted_endpoint_closed": endpoint.get("accepted_endpoint_closed"),
            "checks_passed": point.get("checks_passed")})
    return brief


def _input_anchor_matches(actual: Any, expected: dict[str, Any]) -> bool | None:
    """Compare fixed identities exactly, allowing only tanh's NumPy/Torch ULP drift."""
    if actual is None:
        return None
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        return False
    for key, value in expected.items():
        if key != "shifted_flow_fractions" and actual.get(key) != value:
            return False
    try:
        seen = resume.np.asarray(actual["shifted_flow_fractions"], dtype=resume.np.float64)
        target = resume.np.asarray(expected["shifted_flow_fractions"], dtype=resume.np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if seen.shape != (5,) or target.shape != (5,) or not resume.np.isfinite(seen).all():
        return False
    scale = resume.np.maximum(resume.np.abs(seen), resume.np.abs(target))
    budget = 8.0 * resume.np.finfo(resume.np.float64).eps * resume.np.maximum(
        scale, resume.np.finfo(resume.np.float64).tiny)
    return bool((resume.np.abs(seen-target) <= budget).all())


def _failed_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    """Close the superseded plan's receipt and show it stopped before products."""
    manifest = resume.read_json(FAILED_PREFLIGHT)
    archive_map = plan.get("archive_files", {})
    manifest_rel = FAILED_PREFLIGHT.relative_to(ROOT).as_posix()
    archive_path = FAILED_PREFLIGHT_ATTEMPT / "step.json.gz"
    run_path = FAILED_PREFLIGHT_ATTEMPT / "step.run.json"
    resource_path = FAILED_PREFLIGHT_ATTEMPT / "step.resource.json"
    log_path = FAILED_PREFLIGHT_ATTEMPT / "step.log"
    raw, child = resume.read_child(archive_path)
    run, resource = resume.read_json(run_path), resume.read_json(resource_path)
    raw_sha = hashlib.sha256(raw).hexdigest()
    checks = {"manifest_pin_matches_plan": archive_map.get(manifest_rel) == resume.sha_file(FAILED_PREFLIGHT),
        "failed_plan_snapshot_matches": manifest.get("failed_plan_sha256") == "5756508c5adb97ffb54bbdfb59ab35054f773dc032761fdc654ae3fb050b434a"
            and archive_map.get(manifest.get("failed_plan_snapshot", "").replace(str(ROOT)+"/", ""))
                == manifest.get("failed_plan_sha256"),
        "raw_hash_matches": manifest.get("raw_sha256") == raw_sha,
        "run_hash_matches": manifest.get("run_sha256") == resume.sha_file(run_path),
        "resource_hash_matches": manifest.get("resource_sha256") == resume.sha_file(resource_path),
        "gzip_hash_matches": manifest.get("gzip_sha256") == resume.sha_file(archive_path),
        "log_hash_matches": manifest.get("log_sha256") == resume.sha_file(log_path),
        "lossless_roundtrip": manifest.get("lossless_roundtrip") is True,
        "no_row_products": child.get("jacobian_rows_started") == child.get("jacobian_rows_completed") == 0
            and len(child.get("jacobian_row_history", [])) == 0,
        "no_hvps": child.get("hvp_calls_started") == child.get("hvp_calls_completed") == 0
            and len(child.get("hvp_history", [])) == 0,
        "no_dense_solve_receipt": not child.get("coupled_gn_solve_history"),
        "no_candidate_or_commit": not child.get("iterations")
            and child.get("candidate_committed") is False
            and child.get("active_candidate_committed") is False
            and child.get("optimizer_steps_applied") == 0,
        "resource_is_preflight_failure": run.get("execution_status") == "failed"
            and run.get("numerical_status") == "preflight"
            and resource.get("exit_code") == 1
            and resource.get("resource_termination") is None}
    if not all(checks.values()):
        raise ValueError("superseded preflight archive failed its saved-receipt checks")
    return {"status": "preflight_failed_before_numerical_products",
        "plan_sha256": manifest.get("failed_plan_sha256"),
        "raw_sha256": raw_sha, "elapsed_seconds": resource.get("elapsed_seconds"),
        "sampled_peak_rss_bytes": resource.get("sampled_peak_rss_bytes"),
        "row_vjps_started_completed": [child.get("jacobian_rows_started"),
            child.get("jacobian_rows_completed")],
        "HVPs_started_completed": [child.get("hvp_calls_started"), child.get("hvp_calls_completed")],
        "dense_solves": child.get("dense_solves_started"),
        "candidate_slots": len(child.get("iterations", [])),
        "guarded_launch_count": manifest.get("guarded_launch_count"), "checks": checks}


def _tail_dense_audit(child: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Recheck the latest uncommitted 12x12 solve without claiming a full GN point."""
    current = resume.np.asarray(child.get("current_control", []), dtype=resume.np.float64)
    current_sha = resume.gn.vsha(current)
    current_theta = child.get("current_theta")
    minimum = child.get("base_mixing_minimum", {})
    working_theta = minimum.get("working_theta_star")
    history = [row for row in child.get("jacobian_row_history", [])
        if row.get("base_control_sha256") == current_sha]
    rows = resume._rows_for_point(child, current_sha, float(working_theta)) \
        if isinstance(working_theta, (int, float)) else {"count": 0}
    audit = child.get("dense_solve_audit")
    if not isinstance(audit, dict):
        return {"audit_present": False, "tail_full_model_point_closed": False}

    result: dict[str, Any] = {"audit_present": True,
        "checkpoint_current_control_sha256": current_sha,
        "last_confirmed_control_sha256": child.get("last_confirmed_control_sha256"),
        "audit_carries_base_control_sha256": "base_control_sha256" in audit,
        "tail_rows_at_current_control": rows.get("count"),
        "working_theta_star": working_theta,
        "current_theta": current_theta,
        "point_solve_history_receipts": len(child.get("coupled_gn_solve_history", [])),
        "modeled_point_history_receipts": len(child.get("coupled_gn_point_history", [])),
        "HVP_receipts_at_current_control": sum(h.get("base_control_sha256") == current_sha
            for h in child.get("hvp_history", [])),
        "candidate_trials_at_current_control": sum(
            trial.get("base_control_sha256") == current_sha
            for step in child.get("iterations", [])
            for arm in step.get("model_comparisons", [])
            for trial in arm.get("trials", []))}
    if current_sha != child.get("last_confirmed_control_sha256") or not (
            rows.get("count") == 24 and rows.get("finite")
            and all(rows.get("complete_by_side", {}).values())
            and all(row.get("status") == "completed" and row.get("theta") == working_theta
                for row in history)):
        result.update(tail_linear_solve_audit_closes=False,
            tail_full_model_point_closed=False, reason="current-point row/control context does not close")
        return result

    B = resume.np.asarray(audit.get("B"), dtype=resume.np.float64)
    S = resume.np.asarray(audit.get("S"), dtype=resume.np.float64)
    t = resume.np.asarray(audit.get("t"), dtype=resume.np.float64)
    Bt = resume.np.asarray(audit.get("B_t"), dtype=resume.np.float64)
    solve = resume.np.asarray(audit.get("solve_vector"), dtype=resume.np.float64)
    direction = resume.np.asarray(audit.get("direction"), dtype=resume.np.float64)
    uncorrected = resume.np.asarray(audit.get("uncorrected_direction"), dtype=resume.np.float64)
    correction = resume.np.asarray(audit.get("woodbury_correction"), dtype=resume.np.float64)
    face = plan["face"]
    _, normal, _ = resume.gn.face_geometry(current, face)
    unit = normal / resume.gn.norm(normal)
    projector = resume.np.eye(resume.gn.M) - resume.np.outer(unit, unit)
    robust_d = (2.0 / resume.np.hypot(2.0, rows["r_minus"])) ** 3
    expected_B = resume.np.sqrt(robust_d)[:, None] * (rows["A_minus"] @ projector)
    closure = child.get("last_confirmed_closure", {})
    closure_control = resume.np.asarray(closure.get("control", []), dtype=resume.np.float64)
    closure_minimum = resume.gn.theta_minimum(
        closure.get("side_gradients", {}).get("-1", []),
        closure.get("side_gradients", {}).get("1", []))
    closure_mixture = ((1.0-closure_minimum["theta_star"])
        * resume.np.asarray(closure["side_gradients"]["-1"], dtype=resume.np.float64)
        + closure_minimum["theta_star"]
        * resume.np.asarray(closure["side_gradients"]["1"], dtype=resume.np.float64))
    expected_t = projector @ closure_mixture
    scale_tolerance = 4096.0 * resume.np.finfo(resume.np.float64).eps
    B_error = resume.gn.rel_error(B, expected_B)
    expected_S = resume.np.eye(resume.gn.N) + B @ B.T
    S_error = resume.gn.rel_error(S, expected_S)
    t_error = resume.gn.rel_error(t, expected_t)
    expected_Bt = B @ expected_t
    Bt_error = resume.gn.rel_error(Bt, expected_Bt)
    try:
        expected_S_from_rows = resume.np.eye(resume.gn.N) + expected_B @ expected_B.T
        expected_rhs = expected_B @ expected_t
        expected_solve = resume.np.linalg.solve(expected_S_from_rows, expected_rhs)
        eigen_minimum = float(resume.np.linalg.eigvalsh(S).min())
    except resume.np.linalg.LinAlgError:
        expected_solve = resume.np.full_like(solve, resume.np.nan)
        eigen_minimum = float("-inf")
    solve_error = resume.gn.rel_error(solve, expected_solve)
    solve_residual = resume.gn.norm(S @ solve - expected_rhs)
    expected_correction = expected_B.T @ expected_solve
    correction_error = resume.gn.rel_error(correction, expected_correction)
    expected_uncorrected = -expected_t + expected_correction
    uncorrected_error = resume.gn.rel_error(uncorrected, expected_uncorrected)
    pivot = int(20 + resume.np.argmax(resume.np.abs(normal[20:25])))
    retained = [index for index in range(resume.gn.M) if index != pivot]
    chart = resume.np.zeros((resume.gn.M, resume.gn.M-1), dtype=resume.np.float64)
    chart[retained, resume.np.arange(resume.gn.M-1)] = 1.0
    for column, coordinate in enumerate(retained):
        if 20 <= coordinate < 25:
            chart[pivot, column] = -normal[coordinate]/normal[pivot]
    expected_direction = chart @ expected_uncorrected[retained]
    direction_error = resume.gn.rel_error(direction, expected_direction)
    tangent_normal_residual = abs(float(normal @ direction))
    audits = {"mixture_t_recomputed_from_last_confirmed_P2_gradients": t_error,
        "B_recomputed_from_tail_rows": B_error,
        "S_equals_I_plus_BB_transpose": S_error,
        "B_t_equals_B_times_t": Bt_error,
        "solve_vector_matches_numpy_solve": solve_error,
        "woodbury_correction_matches_Btranspose_solve": correction_error,
        "direction_matches_charted_uncorrected": direction_error,
        "uncorrected_direction_matches_minus_t_plus_correction": uncorrected_error,
        "solve_residual_recomputed": solve_residual,
        "solve_residual_saved": audit.get("solve_residual"),
        "solve_residual_budget_saved": audit.get("solve_residual_budget"),
        "S_minimum_eigenvalue": eigen_minimum,
        "tangent_normal_residual": tangent_normal_residual,
        "tangent_repair_l2_saved": audit.get("tangent_repair_l2"),
        "tangent_repair_budget_saved": audit.get("tangent_repair_budget")}
    numerically_closed = (B.shape == (resume.gn.N, resume.gn.M)
        and S.shape == (resume.gn.N, resume.gn.N) and t.shape == (resume.gn.M,)
        and Bt.shape == (resume.gn.N,) and solve.shape == (resume.gn.N,)
        and direction.shape == (resume.gn.M,) and uncorrected.shape == (resume.gn.M,)
        and correction.shape == (resume.gn.M,)
        and all(resume.np.isfinite(value).all() for value in
            (B, S, t, Bt, solve, direction, uncorrected, correction))
        and closure_control.shape == current.shape
        and resume.gn.vsha(closure_control) == current_sha
        and closure_minimum["strict_interior"]
        and resume.near(float(closure_minimum["theta_star"]), float(working_theta))
        and all(item.get("shape_match") is True
            and item.get("relative_l2_error", float("inf")) <= scale_tolerance
            for item in (t_error, B_error, S_error, Bt_error, solve_error, correction_error,
                direction_error, uncorrected_error))
        and eigen_minimum > 1.0
        and solve_residual <= float(audit.get("solve_residual_budget", -1.0))
        and tangent_normal_residual <= 4096.0*resume.np.finfo(resume.np.float64).eps
            * max(resume.gn.norm(normal)*resume.gn.norm(direction), resume.np.finfo(float).tiny))
    result.update({"dense_solve_dimension": audit.get("dense_solve_dimension"),
        "dense_solves_in_audit": audit.get("dense_solves"),
        "latest_audit_context_matches_current_control": current_sha == child.get("last_confirmed_control_sha256")
            and current_theta == child.get("last_confirmed_theta")
            and working_theta == current_theta,
        "solve_input_gradient_source": "same-control last_confirmed_closure side gradients, independently minimized and P2-closed",
        "solve_input_current_control_matches_closure": (closure_control.shape == current.shape
            and resume.gn.vsha(closure_control) == current_sha),
        "tail_linear_solve_audit_closes": bool(numerically_closed),
        "tail_full_model_point_closed": False,
        "full_model_point_omissions": ["per-point solve-history receipt", "paired HVP receipts",
            "complete direction-model arm receipt", "candidate search", "independent endpoint repeat"],
        "audit_arrays": audits})
    return result


def collect(child_path: Path, run_path: Path, resource_path: Path,
        plan_path: Path, archive_manifest_path: Path | None = None) -> dict[str, Any]:
    if resume.sha_file(plan_path) != EXPECTED_PLAN_SHA:
        raise ValueError("partial-resume plan digest differs from the reviewed frozen plan")
    plan = resume.read_json(plan_path)
    base = _load_partial_prefix(plan, ROOT)
    failed_preflight = _failed_preflight(plan)
    original_loader = resume.load_pr273_base
    resume.load_pr273_base = lambda _plan, _root: base
    try:
        result = resume.collect(child_path, run_path, resource_path, plan_path, archive_manifest_path)
    finally:
        resume.load_pr273_base = original_loader

    raw = resume.read_json(child_path)
    progress = raw.get("comparison_progress")
    progress_base = progress.get("base_control_sha256") if isinstance(progress, dict) else None
    committed = int(result["confirmed_prefix"]["accepted_steps"])
    last_confirmed_sha = raw.get("last_confirmed_control_sha256")
    progress_matches_successor = (isinstance(progress, dict)
        and progress_base == last_confirmed_sha
        and progress.get("index") == committed)
    progress_state_valid = not isinstance(progress, dict) or progress_matches_successor
    anchor = base["anchor"]
    global_identity = result["global_input_runtime_identity"]
    fields_present = all(raw.get(key) is not None for key in
        ("input_before", "input_after", "runtime_before", "runtime_after"))
    anchor_result = {"provenance_matches_plan": raw.get("base_anchor_provenance") == anchor["provenance"],
        "derived_input_control_sha": anchor["input"].get("control_sha256"),
        "derived_input_matches_new_before": _input_anchor_matches(raw.get("input_before"), anchor["input"]),
        "derived_runtime_matches_new_before": raw.get("runtime_before") == anchor["runtime"]
            if raw.get("runtime_before") is not None else None,
        "new_before_after_fields_present": fields_present,
        "new_input_unchanged": raw.get("input_before") == raw.get("input_after")
            if fields_present else None,
        "new_runtime_unchanged": raw.get("runtime_before") == raw.get("runtime_after")
            if fields_present else None,
        "new_source_before_after_matches": raw.get("source_before") == raw.get("source_after")
            if isinstance(raw.get("source_after"), dict) else None,
        "interrupted_predecessor_global_identity": "not_recorded",
        "checks_match_collector": global_identity.get("comparison_passed")}
    result["base"] = {"source": "closed PR274 saved P2 endpoint",
        "control_sha256": base["control_sha256"], "theta_star": base["theta"],
        "native_J": base["objective"], "F_squared": base["F_squared"],
        "partial_raw_sha256": base["raw_sha256"],
        "partial_receipt_hashes": base["producer_receipt_hashes"],
        "interrupted_attempt_terminal_closed": False,
        "source_before_verified": True, "source_after_recorded": False,
        "global_input_runtime_identity": "not_recorded_on_interrupted_predecessor"}
    result["resume_prefix"] = {"confirmed_steps_from_predecessor": 1,
        "predecessor_execution_status": "rss_limit", "predecessor_tail_reused": False,
        "predecessor_tail_rows_started_completed": [base["tail_counts"]["rows_started"],
            base["tail_counts"]["rows_completed"]],
        "predecessor_tail_hvps_dense_solves_candidates_commits": [base["tail_counts"]["HVPs"],
            base["tail_counts"]["dense_solves"], base["tail_counts"]["candidates"],
            base["tail_counts"]["commits"]],
        "predecessor_duplicate_progress_verified": base["predecessor_progress_was_duplicate"],
        "base_side_gradients_source": "last_confirmed_closure side_gradients; P2 rechecked against accepted trial",
        "anchor_provenance": anchor["provenance"],
        "derived_input_fields_changed": base["changed_anchor_input_fields"],
        "shifted_flow_source": "NumPy tanh of saved confirmed control components 20:25; compare to runtime receipt within 8 float64 eps per component"}
    result["resume_anchor_checks"] = anchor_result
    result["checkpoint_progress"] = {"present_in_latest_checkpoint": isinstance(progress, dict),
        "base_control_sha256": progress_base,
        "index": progress.get("index") if isinstance(progress, dict) else None,
        "matches_uncommitted_successor": progress_matches_successor,
        "latest_state_valid": progress_state_valid,
        "absent_during_uncommitted_point_is_permitted": (not isinstance(progress, dict)
            and raw.get("phase") == "running"),
        "absent_at_completed_terminal_checkpoint": (raw.get("phase") == "finished"
            and raw.get("execution_status") == "completed" and not isinstance(progress, dict)),
        "commit_callback_excludes_progress_before_write": True}
    result["point_summaries"] = _brief_metrics(result)
    result["uncommitted_tail_dense_audit"] = _tail_dense_audit(raw, plan)
    result["accounting"]["dense_per_point_receipts"] = result["accounting"]["dense_receipts"]
    result["accounting"]["dense_counter_receipt_gap"] = (
        result["accounting"]["dense_completed"] - result["accounting"]["dense_receipts"])
    result["accounting"]["tail_dense_solve_audit_closes"] = result[
        "uncommitted_tail_dense_audit"].get("tail_linear_solve_audit_closes")
    result["memory_and_claim_limits"] = {
        "rss_savings_guaranteed": False,
        "sampled_resource_guard_closed": result["accounting"].get("resource_closed"),
        "solver_convergence_established": False,
        "minimum_root_response_adjoint_reanalysis_forecast_claims": False}
    result["superseded_preflight"] = failed_preflight
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, default=ATTEMPT / "step.json")
    parser.add_argument("--run", type=Path, default=ATTEMPT / "step.run.json")
    parser.add_argument("--resource", type=Path, default=ATTEMPT / "step.resource.json")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--archive-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = args.archive_manifest
    if manifest is not None and not manifest.exists():
        raise FileNotFoundError(manifest)
    result = collect(args.child, args.run, args.resource, args.plan, manifest)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
        allow_nan=False, default=lambda value: value.item()) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "execution_status", "numerical_status", "base", "confirmed_prefix",
        "point_summaries", "accounting", "uncommitted_tail_dense_audit",
        "resume_anchor_checks", "checkpoint_progress",
        "source_checks", "parent_checks", "superseded_preflight", "terminal_closed")}, indent=2, sort_keys=True,
        default=lambda value: value.item()))


if __name__ == "__main__":
    main()
