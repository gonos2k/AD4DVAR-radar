#!/usr/bin/env python3
"""Audit saved receipts for the prepared-GN dyadic search; never runs the model."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "GN_PREPARED_SEARCH_PLAN_20261010.json"
PREFLIGHT = EVIDENCE / "GN_PREPARED_SEARCH_PREFLIGHT_20261010.json"
PRIOR_RAW = EVIDENCE / "gn_local_commit_20261010_attempt1/step.json"
PRIOR_GZIP = EVIDENCE / "gn_local_commit_20261010_attempt1/step.json.gz"
ATTEMPT = EVIDENCE / "gn_prepared_search_20261010_attempt1"
ACTUAL_RAW = ATTEMPT / "step.json"
ACTUAL_GZIP = ATTEMPT / "step.json.gz"
ACTUAL = ACTUAL_GZIP if ACTUAL_GZIP.exists() else ACTUAL_RAW
RUN = ATTEMPT / "step.run.json"
RESOURCE = ATTEMPT / "step.resource.json"
REPORT = EVIDENCE / "GN_PREPARED_SEARCH_SAVED_AUDIT_20261010.md"
RESULT = EVIDENCE / "GN_PREPARED_SEARCH_SAVED_AUDIT_20261010.json"
PLAN_SHA = "1392032f3f0b859f74f03551a98fa5465216cb7fd981250984b603a495526a1a"
BASE_SHA = "311f27d095be4976ca6131b83b6c4f3d46bc404d4f58921800b5600e38ec0652"
DIRECTION_SHA = "f8862290dfcd7ec73e924182e5bb5376c1a66bd8a890de3e55b84e73b5501595"


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def vector_norm_delta(left: Any, right: Any) -> float | None:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return None
    try:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
    except (TypeError, ValueError, OverflowError):
        return None


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    pairs = {
        "jacobian_rows": ("jacobian_rows_completed", "jacobian_rows_started"),
        "dense_solves": ("dense_solves_completed", "dense_solves_started"),
        "hvps": ("hvp_calls_completed", "hvp_calls_started"),
        "optimizer_steps": ("optimizer_steps_applied",),
    }
    result: dict[str, Any] = {}
    for label, keys in pairs.items():
        result[label] = {}
        for key in keys:
            old, new = before.get(key), after.get(key)
            result[label][key] = {
                "before": old,
                "after": new,
                "delta": (new - old) if isinstance(old, (int, float))
                    and isinstance(new, (int, float)) else None,
            }
    return result


def per_plan_counts(raw: dict[str, Any]) -> dict[str, Any]:
    """New guarded plans start their own counters at zero; never subtract plans."""
    return counter_delta({key: 0 for key in (
        "jacobian_rows_completed", "jacobian_rows_started", "dense_solves_completed",
        "dense_solves_started", "hvp_calls_completed", "hvp_calls_started",
        "optimizer_steps_applied")}, raw)


def saved_tensor_sha(values: Any) -> str | None:
    """Match torch float64 tensor hashing from the JSON-saved flat vector bytes."""
    if not isinstance(values, list):
        return None
    try:
        return hashlib.sha256(struct.pack("<" + "d" * len(values), *(float(v) for v in values))).hexdigest()
    except (TypeError, ValueError, OverflowError, struct.error):
        return None


def max_scaled_vector_error(left: Any, right: Any) -> tuple[float | None, float | None]:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return None, None
    try:
        a = [float(value) for value in left]
        b = [float(value) for value in right]
        error = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        scale = max(max((abs(x) for x in a), default=0.0), max((abs(y) for y in b), default=0.0), 2.2250738585072014e-308)
        return error, 128.0 * 2.220446049250313e-16 * scale
    except (TypeError, ValueError, OverflowError):
        return None, None


def candidate_g_audit(item: dict[str, Any]) -> dict[str, Any]:
    g = item.get("G")
    merit = first(item, "actual_F_squared", "F_squared", "residual_merit")
    theta = first(item, "theta", "theta_minimum")
    sides = item.get("side_gradients", {})
    result: dict[str, Any] = {"theta": theta,
        "theta_strictly_interior": isinstance(theta, (int, float)) and 0.0 < theta < 1.0,
        "G_dimension": len(g) if isinstance(g, list) else None,
        "recorded_R_or_G2": merit}
    try:
        if not isinstance(g, list) or len(g) != 27 or not isinstance(merit, (int, float)):
            raise ValueError
        values = [float(value) for value in g]
        computed = sum(value * value for value in values)
        budget = 128.0 * 2.220446049250313e-16 * max(abs(computed), abs(float(merit)), 2.2250738585072014e-308)
        result.update({"computed_G2": computed, "G2_error": abs(computed - float(merit)),
            "G2_roundoff_budget": budget, "G2_matches_recorded_merit": abs(computed - float(merit)) <= budget})
        minus = sides.get("-1") if isinstance(sides, dict) else None
        plus = sides.get("1") if isinstance(sides, dict) else None
        if not isinstance(minus, list) or not isinstance(plus, list) or len(minus) != 26 or len(plus) != 26:
            raise ValueError
        mixed = [(1.0 - float(theta)) * float(a) + float(theta) * float(b)
                 for a, b in zip(minus, plus)]
        error, side_budget = max_scaled_vector_error(mixed, values[:26])
        result.update({"mixed_side_gradient_max_abs_error": error,
            "mixed_side_gradient_budget": side_budget,
            "G_first26_matches_mixed_sides": error is not None and side_budget is not None and error <= side_budget})
    except (TypeError, ValueError, OverflowError):
        result.update({"G2_matches_recorded_merit": False, "G_first26_matches_mixed_sides": False})
    return result


def verify_pinned_files(plan: dict[str, Any]) -> dict[str, Any]:
    outcome: dict[str, Any] = {}
    for group in ("source_files", "archive_files"):
        pins = plan.get(group, {})
        matches, missing, mismatches = 0, [], []
        for relative, expected_sha in pins.items():
            path = ROOT / relative
            if not path.is_file():
                missing.append(relative)
                continue
            actual_sha = sha256(path)
            if actual_sha != expected_sha:
                mismatches.append({"path": relative, "expected": expected_sha, "actual": actual_sha})
            else:
                matches += 1
        outcome[group] = {"expected_count": len(pins), "matched_count": matches,
            "missing": missing, "mismatches": mismatches,
            "passed": matches == len(pins) and not missing and not mismatches}
    return outcome


def find_candidate_records(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect line-search records from known result containers, preserving order."""
    for key in ("candidate_trials", "trials", "search_trials", "candidate_history", "candidates"):
        value = raw.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for parent in ("search", "line_search", "prepared_search"):
        value = raw.get(parent)
        if isinstance(value, dict):
            for key in ("candidate_trials", "trials", "candidates"):
                records = value.get(key)
                if isinstance(records, list):
                    return [item for item in records if isinstance(item, dict)]
    return []


def main() -> None:
    plan_bytes_sha = sha256(PLAN)
    plan = read_json(PLAN)
    preflight = read_json(PREFLIGHT)
    prior = read_json(PRIOR_GZIP if PRIOR_GZIP.exists() else PRIOR_RAW)
    raw = read_json(ACTUAL)
    run = read_json(RUN) if RUN.exists() else {}
    resource = read_json(RESOURCE) if RESOURCE.exists() else {}

    pins_ok = (
        plan_bytes_sha == PLAN_SHA
        and preflight.get("plan_sha256") == PLAN_SHA
        and preflight.get("load_plan_passed") is True
        and preflight.get("archive_identity_passed") is True
        and preflight.get("source_count") == 151
        and preflight.get("archive_count") == 195
        and plan.get("base_control_sha256") == BASE_SHA
        and plan.get("direction_sha256") == DIRECTION_SHA
    )
    expected = {
        "search_candidate_cap": plan.get("candidate_cap"),
        "optimizer_steps": plan.get("policy", {}).get("optimizer_steps"),
        "postcommit_rows": plan.get("policy", {}).get("jacobian_row_vjp_calls"),
        "postcommit_dense_solves": plan.get("policy", {}).get("dense_solves"),
        "fresh_postcommit_hvps": plan.get("operator_reuse", {}).get("fresh_postcommit_hvp_calls"),
        "same_point_archived_hvp_vectors": plan.get("operator_reuse", {}).get("same_point_archived_hvp_vectors"),
        "fresh_base_hvps": plan.get("operator_reuse", {}).get("fresh_base_hvp_calls"),
    }
    counts = per_plan_counts(raw)
    prior_counts = {key: prior.get(key, 0) for key in (
        "jacobian_rows_completed", "dense_solves_completed", "hvp_calls_completed", "optimizer_steps_applied")}
    trials = find_candidate_records(raw)
    accepted = next((trial for trial in trials if trial.get("accepted") is True), None)
    proposal = raw.get("proposal") if isinstance(raw.get("proposal"), dict) else {}
    repeat = raw.get("final_repeat") if isinstance(raw.get("final_repeat"), dict) else {}
    final = raw.get("postcommit_current_observation")
    if not isinstance(final, dict):
        final = raw.get("postcommit_gn_readiness")
    if not isinstance(final, dict):
        final = {}

    base_j = first(raw, "base_objective", default=prior.get("current_objective", prior.get("base_objective")))
    if base_j is None:
        base_j = prior.get("postcommit_current_observation", {}).get("native_objective")
    base_r = first(raw, "base_F_squared", default=prior.get("base_F_squared"))
    if base_r is None:
        base_r = prior.get("postcommit_current_observation", {}).get("F_squared")
    final_j = first(final, "native_objective", "objective", "native_j")
    if final_j is None:
        final_j = first(proposal, "objective", "native_objective")
    final_r = first(final, "F_squared", "residual_merit")
    if final_r is None:
        final_r = proposal.get("F_squared")
    base_control = raw.get("base_control", prior.get("current_control"))
    final_control = first(raw, "current_control", default=proposal.get("control"))
    progress = {
        "base_objective": base_j,
        "final_objective": final_j,
        "objective_change": (final_j - base_j) if isinstance(base_j, (int, float))
            and isinstance(final_j, (int, float)) else None,
        "objective_decrease_percent": (100.0 * (base_j - final_j) / abs(base_j))
            if isinstance(base_j, (int, float)) and isinstance(final_j, (int, float)) and base_j != 0 else None,
        "base_R_or_G2": base_r,
        "final_R_or_G2": final_r,
        "R_or_G2_change": (final_r - base_r) if isinstance(base_r, (int, float))
            and isinstance(final_r, (int, float)) else None,
        "R_or_G2_decrease_percent": (100.0 * (base_r - final_r) / abs(base_r))
            if isinstance(base_r, (int, float)) and isinstance(final_r, (int, float)) and base_r != 0 else None,
        "control_displacement_l2": vector_norm_delta(base_control, final_control),
        "candidate_alpha": first(proposal, "alpha", default=raw.get("postcommit_candidate_alpha")),
        "accepted_trial_index_1_based": (trials.index(accepted) + 1) if accepted in trials else None,
        "candidate_trials_recorded": len(trials),
    }

    # Reconstruct the saved first-order residual model at the predecessor point.
    # This is arithmetic on archived G and dG vectors, not a new model evaluation.
    prior_ready = prior.get("postcommit_gn_readiness", {})
    base_residual = prior_ready.get("residual") if isinstance(prior_ready, dict) else None
    residual_direction = prior_ready.get("residual_direction") if isinstance(prior_ready, dict) else None
    selected_trial = accepted if accepted is not None else None
    model_math: dict[str, Any] = {"source": "archived 311f residual plus alpha times archived residual_direction"}
    try:
        alpha = float(proposal["alpha"])
        r0 = [float(value) for value in base_residual]
        dr = [float(value) for value in residual_direction]
        g1 = [float(value) for value in selected_trial["G"]]
        if not (len(r0) == len(dr) == len(g1) == 27):
            raise ValueError
        predicted = [base + alpha * slope for base, slope in zip(r0, dr)]
        residual_error = math.sqrt(sum((actual - model) ** 2 for actual, model in zip(g1, predicted)))
        actual_vector_change = math.sqrt(sum((actual - base) ** 2 for actual, base in zip(g1, r0)))
        base_norm = math.sqrt(sum(value * value for value in r0))
        final_norm = math.sqrt(sum(value * value for value in g1))
        predicted_R = sum(value * value for value in predicted)
        actual_R = sum(value * value for value in g1)
        base_R = sum(value * value for value in r0)
        initial_alpha = raw.get("search", {}).get("computed_alpha0")
        alpha_ratio = alpha / float(initial_alpha) if isinstance(initial_alpha, (int, float)) else None
        model_math.update({
            "archived_residual_direction_sha256": prior_ready.get("direction_sha256"),
            "plan_direction_sha256": DIRECTION_SHA,
            "alpha": alpha,
            "initial_alpha": initial_alpha,
            "alpha_over_initial": alpha_ratio,
            "dyadic_halvings": (-math.log2(alpha_ratio)) if alpha_ratio and alpha_ratio > 0 else None,
            "base_G_norm_l2": base_norm,
            "base_G_norm_linf": max(abs(value) for value in r0),
            "final_G_norm_l2": final_norm,
            "final_G_norm_linf": max(abs(value) for value in g1),
            "actual_residual_vector_change_l2": actual_vector_change,
            "linear_model_residual_error_l2": residual_error,
            "model_error_over_actual_residual_change": residual_error / actual_vector_change
                if actual_vector_change > 0 else None,
            "actual_R_decrease": base_R - actual_R,
            "linear_model_R_decrease": base_R - predicted_R,
            "actual_over_model_R_decrease": (base_R - actual_R) / (base_R - predicted_R)
                if base_R != predicted_R else None,
        })
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        model_math["reconstruction_error"] = "required saved vectors or scalar were absent/invalid"

    hvp_history = raw.get("hvp_history", [])
    prior_hvp_history = prior.get("hvp_history", [])
    new_base_sha = first(raw, "current_control_sha256", default=proposal.get("control_sha256"))
    fresh_hvps = [item for item in hvp_history if isinstance(item, dict)
        and item.get("base_control_sha256") == new_base_sha]
    archived_same_point = [item for item in prior_hvp_history if isinstance(item, dict)
        and item.get("base_control_sha256") == BASE_SHA]
    hvps = {
        "plan_reuses_archived_vectors": expected["same_point_archived_hvp_vectors"],
        "verified_prior_same_point_hvp_records": len(archived_same_point),
        "prior_closed_plan_counts_not_subtracted": prior_counts,
        "fresh_hvp_records_at_new_control": len(fresh_hvps),
        "fresh_hvps_from_counter_delta": counts["hvps"]["hvp_calls_completed"]["delta"],
        "recorded_fresh_hvp_details": fresh_hvps,
    }

    candidate_summary = []
    candidate_g_checks = []
    for index, item in enumerate(trials, 1):
        g_audit = candidate_g_audit(item)
        candidate_g_checks.append(g_audit)
        trial_j = first(item, "actual_J", "objective", "native_j")
        trial_j_bound = first(item, "J_armijo_bound", "actual_J_armijo_bound")
        trial_r = first(item, "actual_F_squared", "F_squared", "residual_merit")
        trial_r_bound = first(item, "F_squared_armijo_bound", "actual_F2_armijo_bound")
        candidate_summary.append({
            "index": index,
            "alpha": first(item, "alpha"),
            "accepted": item.get("accepted"),
            "status": item.get("status"),
            "J": trial_j,
            "J_armijo_bound": trial_j_bound,
            "J_armijo_slack": (trial_j_bound - trial_j) if isinstance(trial_j, (int, float))
                and isinstance(trial_j_bound, (int, float)) else None,
            "R_or_G2": trial_r,
            "R_or_G2_armijo_bound": trial_r_bound,
            "R_or_G2_armijo_slack": (trial_r_bound - trial_r) if isinstance(trial_r, (int, float))
                and isinstance(trial_r_bound, (int, float)) else None,
            "theta": first(item, "theta", "theta_minimum"),
            "control_sha256": item.get("control_sha256"),
            "G2_matches_recorded_merit": g_audit.get("G2_matches_recorded_merit"),
            "G_first26_matches_mixed_sides": g_audit.get("G_first26_matches_mixed_sides"),
            "theta_strictly_interior": g_audit.get("theta_strictly_interior"),
            "branch_pair_passed": item.get("branch_pair_passed"),
            "trace_matches_base": item.get("trace_matches_base"),
        })

    pinned_files = verify_pinned_files(plan)
    run_exit = run.get("resource", {}).get("exit_code")
    if run_exit is None:
        run_exit = run.get("exit_code")
    child_sha = run.get("child_sha256")
    if ACTUAL.suffix == ".gz":
        with gzip.open(ACTUAL, "rb") as stream:
            step_raw_sha = hashlib.sha256(stream.read()).hexdigest()
    else:
        step_raw_sha = sha256(ACTUAL)
    reported_control = raw.get("current_control")
    computed_control_sha = saved_tensor_sha(reported_control)
    control_sha_matches = computed_control_sha == raw.get("current_control_sha256")

    # The candidate and final repeat are independent P2 records. Compare both side gradients.
    proposal_sides = proposal.get("side_gradients", {})
    repeat_sides = repeat.get("side_gradients", {})
    side_gradient_audits = {}
    for side in ("-1", "1"):
        err, budget = max_scaled_vector_error(proposal_sides.get(side), repeat_sides.get(side))
        side_gradient_audits[side] = {"max_abs_error": err, "budget": budget,
            "passed": err is not None and budget is not None and err <= budget}
    repeated_theta = first(repeat, "theta", "minimum_theta")
    theta_interior = isinstance(repeated_theta, (int, float)) and 0.0 < repeated_theta < 1.0
    repeat_p2 = {
        "side_gradient_consistency": side_gradient_audits,
        "all_side_gradients_match": all(v["passed"] for v in side_gradient_audits.values()),
        "repeat_theta": repeated_theta,
        "theta_strictly_interior": theta_interior,
        "minimum_theta_matches_proposal": repeat.get("minimum_theta_matches_proposal"),
        "objective_matches_proposal": repeat.get("native_objective_matches_proposal"),
        "merit_matches_proposal": repeat.get("merit_matches_proposal"),
        "trace_matches_proposal": repeat.get("trace_matches_proposal"),
    }
    readiness = raw.get("postcommit_gn_readiness", {})
    current_sha = raw.get("current_control_sha256")
    row_history = raw.get("jacobian_row_history", [])
    row_pairs = {(row.get("side"), row.get("row")) for row in row_history if isinstance(row, dict)}
    solve_audit = raw.get("dense_solve_audit", {})
    readiness_direction_sha = readiness.get("direction_sha256") if isinstance(readiness, dict) else None
    postcommit_counts_close = (
        raw.get("jacobian_rows_completed") == 24 and raw.get("jacobian_rows_started") == 24
        and len(row_history) == 24 and row_pairs == {(side, row) for side in (-1, 1) for row in range(12)}
        and all(row.get("status") == "completed" and row.get("base_control_sha256") == current_sha
            for row in row_history if isinstance(row, dict))
        and raw.get("dense_solves_completed") == 1 and raw.get("dense_solves_started") == 1
        and solve_audit.get("dense_solve_dimension") == 12
        and solve_audit.get("positive_definite_from_cholesky") is True
        and isinstance(solve_audit.get("solve_residual"), (int, float))
        and isinstance(solve_audit.get("solve_residual_budget"), (int, float))
        and solve_audit["solve_residual"] <= solve_audit["solve_residual_budget"]
        and raw.get("hvp_calls_completed") == 2 and raw.get("hvp_calls_started") == 2
        and len(fresh_hvps) == 2
        and all(item.get("direction_sha256") == readiness_direction_sha
            and item.get("base_control_sha256") == current_sha
            and item.get("status") == "completed" for item in fresh_hvps)
        and readiness.get("candidate_count") == 0
    )
    model_math_close = (
        "reconstruction_error" not in model_math
        and model_math.get("archived_residual_direction_sha256") == DIRECTION_SHA
        and isinstance(model_math.get("dyadic_halvings"), (int, float))
        and abs(model_math["dyadic_halvings"] - round(model_math["dyadic_halvings"])) <= 1e-12
    )

    checks = {
        "frozen_plan_and_preflight_pins_match": pins_ok,
        "all_frozen_source_and_archive_hashes_match": all(v["passed"] for v in pinned_files.values()),
        "run_completed_with_explicit_zero_exit": run.get("execution_status") == "completed" and run_exit == 0,
        "run_child_sha_matches_saved_step": isinstance(child_sha, str) and child_sha == step_raw_sha,
        "base_control_matches_plan": raw.get("base_control_sha256") == BASE_SHA,
        "current_control_tensor_byte_sha_matches": control_sha_matches,
        "search_used_at_most_24_trials": len(trials) <= 24,
        "at_most_one_step_applied": raw.get("optimizer_steps_applied") in (0, 1),
        "postcommit_readiness_complete_or_no_commit": raw.get("readiness_complete") is True
            or raw.get("optimizer_steps_applied") == 0,
        "source_input_runtime_closed": raw.get("source_unchanged") is True
            and raw.get("fixed_input_unchanged") is True and raw.get("runtime_unchanged") is True,
        "two_prior_hvps_reused_and_two_new_hvps_computed": len(archived_same_point) == 2
            and len(fresh_hvps) == 2 and counts["hvps"]["hvp_calls_completed"]["delta"] == 2,
        "all_candidate_G_norms_and_mixed_gradients_close": bool(candidate_g_checks)
            and all(item.get("G2_matches_recorded_merit") is True
                and item.get("G_first26_matches_mixed_sides") is True
                and item.get("theta_strictly_interior") is True for item in candidate_g_checks),
        "all_candidate_paired_branches_passed": bool(trials)
            and all(item.get("branch_pair_passed") is True for item in trials),
        "search_stopped_at_first_passing_dyadic_slot": raw.get("search", {}).get("first_pass_candidate_index")
            == progress.get("accepted_trial_index_1_based")
            and raw.get("search", {}).get("evaluated_count") == len(trials),
        "saved_residual_model_math_closes": model_math_close,
        "postcommit_24row_solve_hvp_receipt_closes": postcommit_counts_close,
        "P2_individual_side_gradients_and_theta_close": accepted is None or (
            repeat_p2["all_side_gradients_match"] and repeat_p2["theta_strictly_interior"]
            and repeat_p2["minimum_theta_matches_proposal"] is True
            and repeat_p2["objective_matches_proposal"] is True
            and repeat_p2["merit_matches_proposal"] is True
            and repeat_p2["trace_matches_proposal"] is True),
    }
    def trace_signatures(record: dict[str, Any]) -> dict[str, Any]:
        traces = record.get("branch_trace", {})
        if not isinstance(traces, dict):
            return {}
        return {side: trace.get("signature_sha256") for side, trace in traces.items()
            if isinstance(trace, dict) and trace.get("signature_sha256")}

    def gradient_fingerprints(record: dict[str, Any]) -> dict[str, str | None]:
        sides = record.get("side_gradients", {})
        if not isinstance(sides, dict):
            return {}
        return {side: saved_tensor_sha(vector) for side, vector in sides.items()}

    accepted_summary = None
    if accepted is not None:
        accepted_summary = {key: accepted.get(key) for key in (
            "index", "alpha", "accepted", "status", "objective", "F_squared",
            "J_armijo_bound", "J_armijo_passed", "F_squared_armijo_bound", "F_squared_armijo_passed",
            "actual_path_norm", "theta", "control_sha256", "branch_pair_passed", "face_audit_passed",
            "side_objectives_match_native", "side_gradients_finite", "finite", "mixing_minimum_valid",
            "trace_matches_base")}
        accepted_summary["branch_trace_signatures"] = trace_signatures(accepted)
        accepted_summary["side_gradient_sha256"] = gradient_fingerprints(accepted)
        accepted_summary["G_audit"] = candidate_g_checks[trials.index(accepted)]
    proposal_summary = {key: proposal.get(key) for key in (
        "alpha", "objective", "F_squared", "theta", "control_sha256", "candidate_mixing_minimum",
        "J_armijo_bound", "J_armijo_passed", "F_squared_armijo_bound", "F_squared_armijo_passed",
        "branch_pair_passed", "face_audit_passed", "side_objectives_match_native", "side_gradients_finite")}
    proposal_summary["branch_trace_signatures"] = trace_signatures(proposal)
    proposal_summary["side_gradient_sha256"] = gradient_fingerprints(proposal)
    repeat_summary = {key: repeat.get(key) for key in (
        "objective", "F_squared", "theta", "mixing_minimum_valid", "minimum_theta_matches_proposal",
        "native_objective_matches_proposal", "merit_matches_proposal", "trace_matches_proposal",
        "branch_pair_passed", "face_audit_passed", "side_objectives_match_native", "side_gradients_finite",
        "fixed_input_unchanged", "runtime_unchanged", "source_unchanged", "deadline_passed")}
    repeat_summary["branch_trace_signatures"] = trace_signatures(repeat)
    repeat_summary["side_gradient_sha256"] = gradient_fingerprints(repeat)
    run_summary = {key: run.get(key) for key in (
        "candidate_committed", "child_sha256", "execution_status", "numerical_status", "readiness_complete")}
    resource_path = ATTEMPT / "step.resource.json"
    resource_summary = {key: resource.get(key) for key in (
        "child_pid", "exit_code", "resource_termination", "monitor_error", "elapsed_seconds",
        "sampled_peak_rss_bytes", "wall_limit_seconds", "rss_limit_bytes", "rss_samples",
        "received_sigterm", "child_process_group_cleanup_sent", "child_process_group_cleanup_error")}
    resource_summary.update({"path": resource_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(resource_path) if resource_path.is_file() else None})
    result = {
        "plan_sha256": plan_bytes_sha,
        "preflight": {k: preflight.get(k) for k in (
            "plan_sha256", "load_plan_passed", "archive_identity_passed", "source_count", "archive_count",
            "base_control_sha256", "direction_sha256", "production_executed")},
        "expected_plan_budgets": expected,
        "per_plan_0_to_1_counter_deltas": counts,
        "prior_plan_closed_counts_reference_only": prior_counts,
        "pinned_file_hash_audit": pinned_files,
        "operator_reuse_and_fresh_work": hvps,
        "progress": progress,
        "actual_vs_linear_model_math": model_math,
        "candidate_trials": candidate_summary,
        "candidate_G_audits": candidate_g_checks,
        "accepted_candidate_summary": accepted_summary,
        "proposal_summary": proposal_summary,
        "final_repeat_summary": repeat_summary,
        "independent_P2_audit": repeat_p2,
        "postcommit_readiness_counts_close": postcommit_counts_close,
        "postcommit_readiness": raw.get("postcommit_gn_readiness"),
        "run": run_summary,
        "saved_actual_receipt": {"path": ACTUAL.relative_to(ROOT).as_posix(),
            "uncompressed_step_sha256": step_raw_sha,
            "raw_copy_present_and_equal": (not ACTUAL_RAW.exists() or sha256(ACTUAL_RAW) == step_raw_sha)},
        "resource": resource_summary,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Prepared GN search saved-data audit",
        "",
        f"Plan SHA-256: `{plan_bytes_sha}`; frozen pin checks: **{pins_ok}**.",
        "",
        "This audit reads saved JSON only. It does not execute FV, seed preparation, AD, or HVP work.",
        "",
        "## Work counts",
        "",
        "| Counter | Before | After | Per-plan delta |",
        "|---|---:|---:|---:|",
    ]
    for label in ("jacobian_rows", "dense_solves", "hvps", "optimizer_steps"):
        for key, values in counts[label].items():
            lines.append(f"| {key} | {values['before']} | {values['after']} | {values['delta']} |")
    lines += [
        "",
        f"Archived same-point HVP records verified: {hvps['verified_prior_same_point_hvp_records']} / 2; "
        f"fresh HVP records at new point: {hvps['fresh_hvp_records_at_new_control']}; "
        f"HVP counter delta: {hvps['fresh_hvps_from_counter_delta']}.",
        "",
        f"Explicit run exit code: {run_exit}; child SHA matches saved step: "
        f"{checks['run_child_sha_matches_saved_step']}; current-control byte SHA matches: {control_sha_matches}.",
        f"Frozen source/archive pins: {pinned_files['source_files']['matched_count']}/"
        f"{pinned_files['source_files']['expected_count']} sources and "
        f"{pinned_files['archive_files']['matched_count']}/"
        f"{pinned_files['archive_files']['expected_count']} archives.",
        "",
        "## Candidate and progress",
        "",
        f"Trials recorded: {progress['candidate_trials_recorded']}; first accepted slot: "
        f"{progress['accepted_trial_index_1_based']}; alpha: {progress['candidate_alpha']}.",
        f"Objective decrease: {progress['objective_decrease_percent']}%; "
        f"residual-merit decrease: {progress['R_or_G2_decrease_percent']}%; "
        f"control displacement L2: {progress['control_displacement_l2']}.",
        "",
        "| # | alpha | J | J slack | R/G² | R/G² slack | paired branch | base trace | accepted | status |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|---|",
    ]
    for item in candidate_summary:
        lines.append("| {index} | {alpha} | {J} | {J_armijo_slack} | {R_or_G2} | "
                     "{R_or_G2_armijo_slack} | {branch_pair_passed} | {trace_matches_base} | "
                     "{accepted} | {status} |".format(**item))
    if "reconstruction_error" not in model_math:
        lines += [
            "",
            "The archived first-order residual model predicts a G² decrease of "
            f"{model_math['linear_model_R_decrease']}; the actual decrease is "
            f"{model_math['actual_R_decrease']} (actual/model "
            f"{model_math['actual_over_model_R_decrease']}). Residual-model error is "
            f"{model_math['linear_model_residual_error_l2']} L2, or "
            f"{model_math['model_error_over_actual_residual_change']} of the actual residual-vector change.",
        ]
    lines += ["", "## Closure checks", ""]
    lines += [f"- {key}: **{value}**" for key, value in checks.items()]
    lines += ["", f"All checks passed: **{all(checks.values())}**.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)
    print(RESULT)
    if not all(checks.values()):
        raise SystemExit("saved-data audit found an incomplete or mismatched receipt")


if __name__ == "__main__":
    main()
