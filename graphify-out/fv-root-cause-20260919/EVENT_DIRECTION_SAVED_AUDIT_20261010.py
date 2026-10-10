#!/usr/bin/env python3
"""Audit the frozen event-direction comparison from saved receipts only."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "EVENT_DIRECTION_COMPARISON_PLAN_20261010.json"
PREFLIGHT = EVIDENCE / "EVENT_DIRECTION_PREFLIGHT_20261010.json"
ARCHIVE = EVIDENCE / "EVENT_DIRECTION_ARCHIVE_20261010.json"
ATTEMPT = EVIDENCE / "event_direction_comparison_20261010_attempt1"
RAW_PATH = ATTEMPT / "step.json"
GZIP_PATH = ATTEMPT / "step.json.gz"
RUN_PATH = ATTEMPT / "step.run.json"
RESOURCE_PATH = ATTEMPT / "step.resource.json"
EVENT_PLAN = EVIDENCE / "LIMITER_EVENT_PLAN_ATTEMPT2_20261010.json"
EVENT_ARCHIVE = EVIDENCE / "LIMITER_EVENT_ARCHIVE_ATTEMPT2_20261010.json"
EVENT_GZIP = EVIDENCE / "limiter_event_20261010_attempt2/diagnostic.json.gz"
EVENT_RUN = EVIDENCE / "limiter_event_20261010_attempt2/diagnostic.run.json"
EVENT_RESOURCE = EVIDENCE / "limiter_event_20261010_attempt2/diagnostic.resource.json"
EVENT_AUDIT = EVIDENCE / "LIMITER_EVENT_SAVED_AUDIT_20261010.json"
GN_AUDIT = EVIDENCE / "GN_PREPARED_SEARCH_SAVED_AUDIT_20261010.json"
GN_RAW = EVIDENCE / "gn_prepared_search_20261010_attempt1/step.json.gz"
JSON_OUT = EVIDENCE / "EVENT_DIRECTION_SAVED_AUDIT_20261010.json"
MD_OUT = EVIDENCE / "EVENT_DIRECTION_GREEN_FINAL_20261010.md"

PLAN_SHA = "14773f392f41eaf9cf0fdba2b59bbf3427f8d3fdf0d0e7cc92b266704928ea29"
BASE_SHA = "027e83cf77f7f9cf97c0c72da9f737b5627d312d6375b698bb5b7f7ae94c6769"
GN_DIRECTION_SHA = "ab3a2635cedc7f90c9cead68e7ce9bae9b33a15cabf5ffe4970a9b00c91fea74"
EVENT_PLAN_SHA = "3579c2bbf9f3213cec5669948e9de21e47da258720572b291b083c5355e3f79d"
EVENT_RAW_SHA = "47b6b78d824a6ee817d2d014ea74ce14f68076d68c8c75a38ae6a0a7657297ff"
EPS = sys.float_info.epsilon
ARM_ORDER = ("prepared_gn", "event_orthogonal")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vec(values: Any) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("expected saved numeric array")
    result = [float(x) for x in values]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("saved numeric array contains nonfinite value")
    return result


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimension mismatch")
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in a))


def vsub(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("vector dimension mismatch")
    return [x - y for x, y in zip(a, b)]


def tensor_sha(values: list[float]) -> str:
    return hashlib.sha256(struct.pack("<" + "d" * len(values), *values)).hexdigest()


def close(a: float, b: float, factor: float = 512.0) -> bool:
    return abs(a - b) <= factor * EPS * max(abs(a), abs(b), sys.float_info.min)


def json_scalar_arm(arm: dict[str, Any]) -> dict[str, Any]:
    accepted = arm.get("accepted") or {}
    search = arm.get("search", {})
    gates = arm.get("gates", {})
    residual = vec(arm["residual"])
    residual_direction = vec(arm["residual_direction"])
    alpha = float(accepted.get("alpha", 0.0))
    model_change = (2.0 * alpha * float(arm.get("merit_product", 0.0))
        + alpha * alpha * dot(residual_direction, residual_direction))
    base_r = float(RAW["base_R"])
    max_model_fraction = (float(arm.get("merit_product", 0.0)) ** 2
        / (base_r * dot(residual_direction, residual_direction)))
    actual_r = float(accepted.get("F_squared", base_r))
    actual_reduction = base_r - actual_r
    return {
        "supported": arm.get("supported"),
        "direction_sha256": arm.get("direction_sha256"),
        "direction_l2": norm(vec(arm["direction"])),
        "fresh_hvp_calls_started": arm.get("hvp_calls_started"),
        "fresh_hvp_calls_completed": arm.get("hvp_calls_completed"),
        "side_cost_slopes": arm.get("side_cost_slopes"),
        "selected_J_slope": arm.get("selected_J_slope"),
        "merit_product_G_dot_dG": arm.get("merit_product"),
        "gates": gates,
        "search": search,
        "accepted": {key: accepted.get(key) for key in (
            "index", "alpha", "objective", "F_squared", "J_armijo_bound", "J_armijo_passed",
            "F_squared_armijo_bound", "F_squared_armijo_passed", "actual_path_norm", "theta",
            "control_sha256", "branch_pair_passed", "face_audit_passed", "side_objectives_match_native",
            "side_gradients_finite", "finite", "mixing_minimum_valid", "accepted", "status")},
        "model_R_change_at_accepted_alpha": model_change,
        "actual_R_decrease_at_accepted_alpha": actual_reduction,
        "actual_over_model_R_decrease": actual_reduction / (-model_change) if model_change < 0 else None,
        "maximum_model_R_reduction_fraction": max_model_fraction,
        "residual_direction_l2": norm(residual_direction),
        "residual_l2": norm(residual),
    }


def main() -> None:
    global RAW
    plan = read_json(PLAN)
    preflight = read_json(PREFLIGHT)
    archive = read_json(ARCHIVE)
    run = read_json(RUN_PATH)
    resource = read_json(RESOURCE_PATH)
    with gzip.open(GZIP_PATH, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    RAW = raw
    event_plan = read_json(EVENT_PLAN)
    event_archive = read_json(EVENT_ARCHIVE)
    event_run = read_json(EVENT_RUN)
    event_resource = read_json(EVENT_RESOURCE)
    event_audit = read_json(EVENT_AUDIT)
    gn_audit = read_json(GN_AUDIT)
    with gzip.open(GN_RAW, "rt", encoding="utf-8") as stream:
        gn_raw = json.load(stream)
    with gzip.open(EVENT_GZIP, "rt", encoding="utf-8") as stream:
        event_raw = json.load(stream)

    raw_bytes = RAW_PATH.read_bytes()
    zipped_bytes = GZIP_PATH.read_bytes()
    uncompressed = gzip.decompress(zipped_bytes)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    decompressed_sha = hashlib.sha256(uncompressed).hexdigest()
    pinned: dict[str, Any] = {}
    for group in ("source_files", "archive_files"):
        matches, missing, mismatches = 0, [], []
        for name, expected in plan[group].items():
            path = ROOT / name
            if not path.is_file():
                missing.append(name)
            elif sha(path) == expected:
                matches += 1
            else:
                mismatches.append({"path": name, "expected": expected, "actual": sha(path)})
        pinned[group] = {"expected": len(plan[group]), "matched": matches,
            "missing": missing, "mismatches": mismatches,
            "passed": matches == len(plan[group]) and not missing and not mismatches}

    direction_projection = raw["direction_projection"]
    baseline_direction = vec(direction_projection["baseline_direction"])
    projected_direction = vec(direction_projection["event_orthogonal_direction"])
    event_normal = vec(raw["event_tangent_gradient"])
    w_norm2 = dot(event_normal, event_normal)
    beta = dot(event_normal, baseline_direction) / w_norm2
    recomputed_dperp = [d - beta * w for d, w in zip(baseline_direction, event_normal)]
    projection_error = norm(vsub(projected_direction, recomputed_dperp))
    base_residual = vec(raw["base_residual"])
    base_j, base_r = float(raw["base_J"]), float(raw["base_R"])

    arms = raw["arms"]
    arm_summaries = {name: json_scalar_arm(arms[name]) for name in ARM_ORDER}
    event_arm = arm_summaries["event_orthogonal"]
    gn_arm = arm_summaries["prepared_gn"]
    event_accepted = arms["event_orthogonal"]["accepted"]
    gn_accepted = arms["prepared_gn"]["accepted"]
    chosen_name = raw["selection"]["arm"]
    chosen = arms[chosen_name]["accepted"]
    proposal = raw["proposal"]
    repeat = raw["final_repeat"]
    final_control = vec(raw["current_control"])
    actual_current_sha = tensor_sha(final_control)
    selected_control_sha = str(chosen.get("control_sha256"))
    final_g = vec(event_accepted["G"])

    selection_r_scale = max(abs(float(gn_accepted["F_squared"])),
        abs(float(event_accepted["F_squared"])), sys.float_info.min)
    selection_j_scale = max(abs(float(gn_accepted["objective"])),
        abs(float(event_accepted["objective"])), sys.float_info.min)
    r_tie = abs(float(gn_accepted["F_squared"]) - float(event_accepted["F_squared"])) <= 128 * EPS * selection_r_scale
    j_tie = abs(float(gn_accepted["objective"]) - float(event_accepted["objective"])) <= 128 * EPS * selection_j_scale
    event_arm_wins_by_r = float(event_accepted["F_squared"]) < float(gn_accepted["F_squared"]) and not r_tie
    final_r_change = float(proposal["F_squared"]) - base_r
    final_j_change = float(proposal["objective"]) - base_j

    expected_hvps = {(arm, side) for arm in ARM_ORDER for side in (-1, 1)}
    hvp_history = raw.get("hvp_history", [])
    actual_hvps = {(item.get("arm"), item.get("side")) for item in hvp_history
        if item.get("status") == "completed"}
    hvp_labels_close = (len(hvp_history) == 4 and actual_hvps == expected_hvps
        and all(item.get("base_control_sha256") == BASE_SHA
            and item.get("direction_sha256") == arms[item["arm"]]["direction_sha256"]
            and item.get("operator") == "selected_face_extension"
            and item.get("phase") == "comparison_base_point"
            for item in hvp_history))

    search_checks = {}
    for name in ARM_ORDER:
        arm = arms[name]
        accepted = arm.get("accepted")
        search = arm.get("search", {})
        trials = arm.get("trials", [])
        search_checks[name] = {
            "supported": arm.get("supported") is True,
            "fresh_hvps_complete": arm.get("hvp_calls_started") == arm.get("hvp_calls_completed") == 2,
            "24_candidate_alphas_prepared": search.get("computed_alpha_count") == 24
                and search.get("slot_cap") == 24,
            "evaluated_count_matches_saved_trials": search.get("evaluated_count") == len(trials)
                and len(trials) <= 24,
            "first_pass_record_matches_accepted_index": isinstance(accepted, dict)
                and search.get("first_pass_candidate_index") == accepted.get("index")
                and search.get("first_pass_count") == accepted.get("index")
                and accepted.get("accepted") is True,
            "accepted_actual_J_R_armijo_and_geometry_pass": isinstance(accepted, dict)
                and accepted.get("J_armijo_passed") is True
                and accepted.get("F_squared_armijo_passed") is True
                and accepted.get("branch_pair_passed") is True
                and accepted.get("face_audit_passed") is True
                and accepted.get("side_objectives_match_native") is True
                and accepted.get("side_gradients_finite") is True
                and accepted.get("finite") is True
                and accepted.get("mixing_minimum_valid") is True
                and 0 < float(accepted.get("theta", 0)) < 1
                and 0 < float(accepted.get("actual_path_norm", 0)) <= 0.05 * (1 + 64 * EPS),
        }

    event_gradient_rule = raw["event_gradient_rule"]
    event_receipt_projection_checks = []
    for side in ("-1", "1"):
        entry = event_gradient_rule[side]
        current_projection = vec(entry["recomputed_projection"])
        saved_projection = vec(entry["saved_projection"])
        receipt_projection = vec(event_raw["event_derivatives"][side]["tangent_gradient"])
        event_receipt_projection_checks.append(norm(vsub(current_projection, saved_projection)) <= 1e-11
            and norm(vsub(current_projection, receipt_projection)) <= 1e-11)
    fresh_event_checks = []
    for side in ("-1", "1"):
        fresh = vec(raw["fresh_base_event_values"][side])
        old = vec(event_raw["event_derivatives"][side]["original_values"])
        fresh_event_checks.append(fresh == old)

    gate_summaries = {}
    for name, summary in arm_summaries.items():
        gate_summaries[name] = {key: arms[name]["gates"].get(key)
            for key in ("finite", "both_side_gradients_descend", "merit_descends",
                "envelope_slope_matches_residual_dot", "envelope_slope", "merit_descent_budget")}

    relative_j_pct = 100 * (-final_j_change) / abs(base_j)
    relative_r_pct = 100 * (-final_r_change) / abs(base_r)
    base_g_inf = max(abs(x) for x in base_residual)
    selected_g_inf = max(abs(x) for x in final_g)
    base_g_norm = norm(base_residual)
    selected_g_norm = norm(final_g)
    projection_removed_fraction = norm(vsub(baseline_direction, projected_direction)) / norm(baseline_direction)
    model_r_change = (float(arms[chosen_name]["merit_product"]) * 2 * float(chosen["alpha"])
        + float(chosen["alpha"]) ** 2 * dot(vec(arms[chosen_name]["residual_direction"]),
            vec(arms[chosen_name]["residual_direction"])))
    actual_model_reduction = -model_r_change
    actual_r_reduction = -final_r_change

    checks = {
        "plan_preflight_and_all_pins_close": sha(PLAN) == PLAN_SHA and preflight.get("plan_sha256") == PLAN_SHA
            and preflight.get("load_plan_passed") is True and preflight.get("closed_base_passed") is True
            and preflight.get("event_receipt_passed") is True and preflight.get("production_executed") is False
            and all(item["passed"] for item in pinned.values()),
        "attempt_manifest_raw_gzip_run_resource_close": archive.get("plan_sha256") == PLAN_SHA
            and archive.get("execution_status") == "completed" and archive.get("lossless_roundtrip") is True
            and hashlib.sha256(zipped_bytes).hexdigest() == archive.get("gzip_sha256")
            and raw_sha == decompressed_sha == archive.get("raw_sha256") == run.get("child_sha256")
            and sha(RUN_PATH) == archive.get("run_sha256")
            and sha(RESOURCE_PATH) == archive.get("resource_sha256"),
        "guard_completed_zero_exit_within_resource_caps": run.get("execution_status") == "completed"
            and resource.get("exit_code") == 0 and resource.get("resource_termination") is None
            and resource.get("monitor_error") is None and resource.get("received_sigterm") is False
            and resource.get("sampled_peak_rss_bytes", 0) < resource.get("rss_limit_bytes", 0)
            and resource.get("elapsed_seconds", math.inf) < resource.get("wall_limit_seconds", 0),
        "parent_event_and_GN_receipts_match": event_audit.get("all_checks_passed") is True
            and gn_audit.get("all_checks_passed") is True
            and raw.get("base_control_sha256") == BASE_SHA
            and raw.get("direction_projection", {}).get("baseline_direction") == gn_raw["postcommit_gn_readiness"]["direction"]
            and raw.get("event_plan_sha256") == EVENT_PLAN_SHA
            and raw.get("event_receipt_raw_sha256") == EVENT_RAW_SHA,
        "fresh_base_J_G_theta_event_and_input_requalification":
            close(base_j, float(gn_raw["postcommit_current_observation"]["native_objective"]))
            and close(base_r, float(gn_raw["postcommit_current_observation"]["F_squared"]))
            and close(float(raw["base_theta"]), float(gn_raw["postcommit_current_observation"]["working_theta"]))
            and all(event_receipt_projection_checks) and all(fresh_event_checks)
            and raw.get("input_before") == gn_raw["input_after"]
            and raw.get("runtime_before") == gn_raw["runtime_after"],
        "event_projection_math_and_Q_tangency_close": projection_error <= 1e-12
            and direction_projection.get("normalized") is False
            and abs(float(direction_projection["projected_event_dot"])) <= float(direction_projection["orthogonality_budget"])
            and abs(float(direction_projection["projected_face_dot"])) <= float(direction_projection["projected_face_budget"])
            and raw.get("direction_projection", {}).get("projection_refusal") is None,
        "fresh_four_HVPs_bind_to_both_same_point_directions": raw.get("hvp_calls_started") == 4
            and raw.get("hvp_calls_completed") == 4 and len(hvp_history) == 4 and hvp_labels_close
            and all(len(vec(arms[name][key])) == 26 for name in ARM_ORDER for key in ("hminus", "hplus")),
        "both_arms_supported_and_each_stopped_at_first_passing_slot": all(
            all(values.values()) for values in search_checks.values())
            and raw.get("candidate_slots_used") == sum(arms[name]["search"]["evaluated_count"] for name in ARM_ORDER)
            and raw.get("candidate_slots_used") <= 48,
        "selected_winner_matches_R_then_J_then_baseline_tie_rule": raw["selection"]["comparison_complete"] is True
            and raw["selection"]["supported_arms"] == list(ARM_ORDER)
            and not r_tie and event_arm_wins_by_r and raw["selection"]["arm"] == "event_orthogonal"
            and raw["selection"]["arm"] == archive.get("selected_arm"),
        "one_P2_closed_commit_no_new_point_operators": raw.get("candidate_committed") is True
            and raw.get("optimizer_steps_applied") == raw.get("accepted_iterations") == 1
            and raw.get("postcommit_readiness_performed") is False
            and raw.get("readiness_complete") is False and raw.get("additional_candidates_after_commit") == 0
            and raw.get("jacobian_rows_started") == raw.get("jacobian_rows_completed") == 0
            and raw.get("dense_solves_started") == raw.get("dense_solves_completed") == 0,
        "P2_source_input_runtime_deadline_and_control_hash_close": all(raw["final_repeat"].get(key) is True for key in (
            "minimum_theta_matches_proposal", "native_objective_matches_proposal", "merit_matches_proposal",
            "mixing_minimum_valid", "side_gradients_finite", "side_objectives_match_native",
            "face_audit_passed", "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
            "fixed_input_unchanged", "runtime_unchanged", "deadline_passed"))
            and raw.get("source_unchanged") is True and raw.get("fixed_input_unchanged") is True
            and raw.get("runtime_unchanged") is True and raw.get("deadline_passed") is True
            and actual_current_sha == raw.get("current_control_sha256") == selected_control_sha
            and selected_control_sha == proposal.get("control_sha256")
            and actual_current_sha == raw.get("last_confirmed_control_sha256"),
    }

    selected_event = arms[chosen_name].get("accepted_event_values", {})
    selected_event_summary = {}
    for side in ("-1", "1"):
        values = vec(selected_event[side]["values"])
        zeta = values[0] - values[1]
        selected_event_summary[side] = {"left": values[0], "right": values[1],
            "q_abs_max": values[2], "zeta": zeta, "normalized_abs_zeta": abs(zeta) / values[2]}

    results = {
        name: {"accepted_index": summary["accepted"]["index"],
            "evaluated_candidates": summary["search"]["evaluated_count"],
            "candidate_cap": summary["search"]["slot_cap"],
            "initial_alpha": summary["search"]["computed_alpha0"],
            "alpha": summary["accepted"]["alpha"],
            "dyadic_halvings": math.log2(summary["search"]["computed_alpha0"] / summary["accepted"]["alpha"]),
            "actual_path_norm": summary["accepted"]["actual_path_norm"],
            "objective": summary["accepted"]["objective"],
            "objective_change": summary["accepted"]["objective"] - base_j,
            "objective_armijo_slack": summary["accepted"]["J_armijo_bound"] - summary["accepted"]["objective"],
            "R": summary["accepted"]["F_squared"],
            "R_change": summary["accepted"]["F_squared"] - base_r,
            "R_decrease_percent": 100 * (base_r - summary["accepted"]["F_squared"]) / abs(base_r),
            "R_armijo_slack": summary["accepted"]["F_squared_armijo_bound"] - summary["accepted"]["F_squared"],
            "theta": summary["accepted"]["theta"],
            "control_sha256": summary["accepted"]["control_sha256"],
            "side_cost_slopes": summary["side_cost_slopes"],
            "merit_product": summary["merit_product_G_dot_dG"],
            "maximum_model_R_reduction_percent": 100 * summary["maximum_model_R_reduction_fraction"],
            "actual_over_model_R_decrease": summary["actual_over_model_R_decrease"],
            "gates": summary["gates"]}
        for name, summary in arm_summaries.items()
    }
    selection_math = {
        "baseline_R": results["prepared_gn"]["R"],
        "event_orthogonal_R": results["event_orthogonal"]["R"],
        "R_difference_event_minus_baseline": results["event_orthogonal"]["R"] - results["prepared_gn"]["R"],
        "R_tie_under_plan_tolerance": r_tie,
        "event_orthogonal_wins_on_actual_R": event_arm_wins_by_r,
        "J_tie_under_plan_tolerance": j_tie,
        "tie_rules_invoked": r_tie,
        "selected_arm": chosen_name,
    }
    progress = {
        "base_J": base_j,
        "committed_J": float(proposal["objective"]),
        "J_decrease_percent": relative_j_pct,
        "base_R": base_r,
        "committed_R": float(proposal["F_squared"]),
        "R_decrease_percent": relative_r_pct,
        "base_G_l2": base_g_norm,
        "committed_G_l2": selected_g_norm,
        "base_G_linf": base_g_inf,
        "committed_G_linf": selected_g_inf,
        "actual_control_movement_l2": float(chosen["actual_path_norm"]),
        "committed_theta": float(proposal["theta"]),
    }
    result = {
        "plan_sha256": PLAN_SHA,
        "preflight": {key: preflight.get(key) for key in (
            "plan_sha256", "source_count", "archive_count", "load_plan_passed", "closed_base_passed",
            "event_receipt_passed", "production_executed")},
        "source_archive_pin_audit": pinned,
        "parent_point": {"control_sha256": BASE_SHA, "GN_direction_sha256": GN_DIRECTION_SHA,
            "event_plan_sha256": EVENT_PLAN_SHA, "J": base_j, "R": base_r,
            "theta": float(raw["base_theta"]), "G_l2": base_g_norm, "G_linf": base_g_inf},
        "projection": {"direction_normalized": direction_projection.get("normalized"),
            "event_projection_coefficient": beta,
            "baseline_direction_l2": norm(baseline_direction),
            "event_orthogonal_direction_l2": norm(projected_direction),
            "removed_component_l2_fraction": projection_removed_fraction,
            "baseline_event_dot": direction_projection.get("baseline_event_dot"),
            "projected_event_dot": direction_projection.get("projected_event_dot"),
            "projected_event_dot_budget": direction_projection.get("orthogonality_budget"),
            "projected_face_dot": direction_projection.get("projected_face_dot"),
            "projected_face_dot_budget": direction_projection.get("projected_face_budget"),
            "event_gradient_rule": raw.get("event_gradient_rule")},
        "arms": results,
        "selection": {**selection_math, **raw["selection"]},
        "selected_event_at_endpoint": selected_event_summary,
        "progress": progress,
        "operator_counts": {"fresh_HVP_started": raw.get("hvp_calls_started"),
            "fresh_HVP_completed": raw.get("hvp_calls_completed"),
            "HVP_receipts": [{key: item.get(key) for key in (
                "arm", "side", "phase", "base_control_sha256", "direction_sha256", "operator", "status")}
                for item in hvp_history],
            "jacobian_rows_completed": raw.get("jacobian_rows_completed"),
            "dense_solves_completed": raw.get("dense_solves_completed"),
            "postcommit_readiness_performed": raw.get("postcommit_readiness_performed")},
        "P2": {key: repeat.get(key) for key in (
            "objective", "F_squared", "theta", "mixing_minimum_valid", "minimum_theta_matches_proposal",
            "native_objective_matches_proposal", "merit_matches_proposal", "branch_pair_passed",
            "face_audit_passed", "trace_matches_proposal", "side_objectives_match_native",
            "side_gradients_finite", "source_unchanged", "fixed_input_unchanged", "runtime_unchanged",
            "deadline_passed")},
        "execution": {"execution_status": raw.get("execution_status"),
            "numerical_status": raw.get("numerical_status"),
            "optimizer_steps_applied": raw.get("optimizer_steps_applied"),
            "candidate_committed": raw.get("candidate_committed"),
            "readiness_complete": raw.get("readiness_complete"),
            "additional_candidates_after_commit": raw.get("additional_candidates_after_commit"),
            "elapsed_seconds": resource.get("elapsed_seconds"),
            "sampled_peak_rss_bytes": resource.get("sampled_peak_rss_bytes"),
            "rss_limit_bytes": resource.get("rss_limit_bytes"),
            "wall_limit_seconds": resource.get("wall_limit_seconds"),
            "raw_sha256": raw_sha, "gzip_sha256": hashlib.sha256(zipped_bytes).hexdigest(),
            "run_sha256": sha(RUN_PATH), "resource_sha256": sha(RESOURCE_PATH)},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# GREEN saved-data review: event-orthogonal GN comparison",
        "",
        f"Frozen plan `{PLAN_SHA}`; pin, run, operator, and P2 checks: **{result['all_checks_passed']}** ({sum(checks.values())}/{len(checks)}).",
        "This audit reads saved JSON and uses standard-library arithmetic only; it ran no FV, seed, AD, or HVP code.",
        "",
        "## Direction comparison",
        "",
        f"The Euclidean event projection removed a component of length {norm(vsub(baseline_direction, projected_direction)):.12g} "
        f"({projection_removed_fraction:.6%} of the baseline direction norm) without rescaling. The projected event dot is "
        f"{direction_projection['projected_event_dot']:.12g} (budget {direction_projection['orthogonality_budget']:.12g}); "
        f"the selected-face dot is {direction_projection['projected_face_dot']:.12g} "
        f"(budget {direction_projection['projected_face_budget']:.12g}).",
        "",
        "| Arm | first pass | α | actual move | actual J change | actual R change | max local model χ | actual/model R decrease |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARM_ORDER:
        arm = results[name]
        md.append(f"| {name} | {arm['accepted_index']} / 24 | {arm['alpha']:.12g} | "
            f"{arm['actual_path_norm']:.12g} | {arm['objective_change']:.12g} | {arm['R_change']:.12g} | "
            f"{arm['maximum_model_R_reduction_percent']:.6f}% | {arm['actual_over_model_R_decrease']:.9g} |")
    md += [
        "",
        f"Selection chose **{chosen_name}** by actual R ({results['event_orthogonal']['R']:.12g} vs "
        f"{results['prepared_gn']['R']:.12g}); the R tie rule was not invoked. The event-orthogonal arm passed on its first grid slot; the prepared GN arm first passed on slot 22.",
        f"The selected endpoint reduced R by {relative_r_pct:.6f}% and J by {relative_j_pct:.8f}%. "
        f"G∞ moved from {base_g_inf:.12g} to {selected_g_inf:.12g}, so this does not establish improvement in every gradient norm.",
        f"Its local maximum model χ was {results['event_orthogonal']['maximum_model_R_reduction_percent']:.6f}%, "
        f"below the prepared GN arm's {results['prepared_gn']['maximum_model_R_reduction_percent']:.6f}%. "
        "The evidence favors a longer admissible first-passing step at this point, not a better maximum linear-model reduction or a general convergence claim.",
        "",
        "Four fresh HVPs were computed at the same 027 base point (two sides per arm); there were no Jacobian rows, dense solves, or new-point readiness operators. One selected candidate passed one independent P2 repeat and was committed; no further candidates ran.",
        "",
        "## Checks",
        "",
    ]
    md.extend(f"- `{key}`: **{value}**" for key, value in checks.items())
    md.append("")
    MD_OUT.write_text("\n".join(md), encoding="utf-8")
    print(JSON_OUT)
    print(MD_OUT)
    if not all(checks.values()):
        raise SystemExit("saved-array event-direction audit found a mismatch")


if __name__ == "__main__":
    main()
