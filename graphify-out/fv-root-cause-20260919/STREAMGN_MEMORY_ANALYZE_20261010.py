"""Audit the 2-GiB memory-resume attempt using saved arrays and receipts only.

The analyzer validates the earlier two-step prefix, admits only its final P2
closure at f2ca…, and injects that base into the NumPy-only GN resume collector.
No production, FV, seed, autodiff, HVP, or optimizer code is imported or run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]
if str(EVIDENCE) not in sys.path:
    sys.path.insert(0, str(EVIDENCE))
import GNRESUME_ANALYZE_20261010 as resume
import STREAMGN_ANALYZE_20261010 as stream

PLAN = EVIDENCE / "TANGENT_COUPLED_GN_MEMORY_RESUME_PLAN_20261010.json"
PARENT_PLAN = EVIDENCE / "TANGENT_COUPLED_GN_PARTIAL_RESUME_PLAN_20261010.json"
PARENT_PLAN_SHA = "529110d25ca28ac85ab3ce3fa2fd23a8b2a8ba0122c2f184ab5e26a3c20b4f17"
PARENT_ARCHIVE_MANIFEST = EVIDENCE / "STREAMGN_ARCHIVE_20261010.json"
PARENT_ARCHIVE = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt2/step.json.gz"
PARENT_RAW = PARENT_ARCHIVE.with_suffix("")
PARENT_RUN = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt2/step.run.json"
PARENT_RESOURCE = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt2/step.resource.json"
PARENT_RAW_SHA = "a17272f51451e99b20e03d06122b3cfc79d7415e50c67ea16cb6c205e8f43497"
PARENT_GZIP_SHA = "24c84eab4b4d4f108e65581e02b2bd44baed261dacc2113c2bdb670f0d2ca3e7"
PARENT_RUN_SHA = "03b25b9105ba0133e9ac11350e732b47e4c61149db52d0cac506b2b3cb85676a"
PARENT_RESOURCE_SHA = "b4d6faf9d2bdaf00f5d8fc29a9c9564fb95a2f287e5a7cd30bed2d1bef1c1d4a"
PARENT_MANIFEST_SHA = "4085ca18c607193119bb66427f5161debaf7f9c2481e549c84f6f36247c39953"
ATTEMPT = EVIDENCE / "tangent_coupled_gn_memory_resume_20261010_attempt1"
OUT = EVIDENCE / "STREAMGN_MEMORY_RESULT_20261010.json"
PREFLIGHT_MANIFEST = EVIDENCE / "STREAMGN_PREFLIGHT_ARCHIVE_20261010.json"
BASE_SHA = "f2ca572936a8bf646c920c542b3780ff65a0f94c0d715aa61ba00a5cec57313d"
BASE_THETA = 0.4830637745604483
BASE_OBJECTIVE = 0.06119270839098437
BASE_F2 = 0.004479439946247095
PLAN_SHA = "7c93aa28219c73e7a437feaac7341887da8fc6fc0b9f0b8e4633ad4424224c67"
ANCHOR_PROVENANCE = {
    "input": "derived_from_pr273_input_after_control_sha_and_saved_control_flow_diagnostics_per_fixed_input_contract",
    "runtime": "reused_from_pr273_runtime_after",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return resume.read_json(path)


def _load_two_step_base(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate the interrupted parent, then return only its two-step P2 endpoint."""
    pins = ((PARENT_PLAN, PARENT_PLAN_SHA), (PARENT_ARCHIVE_MANIFEST, PARENT_MANIFEST_SHA),
        (PARENT_ARCHIVE, PARENT_GZIP_SHA), (PARENT_RUN, PARENT_RUN_SHA),
        (PARENT_RESOURCE, PARENT_RESOURCE_SHA))
    if any(_sha(path) != digest for path, digest in pins):
        raise ValueError("preceding 2-GiB resume archive pin mismatch")
    parent_plan = _read_json(PARENT_PLAN)
    manifest = _read_json(PARENT_ARCHIVE_MANIFEST)
    compressed = PARENT_ARCHIVE.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    child = json.loads(raw_bytes)
    parent_run, resource = _read_json(PARENT_RUN), _read_json(PARENT_RESOURCE)
    if (raw_sha != PARENT_RAW_SHA or manifest.get("raw_sha256") != PARENT_RAW_SHA
            or manifest.get("gzip_sha256") != PARENT_GZIP_SHA
            or manifest.get("run_sha256") != PARENT_RUN_SHA
            or manifest.get("resource_sha256") != PARENT_RESOURCE_SHA
            or manifest.get("lossless_roundtrip") is not True
            or manifest.get("confirmed_prefix_steps") != 2
            or manifest.get("terminal_execution_closed") is not False
            or parent_run.get("child_sha256") != PARENT_RAW_SHA
            or parent_run.get("resource") != resource
            or resource.get("execution_status") not in (None, "rss_limit")
            or resource.get("resource_termination") != "rss_limit"
            or resource.get("exit_code") != -15
            or resource.get("sampled_peak_rss_bytes") != 1091633152
            or child.get("plan_sha256") != PARENT_PLAN_SHA
            or child.get("plan_hashes", {}).get("source_files") != parent_plan["source_files"]
            or child.get("plan_hashes", {}).get("archive_files") != parent_plan["archive_files"]
            or child.get("source_after") is not None
            or child.get("input_after") is not None or child.get("runtime_after") is not None
            or child.get("accepted_iterations") != 2 or child.get("optimizer_steps_applied") != 2
            or child.get("jacobian_rows_started") != 72 or child.get("jacobian_rows_completed") != 72
            or child.get("hvp_calls_started") != 4 or child.get("hvp_calls_completed") != 4
            or child.get("dense_solves_started") != 3 or child.get("dense_solves_completed") != 3):
        raise ValueError("interrupted parent receipts do not describe the reviewed two-step prefix")

    # Reuse the existing saved-array collector to close both historical points.
    # Parent attempt 2 starts from PR274's saved 7cec… closure, so seed its
    # two-point audit with that prior partial-prefix base, not PR273's 403590… base.
    prefix_base = stream._load_partial_prefix(parent_plan, ROOT)
    original_loader = resume.load_pr273_base
    resume.load_pr273_base = lambda _plan, _root: prefix_base
    try:
        parent_result = resume.collect(PARENT_RAW, PARENT_RUN, PARENT_RESOURCE,
            PARENT_PLAN, PARENT_ARCHIVE_MANIFEST)
    finally:
        resume.load_pr273_base = original_loader
    tail_audit = stream._tail_dense_audit(child, parent_plan)
    closure = child.get("last_confirmed_closure")
    if (parent_result.get("confirmed_prefix", {}).get("accepted_steps") != 2
            or len(parent_result.get("points", [])) != 2
            or not all(point.get("checks_passed") is True for point in parent_result["points"])
            or parent_result.get("source_checks", {}).get("source_before_matches") is not True
            or parent_result.get("source_checks", {}).get("all_current_plan_pins_match") is not True
            or parent_result.get("parent_checks", {}).get("confirmed_prefix_matches_child_checkpoint") is not True
            or tail_audit.get("tail_linear_solve_audit_closes") is not True
            or not isinstance(closure, dict)
            or closure.get("control") != child.get("current_control")
            or closure.get("theta") != BASE_THETA
            or closure.get("objective") != BASE_OBJECTIVE
            or closure.get("F_squared") != BASE_F2):
        raise ValueError("saved two-step parent prefix or uncommitted tail audit failed")
    control = resume.np.asarray(closure["control"], dtype=resume.np.float64)
    if resume.gn.vsha(control) != BASE_SHA:
        raise ValueError("two-step terminal P2 control hash mismatch")
    minimum = resume.gn.theta_minimum(closure["side_gradients"]["-1"], closure["side_gradients"]["1"])
    _, _, face_q = resume.gn.face_geometry(control, plan["face"])
    mixture = ((1.0-minimum["theta_star"])*resume.np.asarray(closure["side_gradients"]["-1"])
        + minimum["theta_star"]*resume.np.asarray(closure["side_gradients"]["1"]))
    f_squared = float(mixture@mixture + (face_q/resume.gn.FACE_SCALE)**2)
    if (not minimum["strict_interior"] or not resume.near(minimum["theta_star"], BASE_THETA)
            or not resume.near(f_squared, BASE_F2)):
        raise ValueError("saved terminal P2 gradients/minimum do not recompute")

    source_input = prefix_base["anchor"]["input"]
    expected_input = dict(source_input)
    expected_input["control_sha256"] = BASE_SHA
    flow = resume.np.tanh(control[20:25])
    expected_input["shifted_flow_fractions"] = flow.tolist()
    changed_fields = {key for key in set(source_input) | set(expected_input)
        if source_input.get(key) != expected_input.get(key)}
    if (changed_fields != {"control_sha256", "shifted_flow_fractions"}
            or plan.get("anchor_provenance") != ANCHOR_PROVENANCE
            or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("initial_theta") != BASE_THETA):
        raise ValueError("2-GiB restart anchor is not derived from the verified two-step endpoint")
    failed_preflight = stream._failed_preflight(plan)
    return {"endpoint": closure, "control": control, "control_sha256": BASE_SHA,
        "theta": BASE_THETA, "objective": BASE_OBJECTIVE, "F_squared": f_squared,
        "raw_sha256": PARENT_RAW_SHA, "parent_child": child,
        "parent_analysis": parent_result, "tail_dense_audit": tail_audit,
        "producer_receipt_hashes": {"plan_sha256": PARENT_PLAN_SHA, "raw_sha256": PARENT_RAW_SHA,
            "gzip_sha256": PARENT_GZIP_SHA, "run_sha256": PARENT_RUN_SHA,
            "resource_sha256": PARENT_RESOURCE_SHA, "archive_manifest_sha256": PARENT_MANIFEST_SHA},
        "superseded_preflight": failed_preflight,
        "anchor": {"input": expected_input, "runtime": prefix_base["anchor"]["runtime"],
            "provenance": ANCHOR_PROVENANCE, "changed_fields": sorted(changed_fields)},
        "receipt_hashes": {"plan_sha256": PARENT_PLAN_SHA, "raw_sha256": PARENT_RAW_SHA,
            "gzip_sha256": PARENT_GZIP_SHA, "run_sha256": PARENT_RUN_SHA,
            "resource_sha256": PARENT_RESOURCE_SHA,
            "archive_manifest_sha256": PARENT_MANIFEST_SHA}}


def _input_anchor_matches(actual: Any, expected: dict[str, Any]) -> bool | None:
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
    budget = 8.0*resume.np.finfo(resume.np.float64).eps*resume.np.maximum(
        scale, resume.np.finfo(resume.np.float64).tiny)
    return bool((resume.np.abs(seen-target) <= budget).all())


def _input_pair_checks(child: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    before, after = child.get("input_before"), child.get("input_after")
    final_control = resume.np.asarray(child.get("last_confirmed_control", []), dtype=resume.np.float64)
    final_sha = resume.gn.vsha(final_control) if final_control.size else None
    expected_after = dict(anchor["input"])
    if final_sha is not None:
        expected_after["control_sha256"] = final_sha
        expected_after["shifted_flow_fractions"] = resume.np.tanh(final_control[20:25]).tolist()
    before_after_changed = ({key for key in set(before or {}) | set(after or {})
        if (before or {}).get(key) != (after or {}).get(key)} if isinstance(before, dict)
        and isinstance(after, dict) else None)
    return {"before_matches_derived_base_anchor": _input_anchor_matches(before, anchor["input"]),
        "after_matches_last_confirmed_control_anchor": _input_anchor_matches(after, expected_after),
        "after_control_sha256": (after or {}).get("control_sha256") if isinstance(after, dict) else None,
        "last_confirmed_control_sha256": child.get("last_confirmed_control_sha256"),
        "before_after_changed_fields": sorted(before_after_changed) if before_after_changed is not None else None,
        "only_control_dependent_fields_change": before_after_changed == {
            "control_sha256", "shifted_flow_fractions"},
        "fixed_input_unchanged_flag": child.get("fixed_input_unchanged")}


def _brief_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for point in result.get("points", []):
        endpoint, model = point.get("endpoint", {}), point.get("local_model", {})
        actual = endpoint.get("predicted_vs_actual", {})
        summaries.append({"index": point.get("index"),
            "base_control_sha256": point.get("base_control_sha256"),
            "rows": point.get("row_count"), "rows_complete": point.get("rows_complete"),
            "HVPs": point.get("hvp_count"), "HVP_provenance": point.get("hvp_provenance"),
            "dense_solves": point.get("dense_solve_count"),
            "dense_solve_receipt": point.get("dense_solve_receipt"),
            "working_theta_star": point.get("working_theta_star"),
            "accepted": point.get("accepted"),
            "candidate_count": point.get("candidate_count"),
            "candidate_count_within_cap": point.get("candidate_count_within_cap"),
            "refusal": point.get("refusal"),
            "endpoint_theta_star": endpoint.get("theta_star"),
            "native_J": endpoint.get("native_J"), "F_squared": endpoint.get("F_squared"),
            "minimum_gradient_l2": endpoint.get("minimum_gradient_l2"),
            "minimum_gradient_linf": endpoint.get("minimum_gradient_linf"),
            "minimum_gradient_blocks": endpoint.get("minimum_gradient_blocks"),
            "model_chi_cos_squared": actual.get("model_chi_cos_squared"),
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
    return summaries


def _unaccepted_candidate_diagnostics(child: dict[str, Any]) -> dict[str, Any]:
    """Summarize the newest uncommitted search without turning it into a step."""
    progress = child.get("comparison_progress")
    source = "comparison_progress" if isinstance(progress, dict) else "iterations"
    item: dict[str, Any] | None = None
    if isinstance(progress, dict):
        item = progress
    else:
        unaccepted = [step for step in child.get("iterations", [])
            if step.get("accepted") is not True]
        if unaccepted:
            item = unaccepted[-1]
    if not isinstance(item, dict):
        return {"present": False, "candidate_slots": 0,
            "accepted_endpoint": False, "status": "no_uncommitted_candidate_record"}

    arms = [arm for arm in item.get("model_comparisons", [])
        if arm.get("name") == "robust_gn_coupled"]
    trials = [trial for arm in arms for trial in arm.get("trials", [])]
    gate_keys = ("J_armijo_passed", "F_squared_armijo_passed", "face_audit_passed",
        "branch_pair_passed", "side_objectives_match_native", "side_gradients_finite",
        "mixing_minimum_valid")
    failed_counts = {key: sum(trial.get(key) is not True for trial in trials) for key in gate_keys}
    only_f2 = bool(trials) and all(
        trial.get("J_armijo_passed") is True
        and trial.get("F_squared_armijo_passed") is False
        and all(trial.get(key) is True for key in gate_keys if key != "F_squared_armijo_passed")
        for trial in trials)
    status_counts: dict[str, int] = {}
    for trial in trials:
        status = str(trial.get("status", "missing_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    matching_record = next((step for step in child.get("iterations", [])
        if step.get("index") == item.get("index")
        and step.get("base_control_sha256") == item.get("base_control_sha256")), {})
    return {"present": True, "source": source,
        "index": item.get("index"),
        "base_control_sha256": item.get("base_control_sha256"),
        "carried_theta": item.get("theta"),
        "accepted_endpoint": item.get("accepted") is True,
        "candidate_slots": len(trials),
        "accepted_candidates": sum(trial.get("accepted") is True for trial in trials),
        "trial_status_counts": status_counts,
        "gate_not_true_counts": failed_counts,
        "all_candidates_passed_J_and_face_branch": all(
            trial.get("J_armijo_passed") is True
            and trial.get("face_audit_passed") is True
            and trial.get("branch_pair_passed") is True for trial in trials),
        "all_candidates_failed_only_F_squared_Armijo": only_f2,
        "refusal": item.get("refusal") or matching_record.get("refusal"),
        "committed": item.get("accepted") is True and item.get("committed_control") is not None}


def _collect_with_analysis_none_guard(child_path: Path, run_path: Path, resource_path: Path,
        plan_path: Path, archive_manifest_path: Path | None,
        base: dict[str, Any]) -> dict[str, Any]:
    """Temporarily guard the saved analyzer's absent-accepted-alpha division.

    An unaccepted point has `accepted_alpha=None`; the inherited NumPy collector
    divides it by the first alpha while formatting diagnostics. Patch that
    analysis-only expression in memory, leaving the child, collector source,
    production sources, hashes, and all point receipts untouched.
    """
    original_collect = resume.collect
    original_loader = resume.load_pr273_base
    source = inspect.getsource(original_collect)
    old = 'model_vs_actual["accepted_alpha"]/first_alpha if first_alpha else None'
    new = ('model_vs_actual["accepted_alpha"]/first_alpha '
        'if model_vs_actual["accepted_alpha"] is not None and first_alpha else None')
    if source.count(old) != 1:
        raise ValueError("saved analyzer does not contain the expected missing-alpha expression")
    namespace = resume.__dict__
    exec(source.replace(old, new), namespace)
    patched_collect = namespace["collect"]

    resume.load_pr273_base = lambda _plan, _root: base
    resume.collect = patched_collect
    try:
        return resume.collect(child_path, run_path, resource_path, plan_path, archive_manifest_path)
    finally:
        resume.load_pr273_base = original_loader
        resume.collect = original_collect


def collect(child_path: Path, run_path: Path, resource_path: Path,
        plan_path: Path, archive_manifest_path: Path | None = None) -> dict[str, Any]:
    if _sha(plan_path) != PLAN_SHA:
        raise ValueError("memory-resume plan digest differs from frozen 2-GiB plan")
    plan = _read_json(plan_path)
    base = _load_two_step_base(plan)
    result = _collect_with_analysis_none_guard(child_path, run_path, resource_path,
        plan_path, archive_manifest_path, base)

    child = _read_json(child_path)
    anchor = base["anchor"]
    progress = child.get("comparison_progress")
    accepted = int(result["confirmed_prefix"]["accepted_steps"])
    current_sha = child.get("last_confirmed_control_sha256")
    progress_matches = (isinstance(progress, dict)
        and progress.get("base_control_sha256") == current_sha
        and progress.get("index") == accepted)
    fields_present = all(child.get(key) is not None for key in
        ("input_before", "input_after", "runtime", "runtime_after"))
    input_pair = _input_pair_checks(child, anchor)
    runtime = child.get("runtime")
    runtime_after = child.get("runtime_after")
    runtime_same = runtime == runtime_after if runtime is not None and runtime_after is not None else None
    source_same = child.get("source_before") == child.get("source_after") \
        if isinstance(child.get("source_after"), dict) else None
    result["base"] = {"source": "two independently P2-closed memory-resume prefix steps",
        "control_sha256": base["control_sha256"], "theta_star": base["theta"],
        "native_J": base["objective"], "F_squared": base["F_squared"],
        "parent_raw_sha256": base["raw_sha256"],
        "parent_receipt_hashes": base["receipt_hashes"],
        "parent_accepted_steps": 2, "original_1GiB_plan_remains_partial": True}
    result["resume_prefix"] = {"this_plan_maximum_new_commits": int(plan["policy"]["max_accepted_iterations"]),
        "original_commits_not_counted_in_new_plan_cap": True,
        "prior_uncommitted_rows_reused": False, "prior_uncommitted_dense_audit_reused": False,
        "prior_third_hvps": 0, "prior_third_candidates": 0,
        "anchor_provenance": anchor["provenance"],
        "derived_input_fields_changed": anchor["changed_fields"],
        "expected_shifted_flow_source": "NumPy tanh of saved f2ca… control components 20:25; ULP tolerance accounts for Torch/NumPy"}
    result["prior_attempt_tail_audit"] = base["tail_dense_audit"]
    result["superseded_preflight"] = base["superseded_preflight"]
    result["resume_anchor_checks"] = {
        "provenance_matches_plan": child.get("base_anchor_provenance") == anchor["provenance"],
        "input_pair": input_pair,
        "derived_runtime_matches_new_before": runtime == anchor["runtime"] if runtime is not None else None,
        "new_runtime_matches_after": runtime_same,
        "input_runtime_fields_present": fields_present,
        "new_source_before_after_matches": source_same,
        "fixed_input_unchanged_flag": child.get("fixed_input_unchanged"),
        "runtime_unchanged_flag": child.get("runtime_unchanged"),
        "original_1GiB_attempt_terminal_closed": False}
    result["global_input_runtime_identity"] = {
        "status": "recorded",
        "present": {key: key in child for key in ("input_before", "input_after", "runtime", "runtime_after")},
        "base_input_matches_derived_anchor": input_pair["before_matches_derived_base_anchor"],
        "terminal_input_matches_control": input_pair["after_matches_last_confirmed_control_anchor"],
        "fixed_input_unchanged": input_pair["only_control_dependent_fields_change"]
            and child.get("fixed_input_unchanged") is True,
        "runtime_unchanged": runtime_same is True and child.get("runtime_unchanged") is True,
        "source_unchanged": source_same is True and child.get("source_unchanged") is True}
    result["unaccepted_candidate_diagnostics"] = _unaccepted_candidate_diagnostics(child)
    result["point_summaries"] = _brief_metrics(result)
    result["accounting"]["candidate_slots_accepted_iterations"] = sum(
        point.get("candidate_count", 0) for point in result.get("points", [])
        if point.get("accepted") is True)
    result["accounting"]["unaccepted_candidate_slots"] = result[
        "unaccepted_candidate_diagnostics"].get("candidate_slots", 0)
    result["accounting"]["candidate_slots_including_unaccepted"] = result[
        "accounting"].get("candidate_slots_total", 0)
    result["accounting"]["unaccepted_point_model_count"] = int(
        result["unaccepted_candidate_diagnostics"].get("present") is True)
    for point in result["point_summaries"]:
        if (point.get("accepted") is False and point.get("base_control_sha256")
                == result["unaccepted_candidate_diagnostics"].get("base_control_sha256")):
            point["refusal"] = result["unaccepted_candidate_diagnostics"].get("refusal")
    result["checkpoint_progress"] = {"present": isinstance(progress, dict),
        "base_control_sha256": progress.get("base_control_sha256") if isinstance(progress, dict) else None,
        "index": progress.get("index") if isinstance(progress, dict) else None,
        "matches_uncommitted_successor": progress_matches,
        "latest_state_valid": not isinstance(progress, dict) or progress_matches,
        "absent_while_point_uncommitted_permitted": not isinstance(progress, dict)
            and child.get("phase") == "running",
        "absent_at_completed_terminal": child.get("phase") == "finished"
            and child.get("execution_status") == "completed" and not isinstance(progress, dict)}
    result["resource_policy"] = {"rss_limit_bytes": plan["policy"]["rss_bytes"],
        "maximum_new_commits": plan["policy"]["max_accepted_iterations"],
        "row_budget": plan["policy"]["jacobian_row_vjp_calls"],
        "HVP_budget": plan["policy"]["hvp_calls"],
        "dense_budget": plan["policy"]["dense_solves"],
        "predecessor_peak_rss_bytes": 1091633152,
        "predecessor_limit_bytes": 1073741824}
    result["memory_and_claim_limits"] = {"RSS_savings_guaranteed": False,
        "resource_guard_closed": result["accounting"].get("resource_closed"),
        "solver_convergence_established": False,
        "minimum_root_response_adjoint_reanalysis_forecast_claims": False}
    result["saved_analysis_adapter"] = {
        "used_in_memory_guard_for_missing_accepted_alpha": True,
        "inherited_collector_source_modified": False,
        "raw_child_projection_used": False,
        "original_raw_hash_and_counters_preserved": True}
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
    if args.archive_manifest is not None and not args.archive_manifest.exists():
        raise FileNotFoundError(args.archive_manifest)
    result = collect(args.child, args.run, args.resource, args.plan, args.archive_manifest)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
        allow_nan=False, default=lambda value: value.item()) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "execution_status", "numerical_status", "base", "confirmed_prefix", "point_summaries",
        "unaccepted_candidate_diagnostics", "accounting", "prior_attempt_tail_audit",
        "resume_anchor_checks", "source_checks",
        "parent_checks", "resource_policy", "memory_and_claim_limits", "terminal_closed")},
        indent=2, sort_keys=True, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
