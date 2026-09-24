"""Read-only, profile-stratified accounting of four archived FV research cohorts.

This inventory does not run a numerical model or estimate general radar
eligibility. Each cohort retains its own predeclared-case denominator.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "graphify-out/fv-root-cause-20260919"
CASE_IDS = {
    "fv86_full_support": ("seed_a", "seed_b"),
    "point_missing_fixed_control": ("three_time_missing",),
    "long_regular_forward": ("zero_flow", "nonzero_flow"),
    "partial_two_hole": ("masked_case",),
}
AXES = (
    "execution", "forecast", "stationarity", "branch", "local_response_eligibility", "response",
    "response_validation", "physical_validation", "resource",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict[str, Any]:
    value = json.loads((HERE / name).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return value


def _require_manifest(name: str, artifact: str) -> dict[str, Any]:
    manifest = _load(name)
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError(f"{name} has no source/evidence hashes")
    expected = hashes.get(f"graphify-out/fv-root-cause-20260919/{artifact}")
    if expected != _hash(HERE / artifact):
        raise ValueError(f"{artifact} differs from its manifest")
    for relative, fingerprint in hashes.items():
        path = (ROOT / relative).resolve()
        if (not path.is_relative_to(ROOT) or not path.is_file()
                or _hash(path) != fingerprint):
            raise ValueError(f"manifest source/evidence mismatch: {relative}")
    return manifest


def _fv86_rows() -> list[dict[str, Any]]:
    summary = _load("fv86_execution_status_summary.json")
    if (summary.get("declared_case_count") != 2
            or summary.get("eligible_count") != 2
            or summary.get("same_problem_and_source") is not True
            or summary.get("nonlinear_reanalyses") != 0):
        raise ValueError("FV86 execution cohort is not the pinned two-start scope")
    artifact_hashes = summary.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("FV86 summary has no archived artifact hashes")
    for name, expected in artifact_hashes.items():
        if _hash(HERE / name) != expected:
            raise ValueError(f"FV86 archived artifact changed: {name}")
    cases = summary.get("cases")
    if not isinstance(cases, list) or tuple(row.get("mode") for row in cases) != CASE_IDS["fv86_full_support"]:
        raise ValueError("FV86 case list differs from the declared two starts")
    rows = []
    for row in cases:
        mode = row["mode"]
        raw = _load(f"fv86_{mode}.json")
        if (row.get("execution_status") != "completed"
                or row.get("numerical_status") != "eligible"
                or row.get("response_validation") != "not_performed"):
            raise ValueError(f"FV86 {mode} row contradicts the pinned completed/eligible cohort")
        eligible = True
        gradient = row.get("final_gradient_max")
        residual = row.get("adjoint_relative_residual")
        if eligible and (not isinstance(gradient, (int, float)) or gradient >= 1e-10
                         or not isinstance(residual, (int, float)) or residual > 1e-10
                         or not raw.get("nominal_branch") or not raw.get("workflow", {}).get("response")
                         or not math.isfinite(raw.get("final_score", math.nan))):
            raise ValueError(f"FV86 {mode} eligible summary lacks its numerical gates")
        rows.append({
            "case_id": mode,
            "execution": row["execution_status"],
            "forecast": "available" if eligible else "not_certified",
            "stationarity": "passed" if eligible else "not_assessed",
            "branch": "pointwise_passed" if eligible else "not_checked",
            "local_response_eligibility": "eligible" if eligible else "not_assessed",
            "response": "computed" if eligible else "not_computed",
            "response_validation": row["response_validation"],
            "physical_validation": "not_performed",
            "resource": "not_limited" if row["execution_status"] == "completed" else "see_raw_record",
            "proof_level": "archived_exit_source_and_raw_hashes",
            "failure_category": row.get("failure_category"),
            "evidence_sha256": _hash(HERE / f"fv86_{mode}.json"),
        })
    return rows


def _point_rows() -> list[dict[str, Any]]:
    _require_manifest("fv_point_missing_manifest.json", "fv_point_missing_metrics.json")
    metrics = _load("fv_point_missing_metrics.json")
    if (metrics.get("euler_stages") != 54
            or metrics.get("detected_per_time") != [3, 3, 3]
            or not math.isfinite(metrics.get("score", math.nan))):
        raise ValueError("point-missing record is outside its bounded scope")
    return [{
        "case_id": "three_time_missing", "execution": "recorded_only",
        "forecast": "available", "stationarity": "not_assessed",
        "branch": "pointwise_passed", "local_response_eligibility": "not_assessed",
        "response": "not_attempted",
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "resource": "not_measured", "proof_level": "manifest_sources_current_no_child_exit_record",
        "failure_category": None, "evidence_sha256": _hash(HERE / "fv_point_missing_metrics.json"),
    }]


def _long_rows() -> list[dict[str, Any]]:
    _require_manifest("fv_long_horizon_manifest.json", "fv_long_horizon_metrics.json")
    metrics = _load("fv_long_horizon_metrics.json")
    scenarios = metrics.get("scenarios")
    if metrics.get("execution_status") != "completed" or not isinstance(scenarios, dict):
        raise ValueError("long-horizon probe did not report completion")
    if set(scenarios) != set(CASE_IDS["long_regular_forward"]):
        raise ValueError("long-horizon scenarios differ from the declared pair")
    rows = []
    for case_id in CASE_IDS["long_regular_forward"]:
        scenario = scenarios[case_id]
        if (scenario.get("forecast_shape") != [18, 4, 5]
                or scenario.get("strict_branch_probe", {}).get("status") != "refused"
                or not math.isfinite(scenario.get("score_against_same_operator_synthetic_truth", math.nan))):
            raise ValueError(f"long-horizon {case_id} record is incomplete")
        rows.append({
            "case_id": case_id, "execution": "recorded_only",
            "forecast": "available_same_operator_synthetic",
            "stationarity": "not_assessed", "branch": "refused",
            "local_response_eligibility": "refused_branch_under_current_policy",
            "response": "not_attempted", "response_validation": "not_performed",
            "physical_validation": "not_performed", "resource": "local_alarm_only",
            "proof_level": "manifest_sources_current_no_child_exit_record",
            "failure_category": scenario["strict_branch_probe"].get("reason"),
            "evidence_sha256": _hash(HERE / "fv_long_horizon_metrics.json"),
        })
    return rows


def _partial_rows() -> list[dict[str, Any]]:
    folder = HERE / "partial_branch_gate_detail_attempt3"
    raw_path = folder / "fv_partial_reanalysis.json"
    resource_path = folder / "fv_partial_reanalysis.resource.json"
    preflight_path = folder / "fv_partial_reanalysis.preflight.json"
    manifest = _load("fv_partial_branch_gate_detail_run3_manifest.json")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise ValueError("partial attempt has no archived output manifest")
    for relative, fingerprint in output_hashes.items():
        path = (HERE / relative).resolve()
        if not path.is_relative_to(HERE) or _hash(path) != fingerprint:
            raise ValueError(f"partial attempt output changed: {relative}")
    raw = json.loads(raw_path.read_text())
    resource = json.loads(resource_path.read_text())
    if (resource.get("exit_code") == 0 or raw.get("nominal_eligibility") != "refused"
            or raw.get("response_validation") != "not_established"
            or raw.get("nonlinear_reanalyses") != 0
            or raw.get("preflight_report_sha256") != _hash(preflight_path)
            or raw.get("source_sha256") != manifest.get("source_sha256_match_commit")
            or manifest.get("execution", {}).get("numerical_exit_code") != resource.get("exit_code")):
        raise ValueError("partial-refinement refusal record changed meaning")
    return [{
        "case_id": "masked_case", "execution": "failed",
        "forecast": "not_certified", "stationarity": "not_established_due_to_refinement_refusal",
        "branch": "search_policy_refused",
        "local_response_eligibility": "refused_nominal_under_current_policy",
        "response": "not_attempted",
        "response_validation": "not_established", "physical_validation": "not_performed",
        "resource": "not_limited_process_failed" if resource.get("resource_termination") is None else "see_raw_record",
        "proof_level": "preflight_bound_nonzero_exit_no_final_source_check",
        "failure_category": raw.get("failure_category"),
        "evidence_sha256": _hash(raw_path), "resource_sha256": _hash(resource_path),
    }]


def _counts(rows: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("a profile must declare at least one case")
    values = Counter(row[axis] for row in rows)
    result: dict[str, Any] = {
        "declared_case_count": len(rows),
        "status_counts": dict(sorted(values.items())),
    }
    if axis == "local_response_eligibility":
        assessed = len(rows) - values.get("not_assessed", 0)
        eligible = values.get("eligible", 0)
        result.update(
            assessed_count=assessed,
            eligible_count=eligible,
            eligible_evidence_coverage_fraction_of_declared=eligible / len(rows),
            eligible_fraction_given_assessed=eligible / assessed if assessed else None,
        )
    if axis == "response":
        computed = values.get("computed", 0)
        attempted = len(rows) - values.get("not_attempted", 0)
        result.update(
            computed_count=computed,
            computed_evidence_coverage_fraction_of_declared=computed / len(rows),
            computed_fraction_given_attempt=computed / attempted if attempted else None,
        )
    if axis in ("response_validation", "physical_validation"):
        attempted = values.get("passed", 0) + values.get("failed", 0)
        result["independent_validation_attempted_count"] = attempted
        result["attempted_fraction_of_declared"] = attempted / len(rows)
        result["passed_count"] = values.get("passed", 0)
        result["validated_evidence_coverage_fraction_of_declared"] = (
            values.get("passed", 0) / len(rows)
        )
        result["pass_fraction_given_attempt"] = (
            values.get("passed", 0) / attempted if attempted else None
        )
    return result


def run() -> dict[str, Any]:
    groups = {
        "fv86_full_support": _fv86_rows(),
        "point_missing_fixed_control": _point_rows(),
        "long_regular_forward": _long_rows(),
        "partial_two_hole": _partial_rows(),
    }
    if tuple(groups) != tuple(CASE_IDS):
        raise ValueError("profile inventory differs from the frozen plan")
    profiles = {}
    for profile_id, rows in groups.items():
        if tuple(row["case_id"] for row in rows) != CASE_IDS[profile_id]:
            raise ValueError(f"{profile_id} case inventory differs from the frozen plan")
        profiles[profile_id] = {
            "declared_case_ids": CASE_IDS[profile_id],
            "rows": rows,
            "axes": {axis: _counts(rows, axis) for axis in AXES},
        }
    return {
        "scope": "retrospective stratified inventory of archived evidence; no cross-profile success rate or new numerical execution",
        "profile_count": len(profiles),
        "profiles": profiles,
        "cross_profile_fraction": None,
        "input_artifact_sha256": {
            name: _hash(HERE / name) for name in (
                "fv86_execution_status_summary.json",
                "fv_point_missing_manifest.json",
                "fv_long_horizon_manifest.json",
                "partial_branch_gate_detail_attempt3/fv_partial_reanalysis.json",
                "partial_branch_gate_detail_attempt3/fv_partial_reanalysis.resource.json",
                "partial_branch_gate_detail_attempt3/fv_partial_reanalysis.preflight.json",
                "fv_partial_branch_gate_detail_run3_manifest.json",
            )
        },
        "later_seed_a_reanalysis": "separate PR176 evidence, not retroactively included in the FV86 execution cohort",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
