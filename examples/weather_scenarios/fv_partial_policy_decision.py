"""Classify one archived two-hole refusal without rerunning FV numerics."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EVIDENCE = Path(__file__).resolve().parents[2] / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_PARTIAL_POLICY_DECISION_PLAN.md"
PLAN_SHA256 = "e9567c04c631d99b25f1281c94cae592ab09a02fafc1c88b6e693cce123d1001"
MANIFEST = EVIDENCE / "fv_partial_branch_gate_detail_run3_manifest.json"
RAW = EVIDENCE / "partial_branch_gate_detail_attempt3"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify() -> dict[str, Any]:
    if _sha(PLAN) != PLAN_SHA256:
        raise ValueError("partial policy decision plan changed")
    manifest = json.loads(MANIFEST.read_text())
    outputs = manifest["output_sha256"]
    if not isinstance(outputs, dict) or len(outputs) != 6:
        raise ValueError("partial attempt-3 output manifest is incomplete")
    for name, expected in outputs.items():
        actual = EVIDENCE / name
        if not actual.resolve().is_relative_to(RAW.resolve()) or _sha(actual) != expected:
            raise ValueError(f"partial archived output changed: {name}")
    child = json.loads((RAW / "fv_partial_reanalysis.json").read_text())
    resource = json.loads((RAW / "fv_partial_reanalysis.resource.json").read_text())
    preflight = json.loads((RAW / "fv_partial_reanalysis.preflight.json").read_text())
    preflight_resource = json.loads((RAW / "fv_partial_reanalysis.preflight.resource.json").read_text())
    error = child.get("error")
    if not isinstance(error, str):
        raise ValueError("partial refusal has no original error record")
    match = re.search(r"branch_reason_counts=(\{[^}]+\})", error)
    if match is None:
        raise ValueError("partial refusal lacks branch first-failure counts")
    reasons = ast.literal_eval(match.group(1))
    expected_reasons = {
        "partial branch signature changed": 3,
        "minimum face flux margin is at or below 1e-4": 13,
    }
    if reasons != expected_reasons or sum(reasons.values()) != 16:
        raise ValueError("partial archived branch refusal counts changed")
    if not all(fragment in error for fragment in (
        "iteration=4", "no finite candidate evaluation",
        "branch_rejections=16", "nonfinite_candidate_rejections=0",
        "finite_armijo_rejections=0",
    )):
        raise ValueError("partial archived line-search refusal changed")
    if not (
        manifest["execution"]["branch_reason_counts"] == {
            "signature_changed": 3, "face_flux_margin": 13
        }
        and manifest["execution"]["numerical_exit_code"] == 1
        and manifest["execution"]["preflight_exit_code"] == 0
        and manifest["limits"]["numerical_wall_seconds"] == 240
        and manifest["limits"]["sampled_rss_bytes"] == 1024**3
        and child.get("phase") == "partial_newton_refinement"
        and child.get("numerical_status") == "refused"
        and child.get("nominal_eligibility") == "refused"
        and child.get("response_validation") == "not_established"
        and child.get("endpoint_eligibility") == "not_attempted"
        and child.get("nonlinear_reanalyses") == 0
        and child.get("endpoints") == [] and child.get("pairs") == []
        and child.get("preflight_rechecks", {}).get("after") == "pending"
        and child.get("source_sha256") == manifest["source_sha256_match_commit"]
        and preflight.get("status") == "preflight_only"
        and preflight_resource.get("exit_code") == 0
        and resource.get("exit_code") == 1
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("wall_limit_seconds") == 240
        and resource.get("rss_limit_bytes") == 1024**3
        and isinstance(resource.get("rss_samples"), int) and resource["rss_samples"] > 0
        and isinstance(resource.get("sampled_peak_rss_bytes"), int)
        and 0 < resource["sampled_peak_rss_bytes"] < 1024**3
        and isinstance(resource.get("elapsed_seconds"), (int, float))
        and 0 <= resource["elapsed_seconds"] <= 240
    ):
        raise ValueError("partial archived status or resource evidence changed")
    return {
        "plan_sha256": PLAN_SHA256,
        "archive_manifest_sha256": _sha(MANIFEST),
        "archived_child_sha256": outputs["partial_branch_gate_detail_attempt3/fv_partial_reanalysis.json"],
        "archive_source_scope": "preflight/source hashes and output manifest; child after-run preflight recheck pending",
        "execution_status": "nonzero_exit_without_resource_termination",
        "preflight_status": "completed",
        "numerical_status": "refused_by_declared_nominal_policy",
        "support_status": "unsupported_by_declared_nominal_policy",
        "response_computed": False,
        "response_validation": "not_established",
        "proof_of_stationary_root_absence": False,
        "first_failing_branch_reasons": {
            "signature_changed": 3, "face_flux_margin": 13,
        },
        "reason_count_scope": "first failed predicate; 13 face-margin candidates were not signature-tested",
        "finite_armijo_evaluations": 0,
        "historical_failure_category": child["failure_category"],
        "future_method_scope": "branch-aware nominal search would be a new separately validated numerical method",
        "resource": {
            "elapsed_seconds": resource["elapsed_seconds"],
            "sampled_peak_rss_bytes": resource["sampled_peak_rss_bytes"],
            "rss_samples": resource["rss_samples"],
            "wall_limit_seconds": resource["wall_limit_seconds"],
            "rss_limit_bytes": resource["rss_limit_bytes"],
        },
    }


def write_result(output: Path) -> dict[str, Any]:
    resolved = output.resolve()
    if (resolved in (PLAN.resolve(), MANIFEST.resolve())
            or resolved.is_relative_to(RAW.resolve())):
        raise ValueError("partial policy output cannot overwrite archived evidence")
    result = classify()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = write_result(arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
