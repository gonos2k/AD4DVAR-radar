"""Read-only response-support decision for the archived 3-hour point case."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_3H_RESPONSE_POLICY_PLAN.md"
PLAN_SHA256 = "d1d87f91306a2c4c3fed2b99c329314c04670a4ada56c79a884e72ebdd905b00"
RAW = EVIDENCE / "point_3h_forward_attempt1"
RAW_MANIFEST_SHA256 = "056963a6f264a97aa165501a1ff6c07bf62842f17ff26992037cb977e0469de8"
FORWARD_EVIDENCE = EVIDENCE / "FV_POINT_3H_FORWARD_EVIDENCE.json"
FORWARD_EVIDENCE_SHA256 = "a78f09769c5201eafcb7eaedd4f8e506266f80409476780e128d4b08608cfc4e"
BRANCH_REFUSAL = "minmod joint oracle left its strict smooth branch"
EXPECTED_INPUT_IDENTITY_SHA256 = "00665e879d0e370e15f2bf3ef253a2267eb59335c2f51036289ff22632808744"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected archived object: {path}")
    return value


def classify() -> dict[str, Any]:
    if (_sha(PLAN) != PLAN_SHA256
            or _sha(RAW / "manifest.json") != RAW_MANIFEST_SHA256
            or _sha(FORWARD_EVIDENCE) != FORWARD_EVIDENCE_SHA256):
        raise ValueError("point 3-hour support plan or archived evidence changed")
    manifest = _load(RAW / "manifest.json")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != 16:
        raise ValueError("point 3-hour raw manifest is incomplete")
    for name, expected in hashes.items():
        path = (RAW / name).resolve()
        if not path.is_relative_to(RAW.resolve()) or _sha(path) != expected:
            raise ValueError(f"point 3-hour archived artifact changed: {name}")
    child_path = RAW / "point_3h_forward.json"
    child = _load(child_path)
    parent = _load(RAW / "point_3h_forward.run.json")
    resource_sidecar = _load(RAW / "point_3h_forward.resource.json")
    forward_evidence = _load(FORWARD_EVIDENCE)
    resource = parent.get("resource")
    source_hashes = forward_evidence.get("sha256")
    if (not isinstance(resource, dict)
            or resource != resource_sidecar
            or not isinstance(source_hashes, dict)
            or not isinstance(child.get("source_before"), dict)
            or any(source_hashes.get(name) != digest
                   for name, digest in child["source_before"].items())):
        raise ValueError("point 3-hour resource sidecar or source provenance mismatch")
    branch = child.get("strict_branch_diagnostic")
    analysis = child.get("analysis")
    terminal = child.get("terminal")
    layout = child.get("layout")
    if not all(isinstance(value, dict) for value in (branch, analysis, terminal, layout)):
        raise ValueError("point 3-hour archived numerical fields are incomplete")
    assert isinstance(branch, dict) and isinstance(analysis, dict)
    assert isinstance(terminal, dict) and isinstance(layout, dict)
    if not (
        child.get("execution_phase") == "finished"
        and child.get("phase") == "finished"
        and child.get("forward_validation") == "passed"
        and child.get("forecast_available") is True
        and child.get("stationarity_passed") == "not_tested"
        and child.get("response_computed") is False
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("plan_unchanged") is True
        and child.get("archive_unchanged") is True
        and child.get("source_before") == child.get("source_after")
        and child.get("input_before") == child.get("input_after")
        and hashlib.sha256(json.dumps(
            child.get("input_before"), sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode()).hexdigest() == EXPECTED_INPUT_IDENTITY_SHA256
        and parent.get("execution_status") == "completed"
        and parent.get("forward_validation") == "passed"
        and parent.get("response_validation") == "not_performed"
        and parent.get("physical_validation") == "not_performed"
        and type(child.get("pid")) is int and child["pid"] == resource.get("child_pid")
        and resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("wall_limit_seconds") == 180
        and resource.get("rss_limit_bytes") == 1024**3
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= 1024**3
        and isinstance(resource.get("elapsed_seconds"), (int, float))
        and math.isfinite(resource["elapsed_seconds"])
        and 0 <= resource["elapsed_seconds"] <= 180
        and layout.get("observation_times_seconds") == [0.0, 600.0, 1200.0]
        and layout.get("forecast_time_seconds") == 12000.0
        and layout.get("controls") == 26 and layout.get("parameters") == 13
        and layout.get("euler_stages") == 3600
        and child.get("forecast_steps") == 18
        and child.get("substeps_per_interval") == 90
        and child.get("analysis_boundary_stage_pairs") == 180
        and child.get("future_boundary_stage_pairs") == 1620
        and analysis.get("observed_minmod_stages") == 360
        and terminal.get("observed_minmod_stages") == 3600
        and isinstance(analysis.get("point_observation_max_abs_difference_dbz"), (int, float))
        and math.isfinite(analysis["point_observation_max_abs_difference_dbz"])
        and 0 <= analysis["point_observation_max_abs_difference_dbz"] <= 1e-9
        and isinstance(terminal.get("max_abs_difference_from_same_operator_truth_dbz"), (int, float))
        and math.isfinite(terminal["max_abs_difference_from_same_operator_truth_dbz"])
        and 0 <= terminal["max_abs_difference_from_same_operator_truth_dbz"] <= 1e-9
        and branch.get("status") == "refused"
        and branch.get("reason") == BRANCH_REFUSAL
        and branch.get("observed_minmod_stages") == 0
    ):
        raise ValueError("point 3-hour archived forward or branch status changed meaning")
    return {
        "plan_sha256": PLAN_SHA256,
        "raw_manifest_sha256": RAW_MANIFEST_SHA256,
        "forward_evidence_sha256": FORWARD_EVIDENCE_SHA256,
        "archived_child_sha256": _sha(child_path),
        "input_identity": child["input_before"],
        "numerical_solver_runs": 0,
        "execution_status": "completed",
        "forward_validation": "passed_same_operator",
        "stationarity_passed": "not_tested",
        "local_branch_supported": False,
        "response_support_status": "unsupported_by_current_strict_branch_gate",
        "response_computed": False,
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "strict_branch_reason": BRANCH_REFUSAL,
        "strict_branch_admitted_stages": 0,
        "separate_analysis_stages": 360,
        "terminal_forecast_call_stages": 3600,
        "stationary_root_absence_proved": False,
        "scope": "exact fixed 4x5 point-observation 18-lead input/control only; forward success is not response eligibility",
        "future_method": "separately qualified strict 3-hour branch and stationary point, then exact adjoint/VJP and signed endpoints",
    }


if __name__ == "__main__":
    print(json.dumps(classify(), indent=2, sort_keys=True))
