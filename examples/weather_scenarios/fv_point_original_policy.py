"""Read-only support classification for the original correlated point input."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_ORIGINAL_POLICY_PLAN.md"
PLAN_SHA256 = "0510599ecf4d2e3eca50398520d09f7db310641c0fa031ad45627f72c1fe5cc9"
EXPECTED_RAW_MANIFEST_SHA256 = {
    "point_sector_root_attempt1": "88926d99730f75f6d6a9341046f883391525b76cd0576b71faf0af0a64a3de34",
    "point_merit_root_attempt1": "76aa289a628f37b8a4b7eba2e94ca408ef25f7bba24e69f72d2a37f67d44ef9e",
}
EXPECTED_INPUT_IDENTITY = {
    "problem_identity": {
        "fixed_problem_sha256": "3de086e6436c21913899fcc7518fcf3eca62a42d552f94c6eb1e2b293665a6e5",
        "scope": "fixed point-observation identity; caller also binds code, control, parameters and branch",
    },
    "tensor_sha256": {
        "direction": "679fc3ae3a787bbeef8e9efe6f841b4e6eba03cef102198119a820577c5d9fed",
        "observation_coordinates": "7c63a1c24dae48b1abc2d4bc9ce6f0f0e936b9633a9a9259bf7d07f2078cad02",
        "observation_correlation": "7c304612b1f9ef9392ffbd1a1a4b092343746a02dcf200b12a2866f34aaeeaef",
        "parameters": "0344522a5a5ce1a61213e74d3390515a29880cd327bb8a29ea25b20fbffd360a",
        "verification": "b6e21eabc83a8a3588a2ebd893b63d2d1d7afa7500bb83fe57a13584cb8350a9",
        "warm_control": "b61e49b6f5daded19ea1375dc37e3959c1673503309ce6c68cf1deda195462b5",
    },
}
EXPECTED_CONSTRUCTED_PROBLEM_SHA256 = "464664431b230dbbbc997b3e5f5bcf669c8f2fb629ae298b6c0a6d61502e5b79"
EXPECTED_CONSTRUCTED_REPORT_SHA256 = "e0996a5081e20037d39f95049768d15d67e768b8aeda133a00fd638ff39e47aa"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected archived object: {path}")
    return value


def _attempt(name: str, *, status: str, switch_reason: str,
             switch_count: int, gradient_max: float) -> dict[str, Any]:
    directory = EVIDENCE / name
    if _sha(directory / "manifest.json") != EXPECTED_RAW_MANIFEST_SHA256[name]:
        raise ValueError("original point historical raw manifest changed")
    prefix = "point_sector_root" if name == "point_sector_root_attempt1" else "point_merit_root"
    manifest = _load(directory / "manifest.json")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) < 10:
        raise ValueError("original point raw manifest is incomplete")
    for relative, expected in hashes.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or _sha(path) != expected:
            raise ValueError(f"original point archived artifact changed: {relative}")
    child_path = directory / f"{prefix}.json"
    run_path = directory / f"{prefix}.run.json"
    child, run = _load(child_path), _load(run_path)
    resource = run.get("resource")
    if not isinstance(resource, dict):
        raise ValueError("original point resource record is missing")
    trials = child.get("trial_records")
    solves = child.get("linear_solves")
    if not isinstance(trials, list) or not isinstance(solves, list):
        raise ValueError("original point candidate/PCG history missing")
    accepted = [entry for entry in trials if entry.get("accepted") is True]
    final = [entry for entry in trials if entry.get("iteration") == 7]
    switches = [entry for entry in accepted if entry.get("policy_reason") == switch_reason]
    plan_snapshot = directory / ("FV_POINT_SECTOR_ROOT_PLAN.md.txt"
                                  if prefix == "point_sector_root" else
                                  "FV_POINT_MERIT_ROOT_PLAN.md.txt")
    if not (
        child.get("numerical_status") == status
        and child.get("response_validation") == "not_performed"
        and run.get("execution_status") == "completed"
        and run.get("numerical_status") == status
        and run.get("response_validation") == "not_performed"
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("plan_unchanged") is True
        and child.get("plan_sha256") == _sha(plan_snapshot)
        and child.get("source_before") == child.get("source_after")
        and child.get("input_before") == child.get("input_after")
        and child.get("input_before") == EXPECTED_INPUT_IDENTITY
        and child.get("phase") in ("sector_refinement", "merit_refinement")
        and len(accepted) == 6 and accepted[-1].get("iteration") == 6
        and len(switches) == switch_count
        and math.isclose(accepted[-1].get("gradient_max", math.nan), gradient_max,
                         rel_tol=1e-12, abs_tol=1e-15)
        and len(final) == 16 and all(entry.get("accepted") is False
                                      and entry.get("rejection") == "policy"
                                      for entry in final)
        and len(solves) == 7 and all(entry.get("converged") is True
                                     and entry.get("relative_residual", math.inf) <= 1e-10
                                     for entry in solves)
        and resource.get("exit_code") == 2
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("wall_limit_seconds") == 600
        and resource.get("rss_limit_bytes") == 1024**3
        and isinstance(resource.get("rss_samples"), int) and resource["rss_samples"] > 0
        and isinstance(resource.get("sampled_peak_rss_bytes"), int)
        and 0 < resource["sampled_peak_rss_bytes"] <= 1024**3
        and isinstance(resource.get("elapsed_seconds"), (int, float))
        and 0 <= resource["elapsed_seconds"] <= 600
        and not any(key in child for key in ("control", "final_branch", "response"))
    ):
        raise ValueError("original point archived refusal changed meaning")
    return {
        "raw_manifest_sha256": _sha(directory / "manifest.json"),
        "child_sha256": _sha(child_path),
        "input_identity": child["input_before"],
        "status": status,
        "accepted_steps": len(accepted),
        "accepted_signature_switches": len(switches),
        "last_accepted_gradient_max": accepted[-1]["gradient_max"],
        "final_policy_rejections": len(final),
        "converged_pcg_solves": len(solves),
        "child_exit_code": resource["exit_code"],
        "resource_termination": None,
        "elapsed_seconds": resource["elapsed_seconds"],
        "sampled_peak_rss_bytes": resource["sampled_peak_rss_bytes"],
    }


def classify() -> dict[str, Any]:
    if _sha(PLAN) != PLAN_SHA256:
        raise ValueError("original point support plan changed")
    sector = _attempt(
        "point_sector_root_attempt1", status="sector_refused",
        switch_reason="sector_switch_measured_decrease", switch_count=2,
        gradient_max=0.006432750851827679,
    )
    merit = _attempt(
        "point_merit_root_attempt1", status="merit_refused",
        switch_reason="merit_switch_decrease", switch_count=1,
        gradient_max=0.006602351499307973,
    )
    merit_child = _load(EVIDENCE / "point_merit_root_attempt1/point_merit_root.json")
    sector_child = _load(EVIDENCE / "point_sector_root_attempt1/point_sector_root.json")
    sector_last = [entry for entry in sector_child["trial_records"] if entry.get("accepted")][-1]
    constructed_path = EVIDENCE / "point_centered_response_attempt1/point_centered_response.json"
    constructed = _load(constructed_path)
    original_problem = merit["input_identity"]["problem_identity"]["fixed_problem_sha256"]
    constructed_problem = constructed["input_before"]["problem"]["fixed_problem_sha256"]
    if (sector["input_identity"] != merit["input_identity"]
            or _sha(constructed_path) != EXPECTED_CONSTRUCTED_REPORT_SHA256
            or merit_child.get("prior_report_sha256") != sector["child_sha256"]
            or merit_child.get("seed_control_sha256") != sector_last["candidate_control_sha256"]
            or original_problem != EXPECTED_INPUT_IDENTITY["problem_identity"]["fixed_problem_sha256"]
            or constructed_problem != EXPECTED_CONSTRUCTED_PROBLEM_SHA256
            or constructed.get("numerical_status") != "eligible"
            or constructed.get("response_validation") != "passed"):
        raise ValueError("original versus constructed point identity or lineage changed")
    return {
        "plan_sha256": PLAN_SHA256,
        "numerical_solver_runs": 0,
        "original_problem_sha256": original_problem,
        "constructed_problem_sha256": constructed_problem,
        "constructed_response_report_sha256": _sha(constructed_path),
        "attempts": {"triple_decrease": sector, "gradient_merit": merit},
        "support_status": "unsupported_by_current_declared_nominal_search_policies",
        "response_computed": False,
        "response_validation": "not_performed",
        "stationary_root_absence_proved": False,
        "scope": "exact original zero-centered correlated point input only; constructed centered-prior input is different",
        "future_method": "new sector-aware root search with fresh stationarity/curvature/branch/adjoint/signed-endpoint proof",
    }


if __name__ == "__main__":
    print(json.dumps(classify(), indent=2, sort_keys=True))
