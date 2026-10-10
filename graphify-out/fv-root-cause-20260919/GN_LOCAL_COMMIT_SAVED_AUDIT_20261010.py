"""Saved-receipt audit for one GN local commit and its next-point readiness.

This tool reads JSON receipts only. It never imports ADVAR modules or runs FV,
gradient, HVP, optimizer, or candidate-selection work.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PLAN_SHA = "11e4eff337084ba9a9602b477854349969af8306926911fa49c4ab1e8e869b6e"
BASE_CONTROL_SHA = "2ec34eeeef531b94e21b5b4372ced2e07257a5407541072d300f5a3f513e1fcf"
APPROVED_CONTROL_SHA = "311f27d095be4976ca6131b83b6c4f3d46bc404d4f58921800b5600e38ec0652"
ALPHA = 5.574488376651601e-6
BASE_THETA = 0.4830786491678902
BASE_J = 0.06119249248090723
BASE_F2 = 0.00447848922842147
POLICY = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "optimizer_steps": 1, "candidate_count": 1,
    "candidate_alpha": ALPHA, "radius": 0.05, "jacobian_row_vjp_calls": 24,
    "dense_solves": 1, "hvp_calls": 2, "face_scale": 0.84,
    "candidate_grid_per_direction": 0, "max_candidates_per_iteration": 0,
    "root_claim": False, "minimum_claim": False,
    "score_claim": False, "response_claim": False,
}


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    raw = gzip.decompress(data) if path.suffix == ".gz" else data
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _tensor(value: Any, label: str, length: int = 26) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite float64 vector of length {length}")
    return array


def _tensor_sha(value: Any) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _near(a: Any, b: Any) -> bool:
    try:
        left, right = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    eps = np.finfo(np.float64).eps
    return abs(left - right) <= 128 * eps * max(abs(left), abs(right), np.finfo(np.float64).tiny)


def _require_p2(proposal: dict[str, Any], repeat: dict[str, Any]) -> None:
    flags = (
        "mixing_minimum_valid", "minimum_theta_matches_proposal", "side_gradients_finite",
        "native_objective_matches_proposal", "side_objectives_match_native",
        "merit_matches_proposal", "face_audit_passed", "branch_pair_passed",
        "trace_matches_proposal", "source_unchanged", "fixed_input_unchanged",
        "runtime_unchanged", "deadline_passed",
    )
    if not all(repeat.get(key) is True for key in flags):
        raise ValueError("independent P2 closure flags are incomplete")
    if repeat.get("control") != proposal.get("control"):
        raise ValueError("P2 repeat control differs from its proposal")
    if repeat.get("theta") != proposal.get("theta"):
        raise ValueError("P2 repeat theta differs from its proposal")
    for key in ("objective", "F_squared"):
        if not _near(repeat.get(key), proposal.get(key)):
            raise ValueError(f"P2 repeat {key} differs from its proposal")
    expected, actual = proposal.get("side_gradients", {}), repeat.get("side_gradients", {})
    if set(expected) != {"-1", "1"} or set(actual) != {"-1", "1"}:
        raise ValueError("P2 closure is missing a side-gradient vector")
    for side in ("-1", "1"):
        left = _tensor(expected[side], f"proposal side {side} gradient")
        right = _tensor(actual[side], f"repeat side {side} gradient")
        scale = max(float(np.abs(left).max()), float(np.abs(right).max()), np.finfo(np.float64).tiny)
        if float(np.abs(left - right).max()) > 128 * np.finfo(np.float64).eps * scale:
            raise ValueError(f"P2 repeat side {side} gradient differs from its proposal")


def _require_resource(raw_sha: str, run_path: Path | None, resource_path: Path | None) -> dict[str, Any] | None:
    if run_path is None and resource_path is None:
        return None
    if run_path is None or resource_path is None:
        raise ValueError("provide both --run and --resource receipts")
    run, _ = _read_json(run_path)
    resource, _ = _read_json(resource_path)
    if (run.get("execution_status") != "completed" or run.get("child_sha256") != raw_sha
            or run.get("child_read_error") is not None or run.get("resource") != resource
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != 660.0
            or resource.get("rss_limit_bytes") != 2 * 1024**3
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > 660.0
            or not isinstance(resource.get("sampled_peak_rss_bytes"), (int, float))
            or resource["sampled_peak_rss_bytes"] >= 2 * 1024**3):
        raise ValueError("execution/resource receipts do not close")
    return resource


def audit(raw: dict[str, Any], raw_sha: str) -> dict[str, Any]:
    if (raw.get("plan_sha256") != PLAN_SHA or raw.get("base_control_sha256") != BASE_CONTROL_SHA
            or raw.get("policy") != POLICY or not _near(raw.get("base_theta"), BASE_THETA)
            or not _near(raw.get("base_objective"), BASE_J)
            or not _near(raw.get("base_F_squared"), BASE_F2)
            or not _near(raw.get("postcommit_candidate_alpha"), ALPHA)
            or raw.get("root_claim") is not False
            or raw.get("minimum_claim") is not False or raw.get("score_claim") is not False
            or raw.get("response_claim") is not False):
        raise ValueError("plan/base/fixed-step metadata or claim flags changed")

    proposal, repeat = raw.get("proposal"), raw.get("final_repeat")
    if not isinstance(proposal, dict) or not isinstance(repeat, dict):
        raise ValueError("committed proposal or independent final repeat is absent")
    if (not _near(proposal.get("alpha"), ALPHA)
            or proposal.get("candidate_mixing_minimum") is not True
            or proposal.get("J_armijo_passed") is not True
            or proposal.get("F_squared_armijo_passed") is not True
            or proposal.get("face_audit_passed") is not True
            or proposal.get("branch_pair_passed") is not True
            or proposal.get("side_objectives_match_native") is not True
            or proposal.get("side_gradients_finite") is not True):
        raise ValueError("fixed candidate acceptance facts are incomplete")
    _require_p2(proposal, repeat)

    control = _tensor(raw.get("current_control"), "committed control")
    proposal_control = _tensor(proposal.get("control"), "proposal control")
    last_control = _tensor(raw.get("last_confirmed_control"), "last-confirmed control")
    if (not np.array_equal(control, proposal_control) or not np.array_equal(control, last_control)
            or repeat.get("control") != raw.get("current_control")
            or _tensor_sha(control) != APPROVED_CONTROL_SHA
            or raw.get("current_control_sha256") != APPROVED_CONTROL_SHA
            or raw.get("last_confirmed_control_sha256") != APPROVED_CONTROL_SHA
            or raw.get("current_control_sha256") != raw.get("last_confirmed_control_sha256")
            or raw.get("optimizer_steps_applied") != 1 or raw.get("accepted_iterations") != 1
            or raw.get("candidate_committed") is not True
            or raw.get("last_confirmed_iterations") != 1):
        raise ValueError("the single durable commit does not close at the approved control")
    if raw.get("last_confirmed_closure") != repeat:
        raise ValueError("last-confirmed closure differs from the independent repeat")

    identity = raw.get("archive_identity", {})
    if (identity.get("reused_same_point_hvp_vectors") != 2
            or identity.get("fresh_base_hvp_calls") != 0
            or identity.get("fresh_postcommit_hvp_budget") != 2):
        raise ValueError("same-point reused versus new-point HVP accounting is unclear")

    readiness = raw.get("postcommit_gn_readiness", {})
    has_readiness = bool(readiness)
    is_ready = raw.get("readiness_complete") is True
    if raw.get("optimizer_steps_after_commit") not in (None, 0):
        raise ValueError("an additional optimizer step occurred after the commit")
    if has_readiness and (readiness.get("candidate_count") != 0
            or readiness.get("optimizer_steps_after_commit") != 0
            or readiness.get("direction_model") != "robust_gn_coupled"):
        raise ValueError("post-commit readiness includes a second candidate or wrong model")

    if is_ready:
        control_sha = _tensor_sha(control)
        direction = _tensor(readiness.get("direction"), "post-commit direction")
        direction_sha = _tensor_sha(direction)
        if (readiness.get("direction_sha256") != direction_sha
                or raw.get("postcommit_gn_readiness_direction_sha256") != direction_sha
                or raw.get("postcommit_current_observation", {}).get("control_sha256") != control_sha
                or raw.get("jacobian_rows_started") != 24
                or raw.get("jacobian_rows_completed") != 24):
            raise ValueError("post-commit direction or fresh-row counters do not close")

        rows = raw.get("jacobian_row_history", [])
        expected = {(side, row) for side in (-1, 1) for row in range(12)}
        actual = {(item.get("side"), item.get("row")) for item in rows}
        theta = readiness.get("working_theta")
        if (len(rows) != 24 or actual != expected
                or any(item.get("status") != "completed"
                    or item.get("base_control_sha256") != control_sha
                    or item.get("theta") != theta
                    or not isinstance(item.get("residual_value"), (int, float))
                    or not math.isfinite(float(item.get("residual_value", float("nan"))))
                    or len(item.get("gradient", [])) != 26
                    or not all(math.isfinite(float(value)) for value in item.get("gradient", []))
                    for item in rows)):
            raise ValueError("post-commit 24-row VJP history is incomplete or misbound")

        solve = raw.get("dense_solve_audit", {})
        solve_direction = _tensor(solve.get("direction"), "dense-solve direction")
        if (raw.get("dense_solves_started") != 1 or raw.get("dense_solves_completed") != 1
                or solve.get("dense_solves") != 1 or solve.get("dense_solve_dimension") != 12
                or solve.get("S_dimension") != 12 or solve.get("positive_definite_from_cholesky") is not True
                or not isinstance(solve.get("solve_residual"), (int, float))
                or not isinstance(solve.get("solve_residual_budget"), (int, float))
                or not math.isfinite(float(solve.get("solve_residual", float("nan"))))
                or not math.isfinite(float(solve.get("solve_residual_budget", float("nan"))))
                or solve["solve_residual"] < 0 or solve["solve_residual_budget"] <= 0
                or solve["solve_residual"] > solve["solve_residual_budget"]
                or not np.array_equal(solve_direction, direction)):
            raise ValueError("post-commit 12D solve audit failed")

        hvps = raw.get("hvp_history", [])
        started = raw.get("hvp_started_history", [])
        if (raw.get("hvp_calls_started") != 2 or raw.get("hvp_calls_completed") != 2
                or len(hvps) != 2 or {item.get("side") for item in hvps} != {-1, 1}
                or any(item.get("base_control_sha256") != control_sha
                    or item.get("direction_sha256") != direction_sha
                    or item.get("status") != "completed"
                    or item.get("theta") != raw.get("current_theta")
                    or item.get("working_theta") != theta
                    or item.get("operator") != "selected_face_extension"
                    or item.get("direction_model") != "robust_gn_coupled"
                    for item in hvps)
                or len(started) != 2 or {item.get("side") for item in started} != {-1, 1}
                or any(item.get("base_control_sha256") != control_sha
                    or item.get("direction_sha256") != direction_sha
                    or item.get("phase") != "postcommit_current_point"
                    or item.get("theta") != raw.get("current_theta")
                    or item.get("working_theta") != theta
                    or item.get("operator") != "selected_face_extension"
                    or item.get("direction_model") != "robust_gn_coupled"
                    for item in started)):
            raise ValueError("post-commit paired HVP history is incomplete or misbound")
        for name in ("hminus", "hplus"):
            _tensor(readiness.get(name), f"post-commit {name}")
        for name in ("residual", "residual_direction"):
            _tensor(readiness.get(name), f"post-commit {name}", length=27)
        if not all(raw.get(flag) is True for flag in (
                "source_unchanged", "fixed_input_unchanged", "runtime_unchanged", "deadline_passed")):
            raise ValueError("post-commit source/input/runtime/deadline closure is incomplete")
    elif raw.get("numerical_status") == "candidate_committed_current_gn_model_ready_no_second_candidate":
        raise ValueError("raw claims completed readiness without a readiness receipt")

    return {
        "raw_sha256": raw_sha,
        "commit_closed": True,
        "candidate_alpha": ALPHA,
        "control_sha256": APPROVED_CONTROL_SHA,
        "optimizer_steps_applied": 1,
        "P2_closed": True,
        "postcommit_readiness_complete": is_ready,
        "postcommit_new_point": {
            "rows_completed": raw.get("jacobian_rows_completed", 0),
            "dense_solves_completed": raw.get("dense_solves_completed", 0),
            "paired_hvps_completed": raw.get("hvp_calls_completed", 0),
            "additional_candidate_count": readiness.get("candidate_count", 0),
        },
        "claims": {key: raw.get(key, False) for key in (
            "root_claim", "minimum_claim", "score_claim", "response_claim")},
        "numerical_status": raw.get("numerical_status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=Path, help="saved child JSON or .json.gz receipt")
    parser.add_argument("--run", type=Path, help="optional parent run receipt")
    parser.add_argument("--resource", type=Path, help="optional resource receipt; requires --run")
    args = parser.parse_args()
    raw, raw_sha = _read_json(args.step)
    report = audit(raw, raw_sha)
    resource = _require_resource(raw_sha, args.run, args.resource)
    report["execution_resource_closed"] = resource is not None
    if resource is not None:
        report["resource"] = {
            "elapsed_seconds": resource["elapsed_seconds"],
            "sampled_peak_rss_bytes": resource["sampled_peak_rss_bytes"],
            "wall_limit_seconds": resource["wall_limit_seconds"],
            "rss_limit_bytes": resource["rss_limit_bytes"],
        }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
