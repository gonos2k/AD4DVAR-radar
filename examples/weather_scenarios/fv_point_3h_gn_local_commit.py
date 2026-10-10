"""Commit one freshly requalified candidate from the frozen GN local window.

The archived same-point HVP pair is reused only at its original control.
After the independent final repeat is closed and persisted, all GN readiness
work is rebuilt at the committed control.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_gn_local_window_probe as local
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_resume as gn_resume
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "GN_LOCAL_COMMIT_PLAN_20261010.json"
OUTPUT_DIR = EVIDENCE / "gn_local_commit_20261010_attempt1"
SELF = "examples/weather_scenarios/fv_point_3h_gn_local_commit.py"
TEST = "tests/test_fv_point_3h_gn_local_commit.py"
LOCAL_PLAN_SHA = "f0bce6629d3de1071276ce3cf93dcec70a2cba79f8d5661f847f7826670a92a9"
LOCAL_ARCHIVE = EVIDENCE / "GN_LOCAL_WINDOW_ARCHIVE_20261010.json"
LOCAL_ARCHIVE_SHA = "960dc4b872fc82615cd9d8b1384dcb02569534b04d9b5b4ff024a9a27dbeffb7"
LOCAL_GZIP = local.ATTEMPT / "diagnostic.json.gz"
LOCAL_RUN = local.ATTEMPT / "diagnostic.run.json"
LOCAL_RESOURCE = local.ATTEMPT / "diagnostic.resource.json"
LOCAL_GZIP_SHA = "8318092f3e8d49b17af1dc8a4e345bd5a0c4d3d2a6ff498ab227246be0c0d5aa"
LOCAL_RAW_SHA = "1c2139639661a9fb3f1c1937522b97f558010cb7cbcbd765e9ffdcbb3d777f0b"
LOCAL_RUN_SHA = "6b3af5f1792727b4d00ab0434365ff3e2112230e89bb7edaa535a7c096034adc"
LOCAL_RESOURCE_SHA = "79d8988cd3b8cb13b974088559229ce915eec2814fd81d2155998963e1ab5ac7"
BASE_CONTROL_SHA = local.BASE_CONTROL_SHA
BASE_THETA, BASE_J, BASE_F2 = local.BASE_THETA, local.BASE_J, local.BASE_F2
ALPHA = 5.574488376651601e-6
RADIUS = 0.05
FACE_SCALE = 0.84
FACE = local.FACE
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "optimizer_steps": 1, "candidate_count": 1,
    "candidate_alpha": ALPHA, "radius": RADIUS, "jacobian_row_vjp_calls": 24,
    "dense_solves": 1, "hvp_calls": 2, "face_scale": FACE_SCALE,
    "candidate_grid_per_direction": 0, "max_candidates_per_iteration": 0,
    "root_claim": False, "minimum_claim": False,
    "score_claim": False, "response_claim": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("local-commit plan identity mismatch")
    prior = local._load_plan(local.PLAN, LOCAL_PLAN_SHA)
    added_archives = {
        local.PLAN.relative_to(ROOT).as_posix(), LOCAL_ARCHIVE.relative_to(ROOT).as_posix(),
        LOCAL_GZIP.relative_to(ROOT).as_posix(), LOCAL_RUN.relative_to(ROOT).as_posix(),
        LOCAL_RESOURCE.relative_to(ROOT).as_posix(),
    }
    plan = json.loads(path.read_text())
    expected_sources = set(prior["source_files"]) | {SELF, TEST}
    expected_archives = set(prior["archive_files"]) | added_archives
    if (plan.get("experiment_kind") != "one_fixed_gn_local_candidate_commit"
            or plan.get("policy") != POLICY or plan.get("face") != FACE
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA or plan.get("base_objective") != BASE_J
            or plan.get("base_F_squared") != BASE_F2 or plan.get("candidate_alpha") != ALPHA
            or plan.get("predecessor_plan") != local.PLAN.relative_to(ROOT).as_posix()
            or plan.get("predecessor_plan_sha256") != LOCAL_PLAN_SHA
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 148
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 190):
        raise ValueError("local-commit plan scope, base, or fixed policy changed")
    if (any(plan["source_files"].get(name) != value for name, value in prior["source_files"].items())
            or any(plan["archive_files"].get(name) != value for name, value in prior["archive_files"].items())):
        raise ValueError("local-commit plan changed an inherited local-window pin")
    for name, value in {**plan["source_files"], **plan["archive_files"]}.items():
        target = ROOT / name
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()) or _sha(target) != value:
            raise ValueError(f"local-commit source/archive pin mismatch: {name}")
    return plan


def _verify_latest_local_archive(plan: dict[str, Any]) -> dict[str, Any]:
    """Check that the frozen local-window diagnostic is still its latest closed run."""
    for path, digest in ((LOCAL_ARCHIVE, LOCAL_ARCHIVE_SHA), (LOCAL_GZIP, LOCAL_GZIP_SHA),
                         (LOCAL_RUN, LOCAL_RUN_SHA), (LOCAL_RESOURCE, LOCAL_RESOURCE_SHA)):
        if _sha(path) != digest:
            raise ValueError("latest GN local-window archive pin changed")
    manifest = json.loads(LOCAL_ARCHIVE.read_text())
    parent = json.loads(LOCAL_RUN.read_text())
    resource = json.loads(LOCAL_RESOURCE.read_text())
    raw_bytes = gzip.decompress(LOCAL_GZIP.read_bytes())
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    if (manifest.get("raw_sha256") != raw_sha or raw_sha != LOCAL_RAW_SHA
            or manifest.get("gzip_sha256") != LOCAL_GZIP_SHA
            or manifest.get("run_sha256") != LOCAL_RUN_SHA
            or manifest.get("resource_sha256") != LOCAL_RESOURCE_SHA
            or manifest.get("lossless_roundtrip") is not True
            or manifest.get("guarded_launch_count") != 1
            or parent.get("execution_status") != "completed"
            or parent.get("child_sha256") != raw_sha
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != 660.0
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes", POLICY["rss_bytes"]) >= POLICY["rss_bytes"]
            or raw.get("plan_sha256") != LOCAL_PLAN_SHA
            or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "local_window_diagnostic_completed_no_acceptance_or_step_selection"
            or raw.get("base_control_sha256") != BASE_CONTROL_SHA
            or raw.get("direction_sha256") != "0240d2e18fb8dfd4012bd7a0c8d3dd75f13cc12c81c5c941db9afafb3c3d6e94"
            or raw.get("hvp_calls_completed") != 2 or raw.get("candidate_samples_completed") != 8
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True or raw.get("optimizer_steps_applied") != 0):
        raise ValueError("latest frozen local-window archive does not close")
    candidate = next((item for item in raw["samples"] if item.get("alpha") == ALPHA), None)
    if (candidate is None or candidate.get("actual_J_armijo_passed") is not True
            or candidate.get("actual_F2_armijo_passed") is not True
            or candidate.get("control_sha256") != "311f27d095be4976ca6131b83b6c4f3d46bc404d4f58921800b5600e38ec0652"):
        raise ValueError("authorized fixed candidate is absent from the archived local window")
    trace_delta = candidate.get("trace_delta", {})
    if (set(trace_delta) != {"-1", "1"}
            or any(trace_delta[str(side)].get("choices_changed") is not False
                or trace_delta[str(side)].get("face_signs_changed") is not False
                or trace_delta[str(side)].get("nonfinite_or_tie") is not False
                for side in (-1, 1))):
        raise ValueError("authorized sample trace changed selectors/faces or contains ties")
    return {"raw_sha256": raw_sha, "candidate": candidate, "raw": raw}


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    tangent._write(path, value)


def _persist_closed_commit(output: Path, record: dict[str, Any],
                           proposal: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    if not tangent.fresh_final_closure(proposal, repeat):
        raise ValueError("independent P2 final closure failed; base endpoint remains current")
    control = torch.as_tensor(repeat["control"], dtype=torch.float64)
    control_sha = tangent._tensor_sha(control)
    committed = {**record, "phase": "committed", "execution_status": "running",
        "numerical_status": "candidate_committed_postcommit_gn_readiness_pending",
        "current_control": control.tolist(), "current_control_sha256": control_sha,
        "current_theta": float(repeat["theta"]), "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": control_sha, "last_confirmed_theta": float(repeat["theta"]),
        "last_confirmed_iterations": 1, "optimizer_steps_applied": 1,
        "accepted_iterations": 1, "candidate_committed": True,
        "proposal": proposal, "final_repeat": repeat, "last_confirmed_closure": repeat,
        "postcommit_candidate_alpha": ALPHA}
    _atomic_write(output, committed)
    return committed


def _persist_readiness_failure(output: Path, record: dict[str, Any], error: Exception) -> dict[str, Any]:
    failed = {**record, "phase": "finished", "execution_status": "completed",
        "numerical_status": "candidate_committed_gn_readiness_incomplete",
        "postcommit_readiness_error": f"{type(error).__name__}: {error}",
        "readiness_complete": False}
    _atomic_write(output, failed)
    return failed


def _require_postcommit_counts(record: dict[str, Any], control: Tensor,
                               direction: Tensor) -> None:
    direction_sha = tangent._tensor_sha(direction)
    control_sha = tangent._tensor_sha(control)
    history = record.get("hvp_history", [])
    started_history = record.get("hvp_started_history", [])
    row_history = record.get("jacobian_row_history", [])
    row_keys = {(item.get("side"), item.get("row")) for item in row_history}
    readiness = record.get("postcommit_gn_readiness", {})
    working_theta = readiness.get("working_theta")
    solve = record.get("dense_solve_audit", {})
    solve_direction = torch.as_tensor(solve.get("direction", []), dtype=control.dtype)
    readiness_direction = torch.as_tensor(readiness.get("direction", []), dtype=control.dtype)
    if (record.get("jacobian_rows_started") != 24 or record.get("jacobian_rows_completed") != 24
            or len(row_history) != 24
            or row_keys != {(side, row) for side in (-1, 1) for row in range(12)}
            or any(item.get("status") != "completed"
                or item.get("base_control_sha256") != control_sha
                or item.get("theta") != working_theta
                or not isinstance(item.get("residual_value"), (int, float))
                or not math.isfinite(float(item.get("residual_value", float("nan"))))
                or len(item.get("gradient", [])) != 26
                or not all(math.isfinite(float(value)) for value in item.get("gradient", []))
                for item in row_history)
            or record.get("dense_solves_started") != 1 or record.get("dense_solves_completed") != 1
            or solve.get("dense_solves") != 1 or solve.get("dense_solve_dimension") != 12
            or solve.get("S_dimension") != 12 or solve.get("positive_definite_from_cholesky") is not True
            or not isinstance(solve.get("solve_residual"), (int, float))
            or not isinstance(solve.get("solve_residual_budget"), (int, float))
            or not math.isfinite(float(solve.get("solve_residual", float("nan"))))
            or not math.isfinite(float(solve.get("solve_residual_budget", float("nan"))))
            or solve["solve_residual"] < 0 or solve["solve_residual_budget"] <= 0
            or solve["solve_residual"] > solve["solve_residual_budget"]
            or solve_direction.shape != control.shape or not bool(torch.isfinite(solve_direction).all())
            or not torch.equal(solve_direction, direction)
            or readiness.get("direction_model") != "robust_gn_coupled"
            or readiness.get("direction_sha256") != direction_sha
            or readiness_direction.shape != control.shape or not bool(torch.isfinite(readiness_direction).all())
            or not torch.equal(readiness_direction, direction)
            or record.get("hvp_calls_started") != 2 or record.get("hvp_calls_completed") != 2
            or len(history) != 2 or {item.get("side") for item in history} != {-1, 1}
            or any(item.get("base_control_sha256") != control_sha
                or item.get("direction_sha256") != direction_sha or item.get("status") != "completed"
                or item.get("theta") != record.get("current_theta")
                or item.get("working_theta") != working_theta
                or item.get("operator") != "selected_face_extension"
                or item.get("direction_model") != "robust_gn_coupled"
                for item in history)):
        raise ValueError("postcommit GN row/solve/HVP budgets or point-direction linkage failed")
    if (len(started_history) != 2 or {item.get("side") for item in started_history} != {-1, 1}
            or any(item.get("base_control_sha256") != control_sha
                or item.get("direction_sha256") != direction_sha
                or item.get("theta") != record.get("current_theta")
                or item.get("working_theta") != working_theta
                or item.get("operator") != "selected_face_extension"
                or item.get("direction_model") != "robust_gn_coupled"
                or item.get("phase") != "postcommit_current_point"
                for item in started_history)):
        raise ValueError("postcommit started HVP labels do not bind to the ready point and direction")


def _candidate_proposal(candidate: Tensor, observed: dict[str, Any], theta: float,
                        f_squared: float, alpha: float, j_bound: float,
                        f2_bound: float) -> dict[str, Any]:
    return {"control": candidate.tolist(), "control_sha256": tangent._tensor_sha(candidate),
        "theta": theta, "objective": float(observed["native_j"]), "F_squared": f_squared,
        "candidate_mixing_minimum": True, "alpha": alpha,
        "J_armijo_bound": j_bound, "J_armijo_passed": float(observed["native_j"]) <= j_bound,
        "F_squared_armijo_bound": f2_bound, "F_squared_armijo_passed": f_squared <= f2_bound,
        "face_audit_passed": observed["face_ok"], "branch_pair_passed": observed["pair"]["passed"],
        "side_objectives_match_native": observed["side_objectives_match_native"],
        "side_gradients_finite": observed["side_gradients_finite"],
        "side_gradients": {str(side): observed["side"][side][1].tolist() for side in (-1, 1)},
        "branch_trace": {str(side): observed["traces"][side] for side in (-1, 1)}}


def _final_repeat(shared_module: Any, geometry_module: Any, tangent_module: Any,
        plan: dict[str, Any], plan_path: Path, plan_sha: str, problem: Any,
        original: Tensor, parameters: Tensor, truth: Tensor, weights: Tensor,
        proposal: dict[str, Any], input_anchor: dict[str, Any], runtime_anchor: dict[str, Any],
        source_anchor: dict[str, Any], deadline: float) -> dict[str, Any]:
    control = torch.as_tensor(proposal["control"], dtype=torch.float64)
    observed = tangent_module._observe(shared_module, problem, control, parameters, weights)
    theta = float(tangent_module.minimum_mixture_weight(
        observed["side"][-1][1], observed["side"][1][1]))
    g = torch.cat(((1 - theta) * observed["side"][-1][1] + theta * observed["side"][1][1],
                   (observed["q"] / FACE_SCALE).reshape(1)))
    merit = float(torch.dot(g, g))
    native_match = abs(float(observed["native_j"]) - proposal["objective"]) <= 128 * torch.finfo(control.dtype).eps * max(
        abs(float(observed["native_j"])), abs(proposal["objective"]), torch.finfo(control.dtype).tiny)
    merit_match = abs(merit - proposal["F_squared"]) <= 128 * torch.finfo(control.dtype).eps * max(
        abs(merit), abs(proposal["F_squared"]), torch.finfo(control.dtype).tiny)
    theta_budget = 128 * torch.finfo(control.dtype).eps * max(abs(theta), abs(proposal["theta"]), torch.finfo(control.dtype).tiny)
    input_now = shared_module._input_identity(problem, original, control, parameters, truth)
    runtime_now = shared_module._runtime()
    source_now = shared_module._source_hashes(plan, plan_path, plan_sha)
    repeat = {"control": proposal["control"], "theta": theta,
        "objective": float(observed["native_j"]), "F_squared": merit,
        "side_gradients": {str(side): observed["side"][side][1].tolist() for side in (-1, 1)},
        "mixing_minimum_valid": True, "minimum_theta_matches_proposal": abs(theta - proposal["theta"]) <= theta_budget,
        "side_gradients_finite": observed["side_gradients_finite"],
        "native_objective_matches_proposal": native_match,
        "side_objectives_match_native": observed["side_objectives_match_native"],
        "merit_matches_proposal": merit_match, "face_audit_passed": observed["face_ok"],
        "branch_pair_passed": observed["pair"]["passed"],
        "trace_matches_proposal": all(observed["traces"][side]["signature_sha256"]
            == proposal["branch_trace"][str(side)]["signature_sha256"] for side in (-1, 1)),
        "branch_trace": {str(side): observed["traces"][side] for side in (-1, 1)},
        "fixed_input_unchanged": geometry_module.model._fixed_input(input_now, input_anchor,
            tangent_module._tensor_sha(control)),
        "runtime_unchanged": runtime_now == runtime_anchor,
        "source_unchanged": source_now == source_anchor,
        "deadline_passed": time.monotonic() < deadline}
    return repeat


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    start = time.monotonic()
    deadline = start + float(POLICY["internal_seconds"])
    plan = _load_plan(plan_path, plan_sha)
    plan_path = plan_path.resolve()
    source_before = shared._source_hashes(plan, plan_path, plan_sha)
    shared._deadline(deadline, POLICY["internal_seconds"])
    archive = _verify_latest_local_archive(plan)
    shared._deadline(deadline, POLICY["internal_seconds"])
    base = local._archive_identity()
    control, direction = base["control"], base["direction"]
    base_record = {"phase": "running", "execution_status": "running",
        "numerical_status": "fixed_candidate_not_yet_committed", "plan_sha256": plan_sha,
        "policy": POLICY, "current_control": control.tolist(),
        "current_control_sha256": BASE_CONTROL_SHA, "current_theta": BASE_THETA,
        "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": BASE_CONTROL_SHA, "last_confirmed_theta": BASE_THETA,
        "last_confirmed_iterations": 0, "optimizer_steps_applied": 0,
        "candidate_committed": False, "root_claim": False, "minimum_claim": False,
        "score_claim": False, "response_claim": False}
    _atomic_write(output, base_record)
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    shared._deadline(deadline, POLICY["internal_seconds"])
    input_before = shared._input_identity(problem, original, control, parameters, truth)
    runtime_before = shared._runtime()
    if (input_before != base["input_after"] or runtime_before != base["runtime_after"]
            or not geometry.model._fixed_input(input_before, base["input_after"], BASE_CONTROL_SHA)):
        raise ValueError("fixed input/runtime no longer matches the closed original endpoint")
    weights = geometry._face_weights(problem, axis=FACE["axis"], row=FACE["row"], column=FACE["column"])
    observed = tangent._observe(shared, problem, control, parameters, weights)
    shared._deadline(deadline, POLICY["internal_seconds"])
    closure = base["closure"]
    theta = float(tangent.minimum_mixture_weight(observed["side"][-1][1], observed["side"][1][1]))
    theta_budget = 128 * torch.finfo(control.dtype).eps * max(abs(theta), abs(BASE_THETA), torch.finfo(control.dtype).tiny)
    base_g = torch.cat(((1 - theta) * observed["side"][-1][1] + theta * observed["side"][1][1],
                        (observed["q"] / FACE_SCALE).reshape(1)))
    base_f2 = float(torch.dot(base_g, base_g))
    f2_budget = 128 * torch.finfo(control.dtype).eps * max(abs(base_f2), abs(BASE_F2), torch.finfo(control.dtype).tiny)
    if (not shared._gradient_pair_match({s: observed["side"][s][1] for s in (-1, 1)},
            {s: torch.as_tensor(closure["side_gradients"][str(s)], dtype=control.dtype) for s in (-1, 1)})
            or abs(theta - BASE_THETA) > theta_budget or abs(base_f2 - BASE_F2) > f2_budget
            or not observed["pair"]["passed"] or not observed["face_ok"]
            or not observed["side_objectives_match_native"] or not observed["side_gradients_finite"]
            or abs(float(observed["native_j"]) - BASE_J) > 128 * torch.finfo(control.dtype).eps * BASE_J
            or any(observed["traces"][s]["signature_sha256"] != closure["branch_trace"][str(s)]["signature_sha256"]
                   for s in (-1, 1))):
        raise ValueError("fresh base J, side-gradient pair, theta, trace, face, or input failed closure")
    pivot = geometry._pivot(control, weights)
    normal = geometry._face_normal(control, weights)
    _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    gm, gp = observed["side"][-1][1], observed["side"][1][1]
    model = mixing.minimum_tangent_model(gm, gp, base["stored_hminus"], base["stored_hplus"],
        normal, chart, theta, float(observed["q"]), pivot, FACE_SCALE, direction_override=direction)
    shared._deadline(deadline, POLICY["internal_seconds"])
    if (not all(model.gates.get(key) is True for key in (
            "finite", "both_side_gradients_descend", "merit_descends", "envelope_slope_matches_residual_dot"))
            or not local._scaled_error(model.residual, base["stored_residual"], model.residual)["passed"]
            or not local._scaled_error(model.residual_direction, base["stored_residual_direction"],
                                       model.residual_direction)["passed"]):
        raise ValueError("closed same-point minimum tangent model failed reconstruction")
    candidate = local._chart_candidate(control, weights, direction, pivot, ALPHA)
    displacement = float(torch.linalg.vector_norm(candidate - control))
    if not 0.0 < displacement <= RADIUS:
        raise ValueError("fixed candidate has zero displacement or exceeds the actual chart radius")
    trial = tangent._observe(shared, problem, candidate, parameters, weights)
    shared._deadline(deadline, POLICY["internal_seconds"])
    if any(trial["traces"][side]["signature_sha256"]
            != archive["candidate"]["trace_delta"][str(side)]["signature_sha256"]
            for side in (-1, 1)):
        raise ValueError("fresh fixed candidate branch trace differs from the approved archived sample")
    try:
        candidate_theta = float(tangent.minimum_mixture_weight(trial["side"][-1][1], trial["side"][1][1]))
    except ValueError as error:
        raise ValueError("fixed candidate has no interior mixing minimum") from error
    candidate_g = torch.cat(((1 - candidate_theta) * trial["side"][-1][1]
        + candidate_theta * trial["side"][1][1], (trial["q"] / FACE_SCALE).reshape(1)))
    candidate_f2 = float(torch.dot(candidate_g, candidate_g))
    archived_candidate = archive["candidate"]
    scalar_eps = 128 * torch.finfo(control.dtype).eps
    for actual, expected, label in ((candidate_theta, archived_candidate["theta_minimum"], "theta"),
            (float(trial["native_j"]), archived_candidate["actual_J"], "J"),
            (candidate_f2, archived_candidate["actual_F_squared"], "F-squared")):
        if abs(actual - expected) > scalar_eps * max(abs(actual), abs(expected), torch.finfo(control.dtype).tiny):
            raise ValueError(f"fresh fixed candidate {label} differs from the frozen sample")
    archived_g = torch.as_tensor(archived_candidate["G"], dtype=control.dtype)
    if not local._scaled_error(candidate_g, archived_g, archived_g)["passed"]:
        raise ValueError("fresh fixed candidate minimum residual differs from the frozen sample")
    candidate_sha = tangent._tensor_sha(candidate)
    if candidate_sha != archived_candidate["control_sha256"]:
        raise ValueError("fresh fixed candidate control differs from the archived authorized sample")
    j_slope = max(float(torch.dot(gm, direction)), float(torch.dot(gp, direction)))
    j_bound = float(observed["native_j"]) + 1e-4 * ALPHA * j_slope
    f2_bound = base_f2 + 2e-4 * ALPHA * model.merit_product
    j_pass = float(trial["native_j"]) <= j_bound
    f2_pass = candidate_f2 <= f2_bound
    required = (j_pass, f2_pass, trial["pair"]["passed"], trial["face_ok"],
        trial["side_objectives_match_native"], trial["side_gradients_finite"],
        math.isfinite(float(trial["native_j"])), math.isfinite(candidate_f2),
        bool(torch.isfinite(candidate_g).all()))
    if not all(required):
        raise ValueError("fresh fixed candidate failed original dual Armijo, pair, face, finite, or native guards")
    proposal = _candidate_proposal(candidate, trial, candidate_theta, candidate_f2,
        ALPHA, j_bound, f2_bound)
    source_after_proposal = shared._source_hashes(plan, plan_path, plan_sha)
    input_after_proposal = shared._input_identity(problem, original, control, parameters, truth)
    runtime_after_proposal = shared._runtime()
    if source_after_proposal != source_before or input_after_proposal != input_before or runtime_after_proposal != runtime_before:
        raise ValueError("source, fixed input, or runtime changed before independent final closure")
    repeat = _final_repeat(shared, geometry, tangent, plan, plan_path, plan_sha, problem,
        original, parameters, truth, weights, proposal, input_before, runtime_before,
        source_before, deadline)
    shared._deadline(deadline, POLICY["internal_seconds"])
    new_control = torch.as_tensor(repeat["control"], dtype=control.dtype)
    new_sha = tangent._tensor_sha(new_control)
    record: dict[str, Any] = {
        **base_record, "plan_sha256": plan_sha, "policy": POLICY,
        "source_before": source_before, "source_after": source_before,
        "input_before": input_before, "runtime_before": runtime_before,
        "base_control_sha256": BASE_CONTROL_SHA, "base_theta": BASE_THETA,
        "base_objective": BASE_J, "base_F_squared": BASE_F2,
        "archive_identity": {"local_window_plan_sha256": LOCAL_PLAN_SHA,
            "local_window_child_sha256": archive["raw_sha256"],
            "reused_same_point_hvp_vectors": 2, "fresh_base_hvp_calls": 0,
            "fresh_postcommit_hvp_budget": 2},
        "hvp_calls_started": 0, "hvp_calls_completed": 0,
        "jacobian_rows_started": 0, "jacobian_rows_completed": 0,
        "dense_solves_started": 0, "dense_solves_completed": 0,
        "minimum_claim": False, "score_claim": False, "response_claim": False,
    }
    record = _persist_closed_commit(output, record, proposal, repeat)

    # This work is intentionally after the durable commit. Any failure below
    # leaves the accepted point and P2 receipt intact for a later continuation.
    try:
        _build_current_gn_readiness(record, output, plan, plan_path, plan_sha,
            problem, new_control, parameters, weights, original, truth, runtime_before,
            input_before, source_before, deadline, start)
    except Exception as error:
        record = _persist_readiness_failure(output, record, error)
    return record


def _build_current_gn_readiness(record: dict[str, Any], output: Path, plan: dict[str, Any],
        plan_path: Path, plan_sha: str, problem: Any, control: Tensor, parameters: Tensor,
        weights: Tensor, original: Tensor, truth: Tensor, runtime_anchor: dict[str, Any],
        input_anchor: dict[str, Any], source_anchor: dict[str, Any], deadline: float,
        start: float) -> None:
    shared._deadline(deadline, POLICY["internal_seconds"])
    observed = tangent._observe(shared, problem, control, parameters, weights)
    theta = float(record["current_theta"])
    closure = record["final_repeat"]
    reference_j = float(closure["objective"])
    jtol = 128 * torch.finfo(control.dtype).eps * max(
        abs(float(observed["native_j"])), abs(reference_j), torch.finfo(control.dtype).tiny)
    if (abs(float(observed["native_j"]) - reference_j) > jtol
            or not shared._gradient_pair_match({s: observed["side"][s][1] for s in (-1, 1)},
                {s: torch.as_tensor(closure["side_gradients"][str(s)], dtype=control.dtype) for s in (-1, 1)})
            or any(observed["traces"][s]["signature_sha256"] != closure["branch_trace"][str(s)]["signature_sha256"]
                for s in (-1, 1))
            or not observed["pair"]["passed"] or not observed["face_ok"]
            or not observed["side_objectives_match_native"] or not observed["side_gradients_finite"]):
        raise ValueError("fresh current-point observation fails committed candidate closure")
    working_theta = float(tangent.minimum_mixture_weight(
        observed["side"][-1][1], observed["side"][1][1]))
    working_budget = 128 * torch.finfo(control.dtype).eps * max(
        abs(working_theta), abs(float(closure["theta"])), torch.finfo(control.dtype).tiny)
    current_residual = torch.cat(((1 - working_theta) * observed["side"][-1][1]
        + working_theta * observed["side"][1][1], (observed["q"] / FACE_SCALE).reshape(1)))
    current_f2 = float(torch.dot(current_residual, current_residual))
    f2_budget = 128 * torch.finfo(control.dtype).eps * max(
        abs(current_f2), abs(float(closure["F_squared"])), torch.finfo(control.dtype).tiny)
    if (abs(working_theta - float(closure["theta"])) > working_budget
            or abs(current_f2 - float(closure["F_squared"])) > f2_budget):
        raise ValueError("fresh current-point mixing minimum or merit differs from committed closure")
    pivot, normal = geometry._pivot(control, weights), geometry._face_normal(control, weights)
    _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    record["hvp_history"] = []
    record["jacobian_row_history"] = []
    model_state = {"observed": observed, "working_theta": working_theta}
    factory = gn_resume._resume_direction_model_factory({"problem": problem,
        "parameters": parameters, "record": record, "output": output, "deadline": deadline,
        "weights": weights, "plan": plan, "plan_path": plan_path, "plan_sha": plan_sha,
        "model_state": model_state})
    specs = factory(control, theta, observed["side"][-1][1], observed["side"][1][1],
        normal, chart, pivot)
    if len(specs) != 1 or specs[0].get("name") != "robust_gn_coupled":
        raise ValueError("current-point GN factory did not return the single coupled model")
    spec = specs[0]
    direction = spec["direction"]
    record["postcommit_gn_readiness_direction_sha256"] = tangent._tensor_sha(direction)
    grad_fn = torch.func.grad(problem.objective, argnums=0)
    hvps: dict[int, Tensor] = {}
    for side in (-1, 1):
        if record["hvp_calls_started"] >= POLICY["hvp_calls"]:
            raise ValueError("postcommit HVP budget exhausted")
        label = {"phase": "postcommit_current_point", "side": side,
            "base_control_sha256": tangent._tensor_sha(control), "theta": theta,
            "working_theta": float(spec["working_theta"]),
            "direction_sha256": tangent._tensor_sha(direction),
            "operator": "selected_face_extension", "scope": "fresh committed-point GN tangent",
            "direction_model": "robust_gn_coupled"}
        record["hvp_calls_started"] += 1
        record.setdefault("hvp_started_history", []).append(label)
        _atomic_write(output, record)
        shared._deadline(deadline, POLICY["internal_seconds"])
        def calculate(sign: int = side) -> Tensor:
            with shared.transport.selected_face_extension(FACE["axis"], FACE["row"], FACE["column"], sign):
                return torch.func.jvp(lambda point: grad_fn(point, parameters), (control,), (direction,))[1]
        value = calculate()
        if value.shape != control.shape or not bool(torch.isfinite(value).all()):
            raise ValueError("postcommit selected-face HVP is nonfinite or has the wrong shape")
        hvps[side] = value
        record["hvp_calls_completed"] += 1
        record["hvp_history"].append({"base_control_sha256": tangent._tensor_sha(control),
            "direction_sha256": tangent._tensor_sha(direction), "theta": theta,
            "working_theta": float(spec["working_theta"]),
            "direction_model": "robust_gn_coupled", "operator": "selected_face_extension",
            "side": side, "status": "completed"})
        _atomic_write(output, record)
    model = spec["model_builder"](observed["side"][-1][1], observed["side"][1][1],
        hvps[-1], hvps[1], normal, chart, theta, float(observed["q"]), pivot)
    if (not bool(torch.isfinite(direction).all())
            or not all(model.gates.get(key) is True for key in (
                "finite", "both_side_gradients_descend", "merit_descends", "envelope_slope_matches_residual_dot"))):
        raise ValueError("fresh current-point coupled GN model failed numerical gates")
    source_after = shared._source_hashes(plan, plan_path, plan_sha)
    input_after = shared._input_identity(problem, original, control, parameters, truth)
    runtime_after = shared._runtime()
    source_ok = source_after == source_anchor
    input_ok = geometry.model._fixed_input(input_after, input_anchor, tangent._tensor_sha(control))
    runtime_ok = runtime_after == runtime_anchor
    deadline_ok = time.monotonic() < deadline
    record.update({"postcommit_current_observation": {
            "control_sha256": tangent._tensor_sha(control), "reference_theta": theta,
            "working_theta": float(spec["working_theta"]),
            "native_objective": float(observed["native_j"]),
            "F_squared": float(torch.dot(model.residual, model.residual)),
            "branch_pair_passed": observed["pair"]["passed"], "face_audit_passed": observed["face_ok"]},
        "postcommit_gn_readiness": {"direction": direction.tolist(),
            "working_theta": float(spec["working_theta"]),
            "direction_sha256": tangent._tensor_sha(direction), "direction_model": spec["name"],
            "residual": model.residual.tolist(), "residual_direction": model.residual_direction.tolist(),
            "hminus": hvps[-1].tolist(), "hplus": hvps[1].tolist(), "gates": model.gates,
            "diagnostics": spec["diagnostics"], "candidate_count": 0,
            "optimizer_steps_after_commit": 0},
        "runtime_after": runtime_after, "input_after": input_after,
        "source_after": source_after, "source_unchanged": source_ok,
        "fixed_input_unchanged": input_ok, "runtime_unchanged": runtime_ok,
        "deadline_passed": deadline_ok,
        "elapsed_seconds": float(time.monotonic() - start)})
    _atomic_write(output, record)
    _require_postcommit_counts(record, control, direction)
    if not all((source_ok, input_ok, runtime_ok, deadline_ok)):
        raise ValueError("postcommit source/input/runtime/deadline closure failed")
    record.update(phase="finished", execution_status="completed",
        numerical_status="candidate_committed_current_gn_model_ready_no_second_candidate",
        readiness_complete=True)
    _atomic_write(output, record)


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    _load_plan(plan_path, plan_sha)
    plan_path = plan_path.resolve()
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, output.with_suffix(".run.json"))):
        raise ValueError("local-commit output paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
        "--plan", str(plan_path), "--plan-sha256", plan_sha, "--output", str(output)]
    resource_result = run_guarded_diagnostic(command, wall_seconds=POLICY["outer_seconds"],
        rss_bytes=POLICY["rss_bytes"], report_path=resource, log_path=log)
    parent: dict[str, Any] = {"execution_status": shared._execution_status(resource_result,
        wall_seconds=POLICY["outer_seconds"], rss_bytes=POLICY["rss_bytes"]),
        "resource": resource_result, "child_sha256": _sha(output) if output.exists() else None,
        "numerical_status": "not_reached"}
    if output.exists():
        child = json.loads(output.read_text())
        parent["numerical_status"] = child.get("numerical_status")
        parent["candidate_committed"] = child.get("candidate_committed") is True
        parent["readiness_complete"] = child.get("readiness_complete") is True
    shared._write(output.with_suffix(".run.json"), parent)
    return {"parent": parent}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "step.json")
    parser.add_argument("--resource", type=Path, default=OUTPUT_DIR / "step.resource.json")
    parser.add_argument("--log", type=Path, default=OUTPUT_DIR / "step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child_impl(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
