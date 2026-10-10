"""Compare one prepared GN direction with its limiter-event-orthogonal form.

This bounded experiment starts from the closed 027 endpoint. It reuses the
saved PR279 GN direction and PR280 event-gradient receipt, recomputes two
selected-face HVPs for each arm, searches both arms independently, and commits
at most the best first-pass candidate after one independent P2 repeat.
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
from typing import Any, Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared
from examples.weather_scenarios import fv_point_3h_limiter_event_diagnostic as event_diag
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "EVENT_DIRECTION_COMPARISON_PLAN_20261010.json"
OUTPUT_DIR = EVIDENCE / "event_direction_comparison_20261010_attempt1"
SELF = "examples/weather_scenarios/fv_point_3h_event_direction_comparison.py"
TEST = "tests/test_fv_point_3h_event_direction_comparison.py"
EVENT_PLAN_SHA = "3579c2bbf9f3213cec5669948e9de21e47da258720572b291b083c5355e3f79d"
BASE_SHA = "027e83cf77f7f9cf97c0c72da9f737b5627d312d6375b698bb5b7f7ae94c6769"
GN_DIRECTION_SHA = "ab3a2635cedc7f90c9cead68e7ce9bae9b33a15cabf5ffe4970a9b00c91fea74"
EVENT_ARCHIVE = EVIDENCE / "LIMITER_EVENT_ARCHIVE_ATTEMPT2_20261010.json"
EVENT_GZIP = EVIDENCE / "limiter_event_20261010_attempt2/diagnostic.json.gz"
EVENT_RUN = EVIDENCE / "limiter_event_20261010_attempt2/diagnostic.run.json"
EVENT_RESOURCE = EVIDENCE / "limiter_event_20261010_attempt2/diagnostic.resource.json"
EVENT_AUDIT = EVIDENCE / "LIMITER_EVENT_SAVED_AUDIT_20261010.json"
GN_AUDIT = EVIDENCE / "GN_PREPARED_SEARCH_SAVED_AUDIT_20261010.json"
EVENT_ARCHIVE_SHA = "b2fa1593c625853beb2ed110b33447ffade50c46743ff185f907a0ff65318427"
EVENT_GZIP_SHA = "4b0dbb02674f99e8f21d0fa0bf813b61509c4e05bfb82e03b0f853e0ac409b00"
EVENT_RUN_SHA = "ee4fc5f8c630cc9b73142c22eb36d6c7f7fa75be7556b949f39947e0845dbc35"
EVENT_RESOURCE_SHA = "c0a29a3836265badc4bacdc8913db3ff3bde84cf2519d770ca3edb3b57230738"
EVENT_AUDIT_SHA = "d6ce8981c3555878884195b0a9faa4abcffd783608b637cd1072c6c19134ad61"
GN_AUDIT_SHA = "0d762c7b41f9f30981069b5deeca8b5580b2f32efabdd4e753c316656f4305f7"
EVENT_RAW_SHA = "47b6b78d824a6ee817d2d014ea74ce14f68076d68c8c75a38ae6a0a7657297ff"
FACE_AXIS, FACE_ROW, FACE_COLUMN = "y", 4, 3
FACE = {"axis": FACE_AXIS, "row": FACE_ROW, "column": FACE_COLUMN}
FACE_SCALE = 0.84
RADIUS = 0.05
MAX_CANDIDATES_PER_ARM = 24
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "candidate_count": 48,
    "candidate_cap_per_arm": MAX_CANDIDATES_PER_ARM, "max_commits": 1,
    "optimizer_steps": 1, "radius": RADIUS, "jacobian_row_vjp_calls": 0,
    "dense_solves": 0, "hvp_calls": 4, "face_scale": FACE_SCALE,
    "selection_tie_rule": "128eps_relative_R_then_J_then_prepared_gn_order",
    "root_claim": False, "minimum_claim": False, "score_claim": False,
    "response_claim": False,
}
ARM_ORDER = ("prepared_gn", "event_orthogonal")


def _sha(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"pinned path must not be a symbolic link: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned path escapes repository: {path}")
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("event-direction comparison plan identity mismatch")
    prior = event_diag._load_plan(event_diag.PLAN, EVENT_PLAN_SHA)
    add_sources = {SELF, TEST}
    add_archives = {
        event_diag.PLAN.relative_to(ROOT).as_posix(),
        EVENT_ARCHIVE.relative_to(ROOT).as_posix(), EVENT_GZIP.relative_to(ROOT).as_posix(),
        EVENT_RUN.relative_to(ROOT).as_posix(), EVENT_RESOURCE.relative_to(ROOT).as_posix(),
        EVENT_AUDIT.relative_to(ROOT).as_posix(), GN_AUDIT.relative_to(ROOT).as_posix(),
    }
    plan = json.loads(path.read_text())
    expected_sources = set(prior["source_files"]) | add_sources
    expected_archives = set(prior["archive_files"]) | add_archives
    if (plan.get("experiment_kind") != "event_orthogonal_direction_comparison"
            or plan.get("policy") != POLICY or plan.get("face") != FACE
            or plan.get("event_plan") != event_diag.PLAN.relative_to(ROOT).as_posix()
            or plan.get("event_plan_sha256") != EVENT_PLAN_SHA
            or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("gn_direction_sha256") != GN_DIRECTION_SHA
            or plan.get("event_archive_sha256") != _sha(EVENT_ARCHIVE)
            or plan.get("radius") != RADIUS
            or plan.get("candidate_cap_per_arm") != MAX_CANDIDATES_PER_ARM
            or plan.get("predecessor_plan_sha256") != EVENT_PLAN_SHA
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 156
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 207):
        raise ValueError("event-direction comparison plan scope or fixed policy changed")
    for key in ("source_files", "archive_files"):
        if any(plan[key].get(name) != value for name, value in prior[key].items()):
            raise ValueError(f"comparison plan changed inherited {key} pins")
        for name, value in plan[key].items():
            if _sha(ROOT / name) != value:
                raise ValueError(f"comparison {key} pin mismatch: {name}")
    return plan


def _load_event_receipt() -> dict[str, Any]:
    for path, digest in ((EVENT_ARCHIVE, EVENT_ARCHIVE_SHA), (EVENT_GZIP, EVENT_GZIP_SHA),
            (EVENT_RUN, EVENT_RUN_SHA), (EVENT_RESOURCE, EVENT_RESOURCE_SHA),
            (EVENT_AUDIT, EVENT_AUDIT_SHA), (GN_AUDIT, GN_AUDIT_SHA)):
        if _sha(path) != digest:
            raise ValueError(f"successful PR280/PR279 evidence changed: {path.name}")
    manifest = json.loads(EVENT_ARCHIVE.read_text())
    run = json.loads(EVENT_RUN.read_text())
    resource = json.loads(EVENT_RESOURCE.read_text())
    audit = json.loads(EVENT_AUDIT.read_text())
    gn_audit = json.loads(GN_AUDIT.read_text())
    compressed = EVENT_GZIP.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    if (raw_sha != EVENT_RAW_SHA or manifest.get("raw_sha256") != raw_sha
            or manifest.get("gzip_sha256") != EVENT_GZIP_SHA
            or manifest.get("run_sha256") != EVENT_RUN_SHA
            or manifest.get("resource_sha256") != EVENT_RESOURCE_SHA
            or manifest.get("lossless_roundtrip") is not True
            or manifest.get("execution_status") != "completed"
            or run.get("execution_status") != "completed" or run.get("child_sha256") != raw_sha
            or run.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False or resource.get("wall_limit_seconds") != 660.0
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes", POLICY["rss_bytes"]) >= POLICY["rss_bytes"]
            or raw.get("plan_sha256") != EVENT_PLAN_SHA
            or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "limiter_event_diagnostic_complete_no_commit"
            or raw.get("base_control_sha256") != BASE_SHA
            or raw.get("direction_sha256") != GN_DIRECTION_SHA
            or raw.get("optimizer_steps_applied") != 0 or raw.get("fresh_hvp_calls") != 0
            or raw.get("event_grad_completed") != 2 or raw.get("event_jvp_completed") != 2
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True
            or not audit.get("all_checks_passed") or audit.get("plan_sha256") != EVENT_PLAN_SHA
            or not gn_audit.get("all_checks_passed")
            or gn_audit.get("plan_sha256") != event_diag.PRIOR_PLAN_SHA):
        raise ValueError("PR280 event or PR279 prepared-search evidence failed closure")
    return {"raw": raw, "raw_sha256": raw_sha, "audit": audit,
        "gn_audit": gn_audit, "resource": resource}


def _scaled_error(actual: Tensor, expected: Tensor, scale: Tensor) -> dict[str, float | bool]:
    return prepared.local._scaled_error(actual, expected, scale)


def _project_away(direction: Tensor, event_tangent_gradient: Tensor) -> Tensor:
    """Euclidean projection that removes only the event-normal component."""
    if (direction.ndim != 1 or event_tangent_gradient.shape != direction.shape
            or direction.numel() != 26 or direction.dtype != event_tangent_gradient.dtype
            or not bool(torch.isfinite(direction).all() & torch.isfinite(event_tangent_gradient).all())):
        raise ValueError("direction and event gradient must be finite matching 26-vectors")
    denominator = torch.dot(event_tangent_gradient, event_tangent_gradient)
    if not bool(torch.isfinite(denominator) & (denominator > torch.finfo(direction.dtype).tiny)):
        raise ValueError("event tangent gradient has unresolved Euclidean norm")
    projected = direction - event_tangent_gradient * (
        torch.dot(event_tangent_gradient, direction) / denominator)
    if not bool(torch.linalg.vector_norm(projected) > torch.finfo(direction.dtype).tiny):
        raise ValueError("event projection leaves a zero direction")
    return projected


def _common_event_tangent_gradient(event_raw: dict[str, Any], normal: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Reproject both saved side gradients and average only after component checks."""
    derivatives = event_raw.get("event_derivatives", {})
    projected_by_side: dict[int, Tensor] = {}
    saved_by_side: dict[int, Tensor] = {}
    checks: dict[str, Any] = {}
    nn = torch.dot(normal, normal)
    if not bool(torch.isfinite(nn) & (nn > torch.finfo(normal.dtype).tiny)):
        raise ValueError("current selected-face normal has unresolved norm")
    for side in (-1, 1):
        entry = derivatives.get(str(side))
        if not isinstance(entry, dict):
            raise ValueError("successful event receipt is missing a side gradient")
        gradient = torch.as_tensor(entry["gradient"], dtype=normal.dtype)
        saved = torch.as_tensor(entry["tangent_gradient"], dtype=normal.dtype)
        if gradient.shape != normal.shape or saved.shape != normal.shape:
            raise ValueError("saved event gradient has wrong dimension")
        projected = gradient - normal * (torch.dot(normal, gradient) / nn)
        replay_check = _scaled_error(projected, saved, saved)
        if replay_check["passed"] is not True:
            raise ValueError(f"saved event tangent projection mismatch on side {side}")
        projected_by_side[side], saved_by_side[side] = projected, saved
        checks[str(side)] = {"recomputed_projection": projected.tolist(),
            "saved_projection": saved.tolist(), "component_scaled_match": replay_check}
    average = 0.5 * (projected_by_side[-1] + projected_by_side[1])
    common_check = _scaled_error(projected_by_side[-1], projected_by_side[1], average)
    if common_check["passed"] is not True:
        raise ValueError("one-sided event tangent gradients do not share a componentwise common normal")
    checks["side_projection_agreement"] = common_check
    checks["averaging_rule"] = "arithmetic mean of componentwise-agreeing freshly reprojected one-sided gradients"
    return average, checks


def _armijo_bounds(base_j: float, base_r: float, alpha: float,
        side_cost_slopes: tuple[float, float], merit_product: float) -> tuple[float, float]:
    cost_slope = max(side_cost_slopes)
    return base_j + 1e-4 * alpha * cost_slope, base_r + 2e-4 * alpha * merit_product


def _select_arm_candidate(results: dict[str, dict[str, Any]],
                          dtype: torch.dtype) -> dict[str, Any] | None:
    """Choose lowest actual R, then J within roundoff ties, then arm order."""
    supported = [name for name in ARM_ORDER
        if isinstance(results.get(name), dict) and results[name].get("supported", True) is True]
    choices = [(name, value["accepted"]) for name, value in results.items()
        if isinstance(value.get("accepted"), dict)]
    if not choices:
        return None
    winner_name, winner = choices[0]
    for name, candidate in choices[1:]:
        r, best_r = float(candidate["F_squared"]), float(winner["F_squared"])
        scale = max(abs(r), abs(best_r), torch.finfo(dtype).tiny)
        r_tie = abs(r - best_r) <= 128 * torch.finfo(dtype).eps * scale
        if r < best_r and not r_tie:
            winner_name, winner = name, candidate
            continue
        if r_tie:
            j, best_j = float(candidate["objective"]), float(winner["objective"])
            j_scale = max(abs(j), abs(best_j), torch.finfo(dtype).tiny)
            j_tie = abs(j - best_j) <= 128 * torch.finfo(dtype).eps * j_scale
            if j < best_j and not j_tie:
                winner_name, winner = name, candidate
            elif j_tie and ARM_ORDER.index(name) < ARM_ORDER.index(winner_name):
                winner_name, winner = name, candidate
    complete = all(name in supported for name in ARM_ORDER)
    return {"arm": winner_name, "proposal": winner,
        "supported_arms": supported, "comparison_complete": complete,
        "selection_basis": "two_supported_arm_comparison" if complete
            else "single_supported_arm_fallback"}


def _per_arm_first_pass_counts(results: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {name: int(results.get(name, {}).get("search", {}).get("first_pass_count", 0))
        for name in ARM_ORDER}


def _event_plain_values(problem: Any, control: Tensor, parameters: Tensor,
                        side: int) -> Tensor:
    return event_diag._plain_values(problem, control, parameters, side)


def _persist_selected_commit(output: Path, record: dict[str, Any],
        proposal: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    if not tangent.fresh_final_closure(proposal, repeat):
        raise ValueError("selected candidate failed independent P2; original 027 point remains current")
    control = torch.as_tensor(repeat["control"], dtype=torch.float64)
    sha = tangent._tensor_sha(control)
    committed = {**record, "phase": "committed", "execution_status": "running",
        "numerical_status": "event_direction_candidate_committed_no_new_point_readiness",
        "current_control": control.tolist(), "current_control_sha256": sha,
        "current_theta": float(repeat["theta"]), "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": sha, "last_confirmed_theta": float(repeat["theta"]),
        "last_confirmed_iterations": 1, "optimizer_steps_applied": 1,
        "accepted_iterations": 1, "candidate_committed": True,
        "proposal": proposal, "final_repeat": repeat, "last_confirmed_closure": repeat,
        "postcommit_candidate_alpha": float(proposal["alpha"]),
        "postcommit_readiness_performed": False}
    tangent._write(output, committed)
    return committed


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    start = time.monotonic()
    deadline = start + POLICY["internal_seconds"]
    check_deadline = lambda: shared._deadline(deadline, POLICY["internal_seconds"])
    check_deadline()
    plan = _load_plan(plan_path, plan_sha)
    check_deadline()
    event_plan = event_diag._load_plan(event_diag.PLAN, EVENT_PLAN_SHA)
    check_deadline()
    plan_path = plan_path.resolve()
    source_before = shared._source_hashes(plan, plan_path, plan_sha)
    check_deadline()
    event_receipt = _load_event_receipt()
    check_deadline()
    parent = event_diag._load_base()
    check_deadline()
    control = torch.as_tensor(parent["current_control"], dtype=torch.float64)
    baseline_direction = torch.as_tensor(parent["postcommit_gn_readiness"]["direction"], dtype=control.dtype)
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "event_direction_comparison_preflight", "policy": POLICY,
        "plan_sha256": plan_sha, "base_control_sha256": BASE_SHA,
        "current_control": control.tolist(), "current_control_sha256": BASE_SHA,
        "current_theta": float(parent["current_theta"]), "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": BASE_SHA, "last_confirmed_theta": float(parent["current_theta"]),
        "last_confirmed_iterations": 0, "prior_confirmed_commits": 1,
        "candidate_committed": False, "optimizer_steps_applied": 0,
        "hvp_calls_started": 0, "hvp_calls_completed": 0, "hvp_started_history": [],
        "hvp_history": [], "jacobian_rows_started": 0, "jacobian_rows_completed": 0,
        "dense_solves_started": 0, "dense_solves_completed": 0,
        "root_claim": False, "minimum_claim": False, "score_claim": False, "response_claim": False}
    tangent._write(output, record)
    check_deadline()
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    check_deadline()
    input_before = shared._input_identity(problem, original, control, parameters, truth)
    runtime_before = shared._runtime()
    event_raw = event_receipt["raw"]
    if (input_before != parent["input_after"] or runtime_before != parent["runtime_after"]
            or input_before != event_raw["input_after"] or runtime_before != event_raw["runtime_after"]
            or tangent._tensor_sha(control) != BASE_SHA
            or tangent._tensor_sha(baseline_direction) != GN_DIRECTION_SHA):
        raise ValueError("fresh input/runtime/control/direction differs from frozen 027 point")
    event_source = shared._source_hashes(event_plan, event_diag.PLAN, EVENT_PLAN_SHA)
    if event_source != event_raw["source_after"]:
        raise ValueError("successful event receipt source scope no longer closes")
    weights = geometry._face_weights(problem, axis=FACE_AXIS, row=FACE_ROW, column=FACE_COLUMN)
    check_deadline()
    observed = tangent._observe(shared, problem, control, parameters, weights)
    check_deadline()
    closure = parent["final_repeat"]
    theta = float(tangent.minimum_mixture_weight(observed["side"][-1][1], observed["side"][1][1]))
    G = torch.cat(((1 - theta) * observed["side"][-1][1] + theta * observed["side"][1][1],
        (observed["q"] / FACE_SCALE).reshape(1)))
    R = float(torch.dot(G, G))
    eps = 128 * torch.finfo(control.dtype).eps
    if (not shared._gradient_pair_match({s: observed["side"][s][1] for s in (-1, 1)},
            {s: torch.as_tensor(closure["side_gradients"][str(s)], dtype=control.dtype) for s in (-1, 1)})
            or not observed["pair"]["passed"] or not observed["face_ok"]
            or not observed["side_objectives_match_native"] or not observed["side_gradients_finite"]
            or abs(float(observed["native_j"]) - float(closure["objective"])) > eps * abs(float(closure["objective"]))
            or abs(theta - float(closure["theta"])) > eps * abs(float(closure["theta"]))
            or abs(R - float(closure["F_squared"])) > eps * abs(float(closure["F_squared"]))
            or abs(float(observed["native_j"]) - float(event_raw["base_J"])) > eps * abs(float(event_raw["base_J"]))
            or abs(R - float(event_raw["base_R"])) > eps * abs(float(event_raw["base_R"]))
            or abs(theta - float(event_raw["base_theta"])) > eps * abs(float(event_raw["base_theta"]))
            or any(observed["traces"][s]["signature_sha256"] != closure["branch_trace"][str(s)]["signature_sha256"]
                for s in (-1, 1))):
        raise ValueError("fresh J/G/theta/trace/input qualification failed at 027")
    fresh_event_values: dict[str, list[float]] = {}
    for side in (-1, 1):
        check_deadline()
        actual_event = _event_plain_values(problem, control, parameters, side)
        check_deadline()
        saved_event = torch.as_tensor(event_raw["event_derivatives"][str(side)]["original_values"],
            dtype=control.dtype)
        event_check = _scaled_error(actual_event, saved_event, saved_event)
        if event_check["passed"] is not True:
            raise ValueError(f"fresh base event values differ from PR280 side {side}")
        if not bool(torch.isfinite(actual_event).all() & (actual_event[2] > 0)):
            raise ValueError("fresh base event values or q scale are nonfinite/unresolved")
        if not bool((actual_event[:2] < 0).all() & (actual_event[0] < actual_event[1])):
            raise ValueError("fresh base no longer matches the qualified same-negative limiter event")
        fresh_event_values[str(side)] = actual_event.tolist()
    pivot = geometry._pivot(control, weights)
    normal = geometry._face_normal(control, weights)
    _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    normal_norm = torch.linalg.vector_norm(normal)
    tangent_tolerance = 128 * torch.finfo(control.dtype).eps * normal_norm * torch.linalg.vector_norm(baseline_direction)
    if abs(float(torch.dot(normal, baseline_direction))) > float(tangent_tolerance):
        raise ValueError("prepared GN direction is not tangent to the selected face")

    event_normal, event_gradient_checks = _common_event_tangent_gradient(event_raw, normal)
    projection_error: str | None = None
    try:
        projected_direction = _project_away(baseline_direction, event_normal)
    except ValueError as error:
        projection_error = str(error)
        projected_direction = torch.zeros_like(baseline_direction)
    if not bool(torch.isfinite(projected_direction).all()):
        projection_error = "event-orthogonal GN direction is nonfinite"
        projected_direction = torch.zeros_like(baseline_direction)
    orthogonality = float(torch.dot(event_normal, projected_direction))
    orthogonality_budget = 128 * torch.finfo(control.dtype).eps * float(
        torch.linalg.vector_norm(event_normal) * torch.linalg.vector_norm(projected_direction))
    if projection_error is None and abs(orthogonality) > orthogonality_budget:
        raise ValueError("Euclidean event projection does not remove the event-normal component")
    projected_face_dot = float(torch.dot(normal, projected_direction))
    projected_face_budget = 128 * torch.finfo(control.dtype).eps * float(
        torch.linalg.vector_norm(normal) * torch.linalg.vector_norm(projected_direction))
    if projection_error is None and abs(projected_face_dot) > projected_face_budget:
        raise ValueError("event-orthogonal direction left the selected-face tangent space")
    directions = {"prepared_gn": baseline_direction, "event_orthogonal": projected_direction}
    source_anchor = source_before
    record.update({"source_before": source_before, "input_before": input_before,
        "runtime_before": runtime_before, "base_J": float(observed["native_j"]),
        "base_R": R, "base_theta": theta, "base_residual": G.tolist(),
        "fresh_base_event_values": fresh_event_values,
        "event_receipt_raw_sha256": event_receipt["raw_sha256"],
        "event_plan_sha256": EVENT_PLAN_SHA, "event_gradient_rule": event_gradient_checks,
        "event_tangent_gradient": event_normal.tolist(),
        "direction_projection": {"baseline_direction": baseline_direction.tolist(),
            "event_orthogonal_direction": projected_direction.tolist(),
            "baseline_event_dot": float(torch.dot(event_normal, baseline_direction)),
            "projected_event_dot": orthogonality, "orthogonality_budget": orthogonality_budget,
            "projected_face_dot": projected_face_dot,
            "projected_face_budget": projected_face_budget,
            "baseline_l2": float(torch.linalg.vector_norm(baseline_direction)),
            "projected_l2": float(torch.linalg.vector_norm(projected_direction)),
            "normalized": False, "projection_refusal": projection_error},
        "arms": {}, "selection": None})
    grad_fn = torch.func.grad(problem.objective, argnums=0)

    for arm_name in ARM_ORDER:
        direction = directions[arm_name]
        arm_record: dict[str, Any] = {"direction": direction.tolist(),
            "direction_sha256": tangent._tensor_sha(direction),
            "hvp_calls_started": 0, "hvp_calls_completed": 0,
            "hminus": None, "hplus": None, "trials": [], "search": {},
            "accepted": None, "accepted_event_values": None,
            "supported": True, "unsupported_reason": None}
        record["arms"][arm_name] = arm_record
        if arm_name == "event_orthogonal" and projection_error is not None:
            arm_record.update(supported=False, unsupported_reason=projection_error)
            tangent._write(output, record)
            continue
        hvps: dict[int, Tensor] = {}
        hvp_error: str | None = None
        for side in (-1, 1):
            check_deadline()
            if record["hvp_calls_started"] >= POLICY["hvp_calls"]:
                raise ValueError("four-call comparison HVP budget exhausted")
            label = {"phase": "comparison_base_point", "arm": arm_name, "side": side,
                "base_control_sha256": BASE_SHA, "direction_sha256": arm_record["direction_sha256"],
                "operator": "selected_face_extension", "direction_model": arm_name,
                "status": "started"}
            record["hvp_calls_started"] += 1
            arm_record["hvp_calls_started"] += 1
            record["hvp_started_history"].append(label)
            tangent._write(output, record)

            def calculate(sign: int = side) -> Tensor:
                with shared.transport.selected_face_extension(FACE_AXIS, FACE_ROW, FACE_COLUMN, sign):
                    return torch.func.jvp(lambda point: grad_fn(point, parameters),
                        (control,), (direction,))[1]

            try:
                hvp = calculate()
                if hvp.shape != control.shape or not bool(torch.isfinite(hvp).all()):
                    raise ValueError("fresh comparison HVP is nonfinite or has wrong shape")
                hvps[side] = hvp
                record["hvp_calls_completed"] += 1
                arm_record["hvp_calls_completed"] += 1
                complete = {**label, "status": "completed"}
            except Exception as error:
                hvp_error = f"{type(error).__name__}: {error}"
                complete = {**label, "status": "failed", "error": hvp_error}
            source_after_hvp = shared._source_hashes(plan, plan_path, plan_sha)
            input_after_hvp = shared._input_identity(problem, original, control, parameters, truth)
            runtime_after_hvp = shared._runtime()
            complete.update({"source_unchanged": source_after_hvp == source_before,
                "input_unchanged": input_after_hvp == input_before,
                "runtime_unchanged": runtime_after_hvp == runtime_before})
            record["hvp_history"].append(complete)
            tangent._write(output, record)
            if not all((complete["source_unchanged"], complete["input_unchanged"], complete["runtime_unchanged"])):
                raise ValueError("source/input/runtime changed during comparison HVP")
            check_deadline()

        if hvp_error is not None or set(hvps) != {-1, 1}:
            arm_record.update(supported=False, unsupported_reason=hvp_error or "paired HVP incomplete")
            record["arms"][arm_name] = arm_record
            tangent._write(output, record)
            continue
        arm_record["hminus"], arm_record["hplus"] = hvps[-1].tolist(), hvps[1].tolist()
        try:
            model = mixing.minimum_tangent_model(observed["side"][-1][1], observed["side"][1][1],
                hvps[-1], hvps[1], normal, chart, theta, float(observed["q"]), pivot,
                FACE_SCALE, direction_override=direction)
        except Exception as error:
            arm_record.update(supported=False,
                unsupported_reason=f"{type(error).__name__}: {error}")
            record["arms"][arm_name] = arm_record
            tangent._write(output, record)
            continue
        if not torch.equal(model.direction, direction):
            raise ValueError(f"{arm_name} model changed its assigned direction")
        side_cost_slopes: tuple[float, float] = (
            float(torch.dot(observed["side"][-1][1], direction)),
            float(torch.dot(observed["side"][1][1], direction)))
        arm_record.update({"working_theta": model.theta, "theta_prime": model.delta_theta,
            "residual": model.residual.tolist(), "residual_direction": model.residual_direction.tolist(),
            "merit_product": model.merit_product, "side_cost_slopes": list(side_cost_slopes),
            "selected_J_slope": max(side_cost_slopes), "side_products": list(model.side_products),
            "gates": model.gates})
        model_gates = ("finite", "both_side_gradients_descend", "merit_descends",
            "envelope_slope_matches_residual_dot")
        if not all(model.gates.get(key) is True for key in model_gates):
            arm_record.update(supported=False, unsupported_reason="direction model failed descent gates")
            record["arms"][arm_name] = arm_record
            tangent._write(output, record)
            continue

        def evaluate(candidate: Tensor, _theta_prediction: float, alpha: float,
                     active_model: Any = model, active_direction: Tensor = direction,
                     costs: tuple[float, float] = side_cost_slopes) -> dict[str, Any]:
            check_deadline()
            trial = tangent._observe(shared, problem, candidate, parameters, weights)
            check_deadline()
            candidate_theta: float | None = None
            try:
                candidate_theta = float(tangent.minimum_mixture_weight(
                    trial["side"][-1][1], trial["side"][1][1]))
            except ValueError:
                pass
            valid_theta = (candidate_theta is not None and math.isfinite(candidate_theta)
                and eps < candidate_theta < 1.0 - eps)
            result: dict[str, Any] = {"mixing_minimum_valid": valid_theta,
                "theta": candidate_theta if valid_theta else None,
                "branch_pair_passed": trial["pair"]["passed"],
                "face_audit_passed": trial["face_ok"],
                "side_objectives_match_native": trial["side_objectives_match_native"],
                "side_gradients_finite": trial["side_gradients_finite"],
                "side_gradients": {str(s): trial["side"][s][1].tolist() for s in (-1, 1)},
                "branch_trace": {str(s): trial["traces"][s] for s in (-1, 1)},
                "base_direction_sha256": arm_record["direction_sha256"]}
            if not valid_theta or candidate_theta is None:
                result.update(finite=False, J_armijo_passed=False, F_squared_armijo_passed=False)
                return result
            residual = torch.cat(((1 - candidate_theta) * trial["side"][-1][1]
                + candidate_theta * trial["side"][1][1], (trial["q"] / FACE_SCALE).reshape(1)))
            actual_r = float(torch.dot(residual, residual))
            j_bound, r_bound = _armijo_bounds(float(observed["native_j"]), R, alpha,
                costs, active_model.merit_product)
            actual_j = float(trial["native_j"])
            result.update({"objective": actual_j, "F_squared": actual_r,
                "J_armijo_bound": j_bound, "J_armijo_passed": actual_j <= j_bound,
                "F_squared_armijo_bound": r_bound, "F_squared_armijo_passed": actual_r <= r_bound,
                "finite": math.isfinite(actual_j) and math.isfinite(actual_r)
                    and bool(torch.isfinite(residual).all()), "G": residual.tolist(),
                "candidate_direction_sha256": tangent._tensor_sha(active_direction)})
            return result

        accepted, trials, search_meta = prepared._search_candidates(model, control, evaluate,
            chart_candidate=lambda point, vector, alpha: prepared.local._chart_candidate(
                point, weights, vector, pivot, alpha), radius=RADIUS, limit=MAX_CANDIDATES_PER_ARM)
        arm_record["trials"] = trials
        arm_record["search"] = search_meta
        if accepted is not None:
            candidate = torch.as_tensor(accepted["control"], dtype=control.dtype)
            event_values: dict[str, list[float]] = {}
            for side in (-1, 1):
                check_deadline()
                measured = _event_plain_values(problem, candidate, parameters, side)
                check_deadline()
                if not bool(torch.isfinite(measured).all() & (measured[2] > 0)):
                    raise ValueError(f"accepted {arm_name} event value/scale is nonfinite")
                event_values[str(side)] = measured.tolist()
            arm_record["accepted"] = accepted
            arm_record["accepted_event_values"] = {s: {"values": values,
                "zeta": values[0] - values[1], "normalized_abs_zeta": abs(values[0] - values[1]) / values[2]}
                for s, values in event_values.items()}
        record["arms"][arm_name] = arm_record
        tangent._write(output, record)

    record["candidate_slots_used"] = sum(
        int(record["arms"][name].get("search", {}).get("evaluated_count", 0))
        for name in ARM_ORDER)
    if record["candidate_slots_used"] > POLICY["candidate_count"]:
        raise ValueError("two-arm candidate count exceeded the 48-slot plan budget")
    chosen = _select_arm_candidate(record["arms"], control.dtype)
    if chosen is None:
        after = shared._input_identity(problem, original, control, parameters, truth)
        source_after = shared._source_hashes(plan, plan_path, plan_sha)
        runtime_after = shared._runtime()
        if after != input_before or source_after != source_anchor or runtime_after != runtime_before:
            raise ValueError("source/input/runtime changed during complete no-candidate comparison")
        record.update(phase="finished", execution_status="completed",
            numerical_status="event_direction_comparison_complete_no_candidate_passed",
            source_after=source_after, input_after=after, runtime_after=runtime_after,
            source_unchanged=True, fixed_input_unchanged=True, runtime_unchanged=True,
            deadline_passed=time.monotonic() < deadline, optimizer_steps_applied=0,
            candidate_committed=False, readiness_complete=False)
        tangent._write(output, record)
        return record

    arm_name, accepted = chosen["arm"], chosen["proposal"]
    arm = record["arms"][arm_name]
    direction = torch.as_tensor(arm["direction"], dtype=control.dtype)
    trial_observation = {"native_j": accepted["objective"],
        "face_ok": accepted["face_audit_passed"], "pair": {"passed": accepted["branch_pair_passed"]},
        "side_objectives_match_native": accepted["side_objectives_match_native"],
        "side_gradients_finite": accepted["side_gradients_finite"],
        "side": {s: (None, torch.as_tensor(accepted["side_gradients"][str(s)], dtype=control.dtype))
            for s in (-1, 1)},
        "traces": {s: accepted["branch_trace"][str(s)] for s in (-1, 1)}}
    proposal = prepared.gni._candidate_proposal(torch.as_tensor(accepted["control"], dtype=control.dtype),
        trial_observation, float(accepted["theta"]), float(accepted["F_squared"]),
        float(accepted["alpha"]), float(accepted["J_armijo_bound"]),
        float(accepted["F_squared_armijo_bound"]))
    proposal["selected_arm"] = arm_name
    proposal["selected_direction_sha256"] = arm["direction_sha256"]
    check_deadline()
    if (shared._source_hashes(plan, plan_path, plan_sha) != source_before
            or shared._input_identity(problem, original, control, parameters, truth) != input_before
            or shared._runtime() != runtime_before):
        raise ValueError("source, fixed input, or runtime changed before selected-candidate P2")
    repeat = prepared.gni._final_repeat(shared, geometry, tangent, plan, plan_path, plan_sha,
        problem, original, parameters, truth, weights, proposal, input_before, runtime_before,
        source_before, deadline)
    check_deadline()
    if not tangent.fresh_final_closure(proposal, repeat):
        raise ValueError("selected candidate P2 failed; original 027 point remains current")
    record["selection"] = {"arm": arm_name, "direction_sha256": arm["direction_sha256"],
        "comparison_complete": chosen["comparison_complete"],
        "selection_basis": chosen["selection_basis"],
        "supported_arms": chosen["supported_arms"],
        "alpha": float(accepted["alpha"]), "actual_J": float(accepted["objective"]),
        "actual_R": float(accepted["F_squared"]),
        "per_arm_first_pass_counts": _per_arm_first_pass_counts(record["arms"])}
    record["source_after_proposal"] = shared._source_hashes(plan, plan_path, plan_sha)
    record["input_after_proposal"] = shared._input_identity(problem, original, control, parameters, truth)
    record["runtime_after_proposal"] = shared._runtime()
    record = _persist_selected_commit(output, record, proposal, repeat)
    new_control = torch.as_tensor(repeat["control"], dtype=control.dtype)
    source_after = shared._source_hashes(plan, plan_path, plan_sha)
    input_after = shared._input_identity(problem, original, new_control, parameters, truth)
    runtime_after = shared._runtime()
    source_ok = source_after == source_before
    fixed_input_ok = geometry.model._fixed_input(input_after, input_before, tangent._tensor_sha(new_control))
    runtime_ok = runtime_after == runtime_before
    deadline_ok = time.monotonic() < deadline
    record.update({"phase": "finished", "execution_status": "completed",
        "numerical_status": ("event_direction_comparison_committed_one_selected_candidate_no_readiness"
            if record["selection"]["comparison_complete"] else
            "event_direction_single_supported_arm_fallback_committed_no_readiness")
            if all((source_ok, fixed_input_ok, runtime_ok, deadline_ok))
            else "event_direction_candidate_committed_postcommit_identity_incomplete",
        "source_after": source_after, "input_after": input_after, "runtime_after": runtime_after,
        "source_unchanged": source_ok, "fixed_input_unchanged": fixed_input_ok,
        "runtime_unchanged": runtime_ok, "deadline_passed": deadline_ok,
        "readiness_complete": False, "postcommit_readiness_performed": False,
        "additional_candidates_after_commit": 0})
    tangent._write(output, record)
    return record


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    _load_plan(plan_path, plan_sha)
    plan_path = plan_path.resolve()
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, output.with_suffix(".run.json"))):
        raise ValueError("event-direction comparison output paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
        "--plan", str(plan_path), "--plan-sha256", plan_sha, "--output", str(output)]
    result = run_guarded_diagnostic(command, wall_seconds=POLICY["outer_seconds"],
        rss_bytes=POLICY["rss_bytes"], report_path=resource, log_path=log)
    parent: dict[str, Any] = {"execution_status": shared._execution_status(result,
        wall_seconds=POLICY["outer_seconds"], rss_bytes=POLICY["rss_bytes"]),
        "resource": result, "child_sha256": _sha(output) if output.exists() else None,
        "numerical_status": "not_reached"}
    if output.exists():
        child = json.loads(output.read_text())
        parent["numerical_status"] = child.get("numerical_status")
        parent["candidate_committed"] = child.get("candidate_committed") is True
        parent["selected_arm"] = (child.get("selection") or {}).get("arm")
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
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
