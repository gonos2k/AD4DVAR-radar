"""Search a bounded dyadic GN window from the freshly closed prepared point.

The prior commit's same-point HVP vectors and current direction are reused only
after its saved receipt passes the independent NumPy audit. This search makes
one independently repeated, durable commit at most; readiness is then rebuilt
at that new point by the existing GN commit implementation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib import import_module
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
from examples.weather_scenarios import fv_point_3h_gn_local_commit as gni
from examples.weather_scenarios import fv_point_3h_gn_local_window_probe as local
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "GN_PREPARED_SEARCH_PLAN_20261010.json"
OUTPUT_DIR = EVIDENCE / "gn_prepared_search_20261010_attempt1"
SELF = "examples/weather_scenarios/fv_point_3h_gn_prepared_search.py"
TEST = "tests/test_fv_point_3h_gn_prepared_search.py"
AUDIT_PATH = EVIDENCE / "GN_LOCAL_COMMIT_SAVED_AUDIT_20261010.py"
AUDIT_SHA = "ac21aa3c61561064a89a3026d5ad3505b120a2f15c79a28e62fa3fb0dc6f92bc"
COMMIT_ARCHIVE = EVIDENCE / "GN_LOCAL_COMMIT_ARCHIVE_20261010.json"
COMMIT_GZIP = EVIDENCE / "gn_local_commit_20261010_attempt1/step.json.gz"
COMMIT_RUN = EVIDENCE / "gn_local_commit_20261010_attempt1/step.run.json"
COMMIT_RESOURCE = EVIDENCE / "gn_local_commit_20261010_attempt1/step.resource.json"
COMMIT_ARCHIVE_SHA = "a25a8d8bfc0ec374ff149ed4745665d99a7d3dba423307f9676c257e5cced112"
COMMIT_GZIP_SHA = "4e59a33c63b16f0b0c8d62c1bccdd9f46b554427bcf8ec524bb36c802d579a94"
COMMIT_RUN_SHA = "590e0d2835ff75098d2ebffc66bf422745526dcf748d48786fad8d3c62a9f35d"
COMMIT_RESOURCE_SHA = "fbe1a645e18fbb609ceb35581f38f5ff76ae6e57e7ba771d97116f5005e59c66"
COMMIT_RAW_SHA = "0f2892badb2eeebb9ae8d11ff3f3898a7166d68e13e0279e3a3c9625c8c1ff00"
BASE_CONTROL_SHA = "311f27d095be4976ca6131b83b6c4f3d46bc404d4f58921800b5600e38ec0652"
BASE_THETA = 0.483079578890406
BASE_J = 0.06119247898831076
BASE_F2 = 0.004478429816264897
BASE_DIRECTION_SHA = "f8862290dfcd7ec73e924182e5bb5376c1a66bd8a890de3e55b84e73b5501595"
PREDECESSOR_PLAN_SHA = "11e4eff337084ba9a9602b477854349969af8306926911fa49c4ab1e8e869b6e"
FACE = gni.FACE
FACE_SCALE = 0.84
RADIUS = 0.05
MAX_SEARCH_CANDIDATES = 24
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "optimizer_steps": 1, "candidate_count": 24,
    "candidate_cap": 24, "radius": RADIUS, "jacobian_row_vjp_calls": 24,
    "dense_solves": 1, "hvp_calls": 2, "face_scale": FACE_SCALE,
    "root_claim": False, "minimum_claim": False,
    "score_claim": False, "response_claim": False,
}


def _sha(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"pinned path must not be a symbolic link: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned path escapes the repository: {path}")
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("prepared-search plan identity mismatch")
    prior = gni._load_plan(gni.PLAN, PREDECESSOR_PLAN_SHA)
    added_archives = {
        gni.PLAN.relative_to(ROOT).as_posix(), COMMIT_ARCHIVE.relative_to(ROOT).as_posix(),
        COMMIT_GZIP.relative_to(ROOT).as_posix(), COMMIT_RUN.relative_to(ROOT).as_posix(),
        COMMIT_RESOURCE.relative_to(ROOT).as_posix(),
    }
    plan = json.loads(path.read_text())
    expected_sources = set(prior["source_files"]) | {SELF, TEST, AUDIT_PATH.relative_to(ROOT).as_posix()}
    expected_archives = set(prior["archive_files"]) | added_archives
    if (plan.get("experiment_kind") != "prepared_gn_dyadic_search_commit"
            or plan.get("policy") != POLICY or plan.get("face") != FACE
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA or plan.get("base_objective") != BASE_J
            or plan.get("base_F_squared") != BASE_F2
            or plan.get("direction_sha256") != BASE_DIRECTION_SHA
            or plan.get("candidate_cap") != MAX_SEARCH_CANDIDATES
            or plan.get("radius") != RADIUS
            or plan.get("predecessor_plan") != gni.PLAN.relative_to(ROOT).as_posix()
            or plan.get("predecessor_plan_sha256") != PREDECESSOR_PLAN_SHA
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 151
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 195):
        raise ValueError("prepared-search plan scope, endpoint, or policy changed")
    if (any(plan["source_files"].get(name) != value for name, value in prior["source_files"].items())
            or any(plan["archive_files"].get(name) != value for name, value in prior["archive_files"].items())):
        raise ValueError("prepared-search plan changed inherited GN pins")
    for name, value in {**plan["source_files"], **plan["archive_files"]}.items():
        target = ROOT / name
        if _sha(target) != value:
            raise ValueError(f"prepared-search source/archive pin mismatch: {name}")
    if (plan["source_files"].get(SELF) != _sha(ROOT / SELF)
            or plan["source_files"].get(TEST) != _sha(ROOT / TEST)
            or plan["source_files"].get(AUDIT_PATH.relative_to(ROOT).as_posix()) != _sha(AUDIT_PATH)):
        raise ValueError("prepared-search source/test/audit pin mismatch")
    return plan


def _archive_identity() -> dict[str, Any]:
    """Validate the previous closed point from receipts without loading FV."""
    expected = ((AUDIT_PATH, AUDIT_SHA), (COMMIT_ARCHIVE, COMMIT_ARCHIVE_SHA),
        (COMMIT_GZIP, COMMIT_GZIP_SHA), (COMMIT_RUN, COMMIT_RUN_SHA),
        (COMMIT_RESOURCE, COMMIT_RESOURCE_SHA))
    for path, digest in expected:
        if _sha(path) != digest:
            raise ValueError(f"prior prepared point evidence changed: {path.name}")
    manifest = json.loads(COMMIT_ARCHIVE.read_text())
    run = json.loads(COMMIT_RUN.read_text())
    resource = json.loads(COMMIT_RESOURCE.read_text())
    raw_bytes = gzip.decompress(COMMIT_GZIP.read_bytes())
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    if (raw_sha != COMMIT_RAW_SHA or manifest.get("raw_sha256") != raw_sha
            or manifest.get("gzip_sha256") != COMMIT_GZIP_SHA
            or manifest.get("run_sha256") != COMMIT_RUN_SHA
            or manifest.get("resource_sha256") != COMMIT_RESOURCE_SHA
            or manifest.get("lossless_roundtrip") is not True
            or run.get("execution_status") != "completed" or run.get("child_sha256") != raw_sha
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != 660.0
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes", POLICY["rss_bytes"]) >= POLICY["rss_bytes"]):
        raise ValueError("prior GN execution/archive receipts do not close")
    if str(EVIDENCE) not in sys.path:
        sys.path.insert(0, str(EVIDENCE))
    saved_audit = import_module("GN_LOCAL_COMMIT_SAVED_AUDIT_20261010")
    audit = saved_audit.audit(raw, raw_sha)
    saved_audit._require_resource(raw_sha, COMMIT_RUN, COMMIT_RESOURCE)
    if not audit.get("commit_closed") or not audit.get("P2_closed") or not audit.get("postcommit_readiness_complete"):
        raise ValueError("saved audit did not close the prior committed point")
    readiness = raw["postcommit_gn_readiness"]
    if (raw["current_theta"] != BASE_THETA or raw["current_control_sha256"] != BASE_CONTROL_SHA
            or raw["proposal"]["objective"] != BASE_J or raw["proposal"]["F_squared"] != BASE_F2
            or readiness.get("direction_sha256") != BASE_DIRECTION_SHA
            or readiness.get("candidate_count") != 0):
        raise ValueError("prior saved current-point receipt has unexpected base/direction")
    compact = {key: raw[key] for key in ("current_control", "current_control_sha256",
        "current_theta", "final_repeat", "input_after", "runtime_after", "postcommit_gn_readiness")}
    del raw, raw_bytes
    return {"raw": compact, "raw_sha256": raw_sha, "audit": audit}


def _prepare_base(problem: Any, original: Tensor, control: Tensor, parameters: Tensor,
                  truth: Tensor, weights: Tensor, base: dict[str, Any]) -> dict[str, Any]:
    observed = tangent._observe(shared, problem, control, parameters, weights)
    theta = float(tangent.minimum_mixture_weight(observed["side"][-1][1], observed["side"][1][1]))
    g = torch.cat(((1 - theta) * observed["side"][-1][1] + theta * observed["side"][1][1],
        (observed["q"] / FACE_SCALE).reshape(1)))
    merit = float(torch.dot(g, g))
    previous = base["raw"]["final_repeat"]
    eps = 128 * torch.finfo(control.dtype).eps
    if (tangent._tensor_sha(control) != BASE_CONTROL_SHA or abs(theta - BASE_THETA) > eps * abs(BASE_THETA)
            or abs(float(observed["native_j"]) - BASE_J) > eps * abs(BASE_J)
            or abs(merit - BASE_F2) > eps * abs(BASE_F2)
            or not shared._gradient_pair_match({s: observed["side"][s][1] for s in (-1, 1)},
                {s: torch.as_tensor(previous["side_gradients"][str(s)], dtype=control.dtype) for s in (-1, 1)})
            or not observed["pair"]["passed"] or not observed["face_ok"]
            or not observed["side_objectives_match_native"] or not observed["side_gradients_finite"]
            or any(observed["traces"][s]["signature_sha256"] != previous["branch_trace"][str(s)]["signature_sha256"]
                for s in (-1, 1))):
        raise ValueError("fresh prepared point J, side pair, minimum, trace, or face failed closure")
    pivot = geometry._pivot(control, weights)
    normal = geometry._face_normal(control, weights)
    _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    readiness = base["raw"]["postcommit_gn_readiness"]
    direction = torch.as_tensor(readiness["direction"], dtype=control.dtype)
    model = mixing.minimum_tangent_model(observed["side"][-1][1], observed["side"][1][1],
        torch.as_tensor(readiness["hminus"], dtype=control.dtype),
        torch.as_tensor(readiness["hplus"], dtype=control.dtype), normal, chart, theta,
        float(observed["q"]), pivot, FACE_SCALE, direction_override=direction)
    if (tangent._tensor_sha(direction) != BASE_DIRECTION_SHA
            or not all(model.gates.get(key) is True for key in (
                "finite", "both_side_gradients_descend", "merit_descends", "envelope_slope_matches_residual_dot"))
            or not local._scaled_error(model.residual,
                torch.as_tensor(readiness["residual"], dtype=control.dtype), model.residual)["passed"]
            or not local._scaled_error(model.residual_direction,
                torch.as_tensor(readiness["residual_direction"], dtype=control.dtype), model.residual_direction)["passed"]):
        raise ValueError("reconstructed prepared-point tangent model differs from its saved receipt")
    return {"observed": observed, "theta": theta, "G": g, "F_squared": merit,
        "direction": direction, "model": model, "pivot": pivot, "normal": normal, "chart": chart,
        "base_trace_signatures": {s: observed["traces"][s]["signature_sha256"] for s in (-1, 1)}}


def _search_candidates(model: Any, current: Tensor,
        evaluate: Callable[[Tensor, float, float], dict[str, Any]], *,
        chart_candidate: Callable[[Tensor, Tensor, float], Tensor],
        radius: float = RADIUS, limit: int = MAX_SEARCH_CANDIDATES
        ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    if limit < 1 or limit > MAX_SEARCH_CANDIDATES:
        raise ValueError("prepared search candidate cap must be in [1, 24]")
    alphas = tangent.candidate_alphas(model, radius=radius, limit=limit)
    search_meta = {"slot_cap": limit, "computed_alpha0": alphas[0] if alphas else None,
        "minimum_computed_alpha": alphas[-1] if alphas else None,
        "computed_alpha_count": len(alphas), "evaluated_count": 0,
        "first_pass_candidate_index": None, "first_pass_count": 0}
    trials: list[dict[str, Any]] = []
    for index, alpha in enumerate(alphas, 1):
        trial: dict[str, Any] = {"index": index, "alpha": alpha, "accepted": False}
        candidate = chart_candidate(current, model.direction, alpha)
        displacement = candidate - current
        actual_norm = float(torch.linalg.vector_norm(displacement))
        trial["actual_path_norm"] = actual_norm
        search_meta["evaluated_count"] = index
        if not math.isfinite(actual_norm) or not bool(torch.isfinite(candidate).all()):
            trial["status"] = "nonfinite_chart_movement_refused"
            trials.append(trial)
            continue
        trial["control"] = candidate.tolist()
        trial["control_sha256"] = tangent._tensor_sha(candidate)
        if actual_norm <= 0.0 or torch.equal(candidate, current):
            trial["status"] = "zero_control_movement_refused"
            trials.append(trial)
            continue
        if actual_norm > radius * (1 + 64 * torch.finfo(current.dtype).eps):
            trial["status"] = "actual_chart_radius_refused"
            trials.append(trial)
            continue
        facts = evaluate(candidate, model.theta, alpha)
        trial.update(facts)
        trial["accepted"] = all(facts.get(key) is True for key in (
            "mixing_minimum_valid", "J_armijo_passed", "F_squared_armijo_passed",
            "branch_pair_passed", "face_audit_passed", "side_objectives_match_native",
            "side_gradients_finite", "finite"))
        trial["status"] = "accepted" if trial["accepted"] else "gate_refused"
        trials.append(trial)
        if trial["accepted"]:
            search_meta["first_pass_candidate_index"] = index
            search_meta["first_pass_count"] = index
            return trial, trials, search_meta
    return None, trials, search_meta


def _armijo_bounds(base_j: float, base_f2: float, alpha: float,
                   j_slope: float, merit_slope: float) -> tuple[float, float]:
    """Use the original native-J and squared-residual Armijo coefficients."""
    return base_j + 1e-4 * alpha * j_slope, base_f2 + 2e-4 * alpha * merit_slope


def _persist_closed_commit(output: Path, record: dict[str, Any],
                           proposal: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    if not tangent.fresh_final_closure(proposal, repeat):
        raise ValueError("independent P2 final closure failed; prepared point remains current")
    control = torch.as_tensor(repeat["control"], dtype=torch.float64)
    sha = tangent._tensor_sha(control)
    committed = {**record, "phase": "committed", "execution_status": "running",
        "numerical_status": "prepared_candidate_committed_postcommit_gn_readiness_pending",
        "current_control": control.tolist(), "current_control_sha256": sha,
        "current_theta": float(repeat["theta"]), "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": sha, "last_confirmed_theta": float(repeat["theta"]),
        "last_confirmed_iterations": 1, "optimizer_steps_applied": 1,
        "accepted_iterations": 1, "candidate_committed": True,
        "proposal": proposal, "final_repeat": repeat, "last_confirmed_closure": repeat,
        "postcommit_candidate_alpha": float(proposal["alpha"])}
    gni._atomic_write(output, committed)
    return committed


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    start = time.monotonic()
    deadline = start + float(POLICY["internal_seconds"])
    plan = _load_plan(plan_path, plan_sha)
    plan_path = plan_path.resolve()
    source_before = shared._source_hashes(plan, plan_path, plan_sha)
    shared._deadline(deadline, POLICY["internal_seconds"])
    base = _archive_identity()
    previous = base["raw"]
    control = torch.as_tensor(previous["current_control"], dtype=torch.float64)
    direction = torch.as_tensor(previous["postcommit_gn_readiness"]["direction"], dtype=control.dtype)
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "prepared_search_not_yet_committed", "plan_sha256": plan_sha,
        "policy": POLICY, "current_control": control.tolist(), "current_control_sha256": BASE_CONTROL_SHA,
        "current_theta": BASE_THETA, "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": BASE_CONTROL_SHA, "last_confirmed_theta": BASE_THETA,
        "last_confirmed_iterations": 0, "prior_confirmed_iterations": 1,
        "optimizer_steps_applied": 0, "candidate_committed": False,
        "root_claim": False, "minimum_claim": False, "score_claim": False, "response_claim": False}
    gni._atomic_write(output, record)
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    shared._deadline(deadline, POLICY["internal_seconds"])
    input_before = shared._input_identity(problem, original, control, parameters, truth)
    runtime_before = shared._runtime()
    if (input_before != previous["input_after"] or runtime_before != previous["runtime_after"]
            or not geometry.model._fixed_input(input_before, previous["input_after"], BASE_CONTROL_SHA)):
        raise ValueError("fresh fixed input/runtime no longer matches the prepared endpoint")
    weights = geometry._face_weights(problem, axis=FACE["axis"], row=FACE["row"], column=FACE["column"])
    prepared = _prepare_base(problem, original, control, parameters, truth, weights, base)
    model = prepared["model"]

    def evaluate(candidate: Tensor, _theta_prediction: float, alpha: float) -> dict[str, Any]:
        shared._deadline(deadline, POLICY["internal_seconds"])
        trial = tangent._observe(shared, problem, candidate, parameters, weights)
        shared._deadline(deadline, POLICY["internal_seconds"])
        theta: float | None = None
        try:
            theta = float(tangent.minimum_mixture_weight(trial["side"][-1][1], trial["side"][1][1]))
        except ValueError:
            pass
        result: dict[str, Any] = {"mixing_minimum_valid": theta is not None and
            128 * torch.finfo(control.dtype).eps < theta < 1.0 - 128 * torch.finfo(control.dtype).eps,
            "theta": theta, "branch_pair_passed": trial["pair"]["passed"],
            "face_audit_passed": trial["face_ok"],
            "side_objectives_match_native": trial["side_objectives_match_native"],
            "side_gradients_finite": trial["side_gradients_finite"],
            "trace_matches_base": all(trial["traces"][s]["signature_sha256"]
                == prepared["base_trace_signatures"][s] for s in (-1, 1))}
        if theta is None:
            result.update(finite=False, J_armijo_passed=False, F_squared_armijo_passed=False)
            return result
        residual = torch.cat(((1 - theta) * trial["side"][-1][1] + theta * trial["side"][1][1],
            (trial["q"] / FACE_SCALE).reshape(1)))
        f2 = float(torch.dot(residual, residual))
        j_slope = max(float(torch.dot(prepared["observed"]["side"][s][1], direction)) for s in (-1, 1))
        j_bound, f2_bound = _armijo_bounds(float(prepared["observed"]["native_j"]),
            prepared["F_squared"], alpha, j_slope, model.merit_product)
        result.update({"objective": float(trial["native_j"]), "F_squared": f2,
            "J_armijo_bound": j_bound, "J_armijo_passed": float(trial["native_j"]) <= j_bound,
            "F_squared_armijo_bound": f2_bound, "F_squared_armijo_passed": f2 <= f2_bound,
            "finite": math.isfinite(float(trial["native_j"])) and math.isfinite(f2)
                and bool(torch.isfinite(residual).all()),
            "G": residual.tolist(), "side_gradients": {str(s): trial["side"][s][1].tolist() for s in (-1, 1)},
            "branch_trace": {str(s): trial["traces"][s] for s in (-1, 1)}})
        return result

    accepted, trials, search_meta = _search_candidates(model, control,
        evaluate, chart_candidate=lambda point, vector, alpha: local._chart_candidate(
            point, weights, vector, prepared["pivot"], alpha))
    record.update({"source_before": source_before, "input_before": input_before,
        "runtime_before": runtime_before, "base_control_sha256": BASE_CONTROL_SHA,
        "base_theta": BASE_THETA, "base_objective": BASE_J, "base_F_squared": BASE_F2,
        "archive_identity": {"prior_plan_sha256": PREDECESSOR_PLAN_SHA,
            "prior_confirmed_commits": 1,
            "prior_commit_raw_sha256": base["raw_sha256"], "reused_same_point_hvp_vectors": 2,
            "fresh_base_hvp_calls": 0, "fresh_postcommit_hvp_budget": 2},
        "search": search_meta, "candidate_trials": trials,
        "hvp_calls_started": 0, "hvp_calls_completed": 0,
        "jacobian_rows_started": 0, "jacobian_rows_completed": 0,
        "dense_solves_started": 0, "dense_solves_completed": 0})
    shared._deadline(deadline, POLICY["internal_seconds"])
    if accepted is None:
        after = shared._input_identity(problem, original, control, parameters, truth)
        source_after = shared._source_hashes(plan, plan_path, plan_sha)
        runtime_after = shared._runtime()
        if (after != input_before or source_after != source_before or runtime_after != runtime_before):
            raise ValueError("source/input/runtime changed during completed no-candidate search")
        record.update(phase="finished", execution_status="completed",
            numerical_status="prepared_search_completed_no_candidate_passed", source_after=source_after,
            input_after=after, runtime_after=runtime_after, source_unchanged=True,
            fixed_input_unchanged=True, runtime_unchanged=True,
            deadline_passed=time.monotonic() < deadline, optimizer_steps_applied=0,
            candidate_committed=False, readiness_complete=False)
        gni._atomic_write(output, record)
        return record

    proposal = gni._candidate_proposal(torch.as_tensor(accepted["control"], dtype=control.dtype),
        {"native_j": accepted["objective"], "face_ok": accepted["face_audit_passed"],
         "pair": {"passed": accepted["branch_pair_passed"]},
         "side_objectives_match_native": accepted["side_objectives_match_native"],
         "side_gradients_finite": accepted["side_gradients_finite"],
         "side": {s: (None, torch.as_tensor(accepted["side_gradients"][str(s)], dtype=control.dtype))
             for s in (-1, 1)}, "traces": {s: accepted["branch_trace"][str(s)] for s in (-1, 1)}},
        float(accepted["theta"]), float(accepted["F_squared"]), float(accepted["alpha"]),
        float(accepted["J_armijo_bound"]), float(accepted["F_squared_armijo_bound"]))
    source_now = shared._source_hashes(plan, plan_path, plan_sha)
    input_now = shared._input_identity(problem, original, control, parameters, truth)
    runtime_now = shared._runtime()
    if source_now != source_before or input_now != input_before or runtime_now != runtime_before:
        raise ValueError("source, fixed input, or runtime changed before P2 repeat")
    repeat = gni._final_repeat(shared, geometry, tangent, plan, plan_path, plan_sha,
        problem, original, parameters, truth, weights, proposal, input_before, runtime_before,
        source_before, deadline)
    shared._deadline(deadline, POLICY["internal_seconds"])
    if not tangent.fresh_final_closure(proposal, repeat):
        raise ValueError("fresh independent P2 repeat failed; prepared point remains current")
    record["search"]["first_pass_count"] = int(search_meta["evaluated_count"])
    record = _persist_closed_commit(output, record, proposal, repeat)
    new_control = torch.as_tensor(repeat["control"], dtype=control.dtype)
    try:
        gni._build_current_gn_readiness(record, output, plan, plan_path, plan_sha,
            problem, new_control, parameters, weights, original, truth, runtime_before,
            input_before, source_before, deadline, start)
    except Exception as error:
        record = gni._persist_readiness_failure(output, record, error)
    return record


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    _load_plan(plan_path, plan_sha)
    plan_path = plan_path.resolve()
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, output.with_suffix(".run.json"))):
        raise ValueError("prepared-search output paths must be fresh")
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
