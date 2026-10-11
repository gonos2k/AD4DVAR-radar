"""Rebuild current GN at 454396 and evaluate a fixed local window without commit."""
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
from advar import transport

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_gn_continuation_progress as prior
from examples.weather_scenarios import fv_point_3h_event_direction_comparison as comparison
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_comparison as coupled
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "GN_LOCAL_DIAGNOSTIC_PLAN_20261011.json"
OUTPUT_DIR = EVIDENCE / "gn_local_diagnostic_20261011_attempt1"
SELF = "examples/weather_scenarios/fv_point_3h_gn_local_diagnostic.py"
TEST = "tests/test_fv_point_3h_gn_local_diagnostic.py"
PRIOR_PLAN_SHA = "09e51cb8cb74ade0990052c0ef5e62cc95c0dbf1453a003a264655a8bed89af6"
BASE_SHA = "454396e77e2f10492e80e0fd625b55e1a08c2be53fe9721b59a4a27d60be3b7a"
BASE_ARCHIVE = EVIDENCE / "GN_CONTINUATION_ARCHIVE_20261011.json"
BASE_GZIP = EVIDENCE / "gn_continuation_20261011_attempt1/step.json.gz"
BASE_RUN = EVIDENCE / "gn_continuation_20261011_attempt1/step.run.json"
BASE_RESOURCE = EVIDENCE / "gn_continuation_20261011_attempt1/step.resource.json"
BASE_ARCHIVE_SHA = "ac97af49c4db29de9c23ad71979d55eda7423198188ccdda56dff5ff226d1107"
BASE_GZIP_SHA = "3fdc6f32d5a58b5e929e31a4e93be698d9da9738e913a99e05ae77534b86239c"
BASE_RUN_SHA = "92ab45a535ee5fa6f44d6487a29bb1f30c14ca6543430bdab9cbcb5e768973f3"
BASE_RESOURCE_SHA = "c0f1c275c0813e35a92807b7f514cc487bc46e6b3483405632879018084b7d2d"
BASE_RAW_SHA = "f3cd137e9f3e348d9ca679a7b701bad9294ed5c4d2b7693c0c6f150dbb716242"
FACE_AXIS, FACE_ROW, FACE_COLUMN = "y", 4, 3
FACE = {"axis": FACE_AXIS, "row": FACE_ROW, "column": FACE_COLUMN}
FACE_SCALE, RADIUS = 0.84, 0.05
MAX_CANDIDATES = 24
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "optimizer_steps": 0, "max_commits": 0,
    "jacobian_row_vjp_calls": 24, "dense_solves": 1, "hvp_calls": 2,
    "trace_snapshot_sides": 2, "candidate_count": 8, "diagnostic_sample_count": 8,
    "radius": RADIUS, "face_scale": FACE_SCALE,
    "root_claim": False, "minimum_claim": False, "score_claim": False, "response_claim": False,
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
        raise ValueError("released GN resume plan identity mismatch")
    parent = prior._load_plan(prior.PLAN, PRIOR_PLAN_SHA)
    additions = {prior.PLAN.relative_to(ROOT).as_posix(), BASE_ARCHIVE.relative_to(ROOT).as_posix(),
        BASE_GZIP.relative_to(ROOT).as_posix(), BASE_RUN.relative_to(ROOT).as_posix(),
        BASE_RESOURCE.relative_to(ROOT).as_posix()}
    plan = json.loads(path.read_text())
    sources = set(parent["source_files"]) | {SELF, TEST}
    archives = set(parent["archive_files"]) | additions
    if (plan.get("experiment_kind") != "current_gn_local_diagnostic"
            or plan.get("policy") != POLICY or plan.get("face") != FACE
            or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("predecessor_plan") != prior.PLAN.relative_to(ROOT).as_posix()
            or plan.get("predecessor_plan_sha256") != PRIOR_PLAN_SHA
            or plan.get("predecessor_raw_sha256") != BASE_RAW_SHA
            or plan.get("candidate_cap") != MAX_CANDIDATES
            or plan.get("radius") != RADIUS
            or set(plan.get("source_files", {})) != sources or len(plan["source_files"]) != 167
            or set(plan.get("archive_files", {})) != archives or len(plan["archive_files"]) != 233):
        raise ValueError("released GN resume plan scope or fixed policy changed")
    for key in ("source_files", "archive_files"):
        if any(plan[key].get(name) != value for name, value in parent[key].items()):
            raise ValueError(f"resume plan changed inherited {key} pins")
        for name, value in plan[key].items():
            if _sha(ROOT / name) != value:
                raise ValueError(f"resume {key} pin mismatch: {name}")
    return plan


def _load_parent() -> dict[str, Any]:
    for path, digest in ((BASE_ARCHIVE, BASE_ARCHIVE_SHA), (BASE_GZIP, BASE_GZIP_SHA),
            (BASE_RUN, BASE_RUN_SHA), (BASE_RESOURCE, BASE_RESOURCE_SHA)):
        if _sha(path) != digest:
            raise ValueError(f"454396 parent evidence changed: {path.name}")
    manifest, run, resource = (json.loads(BASE_ARCHIVE.read_text()), json.loads(BASE_RUN.read_text()),
        json.loads(BASE_RESOURCE.read_text()))
    raw_bytes = gzip.decompress(BASE_GZIP.read_bytes())
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    parent_plan_path = EVIDENCE / "GN_CONTINUATION_PLAN_20261011.json"
    parent_sha = "319bb7798348643c0198fd09ae9bbe19ddfb88ae7bcfc94c50a80745b29c2d65"
    prior_plan = json.loads(parent_plan_path.read_text())
    prior_source = shared._source_hashes(prior_plan, parent_plan_path, parent_sha)
    if (raw_sha != BASE_RAW_SHA or manifest.get("raw_sha256") != raw_sha
            or manifest.get("gzip_sha256") != BASE_GZIP_SHA or manifest.get("run_sha256") != BASE_RUN_SHA
            or manifest.get("resource_sha256") != BASE_RESOURCE_SHA or manifest.get("lossless_roundtrip") is not True
            or run.get("execution_status") != "completed" or run.get("child_sha256") != raw_sha
            or run.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False or resource.get("wall_limit_seconds") != 660.0
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes", POLICY["rss_bytes"]) >= POLICY["rss_bytes"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or not 0.0 < float(resource["elapsed_seconds"]) <= POLICY["outer_seconds"]
            or raw.get("execution_status") != "completed" or raw.get("plan_sha256") != parent_sha
            or raw.get("source_before") != prior_source or raw.get("source_after") != prior_source
            or raw.get("current_control_sha256") != BASE_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_SHA
            or raw.get("numerical_status") != "gn_continuation_cap_reached"
            or raw.get("candidate_committed") is not True or raw.get("optimizer_steps_applied") != 3
            or raw.get("accepted_iterations") != 3 or raw.get("last_confirmed_iterations") != 3
            or raw.get("readiness_complete") is not False
            or raw.get("jacobian_rows_completed") != 72 or raw.get("dense_solves_completed") != 3
            or raw.get("hvp_calls_completed") != 6
            or raw.get("last_confirmed_closure") != raw.get("closed_steps", [{}])[-1].get("final_repeat")
            or not tangent.fresh_final_closure(raw["closed_steps"][-1]["proposal"], raw["last_confirmed_closure"])):
        raise ValueError("closed 454396 parent archive/P2/plan receipts do not match")
    return {"current_control":raw["current_control"],"current_control_sha256":raw["current_control_sha256"],
        "current_theta":raw["current_theta"],"final_repeat":raw["last_confirmed_closure"],
        "input_after":raw["input_after"],"runtime_after":raw["runtime_after"]}


def _finish_without_candidate(record: dict[str, Any], output: Path, problem: Any,
        original: Tensor, control: Tensor, parameters: Tensor, truth: Tensor,
        plan: dict[str, Any], plan_path: Path, plan_sha: str, input_before: dict[str, Any],
        runtime_before: dict[str, Any], source_before: dict[str, str], deadline: float,
        reason: str) -> dict[str, Any]:
    input_after = shared._input_identity(problem, original, control, parameters, truth)
    source_after = shared._source_hashes(plan, plan_path, plan_sha)
    runtime_after = shared._runtime()
    if input_after != input_before or source_after != source_before or runtime_after != runtime_before:
        raise ValueError("source/input/runtime changed during no-candidate GN refusal")
    record.update(phase="finished", execution_status="completed",
        numerical_status="gn_local_diagnostic_current_model_refused_no_candidate",
        refusal=reason, source_after=source_after, input_after=input_after,
        runtime_after=runtime_after, source_unchanged=True, fixed_input_unchanged=True,
        runtime_unchanged=True, deadline_passed=time.monotonic() < deadline,
        optimizer_steps_applied=0, candidate_committed=False, readiness_complete=False)
    tangent._write(output, record)
    return record


def _event_selector_snapshot(trace: dict[str, Any]) -> dict[str, Any]:
    choice = trace["choices"][124][0]
    left_sign, right_sign = choice["left_sign"][1][2], choice["right_sign"][1][2]
    slope_sign, choose_left = choice["slope_sign"][1][2], choice["choose_left"][1][2]
    return {"left_sign": left_sign, "right_sign": right_sign,
        "slope_sign": slope_sign, "choose_left": choose_left,
        "matches_old_negative_right_selector": bool(left_sign == -1 and right_sign == -1
            and slope_sign == -1 and choose_left is False)}


def _base_model_closure(model: Any, full_residual: Tensor, base_theta: float) -> dict[str, Any]:
    residual_check = prepared.local._scaled_error(model.residual, full_residual, full_residual)
    theta_budget = 128 * torch.finfo(full_residual.dtype).eps * max(
        abs(float(model.theta)), abs(base_theta), torch.finfo(full_residual.dtype).tiny)
    theta_error = abs(float(model.theta) - base_theta)
    return {"full_G_scaled_match": residual_check, "theta_error": theta_error,
        "theta_budget": theta_budget,
        "passed": residual_check["passed"] is True and theta_error <= theta_budget}


def local_sample_alphas(model: Any) -> list[float]:
    alphas = tangent.candidate_alphas(model, radius=RADIUS, limit=24)
    if len(alphas) != 24:
        raise ValueError("current local model does not support the fixed sample window")
    return [alphas[8] / 2**k for k in range(8)]


def diagnostic_samples(model: Any, current: Tensor, alphas: list[float], evaluate: Any, *,
        chart_candidate: Any, progress: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    meta = {"slots_started":0,"slots_completed":0,"evaluations_started":0,"evaluations_completed":0}
    for index,alpha in enumerate(alphas,1):
        trial: dict[str, Any] = {"index":index,"alpha":alpha,"accepted":False,"status":"slot_started"}
        trials.append(trial);meta["slots_started"]+=1;progress(trials,meta)
        try:
            candidate=chart_candidate(current,model.direction,alpha)
            movement=float(torch.linalg.vector_norm(candidate-current))
            if not bool(torch.isfinite(candidate).all()) or not math.isfinite(movement) or not 0<movement<=RADIUS:
                raise ValueError("diagnostic actual chart displacement invalid")
            trial.update(control=candidate.tolist(),control_sha256=tangent._tensor_sha(candidate),actual_path_norm=movement)
            meta["evaluations_started"]+=1;trial["status"]="evaluation_started";progress(trials,meta)
            trial.update(evaluate(candidate,model.theta,alpha));meta["evaluations_completed"]+=1
            trial["accepted"]=False;trial["status"]="diagnostic_evaluated";meta["slots_completed"]+=1
            progress(trials,meta)
        except Exception as error:
            trial.update(accepted=False,status="slot_interrupted" if isinstance(error,TimeoutError) else "slot_failed",
                error_type=type(error).__name__,error_message=str(error));progress(trials,meta);raise
    return trials,meta


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    start = time.monotonic()
    deadline = start + POLICY["internal_seconds"]
    check = lambda: shared._deadline(deadline, POLICY["internal_seconds"])
    check()
    plan = _load_plan(plan_path, plan_sha)
    plan_path = plan_path.resolve()
    source_before = shared._source_hashes(plan, plan_path, plan_sha)
    parent = _load_parent()
    check()
    control = torch.as_tensor(parent["current_control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_SHA:
        raise ValueError("released resume does not start at 454396")
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "gn_local_diagnostic_preflight", "policy": POLICY,
        "plan_sha256": plan_sha, "base_control_sha256": BASE_SHA,
        "current_control": control.tolist(), "current_control_sha256": BASE_SHA,
        "current_theta": float(parent["current_theta"]), "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": BASE_SHA, "last_confirmed_theta": float(parent["current_theta"]),
        "last_confirmed_iterations": 0, "predecessor_commits_in_scope": 3,
        "optimizer_steps_applied": 0, "candidate_committed": False,
        "jacobian_rows_started": 0, "jacobian_rows_completed": 0, "jacobian_row_history": [],
        "dense_solves_started": 0, "dense_solves_completed": 0,
        "hvp_calls_started": 0, "hvp_calls_completed": 0,
        "hvp_started_history": [], "hvp_history": [],
        "root_claim": False, "minimum_claim": False, "response_claim": False, "score_claim": False}
    tangent._write(output, record)
    check()
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    check()
    input_before, runtime_before = shared._input_identity(problem, original, control, parameters, truth), shared._runtime()
    if input_before != parent["input_after"] or runtime_before != parent["runtime_after"]:
        raise ValueError("454396 fixed input/runtime differs from its P2 receipt")
    weights = geometry._face_weights(problem, axis=FACE_AXIS, row=FACE_ROW, column=FACE_COLUMN)
    check()
    observed = tangent._observe(shared, problem, control, parameters, weights)
    check()
    closure = parent["final_repeat"]
    theta = float(tangent.minimum_mixture_weight(observed["side"][-1][1], observed["side"][1][1]))
    G = torch.cat(((1.0 - theta) * observed["side"][-1][1] + theta * observed["side"][1][1],
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
            or any(observed["traces"][s]["signature_sha256"] != closure["branch_trace"][str(s)]["signature_sha256"]
                for s in (-1, 1))):
        raise ValueError("fresh 454396 J/G/theta/strict pair/trace qualification failed")
    pivot, normal = geometry._pivot(control, weights), geometry._face_normal(control, weights)
    _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    event_snapshot = {}
    for side in (-1, 1):
        event_snapshot[str(side)] = _event_selector_snapshot(observed["traces"][side])
    record.update({"source_before": source_before, "input_before": input_before,
        "runtime_before": runtime_before, "base_J": float(observed["native_j"]),
        "base_R": R, "base_theta": theta, "base_full_mixed_residual": G.tolist(),
        "event_124_current_snapshot": event_snapshot, "event_gradient_calls": 0,
        "event_jvp_calls": 0})

    state = {"observed": observed}
    context = {"problem": problem, "parameters": parameters, "record": record,
        "output": output, "deadline": deadline, "weights": weights,
        "plan": plan, "plan_path": plan_path, "plan_sha": plan_sha,
        "model_state": state}
    specs = coupled._direction_model_factory(context)(control, theta,
        observed["side"][-1][1], observed["side"][1][1], normal, chart, pivot)
    gn_specs = [spec for spec in specs if spec.get("name") == "robust_gn_coupled"]
    if (len(gn_specs) != 1 or record["jacobian_rows_completed"] != 24
            or record["dense_solves_completed"] != 1):
        raise ValueError("released current GN factory failed the row/solve budget")
    spec = gn_specs[0]
    direction = spec["direction"]
    record["gn_direction"] = {"direction": direction.tolist(),
        "direction_sha256": tangent._tensor_sha(direction),
        "working_theta": float(spec["working_theta"]), "diagnostics": spec["diagnostics"],
        "dense_solve_audit": record["dense_solve_audit"]}

    grad_fn = torch.func.grad(problem.objective, argnums=0)
    HVP: dict[int, Tensor] = {}
    for side in (-1, 1):
        check()
        label = {"phase": "released_gn_at_454396", "side": side,
            "arm": "prepared_gn", "base_control_sha256": BASE_SHA,
            "direction_sha256": tangent._tensor_sha(direction),
            "reference_theta": theta, "working_theta": float(spec["working_theta"]),
            "operator": "selected_face_extension", "status": "started"}
        record["hvp_calls_started"] += 1
        record["hvp_started_history"].append(label)
        tangent._write(output, record)
        with transport.selected_face_extension(FACE_AXIS, FACE_ROW, FACE_COLUMN, side):
            value = torch.func.jvp(lambda point: grad_fn(point, parameters), (control,), (direction,))[1]
        if value.shape != control.shape or not bool(torch.isfinite(value).all()):
            raise ValueError("released current-point HVP is nonfinite or has wrong shape")
        HVP[side] = value
        record["hvp_calls_completed"] += 1
        record["hvp_history"].append({**label, "status": "completed"})
        tangent._write(output, record)
        after_source = shared._source_hashes(plan, plan_path, plan_sha)
        after_input = shared._input_identity(problem, original, control, parameters, truth)
        after_runtime = shared._runtime()
        if after_source != source_before or after_input != input_before or after_runtime != runtime_before:
            raise ValueError("source/input/runtime changed during released GN HVP")
        check()

    try:
        model = spec["model_builder"](observed["side"][-1][1], observed["side"][1][1],
            HVP[-1], HVP[1], normal, chart, theta, float(observed["q"]), pivot)
    except ValueError as error:
        record["gn_model"] = {"supported": False, "model_error": str(error),
            "base_full_residual": G.tolist(), "base_theta": theta}
        return _finish_without_candidate(record, output, problem, original, control, parameters, truth,
            plan, plan_path, plan_sha, input_before, runtime_before, source_before, deadline, str(error))
    model_closure = _base_model_closure(model, G, theta)
    record["gn_model_closure"] = model_closure
    if not torch.equal(model.direction, direction) or model_closure["passed"] is not True:
        raise ValueError("released GN model changed direction or current full-G/theta")
    if not all(model.gates.get(key) is True for key in (
            "finite", "both_side_gradients_descend", "merit_descends",
            "envelope_slope_matches_residual_dot")):
        record["gn_model"] = {"supported": False, "residual": model.residual.tolist(),
            "theta": float(model.theta), "base_full_residual": G.tolist(),
            "base_theta": theta, "base_model_closure": model_closure, "gates": model.gates}
        return _finish_without_candidate(record, output, problem, original, control, parameters, truth,
            plan, plan_path, plan_sha, input_before, runtime_before, source_before, deadline,
            "released GN model failed current cost/merit descent gates")
    cost_slopes = (float(torch.dot(observed["side"][-1][1], direction)),
        float(torch.dot(observed["side"][1][1], direction)))
    record["gn_model"] = {"residual": model.residual.tolist(),
        "residual_direction": model.residual_direction.tolist(), "theta_prime": model.delta_theta,
        "working_theta": float(model.theta), "merit_product": model.merit_product,
        "side_cost_slopes": list(cost_slopes), "gates": model.gates}

    def evaluate(candidate: Tensor, _theta_prediction: float, alpha: float) -> dict[str, Any]:
        check()
        trial = tangent._observe(shared, problem, candidate, parameters, weights)
        check()
        try:
            candidate_theta = float(tangent.minimum_mixture_weight(trial["side"][-1][1], trial["side"][1][1]))
        except ValueError:
            candidate_theta = math.nan
        valid_theta = math.isfinite(candidate_theta) and eps < candidate_theta < 1.0 - eps
        facts: dict[str, Any] = {"mixing_minimum_valid": valid_theta,
            "theta": candidate_theta if valid_theta else None,
            "branch_pair_passed": trial["pair"]["passed"], "face_audit_passed": trial["face_ok"],
            "side_objectives_match_native": trial["side_objectives_match_native"],
            "side_gradients_finite": trial["side_gradients_finite"],
            "side_gradients": {str(s): trial["side"][s][1].tolist() for s in (-1, 1)},
            "branch_trace": {str(s): trial["traces"][s] for s in (-1, 1)}}
        if not valid_theta:
            facts.update(finite=False, J_armijo_passed=False, F_squared_armijo_passed=False)
            return facts
        residual = torch.cat(((1.0 - candidate_theta) * trial["side"][-1][1]
            + candidate_theta * trial["side"][1][1], (trial["q"] / FACE_SCALE).reshape(1)))
        actual_j, actual_r = float(trial["native_j"]), float(torch.dot(residual, residual))
        j_bound, r_bound = prepared._armijo_bounds(float(observed["native_j"]), R, alpha,
            max(cost_slopes), model.merit_product)
        facts.update({"objective": actual_j, "F_squared": actual_r,
            "J_armijo_bound": j_bound, "J_armijo_passed": actual_j <= j_bound,
            "F_squared_armijo_bound": r_bound, "F_squared_armijo_passed": actual_r <= r_bound,
            "finite": math.isfinite(actual_j) and math.isfinite(actual_r)
                and bool(torch.isfinite(residual).all()), "G": residual.tolist()})
        return facts

    def persist(trials: list[dict[str, Any]], meta: dict[str, Any]) -> None:
        record.update(samples=trials, sample_progress=meta)
        tangent._write(output, record)
    sample_alphas = local_sample_alphas(model)
    record["sample_alphas"] = sample_alphas
    trials, progress = diagnostic_samples(model, control, sample_alphas, evaluate,
        chart_candidate=lambda point,vector,alpha:prepared.local._chart_candidate(point,weights,vector,pivot,alpha),
        progress=persist)
    after = shared._input_identity(problem,original,control,parameters,truth)
    source_after = shared._source_hashes(plan,plan_path,plan_sha)
    runtime_after = shared._runtime()
    if after!=input_before or source_after!=source_before or runtime_after!=runtime_before:
        raise ValueError("local diagnostic original identities changed")
    record.update(phase="finished",execution_status="completed",numerical_status="current_gn_local_diagnostic_complete_no_commit",
        source_after=source_after,input_after=after,runtime_after=runtime_after,source_unchanged=True,
        fixed_input_unchanged=True,runtime_unchanged=True,deadline_passed=time.monotonic()<deadline,
        optimizer_steps_applied=0,candidate_committed=False,readiness_complete=False)
    tangent._write(output,record)
    return record

def run(plan_path: Path, plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    _load_plan(plan_path, plan_sha)
    plan_path = plan_path.resolve()
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, output.with_suffix(".run.json"))):
        raise ValueError("released GN resume output paths must be fresh")
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
