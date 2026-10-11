"""Run at most three current-point GN corrections from the closed a3d2 point."""
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

from examples.weather_scenarios import fv_point_3h_gn_local_candidate_commit as prior
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_comparison as coupled
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios import fv_point_3h_search_progress as search_progress
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "GN_COMMITTED_CONTINUATION_PLAN_20261011.json"
OUTPUT_DIR = EVIDENCE / "gn_committed_continuation_20261011_attempt1"
SELF = "examples/weather_scenarios/fv_point_3h_gn_committed_continuation.py"
TEST = "tests/test_fv_point_3h_gn_committed_continuation.py"
PRIOR_PLAN_SHA = "db51873c025abc29e4b3af25b26e6c06027f5f3697ae064be1bec10968592db8"
BASE_SHA = "a3d2ad7bd5b358619c55ca8c52f9ffe15a276a1b0ec10fd4620e440bea30cc5a"
BASE_ARCHIVE = EVIDENCE / "GN_LOCAL_CANDIDATE_ARCHIVE_20261011.json"
BASE_GZIP = EVIDENCE / "gn_local_candidate_20261011_attempt1/step.json.gz"
BASE_RUN = EVIDENCE / "gn_local_candidate_20261011_attempt1/step.run.json"
BASE_RESOURCE = EVIDENCE / "gn_local_candidate_20261011_attempt1/step.resource.json"
BASE_ARCHIVE_SHA = "4920bb4793d00944f3e47032836939b7095cc15be31ce8ad1a945e90c34d69e8"
BASE_GZIP_SHA = "b01b50406beb164a1fa85c63680be8d0d7946b9750908b390a3369b0f03b96d5"
BASE_RUN_SHA = "7b334de0b80a42e86a52dab2ce7a5a72315434185a7e307024688215f19f5f99"
BASE_RESOURCE_SHA = "511d068a60fc97461a4150b59e1b11d5d21032874d250344c158f20911c65a5c"
BASE_RAW_SHA = "94d60f5df099b08e6b8c9ff315347f3f6705d82f6134c52fa5e5783cf7ba7f0b"
FACE_AXIS, FACE_ROW, FACE_COLUMN = "y", 4, 3
FACE = {"axis": FACE_AXIS, "row": FACE_ROW, "column": FACE_COLUMN}
FACE_SCALE, RADIUS = 0.84, 0.05
MAX_CANDIDATES = 24
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "optimizer_steps": 3, "max_commits": 3,
    "jacobian_row_vjp_calls": 72, "dense_solves": 3, "hvp_calls": 6,
    "candidate_count": 72, "candidate_cap_per_point": 24, "residual_norm_candidate_tolerance": 1e-8,
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
    if (plan.get("experiment_kind") != "bounded_committed_gn_continuation_with_progress"
            or plan.get("policy") != POLICY or plan.get("face") != FACE
            or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("predecessor_plan") != prior.PLAN.relative_to(ROOT).as_posix()
            or plan.get("predecessor_plan_sha256") != PRIOR_PLAN_SHA
            or plan.get("predecessor_raw_sha256") != BASE_RAW_SHA
            or plan.get("candidate_cap") != MAX_CANDIDATES
            or plan.get("radius") != RADIUS
            or set(plan.get("source_files", {})) != sources or len(plan["source_files"]) != 173
            or set(plan.get("archive_files", {})) != archives or len(plan["archive_files"]) != 248):
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
            raise ValueError(f"a3d2 parent evidence changed: {path.name}")
    manifest, run, resource = (json.loads(BASE_ARCHIVE.read_text()), json.loads(BASE_RUN.read_text()),
        json.loads(BASE_RESOURCE.read_text()))
    raw_bytes = gzip.decompress(BASE_GZIP.read_bytes())
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    prior_plan = prior._load_plan(prior.PLAN, PRIOR_PLAN_SHA)
    prior_source = shared._source_hashes(prior_plan, prior.PLAN, PRIOR_PLAN_SHA)
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
            or raw.get("execution_status") != "completed" or raw.get("plan_sha256") != PRIOR_PLAN_SHA
            or raw.get("source_before") != prior_source or raw.get("source_after") != prior_source
            or raw.get("current_control_sha256") != BASE_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_SHA
            or raw.get("numerical_status") != "local_candidate_committed_no_readiness"
            or raw.get("candidate_committed") is not True or raw.get("optimizer_steps_applied") != 1
            or raw.get("accepted_iterations") != 1 or raw.get("last_confirmed_iterations") != 1
            or raw.get("readiness_complete") is not False
            or raw.get("jacobian_rows_completed") != 0 or raw.get("dense_solves_completed") != 0
            or raw.get("hvp_calls_completed") != 2
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True or raw.get("deadline_passed") is not True
            or raw.get("last_confirmed_closure") != raw.get("final_repeat")
            or raw.get("last_confirmed_control") != raw.get("current_control")
            or tangent._tensor_sha(torch.as_tensor(raw.get("current_control"), dtype=torch.float64)) != BASE_SHA
            or not tangent.fresh_final_closure(raw["proposal"], raw["final_repeat"])):
        raise ValueError("closed a3d2 parent archive/P2/plan receipts do not match")
    selected_index = raw.get("selection", {}).get("index")
    selected_trial = next((trial for trial in raw.get("candidate_trials", [])
        if trial.get("index") == selected_index), None)
    if (not isinstance(selected_trial, dict)
            or selected_trial.get("control_sha256") != BASE_SHA
            or selected_trial.get("control") != raw.get("current_control")
            or not isinstance(selected_trial.get("G"), list) or len(selected_trial["G"]) != 27
            or selected_trial.get("F_squared") != raw["proposal"].get("F_squared")
            or selected_trial.get("theta") != raw["proposal"].get("theta")
            or raw["proposal"].get("control_sha256") != BASE_SHA):
        raise ValueError("parent selected candidate does not bind control, theta, R and full G")
    parent_G = torch.as_tensor(selected_trial["G"], dtype=torch.float64)
    parent_R = float(torch.dot(parent_G, parent_G))
    parent_R_saved = float(selected_trial["F_squared"])
    if (not bool(torch.isfinite(parent_G).all())
            or abs(parent_R - parent_R_saved) > 128 * torch.finfo(parent_G.dtype).eps
                * max(abs(parent_R), abs(parent_R_saved), torch.finfo(parent_G.dtype).tiny)):
        raise ValueError("parent full-G vector does not reproduce its recorded residual merit")
    return {key: raw[key] for key in ("current_control", "current_control_sha256", "current_theta",
        "final_repeat", "input_after", "runtime_after", "proposal")} | {"current_full_G": selected_trial["G"]}


def _base_model_closure(model: Any, full_residual: Tensor, base_theta: float) -> dict[str, Any]:
    residual_check = prepared.local._scaled_error(model.residual, full_residual, full_residual)
    theta_budget = 128 * torch.finfo(full_residual.dtype).eps * max(
        abs(float(model.theta)), abs(base_theta), torch.finfo(full_residual.dtype).tiny)
    theta_error = abs(float(model.theta) - base_theta)
    return {"full_G_scaled_match": residual_check, "theta_error": theta_error,
        "theta_budget": theta_budget,
        "passed": residual_check["passed"] is True and theta_error <= theta_budget}


def _full_G_match(actual: Tensor, expected: Tensor) -> dict[str, Any]:
    """Compare residual components, so equal norms cannot hide a changed vector."""
    if actual.shape != expected.shape or not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return {"passed": False, "reason": "shape_or_finiteness_mismatch"}
    return prepared.local._scaled_error(actual, expected, expected)


def _sanitize_json_leaves(value: Any) -> tuple[Any, bool]:
    """Make refusal facts JSON-safe and report whether every numeric leaf was finite."""
    if isinstance(value, float):
        return (value, True) if math.isfinite(value) else (None, False)
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        finite = True
        for key, item in value.items():
            result[key], item_finite = _sanitize_json_leaves(item)
            finite = finite and item_finite
        return result, finite
    if isinstance(value, (list, tuple)):
        sanitized = [_sanitize_json_leaves(item) for item in value]
        return [item for item, _ in sanitized], all(finite for _, finite in sanitized)
    return value, True


def _safe_candidate_facts(facts: dict[str, Any]) -> dict[str, Any]:
    safe, finite = _sanitize_json_leaves(facts)
    if not finite:
        safe["finite"] = False
        safe["J_armijo_passed"] = False
        safe["F_squared_armijo_passed"] = False
    return safe


def _commit_closed_point(record: dict[str, Any], point: dict[str, Any], proposal: dict[str, Any], repeat: dict[str, Any], output: Path) -> dict[str, Any]:
    if not tangent.fresh_final_closure(proposal, repeat):
        raise ValueError("independent point P2 failed")
    control = torch.as_tensor(repeat["control"], dtype=torch.float64)
    sha = tangent._tensor_sha(control)
    count = int(record["last_confirmed_iterations"]) + 1
    if count > POLICY["max_commits"]:
        raise ValueError("continuation commit cap exceeded")
    history = [*record["closed_steps"], {**point, "proposal": proposal, "final_repeat": repeat}]
    full_G = point.get("accepted_full_G")
    if not isinstance(full_G, list) or len(full_G) != 27:
        raise ValueError("committed point lacks its accepted full-G vector")
    full_G_tensor = torch.as_tensor(full_G, dtype=control.dtype)
    if not bool(torch.isfinite(full_G_tensor).all()):
        raise ValueError("committed point full-G vector is nonfinite")
    committed = {**record, "closed_steps": history, "current_control": control.tolist(),
        "current_control_sha256": sha, "current_theta": float(repeat["theta"]),
        "current_full_G": full_G_tensor.tolist(), "last_confirmed_full_G": full_G_tensor.tolist(),
        "last_confirmed_control": control.tolist(), "last_confirmed_control_sha256": sha,
        "last_confirmed_theta": float(repeat["theta"]), "last_confirmed_iterations": count,
        "accepted_iterations": count, "optimizer_steps_applied": count,
        "last_confirmed_closure": repeat, "candidate_committed": True,
        "phase": "running", "numerical_status": "confirmed_point_continuation_pending"}
    # Point data already lives in closed_steps after this successful atomic store.
    for key in ("candidate_trials", "search", "gn_model", "gn_direction", "selection", "gn_model_closure", "active_HVP_minus", "active_HVP_plus"):
        committed.pop(key, None)
    tangent._write(output, committed)
    return committed


def _persist_search_progress(record: dict[str, Any], output: Path, slots_before: int,
        trials: list[dict[str, Any]], search: dict[str, Any]) -> None:
    started = int(search["slots_started"])
    total = slots_before + started
    if total > POLICY["candidate_count"]:
        raise ValueError("continuation candidate budget exceeded")
    record.update(candidate_trials=[dict(trial) for trial in trials], search=dict(search),
        active_candidate_slots_used=started, candidate_slots_used=total)
    tangent._write(output, record)


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    start = time.monotonic(); deadline = start + POLICY["internal_seconds"]
    check = lambda: shared._deadline(deadline, POLICY["internal_seconds"])
    check(); plan = _load_plan(plan_path, plan_sha); plan_path = plan_path.resolve()
    source_before = shared._source_hashes(plan, plan_path, plan_sha)
    parent = _load_parent(); check()
    control = torch.as_tensor(parent["current_control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_SHA: raise ValueError("continuation base differs from a3d2")
    record: dict[str, Any] = {"phase": "running", "execution_status": "running", "policy": POLICY,
        "plan_sha256": plan_sha, "base_control_sha256": BASE_SHA,
        "current_control": control.tolist(), "current_control_sha256": BASE_SHA,
        "current_full_G": parent["current_full_G"], "last_confirmed_full_G": parent["current_full_G"],
        "current_theta": float(parent["current_theta"]), "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": BASE_SHA, "last_confirmed_theta": float(parent["current_theta"]),
        "last_confirmed_iterations": 0, "predecessor_commits_in_scope": 1,
        "accepted_iterations": 0, "optimizer_steps_applied": 0,
        "candidate_committed": False, "closed_steps": [], "point_index": 0,
        "jacobian_rows_started": 0, "jacobian_rows_completed": 0, "jacobian_row_history": [],
        "dense_solves_started": 0, "dense_solves_completed": 0,
        "hvp_calls_started": 0, "hvp_calls_completed": 0, "hvp_started_history": [], "hvp_history": [],
        "root_claim": False,
        "minimum_claim": False, "response_claim": False, "score_claim": False}
    tangent._write(output, record)
    check(); problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed(); check()
    initial_input = shared._input_identity(problem, original, control, parameters, truth)
    runtime_before = shared._runtime()
    if initial_input != parent["input_after"] or runtime_before != parent["runtime_after"]:
        raise ValueError("current original input/runtime differs from parent")
    weights = geometry._face_weights(problem, axis=FACE_AXIS, row=FACE_ROW, column=FACE_COLUMN)
    current_closure = parent["final_repeat"]
    def finish(status: str, reason: str = "", *, failed: bool = False) -> dict[str, Any]:
        source_after = shared._source_hashes(plan, plan_path, plan_sha)
        input_after = shared._input_identity(problem, original, control, parameters, truth)
        runtime_after = shared._runtime()
        source_ok = source_after == source_before
        input_ok = geometry.model._fixed_input(input_after, initial_input, tangent._tensor_sha(control))
        runtime_ok = runtime_after == runtime_before
        deadline_ok = time.monotonic() < deadline
        record.update(phase="finished", execution_status="failed" if failed else "completed", numerical_status=status,
            termination_reason=reason, source_after=source_after, input_after=input_after,
            runtime_after=runtime_after, source_unchanged=source_ok, fixed_input_unchanged=input_ok,
            runtime_unchanged=runtime_ok, deadline_passed=deadline_ok, readiness_complete=False,
            postcommit_readiness_performed=False, elapsed_seconds=time.monotonic()-start)
        tangent._write(output, record)
        if not all((source_ok, input_ok, runtime_ok)):
            raise ValueError("continuation terminal identity failed")
        return record
    try:
        for point_index in range(3):
            check(); record["point_index"] = point_index
            for key in ("candidate_trials", "search", "gn_model", "gn_direction", "selection", "gn_model_closure", "active_HVP_minus", "active_HVP_plus"):
                record.pop(key, None)
            input_before = shared._input_identity(problem, original, control, parameters, truth)
            if (not geometry.model._fixed_input(input_before, initial_input, tangent._tensor_sha(control))
                    or shared._runtime() != runtime_before
                    or shared._source_hashes(plan, plan_path, plan_sha) != source_before):
                raise ValueError("current point fixed input/source/runtime changed")
            rows_before = record["jacobian_rows_completed"]
            solves_before = record["dense_solves_completed"]
            check()
            observed = tangent._observe(shared, problem, control, parameters, weights)
            check()
            closure = current_closure
            theta = float(tangent.minimum_mixture_weight(observed["side"][-1][1], observed["side"][1][1]))
            G = torch.cat(((1.0 - theta) * observed["side"][-1][1] + theta * observed["side"][1][1],
                (observed["q"] / FACE_SCALE).reshape(1)))
            R = float(torch.dot(G, G))
            eps = 128 * torch.finfo(control.dtype).eps
            full_G_check = _full_G_match(G, torch.as_tensor(record["current_full_G"], dtype=control.dtype))
            record["current_full_G_check"] = full_G_check
            if full_G_check.get("passed") is not True:
                raise ValueError("fresh current full-G vector differs componentwise from the carried closed point")
            if (not shared._gradient_pair_match({s: observed["side"][s][1] for s in (-1, 1)},
                    {s: torch.as_tensor(closure["side_gradients"][str(s)], dtype=control.dtype) for s in (-1, 1)})
                    or not observed["pair"]["passed"] or not observed["face_ok"]
                    or not observed["side_objectives_match_native"] or not observed["side_gradients_finite"]
                    or abs(float(observed["native_j"]) - float(closure["objective"])) > eps * abs(float(closure["objective"]))
                    or abs(theta - float(closure["theta"])) > eps * abs(float(closure["theta"]))
                    or abs(R - float(closure["F_squared"])) > eps * abs(float(closure["F_squared"]))
                    or any(observed["traces"][s]["signature_sha256"] != closure["branch_trace"][str(s)]["signature_sha256"]
                        for s in (-1, 1))):
                raise ValueError("fresh current point J/G/theta/strict pair/trace qualification failed")
            if math.sqrt(R) <= POLICY["residual_norm_candidate_tolerance"]:
                record["residual_candidate_R"] = R
                return finish("residual_candidate_requires_final_qualification")
            pivot, normal = geometry._pivot(control, weights), geometry._face_normal(control, weights)
            _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
            record["active_base_control_sha256"] = tangent._tensor_sha(control)
            record.update({"source_before": source_before, "input_before": input_before,
                "runtime_before": runtime_before, "base_J": float(observed["native_j"]),
                "base_R": R, "base_theta": theta, "base_full_mixed_residual": G.tolist()})

            state = {"observed": observed}
            context = {"problem": problem, "parameters": parameters, "record": record,
                "output": output, "deadline": deadline, "weights": weights,
                "plan": plan, "plan_path": plan_path, "plan_sha": plan_sha,
                "model_state": state}
            specs = coupled._direction_model_factory(context)(control, theta,
                observed["side"][-1][1], observed["side"][1][1], normal, chart, pivot)
            gn_specs = [spec for spec in specs if spec.get("name") == "robust_gn_coupled"]
            if (len(gn_specs) != 1 or record["jacobian_rows_completed"] != rows_before + 24
                    or record["dense_solves_completed"] != solves_before + 1):
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
                label = {"phase": "current_gn_continuation", "side": side,
                    "arm": "prepared_gn", "base_control_sha256": tangent._tensor_sha(control),
                    "point_index": point_index, "direction_sha256": tangent._tensor_sha(direction),
                    "reference_theta": theta, "working_theta": float(spec["working_theta"]),
                    "operator": "selected_face_extension", "status": "started"}
                if record["hvp_calls_started"] >= POLICY["hvp_calls"]:
                    raise ValueError("continuation HVP budget exceeded")
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

            record["active_HVP_minus"] = HVP[-1].tolist()
            record["active_HVP_plus"] = HVP[1].tolist()
            try:
                model = spec["model_builder"](observed["side"][-1][1], observed["side"][1][1],
                    HVP[-1], HVP[1], normal, chart, theta, float(observed["q"]), pivot)
            except ValueError as error:
                record["gn_model"] = {"supported": False, "model_error": str(error),
                    "base_full_residual": G.tolist(), "base_theta": theta}
                return finish("current_model_unsupported", str(error))
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
                return finish("current_model_unsupported", "current cost/merit descent gates refused")
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
                    return _safe_candidate_facts(facts)
                residual = torch.cat(((1.0 - candidate_theta) * trial["side"][-1][1]
                    + candidate_theta * trial["side"][1][1], (trial["q"] / FACE_SCALE).reshape(1)))
                actual_j, actual_r = float(trial["native_j"]), float(torch.dot(residual, residual))
                j_bound, r_bound = prepared._armijo_bounds(float(observed["native_j"]), R, alpha,
                    max(cost_slopes), model.merit_product)
                bounds_finite = math.isfinite(j_bound) and math.isfinite(r_bound)
                facts.update({"objective": actual_j, "F_squared": actual_r,
                    "J_armijo_bound": j_bound, "J_armijo_passed": bounds_finite and actual_j <= j_bound,
                    "F_squared_armijo_bound": r_bound, "F_squared_armijo_passed": bounds_finite and actual_r <= r_bound,
                    "finite": math.isfinite(actual_j) and math.isfinite(actual_r)
                        and bounds_finite and bool(torch.isfinite(residual).all()), "G": residual.tolist()})
                return _safe_candidate_facts(facts)

            slots_before = int(record.get("candidate_slots_used", 0))
            def persist_search(trials: list[dict[str, Any]], search: dict[str, Any]) -> None:
                _persist_search_progress(record, output, slots_before, trials, search)
            accepted, trials, search = search_progress.search_candidates(model, control, evaluate,
                chart_candidate=lambda point, vector, alpha: prepared.local._chart_candidate(
                    point, weights, vector, pivot, alpha), radius=RADIUS, limit=MAX_CANDIDATES, progress=persist_search)
            record.update({"candidate_trials": trials, "search": search,
                "active_candidate_slots_used": int(search["evaluated_count"])})
            record["candidate_slots_used"] = slots_before + int(search["evaluated_count"])
            if record["candidate_slots_used"] > POLICY["candidate_count"]:
                raise ValueError("continuation candidate budget exceeded")
            if accepted is None:
                return finish("candidate_grid_exhausted", "24-slot grid exhausted")
            accepted_control = torch.as_tensor(accepted["control"], dtype=control.dtype)
            obs = {"native_j": accepted["objective"], "face_ok": accepted["face_audit_passed"],
                "pair": {"passed": accepted["branch_pair_passed"]},
                "side_objectives_match_native": accepted["side_objectives_match_native"],
                "side_gradients_finite": accepted["side_gradients_finite"],
                "side": {s: (None, torch.as_tensor(accepted["side_gradients"][str(s)], dtype=control.dtype))
                    for s in (-1, 1)}, "traces": {s: accepted["branch_trace"][str(s)] for s in (-1, 1)}}
            proposal = prepared.gni._candidate_proposal(accepted_control, obs, float(accepted["theta"]),
                float(accepted["F_squared"]), float(accepted["alpha"]),
                float(accepted["J_armijo_bound"]), float(accepted["F_squared_armijo_bound"]))
            proposal["selected_arm"] = "prepared_gn"
            proposal["selected_direction_sha256"] = tangent._tensor_sha(direction)
            check()
            repeat = prepared.gni._final_repeat(shared, geometry, tangent, plan, plan_path, plan_sha,
                problem, original, parameters, truth, weights, proposal, input_before, runtime_before,
                source_before, deadline)
            check()
            if not tangent.fresh_final_closure(proposal, repeat):
                raise ValueError("released GN candidate failed independent P2; last confirmed point remains current")
            record["selection"] = {"arm": "prepared_gn", "direction_sha256": tangent._tensor_sha(direction),
                "alpha": float(accepted["alpha"]), "actual_J": float(accepted["objective"]),
                "actual_R": float(accepted["F_squared"]), "first_pass_count": search["first_pass_count"]}
            point = {key: record[key] for key in ("point_index", "base_J", "base_R", "base_theta",
                "base_full_mixed_residual", "gn_direction", "gn_model", "candidate_trials", "search", "selection")}
            point["base_control_sha256"] = tangent._tensor_sha(control)
            point["HVP_minus"] = HVP[-1].tolist()
            point["HVP_plus"] = HVP[1].tolist()
            point["accepted_full_G"] = accepted["G"]
            point["row_calls_completed_at_point"] = record["jacobian_rows_completed"]-rows_before
            point["dense_solves_completed_at_point"] = record["dense_solves_completed"]-solves_before
            record = _commit_closed_point(record, point, proposal, repeat, output)
            control = torch.as_tensor(repeat["control"], dtype=torch.float64)
            current_closure = repeat
        return finish("gn_continuation_cap_reached")
    except TimeoutError as error:
        return finish("internal_time_budget_exhausted", str(error))
    except Exception as error:
        record["failure_type"] = type(error).__name__
        record["failure_message"] = str(error)
        finish("current_point_failure_last_commit_preserved", str(error), failed=True)
        raise

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
