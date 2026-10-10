"""Rebuild the current GN arm and compare event projection at the 48b0 point."""
from __future__ import annotations

import argparse
from dataclasses import replace
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

from advar import transport, variational
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_event_direction_comparison as previous
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared
from examples.weather_scenarios import fv_point_3h_limiter_event_diagnostic as event_diag
from examples.weather_scenarios import fv_point_3h_limiter_event_probe as event
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_comparison as coupled
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "EVENT_DIRECTION_RESUME_PLAN_20261010.json"
OUTPUT_DIR = EVIDENCE / "event_direction_resume_20261010_attempt1"
SELF = "examples/weather_scenarios/fv_point_3h_event_direction_resume.py"
TEST = "tests/test_fv_point_3h_event_direction_resume.py"
COMPARE_PLAN_SHA = "14773f392f41eaf9cf0fdba2b59bbf3427f8d3fdf0d0e7cc92b266704928ea29"
BASE_SHA = "48b0f95c5aa485a6a17046d4497a2267a389b1cfb2eeedb75d79e7ba038d5729"
BASE_EVENT_DIRECTION_SHA = "c9ac394644130f62f0635a753346ad9d4846c08ad59515bde99ab1b3280c1913"
COMPARE_ARCHIVE = EVIDENCE / "EVENT_DIRECTION_ARCHIVE_20261010.json"
COMPARE_GZIP = EVIDENCE / "event_direction_comparison_20261010_attempt1/step.json.gz"
COMPARE_RUN = EVIDENCE / "event_direction_comparison_20261010_attempt1/step.run.json"
COMPARE_RESOURCE = EVIDENCE / "event_direction_comparison_20261010_attempt1/step.resource.json"
COMPARE_AUDIT = EVIDENCE / "EVENT_DIRECTION_SAVED_AUDIT_20261010.json"
COMPARE_ARCHIVE_SHA = "aa49577d6d675470e721d7ab6d406e656121920bee88c4bf58c6a948723da679"
COMPARE_GZIP_SHA = "ef7ddabab034d5ddb0402687ba51247329cae1caf4eb3dea184fc7a0dedb1421"
COMPARE_RUN_SHA = "367c22a689d921f034ad5728d0b9a70ddf6dc9224753c4ad7d3f2069a6331446"
COMPARE_RESOURCE_SHA = "c4bfd995f38e47593c665f23e5b527919b70fb8506c96ffb0f40289f99b83263"
COMPARE_AUDIT_SHA = "38f74ada00a9a84e5708739a7e601a7697d0d4841fbcd342241c08f03d9e141e"
COMPARE_RAW_SHA = "0192d42de82ef0e645e64b2445a8c7ed84feeafb9cd76a142e225eb8ec7348b9"
FACE_AXIS, FACE_ROW, FACE_COLUMN = "y", 4, 3
FACE = {"axis": FACE_AXIS, "row": FACE_ROW, "column": FACE_COLUMN}
FACE_SCALE = 0.84
RADIUS = 0.05
MAX_CANDIDATES_PER_ARM = 24
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "optimizer_steps": 1, "max_commits": 1,
    "jacobian_row_vjp_calls": 24, "dense_solves": 1, "hvp_calls": 4,
    "event_gradient_calls": 2, "event_jvp_calls": 2,
    "candidate_cap_per_arm": MAX_CANDIDATES_PER_ARM, "candidate_count": 48,
    "radius": RADIUS, "face_scale": FACE_SCALE,
    "selection_tie_rule": previous.POLICY["selection_tie_rule"],
    "root_claim": False, "minimum_claim": False, "score_claim": False,
    "response_claim": False,
}
ARM_ORDER = previous.ARM_ORDER


def _sha(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"pinned path must not be a symbolic link: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned path escapes repository: {path}")
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("event-direction resume plan identity mismatch")
    prior = previous._load_plan(previous.PLAN, COMPARE_PLAN_SHA)
    extra_archives = {previous.PLAN.relative_to(ROOT).as_posix(), COMPARE_ARCHIVE.relative_to(ROOT).as_posix(),
        COMPARE_GZIP.relative_to(ROOT).as_posix(), COMPARE_RUN.relative_to(ROOT).as_posix(),
        COMPARE_RESOURCE.relative_to(ROOT).as_posix(), COMPARE_AUDIT.relative_to(ROOT).as_posix()}
    plan = json.loads(path.read_text())
    expected_sources = set(prior["source_files"]) | {SELF, TEST}
    expected_archives = set(prior["archive_files"]) | extra_archives
    if (plan.get("experiment_kind") != "event_direction_gn_resume_comparison"
            or plan.get("policy") != POLICY or plan.get("face") != FACE
            or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("predecessor_plan") != previous.PLAN.relative_to(ROOT).as_posix()
            or plan.get("predecessor_plan_sha256") != COMPARE_PLAN_SHA
            or plan.get("predecessor_raw_sha256") != COMPARE_RAW_SHA
            or plan.get("predecessor_selected_direction_sha256") != BASE_EVENT_DIRECTION_SHA
            or plan.get("candidate_cap_per_arm") != MAX_CANDIDATES_PER_ARM
            or plan.get("radius") != RADIUS
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 158
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 213):
        raise ValueError("event-direction resume plan scope or policy changed")
    for key in ("source_files", "archive_files"):
        if any(plan[key].get(name) != value for name, value in prior[key].items()):
            raise ValueError(f"resume plan changed inherited {key} pins")
        for name, value in plan[key].items():
            if _sha(ROOT / name) != value:
                raise ValueError(f"resume {key} pin mismatch: {name}")
    return plan


def _load_parent() -> dict[str, Any]:
    for path, digest in ((COMPARE_ARCHIVE, COMPARE_ARCHIVE_SHA), (COMPARE_GZIP, COMPARE_GZIP_SHA),
            (COMPARE_RUN, COMPARE_RUN_SHA), (COMPARE_RESOURCE, COMPARE_RESOURCE_SHA),
            (COMPARE_AUDIT, COMPARE_AUDIT_SHA)):
        if _sha(path) != digest:
            raise ValueError(f"PR281 parent receipt changed: {path.name}")
    manifest = json.loads(COMPARE_ARCHIVE.read_text())
    run = json.loads(COMPARE_RUN.read_text())
    resource = json.loads(COMPARE_RESOURCE.read_text())
    audit = json.loads(COMPARE_AUDIT.read_text())
    raw_bytes = gzip.decompress(COMPARE_GZIP.read_bytes())
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    if (raw_sha != COMPARE_RAW_SHA or manifest.get("raw_sha256") != raw_sha
            or manifest.get("gzip_sha256") != COMPARE_GZIP_SHA
            or manifest.get("run_sha256") != COMPARE_RUN_SHA
            or manifest.get("resource_sha256") != COMPARE_RESOURCE_SHA
            or manifest.get("lossless_roundtrip") is not True
            or run.get("execution_status") != "completed" or run.get("child_sha256") != raw_sha
            or run.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False or resource.get("wall_limit_seconds") != 660.0
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes", POLICY["rss_bytes"]) >= POLICY["rss_bytes"]
            or raw.get("plan_sha256") != COMPARE_PLAN_SHA
            or raw.get("execution_status") != "completed"
            or raw.get("current_control_sha256") != BASE_SHA
            or raw.get("numerical_status") != "event_direction_comparison_committed_one_selected_candidate_no_readiness"
            or raw.get("candidate_committed") is not True or raw.get("optimizer_steps_applied") != 1
            or raw.get("readiness_complete") is not False or raw.get("postcommit_readiness_performed") is not False
            or raw.get("jacobian_rows_completed") != 0 or raw.get("dense_solves_completed") != 0
            or raw.get("hvp_calls_completed") != 4 or not tangent.fresh_final_closure(raw["proposal"], raw["final_repeat"])
            or not audit.get("all_checks_passed") or audit.get("plan_sha256") != COMPARE_PLAN_SHA
            or raw.get("selection", {}).get("direction_sha256") != BASE_EVENT_DIRECTION_SHA
            or raw.get("selection", {}).get("comparison_complete") is not True):
        raise ValueError("PR281 48b0 parent receipt or saved audit does not close")
    chosen_arm = raw["selection"]["arm"]
    return {key: raw[key] for key in ("current_control", "current_control_sha256", "current_theta",
        "final_repeat", "input_after", "runtime_after", "selection", "proposal")} | {
        "selected_direction_sha256": raw["arms"][chosen_arm]["direction_sha256"],
        "selected_event_values": raw["arms"][chosen_arm]["accepted_event_values"]}


def _event_support_reason(values: Tensor, trace: dict[str, Any]) -> str | None:
    if values.shape != (3,) or not bool(torch.isfinite(values).all() & (values[2] > 0)):
        return "event values or q scale are nonfinite/unresolved"
    if not bool((values[:2] < 0).all() & (values[0] < values[1])):
        return "event no longer has the supported same-negative right-selected orientation"
    if abs(float(values[0] - values[1])) / float(values[2]) <= 128 * torch.finfo(values.dtype).eps:
        return "event gap is unresolved relative to q scale"
    try:
        choice = trace["choices"][124][0]
        if (choice["left_sign"][1][2] != -1 or choice["right_sign"][1][2] != -1
                or choice["slope_sign"][1][2] != -1 or choice["choose_left"][1][2] is not False):
            return "fresh branch trace no longer selects the supported event side"
    except (KeyError, IndexError, TypeError):
        return "fresh branch trace is missing the event selector"
    return None


def _commit_record(output: Path, record: dict[str, Any],
                   proposal: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    return previous._persist_selected_commit(output, record, proposal, repeat)


def _base_model_closure(model: Any, base_residual: Tensor, base_theta: float) -> dict[str, Any]:
    residual = model.residual
    residual_error = prepared.local._scaled_error(residual, base_residual, base_residual)
    theta_budget = 128 * torch.finfo(base_residual.dtype).eps * max(
        abs(float(model.theta)), abs(base_theta), torch.finfo(base_residual.dtype).tiny)
    theta_error = abs(float(model.theta) - base_theta)
    return {"residual": residual_error, "theta_error": theta_error,
        "theta_budget": theta_budget,
        "passed": residual_error["passed"] is True and theta_error <= theta_budget}


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
        raise ValueError("PR281 base control differs from frozen 48b0 point")
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "event_direction_resume_preflight", "policy": POLICY,
        "plan_sha256": plan_sha, "base_control_sha256": BASE_SHA,
        "current_control": control.tolist(), "current_control_sha256": BASE_SHA,
        "current_theta": float(parent["current_theta"]), "last_confirmed_control": control.tolist(),
        "last_confirmed_control_sha256": BASE_SHA, "last_confirmed_theta": float(parent["current_theta"]),
        "last_confirmed_iterations": 0, "predecessor_commits_in_scope": 1,
        "candidate_committed": False, "optimizer_steps_applied": 0,
        "jacobian_rows_started": 0, "jacobian_rows_completed": 0, "jacobian_row_history": [],
        "dense_solves_started": 0, "dense_solves_completed": 0,
        "hvp_calls_started": 0, "hvp_calls_completed": 0,
        "hvp_started_history": [], "hvp_history": [],
        "event_gradient_calls_started": 0, "event_gradient_calls_completed": 0,
        "event_jvp_calls_started": 0, "event_jvp_calls_completed": 0,
        "root_claim": False, "minimum_claim": False, "score_claim": False, "response_claim": False}
    tangent._write(output, record)
    check()
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    check()
    input_before = shared._input_identity(problem, original, control, parameters, truth)
    runtime_before = shared._runtime()
    if input_before != parent["input_after"] or runtime_before != parent["runtime_after"]:
        raise ValueError("fixed input/runtime differs from PR281 48b0 receipt")
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
        raise ValueError("fresh 48b0 J/G/theta/trace qualification failed")
    pivot, normal = geometry._pivot(control, weights), geometry._face_normal(control, weights)
    _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    record.update(source_before=source_before, input_before=input_before, runtime_before=runtime_before,
        base_J=float(observed["native_j"]), base_R=R, base_theta=theta,
        base_residual=G.tolist(), base_side_gradients={str(s): observed["side"][s][1].tolist() for s in (-1, 1)})

    # The current coupled factory computes 24 residual rows and one dense solve,
    # but deliberately does not form an HVP before the event-normal projection.
    model_state = {"observed": observed}
    context = {"problem": problem, "parameters": parameters, "record": record,
        "output": output, "deadline": deadline, "weights": weights,
        "plan": plan, "plan_path": plan_path, "plan_sha": plan_sha,
        "model_state": model_state}
    specs = coupled._direction_model_factory(context)(control, theta,
        observed["side"][-1][1], observed["side"][1][1], normal, chart, pivot)
    gn_specs = [spec for spec in specs if spec.get("name") == "robust_gn_coupled"]
    if len(gn_specs) != 1 or record["jacobian_rows_completed"] != 24 or record["dense_solves_completed"] != 1:
        raise ValueError("fresh current-point GN factory failed its 24-row/one-solve contract")
    gn_spec = gn_specs[0]
    gn_direction = gn_spec["direction"]
    record["gn_factory"] = {"direction": gn_direction.tolist(),
        "direction_sha256": tangent._tensor_sha(gn_direction),
        "row_vjps_completed": record["jacobian_rows_completed"],
        "dense_solve_audit": record.get("dense_solve_audit"),
        "working_theta": gn_spec["working_theta"], "diagnostics": gn_spec["diagnostics"]}
    if not bool(torch.isfinite(gn_direction).all()):
        raise ValueError("fresh current-point GN direction is nonfinite")

    # Re-evaluate the PR280 event with the same trajectory, disabling only tape
    # replay; the problem and its frozen original replay setting remain intact.
    contract = problem.contract(parameters)
    fv = contract.fv_transport
    if fv is None:
        raise ValueError("fixed 3-hour FV contract is missing")
    direct = replace(contract, fv_transport=replace(fv, replay=False))
    trajectory = lambda point: variational.analysis_trajectory(point, direct)
    event_data: dict[str, Any] = {}
    event_gradients: dict[int, Tensor] = {}
    plain_event_values: dict[int, Tensor] = {}
    event_supported, event_unsupported_reason = True, None
    parent_events = parent["selected_event_values"]
    for side in (-1, 1):
        check()
        function = lambda point, sign=side: event.event_values(trajectory, point, sign, event.EVENT, 360)
        try:
            original_values = event_diag._plain_values(problem, control, parameters, side)
            direct_values = function(control)
            parent_values = torch.as_tensor(parent_events[str(side)]["values"], dtype=control.dtype)
            parity_scale = torch.maximum(direct_values.abs(), parent_values.abs())
            parity_budget = 256 * torch.finfo(control.dtype).eps * parity_scale.clamp_min(
                torch.finfo(control.dtype).tiny)
            if (not torch.equal(original_values, direct_values)
                    or not bool(((direct_values - parent_values).abs() <= parity_budget).all())):
                raise ValueError(f"fresh event values disagree with original/PR281 side {side}")
            reason = _event_support_reason(direct_values, observed["traces"][side])
            if reason is not None:
                raise ValueError(reason)
            with torch.no_grad(), transport.selected_face_extension(FACE_AXIS, FACE_ROW, FACE_COLUMN, side):
                original_frames = variational.analysis_trajectory(control, contract).frames_linear
                direct_frames = variational.analysis_trajectory(control, direct).frames_linear
            if not torch.equal(original_frames, direct_frames):
                raise ValueError("direct/original analysis trajectories differ at 48b0")
            del original_frames, direct_frames
            plain_event_values[side] = direct_values
            event_data[str(side)] = {"values": direct_values.tolist(),
                "original_values": original_values.tolist(), "parent_values": parent_values.tolist(),
                "parent_value_budget": parity_budget.tolist(), "value_parity_exact": True,
                "analysis_frames_parity_exact": True}
        except (ValueError, RuntimeError) as error:
            event_supported = False
            event_unsupported_reason = f"{type(error).__name__}: {error}"
            break
        check()
    if event_supported:
        for side in (-1, 1):
            check()
            function = lambda point, sign=side: event.event_values(trajectory, point, sign, event.EVENT, 360)
            record["event_gradient_calls_started"] += 1
            record["event_jvp_calls_started"] += 1
            tangent._write(output, record)
            try:
                values, directional, zeta_directional, raw_gradient = event.event_derivatives(
                    function, control, gn_direction)
                record["event_gradient_calls_completed"] += 1
                record["event_jvp_calls_completed"] += 1
                if not bool(torch.isfinite(values).all() & torch.isfinite(directional).all()
                            & torch.isfinite(raw_gradient).all() & torch.isfinite(zeta_directional)):
                    raise ValueError("new-point event values/gradient/JVP are nonfinite")
                expected_slope = directional[0] - directional[1]
                gradient_slope = torch.dot(raw_gradient, gn_direction)
                agreement = prepared.local._scaled_error(gradient_slope.reshape(1), expected_slope.reshape(1),
                    (raw_gradient.abs() * gn_direction.abs()).sum().reshape(1))
                if agreement["passed"] is not True or not torch.equal(values, plain_event_values[side]):
                    raise ValueError("new-point event gradient/JVP or value parity failed")
                event_gradients[side] = raw_gradient
                event_data[str(side)].update({"jvp_values": directional.tolist(),
                    "zeta": float(values[0] - values[1]),
                    "zeta_directional_derivative": float(zeta_directional),
                    "gradient": raw_gradient.tolist(), "gradient_dot_direction": float(gradient_slope),
                    "jvp_gradient_dot_check": agreement})
            except (ValueError, RuntimeError) as error:
                event_supported = False
                event_unsupported_reason = f"{type(error).__name__}: {error}"
                break
            tangent._write(output, record)
            check()
    event_normal: Tensor | None = None
    event_checks: dict[str, Any] = {}
    if event_supported and set(event_gradients) == {-1, 1}:
        try:
            event_normal, event_checks = previous._common_event_tangent_gradient({"event_derivatives": {
                str(s): {"gradient": event_gradients[s].tolist(),
                    "tangent_gradient": (event_gradients[s] - normal *
                        (torch.dot(normal, event_gradients[s]) / torch.dot(normal, normal))).tolist()}
                for s in (-1, 1)}}, normal)
        except Exception as error:
            event_supported = False
            event_unsupported_reason = f"{type(error).__name__}: {error}"
    projected_direction: Tensor | None = None
    event_projection: dict[str, Any] = {"supported": False,
        "unsupported_reason": event_unsupported_reason, "normalized": False}
    if event_supported and event_normal is not None:
        try:
            projected_direction = previous._project_away(gn_direction, event_normal)
            event_normal_dot = float(torch.dot(event_normal, projected_direction))
            event_normal_budget = 128 * torch.finfo(control.dtype).eps * float(
                torch.linalg.vector_norm(event_normal) * torch.linalg.vector_norm(projected_direction))
            face_dot = float(torch.dot(normal, projected_direction))
            face_budget = 128 * torch.finfo(control.dtype).eps * float(
                torch.linalg.vector_norm(normal) * torch.linalg.vector_norm(projected_direction))
            if abs(event_normal_dot) > event_normal_budget or abs(face_dot) > face_budget:
                raise ValueError("new-point event projection lost event or selected-face tangency")
            event_projection = {"supported": True, "baseline_gn_direction": gn_direction.tolist(),
                "event_orthogonal_direction": projected_direction.tolist(),
                "event_normal_dot": event_normal_dot, "event_normal_budget": event_normal_budget,
                "face_normal_dot": face_dot, "face_normal_budget": face_budget, "normalized": False}
        except Exception as error:
            event_supported = False
            event_unsupported_reason = f"{type(error).__name__}: {error}"
            projected_direction = None
            event_projection = {"supported": False, "unsupported_reason": event_unsupported_reason,
                "normalized": False}
    if event_normal is not None:
        event_component = float(torch.dot(event_normal, G[:26]))
        event_unit_component = event_component / float(torch.linalg.vector_norm(event_normal))
        event_squared_fraction = event_component**2 / float(torch.dot(event_normal, event_normal)) / R
        event_projection.update({"full_mixed_gradient_event_component": event_unit_component,
            "full_R_event_component_squared_fraction": event_squared_fraction})
    directions = {"prepared_gn": gn_direction}
    if event_supported and projected_direction is not None:
        directions["event_orthogonal"] = projected_direction
    record.update({"direct_contract": {"original_replay": fv.replay, "diagnostic_replay": direct.fv_transport.replay,
            "frame_parity_checked": len(plain_event_values) == 2,
            "event_value_parity_checked": len(plain_event_values) == 2},
        "event_supported": event_supported, "event_unsupported_reason": event_unsupported_reason,
        "event_derivatives": event_data, "event_tangent_gradient": None if event_normal is None else event_normal.tolist(),
        "event_gradient_projection_checks": event_checks, "event_projection": event_projection,
        "arms": {}, "selection": None})

    for arm_name in ARM_ORDER:
        if arm_name not in directions:
            record["arms"][arm_name] = {"supported": False,
                "unsupported_reason": event_unsupported_reason or "event gradient unavailable",
                "hvp_calls_started": 0, "hvp_calls_completed": 0,
                "accepted": None, "trials": [], "search": {}}
            continue
        direction = directions[arm_name]
        arm: dict[str, Any] = {"direction": direction.tolist(), "direction_sha256": tangent._tensor_sha(direction),
            "supported": True, "hvp_calls_started": 0, "hvp_calls_completed": 0,
            "hminus": None, "hplus": None, "accepted": None, "trials": [], "search": {}}
        record["arms"][arm_name] = arm
        hvps: dict[int, Tensor] = {}
        for side in (-1, 1):
            check()
            reference_working_theta = float(gn_spec.get("working_theta", theta))
            label = {"phase": "current_48b0_comparison", "arm": arm_name, "side": side,
                "base_control_sha256": BASE_SHA, "direction_sha256": arm["direction_sha256"],
                "reference_theta": theta, "working_theta": reference_working_theta,
                "operator": "selected_face_extension", "direction_model": arm_name,
                "status": "started"}
            record["hvp_calls_started"] += 1
            arm["hvp_calls_started"] += 1
            record["hvp_started_history"].append(label)
            tangent._write(output, record)
            try:
                with transport.selected_face_extension(FACE_AXIS, FACE_ROW, FACE_COLUMN, side):
                    hvp = torch.func.jvp(lambda point: torch.func.grad(problem.objective, argnums=0)(point, parameters),
                        (control,), (direction,))[1]
                if hvp.shape != control.shape or not bool(torch.isfinite(hvp).all()):
                    raise ValueError("nonfinite or wrong-shape HVP")
                hvps[side] = hvp
                record["hvp_calls_completed"] += 1
                arm["hvp_calls_completed"] += 1
                complete = {**label, "status": "completed"}
            except Exception as error:
                arm["supported"] = False
                arm["unsupported_reason"] = f"{type(error).__name__}: {error}"
                complete = {**label, "status": "failed", "error": arm["unsupported_reason"]}
            complete.update({"source_unchanged": shared._source_hashes(plan, plan_path, plan_sha) == source_before,
                "input_unchanged": shared._input_identity(problem, original, control, parameters, truth) == input_before,
                "runtime_unchanged": shared._runtime() == runtime_before})
            if not all((complete["source_unchanged"], complete["input_unchanged"], complete["runtime_unchanged"])):
                raise ValueError("source/input/runtime changed during current-point HVP")
            record["hvp_history"].append(complete)
            tangent._write(output, record)
            check()
        if not arm["supported"] or set(hvps) != {-1, 1}:
            arm["supported"] = False
            tangent._write(output, record)
            continue
        arm["hminus"], arm["hplus"] = hvps[-1].tolist(), hvps[1].tolist()
        try:
            if arm_name == "prepared_gn":
                model = gn_spec["model_builder"](observed["side"][-1][1], observed["side"][1][1],
                    hvps[-1], hvps[1], normal, chart, theta, float(observed["q"]), pivot)
            else:
                model = mixing.minimum_tangent_model(observed["side"][-1][1], observed["side"][1][1],
                    hvps[-1], hvps[1], normal, chart, theta, float(observed["q"]), pivot,
                    FACE_SCALE, direction_override=direction)
        except ValueError as error:
            arm.update(supported=False, unsupported_reason=f"model closure refused: {error}")
            tangent._write(output, record)
            continue
        if (not torch.equal(model.direction, direction)
                or not all(model.gates.get(key) is True for key in (
                    "finite", "both_side_gradients_descend", "merit_descends",
                    "envelope_slope_matches_residual_dot"))):
            arm.update(supported=False, unsupported_reason="fresh direction model failed its numerical gates",
                gates=model.gates)
            tangent._write(output, record)
            continue
        base_model_check = _base_model_closure(model, G, theta)
        arm["base_model_closure"] = base_model_check
        if base_model_check["passed"] is not True:
            arm.update(supported=False, unsupported_reason="current model residual/theta differs from fresh base")
            tangent._write(output, record)
            continue
        cost_slopes: tuple[float, float] = (
            float(torch.dot(observed["side"][-1][1], direction)),
            float(torch.dot(observed["side"][1][1], direction)))
        arm.update({"working_theta": model.theta, "theta_prime": model.delta_theta,
            "residual": model.residual.tolist(), "residual_direction": model.residual_direction.tolist(),
            "merit_product": model.merit_product, "side_cost_slopes": list(cost_slopes),
            "selected_J_slope": max(cost_slopes), "gates": model.gates})

        def evaluate(candidate: Tensor, _theta_prediction: float, alpha: float) -> dict[str, Any]:
            check()
            trial = tangent._observe(shared, problem, candidate, parameters, weights)
            check()
            try:
                candidate_theta = float(tangent.minimum_mixture_weight(
                    trial["side"][-1][1], trial["side"][1][1]))
            except ValueError:
                candidate_theta = math.nan
            valid_theta = math.isfinite(candidate_theta) and eps < candidate_theta < 1.0 - eps
            result: dict[str, Any] = {"mixing_minimum_valid": valid_theta,
                "theta": candidate_theta if valid_theta else None,
                "branch_pair_passed": trial["pair"]["passed"], "face_audit_passed": trial["face_ok"],
                "side_objectives_match_native": trial["side_objectives_match_native"],
                "side_gradients_finite": trial["side_gradients_finite"],
                "side_gradients": {str(s): trial["side"][s][1].tolist() for s in (-1, 1)},
                "branch_trace": {str(s): trial["traces"][s] for s in (-1, 1)},
                "direction_sha256": arm["direction_sha256"]}
            if not valid_theta:
                result.update(finite=False, J_armijo_passed=False, F_squared_armijo_passed=False)
                return result
            residual = torch.cat(((1 - candidate_theta) * trial["side"][-1][1]
                + candidate_theta * trial["side"][1][1], (trial["q"] / FACE_SCALE).reshape(1)))
            actual_j, actual_r = float(trial["native_j"]), float(torch.dot(residual, residual))
            j_bound, r_bound = previous._armijo_bounds(float(observed["native_j"]), R,
                alpha, cost_slopes, model.merit_product)
            result.update({"objective": actual_j, "F_squared": actual_r,
                "J_armijo_bound": j_bound, "J_armijo_passed": actual_j <= j_bound,
                "F_squared_armijo_bound": r_bound, "F_squared_armijo_passed": actual_r <= r_bound,
                "finite": math.isfinite(actual_j) and math.isfinite(actual_r)
                    and bool(torch.isfinite(residual).all()), "G": residual.tolist()})
            return result

        accepted, trials, search = prepared._search_candidates(model, control, evaluate,
            chart_candidate=lambda point, vector, alpha: prepared.local._chart_candidate(
                point, weights, vector, pivot, alpha), radius=RADIUS, limit=MAX_CANDIDATES_PER_ARM)
        arm["trials"], arm["search"] = trials, search
        if accepted is not None:
            arm["accepted"] = accepted
            candidate = torch.as_tensor(accepted["control"], dtype=control.dtype)
            event_values: dict[str, Any] = {}
            for side in (-1, 1):
                check()
                vals = event_diag._plain_values(problem, candidate, parameters, side)
                check()
                if not bool(torch.isfinite(vals).all() & (vals[2] > 0)):
                    raise ValueError("accepted candidate event values/q scale are nonfinite")
                zeta = float(vals[0] - vals[1])
                event_values[str(side)] = {"values": vals.tolist(), "zeta": zeta,
                    "normalized_abs_zeta": abs(zeta) / float(vals[2])}
            arm["accepted_event_values"] = event_values
        tangent._write(output, record)

    record["candidate_slots_used"] = sum(int(record["arms"][name].get("search", {}).get("evaluated_count", 0))
        for name in ARM_ORDER)
    if record["candidate_slots_used"] > POLICY["candidate_count"]:
        raise ValueError("resume search exceeded the 48-candidate budget")
    chosen = previous._select_arm_candidate(record["arms"], control.dtype)
    if chosen is None:
        after = shared._input_identity(problem, original, control, parameters, truth)
        source_after = shared._source_hashes(plan, plan_path, plan_sha)
        runtime_after = shared._runtime()
        record.update(phase="finished", execution_status="completed",
            numerical_status="event_direction_resume_complete_no_candidate", source_after=source_after,
            input_after=after, runtime_after=runtime_after, source_unchanged=source_after == source_before,
            fixed_input_unchanged=after == input_before, runtime_unchanged=runtime_after == runtime_before,
            optimizer_steps_applied=0, candidate_committed=False, readiness_complete=False,
            deadline_passed=time.monotonic() < deadline)
        tangent._write(output, record)
        return record

    arm_name, accepted = chosen["arm"], chosen["proposal"]
    arm = record["arms"][arm_name]
    observation = {"native_j": accepted["objective"], "face_ok": accepted["face_audit_passed"],
        "pair": {"passed": accepted["branch_pair_passed"]},
        "side_objectives_match_native": accepted["side_objectives_match_native"],
        "side_gradients_finite": accepted["side_gradients_finite"],
        "side": {s: (None, torch.as_tensor(accepted["side_gradients"][str(s)], dtype=control.dtype))
            for s in (-1, 1)}, "traces": {s: accepted["branch_trace"][str(s)] for s in (-1, 1)}}
    proposal = prepared.gni._candidate_proposal(torch.as_tensor(accepted["control"], dtype=control.dtype),
        observation, float(accepted["theta"]), float(accepted["F_squared"]), float(accepted["alpha"]),
        float(accepted["J_armijo_bound"]), float(accepted["F_squared_armijo_bound"]))
    proposal.update(selected_arm=arm_name, selected_direction_sha256=arm["direction_sha256"])
    check()
    if (shared._source_hashes(plan, plan_path, plan_sha) != source_before
            or shared._input_identity(problem, original, control, parameters, truth) != input_before
            or shared._runtime() != runtime_before):
        raise ValueError("source/input/runtime changed before 48b0 selected-candidate P2")
    repeat = prepared.gni._final_repeat(shared, geometry, tangent, plan, plan_path, plan_sha,
        problem, original, parameters, truth, weights, proposal, input_before, runtime_before,
        source_before, deadline)
    check()
    if not tangent.fresh_final_closure(proposal, repeat):
        raise ValueError("selected resume candidate failed P2; 48b0 remains current")
    record["selection"] = {"arm": arm_name, "comparison_complete": chosen["comparison_complete"],
        "selection_basis": chosen["selection_basis"], "supported_arms": chosen["supported_arms"],
        "alpha": float(accepted["alpha"]), "actual_J": float(accepted["objective"]),
        "actual_R": float(accepted["F_squared"]),
        "per_arm_first_pass_counts": previous._per_arm_first_pass_counts(record["arms"])}
    record = previous._persist_selected_commit(output, record, proposal, repeat)
    new_control = torch.as_tensor(repeat["control"], dtype=control.dtype)
    source_after = shared._source_hashes(plan, plan_path, plan_sha)
    input_after = shared._input_identity(problem, original, new_control, parameters, truth)
    runtime_after = shared._runtime()
    source_ok = source_after == source_before
    fixed_ok = geometry.model._fixed_input(input_after, input_before, tangent._tensor_sha(new_control))
    runtime_ok = runtime_after == runtime_before
    deadline_ok = time.monotonic() < deadline
    record.update({"phase": "finished", "execution_status": "completed",
        "numerical_status": ("event_direction_resume_committed_one_candidate_no_readiness"
            if all((source_ok, fixed_ok, runtime_ok, deadline_ok)) else
            "event_direction_resume_committed_postcommit_identity_incomplete"),
        "source_after": source_after, "input_after": input_after, "runtime_after": runtime_after,
        "source_unchanged": source_ok, "fixed_input_unchanged": fixed_ok,
        "runtime_unchanged": runtime_ok, "deadline_passed": deadline_ok,
        "readiness_complete": False, "postcommit_readiness_performed": False,
        "additional_candidates_after_commit": 0})
    tangent._write(output, record)
    return record


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    _load_plan(plan_path, plan_sha)
    plan_path = plan_path.resolve()
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, output.with_suffix(".run.json"))):
        raise ValueError("event-direction resume output paths must be fresh")
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
