"""Read-only local-window diagnostic around the last confirmed GN endpoint.

The samples measure the finite-step residual model along a stored chart tangent.
They do not select or commit a step and do not certify a root or minimum.
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
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_memory_resume as memory
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "GN_LOCAL_WINDOW_PLAN_20261010.json"
MEMORY_ATTEMPT = EVIDENCE / "tangent_coupled_gn_memory_resume_20261010_attempt1"
SELF = "examples/weather_scenarios/fv_point_3h_gn_local_window_probe.py"
TEST = "tests/test_fv_point_3h_gn_local_window_probe.py"
ATTEMPT = EVIDENCE / "gn_local_window_20261010_attempt2"
PREFLIGHT_MANIFEST = EVIDENCE / "GN_LOCAL_WINDOW_PREFLIGHT_20261010.json"
PREFLIGHT_MANIFEST_SHA = "2fba6416002eeb7c4b2aee78c8cc38cacf2983fb2585423d99b7dd759597b8c9"
BASE_CONTROL_SHA = "2ec34eeeef531b94e21b5b4372ced2e07257a5407541072d300f5a3f513e1fcf"
BASE_THETA = 0.4830786491678902
BASE_J = 0.06119249248090723
BASE_F2 = 0.00447848922842147
OLD_ALPHA0 = 0.7306593405044787
OLD_MIN_ALPHA = OLD_ALPHA0 / 2**15
SAMPLE_ALPHAS = tuple(OLD_MIN_ALPHA / 2**k for k in range(1, 9))
STORED_DIRECTION_L2 = 0.04644782248562929
STORED_RESIDUAL_SLOPE = -0.005328963250381516
FACE_SCALE = 0.84
FACE: dict[str, Any] = {"axis": "y", "row": 4, "column": 3}
ANALYSIS_STAGES = 360
POLICY = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "hvp_calls": 2, "candidate_samples": 8,
    "jacobian_row_vjp_calls": 0, "optimizer_steps": 0, "dense_solves": 0,
    "pcg_solves": 0, "acceptance_or_step_selection": False,
    "root_claim": False, "minimum_claim": False, "response_claim": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_identity() -> dict[str, Any]:
    """Close the completed memory receipt and return only the active endpoint."""
    memory_plan = memory.PLAN
    memory_sha = "7c93aa28219c73e7a437feaac7341887da8fc6fc0b9f0b8e4633ad4424224c67"
    manifest_path = EVIDENCE / "STREAMGN_MEMORY_ARCHIVE_20261010.json"
    archive_path = MEMORY_ATTEMPT / "step.json.gz"
    run_path = MEMORY_ATTEMPT / "step.run.json"
    resource_path = MEMORY_ATTEMPT / "step.resource.json"
    expected = {
        memory_plan.relative_to(ROOT).as_posix(): memory_sha,
        manifest_path.relative_to(ROOT).as_posix(): "250b33b6bdf01bcc6d31a632d4f3008682408401238d587100ca0571eb9b6bb9",
        archive_path.relative_to(ROOT).as_posix(): "f573accecc4eaf1b02ce50b4fbedcf15d2333e0907397bd51c9bd7fb777136c1",
        run_path.relative_to(ROOT).as_posix(): "524badff95c820ccd6194f244afda0da0194e48abc086ca709696eabf3ecfb52",
        resource_path.relative_to(ROOT).as_posix(): "a971baf7e5e9ec245211fb5c362172f685f84dabf34fa12e69be1b7ca64b3d5a",
    }
    for name, digest in expected.items():
        file = ROOT / name
        if file.is_symlink() or _sha(file) != digest:
            raise ValueError(f"completed GN memory receipt changed: {name}")
    manifest = json.loads(manifest_path.read_text())
    parent = json.loads(run_path.read_text())
    resource = json.loads(resource_path.read_text())
    if (manifest.get("raw_path") != "graphify-out/fv-root-cause-20260919/tangent_coupled_gn_memory_resume_20261010_attempt1/step.json"
            or manifest.get("gzip_path") != archive_path.relative_to(ROOT).as_posix()
            or manifest.get("run_path") != run_path.relative_to(ROOT).as_posix()
            or manifest.get("resource_path") != resource_path.relative_to(ROOT).as_posix()
            or manifest.get("guarded_launch_count") != 1
            or manifest.get("raw_bytes") <= 0 or manifest.get("gzip_bytes") != len(archive_path.read_bytes())
            or manifest.get("raw_sha256") != "72224458e690cd6675f14479f0787bf7bf4aae878ef0fa7362ed3b0113553552"
            or manifest.get("gzip_sha256") != expected[archive_path.relative_to(ROOT).as_posix()]
            or manifest.get("run_sha256") != expected[run_path.relative_to(ROOT).as_posix()]
            or manifest.get("resource_sha256") != expected[resource_path.relative_to(ROOT).as_posix()]
            or manifest.get("lossless_roundtrip") is not True
            or parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != "tangent_coupled_gn_partial_resume_stopped_after_commit"
            or parent.get("child_sha256") != manifest.get("raw_sha256")
            or parent.get("child_read_error") is not None
            or parent.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or resource.get("wall_limit_seconds") != POLICY["outer_seconds"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > POLICY["outer_seconds"]
            or not isinstance(resource.get("sampled_peak_rss_bytes"), int)
            or resource["sampled_peak_rss_bytes"] >= POLICY["rss_bytes"]):
        raise ValueError("GN memory execution receipt does not close")

    memory_plan_dict = memory._load_plan(memory_plan, memory_sha)
    compressed = archive_path.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    if hashlib.sha256(raw_bytes).hexdigest() != manifest.get("raw_sha256"):
        raise ValueError("completed memory raw child digest does not close")
    raw = json.loads(raw_bytes)
    if (raw["execution_status"] != "completed"
            or raw["numerical_status"] != "tangent_coupled_gn_partial_resume_stopped_after_commit"
            or raw["source_unchanged"] is not True
            or raw["runtime_unchanged"] is not True or raw["fixed_input_unchanged"] is not True
            or raw["accepted_iterations"] != 1 or raw["hvp_calls_completed"] != 4
            or raw["jacobian_rows_completed"] != 48
            or raw["last_confirmed_control_sha256"] != BASE_CONTROL_SHA
            or raw["last_confirmed_theta"] != BASE_THETA):
        raise ValueError("completed memory raw does not close at the expected accepted point")
    closure = raw["last_confirmed_closure"]
    control = torch.as_tensor(raw["last_confirmed_control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_CONTROL_SHA:
        raise ValueError("last confirmed endpoint control digest is invalid")
    _require_endpoint_closure(closure, control)
    iteration = raw["iterations"][1]
    if (iteration.get("base_control_sha256") != BASE_CONTROL_SHA or iteration.get("accepted") is not False
            or iteration.get("selected_direction_model") is not None
            or iteration.get("direction") is None or iteration.get("residual") is None
            or iteration.get("residual_direction") is None):
        raise ValueError("unaccepted local direction is not bound to the current endpoint")
    arms = [arm for arm in iteration.get("model_comparisons", [])
        if arm.get("name") == "robust_gn_coupled"]
    if len(arms) != 1 or arms[0].get("accepted") is not False:
        raise ValueError("unaccepted robust-GN arm is missing or ambiguous")
    arm = arms[0]
    direction = torch.as_tensor(arm["direction"], dtype=control.dtype)
    if (direction.shape != control.shape or not bool(torch.isfinite(direction).all())
            or abs(float(torch.linalg.vector_norm(direction)) - STORED_DIRECTION_L2) > 1e-14
            or tangent._tensor_sha(direction) != "0240d2e18fb8dfd4012bd7a0c8d3dd75f13cc12c81c5c941db9afafb3c3d6e94"):
        raise ValueError("stored unaccepted direction digest is invalid")
    memory_source_receipt = shared._source_hashes(memory_plan_dict, memory_plan, memory_sha)
    if (raw["source_before"] != memory_source_receipt or raw["source_after"] != memory_source_receipt
            or raw["runtime"] != raw["runtime_after"]):
        raise ValueError("completed memory source/runtime receipt does not match its plan")
    current_hvps = [item for item in raw["hvp_history"]
        if item.get("base_control_sha256") == BASE_CONTROL_SHA]
    if (len(current_hvps) != 2
            or {item.get("side") for item in current_hvps} != {-1, 1}
            or any(item.get("theta") != BASE_THETA or item.get("working_theta") != BASE_THETA
                or item.get("direction_model") != "robust_gn_coupled"
                or item.get("operator") != "selected_face_extension"
                or item.get("direction_sha256") != tangent._tensor_sha(direction)
                or item.get("status") != "completed" for item in current_hvps)):
        raise ValueError("current endpoint fresh-HVP history does not bind to the stored arm")
    if (arm.get("hminus_direction") != iteration.get("hminus_direction")
            or arm.get("hplus_direction") != iteration.get("hplus_direction")
            or arm.get("residual") != iteration.get("residual")
            or arm.get("residual_direction") != iteration.get("residual_direction")):
        raise ValueError("unaccepted arm and iteration summary differ")
    side_by_sign = {item["side"]: item for item in current_hvps}
    receipt: dict[str, Any] = {"control": control, "input_after": raw["input_after"],
        "runtime_after": raw["runtime_after"], "closure": closure,
        "source_before": raw["source_before"], "source_after": raw["source_after"],
        "theta": BASE_THETA,
        "objective": BASE_J, "accepted": closure, "direction": direction,
        "stored_residual": torch.as_tensor(arm["residual"], dtype=control.dtype),
        "stored_residual_direction": torch.as_tensor(arm["residual_direction"], dtype=control.dtype),
        "stored_hminus": torch.as_tensor(arm["hminus_direction"], dtype=control.dtype),
        "stored_hplus": torch.as_tensor(arm["hplus_direction"], dtype=control.dtype),
        "stored_hvp_history": side_by_sign,
        "child_sha256": manifest["raw_sha256"]}
    if (abs(float(torch.dot(receipt["stored_residual"], receipt["stored_residual_direction"]))
            - STORED_RESIDUAL_SLOPE) > 1e-14):
        raise ValueError("stored unaccepted-arm residual slope differs from its receipt")
    del iteration, raw, raw_bytes, compressed
    return receipt


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("local-window plan identity mismatch")
    if _sha(memory.PLAN) != "7c93aa28219c73e7a437feaac7341887da8fc6fc0b9f0b8e4633ad4424224c67":
        raise ValueError("inherited 2-GiB GN memory plan changed")
    plan = json.loads(path.read_text())
    prior = json.loads(memory.PLAN.read_text())
    expected_sources = set(prior["source_files"]) | {SELF, TEST}
    added_archives = {
        memory.PLAN.relative_to(ROOT).as_posix(),
        (EVIDENCE / "STREAMGN_MEMORY_ARCHIVE_20261010.json").relative_to(ROOT).as_posix(),
        (MEMORY_ATTEMPT / "step.json.gz").relative_to(ROOT).as_posix(),
        (MEMORY_ATTEMPT / "step.run.json").relative_to(ROOT).as_posix(),
        (MEMORY_ATTEMPT / "step.resource.json").relative_to(ROOT).as_posix(),
    }
    if _sha(PREFLIGHT_MANIFEST) != PREFLIGHT_MANIFEST_SHA:
        raise ValueError("failed local-window preflight manifest changed")
    preflight = json.loads(PREFLIGHT_MANIFEST.read_text())
    added_archives.add(PREFLIGHT_MANIFEST.relative_to(ROOT).as_posix())
    added_archives.update(item["archive_path"] for item in preflight["snapshots"].values())
    added_archives.update(item["path"] for item in preflight["receipts"].values())
    expected_archives = set(prior["archive_files"]) | added_archives
    if (plan.get("experiment_kind") != "gn_local_window_diagnostic"
            or plan.get("policy") != POLICY or plan.get("face") != FACE
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA or plan.get("base_objective") != BASE_J
            or plan.get("base_F_squared") != BASE_F2 or plan.get("direction_sha256")
                != "0240d2e18fb8dfd4012bd7a0c8d3dd75f13cc12c81c5c941db9afafb3c3d6e94"
            or plan.get("direction_l2") != STORED_DIRECTION_L2
            or plan.get("stored_residual_directional_slope") != STORED_RESIDUAL_SLOPE
            or plan.get("sample_alphas") != list(SAMPLE_ALPHAS)
            or plan.get("predecessor_plan") != memory.PLAN.relative_to(ROOT).as_posix()
            or plan.get("predecessor_plan_sha256") != "7c93aa28219c73e7a437feaac7341887da8fc6fc0b9f0b8e4633ad4424224c67"
            or set(plan.get("source_files", {})) != expected_sources or len(plan["source_files"]) != 146
            or plan.get("preflight_manifest") != PREFLIGHT_MANIFEST.relative_to(ROOT).as_posix()
            or plan.get("preflight_manifest_sha256") != PREFLIGHT_MANIFEST_SHA
            or set(plan.get("archive_files", {})) != expected_archives or len(plan["archive_files"]) != 185):
        raise ValueError("local-window plan scope, endpoint, or diagnostic-only policy changed")
    if (any(plan["source_files"].get(name) != value for name, value in prior["source_files"].items())
            or any(plan["archive_files"].get(name) != value for name, value in prior["archive_files"].items())):
        raise ValueError("local-window plan changed an inherited memory source or archive pin")
    for name, value in {**plan["source_files"], **plan["archive_files"]}.items():
        file = ROOT / name
        if file.is_symlink() or not file.resolve().is_relative_to(ROOT.resolve()) or _sha(file) != value:
            raise ValueError(f"local-window source/archive pin mismatch: {name}")
    if (plan["source_files"].get(SELF) != _sha(ROOT / SELF)
            or plan["source_files"].get(TEST) != _sha(ROOT / TEST)):
        raise ValueError("local-window source/test pin mismatch")
    return plan


def _scaled_error(actual: Tensor, expected: Tensor, scale: Tensor) -> dict[str, float | bool]:
    if (actual.shape != expected.shape or actual.shape != scale.shape
            or actual.dtype != expected.dtype or actual.dtype != scale.dtype
            or not all(bool(torch.isfinite(value).all()) for value in (actual, expected, scale))):
        raise ValueError("component-scaled comparison inputs are nonfinite or incompatible")
    eps = torch.finfo(actual.dtype).eps
    # Componentwise relative error plus a reduction-roundoff allowance based on
    # the vector's own norm keeps near-zero components finite and interpretable.
    cancellation = (32 * eps * torch.linalg.vector_norm(scale)).clamp_min(torch.finfo(actual.dtype).tiny)
    budget = 256 * eps * scale.abs() + cancellation
    error = (actual - expected).abs()
    ratio = (error / budget).clamp_max(torch.finfo(actual.dtype).max)
    return {"linf": float(error.max()), "scaled_linf": float(ratio.max()),
        "budget_linf": float(budget.max()), "passed": bool(torch.all(error <= budget))}


def _require_model_matches(hvp: dict[str, Any], model: dict[str, Any]) -> None:
    if not all(item.get("passed") is True for item in (*hvp.values(), *model.values())):
        raise ValueError("fresh HVP or residual model failed its same-point receipt comparison")


def _require_endpoint_closure(closure: dict[str, Any], control: Tensor) -> None:
    required = ("side_gradients_finite", "native_objective_matches_proposal",
        "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
        "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
        "fixed_input_unchanged", "runtime_unchanged", "deadline_passed")
    if (closure.get("control") != control.tolist() or closure.get("theta") != BASE_THETA
            or closure.get("objective") != BASE_J or closure.get("F_squared") != BASE_F2
            or not all(closure.get(key) is True for key in required)
            or any(str(side) not in closure.get("side_gradients", {}) for side in (-1, 1))
            or any(str(side) not in closure.get("branch_trace", {}) for side in (-1, 1))):
        raise ValueError("last confirmed endpoint lacks a complete accepted closure")


def _finite_curve_metrics(g0: Tensor, g: Tensor, r: Tensor, alpha: float,
                          direction: Tensor, actual_displacement: float, f2: float, f20: float,
                          model_slope: float, base_j: float, candidate_j: float,
                          j_slope: float) -> dict[str, Any]:
    """Summarize the finite-step residual expansion without an acceptance gate."""
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("curve sample alpha must be finite and positive")
    remainder = g - g0 - alpha * r
    result = {
        "alpha": alpha,
        "G": g.tolist(),
        "E": remainder.tolist(),
        "E_over_h": (remainder / alpha).tolist(),
        "E_over_h2": (remainder / (alpha * alpha)).tolist(),
        "predicted_displacement_l2": float(torch.linalg.vector_norm(alpha * direction)),
        "actual_displacement_l2": actual_displacement,
        "G_l2": float(torch.linalg.vector_norm(g)),
        "G_inf": float(g.abs().max()),
        "E_l2": float(torch.linalg.vector_norm(remainder)),
        "E_inf": float(remainder.abs().max()),
        "E_over_h_l2": float(torch.linalg.vector_norm(remainder / alpha)),
        "E_over_h2_l2": float(torch.linalg.vector_norm(remainder / (alpha * alpha))),
        "actual_F_squared": f2,
        "actual_F2_armijo_rhs": f20 + 2e-4 * alpha * model_slope,
        "actual_F2_armijo_passed": bool(f2 <= f20 + 2e-4 * alpha * model_slope),
        "actual_J": candidate_j,
        "actual_J_armijo_rhs": base_j + 1e-4 * alpha * j_slope,
        "actual_J_armijo_passed": bool(candidate_j <= base_j + 1e-4 * alpha * j_slope),
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, (int, float))):
        raise ValueError("local-window curve produced a non-finite diagnostic ratio")
    return result


def _trace_delta(base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    def first_difference(left: Any, right: Any, path: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                if key not in left or key not in right:
                    return path + (key,)
                found = first_difference(left[key], right[key], path + (key,))
                if found is not None:
                    return found
            return None
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return path + (min(len(left), len(right)),)
            for index, (a, b) in enumerate(zip(left, right)):
                found = first_difference(a, b, path + (index,))
                if found is not None:
                    return found
            return None
        return None if left == right else path

    selector_path = first_difference(base.get("choices"), current.get("choices"))
    sign_path = first_difference(base.get("face_signs"), current.get("face_signs"))

    def selector_location(path: tuple[Any, ...] | None) -> dict[str, Any] | None:
        if path is None:
            return None
        orientation = path[1] if len(path) > 1 and isinstance(path[1], int) else None
        return {"stage": path[0] if path else None,
            "axis": ("x", "y")[orientation] if orientation in (0, 1) else None,
            "key": path[2] if len(path) > 2 else None,
            "row": path[3] if len(path) > 3 and isinstance(path[3], int) else None,
            "column": path[4] if len(path) > 4 and isinstance(path[4], int) else None}

    def sign_location(path: tuple[Any, ...] | None) -> dict[str, Any] | None:
        if path is None:
            return None
        return {"stage": path[0] if path else None,
            "axis": path[1] if len(path) > 1 and path[1] in ("x", "y") else None,
            "row": path[2] if len(path) > 2 and isinstance(path[2], int) else None,
            "column": path[3] if len(path) > 3 and isinstance(path[3], int) else None}

    return {"signature_sha256": current["signature_sha256"],
        "choices_changed": current.get("choices") != base.get("choices"),
        "face_signs_changed": current.get("face_signs") != base.get("face_signs"),
        "first_changed_selector": selector_location(selector_path),
        "first_changed_face_sign": sign_location(sign_path),
        "stage_count": current.get("stage_count"),
        "minimum_margins": current.get("minimum_margins"),
        "nonfinite_or_tie": current.get("nonfinite_or_tie")}


def _chart_candidate(control: Tensor, weights: Tensor, direction: Tensor,
                     pivot: int, alpha: float) -> Tensor:
    retained = [index for index in range(control.numel()) if index != pivot]
    coordinates = control.clone()
    coordinates[retained] += alpha * direction[retained]
    return geometry._chart(coordinates, weights, pivot, 0.0)


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    start = time.monotonic()
    plan_path = plan_path.resolve()
    plan = _load_plan(plan_path, plan_sha)
    source_before = shared._source_hashes(plan, plan_path, plan_sha)
    base = _archive_identity()
    control, direction = base["control"], base["direction"]
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    input_before = shared._input_identity(problem, original, control, parameters, truth)
    runtime_before = shared._runtime()
    if (input_before != base["input_after"]
            or not geometry.model._fixed_input(input_before, base["input_after"], BASE_CONTROL_SHA)
            or runtime_before != base["runtime_after"]):
        raise ValueError("fresh fixed input/runtime differs from the accepted memory endpoint")
    weights = geometry._face_weights(problem, axis=FACE["axis"], row=FACE["row"], column=FACE["column"])
    observed = tangent._observe(shared, problem, control, parameters, weights)
    closure = base["closure"]
    side_gradients = {side: observed["side"][side][1] for side in (-1, 1)}
    receipt_gradients = {side: torch.as_tensor(closure["side_gradients"][str(side)], dtype=control.dtype)
        for side in (-1, 1)}
    if (not shared._gradient_pair_match(side_gradients, receipt_gradients)
            or any(observed["traces"][s]["signature_sha256"]
                != closure["branch_trace"][str(s)]["signature_sha256"] for s in (-1, 1))
            or not observed["pair"]["passed"] or not observed["face_ok"]
            or not observed["side_objectives_match_native"] or not observed["side_gradients_finite"]):
        raise ValueError("fresh endpoint gradients, traces, strict branch pair, or face check failed")
    fresh_theta = float(tangent.minimum_mixture_weight(
        side_gradients[-1], side_gradients[1]))
    theta_budget = 128 * torch.finfo(control.dtype).eps * max(abs(fresh_theta), abs(BASE_THETA))
    if abs(fresh_theta - BASE_THETA) > theta_budget:
        raise ValueError("fresh endpoint mixing minimum differs from its accepted closure")
    if (abs(float(observed["native_j"]) - BASE_J) > 128 * torch.finfo(control.dtype).eps * BASE_J
            or abs(float(observed["q"])) > observed["face_bound"]):
        raise ValueError("fresh endpoint objective or selected face differs from its receipt")

    pivot = geometry._pivot(control, weights)
    normal = geometry._face_normal(control, weights)
    _, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    chart_error = tangent._validate_direction_override(
        side_gradients[-1], normal, chart, pivot, direction)
    fresh_hvp: dict[int, Tensor] = {}
    grad_fn = torch.func.grad(problem.objective, argnums=0)
    for side in (-1, 1):
        def calculate(sign: int = side) -> Tensor:
            with shared.transport.selected_face_extension(FACE["axis"], FACE["row"], FACE["column"], sign):
                return torch.func.jvp(lambda point: grad_fn(point, parameters),
                    (control,), (direction,))[1]
        fresh_hvp[side] = calculate()
    stored_hvp_errors = {
        "minus": _scaled_error(fresh_hvp[-1], base["stored_hminus"], base["stored_hminus"]),
        "plus": _scaled_error(fresh_hvp[1], base["stored_hplus"], base["stored_hplus"]),
    }
    model = mixing.minimum_tangent_model(side_gradients[-1], side_gradients[1],
        fresh_hvp[-1], fresh_hvp[1], normal, chart, fresh_theta,
        float(observed["q"]), pivot, FACE_SCALE, direction_override=direction)
    if not all(model.gates.get(key) is True for key in (
            "finite", "both_side_gradients_descend", "merit_descends",
            "envelope_slope_matches_residual_dot")):
        raise ValueError("fresh mixing-minimum tangent model failed its strict current-point gates")
    g0 = torch.cat(((1 - fresh_theta) * side_gradients[-1] + fresh_theta * side_gradients[1],
                    (observed["q"] / FACE_SCALE).reshape(1)))
    model_errors = {
        "G": _scaled_error(model.residual, base["stored_residual"], g0),
        "r": _scaled_error(model.residual_direction, base["stored_residual_direction"],
            model.residual_direction),
    }
    _require_model_matches(stored_hvp_errors, model_errors)
    if not bool(torch.isfinite(model.residual).all() and torch.isfinite(model.residual_direction).all()):
        raise ValueError("fresh local tangent model contains non-finite values")

    samples: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    for k, alpha in enumerate(SAMPLE_ALPHAS, 1):
        candidate = _chart_candidate(control, weights, direction, pivot, alpha)
        candidate_observed = tangent._observe(shared, problem, candidate, parameters, weights)
        theta = float(tangent.minimum_mixture_weight(
            candidate_observed["side"][-1][1], candidate_observed["side"][1][1]))
        candidate_g = torch.cat(((1 - theta) * candidate_observed["side"][-1][1]
            + theta * candidate_observed["side"][1][1],
            (candidate_observed["q"] / FACE_SCALE).reshape(1)))
        candidate_f2 = float(torch.dot(candidate_g, candidate_g))
        j_slope = max(float(torch.dot(side_gradients[-1], direction)),
                      float(torch.dot(side_gradients[1], direction)))
        metrics = _finite_curve_metrics(g0, candidate_g, model.residual_direction, alpha,
            direction, float(torch.linalg.vector_norm(candidate - control)), candidate_f2,
            BASE_F2, model.merit_product, BASE_J,
            float(candidate_observed["native_j"]), j_slope)
        trace_pair = {str(s): _trace_delta(observed["traces"][s], candidate_observed["traces"][s])
            for s in (-1, 1)}
        traces[str(k)] = {str(s): candidate_observed["traces"][s] for s in (-1, 1)}
        metrics.update({"k": k, "theta_minimum": theta,
            "G": candidate_g.tolist(),
            "control": candidate.tolist(),
            "gminus": candidate_observed["side"][-1][1].tolist(),
            "gplus": candidate_observed["side"][1][1].tolist(),
            "native_J": float(candidate_observed["native_j"]),
            "face_q": float(candidate_observed["q"]),
            "production_face_value": float(candidate_observed["production_q"]),
            "face_roundoff_bound": candidate_observed["face_bound"],
            "face_audit_passed": bool(candidate_observed["face_ok"]),
            "strict_branch_pair_passed": bool(candidate_observed["pair"]["passed"]),
            "side_objectives_match_native": candidate_observed["side_objectives_match_native"],
            "side_gradients_finite": candidate_observed["side_gradients_finite"],
            "trace_delta": trace_pair,
            "control_sha256": tangent._tensor_sha(candidate),
            "displacement_inf": float((candidate - control).abs().max())})
        samples.append(metrics)
    if time.monotonic() - start > POLICY["internal_seconds"]:
        raise TimeoutError("local-window diagnostic exceeded its internal deadline")
    source_after = shared._source_hashes(plan, plan_path, plan_sha)
    runtime_after = shared._runtime()
    input_after = shared._input_identity(problem, original, control, parameters, truth)
    if source_after != source_before or runtime_after != runtime_before or input_after != input_before:
        raise ValueError("source, fixed input, or runtime changed during local-window diagnostic")
    result = {
        "phase": "finished", "execution_status": "completed",
        "numerical_status": "local_window_diagnostic_completed_no_acceptance_or_step_selection",
        "plan_sha256": plan_sha, "base_control_sha256": BASE_CONTROL_SHA,
        "base_control": control.tolist(),
        "base_theta": BASE_THETA, "base_objective": BASE_J, "base_F_squared": BASE_F2,
        "base_gminus": side_gradients[-1].tolist(), "base_gplus": side_gradients[1].tolist(),
        "base_current_branch_pair": observed["pair"], "initial_side_gradients_match": True,
        "initial_branch_traces_match": True, "fresh_initial_theta": fresh_theta,
        "base_full_branch_traces": {str(s): observed["traces"][s] for s in (-1, 1)},
        "base_production_face_value": float(observed["production_q"]),
        "base_face_audit_passed": bool(observed["face_ok"]),
        "direction_sha256": tangent._tensor_sha(direction),
        "direction_l2": float(torch.linalg.vector_norm(direction)),
        "direction_chart_error": chart_error,
        "direction_face_normal_dot": float(torch.dot(normal, direction)),
        "stored_residual_directional_slope": float(torch.dot(base["stored_residual"], base["stored_residual_direction"])),
        "fresh_hvp_component_scaled_comparison": stored_hvp_errors,
        "fresh_model_G_r_component_scaled_comparison": model_errors,
        "fresh_residual": model.residual.tolist(),
        "fresh_residual_direction": model.residual_direction.tolist(),
        "fresh_hminus": fresh_hvp[-1].tolist(), "fresh_hplus": fresh_hvp[1].tolist(),
        "envelope_slope": model.gates["envelope_slope"],
        "merit_product": model.merit_product,
        "hvp_calls_started": 2, "hvp_calls_completed": 2,
        "hvp_history": [{"side": side, "base_control_sha256": BASE_CONTROL_SHA,
            "direction_sha256": tangent._tensor_sha(direction),
            "stored_receipt": {key: value for key, value in base["stored_hvp_history"][side].items()
                if key in ("theta", "working_theta", "direction_model", "operator", "status")},
            "operator": "selected_face_extension", "scope": "one current chart tangent",
            "status": "completed"} for side in (-1, 1)],
        "jacobian_rows_started": 0, "jacobian_rows_completed": 0,
        "dense_solves_started": 0, "dense_solves_completed": 0,
        "candidate_samples_started": 8, "candidate_samples_completed": len(samples),
        "samples": samples, "full_branch_traces": traces,
        "source_before": source_before,
        "source_after": source_after, "source_unchanged": source_after == source_before,
        "input_before": input_before, "input_after": input_after,
        "fixed_input_unchanged": input_before == input_after,
        "runtime_before": runtime_before, "runtime_after": runtime_after,
        "runtime_unchanged": runtime_before == runtime_after,
        "elapsed_seconds": time.monotonic() - start,
        "optimizer_steps_applied": 0, "candidate_committed": False,
        "acceptance_claim": False, "minimum_claim": False, "root_claim": False,
    }
    tangent._write(output, result)
    return result


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = _load_plan(plan_path, plan_sha)
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, output.with_suffix(".run.json"))):
        raise ValueError("local-window output paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
        "--child", "--plan", str(plan_path), "--plan-sha256", plan_sha, "--output", str(output)]
    resource_result = run_guarded_diagnostic(command,
        wall_seconds=plan["policy"]["outer_seconds"], rss_bytes=plan["policy"]["rss_bytes"],
        report_path=resource, log_path=log)
    parent = {"execution_status": shared._execution_status(resource_result,
        wall_seconds=plan["policy"]["outer_seconds"], rss_bytes=plan["policy"]["rss_bytes"]),
        "resource": resource_result, "child_sha256": _sha(output) if output.exists() else None,
        "numerical_status": "not_reached"}
    if output.exists():
        child = json.loads(output.read_text())
        parent["numerical_status"] = child.get("numerical_status")
    shared._write(output.with_suffix(".run.json"), parent)
    return {"parent": parent}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ATTEMPT / "diagnostic.json")
    parser.add_argument("--resource", type=Path, default=ATTEMPT / "diagnostic.resource.json")
    parser.add_argument("--log", type=Path, default=ATTEMPT / "diagnostic.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child_impl(args.plan, args.plan_sha256, args.output)
    else:
        run(args.plan, args.plan_sha256, args.output, args.resource, args.log)


if __name__ == "__main__":
    main()
