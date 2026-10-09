"""Resume bounded current-point tangent corrections from the accepted PR267 endpoint."""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_RESUME_PLAN_20261009.json"
PREVIOUS_PLAN = EVIDENCE / "TANGENT_CONTINUATION_PLAN_20261009.json"
PREVIOUS_PLAN_SHA = "cc9c3a9baa6fdcbf6216d8a5d1613ebfaf63e708c279c5341aaba77cc35e3c69"
ARCHIVE_MANIFEST = EVIDENCE / "TANGENT_ARCHIVE_20261009.json"
STEP_ARCHIVE = EVIDENCE / "tangent_continuation_20261009_attempt1/step.json.gz"
STEP_RUN = EVIDENCE / "tangent_continuation_20261009_attempt1/step.run.json"
STEP_RESOURCE = EVIDENCE / "tangent_continuation_20261009_attempt1/step.resource.json"
ACTUAL_SOURCE_MANIFEST = EVIDENCE / "tangent_actual_source_20261009/manifest.json"
RESUME_SOURCE_MANIFEST = EVIDENCE / "tangent_resume_source_20261009/manifest.json"
BASE_CONTROL_SHA = "ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7"
BASE_THETA = 0.332026125873074
FACE = tangent.FACE
POLICY = tangent.POLICY
SELF = "examples/weather_scenarios/fv_point_3h_tangent_resume.py"
TEST = "tests/test_fv_point_3h_tangent_resume.py"
CHECKPOINT_TEST = "tests/test_fv_point_3h_tangent_checkpoint.py"
SHARED_NUMERICS = ("tangent_direction", "tangent_model", "candidate_alphas",
    "search_candidates", "fresh_final_closure", "bounded_continuation")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _function_ast(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    functions = {node.name: ast.dump(node, include_attributes=False)
        for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if any(name not in functions for name in names):
        raise ValueError("pre-edit tangent runner is missing a shared numeric function")
    return {name: functions[name] for name in names}


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("tangent-resume plan identity mismatch")
    if _sha(PREVIOUS_PLAN) != PREVIOUS_PLAN_SHA:
        raise ValueError("accepted tangent-continuation plan digest changed")
    previous = json.loads(PREVIOUS_PLAN.read_text())
    actual_manifest = json.loads(ACTUAL_SOURCE_MANIFEST.read_text())
    resume_manifest = json.loads(RESUME_SOURCE_MANIFEST.read_text())
    plan = json.loads(path.read_text())
    prior_sources = previous["source_files"]
    prior_archives = previous["archive_files"]
    actual_snapshots = actual_manifest["snapshots"]
    shared_snapshots = resume_manifest["snapshots"]
    extra_sources = set(plan.get("source_files", {})) - set(prior_sources)
    expected_archives = set(prior_archives) | {
        PREVIOUS_PLAN.relative_to(ROOT).as_posix(),
        STEP_ARCHIVE.relative_to(ROOT).as_posix(),
        STEP_RUN.relative_to(ROOT).as_posix(),
        STEP_RESOURCE.relative_to(ROOT).as_posix(),
        ARCHIVE_MANIFEST.relative_to(ROOT).as_posix(),
        ACTUAL_SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        RESUME_SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        *[row["archive_path"] for row in actual_snapshots.values()],
        *[row["archive_path"] for row in shared_snapshots.values()],
    }
    expected_sources = set(prior_sources) | {SELF, TEST, CHECKPOINT_TEST}
    if (plan.get("experiment_kind") != "current_tangent_resume"
            or plan.get("policy") != POLICY
            or plan.get("producing_plan") != PREVIOUS_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != PREVIOUS_PLAN_SHA
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA
            or plan.get("face") != FACE
            or plan.get("producer_source_snapshots") != actual_snapshots
            or plan.get("shared_source_snapshots") != shared_snapshots
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 129 or len(extra_sources) != 3
            or any(prior_sources.get(name) != snapshot["sha256"]
                   for name, snapshot in actual_snapshots.items())
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 105):
        raise ValueError("tangent-resume plan schema, policy, or map scope changed")
    sources, archives = plan["source_files"], plan["archive_files"]
    if any(archives.get(name) != value for name, value in prior_archives.items()):
        raise ValueError("inherited tangent-continuation archive pin changed")
    if any(sources.get(name) != value for name, value in prior_sources.items()
           if name not in actual_snapshots):
        raise ValueError("inherited tangent-continuation source pin changed")
    for manifest_snapshots in (actual_snapshots, shared_snapshots):
        for name, snapshot in manifest_snapshots.items():
            archive_path = snapshot["archive_path"]
            if archives.get(archive_path) != snapshot["sha256"] or _sha(ROOT / archive_path) != snapshot["sha256"]:
                raise ValueError(f"immutable tangent source copy changed: {name}")
    if any(sources.get(name) != _sha(ROOT / name) for name in actual_snapshots):
        raise ValueError("current tangent runner/test source map differs from live files")
    for name, snapshot in shared_snapshots.items():
        if name == tangent.SELF:
            archived = ROOT / snapshot["archive_path"]
            if _function_ast(archived, SHARED_NUMERICS) != _function_ast(ROOT / name, SHARED_NUMERICS):
                raise ValueError("resume changed the frozen tangent math/search/closure kernel")
    for name, digest_value in {**sources, **archives}.items():
        source = ROOT / name
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT.resolve()) or _sha(source) != digest_value:
            raise ValueError(f"tangent-resume source/archive pin mismatch: {name}")
    archive = json.loads(ARCHIVE_MANIFEST.read_text())
    for field, source in (("direction_archive", STEP_ARCHIVE),
                          ("direction_run", STEP_RUN),
                          ("direction_resource", STEP_RESOURCE)):
        if (plan.get(field) != source.relative_to(ROOT).as_posix()
                or plan.get(field + "_sha256") != _sha(source)):
            raise ValueError(f"tangent-resume {field} receipt changed")
    if plan.get("direction_child_sha256") != archive.get("raw_sha256"):
        raise ValueError("tangent-resume child digest differs from archive manifest")
    return plan


def _check_close(a: float, b: float, dtype: torch.dtype) -> bool:
    scale = max(abs(a), abs(b), torch.finfo(dtype).tiny)
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 128 * torch.finfo(dtype).eps * scale


def _validate_iteration_chain(iterations: list[dict[str, Any]],
                              hvp_history: list[dict[str, Any]],
                              start_control_sha256: str, start_theta: float
                              ) -> tuple[str, float, dict[str, Any], dict[str, Any]]:
    """Close every saved commit and its two current-point HVP labels in order."""
    if len(iterations) != tangent.MAX_ACCEPTED or len(hvp_history) != tangent.MAX_HVP:
        raise ValueError("prior tangent chain does not contain three commits and six HVPs")
    control_sha, theta = start_control_sha256, start_theta
    final_item: dict[str, Any] = {}
    final_trial: dict[str, Any] = {}
    for index, item in enumerate(iterations):
        if (item.get("index") != index or item.get("base_control_sha256") != control_sha
                or item.get("theta") != theta or item.get("accepted") is not True):
            raise ValueError("prior tangent iteration does not start at the preceding commit")
        trials = item.get("trials")
        if not isinstance(trials, list):
            raise ValueError("prior tangent iteration has no candidate-trial list")
        accepted_trials = [trial for trial in trials
            if trial.get("accepted") is True]
        if len(accepted_trials) != 1:
            raise ValueError("prior tangent iteration lacks one unique accepted trial")
        trial = accepted_trials[0]
        try:
            control = torch.as_tensor(trial["control"], dtype=torch.float64)
            committed = torch.as_tensor(item["committed_control"], dtype=torch.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("prior tangent commit control is malformed") from error
        if (control.shape != (26,) or committed.shape != (26,)
                or not bool(torch.isfinite(control).all() & torch.isfinite(committed).all())
                or not torch.equal(control, committed)
                or item.get("committed_theta") != trial.get("theta")
                or not isinstance(trial.get("theta"), (int, float))
                or isinstance(trial.get("theta"), bool)
                or not math.isfinite(float(trial["theta"]))
                or not 0 <= float(trial["theta"]) <= 1):
            raise ValueError("prior accepted trial and committed point/theta differ")
        point_hvps = [row for row in hvp_history
            if row.get("base_control_sha256") == control_sha]
        if (len(point_hvps) != 2 or {row.get("side") for row in point_hvps} != {-1, 1}
                or any(row.get("theta") != theta
                    or row.get("phase") != "current_tangent"
                    or row.get("operator") != "selected_face_extension"
                    or row.get("scope") != "one current chart tangent"
                    or row.get("status") != "completed" for row in point_hvps)):
            raise ValueError("prior tangent point does not have its two fresh labeled side HVPs")
        control_sha, theta = tangent._tensor_sha(control), float(trial["theta"])
        final_item, final_trial = item, trial
    return control_sha, theta, final_item, final_trial


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    if _sha(PREVIOUS_PLAN) != PREVIOUS_PLAN_SHA:
        raise ValueError("accepted tangent-continuation plan digest changed")
    previous = json.loads(PREVIOUS_PLAN.read_text())
    manifest = json.loads(ARCHIVE_MANIFEST.read_text())
    parent, resource = json.loads(STEP_RUN.read_text()), json.loads(STEP_RESOURCE.read_text())
    raw_bytes = gzip.decompress(STEP_ARCHIVE.read_bytes())
    child_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    entry = manifest.get("raw_sha256")
    accepted_command = parent.get("resource", {}).get("command", [])
    previous_receipt = {**previous["source_files"], **previous["archive_files"],
        PREVIOUS_PLAN.relative_to(ROOT).as_posix(): PREVIOUS_PLAN_SHA}
    source_before = raw.get("source_before", {})
    if (manifest.get("raw_path") != "graphify-out/fv-root-cause-20260919/tangent_continuation_20261009_attempt1/step.json"
            or manifest.get("gzip_path") != STEP_ARCHIVE.relative_to(ROOT).as_posix()
            or manifest.get("gzip_sha256") != _sha(STEP_ARCHIVE)
            or manifest.get("raw_bytes") != len(raw_bytes)
            or manifest.get("run_path") != STEP_RUN.relative_to(ROOT).as_posix()
            or manifest.get("run_sha256") != _sha(STEP_RUN)
            or manifest.get("resource_path") != STEP_RESOURCE.relative_to(ROOT).as_posix()
            or manifest.get("resource_sha256") != _sha(STEP_RESOURCE)
            or manifest.get("guarded_launch_count") != 1
            or entry != child_sha
            or parent.get("child_sha256") != child_sha
            or parent.get("execution_status") != "completed"
            or parent.get("child_read_error") is not None
            or parent.get("resource") != resource
            or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != POLICY["outer_seconds"]
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > POLICY["outer_seconds"]
            or resource.get("sampled_peak_rss_bytes", POLICY["rss_bytes"]) >= POLICY["rss_bytes"]
            or not accepted_command or not any(str(arg).endswith("fv_point_3h_tangent_continuation.py")
                for arg in accepted_command)
            or raw.get("plan_sha256") != PREVIOUS_PLAN_SHA
            or raw.get("plan_hashes", {}).get("source_files") != previous["source_files"]
            or raw.get("plan_hashes", {}).get("archive_files") != previous["archive_files"]
            or source_before != previous_receipt or raw.get("source_after") != previous_receipt
            or raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "tangent_continuation_cap_reached"
            or raw.get("candidate_type") != "tangent_gradient_correction"
            or raw.get("accepted_iterations") != 3 or raw.get("optimizer_steps_applied") != 3
            or raw.get("candidate_committed") is not True
            or raw.get("active_candidate_committed") is not False
            or raw.get("hvp_calls_started") != 6 or raw.get("hvp_calls_completed") != 6
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True
            or raw.get("deadline_passed") is not True
            or raw.get("input_before") is None or raw.get("input_after") is None
            or not geometry.model._fixed_input(raw["input_after"], raw["input_before"], BASE_CONTROL_SHA)
            or raw.get("runtime") != raw.get("runtime_after")):
        raise ValueError("archived tangent endpoint or run/resource closure is invalid")
    iterations = raw.get("iterations")
    if not isinstance(iterations, list):
        raise ValueError("prior tangent endpoint does not contain iteration receipts")
    hvp_history = raw.get("hvp_history", [])
    if (len(hvp_history) != 6 or raw.get("hvp_calls_started") != 6
            or raw.get("hvp_calls_completed") != 6
            or any(row.get("status") != "completed" for row in hvp_history)):
        raise ValueError("prior tangent endpoint HVP history is incomplete")
    chain_sha, chain_theta, last, accepted = _validate_iteration_chain(
        iterations, hvp_history, previous["base_control_sha256"], previous["initial_theta"])
    closure = raw.get("last_confirmed_closure")
    if closure != last.get("final_repeat"):
        raise ValueError("top-level last-confirmed closure differs from the last iteration receipt")
    control = torch.as_tensor(raw.get("current_control"), dtype=torch.float64)
    if (not isinstance(closure, dict)
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or chain_sha != BASE_CONTROL_SHA or chain_theta != BASE_THETA
            or _tensor_sha(control) != BASE_CONTROL_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_CONTROL_SHA
            or raw.get("last_confirmed_control") != raw.get("current_control")
            or raw.get("last_confirmed_theta") != BASE_THETA
            or raw.get("current_theta") != BASE_THETA
            or accepted.get("control") != raw.get("current_control")
            or accepted.get("theta") != BASE_THETA
            or closure.get("control") != accepted.get("control")
            or closure.get("theta") != accepted.get("theta")
            or not accepted.get("accepted")
            or not accepted.get("J_armijo_passed")
            or not accepted.get("F_squared_armijo_passed")
            or not accepted.get("face_audit_passed")
            or not accepted.get("branch_pair_passed")
            or not accepted.get("side_objectives_match_native")
            or not accepted.get("side_gradients_finite")):
        raise ValueError("last accepted tangent trial differs from the committed endpoint")
    closure_flags = ("native_objective_matches_proposal", "side_objectives_match_native",
        "merit_matches_proposal", "face_audit_passed", "branch_pair_passed",
        "trace_matches_proposal", "source_unchanged", "fixed_input_unchanged",
        "runtime_unchanged", "deadline_passed")
    if not all(closure.get(key) is True for key in closure_flags):
        raise ValueError("last tangent endpoint final-repeat closure is incomplete")
    try:
        accepted_gradients = {side: torch.as_tensor(accepted["side_gradients"][str(side)], dtype=control.dtype)
            for side in (-1, 1)}
        repeated_gradients = {side: torch.as_tensor(closure["side_gradients"][str(side)], dtype=control.dtype)
            for side in (-1, 1)}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("last tangent endpoint lacks both side gradients") from error
    if not shared._gradient_pair_match(repeated_gradients, accepted_gradients):
        raise ValueError("last tangent endpoint side gradients changed in final repeat")
    accepted_trace = accepted.get("branch_trace", {})
    repeated_trace = closure.get("branch_trace", {})
    if any(accepted_trace.get(str(side), {}).get("signature_sha256")
           != repeated_trace.get(str(side), {}).get("signature_sha256") for side in (-1, 1)):
        raise ValueError("last tangent endpoint branch traces changed in final repeat")
    f2 = float(accepted.get("F_squared", float("nan")))
    repeat_f2 = float(closure.get("F_squared", float("nan")))
    objective = float(accepted.get("objective", float("nan")))
    if (not _check_close(f2, repeat_f2, control.dtype)
            or not _check_close(objective, float(closure.get("objective", float("nan"))), control.dtype)
            or not _check_close(objective, float(raw.get("native_objective", float("nan"))), control.dtype)):
        raise ValueError("last tangent endpoint J or F-squared differs in final repeat")
    residual = torch.as_tensor(last.get("residual"), dtype=control.dtype)
    derivative = torch.as_tensor(last.get("residual_direction"), dtype=control.dtype)
    if (residual.shape != (27,) or derivative.shape != (27,)
            or not bool(torch.isfinite(residual).all() & torch.isfinite(derivative).all())
            or not _check_close(float(torch.dot(residual, derivative)),
                float(last.get("merit_product", float("nan"))), control.dtype)
            or not math.isfinite(float(accepted.get("alpha", float("nan"))))
            or float(accepted["alpha"]) <= 0):
        raise ValueError("last tangent model residual products are invalid")
    return {"raw": raw, "control": control, "theta": BASE_THETA,
        "objective": objective, "accepted": accepted, "child_sha256": child_sha,
        "input_identity": raw["input_after"], "runtime": raw["runtime_after"],
        "resume_efficiency": _resume_efficiency(last, accepted),
        "resume_efficiency_source_control_sha256": last["base_control_sha256"],
        "resume_efficiency_source_theta": last["theta"]}


def _resume_efficiency(iteration: dict[str, Any], accepted: dict[str, Any]) -> dict[str, float | None]:
    """Summarize the saved F/DF prediction without adding an optimization gate."""
    residual = torch.as_tensor(iteration.get("residual"), dtype=torch.float64)
    derivative = torch.as_tensor(iteration.get("residual_direction"), dtype=torch.float64)
    alpha = float(accepted.get("alpha", float("nan")))
    norm2 = float(torch.dot(residual, residual)) if residual.shape == derivative.shape else float("nan")
    derivative_norm2 = float(torch.dot(derivative, derivative)) if residual.shape == derivative.shape else float("nan")
    product = float(torch.dot(residual, derivative)) if residual.shape == derivative.shape else float("nan")
    actual_f2 = float(accepted.get("F_squared", float("nan")))
    predicted_fraction = ((-2.0 * alpha * product - alpha * alpha * derivative_norm2) / norm2
        if (math.isfinite(norm2) and norm2 > 0 and math.isfinite(alpha)
            and math.isfinite(product) and math.isfinite(derivative_norm2)) else float("nan"))
    actual_fraction = ((norm2 - actual_f2) / norm2
        if math.isfinite(norm2) and norm2 > 0 and math.isfinite(actual_f2) else float("nan"))
    cos2 = (product * product / (norm2 * derivative_norm2)
        if (math.isfinite(norm2) and math.isfinite(derivative_norm2)
            and norm2 > 0 and derivative_norm2 > 0 and math.isfinite(product)) else float("nan"))
    tangent_gradient = torch.as_tensor(iteration.get("tangent_gradient"), dtype=torch.float64)
    mixed_gradient = torch.as_tensor(iteration.get("mixed_gradient"), dtype=torch.float64)
    normal_component_norm = float("nan")
    if (tangent_gradient.shape == mixed_gradient.shape == (26,)
            and bool(torch.isfinite(tangent_gradient).all() & torch.isfinite(mixed_gradient).all())):
        normal_component_norm = float(torch.linalg.vector_norm(mixed_gradient - tangent_gradient))
    predicted_f2 = norm2 * (1.0 - predicted_fraction) if math.isfinite(norm2) and math.isfinite(predicted_fraction) else float("nan")
    return {"alpha": alpha,
        "base_F_squared": norm2,
        "predicted_F_squared": predicted_f2,
        "actual_F_squared": actual_f2,
        "residual_norm": math.sqrt(norm2) if math.isfinite(norm2) and norm2 >= 0 else float("nan"),
        "tangent_gradient_norm": float(torch.linalg.vector_norm(tangent_gradient))
            if tangent_gradient.shape == (26,) and bool(torch.isfinite(tangent_gradient).all()) else float("nan"),
        "mixed_gradient_norm": float(torch.linalg.vector_norm(mixed_gradient))
            if mixed_gradient.shape == (26,) and bool(torch.isfinite(mixed_gradient).all()) else float("nan"),
        "ambient_normal_component_norm": normal_component_norm,
        "normalized_predicted_reduction": predicted_fraction,
        "normalized_actual_reduction": actual_fraction, "cos_squared": cos2}


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child_impl(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA)


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA, initial_theta=BASE_THETA)


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    return tangent.run(plan_path, plan_sha, output, resource, log,
        plan_loader=_load_plan, child_script=Path(__file__),
        base_control_sha256=BASE_CONTROL_SHA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path,
        default=EVIDENCE / "tangent_resume_20261009_attempt1/step.json")
    parser.add_argument("--resource", type=Path,
        default=EVIDENCE / "tangent_resume_20261009_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path,
        default=EVIDENCE / "tangent_resume_20261009_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
