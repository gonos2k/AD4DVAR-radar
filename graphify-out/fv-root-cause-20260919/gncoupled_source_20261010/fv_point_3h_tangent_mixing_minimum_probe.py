"""Bounded baseline tangent step with a freshly minimized side-gradient mixture."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_MIXING_MINIMUM_PLAN_20261010.json"
PRODUCING_PLAN = EVIDENCE / "TANGENT_PRECONDITION_COMPARISON_PLAN_20261009.json"
PRODUCING_PLAN_SHA = "be2886f2686cd04e71d3d244967da378bae9c22e39168c5f22bd85e7801169a4"
PRODUCER_ARCHIVE = EVIDENCE / "PRECOND_ARCHIVE_20261009.json"
BASE_ARCHIVE = EVIDENCE / "tangent_precondition_comparison_20261009_attempt1/step.json.gz"
BASE_RUN = EVIDENCE / "tangent_precondition_comparison_20261009_attempt1/step.run.json"
BASE_RESOURCE = EVIDENCE / "tangent_precondition_comparison_20261009_attempt1/step.resource.json"
SOURCE_MANIFEST = EVIDENCE / "mixmin_source_20261010/manifest.json"
PRODUCER_ARCHIVE_SHA = "e410a0bf8bb015a818d4a3e0c73189976a4944c20f7d32161ab686ea99ba7a63"
BASE_RAW_SHA = "c64480cfdc8b011b63b8bec4fd0e704187b28657d47e5e51b63f606406e0c4cf"
BASE_GZIP_SHA = "e7b28a8693e0d1fe62f963bf9e5e714434d8ae0686a23e4ed20407adcb078ddb"
BASE_RUN_SHA = "bfc6c8fd8c4d92ffa7e7ad435993c8148322f0e098ce0daafdf3f3ad60ccaf0a"
BASE_RESOURCE_SHA = "b7d391a2f955c95dcb283afb85b3200210adc5ad99b96774f28d12573930288b"
BASE_CONTROL_SHA = "0da57dbc5a4c101c2f24204835a746514b110f4548fe64de40455f4478dfbed1"
BASE_THETA = 0.3298981720696071
BASE_OBJECTIVE = 0.06124393564803218
BASE_CARRIED_F_SQUARED = 0.005132555144362135
FACE = tangent.FACE
SELF = "examples/weather_scenarios/fv_point_3h_tangent_mixing_minimum_probe.py"
TEST = "tests/test_fv_point_3h_tangent_mixing_minimum_probe.py"
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 240.0, "outer_seconds": 300.0,
    "rss_bytes": 1024**3, "hvp_calls": 2, "jacobian_row_vjp_calls": 0,
    "max_accepted_iterations": 1, "max_candidates_per_iteration": 16,
    "radius": 0.05, "face_scale": 0.84, "optimizer_steps": 1,
    "pcg_solves": 0, "dense_solves": 0, "root_claim": False,
    "minimum_claim": False, "response_claim": False, "score_claim": False,
}
METHOD = {
    "merit": "minimum_squared_mixed_side_gradient_plus_scaled_face_residual",
    "theta_rule": "unique_resolved_strict_interior_argmin_over_theta_of_squared_mixed_side_gradient",
    "direction": "existing_chart_tangent_at_theta_star",
    "model": "two_fresh_selected_face_hvp_envelope_residual_derivative",
    "candidate_theta": "recomputed_from_candidate_side_gradients_no_clipping",
    "linear_theta_prediction": "diagnostic_only",
}
ANALYSIS_STAGES = 360
J_C1 = F_C1 = 1e-4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("mixing-minimum plan identity mismatch")
    if _sha(PRODUCING_PLAN) != PRODUCING_PLAN_SHA or _sha(PRODUCER_ARCHIVE) != PRODUCER_ARCHIVE_SHA:
        raise ValueError("PR270 producer pins changed")
    prior = json.loads(PRODUCING_PLAN.read_text())
    snapshots = json.loads(SOURCE_MANIFEST.read_text())["snapshots"]
    plan = json.loads(path.read_text())
    prior_sources, prior_archives = prior["source_files"], prior["archive_files"]
    expected_sources = set(prior_sources) | {SELF, TEST}
    expected_archives = set(prior_archives) | {
        PRODUCING_PLAN.relative_to(ROOT).as_posix(), PRODUCER_ARCHIVE.relative_to(ROOT).as_posix(),
        BASE_ARCHIVE.relative_to(ROOT).as_posix(), BASE_RUN.relative_to(ROOT).as_posix(),
        BASE_RESOURCE.relative_to(ROOT).as_posix(), SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        *[snapshot["archive_path"] for snapshot in snapshots.values()],
    }
    if (plan.get("experiment_kind") != "current_tangent_mixing_minimum"
            or plan.get("policy") != POLICY or plan.get("method") != METHOD
            or plan.get("producing_plan") != PRODUCING_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != PRODUCING_PLAN_SHA
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA
            or plan.get("base_objective") != BASE_OBJECTIVE
            or plan.get("base_carried_F_squared") != BASE_CARRIED_F_SQUARED
            or plan.get("face") != FACE
            or plan.get("producer_source_snapshots") != snapshots
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 133
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 124):
        raise ValueError("mixing-minimum plan scope or policy changed")
    sources, archives = plan["source_files"], plan["archive_files"]
    if any(archives.get(name) != value for name, value in prior_archives.items()):
        raise ValueError("inherited PR270 archive pin changed")
    snapshot_names = set(snapshots)
    if any(sources.get(name) != value for name, value in prior_sources.items()
            if name not in snapshot_names):
        raise ValueError("inherited PR270 source pin changed")
    if any(prior_sources.get(name) != snapshot["sha256"] for name, snapshot in snapshots.items()):
        raise ValueError("pre-edit tangent source snapshot differs from the PR270 plan")
    if any(sources.get(name) != _sha(ROOT / name) for name in snapshots):
        raise ValueError("current tangent source pins differ from the frozen successor plan")
    for name, snapshot in snapshots.items():
        if (archives.get(snapshot["archive_path"]) != snapshot["sha256"]
                or _sha(ROOT / snapshot["archive_path"]) != snapshot["sha256"]):
            raise ValueError(f"pre-edit source snapshot changed: {name}")
    for name, value in {**sources, **archives}.items():
        target = ROOT / name
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()) or _sha(target) != value:
            raise ValueError(f"mixing-minimum source/archive pin mismatch: {name}")
    if (plan.get("base_archive") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or plan.get("base_archive_sha256") != BASE_GZIP_SHA
            or plan.get("base_run") != BASE_RUN.relative_to(ROOT).as_posix()
            or plan.get("base_run_sha256") != BASE_RUN_SHA
            or plan.get("base_resource") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or plan.get("base_resource_sha256") != BASE_RESOURCE_SHA
            or plan.get("base_archive_manifest") != PRODUCER_ARCHIVE.relative_to(ROOT).as_posix()
            or plan.get("base_archive_manifest_sha256") != PRODUCER_ARCHIVE_SHA
            or plan.get("base_child_sha256") != BASE_RAW_SHA):
        raise ValueError("mixing-minimum predecessor receipt pins changed")
    return plan


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    """Load PR270's selected endpoint from its lossless receipt; no FV replay."""
    if (_sha(PRODUCING_PLAN) != PRODUCING_PLAN_SHA
            or _sha(PRODUCER_ARCHIVE) != PRODUCER_ARCHIVE_SHA
            or _sha(BASE_RUN) != BASE_RUN_SHA or _sha(BASE_RESOURCE) != BASE_RESOURCE_SHA):
        raise ValueError("PR270 plan/run/resource receipt digests changed")
    prior = json.loads(PRODUCING_PLAN.read_text())
    manifest = json.loads(PRODUCER_ARCHIVE.read_text())
    compressed = BASE_ARCHIVE.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    child_sha = hashlib.sha256(raw_bytes).hexdigest()
    if (child_sha != BASE_RAW_SHA or _sha(BASE_ARCHIVE) != BASE_GZIP_SHA
            or manifest.get("raw_sha256") != BASE_RAW_SHA
            or manifest.get("gzip_sha256") != BASE_GZIP_SHA
            or manifest.get("run_sha256") != BASE_RUN_SHA
            or manifest.get("resource_sha256") != BASE_RESOURCE_SHA
            or manifest.get("guarded_launch_count") != 1):
        raise ValueError("PR270 lossless archive manifest does not close")
    raw = json.loads(raw_bytes)
    parent = json.loads(BASE_RUN.read_text())
    resource = json.loads(BASE_RESOURCE.read_text())
    source_receipt = {**prior["source_files"], **prior["archive_files"],
        PRODUCING_PLAN.relative_to(ROOT).as_posix(): PRODUCING_PLAN_SHA}
    if (manifest.get("raw_path") != "graphify-out/fv-root-cause-20260919/tangent_precondition_comparison_20261009_attempt1/step.json"
            or manifest.get("gzip_path") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or manifest.get("raw_bytes") != len(raw_bytes)
            or manifest.get("gzip_bytes") != len(compressed)
            or manifest.get("run_path") != BASE_RUN.relative_to(ROOT).as_posix()
            or manifest.get("resource_path") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != "tangent_precondition_comparison_accepted"
            or parent.get("child_read_error") is not None or parent.get("child_sha256") != child_sha
            or parent.get("resource") != resource
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != 660.0 or resource.get("rss_limit_bytes") != 1024**3
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > 660.0
            or resource.get("sampled_peak_rss_bytes", 1024**3) >= 1024**3
            or raw.get("plan_sha256") != PRODUCING_PLAN_SHA
            or raw.get("plan_hashes", {}).get("source_files") != prior["source_files"]
            or raw.get("plan_hashes", {}).get("archive_files") != prior["archive_files"]
            or raw.get("source_before") != source_receipt or raw.get("source_after") != source_receipt
            or raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "tangent_precondition_comparison_accepted"
            or raw.get("base_control_sha256") != "33cb86ca73a404e6ff9260acbf63ee97f248ac0c1f529427eb1af1ec85a43586"
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or raw.get("candidate_committed") is not True
            or raw.get("active_candidate_committed") is not False
            or raw.get("accepted_iterations") != 1 or raw.get("optimizer_steps_applied") != 1
            or raw.get("hvp_calls_started") != raw.get("hvp_calls_completed")
            or raw.get("hvp_calls_completed") != 4
            or raw.get("jacobian_rows_started") != 24 or raw.get("jacobian_rows_completed") != 24
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True or raw.get("deadline_passed") is not True
            or raw.get("last_confirmed_control_sha256") != BASE_CONTROL_SHA
            or raw.get("last_confirmed_theta") != BASE_THETA
            or raw.get("last_confirmed_iterations") != 1
            or raw.get("current_theta") != BASE_THETA
            or not geometry.model._fixed_input(raw.get("input_after", {}),
                raw.get("input_before", {}), BASE_CONTROL_SHA)):
        raise ValueError("PR270 endpoint/input/resource/source closure is invalid")
    iterations = raw.get("iterations", [])
    if len(iterations) != 1:
        raise ValueError("PR270 receipt does not contain one accepted comparison step")
    iteration = iterations[0]
    arms = iteration.get("model_comparisons", [])
    selected = [arm for arm in arms if arm.get("name") == "baseline_tangent"]
    if (iteration.get("accepted") is not True
            or iteration.get("selected_direction_model") != "baseline_tangent"
            or iteration.get("base_control_sha256") != raw.get("base_control_sha256")
            or iteration.get("theta") != prior.get("initial_theta")
            or iteration.get("committed_control") != raw.get("current_control")
            or iteration.get("committed_theta") != BASE_THETA
            or len(selected) != 1 or selected[0].get("accepted") is not True):
        raise ValueError("PR270 did not select the expected terminal baseline trial")
    accepted_trials = [trial for trial in selected[0].get("trials", [])
        if trial.get("accepted") is True]
    final_repeat = iteration.get("final_repeat")
    if (len(accepted_trials) != 1 or not isinstance(final_repeat, dict)
            or final_repeat != raw.get("last_confirmed_closure")
            or accepted_trials[0].get("control") != raw.get("current_control")
            or accepted_trials[0].get("theta") != BASE_THETA
            or accepted_trials[0].get("objective") != BASE_OBJECTIVE
            or accepted_trials[0].get("F_squared") != BASE_CARRIED_F_SQUARED
            or final_repeat.get("control") != raw.get("current_control")
            or final_repeat.get("theta") != BASE_THETA
            or final_repeat.get("objective") != BASE_OBJECTIVE
            or final_repeat.get("F_squared") != BASE_CARRIED_F_SQUARED
            or final_repeat.get("side_gradients") != accepted_trials[0].get("side_gradients")
            or final_repeat.get("branch_trace") != accepted_trials[0].get("branch_trace")
            or not all(final_repeat.get(key) is True for key in (
                "side_gradients_finite", "native_objective_matches_proposal",
                "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
                "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
                "fixed_input_unchanged", "runtime_unchanged", "deadline_passed"))):
        raise ValueError("PR270 selected trial/final closure differs from the pinned endpoint")
    control = torch.as_tensor(final_repeat["control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_CONTROL_SHA:
        raise ValueError("PR270 endpoint control digest is invalid")
    return {"raw": raw, "control": control, "theta": BASE_THETA,
        "objective": BASE_OBJECTIVE, "accepted": final_repeat, "child_sha256": child_sha}


def minimum_tangent_model(gminus: Tensor, gplus: Tensor, hminus: Tensor, hplus: Tensor,
        normal: Tensor, chart: Tensor, theta_star: float, q: float, pivot: int,
        face_scale: float) -> tangent.TangentModel:
    """Build the full residual derivative of the interior mixing-minimum envelope."""
    mixed, tangent_gradient, direction, support = tangent.tangent_direction(
        gminus, gplus, normal, chart, theta_star, pivot=pivot)
    jump = gplus - gminus
    jump_squared = torch.dot(jump, jump)
    if not bool(torch.isfinite(jump_squared) & (jump_squared > torch.finfo(gminus.dtype).tiny)):
        raise ValueError("minimum-mixture denominator is unresolved after HVP")
    hmix = (1.0 - theta_star) * hminus + theta_star * hplus
    delta_h = hplus - hminus
    theta_numerator = torch.dot(jump, hmix) + torch.dot(mixed, delta_h)
    theta_prime = -theta_numerator / jump_squared
    residual = torch.cat((mixed, mixed.new_tensor([q / face_scale])))
    residual_direction = torch.cat((hmix + jump * theta_prime,
        (torch.dot(normal, direction) / face_scale).reshape(1)))
    envelope_slope = torch.dot(mixed, hmix) + q * torch.dot(normal, direction) / face_scale**2
    merit_product = float(torch.dot(residual, residual_direction))
    dtype = gminus.dtype
    slope_scale = max(abs(float(envelope_slope)),
        float(torch.linalg.vector_norm(residual))
        * float(torch.linalg.vector_norm(residual_direction)), torch.finfo(dtype).tiny)
    slope_budget = 128.0 * torch.finfo(dtype).eps * slope_scale
    envelope_error = abs(merit_product - float(envelope_slope))
    side_products = (float(torch.dot(gminus, direction)), float(torch.dot(gplus, direction)))
    side_budget = tangent._roundoff(dtype,
        float(torch.linalg.vector_norm(gminus)) * float(torch.linalg.vector_norm(direction)),
        float(torch.linalg.vector_norm(gplus)) * float(torch.linalg.vector_norm(direction)))
    finite = all(bool(torch.isfinite(value).all()) for value in (
        mixed, tangent_gradient, direction, hmix, theta_prime, residual,
        residual_direction, envelope_slope))
    gates = {"finite": finite, **support,
        "candidate_mixing_minimum": True,
        "theta_stationarity": float(torch.dot(jump, mixed)),
        "theta_star_interior": True,
        "theta_prime_finite": bool(torch.isfinite(theta_prime)),
        "envelope_slope": float(envelope_slope),
        "envelope_slope_matches_residual_dot": envelope_error <= slope_budget,
        "envelope_slope_match_error": envelope_error,
        "envelope_slope_match_budget": slope_budget,
        "both_side_gradients_descend": max(side_products) < -side_budget,
        "side_descent_budget": side_budget,
        "merit_descends": merit_product < -slope_budget,
        "merit_descent_budget": slope_budget}
    if not finite or envelope_error > slope_budget:
        raise ValueError("minimum-mixture residual derivative failed its envelope identity")
    return tangent.TangentModel(theta_star, mixed, tangent_gradient, direction,
        hminus, hplus, float(theta_prime), float(theta_numerator), float(jump_squared),
        direction, residual, residual_direction, side_products, merit_product, gates)


def _working_model(context: dict[str, Any]):
    def build(control: Tensor, reference_theta: float, gminus: Tensor, gplus: Tensor,
              normal: Tensor, chart: Tensor, pivot: int) -> list[dict[str, Any]]:
        theta_star = tangent.minimum_mixture_weight(gminus, gplus)
        working_theta = float(theta_star)
        mixed, tangent_gradient, direction, support = tangent.tangent_direction(
            gminus, gplus, normal, chart, working_theta, pivot=pivot)
        q = float(context["model_state"]["observed"]["q"])
        context["model_state"]["working_theta"] = working_theta
        residual = torch.cat((mixed, mixed.new_tensor([q / float(context["plan"]["policy"]["face_scale"])])))
        context["record"]["base_mixing_minimum"] = {
            "reference_carried_theta": reference_theta,
            "working_theta_star": working_theta,
            "F_squared": float(torch.dot(residual, residual)),
            "Psi": 0.5 * float(torch.dot(residual, residual)),
            "optimizer_step": False,
        }

        def build_after_hvps(gm: Tensor, gp: Tensor, hm: Tensor, hp: Tensor,
                n: Tensor, z: Tensor, carried_theta: float, q_value: float,
                actual_pivot: int) -> tangent.TangentModel:
            return minimum_tangent_model(gm, gp, hm, hp, n, z,
                float(theta_star), q_value, actual_pivot,
                float(context["plan"]["policy"]["face_scale"]))

        return [{"name": "candidate_mixing_minimum", "direction": direction,
            "direction_override": direction, "working_theta": working_theta,
            "candidate_mixing_minimum": True, "model_builder": build_after_hvps,
            "diagnostics": {"theta_star": working_theta,
                "theta_stationarity": float(torch.dot(gplus - gminus, mixed)),
                "direction_norm": float(torch.linalg.vector_norm(direction)),
                "direction_normal_residual": float(torch.dot(normal, direction)),
                "support": support}}]
    return build


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child_impl(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA, direction_model_factory=_working_model)


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA, direction_model_factory=_working_model,
        initial_theta=BASE_THETA)


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
        default=EVIDENCE / "tangent_mixing_minimum_20261010_attempt1/step.json")
    parser.add_argument("--resource", type=Path,
        default=EVIDENCE / "tangent_mixing_minimum_20261010_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path,
        default=EVIDENCE / "tangent_mixing_minimum_20261010_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
