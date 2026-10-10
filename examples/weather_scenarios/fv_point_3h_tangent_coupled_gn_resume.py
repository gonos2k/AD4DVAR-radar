"""Resume the selected PR273 coupled-GN endpoint for three bounded steps."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_comparison as coupled
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_COUPLED_GN_RESUME_PLAN_20261010.json"
PRODUCING_PLAN = EVIDENCE / "TANGENT_COUPLED_GN_COMPARISON_PLAN_20261010.json"
PRODUCING_PLAN_SHA = "18be87cdbb6d084ff7248339bf81b749edca4e55dbd9d3a3d9d692d5c37f611d"
PRODUCER_ARCHIVE = EVIDENCE / "GNCOUPLED_ARCHIVE_20261010.json"
BASE_ARCHIVE = EVIDENCE / "tangent_coupled_gn_comparison_20261010_attempt1/step.json.gz"
BASE_RUN = EVIDENCE / "tangent_coupled_gn_comparison_20261010_attempt1/step.run.json"
BASE_RESOURCE = EVIDENCE / "tangent_coupled_gn_comparison_20261010_attempt1/step.resource.json"
SOURCE_MANIFEST = EVIDENCE / "gnresume_source_20261010/manifest.json"
PRODUCER_ARCHIVE_SHA = "ca01a3e68b277da672fb989f0713ba3778a8572b95f25d1b3bf94064f5945a3c"
BASE_RAW_SHA = "362e8e0aa7cf12202618c11873633201d23ddc2af835ac67a5d499612edb4599"
BASE_GZIP_SHA = "bdf479a14b0ba0e3342a8dc630e9bce2c89ac83fd4fea4625d17b2a01e507b84"
BASE_RUN_SHA = "627505212d26ca787e4979bcc25706c9ac68ff8fd4ed3fceb262a10611a2b653"
BASE_RESOURCE_SHA = "7585c4dbc8a2fb9bfce1af96d691fb00f92ccefb17de8f4b412e35b8a4027f4a"
BASE_CONTROL_SHA = "4035904e46c11c67a6f937039fe3c88b0516330b05f3ba9d57c4cfa619423ab7"
BASE_THETA = 0.4829044955080727
BASE_OBJECTIVE = 0.061202333947534084
BASE_CARRIED_F_SQUARED = 0.004490021941881948
FACE = mixing.FACE
SELF = "examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_resume.py"
TEST = "tests/test_fv_point_3h_tangent_coupled_gn_resume.py"
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 1024**3, "direction_models": 1,
    "candidate_grid_per_direction": 16, "max_candidates_per_iteration": 16,
    "radius": 0.05, "face_scale": 0.84,
    "jacobian_row_vjp_calls": 72, "hvp_calls": 6,
    "max_accepted_iterations": 3, "optimizer_steps": 3,
    "pcg_solves": 0, "dense_solves": 3, "root_claim": False,
    "minimum_claim": False, "response_claim": False, "score_claim": False,
}
METHOD = {**coupled.METHOD, "direction_models": ["robust_gn_coupled"]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("coupled-GN-resume plan identity mismatch")
    if _sha(PRODUCING_PLAN) != PRODUCING_PLAN_SHA or _sha(PRODUCER_ARCHIVE) != PRODUCER_ARCHIVE_SHA:
        raise ValueError("PR273 producer pins changed")
    prior = json.loads(PRODUCING_PLAN.read_text())
    snapshots = json.loads(SOURCE_MANIFEST.read_text())["snapshots"]
    plan = json.loads(path.read_text())
    expected_sources = set(prior["source_files"]) | {SELF, TEST}
    expected_archives = set(prior["archive_files"]) | {
        PRODUCING_PLAN.relative_to(ROOT).as_posix(), PRODUCER_ARCHIVE.relative_to(ROOT).as_posix(),
        BASE_ARCHIVE.relative_to(ROOT).as_posix(), BASE_RUN.relative_to(ROOT).as_posix(),
        BASE_RESOURCE.relative_to(ROOT).as_posix(), SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        *[snapshot["archive_path"] for snapshot in snapshots.values()],
    }
    if (plan.get("experiment_kind") != "current_tangent_coupled_gn_resume"
            or plan.get("policy") != POLICY or plan.get("method") != METHOD
            or plan.get("producing_plan") != PRODUCING_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != PRODUCING_PLAN_SHA
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA
            or plan.get("base_objective") != BASE_OBJECTIVE
            or plan.get("base_carried_F_squared") != BASE_CARRIED_F_SQUARED
            or plan.get("face") != FACE
            or plan.get("producer_source_snapshots") != snapshots
            or set(plan.get("source_files", {})) != expected_sources or len(plan["source_files"]) != 139
            or set(plan.get("archive_files", {})) != expected_archives or len(plan["archive_files"]) != 148):
        raise ValueError("coupled-GN-resume plan scope or policy changed")
    sources, archives = plan["source_files"], plan["archive_files"]
    if any(archives.get(name) != value for name, value in prior["archive_files"].items()):
        raise ValueError("inherited PR273 archive pin changed")
    snapshot_names = set(snapshots)
    if any(sources.get(name) != value for name, value in prior["source_files"].items()
            if name not in snapshot_names):
        raise ValueError("inherited PR273 source pin changed")
    if any(prior["source_files"].get(name) != snapshot["sha256"] for name, snapshot in snapshots.items()):
        raise ValueError("pre-edit coupled source snapshot differs from PR273 plan")
    if any(sources.get(name) != _sha(ROOT / name) for name in snapshots):
        raise ValueError("current coupled source pins differ from successor plan")
    for name, snapshot in snapshots.items():
        if (archives.get(snapshot["archive_path"]) != snapshot["sha256"]
                or _sha(ROOT / snapshot["archive_path"]) != snapshot["sha256"]):
            raise ValueError(f"pre-edit shared source snapshot changed: {name}")
    for name, value in {**sources, **archives}.items():
        target = ROOT / name
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()) or _sha(target) != value:
            raise ValueError(f"coupled-GN-resume source/archive pin mismatch: {name}")
    pinned = {
        "base_archive": (BASE_ARCHIVE, BASE_GZIP_SHA), "base_run": (BASE_RUN, BASE_RUN_SHA),
        "base_resource": (BASE_RESOURCE, BASE_RESOURCE_SHA),
        "base_archive_manifest": (PRODUCER_ARCHIVE, PRODUCER_ARCHIVE_SHA),
    }
    for key, (file, sha) in pinned.items():
        if plan.get(key) != file.relative_to(ROOT).as_posix() or plan.get(f"{key}_sha256") != sha:
            raise ValueError("PR273 predecessor receipt pins changed")
    if plan.get("base_child_sha256") != BASE_RAW_SHA:
        raise ValueError("PR273 raw child pin changed")
    return plan


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    """Admit PR273's selected robust-GN endpoint from lossless receipts only."""
    for path, digest in ((PRODUCING_PLAN, PRODUCING_PLAN_SHA), (PRODUCER_ARCHIVE, PRODUCER_ARCHIVE_SHA),
            (BASE_RUN, BASE_RUN_SHA), (BASE_RESOURCE, BASE_RESOURCE_SHA)):
        if _sha(path) != digest:
            raise ValueError("PR273 plan/run/resource receipt digests changed")
    prior = json.loads(PRODUCING_PLAN.read_text())
    manifest = json.loads(PRODUCER_ARCHIVE.read_text())
    compressed = BASE_ARCHIVE.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    child_sha = hashlib.sha256(raw_bytes).hexdigest()
    if (child_sha != BASE_RAW_SHA or _sha(BASE_ARCHIVE) != BASE_GZIP_SHA
            or manifest.get("raw_sha256") != BASE_RAW_SHA or manifest.get("gzip_sha256") != BASE_GZIP_SHA
            or manifest.get("run_sha256") != BASE_RUN_SHA or manifest.get("resource_sha256") != BASE_RESOURCE_SHA
            or manifest.get("guarded_launch_count") != 1 or manifest.get("lossless_roundtrip") is not True):
        raise ValueError("PR273 lossless archive manifest does not close")
    raw, parent, resource = json.loads(raw_bytes), json.loads(BASE_RUN.read_text()), json.loads(BASE_RESOURCE.read_text())
    source_receipt = {**prior["source_files"], **prior["archive_files"],
        PRODUCING_PLAN.relative_to(ROOT).as_posix(): PRODUCING_PLAN_SHA}
    if (manifest.get("raw_path") != "graphify-out/fv-root-cause-20260919/tangent_coupled_gn_comparison_20261010_attempt1/step.json"
            or manifest.get("gzip_path") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or manifest.get("raw_bytes") != len(raw_bytes) or manifest.get("gzip_bytes") != len(compressed)
            or manifest.get("run_path") != BASE_RUN.relative_to(ROOT).as_posix()
            or manifest.get("resource_path") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != "tangent_coupled_gn_comparison_accepted"
            or parent.get("child_read_error") is not None or parent.get("child_sha256") != child_sha
            or parent.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False or resource.get("wall_limit_seconds") != coupled.POLICY["outer_seconds"]
            or resource.get("rss_limit_bytes") != coupled.POLICY["rss_bytes"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > coupled.POLICY["outer_seconds"]
            or resource.get("sampled_peak_rss_bytes", coupled.POLICY["rss_bytes"]) >= coupled.POLICY["rss_bytes"]
            or raw.get("plan_sha256") != PRODUCING_PLAN_SHA
            or raw.get("plan_hashes", {}).get("source_files") != prior["source_files"]
            or raw.get("plan_hashes", {}).get("archive_files") != prior["archive_files"]
            or raw.get("source_before") != source_receipt or raw.get("source_after") != source_receipt
            or raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "tangent_coupled_gn_comparison_accepted"
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA or raw.get("current_theta") != BASE_THETA
            or raw.get("accepted_iterations") != 1 or raw.get("optimizer_steps_applied") != 1
            or raw.get("hvp_calls_started") != raw.get("hvp_calls_completed") or raw.get("hvp_calls_completed") != 4
            or raw.get("jacobian_rows_started") != raw.get("jacobian_rows_completed")
            or raw.get("jacobian_rows_completed") != 24
            or raw.get("dense_solves_started") != raw.get("dense_solves_completed")
            or raw.get("dense_solves_completed") != 1
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True or raw.get("deadline_passed") is not True
            or raw.get("last_confirmed_control_sha256") != BASE_CONTROL_SHA
            or raw.get("last_confirmed_theta") != BASE_THETA
            or not geometry.model._fixed_input(raw.get("input_after", {}), raw.get("input_before", {}), BASE_CONTROL_SHA)):
        raise ValueError("PR273 endpoint/input/resource/source closure is invalid")
    final = raw.get("last_confirmed_closure")
    iteration = raw.get("iterations", [{}])[-1]
    gn_arms = [arm for arm in iteration.get("model_comparisons", []) if arm.get("name") == "robust_gn_coupled"]
    accepted = [trial for arm in gn_arms for trial in arm.get("trials", []) if trial.get("accepted") is True]
    if (len(raw.get("iterations", [])) != 1 or iteration.get("accepted") is not True
            or len(gn_arms) != 1 or not gn_arms[0].get("accepted") or len(accepted) != 1
            or iteration.get("selected_direction_model") != "robust_gn_coupled"
            or not isinstance(final, dict) or final != iteration.get("final_repeat")
            or accepted[0].get("control") != final.get("control") or accepted[0].get("theta") != BASE_THETA
            or accepted[0].get("objective") != BASE_OBJECTIVE
            or accepted[0].get("F_squared") != BASE_CARRIED_F_SQUARED
            or not tangent.fresh_final_closure(accepted[0], final)
            or final.get("theta") != BASE_THETA or final.get("objective") != BASE_OBJECTIVE
            or final.get("F_squared") != BASE_CARRIED_F_SQUARED
            or final.get("mixing_minimum_valid") is not True
            or final.get("minimum_theta_matches_proposal") is not True
            or not all(final.get(key) is True for key in ("side_gradients_finite", "native_objective_matches_proposal",
                "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed", "branch_pair_passed",
                "trace_matches_proposal", "source_unchanged", "fixed_input_unchanged", "runtime_unchanged", "deadline_passed"))):
        raise ValueError("PR273 selected robust-GN final closure is invalid")
    control = torch.as_tensor(final["control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_CONTROL_SHA:
        raise ValueError("PR273 terminal control digest is invalid")
    # The full receipt is verified above and remains in the immutable archive.
    # Keep only the input/runtime anchor resident during the new FV trajectory.
    anchor = {key: raw[key] for key in ("input_after", "runtime_after")}
    return {"raw": anchor, "control": control, "theta": BASE_THETA,
        "objective": BASE_OBJECTIVE, "accepted": final, "child_sha256": child_sha}


def _resume_direction_model_factory(context: dict[str, Any]):
    build_coupled = coupled._direction_model_factory(context)

    def build(control: Tensor, reference_theta: float, gminus: Tensor, gplus: Tensor,
            normal: Tensor, chart: Tensor, pivot: int) -> list[dict[str, Any]]:
        specs = build_coupled(control, reference_theta, gminus, gplus, normal, chart, pivot)
        selected = [spec for spec in specs if spec.get("name") == "robust_gn_coupled"]
        if len(selected) != 1:
            raise ValueError("coupled-GN resume must build exactly the robust GN model")
        diagnostics = selected[0]["diagnostics"]
        minimum = context["record"].get("base_mixing_minimum", {})
        f_squared = minimum.get("F_squared")
        working_theta = minimum.get("working_theta_star")
        if (not isinstance(f_squared, (int, float)) or not isinstance(working_theta, (int, float))
                or float(working_theta) != selected[0]["working_theta"]):
            raise ValueError("coupled-GN resume minimum receipt is missing at the current point")
        point = {"index": len(context["record"].get("coupled_gn_point_history", [])),
            "base_control_sha256": tangent._tensor_sha(control), "reference_theta": reference_theta,
            "working_theta_star": float(working_theta), "F_squared": float(f_squared),
            "Psi": 0.5 * float(f_squared),
            "optimizer_step": False}
        context["record"].setdefault("coupled_gn_point_history", []).append(point)
        context["record"].setdefault("coupled_gn_solve_history", []).append({
            "base_control_sha256": point["base_control_sha256"],
            "audit": {key: diagnostics[key] for key in ("B", "S", "dense_solves", "dense_solve_dimension",
                "S_dimension", "positive_definite_from_cholesky", "solve_residual", "solve_residual_budget",
                "tangent_descent_product", "tangent_descent_budget")}})
        return selected
    return build


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child_impl(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA,
        direction_model_factory=_resume_direction_model_factory)


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA,
        direction_model_factory=_resume_direction_model_factory,
        initial_theta=BASE_THETA)


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    return tangent.run(plan_path, plan_sha, output, resource, log,
        plan_loader=_load_plan, child_script=Path(__file__), base_control_sha256=BASE_CONTROL_SHA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path,
        default=EVIDENCE / "tangent_coupled_gn_resume_20261010_attempt1/step.json")
    parser.add_argument("--resource", type=Path,
        default=EVIDENCE / "tangent_coupled_gn_resume_20261010_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path,
        default=EVIDENCE / "tangent_coupled_gn_resume_20261010_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output, args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
