"""Resume the two-commit streamed GN prefix under a 2 GiB RSS cap."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_partial_resume as partial
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_resume as gn_resume

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_COUPLED_GN_MEMORY_RESUME_PLAN_20261010.json"
PARENT_PLAN = partial.PLAN
PARENT_PLAN_SHA = "529110d25ca28ac85ab3ce3fa2fd23a8b2a8ba0122c2f184ab5e26a3c20b4f17"
PARENT_ARCHIVE_MANIFEST = EVIDENCE / "STREAMGN_ARCHIVE_20261010.json"
PARENT_ARCHIVE_MANIFEST_SHA = "4085ca18c607193119bb66427f5161debaf7f9c2481e549c84f6f36247c39953"
PARENT_ARCHIVE = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt2/step.json.gz"
PARENT_RAW = PARENT_ARCHIVE.with_suffix("")
PARENT_RUN = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt2/step.run.json"
PARENT_RESOURCE = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt2/step.resource.json"
PARENT_RAW_SHA = "a17272f51451e99b20e03d06122b3cfc79d7415e50c67ea16cb6c205e8f43497"
PARENT_GZIP_SHA = "24c84eab4b4d4f108e65581e02b2bd44baed261dacc2113c2bdb670f0d2ca3e7"
PARENT_RUN_SHA = "03b25b9105ba0133e9ac11350e732b47e4c61149db52d0cac506b2b3cb85676a"
PARENT_RESOURCE_SHA = "b4d6faf9d2bdaf00f5d8fc29a9c9564fb95a2f287e5a7cd30bed2d1bef1c1d4a"

BASE_CONTROL_SHA = "f2ca572936a8bf646c920c542b3780ff65a0f94c0d715aa61ba00a5cec57313d"
BASE_THETA = 0.4830637745604483
BASE_OBJECTIVE = 0.06119270839098437
BASE_F_SQUARED = 0.004479439946247095
FACE = partial.FACE
SELF = "examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_memory_resume.py"
TEST = "tests/test_fv_point_3h_tangent_coupled_gn_memory_resume.py"
POLICY = {**partial.POLICY, "rss_bytes": 2 * 1024**3}
METHOD = dict(partial.METHOD)
PROVENANCE = dict(partial.PROVENANCE)
RSS_TRANSITION = {
    "predecessor_rss_limit_bytes": partial.POLICY["rss_bytes"],
    "predecessor_sampled_peak_rss_bytes": 1091633152,
    "next_rss_limit_bytes": 2 * 1024**3,
    "strategy": "raise_only_the_bounded_rss_cap; preserve_physics_observations_and_gn_operator",
}
STREAM_ARCHIVE = EVIDENCE / "STREAMGN_ARCHIVE_20261010.json"
STREAM_ATTEMPT = EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closed_two_step_prefix(raw: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    iterations = raw.get("iterations", [])
    if len(iterations) != 2:
        raise ValueError("memory resume requires exactly two accepted prefix iterations")
    last_closure = raw.get("last_confirmed_closure")
    expected_control_sha = raw.get("base_control_sha256")
    expected_theta = raw.get("initial_theta")
    for index, item in enumerate(iterations):
        arms = [arm for arm in item.get("model_comparisons", [])
            if arm.get("name") == "robust_gn_coupled"]
        accepted = [trial for arm in arms for trial in arm.get("trials", [])
            if trial.get("accepted") is True]
        closure = item.get("final_repeat")
        if (item.get("accepted") is not True or item.get("selected_direction_model") != "robust_gn_coupled"
                or item.get("index") != index
                or item.get("base_control_sha256") != expected_control_sha
                or item.get("theta") != expected_theta
                or item.get("reference_theta") != expected_theta
                or len(arms) != 1 or arms[0].get("accepted") is not True or len(accepted) != 1
                or not isinstance(closure, dict)
                or not all(closure.get(key) is True for key in (
                    "mixing_minimum_valid", "minimum_theta_matches_proposal", "side_gradients_finite",
                    "native_objective_matches_proposal", "side_objectives_match_native", "merit_matches_proposal",
                    "face_audit_passed", "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
                    "fixed_input_unchanged", "runtime_unchanged", "deadline_passed"))
                or item.get("committed_control") != closure.get("control")
                or item.get("committed_theta") != closure.get("theta")
                or accepted[0].get("objective") != closure.get("objective")
                or accepted[0].get("F_squared") != closure.get("F_squared")
                or not tangent.fresh_final_closure(accepted[0], closure)):
            raise ValueError(f"memory resume P2 closure failed for prefix iteration {index}")
        expected_control_sha = tangent._tensor_sha(torch.as_tensor(closure["control"], dtype=torch.float64))
        expected_theta = closure["theta"]
    closure = iterations[-1]["final_repeat"]
    if (closure != last_closure or raw.get("current_control") != closure.get("control")
            or raw.get("last_confirmed_control") != closure.get("control")
            or closure.get("theta") != BASE_THETA
            or closure.get("objective") != BASE_OBJECTIVE
            or closure.get("F_squared") != BASE_F_SQUARED):
        raise ValueError("memory resume terminal prefix closure differs from frozen endpoint")
    control = torch.as_tensor(closure["control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_CONTROL_SHA:
        raise ValueError("memory resume terminal prefix control digest mismatch")
    return closure, control


def _validate_two_step_packet(raw: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    if (raw.get("accepted_iterations") != 2 or raw.get("optimizer_steps_applied") != 2
            or raw.get("last_confirmed_iterations") != 2
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_CONTROL_SHA
            or raw.get("current_theta") != BASE_THETA or raw.get("last_confirmed_theta") != BASE_THETA
            or raw.get("hvp_calls_started") != 4 or raw.get("hvp_calls_completed") != 4
            or raw.get("jacobian_rows_started") != 72 or raw.get("jacobian_rows_completed") != 72
            or raw.get("dense_solves_started") != 3 or raw.get("dense_solves_completed") != 3
            or raw.get("candidate_committed") is not True or raw.get("active_candidate_committed") is not False
            or raw.get("minimum_claim") is not False or raw.get("response_claim") is not False
            or raw.get("full_smooth_root") is not False or raw.get("comparison_progress") is not None):
        raise ValueError("two-step prefix counts or current boundary are invalid")
    closure, control = _closed_two_step_prefix(raw)
    if (raw.get("current_control") != closure["control"]
            or raw.get("last_confirmed_control") != closure["control"]):
        raise ValueError("two-step prefix current control differs from final P2 closure")
    first_control_sha = raw["iterations"][0]["base_control_sha256"]
    second_control_sha = raw["iterations"][1]["base_control_sha256"]
    row_history = raw.get("jacobian_row_history", [])
    expected_row_bases = {first_control_sha, second_control_sha, BASE_CONTROL_SHA}
    if set(row.get("base_control_sha256") for row in row_history) != expected_row_bases:
        raise ValueError("two-step raw contains rows at an unexpected control point")
    for point_sha in expected_row_bases:
        rows = [row for row in row_history if row.get("base_control_sha256") == point_sha]
        pairs = {(row.get("side"), row.get("row")) for row in rows}
        if (len(rows) != 24 or any(row.get("status") != "completed" for row in rows)
                or pairs != {(side, row) for side in (-1, 1) for row in range(12)}):
            raise ValueError("two-step raw has an incomplete 24-row point")
    hvps = raw.get("hvp_history", [])
    if (len(hvps) != 4 or any(row.get("status") != "completed" for row in hvps)
            or {row.get("base_control_sha256") for row in hvps} != {first_control_sha, second_control_sha}
            or any(row.get("base_control_sha256") == BASE_CONTROL_SHA for row in hvps)
            or raw.get("current_hvp", {}).get("base_control_sha256") != second_control_sha
            or len(raw.get("coupled_gn_point_history", [])) != 2
            or [point.get("base_control_sha256") for point in raw["coupled_gn_point_history"]]
                != [first_control_sha, second_control_sha]
            or len(raw.get("coupled_gn_solve_history", [])) != 2
            or [point.get("base_control_sha256") for point in raw["coupled_gn_solve_history"]]
                != [first_control_sha, second_control_sha]
            or raw.get("current_jacobian_row", {}).get("base_control_sha256") != BASE_CONTROL_SHA
            or raw.get("current_jacobian_row", {}).get("status") != "completed"
            or raw.get("current_jacobian_row", {}).get("row") != 11
            or raw.get("current_jacobian_row", {}).get("side") != 1
            or raw.get("current_jacobian_row", {}).get("theta") != BASE_THETA):
        raise ValueError("two-step prefix/HVP history is not closed before the third current point")
    current_minimum = raw.get("base_mixing_minimum", {})
    audit = raw.get("dense_solve_audit", {})
    if (current_minimum.get("working_theta_star") != BASE_THETA
            or current_minimum.get("F_squared") != BASE_F_SQUARED
            or current_minimum.get("optimizer_step") is not False
            or audit.get("dense_solves") != 1 or audit.get("dense_solve_dimension") != 12
            or audit.get("S_dimension") != 12 or audit.get("positive_definite_from_cholesky") is not True
            or not isinstance(audit.get("solve_residual"), (int, float))
            or audit["solve_residual"] > audit.get("solve_residual_budget", -1.0)
            or not isinstance(audit.get("tangent_descent_product"), (int, float))
            or audit["tangent_descent_product"] >= 0):
        raise ValueError("third-point GN solve audit is invalid or was already committed")
    return closure, control


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("memory resume plan identity mismatch")
    prior = partial._load_plan(PARENT_PLAN, PARENT_PLAN_SHA)
    plan = json.loads(path.read_text())
    if (plan.get("experiment_kind") != "current_tangent_coupled_gn_partial_resume"
            or plan.get("policy") != POLICY or plan.get("method") != METHOD
            or plan.get("face") != FACE
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA
            or plan.get("base_objective") != BASE_OBJECTIVE
            or plan.get("base_carried_F_squared") != BASE_F_SQUARED
            or plan.get("predecessor_plan") != PARENT_PLAN.relative_to(ROOT).as_posix()
            or plan.get("predecessor_plan_sha256") != PARENT_PLAN_SHA
            or plan.get("predecessor_archive_manifest") != PARENT_ARCHIVE_MANIFEST.relative_to(ROOT).as_posix()
            or plan.get("predecessor_archive_manifest_sha256") != PARENT_ARCHIVE_MANIFEST_SHA
            or plan.get("predecessor_archive") != PARENT_ARCHIVE.relative_to(ROOT).as_posix()
            or plan.get("predecessor_archive_sha256") != PARENT_GZIP_SHA
            or plan.get("predecessor_raw_sha256") != PARENT_RAW_SHA
            or plan.get("predecessor_run") != PARENT_RUN.relative_to(ROOT).as_posix()
            or plan.get("predecessor_run_sha256") != PARENT_RUN_SHA
            or plan.get("predecessor_resource") != PARENT_RESOURCE.relative_to(ROOT).as_posix()
            or plan.get("predecessor_resource_sha256") != PARENT_RESOURCE_SHA
            or plan.get("anchor_provenance") != PROVENANCE
            or plan.get("rss_transition") != RSS_TRANSITION):
        raise ValueError("memory resume plan metadata or predecessor pins changed")
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if not isinstance(sources, dict) or not isinstance(archives, dict):
        raise ValueError("memory resume source/archive pins are missing")
    if (len(sources) != 144 or set(sources) != set(prior["source_files"]) | {SELF, TEST}
            or len(archives) != 173
            or set(archives) != set(prior["archive_files"]) | {
                PARENT_PLAN.relative_to(ROOT).as_posix(), PARENT_ARCHIVE_MANIFEST.relative_to(ROOT).as_posix(),
                PARENT_ARCHIVE.relative_to(ROOT).as_posix(), PARENT_RUN.relative_to(ROOT).as_posix(),
                PARENT_RESOURCE.relative_to(ROOT).as_posix()}
            or any(archives.get(name) != value for name, value in prior["archive_files"].items())
            or any(sources.get(name) != value for name, value in prior["source_files"].items())
            or sources.get(SELF) != _sha(ROOT / SELF) or sources.get(TEST) != _sha(ROOT / TEST)):
        raise ValueError("memory resume plan scope or inherited pins changed")
    if any((ROOT / name).is_symlink() or not (ROOT / name).resolve().is_relative_to(ROOT.resolve())
            or _sha(ROOT / name) != value for name, value in {**sources, **archives}.items()):
        raise ValueError("memory resume source/archive digest mismatch")
    return plan


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    """Admit the two P2 commits and the third-point pre-HVP/commit boundary."""
    partial._load_plan(PARENT_PLAN, PARENT_PLAN_SHA)
    if any(_sha(path) != digest for path, digest in (
            (PARENT_PLAN, PARENT_PLAN_SHA), (PARENT_ARCHIVE_MANIFEST, PARENT_ARCHIVE_MANIFEST_SHA),
            (PARENT_ARCHIVE, PARENT_GZIP_SHA), (PARENT_RUN, PARENT_RUN_SHA),
            (PARENT_RESOURCE, PARENT_RESOURCE_SHA))):
        raise ValueError("two-step predecessor artifact digest mismatch")
    archive_manifest = json.loads(PARENT_ARCHIVE_MANIFEST.read_text())
    compressed = PARENT_ARCHIVE.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha != PARENT_RAW_SHA:
        raise ValueError("two-step predecessor raw digest mismatch")
    raw = json.loads(raw_bytes)
    parent_run = json.loads(PARENT_RUN.read_text())
    resource = json.loads(PARENT_RESOURCE.read_text())
    parent_plan = json.loads(PARENT_PLAN.read_text())
    inherited_receipt = {**parent_plan["source_files"], **parent_plan["archive_files"],
        PARENT_PLAN.relative_to(ROOT).as_posix(): PARENT_PLAN_SHA}
    if (archive_manifest.get("raw_path") != PARENT_RAW.relative_to(ROOT).as_posix()
            or archive_manifest.get("gzip_path") != PARENT_ARCHIVE.relative_to(ROOT).as_posix()
            or archive_manifest.get("raw_sha256") != PARENT_RAW_SHA
            or archive_manifest.get("gzip_sha256") != PARENT_GZIP_SHA
            or archive_manifest.get("run_sha256") != PARENT_RUN_SHA
            or archive_manifest.get("resource_sha256") != PARENT_RESOURCE_SHA
            or archive_manifest.get("confirmed_prefix_steps") != 2
            or archive_manifest.get("terminal_execution_closed") is not False
            or archive_manifest.get("lossless_roundtrip") is not True
            or archive_manifest.get("total_session_guard_starts") != 2
            or parent_run.get("execution_status") != "rss_limit"
            or parent_run.get("numerical_status") != "fresh_current_base_closed"
            or parent_run.get("child_read_error") is not None or parent_run.get("child_sha256") != PARENT_RAW_SHA
            or parent_run.get("resource") != resource
            or resource.get("exit_code") != -15 or resource.get("resource_termination") != "rss_limit"
            or resource.get("monitor_error") is not None or resource.get("received_sigterm") is not False
            or resource.get("rss_limit_bytes") != partial.POLICY["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes") != RSS_TRANSITION["predecessor_sampled_peak_rss_bytes"]
            or resource.get("wall_limit_seconds") != partial.POLICY["outer_seconds"]
            or raw.get("plan_sha256") != PARENT_PLAN_SHA
            or raw.get("plan_hashes", {}).get("source_files") != parent_plan["source_files"]
            or raw.get("plan_hashes", {}).get("archive_files") != parent_plan["archive_files"]
            or raw.get("source_before") != inherited_receipt
            or raw.get("source_after") is not None
            or raw.get("input_after") is not None or raw.get("runtime_after") is not None
            or raw.get("phase") != "running" or raw.get("execution_status") != "running"
            or raw.get("numerical_status") != "fresh_current_base_closed"
            or raw.get("phase") != "running" or raw.get("execution_status") != "running"
            or raw.get("numerical_status") != "fresh_current_base_closed"):
        raise ValueError("two-step RSS predecessor receipt did not close its expected boundary")
    closure, control = _validate_two_step_packet(raw)

    predecessor = gn_resume._load_current_base({})
    input_anchor = dict(predecessor["raw"]["input_after"])
    input_anchor["control_sha256"] = BASE_CONTROL_SHA
    input_anchor["shifted_flow_fractions"] = torch.tanh(control[20:25]).tolist()
    if not geometry.model._fixed_input(input_anchor, predecessor["raw"]["input_after"], BASE_CONTROL_SHA):
        raise ValueError("derived two-step current input violates the fixed-input contract")
    return {"raw": {"input_after": input_anchor, "runtime_after": predecessor["raw"]["runtime_after"]},
        "control": control, "theta": BASE_THETA, "objective": BASE_OBJECTIVE,
        "accepted": closure, "child_sha256": PARENT_RAW_SHA, "anchor_provenance": dict(PROVENANCE)}


def _resume_direction_model_factory(context: dict[str, Any]):
    return partial._resume_direction_model_factory(context)


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child_impl(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA,
        direction_model_factory=_resume_direction_model_factory)


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA,
        direction_model_factory=_resume_direction_model_factory, initial_theta=BASE_THETA)


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    return tangent.run(plan_path, plan_sha, output, resource, log,
        plan_loader=_load_plan, child_script=Path(__file__), base_control_sha256=BASE_CONTROL_SHA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    output = args.output or EVIDENCE / "tangent_coupled_gn_memory_resume_20261010_attempt1/step.json"
    if args.child:
        _run_child_impl(args.plan, args.plan_sha256, output)
    else:
        run(args.plan, args.plan_sha256, output,
            args.resource or output.with_suffix(".resource.json"),
            args.log or output.with_suffix(".log"))


if __name__ == "__main__":
    main()
