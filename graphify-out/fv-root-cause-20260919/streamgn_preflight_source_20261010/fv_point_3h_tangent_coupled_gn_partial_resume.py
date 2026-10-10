"""Resume PR274's closed GN commit after its RSS-interrupted raw tail."""
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
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_resume as gn_resume

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_COUPLED_GN_PARTIAL_RESUME_PLAN_20261010.json"
PARTIAL_PLAN = EVIDENCE / "TANGENT_COUPLED_GN_RESUME_PLAN_20261010.json"
PARTIAL_PLAN_SHA = "1f0eff706a21982407f79b51375687f41aca85ee34433229d8faa8e060a19ce0"
PARTIAL_ARCHIVE = EVIDENCE / "tangent_coupled_gn_resume_20261010_attempt1/step.json.gz"
PARTIAL_RAW = PARTIAL_ARCHIVE.with_suffix("")
PARTIAL_RUN = EVIDENCE / "tangent_coupled_gn_resume_20261010_attempt1/step.run.json"
PARTIAL_RESOURCE = EVIDENCE / "tangent_coupled_gn_resume_20261010_attempt1/step.resource.json"
PARTIAL_MANIFEST = EVIDENCE / "GNRESUME_ARCHIVE_20261010.json"
AUDIT_SCOPE = EVIDENCE / "AUDIT274_SOURCE_SCOPE_20261010.json"
AUDIT_SCOPE_SHA = "973fc1f6e74ff201a6e1884afb167a63cbdd9f47a4ae1f69227568c185bc2a00"
AUDIT_SOURCE_MANIFEST = EVIDENCE / "audit274_source_20261010/manifest.json"
AUDIT_SOURCE_MANIFEST_SHA = "c9415aafa484527f0b64ba3ace33384637d30bc1d779ebf1307eba5ba5d5a5bb"
AUDIT_CORE_SNAPSHOT = EVIDENCE / "audit274_source_20261010/fv_point_3h_tangent_continuation.py"
AUDIT_TEST_SNAPSHOT = EVIDENCE / "audit274_source_20261010/test_fv_point_3h_tangent_coupled_gn_resume.py"
STREAM_SOURCE_MANIFEST = EVIDENCE / "streamgn_source_20261010/manifest.json"
STREAM_SOURCE_MANIFEST_SHA = "8b4ad51d73b19dc6043f191a6e39b567d5ceb166a09d378e8e030abef1eebdd4"
STREAM_CORE_SNAPSHOT = EVIDENCE / "streamgn_source_20261010/fv_point_3h_tangent_continuation.py"
STREAM_TEST = "tests/test_fv_point_3h_tangent_stream_checkpoint.py"
PARTIAL_RAW_SHA = "33adef17be48e9fbc42d28fd4c2f794c1c5c19ff1612adec6d0ef203ddc5ee05"
PARTIAL_GZIP_SHA = "07b5a611a3e0d035841c4136d294ce7cf45a0ba86276eff7d509f4e23f0bf18c"
PARTIAL_RUN_SHA = "da8420ceeca46c486506d9c9e150213759e7b31e5a79614fc342dbeb195ab611"
PARTIAL_RESOURCE_SHA = "de3c552efd3ac93e961eab38c4561982666041abdd71bd38e56670eb76d97ba1"

BASE_CONTROL_SHA = "7cec4c62bc2ebe3a96106086408adc0ecdbc959fe1b53beafa46902cd41fb7dd"
BASE_THETA = 0.4829895933920232
BASE_OBJECTIVE = 0.061195305062512445
BASE_F_SQUARED = 0.004486899769141486
FACE = gn_resume.FACE
SELF = "examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_partial_resume.py"
TEST = "tests/test_fv_point_3h_tangent_coupled_gn_partial_resume.py"
POLICY = dict(gn_resume.POLICY)
METHOD = dict(gn_resume.METHOD)
PROVENANCE = {
    "input": "derived_from_pr273_input_after_control_sha_only_per_fixed_input_contract",
    "runtime": "reused_from_pr273_runtime_after",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closed_first_commit(raw: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    """Return the committed prefix endpoint after separating the unfinished tail."""
    iterations = raw.get("iterations", [])
    closure = raw.get("last_confirmed_closure")
    if len(iterations) != 1 or not isinstance(closure, dict):
        raise ValueError("partial GN resume is missing its single committed closure")
    step = iterations[0]
    if (step.get("accepted") is not True or step.get("selected_direction_model") != "robust_gn_coupled"
            or step.get("committed_control") != raw.get("last_confirmed_control")
            or step.get("committed_control") != raw.get("current_control")
            or step.get("committed_theta") != BASE_THETA
            or step.get("final_repeat") != closure
            or closure.get("control") != step.get("committed_control")
            or closure.get("theta") != BASE_THETA or closure.get("objective") != BASE_OBJECTIVE
            or closure.get("F_squared") != BASE_F_SQUARED
            or closure.get("mixing_minimum_valid") is not True
            or closure.get("minimum_theta_matches_proposal") is not True
            or not all(closure.get(key) is True for key in (
                "side_gradients_finite", "native_objective_matches_proposal", "side_objectives_match_native",
                "merit_matches_proposal", "face_audit_passed", "branch_pair_passed", "trace_matches_proposal",
                "source_unchanged", "fixed_input_unchanged", "runtime_unchanged", "deadline_passed"))):
        raise ValueError("partial GN resume first committed P2 closure is invalid")
    gn_arms = [arm for arm in step.get("model_comparisons", [])
        if arm.get("name") == "robust_gn_coupled"]
    accepted_trials = [trial for arm in gn_arms for trial in arm.get("trials", [])
        if trial.get("accepted") is True]
    if (len(gn_arms) != 1 or gn_arms[0].get("accepted") is not True
            or len(accepted_trials) != 1
            or accepted_trials[0].get("objective") != BASE_OBJECTIVE
            or accepted_trials[0].get("F_squared") != BASE_F_SQUARED
            or not tangent.fresh_final_closure(accepted_trials[0], closure)):
        raise ValueError("partial GN resume accepted trial does not close through P2")
    control = torch.as_tensor(closure["control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_CONTROL_SHA:
        raise ValueError("partial GN resume committed control digest is invalid")

    rows = raw.get("jacobian_row_history", [])
    first_sha = step.get("base_control_sha256")
    first_rows = [row for row in rows if row.get("base_control_sha256") == first_sha]
    tail = [row for row in rows if row.get("base_control_sha256") == BASE_CONTROL_SHA]
    ordered_tail = sorted(tail, key=lambda row: row.get("row", -1))
    row_pairs = {(row.get("side"), row.get("row")) for row in first_rows if row.get("status") == "completed"}
    hvps = raw.get("hvp_history", [])
    if (len(first_rows) != 24 or row_pairs != {(side, row) for side in (-1, 1) for row in range(12)}
            or any(row.get("status") != "completed" for row in first_rows)
            or len(tail) != 2 or sorted(row.get("status") for row in tail) != ["completed", "started"]
            or [(row.get("side"), row.get("row"), row.get("status")) for row in ordered_tail]
                != [(-1, 0, "completed"), (-1, 1, "started")]
            or any(row.get("theta") != BASE_THETA for row in ordered_tail)
            or sum(h.get("base_control_sha256") == first_sha and h.get("status") == "completed"
                for h in hvps) != 2
            or any(h.get("base_control_sha256") == BASE_CONTROL_SHA for h in hvps)
            or raw.get("current_hvp", {}).get("base_control_sha256") != first_sha
            or raw.get("current_jacobian_row", {}).get("base_control_sha256") != BASE_CONTROL_SHA
            or raw.get("current_jacobian_row", {}).get("row") != 1
            or raw.get("current_jacobian_row", {}).get("status") != "started"
            or raw.get("current_jacobian_row", {}).get("theta") != BASE_THETA
            or len(raw.get("coupled_gn_point_history", [])) != 1
            or raw["coupled_gn_point_history"][0].get("base_control_sha256") != first_sha
            or len(raw.get("coupled_gn_solve_history", [])) != 1
            or raw["coupled_gn_solve_history"][0].get("base_control_sha256") != first_sha
            or raw.get("comparison_progress", {}).get("base_control_sha256") != first_sha
            or raw.get("dense_solve_audit", {}).get("dense_solves") != 1):
        raise ValueError("partial GN resume prefix/tail boundary is invalid")
    return closure, control


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("partial GN resume plan identity mismatch")
    plan = json.loads(path.read_text())
    prior = json.loads(PARTIAL_PLAN.read_text())
    if (_sha(PARTIAL_PLAN) != PARTIAL_PLAN_SHA
            or plan.get("experiment_kind") != "current_tangent_coupled_gn_partial_resume"
            or plan.get("policy") != POLICY or plan.get("method") != METHOD
            or plan.get("face") != FACE
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA
            or plan.get("base_objective") != BASE_OBJECTIVE
            or plan.get("base_carried_F_squared") != BASE_F_SQUARED
            or plan.get("predecessor_plan_sha256") != PARTIAL_PLAN_SHA
            or plan.get("predecessor_raw_sha256") != PARTIAL_RAW_SHA
            or plan.get("predecessor_gzip_sha256") != PARTIAL_GZIP_SHA
            or plan.get("predecessor_run_sha256") != PARTIAL_RUN_SHA
            or plan.get("predecessor_resource_sha256") != PARTIAL_RESOURCE_SHA
            or plan.get("predecessor_manifest_sha256") != _sha(PARTIAL_MANIFEST)
            or plan.get("writer_source_manifest") != STREAM_SOURCE_MANIFEST.relative_to(ROOT).as_posix()
            or plan.get("writer_source_manifest_sha256") != STREAM_SOURCE_MANIFEST_SHA
            or plan.get("audited_source_scope") != AUDIT_SCOPE.relative_to(ROOT).as_posix()
            or plan.get("audited_source_scope_sha256") != AUDIT_SCOPE_SHA
            or plan.get("audited_source_manifest") != AUDIT_SOURCE_MANIFEST.relative_to(ROOT).as_posix()
            or plan.get("audited_source_manifest_sha256") != AUDIT_SOURCE_MANIFEST_SHA):
        raise ValueError("partial GN resume plan scope or predecessor pins changed")
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if not isinstance(sources, dict) or not isinstance(archives, dict):
        raise ValueError("partial GN resume source/archive pins are missing")
    if not {SELF, TEST}.issubset(sources):
        raise ValueError("partial GN resume plan must pin its producer and test")
    audit_scope = json.loads(AUDIT_SCOPE.read_text())
    audit_manifest = json.loads(AUDIT_SOURCE_MANIFEST.read_text())
    audit_snapshots = audit_manifest.get("snapshots", [])
    if not isinstance(audit_snapshots, list):
        raise ValueError("audited source snapshot manifest is malformed")
    audit_snapshots_by_source = {entry.get("source_path"): entry
        for entry in audit_snapshots if isinstance(entry, dict)}
    if len(audit_snapshots_by_source) != len(audit_snapshots):
        raise ValueError("audited source snapshot manifest has duplicate or invalid paths")
    stream_manifest = json.loads(STREAM_SOURCE_MANIFEST.read_text())
    additions = set(sources) - set(prior["source_files"])
    stream_tests = {name for name in additions - {SELF, TEST} if name == STREAM_TEST}
    archive_additions = {
        PARTIAL_PLAN.relative_to(ROOT).as_posix(), PARTIAL_MANIFEST.relative_to(ROOT).as_posix(),
        PARTIAL_ARCHIVE.relative_to(ROOT).as_posix(), PARTIAL_RUN.relative_to(ROOT).as_posix(),
        PARTIAL_RESOURCE.relative_to(ROOT).as_posix(), AUDIT_SCOPE.relative_to(ROOT).as_posix(),
        AUDIT_SOURCE_MANIFEST.relative_to(ROOT).as_posix(), AUDIT_CORE_SNAPSHOT.relative_to(ROOT).as_posix(),
        AUDIT_TEST_SNAPSHOT.relative_to(ROOT).as_posix(), STREAM_SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        STREAM_CORE_SNAPSHOT.relative_to(ROOT).as_posix(),
    }
    audited = {
        "examples/weather_scenarios/fv_point_3h_tangent_continuation.py": (
            AUDIT_CORE_SNAPSHOT, "eca3f2232cb08c3db42783b48aab09c09464b6b288d4d6a037f04c8f30f66256"),
        "tests/test_fv_point_3h_tangent_coupled_gn_resume.py": (
            AUDIT_TEST_SNAPSHOT, "4550823578644cf50837e66018e8ddf2a8da9bb0c78d97a35c871230333715e5"),
    }
    stream_snapshot = stream_manifest.get("snapshots", {}).get(
        "examples/weather_scenarios/fv_point_3h_tangent_continuation.py", {})
    if (len(sources) != 142 or additions != {SELF, TEST, *stream_tests} or len(stream_tests) != 1
            or len(archives) != 159
            or set(archives) != set(prior["archive_files"]) | archive_additions
            or any(archives.get(name) != digest for name, digest in prior["archive_files"].items())
            or _sha(AUDIT_SCOPE) != AUDIT_SCOPE_SHA
            or _sha(AUDIT_SOURCE_MANIFEST) != AUDIT_SOURCE_MANIFEST_SHA
            or _sha(STREAM_SOURCE_MANIFEST) != STREAM_SOURCE_MANIFEST_SHA
            or audit_scope.get("old_plan_sha256") != PARTIAL_PLAN_SHA
            or audit_scope.get("source_refs") != 139 or audit_scope.get("archive_refs") != 148
            or any(audit_snapshots_by_source.get(name, {}).get("sha256") != snapshot_sha
                or audit_snapshots_by_source.get(name, {}).get("archive_path")
                    != snapshot_path.relative_to(ROOT).as_posix()
                for name, (snapshot_path, snapshot_sha) in audited.items())
            or any(audit_scope.get("declared_changed_files", {}).get(name, {}).get("executed_sha256")
                    != snapshot_sha
                for name, (_path, snapshot_sha) in audited.items())
            or stream_snapshot.get("sha256") != "e6d6aa78d5f88792fa981978cded6caa9984b2d94ea58ad6cfddea4c1b251c42"
            or stream_snapshot.get("archive_path") != STREAM_CORE_SNAPSHOT.relative_to(ROOT).as_posix()):
        raise ValueError("partial GN resume source/archive successor scope changed")
    for name, (snapshot_path, snapshot_sha) in audited.items():
        if (sources.get(name) != _sha(ROOT / name)
                or archives.get(snapshot_path.relative_to(ROOT).as_posix()) != snapshot_sha
                or _sha(snapshot_path) != snapshot_sha):
            raise ValueError(f"audited PR274 source snapshot/current pin mismatch: {name}")
    for name, digest in prior["source_files"].items():
        if name not in audited and sources.get(name) != digest:
            raise ValueError(f"inherited PR273 source pin changed: {name}")
    if (sources.get("examples/weather_scenarios/fv_point_3h_tangent_continuation.py")
            != _sha(ROOT / "examples/weather_scenarios/fv_point_3h_tangent_continuation.py")
            or archives.get(stream_snapshot["archive_path"]) != stream_snapshot["sha256"]
            or _sha(STREAM_CORE_SNAPSHOT) != stream_snapshot["sha256"]):
        raise ValueError("streaming writer pre-edit source snapshot/current pin mismatch")
    if any((ROOT / name).is_symlink() or not (ROOT / name).resolve().is_relative_to(ROOT.resolve())
            or _sha(ROOT / name) != value for name, value in {**sources, **archives}.items()):
        raise ValueError("partial GN resume source/archive digest mismatch")
    if plan.get("anchor_provenance") != PROVENANCE:
        raise ValueError("partial GN resume anchor provenance is not explicit")
    return plan


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    """Admit only the one fully closed PR274 commit; discard its open raw tail."""
    pins = ((PARTIAL_PLAN, PARTIAL_PLAN_SHA), (PARTIAL_ARCHIVE, PARTIAL_GZIP_SHA),
            (PARTIAL_RUN, PARTIAL_RUN_SHA), (PARTIAL_RESOURCE, PARTIAL_RESOURCE_SHA),
            (PARTIAL_MANIFEST, _sha(PARTIAL_MANIFEST)))
    if any(_sha(path) != digest for path, digest in pins):
        raise ValueError("partial GN resume predecessor receipt digest mismatch")

    old_plan = json.loads(PARTIAL_PLAN.read_text())
    manifest = json.loads(PARTIAL_MANIFEST.read_text())
    compressed = PARTIAL_ARCHIVE.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    if hashlib.sha256(raw_bytes).hexdigest() != PARTIAL_RAW_SHA:
        raise ValueError("partial GN resume raw child digest mismatch")
    raw = json.loads(raw_bytes)
    parent = json.loads(PARTIAL_RUN.read_text())
    resource = json.loads(PARTIAL_RESOURCE.read_text())
    inherited_source_receipt = {**old_plan["source_files"], **old_plan["archive_files"],
        PARTIAL_PLAN.relative_to(ROOT).as_posix(): PARTIAL_PLAN_SHA}
    if (manifest.get("raw_path") != PARTIAL_RAW.relative_to(ROOT).as_posix()
            or manifest.get("gzip_path") != PARTIAL_ARCHIVE.relative_to(ROOT).as_posix()
            or manifest.get("raw_bytes") != len(raw_bytes)
            or manifest.get("gzip_bytes") != len(compressed)
            or manifest.get("run_path") != PARTIAL_RUN.relative_to(ROOT).as_posix()
            or manifest.get("resource_path") != PARTIAL_RESOURCE.relative_to(ROOT).as_posix()
            or manifest.get("raw_sha256") != PARTIAL_RAW_SHA
            or manifest.get("gzip_sha256") != PARTIAL_GZIP_SHA
            or manifest.get("run_sha256") != PARTIAL_RUN_SHA
            or manifest.get("resource_sha256") != PARTIAL_RESOURCE_SHA
            or manifest.get("guarded_launch_count") != 1 or manifest.get("lossless_roundtrip") is not True
            or manifest.get("terminal_execution_closed") is not False
            or parent.get("execution_status") != "rss_limit"
            or parent.get("numerical_status") != "fresh_current_base_closed"
            or parent.get("child_read_error") is not None or parent.get("child_sha256") != PARTIAL_RAW_SHA
            or parent.get("resource") != resource
            or resource.get("resource_termination") != "rss_limit"
            or resource.get("exit_code") != -15 or resource.get("monitor_error") is not None
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or resource.get("wall_limit_seconds") != POLICY["outer_seconds"]
            or resource.get("received_sigterm") is not False
            or not isinstance(resource.get("sampled_peak_rss_bytes"), (int, float))
            or resource["sampled_peak_rss_bytes"] < POLICY["rss_bytes"]
            or raw.get("plan_sha256") != PARTIAL_PLAN_SHA
            or raw.get("plan_hashes", {}).get("source_files") != old_plan["source_files"]
            or raw.get("plan_hashes", {}).get("archive_files") != old_plan["archive_files"]
            or raw.get("source_before") != inherited_source_receipt
            or raw.get("source_after") is not None
            or raw.get("input_after") is not None or raw.get("runtime_after") is not None
            or raw.get("phase") != "running" or raw.get("execution_status") != "running"
            or raw.get("numerical_status") != "fresh_current_base_closed"
            or raw.get("accepted_iterations") != 1 or raw.get("optimizer_steps_applied") != 1
            or raw.get("last_confirmed_iterations") != 1
            or raw.get("candidate_committed") is not True or raw.get("active_candidate_committed") is not False
            or raw.get("hvp_calls_started") != 2 or raw.get("hvp_calls_completed") != 2
            or raw.get("dense_solves_started") != 1 or raw.get("dense_solves_completed") != 1
            or raw.get("jacobian_rows_started") != 26 or raw.get("jacobian_rows_completed") != 25
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_CONTROL_SHA
            or raw.get("current_theta") != BASE_THETA or raw.get("last_confirmed_theta") != BASE_THETA):
        raise ValueError("RSS-interrupted GN resume receipt is not the expected partial prefix")

    closure, control = _closed_first_commit(raw)

    predecessor = gn_resume._load_current_base({})
    predecessor_input = predecessor["raw"]["input_after"]
    current_input = dict(predecessor_input)
    current_input["control_sha256"] = BASE_CONTROL_SHA
    if not geometry.model._fixed_input(current_input, predecessor_input, BASE_CONTROL_SHA):
        raise ValueError("derived current input anchor violates fixed-input contract")
    runtime = predecessor["raw"]["runtime_after"]
    return {"raw": {"input_after": current_input, "runtime_after": runtime},
        "control": control, "theta": BASE_THETA, "objective": BASE_OBJECTIVE,
        "accepted": closure, "child_sha256": PARTIAL_RAW_SHA,
        "anchor_provenance": dict(PROVENANCE)}


def _resume_direction_model_factory(context: dict[str, Any]):
    return gn_resume._resume_direction_model_factory(context)


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
    output = args.output or EVIDENCE / "tangent_coupled_gn_partial_resume_20261010_attempt1/step.json"
    if args.child:
        _run_child_impl(args.plan, args.plan_sha256, output)
    else:
        resource = args.resource or output.with_suffix(".resource.json")
        log = args.log or output.with_suffix(".log")
        run(args.plan, args.plan_sha256, output, resource, log)


if __name__ == "__main__":
    main()
