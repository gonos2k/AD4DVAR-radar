"""Resume the bounded interior mixing-minimum tangent policy from PR271."""
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
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_MIXING_MINIMUM_RESUME_PLAN_20261010.json"
PREDECESSOR_PLAN = EVIDENCE / "TANGENT_MIXING_MINIMUM_PLAN_20261010.json"
PREDECESSOR_PLAN_SHA = "9611b13d3392c198f0a92436c1e9536a6286363a2c64a70a48e5b43f6c480576"
PREDECESSOR_ARCHIVE_MANIFEST = EVIDENCE / "MIXMIN_ARCHIVE_20261010.json"
BASE_ARCHIVE = EVIDENCE / "tangent_mixing_minimum_20261010_attempt1/step.json.gz"
BASE_RUN = EVIDENCE / "tangent_mixing_minimum_20261010_attempt1/step.run.json"
BASE_RESOURCE = EVIDENCE / "tangent_mixing_minimum_20261010_attempt1/step.resource.json"
SOURCE_MANIFEST = EVIDENCE / "mixmin_resume_source_20261010/manifest.json"
PREDECESSOR_ARCHIVE_SHA = "67587f0bac6f549e9c452120a291c8765db8ce35dfa03f8dff12eb8f3b9db5ff"
BASE_RAW_SHA = "8581c6ae0d430c5c9c36c71209bfa03bd53a8627271f6cb900f60790a67d79aa"
BASE_GZIP_SHA = "77ab19d185fb7fafa9caba0395590e7efe78d12b403f02c7a6e9e649a417af3c"
BASE_RUN_SHA = "ce0dc2be160c612e1b856c17bb010c076eeadda118fc8b4c0a2a2bebc1acc4bb"
BASE_RESOURCE_SHA = "55ac5f88b79ef9668ddf4fe4e7aea2c61cb65e89022cb24639f6bb7519a90455"
BASE_CONTROL_SHA = "e801cf4651233ec11b2ec98ac9569e411b8e62e68511440e39db1cb9c3872567"
BASE_THETA = 0.48316874590279574
BASE_OBJECTIVE = 0.061240252230254005
BASE_CARRIED_F_SQUARED = 0.004689202366398875
FACE = mixing.FACE
SELF = "examples/weather_scenarios/fv_point_3h_tangent_mixing_minimum_resume.py"
TEST = "tests/test_fv_point_3h_tangent_mixing_minimum_resume.py"
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 300.0, "outer_seconds": 360.0,
    "rss_bytes": 1024**3, "hvp_calls": 6, "jacobian_row_vjp_calls": 0,
    "max_accepted_iterations": 3, "max_candidates_per_iteration": 16,
    "radius": 0.05, "face_scale": 0.84, "optimizer_steps": 3,
    "pcg_solves": 0, "dense_solves": 0, "root_claim": False,
    "minimum_claim": False, "response_claim": False, "score_claim": False,
}
METHOD = mixing.METHOD


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("mixing-minimum-resume plan identity mismatch")
    if (_sha(PREDECESSOR_PLAN) != PREDECESSOR_PLAN_SHA
            or _sha(PREDECESSOR_ARCHIVE_MANIFEST) != PREDECESSOR_ARCHIVE_SHA):
        raise ValueError("PR271 producer pins changed")
    prior = json.loads(PREDECESSOR_PLAN.read_text())
    snapshots = json.loads(SOURCE_MANIFEST.read_text())["snapshots"]
    plan = json.loads(path.read_text())
    prior_sources, prior_archives = prior["source_files"], prior["archive_files"]
    expected_sources = set(prior_sources) | {SELF, TEST}
    expected_archives = set(prior_archives) | {
        PREDECESSOR_PLAN.relative_to(ROOT).as_posix(),
        PREDECESSOR_ARCHIVE_MANIFEST.relative_to(ROOT).as_posix(),
        BASE_ARCHIVE.relative_to(ROOT).as_posix(), BASE_RUN.relative_to(ROOT).as_posix(),
        BASE_RESOURCE.relative_to(ROOT).as_posix(), SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        *[snapshot["archive_path"] for snapshot in snapshots.values()],
    }
    if (plan.get("experiment_kind") != "current_tangent_mixing_minimum_resume"
            or plan.get("policy") != POLICY or plan.get("method") != METHOD
            or plan.get("producing_plan") != PREDECESSOR_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != PREDECESSOR_PLAN_SHA
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA
            or plan.get("base_objective") != BASE_OBJECTIVE
            or plan.get("base_carried_F_squared") != BASE_CARRIED_F_SQUARED
            or plan.get("face") != FACE
            or plan.get("producer_source_snapshots") != snapshots
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 135
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 132):
        raise ValueError("mixing-minimum-resume plan scope or policy changed")
    sources, archives = plan["source_files"], plan["archive_files"]
    if any(archives.get(name) != value for name, value in prior_archives.items()):
        raise ValueError("inherited PR271 archive pin changed")
    snapshot_names = set(snapshots)
    if any(sources.get(name) != value for name, value in prior_sources.items()
            if name not in snapshot_names):
        raise ValueError("inherited PR271 source pin changed")
    if any(prior_sources.get(name) != snapshot["sha256"] for name, snapshot in snapshots.items()):
        raise ValueError("pre-edit mixing-minimum source snapshot differs from PR271 plan")
    if any(sources.get(name) != _sha(ROOT / name) for name in snapshots):
        raise ValueError("current shared runner source differs from successor plan")
    for name, snapshot in snapshots.items():
        if (archives.get(snapshot["archive_path"]) != snapshot["sha256"]
                or _sha(ROOT / snapshot["archive_path"]) != snapshot["sha256"]):
            raise ValueError(f"pre-edit shared source snapshot changed: {name}")
    for name, value in {**sources, **archives}.items():
        target = ROOT / name
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()) or _sha(target) != value:
            raise ValueError(f"mixing-minimum-resume source/archive pin mismatch: {name}")
    if (plan.get("base_archive") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or plan.get("base_archive_sha256") != BASE_GZIP_SHA
            or plan.get("base_run") != BASE_RUN.relative_to(ROOT).as_posix()
            or plan.get("base_run_sha256") != BASE_RUN_SHA
            or plan.get("base_resource") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or plan.get("base_resource_sha256") != BASE_RESOURCE_SHA
            or plan.get("base_archive_manifest") != PREDECESSOR_ARCHIVE_MANIFEST.relative_to(ROOT).as_posix()
            or plan.get("base_archive_manifest_sha256") != PREDECESSOR_ARCHIVE_SHA
            or plan.get("base_child_sha256") != BASE_RAW_SHA):
        raise ValueError("mixing-minimum-resume predecessor receipt pins changed")
    return plan


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    """Admit PR271's selected, closed endpoint from its saved receipt only."""
    if (_sha(PREDECESSOR_PLAN) != PREDECESSOR_PLAN_SHA
            or _sha(PREDECESSOR_ARCHIVE_MANIFEST) != PREDECESSOR_ARCHIVE_SHA
            or _sha(BASE_RUN) != BASE_RUN_SHA or _sha(BASE_RESOURCE) != BASE_RESOURCE_SHA):
        raise ValueError("PR271 plan/run/resource receipt digests changed")
    prior = json.loads(PREDECESSOR_PLAN.read_text())
    manifest = json.loads(PREDECESSOR_ARCHIVE_MANIFEST.read_text())
    compressed = BASE_ARCHIVE.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    child_sha = hashlib.sha256(raw_bytes).hexdigest()
    if (child_sha != BASE_RAW_SHA or _sha(BASE_ARCHIVE) != BASE_GZIP_SHA
            or manifest.get("raw_sha256") != BASE_RAW_SHA
            or manifest.get("gzip_sha256") != BASE_GZIP_SHA
            or manifest.get("run_sha256") != BASE_RUN_SHA
            or manifest.get("resource_sha256") != BASE_RESOURCE_SHA
            or manifest.get("guarded_launch_count") != 1):
        raise ValueError("PR271 lossless archive manifest does not close")
    raw = json.loads(raw_bytes)
    parent = json.loads(BASE_RUN.read_text())
    resource = json.loads(BASE_RESOURCE.read_text())
    source_receipt = {**prior["source_files"], **prior["archive_files"],
        PREDECESSOR_PLAN.relative_to(ROOT).as_posix(): PREDECESSOR_PLAN_SHA}
    if (manifest.get("raw_path") != "graphify-out/fv-root-cause-20260919/tangent_mixing_minimum_20261010_attempt1/step.json"
            or manifest.get("gzip_path") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or manifest.get("raw_bytes") != len(raw_bytes)
            or manifest.get("gzip_bytes") != len(compressed)
            or manifest.get("run_path") != BASE_RUN.relative_to(ROOT).as_posix()
            or manifest.get("resource_path") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != "tangent_mixing_minimum_accepted"
            or parent.get("child_read_error") is not None or parent.get("child_sha256") != child_sha
            or parent.get("resource") != resource
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != 300.0 or resource.get("rss_limit_bytes") != 1024**3
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > 300.0
            or resource.get("sampled_peak_rss_bytes", 1024**3) >= 1024**3
            or raw.get("plan_sha256") != PREDECESSOR_PLAN_SHA
            or raw.get("plan_hashes", {}).get("source_files") != prior["source_files"]
            or raw.get("plan_hashes", {}).get("archive_files") != prior["archive_files"]
            or raw.get("source_before") != source_receipt or raw.get("source_after") != source_receipt
            or raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "tangent_mixing_minimum_accepted"
            or raw.get("base_control_sha256") != mixing.BASE_CONTROL_SHA
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or raw.get("candidate_committed") is not True
            or raw.get("active_candidate_committed") is not False
            or raw.get("accepted_iterations") != 1 or raw.get("optimizer_steps_applied") != 1
            or raw.get("hvp_calls_started") != raw.get("hvp_calls_completed")
            or raw.get("hvp_calls_completed") != 2
            or raw.get("jacobian_rows_started") != 0 or raw.get("jacobian_rows_completed") != 0
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True or raw.get("deadline_passed") is not True
            or raw.get("last_confirmed_control_sha256") != BASE_CONTROL_SHA
            or raw.get("last_confirmed_theta") != BASE_THETA
            or raw.get("last_confirmed_iterations") != 1
            or raw.get("current_theta") != BASE_THETA
            or not geometry.model._fixed_input(raw.get("input_after", {}),
                raw.get("input_before", {}), BASE_CONTROL_SHA)):
        raise ValueError("PR271 endpoint/input/resource/source closure is invalid")
    iterations = raw.get("iterations", [])
    if len(iterations) != 1:
        raise ValueError("PR271 receipt must contain its one accepted comparison step")
    iteration = iterations[0]
    if (iteration.get("accepted") is not True
            or iteration.get("base_control_sha256") != raw.get("base_control_sha256")
            or iteration.get("theta") != raw.get("initial_theta")
            or iteration.get("committed_control") != raw.get("current_control")
            or iteration.get("committed_theta") != BASE_THETA
            or iteration.get("candidate_mixing_minimum") is not True):
        raise ValueError("PR271 accepted iteration does not close at the endpoint")
    accepted_trials = [trial for trial in iteration.get("trials", [])
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
            or final_repeat.get("mixing_minimum_valid") is not True
            or final_repeat.get("minimum_theta_matches_proposal") is not True
            or not all(final_repeat.get(key) is True for key in (
                "side_gradients_finite", "native_objective_matches_proposal",
                "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
                "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
                "fixed_input_unchanged", "runtime_unchanged", "deadline_passed"))):
        raise ValueError("PR271 terminal accepted trial/final closure mismatch")
    control = torch.as_tensor(final_repeat["control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_CONTROL_SHA:
        raise ValueError("PR271 terminal control digest is invalid")
    return {"raw": raw, "control": control, "theta": BASE_THETA,
        "objective": BASE_OBJECTIVE, "accepted": final_repeat, "child_sha256": child_sha}


def _resume_direction_model_factory(context: dict[str, Any]):
    model_factory = mixing._working_model(context)

    def build(control: Tensor, reference_theta: float, gminus: Tensor, gplus: Tensor,
              normal: Tensor, chart: Tensor, pivot: int) -> list[dict[str, Any]]:
        specs = model_factory(control, reference_theta, gminus, gplus, normal, chart, pivot)
        current = context["record"].get("base_mixing_minimum")
        if isinstance(current, dict) and specs:
            point = {"index": len(context["record"].get("mixing_minimum_point_history", [])),
                "base_control_sha256": tangent._tensor_sha(control),
                "reference_theta": reference_theta,
                "working_theta_star": current.get("working_theta_star"),
                "F_squared": current.get("F_squared"), "Psi": current.get("Psi"),
                "optimizer_step": False}
            history = context["record"].setdefault("mixing_minimum_point_history", [])
            history.append(point)
            context["record"].setdefault("initial_working_mixing_minimum", point.copy())
            context["record"]["current_working_mixing_minimum"] = point.copy()
        return specs
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
        default=EVIDENCE / "tangent_mixing_minimum_resume_20261010_attempt1/step.json")
    parser.add_argument("--resource", type=Path,
        default=EVIDENCE / "tangent_mixing_minimum_resume_20261010_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path,
        default=EVIDENCE / "tangent_mixing_minimum_resume_20261010_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
