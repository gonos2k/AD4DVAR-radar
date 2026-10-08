"""Resume the PR263 accepted point with the existing model-guided kernel."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "CORRECTION_RESUME_PLAN_20261008.json"
PRODUCER_PLAN = EVIDENCE / "EXPLORATION_CORRECTION_CYCLE_PLAN_20261008.json"
PRODUCER_PLAN_SHA = "a91588d2281325c95518c79e677043823b18ed52e51ee0557d55c6e3e51dec12"
BASE_DIR = EVIDENCE / "exploration_correction_cycle_20261008_attempt1"
BASE_STEP = BASE_DIR / "step.json"
BASE_RUN = BASE_DIR / "step.run.json"
BASE_RESOURCE = BASE_DIR / "step.resource.json"
BASE_STEP_SHA = "275107930a951925726f91a68300123ddeebcf302283a26e3d72aca0c5274fae"
BASE_RUN_SHA = "1bc20c78729c631d44f2eead29b795b92ece15318bb5b5a5e9203dae69300372"
BASE_RESOURCE_SHA = "ccdf33208489e58c96cd0aedd1ad56eaca530cd0b80e86fe30911d1936406e4f"
BASE_CONTROL_SHA = "6c0fe4858631513f665f91c77006d4641e564862d8141111c0030c91d2bb74bd"
SOURCE_MANIFEST = EVIDENCE / "correction_resume_source_20261008/manifest.json"
SOURCE_MANIFEST_SHA = "65443c537bf390e174550d91ee9c22028d8124411522ad8f49e8fdeb3ae128e8"
SELF = "examples/weather_scenarios/fv_point_3h_correction_resume.py"
TEST = "tests/test_fv_point_3h_correction_resume.py"
GUIDED = "examples/weather_scenarios/fv_point_3h_model_guided_continuation.py"
def _guided() -> ModuleType:
    return importlib.import_module("examples.weather_scenarios.fv_point_3h_model_guided_continuation")


def policy() -> dict[str, Any]:
    return _guided().policy()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    path = path.resolve()
    if (path != PLAN.resolve() or path.is_symlink() or not path.is_relative_to(ROOT.resolve())
            or _sha(path) != digest):
        raise ValueError("correction-resume plan identity mismatch")
    plan = json.loads(path.read_text())
    inherited = json.loads(PRODUCER_PLAN.read_text())
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    snapshots = plan.get("producer_source_snapshots")
    changed = {GUIDED}
    required_receipts = {PRODUCER_PLAN.relative_to(ROOT).as_posix(),
        BASE_STEP.relative_to(ROOT).as_posix(), BASE_RUN.relative_to(ROOT).as_posix(),
        BASE_RESOURCE.relative_to(ROOT).as_posix(), SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        *(item["archive_path"] for item in manifest["snapshots"].values())}
    expected_sources = set(inherited["source_files"]) | {SELF, TEST}
    expected_archives = set(inherited["archive_files"]) | required_receipts
    if (plan.get("experiment_kind") != "model_guided_correction_resume"
            or plan.get("policy") != policy()
            or plan.get("producing_plan") != PRODUCER_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != PRODUCER_PLAN_SHA
            or plan.get("base_step") != BASE_STEP.relative_to(ROOT).as_posix()
            or plan.get("base_run") != BASE_RUN.relative_to(ROOT).as_posix()
            or plan.get("base_resource") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or not isinstance(sources, dict) or not isinstance(archives, dict)
            or snapshots != manifest.get("snapshots") or set(snapshots) != changed
            or set(sources) != expected_sources or set(archives) != expected_archives
            or _sha(PRODUCER_PLAN) != PRODUCER_PLAN_SHA
            or _sha(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA):
        raise ValueError("correction-resume scope, policy, or inherited lineage changed")
    for name, value in inherited["source_files"].items():
        if name in snapshots:
            snap = snapshots[name]
            if (snap.get("sha256") != value or archives.get(snap.get("archive_path")) != value):
                raise ValueError("resume snapshot does not preserve the PR263 producer bytes")
        elif sources.get(name) != value:
            raise ValueError(f"resume changed inherited producer source: {name}")
    if any(archives.get(name) != value for name, value in inherited["archive_files"].items()):
        raise ValueError("resume changed inherited PR263 archive pins")
    if (archives.get(SOURCE_MANIFEST.relative_to(ROOT).as_posix()) != SOURCE_MANIFEST_SHA
            or any(sources.get(name) != _sha(ROOT / name) for name in (GUIDED, SELF, TEST))
            or any(not (ROOT / item["archive_path"]).is_file()
                   or _sha(ROOT / item["archive_path"]) != item["sha256"]
                   for item in snapshots.values())):
        raise ValueError("resume current sources or producer snapshot do not match their pins")
    for name, value in {**sources, **archives}.items():
        _guided().tangent._pinned_path(name, value)
    for artifact, known in ((BASE_STEP, BASE_STEP_SHA), (BASE_RUN, BASE_RUN_SHA),
                            (BASE_RESOURCE, BASE_RESOURCE_SHA)):
        name = artifact.relative_to(ROOT).as_posix()
        if plan.get({BASE_STEP: "base_step_sha256", BASE_RUN: "base_run_sha256",
                     BASE_RESOURCE: "base_resource_sha256"}[artifact]) != known:
            raise ValueError("resume receipt hash differs from the accepted PR263 pin")
        if archives.get(name) != known:
            raise ValueError("resume plan omits an accepted PR263 receipt")
    return plan


def _load_base(plan: dict[str, Any]) -> dict[str, Any]:
    guided = _guided()
    tangent = guided.tangent
    guard = guided.guard_policy
    for path, known in ((BASE_STEP, BASE_STEP_SHA), (BASE_RUN, BASE_RUN_SHA),
                        (BASE_RESOURCE, BASE_RESOURCE_SHA)):
        if _sha(path) != known or plan["archive_files"].get(path.relative_to(ROOT).as_posix()) != known:
            raise ValueError("accepted PR263 receipt bytes differ from the plan")
    raw, parent, resource = (json.loads(BASE_STEP.read_text()), json.loads(BASE_RUN.read_text()),
                             json.loads(BASE_RESOURCE.read_text()))
    iterations = raw.get("iterations")
    state = raw.get("current_state")
    if not isinstance(iterations, list) or len(iterations) != 2 or not isinstance(state, dict):
        raise ValueError("PR263 endpoint must close exactly two committed corrections")
    expected_sources = {**json.loads(PRODUCER_PLAN.read_text())["source_files"],
        **json.loads(PRODUCER_PLAN.read_text())["archive_files"],
        PRODUCER_PLAN.relative_to(ROOT).as_posix(): PRODUCER_PLAN_SHA}
    accepted = [trial for trial in iterations[-1].get("trials", [])
                if trial.get("status") == "accepted"]
    trial = accepted[0] if len(accepted) == 1 else {}
    if (raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "cycle_recovered"
            or raw.get("plan_sha256") != PRODUCER_PLAN_SHA
            or raw.get("base_control_sha256") != "26e830dc1cb1bce1240b09f0f88ddf616cb279414fa1eb46d01b94e0f52dde14"
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or raw.get("optimizer_steps_applied") != 2 or raw.get("hvp_calls") != 2
            or raw.get("hvp_calls_completed") != 2 or raw.get("candidate_committed") is not True
            or raw.get("active_candidate_committed") is not False
            or raw.get("parameters_sha256") != guided.PARAMETERS_SHA
            or raw.get("source_unchanged") is not True or raw.get("source_before") != raw.get("source_after")
            or raw.get("source_before") != expected_sources
            or raw.get("fixed_input_unchanged") is not True
            or not tangent._check_fixed_input(raw.get("input_after", {}),
                raw.get("input_before", {}), BASE_CONTROL_SHA)
            or raw.get("runtime_after") != raw.get("runtime")
            or len(accepted) != 1 or any(item.get("committed") is not True
                or item.get("status") != "accepted" for item in iterations)
            or [item.get("index") for item in iterations] != [0, 1]
            or any(item.get("hvp_calls_started") != 1 or item.get("hvp_calls_completed") != 1
                   for item in iterations)
            or iterations[0].get("base_control_sha256") != raw.get("base_control_sha256")
            or iterations[1].get("base_control_sha256") != iterations[0].get("accepted_control_sha256")
            or iterations[-1].get("accepted_control_sha256") != BASE_CONTROL_SHA
            or iterations[-1].get("accepted_control") != raw.get("current_control")
            or trial.get("control_sha256") != BASE_CONTROL_SHA
            or trial.get("J_armijo_passed") is not True or trial.get("Phi_armijo_passed") is not True
            or trial.get("strict_point_passed") is not True
            or trial.get("branch", {}).get("status") != "passed_strict_branch"
            or trial.get("objective") != state.get("objective")
            or trial.get("phi") != state.get("phi")
            or trial.get("gradient") != state.get("gradient")
            or trial.get("branch", {}).get("signature_sha256") != state.get("branch", {}).get("signature_sha256")
            or state.get("control_sha256") != BASE_CONTROL_SHA
            or state.get("branch", {}).get("status") != "passed_strict_branch"
            or state.get("branch_partition", {}).get("full_sha256")
                != state.get("branch", {}).get("signature_sha256")
            or iterations[-1].get("accepted_objective") != state.get("objective")
            or iterations[-1].get("accepted_phi") != state.get("phi")
            or iterations[-1].get("accepted_gradient") != state.get("gradient")
            or iterations[-1].get("accepted_gradient_inf") != state.get("gradient_inf")
            or parent.get("execution_status") != "completed" or parent.get("child_read_error") is not None
            or parent.get("child_sha256") != BASE_STEP_SHA
            or parent.get("numerical_status") != raw.get("numerical_status")
            or parent.get("resource") != resource or guard.execution_status(resource) != "completed"):
        raise ValueError("PR263 accepted endpoint or closed receipts are invalid")
    digest = tangent._tensor_sha(torch.tensor(raw["current_control"], dtype=torch.float64))
    if digest != BASE_CONTROL_SHA:
        raise ValueError("PR263 final control values do not match their hash")
    return {"accepted_control": raw["current_control"], "accepted_control_sha256": BASE_CONTROL_SHA,
        "accepted_objective": state["objective"], "accepted_phi": state["phi"],
        "accepted_gradient": state["gradient"], "accepted_gradient_inf": state["gradient_inf"],
        "accepted_branch": state["branch"], "accepted_branch_partition": state["branch_partition"],
        "current_state": state, "parameters_sha256": raw["parameters_sha256"],
        "input_before": raw["input_before"], "input_after": raw["input_after"],
        "runtime": raw["runtime"], "resume_plan_sha256": PRODUCER_PLAN_SHA,
        "resume_step_sha256": BASE_STEP_SHA}


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    return _guided().run(plan_path, plan_sha, output, resource, log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "correction_resume_attempt1/step.json")
    parser.add_argument("--resource", type=Path, default=EVIDENCE / "correction_resume_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path, default=EVIDENCE / "correction_resume_attempt1/step.log")
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.plan_sha256, args.output, args.resource, args.log), indent=2))


if __name__ == "__main__":
    main()
