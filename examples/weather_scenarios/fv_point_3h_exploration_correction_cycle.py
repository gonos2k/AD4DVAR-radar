"""One PR262 exploration step followed by bounded full-space corrections."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as guard_policy
from examples.weather_scenarios import fv_point_3h_qy32_tangent_step as tangent

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "EXPLORATION_CORRECTION_CYCLE_PLAN_20261008.json"
SAMPLE_PLAN = EVIDENCE / "SAMPLE_COMMON_COST_SEARCH_PLAN_20261008.json"
SAMPLE_PLAN_SHA = "617a81486352591f3e17068be78fc4047db5bd4e07333b84dba48297ba9dcc3e"
SAMPLE_DIR = EVIDENCE / "sample_common_search_20261008_attempt1"
SAMPLE_STEP = SAMPLE_DIR / "step.json"
SAMPLE_RUN = SAMPLE_DIR / "step.run.json"
SAMPLE_RESOURCE = SAMPLE_DIR / "step.resource.json"
SAMPLE_STEP_SHA = "7df18a05130cd9e2d12ffde96657bc9217ac42b130095b72212bd317e21c028e"
SAMPLE_RUN_SHA = "e4d0ac82e2a03bdca7642bca98851060ca1a58fcec694701ec64596761855c1d"
SAMPLE_RESOURCE_SHA = "98e65b66e26d4d37c563bea0d33580bb9780c1206bcd41992121905c3b0e63cb"
PR260_PLAN = EVIDENCE / "MODEL_GUIDED_FULL_SPACE_PLAN_20261008.json"
PR260_PLAN_SHA = "9fc5e046ca484737ebd040469990f10cc918fcd9dc245d0762e65f297151440c"
PR260_DIR = EVIDENCE / "model_guided_resume_20261008_attempt1"
PR260_STEP = PR260_DIR / "step.json"
PR260_RUN = PR260_DIR / "step.run.json"
PR260_RESOURCE = PR260_DIR / "step.resource.json"
PR260_STEP_SHA = "66aa57c3056220a7057a5b3c3cca2a09c06d44ed34279b055174e6930ab1838e"
PR260_RUN_SHA = "604d447997c76f358fda4c8f0c895c563c1f2772171de511cf1015d19f2cf6bd"
PR260_RESOURCE_SHA = "67b7294e85b19022b205be03f8cb2f4198be5cd8d50c21935d92354d682344e5"
BASE_CONTROL_SHA = "26e830dc1cb1bce1240b09f0f88ddf616cb279414fa1eb46d01b94e0f52dde14"
CYCLE_REFERENCE_SHA = "2cdccade81a8cf218dd7cad97adf71f6675f8445481dd8f6394ec337ea8e8007"
SNAPSHOT_MANIFEST = EVIDENCE / "cycle_source_20261008/manifest.json"
SNAPSHOT_MANIFEST_SHA = "120138b9465748c65d1c08f73b06f58881f3ba5c642f558a14b4b95563cd5d48"
SELF = "examples/weather_scenarios/fv_point_3h_exploration_correction_cycle.py"
TEST = "tests/test_fv_point_3h_exploration_correction_cycle.py"
SNAPSHOT_SOURCES = {
    "examples/weather_scenarios/fv_point_3h_model_guided_continuation.py":
        "graphify-out/fv-root-cause-20260919/cycle_source_20261008/fv_point_3h_model_guided_continuation.py",
    "tests/test_fv_point_3h_model_guided_continuation.py":
        "graphify-out/fv-root-cause-20260919/cycle_source_20261008/test_fv_point_3h_model_guided_continuation.py",
}
MAX_CORRECTIONS = 3
RECOVERY_EPS = 128 * math.ulp(1.0)


def _guided() -> ModuleType:
    """Load the shared numerical runner without a cyclic static import."""
    return importlib.import_module("examples.weather_scenarios.fv_point_3h_model_guided_continuation")


def policy() -> dict[str, Any]:
    return {**_guided().policy(), "cycle_max_corrections": MAX_CORRECTIONS,
            "recovery_comparator": "resolved_J_and_Phi_below_PR260_reference"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    path = path.resolve()
    if (path != PLAN.resolve() or path.is_symlink()
            or not path.is_relative_to(ROOT.resolve()) or _sha(path) != digest):
        raise ValueError("correction-cycle plan identity mismatch")
    plan = json.loads(path.read_text())
    inherited = json.loads(SAMPLE_PLAN.read_text())
    manifest = json.loads(SNAPSHOT_MANIFEST.read_text())
    snapshots = plan.get("producer_source_snapshots")
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    required_archives = {SAMPLE_PLAN.relative_to(ROOT).as_posix(),
        SAMPLE_STEP.relative_to(ROOT).as_posix(), SAMPLE_RUN.relative_to(ROOT).as_posix(),
        SAMPLE_RESOURCE.relative_to(ROOT).as_posix(),
        SNAPSHOT_MANIFEST.relative_to(ROOT).as_posix(),
        PR260_PLAN.relative_to(ROOT).as_posix(), PR260_STEP.relative_to(ROOT).as_posix(),
        PR260_RUN.relative_to(ROOT).as_posix(), PR260_RESOURCE.relative_to(ROOT).as_posix(),
        *[item["archive_path"] for item in manifest["snapshots"].values()]}
    expected_sources = set(inherited["source_files"]) | {SELF, TEST}
    expected_archives = set(inherited["archive_files"]) | required_archives
    if (plan.get("experiment_kind") != "exploration_correction_cycle"
            or plan.get("producing_plan") != SAMPLE_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != SAMPLE_PLAN_SHA
            or plan.get("policy") != policy()
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("cycle_reference_control_sha256") != CYCLE_REFERENCE_SHA
            or plan.get("base_step") != SAMPLE_STEP.relative_to(ROOT).as_posix()
            or plan.get("base_run") != SAMPLE_RUN.relative_to(ROOT).as_posix()
            or plan.get("base_resource") != SAMPLE_RESOURCE.relative_to(ROOT).as_posix()
            or not isinstance(sources, dict) or not isinstance(archives, dict)
            or not isinstance(snapshots, dict)
            or set(sources) != expected_sources or set(archives) != expected_archives
            or snapshots != manifest.get("snapshots")
            or _sha(SAMPLE_PLAN) != SAMPLE_PLAN_SHA
            or _sha(SNAPSHOT_MANIFEST) != SNAPSHOT_MANIFEST_SHA):
        raise ValueError("correction-cycle scope or inherited lineage changed")
    for name, value in inherited["source_files"].items():
        if name in snapshots:
            snapshot = snapshots[name]
            if (snapshot.get("sha256") != value
                    or archives.get(snapshot.get("archive_path")) != value):
                raise ValueError("correction-cycle snapshot does not preserve the PR262 source pin")
        elif sources.get(name) != value:
            raise ValueError(f"correction-cycle changed inherited sample source: {name}")
    if any(archives.get(name) != value for name, value in inherited["archive_files"].items()):
        raise ValueError("correction-cycle changed an inherited sample archive pin")
    for name in (SELF, TEST):
        if sources.get(name) != _sha(ROOT / name):
            raise ValueError("correction-cycle plan must pin its source and tests")
    if (set(snapshots) != set(SNAPSHOT_SOURCES)
            or any(not (ROOT / item["archive_path"]).is_file()
                   or _sha(ROOT / item["archive_path"]) != item["sha256"]
                   for item in snapshots.values())):
        raise ValueError("correction-cycle snapshots do not preserve changed inherited files")
    for name, value in {**sources, **archives}.items():
        tangent._pinned_path(name, value)
    if _sha(SAMPLE_RUN) != SAMPLE_RUN_SHA:
        raise ValueError("accepted sample-common parent receipt changed")
    for name in required_archives:
        if name not in inherited["archive_files"]:
            path_ = ROOT / name
            expected = plan.get("archive_files", {}).get(name)
            if not expected or _sha(path_) != expected:
                raise ValueError("correction-cycle receipt/archive hash changed")
    return plan


def _load_base(plan: dict[str, Any]) -> dict[str, Any]:
    inherited = json.loads(SAMPLE_PLAN.read_text())
    for path, key, known_sha in ((SAMPLE_STEP, "base_step_sha256", SAMPLE_STEP_SHA),
                      (SAMPLE_RUN, "base_run_sha256", SAMPLE_RUN_SHA),
                      (SAMPLE_RESOURCE, "base_resource_sha256", SAMPLE_RESOURCE_SHA),
                      (PR260_STEP, "cycle_reference_step_sha256", PR260_STEP_SHA),
                      (PR260_RUN, "cycle_reference_run_sha256", PR260_RUN_SHA),
                      (PR260_RESOURCE, "cycle_reference_resource_sha256", PR260_RESOURCE_SHA)):
        digest = plan.get(key)
        if digest != known_sha or _sha(path) != digest or plan["archive_files"].get(
                path.relative_to(ROOT).as_posix()) != digest:
            raise ValueError("correction-cycle accepted receipt hash differs from plan")
    sample = json.loads(SAMPLE_STEP.read_text())
    sample_parent = json.loads(SAMPLE_RUN.read_text())
    sample_resource = json.loads(SAMPLE_RESOURCE.read_text())
    historical = json.loads(PR260_STEP.read_text())
    historical_parent = json.loads(PR260_RUN.read_text())
    historical_resource = json.loads(PR260_RESOURCE.read_text())
    historical_plan = json.loads(PR260_PLAN.read_text())
    expected_historical_sources = {**historical_plan["source_files"],
        **historical_plan["archive_files"],
        PR260_PLAN.relative_to(ROOT).as_posix(): PR260_PLAN_SHA}
    trial_list = [item for item in sample.get("trials", []) if item.get("status") == "accepted"]
    expected_sample_sources = {**inherited["source_files"], **inherited["archive_files"],
        SAMPLE_PLAN.relative_to(ROOT).as_posix(): SAMPLE_PLAN_SHA}
    archived_sample_sources = sample.get("source_before")
    sample_input_fixed = tangent._check_fixed_input(sample.get("input_after", {}),
        sample.get("input_before", {}), BASE_CONTROL_SHA)
    if (sample.get("phase") != "finished" or sample.get("execution_status") != "completed"
            or sample.get("plan_sha256") != SAMPLE_PLAN_SHA
            or sample.get("candidate_committed") is not True
            or sample.get("optimizer_steps_applied") != 1
            or sample.get("accepted_control_sha256") != BASE_CONTROL_SHA
            or sample.get("parameters_sha256") != _guided().PARAMETERS_SHA
            or len(trial_list) != 1
            or trial_list[0].get("control_sha256") != BASE_CONTROL_SHA
            or trial_list[0].get("armijo_passed") is not True
            or trial_list[0].get("strict_point_passed") is not True
            or trial_list[0].get("branch", {}).get("status") != "passed_strict_branch"
            or trial_list[0].get("objective") != sample.get("accepted_objective")
            or trial_list[0].get("phi") != sample.get("accepted_phi_diagnostic")
            or trial_list[0].get("gradient") != sample.get("accepted_gradient")
            or trial_list[0].get("branch", {}).get("signature_sha256")
            != sample.get("accepted_branch", {}).get("signature_sha256")
            or sample.get("source_unchanged") is not True
            or sample.get("source_before") != sample.get("source_after")
            or archived_sample_sources != expected_sample_sources
            or not sample_input_fixed
            or sample.get("runtime_after") != sample.get("runtime")
            or not isinstance(sample_parent.get("resource"), dict)
            or sample_parent.get("execution_status") != "completed"
            or sample_parent.get("child_read_error") is not None
            or sample_parent.get("child_sha256") != SAMPLE_STEP_SHA
            or sample_parent.get("numerical_status") != sample.get("numerical_status")
            or sample_parent.get("resource") != sample_resource
            or guard_policy.execution_status(sample_resource) != "completed"):
        raise ValueError("PR262 accepted sample endpoint or parent/resource closure is invalid")
    if (historical.get("phase") != "finished" or historical.get("execution_status") != "completed"
            or historical.get("plan_sha256") != PR260_PLAN_SHA
            or historical.get("candidate_committed") is not True
            or historical.get("optimizer_steps_applied") != 3
            or historical.get("current_control_sha256") != CYCLE_REFERENCE_SHA
            or historical.get("parameters_sha256") != _guided().PARAMETERS_SHA
            or historical.get("source_unchanged") is not True
            or historical.get("fixed_input_unchanged") is not True
            or historical.get("source_before") != historical.get("source_after")
            or historical.get("source_before") != expected_historical_sources
            or historical.get("runtime_after") != historical.get("runtime")
            or historical_parent.get("execution_status") != "completed"
            or historical_parent.get("child_sha256") != PR260_STEP_SHA
            or historical_parent.get("resource") != historical_resource
            or guard_policy.execution_status(historical_resource) != "completed"):
        raise ValueError("PR260 correction-cycle reference receipts are not closed")
    state = historical.get("current_state", {})
    if (sample.get("base_control_sha256") != CYCLE_REFERENCE_SHA
            or sample.get("base_objective") != state.get("objective")
            or sample.get("base_phi_diagnostic") != state.get("phi")
            or sample.get("base_gradient") != state.get("gradient")
            or sample.get("input_before") != historical.get("input_after")
            or sample.get("runtime") != historical.get("runtime")):
        raise ValueError("PR262 baseline does not match the PR260 pre-exploration state")
    branch = sample["accepted_branch"]
    state = {"objective": sample["accepted_objective"],
        "phi": sample["accepted_phi_diagnostic"], "gradient": sample["accepted_gradient"],
        "gradient_inf": max(abs(value) for value in sample["accepted_gradient"]),
        "branch": {**branch, "signature_sha256": branch.get("signature_sha256")},
        "branch_partition": branch.get("partition", {})}
    return {"accepted_control": sample["accepted_control"],
        "accepted_control_sha256": BASE_CONTROL_SHA,
        "accepted_objective": state["objective"], "accepted_phi": state["phi"],
        "accepted_gradient": state["gradient"], "accepted_gradient_inf": state["gradient_inf"],
        "accepted_branch": state["branch"], "accepted_branch_partition": state["branch_partition"],
        "current_state": state, "parameters_sha256": sample["parameters_sha256"],
        "input_before": sample["input_before"], "input_after": sample["input_after"],
        "runtime": sample["runtime"], "cycle_reference": historical["current_state"],
        "cycle_reference_control_sha256": CYCLE_REFERENCE_SHA,
        "cycle_reference_step_sha256": PR260_STEP_SHA}


def recovery_assessment(reference: dict[str, Any], start: dict[str, Any],
                        current: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    resolved = True
    recovered = True
    for metric in ("objective", "phi"):
        start_value, current_value = float(start[metric]), float(current[metric])
        reference_value = float(reference[metric])
        budget = 128 * math.ulp(1.0) * max(abs(reference_value), abs(current_value),
                                             math.ulp(0.0))
        delta = start_value - current_value
        below_reference = current_value < reference_value - budget
        metric_resolved = delta > 128 * math.ulp(1.0) * max(
            abs(start_value), abs(current_value), math.ulp(0.0))
        metrics[metric] = {"correction_start": start_value, "current": current_value,
            "delta_from_correction_start": current_value - start_value,
            "preexploration_reference": reference_value,
            "delta_from_preexploration_reference": current_value - reference_value,
            "reference_resolution_budget": budget, "below_preexploration_reference": below_reference,
            "correction_decrease_resolved": metric_resolved}
        resolved = resolved and metric_resolved
        recovered = recovered and below_reference
    return {"status": "recovered" if recovered else "unrecovered",
        "recovered": recovered, "resolved_correction_decrease": resolved,
        "metrics": metrics}


def run(plan_sha: str, output: Path, resource: Path, log: Path) -> dict[str, Any]:
    return _guided().run(PLAN, plan_sha, output, resource, log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan_sha256, args.output, args.resource, args.log),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
