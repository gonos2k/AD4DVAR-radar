"""Pinned adapter for another bounded continuation from the accepted 90fc point.

This module validates historical receipts and delegates every numerical step to
``fv_point_3h_dual_merit_continuation``. It never reuses a historical direction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from examples.weather_scenarios import fv_point_3h_accepted_curvature as curvature
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as search
from examples.weather_scenarios import fv_point_3h_current_newton_step as diagnostics
from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation
from examples.weather_scenarios import fv_point_3h_schur_newton_step as block_step
from examples.weather_scenarios import fv_point_3h_hvp_newton_step as live_step

ROOT = curvature.ROOT
EVIDENCE = curvature.EVIDENCE
BASE = EVIDENCE / "e0b_repeat_dual_20261006_attempt1/step.json"
PLAN = EVIDENCE / "90FC_DUAL_MERIT_FOLLOWUP_PLAN_20261007.json"
SELF = "examples/weather_scenarios/fv_point_3h_90fc_followup.py"
TEST = "tests/test_fv_point_3h_90fc_followup.py"
PRIOR_PLAN = EVIDENCE / "E0B_DUAL_MERIT_CONTINUATION_PLAN_20261006.json"
CLASSIFICATION_FIX = EVIDENCE / "REPEAT_DUAL_CLASSIFICATION_FIX_20261006.json"
CONTROL_SHA = "90fc45552d068ae4f1b83eb54ab095362f223dc10b884695ccdbe1e63c9825fc"
LEGACY_RAW_SHA = "6be9cf6e67ef644888334b4cf34fb628483e43655186a1c0a9a0be3e4cc2c003"
LEGACY_PARENT_SHA = "e70abf0987db430670d426a18537012cc7118e34d0d75939458370c665c98704"
LEGACY_RESOURCE_SHA = "de454a9639642a2cffd3d9dc46ed8ffd1d938ccfe746b59ea0d7d6f94852a19c"
LEGACY_PRODUCER_SHA = "b8c02f5cc35032de5aacc79a926a09ee9fe2c669fdba5f0acd80f7676bd5424d"
LEGACY_TEST_SHA = "1608a03b6abc8062c541c9c196f105ae7fb4a10b5fac832f30f5313656855515"
PRIOR_PLAN_SHA = "ee75ca45b03495b055023a984457c4aab73eb76c646c1c17bf8760d2e53f80d9"
CLASSIFICATION_FIX_SHA = "73ef85922aa33e5c91784b0d9cc74634dc86094fd1578fbcc792fe23b7d8a5a1"
HESSIAN_SHA = "3058fec5ebf746c500a02b8712983459f8735e0ad8e762b217474e31379f2255"
CHECKPOINT_SHA = "a249f0947440dada62464c2292e99f03cb4aa44d00ce8705fb8e4b0510d966bd"
HESSIAN_PARENT_SHA = "26261c610df92552a655e8066ae4af384f22693f8754426a6eeb95e5713efb14"
HESSIAN_RESOURCE_SHA = "139c8545f0538625b808e42f4060fbec4a38eacab41fa2ef8a072a53af9ece0a"
HESSIAN = EVIDENCE / "f82c_fresh_curvature_20261005/attempt_1/audit.json"
CHECKPOINT = EVIDENCE / "f82c_fresh_curvature_20261005/checkpoint/hessian_checkpoint.json"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_pins(plan_path: Path, plan_sha: str) -> dict[str, Any]:
    if (not plan_path.resolve().is_relative_to(ROOT.resolve())
            or plan_path.is_symlink() or curvature.sha(plan_path) != plan_sha):
        raise ValueError("follow-up plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    if (plan.get("policy") != continuation.policy_dict()
            or plan.get("base_control_sha256") != CONTROL_SHA
            or not isinstance(plan.get("source_files"), dict)
            or not isinstance(plan.get("archive_files"), dict)):
        raise ValueError("follow-up plan must pin 90fc and the unchanged continuation policy")
    inherited_sources = set(json.loads(PRIOR_PLAN.read_text())["source_files"])
    required_sources = inherited_sources | {SELF, TEST,
        "examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py",
        "tests/test_fv_point_3h_dual_merit_continuation.py",
        "examples/weather_scenarios/fv_point_3h_dual_merit_step.py",
        "examples/weather_scenarios/fv_point_3h_hvp_newton_step.py",
        "examples/weather_scenarios/fv_point_3h_schur_newton_step.py",
        "examples/weather_scenarios/fv_point_3h_current_newton_step.py",
        "src/advar/matrix_free.py", _relative(curvature.R9_ARCHIVE)}
    required_archives = { _relative(BASE), _relative(BASE.with_suffix(".run.json")),
        _relative(BASE.with_suffix(".resource.json")), _relative(PRIOR_PLAN),
        _relative(CLASSIFICATION_FIX), _relative(HESSIAN), _relative(CHECKPOINT),
        _relative(live_step.CURVATURE_PARENT), _relative(live_step.CURVATURE_RESOURCE),
        "graphify-out/fv-root-cause-20260919/e0b_repeat_dual_20261006_attempt1/producer.py",
        "graphify-out/fv-root-cause-20260919/e0b_repeat_dual_20261006_attempt1/producer_test.py"}
    if not required_sources <= set(plan["source_files"]) or not required_archives <= set(plan["archive_files"]):
        raise ValueError("follow-up plan omits current kernel sources or pinned historical receipts")
    if (plan["archive_files"][_relative(PRIOR_PLAN)] != PRIOR_PLAN_SHA
            or plan["archive_files"][_relative(CLASSIFICATION_FIX)] != CLASSIFICATION_FIX_SHA
            or plan["archive_files"][_relative(BASE)] != LEGACY_RAW_SHA
            or plan["archive_files"][_relative(BASE.with_suffix(".run.json"))] != LEGACY_PARENT_SHA
            or plan["archive_files"][_relative(BASE.with_suffix(".resource.json"))] != LEGACY_RESOURCE_SHA
            or plan["archive_files"][_relative(HESSIAN)] != HESSIAN_SHA
            or plan["archive_files"][_relative(CHECKPOINT)] != CHECKPOINT_SHA
            or plan["archive_files"][_relative(live_step.CURVATURE_PARENT)] != HESSIAN_PARENT_SHA
            or plan["archive_files"][_relative(live_step.CURVATURE_RESOURCE)] != HESSIAN_RESOURCE_SHA):
        raise ValueError("follow-up plan changed an immutable historical receipt pin")
    for name, expected in {**plan["source_files"], **plan["archive_files"]}.items():
        path = ROOT / name
        if (not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink()
                or curvature.sha(path) != expected):
            raise ValueError(f"follow-up source/archive pin changed: {name}")
    return plan


def _accepted_endpoint(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the final committed row, rejecting provisional/partial solve data."""
    accepted = [row for row in raw.get("trials", []) if row.get("status") == "accepted"]
    iterations = raw.get("iterations")
    if (raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "budget_refusal"
            or raw.get("optimizer_steps_applied") != 2 or not isinstance(iterations, list)
            or len(iterations) != 2 or len(accepted) != 2
            or any(row.get("status") != "accepted" for row in iterations)):
        raise ValueError("90fc history must contain exactly two completed accepted iterations")
    final = accepted[-1]
    state = raw.get("current_state", {})
    last_iteration = iterations[-1]
    try:
        control = torch.tensor(raw.get("accepted_control"), dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("90fc accepted control is not a numeric FP64 vector") from error
    if (control.shape != (26,) or not bool(torch.isfinite(control).all())
            or curvature.tensor_sha(control) != CONTROL_SHA
            or raw.get("accepted_control_sha256") != CONTROL_SHA
            or final.get("control_sha256") != CONTROL_SHA
            or last_iteration.get("accepted_control_sha256") != CONTROL_SHA
            or last_iteration.get("accepted_control") != raw.get("accepted_control")
            or final.get("control") != raw.get("accepted_control")
            or final.get("objective") != state.get("objective")
            or final.get("phi") != state.get("phi")
            or final.get("gradient") != state.get("gradient")
            or float(torch.tensor(final["gradient"], dtype=torch.float64).abs().max()) != state.get("gradient_inf")
            or final.get("branch") != state.get("branch")
            or final.get("margins") != state.get("margins")
            or final.get("J_armijo_passed") is not True
            or final.get("Phi_armijo_passed") is not True
            or final.get("strict_point_passed") is not True):
        raise ValueError("90fc endpoint does not match the final accepted trial and current_state")
    return final


def load_base(plan_path: Path, plan_sha: str):
    """Return the continuation contract while preserving the original receipt."""
    plan = _require_pins(plan_path, plan_sha)
    prior_plan = json.loads(PRIOR_PLAN.read_text())
    raw = json.loads(BASE.read_text())
    parent = json.loads(BASE.with_suffix(".run.json").read_text())
    resource = json.loads(BASE.with_suffix(".resource.json").read_text())
    fix = json.loads(CLASSIFICATION_FIX.read_text())
    hessian = json.loads(HESSIAN.read_text())
    checkpoint = json.loads(CHECKPOINT.read_text())
    curvature_parent = json.loads(live_step.CURVATURE_PARENT.read_text())
    curvature_resource = json.loads(live_step.CURVATURE_RESOURCE.read_text())
    if (curvature.sha(BASE) != LEGACY_RAW_SHA
            or curvature.sha(BASE.with_suffix(".run.json")) != LEGACY_PARENT_SHA
            or curvature.sha(BASE.with_suffix(".resource.json")) != LEGACY_RESOURCE_SHA
            or curvature.sha(ROOT / "graphify-out/fv-root-cause-20260919/e0b_repeat_dual_20261006_attempt1/producer.py") != LEGACY_PRODUCER_SHA
            or curvature.sha(ROOT / "graphify-out/fv-root-cause-20260919/e0b_repeat_dual_20261006_attempt1/producer_test.py") != LEGACY_TEST_SHA
            or fix.get("historical_parent_execution_status") != "failed"
            or fix.get("child_execution_status") != "completed"
            or fix.get("child_numerical_status") != "budget_refusal"
            or fix.get("exit_code") != 0
            or fix.get("new_classifier_status_on_same_saved_resource") != "completed"
            or fix.get("raw_sha256") != LEGACY_RAW_SHA
            or fix.get("parent_sha256") != LEGACY_PARENT_SHA
            or fix.get("resource_sha256") != LEGACY_RESOURCE_SHA
            or parent.get("execution_status") != "failed"
            or parent.get("child_sha256") != LEGACY_RAW_SHA
            or parent.get("completed_iterations") != 2
            or parent.get("hvp_calls") != 62
            or continuation.execution_status(resource) != "completed"):
        raise ValueError("historical 780-second run does not match its pinned classification-fix record")
    if (raw.get("plan_sha256") != curvature.sha(PRIOR_PLAN)
            or prior_plan.get("accepted_control_sha256") != raw.get("base_control_sha256")
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("runtime") != raw.get("runtime_after")
            or raw.get("accepted_control_sha256") != CONTROL_SHA):
        raise ValueError("90fc raw receipt lost its plan, endpoint, runtime, or source integrity")
    if plan.get("parameters_sha256") != raw.get("parameters_sha256"):
        raise ValueError("follow-up plan parameters differ from the fixed archived input")
    _accepted_endpoint(raw)
    # The partial third solve in current_iteration is preserved in raw, never reused.
    if raw.get("current_iteration", {}).get("status") not in {"budget_refusal", "linear_solve"}:
        raise ValueError("historical partial third solve receipt is absent or inconsistent")
    diagnostics._require_cached_objective_sources(plan, hessian)
    h = hessian.get("curvature", {})
    if (hessian.get("phase") != "finished" or hessian.get("numerical_status") != "curvature_completed"
            or hessian.get("source_before") != hessian.get("source_after")
            or hessian.get("fixed_input_unchanged") is not True
            or hessian.get("runtime") != hessian.get("runtime_after")
            or curvature_parent.get("execution_status") != "completed"
            or curvature_parent.get("child_sha256") != curvature.sha(live_step.CURVATURE)
            or search.execution_status(curvature_resource) != "completed"
            or checkpoint.get("status") != "completed" or h.get("hvp_columns") != 26
            or h.get("hvp_calls_total") != 27
            or not h.get("minimum_eigenpair_audit", {}).get("passed")
            or h.get("hessian") != checkpoint.get("final_result", {}).get("hessian")):
        raise ValueError("pinned f82 curvature/checkpoint receipts are not a completed SPD block source")
    matrix = torch.tensor(h["hessian"], dtype=torch.float64)
    preconditioner, audit = block_step.block_inverse_preconditioner(matrix)
    old = {"audit": hessian, "hessian": matrix, "preconditioner": preconditioner,
           "preconditioner_audit": audit}
    return plan, raw, old, prior_plan


def run(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return continuation.run(plan_path, plan_sha, output, base_loader=load_base, base_path=BASE)


def main() -> None:
    continuation.main(run_fn=run, default_plan=PLAN,
                      module_name="examples.weather_scenarios.fv_point_3h_90fc_followup")


if __name__ == "__main__":
    main()
