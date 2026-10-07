"""Load a hash-pinned, normally completed FV continuation endpoint.

The loader validates receipts and cached curvature only. It never reruns FV or
reuses a historical search direction; callers start a fresh solve at the
committed control.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from examples.weather_scenarios import fv_point_3h_accepted_curvature as curvature
from examples.weather_scenarios import fv_point_3h_current_newton_step as diagnostics
from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation
from examples.weather_scenarios import fv_point_3h_hvp_newton_step as live_step
from examples.weather_scenarios import fv_point_3h_schur_newton_step as block_step

ROOT = curvature.ROOT
EVIDENCE = curvature.EVIDENCE
BASE = EVIDENCE / "90fc_dual_followup_20261007_attempt1/step.json"
BASE_PLAN = EVIDENCE / "90FC_DUAL_MERIT_FOLLOWUP_PLAN_20261007.json"
PLAN = EVIDENCE / "D31_COMMITTED_BASE_COMPARISON_PLAN_20261007.json"
SELF = "examples/weather_scenarios/fv_point_3h_committed_base.py"
TEST = "tests/test_fv_point_3h_committed_base.py"
CONTINUATION = "examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py"
CONTINUATION_TEST = "tests/test_fv_point_3h_dual_merit_continuation.py"
COMPARISON = "examples/weather_scenarios/fv_point_3h_inexact_comparison.py"
COMPARISON_TEST = "tests/test_fv_point_3h_inexact_comparison.py"
PRIOR_PLAN = EVIDENCE / "90FC_DUAL_MERIT_FOLLOWUP_PLAN_20261007.json"
HESSIAN = EVIDENCE / "f82c_fresh_curvature_20261005/attempt_1/audit.json"
CHECKPOINT = EVIDENCE / "f82c_fresh_curvature_20261005/checkpoint/hessian_checkpoint.json"
HESSIAN_PARENT = HESSIAN.with_suffix(".run.json")
HESSIAN_RESOURCE = HESSIAN.with_suffix(".resource.json")
R9_ARCHIVE = curvature.R9_ARCHIVE


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _pinned_file(name: str, digest: str) -> Path:
    """Resolve one repository-relative pin and reject escapes and symlinks."""
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("source/archive pin must be a repository-relative path")
    path = ROOT / name
    if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve())
            or not path.is_file() or curvature.sha(path) != digest):
        raise ValueError(f"source/archive pin changed or escaped repository: {name}")
    return path


def base_path(plan: dict[str, Any]) -> Path:
    """Return the committed raw endpoint path declared by ``plan``."""
    archives = plan.get("archive_files")
    name = plan.get("base_step")
    if not isinstance(archives, dict) or not isinstance(name, str):
        raise ValueError("plan must declare a pinned base_step")
    path = ROOT / name
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("base_step must remain inside the repository and cannot be a symlink")
    if path.suffix != ".json":
        raise ValueError("base_step must be a pinned JSON receipt")
    if name not in archives:
        raise ValueError("plan archive pins must include base_step")
    _pinned_file(name, archives[name])
    for receipt in (path.with_suffix(".run.json"), path.with_suffix(".resource.json")):
        receipt_name = _relative(receipt)
        if receipt_name not in archives:
            raise ValueError("plan archive pins must include base parent and resource receipts")
        _pinned_file(receipt_name, archives[receipt_name])
    return path


def _require_plan(plan_path: Path, plan_sha: str) -> dict[str, Any]:
    if (plan_path.is_symlink() or not plan_path.resolve().is_relative_to(ROOT.resolve())
            or curvature.sha(plan_path) != plan_sha):
        raise ValueError("committed-base plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    if (not isinstance(plan.get("source_files"), dict)
            or not isinstance(plan.get("archive_files"), dict)):
        raise ValueError("plan must declare source and archive pins")
    if "comparison_policy" in plan:
        if plan.get("policy") != continuation.policy_dict() or not isinstance(plan["comparison_policy"], dict):
            raise ValueError("comparison plan must preserve the strict common caps")
        comparison = plan["comparison_policy"]
        common_comparison = {"arm_order": ["strict", "inexact"], "arm_max_iterations": 1,
            "shared_hvp_cap": 90, "internal_seconds": 720.0, "outer_seconds": 780.0}
        if any(comparison.get(key) != value for key, value in common_comparison.items()):
            raise ValueError("comparison plan must declare the fixed one-step cold-start shared-budget arms")
    elif (plan.get("experiment_kind") != "inexact_continuation"
            or plan.get("linear_mode") != "inexact"
            or plan.get("policy") != continuation.policy_dict("inexact", 3)):
        raise ValueError("continuation plan must declare the unchanged bounded inexact policy")

    prior_name = plan.get("base_plan")
    if not isinstance(prior_name, str) or prior_name not in plan["archive_files"]:
        raise ValueError("plan must declare a pinned base_plan")
    prior_path = _pinned_file(prior_name, plan["archive_files"][prior_name])
    prior = json.loads(prior_path.read_text())
    inherited = set(prior["source_files"])
    required_sources = inherited | {SELF, TEST, CONTINUATION, CONTINUATION_TEST,
        COMPARISON, COMPARISON_TEST, _relative(R9_ARCHIVE)}
    if not required_sources <= set(plan["source_files"]):
        raise ValueError("plan omits inherited sources or the committed-base comparison modules")
    endpoint = base_path(plan)
    required_archives = {_relative(endpoint), _relative(endpoint.with_suffix(".run.json")),
        _relative(endpoint.with_suffix(".resource.json")), prior_name,
        _relative(HESSIAN), _relative(CHECKPOINT), _relative(HESSIAN_PARENT),
        _relative(HESSIAN_RESOURCE), _relative(R9_ARCHIVE)}
    if not required_archives <= set(plan["archive_files"]):
        raise ValueError("plan omits committed endpoint or f82 curvature receipts")
    for name, digest in {**plan["source_files"], **plan["archive_files"]}.items():
        _pinned_file(name, digest)
    return plan


def _accepted_endpoint(raw: dict[str, Any], expected_raw_sha: str) -> dict[str, Any]:
    """Select the unique final accepted trial and close it against the iteration."""
    iterations = raw.get("iterations")
    accepted = [row for row in raw.get("trials", []) if row.get("status") == "accepted"]
    state = raw.get("current_state")
    if (raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or isinstance(raw.get("optimizer_steps_applied"), bool)
            or not isinstance(raw.get("optimizer_steps_applied"), int)
            or raw["optimizer_steps_applied"] < 1 or not isinstance(iterations, list)
            or len(iterations) != raw["optimizer_steps_applied"] or not accepted
            or not isinstance(state, dict)):
        raise ValueError("base receipt must be normally completed with closed accepted iterations")
    final = accepted[-1]
    last = iterations[-1]
    if last.get("status") != "accepted" or len(accepted) != len(iterations):
        raise ValueError("base receipt accepted-trial count must match its closed iterations")
    control_values = raw.get("accepted_control")
    try:
        control = torch.tensor(control_values, dtype=torch.float64)
        gradient = torch.tensor(final["gradient"], dtype=torch.float64)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("final accepted trial must contain numeric control and gradient") from error
    if (control.shape != (26,) or gradient.shape != (26,)
            or not bool(torch.isfinite(control).all()) or not bool(torch.isfinite(gradient).all())):
        raise ValueError("final accepted control and full gradient must be finite length-26 vectors")
    control_sha = curvature.tensor_sha(control)
    gradient_inf = float(gradient.abs().max())
    if (control_sha != expected_raw_sha or raw.get("accepted_control_sha256") != control_sha
            or final.get("control_sha256") != control_sha
            or last.get("accepted_control_sha256") != control_sha
            or final.get("control") != control_values
            or last.get("accepted_control") != control_values
            or final.get("objective") != state.get("objective")
            or final.get("phi") != state.get("phi")
            or final.get("gradient") != state.get("gradient")
            or gradient_inf != state.get("gradient_inf")
            or final.get("branch") != state.get("branch")
            or final.get("margins") != state.get("margins")
            or final.get("J_armijo_passed") is not True
            or final.get("Phi_armijo_passed") is not True
            or final.get("strict_point_passed") is not True):
        raise ValueError("last accepted trial, closed iteration, and current_state do not match")
    return final


def _validate_completed_receipt(raw: dict[str, Any], parent: dict[str, Any],
                                resource: dict[str, Any], raw_sha: str) -> dict[str, Any]:
    if (parent.get("execution_status") != "completed"
            or parent.get("child_sha256") != raw_sha
            or parent.get("completed_iterations") != raw.get("optimizer_steps_applied")
            or parent.get("hvp_calls") != raw.get("hvp_calls")
            or continuation.execution_status(resource) != "completed"
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("source_before") != raw.get("source_after")
            or not isinstance(raw.get("input_before"), dict)
            or not isinstance(raw.get("input_after"), dict)
            or raw["input_before"].get("parameters_sha256") != raw.get("parameters_sha256")
            or raw["input_after"].get("parameters_sha256") != raw.get("parameters_sha256")
            or raw.get("runtime") != raw.get("runtime_after")):
        raise ValueError("base parent/resource or source/input/runtime completion receipt is inconsistent")
    expected_control = raw.get("accepted_control_sha256")
    if not isinstance(expected_control, str):
        raise ValueError("base accepted control hash must be a string")
    return _accepted_endpoint(raw, expected_control)


def _validate_curvature(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    hessian = json.loads(HESSIAN.read_text())
    checkpoint = json.loads(CHECKPOINT.read_text())
    parent = json.loads(HESSIAN_PARENT.read_text())
    resource = json.loads(HESSIAN_RESOURCE.read_text())
    h = hessian.get("curvature", {})
    if (hessian.get("phase") != "finished" or hessian.get("execution_status") != "completed"
            or hessian.get("numerical_status") != "curvature_completed"
            or hessian.get("source_unchanged") is not True
            or hessian.get("source_before") != hessian.get("source_after")
            or hessian.get("fixed_input_unchanged") is not True
            or hessian.get("runtime") != hessian.get("runtime_after")
            or parent.get("execution_status") != "completed"
            or parent.get("child_sha256") != curvature.sha(HESSIAN)
            or parent.get("numerical_status") != "curvature_completed"
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("received_sigterm") is not False or resource.get("monitor_error") is not None
            or resource.get("wall_limit_seconds") != 300.0
            or resource.get("rss_limit_bytes") != 1024**3
            or resource.get("elapsed_seconds", 301) > 300.0
            or resource.get("sampled_peak_rss_bytes", 1024**3 + 1) > 1024**3
            or checkpoint.get("status") != "completed"
            or h.get("hvp_columns") != 26 or h.get("hvp_calls_total") != 27
            or h.get("hessian") != checkpoint.get("final_result", {}).get("hessian")
            or not h.get("minimum_eigenpair_audit", {}).get("passed")):
        raise ValueError("pinned completed 26-column plus independent-HVP f82 curvature is inconsistent")
    diagnostics._require_cached_objective_sources(plan, hessian)
    matrix = torch.tensor(h["hessian"], dtype=torch.float64)
    preconditioner, audit = block_step.block_inverse_preconditioner(matrix)
    bundle = {"audit": hessian, "hessian": matrix, "preconditioner": preconditioner,
              "preconditioner_audit": audit}
    return bundle, checkpoint, hessian


def load_base(plan_path: Path, plan_sha: str):
    """Return ``(plan, original_raw, f82_bundle, prior_plan)`` for a fresh solve."""
    plan = _require_plan(plan_path, plan_sha)
    endpoint_path = base_path(plan)
    raw_sha = plan["archive_files"][_relative(endpoint_path)]
    raw = json.loads(endpoint_path.read_text())
    base_plan_path = ROOT / plan["base_plan"]
    base_plan_sha = plan["archive_files"][plan["base_plan"]]
    if (base_plan_sha != curvature.sha(base_plan_path)
            or base_plan_sha != raw.get("plan_sha256")):
        raise ValueError("base plan pin must equal the raw endpoint's producing plan hash")
    prior_plan = json.loads(base_plan_path.read_text())
    prior_controls = [prior_plan[key] for key in ("base_control_sha256", "accepted_control_sha256")
                      if key in prior_plan]
    if (not prior_controls or len(set(prior_controls)) != 1
            or prior_plan.get("parameters_sha256") != raw.get("parameters_sha256")
            or plan.get("parameters_sha256") != raw.get("parameters_sha256")):
        raise ValueError("base plan and completed endpoint parameters do not match")
    parent = json.loads(endpoint_path.with_suffix(".run.json").read_text())
    resource = json.loads(endpoint_path.with_suffix(".resource.json").read_text())
    if raw.get("plan_sha256") != base_plan_sha:
        raise ValueError("base raw receipt does not name its pinned producing plan")
    final = _validate_completed_receipt(raw, parent, resource, raw_sha)
    if (raw.get("base_control_sha256") != prior_controls[0]
            or raw["source_before"].get(plan["base_plan"]) != base_plan_sha):
        raise ValueError("base endpoint does not continue from its producing plan's accepted control")
    if final.get("control_sha256") != raw.get("accepted_control_sha256"):
        raise ValueError("committed endpoint control hash mismatch")
    curvature_bundle, _, _ = _validate_curvature(plan)
    return plan, raw, curvature_bundle, prior_plan
