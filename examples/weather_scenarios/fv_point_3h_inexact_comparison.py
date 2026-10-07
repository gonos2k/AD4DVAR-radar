"""One paired cold-start strict/inexact comparison under a single resource budget."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from examples.weather_scenarios import fv_point_3h_committed_base as committed
from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as search

ROOT, EVIDENCE = committed.ROOT, committed.EVIDENCE
PLAN = EVIDENCE / "D31_COMMITTED_BASE_COMPARISON_PLAN_20261007.json"
SELF = "examples/weather_scenarios/fv_point_3h_inexact_comparison.py"
TEST = "tests/test_fv_point_3h_inexact_comparison.py"
COUNTER_KEYS = ("hvp_calls", "hvp_calls_completed", "pcg_iterations_completed", "pcg_solves_started")


def comparison_policy() -> dict[str, Any]:
    return {"arm_order": ["strict", "inexact"], "arm_max_iterations": 1,
            "selected_arm": "inexact", "shared_hvp_cap": continuation.MAX_HVP,
            "internal_seconds": continuation.INTERNAL_SECONDS,
            "outer_seconds": continuation.WALL_SECONDS, "guarded_launches": 1,
            "cold_start": True, "shared_deadline": True}


def compare_arms(plan_path: Path, plan_sha: str, output: Path,
                 plan: dict[str, Any], base: dict[str, Any], deadline: float, *,
                 arm_runner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Both arms start at the original base; counters/deadline never reset."""
    if output.exists() or output.is_symlink():
        raise ValueError("comparison output must be fresh")
    if plan.get("comparison_policy") != comparison_policy():
        raise ValueError("comparison policy differs from the declared paired experiment")
    runner = arm_runner or continuation.run
    counts = {key: 0 for key in COUNTER_KEYS}
    result: dict[str, Any] = {"phase": "running", "execution_status": "completed",
        "numerical_status": "not_reached", "plan_sha256": plan_sha,
        "comparison_policy": plan["comparison_policy"], "arms": {},
        "arm_counter_deltas": {}, "optimizer_steps_applied": 0, **counts,
        "base_control_sha256": base["accepted_control_sha256"],
        "accepted_control": base["accepted_control"],
        "accepted_control_sha256": base["accepted_control_sha256"],
        "planned_selected_arm": "inexact", "selected_arm": None, "full_root_claim": False,
        "eligible_stationary_point": False, "response_computed": False,
        "forecast_score_computed": False, "accepted_point_curvature": "not_computed",
        "scope": "strict-first cold-start trial with shared remaining budget; no equal per-arm reserve or global speedup claim",
        "comparison_complete": False}
    search._write(output, result)
    mode = "not_started"
    try:
        for mode in plan["comparison_policy"]["arm_order"]:
            if time.monotonic() >= deadline or counts["hvp_calls"] >= continuation.MAX_HVP:
                result.update(numerical_status="budget_refusal", unstarted_arm=mode)
                break
            previous = dict(counts)
            arm = runner(plan_path, plan_sha, output.parent / f"{mode}.json",
                base_loader=committed.load_base, base_path=committed.base_path(plan),
                linear_mode=mode, max_iterations=1, shared_counts=counts,
                absolute_deadline=deadline)
            result["arms"][mode] = arm
            result["arm_counter_deltas"][mode] = {key: counts[key] - previous[key] for key in COUNTER_KEYS}
            result.update(counts)
            search._write(output, result)
            if arm.get("execution_status") != "completed" or arm.get("fixed_input_unchanged") is not True:
                if arm.get("numerical_status") == "budget_refusal":
                    result.update(numerical_status="budget_refusal", budget_refused_arm=mode)
                else:
                    result.update(execution_status="failed", numerical_status="arm_execution_refusal",
                                  failed_arm=mode)
                break
            if arm.get("numerical_status") == "budget_refusal":
                result.update(numerical_status="budget_refusal", budget_refused_arm=mode)
                break
        else:
            result["numerical_status"] = "paired_comparison_completed"
            result["comparison_complete"] = True
        selected = result["arms"].get("inexact")
        if (selected and selected.get("optimizer_steps_applied") == 1
                and selected.get("execution_status") == "completed"
                and selected.get("fixed_input_unchanged") is True
                and selected.get("source_unchanged") is True):
            result["selected_arm"] = "inexact"
            for key in ("optimizer_steps_applied", "accepted_control", "accepted_control_sha256",
                        "iterations", "trials", "current_state", "input_before", "input_after",
                        "parameters_sha256", "runtime", "runtime_after"):
                result[key] = selected[key]
            result["selected_numeric_status"] = selected["numerical_status"]
        else:
            result["selected_numeric_status"] = selected.get("numerical_status") if selected else "not_reached"
    except search.StepRefusal as error:
        result.update(numerical_status="budget_refusal" if isinstance(error, search.BudgetRefusal)
                      else getattr(error, "status", "step_refusal"), refusal=str(error))
        partial_path = output.parent / f"{mode}.json"
        if partial_path.exists():
            try:
                partial = json.loads(partial_path.read_text())
                if isinstance(partial, dict):
                    result["arms"][mode] = partial
            except (OSError, ValueError):
                pass
    except Exception as error:
        result.update(phase="execution_error", execution_status="failed",
                      numerical_status="execution_error", failure=f"{type(error).__name__}: {error}")
        result.update(counts)
        search._write(output, result)
        raise
    result.update(counts)
    return result


def run(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + continuation.INTERNAL_SECONDS
    plan, base, _, _ = committed.load_base(plan_path, plan_sha)
    if (plan.get("comparison_policy") != comparison_policy()
            or plan.get("arm_policies") != {mode: continuation.policy_dict(mode, 1)
                                          for mode in ("strict", "inexact")}):
        raise ValueError("unsupported comparison policy or forcing schedule")
    names = set(plan["source_files"]) | set(plan["archive_files"]) | {plan_path.relative_to(ROOT).as_posix()}
    before = {name: committed.curvature.sha(ROOT / name) for name in names}
    if (before[plan_path.relative_to(ROOT).as_posix()] != plan_sha
            or any(before[name] != digest for name, digest in
                   {**plan["source_files"], **plan["archive_files"]}.items())):
        raise ValueError("comparison pins changed after preflight")
    result = compare_arms(plan_path, plan_sha, output, plan, base, deadline)
    after = {name: committed.curvature.sha(ROOT / name) for name in names}
    verified_arms = [arm for arm in result["arms"].values()
                     if arm.get("fixed_input_unchanged") is True and arm.get("source_unchanged") is True]
    runtime = continuation.curvature.blocks.runtime_identity()
    base_closed = (base.get("source_unchanged") is True
                   and base.get("fixed_input_unchanged") is True
                   and base.get("runtime") == base.get("runtime_after") == runtime)
    intact = before == after and (bool(verified_arms) or base_closed)
    result["base_metadata_only"] = not bool(verified_arms)
    if not verified_arms:
        result.update(current_state=base["current_state"],
            input_before=base["input_after"], input_after=base["input_after"],
            parameters_sha256=base["parameters_sha256"], runtime=runtime, runtime_after=runtime)
    result.update(source_before=before, source_after=after, source_unchanged=before == after,
                  fixed_input_unchanged=intact, elapsed_seconds=time.monotonic() - started,
                  phase="finished" if intact and result["execution_status"] == "completed" else "execution_error")
    if not intact:
        result.update(execution_status="failed", numerical_status="integrity_refusal")
    search._write(output, result)
    return result


def main() -> None:
    continuation.main(run_fn=run, default_plan=PLAN,
                      module_name="examples.weather_scenarios.fv_point_3h_inexact_comparison")


if __name__ == "__main__":
    main()
