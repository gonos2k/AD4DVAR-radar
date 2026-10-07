"""Continue a committed inexact FV endpoint under the fixed bounded policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from examples.weather_scenarios import fv_point_3h_committed_base as committed
from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation

ROOT, EVIDENCE = committed.ROOT, committed.EVIDENCE
PLAN = EVIDENCE / "F462_INEXACT_CONTINUATION_PLAN_20261007.json"
BASE_STEP = "graphify-out/fv-root-cause-20260919/d31_inexact_comparison_20261007_attempt1/step.json"
BASE_PLAN = "graphify-out/fv-root-cause-20260919/D31_COMMITTED_BASE_COMPARISON_PLAN_20261007.json"
SELF = "examples/weather_scenarios/fv_point_3h_inexact_continuation.py"
TEST = "tests/test_fv_point_3h_inexact_continuation.py"


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _validated_loader(plan_path: Path, plan_sha: str):
    plan, raw, preconditioner, base_plan = committed.load_base(plan_path, plan_sha)
    if (plan.get("experiment_kind") != "inexact_continuation"
            or plan.get("policy") != continuation.policy_dict("inexact", 3)
            or plan.get("linear_mode") != "inexact"):
        raise ValueError("plan does not declare the fixed three-step inexact continuation")
    if not {SELF, TEST} <= set(plan.get("source_files", {})):
        raise ValueError("plan must pin the continuation wrapper and its regression tests")
    for name in (SELF, TEST):
        path = ROOT / name
        if path.is_symlink() or committed.curvature.sha(path) != plan["source_files"][name]:
            raise ValueError(f"continuation source pin changed: {name}")
    if (raw.get("accepted_control_sha256") != plan.get("expected_base_control_sha256")
            or raw.get("accepted_control_sha256") != plan.get("base_control_sha256")
            or raw.get("parameters_sha256") != plan.get("parameters_sha256")
            or _digest(raw.get("input_after"))
            != plan.get("frozen_input_sha256")):
        raise ValueError("selected inexact endpoint or frozen input differs from the plan")
    if "arms" in raw:
        arm = raw.get("arms", {}).get("inexact", {})
        if (raw.get("comparison_complete") is not True
                or raw.get("selected_arm") != "inexact" or raw.get("planned_selected_arm") != "inexact"
                or arm.get("execution_status") != "completed"
                or arm.get("fixed_input_unchanged") is not True
                or arm.get("source_unchanged") is not True
                or arm.get("linear_mode") != "inexact"
                or arm.get("accepted_control_sha256") != raw.get("accepted_control_sha256")
                or any(arm.get(key) != raw.get(key) for key in
                       ("accepted_control", "iterations", "trials", "current_state", "input_after"))):
            raise ValueError("comparison receipt does not close its selected inexact arm")
    elif raw.get("linear_mode") != "inexact":
        raise ValueError("base receipt is not an inexact continuation endpoint")
    return plan, raw, preconditioner, base_plan


def run(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    if (plan_path.is_symlink() or not plan_path.resolve().is_relative_to(ROOT.resolve())
            or committed.curvature.sha(plan_path) != plan_sha):
        raise ValueError("inexact continuation plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    return continuation.run(plan_path, plan_sha, output,
        base_loader=_validated_loader, base_path=committed.base_path(plan),
        linear_mode="inexact", max_iterations=3)


def main() -> None:
    continuation.main(run_fn=run, default_plan=PLAN,
        module_name="examples.weather_scenarios.fv_point_3h_inexact_continuation")


if __name__ == "__main__":
    main()
