from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_3h_committed_base as committed
from examples.weather_scenarios import fv_point_3h_inexact_continuation as followup
from examples.weather_scenarios import fv_point_3h_inexact_comparison as comparison
from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation


RAW = (committed.EVIDENCE / "d31_inexact_comparison_20261007_attempt1/step.json")


def _case(tmp_path, monkeypatch):
    raw = json.loads(RAW.read_text())
    assert committed._accepted_endpoint(raw, raw["accepted_control_sha256"])[
        "control_sha256"] == raw["accepted_control_sha256"]
    plan: dict[str, Any] = {"experiment_kind": "inexact_continuation",
        "policy": continuation.policy_dict("inexact", 3), "linear_mode": "inexact",
        "base_step": followup.BASE_STEP, "base_plan": followup.BASE_PLAN,
        "source_files": {name: committed.curvature.sha(committed.ROOT / name)
                         for name in (followup.SELF, followup.TEST)},
        "base_control_sha256": raw["accepted_control_sha256"],
        "expected_base_control_sha256": raw["accepted_control_sha256"],
        "parameters_sha256": raw["parameters_sha256"],
        "frozen_input_sha256": followup._digest(raw["input_after"]),
        "archive_files": {}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = committed.curvature.sha(plan_path)
    monkeypatch.setattr(followup, "ROOT", tmp_path)
    real_sha = committed.curvature.sha
    def sha(path):
        path = Path(path)
        if path == plan_path:
            return plan_sha
        name = path.relative_to(tmp_path).as_posix()
        return plan["source_files"][name] if name in plan["source_files"] else real_sha(path)
    monkeypatch.setattr(committed.curvature, "sha", sha)
    monkeypatch.setattr(committed, "load_base", lambda *_: (plan, raw, {"preconditioner": 1}, {}))
    endpoint = tmp_path / "selected-inexact.json"
    monkeypatch.setattr(committed, "base_path", lambda _plan: endpoint)
    return plan, raw, plan_path, plan_sha, endpoint


def test_delegates_from_normally_completed_selected_endpoint(tmp_path, monkeypatch):
    _, raw, plan_path, plan_sha, endpoint = _case(tmp_path, monkeypatch)
    calls = []
    def runner(path, sha, output, **kwargs):
        loaded = kwargs["base_loader"](path, sha)
        calls.append((path, sha, output, kwargs, loaded[1]["accepted_control_sha256"]))
        return {"optimizer_steps_applied": 0}
    monkeypatch.setattr(continuation, "run", runner)
    output = tmp_path / "step.json"
    result = followup.run(plan_path, plan_sha, output)
    assert result["optimizer_steps_applied"] == 0
    _, _, _, kwargs, selected_sha = calls[0]
    assert selected_sha == raw["accepted_control_sha256"]
    assert kwargs["base_path"] == endpoint
    assert kwargs["linear_mode"] == "inexact" and kwargs["max_iterations"] == 3
    assert "shared_counts" not in kwargs and "absolute_deadline" not in kwargs


@pytest.mark.parametrize("change", [
    lambda plan: plan.update(linear_mode="strict"),
    lambda plan: plan["policy"].update(root_gradient_inf=1e-8),
])
def test_rejects_incompatible_mode_or_final_root_criterion(tmp_path, monkeypatch, change):
    plan, _, plan_path, plan_sha, _ = _case(tmp_path, monkeypatch)
    change(plan)
    with pytest.raises(ValueError, match="fixed three-step inexact continuation"):
        followup._validated_loader(plan_path, plan_sha)


@pytest.mark.parametrize("change", ["selected", "paired_complete", "arm_endpoint"])
def test_rejects_inconsistent_comparison_selector(tmp_path, monkeypatch, change):
    _, raw, plan_path, plan_sha, _ = _case(tmp_path, monkeypatch)
    if change == "selected":
        raw["selected_arm"] = "strict"
    elif change == "paired_complete":
        raw["comparison_complete"] = False
    else:
        raw["arms"]["inexact"]["accepted_control_sha256"] = "different"
    with pytest.raises(ValueError, match="selected inexact arm"):
        followup._validated_loader(plan_path, plan_sha)


def test_real_comparison_parent_and_resource_are_the_base_sidecars():
    raw = json.loads(RAW.read_text())
    assert raw["selected_arm"] == "inexact"
    assert raw["accepted_control_sha256"] == raw["arms"]["inexact"]["accepted_control_sha256"]
    assert RAW.with_suffix(".run.json").exists()
    assert RAW.with_suffix(".resource.json").exists()
    assert not RAW.with_name("inexact.run.json").exists()
