from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_3h_inexact_comparison as comparison


def _inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison.committed, "base_path", lambda _plan: tmp_path / "base.json")
    plan = {"comparison_policy": comparison.comparison_policy()}
    base = {"accepted_control": [0.0] * 26, "accepted_control_sha256": "same-base"}
    return plan, base


def _arm(mode, counts, *, valid=True) -> dict[str, Any]:
    counts["hvp_calls"] += 3 if mode == "strict" else 2
    counts["hvp_calls_completed"] = counts["hvp_calls"]
    counts["pcg_iterations_completed"] += 1
    counts["pcg_solves_started"] += 1
    return {"execution_status": "completed", "fixed_input_unchanged": valid,
        "source_unchanged": valid, "numerical_status": "iteration_limit",
        "optimizer_steps_applied": 1, "accepted_control": [1.0] * 26,
        "accepted_control_sha256": mode, "iterations": [{"arm": mode}], "trials": [],
        "current_state": {"arm": mode}, "input_before": {}, "input_after": {},
        "parameters_sha256": "same-p", "runtime": {}, "runtime_after": {}}


def test_arms_share_deadline_and_counters_but_not_starting_points(tmp_path, monkeypatch):
    plan, base = _inputs(tmp_path, monkeypatch)
    calls = []

    def runner(_path, _sha, _output, **kwargs):
        calls.append(kwargs)
        return _arm(kwargs["linear_mode"], kwargs["shared_counts"])

    result = comparison.compare_arms(tmp_path / "plan.json", "sha", tmp_path / "step.json",
        plan, base, float("inf"), arm_runner=runner)
    assert [c["linear_mode"] for c in calls] == ["strict", "inexact"]
    assert calls[0]["shared_counts"] is calls[1]["shared_counts"]
    assert calls[0]["absolute_deadline"] == calls[1]["absolute_deadline"]
    assert calls[0]["base_path"] == calls[1]["base_path"]
    assert all(c["max_iterations"] == 1 for c in calls)
    assert result["hvp_calls"] == 5
    assert result["arm_counter_deltas"]["strict"]["hvp_calls"] == 3
    assert result["arm_counter_deltas"]["inexact"]["hvp_calls"] == 2
    assert result["optimizer_steps_applied"] == 1
    assert result["accepted_control_sha256"] == "inexact"
    assert result["selected_arm"] == "inexact"


def test_shared_cap_prevents_second_arm_without_reset(tmp_path, monkeypatch):
    plan, base = _inputs(tmp_path, monkeypatch)
    calls = []

    def runner(_path, _sha, _output, **kwargs):
        calls.append(kwargs["linear_mode"])
        counts = kwargs["shared_counts"]
        result = _arm("strict", counts)
        counts["hvp_calls"] = counts["hvp_calls_completed"] = comparison.continuation.MAX_HVP
        return result

    result = comparison.compare_arms(tmp_path / "plan.json", "sha", tmp_path / "step.json",
        plan, base, float("inf"), arm_runner=runner)
    assert calls == ["strict"]
    assert result["hvp_calls"] == comparison.continuation.MAX_HVP
    assert result["numerical_status"] == "budget_refusal"
    assert result["optimizer_steps_applied"] == 0
    assert result["accepted_control_sha256"] == "same-base"
    assert result["selected_arm"] is None


def test_unverified_inexact_endpoint_is_not_selected(tmp_path, monkeypatch):
    plan, base = _inputs(tmp_path, monkeypatch)

    def runner(_path, _sha, _output, **kwargs):
        return _arm(kwargs["linear_mode"], kwargs["shared_counts"],
                    valid=kwargs["linear_mode"] == "strict")

    result = comparison.compare_arms(tmp_path / "plan.json", "sha", tmp_path / "step.json",
        plan, base, float("inf"), arm_runner=runner)
    assert result["execution_status"] == "failed"
    assert result["optimizer_steps_applied"] == 0
    assert result["accepted_control_sha256"] == "same-base"
    assert result["selected_arm"] is None


def test_unexpected_arm_error_preserves_completed_reference_and_counts(tmp_path, monkeypatch):
    plan, base = _inputs(tmp_path, monkeypatch)

    def runner(_path, _sha, _output, **kwargs):
        if kwargs["linear_mode"] == "inexact":
            kwargs["shared_counts"]["hvp_calls"] += 1
            raise RuntimeError("operator failed")
        return _arm("strict", kwargs["shared_counts"])

    with pytest.raises(RuntimeError, match="operator failed"):
        comparison.compare_arms(tmp_path / "plan.json", "sha", tmp_path / "step.json",
            plan, base, float("inf"), arm_runner=runner)
    import json
    saved = json.loads((tmp_path / "step.json").read_text())
    assert "strict" in saved["arms"]
    assert saved["hvp_calls"] == 4
    assert saved["execution_status"] == "failed"


def test_budget_only_arm_keeps_comparison_incomplete_and_unselected(tmp_path, monkeypatch):
    plan, base = _inputs(tmp_path, monkeypatch)
    calls = []

    def runner(_path, _sha, _output, **kwargs):
        calls.append(kwargs["linear_mode"])
        return {"execution_status": "completed", "numerical_status": "budget_refusal",
                "optimizer_steps_applied": 0}

    result = comparison.compare_arms(tmp_path / "plan.json", "sha", tmp_path / "step.json",
        plan, base, float("inf"), arm_runner=runner)
    assert calls == ["strict"]
    assert result["numerical_status"] == "budget_refusal"
    assert result["comparison_complete"] is False
    assert result["selected_arm"] is None


def test_expired_before_any_arm_is_budget_not_integrity_failure(tmp_path, monkeypatch):
    import hashlib
    import json
    plan = {"source_files": {}, "archive_files": {},
            "comparison_policy": comparison.comparison_policy(),
            "arm_policies": {mode: comparison.continuation.policy_dict(mode, 1)
                             for mode in ("strict", "inexact")}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = comparison.continuation.curvature.blocks.runtime_identity()
    base = {"accepted_control": [0.0] * 26, "accepted_control_sha256": "base",
            "current_state": {}, "input_after": {}, "parameters_sha256": "p",
            "source_unchanged": True, "fixed_input_unchanged": True,
            "runtime": runtime, "runtime_after": runtime}
    clock = {"now": 0.0}
    def loader(*_args):
        clock["now"] = 721.0
        return plan, base, {}, {}
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    monkeypatch.setattr(comparison.committed, "load_base", loader)
    monkeypatch.setattr(comparison.time, "monotonic", lambda: clock["now"])
    result = comparison.run(path, digest, tmp_path / "step.json")
    assert result["execution_status"] == "completed"
    assert result["numerical_status"] == "budget_refusal"
    assert result["fixed_input_unchanged"] is True
    assert result["base_metadata_only"] is True
    assert result["selected_arm"] is None
    assert result["arms"] == {}
