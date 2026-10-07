from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_3h_committed_base as committed
from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation


def _receipts():
    raw = json.loads(committed.BASE.read_text())
    parent = json.loads(committed.BASE.with_suffix(".run.json").read_text())
    resource = json.loads(committed.BASE.with_suffix(".resource.json").read_text())
    return raw, parent, resource


def test_real_committed_endpoint_matches_closed_iteration_and_trial():
    raw, parent, resource = _receipts()
    final = committed._validate_completed_receipt(
        raw, parent, resource, committed.curvature.sha(committed.BASE))

    assert final["control_sha256"] == raw["accepted_control_sha256"]
    assert len(raw["iterations"]) == raw["optimizer_steps_applied"] >= 1
    assert parent["execution_status"] == "completed"


@pytest.mark.parametrize("mutation", ["current_state", "uncommitted_candidate"])
def test_rejects_endpoint_that_does_not_close_against_final_iteration(mutation):
    raw, parent, resource = _receipts()
    raw = deepcopy(raw)
    if mutation == "current_state":
        raw["current_state"]["objective"] += 1e-6
    else:
        raw["trials"].append({**deepcopy(raw["trials"][-1]), "status": "accepted",
                              "control_sha256": "not-in-closed-iteration"})
    with pytest.raises(ValueError, match="accepted-trial|accepted trial"):
        committed._validate_completed_receipt(
            raw, parent, resource, committed.curvature.sha(committed.BASE))


@pytest.mark.parametrize("mutation", ["failed_parent", "runtime_mismatch"])
def test_rejects_failed_parent_or_changed_runtime(mutation):
    raw, parent, resource = _receipts()
    raw, parent = deepcopy(raw), deepcopy(parent)
    if mutation == "failed_parent":
        parent["execution_status"] = "failed"
    else:
        raw["runtime_after"] = {**raw["runtime"], "torch": "different"}
    with pytest.raises(ValueError, match="parent/resource|runtime"):
        committed._validate_completed_receipt(
            raw, parent, resource, committed.curvature.sha(committed.BASE))


def _comparison_plan(monkeypatch=None) -> dict[str, Any]:
    # The comparison runner is added by the parent task; these equivalent
    # existing pins keep this loader-only fixture independently runnable.
    if monkeypatch is not None:
        monkeypatch.setattr(committed, "COMPARISON", committed.CONTINUATION)
        monkeypatch.setattr(committed, "COMPARISON_TEST", committed.CONTINUATION_TEST)
    prior = json.loads(committed.PRIOR_PLAN.read_text())
    sources = {name: committed.curvature.sha(committed.ROOT / name)
               for name in prior["source_files"]}
    for name in (committed.SELF, committed.TEST, committed.CONTINUATION,
                 committed.CONTINUATION_TEST, committed.COMPARISON,
                 committed.COMPARISON_TEST, committed._relative(committed.R9_ARCHIVE)):
        sources[name] = committed.curvature.sha(committed.ROOT / name)
    archives = {}
    for path in (committed.BASE, committed.BASE.with_suffix(".run.json"),
                 committed.BASE.with_suffix(".resource.json"), committed.BASE_PLAN,
                 committed.HESSIAN, committed.CHECKPOINT, committed.HESSIAN_PARENT,
                 committed.HESSIAN_RESOURCE, committed.R9_ARCHIVE):
        archives[committed._relative(path)] = committed.curvature.sha(path)
    return {"policy": continuation.policy_dict(), "comparison_policy": {
            "arm_order": ["strict", "inexact"], "arm_max_iterations": 1,
            "selected_arm": "inexact", "shared_hvp_cap": 90,
            "internal_seconds": 720.0, "outer_seconds": 780.0,
            "guarded_launches": 1, "cold_start": True, "shared_deadline": True},
        "source_files": sources, "archive_files": archives,
        "parameters_sha256": json.loads(committed.BASE.read_text())["parameters_sha256"],
        "base_step": committed._relative(committed.BASE),
        "base_plan": committed._relative(committed.BASE_PLAN)}


def test_changed_source_pin_is_rejected_in_small_temp_plan(monkeypatch):
    plan = _comparison_plan(monkeypatch)
    plan["source_files"][committed.SELF] = "0" * 64
    path = committed.ROOT / "graphify-out/fv-root-cause-20260919/.committed-base-tamper-plan.json"
    path.write_text(json.dumps(plan))
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="pin changed"):
            committed._require_plan(path, digest)
    finally:
        path.unlink(missing_ok=True)


def test_base_path_requires_the_plan_archive_pin(monkeypatch):
    plan = _comparison_plan(monkeypatch)
    assert committed.base_path(plan) == committed.BASE
    del plan["archive_files"][committed._relative(committed.BASE)]
    with pytest.raises(ValueError, match="archive pins"):
        committed.base_path(plan)


def test_loader_returns_raw_receipt_and_f82_preconditioner_without_recomputation(monkeypatch):
    plan = _comparison_plan(monkeypatch)
    path = committed.ROOT / "graphify-out/fv-root-cause-20260919/.committed-base-plan.json"
    path.write_text(json.dumps(plan))
    try:
        plan_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded_plan, raw, bundle, prior_plan = committed.load_base(path, plan_sha)
    finally:
        path.unlink(missing_ok=True)

    assert loaded_plan["base_step"] == committed._relative(committed.BASE)
    assert raw["accepted_control_sha256"] == raw["iterations"][-1]["accepted_control_sha256"]
    assert bundle["hessian"].shape == (26, 26)
    assert prior_plan["base_control_sha256"] == raw["base_control_sha256"]
