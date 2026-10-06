from __future__ import annotations

import time
import json
from typing import Any

import pytest
import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation


def _problem(target: float = 0.0) -> tuple[Any, ...]:
    control = torch.zeros(26, dtype=torch.float64)
    control[0] = 0.15
    parameters = torch.zeros(13, dtype=torch.float64)
    hvp_points: list[Tensor] = []

    def objective(candidate, _parameters):
        shifted = candidate.clone()
        shifted[0] -= target
        return shifted @ shifted / 2

    def gradient(candidate, _parameters):
        result = candidate.clone()
        result[0] -= target
        return result

    def branch(candidate, _parameters):
        return ({"status": "passed_strict_branch", "euler_stages": 3600,
                 "choice_stage_count": 3600, "face_sign_stage_count": 3600,
                 "signature_sha256": "same"}, {"complete": True})

    def hvp(point, _parameters, vector):
        hvp_points.append(point.clone())
        return vector.clone()

    def preconditioner(vector):
        return vector.clone()

    return control, parameters, objective, gradient, branch, hvp, preconditioner, hvp_points


def test_repeats_fresh_pcg_directions_at_each_accepted_point(monkeypatch):
    inputs = _problem()
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10, max_iterations=2)

    assert record["execution_status"] == "completed"
    assert record["numerical_status"] == "iteration_limit"
    assert record["optimizer_steps_applied"] == 2
    assert len(record["iterations"]) == 2
    assert record["iterations"][0]["status"] == record["iterations"][1]["status"] == "accepted"
    assert record["hvp_calls"] == record["hvp_calls_completed"]
    assert record["pcg_iterations_completed"] == 2
    assert len(inputs[7]) > 2
    assert any(not torch.equal(point, inputs[0]) for point in inputs[7])
    assert inputs[0][0] == pytest.approx(0.15)
    assert (record["iterations"][0]["accepted_control_sha256"]
            == record["iterations"][1]["base_control_sha256"])


def test_stationarity_stop_is_pending_audit_and_never_eligible(monkeypatch):
    inputs = list(_problem())
    inputs[0][0] = 0.05
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10, max_iterations=3)

    assert record["numerical_status"] == "root_pending_audit"
    assert record["optimizer_steps_applied"] == 1
    assert record["root_pending_audit"] is True
    assert record["eligible_stationary_point"] is False
    assert record["full_root_claim"] is False


def test_total_hvp_cap_keeps_prior_commit_and_records_failed_solve(monkeypatch):
    inputs = _problem()
    monkeypatch.setattr(continuation, "MAX_HVP", 3)
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10, max_iterations=3)

    assert record["numerical_status"] == "budget_refusal"
    assert record["optimizer_steps_applied"] == 1
    assert record["hvp_calls"] == 3
    assert record["hvp_calls_completed"] == 3
    assert record["current_iteration"]["pcg_iterations_failed_solve"] == "not_recorded"
    assert record["accepted_control"][0] == pytest.approx(0.10)


def test_nonpositive_curvature_refuses_without_j_only_fallback(monkeypatch):
    inputs = list(_problem())
    inputs[5] = lambda _point, _parameters, vector: -vector
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10)

    assert record["numerical_status"] in {"linear_solve_refusal", "curvature_refusal"}
    assert record["optimizer_steps_applied"] == 0
    assert record["trials"] == []
    assert record["hvp_calls"] > 0
    assert "fallback" not in record


def test_later_commit_failure_discards_only_provisional_candidate(monkeypatch):
    inputs = _problem()
    calls = 0

    def commit(*_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("diagnostic closure failed")
        return {"closed": True}

    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 10,
        max_iterations=3, commit_candidate=commit)

    assert record["execution_status"] == "failed"
    assert record["optimizer_steps_applied"] == 1
    assert record["iterations"][0]["status"] == "accepted"
    assert record["current_iteration"]["trials"][-1]["status"] == "candidate_not_committed"
    assert record["accepted_control"][0] == pytest.approx(0.10)
    assert record["hvp_calls"] == record["hvp_calls_completed"]


def test_deadline_is_checked_after_slow_branch_callback(monkeypatch):
    inputs = list(_problem())
    normal_branch = inputs[4]

    def slow_branch(candidate, parameters):
        time.sleep(0.02)
        return normal_branch(candidate, parameters)

    inputs[4] = slow_branch
    monkeypatch.setattr(continuation.dual.merit.seed_linear, "_valid_margins",
                        lambda *_args, **_kwargs: True)
    record = continuation.run_iterations(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], time.monotonic() + 0.005)

    assert record["numerical_status"] == "budget_refusal"
    assert record["optimizer_steps_applied"] == 0
    assert "budget exhausted" in record["refusal"]


def test_parent_summary_ignores_nonobject_partial_child(tmp_path):
    child = tmp_path / "step.json"
    child.write_text("[]\n")
    parent = {"completed_iterations": 0, "hvp_calls": 0, "numerical_status": "not_reached"}

    continuation._copy_partial_child_counts(parent, child)

    assert parent == {"completed_iterations": 0, "hvp_calls": 0, "numerical_status": "not_reached"}


@pytest.mark.parametrize("changes,expected", [
    ({}, "completed"),
    ({"wall_limit_seconds": 300.0}, "failed"),
    ({"sampled_peak_rss_bytes": continuation.RSS_BYTES + 1}, "resource_limited"),
    ({"resource_termination": "wall_time_limit"}, "resource_limited"),
    ({"received_sigterm": True}, "failed"),
    ({"monitor_error": "sampling failed"}, "failed"),
])
def test_continuation_classifies_its_own_actual_resource_plan(changes, expected):
    path = continuation.EVIDENCE / "e0b_repeat_dual_20261006_attempt1/step.resource.json"
    resource = json.loads(path.read_text())
    historical = resource.copy()
    resource.update(changes)

    assert continuation.execution_status(resource) == expected
    assert historical["wall_limit_seconds"] == 780.0
    assert continuation.search.execution_status(historical) == "failed"
