from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_gn_committed_continuation as runner
from examples.weather_scenarios import fv_point_3h_search_progress as search
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def _model() -> SimpleNamespace:
    return SimpleNamespace(direction=torch.tensor([1.0, 0.0], dtype=torch.float64), theta=0.5,
        gates={"finite": True, "both_side_gradients_descend": True, "merit_descends": True},
        merit_product=-0.01, residual_direction=torch.tensor([1.0, 0.0], dtype=torch.float64))


def _facts(accepted: bool = False) -> dict[str, bool]:
    keys = ("mixing_minimum_valid", "J_armijo_passed", "branch_pair_passed", "face_audit_passed",
        "side_objectives_match_native", "side_gradients_finite", "finite")
    return {**{key: True for key in keys}, "F_squared_armijo_passed": accepted}


def test_runner_callback_persists_progress_without_double_counting_prior_slots(tmp_path):
    path = tmp_path / "step.json"
    record: dict[str, Any] = {"candidate_slots_used": 7, "last_confirmed_iterations": 1,
        "last_confirmed_control_sha256": runner.BASE_SHA, "closed_steps": [{"P2": True}]}
    tangent._write(path, record)

    def progress(trials: list[dict[str, Any]], meta: dict[str, Any]) -> None:
        runner._persist_search_progress(record, path, 7, trials, meta)

    _, trials, meta = search.search_candidates(_model(), torch.zeros(2, dtype=torch.float64),
        lambda *_: _facts(True), chart_candidate=lambda control, direction, alpha: control + alpha * direction,
        progress=progress, radius=0.05, limit=24)

    saved = json.loads(path.read_text())
    assert meta["slots_started"] == meta["slots_completed"] == 1
    assert meta["evaluations_started"] == meta["evaluations_completed"] == 1
    assert saved["candidate_slots_used"] == 8
    assert saved["last_confirmed_iterations"] == 1
    assert saved["last_confirmed_control_sha256"] == runner.BASE_SHA
    assert saved["closed_steps"] == [{"P2": True}]
    assert saved["candidate_trials"] == trials


def test_nonfinite_facts_are_json_safe_refusals_and_search_continues(tmp_path):
    path = tmp_path / "step.json"
    record: dict[str, Any] = {"candidate_slots_used": 0, "last_confirmed_iterations": 0,
        "last_confirmed_control_sha256": runner.BASE_SHA, "closed_steps": []}
    tangent._write(path, record)
    calls = 0

    def evaluate(*_args: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return runner._safe_candidate_facts({**_facts(), "finite": False,
                "objective": float("nan"), "F_squared": float("inf"),
                "G": [float("nan"), 1.0], "branch_trace": {"nested": [float("-inf")]}})
        return _facts(True)

    accepted, trials, meta = search.search_candidates(_model(), torch.zeros(2, dtype=torch.float64),
        evaluate, chart_candidate=lambda control, direction, alpha: control + alpha * direction,
        progress=lambda t, m: runner._persist_search_progress(record, path, 0, t, m),
        radius=0.05, limit=2)
    saved = json.loads(path.read_text())
    assert accepted is not None and accepted["index"] == 2
    assert calls == 2 and meta["evaluations_completed"] == 2
    assert trials[0]["status"] == "gate_refused" and trials[0]["finite"] is False
    assert trials[0]["objective"] is None and trials[0]["F_squared"] is None
    assert trials[0]["G"] == [None, 1.0]
    assert trials[0]["branch_trace"]["nested"] == [None]
    assert saved["candidate_trials"] == trials
    assert saved["candidate_trials"][1]["status"] == "accepted"


def test_full_G_match_rejects_same_norm_different_components():
    actual = torch.zeros(27, dtype=torch.float64)
    expected = torch.zeros_like(actual)
    actual[0], expected[1] = 1.0, 1.0
    assert torch.linalg.vector_norm(actual) == torch.linalg.vector_norm(expected)
    comparison = runner._full_G_match(actual, expected)
    assert comparison["passed"] is False


def test_timeout_checkpoint_keeps_started_and_completed_counts_separate(tmp_path):
    path = tmp_path / "step.json"
    record: dict[str, Any] = {"candidate_slots_used": 3, "last_confirmed_iterations": 0,
        "last_confirmed_control_sha256": runner.BASE_SHA}
    tangent._write(path, record)
    calls = 0

    def evaluate(*_args: Any) -> dict[str, bool]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("synthetic internal deadline")
        return _facts()

    with pytest.raises(TimeoutError):
        search.search_candidates(_model(), torch.zeros(2, dtype=torch.float64), evaluate,
            chart_candidate=lambda control, direction, alpha: control + alpha * direction,
            progress=lambda trials, meta: runner._persist_search_progress(record, path, 3, trials, meta),
            radius=0.05, limit=24)

    saved = json.loads(path.read_text())
    assert saved["search"]["slots_started"] == 3
    assert saved["search"]["slots_completed"] == 2
    assert saved["search"]["evaluations_started"] == 3
    assert saved["search"]["evaluations_completed"] == 2
    assert saved["candidate_slots_used"] == 6
    assert saved["last_confirmed_control_sha256"] == runner.BASE_SHA


def test_chart_failure_counts_slot_without_starting_evaluation(tmp_path):
    path = tmp_path / "step.json"
    record: dict[str, Any] = {"candidate_slots_used": 0, "last_confirmed_iterations": 0,
        "last_confirmed_control_sha256": runner.BASE_SHA}
    tangent._write(path, record)

    def fail_chart(*_args: Any) -> Tensor:
        raise ValueError("synthetic chart failure")

    with pytest.raises(ValueError, match="synthetic chart failure"):
        search.search_candidates(_model(), torch.zeros(2, dtype=torch.float64),
            lambda *_: pytest.fail("failed chart must not evaluate the objective"),
            chart_candidate=fail_chart,
            progress=lambda trials, meta: runner._persist_search_progress(record, path, 0, trials, meta),
            radius=0.05, limit=24)
    saved = json.loads(path.read_text())
    assert saved["search"]["slots_started"] == 1
    assert saved["search"]["slots_completed"] == 0
    assert saved["search"]["evaluations_started"] == saved["search"]["evaluations_completed"] == 0
    assert saved["candidate_trials"][0]["status"] == "slot_failed"


@pytest.mark.parametrize(
    ("chart", "status"),
    [
        (lambda control, _direction, _alpha: control.clone(), "zero_control_movement_refused"),
        (lambda control, _direction, _alpha: control + torch.tensor([float("inf"), 0.0]),
            "nonfinite_chart_movement_refused"),
        (lambda control, direction, alpha: control + 30.0 * alpha * direction,
            "actual_chart_radius_refused"),
    ],
)
def test_zero_nonfinite_and_over_radius_slots_do_not_start_evaluation(chart, status):
    progress_records: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    accepted, trials, meta = search.search_candidates(_model(), torch.zeros(2, dtype=torch.float64),
        lambda *_: pytest.fail("refused chart points must not evaluate the objective"),
        chart_candidate=chart, progress=lambda t, m: progress_records.append((t.copy(), m.copy())),
        radius=0.05, limit=1)
    assert accepted is None
    assert trials[0]["status"] == status
    assert meta["evaluations_started"] == meta["evaluations_completed"] == 0


def test_current_parent_archive_and_p2_are_qualified_read_only():
    parent = runner._load_parent()
    control = torch.tensor(parent["current_control"], dtype=torch.float64)
    assert tangent._tensor_sha(control) == runner.BASE_SHA
    assert parent["current_control_sha256"] == runner.BASE_SHA
    assert parent["final_repeat"]["control"] == parent["current_control"]


def _p2_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    control = torch.linspace(-0.1, 0.2, 26, dtype=torch.float64)
    minus, plus = torch.zeros_like(control), torch.zeros_like(control)
    minus[0], plus[0] = 1.0, -1.0
    gradients = {"-1": minus.tolist(), "1": plus.tolist()}
    proposal = {"control": control.tolist(), "control_sha256": tangent._tensor_sha(control),
        "theta": 0.5, "objective": 1.0, "F_squared": 0.25, "side_gradients": gradients}
    repeat = {"control": control.tolist(), "theta": 0.5, "objective": 1.0,
        "F_squared": 0.25, "side_gradients": gradients,
        **{key: True for key in ("mixing_minimum_valid", "minimum_theta_matches_proposal",
            "side_gradients_finite", "native_objective_matches_proposal",
            "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
            "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
            "fixed_input_unchanged", "runtime_unchanged", "deadline_passed")}}
    return proposal, repeat


def test_atomic_commit_failure_preserves_last_confirmed_input_record(tmp_path, monkeypatch):
    proposal, repeat = _p2_pair()
    path = tmp_path / "step.json"
    record: dict[str, Any] = {"current_control_sha256": runner.BASE_SHA,
        "last_confirmed_control_sha256": runner.BASE_SHA, "last_confirmed_iterations": 0,
        "closed_steps": [], "optimizer_steps_applied": 0, "candidate_committed": False}
    tangent._write(path, record)

    def fail(_path: Path, _value: dict[str, Any]) -> None:
        raise OSError("simulated atomic commit failure")

    monkeypatch.setattr(runner.tangent, "_write", fail)
    accepted_full_G = [0.0] * 27
    accepted_full_G[0] = 0.5
    with pytest.raises(OSError, match="atomic commit"):
        runner._commit_closed_point(record, {"point_index": 0, "accepted_full_G": accepted_full_G},
            proposal, repeat, path)
    assert json.loads(path.read_text()) == record
    assert record["last_confirmed_control_sha256"] == runner.BASE_SHA
    assert record["optimizer_steps_applied"] == 0
    assert record["closed_steps"] == []
