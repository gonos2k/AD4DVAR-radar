from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_gn_local_commit as commit
from examples.weather_scenarios import fv_point_3h_gn_local_window_probe as local
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def _closed_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    control = torch.linspace(-0.2, 0.3, 26, dtype=torch.float64)
    minus = torch.zeros(26, dtype=torch.float64)
    plus = torch.zeros_like(minus)
    minus[0], plus[0] = 1.0, -1.0
    proposal = {"control": control.tolist(), "theta": 0.5, "objective": 2.0,
        "F_squared": 0.25, "candidate_mixing_minimum": True,
        "side_gradients": {"-1": minus.tolist(), "1": plus.tolist()}}
    repeat = {"control": control.tolist(), "theta": 0.5, "objective": 2.0,
        "F_squared": 0.25, "side_gradients": {"-1": minus.tolist(), "1": plus.tolist()},
        **{key: True for key in (
            "mixing_minimum_valid", "minimum_theta_matches_proposal", "side_gradients_finite",
            "native_objective_matches_proposal", "side_objectives_match_native",
            "merit_matches_proposal", "face_audit_passed", "branch_pair_passed",
            "trace_matches_proposal", "source_unchanged", "fixed_input_unchanged",
            "runtime_unchanged", "deadline_passed")}}
    return proposal, repeat


def test_latest_local_archive_and_closed_base_are_read_only_evidence():
    archived = commit._verify_latest_local_archive({})
    base = local._archive_identity()

    assert archived["raw_sha256"] == commit.LOCAL_RAW_SHA
    assert archived["candidate"]["alpha"] == commit.ALPHA
    assert archived["candidate"]["actual_J_armijo_passed"] is True
    assert archived["candidate"]["actual_F2_armijo_passed"] is True
    assert commit.POLICY["root_claim"] is False
    assert tangent._tensor_sha(base["control"]) == commit.BASE_CONTROL_SHA
    assert set(base["stored_hvp_history"]) == {-1, 1}


def test_gradient_cancellation_does_not_close_the_fresh_side_pair(tmp_path):
    proposal, repeat = _closed_pair()
    repeat["side_gradients"] = {"-1": [0.0] * 26, "1": [0.0] * 26}
    path = tmp_path / "step.json"
    base = {"current_control_sha256": commit.BASE_CONTROL_SHA,
        "current_theta": commit.BASE_THETA, "optimizer_steps_applied": 0,
        "candidate_committed": False}
    commit._atomic_write(path, base)

    with pytest.raises(ValueError, match="P2 final closure"):
        commit._persist_closed_commit(path, base, proposal, repeat)

    assert json.loads(path.read_text()) == base


def test_failed_independent_repeat_keeps_the_original_endpoint(tmp_path):
    proposal, repeat = _closed_pair()
    repeat["runtime_unchanged"] = False
    path = tmp_path / "step.json"
    base = {"current_control": [0.0] * 26,
        "current_control_sha256": commit.BASE_CONTROL_SHA, "current_theta": commit.BASE_THETA,
        "last_confirmed_control_sha256": commit.BASE_CONTROL_SHA,
        "last_confirmed_theta": commit.BASE_THETA, "optimizer_steps_applied": 0,
        "candidate_committed": False}
    commit._atomic_write(path, base)

    with pytest.raises(ValueError, match="P2 final closure"):
        commit._persist_closed_commit(path, base, proposal, repeat)

    assert json.loads(path.read_text()) == base


def test_commit_write_failure_preserves_the_previous_zero_count_record(tmp_path, monkeypatch):
    proposal, repeat = _closed_pair()
    path = tmp_path / "step.json"
    base = {"current_control": [0.0] * 26,
        "current_control_sha256": commit.BASE_CONTROL_SHA, "current_theta": commit.BASE_THETA,
        "last_confirmed_control_sha256": commit.BASE_CONTROL_SHA,
        "last_confirmed_theta": commit.BASE_THETA, "optimizer_steps_applied": 0,
        "candidate_committed": False}
    commit._atomic_write(path, base)

    def fail_write(_path: Path, _value: dict[str, Any]) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(commit, "_atomic_write", fail_write)
    with pytest.raises(OSError, match="atomic replace"):
        commit._persist_closed_commit(path, base, proposal, repeat)

    assert json.loads(path.read_text()) == base
    assert base["optimizer_steps_applied"] == 0
    assert base["candidate_committed"] is False


def test_closed_commit_is_atomic_and_postcommit_failure_preserves_it(tmp_path):
    proposal, repeat = _closed_pair()
    path = tmp_path / "step.json"
    base = {"current_control": [0.0] * 26,
        "current_control_sha256": commit.BASE_CONTROL_SHA, "current_theta": commit.BASE_THETA,
        "last_confirmed_control_sha256": commit.BASE_CONTROL_SHA,
        "last_confirmed_theta": commit.BASE_THETA, "optimizer_steps_applied": 0,
        "candidate_committed": False}
    commit._atomic_write(path, base)

    committed = commit._persist_closed_commit(path, base, proposal, repeat)
    assert committed["candidate_committed"] is True
    assert committed["optimizer_steps_applied"] == 1
    assert committed["current_control_sha256"] == tangent._tensor_sha(torch.tensor(
        proposal["control"], dtype=torch.float64))
    failed = commit._persist_readiness_failure(path, committed, RuntimeError("row budget"))
    disk = json.loads(path.read_text())

    assert failed["candidate_committed"] is True
    assert failed["current_control_sha256"] != commit.BASE_CONTROL_SHA
    assert failed["readiness_complete"] is False
    assert disk["last_confirmed_control_sha256"] == failed["current_control_sha256"]
    assert disk["proposal"] == proposal
    assert disk["final_repeat"] == repeat


def test_postcommit_hvps_must_bind_to_the_new_control_and_direction():
    control = torch.linspace(0.0, 1.0, 26, dtype=torch.float64)
    direction = torch.linspace(-1.0, 1.0, 26, dtype=torch.float64)
    control_sha, direction_sha = tangent._tensor_sha(control), tangent._tensor_sha(direction)
    record: dict[str, Any] = {"jacobian_rows_started": 24, "jacobian_rows_completed": 24,
        "dense_solves_started": 1, "dense_solves_completed": 1,
        "hvp_calls_started": 2, "hvp_calls_completed": 2, "current_theta": 0.5,
        "jacobian_row_history": [{"side": side, "row": row,
            "base_control_sha256": control_sha, "theta": 0.6,
            "residual_value": 0.25, "gradient": [0.0] * 26, "status": "completed"}
            for side in (-1, 1) for row in range(12)],
        "postcommit_gn_readiness": {"working_theta": 0.6, "direction": direction.tolist(),
            "direction_sha256": direction_sha, "direction_model": "robust_gn_coupled"},
        "dense_solve_audit": {"dense_solves": 1, "dense_solve_dimension": 12,
            "S_dimension": 12, "positive_definite_from_cholesky": True,
            "solve_residual": 1e-16, "solve_residual_budget": 1e-14,
            "direction": direction.tolist()},
        "hvp_history": [{"side": side, "base_control_sha256": control_sha,
            "direction_sha256": direction_sha, "theta": 0.5, "working_theta": 0.6,
            "operator": "selected_face_extension", "direction_model": "robust_gn_coupled",
            "status": "completed"} for side in (-1, 1)],
        "hvp_started_history": [{"phase": "postcommit_current_point", "side": side,
            "base_control_sha256": control_sha, "direction_sha256": direction_sha,
            "theta": 0.5, "working_theta": 0.6, "operator": "selected_face_extension",
            "direction_model": "robust_gn_coupled"} for side in (-1, 1)]}

    commit._require_postcommit_counts(record, control, direction)
    record["hvp_history"][0]["base_control_sha256"] = commit.BASE_CONTROL_SHA
    with pytest.raises(ValueError, match="point-direction linkage"):
        commit._require_postcommit_counts(record, control, direction)


def test_postcommit_rows_and_dense_solve_must_close_at_the_committed_point():
    control = torch.linspace(0.0, 1.0, 26, dtype=torch.float64)
    direction = torch.linspace(-1.0, 1.0, 26, dtype=torch.float64)
    control_sha, direction_sha = tangent._tensor_sha(control), tangent._tensor_sha(direction)
    record: dict[str, Any] = {"jacobian_rows_started": 24, "jacobian_rows_completed": 24,
        "dense_solves_started": 1, "dense_solves_completed": 1,
        "hvp_calls_started": 2, "hvp_calls_completed": 2, "current_theta": 0.5,
        "jacobian_row_history": [{"side": side, "row": row,
            "base_control_sha256": control_sha, "theta": 0.6,
            "residual_value": 0.25, "gradient": [0.0] * 26, "status": "completed"}
            for side in (-1, 1) for row in range(12)],
        "postcommit_gn_readiness": {"working_theta": 0.6, "direction": direction.tolist(),
            "direction_sha256": direction_sha, "direction_model": "robust_gn_coupled"},
        "dense_solve_audit": {"dense_solves": 1, "dense_solve_dimension": 12,
            "S_dimension": 12, "positive_definite_from_cholesky": True,
            "solve_residual": 1e-16, "solve_residual_budget": 1e-14,
            "direction": direction.tolist()},
        "hvp_history": [{"side": side, "base_control_sha256": control_sha,
            "direction_sha256": direction_sha, "theta": 0.5, "working_theta": 0.6,
            "operator": "selected_face_extension", "direction_model": "robust_gn_coupled",
            "status": "completed"} for side in (-1, 1)],
        "hvp_started_history": [{"phase": "postcommit_current_point", "side": side,
            "base_control_sha256": control_sha, "direction_sha256": direction_sha,
            "theta": 0.5, "working_theta": 0.6, "operator": "selected_face_extension",
            "direction_model": "robust_gn_coupled"} for side in (-1, 1)]}

    commit._require_postcommit_counts(record, control, direction)
    record["jacobian_row_history"][0]["base_control_sha256"] = commit.BASE_CONTROL_SHA
    with pytest.raises(ValueError, match="row/solve/HVP budgets"):
        commit._require_postcommit_counts(record, control, direction)
    record["jacobian_row_history"][0]["base_control_sha256"] = control_sha
    record["dense_solve_audit"]["solve_residual"] = 1.0
    with pytest.raises(ValueError, match="row/solve/HVP budgets"):
        commit._require_postcommit_counts(record, control, direction)
    record["dense_solve_audit"]["solve_residual"] = 1e-16
    record["dense_solve_audit"]["direction"][0] += 0.01
    with pytest.raises(ValueError, match="point-direction linkage"):
        commit._require_postcommit_counts(record, control, direction)
    record["dense_solve_audit"]["direction"] = direction.tolist()
    record["hvp_history"][0]["direction_sha256"] = commit.BASE_CONTROL_SHA
    with pytest.raises(ValueError, match="point-direction linkage"):
        commit._require_postcommit_counts(record, control, direction)
    record["hvp_history"][0]["direction_sha256"] = direction_sha
    record["hvp_started_history"][0]["direction_sha256"] = commit.BASE_CONTROL_SHA
    with pytest.raises(ValueError, match="started HVP labels"):
        commit._require_postcommit_counts(record, control, direction)
    record["hvp_started_history"][0]["direction_sha256"] = direction_sha
    record["postcommit_gn_readiness"]["direction_sha256"] = commit.BASE_CONTROL_SHA
    with pytest.raises(ValueError, match="point-direction linkage"):
        commit._require_postcommit_counts(record, control, direction)
