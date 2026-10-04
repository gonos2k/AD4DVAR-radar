"""Toy-only checkpoint/resume tests; no FV, HVP workload beyond tiny quadratics."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import pytest
import torch
from torch import Tensor

from examples.weather_scenarios import fv_hessian_checkpoint as checkpoint


def _toy_problem():
    matrix = torch.diag(torch.linspace(1.0, 2.0, checkpoint.NC, dtype=torch.float64))
    control = torch.linspace(-0.3, 0.4, checkpoint.NC, dtype=torch.float64)
    parameters = torch.linspace(-0.2, 0.2, 13, dtype=torch.float64)
    gradient = matrix @ control

    def objective(value: Tensor, params: Tensor) -> Tensor:
        return torch.dot(value, matrix @ value) / 2 + torch.dot(params, params) * 0

    context = {
        "objective": {"sha256": "a" * 64, "name": "quadratic toy"},
        "problem": {"sha256": "b" * 64},
        "coordinates": {"sha256": "c" * 64, "dimension": checkpoint.NC},
        "source": {"sha256": "d" * 64},
        "runtime": {"device": "CPU FP64", "toy": True},
        "branch": {"signature_sha256": "e" * 64},
        "analysis_context": {"scope": "synthetic checkpoint test"},
    }

    def receipt_provider() -> dict[str, Any]:
        return context

    return objective, control, parameters, gradient, context, receipt_provider


def _call(directory: Path, *, max_attempts: int = 3, max_new_columns: int | None = None,
          attempt_seconds: float = 10, receipt_provider=None, parameters: Tensor | None = None,
          objective_fn=None):
    base_objective, control, base_parameters, gradient, _context, base_provider = _toy_problem()
    return checkpoint.checkpoint_hessian(
        base_objective if objective_fn is None else objective_fn, control,
        base_parameters if parameters is None else parameters, gradient,
        receipt_provider=base_provider if receipt_provider is None else receipt_provider,
        checkpoint_dir=directory, deadline=time.monotonic() + 60,
        max_attempts=max_attempts, attempt_seconds=attempt_seconds,
        max_new_columns=max_new_columns,
    )


def test_interrupted_basis_work_resumes_7_12_7_without_recomputation(tmp_path: Path):
    directory = tmp_path / "checkpoint"
    _objective, _control, _parameters, _gradient, context, _ = _toy_problem()
    receipt_calls = 0

    def counted_receipt():
        nonlocal receipt_calls
        receipt_calls += 1
        return context

    for cap, expected in ((7, 7), (12, 19)):
        with pytest.raises(checkpoint.CheckpointBudgetRefusal) as exc:
            _call(directory, max_new_columns=cap, receipt_provider=counted_receipt)
        assert exc.value.state["completed_columns"] == expected
        state = json.loads((directory / "hessian_checkpoint.json").read_text())
        assert len(state["columns"]) == expected
        assert all(row["receipt_before"] == row["receipt_after"] == state["header"]["receipt"]
                   for row in state["columns"])
        assert len(state["attempts"]) == (1 if expected == 7 else 2)
        assert all(item["reserved_seconds"] == 10 for item in state["attempts"])
        assert not (directory / ".writer.lock").exists()

    result = _call(directory, max_new_columns=7, receipt_provider=counted_receipt)
    assert result["hvp_columns"] == checkpoint.NC and result["hvp_calls_total"] == checkpoint.NC + 1
    assert result["block_schur"]["status"] == "schur_evaluated"
    assert result["attempts_reserved"] == 3
    assert len(json.loads((directory / "hessian_checkpoint.json").read_text())["columns"]) == checkpoint.NC
    # Three initial receipts plus two per persisted basis/audit HVP.
    assert receipt_calls == 3 + 2 * (checkpoint.NC + 1)
    assert not (directory / ".writer.lock").exists()


def test_checkpoint_identity_change_refuses_without_committing_hvp(tmp_path: Path):
    directory = tmp_path / "checkpoint"
    with pytest.raises(checkpoint.CheckpointBudgetRefusal):
        _call(directory, max_new_columns=1)
    state_before = json.loads((directory / "hessian_checkpoint.json").read_text())
    bytes_before = (directory / "hessian_checkpoint.json").read_bytes()
    _objective, _control, parameters, _gradient, _context, provider = _toy_problem()
    changed_parameters = parameters.clone()
    changed_parameters[0] += 0.01
    with pytest.raises(checkpoint.ReceiptRefusal) as exc:
        _call(directory, parameters=changed_parameters)
    assert exc.value.state["completed_columns"] == 1
    assert json.loads((directory / "hessian_checkpoint.json").read_text())["columns"] == state_before["columns"]
    assert (directory / "hessian_checkpoint.json").read_bytes() == bytes_before


def test_changed_objective_with_new_attestation_refuses_old_cache(tmp_path: Path):
    directory = tmp_path / "checkpoint"
    with pytest.raises(checkpoint.CheckpointBudgetRefusal):
        _call(directory, max_new_columns=1)
    path = directory / "hessian_checkpoint.json"
    original_bytes = path.read_bytes()
    base_objective, _control, _parameters, _gradient, context, _ = _toy_problem()

    def changed_objective(value: Tensor, params: Tensor) -> Tensor:
        return base_objective(value, params) + value.sum() * 0.125

    changed_context = dict(context)
    changed_context["objective"] = {"sha256": "f" * 64, "name": "changed toy objective"}
    changed_context["source"] = {"sha256": "9" * 64}
    with pytest.raises(checkpoint.ReceiptRefusal):
        _call(directory, objective_fn=changed_objective,
              receipt_provider=lambda: changed_context)
    assert path.read_bytes() == original_bytes


def test_deadline_after_column_26_leaves_only_independent_audit_pending(
    tmp_path: Path, monkeypatch,
):
    directory = tmp_path / "checkpoint"
    clock = [100.0]
    monkeypatch.setattr(checkpoint.time, "monotonic", lambda: clock[0])
    real_hvp = checkpoint._apply_hvp
    calls = 0

    def expire_after_26th_hvp(*args):
        nonlocal calls
        calls += 1
        product = real_hvp(*args)
        if calls == checkpoint.NC:
            clock[0] = 200.0
        return product

    monkeypatch.setattr(checkpoint, "_apply_hvp", expire_after_26th_hvp)
    with pytest.raises(checkpoint.BudgetRefusal) as exc:
        _call(directory)
    assert exc.value.state["status"] == "audit_pending"
    assert exc.value.state["completed_columns"] == checkpoint.NC
    assert exc.value.state["hvp_calls_total"] == checkpoint.NC
    saved = json.loads((directory / "hessian_checkpoint.json").read_text())
    assert len(saved["columns"]) == checkpoint.NC
    assert saved["audit_hvp"] is None
    assert "final_result" not in saved

    result = _call(directory)
    assert calls == checkpoint.NC + 1
    assert result["hvp_calls_total"] == checkpoint.NC + 1
    assert result["checkpoint_status"] == "completed"


def test_audit_hvp_owner_is_bound_to_its_reservation(tmp_path: Path, monkeypatch):
    directory = tmp_path / "checkpoint"
    clock = [100.0]
    monkeypatch.setattr(checkpoint.time, "monotonic", lambda: clock[0])
    real_hvp = checkpoint._apply_hvp
    calls = 0

    def expire_after_audit_hvp(*args):
        nonlocal calls
        calls += 1
        product = real_hvp(*args)
        if calls == checkpoint.NC + 1:
            clock[0] = 200.0
        return product

    monkeypatch.setattr(checkpoint, "_apply_hvp", expire_after_audit_hvp)
    with pytest.raises(checkpoint.BudgetRefusal):
        _call(directory)
    first = json.loads((directory / "hessian_checkpoint.json").read_text())
    assert first["audit_hvp"]["attempt_index"] == 1
    assert first["attempts"][0]["status"] == "budget_refused"

    result = _call(directory)
    assert result["checkpoint_status"] == "completed"
    assert result["attempts_reserved"] == 2
    assert json.loads((directory / "hessian_checkpoint.json").read_text())["audit_hvp"]["attempt_index"] == 1
    path = directory / "hessian_checkpoint.json"
    state = json.loads(path.read_text())
    state["audit_hvp"]["attempt_index"] = 2
    path.write_text(json.dumps(state))
    corrupt_bytes = path.read_bytes()
    monkeypatch.setattr(checkpoint, "_apply_hvp", lambda *_args: pytest.fail("must reject before HVP"))
    with pytest.raises(ValueError, match="independent HVP is malformed"):
        _call(directory)
    assert path.read_bytes() == corrupt_bytes
    assert calls == checkpoint.NC + 1


def test_unexpected_hvp_error_is_terminal_and_retry_does_not_run(tmp_path: Path,
                                                                 monkeypatch):
    directory = tmp_path / "checkpoint"
    calls = 0

    def fail_unexpectedly(*_args):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic programming error")

    monkeypatch.setattr(checkpoint, "_apply_hvp", fail_unexpectedly)
    with pytest.raises(RuntimeError, match="synthetic programming error"):
        _call(directory)
    assert calls == 1
    path = directory / "hessian_checkpoint.json"
    terminal_bytes = path.read_bytes()
    terminal = json.loads(terminal_bytes)
    assert terminal["status"] == "programming_error"
    assert terminal["attempts"][-1]["status"] == "programming_error"
    assert terminal["attempts"][-1]["exception_type"] == "RuntimeError"

    monkeypatch.setattr(checkpoint, "_apply_hvp", lambda *_args: pytest.fail("must not retry"))
    with pytest.raises(ValueError, match="terminal programming error"):
        _call(directory)
    assert path.read_bytes() == terminal_bytes
    assert calls == 1


def test_mismatched_query_preserves_completed_checkpoint_and_correct_reuse(tmp_path: Path,
                                                                           monkeypatch):
    directory = tmp_path / "checkpoint"
    for cap in (7, 12, 7):
        try:
            _call(directory, max_new_columns=cap)
        except checkpoint.CheckpointBudgetRefusal:
            pass
    path = directory / "hessian_checkpoint.json"
    original_bytes = path.read_bytes()
    _objective, _control, _parameters, _gradient, context, _provider = _toy_problem()

    def wrong_provider():
        observed = dict(context)
        observed["source"] = {"sha256": "f" * 64}
        return observed

    with pytest.raises(checkpoint.ReceiptRefusal) as exc:
        _call(directory, receipt_provider=wrong_provider)
    assert exc.value.state["status"] == "receipt_changed"
    assert "observed_receipt" in exc.value.state
    assert path.read_bytes() == original_bytes

    def unexpected_hvp(*_args):
        raise AssertionError("verified completed checkpoint should be reused without an HVP")

    monkeypatch.setattr(checkpoint, "_apply_hvp", unexpected_hvp)
    result = _call(directory)
    assert result["checkpoint_status"] == "completed"
    assert path.read_bytes() == original_bytes
    assert json.loads(original_bytes)["last_status"] == "completed"


def test_receipt_change_across_hvp_does_not_commit_that_product(tmp_path: Path, monkeypatch):
    directory = tmp_path / "checkpoint"
    _objective, _control, _parameters, _gradient, context, _provider = _toy_problem()
    calls = 0

    def changing_receipt():
        nonlocal calls
        calls += 1
        current = dict(context)
        if calls >= 3:
            current["source"] = {"sha256": "f" * 64}
        return current

    with pytest.raises(checkpoint.ReceiptRefusal) as exc:
        _call(directory, receipt_provider=changing_receipt)
    saved = json.loads((directory / "hessian_checkpoint.json").read_text())
    assert exc.value.state["completed_columns"] == 0
    assert saved["columns"] == []
    assert saved["attempts"][0]["reserved_seconds"] == 10
    path = directory / "hessian_checkpoint.json"
    refused_bytes = path.read_bytes()

    def no_retry(*_args):
        pytest.fail("receipt-refused checkpoint must be terminal")

    monkeypatch.setattr(checkpoint, "_apply_hvp", no_retry)
    with pytest.raises(ValueError, match="terminal after a non-budget refusal"):
        _call(directory)
    assert path.read_bytes() == refused_bytes
    altered = json.loads(refused_bytes)
    altered["status"] = altered["last_status"] = "columns_pending"
    path.write_text(json.dumps(altered))
    altered_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="status disagrees with the latest attempt"):
        _call(directory)
    assert path.read_bytes() == altered_bytes


def test_numerical_refusal_is_terminal_and_preserves_cache(tmp_path: Path, monkeypatch):
    directory = tmp_path / "checkpoint"
    calls = 0

    def invalid_hvp(*_args):
        nonlocal calls
        calls += 1
        return torch.full((checkpoint.NC,), float("nan"), dtype=torch.float64)

    monkeypatch.setattr(checkpoint, "_apply_hvp", invalid_hvp)
    with pytest.raises(checkpoint.NumericalRefusal):
        _call(directory)
    assert calls == 1
    path = directory / "hessian_checkpoint.json"
    refused_bytes = path.read_bytes()
    saved = json.loads(refused_bytes)
    assert saved["status"] == "invalid_hvp_product"
    monkeypatch.setattr(checkpoint, "_apply_hvp", lambda *_args: pytest.fail("must not retry"))
    with pytest.raises(ValueError, match="terminal after a non-budget refusal"):
        _call(directory)
    assert path.read_bytes() == refused_bytes
    assert calls == 1


def test_corrupt_column_is_rejected_and_never_dropped(tmp_path: Path):
    directory = tmp_path / "checkpoint"
    with pytest.raises(checkpoint.CheckpointBudgetRefusal):
        _call(directory, max_new_columns=1)
    path = directory / "hessian_checkpoint.json"
    state = json.loads(path.read_text())
    state["columns"][0]["product_sha256"] = "0" * 64
    path.write_text(json.dumps(state))
    corrupt = path.read_bytes()
    with pytest.raises(ValueError, match="checkpoint HVP column is malformed"):
        _call(directory, max_new_columns=1)
    assert path.read_bytes() == corrupt


def test_stale_writer_lock_is_not_removed_automatically(tmp_path: Path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    lock = directory / ".writer.lock"
    lock.write_text("stale owner\n")
    with pytest.raises(FileExistsError):
        _call(directory)
    assert lock.read_text() == "stale owner\n"


def test_attempt_quota_is_nonrefundable_across_resumes(tmp_path: Path):
    directory = tmp_path / "checkpoint"
    with pytest.raises(checkpoint.CheckpointBudgetRefusal):
        _call(directory, max_attempts=1, max_new_columns=1)
    before = json.loads((directory / "hessian_checkpoint.json").read_text())
    with pytest.raises(checkpoint.CheckpointBudgetRefusal, match="attempt quota"):
        _call(directory, max_attempts=1, max_new_columns=1)
    after = json.loads((directory / "hessian_checkpoint.json").read_text())
    assert len(after["attempts"]) == 1
    assert after["columns"] == before["columns"]


@pytest.mark.parametrize("damage", ["clear_ledger", "truncate_ledger", "orphan_anchor",
                                    "truncate_anchor", "wrong_column_owner"])
def test_independent_attempt_reservations_reject_ledger_edits(tmp_path: Path, monkeypatch,
                                                               damage: str):
    directory = tmp_path / "checkpoint"
    for _ in range(2):
        with pytest.raises(checkpoint.CheckpointBudgetRefusal):
            _call(directory, max_new_columns=1)
    path = directory / "hessian_checkpoint.json"
    state = json.loads(path.read_text())
    if damage == "clear_ledger":
        state["attempts"] = []
    elif damage == "truncate_ledger":
        state["attempts"].pop()
    elif damage == "orphan_anchor":
        (directory / "reservation_000003.json").write_text("{}\n")
    elif damage == "truncate_anchor":
        (directory / "reservation_000001.json").write_text("{}\n")
    else:
        state["columns"][0]["attempt_index"] = 2
    if damage in {"clear_ledger", "truncate_ledger", "wrong_column_owner"}:
        path.write_text(json.dumps(state))
    damaged_checkpoint = path.read_bytes()
    damaged_reservations = {p.name: p.read_bytes() for p in directory.glob("reservation_*.json")}
    monkeypatch.setattr(checkpoint, "_apply_hvp", lambda *_args: pytest.fail("must refuse before HVP"))
    with pytest.raises(ValueError, match="reservation|ledger"):
        _call(directory)
    assert path.read_bytes() == damaged_checkpoint
    assert {p.name: p.read_bytes() for p in directory.glob("reservation_*.json")} == damaged_reservations


@pytest.mark.parametrize(
    ("changed_policy", "new_value"),
    [("max_attempts", 3), ("attempt_seconds", 11)],
)
def test_attempt_budget_header_cannot_be_changed_after_progress(
    tmp_path: Path, monkeypatch, changed_policy: str, new_value: int,
):
    directory = tmp_path / "checkpoint"
    for _ in range(2):
        with pytest.raises(checkpoint.CheckpointBudgetRefusal):
            _call(directory, max_attempts=2, max_new_columns=1, attempt_seconds=10)
    path = directory / "hessian_checkpoint.json"
    original_bytes = path.read_bytes()
    saved = json.loads(original_bytes)
    assert len(saved["attempts"]) == 2
    hvp_calls = 0

    def count_hvp(*args):
        nonlocal hvp_calls
        hvp_calls += 1
        return torch.zeros(checkpoint.NC, dtype=torch.float64)

    monkeypatch.setattr(checkpoint, "_apply_hvp", count_hvp)
    with pytest.raises(ValueError, match="attempt-budget header changed"):
        if changed_policy == "max_attempts":
            _call(directory, max_attempts=new_value, max_new_columns=1, attempt_seconds=10)
        else:
            _call(directory, max_attempts=2, max_new_columns=1, attempt_seconds=new_value)
    assert hvp_calls == 0
    assert path.read_bytes() == original_bytes


def test_hvp_direction_argument_mutation_refuses_without_column_commit(tmp_path: Path,
                                                                       monkeypatch):
    directory = tmp_path / "checkpoint"
    def mutate_direction(_objective, _parameters, _control, direction):
        direction[0] += 1
        return torch.ones(checkpoint.NC, dtype=torch.float64)
    monkeypatch.setattr(checkpoint, "_apply_hvp", mutate_direction)
    with pytest.raises(checkpoint.ReceiptRefusal):
        _call(directory, max_new_columns=1)
    saved = json.loads((directory / "hessian_checkpoint.json").read_text())
    assert saved["columns"] == []


def test_completed_checkpoint_is_verified_before_reuse(tmp_path: Path):
    directory = tmp_path / "checkpoint"
    for cap in (7, 12, 7):
        try:
            result = _call(directory, max_new_columns=cap)
        except checkpoint.CheckpointBudgetRefusal:
            continue
    assert result["checkpoint_status"] == "completed"
    path = directory / "hessian_checkpoint.json"
    state = json.loads(path.read_text())
    state["final_result"]["hessian_sha256"] = "0" * 64
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="completed result differs from its committed Hessian columns"):
        _call(directory)
