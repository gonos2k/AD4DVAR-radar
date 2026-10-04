"""Toy-only coordinator durability tests; no FV or HVP execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.weather_scenarios import fv_checkpoint_experiment as experiment
from tests.test_fv_point_3h_endpoint_diagnostics import _make_toy_archive


def _setup(tmp_path: Path, monkeypatch):
    archive, _diagnostic_plan, _unused, raw = _make_toy_archive(tmp_path, monkeypatch)
    root = experiment.diagnostic.ROOT
    evidence = experiment.diagnostic.EVIDENCE
    plan = evidence / "RESUMABLE_HESSIAN_IMPLEMENTATION_PLAN_20261004.md"
    plan.write_text("toy frozen source plan")
    source = root / "examples/weather_scenarios/fv_checkpoint_experiment.py"
    test = root / "tests/test_fv_checkpoint_experiment.py"
    source.parent.mkdir(parents=True, exist_ok=True); source.write_text("toy coordinator source")
    test.parent.mkdir(parents=True, exist_ok=True); test.write_text("toy coordinator test")
    monkeypatch.setattr(experiment, "ROOT", root)
    monkeypatch.setattr(experiment, "EVIDENCE", evidence)
    monkeypatch.setattr(experiment, "ARCHIVE", archive)
    monkeypatch.setattr(experiment, "PLAN", plan)
    monkeypatch.setattr(experiment, "ARCHIVE_SHA", experiment._sha(archive))
    monkeypatch.setattr(experiment, "PARENT_SHA", experiment._sha(archive.with_suffix(".run.json")))
    monkeypatch.setattr(experiment, "RESOURCE_SHA", experiment._sha(archive.with_suffix(".resource.json")))
    monkeypatch.setattr(experiment, "PLAN_SHA", experiment._sha(plan))
    monkeypatch.setattr(experiment, "CONTROL_SHA", raw["last_accepted_control_sha256"])
    monkeypatch.setattr(experiment, "PARAMETERS_SHA", raw["parameters_sha256"])
    monkeypatch.setattr(experiment, "SOURCE_PINS", {})
    monkeypatch.setattr(experiment, "__file__", str(source))
    return archive, plan, evidence, raw


def _fake_guard_result(output: Path, raw: dict[str, Any], *, execution: str,
                       numerical: str, child_read_error: str | None,
                       resource_elapsed: float = 12.0, write_parent: bool = True,
                       child_body: dict[str, Any] | None = None) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    resource = {"elapsed_seconds": resource_elapsed, "exit_code": 0 if execution == "completed" else 1,
                "resource_termination": None, "monitor_error": None}
    parent = {"execution_status": execution, "numerical_status": numerical,
              "child_read_error": child_read_error, "resource": resource,
              "endpoint_identity_matches_archive": execution == "completed",
              "source_unchanged": execution == "completed"}
    output.with_suffix(".resource.json").write_text(json.dumps(resource))
    if child_body is None:
        output.write_text("{malformed child output")
    else:
        output.write_text(json.dumps(child_body))
        parent["child_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        parent["checkpoint"] = child_body.get("checkpoint", {"status": "not_recorded"})
        parent["child_terminal_phase"] = child_body.get("phase")
    if write_parent:
        output.with_suffix(".run.json").write_text(json.dumps(parent))
    return parent


def _run(directory: Path) -> dict[str, Any]:
    return experiment.run_checkpoint_experiment(directory=directory, archive=experiment.ARCHIVE,
        archive_sha256=experiment.ARCHIVE_SHA, parent_sha256=experiment.PARENT_SHA,
        resource_sha256=experiment.RESOURCE_SHA, plan=experiment.PLAN, plan_sha256=experiment.PLAN_SHA)


def test_malformed_child_is_durable_failure_and_does_not_retry(tmp_path: Path, monkeypatch):
    _archive, _plan, evidence, _raw = _setup(tmp_path, monkeypatch)
    calls = 0

    def fake_guarded(**kwargs):
        nonlocal calls
        calls += 1
        return _fake_guard_result(kwargs["output"], _raw, execution="failed",
            numerical="not_reached", child_read_error="child report JSON was malformed")

    monkeypatch.setattr(experiment.diagnostic, "guarded_run", fake_guarded)
    ledger = _run(evidence / "experiment")
    saved = json.loads((evidence / "experiment/experiment.json").read_text())
    row = saved["attempts"][0]
    assert calls == 1
    assert saved["phase"] == "finished" and row["status"] == "failed"
    assert row["child_read_error"] == "child report JSON was malformed"
    assert row["parent_record_sha256"] and row["resource_record_sha256"]
    assert saved["stop_reason"] == "execution/resource/integrity failure; no retry"
    assert ledger == saved


def test_missing_parent_artifact_downgrades_success_and_is_written(tmp_path: Path, monkeypatch):
    _archive, _plan, evidence, _raw = _setup(tmp_path, monkeypatch)

    def fake_guarded(**kwargs):
        return _fake_guard_result(kwargs["output"], _raw, execution="completed",
            numerical="endpoint_diagnostic_completed", child_read_error=None, write_parent=False)

    monkeypatch.setattr(experiment.diagnostic, "guarded_run", fake_guarded)
    ledger = _run(evidence / "experiment")
    row = ledger["attempts"][0]
    assert row["status"] == "failed"
    assert row["missing_artifacts"] == [row["parent_record"]]
    assert ledger["stop_reason"] == "execution/resource/integrity failure; no retry"
    assert ledger["phase"] == "finished"


def test_completed_parent_with_unreadable_child_is_downgraded_durably(tmp_path: Path, monkeypatch):
    _archive, _plan, evidence, _raw = _setup(tmp_path, monkeypatch)

    def fake_guarded(**kwargs):
        return _fake_guard_result(kwargs["output"], _raw, execution="completed",
            numerical="endpoint_diagnostic_completed", child_read_error=None)

    monkeypatch.setattr(experiment.diagnostic, "guarded_run", fake_guarded)
    ledger = _run(evidence / "experiment")
    row = ledger["attempts"][0]
    assert row["status"] == "failed"
    assert row["child_read_error"]
    assert "readable child report" in row["failure"]
    assert row["resource_record_sha256"]
    assert ledger["phase"] == "finished"
    assert ledger["stop_reason"] == "execution/resource/integrity failure; no retry"


def test_budget_checkpoint_progress_resumes_then_stops_on_completion(tmp_path: Path, monkeypatch):
    _archive, _plan, evidence, raw = _setup(tmp_path, monkeypatch)
    calls = 0

    def fake_guarded(**kwargs):
        nonlocal calls
        calls += 1
        output = kwargs["output"]
        numerical = "checkpoint_budget_refusal" if calls == 1 else "endpoint_diagnostic_completed"
        status = "audit_pending" if calls == 1 else "completed"
        child = {"phase": "diagnostic_refused" if calls == 1 else "finished",
                 "numerical_status": numerical, "checkpoint": {"status": status,
                    "completed_columns": 26, "hvp_calls_total": 27, "attempts_reserved": calls}}
        record = _fake_guard_result(output, raw, execution="completed", numerical=numerical,
            child_read_error=None, resource_elapsed=240.0 if calls == 1 else 12.0, child_body=child)
        return record

    monkeypatch.setattr(experiment.diagnostic, "guarded_run", fake_guarded)
    ledger = _run(evidence / "experiment")
    assert calls == 2
    assert len(ledger["attempts"]) == 2
    assert ledger["attempts"][0]["checkpoint_progress"]["status"] == "audit_pending"
    assert ledger["attempts"][1]["checkpoint_progress"]["status"] == "completed"
    assert ledger["stop_reason"] == "fresh curvature diagnostic completed"
    assert ledger["reserved_outer_seconds"] == 600


def test_three_budget_refusals_reserve_but_never_exceed_900_seconds(tmp_path: Path, monkeypatch):
    _archive, _plan, evidence, raw = _setup(tmp_path, monkeypatch)
    calls = 0

    def fake_guarded(**kwargs):
        nonlocal calls
        calls += 1
        checkpoint = {"status": "columns_pending", "completed_columns": 5 * calls,
                      "hvp_calls_total": 5 * calls, "attempts_reserved": calls}
        child = {"phase": "diagnostic_refused", "numerical_status": "checkpoint_budget_refusal",
                 "checkpoint": checkpoint}
        record = _fake_guard_result(kwargs["output"], raw, execution="completed",
            numerical="checkpoint_budget_refusal", child_read_error=None, resource_elapsed=299.0,
            child_body=child)
        return record

    monkeypatch.setattr(experiment.diagnostic, "guarded_run", fake_guarded)
    ledger = _run(evidence / "experiment")
    assert calls == 3 and len(ledger["attempts"]) == 3
    assert ledger["reserved_outer_seconds"] == 900
    assert ledger["stop_reason"] == "declared three-launch limit exhausted"
    assert ledger["elapsed_seconds"] <= 900
