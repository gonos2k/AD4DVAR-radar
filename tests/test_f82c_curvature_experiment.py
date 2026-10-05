"""Toy orchestration tests; the guard and FV child are replaced explicitly."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "graphify-out/fv-root-cause-20260919/F82C_CURVATURE_RUN_EXPERIMENT_20261005.py"
spec = importlib.util.spec_from_file_location("f82c_experiment", SCRIPT)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def run_toy(tmp_path, monkeypatch, statuses):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    evidence = tmp_path / "graphify-out/fv-root-cause-20260919"
    evidence.mkdir(parents=True)
    plan = evidence / "plan.json"
    plan.write_text(json.dumps({"source_files": {}, "archive_files": {}, "scope": "toy",
                               "policy": {"max_guarded_launches": 3, "per_launch_wall_seconds": 300.,
                                          "total_outer_reserved_seconds": 900.},
                               "control_sha256": "c", "parameters_sha256": "p"}))
    plan_sha = probe.sha(plan)
    output = evidence / "run"
    monkeypatch.setattr(probe.sys, "argv", ["experiment", "--plan", str(plan),
        "--plan-sha256", plan_sha, "--directory", str(output)])
    calls = []

    def guard(command, **kwargs):
        assert "--child" in command
        assert kwargs["wall_seconds"] == 300 and kwargs["rss_bytes"] == 1024**3
        directory = Path(command[command.index("--output-directory") + 1])
        status = statuses[len(calls)]
        calls.append(command)
        resource = {"exit_code": 0, "elapsed_seconds": 1., "resource_termination": None,
                    "monitor_error": None, "child_process_group_cleanup_sent": False,
                    "child_process_group_cleanup_error": None}
        Path(kwargs["report_path"]).write_text(json.dumps(resource))
        if status == "non_object":
            (directory / "audit.json").write_text("[]")
        else:
            (directory / "audit.json").write_text(json.dumps({"phase": "finished", "source_unchanged": True,
                "fixed_input_unchanged": True, "runtime": {}, "runtime_after": {}, "control_sha256": "c",
                "parameters_sha256": "p", "plan_sha256": plan_sha, "numerical_status": status,
                "hvp_columns_completed": 26, "hvp_calls_total": 27}))
        return resource

    monkeypatch.setattr(probe, "run_guarded_diagnostic", guard)
    probe.main()
    return output, calls, json.loads((output / "experiment.json").read_text())


def test_only_budget_refusal_resumes_and_completion_stops(tmp_path, monkeypatch):
    output, calls, ledger = run_toy(tmp_path, monkeypatch, ["curvature_budget_refused", "curvature_completed"])
    assert len(calls) == 2 and ledger["reserved_outer_seconds"] == 600
    assert ledger["stop_reason"] == "fresh accepted-point curvature completed"
    assert (output / "attempt_1/audit.run.json").is_file()


def test_three_launch_budget_is_not_extended(tmp_path, monkeypatch):
    _, calls, ledger = run_toy(tmp_path, monkeypatch, ["curvature_budget_refused"] * 3)
    assert len(calls) == 3 and ledger["reserved_outer_seconds"] == 900
    assert ledger["stop_reason"] == "launch quota exhausted"


@pytest.mark.parametrize("status", ["curvature_numerical_refusal", "curvature_receipt_refusal", "non_object"])
def test_nonbudget_or_invalid_child_stops_without_retry(tmp_path, monkeypatch, status):
    output, calls, ledger = run_toy(tmp_path, monkeypatch, [status])
    assert len(calls) == 1 and ledger["reserved_outer_seconds"] == 300
    assert (output / "attempt_1/audit.resource.json").is_file()
    if status == "non_object":
        parent = json.loads((output / "attempt_1/audit.run.json").read_text())
        assert parent["execution_status"] == "failed" and parent["child_read_error"]
