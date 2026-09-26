"""A forward-only 3-hour result cannot be published as a local response."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from examples.weather_scenarios import fv_point_3h_response_policy as policy


def _copy_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    raw = evidence / policy.RAW.name
    shutil.copytree(policy.RAW, raw)
    plan = evidence / policy.PLAN.name
    aggregate = evidence / policy.FORWARD_EVIDENCE.name
    shutil.copyfile(policy.PLAN, plan)
    shutil.copyfile(policy.FORWARD_EVIDENCE, aggregate)
    monkeypatch.setattr(policy, "EVIDENCE", evidence)
    monkeypatch.setattr(policy, "RAW", raw)
    monkeypatch.setattr(policy, "PLAN", plan)
    monkeypatch.setattr(policy, "FORWARD_EVIDENCE", aggregate)
    return raw


def _reseal(raw: Path, filename: str, edit) -> None:
    path = raw / filename
    report = json.loads(path.read_text())
    edit(report)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = raw / "manifest.json"
    pinned = json.loads(manifest.read_text())
    pinned["sha256"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(pinned, indent=2, sort_keys=True) + "\n")


def test_current_strict_branch_refuses_response_but_forward_passes():
    result = policy.classify()
    assert result["numerical_solver_runs"] == 0
    assert result["execution_status"] == "completed"
    assert result["forward_validation"] == "passed_same_operator"
    assert result["stationarity_passed"] == "not_tested"
    assert result["local_branch_supported"] is False
    assert result["response_support_status"] == "unsupported_by_current_strict_branch_gate"
    assert result["response_computed"] is False
    assert result["response_validation"] == "not_performed"
    assert result["physical_validation"] == "not_performed"
    assert result["strict_branch_admitted_stages"] == 0
    assert result["separate_analysis_stages"] == 360
    assert result["terminal_forecast_call_stages"] == 3600
    assert result["stationary_root_absence_proved"] is False


@pytest.mark.parametrize("mutation", [
    "branch_passed", "false_response", "resource_exit", "forward_failed",
    "wrong_stage", "wrong_input", "resource_sidecar", "source_aggregate",
])
def test_resealed_child_cannot_change_response_support_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
):
    raw = _copy_archive(tmp_path, monkeypatch)
    child_name = "point_3h_forward.json"
    parent_name = "point_3h_forward.run.json"
    if mutation == "branch_passed":
        _reseal(raw, child_name, lambda value: value["strict_branch_diagnostic"].update(
            status="passed_pointwise", euler_stages=3600, observed_minmod_stages=3600))
    elif mutation == "false_response":
        _reseal(raw, child_name, lambda value: value.update(response_computed=True,
                                                              response_validation="passed"))
    elif mutation == "resource_exit":
        _reseal(raw, parent_name, lambda value: value["resource"].update(exit_code=1))
    elif mutation == "forward_failed":
        _reseal(raw, child_name, lambda value: value.update(forward_validation="failed"))
    elif mutation == "wrong_stage":
        _reseal(raw, child_name, lambda value: value["terminal"].update(observed_minmod_stages=3240))
    elif mutation == "resource_sidecar":
        _reseal(raw, "point_3h_forward.resource.json",
                lambda value: value.update(exit_code=1))
    elif mutation == "source_aggregate":
        def alter_source(value):
            name = next(iter(value["source_before"]))
            value["source_before"][name] = "a" * 64
            value["source_after"][name] = "a" * 64
        _reseal(raw, child_name, alter_source)
    else:
        def alter(value):
            for key in ("input_before", "input_after"):
                value[key]["control_sha256"] = "a" * 64
        _reseal(raw, child_name, alter)
    monkeypatch.setattr(policy, "RAW_MANIFEST_SHA256",
                        hashlib.sha256((raw / "manifest.json").read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="point 3-hour"):
        policy.classify()


def test_raw_manifest_path_escape_is_refused_even_if_resealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    raw = _copy_archive(tmp_path, monkeypatch)
    manifest = raw / "manifest.json"
    pinned = json.loads(manifest.read_text())
    key = next(iter(pinned["sha256"]))
    pinned["sha256"]["../outside.json"] = pinned["sha256"].pop(key)
    manifest.write_text(json.dumps(pinned))
    monkeypatch.setattr(policy, "RAW_MANIFEST_SHA256",
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="archived artifact changed"):
        policy.classify()
