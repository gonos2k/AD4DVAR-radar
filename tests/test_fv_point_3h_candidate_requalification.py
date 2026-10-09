from __future__ import annotations

from contextlib import nullcontext
import gzip
import hashlib
import json
import importlib
from typing import Any, cast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_candidate_requalification as requal
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as probe
_toy_child = importlib.import_module("test_fv_point_3h_nonsmooth_coupled_probe")._toy_child


def _trace(*, choices, signs, tied=False, count=360, target=0.0):
    return {"stage_count": count, "observed_stage_count": count,
        "expected_stage_count": 360, "nonfinite_or_tie": tied, "choices": choices,
        "face_signs": signs, "target_fluxes": [target], "maximum_face_flux": 2.0}


def test_current_pair_accepts_a_new_shared_signature_and_rejects_pair_mismatch():
    pair = requal.current_branch_pair_gate(
        _trace(choices=[[2]], signs=[[1]]), _trace(choices=[[2]], signs=[[1]]),
        dtype=torch.float64)
    assert pair["passed"]
    assert pair["signature_changed_from_base"] is None

    assert not requal.current_branch_pair_gate(
        _trace(choices=[[2]], signs=[[1]]), _trace(choices=[[3]], signs=[[1]]),
        dtype=torch.float64)["passed"]
    assert not requal.current_branch_pair_gate(
        _trace(choices=[[2]], signs=[[1]], tied=True),
        _trace(choices=[[2]], signs=[[1]]), dtype=torch.float64)["passed"]
    assert not requal.current_branch_pair_gate(
        _trace(choices=[[2]], signs=[[1]], count=359),
        _trace(choices=[[2]], signs=[[1]]), dtype=torch.float64)["passed"]


def test_current_pair_uses_only_the_scaled_target_flux_tolerance():
    accepted = requal.current_branch_pair_gate(
        _trace(choices=[], signs=[], target=1e-14),
        _trace(choices=[], signs=[], target=-1e-14), dtype=torch.float64)
    rejected = requal.current_branch_pair_gate(
        _trace(choices=[], signs=[], target=1e-10),
        _trace(choices=[], signs=[], target=-1e-10), dtype=torch.float64)
    assert accepted["passed"]
    assert not rejected["passed"]


def test_requalification_parent_classifier_uses_its_300_second_plan_limit():
    resource = {"wall_limit_seconds": 300.0, "rss_limit_bytes": 1024**3,
        "exit_code": 0, "resource_termination": None, "monitor_error": None,
        "received_sigterm": False, "elapsed_seconds": 299.0,
        "sampled_peak_rss_bytes": 900_000_000}
    assert probe._execution_status(resource, wall_seconds=300.0,
        rss_bytes=1024**3) == "completed"
    assert probe._execution_status({**resource, "elapsed_seconds": 301.0},
        wall_seconds=300.0, rss_bytes=1024**3) == "wall_timeout"


def test_archived_direction_loader_checks_parent_gzip_and_control_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(requal, "ROOT", tmp_path)
    step_path = tmp_path / "step.json.gz"
    run_path = tmp_path / "step.run.json"
    resource_path = tmp_path / "step.resource.json"
    manifest_path = tmp_path / "archive.json"
    fixed_path = tmp_path / "fixed-plan.json"
    for name, path in (("ARCHIVED_RUN", run_path), ("ARCHIVED_RESOURCE", resource_path),
                       ("ARCHIVE_MANIFEST", manifest_path), ("FIXED_PLAN", fixed_path)):
        monkeypatch.setattr(requal, name, path)
    fixed_path.write_text("{}")
    fixed_name = fixed_path.relative_to(tmp_path).as_posix()
    child_sha = "child-hash"
    child_resource = {"exit_code": 0, "resource_termination": None, "monitor_error": None,
        "received_sigterm": False, "wall_limit_seconds": 660.0, "rss_limit_bytes": 1024**3,
        "elapsed_seconds": 10.0, "sampled_peak_rss_bytes": 10}
    child_resource["exit_code"] = 0
    payload: dict[str, Any] = {
        "phase": "finished", "execution_status": "completed", "plan_sha256": "old-plan",
        "base_control_sha256": "base", "current_control_sha256": "base",
        "candidate_committed": False, "active_candidate_committed": False,
        "optimizer_steps_applied": 0, "hvp_calls_started": 56, "hvp_calls_completed": 56,
        "source_unchanged": True, "fixed_input_unchanged": True, "runtime_unchanged": True,
        "source_before": {fixed_name: "old-plan"}, "source_after": {fixed_name: "old-plan"},
        "input_before": {"control": "base"}, "input_after": {"control": "base"},
        "runtime": {"torch": "old"}, "runtime_after": {"torch": "old"},
        "plan_hashes": {"source_files": {}, "archive_files": {}},
        "iterations": [{"accepted": None}], "theta": 0.5,
        "hessian_scope": "full 26x26 ambient FP64 one-sided Hessians; exact selected branch contexts",
        "hvp_history": ([{"phase": "parity", "side": side, "operator": operator,
            "status": "completed"} for side in (-1, 1)
            for operator in ("native", "selected_extension")]
            + [{"phase": "basis", "side": side, "column": column,
                "operator": "selected_extension", "status": "completed"}
                for side in (-1, 1) for column in range(26)]),
        "hvp_columns": [{"side": side, "column": column}
            for side in (-1, 1) for column in range(26)],
        "delta": [0.0] * 26 + [0.0],
        "hessian_minus": torch.eye(26).tolist(), "hessian_plus": torch.eye(26).tolist(),
        "matrix": torch.eye(27).tolist(), "residual": [0.0] * 27,
    }
    payload["delta"][0] = 1.0
    raw = json.dumps(payload, separators=(",", ":")).encode()
    gzip_bytes = gzip.compress(raw)
    step_path.write_bytes(gzip_bytes)
    actual_sha = hashlib.sha256(raw).hexdigest()
    child_resource_run = dict(child_resource)
    run_path.write_text(json.dumps({"execution_status": "completed", "child_sha256": actual_sha,
        "numerical_status": "coupled_step_refused", "child_read_error": None,
        "resource": child_resource_run}))
    resource_path.write_text(json.dumps(child_resource))
    manifest_path.write_text(json.dumps({"archives": {"raw-step": {
        "gzip_path": "step.json.gz", "raw_sha256": actual_sha,
        "gzip_sha256": hashlib.sha256(gzip_bytes).hexdigest()}}}))

    # The production manifest key is a repository path; use a temporary key that
    # follows the same archive identity contract.
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest["archives"] = {"graphify-out/fv-root-cause-20260919/nonsmooth_coupled_20261009_attempt1/step.json": {
        "gzip_path": "step.json.gz", "raw_sha256": actual_sha,
        "gzip_sha256": hashlib.sha256(gzip_bytes).hexdigest()}}
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(requal, "ARCHIVED_STEP", step_path)

    loaded = requal.load_archived_direction(step_path, base_control_sha256="base",
        old_plan_sha256="old-plan", expected_child_sha256=actual_sha)
    assert loaded["delta"].shape == (27,)
    assert loaded["child_sha256"] == actual_sha
    with pytest.raises(ValueError, match="shared base"):
        requal.load_archived_direction(step_path, base_control_sha256="different",
            old_plan_sha256="old-plan", expected_child_sha256=actual_sha)


@pytest.mark.parametrize(("hessian_scale", "passed"), [(1.0, True), (2.0, False)])
def test_fresh_current_hvps_gate_the_archived_direction_rhs(tmp_path, monkeypatch,
                                                             hessian_scale, passed):
    weights = torch.ones(5, dtype=torch.float64)
    control = torch.zeros(26, dtype=torch.float64)
    direction = torch.zeros_like(control)
    direction[0] = 1.0
    delta = torch.cat((direction, torch.zeros(1, dtype=torch.float64)))
    residual = torch.zeros(27, dtype=torch.float64)
    residual[0] = -1.0
    archived = {"delta": delta, "raw": {"theta": 0.5,
        "matrix": torch.eye(27).tolist(), "residual": residual.tolist()}}

    class Problem:
        def objective(self, value, _parameters):
            return 0.5 * hessian_scale * torch.dot(value, value)

    fake_probe = SimpleNamespace(
        FACE={"axis": "y", "row": 4, "column": 3},
        transport=SimpleNamespace(selected_face_extension=lambda *_args: nullcontext()),
        _face_weights=lambda *_args, **_kwargs: weights,
        _pivot=lambda *_args: 0,
        _path_direction=lambda *_args: (direction, direction),
        _min_norm_mix=lambda *_args: (0.5, torch.zeros(26), {}),
        _face_value=lambda *_args: torch.zeros((), dtype=torch.float64),
        _residual_measure=lambda g, q, scale: torch.cat((g, (q / scale).reshape(1))),
        _counted_hvp=probe._counted_hvp,
    )
    record = {"policy": {"hvp_calls": 2}, "hvp_calls_started": 0,
        "hvp_calls_completed": 0, "hvp_history": []}
    output = tmp_path / "child.json"
    gradient = -direction
    result = requal.validate_current_direction(fake_probe, record, output, 1e20,
        Problem(), control, torch.zeros(1, dtype=torch.float64), archived,
        gradient, gradient, torch.zeros_like(control), torch.tensor(1.0, dtype=torch.float64))
    assert result["passed"] is passed
    assert result["actual_rhs_relative"] == pytest.approx(0.0 if passed else 1.0)
    assert record["hvp_calls_started"] == record["hvp_calls_completed"] == 2
    assert result["archived_matrix_backward_error"] == 0.0


def test_toy_requalification_accepts_a_new_current_pair_and_uses_two_hvps(tmp_path, monkeypatch):
    initial_plan, control, _parameters = _toy_child(monkeypatch)
    plan: dict[str, Any] = cast(dict[str, Any], initial_plan)
    plan.update(experiment_kind="nonsmooth_candidate_requalification",
        policy=requal.POLICY, direction_archive="archived.json",
        producing_plan_sha256="old-plan-sha")
    delta = torch.zeros(27, dtype=torch.float64)
    delta[0] = 0.9
    delta[26] = 0.0
    matrix = torch.eye(27, dtype=torch.float64)
    matrix[20, 26] = 2.0
    matrix[26, 20] = 1.0
    residual = torch.zeros(27, dtype=torch.float64)
    residual[0] = -0.9
    archived = {"delta": delta, "hessian_minus": torch.eye(26, dtype=torch.float64),
        "hessian_plus": torch.eye(26, dtype=torch.float64),
        "raw": {"theta": 0.5, "matrix": matrix.tolist(), "residual": residual.tolist(),
            "face_control_sha256": probe._tensor_sha(torch.cat((control[:20],
                torch.zeros(1, dtype=torch.float64), control[21:]))),
            "face_qualification": {"traces": {
                "-1": {"signature_sha256": "current-1"},
                "1": {"signature_sha256": "current-1"}}},
            "source_before": {"toy-plan.json": "plan-sha"},
            "input_before": {"control_sha256": probe._tensor_sha(control),
                "parameters_sha256": probe._tensor_sha(torch.zeros(1, dtype=torch.float64)),
                "terminal_truth_sha256": probe._tensor_sha(torch.tensor([1.0], dtype=torch.float64)),
                "archived_input": "frozen"},
            "runtime": {"device": "CPU FP64", "torch": "toy", "python": "toy"}}}
    monkeypatch.setattr(requal, "load_archived_direction", lambda *_a, **_k: archived)

    def changed_pair(_problem, candidate, _parameters, side):
        choices = [[2]] if candidate[0] > 0 else [[1]]
        return {"stage_count": 360, "observed_stage_count": 360,
            "expected_stage_count": 360, "choices": choices, "face_signs": [[1]],
            "target_fluxes": [0.0], "minimum_margins": {}, "maximum_face_flux": 1.0,
            "nonfinite_or_tie": False, "signature_sha256": f"current-{choices[0][0]}",
            "side": side}

    monkeypatch.setattr(probe, "_analysis_trace", changed_pair)
    result = probe._run_child(probe.ROOT / "toy-plan.json", "plan-sha", tmp_path / "child.json")

    assert result["execution_status"] == "completed"
    assert result["candidate_committed"] is True
    assert result["candidate_type"] == "one_nonsmooth_requalified_step"
    assert result["hvp_calls_started"] == result["hvp_calls_completed"] == 2
    assert result["solve"]["dense_solves"] == 0
    assert result["solve"]["method"] == "archived direction verified by fresh current HVPs"
    proposal = result["iterations"][0]["proposal"]
    assert proposal["branch_support_passed"] is True
    assert proposal["branch_changes_from_base"]["-1"]["choices_changed_from_base"] is True
    assert proposal["current_branch_pair_gate"]["same_current_choices"] is True
    assert result["final_repeat"]["current_branch_pair_gate"]["passed"] is True
    assert result["final_repeat"]["gradient_matches_proposal"] is True


def test_actual_refused_pr265_direction_can_be_loaded_without_invented_basis_fields():
    entry = json.loads(requal.ARCHIVE_MANIFEST.read_text())["archives"][
        "graphify-out/fv-root-cause-20260919/nonsmooth_coupled_20261009_attempt1/step.json"]
    archive = requal.load_archived_direction(
        base_control_sha256=probe.BASE_CONTROL_SHA,
        old_plan_sha256="83ea077a6d7ba615a84729a3588a92b156687dd02a8fc7320cf9b90ecf5ce299",
        expected_child_sha256=entry["raw_sha256"])
    assert archive["delta"].shape == (27,)
    assert archive["raw"]["hvp_history"][4]["phase"] == "basis"
    assert "operator" not in archive["raw"]["hvp_history"][4]
