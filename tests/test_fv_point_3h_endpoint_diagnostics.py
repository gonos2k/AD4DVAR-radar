"""Toy-only contracts for the read-only endpoint diagnostic; no FV evaluation."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_endpoint_diagnostics as diagnostic
from examples.weather_scenarios import fv_point_3h_precision_hessian_audit as hess_probe
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trial(control: Tensor, gradient: Tensor, branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": 4, "alpha": 0.0625, "status": "accepted", "armijo_passed": True,
        "strict_point_passed": True, "control": control.tolist(),
        "objective": 0.75, "phi": float(torch.dot(gradient, gradient) / 2),
        "gradient": gradient.tolist(), "gradient_blocks": blocks.gradient_blocks(gradient),
        "branch": branch,
    }


def _make_toy_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                      hessian_mode: str = "completed") -> tuple[Path, Path, Path, dict[str, Any]]:
    root = tmp_path / "repo"
    evidence = root / "graphify-out/fv-root-cause-20260919"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "ROOT", root)
    monkeypatch.setattr(diagnostic, "EVIDENCE", evidence)
    plan_rel = "PR238_ENDPOINT_DIAGNOSTIC_PLAN_20261004.md"
    producer_plan_rel = "graphify-out/fv-root-cause-20260919/S4_COUPLED_ORIGINAL_J_CONTINUATION_PLAN_20261003.md"
    for relative, contents in ((diagnostic.SELF, b"toy driver source"),
                               (diagnostic.TEST, b"toy test source"),
                               (plan_rel, b"toy endpoint plan"),
                               (producer_plan_rel, b"toy producer plan")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    control = torch.full((diagnostic.NC,), 0.125, dtype=torch.float64)
    start = torch.full((diagnostic.NC,), 0.25, dtype=torch.float64)
    seed_control = torch.full((diagnostic.NC,), -0.125, dtype=torch.float64)
    parameters = torch.linspace(-0.2, 0.2, diagnostic.NPARAM, dtype=torch.float64)
    truth = torch.tensor([0.5, -0.25], dtype=torch.float64)
    fixed_prior = {"sha256": "f" * 64, "contract": "toy fixed zero prior"}
    archived_input = {
        "control_sha256": diagnostic.tensor_sha(start),
        "parameters_sha256": diagnostic.tensor_sha(parameters),
        "terminal_truth_sha256": diagnostic.tensor_sha(truth),
        "problem": {"fixed_problem_sha256": "a" * 64},
        "prior": fixed_prior,
    }

    def input_identity(_problem, _original, value, p, t):
        assert torch.equal(p, parameters) and torch.equal(t, truth)
        return {"archived_input": archived_input, "control_sha256": diagnostic.tensor_sha(value),
                "parameters_sha256": diagnostic.tensor_sha(p),
                "terminal_truth_sha256": diagnostic.tensor_sha(t)}

    start_identity = input_identity(None, None, start, parameters, truth)
    endpoint_identity = input_identity(None, None, control, parameters, truth)
    branch = {"status": "passed_strict_branch", "euler_stages": diagnostic.NSTAGE,
              "choice_stage_count": diagnostic.NSTAGE, "face_sign_stage_count": diagnostic.NSTAGE,
              "signature_sha256": "b" * 64}
    margins = {"complete": True, "observed_stage_count": diagnostic.NSTAGE,
               "expected_stage_count": diagnostic.NSTAGE}
    gradient = torch.linspace(-0.03, 0.04, diagnostic.NC, dtype=torch.float64)
    row = _trial(control, gradient, branch)
    row["phi"] = 0.75
    raw = {
        "phase": "finished", "numerical_status": "epoch_limit", "optimizer_steps": 4,
        "optimizer_steps_max": 4, "start_control": start.tolist(),
        "start_control_sha256": diagnostic.tensor_sha(start),
        "last_accepted_control": control.tolist(),
        "last_accepted_control_sha256": diagnostic.tensor_sha(control),
        "parameters": parameters.tolist(), "parameters_sha256": diagnostic.tensor_sha(parameters),
        "runtime": {"device": "CPU FP64", "python": "3.12.13", "torch": "2.13.0"},
        "runtime_after": {"device": "CPU FP64", "python": "3.12.13", "torch": "2.13.0"},
        "input_before": start_identity, "input_after": endpoint_identity,
        "input_unchanged": True, "source_unchanged": True,
        "full_root_claim": False, "minimum_claim": False,
        "response_computed": False, "score_computed": False, "adjoint_computed": False,
        "plan_sha256": _hash_bytes((root / producer_plan_rel).read_bytes()),
        "r8_sha256": "c" * 64, "accepted_audit_sha256": "d" * 64,
        "source_before": {}, "source_after": {}, "trials": [dict(row, epoch=i) for i in range(1, 5)],
    }
    archive = evidence / "coupled_original_j_continuation.json"
    archive.write_text(json.dumps(raw))
    source_map = {producer_plan_rel: _hash_bytes((root / producer_plan_rel).read_bytes())}
    raw["source_before"] = dict(source_map)
    raw["source_after"] = dict(source_map)
    archive.write_text(json.dumps(raw))

    resource = {"command": ["toy child"], "exit_code": 0, "resource_termination": None,
                "monitor_error": None, "elapsed_seconds": 988.7, "sampled_peak_rss_bytes": 4096,
                "rss_samples": 2, "wall_limit_seconds": 1500,
                "rss_limit_bytes": diagnostic.RSS_LIMIT_BYTES, "sampling": "toy"}
    (evidence / "coupled_original_j_continuation.resource.json").write_text(json.dumps(resource))
    parent = {"execution_status": "completed", "numerical_status": "epoch_limit",
              "optimizer_steps": 4, "child_read_error": None, "source_bytes_unchanged": True,
              "input_identity_unchanged": True, "r8_unchanged": True, "response_computed": False,
              "adjoint_computed": False, "full_root_claim": False,
              "source_sha256_before": source_map, "source_sha256_after": source_map,
              "source_bytes_base64_after": {
                  producer_plan_rel: base64.b64encode((root / producer_plan_rel).read_bytes()).decode()},
              "resource": resource}
    (evidence / "coupled_original_j_continuation.run.json").write_text(json.dumps(parent))

    problem = SimpleNamespace(objective=lambda c, p: c.square().sum() / 2 + p.sum() * 0)
    original = torch.zeros(diagnostic.NC, dtype=torch.float64)
    base_identity = input_identity(None, None, seed_control, parameters, truth)
    monkeypatch.setattr(diagnostic.blocks, "runtime_identity", lambda: raw["runtime"])
    monkeypatch.setattr(diagnostic.seed, "_prepare_fixed_seed",
                        lambda: (problem, original, seed_control, parameters, truth, base_identity))
    monkeypatch.setattr(diagnostic.seed, "_input_identity", input_identity)
    monkeypatch.setattr(diagnostic.tail, "_full_current_branch", lambda *_args: (branch, margins))
    monkeypatch.setattr(diagnostic.seed, "_valid_margins", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(diagnostic.tail, "_fresh_merit",
                        lambda *_args: (torch.tensor(0.75, dtype=torch.float64), gradient,
                                        torch.tensor(0.75, dtype=torch.float64)))
    hessian_matrix = torch.eye(diagnostic.NC, dtype=torch.float64)
    hessian_result = {"hessian": hessian_matrix.tolist(),
                      "hessian_sha256": diagnostic.tensor_sha(hessian_matrix),
                      "hvp_columns": diagnostic.NC, "hvp_calls_total": diagnostic.NC + 1,
                      "symmetry_relative": 0.0, "eigenvalues": [1.0] * diagnostic.NC,
                      "minimum_eigenpair_audit": {"passed": True},
                      "block_schur": {"status": "schur_evaluated", "field_gradient_inf": 0.1}}
    if hessian_mode == "completed":
        def fresh_hessian(_objective, _control, _parameters, current_gradient, _deadline):
            return {**hessian_result, "schur_input_gradient": current_gradient.tolist()}
        monkeypatch.setattr(diagnostic.hess_probe, "fresh_hessian", fresh_hessian)
    elif hessian_mode == "schur_refused":
        def refused_schur(_objective, _control, _parameters, current_gradient, _deadline):
            return {**hessian_result, "schur_input_gradient": current_gradient.tolist(),
                    "block_schur": {"status": "hff_schur_refused", "refusal": "toy Hff gate"}}
        monkeypatch.setattr(diagnostic.hess_probe, "fresh_hessian", refused_schur)
    elif hessian_mode == "refused":
        monkeypatch.setattr(diagnostic.hess_probe, "fresh_hessian",
                            lambda *_args: (_ for _ in ()).throw(hess_probe.AuditRefusal("toy Hessian refusal")))
    elif hessian_mode == "programming_error":
        monkeypatch.setattr(diagnostic.hess_probe, "fresh_hessian",
                            lambda *_args: (_ for _ in ()).throw(RuntimeError("toy programming error")))
    return archive, root / plan_rel, evidence / "endpoint.json", raw


def test_extract_last_accepted_trial_cross_checks_control_and_step_count():
    control = torch.zeros(diagnostic.NC, dtype=torch.float64)
    gradient = torch.ones(diagnostic.NC, dtype=torch.float64)
    branch = {"status": "passed_strict_branch"}
    accepted = _trial(control, gradient, branch)
    accepted["epoch"] = 1
    raw = {"trials": [accepted], "last_accepted_control": control.tolist(),
           "last_accepted_control_sha256": diagnostic.tensor_sha(control), "optimizer_steps": 1}
    assert diagnostic.extract_last_accepted_trial(raw)["status"] == "accepted"
    raw["last_accepted_control_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="last accepted trial/control"):
        diagnostic.extract_last_accepted_trial(raw)


def test_saved_coupled_archive_schema_selects_last_accepted_trial_without_fv():
    archive = (diagnostic.EVIDENCE / "coupled_original_j_continuation_attempt1"
               / "coupled_original_j_continuation.json")
    raw, _ = diagnostic._read_object(archive)
    parent, _ = diagnostic._read_object(archive.with_suffix(".run.json"))
    resource, _ = diagnostic._read_object(archive.with_suffix(".resource.json"))
    diagnostic._validate_parent_and_resource(raw, parent, resource)
    endpoint = diagnostic.extract_last_accepted_trial(raw)
    assert endpoint["epoch"] == raw["optimizer_steps"] == 4
    assert endpoint["control"] == raw["last_accepted_control"]


def _run(archive: Path, plan: Path, output: Path,
         parent_sha256: str | None = None,
         archive_sha256: str | None = None) -> dict[str, Any]:
    parent, _ = diagnostic._read_object(archive.with_suffix(".run.json"))
    resource, _ = diagnostic._read_object(archive.with_suffix(".resource.json"))
    return diagnostic.run(archive=archive, archive_sha256=(archive_sha256 or diagnostic.sha(archive)),
                          parent_sha256=(parent_sha256 or diagnostic.sha(archive.with_suffix(".run.json"))),
                          resource_sha256=diagnostic.sha(archive.with_suffix(".resource.json")),
                          plan=plan, output=output)


def test_toy_run_reports_only_fresh_endpoint_diagnostics(tmp_path: Path, monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch)
    report = _run(archive, plan, output)
    assert report["phase"] == "finished"
    assert report["numerical_status"] == "endpoint_diagnostic_completed"
    assert report["optimizer_step_applied"] is False
    assert report["response_computed"] is report["score_computed"] is report["adjoint_computed"] is False
    assert report["hvp_columns_completed"] == diagnostic.NC
    assert report["hvp_calls_total"] == diagnostic.NC + 1
    assert report["fresh_hessian"]["block_schur"]["field_gradient_inf"] == 0.1
    assert report["fresh_hessian"]["schur_input_gradient"] == report["gradient"]
    assert max(abs(value) for value in report["fresh_hessian"]["schur_input_gradient"][:diagnostic.NF]) > 0
    assert report["source_unchanged"] and report["input_unchanged"]


def test_typed_hessian_refusal_keeps_jg_and_does_not_claim_hvp_count(tmp_path: Path, monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch, "refused")
    report = _run(archive, plan, output)
    assert report["phase"] == "diagnostic_refused"
    assert report["numerical_status"] == "diagnostic_refusal"
    assert report["objective"] == 0.75 and len(report["gradient"]) == diagnostic.NC
    assert report["hvp_columns_completed"] == report["hvp_calls_total"] == "not_recorded"
    assert report["optimizer_step_applied"] is False


def test_returned_schur_refusal_keeps_completed_hessian_and_hvp_counts(tmp_path: Path,
                                                                      monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch, "schur_refused")
    report = _run(archive, plan, output)
    assert report["phase"] == "diagnostic_refused"
    assert report["numerical_status"] == "diagnostic_refusal"
    assert report["block_schur_status"] == "hff_schur_refused"
    assert report["refusal"].endswith("status=hff_schur_refused; refusal=toy Hff gate")
    assert report["hvp_columns_completed"] == diagnostic.NC
    assert report["hvp_calls_total"] == diagnostic.NC + 1
    assert len(report["fresh_hessian"]["hessian"]) == diagnostic.NC


def test_unexpected_hessian_programming_error_propagates(tmp_path: Path, monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch, "programming_error")
    with pytest.raises(RuntimeError, match="toy programming error"):
        _run(archive, plan, output)


def test_parent_sidecar_hash_mismatch_is_rejected(tmp_path: Path, monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch)
    parent = archive.with_suffix(".run.json")
    pinned_hash = diagnostic.sha(parent)
    parent.write_text(parent.read_text() + " ")
    with pytest.raises(ValueError, match="caller-pinned parent/resource SHA-256 mismatch"):
        _run(archive, plan, output, parent_sha256=pinned_hash)


def test_wrong_archive_hash_is_rejected(tmp_path: Path, monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="caller-pinned continuation archive SHA-256 mismatch"):
        _run(archive, plan, output, archive_sha256="0" * 64)


def test_changed_current_source_is_rejected_against_archived_map(tmp_path: Path, monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch)
    producer_plan = tmp_path / "repo/graphify-out/fv-root-cause-20260919/S4_COUPLED_ORIGINAL_J_CONTINUATION_PLAN_20261003.md"
    producer_plan.write_text("changed producer plan")
    with pytest.raises(ValueError, match="current source tree differs from the continuation's archived source hashes"):
        _run(archive, plan, output)


def test_current_runtime_mismatch_is_rejected(tmp_path: Path, monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(diagnostic.blocks, "runtime_identity",
                        lambda: {"device": "CPU FP64", "python": "wrong", "torch": "wrong"})
    with pytest.raises(ValueError, match="runtime differs from the completed continuation archive"):
        _run(archive, plan, output)


def test_branch_signature_mismatch_is_serialized_as_numerical_refusal(tmp_path: Path,
                                                                    monkeypatch):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch)
    changed_branch = {"status": "passed_strict_branch", "euler_stages": diagnostic.NSTAGE,
                      "choice_stage_count": diagnostic.NSTAGE, "face_sign_stage_count": diagnostic.NSTAGE,
                      "signature_sha256": "e" * 64}
    monkeypatch.setattr(diagnostic.tail, "_full_current_branch",
                        lambda *_args: (changed_branch, {"complete": True}))
    report = _run(archive, plan, output)
    assert report["phase"] == "diagnostic_refused"
    assert report["numerical_status"] == "diagnostic_refusal"
    assert report["branch"]["signature_sha256"] == "e" * 64
    assert report["hvp_calls_total"] == "not_recorded"
