"""Toy-only contracts for the read-only endpoint diagnostic; no FV evaluation."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import threading

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
                               (diagnostic.GUARD, b"toy guarded runner source"),
                               (diagnostic.GUARD_TEST, b"toy guarded runner tests"),
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
    assert diagnostic.GUARD in report["source_before"]


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


@pytest.mark.parametrize("kind", ["existing", "dangling_symlink"])
def test_child_rejects_existing_or_dangling_output_without_clobber(tmp_path: Path,
                                                                  monkeypatch, kind: str):
    archive, plan, output, _ = _make_toy_archive(tmp_path, monkeypatch)
    original = archive.read_bytes()
    if kind == "existing":
        output.write_text("preserve me")
    else:
        output.symlink_to(output.with_name("missing-target.json"))
    with pytest.raises(ValueError, match="output must be fresh"):
        _run(archive, plan, output)
    assert archive.read_bytes() == original
    if kind == "dangling_symlink":
        assert output.is_symlink()
    else:
        assert output.read_text() == "preserve me"


def test_guarded_run_preserves_execution_and_numerical_status_separately(tmp_path: Path,
                                                                         monkeypatch):
    archive, plan, _, raw = _make_toy_archive(tmp_path, monkeypatch)
    output = archive.parent / "future-attempt/audit.json"
    captured: dict[str, Any] = {}

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path, cancel_requested=None):
        captured.update(command=command, wall_seconds=wall_seconds, rss_bytes=rss_bytes,
                        cancel_requested=cancel_requested)
        assert "--guarded" not in command
        sources = json.loads((output.parent / "preflight.json").read_text())["source_before"]
        child = {"phase": "diagnostic_refused", "numerical_status": "diagnostic_refusal",
            "endpoint_control": raw["last_accepted_control"],
            "endpoint_control_sha256": raw["last_accepted_control_sha256"],
            "parameters": raw["parameters"], "parameters_sha256": raw["parameters_sha256"],
            "input_before": raw["input_after"], "input_after": raw["input_after"],
            "source_before": sources, "source_after": sources,
            "source_unchanged": True, "input_unchanged": True,
            "runtime": raw["runtime"], "runtime_after": raw["runtime_after"]}
        diagnostic._write(output, child)
        with log_path.open("x") as stream:
            stream.write("toy child log")
        resource = {"exit_code": 0, "resource_termination": None, "monitor_error": None,
                    "elapsed_seconds": 1.0, "sampled_peak_rss_bytes": 1024,
                    "received_sigterm": False, "child_process_group_cleanup_sent": False,
                    "child_process_group_cleanup_error": None}
        diagnostic._write(report_path, resource)
        return resource

    monkeypatch.setattr(diagnostic, "run_guarded_diagnostic", fake_guard)
    parent = diagnostic.guarded_run(archive=archive, archive_sha256=diagnostic.sha(archive),
        parent_sha256=diagnostic.sha(archive.with_suffix(".run.json")),
        resource_sha256=diagnostic.sha(archive.with_suffix(".resource.json")), plan=plan, output=output)
    assert parent["execution_status"] == "completed"
    assert parent["numerical_status"] == "diagnostic_refusal"
    assert parent["child_terminal_phase"] == "diagnostic_refused"
    assert parent["endpoint_identity_matches_archive"] is True
    assert captured["wall_seconds"] == diagnostic.OUTER_SECONDS
    assert captured["rss_bytes"] == diagnostic.RSS_LIMIT_BYTES
    assert "--guarded" not in captured["command"]
    assert diagnostic.GUARD in parent["source_before"]
    assert diagnostic.GUARD_TEST in parent["source_before"]
    assert (output.parent / "preflight.json").is_file()
    assert (output.parent / ".guarded-launch.lock").is_file()
    assert output.with_suffix(".run.json").is_file()
    assert output.with_suffix(".resource.json").is_file()
    assert output.with_suffix(".log").read_text() == "toy child log"


@pytest.mark.parametrize("sibling", ["log", "resource", "preflight_tmp", "output_tmp", "atomic_temp", "claim_lock"])
def test_guarded_run_refuses_preexisting_or_dangling_output_siblings(tmp_path: Path,
                                                                    monkeypatch, sibling: str):
    archive, plan, _, _ = _make_toy_archive(tmp_path, monkeypatch)
    output = archive.parent / "future-attempt/audit.json"
    output.parent.mkdir(parents=True)
    paths = {"log": output.with_suffix(".log"),
             "resource": output.with_suffix(".resource.json"),
             "preflight_tmp": output.parent / "preflight.json.tmp",
             "output_tmp": output.with_suffix(".json.tmp"),
             "atomic_temp": output.parent / f".{output.name}.stale.tmp",
             "claim_lock": output.parent / ".guarded-launch.lock"}
    paths[sibling].symlink_to(paths[sibling].with_name("missing-target"))
    monkeypatch.setattr(diagnostic, "run_guarded_diagnostic",
                        lambda *_args, **_kwargs: pytest.fail("preexisting sibling launched child"))
    with pytest.raises(ValueError, match="siblings must all be fresh"):
        diagnostic.guarded_run(archive=archive, archive_sha256=diagnostic.sha(archive),
            parent_sha256=diagnostic.sha(archive.with_suffix(".run.json")),
            resource_sha256=diagnostic.sha(archive.with_suffix(".resource.json")), plan=plan, output=output)
    assert paths[sibling].is_symlink()


@pytest.mark.parametrize("name", ["preflight.json", "audit.log"])
def test_guarded_run_rejects_output_paths_colliding_with_derived_sidecars(tmp_path: Path,
                                                                          monkeypatch, name: str):
    archive, plan, _, _ = _make_toy_archive(tmp_path, monkeypatch)
    output = archive.parent / "future-attempt" / name
    monkeypatch.setattr(diagnostic, "run_guarded_diagnostic",
                        lambda *_args, **_kwargs: pytest.fail("colliding outputs launched child"))
    with pytest.raises(ValueError, match="paths must be distinct"):
        diagnostic.guarded_run(archive=archive, archive_sha256=diagnostic.sha(archive),
            parent_sha256=diagnostic.sha(archive.with_suffix(".run.json")),
            resource_sha256=diagnostic.sha(archive.with_suffix(".resource.json")), plan=plan, output=output)


@pytest.mark.parametrize(("resource", "expected", "mismatch"), [
    ({"resource_termination": "cancelled", "received_sigterm": False}, "failed", None),
    ({"resource_termination": None, "received_sigterm": True}, "failed", None),
    ({"resource_termination": "rss_monitor_unavailable", "received_sigterm": False}, "failed", None),
    ({"resource_termination": "resource_monitor_error", "monitor_error": "ps failed",
      "received_sigterm": False}, "failed", None),
    ({"resource_termination": "wall_time_limit", "received_sigterm": False}, "resource_limited", None),
    ({"resource_termination": "rss_limit", "received_sigterm": False}, "resource_limited", None),
    ({"resource_termination": None, "received_sigterm": False,
      "child_process_group_cleanup_sent": True}, "failed", None),
    ({"resource_termination": None, "received_sigterm": False}, "failed", "source"),
    ({"resource_termination": None, "received_sigterm": False}, "failed", "runtime"),
])
def test_guard_execution_failures_do_not_overwrite_child_numerical_status(
        tmp_path: Path, monkeypatch, resource: dict[str, Any], expected: str, mismatch: str | None):
    archive, plan, _, raw = _make_toy_archive(tmp_path, monkeypatch)
    output = archive.parent / "future-attempt/audit.json"

    def fake_guard(_command, *, report_path, log_path, **_kwargs):
        sources = json.loads((output.parent / "preflight.json").read_text())["source_before"]
        child_sources = sources
        runtime = raw["runtime"]
        if mismatch == "source":
            child_sources = {**sources, "wrong.py": "0" * 64}
        if mismatch == "runtime":
            runtime = {"device": "CPU FP64", "python": "changed", "torch": "changed"}
        child = {"phase": "finished", "numerical_status": "endpoint_diagnostic_completed",
            "endpoint_control": raw["last_accepted_control"],
            "endpoint_control_sha256": raw["last_accepted_control_sha256"],
            "parameters": raw["parameters"], "parameters_sha256": raw["parameters_sha256"],
            "input_before": raw["input_after"], "input_after": raw["input_after"],
            "source_before": sources, "source_after": child_sources,
            "source_unchanged": True, "input_unchanged": True,
            "runtime": runtime, "runtime_after": runtime}
        diagnostic._write(output, child)
        with log_path.open("x") as stream:
            stream.write("interrupted child")
        result = {"exit_code": -15, "resource_termination": None, "monitor_error": None,
                  "elapsed_seconds": 2.0, "sampled_peak_rss_bytes": 1024,
                  "received_sigterm": False, "child_process_group_cleanup_sent": False,
                  "child_process_group_cleanup_error": None, **resource}
        diagnostic._write(report_path, result)
        return result

    monkeypatch.setattr(diagnostic, "run_guarded_diagnostic", fake_guard)
    record = diagnostic.guarded_run(archive=archive, archive_sha256=diagnostic.sha(archive),
        parent_sha256=diagnostic.sha(archive.with_suffix(".run.json")),
        resource_sha256=diagnostic.sha(archive.with_suffix(".resource.json")), plan=plan, output=output)
    assert record["execution_status"] == expected
    assert record["numerical_status"] == "endpoint_diagnostic_completed"
    if resource.get("received_sigterm") or resource.get("resource_termination") == "cancelled":
        assert "cancellation" in record["failure_reason"] or "SIGTERM" in record["failure_reason"]


def test_guard_launch_exception_is_recorded_then_propagated(tmp_path: Path, monkeypatch):
    archive, plan, _, _ = _make_toy_archive(tmp_path, monkeypatch)
    output = archive.parent / "future-attempt/audit.json"
    monkeypatch.setattr(diagnostic, "run_guarded_diagnostic",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("toy launch failure")))
    with pytest.raises(RuntimeError, match="toy launch failure"):
        diagnostic.guarded_run(archive=archive, archive_sha256=diagnostic.sha(archive),
            parent_sha256=diagnostic.sha(archive.with_suffix(".run.json")),
            resource_sha256=diagnostic.sha(archive.with_suffix(".resource.json")), plan=plan, output=output)
    parent = json.loads(output.with_suffix(".run.json").read_text())
    assert parent["execution_status"] == "failed"
    assert parent["failure_reason"].endswith("RuntimeError: toy launch failure")


def test_guarded_attempt_claim_allows_only_one_concurrent_launch(tmp_path: Path, monkeypatch):
    archive, plan, _, raw = _make_toy_archive(tmp_path, monkeypatch)
    output = archive.parent / "future-attempt/audit.json"
    ingress_barrier = threading.Barrier(2)
    original_extract = diagnostic.extract_last_accepted_trial
    launches = 0
    preflight_writes = 0
    counter_lock = threading.Lock()

    def synchronized_extract(value):
        result = original_extract(value)
        ingress_barrier.wait(timeout=5)
        return result

    original_write = diagnostic._write

    def counted_write(path, value):
        nonlocal preflight_writes
        if path.name == "preflight.json":
            with counter_lock:
                preflight_writes += 1
        original_write(path, value)

    def fake_guard(_command, *, report_path, log_path, **_kwargs):
        nonlocal launches
        with counter_lock:
            launches += 1
        sources = json.loads((output.parent / "preflight.json").read_text())["source_before"]
        child = {"phase": "diagnostic_refused", "numerical_status": "diagnostic_refusal",
            "endpoint_control": raw["last_accepted_control"],
            "endpoint_control_sha256": raw["last_accepted_control_sha256"],
            "parameters": raw["parameters"], "parameters_sha256": raw["parameters_sha256"],
            "input_before": raw["input_after"], "input_after": raw["input_after"],
            "source_before": sources, "source_after": sources, "source_unchanged": True,
            "input_unchanged": True, "runtime": raw["runtime"], "runtime_after": raw["runtime_after"]}
        diagnostic._write(output, child)
        with log_path.open("x") as stream:
            stream.write("single child")
        resource = {"exit_code": 0, "resource_termination": None, "monitor_error": None,
                    "elapsed_seconds": 1.0, "sampled_peak_rss_bytes": 1024, "received_sigterm": False,
                    "child_process_group_cleanup_sent": False, "child_process_group_cleanup_error": None}
        diagnostic._write(report_path, resource)
        return resource

    monkeypatch.setattr(diagnostic, "extract_last_accepted_trial", synchronized_extract)
    monkeypatch.setattr(diagnostic, "_write", counted_write)
    monkeypatch.setattr(diagnostic, "run_guarded_diagnostic", fake_guard)

    def invoke():
        return diagnostic.guarded_run(archive=archive, archive_sha256=diagnostic.sha(archive),
            parent_sha256=diagnostic.sha(archive.with_suffix(".run.json")),
            resource_sha256=diagnostic.sha(archive.with_suffix(".resource.json")), plan=plan, output=output)

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke) for _ in range(2)]
        for future in futures:
            try:
                results.append(future.result())
            except ValueError as error:
                errors.append(str(error))
    assert len(results) == len(errors) == 1
    assert "already claimed" in errors[0]
    assert launches == preflight_writes == 1
    assert (output.parent / ".guarded-launch.lock").is_file()
    assert json.loads(output.with_suffix(".run.json").read_text())["execution_status"] == "completed"


@pytest.mark.parametrize("guarded", [False, True])
def test_cli_preserves_bare_child_mode_and_optional_guarded_parent(monkeypatch, guarded: bool):
    import sys

    args = ["endpoint-diagnostics", "--archive", "archive.json", "--archive-sha256", "a" * 64,
            "--parent-sha256", "b" * 64, "--resource-sha256", "c" * 64,
            "--plan", "plan.md", "--output", "output.json"]
    if guarded:
        args.append("--guarded")
    monkeypatch.setattr(sys, "argv", args)
    called: list[str] = []
    monkeypatch.setattr(diagnostic, "run", lambda **_kwargs: called.append("child"))
    monkeypatch.setattr(diagnostic, "guarded_run", lambda **_kwargs: (called.append("guarded") or {
        "execution_status": "completed", "numerical_status": "diagnostic_refusal",
        "child_terminal_phase": "diagnostic_refused"}))
    diagnostic.main()
    assert called == (["guarded"] if guarded else ["child"])
