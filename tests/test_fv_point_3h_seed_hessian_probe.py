from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_seed_hessian_probe as probe


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (torch.diag(torch.tensor([-2.0, *range(1, 26)], dtype=torch.float64)),
         "locally_negative_curvature"),
        (torch.diag(torch.tensor([*range(1, 27)], dtype=torch.float64)),
         "locally_positive_curvature"),
        (torch.diag(torch.tensor([0.0, *range(2, 27)], dtype=torch.float64)),
         "near_zero_inconclusive"),
    ],
)
def test_exact_hessian_audit_uses_columns_and_fresh_minimum_eigenpair(matrix, expected):
    control = torch.linspace(-0.5, 0.5, 26, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)

    class Quadratic:
        @staticmethod
        def objective(value, _parameters):
            return 0.5 * torch.dot(value, matrix @ value)

    result = probe._hessian_audit(Quadratic(), control, parameters)

    assert result["status"] == "curvature_verified"
    assert result["curvature_verdict"] == expected
    assert result["hvp_column_count"] == 26
    assert result["hvp_calls"] == control.numel() + 1
    assert result["symmetry_relative"] <= probe.SYMMETRY_TOLERANCE
    assert result["fresh_eigenpair_relative_residual"] <= probe.EIGENPAIR_TOLERANCE
    assert len(result["hessian_sha256"]) == 64
    assert len(result["minimum_eigenvector_sha256"]) == 64
    assert "matrix" not in result and "minimum_eigenvector" not in result


def test_exact_hessian_audit_rejects_nonfinite_hvp_without_curvature_claim(monkeypatch):
    control = torch.ones(26, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)
    problem = SimpleNamespace(objective=lambda value, _p: 0.5 * torch.dot(value, value))
    original_jvp = torch.func.jvp

    def nonfinite_jvp(function, primals, tangents):
        result = original_jvp(function, primals, tangents)
        return result[0], torch.full_like(result[1], float("nan"))

    monkeypatch.setattr(probe.torch.func, "jvp", nonfinite_jvp)
    result = probe._hessian_audit(problem, control, parameters)
    assert result["status"] == "hvp_or_eigensolver_failed"
    assert "curvature_verdict" not in result
    assert result["hvp_calls"] == 1


def test_zero_hessian_is_inconclusive_and_has_no_eigenvalue_ratio():
    control = torch.ones(26, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)
    problem = SimpleNamespace(objective=lambda value, _p: value.sum() * 0.0)

    result = probe._hessian_audit(problem, control, parameters)

    assert result["status"] == "zero_scale_inconclusive"
    assert result["inertia_assigned"] is False
    assert result["lambda_ratio"] is None


def test_nonfinite_hessian_norm_is_not_reported_as_zero_scale(monkeypatch):
    control = torch.ones(26, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)
    problem = SimpleNamespace(objective=lambda value, _p: 0.5 * torch.dot(value, value))
    monkeypatch.setattr(probe.torch.linalg, "matrix_norm", lambda _matrix: torch.tensor(float("inf")))

    result = probe._hessian_audit(problem, control, parameters)

    assert result["status"] == "nonfinite_hessian_scale"
    assert result["inertia_assigned"] is False


def test_zero_lambda_max_is_recorded_but_ratio_is_undefined():
    matrix = torch.diag(torch.tensor([-2.0, *([0.0] * 25)], dtype=torch.float64))
    control = torch.ones(26, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)

    class Quadratic:
        @staticmethod
        def objective(value, _parameters):
            return 0.5 * torch.dot(value, matrix @ value)

    result = probe._hessian_audit(Quadratic(), control, parameters)

    assert result["lambda_max"] == 0.0
    assert result["lambda_ratio"] is None
    assert result["curvature_verdict"] == "locally_negative_curvature"


def test_hessian_audit_does_not_hide_objective_callback_error():
    control = torch.ones(26, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)

    def broken_objective(_value, _parameters):
        raise ValueError("objective callback bug")

    with pytest.raises(ValueError, match="objective callback bug"):
        probe._hessian_audit(SimpleNamespace(objective=broken_objective), control, parameters)


def test_hessian_audit_does_not_classify_callback_runtime_by_message_text():
    control = torch.ones(26, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)

    def broken_objective(_value, _parameters):
        raise RuntimeError("linalg callback bug")

    with pytest.raises(RuntimeError, match="linalg callback bug"):
        probe._hessian_audit(SimpleNamespace(objective=broken_objective), control, parameters)


def _valid_child(source: dict[str, str], archive: dict[str, object]) -> dict[str, Any]:
    input_identity = {
        "archived_input": archive,
        "control_sha256": probe.CONTROL_SHA256,
        "parameters_sha256": probe.PARAMETERS_SHA256,
        "terminal_truth_sha256": probe.TRUTH_SHA256,
        "changed_control_index": probe.seed_linear.EXPECTED_CHANGED_CONTROL,
        "changed_control_indices": [probe.seed_linear.EXPECTED_CHANGED_CONTROL],
    }
    hessian = {
        "status": "curvature_verified", "hvp_column_count": 26, "hvp_calls": 27,
        "hessian_shape": [26, 26], "hessian_dtype": "torch.float64", "hessian_device": "cpu",
        "hessian_frobenius_norm": 6201.0**0.5,
        "eigenvalues": list(range(1, 27)), "lambda_min": 1.0, "lambda_max": 26.0,
        "spectral_scale": 26.0, "lambda_ratio": 1.0 / 26.0,
        "symmetry_relative": 0.0, "fresh_eigenpair_relative_residual": 0.0,
        "fresh_eigenpair_residual_norm": 0.0, "symmetry_absolute_allowance": 0.0,
        "rounding_allowance": 128.0 * torch.finfo(torch.float64).eps * (6201.0**0.5),
        "curvature_verdict": "locally_positive_curvature",
        "curvature_threshold": 1e-8 * 26.0,
        "inertia_assigned": True, "hessian_sha256": "a" * 64,
        "minimum_eigenvector_sha256": "b" * 64,
    }
    return {
        "pid": 42, "execution_status": "completed", "execution_phase": "finished",
        "numerical_status": "curvature_verified", "branch_status": "passed_strict_branch",
        "gradient_status": "finite", "hessian_status": "curvature_verified",
        "child_elapsed_seconds": 1.0,
        "child_phase_seconds": {"source_and_fixture": 0.1, "strict_branch": 0.3,
                                 "analysis_objective_and_gradient": 0.1,
                                 "exact_hessian_audit": 0.5},
        "wall_limit_seconds": probe.WALL_SECONDS,
        "rss_limit_bytes": probe.SAMPLED_RSS_BYTES,
        "termination_reap_grace_seconds": probe.REAP_GRACE_SECONDS,
        "source_before": source, "source_after": source, "source_unchanged": True,
        "plan_sha256": probe.PLAN_SHA256, "plan_unchanged": True,
        "archive_input_sha256": probe.ARCHIVED_INPUT_SHA256,
        "archive_before": archive, "archive_unchanged": True,
        "input_before": input_identity, "input_after": input_identity,
        "input_unchanged": True,
        "layout": {"controls": 26, "parameters": 13, "euler_stages": 3600},
        "branch": {"status": "passed_strict_branch", "euler_stages": 3600,
                   "choice_stage_count": 3600, "face_sign_stage_count": 3600,
                   "signature_sha256": "c" * 64},
        "branch_margins": {"complete": True, "observed_stage_count": 3600,
                           "expected_stage_count": 3600},
        "hessian": hessian, "newton_steps": 0, "pcg_calls": 0,
        "terminal_score_computed": False, "response_computed": False,
        "adjoint_computed": False,
    }


def test_parent_recomputes_curvature_claim_and_requires_all_eigenvalues(monkeypatch):
    source = {"probe.py": "pinned"}
    archive: dict[str, object] = {"archive": "pinned"}
    monkeypatch.setattr(probe, "_input_expected", lambda _identity, _archive: True)
    child = _valid_child(source, archive)

    assert probe._valid_child(child, source, archive)
    del child["hessian"]["eigenvalues"]
    assert not probe._valid_child(child, source, archive)


def test_parent_rejects_curvature_sign_label_inconsistent_with_spectrum(monkeypatch):
    source = {"probe.py": "pinned"}
    archive: dict[str, object] = {"archive": "pinned"}
    monkeypatch.setattr(probe, "_input_expected", lambda _identity, _archive: True)
    child = _valid_child(source, archive)
    child["hessian"]["curvature_verdict"] = "locally_negative_curvature"

    assert not probe._valid_child(child, source, archive)


def test_parent_rejects_inflated_delta_that_masks_positive_curvature(monkeypatch):
    source = {"probe.py": "pinned"}
    archive: dict[str, object] = {"archive": "pinned"}
    monkeypatch.setattr(probe, "_input_expected", lambda _identity, _archive: True)
    child = _valid_child(source, archive)
    hessian = child["hessian"]
    hessian["curvature_threshold"] = 2.0
    hessian["curvature_verdict"] = "near_zero_inconclusive"

    assert not probe._valid_child(child, source, archive)


def _patch_parent_inputs(monkeypatch, source, archive):
    monkeypatch.setattr(probe, "_source_hashes", lambda: source)
    monkeypatch.setattr(probe.first_branch, "_sources_match_archive", lambda _value: True)
    monkeypatch.setattr(probe.first_branch, "_archive_input_identity", lambda: archive)
    monkeypatch.setattr(probe, "_canonical_sha", lambda _value: probe.ARCHIVED_INPUT_SHA256)
    monkeypatch.setattr(probe, "_input_expected", lambda _identity, _archive: True)


def _fake_guard(child, *, exit_code=0):
    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        Path(command[-1]).write_text(json.dumps(child))
        return {
            "command": command, "child_pid": 42, "exit_code": exit_code,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": wall_seconds, "elapsed_seconds": 1.0,
            "rss_limit_bytes": rss_bytes, "rss_samples": 1,
            "sampled_peak_rss_bytes": 100_000_000,
        }
    return guarded


def test_parent_guard_rejects_source_mutation_after_child(monkeypatch, tmp_path):
    source_before = {"probe.py": "before"}
    source_after = {"probe.py": "after"}
    archive: dict[str, object] = {"archive": "pinned"}
    _patch_parent_inputs(monkeypatch, source_before, archive)
    monkeypatch.setattr(probe, "_source_hashes", lambda: source_before)
    child = _valid_child(source_before, archive)
    monkeypatch.setattr(probe, "run_guarded", _fake_guard(child))
    # Parent sees a new source tree only on its post-child identity check.
    calls = iter((source_before, source_after))
    monkeypatch.setattr(probe, "_source_hashes", lambda: next(calls))
    monkeypatch.setattr(probe, "_sha", lambda path: probe.PLAN_SHA256)

    result = probe.run(tmp_path / "source-mutation")

    assert result["execution_status"] == "failed"
    assert result["source_unchanged"] is False
    assert result["child_report_valid"] is True


def test_parent_guard_rejects_plan_mutation_after_child(monkeypatch, tmp_path):
    source = {"probe.py": "pinned"}
    archive: dict[str, object] = {"archive": "pinned"}
    _patch_parent_inputs(monkeypatch, source, archive)
    child = _valid_child(source, archive)
    monkeypatch.setattr(probe, "run_guarded", _fake_guard(child))
    plan_calls = iter((probe.PLAN_SHA256, "changed"))
    monkeypatch.setattr(probe, "_sha", lambda _path: next(plan_calls))

    result = probe.run(tmp_path / "plan-mutation")

    assert result["execution_status"] == "failed"
    assert result["child_report_valid"] is True
