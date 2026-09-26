from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_seed_linear_probe as probe


def test_margin_prepass_covers_full_forecast_before_strict_refusal(monkeypatch):
    observers = []
    stage = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
                         dtype=torch.float64)

    @contextmanager
    def observe(callback):
        observers.append(callback)
        yield

    class Problem:
        def forecast(self, _control, _parameters):
            for _ in range(probe.EXPECTED_STAGES):
                observers[-1](stage, torch.ones((2, 3), dtype=stage.dtype),
                              torch.full((3, 2), 2.0, dtype=stage.dtype))

        def branch_check(self, _control, _parameters):
            raise ValueError("strict branch refused at first stage")

    monkeypatch.setattr(probe.transport, "observe_minmod_stages", observe)
    branch, margins = probe._full_branch_with_margins(
        Problem(), torch.zeros(26, dtype=torch.float64),
        torch.zeros(13, dtype=torch.float64),
    )

    assert branch["status"] == "branch_refused"
    assert margins["complete"] is True
    assert margins["observed_stage_count"] == 3600
    assert margins["absolute_face_flux_over_global_maximum"]["value"] == 0.5
    assert margins["normalized_minima"]["x_left_slope_over_q_scale"]["stage_index"] == 0


def test_full_branch_requires_all_margin_stages(monkeypatch):
    observers = []

    @contextmanager
    def observe(callback):
        observers.append(callback)
        yield

    class Problem:
        def forecast(self, _control, _parameters):
            q = torch.ones((3, 3), dtype=torch.float64)
            observers[-1](q, torch.ones((2, 3), dtype=q.dtype),
                          torch.ones((3, 2), dtype=q.dtype))

        def branch_check(self, *_args):
            return ({"euler_stages": 3600, "choices": [1] * 3600,
                     "face_signs": [1] * 3600}, "strict callback")

    monkeypatch.setattr(probe.transport, "observe_minmod_stages", observe)
    branch, margins = probe._full_branch_with_margins(
        Problem(), torch.zeros(26, dtype=torch.float64),
        torch.zeros(13, dtype=torch.float64),
    )
    assert branch["status"] == "branch_diagnostic_incomplete"
    assert margins["complete"] is False
    assert margins["observed_stage_count"] == 1


def test_linear_preflight_solves_one_rhs_and_recomputes_true_residual():
    control = torch.tensor([1.0, -2.0], dtype=torch.float64)
    before = control.clone()
    problem = SimpleNamespace(objective=lambda value, _p: 0.5 * torch.dot(value, value))
    gradient = control.clone()

    result = probe._linear_preflight(problem, control, torch.zeros(1, dtype=control.dtype), gradient)

    assert result["status"] == "linear_pass"
    assert result["pcg_converged"] is True
    assert result["iterations"] == 1
    # PCG uses one Krylov product and one internal residual product; the probe
    # adds a third, independent product for its reported true residual.
    assert result["hvp_calls"] == 3
    assert result["true_relative_residual"] <= probe.LINEAR_RTOL
    assert torch.equal(control, before)
    assert result["curvature_certificate"] == "not_proved_by_pcg"


def test_stationary_zero_rhs_skips_pcg_and_leaves_curvature_unverified(monkeypatch):
    problem = SimpleNamespace(objective=lambda value, _p: 0.5 * torch.dot(value, value))
    monkeypatch.setattr(probe.matrix_free, "pcg", lambda *_a, **_kw: pytest.fail("zero RHS PCG"))
    result = probe._linear_preflight(
        problem, torch.zeros(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    assert result["status"] == "skipped_stationary_zero_rhs"
    assert result["hvp_calls"] == 0
    assert result["curvature_certificate"] == "unverified_zero_rhs_skipped"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("operator must be symmetric positive definite"), "pcg_refused_non_spd"),
        (RuntimeError("true residual norm is not finite"), "nonfinite_derivative"),
    ],
)
def test_linear_preflight_classifies_only_known_pcg_runtime_failures(monkeypatch, error, expected):
    problem = SimpleNamespace(objective=lambda value, _p: 0.5 * torch.dot(value, value))
    monkeypatch.setattr(probe.matrix_free, "pcg", lambda *_a, **_kw: (_ for _ in ()).throw(error))
    result = probe._linear_preflight(
        problem, torch.ones(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )
    assert result["status"] == expected
    assert result["hvp_calls"] == 0


def test_unexpected_pcg_callback_error_is_not_hidden(monkeypatch):
    problem = SimpleNamespace(objective=lambda value, _p: 0.5 * torch.dot(value, value))
    monkeypatch.setattr(probe.matrix_free, "pcg", lambda *_a, **_kw: (_ for _ in ()).throw(
        RuntimeError("objective callback bug")
    ))
    with pytest.raises(RuntimeError, match="objective callback bug"):
        probe._linear_preflight(
            problem, torch.ones(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
        )


def _minimal_child(source: dict[str, str], archive: dict[str, str]) -> dict[str, object]:
    identity = {
        "archived_input": archive,
        "changed_control_index": probe.EXPECTED_CHANGED_CONTROL,
        "changed_control_indices": [probe.EXPECTED_CHANGED_CONTROL],
        "control_sha256": probe.SHIFTED_SEED_CONTROL_SHA256,
        "parameters_sha256": probe.EXPECTED_PARAMETERS_SHA256,
        "terminal_truth_sha256": probe.EXPECTED_TRUTH_SHA256,
    }
    minimum = {
        "value": 0.1, "stage_index": 0,
        "stage_context": probe.first_branch._stage_context(0),
    }
    margins = {
        "complete": True, "nonfinite": False,
        "observed_stage_count": 3600, "expected_stage_count": 3600,
        "normalized_minima": {
            name: minimum for name in (
                "x_left_slope_over_q_scale", "x_right_slope_over_q_scale",
                "x_active_limiter_gap_over_q_scale", "y_left_slope_over_q_scale",
                "y_right_slope_over_q_scale", "y_active_limiter_gap_over_q_scale",
            )
        },
        "absolute_face_flux_over_global_maximum": {
            "value": 0.5, "absolute_minimum": 0.1,
            "global_maximum_absolute_flux": 0.2,
            "stage_index": 0,
            "stage_context": probe.first_branch._stage_context(0),
            "orientation": "x", "index": [0, 0],
        },
    }
    return {
        "pid": 99, "execution_status": "completed", "execution_phase": "finished",
        "child_elapsed_seconds": 1.0,
        "child_phase_seconds": {"source_and_fixture": 0.1,
                                 "full_forecast_and_strict_branch": 0.9},
        "wall_limit_seconds": probe.WALL_SECONDS,
        "rss_limit_bytes": probe.SAMPLED_RSS_BYTES,
        "termination_reap_grace_seconds": probe.REAP_GRACE_SECONDS,
        "numerical_status": "branch_refused", "branch_status": "branch_refused",
        "stationarity_status": "not_tested", "linear_status": "not_attempted",
        "branch": {"status": "branch_refused", "reason": "strict predicate"},
        "branch_margins": margins,
        "source_before": source, "source_after": source, "source_unchanged": True,
        "plan_sha256": probe.PLAN_SHA256, "plan_unchanged": True,
        "shifted_plan_sha256": probe.SHIFTED_PLAN_SHA256,
        "shifted_plan_unchanged": True,
        "archive_input_sha256": probe.ARCHIVED_INPUT_SHA256,
        "archive_before": archive, "archive_unchanged": True,
        "input_before": identity, "input_after": identity, "input_unchanged": True,
        "response_computed": False, "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "layout": {"controls": 26, "parameters": 13, "euler_stages": 3600},
        "candidate": {
            "control_sha256": probe.SHIFTED_SEED_CONTROL_SHA256,
            "parameters_sha256": probe.EXPECTED_PARAMETERS_SHA256,
            "terminal_truth_sha256": probe.EXPECTED_TRUTH_SHA256,
        },
    }


def test_parent_guard_does_not_rebuild_fixture_and_rejects_bad_exit(monkeypatch, tmp_path):
    source = {
        "probe.py": "pinned",
        "examples/weather_scenarios/fv_point_3h_shifted_branch_probe.py":
            probe.SHIFTED_HELPER_SHA256,
    }
    archive = {"input": "fixed"}
    monkeypatch.setattr(probe, "_source_hashes", lambda: source)
    monkeypatch.setattr(probe.first_branch, "_sources_match_archive", lambda _s: True)
    monkeypatch.setattr(probe.first_branch, "_archive_input_identity", lambda: archive)
    monkeypatch.setattr(probe, "_sha", lambda path: (
        probe.SHIFTED_PLAN_SHA256 if path == probe.shifted_seed.PLAN else probe.PLAN_SHA256
    ))
    monkeypatch.setattr(probe, "_prepare_fixed_seed", lambda: pytest.fail(
        "parent must not rebuild the full FV fixture outside run_guarded"
    ))

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path):
        child = _minimal_child(source, archive)
        Path(command[-1]).write_text(json.dumps(child))
        return {
            "command": command, "child_pid": 99, "exit_code": 2,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": wall_seconds, "elapsed_seconds": 1.0,
            "rss_limit_bytes": rss_bytes, "rss_samples": 1,
            "sampled_peak_rss_bytes": 100_000_000,
        }

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / "attempt")
    assert result["execution_status"] == "failed"
    assert result["numerical_status"] == "branch_refused"
    assert result["child_report_valid"] is True
    assert result["resource"]["exit_code"] == 2


def test_parent_accepts_only_a_valid_completed_child(monkeypatch, tmp_path):
    source = {
        "probe.py": "pinned",
        "examples/weather_scenarios/fv_point_3h_shifted_branch_probe.py":
            probe.SHIFTED_HELPER_SHA256,
    }
    archive = {"input": "fixed"}
    monkeypatch.setattr(probe, "_source_hashes", lambda: source)
    monkeypatch.setattr(probe.first_branch, "_sources_match_archive", lambda _s: True)
    monkeypatch.setattr(probe.first_branch, "_archive_input_identity", lambda: archive)
    monkeypatch.setattr(probe, "_sha", lambda path: (
        probe.SHIFTED_PLAN_SHA256 if path == probe.shifted_seed.PLAN else probe.PLAN_SHA256
    ))
    monkeypatch.setattr(probe, "_prepare_fixed_seed", lambda: pytest.fail("parent fixture build"))

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path):
        Path(command[-1]).write_text(json.dumps(_minimal_child(source, archive)))
        return {
            "command": command, "child_pid": 99, "exit_code": 0,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": wall_seconds, "elapsed_seconds": 1.0,
            "rss_limit_bytes": rss_bytes, "rss_samples": 1,
            "sampled_peak_rss_bytes": 100_000_000,
        }

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / "attempt")
    assert result["execution_status"] == "completed"
    assert result["numerical_status"] == "branch_refused"
    assert result["resource"]["rss_limit_bytes"] == 1024**3


def test_parent_rejects_child_branch_status_contradiction(monkeypatch, tmp_path):
    source = {
        "probe.py": "pinned",
        "examples/weather_scenarios/fv_point_3h_shifted_branch_probe.py":
            probe.SHIFTED_HELPER_SHA256,
    }
    archive = {"input": "fixed"}
    monkeypatch.setattr(probe, "_source_hashes", lambda: source)
    monkeypatch.setattr(probe.first_branch, "_sources_match_archive", lambda _s: True)
    monkeypatch.setattr(probe.first_branch, "_archive_input_identity", lambda: archive)
    monkeypatch.setattr(probe, "_sha", lambda path: (
        probe.SHIFTED_PLAN_SHA256 if path == probe.shifted_seed.PLAN else probe.PLAN_SHA256
    ))
    child = _minimal_child(source, archive)
    child["branch_status"] = "passed_strict_branch"
    monkeypatch.setattr(probe, "_prepare_fixed_seed", lambda: pytest.fail("parent fixture build"))

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path):
        Path(command[-1]).write_text(json.dumps(child))
        return {
            "command": command, "child_pid": 99, "exit_code": 0,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": wall_seconds, "elapsed_seconds": 1.0,
            "rss_limit_bytes": rss_bytes, "rss_samples": 1,
            "sampled_peak_rss_bytes": 100_000_000,
        }

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / "attempt")
    assert result["execution_status"] == "failed"
    assert result["child_report_valid"] is False
