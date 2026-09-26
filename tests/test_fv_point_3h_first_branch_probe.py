from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_first_branch_probe as probe


def _smooth_stage() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    y, x = torch.meshgrid(torch.arange(4, dtype=dtype),
                          torch.arange(5, dtype=dtype), indexing="ij")
    q = 2 + .1 * x + .2 * y + .01 * x * y + .005 * x.square() + .009 * y.square()
    qx = torch.ones((4, 6), dtype=dtype)
    qy = -torch.ones((5, 5), dtype=dtype)
    return q, qx, qy


def test_classifier_keeps_strict_predicate_order_and_counts_simultaneous_failures():
    q = torch.ones((4, 5), dtype=torch.float64)
    qx = torch.zeros((4, 6), dtype=torch.float64)
    qy = torch.ones((5, 5), dtype=torch.float64)

    result = probe.classify_stage(17, q, qx, qy)

    assert result["stage_index"] == 17
    assert result["first_failure"]["predicate"] == "x_left_slope_nonzero"
    assert result["first_failure"]["orientation"] == "x"
    assert result["first_failure"]["first_index"] == [1, 1]
    assert result["first_failure"]["index_space"] == "full_field_interior_center_cell_0based"
    assert result["violation_counts"]["x_right_slope_nonzero"] == 6
    assert result["violation_counts"]["y_left_slope_nonzero"] == 6
    assert result["violation_counts"]["x_face_flux_nonzero"] == 24
    assert result["simultaneous_violation_count"] >= 5


def test_classifier_reports_active_limiter_tie_before_flux_with_global_cell_index():
    dtype = torch.float64
    y, x = torch.meshgrid(torch.arange(4, dtype=dtype),
                          torch.arange(5, dtype=dtype), indexing="ij")
    q = 1 + .1 * x + .2 * y
    result = probe.classify_stage(
        0, q, torch.ones((4, 6), dtype=dtype), -torch.ones((5, 5), dtype=dtype)
    )
    assert result["first_failure"]["predicate"] == "x_active_limiter_operand_difference"
    assert result["first_failure"]["first_index"] == [1, 1]


def test_classifier_reports_y_predicate_after_all_x_predicates_pass():
    dtype = torch.float64
    y, x = torch.meshgrid(torch.arange(4, dtype=dtype),
                          torch.arange(5, dtype=dtype), indexing="ij")
    # Equal y differences produce an active y limiter tie; x differences vary.
    q = 1 + .1 * x + .01 * x.square() + .2 * y + .05 * x * y
    result = probe.classify_stage(
        0, q, torch.ones((4, 6), dtype=dtype), -torch.ones((5, 5), dtype=dtype)
    )
    assert result["first_failure"]["predicate"].startswith("y_")


def test_classifier_reports_first_zero_flux_face_and_full_face_index():
    q, qx, qy = _smooth_stage()
    qx[2, 3] = 0
    result = probe.classify_stage(8, q, qx, qy)
    assert result["first_failure"]["predicate"] == "x_face_flux_nonzero"
    assert result["first_failure"]["first_index"] == [2, 3]
    assert result["first_failure"]["index_space"] == "x_face_array_0based"


def test_classifier_passes_only_when_all_strict_predicates_hold():
    q, qx, qy = _smooth_stage()
    result = probe.classify_stage(3599, q, qx, qy)
    assert result["first_failure"] is None
    assert result["violations"] == []


def test_invalid_field_scale_is_first_and_stops_dependent_predicates():
    q = torch.zeros((4, 5), dtype=torch.float64)
    result = probe.classify_stage(0, q, torch.ones((4, 6)), torch.ones((5, 5)))
    assert result["first_failure"]["predicate"] == "field_scale_finite_positive"
    assert result["simultaneous_violation_count"] == 1


def test_nonfinite_field_scale_can_be_serialized_as_failure_evidence():
    q = torch.full((4, 5), torch.nan, dtype=torch.float64)
    result = probe.classify_stage(0, q, torch.ones((4, 6)), torch.ones((5, 5)))
    assert result["first_failure"]["predicate"] == "field_scale_finite_positive"
    json.dumps(result, allow_nan=False)


def test_first_failure_stops_forecast_after_observer_callback(monkeypatch):
    q = torch.ones((4, 5), dtype=torch.float64)
    qx = torch.zeros((4, 6), dtype=torch.float64)
    qy = torch.ones((5, 5), dtype=torch.float64)
    active = []

    @contextmanager
    def observer_context(callback):
        active.append(callback)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(probe.transport, "observe_minmod_stages", observer_context)
    calls = []

    def fake_forecast():
        calls.append(1)
        for _ in range(5):
            active[-1](q, qx, qy)
        return torch.zeros(())

    result = probe.observe_first_failure(fake_forecast)
    assert len(calls) == 1
    assert result["status"] == "refused_first_branch"
    assert result["stages_observed"] == 1
    assert result["stage_index"] == 0
    assert result["stage_context"] == {
        "phase": "analysis_replay", "interval_index": 0,
        "interval_start_seconds": 0, "interval_end_seconds": 600,
        "forecast_lead_index": None, "substep_index": 0,
        "ssprk_euler_stage_index": 0,
    }


@pytest.mark.parametrize(
    ("stage", "phase", "lead", "substep", "euler_stage"),
    [(359, "analysis_replay", None, 89, 1),
     (360, "future_forecast", 1, 0, 0),
     (539, "future_forecast", 1, 89, 1),
     (3599, "future_forecast", 18, 89, 1)],
)
def test_global_stage_context_maps_analysis_and_forecast_intervals(
    stage, phase, lead, substep, euler_stage,
):
    context = probe._stage_context(stage)
    assert context["phase"] == phase
    assert context["forecast_lead_index"] == lead
    assert context["substep_index"] == substep
    assert context["ssprk_euler_stage_index"] == euler_stage


def test_refused_branch_is_a_completed_diagnostic_when_child_and_guard_succeed(
    monkeypatch, tmp_path: Path,
):
    identity = probe._archive_input_identity()
    source = probe._source_hashes()
    child: dict[str, object] = {
        "pid": 4321, "execution_status": "completed", "execution_phase": "finished",
        "diagnostic_status": "refused_first_branch", "stationarity_passed": "not_tested",
        "response_computed": False, "response_validation": "not_performed",
        "physical_validation": "not_performed", "forecast_score_computed": False,
        "source_before": source, "source_after": source, "source_unchanged": True,
        "input_before": identity, "input_after": identity, "input_unchanged": True,
        "plan_sha256": probe.PLAN_SHA256, "plan_unchanged": True,
        "archive": {"manifest_sha256": probe.ARCHIVE_MANIFEST_SHA256,
                    "forward_evidence_sha256": probe.FORWARD_EVIDENCE_SHA256,
                    "policy_plan_sha256": probe.POLICY_PLAN_SHA256,
                        "policy_results_sha256": probe.POLICY_RESULTS_SHA256,
                        "policy_record_sha256": probe.POLICY_RECORD_SHA256,
                        "policy_evidence_sha256": probe.POLICY_EVIDENCE_SHA256,
                        "input_identity_sha256": probe.ARCHIVED_INPUT_SHA256,
                        "archived_profile_source_sha256": probe.PINNED_ARCHIVE_SOURCES,
                        "current_guard_source_sha256": source[
                            "examples/weather_scenarios/fv86_resource_runner.py"
                        ]},
        "archive_unchanged": True,
        "layout": {"controls": 26, "parameters": 13, "state_shape": [4, 5],
                   "observation_shape": [3, 4],
                   "observation_times_seconds": [0.0, 600.0, 1200.0],
                   "forecast_time_seconds": 12000.0, "euler_stages": 3600},
        "diagnostic": {"status": "refused_first_branch", "stages_observed": 1,
                       "accepted_stages_before_failure": 0, "stage_index": 0,
                       "first_failure": {"predicate": "x_left_slope_nonzero", "count": 6},
                       "violations": [{"predicate": "x_left_slope_nonzero", "count": 6}],
                       "simultaneous_violation_count": 1,
                       "stage_context": probe._stage_context(0)},
    }

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 180 and rss_bytes == 768 * 1024**2
        assert command[-2] == "--output"
        Path(command[-1]).write_text(json.dumps(child))
        report = {"command": command, "child_pid": 4321, "exit_code": 0,
                  "resource_termination": None, "monitor_error": None,
                  "wall_limit_seconds": 180, "elapsed_seconds": 2.0,
                  "rss_limit_bytes": 768 * 1024**2, "rss_samples": 4,
                  "sampled_peak_rss_bytes": 300_000_000}
        Path(report_path).write_text(json.dumps(report))
        Path(log_path).write_text("")
        return report

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / "guarded")
    assert result["execution_status"] == "completed"
    assert result["diagnostic_status"] == "refused_first_branch"
    assert result["stationarity_passed"] == "not_tested"
    assert result["response_computed"] is False


def test_parent_rejects_wrong_child_exit_even_if_refusal_report_is_valid(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_archive_input_identity", lambda: {"fixed": "input"})
    monkeypatch.setattr(probe, "_source_hashes", lambda: {"source": "sha"})
    monkeypatch.setattr(probe, "_sources_match_archive", lambda _source: True)
    child = {"execution_status": "completed", "execution_phase": "finished",
             "diagnostic_status": "refused_first_branch", "stationarity_passed": "not_tested",
             "response_computed": False, "response_validation": "not_performed",
             "physical_validation": "not_performed", "forecast_score_computed": False,
             "diagnostic": {"status": "refused_first_branch", "stages_observed": 1,
                            "accepted_stages_before_failure": 0, "stage_index": 0,
                            "first_failure": {"predicate": "x_left_slope_nonzero"},
                            "simultaneous_violation_count": 1}}

    def fake_guard(command, **kwargs):
        Path(command[-1]).write_text(json.dumps(child))
        return {"command": command, "child_pid": 2, "exit_code": 2,
                "resource_termination": None, "monitor_error": None,
                "wall_limit_seconds": 180, "elapsed_seconds": 1.0,
                "rss_limit_bytes": 768 * 1024**2, "rss_samples": 2,
                "sampled_peak_rss_bytes": 100}

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / "bad-exit")
    assert result["execution_status"] == "failed"
    assert result["diagnostic_status"] == "refused_first_branch"
