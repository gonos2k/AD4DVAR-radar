"""Point-observation terminal-lead contract and guarded publication gates."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_forward_case as case_module
from examples.weather_scenarios import fv_point_3h_forward_probe as probe
from examples.weather_scenarios import fv_point_3h_forward_runner as runner
from examples.weather_scenarios import fv_point_response_preflight as preflight


def test_long_point_layout_schedule_and_external_background():
    problem, control, parameters, truth = case_module.make_case()
    layout = problem.layout
    assert layout["controls"] == 26 and layout["parameters"] == 13
    assert layout["observation_shape"] == (3, 4)
    assert layout["observation_times_seconds"] == (0.0, 600.0, 1200.0)
    assert layout["forecast_time_seconds"] == 12000.0
    assert layout["euler_stages"] == 3600
    assert problem.frozen.nowcast_config.forecast_steps == 18
    assert problem.frozen.fv_transport is not None
    assert problem.frozen.fv_transport.substeps_per_interval == 90
    assert len(problem.frozen.fv_transport.boundary_echo) == 180
    assert len(problem.future_boundary_echo) == 1620
    assert bool(problem.frozen.initial_support_mask.all())
    assert torch.equal(problem.background_dbz, problem.frozen.initial_background_dbz)
    assert problem.observation_status is None
    assert bool((problem.observation_dbz > problem.frozen.analysis_config.detection_limit_dbz).all())
    assert control.shape == (26,) and parameters.shape == (13,) and truth.shape == (4, 5)
    assert probe._sha(probe.PLAN) == probe.PLAN_SHA256


def test_future_boundary_must_cover_all_18_leads():
    problem, _, _, _ = case_module.make_case()
    with pytest.raises(ValueError, match="boundary"):
        replace(problem, future_boundary_echo=problem.future_boundary_echo[:-1])


def test_original_one_lead_layout_and_input_identity_stay_unchanged():
    problem, warm, parameters, direction = preflight.fixed_problem()
    assert problem.frozen.nowcast_config.forecast_steps == 1
    assert problem.layout["forecast_time_seconds"] == 180.0
    assert problem.layout["euler_stages"] == 54
    saved = json.loads((probe.EVIDENCE / "point_sector_root_attempt1/point_sector_root.json").read_text())
    assert preflight._input_identity(problem, warm, parameters, direction) == saved["input_before"]


def test_parent_rejects_incomplete_child_even_with_clean_resource(monkeypatch, tmp_path):
    problem, control, parameters, truth = case_module.make_case()
    monkeypatch.setattr(runner, "make_case", lambda: (problem, control, parameters, truth))
    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 180 and rss_bytes == 1024**3
        assert Path(command[1]).is_absolute() and Path(command[1]).is_file()
        child = {"pid": 123, "forward_validation": "passed", "response_validation": "not_performed"}
        Path(command[-1]).write_text(json.dumps(child))
        resource = {"command": command, "child_pid": 123, "exit_code": 0,
                    "resource_termination": None, "monitor_error": None,
                    "wall_limit_seconds": 180, "elapsed_seconds": 10.0,
                    "rss_limit_bytes": 1024**3, "rss_samples": 3,
                    "sampled_peak_rss_bytes": 300_000_000}
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource
    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / "attempt")
    assert result["execution_status"] == "failed"
    assert result["response_validation"] == "not_performed"


def test_parent_accepts_complete_json_roundtrip_layout(monkeypatch, tmp_path):
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)
    observations = torch.zeros((3, 4), dtype=torch.float64)
    layout = {
        "state_shape": (4, 5), "observation_shape": (3, 4),
        "controls": 26, "parameters": 13,
        "observation_times_seconds": (0.0, 600.0, 1200.0),
        "forecast_time_seconds": 12000.0, "euler_stages": 3600,
    }
    active = [None]
    dummy = torch.zeros(1, dtype=torch.float64)
    @contextmanager
    def observe(callback):
        previous = active[0]
        active[0] = callback
        try:
            yield
        finally:
            active[0] = previous
    def emit(count):
        assert active[0] is not None
        for _ in range(count):
            active[0](dummy, dummy, dummy)
    def analysis(_control, _contract):
        emit(360)
        return SimpleNamespace(frames_linear=torch.zeros((3, 4, 5), dtype=torch.float64))
    class FakeProblem:
        def __init__(self, layout_value):
            self.layout = layout_value
        identity = {"fixture": "json-layout"}
        frozen = SimpleNamespace(nowcast_config=SimpleNamespace(min_dbz=0.0))
        observation_coordinates = torch.zeros((4, 2), dtype=torch.float64)
        observation_dbz = observations
        def contract(self, _parameters):
            return None
        def forecast(self, _control, _parameters):
            emit(3600)
            return truth
        def objective(self, _control, _parameters):
            return torch.zeros((), dtype=torch.float64)
    problem = FakeProblem(layout)
    monkeypatch.setattr(runner, "make_case", lambda: (problem, control, parameters, truth))
    monkeypatch.setattr(runner.transport, "observe_minmod_stages", observe)
    monkeypatch.setattr(runner.v, "analysis_trajectory", analysis)
    monkeypatch.setattr(runner, "echo_to_dbz", lambda frames, **_kwargs: frames)
    monkeypatch.setattr(runner, "point_dbz_bilinear", lambda *_args: observations)
    branch_diagnostic = [{"status": "refused", "reason": "analytic fixture",
                          "observed_minmod_stages": 0}]

    def guarded(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 180 and rss_bytes == 1024**3
        child = {
            "pid": 123, "source_before": probe._sources(),
            "source_after": probe._sources(), "source_unchanged": True,
            "input_before": probe._identity(problem, control, parameters, truth),
            "input_after": probe._identity(problem, control, parameters, truth),
            "input_unchanged": True, "plan_unchanged": True,
            "archive_unchanged": True,
            "plan_sha256": probe.PLAN_SHA256,
            "archive_sha256": probe._sha(probe.preflight.ARCHIVED_WARM),
            "execution_phase": "finished", "forward_validation": "passed",
            "forecast_available": True,
            "stationarity_passed": "not_tested", "response_computed": False,
            "response_validation": "not_performed", "physical_validation": "not_performed",
            "layout": layout, "forecast_steps": 18, "substeps_per_interval": 90,
            "analysis_boundary_stage_pairs": 180,
            "future_boundary_stage_pairs": 1620,
            "analysis": {"observed_minmod_stages": 360,
                         "point_observation_max_abs_difference_dbz": 0.0},
            "terminal": {"observed_minmod_stages": 3600,
                         "forecast_shape": [4, 5], "forecast": truth.tolist(),
                         "forecast_sha256": probe._tensor_sha(truth),
                         "terminal_truth_sha256": probe._tensor_sha(truth),
                         "max_abs_difference_from_same_operator_truth_dbz": 0.0,
                         "objective": 0.0, "score": 0.0,
                         "manual_terminal_mse": 0.0},
            "strict_branch_diagnostic": branch_diagnostic[0],
        }
        Path(command[-1]).write_text(json.dumps(child))
        resource = {"command": command, "child_pid": 123, "exit_code": 0,
                    "resource_termination": None, "monitor_error": None,
                    "wall_limit_seconds": 180, "elapsed_seconds": 10.0,
                    "rss_limit_bytes": 1024**3, "rss_samples": 3,
                    "sampled_peak_rss_bytes": 300_000_000}
        Path(report_path).write_text(json.dumps(resource))
        Path(log_path).write_text("")
        return resource
    monkeypatch.setattr(runner, "run_guarded", guarded)
    result = runner.run(tmp_path / "complete")
    assert result["execution_status"] == "completed"
    assert result["forward_validation"] == "passed"
    assert result["response_validation"] == "not_performed"
    # A branch-success label must cover the whole replayed trajectory.
    branch_diagnostic[0] = {
        "status": "passed_pointwise", "euler_stages": 3600,
        "observed_minmod_stages": 0,
    }
    assert runner.run(tmp_path / "short_branch")["execution_status"] == "failed"


def test_point_three_hour_clis_import_from_unrelated_cwd(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("fv_point_3h_forward_probe.py", "fv_point_3h_forward_runner.py"):
        script = root / "examples/weather_scenarios" / name
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--output" in completed.stdout or "--directory" in completed.stdout
