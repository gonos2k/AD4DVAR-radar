"""Cheap gates for a separately budgeted partial-FV numerical experiment."""
import json

import pytest

from examples.weather_scenarios import fv_partial_reanalysis_preflight as preflight
from examples.weather_scenarios import fv_partial_reanalysis_probe as probe
from examples.weather_scenarios import fv_partial_reanalysis_runner as runner


def test_source_bound_preflight_does_not_solve_and_checks_both_branch_margins():
    report = preflight.run()
    assert report["status"] == "preflight_only"
    assert report["numerical_solver_runs"] == 0
    assert report["observation_counts"] == {
        "valid": 58, "missing": 2, "censored": 0, "qc_rejected": 0,
    }
    branch = report["warm_start_branch"]
    assert branch["euler_stages"] == 54
    assert branch["minimum_scaled_slope_margin"] > 1e-4
    assert branch["minimum_scaled_face_flux_margin"] > 1e-4
    assert report["warm_start_gradient_max"] > 1e-10
    assert report["source_sha256"]["examples/weather_scenarios/fv_partial_reanalysis_preflight.py"]


def test_guarded_runner_refuses_changed_preflight_before_numerical_child(tmp_path, monkeypatch):
    calls = []
    pinned = json.loads(runner.PINNED.read_text())

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path):
        calls.append(command)
        if len(calls) != 1:
            pytest.fail("numerical child launched after preflight identity mismatch")
        report = json.loads(json.dumps(pinned))
        report["tensor_sha256"]["parameters"] = "changed"
        runner._paths(tmp_path)["preflight"].write_text(json.dumps(report))
        return {"exit_code": 0, "resource_termination": None}

    monkeypatch.setattr(runner, "run_guarded", fake_guard)
    with pytest.raises(ValueError, match="tensor_sha256"):
        runner.run(tmp_path)
    assert len(calls) == 1


def test_guarded_runner_preserves_existing_artifacts(tmp_path, monkeypatch):
    runner._paths(tmp_path)["result"].write_text("saved")

    def unexpected_guard(*args, **kwargs):
        pytest.fail("existing experiment was overwritten")

    monkeypatch.setattr(runner, "run_guarded", unexpected_guard)
    with pytest.raises(FileExistsError, match="preserve"):
        runner.run(tmp_path)


def test_numerical_driver_identity_and_directional_arithmetic_are_fail_closed():
    pinned = {"base_commit": "old", "tensor_sha256": {"parameters": "fixed"}}
    probe._require_close_identity(pinned, {**pinned, "base_commit": "new"})
    with pytest.raises(ValueError, match="tensor_sha256"):
        probe._require_close_identity(
            pinned, {**pinned, "tensor_sha256": {"parameters": "changed"}}
        )

    separation, floor, eligible = probe.score_signal_gate(1e-3, 1e-3, 0.1)
    assert eligible and separation > floor
    assert not probe.score_signal_gate(1e-3, 1e-20, 0.1)[2]
    slope, relative_error = probe.central_response_error(
        0.1 + 1e-6, 0.1 - 1e-6, 1e-3, -1e-3
    )
    assert slope == pytest.approx(-1e-3)
    assert relative_error < 1e-10
    with pytest.raises(ValueError, match="nonzero"):
        probe.central_response_error(0.1, 0.1, 1e-3, 0.0)
