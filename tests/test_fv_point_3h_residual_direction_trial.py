from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_netflux_subspace_trial as shared_trial

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "graphify-out/fv-root-cause-20260919/residual_direction_20261012"
sys.path.insert(0, str(WORK))
import direction_trial as config  # noqa: E402

HELPER_PATH = Path(__file__).with_name("test_fv_point_3h_netflux_subspace_trial.py")
HELPER_SPEC = importlib.util.spec_from_file_location("synthetic_subspace_harness", HELPER_PATH)
assert HELPER_SPEC is not None and HELPER_SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = HARNESS
HELPER_SPEC.loader.exec_module(HARNESS)


def _run_residual_trial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                        *, bad_subspace_repeat: bool = False):
    digest = "synthetic-source-pin"
    sha = shared_trial.tangent._tensor_sha(torch.zeros(26, dtype=torch.float64))
    monkeypatch.setattr(config, "BASE_SHA", sha)
    monkeypatch.setattr(config, "ATTEMPT", tmp_path / "attempt")
    monkeypatch.setattr(config, "load_plan", lambda *_args: {})
    monkeypatch.setattr(config.endpoint, "load_endpoint", lambda: shared_trial.endpoint.load_endpoint())
    original = shared_trial._run_child

    def run_with_residual(path, plan_sha, output):
        return original(path, plan_sha, output, experiment=config)

    monkeypatch.setattr(shared_trial, "_run_child", run_with_residual)
    action_events: list[int] = []
    direction_specs = config.direction_specs
    next_direction = config.next_direction
    jvp_count = {"count": 0}
    actual_jvp = torch.func.jvp

    def count_jvp(*args, **kwargs):
        jvp_count["count"] += 1
        return actual_jvp(*args, **kwargs)

    monkeypatch.setattr(torch.func, "jvp", count_jvp)

    def charted_specs(context, control, theta, gm, gp, normal, chart, pivot):
        # The reused synthetic harness returns a square identity placeholder;
        # retain the 25 columns tangent to its x[0] face for this configured run.
        return direction_specs(context, control, theta, gm, gp, normal, chart[:, 1:], pivot)

    def track_next_direction(model, gm, gp, normal, chart, pivot):
        action_events.append(jvp_count["count"])
        return next_direction(model, gm, gp, normal, chart[:, 1:], pivot)

    monkeypatch.setattr(config, "direction_specs", charted_specs)
    monkeypatch.setattr(config, "next_direction", track_next_direction)
    raw, counts = HARNESS._run(monkeypatch, tmp_path, bad_subspace_repeat=bad_subspace_repeat)
    return raw, counts, action_events


def test_configured_trial_orders_seed_pair_residual_action_and_second_pair(monkeypatch, tmp_path):
    raw, counts, actions = _run_residual_trial(monkeypatch, tmp_path)

    assert raw["execution_status"] == "completed", raw.get("termination_reason")
    assert [model["name"] for model in raw["fresh_models"]] == ["baseline_tangent", "residual_merit"]
    assert [entry["direction"] for entry in raw["hvp_history"]] == [
        "baseline_tangent", "baseline_tangent", "residual_merit", "residual_merit"]
    assert len(actions) == 1
    assert actions == [2]
    assert counts["HVP"] == raw["hvp_started"] == raw["hvp_completed"] == 4
    assert counts["row"] == 0
    assert raw["GN_operator"]["jacobian_rows_started"] == 0
    assert raw["GN_operator"]["jacobian_rows_completed"] == 0
    assert raw["GN_operator"]["dense_solves_completed"] == 0
    assert counts["repeat"] == 2
    assert [arm["status"] for arm in raw["arms"]] == [
        "independent_P2_fullG_eligible", "independent_P2_fullG_eligible"]
    assert raw["comparison_complete"] is True


def test_failed_residual_arm_repeat_cannot_fall_back_to_single_commit(monkeypatch, tmp_path):
    raw, counts, _ = _run_residual_trial(monkeypatch, tmp_path, bad_subspace_repeat=True)

    assert raw["execution_status"] == "failed"
    assert counts["repeat"] == 2
    assert raw["optimizer_steps_applied"] == 0
    assert raw["candidate_committed"] is False
    assert raw["current_control"] == raw["last_confirmed_control"]
    assert all(arm["trials"][0]["finite"] is True for arm in raw["arms"])
