"""Saved-evidence and contract checks for external FV background composition."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "weather_scenarios"))

import fv_external_background_phase1 as probe  # noqa: E402


REPORT = Path(__file__).parents[1] / "graphify-out/fv-root-cause-20260919/fv_external_background_phase1.json"


def assert_phase1_evidence(report: dict) -> None:
    chain = report["chain_rule"]
    assert report["parameter_count"] == 3
    assert report["theta_initial"] == [0.1, -0.2, 0.15]
    assert not report["production_learning_eligible"]
    assert not report["typed_neural_prior_application"]
    assert not report["legacy_fsoi_eligible"]
    assert not report["truth_seeded_background"]
    assert report["fixed_support_and_sigma"]
    assert chain["total_y_stationarity_max"] < 1.0e-8
    assert chain["background_stationarity_max"] < 1.0e-8
    assert chain["theta_chain_max_error"] < 1.0e-12
    assert chain["total_y_chain_max_error"] < 1.0e-12
    assert chain["background_y_nonidentity_norm"] > 1.0e-10
    for rows in (report["theta_finite_reanalysis"], report["y_finite_reanalysis"]):
        assert len(rows) == 2
        for row in rows:
            assert all(math.isfinite(row[name]) for name in ("centered_slope", "predicted_slope", "slope_error"))
            assert row["plus_gradient_max"] < 1.0e-8
            assert row["minus_gradient_max"] < 1.0e-8
            assert row["face_box_margin"] > 0.0
            assert row["slope_error"] <= max(1.0e-9, 1.0e-5 * abs(row["predicted_slope"]))
    assert report["train_score_after"] < report["train_score_before"]
    assert math.isfinite(report["holdout_score_before"])
    assert math.isfinite(report["holdout_score_after"])
    assert math.isfinite(report["holdout_gain"])
    reload_errors = report["checkpoint_reload"]
    assert reload_errors is not None
    assert all(value <= 1.0e-12 for value in reload_errors.values())


def test_background_uses_only_first_observation_and_is_bounded() -> None:
    observations, _, _, _ = probe.oracle.make_case()
    theta = torch.tensor((0.2, -0.3, 0.4), dtype=torch.float64)
    baseline = probe.background(theta, observations.dbz)
    later_changed = observations.dbz.clone()
    later_changed[1:] += 20.0
    assert torch.equal(baseline, probe.background(theta, later_changed))
    first_changed = observations.dbz.clone()
    first_changed[0, 1, 1] += 0.01
    assert not torch.equal(baseline, probe.background(theta, first_changed))
    assert bool((baseline - observations.dbz[0]).abs().lt(0.1).all())


def test_saved_phase1_chain_and_reload_evidence() -> None:
    assert_phase1_evidence(json.loads(REPORT.read_text()))
