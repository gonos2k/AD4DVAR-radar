"""Bounded exact-stationary FV observation-scale learning regression."""

from __future__ import annotations

import math
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "weather_scenarios"))

from fv_learning_probe import run_probe  # noqa: E402


def test_exact_stationary_observation_scale_update_has_reanalysis_and_reload_evidence(
    tmp_path: Path,
) -> None:
    report = run_probe(checkpoint_path=tmp_path / "fv_observation_scale.pt")
    assert_learning_evidence(report)


def assert_learning_evidence(report):
    assert report["control_count"] <= 32
    assert report["curvature"] == "exact_robust_hessian"
    assert report["stationarity_max"] < 1.0e-8
    assert report["finite_difference_stationarity_max"] < 1.0e-8
    assert report["adjoint_relative_residual"] < 1.0e-8
    assert report["branch_margin"] > 0.0
    assert report["same_face_signs"]
    assert report["updated_log_scale"] != report["initial_log_scale"]

    finite_difference = report["finite_difference_gradients"]
    assert len(finite_difference) == 3
    assert all(math.isfinite(value) for value in finite_difference)
    errors = [abs(value - report["parameter_gradient"]) for value in finite_difference]
    assert all(3.0 < large / small < 5.0 for large, small in zip(errors, errors[1:]))
    assert errors[-1] < 1e-5 * abs(report["parameter_gradient"])
    assert report["finite_difference_last_error"] <= 2.0 * report[
        "finite_difference_step_error"
    ]

    train_change = report["train_score_after"] - report["train_score_before"]
    assert train_change < 0.0
    assert report["predicted_impact"] < 0.0
    assert abs(train_change - report["predicted_impact"]) <= 0.01 * abs(train_change)
    assert report["holdout_score_after"] < report["holdout_score_before"]
    assert report["holdout_gain_above_error_budget"]
    for key in ("holdout_base_diagnostics", "holdout_updated_diagnostics"):
        assert report[key]["gradient_max"] < 1e-10
        assert report[key]["hessian_min_eigenvalue"] > 0
    assert report["reload_train_score_error"] <= 1.0e-8
    assert report["reload_holdout_score_error"] <= 1.0e-8
    assert report["reload_holdout_control_max_abs"] <= 1.0e-8
    assert report["reload_holdout_forecast_max_abs"] <= 1.0e-8
