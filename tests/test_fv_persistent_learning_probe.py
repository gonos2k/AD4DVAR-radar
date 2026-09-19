"""Saved evidence checks and a reproducible persistent-learning integration test."""

import json
from pathlib import Path
import sys


EVIDENCE = (
    Path(__file__).parents[1]
    / "graphify-out"
    / "fv-root-cause-20260919"
    / "fv_persistent_learning.json"
)


def assert_persistent_evidence(report: dict[str, object]) -> None:
    assert not report["production_learning_eligible"]
    assert not report["neural_prior_eligible"]
    assert report["parameter_count"] == 3
    assert report["distinct_training_windows"]
    assert report["window_schedule"] == ["train_a", "train_b", "train_a"]
    assert report["heldout_used_for_updates"] is False
    assert report["heldout_data_immutable"]

    uninterrupted = report["uninterrupted_steps"]
    resumed = report["resumed_steps"]
    assert len(uninterrupted) == 3
    assert len(resumed) == 2
    assert [item["step"] for item in uninterrupted] == [1, 2, 3]
    assert [item["step"] for item in resumed] == [2, 3]
    assert [item["window"] for item in uninterrupted] == report["window_schedule"]

    for item in uninterrupted:
        assert item["stationarity_max"] < 1.0e-8
        assert item["adjoint_relative_residual"] < 1.0e-8
        assert item["hessian_min_eigenvalue"] > 0.0
        assert item["train_score_after"] < item["train_score_before"]
        check = item["gradient_direction_check"]
        assert check["same_face_signs"]
        assert check["endpoint_stationarity_max"] < 1.0e-8
        assert min(check["face_branch_margins"]) > 0.0
        assert check["finite_difference_errors"][-1] < 1.0e-5 * abs(
            check["predicted_directional_slope"]
        )

    assert report["holdout_score_after_uninterrupted"] < report["holdout_score_before"]
    assert report["holdout_gain_uninterrupted"] > 0.0
    assert report["holdout_gain_above_roundoff"]
    assert report["resume_max_parameter_difference"] <= 1.0e-12
    assert report["resume_max_optimizer_state_difference"] <= 1.0e-12
    assert report["final_forecast_max_difference"] <= 1.0e-12


def test_saved_persistent_learning_evidence() -> None:
    report = json.loads(EVIDENCE.read_text())
    assert_persistent_evidence(report)


def test_persistent_learning_reanalysis_and_resume(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "examples/weather_scenarios"))
    from fv_persistent_learning_probe import run_probe

    assert_persistent_evidence(run_probe(checkpoint_path=tmp_path / "step1.pt"))
