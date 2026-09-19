"""Focused evidence for the exploratory mean-only prior update."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "weather_scenarios"))

from prior_learning import MeanOnlyPrior, run_mean_only_learning  # noqa: E402


@pytest.mark.parametrize("learning_rate", (0.25,))
def test_mean_only_update_has_independent_fso_fd_and_reload_evidence(
    tmp_path: Path,
    learning_rate: float,
) -> None:
    checkpoint = tmp_path / "prior.pt"
    result = run_mean_only_learning(
        checkpoint_path=checkpoint,
        learning_rate=learning_rate,
    )
    assert result.fso_contract.startswith("p1-variational-fso-")
    assert result.fsoi_contract.startswith("p1-linearized-observation-impact-")
    assert result.gB_norm > 0.0 and math.isfinite(result.gB_norm)
    assert result.initial_offset != result.updated_offset
    assert math.isfinite(result.parameter_gradient)
    assert all(math.isfinite(value) for value in result.finite_difference_gradients)
    assert abs(
        result.parameter_gradient - result.finite_difference_gradients[-1]
    ) <= 0.10 * max(abs(result.finite_difference_gradients[-1]), 1.0e-10)
    assert abs(
        result.finite_difference_gradients[-1]
        - result.finite_difference_gradients[-2]
    ) <= 0.10 * max(abs(result.finite_difference_gradients[-1]), 1.0e-10)
    assert result.half_taylor_error <= 0.10
    assert result.full_taylor_error <= 0.10
    assert result.objective_after < result.objective_before
    assert math.isfinite(result.heldout_before)
    assert math.isfinite(result.heldout_after)
    assert result.heldout_after < result.heldout_before
    assert result.fsoi_predicted_impact < 0.0
    assert result.full_resolved_impact < 0.0
    assert result.reload_max_abs_error == 0.0
    assert result.next_assimilation_max_abs_error == 0.0
    assert result.analysis_relative_stationarity_before < 1.0e-3
    assert result.analysis_relative_stationarity_after < 1.0e-3
    assert result.analysis_truth_mae_after <= result.analysis_truth_mae_before
    assert result.next_window_forecast_after <= result.next_window_forecast_before
    assert (
        result.next_window_analysis_truth_mae_after
        <= result.next_window_analysis_truth_mae_before
    )


def test_existing_checkpoint_is_resumed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "prior.pt"
    first = run_mean_only_learning(checkpoint_path=checkpoint)
    second = run_mean_only_learning(checkpoint_path=checkpoint)
    assert second.initial_offset == first.updated_offset
    assert second.updated_offset != second.initial_offset
    assert second.objective_after < second.objective_before


def test_failed_candidate_does_not_replace_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "prior.pt"
    run_mean_only_learning(checkpoint_path=checkpoint)
    before = checkpoint.read_bytes()
    with pytest.raises((RuntimeError, ValueError)):
        run_mean_only_learning(checkpoint_path=checkpoint, learning_rate=1000.0)
    assert checkpoint.read_bytes() == before


def test_only_mean_offset_is_trainable() -> None:
    model = MeanOnlyPrior()
    assert tuple(name for name, _ in model.named_parameters()) == ("offset",)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1
