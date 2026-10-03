"""Pure release-coordinate and pre-optimizer gate tests; no FV evaluation."""
from fractions import Fraction
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_qy_release_probe as probe


def test_insert_y_eta_preserves_other_24_controls():
    tangent = torch.arange(24, dtype=torch.float64)
    released = probe.insert_y_eta(tangent)
    assert released.shape == (25,)
    assert torch.equal(released[:20], tangent[:20])
    assert released[20] == 1e-4
    assert torch.equal(released[21:], tangent[20:])


def test_qy_right_sector_is_read_from_binary_interval_not_status_flag():
    report: dict[str, Any] = {"numerical_status": "oriented_normal_intervals", "normal_intervals": [
        {"normal_axis": 1, "side": 1, "binary": {
            "lower": [1, 1, -1, 1], "upper": [1, 1, -2, 1]},
          "exact_fraction_bounds": ["-1/2", "-1/4"]}
    ]}
    assert probe.qy_right_is_negative(report) == (Fraction(-1, 2), Fraction(-1, 4))
    report["normal_intervals"][0]["binary"]["upper"] = [0, 1, -2, 1]
    with pytest.raises(probe.ReleaseRefusal, match="not strictly negative"):
        probe.qy_right_is_negative(report)


def test_initial_bad_seed_refuses_before_optimizer_callback():
    called = []
    with pytest.raises(probe.ReleaseRefusal, match="below roundoff"):
        probe.optimize_if_decreased(1.0, 1.0, lambda: called.append("optimizer"))
    assert called == []


def test_initial_objective_gate_uses_native_roundoff_scale():
    old, new = 1.0, 1.0 - 1e-12
    comparison, result = probe.optimize_if_decreased(old, new, lambda: "refiner")
    expected = 128 * torch.finfo(torch.float64).eps * max(abs(old), abs(new), torch.finfo(torch.float64).tiny)
    assert comparison["passed"] and comparison["roundoff_budget"] == pytest.approx(expected)
    assert result == "refiner"
