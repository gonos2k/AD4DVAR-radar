"""Synthetic branch checks for the exact selected face-event exception."""
import pytest
import torch

from examples.weather_scenarios.fv_active_face_stationarity_probe import (
    ActiveFaceRefusal,
    _inspect_stage,
)


def _strict_stage():
    row = torch.arange(4, dtype=torch.float64)[:, None]
    column = torch.arange(5, dtype=torch.float64)[None, :]
    echo = 1 + 0.3 * row + 0.05 * row.square() + 0.1 * column.square() + 0.02 * row * column
    qx = torch.ones((4, 6), dtype=torch.float64)
    qy = torch.ones((5, 5), dtype=torch.float64)
    qy[2, 0] = 0.0
    return echo, qx, qy


def test_only_named_face_may_be_zero_and_other_limiter_branches_are_strict():
    echo, qx, qy = _strict_stage()
    trace, slope_margin, face_margin = _inspect_stage(echo, qx, qy, 0, 0)
    assert trace["step"] == 0 and trace["stage"] == 0
    assert trace["qy_sign"][2][0] == 0
    assert all(value == 1 for line in trace["qx_sign"] for value in line)
    assert slope_margin > 1e-4
    assert face_margin == 1.0


def test_wrong_zero_face_and_limiter_tie_refuse():
    echo, qx, qy = _strict_stage()
    wrong_event = qy.clone()
    wrong_event[2, 0] = 1e-20
    with pytest.raises(ActiveFaceRefusal, match="not an exact structural zero"):
        _inspect_stage(echo, qx, wrong_event, 0, 0)

    other_zero = qy.clone()
    other_zero[0, 0] = 0.0
    with pytest.raises(ActiveFaceRefusal, match="non-selected face"):
        _inspect_stage(echo, qx, other_zero, 0, 0)

    other_low = qy.clone()
    other_low[0, 0] = 1e-6
    with pytest.raises(ActiveFaceRefusal, match="scaled margin"):
        _inspect_stage(echo, qx, other_low, 0, 0)

    row = torch.arange(4, dtype=torch.float64)[:, None]
    column = torch.arange(5, dtype=torch.float64)[None, :]
    tied_echo = 1 + 0.3 * row + 0.1 * column
    with pytest.raises(ActiveFaceRefusal, match="limiter input is zero or tied"):
        _inspect_stage(tied_echo, qx, qy, 0, 0)


def test_limiter_margin_below_predeclared_threshold_refuses():
    row = torch.arange(4, dtype=torch.float64)[:, None]
    column = torch.arange(5, dtype=torch.float64)[None, :]
    shallow = 1 + 0.3 * row + 0.05 * row.square() + 0.1 * column + 1e-7 * column.square()
    qx = torch.ones((4, 6), dtype=torch.float64)
    qy = torch.ones((5, 5), dtype=torch.float64)
    qy[2, 0] = 0.0
    with pytest.raises(ActiveFaceRefusal, match="limiter input lacks the fixed 1e-4"):
        _inspect_stage(shallow, qx, qy, 0, 0)
