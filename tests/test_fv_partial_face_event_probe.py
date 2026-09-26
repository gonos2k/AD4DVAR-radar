"""Pure event-location tests; no FV model evaluations are performed."""

import math

import pytest
import torch

from examples.weather_scenarios import fv_partial_face_event_probe as probe
from examples.weather_scenarios.fv_partial_face_event_probe import (
    bisect_zero,
    event_side_parameters,
    event_side_schedule,
)


def test_bisect_zero_finds_linear_event_with_bounded_bracket():
    root = math.sqrt(2.0) / 3.0
    result = bisect_zero(lambda t: t - root)

    assert result["iterations"] <= 64
    assert result["left_t"] <= root <= result["right_t"]
    assert result["right_t"] - result["left_t"] <= 2.0**-48
    assert result["left_value"] <= 0.0 <= result["right_value"]
    assert result["stop_reason"] == "bracket_width"


def test_bisect_zero_accepts_exact_representable_root_without_event_derivatives():
    result = bisect_zero(lambda t: t - 0.375)

    assert result["left_t"] == result["right_t"] == 0.375
    assert result["left_value"] == result["right_value"] == 0.0
    assert result["stop_reason"] == "exact_zero"


@pytest.mark.parametrize(
    ("function", "match"),
    [
        (lambda _t: 1.0, "opposite signs"),
        (lambda t: t, "endpoint event"),
        (lambda t: math.nan if t == 0.5 else (t - 0.4), "finite"),
    ],
)
def test_bisect_zero_refuses_unbracketed_or_nonfinite_events(function, match):
    with pytest.raises(ValueError, match=match):
        bisect_zero(function)


def test_event_side_parameters_keep_declared_offsets_separate():
    points = event_side_parameters(0.5)

    assert points == [
        {"side": "left", "offset": 2.0**-8, "t": 0.5 - 2.0**-8},
        {"side": "left", "offset": 2.0**-10, "t": 0.5 - 2.0**-10},
        {"side": "right", "offset": 2.0**-8, "t": 0.5 + 2.0**-8},
        {"side": "right", "offset": 2.0**-10, "t": 0.5 + 2.0**-10},
    ]


def test_event_side_parameters_omits_offsets_outside_open_chord():
    points = event_side_parameters(2.0**-11)

    assert [point["side"] for point in points] == ["right", "right"]
    assert [point["offset"] for point in points] == [2.0**-8, 2.0**-10]
    assert all(0.0 < point["t"] < 1.0 for point in points)


def test_event_side_schedule_records_each_unavailable_offset_without_replacement():
    schedule = event_side_schedule(2.0**-11)

    assert len(schedule) == 4
    assert [point["available"] for point in schedule] == [False, False, True, True]
    assert [point["t"] for point in schedule[:2]] == [None, None]
    assert [point["offset"] for point in schedule] == [
        2.0**-8, 2.0**-10, 2.0**-8, 2.0**-10,
    ]


@pytest.mark.parametrize("event_t", [math.nan, -0.1, 1.0, 1.1])
def test_event_side_parameters_requires_finite_interior_event(event_t):
    with pytest.raises(ValueError, match="interior event"):
        event_side_parameters(event_t)


def test_known_strict_branch_refusal_is_recorded_without_derivatives():
    class RefusedProblem:
        def branch_check(self, _control, _parameters):
            raise ValueError("minmod joint oracle left its strict smooth branch")

    result = probe._sample_side(
        RefusedProblem(), torch.zeros(1), torch.zeros(1), torch.ones(1),
        {"side": "left", "offset": 2.0**-8, "t": 0.5}, "test_chord",
    )

    assert result["status"] == "branch_refused"
    assert result["finite"] is None
    assert "refusal" in result


def test_unexpected_branch_callback_error_fails_closed():
    class BrokenProblem:
        def branch_check(self, _control, _parameters):
            raise ValueError("unexpected callback regression")

    with pytest.raises(RuntimeError, match="unexpected branch callback"):
        probe._sample_side(
            BrokenProblem(), torch.zeros(1), torch.zeros(1), torch.ones(1),
            {"side": "left", "offset": 2.0**-8, "t": 0.5}, "test_chord",
        )
