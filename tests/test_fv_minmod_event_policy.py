import pytest
import torch

from examples.weather_scenarios.fv_minmod_events import (
    MinmodEvent,
    classify_change,
    classify_event,
)


DTYPE = torch.float64
EPS = torch.finfo(DTYPE).eps


@pytest.mark.parametrize(
    ("left", "right", "other_sign", "positive_gap_selector"),
    [
        (0.5, 0.5 + 64 * EPS, 1, 1),
        (-0.5, -0.5 + 64 * EPS, -1, 2),
    ],
)
def test_gap_event_selector_orientation_uses_common_operand_sign(
    left, right, other_sign, positive_gap_selector
):
    event = classify_event(48, "x", 1, 1, left, right, 1.0)

    assert event == MinmodEvent(48, "x", 1, 1, "gap", other_sign)
    assert event.selector(1) == positive_gap_selector
    assert event.selector(-1) == 3 - positive_gap_selector
    value = event.value(torch.tensor(left, dtype=DTYPE), torch.tensor(right, dtype=DTYPE))
    assert float(value) == right - left


@pytest.mark.parametrize(
    ("kind", "other_sign", "active_selector"),
    [
        ("left_zero", 1, 1),
        ("left_zero", -1, 1),
        ("right_zero", 1, 2),
        ("right_zero", -1, 2),
    ],
)
def test_operand_zero_event_active_side_tracks_other_operand_sign(
    kind, other_sign, active_selector
):
    left, right = (0.0, 0.5 * other_sign) if kind == "left_zero" else (0.5 * other_sign, 0.0)
    event = classify_event(321, "y", 2, 3, left, right, 1.0)

    assert event == MinmodEvent(321, "y", 2, 3, kind, other_sign)
    assert event.selector(other_sign) == active_selector
    assert event.selector(-other_sign) == 0
    left = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    right = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    value = event.value(left, right)
    left_gradient, right_gradient = torch.autograd.grad(value, (left, right), allow_unused=True)
    expected = (1.0, None) if kind == "left_zero" else (None, 1.0)
    actual = (
        None if left_gradient is None else float(left_gradient),
        None if right_gradient is None else float(right_gradient),
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("identity", "old", "new", "before", "after", "kind", "other_sign"),
    [
        ((123, "y", 2, 3), 1, 2, (0.2, 0.6), (0.7, 0.3), "gap", 1),
        ((124, "x", 1, 2), 2, 1, (-0.7, -0.2), (-0.1, -0.8), "gap", -1),
        ((47, "x", 1, 1), 0, 1, (-0.2, 0.5), (0.1, 0.5), "left_zero", 1),
        ((359, "y", 2, 2), 0, 2, (0.6, 0.2), (0.6, -0.1), "right_zero", 1),
        ((8, "x", 2, 1), 1, 0, (-0.1, -0.5), (0.1, -0.5), "left_zero", -1),
        ((250, "y", 1, 3), 2, 0, (0.5, 0.1), (0.5, -0.1), "right_zero", 1),
    ],
)
def test_selector_change_classifier_handles_arbitrary_event_identities(
    identity, old, new, before, after, kind, other_sign
):
    event = classify_change(identity, old, new, before, after)

    assert event == MinmodEvent(identity[0],identity[1],identity[2],identity[3],kind,other_sign)
    assert event.key == identity


def test_selector_change_classifier_refuses_coincident_or_unbracketed_events():
    with pytest.raises(ValueError, match="two event surfaces coincide"):
        classify_event(48, "x", 1, 1, 0.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="operand sign as well"):
        classify_change((124, "y", 2, 3), 1, 2, (0.2, 0.7), (-0.2, -0.7))

    with pytest.raises(ValueError, match="both operand signs changed"):
        classify_change((123, "y", 2, 3), 1, 0, (0.2, 0.7), (-0.1, -0.6))

    with pytest.raises(ValueError, match="unresolved simultaneous"):
        classify_change((48, "x", 1, 1), 0, 3, (-0.1, 0.2), (0.1, 0.2))
