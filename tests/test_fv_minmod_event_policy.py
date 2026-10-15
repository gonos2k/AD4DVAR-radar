import pytest
import torch

from examples.weather_scenarios.fv_minmod_events import (
    MinmodEvent,
    classify_change,
    classify_event,
    outgoing_decision,
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


def _four_sector_rows(vector: torch.Tensor) -> torch.Tensor:
    return vector.reshape(1, -1).expand(4, -1).clone()


def test_outgoing_decision_proposes_release_for_negative_minus_side_cost():
    normal = torch.tensor([1.0, 0.0], dtype=DTYPE)
    minus = _four_sector_rows(torch.tensor([2.0, 0.0], dtype=DTYPE))
    plus = _four_sector_rows(torch.tensor([1.0, 1.0], dtype=DTYPE))

    decision = outgoing_decision(minus, plus, normal)

    assert decision["decision"] == "release"
    assert decision["side"] == -1
    assert all(value < 0 for value in decision["outgoing_minus"])
    assert all(value > 0 for value in decision["outgoing_plus"])


def test_outgoing_decision_proposes_release_for_negative_plus_side_cost():
    normal = torch.tensor([1.0, 0.0], dtype=DTYPE)
    minus = _four_sector_rows(torch.tensor([-1.0, 0.0], dtype=DTYPE))
    plus = _four_sector_rows(torch.tensor([-2.0, 0.0], dtype=DTYPE))

    decision = outgoing_decision(minus, plus, normal)

    assert decision["decision"] == "release"
    assert decision["side"] == 1
    assert all(value > 0 for value in decision["outgoing_minus"])
    assert all(value < 0 for value in decision["outgoing_plus"])


def test_outgoing_decision_maintain_is_only_normal_necessary_condition():
    normal = torch.tensor([1.0, 0.0], dtype=DTYPE)
    tangent_component = torch.tensor([0.0, 4.0], dtype=DTYPE)
    minus = _four_sector_rows(torch.tensor([-1.0, 0.0], dtype=DTYPE) + tangent_component)
    plus = _four_sector_rows(torch.tensor([1.0, 0.0], dtype=DTYPE) + tangent_component)

    decision = outgoing_decision(minus, plus, normal)

    assert decision["decision"] == "maintain"
    assert all(value > 0 for value in decision["outgoing_minus"])
    assert all(value > 0 for value in decision["outgoing_plus"])
    assert bool((minus[:, 1] != 0).any() and (plus[:, 1] != 0).any())
    assert "tangent stationarity" in decision["scope"]


def test_outgoing_decision_keeps_roundoff_sign_unresolved():
    normal = torch.tensor([1.0, 0.0], dtype=DTYPE)
    minus = _four_sector_rows(torch.tensor([-1.0, 0.0], dtype=DTYPE))
    plus = _four_sector_rows(torch.tensor([0.0, 1.0], dtype=DTYPE))

    decision = outgoing_decision(minus, plus, normal)

    assert decision["decision"] == "unresolved"
    assert decision["reason"] == "normal sign unresolved"


@pytest.mark.parametrize(
    "normal",
    [
        torch.tensor([0.0, 0.0], dtype=DTYPE),
        torch.tensor([float("nan"), 0.0], dtype=DTYPE),
        torch.tensor([float("inf"), 0.0], dtype=DTYPE),
    ],
)
def test_outgoing_decision_requires_finite_nonzero_normal(normal):
    minus = _four_sector_rows(torch.tensor([-1.0, 0.0], dtype=DTYPE))
    plus = _four_sector_rows(torch.tensor([1.0, 0.0], dtype=DTYPE))

    with pytest.raises(ValueError, match="normal is not finite and nonzero"):
        outgoing_decision(minus, plus, normal)


def test_outgoing_decision_refuses_flux_sector_tangent_disagreement():
    normal = torch.tensor([1.0, 0.0], dtype=DTYPE)
    minus = _four_sector_rows(torch.tensor([-1.0, 0.0], dtype=DTYPE))
    plus = _four_sector_rows(torch.tensor([1.0, 0.0], dtype=DTYPE))
    plus[-1, 1] = 1.0e-6

    decision = outgoing_decision(minus, plus, normal)

    assert decision["decision"] == "unresolved"
    assert decision["reason"] == "flux-sector tangent gradients disagree"
