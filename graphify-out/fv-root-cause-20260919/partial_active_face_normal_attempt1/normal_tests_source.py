"""Scalar interval-jet checks for the two one-sided face extensions."""
from mpmath import mp
import pytest

from examples.weather_scenarios import fv_slice_precision_reference as reference
from examples.weather_scenarios.fv_active_face_normal_reference import (
    IntervalJet,
    _JetMath,
)


def test_interval_jet_unary_chain_encloses_analytic_derivative():
    with reference._arithmetic(70, True) as iv:
        math = _JetMath(iv, 1)
        x = IntervalJet(iv.mpf("0.25"), iv.one, 1)
        value = (math.exp(x) + math.log(x) + math.log1p(x) + math.expm1(x) + math.sqrt(x)
                 + math.tanh(x) + math.atanh(x))
        lower, upper = value.derivative._mpi_
        with mp.workdps(70):
            point = mp.mpf("0.25")
            expected = (2 * mp.exp(point) + 1 / point + 1 / (1 + point)
                        + 1 / (2 * mp.sqrt(point))
                        + 1 - mp.tanh(point) ** 2 + 1 / (1 - point**2))
            assert mp.make_mpf(lower) <= expected <= mp.make_mpf(upper)


@pytest.mark.parametrize(("side", "expected"), ((1, 2), (-1, 3)))
def test_tagged_face_event_selects_correct_one_sided_flux_derivative(side: int, expected: int):
    with reference._arithmetic(60, True) as iv:
        math = _JetMath(iv, side)
        face = IntervalJet(iv.zero, iv.one, side, event=True)
        flux = max(face, math.zero) * 2 + min(face, math.zero) * 3
        assert flux.value == iv.zero
        assert flux.derivative == iv.mpf(expected)
        assert not flux.event


def test_unrelated_zero_and_ambiguous_event_direction_refuse():
    with reference._arithmetic(60, True) as iv:
        math = _JetMath(iv, 1)
        unrelated_tie = IntervalJet(iv.zero, iv.one, 1)
        with pytest.raises(ValueError, match="nonsmooth tie"):
            max(unrelated_tie, math.zero)

        ambiguous_face = IntervalJet(iv.zero, iv.mpf([-1, 1]), 1, event=True)
        with pytest.raises(ValueError, match="event-side derivative sign"):
            max(ambiguous_face, math.zero)


@pytest.mark.parametrize("value", ("2", "[0,1]"))
def test_event_tag_cannot_resolve_nonzero_or_nonpoint_primal_ties(value: str):
    with reference._arithmetic(60, True) as iv:
        math = _JetMath(iv, 1)
        interval = iv.mpf(value)
        tagged = IntervalJet(interval, iv.one, 1, event=True)
        tied = IntervalJet(interval, iv.zero, 1)
        with pytest.raises(ValueError):
            max(tagged, tied)
