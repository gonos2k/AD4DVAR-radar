"""Directed objective decisions keep the existing one-sided budget."""
from mpmath import iv
from typing import Any, cast

from examples.weather_scenarios.fv_slice_interval_acceptance import decide_enclosure


def bounds(lower, upper):
    return [list(bound) for bound in cast(Any, iv).mpf([lower, upper])._mpi_]


def test_large_true_decrease_is_allowed_without_an_absolute_decrease_cap():
    assert decide_enclosure(bounds(-10, -9), 1e-16) == (True, "interval_objective_passed")


def test_increase_and_unresolved_interval_remain_distinct_refusals():
    assert decide_enclosure(bounds(2e-15, 3e-15), 1e-15) == (False, "interval_objective_increase")
    assert decide_enclosure(bounds(-1e-15, 2e-15), 1e-15) == (False, "interval_objective_precision_uncertain")
    assert decide_enclosure(bounds(0, 5e-16), 1e-15) == (True, "interval_objective_passed")
