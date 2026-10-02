"""Precision-aware objective gate for the declared fixed two-hole slice only."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from mpmath import iv, mp
import torch

from examples.weather_scenarios import fv_slice_precision_reference as reference


def decide_enclosure(binary_bounds, allowance: float) -> tuple[bool, str]:
    lower, upper = [mp.make_mpf(tuple(bound)) for bound in binary_bounds]
    limit = mp.mpf(allowance)
    if upper <= limit:
        return True, "interval_objective_passed"
    if lower > limit:
        return False, "interval_objective_increase"
    return False, "interval_objective_precision_uncertain"


def make_gate(fixture, expected_choices, records):
    """Keep the original roundoff budget; leave gradient Armijo to the refiner.

    Branch metadata must contain the physical controls checked by the caller.
    This is serial process-local research code, not an in-process service API.
    """
    fixed_fixture, choices = deepcopy(fixture), deepcopy(expected_choices)

    def gate(trial):
        controls = [branch["physical_control"] for branch in (trial.current_branch, trial.candidate_branch)]
        try:
            values = [reference.evaluate(fixed_fixture, control, dps=80, interval=True) for control in controls]
        except ValueError as error:
            if str(error) != "interval arithmetic could not certify a numerical branch":
                raise
            records.append({"decision": "interval_objective_branch_uncertain"})
            return False, "interval_objective_branch_uncertain"
        if any(row["branch_choices"] != choices for row in values):
            records.append({"decision": "interval_objective_branch_mismatch"})
            return False, "interval_objective_branch_mismatch"
        previous = iv.dps
        try:
            iv.dps = 80
            enclosures: list[Any] = [iv.make_mpf(tuple(tuple(bound) for bound in row["objective_interval_binary"])) for row in values]
            bounds = [list(bound) for bound in (enclosures[1] - enclosures[0])._mpi_]
        finally:
            iv.dps = previous
        info = torch.finfo(torch.float64)
        allowance = 128 * info.eps * max(abs(trial.current_objective), abs(trial.candidate_objective), info.tiny)
        allowed, reason = decide_enclosure(bounds, allowance)
        records.append({"iteration": trial.iteration, "backtrack": trial.backtrack,
                        "allowance": allowance, "difference_interval_binary": bounds,
                        "decision": reason, "original_armijo_ratio": trial.armijo_ratio,
                        "native_delta": trial.candidate_objective - trial.current_objective})
        # None adds no veto; it never overrides the refiner's Armijo predicate.
        return None if allowed else (False, reason)

    return gate
