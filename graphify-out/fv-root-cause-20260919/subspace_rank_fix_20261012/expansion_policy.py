"""Choose one bounded expansion from freshly qualified current-point KKT facts."""
from __future__ import annotations

import math


def expansion_radius(base_radius: float, entry_radius: float | None) -> float | None:
    """Pick a radius above resolved GN entry, with an upper cap of 2*base_radius.

    Inputs must come from fresh current-point D/V/G. Return None if the model
    does not support GN entry inside the cap. J/R gates and compute limits stay
    unchanged; this choice changes search length only.
    """
    if entry_radius is None or not math.isfinite(entry_radius) or entry_radius <= 0:
        return None
    cap = 2.0 * base_radius
    if entry_radius >= cap:
        return None
    return min(cap, max(1.25 * base_radius, 1.1 * entry_radius))
