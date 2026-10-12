"""Frozen experiment policy for finite residual-model fidelity."""
from __future__ import annotations

import math


def radius_factor(accepted: bool, rho_R: float, eta_vec: float) -> float:
    """Return the next radius multiplier after one finite candidate.

    Shrink rejected or poorly modelled steps. A good scalar decrease alone
    cannot justify expansion when the residual-vector prediction is poor.
    This is search policy, not a change to candidate approval or tolerances.
    """
    if not accepted or not all(math.isfinite(x) for x in (rho_R, eta_vec)):
        return 0.5
    if rho_R < 0.25 or eta_vec > 0.5:
        return 0.5
    return 1.0
