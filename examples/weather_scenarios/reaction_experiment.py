"""Standalone C2 reaction experiment.

This module is an experiment for the weather-scenario review.  It does not
change or select the production reaction law.  The fitted regime is a single
constant non-negative kappa over equally spaced observations.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import Tensor


LOG10_OVER_10 = math.log(10.0) / 10.0


def _check_dbz(name: str, value: Tensor, min_dbz: float) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not math.isfinite(min_dbz):
        raise ValueError("min_dbz must be finite")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} must be finite; missing values are unsupported")
    if bool(torch.any(detached < min_dbz)):
        raise ValueError(f"{name} must be at least min_dbz")


def _reaction_echo(dbz: Tensor, kappa: Tensor, horizon: Tensor, min_dbz: float) -> Tensor:
    """Apply the q-space law through its exactly equivalent dBZ form.

    Since ``log1p(q/A) = log(10) * (z-zmin) / 10``, the C2 law reduces to
    ``z_h = zmin + exp(-kappa*h) * (z_0-zmin)``.  Evaluating this form keeps
    finite large dBZ inputs finite while the tests retain an independent
    q-space oracle for the stated closed form.
    """

    return min_dbz + torch.exp(-kappa * horizon) * (dbz - min_dbz)


def _floor_resolution(dtype: torch.dtype, min_dbz: float) -> float:
    """Return a deterministic dBZ excess that is resolvable in ``dtype``."""

    return 32.0 * torch.finfo(dtype).eps * max(1.0, abs(min_dbz))


def reaction_dbz(
    dbz: Tensor,
    kappa: Tensor | Real,
    horizon_steps: Tensor | Real,
    *,
    min_dbz: float,
) -> Tensor:
    """Return dBZ after ``horizon_steps`` under the C2 decay reaction.

    ``dbz`` may include the exact floor (zero reaction echo), and ``kappa=0``
    is the identity boundary.  Negative kappa (positive growth) and missing
    values are outside this experiment and rejected explicitly.
    """

    _check_dbz("dbz", dbz, min_dbz)
    kappa_tensor = torch.as_tensor(kappa, dtype=dbz.dtype, device=dbz.device)
    horizon_tensor = torch.as_tensor(
        horizon_steps,
        dtype=dbz.dtype,
        device=dbz.device,
    )
    if kappa_tensor.ndim != 0 or horizon_tensor.ndim != 0:
        raise ValueError("kappa and horizon_steps must be scalar")
    if not bool(torch.isfinite(kappa_tensor.detach())):
        raise ValueError("kappa must be finite")
    if not bool(torch.isfinite(horizon_tensor.detach())):
        raise ValueError("horizon_steps must be finite")
    if bool(kappa_tensor.detach() < 0.0):
        raise ValueError("positive growth (negative kappa) is unsupported")
    if bool(horizon_tensor.detach() < 0.0):
        raise ValueError("horizon_steps must be non-negative")
    return _reaction_echo(dbz, kappa_tensor, horizon_tensor, min_dbz)


def estimate_kappa_from_observations(
    observations_dbz: Tensor,
    *,
    min_dbz: float,
    interval_steps: Real = 1.0,
) -> Tensor:
    """Estimate one constant kappa from exactly three past dBZ fields.

    The estimate uses only cells with a resolvable positive dBZ excess in all
    three fields.  A zero-only record has no identifiable decay, missing
    values are unsupported, and positive growth or non-constant decay is
    rejected.
    """

    _check_dbz("observations_dbz", observations_dbz, min_dbz)
    if observations_dbz.ndim < 1 or observations_dbz.shape[0] != 3:
        raise ValueError("observations_dbz must have exactly three time steps")
    if isinstance(interval_steps, bool) or not isinstance(interval_steps, Real):
        raise TypeError("interval_steps must be a real number")
    interval = float(interval_steps)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("interval_steps must be finite and positive")

    reaction = observations_dbz - min_dbz
    resolution = _floor_resolution(observations_dbz.dtype, min_dbz)
    active = torch.all(reaction > resolution, dim=0)
    if not bool(torch.any(active)):
        raise ValueError("zero reaction signal or unresolved excess cannot identify kappa")
    selected = reaction[:, active]
    rates = -torch.log(selected[1:] / selected[:-1]) / interval
    rate_tolerance = 32.0 * torch.finfo(observations_dbz.dtype).eps / interval
    if bool(torch.any(rates < -rate_tolerance)):
        raise ValueError("positive growth is unsupported")
    estimate = torch.mean(rates)
    tolerance = max(
        rate_tolerance,
        1.0e-5 * max(1.0, abs(float(estimate.detach()))),
    )
    if bool(torch.amax(torch.abs(rates - estimate).detach()) > tolerance):
        raise ValueError("observations do not support a constant-kappa regime")
    return estimate


def forecast_from_observations(
    observations_dbz: Tensor,
    *,
    min_dbz: float,
    lead_count: int = 18,
    interval_steps: Real = 1.0,
) -> tuple[Tensor, Tensor]:
    """Fit kappa from three past fields and return kappa plus future fields."""

    if isinstance(lead_count, bool) or not isinstance(lead_count, Integral):
        raise TypeError("lead_count must be an integer")
    if lead_count <= 0:
        raise ValueError("lead_count must be positive")
    kappa = estimate_kappa_from_observations(
        observations_dbz,
        min_dbz=min_dbz,
        interval_steps=interval_steps,
    )
    horizons = [
        (lead + 1) * float(interval_steps)
        for lead in range(int(lead_count))
    ]
    forecasts = torch.stack(
        [
            reaction_dbz(
                observations_dbz[-1],
                kappa,
                horizon,
                min_dbz=min_dbz,
            )
            for horizon in horizons
        ]
    )
    return kappa, forecasts
