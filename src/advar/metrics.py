"""Verification metrics kept separate from state estimation."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def mae(forecast_dbz: Tensor, truth_dbz: Tensor) -> Tensor:
    """Mean absolute dBZ error over finite pairs."""

    error, scale = _scaled_errors(forecast_dbz, truth_dbz)
    return torch.mean(torch.abs(error)) * scale


def rmse(forecast_dbz: Tensor, truth_dbz: Tensor) -> Tensor:
    """Root mean squared dBZ error over finite pairs."""

    error, scale = _scaled_errors(forecast_dbz, truth_dbz)
    return (torch.linalg.vector_norm(error) / math.sqrt(error.numel())) * scale


def critical_success_index(
    forecast_dbz: Tensor,
    truth_dbz: Tensor,
    threshold_dbz: float = 35.0,
) -> Tensor:
    """Threshold exceedance CSI: hits / (hits + misses + false alarms)."""

    if not math.isfinite(threshold_dbz):
        raise ValueError("threshold_dbz must be finite")
    forecast, truth = _finite_pairs(forecast_dbz, truth_dbz)
    forecast_event = forecast >= threshold_dbz
    truth_event = truth >= threshold_dbz
    hits = torch.sum(forecast_event & truth_event)
    misses = torch.sum(~forecast_event & truth_event)
    false_alarms = torch.sum(forecast_event & ~truth_event)
    denominator = hits + misses + false_alarms
    if int(denominator) == 0:
        return forecast.new_tensor(1.0)
    return hits.to(forecast.dtype) / denominator


def _scaled_errors(forecast: Tensor, truth: Tensor) -> tuple[Tensor, Tensor]:
    forecast, truth = _finite_pairs(forecast, truth)
    dtype = torch.promote_types(forecast.dtype, truth.dtype)
    forecast, truth = forecast.to(dtype), truth.to(dtype)
    # Only opposite signs can overflow subtraction. Halve those large pairs
    # first; ordinary differences retain their cancellation accuracy.
    halve = (torch.signbit(forecast) != torch.signbit(truth)) & (
        torch.maximum(forecast.abs(), truth.abs()) > torch.finfo(dtype).max / 2.0
    )
    divisor = torch.where(halve, torch.full_like(forecast, 2.0), torch.ones_like(forecast))
    error = forecast / divisor - truth / divisor
    scale = error.abs().amax().clamp_min(torch.finfo(dtype).tiny)
    return (error / scale) * divisor, scale


def _finite_pairs(forecast: Tensor, truth: Tensor) -> tuple[Tensor, Tensor]:
    if forecast.shape != truth.shape:
        raise ValueError("forecast and truth must have the same shape")
    if not forecast.is_floating_point() or not truth.is_floating_point():
        raise TypeError("forecast and truth must be floating-point tensors")
    valid = torch.isfinite(forecast) & torch.isfinite(truth)
    if not bool(torch.any(valid)):
        raise ValueError("forecast and truth have no finite pairs")
    return forecast[valid], truth[valid]
