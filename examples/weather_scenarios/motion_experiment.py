"""Opt-in subpixel motion experiment.

This module is deliberately outside :mod:`advar.nowcast`.  The local DFT
candidate has not been connected to the P0/P1 branch or sensitivity contracts;
callers must compare its diagnostics with the production estimate before using
it for research.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from advar.nowcast import NowcastConfig, _phase_correlation_shift_and_psr


def local_dft_motion_experiment(
    previous_dbz: Tensor,
    current_dbz: Tensor,
    config: NowcastConfig,
) -> dict[str, Any]:
    """Compare production phase motion with a bounded local-DFT candidate.

    The candidate only changes subpixel interpolation.  It keeps the
    production thresholding, zero-padded FFT, integer peak, and sign
    convention.  A Gaussian low-frequency taper reduces high-frequency
    phase noise for small peaks; the taper is intentionally reported in the
    diagnostics rather than hidden as a production configuration.
    """
    baseline, psr, interior = _phase_correlation_shift_and_psr(
        previous_dbz,
        current_dbz,
        config,
    )
    previous = (previous_dbz - config.echo_threshold_dbz).clamp_min(0.0)
    current = (current_dbz - config.echo_threshold_dbz).clamp_min(0.0)
    previous = previous - previous.mean()
    current = current - current.mean()
    height, width = previous.shape
    padded_shape = (2 * height, 2 * width)
    cross_power = torch.fft.fft2(current, s=padded_shape) * torch.conj(
        torch.fft.fft2(previous, s=padded_shape)
    )
    cross_power = cross_power / cross_power.abs().clamp_min(config.epsilon)
    correlation = torch.fft.ifft2(cross_power).real
    peak_index = int(torch.argmax(correlation).item())
    peak_y, peak_x = divmod(peak_index, padded_shape[1])
    base_y = peak_y - padded_shape[0] if peak_y > height else peak_y
    base_x = peak_x - padded_shape[1] if peak_x > width else peak_x
    base = correlation.new_tensor((float(base_y), float(base_x)))

    frequency_y = torch.fft.fftfreq(
        padded_shape[0], d=1.0, dtype=correlation.dtype,
    )
    frequency_x = torch.fft.fftfreq(
        padded_shape[1], d=1.0, dtype=correlation.dtype,
    )
    angular_y = (2.0 * math.pi * frequency_y)[:, None]
    angular_x = (2.0 * math.pi * frequency_x)[None, :]
    spectral_scale = 5.0 / max(padded_shape)
    low_frequency = torch.exp(
        -0.5
        * (frequency_y[:, None].square() + frequency_x[None, :].square())
        / spectral_scale**2
    )
    peak_radius = math.hypot(base_y, base_x)
    low_frequency_strength = max(0.0, 1.0 - peak_radius / 2.0)
    weight = (1.0 - low_frequency_strength) + (
        low_frequency_strength * low_frequency
    )
    weighted_cross_power = cross_power * weight

    def local_objective(candidate: Tensor) -> Tensor:
        phase = torch.exp(
            1j * (angular_y * candidate[0] + angular_x * candidate[1])
        )
        return (weighted_cross_power * phase).real.sum() / (
            padded_shape[0] * padded_shape[1]
        )

    refined = base.clone()
    for axis in range(2):
        lower = refined.clone()
        upper = refined.clone()
        lower[axis] -= 0.5
        upper[axis] += 0.5
        center_value = local_objective(refined)
        lower_value = local_objective(lower)
        upper_value = local_objective(upper)
        curvature = lower_value - 2.0 * center_value + upper_value
        if float(curvature.detach()) >= -config.epsilon:
            continue
        offset = (
            0.25 * (lower_value - upper_value) / curvature
        ).clamp(-0.5, 0.5)
        candidate = refined.clone()
        candidate[axis] += offset
        candidate_value = local_objective(candidate)
        if float(candidate_value.detach()) >= float(
            center_value.detach()
        ) - config.contract_absolute_tolerance:
            refined = candidate

    objective_before = local_objective(base)
    objective_after = local_objective(refined)
    accepted = bool(
        float(objective_after.detach())
        >= float(objective_before.detach()) - config.contract_absolute_tolerance
    )
    if not accepted:
        refined = base
        objective_after = objective_before
    return {
        "baseline_shift_yx": baseline,
        "candidate_shift_yx": refined,
        "psr": psr,
        "search_interior": interior,
        "integer_peak_yx": (base_y, base_x),
        "spectral_scale_cycles_per_pixel": spectral_scale,
        "objective_before": objective_before,
        "objective_after": objective_after,
        "candidate_alignment_improved": accepted,
    }
