"""Canonical learned-radar feature construction shared by all products."""

from __future__ import annotations

import torch
from torch import Tensor


LEARNED_RADAR_INPUT_FEATURE_CONTRACT = "learned-radar-input-features-v1"
LEARNED_RADAR_INPUT_CHANNELS = (
    "canonical_dbz",
    "qc_valid_mask",
    "quality_weight",
    "normalized_observation_std",
    "source_available_mask",
)
LEARNED_RADAR_OBSERVATION_STD_SCALE_DBZ = 20.0


def learned_radar_input_features(
    frames_dbz: Tensor,
    qc_valid_mask: Tensor,
    quality_weight: Tensor,
    observation_std_dbz: Tensor,
    source_available_mask: Tensor,
) -> Tensor:
    """Build the exact five-channel input used by current learned products."""

    if (
        frames_dbz.ndim != 3
        or not frames_dbz.is_floating_point()
        or qc_valid_mask.shape != frames_dbz.shape
        or qc_valid_mask.dtype is not torch.bool
        or quality_weight.shape != frames_dbz.shape
        or not quality_weight.is_floating_point()
        or observation_std_dbz.shape != frames_dbz.shape
        or not observation_std_dbz.is_floating_point()
        or source_available_mask.shape != frames_dbz.shape
        or source_available_mask.dtype is not torch.bool
        or any(
            item.device != frames_dbz.device
            for item in (
                qc_valid_mask,
                quality_weight,
                observation_std_dbz,
                source_available_mask,
            )
        )
        or not bool(torch.all(torch.isfinite(frames_dbz)))
        or not bool(torch.all(torch.isfinite(quality_weight)))
        or bool(torch.any((quality_weight < 0.0) | (quality_weight > 1.0)))
        or not bool(torch.all(torch.isfinite(observation_std_dbz)))
        or bool(torch.any(observation_std_dbz <= 0.0))
    ):
        raise ValueError("learned radar input components are invalid")
    effective_valid = qc_valid_mask & source_available_mask
    if (
        bool(torch.any(~effective_valid & (quality_weight != 0.0)))
        or bool(torch.any(~effective_valid & (frames_dbz != -10.0)))
    ):
        raise ValueError("learned radar invalid cells are not canonical")
    dtype = frames_dbz.dtype
    return torch.stack(
        (
            frames_dbz,
            effective_valid.to(dtype=dtype),
            quality_weight.to(dtype=dtype),
            observation_std_dbz.to(dtype=dtype)
            / LEARNED_RADAR_OBSERVATION_STD_SCALE_DBZ,
            source_available_mask.to(dtype=dtype),
        )
    ).contiguous()
