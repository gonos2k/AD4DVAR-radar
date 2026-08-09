"""Pure input contracts shared by prospective action execution."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ._digest import json_digest


def action_input_canonicalization_digest(
    *,
    minimum_dbz: float,
    maximum_dbz: float,
    missing_fill_dbz: float,
) -> str:
    """Content-address the exact missing-value and clamp convention."""

    if not all(
        math.isfinite(value)
        for value in (minimum_dbz, maximum_dbz, missing_fill_dbz)
    ) or not minimum_dbz <= missing_fill_dbz <= maximum_dbz:
        raise ValueError("radar action canonicalization bounds are invalid")
    return json_digest(
        {
            "contract": "radar-action-input-canonicalization-v1",
            "minimum_dbz": minimum_dbz,
            "maximum_dbz": maximum_dbz,
            "missing_fill_dbz": missing_fill_dbz,
            "validity": "finite-and-observation-valid",
        }
    )


def canonicalize_action_frames(
    frames_dbz: Tensor,
    valid_mask: Tensor,
    *,
    minimum_dbz: float,
    maximum_dbz: float,
    missing_fill_dbz: float,
) -> Tensor:
    """Map equivalent NaN/Inf/masked observations to one generator input."""

    if valid_mask.shape != frames_dbz.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("radar action canonicalization mask is invalid")
    accepted = torch.isfinite(frames_dbz) & valid_mask
    return torch.where(
        accepted,
        frames_dbz.clamp(minimum_dbz, maximum_dbz),
        frames_dbz.new_full((), missing_fill_dbz),
    )
