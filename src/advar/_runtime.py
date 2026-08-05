"""Canonical numerical-runtime identity used by delayed audit artifacts."""

from __future__ import annotations

import platform
import sys
from typing import Any

import numpy as np
import torch

from ._digest import json_digest


NUMERICAL_RUNTIME_IDENTITY_VERSION = "advar-numerical-runtime-v1"


def numerical_runtime_identity(
    execution_device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Return stable facts that can change floating-point reproduction.

    The identity intentionally records capabilities and configured numerical
    policy, not transient device memory or process identifiers.
    """

    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "version": NUMERICAL_RUNTIME_IDENTITY_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "numpy_version": np.__version__,
        "torch": {
            "version": torch.__version__,
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                mps_backend is not None and mps_backend.is_available()
            ),
            "execution_device": (
                None
                if execution_device is None
                else str(torch.device(execution_device))
            ),
        },
        "byteorder": sys.byteorder,
    }


def numerical_runtime_identity_digest(
    execution_device: str | torch.device | None = None,
) -> str:
    """Content digest for :func:`numerical_runtime_identity`."""

    return json_digest(numerical_runtime_identity(execution_device))
