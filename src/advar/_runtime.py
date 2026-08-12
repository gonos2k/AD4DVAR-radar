"""Canonical numerical-runtime identities used by delayed audit artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from importlib import metadata
import platform
from pathlib import Path
import re
import sys
import numpy as np
import torch

from ._digest import json_digest


NUMERICAL_RUNTIME_IDENTITY_VERSION = "advar-numerical-runtime-v2"
NUMERICAL_RUNTIME_COMPATIBILITY_VERSION = (
    "advar-numerical-runtime-compatibility-v2"
)
NUMERICAL_RUNTIME_EXACT_VERSION = "advar-numerical-runtime-exact-v2"
_DISTRIBUTION_NAME = "advar-radar-nowcast"


def _major_minor(value: str | None) -> str | None:
    if value is None:
        return None
    parts = value.split(".")
    return ".".join(parts[:2])


@lru_cache(maxsize=1)
def _torch_build_configuration() -> str:
    show = getattr(torch.__config__, "show", None)
    return "" if not callable(show) else str(show())


@lru_cache(maxsize=1)
def _numpy_build_configuration() -> object:
    config = getattr(np.__config__, "CONFIG", None)
    if isinstance(config, dict):
        return config
    return {"module": type(np.__config__).__name__}


@lru_cache(maxsize=1)
def _installed_distribution_manifest_digest() -> str:
    """Identify the installed ADVAR distribution and checked-out source bytes.

    Original wheel archives are not retained by Python installers.  RECORD and
    package-source bytes are therefore both addressed so an editable install
    cannot collapse onto the identity of a built wheel with different code.
    """

    distribution_payload: dict[str, object]
    try:
        distribution = metadata.distribution(_DISTRIBUTION_NAME)
        distribution_name = distribution.metadata["Name"]
        distribution_payload = {
            "name": distribution_name or _DISTRIBUTION_NAME,
            "version": distribution.version,
            "metadata_sha256": hashlib.sha256(
                (distribution.read_text("METADATA") or "").encode("utf-8")
            ).hexdigest(),
            "record_sha256": hashlib.sha256(
                (distribution.read_text("RECORD") or "").encode("utf-8")
            ).hexdigest(),
        }
    except metadata.PackageNotFoundError:
        distribution_payload = {"name": _DISTRIBUTION_NAME, "installed": False}

    package_root = Path(__file__).resolve().parent
    sources = [
        {
            "relative_path": path.relative_to(package_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(package_root.rglob("*.py"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return json_digest(
        {
            "contract": "advar-installed-distribution-manifest-v1",
            "distribution": distribution_payload,
            "python_sources": sources,
        }
    )


def _backend_setting(path: str) -> object:
    value: object = torch.backends
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value() if callable(value) else value


def _execution_device_identity(
    execution_device: str | torch.device | None,
) -> dict[str, object]:
    if execution_device is None:
        return {"type": None, "index": None, "model": None}
    device = torch.device(execution_device)
    if device.type == "cuda" and torch.cuda.is_available():
        index = (
            device.index
            if isinstance(device.index, int)
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        driver_version_fn = getattr(torch._C, "_cuda_getDriverVersion", None)
        driver_version = (
            str(driver_version_fn()) if callable(driver_version_fn) else None
        )
        return {
            "type": "cuda",
            "index": index,
            "model": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "driver_version": driver_version,
        }
    if device.type == "mps":
        return {
            "type": "mps",
            "index": device.index if isinstance(device.index, int) else 0,
            "model": platform.processor() or platform.machine(),
            "mac_version": list(platform.mac_ver()),
        }
    return {
        "type": device.type,
        "index": device.index,
        "model": platform.processor() or platform.machine(),
    }


@dataclass(frozen=True)
class NumericalRuntimeManifest:
    """Compatibility and exact identities for one numerical process."""

    compatibility: dict[str, object]
    exact: dict[str, object]
    contract: str = NUMERICAL_RUNTIME_IDENTITY_VERSION
    compatibility_digest: str = field(init=False)
    exact_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != NUMERICAL_RUNTIME_IDENTITY_VERSION
            or self.compatibility.get("contract")
            != NUMERICAL_RUNTIME_COMPATIBILITY_VERSION
            or self.exact.get("contract") != NUMERICAL_RUNTIME_EXACT_VERSION
        ):
            raise ValueError("numerical runtime manifest is invalid")
        object.__setattr__(
            self,
            "compatibility_digest",
            json_digest(self.compatibility),
        )
        object.__setattr__(self, "exact_digest", json_digest(self.exact))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "compatibility": self.compatibility,
            "exact": self.exact,
            "compatibility_digest": self.compatibility_digest,
            "exact_digest": self.exact_digest,
        }


def numerical_runtime_manifest(
    execution_device: str | torch.device | None = None,
) -> NumericalRuntimeManifest:
    """Return compatibility-class and exact-replay runtime identities."""

    mps_backend = getattr(torch.backends, "mps", None)
    execution = _execution_device_identity(execution_device)
    matmul_precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    compatibility: dict[str, object] = {
        "contract": NUMERICAL_RUNTIME_COMPATIBILITY_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "major_minor": _major_minor(platform.python_version()),
        },
        "numpy_major_minor": _major_minor(np.__version__),
        "torch_major_minor": _major_minor(torch.__version__),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": deterministic,
        "deterministic_warn_only": warn_only,
        "execution_device_type": execution["type"],
        "float32_matmul_precision": matmul_precision,
        "cuda_build_major_minor": _major_minor(torch.version.cuda),
    }
    exact: dict[str, object] = {
        "contract": NUMERICAL_RUNTIME_EXACT_VERSION,
        "compatibility_digest": json_digest(compatibility),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "numpy": {
            "version": np.__version__,
            "build_configuration": _numpy_build_configuration(),
        },
        "torch": {
            "version": torch.__version__,
            "build_configuration": _torch_build_configuration(),
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": deterministic,
            "deterministic_warn_only": warn_only,
            "float32_matmul_precision": matmul_precision,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cuda_build_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                mps_backend is not None and mps_backend.is_available()
            ),
            "cudnn_version": _backend_setting("cudnn.version"),
            "cudnn_deterministic": _backend_setting("cudnn.deterministic"),
            "cudnn_benchmark": _backend_setting("cudnn.benchmark"),
            "cuda_matmul_allow_tf32": _backend_setting(
                "cuda.matmul.allow_tf32"
            ),
            "cudnn_allow_tf32": _backend_setting("cudnn.allow_tf32"),
        },
        "execution_device": execution,
        "byteorder": sys.byteorder,
        "installed_distribution_digest": (
            _installed_distribution_manifest_digest()
        ),
    }
    return NumericalRuntimeManifest(compatibility=compatibility, exact=exact)


@dataclass(frozen=True)
class MPSBackendCertificationEvidence:
    """Self-hosted CPU/MPS numerical comparison required for deployment."""

    cpu_runtime_exact_digest: str
    mps_runtime_exact_digest: str
    algorithm_source_manifest_digest: str
    certification_runner_digest: str
    pcg_extreme_scale_relative_error: float
    pcg_relative_error_tolerance: float
    frozen_stationarity_max_abs_difference: float
    robust_stationarity_max_abs_difference: float
    analysis_max_abs_difference_dbz: float
    score_max_abs_difference: float
    numerical_tolerance: float
    decision_statistic_max_abs_difference: float
    minimum_decision_margin: float
    decision_invariant: bool
    nonfinite_fallback_verified: bool
    deterministic_policy_verified: bool
    contract: str = "advar-mps-backend-certification-v1"
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for value in (
            self.cpu_runtime_exact_digest,
            self.mps_runtime_exact_digest,
            self.algorithm_source_manifest_digest,
            self.certification_runner_digest,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("MPS certification digest is invalid")
        measurements = (
            self.pcg_extreme_scale_relative_error,
            self.pcg_relative_error_tolerance,
            self.frozen_stationarity_max_abs_difference,
            self.robust_stationarity_max_abs_difference,
            self.analysis_max_abs_difference_dbz,
            self.score_max_abs_difference,
            self.numerical_tolerance,
            self.decision_statistic_max_abs_difference,
            self.minimum_decision_margin,
        )
        if (
            self.contract != "advar-mps-backend-certification-v1"
            or self.cpu_runtime_exact_digest == self.mps_runtime_exact_digest
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0.0
                for value in measurements
            )
            or self.pcg_relative_error_tolerance <= 0.0
            or self.numerical_tolerance <= 0.0
        ):
            raise ValueError("MPS backend certification is invalid")
        object.__setattr__(self, "evidence_digest", json_digest(self.payload))

    @property
    def eligible(self) -> bool:
        return bool(
            self.pcg_extreme_scale_relative_error
            <= self.pcg_relative_error_tolerance
            and self.frozen_stationarity_max_abs_difference
            <= self.numerical_tolerance
            and self.robust_stationarity_max_abs_difference
            <= self.numerical_tolerance
            and self.analysis_max_abs_difference_dbz <= self.numerical_tolerance
            and self.score_max_abs_difference <= self.numerical_tolerance
            and self.minimum_decision_margin
            > self.decision_statistic_max_abs_difference
            + self.numerical_tolerance
            and self.decision_invariant
            and self.nonfinite_fallback_verified
            and self.deterministic_policy_verified
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "evidence_digest"
        }


def validate_mps_backend_certification(
    evidence: MPSBackendCertificationEvidence,
    *,
    execution_device: str | torch.device,
) -> None:
    """Fail closed unless evidence exactly identifies the active MPS runtime."""

    device = torch.device(execution_device)
    if (
        device.type != "mps"
        or evidence.evidence_digest != json_digest(evidence.payload)
        or evidence.mps_runtime_exact_digest
        != numerical_runtime_identity_digest(device)
        or not evidence.eligible
    ):
        raise ValueError("MPS backend is not deployment-certified")


def numerical_runtime_identity(
    execution_device: str | torch.device | None = None,
) -> dict[str, object]:
    """Backward-compatible exact identity payload.

    New contracts should persist both digests from
    :func:`numerical_runtime_manifest`; existing artifacts continue to use the
    exact digest returned by :func:`numerical_runtime_identity_digest`.
    """

    manifest = numerical_runtime_manifest(execution_device)
    return dict(manifest.exact)


def numerical_runtime_compatibility_digest(
    execution_device: str | torch.device | None = None,
) -> str:
    return numerical_runtime_manifest(execution_device).compatibility_digest


def numerical_runtime_identity_digest(
    execution_device: str | torch.device | None = None,
) -> str:
    """Exact-replay digest for :func:`numerical_runtime_identity`."""

    return numerical_runtime_manifest(execution_device).exact_digest
