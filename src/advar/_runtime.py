"""Canonical numerical-runtime identities used by delayed audit artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import re
import sys
import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

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


_MPS_CERTIFICATION_POLICY_VERSION = "advar-mps-certification-policy-v1"
_MPS_CERTIFICATION_EVIDENCE_VERSION = "advar-mps-certification-evidence-v2"


def _require_digest(name: str, value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} digest is invalid")


def _finite_number(name: str, value: float, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0.0
        or (positive and value <= 0.0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _signature_message(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class MPSBackendCertificationPolicy:
    """Root-approved, measurement-independent CPU/MPS acceptance policy."""

    fixture_set_digest: str
    algorithm_source_manifest_digest: str
    approved_certification_runner_digest: str
    approved_runner_id: str
    approved_runner_public_key_hex: str
    cpu_runtime_compatibility_digest: str
    mps_runtime_compatibility_digest: str
    minimum_pcg_iterations: int = 10
    pcg_solution_relative_error_tolerance: float = 5.0e-4
    pcg_cross_backend_relative_error_tolerance: float = 5.0e-4
    pcg_true_residual_tolerance: float = 5.0e-5
    frozen_stationarity_tolerance: float = 5.0e-3
    robust_stationarity_tolerance: float = 5.0e-3
    analysis_dbz_tolerance: float = 2.0e-2
    forecast_dbz_tolerance: float = 5.0e-2
    metric_score_tolerance: float = 1.0e-2
    decision_statistic_tolerance: float = 1.0e-2
    promotion_decision_threshold: float = 0.0
    minimum_decision_margin: float = 5.0e-2
    deterministic_analysis_dbz_tolerance: float = 0.0
    deterministic_forecast_dbz_tolerance: float = 0.0
    deterministic_decision_statistic_tolerance: float = 0.0
    required_nonfinite_fallback_reason: str = "no_valid_observations"
    contract: str = _MPS_CERTIFICATION_POLICY_VERSION
    policy_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "fixture_set_digest",
            "algorithm_source_manifest_digest",
            "approved_certification_runner_digest",
            "cpu_runtime_compatibility_digest",
            "mps_runtime_compatibility_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.contract != _MPS_CERTIFICATION_POLICY_VERSION
            or not self.approved_runner_id.strip()
            or re.fullmatch(
                r"[0-9a-f]{64}", self.approved_runner_public_key_hex
            )
            is None
            or type(self.minimum_pcg_iterations) is not int
            or self.minimum_pcg_iterations < 10
            or not self.required_nonfinite_fallback_reason.strip()
        ):
            raise ValueError("MPS certification policy is invalid")
        for name in (
            "pcg_solution_relative_error_tolerance",
            "pcg_cross_backend_relative_error_tolerance",
            "pcg_true_residual_tolerance",
            "frozen_stationarity_tolerance",
            "robust_stationarity_tolerance",
            "analysis_dbz_tolerance",
            "forecast_dbz_tolerance",
            "metric_score_tolerance",
            "decision_statistic_tolerance",
            "minimum_decision_margin",
        ):
            _finite_number(name, getattr(self, name), positive=True)
        for name in (
            "deterministic_analysis_dbz_tolerance",
            "deterministic_forecast_dbz_tolerance",
            "deterministic_decision_statistic_tolerance",
        ):
            _finite_number(name, getattr(self, name))
        if not np.isfinite(self.promotion_decision_threshold):
            raise ValueError("promotion decision threshold must be finite")
        object.__setattr__(self, "policy_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "policy_digest"
        }

    def validate_integrity(self) -> None:
        if self.policy_digest != json_digest(self.payload):
            raise ValueError("MPS certification policy digest mismatch")


@dataclass(frozen=True)
class MPSBackendCertificationEvidence:
    """Signed measurements from one approved CPU/MPS paired execution."""

    policy_digest: str
    fixture_set_digest: str
    cpu_runtime_compatibility_digest: str
    cpu_runtime_exact_digest: str
    mps_runtime_compatibility_digest: str
    mps_runtime_exact_digest: str
    algorithm_source_manifest_digest: str
    certification_runner_digest: str
    runner_id: str
    runner_public_key_hex: str
    cpu_raw_result_digest: str
    mps_raw_result_digest: str
    mps_repeat_raw_result_digest: str
    cpu_pcg_solution_relative_error: float
    mps_pcg_solution_relative_error: float
    pcg_cross_backend_relative_error: float
    cpu_pcg_true_relative_residual: float
    mps_pcg_true_relative_residual: float
    cpu_pcg_iterations: int
    mps_pcg_iterations: int
    frozen_stationarity_max_abs_difference: float
    robust_stationarity_max_abs_difference: float
    analysis_max_abs_difference_dbz: float
    forecast_max_abs_difference_dbz: float
    metric_score_max_abs_difference: float
    cpu_promotion_decision_statistic: float
    mps_promotion_decision_statistic: float
    cpu_nonfinite_fallback_reason: str
    mps_nonfinite_fallback_reason: str
    mps_repeat_analysis_max_abs_difference_dbz: float
    mps_repeat_forecast_max_abs_difference_dbz: float
    mps_repeat_decision_statistic_max_abs_difference: float
    runner_signature_hex: str
    contract: str = _MPS_CERTIFICATION_EVIDENCE_VERSION
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "policy_digest",
            "fixture_set_digest",
            "cpu_runtime_compatibility_digest",
            "cpu_runtime_exact_digest",
            "mps_runtime_compatibility_digest",
            "mps_runtime_exact_digest",
            "algorithm_source_manifest_digest",
            "certification_runner_digest",
            "cpu_raw_result_digest",
            "mps_raw_result_digest",
            "mps_repeat_raw_result_digest",
        ):
            _require_digest(name, getattr(self, name))
        measurements = (
            "cpu_pcg_solution_relative_error",
            "mps_pcg_solution_relative_error",
            "pcg_cross_backend_relative_error",
            "cpu_pcg_true_relative_residual",
            "mps_pcg_true_relative_residual",
            "frozen_stationarity_max_abs_difference",
            "robust_stationarity_max_abs_difference",
            "analysis_max_abs_difference_dbz",
            "forecast_max_abs_difference_dbz",
            "metric_score_max_abs_difference",
            "mps_repeat_analysis_max_abs_difference_dbz",
            "mps_repeat_forecast_max_abs_difference_dbz",
            "mps_repeat_decision_statistic_max_abs_difference",
        )
        for name in measurements:
            _finite_number(name, getattr(self, name))
        for name in (
            "cpu_promotion_decision_statistic",
            "mps_promotion_decision_statistic",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if (
            self.contract != _MPS_CERTIFICATION_EVIDENCE_VERSION
            or self.cpu_runtime_exact_digest == self.mps_runtime_exact_digest
            or not self.runner_id.strip()
            or re.fullmatch(r"[0-9a-f]{64}", self.runner_public_key_hex)
            is None
            or re.fullmatch(r"[0-9a-f]{128}", self.runner_signature_hex)
            is None
            or type(self.cpu_pcg_iterations) is not int
            or type(self.mps_pcg_iterations) is not int
            or self.cpu_pcg_iterations <= 0
            or self.mps_pcg_iterations <= 0
            or not self.cpu_nonfinite_fallback_reason.strip()
            or not self.mps_nonfinite_fallback_reason.strip()
        ):
            raise ValueError("MPS backend certification evidence is invalid")
        object.__setattr__(self, "evidence_digest", json_digest(self.payload))

    @classmethod
    def sign(
        cls,
        evidence: MPSBackendCertificationEvidence,
        *,
        runner_private_key: Ed25519PrivateKey,
    ) -> MPSBackendCertificationEvidence:
        public_key_hex = runner_private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        if evidence.runner_public_key_hex != public_key_hex:
            raise ValueError("MPS certification runner key disagrees with signer")
        signature = runner_private_key.sign(
            _signature_message(evidence.signature_payload)
        ).hex()
        return replace(evidence, runner_signature_hex=signature)

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "evidence_digest"
        }

    @property
    def signature_payload(self) -> dict[str, object]:
        return self.payload | {"runner_signature_hex": ""}

    def validate_integrity(self) -> None:
        if self.evidence_digest != json_digest(self.payload):
            raise ValueError("MPS certification evidence digest mismatch")


def validate_mps_backend_certification_evidence(
    evidence: MPSBackendCertificationEvidence,
    policy: MPSBackendCertificationPolicy,
) -> None:
    """Validate signed CPU/MPS measurements against their approved policy."""

    evidence.validate_integrity()
    policy.validate_integrity()
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(policy.approved_runner_public_key_hex)
        ).verify(
            bytes.fromhex(evidence.runner_signature_hex),
            _signature_message(evidence.signature_payload),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("MPS certification runner signature mismatch") from error

    cpu_decision = (
        evidence.cpu_promotion_decision_statistic
        > policy.promotion_decision_threshold
    )
    mps_decision = (
        evidence.mps_promotion_decision_statistic
        > policy.promotion_decision_threshold
    )
    minimum_margin = min(
        abs(
            evidence.cpu_promotion_decision_statistic
            - policy.promotion_decision_threshold
        ),
        abs(
            evidence.mps_promotion_decision_statistic
            - policy.promotion_decision_threshold
        ),
    )
    if (
        evidence.policy_digest != policy.policy_digest
        or evidence.fixture_set_digest != policy.fixture_set_digest
        or evidence.algorithm_source_manifest_digest
        != policy.algorithm_source_manifest_digest
        or evidence.certification_runner_digest
        != policy.approved_certification_runner_digest
        or evidence.runner_id != policy.approved_runner_id
        or evidence.runner_public_key_hex
        != policy.approved_runner_public_key_hex
        or evidence.cpu_runtime_compatibility_digest
        != policy.cpu_runtime_compatibility_digest
        or evidence.mps_runtime_compatibility_digest
        != policy.mps_runtime_compatibility_digest
        or evidence.cpu_pcg_iterations < policy.minimum_pcg_iterations
        or evidence.mps_pcg_iterations < policy.minimum_pcg_iterations
        or evidence.cpu_pcg_solution_relative_error
        > policy.pcg_solution_relative_error_tolerance
        or evidence.mps_pcg_solution_relative_error
        > policy.pcg_solution_relative_error_tolerance
        or evidence.pcg_cross_backend_relative_error
        > policy.pcg_cross_backend_relative_error_tolerance
        or evidence.cpu_pcg_true_relative_residual
        > policy.pcg_true_residual_tolerance
        or evidence.mps_pcg_true_relative_residual
        > policy.pcg_true_residual_tolerance
        or evidence.frozen_stationarity_max_abs_difference
        > policy.frozen_stationarity_tolerance
        or evidence.robust_stationarity_max_abs_difference
        > policy.robust_stationarity_tolerance
        or evidence.analysis_max_abs_difference_dbz
        > policy.analysis_dbz_tolerance
        or evidence.forecast_max_abs_difference_dbz
        > policy.forecast_dbz_tolerance
        or evidence.metric_score_max_abs_difference
        > policy.metric_score_tolerance
        or abs(
            evidence.cpu_promotion_decision_statistic
            - evidence.mps_promotion_decision_statistic
        )
        > policy.decision_statistic_tolerance
        or cpu_decision != mps_decision
        or minimum_margin <= policy.minimum_decision_margin
        or evidence.cpu_nonfinite_fallback_reason
        != policy.required_nonfinite_fallback_reason
        or evidence.mps_nonfinite_fallback_reason
        != policy.required_nonfinite_fallback_reason
        or evidence.mps_repeat_analysis_max_abs_difference_dbz
        > policy.deterministic_analysis_dbz_tolerance
        or evidence.mps_repeat_forecast_max_abs_difference_dbz
        > policy.deterministic_forecast_dbz_tolerance
        or evidence.mps_repeat_decision_statistic_max_abs_difference
        > policy.deterministic_decision_statistic_tolerance
    ):
        raise ValueError("MPS backend is not deployment-certified")


def validate_mps_backend_certification(
    evidence: MPSBackendCertificationEvidence,
    policy: MPSBackendCertificationPolicy,
    *,
    execution_device: str | torch.device,
    active_algorithm_source_manifest_digest: str,
    active_certification_runner_digest: str,
) -> None:
    """Validate evidence plus the active MPS device, runtime, and code."""

    validate_mps_backend_certification_evidence(evidence, policy)
    device = torch.device(execution_device)
    active_runtime = numerical_runtime_manifest(device)
    for name, value in (
        ("active algorithm source", active_algorithm_source_manifest_digest),
        ("active certification runner", active_certification_runner_digest),
    ):
        _require_digest(name, value)
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        device.type != "mps"
        or mps_backend is None
        or not bool(mps_backend.is_available())
        or evidence.algorithm_source_manifest_digest
        != active_algorithm_source_manifest_digest
        or evidence.certification_runner_digest
        != active_certification_runner_digest
        or evidence.mps_runtime_compatibility_digest
        != active_runtime.compatibility_digest
        or evidence.mps_runtime_exact_digest != active_runtime.exact_digest
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
