"""Measured CPU/MPS numerical certification for operational deployment.

This module deliberately owns the fixtures and runner semantics.  Callers may
approve a policy and provide a signing key, but they cannot inject comparison
results or replace the product-code computations used by the certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import numpy as np
import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest
from ._runtime import (
    MPSBackendCertificationEvidence,
    MPSBackendCertificationPolicy,
    numerical_runtime_manifest,
    validate_mps_backend_certification,
)
from .calibration import algorithm_bundle_digest
from .matrix_free import pcg
from .nowcast import NowcastConfig, RadarGridTimeContract
from .physics import echo_to_dbz
from .variational import AnalysisConfig, variational_nowcast


MPS_CERTIFICATION_FIXTURE_VERSION = "advar-mps-certification-fixtures-v1"
MPS_CERTIFICATION_RUNNER_VERSION = "advar-mps-certification-runner-v1"
MPS_CERTIFICATION_RESULT_VERSION = "advar-mps-certification-result-v1"
_PCG_SIZE = 96
_PCG_RTOL = 1.0e-5
_P1_GRID_SIZE = 8


def mps_certification_fixture_set_digest() -> str:
    """Content-address every numerical fixture and its scoring semantics."""

    return json_digest(
        {
            "contract": MPS_CERTIFICATION_FIXTURE_VERSION,
            "pcg": {
                "operator": "variable-diagonal-tridiagonal-spd-v1",
                "size": _PCG_SIZE,
                "diagonal_start": 2.1,
                "diagonal_end": 4.0,
                "off_diagonal": -1.0,
                "preconditioner": "jacobi-diagonal-v1",
                "rtol": _PCG_RTOL,
                "maximum_iterations": 4 * _PCG_SIZE,
                "oracle": "torch-linalg-solve-cpu-float64-v1",
            },
            "p1": {
                "field": "stationary-gaussian-echo-v1",
                "grid_size": _P1_GRID_SIZE,
                "dtype": "float32",
                "horizon_minutes": 30,
                "analysis": {
                    "censored_background_policy": "detection_limit",
                    "maximum_outer_iterations": 5,
                    "maximum_pcg_iterations": 80,
                    "pcg_relative_tolerance": 1.0e-5,
                    "stationarity_tolerance": 1.0e-2,
                    "irls_tolerance": 1.0e-2,
                },
                "score": "finite-domain-forecast-dbz-mse-v1",
                "parent": "latest-frame-persistence-v1",
                "decision_statistic": "parent-mse-minus-p1-mse-v1",
            },
            "fallback": "all-nonfinite-input-no-valid-observations-v1",
            "determinism": "repeat-identical-mps-run-v1",
        }
    )


def mps_certification_runner_digest() -> str:
    """Digest the exact installed runner source used to create evidence."""

    source = Path(__file__).resolve()
    return json_digest(
        {
            "contract": MPS_CERTIFICATION_RUNNER_VERSION,
            "relative_path": "advar/mps_certification.py",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    )


@dataclass(frozen=True)
class MPSCertificationDeviceResult:
    """Raw scalar and tensor identities from one backend execution."""

    runtime_compatibility_digest: str
    runtime_exact_digest: str
    pcg_solution_digest: str
    pcg_solution_relative_error: float
    pcg_true_relative_residual: float
    pcg_iterations: int
    analysis_dbz_digest: str
    forecast_dbz_digest: str
    forecast_valid_mask_digest: str
    frozen_relative_stationarity: float
    robust_relative_stationarity: float
    metric_score: float
    parent_metric_score: float
    promotion_decision_statistic: float
    nonfinite_fallback_reason: str
    contract: str = MPS_CERTIFICATION_RESULT_VERSION
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for value in (
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.pcg_solution_digest,
            self.analysis_dbz_digest,
            self.forecast_dbz_digest,
            self.forecast_valid_mask_digest,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("MPS certification result digest is invalid")
        values = (
            self.pcg_solution_relative_error,
            self.pcg_true_relative_residual,
            self.frozen_relative_stationarity,
            self.robust_relative_stationarity,
            self.metric_score,
            self.parent_metric_score,
            self.promotion_decision_statistic,
        )
        if (
            self.contract != MPS_CERTIFICATION_RESULT_VERSION
            or type(self.pcg_iterations) is not int
            or self.pcg_iterations <= 0
            or not self.nonfinite_fallback_reason
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                for value in values
            )
            or self.pcg_solution_relative_error < 0.0
            or self.pcg_true_relative_residual < 0.0
            or self.frozen_relative_stationarity < 0.0
            or self.robust_relative_stationarity < 0.0
            or self.metric_score < 0.0
            or self.parent_metric_score < 0.0
        ):
            raise ValueError("MPS certification result is invalid")
        object.__setattr__(self, "result_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "result_digest"
        }


@dataclass(frozen=True)
class _DeviceExecution:
    result: MPSCertificationDeviceResult
    arrays: dict[str, Tensor]


def _tridiagonal_problem(
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    diagonal = torch.linspace(
        2.1,
        4.0,
        _PCG_SIZE,
        dtype=torch.float32,
        device=device,
    )
    rhs = (
        torch.sin(
            torch.linspace(
                0.2,
                7.0,
                _PCG_SIZE,
                dtype=torch.float32,
                device=device,
            )
        )
        + 0.1
    )
    oracle_diagonal = diagonal.detach().cpu().to(torch.float64)
    matrix = torch.diag(oracle_diagonal)
    offset = torch.full((_PCG_SIZE - 1,), -1.0, dtype=torch.float64)
    matrix += torch.diag(offset, diagonal=1) + torch.diag(offset, diagonal=-1)
    oracle = torch.linalg.solve(matrix, rhs.detach().cpu().to(torch.float64))
    return diagonal, rhs, oracle


def _apply_tridiagonal(diagonal: Tensor, value: Tensor) -> Tensor:
    zero = torch.zeros((1,), dtype=value.dtype, device=value.device)
    left = torch.cat((zero, value[:-1]))
    right = torch.cat((value[1:], zero))
    return diagonal * value - left - right


def _run_pcg_fixture(device: torch.device) -> tuple[Tensor, float, float, int]:
    diagonal, rhs, oracle = _tridiagonal_problem(device)
    solved = pcg(
        lambda value: _apply_tridiagonal(diagonal, value),
        rhs,
        preconditioner=lambda value: value / diagonal,
        rtol=_PCG_RTOL,
        max_iterations=4 * _PCG_SIZE,
    )
    solution_cpu = solved.solution.detach().cpu().to(torch.float64)
    relative_error = float(
        torch.linalg.vector_norm(solution_cpu - oracle)
        / torch.linalg.vector_norm(oracle)
    )
    true_residual = rhs - _apply_tridiagonal(diagonal, solved.solution)
    true_relative_residual = float(
        torch.linalg.vector_norm(true_residual.detach().cpu().to(torch.float64))
        / torch.linalg.vector_norm(rhs.detach().cpu().to(torch.float64))
    )
    if not solved.converged:
        raise RuntimeError("MPS certification PCG fixture did not converge")
    return (
        solved.solution.detach().cpu(),
        relative_error,
        true_relative_residual,
        solved.iterations,
    )


def _p1_frames(device: torch.device) -> tuple[Tensor, Tensor]:
    coordinates = torch.arange(
        _P1_GRID_SIZE,
        dtype=torch.float32,
        device=device,
    )
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    blob = -10.0 + 40.0 * torch.exp(
        -((y - 3.5).square() + (x - 3.5).square()) / 4.0
    )
    return torch.stack((blob, blob - 1.0, blob)), blob


def _p1_configs() -> tuple[NowcastConfig, AnalysisConfig, RadarGridTimeContract]:
    return (
        NowcastConfig(horizon_minutes=30),
        AnalysisConfig(
            censored_background_policy="detection_limit",
            maximum_outer_iterations=5,
            maximum_pcg_iterations=80,
            pcg_relative_tolerance=1.0e-5,
            final_linearization_relative_stationarity_tolerance=1.0e-2,
            final_robust_relative_stationarity_tolerance=1.0e-2,
            final_field_gradient_max_tolerance=1.0e-2,
            final_irls_relative_weight_tolerance=1.0e-2,
        ),
        RadarGridTimeContract(
            valid_times=(
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="f" * 64,
        ),
    )


def _run_device_fixture(device: torch.device) -> _DeviceExecution:
    runtime = numerical_runtime_manifest(device)
    pcg_solution, pcg_error, true_residual, pcg_iterations = _run_pcg_fixture(
        device
    )
    frames, verification_frame = _p1_frames(device)
    nowcast_config, analysis_config, grid_contract = _p1_configs()
    forecast, analysis = variational_nowcast(
        frames,
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
        grid_time_contract=grid_contract,
    )
    if (
        analysis.used_fallback
        or not analysis.p1_forecast_eligible
        or analysis.linearization_relative_stationarity is None
        or analysis.robust_relative_stationarity is None
    ):
        raise RuntimeError(
            f"MPS certification P1 fixture failed closed: {analysis.reason}"
        )
    analysis_dbz = echo_to_dbz(
        analysis.analyzed_frames_linear,
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    forecast_dbz = forecast.forecast_dbz
    valid_mask = forecast.valid_mask & torch.isfinite(forecast_dbz)
    if not bool(torch.any(valid_mask)):
        raise RuntimeError("MPS certification forecast has no finite domain")
    verification = verification_frame.expand_as(forecast_dbz)
    metric_score = float(
        torch.mean((forecast_dbz[valid_mask] - verification[valid_mask]).square())
    )
    parent = frames[-1].expand_as(forecast_dbz)
    parent_metric_score = float(
        torch.mean((parent[valid_mask] - verification[valid_mask]).square())
    )

    invalid_frames = torch.full_like(frames, torch.nan)
    _, fallback = variational_nowcast(
        invalid_frames,
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
        qc_mask=torch.zeros_like(invalid_frames, dtype=torch.bool),
        grid_time_contract=grid_contract,
    )
    if not fallback.used_fallback:
        raise RuntimeError("MPS certification nonfinite fixture did not fall back")

    arrays = {
        "pcg_solution": pcg_solution,
        "analysis_dbz": analysis_dbz.detach().cpu(),
        "forecast_dbz": forecast_dbz.detach().cpu(),
        "forecast_valid_mask": valid_mask.detach().cpu(),
    }
    result = MPSCertificationDeviceResult(
        runtime_compatibility_digest=runtime.compatibility_digest,
        runtime_exact_digest=runtime.exact_digest,
        pcg_solution_digest=tensor_digest(arrays["pcg_solution"]),
        pcg_solution_relative_error=pcg_error,
        pcg_true_relative_residual=true_residual,
        pcg_iterations=pcg_iterations,
        analysis_dbz_digest=tensor_digest(arrays["analysis_dbz"]),
        forecast_dbz_digest=tensor_digest(arrays["forecast_dbz"]),
        forecast_valid_mask_digest=tensor_digest(
            arrays["forecast_valid_mask"]
        ),
        frozen_relative_stationarity=analysis.linearization_relative_stationarity,
        robust_relative_stationarity=analysis.robust_relative_stationarity,
        metric_score=metric_score,
        parent_metric_score=parent_metric_score,
        promotion_decision_statistic=parent_metric_score - metric_score,
        nonfinite_fallback_reason=fallback.reason,
    )
    return _DeviceExecution(result=result, arrays=arrays)


def _max_abs_difference(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("MPS certification tensor contracts disagree")
    if left.dtype == torch.bool:
        if not torch.equal(left, right):
            raise ValueError("CPU/MPS certification domains disagree")
        return 0.0
    finite = torch.isfinite(left) & torch.isfinite(right)
    if not torch.equal(torch.isfinite(left), torch.isfinite(right)):
        raise ValueError("CPU/MPS finite domains disagree")
    return (
        0.0
        if not bool(torch.any(finite))
        else float(torch.max(torch.abs(left[finite] - right[finite])))
    )


def _relative_l2_difference(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("MPS certification tensor contracts disagree")
    denominator = float(torch.linalg.vector_norm(left.to(torch.float64)))
    numerator = float(
        torch.linalg.vector_norm((left - right).to(torch.float64))
    )
    return numerator / max(denominator, 1.0e-30)


def create_mps_backend_certification_policy(
    *,
    runner_id: str,
    runner_private_key: Ed25519PrivateKey,
) -> MPSBackendCertificationPolicy:
    """Create the default prereview policy for the active code/runtime class."""

    _require_certification_runtime_policy()
    public_key_hex = runner_private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return MPSBackendCertificationPolicy(
        fixture_set_digest=mps_certification_fixture_set_digest(),
        algorithm_source_manifest_digest=algorithm_bundle_digest(),
        approved_certification_runner_digest=mps_certification_runner_digest(),
        approved_runner_id=runner_id,
        approved_runner_public_key_hex=public_key_hex,
        cpu_runtime_compatibility_digest=(
            numerical_runtime_manifest("cpu").compatibility_digest
        ),
        mps_runtime_compatibility_digest=(
            numerical_runtime_manifest("mps").compatibility_digest
        ),
    )


def run_mps_backend_certification(
    policy: MPSBackendCertificationPolicy,
    *,
    runner_private_key: Ed25519PrivateKey,
) -> tuple[
    MPSBackendCertificationEvidence,
    MPSCertificationDeviceResult,
    MPSCertificationDeviceResult,
    MPSCertificationDeviceResult,
    dict[str, dict[str, Tensor]],
]:
    """Measure CPU and MPS product paths and sign the resulting evidence."""

    _require_certification_runtime_policy()
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        raise RuntimeError("MPS certification requires an available MPS backend")
    policy.validate_integrity()
    active_algorithm = algorithm_bundle_digest()
    active_runner = mps_certification_runner_digest()
    public_key_hex = runner_private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    if (
        policy.fixture_set_digest != mps_certification_fixture_set_digest()
        or policy.algorithm_source_manifest_digest != active_algorithm
        or policy.approved_certification_runner_digest != active_runner
        or policy.approved_runner_public_key_hex != public_key_hex
    ):
        raise ValueError("MPS certification policy disagrees with active runner")

    cpu = _run_device_fixture(torch.device("cpu"))
    mps = _run_device_fixture(torch.device("mps"))
    mps_repeat = _run_device_fixture(torch.device("mps"))
    unsigned_evidence = MPSBackendCertificationEvidence(
        policy_digest=policy.policy_digest,
        fixture_set_digest=mps_certification_fixture_set_digest(),
        cpu_runtime_compatibility_digest=(
            cpu.result.runtime_compatibility_digest
        ),
        cpu_runtime_exact_digest=cpu.result.runtime_exact_digest,
        mps_runtime_compatibility_digest=(
            mps.result.runtime_compatibility_digest
        ),
        mps_runtime_exact_digest=mps.result.runtime_exact_digest,
        algorithm_source_manifest_digest=active_algorithm,
        certification_runner_digest=active_runner,
        runner_id=policy.approved_runner_id,
        cpu_raw_result_digest=cpu.result.result_digest,
        mps_raw_result_digest=mps.result.result_digest,
        mps_repeat_raw_result_digest=mps_repeat.result.result_digest,
        cpu_pcg_solution_relative_error=(
            cpu.result.pcg_solution_relative_error
        ),
        mps_pcg_solution_relative_error=(
            mps.result.pcg_solution_relative_error
        ),
        pcg_cross_backend_relative_error=_relative_l2_difference(
            cpu.arrays["pcg_solution"], mps.arrays["pcg_solution"]
        ),
        cpu_pcg_true_relative_residual=(
            cpu.result.pcg_true_relative_residual
        ),
        mps_pcg_true_relative_residual=(
            mps.result.pcg_true_relative_residual
        ),
        cpu_pcg_iterations=cpu.result.pcg_iterations,
        mps_pcg_iterations=mps.result.pcg_iterations,
        frozen_stationarity_max_abs_difference=abs(
            cpu.result.frozen_relative_stationarity
            - mps.result.frozen_relative_stationarity
        ),
        robust_stationarity_max_abs_difference=abs(
            cpu.result.robust_relative_stationarity
            - mps.result.robust_relative_stationarity
        ),
        analysis_max_abs_difference_dbz=_max_abs_difference(
            cpu.arrays["analysis_dbz"], mps.arrays["analysis_dbz"]
        ),
        forecast_max_abs_difference_dbz=_max_abs_difference(
            cpu.arrays["forecast_dbz"], mps.arrays["forecast_dbz"]
        ),
        metric_score_max_abs_difference=abs(
            cpu.result.metric_score - mps.result.metric_score
        ),
        cpu_promotion_decision_statistic=(
            cpu.result.promotion_decision_statistic
        ),
        mps_promotion_decision_statistic=(
            mps.result.promotion_decision_statistic
        ),
        cpu_nonfinite_fallback_reason=cpu.result.nonfinite_fallback_reason,
        mps_nonfinite_fallback_reason=mps.result.nonfinite_fallback_reason,
        mps_repeat_analysis_max_abs_difference_dbz=_max_abs_difference(
            mps.arrays["analysis_dbz"], mps_repeat.arrays["analysis_dbz"]
        ),
        mps_repeat_forecast_max_abs_difference_dbz=_max_abs_difference(
            mps.arrays["forecast_dbz"], mps_repeat.arrays["forecast_dbz"]
        ),
        mps_repeat_decision_statistic_max_abs_difference=abs(
            mps.result.promotion_decision_statistic
            - mps_repeat.result.promotion_decision_statistic
        ),
        runner_public_key_hex=public_key_hex,
        runner_signature_hex="0" * 128,
    )
    evidence = MPSBackendCertificationEvidence.sign(
        unsigned_evidence,
        runner_private_key=runner_private_key,
    )
    validate_mps_backend_certification(
        evidence,
        policy,
        execution_device="mps",
        active_algorithm_source_manifest_digest=active_algorithm,
        active_certification_runner_digest=active_runner,
    )
    return (
        evidence,
        cpu.result,
        mps.result,
        mps_repeat.result,
        {
            "cpu": cpu.arrays,
            "mps": mps.arrays,
            "mps_repeat": mps_repeat.arrays,
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mps_backend_certification(
    output_dir: str | Path,
    *,
    runner_id: str,
    runner_private_key: Ed25519PrivateKey,
) -> MPSBackendCertificationEvidence:
    """Run certification and write signed policy, evidence, and raw outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    policy = create_mps_backend_certification_policy(
        runner_id=runner_id,
        runner_private_key=runner_private_key,
    )
    evidence, cpu, mps, mps_repeat, arrays = run_mps_backend_certification(
        policy,
        runner_private_key=runner_private_key,
    )
    _write_json(
        output / "policy.json",
        policy.payload | {"policy_digest": policy.policy_digest},
    )
    _write_json(
        output / "evidence.json",
        evidence.payload | {"evidence_digest": evidence.evidence_digest},
    )
    for name, result in (("cpu", cpu), ("mps", mps), ("mps-repeat", mps_repeat)):
        _write_json(
            output / f"{name}-result.json",
            result.payload | {"result_digest": result.result_digest},
        )
        np.savez_compressed(
            output / f"{name}-arrays.npz",
            **cast(
                dict[str, Any],
                {
                    role: tensor.detach().cpu().contiguous().numpy()
                    for role, tensor in arrays[
                        name.replace("-", "_")
                    ].items()
                },
            ),
        )
    artifact_files = sorted(path for path in output.iterdir() if path.is_file())
    _write_json(
        output / "artifact-manifest.json",
        {
            "contract": "advar-mps-certification-artifact-manifest-v1",
            "policy_digest": policy.policy_digest,
            "evidence_digest": evidence.evidence_digest,
            "files": [
                {"name": path.name, "sha256": _file_digest(path)}
                for path in artifact_files
            ],
        },
    )
    return evidence


def _private_key_from_environment() -> Ed25519PrivateKey:
    value = os.environ.get("ADVAR_MPS_CERTIFICATION_PRIVATE_KEY_HEX", "")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(
            "ADVAR_MPS_CERTIFICATION_PRIVATE_KEY_HEX must contain a raw "
            "Ed25519 private key"
        )
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(value))


def _require_certification_runtime_policy() -> None:
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.get_float32_matmul_precision() != "highest"
    ):
        raise RuntimeError(
            "MPS certification requires strict deterministic algorithms and "
            "highest float32 matmul precision"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run signed ADVAR MPS certification"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runner-id", required=True)
    args = parser.parse_args(argv)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    evidence = write_mps_backend_certification(
        args.output_dir,
        runner_id=args.runner_id,
        runner_private_key=_private_key_from_environment(),
    )
    print(evidence.evidence_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
