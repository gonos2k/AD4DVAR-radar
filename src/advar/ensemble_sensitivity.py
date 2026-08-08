"""Ensemble forecast sensitivity to observations (EFSO)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import stat

import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest


PRECISION_OPERATOR_TRUST_STORE_CONTRACT = (
    "advar-precision-operator-trust-store-v2"
)
_MAXIMUM_PRECISION_TRUST_STORE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _PrecisionOperatorTrustStore:
    approved_operator_digests: frozenset[str]
    content_digest: str


def _load_precision_operator_trust_store(
    path: str | Path,
) -> _PrecisionOperatorTrustStore:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError("precision trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError("precision trust store must be root-owned and immutable")
        if metadata.st_size > _MAXIMUM_PRECISION_TRUST_STORE_BYTES:
            raise ValueError("precision trust store is too large")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            document = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict) or set(document) != {
        "contract",
        "approved_operator_digests",
    }:
        raise ValueError("invalid precision trust store")
    if document["contract"] != PRECISION_OPERATOR_TRUST_STORE_CONTRACT:
        raise ValueError("unsupported precision trust store")
    raw = document["approved_operator_digests"]
    if not isinstance(raw, list) or not raw or any(
        not isinstance(value, str) for value in raw
    ):
        raise ValueError("precision trust store requires approved digests")
    approved = frozenset(raw)
    if len(approved) != len(raw):
        raise ValueError("precision approval digests must be unique")
    for value in approved:
        _require_digest("precision operator digest", value)
    canonical = {
        "contract": PRECISION_OPERATOR_TRUST_STORE_CONTRACT,
        "approved_operator_digests": sorted(approved),
    }
    return _PrecisionOperatorTrustStore(
        approved_operator_digests=approved,
        content_digest=json_digest(canonical),
    )


def _require_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _owned_float_tensor(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating Tensor")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")
    return value.detach().clone()


@dataclass(frozen=True)
class PrecisionOperatorArtifact:
    """Content-addressed dense precision/covariance pair for research EFSO."""

    precision: Tensor
    covariance: Tensor
    observation_ids: tuple[str, ...]
    forecast_run_digest: str
    observation_error_model_digest: str
    calibration_manifest_digest: str
    maximum_observation_count: int = 512
    maximum_dense_operator_bytes: int = 512 * 1024**2
    maximum_condition_number: float = 1.0e8
    maximum_factorization_flops: int = 1_000_000_000
    observation_index_digest: str = field(init=False)
    minimum_eigenvalue: float = field(init=False)
    condition_number: float = field(init=False)
    contract: str = "precision-operator-artifact-v3"
    operator_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "precision-operator-artifact-v3":
            raise ValueError("unsupported precision operator artifact")
        if (
            type(self.maximum_observation_count) is not int
            or self.maximum_observation_count <= 0
            or type(self.maximum_dense_operator_bytes) is not int
            or self.maximum_dense_operator_bytes <= 0
            or not math.isfinite(self.maximum_condition_number)
            or self.maximum_condition_number < 1.0
            or type(self.maximum_factorization_flops) is not int
            or self.maximum_factorization_flops <= 0
        ):
            raise ValueError("precision operator budgets must be positive")
        if not isinstance(self.precision, Tensor) or not isinstance(
            self.covariance, Tensor
        ):
            raise TypeError("precision operators must be Tensors")
        if (
            self.precision.ndim != 2
            or self.precision.shape[0] != self.precision.shape[1]
        ):
            raise ValueError("precision operators must be square")
        observation_count = self.precision.shape[0]
        factorization_flops = 4 * observation_count**3
        dense_bytes = (
            self.precision.numel() * self.precision.element_size()
            + self.covariance.numel() * self.covariance.element_size()
        )
        if observation_count > self.maximum_observation_count:
            raise ValueError("dense precision operator exceeds observation budget")
        if dense_bytes > self.maximum_dense_operator_bytes:
            raise ValueError("dense precision operator exceeds byte budget")
        if factorization_flops > self.maximum_factorization_flops:
            raise ValueError("dense precision operator exceeds factorization budget")
        if (
            len(self.observation_ids) != observation_count
            or len(set(self.observation_ids)) != observation_count
            or any(not isinstance(value, str) or not value for value in self.observation_ids)
        ):
            raise ValueError("precision observation identifiers must be unique")
        for name in (
            "forecast_run_digest",
            "observation_error_model_digest",
            "calibration_manifest_digest",
        ):
            _require_digest(name, getattr(self, name))
        precision = _owned_float_tensor("precision", self.precision)
        covariance = _owned_float_tensor("covariance", self.covariance)
        if (
            precision.ndim != 2
            or precision.shape[0] != precision.shape[1]
            or covariance.shape != precision.shape
            or precision.dtype != covariance.dtype
            or precision.device != covariance.device
        ):
            raise ValueError("precision operators must be aligned square matrices")
        tolerance = 256.0 * torch.finfo(precision.dtype).eps
        identity = torch.eye(
            precision.shape[0], dtype=precision.dtype, device=precision.device
        )
        if not torch.allclose(precision, precision.T, rtol=0.0, atol=tolerance):
            raise ValueError("precision matrix must be symmetric")
        if not torch.allclose(covariance, covariance.T, rtol=0.0, atol=tolerance):
            raise ValueError("covariance matrix must be symmetric")
        if int(torch.linalg.cholesky_ex(covariance).info.max()) != 0:
            raise ValueError("covariance matrix must be positive definite")
        if int(torch.linalg.cholesky_ex(precision).info.max()) != 0:
            raise ValueError("precision matrix must be positive definite")
        eigenvalues = torch.linalg.eigvalsh(covariance)
        minimum_eigenvalue = float(torch.amin(eigenvalues))
        condition_number = float(torch.amax(eigenvalues) / torch.amin(eigenvalues))
        if condition_number > self.maximum_condition_number:
            raise ValueError("covariance condition number exceeds its budget")
        if not torch.allclose(
            covariance @ precision, identity, rtol=1e-5, atol=tolerance
        ):
            raise ValueError("precision and covariance matrices are inconsistent")
        observation_index_digest = json_digest(
            {
                "contract": "ordered-observation-index-v1",
                "observation_ids": list(self.observation_ids),
                "forecast_run_digest": self.forecast_run_digest,
                "observation_error_model_digest": (
                    self.observation_error_model_digest
                ),
                "calibration_manifest_digest": self.calibration_manifest_digest,
            }
        )
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "minimum_eigenvalue", minimum_eigenvalue)
        object.__setattr__(self, "condition_number", condition_number)
        object.__setattr__(self, "observation_index_digest", observation_index_digest)
        object.__setattr__(self, "operator_digest", _precision_operator_digest(self))


def _precision_operator_digest(value: PrecisionOperatorArtifact) -> str:
    return json_digest(
        {
            "contract": value.contract,
            "precision": tensor_digest(value.precision),
            "covariance": tensor_digest(value.covariance),
            "observation_index_digest": value.observation_index_digest,
            "observation_ids": list(value.observation_ids),
            "forecast_run_digest": value.forecast_run_digest,
            "observation_error_model_digest": value.observation_error_model_digest,
            "calibration_manifest_digest": value.calibration_manifest_digest,
            "maximum_observation_count": value.maximum_observation_count,
            "maximum_dense_operator_bytes": value.maximum_dense_operator_bytes,
            "maximum_condition_number": value.maximum_condition_number,
            "maximum_factorization_flops": value.maximum_factorization_flops,
            "minimum_eigenvalue": value.minimum_eigenvalue,
            "condition_number": value.condition_number,
        }
    )


def validate_precision_operator_artifact(
    value: PrecisionOperatorArtifact,
) -> None:
    """Reject an operator whose retained tensors changed after approval."""

    if value.operator_digest != _precision_operator_digest(value):
        raise ValueError("precision operator digest mismatch")


@dataclass(frozen=True, init=False)
class PrecisionWeightedInnovationEvidence:
    """Verified ``z = R^-1 d`` evidence for one observation index."""

    innovation: Tensor
    precision_weighted_innovation: Tensor
    precision_operator_digest: str
    observation_index_digest: str
    observation_ids: tuple[str, ...]
    trust_store_digest: str | None
    relative_solve_residual: float
    contract: str = "precision-weighted-innovation-evidence-v1"
    evidence_digest: str = field(init=False)

    def __init__(self) -> None:
        raise TypeError("use a precision-evidence factory")

    @classmethod
    def from_operator_artifact(
        cls,
        *,
        innovation: Tensor,
        operator: PrecisionOperatorArtifact,
        trust_store_path: str | Path,
        maximum_relative_residual: float = 1e-6,
    ) -> PrecisionWeightedInnovationEvidence:
        """Apply approved operators and verify ``R(R^-1 d) = d``."""

        validate_precision_operator_artifact(operator)
        trust = _load_precision_operator_trust_store(trust_store_path)
        if operator.operator_digest not in trust.approved_operator_digests:
            raise ValueError("precision operator approval is not trusted")
        original = _owned_float_tensor("innovation", innovation)
        weighted = _owned_float_tensor(
            "precision-weighted innovation", operator.precision @ original
        )
        reconstructed = _owned_float_tensor(
            "reconstructed innovation", operator.covariance @ weighted
        )
        if weighted.shape != original.shape or reconstructed.shape != original.shape:
            raise ValueError("precision operators must preserve observation shape")
        residual = torch.linalg.vector_norm(reconstructed - original)
        scale = torch.linalg.vector_norm(original).clamp_min(
            torch.finfo(original.dtype).eps
        )
        relative = float(residual / scale)
        if not math.isfinite(maximum_relative_residual) or (
            maximum_relative_residual < 0.0
        ):
            raise ValueError("precision residual limit must be nonnegative")
        if relative > maximum_relative_residual:
            raise ValueError("precision-weighted innovation solve is inaccurate")
        return _new_precision_evidence(
            innovation=original,
            precision_weighted_innovation=weighted,
            precision_operator_digest=operator.operator_digest,
            observation_index_digest=operator.observation_index_digest,
            observation_ids=operator.observation_ids,
            trust_store_digest=trust.content_digest,
            relative_solve_residual=relative,
        )

    @classmethod
    def from_diagonal_r(
        cls,
        *,
        innovation: Tensor,
        inverse_observation_variance: Tensor,
        observation_error_model_digest: str,
        observation_ids: tuple[str, ...],
    ) -> PrecisionWeightedInnovationEvidence:
        inverse = _owned_float_tensor(
            "inverse observation variance", inverse_observation_variance
        )
        if inverse.shape != innovation.shape or bool(torch.any(inverse <= 0.0)):
            raise ValueError("diagonal-R precision must be positive and aligned")
        original = _owned_float_tensor("innovation", innovation)
        weighted = original * inverse
        reconstructed = weighted / inverse
        residual = torch.linalg.vector_norm(reconstructed - original)
        scale = torch.linalg.vector_norm(original).clamp_min(
            torch.finfo(original.dtype).eps
        )
        relative = float(residual / scale)
        return _new_precision_evidence(
            innovation=original,
            precision_weighted_innovation=weighted,
            precision_operator_digest=observation_error_model_digest,
            observation_index_digest=json_digest(
                {
                    "contract": "ensemble-observation-index-v1",
                    "observation_ids": list(observation_ids),
                }
            ),
            observation_ids=observation_ids,
            trust_store_digest=None,
            relative_solve_residual=relative,
        )


def _new_precision_evidence(**values: object) -> PrecisionWeightedInnovationEvidence:
    result = object.__new__(PrecisionWeightedInnovationEvidence)
    object.__setattr__(result, "contract", "precision-weighted-innovation-evidence-v1")
    for name, value in values.items():
        object.__setattr__(result, name, value)
    innovation = result.innovation.detach().clone()
    weighted = result.precision_weighted_innovation.detach().clone()
    _require_digest("precision_operator_digest", result.precision_operator_digest)
    _require_digest("observation_index_digest", result.observation_index_digest)
    if result.trust_store_digest is not None:
        _require_digest("trust_store_digest", result.trust_store_digest)
    if (
        len(result.observation_ids) != innovation.numel()
        or len(set(result.observation_ids)) != len(result.observation_ids)
    ):
        raise ValueError("precision observation identifiers are not aligned")
    object.__setattr__(result, "innovation", innovation)
    object.__setattr__(result, "precision_weighted_innovation", weighted)
    object.__setattr__(
        result,
        "evidence_digest",
        json_digest(
            {
                "contract": result.contract,
                "innovation": tensor_digest(innovation),
                "precision_weighted_innovation": tensor_digest(weighted),
                "precision_operator_digest": result.precision_operator_digest,
                "observation_index_digest": result.observation_index_digest,
                "observation_ids": list(result.observation_ids),
                "trust_store_digest": result.trust_store_digest,
                "relative_solve_residual": result.relative_solve_residual,
            }
        ),
    )
    return result


def _precision_evidence_digest(
    value: PrecisionWeightedInnovationEvidence,
) -> str:
    return json_digest(
        {
            "contract": value.contract,
            "innovation": tensor_digest(value.innovation),
            "precision_weighted_innovation": tensor_digest(
                value.precision_weighted_innovation
            ),
            "precision_operator_digest": value.precision_operator_digest,
            "observation_index_digest": value.observation_index_digest,
            "observation_ids": list(value.observation_ids),
            "trust_store_digest": value.trust_store_digest,
            "relative_solve_residual": value.relative_solve_residual,
        }
    )


def validate_precision_weighted_innovation_evidence(
    value: PrecisionWeightedInnovationEvidence,
) -> None:
    """Recompute nested precision evidence before every trusted use."""

    if value.evidence_digest != _precision_evidence_digest(value):
        raise ValueError("precision evidence digest mismatch")


@dataclass(frozen=True)
class EnsembleFSOStatistics:
    """All statistics required by the Kalnay et al. ensemble formula.

    ``forecast_error_projection_by_member`` is
    ``X_f.T @ C @ (e_analysis + e_background)`` for every lead and metric.
    The class deliberately requires this term instead of inventing an
    ensemble from one deterministic P1 analysis.
    """

    precision_evidence: PrecisionWeightedInnovationEvidence
    analysis_observation_perturbations: Tensor
    forecast_error_projection_by_member: Tensor
    lead_minutes: tuple[int, ...]
    metric_names: tuple[str, ...]
    analysis_ensemble_digest: str
    forecast_ensemble_digest: str
    verification_reference_digest: str
    analysis_observation_index_digest: str
    observation_ids: tuple[str, ...]
    ensemble_member_ids: tuple[str, ...]
    analysis_ensemble_member_index_digest: str
    forecast_ensemble_member_index_digest: str
    localization: Tensor | None = None
    localization_digest: str | None = None
    contract: str = "ensemble-fso-full-r-statistics-v2"
    statistics_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "ensemble-fso-full-r-statistics-v2":
            raise ValueError("unsupported ensemble FSO statistics")
        evidence = self.precision_evidence
        validate_precision_weighted_innovation_evidence(evidence)
        evidence = _new_precision_evidence(
            innovation=evidence.innovation,
            precision_weighted_innovation=(
                evidence.precision_weighted_innovation
            ),
            precision_operator_digest=evidence.precision_operator_digest,
            observation_index_digest=evidence.observation_index_digest,
            observation_ids=evidence.observation_ids,
            trust_store_digest=evidence.trust_store_digest,
            relative_solve_residual=evidence.relative_solve_residual,
        )
        object.__setattr__(self, "precision_evidence", evidence)
        innovation = evidence.precision_weighted_innovation
        observation_ensemble = _owned_float_tensor(
            "analysis_observation_perturbations",
            self.analysis_observation_perturbations,
        )
        forecast_projection = _owned_float_tensor(
            "forecast_error_projection_by_member",
            self.forecast_error_projection_by_member,
        )
        compatible = (
            observation_ensemble,
            forecast_projection,
        )
        if any(
            value.dtype != innovation.dtype or value.device != innovation.device
            for value in compatible
        ):
            raise ValueError("ensemble FSO tensors must share dtype and device")
        if innovation.ndim != 1 or innovation.numel() == 0:
            raise ValueError("innovation must have shape [observation]")
        if observation_ensemble.ndim != 2:
            raise ValueError(
                "analysis observation perturbations must have shape "
                "[member, observation]"
            )
        member_count, observation_count = observation_ensemble.shape
        if member_count < 3 or observation_count != innovation.numel():
            raise ValueError("ensemble and innovation dimensions disagree")
        if (
            len(self.observation_ids) != observation_count
            or len(set(self.observation_ids)) != observation_count
            or len(self.ensemble_member_ids) != member_count
            or len(set(self.ensemble_member_ids)) != member_count
        ):
            raise ValueError("ensemble index identifiers must be unique and aligned")
        if forecast_projection.shape != (
            member_count,
            len(self.lead_minutes),
            len(self.metric_names),
        ):
            raise ValueError(
                "forecast error projection must have shape "
                "[member, lead, metric]"
            )
        if not self.lead_minutes or any(
            type(value) is not int or value <= 0 for value in self.lead_minutes
        ):
            raise ValueError("ensemble FSO leads must be positive integers")
        if tuple(sorted(set(self.lead_minutes))) != self.lead_minutes:
            raise ValueError("ensemble FSO leads must be unique and sorted")
        if not self.metric_names or any(
            not isinstance(value, str) or not value for value in self.metric_names
        ):
            raise ValueError("ensemble FSO metric names must be nonempty")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("ensemble FSO metric names must be unique")
        for name, value in (
            ("analysis_ensemble_digest", self.analysis_ensemble_digest),
            ("forecast_ensemble_digest", self.forecast_ensemble_digest),
            (
                "verification_reference_digest",
                self.verification_reference_digest,
            ),
            ("analysis_observation_index_digest", self.analysis_observation_index_digest),
            ("analysis_ensemble_member_index_digest", self.analysis_ensemble_member_index_digest),
            ("forecast_ensemble_member_index_digest", self.forecast_ensemble_member_index_digest),
        ):
            _require_digest(name, value)
        if self.analysis_observation_index_digest != evidence.observation_index_digest:
            raise ValueError("analysis observation ordering disagrees with precision evidence")
        if self.observation_ids != evidence.observation_ids:
            raise ValueError("analysis observation identifiers disagree")
        if self.analysis_ensemble_member_index_digest != self.forecast_ensemble_member_index_digest:
            raise ValueError("analysis and forecast ensemble member ordering disagrees")
        member_index_digest = json_digest(
            {
                "contract": "ensemble-member-index-v1",
                "member_ids": list(self.ensemble_member_ids),
            }
        )
        if self.analysis_ensemble_member_index_digest != member_index_digest:
            raise ValueError("ensemble member identifiers are not canonical")
        expected_analysis_digest = json_digest(
            {
                "contract": "ordered-analysis-ensemble-v1",
                "member_ids": list(self.ensemble_member_ids),
                "observation_ids": list(self.observation_ids),
                "values": tensor_digest(observation_ensemble),
            }
        )
        expected_forecast_digest = json_digest(
            {
                "contract": "ordered-forecast-ensemble-v1",
                "member_ids": list(self.ensemble_member_ids),
                "lead_minutes": list(self.lead_minutes),
                "metric_names": list(self.metric_names),
                "values": tensor_digest(forecast_projection),
            }
        )
        if self.analysis_ensemble_digest != expected_analysis_digest:
            raise ValueError("analysis ensemble content/order digest mismatch")
        if self.forecast_ensemble_digest != expected_forecast_digest:
            raise ValueError("forecast ensemble content/order digest mismatch")
        localization = self.localization
        if localization is None:
            if self.localization_digest is not None:
                raise ValueError("localization digest requires localization")
        else:
            localization = _owned_float_tensor("localization", localization)
            if (
                localization.dtype != innovation.dtype
                or localization.device != innovation.device
            ):
                raise ValueError(
                    "localization must share the ensemble dtype and device"
                )
            if localization.shape != (
                len(self.lead_minutes),
                len(self.metric_names),
                observation_count,
            ):
                raise ValueError(
                    "localization must have shape [lead, metric, observation]"
                )
            if bool(torch.any((localization < 0.0) | (localization > 1.0))):
                raise ValueError("localization must be inside [0, 1]")
            digest = tensor_digest(localization)
            if self.localization_digest is not None and (
                self.localization_digest != digest
            ):
                raise ValueError("localization digest mismatch")
            object.__setattr__(self, "localization_digest", digest)
            object.__setattr__(self, "localization", localization)
        _require_centered_members(
            "analysis observation perturbations",
            observation_ensemble,
        )
        _require_centered_members(
            "forecast error projection",
            forecast_projection,
        )
        object.__setattr__(
            self,
            "analysis_observation_perturbations",
            observation_ensemble,
        )
        object.__setattr__(
            self,
            "forecast_error_projection_by_member",
            forecast_projection,
        )
        object.__setattr__(
            self,
            "statistics_digest",
            _ensemble_statistics_digest(self),
        )

    @classmethod
    def from_diagonal_r(
        cls,
        *,
        innovation: Tensor,
        inverse_observation_variance: Tensor,
        analysis_observation_perturbations: Tensor,
        forecast_error_projection_by_member: Tensor,
        lead_minutes: tuple[int, ...],
        metric_names: tuple[str, ...],
        verification_reference_digest: str,
        observation_ids: tuple[str, ...],
        ensemble_member_ids: tuple[str, ...],
        observation_error_model_digest: str,
        localization: Tensor | None = None,
        localization_digest: str | None = None,
    ) -> EnsembleFSOStatistics:
        """Explicitly build the legacy diagonal-R special case."""

        evidence = PrecisionWeightedInnovationEvidence.from_diagonal_r(
            innovation=innovation,
            inverse_observation_variance=inverse_observation_variance,
            observation_error_model_digest=observation_error_model_digest,
            observation_ids=observation_ids,
        )
        analysis_digest = json_digest(
            {
                "contract": "ordered-analysis-ensemble-v1",
                "member_ids": list(ensemble_member_ids),
                "observation_ids": list(observation_ids),
                "values": tensor_digest(analysis_observation_perturbations),
            }
        )
        forecast_digest = json_digest(
            {
                "contract": "ordered-forecast-ensemble-v1",
                "member_ids": list(ensemble_member_ids),
                "lead_minutes": list(lead_minutes),
                "metric_names": list(metric_names),
                "values": tensor_digest(forecast_error_projection_by_member),
            }
        )
        return cls(
            precision_evidence=evidence,
            analysis_observation_perturbations=(
                analysis_observation_perturbations
            ),
            forecast_error_projection_by_member=(
                forecast_error_projection_by_member
            ),
            lead_minutes=lead_minutes,
            metric_names=metric_names,
            analysis_ensemble_digest=analysis_digest,
            forecast_ensemble_digest=forecast_digest,
            verification_reference_digest=verification_reference_digest,
            analysis_observation_index_digest=evidence.observation_index_digest,
            observation_ids=observation_ids,
            ensemble_member_ids=ensemble_member_ids,
            analysis_ensemble_member_index_digest=json_digest(
                {
                    "contract": "ensemble-member-index-v1",
                    "member_ids": list(ensemble_member_ids),
                }
            ),
            forecast_ensemble_member_index_digest=json_digest(
                {
                    "contract": "ensemble-member-index-v1",
                    "member_ids": list(ensemble_member_ids),
                }
            ),
            localization=localization,
            localization_digest=localization_digest,
        )

    @classmethod
    def from_full_r(
        cls,
        *,
        innovation: Tensor,
        precision_operator: PrecisionOperatorArtifact,
        maximum_relative_residual: float,
        trust_store_path: str | Path,
        analysis_observation_perturbations: Tensor,
        forecast_error_projection_by_member: Tensor,
        lead_minutes: tuple[int, ...],
        metric_names: tuple[str, ...],
        verification_reference_digest: str,
        ensemble_member_ids: tuple[str, ...],
        localization: Tensor | None = None,
        localization_digest: str | None = None,
    ) -> EnsembleFSOStatistics:
        evidence = PrecisionWeightedInnovationEvidence.from_operator_artifact(
            innovation=innovation,
            operator=precision_operator,
            trust_store_path=trust_store_path,
            maximum_relative_residual=maximum_relative_residual,
        )
        observation_ids = precision_operator.observation_ids
        analysis_digest = json_digest(
            {
                "contract": "ordered-analysis-ensemble-v1",
                "member_ids": list(ensemble_member_ids),
                "observation_ids": list(observation_ids),
                "values": tensor_digest(analysis_observation_perturbations),
            }
        )
        forecast_digest = json_digest(
            {
                "contract": "ordered-forecast-ensemble-v1",
                "member_ids": list(ensemble_member_ids),
                "lead_minutes": list(lead_minutes),
                "metric_names": list(metric_names),
                "values": tensor_digest(forecast_error_projection_by_member),
            }
        )
        member_digest = json_digest(
            {
                "contract": "ensemble-member-index-v1",
                "member_ids": list(ensemble_member_ids),
            }
        )
        return cls(
            precision_evidence=evidence,
            analysis_observation_perturbations=analysis_observation_perturbations,
            forecast_error_projection_by_member=forecast_error_projection_by_member,
            lead_minutes=lead_minutes,
            metric_names=metric_names,
            analysis_ensemble_digest=analysis_digest,
            forecast_ensemble_digest=forecast_digest,
            verification_reference_digest=verification_reference_digest,
            analysis_observation_index_digest=(
                precision_operator.observation_index_digest
            ),
            observation_ids=observation_ids,
            ensemble_member_ids=ensemble_member_ids,
            analysis_ensemble_member_index_digest=member_digest,
            forecast_ensemble_member_index_digest=member_digest,
            localization=localization,
            localization_digest=localization_digest,
        )


def _require_centered_members(name: str, value: Tensor) -> None:
    mean = torch.mean(value, dim=0)
    scale = torch.amax(torch.abs(value)).clamp_min(1.0)
    tolerance = 128.0 * torch.finfo(value.dtype).eps * scale
    if bool(torch.any(torch.abs(mean) > tolerance)):
        raise ValueError(f"{name} must be centered across members")


def _ensemble_statistics_digest(value: EnsembleFSOStatistics) -> str:
    return json_digest(
        {
            "contract": value.contract,
            "precision_evidence_digest": _precision_evidence_digest(
                value.precision_evidence
            ),
            "analysis_observation_perturbations": tensor_digest(
                value.analysis_observation_perturbations
            ),
            "forecast_error_projection_by_member": tensor_digest(
                value.forecast_error_projection_by_member
            ),
            "lead_minutes": list(value.lead_minutes),
            "metric_names": list(value.metric_names),
            "analysis_ensemble_digest": value.analysis_ensemble_digest,
            "forecast_ensemble_digest": value.forecast_ensemble_digest,
            "verification_reference_digest": (
                value.verification_reference_digest
            ),
            "analysis_observation_index_digest": (
                value.analysis_observation_index_digest
            ),
            "observation_ids": list(value.observation_ids),
            "ensemble_member_ids": list(value.ensemble_member_ids),
            "analysis_ensemble_member_index_digest": (
                value.analysis_ensemble_member_index_digest
            ),
            "forecast_ensemble_member_index_digest": (
                value.forecast_ensemble_member_index_digest
            ),
            "localization": (
                None
                if value.localization is None
                else tensor_digest(value.localization)
            ),
        }
    )


@dataclass(frozen=True)
class EnsembleFSO:
    """Per-observation ensemble estimate of forecast-error change."""

    observation_impact: Tensor
    total_impact: Tensor
    beneficial_fraction: Tensor
    total_impact_jackknife_std: Tensor
    lead_minutes: tuple[int, ...]
    metric_names: tuple[str, ...]
    statistics_digest: str
    contract: str = "ensemble-fso-v1"
    ensemble_fso_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "ensemble-fso-v1":
            raise ValueError("unsupported ensemble FSO result")
        impact = _owned_float_tensor("observation_impact", self.observation_impact)
        total = _owned_float_tensor("total_impact", self.total_impact)
        fraction = _owned_float_tensor(
            "beneficial_fraction",
            self.beneficial_fraction,
        )
        jackknife = _owned_float_tensor(
            "total_impact_jackknife_std",
            self.total_impact_jackknife_std,
        )
        if impact.ndim != 3 or impact.shape[:2] != (
            len(self.lead_minutes),
            len(self.metric_names),
        ):
            raise ValueError(
                "observation impact must have shape [lead, metric, observation]"
            )
        if (
            total.shape != impact.shape[:2]
            or fraction.shape != impact.shape[:2]
            or jackknife.shape != impact.shape[:2]
        ):
            raise ValueError("ensemble FSO summaries have invalid shapes")
        if bool(torch.any((fraction < 0.0) | (fraction > 1.0))):
            raise ValueError("beneficial fraction must be inside [0, 1]")
        if not torch.allclose(total, impact.sum(dim=-1), rtol=0.0, atol=0.0):
            raise ValueError("ensemble FSO total does not match its impacts")
        _require_digest("statistics_digest", self.statistics_digest)
        object.__setattr__(self, "observation_impact", impact)
        object.__setattr__(self, "total_impact", total)
        object.__setattr__(self, "beneficial_fraction", fraction)
        object.__setattr__(self, "total_impact_jackknife_std", jackknife)
        object.__setattr__(
            self,
            "ensemble_fso_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "observation_impact": tensor_digest(impact),
                    "total_impact": tensor_digest(total),
                    "beneficial_fraction": tensor_digest(fraction),
                    "total_impact_jackknife_std": tensor_digest(jackknife),
                    "lead_minutes": list(self.lead_minutes),
                    "metric_names": list(self.metric_names),
                    "statistics_digest": self.statistics_digest,
                }
            ),
        )


def compute_ensemble_fso(statistics: EnsembleFSOStatistics) -> EnsembleFSO:
    """Compute the direct ensemble observation-impact estimate.

    Negative values reduce the configured forecast-error metric and are
    therefore beneficial. Positive values increase it.
    """

    validate_precision_weighted_innovation_evidence(
        statistics.precision_evidence
    )
    if statistics.statistics_digest != _ensemble_statistics_digest(statistics):
        raise ValueError("ensemble FSO statistics digest mismatch")
    member_count = statistics.analysis_observation_perturbations.shape[0]
    cross_projection = torch.einsum(
        "ko,klm->lmo",
        statistics.analysis_observation_perturbations,
        statistics.forecast_error_projection_by_member,
    ) / (member_count - 1)
    impact = cross_projection * statistics.precision_evidence.precision_weighted_innovation[
        None, None, :
    ]
    if statistics.localization is not None:
        impact = impact * statistics.localization
        support = statistics.localization > 0.0
    else:
        support = torch.ones_like(impact, dtype=torch.bool)
    support_count = support.sum(dim=-1).clamp_min(1)
    beneficial = ((impact < 0.0) & support).sum(dim=-1) / support_count
    precision = statistics.precision_evidence.precision_weighted_innovation
    localized_precision = (
        precision[None, None, :]
        if statistics.localization is None
        else precision[None, None, :] * statistics.localization
    )
    member_contribution = torch.einsum(
        "ko,klm,lmo->klm",
        statistics.analysis_observation_perturbations,
        statistics.forecast_error_projection_by_member,
        localized_precision,
    )
    recentering = member_count / (member_count - 1)
    leave_one_total = (
        member_contribution.sum(dim=0)[None]
        - recentering * member_contribution
    ) / (member_count - 2)
    leave_one_mean = leave_one_total.mean(dim=0)
    jackknife = torch.sqrt(
        (member_count - 1)
        / member_count
        * torch.sum((leave_one_total - leave_one_mean) ** 2, dim=0)
    )
    return EnsembleFSO(
        observation_impact=impact,
        total_impact=impact.sum(dim=-1),
        beneficial_fraction=beneficial.to(impact.dtype),
        total_impact_jackknife_std=jackknife,
        lead_minutes=statistics.lead_minutes,
        metric_names=statistics.metric_names,
        statistics_digest=statistics.statistics_digest,
    )


def validate_ensemble_fso(result: EnsembleFSO) -> None:
    """Reject an EFSO result whose retained tensors were modified."""

    expected = json_digest(
        {
            "contract": result.contract,
            "observation_impact": tensor_digest(result.observation_impact),
            "total_impact": tensor_digest(result.total_impact),
            "beneficial_fraction": tensor_digest(result.beneficial_fraction),
            "total_impact_jackknife_std": tensor_digest(
                result.total_impact_jackknife_std
            ),
            "lead_minutes": list(result.lead_minutes),
            "metric_names": list(result.metric_names),
            "statistics_digest": result.statistics_digest,
        }
    )
    if result.ensemble_fso_digest != expected:
        raise ValueError("ensemble FSO result digest mismatch")
