"""Ensemble forecast sensitivity to observations (EFSO)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest


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
    observation_index_digest: str
    contract: str = "precision-operator-artifact-v1"
    operator_digest: str = field(init=False)

    def __post_init__(self) -> None:
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
        if not torch.allclose(
            covariance @ precision, identity, rtol=1e-5, atol=tolerance
        ):
            raise ValueError("precision and covariance matrices are inconsistent")
        _require_digest("observation_index_digest", self.observation_index_digest)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "operator_digest", json_digest({
            "contract": self.contract,
            "precision": tensor_digest(precision),
            "covariance": tensor_digest(covariance),
            "observation_index_digest": self.observation_index_digest,
        }))


@dataclass(frozen=True, init=False)
class PrecisionWeightedInnovationEvidence:
    """Verified ``z = R^-1 d`` evidence for one observation index."""

    innovation: Tensor
    precision_weighted_innovation: Tensor
    precision_operator_digest: str
    observation_index_digest: str
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
        maximum_relative_residual: float = 1e-6,
    ) -> PrecisionWeightedInnovationEvidence:
        """Apply approved operators and verify ``R(R^-1 d) = d``."""

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
            relative_solve_residual=relative,
        )

    @classmethod
    def from_diagonal_r(
        cls,
        *,
        innovation: Tensor,
        inverse_observation_variance: Tensor,
        observation_error_model_digest: str,
        observation_index_digest: str,
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
            observation_index_digest=observation_index_digest,
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
    object.__setattr__(result, "innovation", innovation)
    object.__setattr__(result, "precision_weighted_innovation", weighted)
    object.__setattr__(result, "evidence_digest", json_digest({
        "contract": result.contract,
        "innovation": tensor_digest(innovation),
        "precision_weighted_innovation": tensor_digest(weighted),
        "precision_operator_digest": result.precision_operator_digest,
        "observation_index_digest": result.observation_index_digest,
        "relative_solve_residual": result.relative_solve_residual,
    }))
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
        if self.analysis_ensemble_member_index_digest != self.forecast_ensemble_member_index_digest:
            raise ValueError("analysis and forecast ensemble member ordering disagrees")
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
        analysis_ensemble_digest: str,
        forecast_ensemble_digest: str,
        verification_reference_digest: str,
        observation_index_digest: str,
        ensemble_member_index_digest: str,
        observation_error_model_digest: str,
        localization: Tensor | None = None,
        localization_digest: str | None = None,
    ) -> EnsembleFSOStatistics:
        """Explicitly build the legacy diagonal-R special case."""

        evidence = PrecisionWeightedInnovationEvidence.from_diagonal_r(
            innovation=innovation,
            inverse_observation_variance=inverse_observation_variance,
            observation_error_model_digest=observation_error_model_digest,
            observation_index_digest=observation_index_digest,
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
            analysis_ensemble_digest=analysis_ensemble_digest,
            forecast_ensemble_digest=forecast_ensemble_digest,
            verification_reference_digest=verification_reference_digest,
            analysis_observation_index_digest=observation_index_digest,
            analysis_ensemble_member_index_digest=ensemble_member_index_digest,
            forecast_ensemble_member_index_digest=ensemble_member_index_digest,
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
        analysis_observation_perturbations: Tensor,
        forecast_error_projection_by_member: Tensor,
        lead_minutes: tuple[int, ...],
        metric_names: tuple[str, ...],
        analysis_ensemble_digest: str,
        forecast_ensemble_digest: str,
        verification_reference_digest: str,
        ensemble_member_index_digest: str,
        localization: Tensor | None = None,
        localization_digest: str | None = None,
    ) -> EnsembleFSOStatistics:
        evidence = PrecisionWeightedInnovationEvidence.from_operator_artifact(
            innovation=innovation,
            operator=precision_operator,
            maximum_relative_residual=maximum_relative_residual,
        )
        return cls(
            precision_evidence=evidence,
            analysis_observation_perturbations=analysis_observation_perturbations,
            forecast_error_projection_by_member=forecast_error_projection_by_member,
            lead_minutes=lead_minutes,
            metric_names=metric_names,
            analysis_ensemble_digest=analysis_ensemble_digest,
            forecast_ensemble_digest=forecast_ensemble_digest,
            verification_reference_digest=verification_reference_digest,
            analysis_observation_index_digest=(
                precision_operator.observation_index_digest
            ),
            analysis_ensemble_member_index_digest=ensemble_member_index_digest,
            forecast_ensemble_member_index_digest=ensemble_member_index_digest,
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
            "analysis_ensemble_member_index_digest": (
                value.analysis_ensemble_member_index_digest
            ),
            "forecast_ensemble_member_index_digest": (
                value.forecast_ensemble_member_index_digest
            ),
            "localization_digest": value.localization_digest,
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
