"""Ensemble forecast sensitivity to observations (EFSO)."""

from __future__ import annotations

from dataclasses import dataclass, field

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
class EnsembleFSOStatistics:
    """All statistics required by the Kalnay et al. ensemble formula.

    ``forecast_error_projection_by_member`` is
    ``X_f.T @ C @ (e_analysis + e_background)`` for every lead and metric.
    The class deliberately requires this term instead of inventing an
    ensemble from one deterministic P1 analysis.
    """

    innovation: Tensor
    analysis_observation_perturbations: Tensor
    forecast_error_projection_by_member: Tensor
    inverse_observation_variance: Tensor
    lead_minutes: tuple[int, ...]
    metric_names: tuple[str, ...]
    analysis_ensemble_digest: str
    forecast_ensemble_digest: str
    verification_reference_digest: str
    observation_error_model_digest: str
    localization: Tensor | None = None
    localization_digest: str | None = None
    contract: str = "ensemble-fso-statistics-v1"
    statistics_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "ensemble-fso-statistics-v1":
            raise ValueError("unsupported ensemble FSO statistics")
        innovation = _owned_float_tensor("innovation", self.innovation)
        observation_ensemble = _owned_float_tensor(
            "analysis_observation_perturbations",
            self.analysis_observation_perturbations,
        )
        forecast_projection = _owned_float_tensor(
            "forecast_error_projection_by_member",
            self.forecast_error_projection_by_member,
        )
        inverse_variance = _owned_float_tensor(
            "inverse_observation_variance",
            self.inverse_observation_variance,
        )
        compatible = (
            observation_ensemble,
            forecast_projection,
            inverse_variance,
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
        if inverse_variance.shape != innovation.shape or bool(
            torch.any(inverse_variance <= 0.0)
        ):
            raise ValueError(
                "inverse observation variance must be positive and match "
                "the innovation"
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
            (
                "observation_error_model_digest",
                self.observation_error_model_digest,
            ),
        ):
            _require_digest(name, value)
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
        object.__setattr__(self, "innovation", innovation)
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
            "inverse_observation_variance",
            inverse_variance,
        )
        object.__setattr__(
            self,
            "statistics_digest",
            _ensemble_statistics_digest(self),
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
            "innovation": tensor_digest(value.innovation),
            "analysis_observation_perturbations": tensor_digest(
                value.analysis_observation_perturbations
            ),
            "forecast_error_projection_by_member": tensor_digest(
                value.forecast_error_projection_by_member
            ),
            "inverse_observation_variance": tensor_digest(
                value.inverse_observation_variance
            ),
            "lead_minutes": list(value.lead_minutes),
            "metric_names": list(value.metric_names),
            "analysis_ensemble_digest": value.analysis_ensemble_digest,
            "forecast_ensemble_digest": value.forecast_ensemble_digest,
            "verification_reference_digest": (
                value.verification_reference_digest
            ),
            "observation_error_model_digest": (
                value.observation_error_model_digest
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
        if impact.ndim != 3 or impact.shape[:2] != (
            len(self.lead_minutes),
            len(self.metric_names),
        ):
            raise ValueError(
                "observation impact must have shape [lead, metric, observation]"
            )
        if total.shape != impact.shape[:2] or fraction.shape != impact.shape[:2]:
            raise ValueError("ensemble FSO summaries have invalid shapes")
        if bool(torch.any((fraction < 0.0) | (fraction > 1.0))):
            raise ValueError("beneficial fraction must be inside [0, 1]")
        if not torch.allclose(total, impact.sum(dim=-1), rtol=0.0, atol=0.0):
            raise ValueError("ensemble FSO total does not match its impacts")
        _require_digest("statistics_digest", self.statistics_digest)
        object.__setattr__(self, "observation_impact", impact)
        object.__setattr__(self, "total_impact", total)
        object.__setattr__(self, "beneficial_fraction", fraction)
        object.__setattr__(
            self,
            "ensemble_fso_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "observation_impact": tensor_digest(impact),
                    "total_impact": tensor_digest(total),
                    "beneficial_fraction": tensor_digest(fraction),
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

    if statistics.statistics_digest != _ensemble_statistics_digest(statistics):
        raise ValueError("ensemble FSO statistics digest mismatch")
    member_count = statistics.analysis_observation_perturbations.shape[0]
    cross_projection = torch.einsum(
        "ko,klm->lmo",
        statistics.analysis_observation_perturbations,
        statistics.forecast_error_projection_by_member,
    ) / (member_count - 1)
    impact = cross_projection * (
        statistics.innovation * statistics.inverse_observation_variance
    )[None, None, :]
    if statistics.localization is not None:
        impact = impact * statistics.localization
    return EnsembleFSO(
        observation_impact=impact,
        total_impact=impact.sum(dim=-1),
        beneficial_fraction=(impact < 0.0).to(impact.dtype).mean(dim=-1),
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
            "lead_minutes": list(result.lead_minutes),
            "metric_names": list(result.metric_names),
            "statistics_digest": result.statistics_digest,
        }
    )
    if result.ensemble_fso_digest != expected:
        raise ValueError("ensemble FSO result digest mismatch")
