"""Deterministic radar range partitions for learned-prior deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from ._digest import json_digest, tensor_digest


def _require_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SourceRadarRegistry:
    """Canonical mapping from mosaic integer indices to physical radar sites."""

    radar_site_digests: tuple[str, ...]
    source_selection_policy_digest: str
    contract: str = "source-radar-registry-v1"
    registry_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "source-radar-registry-v1"
            or not self.radar_site_digests
            or len(set(self.radar_site_digests)) != len(self.radar_site_digests)
        ):
            raise ValueError("source radar registry is invalid")
        for value in self.radar_site_digests:
            _require_digest("source radar site", value)
        _require_digest(
            "source selection policy", self.source_selection_policy_digest
        )
        object.__setattr__(self, "registry_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "radar_site_digests": list(self.radar_site_digests),
            "source_selection_policy_digest": (
                self.source_selection_policy_digest
            ),
        }

    def validate_integrity(self) -> None:
        if self.registry_digest != json_digest(self.payload):
            raise ValueError("source radar registry digest mismatch")


@dataclass(frozen=True)
class RadarSiteLocationRegistry:
    """Ordered projected locations bound to one mosaic source registry."""

    projection_digest: str
    radar_site_digests: tuple[str, ...]
    radar_site_location_digests: tuple[str, ...]
    radar_projected_xy_m: tuple[tuple[float, float], ...]
    contract: str = "radar-site-location-registry-v1"
    registry_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "radar-site-location-registry-v1"
            or not self.radar_site_digests
            or len(set(self.radar_site_digests)) != len(self.radar_site_digests)
            or len(self.radar_site_digests)
            != len(self.radar_site_location_digests)
            or len(self.radar_site_digests) != len(self.radar_projected_xy_m)
            or len(set(self.radar_site_location_digests))
            != len(self.radar_site_location_digests)
            or any(
                len(point) != 2
                or any(not math.isfinite(coordinate) for coordinate in point)
                for point in self.radar_projected_xy_m
            )
        ):
            raise ValueError("radar site-location registry is invalid")
        _require_digest("projection", self.projection_digest)
        for value in (*self.radar_site_digests, *self.radar_site_location_digests):
            _require_digest("radar site location", value)
        object.__setattr__(self, "registry_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "projection_digest": self.projection_digest,
            "radar_site_digests": list(self.radar_site_digests),
            "radar_site_location_digests": list(
                self.radar_site_location_digests
            ),
            "radar_projected_xy_m": [list(point) for point in self.radar_projected_xy_m],
        }

    def validate_integrity(self) -> None:
        if self.registry_digest != json_digest(self.payload):
            raise ValueError("radar site-location registry digest mismatch")


@dataclass(frozen=True)
class RangeGeometryContract:
    """Horizontal radial bands fixed by one radar site and projected grid."""

    radar_site_digest: str
    grid_contract_digest: str
    radar_x_m: float
    radar_y_m: float
    range_regime_labels: tuple[str, ...]
    radial_distance_edges_m: tuple[float, ...]
    horizontal_range_rule_digest: str
    grid_x_m_digest: str
    grid_y_m_digest: str
    radar_site_location_digest: str
    resolver_algorithm: str = "projected-horizontal-euclidean-range-v3"
    contract: str = "radar-horizontal-range-geometry-contract-v3"
    contract_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "radar_site_digest",
            "radar_site_location_digest",
            "grid_contract_digest",
            "horizontal_range_rule_digest",
            "grid_x_m_digest",
            "grid_y_m_digest",
        ):
            _require_digest(name, getattr(self, name))
        edges = self.radial_distance_edges_m
        if (
            self.contract != "radar-horizontal-range-geometry-contract-v3"
            or self.resolver_algorithm
            != "projected-horizontal-euclidean-range-v3"
            or not math.isfinite(self.radar_x_m)
            or not math.isfinite(self.radar_y_m)
            or not self.range_regime_labels
            or len(set(self.range_regime_labels)) != len(self.range_regime_labels)
            or any(
                not value or value.strip() != value
                for value in self.range_regime_labels
            )
            or len(edges) != len(self.range_regime_labels) + 1
            or any(not math.isfinite(value) or value < 0.0 for value in edges)
            or edges[0] != 0.0
            or any(left >= right for left, right in zip(edges, edges[1:]))
        ):
            raise ValueError("range geometry contract is invalid")
        object.__setattr__(self, "contract_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "contract_digest"
        }

    def validate_integrity(self) -> None:
        if self.contract_digest != json_digest(self.payload):
            raise ValueError("range geometry contract digest mismatch")


@dataclass(frozen=True)
class MosaicRangeGeometryContract:
    """Source-aware horizontal bands for a dynamically resolved radar mosaic."""

    source_radar_registry_digest: str
    radar_site_location_registry_digest: str
    source_selection_policy_digest: str
    grid_contract_digest: str
    projection_digest: str
    radar_site_digests: tuple[str, ...]
    radar_site_location_digests: tuple[str, ...]
    radar_projected_xy_m: tuple[tuple[float, float], ...]
    range_regime_labels: tuple[str, ...]
    radial_distance_edges_m: tuple[float, ...]
    horizontal_range_rule_digest: str
    grid_x_m_digest: str
    grid_y_m_digest: str
    resolver_algorithm: str = "source-index-projected-horizontal-range-v3"
    contract: str = "mosaic-horizontal-range-geometry-contract-v3"
    contract_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_radar_registry_digest",
            "radar_site_location_registry_digest",
            "source_selection_policy_digest",
            "grid_contract_digest",
            "projection_digest",
            "horizontal_range_rule_digest",
            "grid_x_m_digest",
            "grid_y_m_digest",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.contract != "mosaic-horizontal-range-geometry-contract-v3"
            or self.resolver_algorithm
            != "source-index-projected-horizontal-range-v3"
            or not self.radar_site_digests
            or len(set(self.radar_site_digests)) != len(self.radar_site_digests)
            or len(self.radar_site_digests)
            != len(self.radar_site_location_digests)
            or len(self.radar_site_digests) != len(self.radar_projected_xy_m)
            or len(set(self.radar_site_location_digests))
            != len(self.radar_site_location_digests)
            or not self.range_regime_labels
            or len(set(self.range_regime_labels)) != len(self.range_regime_labels)
            or any(
                not label or label.strip() != label
                for label in self.range_regime_labels
            )
            or len(self.radial_distance_edges_m)
            != len(self.range_regime_labels) + 1
            or self.radial_distance_edges_m[0] != 0.0
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.radial_distance_edges_m
            )
            or any(
                left >= right
                for left, right in zip(
                    self.radial_distance_edges_m,
                    self.radial_distance_edges_m[1:],
                )
            )
            or any(
                len(point) != 2
                or any(not math.isfinite(coordinate) for coordinate in point)
                for point in self.radar_projected_xy_m
            )
        ):
            raise ValueError("mosaic range geometry contract is invalid")
        for digest in (*self.radar_site_digests, *self.radar_site_location_digests):
            _require_digest("mosaic radar site", digest)
        expected_source_registry_digest = json_digest(
            {
                "contract": "source-radar-registry-v1",
                "radar_site_digests": list(self.radar_site_digests),
                "source_selection_policy_digest": (
                    self.source_selection_policy_digest
                ),
            }
        )
        expected_location_registry_digest = json_digest(
            {
                "contract": "radar-site-location-registry-v1",
                "projection_digest": self.projection_digest,
                "radar_site_digests": list(self.radar_site_digests),
                "radar_site_location_digests": list(
                    self.radar_site_location_digests
                ),
                "radar_projected_xy_m": [
                    list(point) for point in self.radar_projected_xy_m
                ],
            }
        )
        if (
            self.source_radar_registry_digest
            != expected_source_registry_digest
            or self.radar_site_location_registry_digest
            != expected_location_registry_digest
        ):
            raise ValueError("mosaic range registries disagree with site ordering")
        object.__setattr__(self, "contract_digest", json_digest(self.payload))

    @classmethod
    def from_registry(
        cls,
        source_registry: SourceRadarRegistry,
        location_registry: RadarSiteLocationRegistry,
        *,
        grid_contract_digest: str,
        projection_digest: str,
        range_regime_labels: tuple[str, ...],
        radial_distance_edges_m: tuple[float, ...],
        horizontal_range_rule_digest: str,
        grid_x_m_digest: str,
        grid_y_m_digest: str,
        resolver_algorithm: str = "source-index-projected-horizontal-range-v3",
        contract: str = "mosaic-horizontal-range-geometry-contract-v3",
    ) -> MosaicRangeGeometryContract:
        """Build a mosaic geometry only from two integrity-checked registries."""

        if type(source_registry) is not SourceRadarRegistry or type(
            location_registry
        ) is not RadarSiteLocationRegistry:
            raise TypeError("mosaic range geometry requires exact registry types")
        source_registry.validate_integrity()
        location_registry.validate_integrity()
        if source_registry.radar_site_digests != location_registry.radar_site_digests:
            raise ValueError("mosaic source and location registry order disagrees")
        if projection_digest != location_registry.projection_digest:
            raise ValueError("mosaic projection and grid contract disagree")
        return cls(
            source_radar_registry_digest=source_registry.registry_digest,
            radar_site_location_registry_digest=location_registry.registry_digest,
            source_selection_policy_digest=(
                source_registry.source_selection_policy_digest
            ),
            radar_site_digests=source_registry.radar_site_digests,
            radar_site_location_digests=(
                location_registry.radar_site_location_digests
            ),
            radar_projected_xy_m=location_registry.radar_projected_xy_m,
            grid_contract_digest=grid_contract_digest,
            projection_digest=projection_digest,
            range_regime_labels=range_regime_labels,
            radial_distance_edges_m=radial_distance_edges_m,
            horizontal_range_rule_digest=horizontal_range_rule_digest,
            grid_x_m_digest=grid_x_m_digest,
            grid_y_m_digest=grid_y_m_digest,
            resolver_algorithm=resolver_algorithm,
            contract=contract,
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "contract_digest"
        }

    def validate_integrity(self) -> None:
        if self.contract_digest != json_digest(self.payload):
            raise ValueError("mosaic range geometry contract digest mismatch")


@dataclass(frozen=True)
class RangePartitionEvidence:
    """Resolved non-overlapping masks for one immutable range geometry."""

    range_geometry_contract_digest: str
    grid_contract_digest: str
    range_regime_labels: tuple[str, ...]
    masks: tuple[Tensor, ...]
    valid_range_domain_mask: Tensor
    range_band_mask_digests: tuple[str, ...]
    valid_range_domain_mask_digest: str
    active_range_regimes: tuple[str, ...]
    contract: str = "radar-range-partition-evidence-v4"
    evidence_digest: str = field(init=False)

    def _validate_content(self, masks: tuple[Tensor, ...]) -> None:
        _require_digest("range geometry contract", self.range_geometry_contract_digest)
        _require_digest("range partition grid", self.grid_contract_digest)
        if (
            self.contract != "radar-range-partition-evidence-v4"
            or not masks
            or len(masks) != len(self.range_regime_labels)
            or not self.range_regime_labels
            or len(set(self.range_regime_labels))
            != len(self.range_regime_labels)
            or any(not label for label in self.range_regime_labels)
            or len(self.range_band_mask_digests) != len(masks)
            or any(mask.ndim != 2 or mask.dtype is not torch.bool for mask in masks)
            or any(mask.shape != masks[0].shape for mask in masks[1:])
            or self.valid_range_domain_mask.dtype is not torch.bool
            or self.valid_range_domain_mask.ndim != 2
            or self.valid_range_domain_mask.shape != masks[0].shape
            or self.valid_range_domain_mask.device != masks[0].device
            or self.valid_range_domain_mask_digest
            != tensor_digest(self.valid_range_domain_mask)
            or any(
                digest != tensor_digest(mask)
                for digest, mask in zip(
                    self.range_band_mask_digests,
                    masks,
                    strict=True,
                )
            )
            or self.active_range_regimes
            != tuple(
                label
                for label, mask in zip(self.range_regime_labels, masks, strict=True)
                if bool(torch.any(mask))
            )
        ):
            raise ValueError("range partition evidence is invalid")
        membership = torch.stack(
            tuple(mask.to(torch.int8) for mask in masks)
        ).sum(dim=0)
        if not torch.equal(
            membership,
            self.valid_range_domain_mask.to(torch.int8),
        ):
            raise ValueError("range partition is incomplete")

    def __post_init__(self) -> None:
        masks = tuple(mask.detach().clone() for mask in self.masks)
        domain = self.valid_range_domain_mask.detach().clone()
        object.__setattr__(self, "valid_range_domain_mask", domain)
        self._validate_content(masks)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "evidence_digest", json_digest(self.payload))

    def validate_integrity(self) -> None:
        self._validate_content(tuple(self.masks))
        if self.evidence_digest != json_digest(self.payload):
            raise ValueError("range partition evidence digest mismatch")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "range_geometry_contract_digest": self.range_geometry_contract_digest,
            "grid_contract_digest": self.grid_contract_digest,
            "range_regime_labels": list(self.range_regime_labels),
            "range_band_mask_digests": list(self.range_band_mask_digests),
            "valid_range_domain_mask_digest": self.valid_range_domain_mask_digest,
            "active_range_regimes": list(self.active_range_regimes),
            "grid_shape": list(self.masks[0].shape),
        }

    def mask(self, range_regime: str) -> Tensor:
        try:
            index = self.range_regime_labels.index(range_regime)
        except ValueError as error:
            raise ValueError("range regime is outside the physical geometry") from error
        return self.masks[index]


def resolve_range_geometry(
    contract: RangeGeometryContract,
    *,
    grid_x_m: Tensor,
    grid_y_m: Tensor,
) -> RangePartitionEvidence:
    """Resolve range bands from projected coordinates without radar-frame values."""

    x = grid_x_m.detach().clone()
    y = grid_y_m.detach().clone()
    if (
        x.ndim != 2
        or y.shape != x.shape
        or not x.is_floating_point()
        or not y.is_floating_point()
        or not bool(torch.all(torch.isfinite(x)))
        or not bool(torch.all(torch.isfinite(y)))
        or tensor_digest(x) != contract.grid_x_m_digest
        or tensor_digest(y) != contract.grid_y_m_digest
    ):
        raise ValueError("range geometry coordinates disagree with their contract")
    distance = torch.sqrt(
        (x.to(torch.float64) - contract.radar_x_m).square()
        + (y.to(torch.float64) - contract.radar_y_m).square()
    )
    edges = contract.radial_distance_edges_m
    if bool(torch.any(distance > edges[-1])):
        raise ValueError("range geometry does not cover the operational grid")
    masks = tuple(
        (distance >= lower)
        & (
            distance <= upper
            if index == len(contract.range_regime_labels) - 1
            else distance < upper
        )
        for index, (lower, upper) in enumerate(zip(edges, edges[1:]))
    )
    valid_domain = torch.ones_like(distance, dtype=torch.bool)
    return RangePartitionEvidence(
        range_geometry_contract_digest=contract.contract_digest,
        grid_contract_digest=contract.grid_contract_digest,
        range_regime_labels=contract.range_regime_labels,
        masks=masks,
        valid_range_domain_mask=valid_domain,
        range_band_mask_digests=tuple(tensor_digest(mask) for mask in masks),
        valid_range_domain_mask_digest=tensor_digest(valid_domain),
        active_range_regimes=tuple(
            label
            for label, mask in zip(contract.range_regime_labels, masks, strict=True)
            if bool(torch.any(mask))
        ),
    )


def resolve_mosaic_range_geometry(
    contract: MosaicRangeGeometryContract,
    *,
    grid_x_m: Tensor,
    grid_y_m: Tensor,
    source_radar_index_map: Tensor,
) -> tuple[RangePartitionEvidence, Tensor]:
    """Resolve bands from each cell's actual registered source radar."""

    x = grid_x_m.detach().clone()
    y = grid_y_m.detach().clone()
    source = source_radar_index_map.detach().clone()
    contract.validate_integrity()
    if (
        x.ndim != 2
        or y.shape != x.shape
        or source.ndim != 2
        or source.shape != x.shape
        or not x.is_floating_point()
        or not y.is_floating_point()
        or source.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        or not bool(torch.all(torch.isfinite(x)))
        or not bool(torch.all(torch.isfinite(y)))
        or tensor_digest(x) != contract.grid_x_m_digest
        or tensor_digest(y) != contract.grid_y_m_digest
        or bool(
            torch.any(
                (source < -1) | (source >= len(contract.radar_site_digests))
            )
        )
    ):
        raise ValueError("mosaic range geometry inputs disagree with their contract")
    site_xy = torch.tensor(
        contract.radar_projected_xy_m,
        dtype=torch.float64,
        device=x.device,
    )
    valid_source = source >= 0
    source_long = source.clamp_min(0).to(torch.long)
    source_x = site_xy[:, 0][source_long]
    source_y = site_xy[:, 1][source_long]
    resolved_distance = torch.sqrt(
        (x.to(torch.float64) - source_x).square()
        + (y.to(torch.float64) - source_y).square()
    )
    # A zero sentinel keeps the archived tensor finite, while valid_source
    # separately determines statistical range-band membership.
    distance = torch.where(
        valid_source,
        resolved_distance,
        torch.zeros_like(resolved_distance),
    )
    if bool(torch.any(distance > contract.radial_distance_edges_m[-1])):
        raise ValueError("mosaic effective range disagrees with its contract")
    masks = tuple(
        valid_source
        & (distance >= lower)
        & (
            distance <= upper
            if index == len(contract.range_regime_labels) - 1
            else distance < upper
        )
        for index, (lower, upper) in enumerate(
            zip(
                contract.radial_distance_edges_m,
                contract.radial_distance_edges_m[1:],
            )
        )
    )
    return (
        RangePartitionEvidence(
            range_geometry_contract_digest=contract.contract_digest,
            grid_contract_digest=contract.grid_contract_digest,
            range_regime_labels=contract.range_regime_labels,
            masks=masks,
            valid_range_domain_mask=valid_source,
            range_band_mask_digests=tuple(tensor_digest(mask) for mask in masks),
            valid_range_domain_mask_digest=tensor_digest(valid_source),
            active_range_regimes=tuple(
                label
                for label, mask in zip(
                    contract.range_regime_labels, masks, strict=True
                )
                if bool(torch.any(mask))
            ),
        ),
        distance,
    )


def restrict_range_partition_domain(
    partition: RangePartitionEvidence,
    *,
    valid_range_domain_mask: Tensor,
) -> RangePartitionEvidence:
    """Restrict physical range bands to the currently usable source domain."""

    partition.validate_integrity()
    domain = valid_range_domain_mask.detach().clone()
    if (
        domain.dtype is not torch.bool
        or domain.shape != partition.valid_range_domain_mask.shape
        or domain.device != partition.valid_range_domain_mask.device
        or bool(torch.any(domain & ~partition.valid_range_domain_mask))
    ):
        raise ValueError("range partition restriction is outside its source domain")
    masks = tuple(mask & domain for mask in partition.masks)
    return RangePartitionEvidence(
        range_geometry_contract_digest=partition.range_geometry_contract_digest,
        grid_contract_digest=partition.grid_contract_digest,
        range_regime_labels=partition.range_regime_labels,
        masks=masks,
        valid_range_domain_mask=domain,
        range_band_mask_digests=tuple(tensor_digest(mask) for mask in masks),
        valid_range_domain_mask_digest=tensor_digest(domain),
        active_range_regimes=tuple(
            label
            for label, mask in zip(
                partition.range_regime_labels,
                masks,
                strict=True,
            )
            if bool(torch.any(mask))
        ),
        contract="radar-range-partition-evidence-v4",
    )
