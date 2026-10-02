"""A local chart that uses one physical FV face flux as a coordinate.

The chart changes coordinates only. It leaves the complete field, flow, and
growth vector available to the caller, so the same objective can be evaluated
in either representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import torch
from torch import Tensor

from advar.transport import face_volume_fluxes


def face_weights_from_basis(psi_basis: Tensor, axis: str, face: tuple[int, int]) -> Tensor:
    """Return one oriented face-flux weight per production streamfunction basis."""
    if (not isinstance(psi_basis, Tensor) or psi_basis.device.type != "cpu"
            or psi_basis.dtype != torch.float64 or psi_basis.ndim != 3
            or psi_basis.shape[0] < 1 or min(psi_basis.shape[1:]) < 2
            or psi_basis.requires_grad or not bool(torch.isfinite(psi_basis).all())):
        raise ValueError("psi_basis must be fixed finite CPU FP64 [K,H+1,W+1]")
    if axis not in ("x", "y") or not isinstance(face, tuple) or len(face) != 2:
        raise ValueError("face must select an x or y face by (row, column)")
    row, column = face
    if type(row) is not int or type(column) is not int:
        raise ValueError("face indices must be integers")
    h, w = psi_basis.shape[1] - 1, psi_basis.shape[2] - 1
    limits = (h, w + 1) if axis == "x" else (h + 1, w)
    if not (0 <= row < limits[0] and 0 <= column < limits[1]):
        raise ValueError("selected face is outside the production basis grid")
    components = []
    for basis in psi_basis:
        qx, qy = face_volume_fluxes(basis)
        components.append(qx[row, column] if axis == "x" else qy[row, column])
    weights = torch.stack(components)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("selected face weights must be finite")
    return weights.clone()


@dataclass(frozen=True, init=False)
class FVFaceFluxCoordinateChart:
    """Replace one flow latent by its normalized signed physical face flux.

    ``field_count`` and ``growth_count`` pin the unchanged prefix and suffix
    layout. ``weights`` are the selected face's linear weights from
    :func:`face_volume_fluxes`; ``limits`` are the production coefficient
    limits. The chart's normalization is ``abs(weights[p]) * limits[p]``.
    This defines a coordinate scale, not a Euclidean metric on the full
    control vector. The linear face functional is equivalent in real
    arithmetic; FP64 reduction order can differ from the production combined
    streamfunction, and this chart supplies no event or branch certificate.
    A nonzero pivot contribution below 64 FP64 epsilons of the sum of absolute
    nonpivot contributions is outside the numerically representable forward
    chart domain. This guard is conservative: opposing nonpivot terms may cancel
    in this chart's reduction while a production combined-streamfunction
    reduction loses the pivot contribution.
    """

    field_count: int
    growth_count: int
    pivot_index: int
    _limits_bytes: bytes = field(repr=False)
    _weights_bytes: bytes = field(repr=False)
    _flow_count: int = field(repr=False)
    _face: tuple[str, int, int] | None = field(repr=False)

    def __init__(
        self,
        field_count: int,
        growth_count: int,
        pivot_index: int,
        limits: Tensor,
        weights: Tensor,
        face: tuple[str, int, int] | None = None,
    ) -> None:
        if type(field_count) is not int or field_count < 0:
            raise ValueError("field_count must be a non-negative integer")
        if type(growth_count) is not int or growth_count < 0:
            raise ValueError("growth_count must be a non-negative integer")
        if (not isinstance(limits, Tensor) or not isinstance(weights, Tensor)
                or limits.device.type != "cpu" or weights.device.type != "cpu"
                or limits.dtype != torch.float64 or weights.dtype != torch.float64
                or limits.ndim != 1 or limits.numel() < 1 or weights.shape != limits.shape
                or limits.requires_grad or weights.requires_grad
                or not bool(torch.isfinite(limits).all()) or not bool(torch.isfinite(weights).all())
                or bool((limits <= 0).any())):
            raise ValueError("limits and weights must be fixed finite CPU FP64 vectors")
        if type(pivot_index) is not int or not 0 <= pivot_index < limits.numel():
            raise ValueError("pivot_index must select a flow coefficient")
        if not bool(weights[pivot_index] != 0):
            raise ValueError("pivot face weight must be nonzero")
        if (face is not None and (not isinstance(face, tuple) or len(face) != 3
                or face[0] not in ("x", "y")
                or type(face[1]) is not int or type(face[2]) is not int
                or face[1] < 0 or face[2] < 0)):
            raise ValueError("face metadata must be None or (axis, non-negative row, column)")
        scale = weights[pivot_index].abs() * limits[pivot_index]
        if not bool(torch.isfinite(scale) & (scale > 0)):
            raise ValueError("pivot face normalization must be finite and positive")
        coefficient_products = weights * limits
        if not bool(torch.isfinite(coefficient_products).all()):
            raise ValueError("face weight and coefficient-limit products must be finite")
        normalized_row = coefficient_products / scale
        row_abs_bound = normalized_row.abs().sum()
        if not bool(torch.isfinite(normalized_row).all() & torch.isfinite(row_abs_bound)):
            raise ValueError("normalized face row and absolute bound must be finite")
        # Store immutable bytes rather than exposing mutable tensor storage.
        object.__setattr__(self, "field_count", field_count)
        object.__setattr__(self, "growth_count", growth_count)
        object.__setattr__(self, "pivot_index", pivot_index)
        object.__setattr__(self, "_limits_bytes", limits.contiguous().numpy().tobytes())
        object.__setattr__(self, "_weights_bytes", weights.contiguous().numpy().tobytes())
        object.__setattr__(self, "_flow_count", limits.numel())
        object.__setattr__(self, "_face", face)

    @staticmethod
    def _tensor_from_bytes(values: bytes) -> Tensor:
        return torch.frombuffer(bytearray(values), dtype=torch.float64).clone()

    @property
    def _limits(self) -> Tensor:
        return self._tensor_from_bytes(self._limits_bytes)

    @property
    def _weights(self) -> Tensor:
        return self._tensor_from_bytes(self._weights_bytes)

    @classmethod
    def from_basis(
        cls,
        *,
        field_count: int,
        growth_count: int,
        coefficient_limits: Tensor,
        psi_basis: Tensor,
        axis: str,
        face: tuple[int, int],
        pivot_index: int,
    ) -> "FVFaceFluxCoordinateChart":
        weights = face_weights_from_basis(psi_basis, axis, face)
        return cls(field_count, growth_count, pivot_index,
                   coefficient_limits.clone(), weights,
                   (axis, face[0], face[1]))

    @property
    def flow_count(self) -> int:
        return self._flow_count

    @property
    def control_count(self) -> int:
        return self.field_count + self.flow_count + self.growth_count

    @property
    def limits(self) -> Tensor:
        return self._limits.clone()

    @property
    def weights(self) -> Tensor:
        return self._weights.clone()

    @property
    def face_scale(self) -> Tensor:
        return (self._weights[self.pivot_index].abs()
                * self._limits[self.pivot_index]).clone()

    @property
    def normalized_weights(self) -> Tensor:
        return self._weights * self._limits / self.face_scale

    @property
    def layout(self) -> dict[str, object]:
        pivot = self.field_count + self.pivot_index
        growth_start = self.field_count + self.flow_count
        return {
            "field": (0, self.field_count),
            "flow": (self.field_count, growth_start),
            "face_flux_coordinate": pivot,
            "unchanged_flow_coordinates": tuple(
                i for i in range(self.field_count, growth_start) if i != pivot
            ),
            "growth": (growth_start, self.control_count),
            "control_count": self.control_count,
        }

    @property
    def identity(self) -> dict[str, object]:
        digest = hashlib.sha256()
        for fixed_bytes in (self._limits_bytes, self._weights_bytes):
            digest.update(str(torch.float64).encode())
            digest.update(json.dumps((self.flow_count,)).encode())
            digest.update(fixed_bytes)
        digest.update(json.dumps((self.field_count, self.growth_count,
                                  self.pivot_index, self._face)).encode())
        return {
            "chart_sha256": digest.hexdigest(),
            "face": self._face,
            "pivot_flow_index": self.pivot_index,
            "normalization": "abs(pivot face weight) * pivot coefficient limit",
            "metric": (
                "normalized signed face flux in real arithmetic; FP64 reduction order may differ; "
                "no event or branch certificate"
            ),
        }

    def to_face_coordinates(self, original: Tensor) -> Tensor:
        """Map the original complete control vector to the face-flux chart."""
        self._check_control(original)
        torch._assert_async(torch.isfinite(original).all(), "original controls must be finite")
        start = self.field_count
        flows = original[start:start + self.flow_count]
        normalized_weights = self.normalized_weights
        fractions = torch.tanh(flows)
        pivot = start + self.pivot_index
        other_mask = torch.arange(self.flow_count) != self.pivot_index
        other_terms = normalized_weights[other_mask] * fractions[other_mask]
        other_flux = other_terms.sum()
        other_abs_sum = other_terms.abs().sum()
        pivot_flux = normalized_weights[self.pivot_index] * fractions[self.pivot_index]
        eta = other_flux + pivot_flux
        fraction = (eta - other_flux) / normalized_weights[self.pivot_index]
        recovered = torch.atanh(fraction)
        source = flows[self.pivot_index]
        epsilon = torch.finfo(original.dtype).eps
        roundtrip_tolerance = 64 * epsilon * (1 + source.abs())
        resolved = ((pivot_flux == 0)
                    | (pivot_flux.abs() > 64 * epsilon * other_abs_sum))
        valid = (torch.isfinite(eta) & torch.isfinite(fraction) & torch.isfinite(recovered)
                 & (fraction > -1) & (fraction < 1)
                 & resolved
                 & ((recovered - source).abs() <= roundtrip_tolerance))
        torch._assert_async(valid, "original control is outside the strict representable chart domain")
        return torch.cat((original[:pivot], eta.reshape(1), original[pivot + 1:]))

    def from_face_coordinates(self, coordinates: Tensor) -> Tensor:
        """Recover the original controls, refusing values outside the chart."""
        self._check_control(coordinates)
        torch._assert_async(torch.isfinite(coordinates).all(), "face coordinates must be finite")
        start = self.field_count
        pivot = start + self.pivot_index
        eta = coordinates[pivot]
        flow_coordinates = coordinates[start:start + self.flow_count]
        other_mask = torch.arange(self.flow_count) != self.pivot_index
        other = flow_coordinates[other_mask]
        normalized_weights = self.normalized_weights
        other_flux = (normalized_weights[other_mask] * torch.tanh(other)).sum()
        fraction = (eta - other_flux) / normalized_weights[self.pivot_index]
        valid = torch.isfinite(eta) & torch.isfinite(fraction) & (fraction > -1) & (fraction < 1)
        torch._assert_async(valid, "face coordinate is outside the strict representable chart domain")
        recovered = torch.atanh(fraction)
        return torch.cat((coordinates[:pivot], recovered.reshape(1), coordinates[pivot + 1:]))

    def _check_control(self, value: Tensor) -> None:
        if (not isinstance(value, Tensor) or value.shape != (self.control_count,)
                or value.dtype != torch.float64 or value.device.type != "cpu"):
            raise ValueError("control must match this chart's CPU FP64 full-control layout")
