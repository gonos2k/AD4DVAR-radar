"""Restartable, content-addressed P1 frozen-linearization artifacts."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, BinaryIO, cast
import zipfile

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from ._digest import json_digest
from ._runtime import numerical_runtime_identity_digest
from .calibration import algorithm_bundle_digest
from .nowcast import (
    DataStatus,
    DynamicsSource,
    ForecastMetadata,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    StatePathProvenance,
    TendencyPairSelection,
    TendencySource,
)
from .physics import RemapCell
from .variational import (
    AnalysisConfig,
    AnalysisFeasibilityMargins,
    AnalysisLinearization,
    AnalysisObservations,
    AnalysisResult,
    FrozenObservationWhitener,
    FrozenOuterState,
    P1LinearizationState,
    _analysis_trajectory,
    _linearization_stationarity,
    validate_analysis_linearization_content,
)


P1_LINEARIZATION_ARTIFACT_VERSION = "p1-linearization-v7"
DEFAULT_MAXIMUM_MEMBER_COUNT = 96
DEFAULT_MAXIMUM_MEMBER_BYTES = 2 * 1024**3
DEFAULT_MAXIMUM_TOTAL_EXPANDED_BYTES = 8 * 1024**3
_TENSOR_NAME = re.compile(r"tensor_[0-9]{5}")

_DATACLASS_TYPES = {
    value.__name__: value
    for value in (
        AnalysisConfig,
        AnalysisFeasibilityMargins,
        AnalysisLinearization,
        AnalysisObservations,
        ForecastMetadata,
        FrozenObservationWhitener,
        FrozenOuterState,
        NowcastConfig,
        P1LinearizationState,
        RadarGridTimeContract,
        RadarState,
        RemapCell,
        StatePathProvenance,
    )
}
_ENUM_TYPES = {
    value.__name__: value
    for value in (
        DataStatus,
        DynamicsSource,
        TendencyPairSelection,
        TendencySource,
    )
}


def _array_digest(value: NDArray[Any]) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _encode_value(
    value: Any,
    arrays: dict[str, NDArray[Any]],
) -> Any:
    if isinstance(value, Tensor):
        name = f"tensor_{len(arrays):05d}"
        try:
            array = np.array(value.detach().contiguous().cpu().numpy(), copy=True)
        except TypeError as error:
            raise ValueError(
                f"unsupported Tensor dtype in P1 artifact: {value.dtype}"
            ) from error
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise ValueError("P1 artifact tensors require plain dtypes")
        arrays[name] = array
        return {
            "kind": "tensor",
            "name": name,
            "torch_dtype": str(value.dtype),
            "shape": list(value.shape),
            "source_device": str(value.device),
            "digest": _array_digest(array),
        }
    if isinstance(value, Enum):
        return {
            "kind": "enum",
            "type": type(value).__name__,
            "value": value.value,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": type(value).__name__,
            "fields": {
                field.name: _encode_value(getattr(value, field.name), arrays)
                for field in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_value(item, arrays) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_encode_value(item, arrays) for item in value],
        }
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("P1 artifact mappings require string keys")
        return {
            "kind": "dict",
            "items": {
                key: _encode_value(item, arrays)
                for key, item in sorted(value.items())
            },
        }
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            marker = "nan"
        elif value > 0:
            marker = "+inf"
        else:
            marker = "-inf"
        return {"kind": "nonfinite_float", "value": marker}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported P1 artifact value: {type(value).__name__}")


def _decode_value(
    value: Any,
    arrays: dict[str, NDArray[Any]],
    *,
    map_location: torch.device,
) -> Any:
    if not isinstance(value, dict) or "kind" not in value:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise ValueError("invalid P1 artifact value")
    kind = value.get("kind")
    if kind == "tensor":
        name = value.get("name")
        if not isinstance(name, str) or not _TENSOR_NAME.fullmatch(name):
            raise ValueError("invalid P1 artifact tensor name")
        if name not in arrays:
            raise ValueError("P1 artifact tensor is missing")
        array = arrays[name]
        if list(array.shape) != value.get("shape"):
            raise ValueError("P1 artifact tensor shape mismatch")
        if _array_digest(array) != value.get("digest"):
            raise ValueError("P1 artifact tensor digest mismatch")
        tensor = torch.from_numpy(np.array(array, copy=True))
        if str(tensor.dtype) != value.get("torch_dtype"):
            raise ValueError("P1 artifact tensor dtype mismatch")
        return tensor.to(device=map_location)
    if kind == "enum":
        type_name = value.get("type")
        if not isinstance(type_name, str):
            raise ValueError("unsupported P1 artifact enum")
        enum_type = _ENUM_TYPES.get(type_name)
        if enum_type is None:
            raise ValueError("unsupported P1 artifact enum")
        try:
            return enum_type(value.get("value"))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid P1 artifact enum value") from error
    if kind == "dataclass":
        type_name = value.get("type")
        if not isinstance(type_name, str):
            raise ValueError("unsupported P1 artifact dataclass")
        data_type = _DATACLASS_TYPES.get(type_name)
        raw_fields = value.get("fields")
        if data_type is None or not isinstance(raw_fields, dict):
            raise ValueError("unsupported P1 artifact dataclass")
        expected = {field.name for field in fields(data_type)}
        if set(raw_fields) != expected:
            raise ValueError("P1 artifact dataclass fields mismatch")
        decoded = {
            name: _decode_value(item, arrays, map_location=map_location)
            for name, item in raw_fields.items()
        }
        try:
            return data_type(**decoded)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid P1 artifact dataclass") from error
    if kind in ("tuple", "list"):
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("invalid P1 artifact sequence")
        decoded_items = [
            _decode_value(item, arrays, map_location=map_location)
            for item in items
        ]
        return tuple(decoded_items) if kind == "tuple" else decoded_items
    if kind == "dict":
        items = value.get("items")
        if not isinstance(items, dict):
            raise ValueError("invalid P1 artifact mapping")
        return {
            name: _decode_value(item, arrays, map_location=map_location)
            for name, item in items.items()
        }
    if kind == "nonfinite_float":
        marker = value.get("value")
        if marker == "nan":
            return math.nan
        if marker == "+inf":
            return math.inf
        if marker == "-inf":
            return -math.inf
        raise ValueError("invalid non-finite P1 artifact value")
    raise ValueError("invalid P1 artifact value kind")


def _artifact_digest(
    payload_json: str,
    arrays: dict[str, NDArray[Any]],
) -> str:
    return json_digest(
        {
            "version": P1_LINEARIZATION_ARTIFACT_VERSION,
            "payload_sha256": hashlib.sha256(
                payload_json.encode("utf-8")
            ).hexdigest(),
            "arrays": [
                {
                    "name": name,
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "digest": _array_digest(value),
                }
                for name, value in sorted(arrays.items())
            ],
        }
    )


def _linearization_state(
    analysis: AnalysisResult | P1LinearizationState,
) -> P1LinearizationState:
    if analysis.used_fallback or not analysis.converged or analysis.degraded:
        raise ValueError(
            "P1 linearization artifact requires a converged accepted analysis"
        )
    if (
        not analysis.final_linearization_stationary
        or not analysis.fso_eligible
    ):
        raise ValueError(
            "P1 linearization artifact requires an FSO-eligible stationary "
            "analysis"
        )
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("P1 analysis does not retain a final linearization")
    validate_analysis_linearization_content(analysis.control, linearization)
    diagnostics = (
        analysis.linearization_residual_norm,
        analysis.linearization_gradient_norm,
        analysis.linearization_relative_stationarity,
        analysis.linearization_polish_iterations,
    )
    retained = (
        linearization.residual_norm,
        linearization.gradient_norm,
        linearization.relative_stationarity,
        linearization.polish_iterations,
    )
    if diagnostics != retained:
        raise ValueError("P1 analysis linearization diagnostics mismatch")
    if linearization.forecast_run_digest is None:
        raise ValueError("P1 linearization must be bound to a forecast run")
    return P1LinearizationState(
        control=analysis.control.detach().clone(),
        active_field_index=analysis.active_field_index.detach().clone(),
        state=RadarState(
            echo_linear=analysis.state.echo_linear.detach().clone(),
            displacement_yx=analysis.state.displacement_yx.detach().clone(),
            log_growth_per_step=(
                analysis.state.log_growth_per_step.detach().clone()
            ),
        ),
        linearization_residual_norm=linearization.residual_norm,
        linearization_gradient_norm=linearization.gradient_norm,
        linearization_relative_stationarity=(
            linearization.relative_stationarity
        ),
        linearization_polish_iterations=linearization.polish_iterations,
        linearization=linearization,
        final_linearization_stationary=True,
        fso_eligible=True,
        outer_converged=analysis.outer_converged,
    )


def save_p1_linearization(
    analysis: AnalysisResult | P1LinearizationState,
    path: str | Path,
) -> None:
    """Atomically save a replayable accepted P1 final linearization."""

    state = _linearization_state(analysis)
    arrays: dict[str, NDArray[Any]] = {}
    payload = {
        "version": P1_LINEARIZATION_ARTIFACT_VERSION,
        "algorithm_bundle_digest": state.linearization.algorithm_bundle_digest,
        "numerical_runtime_digest": numerical_runtime_identity_digest(),
        "linearization_digest": state.linearization.linearization_digest,
        "state": _encode_value(state, arrays),
    }
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    artifact_digest = _artifact_digest(payload_json, arrays)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            archive_values: dict[str, Any] = {
                "artifact_version": np.asarray(
                    P1_LINEARIZATION_ARTIFACT_VERSION
                ),
                "artifact_digest": np.asarray(artifact_digest),
                "payload_json": np.asarray(payload_json),
                **arrays,
            }
            np.savez(
                stream,
                **archive_values,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
        directory_descriptor = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _preflight_archive(
    artifact: BinaryIO,
    *,
    maximum_member_count: int,
    maximum_member_bytes: int,
    maximum_total_expanded_bytes: int,
) -> None:
    limits = (
        maximum_member_count,
        maximum_member_bytes,
        maximum_total_expanded_bytes,
    )
    if any(type(value) is not int or value <= 0 for value in limits):
        raise ValueError("P1 linearization archive limits must be positive")
    try:
        with zipfile.ZipFile(artifact) as archive:
            members = archive.infolist()
            if len(members) > maximum_member_count:
                raise ValueError("P1 linearization archive has too many members")
            names: set[str] = set()
            total = 0
            declared_total = 0
            for member in members:
                if member.is_dir() or not member.filename.endswith(".npy"):
                    raise ValueError("invalid P1 linearization archive member")
                name = member.filename[:-4]
                if name in names:
                    raise ValueError("duplicate P1 linearization archive member")
                if name not in {
                    "artifact_version",
                    "artifact_digest",
                    "payload_json",
                } and not _TENSOR_NAME.fullmatch(name):
                    raise ValueError("unknown P1 linearization archive member")
                if member.file_size > maximum_member_bytes:
                    raise ValueError("P1 linearization archive member is too large")
                total += member.file_size
                if total > maximum_total_expanded_bytes:
                    raise ValueError("P1 linearization archive is too large")
                declared_total += _declared_array_bytes(
                    archive,
                    member,
                    maximum_member_bytes=maximum_member_bytes,
                )
                if declared_total > maximum_total_expanded_bytes:
                    raise ValueError(
                        "P1 linearization arrays exceed their size limit"
                    )
                names.add(name)
            if not {
                "artifact_version",
                "artifact_digest",
                "payload_json",
            }.issubset(names):
                raise ValueError("P1 linearization archive is incomplete")
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("invalid P1 linearization archive") from error


def _open_preflighted_artifact(
    path: Path,
    *,
    maximum_member_count: int,
    maximum_member_bytes: int,
    maximum_total_expanded_bytes: int,
) -> BinaryIO:
    artifact = path.open("rb")
    try:
        _preflight_archive(
            artifact,
            maximum_member_count=maximum_member_count,
            maximum_member_bytes=maximum_member_bytes,
            maximum_total_expanded_bytes=maximum_total_expanded_bytes,
        )
        artifact.seek(0)
        return artifact
    except BaseException:
        artifact.close()
        raise


def _declared_array_bytes(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    maximum_member_bytes: int,
) -> int:
    try:
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(
                    f"unsupported P1 artifact NPY version: {version}"
                )
            if dtype.hasobject or dtype.fields is not None:
                raise ValueError("P1 artifact arrays require plain dtypes")
            array_bytes = math.prod(shape) * dtype.itemsize
            if array_bytes > maximum_member_bytes:
                raise ValueError("P1 artifact array declares too much data")
            if member.file_size - stream.tell() != array_bytes:
                raise ValueError(
                    "P1 artifact array payload disagrees with its header"
                )
            return array_bytes
    except ValueError:
        raise
    except (OSError, EOFError) as error:
        raise ValueError("invalid P1 artifact array header") from error


def _referenced_tensor_names(value: Any) -> set[str]:
    if isinstance(value, list):
        names: set[str] = set()
        for item in value:
            names.update(_referenced_tensor_names(item))
        return names
    if not isinstance(value, dict):
        return set()
    if value.get("kind") == "tensor":
        name = value.get("name")
        return {name} if isinstance(name, str) else set()
    names = set()
    for item in value.values():
        names.update(_referenced_tensor_names(item))
    return names


def _scalar_string(arrays: dict[str, NDArray[Any]], name: str) -> str:
    value = arrays.get(name)
    if value is None or value.ndim != 0 or value.dtype.kind not in ("U", "S"):
        raise ValueError(f"P1 linearization {name} must be a string scalar")
    return str(value.item())


def _validate_loaded_state(state: P1LinearizationState) -> None:
    linearization = state.linearization
    validate_analysis_linearization_content(state.control, linearization)
    if not torch.equal(
        state.active_field_index,
        linearization.frozen.active_field_index,
    ):
        raise ValueError("P1 artifact active controls mismatch")
    trajectory = _analysis_trajectory(state.control, linearization.frozen)
    tolerance = linearization.frozen.nowcast_config.contract_absolute_tolerance
    expected_state = (
        trajectory.frames_linear[-1],
        trajectory.displacement_yx,
        trajectory.log_growth_per_step,
    )
    actual_state = (
        state.state.echo_linear,
        state.state.displacement_yx,
        state.state.log_growth_per_step,
    )
    if any(
        not torch.allclose(actual, expected, rtol=0.0, atol=tolerance)
        for actual, expected in zip(actual_state, expected_state, strict=True)
    ):
        raise ValueError("P1 artifact does not reproduce its analysis state")
    stationarity = _linearization_stationarity(
        state.control,
        linearization.observations,
        linearization.frozen,
    )
    stored = (
        linearization.residual_norm,
        linearization.gradient_norm,
        linearization.relative_stationarity,
    )
    actual = (
        stationarity.residual_norm,
        stationarity.gradient_norm,
        stationarity.relative_stationarity,
    )
    comparison_tolerance = 64.0 * torch.finfo(state.control.dtype).eps
    if any(
        not math.isclose(
            left,
            right,
            rel_tol=comparison_tolerance,
            abs_tol=comparison_tolerance,
        )
        for left, right in zip(stored, actual, strict=True)
    ):
        raise ValueError("P1 artifact stationarity diagnostics mismatch")
    if (
        stationarity.relative_stationarity
        > linearization.frozen.analysis_config
        .final_linearization_relative_stationarity_tolerance
    ):
        raise ValueError("P1 artifact final linearization is not stationary")


def load_p1_linearization(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    maximum_member_count: int = DEFAULT_MAXIMUM_MEMBER_COUNT,
    maximum_member_bytes: int = DEFAULT_MAXIMUM_MEMBER_BYTES,
    maximum_total_expanded_bytes: int = (
        DEFAULT_MAXIMUM_TOTAL_EXPANDED_BYTES
    ),
) -> P1LinearizationState:
    """Load and independently replay-validate a P1 linearization artifact."""

    source = Path(path)
    artifact = _open_preflighted_artifact(
        source,
        maximum_member_count=maximum_member_count,
        maximum_member_bytes=maximum_member_bytes,
        maximum_total_expanded_bytes=maximum_total_expanded_bytes,
    )
    try:
        with artifact, np.load(artifact, allow_pickle=False) as archive:
            loaded = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("invalid P1 linearization archive") from error
    version = _scalar_string(loaded, "artifact_version")
    if version != P1_LINEARIZATION_ARTIFACT_VERSION:
        raise ValueError(f"unsupported P1 linearization artifact: {version}")
    payload_json = _scalar_string(loaded, "payload_json")
    stored_artifact_digest = _scalar_string(loaded, "artifact_digest")
    tensor_arrays = {
        name: value
        for name, value in loaded.items()
        if _TENSOR_NAME.fullmatch(name)
    }
    if _artifact_digest(payload_json, tensor_arrays) != stored_artifact_digest:
        raise ValueError("P1 linearization artifact digest mismatch")
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid P1 linearization payload") from error
    if not isinstance(payload, dict) or payload.get("version") != version:
        raise ValueError("invalid P1 linearization payload contract")
    referenced_names = _referenced_tensor_names(payload.get("state"))
    if referenced_names != set(tensor_arrays):
        raise ValueError("P1 artifact tensor membership mismatch")
    if payload.get("algorithm_bundle_digest") != algorithm_bundle_digest():
        raise ValueError("P1 artifact algorithm bundle mismatch")
    if (
        payload.get("numerical_runtime_digest")
        != numerical_runtime_identity_digest()
    ):
        raise ValueError("P1 artifact numerical runtime mismatch")
    decoded = _decode_value(
        payload.get("state"),
        tensor_arrays,
        map_location=torch.device(map_location),
    )
    if not isinstance(decoded, P1LinearizationState):
        raise ValueError("P1 artifact does not contain a linearization state")
    if (
        payload.get("linearization_digest")
        != decoded.linearization.linearization_digest
    ):
        raise ValueError("P1 artifact linearization digest mismatch")
    _validate_loaded_state(decoded)
    return cast(P1LinearizationState, decoded)
