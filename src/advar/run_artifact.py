from __future__ import annotations

from dataclasses import asdict
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
from .nowcast import (
    DataStatus,
    ForecastMetadata,
    ForecastResult,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    TendencySource,
)


FORECAST_RUN_ARTIFACT_VERSION = "forecast-run-v5"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_MAXIMUM_MEMBER_COUNT = 128
DEFAULT_MAXIMUM_MEMBER_BYTES = 1024**3
DEFAULT_MAXIMUM_TOTAL_EXPANDED_BYTES = 2 * 1024**3
_CORE_ARRAY_NAMES = frozenset(
    {
        "forecast_run_artifact_version",
        "forecast_run_artifact_digest",
        "forecast_run_digest",
        "nowcast_config_json",
        "nowcast_config_digest",
        "input_bundle_digest",
        "run_background_age_minutes",
        "grid_time_contract_present",
        "grid_time_contract_json",
        "grid_time_contract_digest",
        "forecast_dbz",
        "forecast_dbz_digest",
        "valid_mask",
        "valid_mask_digest",
        "state_echo_linear",
        "displacement_yx",
        "displacement_mps_yx",
        "grid_velocity_mps_yx",
        "projected_velocity_mps_xy",
        "log_growth_per_step",
        "state_metadata_digest",
        "coverage_by_frame",
        "data_status",
        "background_used",
        "background_contribution_fraction",
        "background_state_support_fraction",
        "background_tendency_used",
        "background_age_minutes",
        "source_support",
        "motion_disagreement_px",
        "growth_disagreement",
        "minimum_phase_correlation_psr",
        "tendency_pair_count",
        "tendency_source",
        "provenance",
        "latest_frame_dbz",
        "latest_observation_mask",
        "latest_background_dbz",
        "latest_observation_mask_digest",
        "latest_frame_digest",
        "latest_background_present",
        "latest_background_digest",
        "analysis_config_present",
        "analysis_config_json",
        "analysis_config_digest",
        "analysis_input_digest",
    }
)
_CLI_EXTRA_ARRAY_NAMES = frozenset(
    {
        "output_contract_version",
        "min_publish_support",
        "lead_minutes",
        "analysis_used",
        "analysis_converged",
        "analysis_degraded",
        "analysis_used_fallback",
        "analysis_reason",
        "analysis_initial_objective",
        "analysis_final_objective",
        "analysis_outer_iterations",
        "analysis_pcg_iterations",
        "analysis_minimum_reachability_margin",
        "analysis_unresolved_amplitude_fraction",
        "analysis_unresolved_amplitude_fraction_by_time",
        "analysis_unresolved_pixel_fraction_by_time",
        "analysis_amplitude_violation_score",
        "analysis_amplitude_violation_score_by_time",
        "analysis_integrated_echo_ratio_by_time",
        "analysis_displacement_tolerant_soft_echo_area_ratio_by_time",
        "analysis_effective_precursor_pixel_count_by_time",
        "analysis_bad_quality_weight_by_time",
        "analysis_total_quality_weight_by_time",
        "analysis_amplitude_information_sufficient_by_time",
        "analysis_insufficient_amplitude_information",
        "analysis_established_echo_excess_growth_fraction",
        "analysis_established_echo_excess_growth_fraction_by_time",
        "analysis_maximum_growth_envelope_ratio",
        "analysis_maximum_growth_envelope_ratio_by_time",
        "analysis_amplitude_diagnostics_source",
        "analysis_relative_objective_reduction",
        "analysis_causal_control_cell_count",
        "analysis_causal_seed_cell_count",
        "analysis_causal_seed_prior_cost",
        "analysis_dynamics_reduced_hessian_eigenvalues",
        "analysis_dynamics_reduced_hessian_condition_number",
        "analysis_dynamics_data_gram_eigenvalues",
        "analysis_dynamics_data_information_trace",
        "analysis_dynamics_data_effective_rank",
        "analysis_regularized_dynamics_hessian_eigenvalues",
        "analysis_regularized_dynamics_hessian_condition_number",
        "analysis_field_smoothness_prior_cost",
        "analysis_motion_saturation_margin_yx",
        "analysis_motion_speed_saturation_margin_mps",
        "analysis_growth_saturation_margin",
        "analysis_field_growth_jacobian_cosine",
        "analysis_field_motion_jacobian_cosine_yx",
        "input_minimum_before_fix",
        "input_corrected_count",
        "input_corrected_integral",
        "forecast_minimum_before_fix",
        "forecast_corrected_count",
        "forecast_corrected_integral",
        "echo_integral_before_transport",
        "echo_integral_after_transport",
        "boundary_outflow_integral",
        "echo_budget_error",
        "analysis_minimum_before_fix",
        "analysis_corrected_count",
        "analysis_corrected_integral",
    }
)


def forecast_run_arrays(result: ForecastResult) -> dict[str, Any]:
    """Return the canonical NPZ payload needed to reconstruct ``result``.

    Optional positivity/transport audit objects are intentionally excluded;
    they do not participate in delayed M0 sensitivity.
    """

    result.validate_issuance()
    config = result.run.config
    config_json = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    latest_observation_mask = result.run.latest_observation_mask
    metadata = result.metadata
    grid_time_contract = result.run.grid_time_contract
    displacement_mps = result.displacement_mps_yx
    projected_velocity = result.projected_velocity_mps_xy
    arrays = {
        "forecast_run_artifact_version": np.asarray(
            FORECAST_RUN_ARTIFACT_VERSION
        ),
        "forecast_run_artifact_digest": np.asarray(""),
        "forecast_run_digest": np.asarray(result.forecast_run_digest),
        "nowcast_config_json": np.asarray(config_json),
        "nowcast_config_digest": np.asarray(config.digest),
        "input_bundle_digest": np.asarray(result.run.input_bundle_digest),
        "run_background_age_minutes": np.asarray(
            np.nan
            if result.run.background_age_minutes is None
            else result.run.background_age_minutes
        ),
        "grid_time_contract_present": np.asarray(
            grid_time_contract is not None
        ),
        "grid_time_contract_json": np.asarray(
            ""
            if grid_time_contract is None
            else json.dumps(
                asdict(grid_time_contract),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "grid_time_contract_digest": np.asarray(
            ""
            if result.run.grid_time_contract_digest is None
            else result.run.grid_time_contract_digest
        ),
        "forecast_dbz": _numpy(result.forecast_dbz),
        "forecast_dbz_digest": np.asarray(result.forecast_dbz_digest),
        "valid_mask": _numpy(result.valid_mask),
        "valid_mask_digest": np.asarray(result.valid_mask_digest),
        "state_echo_linear": _numpy(result.state.echo_linear),
        "displacement_yx": _numpy(result.state.displacement_yx),
        "displacement_mps_yx": (
            np.full(2, np.nan, dtype=np.float64)
            if displacement_mps is None
            else _numpy(displacement_mps)
        ),
        "grid_velocity_mps_yx": (
            np.full(2, np.nan, dtype=np.float64)
            if result.grid_velocity_mps_yx is None
            else _numpy(result.grid_velocity_mps_yx)
        ),
        "projected_velocity_mps_xy": (
            np.full(2, np.nan, dtype=np.float64)
            if projected_velocity is None
            else _numpy(projected_velocity)
        ),
        "log_growth_per_step": _numpy(
            result.state.log_growth_per_step
        ),
        "state_metadata_digest": np.asarray(result.state_metadata_digest),
        "coverage_by_frame": _numpy(metadata.coverage_by_frame),
        "data_status": np.asarray(metadata.data_status.value),
        "background_used": np.asarray(metadata.background_used),
        "background_contribution_fraction": np.asarray(
            metadata.background_contribution_fraction
        ),
        "background_state_support_fraction": np.asarray(
            metadata.background_state_support_fraction
        ),
        "background_tendency_used": np.asarray(
            metadata.background_tendency_used
        ),
        "background_age_minutes": np.asarray(
            np.nan
            if metadata.background_age_minutes is None
            else metadata.background_age_minutes
        ),
        "source_support": _numpy(metadata.source_support),
        "motion_disagreement_px": _numpy(
            metadata.motion_disagreement_px
        ),
        "growth_disagreement": _numpy(metadata.growth_disagreement),
        "minimum_phase_correlation_psr": _numpy(
            metadata.minimum_phase_correlation_psr
        ),
        "tendency_pair_count": np.asarray(metadata.tendency_pair_count),
        "tendency_source": np.asarray(metadata.tendency_source.value),
        "provenance": np.asarray(metadata.provenance),
        "latest_observation_mask": _numpy(latest_observation_mask),
        "latest_frame_dbz": _numpy(result.run.latest_frame_dbz),
        "latest_background_dbz": (
            np.empty((0,), dtype=np.float64)
            if result.run.latest_background_dbz is None
            else _numpy(result.run.latest_background_dbz)
        ),
        "latest_observation_mask_digest": np.asarray(
            result.run.latest_observation_mask_digest
        ),
        "latest_frame_digest": np.asarray(result.run.latest_frame_digest),
        "latest_background_present": np.asarray(
            result.run.latest_background_digest is not None
        ),
        "latest_background_digest": np.asarray(
            ""
            if result.run.latest_background_digest is None
            else result.run.latest_background_digest
        ),
        "analysis_config_present": np.asarray(
            result.run.analysis_config_json is not None
        ),
        "analysis_config_json": np.asarray(
            ""
            if result.run.analysis_config_json is None
            else result.run.analysis_config_json
        ),
        "analysis_config_digest": np.asarray(
            ""
            if result.run.analysis_config_digest is None
            else result.run.analysis_config_digest
        ),
        "analysis_input_digest": np.asarray(
            ""
            if result.run.analysis_input_digest is None
            else result.run.analysis_input_digest
        ),
    }
    return seal_forecast_run_arrays(arrays)


def save_forecast_run(result: ForecastResult, path: str | Path) -> None:
    """Atomically persist one issued forecast run as a compressed NPZ."""

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("forecast run artifact path must end with .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = forecast_run_arrays(result)
    atomic_savez_compressed(destination, arrays)


def atomic_savez_compressed(
    destination: Path,
    arrays: dict[str, Any],
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _preflight_archive(
    source: BinaryIO,
    *,
    maximum_member_count: int,
    maximum_member_bytes: int,
    maximum_total_expanded_bytes: int,
) -> None:
    limits = {
        "maximum_member_count": maximum_member_count,
        "maximum_member_bytes": maximum_member_bytes,
        "maximum_total_expanded_bytes": maximum_total_expanded_bytes,
    }
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("forecast run archive limits must be positive integers")
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if len(members) > maximum_member_count:
                raise ValueError("forecast run archive has too many members")
            raw_names = [member.filename for member in members]
            if len(set(raw_names)) != len(raw_names):
                raise ValueError(
                    "forecast run archive has duplicate members"
                )
            names: set[str] = set()
            expanded_bytes = 0
            declared_array_bytes = 0
            for member in members:
                if (
                    member.is_dir()
                    or not member.filename.endswith(".npy")
                    or "/" in member.filename
                    or "\\" in member.filename
                ):
                    raise ValueError(
                        "forecast run archive has an invalid member name"
                    )
                if member.flag_bits & 0x1:
                    raise ValueError(
                        "encrypted forecast run members are unsupported"
                    )
                if member.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                }:
                    raise ValueError(
                        "unsupported forecast run compression method"
                    )
                if member.file_size > maximum_member_bytes:
                    raise ValueError(
                        "forecast run archive member is too large"
                    )
                expanded_bytes += member.file_size
                if expanded_bytes > maximum_total_expanded_bytes:
                    raise ValueError(
                        "forecast run archive expands beyond its limit"
                    )
                declared_array_bytes += _declared_array_bytes(
                    archive,
                    member,
                    maximum_member_bytes=maximum_member_bytes,
                )
                if declared_array_bytes > maximum_total_expanded_bytes:
                    raise ValueError(
                        "forecast run arrays exceed the expanded size limit"
                    )
                names.add(member.filename[:-4])
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("invalid forecast run archive") from error
    allowed = set(_CORE_ARRAY_NAMES)
    if "output_contract_version" in names:
        allowed.update(_CLI_EXTRA_ARRAY_NAMES)
    unknown = names - allowed
    if unknown:
        raise ValueError(
            f"forecast run archive has unknown members: {sorted(unknown)}"
        )


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
                shape, _, dtype = np.lib.format.read_array_header_1_0(
                    stream
                )
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(
                    stream
                )
            else:
                raise ValueError(
                    f"unsupported NPY format version: {version}"
                )
            if dtype.hasobject or dtype.fields is not None:
                raise ValueError(
                    "forecast run arrays must use plain non-object dtypes"
                )
            array_bytes = math.prod(shape) * dtype.itemsize
            if array_bytes > maximum_member_bytes:
                raise ValueError(
                    "forecast run member declares too much array data"
                )
            payload_bytes = member.file_size - stream.tell()
            if payload_bytes != array_bytes:
                raise ValueError(
                    "forecast run member payload disagrees with its header"
                )
            return array_bytes
    except ValueError:
        raise
    except (OSError, EOFError) as error:
        raise ValueError("invalid forecast run array header") from error


def load_forecast_run(
    path: str | Path,
    *,
    maximum_member_count: int = DEFAULT_MAXIMUM_MEMBER_COUNT,
    maximum_member_bytes: int = DEFAULT_MAXIMUM_MEMBER_BYTES,
    maximum_total_expanded_bytes: int = (
        DEFAULT_MAXIMUM_TOTAL_EXPANDED_BYTES
    ),
) -> ForecastResult:
    """Load and independently validate a durable forecast run artifact."""

    source = Path(path)
    artifact = _open_preflighted_artifact(
        source,
        maximum_member_count=maximum_member_count,
        maximum_member_bytes=maximum_member_bytes,
        maximum_total_expanded_bytes=maximum_total_expanded_bytes,
    )
    with artifact, np.load(artifact, allow_pickle=False) as archive:
        version = _string_scalar(archive, "forecast_run_artifact_version")
        if version != FORECAST_RUN_ARTIFACT_VERSION:
            raise ValueError(f"unsupported forecast run artifact: {version}")

        stored_artifact_digest = _digest_scalar(
            archive,
            "forecast_run_artifact_digest",
        )
        loaded_arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
        expected_artifact_digest = _forecast_run_artifact_digest(
            loaded_arrays
        )
        if stored_artifact_digest != expected_artifact_digest:
            raise ValueError("forecast run artifact digest mismatch")

        config_json = _string_scalar(archive, "nowcast_config_json")
        try:
            config_value = json.loads(config_json)
        except json.JSONDecodeError as error:
            raise ValueError("invalid nowcast_config_json") from error
        if not isinstance(config_value, dict):
            raise ValueError("nowcast_config_json must contain an object")
        config = NowcastConfig(**cast(dict[str, Any], config_value))
        config_digest = _digest_scalar(archive, "nowcast_config_digest")
        if config.digest != config_digest:
            raise ValueError("nowcast config digest mismatch")

        forecast_dbz = _tensor(archive, "forecast_dbz")
        valid_mask = _tensor(archive, "valid_mask")
        stored_displacement_mps = _tensor(
            archive,
            "displacement_mps_yx",
        )
        stored_grid_velocity = _tensor(
            archive,
            "grid_velocity_mps_yx",
        )
        stored_projected_velocity = _tensor(
            archive,
            "projected_velocity_mps_xy",
        )
        state = RadarState(
            echo_linear=_tensor(archive, "state_echo_linear"),
            displacement_yx=_tensor(archive, "displacement_yx"),
            log_growth_per_step=_tensor(
                archive,
                "log_growth_per_step",
            ),
        )
        background_age = _float_scalar(
            archive,
            "background_age_minutes",
            allow_nan=True,
        )
        background_state_support_fraction = _float_scalar(
            archive,
            "background_state_support_fraction",
        )
        background_contribution_fraction = _float_scalar(
            archive,
            "background_contribution_fraction",
        )
        if (
            background_state_support_fraction
            != background_contribution_fraction
        ):
            raise ValueError("background state support fraction mismatch")
        stored_background_tendency_used = _bool_scalar(
            archive,
            "background_tendency_used",
        )
        metadata = ForecastMetadata(
            data_status=DataStatus(_string_scalar(archive, "data_status")),
            coverage_by_frame=_tensor(archive, "coverage_by_frame"),
            background_used=_bool_scalar(archive, "background_used"),
            background_contribution_fraction=(
                background_contribution_fraction
            ),
            background_age_minutes=(
                None if math.isnan(background_age) else background_age
            ),
            source_support=_tensor(archive, "source_support"),
            motion_disagreement_px=_tensor(
                archive,
                "motion_disagreement_px",
            ),
            growth_disagreement=_tensor(
                archive,
                "growth_disagreement",
            ),
            minimum_phase_correlation_psr=_floating_scalar_tensor(
                archive,
                "minimum_phase_correlation_psr",
                allow_nan=True,
            ),
            tendency_pair_count=_int_scalar(
                archive,
                "tendency_pair_count",
            ),
            tendency_source=TendencySource(
                _string_scalar(archive, "tendency_source")
            ),
            provenance=_string_scalar(archive, "provenance"),
        )
        if (
            metadata.background_tendency_used
            != stored_background_tendency_used
        ):
            raise ValueError("background tendency provenance mismatch")
        expected_background_used = (
            metadata.background_state_support_fraction > config.epsilon
            or metadata.background_tendency_used
        )
        if metadata.background_used != expected_background_used:
            raise ValueError("background usage provenance mismatch")
        if metadata.background_used != (
            metadata.background_age_minutes is not None
        ):
            raise ValueError("background age provenance mismatch")
        latest_background_present = _bool_scalar(
            archive,
            "latest_background_present",
        )
        stored_latest_background = _tensor(
            archive,
            "latest_background_dbz",
        )
        latest_background_text = _string_scalar(
            archive,
            "latest_background_digest",
        )
        if latest_background_present:
            latest_background_digest = _validate_digest(
                "latest_background_digest",
                latest_background_text,
            )
        else:
            if latest_background_text:
                raise ValueError(
                    "absent latest background must have an empty digest"
                )
            if stored_latest_background.shape != (0,) or (
                not stored_latest_background.is_floating_point()
            ):
                raise ValueError(
                    "absent latest background must use an empty float array"
                )
            latest_background_digest = None

        (
            analysis_config_json,
            analysis_config_digest,
            analysis_input_digest,
        ) = _analysis_lineage(archive)
        latest_observation_mask_digest = _digest_scalar(
            archive,
            "latest_observation_mask_digest",
        )
        run_background_age = _float_scalar(
            archive,
            "run_background_age_minutes",
            allow_nan=True,
        )
        grid_time_contract, grid_time_contract_digest = (
            _grid_time_contract(archive)
        )
        run = ForecastRunContract(
            config=config,
            _latest_frame_dbz=_tensor(archive, "latest_frame_dbz"),
            _latest_observation_mask=_tensor(
                archive,
                "latest_observation_mask",
            ),
            _latest_background_dbz=(
                stored_latest_background
                if latest_background_present
                else None
            ),
            latest_observation_mask_digest=(
                latest_observation_mask_digest
            ),
            latest_frame_digest=_digest_scalar(
                archive,
                "latest_frame_digest",
            ),
            latest_background_digest=latest_background_digest,
            input_bundle_digest=_digest_scalar(
                archive,
                "input_bundle_digest",
            ),
            background_age_minutes=(
                None
                if math.isnan(run_background_age)
                else run_background_age
            ),
            grid_time_contract=grid_time_contract,
            grid_time_contract_digest=grid_time_contract_digest,
            analysis_config_json=analysis_config_json,
            analysis_config_digest=analysis_config_digest,
            analysis_input_digest=analysis_input_digest,
        )
        stored_state_digest = _digest_scalar(
            archive,
            "state_metadata_digest",
        )
        result = ForecastResult(
            forecast_dbz=forecast_dbz,
            valid_mask=valid_mask,
            forecast_dbz_digest=_digest_scalar(
                archive,
                "forecast_dbz_digest",
            ),
            valid_mask_digest=_digest_scalar(
                archive,
                "valid_mask_digest",
            ),
            state=state,
            metadata=metadata,
            run=run,
            state_metadata_digest=stored_state_digest,
            forecast_run_digest=_digest_scalar(
                archive,
                "forecast_run_digest",
            ),
            audit=None,
        )
    _validate_loaded_contract(result)
    _validate_displacement_mps(result, stored_displacement_mps)
    _validate_velocity(
        "grid_velocity_mps_yx",
        result.grid_velocity_mps_yx,
        stored_grid_velocity,
    )
    _validate_velocity(
        "projected_velocity_mps_xy",
        result.projected_velocity_mps_xy,
        stored_projected_velocity,
    )
    return result


def _open_preflighted_artifact(
    source: Path,
    *,
    maximum_member_count: int,
    maximum_member_bytes: int,
    maximum_total_expanded_bytes: int,
) -> BinaryIO:
    artifact = source.open("rb")
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


def _forecast_run_artifact_digest(
    arrays: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if name == "forecast_run_artifact_digest":
            continue
        value = np.asarray(arrays[name])
        if value.dtype.kind == "O":
            raise ValueError("forecast run arrays cannot use object dtype")
        metadata = json.dumps(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(metadata)
        digest.update(b"\0")
        if value.dtype.kind in {"U", "S"}:
            encoded = json.dumps(
                value.tolist(),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(encoded)
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def seal_forecast_run_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(arrays)
    sealed["forecast_run_artifact_digest"] = np.asarray(
        _forecast_run_artifact_digest(sealed)
    )
    return sealed


def _grid_time_contract(
    archive: np.lib.npyio.NpzFile,
) -> tuple[RadarGridTimeContract | None, str | None]:
    present = _bool_scalar(archive, "grid_time_contract_present")
    contract_json = _string_scalar(archive, "grid_time_contract_json")
    digest_text = _string_scalar(archive, "grid_time_contract_digest")
    if not present:
        if contract_json or digest_text:
            raise ValueError(
                "absent grid/time contract must have empty metadata"
            )
        return None, None
    try:
        value = json.loads(contract_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid grid_time_contract_json") from error
    if not isinstance(value, dict):
        raise ValueError("grid_time_contract_json must contain an object")
    expected = {
        "valid_times",
        "dx_m",
        "dy_m",
        "projection",
        "grid_hash",
        "background_valid_times",
        "pixel_to_projected_matrix_m",
    }
    if set(value) != expected:
        raise ValueError("grid_time_contract_json has invalid fields")
    background_times = value["background_valid_times"]
    matrix = value["pixel_to_projected_matrix_m"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
    ):
        raise ValueError("invalid grid_time_contract_json")
    try:
        contract = RadarGridTimeContract(
            valid_times=tuple(value["valid_times"]),
            dx_m=value["dx_m"],
            dy_m=value["dy_m"],
            projection=value["projection"],
            grid_hash=value["grid_hash"],
            pixel_to_projected_matrix_m=(
                (float(matrix[0][0]), float(matrix[0][1])),
                (float(matrix[1][0]), float(matrix[1][1])),
            ),
            background_valid_times=(
                None
                if background_times is None
                else tuple(background_times)
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid grid_time_contract_json") from error
    digest = _validate_digest("grid_time_contract_digest", digest_text)
    if contract.digest != digest:
        raise ValueError("grid/time contract digest mismatch")
    return contract, digest


def _validate_displacement_mps(
    result: ForecastResult,
    stored: Tensor,
) -> None:
    if stored.shape != (2,) or not stored.is_floating_point():
        raise ValueError("displacement_mps_yx must be a floating [2] vector")
    expected = result.displacement_mps_yx
    if expected is None:
        if not bool(torch.all(torch.isnan(stored))):
            raise ValueError(
                "displacement_mps_yx requires a grid/time contract"
            )
        return
    if not torch.allclose(
        stored.to(dtype=expected.dtype),
        expected,
        rtol=1.0e-6,
        atol=1.0e-9,
    ):
        raise ValueError("displacement_mps_yx disagrees with the run contract")


def _validate_velocity(
    name: str,
    expected: Tensor | None,
    stored: Tensor,
) -> None:
    if stored.shape != (2,) or not stored.is_floating_point():
        raise ValueError(f"{name} must be a floating [2] vector")
    if expected is None:
        if not bool(torch.all(torch.isnan(stored))):
            raise ValueError(f"{name} requires a grid/time contract")
        return
    if not torch.allclose(
        stored.to(dtype=expected.dtype),
        expected,
        rtol=1.0e-6,
        atol=1.0e-9,
    ):
        raise ValueError(f"{name} disagrees with the run contract")


def _validate_loaded_contract(result: ForecastResult) -> None:
    forecast = result.forecast_dbz
    valid = result.valid_mask
    state = result.state
    metadata = result.metadata
    config = result.run.config
    floating = (torch.float32, torch.float64)
    float_tensors = (
        forecast,
        state.echo_linear,
        state.displacement_yx,
        state.log_growth_per_step,
        metadata.coverage_by_frame,
        metadata.source_support,
        metadata.motion_disagreement_px,
        metadata.growth_disagreement,
    )
    if any(value.dtype not in floating for value in float_tensors):
        raise ValueError("forecast run tensors must use float32 or float64")
    state_tensors = (
        forecast,
        state.echo_linear,
        state.displacement_yx,
        state.log_growth_per_step,
        metadata.source_support,
    )
    if len({value.dtype for value in state_tensors}) != 1:
        raise ValueError("forecast run state tensors must share one dtype")
    if forecast.ndim != 3 or forecast.shape[0] != config.forecast_steps:
        raise ValueError("forecast_dbz has the wrong lead shape")
    if valid.dtype != torch.bool or valid.shape != forecast.shape:
        raise ValueError("valid_mask must be boolean with the forecast shape")
    if state.echo_linear.ndim != 2:
        raise ValueError("state_echo_linear must be two-dimensional")
    if forecast.shape[1:] != state.echo_linear.shape:
        raise ValueError("forecast and state grids disagree")
    if state.displacement_yx.shape != (2,):
        raise ValueError("displacement_yx must have shape [2]")
    if state.log_growth_per_step.ndim != 0:
        raise ValueError("log_growth_per_step must be scalar")
    if metadata.coverage_by_frame.shape != (3,):
        raise ValueError("coverage_by_frame must have shape [3]")
    if metadata.source_support.shape != state.echo_linear.shape:
        raise ValueError("source_support must match the state grid")
    if metadata.motion_disagreement_px.ndim != 0:
        raise ValueError("motion_disagreement_px must be scalar")
    if metadata.growth_disagreement.ndim != 0:
        raise ValueError("growth_disagreement must be scalar")
    latest_observation_mask = result.run.latest_observation_mask
    if latest_observation_mask.shape != state.echo_linear.shape:
        raise ValueError("latest observation mask must match the state grid")
    if not torch.equal(torch.isfinite(forecast), valid):
        raise ValueError("valid_mask must match finite forecast values")
    if not all(bool(torch.all(torch.isfinite(value))) for value in float_tensors[1:]):
        raise ValueError("forecast run state and metadata must be finite")
    if bool(torch.any(state.echo_linear < 0)):
        raise ValueError("state_echo_linear cannot be negative")
    if not bool(
        torch.all(
            (metadata.coverage_by_frame >= 0)
            & (metadata.coverage_by_frame <= 1)
        )
    ):
        raise ValueError("coverage_by_frame must be in [0, 1]")
    if not bool(
        torch.all(
            (metadata.source_support >= 0)
            & (metadata.source_support <= 1)
        )
    ):
        raise ValueError("source_support must be in [0, 1]")
    if metadata.background_age_minutes is not None and (
        not math.isfinite(metadata.background_age_minutes)
        or metadata.background_age_minutes < 0
    ):
        raise ValueError("background age must be finite and nonnegative")
    result.validate_issuance()


def _numpy(value: Tensor) -> NDArray[Any]:
    return value.detach().contiguous().cpu().numpy()


def _tensor(archive: np.lib.npyio.NpzFile, name: str) -> Tensor:
    return torch.from_numpy(np.array(_array(archive, name), copy=True))


def _array(archive: np.lib.npyio.NpzFile, name: str) -> NDArray[Any]:
    if name not in archive.files:
        raise ValueError(f"forecast run artifact is missing {name}")
    return cast(NDArray[Any], archive[name])


def _string_scalar(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = _array(archive, name)
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a string scalar")
    return str(value.item())


def _digest_scalar(archive: np.lib.npyio.NpzFile, name: str) -> str:
    return _validate_digest(name, _string_scalar(archive, name))


def _analysis_lineage(
    archive: np.lib.npyio.NpzFile,
) -> tuple[str | None, str | None, str | None]:
    present = _bool_scalar(archive, "analysis_config_present")
    config_json = _string_scalar(archive, "analysis_config_json")
    config_digest = _string_scalar(archive, "analysis_config_digest")
    input_digest = _string_scalar(archive, "analysis_input_digest")
    if not present:
        if config_json or config_digest or input_digest:
            raise ValueError("absent analysis config must have empty lineage")
        return None, None, None
    try:
        config_value = json.loads(config_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid analysis_config_json") from error
    validated_config_digest = _validate_digest(
        "analysis_config_digest",
        config_digest,
    )
    if json_digest(config_value) != validated_config_digest:
        raise ValueError("analysis config digest mismatch")
    return (
        config_json,
        validated_config_digest,
        _validate_digest("analysis_input_digest", input_digest),
    )


def _validate_digest(name: str, value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _bool_scalar(archive: np.lib.npyio.NpzFile, name: str) -> bool:
    value = _array(archive, name)
    if value.shape != () or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(value.item())


def _int_scalar(archive: np.lib.npyio.NpzFile, name: str) -> int:
    value = _array(archive, name)
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be an integer scalar")
    return int(value.item())


def _float_scalar(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    allow_nan: bool = False,
) -> float:
    value = _array(archive, name)
    if value.shape != () or value.dtype.kind != "f":
        raise ValueError(f"{name} must be a floating-point scalar")
    result = float(value.item())
    if math.isfinite(result) or (allow_nan and math.isnan(result)):
        return result
    raise ValueError(f"{name} must be finite")


def _floating_scalar_tensor(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    allow_nan: bool = False,
) -> Tensor:
    value = _array(archive, name)
    if value.shape != () or value.dtype.kind != "f":
        raise ValueError(f"{name} must be a floating-point scalar")
    result = _tensor(archive, name)
    scalar = float(result.item())
    if math.isfinite(scalar) or (allow_nan and math.isnan(scalar)):
        return result
    qualifier = "finite or NaN" if allow_nan else "finite"
    raise ValueError(f"{name} must be {qualifier}")
