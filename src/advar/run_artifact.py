from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, cast

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


FORECAST_RUN_ARTIFACT_VERSION = "forecast-run-v4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
    artifact_digest = _forecast_run_artifact_digest(
        result.forecast_run_digest,
    )
    metadata = result.metadata
    grid_time_contract = result.run.grid_time_contract
    displacement_mps = result.displacement_mps_yx
    return {
        "forecast_run_artifact_version": np.asarray(
            FORECAST_RUN_ARTIFACT_VERSION
        ),
        "forecast_run_artifact_digest": np.asarray(artifact_digest),
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


def save_forecast_run(result: ForecastResult, path: str | Path) -> None:
    """Atomically persist one issued forecast run as a compressed NPZ."""

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("forecast run artifact path must end with .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = forecast_run_arrays(result)
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
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_forecast_run(path: str | Path) -> ForecastResult:
    """Load and independently validate a durable forecast run artifact."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        version = _string_scalar(archive, "forecast_run_artifact_version")
        if version != FORECAST_RUN_ARTIFACT_VERSION:
            raise ValueError(f"unsupported forecast run artifact: {version}")

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
            minimum_phase_correlation_psr=_tensor(
                archive,
                "minimum_phase_correlation_psr",
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
            _latest_observation_mask=_tensor(
                archive,
                "latest_observation_mask",
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
        stored_artifact_digest = _digest_scalar(
            archive,
            "forecast_run_artifact_digest",
        )

    _validate_loaded_contract(result)
    _validate_displacement_mps(result, stored_displacement_mps)
    expected_artifact_digest = _forecast_run_artifact_digest(
        result.forecast_run_digest,
    )
    if stored_artifact_digest != expected_artifact_digest:
        raise ValueError("forecast run artifact digest mismatch")
    return result


def _forecast_run_artifact_digest(
    forecast_run_digest: str,
) -> str:
    return json_digest(
        {
            "artifact_version": FORECAST_RUN_ARTIFACT_VERSION,
            "forecast_run_digest": forecast_run_digest,
        }
    )


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
    }
    if set(value) != expected:
        raise ValueError("grid_time_contract_json has invalid fields")
    background_times = value["background_valid_times"]
    try:
        contract = RadarGridTimeContract(
            valid_times=tuple(value["valid_times"]),
            dx_m=value["dx_m"],
            dy_m=value["dy_m"],
            projection=value["projection"],
            grid_hash=value["grid_hash"],
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
