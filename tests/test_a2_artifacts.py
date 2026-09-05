"""Focused regressions for current artifact admission and legacy loading."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from advar.nowcast import (
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    ForecastRunContract,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    NowcastConfig,
    RadarGridTimeContract,
    _forecast_fixed_input_context_digest,
    _forecast_full_analysis_input_digest,
    _forecast_input_bundle_digest_from_digests,
    _forecast_run_identity_digest,
    estimate_state,
    nowcast,
    radar_projected_crs_semantic_digest,
)
from advar.run_artifact import (
    load_forecast_run,
    save_forecast_run,
    seal_forecast_run_arrays,
)


_VALID_TIMES = (
    "2026-08-26T00:00:00Z",
    "2026-08-26T00:10:00Z",
    "2026-08-26T00:20:00Z",
)


def _scientific_grid(contract: str) -> RadarGridTimeContract:
    values: dict[str, object] = {
        "valid_times": _VALID_TIMES,
        "dx_m": 1000.0,
        "dy_m": 1000.0,
        "projection": "EPSG:5179",
        "grid_hash": "1" * 64,
        "grid_shape_yx": (4, 4),
        "projected_crs_digest": radar_projected_crs_semantic_digest(
            "EPSG:5179"
        ),
        "metric_domain_digest": CURRENT_RADAR_METRIC_DOMAIN.digest,
        "cell_center_origin_xy_m": (1_000_000.0, 2_000_000.0),
        "grid_coordinate_dtype": RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        "cell_center_convention": RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
        "spatial_grid_contract": contract,
    }
    if contract.endswith("v6"):
        values["metric_domain_evidence_digest"] = (
            CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
        )
    return RadarGridTimeContract(**values)


def _frames() -> torch.Tensor:
    return torch.full((3, 4, 4), 20.0, dtype=torch.float64)


def test_v5_grid_is_research_usable_but_rejected_by_current_nowcast() -> None:
    grid = _scientific_grid("radar-spatial-grid-identity-v5")

    state, metadata = estimate_state(
        _frames(),
        NowcastConfig(),
        grid_time_contract=grid,
    )
    assert state.echo_linear.shape == (4, 4)
    assert metadata.data_status.value == "OBSERVED"

    with pytest.raises(
        ValueError,
        match="current forecast issuance requires radar-spatial-grid-identity-v6",
    ):
        nowcast(_frames(), grid_time_contract=grid)


def test_current_run_from_inputs_rejects_v5_grid_before_forecast_issuance() -> None:
    grid = _scientific_grid("radar-spatial-grid-identity-v5")

    with pytest.raises(
        ValueError,
        match="current forecast issuance requires radar-spatial-grid-identity-v6",
    ):
        ForecastRunContract.from_inputs(
            NowcastConfig(),
            _frames(),
            torch.ones_like(_frames(), dtype=torch.bool),
            None,
            grid_time_contract=grid,
        )


def _legacy_v5_archive(
    arrays: dict[str, np.ndarray],
    result,
    grid: RadarGridTimeContract,
) -> dict[str, np.ndarray]:
    """Re-address a current archive as an immutable historical v5 run."""

    run = result.run
    assert run.observation_masks_digest is not None
    assert run.observation_quality_weight_digest is not None
    assert run.observation_std_dbz_digest is not None
    assert run.source_available_mask_digest is not None
    assert run.learned_model_input_features_digest is not None
    assert run.fixed_input_context_digest is not None
    assert run.full_analysis_input_digest is not None
    input_bundle_digest = _forecast_input_bundle_digest_from_digests(
        input_frames_digest=run.input_frames_digest,
        observation_masks_digest=run.observation_masks_digest,
        background_frames_digest=run.background_frames_digest,
        background_age_minutes=run.background_age_minutes,
        grid_time_contract_digest=grid.digest,
        operational_calibration_manifest_digest=(
            run.operational_calibration_manifest_digest
        ),
        operational_calibration_approval_digest=(
            run.operational_calibration_approval_digest
        ),
        operational_data_identity_digest=run.operational_data_identity_digest,
    )
    fixed_input_context_digest = _forecast_fixed_input_context_digest(
        observation_masks_digest=run.observation_masks_digest,
        observation_quality_weight_digest=run.observation_quality_weight_digest,
        observation_std_dbz_digest=run.observation_std_dbz_digest,
        source_available_mask_digest=run.source_available_mask_digest,
        learned_model_input_features_digest=(
            run.learned_model_input_features_digest
        ),
        background_frames_digest=run.background_frames_digest,
        background_age_minutes=run.background_age_minutes,
        grid_time_contract_digest=grid.digest,
        operational_calibration_manifest_digest=(
            run.operational_calibration_manifest_digest
        ),
        operational_calibration_approval_digest=(
            run.operational_calibration_approval_digest
        ),
        operational_data_identity_digest=run.operational_data_identity_digest,
        input_plan_digest=run.input_plan_digest,
    )
    full_analysis_input_digest = _forecast_full_analysis_input_digest(
        input_frames_digest=run.input_frames_digest,
        fixed_input_context_digest=fixed_input_context_digest,
    )
    historical_run = replace(
        run,
        input_bundle_digest=input_bundle_digest,
        fixed_input_context_digest=fixed_input_context_digest,
        full_analysis_input_digest=full_analysis_input_digest,
        grid_time_contract=grid,
        grid_time_contract_digest=grid.digest,
    )
    historical = dict(arrays)
    historical["forecast_run_artifact_version"] = np.asarray(
        "forecast-run-v42"
    )
    historical["input_bundle_digest"] = np.asarray(input_bundle_digest)
    historical["fixed_input_context_digest"] = np.asarray(
        fixed_input_context_digest
    )
    historical["full_analysis_input_digest"] = np.asarray(
        full_analysis_input_digest
    )
    historical["grid_time_contract_json"] = np.asarray(
        json.dumps(grid.payload, sort_keys=True, separators=(",", ":"))
    )
    historical["grid_time_contract_digest"] = np.asarray(grid.digest)
    historical["forecast_run_digest"] = np.asarray(
        _forecast_run_identity_digest(
            historical_run,
            result.state_metadata_digest,
            result.forecast_dbz_digest,
            result.valid_mask_digest,
        )
    )
    return seal_forecast_run_arrays(historical)


def test_legacy_v5_grid_artifact_migrates_without_current_issuance_gate(
    tmp_path,
) -> None:
    current_grid = _scientific_grid("radar-spatial-grid-identity-v6")
    result = nowcast(_frames(), grid_time_contract=current_grid)
    current_path = tmp_path / "current.npz"
    legacy_path = tmp_path / "legacy-v5.npz"
    save_forecast_run(result, current_path)
    with np.load(current_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }

    legacy_grid = _scientific_grid("radar-spatial-grid-identity-v5")
    np.savez_compressed(
        legacy_path,
        **_legacy_v5_archive(arrays, result, legacy_grid),
    )

    loaded = load_forecast_run(legacy_path)
    assert loaded.run.grid_time_contract == legacy_grid
    torch.testing.assert_close(
        loaded.forecast_dbz,
        result.forecast_dbz,
        equal_nan=True,
    )
    assert torch.equal(loaded.valid_mask, result.valid_mask)


def test_current_v72_artifact_rejects_v5_grid_but_legacy_version_is_not_gated(
    tmp_path,
) -> None:
    grid = _scientific_grid("radar-spatial-grid-identity-v6")
    result = nowcast(_frames(), grid_time_contract=grid)
    path = tmp_path / "run.npz"
    save_forecast_run(result, path)

    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    v5_grid = _scientific_grid("radar-spatial-grid-identity-v5")
    arrays["grid_time_contract_json"] = np.asarray(
        json.dumps(v5_grid.payload, sort_keys=True, separators=(",", ":"))
    )
    arrays["grid_time_contract_digest"] = np.asarray(v5_grid.digest)
    np.savez_compressed(path, **seal_forecast_run_arrays(arrays))

    with pytest.raises(
        ValueError,
        match="current forecast issuance requires radar-spatial-grid-identity-v6",
    ):
        load_forecast_run(path)


def test_integer_background_age_is_canonicalized_and_serialized_as_float(
    tmp_path,
) -> None:
    frames = _frames()
    result = nowcast(
        frames,
        background_frames_dbz=frames - 1.0,
        background_age_minutes=10,
    )
    assert type(result.run.background_age_minutes) is float
    assert result.run.background_age_minutes == 10.0

    path = tmp_path / "run.npz"
    save_forecast_run(result, path)
    with np.load(path, allow_pickle=False) as archive:
        for name in (
            "run_background_age_minutes",
            "background_age_minutes",
            "state_path_age_minutes",
            "observation_path_age_minutes",
            "background_path_age_minutes",
        ):
            assert archive[name].dtype == np.dtype("float64")
    loaded = load_forecast_run(path)
    assert type(loaded.run.background_age_minutes) is float
    assert loaded.run.background_age_minutes == 10.0


def test_legacy_nan_forecast_migration_compares_nan_masks(tmp_path) -> None:
    frames = _frames()
    qc_mask = torch.ones_like(frames, dtype=torch.bool)
    qc_mask[:, 0, 0] = False
    result = nowcast(frames, qc_mask=qc_mask)
    current_path = tmp_path / "current.npz"
    legacy_path = tmp_path / "legacy-v42.npz"
    save_forecast_run(result, current_path)
    with np.load(current_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    arrays["forecast_run_artifact_version"] = np.asarray("forecast-run-v42")
    np.savez_compressed(legacy_path, **seal_forecast_run_arrays(arrays))

    loaded = load_forecast_run(legacy_path)
    torch.testing.assert_close(
        loaded.forecast_dbz,
        result.forecast_dbz,
        equal_nan=True,
    )
    assert torch.equal(loaded.valid_mask, result.valid_mask)
