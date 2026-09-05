"""Focused regressions for current artifact admission and legacy loading."""

import json

import numpy as np
import pytest
import torch

from advar.nowcast import (
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    NowcastConfig,
    RadarGridTimeContract,
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
