from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar import (  # noqa: E402
    NowcastConfig,
    RadarGridTimeContract,
    SensitivityConfig,
    compute_sensitivity_snapshot,
    load_forecast_run,
    nowcast,
    save_forecast_run,
)
from advar._digest import tensor_digest  # noqa: E402


class ForecastRunArtifactTests(unittest.TestCase):
    def test_tensor_digest_accepts_scalar_tensors(self) -> None:
        value = torch.tensor(0.125, dtype=torch.float64)

        self.assertEqual(tensor_digest(value), tensor_digest(value.clone()))
        self.assertNotEqual(
            tensor_digest(value),
            tensor_digest(value.to(dtype=torch.float32)),
        )

    def frames(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        coordinates = torch.arange(8, dtype=dtype)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        echo = -10.0 + 40.0 * torch.exp(
            -((y - 3.5).square() + (x - 3.5).square()) / 4.0
        )
        return torch.stack((echo, echo, echo))

    def test_restart_round_trip_preserves_m0_snapshot(self) -> None:
        frames = self.frames()
        config = NowcastConfig(
            growth_decay_minutes=120.0,
            max_dbz=60.0,
            min_publish_support=0.7,
        )
        result = nowcast(frames, config)
        sensitivity_config = SensitivityConfig(
            full_map_lead_minutes=(30,),
            tile_size=4,
            soft_fss_window=3,
        )
        in_memory = compute_sensitivity_snapshot(
            frames[-1],
            result,
            result.forecast_dbz.clone(),
            sensitivity_config=sensitivity_config,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        loaded.validate_issuance()
        self.assertEqual(loaded.run.config, config)
        self.assertEqual(
            loaded.state_metadata_digest,
            result.state_metadata_digest,
        )
        self.assertEqual(
            loaded.run.input_bundle_digest,
            result.run.input_bundle_digest,
        )
        self.assertEqual(
            loaded.forecast_run_digest,
            result.forecast_run_digest,
        )
        self.assertIsNone(loaded.run.analysis_config_json)
        self.assertEqual(loaded.forecast_dbz_digest, result.forecast_dbz_digest)
        self.assertEqual(loaded.valid_mask_digest, result.valid_mask_digest)
        self.assertIsNone(loaded.audit)
        torch.testing.assert_close(loaded.forecast_dbz, result.forecast_dbz)
        torch.testing.assert_close(
            loaded.state.echo_linear,
            result.state.echo_linear,
        )
        torch.testing.assert_close(
            loaded.metadata.source_support,
            result.metadata.source_support,
        )
        self.assertEqual(
            loaded.metadata.background_tendency_used,
            result.metadata.background_tendency_used,
        )
        self.assertEqual(
            loaded.metadata.background_state_support_fraction,
            result.metadata.background_state_support_fraction,
        )
        torch.testing.assert_close(
            loaded.metadata.minimum_phase_correlation_psr,
            result.metadata.minimum_phase_correlation_psr,
        )
        torch.testing.assert_close(
            loaded.run.latest_observation_mask,
            result.run.latest_observation_mask,
        )

        reloaded = compute_sensitivity_snapshot(
            frames[-1],
            loaded,
            loaded.forecast_dbz.clone(),
            sensitivity_config=sensitivity_config,
        )
        torch.testing.assert_close(
            reloaded.context_features,
            in_memory.context_features,
        )
        torch.testing.assert_close(
            reloaded.forecast_scores,
            in_memory.forecast_scores,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.control_sensitivity,
            in_memory.control_sensitivity,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.forecast_sensitivity,
            in_memory.forecast_sensitivity,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.direct.maps,
            in_memory.direct.maps,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.direct.tile_norm,
            in_memory.direct.tile_norm,
            equal_nan=True,
        )
        self.assertEqual(reloaded.trust_score, in_memory.trust_score)

    def test_round_trip_preserves_grid_time_contract(self) -> None:
        frames = self.frames()
        background = frames - 1.0
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=500.0,
            dy_m=750.0,
            projection="EPSG:5179",
            grid_hash="c" * 64,
            background_valid_times=(
                "2026-07-30T23:50:00Z",
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
            ),
        )
        result = nowcast(
            frames,
            background_frames_dbz=background,
            background_age_minutes=10.0,
            grid_time_contract=contract,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.grid_time_contract, contract)
        self.assertEqual(
            loaded.run.grid_time_contract_digest,
            contract.digest,
        )
        self.assertEqual(loaded.run.background_age_minutes, 10.0)
        self.assertEqual(loaded.forecast_run_digest, result.forecast_run_digest)
        torch.testing.assert_close(
            loaded.displacement_mps_yx,
            result.displacement_mps_yx,
        )

    def test_load_rejects_tampered_grid_time_contract(self) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="c" * 64,
        )
        result = nowcast(self.frames(), grid_time_contract=contract)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            contract_value = arrays["grid_time_contract_json"].item()
            arrays["grid_time_contract_json"] = np.asarray(
                contract_value.replace('"dx_m":1000.0', '"dx_m":2000.0')
            )
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(
                ValueError,
                "grid/time contract digest mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_tampered_physical_displacement(self) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=500.0,
            projection="EPSG:5179",
            grid_hash="d" * 64,
        )
        result = nowcast(self.frames(), grid_time_contract=contract)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["displacement_mps_yx"] += 1.0
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(
                ValueError,
                "displacement_mps_yx disagrees",
            ):
                load_forecast_run(path)

    def test_input_bundle_digest_covers_all_three_input_times(self) -> None:
        frames = self.frames()
        accepted = torch.ones_like(frames, dtype=torch.bool)
        background = frames - 1.0
        base = nowcast(
            frames,
            qc_mask=accepted,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        earlier_frame = frames.clone()
        earlier_frame[0, 2, 2] += 0.25
        earlier_mask = accepted.clone()
        earlier_mask[0, 2, 2] = False
        earlier_background = background.clone()
        earlier_background[0, 2, 2] += 0.25
        variants = (
            nowcast(
                earlier_frame,
                qc_mask=accepted,
                background_frames_dbz=background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=earlier_mask,
                background_frames_dbz=background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=accepted,
                background_frames_dbz=earlier_background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=accepted,
                background_frames_dbz=background,
                background_age_minutes=20.0,
            ),
        )

        for variant in variants:
            self.assertNotEqual(
                variant.run.input_bundle_digest,
                base.run.input_bundle_digest,
            )
            self.assertNotEqual(
                variant.forecast_run_digest,
                base.forecast_run_digest,
            )

    def test_load_rejects_tampered_state(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["state_echo_linear"][0, 0] += 1.0
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(
                ValueError,
                "state or metadata",
            ):
                load_forecast_run(path)

    def test_save_rejects_mutated_metadata(self) -> None:
        result = nowcast(self.frames())
        result.metadata.source_support[0, 0] = 0.0

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            with self.assertRaisesRegex(
                ValueError,
                "state or metadata",
            ):
                save_forecast_run(result, path)
            self.assertFalse(path.exists())

    def test_load_rejects_tampered_phase_correlation_psr(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["minimum_phase_correlation_psr"] += 1.0
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(
                ValueError,
                "state or metadata",
            ):
                load_forecast_run(path)

    def test_load_rejects_malformed_phase_correlation_psr_schema(self) -> None:
        result = nowcast(self.frames())
        malformed_values = (
            np.asarray(1, dtype=np.int64),
            np.asarray([1.0], dtype=np.float64),
            np.asarray(np.inf, dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                original: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }

            for malformed in malformed_values:
                with self.subTest(value=malformed):
                    arrays = dict(original)
                    arrays["minimum_phase_correlation_psr"] = malformed
                    np.savez_compressed(path, **arrays)

                    with self.assertRaisesRegex(
                        ValueError,
                        "minimum_phase_correlation_psr must be",
                    ):
                        load_forecast_run(path)

    def test_load_rejects_tampered_background_tendency_provenance(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["background_tendency_used"] = np.asarray(True)
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(
                ValueError,
                "background tendency provenance mismatch",
            ):
                load_forecast_run(path)

    def test_save_rejects_mutated_analysis_lineage(self) -> None:
        result = nowcast(self.frames())
        changed_run = replace(
            result.run,
            analysis_config_json="{}",
            analysis_config_digest="0" * 64,
            analysis_input_digest="1" * 64,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            with self.assertRaisesRegex(
                ValueError,
                "analysis config digest mismatch",
            ):
                save_forecast_run(
                    replace(result, run=changed_run),
                    path,
                )
            self.assertFalse(path.exists())

    def test_load_rejects_config_json_that_disagrees_with_digest(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["nowcast_config_json"] = np.asarray(
                '{"interval_minutes":5,"horizon_minutes":180,'
                '"min_dbz":-10.0,"max_dbz":70.0,'
                '"echo_threshold_dbz":5.0,"recent_weight":0.6666666666666666,'
                '"pair_echo_dilation_px":3,"max_displacement_px":20.0,'
                '"max_log_growth_per_step":0.30010459245033816,'
                '"growth_decay_minutes":60.0,"min_publish_support":0.95,'
                '"epsilon":1e-06}'
            )
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(
                ValueError,
                "config digest mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_tampered_input_bundle_identity(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["input_bundle_digest"] = np.asarray("0" * 64)
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(ValueError, "run identity"):
                load_forecast_run(path)


if __name__ == "__main__":
    unittest.main()
