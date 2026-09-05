from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from advar.nowcast import (
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    _forecast_input_bundle_digest_from_digests,
    nowcast,
)
from advar.run_artifact import (
    load_forecast_run,
    save_forecast_run,
    seal_forecast_run_arrays,
)


class RunIdentityReviewTests(unittest.TestCase):
    @staticmethod
    def frames() -> torch.Tensor:
        return torch.full((3, 4, 4), 20.0, dtype=torch.float64)

    @staticmethod
    def masks(frames: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(frames, dtype=torch.bool)

    @staticmethod
    def grid() -> RadarGridTimeContract:
        return RadarGridTimeContract(
            valid_times=(
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:20:00Z",
            ),
            dx_m=1000.5,
            dy_m=2000.5,
            projection="EPSG:5179",
            grid_hash="a" * 64,
        )

    def test_current_run_rejects_direct_input_bundle_tamper(self) -> None:
        frames = self.frames()
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            self.masks(frames),
            None,
        )

        with self.assertRaisesRegex(ValueError, "input bundle digest mismatch"):
            replace(run, input_bundle_digest="0" * 64).validate_integrity()

    def test_current_background_digest_requires_presence_match(self) -> None:
        frames = self.frames()
        masks = self.masks(frames)
        no_background = ForecastRunContract.from_inputs(
            NowcastConfig(), frames, masks, None
        )
        with self.assertRaisesRegex(
            ValueError,
            "background frame digest presence mismatch",
        ):
            replace(
                no_background,
                background_frames_digest="1" * 64,
            ).validate_integrity()

        background = frames - 1.0
        with_background = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            masks,
            background,
            background_age_minutes=0.0,
        )
        with self.assertRaisesRegex(
            ValueError,
            "background frame digest presence mismatch",
        ):
            replace(
                with_background,
                background_frames_digest=None,
            ).validate_integrity()

    def test_current_bundle_builder_and_validator_accept_both_background_states(
        self,
    ) -> None:
        frames = self.frames()
        masks = self.masks(frames)
        for background, age in ((None, None), (frames - 1.0, 0.0)):
            with self.subTest(background_present=background is not None):
                run = ForecastRunContract.from_inputs(
                    NowcastConfig(),
                    frames,
                    masks,
                    background,
                    background_age_minutes=age,
                )
                run.validate_integrity()
                assert run.observation_masks_digest is not None
                expected = _forecast_input_bundle_digest_from_digests(
                    input_frames_digest=run.input_frames_digest,
                    observation_masks_digest=run.observation_masks_digest,
                    background_frames_digest=run.background_frames_digest,
                    background_age_minutes=run.background_age_minutes,
                    grid_time_contract_digest=run.grid_time_contract_digest,
                    operational_calibration_manifest_digest=(
                        run.operational_calibration_manifest_digest
                    ),
                    operational_calibration_approval_digest=(
                        run.operational_calibration_approval_digest
                    ),
                    operational_data_identity_digest=(
                        run.operational_data_identity_digest
                    ),
                )
                self.assertEqual(run.input_bundle_digest, expected)

    def test_reduced_v42_artifact_keeps_legacy_background_migration(self) -> None:
        frames = self.frames()
        background = frames - 1.0
        result = nowcast(
            frames,
            background_frames_dbz=background,
            background_age_minutes=0.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v42.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name
                    not in {
                        "observation_masks_digest",
                        "observation_quality_weight_digest",
                        "observation_std_dbz_digest",
                        "source_available_mask_digest",
                        "learned_model_input_features_digest",
                        "background_frames_digest",
                        "fixed_input_context_digest",
                        "full_analysis_input_digest",
                    }
                }
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v42"
            )
            np.savez_compressed(path, **seal_forecast_run_arrays(arrays))

            loaded = load_forecast_run(path)

        self.assertIsNone(loaded.run.background_frames_digest)
        self.assertIsNotNone(loaded.run.latest_background_digest)
        self.assertIsNone(loaded.run.fixed_input_context_digest)
        torch.testing.assert_close(loaded.forecast_dbz, result.forecast_dbz)

    def test_affine_conversion_rejects_integer_inputs(self) -> None:
        grid = self.grid()
        with self.assertRaisesRegex(ValueError, "float32/float64"):
            grid.projected_displacement_xy(torch.tensor((1, 2)))
        with self.assertRaisesRegex(ValueError, "float32/float64"):
            grid.displacement_yx_from_projected_xy(torch.tensor((1, 2)))

    def test_affine_conversion_preserves_float32_and_float64_paths(self) -> None:
        grid = self.grid()
        expected_projected = torch.tensor((2001.0, -2000.5))
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                displacement = torch.tensor((1.0, 2.0), dtype=dtype)
                projected = grid.projected_displacement_xy(displacement)
                self.assertEqual(projected.dtype, dtype)
                torch.testing.assert_close(
                    projected,
                    expected_projected.to(dtype=dtype),
                )
                recovered = grid.displacement_yx_from_projected_xy(projected)
                self.assertEqual(recovered.dtype, dtype)
                torch.testing.assert_close(recovered, displacement)


if __name__ == "__main__":
    unittest.main()
