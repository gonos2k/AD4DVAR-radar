import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.calibration import (  # noqa: E402
    CalibrationMetric,
    OperationalCalibrationManifest,
)
from advar._digest import json_digest  # noqa: E402
from advar.nowcast import (  # noqa: E402
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    operational_profile_digest,
)


class OperationalCalibrationManifestTests(unittest.TestCase):
    def _manifest(
        self,
        *,
        expected_profile_digest: str = "a" * 64,
        validation_period: tuple[str, str] = (
            "2025-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        ),
        validation_metrics: tuple[CalibrationMetric, ...] = (
            CalibrationMetric("csi_35dbz_60min", 0.61),
            CalibrationMetric("fss_35dbz_60min", 0.72),
        ),
    ) -> OperationalCalibrationManifest:
        return OperationalCalibrationManifest(
            calibration_id="radar-operational-v1",
            expected_profile_digest=expected_profile_digest,
            radar_class="composite-radar-1km",
            training_period=(
                "2024-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ),
            validation_period=validation_period,
            validation_metrics=validation_metrics,
        )

    def test_round_trip_is_canonical_and_content_addressed(self) -> None:
        manifest = self._manifest()

        restored = OperationalCalibrationManifest.from_json(manifest.json)

        self.assertEqual(restored, manifest)
        self.assertEqual(restored.json, manifest.json)
        self.assertRegex(manifest.digest, r"^[0-9a-f]{64}$")
        self.assertEqual(json.loads(manifest.json), manifest.value)

    def test_load_rejects_noncanonical_json(self) -> None:
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            path.write_text(manifest.json + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be canonical"):
                OperationalCalibrationManifest.load(path)

    def test_training_and_validation_periods_cannot_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            self._manifest(
                validation_period=(
                    "2024-12-31T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                )
            )

    def test_validation_metrics_must_be_nonempty_unique_and_sorted(self) -> None:
        invalid_metrics = (
            (),
            (
                CalibrationMetric("same", 1.0),
                CalibrationMetric("same", 2.0),
            ),
            (
                CalibrationMetric("z", 1.0),
                CalibrationMetric("a", 2.0),
            ),
        )
        for metrics in invalid_metrics:
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                self._manifest(validation_metrics=metrics)

    def test_manifest_rejects_nonfinite_metric_and_invalid_profile_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            CalibrationMetric("csi", float("nan"))
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            self._manifest(expected_profile_digest="not-a-digest")

    def test_operational_run_contract_requires_matching_manifest(self) -> None:
        config = NowcastConfig()
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-04T00:00:00Z",
                "2026-08-04T00:10:00Z",
                "2026-08-04T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:3857",
            grid_hash="b" * 64,
        )
        analysis = {
            "execution_mode": "operational",
            "operational_calibration_id": "radar-operational-v1",
        }
        analysis_json = json.dumps(
            analysis,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest = self._manifest(
            expected_profile_digest=operational_profile_digest(
                config,
                analysis,
                grid,
            )
        )
        frames = torch.ones((3, 4, 4), dtype=torch.float64)
        masks = torch.ones_like(frames, dtype=torch.bool)
        lineage = {
            "grid_time_contract": grid,
            "analysis_config_json": analysis_json,
            "analysis_config_digest": json_digest(analysis),
            "analysis_input_digest": "c" * 64,
        }

        with self.assertRaisesRegex(ValueError, "requires a calibration manifest"):
            ForecastRunContract.from_inputs(
                config,
                frames,
                masks,
                None,
                **lineage,
            )

        run = ForecastRunContract.from_inputs(
            config,
            frames,
            masks,
            None,
            operational_calibration_manifest_json=manifest.json,
            operational_calibration_manifest_digest=manifest.digest,
            **lineage,
        )
        self.assertEqual(
            run.operational_calibration_digest,
            manifest.expected_profile_digest,
        )
        run.validate_integrity()

        changed = self._manifest(expected_profile_digest="d" * 64)
        with self.assertRaisesRegex(ValueError, "profile digest mismatch"):
            ForecastRunContract.from_inputs(
                config,
                frames,
                masks,
                None,
                operational_calibration_manifest_json=changed.json,
                operational_calibration_manifest_digest=changed.digest,
                **lineage,
            )


if __name__ == "__main__":
    unittest.main()
