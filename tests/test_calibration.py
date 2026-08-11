import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar._digest import json_digest  # noqa: E402
from advar.calibration import (  # noqa: E402
    CalibrationMetric,
    CalibrationRegime,
    OperationalCalibrationManifest,
    OperationalDataIdentity,
    algorithm_bundle_digest,
)
from advar.nowcast import (  # noqa: E402
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    operational_runtime_profile_digest,
)


class OperationalCalibrationManifestTests(unittest.TestCase):
    def _metric(self) -> CalibrationMetric:
        return CalibrationMetric(
            name="csi_35dbz_60min",
            definition_digest="1" * 64,
            direction="maximize",
            acceptance_threshold=0.55,
            value=0.61,
        )

    def _manifest(
        self,
        *,
        profile_kind: str = "p1",
        expected_runtime_profile_digest: str = "a" * 64,
        expected_algorithm_bundle_digest: str | None = None,
        validation_period: tuple[str, str] = (
            "2025-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        ),
        validation_metrics: tuple[CalibrationMetric, ...] | None = None,
        validation_case_count: int = 120,
        validation_regimes: tuple[CalibrationRegime, ...] = (
            CalibrationRegime("convective", 70),
            CalibrationRegime("stratiform", 50),
        ),
    ) -> OperationalCalibrationManifest:
        return OperationalCalibrationManifest(
            calibration_id="radar-operational-v1",
            profile_kind=profile_kind,
            expected_runtime_profile_digest=expected_runtime_profile_digest,
            expected_algorithm_bundle_digest=(
                algorithm_bundle_digest()
                if expected_algorithm_bundle_digest is None
                else expected_algorithm_bundle_digest
            ),
            calibration_dataset_digest="b" * 64,
            validation_dataset_digest="c" * 64,
            data_identity=OperationalDataIdentity(
                radar_class="composite-radar-1km",
                qc_pipeline_digest="d" * 64,
                observation_error_model_digest="e" * 64,
                background_model_digest="f" * 64,
            ),
            training_period=(
                "2024-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ),
            validation_period=validation_period,
            validation_case_count=validation_case_count,
            validation_regimes=validation_regimes,
            validation_metrics=(
                (self._metric(),)
                if validation_metrics is None
                else validation_metrics
            ),
        )

    def _grid(self) -> RadarGridTimeContract:
        return RadarGridTimeContract(
            valid_times=(
                "2026-08-04T00:00:00Z",
                "2026-08-04T00:10:00Z",
                "2026-08-04T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:3857",
            grid_hash="0" * 64,
        )

    def test_round_trip_is_canonical_and_content_addressed(self) -> None:
        manifest = self._manifest()

        restored = OperationalCalibrationManifest.from_json(manifest.json)

        self.assertEqual(restored, manifest)
        self.assertEqual(restored.json, manifest.json)
        self.assertRegex(manifest.digest, r"^[0-9a-f]{64}$")
        self.assertRegex(manifest.metric_contract_digest, r"^[0-9a-f]{64}$")
        self.assertEqual(json.loads(manifest.json), manifest.value)

    def test_operational_radar_source_identity_round_trips(self) -> None:
        identities = (
            OperationalDataIdentity(
                radar_class="single-site",
                qc_pipeline_digest="1" * 64,
                observation_error_model_digest="2" * 64,
                background_model_digest="3" * 64,
                radar_source_kind="single_site",
                radar_site_digest="4" * 64,
                radar_site_location_digest="5" * 64,
                radar_source_contract_digest="6" * 64,
            ),
            OperationalDataIdentity(
                radar_class="mosaic",
                qc_pipeline_digest="1" * 64,
                observation_error_model_digest="2" * 64,
                background_model_digest="3" * 64,
                radar_source_kind="mosaic",
                radar_source_contract_digest="6" * 64,
                source_radar_index_map_digest="7" * 64,
                effective_horizontal_range_map_digest="8" * 64,
                source_selection_policy_digest="9" * 64,
            ),
        )

        for identity in identities:
            with self.subTest(kind=identity.radar_source_kind):
                self.assertEqual(
                    OperationalDataIdentity.from_json(identity.json),
                    identity,
                )

        with self.assertRaisesRegex(ValueError, "source identity"):
            OperationalDataIdentity(
                radar_class="single-site",
                qc_pipeline_digest="1" * 64,
                observation_error_model_digest="2" * 64,
                background_model_digest="3" * 64,
                radar_source_kind="single_site",
                radar_site_digest="4" * 64,
            )

    def test_algorithm_bundle_digest_tracks_source_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = root / "algorithm.py"
            module.write_text("VALUE = 1\n", encoding="utf-8")
            initial = algorithm_bundle_digest(root)
            module.write_text("VALUE = 2\n", encoding="utf-8")
            changed = algorithm_bundle_digest(root)

        self.assertNotEqual(initial, changed)

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

    def test_metrics_and_regimes_are_certified(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not meet"):
            CalibrationMetric(
                name="csi",
                definition_digest="1" * 64,
                direction="maximize",
                acceptance_threshold=0.6,
                value=0.5,
            )

        manifest = self._manifest()
        changed = manifest.value
        changed["metric_contract_digest"] = "7" * 64
        with self.assertRaisesRegex(ValueError, "metric contract digest"):
            OperationalCalibrationManifest.from_json(
                json.dumps(
                    changed,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        with self.assertRaisesRegex(ValueError, "regime counts"):
            self._manifest(validation_case_count=121)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            CalibrationRegime("invalid", 1.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._manifest(validation_case_count=1.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "canonical name order"):
            self._manifest(
                validation_regimes=(
                    CalibrationRegime("stratiform", 50),
                    CalibrationRegime("convective", 70),
                )
            )

    def test_manifest_rejects_invalid_lineage(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            self._manifest(expected_runtime_profile_digest="not-a-digest")
        with self.assertRaisesRegex(ValueError, "datasets must differ"):
            manifest = self._manifest()
            OperationalCalibrationManifest(
                **{
                    **manifest.__dict__,
                    "validation_dataset_digest": (
                        manifest.calibration_dataset_digest
                    ),
                }
            )

    def test_operational_p1_run_requires_matching_approved_manifest(
        self,
    ) -> None:
        config = NowcastConfig()
        grid = self._grid()
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
            expected_runtime_profile_digest=(
                operational_runtime_profile_digest(
                    config,
                    grid,
                    analysis_config=analysis,
                )
            )
        )
        frames = torch.ones((3, 4, 4), dtype=torch.float64)
        masks = torch.ones_like(frames, dtype=torch.bool)
        lineage = {
            "grid_time_contract": grid,
            "analysis_config_json": analysis_json,
            "analysis_config_digest": json_digest(analysis),
            "analysis_input_digest": "9" * 64,
        }

        with self.assertRaisesRegex(ValueError, "requires a calibration manifest"):
            ForecastRunContract.from_inputs(
                config,
                frames,
                masks,
                None,
                **lineage,
            )
        with self.assertRaisesRegex(ValueError, "provided together"):
            ForecastRunContract.from_inputs(
                config,
                frames,
                masks,
                None,
                operational_calibration_manifest_json=manifest.json,
                operational_calibration_manifest_digest=manifest.digest,
                **lineage,
            )
        with self.assertRaisesRegex(ValueError, "not approved"):
            ForecastRunContract.from_inputs(
                config,
                frames,
                masks,
                None,
                operational_calibration_manifest_json=manifest.json,
                operational_calibration_manifest_digest=manifest.digest,
                operational_calibration_approval_digest="7" * 64,
                operational_data_identity_json=manifest.data_identity.json,
                operational_data_identity_digest=manifest.data_identity.digest,
                **lineage,
            )

        run = ForecastRunContract.from_inputs(
            config,
            frames,
            masks,
            None,
            operational_calibration_manifest_json=manifest.json,
            operational_calibration_manifest_digest=manifest.digest,
            operational_calibration_approval_digest=manifest.digest,
            operational_data_identity_json=manifest.data_identity.json,
            operational_data_identity_digest=manifest.data_identity.digest,
            **lineage,
        )
        self.assertEqual(
            run.operational_runtime_profile_digest,
            manifest.expected_runtime_profile_digest,
        )
        run.validate_integrity()

        wrong_identity = OperationalDataIdentity(
            radar_class="other-radar",
            qc_pipeline_digest=manifest.data_identity.qc_pipeline_digest,
            observation_error_model_digest=(
                manifest.data_identity.observation_error_model_digest
            ),
            background_model_digest=(
                manifest.data_identity.background_model_digest
            ),
        )
        with self.assertRaisesRegex(ValueError, "identity is not calibrated"):
            ForecastRunContract.from_inputs(
                config,
                frames,
                masks,
                None,
                operational_calibration_manifest_json=manifest.json,
                operational_calibration_manifest_digest=manifest.digest,
                operational_calibration_approval_digest=manifest.digest,
                operational_data_identity_json=wrong_identity.json,
                operational_data_identity_digest=wrong_identity.digest,
                **lineage,
            )

        changed = self._manifest(
            expected_runtime_profile_digest=(
                manifest.expected_runtime_profile_digest
            ),
            expected_algorithm_bundle_digest="8" * 64,
        )
        with self.assertRaisesRegex(ValueError, "algorithm bundle"):
            ForecastRunContract.from_inputs(
                config,
                frames,
                masks,
                None,
                operational_calibration_manifest_json=changed.json,
                operational_calibration_manifest_digest=changed.digest,
                operational_calibration_approval_digest=changed.digest,
                operational_data_identity_json=changed.data_identity.json,
                operational_data_identity_digest=changed.data_identity.digest,
                **lineage,
            )

    def test_operational_p0_run_has_independent_certified_profile(self) -> None:
        config = NowcastConfig()
        grid = self._grid()
        manifest = self._manifest(
            profile_kind="p0",
            expected_runtime_profile_digest=(
                operational_runtime_profile_digest(config, grid)
            ),
        )
        frames = torch.ones((3, 4, 4), dtype=torch.float64)
        masks = torch.ones_like(frames, dtype=torch.bool)

        run = ForecastRunContract.from_inputs(
            config,
            frames,
            masks,
            None,
            grid_time_contract=grid,
            operational_calibration_manifest_json=manifest.json,
            operational_calibration_manifest_digest=manifest.digest,
            operational_calibration_approval_digest=manifest.digest,
            operational_data_identity_json=manifest.data_identity.json,
            operational_data_identity_digest=manifest.data_identity.digest,
        )

        self.assertIsNone(run.analysis_config_json)
        self.assertEqual(
            run.operational_runtime_profile_digest,
            manifest.expected_runtime_profile_digest,
        )
        run.validate_integrity()

    def test_p0_runtime_profile_excludes_p1_only_settings(self) -> None:
        grid = self._grid()
        base = NowcastConfig()
        changed = NowcastConfig(
            p1_motion_saturation_safe_margin_mps=3.0,
            p1_growth_saturation_safe_margin_per_step=0.06,
            p1_saturation_uncertainty_multiplier=6.0,
        )
        analysis = {
            "execution_mode": "operational",
            "operational_calibration_id": "radar-operational-v1",
        }

        self.assertEqual(
            operational_runtime_profile_digest(base, grid),
            operational_runtime_profile_digest(changed, grid),
        )
        self.assertNotEqual(
            operational_runtime_profile_digest(
                base,
                grid,
                analysis_config=analysis,
            ),
            operational_runtime_profile_digest(
                changed,
                grid,
                analysis_config=analysis,
            ),
        )


if __name__ == "__main__":
    unittest.main()
