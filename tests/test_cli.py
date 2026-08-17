from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar import cli  # noqa: E402
from advar import load_forecast_run  # noqa: E402
from advar.calibration import (  # noqa: E402
    CalibrationMetric,
    CalibrationRegime,
    OperationalCalibrationManifest,
    OperationalDataIdentity,
    algorithm_bundle_digest,
)
from advar.nowcast import (  # noqa: E402
    NowcastConfig,
    RadarGridTimeContract,
    operational_runtime_profile_digest,
)
from advar.variational import (  # noqa: E402
    AnalysisConfig,
    observation_common_bias_group_map_digest,
    observation_common_bias_mode_weights_digest,
)


class CliTests(unittest.TestCase):
    def _stationary_frames(self) -> np.ndarray:
        y, x = np.meshgrid(
            np.arange(8, dtype=np.float32),
            np.arange(8, dtype=np.float32),
            indexing="ij",
        )
        frame = 35.0 * np.exp(-((y - 3.5) ** 2 + (x - 3.5) ** 2) / 5.0)
        return np.stack((frame, frame, frame)).astype(np.float32)

    def _run_cli(
        self,
        directory: Path,
        frames: np.ndarray,
        *extra_arguments: str,
    ) -> Path:
        input_path = directory / "input.npy"
        output_path = directory / "forecast.npz"
        np.save(input_path, frames, allow_pickle=False)
        arguments = [
            "advar-nowcast",
            str(input_path),
            str(output_path),
            *extra_arguments,
        ]
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            cli.main()
        return output_path

    def _operational_profile_arguments(
        self,
        directory: Path,
        *,
        calibration_id: str = "test-calibration-v1",
        forecast_confidence_length_scale_m: float = 10_000.0,
        variational: bool = True,
    ) -> tuple[str, ...]:
        nowcast_config = NowcastConfig(
            minimum_publish_verified_support=0.95,
            minimum_publish_confidence=0.5,
            minimum_publish_observation_verified_support=0.95,
            maximum_publish_background_fraction=0.25,
            forecast_confidence_length_scale_m=(
                forecast_confidence_length_scale_m
            ),
            maximum_motion_speed_mps=100.0,
            minimum_phase_correlation_psr=0.0,
            pair_echo_dilation_m=3000.0,
            phase_correlation_sidelobe_radius_m=2000.0,
            maximum_pair_velocity_disagreement_mps=10.0,
            maximum_pair_growth_disagreement=0.0953,
            maximum_local_growth_log_error_per_step=0.4055,
            p1_motion_saturation_safe_margin_mps=2.0,
            p1_growth_saturation_safe_margin_per_step=0.04879,
            p1_posterior_saturation_sigma_multiplier=2.0,
            p1_saturation_uncertainty_multiplier=4.0,
            minimum_pair_psr_advantage=3.0,
            minimum_pair_confidence_ratio=1.5,
            long_pair_confidence_penalty=0.5,
            minimum_growth_overlap_support=4.0,
            minimum_growth_overlap_area_km2=4.0,
        )
        analysis_config = AnalysisConfig(
            execution_mode="operational",
            operational_calibration_id=calibration_id,
            motion_increment_scale_mps=2.0,
            causal_support_uncertainty_m=1000.0,
            amplitude_displacement_tolerance_m=1000.0,
            maximum_latest_detected_error_std=10.0,
            minimum_local_verification_precision=0.01,
            maximum_local_analysis_verification_error_dbz=6.0,
            maximum_unresolved_amplitude_fraction=1.0,
            minimum_amplitude_total_quality_weight=0.001,
            minimum_amplitude_effective_pixel_count=1.0,
            amplitude_information_policy="operational_fallback",
            minimum_integrated_echo_ratio_for_confidence=0.01,
            maximum_integrated_echo_ratio_for_confidence=100.0,
            minimum_soft_echo_area_ratio_for_confidence=0.01,
            maximum_soft_echo_area_ratio_for_confidence=100.0,
            maximum_established_excess_growth_fraction_for_confidence=1.0,
            minimum_object_count_ratio_for_confidence=0.75,
            amplitude_confidence_policy="operational_fallback",
            observation_common_bias_std_dbz=0.0,
            observation_common_bias_scope="per_frame",
            observation_common_bias_tile_size_px=0,
        )
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:3857",
            grid_hash="0" * 64,
        )
        profile_digest = operational_runtime_profile_digest(
            nowcast_config,
            grid,
            analysis_config=(
                asdict(analysis_config) if variational else None
            ),
        )
        data_identity = OperationalDataIdentity(
            radar_class="test-radar-class",
            qc_pipeline_digest="3" * 64,
            observation_error_model_digest="4" * 64,
            background_model_digest="5" * 64,
        )
        manifest = OperationalCalibrationManifest(
            calibration_id=calibration_id,
            profile_kind="p1" if variational else "p0",
            expected_runtime_profile_digest=profile_digest,
            expected_algorithm_bundle_digest=algorithm_bundle_digest(),
            calibration_dataset_digest="1" * 64,
            validation_dataset_digest="2" * 64,
            data_identity=data_identity,
            training_period=(
                "2025-01-01T00:00:00Z",
                "2025-07-01T00:00:00Z",
            ),
            validation_period=(
                "2025-07-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
            validation_case_count=20,
            validation_regimes=(
                CalibrationRegime("convective", 12),
                CalibrationRegime("stratiform", 8),
            ),
            validation_metrics=(
                CalibrationMetric(
                    name="csi_35",
                    definition_digest="6" * 64,
                    direction="maximize",
                    acceptance_threshold=0.4,
                    value=0.5,
                ),
            ),
        )
        manifest_path = directory / "operational-calibration.json"
        manifest_path.write_text(manifest.json, encoding="utf-8")
        arguments = [
            "--mode",
            "operational",
            "--operational-calibration-id",
            calibration_id,
            "--operational-calibration-manifest",
            str(manifest_path),
            "--approved-operational-calibration-manifest-digest",
            manifest.digest,
            "--radar-class",
            data_identity.radar_class,
            "--qc-pipeline-digest",
            data_identity.qc_pipeline_digest,
            "--observation-error-model-digest",
            data_identity.observation_error_model_digest,
            "--background-model-digest",
            data_identity.background_model_digest,
            "--valid-times",
            "2026-07-31T00:00:00Z",
            "2026-07-31T00:10:00Z",
            "2026-07-31T00:20:00Z",
            "--dx-m",
            "1000",
            "--dy-m",
            "1000",
            "--projection",
            "EPSG:3857",
            "--grid-hash",
            "0" * 64,
            "--maximum-motion-speed-mps",
            "100",
            "--minimum-phase-correlation-psr",
            "0",
            "--pair-echo-dilation-m",
            "3000",
            "--phase-correlation-sidelobe-radius-m",
            "2000",
            "--long-pair-confidence-penalty",
            "0.5",
            "--maximum-pair-velocity-disagreement-mps",
            "10",
            "--maximum-pair-growth-disagreement",
            "0.0953",
            "--maximum-local-growth-log-error-per-step",
            "0.4055",
            "--minimum-pair-psr-advantage",
            "3",
            "--minimum-pair-confidence-ratio",
            "1.5",
            "--minimum-growth-overlap-support",
            "4",
            "--minimum-growth-overlap-area-km2",
            "4",
            "--minimum-publish-verified-support",
            "0.95",
            "--minimum-publish-confidence",
            "0.5",
            "--minimum-publish-observation-verified-support",
            "0.95",
            "--maximum-publish-background-fraction",
            "0.25",
            "--forecast-velocity-uncertainty-mps",
            "1",
            "--forecast-confidence-length-scale-m",
            str(forecast_confidence_length_scale_m),
            "--forecast-log-growth-uncertainty-per-step",
            "0.05",
            "--forecast-log-growth-confidence-scale",
            "1",
            "--single-pair-uncertainty-multiplier",
            "2",
            "--persistence-uncertainty-multiplier",
            "4",
            "--background-tendency-age-uncertainty-scale-minutes",
            "60",
        ]
        if variational:
            arguments = ["--variational", *arguments]
            arguments.extend(
                (
                    "--motion-increment-scale-mps",
                    "2",
                    "--p1-motion-saturation-safe-margin-mps",
                    "2",
                    "--p1-growth-saturation-safe-margin-per-step",
                    "0.04879",
                    "--p1-posterior-saturation-sigma-multiplier",
                    "2",
                    "--p1-saturation-uncertainty-multiplier",
                    "4",
                    "--causal-support-uncertainty-m",
                    "1000",
                    "--amplitude-displacement-tolerance-m",
                    "1000",
                    "--observation-std-dbz",
                    "2",
                    "--observation-common-bias-std-dbz",
                    "0",
                    "--observation-common-bias-scope",
                    "per_frame",
                    "--observation-common-bias-tile-size-px",
                    "0",
                    "--maximum-detected-error-std",
                    "10",
                    "--minimum-local-verification-precision",
                    "0.01",
                    "--maximum-local-analysis-verification-error-dbz",
                    "6",
                    "--maximum-unresolved-amplitude-fraction",
                    "1",
                    "--minimum-amplitude-total-quality-weight",
                    "0.001",
                    "--minimum-amplitude-effective-pixel-count",
                    "1",
                    "--minimum-integrated-echo-ratio-for-confidence",
                    "0.01",
                    "--maximum-integrated-echo-ratio-for-confidence",
                    "100",
                    "--minimum-soft-echo-area-ratio-for-confidence",
                    "0.01",
                    "--maximum-soft-echo-area-ratio-for-confidence",
                    "100",
                    "--maximum-established-excess-growth-fraction-for-confidence",
                    "1",
                    "--minimum-object-count-ratio-for-confidence",
                    "0.75",
                )
            )
        return tuple(arguments)

    def _assert_common_status_fields(self, result: np.lib.npyio.NpzFile) -> None:
        expected = {
            "output_contract_version",
            "forecast_run_artifact_version",
            "forecast_run_digest",
            "nowcast_config_json",
            "input_bundle_digest",
            "grid_time_contract_present",
            "grid_time_contract_json",
            "grid_time_contract_digest",
            "displacement_mps_yx",
            "grid_velocity_mps_yx",
            "projected_velocity_mps_xy",
            "analysis_config_present",
            "analysis_config_json",
            "analysis_config_digest",
            "analysis_input_digest",
            "valid_mask",
            "state_echo_linear",
            "source_support",
            "path_verified_source_support",
            "verified_source_support",
            "local_motion_verified_support",
            "local_growth_verified_support",
            "local_dynamics_verified_support",
            "observation_verified_source_support",
            "background_verified_source_support",
            "forecast_path_verified_support",
            "forecast_verified_support",
            "forecast_local_motion_verified_support",
            "forecast_local_growth_verified_support",
            "forecast_local_dynamics_verified_support",
            "forecast_observation_verified_support",
            "forecast_background_verified_support",
            "forecast_velocity_uncertainty_mps",
            "motion_evidence_uncertainty_multiplier",
            "growth_evidence_uncertainty_multiplier",
            "forecast_position_uncertainty_m",
            "forecast_log_growth_uncertainty",
            "maximum_growth_saturation_excess",
            "posterior_velocity_uncertainty_mps",
            "posterior_log_growth_uncertainty_per_step",
            "p1_velocity_saturation_uncertainty_mps",
            "p1_log_growth_saturation_uncertainty_per_step",
            "forecast_confidence",
            "radar_anchored_valid_mask",
            "radar_state_anchored_valid_mask",
            "radar_dynamics_anchored_valid_mask",
            "background_dynamics_mask",
            "background_fallback_mask",
            "latest_frame_dbz",
            "latest_background_dbz",
            "latest_observation_mask",
            "data_status",
            "coverage_by_frame",
            "background_used",
            "background_contribution_fraction",
            "background_state_support_fraction",
            "observation_state_support_fraction",
            "background_tendency_used",
            "background_age_minutes",
            "minimum_phase_correlation_psr",
            "tendency_pair_count",
            "motion_pair_count",
            "growth_pair_count",
            "motion_pair_selection",
            "growth_pair_selection",
            "motion_pair_conflict",
            "growth_pair_conflict",
            "motion_disagreement_mps",
            "tendency_source",
            "dynamics_source",
            "state_path_source",
            "state_path_mode",
            "state_path_pair_count",
            "state_path_minimum_psr",
            "state_path_conflict",
            "state_path_extrapolated",
            "state_path_age_minutes",
            "observation_path_mode",
            "observation_path_pair_count",
            "observation_path_minimum_psr",
            "observation_path_conflict",
            "observation_path_extrapolated",
            "observation_path_age_minutes",
            "background_path_mode",
            "background_path_pair_count",
            "background_path_minimum_psr",
            "background_path_conflict",
            "background_path_extrapolated",
            "background_path_age_minutes",
            "minimum_growth_overlap_support",
            "minimum_growth_overlap_area_km2",
            "operational_runtime_profile_digest",
            "operational_calibration_manifest_present",
            "operational_calibration_manifest_json",
            "operational_calibration_manifest_digest",
            "operational_calibration_approval_digest",
            "operational_data_identity_present",
            "operational_data_identity_json",
            "operational_data_identity_digest",
            "min_publish_support",
            "minimum_publish_verified_support",
            "minimum_publish_confidence",
            "minimum_publish_observation_verified_support",
            "maximum_publish_background_fraction",
            "analysis_converged",
            "analysis_outer_converged",
            "analysis_final_linearization_stationary",
            "analysis_final_robust_stationary",
            "analysis_final_irls_fixed_point",
            "analysis_p1_forecast_eligible",
            "analysis_posterior_eligible",
            "analysis_fso_eligible",
            "analysis_degraded",
            "analysis_used_fallback",
            "analysis_reason",
        }
        self.assertTrue(expected.issubset(result.files))

    def test_normal_p0_output_contains_operational_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = self._run_cli(
                Path(temporary),
                self._stationary_frames(),
            )

            with np.load(output_path, allow_pickle=False) as result:
                self._assert_common_status_fields(result)
                self.assertEqual(
                    result["output_contract_version"].item(),
                    "nowcast-npz-v72",
                )
                self.assertEqual(
                    result["forecast_run_artifact_version"].item(),
                    "forecast-run-v66",
                )
                self.assertEqual(result["data_status"].item(), "OBSERVED")
                self.assertEqual(result["forecast_dbz"].shape, (18, 8, 8))
                self.assertTrue(np.isfinite(result["forecast_dbz"]).all())
                np.testing.assert_array_equal(
                    result["coverage_by_frame"],
                    np.ones(3),
                )
                self.assertTrue(np.isnan(result["background_age_minutes"].item()))
                self.assertFalse(result["background_used"].item())
                self.assertFalse(result["background_tendency_used"].item())
                self.assertEqual(
                    result["background_contribution_fraction"].item(),
                    0.0,
                )
                self.assertEqual(
                    result["tendency_source"].item(),
                    "OBSERVATION",
                )
                self.assertEqual(result["min_publish_support"].item(), 0.95)
                self.assertEqual(result["tendency_pair_count"].item(), 2)
                self.assertEqual(result["motion_pair_count"].item(), 2)
                self.assertEqual(result["growth_pair_count"].item(), 2)
                self.assertEqual(
                    result["motion_pair_selection"].item(),
                    "BLENDED",
                )
                self.assertEqual(
                    result["growth_pair_selection"].item(),
                    "BLENDED",
                )
                self.assertFalse(result["motion_pair_conflict"].item())
                self.assertFalse(result["growth_pair_conflict"].item())
                self.assertGreaterEqual(
                    result["minimum_phase_correlation_psr"].item(),
                    8.0,
                )
                self.assertFalse(result["analysis_converged"].item())
                self.assertFalse(result["analysis_outer_converged"].item())
                self.assertFalse(
                    result["analysis_final_linearization_stationary"].item()
                )
                self.assertFalse(result["analysis_fso_eligible"].item())
                self.assertFalse(result["analysis_config_present"].item())
                self.assertFalse(result["analysis_degraded"].item())
                self.assertFalse(result["analysis_used_fallback"].item())
                self.assertEqual(result["analysis_reason"].item(), "not_requested")
                self.assertNotIn("forecast_corrected_count", result.files)
                self.assertNotIn(
                    "echo_integral_before_transport",
                    result.files,
                )
            loaded = load_forecast_run(output_path)
            loaded.validate_issuance()
            self.assertEqual(loaded.forecast_dbz.shape, (18, 8, 8))
            self.assertEqual(
                loaded.run.config.min_publish_support,
                0.95,
            )

    def test_cli_persists_grid_time_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = self._run_cli(
                Path(temporary),
                self._stationary_frames(),
                "--valid-times",
                "2026-07-31T09:00:00+09:00",
                "2026-07-31T09:10:00+09:00",
                "2026-07-31T09:20:00+09:00",
                "--dx-m",
                "1000",
                "--dy-m",
                "1000",
                "--projection",
                "EPSG:5179",
                "--grid-hash",
                "d" * 64,
                "--pixel-to-projected-matrix-m",
                "0",
                "-1000",
                "1000",
                "0",
                "--maximum-motion-speed-mps",
                "30",
                "--maximum-pair-motion-disagreement-px",
                "3.5",
                "--maximum-pair-velocity-disagreement-mps",
                "7.5",
                "--maximum-pair-growth-disagreement",
                "0.08",
                "--minimum-pair-psr-advantage",
                "4.5",
                "--minimum-pair-confidence-ratio",
                "1.8",
            )

            loaded = load_forecast_run(output_path)
            contract = loaded.run.grid_time_contract
            self.assertIsNotNone(contract)
            assert contract is not None
            self.assertEqual(
                contract.valid_times,
                (
                    "2026-07-31T00:00:00Z",
                    "2026-07-31T00:10:00Z",
                    "2026-07-31T00:20:00Z",
                ),
            )
            self.assertEqual(contract.dx_m, 1000.0)
            self.assertEqual(contract.projection, "EPSG:5179")
            self.assertEqual(
                contract.pixel_to_projected_matrix_m,
                ((0.0, -1000.0), (1000.0, 0.0)),
            )
            self.assertEqual(
                loaded.run.config.maximum_motion_speed_mps,
                30.0,
            )
            self.assertEqual(
                loaded.run.config.maximum_pair_motion_disagreement_px,
                3.5,
            )
            self.assertEqual(
                loaded.run.config.maximum_pair_velocity_disagreement_mps,
                7.5,
            )
            self.assertEqual(
                loaded.run.config.maximum_pair_growth_disagreement,
                0.08,
            )
            self.assertEqual(
                loaded.run.config.minimum_pair_psr_advantage,
                4.5,
            )
            self.assertEqual(
                loaded.run.config.minimum_pair_confidence_ratio,
                1.8,
            )

    def test_variational_output_records_feasibility_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frames = np.full((3, 8, 8), 20.0, dtype=np.float32)
            frames[1] = 19.0
            output_path = self._run_cli(
                Path(temporary),
                frames,
                "--variational",
            )

            with np.load(output_path, allow_pickle=False) as result:
                self.assertTrue(result["analysis_config_present"].item())
                self.assertTrue(result["analysis_config_json"].item())
                self.assertEqual(
                    len(result["analysis_config_digest"].item()),
                    64,
                )
                self.assertEqual(
                    len(result["analysis_input_digest"].item()),
                    64,
                )
                self.assertFalse(result["analysis_used_fallback"].item())
                self.assertTrue(
                    result["analysis_final_linearization_stationary"].item()
                )
                self.assertTrue(result["analysis_fso_eligible"].item())
                self.assertEqual(
                    result["dynamics_source"].item(),
                    "P1_VARIATIONAL",
                )
                self.assertEqual(result["state_path_source"].item(), "NONE")
                self.assertEqual(result["state_path_mode"].item(), "NONE")
                self.assertTrue(
                    np.isnan(result["minimum_growth_overlap_support"].item())
                )
                self.assertLess(
                    result["analysis_final_objective"].item(),
                    result["analysis_initial_objective"].item(),
                )
                self.assertEqual(
                    result["analysis_unresolved_amplitude_fraction"].item(),
                    0.0,
                )
                np.testing.assert_array_equal(
                    result[
                        "analysis_unresolved_amplitude_fraction_by_time"
                    ],
                    np.zeros(2),
                )
                np.testing.assert_array_equal(
                    result["analysis_amplitude_violation_score_by_time"],
                    np.zeros(2),
                )
                np.testing.assert_array_equal(
                    result[
                        "analysis_effective_precursor_pixel_count_by_time"
                    ],
                    np.zeros(2),
                )
                np.testing.assert_array_equal(
                    result[
                        "analysis_amplitude_information_sufficient_by_time"
                    ],
                    np.ones(2, dtype=np.bool_),
                )
                self.assertFalse(
                    result[
                        "analysis_insufficient_amplitude_information"
                    ].item()
                )
                np.testing.assert_array_equal(
                    result[
                        "analysis_established_echo_excess_growth_fraction_by_time"
                    ],
                    np.zeros(2),
                )
                self.assertEqual(
                    result[
                        "analysis_established_echo_excess_growth_fraction"
                    ].item(),
                    0.0,
                )

                self.assertEqual(
                    result[
                        "analysis_maximum_growth_envelope_ratio_by_time"
                    ].shape,
                    (2,),
                )
                self.assertLessEqual(
                    result[
                        "analysis_maximum_growth_envelope_ratio"
                    ].item(),
                    1.0,
                )
                np.testing.assert_array_equal(
                    result[
                        "analysis_displacement_tolerant_soft_echo_area_ratio_by_time"
                    ],
                    np.full(2, np.nan),
                )
                self.assertEqual(
                    result["analysis_amplitude_diagnostics_source"].item(),
                    "returned_analysis",
                )
                self.assertEqual(
                    result["analysis_relative_objective_reduction"].shape,
                    (),
                )
                self.assertEqual(
                    result["analysis_causal_control_cell_count"].item(),
                    0,
                )
                self.assertEqual(
                    result["analysis_causal_seed_cell_count"].item(),
                    0,
                )
                self.assertEqual(
                    result["analysis_causal_seed_prior_cost"].item(),
                    0.0,
                )
                eigenvalues = result[
                    "analysis_regularized_dynamics_hessian_eigenvalues"
                ]
                self.assertEqual(eigenvalues.shape, (3,))
                self.assertTrue(np.isfinite(eigenvalues).all())
                self.assertTrue(np.all(eigenvalues >= 1.0))
                self.assertTrue(
                    np.isfinite(
                        result[
                            "analysis_regularized_dynamics_hessian_condition_number"
                        ]
                    )
                )
                self.assertTrue(
                    np.isfinite(
                        result["analysis_field_growth_jacobian_cosine"]
                    )
                )
                motion_cosines = result[
                    "analysis_field_motion_jacobian_cosine_by_control"
                ]
                self.assertEqual(motion_cosines.shape, (2,))
                self.assertTrue(np.isfinite(motion_cosines).all())
                data_eigenvalues = result[
                    "analysis_dynamics_data_gram_eigenvalues"
                ]
                self.assertEqual(data_eigenvalues.shape, (3,))
                self.assertTrue(np.isfinite(data_eigenvalues).all())
                self.assertTrue(np.all(data_eigenvalues >= 0.0))
                self.assertGreater(
                    result["analysis_dynamics_data_information_trace"].item(),
                    0.0,
                )
                self.assertGreaterEqual(
                    result["analysis_dynamics_data_numerical_rank"].item(),
                    1,
                )
                self.assertGreaterEqual(
                    result["analysis_field_smoothness_prior_cost"].item(),
                    0.0,
                )
                self.assertEqual(
                    result["analysis_motion_saturation_margin_yx"].shape,
                    (2,),
                )
                self.assertGreaterEqual(
                    result["analysis_growth_saturation_margin"].item(),
                    0.0,
                )
                self.assertTrue(
                    np.isnan(
                        result[
                            "analysis_motion_speed_saturation_margin_mps"
                        ].item()
                    )
                )
            loaded = load_forecast_run(output_path)
            loaded.validate_issuance()
            self.assertEqual(
                loaded.metadata.dynamics_source.value,
                "P1_VARIATIONAL",
            )
            self.assertTrue(
                torch.isnan(
                    loaded.metadata.posterior_velocity_uncertainty_mps
                )
            )
            self.assertTrue(
                torch.isnan(
                    loaded.metadata.p1_velocity_saturation_uncertainty_mps
                )
            )
            self.assertFalse(bool(torch.any(loaded.forecast_confidence)))
            self.assertTrue(
                np.isnan(loaded.metadata.minimum_growth_overlap_support)
            )
            self.assertIsNotNone(loaded.run.analysis_config_json)
            self.assertIsNotNone(loaded.run.analysis_config_digest)
            self.assertIsNotNone(loaded.run.analysis_input_digest)

    def test_variational_cli_binds_common_bias_group_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            groups = np.zeros((8, 8), dtype=np.int32)
            groups[:, 4:] = 10
            groups[4:, :] += 20
            group_path = directory / "groups.npy"
            np.save(group_path, groups, allow_pickle=False)
            output_path = self._run_cli(
                directory,
                np.full((3, 8, 8), 20.0, dtype=np.float32),
                "--variational",
                "--observation-common-bias-std-dbz",
                "0.2",
                "--observation-common-bias-scope",
                "all_times",
                "--observation-common-bias-group-map",
                str(group_path),
            )
            expected_digest = observation_common_bias_group_map_digest(
                torch.as_tensor(groups, dtype=torch.long),
                temporal_scope="all_times",
            )
            with np.load(output_path, allow_pickle=False) as result:
                config = json.loads(result["analysis_config_json"].item())
                self.assertEqual(
                    config["observation_common_bias_group_map_digest"],
                    expected_digest,
                )
                self.assertEqual(
                    config["observation_common_bias_tile_size_px"],
                    0,
                )
                self.assertEqual(len(result["analysis_input_digest"].item()), 64)

    def test_variational_cli_binds_overlapping_common_bias_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            x_fraction = np.linspace(0.0, 1.0, 8, dtype=np.float32)
            mode_weights = np.stack(
                (
                    np.broadcast_to(np.sqrt(1.0 - x_fraction), (8, 8)),
                    np.broadcast_to(np.sqrt(x_fraction), (8, 8)),
                )
            ).copy()
            mode_path = directory / "mode-weights.npy"
            np.save(mode_path, mode_weights, allow_pickle=False)
            output_path = self._run_cli(
                directory,
                np.full((3, 8, 8), 20.0, dtype=np.float32),
                "--variational",
                "--observation-common-bias-std-dbz",
                "0.2",
                "--observation-common-bias-mode-weights",
                str(mode_path),
            )
            expected_digest = observation_common_bias_mode_weights_digest(
                torch.as_tensor(mode_weights)
            )
            with np.load(output_path, allow_pickle=False) as result:
                config = json.loads(result["analysis_config_json"].item())
                self.assertEqual(
                    config["observation_common_bias_mode_weights_digest"],
                    expected_digest,
                )
                self.assertIsNone(
                    config["observation_common_bias_group_map_digest"]
                )

    def test_variational_cli_rejects_modes_before_materializing_them(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            mode_path = directory / "mode-weights.npy"
            np.save(
                mode_path,
                np.ones((2, 8, 8), dtype=np.float32),
                allow_pickle=False,
            )
            with (
                patch(
                    "advar.cli.estimate_common_bias_resources"
                ) as estimate,
                patch(
                    "advar.cli.np.array",
                    side_effect=AssertionError("mode array was materialized"),
                ),
                self.assertRaisesRegex(ValueError, "preflight rejected"),
            ):
                estimate.return_value.within_budget = False
                estimate.return_value.rejection_reasons = (
                    "whitener_operations_per_apply",
                )
                self._run_cli(
                    directory,
                    np.full((3, 8, 8), 20.0, dtype=np.float32),
                    "--variational",
                    "--observation-common-bias-std-dbz",
                    "0.2",
                    "--observation-common-bias-mode-weights",
                    str(mode_path),
                )

            estimate.assert_called_once_with(
                (2, 8, 8),
                (3, 8, 8),
                dtype=torch.float32,
                temporal_scope="per_frame",
            )

    def test_cli_records_operational_amplitude_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = self._run_cli(
                Path(temporary),
                self._stationary_frames(),
                "--variational",
                "--amplitude-information-policy",
                "operational_fallback",
            )

            with np.load(output_path, allow_pickle=False) as result:
                config = json.loads(result["analysis_config_json"].item())
                self.assertEqual(
                    config["amplitude_information_policy"],
                    "operational_fallback",
                )

    def test_operational_mode_requires_explicit_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                self._run_cli(
                    Path(temporary),
                    self._stationary_frames(),
                    "--variational",
                    "--mode",
                    "operational",
                )

    def test_operational_mode_requires_manifest_with_matching_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arguments = list(self._operational_profile_arguments(directory))
            manifest_position = arguments.index(
                "--operational-calibration-manifest"
            )
            without_manifest = arguments.copy()
            del without_manifest[manifest_position : manifest_position + 2]
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                self._run_cli(
                    directory,
                    self._stationary_frames(),
                    *without_manifest,
                )

            approval_position = arguments.index(
                "--approved-operational-calibration-manifest-digest"
            )
            without_approval = arguments.copy()
            del without_approval[approval_position : approval_position + 2]
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                self._run_cli(
                    directory,
                    self._stationary_frames(),
                    *without_approval,
                )

            identifier_position = arguments.index(
                "--operational-calibration-id"
            )
            arguments[identifier_position + 1] = "wrong-calibration"
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                self._run_cli(
                    directory,
                    self._stationary_frames(),
                    *arguments,
                )

    def test_operational_manifest_rejects_changed_runtime_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arguments = list(self._operational_profile_arguments(directory))
            scale_position = arguments.index(
                "--forecast-confidence-length-scale-m"
            )
            arguments[scale_position + 1] = "9000"

            with self.assertRaisesRegex(
                ValueError,
                "runtime profile digest mismatch",
            ):
                self._run_cli(
                    directory,
                    self._stationary_frames(),
                    *arguments,
                )

    def test_operational_mode_requires_pair_calibration(self) -> None:
        required = (
            "--pair-echo-dilation-m",
            "--phase-correlation-sidelobe-radius-m",
            "--maximum-pair-velocity-disagreement-mps",
            "--maximum-pair-growth-disagreement",
            "--minimum-pair-psr-advantage",
            "--minimum-pair-confidence-ratio",
            "--minimum-growth-overlap-support",
            "--minimum-growth-overlap-area-km2",
            "--minimum-publish-verified-support",
            "--minimum-publish-confidence",
            "--minimum-publish-observation-verified-support",
            "--maximum-publish-background-fraction",
            "--forecast-velocity-uncertainty-mps",
            "--forecast-confidence-length-scale-m",
            "--forecast-log-growth-uncertainty-per-step",
            "--forecast-log-growth-confidence-scale",
            "--minimum-local-verification-precision",
            "--maximum-local-analysis-verification-error-dbz",
            "--minimum-object-count-ratio-for-confidence",
            "--observation-common-bias-std-dbz",
            "--observation-common-bias-scope",
            "--observation-common-bias-tile-size-px",
        )
        for name in required:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    arguments = list(
                        self._operational_profile_arguments(directory)
                    )
                    position = arguments.index(name)
                    del arguments[position : position + 2]
                    with self.assertRaises(SystemExit), redirect_stderr(
                        io.StringIO()
                    ):
                        self._run_cli(
                            Path(temporary),
                            self._stationary_frames(),
                            *arguments,
                        )

    def test_operational_mode_records_complete_fail_closed_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output_path = self._run_cli(
                directory,
                self._stationary_frames(),
                *self._operational_profile_arguments(directory),
            )

            with np.load(output_path, allow_pickle=False) as result:
                config = json.loads(result["analysis_config_json"].item())
                nowcast_config = json.loads(
                    result["nowcast_config_json"].item()
                )
                self.assertEqual(config["execution_mode"], "operational")
                self.assertEqual(
                    config["operational_calibration_id"],
                    "test-calibration-v1",
                )
                self.assertEqual(
                    len(result["operational_runtime_profile_digest"].item()),
                    64,
                )
                manifest = OperationalCalibrationManifest.from_json(
                    result["operational_calibration_manifest_json"].item()
                )
                self.assertTrue(
                    result["operational_calibration_manifest_present"].item()
                )
                self.assertEqual(
                    result["operational_calibration_manifest_digest"].item(),
                    manifest.digest,
                )
                self.assertEqual(
                    result["operational_calibration_approval_digest"].item(),
                    manifest.digest,
                )
                self.assertEqual(
                    result["operational_data_identity_digest"].item(),
                    manifest.data_identity.digest,
                )
                self.assertEqual(manifest.calibration_id, "test-calibration-v1")
                self.assertEqual(
                    config["amplitude_information_policy"],
                    "operational_fallback",
                )
                self.assertEqual(
                    config["amplitude_confidence_policy"],
                    "operational_fallback",
                )
                self.assertEqual(
                    config["minimum_object_count_ratio_for_confidence"],
                    0.75,
                )
                self.assertEqual(config["motion_increment_scale_mps"], 2.0)
                self.assertEqual(
                    config["observation_common_bias_std_dbz"],
                    0.0,
                )
                self.assertEqual(
                    config["observation_common_bias_scope"],
                    "per_frame",
                )
                self.assertEqual(
                    config["observation_common_bias_tile_size_px"],
                    0,
                )
                self.assertEqual(
                    nowcast_config["minimum_growth_overlap_support"],
                    4.0,
                )
                self.assertEqual(
                    nowcast_config["minimum_growth_overlap_area_km2"],
                    4.0,
                )
                self.assertEqual(
                    nowcast_config["minimum_publish_verified_support"],
                    0.95,
                )
                self.assertEqual(
                    nowcast_config["minimum_publish_confidence"],
                    0.5,
                )
                self.assertEqual(
                    nowcast_config[
                        "minimum_publish_observation_verified_support"
                    ],
                    0.95,
                )
                self.assertEqual(
                    nowcast_config["maximum_publish_background_fraction"],
                    0.25,
                )
                self.assertEqual(
                    nowcast_config["forecast_velocity_uncertainty_mps"],
                    1.0,
                )
                self.assertEqual(
                    nowcast_config["forecast_confidence_length_scale_m"],
                    10_000.0,
                )
                self.assertEqual(
                    nowcast_config[
                        "forecast_log_growth_uncertainty_per_step"
                    ],
                    0.05,
                )
                self.assertEqual(
                    nowcast_config[
                        "p1_posterior_saturation_sigma_multiplier"
                    ],
                    2.0,
                )
                self.assertEqual(
                    nowcast_config["forecast_log_growth_confidence_scale"],
                    1.0,
                )
                self.assertEqual(
                    nowcast_config["single_pair_uncertainty_multiplier"],
                    2.0,
                )
                self.assertEqual(
                    nowcast_config["persistence_uncertainty_multiplier"],
                    4.0,
                )
                self.assertEqual(
                    nowcast_config[
                        "background_tendency_age_uncertainty_scale_minutes"
                    ],
                    60.0,
                )
                self.assertEqual(
                    result["analysis_motion_control_coordinate_system"].item(),
                    "projected_xy_mps_radial_ball",
                )
                self.assertEqual(
                    result[
                        "analysis_field_smoothness_coordinate_system"
                    ].item(),
                    "projected_orthogonal_graph",
                )
                self.assertIn(
                    "analysis_amplitude_confidence_failed",
                    result.files,
                )
                self.assertIn(
                    "analysis_dynamics_data_effective_dimension",
                    result.files,
                )
                self.assertIn(
                    "analysis_dynamics_data_to_prior_ratio_by_mode",
                    result.files,
                )
                self.assertIn(
                    "analysis_field_conditioned_dynamics_data_gram_eigenvalues",
                    result.files,
                )
                self.assertIn(
                    "analysis_field_conditioned_dynamics_data_effective_dimension",
                    result.files,
                )
                if result["dynamics_source"].item() == "P1_VARIATIONAL":
                    self.assertTrue(
                        np.isfinite(
                            result[
                                "posterior_velocity_uncertainty_mps"
                            ].item()
                        )
                    )
                    self.assertTrue(
                        np.isfinite(
                            result[
                                "posterior_log_growth_uncertainty_per_step"
                            ].item()
                        )
                    )
                    self.assertTrue(
                        np.isfinite(
                            result[
                                "p1_velocity_saturation_uncertainty_mps"
                            ].item()
                        )
                    )
                    self.assertTrue(
                        np.isfinite(
                            result[
                                "p1_log_growth_saturation_uncertainty_per_step"
                            ].item()
                        )
                    )

            loaded = load_forecast_run(output_path)
            loaded.validate_issuance()
            self.assertEqual(
                loaded.run.operational_calibration_manifest_json,
                manifest.json,
            )
            self.assertEqual(
                loaded.run.operational_calibration_manifest_digest,
                manifest.digest,
            )
            self.assertEqual(
                loaded.run.operational_calibration_approval_digest,
                manifest.digest,
            )
            self.assertEqual(
                loaded.run.operational_data_identity_digest,
                manifest.data_identity.digest,
            )

    def test_operational_p0_profile_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output_path = self._run_cli(
                directory,
                self._stationary_frames(),
                *self._operational_profile_arguments(
                    directory,
                    variational=False,
                ),
            )

            with np.load(output_path, allow_pickle=False) as result:
                self.assertFalse(result["analysis_used"].item())
                self.assertFalse(result["analysis_config_present"].item())
                manifest = OperationalCalibrationManifest.from_json(
                    result["operational_calibration_manifest_json"].item()
                )
                self.assertEqual(manifest.profile_kind, "p0")
                self.assertEqual(
                    result["operational_calibration_approval_digest"].item(),
                    manifest.digest,
                )
                self.assertEqual(
                    result["operational_data_identity_digest"].item(),
                    manifest.data_identity.digest,
                )

            loaded = load_forecast_run(output_path)
            loaded.validate_issuance()
            self.assertIsNone(loaded.run.analysis_config_json)
            self.assertEqual(
                loaded.run.operational_calibration_approval_digest,
                manifest.digest,
            )

    def test_operational_runtime_profile_tracks_content_not_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("base", "renamed", "changed"):
                (root / name).mkdir()
            profiles = (
                self._operational_profile_arguments(root / "base"),
                self._operational_profile_arguments(
                    root / "renamed",
                    calibration_id="renamed-profile",
                ),
                self._operational_profile_arguments(
                    root / "changed",
                    forecast_confidence_length_scale_m=9000.0,
                ),
            )
            outputs = (
                self._run_cli(
                    root / "base",
                    self._stationary_frames(),
                    *profiles[0],
                ),
                self._run_cli(
                    root / "renamed",
                    self._stationary_frames(),
                    *profiles[1],
                ),
                self._run_cli(
                    root / "changed",
                    self._stationary_frames(),
                    *profiles[2],
                ),
            )
            digests = []
            manifest_digests = []
            for output in outputs:
                with np.load(output, allow_pickle=False) as result:
                    digests.append(
                        result["operational_runtime_profile_digest"].item()
                    )
                    manifest_digests.append(
                        result[
                            "operational_calibration_manifest_digest"
                        ].item()
                    )
            base_run = load_forecast_run(outputs[0]).run

        self.assertRegex(digests[0], r"^[0-9a-f]{64}$")
        self.assertEqual(digests[0], digests[1])
        self.assertNotEqual(digests[0], digests[2])
        self.assertNotEqual(manifest_digests[0], manifest_digests[1])
        self.assertNotEqual(manifest_digests[0], manifest_digests[2])

        grid = base_run.grid_time_contract
        assert grid is not None
        shifted_grid = replace(
            grid,
            valid_times=(
                "2026-08-01T00:00:00Z",
                "2026-08-01T00:10:00Z",
                "2026-08-01T00:20:00Z",
            ),
        )
        shifted_run = replace(
            base_run,
            grid_time_contract=shifted_grid,
            grid_time_contract_digest=shifted_grid.digest,
        )
        self.assertEqual(
            base_run.operational_runtime_profile_digest,
            shifted_run.operational_runtime_profile_digest,
        )
        different_grid = replace(grid, grid_hash="1" * 64)
        different_grid_run = replace(
            base_run,
            grid_time_contract=different_grid,
            grid_time_contract_digest=different_grid.digest,
        )
        self.assertNotEqual(
            base_run.operational_runtime_profile_digest,
            different_grid_run.operational_runtime_profile_digest,
        )

    def test_all_qc_rejected_uses_stale_background(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frames = self._stationary_frames()
            qc_path = directory / "qc.npy"
            background_path = directory / "background.npy"
            np.save(qc_path, np.zeros_like(frames, dtype=np.bool_))
            np.save(background_path, frames - 2.0)

            output_path = self._run_cli(
                directory,
                frames,
                "--qc-mask",
                str(qc_path),
                "--background",
                str(background_path),
                "--background-age-minutes",
                "10",
            )

            with np.load(output_path, allow_pickle=False) as result:
                self._assert_common_status_fields(result)
                self.assertFalse(result["analysis_config_present"].item())
                self.assertEqual(
                    result["data_status"].item(),
                    "STALE_BACKGROUND",
                )
                self.assertTrue(np.isfinite(result["forecast_dbz"]).all())
                np.testing.assert_array_equal(
                    result["coverage_by_frame"],
                    np.zeros(3),
                )
                self.assertEqual(result["background_age_minutes"].item(), 10.0)
                self.assertTrue(result["background_used"].item())
                self.assertTrue(result["background_tendency_used"].item())
                self.assertEqual(
                    result["background_contribution_fraction"].item(),
                    1.0,
                )
                self.assertEqual(
                    result["tendency_source"].item(),
                    "BACKGROUND",
                )

    def test_background_requires_age(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            background_path = directory / "background.npy"
            np.save(background_path, self._stationary_frames())

            with self.assertRaises(SystemExit):
                self._run_cli(
                    directory,
                    self._stationary_frames(),
                    "--background",
                    str(background_path),
                )

    def test_all_qc_rejected_without_background_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frames = self._stationary_frames()
            qc_path = directory / "qc.npy"
            np.save(qc_path, np.zeros_like(frames, dtype=np.bool_))

            output_path = self._run_cli(
                directory,
                frames,
                "--qc-mask",
                str(qc_path),
            )

            with np.load(output_path, allow_pickle=False) as result:
                self._assert_common_status_fields(result)
                self.assertEqual(
                    result["data_status"].item(),
                    "UNAVAILABLE",
                )
                self.assertTrue(np.isnan(result["forecast_dbz"]).all())
                np.testing.assert_array_equal(
                    result["coverage_by_frame"],
                    np.zeros(3),
                )
                self.assertFalse(result["background_used"].item())
                self.assertEqual(result["tendency_source"].item(), "NONE")

    def test_audit_is_optional_and_reuses_the_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch("advar.cli.nowcast", wraps=cli.nowcast) as science:
                output_path = self._run_cli(
                    Path(temporary),
                    self._stationary_frames(),
                    "--audit",
                )

            self.assertEqual(science.call_count, 1)
            with np.load(output_path, allow_pickle=False) as result:
                self.assertEqual(result["forecast_corrected_count"].item(), 0)
                self.assertEqual(
                    result["echo_integral_before_transport"].shape,
                    (18,),
                )
                self.assertTrue(
                    np.allclose(result["echo_budget_error"], 0.0)
                )
            loaded = load_forecast_run(output_path)
            loaded.validate_issuance()

    def test_atomic_save_preserves_existing_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output_path = directory / "forecast.npz"
            original = b"existing-output"
            output_path.write_bytes(original)

            with patch(
                "advar.run_artifact.np.savez_compressed",
                side_effect=RuntimeError("synthetic write failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic write failure",
                ):
                    cli.atomic_savez_compressed(
                        output_path,
                        {"forecast_dbz": np.zeros((1, 2, 2))},
                    )

            self.assertEqual(output_path.read_bytes(), original)
            self.assertEqual(
                list(directory.glob(f".{output_path.name}.*.tmp")),
                [],
            )

    def test_atomic_save_fsyncs_file_and_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forecast.npz"
            with patch("advar.run_artifact.os.fsync") as fsync:
                cli.atomic_savez_compressed(
                    path,
                    {"forecast_dbz": np.zeros((1, 2, 2))},
                )

            self.assertTrue(path.exists())
            self.assertEqual(fsync.call_count, 2)

    def test_cli_diagnostic_tampering_breaks_artifact_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = self._run_cli(
                Path(temporary),
                self._stationary_frames(),
                "--variational",
            )
            with np.load(output_path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["analysis_field_smoothness_prior_cost"] += 1.0
            np.savez_compressed(output_path, **arrays)

            with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                load_forecast_run(output_path)


if __name__ == "__main__":
    unittest.main()
