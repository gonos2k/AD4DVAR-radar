from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar import cli  # noqa: E402
from advar import load_forecast_run  # noqa: E402


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

    def _operational_profile_arguments(self) -> tuple[str, ...]:
        return (
            "--variational",
            "--mode",
            "operational",
            "--operational-calibration-id",
            "test-calibration-v1",
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
            "--motion-increment-scale-mps",
            "2",
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
            "--minimum-pair-psr-advantage",
            "3",
            "--minimum-pair-confidence-ratio",
            "1.5",
            "--minimum-growth-overlap-support",
            "4",
            "--minimum-growth-overlap-area-km2",
            "4",
            "--causal-support-uncertainty-m",
            "1000",
            "--amplitude-displacement-tolerance-m",
            "1000",
            "--observation-std-dbz",
            "2",
            "--maximum-detected-error-std",
            "10",
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
        )

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
            "latest_frame_dbz",
            "latest_background_dbz",
            "latest_observation_mask",
            "data_status",
            "coverage_by_frame",
            "background_used",
            "background_contribution_fraction",
            "background_state_support_fraction",
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
            "minimum_growth_overlap_support",
            "minimum_growth_overlap_area_km2",
            "min_publish_support",
            "analysis_converged",
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
                    "nowcast-npz-v24",
                )
                self.assertEqual(
                    result["forecast_run_artifact_version"].item(),
                    "forecast-run-v16",
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
                np.isnan(loaded.metadata.minimum_growth_overlap_support)
            )
            self.assertIsNotNone(loaded.run.analysis_config_json)
            self.assertIsNotNone(loaded.run.analysis_config_digest)
            self.assertIsNotNone(loaded.run.analysis_input_digest)

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
        )
        for name in required:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    arguments = list(self._operational_profile_arguments())
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
            output_path = self._run_cli(
                Path(temporary),
                self._stationary_frames(),
                *self._operational_profile_arguments(),
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
                    config["amplitude_information_policy"],
                    "operational_fallback",
                )
                self.assertEqual(
                    config["amplitude_confidence_policy"],
                    "operational_fallback",
                )
                self.assertEqual(config["motion_increment_scale_mps"], 2.0)
                self.assertEqual(
                    nowcast_config["minimum_growth_overlap_support"],
                    4.0,
                )
                self.assertEqual(
                    nowcast_config["minimum_growth_overlap_area_km2"],
                    4.0,
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
