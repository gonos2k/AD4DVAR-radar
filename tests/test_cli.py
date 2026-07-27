from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar import cli  # noqa: E402


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

    def _assert_common_status_fields(self, result: np.lib.npyio.NpzFile) -> None:
        expected = {
            "output_contract_version",
            "forecast_status",
            "data_coverage_fraction",
            "latest_data_coverage_fraction",
            "background_used",
            "background_age_minutes",
            "analysis_converged",
            "analysis_degraded",
            "analysis_used_fallback",
            "analysis_reason",
            "minimum_echo_linear",
            "negative_echo_count_before_roundoff_fix",
            "negative_echo_integral_before_fix",
            "roundoff_correction_count",
            "roundoff_correction_integral",
            "minimum_transport_weight",
            "maximum_transport_weight",
            "transport_weight_sum_error",
            "echo_integral_before_transport",
            "echo_integral_after_transport",
            "boundary_outflow_integral",
            "echo_budget_error",
            "positivity_gate_passed",
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
                    "nowcast-npz-v2",
                )
                self.assertEqual(result["forecast_status"].item(), "OBSERVED")
                self.assertEqual(result["forecast_dbz"].shape, (18, 8, 8))
                self.assertTrue(np.isfinite(result["forecast_dbz"]).all())
                self.assertEqual(result["data_coverage_fraction"].item(), 1.0)
                self.assertEqual(
                    result["latest_data_coverage_fraction"].item(),
                    1.0,
                )
                self.assertTrue(np.isnan(result["background_age_minutes"].item()))
                self.assertFalse(result["background_used"].item())
                self.assertFalse(result["analysis_converged"].item())
                self.assertFalse(result["analysis_degraded"].item())
                self.assertFalse(result["analysis_used_fallback"].item())
                self.assertEqual(result["analysis_reason"].item(), "not_requested")
                self.assertGreaterEqual(
                    result["minimum_echo_linear"].item(),
                    0.0,
                )
                self.assertEqual(
                    result["negative_echo_count_before_roundoff_fix"].item(),
                    0,
                )
                self.assertEqual(
                    result["negative_echo_integral_before_fix"].item(),
                    0.0,
                )
                self.assertEqual(
                    result["roundoff_correction_count"].item(),
                    0,
                )
                self.assertEqual(
                    result["roundoff_correction_integral"].item(),
                    0.0,
                )
                self.assertEqual(
                    result["minimum_transport_weight"].shape,
                    (18,),
                )
                self.assertEqual(
                    result["echo_integral_before_transport"].shape,
                    (18,),
                )
                self.assertEqual(
                    result["echo_integral_after_transport"].shape,
                    (18,),
                )
                self.assertTrue(
                    np.allclose(result["echo_budget_error"], 0.0)
                )
                self.assertTrue(result["positivity_gate_passed"].item())

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
                self.assertEqual(
                    result["forecast_status"].item(),
                    "STALE_BACKGROUND",
                )
                self.assertTrue(np.isfinite(result["forecast_dbz"]).all())
                self.assertEqual(result["data_coverage_fraction"].item(), 0.0)
                self.assertEqual(
                    result["latest_data_coverage_fraction"].item(),
                    0.0,
                )
                self.assertEqual(result["background_age_minutes"].item(), 10.0)
                self.assertTrue(result["background_used"].item())

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
                    result["forecast_status"].item(),
                    "UNAVAILABLE",
                )
                self.assertTrue(np.isnan(result["forecast_dbz"]).all())
                self.assertEqual(result["data_coverage_fraction"].item(), 0.0)
                self.assertEqual(
                    result["latest_data_coverage_fraction"].item(),
                    0.0,
                )
                self.assertFalse(result["background_used"].item())

    def test_atomic_save_preserves_existing_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output_path = directory / "forecast.npz"
            original = b"existing-output"
            output_path.write_bytes(original)

            with patch(
                "advar.cli.np.savez_compressed",
                side_effect=RuntimeError("synthetic write failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic write failure",
                ):
                    cli._atomic_savez_compressed(
                        output_path,
                        {"forecast_dbz": np.zeros((1, 2, 2))},
                    )

            self.assertEqual(output_path.read_bytes(), original)
            self.assertEqual(
                list(directory.glob(f".{output_path.name}.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
