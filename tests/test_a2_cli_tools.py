from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import advar.cli as cli


SCRIPT = Path(__file__).parents[1] / ".github/scripts/build_deployment_bundle.py"
SPEC = importlib.util.spec_from_file_location("advar_a2_deployment_bundle", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("deployment bundle script cannot be loaded")
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


class A2CliToolTests(unittest.TestCase):
    @staticmethod
    def _frames() -> np.ndarray:
        frame = np.zeros((16, 16), dtype=np.float32)
        frame[4:8, 4:8] = 30.0
        return np.stack(
            (frame, np.roll(frame, 1, axis=1), np.roll(frame, 2, axis=1))
        )

    @staticmethod
    def _grid_arguments() -> tuple[str, ...]:
        return (
            "--valid-times",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:10:00Z",
            "2026-08-01T00:20:00Z",
            "--dx-m",
            "1000",
            "--dy-m",
            "1000",
            "--projection",
            "EPSG:5179",
            "--grid-hash",
            "0" * 64,
        )

    def _run_cli(
        self,
        root: Path,
        *extra_arguments: str,
    ) -> Path:
        input_path = root / "input.npy"
        output_path = root / "forecast.npz"
        np.save(input_path, self._frames(), allow_pickle=False)
        old_argv = sys.argv
        sys.argv = [
            "advar-nowcast",
            str(input_path),
            str(output_path),
            *extra_arguments,
        ]
        try:
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                cli.main()
        finally:
            sys.argv = old_argv
        return output_path

    def test_physical_pair_options_require_grid_time_metadata(self) -> None:
        options = (
            "--maximum-pair-velocity-disagreement-mps",
            "--pair-echo-dilation-m",
            "--phase-correlation-sidelobe-radius-m",
            "--minimum-growth-overlap-area-km2",
        )
        for option in options:
            with (
                self.subTest(option=option),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                input_path = root / "input.npy"
                output_path = root / "forecast.npz"
                np.save(input_path, self._frames(), allow_pickle=False)
                old_argv = sys.argv
                sys.argv = [
                    "advar-nowcast",
                    str(input_path),
                    str(output_path),
                    option,
                    "1",
                ]
                try:
                    with (
                        mock.patch("sys.stdout", new_callable=io.StringIO),
                        mock.patch("sys.stderr", new_callable=io.StringIO),
                        self.assertRaises(SystemExit),
                    ):
                        cli.main()
                finally:
                    sys.argv = old_argv
                self.assertFalse(output_path.exists())

    def test_physical_pair_options_remain_supported_with_grid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = self._run_cli(
                Path(directory),
                *self._grid_arguments(),
                "--maximum-pair-velocity-disagreement-mps",
                "10",
                "--pair-echo-dilation-m",
                "500",
                "--phase-correlation-sidelobe-radius-m",
                "500",
                "--minimum-growth-overlap-area-km2",
                "1",
            )
            self.assertTrue(output_path.is_file())
            with np.load(output_path, allow_pickle=False) as result:
                config = json.loads(result["nowcast_config_json"].item())
            self.assertEqual(config["pair_echo_dilation_m"], 500.0)
            self.assertEqual(
                config["phase_correlation_sidelobe_radius_m"],
                500.0,
            )
            self.assertEqual(
                config["maximum_pair_velocity_disagreement_mps"],
                10.0,
            )
            self.assertEqual(config["minimum_growth_overlap_area_km2"], 1.0)

    def test_builder_rejects_non_x86_linux_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "advar_radar_nowcast-0.114.0-py3-none-any.whl"
            lock = root / (
                f"runtime-py{sys.version_info.major}{sys.version_info.minor}-linux.lock"
            )
            audit = root / "audit.json"
            wheelhouse = root / "wheelhouse"
            output = root / "bundle"
            wheel.write_bytes(b"placeholder")
            lock.write_text("", encoding="utf-8")
            audit.write_text('{"dependencies": []}', encoding="utf-8")
            wheelhouse.mkdir()
            signing_key = Ed25519PrivateKey.from_private_bytes(b"1" * 32)
            with (
                mock.patch.object(BUNDLE.platform, "system", return_value="Linux"),
                mock.patch.object(BUNDLE.platform, "machine", return_value="aarch64"),
                mock.patch.object(BUNDLE, "_locked_packages", return_value=[]),
                mock.patch.object(BUNDLE, "_wheel_identity", return_value=(
                    "advar-radar-nowcast",
                    "0.114.0",
                )),
                mock.patch.object(
                    BUNDLE.importlib.metadata,
                    "version",
                    return_value="0.114.0",
                ),
                mock.patch.object(BUNDLE, "_runtime_tree_snapshot", return_value={}),
                mock.patch.object(
                    BUNDLE,
                    "_validate_runtime_tree_snapshot",
                    return_value={},
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "current deployment bundle requires Linux x86_64",
                ),
            ):
                BUNDLE.build_bundle(
                    wheel=wheel,
                    lock=lock,
                    audit=audit,
                    wheelhouse=wheelhouse,
                    output=output,
                    source_commit="a" * 40,
                    repository="gonos2k/AD4DVAR-radar",
                    source_ref="refs/pull/156/merge",
                    workflow_sha="b" * 40,
                    mode="candidate-smoke",
                    signer_id="ci-candidate-smoke",
                    signing_key=signing_key,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
