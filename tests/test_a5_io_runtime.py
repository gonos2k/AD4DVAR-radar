from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import advar.acceptance as acceptance
import advar.runtime_closure as runtime_closure
from advar.acceptance import RealCaseAcceptanceManifest
from advar.diagnostics import audit_transport
import test_acceptance as acceptance_fixtures


REPOSITORY = Path(__file__).resolve().parents[1]
ACCEPTANCE_SCRIPT = REPOSITORY / ".github/scripts/run_real_case_acceptance.py"


class A5IoRuntimeTests(unittest.TestCase):
    def _manifest_fixture(
        self,
        root: Path,
    ) -> tuple[Path, RealCaseAcceptanceManifest]:
        manifest = acceptance_fixtures.RealCaseAcceptanceTests().fixture(root)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(manifest.json, encoding="utf-8")
        return manifest_path, manifest

    def _run_acceptance(
        self,
        manifest_path: Path,
        artifact_root: Path,
        report: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(ACCEPTANCE_SCRIPT),
                "--manifest",
                str(manifest_path),
                "--artifact-root",
                str(artifact_root),
                "--report",
                str(report),
            ),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_report_rejects_manifest_and_artifact_aliases(self) -> None:
        for target_kind in ("manifest", "artifact"):
            for alias_kind in ("direct", "symlink", "hardlink"):
                with (
                    self.subTest(
                        target_kind=target_kind,
                        alias_kind=alias_kind,
                    ),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    workspace = Path(directory).resolve()
                    root = workspace / "artifacts"
                    root.mkdir()
                    manifest_path, manifest = self._manifest_fixture(root)
                    if target_kind == "manifest":
                        victim = manifest_path
                    else:
                        victim = root / manifest.cases[0].artifacts[0].relative_path
                    if alias_kind == "direct":
                        report = victim
                    elif alias_kind == "symlink":
                        report = workspace / f"{target_kind}-symlink.json"
                        report.symlink_to(victim)
                    else:
                        report = workspace / f"{target_kind}-hardlink.json"
                        os.link(victim, report)
                    before = victim.read_bytes()

                    completed = self._run_acceptance(manifest_path, root, report)

                    self.assertNotEqual(completed.returncode, 0)
                    self.assertEqual(victim.read_bytes(), before)

    def test_report_rejects_a_path_inside_the_artifact_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest_path, manifest = self._manifest_fixture(root)
            victim = root / manifest.cases[0].artifacts[0].relative_path
            before = victim.read_bytes()

            completed = self._run_acceptance(
                manifest_path,
                root,
                root / "new-report.json",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(victim.read_bytes(), before)
            self.assertFalse((root / "new-report.json").exists())

    def test_report_outside_inputs_is_complete_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            root = workspace / "artifacts"
            root.mkdir()
            manifest_path, manifest = self._manifest_fixture(root)
            report = workspace / "report.json"

            completed = self._run_acceptance(manifest_path, root, report)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            decoded = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(
                decoded["contract"],
                "advar-real-case-acceptance-report-v2",
            )
            self.assertEqual(decoded["manifest_digest"], manifest.manifest_digest)

    def test_acceptance_fd_walk_does_not_follow_a_parent_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            root = workspace / "artifacts"
            outside = workspace / "outside"
            root.mkdir()
            outside.mkdir()
            _, manifest = self._manifest_fixture(root)
            first = manifest.cases[0].artifacts[0]
            case_directory = root / "case-1"
            outside_case = outside / "case-1"
            outside_case.mkdir()
            outside_file = outside_case / Path(first.relative_path).name
            outside_file.write_bytes(b"outside")
            outside_file.chmod(0o600)

            original_open = acceptance.os.open
            original_read = acceptance._read_snapshot_descriptor
            swapped = False

            def open_hook(
                path: object,
                flags: int,
                *args: object,
                **kwargs: object,
            ) -> int:
                nonlocal swapped
                descriptor = original_open(path, flags, *args, **kwargs)
                if (
                    not swapped
                    and kwargs.get("dir_fd") is not None
                    and path == "case-1"
                ):
                    swapped = True
                    case_directory.rename(root / "case-1-raced")
                    case_directory.symlink_to(
                        outside_case,
                        target_is_directory=True,
                    )
                return descriptor

            def read_hook(descriptor: int, *, maximum_bytes: int) -> bytes:
                data = original_read(descriptor, maximum_bytes=maximum_bytes)
                if swapped and case_directory.is_symlink():
                    case_directory.unlink()
                    (root / "case-1-raced").rename(case_directory)
                return data

            with (
                patch.object(acceptance.os, "open", side_effect=open_hook),
                patch.object(
                    acceptance,
                    "_read_snapshot_descriptor",
                    side_effect=read_hook,
                ),
            ):
                acceptance.verify_real_case_acceptance(manifest, artifact_root=root)

            self.assertTrue(swapped)
            self.assertTrue(case_directory.is_dir())
            self.assertFalse(case_directory.is_symlink())

    def test_runtime_rejects_fifo_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "runtime.fifo"
            os.mkfifo(fifo)
            with patch.object(runtime_closure.os, "open") as opened:
                with self.assertRaisesRegex(ValueError, "must be regular"):
                    runtime_closure._runtime_file_snapshot(fifo, deployable=False)
            opened.assert_not_called()

    def test_runtime_rejects_a_fifo_replacing_a_checked_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.bin"
            path.write_bytes(b"regular")
            original_open = runtime_closure.os.open

            def replace_with_fifo(
                opened_path: object,
                flags: int,
                *args: object,
                **kwargs: object,
            ) -> int:
                path.unlink()
                os.mkfifo(path)
                return original_open(opened_path, flags, *args, **kwargs)

            with patch.object(
                runtime_closure.os,
                "open",
                side_effect=replace_with_fifo,
            ):
                with self.assertRaisesRegex(ValueError, "immutable and regular"):
                    runtime_closure._runtime_file_snapshot(path, deployable=False)

    def test_runtime_rejects_unresolved_ldd_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "site-packages"
            source = root / "package/extension.so"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"extension")
            completed = subprocess.CompletedProcess(
                args=("ldd", str(source)),
                returncode=0,
                stdout="libcritical.so => not found\n",
                stderr="",
            )
            with (
                patch.object(
                    runtime_closure.platform,
                    "system",
                    return_value="Linux",
                ),
                patch.object(
                    runtime_closure.subprocess,
                    "run",
                    return_value=completed,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unresolved dependency"):
                    runtime_closure._linked_native_libraries(
                        (source,), deployable=False, import_roots=(root,)
                    )

    def test_transport_rejects_moved_grid_shape_mismatch(self) -> None:
        echo = torch.ones((2, 2), dtype=torch.float64)
        moved = torch.ones((3, 3), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "same grid shape"):
            audit_transport(
                echo,
                torch.zeros(2, dtype=torch.float64),
                moved=moved,
            )

    def test_mps_workflow_pins_the_certified_torch_version(self) -> None:
        workflow = (REPOSITORY / ".github/workflows/mps-certification.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch==2.13.0", workflow)
        self.assertIn(
            "torch.__version__.split('+', 1)[0] == '2.13.0'",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
