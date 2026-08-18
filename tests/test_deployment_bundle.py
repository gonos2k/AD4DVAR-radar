from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


SCRIPT = Path(__file__).parents[1] / ".github/scripts/build_deployment_bundle.py"
SPEC = importlib.util.spec_from_file_location("advar_deployment_bundle", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("deployment bundle script cannot be loaded")
bundle_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle_module)


def _write_test_wheel(
    path: Path,
    *,
    distribution: str,
    version: str,
) -> None:
    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    package = distribution.replace("-", "_")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.3\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n",
        )
        archive.writestr(f"{package}/__init__.py", "VALUE = 1\n")


class DeploymentBundleTests(unittest.TestCase):
    def test_runtime_tree_excludes_distribution_files_outside_import_roots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "venv/lib/python3.12/site-packages"
            torch_package = site / "torch"
            advar_package = site / "advar"
            torch_metadata = site / "torch-2.13.0+cpu.dist-info"
            advar_metadata = site / "advar_radar_nowcast-0.91.0.dist-info"
            torch_package.mkdir(parents=True)
            advar_package.mkdir()
            torch_metadata.mkdir()
            advar_metadata.mkdir()
            (torch_package / "__init__.py").write_text("", encoding="utf-8")
            (advar_package / "__init__.py").write_text("", encoding="utf-8")
            (torch_metadata / "RECORD").write_text(
                "../../../bin/torchrun,venv-specific-hash,1\n",
                encoding="utf-8",
            )
            (advar_metadata / "direct_url.json").write_text(
                '{"url":"file:///venv-specific-wheel-path"}',
                encoding="utf-8",
            )

            class FakeDistribution:
                def __init__(self, version: str, files: list[Path]) -> None:
                    self.version = version
                    self.files = files

                def locate_file(self, path: Path) -> Path:
                    return site / path

            distributions = {
                "torch": FakeDistribution(
                    "2.13.0+cpu",
                    [
                        Path("torch/__init__.py"),
                        Path("torch-2.13.0+cpu.dist-info/RECORD"),
                        Path("../../../bin/torchrun"),
                    ],
                ),
                "advar-radar-nowcast": FakeDistribution(
                    "0.91.0",
                    [
                        Path("advar/__init__.py"),
                        Path(
                            "advar_radar_nowcast-0.91.0.dist-info/"
                            "direct_url.json"
                        ),
                    ],
                ),
            }
            with (
                mock.patch.object(
                    bundle_module.sysconfig,
                    "get_path",
                    return_value=str(site),
                ),
                mock.patch.object(
                    bundle_module.importlib.metadata,
                    "distribution",
                    side_effect=lambda name: distributions[name],
                ),
            ):
                snapshot = bundle_module._runtime_tree_snapshot(
                    selected_wheels=[
                        {
                            "name": "torch",
                            "wheel_version": "2.13.0+cpu",
                        }
                    ],
                    application_version="0.91.0",
                )

            self.assertEqual(
                [item["path"] for item in snapshot["files"]],
                ["site/advar/__init__.py", "site/torch/__init__.py"],
            )

    def test_public_lock_accepts_exact_hashed_local_cpu_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            torch_wheel = wheelhouse / "torch-2.13.0+cpu-py3-none-any.whl"
            _write_test_wheel(
                torch_wheel,
                distribution="torch",
                version="2.13.0+cpu",
            )
            locked = [
                {
                    "name": "torch",
                    "version": "2.13.0",
                    "sha256": [bundle_module._sha256(torch_wheel)],
                }
            ]

            self.assertEqual(
                bundle_module._validate_wheelhouse(wheelhouse, locked),
                [
                    {
                        "name": "torch",
                        "locked_version": "2.13.0",
                        "wheel_version": "2.13.0+cpu",
                        "filename": torch_wheel.name,
                        "size_bytes": torch_wheel.stat().st_size,
                        "sha256": bundle_module._sha256(torch_wheel),
                    }
                ],
            )

    def test_bundle_seals_wheel_lock_sbom_audit_and_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "advar_radar_nowcast-0.91.0-py3-none-any.whl"
            _write_test_wheel(
                wheel,
                distribution="advar-radar-nowcast",
                version="0.91.0",
            )
            lock = root / (
                "runtime-py"
                f"{sys.version_info.major}{sys.version_info.minor}-linux.lock"
            )
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            numpy_wheel = wheelhouse / "numpy-2.2.0-py3-none-any.whl"
            _write_test_wheel(
                numpy_wheel,
                distribution="numpy",
                version="2.2.0",
            )
            lock.write_text(
                "numpy==2.2.0 \\\n"
                f"    --hash=sha256:{bundle_module._sha256(numpy_wheel)}\n",
                encoding="utf-8",
            )
            audit = root / "audit.json"
            audit.write_text(
                json.dumps({
                    "dependencies": [
                        {"name": "numpy", "version": "2.2.0", "vulns": []}
                    ],
                    "fixes": [],
                }),
                encoding="utf-8",
            )
            output = root / "bundle"
            signing_key = Ed25519PrivateKey.from_private_bytes(b"\x31" * 32)
            fake_torch = types.SimpleNamespace(
                __version__="2.13.0",
                version=types.SimpleNamespace(cuda=None),
            )
            runtime_tree_unsigned = {
                "contract": "advar-import-runtime-tree-v1",
                "distributions": [
                    {"name": "advar-radar-nowcast", "version": "0.91.0"},
                    {"name": "numpy", "version": "2.2.0"},
                ],
                "files": [
                    {
                        "distribution": "numpy",
                        "path": "site/numpy/__init__.py",
                        "size_bytes": 10,
                        "sha256": "8" * 64,
                    }
                ],
            }
            runtime_tree = runtime_tree_unsigned | {
                "runtime_tree_digest": bundle_module._json_digest(
                    runtime_tree_unsigned
                )
            }
            with (
                mock.patch.dict("sys.modules", {"torch": fake_torch}),
                mock.patch.object(bundle_module.platform, "system", return_value="Linux"),
                mock.patch.object(bundle_module.platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    bundle_module.importlib.metadata,
                    "version",
                    return_value="0.91.0",
                ),
                mock.patch.object(
                    bundle_module,
                    "_runtime_tree_snapshot",
                    return_value=runtime_tree,
                ),
            ):
                bundle_module.build_bundle(
                    wheel=wheel,
                    lock=lock,
                    audit=audit,
                    wheelhouse=wheelhouse,
                    output=output,
                    source_commit="a" * 40,
                    repository="gonos2k/AD4DVAR-radar",
                    source_ref="refs/pull/126/merge",
                    workflow_sha="b" * 40,
                    mode="candidate-smoke",
                    signer_id="ci-candidate-smoke",
                    signing_key=signing_key,
                )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["contract"],
                "advar-linux-cpu-deployment-bundle-v3",
            )
            self.assertEqual(manifest["mode"], "candidate-smoke")
            self.assertEqual(manifest["source_commit"], "a" * 40)
            self.assertEqual(
                {item["name"] for item in manifest["files"]},
                {
                    wheel.name,
                    lock.name,
                    "installation-attestation.json",
                    "runtime-tree.json",
                    "sbom.cyclonedx.json",
                    "vulnerability-audit.json",
                    f"wheelhouse/{numpy_wheel.name}",
                },
            )
            self.assertEqual(len(manifest["bundle_digest"]), 64)
            verified_digest = bundle_module.verify_bundle(
                output,
                trusted_public_key_hex=(
                    signing_key.public_key().public_bytes_raw().hex()
                ),
                expected_mode="candidate-smoke",
                expected_repository="gonos2k/AD4DVAR-radar",
                expected_source_ref="refs/pull/126/merge",
                expected_source_commit="a" * 40,
                expected_workflow_sha="b" * 40,
                expected_signer_id="ci-candidate-smoke",
            )
            self.assertEqual(verified_digest, manifest["bundle_digest"])
            with (
                mock.patch.object(bundle_module.platform, "system", return_value="Linux"),
                mock.patch.object(bundle_module.platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    bundle_module,
                    "_runtime_tree_snapshot",
                    return_value=runtime_tree,
                ),
            ):
                self.assertEqual(
                    bundle_module.verify_current_installation(output),
                    runtime_tree["runtime_tree_digest"],
                )

            activation_key = Ed25519PrivateKey.from_private_bytes(b"\x41" * 32)
            with (
                mock.patch.object(bundle_module.platform, "system", return_value="Linux"),
                mock.patch.object(bundle_module.platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    bundle_module,
                    "_runtime_tree_snapshot",
                    return_value=runtime_tree,
                ),
            ):
                activation = bundle_module.issue_runtime_activation_receipt(
                    output,
                    trusted_bundle_public_key_hex=(
                        signing_key.public_key().public_bytes_raw().hex()
                    ),
                    expected_mode="candidate-smoke",
                    expected_repository="gonos2k/AD4DVAR-radar",
                    expected_source_ref="refs/pull/126/merge",
                    expected_source_commit="a" * 40,
                    expected_workflow_sha="b" * 40,
                    expected_bundle_signer_id="ci-candidate-smoke",
                    deployment_instance_id="ci-offline-replay",
                    activated_at="2026-08-18T00:00:00Z",
                    activation_signer_id="ci-runtime-activation",
                    activation_signing_key=activation_key,
                )
            activation_digest = (
                bundle_module.verify_runtime_activation_receipt(
                    activation,
                    trusted_activation_public_key_hex=(
                        activation_key.public_key().public_bytes_raw().hex()
                    ),
                    expected_bundle_digest=manifest["bundle_digest"],
                    expected_runtime_tree_digest=(
                        runtime_tree["runtime_tree_digest"]
                    ),
                    expected_bundle_mode="candidate-smoke",
                    expected_deployment_instance_id="ci-offline-replay",
                    expected_activation_signer_id="ci-runtime-activation",
                )
            )
            self.assertEqual(activation_digest, activation["receipt_digest"])
            relabeled_activation = dict(activation)
            relabeled_activation["deployment_instance_id"] = "production"
            with self.assertRaisesRegex(ValueError, "identity"):
                bundle_module.verify_runtime_activation_receipt(
                    relabeled_activation,
                    trusted_activation_public_key_hex=(
                        activation_key.public_key().public_bytes_raw().hex()
                    ),
                    expected_bundle_digest=manifest["bundle_digest"],
                    expected_runtime_tree_digest=(
                        runtime_tree["runtime_tree_digest"]
                    ),
                    expected_bundle_mode="candidate-smoke",
                    expected_deployment_instance_id="production",
                    expected_activation_signer_id="ci-runtime-activation",
                )

            changed_tree = dict(runtime_tree)
            changed_tree["runtime_tree_digest"] = "f" * 64
            with (
                mock.patch.object(bundle_module.platform, "system", return_value="Linux"),
                mock.patch.object(bundle_module.platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    bundle_module,
                    "_runtime_tree_snapshot",
                    return_value=changed_tree,
                ),
                self.assertRaisesRegex(ValueError, "runtime tree"),
            ):
                bundle_module.verify_current_installation(output)

            bundled_numpy = output / "wheelhouse" / numpy_wheel.name
            original_numpy = bundled_numpy.read_bytes()
            bundled_numpy.write_bytes(original_numpy + b"attacker")
            with self.assertRaisesRegex(ValueError, "modified"):
                bundle_module.verify_bundle(
                    output,
                    trusted_public_key_hex=(
                        signing_key.public_key().public_bytes_raw().hex()
                    ),
                    expected_mode="candidate-smoke",
                    expected_repository="gonos2k/AD4DVAR-radar",
                    expected_source_ref="refs/pull/126/merge",
                    expected_source_commit="a" * 40,
                    expected_workflow_sha="b" * 40,
                    expected_signer_id="ci-candidate-smoke",
                )
            bundled_numpy.write_bytes(original_numpy)

            missing_wheelhouse = root / "missing-wheelhouse"
            missing_wheelhouse.mkdir()
            with (
                mock.patch.dict("sys.modules", {"torch": fake_torch}),
                mock.patch.object(bundle_module.platform, "system", return_value="Linux"),
                mock.patch.object(
                    bundle_module.importlib.metadata,
                    "version",
                    return_value="0.91.0",
                ),
                self.assertRaisesRegex(ValueError, "wheelhouse"),
            ):
                bundle_module.build_bundle(
                    wheel=wheel,
                    lock=lock,
                    audit=audit,
                    wheelhouse=missing_wheelhouse,
                    output=root / "missing-bundle",
                    source_commit="a" * 40,
                    repository="gonos2k/AD4DVAR-radar",
                    source_ref="refs/pull/126/merge",
                    workflow_sha="b" * 40,
                    mode="candidate-smoke",
                    signer_id="ci-candidate-smoke",
                    signing_key=signing_key,
                )

            extra_wheel = wheelhouse / "unapproved-1.0-py3-none-any.whl"
            _write_test_wheel(
                extra_wheel,
                distribution="unapproved",
                version="1.0",
            )
            with self.assertRaisesRegex(ValueError, "wheelhouse"):
                bundle_module._validate_wheelhouse(
                    wheelhouse,
                    bundle_module._locked_packages(lock),
                )
            extra_wheel.unlink()

            original_source_numpy = numpy_wheel.read_bytes()
            _write_test_wheel(
                numpy_wheel,
                distribution="numpy",
                version="9.9.0",
            )
            with self.assertRaisesRegex(ValueError, "wheelhouse"):
                bundle_module._validate_wheelhouse(
                    wheelhouse,
                    bundle_module._locked_packages(lock),
                )
            numpy_wheel.write_bytes(original_source_numpy)

            with mock.patch.object(
                bundle_module,
                "_validate_deployable_bundle_permissions",
            ), self.assertRaisesRegex(ValueError, "identity"):
                bundle_module.verify_bundle(
                    output,
                    trusted_public_key_hex=(
                        signing_key.public_key().public_bytes_raw().hex()
                    ),
                    expected_mode="deployable",
                    expected_repository="gonos2k/AD4DVAR-radar",
                    expected_source_ref="refs/pull/126/merge",
                    expected_source_commit="a" * 40,
                    expected_workflow_sha="b" * 40,
                    expected_signer_id="ci-candidate-smoke",
                )

            original_manifest_bytes = (output / "manifest.json").read_bytes()
            relabeled = json.loads(original_manifest_bytes)
            relabeled["signer_id"] = "advar-release"
            unsigned_relabeled = {
                key: value
                for key, value in relabeled.items()
                if key not in {"bundle_digest", "signature_hex"}
            }
            relabeled["bundle_digest"] = bundle_module.hashlib.sha256(
                bundle_module._canonical_bytes(unsigned_relabeled)
            ).hexdigest()
            (output / "manifest.json").write_bytes(
                bundle_module._canonical_bytes(relabeled) + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "signature"):
                bundle_module.verify_bundle(
                    output,
                    trusted_public_key_hex=(
                        signing_key.public_key().public_bytes_raw().hex()
                    ),
                    expected_mode="candidate-smoke",
                    expected_repository="gonos2k/AD4DVAR-radar",
                    expected_source_ref="refs/pull/126/merge",
                    expected_source_commit="a" * 40,
                    expected_workflow_sha="b" * 40,
                    expected_signer_id="advar-release",
                )
            (output / "manifest.json").write_bytes(original_manifest_bytes)

            wheel_output = output / wheel.name
            wheel_output.write_bytes(b"attacker-wheel")
            forged = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            unsigned = {
                key: value
                for key, value in forged.items()
                if key not in {"bundle_digest", "signature_hex"}
            }
            for item in unsigned["files"]:
                if item["name"] == wheel.name:
                    item["size_bytes"] = wheel_output.stat().st_size
                    item["sha256"] = bundle_module._sha256(wheel_output)
            forged["files"] = unsigned["files"]
            forged["bundle_digest"] = bundle_module.hashlib.sha256(
                bundle_module._canonical_bytes(unsigned)
            ).hexdigest()
            (output / "manifest.json").write_bytes(
                bundle_module._canonical_bytes(forged) + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "signature"):
                bundle_module.verify_bundle(
                    output,
                    trusted_public_key_hex=(
                        signing_key.public_key().public_bytes_raw().hex()
                    ),
                    expected_mode="candidate-smoke",
                    expected_repository="gonos2k/AD4DVAR-radar",
                    expected_source_ref="refs/pull/126/merge",
                    expected_source_commit="a" * 40,
                    expected_workflow_sha="b" * 40,
                    expected_signer_id="ci-candidate-smoke",
                )

            vulnerable = root / "vulnerable.json"
            vulnerable.write_text(
                json.dumps({
                    "dependencies": [
                        {
                            "name": "numpy",
                            "version": "2.2.0",
                            "vulns": [{"id": "CVE-test"}],
                        }
                    ]
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "audit"):
                bundle_module._validate_audit(vulnerable)

            public_key = root / "trusted-public-key.hex"
            public_key.write_text(
                signing_key.public_key().public_bytes_raw().hex(),
                encoding="ascii",
            )
            linked_key = root / "linked-public-key.hex"
            linked_key.symlink_to(public_key)
            with self.assertRaises((OSError, ValueError)):
                bundle_module._load_trusted_public_key(
                    linked_key.absolute(),
                    deployable=False,
                )


if __name__ == "__main__":
    unittest.main()
