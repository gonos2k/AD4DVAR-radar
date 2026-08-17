from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


SCRIPT = Path(__file__).parents[1] / ".github/scripts/build_deployment_bundle.py"
SPEC = importlib.util.spec_from_file_location("advar_deployment_bundle", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("deployment bundle script cannot be loaded")
bundle_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle_module)


class DeploymentBundleTests(unittest.TestCase):
    def test_bundle_seals_wheel_lock_sbom_audit_and_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "advar_radar_nowcast-0.90.0-py3-none-any.whl"
            wheel.write_bytes(b"wheel-bytes")
            lock = root / (
                "runtime-py"
                f"{sys.version_info.major}{sys.version_info.minor}-linux.lock"
            )
            lock.write_text(
                "numpy==2.2.0 \\\n"
                f"    --hash=sha256:{'1' * 64}\n",
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
            with (
                mock.patch.dict("sys.modules", {"torch": fake_torch}),
                mock.patch.object(bundle_module.platform, "system", return_value="Linux"),
                mock.patch.object(bundle_module.platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    bundle_module.importlib.metadata,
                    "version",
                    return_value="0.90.0",
                ),
            ):
                bundle_module.build_bundle(
                    wheel=wheel,
                    lock=lock,
                    audit=audit,
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
                "advar-linux-cpu-deployment-bundle-v2",
            )
            self.assertEqual(manifest["mode"], "candidate-smoke")
            self.assertEqual(manifest["source_commit"], "a" * 40)
            self.assertEqual(
                {item["name"] for item in manifest["files"]},
                {
                    wheel.name,
                    lock.name,
                    "installation-attestation.json",
                    "sbom.cyclonedx.json",
                    "vulnerability-audit.json",
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
