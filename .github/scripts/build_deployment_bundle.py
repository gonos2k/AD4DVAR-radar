from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import sys
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


BUNDLE_CONTRACT = "advar-linux-cpu-deployment-bundle-v2"
SIGNATURE_DOMAIN = b"advar-linux-cpu-deployment-bundle-v2\x00"
BundleMode = Literal["candidate-smoke", "deployable"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _canonical_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _locked_packages(path: Path) -> list[dict[str, object]]:
    packages: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("    #"):
            continue
        if not line.startswith(" "):
            retained = line.removesuffix(" \\")
            if "==" not in retained:
                raise ValueError("deployment lock contains a non-exact pin")
            name, version = retained.split("==", 1)
            current = {"name": name, "version": version, "sha256": []}
            packages.append(current)
            continue
        prefix = "    --hash=sha256:"
        if current is None or not line.startswith(prefix):
            raise ValueError("deployment lock is noncanonical")
        hashes = current["sha256"]
        if not isinstance(hashes, list):
            raise TypeError("deployment lock hash list is invalid")
        hashes.append(line[len(prefix):].removesuffix(" \\"))
    if not packages or any(not item["sha256"] for item in packages):
        raise ValueError("deployment lock has an unhashed package")
    return packages


def _validate_audit(path: Path) -> dict[str, object]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    dependencies = value.get("dependencies") if isinstance(value, dict) else None
    if not isinstance(dependencies, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("vulns"), list)
        or item["vulns"]
        for item in dependencies
    ):
        raise ValueError("vulnerability audit is incomplete or nonclean")
    return value


def _validate_identity(
    *,
    source_commit: str,
    repository: str,
    source_ref: str,
    workflow_sha: str,
    signer_id: str,
    mode: BundleMode,
) -> None:
    if (
        re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", workflow_sha) is None
        or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None
        or not source_ref.startswith("refs/")
        or not signer_id
        or signer_id.strip() != signer_id
        or mode not in ("candidate-smoke", "deployable")
    ):
        raise ValueError("deployment bundle identity is invalid")


def _signature_preimage(unsigned_manifest: dict[str, object]) -> bytes:
    return SIGNATURE_DOMAIN + _canonical_bytes(unsigned_manifest)


def _require_root_owned_nonwritable_ancestry(path: Path) -> None:
    for retained in (path, *path.parents):
        metadata = retained.lstat()
        expected_directory = retained == path or retained in path.parents
        if (
            (expected_directory and not stat.S_ISDIR(metadata.st_mode))
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError(
                "deployable bundle ancestry must be root-owned and non-writable"
            )
        if retained == retained.parent:
            break


def _validate_deployable_bundle_permissions(bundle: Path) -> None:
    if not bundle.is_absolute() or bundle.is_symlink():
        raise ValueError("deployable bundle path must be absolute and unsymlinked")
    _require_root_owned_nonwritable_ancestry(bundle)
    for path in bundle.iterdir():
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError(
                "deployable bundle payload must be root-owned and non-writable"
            )


def build_bundle(
    *,
    wheel: Path,
    lock: Path,
    audit: Path,
    output: Path,
    source_commit: str,
    repository: str,
    source_ref: str,
    workflow_sha: str,
    mode: BundleMode,
    signer_id: str,
    signing_key: Ed25519PrivateKey,
) -> Path:
    _validate_identity(
        source_commit=source_commit,
        repository=repository,
        source_ref=source_ref,
        workflow_sha=workflow_sha,
        signer_id=signer_id,
        mode=mode,
    )
    if (
        not wheel.is_file()
        or wheel.is_symlink()
        or wheel.suffix != ".whl"
        or not lock.is_file()
        or lock.is_symlink()
        or not audit.is_file()
        or audit.is_symlink()
    ):
        raise ValueError("deployment bundle inputs are invalid")
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    if python_tag not in lock.name or "linux" not in lock.name:
        raise ValueError("deployment lock does not match this Python/Linux target")
    if platform.system() != "Linux":
        raise ValueError("current deployment bundle is Linux-only")
    locked = _locked_packages(lock)
    audit_value = _validate_audit(audit)
    try:
        import torch
    except ImportError as error:
        raise ValueError("deployment runtime is not installed") from error
    if torch.version.cuda is not None:
        raise ValueError("deployment bundle must be CPU-only")
    package_version = importlib.metadata.version("advar-radar-nowcast")
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    wheel_output = output / wheel.name
    lock_output = output / lock.name
    shutil.copyfile(wheel, wheel_output)
    shutil.copyfile(lock, lock_output)
    audit_output = output / "vulnerability-audit.json"
    _canonical_json(audit_output, audit_value)
    sbom_output = output / "sbom.cyclonedx.json"
    _canonical_json(
        sbom_output,
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "version": 1,
            "metadata": {
                "component": {
                    "type": "application",
                    "name": "advar-radar-nowcast",
                    "version": package_version,
                    "hashes": [{"alg": "SHA-256", "content": _sha256(wheel)}],
                }
            },
            "components": [
                {
                    "type": "library",
                    "name": item["name"],
                    "version": item["version"],
                    "hashes": [
                        {"alg": "SHA-256", "content": digest}
                        for digest in item["sha256"]
                    ],
                }
                for item in locked
            ],
        },
    )
    installation_output = output / "installation-attestation.json"
    _canonical_json(
        installation_output,
        {
            "contract": "advar-installation-attestation-v1",
            "source_commit": source_commit,
            "package_version": package_version,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "torch_version": torch.__version__,
            "torch_cuda": None,
            "wheel_filename": wheel.name,
            "wheel_sha256": _sha256(wheel),
            "lock_filename": lock.name,
            "lock_sha256": _sha256(lock),
            "install_command_contract": (
                "pip-require-hashes-runtime-then-wheel-no-deps-v1"
            ),
        },
    )
    payload_files = tuple(sorted(output.iterdir(), key=lambda item: item.name))
    unsigned_manifest: dict[str, object] = {
        "contract": BUNDLE_CONTRACT,
        "mode": mode,
        "repository": repository,
        "source_ref": source_ref,
        "source_commit": source_commit,
        "workflow_sha": workflow_sha,
        "signer_id": signer_id,
        "package_version": package_version,
        "python_tag": python_tag,
        "platform": "linux-cpu",
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in payload_files
        ],
    }
    bundle_digest = hashlib.sha256(
        _canonical_bytes(unsigned_manifest)
    ).hexdigest()
    manifest = unsigned_manifest | {
        "bundle_digest": bundle_digest,
        "signature_hex": signing_key.sign(
            _signature_preimage(unsigned_manifest)
        ).hex(),
    }
    _canonical_json(output / "manifest.json", manifest)
    return output


def verify_bundle(
    bundle: Path,
    *,
    trusted_public_key_hex: str,
    expected_mode: BundleMode,
    expected_repository: str,
    expected_source_ref: str,
    expected_source_commit: str,
    expected_workflow_sha: str,
    expected_signer_id: str,
) -> str:
    if expected_mode == "deployable":
        _validate_deployable_bundle_permissions(bundle)
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or bundle.is_symlink() or not manifest_path.is_file():
        raise ValueError("deployment bundle is missing its manifest")
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("deployment bundle manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("deployment bundle manifest is invalid")
    signature_hex = manifest.pop("signature_hex", None)
    bundle_digest = manifest.pop("bundle_digest", None)
    signer_id = manifest.get("signer_id")
    _validate_identity(
        source_commit=expected_source_commit,
        repository=expected_repository,
        source_ref=expected_source_ref,
        workflow_sha=expected_workflow_sha,
        signer_id=expected_signer_id,
        mode=expected_mode,
    )
    if (
        manifest.get("contract") != BUNDLE_CONTRACT
        or manifest.get("mode") != expected_mode
        or manifest.get("repository") != expected_repository
        or manifest.get("source_ref") != expected_source_ref
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("workflow_sha") != expected_workflow_sha
        or signer_id != expected_signer_id
        or not isinstance(bundle_digest, str)
        or bundle_digest
        != hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
        or not isinstance(signature_hex, str)
    ):
        raise ValueError("deployment bundle identity or digest is invalid")
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(trusted_public_key_hex)
        ).verify(
            bytes.fromhex(signature_hex),
            _signature_preimage(manifest),
        )
    except (InvalidSignature, TypeError, ValueError) as error:
        raise ValueError("deployment bundle signature is untrusted") from error
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("deployment bundle file manifest is invalid")
    expected_names: set[str] = set()
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "size_bytes", "sha256"}
            or not isinstance(item["name"], str)
            or Path(item["name"]).name != item["name"]
            or item["name"] == "manifest.json"
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is None
            or item["name"] in expected_names
        ):
            raise ValueError("deployment bundle file manifest is invalid")
        expected_names.add(item["name"])
        path = bundle / item["name"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError("deployment bundle payload was modified")
    actual_names = {path.name for path in bundle.iterdir()}
    if actual_names != expected_names | {"manifest.json"}:
        raise ValueError("deployment bundle contains unmanifested files")
    return bundle_digest


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    if not path.is_absolute():
        raise ValueError("bundle signing key path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if not stat.S_ISREG(metadata.st_mode) or mode & 0o077:
            raise ValueError("bundle signing key must be a private regular file")
        value = os.read(descriptor, 33)
    finally:
        os.close(descriptor)
    if len(value) != 32:
        raise ValueError("bundle signing key must contain 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(value)


def _load_trusted_public_key(path: Path, *, deployable: bool) -> str:
    if not path.is_absolute():
        raise ValueError("trusted bundle public key path must be absolute")
    if deployable:
        _require_root_owned_nonwritable_ancestry(path.parent)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > 128
            or (
                deployable
                and (metadata.st_uid != 0 or metadata.st_mode & 0o022)
            )
        ):
            raise ValueError("trusted bundle public key is invalid")
        value = os.read(descriptor, 129).decode("ascii").strip()
    finally:
        os.close(descriptor)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("trusted bundle public key is invalid")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--wheel", type=Path, required=True)
    build.add_argument("--lock", type=Path, required=True)
    build.add_argument("--audit", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--repository", required=True)
    build.add_argument("--source-ref", required=True)
    build.add_argument("--workflow-sha", required=True)
    build.add_argument(
        "--mode", choices=("candidate-smoke", "deployable"), required=True
    )
    build.add_argument("--signer-id", required=True)
    build.add_argument("--signing-private-key", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--trusted-public-key", type=Path, required=True)
    verify.add_argument(
        "--expected-mode",
        choices=("candidate-smoke", "deployable"),
        required=True,
    )
    verify.add_argument("--expected-repository", required=True)
    verify.add_argument("--expected-source-ref", required=True)
    verify.add_argument("--expected-source-commit", required=True)
    verify.add_argument("--expected-workflow-sha", required=True)
    verify.add_argument("--expected-signer-id", required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "build":
            build_bundle(
                wheel=arguments.wheel.resolve(),
                lock=arguments.lock.resolve(),
                audit=arguments.audit.resolve(),
                output=arguments.output.resolve(),
                source_commit=arguments.source_commit,
                repository=arguments.repository,
                source_ref=arguments.source_ref,
                workflow_sha=arguments.workflow_sha,
                mode=arguments.mode,
                signer_id=arguments.signer_id,
                signing_key=_load_private_key(
                    arguments.signing_private_key.absolute()
                ),
            )
        else:
            public_key_hex = _load_trusted_public_key(
                arguments.trusted_public_key.absolute(),
                deployable=arguments.expected_mode == "deployable",
            )
            verify_bundle(
                arguments.bundle.absolute(),
                trusted_public_key_hex=public_key_hex,
                expected_mode=arguments.expected_mode,
                expected_repository=arguments.expected_repository,
                expected_source_ref=arguments.expected_source_ref,
                expected_source_commit=arguments.expected_source_commit,
                expected_workflow_sha=arguments.expected_workflow_sha,
                expected_signer_id=arguments.expected_signer_id,
            )
    except (OSError, TypeError, ValueError) as error:
        print(f"deployment bundle generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
