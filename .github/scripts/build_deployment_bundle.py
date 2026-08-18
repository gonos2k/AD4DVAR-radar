from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
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
import sysconfig
from typing import Any, Literal
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


BUNDLE_CONTRACT = "advar-linux-cpu-deployment-bundle-v3"
SIGNATURE_DOMAIN = b"advar-linux-cpu-deployment-bundle-v3\x00"
ACTIVATION_SIGNATURE_DOMAIN = b"advar-runtime-closure-activation-v1\x00"
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


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


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


def _wheel_identity(path: Path) -> tuple[str, str]:
    if not path.is_file() or path.is_symlink() or path.suffix != ".whl":
        raise ValueError("deployment wheelhouse contains an invalid wheel")
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_members = tuple(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
                and name.count("/") == 1
            )
            if len(metadata_members) != 1:
                raise ValueError("deployment wheel metadata is ambiguous")
            metadata = BytesParser().parsebytes(
                archive.read(metadata_members[0], pwd=None)
            )
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise ValueError("deployment wheel is unreadable") from error
    name = metadata.get("Name")
    version = metadata.get("Version")
    if (
        not isinstance(name, str)
        or not isinstance(version, str)
        or not name
        or not version
        or name.strip() != name
        or version.strip() != version
    ):
        raise ValueError("deployment wheel metadata is incomplete")
    return _normalized_distribution_name(name), version


def _validate_wheelhouse(
    wheelhouse: Path,
    locked: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise ValueError("deployment wheelhouse must be an unsymlinked directory")
    expected: dict[str, dict[str, object]] = {}
    for package in locked:
        name = _normalized_distribution_name(str(package["name"]))
        if name in expected:
            raise ValueError("deployment lock contains duplicate packages")
        expected[name] = package
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for wheel in sorted(wheelhouse.iterdir(), key=lambda item: item.name):
        if not wheel.is_file() or wheel.is_symlink() or wheel.suffix != ".whl":
            raise ValueError("deployment wheelhouse must contain only wheel files")
        name, version = _wheel_identity(wheel)
        package = expected.get(name)
        digest = _sha256(wheel)
        allowed_hashes = package.get("sha256") if package is not None else None
        if (
            package is None
            or name in seen
            or version != package["version"]
            or not isinstance(allowed_hashes, list)
            or digest not in allowed_hashes
        ):
            raise ValueError("deployment wheelhouse disagrees with its lock")
        seen.add(name)
        selected.append(
            {
                "name": name,
                "version": version,
                "filename": wheel.name,
                "size_bytes": wheel.stat().st_size,
                "sha256": digest,
            }
        )
    if seen != set(expected):
        raise ValueError("deployment wheelhouse is incomplete")
    return selected


def _runtime_tree_snapshot(
    *,
    locked: list[dict[str, object]],
    application_version: str,
) -> dict[str, object]:
    distributions = [
        {
            "name": _normalized_distribution_name(str(item["name"])),
            "version": str(item["version"]),
        }
        for item in locked
    ] + [
        {"name": "advar-radar-nowcast", "version": application_version}
    ]
    distributions.sort(key=lambda item: str(item["name"]))
    roots = tuple(
        sorted(
            {
                Path(value).absolute()
                for value in (
                    sysconfig.get_path("purelib"),
                    sysconfig.get_path("platlib"),
                )
                if value
            },
            key=str,
        )
    )
    if not roots:
        raise ValueError("deployment runtime has no import roots")
    files: list[dict[str, object]] = []
    claimed_paths: set[tuple[str, str]] = set()
    for expected in distributions:
        name = str(expected["name"])
        version = str(expected["version"])
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError("deployment runtime is incomplete") from error
        if distribution.version != version or distribution.files is None:
            raise ValueError("deployment runtime disagrees with its lock")
        for package_path in distribution.files:
            located = Path(distribution.locate_file(package_path)).absolute()
            relative: Path | None = None
            for root in roots:
                try:
                    relative = located.relative_to(root)
                except ValueError:
                    continue
                break
            if relative is None or "__pycache__" in relative.parts or (
                located.suffix == ".pyc"
            ):
                continue
            try:
                metadata = located.lstat()
            except FileNotFoundError as error:
                raise ValueError("deployment runtime file is missing") from error
            if not stat.S_ISREG(metadata.st_mode) or located.is_symlink():
                raise ValueError("deployment runtime contains a non-regular file")
            canonical_path = relative.as_posix()
            key = (name, canonical_path)
            if key in claimed_paths:
                raise ValueError("deployment runtime file identity is duplicated")
            claimed_paths.add(key)
            files.append(
                {
                    "distribution": name,
                    "path": f"site/{canonical_path}",
                    "size_bytes": metadata.st_size,
                    "sha256": _sha256(located),
                }
            )
    files.sort(key=lambda item: (str(item["distribution"]), str(item["path"])))
    unsigned: dict[str, object] = {
        "contract": "advar-import-runtime-tree-v1",
        "distributions": distributions,
        "files": files,
    }
    return unsigned | {"runtime_tree_digest": _json_digest(unsigned)}


def _validate_runtime_tree_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("runtime tree attestation is invalid")
    unsigned = dict(value)
    digest = unsigned.pop("runtime_tree_digest", None)
    distributions = unsigned.get("distributions")
    files = unsigned.get("files")
    if (
        unsigned.get("contract") != "advar-import-runtime-tree-v1"
        or set(unsigned) != {"contract", "distributions", "files"}
        or not isinstance(distributions, list)
        or not distributions
        or not isinstance(files, list)
        or not files
        or not isinstance(digest, str)
        or digest != _json_digest(unsigned)
    ):
        raise ValueError("runtime tree attestation is invalid")
    previous_distribution = ""
    for item in distributions:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "version"}
            or not isinstance(item["name"], str)
            or not isinstance(item["version"], str)
            or item["name"] != _normalized_distribution_name(item["name"])
            or item["name"] <= previous_distribution
        ):
            raise ValueError("runtime tree distributions are invalid")
        previous_distribution = item["name"]
    previous_file: tuple[str, str] = ("", "")
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != {"distribution", "path", "size_bytes", "sha256"}
            or not isinstance(item["distribution"], str)
            or not isinstance(item["path"], str)
            or not item["path"].startswith("site/")
            or ".." in Path(item["path"]).parts
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is None
        ):
            raise ValueError("runtime tree files are invalid")
        current = (item["distribution"], item["path"])
        if current <= previous_file:
            raise ValueError("runtime tree files are noncanonical")
        previous_file = current
    distribution_names = {str(item["name"]) for item in distributions}
    if any(str(item["distribution"]) not in distribution_names for item in files):
        raise ValueError("runtime tree file has no locked distribution")
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


def _canonical_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("runtime activation time is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("runtime activation time must include UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    if value != canonical:
        raise ValueError("runtime activation time must be canonical UTC")
    return canonical


def _activation_signature_preimage(unsigned_receipt: dict[str, object]) -> bytes:
    return ACTIVATION_SIGNATURE_DOMAIN + _canonical_bytes(unsigned_receipt)


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
    for path in bundle.rglob("*"):
        metadata = path.lstat()
        if (
            not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode))
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
    wheelhouse: Path,
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
        or not wheelhouse.is_dir()
        or wheelhouse.is_symlink()
    ):
        raise ValueError("deployment bundle inputs are invalid")
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    if python_tag not in lock.name or "linux" not in lock.name:
        raise ValueError("deployment lock does not match this Python/Linux target")
    if platform.system() != "Linux":
        raise ValueError("current deployment bundle is Linux-only")
    locked = _locked_packages(lock)
    selected_wheels = _validate_wheelhouse(wheelhouse, locked)
    audit_value = _validate_audit(audit)
    try:
        import torch
    except ImportError as error:
        raise ValueError("deployment runtime is not installed") from error
    if torch.version.cuda is not None:
        raise ValueError("deployment bundle must be CPU-only")
    package_version = importlib.metadata.version("advar-radar-nowcast")
    application_name, application_wheel_version = _wheel_identity(wheel)
    if (
        application_name != "advar-radar-nowcast"
        or application_wheel_version != package_version
    ):
        raise ValueError("application wheel disagrees with the installed runtime")
    runtime_tree = _validate_runtime_tree_snapshot(
        _runtime_tree_snapshot(
            locked=locked,
            application_version=package_version,
        )
    )
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    wheel_output = output / wheel.name
    lock_output = output / lock.name
    shutil.copyfile(wheel, wheel_output)
    shutil.copyfile(lock, lock_output)
    wheelhouse_output = output / "wheelhouse"
    wheelhouse_output.mkdir(mode=0o755)
    for selected in selected_wheels:
        filename = str(selected["filename"])
        shutil.copyfile(wheelhouse / filename, wheelhouse_output / filename)
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
            "contract": "advar-installation-attestation-v2",
            "source_commit": source_commit,
            "package_version": package_version,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "python_abi": sys.implementation.cache_tag,
            "torch_version": torch.__version__,
            "torch_cuda": None,
            "wheel_filename": wheel.name,
            "wheel_sha256": _sha256(wheel),
            "lock_filename": lock.name,
            "lock_sha256": _sha256(lock),
            "wheelhouse_digest": _json_digest(selected_wheels),
            "runtime_tree_digest": runtime_tree["runtime_tree_digest"],
            "install_command_contract": (
                "pip-no-index-wheelhouse-require-hashes-then-wheel-no-deps-v2"
            ),
        },
    )
    _canonical_json(output / "runtime-tree.json", runtime_tree)
    payload_files = tuple(
        sorted(
            (path for path in output.rglob("*") if path.is_file()),
            key=lambda item: item.relative_to(output).as_posix(),
        )
    )
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
        "platform": f"linux-{platform.machine()}-cpu",
        "wheelhouse": selected_wheels,
        "files": [
            {
                "name": path.relative_to(output).as_posix(),
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
        or manifest.get("python_tag")
        not in {"py310", "py312"}
        or manifest.get("platform") != "linux-x86_64-cpu"
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
    selected_wheels = manifest.get("wheelhouse")
    if not isinstance(files, list) or not isinstance(selected_wheels, list):
        raise ValueError("deployment bundle file manifest is invalid")
    expected_names: set[str] = set()
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "size_bytes", "sha256"}
            or not isinstance(item["name"], str)
            or Path(item["name"]).is_absolute()
            or ".." in Path(item["name"]).parts
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
    actual_names = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    if actual_names != expected_names | {"manifest.json"}:
        raise ValueError("deployment bundle contains unmanifested files")
    directories = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_dir()
    }
    if directories != {"wheelhouse"} or any(
        path.is_symlink() for path in bundle.rglob("*")
    ):
        raise ValueError("deployment bundle layout is invalid")
    lock_names = [
        name
        for name in expected_names
        if name.startswith("runtime-py") and name.endswith("-linux.lock")
    ]
    if len(lock_names) != 1:
        raise ValueError("deployment bundle lock identity is invalid")
    locked = _locked_packages(bundle / lock_names[0])
    if _validate_wheelhouse(bundle / "wheelhouse", locked) != selected_wheels:
        raise ValueError("deployment bundle wheelhouse manifest is invalid")
    runtime_tree = _validate_runtime_tree_snapshot(
        json.loads((bundle / "runtime-tree.json").read_text(encoding="utf-8"))
    )
    installation: Any = json.loads(
        (bundle / "installation-attestation.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(installation, dict)
        or installation.get("contract") != "advar-installation-attestation-v2"
        or installation.get("runtime_tree_digest")
        != runtime_tree["runtime_tree_digest"]
        or installation.get("wheelhouse_digest") != _json_digest(selected_wheels)
        or installation.get("install_command_contract")
        != "pip-no-index-wheelhouse-require-hashes-then-wheel-no-deps-v2"
        or installation.get("source_commit") != manifest.get("source_commit")
        or installation.get("package_version") != manifest.get("package_version")
        or installation.get("lock_filename") != lock_names[0]
        or installation.get("lock_sha256") != _sha256(bundle / lock_names[0])
    ):
        raise ValueError("deployment installation attestation is invalid")
    wheel_names = [
        name
        for name in expected_names
        if name.endswith(".whl") and not name.startswith("wheelhouse/")
    ]
    if (
        len(wheel_names) != 1
        or installation.get("wheel_filename") != wheel_names[0]
        or installation.get("wheel_sha256") != _sha256(bundle / wheel_names[0])
    ):
        raise ValueError("deployment application wheel identity is invalid")
    expected_distributions = sorted(
        [
            {"name": str(item["name"]), "version": str(item["version"])}
            for item in selected_wheels
        ]
        + [
            {
                "name": "advar-radar-nowcast",
                "version": str(manifest.get("package_version", "")),
            }
        ],
        key=lambda item: item["name"],
    )
    if runtime_tree.get("distributions") != expected_distributions:
        raise ValueError("deployment runtime distributions are invalid")
    return bundle_digest


def verify_current_installation(bundle: Path) -> str:
    manifest: Any = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    installation: Any = json.loads(
        (bundle / "installation-attestation.json").read_text(encoding="utf-8")
    )
    runtime_tree = _validate_runtime_tree_snapshot(
        json.loads((bundle / "runtime-tree.json").read_text(encoding="utf-8"))
    )
    if not isinstance(manifest, dict) or not isinstance(installation, dict):
        raise ValueError("deployment installation attestation is invalid")
    lock_names = [
        str(item["name"])
        for item in manifest.get("files", [])
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and str(item["name"]).startswith("runtime-py")
        and str(item["name"]).endswith("-linux.lock")
    ]
    if len(lock_names) != 1:
        raise ValueError("deployment bundle lock identity is invalid")
    current = _validate_runtime_tree_snapshot(
        _runtime_tree_snapshot(
            locked=_locked_packages(bundle / lock_names[0]),
            application_version=str(manifest.get("package_version", "")),
        )
    )
    expected_platform = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_abi": sys.implementation.cache_tag,
    }
    if (
        current != runtime_tree
        or any(installation.get(name) != value for name, value in expected_platform.items())
        or platform.system() != "Linux"
        or manifest.get("python_tag")
        != f"py{sys.version_info.major}{sys.version_info.minor}"
        or manifest.get("platform") != f"linux-{platform.machine()}-cpu"
    ):
        raise ValueError("installed runtime disagrees with the deployment bundle")
    return str(runtime_tree["runtime_tree_digest"])


def issue_runtime_activation_receipt(
    bundle: Path,
    *,
    trusted_bundle_public_key_hex: str,
    expected_mode: BundleMode,
    expected_repository: str,
    expected_source_ref: str,
    expected_source_commit: str,
    expected_workflow_sha: str,
    expected_bundle_signer_id: str,
    deployment_instance_id: str,
    activated_at: str,
    activation_signer_id: str,
    activation_signing_key: Ed25519PrivateKey,
) -> dict[str, object]:
    """Sign the exact verified import tree installed on one deployment host."""

    if (
        not deployment_instance_id
        or deployment_instance_id.strip() != deployment_instance_id
        or not activation_signer_id
        or activation_signer_id.strip() != activation_signer_id
    ):
        raise ValueError("runtime activation identity is invalid")
    bundle_digest = verify_bundle(
        bundle,
        trusted_public_key_hex=trusted_bundle_public_key_hex,
        expected_mode=expected_mode,
        expected_repository=expected_repository,
        expected_source_ref=expected_source_ref,
        expected_source_commit=expected_source_commit,
        expected_workflow_sha=expected_workflow_sha,
        expected_signer_id=expected_bundle_signer_id,
    )
    runtime_tree_digest = verify_current_installation(bundle)
    unsigned: dict[str, object] = {
        "contract": "advar-runtime-closure-activation-receipt-v1",
        "bundle_digest": bundle_digest,
        "bundle_mode": expected_mode,
        "runtime_tree_digest": runtime_tree_digest,
        "installation_attestation_sha256": _sha256(
            bundle / "installation-attestation.json"
        ),
        "deployment_instance_id": deployment_instance_id,
        "activated_at": _canonical_utc(activated_at),
        "activation_signer_id": activation_signer_id,
    }
    return unsigned | {
        "receipt_digest": _json_digest(unsigned),
        "signature_hex": activation_signing_key.sign(
            _activation_signature_preimage(unsigned)
        ).hex(),
    }


def verify_runtime_activation_receipt(
    receipt: object,
    *,
    trusted_activation_public_key_hex: str,
    expected_bundle_digest: str,
    expected_runtime_tree_digest: str,
    expected_bundle_mode: BundleMode,
    expected_deployment_instance_id: str,
    expected_activation_signer_id: str,
) -> str:
    if not isinstance(receipt, dict):
        raise ValueError("runtime activation receipt is invalid")
    unsigned = dict(receipt)
    signature_hex = unsigned.pop("signature_hex", None)
    receipt_digest = unsigned.pop("receipt_digest", None)
    expected_keys = {
        "contract",
        "bundle_digest",
        "bundle_mode",
        "runtime_tree_digest",
        "installation_attestation_sha256",
        "deployment_instance_id",
        "activated_at",
        "activation_signer_id",
    }
    if (
        set(unsigned) != expected_keys
        or unsigned.get("contract")
        != "advar-runtime-closure-activation-receipt-v1"
        or unsigned.get("bundle_digest") != expected_bundle_digest
        or unsigned.get("runtime_tree_digest") != expected_runtime_tree_digest
        or unsigned.get("bundle_mode") != expected_bundle_mode
        or unsigned.get("deployment_instance_id")
        != expected_deployment_instance_id
        or unsigned.get("activation_signer_id")
        != expected_activation_signer_id
        or not isinstance(unsigned.get("activated_at"), str)
        or _canonical_utc(str(unsigned["activated_at"]))
        != unsigned["activated_at"]
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(unsigned.get("installation_attestation_sha256")),
        )
        is None
        or receipt_digest != _json_digest(unsigned)
        or not isinstance(signature_hex, str)
    ):
        raise ValueError("runtime activation receipt identity is invalid")
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(trusted_activation_public_key_hex)
        ).verify(
            bytes.fromhex(signature_hex),
            _activation_signature_preimage(unsigned),
        )
    except (InvalidSignature, TypeError, ValueError) as error:
        raise ValueError("runtime activation receipt signature is untrusted") from error
    return str(receipt_digest)


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
    build.add_argument("--wheelhouse", type=Path, required=True)
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
    verify.add_argument("--verify-installation", action="store_true")
    activate = subparsers.add_parser("activate-runtime")
    activate.add_argument("--bundle", type=Path, required=True)
    activate.add_argument("--trusted-bundle-public-key", type=Path, required=True)
    activate.add_argument(
        "--expected-mode",
        choices=("candidate-smoke", "deployable"),
        required=True,
    )
    activate.add_argument("--expected-repository", required=True)
    activate.add_argument("--expected-source-ref", required=True)
    activate.add_argument("--expected-source-commit", required=True)
    activate.add_argument("--expected-workflow-sha", required=True)
    activate.add_argument("--expected-bundle-signer-id", required=True)
    activate.add_argument("--deployment-instance-id", required=True)
    activate.add_argument("--activated-at")
    activate.add_argument("--activation-signer-id", required=True)
    activate.add_argument(
        "--activation-signing-private-key", type=Path, required=True
    )
    activate.add_argument("--receipt", type=Path, required=True)
    verify_activation = subparsers.add_parser("verify-runtime-activation")
    verify_activation.add_argument("--receipt", type=Path, required=True)
    verify_activation.add_argument(
        "--trusted-activation-public-key", type=Path, required=True
    )
    verify_activation.add_argument("--expected-bundle-digest", required=True)
    verify_activation.add_argument("--expected-runtime-tree-digest", required=True)
    verify_activation.add_argument(
        "--expected-bundle-mode",
        choices=("candidate-smoke", "deployable"),
        required=True,
    )
    verify_activation.add_argument(
        "--expected-deployment-instance-id", required=True
    )
    verify_activation.add_argument(
        "--expected-activation-signer-id", required=True
    )
    arguments = parser.parse_args()
    try:
        if arguments.command == "build":
            build_bundle(
                wheel=arguments.wheel.resolve(),
                lock=arguments.lock.resolve(),
                audit=arguments.audit.resolve(),
                wheelhouse=arguments.wheelhouse.resolve(),
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
        elif arguments.command == "verify":
            public_key_hex = _load_trusted_public_key(
                arguments.trusted_public_key.absolute(),
                deployable=arguments.expected_mode == "deployable",
            )
            verified_digest = verify_bundle(
                arguments.bundle.absolute(),
                trusted_public_key_hex=public_key_hex,
                expected_mode=arguments.expected_mode,
                expected_repository=arguments.expected_repository,
                expected_source_ref=arguments.expected_source_ref,
                expected_source_commit=arguments.expected_source_commit,
                expected_workflow_sha=arguments.expected_workflow_sha,
                expected_signer_id=arguments.expected_signer_id,
            )
            if arguments.verify_installation:
                print(verify_current_installation(arguments.bundle.absolute()))
            else:
                print(verified_digest)
        elif arguments.command == "activate-runtime":
            bundle_public_key = _load_trusted_public_key(
                arguments.trusted_bundle_public_key.absolute(),
                deployable=arguments.expected_mode == "deployable",
            )
            activated_at = arguments.activated_at or (
                datetime.now(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
            receipt = issue_runtime_activation_receipt(
                arguments.bundle.absolute(),
                trusted_bundle_public_key_hex=bundle_public_key,
                expected_mode=arguments.expected_mode,
                expected_repository=arguments.expected_repository,
                expected_source_ref=arguments.expected_source_ref,
                expected_source_commit=arguments.expected_source_commit,
                expected_workflow_sha=arguments.expected_workflow_sha,
                expected_bundle_signer_id=(
                    arguments.expected_bundle_signer_id
                ),
                deployment_instance_id=arguments.deployment_instance_id,
                activated_at=activated_at,
                activation_signer_id=arguments.activation_signer_id,
                activation_signing_key=_load_private_key(
                    arguments.activation_signing_private_key.absolute()
                ),
            )
            _canonical_json(arguments.receipt.absolute(), receipt)
            print(receipt["receipt_digest"])
        else:
            activation_public_key = _load_trusted_public_key(
                arguments.trusted_activation_public_key.absolute(),
                deployable=arguments.expected_bundle_mode == "deployable",
            )
            receipt_value: Any = json.loads(
                arguments.receipt.read_text(encoding="utf-8")
            )
            print(
                verify_runtime_activation_receipt(
                    receipt_value,
                    trusted_activation_public_key_hex=activation_public_key,
                    expected_bundle_digest=arguments.expected_bundle_digest,
                    expected_runtime_tree_digest=(
                        arguments.expected_runtime_tree_digest
                    ),
                    expected_bundle_mode=arguments.expected_bundle_mode,
                    expected_deployment_instance_id=(
                        arguments.expected_deployment_instance_id
                    ),
                    expected_activation_signer_id=(
                        arguments.expected_activation_signer_id
                    ),
                )
            )
    except (OSError, TypeError, ValueError) as error:
        print(f"deployment bundle generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
