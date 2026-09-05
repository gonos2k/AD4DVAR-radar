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
import site
import stat
import subprocess
import sys
import sysconfig
from typing import Any, Literal
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from advar.runtime_closure import (
    active_import_path_snapshot,
    validate_current_runtime_closure,
)


BUNDLE_CONTRACT = "advar-linux-cpu-deployment-bundle-v4"
SIGNATURE_DOMAIN = b"advar-linux-cpu-deployment-bundle-v4\x00"
RELEASE_APPROVAL_SIGNATURE_DOMAIN = (
    b"ADVAR_DEPLOYMENT_BUNDLE_RELEASE_APPROVAL_V1\x00"
)
ACTIVATION_SIGNATURE_DOMAIN = b"ADVAR_DEPLOYMENT_RUNTIME_ACTIVATION_V3\x00"
RUNTIME_ACTIVATION_GENESIS_DIGEST = hashlib.sha256(
    b"ADVAR_DEPLOYMENT_RUNTIME_ACTIVATION_GENESIS_V1\x00"
).hexdigest()
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


def _is_forbidden_runtime_artifact(relative: Path) -> bool:
    return "__pycache__" in relative.parts or relative.suffix in {
        ".pyc",
        ".pyo",
        ".pth",
    }


def _is_canonical_runtime_omission(relative: Path) -> bool:
    return relative.name == "__pycache__" or relative.suffix in {
        ".pyc",
        ".pyo",
        ".pth",
    }


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


def _wheel_version_matches_lock(
    *,
    locked_version: str,
    wheel_version: str,
) -> bool:
    """Match PEP 440 public pins while preserving an authenticated local tag."""
    if wheel_version == locked_version:
        return True
    if "+" in locked_version:
        return False
    return re.fullmatch(
        rf"{re.escape(locked_version)}\+[a-z0-9]+(?:[._-][a-z0-9]+)*",
        wheel_version,
        flags=re.IGNORECASE,
    ) is not None


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
            or not _wheel_version_matches_lock(
                locked_version=str(package["version"]),
                wheel_version=version,
            )
            or not isinstance(allowed_hashes, list)
            or digest not in allowed_hashes
        ):
            raise ValueError("deployment wheelhouse disagrees with its lock")
        seen.add(name)
        selected.append(
            {
                "name": name,
                "locked_version": str(package["version"]),
                "wheel_version": version,
                "filename": wheel.name,
                "size_bytes": wheel.stat().st_size,
                "sha256": digest,
            }
        )
    if seen != set(expected):
        raise ValueError("deployment wheelhouse is incomplete")
    return selected


def _runtime_import_roots() -> tuple[Path, ...]:
    candidates = {
        Path(value).absolute()
        for value in (
            *site.getsitepackages(),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        )
        if value
    }
    return tuple(sorted(candidates, key=str))


def _require_runtime_permissions(path: Path, *, deployable: bool) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("deployment runtime contains a symlink")
    if deployable and (metadata.st_uid != 0 or metadata.st_mode & 0o022):
        raise ValueError(
            "deployable runtime must be root-owned and non-writable"
        )
    return metadata


def _require_deployable_ancestry(path: Path) -> None:
    for retained in (path, *path.parents):
        metadata = retained.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError(
                "deployable runtime ancestry must be root-owned and non-writable"
            )
        if retained == retained.parent:
            break


def _runtime_file_snapshot(
    path: Path,
    *,
    deployable: bool,
) -> tuple[int, str]:
    if deployable:
        _require_deployable_ancestry(path.parent)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("deployment runtime file cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (deployable and (before.st_uid != 0 or before.st_mode & 0o022))
        ):
            raise ValueError("deployment runtime file is not immutable and regular")
        digest = hashlib.sha256()
        retained_size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            retained_size += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_mode,
            before.st_uid,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_mode,
            after.st_uid,
        )
        if retained_size != before.st_size or before_identity != after_identity:
            raise ValueError("deployment runtime file changed during validation")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _linked_native_libraries(
    paths: tuple[Path, ...],
    *,
    deployable: bool,
    import_roots: tuple[Path, ...],
) -> list[dict[str, object]]:
    if platform.system() != "Linux":
        return []
    canonical_import_roots = tuple(
        root.resolve(strict=True) for root in import_roots
    )
    libraries: set[Path] = set()
    for source in paths:
        try:
            completed = subprocess.run(
                ("ldd", str(source)),
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("deployment native library closure is unreadable") from error
        for line in completed.stdout.splitlines():
            match = re.search(r"(?:=>\s+)?(/[^\s]+)", line)
            if match is None:
                continue
            library = Path(match.group(1)).resolve(strict=True)
            libraries.add(library)
    result = []
    for path in sorted(libraries, key=lambda item: item.as_posix()):
        size_bytes, sha256 = _runtime_file_snapshot(
            path,
            deployable=deployable,
        )
        runtime_root_index = next(
            (
                index
                for index, root in enumerate(canonical_import_roots)
                if path.is_relative_to(root)
            ),
            None,
        )
        identity_path = path.as_posix()
        if runtime_root_index is not None:
            runtime_root = canonical_import_roots[runtime_root_index]
            identity_path = (
                f"site-{runtime_root_index}/"
                f"{path.relative_to(runtime_root).as_posix()}"
            )
        result.append(
            {
                "name": path.name,
                "path": identity_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
    result.sort(key=lambda item: (str(item["path"]), str(item["sha256"])))
    return result


def _interpreter_closure_snapshot(
    *,
    native_extension_paths: tuple[Path, ...],
    deployable: bool,
) -> dict[str, object]:
    if not sys.dont_write_bytecode:
        raise ValueError("deployment Python must disable bytecode writes")
    executable = Path(sys.executable).resolve(strict=True)
    executable_size, executable_sha256 = _runtime_file_snapshot(
        executable,
        deployable=deployable,
    )
    stdlib_root = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    if deployable:
        _require_deployable_ancestry(stdlib_root)
    _require_runtime_permissions(stdlib_root, deployable=deployable)
    stdlib_files: list[dict[str, object]] = []
    import_roots = _runtime_import_roots()
    for path in sorted(stdlib_root.rglob("*"), key=lambda item: item.as_posix()):
        if any(path == root or root in path.parents for root in import_roots):
            continue
        metadata = _require_runtime_permissions(path, deployable=deployable)
        if stat.S_ISDIR(metadata.st_mode):
            continue
        size_bytes, sha256 = _runtime_file_snapshot(
            path,
            deployable=deployable,
        )
        relative = path.relative_to(stdlib_root).as_posix()
        stdlib_files.append(
            {
                "path": relative,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
    unsigned: dict[str, object] = {
        "contract": "advar-python-interpreter-closure-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_abi": sys.implementation.cache_tag,
        "bytecode_write_disabled": True,
        "executable_size_bytes": executable_size,
        "executable_sha256": executable_sha256,
        "active_import_path": active_import_path_snapshot(
            import_roots=import_roots,
            stdlib_root=stdlib_root,
            deployable=deployable,
        ),
        "stdlib_files": stdlib_files,
        "native_libraries": _linked_native_libraries(
            (executable, *native_extension_paths),
            deployable=deployable,
            import_roots=import_roots,
        ),
    }
    return unsigned | {"interpreter_closure_digest": _json_digest(unsigned)}


def _runtime_tree_snapshot(
    *,
    selected_wheels: list[dict[str, object]],
    application_version: str,
    runtime_mode: BundleMode = "candidate-smoke",
) -> dict[str, object]:
    distributions = [
        {
            "name": _normalized_distribution_name(str(item["name"])),
            "version": str(item["wheel_version"]),
        }
        for item in selected_wheels
    ] + [
        {"name": "advar-radar-nowcast", "version": application_version}
    ]
    distributions.sort(key=lambda item: str(item["name"]))
    roots = _runtime_import_roots()
    if not roots:
        raise ValueError("deployment runtime has no import roots")
    deployable = runtime_mode == "deployable"
    files: list[dict[str, object]] = []
    seen_claims: set[tuple[Path, str]] = set()
    claimed_paths: dict[tuple[Path, str], str] = {}
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
            if relative is None or ".." in relative.parts:
                continue
            root = next(root for root in roots if located.is_relative_to(root))
            key = (root, relative.as_posix())
            previous = claimed_paths.setdefault(key, name)
            if previous != name:
                raise ValueError("deployment runtime file identity is duplicated")
    actual_distributions = {
        _normalized_distribution_name(distribution.metadata["Name"])
        for root in roots
        for distribution in importlib.metadata.distributions(path=[str(root)])
        if distribution.metadata.get("Name")
    }
    expected_distribution_names = {
        str(item["name"]) for item in distributions
    }
    if actual_distributions != expected_distribution_names:
        raise ValueError("deployment runtime contains an extra distribution")
    native_extensions: list[Path] = []
    excluded_metadata = {"RECORD", "direct_url.json"}
    for root_index, root in enumerate(roots):
        if deployable:
            _require_deployable_ancestry(root)
        root_metadata = _require_runtime_permissions(root, deployable=deployable)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError("deployment import root is not a directory")
        for located in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            metadata = _require_runtime_permissions(located, deployable=deployable)
            relative = located.relative_to(root)
            if stat.S_ISDIR(metadata.st_mode):
                if (root, relative.as_posix()) in claimed_paths:
                    seen_claims.add((root, relative.as_posix()))
                continue
            if (
                _is_forbidden_runtime_artifact(relative)
                or located.name in {"sitecustomize.py", "usercustomize.py"}
            ):
                raise ValueError(
                    "deployment runtime contains a forbidden import artifact: "
                    f"site-{root_index}/{relative.as_posix()}"
                )
            owner = claimed_paths.get((root, relative.as_posix()))
            if owner is None:
                raise ValueError("deployment runtime contains an unowned file")
            seen_claims.add((root, relative.as_posix()))
            if (
                len(relative.parts) >= 2
                and relative.parts[-2].endswith(".dist-info")
                and relative.name in excluded_metadata
            ):
                continue
            if located.suffix in {".so", ".pyd", ".dylib"}:
                native_extensions.append(located)
            size_bytes, sha256 = _runtime_file_snapshot(
                located,
                deployable=deployable,
            )
            files.append(
                {
                    "distribution": owner,
                    "path": f"site-{root_index}/{relative.as_posix()}",
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            )
    files.sort(key=lambda item: (str(item["distribution"]), str(item["path"])))
    omitted_forbidden_files: list[dict[str, str]] = []
    for (root, relative_text), owner in claimed_paths.items():
        if (root, relative_text) in seen_claims:
            continue
        relative = Path(relative_text)
        if not _is_canonical_runtime_omission(relative):
            raise ValueError("deployment runtime is missing a locked file")
        omitted_forbidden_files.append(
            {
                "distribution": owner,
                "path": (
                    f"site-{roots.index(root)}/{relative.as_posix()}"
                ),
            }
        )
    omitted_forbidden_files.sort(
        key=lambda item: (item["distribution"], item["path"])
    )
    interpreter_closure = _interpreter_closure_snapshot(
        native_extension_paths=tuple(native_extensions),
        deployable=deployable,
    )
    unsigned: dict[str, object] = {
        "contract": "advar-import-runtime-tree-v2",
        "runtime_mode": runtime_mode,
        "import_root_count": len(roots),
        "distributions": distributions,
        "files": files,
        "omitted_forbidden_files": omitted_forbidden_files,
        "interpreter_closure": interpreter_closure,
    }
    return unsigned | {"runtime_tree_digest": _json_digest(unsigned)}


def _validate_runtime_tree_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("runtime tree attestation is invalid")
    unsigned = dict(value)
    digest = unsigned.pop("runtime_tree_digest", None)
    distributions = unsigned.get("distributions")
    files = unsigned.get("files")
    omitted_forbidden_files = unsigned.get("omitted_forbidden_files")
    interpreter_closure = unsigned.get("interpreter_closure")
    if (
        unsigned.get("contract") != "advar-import-runtime-tree-v2"
        or set(unsigned)
        != {
            "contract",
            "runtime_mode",
            "import_root_count",
            "distributions",
            "files",
            "omitted_forbidden_files",
            "interpreter_closure",
        }
        or unsigned.get("runtime_mode") not in {"candidate-smoke", "deployable"}
        or not isinstance(unsigned.get("import_root_count"), int)
        or int(unsigned["import_root_count"]) <= 0
        or not isinstance(distributions, list)
        or not distributions
        or not isinstance(files, list)
        or not files
        or not isinstance(omitted_forbidden_files, list)
        or not isinstance(digest, str)
        or digest != _json_digest(unsigned)
        or not isinstance(interpreter_closure, dict)
        or interpreter_closure.get("contract")
        != "advar-python-interpreter-closure-v1"
        or interpreter_closure.get("bytecode_write_disabled") is not True
        or interpreter_closure.get("interpreter_closure_digest")
        != _json_digest(
            {
                key: value
                for key, value in interpreter_closure.items()
                if key != "interpreter_closure_digest"
            }
        )
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
            or re.fullmatch(r"site-[0-9]+/.+", item["path"]) is None
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
    previous_omission: tuple[str, str] = ("", "")
    retained_paths = {str(item["path"]) for item in files}
    for item in omitted_forbidden_files:
        if (
            not isinstance(item, dict)
            or set(item) != {"distribution", "path"}
            or not isinstance(item["distribution"], str)
            or item["distribution"] not in distribution_names
            or not isinstance(item["path"], str)
            or re.fullmatch(r"site-[0-9]+/.+", item["path"]) is None
            or ".." in Path(item["path"]).parts
            or item["path"] in retained_paths
        ):
            raise ValueError("runtime tree omission is invalid")
        relative = Path(item["path"].split("/", 1)[1])
        if not _is_canonical_runtime_omission(relative):
            raise ValueError("runtime tree omission is not forbidden")
        current = (item["distribution"], item["path"])
        if current <= previous_omission:
            raise ValueError("runtime tree omissions are noncanonical")
        previous_omission = current
    return value


def _runtime_tree_mismatch_fields(
    expected: dict[str, object],
    current: dict[str, object],
) -> tuple[str, ...]:
    """Return only schema field names, never runtime paths or file digests."""
    mismatches: list[str] = []
    for name in sorted(set(expected) | set(current)):
        if name == "runtime_tree_digest" or expected.get(name) == current.get(name):
            continue
        if name != "interpreter_closure":
            mismatches.append(name)
            continue
        expected_interpreter = expected.get(name)
        current_interpreter = current.get(name)
        if not isinstance(expected_interpreter, dict) or not isinstance(
            current_interpreter, dict
        ):
            mismatches.append(name)
            continue
        for interpreter_name in sorted(
            set(expected_interpreter) | set(current_interpreter)
        ):
            if interpreter_name == "interpreter_closure_digest":
                continue
            if expected_interpreter.get(interpreter_name) != current_interpreter.get(
                interpreter_name
            ):
                mismatches.append(f"{name}.{interpreter_name}")
    return tuple(mismatches)


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


def _release_approval_signature_preimage(
    unsigned_approval: dict[str, object],
) -> bytes:
    return RELEASE_APPROVAL_SIGNATURE_DOMAIN + _canonical_bytes(
        unsigned_approval
    )


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
            selected_wheels=selected_wheels,
            application_version=package_version,
            runtime_mode=mode,
        )
    )
    if platform.machine() != "x86_64":
        raise ValueError("current deployment bundle requires Linux x86_64")
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
                    "version": item["wheel_version"],
                    "hashes": [
                        {"alg": "SHA-256", "content": item["sha256"]}
                    ],
                    "properties": [
                        {
                            "name": "advar:locked-public-version",
                            "value": item["locked_version"],
                        }
                    ],
                }
                for item in selected_wheels
            ],
        },
    )
    installation_output = output / "installation-attestation.json"
    _canonical_json(
        installation_output,
        {
            "contract": "advar-installation-attestation-v3",
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
            "interpreter_closure_digest": runtime_tree[
                "interpreter_closure"
            ]["interpreter_closure_digest"],
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
        or installation.get("contract") != "advar-installation-attestation-v3"
        or installation.get("runtime_tree_digest")
        != runtime_tree["runtime_tree_digest"]
        or installation.get("interpreter_closure_digest")
        != runtime_tree["interpreter_closure"]["interpreter_closure_digest"]
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
            {
                "name": str(item["name"]),
                "version": str(item["wheel_version"]),
            }
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


def issue_bundle_release_approval(
    bundle: Path,
    *,
    trusted_bundle_public_key_hex: str,
    expected_mode: BundleMode,
    expected_repository: str,
    expected_source_ref: str,
    expected_source_commit: str,
    expected_workflow_sha: str,
    expected_bundle_signer_id: str,
    approved_at: str,
    expires_at: str,
    release_authority_id: str,
    release_authority_trust_store_digest: str,
    release_signing_key: Ed25519PrivateKey,
) -> dict[str, object]:
    """Approve one verified bundle with a detached release-authority proof."""

    deployment_bundle_digest = verify_bundle(
        bundle,
        trusted_public_key_hex=trusted_bundle_public_key_hex,
        expected_mode=expected_mode,
        expected_repository=expected_repository,
        expected_source_ref=expected_source_ref,
        expected_source_commit=expected_source_commit,
        expected_workflow_sha=expected_workflow_sha,
        expected_signer_id=expected_bundle_signer_id,
    )
    manifest_value: Any = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(manifest_value, dict):
        raise ValueError("deployment bundle manifest is invalid")
    canonical_approved_at = _canonical_utc(approved_at)
    canonical_expires_at = _canonical_utc(expires_at)
    approved = datetime.fromisoformat(
        canonical_approved_at.replace("Z", "+00:00")
    )
    expiry = datetime.fromisoformat(
        canonical_expires_at.replace("Z", "+00:00")
    )
    if (
        re.fullmatch(
            r"[0-9a-f]{64}", release_authority_trust_store_digest
        )
        is None
        or not release_authority_id
        or release_authority_id.strip() != release_authority_id
        or approved >= expiry
        or approved > datetime.now(timezone.utc)
    ):
        raise ValueError("deployment bundle release approval is invalid")
    unsigned: dict[str, object] = {
        "deployment_bundle_digest": deployment_bundle_digest,
        "bundle_manifest_digest": _json_digest(manifest_value),
        "source_commit": expected_source_commit,
        "repository": expected_repository,
        "source_ref": expected_source_ref,
        "platform": str(manifest_value.get("platform", "")),
        "runtime_mode": expected_mode,
        "approved_at": canonical_approved_at,
        "expires_at": canonical_expires_at,
        "authority_id": release_authority_id,
        "authority_public_key_hex": (
            release_signing_key.public_key().public_bytes_raw().hex()
        ),
        "authority_trust_store_digest": (
            release_authority_trust_store_digest
        ),
        "contract": "deployment-bundle-release-approval-v1",
    }
    payload = unsigned | {
        "authority_signature_hex": release_signing_key.sign(
            _release_approval_signature_preimage(unsigned)
        ).hex(),
    }
    return payload | {"approval_digest": _json_digest(payload)}


def verify_bundle_release_approval(
    approval: object,
    *,
    trusted_release_public_key_hex: str,
    expected_deployment_bundle_digest: str,
    expected_bundle_manifest_digest: str,
    expected_repository: str,
    expected_source_ref: str,
    expected_source_commit: str,
    expected_platform: str,
    expected_mode: BundleMode,
    expected_release_authority_id: str,
    expected_release_authority_trust_store_digest: str,
    required_valid_through: str,
) -> str:
    """Verify the detached release proof and its exact bundle identity."""

    if not isinstance(approval, dict):
        raise ValueError("deployment bundle release approval is invalid")
    payload = dict(approval)
    approval_digest = payload.pop("approval_digest", None)
    signature_hex = payload.get("authority_signature_hex")
    unsigned = dict(payload)
    unsigned.pop("authority_signature_hex", None)
    expected_keys = {
        "deployment_bundle_digest",
        "bundle_manifest_digest",
        "source_commit",
        "repository",
        "source_ref",
        "platform",
        "runtime_mode",
        "approved_at",
        "expires_at",
        "authority_id",
        "authority_public_key_hex",
        "authority_trust_store_digest",
        "contract",
    }
    approved_at = unsigned.get("approved_at")
    expires_at = unsigned.get("expires_at")
    required = _canonical_utc(required_valid_through)
    if (
        set(unsigned) != expected_keys
        or unsigned.get("contract")
        != "deployment-bundle-release-approval-v1"
        or unsigned.get("deployment_bundle_digest")
        != expected_deployment_bundle_digest
        or unsigned.get("bundle_manifest_digest")
        != expected_bundle_manifest_digest
        or unsigned.get("source_commit") != expected_source_commit
        or unsigned.get("repository") != expected_repository
        or unsigned.get("source_ref") != expected_source_ref
        or unsigned.get("platform") != expected_platform
        or unsigned.get("runtime_mode") != expected_mode
        or unsigned.get("authority_id") != expected_release_authority_id
        or unsigned.get("authority_public_key_hex")
        != trusted_release_public_key_hex
        or unsigned.get("authority_trust_store_digest")
        != expected_release_authority_trust_store_digest
        or not isinstance(approved_at, str)
        or _canonical_utc(approved_at) != approved_at
        or not isinstance(expires_at, str)
        or _canonical_utc(expires_at) != expires_at
        or datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
        >= datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        or datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
        > datetime.now(timezone.utc)
        or datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
        > datetime.fromisoformat(required.replace("Z", "+00:00"))
        or datetime.fromisoformat(required.replace("Z", "+00:00"))
        > datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        or approval_digest != _json_digest(payload)
        or not isinstance(signature_hex, str)
    ):
        raise ValueError("deployment bundle release approval identity is invalid")
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(trusted_release_public_key_hex)
        ).verify(
            bytes.fromhex(signature_hex),
            _release_approval_signature_preimage(unsigned),
        )
    except (InvalidSignature, TypeError, ValueError) as error:
        raise ValueError(
            "deployment bundle release approval signature is untrusted"
        ) from error
    return str(approval_digest)


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
    selected_wheels = _validate_wheelhouse(
        bundle / "wheelhouse",
        _locked_packages(bundle / lock_names[0]),
    )
    if selected_wheels != manifest.get("wheelhouse"):
        raise ValueError("deployment bundle wheelhouse manifest is invalid")
    current = _validate_runtime_tree_snapshot(
        _runtime_tree_snapshot(
            selected_wheels=selected_wheels,
            application_version=str(manifest.get("package_version", "")),
            runtime_mode=(
                "deployable"
                if manifest.get("mode") == "deployable"
                else "candidate-smoke"
            ),
        )
    )
    expected_platform = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_abi": sys.implementation.cache_tag,
    }
    runtime_mismatches = _runtime_tree_mismatch_fields(runtime_tree, current)
    if (
        runtime_mismatches
        or any(installation.get(name) != value for name, value in expected_platform.items())
        or platform.system() != "Linux"
        or manifest.get("python_tag")
        != f"py{sys.version_info.major}{sys.version_info.minor}"
        or manifest.get("platform") != f"linux-{platform.machine()}-cpu"
    ):
        detail = ", ".join(runtime_mismatches) or "platform_identity"
        raise ValueError(
            "installed runtime disagrees with the deployment bundle: " + detail
        )
    interpreter = runtime_tree["interpreter_closure"]
    assert isinstance(interpreter, dict)
    runtime_mode: BundleMode = (
        "deployable"
        if manifest.get("mode") == "deployable"
        else "candidate-smoke"
    )
    validate_current_runtime_closure(
        expected_runtime_tree_digest=str(runtime_tree["runtime_tree_digest"]),
        expected_interpreter_closure_digest=str(
            interpreter["interpreter_closure_digest"]
        ),
        runtime_mode=runtime_mode,
    )
    return str(runtime_tree["runtime_tree_digest"])


def issue_runtime_activation_receipt(
    bundle: Path,
    *,
    release_approval: object,
    trusted_bundle_public_key_hex: str,
    trusted_release_public_key_hex: str,
    expected_mode: BundleMode,
    expected_repository: str,
    expected_source_ref: str,
    expected_source_commit: str,
    expected_workflow_sha: str,
    expected_bundle_signer_id: str,
    deployment_instance_digest: str,
    host_identity_digest: str,
    activation_sequence_number: int,
    previous_activation_receipt_digest: str,
    rollback_reason_digest: str | None,
    activated_at: str,
    expires_at: str,
    expected_release_authority_id: str,
    expected_release_authority_trust_store_digest: str,
    activation_authority_id: str,
    activation_authority_trust_store_digest: str,
    activation_signing_key: Ed25519PrivateKey,
) -> dict[str, object]:
    """Sign the exact verified import tree installed on one deployment host."""

    if (
        re.fullmatch(r"[0-9a-f]{64}", deployment_instance_digest) is None
        or re.fullmatch(r"[0-9a-f]{64}", host_identity_digest) is None
        or re.fullmatch(
            r"[0-9a-f]{64}", activation_authority_trust_store_digest
        )
        is None
        or isinstance(activation_sequence_number, bool)
        or activation_sequence_number <= 0
        or re.fullmatch(
            r"[0-9a-f]{64}", previous_activation_receipt_digest
        )
        is None
        or (
            rollback_reason_digest is not None
            and re.fullmatch(r"[0-9a-f]{64}", rollback_reason_digest) is None
        )
        or not activation_authority_id
        or activation_authority_id.strip() != activation_authority_id
    ):
        raise ValueError("runtime activation identity is invalid")
    activation_public_key_hex = (
        activation_signing_key.public_key().public_bytes_raw().hex()
    )
    if activation_public_key_hex == trusted_release_public_key_hex:
        raise ValueError(
            "release approval and runtime activation require separate keys"
        )
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
    runtime_tree: Any = json.loads(
        (bundle / "runtime-tree.json").read_text(encoding="utf-8")
    )
    manifest_value: Any = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(runtime_tree, dict) or not isinstance(
        runtime_tree.get("interpreter_closure"), dict
    ) or not isinstance(manifest_value, dict):
        raise ValueError("runtime interpreter closure is invalid")
    activated = _canonical_utc(activated_at)
    expiry = _canonical_utc(expires_at)
    activated_datetime = datetime.fromisoformat(
        activated.replace("Z", "+00:00")
    )
    if (
        activated_datetime >= datetime.fromisoformat(
            expiry.replace("Z", "+00:00")
        )
        or activated_datetime > datetime.now(timezone.utc)
        or (
            activation_sequence_number == 1
            and previous_activation_receipt_digest
            != RUNTIME_ACTIVATION_GENESIS_DIGEST
        )
        or (
            activation_sequence_number > 1
            and previous_activation_receipt_digest
            == RUNTIME_ACTIVATION_GENESIS_DIGEST
        )
    ):
        raise ValueError("runtime activation expiry is invalid")
    release_approval_digest = verify_bundle_release_approval(
        release_approval,
        trusted_release_public_key_hex=trusted_release_public_key_hex,
        expected_deployment_bundle_digest=bundle_digest,
        expected_bundle_manifest_digest=_json_digest(manifest_value),
        expected_repository=expected_repository,
        expected_source_ref=expected_source_ref,
        expected_source_commit=expected_source_commit,
        expected_platform=str(manifest_value.get("platform", "")),
        expected_mode=expected_mode,
        expected_release_authority_id=expected_release_authority_id,
        expected_release_authority_trust_store_digest=(
            expected_release_authority_trust_store_digest
        ),
        required_valid_through=expiry,
    )
    assert isinstance(release_approval, dict)
    if datetime.fromisoformat(
        str(release_approval["approved_at"]).replace("Z", "+00:00")
    ) > activated_datetime:
        raise ValueError("runtime activation chronology is invalid")
    unsigned: dict[str, object] = {
        "deployment_bundle_digest": bundle_digest,
        "release_approval_digest": release_approval_digest,
        "runtime_tree_digest": runtime_tree_digest,
        "interpreter_closure_digest": runtime_tree[
            "interpreter_closure"
        ]["interpreter_closure_digest"],
        "installation_attestation_sha256": _sha256(
            bundle / "installation-attestation.json"
        ),
        "deployment_instance_digest": deployment_instance_digest,
        "host_identity_digest": host_identity_digest,
        "runtime_mode": expected_mode,
        "activation_sequence_number": activation_sequence_number,
        "previous_activation_receipt_digest": (
            previous_activation_receipt_digest
        ),
        "rollback_reason_digest": rollback_reason_digest,
        "activated_at": activated,
        "expires_at": expiry,
        "authority_id": activation_authority_id,
        "authority_public_key_hex": activation_public_key_hex,
        "authority_trust_store_digest": (
            activation_authority_trust_store_digest
        ),
        "contract": "deployment-runtime-activation-receipt-v3",
    }
    payload = unsigned | {
        "authority_signature_hex": activation_signing_key.sign(
            _activation_signature_preimage(unsigned)
        ).hex(),
    }
    return payload | {"receipt_digest": _json_digest(payload)}


def verify_runtime_activation_receipt(
    receipt: object,
    *,
    release_approval: object,
    trusted_release_public_key_hex: str,
    trusted_activation_public_key_hex: str,
    expected_deployment_bundle_digest: str,
    expected_bundle_manifest_digest: str,
    expected_runtime_tree_digest: str,
    expected_interpreter_closure_digest: str,
    expected_deployment_instance_digest: str,
    expected_host_identity_digest: str,
    expected_mode: BundleMode,
    expected_activation_sequence_number: int,
    expected_previous_activation_receipt_digest: str,
    expected_rollback_reason_digest: str | None,
    expected_release_authority_id: str,
    expected_release_authority_trust_store_digest: str,
    expected_activation_authority_id: str,
    expected_activation_authority_trust_store_digest: str,
) -> str:
    if not isinstance(receipt, dict):
        raise ValueError("runtime activation receipt is invalid")
    payload = dict(receipt)
    receipt_digest = payload.pop("receipt_digest", None)
    signature_hex = payload.get("authority_signature_hex")
    unsigned = dict(payload)
    unsigned.pop("authority_signature_hex", None)
    expected_keys = {
        "contract",
        "deployment_bundle_digest",
        "release_approval_digest",
        "runtime_tree_digest",
        "interpreter_closure_digest",
        "installation_attestation_sha256",
        "deployment_instance_digest",
        "host_identity_digest",
        "runtime_mode",
        "activation_sequence_number",
        "previous_activation_receipt_digest",
        "rollback_reason_digest",
        "activated_at",
        "expires_at",
        "authority_id",
        "authority_public_key_hex",
        "authority_trust_store_digest",
    }
    if (
        set(unsigned) != expected_keys
        or unsigned.get("contract")
        != "deployment-runtime-activation-receipt-v3"
        or unsigned.get("deployment_bundle_digest")
        != expected_deployment_bundle_digest
        or unsigned.get("runtime_tree_digest") != expected_runtime_tree_digest
        or unsigned.get("interpreter_closure_digest")
        != expected_interpreter_closure_digest
        or unsigned.get("deployment_instance_digest")
        != expected_deployment_instance_digest
        or unsigned.get("host_identity_digest")
        != expected_host_identity_digest
        or unsigned.get("runtime_mode") != expected_mode
        or unsigned.get("activation_sequence_number")
        != expected_activation_sequence_number
        or unsigned.get("previous_activation_receipt_digest")
        != expected_previous_activation_receipt_digest
        or unsigned.get("rollback_reason_digest")
        != expected_rollback_reason_digest
        or unsigned.get("authority_id")
        != expected_activation_authority_id
        or unsigned.get("authority_public_key_hex")
        != trusted_activation_public_key_hex
        or unsigned.get("authority_trust_store_digest")
        != expected_activation_authority_trust_store_digest
        or not isinstance(unsigned.get("activated_at"), str)
        or _canonical_utc(str(unsigned["activated_at"]))
        != unsigned["activated_at"]
        or not isinstance(unsigned.get("expires_at"), str)
        or _canonical_utc(str(unsigned["expires_at"]))
        != unsigned["expires_at"]
        or datetime.fromisoformat(
            str(unsigned["activated_at"]).replace("Z", "+00:00")
        )
        >= datetime.fromisoformat(
            str(unsigned["expires_at"]).replace("Z", "+00:00")
        )
        or datetime.fromisoformat(
            str(unsigned["activated_at"]).replace("Z", "+00:00")
        )
        > datetime.now(timezone.utc)
        or (
            expected_activation_sequence_number == 1
            and expected_previous_activation_receipt_digest
            != RUNTIME_ACTIVATION_GENESIS_DIGEST
        )
        or (
            expected_activation_sequence_number > 1
            and expected_previous_activation_receipt_digest
            == RUNTIME_ACTIVATION_GENESIS_DIGEST
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(unsigned.get("installation_attestation_sha256")),
        )
        is None
        or receipt_digest != _json_digest(payload)
        or not isinstance(signature_hex, str)
    ):
        raise ValueError("runtime activation receipt identity is invalid")
    release_approval_digest = verify_bundle_release_approval(
        release_approval,
        trusted_release_public_key_hex=trusted_release_public_key_hex,
        expected_deployment_bundle_digest=expected_deployment_bundle_digest,
        expected_bundle_manifest_digest=expected_bundle_manifest_digest,
        expected_repository=str(
            release_approval.get("repository", "")
            if isinstance(release_approval, dict)
            else ""
        ),
        expected_source_ref=str(
            release_approval.get("source_ref", "")
            if isinstance(release_approval, dict)
            else ""
        ),
        expected_source_commit=str(
            release_approval.get("source_commit", "")
            if isinstance(release_approval, dict)
            else ""
        ),
        expected_platform=str(
            release_approval.get("platform", "")
            if isinstance(release_approval, dict)
            else ""
        ),
        expected_mode=expected_mode,
        expected_release_authority_id=expected_release_authority_id,
        expected_release_authority_trust_store_digest=(
            expected_release_authority_trust_store_digest
        ),
        required_valid_through=str(unsigned.get("expires_at", "")),
    )
    if unsigned.get("release_approval_digest") != release_approval_digest:
        raise ValueError("runtime activation release approval is invalid")
    assert isinstance(release_approval, dict)
    if (
        trusted_activation_public_key_hex == trusted_release_public_key_hex
        or expected_activation_authority_id == expected_release_authority_id
        or datetime.fromisoformat(
            str(release_approval["approved_at"]).replace("Z", "+00:00")
        )
        > datetime.fromisoformat(
            str(unsigned["activated_at"]).replace("Z", "+00:00")
        )
    ):
        raise ValueError("runtime activation authority or chronology is invalid")
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
    approve_release = subparsers.add_parser("approve-release")
    approve_release.add_argument("--bundle", type=Path, required=True)
    approve_release.add_argument(
        "--trusted-bundle-public-key", type=Path, required=True
    )
    approve_release.add_argument(
        "--expected-mode",
        choices=("candidate-smoke", "deployable"),
        required=True,
    )
    approve_release.add_argument("--expected-repository", required=True)
    approve_release.add_argument("--expected-source-ref", required=True)
    approve_release.add_argument("--expected-source-commit", required=True)
    approve_release.add_argument("--expected-workflow-sha", required=True)
    approve_release.add_argument("--expected-bundle-signer-id", required=True)
    approve_release.add_argument("--approved-at")
    approve_release.add_argument("--expires-at", required=True)
    approve_release.add_argument("--release-authority-id", required=True)
    approve_release.add_argument(
        "--release-authority-trust-store-digest", required=True
    )
    approve_release.add_argument(
        "--release-signing-private-key", type=Path, required=True
    )
    approve_release.add_argument("--approval", type=Path, required=True)
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
    activate.add_argument("--deployment-instance-digest", required=True)
    activate.add_argument("--host-identity-digest", required=True)
    activate.add_argument("--activation-sequence-number", type=int, required=True)
    activate.add_argument(
        "--previous-activation-receipt-digest", required=True
    )
    activate.add_argument("--rollback-reason-digest")
    activate.add_argument("--activated-at")
    activate.add_argument("--expires-at", required=True)
    activate.add_argument("--release-approval", type=Path, required=True)
    activate.add_argument(
        "--trusted-release-public-key", type=Path, required=True
    )
    activate.add_argument("--expected-release-authority-id", required=True)
    activate.add_argument(
        "--expected-release-authority-trust-store-digest", required=True
    )
    activate.add_argument("--activation-authority-id", required=True)
    activate.add_argument(
        "--activation-authority-trust-store-digest", required=True
    )
    activate.add_argument(
        "--activation-signing-private-key", type=Path, required=True
    )
    activate.add_argument("--receipt", type=Path, required=True)
    verify_activation = subparsers.add_parser("verify-runtime-activation")
    verify_activation.add_argument("--receipt", type=Path, required=True)
    verify_activation.add_argument("--release-approval", type=Path, required=True)
    verify_activation.add_argument(
        "--trusted-release-public-key", type=Path, required=True
    )
    verify_activation.add_argument(
        "--trusted-activation-public-key", type=Path, required=True
    )
    verify_activation.add_argument(
        "--expected-mode",
        choices=("candidate-smoke", "deployable"),
        required=True,
    )
    verify_activation.add_argument(
        "--expected-deployment-bundle-digest", required=True
    )
    verify_activation.add_argument("--expected-runtime-tree-digest", required=True)
    verify_activation.add_argument(
        "--expected-interpreter-closure-digest", required=True
    )
    verify_activation.add_argument(
        "--expected-deployment-instance-digest", required=True
    )
    verify_activation.add_argument("--expected-host-identity-digest", required=True)
    verify_activation.add_argument(
        "--expected-activation-sequence-number", type=int, required=True
    )
    verify_activation.add_argument(
        "--expected-previous-activation-receipt-digest", required=True
    )
    verify_activation.add_argument("--expected-rollback-reason-digest")
    verify_activation.add_argument(
        "--expected-release-authority-id", required=True
    )
    verify_activation.add_argument(
        "--expected-release-authority-trust-store-digest", required=True
    )
    verify_activation.add_argument(
        "--expected-activation-authority-id", required=True
    )
    verify_activation.add_argument(
        "--expected-activation-authority-trust-store-digest", required=True
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
        elif arguments.command == "approve-release":
            bundle_public_key = _load_trusted_public_key(
                arguments.trusted_bundle_public_key.absolute(),
                deployable=arguments.expected_mode == "deployable",
            )
            approved_at = arguments.approved_at or (
                datetime.now(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
            approval = issue_bundle_release_approval(
                arguments.bundle.absolute(),
                trusted_bundle_public_key_hex=bundle_public_key,
                expected_mode=arguments.expected_mode,
                expected_repository=arguments.expected_repository,
                expected_source_ref=arguments.expected_source_ref,
                expected_source_commit=arguments.expected_source_commit,
                expected_workflow_sha=arguments.expected_workflow_sha,
                expected_bundle_signer_id=arguments.expected_bundle_signer_id,
                approved_at=approved_at,
                expires_at=arguments.expires_at,
                release_authority_id=arguments.release_authority_id,
                release_authority_trust_store_digest=(
                    arguments.release_authority_trust_store_digest
                ),
                release_signing_key=_load_private_key(
                    arguments.release_signing_private_key.absolute()
                ),
            )
            _canonical_json(arguments.approval.absolute(), approval)
            print(approval["approval_digest"])
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
            release_public_key = _load_trusted_public_key(
                arguments.trusted_release_public_key.absolute(),
                deployable=arguments.expected_mode == "deployable",
            )
            release_approval: Any = json.loads(
                arguments.release_approval.read_text(encoding="utf-8")
            )
            receipt = issue_runtime_activation_receipt(
                arguments.bundle.absolute(),
                release_approval=release_approval,
                trusted_bundle_public_key_hex=bundle_public_key,
                trusted_release_public_key_hex=release_public_key,
                expected_mode=arguments.expected_mode,
                expected_repository=arguments.expected_repository,
                expected_source_ref=arguments.expected_source_ref,
                expected_source_commit=arguments.expected_source_commit,
                expected_workflow_sha=arguments.expected_workflow_sha,
                expected_bundle_signer_id=(
                    arguments.expected_bundle_signer_id
                ),
                deployment_instance_digest=(
                    arguments.deployment_instance_digest
                ),
                host_identity_digest=arguments.host_identity_digest,
                activation_sequence_number=(
                    arguments.activation_sequence_number
                ),
                previous_activation_receipt_digest=(
                    arguments.previous_activation_receipt_digest
                ),
                rollback_reason_digest=arguments.rollback_reason_digest,
                activated_at=activated_at,
                expires_at=arguments.expires_at,
                expected_release_authority_id=(
                    arguments.expected_release_authority_id
                ),
                expected_release_authority_trust_store_digest=(
                    arguments.expected_release_authority_trust_store_digest
                ),
                activation_authority_id=arguments.activation_authority_id,
                activation_authority_trust_store_digest=(
                    arguments.activation_authority_trust_store_digest
                ),
                activation_signing_key=_load_private_key(
                    arguments.activation_signing_private_key.absolute()
                ),
            )
            _canonical_json(arguments.receipt.absolute(), receipt)
            print(receipt["receipt_digest"])
        else:
            release_public_key = _load_trusted_public_key(
                arguments.trusted_release_public_key.absolute(),
                deployable=arguments.expected_mode == "deployable",
            )
            activation_public_key = _load_trusted_public_key(
                arguments.trusted_activation_public_key.absolute(),
                deployable=arguments.expected_mode == "deployable",
            )
            receipt_value: Any = json.loads(
                arguments.receipt.read_text(encoding="utf-8")
            )
            release_approval_value: Any = json.loads(
                arguments.release_approval.read_text(encoding="utf-8")
            )
            print(
                verify_runtime_activation_receipt(
                    receipt_value,
                    release_approval=release_approval_value,
                    trusted_release_public_key_hex=release_public_key,
                    trusted_activation_public_key_hex=activation_public_key,
                    expected_deployment_bundle_digest=(
                        arguments.expected_deployment_bundle_digest
                    ),
                    expected_bundle_manifest_digest=str(
                        release_approval_value.get("bundle_manifest_digest", "")
                        if isinstance(release_approval_value, dict)
                        else ""
                    ),
                    expected_runtime_tree_digest=(
                        arguments.expected_runtime_tree_digest
                    ),
                    expected_interpreter_closure_digest=(
                        arguments.expected_interpreter_closure_digest
                    ),
                    expected_deployment_instance_digest=(
                        arguments.expected_deployment_instance_digest
                    ),
                    expected_host_identity_digest=(
                        arguments.expected_host_identity_digest
                    ),
                    expected_mode=arguments.expected_mode,
                    expected_activation_sequence_number=(
                        arguments.expected_activation_sequence_number
                    ),
                    expected_previous_activation_receipt_digest=(
                        arguments
                        .expected_previous_activation_receipt_digest
                    ),
                    expected_rollback_reason_digest=(
                        arguments.expected_rollback_reason_digest
                    ),
                    expected_release_authority_id=(
                        arguments.expected_release_authority_id
                    ),
                    expected_release_authority_trust_store_digest=(
                        arguments
                        .expected_release_authority_trust_store_digest
                    ),
                    expected_activation_authority_id=(
                        arguments.expected_activation_authority_id
                    ),
                    expected_activation_authority_trust_store_digest=(
                        arguments
                        .expected_activation_authority_trust_store_digest
                    ),
                )
            )
    except (OSError, TypeError, ValueError) as error:
        print(f"deployment bundle generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
