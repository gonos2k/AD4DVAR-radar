from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import site
import stat
import subprocess
import sys
import sysconfig
from typing import Literal


RuntimeMode = Literal["candidate-smoke", "deployable"]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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


def _require_runtime_permissions(
    path: Path,
    *,
    deployable: bool,
) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("deployment runtime contains a symlink")
    if deployable and (metadata.st_uid != 0 or metadata.st_mode & 0o022):
        raise ValueError("deployable runtime must be root-owned and non-writable")
    return metadata


def _linked_native_libraries(
    paths: tuple[Path, ...],
    *,
    deployable: bool,
) -> list[dict[str, object]]:
    if platform.system() != "Linux":
        return []
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
            raise ValueError(
                "deployment native library closure is unreadable"
            ) from error
        for line in completed.stdout.splitlines():
            match = re.search(r"(?:=>\s+)?(/[^\s]+)", line)
            if match is not None:
                libraries.add(Path(match.group(1)).resolve(strict=True))
    result: list[dict[str, object]] = []
    for path in sorted(libraries, key=lambda item: item.as_posix()):
        size_bytes, sha256 = _runtime_file_snapshot(
            path,
            deployable=deployable,
        )
        result.append(
            {
                "name": path.name,
                "path": path.as_posix(),
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
    return result


def active_import_path_snapshot(
    *,
    import_roots: tuple[Path, ...],
    stdlib_root: Path,
    deployable: bool,
) -> list[dict[str, object]]:
    """Seal ordered import roots and reject paths outside the runtime closure."""

    entries: list[dict[str, object]] = []
    seen: set[Path] = set()
    zip_name = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    for raw_path in sys.path:
        if not raw_path:
            raise ValueError("deployment runtime contains an ambient import root")
        path = Path(raw_path).absolute()
        if path in seen:
            raise ValueError("deployment runtime import path is duplicated")
        seen.add(path)
        allowed_directory = (
            path in import_roots
            or path == stdlib_root
            or stdlib_root in path.parents
        )
        allowed_zip = path.parent == stdlib_root.parent and path.name == zip_name
        if not path.exists():
            if not allowed_zip:
                raise ValueError("deployment runtime contains an unexpected import root")
            entries.append(
                {"role": "stdlib-zip", "kind": "absent-stdlib-zip"}
            )
            continue
        metadata = _require_runtime_permissions(path, deployable=deployable)
        if allowed_directory and stat.S_ISDIR(metadata.st_mode):
            if path in import_roots:
                role = f"site-{import_roots.index(path)}"
            elif path == stdlib_root:
                role = "stdlib"
            else:
                role = f"stdlib/{path.relative_to(stdlib_root).as_posix()}"
            entries.append({"role": role, "kind": "directory"})
        elif allowed_zip and stat.S_ISREG(metadata.st_mode):
            size_bytes, sha256 = _runtime_file_snapshot(
                path,
                deployable=deployable,
            )
            entries.append(
                {
                    "role": "stdlib-zip",
                    "kind": "stdlib-zip",
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            )
        else:
            raise ValueError("deployment runtime contains an unexpected import root")
    return entries


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
        stdlib_files.append(
            {
                "path": path.relative_to(stdlib_root).as_posix(),
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
        ),
    }
    return unsigned | {"interpreter_closure_digest": _json_digest(unsigned)}


def _installed_distributions(
    roots: tuple[Path, ...],
) -> list[dict[str, str]]:
    retained: dict[str, str] = {}
    for root in roots:
        for distribution in importlib.metadata.distributions(path=[str(root)]):
            raw_name = distribution.metadata.get("Name")
            if not raw_name:
                continue
            name = _normalized_distribution_name(raw_name)
            previous = retained.setdefault(name, distribution.version)
            if previous != distribution.version:
                raise ValueError("deployment runtime distribution is ambiguous")
    if not retained:
        raise ValueError("deployment runtime contains no distributions")
    return [
        {"name": name, "version": version}
        for name, version in sorted(retained.items())
    ]


def _runtime_tree_snapshot(
    *,
    distributions: list[dict[str, str]],
    runtime_mode: RuntimeMode,
) -> dict[str, object]:
    expected_distributions = sorted(
        (
            {
                "name": _normalized_distribution_name(item["name"]),
                "version": item["version"],
            }
            for item in distributions
        ),
        key=lambda item: item["name"],
    )
    if (
        runtime_mode not in {"candidate-smoke", "deployable"}
        or not expected_distributions
        or len({item["name"] for item in expected_distributions})
        != len(expected_distributions)
    ):
        raise ValueError("deployment runtime identity is invalid")
    roots = _runtime_import_roots()
    if not roots:
        raise ValueError("deployment runtime has no import roots")
    deployable = runtime_mode == "deployable"
    claimed_paths: dict[tuple[Path, str], str] = {}
    for expected in expected_distributions:
        name = expected["name"]
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError("deployment runtime is incomplete") from error
        if (
            distribution.version != expected["version"]
            or distribution.files is None
        ):
            raise ValueError("deployment runtime disagrees with its lock")
        for package_path in distribution.files:
            located = Path(str(distribution.locate_file(package_path))).absolute()
            root = next(
                (candidate for candidate in roots if located.is_relative_to(candidate)),
                None,
            )
            if root is None:
                continue
            relative = located.relative_to(root)
            if ".." in relative.parts:
                continue
            key = (root, relative.as_posix())
            previous = claimed_paths.setdefault(key, name)
            if previous != name:
                raise ValueError("deployment runtime file identity is duplicated")
    actual_distributions = {
        item["name"] for item in _installed_distributions(roots)
    }
    expected_names = {item["name"] for item in expected_distributions}
    if actual_distributions != expected_names:
        raise ValueError("deployment runtime contains an extra distribution")
    files: list[dict[str, object]] = []
    seen_claims: set[tuple[Path, str]] = set()
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
        "distributions": expected_distributions,
        "files": files,
        "omitted_forbidden_files": omitted_forbidden_files,
        "interpreter_closure": interpreter_closure,
    }
    return unsigned | {"runtime_tree_digest": _json_digest(unsigned)}


def snapshot_current_runtime(
    *,
    runtime_mode: RuntimeMode,
) -> dict[str, object]:
    """Hash the complete import/interpreter closure of this Python process."""

    roots = _runtime_import_roots()
    return _runtime_tree_snapshot(
        distributions=_installed_distributions(roots),
        runtime_mode=runtime_mode,
    )


def validate_current_runtime_closure(
    *,
    expected_runtime_tree_digest: str,
    expected_interpreter_closure_digest: str,
    runtime_mode: RuntimeMode,
) -> None:
    """Fail closed unless the executing process matches a sealed runtime."""

    snapshot = snapshot_current_runtime(runtime_mode=runtime_mode)
    interpreter = snapshot.get("interpreter_closure")
    if (
        snapshot.get("runtime_tree_digest") != expected_runtime_tree_digest
        or not isinstance(interpreter, dict)
        or interpreter.get("bytecode_write_disabled") is not True
        or interpreter.get("interpreter_closure_digest")
        != expected_interpreter_closure_digest
    ):
        raise ValueError("current process runtime closure disagrees with activation")
