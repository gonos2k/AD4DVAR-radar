from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 CI lock supplies tomli.
    import tomli as tomllib  # type: ignore[no-redef]

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
LOCKS = {
    "runtime-py310-linux.lock": ROOT / "requirements/runtime-py310-linux.lock",
    "runtime-py312-linux.lock": ROOT / "requirements/runtime-py312-linux.lock",
    "ci-py310-linux.lock": ROOT / "requirements/ci-py310-linux.lock",
    "ci-py312-linux.lock": ROOT / "requirements/ci-py312-linux.lock",
}
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)(?:\s+\\)?$")
HASH = re.compile(r"^\s+--hash=sha256:([0-9a-f]{64})(?:\s+\\)?$")


@dataclass(frozen=True)
class LockedPackage:
    version: str
    hashes: frozenset[str]


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_lock(path: Path) -> dict[str, LockedPackage]:
    pins: dict[str, LockedPackage] = {}
    current: str | None = None
    current_version: str | None = None
    current_hashes: set[str] = set()

    def finish() -> None:
        if current is None:
            return
        if current_version is None or not current_hashes:
            raise ValueError(f"{path}: {current} has no SHA-256 hash")
        pins[current] = LockedPackage(current_version, frozenset(current_hashes))

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#") or line.startswith("    #"):
            continue
        match = PIN.fullmatch(line)
        if match is not None:
            finish()
            name = _normalized(match.group(1))
            if name in pins:
                raise ValueError(f"{path}:{line_number}: duplicate pin for {name}")
            current = name
            current_version = match.group(2)
            current_hashes = set()
            continue
        hash_match = HASH.fullmatch(line)
        if current is not None and hash_match is not None:
            digest = hash_match.group(1)
            if digest in current_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash for {current}")
            current_hashes.add(digest)
            continue
        raise ValueError(f"{path}:{line_number}: non-canonical lock line: {line!r}")
    finish()
    if not pins:
        raise ValueError(f"{path}: empty dependency closure")
    return pins


def _requirements_from_pyproject() -> tuple[list[Requirement], list[Requirement]]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    runtime = [Requirement(item) for item in project["project"]["dependencies"]]
    build = [Requirement(item) for item in project["build-system"]["requires"]]
    return runtime, build


def _requirements_from_input(name: str) -> list[Requirement]:
    result: list[Requirement] = []
    for line in (ROOT / "requirements" / name).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            result.append(Requirement(stripped))
    return result


def _require_satisfied(
    requirement: Requirement,
    pins: dict[str, LockedPackage],
    *,
    lock_name: str,
) -> None:
    name = _normalized(requirement.name)
    if name not in pins:
        raise ValueError(f"{lock_name}: missing direct requirement {requirement}")
    if Version(pins[name].version) not in requirement.specifier:
        raise ValueError(
            f"{lock_name}: {name}=={pins[name].version} does not satisfy "
            f"{requirement.specifier}"
        )


def _require_cpu_only(pins: dict[str, LockedPackage], *, lock_name: str) -> None:
    forbidden = sorted(
        name for name in pins if name == "triton" or name.startswith("nvidia-")
    )
    if forbidden:
        raise ValueError(f"{lock_name}: GPU dependency pins are forbidden: {forbidden}")
    torch = pins.get("torch")
    if torch is None:
        raise ValueError(f"{lock_name}: missing torch CPU runtime")
    torch_version = Version(torch.version)
    if str(torch_version) != "2.13.0" or torch_version.local is not None:
        raise ValueError(
            f"{lock_name}: torch must use the auditable public 2.13.0 pin"
        )


def main() -> int:
    try:
        closures = {name: _read_lock(path) for name, path in LOCKS.items()}
        runtime_requirements, build_requirements = _requirements_from_pyproject()
        deployment_requirements = _requirements_from_input("runtime.in")
        ci_requirements = _requirements_from_input("ci.in")
        for lock_name, closure in closures.items():
            _require_cpu_only(closure, lock_name=lock_name)
        for python_tag in ("310", "312"):
            runtime_name = f"runtime-py{python_tag}-linux.lock"
            ci_name = f"ci-py{python_tag}-linux.lock"
            runtime = closures[runtime_name]
            ci = closures[ci_name]
            for requirement in (*runtime_requirements, *deployment_requirements):
                _require_satisfied(requirement, runtime, lock_name=runtime_name)
                _require_satisfied(requirement, ci, lock_name=ci_name)
            for requirement in (*build_requirements, *ci_requirements):
                _require_satisfied(requirement, ci, lock_name=ci_name)
            for name, locked_package in runtime.items():
                if ci.get(name) != locked_package:
                    raise ValueError(
                        f"{ci_name}: runtime pin or hash set for {name} is absent or differs"
                    )
        for requirement in deployment_requirements:
            name = _normalized(requirement.name)
            versions = {
                closures[f"runtime-py{tag}-linux.lock"][name].version
                for tag in ("310", "312")
            }
            if len(versions) != 1:
                raise ValueError(
                    f"deployment direct pin differs between Python versions: {name}"
                )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"dependency lock validation failed: {exc}", file=sys.stderr)
        return 1
    print("dependency locks: 4 hashed Linux CPU closures are synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
