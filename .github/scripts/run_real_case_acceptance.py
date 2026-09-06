#!/usr/bin/env python3
"""Verify a report-only real-case acceptance manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from advar.acceptance import (
    RealCaseAcceptanceManifest,
    verify_real_case_acceptance,
)


def _same_file(left: Path, right: Path) -> bool:
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except OSError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def _reject_report_collision(
    *,
    report: Path,
    manifest: Path,
    artifact_root: Path,
    input_paths: tuple[Path, ...] = (),
) -> None:
    """Keep report publication outside every input identity and tree."""

    report = report.absolute()
    manifest = manifest.absolute()
    artifact_root = artifact_root.absolute()
    if report.is_symlink():
        raise ValueError("acceptance report path must not be a symlink")
    try:
        report_resolved = report.resolve(strict=False)
        manifest_resolved = manifest.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("acceptance report path cannot be resolved safely") from error
    if report_resolved == manifest_resolved:
        raise ValueError("acceptance report path collides with the manifest")
    try:
        root_resolved = artifact_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("acceptance report path cannot be resolved safely") from error
    try:
        report_resolved.relative_to(root_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("acceptance report path must be outside artifact root")
    for input_path in (manifest, *input_paths):
        if _same_file(report, input_path):
            raise ValueError("acceptance report path collides with an input artifact")


def _write_report(path: Path, report: dict[str, object]) -> None:
    """Publish complete report bytes through a temporary sibling."""

    path = path.absolute()
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        if temporary_name is None:
            raise RuntimeError("acceptance report temporary file was not created")
        os.replace(temporary_name, path)
        temporary_name = None
        try:
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(path.parent, directory_flags)
        except OSError:
            return
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    _reject_report_collision(
        report=arguments.report,
        manifest=arguments.manifest,
        artifact_root=arguments.artifact_root,
    )
    manifest = RealCaseAcceptanceManifest.from_json(
        arguments.manifest.read_text(encoding="utf-8")
    )
    input_paths = (
        arguments.artifact_root
        / manifest.sample_size_preflight_relative_path,
        *(
            arguments.artifact_root / reference.relative_path
            for case in manifest.cases
            for reference in case.artifacts
        ),
    )
    _reject_report_collision(
        report=arguments.report,
        manifest=arguments.manifest,
        artifact_root=arguments.artifact_root,
        input_paths=input_paths,
    )
    report = verify_real_case_acceptance(
        manifest,
        artifact_root=arguments.artifact_root,
    )
    _write_report(arguments.report, report)
    print(report["report_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
