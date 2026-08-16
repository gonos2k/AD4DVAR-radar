from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _relative_path(raw_path: object) -> str:
    path = Path(str(raw_path))
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _print_diagnostic(diagnostic: dict[str, Any]) -> None:
    location = diagnostic.get("range", {}).get("start", {})
    line = int(location.get("line", 0)) + 1
    column = int(location.get("character", 0)) + 1
    severity = str(diagnostic.get("severity", "unknown"))
    message = str(diagnostic.get("message", "missing diagnostic message"))
    rule = diagnostic.get("rule")
    suffix = f" ({rule})" if rule else ""
    print(
        f"{_relative_path(diagnostic.get('file', '<unknown>'))}:"
        f"{line}:{column}: {severity}: {message}{suffix}"
    )


def main() -> int:
    targets = sys.argv[1:] or ["src/advar"]
    isolated = ["-I"] if os.environ.get("CI") == "true" else []
    completed = subprocess.run(
        [
            sys.executable,
            *isolated,
            "-m",
            "basedpyright",
            "--level",
            "error",
            "--outputjson",
            "--pythonpath",
            sys.executable,
            *targets,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")

    try:
        report = json.loads(completed.stdout.lstrip("\ufeff"))
        diagnostics = report["generalDiagnostics"]
        summary = report["summary"]
        error_count = int(summary["errorCount"])
        warning_count = int(summary["warningCount"])
        information_count = int(summary["informationCount"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print("basedpyright did not produce a valid JSON report", file=sys.stderr)
        print(completed.stdout, file=sys.stderr, end="")
        print(str(exc), file=sys.stderr)
        return completed.returncode or 1

    for diagnostic in diagnostics:
        if isinstance(diagnostic, dict):
            _print_diagnostic(diagnostic)

    print(
        "basedpyright: "
        f"{error_count} errors, {warning_count} warnings, "
        f"{information_count} notes"
    )
    if error_count:
        return 1
    if completed.returncode == 0:
        return 0
    if completed.returncode == 1 and warning_count:
        return 0
    print(
        f"basedpyright exited unexpectedly with code {completed.returncode}",
        file=sys.stderr,
    )
    return completed.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
