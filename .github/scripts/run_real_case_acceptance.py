#!/usr/bin/env python3
"""Verify a report-only real-case acceptance manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from advar.acceptance import (
    RealCaseAcceptanceManifest,
    verify_real_case_acceptance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = RealCaseAcceptanceManifest.from_json(
        arguments.manifest.read_text(encoding="utf-8")
    )
    report = verify_real_case_acceptance(
        manifest,
        artifact_root=arguments.artifact_root,
    )
    arguments.report.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(report["report_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
