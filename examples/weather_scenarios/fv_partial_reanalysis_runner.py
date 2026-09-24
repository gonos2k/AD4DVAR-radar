"""Approval-gated resource wrapper for the planned partial FV reanalysis.

Do not invoke this script until its separately proposed 1200-second/1-GiB
numerical budget is approved. It preserves every child artifact on refusal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded


PINNED = ROOT / "graphify-out/fv-root-cause-20260919/fv_partial_reanalysis_preflight.json"


def _paths(directory: Path) -> dict[str, Path]:
    stem = directory / "fv_partial_reanalysis"
    return {
        "preflight": stem.with_suffix(".preflight.json"),
        "preflight_resource": stem.with_suffix(".preflight.resource.json"),
        "preflight_log": stem.with_suffix(".preflight.log"),
        "result": stem.with_suffix(".json"),
        "resource": stem.with_suffix(".resource.json"),
        "log": stem.with_suffix(".log"),
    }


def run(directory: Path) -> dict[str, Any]:
    paths = _paths(directory)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("preserve the existing partial reanalysis artifacts")
    directory.mkdir(parents=True, exist_ok=True)

    preflight_command = [
        sys.executable,
        str(Path(__file__).with_name("fv_partial_reanalysis_preflight.py")),
        "--output",
        str(paths["preflight"]),
    ]
    preflight_resource = run_guarded(
        preflight_command,
        wall_seconds=120,
        rss_bytes=1 * 1024**3,
        report_path=paths["preflight_resource"],
        log_path=paths["preflight_log"],
    )
    if preflight_resource["exit_code"] != 0 or preflight_resource["resource_termination"]:
        raise RuntimeError("guarded partial preflight did not complete")
    current = json.loads(paths["preflight"].read_text())
    pinned = json.loads(PINNED.read_text())
    for key in (
        "status", "problem_identity", "observation_counts", "missing_indices",
        "common_bias_std_dbz", "verification_constructor", "tensor_sha256",
        "warm_start_gradient_max", "warm_start_branch", "source_sha256",
        "plan_sha256", "warm_start_report_sha256",
    ):
        if current[key] != pinned[key]:
            raise ValueError(f"guarded preflight changed its {key}")

    command = [
        sys.executable,
        str(Path(__file__).with_name("fv_partial_reanalysis_probe.py")),
        "--output",
        str(paths["result"]),
        "--execute",
    ]
    resource = run_guarded(
        command,
        wall_seconds=1200,
        rss_bytes=1 * 1024**3,
        report_path=paths["resource"],
        log_path=paths["log"],
    )
    if resource["exit_code"] != 0 or resource["resource_termination"]:
        raise RuntimeError("guarded partial reanalysis did not complete; inspect artifacts")
    return resource


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("numerical execution requires explicit --execute after budget approval")
    print(json.dumps(run(args.directory), indent=2))
