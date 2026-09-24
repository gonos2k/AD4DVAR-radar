"""Guard one source-bound concurrent FV response child process."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from examples.weather_scenarios.fv86_resource_runner import run_guarded


ROOT = Path(__file__).resolve().parents[2]
WALL_SECONDS = 300
RSS_BYTES = 1536 * 1024**2
SOURCE_PATHS = (
    "src/advar/matrix_free.py", "src/advar/transport.py",
    "src/advar/local_response.py", "src/advar/fv_research_problem.py",
    "src/advar/variational.py", "src/advar/physics.py", "src/advar/nowcast.py",
    "examples/weather_scenarios/fv_minmod_inverse_probe.py",
    "examples/weather_scenarios/fv_sensitivity_probe.py",
    "examples/weather_scenarios/fv_scaled_research_case.py",
    "examples/weather_scenarios/fv_concurrent_response_probe.py",
    "examples/weather_scenarios/fv_concurrent_response_runner.py",
    "examples/weather_scenarios/fv86_resource_runner.py",
)


def _hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


def run(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    source_before = _hashes()
    report_path = directory / "fv_concurrent_response.json"
    resource_path = directory / "fv_concurrent_response.resource.json"
    log_path = directory / "fv_concurrent_response.log"
    command = [
        sys.executable, "-m", "examples.weather_scenarios.fv_concurrent_response_probe",
        "--output", str(report_path),
    ]
    resource = run_guarded(
        command,
        wall_seconds=WALL_SECONDS,
        rss_bytes=RSS_BYTES,
        report_path=resource_path,
        log_path=log_path,
    )
    source_after = _hashes()
    child_error = None
    try:
        child = json.loads(report_path.read_text()) if report_path.exists() else None
    except (OSError, json.JSONDecodeError) as error:
        child = None
        child_error = f"{type(error).__name__}: {error}"
    completed = (
        resource["exit_code"] == 0
        and resource["resource_termination"] is None
        and resource["monitor_error"] is None
        and isinstance(resource.get("elapsed_seconds"), (int, float))
        and math.isfinite(resource["elapsed_seconds"])
        and resource["elapsed_seconds"] <= WALL_SECONDS
        and source_after == source_before
        and isinstance(child, dict)
        and child.get("status") == "completed"
        and child.get("phase") == "finished"
        and child.get("source_unchanged") is True
        and child.get("inputs_unchanged") is True
        and child.get("source_sha256") == source_before
        and set(child.get("workers", {})) == {"fv4x5", "fv8x10"}
    )
    result: dict[str, object] = {
        "execution_status": "completed" if completed else "failed",
        "source_before": source_before,
        "source_after": source_after,
        "resource": resource,
        "child_status": None if child is None else child.get("status"),
        "child_read_error": child_error,
        "child_report_sha256": (
            hashlib.sha256(report_path.read_bytes()).hexdigest()
            if report_path.exists() else None
        ),
    }
    (directory / "fv_concurrent_response.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
