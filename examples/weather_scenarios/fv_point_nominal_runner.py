"""Guard one point-objective Newton feasibility attempt and retain refusals."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded


WALL_SECONDS = 240
SAMPLED_RSS_BYTES = 1024**3
PREFLIGHT = ROOT / "graphify-out/fv-root-cause-20260919/point_response_preflight_attempt1/point_response_preflight.json"


def run(directory: Path) -> dict[str, object]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("nominal point output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_nominal.json"
    command = [sys.executable, "-m", "examples.weather_scenarios.fv_point_nominal_probe",
               "--output", str(output), "--preflight", str(PREFLIGHT)]
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "point_nominal.resource.json",
        log_path=directory / "point_nominal.log",
    )
    child = None
    read_error = None
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    valid_resource = (
        resource.get("exit_code") in (0, 2)
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("command") == command
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and type(resource.get("elapsed_seconds")) in (int, float)
        and math.isfinite(resource["elapsed_seconds"])
        and 0 <= resource["elapsed_seconds"] <= WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
    )
    valid_child = (
        isinstance(child, dict)
        and child.get("numerical_status") in ("nominal_stationarity", "refused")
        and child.get("response_validation") == "not_performed"
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("plan_unchanged") is True
        and child.get("preflight_plan_unchanged") is True
        and child.get("preflight_unchanged") is True
        and type(child.get("pid")) is int
        and child["pid"] == resource.get("child_pid")
        and ((child["numerical_status"] == "nominal_stationarity" and resource["exit_code"] == 0)
             or (child["numerical_status"] == "refused" and resource["exit_code"] == 2))
    )
    result: dict[str, object] = {
        "execution_status": "completed" if valid_resource and valid_child else "failed",
        "numerical_status": child.get("numerical_status") if isinstance(child, dict) else "not_reached",
        "response_validation": "not_performed",
        "child_read_error": read_error,
        "resource": resource,
        "scope": "one warm-start point-objective Newton attempt only; no adjoint/reanalysis",
    }
    (directory / "point_nominal.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
