"""Guard and classify the bounded point-response no-solver preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded


WALL_SECONDS = 120
SAMPLED_RSS_BYTES = 1024**3


def run(directory: Path) -> dict[str, object]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("point preflight output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_response_preflight.json"
    command = [
        sys.executable, "-m", "examples.weather_scenarios.fv_point_response_preflight",
        "--output", str(output),
    ]
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "point_response_preflight.resource.json",
        log_path=directory / "point_response_preflight.log",
    )
    child = None
    read_error = None
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    valid_resource = (
        resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("command") == command
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
    )
    valid_child = (
        isinstance(child, dict)
        and child.get("status") == "preflight_only"
        and child.get("numerical_solver_runs") == 0
        and child.get("source_unchanged") is True
        and child.get("response_validation") == "not_performed"
        and type(child.get("pid")) is int
        and child["pid"] == resource.get("child_pid")
        and isinstance(child.get("warm_branch"), dict)
        and len(child["warm_branch"].get("choices", [])) == 54
        and len(child["warm_branch"].get("face_signs", [])) == 54
    )
    result: dict[str, object] = {
        "execution_status": "completed" if valid_resource and valid_child else "failed",
        "resource": resource,
        "child_read_error": read_error,
        "child_status": child.get("status") if isinstance(child, dict) else None,
        "scope": "one fixed point problem preflight only; no GN, PCG, adjoint or reanalysis",
    }
    (directory / "point_response_preflight.run.json").write_text(
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
