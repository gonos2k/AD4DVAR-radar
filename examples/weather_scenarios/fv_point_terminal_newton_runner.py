"""Guard the single locked-terminal exact Newton feasibility diagnostic."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios.fv_point_basin_runner import _valid_branch, _valid_curvature
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios import fv_point_terminal_newton_probe as probe


WALL_SECONDS = 600
SAMPLED_RSS_BYTES = 1024**3
REFUSALS = {"seed_curvature_refused", "newton_refused"}
EXPECTED_SEED_SHA256 = "8f5110f2755217dece8ca9f6bbab23ceadb184b4f1a32b8e4654ddfde5f4b258"
EXPECTED_BRANCH_SHA256 = "0783f66a062d5144894513c65a4e221befca4d23a22d8fba0adc43ead5011327"


def _current_provenance() -> dict[str, Any]:
    problem, warm, parameters, direction = preflight.fixed_problem()
    pinned_preflight = json.loads(probe.PREFLIGHT.read_text())
    return {
        "source": probe._sources(),
        "input": preflight._input_identity(problem, warm, parameters, direction),
        "plan": probe._sha(probe.PLAN),
        "preflight_plan": probe._sha(preflight.PLAN),
        "expected_preflight_plan": pinned_preflight["plan_sha256"],
        "preflight": probe._sha(probe.PREFLIGHT),
        "attempt2": probe._sha(probe.ATTEMPT2),
    }


def _valid_success(child: dict[str, Any]) -> bool:
    control = child.get("control")
    seed = child.get("seed_branch")
    final = child.get("final_branch")
    return (
        child.get("phase") == "finished"
        and isinstance(control, list) and len(control) == 26
        and all(type(value) in (int, float) and math.isfinite(value) for value in control)
        and type(child.get("gradient_max")) in (int, float)
        and math.isfinite(child["gradient_max"])
        and 0 <= child["gradient_max"] < 1e-10
        and all(type(child.get(key)) in (int, float) and math.isfinite(child[key])
                for key in ("objective", "score", "gradient_norm"))
        and type(child.get("refinement_iterations")) is int
        and 0 <= child["refinement_iterations"] <= 8
        and _valid_branch(seed) and _valid_branch(final)
        and child.get("seed_control_sha256") == EXPECTED_SEED_SHA256
        and seed["signature_sha256"] == EXPECTED_BRANCH_SHA256
        and seed["signature_sha256"] == final["signature_sha256"]
        and _valid_curvature(child.get("seed_curvature"))
        and _valid_curvature(child.get("final_curvature"))
    )


def run(directory: Path) -> dict[str, object]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("terminal Newton output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_terminal_newton.json"
    command = [sys.executable, "-m", "examples.weather_scenarios.fv_point_terminal_newton_probe",
               "--output", str(output)]
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "point_terminal_newton.resource.json",
        log_path=directory / "point_terminal_newton.log",
    )
    child = None
    read_error = None
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    current = None
    provenance_error = None
    if isinstance(child, dict):
        try:
            current = _current_provenance()
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            provenance_error = f"{type(error).__name__}: {error}"
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
        and child.get("numerical_status") in (REFUSALS | {"nominal_stationarity"})
        and child.get("response_validation") == "not_performed"
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("seed_control_unchanged") is True
        and child.get("plan_unchanged") is True
        and child.get("preflight_plan_unchanged") is True
        and child.get("preflight_unchanged") is True
        and child.get("attempt2_unchanged") is True
        and isinstance(current, dict)
        and current["plan"] == probe.PLAN_SHA256
        and current["preflight"] == probe.PREFLIGHT_SHA256
        and current["attempt2"] == probe.ATTEMPT2_SHA256
        and current["preflight_plan"] == current["expected_preflight_plan"]
        and child.get("plan_sha256") == current["plan"]
        and child.get("preflight_plan_sha256") == current["preflight_plan"]
        and child.get("preflight_sha256") == current["preflight"]
        and child.get("attempt2_sha256") == current["attempt2"]
        and child.get("source_before") == child.get("source_after") == current["source"]
        and child.get("input_before") == child.get("input_after") == current["input"]
        and child.get("seed_control_sha256") == EXPECTED_SEED_SHA256
        and type(child.get("pid")) is int
        and child["pid"] == resource.get("child_pid")
        and ((child["numerical_status"] == "nominal_stationarity"
              and resource["exit_code"] == 0 and _valid_success(child))
             or (child["numerical_status"] in REFUSALS
                 and resource["exit_code"] == 2
                 and isinstance(child.get("refusal"), str)
                 and child.get("phase") == (
                     "seed_curvature" if child["numerical_status"] == "seed_curvature_refused"
                     else "exact_refinement"
                 )))
    )
    result: dict[str, object] = {
        "execution_status": "completed" if valid_resource and valid_child else "failed",
        "numerical_status": child.get("numerical_status") if isinstance(child, dict) else "not_reached",
        "response_validation": "not_performed",
        "child_read_error": read_error,
        "provenance_error": provenance_error,
        "resource": resource,
        "scope": "one locked point terminal curvature and exact Newton diagnostic; no adjoint/reanalysis",
    }
    (directory / "point_terminal_newton.run.json").write_text(
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
