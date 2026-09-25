"""Guard and independently classify one gradient-merit point-root run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios.fv_point_basin_runner import _valid_curvature
from examples.weather_scenarios import fv_point_merit_root_probe as probe
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios.fv_point_sector_root_runner import _valid_branch


WALL_SECONDS = 600
SAMPLED_RSS_BYTES = 1024**3
ROOT_STATUSES = {"merit_root_margin_qualified", "merit_root_low_margin"}
REFUSALS = {"seed_curvature_refused", "merit_refused"}


def _valid_root(child: dict[str, Any]) -> bool:
    control = child.get("control")
    final = child.get("final_branch")
    calls = child.get("branch_calls")
    if not (
        child.get("phase") == "finished"
        and isinstance(control, list) and len(control) == 26
        and all(type(v) in (int, float) and math.isfinite(v) for v in control)
        and type(child.get("gradient_max")) in (int, float)
        and math.isfinite(child["gradient_max"])
        and 0 <= child["gradient_max"] < 1e-10
        and all(type(child.get(k)) in (int, float) and math.isfinite(child[k])
                for k in ("objective", "score", "gradient_norm"))
        and type(child.get("refinement_iterations")) is int
        and 0 <= child["refinement_iterations"] <= 8
        and _valid_branch(final)
        and _valid_curvature(child.get("final_curvature"))
        and child.get("final_control_sha256") == preflight._tensor_sha(
            torch.tensor(control, dtype=torch.float64))
        and isinstance(calls, list) and bool(calls)
        and isinstance(calls[-1], dict)
        and calls[-1].get("status") == "core_branch_admitted"
        and calls[-1].get("control_sha256") == child["final_control_sha256"]
        and all(calls[-1].get(k) == final[k] for k in (
            "signature_sha256", "euler_stages",
            "minimum_scaled_slope_margin", "minimum_scaled_face_flux_margin"
        ))
    ):
        return False
    high = final["minimum_scaled_slope_margin"] > 1e-4 and final["minimum_scaled_face_flux_margin"] > 1e-4
    return child.get("response_margin_qualified") is high and child.get("numerical_status") == (
        "merit_root_margin_qualified" if high else "merit_root_low_margin")


def run(directory: Path) -> dict[str, object]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("merit root output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_merit_root.json"
    command = [sys.executable, "-m", "examples.weather_scenarios.fv_point_merit_root_probe",
               "--output", str(output)]
    source_before = probe._sources()
    plan_before, prior_before = probe._sha(probe.PLAN), probe._sha(probe.PRIOR)
    resource = run_guarded(command, wall_seconds=WALL_SECONDS,
                           rss_bytes=SAMPLED_RSS_BYTES,
                           report_path=directory / "point_merit_root.resource.json",
                           log_path=directory / "point_merit_root.log")
    child = None
    read_error = None
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    current = None
    provenance_error = None
    try:
        problem, warm, parameters, direction = preflight.fixed_problem()
        current = preflight._input_identity(problem, warm, parameters, direction)
    except (OSError, ValueError, RuntimeError) as error:
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
        and child.get("numerical_status") in ROOT_STATUSES | REFUSALS
        and child.get("response_validation") == "not_performed"
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("plan_unchanged") is True
        and child.get("prior_unchanged") is True
        and child.get("seed_control_unchanged") is True
        and child.get("source_before") == child.get("source_after") == source_before == probe._sources()
        and child.get("input_before") == child.get("input_after") == current
        and child.get("plan_sha256") == plan_before == probe.PLAN_SHA256 == probe._sha(probe.PLAN)
        and child.get("prior_report_sha256") == prior_before == probe.PRIOR_SHA256 == probe._sha(probe.PRIOR)
        and child.get("seed_control_sha256") == probe.SEED_SHA256
        and isinstance(child.get("seed_control"), list)
        and len(child["seed_control"]) == 26
        and all(type(v) in (int, float) and math.isfinite(v) for v in child["seed_control"])
        and preflight._tensor_sha(torch.tensor(child["seed_control"], dtype=torch.float64)) == probe.SEED_SHA256
        and type(child.get("pid")) is int and child["pid"] == resource.get("child_pid")
        and _valid_branch(child.get("seed_branch"))
        and child["seed_branch"]["signature_sha256"] == probe.SEED_SIGNATURE_SHA256
        and (_valid_curvature(child.get("seed_curvature"))
             or (child.get("numerical_status") == "seed_curvature_refused"
                 and child.get("phase") == "seed_curvature"))
        and ((child["numerical_status"] in ROOT_STATUSES
              and resource["exit_code"] == 0 and _valid_root(child))
             or (child["numerical_status"] in REFUSALS
                 and resource["exit_code"] == 2
                 and isinstance(child.get("refusal"), str)
                 and (child.get("phase") == "seed_curvature"
                      if child["numerical_status"] == "seed_curvature_refused"
                      else child.get("phase") in ("merit_refinement", "final_curvature"))))
    )
    result: dict[str, object] = {
        "execution_status": "completed" if valid_resource and valid_child else "failed",
        "numerical_status": child.get("numerical_status") if isinstance(child, dict) else "not_reached",
        "response_validation": "not_performed",
        "resource": resource,
        "child_read_error": read_error,
        "provenance_error": provenance_error,
        "scope": "one gradient-merit sector-root diagnostic; no adjoint/reanalysis",
    }
    (directory / "point_merit_root.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
