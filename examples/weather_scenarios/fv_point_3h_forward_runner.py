"""Resource guard and independent terminal-field check for point FV 3 hours."""
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

from advar import transport, variational as v
from advar.fv_point_sampler import point_dbz_bilinear
from advar.physics import echo_to_dbz
from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios import fv_point_3h_forward_probe as probe
from examples.weather_scenarios.fv_point_3h_forward_case import make_case


WALL_SECONDS = 180
SAMPLED_RSS_BYTES = 1024**3


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_layout(layout: dict[str, Any]) -> dict[str, Any]:
    """Compare the child's JSON shape with the same serialized local layout."""
    return json.loads(json.dumps(layout))


def _valid_numerics(child: dict[str, Any], problem: Any,
                    control: torch.Tensor, parameters: torch.Tensor,
                    truth: torch.Tensor) -> bool:
    layout = problem.layout
    analysis = child.get("analysis")
    terminal = child.get("terminal")
    branch = child.get("strict_branch_diagnostic")
    if not all(isinstance(value, dict) for value in (analysis, terminal, branch)):
        return False
    assert isinstance(analysis, dict) and isinstance(terminal, dict) and isinstance(branch, dict)
    if not (
        child.get("execution_phase") == "finished"
        and child.get("forward_validation") == "passed"
        and child.get("forecast_available") is True
        and child.get("stationarity_passed") == "not_tested"
        and child.get("response_computed") is False
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
        and child.get("layout") == _json_layout(layout)
        and layout["controls"] == 26 and layout["parameters"] == 13
        and layout["observation_times_seconds"] == (0.0, 600.0, 1200.0)
        and layout["forecast_time_seconds"] == 12000.0
        and layout["euler_stages"] == 3600
        and child.get("forecast_steps") == 18
        and child.get("substeps_per_interval") == 90
        and child.get("analysis_boundary_stage_pairs") == 180
        and child.get("future_boundary_stage_pairs") == 1620
        and analysis.get("observed_minmod_stages") == 360
        and terminal.get("observed_minmod_stages") == 3600
        and terminal.get("forecast_shape") == [4, 5]
        and terminal.get("terminal_truth_sha256") == probe._tensor_sha(truth)
        and _finite(analysis.get("point_observation_max_abs_difference_dbz"))
        and analysis["point_observation_max_abs_difference_dbz"] <= 1e-9
        and _finite(terminal.get("max_abs_difference_from_same_operator_truth_dbz"))
        and terminal["max_abs_difference_from_same_operator_truth_dbz"] <= 1e-9
        and all(_finite(terminal.get(key)) for key in ("objective", "score", "manual_terminal_mse"))
        and branch.get("status") in ("passed_pointwise", "refused")
        and type(branch.get("observed_minmod_stages")) is int
        and 0 <= branch["observed_minmod_stages"] <= 3600
        and (branch.get("euler_stages") == 3600
             and branch["observed_minmod_stages"] == 3600
             if branch["status"] == "passed_pointwise"
             else isinstance(branch.get("reason"), str))
    ):
        return False
    reported = torch.as_tensor(terminal.get("forecast"), dtype=torch.float64)
    if reported.shape != (4, 5) or not bool(torch.isfinite(reported).all()):
        return False
    if terminal.get("forecast_sha256") != probe._tensor_sha(reported):
        return False
    analysis_stages = 0
    def count_analysis(_echo: torch.Tensor, _qx: torch.Tensor, _qy: torch.Tensor) -> None:
        nonlocal analysis_stages
        analysis_stages += 1
    with torch.no_grad(), transport.observe_minmod_stages(count_analysis):
        trajectory = v.analysis_trajectory(control, problem.contract(parameters))
    sampled = point_dbz_bilinear(
        echo_to_dbz(trajectory.frames_linear,
                    min_dbz=problem.frozen.nowcast_config.min_dbz),
        problem.observation_coordinates,
    )
    observation_difference = float((sampled - problem.observation_dbz).abs().max())
    forecast_stages = 0
    def count_forecast(_echo: torch.Tensor, _qx: torch.Tensor, _qy: torch.Tensor) -> None:
        nonlocal forecast_stages
        forecast_stages += 1
    with torch.no_grad(), transport.observe_minmod_stages(count_forecast):
        actual = problem.forecast(control, parameters)
    actual_difference = float((actual - truth).abs().max())
    actual_score = float((actual - truth).square().mean())
    return (
        analysis_stages == analysis["observed_minmod_stages"]
        and forecast_stages == terminal["observed_minmod_stages"]
        and math.isclose(observation_difference,
                         analysis["point_observation_max_abs_difference_dbz"],
                         rel_tol=1e-10, abs_tol=1e-12)
        and observation_difference <= 1e-9
        and torch.allclose(reported, actual, rtol=0, atol=1e-10)
        and math.isclose(actual_difference,
                         terminal["max_abs_difference_from_same_operator_truth_dbz"],
                         rel_tol=1e-10, abs_tol=1e-12)
        and actual_difference <= 1e-9
        and math.isclose(actual_score, terminal["score"], rel_tol=1e-10, abs_tol=1e-12)
        and math.isclose(actual_score, terminal["manual_terminal_mse"],
                         rel_tol=1e-10, abs_tol=1e-12)
        and math.isclose(float(problem.objective(control, parameters)),
                         terminal["objective"], rel_tol=1e-10, abs_tol=1e-12)
    )


def run(directory: Path) -> dict[str, object]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("point 3-hour output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_3h_forward.json"
    command = [sys.executable, str(probe.ROOT /
               "examples/weather_scenarios/fv_point_3h_forward_probe.py"),
               "--output", str(output)]
    source_before = probe._sources()
    plan_before, archive_before = probe._sha(probe.PLAN), probe._sha(probe.preflight.ARCHIVED_WARM)
    problem, control, parameters, truth = make_case()
    input_before = probe._identity(problem, control, parameters, truth)
    resource = run_guarded(command, wall_seconds=WALL_SECONDS,
                           rss_bytes=SAMPLED_RSS_BYTES,
                           report_path=directory / "point_3h_forward.resource.json",
                           log_path=directory / "point_3h_forward.log")
    child = None
    read_error = None
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    resource_ok = (
        resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("command") == command
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and _finite(resource.get("elapsed_seconds"))
        and 0 <= resource["elapsed_seconds"] <= WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
    )
    identity_ok = (
        isinstance(child, dict)
        and child.get("pid") == resource.get("child_pid")
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("plan_unchanged") is True
        and child.get("archive_unchanged") is True
        and child.get("source_before") == child.get("source_after") == source_before == probe._sources()
        and child.get("input_before") == child.get("input_after") == input_before == probe._identity(*make_case())
        and child.get("plan_sha256") == plan_before == probe.PLAN_SHA256 == probe._sha(probe.PLAN)
        and child.get("archive_sha256") == archive_before == probe._sha(probe.preflight.ARCHIVED_WARM)
    )
    numeric_ok = bool(isinstance(child, dict) and resource_ok and identity_ok
                      and _valid_numerics(child, problem, control, parameters, truth))
    result: dict[str, object] = {
        "execution_status": "completed" if numeric_ok else "failed",
        "forward_validation": child.get("forward_validation") if isinstance(child, dict) else "not_performed",
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "child_read_error": read_error,
        "resource": resource,
        "scope": "fixed off-grid point observations with terminal 18-lead/3-hour same-operator FV forward only",
    }
    (directory / "point_3h_forward.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
