"""One bounded point-observation FV terminal forecast over 18 ten-minute leads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport, variational as v
from advar.fv_point_sampler import point_dbz_bilinear
from advar.physics import echo_to_dbz
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios.fv_point_3h_forward_case import make_case


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_3H_FORWARD_PLAN.md"
PLAN_SHA256 = "bd0898de43c5c3eba0001c50d89e223af1fc6145b897c42ff68b1f49243060fc"
SOURCE_PATHS = tuple(dict.fromkeys((*preflight.SOURCE_PATHS,
    "examples/weather_scenarios/fv_long_horizon_case.py",
    "examples/weather_scenarios/fv_point_3h_forward_case.py",
    "examples/weather_scenarios/fv_point_3h_forward_probe.py",
    "examples/weather_scenarios/fv_point_3h_forward_runner.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _identity(problem: Any, control: Tensor, parameters: Tensor, truth: Tensor) -> dict[str, Any]:
    return {"problem": problem.identity, "control_sha256": _tensor_sha(control),
            "parameters_sha256": _tensor_sha(parameters),
            "terminal_truth_sha256": _tensor_sha(truth)}


def run(output: Path) -> dict[str, Any]:
    source_before = _sources()
    plan_before = _sha(PLAN)
    archive_before = _sha(preflight.ARCHIVED_WARM)
    if plan_before != PLAN_SHA256:
        raise ValueError("point 3-hour forward declared plan changed")
    problem, control, parameters, truth = make_case()
    input_before = _identity(problem, control, parameters, truth)
    started = time.monotonic()
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "forward", "execution_phase": "running",
        "forward_validation": "not_performed",
        "stationarity_passed": "not_tested", "response_computed": False,
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "plan_sha256": PLAN_SHA256,
        "archive_sha256": archive_before,
        "source_before": source_before, "input_before": input_before,
        "scope": "12 fixed off-grid dBZ point values at 0/10/20 minutes; one terminal 3-hour forecast at 200 minutes absolute; same-operator synthetic truth only",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        spec = problem.frozen.fv_transport
        if spec is None:
            raise ValueError("point 3-hour fixture requires FV transport")
        layout = problem.layout
        steps = problem.frozen.nowcast_config.forecast_steps
        substeps = spec.substeps_per_interval
        if (layout["controls"] != 26 or layout["parameters"] != 13
                or layout["observation_times_seconds"] != (0.0, 600.0, 1200.0)
                or layout["forecast_time_seconds"] != 12000.0
                or layout["euler_stages"] != 3600
                or steps != 18 or substeps != 90
                or len(spec.boundary_echo) != 2 * substeps
                or len(problem.future_boundary_echo) != steps * substeps
                or not bool(problem.frozen.initial_support_mask.all())
                or not torch.equal(problem.background_dbz,
                                   problem.frozen.initial_background_dbz)):
            raise ValueError("point 3-hour grid/time/support contract changed")
        report["layout"] = layout
        report["analysis_boundary_stage_pairs"] = len(spec.boundary_echo)
        report["future_boundary_stage_pairs"] = len(problem.future_boundary_echo)
        report["forecast_steps"] = steps
        report["substeps_per_interval"] = substeps
        save()

        analysis_stages = 0
        def count_analysis(_echo: Tensor, _qx: Tensor, _qy: Tensor) -> None:
            nonlocal analysis_stages
            analysis_stages += 1
        analysis_tick = time.monotonic()
        with torch.no_grad(), transport.observe_minmod_stages(count_analysis):
            trajectory = v.analysis_trajectory(control, problem.contract(parameters))
        sampled = point_dbz_bilinear(
            echo_to_dbz(trajectory.frames_linear,
                        min_dbz=problem.frozen.nowcast_config.min_dbz),
            problem.observation_coordinates,
        )
        observation_difference = float((sampled - problem.observation_dbz).abs().max())
        report["analysis"] = {
            "observed_minmod_stages": analysis_stages,
            "point_observation_max_abs_difference_dbz": observation_difference,
            "point_observation_sha256": _tensor_sha(problem.observation_dbz),
            "seconds": time.monotonic() - analysis_tick,
        }
        if analysis_stages != 4 * substeps or observation_difference > 1e-9:
            raise ValueError("point 3-hour analysis/observation reproduction failed")
        report["phase"] = "terminal_forecast"
        save()

        forecast_stages = 0
        def count_forecast(_echo: Tensor, _qx: Tensor, _qy: Tensor) -> None:
            nonlocal forecast_stages
            forecast_stages += 1
        forecast_tick = time.monotonic()
        with torch.no_grad(), transport.observe_minmod_stages(count_forecast):
            forecast = problem.forecast(control, parameters)
        difference = float((forecast - truth).abs().max())
        manual_score = float((forecast - truth).square().mean())
        objective = float(problem.objective(control, parameters))
        score = float(problem.score(control, parameters))
        report["terminal"] = {
            "forecast_shape": list(forecast.shape),
            "forecast_sha256": _tensor_sha(forecast),
            "terminal_truth_sha256": _tensor_sha(truth),
            "forecast": forecast.tolist(),
            "observed_minmod_stages": forecast_stages,
            "max_abs_difference_from_same_operator_truth_dbz": difference,
            "objective": objective, "score": score,
            "manual_terminal_mse": manual_score,
            "seconds": time.monotonic() - forecast_tick,
        }
        if (forecast.shape != truth.shape or forecast_stages != layout["euler_stages"]
                or not all(math.isfinite(value) for value in
                           (difference, objective, score, manual_score))
                or difference > 1e-9
                or abs(score - manual_score) > 1e-12):
            report["forward_validation"] = "failed"
            report["execution_phase"] = "finished"
            save()
            return report
        report["phase"] = "branch_diagnostic"
        save()

        branch_stages = 0
        def count_branch(_echo: Tensor, _qx: Tensor, _qy: Tensor) -> None:
            nonlocal branch_stages
            branch_stages += 1
        try:
            with transport.observe_minmod_stages(count_branch):
                branch, scope = problem.branch_check(control, parameters)
            branch_record: dict[str, Any] = {
                "status": "passed_pointwise", "euler_stages": branch["euler_stages"],
                "scope": scope,
            }
        except ValueError as error:
            branch_record = {"status": "refused", "reason": str(error)}
        branch_record["observed_minmod_stages"] = branch_stages
        report["strict_branch_diagnostic"] = branch_record
        report.update(
            forward_validation="passed", execution_phase="finished",
            phase="finished", forecast_available=True,
        )
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = _identity(problem, control, parameters, truth)
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["archive_unchanged"] = _sha(preflight.ARCHIVED_WARM) == archive_before
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "plan_unchanged", "archive_unchanged"
        )):
            report.update(forward_validation="not_performed", execution_phase="identity_refused",
                          forecast_available=False)
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.output)
    raise SystemExit(0 if result["forward_validation"] == "passed" else 2)
