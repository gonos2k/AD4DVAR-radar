"""One bounded concurrent 4x5/8x10 FV local-response integration check.

Uses archived stationary controls and fixed synthetic scores. No GN,
refinement, signed reanalysis, finite-path or physical-skill claim.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from threading import Barrier
import time
from typing import Any, cast

import torch
from torch import Tensor

from advar import matrix_free, transport, variational as v
from advar.fv_research_problem import FVResearchProblem
from advar.local_response import compute_local_response
from advar.physics import echo_to_dbz
from advar.transport import BoundarySchedule
from examples.weather_scenarios import fv_minmod_inverse_probe as small_fixture
from examples.weather_scenarios import fv_scaled_research_case as large_fixture


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
ARCHIVED = {
    "minmod_middle_time_bias_final.json": "a0f9568b59bce283b39d1527d3ab64f7034d41c3d41eb3e843ca0ffd7b8fce88",
    "minmod_parameter_vjp.json": "b207c838945baaedffdd3d53e6a41ad124950b9b4a96ea771f59953c0bd25456",
    "fv86_seed_a.json": "c8de13325e4d76a5e904a6ef457deed853aa762cc6d051afd7fb59cd6b9e3bd0",
    "fv86_reanalysis.json": "c80282301138ccae67ce49b86c1d1f2dcc88bc8a144b571eecbe23904ed5a302",
}
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


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_hash(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _tensor_identity(value: Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _bytes_hash(value)}


def _archive(name: str) -> dict[str, Any]:
    path = EVIDENCE / name
    if _hash(path) != ARCHIVED[name]:
        raise ValueError(f"archived baseline changed: {name}")
    return json.loads(path.read_text())


def _small_case() -> tuple[FVResearchProblem, Tensor, Tensor, Tensor, float]:
    saved = _archive("minmod_middle_time_bias_final.json")
    archived_response = _archive("minmod_parameter_vjp.json")
    observations, frozen, future_echo, future_support = small_fixture.make_spatial_case()
    future_support = cast(BoundarySchedule, future_support)
    control = torch.tensor(saved["nominal_control"], dtype=torch.float64)
    parameters = torch.cat((observations.dbz.flatten(), observations.dbz.new_tensor([0.02])))
    if (_bytes_hash(control) != saved["input_identity"]["control_sha256"]
            or _bytes_hash(parameters) != saved["input_identity"]["parameters_sha256"]):
        raise ValueError("4x5 archived stationary input identity mismatch")
    pattern = torch.linspace(-0.2, 0.3, 20, dtype=control.dtype).reshape(4, 5)
    contract = replace(frozen, initial_background_dbz=observations.dbz[0] + parameters[-1] * pattern)
    forecast = v.forecast_fv_analysis(
        control, contract, leads=1, boundary_start_interval=2,
        boundary_echo=future_echo, boundary_support=future_support,
    ).frames_linear[-1]
    verification = (echo_to_dbz(forecast, min_dbz=frozen.nowcast_config.min_dbz) + 0.1 * pattern).detach()
    if _bytes_hash(verification) != archived_response["input_identity"]["verification"]:
        raise ValueError("4x5 verification differs from archived score")
    problem = FVResearchProblem(
        observations, frozen, future_echo, future_support, pattern, verification,
        small_fixture.inspect_branches, saved["nominal_branch"],
    )
    direction = torch.zeros_like(parameters)
    direction[20:40] = 1.0
    if not torch.equal(direction, torch.tensor(
        saved["tangents"]["middle_time_bias"]["parameter_direction"], dtype=direction.dtype,
    )):
        raise ValueError("4x5 direction differs from archived signed response")
    expected = archived_response["directions"]["middle_time_bias"]["total"]
    return problem, control, parameters, direction, expected


def _large_case() -> tuple[FVResearchProblem, Tensor, Tensor, Tensor, float]:
    saved = _archive("fv86_seed_a.json")
    case = large_fixture.make_case()
    control = torch.tensor(saved["workflow"]["control"], dtype=torch.float64)
    parameters = case.parameters
    if (_bytes_hash(parameters) != saved["input_identity"]["parameters"]
            or _bytes_hash(case.verification) != saved["input_identity"]["verification"]):
        raise ValueError("8x10 archived input/verification identity mismatch")
    matching = [row for row in saved["branch_checks"] if row["control_sha256"] == _bytes_hash(control)]
    if not matching:
        raise ValueError("8x10 refined control lacks archived branch check")
    problem = large_fixture.make_problem(case, saved["nominal_branch"])
    direction = torch.zeros_like(parameters)
    direction[80:160] = 1.0
    archived_direction = _archive("fv86_reanalysis.json")["direction"]
    if (not torch.equal(direction, torch.tensor(archived_direction["values"], dtype=direction.dtype))
            or _bytes_hash(direction) != archived_direction["sha256"]):
        raise ValueError("8x10 direction differs from archived signed response")
    expected = saved["workflow"]["response"]["total"]["middle_time_bias"]
    return problem, control, parameters, direction, expected


def run(output: Path) -> dict[str, Any]:
    source_before = {name: _hash(ROOT / name) for name in SOURCE_PATHS}
    report: dict[str, Any] = {
        "status": "running", "phase": "prepare", "source_sha256": source_before,
        "archived_report_sha256": ARCHIVED, "workers": {},
        "nonlinear_analyses": 0, "nonlinear_reanalyses": 0,
        "finite_path_certified": False, "physical_validation": "not_performed",
    }

    def save() -> None:
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    try:
        cases = {"fv4x5": _small_case(), "fv8x10": _large_case()}
        report["phase"] = "concurrent_response"
        report["input_identity"] = {
            name: {"problem": problem.identity["fixed_problem_sha256"],
                   "control": _tensor_identity(control), "parameters": _tensor_identity(parameters),
                   "verification": _tensor_identity(problem.verification),
                   "direction": _tensor_identity(direction)}
            for name, (problem, control, parameters, direction, _) in cases.items()
        }
        save()
        rendezvous = Barrier(2)
        stage_rendezvous = Barrier(2)
        pcg_rendezvous = Barrier(2)

        def worker(name: str) -> dict[str, Any]:
            problem, control, parameters, direction, expected = cases[name]
            solves: list[dict[str, Any]] = []
            stage_count = 0
            stage_first: float | None = None
            stage_last: float | None = None

            def count_stage(q: Tensor, qx: Tensor, qy: Tensor) -> None:
                nonlocal stage_count, stage_first, stage_last
                if stage_count == 0:
                    stage_first = time.monotonic()
                    stage_rendezvous.wait(timeout=30)
                stage_count += 1
                stage_last = time.monotonic()

            def checked_branch(c: Tensor, p: Tensor):
                # Stage snapshots belong only to the ordinary branch forecast.
                # HVP/VJP/replay calculations below do not inherit this observer.
                with transport.observe_minmod_stages(count_stage):
                    return problem.branch_check(c, p)

            def monitor(original):
                def solve(operator, rhs, **kwargs):
                    entry: dict[str, Any] = {
                        "rhs": _tensor_identity(rhs),
                        "rtol": kwargs.get("rtol"),
                        "max_iterations": kwargs.get("max_iterations"),
                        "hvp_calls": 0,
                        "hvp_event_times": [],
                    }
                    started = time.monotonic()
                    entry["started"] = started

                    def measured(value):
                        if entry["hvp_calls"] == 0:
                            pcg_rendezvous.wait(timeout=30)
                        result = operator(value)
                        entry["hvp_calls"] += 1
                        entry["hvp_event_times"].append(time.monotonic())
                        return result

                    try:
                        result = original(measured, rhs, **kwargs)
                        entry.update(converged=result.converged, iterations=result.iterations,
                                     relative_residual=result.relative_residual)
                        return result
                    finally:
                        entry["ended"] = time.monotonic()
                        entry["elapsed_seconds"] = entry["ended"] - started
                        solves.append(entry)

                return solve

            with matrix_free.observe_pcg_calls(monitor):
                rendezvous.wait(timeout=30)
                started = time.monotonic()
                response = compute_local_response(
                    problem.objective, problem.score, control, parameters,
                    {"middle_time_bias": direction},
                    branch_check=checked_branch,
                    input_identity=report["input_identity"][name],
                )
                ended = time.monotonic()
            total = float(response.total["middle_time_bias"])
            error = abs(total - expected) / abs(expected)
            if (response.gradient_max >= 1e-10
                    or response.true_adjoint_relative_residual > 1e-10
                    or response.branch_signature["euler_stages"] != (54 if name == "fv4x5" else 108)
                    or stage_count != response.branch_signature["euler_stages"]
                    or stage_first is None or stage_last is None
                    or error > 1e-6 or len(solves) != 1
                    or solves[0]["hvp_calls"] != response.hvp_count
                    or solves[0]["iterations"] != response.pcg_iterations
                    or solves[0]["relative_residual"] != response.pcg_relative_residual):
                raise ValueError(f"{name} local response or instrumentation gate failed")
            return {
                "started": started, "ended": ended,
                "gradient_max": response.gradient_max,
                "true_adjoint_residual": response.true_adjoint_residual,
                "true_adjoint_relative_residual": response.true_adjoint_relative_residual,
                "branch_euler_stages": response.branch_signature["euler_stages"],
                "outer_stage_observer_events": stage_count,
                "stage_observer_first": stage_first,
                "stage_observer_last": stage_last,
                "response_hvp_count": response.hvp_count,
                "response_pcg_iterations": response.pcg_iterations,
                "response_pcg_relative_residual": response.pcg_relative_residual,
                "branch_choices_sha256": hashlib.sha256(json.dumps(response.branch_signature["choices"], sort_keys=True).encode()).hexdigest(),
                "branch_face_signs_sha256": hashlib.sha256(json.dumps(response.branch_signature["face_signs"], sort_keys=True).encode()).hexdigest(),
                "direct": float(response.direct["middle_time_bias"]),
                "indirect": float(response.indirect["middle_time_bias"]),
                "total": total, "archived_total": expected,
                "relative_difference": error,
                "pcg": solves,
            }

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(worker, name) for name in cases}
            for name, future in futures.items():
                report["workers"][name] = future.result(timeout=260)
                save()
        windows = list(report["workers"].values())
        overlap = min(row["ended"] for row in windows) - max(row["started"] for row in windows)
        if overlap <= 0:
            raise ValueError("concurrent FV response intervals did not overlap")
        report["worker_overlap_seconds"] = overlap
        stage_overlap = min(row["stage_observer_last"] for row in windows) - max(
            row["stage_observer_first"] for row in windows
        )
        if stage_overlap <= 0:
            raise ValueError("concurrent minmod stage observer windows did not overlap")
        report["stage_observer_overlap_seconds"] = stage_overlap
        event_windows = [row["pcg"][0]["hvp_event_times"] for row in windows]
        if any(len(events) < 2 for events in event_windows):
            raise ValueError("both PCG solves need multiple observed HVP events")
        hvp_overlap = min(events[-1] for events in event_windows) - max(events[0] for events in event_windows)
        if hvp_overlap <= 0:
            raise ValueError("concurrent PCG HVP event windows did not overlap")
        report["pcg_hvp_event_overlap_seconds"] = hvp_overlap
        source_after = {name: _hash(ROOT / name) for name in SOURCE_PATHS}
        if source_after != source_before:
            raise RuntimeError("concurrent response source changed during execution")
        for name, (_, control, parameters, direction, _) in cases.items():
            expected_identity = report["input_identity"][name]
            if (expected_identity["problem"] != cases[name][0].identity["fixed_problem_sha256"]
                    or expected_identity["control"] != _tensor_identity(control)
                    or expected_identity["parameters"] != _tensor_identity(parameters)
                    or expected_identity["verification"] != _tensor_identity(cases[name][0].verification)
                    or expected_identity["direction"] != _tensor_identity(direction)):
                raise RuntimeError("concurrent response inputs changed during execution")
        report.update(status="completed", phase="finished", source_unchanged=True,
                      inputs_unchanged=True)
    except Exception as error:
        report.update(status="execution_error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
