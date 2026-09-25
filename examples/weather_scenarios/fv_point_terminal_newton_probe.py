"""Exact Hessian and Newton feasibility from one pinned point-search endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import torch
from torch import Tensor

from advar import matrix_free
from advar.local_refinement import refine_stationary
from examples.weather_scenarios import fv_point_basin_probe as basin_probe
from examples.weather_scenarios import fv_point_response_preflight as preflight


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_TERMINAL_NEWTON_PLAN.md"
PLAN_SHA256 = "396de67a8fe177234ef2bf1eb69d57d20e7f2dbb90de522b33f840e789393ccd"
PREFLIGHT = EVIDENCE / "point_response_preflight_attempt1/point_response_preflight.json"
PREFLIGHT_SHA256 = "d8377046433ad8d63bf32f3f60b81085061d4be4d7ccf6591e5df69f9a3dfbd5"
ATTEMPT2 = EVIDENCE / "point_basin_attempt2/point_basin.json"
ATTEMPT2_SHA256 = "fbafbab3736abac64421356b67ab0896ea31976e31b0c43f41eea933414e7a51"
CONTROL_SHA256 = "8f5110f2755217dece8ca9f6bbab23ceadb184b4f1a32b8e4654ddfde5f4b258"
BRANCH_SHA256 = "0783f66a062d5144894513c65a4e221befca4d23a22d8fba0adc43ead5011327"
SOURCE_PATHS = tuple(dict.fromkeys((*basin_probe.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_terminal_newton_probe.py",
    "examples/weather_scenarios/fv_point_terminal_newton_runner.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _close(label: str, actual: float, expected: float) -> None:
    if (not math.isfinite(actual) or not math.isfinite(expected)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError(f"terminal Newton {label} changed from pinned search endpoint")


def run(output: Path) -> dict[str, Any]:
    source_before = _sources()
    plan_before = _sha(PLAN)
    if plan_before != PLAN_SHA256:
        raise ValueError("terminal Newton declared plan changed")
    preflight_plan_before = _sha(preflight.PLAN)
    if _sha(PREFLIGHT) != PREFLIGHT_SHA256 or _sha(ATTEMPT2) != ATTEMPT2_SHA256:
        raise ValueError("terminal Newton pinned input report changed")
    pinned_preflight = json.loads(PREFLIGHT.read_text())
    pinned_search = json.loads(ATTEMPT2.read_text())
    if (pinned_preflight.get("status") != "preflight_only"
            or pinned_preflight.get("plan_sha256") != preflight_plan_before
            or pinned_search.get("numerical_status") != "basin_incomplete"
            or pinned_search.get("basin_status") != "step_budget"
            or pinned_search.get("basin_accepted_steps") != 100
            or pinned_search.get("response_validation") != "not_performed"):
        raise ValueError("terminal Newton pinned status or plan changed")
    if pinned_search.get("source_before") != {
        name: source_before[name] for name in basin_probe.SOURCE_PATHS
    }:
        raise ValueError("terminal Newton search-source identity changed")
    problem, warm, parameters, direction = preflight.fixed_problem()
    input_before = preflight._input_identity(problem, warm, parameters, direction)
    if (pinned_search.get("input_before") != input_before
            or pinned_preflight.get("problem_identity") != input_before["problem_identity"]
            or pinned_preflight.get("tensor_sha256") != input_before["tensor_sha256"]):
        raise ValueError("terminal Newton fixed problem/input changed")
    events = pinned_search["accepted_events"]
    last = events[-1]
    if (len(events) != 100 or last.get("iteration") != 100
            or last.get("status") != "accepted"):
        raise ValueError("terminal Newton selected event is not the deterministic last step")
    control = torch.tensor(last["control"], dtype=torch.float64)
    if (control.shape != warm.shape
            or preflight._tensor_sha(control) != CONTROL_SHA256
            or pinned_search.get("last_accepted_control_sha256") != CONTROL_SHA256
            or last.get("signature_sha256") != BRANCH_SHA256):
        raise ValueError("terminal Newton selected control or branch digest changed")
    expected_signature = json.loads(last["branch_key"])
    actual_branch, _ = basin_probe._strict_branch(problem, control, parameters)
    if (basin_probe._signature(actual_branch) != expected_signature
            or basin_probe._signature_sha(actual_branch) != BRANCH_SHA256):
        raise ValueError("terminal Newton actual branch differs from pinned endpoint")
    _close("objective", float(problem.objective(control, parameters)), last["objective"])
    initial_gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    _close("gradient maximum", float(initial_gradient.abs().max()), last["gradient_max"])
    saved_gradient = torch.tensor(last["gradient"], dtype=torch.float64)
    if (saved_gradient.shape != initial_gradient.shape
            or not torch.allclose(initial_gradient, saved_gradient,
                                  rtol=1e-10, atol=1e-12)):
        raise ValueError("terminal Newton full seed gradient changed")
    _close("slope margin", float(actual_branch["minimum_scaled_slope_margin"]),
           last["minimum_scaled_slope_margin"])
    _close("face margin", float(actual_branch["minimum_scaled_face_flux_margin"]),
           last["minimum_scaled_face_flux_margin"])

    started = time.monotonic()
    branch_calls: list[dict[str, Any]] = []
    linear_solves: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "seed_curvature", "numerical_status": "running",
        "response_validation": "not_performed", "nonlinear_reanalyses": 0,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "plan_sha256": plan_before, "preflight_sha256": PREFLIGHT_SHA256,
        "preflight_plan_sha256": preflight_plan_before,
        "attempt2_sha256": ATTEMPT2_SHA256,
        "source_before": source_before, "input_before": input_before,
        "seed_control": control.detach().tolist(),
        "seed_control_sha256": CONTROL_SHA256,
        "seed_branch": {
            "euler_stages": actual_branch["euler_stages"],
            "minimum_scaled_slope_margin": actual_branch["minimum_scaled_slope_margin"],
            "minimum_scaled_face_flux_margin": actual_branch["minimum_scaled_face_flux_margin"],
            "signature_sha256": BRANCH_SHA256,
        },
        "seed_objective": float(problem.objective(control, parameters)),
        "seed_gradient": initial_gradient.detach().tolist(),
        "seed_gradient_max": float(initial_gradient.abs().max()),
        "branch_calls": branch_calls, "linear_solves": linear_solves,
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def pinned_branch(candidate: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
        record: dict[str, Any] = {"control_sha256": preflight._tensor_sha(candidate)}
        try:
            branch, scope = basin_probe._strict_branch(problem, candidate, p)
            record.update(
                euler_stages=branch["euler_stages"],
                minimum_scaled_slope_margin=branch["minimum_scaled_slope_margin"],
                minimum_scaled_face_flux_margin=branch["minimum_scaled_face_flux_margin"],
                signature_sha256=basin_probe._signature_sha(branch),
            )
            if basin_probe._signature(branch) != expected_signature:
                raise ValueError("terminal Newton branch changed from locked endpoint")
            record["status"] = "accepted_branch"
            return branch, scope
        except ValueError as error:
            record.update(status="refused_branch", reason=str(error))
            raise
        finally:
            branch_calls.append(record)

    def monitor(original):
        def solve(operator, rhs, **kwargs):
            entry: dict[str, Any] = {"hvp_calls": 0, "rtol": kwargs.get("rtol"),
                                     "max_iterations": kwargs.get("max_iterations")}

            def measured(value):
                result = operator(value)
                entry["hvp_calls"] += 1
                return result

            tick = time.monotonic()
            try:
                result = original(measured, rhs, **kwargs)
                entry.update(converged=result.converged, iterations=result.iterations,
                             relative_residual=result.relative_residual)
                return result
            except (RuntimeError, ValueError) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                entry["seconds"] = time.monotonic() - tick
                linear_solves.append(entry)

        return solve

    save()
    try:
        report["seed_curvature"] = basin_probe._hessian_audit(problem, control, parameters)
        report["phase"] = "exact_refinement"
        save()
        try:
            with matrix_free.observe_pcg_calls(monitor):
                refined = refine_stationary(
                    problem.objective, control, parameters,
                    branch_check=pinned_branch, max_iterations=8,
                    max_backtracks=16, pcg_max_iterations=104,
                )
            final_branch, _ = pinned_branch(refined.control, parameters)
            gradient = torch.func.grad(problem.objective, argnums=0)(
                refined.control, parameters
            )
            if (not bool(torch.isfinite(gradient).all())
                    or float(gradient.abs().max()) >= 1e-10):
                raise ValueError("terminal Newton final stationarity refused")
            report.update(
                control=refined.control.detach().tolist(),
                objective=float(problem.objective(refined.control, parameters)),
                score=float(problem.score(refined.control, parameters)),
                gradient_max=float(gradient.abs().max()),
                gradient_norm=float(torch.linalg.vector_norm(gradient)),
                refinement_iterations=refined.iterations,
                refinement_hvp_count=refined.hvp_count,
                refinement_history=refined.history,
                final_branch={
                    "euler_stages": final_branch["euler_stages"],
                    "minimum_scaled_slope_margin": final_branch["minimum_scaled_slope_margin"],
                    "minimum_scaled_face_flux_margin": final_branch["minimum_scaled_face_flux_margin"],
                    "signature_sha256": basin_probe._signature_sha(final_branch),
                },
            )
            save()
            report["final_curvature"] = basin_probe._hessian_audit(
                problem, refined.control, parameters
            )
            report.update(numerical_status="nominal_stationarity", phase="finished")
        except (RuntimeError, ValueError) as error:
            report.update(numerical_status="newton_refused",
                          refusal=f"{type(error).__name__}: {error}")
    except ValueError as error:
        report.update(numerical_status="seed_curvature_refused",
                      refusal=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = preflight._input_identity(problem, warm, parameters, direction)
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["seed_control_unchanged"] = preflight._tensor_sha(control) == CONTROL_SHA256
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["preflight_plan_unchanged"] = _sha(preflight.PLAN) == preflight_plan_before
        report["preflight_unchanged"] = _sha(PREFLIGHT) == PREFLIGHT_SHA256
        report["attempt2_unchanged"] = _sha(ATTEMPT2) == ATTEMPT2_SHA256
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "plan_unchanged",
            "seed_control_unchanged", "preflight_plan_unchanged",
            "preflight_unchanged", "attempt2_unchanged",
        )):
            report.update(numerical_status="identity_refused", phase="identity_recheck")
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.output)
    raise SystemExit(0 if result["numerical_status"] == "nominal_stationarity" else 2)
