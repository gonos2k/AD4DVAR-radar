"""One bounded nominal Newton feasibility check for the point FV objective."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from advar import matrix_free
from advar.local_refinement import refine_stationary
from examples.weather_scenarios import fv_point_response_preflight as preflight


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_NOMINAL_DIAGNOSTIC_PLAN.md"
PREFLIGHT_SHA256 = "d8377046433ad8d63bf32f3f60b81085061d4be4d7ccf6591e5df69f9a3dfbd5"
SOURCE_PATHS = (*preflight.SOURCE_PATHS,
                "examples/weather_scenarios/fv_point_nominal_probe.py",
                "examples/weather_scenarios/fv_point_nominal_runner.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _signature(branch: dict[str, Any]) -> dict[str, Any]:
    return {name: branch[name] for name in ("choices", "face_signs")}


def _signature_sha(branch: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        _signature(branch), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def run(output: Path, preflight_path: Path) -> dict[str, Any]:
    source_before = _sources()
    plan_before = _sha(PLAN)
    preflight_plan_before = _sha(preflight.PLAN)
    if _sha(preflight_path) != PREFLIGHT_SHA256:
        raise ValueError("nominal point preflight artifact changed")
    pinned = json.loads(preflight_path.read_text())
    if (pinned.get("status") != "preflight_only"
            or pinned.get("numerical_solver_runs") != 0
            or pinned.get("plan_sha256") != preflight_plan_before
            or pinned.get("source_sha256") != {
                name: source_before[name] for name in preflight.SOURCE_PATHS
            }):
        raise ValueError("nominal point preflight source contract changed")
    problem, warm, parameters, direction = preflight.fixed_problem()
    input_before = preflight._input_identity(problem, warm, parameters, direction)
    if (pinned.get("problem_identity") != input_before["problem_identity"]
            or pinned.get("tensor_sha256") != input_before["tensor_sha256"]):
        raise ValueError("nominal point preflight input identity changed")
    expected = pinned["warm_branch"]
    warm_branch, _, warm_face_margin = preflight._branch(problem, warm, parameters)
    if (_signature(warm_branch) != _signature(expected)
            or warm_branch["euler_stages"] != 54
            or warm_branch["minimum_scaled_slope_margin"] <= 1e-4
            or warm_face_margin <= 1e-4):
        raise ValueError("nominal point warm branch changed")

    started = time.monotonic()
    branch_calls: list[dict[str, Any]] = []
    linear_solves: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "phase": "nominal_refinement", "numerical_status": "running",
        "pid": os.getpid(),
        "response_validation": "not_performed", "nonlinear_reanalyses": 0,
        "preflight_sha256": PREFLIGHT_SHA256, "plan_sha256": plan_before,
        "preflight_plan_sha256": preflight_plan_before,
        "source_before": source_before, "input_before": input_before,
        "warm_objective": pinned["warm_objective"],
        "warm_gradient_max": pinned["warm_gradient_max"],
        "warm_branch_signature_sha256": _signature_sha(warm_branch),
        "branch_calls": branch_calls, "linear_solves": linear_solves,
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def checked_branch(control: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
        record: dict[str, Any] = {"control_sha256": preflight._tensor_sha(control)}
        try:
            branch, scope, face_margin = preflight._branch(problem, control, p)
            record.update(
                euler_stages=branch["euler_stages"],
                minimum_scaled_slope_margin=branch["minimum_scaled_slope_margin"],
                minimum_scaled_face_flux_margin=face_margin,
                signature_sha256=_signature_sha(branch),
            )
            if branch["euler_stages"] != 54:
                raise ValueError("point nominal stage count changed")
            if branch["minimum_scaled_slope_margin"] <= 1e-4:
                raise ValueError("point nominal slope margin at or below 1e-4")
            if face_margin <= 1e-4:
                raise ValueError("point nominal face margin at or below 1e-4")
            if _signature(branch) != _signature(expected):
                raise ValueError("point nominal branch signature changed")
            record["status"] = "accepted_branch"
            return branch, scope
        except ValueError as error:
            record.update(status="refused_branch", reason=str(error))
            raise
        finally:
            branch_calls.append(record)

    def monitor(original):
        def solve(operator, rhs, **kwargs):
            entry: dict[str, Any] = {
                "rtol": kwargs.get("rtol"),
                "max_iterations": kwargs.get("max_iterations"),
                "hvp_calls": 0,
            }

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
            except (ValueError, RuntimeError) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                entry["seconds"] = time.monotonic() - tick
                linear_solves.append(entry)

        return solve

    save()
    try:
        with matrix_free.observe_pcg_calls(monitor):
            refined = refine_stationary(
                problem.objective, warm, parameters, branch_check=checked_branch,
                max_iterations=8, max_backtracks=16, pcg_max_iterations=104,
            )
        final_branch, _, final_face_margin = preflight._branch(
            problem, refined.control, parameters
        )
        gradient = torch.func.grad(problem.objective, argnums=0)(refined.control, parameters)
        if (refined.gradient_max >= 1e-10
                or not bool(torch.isfinite(gradient).all())
                or float(gradient.abs().max()) >= 1e-10
                or _signature(final_branch) != _signature(expected)
                or final_branch["minimum_scaled_slope_margin"] <= 1e-4
                or final_face_margin <= 1e-4):
            raise ValueError("point nominal terminal stationarity or branch gate failed")
        report.update(
            numerical_status="nominal_stationarity", phase="finished",
            control=refined.control.detach().tolist(),
            objective=float(problem.objective(refined.control, parameters)),
            score=float(problem.score(refined.control, parameters)),
            gradient_max=float(gradient.abs().max()),
            gradient_norm=float(torch.linalg.vector_norm(gradient)),
            iterations=refined.iterations, hvp_count=refined.hvp_count,
            refinement_history=refined.history,
            final_branch={"euler_stages": 54,
                          "minimum_scaled_slope_margin": final_branch["minimum_scaled_slope_margin"],
                          "minimum_scaled_face_flux_margin": final_face_margin,
                          "signature_sha256": _signature_sha(final_branch)},
        )
    except (ValueError, RuntimeError) as error:
        report.update(numerical_status="refused", phase="nominal_refinement",
                      refusal=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = preflight._input_identity(
            problem, warm, parameters, direction
        )
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == plan_before
        report["preflight_plan_unchanged"] = _sha(preflight.PLAN) == preflight_plan_before
        report["preflight_unchanged"] = _sha(preflight_path) == PREFLIGHT_SHA256
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "plan_unchanged",
            "preflight_plan_unchanged", "preflight_unchanged"
        )):
            report.update(numerical_status="identity_refused", phase="identity_recheck")
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.output, arguments.preflight)
    raise SystemExit(0 if result["numerical_status"] == "nominal_stationarity" else 2)
