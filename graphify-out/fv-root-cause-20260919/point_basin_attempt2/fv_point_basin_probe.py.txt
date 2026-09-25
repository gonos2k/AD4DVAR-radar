"""One source-bound point-objective basin search and exact-root handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from advar import matrix_free
from advar.local_refinement import refine_stationary
from examples.weather_scenarios import fv_point_basin_search as basin
from examples.weather_scenarios import fv_point_response_preflight as preflight


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_BASIN_EXPLORATION2_PLAN.md"
PREFLIGHT = EVIDENCE / "point_response_preflight_attempt1/point_response_preflight.json"
PREFLIGHT_SHA256 = "d8377046433ad8d63bf32f3f60b81085061d4be4d7ccf6591e5df69f9a3dfbd5"
ATTEMPT1 = EVIDENCE / "point_basin_attempt1/point_basin.json"
ATTEMPT1_SHA256 = "e0cfde720bd2002a7d8149eca3553c244a9f93f45d256e3324066c58e9a158dc"
SOURCE_PATHS = (*preflight.SOURCE_PATHS,
                "examples/weather_scenarios/fv_point_basin_search.py",
                "examples/weather_scenarios/fv_point_basin_probe.py",
                "examples/weather_scenarios/fv_point_basin_runner.py")


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


def _strict_branch(problem: Any, control: Tensor, parameters: Tensor) -> tuple[dict[str, Any], str]:
    branch, scope, face = preflight._branch(problem, control, parameters)
    if (branch["euler_stages"] != 54
            or not math.isfinite(float(branch["minimum_scaled_slope_margin"]))
            or not math.isfinite(face)
            or branch["minimum_scaled_slope_margin"] <= 1e-4
            or face <= 1e-4):
        raise ValueError("basin stage, slope or face margin refused")
    return {**branch, "minimum_scaled_face_flux_margin": face}, scope


def _warm_branch(
    problem: Any, control: Tensor, parameters: Tensor, expected: dict[str, Any],
) -> dict[str, Any]:
    branch, _ = _strict_branch(problem, control, parameters)
    if _signature(branch) != _signature(expected):
        raise ValueError("basin warm branch signature changed")
    return branch


def _exploration_branch(problem: Any, control: Tensor, parameters: Tensor) -> tuple[dict[str, Any], str]:
    branch, scope, face = preflight._branch(problem, control, parameters)
    slope = float(branch["minimum_scaled_slope_margin"])
    if (branch["euler_stages"] != 54 or not math.isfinite(slope)
            or not math.isfinite(face) or slope <= 0 or face <= 0):
        raise ValueError("exploration stage or pointwise smoothness refused")
    return {**branch, "minimum_scaled_face_flux_margin": face}, scope


def _hessian_audit(problem: Any, control: Tensor, parameters: Tensor) -> dict[str, float]:
    gradient = torch.func.grad(problem.objective, argnums=0)
    columns = []
    for direction in torch.eye(control.numel(), dtype=control.dtype):
        column = torch.func.jvp(
            lambda c: gradient(c, parameters), (control,), (direction,)
        )[1]
        if column.shape != control.shape or not bool(torch.isfinite(column).all()):
            raise ValueError("basin terminal exact HVP is invalid")
        columns.append(column)
    hessian = torch.stack(columns, dim=1)
    scale = float(torch.linalg.matrix_norm(hessian))
    symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T)) / max(
        scale, torch.finfo(control.dtype).tiny
    )
    if not math.isfinite(scale) or not math.isfinite(symmetry) or symmetry > 1e-10:
        raise ValueError("basin terminal exact Hessian is nonfinite or asymmetric")
    eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    minimum, maximum = float(eigenvalues[0]), float(eigenvalues[-1])
    ratio = minimum / maximum if maximum > 0 else -math.inf
    if (not all(math.isfinite(value) for value in (minimum, maximum, ratio))
            or minimum <= 0 or ratio <= math.sqrt(torch.finfo(control.dtype).eps)):
        raise ValueError("basin terminal exact Hessian is not sufficiently SPD")
    return {"symmetry_relative": symmetry, "lambda_min": minimum,
            "lambda_max": maximum, "lambda_ratio": ratio,
            "hvp_columns": control.numel()}


def run(output: Path) -> dict[str, Any]:
    source_before = _sources()
    plan_before = _sha(PLAN)
    preflight_plan_before = _sha(preflight.PLAN)
    if _sha(ATTEMPT1) != ATTEMPT1_SHA256:
        raise ValueError("basin attempt1 evidence changed")
    if _sha(PREFLIGHT) != PREFLIGHT_SHA256:
        raise ValueError("basin preflight artifact changed")
    pinned = json.loads(PREFLIGHT.read_text())
    if (pinned.get("status") != "preflight_only"
            or pinned.get("numerical_solver_runs") != 0
            or pinned.get("plan_sha256") != preflight_plan_before
            or pinned.get("source_sha256") != {
                name: source_before[name] for name in preflight.SOURCE_PATHS
            }):
        raise ValueError("basin preflight source or plan changed")
    problem, warm, parameters, direction = preflight.fixed_problem()
    input_before = preflight._input_identity(problem, warm, parameters, direction)
    if (pinned.get("problem_identity") != input_before["problem_identity"]
            or pinned.get("tensor_sha256") != input_before["tensor_sha256"]):
        raise ValueError("basin fixed inputs changed")
    warm_branch = _warm_branch(problem, warm, parameters, pinned["warm_branch"])

    started = time.monotonic()
    accepted_events: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    first_refusal_examples: dict[str, dict[str, Any]] = {}
    linear_solves: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "basin_search",
        "numerical_status": "running", "response_validation": "not_performed",
        "preflight_sha256": PREFLIGHT_SHA256,
        "attempt1_sha256": ATTEMPT1_SHA256,
        "preflight_plan_sha256": preflight_plan_before,
        "plan_sha256": plan_before,
        "source_before": source_before, "input_before": input_before,
        "warm_control": warm.detach().tolist(),
        "warm_objective": pinned["warm_objective"],
        "warm_gradient_max": pinned["warm_gradient_max"],
        "accepted_events": accepted_events,
        "rejection_counts": rejection_counts,
        "first_refusal_examples": first_refusal_examples,
        "linear_solves": linear_solves,
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def strict_branch(control: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
        return _strict_branch(problem, control, p)

    def checkpoint(record: dict[str, Any]) -> None:
        category = str(record["status"])
        if category == "accepted":
            accepted_events.append(record)
            report["last_accepted_control_sha256"] = preflight._tensor_sha(
                torch.tensor(record["control"], dtype=torch.float64)
            )
            save()
        else:
            previous = rejection_counts.get(category, 0)
            rejection_counts[category] = previous + 1
            if previous == 0:
                first_refusal_examples[category] = record
                save()

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
        search = basin.search_basin(
            problem.objective, warm, parameters,
            lambda control, p: _exploration_branch(problem, control, p),
            checkpoint=checkpoint, max_steps=100, max_backtracks=16,
            max_trials=1600, memory=10, phase_seconds=600,
            handoff_gradient_max=1e-4,
        )
        report.update(
            basin_status=search.status,
            basin_objective=search.objective,
            basin_gradient_max=search.gradient_max,
            basin_accepted_steps=search.accepted_steps,
            basin_trial_evaluations=search.trial_evaluations,
            basin_branch_changes=search.branch_changes,
            basin_history=search.history,
            basin_control=search.control.detach().tolist(),
        )
        save()
        if search.status != "candidate":
            report.update(numerical_status="basin_incomplete", phase="basin_search")
        else:
            report["phase"] = "terminal_eligibility"
            terminal_branch, _ = strict_branch(search.control, parameters)
            report["basin_terminal_branch"] = {
                "signature_sha256": _signature_sha(terminal_branch),
                "minimum_scaled_slope_margin": terminal_branch["minimum_scaled_slope_margin"],
                "minimum_scaled_face_flux_margin": terminal_branch["minimum_scaled_face_flux_margin"],
            }
            report["phase"] = "curvature"
            save()
            try:
                report["curvature"] = _hessian_audit(problem, search.control, parameters)
            except ValueError as error:
                report.update(numerical_status="curvature_refused", refusal=str(error))
            else:
                report["phase"] = "exact_refinement"
                save()
                expected = _signature(terminal_branch)

                def pinned_branch(control: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
                    branch, scope = strict_branch(control, p)
                    if _signature(branch) != expected:
                        raise ValueError("basin terminal refinement branch changed")
                    return branch, scope

                try:
                    with matrix_free.observe_pcg_calls(monitor):
                        refined = refine_stationary(
                            problem.objective, search.control, parameters,
                            branch_check=pinned_branch, max_iterations=8,
                            max_backtracks=16, pcg_max_iterations=104,
                        )
                    final_branch, _ = pinned_branch(refined.control, parameters)
                    gradient = torch.func.grad(problem.objective, argnums=0)(
                        refined.control, parameters
                    )
                    if (not bool(torch.isfinite(gradient).all())
                            or float(gradient.abs().max()) >= 1e-10):
                        raise ValueError("basin exact refinement stationarity refused")
                    final_curvature = _hessian_audit(
                        problem, refined.control, parameters
                    )
                    report.update(
                        numerical_status="nominal_stationarity", phase="finished",
                        control=refined.control.detach().tolist(),
                        objective=float(problem.objective(refined.control, parameters)),
                        score=float(problem.score(refined.control, parameters)),
                        gradient_max=float(gradient.abs().max()),
                        gradient_norm=float(torch.linalg.vector_norm(gradient)),
                        refinement_iterations=refined.iterations,
                        refinement_hvp_count=refined.hvp_count,
                        refinement_history=refined.history,
                        final_branch_signature_sha256=_signature_sha(final_branch),
                        final_branch={
                            "euler_stages": final_branch["euler_stages"],
                            "minimum_scaled_slope_margin": final_branch[
                                "minimum_scaled_slope_margin"
                            ],
                            "minimum_scaled_face_flux_margin": final_branch[
                                "minimum_scaled_face_flux_margin"
                            ],
                            "signature_sha256": _signature_sha(final_branch),
                        },
                        final_curvature=final_curvature,
                    )
                except (RuntimeError, ValueError) as error:
                    report.update(numerical_status="refinement_refused",
                                  refusal=f"{type(error).__name__}: {error}")
    except (RuntimeError, ValueError) as error:
        category = ("exploration_terminal_ineligible"
                    if report["phase"] == "terminal_eligibility" else "basin_refused")
        report.update(numerical_status=category,
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
        report["preflight_unchanged"] = _sha(PREFLIGHT) == PREFLIGHT_SHA256
        report["attempt1_unchanged"] = _sha(ATTEMPT1) == ATTEMPT1_SHA256
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "plan_unchanged",
            "preflight_plan_unchanged", "preflight_unchanged", "attempt1_unchanged",
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
