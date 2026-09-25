"""One branch-conditioned point root diagnostic with unchanged Newton math."""
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
from examples.weather_scenarios import fv_point_terminal_newton_probe as previous


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_CORE_STRICT_ROOT_PLAN.md"
PLAN_SHA256 = "e2e0c49e1f59e05432bd127c33211d2f8d95532cd9560088203a8255958678d7"
PRIOR_REPORT = EVIDENCE / "point_terminal_newton_attempt1/point_terminal_newton.json"
PRIOR_SHA256 = "aff2a5e1374888b190c783f30bd7b8a468a0cc8bdb67a2cb562603308021d4f9"
SOURCE_PATHS = tuple(dict.fromkeys((*previous.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_core_strict_root_probe.py",
    "examples/weather_scenarios/fv_point_core_strict_root_runner.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _core_branch(
    problem: Any, control: Tensor, parameters: Tensor,
) -> tuple[dict[str, Any], str]:
    """Require the existing pointwise roundoff-aware oracle, not a bare sign."""
    branch, scope, face = preflight._branch(problem, control, parameters)
    slope = float(branch["minimum_scaled_slope_margin"])
    if (branch["euler_stages"] != 54 or not math.isfinite(slope)
            or not math.isfinite(face) or slope <= 0 or face <= 0):
        raise ValueError("core-strict stage or branch resolvability refused")
    return {**branch, "minimum_scaled_face_flux_margin": face}, scope


def run(output: Path) -> dict[str, Any]:
    source_before = _sources()
    plan_before = _sha(PLAN)
    preflight_plan_before = _sha(preflight.PLAN)
    if plan_before != PLAN_SHA256:
        raise ValueError("core-strict root declared plan changed")
    if (_sha(PRIOR_REPORT) != PRIOR_SHA256
            or _sha(previous.ATTEMPT2) != previous.ATTEMPT2_SHA256
            or _sha(previous.PREFLIGHT) != previous.PREFLIGHT_SHA256):
        raise ValueError("core-strict root pinned report changed")
    pinned_preflight = json.loads(previous.PREFLIGHT.read_text())
    if pinned_preflight.get("plan_sha256") != preflight_plan_before:
        raise ValueError("core-strict root preflight plan changed")
    prior = json.loads(PRIOR_REPORT.read_text())
    if (prior.get("numerical_status") != "newton_refused"
            or prior.get("phase") != "exact_refinement"
            or prior.get("seed_control_sha256") != previous.CONTROL_SHA256
            or prior.get("source_before") != {
                name: source_before[name] for name in previous.SOURCE_PATHS
            }):
        raise ValueError("core-strict root prior run/source status changed")
    problem, warm, parameters, direction = preflight.fixed_problem()
    input_before = preflight._input_identity(problem, warm, parameters, direction)
    if prior.get("input_before") != input_before:
        raise ValueError("core-strict root problem inputs changed")
    control = torch.tensor(prior["seed_control"], dtype=torch.float64)
    if (control.shape != warm.shape
            or preflight._tensor_sha(control) != previous.CONTROL_SHA256):
        raise ValueError("core-strict root seed control changed")
    seed_branch, _ = basin_probe._strict_branch(problem, control, parameters)
    expected = basin_probe._signature(seed_branch)
    if basin_probe._signature_sha(seed_branch) != previous.BRANCH_SHA256:
        raise ValueError("core-strict root seed branch changed")
    previous._close("seed objective", float(problem.objective(control, parameters)),
                    prior["seed_objective"])
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    seed_gradient = gradient_fn(control, parameters)
    if (not bool(torch.isfinite(seed_gradient).all())
            or not torch.allclose(seed_gradient,
                                  torch.tensor(prior["seed_gradient"], dtype=torch.float64),
                                  rtol=1e-10, atol=1e-12)):
        raise ValueError("core-strict root seed gradient changed")

    started = time.monotonic()
    branch_calls: list[dict[str, Any]] = []
    linear_solves: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "seed_curvature",
        "numerical_status": "running", "response_validation": "not_performed",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "plan_sha256": PLAN_SHA256, "prior_report_sha256": PRIOR_SHA256,
        "preflight_plan_sha256": preflight_plan_before,
        "preflight_sha256": previous.PREFLIGHT_SHA256,
        "attempt2_sha256": previous.ATTEMPT2_SHA256,
        "source_before": source_before, "input_before": input_before,
        "seed_control": control.detach().tolist(),
        "seed_control_sha256": previous.CONTROL_SHA256,
        "seed_gradient": seed_gradient.detach().tolist(),
        "seed_branch": {
            "euler_stages": seed_branch["euler_stages"],
            "minimum_scaled_slope_margin": seed_branch["minimum_scaled_slope_margin"],
            "minimum_scaled_face_flux_margin": seed_branch["minimum_scaled_face_flux_margin"],
            "signature_sha256": previous.BRANCH_SHA256,
        },
        "branch_calls": branch_calls, "linear_solves": linear_solves,
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def pinned_core_branch(candidate: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
        record: dict[str, Any] = {
            "control_sha256": preflight._tensor_sha(candidate),
            "control": candidate.detach().tolist(),
            "status": "trace_pending",
        }
        try:
            branch, scope = _core_branch(problem, candidate, p)
            matches = basin_probe._signature(branch) == expected
            record.update(
                euler_stages=branch["euler_stages"],
                minimum_scaled_slope_margin=branch["minimum_scaled_slope_margin"],
                minimum_scaled_face_flux_margin=branch["minimum_scaled_face_flux_margin"],
                signature_sha256=basin_probe._signature_sha(branch),
                signature_matches_seed=matches,
                response_margin_qualified=(
                    branch["minimum_scaled_slope_margin"] > 1e-4
                    and branch["minimum_scaled_face_flux_margin"] > 1e-4
                ),
            )
            # Diagnostic evaluation is deliberately separate from the
            # refiner's later merit calculation, including signature misses.
            candidate_objective = problem.objective(candidate, p)
            candidate_gradient = gradient_fn(candidate, p)
            if (not bool(torch.isfinite(candidate_objective))
                    or not bool(torch.isfinite(candidate_gradient).all())):
                raise ValueError("core-strict diagnostic objective/gradient nonfinite")
            objective_value = float(candidate_objective)
            gradient_max = float(candidate_gradient.abs().max())
            gradient_norm = float(torch.linalg.vector_norm(candidate_gradient))
            if not all(math.isfinite(value) for value in (
                objective_value, gradient_max, gradient_norm
            )):
                raise ValueError("core-strict diagnostic norm nonfinite")
            record.update(diagnostic_objective=objective_value,
                          diagnostic_gradient_max=gradient_max,
                          diagnostic_gradient_norm=gradient_norm)
            if not matches:
                raise ValueError("core-strict root full branch signature changed")
            record["status"] = "admitted_branch"
            return branch, scope
        except (ValueError, RuntimeError) as error:
            record.update(status="refused_branch", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            branch_calls.append(record)
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
                save()

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
                    branch_check=pinned_core_branch, max_iterations=8,
                    max_backtracks=16, pcg_max_iterations=104,
                )
            final_gradient = gradient_fn(refined.control, parameters)
            if (not bool(torch.isfinite(final_gradient).all())
                    or float(final_gradient.abs().max()) >= 1e-10):
                raise ValueError("core-strict final stationarity refused")
            report.update(
                phase="final_curvature",
                control=refined.control.detach().tolist(),
                objective=float(problem.objective(refined.control, parameters)),
                score=float(problem.score(refined.control, parameters)),
                gradient_max=float(final_gradient.abs().max()),
                gradient_norm=float(torch.linalg.vector_norm(final_gradient)),
                refinement_iterations=refined.iterations,
                refinement_hvp_count=refined.hvp_count,
                refinement_history=refined.history,
            )
            save()
            report["final_curvature"] = basin_probe._hessian_audit(
                problem, refined.control, parameters
            )
            final_branch, _ = pinned_core_branch(refined.control, parameters)
            slope = float(final_branch["minimum_scaled_slope_margin"])
            face = float(final_branch["minimum_scaled_face_flux_margin"])
            margin_qualified = slope > 1e-4 and face > 1e-4
            report.update(
                numerical_status=("core_strict_root_margin_qualified" if margin_qualified
                                  else "core_strict_root_low_margin"),
                phase="finished",
                response_margin_qualified=margin_qualified,
                final_branch={
                    "euler_stages": final_branch["euler_stages"],
                    "minimum_scaled_slope_margin": slope,
                    "minimum_scaled_face_flux_margin": face,
                    "signature_sha256": basin_probe._signature_sha(final_branch),
                },
            )
        except (RuntimeError, ValueError) as error:
            report.update(numerical_status="root_refused",
                          refusal=f"{type(error).__name__}: {error}")
    except ValueError as error:
        report.update(numerical_status="seed_curvature_refused",
                      refusal=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = preflight._input_identity(problem, warm, parameters, direction)
        report["seed_control_unchanged"] = preflight._tensor_sha(control) == previous.CONTROL_SHA256
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["preflight_plan_unchanged"] = _sha(preflight.PLAN) == preflight_plan_before
        report["prior_unchanged"] = _sha(PRIOR_REPORT) == PRIOR_SHA256
        report["preflight_unchanged"] = _sha(previous.PREFLIGHT) == previous.PREFLIGHT_SHA256
        report["attempt2_unchanged"] = _sha(previous.ATTEMPT2) == previous.ATTEMPT2_SHA256
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "seed_control_unchanged",
            "plan_unchanged", "preflight_plan_unchanged", "prior_unchanged",
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
    raise SystemExit(0 if result["numerical_status"].startswith("core_strict_root_") else 2)
