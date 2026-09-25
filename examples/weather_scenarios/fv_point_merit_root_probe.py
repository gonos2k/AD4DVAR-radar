"""One bounded gradient-merit sector-root diagnostic from the prior run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from advar import matrix_free
from advar.local_refinement import RefinementNumericalRefusal, RefinementTrial, refine_stationary
from examples.weather_scenarios import fv_point_basin_probe as basin
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios import fv_point_sector_root_probe as sector
from examples.weather_scenarios.fv_point_sector_policy import _digest, _key


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_MERIT_ROOT_PLAN.md"
PLAN_SHA256 = "c8a43e522cee2064b7ccf0144cdfab5d4bf52ff6f72d19dbd3ab6f9623769abf"
PRIOR = EVIDENCE / "point_sector_root_attempt1/point_sector_root.json"
PRIOR_SHA256 = "c32d148cd08523661f4838ceebd518f57c26e6b92f0ebcf25d6493a4c1658aab"
SEED_SHA256 = "17fe09298738471223846280f77095f7f3bcb96d9e1d527dacf71a62b0628a8f"
SEED_SIGNATURE_SHA256 = "ae447d2aeb384f7c39f7c97ba9b034fe3cdd6b7222f5d58d2b8587b2972b1da5"
SOURCE_PATHS = (*sector.SOURCE_PATHS,
                "examples/weather_scenarios/fv_point_merit_root_probe.py",
                "examples/weather_scenarios/fv_point_merit_root_runner.py")


class _NumericalGateRefusal(ValueError):
    """A declared pointwise curvature or final branch qualification failure."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def merit_trial_acceptance(
    trial: RefinementTrial,
    record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[bool, str] | None:
    """Use measured squared stationarity residual only after a sector switch."""
    current = _key(trial.current_branch)
    candidate = _key(trial.candidate_branch)
    if current == candidate:
        return None
    safe_norm = math.sqrt(torch.finfo(torch.float64).max)
    old_norm = trial.current_gradient_norm
    new_norm = trial.candidate_gradient_norm
    old = 0.5 * old_norm * old_norm if math.isfinite(old_norm) and abs(old_norm) <= safe_norm else math.inf
    new = 0.5 * new_norm * new_norm if math.isfinite(new_norm) and abs(new_norm) <= safe_norm else math.inf
    finite = math.isfinite(old) and math.isfinite(new)
    delta = old - new if finite else math.nan
    floor = (128 * torch.finfo(torch.float64).eps *
             max(abs(old), abs(new), torch.finfo(torch.float64).tiny)) if finite else math.nan
    accepted = finite and math.isfinite(delta) and math.isfinite(floor) and delta > floor
    reason = "merit_switch_decrease" if accepted else "merit_switch_refused"
    if record is not None:
        record({"iteration": trial.iteration, "backtrack": trial.backtrack,
                "step_scale": trial.step_scale,
                "current_signature_sha256": _digest(current),
                "candidate_signature_sha256": _digest(candidate),
                "old_merit": old if finite else None,
                "new_merit": new if finite else None,
                "delta": delta if finite and math.isfinite(delta) else None,
                "floor": floor if finite and math.isfinite(floor) else None,
                "current_objective": trial.current_objective,
                "candidate_objective": trial.candidate_objective,
                "current_gradient_max": trial.current_gradient_max,
                "candidate_gradient_max": trial.candidate_gradient_max,
                "accepted": accepted, "reason": reason})
    return accepted, reason


def run(output: Path) -> dict[str, Any]:
    source_before = _sources()
    plan_before, prior_before = _sha(PLAN), _sha(PRIOR)
    if plan_before != PLAN_SHA256 or prior_before != PRIOR_SHA256:
        raise ValueError("merit root plan or prior report changed")
    prior = json.loads(PRIOR.read_text())
    if (prior.get("numerical_status") != "sector_refused"
            or prior.get("response_validation") != "not_performed"
            or prior.get("source_before") != prior.get("source_after")
            or sector._sources() != prior.get("source_before")):
        raise ValueError("merit root prior source or status changed")
    accepted = [trial for trial in prior["trial_records"] if trial.get("accepted")]
    if len(accepted) != 6 or accepted[-1].get("iteration") != 6:
        raise ValueError("merit root seed trial is not the declared sixth acceptance")
    seed = torch.tensor(accepted[-1]["candidate_control"], dtype=torch.float64)
    if preflight._tensor_sha(seed) != SEED_SHA256:
        raise ValueError("merit root seed control changed")
    problem, warm, parameters, direction = preflight.fixed_problem()
    input_before = preflight._input_identity(problem, warm, parameters, direction)
    if input_before != prior.get("input_before") or prior.get("input_before") != prior.get("input_after"):
        raise ValueError("merit root fixed input changed")
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    seed_gradient = gradient_fn(seed, parameters)
    if (not bool(torch.isfinite(seed_gradient).all())
            or not math.isclose(float(seed_gradient.abs().max()), accepted[-1]["gradient_max"],
                                rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("merit root seed gradient changed")
    seed_branch, _ = sector.previous._core_branch(problem, seed, parameters)
    if basin._signature_sha(seed_branch) != SEED_SIGNATURE_SHA256:
        raise ValueError("merit root seed branch changed")

    started = time.monotonic()
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "seed_curvature", "numerical_status": "running",
        "response_validation": "not_performed", "plan_sha256": PLAN_SHA256,
        "prior_report_sha256": PRIOR_SHA256, "seed_control_sha256": SEED_SHA256,
        "seed_control": seed.detach().tolist(), "seed_gradient": seed_gradient.detach().tolist(),
        "seed_branch": {
            "signature_sha256": SEED_SIGNATURE_SHA256,
            "euler_stages": seed_branch["euler_stages"],
            "minimum_scaled_slope_margin": seed_branch["minimum_scaled_slope_margin"],
            "minimum_scaled_face_flux_margin": seed_branch["minimum_scaled_face_flux_margin"],
        },
        "source_before": source_before, "input_before": input_before,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "branch_calls": [], "trial_records": [], "policy_records": [], "linear_solves": [],
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def branch_check(control: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
        entry: dict[str, Any] = {"control_sha256": preflight._tensor_sha(control),
                                 "status": "trace_pending"}
        try:
            branch, scope = sector.previous._core_branch(problem, control, p)
            entry.update(status="core_branch_admitted",
                         signature_sha256=basin._signature_sha(branch),
                         euler_stages=branch["euler_stages"],
                         minimum_scaled_slope_margin=branch["minimum_scaled_slope_margin"],
                         minimum_scaled_face_flux_margin=branch["minimum_scaled_face_flux_margin"])
            return branch, scope
        except (ValueError, RuntimeError) as error:
            entry.update(status="core_branch_refused", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["branch_calls"].append(entry)
            save()

    def record_policy(entry: dict[str, Any]) -> None:
        report["policy_records"].append(entry)
        save()

    def record_trial(entry: dict[str, Any]) -> None:
        if "candidate_control" in entry:
            entry["candidate_control_sha256"] = preflight._tensor_sha(
                torch.tensor(entry["candidate_control"], dtype=torch.float64))
        branch = entry.pop("branch", None)
        if branch is not None:
            entry["branch_signature_sha256"] = basin._signature_sha(branch)
        report["trial_records"].append(entry)
        save()

    def monitor(original: Any) -> Any:
        def solve(operator: Any, rhs: Tensor, **kwargs: Any) -> Any:
            entry: dict[str, Any] = {"hvp_calls": 0, "rtol": kwargs.get("rtol"),
                                     "max_iterations": kwargs.get("max_iterations")}
            def measured(value: Tensor) -> Tensor:
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
                report["linear_solves"].append(entry)
                save()
        return solve

    def audit_curvature(value: Tensor) -> dict[str, float]:
        try:
            return basin._hessian_audit(problem, value, parameters)
        except ValueError as error:
            if not str(error).startswith("basin terminal exact "):
                raise
            raise _NumericalGateRefusal(str(error)) from error

    save()
    try:
        report["seed_curvature"] = audit_curvature(seed)
        report["phase"] = "merit_refinement"
        save()
        try:
            with matrix_free.observe_pcg_calls(monitor):
                refined = refine_stationary(
                    problem.objective, seed, parameters,
                    branch_check=branch_check,
                    trial_acceptance=lambda trial: merit_trial_acceptance(trial, record_policy),
                    trial_observer=record_trial,
                    max_iterations=8, max_backtracks=16, pcg_max_iterations=104,
                )
        except RefinementNumericalRefusal as error:
            report.update(numerical_status="merit_refused",
                          refusal=f"{type(error).__name__}: {error}")
        else:
            report["phase"] = "final_curvature"
            final_gradient = gradient_fn(refined.control, parameters)
            if (not bool(torch.isfinite(final_gradient).all())
                    or float(final_gradient.abs().max()) >= 1e-10):
                report.update(numerical_status="merit_refused",
                              refusal="final gradient failed stationarity gate")
            else:
                report["final_curvature"] = audit_curvature(refined.control)
                try:
                    final_branch, _ = branch_check(refined.control, parameters)
                except ValueError as error:
                    raise _NumericalGateRefusal(f"final branch refused: {error}") from error
                slope = float(final_branch["minimum_scaled_slope_margin"])
                face = float(final_branch["minimum_scaled_face_flux_margin"])
                high_margin = slope > 1e-4 and face > 1e-4
                report.update(
                    numerical_status="merit_root_margin_qualified" if high_margin else "merit_root_low_margin",
                    response_margin_qualified=high_margin,
                    control=refined.control.detach().tolist(),
                    final_control_sha256=preflight._tensor_sha(refined.control),
                    final_branch={"signature_sha256": basin._signature_sha(final_branch),
                                  "euler_stages": final_branch["euler_stages"],
                                  "minimum_scaled_slope_margin": slope,
                                  "minimum_scaled_face_flux_margin": face},
                    objective=float(problem.objective(refined.control, parameters)),
                    score=float(problem.score(refined.control, parameters)),
                    gradient_max=float(final_gradient.abs().max()),
                    gradient_norm=float(torch.linalg.vector_norm(final_gradient)),
                    refinement_iterations=refined.iterations,
                    refinement_hvp_count=refined.hvp_count,
                    phase="finished",
                )
    except _NumericalGateRefusal as error:
        if report["phase"] in ("seed_curvature", "final_curvature"):
            report.update(numerical_status="seed_curvature_refused" if report["phase"] == "seed_curvature"
                          else "merit_refused", refusal=f"{type(error).__name__}: {error}")
        else:
            raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = preflight._input_identity(problem, warm, parameters, direction)
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["prior_unchanged"] = _sha(PRIOR) == PRIOR_SHA256
        report["seed_control_unchanged"] = preflight._tensor_sha(seed) == SEED_SHA256
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "plan_unchanged",
            "prior_unchanged", "seed_control_unchanged")):
            report.update(numerical_status="identity_refused", phase="identity_recheck")
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.output)
    raise SystemExit(0 if result["numerical_status"].startswith("merit_root_") else 2)
