"""One measured-endpoint branch-adaptive point-root feasibility run."""
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
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from examples.weather_scenarios import fv_point_basin_probe as basin_probe
from examples.weather_scenarios import fv_point_core_strict_root_probe as previous
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios.fv_point_sector_policy import sector_trial_acceptance


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_SECTOR_ROOT_PLAN.md"
PLAN_SHA256 = "79495279024f2140a91885f615e718a77b9f372a03f3233d9ebabf7293679eac"
PRIOR = EVIDENCE / "point_core_strict_root_attempt1/point_core_strict_root.json"
PRIOR_SHA256 = "95e14c120d37c816aceed4f7ed0b98bb406549b5e14e1fd967aff133b8814a76"
SOURCE_PATHS = tuple(dict.fromkeys((*previous.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_sector_policy.py",
    "examples/weather_scenarios/fv_point_sector_root_probe.py",
    "examples/weather_scenarios/fv_point_sector_root_runner.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def run(output: Path) -> dict[str, Any]:
    source_before = _sources()
    plan_before = _sha(PLAN)
    preflight_plan_before = _sha(preflight.PLAN)
    previous_plan_before = _sha(previous.PLAN)
    if plan_before != PLAN_SHA256:
        raise ValueError("sector root declared plan changed")
    if (_sha(PRIOR) != PRIOR_SHA256
            or _sha(previous.previous.PREFLIGHT) != previous.previous.PREFLIGHT_SHA256
            or _sha(previous.previous.ATTEMPT2) != previous.previous.ATTEMPT2_SHA256):
        raise ValueError("sector root pinned report changed")
    prior = json.loads(PRIOR.read_text())
    pinned_preflight = json.loads(previous.previous.PREFLIGHT.read_text())
    if (prior.get("numerical_status") != "root_refused"
            or prior.get("response_validation") != "not_performed"
            or prior.get("seed_control_sha256") != previous.previous.CONTROL_SHA256
            or pinned_preflight.get("plan_sha256") != preflight_plan_before
            or prior.get("plan_sha256") != previous_plan_before
            or prior.get("plan_sha256") != previous_plan_before):
        raise ValueError("sector root prior status or plan changed")
    # The additive trial hook intentionally changes this one source file;
    # every other derivative-critical source must still match the prior run.
    changed_source = "src/advar/local_refinement.py"
    if any(prior["source_before"].get(name) != source_before[name]
           for name in previous.SOURCE_PATHS if name != changed_source):
        raise ValueError("sector root unrelated source changed from prior run")
    problem, warm, parameters, direction = preflight.fixed_problem()
    input_before = preflight._input_identity(problem, warm, parameters, direction)
    if prior.get("input_before") != input_before:
        raise ValueError("sector root fixed inputs changed")
    control = torch.tensor(prior["seed_control"], dtype=torch.float64)
    if (control.shape != warm.shape
            or preflight._tensor_sha(control) != previous.previous.CONTROL_SHA256):
        raise ValueError("sector root seed control changed")
    seed_branch, _ = previous._core_branch(problem, control, parameters)
    if basin_probe._signature_sha(seed_branch) != previous.previous.BRANCH_SHA256:
        raise ValueError("sector root seed signature changed")
    prior_gradient = torch.tensor(prior["seed_gradient"], dtype=torch.float64)
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    seed_gradient = gradient_fn(control, parameters)
    if (prior_gradient.shape != seed_gradient.shape
            or not torch.allclose(prior_gradient, seed_gradient,
                                  rtol=1e-10, atol=1e-12)):
        raise ValueError("sector root seed gradient changed")

    started = time.monotonic()
    branch_calls: list[dict[str, Any]] = []
    trial_records: list[dict[str, Any]] = []
    policy_records: list[dict[str, Any]] = []
    linear_solves: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "seed_curvature",
        "numerical_status": "running", "response_validation": "not_performed",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "plan_sha256": PLAN_SHA256, "prior_report_sha256": PRIOR_SHA256,
        "previous_plan_sha256": previous_plan_before,
        "preflight_plan_sha256": preflight_plan_before,
        "preflight_sha256": previous.previous.PREFLIGHT_SHA256,
        "attempt2_sha256": previous.previous.ATTEMPT2_SHA256,
        "source_before": source_before, "input_before": input_before,
        "seed_control": control.detach().tolist(),
        "seed_control_sha256": previous.previous.CONTROL_SHA256,
        "seed_gradient": seed_gradient.detach().tolist(),
        "seed_branch": {
            "euler_stages": seed_branch["euler_stages"],
            "minimum_scaled_slope_margin": seed_branch["minimum_scaled_slope_margin"],
            "minimum_scaled_face_flux_margin": seed_branch["minimum_scaled_face_flux_margin"],
            "signature_sha256": basin_probe._signature_sha(seed_branch),
        },
        "branch_calls": branch_calls, "trial_records": trial_records,
        "policy_records": policy_records, "linear_solves": linear_solves,
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def any_core_branch(candidate: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
        record: dict[str, Any] = {
            "control_sha256": preflight._tensor_sha(candidate),
            "status": "trace_pending",
        }
        try:
            branch, scope = previous._core_branch(problem, candidate, p)
            record.update(
                status="core_branch_admitted",
                euler_stages=branch["euler_stages"],
                minimum_scaled_slope_margin=branch["minimum_scaled_slope_margin"],
                minimum_scaled_face_flux_margin=branch["minimum_scaled_face_flux_margin"],
                signature_sha256=basin_probe._signature_sha(branch),
            )
            return branch, scope
        except (ValueError, RuntimeError) as error:
            record.update(status="core_branch_refused",
                          reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            branch_calls.append(record)
            save()

    def record_policy(event: dict[str, Any]) -> None:
        policy_records.append(event)
        save()

    def record_trial(event: dict[str, Any]) -> None:
        if "candidate_control" in event:
            event["candidate_control_sha256"] = preflight._tensor_sha(
                torch.tensor(event["candidate_control"], dtype=torch.float64)
            )
        branch = event.pop("branch", None)
        if branch is not None:
            event["branch_signature_sha256"] = basin_probe._signature_sha(branch)
            if event["accepted"]:
                event["accepted_branch_signature"] = basin_probe._signature(branch)
        trial_records.append(event)
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
        report["phase"] = "sector_refinement"
        save()
        try:
            with matrix_free.observe_pcg_calls(monitor):
                refined = refine_stationary(
                    problem.objective, control, parameters,
                    branch_check=any_core_branch,
                    trial_acceptance=lambda trial: sector_trial_acceptance(trial, record_policy),
                    trial_observer=record_trial,
                    max_iterations=8, max_backtracks=16, pcg_max_iterations=104,
                )
            report["phase"] = "final_curvature"
            gradient = gradient_fn(refined.control, parameters)
            if (not bool(torch.isfinite(gradient).all())
                    or float(gradient.abs().max()) >= 1e-10):
                raise ValueError("sector root final stationarity refused")
            report.update(
                phase="final_curvature",
                control=refined.control.detach().tolist(),
                objective=float(problem.objective(refined.control, parameters)),
                score=float(problem.score(refined.control, parameters)),
                gradient_max=float(gradient.abs().max()),
                gradient_norm=float(torch.linalg.vector_norm(gradient)),
                refinement_iterations=refined.iterations,
                refinement_hvp_count=refined.hvp_count,
                refinement_history=refined.history,
            )
            save()
            report["final_curvature"] = basin_probe._hessian_audit(
                problem, refined.control, parameters
            )
            final_branch, _ = any_core_branch(refined.control, parameters)
            slope = float(final_branch["minimum_scaled_slope_margin"])
            face = float(final_branch["minimum_scaled_face_flux_margin"])
            high_margin = slope > 1e-4 and face > 1e-4
            report.update(
                numerical_status=("sector_root_margin_qualified" if high_margin
                                  else "sector_root_low_margin"),
                response_margin_qualified=high_margin,
                phase="finished",
                final_control_sha256=preflight._tensor_sha(refined.control),
                final_branch={
                    "euler_stages": final_branch["euler_stages"],
                    "minimum_scaled_slope_margin": slope,
                    "minimum_scaled_face_flux_margin": face,
                    "signature_sha256": basin_probe._signature_sha(final_branch),
                },
            )
        except RefinementNumericalRefusal as error:
            report.update(numerical_status="sector_refused",
                          refusal=f"{type(error).__name__}: {error}")
    except ValueError as error:
        if report["phase"] == "seed_curvature":
            report.update(numerical_status="seed_curvature_refused",
                          refusal=f"{type(error).__name__}: {error}")
        elif report["phase"] == "final_curvature":
            report.update(numerical_status="sector_refused",
                          refusal=f"{type(error).__name__}: {error}")
        else:
            raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = preflight._input_identity(problem, warm, parameters, direction)
        report["seed_control_unchanged"] = preflight._tensor_sha(control) == previous.previous.CONTROL_SHA256
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["prior_unchanged"] = _sha(PRIOR) == PRIOR_SHA256
        report["previous_plan_unchanged"] = _sha(previous.PLAN) == previous_plan_before
        report["preflight_plan_unchanged"] = _sha(preflight.PLAN) == preflight_plan_before
        report["preflight_unchanged"] = _sha(previous.previous.PREFLIGHT) == previous.previous.PREFLIGHT_SHA256
        report["attempt2_unchanged"] = _sha(previous.previous.ATTEMPT2) == previous.previous.ATTEMPT2_SHA256
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "seed_control_unchanged",
            "plan_unchanged", "prior_unchanged", "previous_plan_unchanged",
            "preflight_plan_unchanged", "preflight_unchanged", "attempt2_unchanged",
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
    raise SystemExit(0 if result["numerical_status"].startswith("sector_root_") else 2)
