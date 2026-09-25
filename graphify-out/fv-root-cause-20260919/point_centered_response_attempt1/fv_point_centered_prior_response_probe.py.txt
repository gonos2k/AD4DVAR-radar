"""Bounded stationary response and signed reanalysis for a constructed point FV case."""
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

from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from advar.local_response import compute_local_response
from advar.matrix_free import pcg
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios.fv_point_centered_prior_case import (
    CenteredBranchRefusal, CenteredPointCase, make_case, tensor_sha,
)
from examples.weather_scenarios.fv_point_sector_policy import _digest, _key


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_CENTERED_PRIOR_RESPONSE_PLAN.md"
PLAN_SHA256 = "0e4469531dc050dc7c19933d035ad1063564260abac78b9c8e16d7d9395a72b3"
ARCHIVE = preflight.ARCHIVED_WARM
SOURCE_PATHS = tuple(dict.fromkeys((*preflight.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_core_strict_root_probe.py",
    "examples/weather_scenarios/fv_point_sector_policy.py",
    "examples/weather_scenarios/fv_point_centered_prior_case.py",
    "examples/weather_scenarios/fv_point_centered_prior_response_probe.py",
    "examples/weather_scenarios/fv_point_centered_prior_response_runner.py",
)))
STEP_SIZES = (0.001, 0.0005)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _branch_summary(branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "euler_stages": branch["euler_stages"],
        "signature_sha256": _digest(_key(branch)),
        "minimum_scaled_slope_margin": branch["minimum_scaled_slope_margin"],
        "minimum_scaled_face_flux_margin": branch["minimum_scaled_face_flux_margin"],
    }


def _finite_scalar(name: str, value: Tensor) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _projection_limit(direction_response: float) -> float:
    return max(1e-6 * abs(direction_response), 1e-12)


def run(output: Path) -> dict[str, Any]:
    source_before = _sources()
    plan_before, archive_before = _sha(PLAN), _sha(ARCHIVE)
    if plan_before != PLAN_SHA256:
        raise ValueError("centered point declared plan changed")
    case = make_case()
    input_before = case.identity
    control, p, direction = case.control, case.parameters, case.direction
    if (control.shape != (26,) or p.shape != (13,) or direction.shape != p.shape
            or control.dtype != torch.float64 or p.dtype != torch.float64
            or not torch.equal(direction.nonzero().flatten(), torch.tensor([4, 5, 6, 7]))):
        raise ValueError("centered point fixed input layout changed")
    started = time.monotonic()
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "nominal_audit", "numerical_status": "running",
        "response_validation": "not_performed", "plan_sha256": PLAN_SHA256,
        "archive_sha256": archive_before, "source_before": source_before,
        "input_before": input_before,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "scope": "constructed centered dynamics prior; full-valid correlated point FV; one local direction",
        "endpoints": [],
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def checked_branch(value: Tensor, parameters: Tensor) -> tuple[dict[str, Any], str]:
        branch, scope = case.branch_check(value, parameters)
        return branch, scope

    save()
    try:
        gradient_fn = torch.func.grad(case.objective, argnums=0)
        gradient = gradient_fn(control, p)
        branch, _ = checked_branch(control, p)
        nominal_objective = _finite_scalar("nominal objective", case.objective(control, p))
        nominal_score = _finite_scalar("nominal score", case.score(control, p))
        gradient_max = _finite_scalar("nominal gradient", gradient.abs().max())
        if gradient_max >= 1e-10:
            raise ValueError("centered point manufactured control is not stationary")
        # Dense columns are an independent small-problem audit, never the solve.
        columns = [torch.func.jvp(lambda c: gradient_fn(c, p), (control,), (basis,))[1]
                   for basis in torch.eye(control.numel(), dtype=control.dtype)]
        hessian = torch.stack(columns, dim=1)
        if not bool(torch.isfinite(hessian).all()):
            raise ValueError("centered point exact Hessian has nonfinite entries")
        hessian_scale = float(torch.linalg.matrix_norm(hessian))
        symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T)) / max(
            hessian_scale, torch.finfo(control.dtype).tiny)
        eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        eigen_min, eigen_max = float(eigenvalues[0]), float(eigenvalues[-1])
        if (not all(math.isfinite(v) for v in (symmetry, eigen_min, eigen_max))
                or symmetry > 1e-10 or eigen_min <= 0):
            raise ValueError("centered point exact Hessian failed local SPD gate")
        report["nominal"] = {
            "control": control.tolist(), "control_sha256": tensor_sha(control),
            "parameters": p.tolist(), "parameters_sha256": tensor_sha(p),
            "direction": direction.tolist(), "direction_sha256": tensor_sha(direction),
            "dynamics_prior_mean": case.dynamics_prior_mean.tolist(),
            "dynamics_prior_mean_sha256": tensor_sha(case.dynamics_prior_mean),
            "objective": nominal_objective, "score": nominal_score,
            "gradient_max": gradient_max, "gradient_norm": float(torch.linalg.vector_norm(gradient)),
            "branch": _branch_summary(branch),
            "hessian_audit": {"hvp_columns": len(columns), "symmetry_relative": symmetry,
                              "lambda_min": eigen_min, "lambda_max": eigen_max},
        }
        report["phase"] = "adjoint"
        save()

        tick = time.monotonic()
        response = compute_local_response(
            case.objective, case.score, control, p,
            {"middle_time_bias": direction},
            branch_check=checked_branch,
            input_identity=input_before,
        )
        response_time = time.monotonic() - tick
        directional = float(response.total["middle_time_bias"])
        projected = float(torch.dot(response.total_gradient, direction))
        projection_error = abs(directional - projected)
        projection_limit = _projection_limit(directional)
        if projection_error > projection_limit:
            raise ValueError("full VJP projection disagrees with directional JVP")
        report["response"] = {
            "direct_gradient": response.direct_gradient.tolist(),
            "indirect_gradient": response.indirect_gradient.tolist(),
            "total_gradient": response.total_gradient.tolist(),
            "direct": float(response.direct["middle_time_bias"]),
            "indirect": float(response.indirect["middle_time_bias"]),
            "total": directional, "total_projection": projected,
            "projection_absolute_error": projection_error,
            "projection_limit": projection_limit,
            "adjoint": response.adjoint.tolist(),
            "true_adjoint_relative_residual": response.true_adjoint_relative_residual,
            "pcg_relative_residual": response.pcg_relative_residual,
            "pcg_iterations": response.pcg_iterations,
            "hvp_count": response.hvp_count,
            "seconds": response_time,
        }
        report["phase"] = "tangent"
        save()

        cross = torch.func.jvp(lambda q: gradient_fn(control, q), (p,), (direction,))[1]
        tangent_rhs = -cross
        def hessian_vector(delta: Tensor) -> Tensor:
            return torch.func.jvp(lambda c: gradient_fn(c, p), (control,), (delta,))[1]
        tick = time.monotonic()
        tangent_solve = pcg(hessian_vector, tangent_rhs, rtol=1e-10, max_iterations=104)
        tangent = tangent_solve.solution
        residual = hessian_vector(tangent) - tangent_rhs
        rhs_norm = float(torch.linalg.vector_norm(tangent_rhs))
        relative_residual = float(torch.linalg.vector_norm(residual)) / rhs_norm if rhs_norm else float(torch.linalg.vector_norm(residual))
        if (not tangent_solve.converged or not bool(torch.isfinite(tangent).all())
                or not math.isfinite(relative_residual) or relative_residual > 1e-10):
            raise RuntimeError("centered point tangent failed true residual gate")
        report["tangent"] = {
            "control_direction": tangent.tolist(), "control_direction_sha256": tensor_sha(tangent),
            "rhs_sha256": tensor_sha(tangent_rhs),
            "iterations": tangent_solve.iterations,
            "pcg_relative_residual": tangent_solve.relative_residual,
            "true_relative_residual": relative_residual,
            "seconds": time.monotonic() - tick,
        }
        save()

        absolute_errors: list[float] = []
        for index, h in enumerate(STEP_SIZES):
            scores: dict[str, float] = {}
            for name, sign in (("plus", 1.0), ("minus", -1.0)):
                report["phase"] = f"endpoint_{index}_{name}"
                parameters = p + sign * h * direction
                predictor = control + sign * h * tangent
                endpoint: dict[str, Any] = {
                    "h": h, "sign": name, "status": "running",
                    "parameters_sha256": tensor_sha(parameters),
                    "predictor_sha256": tensor_sha(predictor),
                }
                report["endpoints"].append(endpoint)
                save()
                try:
                    checked_branch(predictor, parameters)
                except CenteredBranchRefusal as error:
                    endpoint.update(status="refused",
                                    refusal=f"{type(error).__name__}: {error}")
                    report.update(numerical_status="endpoint_refused")
                    save()
                    return report
                tick = time.monotonic()
                try:
                    refined = refine_stationary(
                        case.objective, predictor, parameters,
                        branch_check=checked_branch,
                        max_iterations=8, max_backtracks=16,
                        pcg_max_iterations=104,
                    )
                except RefinementNumericalRefusal as error:
                    endpoint.update(status="refused",
                                    refusal=f"{type(error).__name__}: {error}")
                    report.update(numerical_status="endpoint_refused")
                    save()
                    return report
                try:
                    final_branch, _ = checked_branch(refined.control, parameters)
                except CenteredBranchRefusal as error:
                    endpoint.update(status="refused",
                                    refusal=f"{type(error).__name__}: {error}")
                    report.update(numerical_status="endpoint_refused")
                    save()
                    return report
                final_gradient = gradient_fn(refined.control, parameters)
                final_max = _finite_scalar("endpoint gradient", final_gradient.abs().max())
                if final_max >= 1e-10:
                    raise ValueError("centered point endpoint failed fresh stationarity gate")
                score = _finite_scalar("endpoint score", case.score(refined.control, parameters))
                endpoint.update(
                    status="eligible", control=refined.control.tolist(),
                    control_sha256=tensor_sha(refined.control),
                    objective=_finite_scalar("endpoint objective", case.objective(refined.control, parameters)),
                    score=score, gradient_max=final_max,
                    branch=_branch_summary(final_branch),
                    refinement_iterations=refined.iterations,
                    refinement_hvp_count=refined.hvp_count,
                    refinement_pcg_iterations=[item["pcg_iterations"] for item in refined.history if "pcg_iterations" in item],
                    seconds=time.monotonic() - tick,
                )
                scores[name] = score
                save()
            central = (scores["plus"] - scores["minus"]) / (2 * h)
            absolute_error = abs(central - directional)
            limit = 1e-4 * abs(directional) if abs(directional) >= 1e-8 else 1e-10
            report.setdefault("pairs", []).append({
                "h": h, "central_difference": central,
                "adjoint_directional": directional,
                "absolute_error": absolute_error,
                "relative_error": absolute_error / abs(directional) if directional else None,
                "limit": limit, "passed": absolute_error < limit,
            })
            absolute_errors.append(absolute_error)
            save()
        passed = (all(pair["passed"] for pair in report["pairs"])
                  and absolute_errors[1] < absolute_errors[0])
        report.update(numerical_status="eligible" if passed else "validation_failed",
                      response_validation="passed" if passed else "failed",
                      phase="finished")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = case.identity
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["archive_unchanged"] = _sha(ARCHIVE) == archive_before
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "plan_unchanged", "archive_unchanged"
        )):
            report.update(numerical_status="identity_refused", phase="identity_recheck",
                          response_validation="not_performed")
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.output)
    raise SystemExit(0 if result["numerical_status"] == "eligible" else 2)
