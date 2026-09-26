"""One bounded current-source sector-aware root search for two missing FV cells."""
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

from advar import matrix_free, variational
from advar.local_refinement import (RefinementCallbackError,
                                    RefinementNumericalRefusal, refine_stationary)
from examples.weather_scenarios import fv_partial_reanalysis_preflight as preflight
from examples.weather_scenarios import fv_point_basin_probe as basin
from examples.weather_scenarios.fv_point_merit_root_probe import merit_trial_acceptance
from examples.weather_scenarios.fv_point_sector_policy import _digest, _key
from tests.test_fv_research_partial_observation import _problem


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_PARTIAL_SECTOR_ROOT_PLAN.md"
PLAN_SHA256 = "a92b977ea481fb0abcc43a8bd45aa14ded45e2744450ed7a52023ba56aa24560"
HISTORICAL_PREFLIGHT = EVIDENCE / "partial_branch_gate_detail_attempt3/fv_partial_reanalysis.preflight.json"
HISTORICAL_PREFLIGHT_SHA256 = "4a2fd9c57e6f271e0505e6ffb9bb7bc9bca66fff724ce0d6e885df90d7a7657b"
HISTORICAL_MANIFEST = EVIDENCE / "fv_partial_branch_gate_detail_run3_manifest.json"
HISTORICAL_MANIFEST_SHA256 = "6d859284dbb737910e2a94252e37cf87322924cae2fa9e88381f9e29dd18a5a3"
CURRENT_PROBLEM_SHA256 = "838fcf77a43b209b53dfe62f6a1657d0f8eba5a41797e431451059f3f8d18158"
HISTORICAL_PROBLEM_SHA256 = "ffc5a82cb9e4cda7667378b1932c6703e61a6a0c2d16d37ca1c6268b2f860e48"
SOURCE_PATHS = tuple(dict.fromkeys((
    "examples/weather_scenarios/fv_partial_reanalysis_preflight.py",
    "examples/weather_scenarios/fv_partial_reanalysis_probe.py",
    "examples/weather_scenarios/fv_minmod_inverse_probe.py",
    "examples/weather_scenarios/fv86_resource_runner.py",
    "examples/weather_scenarios/fv_point_basin_probe.py",
    "examples/weather_scenarios/fv_point_merit_root_probe.py",
    "examples/weather_scenarios/fv_point_sector_policy.py",
    "examples/weather_scenarios/fv_partial_sector_root_probe.py",
    "examples/weather_scenarios/fv_partial_sector_root_runner.py",
    "tests/test_fv_research_partial_observation.py",
    "src/advar/fv_research_problem.py",
    "src/advar/local_refinement.py",
    "src/advar/local_response.py",
    "src/advar/matrix_free.py",
    "src/advar/variational.py",
    "src/advar/transport.py",
    "src/advar/physics.py",
    "src/advar/nowcast.py",
)))
COMPARED_PREFLIGHT_FIELDS = (
    "observation_counts", "missing_indices", "common_bias_std_dbz",
    "verification_constructor", "tensor_sha256", "warm_start_gradient_max",
    "warm_start_objective", "warm_start_branch",
)


class _NumericalGateRefusal(ValueError):
    """Known seed/final exact-curvature or branch qualification refusal."""


_KNOWN_BRANCH_REFUSALS = frozenset({
    "minmod joint oracle left its strict smooth branch",
    "partial sector core margin resolvability refused",
})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _preflight_identity() -> dict[str, Any]:
    if (_sha(HISTORICAL_PREFLIGHT) != HISTORICAL_PREFLIGHT_SHA256
            or _sha(HISTORICAL_MANIFEST) != HISTORICAL_MANIFEST_SHA256):
        raise ValueError("historical two-hole preflight or manifest changed")
    archived = json.loads(HISTORICAL_PREFLIGHT.read_text())
    current = preflight.run()
    if (archived["problem_identity"]["fixed_problem_sha256"] != HISTORICAL_PROBLEM_SHA256
            or current["problem_identity"]["fixed_problem_sha256"] != CURRENT_PROBLEM_SHA256
            or any(archived[name] != current[name] for name in COMPARED_PREFLIGHT_FIELDS)
            or current.get("status") != "preflight_only"
            or current.get("numerical_solver_runs") != 0):
        raise ValueError("current-source two-hole input differs from declared fixed tensors")
    return {
        "historical_problem_identity": archived["problem_identity"],
        "current_problem_identity": current["problem_identity"],
        "fixed_input_fields": {name: current[name] for name in COMPARED_PREFLIGHT_FIELDS},
        "historical_preflight_sha256": HISTORICAL_PREFLIGHT_SHA256,
        "historical_manifest_sha256": HISTORICAL_MANIFEST_SHA256,
    }


def _branch_summary(branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "euler_stages": branch["euler_stages"],
        "minimum_scaled_slope_margin": branch["minimum_scaled_slope_margin"],
        "minimum_scaled_face_flux_margin": branch["minimum_scaled_face_flux_margin"],
        "signature_sha256": _digest(_key(branch)),
    }


def _hessian_audit(problem: Any, control: Tensor, parameters: Tensor) -> dict[str, float]:
    try:
        return basin._hessian_audit(problem, control, parameters)
    except ValueError as error:
        if not str(error).startswith("basin terminal exact "):
            raise
        raise _NumericalGateRefusal(str(error)) from error


def run(output: Path) -> dict[str, Any]:
    source_before, plan_before = _sources(), _sha(PLAN)
    if plan_before != PLAN_SHA256:
        raise ValueError("partial sector root plan changed")
    input_before = _preflight_identity()
    problem, warm, parameters = _problem()
    if problem.identity != input_before["current_problem_identity"]:
        raise ValueError("partial sector current problem identity changed")
    started = time.monotonic()
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "gauss_newton", "numerical_status": "running",
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "plan_sha256": PLAN_SHA256, "source_before": source_before,
        "input_before": input_before, "warm_control_sha256": _tensor_sha(warm),
        "parameters_sha256": _tensor_sha(parameters),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "scope": "one current-source two-hole partial FV nominal root search; no adjoint/reanalysis",
        "branch_calls": [], "trial_records": [], "policy_records": [], "linear_solves": [],
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def branch_check(control: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
        entry: dict[str, Any] = {"control_sha256": _tensor_sha(control),
                                 "status": "trace_pending"}
        try:
            branch, scope, face = preflight.branch_with_face_margin(problem, control, p)
            slope = float(branch["minimum_scaled_slope_margin"])
            if branch["euler_stages"] != 54:
                raise RuntimeError("partial sector core RK stage count changed")
            if not math.isfinite(slope) or not math.isfinite(face) or slope <= 0 or face <= 0:
                raise ValueError("partial sector core margin resolvability refused")
            complete = {**branch, "minimum_scaled_face_flux_margin": face}
            entry.update(status="core_branch_admitted", **_branch_summary(complete))
            return complete, scope
        except ValueError as error:
            if str(error) not in _KNOWN_BRANCH_REFUSALS:
                entry.update(status="callback_error",
                             reason=f"{type(error).__name__}: {error}")
                raise RefinementCallbackError(
                    f"unexpected branch oracle ValueError: {error}") from error
            entry.update(status="core_branch_refused",
                         reason=f"{type(error).__name__}: {error}")
            raise
        except RuntimeError as error:
            entry.update(status="callback_error",
                         reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["branch_calls"].append(entry)
            save()

    def record_policy(entry: dict[str, Any]) -> None:
        report["policy_records"].append(entry)
        save()

    def record_trial(entry: dict[str, Any]) -> None:
        if "candidate_control" in entry:
            candidate = torch.tensor(entry["candidate_control"], dtype=torch.float64)
            entry["candidate_control_sha256"] = _tensor_sha(candidate)
            if not bool(torch.isfinite(candidate).all()):
                entry["candidate_control"] = None
                entry["candidate_nonfinite"] = True
        branch = entry.pop("branch", None)
        if branch is not None:
            entry["branch"] = _branch_summary(branch)
        report["trial_records"].append(entry)
        save()

    def monitor(original: Any) -> Any:
        def solve(operator: Any, rhs: Tensor, **kwargs: Any) -> Any:
            entry: dict[str, Any] = {"phase": report["phase"],
                                     "hvp_calls": 0, "rtol": kwargs.get("rtol"),
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

    save()
    try:
        gn_tick = time.monotonic()
        with matrix_free.observe_pcg_calls(monitor):
            gn = variational.solve_analysis(
                problem.observations, problem.contract(parameters), control=warm)
        seed = gn.control.detach().clone()
        report["gauss_newton"] = {
            "reason": gn.reason, "outer_iterations": gn.outer_iterations,
            "pcg_iterations": gn.pcg_iterations,
            "initial_objective": gn.initial_objective,
            "final_objective": gn.final_objective,
            "control": seed.tolist(), "control_sha256": _tensor_sha(seed),
            "seconds": time.monotonic() - gn_tick,
        }
        seed_gradient = torch.func.grad(problem.objective, argnums=0)(seed, parameters)
        report["seed_gradient_norm"] = float(torch.linalg.vector_norm(seed_gradient))
        report["seed_gradient_max"] = float(seed_gradient.abs().max())
        report["phase"] = "seed_curvature"
        save()
        report["seed_branch"] = _branch_summary(branch_check(seed, parameters)[0])
        report["seed_curvature"] = _hessian_audit(problem, seed, parameters)
        report["seed_curvature_control_sha256"] = _tensor_sha(seed)
        report["phase"] = "sector_refinement"
        save()
        try:
            with matrix_free.observe_pcg_calls(monitor):
                refined = refine_stationary(
                    problem.objective, seed, parameters,
                    branch_check=branch_check,
                    trial_acceptance=lambda trial: merit_trial_acceptance(trial, record_policy),
                    trial_observer=record_trial,
                    max_iterations=8, max_backtracks=16,
                    pcg_max_iterations=104,
                )
        except RefinementNumericalRefusal as error:
            report.update(numerical_status="root_refused",
                          refusal=f"{type(error).__name__}: {error}")
        else:
            report["phase"] = "final_curvature"
            gradient = torch.func.grad(problem.objective, argnums=0)(refined.control, parameters)
            maximum = float(gradient.abs().max())
            if not math.isfinite(maximum) or maximum >= 1e-10:
                report.update(numerical_status="root_refused",
                              refusal="final gradient failed stationarity gate")
            else:
                report["final_curvature"] = _hessian_audit(problem, refined.control, parameters)
                report["final_curvature_control_sha256"] = _tensor_sha(refined.control)
                final_branch, _ = branch_check(refined.control, parameters)
                margin_qualified = (
                    final_branch["minimum_scaled_slope_margin"] > 1e-4
                    and final_branch["minimum_scaled_face_flux_margin"] > 1e-4
                )
                report.update(
                    phase="finished",
                    numerical_status="root_margin_qualified" if margin_qualified else "root_low_margin",
                    response_margin_qualified=margin_qualified,
                    control=refined.control.tolist(),
                    final_control_sha256=_tensor_sha(refined.control),
                    final_branch=_branch_summary(final_branch),
                    final_gradient_max=maximum,
                    final_objective=float(problem.objective(refined.control, parameters)),
                    final_score=float(problem.score(refined.control, parameters)),
                    refinement_iterations=refined.iterations,
                    refinement_hvp_count=refined.hvp_count,
                )
    except _NumericalGateRefusal as error:
        report.update(numerical_status=("seed_curvature_refused"
                                        if report["phase"] == "seed_curvature"
                                        else "root_refused"),
                      refusal=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = _preflight_identity()
        report["warm_control_unchanged"] = _tensor_sha(warm) == report["warm_control_sha256"]
        report["parameters_unchanged"] = _tensor_sha(parameters) == report["parameters_sha256"]
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        if not all(report[key] for key in (
            "warm_control_unchanged", "parameters_unchanged", "source_unchanged",
            "input_unchanged", "plan_unchanged",
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
    raise SystemExit(0 if result["numerical_status"] in (
        "root_margin_qualified", "root_low_margin") else 2)
