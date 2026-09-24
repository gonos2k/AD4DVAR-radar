"""Bounded G3b partial-observation stationary-response experiment.

The probe is intentionally serial and writes an atomic JSON checkpoint after
each phase and signed endpoint. Run it under ``fv86_resource_runner.run_guarded``
for the separately declared 1200-second / 1-GiB resource boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import local_refinement, local_response, matrix_free, variational
from examples.weather_scenarios import fv_partial_reanalysis_preflight as preflight
from tests.test_fv_research_partial_observation import _problem


PREDECLARED_STEPS = tuple(0.001 * (2.0 ** -index) for index in range(6))
STATIONARITY_TOLERANCE = 1.0e-10
LINEAR_RELATIVE_TOLERANCE = 1.0e-10
MINIMUM_BRANCH_MARGIN = 1.0e-4
MINIMUM_CONSECUTIVE_PAIRS = 2
MAX_REFINEMENT_ITERATIONS = 8
MAX_BACKTRACKS = 16
PCG_MAX_ITERATIONS = 104
SIGNAL_EPS_MULTIPLIER = 1024.0


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _serializable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _branch_signature(branch: dict[str, Any]) -> dict[str, Any]:
    return {"choices": branch["choices"], "face_signs": branch["face_signs"]}


def _branch_summary(branch: dict[str, Any], face_margin: float) -> dict[str, Any]:
    return {
        "euler_stages": branch["euler_stages"],
        "minimum_scaled_slope_margin": branch["minimum_scaled_slope_margin"],
        "minimum_scaled_face_flux_margin": face_margin,
        "signature_sha256": _json_digest(_branch_signature(branch)),
    }


def _failure_category(message: str, phase: str) -> str:
    lowered = message.lower()
    if "face" in lowered and "margin" in lowered:
        return "face_flux_margin_refusal"
    if "slope margin" in lowered or "branch margin" in lowered:
        return "slope_margin_refusal"
    if "branch" in lowered or "strict positive growth" in lowered:
        return "branch_refusal"
    if "positive definite" in lowered or "positive curvature" in lowered:
        return "nonpositive_curvature"
    if "response is zero" in lowered or "directional signal" in lowered:
        return "zero_directional_signal"
    if "pcg" in lowered and ("converg" in lowered or "budget" in lowered):
        return "linear_solve_refusal"
    if "residual" in lowered:
        return "linear_residual_refusal"
    if "armijo" in lowered or "descent direction" in lowered:
        return "refinement_line_search_refusal"
    if "stationary" in lowered or "iteration budget" in lowered:
        return "stationarity_refusal"
    if "cfl" in lowered:
        return "cfl_refusal"
    if phase == "preflight":
        return "preflight_identity_refusal"
    return "numerical_refusal"


def _require_close_identity(pinned: dict[str, Any], current: dict[str, Any]) -> None:
    """Match the pinned inputs and sources; commit IDs are provenance only."""
    pinned_identity = {key: value for key, value in pinned.items() if key != "base_commit"}
    current_identity = {key: value for key, value in current.items() if key != "base_commit"}
    if pinned_identity != current_identity:
        differing = sorted(
            key for key in set(pinned_identity) | set(current_identity)
            if pinned_identity.get(key) != current_identity.get(key)
        )
        raise ValueError(f"preflight identity changed: {differing}")


def _check_sources(expected: dict[str, str]) -> dict[str, str]:
    actual = {name: _file_digest(ROOT / name) for name in sorted(expected)}
    mismatches = [name for name, digest in expected.items() if actual.get(name) != digest]
    if mismatches:
        raise ValueError(f"pinned source SHA256 mismatch: {mismatches}")
    return actual


def _fresh_preflight(pinned: dict[str, Any]) -> dict[str, Any]:
    current = preflight.run()
    _require_close_identity(pinned, current)
    if current.get("status") != "preflight_only" or current.get("numerical_solver_runs") != 0:
        raise ValueError("preflight did not remain a no-solver check")
    _check_sources(current["source_sha256"])
    return current


def _make_direction(problem: Any, parameters: torch.Tensor) -> torch.Tensor:
    direction = torch.zeros_like(parameters)
    direction[20:40] = problem.observations.valid_mask[1].to(parameters.dtype).flatten()
    if int(torch.count_nonzero(direction)) != 19:
        raise ValueError("valid-only middle-time direction changed")
    missing = problem.observations.missing_mask.flatten()
    if bool((direction[:-1][missing] != 0).any()) or bool(direction[-1] != 0):
        raise ValueError("direction activates a missing observation or theta")
    return direction


def score_signal_gate(step: float, response: float, nominal_score: float) -> tuple[float, float, bool]:
    """Check that the predicted signed score separation clears roundoff."""
    separation = 2.0 * step * abs(response)
    floor = SIGNAL_EPS_MULTIPLIER * torch.finfo(torch.float64).eps * max(
        abs(nominal_score), torch.finfo(torch.float64).tiny
    )
    return separation, floor, separation > floor


def central_response_error(
    score_minus: float, score_plus: float, step: float, response: float,
) -> tuple[float, float]:
    """Return central score slope and error scaled by the nonzero response."""
    if not math.isfinite(response) or response == 0.0:
        raise ValueError("directional signal must be finite and nonzero")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    central = (score_plus - score_minus) / (2.0 * step)
    return central, abs(central - response) / abs(response)


def _branch_with_gates(
    problem: Any,
    control: torch.Tensor,
    parameters: torch.Tensor,
    expected_signature: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, float]:
    branch, scope, face_margin = preflight.branch_with_face_margin(
        problem, control, parameters
    )
    if branch["euler_stages"] != 54:
        raise ValueError("partial branch did not inspect all 54 Euler stages")
    if branch["minimum_scaled_slope_margin"] <= MINIMUM_BRANCH_MARGIN:
        raise ValueError("minimum slope margin is at or below 1e-4")
    if face_margin <= MINIMUM_BRANCH_MARGIN:
        raise ValueError("minimum face flux margin is at or below 1e-4")
    if expected_signature is not None and _branch_signature(branch) != expected_signature:
        raise ValueError("partial branch signature changed")
    return branch, scope, face_margin


def _monitor_pcg(
    original: Callable[..., Any], report: dict[str, Any], phase_name: str
) -> Callable[..., Any]:
    def solve(operator: Callable[[torch.Tensor], torch.Tensor], rhs: torch.Tensor, **kwargs: Any) -> Any:
        entry: dict[str, Any] = {
            "phase": phase_name,
            "rtol": kwargs.get("rtol"),
            "max_iterations": kwargs.get("max_iterations"),
            "hvp_calls": 0,
        }

        def measured(direction: torch.Tensor) -> torch.Tensor:
            result = operator(direction)
            entry["hvp_calls"] += 1
            return result

        tick = time.monotonic()
        try:
            result = original(measured, rhs, **kwargs)
            entry.update(
                converged=result.converged,
                iterations=result.iterations,
                pcg_relative_residual=result.relative_residual,
            )
            return result
        except (ValueError, RuntimeError) as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            entry["seconds"] = time.monotonic() - tick
            report["linear_solves"].append(entry)

    return solve


def run(output: Path, preflight_path: Path, *, stop_after_nominal: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        pinned = json.loads(preflight_path.read_text())
        preflight_file_hash = _file_digest(preflight_path)
        initial_preflight = _fresh_preflight(pinned)
        driver_relative = str(Path(__file__).resolve().relative_to(ROOT))
        driver_hash = _file_digest(Path(__file__).resolve())
        if pinned["source_sha256"].get(driver_relative) != driver_hash:
            raise ValueError("pinned preflight does not contain this numerical driver SHA256")
    except Exception as error:
        failure = {
            "execution_status": "error",
            "numerical_status": "not_started",
            "response_validation": "not_established",
            "phase": "preflight_identity",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.monotonic() - started,
        }
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        temp.replace(output)
        raise
    problem, warm_start, parameters = _problem()
    runtime_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    objective, score, _ = problem.functions()
    direction = _make_direction(problem, parameters)

    source_hashes = dict(pinned["source_sha256"])
    source_hashes[driver_relative] = driver_hash
    input_identity = {
        "fixed_problem_sha256": problem.identity["fixed_problem_sha256"],
        "preflight_tensor_sha256": pinned["tensor_sha256"],
        "warm_start_sha256": _tensor_digest(warm_start),
        "direction_sha256": _tensor_digest(direction),
        "step_sizes": list(PREDECLARED_STEPS),
    }
    report: dict[str, Any] = {
        "execution_status": "running",
        "numerical_status": "not_started",
        "response_validation": "not_established",
        "phase": "preflight_identity",
        "scope": "one-lead 4x5 / 26-control fixed-mask partial-observation local response",
        "stop_after_nominal": stop_after_nominal,
        "pinned_source_commit": pinned["base_commit"],
        "runtime_commit": runtime_commit,
        "environment": pinned["environment"],
        "preflight_path": str(preflight_path),
        "preflight_report_sha256": preflight_file_hash,
        "preflight_identity": initial_preflight,
        "source_sha256": source_hashes,
        "input_identity": input_identity,
        "direction": {
            "name": "middle_time_valid_bias",
            "units": "dBZ per unit step",
            "values": direction.tolist(),
        },
        "step_sizes": list(PREDECLARED_STEPS),
        "gates": {
            "stationarity_max_gradient": STATIONARITY_TOLERANCE,
            "tangent_actual_relative_residual": LINEAR_RELATIVE_TOLERANCE,
            "adjoint_actual_relative_residual": LINEAR_RELATIVE_TOLERANCE,
            "minimum_scaled_slope_margin": MINIMUM_BRANCH_MARGIN,
            "minimum_scaled_face_flux_margin": MINIMUM_BRANCH_MARGIN,
            "score_signal_epsilon_multiplier": SIGNAL_EPS_MULTIPLIER,
            "central_response_relative_error": 1.0e-4,
            "consecutive_valid_pairs": MINIMUM_CONSECUTIVE_PAIRS,
        },
        "timings": {},
        "linear_solves": [],
        "endpoints": [],
        "pairs": [],
        "nonlinear_reanalyses": 0,
        "nominal_branch_policy": "pin the fresh GN seed's strict branch through nominal Newton refinement; refuse if it changes",
        "nominal_eligibility": "not_attempted",
        "endpoint_eligibility": "not_attempted",
        "finite_path_certified": False,
        "general_minmod_response_eligible": False,
    }
    def checkpoint(phase_name: str | None = None) -> None:
        if phase_name is not None:
            report["phase"] = phase_name
        report["elapsed_seconds"] = time.monotonic() - started
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(output)

    def require_unchanged() -> None:
        _check_sources(pinned["source_sha256"])
        if _file_digest(Path(__file__).resolve()) != driver_hash:
            raise ValueError("numerical driver changed during execution")
        if _file_digest(preflight_path) != preflight_file_hash:
            raise ValueError("pinned preflight report changed during execution")

    def fail(error: BaseException, phase_name: str) -> None:
        report["execution_status"] = "error"
        report["numerical_status"] = "refused"
        report["response_validation"] = "not_established"
        if report["nominal_eligibility"] == "refinement_attempted":
            report["nominal_eligibility"] = "refused"
        report["error"] = f"{type(error).__name__}: {error}"
        report["failure_category"] = _failure_category(str(error), phase_name)
        checkpoint(phase_name)

    checkpoint("preflight_identity")
    try:
        require_unchanged()
        # The first recheck above is the guarded, pre-solve preflight.
        report["preflight_rechecks"] = {"before": "passed", "after": "pending"}

        report["numerical_status"] = "running"
        checkpoint("partial_gauss_newton")
        contract = problem.contract(parameters)
        gn_started = time.monotonic()
        with patch.object(
            variational,
            "pcg",
            _monitor_pcg(variational.pcg, report, "partial_gauss_newton"),
        ):
            gn = variational.solve_analysis(
                problem.observations, contract, control=warm_start
            )
        gn_control = gn.control.detach().clone()
        report["timings"]["partial_gauss_newton_seconds"] = time.monotonic() - gn_started
        report["partial_gauss_newton"] = {
            "reason": gn.reason,
            "outer_iterations": gn.outer_iterations,
            "pcg_iterations": gn.pcg_iterations,
            "initial_objective": gn.initial_objective,
            "final_objective": gn.final_objective,
            "control_sha256": _tensor_digest(gn_control),
        }
        checkpoint("partial_gauss_newton_complete")

        gn_branch, _, gn_face_margin = _branch_with_gates(
            problem, gn_control, parameters, expected_signature=None
        )
        nominal_signature = _branch_signature(gn_branch)

        def fixed_branch(c: torch.Tensor, p: torch.Tensor) -> tuple[Any, str]:
            branch, scope, _ = _branch_with_gates(
                problem, c, p, expected_signature=nominal_signature
            )
            return branch, scope

        report["nominal_eligibility"] = "refinement_attempted"
        checkpoint("partial_newton_refinement")
        refinement_started = time.monotonic()
        with patch.object(
            local_refinement,
            "pcg",
            _monitor_pcg(local_refinement.pcg, report, "partial_newton_refinement"),
        ):
            refined = local_refinement.refine_stationary(
                objective,
                gn_control,
                parameters,
                branch_check=fixed_branch,
                max_iterations=MAX_REFINEMENT_ITERATIONS,
                max_backtracks=MAX_BACKTRACKS,
                pcg_max_iterations=PCG_MAX_ITERATIONS,
            )
        nominal = refined.control
        nominal_gradient = torch.func.grad(objective, argnums=0)(nominal, parameters)
        nominal_gradient_max = float(nominal_gradient.abs().max())
        nominal_branch, nominal_scope, nominal_face_margin = _branch_with_gates(
            problem, nominal, parameters, expected_signature=nominal_signature
        )
        if not bool(torch.isfinite(nominal_gradient).all()):
            raise ValueError("refined nominal gradient is nonfinite")
        if nominal_gradient_max >= STATIONARITY_TOLERANCE:
            raise ValueError("refined partial nominal is not stationary")
        nominal_objective = objective(nominal, parameters)
        nominal_score = score(nominal, parameters)
        if not bool(torch.isfinite(nominal_objective)) or not bool(torch.isfinite(nominal_score)):
            raise ValueError("nominal objective or score is nonfinite")
        report["timings"]["nominal_refinement_seconds"] = time.monotonic() - refinement_started
        report["nominal"] = {
            "control": nominal.tolist(),
            "control_sha256": _tensor_digest(nominal),
            "objective": float(nominal_objective),
            "score": float(nominal_score),
            "gradient_max": nominal_gradient_max,
            "branch": _branch_summary(nominal_branch, nominal_face_margin),
            "branch_scope": nominal_scope,
            "gauss_newton_branch_face_margin": gn_face_margin,
            "refinement": _serializable(refined),
        }
        report["nominal_eligibility"] = "passed"
        report["numerical_status"] = "nominal_eligible"

        if stop_after_nominal:
            report["response_validation"] = "not_performed"
            report["stopping_reason"] = "diagnostic stopped after nominal stationarity check"
            require_unchanged()
            _fresh_preflight(pinned)
            report["preflight_rechecks"]["after"] = "passed"
            report["execution_status"] = "completed"
            report["timings"]["total_seconds"] = time.monotonic() - started
            checkpoint("finished")
            return report
        checkpoint("nominal_eligible")

        gradient = torch.func.grad(objective, argnums=0)

        def hessian_vector(delta: torch.Tensor) -> torch.Tensor:
            return torch.func.jvp(
                lambda c: gradient(c, parameters), (nominal,), (delta,)
            )[1]

        cross = torch.func.jvp(
            lambda p: gradient(nominal, p), (parameters,), (direction,)
        )[1]
        tangent_rhs = -cross
        tangent_rhs_norm = float(tangent_rhs.norm())
        if not math.isfinite(tangent_rhs_norm) or tangent_rhs_norm == 0:
            raise ValueError("partial tangent right-hand side is zero or nonfinite")
        checkpoint("tangent_solve")
        tangent_started = time.monotonic()
        with patch.object(
            matrix_free, "pcg", _monitor_pcg(matrix_free.pcg, report, "tangent_solve")
        ):
            tangent_result = matrix_free.pcg(
                hessian_vector,
                tangent_rhs,
                rtol=LINEAR_RELATIVE_TOLERANCE,
                max_iterations=PCG_MAX_ITERATIONS,
            )
        tangent = tangent_result.solution
        tangent_residual = hessian_vector(tangent) - tangent_rhs
        tangent_actual_relative = float(tangent_residual.norm()) / tangent_rhs_norm
        if (not tangent_result.converged or not bool(torch.isfinite(tangent).all())
                or not math.isfinite(tangent_actual_relative)
                or tangent_actual_relative > LINEAR_RELATIVE_TOLERANCE):
            raise RuntimeError("partial tangent failed actual residual gate")
        report["tangent"] = {
            "solution": tangent.tolist(),
            "solution_sha256": _tensor_digest(tangent),
            "cross_gradient": cross.tolist(),
            "cross_gradient_sha256": _tensor_digest(cross),
            "converged": tangent_result.converged,
            "iterations": tangent_result.iterations,
            "pcg_relative_residual": tangent_result.relative_residual,
            "actual_relative_residual": tangent_actual_relative,
            "rtol": LINEAR_RELATIVE_TOLERANCE,
            "max_iterations": PCG_MAX_ITERATIONS,
            "seconds": time.monotonic() - tangent_started,
        }

        identity = {
            **input_identity,
            "nominal_control_sha256": _tensor_digest(nominal),
            "nominal_branch_sha256": _json_digest(nominal_signature),
        }
        checkpoint("adjoint_response")
        response_started = time.monotonic()
        with patch.object(
            local_response,
            "pcg",
            _monitor_pcg(local_response.pcg, report, "adjoint_response"),
        ):
            response = local_response.compute_local_response(
                objective,
                score,
                nominal,
                parameters,
                {"middle_time_valid_bias": direction},
                branch_check=fixed_branch,
                input_identity=identity,
            )
        adjoint_response = float(response.total["middle_time_valid_bias"])
        if not math.isfinite(adjoint_response) or adjoint_response == 0:
            raise ValueError("partial adjoint response is zero or nonfinite")
        report["timings"]["adjoint_response_seconds"] = time.monotonic() - response_started
        report["response"] = {
            "direct": float(response.direct["middle_time_valid_bias"]),
            "indirect": float(response.indirect["middle_time_valid_bias"]),
            "total": adjoint_response,
            "score_control_gradient": response.score_control_gradient.tolist(),
            "direct_parameter_gradient": response.direct_gradient.tolist(),
            "indirect_parameter_gradient": response.indirect_gradient.tolist(),
            "total_parameter_gradient": response.total_gradient.tolist(),
            "mixed_gradient": response.mixed_gradients["middle_time_valid_bias"].tolist(),
            "adjoint": response.adjoint.tolist(),
            "gradient_max": response.gradient_max,
            "true_adjoint_residual": response.true_adjoint_residual,
            "true_adjoint_relative_residual": response.true_adjoint_relative_residual,
            "pcg_relative_residual": response.pcg_relative_residual,
            "pcg_iterations": response.pcg_iterations,
            "hvp_count": response.hvp_count,
            "branch": _branch_summary(nominal_branch, nominal_face_margin),
            "scope": response.scope,
        }
        if response.true_adjoint_relative_residual > LINEAR_RELATIVE_TOLERANCE:
            raise RuntimeError("partial transposed-adjoint residual exceeds tolerance")

        predicted = abs(adjoint_response)
        consecutive_passes = 0
        last_valid_pair: dict[str, Any] | None = None
        report["response_validation"] = "running"
        report["numerical_status"] = "response_ready"
        for step_index, step_size in enumerate(PREDECLARED_STEPS):
            signal_separation, signal_floor, signal_ok = score_signal_gate(
                step_size, adjoint_response, float(nominal_score)
            )
            pair: dict[str, Any] = {
                "step_index": step_index,
                "h": step_size,
                "expected_score_separation": signal_separation,
                "score_signal_floor": signal_floor,
                "signal_gate_passed": signal_ok,
                "endpoints": {},
            }
            if not pair["signal_gate_passed"]:
                pair["status"] = "signal_below_roundoff_floor"
                report["pairs"].append(pair)
                report["response_validation"] = "not_established"
                report["numerical_status"] = "unresolved"
                report["stopping_reason"] = "predicted score separation is roundoff dominated"
                checkpoint(f"step_{step_index}_signal_refusal")
                break

            checkpoint(f"step_{step_index}_endpoints")
            for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
                endpoint_p = parameters + (sign * step_size) * direction
                endpoint_start = nominal + (sign * step_size) * tangent
                endpoint: dict[str, Any] = {
                    "sign": sign_name,
                    "parameters_sha256": _tensor_digest(endpoint_p),
                    "start_control_sha256": _tensor_digest(endpoint_start),
                    "status": "running",
                }
                pair["endpoints"][sign_name] = endpoint
                report["endpoints"].append(endpoint)
                endpoint_started = time.monotonic()
                try:
                    start_branch, start_scope, start_face_margin = _branch_with_gates(
                        problem,
                        endpoint_start,
                        endpoint_p,
                        expected_signature=nominal_signature,
                    )
                    report["nonlinear_reanalyses"] += 1
                    checkpoint(f"step_{step_index}_{sign_name}_refinement")
                    with patch.object(
                        local_refinement,
                        "pcg",
                        _monitor_pcg(
                            local_refinement.pcg,
                            report,
                            f"step_{step_index}_{sign_name}_refinement",
                        ),
                    ):
                        endpoint_refined = local_refinement.refine_stationary(
                            objective,
                            endpoint_start,
                            endpoint_p,
                            branch_check=fixed_branch,
                            max_iterations=MAX_REFINEMENT_ITERATIONS,
                            max_backtracks=MAX_BACKTRACKS,
                            pcg_max_iterations=PCG_MAX_ITERATIONS,
                        )
                    endpoint_control = endpoint_refined.control
                    endpoint_gradient = torch.func.grad(objective, argnums=0)(
                        endpoint_control, endpoint_p
                    )
                    endpoint_gradient_max = float(endpoint_gradient.abs().max())
                    endpoint_branch, endpoint_scope, endpoint_face_margin = _branch_with_gates(
                        problem,
                        endpoint_control,
                        endpoint_p,
                        expected_signature=nominal_signature,
                    )
                    endpoint_score = score(endpoint_control, endpoint_p)
                    endpoint_objective = objective(endpoint_control, endpoint_p)
                    if not bool(torch.isfinite(endpoint_gradient).all()):
                        raise ValueError("endpoint gradient is nonfinite")
                    if (endpoint_gradient_max >= STATIONARITY_TOLERANCE
                            or not endpoint_refined.gradient_max < STATIONARITY_TOLERANCE):
                        raise ValueError("endpoint did not meet stationarity gate")
                    if not bool(torch.isfinite(endpoint_score)) or not bool(torch.isfinite(endpoint_objective)):
                        raise ValueError("endpoint score or objective is nonfinite")
                    endpoint.update({
                        "status": "eligible",
                        "control": endpoint_control.tolist(),
                        "control_sha256": _tensor_digest(endpoint_control),
                        "objective": float(endpoint_objective),
                        "score": float(endpoint_score),
                        "gradient_max": endpoint_gradient_max,
                        "branch": _branch_summary(endpoint_branch, endpoint_face_margin),
                        "branch_scope": endpoint_scope,
                        "predictor_branch": _branch_summary(start_branch, start_face_margin),
                        "predictor_scope": start_scope,
                        "refinement": _serializable(endpoint_refined),
                    })
                except (ValueError, RuntimeError) as error:
                    endpoint.update({
                        "status": "refused",
                        "error": f"{type(error).__name__}: {error}",
                        "failure_category": _failure_category(str(error), f"step_{step_index}_{sign_name}"),
                    })
                finally:
                    endpoint["seconds"] = time.monotonic() - endpoint_started
                    checkpoint(f"step_{step_index}_{sign_name}_complete")

            minus = pair["endpoints"]["minus"]
            plus = pair["endpoints"]["plus"]
            if minus["status"] == "eligible" and plus["status"] == "eligible":
                central, relative_error = central_response_error(
                    minus["score"], plus["score"], step_size, adjoint_response
                )
                if report["endpoint_eligibility"] == "not_attempted":
                    report["endpoint_eligibility"] = "all_attempted_endpoints_eligible"
                pair.update({
                    "status": "valid_pair",
                    "central_score_slope": central,
                    "adjoint_response": adjoint_response,
                    "relative_error": relative_error,
                    "passed": relative_error <= 1.0e-4,
                    "response_scale": predicted,
                })
                if pair["passed"]:
                    consecutive_passes += 1
                else:
                    consecutive_passes = 0
                last_valid_pair = pair
            else:
                report["endpoint_eligibility"] = "one_or_more_endpoints_refused"
                pair.update({
                    "status": "endpoint_refused",
                    "failure": "one or both signed endpoint refinements refused",
                })
                consecutive_passes = 0
            pair["consecutive_passing_pairs"] = consecutive_passes
            report["pairs"].append(pair)
            checkpoint(f"step_{step_index}_pair_complete")
            if consecutive_passes >= MINIMUM_CONSECUTIVE_PAIRS:
                report["response_validation"] = "passed"
                report["numerical_status"] = "response_validated"
                report["stopping_reason"] = "two consecutive valid pairs passed relative-error gate"
                break
        else:
            report["response_validation"] = "not_established"
            report["numerical_status"] = "unresolved"
            report["stopping_reason"] = "predeclared step sequence exhausted"

        if report["response_validation"] == "running":
            report["response_validation"] = "not_established"
        report["last_valid_pair"] = last_valid_pair
        report["consecutive_passing_pairs"] = consecutive_passes
        report["nominal_eligibility"] = "passed"
        require_unchanged()
        fresh_after = _fresh_preflight(pinned)
        if fresh_after != initial_preflight:
            raise ValueError("preflight inputs changed during numerical work")
        runtime_commit_after = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if runtime_commit_after != runtime_commit:
            raise ValueError("repository commit changed during numerical work")
        report["runtime_commit_after"] = runtime_commit_after
        report["preflight_rechecks"]["after"] = "passed"
        report["execution_status"] = "completed"
        report["timings"]["total_seconds"] = time.monotonic() - started
        checkpoint("finished")
        return report
    except Exception as error:
        fail(error, str(report.get("phase", "unknown")))
        raise


def validate_only(output: Path, preflight_path: Path) -> dict[str, Any]:
    """Exercise the pinned preflight path without starting GN, PCG, or Newton."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        pinned = json.loads(preflight_path.read_text())
        report_hash = _file_digest(preflight_path)
        current = _fresh_preflight(pinned)
        relative = str(Path(__file__).resolve().relative_to(ROOT))
        driver_hash = _file_digest(Path(__file__).resolve())
        if pinned["source_sha256"].get(relative) != driver_hash:
            raise ValueError("pinned preflight does not contain this numerical driver SHA256")
        if _file_digest(preflight_path) != report_hash:
            raise ValueError("pinned preflight report changed during validation")
        result: dict[str, Any] = {
            "execution_status": "completed",
            "numerical_status": "not_started",
            "response_validation": "not_established",
            "phase": "validate_only_complete",
            "validate_only": True,
            "numerical_solver_runs": 0,
            "pinned_source_commit": pinned["base_commit"],
            "runtime_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "preflight_path": str(preflight_path),
            "preflight_report_sha256": report_hash,
            "preflight_identity": current,
            "driver_sha256": driver_hash,
            "source_sha256": pinned["source_sha256"],
        }
    except Exception as error:
        result = {
            "execution_status": "error",
            "numerical_status": "not_started",
            "response_validation": "not_established",
            "phase": "validate_only_preflight",
            "validate_only": True,
            "numerical_solver_runs": 0,
            "error": f"{type(error).__name__}: {error}",
        }
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(output)
    if result["execution_status"] != "completed":
        raise ValueError(str(result["error"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        type=Path,
        default=ROOT / "graphify-out/fv-root-cause-20260919/fv_partial_reanalysis_preflight.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-after-nominal", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and not args.execute:
        parser.error("numerical execution requires --execute after budget approval")
    result = (
        validate_only(args.output, args.preflight)
        if args.validate_only
        else run(args.output, args.preflight, stop_after_nominal=args.stop_after_nominal)
    )
    print(json.dumps({
        "execution_status": result["execution_status"],
        "numerical_status": result["numerical_status"],
        "response_validation": result["response_validation"],
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
