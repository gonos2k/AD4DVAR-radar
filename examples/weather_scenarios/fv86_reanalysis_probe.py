"""Resource-bounded local reanalysis check for the recorded FV86 seed-A point.

This probe checks one conditional direction and local step sequence. It does
not rerun Gauss-Newton and makes no general response or finite-path claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, cast
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES = ROOT / "examples/weather_scenarios"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from advar import local_refinement
from advar.matrix_free import pcg
from examples.weather_scenarios.fv86_execution_probe import branch_summary, digest, load, failure_category

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
BASELINE_PATH = EVIDENCE / "fv86_seed_a.json"
BASELINE_SHA256 = "c8de13325e4d76a5e904a6ef457deed853aa762cc6d051afd7fb59cd6b9e3bd0"
PLAN_PATH = EVIDENCE / "FV86_REANALYSIS_PLAN.md"
STEP_SIZES = tuple(0.001 * (2.0 ** -index) for index in range(6))
MAX_SECONDS = 1200.0
MAX_RSS_BYTES = 2 * 1024**3


def relative_error(actual: float, expected: float, *, scale: float | None = None) -> float:
    """Compare scalars using a caller-selected physically relevant scale."""
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError("response comparison must be finite")
    denominator = max(abs(actual), abs(expected)) if scale is None else scale
    if denominator == 0 and actual == expected:
        return 0.0
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("comparison scale must be finite and positive")
    return abs(actual - expected) / denominator


def pair_error(
    plus_score: float,
    minus_score: float,
    step_size: float,
    response: float,
    *,
    scale: float | None = None,
) -> tuple[float, float]:
    """Return central score slope and scale-aware error against a response."""
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    slope = (plus_score - minus_score) / (2.0 * step_size)
    return slope, relative_error(slope, response, scale=scale)


def serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {key: serializable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = load("fv_scaled_research_case")
    baseline = json.loads(BASELINE_PATH.read_text())
    plan_hash = sha256_file(PLAN_PATH)
    baseline_hash = sha256_file(BASELINE_PATH)

    source_paths = [
        Path(__file__),
        Path(fixture.__file__),
        Path(__file__).with_name("fv86_execution_probe.py"),
        Path(__file__).with_name("fv_gn_response_probe.py"),
        Path(__file__).with_name("fv_minmod_inverse_probe.py"),
        Path(__file__).with_name("fv_sensitivity_probe.py"),
        Path(__file__).with_name("fv_analysis_response.py"),
        Path(__file__).with_name("fv86_resource_runner.py"),
        Path(__file__).with_name("fv86_reanalysis_runner.py"),
        ROOT / "src/advar/local_refinement.py",
        ROOT / "src/advar/local_response.py",
        ROOT / "src/advar/fv_research_problem.py",
        ROOT / "src/advar/matrix_free.py",
        ROOT / "src/advar/variational.py",
        ROOT / "src/advar/transport.py",
        ROOT / "src/advar/physics.py",
        ROOT / "src/advar/nowcast.py",
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
    }
    archived_sources = baseline["source_sha256"]
    archived_source_matches = {
        name: (source_hashes.get(name) == expected)
        for name, expected in archived_sources.items()
    }
    report: dict[str, Any] = {
        "status": "running",
        "phase": "prepare",
        "scope": "one fixed 86-control nominal point and middle-frame uniform dBZ direction",
        "environment": {"torch": torch.__version__, "device": "CPU FP64"},
        "authorization": {
            "plan_sha256": plan_hash,
            "wall_cap_seconds": MAX_SECONDS,
            "rss_cap_bytes": MAX_RSS_BYTES,
        },
        "baseline": {
            "path": str(BASELINE_PATH.relative_to(ROOT)) if BASELINE_PATH.is_relative_to(ROOT) else str(BASELINE_PATH),
            "sha256": baseline_hash,
            "expected_sha256": BASELINE_SHA256,
            "sha256_matches": baseline_hash == BASELINE_SHA256,
            "status": baseline.get("status"),
            "mode": baseline.get("mode"),
            "source_sha256": archived_sources,
            "archived_source_matches": archived_source_matches,
        },
        "source_sha256": source_hashes,
        "timings": {},
        "linear_solves": [],
        "endpoints": [],
        "pairs": [],
        "general_minmod_response_eligible": False,
        "finite_path_certified": False,
        "nonlinear_reanalyses": 0,
        "execution_success": False,
        "numerical_validation": "not_started",
        "response_validation": "not_performed",
    }

    def checkpoint() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    def phase(name: str) -> None:
        report["phase"] = name
        checkpoint()
        require_budget()

    def require_budget() -> None:
        elapsed = time.monotonic() - started
        if elapsed >= MAX_SECONDS:
            raise TimeoutError(f"approved wall-time cap reached at {elapsed:.3f}s")

    def monitored_pcg(original):
        def solve(operator, rhs, **kwargs):
            entry: dict[str, Any] = {
                "phase": report["phase"],
                "rtol": kwargs.get("rtol"),
                "max_iterations": kwargs.get("max_iterations"),
                "hvp_calls": 0,
            }
            report["linear_solves"].append(entry)
            tick = time.monotonic()

            def measured(direction):
                require_budget()
                result = operator(direction)
                entry["hvp_calls"] += 1
                if entry["hvp_calls"] % 16 == 0:
                    checkpoint()
                return result

            try:
                result = original(measured, rhs, **kwargs)
                entry.update(
                    converged=result.converged,
                    iterations=result.iterations,
                    pcg_relative_residual=result.relative_residual,
                )
                return result
            except (ValueError, RuntimeError) as error:
                entry["raw_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                entry["seconds"] = time.monotonic() - tick
                checkpoint()

        return solve

    checkpoint()
    try:
        if baseline_hash != BASELINE_SHA256:
            raise ValueError("recorded seed-A report SHA-256 does not match the pinned artifact")
        if baseline.get("mode") != "seed_a" or baseline.get("status") != "eligible":
            raise ValueError("recorded seed-A baseline is not eligible")
        changed_adapters = {"examples/weather_scenarios/fv86_execution_probe.py",
                            "examples/weather_scenarios/fv_scaled_research_case.py",
                            "examples/weather_scenarios/fv_gn_response_probe.py"}
        mismatches = [name for name, matches in archived_source_matches.items()
                      if not matches and name not in changed_adapters]
        if mismatches:
            raise ValueError(f"baseline numerical source mismatch: {mismatches}")
        report["baseline"]["refactored_adapters"] = [
            name for name in changed_adapters if not archived_source_matches.get(name, True)]
        report["baseline"]["compatibility_evidence"] = "PR175 common-problem frozen-source parity; fresh cache derivatives checked below"

        case = fixture.make_case()
        p = case.parameters
        nominal_control = torch.tensor(baseline["workflow"]["control"], dtype=torch.float64)
        expected_branch = baseline["nominal_branch"]
        objective, score, branch_check = fixture.functions(case, expected_branch=expected_branch)
        problem_contract = {
            "parameters": digest(p),
            "verification": digest(case.verification),
            "initial_echo": digest(case.truth_initial_echo),
            "basis": digest(case.frozen.fv_transport.psi_basis),
            "problem": digest(case.definition),
        }
        if problem_contract != baseline["input_identity"]:
            raise ValueError("reconstructed common-problem identity differs from seed-A baseline")
        baseline_branch_row = next(
            row for row in baseline["branch_checks"]
            if row["control_sha256"] == digest(nominal_control)
        )
        if digest(nominal_control) != baseline_branch_row["control_sha256"]:
            raise ValueError("seed-A nominal control hash does not match its archived branch check")

        current_input_contract = {
            "problem": problem_contract,
            "control": digest(nominal_control),
            "baseline_report_sha256": baseline_hash,
            "plan_sha256": plan_hash,
        }
        report["input_identity"] = current_input_contract
        nominal_branch, branch_scope = branch_check(nominal_control, p)
        if any(branch_summary(nominal_branch)[key] != baseline_branch_row[key]
               for key in ("euler_stages", "selectors_sha256", "face_signs_sha256")):
            raise ValueError("fresh nominal branch differs from the archived seed-A branch")
        gradient = torch.func.grad(objective, argnums=0)
        nominal_gradient = gradient(nominal_control, p)
        nominal_gradient_max = float(nominal_gradient.abs().max())
        if not bool(torch.isfinite(nominal_gradient).all()) or nominal_gradient_max >= 1e-10:
            raise ValueError(f"seed-A nominal point failed fresh stationarity check: {nominal_gradient_max}")
        nominal_objective = objective(nominal_control, p)
        nominal_score = score(nominal_control, p)
        if not bool(torch.isfinite(nominal_objective)) or not bool(torch.isfinite(nominal_score)):
            raise ValueError("fresh nominal objective or score is nonfinite")

        if relative_error(float(nominal_objective),baseline["workflow"]["after"]["objective"]) > 1e-10:
            raise ValueError("nominal objective changed from baseline")
        if relative_error(float(nominal_score),baseline["final_score"]) > 1e-10:
            raise ValueError("nominal score changed from baseline")
        frame_size = case.observations.dbz[0].numel()
        direction = torch.zeros_like(p)
        direction[frame_size : 2 * frame_size] = 1.0
        middle_bias = direction
        cached = baseline["workflow"]["response"]
        cached_direct = float(cached["direct"]["middle_time_bias"])
        cached_indirect = float(cached["indirect"]["middle_time_bias"])
        cached_total = float(cached["total"]["middle_time_bias"])
        score_control_gradient, score_parameter_gradient = torch.func.grad(
            score, argnums=(0, 1)
        )(nominal_control, p)
        for name, fresh, saved_values in (
            ("Ec", score_control_gradient, cached["score_control_gradient"]),
            ("Ep", score_parameter_gradient, cached["direct_gradient"]),
        ):
            saved_vector = torch.tensor(saved_values, dtype=fresh.dtype)
            if fresh.shape != saved_vector.shape or not torch.allclose(fresh, saved_vector, rtol=1e-10, atol=0):
                raise ValueError(f"fresh {name} differs from cached score gradient")
        report["direction"] = {"name": "middle_time_bias", "values": direction.tolist(),
                               "sha256": digest(direction), "units": "dBZ offset at middle observation time"}
        fresh_cross = torch.func.jvp(
            lambda parameters: gradient(nominal_control, parameters),
            (p,),
            (middle_bias,),
        )[1]
        fresh_transpose = torch.func.vjp(
            lambda control: gradient(control, p), nominal_control
        )[1]
        cached_adjoint = torch.tensor(cached["adjoint"], dtype=nominal_control.dtype)
        fresh_ht_lambda = fresh_transpose(cached_adjoint)[0]
        adjoint_rhs = score_control_gradient
        rhs_norm = float(adjoint_rhs.norm())
        adjoint_residual = float((fresh_ht_lambda - adjoint_rhs).norm())
        adjoint_relative = adjoint_residual / rhs_norm if rhs_norm else adjoint_residual
        if not math.isfinite(adjoint_relative) or adjoint_relative > 1e-10:
            raise ValueError(f"fresh VJP does not verify cached adjoint: relative residual {adjoint_relative}")

        fresh_direct = float(torch.dot(score_parameter_gradient, middle_bias))
        fresh_indirect = -float(torch.dot(cached_adjoint, fresh_cross))
        fresh_total = fresh_direct + fresh_indirect
        cached_mixed = torch.tensor(
            cached["mixed_gradients"]["middle_time_bias"], dtype=nominal_control.dtype
        )
        if cached_mixed.shape != fresh_cross.shape or not torch.allclose(fresh_cross, cached_mixed, rtol=1e-10, atol=0):
            raise ValueError("fresh mixed derivative differs from cached direction")
        mixed_scale = max(float(fresh_cross.norm()), float(cached_mixed.norm()))
        mixed_relative_error = float((fresh_cross - cached_mixed).norm()) / mixed_scale if mixed_scale else 0.0
        component_errors = {
            "direct": relative_error(fresh_direct, cached_direct),
            "indirect": relative_error(fresh_indirect, cached_indirect),
            "total": relative_error(fresh_total, cached_total),
        }
        if fresh_total == 0:
            raise ValueError("predeclared relative response gate is undefined for zero response")
        if mixed_relative_error > 1e-6 or any(error > 1e-6 for error in component_errors.values()):
            raise ValueError(
                "fresh score response differs from cached mixed derivative/components: "
                f"mixed={mixed_relative_error}, components={component_errors}"
            )
        report["nominal"] = {
            "control_sha256": digest(nominal_control),
            "objective": float(nominal_objective),
            "score": float(nominal_score),
            "gradient_max": nominal_gradient_max,
            "branch": branch_summary(nominal_branch),
            "branch_scope": branch_scope,
            "cached_response": {
                "direct": cached_direct,
                "indirect": cached_indirect,
                "total": cached_total,
            },
            "fresh_response": {
                "direct": fresh_direct,
                "indirect": fresh_indirect,
                "total": fresh_total,
                "component_relative_errors": component_errors,
                "mixed_gradient_relative_error": mixed_relative_error,
            },
            "cached_adjoint_fresh_vjp": {
                "residual_norm": adjoint_residual,
                "relative_residual": adjoint_relative,
                "tolerance": 1e-10,
            },
            "fresh_cross_sha256": digest(fresh_cross),
        }
        phase("tangent")

        def hessian_vector(delta):
            return torch.func.jvp(
                lambda control: gradient(control, p), (nominal_control,), (delta,)
            )[1]

        tangent_rhs = -fresh_cross
        tangent_tick = time.monotonic()
        with patch.object(local_refinement, "pcg", monitored_pcg(pcg)):
            tangent_solve = local_refinement.pcg(
                hessian_vector,
                tangent_rhs,
                rtol=1e-10,
                max_iterations=344,
            )
        tangent = tangent_solve.solution
        tangent_residual_vector = hessian_vector(tangent) - tangent_rhs
        tangent_rhs_norm = float(tangent_rhs.norm())
        tangent_residual = float(tangent_residual_vector.norm())
        tangent_relative = tangent_residual / tangent_rhs_norm if tangent_rhs_norm else tangent_residual
        report["tangent"] = {
            "solution": tangent.tolist(),
            "solution_sha256": digest(tangent),
            "rhs_sha256": digest(tangent_rhs),
            "converged": tangent_solve.converged,
            "iterations": tangent_solve.iterations,
            "pcg_relative_residual": tangent_solve.relative_residual,
            "actual_residual_norm": tangent_residual,
            "actual_relative_residual": tangent_relative,
            "rtol": 1e-10,
            "max_iterations": 344,
            "seconds": time.monotonic() - tangent_tick,
        }
        if (not tangent_solve.converged or not bool(torch.isfinite(tangent).all())
                or not math.isfinite(tangent_relative) or tangent_relative > 1e-10):
            raise RuntimeError(f"tangent solve failed actual residual gate: {tangent_relative}")
        cast(dict[str, Any], report["nominal"]["cached_adjoint_fresh_vjp"])["tangent_projection_error_diagnostic"] = float(
            torch.dot(fresh_ht_lambda - adjoint_rhs, tangent)
        )
        cast(dict[str, Any], report["nominal"]["cached_adjoint_fresh_vjp"])["diagnostic_scope"] = (
            "adjoint residual projected onto the tangent; diagnostic only, not the "
            "finite-difference error certificate"
        )

        pairs_consecutive_passes = 0
        last_valid_pair: dict[str, Any] | None = None
        for step_index, step_size in enumerate(STEP_SIZES):
            require_budget()
            phase(f"step_{step_index}_endpoints")
            pair_record: dict[str, Any] = {
                "step_index": step_index,
                "h": step_size,
                "endpoints": {},
                "valid_pair": False,
            }
            for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                require_budget()
                report["phase"] = f"step_{step_index}_{sign_name}"
                checkpoint()
                require_budget()
                endpoint_parameters = p + (sign * step_size) * direction
                start_control = nominal_control + (sign * step_size) * tangent
                endpoint: dict[str, Any] = {
                    "sign": sign_name,
                    "parameters_sha256": digest(endpoint_parameters),
                    "parameters": endpoint_parameters.tolist(),
                    "start_control_sha256": digest(start_control),
                    "start_control": start_control.tolist(),
                    "status": "running",
                }
                report["endpoints"].append(endpoint)
                pair_record["endpoints"][sign_name] = endpoint
                endpoint_tick = time.monotonic()
                candidates: list[dict[str, Any]] = []

                def checked_branch(trial, parameters):
                    branch, scope = branch_check(trial, parameters)
                    candidates.append({
                        "control_sha256": digest(trial),
                        "branch": branch_summary(branch),
                    })
                    return branch_summary(branch), scope

                try:
                    checked_branch(start_control, endpoint_parameters)
                    report["nonlinear_reanalyses"] += 1
                    with patch.object(local_refinement, "pcg", monitored_pcg(local_refinement.pcg)):
                        refined = local_refinement.refine_stationary(
                            objective,
                            start_control,
                            endpoint_parameters,
                            branch_check=checked_branch,
                            max_iterations=8,
                            max_backtracks=16,
                            pcg_max_iterations=344,
                        )
                    final_branch, final_scope = branch_check(refined.control, endpoint_parameters)
                    final_gradient = gradient(refined.control, endpoint_parameters)
                    final_objective = objective(refined.control, endpoint_parameters)
                    final_score = score(refined.control, endpoint_parameters)
                    if not bool(torch.isfinite(final_score)) or not bool(torch.isfinite(final_objective)):
                        raise ValueError("endpoint objective or score is nonfinite")
                    if not bool(torch.isfinite(final_gradient).all()):
                        raise ValueError("refined endpoint gradient is nonfinite")
                    if float(final_gradient.abs().max()) >= 1e-10:
                        raise ValueError("refined endpoint failed fresh stationarity gate")
                    endpoint.update({
                        "status": "eligible",
                        "control": refined.control.tolist(),
                        "control_sha256": digest(refined.control),
                        "objective": float(final_objective),
                        "score": float(final_score),
                        "gradient_max": float(final_gradient.abs().max()),
                        "branch": branch_summary(final_branch),
                        "branch_scope": final_scope,
                        "actual_cfl": fixture.actual_cfl(case, refined.control),
                        "refinement": serializable(refined),
                        "branch_candidates": candidates,
                    })
                except (ValueError, RuntimeError) as error:
                    endpoint.update({
                        "status": "refused",
                        "raw_error": f"{type(error).__name__}: {error}",
                        "failure_category": failure_category(str(error), "refinement"),
                        "branch_candidates": candidates,
                    })
                finally:
                    endpoint["seconds"] = time.monotonic() - endpoint_tick
                    checkpoint()
                pair_record["endpoints"][sign_name] = endpoint
            plus = pair_record["endpoints"]["plus"]
            minus = pair_record["endpoints"]["minus"]
            if plus["status"] == "eligible" and minus["status"] == "eligible":
                response_scale = abs(fresh_total)
                finite_difference, error = pair_error(
                    plus["score"], minus["score"], step_size, fresh_total,
                    scale=response_scale,
                )
                pair_record.update(
                    valid_pair=True,
                    central_score_slope=finite_difference,
                    fresh_adjoint_directional_response=fresh_total,
                    relative_error=error,
                    passed=error <= 1e-4,
                    response_scale=response_scale,
                )
                if pair_record["passed"]:
                    pairs_consecutive_passes += 1
                else:
                    pairs_consecutive_passes = 0
                last_valid_pair = pair_record
            else:
                pairs_consecutive_passes = 0
                pair_record["failure"] = "one or both endpoints refused"
            report["pairs"].append(pair_record)
            checkpoint()
            if pairs_consecutive_passes >= 2:
                report["stopping_reason"] = "two consecutive valid pairs passed relative-error gate"
                break
        else:
            report["stopping_reason"] = "predeclared step sequence exhausted"

        report["response_validation"] = "passed" if pairs_consecutive_passes >= 2 else "failed"
        report["last_valid_pair"] = last_valid_pair
        report["numerical_validation"] = (
            "passed for this nominal point, direction, and local steps"
            if pairs_consecutive_passes >= 2
            else "not established; see endpoint and pair records"
        )
        source_unchanged = all(
            sha256_file(path) == source_hashes[str(path.relative_to(ROOT))]
            for path in source_paths
        )
        baseline_unchanged = sha256_file(BASELINE_PATH) == baseline_hash
        plan_unchanged = sha256_file(PLAN_PATH) == plan_hash
        input_identity_after = {
            "parameters": digest(case.parameters),
            "verification": digest(case.verification),
            "initial_echo": digest(case.truth_initial_echo),
            "basis": digest(case.frozen.fv_transport.psi_basis),
            "problem": digest(case.definition),
            "control": digest(nominal_control),
        }
        report["input_identity_before"] = current_input_contract
        report["input_identity_after"] = input_identity_after
        inputs_unchanged = (
            input_identity_after["parameters"] == problem_contract["parameters"]
            and input_identity_after["verification"] == problem_contract["verification"]
            and input_identity_after["initial_echo"] == problem_contract["initial_echo"]
            and input_identity_after["basis"] == problem_contract["basis"]
            and input_identity_after["problem"] == problem_contract["problem"]
            and input_identity_after["control"] == current_input_contract["control"]
        )
        report["inputs_unchanged"] = inputs_unchanged
        report["source_unchanged"] = source_unchanged
        report["baseline_unchanged"] = baseline_unchanged
        report["plan_unchanged"] = plan_unchanged
        if not (source_unchanged and baseline_unchanged and plan_unchanged and inputs_unchanged):
            raise RuntimeError("input or source snapshot changed during the probe")
        report["execution_success"] = True
        report["status"] = "completed"
        phase("finished")
    except Exception as error:
        report.update(
            status="execution_error",
            execution_success=False,
            failure_category="wall_time_limit" if isinstance(error, TimeoutError) else failure_category(str(error), report["phase"]),
            error_type=type(error).__name__,
            error=f"{type(error).__name__}: {error}",
        )
        checkpoint()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output)["status"])
