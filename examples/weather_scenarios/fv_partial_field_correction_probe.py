"""Run one guarded fixed-dynamics field-block correction from the PR212 seed."""
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

from advar import matrix_free
from advar.local_refinement import (RefinementCallbackError,
                                    RefinementNumericalRefusal,
                                    RefinementTrial, refine_stationary)
from examples.weather_scenarios import fv_partial_alternate_root_probe as seed_probe
from examples.weather_scenarios import fv_partial_reanalysis_preflight as preflight
from examples.weather_scenarios import fv_partial_sector_root_probe as prior
from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios.fv_point_sector_policy import _digest, _key
from tests.test_fv_research_partial_observation import _problem


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_PARTIAL_FIELD_CORRECTION_PLAN.md"
PLAN_SHA256 = "391bfdbeb4f97af617dc00840700cadba36c36be8ed797e78026ba8cc3aed1f8"
RUNNER_PATH = "examples/weather_scenarios/fv86_resource_runner.py"
RUNNER_SHA256 = "6242598d8b11a74ed44fcd419d83fd14cfcd637f34ee4746e1f5432554aeeb6d"
SELF_PATH = "examples/weather_scenarios/fv_partial_field_correction_probe.py"
SOURCE_PATHS = tuple(dict.fromkeys((
    *seed_probe.SOURCE_PATHS,
    RUNNER_PATH,
    SELF_PATH,
)))
SEED_CONTROL_SHA256 = "125e0200fdf85f996715ba1bc04aa3f631333614bf15cf2c16630b94c1897421"
SEED_SIGNATURE_SHA256 = "50d3b1a4bad6761b14806c62708400d87d16e506ab08ef545f7ba9d870bece6d"
WALL_SECONDS = 300
SAMPLED_RSS_BYTES = 1024**3
FIELD_SIZE = 20
CONTROL_SIZE = 26
STATIONARITY_TOLERANCE = 1e-10
TRUE_RESIDUAL_TOLERANCE = 1e-10
OBJECTIVE_ROUNDOFF_FACTOR = 128.0
VALID_NUMERICAL_STATUSES = {"block_stationary_candidate", "field_correction_refused"}
KNOWN_PCG_ERRORS = (
    "residual norm is not finite", "preconditioner must be positive definite",
    "operator must be symmetric positive definite", "PCG step is not finite",
    "true residual norm is not finite", "PCG direction update is not finite",
)


class _SeedIdentityError(ValueError):
    """The pinned seed, source, input, or fixed parameter identity changed."""


class _BranchCandidateRefusal(ValueError):
    """A candidate does not satisfy the frozen strict branch contract."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _field_objective(objective: Any, fixed_dynamics: Tensor):
    """Restrict a full objective to z while keeping d constant and detached."""
    fixed = fixed_dynamics.detach().clone()
    if fixed.shape != (CONTROL_SIZE - FIELD_SIZE,):
        raise ValueError("fixed dynamics must contain six controls")

    def reduced(field: Tensor, parameters: Tensor) -> Tensor:
        if field.shape != (FIELD_SIZE,):
            raise ValueError("field control must contain twenty values")
        return objective(torch.cat((field, fixed)), parameters)

    return reduced


def _full_control(field: Tensor, fixed_dynamics: Tensor) -> Tensor:
    if field.shape != (FIELD_SIZE,) or fixed_dynamics.shape != (CONTROL_SIZE - FIELD_SIZE,):
        raise ValueError("field/dynamics partition must be 20 plus 6 controls")
    return torch.cat((field, fixed_dynamics))


def _gradient_blocks(gradient: Tensor) -> dict[str, Any]:
    if gradient.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(gradient).all()):
        raise ValueError("full objective gradient must be a finite 26-vector")
    field = gradient[:FIELD_SIZE]
    dynamics = gradient[FIELD_SIZE:]
    return {
        "field_l2": float(torch.linalg.vector_norm(field)),
        "field_inf": float(field.abs().max()),
        "dynamics_l2": float(torch.linalg.vector_norm(dynamics)),
        "dynamics_components": dynamics.tolist(),
        "full_l2": float(torch.linalg.vector_norm(gradient)),
        "full_inf": float(gradient.abs().max()),
    }


def _gradient_audit(problem: Any, control: Tensor, parameters: Tensor) -> dict[str, Any]:
    gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    return {"control_sha256": _tensor_sha(control), **_gradient_blocks(gradient)}


def _branch_summary(branch: dict[str, Any], face_margin: float) -> dict[str, Any]:
    complete = {**branch, "minimum_scaled_face_flux_margin": face_margin}
    return {
        "euler_stages": complete["euler_stages"],
        "minimum_scaled_slope_margin": complete["minimum_scaled_slope_margin"],
        "minimum_scaled_face_flux_margin": face_margin,
        "signature_sha256": _digest(_key(complete)),
    }


def objective_gate(trial: RefinementTrial) -> tuple[bool, str] | None:
    """Reject material J increases while preserving the refiner's Armijo gate."""
    old = trial.current_objective
    new = trial.candidate_objective
    if not math.isfinite(old) or not math.isfinite(new):
        return False, "actual_objective_nonfinite"
    increase = new - old
    tolerance = (OBJECTIVE_ROUNDOFF_FACTOR * torch.finfo(torch.float64).eps
                 * max(abs(old), abs(new), torch.finfo(torch.float64).tiny))
    if not math.isfinite(increase) or not math.isfinite(tolerance) or increase > tolerance:
        return False, "actual_objective_increase"
    # None deliberately leaves normalized gradient-merit Armijo to the refiner.
    return None


def _branch_check(
    problem: Any, fixed_dynamics: Tensor,
    seed_branch: dict[str, Any], face_margin_seed: float,
    calls: list[dict[str, Any]],
):
    def check(field: Tensor, _p: Tensor) -> tuple[dict[str, Any], str]:
        full = _full_control(field, fixed_dynamics)
        entry: dict[str, Any] = {"control_sha256": _tensor_sha(full), "status": "trace_pending"}
        try:
            branch, scope, face_margin = preflight.branch_with_face_margin(
                problem, full, _p,
            )
            summary = _branch_summary(branch, face_margin)
            if (summary["euler_stages"] != 54
                    or not math.isfinite(summary["minimum_scaled_slope_margin"])
                    or summary["minimum_scaled_slope_margin"] <= 0.0
                    or not math.isfinite(face_margin) or face_margin <= 0.0):
                raise _BranchCandidateRefusal("candidate branch margins are not positive")
            if face_margin != face_margin_seed:
                raise RuntimeError("fixed dynamics changed the measured face margin")
            if summary["signature_sha256"] != seed_branch["signature_sha256"]:
                raise _BranchCandidateRefusal("candidate full signature differs from frozen seed")
            entry.update(status="positive_strict_branch", **summary)
            return {**branch, "minimum_scaled_face_flux_margin": face_margin}, scope
        except ValueError as error:
            if str(error) == "minmod joint oracle left its strict smooth branch" or isinstance(
                error, _BranchCandidateRefusal
            ):
                entry.update(status="oracle_branch_refused", reason=f"{type(error).__name__}: {error}")
                raise
            entry.update(status="callback_error", reason=f"{type(error).__name__}: {error}")
            raise RefinementCallbackError(f"unexpected branch oracle ValueError: {error}") from error
        except RuntimeError as error:
            entry.update(status="callback_error", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            calls.append(entry)

    return check


def _run_child(output: Path, *, expected_plan_sha256: str,
               expected_probe_sha256: str) -> dict[str, Any]:
    if (output.exists() or output.is_symlink()
            or output.with_suffix(output.suffix + ".tmp").exists()
            or output.with_suffix(output.suffix + ".tmp").is_symlink()):
        raise ValueError("field correction child output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    if (_sha(PLAN) != expected_plan_sha256 or expected_plan_sha256 != PLAN_SHA256
            or _sha(Path(__file__)) != expected_probe_sha256):
        raise _SeedIdentityError("reviewed field correction plan or probe source changed")
    archives = (seed_probe.SEED_ARCHIVE.resolve(strict=True),
                seed_probe.seed_gate.ARCHIVE.resolve(strict=True))
    temporary = output.with_suffix(output.suffix + ".tmp")
    if (any(path.resolve(strict=False).is_relative_to(archive)
            for path in (output, temporary) for archive in archives)):
        raise ValueError("field correction output must be outside raw archives")

    seed, seed_control = seed_probe._seed_evidence()
    if (_tensor_sha(seed_control) != SEED_CONTROL_SHA256
            or seed.get("selected_signature_sha256") != SEED_SIGNATURE_SHA256):
        raise _SeedIdentityError("PR212 field correction seed differs from frozen identity")
    source_before = _sources()
    input_before = prior._preflight_identity()
    seed_raw_before = seed_probe._seed_record_hashes()
    problem, _, parameters = _problem()
    if (problem.identity != input_before["current_problem_identity"]
            or seed.get("input_before") != input_before):
        raise _SeedIdentityError("current fixed problem differs from validated PR212 seed")

    fixed_dynamics = seed_control[FIELD_SIZE:].detach().clone()
    field_seed = seed_control[:FIELD_SIZE].detach().clone()
    reduced = _field_objective(problem.objective, fixed_dynamics)
    seed_branch_raw, _, seed_face = preflight.branch_with_face_margin(
        problem, seed_control, parameters,
    )
    seed_branch = _branch_summary(seed_branch_raw, seed_face)
    if (seed_branch["euler_stages"] != 54
            or seed_branch["signature_sha256"] != SEED_SIGNATURE_SHA256
            or not math.isfinite(seed_branch["minimum_scaled_slope_margin"])
            or seed_branch["minimum_scaled_slope_margin"] <= 0.0
            or not math.isfinite(seed_face)
            or seed_face <= 0.0):
        raise _SeedIdentityError("fresh frozen seed branch does not match PR212")

    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "field_refinement", "numerical_status": "running",
        "scope": "one fixed-dynamics 20-control field correction; no product GN, response, score, or perturbed reanalysis",
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "plan_sha256": expected_plan_sha256,
        "reviewed_probe_sha256": expected_probe_sha256,
        "shared_resource_runner_sha256": RUNNER_SHA256,
        "seed_control_sha256": _tensor_sha(seed_control),
        "seed_control": seed_control.tolist(),
        "seed_signature_sha256": seed_branch["signature_sha256"],
        "fixed_dynamics_sha256": _tensor_sha(fixed_dynamics),
        "parameters_sha256": _tensor_sha(parameters),
        "archive_manifest_sha256": seed_probe.SEED_MANIFEST_SHA256,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "source_before": source_before, "input_before": input_before,
        "seed_raw_hashes_before": seed_raw_before,
        "seed_branch": seed_branch,
        "branch_calls": [], "trial_records": [], "linear_solves": [],
        "gradient_audits": [],
    }

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        seed_control_gradient = _gradient_audit(problem, seed_control, parameters)
        report["gradient_audits"].append({"stage": "seed", **seed_control_gradient})
        report["seed_objective"] = float(problem.objective(seed_control, parameters))
        report["seed_field_control"] = field_seed.tolist()
        report["seed_field_control_sha256"] = _tensor_sha(field_seed)
        report["seed_gradient"] = seed_control_gradient
        report["seed_field_gradient_max"] = seed_control_gradient["field_inf"]
        report["branch_calls"].append({
            "control_sha256": _tensor_sha(seed_control), "status": "positive_strict_branch",
            **seed_branch,
        })
        save()

        policy_records: dict[tuple[int, int], dict[str, Any]] = {}
        current_objective = report["seed_objective"]

        def acceptance(trial: RefinementTrial) -> tuple[bool, str] | None:
            nonlocal current_objective
            decision = objective_gate(trial)
            delta = trial.candidate_objective - trial.current_objective
            tolerance = (OBJECTIVE_ROUNDOFF_FACTOR * torch.finfo(torch.float64).eps
                         * max(abs(trial.current_objective), abs(trial.candidate_objective),
                               torch.finfo(torch.float64).tiny))
            policy_records[(trial.iteration, trial.backtrack)] = {
                "current_objective": trial.current_objective,
                "candidate_objective": trial.candidate_objective,
                "objective_increase": delta,
                "objective_roundoff_tolerance": tolerance,
                "objective_gate_passed": decision is None,
                "current_phi": 0.5 * trial.current_gradient_norm**2,
                "candidate_phi": 0.5 * trial.candidate_gradient_norm**2,
                "phi_decrease": 0.5 * (trial.current_gradient_norm**2
                                        - trial.candidate_gradient_norm**2),
                "armijo_ratio": trial.armijo_ratio,
                "normalized_slope": trial.normalized_slope,
            }
            return decision

        def observe_trial(entry: dict[str, Any]) -> None:
            nonlocal current_objective
            key = (int(entry["iteration"]), int(entry["backtrack"]))
            policy = policy_records.get(key)
            if policy is not None:
                entry["actual_objective_gate"] = policy
            candidate_values = entry.get("candidate_control")
            if isinstance(candidate_values, list) and len(candidate_values) == FIELD_SIZE:
                field = torch.tensor(candidate_values, dtype=torch.float64)
                full = _full_control(field, fixed_dynamics)
                entry["candidate_control"] = full.tolist()
                entry["candidate_control_sha256"] = _tensor_sha(full)
                if entry.get("accepted") is True:
                    audit = _gradient_audit(problem, full, parameters)
                    entry["gradient_blocks"] = audit
                    report["gradient_audits"].append({
                        "stage": "accepted", "iteration": entry["iteration"],
                        "backtrack": entry["backtrack"], **audit,
                    })
                    current_objective = float(entry["objective"])
            branch = entry.pop("branch", None)
            if isinstance(branch, dict):
                entry["branch"] = _branch_summary(
                    branch, float(branch["minimum_scaled_face_flux_margin"]),
                )
            report["trial_records"].append(entry)
            save()

        solve_index = 0
        def monitor(original: Any):
            def solve(operator: Any, rhs: Tensor, **kwargs: Any):
                nonlocal solve_index
                solve_index += 1
                calls = 0
                pcg_calls = 0
                returned = False
                entry: dict[str, Any] = {
                    "iteration": solve_index, "rtol": kwargs.get("rtol"),
                    "max_iterations": kwargs.get("max_iterations"),
                }

                def measured(direction: Tensor) -> Tensor:
                    nonlocal calls
                    calls += 1
                    return operator(direction)

                try:
                    result = original(measured, rhs, **kwargs)
                    returned = True
                    pcg_calls = calls
                    pcg_relative = float(result.relative_residual)
                    true = measured(result.solution) - rhs
                    true_relative = float(torch.linalg.vector_norm(true)) / float(
                        torch.linalg.vector_norm(rhs)
                    )
                    entry.update(
                        converged=result.converged, iterations=result.iterations,
                        reported_relative_residual=(pcg_relative
                                                    if math.isfinite(pcg_relative) else None),
                        true_relative_residual=(true_relative
                                                if math.isfinite(true_relative) else None),
                        pcg_hvp_calls=calls - 1, audit_hvp_calls=1,
                        refiner_true_residual_hvp_calls=int(result.converged),
                        measured_hvp_calls=calls,
                        total_hvp_calls=calls + int(result.converged),
                    )
                    if (not math.isfinite(true_relative)
                            or true_relative > TRUE_RESIDUAL_TOLERANCE):
                        entry["refiner_true_residual_hvp_calls"] = 0
                        entry["total_hvp_calls"] = calls
                        raise RefinementNumericalRefusal(
                            "matrix-free Newton PCG true residual exceeds tolerance"
                        )
                    return result
                except (RuntimeError, ValueError) as error:
                    entry["error"] = f"{type(error).__name__}: {error}"
                    entry["pcg_hvp_calls"] = pcg_calls if returned else calls
                    entry["audit_hvp_calls"] = calls - pcg_calls if returned else 0
                    entry["refiner_true_residual_hvp_calls"] = 0
                    entry["measured_hvp_calls"] = calls
                    entry["total_hvp_calls"] = calls
                    raise
                finally:
                    report["linear_solves"].append(entry)
                    save()
            return solve

        report["phase"] = "field_refinement"
        save()
        refiner_reported_hvp_count: int | None = None
        try:
            with matrix_free.observe_pcg_calls(monitor):
                refined = refine_stationary(
                    reduced, field_seed, parameters,
                    branch_check=_branch_check(
                        problem, fixed_dynamics, seed_branch,
                        seed_face, report["branch_calls"],
                    ),
                    max_iterations=8, max_backtracks=16,
                    pcg_max_iterations=80, trial_acceptance=acceptance,
                    trial_observer=observe_trial,
                )
            final_field = refined.control.detach().clone()
            numerical_status = "block_stationary_candidate"
            refusal = None
            iterations, hvp_count = refined.iterations, refined.hvp_count
            refiner_reported_hvp_count = refined.hvp_count
            measured_hvp_count = sum(
                int(row["total_hvp_calls"]) for row in report["linear_solves"]
            )
            if measured_hvp_count != hvp_count:
                raise RuntimeError("refinement and observed HVP costs differ")
        except RefinementNumericalRefusal as error:
            numerical_status = "field_correction_refused"
            refusal = f"{type(error).__name__}: {error}"
            accepted = [row for row in report["trial_records"] if row.get("accepted") is True]
            if accepted:
                final_field = torch.tensor(accepted[-1]["candidate_control"][:FIELD_SIZE],
                                           dtype=torch.float64)
            else:
                final_field = field_seed
            iterations = len(report["linear_solves"])
            measured_hvp_count = sum(
                int(row.get("total_hvp_calls", 0)) for row in report["linear_solves"]
            )
            hvp_count = measured_hvp_count
            refiner_reported_hvp_count = next(
                (int(row["hvp_count"]) for row in reversed(report["trial_records"])
                 if type(row.get("hvp_count")) is int),
                None,
            )
        except RefinementCallbackError:
            raise

        final_control = _full_control(final_field, fixed_dynamics)
        if (not torch.equal(final_control[FIELD_SIZE:], seed_control[FIELD_SIZE:])
                or _tensor_sha(final_control[FIELD_SIZE:]) != _tensor_sha(seed_control[FIELD_SIZE:])):
            raise RuntimeError("fixed flow/growth controls changed during field correction")
        final_branch_raw, _, final_face = preflight.branch_with_face_margin(
            problem, final_control, parameters,
        )
        final_branch = _branch_summary(final_branch_raw, final_face)
        if (final_branch["euler_stages"] != 54
                or final_branch["signature_sha256"] != seed_branch["signature_sha256"]
                or final_branch["minimum_scaled_slope_margin"] <= 0.0
                or final_face != seed_face):
            raise RuntimeError("fresh final branch does not preserve the frozen field trace")
        final_reduced_gradient = torch.func.grad(reduced, argnums=0)(final_field, parameters)
        final_full_gradient = _gradient_audit(problem, final_control, parameters)
        final_field_gradient_max = float(final_reduced_gradient.abs().max())
        reduced_stationary = final_field_gradient_max < STATIONARITY_TOLERANCE
        full_stationary = final_full_gradient["full_inf"] < STATIONARITY_TOLERANCE
        response_eligible = (
            reduced_stationary
            and final_branch["minimum_scaled_slope_margin"] > 1e-4
            and final_branch["minimum_scaled_face_flux_margin"] > 1e-4
        )
        report.update(
            phase="finished", numerical_status=numerical_status,
            refusal=refusal, iterations=iterations, hvp_count=hvp_count,
            pcg_and_observer_hvp_count=sum(
                int(row.get("pcg_hvp_calls", 0)) + int(row.get("audit_hvp_calls", 0))
                for row in report["linear_solves"]
            ),
            refiner_reported_hvp_count=refiner_reported_hvp_count,
            refiner_reported_hvp_scope=(
                "RefinementResult total" if numerical_status == "block_stationary_candidate"
                else "last trial record, null if refusal preceded a trial"
            ),
            hvp_count_scope=("refiner total; cross-checked with per-solve counts"
                             if numerical_status == "block_stationary_candidate"
                             else "per-solve PCG, observer audit, and inferred refiner residual calls"),
            final_field_control=final_field.tolist(),
            final_control=final_control.tolist(),
            final_control_sha256=_tensor_sha(final_control),
            final_fixed_dynamics_sha256=_tensor_sha(final_control[FIELD_SIZE:]),
            fixed_dynamics_byte_identical=True,
            final_objective=float(problem.objective(final_control, parameters)),
            final_branch=final_branch,
            final_reduced_gradient_max=final_field_gradient_max,
            final_full_gradient=final_full_gradient,
            final_gradient_audit={"stage": "final", **final_full_gradient},
            reduced_stationarity=reduced_stationary,
            full_stationarity=full_stationary,
            response_eligibility=("margin_supported_handoff" if response_eligible
                                  else "not_eligible"),
            execution_completion="child_completed",
        )
        report["gradient_audits"].append(report["final_gradient_audit"])
    except Exception as error:
        report.update(
            phase="execution_error", numerical_status="execution_error",
            execution_error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        report["parameters_sha256_after"] = _tensor_sha(parameters)
        report["source_after"] = _sources()
        report["input_after"] = prior._preflight_identity()
        report["seed_raw_hashes_after"] = seed_probe._seed_record_hashes()
        report["raw_unchanged"] = (
            report["seed_raw_hashes_before"] == report["seed_raw_hashes_after"]
        )
        if (report["source_before"] != report["source_after"]
                or report["input_before"] != report["input_after"]
                or report["seed_raw_hashes_before"] != report["seed_raw_hashes_after"]
                or report["parameters_sha256"] != report["parameters_sha256_after"]
                or _sha(PLAN) != expected_plan_sha256
                or _sha(Path(__file__)) != expected_probe_sha256
                or _sha(ROOT / RUNNER_PATH) != RUNNER_SHA256):
            report.update(phase="identity_recheck", numerical_status="identity_error")
        save()
    return report


def _finite(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(value)


def _finite_number(value: object) -> float | None:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value)):
        return None
    return float(value)


def _valid_resource(resource: object, command: list[str]) -> bool:
    return (isinstance(resource, dict)
            and resource.get("command") == command
            and type(resource.get("child_pid")) is int and resource["child_pid"] > 0
            and resource.get("wall_limit_seconds") == WALL_SECONDS
            and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
            and resource.get("resource_termination") is None
            and resource.get("monitor_error") is None
            and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
            and type(resource.get("sampled_peak_rss_bytes")) is int
            and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
            and _finite(resource.get("elapsed_seconds"))
            and 0 <= resource["elapsed_seconds"] <= WALL_SECONDS
            and type(resource.get("exit_code")) is int
            and resource.get("exit_code") in (0, 2))


def _valid_child(child: object, resource: dict[str, Any], source_before: dict[str, str],
                 input_before: dict[str, Any], raw_before: dict[str, str],
                 expected_probe_sha256: str) -> bool:
    if not isinstance(child, dict):
        return False
    numerical = child.get("numerical_status")
    exit_code = resource.get("exit_code")
    fixed_inputs = input_before.get("fixed_input_fields")
    input_hashes = fixed_inputs.get("tensor_sha256") if isinstance(fixed_inputs, dict) else None
    expected_parameters_sha = (
        input_hashes.get("parameters") if isinstance(input_hashes, dict) else None
    )
    seed_control_values = child.get("seed_control")
    final_control_values = child.get("final_control")
    if (not isinstance(seed_control_values, list) or len(seed_control_values) != CONTROL_SIZE
            or not isinstance(final_control_values, list) or len(final_control_values) != CONTROL_SIZE
            or not all(_finite(value) for value in seed_control_values + final_control_values)):
        return False
    seed_control = torch.tensor(seed_control_values, dtype=torch.float64)
    final_control = torch.tensor(final_control_values, dtype=torch.float64)
    final_branch = child.get("final_branch")
    seed_branch = child.get("seed_branch")
    full_gradient = child.get("final_full_gradient")
    if (not isinstance(final_branch, dict) or not isinstance(seed_branch, dict)
            or not isinstance(full_gradient, dict)):
        return False
    field_max = _finite_number(child.get("final_reduced_gradient_max"))
    full_max = _finite_number(full_gradient.get("full_inf"))
    field_inf = _finite_number(full_gradient.get("field_inf"))
    slope_margin = _finite_number(final_branch.get("minimum_scaled_slope_margin"))
    face_margin = _finite_number(final_branch.get("minimum_scaled_face_flux_margin"))
    response_eligible = (
        field_max is not None and field_max < STATIONARITY_TOLERANCE
        and slope_margin is not None and slope_margin > 1e-4
        and face_margin is not None and face_margin > 1e-4
    )
    reduced_status = field_max is not None and field_max < STATIONARITY_TOLERANCE
    full_status = full_max is not None and full_max < STATIONARITY_TOLERANCE
    field_gradient_consistent = (
        field_max is not None and field_inf is not None
        and math.isclose(field_max, field_inf, rel_tol=1e-10, abs_tol=1e-12)
    )
    linear_solves = child.get("linear_solves")
    iterations = child.get("iterations")
    if (not isinstance(linear_solves, list) or type(iterations) is not int
            or not 0 <= iterations <= 8 or len(linear_solves) != iterations):
        return False
    for expected_iteration, solve in enumerate(linear_solves, start=1):
        if (not isinstance(solve, dict) or solve.get("iteration") != expected_iteration
                or solve.get("rtol") != TRUE_RESIDUAL_TOLERANCE
                or solve.get("max_iterations") != 80):
            return False
        terminal_refusal = (numerical == "field_correction_refused"
                            and expected_iteration == len(linear_solves))
        residual = _finite_number(solve.get("true_relative_residual"))
        if numerical == "block_stationary_candidate" or not terminal_refusal:
            if (solve.get("converged") is not True or residual is None
                    or residual > TRUE_RESIDUAL_TOLERANCE):
                return False
        else:
            refusal = child.get("refusal")
            error = solve.get("error")
            if isinstance(error, str) and error.startswith("RuntimeError: "):
                reason = error.removeprefix("RuntimeError: ")
                if (reason not in KNOWN_PCG_ERRORS
                        or refusal != f"RefinementNumericalRefusal: {reason}"):
                    return False
            elif (isinstance(error, str)
                  and error == (
                      "RefinementNumericalRefusal: matrix-free Newton PCG true residual exceeds tolerance"
                  )):
                if (type(solve.get("converged")) is not bool
                        or refusal != error or residual is not None
                        and residual <= TRUE_RESIDUAL_TOLERANCE):
                    return False
            elif (refusal == (
                    "RefinementNumericalRefusal: matrix-free Newton PCG true residual exceeds tolerance"
                  ) and solve.get("converged") is True
                  and residual is not None and residual > TRUE_RESIDUAL_TOLERANCE):
                pass
            elif solve.get("converged") is False:
                if (residual is None or residual > TRUE_RESIDUAL_TOLERANCE
                        or refusal != (
                            "RefinementNumericalRefusal: matrix-free Newton PCG did not converge"
                        )):
                    return False
            elif (residual is None or residual > TRUE_RESIDUAL_TOLERANCE
                  or not isinstance(refusal, str)
                  or not refusal.startswith("RefinementNumericalRefusal: ")):
                return False
    branch_valid = (
        final_branch.get("euler_stages") == 54
        and final_branch.get("signature_sha256") == SEED_SIGNATURE_SHA256
        and seed_branch.get("signature_sha256") == SEED_SIGNATURE_SHA256
        and slope_margin is not None and slope_margin > 0.0
        and face_margin is not None and face_margin > 0.0
        and face_margin == seed_branch.get("minimum_scaled_face_flux_margin")
    )
    fixed_dynamics_valid = (
        torch.equal(seed_control[FIELD_SIZE:], final_control[FIELD_SIZE:])
        and child.get("fixed_dynamics_byte_identical") is True
        and child.get("fixed_dynamics_sha256") == _tensor_sha(seed_control[FIELD_SIZE:])
        and child.get("final_fixed_dynamics_sha256") == _tensor_sha(final_control[FIELD_SIZE:])
    )
    qualification = (
        _finite(child.get("final_objective"))
        and child.get("final_control_sha256") == _tensor_sha(final_control)
        and child.get("seed_control_sha256") == _tensor_sha(seed_control) == SEED_CONTROL_SHA256
        and child.get("reduced_stationarity") is reduced_status
        and child.get("full_stationarity") is full_status
        and field_gradient_consistent and branch_valid and fixed_dynamics_valid
        and child.get("response_eligibility") == (
            "margin_supported_handoff" if response_eligible else "not_eligible"
        )
    )
    return (
        numerical in VALID_NUMERICAL_STATUSES
        and ((numerical == "block_stationary_candidate" and exit_code == 0)
             or (numerical == "field_correction_refused" and exit_code == 2))
        and child.get("phase") == "finished"
        and child.get("pid") == resource.get("child_pid")
        and child.get("plan_sha256") == PLAN_SHA256
        and child.get("reviewed_probe_sha256") == expected_probe_sha256
        and child.get("shared_resource_runner_sha256") == RUNNER_SHA256
        and child.get("source_before") == child.get("source_after") == source_before
        and child.get("input_before") == child.get("input_after") == input_before
        and child.get("parameters_sha256") == child.get("parameters_sha256_after")
        and child.get("parameters_sha256") == expected_parameters_sha
        and qualification
        and (numerical != "block_stationary_candidate" or reduced_status)
        and (numerical != "field_correction_refused"
             or isinstance(child.get("refusal"), str)
             and child["refusal"].startswith("RefinementNumericalRefusal: "))
        and child.get("raw_unchanged") is True
        and child.get("seed_raw_hashes_before") == child.get("seed_raw_hashes_after") == raw_before
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
        and child.get("execution_completion") == "child_completed"
    )


def _execution_status(resource: object, child: object, command: list[str],
                      source_before: dict[str, str], input_before: dict[str, Any],
                      raw_before: dict[str, str], expected_probe_sha256: str) -> str:
    if isinstance(resource, dict) and resource.get("resource_termination") is not None:
        return "resource_limited"
    if (not _valid_resource(resource, command) or not isinstance(resource, dict)
            or not _valid_child(child, resource, source_before, input_before,
                                raw_before, expected_probe_sha256)):
        return "failed"
    return "completed"


def _audit_final_child(child: dict[str, Any], input_before: dict[str, Any]) -> dict[str, Any]:
    """Recompute the reported final objective, gradients, branch, and fixed block."""
    started = time.monotonic()
    passed = False
    checks: dict[str, bool] = {}
    try:
        expected_parameter_sha = input_before["fixed_input_fields"]["tensor_sha256"]["parameters"]
        seed, seed_control = seed_probe._seed_evidence()
        problem, _, parameters = _problem()
        final_control = torch.tensor(child["final_control"], dtype=torch.float64)
        final_field = final_control[:FIELD_SIZE]
        fixed_dynamics = seed_control[FIELD_SIZE:].detach().clone()
        reduced = _field_objective(problem.objective, fixed_dynamics)
        objective = float(problem.objective(final_control, parameters))
        full_gradient = _gradient_audit(problem, final_control, parameters)
        reduced_gradient = torch.func.grad(reduced, argnums=0)(final_field, parameters)
        reduced_inf = float(reduced_gradient.abs().max())
        branch_raw, _, face = preflight.branch_with_face_margin(
            problem, final_control, parameters,
        )
        branch = _branch_summary(branch_raw, face)
        reported = child["final_full_gradient"]
        reported_branch = child["final_branch"]
        tolerance = 1e-10

        def close(left: object, right: object) -> bool:
            left_value, right_value = _finite_number(left), _finite_number(right)
            return (left_value is not None and right_value is not None
                    and math.isclose(left_value, right_value,
                                     rel_tol=1e-10, abs_tol=1e-12))

        checks = {
            "parameters_match_preflight": _tensor_sha(parameters) == expected_parameter_sha,
            "seed_control_matches_archive": (
                _tensor_sha(seed_control) == SEED_CONTROL_SHA256
                and child["seed_control_sha256"] == _tensor_sha(seed_control)
            ),
            "final_control_hash_matches": child["final_control_sha256"]
            == _tensor_sha(final_control),
            "fixed_dynamics_match_seed": torch.equal(
                final_control[FIELD_SIZE:], seed_control[FIELD_SIZE:]
            ),
            "objective_matches": close(child["final_objective"], objective),
            "reduced_gradient_matches_full_field_block": close(
                child["final_reduced_gradient_max"], reduced_inf,
            ) and close(reported.get("field_inf"), full_gradient["field_inf"]),
            "full_gradient_matches": all(close(reported.get(key), full_gradient[key])
                                          for key in (
                                              "field_l2", "field_inf", "dynamics_l2",
                                              "full_l2", "full_inf",
                                          )) and isinstance(reported.get("dynamics_components"), list)
            and len(reported["dynamics_components"]) == 6
            and all(close(value, expected) for value, expected in zip(
                reported["dynamics_components"], full_gradient["dynamics_components"], strict=True
            )),
            "stationarity_flags_match": (
                child["reduced_stationarity"] is (reduced_inf < tolerance)
                and child["full_stationarity"] is (
                    full_gradient["full_inf"] < tolerance
                )
            ),
            "final_branch_matches": (
                branch["euler_stages"] == 54
                and branch["signature_sha256"] == SEED_SIGNATURE_SHA256
                and branch["signature_sha256"] == reported_branch["signature_sha256"]
                and close(branch["minimum_scaled_slope_margin"],
                          reported_branch["minimum_scaled_slope_margin"])
                and close(face, reported_branch["minimum_scaled_face_flux_margin"])
                and face == seed["seed_branch"]["minimum_scaled_face_flux_margin"]
            ),
        }
        response = (
            "margin_supported_handoff"
            if reduced_inf < tolerance
            and branch["minimum_scaled_slope_margin"] > 1e-4
            and face > 1e-4 else "not_eligible"
        )
        checks["response_eligibility_matches"] = child["response_eligibility"] == response
        passed = all(checks.values())
    except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError, IndexError):
        passed = False
    return {
        "passed": passed, "checks": checks,
        "scope": "one final J, reduced/full gradient, fixed-control, and 54-stage branch audit; no solver or PCG",
        "elapsed_seconds": time.monotonic() - started,
    }


def run(directory: Path, *, expected_plan_sha256: str,
        expected_probe_sha256: str) -> dict[str, Any]:
    archives = (seed_probe.SEED_ARCHIVE.resolve(strict=True),
                seed_probe.seed_gate.ARCHIVE.resolve(strict=True))
    if (directory.is_symlink() or directory.exists()
            or directory.resolve(strict=False).is_relative_to(archives[0])
            or directory.resolve(strict=False).is_relative_to(archives[1])):
        raise ValueError("field correction directory must be fresh and outside raw archives")
    if (_sha(PLAN) != expected_plan_sha256 or expected_plan_sha256 != PLAN_SHA256
            or _sha(Path(__file__)) != expected_probe_sha256
            or _sha(ROOT / RUNNER_PATH) != RUNNER_SHA256):
        raise _SeedIdentityError("reviewed field correction plan, probe, or resource source changed")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "field_correction.json"
    resource_path = directory / "field_correction.resource.json"
    log_path = directory / "field_correction.log"
    run_path = directory / "field_correction.run.json"
    if any(path.exists() or path.is_symlink() for path in (output, resource_path, log_path, run_path)):
        raise ValueError("field correction attempt outputs must be fresh")
    source_before = _sources()
    input_before = prior._preflight_identity()
    raw_before = seed_probe._seed_record_hashes()
    command = [
        sys.executable, str(Path(__file__)), "--child", "--output", str(output),
        "--expected-plan-sha256", expected_plan_sha256,
        "--expected-probe-sha256", expected_probe_sha256,
    ]
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=resource_path, log_path=log_path,
    )
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError):
        child = None
    execution_status = _execution_status(
        resource, child, command, source_before, input_before, raw_before,
        expected_probe_sha256,
    )
    final_audit = None
    if execution_status == "completed" and isinstance(child, dict):
        final_audit = _audit_final_child(child, input_before)
        if not final_audit["passed"]:
            execution_status = "failed"
    source_after = _sources()
    input_after = prior._preflight_identity()
    raw_after = seed_probe._seed_record_hashes()
    parent_identity_unchanged = (
        source_before == source_after and input_before == input_after
        and raw_before == raw_after and _sha(PLAN) == expected_plan_sha256
        and _sha(Path(__file__)) == expected_probe_sha256
        and _sha(ROOT / RUNNER_PATH) == RUNNER_SHA256
    )
    if not parent_identity_unchanged:
        execution_status = "failed"
    result = {
        "execution_status": execution_status,
        "numerical_status": child.get("numerical_status", "not_reached")
        if isinstance(child, dict) else "not_reached",
        "reduced_stationarity": child.get("reduced_stationarity")
        if isinstance(child, dict) else None,
        "full_stationarity": child.get("full_stationarity")
        if isinstance(child, dict) else None,
        "response_eligibility": child.get("response_eligibility", "not_eligible")
        if isinstance(child, dict) else "not_eligible",
        "parent_identity_unchanged": parent_identity_unchanged,
        "parent_final_audit": final_audit,
        "source_before": source_before, "source_after": source_after,
        "input_before": input_before, "input_after": input_after,
        "raw_hashes_before": raw_before, "raw_hashes_after": raw_after,
        "resource": resource,
        "child": child,
    }
    run_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    args = parser.parse_args()
    if args.child:
        if args.output is None or args.directory is not None:
            parser.error("--child requires --output and forbids --directory")
        result = _run_child(
            args.output, expected_plan_sha256=args.expected_plan_sha256,
            expected_probe_sha256=args.expected_probe_sha256,
        )
        print(json.dumps({key: result.get(key) for key in (
            "phase", "numerical_status", "reduced_stationarity",
            "full_stationarity", "response_eligibility", "refusal",
        )}, sort_keys=True))
        return 0 if result["numerical_status"] == "block_stationary_candidate" else 2
    if args.directory is None or args.output is not None:
        parser.error("parent mode requires --directory and forbids --output")
    result = run(args.directory,
                 expected_plan_sha256=args.expected_plan_sha256,
                 expected_probe_sha256=args.expected_probe_sha256)
    print(json.dumps({key: result.get(key) for key in (
        "execution_status", "numerical_status", "reduced_stationarity",
        "full_stationarity", "response_eligibility",
    )}, sort_keys=True))
    return 0 if result["execution_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
