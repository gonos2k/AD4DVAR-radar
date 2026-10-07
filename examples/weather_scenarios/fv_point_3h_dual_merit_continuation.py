"""Bounded repeated dual-merit correction from a pinned accepted point.

The original objective and fixed parameters remain unchanged. Each accepted
point gets a fresh live-HVP PCG direction; line-search trials reuse that
direction and its single fresh ``H s`` action. This runner records optimizer
progress only and does not certify a stationary point or forecast.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from advar import matrix_free
from examples.weather_scenarios import fv_point_3h_accepted_curvature as curvature
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as search
from examples.weather_scenarios import fv_point_3h_current_newton_step as diagnostics
from examples.weather_scenarios import fv_point_3h_dual_merit_step as dual
from examples.weather_scenarios import fv_point_3h_hvp_newton_step as live_step
from examples.weather_scenarios import fv_point_3h_schur_newton_step as block_step
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

ROOT = curvature.ROOT
EVIDENCE = curvature.EVIDENCE
PLAN = EVIDENCE / "E0B_DUAL_MERIT_CONTINUATION_PLAN_20261006.json"
BASE = EVIDENCE / "a35_dual_merit_20261006_attempt1/step.json"
SELF = "examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py"
TEST = "tests/test_fv_point_3h_dual_merit_continuation.py"
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 720.0, 780.0, 1024**3
MAX_ITERATIONS, MAX_HVP, PCG_MAX, PCG_RTOL = 3, 90, 40, 1e-10
MAX_CANDIDATES, RADIUS, C1_J, C1_PHI = 16, 0.05, 1e-4, 1e-4
ROOT_GRADIENT_INF = 1e-10
EPS = torch.finfo(torch.float64).eps


class ContinuationRefusal(search.StepRefusal):
    """Expected numerical, policy, or cooperative-budget stop."""

    def __init__(self, message: str, status: str = "step_refusal") -> None:
        super().__init__(message)
        self.status = status


def _policy(mode: str = "strict", max_iterations: int = MAX_ITERATIONS) -> dict[str, Any]:
    if mode not in {"strict", "inexact"}:
        raise ValueError("linear_mode must be 'strict' or 'inexact'")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError(f"max_iterations must be in [1, {MAX_ITERATIONS}]")
    result = {"max_iterations": max_iterations, "max_hvp_calls": MAX_HVP,
            "pcg_max_iterations": PCG_MAX, "pcg_relative_tolerance": PCG_RTOL,
            "max_candidates": MAX_CANDIDATES, "radius": RADIUS,
            "j_armijo_c1": C1_J, "phi_armijo_c1": C1_PHI,
            "root_gradient_inf": ROOT_GRADIENT_INF,
            "internal_seconds": INTERNAL_SECONDS, "outer_seconds": WALL_SECONDS,
            "rss_bytes": RSS_BYTES, "guarded_launches": 1,
            "cached_preconditioner_scope": "f82c SPD block inverse preconditioner only",
            "hvp_cap_includes": "all PCG products and independently recomputed true-residual Hs; 3 solves are not guaranteed"}
    if mode == "inexact":
        result.update(linear_mode="inexact", forcing_tolerance_min=PCG_RTOL,
                      forcing_tolerance_max=1e-3, forcing_tolerance_scale=0.1,
                      require_resolved_descent_for_relaxed_tolerance=True,
                      forcing_tolerance_schedule="clip(0.1 * current_gradient_inf, [1e-10, 1e-3]); heuristic, not Eisenstat-Walker")
    return result


def policy_dict(mode: str = "strict", max_iterations: int = MAX_ITERATIONS) -> dict[str, Any]:
    """Public frozen policy for the evidence plan and review checklist."""
    return _policy(mode, max_iterations)



def forcing_tolerance(gradient_inf: float, mode: str = "strict") -> float:
    """Return the strict tolerance or bounded current-gradient heuristic."""
    if mode == "strict":
        return PCG_RTOL
    if mode != "inexact" or not math.isfinite(gradient_inf) or gradient_inf < 0:
        raise ValueError("invalid linear mode or gradient scale")
    return min(1e-3, max(PCG_RTOL, 0.1 * gradient_inf))

def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise search.BudgetRefusal("720-second internal continuation budget exhausted")


def execution_status(resource: dict[str, Any]) -> str:
    """Classify this runner's 780-second receipt, independently of old runners."""
    elapsed = resource.get("elapsed_seconds")
    peak = resource.get("sampled_peak_rss_bytes")
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (float, int))
            or not math.isfinite(elapsed) or elapsed < 0
            or isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
            or resource.get("wall_limit_seconds") != WALL_SECONDS
            or resource.get("rss_limit_bytes") != RSS_BYTES):
        return "failed"
    if (resource.get("received_sigterm") or resource.get("monitor_error")
            or resource.get("child_process_group_cleanup_error")
            or resource.get("child_process_group_cleanup_sent") is not False
            or resource.get("resource_termination") not in {None, "wall_time_limit", "rss_limit"}):
        return "failed"
    if (resource.get("resource_termination") in {"wall_time_limit", "rss_limit"}
            or elapsed > WALL_SECONDS or peak > RSS_BYTES):
        return "resource_limited"
    return "completed" if resource.get("exit_code") == 0 else "failed"


def _finite_vector(value: Any, name: str) -> Tensor:
    if (not isinstance(value, Tensor) or value.shape != (26,) or value.dtype != torch.float64
            or value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be finite CPU FP64 length 26")
    return value


def _strict_branch(branch: dict[str, Any], margins: dict[str, Any]) -> bool:
    return (branch.get("status") == "passed_strict_branch" and branch.get("euler_stages") == 3600
            and branch.get("choice_stage_count") == 3600 and branch.get("face_sign_stage_count") == 3600
            and dual.merit.seed_linear._valid_margins(margins, complete=True))


def _true_merit(control: Tensor, parameters: Tensor, objective: Callable[[Tensor, Tensor], Tensor],
                gradient_fn: Callable[[Tensor, Tensor], Tensor], deadline: float) -> tuple[Tensor, Tensor, Tensor]:
    _check_deadline(deadline)
    value = objective(control.clone(), parameters.clone())
    _check_deadline(deadline)
    gradient = gradient_fn(control.clone(), parameters.clone())
    _check_deadline(deadline)
    if (not isinstance(value, Tensor) or value.shape != () or value.dtype != torch.float64
            or value.device.type != "cpu" or not bool(torch.isfinite(value))):
        raise ContinuationRefusal("current original J is not finite CPU FP64 scalar")
    _finite_vector(gradient, "current full gradient")
    phi = gradient @ gradient / 2
    if not bool(torch.isfinite(phi)):
        raise ContinuationRefusal("current Phi is nonfinite")
    return value, gradient, phi


def _solve_direction(control: Tensor, parameters: Tensor, gradient: Tensor,
                     hvp_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
                     preconditioner: Callable[[Tensor], Tensor], deadline: float,
                     counts: dict[str, int], *, rtol: float = PCG_RTOL) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Use the existing matrix-free PCG recurrence with the continuation cap."""
    def operator(vector: Tensor) -> Tensor:
        _check_deadline(deadline)
        if counts["hvp_calls"] >= MAX_HVP:
            raise search.BudgetRefusal(f"continuation live HVP cap {MAX_HVP} reached")
        counts["hvp_calls"] += 1
        result = hvp_fn(control.clone(), parameters.clone(), vector.clone())
        counts["hvp_calls_completed"] += 1
        _check_deadline(deadline)
        return _finite_vector(result, "live current-point HVP")

    def checked_preconditioner(vector: Tensor) -> Tensor:
        try:
            return preconditioner(vector.clone())
        except block_step.StepRefusal as error:
            raise ContinuationRefusal(f"cached f82c preconditioner refusal: {error}",
                                      "preconditioner_refusal") from error

    try:
        solve = matrix_free.pcg(operator, -gradient, preconditioner=checked_preconditioner,
                                rtol=rtol, max_iterations=PCG_MAX)
    except RuntimeError as error:
        known = {"operator must be symmetric positive definite", "preconditioner must be positive definite",
                 "PCG step is not finite", "PCG direction update is not finite",
                 "residual norm is not finite", "true residual norm is not finite"}
        if str(error) in known:
            raise ContinuationRefusal(f"PCG numerical refusal: {error}", "linear_solve_refusal") from error
        raise
    direction = _finite_vector(solve.solution, "PCG direction")
    # Recompute Hs at the accepted current point. The same product supplies
    # both the true residual and the Phi directional slope for this search.
    hs = operator(direction)
    residual = hs + gradient
    g_norm = torch.linalg.vector_norm(gradient)
    relative = float(torch.linalg.vector_norm(residual) / g_norm) if bool(g_norm > 0) else math.inf
    g_dot_s, g_dot_hs = float(gradient @ direction), float(gradient @ hs)
    curvature_value = float(direction @ hs)
    curvature_roundoff = 128 * EPS * float(torch.sum(torch.abs(direction * hs)))
    slope_roundoff = 128 * EPS * float(torch.sum(torch.abs(gradient * direction)))
    phi_slope_roundoff = 128 * EPS * float(torch.sum(torch.abs(gradient * hs)))
    audit = {"converged": bool(solve.converged), "iterations": int(solve.iterations),
             "pcg_relative_residual": float(solve.relative_residual),
             "true_relative_residual": relative, "g_dot_s": g_dot_s,
             "g_dot_Hs": g_dot_hs, "directional_curvature_s_H_s": curvature_value,
             "curvature_roundoff_budget": curvature_roundoff,
             "g_dot_s_roundoff_budget": slope_roundoff,
             "g_dot_Hs_roundoff_budget": phi_slope_roundoff,
             "rtol": rtol, "strict_rtol_floor": PCG_RTOL, "max_iterations": PCG_MAX,
             "rho": relative,
             "hvp_calls_including_true_residual": None}
    if (not solve.converged or not math.isfinite(relative) or relative > rtol
            or not math.isfinite(g_dot_s) or not math.isfinite(g_dot_hs)
            or not math.isfinite(curvature_value)):
        raise ContinuationRefusal("PCG convergence or true-residual gate failed", "linear_solve_refusal")
    if not curvature_value > curvature_roundoff:
        raise ContinuationRefusal("observed direction curvature is nonpositive or unresolved", "curvature_refusal")
    if (not g_dot_s < 0 or not g_dot_hs < 0
            or (rtol > PCG_RTOL
                and (not g_dot_s < -slope_roundoff or not g_dot_hs < -phi_slope_roundoff))):
        raise ContinuationRefusal("current Newton direction lacks strict descent in both J and Phi",
                                  "dual_slope_refusal")
    return direction, hs, audit


def run_iterations(control: Tensor, parameters: Tensor,
                   objective: Callable[[Tensor, Tensor], Tensor],
                   gradient_fn: Callable[[Tensor, Tensor], Tensor],
                   branch_fn: Callable[[Tensor, Tensor], tuple[dict[str, Any], dict[str, Any]]],
                   hvp_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
                   preconditioner: Callable[[Tensor], Tensor], deadline: float,
                   *, max_iterations: int = MAX_ITERATIONS,
                   linear_mode: str = "strict",
                   shared_counts: dict[str, int] | None = None,
                   initial_metadata: dict[str, Any] | None = None,
                   commit_candidate: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                               dict[str, Any], float], dict[str, Any]] | None = None,
                   on_update: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run the bounded numerical loop with injectable callbacks for small tests.

    ``commit_candidate`` performs point-specific diagnostics and integrity
    closure. Its return value is attached to an iteration receipt; state is
    advanced only after it and the deadline check succeed.
    """
    started = time.monotonic()
    c = _finite_vector(control, "starting control").clone()
    if (not isinstance(parameters, Tensor) or parameters.shape != (13,) or parameters.dtype != torch.float64
            or parameters.device.type != "cpu" or not bool(torch.isfinite(parameters).all())):
        raise ValueError("parameters must be finite CPU FP64 length 13")
    actual_policy = _policy(linear_mode, max_iterations)
    counts = shared_counts if shared_counts is not None else {}
    defaults = {"hvp_calls": 0, "hvp_calls_completed": 0,
                "pcg_iterations_completed": 0, "pcg_solves_started": 0}
    for key, default in defaults.items():
        value = counts.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"shared_counts[{key!r}] must be a nonnegative integer")
    if counts["hvp_calls"] > MAX_HVP or counts["hvp_calls_completed"] > counts["hvp_calls"]:
        raise ValueError("shared HVP counters exceed or contradict the global cap")
    record: dict[str, Any] = {"phase": "running", "execution_status": "completed",
        "numerical_status": "not_reached", "optimizer_steps_applied": 0,
        "accepted_control": c.tolist(), "accepted_control_sha256": curvature.tensor_sha(c),
        "parameters_sha256": curvature.tensor_sha(parameters), "iterations": [], "trials": [],
        **counts, "policy": actual_policy, "linear_mode": linear_mode, "full_root_claim": False,
        "eligible_stationary_point": False, "accepted_point_curvature": "not_computed",
        "response_computed": False, "forecast_score_computed": False,
        "scope": "bounded repeated original-J dual-merit correction; no final root, adjoint, or forecast claim"}
    if initial_metadata:
        record.update(initial_metadata)
    # Input metadata describes provenance only; the executed policy is derived
    # from this invocation's actual keyword arguments.
    record["policy"] = actual_policy
    record["linear_mode"] = linear_mode

    def persist() -> None:
        record.update(counts)
        record["accepted_control"] = c.tolist()
        record["accepted_control_sha256"] = curvature.tensor_sha(c)
        if on_update is not None:
            on_update(record)

    persist()
    try:
        for iteration_index in range(max_iterations):
            _check_deadline(deadline)
            base_j, gradient, base_phi = _true_merit(c, parameters, objective, gradient_fn, deadline)
            _check_deadline(deadline)
            branch, margins = branch_fn(c.clone(), parameters.clone())
            _check_deadline(deadline)
            if not _strict_branch(branch, margins):
                raise ContinuationRefusal("current point no longer passes its complete strict branch gate",
                                          "branch_refusal")
            gradient_inf = float(gradient.abs().max())
            state = {"objective": float(base_j), "phi": float(base_phi),
                     "gradient": gradient.tolist(), "gradient_l2": float(gradient.norm()),
                     "gradient_inf": gradient_inf, "branch": branch, "margins": margins}
            if gradient_inf <= ROOT_GRADIENT_INF:
                record.update(numerical_status="root_pending_audit", current_state=state,
                              root_pending_audit=True)
                persist()
                break
            start_hvp = counts["hvp_calls"]
            iteration: dict[str, Any] = {"iteration": iteration_index + 1,
                "base_control_sha256": curvature.tensor_sha(c), "base_state": state,
                "trials": [], "status": "linear_solve",
                "hvp_calls_before_solve": counts["hvp_calls"],
                "pcg_iterations_failed_solve": None}
            record["current_iteration"] = iteration
            counts["pcg_solves_started"] += 1
            persist()
            rtol = forcing_tolerance(gradient_inf, linear_mode)
            direction, hs, solve = _solve_direction(c, parameters, gradient, hvp_fn,
                preconditioner, deadline, counts, rtol=rtol)
            solve["linear_mode"] = linear_mode
            solve["forcing_tolerance"] = rtol
            counts["pcg_iterations_completed"] += solve["iterations"]
            solve["hvp_calls_including_true_residual"] = counts["hvp_calls"] - start_hvp
            iteration.update(direction=direction.tolist(), H_s=hs.tolist(), solve=solve,
                             status="direction_ready")
            persist()

            def save_trial(row: dict[str, Any]) -> None:
                row["iteration"] = iteration_index + 1
                iteration["trials"].append(row)
                record["trials"].append(row)
                persist()

            def safe_callback(callback: Callable[..., Any]) -> Callable[..., Any]:
                return lambda x, p: callback(x.clone(), p.clone())

            accepted, trials, status = dual.dual_merit_search(
                c.clone(), direction.clone(), base_j, gradient.clone(), base_phi, hs.clone(),
                safe_callback(objective), safe_callback(gradient_fn), parameters.clone(),
                safe_callback(branch_fn), deadline, branch["signature_sha256"], on_trial=save_trial)
            iteration["search_status"] = status
            if accepted is None:
                iteration["status"] = status
                record.update(numerical_status=status, current_iteration=iteration)
                persist()
                break
            candidate_row = trials[-1]
            displacement = accepted - c
            delta_norm = float(torch.linalg.vector_norm(displacement))
            scale = max(1.0, float(torch.linalg.vector_norm(c)))
            candidate_gradient_inf = float(
                torch.tensor(candidate_row["gradient"], dtype=torch.float64).abs().max())
            root_candidate = candidate_gradient_inf <= ROOT_GRADIENT_INF
            if root_candidate:
                # Confirm the pending-root path with an independent fresh
                # evaluation: the search receipt alone must not establish it.
                fresh_j, fresh_gradient, fresh_phi = _true_merit(
                    accepted, parameters, objective, gradient_fn, deadline)
                fresh_branch, fresh_margins = branch_fn(accepted.clone(), parameters.clone())
                _check_deadline(deadline)
                saved_gradient = torch.tensor(candidate_row["gradient"], dtype=torch.float64)
                scalar_consistent = all(
                    abs(float(fresh) - float(saved)) <= 128 * EPS * max(
                        abs(float(fresh)), abs(float(saved)), torch.finfo(torch.float64).tiny)
                    for fresh, saved in ((fresh_j, candidate_row["objective"]),
                                         (fresh_phi, candidate_row["phi"])))
                gradient_consistent = bool(torch.max(torch.abs(fresh_gradient - saved_gradient)) <=
                    128 * EPS * max(torch.finfo(torch.float64).tiny,
                                    float(fresh_gradient.abs().max()),
                                    float(saved_gradient.abs().max())))
                fresh_gradient_inf = float(fresh_gradient.abs().max())
                fresh_branch_valid = (_strict_branch(fresh_branch, fresh_margins)
                    and fresh_branch.get("signature_sha256") ==
                        candidate_row["branch"].get("signature_sha256"))
                fresh_armijo = (float(fresh_j) <= candidate_row["J_armijo_threshold"]
                    and float(fresh_phi) <= candidate_row["Phi_armijo_threshold"])
                if (not scalar_consistent or not gradient_consistent or not fresh_branch_valid
                        or not fresh_armijo or fresh_gradient_inf > ROOT_GRADIENT_INF):
                    _mark_current_candidate_not_committed(record,
                        "fresh root candidate re-evaluation changed or failed a root gate")
                    iteration.update(status="root_candidate_recheck_failed",
                        candidate_not_committed=True,
                        fresh_gradient_inf=fresh_gradient_inf,
                        fresh_objective=float(fresh_j), fresh_phi=float(fresh_phi),
                        recheck_objective_consistent=scalar_consistent,
                        recheck_gradient_consistent=gradient_consistent,
                        recheck_strict_branch_passed=fresh_branch_valid,
                        recheck_dual_armijo_passed=fresh_armijo)
                    record.update(numerical_status="root_candidate_recheck_failed",
                                  current_iteration=iteration)
                    persist()
                    break
                # Use the confirmed values in the commit receipt.
                candidate_row.update(objective=float(fresh_j),
                    gradient=fresh_gradient.tolist(), gradient_l2=float(fresh_gradient.norm()),
                    gradient_blocks=block_step.frozen.gradient_blocks(fresh_gradient),
                    phi=float(fresh_phi),
                    branch=fresh_branch, margins=fresh_margins)
                candidate_gradient_inf = fresh_gradient_inf
            # Large positive curvature can turn a roundoff-size move into a
            # valid root, so only the independently confirmed root bypasses this guard.
            if delta_norm <= 128 * EPS * scale and not root_candidate:
                _mark_current_candidate_not_committed(record, "candidate displacement is numerically stagnant")
                iteration.update(status="displacement_stagnation", candidate_not_committed=True)
                record.update(numerical_status="displacement_stagnation", current_iteration=iteration)
                persist()
                break
            base_j_delta = float(base_j) - candidate_row["objective"]
            base_phi_delta = float(base_phi) - candidate_row["phi"]
            j_floor = 128 * EPS * max(abs(float(base_j)), abs(candidate_row["objective"]), torch.finfo(torch.float64).tiny)
            phi_floor = 128 * EPS * max(abs(float(base_phi)), abs(candidate_row["phi"]), torch.finfo(torch.float64).tiny)
            # Both actual Armijo checks remain mandatory; only subtraction-
            # scale decrease floors are skipped for a confirmed root.
            if not root_candidate and (base_j_delta <= j_floor or base_phi_delta <= phi_floor):
                _mark_current_candidate_not_committed(record, "accepted decrease is numerically zero")
                iteration.update(status="numerically_zero_decrease", candidate_not_committed=True,
                                 J_reduction=base_j_delta, Phi_reduction=base_phi_delta)
                record.update(numerical_status="numerically_zero_decrease", current_iteration=iteration)
                persist()
                break
            _check_deadline(deadline)
            closure = commit_candidate(c.clone(), accepted.clone(), parameters.clone(), base_j.clone(),
                gradient.clone(), hs.clone(), candidate_row, deadline) if commit_candidate else {}
            if not isinstance(closure, dict):
                raise TypeError("commit_candidate must return a dict of completed diagnostics")
            _check_deadline(deadline)
            iteration.update(status="accepted", accepted_control=accepted.tolist(),
                accepted_control_sha256=curvature.tensor_sha(accepted),
                accepted_objective=candidate_row["objective"], accepted_phi=candidate_row["phi"],
                accepted_gradient=candidate_row["gradient"],
                accepted_gradient_inf=candidate_gradient_inf,
                root_candidate_recheck_passed=root_candidate,
                root_candidate_recheck_gradient_inf=candidate_gradient_inf if root_candidate else None,
                J_reduction=base_j_delta, Phi_reduction=base_phi_delta,
                displacement_l2=delta_norm,
                displacement_below_roundoff=delta_norm <= 128 * EPS * scale,
                diagnostics=closure)
            # Commit only after all candidate diagnostics, integrity checks,
            # and deadline checks have completed successfully.
            c = accepted
            record["iterations"].append(iteration)
            record.pop("current_iteration", None)
            record["optimizer_steps_applied"] += 1
            record.update(numerical_status="iteration_accepted",
                          current_state={"objective": candidate_row["objective"],
                              "phi": candidate_row["phi"], "gradient": candidate_row["gradient"],
                              "gradient_inf": iteration["accepted_gradient_inf"],
                              "branch": candidate_row["branch"], "margins": candidate_row["margins"]})
            persist()
            if iteration["accepted_gradient_inf"] <= ROOT_GRADIENT_INF:
                record.update(numerical_status="root_pending_audit", root_pending_audit=True)
                break
        else:
            record["numerical_status"] = "iteration_limit"
        if record["numerical_status"] == "iteration_accepted":
            record["numerical_status"] = "iteration_limit"
    except search.StepRefusal as error:
        if record.get("current_iteration", {}).get("trials"):
            _mark_current_candidate_not_committed(record, str(error))
        record.update(numerical_status="budget_refusal" if isinstance(error, search.BudgetRefusal)
                      else getattr(error, "status", "step_refusal"), refusal=str(error))
        if record.get("current_iteration", {}).get("status") == "linear_solve":
            record["current_iteration"]["pcg_iterations_failed_solve"] = "not_recorded"
            record["current_iteration"]["hvp_calls_at_refusal"] = counts["hvp_calls"]
        if record.get("current_iteration", {}).get("trials", [])[-1:]:
            record["current_iteration"]["status"] = record["numerical_status"]
    except Exception as error:
        solve_in_progress = record.get("current_iteration", {}).get("status") == "linear_solve"
        if record.get("current_iteration", {}).get("trials"):
            _mark_current_candidate_not_committed(record, f"{type(error).__name__}: {error}")
        record.update(phase="execution_error", execution_status="failed",
                      numerical_status="execution_error", failure=f"{type(error).__name__}: {error}")
        if record.get("current_iteration"):
            if solve_in_progress:
                record["current_iteration"]["pcg_iterations_failed_solve"] = "not_recorded"
                record["current_iteration"]["hvp_calls_at_refusal"] = counts["hvp_calls"]
            record["current_iteration"]["status"] = "execution_error"
    record.update(phase="finished" if record["execution_status"] == "completed" else "execution_error",
                  elapsed_seconds=time.monotonic() - started)
    persist()
    return record


def _mark_current_candidate_not_committed(record: dict[str, Any], reason: str) -> None:
    """Clear only this iteration's provisional row; keep earlier commits intact."""
    iteration = record.get("current_iteration")
    trials = iteration.get("trials", []) if isinstance(iteration, dict) else []
    if trials and trials[-1].get("status") == "accepted":
        trials[-1].update(status="candidate_not_committed", accepted=False,
                          commit_refusal=reason)


def _copy_partial_child_counts(parent: dict[str, Any], child_path: Path) -> None:
    """Copy trustworthy progress only from a JSON object child receipt."""
    if not child_path.exists():
        return
    try:
        partial = json.loads(child_path.read_text())
    except (OSError, ValueError):
        return
    if isinstance(partial, dict):
        parent.update(completed_iterations=partial.get("optimizer_steps_applied", 0),
                      hvp_calls=partial.get("hvp_calls", 0),
                      numerical_status=partial.get("numerical_status", "partial_receipt"))


def _archive_names(base_plan: dict[str, Any]) -> set[str]:
    return (set(base_plan["archive_files"]) |
            {BASE.relative_to(ROOT).as_posix(), BASE.with_suffix(".run.json").relative_to(ROOT).as_posix(),
             BASE.with_suffix(".resource.json").relative_to(ROOT).as_posix(),
             dual.PLAN.relative_to(ROOT).as_posix(), dual.OLD_CURVATURE.relative_to(ROOT).as_posix(),
             live_step.CHECKPOINT.relative_to(ROOT).as_posix(),
             live_step.CURVATURE_PARENT.relative_to(ROOT).as_posix(),
             live_step.CURVATURE_RESOURCE.relative_to(ROOT).as_posix()})


def load_base(plan_path: Path, plan_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not plan_path.resolve().is_relative_to(ROOT) or curvature.sha(plan_path) != plan_sha:
        raise ValueError("continuation plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    base_plan, direction_raw = dual.load_base(dual.PLAN, curvature.sha(dual.PLAN))
    accepted_raw = json.loads(BASE.read_text())
    parent = json.loads(BASE.with_suffix(".run.json").read_text())
    resource = json.loads(BASE.with_suffix(".resource.json").read_text())
    curvature_audit = json.loads(live_step.CURVATURE.read_text())
    checkpoint = json.loads(live_step.CHECKPOINT.read_text())
    curvature_parent = json.loads(live_step.CURVATURE_PARENT.read_text())
    curvature_resource = json.loads(live_step.CURVATURE_RESOURCE.read_text())
    required_sources = set(base_plan["source_files"]) | {SELF, TEST,
        "examples/weather_scenarios/fv_point_3h_dual_merit_step.py",
        "examples/weather_scenarios/fv_point_3h_hvp_newton_step.py",
        "examples/weather_scenarios/fv_point_3h_schur_newton_step.py",
        "src/advar/matrix_free.py"}
    required_archives = _archive_names(base_plan)
    if (not isinstance(plan.get("source_files"), dict) or not required_sources <= set(plan["source_files"])
            or not isinstance(plan.get("archive_files"), dict) or not required_archives <= set(plan["archive_files"])
            or plan.get("policy") != _policy()
            or plan.get("accepted_control_sha256") != accepted_raw.get("accepted_control_sha256")
            or plan.get("parameters_sha256") != direction_raw.get("parameters_sha256")):
        raise ValueError("continuation plan must pin the e0b04a inputs, numerical sources, and fixed policy")
    for name, digest in {**plan["source_files"], **plan["archive_files"]}.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or (ROOT / name).is_symlink() or curvature.sha(path) != digest:
            raise ValueError(f"continuation input/source pin changed: {name}")
    diagnostics._require_cached_objective_sources(plan, curvature_audit)
    h = curvature_audit.get("curvature", {})
    if (curvature_audit.get("phase") != "finished"
            or curvature_audit.get("numerical_status") != "curvature_completed"
            or curvature_audit.get("source_unchanged") is not True
            or curvature_audit.get("fixed_input_unchanged") is not True
            or curvature_audit.get("source_before") != curvature_audit.get("source_after")
            or curvature_parent.get("execution_status") != "completed"
            or curvature_parent.get("child_read_error") is not None
            or curvature_parent.get("child_sha256") != curvature.sha(live_step.CURVATURE)
            or search.execution_status(curvature_resource) != "completed"
            or checkpoint.get("status") != "completed" or h.get("hvp_columns") != 26
            or h.get("hvp_calls_total") != 27
            or not h.get("minimum_eigenpair_audit", {}).get("passed")
            or h.get("hessian") != checkpoint.get("final_result", {}).get("hessian")):
        raise ValueError("archived f82c curvature/checkpoint/parent/resource receipts are incomplete")
    accepted_trials = [row for row in accepted_raw.get("trials", []) if row.get("status") == "accepted"]
    trial = accepted_trials[0] if len(accepted_trials) == 1 else None
    if (accepted_raw.get("phase") != "finished" or accepted_raw.get("numerical_status") != "one_dual_merit_step_accepted"
            or accepted_raw.get("optimizer_steps_applied") != 1 or accepted_raw.get("fixed_input_unchanged") is not True
            or accepted_raw.get("source_unchanged") is not True
            or accepted_raw.get("source_before") != accepted_raw.get("source_after")
            or accepted_raw.get("runtime") != accepted_raw.get("runtime_after")
            or accepted_raw.get("plan_sha256") != curvature.sha(dual.PLAN)
            or accepted_raw.get("accepted_control_sha256") != "e0b04a4dc019cea2664af811e7cd956453dd6b34c9d2f7a35bff171fdc57132a"
            or accepted_raw.get("base_control_sha256") != direction_raw.get("base_control_sha256")
            or not trial or trial.get("control_sha256") != accepted_raw.get("accepted_control_sha256")
            or trial.get("J_armijo_passed") is not True or trial.get("Phi_armijo_passed") is not True
            or trial.get("strict_point_passed") is not True
            or parent.get("execution_status") != "completed" or parent.get("child_read_error") is not None
            or parent.get("child_sha256") != curvature.sha(BASE) or search.execution_status(resource) != "completed"):
        raise ValueError("accepted e0b04a dual-merit raw, parent, or resource receipt is incomplete")
    # The preconditioner is the archived f82c SPD matrix; it is never an H operator.
    hessian = torch.tensor(curvature_audit["curvature"]["hessian"], dtype=torch.float64)
    try:
        preconditioner, preconditioner_audit = block_step.block_inverse_preconditioner(hessian)
    except block_step.StepRefusal as error:
        raise ContinuationRefusal(f"cached f82c preconditioner refusal: {error}",
                                  "preconditioner_refusal") from error
    return plan, accepted_raw, {"audit": curvature_audit, "hessian": hessian,
                                "preconditioner": preconditioner, "preconditioner_audit": preconditioner_audit}, base_plan


def run(plan_path: Path, plan_sha: str, output: Path, *,
        base_loader: Callable[[Path, str], tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] | None = None,
        base_path: Path = BASE,
        max_iterations: int = MAX_ITERATIONS,
        linear_mode: str = "strict",
        shared_counts: dict[str, int] | None = None,
        absolute_deadline: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS if absolute_deadline is None else absolute_deadline
    actual_policy = _policy(linear_mode, max_iterations)
    counts = shared_counts if shared_counts is not None else {}
    for key, default in {"hvp_calls": 0, "hvp_calls_completed": 0,
                         "pcg_iterations_completed": 0, "pcg_solves_started": 0}.items():
        value = counts.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"shared_counts[{key!r}] must be a nonnegative integer")
    if (counts["hvp_calls"] > MAX_HVP or counts["hvp_calls_completed"] > counts["hvp_calls"]):
        raise ValueError("shared HVP counters exceed or contradict the global cap")
    initial_hvp_calls = counts["hvp_calls"]
    if output.exists() or output.is_symlink():
        raise ValueError("continuation output must be fresh")
    preflight: dict[str, Any] = {"phase": "preflight", "execution_status": "running",
        "numerical_status": "not_reached", "optimizer_steps_applied": 0,
        **counts, "policy": actual_policy, "linear_mode": linear_mode,
        "plan_sha256": plan_sha}
    search._write(output, preflight)
    if time.monotonic() >= deadline:
        preflight.update(phase="finished", execution_status="completed",
            numerical_status="budget_refusal", refusal="shared absolute deadline exhausted",
            elapsed_seconds=time.monotonic() - started)
        search._write(output, preflight)
        return preflight
    try:
        plan, accepted_raw, old, _ = (base_loader or load_base)(plan_path, plan_sha)
    except Exception as error:
        preflight.update(phase="execution_error", execution_status="failed", numerical_status="execution_error",
                         failure=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        search._write(output, preflight)
        raise
    _check_deadline(deadline)
    names = set(plan["source_files"]) | set(plan["archive_files"]) | {plan_path.relative_to(ROOT).as_posix()}
    before = {name: curvature.sha(ROOT / name) for name in names}
    if (before[plan_path.relative_to(ROOT).as_posix()] != plan_sha
            or any(before[name] != digest for name, digest in
                   {**plan["source_files"], **plan["archive_files"]}.items())):
        raise ValueError("continuation source/input/plan changed after preflight")
    problem, original, _, parameters, truth, _ = curvature.seed._prepare_fixed_seed()
    _check_deadline(deadline)
    control = torch.tensor(accepted_raw["accepted_control"], dtype=torch.float64)
    if (curvature.tensor_sha(control) != accepted_raw["accepted_control_sha256"]
            or curvature.tensor_sha(parameters) != accepted_raw["parameters_sha256"]
            or curvature.blocks.runtime_identity() != accepted_raw["runtime"]):
        raise ValueError("accepted start control, fixed parameters, or runtime identity changed")
    identity = curvature.seed._input_identity(problem, original, control, parameters, truth)
    if identity != accepted_raw["input_after"]:
        raise ValueError("reconstructed accepted-start fixed input identity changed")
    gradient_fn = torch.func.grad(problem.objective, argnums=0)

    def objective(c: Tensor, p: Tensor) -> Tensor:
        return problem.objective(c, p)

    def branch_fn(c: Tensor, p: Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        return curvature.tail._full_current_branch(problem, c, p)

    def hvp_fn(c: Tensor, p: Tensor, vector: Tensor) -> Tensor:
        return torch.func.jvp(lambda x: gradient_fn(x, p), (c,), (vector,))[1]

    endpoint_trials = [row for row in accepted_raw["trials"]
                       if row.get("status") == "accepted"
                       and row.get("control_sha256") == accepted_raw["accepted_control_sha256"]]
    if len(endpoint_trials) != 1:
        raise ValueError("accepted start point must match exactly one committed trial")
    expected_trial = endpoint_trials[0]
    _check_deadline(deadline)
    fresh = curvature.tail._fresh_merit(problem, control.clone(), parameters.clone(), gradient_fn)
    _check_deadline(deadline)
    current_branch, current_margins = branch_fn(control.clone(), parameters.clone())
    _check_deadline(deadline)
    expected_gradient = torch.tensor(expected_trial["gradient"], dtype=torch.float64)
    checks = curvature._endpoint_metrics({"objective": expected_trial["objective"],
        "phi": expected_trial["phi"], "gradient_inf": float(expected_gradient.abs().max()),
        "gradient": expected_trial["gradient"]}, fresh[0], fresh[1], fresh[2]) if fresh is not None else None
    if (fresh is None or not torch.equal(fresh[1], expected_gradient)
            or current_branch.get("signature_sha256") != expected_trial["branch"].get("signature_sha256")
            or not _strict_branch(current_branch, current_margins)
            or checks is None or not curvature._endpoint_metrics_pass(checks)):
        raise ContinuationRefusal("fresh start J/g/Phi/strict branch differs from accepted-step receipt")

    def commit_candidate(base: Tensor, candidate: Tensor, p: Tensor, base_j: Tensor,
                         base_gradient: Tensor, hs: Tensor, row: dict[str, Any], deadline_at: float) -> dict[str, Any]:
        _check_deadline(deadline_at)
        signature, scope = problem.branch_check(candidate, p)
        _check_deadline(deadline_at)
        spec = problem.frozen.fv_transport
        if spec is None:
            raise ValueError("FV transport disappeared")
        partition = curvature._branch_partition(signature, tuple(problem.layout["observation_times_seconds"]),
                                                spec.substeps_per_interval)
        if partition["full_signature_sha256"] != row["branch"]["signature_sha256"]:
            raise ContinuationRefusal("candidate branch changed during independent signature capture")
        state0 = diagnostics.physical_state(problem, base, p)
        _check_deadline(deadline_at)
        state1 = diagnostics.physical_state(problem, candidate, p)
        changes = {key: (state1[key] - state0[key]).tolist() for key in state0}
        _check_deadline(deadline_at)
        final_identity = curvature.seed._input_identity(problem, original, candidate, p, truth)
        after = {name: curvature.sha(ROOT / name) for name in names}
        intact = (before == after and curvature.blocks.runtime_identity() == accepted_raw["runtime"]
                  and curvature.audit._fixed_input_matches(final_identity, identity, curvature.tensor_sha(candidate)))
        if not intact:
            raise ContinuationRefusal("candidate fixed-input/source integrity closure failed")
        delta = candidate - base
        return {"accepted_branch_partition": partition, "branch_scope": scope,
            "model_diagnostics": diagnostics.model_diagnostics(float(base_j), base_gradient, None,
                delta, row["objective"], torch.tensor(row["gradient"], dtype=torch.float64),
                hessian_delta=row["alpha"] * hs),
            "physical_changes": {"values": changes,
                "base": {key: value.tolist() for key, value in state0.items()},
                "candidate": {key: value.tolist() for key, value in state1.items()},
                **diagnostics.physical_unit_labels()},
            "source_fixed_input_integrity_passed": True}

    record = run_iterations(control, parameters, objective, gradient_fn, branch_fn, hvp_fn,
        old["preconditioner"], deadline, max_iterations=max_iterations,
        linear_mode=linear_mode, shared_counts=counts,
        initial_metadata={"plan_sha256": plan_sha, "source_before": before,
            "accepted_step_sha256": curvature.sha(base_path), "input_before": identity,
            "runtime": accepted_raw["runtime"], "base_control_sha256": curvature.tensor_sha(control),
            "parameters_sha256": curvature.tensor_sha(parameters),
            "preconditioner_scope": "cached f82c SPD block inverse preconditioner only"},
        commit_candidate=commit_candidate,
        on_update=lambda current: search._write(output, current))
    after = {name: curvature.sha(ROOT / name) for name in names}
    final_control = torch.tensor(record["accepted_control"], dtype=torch.float64)
    final_identity = curvature.seed._input_identity(problem, original, final_control, parameters, truth)
    intact = (before == after and curvature.blocks.runtime_identity() == accepted_raw["runtime"]
              and curvature.audit._fixed_input_matches(final_identity, identity, curvature.tensor_sha(final_control)))
    record.update(source_before=before, source_after=after, source_unchanged=before == after,
        fixed_input_unchanged=intact, input_before=identity, input_after=final_identity,
        runtime=accepted_raw["runtime"], runtime_after=curvature.blocks.runtime_identity(),
        plan_sha256=plan_sha, accepted_step_sha256=curvature.sha(base_path),
        preconditioner_scope="cached f82c SPD block inverse preconditioner only",
        preconditioner_audit=old["preconditioner_audit"], elapsed_seconds=time.monotonic() - started,
        new_hvp_calls=record["hvp_calls"] - initial_hvp_calls,
        policy=actual_policy, linear_mode=linear_mode)
    if not intact:
        record.update(numerical_status="integrity_refusal", phase="integrity_refused",
                      execution_status="failed")
    search._write(output, record)
    return record


def main(*, run_fn: Callable[[Path, str, Path], dict[str, Any]] | None = None,
         default_plan: Path = PLAN,
         module_name: str = "examples.weather_scenarios.fv_point_3h_dual_merit_continuation") -> None:
    runner = run_fn or run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=default_plan)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.child:
        output = directory / "step.json"
        try:
            runner(args.plan.resolve(), args.plan_sha256, output)
        except search.StepRefusal as error:
            try:
                partial = json.loads(output.read_text()) if output.exists() else {}
            except (OSError, ValueError, UnicodeError):
                partial = {}
            if not isinstance(partial, dict):
                partial = {}
            partial.update(phase="finished", execution_status="completed",
                numerical_status="budget_refusal" if isinstance(error, search.BudgetRefusal)
                else getattr(error, "status", "step_refusal"), refusal=str(error))
            search._write(output, partial)
        except Exception as error:
            try:
                partial = json.loads(output.read_text()) if output.exists() else {}
            except (OSError, ValueError):
                partial = {}
            if not isinstance(partial, dict):
                partial = {}
            partial.update(phase="execution_error", execution_status="failed",
                numerical_status="execution_error", failure=f"{type(error).__name__}: {error}")
            search._write(output, partial)
            raise
        return
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(ROOT / ".venv/bin/python"), "-m", module_name,
        "--child", "--plan", str(args.plan.resolve()), "--plan-sha256", args.plan_sha256,
        "--directory", str(directory)]
    resource = run_guarded_diagnostic(command, wall_seconds=WALL_SECONDS, rss_bytes=RSS_BYTES,
        report_path=directory / "step.resource.json", log_path=directory / "step.log")
    parent = {"execution_status": execution_status(resource), "resource": resource,
              "numerical_status": "not_reached", "child_read_error": None,
              "completed_iterations": 0, "hvp_calls": 0}
    search._write(directory / "step.run.json", parent)
    try:
        child = json.loads((directory / "step.json").read_text())
        if not isinstance(child, dict):
            raise ValueError("child result must be a JSON object")
        parent.update(numerical_status=child["numerical_status"],
            completed_iterations=child.get("optimizer_steps_applied", 0),
            hvp_calls=child.get("hvp_calls", 0), child_sha256=curvature.sha(directory / "step.json"))
        if parent["execution_status"] == "completed" and not (child.get("phase") == "finished"
                and child.get("fixed_input_unchanged") is True and child.get("execution_status") == "completed"):
            parent["execution_status"] = "failed"
    except (OSError, ValueError, KeyError) as error:
        _copy_partial_child_counts(parent, directory / "step.json")
        parent.update(execution_status="failed" if parent["execution_status"] == "completed" else parent["execution_status"],
                      child_read_error=str(error))
    search._write(directory / "step.run.json", parent)
    print(json.dumps({k: parent[k] for k in ("execution_status", "numerical_status", "completed_iterations", "hvp_calls")}))


if __name__ == "__main__":
    main()
