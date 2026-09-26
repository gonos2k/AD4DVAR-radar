"""Resource guard and fail-closed publication checks for PR213 root-only work."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios import fv_partial_alternate_root_probe as probe


WALL_SECONDS = 600
SAMPLED_RSS_BYTES = 1024**3
_ROOT_STATUSES = {"root_margin_qualified", "root_low_margin"}
_NUMERICAL_STATUSES = _ROOT_STATUSES | {"root_refused"}
_KNOWN_PCG_ERRORS = (
    "residual norm is not finite",
    "preconditioner must be positive definite",
    "operator must be symmetric positive definite",
    "PCG step is not finite",
    "true residual norm is not finite",
    "PCG direction update is not finite",
)


def _known_refinement_refusal(value: object) -> bool:
    if not isinstance(value, str):
        return False
    numerical_prefix = "RefinementNumericalRefusal: "
    if value.startswith(numerical_prefix):
        reason = value[len(numerical_prefix):]
        return (reason in {
            "matrix-free Newton PCG did not converge",
            "matrix-free Newton PCG true residual exceeds tolerance",
            "Newton merit slope is nonfinite",
            "Newton step is not a descent direction for gradient merit",
        }
                or reason.startswith((
                    "stationarity refinement failed to find an admissible step:",
                    "stationarity refinement iteration budget exhausted:",
                ))
                or reason in _KNOWN_PCG_ERRORS)
    qualification_prefix = "_FinalQualificationRefusal: "
    if not value.startswith(qualification_prefix):
        return False
    reason = value[len(qualification_prefix):]
    return (reason == "final gradient infinity norm is not below 1e-10"
            or reason.startswith("basin terminal exact ")
            or reason in {
                "final strict branch refused: minmod joint oracle left its strict smooth branch",
                "final strict branch refused: partial FV strict branch has nonpositive scaled margins",
            })


def _terminal_failed_solve(child: dict[str, Any], solve: dict[str, Any],
                           trials: list[Any], solves: list[Any]) -> bool:
    """Validate one final PCG refusal and forbid work after that iteration."""
    refusal = child.get("refusal")
    iteration = solve.get("iteration")
    if (child.get("phase") != "root_refinement"
            or type(iteration) is not int
            or not _known_refinement_refusal(refusal)
            or not isinstance(refusal, str)
            or not refusal.startswith("RefinementNumericalRefusal: ")
            or not solves or solves[-1] is not solve
            or any(isinstance(trial, dict)
                   and type(trial.get("iteration")) is int
                   and trial["iteration"] >= iteration for trial in trials)
            or any(isinstance(later, dict)
                   and type(later.get("iteration")) is int
                   and later["iteration"] > iteration for later in solves)):
        return False
    solve_iterations = [row.get("iteration") for row in solves if isinstance(row, dict)]
    if solve_iterations != list(range(1, iteration + 1)):
        return False
    for earlier_iteration in range(1, iteration):
        earlier_trials = [trial for trial in trials
                          if isinstance(trial, dict)
                          and trial.get("iteration") == earlier_iteration]
        if (not earlier_trials or earlier_trials[-1].get("accepted") is not True
                or sum(trial.get("accepted") is True for trial in earlier_trials) != 1):
            return False
    error = solve.get("error")
    if isinstance(error, str):
        message = error.removeprefix("RuntimeError: ")
        return (error.startswith("RuntimeError: ")
                and message in _KNOWN_PCG_ERRORS
                and refusal == "RefinementNumericalRefusal: " + message)
    iterations = solve.get("iterations")
    reported = solve.get("reported_relative_residual")
    true_residual = solve.get("true_relative_residual")
    if (type(iterations) is not int or not 0 <= iterations <= 104
            or not _finite(reported) or not _finite(true_residual)):
        return False
    if solve.get("converged") is False:
        return (refusal == "RefinementNumericalRefusal: matrix-free Newton PCG did not converge"
                and iterations == 104
                and isinstance(reported, (int, float))
                and reported > probe.PCG_TRUE_RESIDUAL_TOLERANCE
                and isinstance(true_residual, (int, float))
                and true_residual > probe.PCG_TRUE_RESIDUAL_TOLERANCE)
    return (solve.get("converged") is True
            and isinstance(true_residual, (int, float))
            and true_residual > probe.PCG_TRUE_RESIDUAL_TOLERANCE
            and refusal == (
                "RefinementNumericalRefusal: matrix-free Newton PCG true residual exceeds tolerance"))


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _finite(value: object) -> bool:
    return (isinstance(value, (int, float)) and type(value) in (int, float)
            and math.isfinite(value))


def _valid_curvature(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (_finite(value.get("symmetry_relative"))
            and _finite(value.get("lambda_min"))
            and _finite(value.get("lambda_max"))
            and _finite(value.get("lambda_ratio"))
            and type(value.get("hvp_columns")) is int
            and value["hvp_columns"] == 26
            and 0 <= value["symmetry_relative"] <= 1e-10
            and value["lambda_min"] > 0 and value["lambda_max"] > 0
            and value["lambda_ratio"] > math.sqrt(sys.float_info.epsilon))


def _valid_branch(value: object, *, threshold: float) -> bool:
    return (isinstance(value, dict)
            and type(value.get("euler_stages")) is int
            and value["euler_stages"] == 54
            and _digest(value.get("signature_sha256"))
            and all(_finite(value.get(name)) and value[name] > threshold for name in (
                "minimum_scaled_slope_margin", "minimum_scaled_face_flux_margin")))


def _valid_candidate_trace(child: dict[str, Any], *, allow_solve_error: bool,
                          require_complete: bool = False) -> bool:
    trials = child.get("trial_records")
    policies = child.get("policy_records")
    solves = child.get("linear_solves")
    branch_calls = child.get("branch_calls")
    if (not isinstance(trials, list) or not isinstance(policies, list)
            or not isinstance(solves, list) or not isinstance(branch_calls, list)
            or not _finite(child.get("seed_gradient_norm"))
            or not _digest(child.get("seed_signature_sha256"))):
        return False
    policy_by_trial: dict[tuple[object, object], dict[str, Any]] = {}
    for policy in policies:
        if not isinstance(policy, dict):
            return False
        key = (policy.get("iteration"), policy.get("backtrack"))
        if key in policy_by_trial:
            return False
        policy_by_trial[key] = policy
    solve_by_iteration: dict[object, dict[str, Any]] = {}
    for solve in solves:
        if not isinstance(solve, dict):
            return False
        iteration = solve.get("iteration")
        if iteration in solve_by_iteration:
            return False
        solve_by_iteration[iteration] = solve
    positive_branch_calls = {
        row.get("control_sha256"): row for row in branch_calls
        if isinstance(row, dict) and row.get("status") == "positive_strict_branch"
    }
    current_signature = child["seed_signature_sha256"]
    current_gradient_norm = child["seed_gradient_norm"]
    accepted_control_hash = child.get("seed_control_sha256")
    consumed_policies: set[tuple[object, object]] = set()
    trial_order: list[tuple[int, int]] = []
    accepted_by_iteration: dict[int, int] = {}
    for trial in trials:
        if not isinstance(trial, dict) or not _digest(trial.get("candidate_control_sha256")):
            return False
        candidate_values = trial.get("candidate_control")
        if candidate_values is not None:
            if (not isinstance(candidate_values, list)
                    or not all(_finite(value) for value in candidate_values)
                    or probe._tensor_sha(torch.tensor(candidate_values, dtype=torch.float64))
                    != trial["candidate_control_sha256"]):
                return False
        iteration, backtrack = trial.get("iteration"), trial.get("backtrack")
        solve = solve_by_iteration.get(iteration)
        if (type(iteration) is not int or not 1 <= iteration <= 8
                or type(backtrack) is not int or not 0 <= backtrack < 16
                or solve is None):
            return False
        trial_order.append((iteration, backtrack))
        candidate_values = trial.get("candidate_control")
        if trial.get("accepted") is True:
            if (not isinstance(candidate_values, list) or len(candidate_values) != 26
                    or not all(_finite(value) for value in candidate_values)
                    or probe._tensor_sha(torch.tensor(candidate_values, dtype=torch.float64))
                    != trial["candidate_control_sha256"]):
                return False
        branch = trial.get("branch")
        if trial.get("candidate_nonfinite") is True:
            if (trial.get("candidate_control") is not None
                    or trial.get("accepted") is True
                    or trial.get("rejection") != "nonfinite_candidate"):
                return False
            continue
        if trial.get("rejection") == "nonfinite_candidate":
            if isinstance(branch, dict):
                if (not _valid_branch(branch, threshold=0.0)
                        or trial["candidate_control_sha256"] not in positive_branch_calls):
                    return False
            continue
        if trial.get("rejection") == "branch":
            if isinstance(branch, dict):
                return False
            rejected_call = next((row for row in branch_calls
                                  if isinstance(row, dict)
                                  and row.get("control_sha256") == trial[
                                      "candidate_control_sha256"]), None)
            if (not isinstance(rejected_call, dict)
                    or rejected_call.get("status") != "oracle_branch_refused"):
                return False
            continue
        if not isinstance(branch, dict) or not _valid_branch(branch, threshold=0.0):
            return False
        if (not _finite(trial.get("objective"))
                or not _finite(trial.get("gradient_norm"))
                or not _finite(trial.get("gradient_max"))):
            return False
        candidate_signature = branch.get("signature_sha256")
        if (not _digest(candidate_signature)
                or trial["candidate_control_sha256"] not in positive_branch_calls):
            return False
        branch_call = positive_branch_calls[trial["candidate_control_sha256"]]
        if (branch_call.get("signature_sha256") != candidate_signature
                or branch_call.get("minimum_scaled_slope_margin")
                != branch.get("minimum_scaled_slope_margin")
                or branch_call.get("minimum_scaled_face_flux_margin")
                != branch.get("minimum_scaled_face_flux_margin")):
            return False
        policy_key = (iteration, backtrack)
        policy = policy_by_trial.get(policy_key)
        switched = candidate_signature != current_signature
        if switched:
            if (not isinstance(policy, dict)
                    or policy.get("current_signature_sha256") != current_signature
                    or policy.get("candidate_signature_sha256") != candidate_signature
                    or not _finite(trial.get("gradient_norm"))):
                return False
            old_phi = 0.5 * current_gradient_norm * current_gradient_norm
            new_phi = 0.5 * trial["gradient_norm"] * trial["gradient_norm"]
            floor = (128 * sys.float_info.epsilon
                     * max(abs(old_phi), abs(new_phi), sys.float_info.min))
            delta = old_phi - new_phi
            accepted_switch = math.isfinite(delta) and delta > floor
            if (not _finite(policy.get("old_phi"))
                    or not _finite(policy.get("new_phi"))
                    or not _finite(policy.get("measured_phi_decrease"))
                    or not _finite(policy.get("required_decrease"))
                    or not math.isclose(policy["old_phi"], old_phi,
                                        rel_tol=1e-14, abs_tol=0.0)
                    or not math.isclose(policy["new_phi"], new_phi,
                                        rel_tol=1e-14, abs_tol=0.0)
                    or not math.isclose(policy["measured_phi_decrease"], delta,
                                        rel_tol=1e-14, abs_tol=0.0)
                    or not math.isclose(policy["required_decrease"], floor,
                                        rel_tol=1e-14, abs_tol=0.0)
                    or policy.get("accepted") is not accepted_switch
                    or trial.get("accepted") is not accepted_switch
                    or policy.get("reason") != (
                        "measured_phi_decrease" if accepted_switch
                        else "measured_phi_refused")):
                return False
            consumed_policies.add(policy_key)
        elif policy is not None or "acceptance_policy" in trial:
            return False
        elif (not _finite(trial.get("armijo_ratio"))
              or trial.get("accepted") is not (trial["armijo_ratio"] <= 1.0)):
            return False
        if trial.get("accepted") is True:
            if (not _finite(trial.get("objective"))
                    or not _finite(trial.get("gradient_norm"))
                    or not _finite(trial.get("gradient_max"))
                    or not _finite(trial.get("linear_relative_residual"))
                    or trial["linear_relative_residual"] > probe.PCG_TRUE_RESIDUAL_TOLERANCE
                    or not _finite(trial.get("pcg_relative_residual"))
                    or not 0 <= trial["pcg_relative_residual"]
                    <= probe.PCG_TRUE_RESIDUAL_TOLERANCE
                    or type(trial.get("pcg_iterations")) is not int
                    or trial["pcg_iterations"] != solve.get("iterations")
                    or not _finite(solve.get("true_relative_residual"))
                    or not math.isclose(solve["true_relative_residual"],
                                        trial["linear_relative_residual"],
                                        rel_tol=1e-12, abs_tol=1e-15)
                    or not _finite(solve.get("reported_relative_residual"))
                    or not math.isclose(solve["reported_relative_residual"],
                                        trial["pcg_relative_residual"],
                                        rel_tol=1e-12, abs_tol=1e-15)):
                return False
            current_signature = candidate_signature
            current_gradient_norm = trial["gradient_norm"]
            accepted_control_hash = trial["candidate_control_sha256"]
            accepted_by_iteration[iteration] = accepted_by_iteration.get(iteration, 0) + 1
        elif not switched and (not _finite(trial.get("armijo_ratio"))
                               or trial["armijo_ratio"] <= 1.0):
            return False
    if consumed_policies != set(policy_by_trial):
        return False
    if trial_order != sorted(trial_order):
        return False
    for iteration in {item[0] for item in trial_order}:
        backtracks = [backtrack for step, backtrack in trial_order if step == iteration]
        if backtracks != list(range(len(backtracks))):
            return False
    if require_complete:
        iterations = child.get("refinement_iterations")
        if (type(iterations) is not int or iterations < 1
                or len(accepted_by_iteration) != iterations
                or set(accepted_by_iteration) != set(range(1, iterations + 1))
                or any(count != 1 for count in accepted_by_iteration.values())
                or len(solves) != iterations
                or any(isinstance(solve, dict) and isinstance(solve.get("error"), str)
                       for solve in solves)):
            return False
        for iteration in range(1, iterations + 1):
            step_trials = [trial for trial in trials
                           if trial.get("iteration") == iteration]
            if (not step_trials or step_trials[-1].get("accepted") is not True
                    or sum(trial.get("accepted") is True for trial in step_trials) != 1):
                return False
    for solve in solves:
        if (type(solve.get("iteration")) is not int
                or not 1 <= solve["iteration"] <= 8
                or solve.get("rtol") != probe.PCG_TRUE_RESIDUAL_TOLERANCE
                or solve.get("max_iterations") != 104):
            return False
        failed_solve = (
            isinstance(solve.get("error"), str)
            or solve.get("converged") is False
            or (_finite(solve.get("true_relative_residual"))
                and solve["true_relative_residual"]
                > probe.PCG_TRUE_RESIDUAL_TOLERANCE)
        )
        if failed_solve:
            if (not allow_solve_error
                    or not _terminal_failed_solve(child, solve, trials, solves)):
                return False
            continue
        if (type(solve.get("iterations")) is not int
                or not 0 <= solve["iterations"] <= 104
                or solve.get("converged") is not True
                or not _finite(solve.get("true_relative_residual"))
                or solve["true_relative_residual"] > probe.PCG_TRUE_RESIDUAL_TOLERANCE):
            return False
    return _digest(accepted_control_hash)


def _valid_root(child: dict[str, Any], status: str) -> bool:
    if (child.get("phase") != "finished"
            or not isinstance(child.get("control"), list)
            or len(child["control"]) != 26
            or not all(_finite(value) for value in child["control"])
            or not _digest(child.get("final_control_sha256"))
            or child.get("final_control_sha256") != child.get("final_curvature_control_sha256")
            or not _finite(child.get("final_gradient_max"))
            or not 0 <= child["final_gradient_max"] < probe.STATIONARITY_TOLERANCE
            or not _finite(child.get("final_gradient_norm"))
            or not _finite(child.get("final_objective"))
            or type(child.get("refinement_iterations")) is not int
            or not 0 <= child["refinement_iterations"] <= 8
            or not _valid_curvature(child.get("final_curvature"))
            or child.get("response_validation") != "not_performed"
            or child.get("physical_validation") != "not_performed"):
        return False
    if not _valid_candidate_trace(child, allow_solve_error=False, require_complete=True):
        return False
    control_hash = probe._tensor_sha(torch.tensor(child["control"], dtype=torch.float64))
    accepted_trials = [trial for trial in child["trial_records"]
                       if isinstance(trial, dict) and trial.get("accepted") is True]
    expected_hash = (accepted_trials[-1]["candidate_control_sha256"]
                     if accepted_trials else child.get("seed_control_sha256"))
    if control_hash != child.get("final_control_sha256") or control_hash != expected_hash:
        return False
    branch = child.get("final_branch")
    if not isinstance(branch, dict):
        return False
    expected_signature = (accepted_trials[-1].get("branch", {}).get("signature_sha256")
                          if accepted_trials else child.get("seed_signature_sha256"))
    if branch.get("signature_sha256") != expected_signature:
        return False
    final_branch_call = next((row for row in child["branch_calls"]
                              if isinstance(row, dict)
                              and row.get("phase") == "final_branch"
                              and row.get("control_sha256") == control_hash
                              and row.get("status") == "positive_strict_branch"), None)
    if not child["branch_calls"] or child["branch_calls"][-1] is not final_branch_call:
        return False
    if (not isinstance(final_branch_call, dict)
            or final_branch_call.get("signature_sha256") != branch.get("signature_sha256")
            or final_branch_call.get("minimum_scaled_slope_margin")
            != branch.get("minimum_scaled_slope_margin")
            or final_branch_call.get("minimum_scaled_face_flux_margin")
            != branch.get("minimum_scaled_face_flux_margin")):
        return False
    if status == "root_margin_qualified":
        return (_valid_branch(branch, threshold=1e-4)
                and child.get("response_margin_qualified") is True)
    if not _valid_branch(branch, threshold=0.0) or not isinstance(branch, dict):
        return False
    slope_margin = branch.get("minimum_scaled_slope_margin")
    face_margin = branch.get("minimum_scaled_face_flux_margin")
    return (isinstance(slope_margin, (int, float))
            and isinstance(face_margin, (int, float))
            and (slope_margin <= 1e-4 or face_margin <= 1e-4)
            and child.get("response_margin_qualified") is False)


def _valid_refusal(child: dict[str, Any], status: str, exit_code: int) -> bool:
    if status == "root_refused":
        if (exit_code != 2 or not _known_refinement_refusal(child.get("refusal"))):
            return False
        phase = child.get("phase")
        if phase == "root_refinement":
            solves = child.get("linear_solves")
            trials = child.get("trial_records")
            if not isinstance(solves, list) or not solves or not isinstance(trials, list):
                return False
            failed = [solve for solve in solves if isinstance(solve, dict) and (
                isinstance(solve.get("error"), str)
                or solve.get("converged") is False
                or (_finite(solve.get("true_relative_residual"))
                    and solve["true_relative_residual"]
                    > probe.PCG_TRUE_RESIDUAL_TOLERANCE)
            )]
            if failed:
                return len(failed) == 1 and _terminal_failed_solve(
                    child, failed[0], trials, solves)
            if not all(isinstance(solve, dict)
                       and solve.get("converged") is True
                       and _finite(solve.get("true_relative_residual"))
                       and solve["true_relative_residual"]
                       <= probe.PCG_TRUE_RESIDUAL_TOLERANCE
                       for solve in solves):
                return False
            refusal = str(child["refusal"])
            if refusal.startswith("RefinementNumericalRefusal: stationarity refinement failed to find"):
                last_iteration = solves[-1].get("iteration")
                last_trials = [trial for trial in trials
                               if isinstance(trial, dict)
                               and trial.get("iteration") == last_iteration]
                return (len(last_trials) == 16
                        and all(trial.get("accepted") is not True for trial in last_trials))
            if refusal.startswith("RefinementNumericalRefusal: stationarity refinement iteration budget exhausted:"):
                accepted = [trial for trial in trials if isinstance(trial, dict)
                            and trial.get("accepted") is True]
                return (len(solves) == 8 and len(accepted) == 8
                        and all(sum(trial.get("accepted") is True for trial in trials
                                    if isinstance(trial, dict)
                                    and trial.get("iteration") == i) == 1
                                for i in range(1, 9)))
            last_iteration = solves[-1].get("iteration")
            return not any(isinstance(trial, dict)
                           and trial.get("iteration") == last_iteration
                           for trial in trials)
        return phase in {"final_curvature", "final_branch"} and str(
            child.get("refusal", "")).startswith("_FinalQualificationRefusal: ")
    return False


def _seed_archive_matches_child(child: dict[str, Any]) -> bool:
    try:
        archive = probe.SEED_ARCHIVE
        manifest_path = archive / "manifest.json"
        if probe._sha(manifest_path) != probe.SEED_MANIFEST_SHA256:
            return False
        manifest = json.loads(manifest_path.read_text())
        hashes = manifest.get("file_sha256")
        if not isinstance(hashes, dict):
            return False
        for name in probe.SEED_RECORDS:
            path = archive / name
            relative = str(path.relative_to(ROOT))
            if relative not in hashes or probe._sha(path) != hashes[relative]:
                return False
        seed = json.loads((archive / "alternate_seed.json").read_text())
        parent = json.loads((archive / "alternate_seed.run.json").read_text())
        resource = json.loads((archive / "alternate_seed.resource.json").read_text())
        control, row, prior = probe._archived_seed_control()
        branch = child.get("seed_branch")
        archived_branch = seed.get("seed_branch")
        fresh_matches = (
            isinstance(branch, dict) and isinstance(archived_branch, dict)
            and branch.get("euler_stages") == archived_branch.get("euler_stages") == 54
            and branch.get("signature_sha256") == archived_branch.get("signature_sha256")
            == probe.SEED_SIGNATURE_SHA256
            and all(_finite(branch.get(name)) and _finite(archived_branch.get(name))
                    and math.isclose(branch[name], archived_branch[name],
                                     rel_tol=1e-10, abs_tol=1e-12)
                    for name in ("minimum_scaled_slope_margin",
                                 "minimum_scaled_face_flux_margin"))
        )
        if not isinstance(branch, dict):
            return False
        for name in ("objective", "gradient_norm", "gradient_max"):
            if (not _finite(child.get("seed_" + name))
                    or not _finite(seed.get(name))
                    or not math.isclose(child["seed_" + name], seed[name],
                                        rel_tol=1e-10, abs_tol=1e-12)):
                return False
        seed_branch_call = next((entry for entry in child.get("branch_calls", [])
                                 if isinstance(entry, dict)
                                 and entry.get("phase") == "seed_branch"
                                 and entry.get("control_sha256") == probe.SEED_CONTROL_SHA256
                                 and entry.get("status") == "positive_strict_branch"), None)
        return (
            fresh_matches
            and seed.get("numerical_status") == "seed_locally_spd"
            and parent.get("execution_status") == "completed"
            and parent.get("numerical_status") == "seed_locally_spd"
            and resource.get("exit_code") == 0
            and resource.get("resource_termination") is None
            and resource.get("monitor_error") is None
            and child.get("seed_manifest_sha256") == probe.SEED_MANIFEST_SHA256
            and child.get("seed_control_sha256") == probe._tensor_sha(control)
            == probe.SEED_CONTROL_SHA256 == row.get("candidate_control_sha256")
            and child.get("seed_signature_sha256") == seed.get("selected_signature_sha256")
            and seed.get("curvature_control_sha256") == probe.SEED_CONTROL_SHA256
            and child.get("seed_curvature_control_sha256") == probe.SEED_CONTROL_SHA256
            and child.get("seed_curvature") == seed.get("curvature")
            and child.get("seed_input_identity") == seed.get("input_before")
            == prior.get("input_before") == prior.get("input_after")
            and seed.get("source_before") == seed.get("source_after")
            and child.get("environment") == seed.get("environment")
            and isinstance(seed_branch_call, dict)
            and seed_branch_call.get("signature_sha256") == probe.SEED_SIGNATURE_SHA256
            and seed_branch_call.get("minimum_scaled_slope_margin")
            == branch.get("minimum_scaled_slope_margin")
            and seed_branch_call.get("minimum_scaled_face_flux_margin")
            == branch.get("minimum_scaled_face_flux_margin")
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _valid_child(child: object, exit_code: int, *, plan_hash: str,
                 probe_hash: str, runner_hash: str,
                 sources: dict[str, str]) -> bool:
    if not isinstance(child, dict):
        return False
    status = child.get("numerical_status")
    common = (
        child.get("pid", 0) > 0
        and child.get("phase") != "running"
        and child.get("plan_sha256") == plan_hash == probe._sha(probe.PLAN)
        and child.get("plan_sha256_before") == child.get("plan_sha256_after") == plan_hash
        and child.get("reviewed_probe_sha256") == probe_hash == probe._sha(Path(probe.__file__))
        and child.get("reviewed_runner_sha256") == runner_hash == probe._sha(Path(__file__))
        and child.get("seed_manifest_sha256") == probe.SEED_MANIFEST_SHA256
        and child.get("seed_control_sha256") == probe.SEED_CONTROL_SHA256
        and child.get("seed_signature_sha256") == probe.SEED_SIGNATURE_SHA256
        and child.get("new_product_gn_runs") == 0
        and child.get("new_adjoint_reanalysis_runs") == 0
        and child.get("new_vjp_runs") == 0
        and child.get("new_nonlinear_reanalysis_runs") == 0
        and child.get("new_newton_refinement_runs") == 1
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
        and child.get("source_before") == child.get("source_after") == sources == probe._sources()
        and child.get("input_before") == child.get("input_after")
        and _seed_archive_matches_child(child)
        and child.get("plan_unchanged") is True
        and child.get("seed_archive_unchanged") is True
        and child.get("seed_record_hashes_before")
        == child.get("seed_record_hashes_after") == probe._seed_record_hashes()
        and child.get("seed_archives_unchanged") is True
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("parameters_unchanged") is True
        and _valid_candidate_trace(child, allow_solve_error=status == "root_refused")
    )
    if not common:
        return False
    if status in _ROOT_STATUSES:
        return exit_code == 0 and _valid_root(child, status)
    return status in _NUMERICAL_STATUSES and _valid_refusal(child, status, exit_code)


def run(directory: Path, *, expected_plan_sha256: str,
        expected_probe_sha256: str,
        expected_runner_sha256: str) -> dict[str, Any]:
    protected = (probe.SEED_ARCHIVE.resolve(strict=True),
                 probe.seed_gate.ARCHIVE.resolve(strict=True))
    resolved = directory.resolve(strict=False)
    if (any(resolved.is_relative_to(path) for path in protected)
            or directory.is_symlink()):
        raise ValueError("alternate root output must be outside frozen raw archives")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("alternate root output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "partial_alternate_root.json"
    command = [
        sys.executable, str(Path(probe.__file__)), "--output", str(output),
        "--expected-plan-sha256", expected_plan_sha256,
        "--expected-probe-sha256", expected_probe_sha256,
        "--expected-runner-sha256", expected_runner_sha256,
    ]
    source_before = probe._sources()
    if (probe._sha(probe.PLAN) != expected_plan_sha256
            or probe._sha(Path(probe.__file__)) != expected_probe_sha256
            or probe._sha(Path(__file__)) != expected_runner_sha256
            or source_before.get("examples/weather_scenarios/fv_partial_alternate_root_runner.py")
            != expected_runner_sha256):
        raise ValueError("reviewed alternate root source or plan changed before launch")
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "partial_alternate_root.resource.json",
        log_path=directory / "partial_alternate_root.log",
    )
    child: dict[str, Any] | None = None
    read_error = None
    try:
        parsed = json.loads(output.read_text())
        if isinstance(parsed, dict):
            child = parsed
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    valid_resource = (
        resource.get("exit_code") in (0, 2)
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("command") == command
        and type(resource.get("child_pid")) is int and resource["child_pid"] > 0
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
        and _finite(resource.get("elapsed_seconds"))
        and 0 <= resource["elapsed_seconds"] <= WALL_SECONDS
    )
    child_ok = (
        child is not None
        and child.get("pid") == resource.get("child_pid")
        and _valid_child(child, resource.get("exit_code"), plan_hash=expected_plan_sha256,
                         probe_hash=expected_probe_sha256, runner_hash=expected_runner_sha256,
                         sources=source_before)
    )
    execution_status = (
        "resource_limited" if resource.get("resource_termination") is not None else
        "completed" if valid_resource and child_ok else "failed"
    )
    result = {
        "execution_status": execution_status,
        "numerical_status": child.get("numerical_status") if child else "not_reached",
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "resource": resource,
        "child_read_error": read_error,
        "scope": "one PR212 alternate-seed root-only refinement; no GN/adjoint/VJP/reanalysis",
    }
    (directory / "partial_alternate_root.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    args = parser.parse_args()
    result = run(args.directory, expected_plan_sha256=args.expected_plan_sha256,
                 expected_probe_sha256=args.expected_probe_sha256,
                 expected_runner_sha256=args.expected_runner_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
