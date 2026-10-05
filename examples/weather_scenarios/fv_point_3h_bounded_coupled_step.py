"""One guarded original-J step using the pinned cd6b Hessian; no new HVP."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit
from examples.weather_scenarios import fv_point_3h_endpoint_diagnostics as diagnostic
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_schur_newton_step as block_step
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks
from examples.weather_scenarios.fv_diagnostic_guard import atomic_write_text, run_guarded_diagnostic

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
RUN_DIR = EVIDENCE / "resumable_cd6b_hessian_20261004"
ARCHIVE = EVIDENCE / "coupled_original_j_continuation_attempt1/coupled_original_j_continuation.json"
EXPERIMENT = RUN_DIR / "experiment.json"
CHECKPOINT = RUN_DIR / "checkpoint/hessian_checkpoint.json"
BASE_AUDIT = RUN_DIR / "attempt_2/audit.json"
BASE_PARENT = RUN_DIR / "attempt_2/audit.run.json"
BASE_RESOURCE = RUN_DIR / "attempt_2/audit.resource.json"
SELF = "examples/weather_scenarios/fv_point_3h_bounded_coupled_step.py"
TEST = "tests/test_fv_point_3h_bounded_coupled_step.py"
PLAN = EVIDENCE / "PR243_BOUNDED_COUPLED_PLAN_20261005.json"
ARCHIVE_SHA = "3e2e1e247be0de1c6b5e27d4c835dbbe6b456f7a0c0dc2e0e90ac18d71fd8385"
EXPERIMENT_SHA = "76eaca929840aba29aa9665a745b9f4ec2c709517c5069f5e84ff6def81db4ae"
CHECKPOINT_SHA = "d0064b2f5ec6a570cd679a683f3f38667e2f9f73d73d6b30c46dbe1d92874797"
BASE_AUDIT_SHA = "20b7ce7bab9e960d8feb857a4d43a9b51c67e6e1d60a1cce549d3d37cf71d8e5"
BASE_PARENT_SHA = "39250737419ce21f3627a90fda96414ae7e449f1095c0530f3d2c713ef212b9e"
BASE_RESOURCE_SHA = "f248b2011dc3b67d324b344099b217084a2fbd9dd8cbb5a93f74f665919eba91"
CONTROL_SHA = "cd6b5693e2bd32a8a41a3509b87b61e7cf647b03d864dfccc0a4cccce96e908d"
PARAMETERS_SHA = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
HESSIAN_SHA = "9abaf1e5d643bce7c2d1426245d64c04feec119b7eddecdc3853a40dbb297003"
GRADIENT_SHA = "c9dfc589a4b83665e48af0266d68514af9189738b6cf376891c081ebbd7e5a94"
MU, RADIUS, MAX_BACKTRACKS = 40.0, 0.05, 16
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 240.0, 300.0, 1024**3
ALPHA_C1 = 1e-4


class StepRefusal(RuntimeError):
    """Known one-step numerical or budget refusal."""


class BudgetRefusal(StepRefusal):
    """Cooperative deadline reached before starting another operation."""


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise BudgetRefusal("240-second internal budget exhausted during setup")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def exact_shifted_direction(hessian: Tensor, gradient: Tensor, mu: float = MU) -> tuple[Tensor, dict[str, Any]]:
    """Apply the exact cached 20+6 block inverse of H+mu*P_d; performs no HVP."""
    if (hessian.shape != (26, 26) or hessian.dtype != torch.float64 or hessian.device.type != "cpu"
            or gradient.shape != (26,) or gradient.dtype != torch.float64 or gradient.device.type != "cpu"
            or not bool(torch.isfinite(hessian).all() & torch.isfinite(gradient).all())):
        raise ValueError("cached Hessian/gradient must be finite CPU FP64 26-control values")
    shifted = hessian.clone()
    shifted[20:, 20:] += mu * torch.eye(6, dtype=torch.float64)
    try:
        inverse, inverse_audit = block_step.block_inverse_preconditioner(shifted)
        step = inverse(-gradient)
    except block_step.StepRefusal as error:
        raise StepRefusal(f"cached shifted block qualification refused: {error}") from error
    residual = shifted @ step + gradient
    rhs_norm = torch.linalg.vector_norm(gradient)
    relative = float(torch.linalg.vector_norm(residual) / rhs_norm) if float(rhs_norm) > 0 else float(torch.linalg.vector_norm(residual))
    products = gradient * step
    slope = float(torch.dot(gradient, step))
    h_step = hessian @ step
    phi_directional_derivative = float(torch.dot(gradient, h_step))
    original_h_quadratic_form = float(torch.dot(step, h_step))
    dot_budget = float(128 * torch.finfo(torch.float64).eps * products.abs().sum())
    if not bool(torch.isfinite(step).all()) or not math.isfinite(relative) or relative > 1e-10:
        raise StepRefusal("cached shifted block-inverse true residual gate failed")
    if not slope < -dot_budget:
        raise StepRefusal("cached modified-Newton direction is not descending beyond dot rounding")
    original_residual = hessian @ step + gradient
    return step, {"mu": mu, "modified_operator": "cached H + mu*P_d",
        "preconditioner_audit": inverse_audit, "modified_residual": residual.tolist(),
        "modified_relative_residual": relative, "original_residual": original_residual.tolist(),
        "original_relative_residual": float(torch.linalg.vector_norm(original_residual) / rhs_norm),
        "g_dot_s": slope, "g_dot_s_rounding_budget": dot_budget,
        "original_H_step_quadratic_form": original_h_quadratic_form,
        "phi_directional_derivative": phi_directional_derivative,
        "modified_condition_2": inverse_audit["full"]["lambda_max"] / inverse_audit["full"]["lambda_min"],
        "step_l2": float(torch.linalg.vector_norm(step)),
        "step_gradient_blocks": blocks.gradient_blocks(step)}


def bounded_original_j_search(control: Tensor, parameters: Tensor, base_objective: Tensor,
                              gradient: Tensor, step: Tensor,
                              objective: Callable[[Tensor, Tensor], Tensor],
                              gradient_fn: Callable[[Tensor, Tensor], Tensor],
                              branch_fn: Callable[[Tensor, Tensor], tuple[dict[str, Any], dict[str, Any]]],
                              base_branch_signature: str,
                              *, deadline: float | None = None,
                              on_trial: Callable[[dict[str, Any]], None] | None = None
                              ) -> tuple[Tensor | None, list[dict[str, Any]], str]:
    step_norm = float(torch.linalg.vector_norm(step))
    if not math.isfinite(step_norm) or step_norm <= 0:
        raise StepRefusal("cached modified-Newton direction has no positive finite norm")
    alpha_start = min(1.0, RADIUS / step_norm)
    slope = float(torch.dot(gradient, step))
    trials: list[dict[str, Any]] = []
    for backtrack in range(MAX_BACKTRACKS):
        alpha = alpha_start * 2.0**-backtrack
        candidate = control + alpha * step
        row: dict[str, Any] = {"backtrack": backtrack, "alpha": alpha,
                               "control": candidate.tolist(), "control_sha256": tensor_sha(candidate)}
        if deadline is not None and time.monotonic() >= deadline:
            row.update(status="budget_refused_before_evaluation")
            trials.append(row)
            if on_trial: on_trial(row)
            return None, trials, "step_budget_refusal"
        value = objective(candidate, parameters)
        if not isinstance(value, Tensor) or value.shape != () or value.dtype != torch.float64 or value.device.type != "cpu":
            raise ValueError("candidate original J callback contract changed")
        if not bool(torch.isfinite(value)):
            row.update(status="nonfinite_objective")
            trials.append(row)
            if on_trial: on_trial(row)
            continue
        candidate_gradient = gradient_fn(candidate, parameters)
        if (candidate_gradient.shape != gradient.shape or candidate_gradient.dtype != torch.float64
                or candidate_gradient.device.type != "cpu"):
            raise ValueError("candidate full-gradient callback contract changed")
        row["objective"] = float(value)
        if not bool(torch.isfinite(candidate_gradient).all()):
            row.update(status="nonfinite_gradient")
            trials.append(row)
            if on_trial: on_trial(row)
            continue
        phi = torch.dot(candidate_gradient, candidate_gradient) / 2
        row.update(gradient=candidate_gradient.tolist(), gradient_inf=float(candidate_gradient.abs().max()),
                   phi=float(phi) if bool(torch.isfinite(phi)) else None)
        if not bool(torch.isfinite(phi)):
            row.update(status="nonfinite_phi")
            trials.append(row)
            if on_trial: on_trial(row)
            continue
        branch, margins = branch_fn(candidate, parameters)
        strict = (branch.get("status") == "passed_strict_branch"
                  and branch.get("euler_stages") == 3600
                  and branch.get("choice_stage_count") == 3600
                  and branch.get("face_sign_stage_count") == 3600
                  and seed._valid_margins(margins, complete=True))
        armijo = block_step.armijo_holds(float(base_objective), float(value), alpha, slope)
        row.update(branch=branch, margins=margins, strict_point_passed=strict,
                   branch_signature_changed=branch.get("signature_sha256") != base_branch_signature,
                   armijo_passed=armijo,
                   status="candidate_evaluated")
        if deadline is not None and time.monotonic() >= deadline:
            row["status"] = "budget_refused_after_evaluation"
            trials.append(row)
            if on_trial: on_trial(row)
            return None, trials, "step_budget_refusal"
        if armijo and strict:
            row["status"] = "accepted"
            trials.append(row)
            if on_trial: on_trial(row)
            return candidate, trials, "one_original_J_step_accepted"
        row["status"] = "rejected"
        trials.append(row)
        if on_trial: on_trial(row)
    return None, trials, "original_J_armijo_grid_exhausted"


def _source_paths(plan_path: Path, archive: dict[str, Any]) -> list[str]:
    archive_path = ARCHIVE.relative_to(ROOT).as_posix()
    plan_sources = json.loads(plan_path.read_text())["source_files"]
    return sorted(set(plan_sources) | _required_source_files(archive) | {
        str(plan_path.relative_to(ROOT)), archive_path,
        str(ARCHIVE.with_suffix(".run.json").relative_to(ROOT)),
        str(ARCHIVE.with_suffix(".resource.json").relative_to(ROOT)),
        str(EXPERIMENT.relative_to(ROOT)), str(CHECKPOINT.relative_to(ROOT)),
        str(BASE_AUDIT.relative_to(ROOT)), str(BASE_PARENT.relative_to(ROOT)),
        str(BASE_RESOURCE.relative_to(ROOT))})


def _required_source_files(archive: dict[str, Any]) -> set[str]:
    return set(archive["source_before"]) | {SELF, TEST, block_step.SELF, block_step.TEST,
        blocks.SELF, blocks.TEST,
        "examples/weather_scenarios/fv_point_3h_seed_linear_probe.py", "tests/test_fv_point_3h_seed_linear_probe.py",
        "examples/weather_scenarios/fv_point_3h_merit_continuation.py",
        "examples/weather_scenarios/fv_point_3h_accepted_endpoint_audit.py",
        "examples/weather_scenarios/fv_point_3h_endpoint_diagnostics.py",
        "examples/weather_scenarios/fv_diagnostic_guard.py",
        "examples/weather_scenarios/fv86_resource_runner.py",
        "src/advar/fv_point_research_problem.py", "src/advar/transport.py",
        "src/advar/variational.py", "src/advar/nowcast.py"}


def _load_cached(plan_path: Path, plan_sha256: str):
    if not plan_path.resolve().is_relative_to(ROOT) or sha(plan_path) != plan_sha256:
        raise ValueError("bounded-step caller plan identity mismatch")
    for path, expected in ((ARCHIVE, ARCHIVE_SHA), (EXPERIMENT, EXPERIMENT_SHA),
        (CHECKPOINT, CHECKPOINT_SHA), (BASE_AUDIT, BASE_AUDIT_SHA),
        (BASE_PARENT, BASE_PARENT_SHA), (BASE_RESOURCE, BASE_RESOURCE_SHA)):
        if sha(path) != expected:
            raise ValueError(f"pinned cached endpoint artifact changed: {path.name}")
    raw = json.loads(ARCHIVE.read_text())
    plan = json.loads(plan_path.read_text())
    source_files = plan.get("source_files")
    if not isinstance(source_files, dict) or not _required_source_files(raw) <= set(source_files):
        raise ValueError("bounded-step plan must pin the current producer/test and original objective dependencies")
    for name, digest in source_files.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or sha(source) != digest:
            raise ValueError(f"bounded-step plan source pin changed: {name}")
    ledger = json.loads(EXPERIMENT.read_text())
    parent, resource = json.loads(BASE_PARENT.read_text()), json.loads(BASE_RESOURCE.read_text())
    saved = json.loads(BASE_AUDIT.read_text())
    checkpoint = json.loads(CHECKPOINT.read_text())
    accepted = diagnostic.extract_last_accepted_trial(raw)
    if (ledger.get("phase") != "finished" or ledger.get("stop_reason") != "fresh curvature diagnostic completed"
            or ledger.get("source_unchanged") is not True or ledger.get("original_archive_unchanged") is not True
            or ledger.get("source_before") != ledger.get("source_after")
            or len(ledger.get("attempts", [])) != 2
            or ledger["attempts"][-1].get("numerical_status") != "endpoint_diagnostic_completed"
            or parent.get("execution_status") != "completed"
            or parent.get("child_read_error") is not None or parent.get("endpoint_identity_matches_archive") is not True
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("elapsed_seconds", math.inf) > WALL_SECONDS
            or resource.get("wall_limit_seconds") != WALL_SECONDS
            or resource.get("rss_limit_bytes") != RSS_BYTES
            or resource.get("sampled_peak_rss_bytes", math.inf) > RSS_BYTES):
        raise ValueError("pinned checkpoint experiment is not a completed same-point diagnostic")
    if (raw.get("last_accepted_control_sha256") != CONTROL_SHA
            or raw.get("parameters_sha256") != PARAMETERS_SHA
            or saved.get("endpoint_control_sha256") != CONTROL_SHA
            or saved.get("parameters_sha256") != PARAMETERS_SHA
            or saved.get("phase") != "finished"
            or saved.get("numerical_status") != "endpoint_diagnostic_completed"
            or saved.get("source_unchanged") is not True
            or saved.get("source_before") != saved.get("source_after")
            or saved.get("input_unchanged") is not True
            or saved.get("optimizer_step_applied") is not False
            or saved.get("full_root_claim") is not False
            or saved.get("response_computed") is not False):
        raise ValueError("stored H/g report is not the declared read-only cd6b endpoint")
    if (checkpoint.get("status") != "completed" or len(checkpoint.get("columns", [])) != 26
            or checkpoint.get("audit_hvp", {}).get("status") != "passed"):
        raise ValueError("cached Hessian checkpoint is not fully qualified")
    result = checkpoint.get("final_result")
    hessian_data = saved.get("fresh_hessian")
    stored_gradient = torch.tensor(saved.get("gradient"), dtype=torch.float64)
    header = checkpoint.get("header", {})
    if (stored_gradient.shape != (26,) or tensor_sha(stored_gradient) != GRADIENT_SHA
            or header.get("control", {}).get("sha256") != CONTROL_SHA
            or header.get("parameters", {}).get("sha256") != PARAMETERS_SHA
            or header.get("gradient", {}).get("sha256") != GRADIENT_SHA):
        raise ValueError("stored gradient/checkpoint endpoint header differs from the pinned c/p/g")
    if (not isinstance(result, dict) or not isinstance(hessian_data, dict)
            or result.get("hessian_sha256") != HESSIAN_SHA
            or hessian_data.get("hessian_sha256") != HESSIAN_SHA
            or result.get("hessian") != hessian_data.get("hessian")
            or result.get("hvp_columns") != 26 or result.get("hvp_calls_total") != 27):
        raise ValueError("stored Hessian bytes/report/checkpoint do not agree")
    if (saved.get("input_before") != raw.get("input_after")
            or saved.get("input_after") != raw.get("input_after")
            or saved.get("branch", {}).get("signature_sha256")
               != accepted.get("branch", {}).get("signature_sha256")):
        raise ValueError("stored endpoint input/branch differs from accepted trial")
    return raw, accepted, saved, checkpoint, torch.tensor(result["hessian"], dtype=torch.float64)


def run(*, plan_path: Path, plan_sha256: str, output: Path) -> dict[str, Any]:
    """Evaluate one bounded cached-H direction against the original full J."""
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    if output.exists() or output.is_symlink():
        raise ValueError("bounded-step output must be fresh")
    root = ROOT.resolve()
    raw, accepted, saved, checkpoint, hessian = _load_cached(plan_path, plan_sha256)
    _check_deadline(deadline)
    paths = _source_paths(plan_path.resolve(), raw)
    before = {name: sha(root / name) for name in paths}
    if any(before.get(name) != digest for name, digest in raw["source_before"].items()):
        raise ValueError("one or more original R9 objective dependencies changed")
    original_control = torch.tensor(accepted["control"], dtype=torch.float64)
    parameters = torch.tensor(raw["parameters"], dtype=torch.float64)
    if (tensor_sha(original_control) != CONTROL_SHA or tensor_sha(parameters) != PARAMETERS_SHA
            or tensor_sha(hessian) != HESSIAN_SHA):
        raise ValueError("pinned control, parameters or cached Hessian hash changed")
    runtime = blocks.runtime_identity()
    if runtime != saved.get("runtime") or runtime != saved.get("runtime_after"):
        raise ValueError("runtime differs from the saved curvature diagnostic")
    _check_deadline(deadline)
    problem, original, _seed_control, fixed_parameters, truth, base_identity = seed._prepare_fixed_seed()
    if not torch.equal(parameters, fixed_parameters):
        raise ValueError("reconstructed original parameter vector changed")
    input_before = seed._input_identity(problem, original, original_control, fixed_parameters, truth)
    if (input_before != raw.get("input_after") or input_before != saved.get("input_after")
            or base_identity.get("archived_input") != input_before.get("archived_input")):
        raise ValueError("reconstructed original problem/input identity differs from stored endpoint")
    _check_deadline(deadline)
    branch, margins = tail._full_current_branch(problem, original_control, fixed_parameters)
    _check_deadline(deadline)
    if (branch.get("status") != "passed_strict_branch" or branch.get("euler_stages") != 3600
            or branch.get("choice_stage_count") != 3600 or branch.get("face_sign_stage_count") != 3600
            or branch.get("signature_sha256") != saved.get("branch", {}).get("signature_sha256")
            or not seed._valid_margins(margins, complete=True)):
        raise StepRefusal("fresh original endpoint strict branch/margins differ from cached base")
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    fresh = tail._fresh_merit(problem, original_control, fixed_parameters, gradient_fn)
    _check_deadline(deadline)
    if fresh is None:
        raise StepRefusal("fresh original endpoint J/g/Phi is nonfinite")
    base_j, gradient, phi = fresh
    checks: dict[str, Any] = {"objective": audit._metric(float(accepted["objective"]), float(base_j)),
              "phi": audit._metric(float(accepted["phi"]), float(phi))}
    checks["gradient_components"] = [audit._metric(float(a), float(b)) for a, b in
        zip(torch.tensor(accepted["gradient"], dtype=torch.float64), gradient, strict=True)]
    if (not checks["objective"]["passed"] or not checks["phi"]["passed"]
            or not all(check["passed"] for check in checks["gradient_components"])):
        raise StepRefusal("fresh base J/Phi/full gradient differs from archived accepted point")
    step, direction_audit = exact_shifted_direction(hessian, gradient, MU)
    _check_deadline(deadline)
    alpha_start = min(1.0, RADIUS / float(torch.linalg.vector_norm(step)))
    report: dict[str, Any] = {"schema": "advar.point3h-bounded-coupled-step.v1", "phase": "running",
        "numerical_status": "not_reached", "scope": "at most one cached-H shifted direction; original-J backtracking",
        "archive_sha256": ARCHIVE_SHA, "experiment_sha256": EXPERIMENT_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA, "base_audit_sha256": BASE_AUDIT_SHA,
        "plan_sha256": plan_sha256, "source_before": before, "source_unchanged": None,
        "runtime": runtime, "runtime_after": None, "input_before": input_before, "input_after": None,
        "base_control": original_control.tolist(), "base_control_sha256": tensor_sha(original_control),
        "parameters": parameters.tolist(), "parameters_sha256": tensor_sha(parameters),
        "base_objective": float(base_j), "base_phi": float(phi), "base_gradient": gradient.tolist(),
        "base_gradient_inf": float(gradient.abs().max()), "base_gradient_blocks": blocks.gradient_blocks(gradient),
        "base_branch": branch, "base_margins": margins, "cached_hessian_sha256": tensor_sha(hessian),
        "direction": step.tolist(), "direction_audit": direction_audit,
        "policy": {"dynamics_shift_mu": MU, "whole_control_radius": RADIUS,
                   "alpha_start": alpha_start, "dyadic_backtracks": MAX_BACKTRACKS,
                   "armijo_c1": ALPHA_C1, "acceptance": "original J Armijo and own strict endpoint branch/margins",
                   "branch_path_certificate": False},
        "trials": [], "optimizer_steps_max": 1, "optimizer_steps_applied": 0,
        "response_computed": False, "adjoint_computed": False, "score_computed": False,
        "full_root_claim": False, "minimum_claim": False, "physical_validated": False,
        "internal_budget_seconds": INTERNAL_SECONDS, "outer_wall_seconds": WALL_SECONDS,
        "rss_limit_bytes": RSS_BYTES}
    _write(output, report)

    def record_trial(row: dict[str, Any]) -> None:
        report["trials"].append(row)
        _write(output, report)

    def branch_fn(candidate: Tensor, p: Tensor):
        return tail._full_current_branch(problem, candidate, p)

    final_control = original_control
    try:
        accepted_control, trials, status = bounded_original_j_search(
            original_control, fixed_parameters, base_j, gradient, step, problem.objective,
            gradient_fn, branch_fn, branch["signature_sha256"], deadline=deadline,
            on_trial=record_trial)
        if not report["trials"] and trials:
            report["trials"].extend(trials)
        if status == "one_original_J_step_accepted":
            if accepted_control is None:
                raise ValueError("accepted search omitted its control")
            final_control = accepted_control
            report.update(numerical_status=status, optimizer_steps_applied=1,
                          accepted_control=accepted_control.tolist(),
                          accepted_control_sha256=tensor_sha(accepted_control),
                          accepted_branch_signature_changed=report["trials"][-1]["branch_signature_changed"])
        else:
            report.update(numerical_status=status, refusal=status)
    except StepRefusal as error:
        report.update(numerical_status="step_refusal", refusal=f"{type(error).__name__}: {error}")
    input_after = seed._input_identity(problem, original, final_control,
                                       fixed_parameters, truth)
    after = {name: sha(root / name) for name in paths}
    runtime_after = blocks.runtime_identity()
    if (after != before or runtime_after != runtime
            or not audit._fixed_input_matches(input_after, input_before,
                tensor_sha(final_control))):
        report.update(phase="integrity_refused", numerical_status="integrity_refusal",
                      source_after=after, source_unchanged=after == before,
                      input_after=input_after, runtime_after=runtime_after,
                      refusal="source/input/runtime changed during bounded step")
        _write(output, report)
        raise ValueError("bounded-step integrity check failed")
    report.update(phase="finished", source_after=after, source_unchanged=True,
                  input_after=input_after, fixed_input_unchanged=True,
                  runtime_after=runtime_after, elapsed_seconds=time.monotonic() - started)
    _write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.output_directory.resolve()
    output = directory / "step.json"
    if args.child:
        # Internal bare mode requires the outer guard; it is not a public run path.
        try:
            run(plan_path=args.plan.resolve(), plan_sha256=args.plan_sha256, output=output)
        except StepRefusal as error:
            if not output.exists():
                _write(output, {"phase": "finished", "numerical_status": "step_budget_refusal" if isinstance(error, BudgetRefusal) else "initial_step_refusal",
                    "refusal": str(error), "source_unchanged": None,
                    "integrity_status": "not_verified",
                    "optimizer_steps_applied": 0, "response_computed": False, "full_root_claim": False})
        return
    # Validate the fixed inputs before reserving the only guarded launch.
    _load_cached(args.plan.resolve(), args.plan_sha256)
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(ROOT / ".venv/bin/python"), "-m",
               "examples.weather_scenarios.fv_point_3h_bounded_coupled_step",
               "--child", "--plan", str(args.plan.resolve()),
               "--plan-sha256", args.plan_sha256, "--output-directory", str(directory)]
    resource = run_guarded_diagnostic(
        command, wall_seconds=WALL_SECONDS, rss_bytes=RSS_BYTES,
        report_path=directory / "step.resource.json", log_path=directory / "step.log")
    execution = execution_status(resource)
    parent: dict[str, Any] = {"execution_status": execution, "resource": resource,
                             "child_read_error": None, "numerical_status": "not_reached"}
    parent_path = directory / "step.run.json"
    # Keep process/resource failure evidence even if child JSON cannot be read.
    _write(parent_path, parent)
    try:
        child = json.loads(output.read_text())
        if not isinstance(child, dict):
            raise ValueError("child step report must be a JSON object")
        parent.update(child_sha256=sha(output), numerical_status=child.get("numerical_status", "not_reached"))
        if execution == "completed" and (child.get("phase") != "finished" or child.get("source_unchanged") is not True
                or child.get("fixed_input_unchanged") is not True or child.get("runtime_after") != child.get("runtime")):
            parent["execution_status"] = "failed"
            parent["execution_failure_reason"] = "child_completion_or_integrity_not_verified"
    except (OSError, ValueError) as error:
        parent.update(execution_status="failed" if execution == "completed" else execution,
                      child_read_error=f"{type(error).__name__}: {error}")
    _write(parent_path, parent)
    print(json.dumps({"execution_status": parent["execution_status"],
                      "numerical_status": parent["numerical_status"], "output": str(output)}))


def execution_status(resource: dict[str, Any]) -> str:
    elapsed = resource.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
        return "failed"
    if resource.get("received_sigterm") or resource.get("resource_termination") == "cancelled":
        return "failed"
    if (resource.get("monitor_error") or resource.get("child_process_group_cleanup_error")
            or resource.get("child_process_group_cleanup_sent") is not False):
        return "failed"
    if resource.get("resource_termination") not in {None, "wall_time_limit", "rss_limit"}:
        return "failed"
    if resource.get("resource_termination") in {"wall_time_limit", "rss_limit"} or elapsed > WALL_SECONDS:
        return "resource_limited"
    if resource.get("exit_code") != 0:
        return "failed"
    return "completed"


if __name__ == "__main__":
    main()
