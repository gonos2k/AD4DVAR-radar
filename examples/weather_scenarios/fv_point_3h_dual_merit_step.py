"""One cached-direction step that requires descent in both J and gradient merit.

The stored a35 direction and its stored live ``H s`` are fixed inputs.  This
diagnostic makes no Hessian-vector products and does not claim a root, response,
score, or forecast validation.
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

from examples.weather_scenarios import fv_point_3h_accepted_curvature as curvature
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as search
from examples.weather_scenarios import fv_point_3h_current_newton_step as diagnostics
from examples.weather_scenarios import fv_point_3h_merit_continuation as merit
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

ROOT = curvature.ROOT
EVIDENCE = curvature.EVIDENCE
BASE = EVIDENCE / "a35_live_hvp_newton_20261006_attempt1/step.json"
BASE_PARENT = BASE.with_suffix(".run.json")
BASE_RESOURCE = BASE.with_suffix(".resource.json")
BASE_PLAN = EVIDENCE / "NEWA35_LIVE_HVP_NEWTON_PLAN_20261006.json"
OLD_CURVATURE = EVIDENCE / "f82c_fresh_curvature_20261005/attempt_1/audit.json"
SELF = "examples/weather_scenarios/fv_point_3h_dual_merit_step.py"
TEST = "tests/test_fv_point_3h_dual_merit_step.py"
PLAN = EVIDENCE / "A35_DUAL_MERIT_STEP_PLAN_20261006.json"
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 240., 300., 1024**3
RADIUS, MAX_CANDIDATES, C1_J, C1_PHI = .05, 16, 1e-4, 1e-4
RESIDUAL_RTOL = 1e-10


class StepRefusal(search.StepRefusal):
    """Known policy, numerical, or cooperative-budget refusal."""


def _policy() -> dict[str, Any]:
    return {"j_armijo_c1": C1_J, "phi_armijo_c1": C1_PHI,
            "radius": RADIUS, "max_candidates": MAX_CANDIDATES,
            "internal_seconds": INTERNAL_SECONDS, "outer_seconds": WALL_SECONDS,
            "rss_bytes": RSS_BYTES, "guarded_launches": 1, "new_hvp": 0}


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise search.BudgetRefusal("240-second internal budget exhausted during dual-merit step")


def _direction_audit(raw: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
    """Validate the saved solve receipt and return immutable s, g, and Hs."""
    try:
        direction = torch.tensor(raw["direction"], dtype=torch.float64)
        gradient = torch.tensor(raw["base_gradient"], dtype=torch.float64)
        hs = torch.tensor(raw["live_directional_model"]["H_s"], dtype=torch.float64)
        residual = torch.tensor(raw["solve"]["true_residual"], dtype=torch.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cached a35 direction receipt is malformed") from error
    tensors = (direction, gradient, hs, residual)
    if any(x.shape != (26,) or x.device.type != "cpu" or not bool(torch.isfinite(x).all()) for x in tensors):
        raise ValueError("cached direction, gradient, Hs, and residual must be finite CPU FP64 length 26")
    g_norm = torch.linalg.vector_norm(gradient)
    residual_relative = float(torch.linalg.vector_norm(residual) / g_norm) if float(g_norm) > 0 else math.inf
    residual_consistency = torch.linalg.vector_norm(hs + gradient - residual)
    consistency_scale = torch.maximum(g_norm, torch.tensor(torch.finfo(torch.float64).tiny))
    gts, gths = float(gradient @ direction), float(gradient @ hs)
    if (not math.isfinite(residual_relative) or residual_relative > RESIDUAL_RTOL
            or abs(residual_relative - raw["solve"].get("true_relative_residual", math.inf)) > 1e-12
            or float(residual_consistency / consistency_scale) > 1e-12
            or not gts < 0 or not gths < 0
            or abs(gts - raw["solve"].get("g_dot_s", math.inf)) > 1e-12 * max(1., abs(gts))):
        raise ValueError("cached direction fails its true-residual or dual-descent receipt")
    return direction, gradient, hs


def _archive_names() -> set[str]:
    return {path.relative_to(ROOT).as_posix() for path in
            (BASE, BASE_PARENT, BASE_RESOURCE, BASE_PLAN, OLD_CURVATURE)}


def _mark_candidate_not_committed(record: dict[str, Any], reason: str) -> None:
    """Clear provisional acceptance when closure fails before commit."""
    if record.get("optimizer_steps_applied") == 0 and record.get("trials"):
        row = record["trials"][-1]
        if row.get("status") == "accepted":
            row.update(status="candidate_not_committed", accepted=False,
                       commit_refusal=reason)


def load_base(plan_path: Path, plan_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not plan_path.resolve().is_relative_to(ROOT) or curvature.sha(plan_path) != plan_sha:
        raise ValueError("dual-merit plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    raw = json.loads(BASE.read_text())
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    required_sources = set(raw["source_before"]) | {
        SELF, TEST, "examples/weather_scenarios/fv_point_3h_current_newton_step.py",
        "examples/weather_scenarios/fv_point_3h_merit_continuation.py",
        "examples/weather_scenarios/fv_point_3h_accepted_curvature.py",
        "examples/weather_scenarios/fv_point_3h_hvp_newton_step.py",
        "examples/weather_scenarios/fv_point_3h_seed_linear_probe.py",
        "examples/weather_scenarios/fv_point_3h_terminal_block_schur.py",
        "examples/weather_scenarios/fv_diagnostic_guard.py",
    }
    if not isinstance(sources, dict) or not required_sources <= set(sources):
        raise ValueError("dual-merit plan must pin the producer sources, caller, and focused tests")
    if not isinstance(archives, dict) or not _archive_names() <= set(archives):
        raise ValueError("dual-merit plan must pin the raw step, parent, resource, and producer plan")
    if plan.get("policy") != _policy():
        raise ValueError("dual-merit plan policy differs from the fixed two-merit search")
    for name, digest in {**sources, **archives}.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or (ROOT / name).is_symlink() or curvature.sha(path) != digest:
            raise ValueError(f"dual-merit source/evidence pin changed: {name}")
    old_audit = json.loads(OLD_CURVATURE.read_text())
    if (curvature.sha(OLD_CURVATURE) != raw.get("curvature_sha256")
            or json.loads(BASE_PLAN.read_text()).get("archive_files", {}).get(OLD_CURVATURE.relative_to(ROOT).as_posix())
            != curvature.sha(OLD_CURVATURE)):
        raise ValueError("cached live-HVP step curvature source chain changed")
    diagnostics._require_cached_objective_sources(plan, old_audit)
    parent = json.loads(BASE_PARENT.read_text())
    resource = json.loads(BASE_RESOURCE.read_text())
    if (raw.get("phase") != "finished" or raw.get("numerical_status") != "one_live_HVP_original_J_step_accepted"
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("base_control_sha256") != plan.get("control_sha256")
            or raw.get("parameters_sha256") != plan.get("parameters_sha256")
            or raw.get("base_control_sha256") != "a35e2f6ef000cca4b593a3e5ed49f1c3e94b951450f2f235902a50e1dde6647c"
            or raw.get("accepted_control_sha256") == raw.get("base_control_sha256")
            or raw.get("plan_sha256") != curvature.sha(BASE_PLAN)
            or len([item for item in raw.get("trials", []) if item.get("status") == "accepted"]) != 1
            or raw["trials"][0].get("control_sha256") != raw.get("accepted_control_sha256")
            or parent.get("execution_status") != "completed" or parent.get("child_read_error") is not None
            or parent.get("child_sha256") != curvature.sha(BASE)
            or search.execution_status(resource) != "completed"):
        raise ValueError("raw a35 live-HVP direction receipts are not complete and unchanged")
    _direction_audit(raw)
    return plan, raw


def dual_merit_search(control: Tensor, direction: Tensor, base_j: Tensor, base_gradient: Tensor,
                      base_phi: Tensor, hs: Tensor, objective: Callable[[Tensor, Tensor], Tensor],
                      gradient_fn: Callable[[Tensor, Tensor], Tensor], parameters: Tensor,
                      branch_fn: Callable[[Tensor, Tensor], tuple[dict[str, Any], dict[str, Any]]],
                      deadline: float, base_branch_signature: str,
                      on_trial: Callable[[dict[str, Any]], None] | None = None
                      ) -> tuple[Tensor | None, list[dict[str, Any]], str]:
    """Backtrack along fixed s; require actual J and Phi Armijo plus strict branch margins."""
    controls = (control, direction, base_gradient, hs)
    if any(x.shape != (26,) for x in controls) or parameters.shape != (13,):
        raise ValueError("dual-merit search expects 26 controls and a 13-parameter vector")
    if any(x.dtype != torch.float64 or x.device.type != "cpu" or not bool(torch.isfinite(x).all())
           for x in (*controls, parameters)):
        raise ValueError("dual-merit search inputs must be finite CPU FP64")
    if any(not isinstance(x, Tensor) or x.shape != () or x.dtype != torch.float64
           or x.device.type != "cpu" or not bool(torch.isfinite(x)) for x in (base_j, base_phi)):
        raise ValueError("dual-merit baselines must be finite CPU FP64 scalars")
    slope_j, slope_phi = float(base_gradient @ direction), float(base_gradient @ hs)
    norm = float(torch.linalg.vector_norm(direction))
    if not math.isfinite(norm) or norm <= 0 or not slope_j < 0 or not slope_phi < 0:
        raise StepRefusal("cached direction is not a finite dual-descent direction")
    alpha_start = min(1., RADIUS / norm)
    trials: list[dict[str, Any]] = []
    for backtrack in range(MAX_CANDIDATES):
        alpha = alpha_start * 2.**-backtrack
        candidate = control + alpha * direction
        threshold_j = float(base_j) + C1_J * alpha * slope_j
        threshold_phi = float(base_phi) + C1_PHI * alpha * slope_phi
        row: dict[str, Any] = {"backtrack": backtrack, "alpha": alpha,
            "control": candidate.tolist(), "control_sha256": curvature.tensor_sha(candidate),
            "J_armijo_threshold": threshold_j, "Phi_armijo_threshold": threshold_phi,
            "J_directional_slope": slope_j, "Phi_directional_slope": slope_phi,
            "base_branch_signature_sha256": base_branch_signature,
            "new_hvp": 0, "accepted": False}
        def save(status: str) -> None:
            row["status"] = status
            trials.append(row)
            if on_trial is not None:
                on_trial(row)
        def evaluate(phase: str, callback: Callable[[], Any]) -> Any:
            try:
                _check_deadline(deadline)
                value = callback()
                _check_deadline(deadline)
                return value
            except Exception as error:
                row.update(failed_phase=phase, failure=f"{type(error).__name__}: {error}",
                           reasons=[f"{phase} evaluation raised"])
                save("budget_refused_during_evaluation" if isinstance(error, search.BudgetRefusal) else "evaluation_error")
                raise
        if time.monotonic() >= deadline:
            row["reasons"] = ["internal deadline reached before evaluation"]
            save("budget_refused_before_evaluation")
            return None, trials, "dual_merit_budget_refusal"
        value = evaluate("objective", lambda: objective(candidate, parameters))
        if not isinstance(value, Tensor) or value.shape != () or value.dtype != torch.float64 or value.device.type != "cpu":
            row["reasons"] = ["candidate original-J callback contract changed"]
            save("callback_contract_error")
            raise ValueError("candidate original-J callback contract changed")
        j_finite = bool(torch.isfinite(value))
        row["objective"] = float(value) if j_finite else None
        if not j_finite:
            row["reasons"] = ["candidate J is nonfinite"]
            save("nonfinite_objective")
            continue
        gradient = evaluate("gradient", lambda: gradient_fn(candidate, parameters))
        if not isinstance(gradient, Tensor) or gradient.shape != (26,) or gradient.dtype != torch.float64 or gradient.device.type != "cpu":
            row["reasons"] = ["candidate full-gradient callback contract changed"]
            save("callback_contract_error")
            raise ValueError("candidate full-gradient callback contract changed")
        if not bool(torch.isfinite(gradient).all()):
            row.update(gradient=None, gradient_blocks=None, gradient_l2=None,
                       phi=None, reasons=["candidate full gradient is nonfinite"])
            save("nonfinite_gradient")
            continue
        phi = torch.dot(gradient, gradient) / 2
        row.update(gradient=gradient.tolist(),
                   gradient_blocks=blocks.gradient_blocks(gradient), gradient_l2=float(gradient.norm()))
        if not bool(torch.isfinite(phi)):
            row.update(phi=None, reasons=["candidate Phi is nonfinite"])
            save("nonfinite_phi")
            continue
        row["phi"] = float(phi)
        def branch_evaluation() -> tuple[dict[str, Any], dict[str, Any], bool]:
            branch, margins = branch_fn(candidate, parameters)
            strict = (branch.get("status") == "passed_strict_branch"
                      and branch.get("euler_stages") == 3600
                      and branch.get("choice_stage_count") == 3600
                      and branch.get("face_sign_stage_count") == 3600
                      and merit.seed_linear._valid_margins(margins, complete=True))
            return branch, margins, strict
        branch, margins, strict = evaluate("branch", branch_evaluation)
        j_pass = float(value) <= threshold_j
        phi_pass = float(phi) <= threshold_phi
        reasons = []
        if not j_pass: reasons.append("actual J exceeds J Armijo threshold")
        if not phi_pass: reasons.append("actual Phi exceeds Phi Armijo threshold")
        if not strict: reasons.append("candidate full strict branch or margins failed")
        row.update(branch=branch, margins=margins, strict_point_passed=strict,
                   branch_signature_changed=branch.get("signature_sha256") != base_branch_signature,
                   J_armijo_passed=j_pass, Phi_armijo_passed=phi_pass, reasons=reasons)
        if time.monotonic() >= deadline:
            row["reasons"].append("internal deadline reached after evaluation")
            save("budget_refused_after_evaluation")
            return None, trials, "dual_merit_budget_refusal"
        if j_pass and phi_pass and strict:
            row["accepted"] = True
            save("accepted")
            return candidate, trials, "one_dual_merit_step_accepted"
        save("rejected")
    return None, trials, "dual_merit_grid_exhausted"


def run(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    if output.exists() or output.is_symlink():
        raise ValueError("dual-merit output must be fresh")
    plan, raw = load_base(plan_path, plan_sha)
    names = set(plan["source_files"]) | set(plan["archive_files"]) | {plan_path.relative_to(ROOT).as_posix()}
    before = {name: curvature.sha(ROOT / name) for name in names}
    if (before[plan_path.relative_to(ROOT).as_posix()] != plan_sha or
            any(before[n] != digest for n, digest in {**plan["source_files"], **plan["archive_files"]}.items())):
        raise ValueError("dual-merit plan changed after preflight")
    c = torch.tensor(raw["base_control"], dtype=torch.float64)
    p = torch.tensor(raw["parameters"], dtype=torch.float64)
    direction, saved_gradient, hs = _direction_audit(raw)
    problem, original, _seed_control, fixed_p, truth, _ = curvature.seed._prepare_fixed_seed()
    identity = curvature.seed._input_identity(problem, original, c, p, truth)
    if (curvature.tensor_sha(c) != raw["base_control_sha256"]
            or not torch.equal(p, fixed_p) or identity != raw["input_before"]
            or curvature.blocks.runtime_identity() != raw["runtime"]):
        raise ValueError("reconstructed original a35 base control, parameters, input, or runtime changed")
    record: dict[str, Any] = {"phase": "running", "numerical_status": "not_reached",
        "scope": "one same-direction a35 dual-merit diagnostic; cached Hs only, no new HVP/PCG/root/response/score/forecast claim",
        "source_before": before, "plan_sha256": plan_sha, "base_step_sha256": curvature.sha(BASE),
        "base_control": c.tolist(), "base_control_sha256": curvature.tensor_sha(c),
        "parameters_sha256": curvature.tensor_sha(p), "input_before": identity,
        "runtime": raw["runtime"], "policy": plan["policy"], "trials": [],
        "optimizer_steps_applied": 0, "new_hvp": 0, "pcg_iterations": 0,
        "score_computed": False, "response_computed": False, "full_root_claim": False,
        "global_spd_claim": False, "physical_validated": False,
        "accepted_point_curvature": "not_computed", "forecast_score_computed": False}
    search._write(output, record)
    final_c = c
    candidate_update: dict[str, Any] | None = None
    try:
        _check_deadline(deadline)
        branch, margins = curvature.tail._full_current_branch(problem, c, p)
        _check_deadline(deadline)
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        fresh = curvature.tail._fresh_merit(problem, c, p, gradient_fn)
        if fresh is None:
            raise StepRefusal("fresh a35 base J/g/Phi is nonfinite")
        j, gradient, phi = fresh
        checks = curvature._endpoint_metrics({"objective": raw["base_objective"],
            "phi": raw["base_phi"], "gradient_inf": float(torch.tensor(raw["base_gradient"], dtype=torch.float64).abs().max()),
            "gradient": raw["base_gradient"]}, j, gradient, phi)
        if (branch != raw["base_branch"] or margins != raw["base_margins"]
                or not curvature.seed._valid_margins(margins, complete=True)
                or not curvature._endpoint_metrics_pass(checks)
                or not torch.equal(gradient, saved_gradient)
                or abs(float(phi) - float(gradient @ gradient / 2)) > 1e-14):
            raise StepRefusal("fresh a35 J/g/Phi or complete strict branch differs from cached live-HVP base")
        record.update(base_objective=float(j), base_phi=float(phi), base_gradient=gradient.tolist(),
            base_gradient_blocks=blocks.gradient_blocks(gradient), base_gradient_l2=float(gradient.norm()),
            base_branch=branch, base_margins=margins, base_metric_checks=checks,
            direction=direction.tolist(), direction_sha256=curvature.tensor_sha(direction),
            live_directional_model=raw["live_directional_model"],
            direction_audit={"true_relative_residual": raw["solve"]["true_relative_residual"],
                "g_dot_s": float(gradient @ direction), "g_dot_Hs": float(gradient @ hs),
                "immutable_cached_direction": True, "new_hvp": 0})
        search._write(output, record)
        def save_trial(row: dict[str, Any]) -> None:
            row["base_branch_signature_sha256"] = branch["signature_sha256"]
            record["trials"].append(row)
            search._write(output, record)
        accepted, _, status = dual_merit_search(c, direction, j, gradient, phi, hs,
            problem.objective, gradient_fn, p,
            lambda x, pp: curvature.tail._full_current_branch(problem, x, pp),
            deadline, branch["signature_sha256"], on_trial=save_trial)
        record["numerical_status"] = status
        if accepted is not None:
            _check_deadline(deadline)
            signature, scope = problem.branch_check(accepted, p)
            _check_deadline(deadline)
            spec = problem.frozen.fv_transport
            if spec is None:
                raise ValueError("FV transport disappeared")
            partition = curvature._branch_partition(signature, tuple(problem.layout["observation_times_seconds"]),
                spec.substeps_per_interval)
            row = record["trials"][-1]
            if partition["full_signature_sha256"] != row["branch"]["signature_sha256"]:
                raise StepRefusal("accepted branch signature changed during independent partition capture")
            delta = accepted - c
            _check_deadline(deadline)
            state0 = diagnostics.physical_state(problem, c, p)
            _check_deadline(deadline)
            state1 = diagnostics.physical_state(problem, accepted, p)
            _check_deadline(deadline)
            changes = {key: (state1[key] - state0[key]).tolist() for key in state0}
            candidate_update = dict(optimizer_steps_applied=1, accepted_control=accepted.tolist(),
                accepted_control_sha256=curvature.tensor_sha(accepted), accepted_branch_partition=partition,
                branch_scope=scope, model_diagnostics=diagnostics.model_diagnostics(float(j), gradient, None,
                    delta, row["objective"], torch.tensor(row["gradient"], dtype=torch.float64),
                    hessian_delta=row["alpha"] * hs),
                physical_changes={"values": changes,
                    "base": {key: value.tolist() for key, value in state0.items()},
                    "candidate": {key: value.tolist() for key, value in state1.items()},
                    **diagnostics.physical_unit_labels()},
                numerical_status="one_dual_merit_step_accepted")
            _check_deadline(deadline)
            final_c = accepted
    except search.StepRefusal as error:
        record.update(numerical_status="budget_refusal" if isinstance(error, search.BudgetRefusal) else "step_refusal",
                      refusal=str(error))
        _mark_candidate_not_committed(record, str(error))
    except Exception as error:
        record.update(phase="programming_error", numerical_status="programming_error",
                      failure=f"{type(error).__name__}: {error}")
        _mark_candidate_not_committed(record, record["failure"])
        search._write(output, record)
        raise
    after = {name: curvature.sha(ROOT / name) for name in names}
    final_identity = curvature.seed._input_identity(problem, original, final_c, p, truth)
    intact = (before == after and curvature.blocks.runtime_identity() == raw["runtime"]
              and curvature.audit._fixed_input_matches(final_identity, identity, curvature.tensor_sha(final_c)))
    if candidate_update is not None:
        try:
            _check_deadline(deadline)
        except search.BudgetRefusal as error:
            record.update(numerical_status="budget_refusal", refusal=str(error))
            candidate_update = None
            final_c = c
            final_identity = curvature.seed._input_identity(problem, original, c, p, truth)
            intact = (before == after and curvature.blocks.runtime_identity() == raw["runtime"]
                      and curvature.audit._fixed_input_matches(final_identity, identity, curvature.tensor_sha(c)))
        if candidate_update is not None and intact:
            record.update(candidate_update)
    if record["trials"] and record["trials"][-1].get("status") == "accepted" and not record["optimizer_steps_applied"]:
        _mark_candidate_not_committed(record, "post-candidate integrity closure refused")
    record.update(phase="finished" if intact else "integrity_refused", source_after=after,
        source_unchanged=before == after, fixed_input_unchanged=intact, input_after=final_identity,
        runtime_after=curvature.blocks.runtime_identity(), elapsed_seconds=time.monotonic() - started)
    if not intact:
        record["numerical_status"] = "integrity_refusal"
    search._write(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.child:
        run(args.plan.resolve(), args.plan_sha256, directory / "step.json")
        return
    load_base(args.plan.resolve(), args.plan_sha256)
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(ROOT / ".venv/bin/python"), "-m", "examples.weather_scenarios.fv_point_3h_dual_merit_step",
        "--child", "--plan", str(args.plan.resolve()), "--plan-sha256", args.plan_sha256,
        "--directory", str(directory)]
    resource = run_guarded_diagnostic(command, wall_seconds=WALL_SECONDS, rss_bytes=RSS_BYTES,
        report_path=directory / "step.resource.json", log_path=directory / "step.log")
    parent = {"execution_status": search.execution_status(resource), "resource": resource,
              "numerical_status": "not_reached", "child_read_error": None}
    search._write(directory / "step.run.json", parent)
    try:
        child = json.loads((directory / "step.json").read_text())
        if not isinstance(child, dict):
            raise ValueError("child result must be a JSON object")
        parent.update(numerical_status=child["numerical_status"], child_sha256=curvature.sha(directory / "step.json"))
        if parent["execution_status"] == "completed" and not (child.get("phase") == "finished" and child.get("fixed_input_unchanged") is True):
            parent["execution_status"] = "failed"
    except (OSError, ValueError, KeyError) as error:
        parent.update(execution_status="failed" if parent["execution_status"] == "completed" else parent["execution_status"],
                      child_read_error=str(error))
    search._write(directory / "step.run.json", parent)
    print(json.dumps({key: parent[key] for key in ("execution_status", "numerical_status")}))


if __name__ == "__main__":
    main()
