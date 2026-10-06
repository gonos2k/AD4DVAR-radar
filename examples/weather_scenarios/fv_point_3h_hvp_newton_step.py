"""One live-HVP Newton step at a35; the f82c matrix is a preconditioner only."""
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
from examples.weather_scenarios import fv_point_3h_schur_newton_step as block_step
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

ROOT = curvature.ROOT
EVIDENCE = curvature.EVIDENCE
BASE = EVIDENCE / "f82c_unshifted_newton_20261005_attempt1/step.json"
BASE_PARENT = BASE.with_suffix(".run.json")
BASE_RESOURCE = BASE.with_suffix(".resource.json")
CHECKPOINT = EVIDENCE / "f82c_fresh_curvature_20261005/checkpoint/hessian_checkpoint.json"
CURVATURE = EVIDENCE / "f82c_fresh_curvature_20261005/attempt_1/audit.json"
CURVATURE_PARENT = CURVATURE.with_suffix(".run.json")
CURVATURE_RESOURCE = CURVATURE.with_suffix(".resource.json")
SELF = "examples/weather_scenarios/fv_point_3h_hvp_newton_step.py"
TEST = "tests/test_fv_point_3h_hvp_newton_step.py"
PLAN = EVIDENCE / "NEWA35_LIVE_HVP_NEWTON_PLAN_20261006.json"
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 240., 300., 1024**3
PCG_RTOL, PCG_MAX = 1e-10, 104
HVP_MAX = 106


def counted_hvp(operator: Callable[[Tensor], Tensor], deadline: float,
                limit: int = HVP_MAX) -> tuple[Callable[[Tensor], Tensor], dict[str, int]]:
    """Count started/completed live products and stop before product limit+1."""
    counts = {"started": 0, "completed": 0}

    def apply(vector: Tensor) -> Tensor:
        search._check_deadline(deadline)
        if counts["started"] >= limit:
            raise search.BudgetRefusal(f"live HVP cap {limit} reached")
        counts["started"] += 1
        result = operator(vector)
        counts["completed"] += 1
        search._check_deadline(deadline)
        return result

    return apply, counts


def _archive_names() -> set[str]:
    return {p.relative_to(ROOT).as_posix() for p in
            (BASE, BASE_PARENT, BASE_RESOURCE, CURVATURE, CURVATURE_PARENT,
             CURVATURE_RESOURCE, CHECKPOINT)}


def _accepted_trial(base: dict[str, Any]) -> dict[str, Any]:
    accepted = [trial for trial in base.get("trials", []) if trial.get("status") == "accepted"]
    if (len(accepted) != 1 or accepted[0].get("control_sha256") != base.get("accepted_control_sha256")
            or accepted[0].get("strict_point_passed") is not True
            or accepted[0].get("armijo_passed") is not True):
        raise ValueError("raw step must contain exactly one accepted strict original-J point")
    return accepted[0]


def load_base(plan_path: Path, plan_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not plan_path.resolve().is_relative_to(ROOT) or curvature.sha(plan_path) != plan_sha:
        raise ValueError("live-HVP Newton plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    base = json.loads(BASE.read_text())
    audit = json.loads(CURVATURE.read_text())
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    required_sources = set(base["source_before"]) | {SELF, TEST,
        "examples/weather_scenarios/fv_point_3h_current_newton_step.py",
        "tests/test_fv_point_3h_current_newton_step.py",
        block_step.SELF, block_step.TEST,
        "examples/weather_scenarios/fv_point_3h_bounded_coupled_step.py",
        "tests/test_fv_point_3h_bounded_coupled_step.py"}
    if not isinstance(sources, dict) or not required_sources <= set(sources):
        raise ValueError("plan must pin the caller, focused tests, solver, and objective dependencies")
    if not isinstance(archives, dict) or not _archive_names() <= set(archives):
        raise ValueError("plan must pin accepted step, curvature, checkpoint, parent, and resource receipts")
    if (plan.get("control_sha256") != base.get("accepted_control_sha256")
            or plan.get("parameters_sha256") != base.get("parameters_sha256")):
        raise ValueError("plan control or fixed-parameter identity differs from accepted a35e2f point")
    for name, digest in {**sources, **archives}.items():
        path = (ROOT / name).resolve()
        if (not path.is_relative_to(ROOT.resolve()) or (ROOT / name).is_symlink()
                or curvature.sha(path) != digest):
            raise ValueError(f"live-HVP input/source pin changed: {name}")
    diagnostics._require_cached_objective_sources(plan, audit)
    base_parent = json.loads(BASE_PARENT.read_text())
    curvature_parent = json.loads(CURVATURE_PARENT.read_text())
    if (base.get("phase") != "finished" or base.get("numerical_status") != "one_original_J_step_accepted"
            or base.get("optimizer_steps_applied") != 1 or base.get("fixed_input_unchanged") is not True
            or base.get("source_unchanged") is not True
            or base.get("source_before") != base.get("source_after")
            or base_parent.get("execution_status") != "completed"
            or base_parent.get("child_read_error") is not None
            or base_parent.get("child_sha256") != curvature.sha(BASE)
            or search.execution_status(json.loads(BASE_RESOURCE.read_text())) != "completed"):
        raise ValueError("accepted f82c step receipts are not completed and unchanged")
    h = audit.get("curvature", {})
    checkpoint = json.loads(CHECKPOINT.read_text())
    if (audit.get("phase") != "finished" or audit.get("numerical_status") != "curvature_completed"
            or audit.get("source_unchanged") is not True or audit.get("fixed_input_unchanged") is not True
            or audit.get("source_before") != audit.get("source_after")
            or curvature_parent.get("execution_status") != "completed"
            or curvature_parent.get("child_read_error") is not None
            or curvature_parent.get("child_sha256") != curvature.sha(CURVATURE)
            or search.execution_status(json.loads(CURVATURE_RESOURCE.read_text())) != "completed"
            or checkpoint.get("status") != "completed" or h.get("hvp_columns") != 26
            or h.get("hvp_calls_total") != 27 or not h.get("minimum_eigenpair_audit", {}).get("passed")
            or h.get("hessian") != checkpoint.get("final_result", {}).get("hessian")):
        raise ValueError("f82c curvature is not a qualified completed SPD preconditioner source")
    policy = {"armijo_c1": 1e-4, "guarded_launches": 1,
              "internal_seconds": INTERNAL_SECONDS, "linear_relative_tolerance": PCG_RTOL,
              "max_candidates": 16, "max_hvp_calls": HVP_MAX,
              "outer_seconds": WALL_SECONDS, "pcg_max_iterations": PCG_MAX,
              "phi_is_acceptance_gate": False, "radius": .05, "rss_bytes": RSS_BYTES}
    if plan.get("policy") != policy:
        raise ValueError("live-HVP plan policy differs from the fixed one-step policy")
    _accepted_trial(base)
    return plan, base, audit


def run(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    if output.exists() or output.is_symlink():
        raise ValueError("live-HVP output must be fresh")
    plan, base, audit = load_base(plan_path, plan_sha)
    names = set(plan["source_files"]) | set(plan["archive_files"]) | {plan_path.relative_to(ROOT).as_posix()}
    before = {name: curvature.sha(ROOT / name) for name in names}
    expected_pins = {**plan["source_files"], **plan["archive_files"]}
    if (before[plan_path.relative_to(ROOT).as_posix()] != plan_sha
            or any(before.get(name) != digest for name, digest in expected_pins.items())):
        raise ValueError("live-HVP plan/input/source changed after preflight")
    c = torch.tensor(base["accepted_control"], dtype=torch.float64)
    p = torch.tensor(base["parameters"], dtype=torch.float64)
    h = torch.tensor(audit["curvature"]["hessian"], dtype=torch.float64)
    if (curvature.tensor_sha(c) != base["accepted_control_sha256"]
            or curvature.tensor_sha(p) != base["parameters_sha256"]
            or curvature.tensor_sha(h) != audit["curvature"]["hessian_sha256"]):
        raise ValueError("accepted control, parameter, or preconditioner tensor identity changed")
    problem, original, _, fixed_p, truth, _ = curvature.seed._prepare_fixed_seed()
    if not torch.equal(p, fixed_p):
        raise ValueError("accepted-step parameters differ from reconstructed fixed problem")
    identity = curvature.seed._input_identity(problem, original, c, p, truth)
    if identity != base["input_after"] or curvature.blocks.runtime_identity() != base["runtime"]:
        raise ValueError("accepted control/input/runtime identity changed")

    record: dict[str, Any] = {"phase": "running", "numerical_status": "not_reached",
        "scope": "one bounded original-J live-HVP Newton step at the accepted a35e2f point",
        "source_before": before, "source_after": None, "plan_sha256": plan_sha,
        "accepted_step_sha256": curvature.sha(BASE), "curvature_sha256": curvature.sha(CURVATURE),
        "base_control": c.tolist(), "base_control_sha256": curvature.tensor_sha(c),
        "parameters": p.tolist(), "parameters_sha256": curvature.tensor_sha(p),
        "input_before": identity, "runtime": base["runtime"], "policy": plan["policy"],
        "trials": [], "optimizer_steps_applied": 0, "hvp_calls": 0, "hvp_calls_completed": 0,
        "pcg_iterations": None, "true_relative_residual": None, "score_computed": False,
        "response_computed": False, "full_root_claim": False, "physical_validated": False,
        "accepted_point_curvature": "not_computed", "global_spd_claim": False}
    search._write(output, record)
    final_c = c
    try:
        search._check_deadline(deadline)
        branch, margins = curvature.tail._full_current_branch(problem, c, p)
        search._check_deadline(deadline)
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        fresh = curvature.tail._fresh_merit(problem, c, p, gradient_fn)
        if fresh is None:
            raise search.StepRefusal("fresh accepted-point J/g/Phi is nonfinite")
        j, g, phi = fresh
        accepted_trial = _accepted_trial(base)
        expected = {"objective": accepted_trial["objective"], "phi": accepted_trial["phi"],
                    "gradient_inf": accepted_trial["gradient_inf"],
                    "gradient": accepted_trial["gradient"]}
        checks = curvature._endpoint_metrics(expected, j, g, phi)
        if (branch.get("status") != "passed_strict_branch" or branch.get("euler_stages") != 3600
                or branch.get("choice_stage_count") != 3600 or branch.get("face_sign_stage_count") != 3600
                or not curvature.seed._valid_margins(margins, complete=True)
                or branch.get("signature_sha256") != accepted_trial["branch"].get("signature_sha256")
                or not curvature._endpoint_metrics_pass(checks)):
            raise search.StepRefusal("fresh accepted-point J/g/branch differs from f82c receipt")
        search._check_deadline(deadline)
        preconditioner, preconditioner_audit = block_step.block_inverse_preconditioner(h)
        record.update(base_objective=float(j), base_phi=float(phi), base_gradient=g.tolist(),
            base_branch=branch, base_margins=margins, base_metric_checks=checks,
            preconditioner_audit=preconditioner_audit, preconditioner_scope="cached f82c Hessian block inverse only",
            phase="linear_solve")
        search._write(output, record)

        def operator(vector: Tensor) -> Tensor:
            product = torch.func.jvp(lambda x: gradient_fn(x, p), (c,), (vector,))[1]
            if bool(torch.isfinite(product).all()):
                dot = float(vector @ product)
                record["last_live_product"] = {"direction": vector.tolist(), "product": product.tolist(),
                    "direction_dot_product": dot if math.isfinite(dot) else None,
                    "scope": "one live operator action; Krylov/residual vector, not global SPD evidence"}
            else:
                record["last_live_product"] = {"finite": False}
            return product

        live_hvp, hvp_counts = counted_hvp(operator, deadline)
        try:
            step, solve = block_step.solve_newton_direction(live_hvp, g, preconditioner, deadline)
        finally:
            record["hvp_calls"] = hvp_counts["started"]
            record["hvp_calls_completed"] = hvp_counts["completed"]
        residual = torch.tensor(solve["true_residual"], dtype=torch.float64)
        hs = residual - g
        record.update(base_objective=float(j), base_phi=float(phi), base_gradient=g.tolist(),
            base_branch=branch, base_margins=margins, base_metric_checks=checks,
            preconditioner_audit=preconditioner_audit, preconditioner_scope="cached f82c Hessian block inverse only",
            direction=step.tolist(), solve=solve, pcg_iterations=solve["iterations"],
            true_relative_residual=solve["true_relative_residual"], phase="line_search")
        search._write(output, record)

        def save_trial(row: dict[str, Any]) -> None:
            record["trials"].append(row)
            search._write(output, record)

        accepted, _, status = search.bounded_original_j_search(
            c, p, j, g, step, problem.objective, gradient_fn,
            lambda x, pp: curvature.tail._full_current_branch(problem, x, pp),
            branch["signature_sha256"], deadline=deadline, on_trial=save_trial)
        search._check_deadline(deadline)
        record["numerical_status"] = status
        if accepted is not None:
            search._check_deadline(deadline)
            signature, scope = problem.branch_check(accepted, p)
            search._check_deadline(deadline)
            spec = problem.frozen.fv_transport
            if spec is None:
                raise ValueError("FV transport disappeared")
            search._check_deadline(deadline)
            partition = curvature._branch_partition(signature, tuple(problem.layout["observation_times_seconds"]),
                                                    spec.substeps_per_interval)
            row = record["trials"][-1]
            if partition["full_signature_sha256"] != row["branch"]["signature_sha256"]:
                raise search.StepRefusal("accepted candidate branch changed during signature capture")
            delta = accepted - c
            search._check_deadline(deadline)
            state0 = diagnostics.physical_state(problem, c, p)
            search._check_deadline(deadline)
            state1 = diagnostics.physical_state(problem, accepted, p)
            search._check_deadline(deadline)
            changes = {k: (state1[k] - state0[k]).tolist() for k in state0}
            search._check_deadline(deadline)
            record.update(optimizer_steps_applied=1, accepted_control=accepted.tolist(),
                accepted_control_sha256=curvature.tensor_sha(accepted), accepted_branch_partition=partition,
                branch_scope=scope, model_diagnostics=diagnostics.model_diagnostics(
                    float(j), g, None, delta, row["objective"],
                    torch.tensor(row["gradient"], dtype=torch.float64),
                    hessian_delta=row["alpha"] * hs),
                live_directional_model={"H_s": hs.tolist(),
                    "scope": "live Hessian action on accepted Newton direction only; no full matrix"},
                physical_changes={"values": changes,
                    "base": {k: x.tolist() for k, x in state0.items()},
                    "candidate": {k: x.tolist() for k, x in state1.items()},
                    **diagnostics.physical_unit_labels()},
                numerical_status="one_live_HVP_original_J_step_accepted")
            final_c = accepted
    except block_step.StepRefusal as error:
        message = str(error)
        expired = message == "300-second internal budget exhausted during PCG"
        record.update(numerical_status="budget_refusal" if expired else "step_refusal",
                      refusal="240-second internal budget exhausted during PCG" if expired else message,
                      pcg_iterations=record.get("pcg_iterations", "iterations_not_recorded"))
    except search.StepRefusal as error:
        record.update(numerical_status="budget_refusal" if isinstance(error, search.BudgetRefusal) else "step_refusal",
                      refusal=str(error), pcg_iterations=record.get("pcg_iterations", "iterations_not_recorded"))
        if record["trials"] and record["trials"][-1].get("status") == "accepted" and not record["optimizer_steps_applied"]:
            record["trials"][-1].update(status="candidate_not_committed", original_J_candidate_passed=True)
    after = {name: curvature.sha(ROOT / name) for name in names}
    final_identity = curvature.seed._input_identity(problem, original, final_c, p, truth)
    intact = (before == after and curvature.blocks.runtime_identity() == base["runtime"]
              and curvature.audit._fixed_input_matches(final_identity, identity, curvature.tensor_sha(final_c)))
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
    command = [str(ROOT / ".venv/bin/python"), "-m", "examples.weather_scenarios.fv_point_3h_hvp_newton_step",
               "--child", "--plan", str(args.plan.resolve()), "--plan-sha256", args.plan_sha256,
               "--directory", str(directory)]
    resource = run_guarded_diagnostic(command, wall_seconds=WALL_SECONDS, rss_bytes=RSS_BYTES,
        report_path=directory / "step.resource.json", log_path=directory / "step.log")
    parent = {"execution_status": search.execution_status(resource), "resource": resource,
              "numerical_status": "not_reached", "child_read_error": None}
    search._write(directory / "step.run.json", parent)
    try:
        child = json.loads((directory / "step.json").read_text())
        parent.update(numerical_status=child["numerical_status"], child_sha256=curvature.sha(directory / "step.json"))
        if parent["execution_status"] == "completed" and not (child.get("phase") == "finished" and child.get("fixed_input_unchanged") is True):
            parent["execution_status"] = "failed"
    except (OSError, ValueError, KeyError) as error:
        parent.update(execution_status="failed" if parent["execution_status"] == "completed" else parent["execution_status"],
                      child_read_error=str(error))
    search._write(directory / "step.run.json", parent)
    print(json.dumps({k: parent[k] for k in ("execution_status", "numerical_status")}))


if __name__ == "__main__":
    main()
