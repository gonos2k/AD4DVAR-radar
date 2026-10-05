"""One unshifted coupled Newton search from the qualified f82c curvature."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor
from advar import variational as v
from advar.physics import dbz_to_echo
from advar.transport import bounded_fv_coefficients, face_volume_fluxes
from examples.weather_scenarios import fv_point_3h_accepted_curvature as curvature
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as search
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

ROOT = curvature.ROOT
EVIDENCE = curvature.EVIDENCE
BASE = EVIDENCE / "f82c_fresh_curvature_20261005/attempt_1/audit.json"
CHECKPOINT = EVIDENCE / "f82c_fresh_curvature_20261005/checkpoint/hessian_checkpoint.json"
SELF = "examples/weather_scenarios/fv_point_3h_current_newton_step.py"
TEST = "tests/test_fv_point_3h_current_newton_step.py"
PLAN = EVIDENCE / "F82C_NEWTON_PLAN_20261005.json"


def load_base(plan_path: Path, plan_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if curvature.sha(plan_path) != plan_sha:
        raise ValueError("Newton plan hash changed")
    plan = json.loads(plan_path.read_text())
    base = json.loads(BASE.read_text())
    required = set(base["source_before"]) | {SELF, TEST,
        "examples/weather_scenarios/fv_point_3h_bounded_coupled_step.py"}
    if not required <= set(plan["source_files"]):
        raise ValueError("plan must bind the current caller and all original curvature dependencies")
    for name, digest in {**plan["source_files"], **plan["archive_files"]}.items():
        if curvature.sha(ROOT / name) != digest:
            raise ValueError(f"Newton input/source pin changed: {name}")
    if not {BASE.relative_to(ROOT).as_posix(), CHECKPOINT.relative_to(ROOT).as_posix(),
            BASE.with_suffix(".run.json").relative_to(ROOT).as_posix(),
            BASE.with_suffix(".resource.json").relative_to(ROOT).as_posix()} <= set(plan["archive_files"]):
        raise ValueError("plan must bind curvature raw/checkpoint/parent/resource")
    checkpoint = json.loads(CHECKPOINT.read_text())
    parent = json.loads(BASE.with_suffix(".run.json").read_text())
    resource = json.loads(BASE.with_suffix(".resource.json").read_text())
    h = base["curvature"]
    if (base["phase"] != "finished" or base["numerical_status"] != "curvature_completed"
            or base["source_before"] != base["source_after"] or not base["fixed_input_unchanged"]
            or base["runtime"] != base["runtime_after"] or parent["execution_status"] != "completed"
            or parent["child_sha256"] != curvature.sha(BASE) or search.execution_status(resource) != "completed"
            or checkpoint["status"] != "completed" or h["hvp_columns"] != 26 or h["hvp_calls_total"] != 27
            or not h["minimum_eigenpair_audit"]["passed"]
            or h["hessian"] != checkpoint["final_result"]["hessian"]):
        raise ValueError("base is not a qualified completed same-point curvature result")
    if base["control_sha256"] != curvature.CONTROL_SHA or base["parameters_sha256"] != curvature.PARAMETERS_SHA:
        raise ValueError("Newton base must be the accepted f82c point")
    policy = plan["policy"]
    if policy != {"mu": 0., "radius": .05, "candidate_count": 16, "armijo_c1": 1e-4,
                  "internal_seconds": 240., "outer_seconds": 300., "rss_bytes": 1024**3,
                  "guarded_launches": 1, "phi_is_acceptance_gate": False}:
        raise ValueError("Newton policy differs from the reviewed unshifted one-step policy")
    return plan, base, checkpoint


def model_diagnostics(j0: float, g0: Tensor, h0: Tensor, delta: Tensor,
                      j1: float, g1: Tensor) -> dict[str, Any]:
    linear_g = g0 + h0 @ delta
    predicted = -float(g0 @ delta + .5 * delta @ (h0 @ delta))
    actual = j0 - j1
    return {"predicted_J": j0 - predicted, "actual_J_reduction": actual,
        "predicted_J_reduction": predicted, "actual_over_predicted_J_reduction": actual / predicted if predicted > 0 else None,
        "linear_gradient": linear_g.tolist(), "linear_phi": float(linear_g @ linear_g / 2),
        "actual_phi": float(g1 @ g1 / 2), "gradient_linearization_error_l2": float((g1 - linear_g).norm()),
        "gradient_linearization_relative_error": float((g1 - linear_g).norm() / g1.norm()) if bool(g1.norm() > 0) else None,
        "step_l2": float(delta.norm()), "field_step_l2": float(delta[:20].norm()),
        "dynamics_step_l2": float(delta[20:].norm()), "directional_secant": float(delta @ (g1 - g0)),
        "old_point_directional_curvature": float(delta @ (h0 @ delta))}


def physical_state(problem: Any, c: Tensor, p: Tensor) -> dict[str, Tensor]:
    contract = problem.contract(p)
    spec = contract.fv_transport
    if spec is None:
        raise ValueError("FV transport is required for physical-coordinate diagnostics")
    field = torch.zeros_like(contract.initial_background_dbz).flatten().scatter(
        0, contract.active_field_index, c[:contract.active_field_index.numel()]).reshape_as(contract.initial_background_dbz)
    dbz = v._initial_analysis_dbz(field, contract)
    count = spec.coefficient_limits.numel()
    coeff = bounded_fv_coefficients(c[20:20+count], psi_basis=spec.psi_basis,
        coefficient_limits=spec.coefficient_limits, dt_seconds=contract.nowcast_config.interval_minutes*60/spec.substeps_per_interval,
        spacing_yx=spec.spacing_yx, reconstruction=spec.reconstruction, max_courant=spec.max_courant)
    qx, qy = face_volume_fluxes(torch.einsum("k,kij->ij", coeff, spec.psi_basis))
    return {"initial_dbz": dbz, "initial_echo_proxy": dbz_to_echo(dbz, min_dbz=contract.nowcast_config.min_dbz),
        "flow_coefficients": coeff, "qx": qx, "qy": qy,
        "normal_x_speed": qx/spec.spacing_yx[0], "normal_y_speed": qy/spec.spacing_yx[1],
        "interval_log_echo_growth": contract.nowcast_config.max_log_growth_per_step*c[-1].tanh()}


def run(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 240.
    if output.exists() or output.is_symlink():
        raise ValueError("Newton output must be fresh")
    plan, base, checkpoint = load_base(plan_path, plan_sha)
    names = set(plan["source_files"]) | set(plan["archive_files"]) | {plan_path.relative_to(ROOT).as_posix()}
    before = {n: curvature.sha(ROOT/n) for n in names}
    if (before[plan_path.relative_to(ROOT).as_posix()] != plan_sha or
            any(before[n] != digest for n, digest in {**plan["source_files"], **plan["archive_files"]}.items())):
        raise ValueError("Newton plan/input/source changed after preflight")
    c, p = (torch.tensor(base[k], dtype=torch.float64) for k in ("control", "parameters"))
    h = torch.tensor(base["curvature"]["hessian"], dtype=torch.float64)
    if (curvature.tensor_sha(h) != base["curvature"]["hessian_sha256"] or curvature.tensor_sha(c) != base["control_sha256"]
            or curvature.tensor_sha(p) != base["parameters_sha256"]):
        raise ValueError("base tensor identities changed")
    problem, original, _, fixed_p, truth, _ = curvature.seed._prepare_fixed_seed()
    identity = curvature.seed._input_identity(problem, original, c, p, truth)
    if identity != base["input_after"] or not torch.equal(p, fixed_p) or curvature.blocks.runtime_identity() != base["runtime"]:
        raise ValueError("original problem/parameters/runtime changed")
    record: dict[str, Any] = {"phase": "running", "numerical_status": "not_reached", "source_before": before,
        "plan_sha256": plan_sha, "base_curvature_sha256": curvature.sha(BASE), "base_control": c.tolist(),
        "base_control_sha256": curvature.tensor_sha(c), "parameters": p.tolist(), "parameters_sha256": curvature.tensor_sha(p),
        "input_before": identity, "runtime": base["runtime"], "policy": plan["policy"], "trials": [],
        "optimizer_steps_applied": 0, "new_hvp": 0, "score_computed": False, "response_computed": False,
        "full_root_claim": False, "physical_validated": False, "accepted_point_curvature": "not_computed"}
    search._write(output, record)
    final_c = c
    try:
        search._check_deadline(deadline)
        branch, margins = curvature.tail._full_current_branch(problem, c, p)
        search._check_deadline(deadline)
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        fresh = curvature.tail._fresh_merit(problem, c, p, gradient_fn)
        if fresh is None:
            raise search.StepRefusal("fresh base J/g/Phi is nonfinite")
        j, g, phi = fresh
        search._check_deadline(deadline)
        expected = {"objective": base["objective"], "phi": base["phi"], "gradient_inf": base["gradient_inf"], "gradient": base["gradient"]}
        if (branch != base["endpoint_branch"] or margins != base["endpoint_margins"]
                or not curvature._endpoint_metrics_pass(curvature._endpoint_metrics(expected, j, g, phi))):
            raise search.StepRefusal("fresh base J/g/strict branch differs from f82c curvature receipt")
        s, direction = search.exact_shifted_direction(h, g, mu=0.)
        search._check_deadline(deadline)
        record.update(base_objective=float(j), base_phi=float(phi), base_gradient=g.tolist(),
                      base_gradient_blocks=curvature.blocks.gradient_blocks(g), direction=s.tolist(), direction_audit=direction,
                      base_branch=branch, base_margins=margins, base_branch_partition=base["branch_partition"])
        weak = torch.tensor(checkpoint["audit_hvp"]["direction"], dtype=torch.float64)
        record["weak_mode"] = {"eigenvalue": base["curvature"]["eigenvalues"][0],
            "vector": weak.tolist(), "direction_projection": float(weak @ s),
            "direction_squared_norm_fraction": float((weak @ s).square() / (s @ s)),
            "scope": "standardized original controls; not physical uncertainty"}
        search._write(output, record)

        def save_trial(row: dict[str, Any]) -> None:
            record["trials"].append(row)
            search._write(output, record)

        accepted, _, status = search.bounded_original_j_search(c, p, j, g, s, problem.objective,
            gradient_fn, lambda x, pp: curvature.tail._full_current_branch(problem, x, pp),
            branch["signature_sha256"], deadline=deadline, on_trial=save_trial)
        record["numerical_status"] = status
        if accepted is not None:
            search._check_deadline(deadline)
            signature, scope = problem.branch_check(accepted, p)
            spec = problem.frozen.fv_transport
            if spec is None:
                raise ValueError("FV transport disappeared")
            partition = curvature._branch_partition(signature, tuple(problem.layout["observation_times_seconds"]),
                                                    spec.substeps_per_interval)
            search._check_deadline(deadline)
            row = record["trials"][-1]
            if partition["full_signature_sha256"] != row["branch"]["signature_sha256"]:
                raise search.StepRefusal("accepted point branch changed during independent signature capture")
            delta = accepted - c
            search._check_deadline(deadline)
            base_state = physical_state(problem, c, p)
            search._check_deadline(deadline)
            states = base_state, physical_state(problem, accepted, p)
            changes = {k: (states[1][k]-states[0][k]).tolist() for k in states[0]}
            search._check_deadline(deadline)
            record.update(optimizer_steps_applied=1, accepted_control=accepted.tolist(),
                accepted_control_sha256=curvature.tensor_sha(accepted), accepted_branch_partition=partition,
                branch_scope=scope, model_diagnostics=model_diagnostics(float(j), g, h, delta,
                    row["objective"], torch.tensor(row["gradient"], dtype=torch.float64)),
                physical_changes={"values": changes, "base": {k: x.tolist() for k, x in states[0].items()},
                    "candidate": {k: x.tolist() for k, x in states[1].items()},
                    "scope": "model echo proxy, not water mass; model flow, not observed air wind",
                    "speed_units": "configured model-coordinate lengths per second", "growth_units": "log echo growth per 600-second interval"})
            record["weak_mode"].update(actual_step_projection=float(weak @ delta),
                actual_step_squared_norm_fraction=float((weak @ delta).square()/(delta @ delta)))
            final_c = accepted
    except search.StepRefusal as error:
        record.update(numerical_status="budget_refusal" if isinstance(error, search.BudgetRefusal) else "step_refusal", refusal=str(error))
        if record["trials"] and record["trials"][-1]["status"] == "accepted" and not record["optimizer_steps_applied"]:
            record["trials"][-1].update(status="candidate_not_committed", original_J_candidate_passed=True)
    after = {n: curvature.sha(ROOT/n) for n in names}
    final_identity = curvature.seed._input_identity(problem, original, final_c, p, truth)
    intact = before == after and curvature.blocks.runtime_identity() == base["runtime"] and curvature.audit._fixed_input_matches(final_identity, identity, curvature.tensor_sha(final_c))
    record.update(phase="finished" if intact else "integrity_refused", source_after=after, source_unchanged=before == after,
        fixed_input_unchanged=intact, input_after=final_identity, runtime_after=curvature.blocks.runtime_identity())
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
        run(args.plan.resolve(), args.plan_sha256, directory/"step.json")
        return
    load_base(args.plan.resolve(), args.plan_sha256)
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(ROOT/".venv/bin/python"), "-m", "examples.weather_scenarios.fv_point_3h_current_newton_step",
               "--child", "--plan", str(args.plan.resolve()), "--plan-sha256", args.plan_sha256, "--directory", str(directory)]
    resource = run_guarded_diagnostic(command, wall_seconds=300., rss_bytes=1024**3,
                                     report_path=directory/"step.resource.json", log_path=directory/"step.log")
    parent = {"execution_status": search.execution_status(resource), "resource": resource,
              "numerical_status": "not_reached", "child_read_error": None}
    search._write(directory/"step.run.json", parent)
    try:
        child = json.loads((directory/"step.json").read_text())
        if not isinstance(child, dict):
            raise ValueError("child result must be a JSON object")
        parent.update(numerical_status=child["numerical_status"], child_sha256=curvature.sha(directory/"step.json"))
        if parent["execution_status"] == "completed" and not (child.get("phase") == "finished" and child.get("fixed_input_unchanged") is True):
            parent["execution_status"] = "failed"
    except (OSError, ValueError, KeyError) as error:
        parent.update(execution_status="failed" if parent["execution_status"] == "completed" else parent["execution_status"], child_read_error=str(error))
    search._write(directory/"step.run.json", parent)
    print(json.dumps({k: parent[k] for k in ("execution_status", "numerical_status")}))


if __name__ == "__main__":
    main()
