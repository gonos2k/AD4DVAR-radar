"""One bounded field-only correction at the pinned accepted S4 endpoint."""
from __future__ import annotations

import argparse, hashlib, json, math, platform, re, time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from advar import matrix_free
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit_probe
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks
from examples.weather_scenarios import fv_partial_signed_face_slices as slices

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
AUDIT = EVIDENCE / "point_3h_accepted_endpoint_audit_attempt1/audit.json"
AUDIT_SHA = "b20c5f85cc8921d8cbff69eddac6167c453739906b7a05f5ac1737df405147e6"
PLAN = EVIDENCE / "S4_POINT_TERMINAL_FIELD_REFINEMENT_PLAN_20261003.md"
PLAN_SHA = "bcbf5a548f3e3d79dd36eac2ee05158427c327f8fb11edea802d0fdf74659ac9"
SELF = "examples/weather_scenarios/fv_point_3h_field_refinement.py"
TEST = "tests/test_fv_point_3h_field_refinement.py"
SELF_CANONICAL_SHA = "1ca3fd3f8abeeea5673a2fea5ef7658a04f5c6f49e188a250a65e075e28a0254"
TEST_SHA = "c8d075866e08c48fc08bba4b7455f11bd7ff8d1bf36ba7ba900b108ec6fd2821"
_SELF_RE = re.compile(rb'(?m)^SELF_CANONICAL_SHA = "[0-9a-f]{64}"$')
FIELD_COUNT, CONTROL_COUNT, PCG_MAX = 20, 26, 104
FIELD_TOL, INTERNAL_SECONDS = 1e-10, 540.0
RUNTIME = {"device": "CPU FP64", "python": "3.12.13", "torch": "2.13.0"}


class FieldBudgetRefusal(RuntimeError):
    """Declared internal deadline or strict-point qualification refusal."""


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor_sha(value: Tensor) -> str: return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"); tmp.replace(path)


def require_pins() -> None:
    source,count=_SELF_RE.subn(b'SELF_CANONICAL_SHA = "<canonical-self-pin>"',(ROOT/SELF).read_bytes())
    if count!=1 or hashlib.sha256(source).hexdigest()!=SELF_CANONICAL_SHA or sha(ROOT/TEST)!=TEST_SHA:
        raise ValueError("field-refinement producer/test source pin changed")
    if sha(PLAN)!=PLAN_SHA or sha(AUDIT)!=AUDIT_SHA: raise ValueError("field-refinement plan/audit pin changed")


def require_resolved_hff(accepted: dict[str, Any]) -> None:
    schur=accepted.get("block_schur",{}); eig=schur.get("hff_eigenvalues",[]); residuals=schur.get("field_solve_residuals",{})
    if (accepted.get("hessian_status")!="schur_evaluated" or schur.get("status")!="schur_evaluated"
            or not eig or not all(math.isfinite(v) for v in eig)
            or min(eig)<=schur.get("hff_positive_floor",math.inf)
            or not math.isfinite(schur.get("hff_condition",math.inf))
            or schur["hff_condition"]>schur.get("hff_condition_limit",0)
            or any(not math.isfinite(residuals.get(k,math.inf)) or residuals[k]>1e-10
                   for k in ("hff_inverse_gf","hff_inverse_hfd"))):
        raise FieldBudgetRefusal("accepted audit does not qualify resolved SPD Hff for field-only PCG")


def field_objective(problem: Any, fixed_dynamics: Tensor):
    """Restrict the untouched 26-control objective to its first 20 coordinates."""
    if (fixed_dynamics.shape != (6,) or fixed_dynamics.dtype != torch.float64
            or fixed_dynamics.device.type != "cpu" or not bool(torch.isfinite(fixed_dynamics).all())):
        raise ValueError("fixed dynamics must be finite CPU FP64 length six")
    fixed = fixed_dynamics.clone()
    def objective(field: Tensor, parameters: Tensor) -> Tensor:
        if field.shape != (FIELD_COUNT,) or field.dtype != torch.float64 or field.device.type != "cpu":
            raise ValueError("field control must be CPU FP64 length 20")
        return problem.objective(torch.cat((field, fixed)), parameters)
    return objective


def audit_field_solves(objective, parameters: Tensor, solves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gradient_fn = torch.func.grad(objective, argnums=0)
    audits = []
    for row in solves:
        if row.get("error") is not None:
            audits.append({"status": "solver_refusal", "error": row["error"]}); continue
        if row.get("rtol") != 1e-10 or row.get("max_iterations") != PCG_MAX:
            raise ValueError("saved field PCG budget differs from the pinned contract")
        point, rhs, step = (torch.tensor(row[k], dtype=torch.float64)
                            for k in ("input_tangent", "rhs", "solution"))
        if any(v.shape != (FIELD_COUNT,) for v in (point,rhs,step)):
            raise ValueError("saved field PCG vector is malformed")
        if any(not bool(torch.isfinite(v).all()) for v in (point,rhs,step)):
            audits.append({"status":"nonfinite_solve_vector"}); continue
        gradient = gradient_fn(point, parameters)
        rhs_budget=128*torch.finfo(torch.float64).eps*torch.maximum(rhs.abs(),gradient.abs()).clamp_min(torch.finfo(torch.float64).tiny)
        if bool(((rhs + gradient).abs() > rhs_budget).any()): raise ValueError("field PCG RHS differs from fresh -g_f")
        hvp = torch.func.jvp(lambda f: gradient_fn(f, parameters), (point,), (step,))[1]
        residual = hvp - rhs
        rhs_norm = torch.linalg.vector_norm(rhs)
        relative = float(torch.linalg.vector_norm(residual) / rhs_norm) if float(rhs_norm) > 0 else math.inf
        passed=math.isfinite(relative) and relative<=1e-10
        audits.append({"status":"passed" if passed else "residual_refused",
            "true_residual":residual.tolist() if bool(torch.isfinite(residual).all()) else None,
            "true_relative_residual":relative if math.isfinite(relative) else None})
    return audits


def run(output: Path) -> dict[str, Any]:
    if output.exists(): raise ValueError("field-refinement output must be fresh")
    require_pins()
    accepted=json.loads(AUDIT.read_text())
    runtime={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if (accepted.get("phase")!="finished" or accepted.get("numerical_status")!="diagnostic_completed"
            or accepted.get("source_unchanged") is not True or accepted.get("input_unchanged") is not True
            or runtime!=RUNTIME or runtime!=accepted.get("runtime") or runtime!=accepted.get("runtime_after")):
        raise ValueError("accepted endpoint audit/runtime is not eligible")
    paths=sorted(set(accepted["source_before"])|{SELF,TEST,str(PLAN.relative_to(ROOT)),str(AUDIT.relative_to(ROOT))})
    before={name:sha(ROOT/name) for name in paths}
    if accepted["source_before"]!=accepted["source_after"] or any(before.get(k)!=v for k,v in accepted["source_before"].items()):
        raise ValueError("accepted endpoint source map changed")
    problem, original, _, parameters, truth, base_identity=seed._prepare_fixed_seed()
    control=torch.tensor(accepted["accepted_control"],dtype=torch.float64)
    identity=seed._input_identity(problem,original,control,parameters,truth)
    if (control.shape!=(CONTROL_COUNT,) or tensor_sha(control)!=accepted["accepted_control_sha256"]
            or tensor_sha(parameters)!=accepted["parameters_sha256"] or identity!=accepted["input_after"]
            or base_identity.get("archived_input")!=identity.get("archived_input")):
        raise ValueError("accepted control/parameters/fixed input identity changed")
    fixed_parameters=parameters.clone(); original_d=control[FIELD_COUNT:].clone(); fixed_d=original_d.clone()
    field_start=control[:FIELD_COUNT].clone(); objective=field_objective(problem,fixed_d)
    gradient_full=torch.func.grad(problem.objective,argnums=0)
    started=time.monotonic(); deadline=started+INTERNAL_SECONDS
    report: dict[str,Any]={"phase":"input_ready","numerical_status":"not_reached","scope":"conditional field-only local correction; dynamics fixed at accepted values",
        "audit_sha256":AUDIT_SHA,"plan_sha256":PLAN_SHA,"source_before":before,"input_before":identity,"runtime":runtime,
        "accepted_control":control.tolist(),"accepted_control_sha256":tensor_sha(control),"parameters":parameters.tolist(),
        "parameters_sha256":tensor_sha(parameters),"fixed_dynamics":fixed_d.tolist(),"field_start":field_start.tolist(),
        "field_issuance":False,"full_root_claim":False,"full_hessian_or_minimum_claim":False,"response_computed":False,
        "score_computed":False,"trials":[],"linear_solves":[]}
    save(output,report)
    state={"tangent":field_start.clone(),"signature":accepted["branch"]["signature_sha256"]}
    branch_events=[]; solves=report["linear_solves"]
    def budget() -> None:
        if time.monotonic()>=deadline: raise FieldBudgetRefusal("540-second internal field-refinement budget exhausted")
    def branch_check(field: Tensor,p: Tensor):
        budget()
        if not torch.equal(p,fixed_parameters): raise RuntimeError("field refinement changed fixed parameters")
        full=torch.cat((field,fixed_d))
        branch,margins=tail._full_current_branch(problem,full,p)
        budget()
        if branch.get("status")!="passed_strict_branch" or not seed._valid_margins(margins,complete=True):
            raise ValueError("field candidate failed strict 3600-stage branch/margin qualification")
        branch_events.append({"field":field.tolist(),"full_control_sha256":tensor_sha(full),
            "signature_sha256":branch["signature_sha256"],
            "signature_changed_from_accepted":branch["signature_sha256"]!=accepted["branch"]["signature_sha256"],
            "margins":margins})
        state.update(tangent=field.clone(),signature=branch["signature_sha256"])
        return branch,"fixed original point problem; strict 3600-stage pointwise branch, signature may change"
    def checked_objective(field: Tensor,p: Tensor):
        budget(); value=objective(field,p); budget(); return value
    def observer(row: dict[str, Any]) -> None:
        trial=dict(row)
        if isinstance(row.get("candidate_control"),list):
            candidate=torch.tensor(row["candidate_control"],dtype=torch.float64)
            trial["candidate_full_control"] = torch.cat((candidate,fixed_d)).tolist()
        trial["branch_event_index"] = len(branch_events)-1 if branch_events else None
        report["trials"].append(trial)
        save(output,report)
    try:
        require_resolved_hff(accepted)
        budget()
        base_branch,base_margins=tail._full_current_branch(problem,control,parameters)
        budget()
        if (base_branch.get("status")!="passed_strict_branch" or base_branch.get("signature_sha256")!=accepted["branch"]["signature_sha256"]
                or not seed._valid_margins(base_margins,complete=True)):
            raise FieldBudgetRefusal("accepted endpoint failed fresh strict branch/margin replay")
        state["signature"]=base_branch["signature_sha256"]
        fresh=tail._fresh_merit(problem,control,parameters,gradient_full)
        budget()
        if fresh is None: raise FieldBudgetRefusal("accepted endpoint original J/Phi/full gradient is nonfinite")
        base_j,full_gradient,base_phi=fresh
        if not (audit_probe._metric(accepted["objective"],float(base_j))["passed"]
                and audit_probe._metric(accepted["phi"],float(base_phi))["passed"]):
            raise FieldBudgetRefusal("fresh original J/Phi differs from accepted endpoint audit")
        saved_full_gradient=torch.tensor(accepted["gradient"],dtype=torch.float64)
        if saved_full_gradient.shape!=full_gradient.shape: raise FieldBudgetRefusal("accepted audit full-gradient shape changed")
        gradient_checks=[audit_probe._metric(float(saved),float(fresh_value))
                         for saved,fresh_value in zip(saved_full_gradient,full_gradient,strict=True)]
        if not all(row["passed"] for row in gradient_checks):
            raise FieldBudgetRefusal("fresh accepted endpoint full gradient differs from audit")
        report.update(phase="field_refining",base_objective=float(base_j),base_full_phi=float(base_phi),
            base_full_gradient=full_gradient.tolist(),base_gradient_blocks=blocks.gradient_blocks(full_gradient),
            base_branch=base_branch,base_margins=base_margins)
        save(output,report)
        if not bool(torch.isfinite(full_gradient[:FIELD_COUNT]).all()): raise FieldBudgetRefusal("initial field gradient is nonfinite")
        with matrix_free.observe_pcg_calls(slices._true_residual_monitor(solves,0.0,state)):
            refined=refine_stationary(checked_objective,field_start,fixed_parameters,branch_check=branch_check,
                max_iterations=4,max_backtracks=16,pcg_max_iterations=PCG_MAX,trial_observer=observer)
        budget()
        field=refined.control.clone()
        if not torch.equal(parameters,fixed_parameters) or not torch.equal(torch.cat((field,fixed_d))[FIELD_COUNT:],original_d):
            raise RuntimeError("field refinement altered a fixed input")
        budget(); solve_audits=audit_field_solves(objective,fixed_parameters,solves); budget()
        if any(row["status"]!="passed" for row in solve_audits):
            raise RefinementNumericalRefusal("independent saved field-solve residual audit failed")
        fresh_branch,fresh_margins=tail._full_current_branch(problem,torch.cat((field,fixed_d)),parameters)
        budget()
        if fresh_branch.get("status")!="passed_strict_branch" or not seed._valid_margins(fresh_margins,complete=True):
            raise FieldBudgetRefusal("final field candidate failed strict branch/margin qualification")
        field_gradient=torch.func.grad(objective,argnums=0)(field,fixed_parameters)
        if not bool(torch.isfinite(field_gradient).all()) or float(field_gradient.abs().max())>=FIELD_TOL:
            raise RefinementNumericalRefusal("fresh final field-gradient gate exceeds 1e-10")
        full=torch.cat((field,fixed_d)); full_gradient=gradient_full(full,parameters)
        final_j=problem.objective(full,parameters); budget(); full_phi=torch.dot(full_gradient,full_gradient)/2
        field_phi=torch.dot(field_gradient,field_gradient)/2
        if not bool(torch.isfinite(final_j)&torch.isfinite(full_phi)&torch.isfinite(field_phi)&torch.isfinite(full_gradient).all()):
            raise RefinementNumericalRefusal("final field/full objective or gradient is nonfinite")
        report.update(phase="finished",numerical_status="field_stationary_candidate",field_issuance=True,
            corrected_field=field.tolist(),corrected_control=full.tolist(),corrected_control_sha256=tensor_sha(full),
            fixed_dynamics_unchanged=torch.equal(full[FIELD_COUNT:],fixed_d),objective=float(final_j),
            field_phi=float(field_phi),full_phi=float(full_phi),field_gradient=field_gradient.tolist(),
            field_gradient_max=float(field_gradient.abs().max()),full_gradient=full_gradient.tolist(),
            full_gradient_blocks=blocks.gradient_blocks(full_gradient),final_branch=fresh_branch,
            final_margins=fresh_margins,branch_events=branch_events,solve_audits=solve_audits,
            newton_iterations=refined.iterations,refiner_hvp_count=refined.hvp_count)
    except (FieldBudgetRefusal,RefinementNumericalRefusal) as error:
        report.update(phase="finished",numerical_status="field_refused",field_issuance=False,
                      refusal=f"{type(error).__name__}: {error}",branch_events=branch_events,
                      solve_audits=audit_field_solves(objective,fixed_parameters,solves))
    after={name:sha(ROOT/name) for name in paths}; input_after=seed._input_identity(problem,original,control,parameters,truth)
    runtime_after={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if (after!=before or input_after!=identity or runtime_after!=runtime or sha(AUDIT)!=AUDIT_SHA or sha(PLAN)!=PLAN_SHA):
        raise ValueError("field-refinement source/input/runtime changed during work")
    report.update(source_after=after,source_unchanged=True,input_after=input_after,input_unchanged=True,runtime_after=runtime_after,
                  elapsed_seconds=time.monotonic()-started)
    save(output,report); return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
