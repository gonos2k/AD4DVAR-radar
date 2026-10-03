"""Continue the pinned S4 field-only refinement from its last accepted trial."""
from __future__ import annotations

import argparse, hashlib, json, math, platform, re, time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from advar import matrix_free
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit_probe
from examples.weather_scenarios import fv_point_3h_field_refinement as field_core
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks
from examples.weather_scenarios import fv_partial_signed_face_slices as slices

ROOT=Path(__file__).resolve().parents[2]; EVIDENCE=ROOT/"graphify-out/fv-root-cause-20260919"
RAW=EVIDENCE/"point_3h_field_continuation_attempt1/field_continuation.json"; RAW_SHA="a5759c3eca7ca537e0038b3dbe95f3ff67abdabcc43f5f5a2b3aa3cc5982fb1b"
BASE_AUDIT=EVIDENCE/"point_3h_accepted_endpoint_audit_attempt1/audit.json"; BASE_AUDIT_SHA="b20c5f85cc8921d8cbff69eddac6167c453739906b7a05f5ac1737df405147e6"; BASE_PLAN=EVIDENCE/"S4_POINT_TERMINAL_FIELD_CONTINUATION_PLAN_20261003.md"; BASE_PLAN_SHA="25eb56f5123bd22d32096099019571f81176b26485976250260c36a80088e68d"
PLAN=EVIDENCE/"S4_POINT_TERMINAL_FIELD_PRECISION_CONTINUATION_PLAN_20261003.md"; PLAN_SHA="b21b19832138515964f21c434d7fc6e108148ed0085ac903afd860a882d1e3ae"; SELF="examples/weather_scenarios/fv_point_3h_field_precision_continuation.py"; TEST="tests/test_fv_point_3h_field_precision_continuation.py"
SELF_CANONICAL_SHA = "376e65ec8b2f11688f49dabbd4501d659115d3bc019e1789857e2360dfa91f59"
TEST_SHA = "1a1a38854e29d68dd010046a3b31a4c554947436e4e039478418fc3e09459b59"
_SELF_RE=re.compile(rb'(?m)^SELF_CANONICAL_SHA = "[0-9a-f]{64}"$')
FIELD_COUNT,CONTROL_COUNT,PCG_MAX,MAX_ITERATIONS=20,26,104,6; FIELD_TOL,INTERNAL_SECONDS,WALL_SECONDS=1e-10,1680.0,1800.0; RUNTIME={"device":"CPU FP64","python":"3.12.13","torch":"2.13.0"}


class ContinuationRefusal(RuntimeError):
    """Declared Hff, strict-point, or internal-budget numerical refusal."""


def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor_sha(value:Tensor)->str: return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def save(path:Path,report:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"); tmp.replace(path)
def require_pins()->None:
    source,count=_SELF_RE.subn(b'SELF_CANONICAL_SHA = "<canonical-self-pin>"',(ROOT/SELF).read_bytes())
    if count!=1 or hashlib.sha256(source).hexdigest()!=SELF_CANONICAL_SHA or sha(ROOT/TEST)!=TEST_SHA: raise ValueError("field-continuation source/test pin changed")
    if any(sha(p)!=h for p,h in ((PLAN,PLAN_SHA),(BASE_PLAN,BASE_PLAN_SHA),(RAW,RAW_SHA),(BASE_AUDIT,BASE_AUDIT_SHA))): raise ValueError("field-continuation plan/raw/audit pin changed")
def select_last_accepted_control(raw:dict[str,Any])->Tensor:
    accepted=[row for row in raw.get("trials",[]) if row.get("accepted") is True]
    if not accepted: raise ValueError("continuation attempt1 contains no accepted field trial")
    row=accepted[-1]; field=torch.tensor(row.get("candidate_control"),dtype=torch.float64)
    full=torch.tensor(row.get("candidate_full_control"),dtype=torch.float64)
    if (field.shape!=(FIELD_COUNT,) or full.shape!=(CONTROL_COUNT,) or not bool(torch.isfinite(full).all())
            or not torch.equal(full[:FIELD_COUNT],field) or tensor_sha(full)!="ecfbbbd4919b8b783c93dce4a8dbb27598eac8eb9ca698bcf7ecd510f737b38b"):
        raise ValueError("last accepted field trial lacks its pinned full control")
    if (raw.get("numerical_status")!="continuation_refused" or raw.get("field_issuance") is not False
            or not str(raw.get("refusal","")).startswith("RefinementNumericalRefusal: stationarity refinement iteration budget exhausted")):
        raise ValueError("attempt1 is not the declared iteration-refused no-issuance artifact")
    return full
def field_hessian_audit(objective,field:Tensor,parameters:Tensor,deadline:float)->dict[str,Any]:
    grad=torch.func.grad(objective,argnums=0); eye=torch.eye(FIELD_COUNT,dtype=torch.float64); columns=[]
    def product(v:Tensor)->Tensor:
        if time.monotonic()>=deadline: raise ContinuationRefusal("1680-second budget exhausted during fresh Hff HVPs")
        return torch.func.jvp(lambda x:grad(x,parameters),(field,),(v,))[1]
    for i in range(FIELD_COUNT):
        col=product(eye[i])
        if col.shape!=(FIELD_COUNT,) or not bool(torch.isfinite(col).all()): raise ContinuationRefusal("fresh Hff HVP column is invalid")
        columns.append(col)
    h=torch.stack(columns,dim=1); norm=torch.linalg.matrix_norm(h)
    if not bool(torch.isfinite(norm)) or float(norm)<=0: raise ContinuationRefusal("fresh Hff has no finite scale")
    symmetry=float(torch.linalg.matrix_norm(h-h.T)/norm)
    if not math.isfinite(symmetry) or symmetry>1e-8: raise ContinuationRefusal("fresh Hff symmetry gate failed")
    try: eig,vec=torch.linalg.eigh(.5*(h+h.T))
    except torch.linalg.LinAlgError as error: raise ContinuationRefusal(f"fresh Hff eigensolver failed: {error}") from error
    floor=max(1e-8*float(eig[-1].abs()),128*torch.finfo(torch.float64).eps*float(norm))
    if not bool(torch.isfinite(eig).all()&torch.isfinite(vec).all()) or float(eig[0])<=floor:
        raise ContinuationRefusal("fresh Hff is not resolved SPD")
    cond=float(eig[-1]/eig[0])
    if not math.isfinite(cond) or cond>1e8: raise ContinuationRefusal("fresh Hff condition exceeds 1e8")
    fresh_min=product(vec[:,0]); residual=fresh_min-eig[0]*vec[:,0]
    relative=float(torch.linalg.vector_norm(residual)/norm)
    if not math.isfinite(relative) or relative>1e-8: raise ContinuationRefusal("independent 21st Hff eigenpair residual failed")
    return {"hessian":h.tolist(),"hessian_sha256":tensor_sha(h),"eigenvalues":eig.tolist(),
        "lambda_min":float(eig[0]),"lambda_max":float(eig[-1]),"positive_floor":floor,
        "condition":cond,"symmetry_relative":symmetry,"hvp_columns":FIELD_COUNT,
        "independent_eigenpair_hvp":True,"eigenpair_relative_residual":relative}
def terminal_diagnostics(problem:Any,control:Tensor,parameters:Tensor,deadline:float|None=None)->dict[str,Any]:
    started=time.monotonic()
    if deadline is not None and started>=deadline: return {"terminal_diagnostics_status":"post_audit_budget_refused","terminal_diagnostics_reason":"1800-second wall budget exhausted"}
    gradient=torch.func.grad(problem.objective,argnums=0)(control,parameters); objective=problem.objective(control,parameters); phi=torch.dot(gradient,gradient)/2
    if not bool(torch.isfinite(gradient).all()&torch.isfinite(objective)&torch.isfinite(phi)):
        return {"terminal_diagnostics_status":"numerical_refusal","terminal_diagnostics_reason":"terminal full-control objective/gradient is nonfinite"}
    return {"terminal_diagnostics_status":"completed","terminal_diagnostics_seconds":time.monotonic()-started,
        "terminal_diagnostics_after_wall_deadline":deadline is not None and time.monotonic()>deadline,
        "terminal_control":control.tolist(),"terminal_control_sha256":tensor_sha(control),"terminal_objective":float(objective),
        "terminal_full_gradient":gradient.tolist(),"terminal_gradient_blocks":blocks.gradient_blocks(gradient),"terminal_full_phi":float(phi),
        "terminal_field_gradient":gradient[:FIELD_COUNT].tolist(),"terminal_field_gradient_max":float(gradient[:FIELD_COUNT].abs().max()),
        "terminal_field_phi":float(torch.dot(gradient[:FIELD_COUNT],gradient[:FIELD_COUNT])/2),
        "terminal_dynamics_gradient":gradient[FIELD_COUNT:].tolist(),"terminal_dynamics_gradient_max":float(gradient[FIELD_COUNT:].abs().max())}
def terminal_issuance_gate(record:dict[str,Any])->tuple[bool|None,bool]:
    measured=record.get("terminal_diagnostics_status")=="completed" and isinstance(record.get("terminal_field_gradient_max"),(int,float)) and math.isfinite(record["terminal_field_gradient_max"])
    passed=record["terminal_field_gradient_max"]<FIELD_TOL if measured else None
    return passed,(record.get("terminal_diagnostics_status")=="completed"
        and record.get("terminal_diagnostics_after_wall_deadline") is False and passed is True)
def select_terminal_control(report:dict[str,Any],start:Tensor)->tuple[Tensor,str]:
    accepted=[row for row in report.get("trials",[]) if row.get("accepted") is True]; issued=report.get("field_issuance") is True
    control=torch.tensor(report["corrected_control"],dtype=torch.float64) if issued else torch.tensor(accepted[-1]["candidate_full_control"],dtype=torch.float64) if accepted else start.clone()
    source="corrected_field" if issued else "last_accepted_precision_trial" if accepted else "selected_latest_continuation_start"
    if control.shape!=(CONTROL_COUNT,) or not bool(torch.isfinite(control).all()) or not torch.equal(control[FIELD_COUNT:],start[FIELD_COUNT:]): raise ValueError("terminal control is malformed or changed fixed dynamics")
    return control,source
def audit_saved_solves(objective,parameters:Tensor,solves:list[dict[str,Any]],deadline:float)->list[dict[str,Any]]:
    if time.monotonic()>=deadline: return [{"status":"post_audit_budget_refused","reason":"wall budget exhausted before residual audit"}]
    result=field_core.audit_field_solves(objective,parameters,solves)
    if time.monotonic()>=deadline: result.append({"status":"post_audit_budget_overrun","reason":"residual audit completed after wall budget"})
    return result

def run(output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("field-continuation output must be fresh")
    require_pins(); raw=json.loads(RAW.read_text()); accepted=json.loads(BASE_AUDIT.read_text())
    runtime={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if runtime!=RUNTIME or raw.get("runtime")!=runtime or raw.get("runtime_after")!=runtime:
        raise ValueError("attempt1/runtime differs from pinned environment")
    start=select_last_accepted_control(raw)
    if raw.get("plan_sha256")!=BASE_PLAN_SHA or raw.get("accepted_audit_sha256")!=BASE_AUDIT_SHA:
        raise ValueError("continuation attempt1 plan/audit provenance changed")
    selected_trial=[row for row in raw["trials"] if row.get("accepted") is True][-1]
    paths=sorted(set(raw["source_before"])|{SELF,TEST,str(PLAN.relative_to(ROOT)),str(BASE_PLAN.relative_to(ROOT)),str(RAW.relative_to(ROOT)),str(BASE_AUDIT.relative_to(ROOT))})
    before={name:sha(ROOT/name) for name in paths}
    if raw["source_before"]!=raw["source_after"] or any(before.get(k)!=v for k,v in raw["source_before"].items()):
        raise ValueError("field attempt1 source map changed")
    problem,original,_,parameters,truth,base_identity=seed._prepare_fixed_seed()
    original_control=torch.tensor(accepted["accepted_control"],dtype=torch.float64)
    original_input=seed._input_identity(problem,original,original_control,parameters,truth)
    start_input=seed._input_identity(problem,original,start,parameters,truth)
    if (original_input!=accepted["input_after"] or raw["input_before"]!=accepted["input_after"]
            or tensor_sha(parameters)!=raw["parameters_sha256"] or tensor_sha(parameters)!=accepted["parameters_sha256"]
            or base_identity.get("archived_input")!=original_input.get("archived_input")
            or not torch.equal(start[FIELD_COUNT:],torch.tensor(raw["fixed_dynamics"],dtype=torch.float64))):
        raise ValueError("continuation start changed accepted fixed input, parameters, or dynamics")
    fixed_p=parameters.clone(); fixed_d=start[FIELD_COUNT:].clone(); field_start=start[:FIELD_COUNT].clone()
    objective=field_core.field_objective(problem,fixed_d); full_gradient_fn=torch.func.grad(problem.objective,argnums=0)
    started=time.monotonic(); deadline=started+INTERNAL_SECONDS; wall_deadline=started+WALL_SECONDS
    report:dict[str,Any]={"phase":"input_ready","numerical_status":"not_reached","scope":"continuation of one field-only local correction; no full-root/minimum/response claim",
        "attempt1_sha256":RAW_SHA,"accepted_audit_sha256":BASE_AUDIT_SHA,"plan_sha256":PLAN_SHA,
        "source_before":before,"input_before":original_input,"start_input":start_input,"runtime":runtime,
        "attempt1_refusal":raw["refusal"],"attempt1_elapsed_seconds":raw["elapsed_seconds"],
        "start_control":start.tolist(),"start_control_sha256":tensor_sha(start),"fixed_dynamics":fixed_d.tolist(),
        "parameters":parameters.tolist(),"parameters_sha256":tensor_sha(parameters),"field_issuance":False,
        "full_root_claim":False,"full_hessian_or_minimum_claim":False,"response_computed":False,"score_computed":False,
        "trials":[],"linear_solves":[]}
    save(output,report); state={"tangent":field_start.clone(),"signature":selected_trial["branch"]["signature_sha256"]}
    branch_events=[]; solves=report["linear_solves"]
    def budget()->None:
        if time.monotonic()>=deadline: raise ContinuationRefusal("1680-second continuation budget exhausted")
    def branch_check(field:Tensor,p:Tensor):
        budget()
        if not torch.equal(p,fixed_p): raise RuntimeError("continuation changed fixed parameters")
        full=torch.cat((field,fixed_d)); branch,margins=tail._full_current_branch(problem,full,p); budget()
        if branch.get("status")!="passed_strict_branch" or not seed._valid_margins(margins,complete=True):
            raise ValueError("continuation candidate failed strict 3600-stage branch/margin check")
        branch_events.append({"control_sha256":tensor_sha(full),"signature_sha256":branch["signature_sha256"],
            "signature_changed_from_start":branch["signature_sha256"]!=state["start_signature"],"margins":margins})
        state.update(tangent=field.clone(),signature=branch["signature_sha256"])
        return branch,"fixed p and d; complete strict pointwise branch, signature may change"
    def checked_objective(field:Tensor,p:Tensor):
        budget(); value=objective(field,p); budget(); return value
    def observer(row:dict[str,Any])->None:
        trial=dict(row)
        if isinstance(row.get("candidate_control"),list): trial["candidate_full_control"]=torch.cat((torch.tensor(row["candidate_control"],dtype=torch.float64),fixed_d)).tolist()
        trial["branch_event_index"]=len(branch_events)-1 if branch_events else None
        report["trials"].append(trial); save(output,report)
    solve_audits: list[dict[str,Any]]|None=None
    try:
        budget(); fresh_branch,fresh_margins=tail._full_current_branch(problem,start,parameters); budget()
        saved_trial=selected_trial
        if (fresh_branch.get("status")!="passed_strict_branch"
                or fresh_branch.get("signature_sha256")!=saved_trial["branch"].get("signature_sha256")
                or not seed._valid_margins(fresh_margins,complete=True)):
            raise ContinuationRefusal("selected accepted trial no longer passes its strict branch")
        field_grad_fn=torch.func.grad(objective,argnums=0); field_gradient=field_grad_fn(field_start,parameters)
        j=float(problem.objective(start,parameters)); budget()
        if (not audit_probe._metric(saved_trial["objective"],j)["passed"]
                or not audit_probe._metric(saved_trial["gradient_max"],float(field_gradient.abs().max()))["passed"]
                or not audit_probe._metric(saved_trial["gradient_norm"],float(torch.linalg.vector_norm(field_gradient)))["passed"]):
            raise ContinuationRefusal("fresh selected-start J/field-gradient differs from accepted attempt1 trial")
        restricted_j=float(objective(field_start,parameters))
        if not audit_probe._metric(j,restricted_j)["passed"]:
            raise ContinuationRefusal("field-restricted objective differs from original full-control objective")
        hff=field_hessian_audit(objective,field_start,parameters,deadline)
        report.update(phase="continuing",start_objective=j,start_field_gradient=field_gradient.tolist(),
            start_field_gradient_max=float(field_gradient.abs().max()),start_field_gradient_norm=float(field_gradient.norm()),
            field_restricted_start_objective=restricted_j,start_branch=fresh_branch,start_margins=fresh_margins,fresh_hff=hff)
        save(output,report)
        state["start_signature"]=fresh_branch["signature_sha256"]
        with matrix_free.observe_pcg_calls(slices._true_residual_monitor(solves,0.,state)):
            refined=refine_stationary(checked_objective,field_start,fixed_p,branch_check=branch_check,
                max_iterations=MAX_ITERATIONS,max_backtracks=16,pcg_max_iterations=PCG_MAX,trial_observer=observer)
        budget(); field=refined.control.clone(); full=torch.cat((field,fixed_d))
        if not torch.equal(parameters,fixed_p) or not torch.equal(full[FIELD_COUNT:],start[FIELD_COUNT:]):
            raise RuntimeError("continuation modified fixed parameters or dynamics")
        solve_audits=audit_saved_solves(objective,fixed_p,solves,wall_deadline); budget()
        if any(row["status"]!="passed" for row in solve_audits): raise RefinementNumericalRefusal("continuation solve residual audit failed")
        branch,margins=tail._full_current_branch(problem,full,parameters); budget()
        if branch.get("status")!="passed_strict_branch" or not seed._valid_margins(margins,complete=True):
            raise ContinuationRefusal("final continuation field failed strict branch/margin gate")
        gf=field_grad_fn(field,fixed_p)
        if not bool(torch.isfinite(gf).all()) or float(gf.abs().max())>=1e-10: raise RefinementNumericalRefusal("fresh final field-gradient gate exceeds 1e-10")
        g=full_gradient_fn(full,parameters); objective_final=problem.objective(full,parameters); budget(); phi=torch.dot(g,g)/2
        report.update(phase="finished",numerical_status="field_stationary_candidate",field_issuance=True,
            corrected_field=field.tolist(),corrected_control=full.tolist(),corrected_control_sha256=tensor_sha(full),
            fixed_dynamics_unchanged=torch.equal(full[FIELD_COUNT:],start[FIELD_COUNT:]),objective=float(objective_final),
            field_phi=float(torch.dot(gf,gf)/2),full_phi=float(phi),full_gradient=g.tolist(),
            full_gradient_blocks=blocks.gradient_blocks(g),field_gradient=gf.tolist(),field_gradient_max=float(gf.abs().max()),
            final_branch=branch,final_margins=margins,branch_events=branch_events,solve_audits=solve_audits,
            newton_iterations=refined.iterations,refiner_hvp_count=refined.hvp_count)
    except (ContinuationRefusal,RefinementNumericalRefusal) as error:
        if solve_audits is None: solve_audits=audit_saved_solves(objective,fixed_p,solves,wall_deadline)
        report.update(phase="finished",numerical_status="continuation_refused",field_issuance=False,
            refusal=f"{type(error).__name__}: {error}",branch_events=branch_events,
            solve_audits=solve_audits)
    terminal,terminal_source=select_terminal_control(report,start)
    terminal_record=terminal_diagnostics(problem,terminal,parameters,deadline=wall_deadline)
    terminal_gradient_passed,terminal_eligible=terminal_issuance_gate(terminal_record)
    if not terminal_eligible and report["field_issuance"]:
        refusal=terminal_record.get("terminal_diagnostics_reason") or ("terminal diagnostics wall-deadline overrun" if terminal_record.get("terminal_diagnostics_after_wall_deadline") is True else "terminal field-gradient issuance gate failed")
        report.update(field_issuance=False,numerical_status="terminal_diagnostics_refused",
            refusal=refusal)
        for key in ("corrected_field","corrected_control","corrected_control_sha256"):
            report.pop(key,None)
    report.update(terminal_record,terminal_control_source=terminal_source,
        terminal_fixed_dynamics_unchanged=torch.equal(terminal[FIELD_COUNT:],start[FIELD_COUNT:]),
        terminal_field_gradient_gate_passed=terminal_gradient_passed)
    after={name:sha(ROOT/name) for name in paths}; input_after=seed._input_identity(problem,original,original_control,parameters,truth)
    runtime_after={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if after!=before or input_after!=original_input or runtime_after!=runtime or sha(RAW)!=RAW_SHA or sha(PLAN)!=PLAN_SHA or sha(BASE_PLAN)!=BASE_PLAN_SHA or sha(BASE_AUDIT)!=BASE_AUDIT_SHA:
        raise ValueError("continuation source/input/runtime changed during work")
    report.update(source_after=after,source_unchanged=True,input_after=input_after,input_unchanged=True,
        runtime_after=runtime_after,elapsed_seconds=time.monotonic()-started)
    save(output,report); return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
