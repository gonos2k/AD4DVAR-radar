"""At most four original-J Armijo epochs with fresh modified-H block solves."""
from __future__ import annotations

import argparse,hashlib,json,math,platform,re,time
from pathlib import Path
from typing import Any
import torch
from torch import Tensor
from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_precision_hessian_audit as hess_probe
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_schur_newton_step as block_probe
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks

ROOT=Path(__file__).resolve().parents[2]; EVIDENCE=ROOT/"graphify-out/fv-root-cause-20260919"
R8=EVIDENCE/"block_modified_newton_search_attempt1/block_modified_newton_search.json"; R8_SHA="73b4e51c76ad6fb6f7ffe0721e1a4b509d6e4371b667734c68b8789ed47dffb7"
AUDIT=EVIDENCE/"point_3h_accepted_endpoint_audit_attempt1/audit.json"; AUDIT_SHA="b20c5f85cc8921d8cbff69eddac6167c453739906b7a05f5ac1737df405147e6"
PLAN=EVIDENCE/"S4_COUPLED_ORIGINAL_J_CONTINUATION_PLAN_20261003.md"; PLAN_SHA="5e83f2f8bd6740a584fdd81b5d3c37f2c77d0e9e8a8c663564c5e432e073269c"
SELF="examples/weather_scenarios/fv_point_3h_coupled_original_j_continuation.py"; TEST="tests/test_fv_point_3h_coupled_original_j_continuation.py"
SELF_SHA="292c88dca84bc1d95a4487df8ab9117877bff28e13441b80ac275ba2156c4907"
TEST_SHA="a527349062bcaf4ee06ab167e82f75b1260dc115ce2909ffe6d4ce5dccb7218e"
_SELF_RE=re.compile(rb'(?m)^SELF_SHA="[0-9a-f]{64}"$')
NC,NF,ND,MAX_EPOCHS,MAX_PCG=26,20,6,4,104
RTOL,STATIONARITY_TOL,INTERNAL_SECONDS,WALL_SECONDS=1e-10,1e-10,1200.,1500.
ALPHAS=tuple(2.**-i for i in range(16)); C1=1e-4
RUNTIME={"device":"CPU FP64","python":"3.12.13","torch":"2.13.0"}


class ContinuationRefusal(RuntimeError):
    """Declared curvature, linear-solve, line-search, branch or budget refusal."""


def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor_sha(x:Tensor)->str: return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def write(path:Path,data:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n"); tmp.replace(path)


def require_pins()->None:
    source,n=_SELF_RE.subn(b'SELF_SHA="<canonical-self-pin>"',(ROOT/SELF).read_bytes())
    if n!=1 or hashlib.sha256(source).hexdigest()!=SELF_SHA or sha(ROOT/TEST)!=TEST_SHA: raise ValueError("continuation source/test pin changed")
    if any(sha(p)!=h for p,h in ((R8,R8_SHA),(AUDIT,AUDIT_SHA),(PLAN,PLAN_SHA))): raise ValueError("continuation raw/audit/plan pin changed")


def candidate_metrics(value:Tensor,gradient:Tensor)->dict[str,Any]:
    if value.ndim!=0 or value.dtype!=torch.float64 or value.device.type!="cpu": raise ValueError("candidate J contract changed")
    if not bool(torch.isfinite(value)): return {"status":"nonfinite_objective"}
    if gradient.shape!=(NC,) or gradient.dtype!=torch.float64 or gradient.device.type!="cpu": raise ValueError("candidate gradient contract changed")
    if not bool(torch.isfinite(gradient).all()): return {"status":"nonfinite_gradient","objective":float(value)}
    phi=torch.dot(gradient,gradient)/2
    if not bool(torch.isfinite(phi)): return {"status":"nonfinite_phi","objective":float(value)}
    return {"status":"finite","objective":float(value),"gradient":gradient.tolist(),
            "phi":float(phi),"gradient_blocks":blocks.gradient_blocks(gradient)}


def accepted_endpoint_stationary(metrics:dict[str,Any])->bool:
    if metrics.get("status")!="finite": return False
    gradient=torch.tensor(metrics.get("gradient"),dtype=torch.float64)
    return gradient.shape==(NC,) and bool(torch.isfinite(gradient).all()) and float(gradient.abs().max())<STATIONARITY_TOL


def modified_operator(original_hvp:Any,mu:float):
    return lambda vector: original_hvp(vector) + torch.cat((torch.zeros(NF,dtype=torch.float64),mu*vector[NF:]))


def choose_dynamics_shift(h:Tensor,schur:dict[str,Any],full_eigenvalues:list[float]|None=None)->tuple[float,Any,dict[str,Any]]:
    if schur.get("status")!="schur_evaluated" or not schur.get("schur_eigenvalues"):
        raise ContinuationRefusal("fresh Hff/Schur qualification is unresolved")
    if min(schur["hff_eigenvalues"])<=schur["hff_positive_floor"]: raise ContinuationRefusal("fresh Hff is not resolved SPD")
    if full_eigenvalues is None: full_eigenvalues=schur["full_hessian_eigenvalues"]
    if full_eigenvalues is None: raise ContinuationRefusal("fresh full-H eigenvalues are unavailable")
    s=torch.tensor(schur["schur_complement"],dtype=torch.float64); seig=torch.tensor(schur["schur_eigenvalues"],dtype=torch.float64)
    hnorm=torch.linalg.matrix_norm(h); snorm=torch.linalg.matrix_norm(s)
    hfloor=max(blocks.CURVATURE_REL_TOL*max(abs(x) for x in full_eigenvalues),blocks.ROUNDING_MULT*torch.finfo(torch.float64).eps*float(hnorm))
    sfloor=max(blocks.CURVATURE_REL_TOL*float(seig.abs().max()),blocks.ROUNDING_MULT*torch.finfo(torch.float64).eps*float(snorm))
    full_spd=min(full_eigenvalues)>hfloor; schur_spd=float(seig[0])>sfloor
    mu=0. if full_spd and schur_spd else max(0.,1.-float(seig[0]))
    shifted=h.clone(); shifted[NF:,NF:]+=mu*torch.eye(ND,dtype=torch.float64)
    try: preconditioner,qualification=block_probe.block_inverse_preconditioner(shifted)
    except block_probe.StepRefusal as error: raise ContinuationRefusal(f"shifted SPD block preconditioner refused: {error}") from error
    return mu,preconditioner,{**qualification,"full_spd_resolved":full_spd,"schur_spd_resolved":schur_spd,"full_positive_floor":hfloor,"schur_positive_floor":sfloor}


def refusal_message(error:Exception)->str:
    message=str(error)
    if isinstance(error,block_probe.StepRefusal) and message=="300-second internal budget exhausted during PCG": return "1200-second continuation budget exhausted during modified PCG"
    if isinstance(error,hess_probe.AuditRefusal) and message=="240-second internal Hessian audit budget exhausted": return "1200-second continuation budget exhausted during fresh Hessian audit"
    if isinstance(error,hess_probe.AuditRefusal) and message=="240-second internal budget exhausted after independent 27th HVP": return "1200-second continuation budget exhausted after independent 27th HVP"
    if isinstance(error,hess_probe.AuditRefusal) and message=="240-second internal budget exhausted after Schur diagnostic": return "1200-second continuation budget exhausted after Schur diagnostic"
    if isinstance(error,block_probe.StepRefusal) and message=="PCG or fresh original-H true residual gate failed": return "PCG or fresh modified-A_mu true residual gate failed"
    return message


def run(output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("coupled continuation output must be fresh")
    require_pins(); r8=json.loads(R8.read_text()); accepted=json.loads(AUDIT.read_text())
    hess_raw=json.loads(hess_probe.RAW.read_text()); hess_base=json.loads(hess_probe.BASE_AUDIT.read_text())
    runtime={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if runtime!=RUNTIME or r8.get("phase")!="finished" or r8.get("numerical_status")!="one_original_J_step_accepted" or r8.get("accepted_control_sha256")!="3aa913cd71d5916b4f3146efad75b0f5dd6766f6bdc81ee6e8c6706cee03fc32":
        raise ValueError("R8 accepted continuation point/runtime is not eligible")
    c=torch.tensor(r8["accepted_control"],dtype=torch.float64); p=torch.tensor(r8["parameters"],dtype=torch.float64)
    if c.shape!=(NC,) or tensor_sha(c)!=r8["accepted_control_sha256"]: raise ValueError("R8 continuation control/hash malformed")
    block_probe.require_pins(); hess_probe.require_pins()
    paths=sorted(set(r8["source_before"])|set(accepted["source_before"])|
        set(hess_raw["source_before"])|set(hess_base["source_before"])|set(block_probe.SOURCE_PATHS)|
        set(tail.SOURCE_PATHS)|{hess_probe.SELF,hess_probe.TEST,str(hess_probe.PLAN.relative_to(ROOT)),SELF,TEST,
         str(PLAN.relative_to(ROOT)),str(R8.relative_to(ROOT)),str(AUDIT.relative_to(ROOT)),
         str(hess_probe.RAW.relative_to(ROOT)),str(hess_probe.BASE_AUDIT.relative_to(ROOT))})
    before={n:sha(ROOT/n) for n in paths}
    source_maps=(r8["source_before"],r8["source_after"],accepted["source_before"],accepted["source_after"],
        hess_raw["source_before"],hess_raw["source_after"],hess_base["source_before"],hess_base["source_after"])
    if any(before.get(k)!=v for source_map in source_maps for k,v in source_map.items()):
        raise ValueError("continuation dependency source map changed")
    problem,original,_,parameters,truth,base=seed._prepare_fixed_seed()
    base_control=torch.tensor(accepted["accepted_control"],dtype=torch.float64)
    base_input=seed._input_identity(problem,original,base_control,parameters,truth)
    current_input=seed._input_identity(problem,original,c,parameters,truth)
    if (tensor_sha(parameters)!=r8["parameters_sha256"] or tensor_sha(parameters)!=accepted["parameters_sha256"]
            or base_input!=accepted["input_after"] or not audit._fixed_input_matches(current_input,base_input,tensor_sha(c))
            or not audit._fixed_input_matches(r8["input_before"],base_input,r8["control_sha256"])
            or r8["input_before"]!=r8["input_after"]
            or base.get("archived_input")!=current_input.get("archived_input")):
        raise ValueError("R8 point changed fixed p/data/time/prior identity")
    fixed_p=parameters.clone(); gradient_fn=torch.func.grad(problem.objective,argnums=0)
    started=time.monotonic(); deadline=started+INTERNAL_SECONDS; wall=started+WALL_SECONDS
    report:dict[str,Any]={"phase":"input_ready","numerical_status":"not_reached","scope":"at most four coupled original-J epochs; no response/root/minimum/physical claim",
        "r8_sha256":R8_SHA,"accepted_audit_sha256":AUDIT_SHA,"plan_sha256":PLAN_SHA,"source_before":before,
        "input_before":current_input,"runtime":runtime,"start_control":c.tolist(),"start_control_sha256":tensor_sha(c),
        "parameters":parameters.tolist(),"parameters_sha256":tensor_sha(parameters),"optimizer_steps_max":MAX_EPOCHS,
        "full_root_claim":False,"minimum_claim":False,"response_computed":False,"score_computed":False,"trials":[],"epochs":[]}
    write(output,report); accepted_steps=0
    def budget()->None:
        if time.monotonic()>=deadline: raise ContinuationRefusal("1200-second shared continuation budget exhausted")
    try:
        for epoch in range(1,MAX_EPOCHS+1):
            budget(); branch,margins=tail._full_current_branch(problem,c,parameters); budget()
            if branch.get("status")!="passed_strict_branch" or not seed._valid_margins(margins,complete=True): raise ContinuationRefusal("current epoch failed strict 3600-stage branch/margin gate")
            fresh=tail._fresh_merit(problem,c,parameters,gradient_fn); budget()
            if fresh is None: raise ContinuationRefusal("fresh current original J/Phi/full gradient is nonfinite")
            base_j,g,phi=fresh; ginf=float(g.abs().max())
            if epoch==1 and (not audit._metric(r8["accepted_objective"],float(base_j))["passed"]
                    or not audit._metric(r8["accepted_phi"],float(phi))["passed"]
                    or not all(audit._metric(float(a),float(b))["passed"] for a,b in zip(r8["accepted_gradient"],g,strict=True))
                    or branch.get("signature_sha256")!=r8["accepted_branch"].get("signature_sha256")):
                raise ContinuationRefusal("fresh R8 accepted J/g/Phi/branch differs from recorded step")
            report["epochs"].append({"epoch":epoch,"control_sha256":tensor_sha(c),"objective":float(base_j),"phi":float(phi),
                "gradient":g.tolist(),"gradient_blocks":blocks.gradient_blocks(g),"branch":branch,"margins":margins})
            write(output,report)
            if ginf<STATIONARITY_TOL:
                report.update(phase="finished",numerical_status="stationary_candidate",full_gradient_inf=ginf,optimizer_steps=accepted_steps)
                break
            fresh_h=hess_probe.fresh_hessian(problem.objective,c,parameters,g,deadline)
            bs=fresh_h["block_schur"]; eig_s=bs.get("schur_eigenvalues")
            if bs.get("status")!="schur_evaluated" or not eig_s: raise ContinuationRefusal("fresh Hff/Schur qualification refused")
            if min(bs["hff_eigenvalues"])<=bs["hff_positive_floor"]: raise ContinuationRefusal("fresh Hff is not resolved SPD")
            full_eig=fresh_h["eigenvalues"]
            h=torch.tensor(fresh_h["hessian"],dtype=torch.float64)
            mu,preconditioner,preaudit=choose_dynamics_shift(h,bs,full_eig)
            mode="original_spd_block_inverse" if mu==0 else "shifted_dynamics_block_inverse"
            def original_hvp(v:Tensor)->Tensor:
                budget(); value=torch.func.jvp(lambda x:gradient_fn(x,parameters),(c,),(v,))[1]; budget(); return value
            modified=original_hvp if mu==0 else modified_operator(original_hvp,mu)
            step,modified_solve=block_probe.solve_newton_direction(modified,g,preconditioner,deadline)
            orig_res=original_hvp(step)+g; orig_rel=float(orig_res.norm()/g.norm()); slope=float(g@step)
            report["epochs"][-1].update(fresh_hessian_sha256=fresh_h["hessian_sha256"],fresh_hessian_eigenvalues=full_eig,
                hessian_symmetry_relative=fresh_h["symmetry_relative"],eigenpair_audit=fresh_h["minimum_eigenpair_audit"],
                block_schur=bs,dynamics_shift_mu=mu,preconditioner_mode=mode,preconditioner_audit=preaudit,
                modified_solve=modified_solve,original_hessian_residual=orig_res.tolist(),original_hessian_relative_residual=orig_rel,
                g_dot_s=slope)
            write(output,report)
            accepted=False
            accepted_metrics:dict[str,Any]|None=None
            for alpha in ALPHAS:
                budget(); candidate=c+alpha*step; value=problem.objective(candidate,parameters)
                if value.shape!=() or value.dtype!=torch.float64 or value.device.type!="cpu": raise ValueError("candidate J callback contract changed")
                trial:dict[str,Any]={"epoch":epoch,"alpha":alpha,"control":candidate.tolist()}
                if not bool(torch.isfinite(value)):
                    trial["status"]="nonfinite_objective"; report["trials"].append(trial); write(output,report); continue
                candidate_g=gradient_fn(candidate,parameters); metrics=candidate_metrics(value,candidate_g)
                candidate_branch,candidate_margins=tail._full_current_branch(problem,candidate,parameters); budget()
                armijo=block_probe.armijo_holds(float(base_j),float(value),alpha,slope)
                strict=candidate_branch.get("status")=="passed_strict_branch" and candidate_branch.get("euler_stages")==3600 and seed._valid_margins(candidate_margins,complete=True)
                trial.update(metrics,branch=candidate_branch,branch_margins=candidate_margins,
                    branch_signature_changed=candidate_branch.get("signature_sha256")!=branch.get("signature_sha256"),
                    armijo_passed=armijo,strict_point_passed=strict,
                    status="accepted" if armijo and strict and metrics["status"]=="finite" else metrics["status"] if metrics["status"]!="finite" else "rejected")
                report["trials"].append(trial); write(output,report)
                if armijo and strict and metrics["status"]=="finite":
                    c=candidate; accepted=True; accepted_metrics=metrics; accepted_steps+=1; break
            if not accepted:
                report.update(phase="finished",numerical_status="original_J_armijo_grid_exhausted",optimizer_steps=accepted_steps); break
            report.update(phase="epoch_accepted",optimizer_steps=accepted_steps,last_accepted_control=c.tolist(),last_accepted_control_sha256=tensor_sha(c)); write(output,report)
            if accepted_metrics is not None and accepted_endpoint_stationary(accepted_metrics):
                report.update(phase="finished",numerical_status="stationary_candidate",
                    full_gradient_inf=max(abs(float(value)) for value in accepted_metrics["gradient"]))
                break
        else: report.update(phase="finished",numerical_status="epoch_limit",optimizer_steps=accepted_steps)
    except (ContinuationRefusal,block_probe.StepRefusal,hess_probe.AuditRefusal) as error:
        report.update(phase="finished",numerical_status="continuation_refused",optimizer_steps=accepted_steps,refusal=f"{type(error).__name__}: {refusal_message(error)}")
    after={n:sha(ROOT/n) for n in paths}; input_after=seed._input_identity(problem,original,c,parameters,truth)
    runtime_after={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if (after!=before or not audit._fixed_input_matches(input_after,current_input,tensor_sha(c))
            or runtime_after!=runtime or sha(R8)!=R8_SHA or sha(AUDIT)!=AUDIT_SHA or sha(PLAN)!=PLAN_SHA):
        raise ValueError("coupled continuation source/input/runtime changed")
    report.update(source_after=after,source_unchanged=True,input_after=input_after,input_unchanged=True,runtime_after=runtime_after,elapsed_seconds=time.monotonic()-started)
    write(output,report); return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
