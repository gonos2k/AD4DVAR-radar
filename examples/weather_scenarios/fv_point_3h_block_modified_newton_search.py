"""One original-objective Armijo search with shifted block-Schur preconditioning."""
from __future__ import annotations

import argparse,hashlib,json,math,platform,re,time
from pathlib import Path
from typing import Any
import torch
from torch import Tensor
from advar import matrix_free
from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_precision_hessian_audit as hessian_probe
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_schur_newton_step as schur_probe
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks
from examples.weather_scenarios import fv_point_3h_schur_newton_step as step_probe

ROOT=Path(__file__).resolve().parents[2]; EVIDENCE=ROOT/"graphify-out/fv-root-cause-20260919"
RAW=EVIDENCE/"precision_terminal_hessian_audit_attempt1/precision_hessian_audit.json"; RAW_SHA="eb0f716bb36743ccaeb9e42845debedbbf92c110f49e2a4779184541a0e263af"
BASE_AUDIT=EVIDENCE/"point_3h_accepted_endpoint_audit_attempt1/audit.json"; BASE_AUDIT_SHA="b20c5f85cc8921d8cbff69eddac6167c453739906b7a05f5ac1737df405147e6"
PLAN=EVIDENCE/"S4_BLOCK_MODIFIED_NEWTON_SEARCH_PLAN_20261003.md"; PLAN_SHA="f386c7e39e08c2c813c5dbc65566e95cafd4cecbd807158c29d47bfc4c34102d"
SELF="examples/weather_scenarios/fv_point_3h_block_modified_newton_search.py"; TEST="tests/test_fv_point_3h_block_modified_newton_search.py"
SELF_SHA="5e82dd1e0f5f59ee223a7cbbe8f169dd30398bd66db689d76dbb2d4c062c293d"
TEST_SHA="e9a8d17711166c04990c69eb3495f7727ff7e4cfce71529c3f4a0a55d0c4ecab"
_SELF_RE=re.compile(rb'(?m)^SELF_SHA="[0-9a-f]{64}"$')
NC,NF,NSTAGE=26,20,3600; MU_FLOOR=1.; ALPHAS=tuple(2.**-i for i in range(16))
INTERNAL_SECONDS=240.; WALL_SECONDS=300.; ARMIJO_C1=1e-4
RUNTIME={"device":"CPU FP64","python":"3.12.13","torch":"2.13.0"}


class SearchRefusal(ValueError):
    """Declared numerical or strict-point search refusal."""


def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor_sha(value:Tensor)->str: return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def write(path:Path,report:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"); tmp.replace(path)


def require_pins()->None:
    source,count=_SELF_RE.subn(b'SELF_SHA="<canonical-self-pin>"',(ROOT/SELF).read_bytes())
    if count!=1 or hashlib.sha256(source).hexdigest()!=SELF_SHA or sha(ROOT/TEST)!=TEST_SHA:
        raise ValueError("modified-Newton producer/test source pin changed")
    if any(sha(p)!=h for p,h in ((RAW,RAW_SHA),(BASE_AUDIT,BASE_AUDIT_SHA),(PLAN,PLAN_SHA))):
        raise ValueError("modified-Newton raw/audit/plan pin changed")


def shift_and_precondition(raw:dict[str,Any])->tuple[float,Tensor,Any,dict[str,Any]]:
    schur=raw.get("block_schur",{}); eig=schur.get("schur_eigenvalues",[])
    if not eig or not all(math.isfinite(v) for v in eig): raise SearchRefusal("fresh Schur spectrum is absent/nonfinite")
    mu=max(0.,MU_FLOOR-min(eig)); h=torch.tensor(raw.get("hessian"),dtype=torch.float64)
    if h.shape!=(NC,NC) or tensor_sha(h)!=raw.get("hessian_sha256"):
        raise ValueError("pinned fresh terminal Hessian shape/hash changed")
    shifted=h.clone(); shifted[NF:,NF:]+=mu*torch.eye(6,dtype=torch.float64)
    try: preconditioner,qualification=schur_probe.block_inverse_preconditioner(shifted)
    except schur_probe.StepRefusal as error: raise SearchRefusal(f"shifted SPD block preconditioner refused: {error}") from error
    return mu,h,preconditioner,qualification


def modified_operator(original_hvp,mu:float):
    def apply(vector:Tensor)->Tensor:
        result=original_hvp(vector).clone(); result[NF:]+=mu*vector[NF:]; return result
    return apply


def terminal_input_matches(raw:dict[str,Any],terminal_input:dict[str,Any],base_input:dict[str,Any],control_sha:str)->bool:
    return (raw.get("input_before")==terminal_input and raw.get("input_after")==terminal_input
        and audit._fixed_input_matches(terminal_input,base_input,control_sha))


def candidate_metrics(value:Tensor,gradient:Tensor)->dict[str,Any]:
    if value.ndim!=0 or value.dtype!=torch.float64 or value.device.type!="cpu": raise ValueError("candidate J must be a CPU FP64 scalar")
    if not bool(torch.isfinite(value)): return {"status":"nonfinite_objective"}
    if gradient.shape!=(NC,) or gradient.dtype!=torch.float64 or gradient.device.type!="cpu": raise ValueError("candidate gradient must be CPU FP64 length 26")
    if not bool(torch.isfinite(gradient).all()): return {"status":"nonfinite_gradient","objective":float(value)}
    phi=torch.dot(gradient,gradient)/2
    if not bool(torch.isfinite(phi)): return {"status":"nonfinite_phi","objective":float(value)}
    return {"status":"finite","objective":float(value),"gradient":gradient.tolist(),"phi":float(phi),
        "gradient_blocks":blocks.gradient_blocks(gradient)}


def refusal_message(error:Exception)->str:
    message=str(error)
    if isinstance(error,schur_probe.StepRefusal) and message=="300-second internal budget exhausted during PCG":
        return "240-second modified-Newton internal budget exhausted during PCG"
    if isinstance(error,schur_probe.StepRefusal) and message=="PCG or fresh original-H true residual gate failed":
        return "PCG or fresh modified-A_mu true residual gate failed"
    return message


def run(output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("modified-Newton search output must be fresh")
    require_pins(); raw=json.loads(RAW.read_text()); accepted=json.loads(BASE_AUDIT.read_text())
    runtime={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if (runtime!=RUNTIME or raw.get("phase")!="finished" or raw.get("numerical_status")!="diagnostic_completed"
            or raw.get("source_unchanged") is not True or raw.get("input_unchanged") is not True
            or raw.get("hessian_status")!="schur_evaluated" or raw.get("response_computed") is not False
            or raw.get("score_computed") is not False or raw.get("adjoint_computed") is not False
            or raw.get("full_root_claim") is not False or runtime!=raw.get("runtime") or runtime!=raw.get("runtime_after")):
        raise ValueError("precision terminal Hessian raw/runtime is not eligible")
    if (accepted.get("phase")!="finished" or accepted.get("numerical_status")!="diagnostic_completed"
            or accepted.get("source_unchanged") is not True or accepted.get("input_unchanged") is not True):
        raise ValueError("accepted endpoint audit provenance is not eligible")
    control=torch.tensor(raw["terminal_control"],dtype=torch.float64); p=torch.tensor(raw["parameters"],dtype=torch.float64)
    if control.shape!=(NC,) or tensor_sha(control)!="a890f344503cb3da02919de45f1c5b244beac300b1d4d2ca36f403c653e42b6d":
        raise ValueError("modified-Newton start control is not the pinned terminal point")
    blocks.require_source_pins(); step_probe.require_pins()
    paths=sorted(set(raw["source_before"])|set(accepted["source_before"])|set(step_probe.SOURCE_PATHS)|{SELF,TEST,str(PLAN.relative_to(ROOT)),str(RAW.relative_to(ROOT)),str(BASE_AUDIT.relative_to(ROOT))})
    before={name:sha(ROOT/name) for name in paths}
    for sm in (raw["source_before"],raw["source_after"],accepted["source_before"],accepted["source_after"]):
        if any(before.get(k)!=v for k,v in sm.items()): raise ValueError("modified-Newton dependency source map changed")
    problem,original,_,parameters,truth,base=seed._prepare_fixed_seed()
    base_control=torch.tensor(accepted["accepted_control"],dtype=torch.float64)
    base_input=seed._input_identity(problem,original,base_control,parameters,truth)
    terminal_input=seed._input_identity(problem,original,control,parameters,truth)
    if (tensor_sha(parameters)!=raw["parameters_sha256"] or tensor_sha(parameters)!=accepted["parameters_sha256"]
            or base_input!=accepted["input_after"] or not terminal_input_matches(raw,terminal_input,base_input,tensor_sha(control))
            or base.get("archived_input")!=terminal_input.get("archived_input")):
        raise ValueError("modified-Newton terminal fixed data/time/prior identity changed")
    prior_schur=raw["block_schur"].get("schur_eigenvalues",[])
    if not prior_schur or min(prior_schur)>=MU_FLOOR: raise ValueError("terminal Schur block is not the declared negative-curvature case")
    mu,hessian,preconditioner,preconditioner_audit=shift_and_precondition(raw)
    fixed_p=parameters.clone(); gradient_fn=torch.func.grad(problem.objective,argnums=0)
    started=time.monotonic(); deadline=started+INTERNAL_SECONDS
    report:dict[str,Any]={"phase":"input_ready","numerical_status":"not_reached",
        "scope":"one shifted-block-preconditioned original-J Armijo search; no original-H Newton-convergence claim",
        "raw_sha256":RAW_SHA,"accepted_audit_sha256":BASE_AUDIT_SHA,"plan_sha256":PLAN_SHA,
        "source_before":before,"input_before":terminal_input,"runtime":runtime,"control":control.tolist(),
        "control_sha256":tensor_sha(control),"parameters":parameters.tolist(),"parameters_sha256":tensor_sha(parameters),
        "original_terminal_hessian_min_eigenvalue":min(raw["full_hessian_eigenvalues"]),
        "original_schur_eigenvalues":prior_schur,"dynamics_shift_mu":mu,"preconditioner_audit":preconditioner_audit,
        "full_root_claim":False,"minimum_claim":False,"physical_validated":False,"adjoint_computed":False,
        "response_computed":False,"score_computed":False,"optimizer_steps_max":1,"trials":[]}
    write(output,report)
    try:
        branch,margins=tail._full_current_branch(problem,control,parameters)
        if (branch.get("status")!="passed_strict_branch" or branch.get("euler_stages")!=NSTAGE
                or branch.get("signature_sha256")!=raw["branch"]["signature_sha256"] or not seed._valid_margins(margins,complete=True)):
            raise SearchRefusal("terminal start failed fresh strict branch/margin replay")
        fresh=tail._fresh_merit(problem,control,parameters,gradient_fn)
        if fresh is None: raise SearchRefusal("terminal original J/Phi/full gradient is nonfinite")
        base_j,g,phi=fresh
        if (not audit._metric(raw["objective"],float(base_j))["passed"]
                or not audit._metric(raw["phi"],float(phi))["passed"]
                or not all(audit._metric(float(a),float(b))["passed"] for a,b in zip(raw["gradient"],g,strict=True))):
            raise SearchRefusal("fresh terminal J/Phi/full gradient differs from precision-Hessian raw")
        def original_hvp(v:Tensor)->Tensor:
            if time.monotonic()>=deadline: raise SearchRefusal("240-second modified-Newton budget exhausted during HVP")
            return torch.func.jvp(lambda c:gradient_fn(c,parameters),(control,),(v,))[1]
        modified_hvp=modified_operator(original_hvp,mu)
        step,modified_solve=schur_probe.solve_newton_direction(modified_hvp,g,preconditioner,deadline)
        original_residual=original_hvp(step)+g
        original_relative=float(torch.linalg.vector_norm(original_residual)/torch.linalg.vector_norm(g))
        report.update(phase="line_search",objective=float(base_j),full_gradient=g.tolist(),
            gradient_blocks=blocks.gradient_blocks(g),phi=float(phi),branch=branch,branch_margins=margins,
            step=step.tolist(),modified_system=modified_solve,
            original_hessian_residual=original_residual.tolist(),original_hessian_relative_residual=original_relative)
        write(output,report)
        slope=float(torch.dot(g,step))
        for alpha in ALPHAS:
            if time.monotonic()>=deadline: raise SearchRefusal("240-second modified-Newton budget exhausted in line search")
            candidate=control+alpha*step; value=problem.objective(candidate,parameters)
            trial:dict[str,Any]={"alpha":alpha,"control":candidate.tolist()}
            if value.ndim!=0 or value.dtype!=torch.float64 or value.device.type!="cpu": raise ValueError("trial objective contract changed")
            if not bool(torch.isfinite(value)):
                trial["status"]="nonfinite_objective"; report["trials"].append(trial); write(output,report); continue
            candidate_gradient=gradient_fn(candidate,parameters); numeric=candidate_metrics(value,candidate_gradient)
            candidate_branch,candidate_margins=tail._full_current_branch(problem,candidate,parameters)
            if time.monotonic()>=deadline: raise SearchRefusal("240-second modified-Newton budget exhausted after endpoint audit")
            armijo=schur_probe.armijo_holds(float(base_j),float(value),alpha,slope)
            strict=(candidate_branch.get("status")=="passed_strict_branch" and candidate_branch.get("euler_stages")==NSTAGE
                and seed._valid_margins(candidate_margins,complete=True))
            trial.update(status="accepted" if armijo and strict and numeric["status"]=="finite" else numeric["status"] if numeric["status"]!="finite" else "rejected",
                **{k:v for k,v in numeric.items() if k!="status"},
                branch=candidate_branch,branch_margins=candidate_margins,
                branch_signature_changed=candidate_branch.get("signature_sha256")!=branch.get("signature_sha256"),
                armijo_passed=armijo,strict_point_passed=strict)
            report["trials"].append(trial); write(output,report)
            if armijo and strict and numeric["status"]=="finite":
                report.update(phase="finished",numerical_status="one_original_J_step_accepted",optimizer_steps=1,
                    accepted_alpha=alpha,accepted_control=candidate.tolist(),accepted_control_sha256=tensor_sha(candidate),
                    accepted_objective=numeric["objective"],accepted_gradient=numeric["gradient"],accepted_phi=numeric["phi"],
                    accepted_gradient_blocks=numeric["gradient_blocks"],accepted_branch=candidate_branch,accepted_margins=candidate_margins)
                break
        else: report.update(phase="finished",numerical_status="original_J_armijo_grid_exhausted",optimizer_steps=0)
    except (SearchRefusal,schur_probe.StepRefusal) as error:
        report.update(phase="finished",numerical_status="modified_search_refused",optimizer_steps=0,
                      refusal=f"{type(error).__name__}: {refusal_message(error)}")
    after={name:sha(ROOT/name) for name in paths}; input_after=seed._input_identity(problem,original,control,parameters,truth)
    runtime_after={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if after!=before or input_after!=terminal_input or runtime_after!=runtime or sha(RAW)!=RAW_SHA or sha(BASE_AUDIT)!=BASE_AUDIT_SHA or sha(PLAN)!=PLAN_SHA:
        raise ValueError("modified-Newton source/input/runtime changed during search")
    report.update(source_after=after,source_unchanged=True,input_after=input_after,input_unchanged=True,runtime_after=runtime_after,
        elapsed_seconds=time.monotonic()-started)
    write(output,report); return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
