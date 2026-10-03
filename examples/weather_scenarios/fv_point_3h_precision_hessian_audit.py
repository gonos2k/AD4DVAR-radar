"""Fresh read-only full-Hessian and Schur audit at the S4 precision terminal."""
from __future__ import annotations

import argparse,hashlib,json,math,platform,re,time
from pathlib import Path
from typing import Any
import torch
from torch import Tensor
from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit
from examples.weather_scenarios import fv_point_3h_field_precision_continuation as precision
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks

ROOT=Path(__file__).resolve().parents[2]; EVIDENCE=ROOT/"graphify-out/fv-root-cause-20260919"
RAW=EVIDENCE/"field_precision_continuation_attempt1/field_precision_continuation.json"
RAW_SHA="e8e0d395a212a37227185b40ccb977f26023795527a9c820e414ed46b6202cf9"
BASE_AUDIT=EVIDENCE/"point_3h_accepted_endpoint_audit_attempt1/audit.json"
BASE_AUDIT_SHA="b20c5f85cc8921d8cbff69eddac6167c453739906b7a05f5ac1737df405147e6"
PLAN=EVIDENCE/"S4_POINT_TERMINAL_PRECISION_HESSIAN_AUDIT_PLAN_20261003.md"
PLAN_SHA="0228acc3f56d9d580d2273abf40935aff8192334e18735b0b6be8ddfc0d2e005"
SELF="examples/weather_scenarios/fv_point_3h_precision_hessian_audit.py"
TEST="tests/test_fv_point_3h_precision_hessian_audit.py"
SELF_SHA="493d95ec3fe43a10f2e8d42951b99279310ef9e4c39e30f9bae112c13a6d335b"
TEST_SHA="bb816c1da2196a80c17fa4bba219710f5ca8367159ad8e5d48b06cd874f67cf3"
_SELF_RE=re.compile(rb'(?m)^SELF_SHA="[0-9a-f]{64}"$')
NC,NF,NSTAGE=26,20,3600
INTERNAL_SECONDS=240.0
RUNTIME={"device":"CPU FP64","python":"3.12.13","torch":"2.13.0"}


class AuditRefusal(ValueError):
    """A declared terminal-point numerical qualification refusal."""


def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor_sha(value:Tensor)->str: return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def write(path:Path,data:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n"); tmp.replace(path)


def require_pins()->None:
    source,count=_SELF_RE.subn(b'SELF_SHA="<canonical-self-pin>"',(ROOT/SELF).read_bytes())
    if count!=1 or hashlib.sha256(source).hexdigest()!=SELF_SHA or sha(ROOT/TEST)!=TEST_SHA:
        raise ValueError("precision Hessian audit source/test pin changed")
    if any(sha(p)!=h for p,h in ((RAW,RAW_SHA),(BASE_AUDIT,BASE_AUDIT_SHA),(PLAN,PLAN_SHA))):
        raise ValueError("precision Hessian audit raw/audit/plan pin changed")


def terminal_control(raw:dict[str,Any])->Tensor:
    value=torch.tensor(raw.get("terminal_control"),dtype=torch.float64)
    if (raw.get("numerical_status")!="continuation_refused" or raw.get("field_issuance") is not False
            or raw.get("terminal_diagnostics_status")!="completed" or value.shape!=(NC,)
            or tensor_sha(value)!="a890f344503cb3da02919de45f1c5b244beac300b1d4d2ca36f403c653e42b6d"):
        raise ValueError("precision continuation raw does not contain the pinned finite terminal point")
    return value


def fresh_hessian(objective,control:Tensor,parameters:Tensor,gradient:Tensor,deadline:float)->dict[str,Any]:
    grad=torch.func.grad(objective,argnums=0); eye=torch.eye(NC,dtype=torch.float64); columns=[]
    def product(vector:Tensor)->Tensor:
        if time.monotonic()>=deadline: raise AuditRefusal("240-second internal Hessian audit budget exhausted")
        result=torch.func.jvp(lambda c:grad(c,parameters),(control,),(vector,))[1]
        if result.shape!=(NC,) or not bool(torch.isfinite(result).all()): raise AuditRefusal("fresh full-H HVP is invalid/nonfinite")
        return result
    for i in range(NC): columns.append(product(eye[i]))
    h=torch.stack(columns,dim=1); norm=torch.linalg.matrix_norm(h)
    if not bool(torch.isfinite(norm)) or float(norm)<=0: raise AuditRefusal("fresh full Hessian has no finite scale")
    symmetry=float(torch.linalg.matrix_norm(h-h.T)/norm)
    if not math.isfinite(symmetry) or symmetry>blocks.SYMMETRY_TOL: raise AuditRefusal("fresh full-H symmetry gate failed")
    try: eig,vec=torch.linalg.eigh(.5*(h+h.T))
    except torch.linalg.LinAlgError as error: raise AuditRefusal(f"full-H eigensolver failed: {error}") from error
    if not bool(torch.isfinite(eig).all()&torch.isfinite(vec).all()): raise AuditRefusal("full-H eigensystem is nonfinite")
    independent=product(vec[:,0]); eigenpair=blocks.eigenpair_audit(independent,eig[0],vec[:,0],norm)
    if not eigenpair["passed"]: raise AuditRefusal("independent 27th minimum-eigenpair HVP residual failed")
    if time.monotonic()>=deadline: raise AuditRefusal("240-second internal budget exhausted after independent 27th HVP")
    try: schur=blocks.schur_diagnostic(h,gradient)
    except blocks.DiagnosticRefusal as error: raise AuditRefusal(f"block/Schur diagnostic refused: {error}") from error
    except torch.linalg.LinAlgError as error: raise AuditRefusal(f"Schur eigensolver failed: {error}") from error
    if time.monotonic()>=deadline: raise AuditRefusal("240-second internal budget exhausted after Schur diagnostic")
    return {"hessian":h.tolist(),"hessian_sha256":tensor_sha(h),"hvp_columns":NC,
        "hvp_calls_total":NC+1,"symmetry_relative":symmetry,"eigenvalues":eig.tolist(),
        "minimum_eigenpair_audit":eigenpair,"block_schur":schur}


def run(output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("precision Hessian audit output must be fresh")
    require_pins(); raw=json.loads(RAW.read_text()); accepted=json.loads(BASE_AUDIT.read_text())
    runtime={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if runtime!=RUNTIME or raw.get("runtime")!=runtime or raw.get("runtime_after")!=runtime or raw.get("source_unchanged") is not True or raw.get("input_unchanged") is not True:
        raise ValueError("precision terminal source/input/runtime record is not eligible")
    if raw.get("source_before")!=raw.get("source_after") or accepted.get("source_before")!=accepted.get("source_after"):
        raise ValueError("pinned terminal or accepted-audit source map changed")
    control=terminal_control(raw); params=torch.tensor(raw["parameters"],dtype=torch.float64)
    paths=sorted(set(raw["source_before"])|set(accepted["source_before"])|{SELF,TEST,str(PLAN.relative_to(ROOT)),str(RAW.relative_to(ROOT)),str(BASE_AUDIT.relative_to(ROOT))})
    before={name:sha(ROOT/name) for name in paths}
    for source_map in (raw["source_before"],raw["source_after"],accepted["source_before"],accepted["source_after"]):
        if any(before.get(k)!=v for k,v in source_map.items()): raise ValueError("precision Hessian dependency source map changed")
    problem,original,_,parameters,truth,base=seed._prepare_fixed_seed()
    original_control=torch.tensor(accepted["accepted_control"],dtype=torch.float64)
    original_input=seed._input_identity(problem,original,original_control,parameters,truth)
    terminal_input=seed._input_identity(problem,original,control,parameters,truth)
    if (raw["parameters_sha256"]!=tensor_sha(parameters) or tensor_sha(params)!=tensor_sha(parameters)
            or original_input!=accepted["input_after"]
            or not audit._fixed_input_matches(terminal_input,original_input,tensor_sha(control))
            or base.get("archived_input")!=terminal_input.get("archived_input")):
        raise ValueError("terminal control changed fixed parameters, data, time, or prior identity")
    accepted_trials=[row for row in raw["trials"] if row.get("accepted") is True]
    if (not accepted_trials or raw.get("input_before")!=accepted.get("input_after")
            or raw.get("input_after")!=accepted.get("input_after")
            or not torch.equal(control,torch.tensor(accepted_trials[-1]["candidate_full_control"],dtype=torch.float64))
            or not torch.equal(control[20:],torch.tensor(raw["fixed_dynamics"],dtype=torch.float64))):
        raise ValueError("terminal control/trial/fixed input provenance does not match the accepted attempt")
    expected=accepted_trials[-1]["branch"]["signature_sha256"]
    report:dict[str,Any]={"phase":"input_ready","numerical_status":"not_reached",
        "scope":"fresh terminal full-Hessian/20+6 block Schur audit only; no step, score, response, root or minimum claim",
        "raw_sha256":RAW_SHA,"accepted_audit_sha256":BASE_AUDIT_SHA,"plan_sha256":PLAN_SHA,
        "source_before":before,"runtime":runtime,"input_before":terminal_input,
        "terminal_control":control.tolist(),"terminal_control_sha256":tensor_sha(control),
        "parameters":parameters.tolist(),"parameters_sha256":tensor_sha(parameters),
        "full_root_claim":False,"optimizer_step_applied":False,"score_computed":False,
        "adjoint_computed":False,"response_computed":False}
    write(output,report); started=time.monotonic()
    try:
        branch,margins=tail._full_current_branch(problem,control,parameters)
        if branch.get("status")!="passed_strict_branch" or branch.get("euler_stages")!=NSTAGE or branch.get("signature_sha256")!=expected or not seed._valid_margins(margins,complete=True):
            raise AuditRefusal("fresh terminal strict 3600-stage branch/margin replay failed")
        gradient_fn=torch.func.grad(problem.objective,argnums=0)
        fresh=tail._fresh_merit(problem,control,parameters,gradient_fn)
        if fresh is None: raise AuditRefusal("fresh terminal J/Phi/full gradient is nonfinite")
        objective,gradient,phi=fresh
        checks:dict[str,Any]={"J":audit._metric(raw["terminal_objective"],float(objective)),
            "Phi":audit._metric(raw["terminal_full_phi"],float(phi)),
            "gradient_components":[audit._metric(float(a),float(b)) for a,b in zip(raw["terminal_full_gradient"],gradient,strict=True)]}
        if not checks["J"]["passed"] or not checks["Phi"]["passed"] or not all(x["passed"] for x in checks["gradient_components"]):
            raise AuditRefusal("fresh terminal J/Phi/full-gradient differs from precision raw")
        report.update(phase="endpoint_checked",numerical_status="terminal_metrics_reproduced",
            objective=float(objective),phi=float(phi),gradient=gradient.tolist(),
            gradient_blocks=blocks.gradient_blocks(gradient),saved_metric_checks=checks,
            branch=branch,branch_margins=margins)
        write(output,report)
        hessian=fresh_hessian(problem.objective,control,parameters,gradient,started+INTERNAL_SECONDS)
        report.update(phase="finished",numerical_status="diagnostic_completed",hessian_status=hessian["block_schur"]["status"],
            hessian=hessian["hessian"],hessian_sha256=hessian["hessian_sha256"],hvp_columns_completed=NC,
            hvp_calls_total=hessian["hvp_calls_total"],full_hessian_symmetry_relative=hessian["symmetry_relative"],
            full_hessian_eigenvalues=hessian["eigenvalues"],eigenpair_audit=hessian["minimum_eigenpair_audit"],
            block_schur=hessian["block_schur"])
    except AuditRefusal as error:
        report.update(phase="finished",numerical_status="diagnostic_refusal",refusal=str(error))
    after={name:sha(ROOT/name) for name in paths}; input_after=seed._input_identity(problem,original,control,parameters,truth)
    runtime_after={"device":"CPU FP64","python":platform.python_version(),"torch":torch.__version__}
    if (after!=before or input_after!=terminal_input or runtime_after!=runtime
            or sha(RAW)!=RAW_SHA or sha(BASE_AUDIT)!=BASE_AUDIT_SHA or sha(PLAN)!=PLAN_SHA):
        raise ValueError("terminal precision Hessian audit source/input/runtime changed")
    report.update(source_after=after,source_unchanged=True,input_after=input_after,input_unchanged=True,runtime_after=runtime_after,
        elapsed_seconds=time.monotonic()-started)
    write(output,report); return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
