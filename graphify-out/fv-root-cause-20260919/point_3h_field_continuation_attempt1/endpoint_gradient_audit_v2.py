from __future__ import annotations
import base64, hashlib, json, math, platform
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from examples.weather_scenarios import fv_point_3h_field_continuation as probe
from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as endpoint
OUT=ROOT/"graphify-out/fv-root-cause-20260919/point_3h_field_continuation_attempt1"
PREFLIGHT=json.loads((OUT/"preflight.json").read_text()); RUN=json.loads((OUT/"field_continuation.run.json").read_text())
CHILD=json.loads((OUT/"field_continuation.json").read_text()); RAW=json.loads(probe.RAW.read_text()); ACCEPTED=json.loads(probe.BASE_AUDIT.read_text())
if CHILD.get("numerical_status")!="continuation_refused" or CHILD.get("field_issuance") is not False: raise ValueError("unexpected child status")
if RUN.get("execution_status")!="completed" or RUN.get("source_bytes_unchanged") is not True: raise ValueError("guard or source-byte audit failed")
for name, b64 in PREFLIGHT["source_bytes_base64_before"].items():
    now=(ROOT/name).read_bytes()
    if base64.b64encode(now).decode("ascii") != b64: raise ValueError(f"exact source bytes changed: {name}")
trial=next(row for row in reversed(CHILD["trials"]) if row.get("accepted") is True)
full=torch.tensor(trial["candidate_full_control"],dtype=torch.float64); warm=torch.tensor(CHILD["start_control"],dtype=torch.float64)
problem,original,_,parameters,truth,_=probe.seed._prepare_fixed_seed()
identity=probe.seed._input_identity(problem,original,torch.tensor(ACCEPTED["accepted_control"],dtype=torch.float64),parameters,truth)
if identity != ACCEPTED["input_after"] or identity != CHILD["input_after"]: raise ValueError("accepted data/time/prior identity changed")
if (full.shape!=(26,) or not torch.equal(full[20:],warm[20:]) or not torch.equal(full[20:],torch.tensor(RAW["fixed_dynamics"],dtype=torch.float64))
    or probe.tensor_sha(parameters)!=CHILD["parameters_sha256"]): raise ValueError("final candidate changed fixed dynamics or parameters")
grad=torch.func.grad(problem.objective,argnums=0)(full,parameters); j=problem.objective(full,parameters)
if not bool(torch.isfinite(grad).all()) or not bool(torch.isfinite(j)): raise ValueError("terminal full gradient/objective is nonfinite")
gf,gd=grad[:20],grad[20:]
if not endpoint._metric(trial["objective"],float(j))["passed"] or not endpoint._metric(trial["gradient_max"],float(gf.abs().max()))["passed"]:
    raise ValueError("terminal trial summary differs from independent fresh full-objective evaluation")
report={"scope":"read-only endpoint audit of last accepted continuation trial; diagnostic only, no issuance/root/response/physical claim",
 "execution_status":"completed","numerical_status":"diagnostic_only","child_numerical_status":CHILD["numerical_status"],"field_issuance":False,
 "attempt1_sha256":probe.RAW_SHA,"accepted_audit_sha256":probe.BASE_AUDIT_SHA,"continuation_report_sha256":hashlib.sha256((OUT/"field_continuation.json").read_bytes()).hexdigest(),
 "preflight_exact_source_bytes_match":True,"source_sha256_before":PREFLIGHT["source_sha256_before"],
 "input_identity_matches_accepted_audit":True,"input_identity":identity,"runtime":{"python":platform.python_version(),"torch":torch.__version__},
 "terminal_trial":{"iteration":trial["iteration"],"step_scale":trial["step_scale"],"candidate_full_control_sha256":probe.tensor_sha(full),"branch":trial["branch"],"objective":float(j),"saved_objective":trial["objective"]},
 "fixed_parameters_sha256":probe.tensor_sha(parameters),"fixed_dynamics":full[20:].tolist(),"fixed_dynamics_unchanged":True,
 "full_gradient":grad.tolist(),"field_gradient":gf.tolist(),"field_gradient_max":float(gf.abs().max()),"field_gradient_l2":float(torch.linalg.vector_norm(gf)),
 "dynamics_gradient":gd.tolist(),"dynamics_gradient_max":float(gd.abs().max()),"dynamics_gradient_l2":float(torch.linalg.vector_norm(gd)),
 "full_gradient_max":float(grad.abs().max()),"full_gradient_l2":float(torch.linalg.vector_norm(grad)),"full_phi":float(torch.dot(grad,grad)/2),
 "saved_field_gradient_max":trial["gradient_max"],"saved_objective":trial["objective"],"field_stationarity_gate_passed":float(gf.abs().max())<1e-10,
 "strict_endpoint_branch_recorded":trial["branch"].get("status")=="passed_strict_branch","response_computed":False,"full_root_claim":False,"physical_validated":False}
(OUT/"endpoint_gradient_audit_v2.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({k:report[k] for k in ("execution_status","numerical_status","field_gradient_max","dynamics_gradient_max","full_gradient_max","objective","field_stationarity_gate_passed","fixed_dynamics_unchanged","strict_endpoint_branch_recorded")},indent=2,sort_keys=True))
