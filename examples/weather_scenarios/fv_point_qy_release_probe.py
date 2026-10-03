"""Release qy into its evidence-backed positive sector; keep qx structurally zero."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable
import torch
from torch import Tensor
from advar import matrix_free
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from examples.weather_scenarios import fv_point_active_face_stationarity as single
from examples.weather_scenarios import fv_point_event_diagnostic as event
from examples.weather_scenarios.fv_partial_signed_face_slices import _true_residual_monitor
from examples.weather_scenarios.fv_point_paired_face_binding import FVPointPairedFaceBinding
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"graphify-out/fv-root-cause-20260919"
PAIRED,NORMAL,PLAN=(OUT/"point_paired_face_stationarity_attempt1/stationarity.json",
                    OUT/"point_paired_face_normal_attempt2/normal.json",OUT/"R2_POINT_QY_RELEASE_PLAN_20261003.md")
NORMAL_INPUT=OUT/"R2_POINT_PAIRED_FACE_NORMAL_INPUT_20261003.json"
PAIRED_SHA,NORMAL_SHA,PLAN_SHA,NORMAL_INPUT_SHA=(
    "d38d1cec4550a6347ceb14db23395679a59a7cad89f09096ec7185f984f6a8cf",
    "20f82da0f90e944532661ae29985d929b4cf07b21afcfec1b85c69b5d88bb18e",
    "c67a6d92e700366318322bedc402058e3bf5d034af5aa23eeafb6b3768cc93ba",
    "dd9caee732c44ed0961cadd5265005ddfa95bbd5b1d61abab2a08ff6d24a4320")
SOURCE_PATHS=("examples/weather_scenarios/fv_point_qy_release_probe.py","tests/test_fv_point_qy_release_probe.py")
ETA_Y=1e-4
class ReleaseRefusal(single.PointFaceRefusal):
    """Known failure of the prescribed positive-qy start or branch gates."""
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _fraction(raw: Any) -> Fraction:
    if not isinstance(raw,list) or len(raw)!=4 or any(type(x)is not int for x in raw): raise ValueError("normal endpoint is not an exact mpmath tuple")
    sign,mantissa,exponent,bits=raw
    if sign not in (0,1) or mantissa<0 or bits<0 or mantissa.bit_length()!=bits: raise ValueError("normal interval endpoint malformed")
    n=-mantissa if sign else mantissa
    return Fraction(n)*2**exponent if exponent>=0 else Fraction(n,2**-exponent)
def qy_right_is_negative(report: dict[str, Any]) -> tuple[Fraction,Fraction]:
    rows=[r for r in report.get("normal_intervals",[]) if r.get("normal_axis")==1 and r.get("side")==1]
    if len(rows)!=1 or not isinstance(rows[0].get("binary"),dict): raise ValueError("normal evidence lacks named qy right interval")
    raw=rows[0]["binary"]; lo,hi=_fraction(raw.get("lower")),_fraction(raw.get("upper"))
    if lo>hi: raise ValueError("qy right interval reversed")
    if hi>=0: raise ReleaseRefusal("captured qy right derivative is not strictly negative")
    return lo,hi
def insert_y_eta(tangent: Tensor, eta: float = ETA_Y) -> Tensor:
    if (not isinstance(tangent,Tensor) or tangent.shape!=(24,) or tangent.dtype!=torch.float64 or tangent.device.type!="cpu" or not bool(torch.isfinite(tangent).all()) or not math.isfinite(eta) or eta<=0):
        raise ValueError("paired tangent and positive y coordinate must be finite CPU FP64")
    return torch.cat((tangent[:20],tangent.new_tensor([eta]),tangent[20:]))
def objective_decrease(old: float, new: float) -> dict[str,float|bool]:
    if not(math.isfinite(old) and math.isfinite(new)): raise ReleaseRefusal("initial paired/released objective is nonfinite")
    budget=128*torch.finfo(torch.float64).eps*max(abs(old),abs(new),torch.finfo(torch.float64).tiny)
    return {"paired":old,"released":new,"decrease":old-new,"roundoff_budget":budget,"passed":old-new>budget}
def optimize_if_decreased(old: float, new: float, optimizer: Callable[[],Any]) -> tuple[dict[str,float|bool],Any]:
    comparison=objective_decrease(old,new)
    if not comparison["passed"]: raise ReleaseRefusal("release objective decrease is below roundoff")
    return comparison,optimizer()
def _strict_full(binding: Any, control: Tensor) -> Tensor:
    try: full=binding.lift(control); binding.chart.to_face_coordinates(full); return full
    except RuntimeError as error:
        if str(error) not in ("face coordinate is outside the strict representable chart domain",
                              "original control is outside the strict representable chart domain"): raise
        raise ReleaseRefusal(str(error)) from error
def _trace(binding: Any, control: Tensor, p: Tensor):
    _strict_full(binding,control); return single.trace(binding,control,p)
def _write(path: Path, report: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"); temp.replace(path)
def run(output: Path) -> dict[str,Any]:
    if output.exists() or sha(PAIRED)!=PAIRED_SHA or sha(NORMAL)!=NORMAL_SHA or sha(PLAN)!=PLAN_SHA or sha(NORMAL_INPUT)!=NORMAL_INPUT_SHA:
        raise ValueError("release output must be fresh and paired/normal/plan pins unchanged")
    paired,normal,capture=json.loads(PAIRED.read_text()),json.loads(NORMAL.read_text()),json.loads(NORMAL_INPUT.read_text())
    if (paired.get("phase")!="finished" or paired.get("numerical_status")!="tangent_stationary_candidate"
            or paired.get("full_root_claim") is not False or paired.get("response_validation")!="not_performed"
            or normal.get("paired_report_sha256")!=PAIRED_SHA or normal.get("input_sha256")!=NORMAL_INPUT_SHA
            or normal.get("numerical_status")!="unsupported_normal" or normal.get("full_root_claim") is not False
            or normal.get("response_validation")!="not_performed"):
        raise ValueError("paired root or normal evidence does not match the declared captures")
    normal_interval=qy_right_is_negative(normal)
    problem,warm,p,direction=event.preflight.fixed_problem()
    identity=event.preflight._input_identity(problem,warm,p,direction)
    if (identity!=paired.get("input_before") or identity!=normal.get("input_identity_before")
            or identity!=capture.get("input_identity") or capture.get("tangent")!=paired.get("last_accepted_tangent")
            or capture.get("parameters_sha256")!=paired.get("parameters_sha256") or capture.get("paired_report_sha256")!=PAIRED_SHA
            or capture.get("native_objective")!=paired.get("final_objective")
            or capture.get("native_analysis_trace")!=paired.get("final_trace",[])[:36]):
        raise ValueError("fixed input differs from paired root/normal capture")
    if event._tensor_sha(p)!=paired.get("parameters_sha256"):
        raise ValueError("fixed parameters differ from paired root")
    runtime={"python":platform.python_version(),"torch":torch.__version__,"device":"CPU FP64"}
    if runtime!=paired.get("runtime") or runtime["python"]!=normal.get("runtime",{}).get("python"):
        raise ValueError("release runtime differs from pinned paired/normal provenance")
    oldsrc,normsrc=paired.get("source_after"),normal.get("source_after")
    if (not isinstance(oldsrc,dict) or paired.get("source_before")!=oldsrc
            or not isinstance(normsrc,dict) or normal.get("source_before")!=normsrc):
        raise ValueError("paired/normal source maps are malformed")
    paths=tuple(sorted(set(oldsrc)|set(normsrc)|set(SOURCE_PATHS)|{"graphify-out/fv-root-cause-20260919/R2_POINT_PAIRED_FACE_NORMAL_INPUT_20261003.json"}))
    before={name:sha(ROOT/name) for name in paths}
    if any(before[k]!=v for m in (oldsrc,normsrc) for k,v in m.items()):
        raise ValueError("captured paired/normal sources differ from current source")
    pair_binding=FVPointPairedFaceBinding(problem); binding=pair_binding.first_binding
    seed=torch.tensor(paired["last_accepted_tangent"],dtype=torch.float64)
    saved_control=torch.tensor(paired["last_accepted_control"],dtype=torch.float64)
    if saved_control.shape!=(26,) or not torch.equal(pair_binding.lift(seed),saved_control):
        raise ValueError("paired tangent no longer lifts to its saved full control")
    first_control=pair_binding.second_chart.from_face_coordinates(insert_y_eta(seed))
    pair_binding.second_chart.to_face_coordinates(first_control)
    full_start=_strict_full(binding,first_control)
    qx,qy=binding.face_fluxes(first_control)
    if not torch.equal(qx[3,4],qx.new_zeros(())) or not bool(qy[3,0]>0):
        raise ReleaseRefusal("release seed must preserve qx zero and enter the positive qy sector")
    fixed_p=p.clone()
    def objective(c: Tensor, parameters: Tensor) -> Tensor:
        full=_strict_full(binding,c)
        return binding.reduced_problem.objective(c,parameters)+0.5*full[20].square()
    old_j=float(paired["final_objective"]); new_j=float(objective(first_control,p))
    comparison=objective_decrease(old_j,new_j)
    state: dict[str,Any]={"tangent":first_control.clone(),"signature":None}
    solves: list[dict[str,Any]]=[]; trials: list[dict[str,Any]]=[]; last=first_control.clone()
    report: dict[str,Any]={"pid":os.getpid(),"phase":"running","numerical_status":"running",
        "paired_report_sha256":PAIRED_SHA,"normal_report_sha256":NORMAL_SHA,"plan_sha256":PLAN_SHA,
        "source_before":before,"input_before":identity,"runtime":runtime,
        "normal_qy_right_exact_fraction":[str(normal_interval[0]),str(normal_interval[1])],
        "release_eta_y":ETA_Y,"initial_objective_comparison":comparison,
        "start_tangent":first_control.tolist(),"start_control":full_start.tolist(),
        "start_qx_zero":True,"start_qy_flux":float(qy[3,0]),"full_root_claim":False,
        "normal_validation":"prior qy right sign used only for sector choice","response_validation":"not_performed",
        "trials":trials,"linear_solves":solves}
    def save() -> None: _write(output,report)
    save()
    try:
        if not comparison["passed"]: raise ReleaseRefusal("positive-qy seed does not lower native objective beyond roundoff")
        start_branch,start_trace=_trace(binding,first_control,p)
        state["signature"]=start_branch["signature_sha256"]
        gradient=torch.func.grad(objective)(first_control,p)
        report.update(start_branch=start_branch,start_trace=start_trace,start_objective=new_j,
                      start_gradient=gradient.tolist(),start_gradient_max=float(gradient.abs().max()))
        report["start_curvature"]=single.curvature(objective,first_control,p)
        if not report["start_curvature"]["qualified"]:
            raise ReleaseRefusal("released start lacks numerical 25D SPD curvature")
        def branch_check(c: Tensor, parameters: Tensor):
            if not torch.equal(parameters,fixed_p): raise RuntimeError("refiner changed fixed parameters")
            branch,_=_trace(binding,c,parameters)
            if branch["signature_sha256"]!=state["signature"]:
                raise single.PointFaceRefusal("positive-qy single-face branch changed")
            state["tangent"]=c.detach().clone()
            return branch,"fixed qx zero and positive-qy branch; other faces/limiters strict"
        def observer(row: dict[str,Any]):
            nonlocal last
            trials.append(row)
            if row.get("accepted"): last=torch.tensor(row["candidate_control"],dtype=torch.float64)
            report["last_accepted_tangent"]=last.tolist(); save()
        with matrix_free.observe_pcg_calls(_true_residual_monitor(solves,0.0,state)):
            comparison,result=optimize_if_decreased(old_j,new_j,lambda: refine_stationary(objective,first_control,p,branch_check=branch_check,
                max_iterations=4,max_backtracks=16,pcg_max_iterations=104,trial_observer=observer)
            )
        report["initial_objective_comparison"]=comparison
        last=result.control.clone(); final_branch,final_trace=_trace(binding,last,p)
        final_gradient=torch.func.grad(objective)(last,p)
        report["final_curvature"]=single.curvature(objective,last,p)
        if not bool(torch.isfinite(final_gradient).all()) or float(final_gradient.abs().max())>=1e-10:
            raise ReleaseRefusal("released tangent gradient exceeds 1e-10")
        if not report["final_curvature"]["qualified"]:
            raise ReleaseRefusal("released final tangent lacks numerical 25D SPD curvature")
        report.update(numerical_status="tangent_stationary_candidate",tangent_gradient=final_gradient.tolist(),
            tangent_gradient_max=float(final_gradient.abs().max()),final_branch=final_branch,final_trace=final_trace,
            final_objective=float(objective(last,p)),newton_iterations=result.iterations,refiner_hvp_count=result.hvp_count)
    except (RefinementNumericalRefusal,single.PointFaceRefusal) as error:
        report.update(numerical_status="numerical_refusal",reason=str(error))
    single.audit_solves(objective,p,solves,state["signature"])
    after={name:sha(ROOT/name) for name in paths}
    if (after!=before or event.preflight._input_identity(problem,warm,p,direction)!=identity
            or not torch.equal(p,fixed_p) or sha(PAIRED)!=PAIRED_SHA or sha(NORMAL)!=NORMAL_SHA or sha(PLAN)!=PLAN_SHA or sha(NORMAL_INPUT)!=NORMAL_INPUT_SHA
            or {"python":platform.python_version(),"torch":torch.__version__,"device":"CPU FP64"}!=runtime):
        raise ValueError("release source/input/report/plan/runtime changed")
    try: report["last_accepted_control"]=binding.lift(last).tolist()
    except RuntimeError as error:
        if "strict representable chart domain" not in str(error) or report["numerical_status"]!="numerical_refusal": raise
        report["last_accepted_control"]=None
    report.update(phase="finished",source_after=after,input_after=identity,source_unchanged=True,
        input_unchanged=True,runtime_after=runtime,full_root_claim=False,response_validation="not_performed")
    save(); return report
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
