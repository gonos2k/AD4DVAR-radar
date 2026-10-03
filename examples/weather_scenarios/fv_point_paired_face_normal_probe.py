"""Audit one-sided interval normals at the pinned paired-face tangent point."""
from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import mpmath

from examples.weather_scenarios import fv_point_paired_face_normal_reference as reference

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
INPUT, PLAN, PAIRED = (EVIDENCE / "R2_POINT_PAIRED_FACE_NORMAL_INPUT_20261003.json",
                       EVIDENCE / "R2_POINT_PAIRED_FACE_NORMAL_PLAN_20261003.md",
                       EVIDENCE / "point_paired_face_stationarity_attempt1/stationarity.json")
INPUT_SHA = "dd9caee732c44ed0961cadd5265005ddfa95bbd5b1d61abab2a08ff6d24a4320"
PLAN_SHA = "db6e7b247b01c70eca72113e92b78fced46df3135500e8f52c06485286a2e9b2"
PAIRED_SHA = "d38d1cec4550a6347ceb14db23395679a59a7cad89f09096ec7185f984f6a8cf"
SOURCE_PATHS = ("examples/weather_scenarios/fv_point_paired_face_normal_reference.py", "examples/weather_scenarios/fv_active_face_normal_reference.py",
                "examples/weather_scenarios/fv_slice_precision_reference.py", "examples/weather_scenarios/fv_point_paired_face_normal_probe.py",
                "tests/test_fv_point_paired_face_normal_probe.py")
UnsupportedNormal = reference.NormalReferenceRefusal

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def _flat(value: Any) -> list[Any]:
    if not isinstance(value,list): raise ValueError("trace field must be a list")
    return [v for row in value for v in row] if value and isinstance(value[0],list) else value
def _choice(row: dict[str, Any], axis: str) -> list[int]:
    data = row.get(axis); 
    if not isinstance(data, dict): raise ValueError("native trace lacks limiter inputs")
    signs, left = _flat(data.get("slope_sign")), _flat(data.get("choose_left"))
    if len(signs)!=6 or len(left)!=6 or any(type(s)is not int or s not in (-1,0,1) for s in signs) or any(type(v)is not bool for v in left):
        raise ValueError("native limiter trace must have six signed choices and bool selectors")
    return [0 if s == 0 else (1 if l else 2) for s, l in zip(signs, left, strict=True)]

def audit_trace(native: list[dict[str, Any]], branch: list[dict[str, Any]], sides: tuple[int, int]) -> None:
    if len(native)!=36 or len(branch)!=36: raise ValueError("branch audit requires 36 analysis rows")
    for i, (src, row) in enumerate(zip(native, branch, strict=True)):
        if any(x.get(k)!=n for x in (src,row) for k,n in (("step",i//2),("stage",i%2))):
            raise ValueError("branch stage order malformed")
        if any(row.get(a)!=_choice(src,a) for a in ("x","y")):
            raise UnsupportedNormal("normal changed a native limiter-choice bin")
        qx, qy = _flat(src.get("qx_sign")), _flat(src.get("qy_sign"))
        rx, ry = row.get("qx_sign"), row.get("qy_sign")
        if not isinstance(rx,list) or not isinstance(ry,list) or (len(qx),len(qy),len(rx),len(ry))!=(24,25,24,25):
            raise ValueError("face-sign trace layout malformed")
        if any(type(v)is not int or v not in (-1,0,1) for v in qx+qy+rx+ry): raise ValueError("invalid face sign")
        if qx[22]!=0 or qy[15]!=0 or any(v==0 for j,v in enumerate(qx) if j!=22) or any(v==0 for j,v in enumerate(qy) if j!=15):
            raise ValueError("native trace must have only the selected zero faces")
        if (any(qx[j]!=rx[j] for j in range(24) if j!=22) or any(qy[j]!=ry[j] for j in range(25) if j!=15)
                or rx[22]!=sides[0] or ry[15]!=sides[1]):
            raise UnsupportedNormal("face signs differ from native other faces or requested orthant")

def _point(raw: Any) -> Fraction:
    if not isinstance(raw,list) or len(raw)!=4 or any(type(v)is not int for v in raw): raise ValueError("binary endpoint must be an mpmath tuple")
    sign,mantissa,exponent,bits=raw
    if sign not in (0,1) or mantissa<0 or bits<0 or mantissa.bit_length()!=bits:
        raise ValueError("binary endpoint malformed or nonfinite")
    return Fraction(-mantissa if sign else mantissa) * 2**exponent if exponent >= 0 else Fraction(-mantissa if sign else mantissa) / 2**-exponent
def _iv(raw: Any, name: str) -> tuple[Fraction, Fraction, list[int], list[int]]:
    if not isinstance(raw,dict) or set(raw)!={"lower","upper"}: raise ValueError(f"missing binary interval for {name}")
    lo, hi = _point(raw["lower"]), _point(raw["upper"])
    if lo>hi: raise ValueError(f"reversed interval for {name}")
    return lo, hi, raw["lower"], raw["upper"]

def interval(row: dict[str, Any], name: str):
    keys=("objective_mpi","J_binary") if name=="objective" else ("sigma_eta_mpi","sigma_eta_binary")
    return _iv(next((row[k] for k in keys if k in row),None),name)
def intersection(rows: list[tuple[Fraction, Fraction, list[int], list[int]]]):
    if not rows: return None
    lo, hi = max(rows, key=lambda r: r[0]), min(rows, key=lambda r: r[1])
    return (lo[0], hi[1], lo[2], hi[3]) if lo[0] <= hi[1] else None

def strict_orientation(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> bool: return left[1] < 0 < right[0]
def audit_evaluation_set(rows: list[dict[str, Any]]) -> None:
    expected={(a,x,y) for a in (0,1) for x in (-1,1) for y in (-1,1)}; keys=[]
    for r in rows:
        a,s=r.get("normal_axis"),r.get("requested_sides")
        if type(a)is not int or a not in (0,1) or not isinstance(s,list) or len(s)!=2 or any(type(v)is not int or v not in (-1,1) for v in s) or r.get("sector_sides")!=s:
            raise ValueError("normal evaluation has unknown axis or malformed sector metadata")
        keys.append((a,*s))
    if len(keys)!=8 or len(set(keys))!=8 or set(keys)!=expected:
        raise ValueError("normal evaluations must cover each axis/orthant once")
def audit_fluxes(row: dict[str, Any], axis: int, scales: list[float]) -> None:
    vals, derivs = row.get("selected_flux_values_binary"), row.get("selected_flux_derivatives_binary")
    if not isinstance(vals,list) or not isinstance(derivs,list) or len(vals)!=2 or len(derivs)!=2: raise ValueError("reference omitted selected-flux intervals")
    flux = [_iv(v, "selected flux")[:2] for v in vals]
    rate = [_iv(v, "selected derivative")[:2] for v in derivs]
    if any(v!=(Fraction(0),Fraction(0)) for v in flux): raise ValueError("selected flux is not exact zero")
    active = Fraction.from_float(float(scales[axis]))
    if not rate[axis][0]<=active<=rate[axis][1] or rate[1-axis]!=(Fraction(0),Fraction(0)):
        raise ValueError("selected derivative differs from normal scale/inactive zero")
def primal_crosscheck(native: float, row: dict[str, Any]) -> dict[str, Any]:
    """Compare native J using a term-scaled FP64 gamma budget, without a unit floor."""
    lo,hi,*_ = interval(row,"objective"); j=Fraction.from_float(native)
    delta=Fraction(0) if lo<=j<=hi else min(abs(j-lo),abs(j-hi)); costs=[]
    for key in ("robust_cost_bounds", "prior_cost_bounds", "smooth_cost_bounds"):
        raw = row.get(key)
        if not isinstance(raw,list) or len(raw)!=2: raise ValueError(f"reference omitted {key}")
        bounds=[Fraction(Decimal(v)) for v in raw]
        if bounds[0]<0 or bounds[1]<bounds[0]: raise ValueError(f"{key} must bound a nonnegative term")
        costs.append(bounds[1])
    scale=abs(j)+max(abs(lo),abs(hi))+sum(costs); n=36*20*128+3*4*4+2*26+4*4*8+31*8
    ne=n*Fraction(1,2**52); budget=ne/(1-ne)*scale
    return {"passed":delta<=budget,"delta":float(delta),"budget":float(budget),"scale":float(scale),
            "operations":n,"delta_exact":str(delta),"budget_exact":str(budget)}

def grouped_normals(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    summaries,refusal=[],None
    for axis in (0,1):
        sides={}
        for side in (-1,1):
            sector=[r for r in rows if r["normal_axis"]==axis and r["requested_sides"][axis]==side]
            if len(sector)!=2 or {r["requested_sides"][1-axis] for r in sector}!={-1,1}:
                raise ValueError("normal side lacks unique evaluations from both other sectors")
            iv=intersection([interval(r,"sigma") for r in sector])
            if iv is None: refusal=refusal or f"axis {axis} derivative intervals disagree across sectors"; continue
            sides[side]=iv[:2]
            summaries.append({"normal_axis":axis,"side":side,"binary":{"lower":iv[2],"upper":iv[3]},
                              "exact_fraction_bounds":[str(iv[0]),str(iv[1])],"source_evaluations":[r["sector_sides"] for r in sector]})
        if len(sides)==2 and not strict_orientation(sides[-1],sides[1]):
            refusal=refusal or f"axis {axis} lacks strict left-negative/right-positive intervals"
    return summaries, refusal

def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n"); temp.replace(path)

def run(output: Path) -> dict[str, Any]:
    if output.exists() or sha(INPUT)!=INPUT_SHA or sha(PLAN)!=PLAN_SHA or sha(PAIRED)!=PAIRED_SHA:
        raise ValueError("normal output must be fresh and input/report/plan pins unchanged")
    data, paired = json.loads(INPUT.read_text()), json.loads(PAIRED.read_text())
    sources = paired.get("source_after")
    if (not isinstance(sources,dict) or paired.get("source_before")!=sources or paired.get("input_before")!=paired.get("input_after")
            or data.get("input_identity")!=paired.get("input_before") or data.get("parameters_sha256")!=paired.get("parameters_sha256")
            or data.get("paired_report_sha256")!=PAIRED_SHA or paired.get("phase")!="finished"
            or paired.get("numerical_status")!="tangent_stationary_candidate" or paired.get("full_root_claim") is not False
            or paired.get("normal_validation")!="not_performed" or paired.get("response_validation")!="not_performed"
            or paired.get("selected_faces")!=[["x",3,4],["y",3,0]] or paired.get("normal_coordinates")!=[0.,0.]
            or data.get("tangent")!=paired.get("last_accepted_tangent")
            or data.get("native_analysis_trace")!=paired.get("final_trace",[])[:36]
            or data.get("native_objective")!=paired.get("final_objective")):
        raise ValueError("normal capture does not match the finished paired tangent report")
    runtime = {"python":platform.python_version(),"mpmath":mpmath.__version__}
    if runtime["python"]!=paired.get("runtime",{}).get("python"):
        raise ValueError("pure runtime Python differs from paired report provenance")
    reference.validate_projection(data["fixture"])
    paths = tuple(sorted(set(sources)|set(SOURCE_PATHS)))
    before = {name:sha(ROOT/name) for name in paths}
    if any(before[k]!=v for k,v in sources.items()):
        raise ValueError("paired producer sources differ from the captured root")
    rows, refusal = [], None
    try:
        for sx in (-1,1):
            for sy in (-1,1):
                for axis in (0,1):
                    row = reference.evaluate(data["fixture"],data["tangent"],axis,(sx,sy),dps=80)
                    if row.get("normal_axis")!=axis or row.get("sector_sides")!=[sx,sy]:
                        raise ValueError("reference returned mismatched sector metadata")
                    audit_trace(data["native_analysis_trace"],row.get("branch_choices",[]),(sx,sy))
                    audit_fluxes(row,axis,data["fixture"]["normal_scales"])
                    cmp = primal_crosscheck(data["native_objective"],row)
                    if not cmp["passed"]:
                        raise ValueError("independent J differs from native beyond its FP64 term-scale budget")
                    row.update(requested_sides=[sx,sy],native_objective_crosscheck=cmp)
                    rows.append(row)
    except UnsupportedNormal as error:
        refusal = str(error)
    if rows and refusal is None:
        audit_evaluation_set(rows)
    jout = intersection([interval(r,"objective") for r in rows]) if rows else None
    if rows and jout is None:
        refusal = refusal or "objective intervals across paired orthants do not overlap"
    normals = []
    if rows and refusal is None:
        normals, refusal = grouped_normals(rows)
    after = {name:sha(ROOT/name) for name in paths}
    if (after!=before or sha(INPUT)!=INPUT_SHA or sha(PLAN)!=PLAN_SHA or sha(PAIRED)!=PAIRED_SHA
            or platform.python_version()!=runtime["python"] or mpmath.__version__!=runtime["mpmath"]):
        raise ValueError("normal source/input/report/runtime changed during evaluation")
    payload = {"phase":"finished","numerical_status":"unsupported_normal" if refusal else "oriented_normal_intervals",
               "reason":refusal,"input_sha256":INPUT_SHA,"paired_report_sha256":PAIRED_SHA,"plan_sha256":PLAN_SHA,
               "source_before":before,"source_after":after,"input_identity_before":data["input_identity"],
               "input_identity_after":data["input_identity"],"runtime":runtime,"runtime_after":runtime,
               "normal_evaluations":rows,"objective_overlap_binary":None if jout is None else {"lower":jout[2],"upper":jout[3]},
               "normal_intervals":normals,"full_root_claim":False,"normal_minimum_claim":False,
               "response_validation":"not_performed","physical_validated":False}
    _write(output,payload); return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
