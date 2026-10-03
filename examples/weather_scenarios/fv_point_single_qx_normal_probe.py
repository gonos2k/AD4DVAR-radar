"""Independent binary-interval one-sided normal audit on the released qx-zero state."""
from __future__ import annotations

import argparse, hashlib, json, platform
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath

from examples.weather_scenarios import fv_point_paired_face_normal_probe as audit
from examples.weather_scenarios import fv_point_single_face_normal_reference as reference

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"graphify-out/fv-root-cause-20260919"
INPUT=OUT/"R2_POINT_SINGLE_QX_NORMAL_INPUT_20261003_RELEASE2.json"
INPUT_SHA="dd8519440ac989944e787e9365115b210e3f1aa22a19048f0f8113990d6313e4"
RELEASE=OUT/"point_qy_release_attempt2/release.json"
RELEASE_SHA="e3f4f083ac9bfb002d38e4e2a0352672f179775d18b22fbf26fd294bedd2ad24"
PAIRED=OUT/"point_paired_face_stationarity_attempt1/stationarity.json"
PAIRED_SHA="d38d1cec4550a6347ceb14db23395679a59a7cad89f09096ec7185f984f6a8cf"
PLAN=OUT/"R2_POINT_SINGLE_QX_NORMAL_PLAN_20261003.md"
PLAN_SHA="ed1d3b60305e20ea9b9c4b72185bac33c178675db2bf0bb8d1f34d58a8c09b71"
SOURCE_PATHS=("examples/weather_scenarios/fv_point_single_face_normal_reference.py",
              "examples/weather_scenarios/fv_point_single_qx_normal_probe.py",
              "examples/weather_scenarios/fv_point_paired_face_normal_probe.py",
              "examples/weather_scenarios/fv_point_paired_face_normal_reference.py",
              "examples/weather_scenarios/fv_active_face_normal_reference.py",
              "examples/weather_scenarios/fv_slice_precision_reference.py",
              "tests/test_fv_point_single_qx_normal_probe.py")
Refusal=reference.SingleFaceNormalRefusal


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flat(value: Any) -> list[Any]:
    if not isinstance(value,list): raise ValueError("native trace field must be nested list")
    return [x for row in value for x in row] if value and isinstance(value[0],list) else value


def _choice(row: dict[str, Any], axis: str) -> list[int]:
    data=row.get(axis)
    if not isinstance(data,dict): raise ValueError("native limiter record is malformed")
    signs,left=_flat(data.get("slope_sign")),_flat(data.get("choose_left"))
    if len(signs)!=6 or len(left)!=6 or any(type(s)is not int or s not in (-1,0,1) for s in signs) or any(type(v)is not bool for v in left):
        raise ValueError("native limiter record has malformed 2x3 choices")
    return [0 if s==0 else (1 if choose else 2) for s,choose in zip(signs,left,strict=True)]


def audit_trace(native: list[dict[str, Any]], branch: list[dict[str, Any]], side: int) -> None:
    if len(native)!=36 or len(branch)!=36: raise ValueError("qx normal audit requires 36 native analysis rows")
    for i,(src,row) in enumerate(zip(native,branch,strict=True)):
        if any(x.get(k)!=v for x in (src,row) for k,v in (("step",i//2),("stage",i%2))):
            raise ValueError("single-qx stage order differs from native trace")
        if any(row.get(axis)!=_choice(src,axis) for axis in ("x","y")):
            raise Refusal("single-qx normal changed a native limiter-choice bin")
        qx,qy=_flat(src.get("qx_sign")),_flat(src.get("qy_sign")); rx,ry=row.get("qx_sign"),row.get("qy_sign")
        if not isinstance(rx,list) or not isinstance(ry,list) or (len(qx),len(qy),len(rx),len(ry))!=(24,25,24,25):
            raise ValueError("single-qx sign arrays have malformed grid shape")
        if any(type(v)is not int or v not in (-1,0,1) for v in qx+qy+rx+ry):
            raise ValueError("face sign must be -1, 0, or +1")
        if qx[22]!=0 or any(v==0 for j,v in enumerate(qx) if j!=22) or any(v==0 for v in qy):
            raise ValueError("native stage must have exactly the selected qx zero")
        if any(qx[j]!=rx[j] for j in range(24) if j!=22) or qy!=ry:
            raise Refusal("single-qx normal changed one of 48 other face signs")
        if rx[22]!=side: raise Refusal("selected qx sign does not match requested side")


def audit_flux(row: dict[str, Any]) -> None:
    values,rates=row.get("selected_flux_values_binary"),row.get("selected_flux_derivatives_binary")
    if not isinstance(values,list) or not isinstance(rates,list) or len(values)!=2 or len(rates)!=2:
        raise ValueError("reference omitted selected-flux intervals")
    qx,qy=(audit._iv(v,"flux")[:2] for v in values)
    dx,dy=(audit._iv(v,"derivative")[:2] for v in rates)
    if qx!=(Fraction(0),Fraction(0)) or not dx[0]<=Fraction.from_float(.11)<=dx[1] or dy!=(Fraction(0),Fraction(0)):
        raise ValueError("qx event primal/normal derivative intervals violate the pinned face contract")


def verify_capture(data: dict[str, Any], release: dict[str, Any], paired: dict[str, Any]) -> None:
    if (data.get("release_report_sha256")!=RELEASE_SHA or data.get("paired_report_sha256")!=PAIRED_SHA
            or data.get("release_source_sha256")!=release.get("source_after")
            or release.get("last_accepted_tangent")!=data.get("tangent")
            or release.get("final_objective")!=data.get("native_objective")
            or release.get("final_branch",{}).get("signature_sha256")!=data.get("native_signature_sha256")
            or release.get("input_before")!=data.get("input_identity")
            or release.get("input_before")!=paired.get("input_after")
            or paired.get("input_before")!=paired.get("input_after")
            or data.get("input_identity",{}).get("tensor_sha256",{}).get("parameters")!=data.get("parameters_sha256")
            or set(data.get("source_sha256",{}))!=set(data.get("release_source_sha256",{}))
            or data.get("parameters_sha256")!=release.get("input_before",{}).get("tensor_sha256",{}).get("parameters")
            or release.get("final_trace",[])[:36]!=data.get("native_analysis_trace")):
        raise ValueError("single-qx input capture differs from the released native point")


def evaluate_side(fixture: dict[str, Any], tangent: list[float|str], side: int) -> dict[str, Any]:
    return reference.evaluate(fixture,tangent,side,dps=80)


def _write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"); temp.replace(path)


def run(output: Path) -> dict[str, Any]:
    if output.exists() or sha(INPUT)!=INPUT_SHA or sha(RELEASE)!=RELEASE_SHA or sha(PAIRED)!=PAIRED_SHA or sha(PLAN)!=PLAN_SHA:
        raise ValueError("normal output must be fresh and captured input/release/plan pins unchanged")
    data,release,paired=(json.loads(p.read_text()) for p in (INPUT,RELEASE,PAIRED))
    if (paired.get("phase")!="finished" or paired.get("numerical_status")!="tangent_stationary_candidate"
            or paired.get("full_root_claim") is not False or paired.get("response_validation")!="not_performed"
            or release.get("paired_report_sha256")!=PAIRED_SHA
            or release.get("phase")!="finished" or release.get("numerical_status")!="tangent_stationary_candidate"
            or release.get("full_root_claim") is not False or release.get("response_validation")!="not_performed"
            or paired.get("full_root_claim") is not False or data.get("source_sha256") is None):
        raise ValueError("released root/report provenance is not the declared normal input")
    verify_capture(data,release,paired)
    if (data.get("source_drift_since_release")!={k:{"released_sha256":release.get("source_after",{}).get(k),"current_sha256":v}
            for k,v in data.get("source_sha256",{}).items() if release.get("source_after",{}).get(k)!=v}):
        raise ValueError("release-to-capture source drift record is inconsistent")
    runtime={"python":platform.python_version(),"mpmath":mpmath.__version__}
    if runtime["mpmath"]!="1.3.0":
        raise ValueError("single-qx interval reference requires pinned mpmath 1.3.0")
    if runtime["python"]!=data.get("runtime",{}).get("python"):
        raise ValueError("pure qx reference Python differs from native capture")
    reference.validate_projection(data["fixture"])
    paths=tuple(sorted(set(data["source_sha256"])|set(SOURCE_PATHS)|{"graphify-out/fv-root-cause-20260919/R2_POINT_SINGLE_QX_NORMAL_INPUT_20261003.json"}))
    before={name:sha(ROOT/name) for name in paths}
    if any(before[k]!=v for k,v in data["source_sha256"].items()): raise ValueError("captured native source map changed")
    rows=[]; refusal=None
    try:
        for side in (-1,1):
            row=evaluate_side(data["fixture"],data["tangent"],side)
            if row.get("side")!=side: raise ValueError("single-qx reference returned mismatched side")
            audit_trace(data["native_analysis_trace"],row.get("branch_choices",[]),side); audit_flux(row)
            comparison=audit.primal_crosscheck(data["native_objective"],row)
            if not comparison["passed"]: raise ValueError("interval J differs from native beyond its term-scale budget")
            row["native_objective_crosscheck"]=comparison; rows.append(row)
    except Refusal as error:
        refusal=str(error)
    jout=audit.intersection([audit.interval(r,"objective") for r in rows]) if rows else None
    if rows and jout is None: refusal=refusal or "single-qx primal J intervals do not overlap"
    orientation=None
    if len(rows)==2 and refusal is None:
        left,right=(audit.interval(r,"sigma")[:2] for r in rows)
        orientation=reference.strict_normal_orientation(left,right)
        if not orientation: refusal="single-qx normal lacks strict left-negative/right-positive signs"
    after={name:sha(ROOT/name) for name in paths}
    if (after!=before or sha(INPUT)!=INPUT_SHA or sha(RELEASE)!=RELEASE_SHA or sha(PAIRED)!=PAIRED_SHA or sha(PLAN)!=PLAN_SHA
            or platform.python_version()!=runtime["python"] or mpmath.__version__!=runtime["mpmath"]):
        raise ValueError("single-qx source/input/report/plan/runtime changed during evaluation")
    result={"phase":"finished","numerical_status":"unsupported_normal" if refusal else "oriented_single_qx_normal",
        "reason":refusal,"input_sha256":INPUT_SHA,"release_report_sha256":RELEASE_SHA,"paired_report_sha256":PAIRED_SHA,
        "plan_sha256":PLAN_SHA,"source_before":before,"source_after":after,"input_identity_before":data["input_identity"],
        "input_identity_after":data["input_identity"],"runtime":runtime,"runtime_after":runtime,"evaluations":rows,
        "objective_overlap_binary":None if jout is None else {"lower":jout[2],"upper":jout[3]},
        "strict_single_qx_orientation":orientation,"full_root_claim":False,"normal_minimum_claim":False,
        "positive_parameter_ball_claim":False,"response_validation":"not_performed","physical_validated":False}
    _write(output,result); return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
