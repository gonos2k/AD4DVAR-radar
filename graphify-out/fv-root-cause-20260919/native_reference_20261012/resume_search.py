"""Continue the frozen two-start Powell path using the support-only source fix."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
OLD_PLAN = WORK / "PLAN.json"
OLD_PLAN_SHA = "b6f3a08e02ec6da684ea8ea7c9611e6eedf3b8de9de8a9ebc6830347dd13d9ac"
OLD_PLAN_SHA_FILE = WORK / "PLAN_SHA256.txt"
OLD_ATTEMPT = WORK / "attempt1"
REPLAY_RECEIPT = WORK / "zero_powell_replay.json"
SUPPORT_FAILURE = WORK / "zero_powell_support_failure.json"
SUPPORT_DIAGNOSIS = WORK / "zero_powell_support_diagnosis.json"
SIGNED = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012"
CONT = WORK / "continuation3"
PLAN = CONT / "CONTINUATION_PLAN.json"
PLAN_SHA_FILE = CONT / "CONTINUATION_PLAN_SHA256.txt"
ATTEMPT = CONT / "attempt1"
CONT1 = WORK / "continuation1"
CONT2 = WORK / "continuation2"
SELF = "graphify-out/fv-root-cause-20260919/native_reference_20261012/resume_search.py"
REPLAY = "graphify-out/fv-root-cause-20260919/native_reference_20261012/replay_zero_powell.py"
SEARCH_DRIVER = "graphify-out/fv-root-cause-20260919/native_reference_20261012/reference_search.py"
CASE = ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"
MAX_TOTAL_CALLS = 2000
OLD_SUCCESSFUL_CALLS = 1699
OLD_SEARCH_SECONDS = 355.31765737500973
FAILED_PREFLIGHT_SECONDS = 3.443094667047262
OLD_FB0A_CALLS = 1000
OLD_ZERO_CALLS = 699
FB0A_SHA = "fb0a2efaeefe3a8b07a9385a81260757e4923b181a9430c384cfb96b8d8ea7b4"
FB0A_J = 0.060988878373729814
ZERO_SHA = "46f531b7ea0428fbf2c3ca2b60e8dc33d6bbfa000e0fd1b489c5e39140a47006"
FAILED_ZERO_SHA = "5556eec11e69aebb4a6e76a7b2e33787c76cf6145ad87eec7dd201ec3f5b9e62"

sys.path.insert(0, str(ROOT))
from examples.weather_scenarios import fv_point_3h_netflux_future_observation as obs  # noqa: E402
from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle  # noqa: E402


class NativeBudgetReached(Exception):
    """The original cumulative native-objective call allowance is exhausted."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"continuation pin is not an in-repository regular file: {path}")
    return _sha_bytes(path.read_bytes())


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _tensor_sha(value: torch.Tensor) -> str:
    if value.dtype != torch.float64 or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
        raise ValueError("continuation controls must be finite CPU FP64")
    return _sha_bytes(value.detach().contiguous().numpy().tobytes())


def _load_case_and_controls() -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if _sha(OLD_PLAN) != OLD_PLAN_SHA or OLD_PLAN_SHA_FILE.read_text().strip() != OLD_PLAN_SHA:
        raise ValueError("original native Powell plan changed")
    if _sha(SIGNED / "PLAN.json") != "707bf06bf0432a3f46b6fcc8901908186d70336e075ea5dc72b3cf4658bcfe9b":
        raise ValueError("signed 2a4d input plan changed")
    step = _read(SIGNED / "attempt1/step.json")
    arms = {arm["name"]: arm for arm in step["arms"]}
    fb = torch.as_tensor(arms["baseline_tangent"]["trials"][0]["control"], dtype=torch.float64)
    zero = torch.zeros_like(fb)
    if _tensor_sha(fb) != FB0A_SHA or _tensor_sha(zero) != ZERO_SHA:
        raise ValueError("native Powell starting controls differ from frozen fb0a/zero")
    case = obs._load_case()
    problem, original, parameters, truth = obs._fixed_case_tensors(case, {
        "parameters_sha256": obs.PARAMETERS_SHA,"truth_sha256": obs.TRUTH_SHA})
    return problem, original, parameters, truth, {"fb0a": fb,"zero": zero}


def _archive_inputs() -> dict[str, str]:
    old_evals = OLD_ATTEMPT / "evaluations.jsonl"
    paths = {
        OLD_PLAN,
        OLD_PLAN_SHA_FILE,
        SIGNED / "source.tar.gz",
        OLD_ATTEMPT / "checkpoint.json",
        old_evals,
        OLD_ATTEMPT / "run.json",
        OLD_ATTEMPT / "resource.json",
        OLD_ATTEMPT / "search.log",
        OLD_ATTEMPT / "fb0a.candidate_checkpoint.json",
        OLD_ATTEMPT / "zero.candidate_checkpoint.json",
        REPLAY_RECEIPT,
        SUPPORT_FAILURE,
        SUPPORT_DIAGNOSIS,
        CONT1 / "CONTINUATION_PLAN.json",
        CONT1 / "CONTINUATION_PLAN_SHA256.txt",
        CONT1 / "source.tar.gz",
        CONT1 / "attempt1/run.json",
        CONT1 / "attempt1/resource.json",
        CONT1 / "attempt1/search.log",
        CONT2 / "CONTINUATION_PLAN.json",
        CONT2 / "CONTINUATION_PLAN_SHA256.txt",
        CONT2 / "source.tar.gz",
        CASE,
        CASE.with_name("FIXED_CASE_MANIFEST.json"),
        SIGNED / "PLAN.json",
        SIGNED / "PLAN_SHA256.txt",
        SIGNED / "source.tar.gz",
        SIGNED / "attempt1/step.json",
        SIGNED / "attempt1/step.run.json",
        SIGNED / "attempt1/step.resource.json",
        SIGNED / "attempt1/baseline_tangent/candidate_01.json",
    }
    return {path.relative_to(ROOT).as_posix(): _sha(path) for path in sorted(paths)}


def freeze_plan() -> dict[str, Any]:
    if PLAN.exists() or PLAN_SHA_FILE.exists():
        raise ValueError("native continuation plan already frozen")
    old = _read(OLD_PLAN)
    source_files = dict(old["source_files"])
    old_transport = source_files["src/advar/transport.py"]
    new_transport = _sha(ROOT / "src/advar/transport.py")
    if old_transport == new_transport:
        raise ValueError("expected the documented support-only transport transition")
    transition = {"src/advar/transport.py": {"before": old_transport,"after": new_transport}}
    source_files["src/advar/transport.py"] = new_transport
    for name, digest in source_files.items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"continuation source scope drift: {name}")
    new_scripts = {SEARCH_DRIVER: _sha(ROOT / SEARCH_DRIVER),
        REPLAY: _sha(ROOT / REPLAY), SELF: _sha(ROOT / SELF)}
    extra_archive = _archive_inputs()
    inherited_archives = dict(old["archive_files"])
    for name, digest in inherited_archives.items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"inherited archive changed: {name}")
    previous_run = _read(OLD_ATTEMPT / "run.json")
    previous_resource = previous_run["resource"]
    previous_checkpoint = _read(OLD_ATTEMPT / "checkpoint.json")
    rows = [json.loads(line) for line in (OLD_ATTEMPT / "evaluations.jsonl").read_text().splitlines()]
    previous_fb = [row for row in rows if row["start"] == "fb0a"]
    previous_zero = [row for row in rows if row["start"] == "zero"]
    if (len(previous_fb) != 1001 or len(previous_zero) != 700
            or previous_checkpoint["starts"]["fb0a"]["objective_evaluations"] != 1000
            or previous_checkpoint["starts"]["zero"]["objective_evaluations"] != 699
            or previous_resource["elapsed_seconds"] != OLD_SEARCH_SECONDS):
        raise ValueError("old Powell counters/runtime differ from the recorded failed run")
    failed = _read(REPLAY_RECEIPT)
    failure = _read(SUPPORT_FAILURE)
    diagnosis = _read(SUPPORT_DIAGNOSIS)
    if (failed.get("status") != "first_unseen_reproduced"
            or failed.get("failed_control_sha256") != FAILED_ZERO_SHA
            or failure.get("status") != "native_support_refusal_reproduced"
            or diagnosis.get("status") != "support_roundoff_cause_identified"):
        raise ValueError("cached Powell failure replay/physical support diagnosis changed")
    plan = {
        "experiment_kind": "native_J_powell_continuation_after_material_support_roundoff_fix",
        "source_files": source_files,
        "extra_source_files": new_scripts,
        "observer_source_sha256": new_scripts[SELF],
        "source_transition": transition,
        "parent_native_plan_sha256": OLD_PLAN_SHA,
        "parent_source_bundle_sha256": _sha(SIGNED / "source.tar.gz"),
        "parent_source_count": len(old["source_files"]),
        "parent_archive_count": len(old["archive_files"]),
        "source_tar_base": "original native plan sources with only transport.py updated",
        "source_bundle_sha256": None,
        "archive_files": {**inherited_archives, **extra_archive},
        "archive_count_inherited": len(inherited_archives),
        "points": old["points"],
        "parameters_sha256": old["parameters_sha256"],
        "terminal_truth_sha256": old["terminal_truth_sha256"],
        "fixed_problem_identity": old["fixed_problem_identity"],
        "runtime_identity": old["runtime_identity"],
        "old_search": {"elapsed_seconds": OLD_SEARCH_SECONDS,
            "successful_native_J_evaluations": OLD_SUCCESSFUL_CALLS,
            "fb0a_successful_evaluations": OLD_FB0A_CALLS,
            "zero_successful_evaluations": OLD_ZERO_CALLS,
            "zero_first_unseen_callback": failed["first_unseen_powell_callback"],
            "zero_failed_attempt_control_sha256": FAILED_ZERO_SHA},
        "continuation": {"sciPy_version": failed["sciPy_version"],
            "zero_start_replay": "same SciPy Powell from original zero, feeding exact old float64 controls/J cache through callback 699",
            "continue_from_callback": 700,
            "remaining_zero_start_successful_evaluations_before_equivalence_probes": 301,
            "remaining_total_successful_native_J_call_budget": 301,
            "native_J_equivalence_probe_budget": 3,
            "powell_original_maxfev_per_start": 1000,
            "powell_original_maxiter_per_start": 100,
            "optimizer_options": old["optimizer"],
            "equivalence_probes": ["fb0a baseline J","zero start J","saved zero best-checkpoint J"],
            "source_fix_scope": "only material-support donor-cell algebra; echo minmod and default donor-cell arithmetic unchanged; strict support tolerance retained"},
        "cumulative_budget": {"max_successful_native_J_evaluations": MAX_TOTAL_CALLS,
            "previous_successful_native_J_evaluations": OLD_SUCCESSFUL_CALLS,
            "remaining_successful_native_J_evaluations": MAX_TOTAL_CALLS-OLD_SUCCESSFUL_CALLS,
            "budget_after_equivalence_probes": MAX_TOTAL_CALLS-OLD_SUCCESSFUL_CALLS-3,
            "previous_failed_support_objective_attempts_not_completed": 1,
            "separate_failed_support_reproduction_attempts": 1,
            "previous_search_seconds": OLD_SEARCH_SECONDS,
            "previous_failed_preflight_seconds": FAILED_PREFLIGHT_SECONDS,
            "original_search_internal_cap_seconds": 3000.0,
            "remaining_original_search_internal_seconds": 3000.0-OLD_SEARCH_SECONDS-FAILED_PREFLIGHT_SECONDS,
            "total_original_search_wall_cap_seconds": 3600.0,
            "remaining_search_outer_seconds": 3600.0-OLD_SEARCH_SECONDS-FAILED_PREFLIGHT_SECONDS,
            "rss_bytes": 2*1024**3,"no_new_budget_grant": True},
        "output_contract": {"Q_calls": 0,"R_calls": 0,"E_calls": 0,"HVP_calls": 0,
            "HVP_status_after_failed_partial_forward": "unknown_aborted_partial_native_attempt; no completed HVP",
            "cross_start_E_selection": False},
    }
    # Freeze a new source tar with the updated core and both continuation/replay drivers.
    import io
    bundle_files = {**source_files, **new_scripts}
    bundle_path = CONT / "source.tar.gz"
    CONT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path,"w:gz") as archive:
        for name in sorted(bundle_files):
            data=(ROOT/name).read_bytes()
            info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    plan["source_bundle_sha256"]=_sha(bundle_path)
    plan["source_member_count"]=len(bundle_files)
    plan["archive_files"].update({bundle_path.relative_to(ROOT).as_posix():_sha(bundle_path)})
    plan["archive_count_new"]=len(plan["archive_files"])-len(inherited_archives)
    plan["archive_files"]={k:plan["archive_files"][k] for k in sorted(plan["archive_files"])}
    PLAN.write_text(json.dumps(plan,indent=2,sort_keys=True,allow_nan=False)+"\n")
    digest=_sha(PLAN);PLAN_SHA_FILE.write_text(digest+"\n")
    return {"plan_sha256":digest,"source_members":len(bundle_files),
        "archive_count":len(plan["archive_files"]),"source_bundle_sha256":plan["source_bundle_sha256"],
        "source_transition":transition,"remaining_objective_calls":301,
        "remaining_search_seconds":3000.0-OLD_SEARCH_SECONDS}


def load_plan(digest: str) -> dict[str,Any]:
    if _sha(PLAN)!=digest or PLAN_SHA_FILE.read_text().strip()!=digest:
        raise ValueError("continuation plan seal mismatch")
    plan=_read(PLAN)
    if (plan.get("observer_source_sha256") not in (None,_sha(ROOT/SELF))
            or plan.get("parent_native_plan_sha256")!=OLD_PLAN_SHA
            or plan.get("source_transition",{}).get("src/advar/transport.py",{}).get("after")!=_sha(ROOT/"src/advar/transport.py")
            or plan.get("max_successful_native_J_evaluations") not in (None,MAX_TOTAL_CALLS)):
        raise ValueError("continuation source/budget identity mismatch")
    for name,digest_value in plan["source_files"].items():
        if _sha(ROOT/name)!=digest_value:raise ValueError(f"continuation source hash mismatch: {name}")
    for name,digest_value in plan["extra_source_files"].items():
        if _sha(ROOT/name)!=digest_value:raise ValueError(f"continuation driver hash mismatch: {name}")
    for name,digest_value in plan["archive_files"].items():
        if _sha(ROOT/name)!=digest_value:raise ValueError(f"continuation archive hash mismatch: {name}")
    import tarfile
    with tarfile.open(CONT/"source.tar.gz","r:gz") as archive:
        members={member.name:archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    bundle={**plan["source_files"],**plan["extra_source_files"]}
    if set(members)!=set(bundle) or any(_sha_bytes(members[name])!=value for name,value in bundle.items()):
        raise ValueError("continuation source tar member/hash mismatch")
    if _sha(CONT/"source.tar.gz")!=plan["source_bundle_sha256"]:
        raise ValueError("continuation source bundle digest mismatch")
    return plan


def _write(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temp.replace(path)


def _run(digest:str)->dict[str,Any]:
    from scipy.optimize import minimize
    import scipy
    plan=load_plan(digest)
    problem, original, parameters, truth, points=_load_case_and_controls()
    source_before={name:_sha(ROOT/name) for name in plan["source_files"]}
    input_before={name:cycle.shared._input_identity(problem,original,point,parameters,truth)
        for name,point in points.items()}
    old_rows=[json.loads(line) for line in (OLD_ATTEMPT/"evaluations.jsonl").read_text().splitlines()]
    zero_rows=[row for row in old_rows if row["start"]=="zero"]
    cache:dict[bytes,float]={}
    zero_best=None
    zero_best_cost=math.inf
    for row in zero_rows:
        key=torch.as_tensor(row["control"],dtype=torch.float64).numpy().tobytes()
        value=float(row["J"])
        if key in cache and cache[key]!=value:raise ValueError("old duplicate point cost differs")
        cache[key]=value
        if value<zero_best_cost:
            zero_best_cost=value
            zero_best=(key,value)
    failed=json.loads(REPLAY_RECEIPT.read_text())
    failed_key=torch.as_tensor(failed["failed_control"],dtype=torch.float64).numpy().tobytes()
    if failed_key in cache:raise ValueError("failed point unexpectedly occurs in old successful cache")
    if _tensor_sha(torch.as_tensor(failed["failed_control"],dtype=torch.float64))!=FAILED_ZERO_SHA:
        raise ValueError("failed point hash differs from cached replay")
    fb_expected=next(float(row["J"]) for row in old_rows if row["start"]=="fb0a" and row["kind"]=="initial_native_J")
    zero_expected=next(float(row["J"]) for row in zero_rows if row["kind"]=="initial_native_J")
    probes=[]
    def native_j(control:torch.Tensor,label:str)->float:
        start=time.monotonic()
        with torch.no_grad():value=problem.objective(control,parameters)
        if not isinstance(value,torch.Tensor) or value.ndim!=0 or not bool(torch.isfinite(value)):
            raise ValueError(f"native J probe {label} is nonfinite/malformed")
        result=float(value);probes.append({"label":label,"control_sha256":_tensor_sha(control),
            "J":result,"elapsed_seconds":time.monotonic()-start})
        _write(ATTEMPT/"equivalence_probes.json",{"plan_sha256":digest,"probes":probes})
        return result
    fb_j=native_j(points["fb0a"],"fb0a-baseline")
    if fb_j!=fb_expected or fb_j!=0.060988878373729814:
        raise ValueError("patched transport changed the sealed fb0a native J")
    # The zero start is re-evaluated at its original Powell callback below.
    zero_checkpoint=torch.as_tensor(json.loads((OLD_ATTEMPT/"zero.candidate_checkpoint.json").read_text())["best_control"],dtype=torch.float64)
    zero_checkpoint_key=zero_checkpoint.numpy().tobytes()
    if zero_checkpoint_key not in cache or cache[zero_checkpoint_key]!=zero_best_cost:
        raise ValueError("saved zero checkpoint is not one of the old exact cached controls")
    # Freeze the state. The cache replay below performs the zero-start and saved-best native checks inline.
    _write(ATTEMPT/"equivalence_probes.json",{"plan_sha256":digest,"probes":probes,
        "zero_start_and_checkpoint":"to be checked at their original replay callbacks"})
    old_elapsed=OLD_SEARCH_SECONDS
    external_validation_calls=len(probes)
    global_seconds_left=float(plan["cumulative_budget"]["remaining_original_search_internal_seconds"])
    deadline=time.monotonic()+global_seconds_left
    zero_calls=0;scipy_calls=0;new_native_calls=0;fb_probed=True;zero_start_probed=False;zero_checkpoint_probed=False
    # The 301-call old allowance includes 3 equivalence probes; fb0a uses one,
    # zero-start and its old best checkpoint are rechecked inline during replay.
    zero_remaining=301-len(probes)
    if zero_remaining<=0:raise ValueError("continuation native objective budget exhausted by preflight")
    eval_path=ATTEMPT/"continuation_evaluations.jsonl"
    checkpoint={"plan_sha256":digest,"status":"running","prior_search_elapsed_seconds":OLD_SEARCH_SECONDS,
        "prior_successful_native_J_evaluations":OLD_SUCCESSFUL_CALLS,
        "prior_zero_successful_native_J_evaluations":OLD_ZERO_CALLS,
        "new_validation_objective_calls":len(probes),"new_zero_native_objective_budget":zero_remaining,
        "new_zero_native_objective_calls":0,"powell_function_calls_replayed":0,
        "scipy_version":scipy.__version__,"source_before":source_before,
        "input_before":input_before,"equivalence_probes":probes,
        "best_J":zero_best_cost,"best_control":zero_checkpoint.tolist(),
        "failed_control_sha256":FAILED_ZERO_SHA,"failed_control_recovered":False}
    _write(ATTEMPT/"continuation_checkpoint.json",checkpoint)

    def total_successful_native_calls()->int:
        # Inline zero-start/checkpoint probes are already counted in new_native_calls.
        return OLD_SUCCESSFUL_CALLS+len(probes)+new_native_calls-int(zero_start_probed)-int(zero_checkpoint_probed)

    def f(vector:Any)->float:
        nonlocal scipy_calls,zero_calls,new_native_calls,zero_start_probed,zero_checkpoint_probed,zero_best_cost
        if time.monotonic()>=deadline:raise NativeBudgetReached("cumulative original search time reached")
        scipy_calls+=1;zero_calls+=1
        array=np.asarray(vector,dtype=np.float64)
        key=np.ascontiguousarray(array).tobytes()
        cache_hit=key in cache
        label="cached_prior_J"
        if not cache_hit and scipy_calls<700:
            raise ValueError(f"cached replay diverged before original callback 700 (callback {scipy_calls})")
        if scipy_calls==700 and key!=failed_key:
            raise ValueError("first unseen continuation control differs from reproduced callback-700 failure")
        if key==failed_key and scipy_calls!=700:
            raise ValueError("reproduced failed control appeared at an unexpected Powell callback")
        if cache_hit:
            value=cache[key]
            if not zero_start_probed and scipy_calls==1:
                # One fresh native check replaces callback 1's old cache return.
                control=torch.as_tensor(array,dtype=torch.float64)
                fresh=native_j(control,"zero-start-equivalence")
                if fresh!=zero_expected or fresh!=value:raise ValueError("zero-start J changed under support fix")
                new_native_calls+=1;zero_start_probed=True
                label="fresh_zero_start_equivalence"
            elif key==zero_checkpoint_key and not zero_checkpoint_probed:
                control=torch.as_tensor(array,dtype=torch.float64)
                fresh=native_j(control,"zero-checkpoint-equivalence")
                if fresh!=zero_best_cost or fresh!=value:raise ValueError("saved zero checkpoint J changed under support fix")
                new_native_calls+=1;zero_checkpoint_probed=True
                label="fresh_zero_checkpoint_equivalence"
        else:
            if new_native_calls>=zero_remaining:raise NativeBudgetReached("cumulative native-J call cap reached")
            control=torch.as_tensor(array,dtype=torch.float64)
            call_start=time.monotonic()
            event={"start":"zero","powell_callback":scipy_calls,
                "control":array.tolist(),"control_sha256":_tensor_sha(control),
                "status":"native_J_started","recovered_old_failure":key==failed_key}
            _write(ATTEMPT/"in_progress.json",event)
            try:
                with torch.no_grad():native=problem.objective(control,parameters)
            except ValueError as error:
                event.update(status="native_validation_refused",error=str(error),
                    elapsed_seconds=time.monotonic()-call_start)
                with (ATTEMPT/"continuation_evaluations.jsonl").open("a") as out:
                    out.write(json.dumps(event,allow_nan=False)+"\n")
                _write(ATTEMPT/"in_progress.json",event)
                raise
            if not isinstance(native,torch.Tensor) or native.ndim!=0 or not bool(torch.isfinite(native)):
                raise ValueError("continued native J is nonfinite or malformed")
            value=float(native);label="new_native_J"
            new_native_calls+=1;cache[key]=value
            if value<zero_best_cost:
                zero_best_cost=value
                checkpoint["best_J"]=value
                checkpoint["best_control"]=array.tolist()
            event.update(status="native_J_completed",J=value,
                elapsed_seconds=time.monotonic()-call_start,
                cumulative_native_J_calls=total_successful_native_calls())
            with (ATTEMPT/"continuation_evaluations.jsonl").open("a") as out:
                out.write(json.dumps(event,allow_nan=False)+"\n")
            if key==failed_key:checkpoint["failed_control_recovered"]=True
        checkpoint["new_zero_native_objective_calls"]=new_native_calls
        checkpoint["powell_function_calls_replayed"]=scipy_calls
        checkpoint["status"]="running"
        with (ATTEMPT/"powell_replay.jsonl").open("a") as out:
            out.write(json.dumps({"callback":scipy_calls,"control_sha256":_tensor_sha(torch.as_tensor(array)),
                "cache_hit":cache_hit,"source":label,"J":value},allow_nan=False)+"\n")
        _write(ATTEMPT/"continuation_checkpoint.json",checkpoint)
        _write(ATTEMPT/"zero.candidate_checkpoint.json",{
            "plan_sha256":digest,"status":"running","powell_callback":scipy_calls,
            "new_zero_objective_calls":new_native_calls,"cumulative_successful_native_J_calls":total_successful_native_calls(),
            "best_J":checkpoint["best_J"],"best_control":checkpoint["best_control"],
            "failed_control_recovered":checkpoint["failed_control_recovered"]})
        return value

    zero_input=np.zeros(26,dtype=np.float64)
    start_j=zero_expected
    box=math.sqrt(2*start_j)
    start_clock=time.monotonic()
    try:
        result=__import__("scipy.optimize",fromlist=["minimize"]).minimize(
            f,zero_input,method="Powell",bounds=[(-box,box)]*26,
            options={"maxfev":1000,"maxiter":100,"xtol":1e-11,"ftol":1e-13,"disp":False})
        checkpoint.update(status="powell_completed",scipy_success=bool(result.success),
            scipy_message=str(result.message),scipy_nfev=int(result.nfev),scipy_nit=int(result.nit),
            result_J=float(result.fun),result_control=np.asarray(result.x,dtype=np.float64).tolist(),
            result_control_sha256=_tensor_sha(torch.as_tensor(result.x,dtype=torch.float64)))
    except NativeBudgetReached as error:
        checkpoint.update(status="cumulative_call_budget_refusal_with_best_so_far",
            stop_reason=str(error))
    except ValueError as error:
        checkpoint.update(status="native_domain_refusal_without_penalty",
            stop_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        checkpoint.update({"new_zero_native_objective_calls":new_native_calls,
            "new_validation_objective_calls":len(probes),
            "total_successful_native_J_calls":total_successful_native_calls(),
            "powell_function_calls_replayed":scipy_calls,
            "cumulative_search_elapsed_seconds":old_elapsed+time.monotonic()-start_clock,
            "Q_calls":0,"R_calls":0,"E_calls":0,
            "HVP_status_after_previous_failed_partial_attempt":"unknown_aborted_partial_native_attempt"})
        checkpoint["equivalence_probes"]=[*probes,
            {"label":"zero-start-equivalence","completed":zero_start_probed},
            {"label":"zero-checkpoint-equivalence","completed":zero_checkpoint_probed}]
        _write(ATTEMPT/"continuation_checkpoint.json",checkpoint)
        if source_before!={name:_sha(ROOT/name) for name in plan["source_files"]}:
            raise ValueError("pinned native source changed during continuation")
        if input_before!={name:cycle.shared._input_identity(problem,original,point,parameters,truth)
                for name,point in points.items()}:
            raise ValueError("fixed native problem inputs changed during continuation")
    return checkpoint


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze_plan(),sort_keys=True));return
    if not args.plan_sha256:raise SystemExit("--plan-sha256 is required")
    load_plan(args.plan_sha256)
    if args.child:
        result=_run(args.plan_sha256);print(json.dumps({k:v for k,v in result.items() if k!="source_before" and k!="input_before"},sort_keys=True));return
    from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic
    CONT.mkdir(parents=True,exist_ok=True);ATTEMPT.mkdir(parents=True,exist_ok=True)
    if (ATTEMPT/"continuation_checkpoint.json").exists():
        raise ValueError("continuation already launched; resume through the same continuation plan only")
    remaining=3600.0-OLD_SEARCH_SECONDS-FAILED_PREFLIGHT_SECONDS
    resource=run_guarded_diagnostic([sys.executable,str(Path(__file__).resolve()),"--child","--plan-sha256",args.plan_sha256],
        wall_seconds=remaining,rss_bytes=2*1024**3,report_path=ATTEMPT/"resource.json",log_path=ATTEMPT/"search.log")
    _write(ATTEMPT/"run.json",{"resource":resource,"plan_sha256":args.plan_sha256,
        "execution_status":"completed" if resource.get("exit_code")==0 else "failed"})
    print(json.dumps({"resource":resource,"execution_status":"completed" if resource.get("exit_code")==0 else "failed"},sort_keys=True))
    if resource.get("exit_code")!=0:raise SystemExit(1)


if __name__=="__main__":main()
