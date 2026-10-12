"""Same-point gradient/robust-GN comparison under fixed inputs and one budget."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.weather_scenarios import fv_point_3h_netflux_continuation as previous
from examples.weather_scenarios import fv_point_3h_netflux_continuous_endpoint as endpoint
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_comparison as coupled

tangent, geometry, shared = previous.tangent, previous.geometry, previous.shared
finite, prepared, search = previous.finite, previous.prepared, previous.search
CurrentBackend, _sha = previous.CurrentBackend, previous._sha
WORK = previous.previous.EVIDENCE / "netflux_direction_comparison_20261012"
PLAN, ATTEMPT = WORK / "PLAN.json", WORK / "attempt1"
VERSION, BASE_SHA = previous.VERSION, endpoint.BASE_SHA
ARM_ORDER = ("baseline_tangent", "robust_gn_coupled")
NEW_SOURCES = {"examples/weather_scenarios/fv_point_3h_netflux_direction_comparison.py",
    "tests/test_fv_point_3h_netflux_direction_comparison.py",
    "examples/weather_scenarios/fv_point_3h_netflux_continuous_endpoint.py",
    "tests/test_fv_point_3h_netflux_continuous_endpoint.py"}
POLICY: dict[str, Any] = {**previous.POLICY, "max_commits": 1, "hvp_calls": 4,
    "jacobian_row_vjp_calls": 24, "dense_solves": 1, "face_scale": previous.FACE_SCALE,
    "max_candidates": 48, "max_candidates_per_point": 24,
    "arms": list(ARM_ORDER), "budget_scope": "one global600s, including all common and GN preparation",
    "comparison_required_for_commit": "both arms independently P2/fullG eligible",
    "selection": "actual repeated R then J, scale-aware ties prefer gradient; no E",
    "base_qualification": "same_version_closed_budget_prefix_then_fresh_current_P2"}


def load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("direction comparison plan identity mismatch")
    parent = endpoint.load_endpoint()["parent_plan"]
    plan = json.loads(path.read_text())
    archives = set(parent["archive_files"]) | {p.relative_to(ROOT).as_posix() for p in endpoint.RECEIPTS}
    if (plan.get("experiment_kind") != "same_point_fixed_version_gradient_robust_GN"
            or plan.get("numerical_version") != VERSION or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("policy") != POLICY
            or set(plan.get("source_files", {})) != set(parent["source_files"]) | NEW_SOURCES
            or set(plan.get("archive_files", {})) != archives):
        raise ValueError("direction plan scope/version/input/policy changed")
    for group in ("source_files", "archive_files"):
        if any(plan[group].get(n) != h for n, h in parent[group].items()):
            raise ValueError("direction plan changed an inherited numerical/input pin")
        for name, digest0 in plan[group].items():
            if _sha(ROOT / name) != digest0:
                raise ValueError(f"direction source/archive pin mismatch: {name}")
    return plan


def direction_efficiency(model: Any) -> dict[str, float]:
    G, r = model.residual, model.residual_direction
    R, rr, slope = float(torch.dot(G, G)), float(torch.dot(r, r)), float(torch.dot(G, r))
    if not all(math.isfinite(x) for x in (R, rr, slope)) or R <= 0 or rr <= 0:
        raise ValueError("direction efficiency needs resolved finite nonzero G and DGd")
    alpha = -slope / rr
    cosine = -slope / math.sqrt(R * rr)
    return {"R": R, "G_dot_DGd": slope, "DGd_squared": rr,
        "unbounded_model_alpha": alpha, "chi": cosine * cosine,
        "cosine_G_minus_DGd": cosine}


def choose_repeated_arm(arms: list[dict[str, Any]], dtype: torch.dtype) -> dict[str, Any] | None:
    if len(arms) != 2 or any(a.get("status") != "independent_P2_fullG_eligible" for a in arms):
        return None
    proposals = [{**a["repeat"], "direction_model": a["name"]} for a in arms]
    winner = tangent._select_direction_candidate(proposals, dtype)
    return next(a for a in arms if winner is not None and a["name"] == winner["direction_model"])


def _run_child(path: Path, digest: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    plan = load_plan(path, digest)
    base = endpoint.load_endpoint()
    deadline = started + POLICY["internal_seconds"]
    backend = CurrentBackend(plan, path, digest, ATTEMPT / "operators", deadline, base)
    c = backend.control
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "current_P2", "plan_sha256": digest, "policy": POLICY,
        "numerical_version": VERSION, "base_control_sha256": BASE_SHA,
        "current_control": c.tolist(), "current_control_sha256": BASE_SHA,
        "current_theta": base["theta"], "current_full_G": base["full_G"].tolist(),
        "last_confirmed_control": c.tolist(), "last_confirmed_control_sha256": BASE_SHA,
        "last_confirmed_theta": base["theta"], "last_confirmed_full_G": base["full_G"].tolist(),
        "last_confirmed_numerical_version": VERSION, "closed_steps": [],
        "optimizer_steps_applied": 0, "candidate_committed": False, "arms": [],
        "hvp_started": 0, "hvp_completed": 0, "hvp_history": [],
        "candidate_evaluations_started": 0, "candidate_evaluations_completed": 0,
        "root_claim": False, "minimum_claim": False, "response_claim": False, "score_claim": False,
        "original_problem_solved": False, "terminal_E_recomputed": False,
        "Q0_scope": "conditional active-face hypothesis",
        "source_before": backend.source_anchor, "input_before": backend.input_anchor,
        "runtime_before": backend.runtime_anchor}
    def write() -> None:
        tangent._write(output, record)
    def finish(status: str, reason: str = "", *, failed: bool = False) -> dict[str, Any]:
        record["stop_stage"] = record["numerical_status"]
        control = torch.as_tensor(record["current_control"], dtype=torch.float64)
        source = shared._source_hashes(plan, path.resolve(), digest)
        inp = shared._input_identity(backend.problem, backend.original, control, backend.parameters, backend.truth)
        runtime = shared._runtime()
        input_ok = geometry.model._fixed_input(inp, backend.input_anchor, tangent._tensor_sha(control))
        record.update(phase="finished", execution_status="failed" if failed else "completed",
            numerical_status=status, termination_reason=reason, source_after=source,
            source_unchanged=source == backend.source_anchor, input_after=inp, fixed_input_unchanged=input_ok,
            runtime_after=runtime, runtime_unchanged=runtime == backend.runtime_anchor,
            deadline_passed=time.monotonic() < deadline, elapsed_seconds=time.monotonic()-started,
            backend_timing=backend.timings,
            GN_rows_started=sum(a.get("GN_operator", {}).get("jacobian_rows_started", 0) for a in record["arms"]),
            GN_rows_completed=sum(a.get("GN_operator", {}).get("jacobian_rows_completed", 0) for a in record["arms"]),
            GN_solves_completed=sum(a.get("GN_operator", {}).get("dense_solves_completed", 0) for a in record["arms"]),
            postcommit_readiness_performed=False)
        for arm in record["arms"]:
            if "inclusive_arm_seconds" not in arm:
                arm["inclusive_arm_seconds"]=time.monotonic()-started-arm["start_offset_seconds"]
                arm["status"]="interrupted"
        write()
        if not (record["source_unchanged"] and input_ok and record["runtime_unchanged"]):
            raise ValueError("comparison fixed source/input/runtime changed")
        return record
    write()
    try:
        backend._check(c)
        observed = tangent._observe(shared, backend.problem, c, backend.parameters, backend.weights)
        gm, gp = observed["side"][-1][1], observed["side"][1][1]
        if not all((observed["pair"]["passed"], observed["face_ok"],
                    observed["side_objectives_match_native"], observed["side_gradients_finite"])):
            return finish("unsupported_native_pair")
        oldtheta = float(base["theta"])
        stored = torch.cat(((1-oldtheta)*gm+oldtheta*gp,(observed["q"]/previous.FACE_SCALE).reshape(1)))
        match = previous.match_previous_P2(observed,oldtheta,stored,base["full_G"],base["repeat"])
        record["previous_P2_match"] = match
        if not match["passed"]:
            raise ValueError("current comparison point differs from its closed P2")
        normal = geometry._face_normal(c,backend.weights)
        outward = previous.native_face_release(gm,gp,normal)
        record["outward_derivatives"] = outward
        write()
        if outward["release_required"]:
            return finish("native_face_release_required_no_Q0_commit")
        theta = float(tangent.minimum_mixture_weight(gm,gp))
        G = torch.cat(((1-theta)*gm+theta*gp,(observed["q"]/previous.FACE_SCALE).reshape(1)))
        rematch = previous.match_previous_P2(observed,theta,G,base["full_G"],base["repeat"])
        if not rematch["passed"]:
            raise ValueError("current minimum mixture differs from closed P2")
        pivot = geometry._pivot(c,backend.weights)
        _, chart, _ = geometry._chart_jacobian(c,backend.weights,pivot,0.)
        _, _, grad_direction, support = tangent.tangent_direction(gm,gp,normal,chart,theta,pivot=pivot)
        record.update(base_J=float(observed["native_j"]),base_R=float(torch.dot(G,G)),
            minimum_mixture_P2_match=rematch,common_face_support=support,
            common_preparation_seconds=time.monotonic()-started)
        if float(G.norm()) <= POLICY["residual_norm_candidate_tolerance"]:
            return finish("residual_candidate_requires_exact_qualification")
        specs: dict[str,Any] = {"baseline_tangent":{"direction":grad_direction}}
        bundles: dict[str,Any] = {}
        for index,name in enumerate(ARM_ORDER):
            arm_started = time.monotonic()
            arm: dict[str,Any] = {"name":name,"order":index,"status":"preparing","candidate":None,
                "trials":[],"hvp_started":0,"hvp_completed":0,"start_offset_seconds":arm_started-started}
            record["arms"].append(arm);write()
            if name == "robust_gn_coupled":
                op = {"jacobian_rows_started":0,"jacobian_rows_completed":0,"jacobian_row_history":[],
                    "dense_solves_started":0,"dense_solves_completed":0}
                context = {"problem":backend.problem,"parameters":backend.parameters,"record":op,
                    "output":ATTEMPT/"operators/GN.json","deadline":deadline,"weights":backend.weights,
                    "plan":plan,"plan_path":path,"plan_sha":digest,"model_state":{"observed":observed}}
                record["numerical_status"]="GN_rows_and_solve";write()
                factory_started=time.monotonic()
                arm["GN_operator"]=op
                try:
                    built=coupled._direction_model_factory(context)(c,theta,gm,gp,normal,chart,pivot)
                finally:
                    arm["GN_preparation_seconds"]=time.monotonic()-factory_started
                    backend.timings["row_solve_seconds_inclusive"] += arm["GN_preparation_seconds"]
                backend.row_count=int(op["jacobian_rows_completed"]);backend.solve_count=int(op["dense_solves_completed"])
                arm["GN_operator"]=op
                if backend.row_count!=24 or backend.solve_count!=1:
                    raise ValueError("GN construction did not close exactly24rows/one solve")
                if not torch.equal(built[0]["direction"],grad_direction):
                    raise ValueError("GN factory changed the shared gradient baseline")
                specs[name]=next(s for s in built if s["name"]==name)
            direction=specs[name]["direction"]
            tangent._validate_direction_override(gm,normal,chart,pivot,direction)
            arm.update(direction=direction.tolist(),direction_sha256=tangent._tensor_sha(direction))
            products={}
            grad_fn=torch.func.grad(backend.problem.objective,argnums=0)
            record["numerical_status"] = f"{name}_fresh_HVP"
            for side in (-1,1):
                if record["hvp_started"]>=POLICY["hvp_calls"]:
                    raise ValueError("comparison HVP cap reached")
                record["hvp_started"]+=1;arm["hvp_started"]+=1
                event={"arm":name,"side":side,"control_sha256":BASE_SHA,"stage":"precheck"}
                record["hvp_history"].append(event);write()
                backend._check(c)
                event["stage"]="jvp_started";write()
                hvp_started_at = time.perf_counter()
                with previous.transport.selected_face_extension("y",4,3,side):
                    value=torch.func.jvp(lambda x:grad_fn(x,backend.parameters),(c,),(direction,))[1]
                hvp_seconds = time.perf_counter() - hvp_started_at
                backend.timings["pure_hvp_seconds"] += hvp_seconds
                if value.shape!=c.shape or value.dtype!=c.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError("comparison HVP malformed/nonfinite")
                event.update(stage="jvp_returned",vector=value.tolist(),seconds=hvp_seconds);write()
                backend._check(c)
                products[side]=value
                event["stage"]="qualified_complete"
                record["hvp_completed"]+=1;arm["hvp_completed"]+=1;write()
            model=prepared.mixing.minimum_tangent_model(gm,gp,products[-1],products[1],normal,chart,theta,
                float(observed["q"]),pivot,previous.FACE_SCALE,direction_override=direction)
            if not finite._base_model_closure(model,G,theta)["passed"] or not torch.equal(model.direction,direction):
                raise ValueError("comparison model does not reproduce current G/theta/direction")
            arm.update(side_HVP={str(s):products[s].tolist() for s in (-1,1)},model_gates=model.gates,
                residual=model.residual.tolist(),residual_direction=model.residual_direction.tolist(),
                efficiency=direction_efficiency(model),merit_product=model.merit_product)
            if not all(model.gates.get(k) is True for k in
                    ("finite","both_side_gradients_descend","merit_descends","envelope_slope_matches_residual_dot")):
                arm.update(status="model_refused",inclusive_arm_seconds=time.monotonic()-arm_started);write();continue
            bundle={"observed":observed,"theta":theta,"G":G,"R":record["base_R"],"J":record["base_J"],
                "direction":direction,"model":model,"pivot":pivot,"cost_slopes":model.side_products}
            bundles[name]=bundle;receipts={}
            def evaluate(candidate: Tensor,_theta: float,alpha: float) -> dict[str,Any]:
                record["candidate_evaluations_started"]+=1;write()
                facts=backend.evaluate(bundle,candidate,alpha)
                record["candidate_evaluations_completed"]+=1
                file=ATTEMPT/name/f"candidate_{len(receipts)+1:02d}.json";tangent._write(file,facts)
                receipts[alpha]=(file,_sha(file),facts)
                compact={k:v for k,v in facts.items() if k!="branch_trace"}
                compact.update(candidate_receipt=str(file.relative_to(ROOT)),candidate_receipt_sha256=_sha(file))
                return compact
            def progress(trials: list[dict[str,Any]],meta: dict[str,Any]) -> None:
                arm.update(trials=trials,search=meta);write()
            record["numerical_status"]="actual_JR_search";write()
            selected,trials,meta=search.search(model,c,evaluate,
                chart_candidate=lambda x,d,a:backend.chart(bundle,x,d,a),progress=progress,
                policy={"name":"dyadic_JR","max_evaluations":24,"max_backtracks":24,"radius":previous.RADIUS})
            arm.update(candidate=selected,trials=trials,search=meta)
            if selected is not None:
                file,file_sha,facts=receipts[selected["alpha"]]
                if _sha(file)!=file_sha:
                    raise ValueError("comparison candidate receipt changed")
                selected["branch_trace"]=facts["branch_trace"]
                record["numerical_status"]="arm_independent_P2_fullG";write()
                proposal,repeat=backend.repeat(bundle,selected)
                check=previous.fresh_full_G(repeat,torch.as_tensor(selected["G"],dtype=c.dtype),backend.weights)
                proposal.update(selected_arm=name,selected_direction=name,conditional_Q0=True,
                    acceptance_policy="original_JR_conditional_Q0")
                if not (tangent.fresh_final_closure(proposal,repeat) and check["passed"]
                        and float(repeat["objective"])<=float(proposal["J_armijo_bound"])
                        and float(repeat["F_squared"])<=float(proposal["F_squared_armijo_bound"])):
                    raise ValueError("arm independent P2/fullG approval failed")
                arm.update(proposal=proposal,repeat=repeat,full_G_check=check,status="independent_P2_fullG_eligible",
                    mixed_gradient_max=max(abs(x) for x in check["G"][:26]))
            else:
                arm["status"]="candidate_search_exhausted"
            arm["inclusive_arm_seconds"]=time.monotonic()-arm_started;write()
        winner=choose_repeated_arm(record["arms"],c.dtype)
        record["comparison_complete"]=winner is not None
        if winner is None:
            return finish("comparison_incomplete_no_commit")
        record["selected_direction"]=winner["name"];write()
        point={"point_index":0,"base_control_sha256":BASE_SHA,"full_G":G.tolist(),
            "previous_P2_match":match,"minimum_mixture_P2_match":rematch,"outward_derivatives":outward,
            "direction":winner["direction"],"side_HVP":winner["side_HVP"],"hvp_started":2,"hvp_completed":2,
            "model_gates":winner["model_gates"],"selected_candidate":winner["candidate"]}
        record=previous.persist_commit(output,record,point,winner["proposal"],winner["repeat"],winner["full_G_check"])
        return finish("same_point_comparison_one_candidate_committed")
    except TimeoutError as error:
        return finish("budget_exhausted_comparison_incomplete_no_commit",str(error))
    except Exception as error:
        return finish("comparison_support_or_contract_failure",f"{type(error).__name__}: {error}",failed=True)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--plan-sha256",required=True);parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    if args.child:
        result=_run_child(PLAN,args.plan_sha256,ATTEMPT/"step.json")
        if result["execution_status"]=="failed": raise SystemExit(1)
        return
    load_plan(PLAN,args.plan_sha256)
    if ATTEMPT.exists(): raise ValueError("comparison attempt must be fresh")
    ATTEMPT.mkdir()
    resource=previous.run_guarded_diagnostic([str(ROOT/".venv/bin/python"),str(Path(__file__).resolve()),
        "--child","--plan-sha256",args.plan_sha256],wall_seconds=POLICY["outer_seconds"],
        rss_bytes=POLICY["rss_bytes"],report_path=ATTEMPT/"step.resource.json",log_path=ATTEMPT/"step.log")
    result={"execution_status":shared._execution_status(resource,wall_seconds=POLICY["outer_seconds"],rss_bytes=POLICY["rss_bytes"]),
        "resource":resource,"child_sha256":_sha(ATTEMPT/"step.json") if (ATTEMPT/"step.json").exists() else None}
    tangent._write(ATTEMPT/"step.run.json",result);print(json.dumps(result))


if __name__=="__main__": main()
