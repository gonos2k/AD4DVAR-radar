"""One fresh-point gradient versus two-direction trust-radius trial.

All candidate gates and independent final qualification reuse the original J/R
backend. Coefficients change within a fixed point; no additional HVP is needed.
"""
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
from examples.weather_scenarios import fv_point_3h_netflux_direction_comparison as comparison
from examples.weather_scenarios import fv_point_3h_netflux_comparison_endpoint as endpoint
from examples.weather_scenarios.fv_point_3h_subspace_model import solve_two_direction_subspace
from examples.weather_scenarios.fv_point_3h_subspace_policy import radius_factor

previous = comparison.previous
geometry, tangent, shared = comparison.geometry, comparison.tangent, comparison.shared
prepared, finite = comparison.prepared, comparison.finite
WORK = comparison.WORK.parent / "netflux_subspace_20261012"
PLAN, ATTEMPT = WORK / "PLAN.json", WORK / "attempt1"
VERSION, BASE_SHA = comparison.VERSION, endpoint.BASE_SHA
ARM_ORDER = ("baseline_tangent", "two_direction_subspace")
INITIAL_RADIUS = 1.8757177916958825e-5  # Closed PR300 gradient displacement, in control units.
NEW_SOURCES = {"examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py",
    "examples/weather_scenarios/fv_point_3h_subspace_model.py",
    "examples/weather_scenarios/fv_point_3h_subspace_policy.py",
    "examples/weather_scenarios/fv_point_3h_netflux_comparison_endpoint.py",
    "tests/test_fv_point_3h_netflux_subspace_trial.py",
    "tests/test_fv_point_3h_subspace_model.py",
    "tests/test_fv_point_3h_netflux_comparison_endpoint.py"}
POLICY = {**comparison.POLICY, "max_candidates": 12, "max_candidates_per_point": 6,
    "candidate_cap_per_point": 6, "hvp_calls_per_point": 4,
    "max_candidates_per_arm": 6, "arms": list(ARM_ORDER), "radius": INITIAL_RADIUS,
    "candidate_policy": "nonnegative_two_coefficient_trust_radius",
    "radius_update": "half if rejected/nonfinite/rho_R<0.25/eta_vec>0.5; otherwise hold; never expand",
    "radius_update_after_accept": "record next radius then stop arm at first qualified candidate",
    "chart_safeguard": "at most 33 chart calls per candidate; contract alpha by 32 bisections if actual displacement exceeds radius",
    "base_qualification": "PR300_closed_winner_then_fresh_current_P2",
    "coefficient_solves": "at most 12 tiny model solves; no additional HVP",
    "selection": "actual repeated R then J; both arms required; no E"}


def load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.resolve() != PLAN.resolve() or comparison._sha(path) != digest:
        raise ValueError("subspace plan identity mismatch")
    parent = endpoint.load_endpoint()["parent_plan"]
    plan = json.loads(path.read_text())
    archives = set(parent["archive_files"]) | {p.relative_to(ROOT).as_posix() for p in endpoint.RECEIPTS}
    if (plan.get("experiment_kind") != "fresh_point_two_direction_trust_radius"
            or plan.get("policy") != POLICY or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("numerical_version") != VERSION
            or set(plan.get("source_files", {})) != set(parent["source_files"]) | NEW_SOURCES
            or set(plan.get("archive_files", {})) != archives):
        raise ValueError("subspace plan changed scope, version, source or budget")
    for group in ("source_files", "archive_files"):
        if any(plan[group].get(n) != h for n, h in parent[group].items()):
            raise ValueError("subspace trial changed inherited fixed inputs/source")
        if any(comparison._sha(ROOT / n) != h for n, h in plan[group].items()):
            raise ValueError("subspace trial source/archive pin mismatch")
    return plan


def composed_model(gm: Tensor, gp: Tensor, normal: Tensor, chart: Tensor, theta: float,
        q: float, pivot: int, models: list[Any], products: list[dict[int, Tensor]], coefficients: Tensor) -> Any:
    direction = sum((coefficients[i] * models[i].direction for i in range(2)), torch.zeros_like(gm))
    hvp = {s: sum((coefficients[i] * products[i][s] for i in range(2)), torch.zeros_like(gm)) for s in (-1, 1)}
    model = prepared.mixing.minimum_tangent_model(gm, gp, hvp[-1], hvp[1], normal, chart,
        theta, q, pivot, previous.FACE_SCALE, direction_override=direction)
    expected = sum((coefficients[i] * models[i].residual_direction for i in range(2)),
        torch.zeros_like(model.residual_direction))
    if not finite._full_G_match(model.residual_direction, expected)["passed"]:
        raise ValueError("composed minimum-mixture derivative lost linearity")
    return model


def model_fidelity(G: Tensor, r: Tensor, actual: Tensor) -> dict[str, float]:
    predicted = G + r
    pred_drop = torch.dot(G, G) - torch.dot(predicted, predicted)
    actual_drop = torch.dot(G, G) - torch.dot(actual, actual)
    actual_delta = torch.linalg.vector_norm(actual - G)
    err = torch.linalg.vector_norm(actual - predicted)
    return {"rho_R": float(actual_drop / pred_drop) if bool(pred_drop > 0) else math.nan,
        "eta_vec": float(err / actual_delta) if bool(actual_delta > 0) else math.inf,
        "predicted_R": float(torch.dot(predicted, predicted)), "predicted_decrease": float(pred_drop),
        "actual_decrease": float(actual_drop), "vector_error_norm": float(err),
        "actual_change_norm": float(actual_delta), "predicted_change_norm": float(r.norm())}


def chart_within_radius(backend: Any, bundle: dict[str, Any], c: Tensor,
        radius: float) -> tuple[Tensor, float, int]:
    backend._check(c)
    candidate = backend.chart(bundle,c,bundle["direction"],1.)
    if float((candidate-c).norm()) <= radius:
        return candidate, 1., 1
    # Bound actual retracted displacement, not just its tangent prediction.
    lower, upper = 0., 1.
    candidate = c.clone()
    for _ in range(32):
        alpha = (lower+upper)/2
        trial = backend.chart(bundle,c,bundle["direction"],alpha)
        if float((trial-c).norm()) <= radius:
            candidate, lower = trial, alpha
        else:
            upper = alpha
    backend._check(c)
    if lower == 0. or torch.equal(candidate,c):
        raise ValueError("chart safeguard could not resolve a nonzero in-radius candidate")
    return candidate, lower, 33


def choose_radius(G: Tensor, D: Tensor, V: Tensor, record: dict[str,Any]) -> float:
    """The original fixed-radius experiment keeps its frozen search length."""
    return INITIAL_RADIUS


def _run_child(path: Path, digest: str, output: Path, *, experiment: Any | None = None) -> dict[str, Any]:
    config = sys.modules[__name__] if experiment is None else experiment
    started = time.monotonic()
    plan, base = config.load_plan(path, digest), config.endpoint.load_endpoint()
    deadline = started + config.POLICY["internal_seconds"]
    backend = comparison.CurrentBackend(plan, path, digest, config.ATTEMPT / "operators", deadline, base)
    c = backend.control
    record: dict[str, Any] = {"phase": "running", "execution_status": "running", "numerical_status": "current_P2",
        "plan_sha256": digest, "policy": config.POLICY, "numerical_version": VERSION, "base_control_sha256": config.BASE_SHA,
        "current_control": c.tolist(), "current_control_sha256": config.BASE_SHA, "current_theta": base["theta"],
        "current_full_G": base["full_G"].tolist(), "last_confirmed_control": c.tolist(),
        "last_confirmed_control_sha256": config.BASE_SHA, "last_confirmed_theta": base["theta"],
        "last_confirmed_full_G": base["full_G"].tolist(), "last_confirmed_numerical_version": VERSION,
        "closed_steps": [], "optimizer_steps_applied": 0, "candidate_committed": False,
        "hvp_started": 0, "hvp_completed": 0, "hvp_history": [], "arms": [], "fresh_models": [],
        "candidate_evaluations_started": 0, "candidate_evaluations_completed": 0,
        "root_claim": False, "minimum_claim": False, "response_claim": False, "score_claim": False,
        "original_problem_solved": False, "terminal_E_recomputed": False,
        "Q0_scope": "conditional active-face hypothesis", "source_before": backend.source_anchor,
        "input_before": backend.input_anchor, "runtime_before": backend.runtime_anchor}
    def write() -> None:
        tangent._write(output, record)
    def finish(status: str, reason: str = "", *, failed: bool = False) -> dict[str, Any]:
        control = torch.as_tensor(record["current_control"], dtype=c.dtype)
        source = shared._source_hashes(plan, path.resolve(), digest)
        inp = shared._input_identity(backend.problem, backend.original, control, backend.parameters, backend.truth)
        runtime = shared._runtime()
        input_ok = geometry.model._fixed_input(inp, backend.input_anchor, tangent._tensor_sha(control))
        record.update(phase="finished", execution_status="failed" if failed else "completed", numerical_status=status,
            termination_reason=reason, source_after=source, source_unchanged=source == backend.source_anchor,
            input_after=inp, fixed_input_unchanged=input_ok, runtime_after=runtime,
            runtime_unchanged=runtime == backend.runtime_anchor, elapsed_seconds=time.monotonic()-started,
            deadline_passed=time.monotonic() < deadline, backend_timing=backend.timings,
            postcommit_readiness_performed=False)
        write()
        if not (record["source_unchanged"] and input_ok and record["runtime_unchanged"]):
            raise ValueError("subspace source/input/runtime changed")
        return record
    write()
    try:
        observed = tangent._observe(shared, backend.problem, c, backend.parameters, backend.weights)
        gm, gp = observed["side"][-1][1], observed["side"][1][1]
        if not all((observed["pair"]["passed"], observed["face_ok"], observed["side_objectives_match_native"], observed["side_gradients_finite"])):
            return finish("unsupported_current_pair_no_commit")
        normal = geometry._face_normal(c, backend.weights)
        theta = float(base["theta"])
        G = torch.cat(((1-theta)*gm + theta*gp, (observed["q"]/previous.FACE_SCALE).reshape(1)))
        match = previous.match_previous_P2(observed, theta, G, base["full_G"], base["repeat"])
        record["previous_P2_match"] = match; write()
        if not match["passed"]:
            raise ValueError("fresh current point differs from PR300 P2")
        outward = previous.native_face_release(gm, gp, normal)
        record["outward_derivatives"] = outward; write()
        if outward["release_required"]:
            return finish("native_face_release_required_no_Q0_commit")
        theta = float(tangent.minimum_mixture_weight(gm, gp))
        G = torch.cat(((1-theta)*gm+theta*gp, (observed["q"]/previous.FACE_SCALE).reshape(1)))
        rematch = previous.match_previous_P2(observed, theta, G, base["full_G"], base["repeat"])
        if not rematch["passed"]:
            raise ValueError("current minimum mixture differs from PR300 P2")
        record.update(minimum_mixture_P2_match=rematch, base_J=float(observed["native_j"]), base_R=float(torch.dot(G,G)))
        if float(G.norm()) <= config.POLICY["residual_norm_candidate_tolerance"]:
            return finish("residual_candidate_requires_exact_qualification")
        pivot = geometry._pivot(c, backend.weights)
        _, chart, _ = geometry._chart_jacobian(c, backend.weights, pivot, 0.)
        op = {"jacobian_rows_started": 0, "jacobian_rows_completed": 0, "jacobian_row_history": [],
            "dense_solves_started": 0, "dense_solves_completed": 0}
        record["GN_operator"] = op; record["numerical_status"] = "GN_rows_and_solve"; write()
        context = {"problem": backend.problem, "parameters": backend.parameters, "record": op,
            "output": config.ATTEMPT/"operators/GN.json", "deadline": deadline, "weights": backend.weights,
            "plan": plan, "plan_path": path, "plan_sha": digest, "model_state": {"observed": observed}}
        gn_started = time.monotonic()
        try:
            specs = comparison.coupled._direction_model_factory(context)(c,theta,gm,gp,normal,chart,pivot)
        finally:
            record["GN_preparation_seconds"] = time.monotonic()-gn_started
            backend.timings["row_solve_seconds_inclusive"] += record["GN_preparation_seconds"]
        if op["jacobian_rows_completed"] != 24 or op["dense_solves_completed"] != 1:
            raise ValueError("GN row/solve budget closure failed")
        models, products = [], []
        for spec in specs:
            direction = spec["direction"]
            tangent._validate_direction_override(gm,normal,chart,pivot,direction)
            pair = {}
            for side in (-1,1):
                backend._check(c)
                if record["hvp_started"] >= 4:
                    raise ValueError("HVP cap reached")
                event = {"direction": spec["name"], "side": side, "control_sha256": config.BASE_SHA, "stage": "precheck"}
                record["hvp_started"] += 1; record["hvp_history"].append(event); write()
                event["stage"] = "jvp_started"; write()
                hs = time.perf_counter()
                with previous.transport.selected_face_extension("y",4,3,side):
                    pair[side] = torch.func.jvp(lambda x: torch.func.grad(backend.problem.objective,argnums=0)(x,backend.parameters), (c,), (direction,))[1]
                backend.timings["pure_hvp_seconds"] += time.perf_counter()-hs
                value = pair[side]
                if value.shape != c.shape or value.dtype != c.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError("fresh HVP malformed/nonfinite")
                event.update(stage="jvp_returned",vector=value.tolist()); write(); backend._check(c)
                event["stage"] = "qualified_complete"; record["hvp_completed"] += 1; write()
            model = prepared.mixing.minimum_tangent_model(gm,gp,pair[-1],pair[1],normal,chart,theta,
                float(observed["q"]),pivot,previous.FACE_SCALE,direction_override=direction)
            if not finite._base_model_closure(model,G,theta)["passed"] or not all(model.gates.get(k) is True for k in
                    ("finite","both_side_gradients_descend","merit_descends","envelope_slope_matches_residual_dot")):
                return finish("fresh_direction_model_refused_no_commit")
            models.append(model); products.append(pair)
            record["fresh_models"].append({"name":spec["name"],"direction":direction.tolist(),
                "residual_direction":model.residual_direction.tolist(),"side_HVP":{str(s):pair[s].tolist() for s in (-1,1)},
                "model_gates":model.gates,"efficiency":comparison.direction_efficiency(model)})
            write()
        if len(models) != 2 or [s["name"] for s in specs] != list(comparison.ARM_ORDER):
            raise ValueError("unexpected shared direction factory order")
        D = torch.stack([m.direction for m in models],dim=1)
        V = torch.stack([m.residual_direction for m in models],dim=1)
        radius0 = config.choose_radius(G,D,V,record)
        record["initial_radius"] = radius0
        record["common_preparation_seconds"] = time.monotonic()-started; write()
        if radius0 is None:
            return finish("no_resolved_subspace_entry_within_frozen_radius_cap")
        bundles = {}
        for name in ARM_ORDER:
            arm_started = time.monotonic()
            arm: dict[str,Any] = {"name":name,"status":"searching","trials":[],"candidate":None}
            record["arms"].append(arm); write()
            radius = radius0
            for index in range(6):
                backend._check(c)
                if name == "baseline_tangent":
                    alpha = min(comparison.direction_efficiency(models[0])["unbounded_model_alpha"], radius/float(D[:,0].norm()))
                    coefficients = torch.tensor([alpha,0.],dtype=c.dtype)
                    solver_diag: dict[str,Any] = {"scope":"gradient ray within same radius"}
                else:
                    solution = solve_two_direction_subspace(G,D,V,radius)
                    coefficients = solution.coefficients
                    solver_diag = {k: float(v) if isinstance(v,Tensor) else v for k,v in solution.diagnostics.items()}
                model = composed_model(gm,gp,normal,chart,theta,float(observed["q"]),pivot,models,products,coefficients)
                if not all(model.gates.get(k) is True for k in ("finite","both_side_gradients_descend","merit_descends","envelope_slope_matches_residual_dot")):
                    raise ValueError("composed model refused")
                bundle = {"observed":observed,"theta":theta,"G":G,"R":record["base_R"],"J":record["base_J"],
                    "direction":model.direction,"model":model,"pivot":pivot,"cost_slopes":model.side_products}
                candidate, chart_alpha, chart_calls = chart_within_radius(backend,bundle,c,radius)
                path_norm = float((candidate-c).norm())
                trial: dict[str,Any] = {"index":index,"radius":radius,"model_coefficients":coefficients.tolist(),
                    "coefficients":(chart_alpha*coefficients).tolist(),"chart_alpha":chart_alpha,"chart_calls":chart_calls,
                    "solver":solver_diag,"control":candidate.tolist(),"control_sha256":tangent._tensor_sha(candidate),
                    "actual_path_norm":path_norm,"direction":model.direction.tolist(),"DGd":model.residual_direction.tolist()}
                record["candidate_evaluations_started"] += 1; write()
                facts = backend.evaluate(bundle,candidate,chart_alpha)
                record["candidate_evaluations_completed"] += 1
                accepted = all(facts.get(k) is True for k in comparison.search._ALL_GATES)
                fidelity = model_fidelity(G,chart_alpha*model.residual_direction,torch.as_tensor(facts["G"],dtype=c.dtype)) if "G" in facts else {"rho_R":math.nan,"eta_vec":math.inf}
                factor = radius_factor(accepted,fidelity["rho_R"],fidelity["eta_vec"])
                trial.update(**facts,accepted=accepted,status="accepted" if accepted else "rejected",
                    fidelity={k:v if math.isfinite(v) else None for k,v in fidelity.items()},next_radius=radius*factor)
                receipt = config.ATTEMPT/name/f"candidate_{index+1:02d}.json"
                tangent._write(receipt,trial)
                trial.update(candidate_receipt=receipt.relative_to(ROOT).as_posix(),candidate_receipt_sha256=comparison._sha(receipt))
                arm["trials"].append(trial); write()
                if accepted:
                    proposal,repeat = backend.repeat(bundle,trial)
                    check = previous.fresh_full_G(repeat,torch.as_tensor(trial["G"],dtype=c.dtype),backend.weights)
                    proposal.update(selected_arm=name,selected_direction=name,conditional_Q0=True,acceptance_policy="original_JR_conditional_Q0")
                    if not (tangent.fresh_final_closure(proposal,repeat) and check["passed"]
                            and repeat["objective"] <= proposal["J_armijo_bound"] and repeat["F_squared"] <= proposal["F_squared_armijo_bound"]):
                        raise ValueError("independent candidate closure failed")
                    arm.update(status="independent_P2_fullG_eligible",candidate=trial,proposal=proposal,repeat=repeat,full_G_check=check)
                    bundles[name] = bundle; write(); break
                radius *= factor
            if arm["candidate"] is None:
                arm["status"] = "candidate_cap_reached"
            arm["search_seconds"] = time.monotonic()-arm_started; write()
        winner = comparison.choose_repeated_arm(record["arms"],c.dtype)
        record["comparison_complete"] = winner is not None
        if winner is None:
            return finish("comparison_incomplete_no_commit")
        name = winner["name"]
        record["selected_direction"] = name
        point = {"point_index":0,"base_control_sha256":config.BASE_SHA,"J":record["base_J"],"R":record["base_R"],"full_G":G.tolist(),
            "previous_P2_match":match,"minimum_mixture_P2_match":rematch,"outward_derivatives":outward,
            "direction":winner["candidate"]["direction"],"selected_candidate":winner["candidate"],
            "hvp_started":4,"hvp_completed":4,"model_gates":bundles[name]["model"].gates}
        record = previous.persist_commit(output,record,point,winner["proposal"],winner["repeat"],winner["full_G_check"])
        return finish("subspace_comparison_one_candidate_committed")
    except TimeoutError as error:
        return finish("budget_exhausted_no_commit",str(error))
    except Exception as error:
        return finish("support_or_contract_failure_no_commit",f"{type(error).__name__}: {error}",failed=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--plan-sha256",required=True); parser.add_argument("--child",action="store_true")
    args = parser.parse_args()
    if args.child:
        result = _run_child(PLAN,args.plan_sha256,ATTEMPT/"step.json")
        if result["execution_status"] == "failed":
            raise SystemExit(1)
        return
    load_plan(PLAN,args.plan_sha256)
    if ATTEMPT.exists():
        raise ValueError("subspace attempt must be fresh")
    ATTEMPT.mkdir()
    resource = previous.run_guarded_diagnostic([str(ROOT/".venv/bin/python"),str(Path(__file__).resolve()),"--child","--plan-sha256",args.plan_sha256],
        wall_seconds=POLICY["outer_seconds"],rss_bytes=POLICY["rss_bytes"],report_path=ATTEMPT/"step.resource.json",log_path=ATTEMPT/"step.log")
    result = {"execution_status":shared._execution_status(resource,wall_seconds=POLICY["outer_seconds"],rss_bytes=POLICY["rss_bytes"]),
        "resource":resource,"child_sha256":comparison._sha(ATTEMPT/"step.json") if (ATTEMPT/"step.json").exists() else None}
    tangent._write(ATTEMPT/"step.run.json",result); print(json.dumps(result))


if __name__ == "__main__":
    main()
