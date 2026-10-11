"""Fixed-version continuous current-gradient trial from the admitted net-flux endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as previous
from examples.weather_scenarios import fv_point_3h_netflux_endpoint as endpoint

transport, geometry, finite, prepared = previous.transport, previous.geometry, previous.finite, previous.prepared
shared, search, tangent = previous.shared, previous.search, previous.tangent
match_previous_P2, native_face_release = previous.match_previous_P2, previous.native_face_release
fresh_full_G, persist_commit = previous.fresh_full_G, previous.persist_commit
classify_no_candidate, _sha = previous.classify_no_candidate, previous._sha
run_guarded_diagnostic = previous.run_guarded_diagnostic
Tensor = torch.Tensor
CurrentBackend = previous.CurrentBackend
VERSION, FACE_SCALE, RADIUS = previous.VERSION, previous.FACE_SCALE, previous.RADIUS
WORK = previous.EVIDENCE / "netflux_continuous_20261012"
PLAN, ATTEMPT = WORK / "PLAN.json", WORK / "attempt1"
BASE_SHA = endpoint.BASE_SHA
PARENT_PLAN, PARENT_SHA = previous.PLAN, "7f26c76bad511e1345c2d3c27b2073908c574587fc3aefd433fb44b98a932441"
MAX_COMMITS, MAX_HVP, MAX_CANDIDATES_PER_POINT = 20, 40, 24
SELF = "examples/weather_scenarios/fv_point_3h_netflux_continuation.py"
NEW_SOURCES = {SELF, "tests/test_fv_point_3h_netflux_continuation.py",
    "examples/weather_scenarios/fv_point_3h_netflux_endpoint.py",
    "tests/test_fv_point_3h_netflux_endpoint.py"}
POLICY: dict[str, Any] = {**previous.POLICY, "max_commits": MAX_COMMITS,
    "hvp_calls": MAX_HVP, "max_candidates": MAX_COMMITS * MAX_CANDIDATES_PER_POINT,
    "max_candidates_per_point": MAX_CANDIDATES_PER_POINT,
    "base_qualification": "same_version_receipt_endpoint_then_fresh_current_P2",
    "stagnation_window": 3, "stagnation_relative_decrease": 0.003,
    "stagnation_action": "stop_for_same_point_direction_comparison_on_small_progress_or_regression"}


def load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("continuous plan identity mismatch")
    base = endpoint.load_endpoint()
    parent = base["parent_plan"]
    plan = json.loads(path.read_text())
    archives = set(parent["archive_files"]) | {p.relative_to(ROOT).as_posix() for p in endpoint.RECEIPTS}
    if (_sha(PARENT_PLAN) != PARENT_SHA
            or plan.get("experiment_kind") != "netflux_fixed_version_continuous_gradient"
            or plan.get("numerical_version") != VERSION
            or plan.get("policy") != POLICY or plan.get("base_control_sha256") != BASE_SHA
            or set(plan.get("source_files", {})) != set(parent["source_files"]) | NEW_SOURCES
            or set(plan.get("archive_files", {})) != archives):
        raise ValueError("continuous plan changed input, version, sources, or policy")
    for name, pinned in parent["source_files"].items():
        if plan["source_files"].get(name) != pinned:
            raise ValueError(f"fixed numerical source changed: {name}")
    for name, pinned in parent["archive_files"].items():
        if plan["archive_files"].get(name) != pinned:
            raise ValueError(f"inherited archive changed: {name}")
    for name, pinned in {**plan["source_files"], **plan["archive_files"]}.items():
        if _sha(ROOT / name) != pinned:
            raise ValueError(f"continuous source/archive pin mismatch: {name}")
    return plan


def convergence_progress(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """A small three-step decrease is a comparison trigger, never a root claim."""
    history = [{"point_index": step["point_index"],
        "control_sha256": step["proposal"]["control_sha256"],
        "J": float(step["final_repeat"]["objective"]),
        "R": float(step["final_repeat"]["F_squared"]),
        "mixed_gradient_max": max(abs(x) for x in step["full_G_check"]["G"][:26])}
        for step in steps]
    window = int(POLICY["stagnation_window"])
    result: dict[str, Any] = {"history": history, "stagnation_detected": False, "no_progress_detected": False, "window_metric_regression": False, "direction_comparison_required": False,
        "window": window, "relative_decrease_threshold": POLICY["stagnation_relative_decrease"]}
    if len(steps) >= window:
        initial = steps[-window]
        final = history[-1]
        initial_max = max(abs(x) for x in initial["full_G"][:26])
        decreases = {"R": (float(initial["R"]) - final["R"]) / float(initial["R"]),
            "mixed_gradient_max": (initial_max - final["mixed_gradient_max"]) / initial_max}
        result.update(window_relative_decrease=decreases,
            stagnation_detected=all(0 <= v <= POLICY["stagnation_relative_decrease"] for v in decreases.values()),
            no_progress_detected=all(v <= POLICY["stagnation_relative_decrease"] for v in decreases.values()),
            window_metric_regression=any(v < 0 for v in decreases.values()),
            direction_comparison_required=(any(v < 0 for v in decreases.values())
                or all(0 <= v <= POLICY["stagnation_relative_decrease"] for v in decreases.values())))
    return result


def _run_child(path: Path, digest: str, output: Path) -> dict[str, Any]:
    start = time.monotonic()
    plan = load_plan(path, digest)
    base: dict[str, Any] = endpoint.load_endpoint()
    deadline = start + POLICY["internal_seconds"]
    backend = CurrentBackend(plan, path, digest, ATTEMPT / "operators", deadline, base)
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "preflight", "plan_sha256": digest, "policy": POLICY,
        "physical_problem": "original_J_local_minimization",
        "search_domain": "selected_Q0_conditional_continuation",
        "base_control_sha256": BASE_SHA,
        "current_control": base["control"].tolist(), "current_control_sha256": BASE_SHA,
        "current_theta": base["theta"], "current_full_G": base["full_G"].tolist(),
        "last_confirmed_control": base["control"].tolist(), "last_confirmed_control_sha256": BASE_SHA,
        "last_confirmed_theta": base["theta"], "last_confirmed_full_G": base["full_G"].tolist(),
        "closed_steps": [], "candidate_committed": False, "optimizer_steps_applied": 0,
        "candidate_evaluations_started": 0, "candidate_evaluations_completed": 0,
        "hvp_started": 0, "hvp_completed": 0, "root_claim": False,
        "minimum_claim": False, "response_claim": False, "score_claim": False,
        "original_problem_solved": False, "numerical_version": VERSION,
        "base_endpoint_admitted": True, "last_confirmed_numerical_version": VERSION, "Q0_scope": "conditional hypothesis; not physical constraint",
        "base_future_score": None,
        "base_future_score_source": "not evaluated; no new-version E baseline"}
    def write() -> None:
        tangent._write(output, record)
    write()
    record.update(source_before=backend.source_anchor, input_before=backend.input_anchor,
        runtime_before=backend.runtime_anchor)

    stage = "same_version_current_P2"

    def finish(status: str, reason: str = "", *, failed: bool = False) -> dict[str, Any]:
        c = torch.as_tensor(record["current_control"], dtype=torch.float64)
        record["terminal_E_recomputed"] = False
        inp = shared._input_identity(backend.problem, backend.original, c, backend.parameters, backend.truth)
        source = shared._source_hashes(plan, path.resolve(), digest)
        runtime = shared._runtime()
        input_ok = geometry.model._fixed_input(inp, backend.input_anchor, tangent._tensor_sha(c))
        record.update(phase="finished", execution_status="failed" if failed else "completed", numerical_status=status,
            failure_stage=stage if failed else None,
            termination_reason=reason, source_after=source, source_unchanged=source == backend.source_anchor,
            input_after=inp, fixed_input_unchanged=input_ok, runtime_after=runtime,
            runtime_unchanged=runtime == backend.runtime_anchor,
            deadline_passed=time.monotonic() < deadline, elapsed_seconds=time.monotonic() - start,
            backend_timing=backend.timings)
        write()
        if not (record["source_unchanged"] and input_ok and record["runtime_unchanged"]):
            raise ValueError("continuation source/input/runtime changed")
        return record

    try:
        for point_index in range(MAX_COMMITS):
            stage = "fresh_current_direction"
            record["numerical_status"] = stage
            write()
            c = backend.control
            backend._check(c)
            observed = tangent._observe(shared, backend.problem, c, backend.parameters, backend.weights)
            gminus, gplus = observed["side"][-1][1], observed["side"][1][1]
            previous_repeat = (base["repeat"] if point_index == 0
                else record["closed_steps"][-1]["final_repeat"])
            previous_G = torch.as_tensor(record["current_full_G"], dtype=c.dtype)
            point = {"point_index": point_index,
                "base_control_sha256": tangent._tensor_sha(c),
                "hvp_started": 0, "hvp_completed": 0}
            record["current_point"] = point
            write()
            if not (observed["pair"]["passed"] and observed["face_ok"]
                    and observed["side_objectives_match_native"] and observed["side_gradients_finite"]):
                return finish("unsupported_current_native_pair")
            # Reconstruct with the previously certified theta before minimizing again.
            # A changed P2 is an error, even if the new gradients suggest release.
            previous_theta = float(previous_repeat["theta"])
            stored_G = torch.cat(((1 - previous_theta) * gminus + previous_theta * gplus,
                (observed["q"] / FACE_SCALE).reshape(1)))
            p2_match = match_previous_P2(observed, previous_theta, stored_G, previous_G, previous_repeat)
            point["previous_P2_match"] = p2_match
            write()
            if not p2_match["passed"]:
                raise ValueError("fresh current point differs from its previous closed P2")
            normal = geometry._face_normal(c, backend.weights)
            outward = native_face_release(gminus, gplus, normal)
            point["outward_derivatives"] = outward
            write()
            if outward["release_required"]:
                return finish("native_face_release_required_no_Q0_commit")
            try:
                theta_t = tangent.minimum_mixture_weight(gminus, gplus)
            except ValueError as error:
                return finish("unsupported_current_mixture", str(error))
            theta = float(theta_t)
            G = torch.cat(((1 - theta_t) * gminus + theta_t * gplus,
                (observed["q"] / FACE_SCALE).reshape(1)))
            mixture_match = match_previous_P2(observed, theta, G, previous_G, previous_repeat)
            point["minimum_mixture_P2_match"] = mixture_match
            if not mixture_match["passed"]:
                raise ValueError("fresh minimum mixture differs from its previous closed P2")
            pivot = geometry._pivot(c, backend.weights)
            _, chart, _ = geometry._chart_jacobian(c, backend.weights, pivot, 0.0)
            jump = gplus - gminus
            unit = normal / torch.linalg.vector_norm(normal)
            support_gap = jump - unit * torch.dot(unit, jump)
            point["common_face_measurement"] = {
                "tangent_jump_norm": float(torch.linalg.vector_norm(support_gap)),
                "budget": tangent._roundoff(c.dtype, float(gminus.norm()), float(gplus.norm()), float(jump.norm()))}
            write()
            _, _, direction, direction_audit = tangent.tangent_direction(
                gminus, gplus, normal, chart, theta, pivot=pivot)
            point.update(theta=theta, J=float(observed["native_j"]), R=float(torch.dot(G, G)),
                full_G=G.tolist(), direction=direction.tolist(),
                direction_sha256=tangent._tensor_sha(direction), direction_audit=direction_audit)
            write()
            residual_norm = float(torch.linalg.vector_norm(G))
            if residual_norm <= POLICY["residual_norm_candidate_tolerance"]:
                return finish("residual_candidate_requires_qualification")
            stage = "fresh_current_direction_HVP"
            record["numerical_status"] = stage
            write()
            products: dict[int, Tensor] = {}
            grad_fn = torch.func.grad(backend.problem.objective, argnums=0)
            for side in (-1, 1):
                if record["hvp_started"] >= MAX_HVP or point["hvp_started"] >= 2:
                    return finish("budget_exhausted", "fresh HVP cap reached")
                record["hvp_started"] += 1
                point["hvp_started"] += 1
                write()
                backend._check(c)
                with transport.selected_face_extension("y", 4, 3, side):
                    products[side] = torch.func.jvp(lambda x: grad_fn(x, backend.parameters), (c,), (direction,))[1]
                backend._check(c)
                if (products[side].shape != c.shape or products[side].dtype != c.dtype
                        or products[side].device != c.device
                        or not bool(torch.isfinite(products[side]).all())):
                    raise ValueError("fresh current-point HVP is malformed or nonfinite")
                record["hvp_completed"] += 1
                point["hvp_completed"] += 1
                backend.hvp_count += 1
                write()
            model = prepared.mixing.minimum_tangent_model(gminus, gplus,
                products[-1], products[1], normal, chart, theta, float(observed["q"]),
                pivot, FACE_SCALE, direction_override=direction)
            if not finite._base_model_closure(model, G, theta)["passed"]:
                raise ValueError("current point model does not reproduce the fresh full G")
            point.update(side_HVP={str(s): products[s].tolist() for s in (-1, 1)},
                model_gates=model.gates, merit_product=model.merit_product)
            if not all(model.gates.get(k) is True for k in
                    ("finite", "both_side_gradients_descend", "merit_descends", "envelope_slope_matches_residual_dot")):
                return finish("unsupported_current_direction")
            bundle = {"observed": observed, "theta": theta, "G": G,
                "R": float(torch.dot(G, G)), "J": float(observed["native_j"]),
                "direction": direction, "model": model, "pivot": pivot,
                "cost_slopes": model.side_products}
            receipts: dict[float, dict[str, Any]] = {}

            def evaluate(candidate: Tensor, _theta: float, alpha: float) -> dict[str, Any]:
                record["candidate_evaluations_started"] += 1
                write()
                facts = backend.evaluate(bundle, candidate, alpha)
                record["candidate_evaluations_completed"] += 1
                receipt_path = ATTEMPT / f"point_{point_index:02d}" / f"candidate_{record['candidate_evaluations_completed']:02d}.json"
                tangent._write(receipt_path, facts)
                receipts[alpha] = {"facts": facts, "path": receipt_path, "sha256": _sha(receipt_path)}
                compact = {key: value for key, value in facts.items() if key != "branch_trace"}
                compact.update(candidate_receipt=str(receipt_path.relative_to(ROOT)),
                    candidate_receipt_sha256=receipts[alpha]["sha256"])
                return compact

            def progress(trials: list[dict[str, Any]], meta: dict[str, Any]) -> None:
                point["trials"], point["search"] = trials, meta
                write()

            stage = "actual_JR_candidate_search"
            record["numerical_status"] = stage
            write()
            selected, trials, meta = search.search(model, c, evaluate,
                chart_candidate=lambda x, d, a: backend.chart(bundle, x, d, a),
                progress=progress, policy={"name": "dyadic_JR", "max_evaluations": 24,
                    "max_backtracks": 24, "radius": RADIUS})
            point.update(trials=trials, search=meta)
            if selected is None:
                if residual_norm <= POLICY["residual_norm_candidate_tolerance"]:
                    return finish("residual_candidate_requires_qualification")
                status = classify_no_candidate(residual_norm=residual_norm,
                    cap_hit=len(trials) >= MAX_CANDIDATES_PER_POINT,
                    unsupported=not all(model.gates.get(k) is True for k in
                        ("finite", "both_side_gradients_descend", "merit_descends", "envelope_slope_matches_residual_dot")),
                    budget_hit=time.monotonic() >= deadline)
                return finish(status)
            receipt = receipts[selected["alpha"]]
            if _sha(receipt["path"]) != receipt["sha256"]:
                raise ValueError("selected candidate receipt changed before P2 repeat")
            selected["branch_trace"] = receipt["facts"]["branch_trace"]
            stage = "selected_candidate_P2_full_G"
            record["numerical_status"] = stage
            write()
            proposal, repeat = backend.repeat(bundle, selected)
            proposal.update(selected_arm="current_tangent_gradient", selected_direction="d=-P_Q minimum mixed-gradient",
                acceptance_policy="original_JR_conditional_Q0", conditional_Q0=True)
            check = fresh_full_G(repeat, torch.as_tensor(selected["G"], dtype=c.dtype), backend.weights)
            point["selected_candidate"] = selected
            record = persist_commit(output, record, point, proposal, repeat, check)
            record["postcommit_readiness_performed"] = False
            record["last_confirmed_numerical_version"] = VERSION
            backend.control = torch.as_tensor(record["current_control"], dtype=torch.float64)
            stage = "postcommit_progress_classification"
            record["convergence_progress"] = convergence_progress(record["closed_steps"])
            write()
            if float(torch.linalg.vector_norm(torch.as_tensor(record["current_full_G"], dtype=torch.float64))) <= POLICY["residual_norm_candidate_tolerance"]:
                return finish("residual_candidate_requires_qualification")
            if record["convergence_progress"]["direction_comparison_required"]:
                status = ("rolling_metric_regression_requires_same_point_direction_comparison"
                    if record["convergence_progress"]["window_metric_regression"]
                    else "rolling_stagnation_requires_same_point_direction_comparison")
                return finish(status)
        return finish("commit_cap_reached")
    except TimeoutError as error:
        return finish("budget_exhausted", str(error))
    except Exception as error:
        status = ("native_face_release_required_no_Q0_commit"
            if "native face release required" in str(error) else "current_support_or_contract_failure")
        return finish(status, f"{type(error).__name__}: {error}", failed=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        result = _run_child(PLAN, args.plan_sha256, ATTEMPT / "step.json")
        if result["execution_status"] == "failed":
            raise SystemExit(1)
        return
    load_plan(PLAN, args.plan_sha256)
    if ATTEMPT.exists():
        raise ValueError("net-flux continuous attempt must be fresh")
    ATTEMPT.mkdir()
    resource = run_guarded_diagnostic([str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
        "--child", "--plan-sha256", args.plan_sha256], wall_seconds=POLICY["outer_seconds"],
        rss_bytes=POLICY["rss_bytes"], report_path=ATTEMPT / "step.resource.json",
        log_path=ATTEMPT / "step.log")
    result = {"execution_status": shared._execution_status(resource,
        wall_seconds=POLICY["outer_seconds"], rss_bytes=POLICY["rss_bytes"]),
        "resource": resource, "child_sha256": _sha(ATTEMPT / "step.json") if (ATTEMPT / "step.json").exists() else None}
    tangent._write(ATTEMPT / "step.run.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
