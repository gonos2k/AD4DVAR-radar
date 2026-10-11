"""Fresh gradient continuation from 556a with P2-first face-release decisions.

The frozen predecessor remains replayable. This successor uses the same
conditional Q=0 direction and J/R acceptance, with no general face-exit solver.
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

from examples.weather_scenarios import fv_point_3h_current_gradient_continuation as previous
from examples.weather_scenarios import fv_point_3h_current_gradient_endpoint as endpoint

transport = previous.transport
geometry, finite, prepared = previous.geometry, previous.finite, previous.prepared
shared, search, tangent = previous.shared, previous.search, previous.tangent
run_guarded_diagnostic = previous.run_guarded_diagnostic
CurrentBackend = previous.CurrentBackend
match_previous_P2 = previous.match_previous_P2
native_face_release = previous.native_face_release
fresh_full_G, persist_commit = previous.fresh_full_G, previous.persist_commit
classify_no_candidate = previous.classify_no_candidate
_sha = previous._sha

EVIDENCE = previous.EVIDENCE
PLAN = EVIDENCE / "GRADIENT_RESUME_PLAN_20261011.json"
ATTEMPT = EVIDENCE / "current_gradient_resume_20261011_attempt1"
BASE_SHA = "556a25d910cad707b108cc8c6770af9ebf6b34f5a84196736e9fee82ed298a8b"
SELF = "examples/weather_scenarios/fv_point_3h_current_gradient_resume.py"
TEST = "tests/test_fv_point_3h_current_gradient_resume.py"
NEW_SOURCES = {SELF, TEST,
    "examples/weather_scenarios/fv_point_3h_current_gradient_endpoint.py",
    "tests/test_fv_point_3h_current_gradient_endpoint.py"}
FACE_SCALE, RADIUS = previous.FACE_SCALE, previous.RADIUS
MAX_COMMITS, MAX_HVP = previous.MAX_COMMITS, previous.MAX_HVP
MAX_CANDIDATES_PER_POINT = previous.MAX_CANDIDATES_PER_POINT
POLICY: dict[str, Any] = {**previous.POLICY,
    "qualification_order": "native_pair_then_previous_P2_then_release_then_minimum_mixture"}


def load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("gradient-resume plan identity mismatch")
    base = endpoint.load_endpoint()
    parent = base["parent_plan"]
    plan = json.loads(path.read_text())
    source_names = set(parent["source_files"]) | NEW_SOURCES
    archive_names = set(parent["archive_files"]) | set(base["receipt_files"])
    if (plan.get("experiment_kind") != "current_gradient_resume_from_closed_556a"
            or plan.get("policy") != POLICY or plan.get("base_control_sha256") != BASE_SHA
            or set(plan.get("source_files", {})) != source_names
            or set(plan.get("archive_files", {})) != archive_names):
        raise ValueError("gradient-resume plan scope or policy changed")
    for group in ("source_files", "archive_files"):
        for name, pinned in parent[group].items():
            if plan[group].get(name) != pinned:
                raise ValueError(f"inherited pin changed: {name}")
    for name, pinned in base["receipt_files"].items():
        if plan["archive_files"].get(name) != pinned:
            raise ValueError(f"endpoint receipt pin changed: {name}")
    for name, pinned in {**plan["source_files"], **plan["archive_files"]}.items():
        if _sha(ROOT / name) != pinned:
            raise ValueError(f"gradient-resume source/archive changed: {name}")
    return plan


def _run_child(path: Path, digest: str, output: Path) -> dict[str, Any]:
    start = time.monotonic()
    plan = load_plan(path, digest)
    base = endpoint.load_endpoint()
    deadline = start + POLICY["internal_seconds"]
    backend = CurrentBackend(plan, path, digest, ATTEMPT / "operators", deadline, base)
    raw0 = base["raw"]
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
        "original_problem_solved": False, "Q0_scope": "conditional hypothesis; not physical constraint",
        "base_future_score": raw0.get("terminal_future_score"),
        "base_future_score_source": "explicit_cached_parent_terminal_E; no reranking"}
    def write() -> None:
        tangent._write(output, record)
    write()
    record.update(source_before=backend.source_anchor, input_before=backend.input_anchor,
        runtime_before=backend.runtime_anchor)

    terminal_score_written = False

    def finish(status: str, reason: str = "") -> dict[str, Any]:
        nonlocal terminal_score_written
        c = torch.as_tensor(record["current_control"], dtype=torch.float64)
        if not terminal_score_written:
            if record["optimizer_steps_applied"] == 0:
                score = record["base_future_score"]
                score_source = "explicit_cached_parent_terminal_E; no reranking"
            else:
                try:
                    backend._check(c)
                    with torch.no_grad():
                        score = float(backend.problem.score(c, backend.parameters))
                    if not math.isfinite(score):
                        score = None
                except Exception as error:
                    score = None
                    record["terminal_future_score_error"] = f"{type(error).__name__}: {error}"
                score_source = "matched-twin terminal E at final closed point; observation only"
            record["terminal_future_score"] = score
            record["terminal_future_score_scope"] = score_source
            terminal_score_written = True
        inp = shared._input_identity(backend.problem, backend.original, c, backend.parameters, backend.truth)
        source = shared._source_hashes(plan, path.resolve(), digest)
        runtime = shared._runtime()
        input_ok = geometry.model._fixed_input(inp, backend.input_anchor, tangent._tensor_sha(c))
        record.update(phase="finished", execution_status="completed", numerical_status=status,
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
            _, _, direction, direction_audit = tangent.tangent_direction(
                gminus, gplus, normal, chart, theta, pivot=pivot)
            point.update(theta=theta, J=float(observed["native_j"]), R=float(torch.dot(G, G)),
                full_G=G.tolist(), direction=direction.tolist(),
                direction_sha256=tangent._tensor_sha(direction), direction_audit=direction_audit)
            write()
            residual_norm = float(torch.linalg.vector_norm(G))
            if residual_norm <= POLICY["residual_norm_candidate_tolerance"]:
                return finish("residual_candidate_requires_qualification")
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
            proposal, repeat = backend.repeat(bundle, selected)
            proposal.update(selected_arm="current_tangent_gradient", selected_direction="d=-P_Q minimum mixed-gradient",
                acceptance_policy="original_JR_conditional_Q0", conditional_Q0=True)
            check = fresh_full_G(repeat, torch.as_tensor(selected["G"], dtype=c.dtype), backend.weights)
            point["selected_candidate"] = selected
            record = persist_commit(output, record, point, proposal, repeat, check)
            record["postcommit_readiness_performed"] = False
            backend.control = torch.as_tensor(record["current_control"], dtype=torch.float64)
        return finish("commit_cap_reached")
    except TimeoutError as error:
        return finish("budget_exhausted", str(error))
    except Exception as error:
        record["execution_status"] = "failed"
        record["termination_reason"] = str(error)
        write()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(PLAN, args.plan_sha256, ATTEMPT / "step.json")
        return
    load_plan(PLAN, args.plan_sha256)
    if ATTEMPT.exists():
        raise ValueError("gradient-resume attempt must be fresh")
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
