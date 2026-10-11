"""Bounded d=-P_Q continuation from the closed current tangent endpoint."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
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

from advar import transport  # noqa: E402
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry  # noqa: E402
from examples.weather_scenarios import fv_point_3h_gn_finite_continuation as finite  # noqa: E402
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared  # noqa: E402
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared  # noqa: E402
from examples.weather_scenarios import fv_point_3h_policy_search as search  # noqa: E402
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed  # noqa: E402
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent  # noqa: E402
from examples.weather_scenarios import fv_point_3h_current_tangent_search as frozen  # noqa: E402
from examples.weather_scenarios.fv_point_3h_policy_backend import FVBackend  # noqa: E402
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic  # noqa: E402

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PARENT_PLAN = EVIDENCE / "CURRENT_TANGENT_SEARCH_PLAN_20261011.json"
PARENT_PLAN_SHA = "82ba9f107658f4fd75016c434692073f31690c7fa94816a380e45146a7d9227b"
ARCHIVE = EVIDENCE / "CURRENT_TANGENT_ARCHIVE_20261011.json"
BASE_DIR = EVIDENCE / "current_tangent_search_20261011_attempt1"
BASE_GZIP = BASE_DIR / "step.json.gz"
BASE_RUN = BASE_DIR / "step.run.json"
BASE_RESOURCE = BASE_DIR / "step.resource.json"
BASE_SHA = "98ee63abf614756ddbdbba5cc480d66a3235a17b4def8c50b084c4f20dcdd957"
BASE_RAW_SHA = "1bffd3c6cf79edef1d393434585d3aebfcc631a2d3896175647f40d7d3a671a0"
PLAN = EVIDENCE / "GRADIENT_CONTINUATION_PLAN_20261011.json"
ATTEMPT = EVIDENCE / "current_gradient_continuation_20261011_attempt1"
SELF = "examples/weather_scenarios/fv_point_3h_current_gradient_continuation.py"
TEST = "tests/test_fv_point_3h_current_gradient_continuation.py"
FACE = {"axis": "y", "row": 4, "column": 3}
FACE_SCALE = 0.84
RADIUS = 0.05
MAX_COMMITS = 3
MAX_HVP = 6
MAX_CANDIDATES_PER_POINT = 24
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 2 * 1024**3, "max_commits": MAX_COMMITS,
    "max_candidates_per_point": MAX_CANDIDATES_PER_POINT, "max_candidates": 72,
    "hvp_calls": MAX_HVP, "hvp_calls_per_point": 2,
    "jacobian_row_vjp_calls": 0, "dense_solves": 0, "radius": RADIUS,
    "face_scale": FACE_SCALE, "candidate_policy": "dyadic_JR",
    "candidate_cap_per_point": MAX_CANDIDATES_PER_POINT,
    "residual_norm_candidate_tolerance": 1e-8,
    "root_claim": False, "minimum_claim": False, "response_claim": False,
    "score_claim": False,
}


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned path is not a regular in-repository file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("current-gradient plan identity mismatch")
    parent = frozen.load_plan(PARENT_PLAN, PARENT_PLAN_SHA)
    plan = json.loads(path.read_text())
    extra_archives = {PARENT_PLAN.relative_to(ROOT).as_posix(), ARCHIVE.relative_to(ROOT).as_posix(),
        BASE_GZIP.relative_to(ROOT).as_posix(), BASE_RUN.relative_to(ROOT).as_posix(),
        BASE_RESOURCE.relative_to(ROOT).as_posix()}
    source_names = set(parent["source_files"]) | {SELF, TEST}
    archive_names = set(parent["archive_files"]) | extra_archives
    if (plan.get("experiment_kind") != "current_face_gradient_continuation_from_closed_98"
            or plan.get("policy") != POLICY or plan.get("base_control_sha256") != BASE_SHA
            or set(plan.get("source_files", {})) != source_names
            or set(plan.get("archive_files", {})) != archive_names):
        raise ValueError("current-gradient plan scope or policy changed")
    for name, digest_value in parent["source_files"].items():
        if plan["source_files"].get(name) != digest_value:
            raise ValueError(f"inherited source pin changed: {name}")
    for name, digest_value in parent["archive_files"].items():
        if plan["archive_files"].get(name) != digest_value:
            raise ValueError(f"inherited archive pin changed: {name}")
    for name, digest_value in {**plan["source_files"], **plan["archive_files"]}.items():
        if _sha(ROOT / name) != digest_value:
            raise ValueError(f"successor source/archive pin mismatch: {name}")
    return plan


def validate_archived_endpoint(raw: dict[str, Any], run: dict[str, Any],
        resource: dict[str, Any], *, raw_sha: str, gzip_sha: str,
        expected_source: dict[str, str]) -> dict[str, Any]:
    """Admit the committed point from receipts alone; this function runs no FV."""
    control = torch.as_tensor(raw.get("current_control"), dtype=torch.float64)
    full_g = torch.as_tensor(raw.get("current_full_G"), dtype=torch.float64)
    repeat = raw.get("canonical_final_repeat")
    proposal = raw.get("canonical_proposal")
    if raw_sha != BASE_RAW_SHA:
        raise ValueError("archived raw endpoint digest mismatch")
    if (run.get("execution_status") != "completed" or run.get("child_sha256") != raw_sha
            or run.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != 660.0
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes", POLICY["rss_bytes"]) >= POLICY["rss_bytes"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or not 0.0 < float(resource["elapsed_seconds"]) <= POLICY["outer_seconds"]
            or raw.get("execution_status") != "completed"
            or raw.get("plan_sha256") != PARENT_PLAN_SHA
            or raw.get("numerical_status") != "one_conditional_current_tangent_candidate_committed"
            or raw.get("candidate_committed") is not True or raw.get("optimizer_steps_applied") != 1
            or raw.get("current_control_sha256") != BASE_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_SHA
            or tangent._tensor_sha(control) != BASE_SHA
            or raw.get("last_confirmed_control") != raw.get("current_control")
            or raw.get("source_before") != expected_source or raw.get("source_after") != expected_source
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True or raw.get("deadline_passed") is not True
            or raw.get("runtime_before") != raw.get("runtime_after")
            or not isinstance(repeat, dict) or not isinstance(proposal, dict)
            or raw.get("canonical_final_repeat") != repeat
            or not tangent.fresh_final_closure(proposal, repeat)
            or repeat.get("control") != raw.get("current_control")
            or repeat.get("theta") != raw.get("current_theta")
            or raw.get("current_full_G") != raw.get("last_confirmed_full_G")
            or full_g.shape != (27,) or not bool(torch.isfinite(full_g).all())):
        raise ValueError("current tangent archive endpoint/P2/source/resource closure failed")
    before_input, after_input = raw.get("input_before"), raw.get("input_after")
    stable_input_keys = ("parameters_sha256", "terminal_truth_sha256", "archived_input",
        "original_flow_fractions")
    if (not isinstance(before_input, dict) or not isinstance(after_input, dict)
            or not all(before_input.get(key) == after_input.get(key) for key in stable_input_keys)
            or after_input.get("control_sha256") != BASE_SHA):
        raise ValueError("archived current direct-source inputs do not close")
    native_g = float(torch.dot(full_g, full_g))
    residual_budget = 128 * torch.finfo(control.dtype).eps * max(
        abs(native_g), abs(float(repeat["F_squared"])), torch.finfo(control.dtype).tiny)
    objective_budget = 128 * torch.finfo(control.dtype).eps * max(
        abs(float(repeat["objective"])), abs(float(proposal["objective"])), torch.finfo(control.dtype).tiny)
    if (proposal.get("J_armijo_passed") is not True
            or proposal.get("F_squared_armijo_passed") is not True
            or abs(float(repeat["objective"]) - float(proposal["objective"])) > objective_budget
            or abs(float(repeat["F_squared"]) - float(proposal["F_squared"])) > residual_budget
            or abs(native_g - float(repeat["F_squared"])) > residual_budget
            or float(repeat["objective"]) > float(proposal["J_armijo_bound"])
            or float(repeat["F_squared"]) > float(proposal["F_squared_armijo_bound"])
            or not raw.get("final_full_G_check", {}).get("passed")):
        raise ValueError("archived endpoint does not close original J/R and full-G checks")
    side = repeat["side_gradients"]
    theta = float(repeat["theta"])
    mixed = ((1 - theta) * torch.as_tensor(side["-1"], dtype=control.dtype)
        + theta * torch.as_tensor(side["1"], dtype=control.dtype))
    budget = 128 * torch.finfo(control.dtype).eps * max(
        float(torch.linalg.vector_norm(full_g)), float(torch.linalg.vector_norm(mixed)),
        torch.finfo(control.dtype).tiny)
    recorded_full_g = torch.as_tensor(raw["final_full_G_check"].get("G"), dtype=control.dtype)
    if (not torch.allclose(full_g[:-1], mixed, rtol=0.0, atol=budget)
            or full_g[-1].abs() > budget or not torch.equal(recorded_full_g, full_g)):
        raise ValueError("archived full-G components do not reconstruct from the final P2")
    return {"raw": raw, "control": control, "theta": float(raw["current_theta"]),
        "full_G": full_g, "repeat": repeat, "input_after": raw["input_after"],
        "runtime_after": raw["runtime_after"]}


def load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    frozen.load_plan(PARENT_PLAN, PARENT_PLAN_SHA)
    archive_pins = plan["archive_files"]
    for path in (ARCHIVE, BASE_GZIP, BASE_RUN, BASE_RESOURCE):
        rel = path.relative_to(ROOT).as_posix()
        if archive_pins.get(rel) != _sha(path):
            raise ValueError(f"successor plan does not pin parent receipt: {rel}")
    archive_doc = json.loads(ARCHIVE.read_text())
    raw_bytes = gzip.decompress(BASE_GZIP.read_bytes())
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    gzip_sha = _sha(BASE_GZIP)
    run = json.loads(BASE_RUN.read_text())
    resource = json.loads(BASE_RESOURCE.read_text())
    raw = json.loads(raw_bytes)
    parent = json.loads(PARENT_PLAN.read_text())
    expected_source = {**parent["source_files"], **parent["archive_files"],
        PARENT_PLAN.relative_to(ROOT).as_posix(): PARENT_PLAN_SHA}
    manifest = archive_doc.get("manifest", {})
    for path, data in ((BASE_DIR / "step.json", raw_bytes), (BASE_RUN, BASE_RUN.read_bytes()),
            (BASE_RESOURCE, BASE_RESOURCE.read_bytes())):
        entry = manifest.get(str(path.relative_to(ROOT)))
        if not isinstance(entry, dict) or hashlib.sha256(data).hexdigest() != entry.get("sha256") or len(data) != entry.get("bytes"):
            raise ValueError(f"current tangent archived receipt mismatch: {path.name}")
    if (archive_doc.get("raw_sha256") != BASE_RAW_SHA
            or archive_doc.get("gzip_sha256") != gzip_sha
            or raw_sha != BASE_RAW_SHA or archive_doc.get("plan_sha256") != PARENT_PLAN_SHA):
        raise ValueError("current tangent archive identity changed")
    return validate_archived_endpoint(raw, run, resource, raw_sha=raw_sha,
        gzip_sha=gzip_sha, expected_source=expected_source)


def fresh_full_G(repeat: dict[str, Any], expected: Tensor, weights: Tensor) -> dict[str, Any]:
    control = torch.as_tensor(repeat["control"], dtype=torch.float64)
    minus = torch.as_tensor(repeat["side_gradients"]["-1"], dtype=control.dtype)
    plus = torch.as_tensor(repeat["side_gradients"]["1"], dtype=control.dtype)
    theta = float(repeat["theta"])
    actual = torch.cat(((1 - theta) * minus + theta * plus,
        (shared._face_value(control, weights) / FACE_SCALE).reshape(1)))
    match = finite._full_G_match(actual, expected)
    merit = float(torch.dot(actual, actual))
    budget = 128 * torch.finfo(actual.dtype).eps * max(abs(merit), abs(float(repeat["F_squared"])), torch.finfo(actual.dtype).tiny)
    return {"G": actual.tolist(), "component_match": match,
        "merit_match": abs(merit - float(repeat["F_squared"])) <= budget,
        "passed": match["passed"] is True and abs(merit - float(repeat["F_squared"])) <= budget}


def persist_commit(output: Path, prefix: dict[str, Any], point: dict[str, Any],
        proposal: dict[str, Any], repeat: dict[str, Any], full_g: dict[str, Any]) -> dict[str, Any]:
    if (not tangent.fresh_final_closure(proposal, repeat)
            or proposal.get("J_armijo_passed") is not True
            or proposal.get("F_squared_armijo_passed") is not True
            or full_g.get("passed") is not True
            or float(repeat["objective"]) > float(proposal["J_armijo_bound"])
            or float(repeat["F_squared"]) > float(proposal["F_squared_armijo_bound"])):
        raise ValueError("native J/R, P2 or full-G approval failed")
    result = copy.deepcopy(prefix)
    control = torch.as_tensor(repeat["control"], dtype=torch.float64)
    sha = tangent._tensor_sha(control)
    closed = {**point, "proposal": proposal, "final_repeat": repeat, "full_G_check": full_g}
    result["closed_steps"] = [*result.get("closed_steps", []), closed]
    result.update(current_control=control.tolist(), current_control_sha256=sha,
        current_theta=float(repeat["theta"]), current_full_G=full_g["G"],
        last_confirmed_control=control.tolist(), last_confirmed_control_sha256=sha,
        last_confirmed_theta=float(repeat["theta"]), last_confirmed_full_G=full_g["G"],
        candidate_committed=True, optimizer_steps_applied=len(result["closed_steps"]),
        postcommit_readiness_performed=False)
    tangent._write(output, result)
    return result


def classify_no_candidate(*, residual_norm: float, cap_hit: bool = False,
        unsupported: bool = False, budget_hit: bool = False) -> str:
    if residual_norm <= POLICY["residual_norm_candidate_tolerance"]:
        return "residual_candidate_requires_qualification"
    if budget_hit:
        return "budget_exhausted"
    if unsupported:
        return "unsupported_current_direction"
    return "candidate_cap_reached" if cap_hit else "search_exhausted"


class CurrentBackend(FVBackend):
    """Reuse evaluator/repeat/check only, with the archived 98ee endpoint."""
    # Deliberately skip FVBackend.__init__, which pins the older 68 endpoint.
    def __init__(self, plan: dict[str, Any], plan_path: Path, plan_sha: str,  # pyright: ignore[reportMissingSuperCall]
                 output_dir: Path, deadline: float, base: dict[str, Any]):
        self.plan, self.plan_path, self.plan_sha = plan, plan_path.resolve(), plan_sha
        self.output_dir, self.deadline = output_dir, deadline
        self.build_count = self.row_count = self.solve_count = self.hvp_count = 0
        self.timings = {key: 0.0 for key in ("identity_seconds", "own_write_seconds",
            "candidate_observe_seconds", "repeat_seconds", "row_solve_seconds_inclusive", "pure_hvp_seconds")}
        self.problem, self.original, _, self.parameters, self.truth, _ = seed._prepare_fixed_seed()
        self.parent = base
        self.control = base["control"].clone()
        self.input_anchor = shared._input_identity(self.problem, self.original, self.control, self.parameters, self.truth)
        self.runtime_anchor = shared._runtime()
        self.source_anchor = shared._source_hashes(plan, self.plan_path, plan_sha)
        self.weights = geometry._face_weights(self.problem, axis="y", row=4, column=3)
        if self.input_anchor != base["input_after"] or self.runtime_anchor != base["runtime_after"]:
            raise ValueError("current fixed inputs/runtime differ from the archived endpoint")
        self._check(self.control)


def match_previous_P2(observed: dict[str, Any], theta: float, full_g: Tensor,
        previous_g: Tensor, previous_repeat: dict[str, Any]) -> dict[str, Any]:
    component = finite._full_G_match(full_g, previous_g)
    previous_theta = float(previous_repeat["theta"])
    theta_budget = 128 * torch.finfo(full_g.dtype).eps * max(
        abs(previous_theta), abs(theta), torch.finfo(full_g.dtype).tiny)
    side_match = shared._gradient_pair_match(
        {s: observed["side"][s][1] for s in (-1, 1)},
        {s: torch.as_tensor(previous_repeat["side_gradients"][str(s)], dtype=full_g.dtype)
            for s in (-1, 1)})
    objective = float(observed["native_j"])
    previous_objective = float(previous_repeat["objective"])
    objective_budget = 128 * torch.finfo(full_g.dtype).eps * max(
        abs(objective), abs(previous_objective), torch.finfo(full_g.dtype).tiny)
    trace_match = all(observed["traces"][s]["signature_sha256"]
        == previous_repeat["branch_trace"][str(s)]["signature_sha256"] for s in (-1, 1))
    return {"component_match": component, "theta_error": abs(theta - previous_theta),
        "theta_budget": theta_budget, "side_gradient_match": side_match,
        "objective_error": abs(objective - previous_objective),
        "objective_budget": objective_budget, "trace_match": trace_match,
        "passed": component["passed"] is True and abs(theta - previous_theta) <= theta_budget
            and side_match is True and abs(objective - previous_objective) <= objective_budget
            and trace_match}


def native_face_release(gminus: Tensor, gplus: Tensor, normal: Tensor) -> dict[str, Any]:
    norm = torch.linalg.vector_norm(normal)
    if (not bool(torch.isfinite(norm)) or float(norm) <= torch.finfo(normal.dtype).tiny
            or not bool(torch.isfinite(gminus).all() and torch.isfinite(gplus).all())):
        raise ValueError("native outward derivative inputs are unresolved")
    unit_normal = normal / norm
    minus = torch.dot(gminus, -unit_normal)
    plus = torch.dot(gplus, unit_normal)
    minus_budget = 128 * torch.finfo(gminus.dtype).eps * max(
        float(torch.linalg.vector_norm(gminus)), torch.finfo(gminus.dtype).tiny)
    plus_budget = 128 * torch.finfo(gplus.dtype).eps * max(
        float(torch.linalg.vector_norm(gplus)), torch.finfo(gplus.dtype).tiny)
    minus_resolved = float(minus) < -minus_budget
    plus_resolved = float(plus) < -plus_budget
    return {"gminus_dot_minus_n": float(minus), "gplus_dot_plus_n": float(plus),
        "gminus_budget": minus_budget, "gplus_budget": plus_budget,
        "release_required": minus_resolved or plus_resolved}


def _run_child(path: Path, digest: str, output: Path) -> dict[str, Any]:
    start = time.monotonic()
    plan = load_plan(path, digest)
    base = load_current_base(plan)
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
        "base_future_score": raw0.get("final_future_score"),
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
            theta_t = tangent.minimum_mixture_weight(gminus, gplus)
            theta = float(theta_t)
            G = torch.cat(((1 - theta_t) * gminus + theta_t * gplus,
                (observed["q"] / FACE_SCALE).reshape(1)))
            previous_repeat = (base["repeat"] if point_index == 0
                else record["closed_steps"][-1]["final_repeat"])
            previous_G = torch.as_tensor(record["current_full_G"], dtype=c.dtype)
            p2_match = match_previous_P2(observed, theta, G, previous_G, previous_repeat)
            if not p2_match["passed"]:
                raise ValueError("fresh current point differs from its previous closed P2")
            normal = geometry._face_normal(c, backend.weights)
            pivot = geometry._pivot(c, backend.weights)
            _, chart, _ = geometry._chart_jacobian(c, backend.weights, pivot, 0.0)
            _, _, direction, direction_audit = tangent.tangent_direction(
                gminus, gplus, normal, chart, theta, pivot=pivot)
            outward = native_face_release(gminus, gplus, normal)
            point: dict[str, Any] = {"point_index": point_index, "base_control_sha256": tangent._tensor_sha(c),
                "theta": theta, "J": float(observed["native_j"]), "R": float(torch.dot(G, G)),
                "full_G": G.tolist(), "direction": direction.tolist(),
                "direction_sha256": tangent._tensor_sha(direction), "direction_audit": direction_audit,
                "previous_P2_match": p2_match,
                "outward_derivatives": outward, "hvp_started": 0, "hvp_completed": 0}
            record["current_point"] = point
            record.update(current_control=c.tolist(), current_control_sha256=tangent._tensor_sha(c),
                current_theta=theta, current_full_G=G.tolist())
            write()
            if not (observed["pair"]["passed"] and observed["face_ok"]
                    and observed["side_objectives_match_native"] and observed["side_gradients_finite"]):
                return finish("unsupported_current_native_pair")
            if outward["release_required"]:
                return finish("native_face_release_required_no_Q0_commit")
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
            proposal.update(selected_direction="d=-P_Q minimum mixed-gradient",
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
        raise ValueError("continuation attempt must be fresh")
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
