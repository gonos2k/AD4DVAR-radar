"""Compare nonnegative and signed coefficient spans using sealed same-point HVPs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
RUN = ROOT / "graphify-out/fv-root-cause-20260919/residual_direction_20261012"
PARENT_PLAN = RUN / "PLAN.json"
PARENT_PLAN_SHA = "cd9db66869652c8807264dcdb9172f94e4c65118fc3982465e516e706502fa64"
PARENT_SOURCE_TAR = RUN / "source.tar.gz"
PARENT_SOURCE_TAR_SHA = "da936a87c96bcb5970afbed650ba00760949763652f77febfebdd9b44d6ac2b8"
PARENT_RECEIPT_MANIFEST_SHA = "c7f55b3b44b83c895f8340c22181546cb307333f56f79ea0f58df380bdc29263"
PARENT_RECEIPTS_TAR_SHA = "5fae12a5f5a8a39c1556a6e62180c399d654f837882026f650bd2c2cb7b1aa11"
PARENT_FINAL_CONTROL_SHA = "fb0a2efaeefe3a8b07a9385a81260757e4923b181a9430c384cfb96b8d8ea7b4"
BASE_SHA = "2a4dba130d28273a63e97d0de39a11f64fdee09b0ad754559bc8902ba6cddbca"
RADIUS = 3.751435583391765e-5
PRIOR_GUARDED_SECONDS = 54.004789291881025
PLAN = WORK / "PLAN.json"
ATTEMPT = WORK / "attempt1"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK))
import audit_stored_model as stored  # noqa: E402
from examples.weather_scenarios import fv_point_3h_netflux_subspace_trial as trial  # noqa: E402
from examples.weather_scenarios import fv_point_3h_residual_direction as residual  # noqa: E402
from examples.weather_scenarios.fv_point_3h_subspace_model import (  # noqa: E402
    SubspaceModel, solve_two_direction_subspace,
)

_endpoint_path = RUN / "endpoint.py"
_endpoint_spec = importlib.util.spec_from_file_location("signed_span_base_endpoint", _endpoint_path)
if _endpoint_spec is None or _endpoint_spec.loader is None:
    raise ImportError(f"could not load frozen base endpoint: {_endpoint_path}")
endpoint = importlib.util.module_from_spec(_endpoint_spec)
sys.modules[_endpoint_spec.name] = endpoint
_endpoint_spec.loader.exec_module(endpoint)

VERSION = trial.VERSION
DIRECTION_ORDER = ("baseline_tangent", "residual_merit")
ARM_ORDER = ("baseline_tangent", "two_direction_subspace")
CHANGED_SOURCES = {
    "examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py",
    "examples/weather_scenarios/fv_point_3h_residual_direction.py",
    "tests/test_fv_point_3h_residual_direction.py",
    "tests/test_fv_point_3h_subspace_model.py",
}
NEW_SOURCES = {
    str((WORK / name).relative_to(ROOT))
    for name in ("signed_trial.py", "audit_stored_model.py")
} | {"tests/test_fv_point_3h_signed_span_trial.py"}
NEW_ARCHIVES = {
    str((RUN / name).relative_to(ROOT))
    for name in ("PLAN.json", "source.tar.gz", "RECEIPT_MANIFEST.json", "receipts.tar.gz")
}
POLICY = {
    **trial.POLICY,
    "experiment_kind": "cached_2a4d_signed_span_vs_nonnegative_span",
    "arms": list(ARM_ORDER),
    "direction_columns": list(DIRECTION_ORDER),
    "selection": "actual repeated J/R; both arms required; no E",
    "radius": RADIUS,
    "candidate_policy": "baseline ray versus four signed coefficient quadrants at the same radius",
    "coefficient_solve": "existing trust-region solver applied to each sign-transformed D/V pair",
    "coefficient_solves": "at most 25 tiny cone/quadrant solves including preflight; no additional HVP",
    "cached_HVP_pair_count": 4,
    "new_hvp_cap": 0,
    "jacobian_row_vjp_calls": 0,
    "dense_solves": 0,
    "max_candidates": 12,
    "max_candidates_per_arm": 6,
    "candidate_cap_per_point": 6,
    "max_commits": 1,
    "prior_guarded_seconds_charged": PRIOR_GUARDED_SECONDS,
    "internal_seconds": 600.0 - PRIOR_GUARDED_SECONDS,
    "outer_seconds": 660.0 - PRIOR_GUARDED_SECONDS,
    "budget_scope": "previous 54.004789s guarded PR304 run charged once plus this cached-HVP comparison; total caps 600/660s, 2GiB",
    "signed_preflight": "same-radius predicted-R gain above FP64 budget and both base side costs strictly negative, otherwise no candidates",
    "response_claim": False,
    "root_claim": False,
    "minimum_claim": False,
    "score_claim": False,
    "future_score_evaluations": 0,
}

_CACHE: dict[str, Any] | None = None
_CONTEXT: dict[str, Any] | None = None
_SIGNED_RESULT: SubspaceModel | None = None
_SIGNED_SIGNATURE: tuple[str, str, str, float] | None = None
_SIGNED_PLAN_SHA: str | None = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device or not torch.equal(a, b):
        raise ValueError(f"cached signed-span {label} differs from this current point")


def _ensure_cache() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = stored.load_stored_model()
    return _CACHE


def load_plan(path: Path, digest: str) -> dict[str, Any]:
    global _SIGNED_PLAN_SHA
    if path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("signed-span plan identity mismatch")
    parent_bytes = stored.read_pinned(PARENT_PLAN, PARENT_PLAN_SHA)
    parent = json.loads(parent_bytes)
    plan = json.loads(path.read_text())
    expected_transition = {
        name: {"before": parent["source_files"][name], "after": _sha(ROOT / name)}
        for name in sorted(CHANGED_SOURCES)
    }
    expected_source_set = set(parent["source_files"]) | NEW_SOURCES
    expected_archive_set = set(parent["archive_files"]) | NEW_ARCHIVES
    if (plan.get("experiment_kind") != "cached_2a4d_signed_span_vs_nonnegative_span"
            or plan.get("policy") != POLICY or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("numerical_version") != VERSION
            or plan.get("source_transition") != expected_transition
            or set(plan.get("source_files", {})) != expected_source_set
            or set(plan.get("archive_files", {})) != expected_archive_set
            or plan.get("parent_plan_sha256") != PARENT_PLAN_SHA
            or plan.get("parent_source_bundle_sha256") != PARENT_SOURCE_TAR_SHA
            or plan.get("parent_receipt_bundle_sha256") != PARENT_RECEIPTS_TAR_SHA):
        raise ValueError("signed-span source, archive, parent, or policy scope changed")
    for name, expected in parent["source_files"].items():
        if name not in CHANGED_SOURCES and plan["source_files"].get(name) != expected:
            raise ValueError(f"unapproved inherited numerical source change: {name}")
    for name, expected in parent["archive_files"].items():
        if plan["archive_files"].get(name) != expected:
            raise ValueError(f"inherited frozen archive changed: {name}")
    for group in ("source_files", "archive_files"):
        for name, expected in plan[group].items():
            path_in_repo = ROOT / name
            if path_in_repo.is_symlink() or not path_in_repo.is_file() or _sha(path_in_repo) != expected:
                raise ValueError(f"signed-span {group[:-6]} pin mismatch: {name}")
    bundle_data = stored.read_pinned(PARENT_SOURCE_TAR, PARENT_SOURCE_TAR_SHA)
    if plan.get("parent_source_bundle_sha256") != hashlib.sha256(bundle_data).hexdigest():
        raise ValueError("cached PR304 source bundle changed")
    cache = _ensure_cache()
    if (cache["pins"]["receipt_tar_sha256"] != PARENT_RECEIPTS_TAR_SHA
            or cache["control"].shape != (26,)
            or trial.tangent._tensor_sha(cache["control"]) != BASE_SHA
            or plan.get("stored_model_audit_sha256") != _sha(WORK / "AUDIT.json")):
        raise ValueError("signed-span cached base receipt or model audit identity changed")
    base = endpoint.load_endpoint()
    _same(cache["control"], base["control"], "2a4d base control")
    _same(cache["G"], base["full_G"], "2a4d base G")
    if cache["theta"] != base["theta"]:
        raise ValueError("signed-span cached theta differs from admitted 2a4d endpoint")
    _SIGNED_PLAN_SHA = digest
    return plan


def _verify_context(context: dict[str, Any], control: torch.Tensor, theta: float,
        gm: torch.Tensor, gp: torch.Tensor, normal: torch.Tensor,
        chart: torch.Tensor, pivot: int) -> dict[str, Any]:
    global _CONTEXT
    cache = _ensure_cache()
    if _CONTEXT is not None and context is not _CONTEXT:
        raise ValueError("cached HVP reuse refused for a different backend context")
    if (context.get("plan_path") is None or context["plan_path"].resolve() != PLAN.resolve()
            or context.get("plan_sha") != _SIGNED_PLAN_SHA
            or context.get("plan", {}).get("base_control_sha256") != BASE_SHA
            or context.get("plan", {}).get("experiment_kind") != "cached_2a4d_signed_span_vs_nonnegative_span"):
        raise ValueError("cached HVP context has a different plan or base point")
    _same(control, cache["control"], "control")
    _same(gm, cache["gm"], "minus gradient")
    _same(gp, cache["gp"], "plus gradient")
    if theta != cache["theta"]:
        raise ValueError("cached HVP context theta differs from the frozen 2a4d repeat")
    observed = context.get("model_state", {}).get("observed")
    if not isinstance(observed, dict):
        raise ValueError("cached HVP context has no fresh same-point observation")
    if (observed.get("pair", {}).get("passed") is not True
            or observed.get("face_ok") is not True
            or observed.get("side_objectives_match_native") is not True
            or observed.get("side_gradients_finite") is not True):
        raise ValueError("current same-point native/P2 side observation is not qualified")
    _same(observed["side"][-1][1], cache["gm"], "observed gminus")
    _same(observed["side"][1][1], cache["gp"], "observed gplus")
    q_tensor = (observed["q"] / trial.previous.FACE_SCALE).reshape(1)
    mixed = (1.0 - theta) * gm + theta * gp
    _same(torch.cat((mixed, q_tensor)), cache["G"], "fresh current full-G")
    j_budget = 128.0 * torch.finfo(control.dtype).eps * max(
        abs(cache["parent_J"]), torch.finfo(control.dtype).tiny)
    if abs(float(observed["native_j"]) - cache["parent_J"]) > j_budget:
        raise ValueError("fresh native J differs from the frozen 2a4d repeat")
    expected_normal = trial.geometry._face_normal(control, context["weights"])
    _same(normal, expected_normal, "face normal")
    expected_pivot = trial.geometry._pivot(control, context["weights"])
    if pivot != expected_pivot:
        raise ValueError("cached HVP context face pivot differs")
    _, expected_chart, _ = trial.geometry._chart_jacobian(control, context["weights"], pivot, 0.0)
    _same(chart, expected_chart, "face chart")
    record = context.get("record", {})
    if record.get("jacobian_rows_started") != 0 or record.get("dense_solves_started") != 0:
        raise ValueError("signed-span preparation must not perform new GN work")
    _CONTEXT = context
    return cache


def direction_specs(context: dict[str, Any], c: torch.Tensor, theta: float,
        gm: torch.Tensor, gp: torch.Tensor, normal: torch.Tensor,
        chart: torch.Tensor, pivot: int) -> list[dict[str, Any]]:
    cache = _verify_context(context, c, theta, gm, gp, normal, chart, pivot)
    model = cache["fresh_models"]
    d = torch.as_tensor(model["baseline_tangent"]["direction"], dtype=c.dtype)
    s = torch.as_tensor(model["residual_merit"]["direction"], dtype=c.dtype)
    _, _, expected_direction, _ = trial.tangent.tangent_direction(
        gm, gp, normal, chart, theta, pivot=pivot)
    _same(d, expected_direction, "fresh baseline tangent direction")
    base_action = torch.as_tensor(
        model["baseline_tangent"]["residual_direction"][:gm.numel()], dtype=c.dtype)

    def action(vector: torch.Tensor) -> torch.Tensor:
        _same(vector, d, "cached DG[d] action input")
        return base_action

    reconstructed_s, info = residual.build_residual_direction(
        action, normal, gm, gp, chart, pivot, normalize_to=d, seed_direction=d)
    _same(s, reconstructed_s, "cached normalized DG[d] direction")
    if not all(info[key] for key in ("finite", "both_side_gradients_descend", "seed_merit_descends")):
        raise ValueError("cached residual direction fails fresh same-point helper gates")
    products = []
    for name, direction in (("baseline_tangent", d), ("residual_merit", s)):
        side_hvp = {side: torch.as_tensor(
            model[name]["side_HVP"][str(side)], dtype=c.dtype) for side in (-1, 1)}
        rebuilt = trial.prepared.mixing.minimum_tangent_model(gm, gp,
            side_hvp[-1], side_hvp[1], normal, chart, theta,
            float(context["model_state"]["observed"]["q"]), pivot,
            trial.previous.FACE_SCALE, direction_override=direction)
        closure = trial.finite._base_model_closure(rebuilt, cache["G"], theta)
        if (not closure["passed"] or not all(rebuilt.gates.get(key) is True for key in (
                "finite", "both_side_gradients_descend", "merit_descends",
                "envelope_slope_matches_residual_dot"))):
            raise ValueError(f"stored {name} HVP pair fails reconstructed same-point model closure")
        _same(rebuilt.residual_direction,
            torch.as_tensor(model[name]["residual_direction"], dtype=c.dtype),
            f"stored {name} V closure")
        products.append(side_hvp)
    context["record"]["cached_model_reconstruction"] = {
        "baseline_direction_exact": True, "residual_direction_exact": True,
        "fresh_model_V_closure": True, "DGd_source": "PR304 baseline fresh_models residual_direction[:26]",
        "cached_HVP_pairs": 2}
    return [{"name": "baseline_tangent", "direction": d},
        {"name": "residual_merit", "direction": s}]


def hvp_pair(context: dict[str, Any], c: torch.Tensor,
        spec: dict[str, Any], side: int) -> torch.Tensor:
    cache = _ensure_cache()
    if _CONTEXT is None or context is not _CONTEXT:
        raise ValueError("cached HVP refused for an unqualified context")
    if side not in (-1, 1) or spec.get("name") not in DIRECTION_ORDER:
        raise ValueError("cached HVP refused for an unknown direction/side")
    _same(c, cache["control"], "reused HVP control")
    observed = context.get("model_state", {}).get("observed", {})
    side_data = observed.get("side", {})
    if -1 not in side_data or 1 not in side_data:
        raise ValueError("cached HVP refused after context side data disappeared")
    gm, gp = side_data[-1][1], side_data[1][1]
    weights = context.get("weights")
    if not isinstance(weights, torch.Tensor):
        raise ValueError("cached HVP refused after context weights disappeared")
    normal = trial.geometry._face_normal(c, weights)
    pivot = trial.geometry._pivot(c, weights)
    _, chart, _ = trial.geometry._chart_jacobian(c, weights, pivot, 0.0)
    _verify_context(context, c, _ensure_cache()["theta"], gm, gp, normal, chart, pivot)
    model = cache["fresh_models"][spec["name"]]
    direction = torch.as_tensor(model["direction"], dtype=c.dtype)
    _same(spec["direction"], direction, "reused HVP direction")
    vector = torch.as_tensor(model["side_HVP"][str(side)], dtype=c.dtype)
    if vector.shape != c.shape or not bool(torch.isfinite(vector).all()):
        raise ValueError("sealed cached HVP is malformed or nonfinite")
    return vector


def _signed_solve(G: torch.Tensor, D: torch.Tensor, V: torch.Tensor,
        radius: float) -> tuple[SubspaceModel, tuple[int, int]]:
    if not math.isfinite(radius) or not 0.0 < radius <= RADIUS:
        raise ValueError("signed-span solver radius is outside the frozen trust region")
    signs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    results = []
    for pair in signs:
        sign = torch.tensor(pair, dtype=G.dtype, device=G.device)
        result = solve_two_direction_subspace(
            G, D * sign.reshape(1, 2), V * sign.reshape(1, 2), radius)
        coeff = result.coefficients * sign
        direction = D @ coeff
        residual_direction = V @ coeff
        predicted = torch.linalg.vector_norm(G + residual_direction).square()
        diagnostics = dict(result.diagnostics)
        diagnostics.update(signed_quadrant=f"{pair[0]:+d},{pair[1]:+d}",
            signed_predicted_R=predicted, original_nonnegative_cone=False)
        results.append((SubspaceModel(coeff, direction, residual_direction, predicted, diagnostics), pair))
    return min(results, key=lambda item: float(item[0].predicted_R))


def coefficient_solve(G: torch.Tensor, D: torch.Tensor, V: torch.Tensor,
        radius: float) -> SubspaceModel:
    cache = _ensure_cache()
    _same(G, cache["G"], "coefficient-solver G")
    _same(D, cache["D"], "coefficient-solver D")
    _same(V, cache["V"], "coefficient-solver V")
    signature = (hashlib.sha256(G.contiguous().numpy().tobytes()).hexdigest(),
        hashlib.sha256(D.contiguous().numpy().tobytes()).hexdigest(),
        hashlib.sha256(V.contiguous().numpy().tobytes()).hexdigest(), float(radius))
    if _SIGNED_RESULT is not None and signature == _SIGNED_SIGNATURE:
        return _SIGNED_RESULT
    result, pair = _signed_solve(G, D, V, radius)
    result.diagnostics.update(selected_signed_quadrant=f"{pair[0]:+d},{pair[1]:+d}")
    return result


def choose_radius(G: torch.Tensor, D: torch.Tensor, V: torch.Tensor,
        record: dict[str, Any]) -> float | None:
    global _SIGNED_RESULT, _SIGNED_SIGNATURE
    cache = _ensure_cache()
    _same(G, cache["G"], "preflight G")
    _same(D, cache["D"], "preflight D")
    _same(V, cache["V"], "preflight V")
    nonnegative = solve_two_direction_subspace(G, D, V, RADIUS)
    signed, pair = _signed_solve(G, D, V, RADIUS)
    direction = signed.direction
    gminus_dot = torch.dot(cache["gm"], direction)
    gplus_dot = torch.dot(cache["gp"], direction)
    side_budget = trial.tangent._roundoff(G.dtype,
        float(torch.linalg.vector_norm(cache["gm"])) * float(torch.linalg.vector_norm(direction)),
        float(torch.linalg.vector_norm(cache["gp"])) * float(torch.linalg.vector_norm(direction)))
    gain = nonnegative.predicted_R - signed.predicted_R
    gain_budget = 128.0 * torch.finfo(G.dtype).eps * max(
        abs(float(nonnegative.predicted_R)), abs(float(signed.predicted_R)),
        float(torch.dot(G, G)), torch.finfo(G.dtype).tiny)
    gain_passed = bool(torch.isfinite(gain) & (gain > gain_budget))
    side_passed = bool(gminus_dot < -side_budget and gplus_dot < -side_budget)
    record["signed_span_preflight"] = {
        "nonnegative_coefficients": nonnegative.coefficients.tolist(),
        "nonnegative_predicted_R": float(nonnegative.predicted_R),
        "best_signed_quadrant": list(pair),
        "signed_coefficients": signed.coefficients.tolist(),
        "signed_predicted_R": float(signed.predicted_R),
        "predicted_R_gain": float(gain), "gain_budget": float(gain_budget),
        "gain_passed": gain_passed,
        "gminus_dot_Da": float(gminus_dot), "gplus_dot_Da": float(gplus_dot),
        "side_descent_budget": float(side_budget), "both_side_costs_descend": side_passed,
        "radius": RADIUS, "actual_candidate_calls": 0}
    if not (gain_passed and side_passed):
        record["signed_span_preflight"]["status"] = "no_candidate_no_signed_gain_or_side_descent"
        return None
    _SIGNED_RESULT = signed
    _SIGNED_SIGNATURE = (hashlib.sha256(G.contiguous().numpy().tobytes()).hexdigest(),
        hashlib.sha256(D.contiguous().numpy().tobytes()).hexdigest(),
        hashlib.sha256(V.contiguous().numpy().tobytes()).hexdigest(), float(RADIUS))
    record["signed_span_preflight"]["status"] = "signed_model_gain_and_side_costs_pass"
    return RADIUS


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        result = trial._run_child(PLAN, args.plan_sha256, ATTEMPT / "step.json",
            experiment=sys.modules[__name__])
        if result["execution_status"] == "failed":
            raise SystemExit(1)
        return
    load_plan(PLAN, args.plan_sha256)
    if ATTEMPT.exists():
        raise ValueError("signed-span attempt must be fresh")
    ATTEMPT.mkdir()
    resource = trial.previous.run_guarded_diagnostic([
        str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
        "--plan-sha256", args.plan_sha256],
        wall_seconds=POLICY["outer_seconds"], rss_bytes=POLICY["rss_bytes"],
        report_path=ATTEMPT / "step.resource.json", log_path=ATTEMPT / "step.log")
    status = trial.shared._execution_status(resource,
        wall_seconds=POLICY["outer_seconds"], rss_bytes=POLICY["rss_bytes"])
    child_path = ATTEMPT / "step.json"
    result = {"execution_status": status, "resource": resource,
        "charged_prior_guarded_seconds": PRIOR_GUARDED_SECONDS,
        "total_guarded_seconds": PRIOR_GUARDED_SECONDS + float(resource["elapsed_seconds"]),
        "child_sha256": trial.comparison._sha(child_path) if child_path.exists() else None}
    trial.tangent._write(ATTEMPT / "step.run.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
