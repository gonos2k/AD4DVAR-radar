"""Guarded fresh-HVP continuation along the selected-face tangent.

This is a model-guided tangent-gradient correction. It is neither a Newton
solve nor a root/minimum/response certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Literal

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_candidate_requalification as requal
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_CONTINUATION_PLAN_20261009.json"
REQUAL_PLAN = EVIDENCE / "CANDIDATE_REQUALIFICATION_PLAN_20261009.json"
REQUAL_PLAN_SHA = "5b7ebac50ebc38c0eaf34539c5bdf89e00c4d342ef96f2a96bc12a93fa30bfb4"
REQUAL_ARCHIVE = EVIDENCE / "REQUAL_ARCHIVE_20261009.json"
CURRENT_ARCHIVE = EVIDENCE / "candidate_requalification_20261009_attempt1/step.json.gz"
CURRENT_RUN = EVIDENCE / "candidate_requalification_20261009_attempt1/step.run.json"
CURRENT_RESOURCE = EVIDENCE / "candidate_requalification_20261009_attempt1/step.resource.json"
BASE_CONTROL_SHA = "6b29dacd01fc30ee4041e93dd3ad110c29699c4553d098694086eadeb580a743"
BASE_THETA = 0.3504554198817298
FACE_AXIS: Literal["y"] = "y"
FACE_ROW, FACE_COLUMN = 4, 3
FACE: dict[str, Any] = {"axis": FACE_AXIS, "row": FACE_ROW, "column": FACE_COLUMN}
SELF = "examples/weather_scenarios/fv_point_3h_tangent_continuation.py"
TEST = "tests/test_fv_point_3h_tangent_continuation.py"
POLICY = {
    "guarded_launches": 1, "internal_seconds": 240.0, "outer_seconds": 300.0,
    "rss_bytes": 1024**3, "hvp_calls": 6, "max_accepted_iterations": 3,
    "max_candidates_per_iteration": 16, "radius": 0.05, "face_scale": 0.84,
    "optimizer_steps": 3, "pcg_solves": 0, "dense_solves": 0,
    "root_claim": False, "minimum_claim": False, "response_claim": False,
}
J_C1 = F_C1 = 1e-4
ANALYSIS_STAGES = 360

MAX_ACCEPTED = 3
MAX_HVP = 6
MAX_CANDIDATES = 16
RADIUS = 0.05
FACE_SCALE = 0.84
EPS_MULTIPLIER = 128.0


@dataclass(frozen=True)
class TangentModel:
    """One current point's face-tangent model and its two fresh HVPs."""

    theta: float
    mixed_gradient: Tensor
    tangent_gradient: Tensor
    chart_direction: Tensor
    hminus: Tensor
    hplus: Tensor
    delta_theta: float
    delta_theta_numerator: float
    delta_theta_denominator: float
    direction: Tensor
    residual: Tensor
    residual_direction: Tensor
    side_products: tuple[float, float]
    merit_product: float
    gates: dict[str, Any]


def _finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _roundoff(dtype: torch.dtype, *scales: float) -> float:
    scale = max(*(abs(float(x)) for x in scales), torch.finfo(dtype).tiny)
    return EPS_MULTIPLIER * torch.finfo(dtype).eps * scale


def tangent_direction(gminus: Tensor, gplus: Tensor, normal: Tensor,
                      chart_jacobian: Tensor, theta: float, *, pivot: int | None = None
                      ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Return current mixture, projected gradient, and exact chart tangent."""
    size = gminus.numel()
    if (not 0 <= theta <= 1 or gplus.shape != (size,) or normal.shape != (size,)
            or chart_jacobian.shape != (size, size - 1)
            or not all(_finite(t) for t in (gminus, gplus, normal, chart_jacobian))):
        raise ValueError("theta or current tangent inputs are invalid")
    nnorm = torch.linalg.vector_norm(normal)
    if not bool(torch.isfinite(nnorm) & (nnorm > torch.finfo(normal.dtype).tiny)):
        raise ValueError("selected face normal is unresolved")
    jump = gplus - gminus
    jnorm = torch.linalg.vector_norm(jump)
    unit_normal = normal / nnorm
    tangent_jump = jump - unit_normal * torch.dot(unit_normal, jump)
    tangent_jump_norm = torch.linalg.vector_norm(tangent_jump)
    support_budget = _roundoff(gminus.dtype, float(torch.linalg.vector_norm(gminus)),
                               float(torch.linalg.vector_norm(gplus)), float(jnorm))
    if not bool(torch.isfinite(tangent_jump_norm)) or float(tangent_jump_norm) > support_budget:
        raise ValueError("one-sided gradient jump is not supported by the common face")
    mixed = (1 - theta) * gminus + theta * gplus
    tangent_gradient = mixed - unit_normal * torch.dot(unit_normal, mixed)
    pivot = int(torch.argmax(normal.abs())) if pivot is None else int(pivot)
    if not 0 <= pivot < size:
        raise ValueError("selected-face chart pivot is outside the control")
    retained = [i for i in range(size) if i != pivot]
    direction = chart_jacobian @ (-tangent_gradient[retained])
    if not _finite(direction):
        raise ValueError("chart tangent direction is nonfinite")
    return mixed, tangent_gradient, direction, {"common_face_gradient_jump": True,
        "gradient_jump_tangent_norm": float(tangent_jump_norm),
        "gradient_jump_support_budget": support_budget}


def tangent_model(gminus: Tensor, gplus: Tensor, hminus: Tensor, hplus: Tensor,
                  normal: Tensor, chart_jacobian: Tensor, theta: float,
                  *, q: float = 0.0, face_scale: float = FACE_SCALE,
                  pivot: int | None = None) -> TangentModel:
    """Build the correction from current side products only.

    ``chart_jacobian`` maps retained coordinates to the ambient chart tangent.
    ``hminus`` and ``hplus`` are fresh HVP vectors along the returned chart
    direction, not cached Hessian rows or full Hessian matrices.
    """
    tensors = (gminus, gplus, hminus, hplus, normal, chart_jacobian)
    if not 0.0 <= theta <= 1.0 or not all(_finite(t) for t in tensors):
        raise ValueError("theta or current side products are outside the finite domain")
    size = gminus.numel()
    if (gplus.shape != (size,) or normal.shape != (size,) or hminus.shape != (size,)
            or hplus.shape != (size,) or chart_jacobian.shape != (size, size - 1)
            or not math.isfinite(face_scale) or face_scale <= 0):
        raise ValueError("current tangent model dimensions or face scale are invalid")
    normal_norm = torch.linalg.vector_norm(normal)
    if not bool(torch.isfinite(normal_norm) & (normal_norm > torch.finfo(normal.dtype).tiny)):
        raise ValueError("selected face normal is unresolved")

    jump = gplus - gminus
    unit_normal = normal / normal_norm
    jump_norm = torch.linalg.vector_norm(jump)
    normal_jump = torch.dot(unit_normal, jump)
    denominator_budget = EPS_MULTIPLIER * torch.finfo(normal.dtype).eps * float(jump_norm)
    if abs(float(normal_jump)) <= denominator_budget:
        raise ValueError("one-sided gradient jump has unresolved normal component")

    mixed, tangent_gradient, chart_direction, support = tangent_direction(
        gminus, gplus, normal, chart_jacobian, theta, pivot=pivot)
    hmix = (1.0 - theta) * hminus + theta * hplus
    numerator = torch.dot(unit_normal, mixed + hmix)
    delta_theta = -numerator / normal_jump
    residual = torch.cat((mixed, normal.new_tensor([q / face_scale])))
    residual_direction = torch.cat((hmix + jump * delta_theta,
        (torch.dot(unit_normal, chart_direction) * normal_norm / face_scale).reshape(1)))
    side_products = (float(torch.dot(gminus, chart_direction)),
                     float(torch.dot(gplus, chart_direction)))
    merit_product = float(torch.dot(residual, residual_direction))
    finite = all(_finite(t) for t in (mixed, tangent_gradient, chart_direction,
                                      hmix, numerator, normal_jump, residual_direction))
    side_budget = _roundoff(gminus.dtype, float(torch.linalg.vector_norm(gminus)) *
        float(torch.linalg.vector_norm(chart_direction)), float(torch.linalg.vector_norm(gplus)) *
        float(torch.linalg.vector_norm(chart_direction)))
    merit_budget = _roundoff(gminus.dtype, float(torch.dot(residual, residual)),
        float(torch.linalg.vector_norm(residual)) * float(torch.linalg.vector_norm(residual_direction)))
    gates = {"finite": finite,
        **support,
        "normal_denominator": float(normal_jump),
        "normal_denominator_budget": denominator_budget,
        "both_side_gradients_descend": max(side_products) < -side_budget,
        "side_descent_budget": side_budget,
        "merit_descends": merit_product < -merit_budget,
        "merit_descent_budget": merit_budget}
    if not finite:
        raise ValueError("tangent correction products are nonfinite")
    return TangentModel(theta, mixed, tangent_gradient, chart_direction, hminus, hplus,
        float(delta_theta), float(numerator), float(normal_jump), chart_direction,
        residual, residual_direction, side_products, merit_product, gates)


def candidate_alphas(model: TangentModel, *, radius: float = RADIUS,
                     limit: int = MAX_CANDIDATES) -> list[float]:
    """Stable radius/merit cap followed by bounded halvings."""
    if not model.gates["finite"] or not model.gates["both_side_gradients_descend"] \
            or not model.gates["merit_descends"]:
        return []
    norm = torch.linalg.vector_norm(model.direction)
    direction_norm = float(norm)
    if not math.isfinite(direction_norm) or direction_norm <= torch.finfo(norm.dtype).tiny:
        return []
    derivative_norm = float(torch.linalg.vector_norm(model.residual_direction))
    if not math.isfinite(derivative_norm) or derivative_norm <= torch.finfo(norm.dtype).tiny:
        return []
    # Ratios avoid forming squared norms that may overflow.
    merit_cap = (-model.merit_product / derivative_norm) / derivative_norm
    alpha0 = min(1.0, radius / direction_norm, merit_cap)
    if not math.isfinite(alpha0) or alpha0 <= 0:
        return []
    return [alpha0 / (2 ** i) for i in range(limit)]


def search_candidates(model: TangentModel, current: Tensor,
                      candidate_eval: Callable[[Tensor, float, float], dict[str, Any]], *,
                      chart_candidate: Callable[[Tensor, Tensor, float], Tensor] | None = None,
                      radius: float = RADIUS, limit: int = MAX_CANDIDATES
                      ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Search actual chart points; candidate_eval returns independent gate facts.

    The shared search kernel owns alpha bounds, theta-domain checks and actual
    chart-radius checks. Product-specific J/F/Q/trace checks remain in the
    evaluator and must all be true for ``accepted``.
    """
    if limit < 1 or limit > MAX_CANDIDATES:
        raise ValueError("tangent search exceeds the 16-candidate hard limit")
    trials: list[dict[str, Any]] = []
    for alpha in candidate_alphas(model, radius=radius, limit=limit):
        theta = model.theta + alpha * model.delta_theta
        trial: dict[str, Any] = {"alpha": alpha, "theta": theta, "accepted": False}
        if not 0.0 <= theta <= 1.0:
            trial["status"] = "theta_domain_refused"
            trials.append(trial)
            continue
        candidate = (chart_candidate(current, model.direction, alpha) if chart_candidate
                     else current + alpha * model.direction)
        actual_delta = candidate - current
        actual_norm = float(torch.linalg.vector_norm(actual_delta))
        trial["actual_path_norm"] = actual_norm
        if not math.isfinite(actual_norm) or actual_norm > radius * (1 + 64 * torch.finfo(current.dtype).eps):
            trial["status"] = "actual_chart_radius_refused"
            trials.append(trial)
            continue
        facts = candidate_eval(candidate, theta, alpha)
        trial.update(facts)
        required = ("J_armijo_passed", "F_squared_armijo_passed", "face_audit_passed",
                    "branch_pair_passed", "side_objectives_match_native", "side_gradients_finite")
        trial["accepted"] = all(facts.get(name) is True for name in required)
        trial["status"] = "accepted" if trial["accepted"] else "rejected"
        trial["control"] = candidate.detach().tolist()
        trials.append(trial)
        if trial["accepted"]:
            return trial, trials
    return None, trials


def fresh_final_closure(proposal: dict[str, Any], repeat: dict[str, Any]) -> bool:
    """P2 closure: require both fresh side gradients to match the proposal."""
    # A mixed-gradient match alone cannot detect equal-and-opposite side drift.
    required = ("side_gradients_finite", "native_objective_matches_proposal",
        "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
        "branch_pair_passed", "trace_matches_proposal", "source_unchanged", "fixed_input_unchanged",
        "runtime_unchanged", "deadline_passed")
    if not all(repeat.get(key) is True for key in required):
        return False
    for key in ("theta", "control", "objective", "F_squared", "side_gradients"):
        if key not in proposal:
            return False
    try:
        if float(repeat["theta"]) != float(proposal["theta"]):
            return False
        actual_control = torch.as_tensor(repeat["control"], dtype=torch.float64)
        expected_control = torch.as_tensor(proposal["control"], dtype=torch.float64)
        if not torch.equal(actual_control, expected_control):
            return False
        actual = {side: torch.as_tensor(repeat["side_gradients"][str(side)], dtype=torch.float64)
                  for side in (-1, 1)}
        expected = {side: torch.as_tensor(proposal["side_gradients"][str(side)], dtype=torch.float64)
                    for side in (-1, 1)}
    except (KeyError, TypeError, ValueError):
        return False
    return shared._gradient_pair_match(actual, expected)


def bounded_continuation(initial_control: Tensor, theta: float,
                         current_products: Callable[[Tensor, float], tuple[Tensor, Tensor, Tensor, Tensor, float]],
                         side_hvp: Callable[[Tensor, Tensor, int], Tensor],
                         chart_candidate: Callable[[Tensor, Tensor, float], Tensor],
                         candidate_eval: Callable[[Tensor, float, float], dict[str, Any]],
                         final_repeat: Callable[[dict[str, Any]], dict[str, Any]], *,
                         chart_pivot: Callable[[Tensor], int] | None = None,
                         committed_progress: Callable[[list[dict[str, Any]]], None] | None = None,
                         max_accepted: int = MAX_ACCEPTED) -> dict[str, Any]:
    """Run at most three accepted tangent corrections with transactional commit.

    ``current_products`` returns fresh ``(g-, g+, normal, chart_jacobian, Q)``.
    ``side_hvp`` performs exactly two current HVPs along the computed chart
    direction. Both callbacks run again at each committed point.
    ``final_repeat`` must recompute all stated closure facts without reusing the
    line-search result. The last confirmed control and theta survive refusal.
    """
    if max_accepted < 1 or max_accepted > MAX_ACCEPTED:
        raise ValueError("tangent continuation exceeds the three-commit hard limit")
    control, current_theta = initial_control.clone(), float(theta)
    records: list[dict[str, Any]] = []
    for index in range(max_accepted):
        try:
            gminus, gplus, normal, z, q = current_products(control, current_theta)
            pivot = chart_pivot(control) if chart_pivot is not None else None
            _, _, direction, _ = tangent_direction(gminus, gplus, normal, z, current_theta,
                                                    pivot=pivot)
            hminus = side_hvp(control, direction, -1)
            hplus = side_hvp(control, direction, 1)
            model = tangent_model(gminus, gplus, hminus, hplus, normal, z, current_theta,
                                  q=q, pivot=pivot)
            accepted, trials = search_candidates(model, control, candidate_eval,
                chart_candidate=chart_candidate)
        except Exception as error:
            records.append({"index": index, "theta": current_theta, "accepted": False,
                "refusal": f"current tangent evaluation raised {type(error).__name__}: {error}"})
            break
        item: dict[str, Any] = {"index": index, "base_control_sha256": _tensor_sha(control),
            "theta": current_theta,
            "direction": model.direction.detach().tolist(), "hminus_direction": hminus.detach().tolist(),
            "hplus_direction": hplus.detach().tolist(), "delta_theta": model.delta_theta,
            "delta_theta_numerator": model.delta_theta_numerator,
            "delta_theta_denominator": model.delta_theta_denominator,
            "mixed_gradient": model.mixed_gradient.detach().tolist(),
            "tangent_gradient": model.tangent_gradient.detach().tolist(),
            "residual": model.residual.detach().tolist(),
            "residual_direction": model.residual_direction.detach().tolist(),
            "side_products": list(model.side_products), "merit_product": model.merit_product,
            "gates": model.gates, "trials": trials, "accepted": False}
        if accepted is None:
            item["refusal"] = "no candidate passed the current tangent search gates"
            records.append(item)
            break
        try:
            repeat = final_repeat(accepted)
        except Exception as error:
            item["refusal"] = f"final closure raised {type(error).__name__}: {error}"
            records.append(item)
            break
        item["final_repeat"] = repeat
        item["accepted"] = fresh_final_closure(accepted, repeat)
        if not item["accepted"]:
            item["refusal"] = "independent fresh final closure failed"
            records.append(item)
            break
        control = torch.as_tensor(accepted["control"], dtype=initial_control.dtype,
                                  device=initial_control.device)
        current_theta = float(accepted["theta"])
        item["committed_control"] = control.detach().tolist()
        item["committed_theta"] = current_theta
        records.append(item)
        if committed_progress is not None:
            committed_progress(records)
    return {"control": control, "theta": current_theta, "iterations": records,
        "accepted_iterations": sum(bool(item["accepted"]) for item in records),
        "candidate_type": "tangent_gradient_correction",
        "full_smooth_root": False, "minimum_claim": False, "response_claim": False}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("tangent-continuation plan identity mismatch")
    plan = json.loads(path.read_text())
    if _sha(REQUAL_PLAN) != REQUAL_PLAN_SHA:
        raise ValueError("producing candidate-requalification plan digest changed")
    prior = json.loads(REQUAL_PLAN.read_text())
    snapshots = json.loads((EVIDENCE / "tangent_source_20261009/manifest.json").read_text())
    expected_sources = set(prior["source_files"]) | {SELF, TEST}
    expected_archives = (set(prior["archive_files"]) | {
        REQUAL_PLAN.relative_to(ROOT).as_posix(),
        REQUAL_ARCHIVE.relative_to(ROOT).as_posix(),
        (EVIDENCE / "tangent_source_20261009/manifest.json").relative_to(ROOT).as_posix(),
        CURRENT_ARCHIVE.relative_to(ROOT).as_posix(), CURRENT_RUN.relative_to(ROOT).as_posix(),
        CURRENT_RESOURCE.relative_to(ROOT).as_posix(),
        *[row["archive_path"] for row in snapshots["snapshots"].values()]})
    if (plan.get("experiment_kind") != "current_tangent_continuation"
            or plan.get("policy") != POLICY
            or plan.get("producing_plan") != REQUAL_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != REQUAL_PLAN_SHA
            or _sha(REQUAL_PLAN) != REQUAL_PLAN_SHA
            or plan.get("scope") != (
                "Fresh current-point two-side HVP tangent-gradient corrections; "
                "no old Hessian operator or minimum/response certificate")
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != 0.3504554198817298
            or plan.get("face") != FACE
            or plan.get("producer_source_snapshots") != snapshots.get("snapshots")
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 126
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 94):
        raise ValueError("tangent-continuation plan scope or policy changed")
    sources, archives = plan["source_files"], plan["archive_files"]
    if any(archives.get(name) != value for name, value in prior["archive_files"].items()):
        raise ValueError("inherited candidate-requalification archive pin changed")
    if any(sources.get(name) != digest_value for name, digest_value in prior["source_files"].items()
           if name not in snapshots["snapshots"]):
        raise ValueError("inherited candidate-requalification source pin changed")
    for name, snapshot in snapshots["snapshots"].items():
        if (prior["source_files"].get(name) != snapshot["sha256"]
                or sources.get(name) != _sha(ROOT / name)
                or archives.get(snapshot["archive_path"]) != snapshot["sha256"]
                or _sha(ROOT / snapshot["archive_path"]) != snapshot["sha256"]):
            raise ValueError(f"live tangent source or archived producer snapshot changed: {name}")
    for name, digest_value in {**sources, **archives}.items():
        source = ROOT / name
        if (source.is_symlink() or not source.resolve().is_relative_to(ROOT.resolve())
                or _sha(source) != digest_value):
            raise ValueError(f"tangent source/archive pin mismatch: {name}")
    for field, source in (("direction_archive", CURRENT_ARCHIVE),
                          ("direction_run", CURRENT_RUN),
                          ("direction_resource", CURRENT_RESOURCE)):
        if (plan.get(field) != source.relative_to(ROOT).as_posix()
                or plan.get(field + "_sha256") != _sha(source)):
            raise ValueError(f"tangent {field} receipt changed")
    if plan.get("direction_child_sha256") != json.loads(REQUAL_ARCHIVE.read_text())["raw_sha256"]:
        raise ValueError("tangent base child digest differs from archive manifest")
    return plan


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    archive = json.loads(REQUAL_ARCHIVE.read_text())
    run, resource = json.loads(CURRENT_RUN.read_text()), json.loads(CURRENT_RESOURCE.read_text())
    raw_bytes = gzip.decompress(CURRENT_ARCHIVE.read_bytes())
    child_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected_resource = run.get("resource", {})
    if (archive.get("gzip_path") != CURRENT_ARCHIVE.relative_to(ROOT).as_posix()
            or archive.get("gzip_sha256") != _sha(CURRENT_ARCHIVE)
            or archive.get("raw_sha256") != child_sha
            or run.get("child_sha256") != child_sha
            or run.get("execution_status") != "completed"
            or run.get("child_read_error") is not None
            or expected_resource != resource
            or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != POLICY["outer_seconds"]
            or resource.get("rss_limit_bytes") != POLICY["rss_bytes"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > POLICY["outer_seconds"]
            or resource.get("sampled_peak_rss_bytes", POLICY["rss_bytes"])
                >= POLICY["rss_bytes"]):
        raise ValueError("accepted PR266 base archive/run/resource closure is invalid")
    raw = json.loads(raw_bytes)
    iterations = raw.get("iterations", [])
    accepted = iterations[0].get("accepted") if len(iterations) == 1 else None
    control = torch.as_tensor(raw.get("current_control"), dtype=torch.float64)
    producing = json.loads(REQUAL_PLAN.read_text())
    saved_hashes = raw.get("plan_hashes", {})
    expected_source_receipt = {**producing.get("source_files", {}),
        **producing.get("archive_files", {}),
        REQUAL_PLAN.relative_to(ROOT).as_posix(): REQUAL_PLAN_SHA}
    accepted_input = raw.get("input_before", {})
    after_input = raw.get("input_after", {})
    if (raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("plan_sha256") != REQUAL_PLAN_SHA
            or saved_hashes.get("source_files") != producing.get("source_files")
            or saved_hashes.get("archive_files") != producing.get("archive_files")
            or raw.get("source_before") != expected_source_receipt
            or raw.get("source_after") != expected_source_receipt
            or raw.get("candidate_type") != "one_nonsmooth_requalified_step"
            or raw.get("candidate_committed") is not True
            or raw.get("active_candidate_committed") is not False
            or raw.get("optimizer_steps_applied") != 1 or not isinstance(accepted, dict)
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or _tensor_sha(control) != BASE_CONTROL_SHA
            or accepted.get("control_sha256") != BASE_CONTROL_SHA
            or torch.as_tensor(accepted.get("control"), dtype=torch.float64).shape != (26,)
            or not torch.equal(torch.as_tensor(accepted.get("control"), dtype=torch.float64), control)
            or float(accepted.get("theta", float("nan"))) != plan["initial_theta"]
            or not isinstance(raw.get("final_repeat"), dict)
            or not all(raw["final_repeat"].get(key) is True for key in (
                "gradient_matches_proposal", "objective_matches_proposal",
                "side_objectives_match_native", "branch_matches_proposal",
                "merit_matches_proposal", "face_roundoff_passed",
                "source_unchanged", "fixed_input_unchanged", "runtime_unchanged",
                "deadline_passed"))
            or not raw.get("source_unchanged") or not raw.get("fixed_input_unchanged")
            or not raw.get("runtime_unchanged") or not raw.get("deadline_passed")
            or raw.get("source_before") != raw.get("source_after")
            or not geometry.model._fixed_input(after_input, accepted_input, BASE_CONTROL_SHA)
            or raw.get("runtime") != raw.get("runtime_after")):
        raise ValueError("PR266 base is not a closed accepted candidate")
    return {"raw": raw, "control": control, "theta": float(accepted["theta"]),
        "objective": float(accepted["objective"]), "accepted": accepted,
        "child_sha256": child_sha}


def _observe(probe: Any, problem: Any, control: Tensor, parameters: Tensor,
             weights: Tensor) -> dict[str, Any]:
    native_j = problem.objective(control, parameters)
    side = {sign: probe._eval_side(problem, control, parameters, sign) for sign in (-1, 1)}
    traces = {sign: probe._analysis_trace(problem, control, parameters, sign) for sign in (-1, 1)}
    pair = requal.current_branch_pair_gate(traces[-1], traces[1], dtype=control.dtype,
                                           stages=ANALYSIS_STAGES)
    face_q = probe._face_value(control, weights)
    production_q, face_bound, face_ok = probe._face_audit(problem, control, weights, FACE, 0.0)
    scale = 128 * torch.finfo(control.dtype).eps * max(abs(float(native_j)),
        abs(float(side[-1][0])), abs(float(side[1][0])), torch.finfo(control.dtype).tiny)
    objectives_match = all(abs(float(side[s][0] - native_j)) <= scale for s in (-1, 1))
    gradients_finite = all(bool(torch.isfinite(side[s][1]).all()) for s in (-1, 1))
    return {"native_j": native_j, "side": side, "traces": traces, "pair": pair,
        "q": face_q, "production_q": production_q, "face_bound": face_bound,
        "face_ok": bool(face_ok and abs(float(face_q)) <= face_bound
                         and abs(float(production_q)) <= face_bound),
        "side_objectives_match_native": objectives_match,
        "side_gradients_finite": gradients_finite}


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path, *,
                    plan_loader: Callable[[Path, str], dict[str, Any]] | None = None,
                    base_loader: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                    base_control_sha256: str | None = None) -> dict[str, Any]:
    start = time.monotonic()
    plan = (plan_loader or _load_plan)(plan_path, plan_sha)
    base = (base_loader or _load_current_base)(plan)
    actual_base_sha = _tensor_sha(base["control"])
    if base_control_sha256 is not None and actual_base_sha != base_control_sha256:
        raise ValueError("loaded continuation control differs from its frozen base hash")
    expected_base_sha = base_control_sha256 or actual_base_sha
    deadline = start + POLICY["internal_seconds"]
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "preflight", "plan_sha256": plan_sha,
        "base_control_sha256": actual_base_sha, "policy": POLICY,
        "current_control": base["control"].tolist(),
        "current_control_sha256": actual_base_sha, "current_theta": base["theta"],
        "last_confirmed_control": base["control"].tolist(),
        "last_confirmed_control_sha256": actual_base_sha,
        "last_confirmed_theta": base["theta"], "last_confirmed_iterations": 0,
        "hvp_calls_started": 0, "hvp_calls_completed": 0, "hvp_history": [],
        "iterations": [], "candidate_committed": False, "active_candidate_committed": False,
        "optimizer_steps_applied": 0, "full_smooth_root": False,
        "minimum_claim": False, "response_claim": False}
    source_before = shared._source_hashes(plan, plan_path, plan_sha)
    record["source_before"] = source_before
    record["plan_hashes"] = {"source_files": plan["source_files"],
                             "archive_files": plan["archive_files"]}
    if isinstance(base.get("resume_efficiency"), dict):
        record["resume_anchor"] = {"source_child_sha256": base.get("child_sha256"),
            "control_sha256": actual_base_sha, "theta": base["theta"],
            "objective": base["objective"],
            "F_squared_at_endpoint": base["accepted"].get("F_squared"),
            "preceding_step_efficiency_source_control_sha256":
                base.get("resume_efficiency_source_control_sha256"),
            "preceding_step_efficiency_source_theta":
                base.get("resume_efficiency_source_theta"),
            "preceding_step_efficiency": base["resume_efficiency"]}
    _write(output, record)
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    input_before = shared._input_identity(problem, original, base["control"], parameters, truth)
    runtime_before = shared._runtime()
    if (input_before != base["raw"].get("input_after")
            or runtime_before != base["raw"].get("runtime_after")):
        raise ValueError("reconstructed fixed input/runtime differs from predecessor endpoint receipt")
    weights = geometry._face_weights(problem, axis=FACE_AXIS, row=FACE_ROW, column=FACE_COLUMN)
    if abs(float(weights.abs().max()) - FACE_SCALE) > 128 * torch.finfo(weights.dtype).eps * FACE_SCALE:
        raise ValueError("selected-face scale differs from the frozen 0.84 policy")
    shared._deadline(deadline, POLICY["internal_seconds"])
    initial = _observe(shared, problem, base["control"], parameters, weights)
    jtol = 128 * torch.finfo(base["control"].dtype).eps * max(
        abs(float(initial["native_j"])), abs(base["objective"]), torch.finfo(base["control"].dtype).tiny)
    initial_f = torch.cat(((1 - base["theta"]) * initial["side"][-1][1]
                           + base["theta"] * initial["side"][1][1],
                           (initial["q"] / FACE_SCALE).reshape(1)))
    initial_f2 = float(torch.dot(initial_f, initial_f))
    accepted_f2 = float(base["accepted"].get("F_squared", float("nan")))
    f2_tol = 128 * torch.finfo(base["control"].dtype).eps * max(
        abs(initial_f2), abs(accepted_f2), torch.finfo(base["control"].dtype).tiny)
    try:
        receipt_gradients = {side: torch.as_tensor(base["accepted"]["side_gradients"][str(side)],
            dtype=base["control"].dtype) for side in (-1, 1)}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("accepted continuation base lacks both side-gradient vectors") from error
    gradients_match = shared._gradient_pair_match(
        {side: initial["side"][side][1] for side in (-1, 1)}, receipt_gradients)
    accepted_traces = base["accepted"].get("branch_trace", {})
    traces_match = all(initial["traces"][side]["signature_sha256"]
        == accepted_traces.get(str(side), {}).get("signature_sha256") for side in (-1, 1))
    if (abs(float(initial["native_j"]) - base["objective"]) > jtol
            or not math.isfinite(accepted_f2) or abs(initial_f2 - accepted_f2) > f2_tol
            or not gradients_match or not traces_match
            or not initial["pair"]["passed"] or not initial["face_ok"]
            or not initial["side_objectives_match_native"] or not initial["side_gradients_finite"]):
        raise ValueError("fresh J/F/side gradients/traces/branch pair failed at continuation base")
    record.update(numerical_status="fresh_current_base_closed", initial_theta=base["theta"],
        current_native_objective=float(initial["native_j"]),
        initial_F_squared=initial_f2, accepted_F_squared=accepted_f2,
        initial_F=initial_f.tolist(),
        initial_side_gradients_match=True, initial_branch_traces_match=True,
        base_current_branch_pair=initial["pair"],
        base_signature_changed_from_old=None,
        initial_branch_trace={str(s): initial["traces"][s] for s in (-1, 1)},
        face_qualification={"value": float(initial["q"]),
            "production_value": float(initial["production_q"]), "roundoff_bound": initial["face_bound"],
            "passed": initial["face_ok"]})
    closure_state: dict[str, Any] = {"control": base["control"], "theta": base["theta"],
                                    "accepted_count": 0}
    record["last_confirmed_control"] = base["control"].tolist()
    record["last_confirmed_control_sha256"] = actual_base_sha
    record["last_confirmed_theta"] = base["theta"]
    _write(output, record)
    model_state: dict[str, Any] = {}
    model_state["current_observed"] = initial

    def current_products(control: Tensor, theta: float):
        shared._deadline(deadline, POLICY["internal_seconds"])
        if (torch.equal(control, closure_state["control"]) and theta == closure_state["theta"]
                and "observed" in closure_state):
            observed = closure_state["observed"]
        elif torch.equal(control, base["control"]) and theta == base["theta"]:
            observed = initial
        else:
            observed = _observe(shared, problem, control, parameters, weights)
        if (not observed["pair"]["passed"] or not observed["face_ok"]
                or not observed["side_objectives_match_native"]
                or not observed["side_gradients_finite"]):
            raise ValueError("current native J, face, or strict two-side branch pair gate failed")
        normal = geometry._face_normal(control, weights)
        pivot = geometry._pivot(control, weights)
        _, z, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
        gm, gp = observed["side"][-1][1], observed["side"][1][1]
        model_state.clear()
        model_state.update(observed=observed, pivot=pivot, normal=normal, z=z,
            gm=gm, gp=gp, theta=theta,
            f=torch.cat(((1 - theta) * gm + theta * gp,
                (observed["q"] / FACE_SCALE).reshape(1))))
        model_state["current_observed"] = observed
        return gm, gp, normal, z, float(observed["q"])

    def side_hvp(control: Tensor, direction: Tensor, sign: int) -> Tensor:
        gfn = torch.func.grad(problem.objective, argnums=0)
        def calculate():
            with shared.transport.selected_face_extension(
                    FACE_AXIS, FACE_ROW, FACE_COLUMN, sign):
                return torch.func.jvp(lambda point: gfn(point, parameters),
                    (control,), (direction,))[1]
        return shared._counted_hvp(record, output, deadline,
            {"phase": "current_tangent", "side": sign,
             "base_control_sha256": _tensor_sha(control), "theta": model_state["theta"],
             "operator": "selected_face_extension", "scope": "one current chart tangent"}, calculate)

    def chart_pivot(control: Tensor) -> int:
        return geometry._pivot(control, weights)

    def chart_candidate(control: Tensor, direction: Tensor, alpha: float) -> Tensor:
        pivot = model_state["pivot"]
        retained = [i for i in range(control.numel()) if i != pivot]
        coordinates = control.clone()
        coordinates[retained] = coordinates[retained] + alpha * direction[retained]
        return geometry._chart(coordinates, weights, pivot, 0.0)

    def candidate_eval(candidate: Tensor, theta: float, alpha: float) -> dict[str, Any]:
        shared._deadline(deadline, POLICY["internal_seconds"])
        observed = _observe(shared, problem, candidate, parameters, weights)
        gm0, gp0 = model_state["gm"], model_state["gp"]
        model = tangent_model(gm0, gp0, model_state["hminus"], model_state["hplus"],
            model_state["normal"], model_state["z"], model_state["theta"],
            q=float(model_state["observed"]["q"]), pivot=model_state["pivot"])
        ftrial = torch.cat(((1 - theta) * observed["side"][-1][1]
                             + theta * observed["side"][1][1],
                             (observed["q"] / FACE_SCALE).reshape(1)))
        f2 = float(torch.dot(ftrial, ftrial))
        fbase2 = float(torch.dot(model.residual, model.residual))
        fbound = fbase2 + 2 * F_C1 * alpha * model.merit_product
        jslope = max(float(torch.dot(gm0, model.direction)),
                     float(torch.dot(gp0, model.direction)))
        jbound = float(model_state["observed"]["native_j"]) + J_C1 * alpha * jslope
        return {"objective": float(observed["native_j"]), "theta": theta,
            "J_armijo_passed": bool(float(observed["native_j"]) <= jbound),
            "J_armijo_bound": jbound, "J_actual_path_slope": jslope,
            "F_squared": f2, "F_squared_armijo_bound": fbound,
            "F_squared_armijo_passed": f2 <= fbound,
            "face_value": float(observed["q"]),
            "production_face_value": float(observed["production_q"]),
            "face_roundoff_bound": observed["face_bound"],
            "face_audit_passed": observed["face_ok"],
            "branch_pair_passed": observed["pair"]["passed"],
            "current_branch_pair_gate": observed["pair"],
            "side_objectives_match_native": observed["side_objectives_match_native"],
            "side_gradients_finite": observed["side_gradients_finite"],
            "side_gradients": {str(s): observed["side"][s][1].tolist() for s in (-1, 1)},
            "branch_trace": {str(s): observed["traces"][s] for s in (-1, 1)}}

    def final_repeat(proposal: dict[str, Any]) -> dict[str, Any]:
        shared._deadline(deadline, POLICY["internal_seconds"])
        candidate = torch.as_tensor(proposal["control"], dtype=base["control"].dtype)
        observed = _observe(shared, problem, candidate, parameters, weights)
        native_match = abs(float(observed["native_j"]) - float(proposal["objective"])) <= 128 * torch.finfo(candidate.dtype).eps * max(
            abs(float(observed["native_j"])), abs(float(proposal["objective"])), torch.finfo(candidate.dtype).tiny)
        f = torch.cat(((1 - proposal["theta"]) * observed["side"][-1][1]
                       + proposal["theta"] * observed["side"][1][1],
                       (observed["q"] / FACE_SCALE).reshape(1)))
        merit = float(torch.dot(f, f))
        merit_match = abs(merit - float(proposal["F_squared"])) <= 128 * torch.finfo(candidate.dtype).eps * max(
            abs(merit), abs(float(proposal["F_squared"])), torch.finfo(candidate.dtype).tiny)
        repeated_input = shared._input_identity(problem, original, candidate, parameters, truth)
        fixed_ok = geometry.model._fixed_input(repeated_input, input_before, _tensor_sha(candidate))
        runtime_ok = shared._runtime() == runtime_before
        source_ok = shared._source_hashes(plan, plan_path, plan_sha) == source_before
        deadline_ok = time.monotonic() < deadline
        closure_state["pending_observed"] = observed
        return {"theta": float(proposal["theta"]), "control": proposal["control"],
            "objective": float(observed["native_j"]), "F_squared": merit,
            "side_gradients_finite": observed["side_gradients_finite"],
            "side_gradients": {str(s): observed["side"][s][1].tolist() for s in (-1, 1)},
            "native_objective_matches_proposal": native_match,
            "side_objectives_match_native": observed["side_objectives_match_native"],
            "merit_matches_proposal": merit_match,
            "face_audit_passed": observed["face_ok"],
            "branch_pair_passed": observed["pair"]["passed"],
            "trace_matches_proposal": all(observed["traces"][s]["signature_sha256"]
                == proposal.get("branch_trace", {}).get(str(s), {}).get("signature_sha256")
                for s in (-1, 1)),
            "branch_trace": {str(s): observed["traces"][s] for s in (-1, 1)},
            "current_branch_pair_gate": observed["pair"],
            "fixed_input_unchanged": bool(fixed_ok), "runtime_unchanged": runtime_ok,
            "source_unchanged": source_ok, "deadline_passed": deadline_ok}

    # Retain the accepted point in state only after fresh final closure succeeds.
    def closed_repeat(proposal: dict[str, Any]) -> dict[str, Any]:
        repeat = final_repeat(proposal)
        if fresh_final_closure(proposal, repeat):
            closure_state["control"] = torch.as_tensor(proposal["control"], dtype=base["control"].dtype)
            closure_state["theta"] = float(proposal["theta"])
            closure_state["observed"] = closure_state.pop("pending_observed")
            closure_state["accepted_count"] += 1
            record["last_confirmed_control"] = closure_state["control"].tolist()
            record["last_confirmed_control_sha256"] = _tensor_sha(closure_state["control"])
            record["last_confirmed_theta"] = closure_state["theta"]
            record["last_confirmed_iterations"] = closure_state["accepted_count"]
            record["last_confirmed_closure"] = repeat
        return repeat

    def candidate_eval_with_model(candidate: Tensor, theta: float, alpha: float):
        model_state["hminus"], model_state["hplus"] = (
            model_state["latest_hminus"], model_state["latest_hplus"])
        return candidate_eval(candidate, theta, alpha)

    # Capture each fresh HVP for the current point; the model itself is built by the shared kernel.
    def tracked_hvp(control: Tensor, direction: Tensor, sign: int) -> Tensor:
        result = side_hvp(control, direction, sign)
        model_state["latest_hminus" if sign < 0 else "latest_hplus"] = result
        return result

    def committed_progress(iterations: list[dict[str, Any]]) -> None:
        confirmed = closure_state["control"]
        record["iterations"] = iterations
        record["optimizer_steps_applied"] = closure_state["accepted_count"]
        record["accepted_iterations"] = closure_state["accepted_count"]
        record["candidate_committed"] = True
        record["current_control"] = confirmed.tolist()
        record["current_control_sha256"] = _tensor_sha(confirmed)
        record["current_theta"] = closure_state["theta"]
        record["last_confirmed_control"] = confirmed.tolist()
        record["last_confirmed_control_sha256"] = _tensor_sha(confirmed)
        record["last_confirmed_theta"] = closure_state["theta"]
        record["last_confirmed_iterations"] = closure_state["accepted_count"]
        _write(output, record)

    result = bounded_continuation(base["control"], base["theta"], current_products,
        tracked_hvp, chart_candidate, candidate_eval_with_model, closed_repeat,
        chart_pivot=chart_pivot, committed_progress=committed_progress)
    control, theta = result["control"], result["theta"]
    final_observed = (closure_state.get("observed") if torch.equal(control, closure_state["control"])
        else None) or model_state.get("current_observed", initial)
    final_input = shared._input_identity(problem, original, control, parameters, truth)
    fixed_ok = geometry.model._fixed_input(final_input, input_before, _tensor_sha(control))
    runtime_after = shared._runtime()
    source_after = shared._source_hashes(plan, plan_path, plan_sha)
    source_ok, runtime_ok = source_after == source_before, runtime_after == runtime_before
    deadline_ok = time.monotonic() < deadline
    closure_ok = (final_observed["pair"]["passed"] and final_observed["face_ok"]
        and final_observed["side_objectives_match_native"] and final_observed["side_gradients_finite"]
        and fixed_ok and runtime_ok and source_ok and deadline_ok
        and record["hvp_calls_started"] == record["hvp_calls_completed"])
    committed = result["accepted_iterations"]
    if committed == MAX_ACCEPTED:
        numerical_status = "tangent_continuation_cap_reached"
    elif committed:
        numerical_status = "tangent_continuation_stopped_after_commit"
    else:
        numerical_status = "tangent_continuation_refused"
    record.update(phase="finished", execution_status="completed" if closure_ok else "failed",
        numerical_status=numerical_status if closure_ok else "diagnostic_closure_failed",
        candidate_type=result["candidate_type"], iterations=result["iterations"],
        accepted_iterations=committed, optimizer_steps_applied=committed,
        candidate_committed=committed > 0, active_candidate_committed=False,
        current_control=control.tolist(), current_control_sha256=_tensor_sha(control),
        current_theta=theta, native_objective=float(final_observed["native_j"]),
        final_current_branch_pair=final_observed["pair"],
        final_face_audit={"passed": final_observed["face_ok"],
            "value": float(final_observed["q"]),
            "production_value": float(final_observed["production_q"]),
            "roundoff_bound": final_observed["face_bound"]},
        input_before=input_before, input_after=final_input,
        fixed_input_unchanged=bool(fixed_ok), runtime=runtime_before,
        runtime_after=runtime_after, runtime_unchanged=runtime_ok,
        source_before=source_before, source_after=source_after, source_unchanged=source_ok,
        deadline_passed=deadline_ok, elapsed_seconds=time.monotonic() - start,
        full_smooth_root=False, minimum_claim=False, response_claim=False)
    if not closure_ok:
        record.update(refusal="final source/input/runtime/deadline or numerical closure failed",
            current_control=closure_state["control"].tolist(),
            current_control_sha256=_tensor_sha(closure_state["control"]),
            current_theta=closure_state["theta"])
    _write(output, record)
    return record


def _run_child(plan_path: Path, plan_sha: str, output: Path, *,
               plan_loader: Callable[[Path, str], dict[str, Any]] | None = None,
               base_loader: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
               base_control_sha256: str | None = None,
               initial_theta: float | None = None) -> dict[str, Any]:
    expected_base_sha = base_control_sha256 or BASE_CONTROL_SHA
    fallback_theta = BASE_THETA if initial_theta is None else initial_theta
    try:
        if not output.exists():
            _write(output, {"phase": "running", "execution_status": "running",
                "numerical_status": "preflight", "plan_sha256": plan_sha,
                "base_control_sha256": expected_base_sha, "current_theta": fallback_theta,
                "candidate_committed": False,
                "active_candidate_committed": False, "optimizer_steps_applied": 0,
                "hvp_calls_started": 0, "hvp_calls_completed": 0,
                "full_smooth_root": False, "minimum_claim": False, "response_claim": False})
        injected: dict[str, Any] = {}
        if plan_loader is not None:
            injected["plan_loader"] = plan_loader
        if base_loader is not None:
            injected["base_loader"] = base_loader
        if base_control_sha256 is not None:
            injected["base_control_sha256"] = base_control_sha256
        return _run_child_impl(plan_path, plan_sha, output, **injected)
    except Exception as error:
        try:
            record = json.loads(output.read_text())
        except (OSError, ValueError):
            record = {}
        checkpoint_fields = ("last_confirmed_control", "last_confirmed_control_sha256",
            "last_confirmed_theta", "last_confirmed_iterations")
        has_checkpoint = any(key in record for key in checkpoint_fields)
        try:
            confirmed = torch.as_tensor(record.get("last_confirmed_control", []),
                                        dtype=torch.float64)
        except (OverflowError, TypeError, ValueError, RuntimeError):
            confirmed = torch.empty(0, dtype=torch.float64)
        confirmed_sha = record.get("last_confirmed_control_sha256")
        confirmed_theta = record.get("last_confirmed_theta")
        confirmed_count = record.get("last_confirmed_iterations")
        theta_valid = (isinstance(confirmed_theta, (int, float))
            and not isinstance(confirmed_theta, bool)
            and 0.0 <= confirmed_theta <= 1.0 and math.isfinite(float(confirmed_theta)))
        count_valid = (isinstance(confirmed_count, int) and not isinstance(confirmed_count, bool)
            and 0 <= confirmed_count <= MAX_ACCEPTED)
        committed = confirmed_sha is not None and confirmed_sha != expected_base_sha
        control_valid = (confirmed.shape == (26,) and bool(torch.isfinite(confirmed).all())
                         and confirmed_sha == _tensor_sha(confirmed))
        commit_count_valid = (count_valid and
            (confirmed_count != 0 if committed else confirmed_count == 0))
        checkpoint_valid = (has_checkpoint and control_valid and theta_valid
                            and commit_count_valid)
        committed_count = confirmed_count if checkpoint_valid and committed else 0
        invalid_checkpoint = has_checkpoint and not checkpoint_valid
        record.update(phase="finished", execution_status="failed",
            numerical_status=("invalid_checkpoint" if invalid_checkpoint else
                "internal_deadline_refused" if isinstance(error, TimeoutError)
                else "prerequisite_or_execution_refused"),
            refusal=f"{type(error).__name__}: {error}", candidate_committed=committed,
            active_candidate_committed=False, optimizer_steps_applied=committed_count,
            accepted_iterations=committed_count,
            full_smooth_root=False, minimum_claim=False, response_claim=False)
        if invalid_checkpoint:
            record["refusal"] = f"invalid last-confirmed checkpoint; {record['refusal']}"
            record.update(candidate_committed=False, optimizer_steps_applied=0,
                          accepted_iterations=0)
            record.pop("last_confirmed_closure", None)
            for key in checkpoint_fields:
                record.pop(key, None)
            try:
                base_control = torch.as_tensor(record.get("current_control", []),
                                               dtype=torch.float64)
            except (OverflowError, TypeError, ValueError, RuntimeError):
                base_control = torch.empty(0, dtype=torch.float64)
            exact_base = (base_control.shape == (26,)
                and record.get("current_control_sha256") == expected_base_sha
                and _tensor_sha(base_control) == expected_base_sha)
            record["current_control_sha256"] = expected_base_sha
            if exact_base:
                record["current_control"] = base_control.tolist()
                record["current_theta"] = fallback_theta
            else:
                record.pop("current_control", None)
                record.pop("current_theta", None)
        elif checkpoint_valid:
            record.update(current_control=confirmed.tolist(), current_control_sha256=confirmed_sha,
                          current_theta=confirmed_theta)
        else:
            record.update(candidate_committed=False, optimizer_steps_applied=0,
                          accepted_iterations=0, current_control_sha256=expected_base_sha)
            record.pop("current_control", None)
            record.pop("current_theta", None)
        _write(output, record)
        return record


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path, *,
        plan_loader: Callable[[Path, str], dict[str, Any]] | None = None,
        child_script: Path | None = None,
        base_control_sha256: str | None = None) -> dict[str, Any]:
    if plan_path.is_symlink():
        raise ValueError("tangent-continuation plan must not be a symbolic link")
    plan_path = plan_path.resolve()
    plan = (plan_loader or _load_plan)(plan_path, plan_sha)
    expected_base_sha = base_control_sha256 or plan.get("base_control_sha256", BASE_CONTROL_SHA)
    if any(path.exists() or path.is_symlink() for path in
           (output, resource, log, output.with_suffix(".run.json"))):
        raise ValueError("output/resource/log paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"),
        str((child_script or Path(__file__)).resolve()),
        "--child", "--plan", str(plan_path), "--plan-sha256", plan_sha,
        "--output", str(output)]
    resource_result = run_guarded_diagnostic(command,
        wall_seconds=float(plan["policy"]["outer_seconds"]),
        rss_bytes=int(plan["policy"]["rss_bytes"]), report_path=resource, log_path=log)
    status = shared._execution_status(resource_result,
        wall_seconds=float(plan["policy"]["outer_seconds"]),
        rss_bytes=int(plan["policy"]["rss_bytes"]))
    parent: dict[str, Any] = {"execution_status": status, "resource": resource_result,
        "child_sha256": _sha(output) if output.exists() else None,
        "numerical_status": "not_reached", "child_read_error": None}
    child: dict[str, Any] | None = None
    if output.exists():
        try:
            child = json.loads(output.read_text())
            if not isinstance(child, dict):
                raise ValueError("child report must be a JSON object")
        except (OSError, ValueError) as error:
            parent.update(execution_status="failed",
                child_read_error=f"{type(error).__name__}: {error}")
            _write(output.with_suffix(".run.json"), parent)
            return {"parent": parent, "child": None}
        parent["numerical_status"] = child.get("numerical_status")
        iterations = child.get("iterations", [])
        accepted = [item for item in iterations if item.get("accepted") is True] \
            if isinstance(iterations, list) else []
        accepted_controls = [item.get("committed_control") for item in accepted]
        final_commit_matches = ((not accepted
            and child.get("current_control_sha256") == expected_base_sha
            and child.get("current_theta") == plan.get("initial_theta"))
            or (len(accepted) > 0 and child.get("current_control") == accepted_controls[-1]
                and child.get("current_theta") == accepted[-1].get("committed_theta")))
        history = child.get("hvp_history", [])
        history_ok = (isinstance(history, list)
            and len(history) == child.get("hvp_calls_completed")
            and all(item.get("status") == "completed" for item in history))
        for item in accepted:
            current_point_hvps = [entry for entry in history
                if entry.get("base_control_sha256") == item.get("base_control_sha256")]
            history_ok = history_ok and len(current_point_hvps) == 2 \
                and {entry.get("side") for entry in current_point_hvps} == {-1, 1}
        closed = (child.get("phase") == "finished"
            and child.get("plan_sha256") == plan_sha
            and child.get("base_control_sha256") == expected_base_sha
            and child.get("execution_status") == "completed"
            and child.get("source_unchanged") is True
            and child.get("fixed_input_unchanged") is True
            and child.get("runtime_unchanged") is True
            and child.get("deadline_passed") is True
            and child.get("active_candidate_committed") is False
            and child.get("candidate_committed") is (len(accepted) > 0)
            and child.get("accepted_iterations") == len(accepted)
            and child.get("optimizer_steps_applied") == len(accepted)
            and child.get("hvp_calls_started") == child.get("hvp_calls_completed")
            and child.get("hvp_calls_completed", 0) <= int(plan["policy"]["hvp_calls"])
            and history_ok
            and len(accepted) <= int(plan["policy"]["max_accepted_iterations"])
            and final_commit_matches
            and child.get("current_control_sha256") == _tensor_sha(torch.as_tensor(
                child.get("current_control", []), dtype=torch.float64)))
        if status == "completed" and not closed:
            parent.update(execution_status="failed", execution_failure_reason="child closure refused")
        elif status != "completed":
            parent["execution_status"] = status
    elif status == "completed":
        parent.update(execution_status="failed", execution_failure_reason="guard exited zero without a child receipt")
    _write(output.with_suffix(".run.json"), parent)
    return {"parent": parent, "child": child}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path,
        default=EVIDENCE / "tangent_continuation_20261009_attempt1/step.json")
    parser.add_argument("--resource", type=Path,
        default=EVIDENCE / "tangent_continuation_20261009_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path,
        default=EVIDENCE / "tangent_continuation_20261009_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
