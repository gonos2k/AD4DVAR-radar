"""Provisional, bounded candidate policies for the point FV experiment."""
from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


POLICY_VERSION = "fv-point-3h-policy-search-v1"
ELIGIBILITY_VERSION = "all-gates-plus-policy-v1"
DEFAULT_POLICY = {
    "name": "dyadic_JR", "gamma": 4.0, "expansion_factor": 2.0,
    "max_expansions": 3, "max_evaluations": 24, "max_backtracks": 24,
    "Rcap_fraction": 0.01, "radius": 0.05,
}
_ALL_GATES = (
    "mixing_minimum_valid", "J_armijo_passed", "F_squared_armijo_passed",
    "branch_pair_passed", "face_audit_passed", "side_objectives_match_native",
    "side_gradients_finite", "finite",
)


def search(model: Any, current: Tensor,
        evaluate: Callable[[Tensor, float, float], dict[str, Any]], *,
        chart_candidate: Callable[[Tensor, Tensor, float], Tensor],
        progress: Callable[[list[dict[str, Any]], dict[str, Any]], None],
        policy: Mapping[str, Any] | None = None,
        previous_movement: float | None = None,
        reference_R: float | None = None
        ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """Return a provisional eligible trial; this helper never commits state."""
    config = {**DEFAULT_POLICY, **dict(policy or {})}
    name = config["name"]
    if name not in {"dyadic_JR", "seeded_JR", "bounded_J_Rcap"}:
        raise ValueError(f"unknown candidate policy: {name}")
    radius = float(config["radius"])
    gamma = float(config["gamma"])
    expansion_factor = float(config["expansion_factor"])
    max_expansions = int(config["max_expansions"])
    rcap_fraction = float(config["Rcap_fraction"])
    max_evaluations = int(config["max_evaluations"])
    max_backtracks = int(config["max_backtracks"])
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(gamma) or gamma <= 0 \
            or not math.isfinite(expansion_factor) or expansion_factor <= 1 \
            or not 0 <= max_expansions <= 3 or not math.isfinite(rcap_fraction) or rcap_fraction < 0 \
            or not 0 <= max_evaluations <= 24 \
            or not 0 <= max_backtracks <= 24:
        raise ValueError("policy radius and candidate budgets are invalid")

    norm = float(torch.linalg.vector_norm(model.direction))
    model_alphas = tangent.candidate_alphas(model, radius=radius, limit=24) \
        if model.gates.get("merit_descends") else []
    model_alpha = model_alphas[0] if model_alphas else None
    initial_alpha = model_alpha
    proposal_cap = model_alpha
    if name == "seeded_JR":
        if model_alpha is not None and previous_movement is not None \
                and math.isfinite(float(previous_movement)) and previous_movement > 0 \
                and math.isfinite(norm) and norm > 0:
            initial_alpha = min(model_alpha, gamma * float(previous_movement) / norm)
        else:
            initial_alpha = None
    elif name == "bounded_J_Rcap":
        sides = getattr(model, "side_products", ())
        if model_alpha is None and len(sides) == 2 \
                and all(math.isfinite(float(s)) and float(s) < 0 for s in sides) \
                and math.isfinite(norm) and norm > 0:
            proposal_cap = radius / norm
        if proposal_cap is not None and previous_movement is not None \
                and math.isfinite(float(previous_movement)) and previous_movement > 0 \
                and math.isfinite(norm) and norm > 0:
            initial_alpha = min(proposal_cap, gamma * float(previous_movement) / norm)
        else:
            initial_alpha = proposal_cap

    meta: dict[str, Any] = {
        "policy_version": POLICY_VERSION, "eligibility_version": ELIGIBILITY_VERSION,
        "policy": name, "policy_parameters": {k: config[k] for k in (
            "gamma", "expansion_factor", "max_expansions", "max_evaluations",
            "max_backtracks", "Rcap_fraction", "radius")},
        "slot_cap": max_evaluations, "computed_model_alpha": model_alpha,
        "initial_alpha": initial_alpha, "previous_movement": previous_movement,
        "reference_R": reference_R, "evaluated_count": 0,
        "slots_started": 0, "slots_completed": 0,
        "evaluations_started": 0, "evaluations_completed": 0,
        "first_pass_candidate_index": None, "first_pass_count": 0,
        "selected_index": None, "selected_reason": None,
    }
    trials: list[dict[str, Any]] = []
    progress(trials, copy.deepcopy(meta))
    if max_evaluations == 0 or max_backtracks == 0:
        meta["status"] = "gate_refused"
        progress(trials, copy.deepcopy(meta))
        return None, trials, meta
    if initial_alpha is None or initial_alpha <= 0 or not math.isfinite(initial_alpha):
        meta["status"] = "gate_refused"
        progress(trials, copy.deepcopy(meta))
        return None, trials, meta

    def slot(alpha: float) -> dict[str, Any]:
        index = len(trials) + 1
        trial: dict[str, Any] = {"index": index, "alpha": alpha, "accepted": False,
            "status": "slot_started", "policy": name}
        trials.append(trial)
        meta["slots_started"] += 1
        meta["evaluated_count"] = meta["slots_started"]
        progress(trials, copy.deepcopy(meta))
        try:
            candidate = chart_candidate(current, model.direction, alpha)
            movement_tensor = torch.linalg.vector_norm(candidate - current)
            movement = float(movement_tensor)
            trial["actual_path_norm"] = movement if math.isfinite(movement) else None
            finite_candidate = bool(torch.isfinite(candidate).all())
            radius_ok = math.isfinite(movement) and movement > 0 and movement <= radius * (
                1.0 + 64.0 * torch.finfo(current.dtype).eps)
            if not finite_candidate or not math.isfinite(movement):
                trial["status"] = "nonfinite_chart_movement_refused"
            elif not radius_ok:
                trial["status"] = "actual_chart_radius_refused" if movement > 0 else "zero_control_movement_refused"
            else:
                trial.update(control=candidate.tolist(), control_sha256=tangent._tensor_sha(candidate))
                trial["status"] = "evaluation_started"
                meta["evaluations_started"] += 1
                progress(trials, copy.deepcopy(meta))
                facts = evaluate(candidate, model.theta, alpha)
                meta["evaluations_completed"] += 1
                trial.update(facts)
                required_gates = _ALL_GATES if name != "bounded_J_Rcap" else tuple(
                    key for key in _ALL_GATES if key != "F_squared_armijo_passed")
                base_gates = all(facts.get(key) is True for key in required_gates)
                cap_ok = True
                if name == "bounded_J_Rcap":
                    actual_R = facts.get("F_squared", facts.get("actual_R", facts.get("actual_F_squared")))
                    cap = None if reference_R is None else float(reference_R) * (1.0 + rcap_fraction)
                    cap_ok = actual_R is not None and cap is not None and math.isfinite(float(actual_R)) \
                        and math.isfinite(cap) and float(actual_R) <= cap
                    trial["R_cap"] = cap
                    trial["R_cap_passed"] = bool(cap_ok)
                    if not facts.get("F_squared_armijo_passed"):
                        trial["F_squared_armijo_passed"] = False
                eligible = base_gates and (cap_ok if name == "bounded_J_Rcap" else
                    facts.get("F_squared_armijo_passed") is True)
                trial["policy_eligible"] = bool(eligible)
                trial["accepted"] = bool(eligible)
                trial["status"] = "eligible" if eligible else "gate_refused"
            meta["slots_completed"] += 1
            if trial["accepted"]:
                if meta["first_pass_candidate_index"] is None:
                    meta["first_pass_candidate_index"] = index
                    meta["first_pass_count"] = index
            progress(trials, copy.deepcopy(meta))
        except Exception as error:
            trial["accepted"] = False
            trial["policy_eligible"] = False
            trial["status"] = "slot_interrupted" if isinstance(error, TimeoutError) else "slot_failed"
            trial["error_type"] = type(error).__name__
            trial["error_message"] = str(error)
            progress(trials, copy.deepcopy(meta))
            raise
        return trial

    # The ordinary policy reproduces the frozen dyadic first-pass stopping rule.
    if name == "dyadic_JR":
        for alpha in model_alphas[:min(max_evaluations, max_backtracks)]:
            candidate = slot(alpha)
            if candidate["accepted"]:
                meta["selected_index"] = candidate["index"]
                meta["selected_reason"] = "first eligible dyadic candidate"
                meta["status"] = "eligible"
                return candidate, trials, meta
        meta["status"] = "gate_refused"
        return None, trials, meta

    # Seeded and bounded policies backtrack to their first eligible point.
    first = slot(initial_alpha)
    for _ in range(min(max_backtracks, max_evaluations) - 1):
        if first["accepted"] or len(trials) >= max_evaluations:
            break
        first = slot(initial_alpha / (2 ** len(trials)))
    passed = [trial for trial in trials if trial.get("accepted") is True]
    if not passed:
        meta["status"] = "gate_refused"
        return None, trials, meta

    # Expand only when the initial point itself passed; stop on the first refusal.
    if len(trials) == 1 and passed[0] is first and proposal_cap is not None:
        for _ in range(max_expansions):
            if len(trials) >= max_evaluations:
                break
            alpha = min(proposal_cap, first["alpha"] * expansion_factor)
            if alpha <= first["alpha"]:
                break
            candidate = slot(alpha)
            if not candidate["accepted"]:
                break
            passed.append(candidate)
            first = candidate

    if name == "seeded_JR":
        selected = min(passed, key=lambda t: (float(t.get("F_squared", t.get("actual_R", math.inf))),
                                                float(t.get("objective", math.inf)), t["index"]))
        reason = "lowest R, then native J among eligible trials"
    else:
        selected = min(passed, key=lambda t: (float(t.get("objective", math.inf)),
                                                float(t.get("F_squared", t.get("actual_R", math.inf))), t["index"]))
        reason = "lowest native J, then R among eligible trials under fixed reference cap"
    meta["selected_index"] = selected["index"]
    meta["selected_reason"] = reason
    meta["status"] = "eligible"
    return selected, trials, meta
