"""Measured endpoint filter for one branch-adaptive point-root experiment."""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import math
import sys
from typing import Any

import torch

from advar.local_refinement import RefinementTrial


def _key(signature: Any) -> dict[str, Any]:
    if not isinstance(signature, dict) or not all(
        name in signature for name in ("choices", "face_signs")
    ):
        raise ValueError("sector policy needs full limiter choices and face signs")
    return {name: signature[name] for name in ("choices", "face_signs")}


def _digest(signature: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        signature, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def sector_trial_acceptance(
    trial: RefinementTrial,
    record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[bool, str] | None:
    """Use ordinary Armijo within a signature; filter switches by true data.

    No Hessian-slope model from the old signature is used for a switched
    endpoint. This checks only endpoint quantities, not a smooth path.
    """
    current = _key(trial.current_branch)
    candidate = _key(trial.candidate_branch)
    if current == candidate:
        return None
    safe_norm = math.sqrt(sys.float_info.max)
    current_phi = (0.5 * trial.current_gradient_norm * trial.current_gradient_norm
                   if math.isfinite(trial.current_gradient_norm)
                   and abs(trial.current_gradient_norm) <= safe_norm else math.inf)
    candidate_phi = (0.5 * trial.candidate_gradient_norm * trial.candidate_gradient_norm
                     if math.isfinite(trial.candidate_gradient_norm)
                     and abs(trial.candidate_gradient_norm) <= safe_norm else math.inf)
    metrics = {
        "objective": (trial.current_objective, trial.candidate_objective),
        "gradient_merit": (current_phi, candidate_phi),
        "gradient_max": (trial.current_gradient_max, trial.candidate_gradient_max),
    }
    epsilon = torch.finfo(torch.float64).eps
    tiny = torch.finfo(torch.float64).tiny
    comparisons: dict[str, dict[str, float | bool | None]] = {}
    failed: list[str] = []
    for name, (old, new) in metrics.items():
        if not math.isfinite(old) or not math.isfinite(new):
            comparisons[name] = {"old": old if math.isfinite(old) else None,
                                 "new": new if math.isfinite(new) else None,
                                 "delta": None, "floor": None, "passed": False}
            failed.append(name)
            continue
        delta = old - new
        floor = 128 * epsilon * max(abs(old), abs(new), tiny)
        passed = math.isfinite(delta) and math.isfinite(floor) and delta > floor
        comparisons[name] = {"old": old, "new": new,
                             "delta": delta if math.isfinite(delta) else None,
                             "floor": floor if math.isfinite(floor) else None,
                             "passed": passed}
        if not passed:
            failed.append(name)
    accepted = not failed
    reason = ("sector_switch_measured_decrease" if accepted else
              "sector_switch_refused_" + "_".join(failed))
    if record is not None:
        record({
            "iteration": trial.iteration, "backtrack": trial.backtrack,
            "step_scale": trial.step_scale,
            "current_signature_sha256": _digest(current),
            "candidate_signature_sha256": _digest(candidate),
            "comparisons": comparisons,
            "accepted": accepted, "reason": reason,
        })
    return accepted, reason
