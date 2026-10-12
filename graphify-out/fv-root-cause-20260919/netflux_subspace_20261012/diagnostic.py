"""Reconstruct stored direction-comparison finite errors without model calls."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_netflux_comparison_endpoint as endpoint

WORK = endpoint.WORK
OUTPUT = WORK / "candidate_model_error_blocks.json"
# The 26-control order is 20 echo values, five flow latents, and one growth latent.
# The constrained face residual is appended as the 27th component of full G.
BLOCKS = {"echo": (0, 20), "flow": (20, 25), "growth": (25, 26), "face": (26, 27)}


def _norm(values: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _block(values: list[float]) -> dict[str, Any]:
    return {"l2": _norm(values), "linf": max((abs(value) for value in values), default=0.0),
        "values": values}


def _selector_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                if isinstance(child, list):
                    found.extend(_flatten(child))
            else:
                found.extend(_selector_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_selector_values(child, key))
    return found


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [leaf for child in value for leaf in _flatten(child)]
    return [value]


def _selector_changes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for side in ("-1", "1"):
        a, b = left["branch_trace"][side], right["branch_trace"][side]
        for key in ("choose_left", "left_sign", "right_sign", "slope_sign"):
            before, after = _selector_values(a["choices"], key), _selector_values(b["choices"], key)
            if len(before) != len(after):
                raise ValueError("neighboring branch traces have different selector layouts")
            counts[f"side_{side}_{key}"] = sum(x != y for x, y in zip(before, after))
    return counts


def build_report() -> dict[str, Any]:
    admitted = endpoint.load_endpoint()
    raw = admitted["raw"]
    previous = endpoint.comparison.endpoint.load_endpoint()
    gn = next(arm for arm in raw["arms"] if arm["name"] == "robust_gn_coupled")
    base_g = [float(value) for value in gn["residual"]]
    direction_g = [float(value) for value in gn["residual_direction"]]
    if len(base_g) != 27 or len(direction_g) != 27:
        raise ValueError("stored GN residual vectors must have 27 entries")
    manifest = json.loads(endpoint._read(endpoint.RECEIPT_MANIFEST,
        endpoint.RECEIPT_MANIFEST_SHA))
    receipts = endpoint.original_endpoint._bundle_members(
        endpoint._read(endpoint.RECEIPTS_TAR, endpoint.RECEIPTS_TAR_SHA), set(manifest["files"]))
    output: dict[str, Any] = {
        "kind": "stored_vector_arithmetic_only",
        "base_control_sha256": endpoint.PARENT_CONTROL_SHA,
        "accepted_control_sha256": endpoint.BASE_SHA,
        "base_theta": previous["theta"],
        "endpoint_theta": admitted["theta"],
        "theta_chain": {"base": previous["theta"],
            "old_theta_P2_match": raw["previous_P2_match"],
            "minimum_mixture_theta_P2_match": raw["minimum_mixture_P2_match"],
            "selected_repeat": admitted["repeat"]["theta"]},
        "endpoint_theta_correction": {},
        "blocks": {name: {"start": start, "stop": stop} for name, (start, stop) in BLOCKS.items()},
        "candidates": [],
        "selector_transition": {"from": 9, "to": 10,
            "scope": "neighboring rejected-to-accepted candidate branch traces; no base selector trace is stored",
            "changed_counts": None},
        "receipt_manifest_sha256": endpoint.RECEIPT_MANIFEST_SHA,
        "receipt_count": len(manifest["files"]),
    }
    candidates: dict[int, dict[str, Any]] = {}
    base_theta = float(previous["theta"])
    for index in (9, 10):
        receipt_name = f"attempt1/robust_gn_coupled/candidate_{index:02d}.json"
        payload = json.loads(receipts[receipt_name])
        expected = manifest["files"][receipt_name]
        if endpoint._sha_bytes(receipts[receipt_name]) != expected["sha256"]:
            raise ValueError(f"candidate receipt changed: {receipt_name}")
        alpha = float(payload["alpha"])
        actual_g = [float(value) for value in payload["G"]]
        predicted_g = [base + alpha * slope for base, slope in zip(base_g, direction_g)]
        error = [actual - predicted for actual, predicted in zip(actual_g, predicted_g)]
        if len(actual_g) != 27:
            raise ValueError("stored candidate G must have 27 entries")
        candidates[index] = payload
        side_minus = [float(value) for value in payload["side_gradients"]["-1"]]
        side_plus = [float(value) for value in payload["side_gradients"]["1"]]
        candidate_theta = float(payload["theta"])
        mixed_theta_correction = [(candidate_theta - base_theta) * (plus - minus)
            for minus, plus in zip(side_minus, side_plus)]
        mixed_at_candidate = [(1.0 - candidate_theta) * minus + candidate_theta * plus
            for minus, plus in zip(side_minus, side_plus)]
        if max(abs(a - b) for a, b in zip(mixed_at_candidate, actual_g[:26])) > 1e-12:
            raise ValueError("candidate G first 26 components do not match its stored theta mixture")
        output["candidates"].append({
            "candidate_index": index,
            "receipt": receipt_name,
            "receipt_sha256": expected["sha256"],
            "status": "accepted" if payload["F_squared_armijo_passed"] else "rejected_R_armijo",
            "alpha": alpha,
            "theta": payload["theta"],
            "theta_delta_from_base": candidate_theta - base_theta,
            "theta_mixed_correction_blocks": {name: _block(mixed_theta_correction[a:b])
                for name, (a, b) in BLOCKS.items() if b <= 26},
            "J": payload["objective"],
            "R": payload["F_squared"],
            "model_error_norm_stored": payload["model_error_norm"],
            "model_error_norm_recomputed": _norm(error),
            "actual_G_blocks": {name: _block(actual_g[a:b]) for name, (a, b) in BLOCKS.items()},
            "predicted_G_blocks": {name: _block(predicted_g[a:b]) for name, (a, b) in BLOCKS.items()},
            "finite_error_blocks": {name: _block(error[a:b]) for name, (a, b) in BLOCKS.items()},
            "branch_signature_matches_base": payload["branch_signature_matches_base"],
            "branch_signature_sha256": {side: payload["branch_trace"][side]["signature_sha256"]
                for side in ("-1", "1")},
            "branch_nonfinite_or_tie": {side: payload["branch_trace"][side]["nonfinite_or_tie"]
                for side in ("-1", "1")},
        })
    output["selector_transition"]["changed_counts"] = _selector_changes(candidates[9], candidates[10])
    final_repeat = admitted["repeat"]
    side_minus = [float(value) for value in final_repeat["side_gradients"]["-1"]]
    side_plus = [float(value) for value in final_repeat["side_gradients"]["1"]]
    final_theta = float(final_repeat["theta"])
    theta_jump = [plus - minus for minus, plus in zip(side_minus, side_plus)]
    output["endpoint_theta_correction"] = {
        "base_theta": base_theta,
        "endpoint_theta": final_theta,
        "delta_theta": final_theta - base_theta,
        "side_gradient_jump_blocks": {name: _block(theta_jump[a:b])
            for name, (a, b) in BLOCKS.items() if b <= 26},
        "delta_theta_times_jump_blocks": {name: _block(
            [(final_theta - base_theta) * value for value in theta_jump[a:b]])
            for name, (a, b) in BLOCKS.items() if b <= 26},
    }
    return output


def main() -> None:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
