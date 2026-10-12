"""Offline current-model audit from the byte-pinned PR302 operator cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
RANK_FIX_WORK = ROOT / "graphify-out/fv-root-cause-20260919/subspace_rank_fix_20261012"
OUTPUT = WORK / "current_model_audit.json"
RADIUS = 1.8757177916958825e-5
RADIUS_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0)
BASE_SHA256 = "7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
if str(RANK_FIX_WORK) not in sys.path:
    sys.path.insert(0, str(RANK_FIX_WORK))

import cached_operators as cache
from examples.weather_scenarios.fv_point_3h_subspace_model import solve_two_direction_subspace


def _tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _finite_list(values: list[float], label: str) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64, device="cpu")
    if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"cached {label} must be a finite CPU FP64 vector")
    return tensor


def _model_by_name(models: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [model for model in models if model.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"cache must contain exactly one {name} model")
    return matches[0]


def _ray_metrics(G: torch.Tensor, direction: torch.Tensor, DGd: torch.Tensor,
                 fraction: float, direction_model: str) -> dict[str, float | str]:
    direction_norm = torch.linalg.vector_norm(direction)
    step_scale = fraction * RADIUS / float(direction_norm)
    predicted_residual = G + step_scale * DGd
    predicted_R = torch.dot(predicted_residual, predicted_residual)
    return {
        "direction_model": direction_model,
        "radius_multiple": fraction,
        "control_radius": fraction * RADIUS,
        "direction_scale": step_scale,
        "predicted_R": float(predicted_R),
        "predicted_R_decrease": float(torch.dot(G, G) - predicted_R),
    }


def run(output: Path = OUTPUT) -> dict[str, Any]:
    """Admit and summarize cached operators without FV, HVP, or score calls."""
    admitted = cache.admit()
    step, operator = admitted["step"], admitted["operator"]
    endpoint = admitted["endpoint_snapshot"]["base"]
    control = _finite_list(step["current_control"], "control")
    G = _finite_list(step["current_full_G"], "full_G")
    endpoint_control = _finite_list(endpoint["control"], "closed endpoint control")
    endpoint_G = _finite_list(endpoint["full_G"], "closed endpoint full_G")
    if (control.numel() != 26 or G.numel() != 27
            or endpoint_control.shape != control.shape
            or not torch.equal(control, endpoint_control)
            or not torch.equal(G, endpoint_G)
            or step.get("current_control_sha256") != BASE_SHA256
            or step.get("input_after") != endpoint["context"]):
        raise ValueError("cached current model differs from the closed 7839 endpoint/context")
    repeat = endpoint["repeat"]
    if (step.get("base_J") != repeat.get("objective")
            or step.get("base_R") != repeat.get("F_squared")
            or step.get("current_theta") != endpoint["theta"]):
        raise ValueError("cached base J/R/theta differ from the closed endpoint repeat")

    models = step.get("fresh_models")
    if not isinstance(models, list):
        raise ValueError("cached fresh model list is missing")
    gradient = _model_by_name(models, "baseline_tangent")
    gn = _model_by_name(models, "robust_gn_coupled")
    directions = torch.stack([
        _finite_list(gradient["direction"], "cached gradient direction"),
        _finite_list(gn["direction"], "cached GN direction"),
    ], dim=1)
    residual_directions = torch.stack([
        _finite_list(gradient["residual_direction"], "cached gradient DGd"),
        _finite_list(gn["residual_direction"], "cached GN DGd"),
    ], dim=1)
    if directions.shape != (26, 2) or residual_directions.shape != (27, 2):
        raise ValueError("cached directions must have control/residual dimensions 26 and 27")

    # Preserve both one-sided objective HVPs as receipt data; do not call an HVP.
    side_hvps: dict[str, dict[str, Any]] = {}
    for model in (gradient, gn):
        sides = model.get("side_HVP")
        if not isinstance(sides, dict) or set(sides) != {"-1", "1"}:
            raise ValueError(f"{model['name']} is missing its cached two-sided HVP pair")
        side_hvps[model["name"]] = {}
        for side in ("-1", "1"):
            hvp = _finite_list(sides[side], f"{model['name']} side HVP {side}")
            if hvp.shape != control.shape:
                raise ValueError("cached side HVP is not a 26-control vector")
            side_hvps[model["name"]][side] = {
                "dimension": int(hvp.numel()),
                "l2": float(torch.linalg.vector_norm(hvp)),
                "sha256": _tensor_sha(hvp),
                "vector": hvp.tolist(),
            }

    # The first direction is the no-GN baseline. The second coefficient is the
    # incremental GN contribution and must clear a scale-aware roundoff gate.
    baseline_D = torch.stack((directions[:, 0], directions[:, 0]), dim=1)
    baseline_V = torch.stack((residual_directions[:, 0], residual_directions[:, 0]), dim=1)
    comparisons: list[dict[str, Any]] = []
    eps = torch.finfo(G.dtype).eps
    for multiple in RADIUS_MULTIPLIERS:
        radius = multiple * RADIUS
        baseline = solve_two_direction_subspace(G, baseline_D, baseline_V, radius)
        combined = solve_two_direction_subspace(G, directions, residual_directions, radius)
        scales = torch.linalg.vector_norm(directions, dim=0)
        normalized_coefficients = combined.coefficients * scales
        baseline_R = baseline.predicted_R
        combined_R = combined.predicted_R
        added_decrease = baseline_R - combined_R
        model_budget = 128.0 * eps * torch.maximum(baseline_R.abs(), combined_R.abs()).clamp_min(
            torch.finfo(G.dtype).tiny)
        coefficient_budget = 128.0 * eps * normalized_coefficients.norm().clamp_min(
            torch.finfo(G.dtype).tiny)
        gn_resolved = bool(normalized_coefficients[1] > coefficient_budget)
        gain_resolved = bool(added_decrease > model_budget)
        comparisons.append({
            "radius_multiple": multiple,
            "radius": radius,
            "baseline_tangent_R": float(baseline_R),
            "combined_GN_R": float(combined_R),
            "added_predicted_R_decrease": float(added_decrease),
            "normalized_coefficients": normalized_coefficients.tolist(),
            "GN_coefficient_budget": float(coefficient_budget),
            "model_decrease_budget": float(model_budget),
            "GN_coefficient_resolved": gn_resolved,
            "added_model_gain_resolved": gain_resolved,
            "GN_gate_passed": gn_resolved and gain_resolved,
            "selected_active_set": combined.diagnostics["selected_active_set"],
            "residual_derivative_rank": int(combined.diagnostics["residual_derivative_rank"]),
        })

    row_gradients = torch.tensor(
        [row["gradient"] for row in operator["jacobian_row_history"]], dtype=torch.float64)
    singular_values = torch.linalg.svdvals(row_gradients)
    rank_tolerance = 64.0 * eps * max(row_gradients.shape) * singular_values[0]
    row_rank = int((singular_values > rank_tolerance).sum())
    dense = operator["dense_solve_audit"]
    rays = [_ray_metrics(G, _finite_list(model["direction"], model["name"]),
                         _finite_list(model["residual_direction"], f"{model['name']} DGd"),
                         multiple, model["name"])
            for model in (gradient, gn) for multiple in RADIUS_MULTIPLIERS]
    report: dict[str, Any] = {
        "status": "offline_cached_current_model_audit_complete",
        "scope": "pinned PR302 bytes and stored-vector arithmetic only; no FV, HVP, candidate, or score calls",
        "admission": admitted["pins"],
        "base": {
            "control_sha256": BASE_SHA256,
            "control_dimension": int(control.numel()),
            "control": control.tolist(),
            "full_G_dimension": int(G.numel()),
            "full_G": G.tolist(),
            "theta": endpoint["theta"],
            "J": float(repeat["objective"]),
            "R": float(repeat["F_squared"]),
            "full_G_l2": float(torch.linalg.vector_norm(G)),
            "full_G_sha256": _tensor_sha(G),
            "repeat": repeat,
            "context": endpoint["context"],
            "parent_plan_sha256": endpoint["parent_plan_sha256"],
            "closed_endpoint_snapshot_sha256": admitted["pins"]["closed_endpoint_snapshot_sha256"],
        },
        "cached_gradient_direction": {
            "source_model": gradient["name"],
            "control_dimension": int(directions.shape[0]),
            "l2": float(torch.linalg.vector_norm(directions[:, 0])),
            "direction_sha256": _tensor_sha(directions[:, 0]),
            "direction": directions[:, 0].tolist(),
            "DGd_dimension": int(residual_directions.shape[0]),
            "DGd_l2": float(torch.linalg.vector_norm(residual_directions[:, 0])),
            "DGd_sha256": _tensor_sha(residual_directions[:, 0]),
            "DGd": residual_directions[:, 0].tolist(),
            "GN_direction": directions[:, 1].tolist(),
            "GN_DGd": residual_directions[:, 1].tolist(),
        },
        "cached_side_HVPs": side_hvps,
        "cached_operator_audit": {
            "jacobian_rows": int(row_gradients.shape[0]),
            "control_dimension": int(row_gradients.shape[1]),
            "row_gradient_rank": row_rank,
            "row_gradient_rank_tolerance": float(rank_tolerance),
            "legacy_GN_B_shape": [len(dense["B"]), len(dense["B"][0])],
            "legacy_GN_S_shape": [len(dense["S"]), len(dense["S"][0])],
            "legacy_dense_solves": int(operator["dense_solves_completed"]),
            "HVPs_started_completed": [int(step["hvp_started"]), int(step["hvp_completed"])],
            "elapsed_seconds": float(step["elapsed_seconds"]),
            "cache_run_wall_seconds": float(admitted["resource"]["elapsed_seconds"]),
            "candidate_evaluations": [int(step["candidate_evaluations_started"]),
                                       int(step["candidate_evaluations_completed"])],
            "commits": int(bool(step["candidate_committed"])),
            "current_kkt_radius_selection": step["current_KKT_radius_selection"],
        },
        "cached_ray_R_models": rays,
        "incremental_GN_gate_by_radius": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": result["status"],
        "control_sha256": result["base"]["control_sha256"],
        "row_gradient_shape": [result["cached_operator_audit"]["jacobian_rows"],
                               result["cached_operator_audit"]["control_dimension"]],
        "incremental_GN_gate_by_radius": result["incremental_GN_gate_by_radius"],
        "output": str(OUTPUT),
    }, indent=2))
