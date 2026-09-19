"""Bounded shared-solver FV reanalysis check for the C5c GN response.

The nominal analysis is followed by four nominal-control-initialized perturbed analyses at +/- h for
h=1e-3 and 5e-4.  This is a fixed-statistics experiment: ``y`` and the
declared ``B=y[0]`` dependency change, while masks, support, weights,
verification, geometry, and future boundary schedules remain fixed.  It is
finite reanalysis evidence, not an optimizer-path or FSOI certificate.
"""

from __future__ import annotations

import argparse
import json
import runpy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from advar import variational as v
from advar.fv_sensitivity import compute_fv_observation_response, face_branch_margin, face_signs
from advar.physics import echo_to_dbz


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "examples/weather_scenarios/fv_sensitivity_probe.py"


def _config(frozen: Any) -> Any:
    return replace(
        frozen.analysis_config,
        maximum_outer_iterations=16,
        maximum_pcg_iterations=96,
        gradient_tolerance=1e-10,
        step_tolerance=1e-12,
        pcg_relative_tolerance=1e-10,
    )


def _solve(y: torch.Tensor, observations: Any, frozen: Any, control: torch.Tensor) -> tuple[Any, Any]:
    """Run the same public solver path for an observation reanalysis."""
    # Replace only y and B in the fixed contract returned by make_case.
    endpoint_observations = replace(observations, dbz=y)
    endpoint_frozen = replace(
        frozen,
        analysis_config=_config(frozen),
        initial_background_dbz=y[0],
    )
    return v.solve_analysis(endpoint_observations, endpoint_frozen, control=control), endpoint_frozen


def _diagnostics(result: Any) -> dict[str, Any]:
    return {
        "reason": result.reason,
        "selected_gradient_norm": result.selected_gradient_norm,
        "outer_iterations": result.outer_iterations,
        "pcg_iterations": result.pcg_iterations,
        "initial_objective": result.initial_objective,
        "final_objective": result.final_objective,
        "stationarity_verified": result.stationarity_verified,
    }


def run_probe() -> dict[str, Any]:
    started = time.monotonic()
    namespace = runpy.run_path(str(PROBE))
    make_case = namespace["make_case"]
    observations, frozen, future, support = make_case()
    frozen = replace(frozen, analysis_config=_config(frozen))

    # Required nominal solve.  Perturbation budget below is exactly four.
    nominal = v.solve_analysis(observations, frozen)
    base_y = observations.dbz
    control = nominal.control
    def contract(y: torch.Tensor, base: Any) -> Any:
        return replace(base, initial_background_dbz=y[0])

    def forecast(c: torch.Tensor, y: torch.Tensor, base: Any) -> torch.Tensor:
        trajectory = v.forecast_fv_analysis(
            c,
            contract(y, base),
            leads=2,
            boundary_start_interval=2,
            boundary_echo=future,
            boundary_support=support,
        )
        return echo_to_dbz(trajectory.frames_linear[1:], min_dbz=-10.0)

    prediction = forecast(control, base_y, frozen).detach()
    metric_weight = torch.linspace(
        0.2, 1.0, prediction.numel(), dtype=control.dtype
    ).reshape_as(prediction)
    verification_dbz = prediction + 0.3 * torch.cos(
        torch.arange(prediction.numel(), dtype=control.dtype).reshape_as(prediction)
    )
    # Same normalized weighted dBZ MSE as compute_fv_observation_response.
    normalized_weight = metric_weight / metric_weight.max()
    normalized_weight = normalized_weight / normalized_weight.sum()

    def score(c: torch.Tensor, y: torch.Tensor, base: Any) -> torch.Tensor:
        difference = torch.where(
            normalized_weight > 0,
            forecast(c, y, base) - verification_dbz,
            torch.zeros_like(verification_dbz),
        )
        return (normalized_weight * difference.square()).sum()

    response = compute_fv_observation_response(
        control,
        observations,
        frozen,
        verification_dbz=verification_dbz,
        metric_weight=metric_weight,
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future,
        boundary_support=support,
        background_dependency="first_observation",
    )
    direction = torch.sin(
        torch.arange(base_y.numel(), dtype=control.dtype)
    ).reshape_as(base_y)
    gn_directional_slope = (response.sensitivity_dbz * direction).sum()
    base_score = score(control, base_y, frozen)
    base_signs = face_signs(control, frozen)
    fixed_support = all(
        bool(edge.eq(1).all())
        for stages in support
        for edges in stages
        for edge in edges
    )

    perturbations: list[dict[str, Any]] = []
    by_h: dict[float, dict[int, tuple[float, Any, Any]]] = {}
    for h in (1e-3, 5e-4):
        by_h[h] = {}
        for sign in (-1, 1):
            y = base_y + sign * h * direction
            result, endpoint_frozen = _solve(y, observations, frozen, control)
            endpoint_signs = face_signs(result.control, endpoint_frozen)
            same_signs = bool(torch.equal(endpoint_signs, base_signs))
            box_margin = face_branch_margin(control, result.control, frozen)
            endpoint_score = score(result.control, y, endpoint_frozen)
            by_h[h][sign] = (float(endpoint_score), result, endpoint_frozen)
            perturbations.append(
                {
                    "h": h,
                    "sign": sign,
                    "score": float(endpoint_score),
                    "score_change": float(endpoint_score - base_score),
                    "solver": _diagnostics(result),
                    "gradient_over_h": abs(result.selected_gradient_norm) / h,
                    "detected_mask_all": bool(observations.detected_mask.all()),
                    "valid_mask_all": bool(observations.valid_mask.all()),
                    "initial_support_all": bool(frozen.initial_support_mask.all()),
                    "future_support_all": fixed_support,
                    "same_face_signs_as_nominal": same_signs,
                    "face_box_margin": box_margin,
                    "branch_check": "pass" if same_signs and box_margin > 0 else "fail",
                }
            )

    slopes: list[dict[str, Any]] = []
    for h in (1e-3, 5e-4):
        plus_score = by_h[h][1][0]
        minus_score = by_h[h][-1][0]
        central_slope = (plus_score - minus_score) / (2.0 * h)
        slopes.append(
            {
                "h": h,
                "plus_score": plus_score,
                "minus_score": minus_score,
                "central_slope": central_slope,
                "gn_sensitivity_dot_direction": float(gn_directional_slope),
                "central_minus_gn": central_slope - float(gn_directional_slope),
                "absolute_error": abs(central_slope - float(gn_directional_slope)),
                "relative_error_to_gn": abs(
                    central_slope - float(gn_directional_slope)
                ) / max(abs(float(gn_directional_slope)), 1e-30),
                "same_face_signs": all(
                    item["same_face_signs_as_nominal"]
                    and item["face_box_margin"] > 0
                    for item in perturbations
                    if item["h"] == h
                ),
            }
        )

    report = {
        "scope": (
            "shared public matrix-free solve_analysis, endpoints initialized from nominal control; four perturbed "
            "solves; no dense polish; GN approximate response"
        ),
        "fixed_statistics_semantics": (
            "Only y and B=y[0] are changed. Detection/valid masks, support, "
            "geometry, observation uncertainties, verification, metric weights, "
            "and future boundary schedules are fixed; prepare_analysis is not rerun."
        ),
        "branch_scope": (
            "Endpoint signs and sufficient latent-box margins checked; this does "
            "not certify an unknown continuous optimizer path."
        ),
        "nominal_solver": _diagnostics(nominal),
        "nominal_response": {
            "gradient_max": response.gradient_max,
            "normal_products": response.normal_products,
            "adjoint_relative_residual": response.adjoint_relative_residual,
            "face_margin": response.face_margin,
            "score": float(base_score),
            "sensitivity_norm": float(response.sensitivity_dbz.norm()),
            "gn_directional_slope": float(gn_directional_slope),
            "curvature": response.curvature,
        },
        "perturbed_solver_count": 4,
        "perturbations": perturbations,
        "central_slope_comparison": slopes,
        "elapsed_seconds": time.monotonic() - started,
        "total_reanalysis": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
