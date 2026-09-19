"""One visible, observation-only FV rotation demonstration.

The 48 km square is refined to 240x240 cells (200 m spacing), so the compact
asymmetric profile remains identifiable. The truth flow supplies analytic
boundary traces for the PDE test; no truth coefficients are passed to analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from advar.nowcast import NowcastConfig, nowcast  # noqa: E402
from advar.physics import echo_to_dbz  # noqa: E402
from advar.transport import BoundarySchedule  # noqa: E402
from advar.variational import (  # noqa: E402
    AnalysisConfig,
    FVAnalysisResult,
    FVAnalysisTransport,
    forecast_fv_analysis,
    prepare_analysis,
    solve_analysis,
)
from finite_volume_probe import cell_averages, face_averages  # noqa: E402
from fv_original_cases import _metric_fixed, _source_commit  # noqa: E402


GRID = 240
SPACING_M = 200.0
INTERVAL_MINUTES = 10
INTERVAL_SECONDS = INTERVAL_MINUTES * 60.0
LEADS = 18
SUBSTEPS = 64
OMEGA = 5.0e-5
BASE_ECHO = 100.0
PROFILE_ECHO = 900.0
MIN_DBZ = -10.0
MAX_DBZ = 70.0


def _json_grid(value: torch.Tensor) -> list[list[float | None]]:
    return [
        [
            None if not math.isfinite(float(x)) else float(x)
            for x in row
        ]
        for row in value.detach().cpu()
    ]


def _q_field(elapsed_seconds: float) -> torch.Tensor:
    return BASE_ECHO + PROFILE_ECHO * cell_averages(
        GRID, SPACING_M, elapsed_seconds, (0.0, 0.0), OMEGA, 0.0,
        quadrature_order=12,
    )


def _psi_basis() -> torch.Tensor:
    vertices = (torch.arange(GRID + 1, dtype=torch.float64) - GRID / 2) * SPACING_M
    y, x = torch.meshgrid(vertices, vertices, indexing="ij")
    basis = torch.stack((y, -x, -0.5 * (x.square() + y.square())))
    return basis - basis[:, :1, :1]


def _boundary_schedule(start_seconds: float, intervals: int) -> BoundarySchedule:
    dt = INTERVAL_SECONDS / SUBSTEPS
    schedules = []
    for step in range(intervals * SUBSTEPS):
        times = (start_seconds + step * dt, start_seconds + (step + 1) * dt)
        schedules.append(tuple(
            tuple(BASE_ECHO + PROFILE_ECHO * edge for edge in face_averages(
                GRID, SPACING_M, time, (0.0, 0.0), OMEGA, 0.0,
                quadrature_order=12,
            ))
            for time in times
        ))
    return tuple(schedules)


def _support_schedule(intervals: int) -> BoundarySchedule:
    edge = torch.ones(GRID, dtype=torch.float64)
    pair = (edge, edge, edge, edge)
    return tuple((pair, pair) for _ in range(intervals * SUBSTEPS))


def _metrics(
    forecast: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    valid_frames = torch.ones_like(truth, dtype=torch.bool) if valid is None else valid
    rows = [
        _metric_fixed(forecast[i], persistence, truth[i], None, valid_frames[i])
        for i in range(LEADS)
    ]
    for i, row in enumerate(rows):
        row["lead_minutes"] = (i + 1) * INTERVAL_MINUTES
    return rows


def build_payload(*, maximum_outer_iterations: int = 4, step_tolerance: float = 1.0e-5,
                  analysis_checkpoint: Path | None = None,
                  verify_stationarity: bool = False) -> dict[str, Any]:
    config = NowcastConfig(
        interval_minutes=INTERVAL_MINUTES,
        horizon_minutes=LEADS * INTERVAL_MINUTES,
        min_dbz=MIN_DBZ,
        max_dbz=MAX_DBZ,
    )
    print('Generating independent240x240 observations/truth', flush=True)
    observations_q = torch.stack(tuple(_q_field(t) for t in (-1200.0, -600.0, 0.0)))
    truth_q = torch.stack(tuple(_q_field((i + 1) * INTERVAL_SECONDS) for i in range(LEADS)))
    observations = echo_to_dbz(observations_q, min_dbz=MIN_DBZ, max_dbz=MAX_DBZ)
    truth = echo_to_dbz(truth_q, min_dbz=MIN_DBZ, max_dbz=MAX_DBZ)
    persistence = observations[-1]

    analysis_boundary = _boundary_schedule(-1200.0, 2)
    future_boundary = _boundary_schedule(0.0, LEADS)
    analysis_support = _support_schedule(2)
    future_support = _support_schedule(LEADS)
    fv_spec = FVAnalysisTransport(
        psi_basis=_psi_basis(),
        coefficient_limits=torch.tensor([2.0, 2.0, 1.0e-4], dtype=torch.float64),
        substeps_per_interval=SUBSTEPS,
        spacing_yx=(SPACING_M, SPACING_M),
        boundary_echo=analysis_boundary,
        boundary_support=analysis_support,
        reconstruction="donorcell",
        max_courant=0.5,
        replay=True,
    )
    analysis_config = AnalysisConfig(
        field_smoothness_weight=0.0,
        maximum_outer_iterations=maximum_outer_iterations,
        maximum_pcg_iterations=40,
        gradient_tolerance=1.0e-6,
        step_tolerance=step_tolerance,
        pcg_relative_tolerance=1.0e-5,
    )
    observations_for_da, frozen = prepare_analysis(
        observations,
        nowcast_config=config,
        analysis_config=analysis_config,
        observation_std_dbz=0.5,
        fv_transport=fv_spec,
    )
    p0_result = nowcast(observations, config)
    p0 = p0_result.forecast_dbz[:LEADS]
    p0_method = {
        "forecast": [_json_grid(x.masked_fill(~valid, float("nan"))) for x, valid in zip(p0, p0_result.valid_mask[:LEADS], strict=True)],
        "metrics": _metrics(p0, truth, persistence, p0_result.valid_mask[:LEADS]),
        "state": {"status": "completed", "reason": p0_result.metadata.tendency_source.value},
    }

    print('Starting240x240 FV analysis', flush=True)
    fv_result = solve_analysis(observations_for_da, frozen)
    if not isinstance(fv_result, FVAnalysisResult):
        raise RuntimeError("FV solve returned a non-FV result")
    verification = None
    if verify_stationarity:
        from advar.fv_sensitivity import verify_fv_stationarity
        fv_result, verification = verify_fv_stationarity(fv_result, observations_for_da, frozen)
    if analysis_checkpoint is not None:
        # Retain the full local research contract for independent derivative checks.
        analysis_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(result=fv_result, control=fv_result.control,
                        observations=observations_for_da, frozen=frozen), analysis_checkpoint)
    print(f'Analysis ended: {fv_result.reason}; forecasting18leads', flush=True)
    with torch.no_grad():
        forecast_result = forecast_fv_analysis(
            fv_result.control.detach(),
            frozen,
            leads=LEADS,
            boundary_start_interval=2,
            boundary_echo=future_boundary,
            boundary_support=future_support,
        )
    forecast_q = forecast_result.frames_linear[1:]
    forecast = echo_to_dbz(forecast_q, min_dbz=MIN_DBZ, max_dbz=MAX_DBZ)
    fv_method = {
        "forecast": [_json_grid(x) for x in forecast],
        "metrics": _metrics(forecast, truth, persistence),
        "state": {
            "status": "completed" if fv_result.stationarity_verified else "unverified",
            "reason": fv_result.reason,
            "stationarity_verified": bool(fv_result.stationarity_verified),
            "selected_gradient_norm": fv_result.selected_gradient_norm,
            "outer_iterations": fv_result.outer_iterations,
            "pcg_iterations": fv_result.pcg_iterations,
            "psi_coefficients": [float(x) for x in fv_result.trajectory.psi_coefficients],
            "log_growth_per_interval": float(fv_result.trajectory.log_growth_per_step),
        },
        "solver": {
            "stationarity_verified": bool(fv_result.stationarity_verified),
            "selected_gradient_norm": fv_result.selected_gradient_norm,
            "initial_objective": fv_result.initial_objective,
            "final_objective": fv_result.final_objective,
            "reason": fv_result.reason,
        },
    }
    return {
        "meta": {
            "analysis_config": {"maximum_outer_iterations": maximum_outer_iterations, "step_tolerance": step_tolerance, "gradient_tolerance": analysis_config.gradient_tolerance, "maximum_pcg_iterations": analysis_config.maximum_pcg_iterations, "pcg_relative_tolerance": analysis_config.pcg_relative_tolerance},
            "grid_shape": [GRID, GRID],
            "spacing_m": SPACING_M,
            "min_dbz": MIN_DBZ, "max_dbz": MAX_DBZ,
            "generator": "fv_rotation_demo.py", "source_commit": _source_commit(),
            "interval_minutes": INTERVAL_MINUTES,
            "leads": LEADS,
            "substeps_per_interval": SUBSTEPS,
            "background_echo": BASE_ECHO,
            "analytic_flow": {"velocity_yx": [0.0, 0.0], "omega_per_second": OMEGA},
            "scope": "one PDE-consistent rigid-rotation FV demonstration; diagnostic, not weather skill",
            "known_boundary": "analytic face averages at every SSPRK stage; support is one",
            "provenance": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), Path(__file__).with_name("finite_volume_probe.py"), Path(__file__).with_name("fv_original_cases.py"), ROOT / "src/advar/transport.py", ROOT / "src/advar/variational.py", ROOT / "src/advar/fv_sensitivity.py")
            },
            "local_stationarity_verification": verification,
        },
        "cases": [{
            "id": "pde_rigid_rotation_visible",
            "name": "PDE · 31° 강체 회전",
            "description": "비대칭 타원형 에코장이 180분 동안 약 31° 회전합니다.",
            "expectation": "관측 세 시각만으로 FV ψ 회전 성분을 조건부 추정합니다.",
            "model_limit": "분석은 명시된 analytic boundary에 의존하며 운영 FSO/FSOI 적격성을 주장하지 않습니다.",
            "observations": [_json_grid(x) for x in observations],
            "truth": [_json_grid(x) for x in truth],
            "persistence": _json_grid(persistence),
            "methods": {"p0": p0_method, "fv": fv_method},
            "grid_shape": [GRID, GRID],
            "interval_minutes": INTERVAL_MINUTES,
            "min_dbz": MIN_DBZ,
            "max_dbz": MAX_DBZ,
            "assumptions": {"truth_coefficients_in_inference": False, "truth_used_for_initialization": False, "future_boundaries_explicit": True},
        }],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-outer-iterations", type=int, default=4)
    parser.add_argument("--step-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--analysis-checkpoint", type=Path)
    parser.add_argument("--verify-stationarity", action="store_true",
                        help="Require the explicit local first-order checks; not FSO eligibility")
    args = parser.parse_args()
    start = time.perf_counter()
    payload = build_payload(maximum_outer_iterations=args.maximum_outer_iterations,
                            step_tolerance=args.step_tolerance,
                            analysis_checkpoint=args.analysis_checkpoint,
                            verify_stationarity=args.verify_stationarity)
    payload["meta"]["elapsed_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
