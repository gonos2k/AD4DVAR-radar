"""Build two PDE-consistent FV comparisons for the original weather page.

This is a bounded research fixture: two 12x12 CPU/FP64 cases, two future
leads, prescribed analytic inflow, and observation-only FV analysis. The
analytic truth is independent of the production FV trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from advar.nowcast import NowcastConfig, nowcast  # noqa: E402
from advar.physics import echo_to_dbz  # noqa: E402
from advar.transport import BoundarySchedule, finite_volume_trajectory  # noqa: E402
from advar.variational import (  # noqa: E402
    AnalysisConfig,
    FVAnalysisResult,
    FVAnalysisTransport,
    forecast_fv_analysis,
    prepare_analysis,
    solve_analysis,
)
from finite_volume_probe import cell_averages, face_averages  # noqa: E402
from scenarios import _metric  # noqa: E402


GRID = 12
SPACING_M = 1_000.0
INTERVAL_MINUTES = 10
SUBSTEPS = 8
MIN_DBZ = -10.0
MAX_DBZ = 70.0
BASE_ECHO = 100.0
PROFILE_ECHO = 900.0
LEADS = 2

CASE_SPECS = (
    dict(
        id="pde_translation",
        name="PDE · 일정 병진",
        description="비발산 일정 속도의 두드러진 에코장을 FV로 운반합니다.",
        expectation="관측 세 시각만으로 일정 병진 성분을 조건부로 추정합니다.",
        lesson="같은 격자와 같은 truth에서 P0와 FV의 고정 영역 오차를 비교합니다.",
        model_limit="정지한 비발산 유동과 명시한 inflow만 다루며 새 세포를 만들지 않습니다.",
        velocity=(1.0, 0.0),
        omega=0.0,
    ),
    dict(
        id="pde_rigid_rotation",
        name="PDE · 강체 회전",
        description="고정된 streamfunction으로 비발산 강체 회전을 구성합니다.",
        expectation="회전 성분을 포함한 FV 분석·예측의 조건부 수치 응답을 확인합니다.",
        lesson="P0 전역 병진과 FV 회전 표현을 같은 observation/truth에 대조합니다.",
        model_limit="ψ는 두 lead 동안 고정이며 식별성·운영 날씨 skill을 주장하지 않습니다.",
        velocity=(0.0, 0.0),
        omega=5.0e-5,
    ),
)


def _json_grid(value: torch.Tensor, valid: torch.Tensor | None = None) -> list[list[float | None]]:
    return [
        [
            None if (valid is not None and not bool(valid[row_index, col_index])) or not math.isfinite(float(x)) else float(x)
            for col_index, x in enumerate(row)
        ]
        for row_index, row in enumerate(value.detach().cpu())
    ]


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _null_grid() -> list[list[None]]:
    return [[None for _ in range(GRID)] for _ in range(GRID)]


def _q_field(elapsed_seconds: float, velocity: tuple[float, float], omega: float) -> torch.Tensor:
    return BASE_ECHO + PROFILE_ECHO * cell_averages(
        GRID,
        SPACING_M,
        elapsed_seconds,
        velocity,
        omega,
        0.0,
        quadrature_order=12,
    )


def _psi_basis() -> torch.Tensor:
    vertices = (torch.arange(GRID + 1, dtype=torch.float64) - GRID / 2) * SPACING_M
    y, x = torch.meshgrid(vertices, vertices, indexing="ij")
    basis = torch.stack((y, -x, -0.5 * (x.square() + y.square())))
    return basis - basis[:, :1, :1]


def _boundary_schedule(
    start_seconds: float,
    intervals: int,
    velocity: tuple[float, float],
    omega: float,
) -> BoundarySchedule:
    dt = INTERVAL_MINUTES * 60.0 / SUBSTEPS
    schedules = []
    for step in range(intervals * SUBSTEPS):
        edges0 = tuple(
            PROFILE_ECHO * edge + BASE_ECHO * torch.ones_like(edge)
            for edge in face_averages(
                GRID, SPACING_M, start_seconds + step * dt,
                velocity, omega, 0.0, quadrature_order=12,
            )
        )
        edges1 = tuple(
            PROFILE_ECHO * edge + BASE_ECHO * torch.ones_like(edge)
            for edge in face_averages(
                GRID, SPACING_M, start_seconds + (step + 1) * dt,
                velocity, omega, 0.0, quadrature_order=12,
            )
        )
        schedules.append((edges0, edges1))
    return tuple(schedules)


def _metric_fixed(
    forecast: torch.Tensor,
    persistence: torch.Tensor,
    truth: torch.Tensor,
    confidence: torch.Tensor | None,
    valid: torch.Tensor | None = None,
) -> dict[str, Any]:
    candidate_valid = torch.ones_like(truth, dtype=torch.bool) if valid is None else valid.to(dtype=torch.bool)
    fixed = torch.ones_like(truth, dtype=torch.bool)
    metric = _metric(forecast, persistence, truth, candidate_valid, confidence, fixed)
    missing = int((~torch.isfinite(forecast) | ~candidate_valid).sum())
    total = int(fixed.sum())
    persistence_finite = torch.isfinite(persistence) & torch.isfinite(truth)
    persistence_mae = None if not bool(persistence_finite.all()) else float((persistence[persistence_finite] - truth[persistence_finite]).abs().mean())
    metric["missing_pixels"] = missing
    metric["domain_pixels"] = total
    metric["issued_pixels"] = int((torch.isfinite(forecast) & candidate_valid).sum())
    metric["issued_missing_pixels"] = total - metric["issued_pixels"]
    metric["valid_fraction"] = round(float(candidate_valid.float().mean()), 4)
    metric["persistence_mae"] = persistence_mae
    if missing or not bool(persistence_finite.all()):
        metric["mae"] = None
        metric["scored_pixels"] = 0
        metric["scored_fraction"] = 0.0
        metric["unscored_pixels"] = total
        for label, detection in metric["detection"].items():
            truth_positive = int((torch.isfinite(truth) & (truth >= float(label))).sum())
            detection.update({"hits": None, "misses": None, "false_alarms": None, "csi": None, "pod": None, "scored_truth_echo_pixels": 0, "excluded_truth_echo_pixels": truth_positive})
    return metric


def _failed_metrics(persistence: torch.Tensor, truth: torch.Tensor) -> list[dict[str, Any]]:
    metrics = [_metric_fixed(torch.full_like(truth[i], float("nan")), persistence, truth[i], None) for i in range(LEADS)]
    for i, metric in enumerate(metrics):
        assert metric is not None
        metric["lead_minutes"] = (i + 1) * INTERVAL_MINUTES
    return metrics


def _p0_method(observations: torch.Tensor, truth: torch.Tensor, persistence: torch.Tensor, config: NowcastConfig) -> dict[str, Any]:
    try:
        result = nowcast(observations, config)
        forecast = result.forecast_dbz[:LEADS]
        metrics = [
            _metric_fixed(
                forecast[i], persistence, truth[i], result.forecast_confidence[i],
                result.valid_mask[i],
            )
            for i in range(LEADS)
        ]
        state = {
            "status": "completed",
            "reason": getattr(result.metadata, "tendency_source", "unknown").value,
            "displacement_yx": [float(x) for x in result.state.displacement_yx],
            "log_growth_per_step": float(result.state.log_growth_per_step),
        }
        for i, metric in enumerate(metrics):
            assert metric is not None
            metric["lead_minutes"] = (i + 1) * INTERVAL_MINUTES
        return {"forecast": [_json_grid(forecast[i], result.valid_mask[i]) for i in range(LEADS)], "metrics": metrics, "state": state}
    except Exception as error:
        return {"forecast": [_null_grid() for _ in range(LEADS)], "metrics": _failed_metrics(persistence, truth), "state": {"status": "failed", "reason": f"{type(error).__name__}: {error}"}}


def _fv_method(
    observations: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    frozen: Any,
    future_echo: BoundarySchedule,
    future_support: BoundarySchedule,
) -> dict[str, Any]:
    try:
        result = solve_analysis(observations, frozen)
        if not isinstance(result, FVAnalysisResult):
            raise RuntimeError("FV solve returned a non-FV result")
        forecast_result = forecast_fv_analysis(
            result.control,
            frozen,
            leads=LEADS,
            boundary_start_interval=2,
            boundary_echo=future_echo,
            boundary_support=future_support,
        )
        forecast = echo_to_dbz(forecast_result.frames_linear[1:], min_dbz=MIN_DBZ, max_dbz=MAX_DBZ)
        metrics = [_metric_fixed(forecast[i], persistence, truth[i], None) for i in range(LEADS)]
        for i, metric in enumerate(metrics):
            assert metric is not None
            metric["lead_minutes"] = (i + 1) * INTERVAL_MINUTES
        state = {
            "status": "completed" if result.stationarity_verified else "unverified",
            "reason": result.reason,
            "stationarity_verified": bool(result.stationarity_verified),
            "selected_gradient_norm": result.selected_gradient_norm,
            "outer_iterations": result.outer_iterations,
            "pcg_iterations": result.pcg_iterations,
            "escape_status": result.escape_status,
            "improved": bool(result.improved),
            "psi_coefficients": [float(x) for x in result.trajectory.psi_coefficients],
            "log_growth_per_interval": float(result.trajectory.log_growth_per_step),
        }
        solver = {
            "reason": result.reason,
            "stationarity_verified": bool(result.stationarity_verified),
            "initial_objective": result.initial_objective,
            "final_objective": result.final_objective,
            "outer_iterations": result.outer_iterations,
            "pcg_iterations": result.pcg_iterations,
            "selected_gradient_norm": result.selected_gradient_norm,
            "escape_status": result.escape_status,
            "decoded_psi_coefficients": [float(x) for x in result.trajectory.psi_coefficients],
            "decoded_log_growth_per_interval": float(result.trajectory.log_growth_per_step),
        }
        return {"forecast": [_json_grid(x) for x in forecast], "metrics": metrics, "state": state, "solver": solver}
    except Exception as error:
        return {"forecast": [_null_grid() for _ in range(LEADS)], "metrics": _failed_metrics(persistence, truth), "state": {"status": "failed", "reason": f"{type(error).__name__}: {error}"}, "solver": None}


def _case_payload(
    spec: dict[str, Any],
    config: NowcastConfig,
    maximum_outer_iterations: int,
) -> dict[str, Any]:
    velocity, omega = spec["velocity"], spec["omega"]
    observations_q = torch.stack(tuple(_q_field(seconds, velocity, omega) for seconds in (-1200.0, -600.0, 0.0)))
    truth_q = torch.stack(tuple(_q_field(seconds, velocity, omega) for seconds in (600.0, 1200.0)))
    observations = echo_to_dbz(observations_q, min_dbz=MIN_DBZ, max_dbz=MAX_DBZ)
    truth = echo_to_dbz(truth_q, min_dbz=MIN_DBZ, max_dbz=MAX_DBZ)
    persistence = observations[-1]
    basis = _psi_basis()
    limits = torch.tensor([2.0, 2.0, 1.0e-4], dtype=torch.float64)
    analysis_boundary = _boundary_schedule(-1200.0, 2, velocity, omega)
    future_boundary = _boundary_schedule(0.0, LEADS, velocity, omega)
    support_edges = tuple(torch.ones(n, dtype=torch.float64) for n in (GRID, GRID, GRID, GRID))
    support_boundary = tuple(
        (support_edges, support_edges) for _ in range(LEADS * SUBSTEPS)
    )
    fv_spec = FVAnalysisTransport(
        psi_basis=basis,
        coefficient_limits=limits,
        substeps_per_interval=SUBSTEPS,
        spacing_yx=(SPACING_M, SPACING_M),
        boundary_echo=analysis_boundary,
        boundary_support=support_boundary,
        reconstruction="donorcell",
        max_courant=0.5,
        replay=True,
    )
    analysis_config = AnalysisConfig(
        field_smoothness_weight=0.0,
        maximum_outer_iterations=maximum_outer_iterations,
        maximum_pcg_iterations=40,
        gradient_tolerance=1.0e-6,
        step_tolerance=1.0e-5,
        pcg_relative_tolerance=1.0e-5,
    )
    observations_for_da, frozen = prepare_analysis(
        observations,
        nowcast_config=config,
        analysis_config=analysis_config,
        observation_std_dbz=0.5,
        fv_transport=fv_spec,
    )
    p0 = _p0_method(observations, truth, persistence, config)
    fv = _fv_method(observations_for_da, truth, persistence, frozen, future_boundary, support_boundary)
    return {
        **{key: spec[key] for key in ("id", "name", "description", "expectation", "lesson", "model_limit")},
        "grid_shape": [GRID, GRID],
        "interval_minutes": INTERVAL_MINUTES,
        "analytic_flow": {"velocity_yx": list(velocity), "omega": omega},
        "min_dbz": MIN_DBZ,
        "max_dbz": MAX_DBZ,
        "observations": [_json_grid(x) for x in observations],
        "truth": [_json_grid(x) for x in truth],
        "persistence": _json_grid(persistence),
        "methods": {"p0": p0, "fv": fv},
        "forecast": p0["forecast"],
        "metrics": p0["metrics"],
        "state": p0["state"],
        "known_boundary": {"echo": "analytic cell-face averages at both SSPRK stages", "support": "ones on every face/stage", "relative_start_intervals": {"analysis": 0, "forecast": 2}},
        "assumptions": {"dtype": "float64", "device": "cpu", "quadrature_order": 12, "truth_coefficients_in_inference": False, "truth_used_for_initialization": False, "known_boundary_generated_from": "same analytic truth q at each SSPRK stage time", "known_truth_flow_used_only_for_generation": True, "fixed_domain_pixels": GRID * GRID},
    }


def build_payload(maximum_outer_iterations: int = 4) -> dict[str, Any]:
    if maximum_outer_iterations <= 0:
        raise ValueError("maximum_outer_iterations must be positive")
    config = NowcastConfig(interval_minutes=INTERVAL_MINUTES, horizon_minutes=LEADS * INTERVAL_MINUTES, min_dbz=MIN_DBZ, max_dbz=MAX_DBZ)
    cases = [_case_payload(spec, config, maximum_outer_iterations) for spec in CASE_SPECS]
    source_paths = [Path(__file__), Path(__file__).with_name("finite_volume_probe.py"), ROOT / "src/advar/transport.py", ROOT / "src/advar/variational.py"]
    provenance = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    return {
        "meta": {"grid_shape": [GRID, GRID], "interval_minutes": INTERVAL_MINUTES, "min_dbz": MIN_DBZ, "max_dbz": MAX_DBZ, "background_echo": BASE_ECHO, "spacing_m": SPACING_M, "leads": LEADS, "substeps_per_interval": SUBSTEPS, "analysis_config": {"maximum_outer_iterations": maximum_outer_iterations, "maximum_pcg_iterations": 40, "gradient_tolerance": 1.0e-6, "step_tolerance": 1.0e-5, "pcg_relative_tolerance": 1.0e-5}, "model": "shared FV PDE q_t+div(uq)=gamma q; research comparison", "oracle": "independent characteristic cell/face quadrature", "fixed_domain": "all truth cells; candidate missing pixels make mae null", "scope": "two new PDE-consistent cases only; original seven cases remain unchanged", "generator": "fv_original_cases.py", "source_commit": _source_commit(), "provenance": provenance},
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "graphify-out/fv-root-cause-20260913/fv_original_cases.json")
    parser.add_argument("--maximum-outer-iterations", type=int, default=4)
    args = parser.parse_args()
    payload = build_payload(args.maximum_outer_iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
