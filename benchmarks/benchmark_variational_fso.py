"""Measure delayed P1 FSO wall time and process peak RSS on real artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from advar import (
    SensitivityConfig,
    VariationalAdjointConfig,
    compute_variational_fso,
    load_forecast_run,
    load_p1_linearization,
)


def _positive_minutes(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("lead minutes must be positive")
    return result


def _metric_names(value: str) -> tuple[str, ...]:
    result = tuple(item for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("at least one metric is required")
    return result


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast_run", type=Path)
    parser.add_argument("linearization", type=Path)
    parser.add_argument("verification_npy", type=Path)
    parser.add_argument(
        "--lead-minutes",
        type=_positive_minutes,
        default=(30, 60, 120, 180),
    )
    parser.add_argument(
        "--metrics",
        type=_metric_names,
        default=("log_echo_mse",),
    )
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument(
        "--metric-domain",
        choices=(
            "issued",
            "radar_dynamics_anchored",
            "confidence_weighted",
        ),
        default="issued",
    )
    parser.add_argument("--max-normal-products", type=int, default=10_000)
    parser.add_argument("--max-output-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--gauss-newton-probes", type=int, default=4)
    parser.add_argument("--cold-start", action="store_true")
    args = parser.parse_args()

    forecast = load_forecast_run(args.forecast_run)
    analysis = load_p1_linearization(args.linearization)
    verification_array = np.load(args.verification_npy, allow_pickle=False)
    verification = torch.as_tensor(
        verification_array,
        dtype=forecast.state.echo_linear.dtype,
        device=forecast.state.echo_linear.device,
    )
    sensitivity_config = SensitivityConfig(
        metric_names=args.metrics,
        metric_domain=args.metric_domain,
        full_map_lead_minutes=args.lead_minutes,
        tile_size=args.tile_size,
    )
    adjoint_config = VariationalAdjointConfig(
        lead_minutes=args.lead_minutes,
        maximum_normal_products=args.max_normal_products,
        maximum_materialized_output_bytes=args.max_output_bytes,
        warm_start_by_metric=not args.cold_start,
        gauss_newton_probe_count=args.gauss_newton_probes,
    )

    rss_before = _peak_rss_bytes()
    started = time.perf_counter()
    result = compute_variational_fso(
        forecast,
        analysis,
        verification,
        sensitivity_config=sensitivity_config,
        adjoint_config=adjoint_config,
    )
    elapsed = time.perf_counter() - started
    report = {
        "contract": "p1-variational-fso-benchmark-v1",
        "forecast_run_digest": forecast.forecast_run_digest,
        "linearization_digest": result.linearization_digest,
        "variational_fso_digest": result.variational_fso_digest,
        "grid_shape": list(forecast.state.echo_linear.shape),
        "dtype": str(forecast.state.echo_linear.dtype),
        "device": str(forecast.state.echo_linear.device),
        "lead_minutes": list(result.lead_minutes),
        "metric_names": list(result.metric_names),
        "metric_domain": result.metric_domain,
        "metric_domain_weight_fraction": (
            result.metric_domain_weight_fraction.cpu().tolist()
        ),
        "wall_seconds": elapsed,
        "peak_rss_before_bytes": rss_before,
        "peak_rss_after_bytes": _peak_rss_bytes(),
        "materialized_output_bytes": result.materialized_output_bytes,
        "normal_products": result.total_normal_products,
        "adjoint_iterations": result.adjoint_iterations.cpu().tolist(),
        "adjoint_relative_residual": (
            result.adjoint_relative_residual.cpu().tolist()
        ),
        "adjoint_detected_sensitivity_l2_error_bound": (
            result.adjoint_detected_sensitivity_l2_error_bound.cpu().tolist()
        ),
        "adjoint_warm_started": result.adjoint_warm_started.cpu().tolist(),
        "low_local_validity": result.active_set_margins.low_local_validity,
        "gauss_newton_maximum_relative_curvature_defect": (
            result.gauss_newton_diagnostics
            .maximum_relative_curvature_defect
        ),
        "gauss_newton_reliable": result.gauss_newton_diagnostics.reliable,
    }
    print(
        json.dumps(
            _json_safe(report),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
