"""Command-line entry point for NumPy radar volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .nowcast import NowcastConfig, nowcast
from .variational import AnalysisConfig, variational_nowcast


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast 18 radar frames from 3 ten-minute dBZ frames."
    )
    parser.add_argument("input", type=Path, help=".npy file shaped [3, H, W]")
    parser.add_argument("output", type=Path, help="output .npz file")
    parser.add_argument("--min-dbz", type=float, default=-10.0)
    parser.add_argument("--max-dbz", type=float, default=70.0)
    parser.add_argument("--echo-threshold-dbz", type=float, default=5.0)
    parser.add_argument(
        "--variational",
        action="store_true",
        help="jointly analyze the three frames before forecasting",
    )
    parser.add_argument(
        "--observation-std-dbz",
        type=float,
        default=2.0,
        help="observation error used by --variational",
    )
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("output path must end with .npz")

    frames = np.load(args.input)
    if frames.ndim != 3 or frames.shape[0] != 3:
        raise ValueError("input must have shape [3, height, width]")

    config = NowcastConfig(
        min_dbz=args.min_dbz,
        max_dbz=args.max_dbz,
        echo_threshold_dbz=args.echo_threshold_dbz,
    )
    frames_tensor = torch.as_tensor(frames, dtype=torch.float32)
    if args.variational:
        forecast, analysis = variational_nowcast(
            frames_tensor,
            nowcast_config=config,
            analysis_config=AnalysisConfig(
                detection_limit_dbz=args.echo_threshold_dbz,
            ),
            observation_std_dbz=args.observation_std_dbz,
        )
        state = analysis.state
    else:
        forecast, state = nowcast(frames_tensor, config)
        analysis = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "forecast_dbz": forecast.detach().cpu().numpy(),
        "displacement_yx": state.displacement_yx.detach().cpu().numpy(),
        "log_growth_per_step": float(state.log_growth_per_step),
        "motion_disagreement_px": float(state.motion_disagreement_px),
        "growth_disagreement": float(state.growth_disagreement),
        "lead_minutes": np.arange(
            config.interval_minutes,
            config.horizon_minutes + 1,
            config.interval_minutes,
        ),
    }
    if analysis is not None:
        output.update(
            analysis_used=np.asarray(True),
            analysis_used_fallback=np.asarray(analysis.used_fallback),
            analysis_reason=np.asarray(analysis.reason),
            analysis_initial_objective=np.asarray(
                analysis.initial_objective
            ),
            analysis_final_objective=np.asarray(analysis.final_objective),
            analysis_outer_iterations=np.asarray(analysis.outer_iterations),
            analysis_pcg_iterations=np.asarray(analysis.pcg_iterations),
        )
    np.savez_compressed(
        args.output,
        **output,
    )

    dy, dx = state.displacement_yx.tolist()
    print(
        f"saved {config.forecast_steps} frames to {args.output} "
        f"(motion dy={dy:.2f}, dx={dx:.2f} px/step)"
    )


if __name__ == "__main__":
    main()
