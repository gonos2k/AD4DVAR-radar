"""Command-line entry point for NumPy radar volumes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any

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
    parser.add_argument(
        "--qc-mask",
        type=Path,
        help="optional boolean .npy mask shaped [3, H, W]",
    )
    parser.add_argument(
        "--background",
        type=Path,
        help="optional time-aligned previous-cycle .npy background",
    )
    parser.add_argument(
        "--background-age-minutes",
        type=float,
        help="age of --background for output diagnostics",
    )
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("output path must end with .npz")

    frames = np.load(args.input, allow_pickle=False)
    if frames.ndim != 3 or frames.shape[0] != 3:
        raise ValueError("input must have shape [3, height, width]")
    qc_mask = None
    if args.qc_mask is not None:
        qc_array = np.load(args.qc_mask, allow_pickle=False)
        if qc_array.shape != frames.shape or qc_array.dtype != np.bool_:
            raise ValueError("QC mask must be boolean with the input shape")
        qc_mask = torch.as_tensor(qc_array, dtype=torch.bool)
    background = None
    if args.background is not None:
        background_array = np.load(args.background, allow_pickle=False)
        if background_array.shape != frames.shape:
            raise ValueError("background must have the input shape")
        background = torch.as_tensor(background_array, dtype=torch.float32)
    if args.background_age_minutes is not None and background is None:
        parser.error("--background-age-minutes requires --background")

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
            qc_mask=qc_mask,
            background_frames_dbz=background,
            background_age_minutes=args.background_age_minutes,
        )
        state = analysis.state
    else:
        forecast, state = nowcast(
            frames_tensor,
            config,
            qc_mask=qc_mask,
            background_frames_dbz=background,
            background_age_minutes=args.background_age_minutes,
        )
        analysis = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "forecast_dbz": forecast.detach().cpu().numpy(),
        "displacement_yx": state.displacement_yx.detach().cpu().numpy(),
        "log_growth_per_step": float(state.log_growth_per_step),
        "motion_disagreement_px": float(state.motion_disagreement_px),
        "growth_disagreement": float(state.growth_disagreement),
        "forecast_status": np.asarray(state.forecast_status.value),
        "data_coverage_fraction": np.asarray(
            state.data_coverage_fraction
        ),
        "latest_data_coverage_fraction": np.asarray(
            state.latest_data_coverage_fraction
        ),
        "background_used": np.asarray(state.background_used),
        "background_age_minutes": np.asarray(
            np.nan
            if state.background_age_minutes is None
            else state.background_age_minutes
        ),
        "lead_minutes": np.arange(
            config.interval_minutes,
            config.horizon_minutes + 1,
            config.interval_minutes,
        ),
    }
    if analysis is None:
        output.update(
            analysis_used=np.asarray(False),
            analysis_converged=np.asarray(False),
            analysis_degraded=np.asarray(False),
            analysis_used_fallback=np.asarray(False),
            analysis_reason=np.asarray("not_requested"),
        )
    else:
        output.update(
            analysis_used=np.asarray(True),
            analysis_converged=np.asarray(analysis.converged),
            analysis_degraded=np.asarray(analysis.degraded),
            analysis_used_fallback=np.asarray(analysis.used_fallback),
            analysis_reason=np.asarray(analysis.reason),
            analysis_initial_objective=np.asarray(
                analysis.initial_objective
            ),
            analysis_final_objective=np.asarray(analysis.final_objective),
            analysis_outer_iterations=np.asarray(analysis.outer_iterations),
            analysis_pcg_iterations=np.asarray(analysis.pcg_iterations),
        )
    _atomic_savez_compressed(args.output, output)

    dy, dx = state.displacement_yx.tolist()
    print(
        f"saved {config.forecast_steps} frames to {args.output} "
        f"(motion dy={dy:.2f}, dx={dx:.2f} px/step)"
    )


def _atomic_savez_compressed(
    path: Path,
    arrays: dict[str, Any],
) -> None:
    """Publish one complete NPZ or leave the previous output untouched."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
