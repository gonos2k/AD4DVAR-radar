"""Command-line entry point for NumPy radar volumes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from .nowcast import (
    ForecastResult,
    NowcastConfig,
    nowcast,
)
from .variational import AnalysisConfig, AnalysisResult, variational_nowcast


OUTPUT_CONTRACT_VERSION = "nowcast-npz-v5"


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
        "--min-publish-support",
        type=float,
        default=0.95,
        help="minimum propagated source support required for publication",
    )
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
        help="required age of --background in minutes",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="include optional positivity and transport audits",
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
    if background is not None and args.background_age_minutes is None:
        parser.error("--background requires --background-age-minutes")

    config = NowcastConfig(
        min_dbz=args.min_dbz,
        max_dbz=args.max_dbz,
        echo_threshold_dbz=args.echo_threshold_dbz,
        min_publish_support=args.min_publish_support,
    )
    frames_tensor = torch.as_tensor(frames, dtype=torch.float32)
    if args.variational:
        result, analysis = variational_nowcast(
            frames_tensor,
            nowcast_config=config,
            analysis_config=AnalysisConfig(
                detection_limit_dbz=args.echo_threshold_dbz,
            ),
            observation_std_dbz=args.observation_std_dbz,
            qc_mask=qc_mask,
            background_frames_dbz=background,
            background_age_minutes=args.background_age_minutes,
            audit=args.audit,
        )
    else:
        result = nowcast(
            frames_tensor,
            config,
            qc_mask=qc_mask,
            background_frames_dbz=background,
            background_age_minutes=args.background_age_minutes,
            audit=args.audit,
        )
        analysis = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = _output_arrays(result, analysis, config)
    if args.audit:
        output.update(_audit_arrays(result, analysis))
    _atomic_savez_compressed(args.output, output)

    dy, dx = result.state.displacement_yx.tolist()
    print(
        f"saved {config.forecast_steps} frames to {args.output} "
        f"(motion dy={dy:.2f}, dx={dx:.2f} px/step)"
    )


def _output_arrays(
    result: ForecastResult,
    analysis: AnalysisResult | None,
    config: NowcastConfig,
) -> dict[str, Any]:
    metadata = result.metadata
    output: dict[str, Any] = {
        "output_contract_version": np.asarray(OUTPUT_CONTRACT_VERSION),
        "forecast_dbz": result.forecast_dbz.detach().cpu().numpy(),
        "displacement_yx": (
            result.state.displacement_yx.detach().cpu().numpy()
        ),
        "log_growth_per_step": float(result.state.log_growth_per_step),
        "motion_disagreement_px": float(metadata.motion_disagreement_px),
        "growth_disagreement": float(metadata.growth_disagreement),
        "tendency_pair_count": np.asarray(metadata.tendency_pair_count),
        "tendency_source": np.asarray(metadata.tendency_source.value),
        "min_publish_support": np.asarray(config.min_publish_support),
        "data_status": np.asarray(metadata.data_status.value),
        "coverage_by_frame": metadata.coverage_by_frame.cpu().numpy(),
        "background_used": np.asarray(metadata.background_used),
        "background_contribution_fraction": np.asarray(
            metadata.background_contribution_fraction
        ),
        "background_age_minutes": np.asarray(
            np.nan
            if metadata.background_age_minutes is None
            else metadata.background_age_minutes
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
    return output


def _audit_arrays(
    result: ForecastResult,
    analysis: AnalysisResult | None,
) -> dict[str, Any]:
    if result.audit is None:
        raise RuntimeError("audit output was requested but no audit was produced")
    final = result.audit.forecast_final
    transport = result.audit.transport
    output: dict[str, Any] = {
        "forecast_minimum_before_fix": np.asarray(final.minimum_before_fix),
        "forecast_corrected_count": np.asarray(final.corrected_count),
        "forecast_corrected_integral": np.asarray(final.corrected_integral),
        "echo_integral_before_transport": np.asarray(
            [item.echo_integral_before for item in transport]
        ),
        "echo_integral_after_transport": np.asarray(
            [item.echo_integral_after for item in transport]
        ),
        "boundary_outflow_integral": np.asarray(
            [item.boundary_outflow_integral for item in transport]
        ),
        "echo_budget_error": np.asarray(
            [item.echo_budget_error for item in transport]
        ),
    }
    if analysis is not None:
        output.update(
            analysis_minimum_before_fix=np.asarray(
                analysis.audit.minimum_before_fix
            ),
            analysis_corrected_count=np.asarray(
                analysis.audit.corrected_count
            ),
            analysis_corrected_integral=np.asarray(
                analysis.audit.corrected_integral
            ),
        )
    return output


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
