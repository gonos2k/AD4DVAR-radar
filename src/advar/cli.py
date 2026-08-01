"""Command-line entry point for NumPy radar volumes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from .nowcast import (
    ForecastResult,
    NowcastConfig,
    RadarGridTimeContract,
    nowcast,
)
from .run_artifact import forecast_run_arrays
from .variational import AnalysisConfig, AnalysisResult, variational_nowcast


OUTPUT_CONTRACT_VERSION = "nowcast-npz-v10"


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
        "--amplitude-information-policy",
        choices=("research_degraded", "operational_fallback"),
        default="research_degraded",
        help=(
            "P1 behavior when precursor amplitude information is insufficient"
        ),
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
        "--maximum-background-age-minutes",
        type=float,
        default=60.0,
    )
    parser.add_argument("--valid-times", nargs=3)
    parser.add_argument("--background-valid-times", nargs=3)
    parser.add_argument("--dx-m", type=float)
    parser.add_argument("--dy-m", type=float)
    parser.add_argument("--projection")
    parser.add_argument("--grid-hash")
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
    grid_time_contract = _grid_time_contract_from_args(
        parser,
        args,
        background_present=background is not None,
    )

    config = NowcastConfig(
        min_dbz=args.min_dbz,
        max_dbz=args.max_dbz,
        echo_threshold_dbz=args.echo_threshold_dbz,
        min_publish_support=args.min_publish_support,
        maximum_background_age_minutes=(
            args.maximum_background_age_minutes
        ),
    )
    frames_tensor = torch.as_tensor(frames, dtype=torch.float32)
    if args.variational:
        result, analysis = variational_nowcast(
            frames_tensor,
            nowcast_config=config,
            analysis_config=AnalysisConfig(
                detection_limit_dbz=args.echo_threshold_dbz,
                amplitude_information_policy=args.amplitude_information_policy,
            ),
            observation_std_dbz=args.observation_std_dbz,
            qc_mask=qc_mask,
            background_frames_dbz=background,
            background_age_minutes=args.background_age_minutes,
            grid_time_contract=grid_time_contract,
            audit=args.audit,
        )
    else:
        result = nowcast(
            frames_tensor,
            config,
            qc_mask=qc_mask,
            background_frames_dbz=background,
            background_age_minutes=args.background_age_minutes,
            grid_time_contract=grid_time_contract,
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


def _grid_time_contract_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    background_present: bool,
) -> RadarGridTimeContract | None:
    required = (
        args.valid_times,
        args.dx_m,
        args.dy_m,
        args.projection,
        args.grid_hash,
    )
    if not any(value is not None for value in required):
        if args.background_valid_times is not None:
            parser.error("--background-valid-times requires grid/time metadata")
        return None
    if any(value is None for value in required):
        parser.error(
            "--valid-times, --dx-m, --dy-m, --projection, and --grid-hash "
            "must be provided together"
        )
    if background_present != (args.background_valid_times is not None):
        parser.error(
            "--background-valid-times must match --background availability"
        )
    try:
        return RadarGridTimeContract(
            valid_times=tuple(args.valid_times),
            dx_m=args.dx_m,
            dy_m=args.dy_m,
            projection=args.projection,
            grid_hash=args.grid_hash,
            background_valid_times=(
                None
                if args.background_valid_times is None
                else tuple(args.background_valid_times)
            ),
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))


def _output_arrays(
    result: ForecastResult,
    analysis: AnalysisResult | None,
    config: NowcastConfig,
) -> dict[str, Any]:
    output = forecast_run_arrays(result)
    output.update(
        {
            "output_contract_version": np.asarray(OUTPUT_CONTRACT_VERSION),
            "min_publish_support": np.asarray(config.min_publish_support),
            "lead_minutes": np.arange(
                config.interval_minutes,
                config.horizon_minutes + 1,
                config.interval_minutes,
            ),
        }
    )
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
            analysis_minimum_reachability_margin=np.asarray(
                np.nan
                if analysis.minimum_reachability_margin is None
                else analysis.minimum_reachability_margin
            ),
            analysis_unresolved_amplitude_fraction=np.asarray(
                np.nan
                if analysis.unresolved_amplitude_fraction is None
                else analysis.unresolved_amplitude_fraction
            ),
            analysis_unresolved_amplitude_fraction_by_time=(
                _optional_pair_array(
                    analysis.unresolved_amplitude_fraction_by_time
                )
            ),
            analysis_unresolved_pixel_fraction_by_time=_optional_pair_array(
                analysis.unresolved_pixel_fraction_by_time
            ),
            analysis_amplitude_violation_score=np.asarray(
                np.nan
                if analysis.amplitude_violation_score is None
                else analysis.amplitude_violation_score
            ),
            analysis_amplitude_violation_score_by_time=_optional_pair_array(
                analysis.amplitude_violation_score_by_time
            ),
            analysis_integrated_echo_ratio_by_time=_optional_pair_array(
                analysis.integrated_echo_ratio_by_time
            ),
            analysis_displacement_tolerant_soft_echo_area_ratio_by_time=(
                _optional_pair_array(
                    analysis
                    .displacement_tolerant_soft_echo_area_ratio_by_time
                )
            ),
            analysis_amplitude_diagnostics_source=np.asarray(
                analysis.amplitude_diagnostics_source
            ),
            analysis_effective_precursor_pixel_count_by_time=(
                _optional_pair_array(
                    analysis.effective_precursor_pixel_count_by_time
                )
            ),
            analysis_bad_quality_weight_by_time=_optional_pair_array(
                analysis.bad_quality_weight_by_time
            ),
            analysis_total_quality_weight_by_time=_optional_pair_array(
                analysis.total_quality_weight_by_time
            ),
            analysis_amplitude_information_sufficient_by_time=(
                _optional_bool_pair_array(
                    analysis.amplitude_information_sufficient_by_time
                )
            ),
            analysis_insufficient_amplitude_information=np.asarray(
                analysis.insufficient_amplitude_information
            ),
            analysis_established_echo_excess_growth_fraction=np.asarray(
                np.nan
                if analysis.established_echo_excess_growth_fraction is None
                else analysis.established_echo_excess_growth_fraction
            ),
            analysis_established_echo_excess_growth_fraction_by_time=(
                _optional_pair_array(
                    analysis
                    .established_echo_excess_growth_fraction_by_time
                )
            ),
            analysis_maximum_growth_envelope_ratio=np.asarray(
                np.nan
                if analysis.maximum_growth_envelope_ratio is None
                else analysis.maximum_growth_envelope_ratio
            ),
            analysis_maximum_growth_envelope_ratio_by_time=(
                _optional_pair_array(
                    analysis.maximum_growth_envelope_ratio_by_time
                )
            ),
            analysis_relative_objective_reduction=np.asarray(
                np.nan
                if analysis.relative_objective_reduction is None
                else analysis.relative_objective_reduction
            ),
            analysis_causal_control_cell_count=np.asarray(
                analysis.causal_control_cell_count
            ),
            analysis_causal_seed_cell_count=np.asarray(
                analysis.causal_seed_cell_count
            ),
            analysis_causal_seed_prior_cost=np.asarray(
                analysis.causal_seed_prior_cost
            ),
            analysis_dynamics_reduced_hessian_eigenvalues=(
                _optional_triple_array(
                    analysis.dynamics_reduced_hessian_eigenvalues
                )
            ),
            analysis_dynamics_reduced_hessian_condition_number=np.asarray(
                np.nan
                if analysis.dynamics_reduced_hessian_condition_number is None
                else analysis.dynamics_reduced_hessian_condition_number
            ),
            analysis_field_growth_jacobian_cosine=np.asarray(
                np.nan
                if analysis.field_growth_jacobian_cosine is None
                else analysis.field_growth_jacobian_cosine
            ),
            analysis_field_motion_jacobian_cosine_yx=(
                _optional_nullable_pair_array(
                    analysis.field_motion_jacobian_cosine_yx
                )
            ),
        )
    return output


def _optional_pair_array(
    value: tuple[float, float] | None,
) -> NDArray[Any]:
    return np.asarray((np.nan, np.nan) if value is None else value)


def _optional_bool_pair_array(
    value: tuple[bool, bool] | None,
) -> NDArray[Any]:
    return np.asarray((False, False) if value is None else value)


def _optional_nullable_pair_array(
    value: tuple[float | None, float | None] | None,
) -> NDArray[Any]:
    if value is None:
        return np.asarray((np.nan, np.nan))
    return np.asarray(
        tuple(np.nan if item is None else item for item in value)
    )


def _optional_triple_array(
    value: tuple[float, float, float] | None,
) -> NDArray[Any]:
    return np.asarray((np.nan, np.nan, np.nan) if value is None else value)


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
