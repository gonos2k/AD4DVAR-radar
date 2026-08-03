"""Command-line entry point for NumPy radar volumes."""

from __future__ import annotations

import argparse
from pathlib import Path
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
from .run_artifact import (
    atomic_savez_compressed,
    forecast_run_arrays,
    seal_forecast_run_arrays,
)
from .variational import AnalysisConfig, AnalysisResult, variational_nowcast


OUTPUT_CONTRACT_VERSION = "nowcast-npz-v30"


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
        "--minimum-publish-verified-support",
        type=float,
        help="minimum pair-verified support required for publication",
    )
    parser.add_argument(
        "--variational",
        action="store_true",
        help="jointly analyze the three frames before forecasting",
    )
    parser.add_argument(
        "--mode",
        choices=("research", "operational"),
        default="research",
        help="research diagnostics or a calibrated fail-closed P1 profile",
    )
    parser.add_argument(
        "--operational-calibration-id",
        help="immutable identifier for the hindcast calibration profile",
    )
    parser.add_argument(
        "--observation-std-dbz",
        type=float,
        help="observation error used by --variational",
    )
    parser.add_argument(
        "--motion-increment-scale-mps",
        type=float,
        help="projected x/y velocity increment scale used by --variational",
    )
    parser.add_argument(
        "--amplitude-information-policy",
        choices=("research_degraded", "operational_fallback"),
        help=(
            "P1 behavior when precursor amplitude information is insufficient"
        ),
    )
    parser.add_argument(
        "--amplitude-confidence-policy",
        choices=("research_degraded", "operational_fallback"),
        help="P1 behavior when amplitude amount, area, or growth is implausible",
    )
    parser.add_argument("--maximum-detected-error-std", type=float)
    parser.add_argument("--maximum-unresolved-amplitude-fraction", type=float)
    parser.add_argument("--minimum-amplitude-total-quality-weight", type=float)
    parser.add_argument("--minimum-amplitude-effective-pixel-count", type=float)
    parser.add_argument(
        "--minimum-integrated-echo-ratio-for-confidence",
        type=float,
    )
    parser.add_argument(
        "--maximum-integrated-echo-ratio-for-confidence",
        type=float,
    )
    parser.add_argument(
        "--minimum-soft-echo-area-ratio-for-confidence",
        type=float,
    )
    parser.add_argument(
        "--maximum-soft-echo-area-ratio-for-confidence",
        type=float,
    )
    parser.add_argument(
        "--maximum-established-excess-growth-fraction-for-confidence",
        type=float,
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
    parser.add_argument("--maximum-motion-speed-mps", type=float)
    parser.add_argument("--minimum-phase-correlation-psr", type=float)
    parser.add_argument("--pair-echo-dilation-m", type=float)
    parser.add_argument("--phase-correlation-sidelobe-radius-m", type=float)
    parser.add_argument(
        "--maximum-pair-motion-disagreement-px",
        type=float,
    )
    parser.add_argument(
        "--maximum-pair-velocity-disagreement-mps",
        type=float,
    )
    parser.add_argument(
        "--maximum-pair-growth-disagreement",
        type=float,
    )
    parser.add_argument(
        "--minimum-pair-psr-advantage",
        type=float,
    )
    parser.add_argument(
        "--minimum-pair-confidence-ratio",
        type=float,
    )
    parser.add_argument(
        "--long-pair-confidence-penalty",
        type=float,
    )
    parser.add_argument("--minimum-growth-overlap-support", type=float)
    parser.add_argument("--minimum-growth-overlap-area-km2", type=float)
    parser.add_argument("--valid-times", nargs=3)
    parser.add_argument("--background-valid-times", nargs=3)
    parser.add_argument("--dx-m", type=float)
    parser.add_argument("--dy-m", type=float)
    parser.add_argument("--projection")
    parser.add_argument("--grid-hash")
    parser.add_argument(
        "--pixel-to-projected-matrix-m",
        type=float,
        nargs=4,
        metavar=("XX", "XR", "YX", "YR"),
    )
    parser.add_argument("--causal-support-uncertainty-m", type=float)
    parser.add_argument("--amplitude-displacement-tolerance-m", type=float)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="include optional positivity and transport audits",
    )
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("output path must end with .npz")
    if (
        args.mode != "operational"
        and args.operational_calibration_id is not None
    ):
        parser.error(
            "--operational-calibration-id requires --mode operational"
        )
    if not args.variational and (
        args.mode == "operational"
        or args.amplitude_information_policy is not None
        or args.amplitude_confidence_policy is not None
        or args.operational_calibration_id is not None
        or args.observation_std_dbz is not None
        or args.motion_increment_scale_mps is not None
        or args.maximum_detected_error_std is not None
        or args.maximum_unresolved_amplitude_fraction is not None
        or args.minimum_amplitude_total_quality_weight is not None
        or args.minimum_amplitude_effective_pixel_count is not None
        or args.minimum_integrated_echo_ratio_for_confidence is not None
        or args.maximum_integrated_echo_ratio_for_confidence is not None
        or args.minimum_soft_echo_area_ratio_for_confidence is not None
        or args.maximum_soft_echo_area_ratio_for_confidence is not None
        or (
            args.maximum_established_excess_growth_fraction_for_confidence
            is not None
        )
        or args.causal_support_uncertainty_m is not None
        or args.amplitude_displacement_tolerance_m is not None
    ):
        parser.error(
            "operational mode and P1 analysis settings require --variational"
        )

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

    default_nowcast_config = NowcastConfig()
    config = NowcastConfig(
        min_dbz=args.min_dbz,
        max_dbz=args.max_dbz,
        echo_threshold_dbz=args.echo_threshold_dbz,
        min_publish_support=args.min_publish_support,
        minimum_publish_verified_support=(
            args.minimum_publish_verified_support
        ),
        maximum_background_age_minutes=(
            args.maximum_background_age_minutes
        ),
        maximum_motion_speed_mps=args.maximum_motion_speed_mps,
        minimum_phase_correlation_psr=(
            default_nowcast_config.minimum_phase_correlation_psr
            if args.minimum_phase_correlation_psr is None
            else args.minimum_phase_correlation_psr
        ),
        pair_echo_dilation_m=args.pair_echo_dilation_m,
        phase_correlation_sidelobe_radius_m=(
            args.phase_correlation_sidelobe_radius_m
        ),
        maximum_pair_motion_disagreement_px=(
            default_nowcast_config.maximum_pair_motion_disagreement_px
            if args.maximum_pair_motion_disagreement_px is None
            else args.maximum_pair_motion_disagreement_px
        ),
        maximum_pair_velocity_disagreement_mps=(
            default_nowcast_config.maximum_pair_velocity_disagreement_mps
            if args.maximum_pair_velocity_disagreement_mps is None
            else args.maximum_pair_velocity_disagreement_mps
        ),
        maximum_pair_growth_disagreement=(
            default_nowcast_config.maximum_pair_growth_disagreement
            if args.maximum_pair_growth_disagreement is None
            else args.maximum_pair_growth_disagreement
        ),
        minimum_pair_psr_advantage=(
            default_nowcast_config.minimum_pair_psr_advantage
            if args.minimum_pair_psr_advantage is None
            else args.minimum_pair_psr_advantage
        ),
        minimum_pair_confidence_ratio=(
            default_nowcast_config.minimum_pair_confidence_ratio
            if args.minimum_pair_confidence_ratio is None
            else args.minimum_pair_confidence_ratio
        ),
        long_pair_confidence_penalty=(
            default_nowcast_config.long_pair_confidence_penalty
            if args.long_pair_confidence_penalty is None
            else args.long_pair_confidence_penalty
        ),
        minimum_growth_overlap_support=(
            default_nowcast_config.minimum_growth_overlap_support
            if args.minimum_growth_overlap_support is None
            else args.minimum_growth_overlap_support
        ),
        minimum_growth_overlap_area_km2=(
            args.minimum_growth_overlap_area_km2
        ),
    )
    frames_tensor = torch.as_tensor(frames, dtype=torch.float32)
    if args.variational:
        analysis_config = _analysis_config_from_args(
            parser,
            args,
            grid_time_contract=grid_time_contract,
        )
        result, analysis = variational_nowcast(
            frames_tensor,
            nowcast_config=config,
            analysis_config=analysis_config,
            observation_std_dbz=analysis_config.observation_std_dbz,
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
    atomic_savez_compressed(
        args.output,
        seal_forecast_run_arrays(output),
    )

    dy, dx = result.state.displacement_yx.tolist()
    print(
        f"saved {config.forecast_steps} frames to {args.output} "
        f"(motion dy={dy:.2f}, dx={dx:.2f} px/step)"
    )


def _analysis_config_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    grid_time_contract: RadarGridTimeContract | None,
) -> AnalysisConfig:
    defaults = AnalysisConfig()
    calibrated_values = {
        "--operational-calibration-id": args.operational_calibration_id,
        "--observation-std-dbz": args.observation_std_dbz,
        "--minimum-phase-correlation-psr": (
            args.minimum_phase_correlation_psr
        ),
        "--pair-echo-dilation-m": args.pair_echo_dilation_m,
        "--phase-correlation-sidelobe-radius-m": (
            args.phase_correlation_sidelobe_radius_m
        ),
        "--long-pair-confidence-penalty": (
            args.long_pair_confidence_penalty
        ),
        "--maximum-pair-velocity-disagreement-mps": (
            args.maximum_pair_velocity_disagreement_mps
        ),
        "--maximum-pair-growth-disagreement": (
            args.maximum_pair_growth_disagreement
        ),
        "--minimum-pair-psr-advantage": args.minimum_pair_psr_advantage,
        "--minimum-pair-confidence-ratio": (
            args.minimum_pair_confidence_ratio
        ),
        "--minimum-growth-overlap-support": (
            args.minimum_growth_overlap_support
        ),
        "--minimum-growth-overlap-area-km2": (
            args.minimum_growth_overlap_area_km2
        ),
        "--minimum-publish-verified-support": (
            args.minimum_publish_verified_support
        ),
        "--maximum-motion-speed-mps": args.maximum_motion_speed_mps,
        "--motion-increment-scale-mps": args.motion_increment_scale_mps,
        "--causal-support-uncertainty-m": (
            args.causal_support_uncertainty_m
        ),
        "--amplitude-displacement-tolerance-m": (
            args.amplitude_displacement_tolerance_m
        ),
        "--maximum-detected-error-std": args.maximum_detected_error_std,
        "--maximum-unresolved-amplitude-fraction": (
            args.maximum_unresolved_amplitude_fraction
        ),
        "--minimum-amplitude-total-quality-weight": (
            args.minimum_amplitude_total_quality_weight
        ),
        "--minimum-amplitude-effective-pixel-count": (
            args.minimum_amplitude_effective_pixel_count
        ),
        "--minimum-integrated-echo-ratio-for-confidence": (
            args.minimum_integrated_echo_ratio_for_confidence
        ),
        "--maximum-integrated-echo-ratio-for-confidence": (
            args.maximum_integrated_echo_ratio_for_confidence
        ),
        "--minimum-soft-echo-area-ratio-for-confidence": (
            args.minimum_soft_echo_area_ratio_for_confidence
        ),
        "--maximum-soft-echo-area-ratio-for-confidence": (
            args.maximum_soft_echo_area_ratio_for_confidence
        ),
        "--maximum-established-excess-growth-fraction-for-confidence": (
            args.maximum_established_excess_growth_fraction_for_confidence
        ),
    }
    if args.mode == "operational":
        missing = [
            name for name, value in calibrated_values.items() if value is None
        ]
        if grid_time_contract is None:
            missing.append("grid/time metadata")
        if missing:
            parser.error(
                "operational mode requires explicitly calibrated values for: "
                + ", ".join(missing)
            )
        if args.amplitude_information_policy not in (
            None,
            "operational_fallback",
        ) or args.amplitude_confidence_policy not in (
            None,
            "operational_fallback",
        ):
            parser.error(
                "operational mode requires operational_fallback amplitude "
                "policies"
            )
    information_policy = args.amplitude_information_policy or (
        "operational_fallback"
        if args.mode == "operational"
        else defaults.amplitude_information_policy
    )
    confidence_policy = args.amplitude_confidence_policy or (
        "operational_fallback"
        if args.mode == "operational"
        else defaults.amplitude_confidence_policy
    )

    def value(name: str, default: float) -> float:
        candidate = getattr(args, name)
        return default if candidate is None else candidate

    return AnalysisConfig(
        execution_mode=args.mode,
        operational_calibration_id=args.operational_calibration_id,
        detection_limit_dbz=args.echo_threshold_dbz,
        observation_std_dbz=value(
            "observation_std_dbz",
            defaults.observation_std_dbz,
        ),
        motion_increment_scale_mps=args.motion_increment_scale_mps,
        maximum_latest_detected_error_std=value(
            "maximum_detected_error_std",
            defaults.maximum_latest_detected_error_std,
        ),
        maximum_unresolved_amplitude_fraction=value(
            "maximum_unresolved_amplitude_fraction",
            defaults.maximum_unresolved_amplitude_fraction,
        ),
        minimum_amplitude_total_quality_weight=value(
            "minimum_amplitude_total_quality_weight",
            defaults.minimum_amplitude_total_quality_weight,
        ),
        minimum_amplitude_effective_pixel_count=value(
            "minimum_amplitude_effective_pixel_count",
            defaults.minimum_amplitude_effective_pixel_count,
        ),
        amplitude_information_policy=information_policy,
        amplitude_confidence_policy=confidence_policy,
        minimum_integrated_echo_ratio_for_confidence=value(
            "minimum_integrated_echo_ratio_for_confidence",
            defaults.minimum_integrated_echo_ratio_for_confidence,
        ),
        maximum_integrated_echo_ratio_for_confidence=value(
            "maximum_integrated_echo_ratio_for_confidence",
            defaults.maximum_integrated_echo_ratio_for_confidence,
        ),
        minimum_soft_echo_area_ratio_for_confidence=value(
            "minimum_soft_echo_area_ratio_for_confidence",
            defaults.minimum_soft_echo_area_ratio_for_confidence,
        ),
        maximum_soft_echo_area_ratio_for_confidence=value(
            "maximum_soft_echo_area_ratio_for_confidence",
            defaults.maximum_soft_echo_area_ratio_for_confidence,
        ),
        maximum_established_excess_growth_fraction_for_confidence=value(
            "maximum_established_excess_growth_fraction_for_confidence",
            defaults.maximum_established_excess_growth_fraction_for_confidence,
        ),
        causal_support_uncertainty_m=args.causal_support_uncertainty_m,
        amplitude_displacement_tolerance_m=(
            args.amplitude_displacement_tolerance_m
        ),
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
        physical_metadata = (
            args.background_valid_times,
            args.pixel_to_projected_matrix_m,
            args.maximum_motion_speed_mps,
            args.motion_increment_scale_mps,
            args.causal_support_uncertainty_m,
            args.amplitude_displacement_tolerance_m,
        )
        if any(value is not None for value in physical_metadata):
            parser.error("physical settings require grid/time metadata")
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
            pixel_to_projected_matrix_m=(
                None
                if args.pixel_to_projected_matrix_m is None
                else (
                    (
                        args.pixel_to_projected_matrix_m[0],
                        args.pixel_to_projected_matrix_m[1],
                    ),
                    (
                        args.pixel_to_projected_matrix_m[2],
                        args.pixel_to_projected_matrix_m[3],
                    ),
                )
            ),
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
            "minimum_publish_verified_support": np.asarray(
                np.nan
                if config.minimum_publish_verified_support is None
                else config.minimum_publish_verified_support
            ),
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
            analysis_amplitude_confidence_failed=np.asarray(
                analysis.amplitude_confidence_failed
            ),
            analysis_precursor_object_count_by_time=_optional_int_pair_array(
                analysis.precursor_object_count_by_time
            ),
            analysis_insufficient_amplitude_object_count_by_time=(
                _optional_int_pair_array(
                    analysis.insufficient_amplitude_object_count_by_time
                )
            ),
            analysis_maximum_object_unresolved_fraction_by_time=(
                _optional_pair_array(
                    analysis.maximum_object_unresolved_fraction_by_time
                )
            ),
            analysis_minimum_object_integrated_echo_ratio_by_time=(
                _optional_pair_array(
                    analysis.minimum_object_integrated_echo_ratio_by_time
                )
            ),
            analysis_maximum_object_integrated_echo_ratio_by_time=(
                _optional_pair_array(
                    analysis.maximum_object_integrated_echo_ratio_by_time
                )
            ),
            analysis_minimum_object_soft_echo_area_ratio_by_time=(
                _optional_pair_array(
                    analysis.minimum_object_soft_echo_area_ratio_by_time
                )
            ),
            analysis_maximum_object_soft_echo_area_ratio_by_time=(
                _optional_pair_array(
                    analysis.maximum_object_soft_echo_area_ratio_by_time
                )
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
            analysis_dynamics_data_gram_eigenvalues=(
                _optional_triple_array(
                    analysis.dynamics_data_gram_eigenvalues
                )
            ),
            analysis_dynamics_data_information_trace=np.asarray(
                np.nan
                if analysis.dynamics_data_information_trace is None
                else analysis.dynamics_data_information_trace
            ),
            analysis_dynamics_data_numerical_rank=np.asarray(
                -1
                if analysis.dynamics_data_numerical_rank is None
                else analysis.dynamics_data_numerical_rank
            ),
            analysis_dynamics_data_effective_dimension=np.asarray(
                np.nan
                if analysis.dynamics_data_effective_dimension is None
                else analysis.dynamics_data_effective_dimension
            ),
            analysis_dynamics_data_to_prior_ratio_by_mode=(
                _optional_triple_array(
                    analysis.dynamics_data_to_prior_ratio_by_mode
                )
            ),
            analysis_regularized_dynamics_hessian_eigenvalues=(
                _optional_triple_array(
                    analysis.regularized_dynamics_hessian_eigenvalues
                )
            ),
            analysis_regularized_dynamics_hessian_condition_number=(
                np.asarray(
                    np.nan
                    if analysis.regularized_dynamics_hessian_condition_number
                    is None
                    else analysis.regularized_dynamics_hessian_condition_number
                )
            ),
            analysis_field_smoothness_prior_cost=np.asarray(
                analysis.field_smoothness_prior_cost
            ),
            analysis_motion_control_coordinate_system=np.asarray(
                analysis.motion_control_coordinate_system
            ),
            analysis_field_smoothness_coordinate_system=np.asarray(
                analysis.field_smoothness_coordinate_system
            ),
            analysis_motion_saturation_margin_yx=(
                _optional_pair_array(analysis.motion_saturation_margin_yx)
            ),
            analysis_motion_speed_saturation_margin_mps=np.asarray(
                np.nan
                if analysis.motion_speed_saturation_margin_mps is None
                else analysis.motion_speed_saturation_margin_mps
            ),
            analysis_growth_saturation_margin=np.asarray(
                np.nan
                if analysis.growth_saturation_margin is None
                else analysis.growth_saturation_margin
            ),
            analysis_field_growth_jacobian_cosine=np.asarray(
                np.nan
                if analysis.field_growth_jacobian_cosine is None
                else analysis.field_growth_jacobian_cosine
            ),
            analysis_field_motion_jacobian_cosine_by_control=(
                _optional_nullable_pair_array(
                    analysis.field_motion_jacobian_cosine_by_control
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


def _optional_int_pair_array(
    value: tuple[int, int] | None,
) -> NDArray[Any]:
    return np.asarray((0, 0) if value is None else value)


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


if __name__ == "__main__":
    main()
