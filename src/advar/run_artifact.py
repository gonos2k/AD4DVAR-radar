from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, BinaryIO, cast
import zipfile

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from ._digest import json_digest
from .nowcast import (
    DataStatus,
    DynamicsSource,
    ForecastMetadata,
    ForecastResult,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    StatePathProvenance,
    TendencyPairSelection,
    TendencySource,
    forecast_evidence_fields,
    forecast_from_state,
)


FORECAST_RUN_ARTIFACT_VERSION = "forecast-run-v49"
_LEGACY_FORECAST_RUN_ARTIFACT_VERSIONS = {
    "forecast-run-v42",
    "forecast-run-v43",
    "forecast-run-v44",
    "forecast-run-v45",
    "forecast-run-v46",
    "forecast-run-v47",
    "forecast-run-v48",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_MAXIMUM_MEMBER_COUNT = 224
DEFAULT_MAXIMUM_MEMBER_BYTES = 1024**3
DEFAULT_MAXIMUM_TOTAL_EXPANDED_BYTES = 2 * 1024**3
_ArtifactArrays = Mapping[str, NDArray[Any]]
_CORE_ARRAY_NAMES = frozenset(
    {
        "forecast_run_artifact_version",
        "forecast_run_artifact_digest",
        "forecast_run_digest",
        "forecast_integrator_version",
        "nowcast_config_json",
        "nowcast_config_digest",
        "input_bundle_digest",
        "input_frames_digest",
        "observation_masks_digest",
        "observation_quality_weight_digest",
        "observation_std_dbz_digest",
        "background_frames_digest",
        "fixed_input_context_digest",
        "full_analysis_input_digest",
        "run_background_age_minutes",
        "grid_time_contract_present",
        "grid_time_contract_json",
        "grid_time_contract_digest",
        "forecast_dbz",
        "forecast_dbz_digest",
        "valid_mask",
        "valid_mask_digest",
        "forecast_path_verified_support",
        "forecast_verified_support",
        "forecast_local_motion_verified_support",
        "forecast_local_growth_verified_support",
        "forecast_local_dynamics_verified_support",
        "forecast_observation_verified_support",
        "forecast_background_verified_support",
        "forecast_velocity_uncertainty_mps",
        "motion_evidence_uncertainty_multiplier",
        "growth_evidence_uncertainty_multiplier",
        "forecast_position_uncertainty_m",
        "forecast_log_growth_uncertainty",
        "forecast_confidence",
        "radar_anchored_valid_mask",
        "radar_state_anchored_valid_mask",
        "radar_dynamics_anchored_valid_mask",
        "background_dynamics_mask",
        "background_fallback_mask",
        "state_echo_linear",
        "displacement_yx",
        "displacement_mps_yx",
        "grid_velocity_mps_yx",
        "projected_velocity_mps_xy",
        "log_growth_per_step",
        "state_metadata_digest",
        "coverage_by_frame",
        "data_status",
        "background_used",
        "background_contribution_fraction",
        "background_state_support_fraction",
        "observation_state_support_fraction",
        "background_tendency_used",
        "background_age_minutes",
        "source_support",
        "observation_source_support",
        "background_source_support",
        "path_verified_source_support",
        "verified_source_support",
        "local_motion_verified_support",
        "local_growth_verified_support",
        "local_dynamics_verified_support",
        "observation_verified_source_support",
        "background_verified_source_support",
        "motion_disagreement_px",
        "motion_disagreement_mps",
        "growth_disagreement",
        "maximum_growth_saturation_excess",
        "posterior_velocity_uncertainty_mps",
        "posterior_log_growth_uncertainty_per_step",
        "p1_velocity_saturation_uncertainty_mps",
        "p1_log_growth_saturation_uncertainty_per_step",
        "minimum_phase_correlation_psr",
        "tendency_pair_count",
        "motion_pair_count",
        "growth_pair_count",
        "motion_pair_selection",
        "growth_pair_selection",
        "motion_pair_conflict",
        "growth_pair_conflict",
        "tendency_source",
        "dynamics_source",
        "state_path_source",
        "state_path_mode",
        "state_path_pair_count",
        "state_path_minimum_psr",
        "state_path_conflict",
        "state_path_extrapolated",
        "state_path_age_minutes",
        "observation_path_mode",
        "observation_path_pair_count",
        "observation_path_minimum_psr",
        "observation_path_conflict",
        "observation_path_extrapolated",
        "observation_path_age_minutes",
        "background_path_mode",
        "background_path_pair_count",
        "background_path_minimum_psr",
        "background_path_conflict",
        "background_path_extrapolated",
        "background_path_age_minutes",
        "minimum_growth_overlap_support",
        "minimum_growth_overlap_area_km2",
        "provenance",
        "latest_frame_dbz",
        "latest_observation_mask",
        "latest_background_dbz",
        "latest_observation_mask_digest",
        "latest_frame_digest",
        "latest_background_present",
        "latest_background_digest",
        "analysis_config_present",
        "analysis_config_json",
        "analysis_config_digest",
        "analysis_input_digest",
        "operational_runtime_profile_digest",
        "operational_calibration_manifest_present",
        "operational_calibration_manifest_json",
        "operational_calibration_manifest_digest",
        "operational_calibration_approval_digest",
        "operational_data_identity_present",
        "operational_data_identity_json",
        "operational_data_identity_digest",
        "neural_prior_digest",
        "prior_application_digest",
        "prior_model_contract_digest",
        "prior_feature_schema_digest",
        "prior_training_manifest_digest",
        "prior_inference_evidence_digest",
        "prior_inference_algorithm_digest",
        "prior_numerical_runtime_digest",
        "prior_dependency",
        "prior_role",
        "input_plan_json",
        "input_plan_digest",
        "input_plan_resolution_digest",
    }
)
_CLI_EXTRA_ARRAY_NAMES = frozenset(
    {
        "output_contract_version",
        "min_publish_support",
        "minimum_publish_verified_support",
        "minimum_publish_confidence",
        "minimum_publish_observation_verified_support",
        "maximum_publish_background_fraction",
        "lead_minutes",
        "analysis_used",
        "analysis_converged",
        "analysis_outer_converged",
        "analysis_final_linearization_stationary",
        "analysis_final_robust_stationary",
        "analysis_final_irls_fixed_point",
        "analysis_p1_forecast_eligible",
        "analysis_posterior_eligible",
        "analysis_fso_eligible",
        "analysis_degraded",
        "analysis_used_fallback",
        "analysis_reason",
        "analysis_initial_objective",
        "analysis_final_objective",
        "analysis_robust_gradient_norm",
        "analysis_linearization_field_gradient_rms",
        "analysis_linearization_field_gradient_max",
        "analysis_linearization_dynamics_gradient_max",
        "analysis_robust_field_gradient_rms",
        "analysis_robust_field_gradient_max",
        "analysis_robust_dynamics_gradient_max",
        "analysis_robust_relative_stationarity",
        "analysis_irls_relative_weight_change",
        "analysis_outer_iterations",
        "analysis_pcg_iterations",
        "analysis_minimum_reachability_margin",
        "analysis_unresolved_amplitude_fraction",
        "analysis_unresolved_amplitude_fraction_by_time",
        "analysis_unresolved_pixel_fraction_by_time",
        "analysis_amplitude_violation_score",
        "analysis_amplitude_violation_score_by_time",
        "analysis_integrated_echo_ratio_by_time",
        "analysis_displacement_tolerant_soft_echo_area_ratio_by_time",
        "analysis_effective_precursor_pixel_count_by_time",
        "analysis_bad_quality_weight_by_time",
        "analysis_total_quality_weight_by_time",
        "analysis_amplitude_information_sufficient_by_time",
        "analysis_insufficient_amplitude_information",
        "analysis_amplitude_confidence_failed",
        "analysis_precursor_object_count_by_time",
        "analysis_insufficient_amplitude_object_count_by_time",
        "analysis_maximum_object_unresolved_fraction_by_time",
        "analysis_minimum_object_integrated_echo_ratio_by_time",
        "analysis_maximum_object_integrated_echo_ratio_by_time",
        "analysis_minimum_object_soft_echo_area_ratio_by_time",
        "analysis_maximum_object_soft_echo_area_ratio_by_time",
        "analysis_minimum_object_count_ratio_by_time",
        "analysis_established_echo_excess_growth_fraction",
        "analysis_established_echo_excess_growth_fraction_by_time",
        "analysis_maximum_growth_envelope_ratio",
        "analysis_maximum_growth_envelope_ratio_by_time",
        "analysis_amplitude_diagnostics_source",
        "analysis_relative_objective_reduction",
        "analysis_causal_control_cell_count",
        "analysis_causal_seed_cell_count",
        "analysis_causal_seed_prior_cost",
        "analysis_dynamics_data_gram_eigenvalues",
        "analysis_dynamics_data_information_trace",
        "analysis_dynamics_data_numerical_rank",
        "analysis_dynamics_data_effective_dimension",
        "analysis_dynamics_data_to_prior_ratio_by_mode",
        "analysis_field_conditioned_dynamics_data_gram_eigenvalues",
        "analysis_field_conditioned_dynamics_data_information_trace",
        "analysis_field_conditioned_dynamics_data_effective_dimension",
        "analysis_field_conditioning_maximum_relative_residual",
        "analysis_regularized_dynamics_hessian_eigenvalues",
        "analysis_regularized_dynamics_hessian_condition_number",
        "analysis_field_smoothness_prior_cost",
        "analysis_motion_control_coordinate_system",
        "analysis_field_smoothness_coordinate_system",
        "analysis_motion_saturation_margin_yx",
        "analysis_motion_speed_saturation_margin_mps",
        "analysis_growth_saturation_margin",
        "analysis_field_growth_jacobian_cosine",
        "analysis_field_motion_jacobian_cosine_by_control",
        "input_minimum_before_fix",
        "input_corrected_count",
        "input_corrected_integral",
        "forecast_minimum_before_fix",
        "forecast_corrected_count",
        "forecast_corrected_integral",
        "echo_integral_before_transport",
        "echo_integral_after_transport",
        "boundary_outflow_integral",
        "echo_budget_error",
        "analysis_minimum_before_fix",
        "analysis_corrected_count",
        "analysis_corrected_integral",
    }
)


def forecast_run_arrays(result: ForecastResult) -> dict[str, Any]:
    """Return the canonical NPZ payload needed to reconstruct ``result``.

    Optional positivity/transport audit objects are intentionally excluded;
    they do not participate in delayed M0 sensitivity.
    """

    result.validate_issuance()
    if any(
        value is None
        for value in (
            result.run.observation_masks_digest,
            result.run.observation_quality_weight_digest,
            result.run.observation_std_dbz_digest,
            result.run.fixed_input_context_digest,
            result.run.full_analysis_input_digest,
        )
    ):
        raise ValueError("current run artifacts require complete input context")
    config = result.run.config
    config_json = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    latest_observation_mask = result.run.latest_observation_mask
    metadata = result.metadata
    grid_time_contract = result.run.grid_time_contract
    operational_runtime_profile_digest = (
        result.run.operational_runtime_profile_digest
    )
    displacement_mps = result.displacement_mps_yx
    projected_velocity = result.projected_velocity_mps_xy
    arrays = {
        "forecast_run_artifact_version": np.asarray(
            FORECAST_RUN_ARTIFACT_VERSION
        ),
        "forecast_run_artifact_digest": np.asarray(""),
        "forecast_run_digest": np.asarray(result.forecast_run_digest),
        "forecast_integrator_version": np.asarray(
            result.run.forecast_integrator_version
        ),
        "nowcast_config_json": np.asarray(config_json),
        "nowcast_config_digest": np.asarray(config.digest),
        "input_bundle_digest": np.asarray(result.run.input_bundle_digest),
        "input_frames_digest": np.asarray(result.run.input_frames_digest),
        "observation_masks_digest": np.asarray(
            result.run.observation_masks_digest
        ),
        "observation_quality_weight_digest": np.asarray(
            result.run.observation_quality_weight_digest
        ),
        "observation_std_dbz_digest": np.asarray(
            result.run.observation_std_dbz_digest
        ),
        "background_frames_digest": np.asarray(
            ""
            if result.run.background_frames_digest is None
            else result.run.background_frames_digest
        ),
        "fixed_input_context_digest": np.asarray(
            result.run.fixed_input_context_digest
        ),
        "full_analysis_input_digest": np.asarray(
            result.run.full_analysis_input_digest
        ),
        "neural_prior_digest": np.asarray(
            "" if result.run.neural_prior_digest is None else result.run.neural_prior_digest
        ),
        "prior_application_digest": np.asarray(
            ""
            if result.run.prior_application_digest is None
            else result.run.prior_application_digest
        ),
        "prior_model_contract_digest": np.asarray(
            ""
            if result.run.prior_model_contract_digest is None
            else result.run.prior_model_contract_digest
        ),
        "prior_feature_schema_digest": np.asarray(
            ""
            if result.run.prior_feature_schema_digest is None
            else result.run.prior_feature_schema_digest
        ),
        "prior_training_manifest_digest": np.asarray(
            ""
            if result.run.prior_training_manifest_digest is None
            else result.run.prior_training_manifest_digest
        ),
        "prior_inference_evidence_digest": np.asarray(
            ""
            if result.run.prior_inference_evidence_digest is None
            else result.run.prior_inference_evidence_digest
        ),
        "prior_inference_algorithm_digest": np.asarray(
            ""
            if result.run.prior_inference_algorithm_digest is None
            else result.run.prior_inference_algorithm_digest
        ),
        "prior_numerical_runtime_digest": np.asarray(
            ""
            if result.run.prior_numerical_runtime_digest is None
            else result.run.prior_numerical_runtime_digest
        ),
        "prior_dependency": np.asarray(
            "" if result.run.prior_dependency is None else result.run.prior_dependency
        ),
        "prior_role": np.asarray(
            "" if result.run.prior_role is None else result.run.prior_role
        ),
        "input_plan_json": np.asarray(
            "" if result.run.input_plan_json is None else result.run.input_plan_json
        ),
        "input_plan_digest": np.asarray(
            "" if result.run.input_plan_digest is None else result.run.input_plan_digest
        ),
        "input_plan_resolution_digest": np.asarray(
            ""
            if result.run.input_plan_resolution_digest is None
            else result.run.input_plan_resolution_digest
        ),
        "run_background_age_minutes": np.asarray(
            np.nan
            if result.run.background_age_minutes is None
            else result.run.background_age_minutes
        ),
        "grid_time_contract_present": np.asarray(
            grid_time_contract is not None
        ),
        "grid_time_contract_json": np.asarray(
            ""
            if grid_time_contract is None
            else json.dumps(
                asdict(grid_time_contract),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "grid_time_contract_digest": np.asarray(
            ""
            if result.run.grid_time_contract_digest is None
            else result.run.grid_time_contract_digest
        ),
        "forecast_dbz": _numpy(result.forecast_dbz),
        "forecast_dbz_digest": np.asarray(result.forecast_dbz_digest),
        "valid_mask": _numpy(result.valid_mask),
        "valid_mask_digest": np.asarray(result.valid_mask_digest),
        "forecast_verified_support": _numpy(
            result.forecast_verified_support
        ),
        "forecast_local_motion_verified_support": _numpy(
            result.forecast_local_motion_verified_support
        ),
        "forecast_local_growth_verified_support": _numpy(
            result.forecast_local_growth_verified_support
        ),
        "forecast_local_dynamics_verified_support": _numpy(
            result.forecast_local_dynamics_verified_support
        ),
        "forecast_path_verified_support": _numpy(
            result.forecast_path_verified_support
        ),
        "forecast_observation_verified_support": _numpy(
            result.forecast_observation_verified_support
        ),
        "forecast_background_verified_support": _numpy(
            result.forecast_background_verified_support
        ),
        "forecast_velocity_uncertainty_mps": _numpy(
            result.forecast_velocity_uncertainty_mps
        ),
        "motion_evidence_uncertainty_multiplier": _numpy(
            result.motion_evidence_uncertainty_multiplier
        ),
        "growth_evidence_uncertainty_multiplier": _numpy(
            result.growth_evidence_uncertainty_multiplier
        ),
        "forecast_position_uncertainty_m": _numpy(
            result.forecast_position_uncertainty_m
        ),
        "forecast_log_growth_uncertainty": _numpy(
            result.forecast_log_growth_uncertainty
        ),
        "forecast_confidence": _numpy(result.forecast_confidence),
        "radar_anchored_valid_mask": _numpy(
            result.radar_anchored_valid_mask
        ),
        "radar_state_anchored_valid_mask": _numpy(
            result.radar_state_anchored_valid_mask
        ),
        "radar_dynamics_anchored_valid_mask": _numpy(
            result.radar_dynamics_anchored_valid_mask
        ),
        "background_dynamics_mask": _numpy(
            result.background_dynamics_mask
        ),
        "background_fallback_mask": _numpy(
            result.background_fallback_mask
        ),
        "state_echo_linear": _numpy(result.state.echo_linear),
        "displacement_yx": _numpy(result.state.displacement_yx),
        "displacement_mps_yx": (
            np.full(2, np.nan, dtype=np.float64)
            if displacement_mps is None
            else _numpy(displacement_mps)
        ),
        "grid_velocity_mps_yx": (
            np.full(2, np.nan, dtype=np.float64)
            if result.grid_velocity_mps_yx is None
            else _numpy(result.grid_velocity_mps_yx)
        ),
        "projected_velocity_mps_xy": (
            np.full(2, np.nan, dtype=np.float64)
            if projected_velocity is None
            else _numpy(projected_velocity)
        ),
        "log_growth_per_step": _numpy(
            result.state.log_growth_per_step
        ),
        "state_metadata_digest": np.asarray(result.state_metadata_digest),
        "coverage_by_frame": _numpy(metadata.coverage_by_frame),
        "data_status": np.asarray(metadata.data_status.value),
        "background_used": np.asarray(metadata.background_used),
        "background_contribution_fraction": np.asarray(
            metadata.background_contribution_fraction
        ),
        "background_state_support_fraction": np.asarray(
            metadata.background_state_support_fraction
        ),
        "observation_state_support_fraction": np.asarray(
            metadata.observation_state_support_fraction
        ),
        "background_tendency_used": np.asarray(
            metadata.background_tendency_used
        ),
        "background_age_minutes": np.asarray(
            np.nan
            if metadata.background_age_minutes is None
            else metadata.background_age_minutes
        ),
        "source_support": _numpy(metadata.source_support),
        "observation_source_support": _numpy(
            metadata.observation_source_support
        ),
        "background_source_support": _numpy(
            metadata.background_source_support
        ),
        "path_verified_source_support": _numpy(
            metadata.path_verified_source_support
        ),
        "verified_source_support": _numpy(
            metadata.verified_source_support
        ),
        "local_motion_verified_support": _numpy(
            metadata.local_motion_verified_support
        ),
        "local_growth_verified_support": _numpy(
            metadata.local_growth_verified_support
        ),
        "local_dynamics_verified_support": _numpy(
            metadata.local_dynamics_verified_support
        ),
        "observation_verified_source_support": _numpy(
            metadata.observation_verified_source_support
        ),
        "background_verified_source_support": _numpy(
            metadata.background_verified_source_support
        ),
        "motion_disagreement_px": _numpy(
            metadata.motion_disagreement_px
        ),
        "motion_disagreement_mps": _numpy(
            metadata.motion_disagreement_mps
        ),
        "growth_disagreement": _numpy(metadata.growth_disagreement),
        "maximum_growth_saturation_excess": _numpy(
            metadata.maximum_growth_saturation_excess
        ),
        "posterior_velocity_uncertainty_mps": _numpy(
            metadata.posterior_velocity_uncertainty_mps
        ),
        "posterior_log_growth_uncertainty_per_step": _numpy(
            metadata.posterior_log_growth_uncertainty_per_step
        ),
        "p1_velocity_saturation_uncertainty_mps": _numpy(
            metadata.p1_velocity_saturation_uncertainty_mps
        ),
        "p1_log_growth_saturation_uncertainty_per_step": _numpy(
            metadata.p1_log_growth_saturation_uncertainty_per_step
        ),
        "minimum_phase_correlation_psr": _numpy(
            metadata.minimum_phase_correlation_psr
        ),
        "tendency_pair_count": np.asarray(metadata.tendency_pair_count),
        "motion_pair_count": np.asarray(metadata.motion_pair_count),
        "growth_pair_count": np.asarray(metadata.growth_pair_count),
        "motion_pair_selection": np.asarray(
            metadata.motion_pair_selection.value
        ),
        "growth_pair_selection": np.asarray(
            metadata.growth_pair_selection.value
        ),
        "motion_pair_conflict": np.asarray(metadata.motion_pair_conflict),
        "growth_pair_conflict": np.asarray(metadata.growth_pair_conflict),
        "tendency_source": np.asarray(metadata.tendency_source.value),
        "dynamics_source": np.asarray(metadata.dynamics_source.value),
        "state_path_source": np.asarray(metadata.state_path_source.value),
        "state_path_mode": np.asarray(metadata.state_path_mode.value),
        "state_path_pair_count": np.asarray(metadata.state_path_pair_count),
        "state_path_minimum_psr": np.asarray(
            metadata.state_path_minimum_psr
        ),
        "state_path_conflict": np.asarray(metadata.state_path_conflict),
        "state_path_extrapolated": np.asarray(
            metadata.state_path_extrapolated
        ),
        "state_path_age_minutes": np.asarray(
            np.nan
            if metadata.state_path_age_minutes is None
            else metadata.state_path_age_minutes
        ),
        **_path_provenance_arrays(
            "observation_path",
            metadata.observation_path,
        ),
        **_path_provenance_arrays("background_path", metadata.background_path),
        "minimum_growth_overlap_support": np.asarray(
            metadata.minimum_growth_overlap_support
        ),
        "minimum_growth_overlap_area_km2": np.asarray(
            metadata.minimum_growth_overlap_area_km2
        ),
        "provenance": np.asarray(metadata.provenance),
        "latest_observation_mask": _numpy(latest_observation_mask),
        "latest_frame_dbz": _numpy(result.run.latest_frame_dbz),
        "latest_background_dbz": (
            np.empty((0,), dtype=np.float64)
            if result.run.latest_background_dbz is None
            else _numpy(result.run.latest_background_dbz)
        ),
        "latest_observation_mask_digest": np.asarray(
            result.run.latest_observation_mask_digest
        ),
        "latest_frame_digest": np.asarray(result.run.latest_frame_digest),
        "latest_background_present": np.asarray(
            result.run.latest_background_digest is not None
        ),
        "latest_background_digest": np.asarray(
            ""
            if result.run.latest_background_digest is None
            else result.run.latest_background_digest
        ),
        "analysis_config_present": np.asarray(
            result.run.analysis_config_json is not None
        ),
        "analysis_config_json": np.asarray(
            ""
            if result.run.analysis_config_json is None
            else result.run.analysis_config_json
        ),
        "analysis_config_digest": np.asarray(
            ""
            if result.run.analysis_config_digest is None
            else result.run.analysis_config_digest
        ),
        "analysis_input_digest": np.asarray(
            ""
            if result.run.analysis_input_digest is None
            else result.run.analysis_input_digest
        ),
        "operational_runtime_profile_digest": np.asarray(
            ""
            if operational_runtime_profile_digest is None
            else operational_runtime_profile_digest
        ),
        "operational_calibration_manifest_present": np.asarray(
            result.run.operational_calibration_manifest_json is not None
        ),
        "operational_calibration_manifest_json": np.asarray(
            ""
            if result.run.operational_calibration_manifest_json is None
            else result.run.operational_calibration_manifest_json
        ),
        "operational_calibration_manifest_digest": np.asarray(
            ""
            if result.run.operational_calibration_manifest_digest is None
            else result.run.operational_calibration_manifest_digest
        ),
        "operational_calibration_approval_digest": np.asarray(
            ""
            if result.run.operational_calibration_approval_digest is None
            else result.run.operational_calibration_approval_digest
        ),
        "operational_data_identity_present": np.asarray(
            result.run.operational_data_identity_json is not None
        ),
        "operational_data_identity_json": np.asarray(
            ""
            if result.run.operational_data_identity_json is None
            else result.run.operational_data_identity_json
        ),
        "operational_data_identity_digest": np.asarray(
            ""
            if result.run.operational_data_identity_digest is None
            else result.run.operational_data_identity_digest
        ),
    }
    return seal_forecast_run_arrays(arrays)


def _path_provenance_arrays(
    prefix: str,
    path: StatePathProvenance,
) -> dict[str, NDArray[Any]]:
    return {
        f"{prefix}_mode": np.asarray(path.mode.value),
        f"{prefix}_pair_count": np.asarray(path.pair_count),
        f"{prefix}_minimum_psr": np.asarray(path.minimum_psr),
        f"{prefix}_conflict": np.asarray(path.conflict),
        f"{prefix}_extrapolated": np.asarray(path.extrapolated),
        f"{prefix}_age_minutes": np.asarray(
            np.nan if path.age_minutes is None else path.age_minutes
        ),
    }


def save_forecast_run(result: ForecastResult, path: str | Path) -> None:
    """Atomically persist one issued forecast run as a compressed NPZ."""

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("forecast run artifact path must end with .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = forecast_run_arrays(result)
    atomic_savez_compressed(destination, arrays)


def atomic_savez_compressed(
    destination: Path,
    arrays: dict[str, Any],
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _preflight_archive(
    source: BinaryIO,
    *,
    maximum_member_count: int,
    maximum_member_bytes: int,
    maximum_total_expanded_bytes: int,
) -> None:
    limits = {
        "maximum_member_count": maximum_member_count,
        "maximum_member_bytes": maximum_member_bytes,
        "maximum_total_expanded_bytes": maximum_total_expanded_bytes,
    }
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("forecast run archive limits must be positive integers")
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if len(members) > maximum_member_count:
                raise ValueError("forecast run archive has too many members")
            raw_names = [member.filename for member in members]
            if len(set(raw_names)) != len(raw_names):
                raise ValueError(
                    "forecast run archive has duplicate members"
                )
            names: set[str] = set()
            expanded_bytes = 0
            declared_array_bytes = 0
            for member in members:
                if (
                    member.is_dir()
                    or not member.filename.endswith(".npy")
                    or "/" in member.filename
                    or "\\" in member.filename
                ):
                    raise ValueError(
                        "forecast run archive has an invalid member name"
                    )
                if member.flag_bits & 0x1:
                    raise ValueError(
                        "encrypted forecast run members are unsupported"
                    )
                if member.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                }:
                    raise ValueError(
                        "unsupported forecast run compression method"
                    )
                if member.file_size > maximum_member_bytes:
                    raise ValueError(
                        "forecast run archive member is too large"
                    )
                expanded_bytes += member.file_size
                if expanded_bytes > maximum_total_expanded_bytes:
                    raise ValueError(
                        "forecast run archive expands beyond its limit"
                    )
                declared_array_bytes += _declared_array_bytes(
                    archive,
                    member,
                    maximum_member_bytes=maximum_member_bytes,
                )
                if declared_array_bytes > maximum_total_expanded_bytes:
                    raise ValueError(
                        "forecast run arrays exceed the expanded size limit"
                    )
                names.add(member.filename[:-4])
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("invalid forecast run archive") from error
    allowed = set(_CORE_ARRAY_NAMES)
    if "output_contract_version" in names:
        allowed.update(_CLI_EXTRA_ARRAY_NAMES)
    unknown = names - allowed
    if unknown:
        raise ValueError(
            f"forecast run archive has unknown members: {sorted(unknown)}"
        )


def _declared_array_bytes(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    maximum_member_bytes: int,
) -> int:
    try:
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(
                    stream
                )
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(
                    stream
                )
            else:
                raise ValueError(
                    f"unsupported NPY format version: {version}"
                )
            if dtype.hasobject or dtype.fields is not None:
                raise ValueError(
                    "forecast run arrays must use plain non-object dtypes"
                )
            array_bytes = math.prod(shape) * dtype.itemsize
            if array_bytes > maximum_member_bytes:
                raise ValueError(
                    "forecast run member declares too much array data"
                )
            payload_bytes = member.file_size - stream.tell()
            if payload_bytes != array_bytes:
                raise ValueError(
                    "forecast run member payload disagrees with its header"
                )
            return array_bytes
    except ValueError:
        raise
    except (OSError, EOFError) as error:
        raise ValueError("invalid forecast run array header") from error


def load_forecast_run(
    path: str | Path,
    *,
    maximum_member_count: int = DEFAULT_MAXIMUM_MEMBER_COUNT,
    maximum_member_bytes: int = DEFAULT_MAXIMUM_MEMBER_BYTES,
    maximum_total_expanded_bytes: int = (
        DEFAULT_MAXIMUM_TOTAL_EXPANDED_BYTES
    ),
) -> ForecastResult:
    """Load and independently validate a durable forecast run artifact."""

    source = Path(path)
    artifact = _open_preflighted_artifact(
        source,
        maximum_member_count=maximum_member_count,
        maximum_member_bytes=maximum_member_bytes,
        maximum_total_expanded_bytes=maximum_total_expanded_bytes,
    )
    with artifact, np.load(artifact, allow_pickle=False) as archive:
        version_name = "forecast_run_artifact_version"
        loaded_arrays: dict[str, NDArray[Any]] = {
            version_name: np.array(archive[version_name], copy=True)
        }
        version = _string_scalar(loaded_arrays, version_name)
        if version not in (
            {FORECAST_RUN_ARTIFACT_VERSION}
            | _LEGACY_FORECAST_RUN_ARTIFACT_VERSIONS
        ):
            raise ValueError(f"unsupported forecast run artifact: {version}")

        loaded_arrays.update(
            {
                name: np.array(archive[name], copy=True)
                for name in archive.files
                if name != version_name
            }
        )
        stored_artifact_digest = _digest_scalar(
            loaded_arrays,
            "forecast_run_artifact_digest",
        )
        expected_artifact_digest = _forecast_run_artifact_digest(
            loaded_arrays
        )
        if stored_artifact_digest != expected_artifact_digest:
            raise ValueError("forecast run artifact digest mismatch")

        config_json = _string_scalar(loaded_arrays, "nowcast_config_json")
        try:
            config_value = json.loads(config_json)
        except json.JSONDecodeError as error:
            raise ValueError("invalid nowcast_config_json") from error
        if not isinstance(config_value, dict):
            raise ValueError("nowcast_config_json must contain an object")
        config = NowcastConfig(**cast(dict[str, Any], config_value))
        config_digest = _digest_scalar(
            loaded_arrays,
            "nowcast_config_digest",
        )
        if config.digest != config_digest:
            raise ValueError("nowcast config digest mismatch")

        forecast_dbz = _tensor(loaded_arrays, "forecast_dbz")
        valid_mask = _tensor(loaded_arrays, "valid_mask")
        stored_forecast_verified_support = _tensor(
            loaded_arrays,
            "forecast_verified_support",
        )
        stored_forecast_local_motion_verified_support = _tensor(
            loaded_arrays,
            "forecast_local_motion_verified_support",
        )
        stored_forecast_local_growth_verified_support = _tensor(
            loaded_arrays,
            "forecast_local_growth_verified_support",
        )
        stored_forecast_local_dynamics_verified_support = _tensor(
            loaded_arrays,
            "forecast_local_dynamics_verified_support",
        )
        stored_forecast_path_verified_support = _tensor(
            loaded_arrays,
            "forecast_path_verified_support",
        )
        stored_forecast_observation_verified_support = _tensor(
            loaded_arrays,
            "forecast_observation_verified_support",
        )
        stored_forecast_background_verified_support = _tensor(
            loaded_arrays,
            "forecast_background_verified_support",
        )
        stored_forecast_velocity_uncertainty_mps = _tensor(
            loaded_arrays,
            "forecast_velocity_uncertainty_mps",
        )
        stored_motion_evidence_multiplier = _tensor(
            loaded_arrays,
            "motion_evidence_uncertainty_multiplier",
        )
        stored_growth_evidence_multiplier = _tensor(
            loaded_arrays,
            "growth_evidence_uncertainty_multiplier",
        )
        stored_forecast_position_uncertainty_m = _tensor(
            loaded_arrays,
            "forecast_position_uncertainty_m",
        )
        stored_forecast_log_growth_uncertainty = _tensor(
            loaded_arrays,
            "forecast_log_growth_uncertainty",
        )
        stored_forecast_confidence = _tensor(
            loaded_arrays,
            "forecast_confidence",
        )
        stored_radar_anchored_valid_mask = _tensor(
            loaded_arrays,
            "radar_anchored_valid_mask",
        )
        stored_radar_state_anchored_valid_mask = _tensor(
            loaded_arrays,
            "radar_state_anchored_valid_mask",
        )
        stored_radar_dynamics_anchored_valid_mask = _tensor(
            loaded_arrays,
            "radar_dynamics_anchored_valid_mask",
        )
        stored_background_dynamics_mask = _tensor(
            loaded_arrays,
            "background_dynamics_mask",
        )
        stored_background_fallback_mask = _tensor(
            loaded_arrays,
            "background_fallback_mask",
        )
        stored_displacement_mps = _tensor(
            loaded_arrays,
            "displacement_mps_yx",
        )
        stored_grid_velocity = _tensor(
            loaded_arrays,
            "grid_velocity_mps_yx",
        )
        stored_projected_velocity = _tensor(
            loaded_arrays,
            "projected_velocity_mps_xy",
        )
        state = RadarState(
            echo_linear=_tensor(loaded_arrays, "state_echo_linear"),
            displacement_yx=_tensor(loaded_arrays, "displacement_yx"),
            log_growth_per_step=_tensor(
                loaded_arrays,
                "log_growth_per_step",
            ),
        )
        background_age = _float_scalar(
            loaded_arrays,
            "background_age_minutes",
            allow_nan=True,
        )
        background_state_support_fraction = _float_scalar(
            loaded_arrays,
            "background_state_support_fraction",
        )
        background_contribution_fraction = _float_scalar(
            loaded_arrays,
            "background_contribution_fraction",
        )
        if (
            background_state_support_fraction
            != background_contribution_fraction
        ):
            raise ValueError("background state support fraction mismatch")
        stored_background_tendency_used = _bool_scalar(
            loaded_arrays,
            "background_tendency_used",
        )
        state_path_age = _float_scalar(
            loaded_arrays,
            "state_path_age_minutes",
            allow_nan=True,
        )
        observation_path = _path_provenance_from_arrays(
            loaded_arrays,
            "observation_path",
        )
        background_path = _path_provenance_from_arrays(
            loaded_arrays,
            "background_path",
        )
        metadata = ForecastMetadata(
            data_status=DataStatus(
                _string_scalar(loaded_arrays, "data_status")
            ),
            coverage_by_frame=_tensor(
                loaded_arrays,
                "coverage_by_frame",
            ),
            background_used=_bool_scalar(
                loaded_arrays,
                "background_used",
            ),
            background_contribution_fraction=(
                background_contribution_fraction
            ),
            background_age_minutes=(
                None if math.isnan(background_age) else background_age
            ),
            source_support=_tensor(loaded_arrays, "source_support"),
            observation_source_support=_tensor(
                loaded_arrays,
                "observation_source_support",
            ),
            background_source_support=_tensor(
                loaded_arrays,
                "background_source_support",
            ),
            path_verified_source_support=_tensor(
                loaded_arrays,
                "path_verified_source_support",
            ),
            verified_source_support=_tensor(
                loaded_arrays,
                "verified_source_support",
            ),
            local_motion_verified_support=_tensor(
                loaded_arrays,
                "local_motion_verified_support",
            ),
            local_growth_verified_support=_tensor(
                loaded_arrays,
                "local_growth_verified_support",
            ),
            local_dynamics_verified_support=_tensor(
                loaded_arrays,
                "local_dynamics_verified_support",
            ),
            observation_verified_source_support=_tensor(
                loaded_arrays,
                "observation_verified_source_support",
            ),
            background_verified_source_support=_tensor(
                loaded_arrays,
                "background_verified_source_support",
            ),
            motion_disagreement_px=_tensor(
                loaded_arrays,
                "motion_disagreement_px",
            ),
            motion_disagreement_mps=_floating_scalar_tensor(
                loaded_arrays,
                "motion_disagreement_mps",
                allow_nan=True,
            ),
            growth_disagreement=_tensor(
                loaded_arrays,
                "growth_disagreement",
            ),
            maximum_growth_saturation_excess=_tensor(
                loaded_arrays,
                "maximum_growth_saturation_excess",
            ),
            posterior_velocity_uncertainty_mps=_floating_scalar_tensor(
                loaded_arrays,
                "posterior_velocity_uncertainty_mps",
                allow_nan=True,
            ),
            posterior_log_growth_uncertainty_per_step=(
                _floating_scalar_tensor(
                    loaded_arrays,
                    "posterior_log_growth_uncertainty_per_step",
                    allow_nan=True,
                )
            ),
            p1_velocity_saturation_uncertainty_mps=(
                _floating_scalar_tensor(
                    loaded_arrays,
                    "p1_velocity_saturation_uncertainty_mps",
                    allow_nan=True,
                )
            ),
            p1_log_growth_saturation_uncertainty_per_step=(
                _floating_scalar_tensor(
                    loaded_arrays,
                    "p1_log_growth_saturation_uncertainty_per_step",
                    allow_nan=True,
                )
            ),
            minimum_phase_correlation_psr=_floating_scalar_tensor(
                loaded_arrays,
                "minimum_phase_correlation_psr",
                allow_nan=True,
            ),
            tendency_pair_count=_int_scalar(
                loaded_arrays,
                "tendency_pair_count",
            ),
            motion_pair_count=_int_scalar(
                loaded_arrays,
                "motion_pair_count",
            ),
            growth_pair_count=_int_scalar(
                loaded_arrays,
                "growth_pair_count",
            ),
            motion_pair_selection=TendencyPairSelection(
                _string_scalar(loaded_arrays, "motion_pair_selection")
            ),
            growth_pair_selection=TendencyPairSelection(
                _string_scalar(loaded_arrays, "growth_pair_selection")
            ),
            motion_pair_conflict=_bool_scalar(
                loaded_arrays,
                "motion_pair_conflict",
            ),
            growth_pair_conflict=_bool_scalar(
                loaded_arrays,
                "growth_pair_conflict",
            ),
            tendency_source=TendencySource(
                _string_scalar(loaded_arrays, "tendency_source")
            ),
            dynamics_source=DynamicsSource(
                _string_scalar(loaded_arrays, "dynamics_source")
            ),
            state_path_source=TendencySource(
                _string_scalar(loaded_arrays, "state_path_source")
            ),
            state_path_mode=TendencyPairSelection(
                _string_scalar(loaded_arrays, "state_path_mode")
            ),
            state_path_pair_count=_int_scalar(
                loaded_arrays,
                "state_path_pair_count",
            ),
            state_path_minimum_psr=_float_scalar(
                loaded_arrays,
                "state_path_minimum_psr",
                allow_nan=True,
            ),
            state_path_conflict=_bool_scalar(
                loaded_arrays,
                "state_path_conflict",
            ),
            state_path_extrapolated=_bool_scalar(
                loaded_arrays,
                "state_path_extrapolated",
            ),
            state_path_age_minutes=(
                None if math.isnan(state_path_age) else state_path_age
            ),
            observation_path=observation_path,
            background_path=background_path,
            minimum_growth_overlap_support=_float_scalar(
                loaded_arrays,
                "minimum_growth_overlap_support",
                allow_nan=True,
            ),
            minimum_growth_overlap_area_km2=_float_scalar(
                loaded_arrays,
                "minimum_growth_overlap_area_km2",
                allow_nan=True,
            ),
            provenance=_string_scalar(loaded_arrays, "provenance"),
        )
        stored_observation_fraction = _float_scalar(
            loaded_arrays,
            "observation_state_support_fraction",
        )
        if (
            stored_observation_fraction
            != metadata.observation_state_support_fraction
        ):
            raise ValueError("observation state support fraction mismatch")
        if (
            metadata.background_tendency_used
            != stored_background_tendency_used
        ):
            raise ValueError("background tendency provenance mismatch")
        latest_background_present = _bool_scalar(
            loaded_arrays,
            "latest_background_present",
        )
        stored_latest_background = _tensor(
            loaded_arrays,
            "latest_background_dbz",
        )
        latest_background_text = _string_scalar(
            loaded_arrays,
            "latest_background_digest",
        )
        if latest_background_present:
            latest_background_digest = _validate_digest(
                "latest_background_digest",
                latest_background_text,
            )
        else:
            if latest_background_text:
                raise ValueError(
                    "absent latest background must have an empty digest"
                )
            if stored_latest_background.shape != (0,) or (
                not stored_latest_background.is_floating_point()
            ):
                raise ValueError(
                    "absent latest background must use an empty float array"
                )
            latest_background_digest = None

        (
            analysis_config_json,
            analysis_config_digest,
            analysis_input_digest,
        ) = _analysis_lineage(loaded_arrays)
        (
            operational_calibration_manifest_json,
            operational_calibration_manifest_digest,
            operational_calibration_approval_digest,
        ) = _operational_calibration_manifest_lineage(loaded_arrays)
        (
            operational_data_identity_json,
            operational_data_identity_digest,
        ) = _operational_data_identity_lineage(loaded_arrays)
        latest_observation_mask_digest = _digest_scalar(
            loaded_arrays,
            "latest_observation_mask_digest",
        )
        run_background_age = _float_scalar(
            loaded_arrays,
            "run_background_age_minutes",
            allow_nan=True,
        )
        grid_time_contract, grid_time_contract_digest = (
            _grid_time_contract(loaded_arrays)
        )
        run = ForecastRunContract(
            config=config,
            _latest_frame_dbz=_tensor(loaded_arrays, "latest_frame_dbz"),
            _latest_observation_mask=_tensor(
                loaded_arrays,
                "latest_observation_mask",
            ),
            _latest_background_dbz=(
                stored_latest_background
                if latest_background_present
                else None
            ),
            latest_observation_mask_digest=(
                latest_observation_mask_digest
            ),
            latest_frame_digest=_digest_scalar(
                loaded_arrays,
                "latest_frame_digest",
            ),
            latest_background_digest=latest_background_digest,
            input_frames_digest=(
                _digest_scalar(loaded_arrays, "input_frames_digest")
                if "input_frames_digest" in loaded_arrays
                else _digest_scalar(loaded_arrays, "latest_frame_digest")
            ),
            observation_masks_digest=(
                _digest_scalar(loaded_arrays, "observation_masks_digest")
                if "observation_masks_digest" in loaded_arrays
                else None
            ),
            observation_quality_weight_digest=(
                _digest_scalar(
                    loaded_arrays,
                    "observation_quality_weight_digest",
                )
                if "observation_quality_weight_digest" in loaded_arrays
                else None
            ),
            observation_std_dbz_digest=(
                _digest_scalar(loaded_arrays, "observation_std_dbz_digest")
                if "observation_std_dbz_digest" in loaded_arrays
                else None
            ),
            background_frames_digest=(
                _optional_digest_scalar(
                    loaded_arrays,
                    "background_frames_digest",
                )
                if "background_frames_digest" in loaded_arrays
                else None
            ),
            fixed_input_context_digest=(
                _digest_scalar(loaded_arrays, "fixed_input_context_digest")
                if "fixed_input_context_digest" in loaded_arrays
                else None
            ),
            full_analysis_input_digest=(
                _digest_scalar(loaded_arrays, "full_analysis_input_digest")
                if "full_analysis_input_digest" in loaded_arrays
                else None
            ),
            input_bundle_digest=_digest_scalar(
                loaded_arrays,
                "input_bundle_digest",
            ),
            background_age_minutes=(
                None
                if math.isnan(run_background_age)
                else run_background_age
            ),
            grid_time_contract=grid_time_contract,
            grid_time_contract_digest=grid_time_contract_digest,
            analysis_config_json=analysis_config_json,
            analysis_config_digest=analysis_config_digest,
            analysis_input_digest=analysis_input_digest,
            operational_calibration_manifest_json=(
                operational_calibration_manifest_json
            ),
            operational_calibration_manifest_digest=(
                operational_calibration_manifest_digest
            ),
            operational_calibration_approval_digest=(
                operational_calibration_approval_digest
            ),
            operational_data_identity_json=operational_data_identity_json,
            operational_data_identity_digest=operational_data_identity_digest,
            neural_prior_digest=_optional_digest_scalar(
                loaded_arrays, "neural_prior_digest"
            ) if "neural_prior_digest" in loaded_arrays else None,
            prior_application_digest=_optional_digest_scalar(
                loaded_arrays, "prior_application_digest"
            ) if "prior_application_digest" in loaded_arrays else None,
            prior_model_contract_digest=_optional_digest_scalar(
                loaded_arrays, "prior_model_contract_digest"
            ) if "prior_model_contract_digest" in loaded_arrays else None,
            prior_feature_schema_digest=_optional_digest_scalar(
                loaded_arrays, "prior_feature_schema_digest"
            ) if "prior_feature_schema_digest" in loaded_arrays else None,
            prior_training_manifest_digest=_optional_digest_scalar(
                loaded_arrays, "prior_training_manifest_digest"
            ) if "prior_training_manifest_digest" in loaded_arrays else None,
            prior_inference_evidence_digest=_optional_digest_scalar(
                loaded_arrays, "prior_inference_evidence_digest"
            ) if "prior_inference_evidence_digest" in loaded_arrays else None,
            prior_inference_algorithm_digest=_optional_digest_scalar(
                loaded_arrays, "prior_inference_algorithm_digest"
            ) if "prior_inference_algorithm_digest" in loaded_arrays else None,
            prior_numerical_runtime_digest=_optional_digest_scalar(
                loaded_arrays, "prior_numerical_runtime_digest"
            ) if "prior_numerical_runtime_digest" in loaded_arrays else None,
            prior_dependency=(
                (_string_scalar(loaded_arrays, "prior_dependency") or None)
                if "prior_dependency" in loaded_arrays
                else None
            ),
            prior_role=(
                (_string_scalar(loaded_arrays, "prior_role") or None)
                if "prior_role" in loaded_arrays
                else None
            ),
            prior_lineage_contract=(
                "neural-prior-run-lineage-v1-audit"
                if version == "forecast-run-v44"
                and "neural_prior_digest" in loaded_arrays
                and bool(_string_scalar(loaded_arrays, "neural_prior_digest"))
                and "prior_inference_evidence_digest" not in loaded_arrays
                else "neural-prior-run-lineage-v2"
            ),
            input_plan_digest=(
                _optional_digest_scalar(loaded_arrays, "input_plan_digest")
                if "input_plan_digest" in loaded_arrays
                else None
            ),
            input_plan_json=(
                (_string_scalar(loaded_arrays, "input_plan_json") or None)
                if "input_plan_json" in loaded_arrays
                else (
                    None
                    if "input_plan_digest" not in loaded_arrays
                    or not _string_scalar(loaded_arrays, "input_plan_digest")
                    else json.dumps(
                        {
                            "contract": "legacy-opaque-input-plan-v1",
                            "legacy_digest": _string_scalar(
                                loaded_arrays, "input_plan_digest"
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            ),
            input_plan_resolution_digest=(
                _optional_digest_scalar(
                    loaded_arrays, "input_plan_resolution_digest"
                )
                if "input_plan_resolution_digest" in loaded_arrays
                else (
                    None
                    if "input_plan_digest" not in loaded_arrays
                    or not _string_scalar(loaded_arrays, "input_plan_digest")
                    else json_digest(
                        {
                            "contract": "forecast-input-plan-resolution-v1",
                            "input_plan_digest": _string_scalar(
                                loaded_arrays, "input_plan_digest"
                            ),
                            "input_bundle_digest": _digest_scalar(
                                loaded_arrays, "input_bundle_digest"
                            ),
                        }
                    )
                )
            ),
            forecast_integrator_version=_string_scalar(
                loaded_arrays,
                "forecast_integrator_version",
            ),
        )
        stored_runtime_profile_digest = _string_scalar(
            loaded_arrays,
            "operational_runtime_profile_digest",
        )
        expected_runtime_profile_digest = (
            run.operational_runtime_profile_digest
        )
        if stored_runtime_profile_digest != (
            ""
            if expected_runtime_profile_digest is None
            else expected_runtime_profile_digest
        ):
            raise ValueError("operational runtime profile digest mismatch")
        stored_state_digest = _digest_scalar(
            loaded_arrays,
            "state_metadata_digest",
        )
        result = ForecastResult(
            forecast_dbz=forecast_dbz,
            valid_mask=valid_mask,
            forecast_dbz_digest=_digest_scalar(
                loaded_arrays,
                "forecast_dbz_digest",
            ),
            valid_mask_digest=_digest_scalar(
                loaded_arrays,
                "valid_mask_digest",
            ),
            state=state,
            metadata=metadata,
            run=run,
            state_metadata_digest=stored_state_digest,
            forecast_run_digest=_digest_scalar(
                loaded_arrays,
                "forecast_run_digest",
            ),
            evidence=forecast_evidence_fields(state, metadata, config),
            audit=None,
        )
        if version in _LEGACY_FORECAST_RUN_ARTIFACT_VERSIONS:
            migrated = forecast_from_state(state, metadata, config, run=run)
            if not torch.equal(migrated.forecast_dbz, forecast_dbz) or not torch.equal(
                migrated.valid_mask, valid_mask
            ):
                raise ValueError("legacy forecast payload cannot be migrated")
            result = migrated
    result.validate_issuance()
    if not torch.equal(
        result.forecast_path_verified_support,
        stored_forecast_path_verified_support,
    ):
        raise ValueError("forecast path verified support mismatch")
    if not torch.equal(
        result.forecast_verified_support,
        stored_forecast_verified_support,
    ):
        raise ValueError("forecast verified support mismatch")
    if not torch.equal(
        result.forecast_local_motion_verified_support,
        stored_forecast_local_motion_verified_support,
    ):
        raise ValueError("forecast local motion verified support mismatch")
    if not torch.equal(
        result.forecast_local_growth_verified_support,
        stored_forecast_local_growth_verified_support,
    ):
        raise ValueError("forecast local growth verified support mismatch")
    if not torch.equal(
        result.forecast_local_dynamics_verified_support,
        stored_forecast_local_dynamics_verified_support,
    ):
        raise ValueError("forecast local dynamics verified support mismatch")
    if not torch.equal(
        result.forecast_observation_verified_support,
        stored_forecast_observation_verified_support,
    ):
        raise ValueError("forecast observation verified support mismatch")
    if not torch.equal(
        result.forecast_background_verified_support,
        stored_forecast_background_verified_support,
    ):
        raise ValueError("forecast background verified support mismatch")
    if not torch.equal(
        result.forecast_velocity_uncertainty_mps,
        stored_forecast_velocity_uncertainty_mps,
    ):
        raise ValueError("forecast velocity uncertainty mismatch")
    if not torch.equal(
        result.motion_evidence_uncertainty_multiplier,
        stored_motion_evidence_multiplier,
    ):
        raise ValueError("motion evidence multiplier mismatch")
    if not torch.equal(
        result.growth_evidence_uncertainty_multiplier,
        stored_growth_evidence_multiplier,
    ):
        raise ValueError("growth evidence multiplier mismatch")
    if not torch.equal(
        result.forecast_position_uncertainty_m,
        stored_forecast_position_uncertainty_m,
    ):
        raise ValueError("forecast position uncertainty mismatch")
    if not torch.equal(
        result.forecast_log_growth_uncertainty,
        stored_forecast_log_growth_uncertainty,
    ):
        raise ValueError("forecast log-growth uncertainty mismatch")
    if not torch.equal(
        result.forecast_confidence,
        stored_forecast_confidence,
    ):
        raise ValueError("forecast confidence mismatch")
    if not torch.equal(
        result.radar_anchored_valid_mask,
        stored_radar_anchored_valid_mask,
    ):
        raise ValueError("radar-anchored valid mask mismatch")
    if not torch.equal(
        result.radar_state_anchored_valid_mask,
        stored_radar_state_anchored_valid_mask,
    ):
        raise ValueError("radar-state-anchored valid mask mismatch")
    if not torch.equal(
        result.radar_dynamics_anchored_valid_mask,
        stored_radar_dynamics_anchored_valid_mask,
    ):
        raise ValueError("radar-dynamics-anchored valid mask mismatch")
    if not torch.equal(
        result.background_dynamics_mask,
        stored_background_dynamics_mask,
    ):
        raise ValueError("background dynamics mask mismatch")
    if not torch.equal(
        result.background_fallback_mask,
        stored_background_fallback_mask,
    ):
        raise ValueError("background fallback mask mismatch")
    _validate_displacement_mps(result, stored_displacement_mps)
    _validate_velocity(
        "grid_velocity_mps_yx",
        result.grid_velocity_mps_yx,
        stored_grid_velocity,
    )
    _validate_velocity(
        "projected_velocity_mps_xy",
        result.projected_velocity_mps_xy,
        stored_projected_velocity,
    )
    return result


def _open_preflighted_artifact(
    source: Path,
    *,
    maximum_member_count: int,
    maximum_member_bytes: int,
    maximum_total_expanded_bytes: int,
) -> BinaryIO:
    artifact = source.open("rb")
    try:
        _preflight_archive(
            artifact,
            maximum_member_count=maximum_member_count,
            maximum_member_bytes=maximum_member_bytes,
            maximum_total_expanded_bytes=maximum_total_expanded_bytes,
        )
        artifact.seek(0)
        return artifact
    except BaseException:
        artifact.close()
        raise


def _forecast_run_artifact_digest(
    arrays: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if name == "forecast_run_artifact_digest":
            continue
        value = np.asarray(arrays[name])
        if value.dtype.kind == "O":
            raise ValueError("forecast run arrays cannot use object dtype")
        metadata = json.dumps(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(metadata)
        digest.update(b"\0")
        if value.dtype.kind in {"U", "S"}:
            encoded = json.dumps(
                value.tolist(),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(encoded)
        else:
            _update_digest_with_array_bytes(digest, value)
        digest.update(b"\0")
    return digest.hexdigest()


def _update_digest_with_array_bytes(
    digest: Any,
    value: NDArray[Any],
) -> None:
    if value.flags.c_contiguous:
        digest.update(value.data.cast("B"))
        return
    buffer_elements = max(1, (1024**2) // max(1, value.dtype.itemsize))
    for index in np.ndindex(value.shape[:-1]):
        row = np.asarray(value[index]).reshape(-1)
        for start in range(0, row.size, buffer_elements):
            chunk = np.ascontiguousarray(
                row[start : start + buffer_elements]
            )
            digest.update(chunk.data.cast("B"))


def seal_forecast_run_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(arrays)
    sealed["forecast_run_artifact_digest"] = np.asarray(
        _forecast_run_artifact_digest(sealed)
    )
    return sealed


def _grid_time_contract(
    arrays: _ArtifactArrays,
) -> tuple[RadarGridTimeContract | None, str | None]:
    present = _bool_scalar(arrays, "grid_time_contract_present")
    contract_json = _string_scalar(arrays, "grid_time_contract_json")
    digest_text = _string_scalar(arrays, "grid_time_contract_digest")
    if not present:
        if contract_json or digest_text:
            raise ValueError(
                "absent grid/time contract must have empty metadata"
            )
        return None, None
    try:
        value = json.loads(contract_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid grid_time_contract_json") from error
    if not isinstance(value, dict):
        raise ValueError("grid_time_contract_json must contain an object")
    expected = {
        "valid_times",
        "dx_m",
        "dy_m",
        "projection",
        "grid_hash",
        "background_valid_times",
        "pixel_to_projected_matrix_m",
    }
    if set(value) != expected:
        raise ValueError("grid_time_contract_json has invalid fields")
    background_times = value["background_valid_times"]
    matrix = value["pixel_to_projected_matrix_m"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
    ):
        raise ValueError("invalid grid_time_contract_json")
    try:
        contract = RadarGridTimeContract(
            valid_times=tuple(value["valid_times"]),
            dx_m=value["dx_m"],
            dy_m=value["dy_m"],
            projection=value["projection"],
            grid_hash=value["grid_hash"],
            pixel_to_projected_matrix_m=(
                (float(matrix[0][0]), float(matrix[0][1])),
                (float(matrix[1][0]), float(matrix[1][1])),
            ),
            background_valid_times=(
                None
                if background_times is None
                else tuple(background_times)
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid grid_time_contract_json") from error
    digest = _validate_digest("grid_time_contract_digest", digest_text)
    if contract.digest != digest:
        raise ValueError("grid/time contract digest mismatch")
    return contract, digest


def _validate_displacement_mps(
    result: ForecastResult,
    stored: Tensor,
) -> None:
    if stored.shape != (2,) or not stored.is_floating_point():
        raise ValueError("displacement_mps_yx must be a floating [2] vector")
    expected = result.displacement_mps_yx
    if expected is None:
        if not bool(torch.all(torch.isnan(stored))):
            raise ValueError(
                "displacement_mps_yx requires a grid/time contract"
            )
        return
    if not torch.allclose(
        stored.to(dtype=expected.dtype),
        expected,
        rtol=1.0e-6,
        atol=1.0e-9,
    ):
        raise ValueError("displacement_mps_yx disagrees with the run contract")


def _validate_velocity(
    name: str,
    expected: Tensor | None,
    stored: Tensor,
) -> None:
    if stored.shape != (2,) or not stored.is_floating_point():
        raise ValueError(f"{name} must be a floating [2] vector")
    if expected is None:
        if not bool(torch.all(torch.isnan(stored))):
            raise ValueError(f"{name} requires a grid/time contract")
        return
    if not torch.allclose(
        stored.to(dtype=expected.dtype),
        expected,
        rtol=1.0e-6,
        atol=1.0e-9,
    ):
        raise ValueError(f"{name} disagrees with the run contract")


def _numpy(value: Tensor) -> NDArray[Any]:
    return value.detach().contiguous().cpu().numpy()


def _tensor(arrays: _ArtifactArrays, name: str) -> Tensor:
    return torch.from_numpy(_array(arrays, name))


def _array(arrays: _ArtifactArrays, name: str) -> NDArray[Any]:
    if name not in arrays:
        raise ValueError(f"forecast run artifact is missing {name}")
    return arrays[name]


def _string_scalar(arrays: _ArtifactArrays, name: str) -> str:
    value = _array(arrays, name)
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a string scalar")
    return str(value.item())


def _digest_scalar(arrays: _ArtifactArrays, name: str) -> str:
    return _validate_digest(name, _string_scalar(arrays, name))


def _optional_digest_scalar(
    arrays: _ArtifactArrays,
    name: str,
) -> str | None:
    value = _string_scalar(arrays, name)
    return None if not value else _validate_digest(name, value)


def _analysis_lineage(
    arrays: _ArtifactArrays,
) -> tuple[str | None, str | None, str | None]:
    present = _bool_scalar(arrays, "analysis_config_present")
    config_json = _string_scalar(arrays, "analysis_config_json")
    config_digest = _string_scalar(arrays, "analysis_config_digest")
    input_digest = _string_scalar(arrays, "analysis_input_digest")
    if not present:
        if config_json or config_digest or input_digest:
            raise ValueError("absent analysis config must have empty lineage")
        return None, None, None
    try:
        config_value = json.loads(config_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid analysis_config_json") from error
    validated_config_digest = _validate_digest(
        "analysis_config_digest",
        config_digest,
    )
    if json_digest(config_value) != validated_config_digest:
        raise ValueError("analysis config digest mismatch")
    return (
        config_json,
        validated_config_digest,
        _validate_digest("analysis_input_digest", input_digest),
    )


def _operational_calibration_manifest_lineage(
    arrays: _ArtifactArrays,
) -> tuple[str | None, str | None, str | None]:
    present = _bool_scalar(
        arrays,
        "operational_calibration_manifest_present",
    )
    manifest_json = _string_scalar(
        arrays,
        "operational_calibration_manifest_json",
    )
    manifest_digest = _string_scalar(
        arrays,
        "operational_calibration_manifest_digest",
    )
    approval_digest = _string_scalar(
        arrays,
        "operational_calibration_approval_digest",
    )
    if not present:
        if manifest_json or manifest_digest or approval_digest:
            raise ValueError(
                "absent calibration manifest must have empty lineage"
            )
        return None, None, None
    validated_digest = _validate_digest(
        "operational_calibration_manifest_digest",
        manifest_digest,
    )
    try:
        value = json.loads(manifest_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid calibration manifest JSON") from error
    if json_digest(value) != validated_digest:
        raise ValueError("operational calibration manifest digest mismatch")
    validated_approval = _validate_digest(
        "operational_calibration_approval_digest",
        approval_digest,
    )
    if validated_approval != validated_digest:
        raise ValueError("operational calibration manifest is not approved")
    return manifest_json, validated_digest, validated_approval


def _operational_data_identity_lineage(
    arrays: _ArtifactArrays,
) -> tuple[str | None, str | None]:
    present = _bool_scalar(arrays, "operational_data_identity_present")
    identity_json = _string_scalar(
        arrays,
        "operational_data_identity_json",
    )
    identity_digest = _string_scalar(
        arrays,
        "operational_data_identity_digest",
    )
    if not present:
        if identity_json or identity_digest:
            raise ValueError(
                "absent operational data identity must have empty lineage"
            )
        return None, None
    try:
        value = json.loads(identity_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid operational data identity JSON") from error
    validated_digest = _validate_digest(
        "operational_data_identity_digest",
        identity_digest,
    )
    if json_digest(value) != validated_digest:
        raise ValueError("operational data identity digest mismatch")
    return identity_json, validated_digest


def _validate_digest(name: str, value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _bool_scalar(arrays: _ArtifactArrays, name: str) -> bool:
    value = _array(arrays, name)
    if value.shape != () or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(value.item())


def _int_scalar(arrays: _ArtifactArrays, name: str) -> int:
    value = _array(arrays, name)
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be an integer scalar")
    return int(value.item())


def _float_scalar(
    arrays: _ArtifactArrays,
    name: str,
    *,
    allow_nan: bool = False,
) -> float:
    value = _array(arrays, name)
    if value.shape != () or value.dtype.kind != "f":
        raise ValueError(f"{name} must be a floating-point scalar")
    result = float(value.item())
    if math.isfinite(result) or (allow_nan and math.isnan(result)):
        return result
    raise ValueError(f"{name} must be finite")


def _path_provenance_from_arrays(
    arrays: _ArtifactArrays,
    prefix: str,
) -> StatePathProvenance:
    age = _float_scalar(
        arrays,
        f"{prefix}_age_minutes",
        allow_nan=True,
    )
    return StatePathProvenance(
        mode=TendencyPairSelection(
            _string_scalar(arrays, f"{prefix}_mode")
        ),
        pair_count=_int_scalar(arrays, f"{prefix}_pair_count"),
        minimum_psr=_float_scalar(
            arrays,
            f"{prefix}_minimum_psr",
            allow_nan=True,
        ),
        conflict=_bool_scalar(arrays, f"{prefix}_conflict"),
        extrapolated=_bool_scalar(arrays, f"{prefix}_extrapolated"),
        age_minutes=None if math.isnan(age) else age,
    )


def _floating_scalar_tensor(
    arrays: _ArtifactArrays,
    name: str,
    *,
    allow_nan: bool = False,
) -> Tensor:
    value = _array(arrays, name)
    if value.shape != () or value.dtype.kind != "f":
        raise ValueError(f"{name} must be a floating-point scalar")
    result = _tensor(arrays, name)
    scalar = float(result.item())
    if math.isfinite(scalar) or (allow_nan and math.isnan(scalar)):
        return result
    qualifier = "finite or NaN" if allow_nan else "finite"
    raise ValueError(f"{name} must be {qualifier}")
