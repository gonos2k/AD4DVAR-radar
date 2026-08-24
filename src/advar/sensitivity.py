"""Forecast sensitivities that can be stored as conditional experience."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import IntEnum
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ._digest import dataclass_digest, json_digest, tensor_digest
from .calibration import OperationalCalibrationManifest, OperationalDataIdentity
from .matrix_free import pcg
from .nowcast import (
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    DataStatus,
    DynamicsSource,
    ForecastMetadata,
    ForecastResult,
    ForecastRunContract,
    NowcastConfig,
    RADAR_GRID_AFFINE_ABSOLUTE_TOLERANCE_M,
    RADAR_GRID_AFFINE_RELATIVE_TOLERANCE,
    RADAR_GRID_MAXIMUM_AFFINE_CONDITION_NUMBER,
    RADAR_GRID_MINIMUM_AXIS_SINE,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    RADAR_SCIENTIFIC_MAXIMUM_AXIS_SPACING_M,
    RADAR_SCIENTIFIC_MINIMUM_AXIS_SPACING_M,
    RadarGridTimeContract,
    RadarSpatialGridIdentity,
    RadarState,
    TendencyPairSelection,
    TendencySource,
    _estimate_available_pair,
    _estimate_source_tendencies,
    _forecast_linear_at_step_core,
    _phase_correlation_details,
    forecast_from_state,
    forecast_linear_at_step,
    motion_displacement_limits_yx,
)
from .physics import RemapCell, dbz_to_echo, echo_to_dbz, freeze_remap_cell
from .variational import (
    P1_LINEARIZATION_CONTRACT,
    AnalysisFeasibilityMargins,
    AnalysisLinearization,
    AnalysisObservations,
    AnalysisResult,
    BoundNeuralPriorInput,
    FrozenOuterState,
    NeuralPriorApplication,
    NeuralPriorInferenceRunner,
    P1LinearizationState,
    _apply_observation_error_whitener,
    _analysis_trajectory,
    _analysis_input_lineage,
    _count_observation_whitener_applies,
    _linearization_stationarity,
    _relative_irls_weight_change,
    _observation_whitener_operations_per_apply,
    _robust_stationarity,
    _stationarity_is_acceptable,
    freeze_irls_weights,
    prepare_analysis,
    residual_vector,
    solve_analysis,
    validate_analysis_linearization_content,
    variational_nowcast,
)


SUPPORTED_METRICS = (
    "log_echo_mse",
    "soft_fss_error_35",
    "centroid_error",
    "centroid_error_m2",
)
DEFAULT_METRICS = (
    "log_echo_mse",
    "soft_fss_error_35",
    "centroid_error",
)

FSOMetricDomain = Literal[
    "issued",
    "radar_dynamics_anchored",
    "confidence_weighted",
]
PerturbationSemantics = Literal[
    "augmented_parameter",
    "physical_radar_value",
]
BaselineDynamicsBranchStatus = Literal[
    "not_applicable",
    "unknown",
    "certified",
    "invalid",
]
CandidateRankingObjective = Literal[
    "absolute_influence",
    "expected_error_reduction",
    "two_sided_diagnostic",
]
LearningSelectionMode = Literal["direct", "ranked_top_k"]
FirstOrderMetricDomain = Literal[
    "frozen_metric_domain",
    "resolved_issuance_domain",
]
TileShape = tuple[int, int]
LEARNING_POLICY_TRUST_STORE_CONTRACT = "advar-learning-policy-trust-store-v1"
MAXIMUM_LEARNING_POLICY_TRUST_STORE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _LearningPolicyTrustStore:
    approved_policy_digests: frozenset[str]
    content_digest: str


def _canonical_verification_time(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("verification valid times must be ISO-8601 strings")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            "verification valid times must be ISO-8601 strings"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("verification valid times must include timezones")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _load_learning_policy_trust_store(
    path: str | Path,
) -> _LearningPolicyTrustStore:
    """Read approved policy digests from a root-owned immutable JSON file."""

    trust_store = Path(path)
    if not trust_store.is_absolute():
        raise ValueError("learning policy trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(trust_store, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("learning policy trust store must be a file")
        if metadata.st_uid != 0 or metadata.st_mode & 0o022:
            raise ValueError(
                "learning policy trust store must be root-owned and not "
                "group/world-writable"
            )
        if metadata.st_size > MAXIMUM_LEARNING_POLICY_TRUST_STORE_BYTES:
            raise ValueError("learning policy trust store is too large")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            content = stream.read(MAXIMUM_LEARNING_POLICY_TRUST_STORE_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(content) > MAXIMUM_LEARNING_POLICY_TRUST_STORE_BYTES:
        raise ValueError("learning policy trust store is too large")
    document = json.loads(content)
    if not isinstance(document, dict) or set(document) != {
        "contract",
        "approved_policy_digests",
    }:
        raise ValueError("invalid learning policy trust store")
    if document["contract"] != LEARNING_POLICY_TRUST_STORE_CONTRACT:
        raise ValueError("unsupported learning policy trust store")
    raw_digests = document["approved_policy_digests"]
    if not isinstance(raw_digests, list) or any(
        not isinstance(digest, str) for digest in raw_digests
    ):
        raise ValueError("approved policy digests must be a list")
    digests = frozenset(raw_digests)
    if len(digests) != len(raw_digests):
        raise ValueError("approved policy digests must be unique")
    for digest in digests:
        _require_sha256("approved_policy_digest", digest)
    return _LearningPolicyTrustStore(
        approved_policy_digests=digests,
        content_digest=json_digest(
            {
                "contract": LEARNING_POLICY_TRUST_STORE_CONTRACT,
                "approved_policy_digests": sorted(digests),
            }
        ),
    )


OBSERVATION_ERROR_DERIVATION_ALGORITHM_CONTRACT = (
    "verification-observation-error-deterministic-derivation-v1"
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_DIGEST = json_digest(
    {
        "contract": OBSERVATION_ERROR_DERIVATION_ALGORITHM_CONTRACT,
        "state_precedence": [
            "mosaic_source_unassigned",
            "source_missing",
            "beam_blocked",
            "qc_invalid",
            "below_detection_censored",
            "observed_echo",
            "observed_clear",
        ],
        "quality_rule": "registry-quality-on-valid-cells-v1",
        "observation_std_rule": "registry-std-on-valid-cells-v1",
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V2_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-spatial-derivation-v2",
        "state_precedence": [
            "mosaic_source_unassigned",
            "source_missing",
            "beam_blocked",
            "qc_invalid",
            "below_detection_censored",
            "observed_echo",
            "observed_clear",
        ],
        "quality_rule": (
            "registry-quality-times-attenuation-times-unblocked-times-"
            "range-reliability-v1"
        ),
        "observation_std_rule": (
            "registry-std-times-one-plus-range-elevation-blockage-"
            "attenuation-penalties-v1"
        ),
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V3_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-source-composition-v3",
        "state_precedence": [
            "mosaic_source_unassigned",
            "source_missing",
            "beam_blocked",
            "qc_invalid",
            "below_detection_censored",
            "observed_echo",
            "observed_clear",
        ],
        "observation_value_rule": (
            "axis0-gather-reflectivity-by-selected-source-index-v1"
        ),
        "detection_limit_rule": (
            "axis0-gather-detection-limit-by-selected-source-index-v1"
        ),
        "acquisition_time_rule": (
            "axis0-gather-acquisition-offset-by-selected-source-index-v1"
        ),
        "quality_rule": (
            "selected-source-registry-quality-times-attenuation-times-"
            "unblocked-times-range-reliability-v1"
        ),
        "observation_std_rule": (
            "selected-source-registry-std-times-one-plus-range-elevation-"
            "blockage-attenuation-penalties-v1"
        ),
    }
)
OBSERVATION_TEMPORAL_QUALITY_DECAY_ALGORITHM_V1_DIGEST = json_digest(
    {
        "contract": "verification-observation-temporal-quality-decay-v1",
        "rule": "exp-negative-age-over-scale-to-power-v1",
    }
)
OBSERVATION_TEMPORAL_ERROR_ALGORITHM_V1_DIGEST = json_digest(
    {
        "contract": "verification-observation-temporal-error-growth-v1",
        "rule": "quadrature-spatial-and-linear-age-error-v1",
    }
)
OBSERVATION_DETECTION_LIMIT_ALGORITHM_V1_DIGEST = json_digest(
    {
        "contract": "verification-detection-limit-derivation-v1",
        "rule": "ordered-radar-registry-constant-limit-v1",
    }
)
OBSERVATION_CENSOR_STATE_ALGORITHM_V1_DIGEST = json_digest(
    {
        "contract": "verification-censor-state-derivation-v1",
        "rule": "selected-reflectivity-at-or-below-registered-limit-v1",
    }
)
OBSERVATION_DETECTION_LIMIT_ALGORITHM_V2_DIGEST = json_digest(
    {
        "contract": "verification-detection-limit-derivation-v2",
        "rule": "ordered-radar-range-quadratic-elevation-excess-v1",
        "attenuation_double_penalty": "excluded-v1",
    }
)
OBSERVATION_REPORT_KIND_ALGORITHM_V1_DIGEST = json_digest(
    {
        "contract": "verification-observation-report-kind-v1",
        "native_states": [
            "detected_echo",
            "confirmed_clear",
            "below_detection_censored",
        ],
        "rule": "source-signed-category-product-validated-v1",
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V4_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-temporal-composition-v4",
        "parent_algorithm_digest": OBSERVATION_ERROR_DERIVATION_ALGORITHM_V3_DIGEST,
        "temporal_quality_decay_algorithm_digest": (
            OBSERVATION_TEMPORAL_QUALITY_DECAY_ALGORITHM_V1_DIGEST
        ),
        "temporal_error_algorithm_digest": (
            OBSERVATION_TEMPORAL_ERROR_ALGORITHM_V1_DIGEST
        ),
    }
)

LEGACY_OBSERVATION_MASK_DERIVATION_ALGORITHM_CONTRACT_V1 = (
    "verification-observation-mask-deterministic-derivation-v1"
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V1_DIGEST = json_digest(
    {
        "contract": LEGACY_OBSERVATION_MASK_DERIVATION_ALGORITHM_CONTRACT_V1,
        "source_assignment_rule": "highest-positive-score-first-index-v1",
        "source_presence_rule": "assigned-source-availability-v1",
        "range_elevation_rule": "inclusive-plan-bounds-v1",
        "beam_blockage_rule": "fraction-above-plan-maximum-v1",
        "attenuation_qc_rule": "score-at-least-plan-minimum-v1",
        "censoring_rule": "reported-and-at-or-below-detection-limit-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_CONTRACT = (
    "verification-observation-mask-source-composition-v2"
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V2_DIGEST = json_digest(
    {
        "contract": OBSERVATION_MASK_DERIVATION_ALGORITHM_CONTRACT,
        "source_assignment_rule": "highest-positive-score-first-index-v1",
        "source_presence_rule": "assigned-source-availability-v1",
        "source_registry_order_binding": "exact-ordered-source-digests-v1",
        "selected_source_spatial_gather": (
            "axis0-gather-by-selected-source-index-v1"
        ),
        "selected_source_observation_gather": (
            "reflectivity-detection-censor-time-axis0-gather-v1"
        ),
        "range_elevation_rule": "inclusive-plan-bounds-v1",
        "beam_blockage_rule": "fraction-above-plan-maximum-v1",
        "attenuation_qc_rule": "score-at-least-plan-minimum-v1",
        "censoring_rule": "selected-source-reported-at-selected-limit-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V3_DIGEST = json_digest(
    {
        "contract": "verification-observation-mask-preregistered-source-v3",
        "parent_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V2_DIGEST,
        "maximum_acquisition_age_rule": "inclusive-registered-age-bound-v1",
        "detection_limit_algorithm_digest": (
            OBSERVATION_DETECTION_LIMIT_ALGORITHM_V1_DIGEST
        ),
        "censor_state_algorithm_digest": (
            OBSERVATION_CENSOR_STATE_ALGORITHM_V1_DIGEST
        ),
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V4_DIGEST = json_digest(
    {
        "contract": "verification-observation-mask-chronology-report-kind-v4",
        "parent_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V3_DIGEST,
        "acquisition_age_rule": (
            "valid-time-minus-source-nominal-time-minus-cell-offset-v1"
        ),
        "detection_limit_algorithm_digest": (
            OBSERVATION_DETECTION_LIMIT_ALGORITHM_V2_DIGEST
        ),
        "report_kind_algorithm_digest": (
            OBSERVATION_REPORT_KIND_ALGORITHM_V1_DIGEST
        ),
        "stale_reason_rule": "separate-from-attenuation-qc-v1",
    }
)
OBSERVATION_SOURCE_SELECTION_ALGORITHM_V1_DIGEST = json_digest(
    {
        "contract": "verification-observation-source-selection-v1",
        "eligibility_rule": (
            "availability-age-range-elevation-blockage-attenuation-first-v1"
        ),
        "score_rule": (
            "one-plus-inverse-one-plus-range-km-elevation-time-unblocked-"
            "attenuation-quality-v1"
        ),
        "tie_break_rule": "first-exact-ordered-registry-source-v1",
        "fallback_rule": "highest-scoring-eligible-source-v1",
        "invalid_score": "zero-v1",
    }
)
OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V1_DIGEST = json_digest(
    {
        "contract": "verification-observation-spatial-age-gate-v1",
        "rule": "reference-speed-times-age-at-most-grid-fraction-v1",
        "metric_role": "spatial-skill-domain-only-v1",
    }
)
OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V2_DIGEST = json_digest(
    {
        "contract": "verification-observation-spatial-age-gate-v2",
        "rule": "reference-speed-times-age-at-most-l2-index-grid-fraction-v1",
        "grid_displacement_norm": "minimum-affine-singular-value-v1",
        "metric_role": "spatial-skill-domain-only-v1",
    }
)
OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V3_DIGEST = json_digest(
    {
        "contract": "verification-observation-spatial-age-gate-v3",
        "rule": "reference-speed-times-age-at-most-active-index-grid-fraction-v1",
        "two_dimensional_spacing": "minimum-affine-singular-value-v1",
        "single_row_spacing": "affine-column-norm-v1",
        "single_column_spacing": "affine-row-norm-v1",
        "single_cell_policy": "spatial-metric-unsupported-v1",
        "metric_role": "spatial-skill-domain-only-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V5_DIGEST = json_digest(
    {
        "contract": "verification-observation-mask-product-selection-v5",
        "parent_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V4_DIGEST,
        "source_selection_algorithm_digest": (
            OBSERVATION_SOURCE_SELECTION_ALGORITHM_V1_DIGEST
        ),
        "spatial_age_gate_algorithm_digest": (
            OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V1_DIGEST
        ),
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V5_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-report-kind-temporal-v5",
        "parent_algorithm_digest": OBSERVATION_ERROR_DERIVATION_ALGORITHM_V4_DIGEST,
        "mask_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V4_DIGEST,
        "state_rule": "signed-clear-censored-and-explicit-stale-v1",
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V6_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-product-selection-v6",
        "parent_algorithm_digest": OBSERVATION_ERROR_DERIVATION_ALGORITHM_V5_DIGEST,
        "mask_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V5_DIGEST,
        "source_selection_algorithm_digest": (
            OBSERVATION_SOURCE_SELECTION_ALGORITHM_V1_DIGEST
        ),
        "clear_intensity_rule": "categorical-support-only-v1",
        "threshold_partition_rule": "echo-strictly-above-censored-at-or-below-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V6_DIGEST = json_digest(
    {
        "contract": "verification-observation-mask-geometry-closure-v6",
        "parent_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V5_DIGEST,
        "geometry_contract": "radar-observation-geometry-v2",
        "grid_identity_rule": "grid-contract-in-geometry-content-address-v1",
        "grid_spacing_rule": (
            "uniform-rectilinear-isotropic-coordinate-derived-spacing-v1"
        ),
        "coordinate_frame_rule": "registry-and-grid-exact-crs-binding-v1",
        "geometry_model": "projected-horizontal-representative-tilt-v1",
        "radar_altitude_role": "provenance_only",
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V7_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-geometry-closure-v7",
        "parent_algorithm_digest": OBSERVATION_ERROR_DERIVATION_ALGORITHM_V6_DIGEST,
        "mask_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V6_DIGEST,
        "coordinate_frame_rule": "registry-and-grid-exact-crs-binding-v1",
        "geometry_model": "projected-horizontal-representative-tilt-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V7_DIGEST = json_digest(
    {
        "contract": "verification-observation-mask-shared-affine-grid-v7",
        "parent_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V6_DIGEST,
        "geometry_contract": "radar-observation-geometry-v3",
        "spatial_grid_contract": "radar-spatial-grid-identity-v2",
        "coordinate_generation_rule": (
            "origin-plus-affine-column-row-cell-center-v1"
        ),
        "coordinate_dtype": RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        "cell_center_convention": (
            RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION
        ),
        "coordinate_equality_rule": "exact-torch-equal-v1",
        "derived_spatial_field_dtype_rule": (
            "float64-geometry-then-source-reflectivity-dtype-v1"
        ),
        "grid_spacing_rule": "minimum-affine-axis-norm-v1",
        "affine_relative_tolerance": RADAR_GRID_AFFINE_RELATIVE_TOLERANCE,
        "affine_absolute_tolerance_m": (
            RADAR_GRID_AFFINE_ABSOLUTE_TOLERANCE_M
        ),
        "coordinate_frame_rule": (
            "canonical-projection-derived-crs-and-registry-exact-binding-v1"
        ),
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V8_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-shared-affine-grid-v8",
        "parent_algorithm_digest": OBSERVATION_ERROR_DERIVATION_ALGORITHM_V7_DIGEST,
        "mask_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V7_DIGEST,
        "geometry_contract": "radar-observation-geometry-v3",
        "spatial_grid_contract": "radar-spatial-grid-identity-v2",
        "grid_spacing_rule": "minimum-affine-axis-norm-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V8_DIGEST = json_digest(
    {
        "contract": "verification-observation-mask-affine-domain-v8",
        "parent_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V7_DIGEST,
        "geometry_contract": "radar-observation-geometry-v4",
        "spatial_grid_contract": "radar-spatial-grid-identity-v3",
        "projected_crs_contract": "radar-projected-crs-identity-v2",
        "coordinate_generation_rule": (
            "origin-plus-affine-column-row-cell-center-v1"
        ),
        "coordinate_dtype": RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        "cell_center_convention": (
            RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION
        ),
        "coordinate_equality_rule": "exact-torch-equal-v1",
        "scientific_source_dtype_policy": "float32-or-float64-only-v1",
        "grid_spacing_rule": "minimum-affine-singular-value-l2-index-v1",
        "minimum_axis_sine": RADAR_GRID_MINIMUM_AXIS_SINE,
        "maximum_affine_condition_number": (
            RADAR_GRID_MAXIMUM_AFFINE_CONDITION_NUMBER
        ),
        "affine_relative_tolerance": RADAR_GRID_AFFINE_RELATIVE_TOLERANCE,
        "affine_absolute_tolerance_m": (
            RADAR_GRID_AFFINE_ABSOLUTE_TOLERANCE_M
        ),
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V9_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-affine-domain-v9",
        "parent_algorithm_digest": OBSERVATION_ERROR_DERIVATION_ALGORITHM_V8_DIGEST,
        "mask_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V8_DIGEST,
        "geometry_contract": "radar-observation-geometry-v4",
        "spatial_grid_contract": "radar-spatial-grid-identity-v3",
        "grid_spacing_rule": "minimum-affine-singular-value-l2-index-v1",
        "scientific_source_dtype_policy": "float32-or-float64-only-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V9_DIGEST = json_digest(
    {
        "contract": "verification-observation-mask-metric-crs-v9",
        "parent_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V8_DIGEST,
        "geometry_contract": "radar-observation-geometry-v5",
        "spatial_grid_contract": "radar-spatial-grid-identity-v4",
        "projected_crs_contract": "radar-projected-crs-identity-v3",
        "metric_crs_policy": "epsg-5179-bounded-ground-distance-only-v1",
        "coordinate_generation_rule": (
            "origin-plus-affine-column-row-cell-center-v1"
        ),
        "coordinate_dtype": RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        "cell_center_convention": (
            RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION
        ),
        "coordinate_equality_rule": "exact-torch-equal-v1",
        "scientific_source_dtype_policy": "float32-or-float64-only-v1",
        "grid_spacing_rule": "active-axis-or-minimum-singular-value-v1",
        "affine_metric_calculation": "maximum-entry-scaled-finite-svd-v1",
        "minimum_axis_sine": RADAR_GRID_MINIMUM_AXIS_SINE,
        "maximum_affine_condition_number": (
            RADAR_GRID_MAXIMUM_AFFINE_CONDITION_NUMBER
        ),
        "affine_relative_tolerance": RADAR_GRID_AFFINE_RELATIVE_TOLERANCE,
        "affine_absolute_tolerance_m": (
            RADAR_GRID_AFFINE_ABSOLUTE_TOLERANCE_M
        ),
        "spatial_age_gate_algorithm_digest": (
            OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V3_DIGEST
        ),
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V10_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-metric-crs-v10",
        "parent_algorithm_digest": OBSERVATION_ERROR_DERIVATION_ALGORITHM_V9_DIGEST,
        "mask_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V9_DIGEST,
        "geometry_contract": "radar-observation-geometry-v5",
        "spatial_grid_contract": "radar-spatial-grid-identity-v4",
        "grid_spacing_rule": "active-axis-or-minimum-singular-value-v1",
        "scientific_source_dtype_policy": "float32-or-float64-only-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST = json_digest(
    {
        "contract": "verification-observation-mask-metric-domain-v10",
        "parent_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V9_DIGEST,
        "geometry_contract": "radar-observation-geometry-v6",
        "spatial_grid_contract": "radar-spatial-grid-identity-v5",
        "projected_crs_contract": "radar-projected-crs-identity-v4",
        "metric_domain_contract": CURRENT_RADAR_METRIC_DOMAIN.contract,
        "metric_domain_digest": CURRENT_RADAR_METRIC_DOMAIN.digest,
        "metric_domain_evidence_digest": (
            CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
        ),
        "coordinate_membership": "all-cell-centers-and-radar-sites-v1",
        "maximum_linear_scale_error": (
            CURRENT_RADAR_METRIC_DOMAIN.maximum_linear_scale_error
        ),
        "minimum_axis_spacing_m": RADAR_SCIENTIFIC_MINIMUM_AXIS_SPACING_M,
        "maximum_axis_spacing_m": RADAR_SCIENTIFIC_MAXIMUM_AXIS_SPACING_M,
        "coordinate_generation_rule": (
            "origin-plus-affine-column-row-cell-center-v1"
        ),
        "coordinate_dtype": RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        "cell_center_convention": (
            RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION
        ),
        "coordinate_equality_rule": "exact-torch-equal-v1",
        "scientific_source_dtype_policy": "float32-or-float64-only-v1",
        "grid_spacing_rule": "active-axis-or-minimum-singular-value-v1",
        "affine_metric_calculation": "maximum-entry-scaled-finite-svd-v2",
        "minimum_axis_sine": RADAR_GRID_MINIMUM_AXIS_SINE,
        "maximum_affine_condition_number": (
            RADAR_GRID_MAXIMUM_AFFINE_CONDITION_NUMBER
        ),
        "affine_relative_tolerance": RADAR_GRID_AFFINE_RELATIVE_TOLERANCE,
        "affine_absolute_tolerance_m": (
            RADAR_GRID_AFFINE_ABSOLUTE_TOLERANCE_M
        ),
        "spatial_age_gate_algorithm_digest": (
            OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V3_DIGEST
        ),
    }
)
OBSERVATION_ERROR_DERIVATION_ALGORITHM_V11_DIGEST = json_digest(
    {
        "contract": "verification-observation-error-metric-domain-v11",
        "parent_algorithm_digest": OBSERVATION_ERROR_DERIVATION_ALGORITHM_V10_DIGEST,
        "mask_algorithm_digest": OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST,
        "geometry_contract": "radar-observation-geometry-v6",
        "spatial_grid_contract": "radar-spatial-grid-identity-v5",
        "metric_domain_digest": CURRENT_RADAR_METRIC_DOMAIN.digest,
        "metric_domain_evidence_digest": (
            CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
        ),
        "grid_spacing_rule": "active-axis-or-minimum-singular-value-v1",
        "scientific_source_dtype_policy": "float32-or-float64-only-v1",
        "physical_learning_tile_policy": "orthogonal-grid-only-v1",
    }
)
OBSERVATION_MASK_DERIVATION_ALGORITHM_DIGEST = (
    OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST
)

RADAR_GEOMETRY_SPACING_ABSOLUTE_TOLERANCE_M = 1.0e-3


def _derived_uniform_rectilinear_grid_spacing_m(
    grid_x_m: Tensor,
    grid_y_m: Tensor,
) -> float:
    """Derive one isotropic spacing from a uniform projected grid."""

    step_vectors: list[Tensor] = []
    if grid_x_m.shape[1] > 1:
        horizontal = torch.stack(
            (
                torch.diff(grid_x_m, dim=1),
                torch.diff(grid_y_m, dim=1),
            ),
            dim=-1,
        )
        reference = horizontal[0, 0]
        if not torch.allclose(
            horizontal,
            reference.expand_as(horizontal),
            rtol=0.0,
            atol=RADAR_GEOMETRY_SPACING_ABSOLUTE_TOLERANCE_M,
        ):
            raise ValueError("radar observation grid is not uniformly rectilinear")
        step_vectors.append(reference)
    if grid_x_m.shape[0] > 1:
        vertical = torch.stack(
            (
                torch.diff(grid_x_m, dim=0),
                torch.diff(grid_y_m, dim=0),
            ),
            dim=-1,
        )
        reference = vertical[0, 0]
        if not torch.allclose(
            vertical,
            reference.expand_as(vertical),
            rtol=0.0,
            atol=RADAR_GEOMETRY_SPACING_ABSOLUTE_TOLERANCE_M,
        ):
            raise ValueError("radar observation grid is not uniformly rectilinear")
        step_vectors.append(reference)
    if not step_vectors:
        raise ValueError("radar observation grid needs a neighboring cell")
    spacings = tuple(float(torch.linalg.vector_norm(step)) for step in step_vectors)
    if any(
        not math.isfinite(spacing)
        or spacing <= RADAR_GEOMETRY_SPACING_ABSOLUTE_TOLERANCE_M
        for spacing in spacings
    ):
        raise ValueError("radar observation grid spacing is invalid")
    if len(step_vectors) == 2:
        unit_dot = float(
            torch.dot(step_vectors[0], step_vectors[1])
            / (spacings[0] * spacings[1])
        )
        if abs(unit_dot) > 1.0e-6 or not math.isclose(
            spacings[0],
            spacings[1],
            rel_tol=0.0,
            abs_tol=RADAR_GEOMETRY_SPACING_ABSOLUTE_TOLERANCE_M,
        ):
            raise ValueError(
                "radar observation grid must be orthogonal and isotropic"
            )
    return spacings[0]


@dataclass(frozen=True)
class VerificationObservationErrorPlan:
    """Pre-registered policy for deriving verification observation errors."""

    radar_source_kind: Literal["single_site", "mosaic"]
    source_registry_digest: str
    calibration_registry_digest: str
    range_elevation_validity_algorithm_digest: str
    beam_blockage_algorithm_digest: str
    attenuation_qc_digest: str
    censoring_rule_digest: str
    spatial_correlation_block_algorithm_digest: str
    quality_weight_interpretation_digest: str
    quality_weight_algorithm_digest: str
    observation_std_algorithm_digest: str
    observation_error_model_digest: str
    source_assignment_algorithm_digest: str
    minimum_detectable_echo_dbz: float
    observation_error_reference_std_dbz: float
    derivation_algorithm_digest: str | None = None
    mask_derivation_algorithm_digest: str | None = None
    maximum_range_km: float | None = None
    minimum_elevation_deg: float | None = None
    maximum_elevation_deg: float | None = None
    maximum_beam_blockage_fraction: float | None = None
    minimum_attenuation_qc_score: float | None = None
    verification_source_authority_id: str | None = None
    verification_source_authority_public_key_hex: str | None = None
    maximum_acquisition_age_seconds: float | None = None
    temporal_quality_decay_scale_seconds: float | None = None
    temporal_quality_decay_power: float | None = None
    temporal_error_growth_dbz_per_second: float | None = None
    temporal_quality_decay_algorithm_digest: str | None = None
    temporal_error_algorithm_digest: str | None = None
    detection_limit_derivation_algorithm_digest: str | None = None
    censor_state_derivation_algorithm_digest: str | None = None
    geometry_contract_digest: str | None = None
    acquisition_timestamp_reference: Literal["volume_end"] | None = None
    spatial_metric_reference_speed_mps: float | None = None
    spatial_metric_maximum_displacement_fraction_cells: float | None = None
    spatial_age_gate_algorithm_digest: str | None = None
    spatial_correlation_role: Literal["diagnostic_only"] = "diagnostic_only"
    contract: str = "verification-observation-error-plan-v1"
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract not in {
                "verification-observation-error-plan-v1",
                "verification-observation-error-plan-v2",
                "verification-observation-error-plan-v3",
                "verification-observation-error-plan-v4",
                "verification-observation-error-plan-v5",
                "verification-observation-error-plan-v6",
                "verification-observation-error-plan-v7",
                "verification-observation-error-plan-v8",
                "verification-observation-error-plan-v9",
                "verification-observation-error-plan-v11",
                "verification-observation-error-plan-v12",
            }
            or self.radar_source_kind not in {"single_site", "mosaic"}
            or not math.isfinite(self.minimum_detectable_echo_dbz)
            or not math.isfinite(self.observation_error_reference_std_dbz)
            or self.observation_error_reference_std_dbz <= 0.0
            or self.spatial_correlation_role != "diagnostic_only"
        ):
            raise ValueError("verification observation-error plan is invalid")
        if self.contract == "verification-observation-error-plan-v1":
            if any(
                value is not None
                for value in (
                    self.derivation_algorithm_digest,
                    self.mask_derivation_algorithm_digest,
                    self.maximum_range_km,
                    self.minimum_elevation_deg,
                    self.maximum_elevation_deg,
                    self.maximum_beam_blockage_fraction,
                    self.minimum_attenuation_qc_score,
                    self.verification_source_authority_id,
                    self.verification_source_authority_public_key_hex,
                    self.maximum_acquisition_age_seconds,
                    self.temporal_quality_decay_scale_seconds,
                    self.temporal_quality_decay_power,
                    self.temporal_error_growth_dbz_per_second,
                    self.temporal_quality_decay_algorithm_digest,
                    self.temporal_error_algorithm_digest,
                    self.detection_limit_derivation_algorithm_digest,
                    self.censor_state_derivation_algorithm_digest,
                    self.geometry_contract_digest,
                    self.acquisition_timestamp_reference,
                    self.spatial_metric_reference_speed_mps,
                    self.spatial_metric_maximum_displacement_fraction_cells,
                    self.spatial_age_gate_algorithm_digest,
                )
            ):
                raise ValueError(
                    "legacy observation-error plans cannot claim replay"
                )
        elif self.contract == "verification-observation-error-plan-v2":
            if (
                self.derivation_algorithm_digest
                != OBSERVATION_ERROR_DERIVATION_ALGORITHM_DIGEST
                or self.mask_derivation_algorithm_digest is not None
                or any(
                    value is not None
                    for value in (
                        self.maximum_range_km,
                        self.minimum_elevation_deg,
                        self.maximum_elevation_deg,
                        self.maximum_beam_blockage_fraction,
                        self.minimum_attenuation_qc_score,
                        self.verification_source_authority_id,
                        self.verification_source_authority_public_key_hex,
                        self.maximum_acquisition_age_seconds,
                        self.temporal_quality_decay_scale_seconds,
                        self.temporal_quality_decay_power,
                        self.temporal_error_growth_dbz_per_second,
                        self.temporal_quality_decay_algorithm_digest,
                        self.temporal_error_algorithm_digest,
                        self.detection_limit_derivation_algorithm_digest,
                        self.censor_state_derivation_algorithm_digest,
                        self.geometry_contract_digest,
                        self.acquisition_timestamp_reference,
                        self.spatial_metric_reference_speed_mps,
                        self.spatial_metric_maximum_displacement_fraction_cells,
                        self.spatial_age_gate_algorithm_digest,
                    )
                )
                or any(
                    getattr(self, name)
                    != OBSERVATION_ERROR_DERIVATION_ALGORITHM_DIGEST
                    for name in (
                        "range_elevation_validity_algorithm_digest",
                        "beam_blockage_algorithm_digest",
                        "quality_weight_algorithm_digest",
                        "observation_std_algorithm_digest",
                        "source_assignment_algorithm_digest",
                    )
                )
            ):
                raise ValueError(
                    "deterministic observation-error derivation is unsupported"
                )
        else:
            expected_error_algorithm = (
                OBSERVATION_ERROR_DERIVATION_ALGORITHM_V11_DIGEST
                if self.contract == "verification-observation-error-plan-v12"
                else (
                    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V10_DIGEST
                    if self.contract == "verification-observation-error-plan-v11"
                    else (
                        OBSERVATION_ERROR_DERIVATION_ALGORITHM_V8_DIGEST
                        if self.contract == "verification-observation-error-plan-v9"
                        else (
                            OBSERVATION_ERROR_DERIVATION_ALGORITHM_V7_DIGEST
                            if self.contract == "verification-observation-error-plan-v8"
                            else (
                                OBSERVATION_ERROR_DERIVATION_ALGORITHM_V6_DIGEST
                                if self.contract == "verification-observation-error-plan-v7"
                                else (
                                    OBSERVATION_ERROR_DERIVATION_ALGORITHM_V5_DIGEST
                                    if self.contract == "verification-observation-error-plan-v6"
                                    else (
                                        OBSERVATION_ERROR_DERIVATION_ALGORITHM_V4_DIGEST
                                        if self.contract == "verification-observation-error-plan-v5"
                                        else (
                                            OBSERVATION_ERROR_DERIVATION_ALGORITHM_V3_DIGEST
                                            if self.contract == "verification-observation-error-plan-v4"
                                            else OBSERVATION_ERROR_DERIVATION_ALGORITHM_V2_DIGEST
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            expected_mask_algorithm = (
                OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST
                if self.contract == "verification-observation-error-plan-v12"
                else (
                    OBSERVATION_MASK_DERIVATION_ALGORITHM_V9_DIGEST
                    if self.contract == "verification-observation-error-plan-v11"
                    else (
                        OBSERVATION_MASK_DERIVATION_ALGORITHM_V7_DIGEST
                        if self.contract == "verification-observation-error-plan-v9"
                        else (
                            OBSERVATION_MASK_DERIVATION_ALGORITHM_V6_DIGEST
                            if self.contract == "verification-observation-error-plan-v8"
                            else (
                                OBSERVATION_MASK_DERIVATION_ALGORITHM_V5_DIGEST
                                if self.contract == "verification-observation-error-plan-v7"
                                else (
                                    OBSERVATION_MASK_DERIVATION_ALGORITHM_V4_DIGEST
                                    if self.contract == "verification-observation-error-plan-v6"
                                    else (
                                        OBSERVATION_MASK_DERIVATION_ALGORITHM_V3_DIGEST
                                        if self.contract == "verification-observation-error-plan-v5"
                                        else (
                                            OBSERVATION_MASK_DERIVATION_ALGORITHM_V2_DIGEST
                                            if self.contract == "verification-observation-error-plan-v4"
                                            else OBSERVATION_MASK_DERIVATION_ALGORITHM_V1_DIGEST
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            scalar_values = (
                self.maximum_range_km,
                self.minimum_elevation_deg,
                self.maximum_elevation_deg,
                self.maximum_beam_blockage_fraction,
                self.minimum_attenuation_qc_score,
            )
            if (
                self.derivation_algorithm_digest != expected_error_algorithm
                or self.mask_derivation_algorithm_digest
                != expected_mask_algorithm
                or any(value is None or not math.isfinite(value) for value in scalar_values)
                or cast(float, self.maximum_range_km) <= 0.0
                or cast(float, self.minimum_elevation_deg)
                > cast(float, self.maximum_elevation_deg)
                or not 0.0
                <= cast(float, self.maximum_beam_blockage_fraction)
                <= 1.0
                or not 0.0 <= cast(float, self.minimum_attenuation_qc_score) <= 1.0
                or self.range_elevation_validity_algorithm_digest
                != expected_mask_algorithm
                or self.beam_blockage_algorithm_digest
                != expected_mask_algorithm
                or self.source_assignment_algorithm_digest
                != (
                    OBSERVATION_SOURCE_SELECTION_ALGORITHM_V1_DIGEST
                    if self.contract
                    in {
                        "verification-observation-error-plan-v7",
                        "verification-observation-error-plan-v8",
                        "verification-observation-error-plan-v9",
                        "verification-observation-error-plan-v11",
                        "verification-observation-error-plan-v12",
                    }
                    else expected_mask_algorithm
                )
                or self.quality_weight_algorithm_digest
                != expected_error_algorithm
                or self.observation_std_algorithm_digest
                != expected_error_algorithm
                or not isinstance(self.verification_source_authority_id, str)
                or not self.verification_source_authority_id
                or self.verification_source_authority_id.strip()
                != self.verification_source_authority_id
                or not isinstance(
                    self.verification_source_authority_public_key_hex,
                    str,
                )
            ):
                raise ValueError(
                    "deterministic observation-mask derivation is unsupported"
                )
            temporal_values = (
                self.maximum_acquisition_age_seconds,
                self.temporal_quality_decay_scale_seconds,
                self.temporal_quality_decay_power,
                self.temporal_error_growth_dbz_per_second,
            )
            if self.contract in {
                "verification-observation-error-plan-v5",
                "verification-observation-error-plan-v6",
                "verification-observation-error-plan-v7",
                "verification-observation-error-plan-v8",
                "verification-observation-error-plan-v9",
                "verification-observation-error-plan-v11",
                "verification-observation-error-plan-v12",
            }:
                expected_detection_algorithm = (
                    OBSERVATION_DETECTION_LIMIT_ALGORITHM_V2_DIGEST
                    if self.contract in {
                        "verification-observation-error-plan-v6",
                        "verification-observation-error-plan-v7",
                        "verification-observation-error-plan-v8",
                        "verification-observation-error-plan-v9",
                        "verification-observation-error-plan-v11",
                        "verification-observation-error-plan-v12",
                    }
                    else OBSERVATION_DETECTION_LIMIT_ALGORITHM_V1_DIGEST
                )
                expected_censor_algorithm = (
                    OBSERVATION_REPORT_KIND_ALGORITHM_V1_DIGEST
                    if self.contract in {
                        "verification-observation-error-plan-v6",
                        "verification-observation-error-plan-v7",
                        "verification-observation-error-plan-v8",
                        "verification-observation-error-plan-v9",
                        "verification-observation-error-plan-v11",
                        "verification-observation-error-plan-v12",
                    }
                    else OBSERVATION_CENSOR_STATE_ALGORITHM_V1_DIGEST
                )
                if (
                    any(
                        value is None or not math.isfinite(value)
                        for value in temporal_values
                    )
                    or cast(float, self.maximum_acquisition_age_seconds) <= 0.0
                    or cast(float, self.temporal_quality_decay_scale_seconds)
                    <= 0.0
                    or cast(float, self.temporal_quality_decay_power) <= 0.0
                    or cast(float, self.temporal_error_growth_dbz_per_second)
                    < 0.0
                    or self.temporal_quality_decay_algorithm_digest
                    != OBSERVATION_TEMPORAL_QUALITY_DECAY_ALGORITHM_V1_DIGEST
                    or self.temporal_error_algorithm_digest
                    != OBSERVATION_TEMPORAL_ERROR_ALGORITHM_V1_DIGEST
                    or self.detection_limit_derivation_algorithm_digest
                    != expected_detection_algorithm
                    or self.censor_state_derivation_algorithm_digest
                    != expected_censor_algorithm
                ):
                    raise ValueError(
                        "preregistered temporal/censor derivation is unsupported"
                    )
            elif any(value is not None for value in (
                *temporal_values,
                self.temporal_quality_decay_algorithm_digest,
                self.temporal_error_algorithm_digest,
                self.detection_limit_derivation_algorithm_digest,
                self.censor_state_derivation_algorithm_digest,
            )):
                raise ValueError("legacy plans cannot claim temporal/censor replay")
            scientific_geometry_fields = (
                self.geometry_contract_digest,
                self.acquisition_timestamp_reference,
                self.spatial_metric_reference_speed_mps,
                self.spatial_metric_maximum_displacement_fraction_cells,
                self.spatial_age_gate_algorithm_digest,
            )
            if self.contract in {
                "verification-observation-error-plan-v7",
                "verification-observation-error-plan-v8",
                "verification-observation-error-plan-v9",
                "verification-observation-error-plan-v11",
                "verification-observation-error-plan-v12",
            }:
                if (
                    not isinstance(self.geometry_contract_digest, str)
                    or self.acquisition_timestamp_reference != "volume_end"
                    or self.spatial_metric_reference_speed_mps is None
                    or not math.isfinite(self.spatial_metric_reference_speed_mps)
                    or self.spatial_metric_reference_speed_mps <= 0.0
                    or self.spatial_metric_maximum_displacement_fraction_cells
                    is None
                    or not math.isfinite(
                        self.spatial_metric_maximum_displacement_fraction_cells
                    )
                    or not 0.0
                    < self.spatial_metric_maximum_displacement_fraction_cells
                    <= 1.0
                    or self.spatial_age_gate_algorithm_digest
                    != (
                        OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V3_DIGEST
                        if self.contract
                        in {
                            "verification-observation-error-plan-v11",
                            "verification-observation-error-plan-v12",
                        }
                        else OBSERVATION_SPATIAL_AGE_GATE_ALGORITHM_V1_DIGEST
                    )
                ):
                    raise ValueError(
                        "scientific geometry/time policy is unsupported"
                    )
                _require_sha256(
                    "verification geometry contract",
                    self.geometry_contract_digest,
                )
            elif any(value is not None for value in scientific_geometry_fields):
                raise ValueError(
                    "legacy plans cannot claim geometry/time replay"
                )
            try:
                source_authority_key = bytes.fromhex(
                    cast(str, self.verification_source_authority_public_key_hex)
                )
                Ed25519PublicKey.from_public_bytes(source_authority_key)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "verification source authority key is invalid"
                ) from error
            if len(source_authority_key) != 32:
                raise ValueError("verification source authority key is invalid")
        for name in (
            "source_registry_digest",
            "calibration_registry_digest",
            "range_elevation_validity_algorithm_digest",
            "beam_blockage_algorithm_digest",
            "attenuation_qc_digest",
            "censoring_rule_digest",
            "spatial_correlation_block_algorithm_digest",
            "quality_weight_interpretation_digest",
            "quality_weight_algorithm_digest",
            "observation_std_algorithm_digest",
            "observation_error_model_digest",
            "source_assignment_algorithm_digest",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "temporal_quality_decay_algorithm_digest",
            "temporal_error_algorithm_digest",
            "detection_limit_derivation_algorithm_digest",
            "censor_state_derivation_algorithm_digest",
            "geometry_contract_digest",
            "spatial_age_gate_algorithm_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(name, value)
        object.__setattr__(self, "plan_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if key != "plan_digest"
        }
        deterministic_fields = {
            "derivation_algorithm_digest",
            "mask_derivation_algorithm_digest",
            "maximum_range_km",
            "minimum_elevation_deg",
            "maximum_elevation_deg",
            "maximum_beam_blockage_fraction",
            "minimum_attenuation_qc_score",
            "verification_source_authority_id",
            "verification_source_authority_public_key_hex",
        }
        temporal_censor_fields = {
            "maximum_acquisition_age_seconds",
            "temporal_quality_decay_scale_seconds",
            "temporal_quality_decay_power",
            "temporal_error_growth_dbz_per_second",
            "temporal_quality_decay_algorithm_digest",
            "temporal_error_algorithm_digest",
            "detection_limit_derivation_algorithm_digest",
            "censor_state_derivation_algorithm_digest",
        }
        scientific_geometry_fields = {
            "geometry_contract_digest",
            "acquisition_timestamp_reference",
            "spatial_metric_reference_speed_mps",
            "spatial_metric_maximum_displacement_fraction_cells",
            "spatial_age_gate_algorithm_digest",
        }
        if self.contract == "verification-observation-error-plan-v1":
            for key in (
                deterministic_fields
                | temporal_censor_fields
                | scientific_geometry_fields
            ):
                payload.pop(key)
        elif self.contract == "verification-observation-error-plan-v2":
            for key in (
                deterministic_fields
                - {"derivation_algorithm_digest"}
            ) | temporal_censor_fields | scientific_geometry_fields:
                payload.pop(key)
        elif self.contract in {
            "verification-observation-error-plan-v3",
            "verification-observation-error-plan-v4",
        }:
            for key in temporal_censor_fields:
                payload.pop(key)
            for key in scientific_geometry_fields:
                payload.pop(key)
        elif self.contract in {
            "verification-observation-error-plan-v5",
            "verification-observation-error-plan-v6",
        }:
            for key in scientific_geometry_fields:
                payload.pop(key)
        return payload

    def validate_integrity(self) -> None:
        if self.plan_digest != json_digest(self.payload):
            raise ValueError("verification observation-error plan digest mismatch")


@dataclass(frozen=True)
class RadarObservationGeometryContract:
    """Product-owned projected geometry for scientific verification replay."""

    grid_contract_digest: str
    projected_crs_digest: str
    grid_x_m: Tensor
    grid_y_m: Tensor
    grid_spacing_m: float
    projected_grid_identity: RadarSpatialGridIdentity | None = None
    contract: str = "radar-observation-geometry-v3"
    geometry_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract not in {
                "radar-observation-geometry-v1",
                "radar-observation-geometry-v2",
                "radar-observation-geometry-v3",
                "radar-observation-geometry-v5",
                "radar-observation-geometry-v6",
            }
            or self.grid_x_m.ndim != 2
            or self.grid_y_m.shape != self.grid_x_m.shape
            or not self.grid_x_m.is_floating_point()
            or self.grid_y_m.dtype != self.grid_x_m.dtype
            or self.grid_y_m.device != self.grid_x_m.device
            or not bool(torch.all(torch.isfinite(self.grid_x_m)))
            or not bool(torch.all(torch.isfinite(self.grid_y_m)))
            or not math.isfinite(self.grid_spacing_m)
            or self.grid_spacing_m <= 0.0
        ):
            raise ValueError("radar observation geometry is invalid")
        _require_sha256("radar observation grid", self.grid_contract_digest)
        _require_sha256("radar observation CRS", self.projected_crs_digest)
        if self.contract == "radar-observation-geometry-v2":
            if self.projected_grid_identity is not None:
                raise ValueError("geometry v2 cannot claim a projected-grid identity")
            derived_spacing_m = _derived_uniform_rectilinear_grid_spacing_m(
                self.grid_x_m,
                self.grid_y_m,
            )
            if not math.isclose(
                self.grid_spacing_m,
                derived_spacing_m,
                rel_tol=0.0,
                abs_tol=RADAR_GEOMETRY_SPACING_ABSOLUTE_TOLERANCE_M,
            ):
                raise ValueError(
                    "radar observation spacing disagrees with grid coordinates"
                )
            object.__setattr__(self, "grid_spacing_m", derived_spacing_m)
        elif self.contract in {
            "radar-observation-geometry-v3",
            "radar-observation-geometry-v5",
            "radar-observation-geometry-v6",
        }:
            identity = self.projected_grid_identity
            expected_identity_contract = (
                "radar-spatial-grid-identity-v4"
                if self.contract == "radar-observation-geometry-v5"
                else (
                    "radar-spatial-grid-identity-v5"
                    if self.contract == "radar-observation-geometry-v6"
                    else "radar-spatial-grid-identity-v2"
                )
            )
            expected_spacing_m = (
                identity.spatial_metric_spacing_m
                if type(identity) is RadarSpatialGridIdentity
                and self.contract
                in {
                    "radar-observation-geometry-v5",
                    "radar-observation-geometry-v6",
                }
                else (
                    identity.minimum_axis_spacing_m
                    if type(identity) is RadarSpatialGridIdentity
                    else None
                )
            )
            if (
                type(identity) is not RadarSpatialGridIdentity
                or identity.contract != expected_identity_contract
                or identity.shape_yx != tuple(self.grid_x_m.shape)
                or identity.projected_crs_digest != self.projected_crs_digest
                or self.grid_x_m.dtype != torch.float64
                or self.grid_y_m.dtype != torch.float64
                or self.grid_spacing_m != expected_spacing_m
            ):
                raise ValueError("scientific radar affine geometry is invalid")
            expected_x_m, expected_y_m = (
                identity.projected_cell_center_coordinates(
                    device=self.grid_x_m.device,
                )
            )
            if not torch.equal(
                self.grid_x_m,
                expected_x_m,
            ) or not torch.equal(self.grid_y_m, expected_y_m):
                raise ValueError(
                    "radar observation coordinates disagree with the projected grid"
                )
        elif self.projected_grid_identity is not None:
            raise ValueError("legacy geometry cannot claim a projected-grid identity")
        object.__setattr__(self, "grid_x_m", self.grid_x_m.detach().clone())
        object.__setattr__(self, "grid_y_m", self.grid_y_m.detach().clone())
        object.__setattr__(self, "geometry_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract": self.contract,
            "projected_crs_digest": self.projected_crs_digest,
            "grid_x_m_digest": tensor_digest(self.grid_x_m),
            "grid_y_m_digest": tensor_digest(self.grid_y_m),
            "grid_spacing_m": self.grid_spacing_m,
        }
        if self.contract in {
            "radar-observation-geometry-v2",
            "radar-observation-geometry-v3",
            "radar-observation-geometry-v5",
            "radar-observation-geometry-v6",
        }:
            payload["grid_contract_digest"] = self.grid_contract_digest
        if self.contract in {
            "radar-observation-geometry-v3",
            "radar-observation-geometry-v5",
            "radar-observation-geometry-v6",
        }:
            assert self.projected_grid_identity is not None
            payload.update(
                {
                    "projected_grid_identity": (
                        self.projected_grid_identity.payload
                    ),
                    "projected_grid_identity_digest": (
                        self.projected_grid_identity.digest
                    ),
                }
            )
        return payload

    def validate_integrity(self) -> None:
        if self.geometry_digest != json_digest(self.payload):
            raise ValueError("radar observation geometry digest mismatch")

    @classmethod
    def from_grid_time_contract(
        cls,
        grid: RadarGridTimeContract,
        *,
        device: torch.device | str | None = None,
    ) -> RadarObservationGeometryContract:
        """Create current geometry only from the forecast grid authority."""

        if type(grid) is not RadarGridTimeContract:
            raise ValueError("radar grid/time contract is required")
        identity = grid.spatial_grid_identity
        if identity.contract != "radar-spatial-grid-identity-v5":
            raise ValueError("scientific verification requires projected-grid v5")
        identity.validate_current_metric_domain_evidence()
        grid_x_m, grid_y_m = identity.projected_cell_center_coordinates(
            device=device,
        )
        assert identity.projected_crs_digest is not None
        return cls(
            grid_contract_digest=grid.digest,
            projected_crs_digest=identity.projected_crs_digest,
            grid_x_m=grid_x_m,
            grid_y_m=grid_y_m,
            grid_spacing_m=identity.spatial_metric_spacing_m,
            projected_grid_identity=identity,
            contract="radar-observation-geometry-v6",
        )


@dataclass(frozen=True)
class ObservationRadarSource:
    """One ordered radar/calibration entry used by observation-error replay."""

    radar_site_digest: str
    calibration_epoch_digest: str
    quality_weight: float
    observation_std_dbz: float
    detection_limit_dbz: float = -10.0
    detection_limit_range_quadratic_dbz_per_km2: float = 0.0
    detection_limit_elevation_excess_dbz_per_degree: float = 0.0
    detection_limit_reference_elevation_deg: float = 0.0
    projected_x_m: float | None = None
    projected_y_m: float | None = None
    radar_altitude_m: float | None = None
    representative_scan_elevation_deg: float | None = None
    contract: str = "observation-radar-source-v3"
    source_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract not in {
                "observation-radar-source-v1",
                "observation-radar-source-v2",
                "observation-radar-source-v3",
                "observation-radar-source-v4",
            }
            or not math.isfinite(self.quality_weight)
            or not 0.0 <= self.quality_weight <= 1.0
            or not math.isfinite(self.observation_std_dbz)
            or self.observation_std_dbz <= 0.0
            or not math.isfinite(self.detection_limit_dbz)
            or not math.isfinite(
                self.detection_limit_range_quadratic_dbz_per_km2
            )
            or self.detection_limit_range_quadratic_dbz_per_km2 < 0.0
            or not math.isfinite(
                self.detection_limit_elevation_excess_dbz_per_degree
            )
            or self.detection_limit_elevation_excess_dbz_per_degree < 0.0
            or not math.isfinite(
                self.detection_limit_reference_elevation_deg
            )
            or (
                self.contract == "observation-radar-source-v1"
                and self.detection_limit_dbz != -10.0
            )
            or (
                self.contract not in {
                    "observation-radar-source-v3",
                    "observation-radar-source-v4",
                }
                and any(
                    value != 0.0
                    for value in (
                        self.detection_limit_range_quadratic_dbz_per_km2,
                        self.detection_limit_elevation_excess_dbz_per_degree,
                        self.detection_limit_reference_elevation_deg,
                    )
                )
            )
            or (
                self.contract == "observation-radar-source-v4"
                and any(
                    value is None or not math.isfinite(value)
                    for value in (
                        self.projected_x_m,
                        self.projected_y_m,
                        self.radar_altitude_m,
                        self.representative_scan_elevation_deg,
                    )
                )
            )
            or (
                self.contract != "observation-radar-source-v4"
                and any(
                    value is not None
                    for value in (
                        self.projected_x_m,
                        self.projected_y_m,
                        self.radar_altitude_m,
                        self.representative_scan_elevation_deg,
                    )
                )
            )
        ):
            raise ValueError("observation radar source is invalid")
        _require_sha256("observation radar site", self.radar_site_digest)
        _require_sha256(
            "observation radar calibration epoch",
            self.calibration_epoch_digest,
        )
        object.__setattr__(self, "source_digest", json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if key != "source_digest"
        }
        if self.contract == "observation-radar-source-v1":
            payload.pop("detection_limit_dbz")
        if self.contract not in {
            "observation-radar-source-v3",
            "observation-radar-source-v4",
        }:
            payload.pop("detection_limit_range_quadratic_dbz_per_km2")
            payload.pop("detection_limit_elevation_excess_dbz_per_degree")
            payload.pop("detection_limit_reference_elevation_deg")
        if self.contract != "observation-radar-source-v4":
            payload.pop("projected_x_m")
            payload.pop("projected_y_m")
            payload.pop("radar_altitude_m")
            payload.pop("representative_scan_elevation_deg")
        return payload

    def validate_integrity(self) -> None:
        if self.source_digest != json_digest(self.payload):
            raise ValueError("observation radar source digest mismatch")


@dataclass(frozen=True)
class MosaicObservationSourceRegistry:
    """Canonical index-to-radar/calibration registry for scientific replay."""

    radar_source_kind: Literal["single_site", "mosaic"]
    ordered_sources: tuple[ObservationRadarSource, ...]
    projected_crs_digest: str | None = None
    metric_domain_digest: str | None = None
    geometry_model: Literal[
        "projected-horizontal-representative-tilt-v1"
    ] | None = None
    radar_altitude_role: Literal["provenance_only"] | None = None
    contract: str = "mosaic-observation-source-registry-v3"
    source_registry_digest: str = field(init=False)
    calibration_registry_digest: str = field(init=False)
    registry_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract not in {
                "mosaic-observation-source-registry-v1",
                "mosaic-observation-source-registry-v2",
                "mosaic-observation-source-registry-v3",
                "mosaic-observation-source-registry-v4",
                "mosaic-observation-source-registry-v5",
                "mosaic-observation-source-registry-v6",
            }
            or self.radar_source_kind not in {"single_site", "mosaic"}
            or not self.ordered_sources
            or any(type(source) is not ObservationRadarSource for source in self.ordered_sources)
            or len({source.radar_site_digest for source in self.ordered_sources})
            != len(self.ordered_sources)
            or (
                self.radar_source_kind == "single_site"
                and len(self.ordered_sources) != 1
            )
            or any(
                source.contract
                != (
                    "observation-radar-source-v4"
                    if self.contract
                    in {
                        "mosaic-observation-source-registry-v4",
                        "mosaic-observation-source-registry-v5",
                        "mosaic-observation-source-registry-v6",
                    }
                    else (
                        "observation-radar-source-v3"
                        if self.contract == "mosaic-observation-source-registry-v3"
                        else (
                            "observation-radar-source-v2"
                            if self.contract == "mosaic-observation-source-registry-v2"
                            else "observation-radar-source-v1"
                        )
                    )
                )
                for source in self.ordered_sources
            )
            or (
                self.contract
                in {
                    "mosaic-observation-source-registry-v5",
                    "mosaic-observation-source-registry-v6",
                }
                and (
                    self.geometry_model
                    != "projected-horizontal-representative-tilt-v1"
                    or self.radar_altitude_role != "provenance_only"
                    or not isinstance(self.projected_crs_digest, str)
                    or (
                        self.metric_domain_digest
                        != (
                            CURRENT_RADAR_METRIC_DOMAIN.digest
                            if self.contract
                            == "mosaic-observation-source-registry-v6"
                            else None
                        )
                    )
                )
            )
            or (
                self.contract
                not in {
                    "mosaic-observation-source-registry-v5",
                    "mosaic-observation-source-registry-v6",
                }
                and any(
                    value is not None
                    for value in (
                        self.projected_crs_digest,
                        self.metric_domain_digest,
                        self.geometry_model,
                        self.radar_altitude_role,
                    )
                )
            )
        ):
            raise ValueError("mosaic observation source registry is invalid")
        if self.projected_crs_digest is not None:
            _require_sha256("observation radar registry CRS", self.projected_crs_digest)
        if self.contract == "mosaic-observation-source-registry-v6":
            for source in self.ordered_sources:
                CURRENT_RADAR_METRIC_DOMAIN.validate_projected_point(
                    cast(float, source.projected_x_m),
                    cast(float, source.projected_y_m),
                )
        source_registry_digest = json_digest(self._source_registry_payload())
        calibration_registry_digest = json_digest(
            self._calibration_registry_payload()
        )
        object.__setattr__(
            self,
            "source_registry_digest",
            source_registry_digest,
        )
        object.__setattr__(
            self,
            "calibration_registry_digest",
            calibration_registry_digest,
        )
        object.__setattr__(self, "registry_digest", json_digest(self.payload))

    def _source_registry_payload(self) -> dict[str, object]:
        generation = {
            "mosaic-observation-source-registry-v1": "v1",
            "mosaic-observation-source-registry-v2": "v2",
            "mosaic-observation-source-registry-v3": "v3",
            "mosaic-observation-source-registry-v4": "v4",
            "mosaic-observation-source-registry-v5": "v5",
            "mosaic-observation-source-registry-v6": "v6",
        }[self.contract]
        payload: dict[str, object] = {
            "contract": f"observation-radar-index-registry-{generation}",
            "ordered_radar_site_digests": [
                source.radar_site_digest for source in self.ordered_sources
            ],
        }
        if self.contract in {
            "mosaic-observation-source-registry-v5",
            "mosaic-observation-source-registry-v6",
        }:
            payload.update(
                {
                    "projected_crs_digest": self.projected_crs_digest,
                    "geometry_model": self.geometry_model,
                    "radar_altitude_role": self.radar_altitude_role,
                }
            )
        if self.contract == "mosaic-observation-source-registry-v6":
            payload["metric_domain_digest"] = self.metric_domain_digest
        return payload

    def _calibration_registry_payload(self) -> dict[str, object]:
        generation = {
            "mosaic-observation-source-registry-v1": "v1",
            "mosaic-observation-source-registry-v2": "v2",
            "mosaic-observation-source-registry-v3": "v3",
            "mosaic-observation-source-registry-v4": "v4",
            "mosaic-observation-source-registry-v5": "v5",
            "mosaic-observation-source-registry-v6": "v6",
        }[self.contract]
        return {
            "contract": f"observation-radar-calibration-registry-{generation}",
            "ordered_sources": [source.payload for source in self.ordered_sources],
        }

    def validate_against_plan(
        self,
        plan: VerificationObservationErrorPlan,
    ) -> None:
        self.validate_integrity()
        plan.validate_integrity()
        if (
            type(plan) is not VerificationObservationErrorPlan
            or self.registry_digest != json_digest(self.payload)
            or self.radar_source_kind != plan.radar_source_kind
            or self.source_registry_digest != plan.source_registry_digest
            or self.calibration_registry_digest
            != plan.calibration_registry_digest
        ):
            raise ValueError(
                "mosaic observation source registry disagrees with its plan"
            )

    def validate_integrity(self) -> None:
        for source in self.ordered_sources:
            source.validate_integrity()
        expected_source_registry_digest = json_digest(
            self._source_registry_payload()
        )
        expected_calibration_registry_digest = json_digest(
            self._calibration_registry_payload()
        )
        if (
            self.source_registry_digest != expected_source_registry_digest
            or self.calibration_registry_digest
            != expected_calibration_registry_digest
            or self.registry_digest != json_digest(self.payload)
        ):
            raise ValueError("mosaic observation source registry digest mismatch")

    def validate_source_map(
        self,
        source_radar_index_map: Tensor | None,
        *,
        expected_shape: torch.Size,
        expected_device: torch.device,
    ) -> None:
        if self.radar_source_kind == "single_site":
            if source_radar_index_map is not None:
                raise ValueError("single-site observation source map is invalid")
            return
        if (
            source_radar_index_map is None
            or source_radar_index_map.dtype is not torch.int64
            or source_radar_index_map.shape != expected_shape
            or source_radar_index_map.device != expected_device
            or not bool(torch.all(source_radar_index_map >= -1))
            or not bool(
                torch.all(
                    source_radar_index_map
                    < len(self.ordered_sources)
                )
            )
        ):
            raise ValueError(
                "mosaic observation source index is outside its registry"
            )

    @property
    def source_calibration_epochs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (source.radar_site_digest, source.calibration_epoch_digest)
            for source in self.ordered_sources
        )

    @property
    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract": self.contract,
            "radar_source_kind": self.radar_source_kind,
            "ordered_source_digests": [
                source.source_digest for source in self.ordered_sources
            ],
            "source_registry_digest": self.source_registry_digest,
            "calibration_registry_digest": self.calibration_registry_digest,
        }
        if self.contract in {
            "mosaic-observation-source-registry-v5",
            "mosaic-observation-source-registry-v6",
        }:
            payload.update(
                {
                    "projected_crs_digest": self.projected_crs_digest,
                    "geometry_model": self.geometry_model,
                    "radar_altitude_role": self.radar_altitude_role,
                }
            )
        if self.contract == "mosaic-observation-source-registry-v6":
            payload["metric_domain_digest"] = self.metric_domain_digest
        return payload


@dataclass(frozen=True)
class VerificationObservationSourceIdentity:
    """Typed time/grid/product identity of future verification evidence."""

    valid_times: tuple[str, ...]
    acquisition_valid_times_by_source: tuple[tuple[str, ...], ...]
    grid_contract_digest: str
    radar_product_digest: str
    native_verification_source_identity_digest: str
    upstream_verification_artifact_digest: str
    source_authority_id: str
    source_authority_public_key_hex: str
    source_observed_at: str
    source_signature_hex: str
    acquisition_timestamp_reference: Literal["volume_end"] | None = None
    contract: str = "verification-observation-source-identity-v3"
    source_acquisition_time_identity_digest: str = field(init=False)
    identity_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract not in {
                "verification-observation-source-identity-v2",
                "verification-observation-source-identity-v3",
            }
            or not self.valid_times
            or not self.acquisition_valid_times_by_source
            or any(
                len(row) != len(self.valid_times)
                for row in self.acquisition_valid_times_by_source
            )
        ):
            raise ValueError("verification observation source identity is invalid")
        if (
            self.contract == "verification-observation-source-identity-v3"
            and self.acquisition_timestamp_reference != "volume_end"
        ) or (
            self.contract == "verification-observation-source-identity-v2"
            and self.acquisition_timestamp_reference is not None
        ):
            raise ValueError("verification acquisition timestamp reference is invalid")
        valid_times = tuple(
            _canonical_verification_time(value) for value in self.valid_times
        )
        acquisition_times_by_source = tuple(
            tuple(_canonical_verification_time(value) for value in row)
            for row in self.acquisition_valid_times_by_source
        )
        valid_datetimes = tuple(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in valid_times
        )
        acquisition_datetimes_by_source = tuple(
            tuple(
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                for value in row
            )
            for row in acquisition_times_by_source
        )
        if (
            any(
                later <= earlier
                for earlier, later in zip(
                    valid_datetimes,
                    valid_datetimes[1:],
                )
            )
            or any(
                acquisition > valid_datetimes[index]
                for row in acquisition_datetimes_by_source
                for index, acquisition in enumerate(row)
            )
        ):
            raise ValueError("verification observation source chronology is invalid")
        for name in (
            "grid_contract_digest",
            "radar_product_digest",
            "native_verification_source_identity_digest",
            "upstream_verification_artifact_digest",
        ):
            _require_sha256(name, getattr(self, name))
        observed_at = _canonical_verification_time(self.source_observed_at)
        observed_datetime = datetime.fromisoformat(
            observed_at.replace("Z", "+00:00")
        )
        if (
            not self.source_authority_id
            or self.source_authority_id.strip() != self.source_authority_id
            or any(
                observed_datetime < value
                for row in acquisition_datetimes_by_source
                for value in row
            )
        ):
            raise ValueError("verification source authority identity is invalid")
        object.__setattr__(self, "valid_times", valid_times)
        object.__setattr__(
            self,
            "acquisition_valid_times_by_source",
            acquisition_times_by_source,
        )
        object.__setattr__(self, "source_observed_at", observed_at)
        object.__setattr__(
            self,
            "source_acquisition_time_identity_digest",
            json_digest(
                {
                    "contract": (
                        "verification-source-acquisition-times-v3"
                        if self.contract == "verification-observation-source-identity-v3"
                        else "verification-source-acquisition-times-v2"
                    ),
                    "native_verification_source_identity_digest": (
                        self.native_verification_source_identity_digest
                    ),
                    "acquisition_valid_times_by_source": [
                        list(row) for row in acquisition_times_by_source
                    ],
                    **(
                        {
                            "acquisition_timestamp_reference": (
                                self.acquisition_timestamp_reference
                            )
                        }
                        if self.contract == "verification-observation-source-identity-v3"
                        else {}
                    ),
                }
            ),
        )
        try:
            source_public_key = bytes.fromhex(
                self.source_authority_public_key_hex
            )
            Ed25519PublicKey.from_public_bytes(source_public_key).verify(
                bytes.fromhex(self.source_signature_hex),
                json_digest(self.unsigned_payload).encode("ascii"),
            )
        except (InvalidSignature, TypeError, ValueError) as error:
            raise ValueError("verification source signature is invalid") from error
        if len(source_public_key) != 32:
            raise ValueError("verification source authority key is invalid")
        object.__setattr__(self, "identity_digest", json_digest(self.payload))

    def validate_integrity(self) -> None:
        expected_acquisition_digest = json_digest(
            {
                "contract": (
                    "verification-source-acquisition-times-v3"
                    if self.contract == "verification-observation-source-identity-v3"
                    else "verification-source-acquisition-times-v2"
                ),
                "native_verification_source_identity_digest": (
                    self.native_verification_source_identity_digest
                ),
                "acquisition_valid_times_by_source": [
                    list(row) for row in self.acquisition_valid_times_by_source
                ],
                **(
                    {
                        "acquisition_timestamp_reference": (
                            self.acquisition_timestamp_reference
                        )
                    }
                    if self.contract == "verification-observation-source-identity-v3"
                    else {}
                ),
            }
        )
        if (
            self.source_acquisition_time_identity_digest
            != expected_acquisition_digest
            or self.identity_digest != json_digest(self.payload)
        ):
            raise ValueError("verification observation source identity mismatch")
        try:
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.source_authority_public_key_hex)
            ).verify(
                bytes.fromhex(self.source_signature_hex),
                json_digest(self.unsigned_payload).encode("ascii"),
            )
        except (InvalidSignature, TypeError, ValueError) as error:
            raise ValueError("verification source signature is invalid") from error

    @property
    def unsigned_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract": self.contract,
            "valid_times": list(self.valid_times),
            "acquisition_valid_times_by_source": [
                list(row) for row in self.acquisition_valid_times_by_source
            ],
            "grid_contract_digest": self.grid_contract_digest,
            "radar_product_digest": self.radar_product_digest,
            "native_verification_source_identity_digest": (
                self.native_verification_source_identity_digest
            ),
            "upstream_verification_artifact_digest": (
                self.upstream_verification_artifact_digest
            ),
            "source_acquisition_time_identity_digest": (
                self.source_acquisition_time_identity_digest
            ),
            "source_authority_id": self.source_authority_id,
            "source_authority_public_key_hex": (
                self.source_authority_public_key_hex
            ),
            "source_observed_at": self.source_observed_at,
        }
        if self.contract == "verification-observation-source-identity-v3":
            payload["acquisition_timestamp_reference"] = (
                self.acquisition_timestamp_reference
            )
        return payload

    @property
    def payload(self) -> dict[str, object]:
        return self.unsigned_payload | {
            "source_signature_hex": self.source_signature_hex,
        }

    @classmethod
    def issue(
        cls,
        *,
        valid_times: tuple[str, ...],
        acquisition_valid_times_by_source: tuple[tuple[str, ...], ...],
        grid_contract_digest: str,
        radar_product_digest: str,
        native_verification_source_identity_digest: str,
        upstream_verification_artifact_digest: str,
        source_authority_id: str,
        source_authority_private_key: Ed25519PrivateKey,
        source_observed_at: str,
        acquisition_timestamp_reference: Literal["volume_end"] = "volume_end",
    ) -> VerificationObservationSourceIdentity:
        public_key_hex = source_authority_private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        canonical_valid_times = tuple(
            _canonical_verification_time(value) for value in valid_times
        )
        canonical_acquisition_times_by_source = tuple(
            tuple(_canonical_verification_time(value) for value in row)
            for row in acquisition_valid_times_by_source
        )
        acquisition_digest = json_digest(
            {
                "contract": "verification-source-acquisition-times-v3",
                "native_verification_source_identity_digest": (
                    native_verification_source_identity_digest
                ),
                "acquisition_valid_times_by_source": [
                    list(row) for row in canonical_acquisition_times_by_source
                ],
                "acquisition_timestamp_reference": acquisition_timestamp_reference,
            }
        )
        unsigned = {
            "contract": "verification-observation-source-identity-v3",
            "valid_times": list(canonical_valid_times),
            "acquisition_valid_times_by_source": [
                list(row) for row in canonical_acquisition_times_by_source
            ],
            "grid_contract_digest": grid_contract_digest,
            "radar_product_digest": radar_product_digest,
            "native_verification_source_identity_digest": (
                native_verification_source_identity_digest
            ),
            "upstream_verification_artifact_digest": (
                upstream_verification_artifact_digest
            ),
            "source_acquisition_time_identity_digest": acquisition_digest,
            "source_authority_id": source_authority_id,
            "source_authority_public_key_hex": public_key_hex,
            "source_observed_at": _canonical_verification_time(
                source_observed_at
            ),
            "acquisition_timestamp_reference": acquisition_timestamp_reference,
        }

        return cls(
            valid_times=canonical_valid_times,
            acquisition_valid_times_by_source=(
                canonical_acquisition_times_by_source
            ),
            grid_contract_digest=grid_contract_digest,
            radar_product_digest=radar_product_digest,
            native_verification_source_identity_digest=(
                native_verification_source_identity_digest
            ),
            upstream_verification_artifact_digest=(
                upstream_verification_artifact_digest
            ),
            source_authority_id=source_authority_id,
            source_authority_public_key_hex=public_key_hex,
            source_observed_at=_canonical_verification_time(source_observed_at),
            source_signature_hex=source_authority_private_key.sign(
                json_digest(unsigned).encode("ascii")
            ).hex(),
            acquisition_timestamp_reference=acquisition_timestamp_reference,
        )


def _verification_observation_upstream_artifact_digest(
    *,
    contract: Literal[
        "typed-upstream-verification-observation-v4",
        "typed-upstream-verification-observation-v5",
        "typed-upstream-verification-observation-v6",
        "typed-upstream-verification-observation-v7",
        "typed-upstream-verification-observation-v9",
        "typed-upstream-verification-observation-v10",
    ],
    valid_times: tuple[str, ...],
    acquisition_valid_times_by_source: tuple[tuple[str, ...], ...],
    grid_contract_digest: str,
    radar_product_digest: str,
    native_verification_source_identity_digest: str,
    source_registry_artifact_digest: str,
    ordered_source_digests: tuple[str, ...],
    reflectivity_dbz_by_source: Tensor,
    detection_limit_dbz_by_source: Tensor,
    acquisition_time_offset_seconds_by_source: Tensor,
    observation_report_kind_by_source: Tensor,
    source_assignment_scores: Tensor,
    source_availability_by_time: Tensor,
    range_km_by_source: Tensor,
    elevation_deg_by_source: Tensor,
    beam_blockage_fraction_by_source: Tensor,
    attenuation_qc_score_by_source: Tensor,
    range_elevation_validity_domain_digest: str,
    beam_blockage_visibility_mask_digest: str,
    spatial_correlation_block_digest: str,
    geometry_contract_digest: str | None = None,
) -> str:
    payload: dict[str, object] = {
            "contract": contract,
            "valid_times": list(valid_times),
            "acquisition_valid_times_by_source": [
                list(row) for row in acquisition_valid_times_by_source
            ],
            "grid_contract_digest": grid_contract_digest,
            "radar_product_digest": radar_product_digest,
            "native_verification_source_identity_digest": (
                native_verification_source_identity_digest
            ),
            "source_registry_artifact_digest": (
                source_registry_artifact_digest
            ),
            "ordered_source_digests": list(ordered_source_digests),
            "reflectivity_dbz_by_source_digest": tensor_digest(
                reflectivity_dbz_by_source
            ),
            "detection_limit_dbz_by_source_digest": tensor_digest(
                detection_limit_dbz_by_source
            ),
            "acquisition_time_offset_seconds_by_source_digest": tensor_digest(
                acquisition_time_offset_seconds_by_source
            ),
            "observation_report_kind_by_source_digest": tensor_digest(
                observation_report_kind_by_source
            ),
            "source_assignment_scores_digest": tensor_digest(
                source_assignment_scores
            ),
            "source_availability_by_time_digest": tensor_digest(
                source_availability_by_time
            ),
            "range_km_by_source_digest": tensor_digest(
                range_km_by_source
            ),
            "elevation_deg_by_source_digest": tensor_digest(
                elevation_deg_by_source
            ),
            "beam_blockage_fraction_by_source_digest": tensor_digest(
                beam_blockage_fraction_by_source
            ),
            "attenuation_qc_score_by_source_digest": tensor_digest(
                attenuation_qc_score_by_source
            ),
            "range_elevation_validity_domain_digest": (
                range_elevation_validity_domain_digest
            ),
            "beam_blockage_visibility_mask_digest": (
                beam_blockage_visibility_mask_digest
            ),
            "spatial_correlation_block_digest": (
                spatial_correlation_block_digest
            ),
        }
    if contract in {
        "typed-upstream-verification-observation-v5",
        "typed-upstream-verification-observation-v6",
        "typed-upstream-verification-observation-v7",
        "typed-upstream-verification-observation-v9",
        "typed-upstream-verification-observation-v10",
    }:
        if geometry_contract_digest is None:
            raise ValueError("verification geometry identity is missing")
        payload["geometry_contract_digest"] = geometry_contract_digest
    elif geometry_contract_digest is not None:
        raise ValueError("legacy verification evidence cannot claim geometry")
    return json_digest(payload)


class VerificationObservationReportKind(IntEnum):
    """Source-attested meaning of one radar observation sample."""

    DETECTED_ECHO = 0
    CONFIRMED_CLEAR = 1
    BELOW_DETECTION_CENSORED = 2


def _registered_observation_geometry_fields(
    *,
    source_registry: MosaicObservationSourceRegistry,
    geometry: RadarObservationGeometryContract,
    time_count: int,
    output_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    if (
        source_registry.contract != "mosaic-observation-source-registry-v6"
        or geometry.contract != "radar-observation-geometry-v6"
        or source_registry.projected_crs_digest
        != geometry.projected_crs_digest
        or source_registry.metric_domain_digest
        != CURRENT_RADAR_METRIC_DOMAIN.digest
        or geometry.projected_grid_identity is None
        or geometry.projected_grid_identity.metric_domain_digest
        != source_registry.metric_domain_digest
        or geometry.projected_grid_identity.metric_domain_evidence_digest
        != CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
        or source_registry.geometry_model
        != "projected-horizontal-representative-tilt-v1"
        or source_registry.radar_altitude_role != "provenance_only"
        or time_count <= 0
    ):
        raise ValueError("registered radar geometry is invalid")
    source_registry.validate_integrity()
    geometry.validate_integrity()
    ranges = []
    elevations = []
    for source in source_registry.ordered_sources:
        range_km = torch.sqrt(
            (geometry.grid_x_m - cast(float, source.projected_x_m)).square()
            + (geometry.grid_y_m - cast(float, source.projected_y_m)).square()
        ) / 1000.0
        if output_dtype is not None:
            if output_dtype not in {torch.float32, torch.float64}:
                raise ValueError(
                    "scientific radar source values require float32 or float64"
                )
            range_km = range_km.to(dtype=output_dtype)
        ranges.append(range_km.unsqueeze(0).expand(time_count, -1, -1))
        elevations.append(
            torch.full_like(
                range_km,
                cast(float, source.representative_scan_elevation_deg),
            ).unsqueeze(0).expand(time_count, -1, -1)
        )
    return torch.stack(ranges), torch.stack(elevations)


def _registered_detection_limit_field(
    *,
    source_registry: MosaicObservationSourceRegistry,
    range_km_by_source: Tensor,
    elevation_deg_by_source: Tensor,
) -> Tensor:
    sources = source_registry.ordered_sources
    base = range_km_by_source.new_tensor(
        [source.detection_limit_dbz for source in sources]
    )[:, None, None, None]
    range_coefficient = range_km_by_source.new_tensor(
        [
            source.detection_limit_range_quadratic_dbz_per_km2
            for source in sources
        ]
    )[:, None, None, None]
    elevation_coefficient = range_km_by_source.new_tensor(
        [
            source.detection_limit_elevation_excess_dbz_per_degree
            for source in sources
        ]
    )[:, None, None, None]
    reference_elevation = range_km_by_source.new_tensor(
        [source.detection_limit_reference_elevation_deg for source in sources]
    )[:, None, None, None]
    return (
        base
        + range_coefficient * range_km_by_source.square()
        + elevation_coefficient
        * torch.clamp(elevation_deg_by_source - reference_elevation, min=0.0)
    )


def _verification_acquisition_age_seconds_by_source(
    *,
    source_identity: VerificationObservationSourceIdentity,
    acquisition_time_offset_seconds_by_source: Tensor,
) -> Tensor:
    valid_datetimes = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in source_identity.valid_times
    )
    nominal_ages = acquisition_time_offset_seconds_by_source.new_tensor(
        [
            [
                (valid_datetimes[index] - datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )).total_seconds()
                for index, value in enumerate(row)
            ]
            for row in source_identity.acquisition_valid_times_by_source
        ]
    )[:, :, None, None]
    return nominal_ages - acquisition_time_offset_seconds_by_source


def _product_owned_source_assignment_scores(
    *,
    plan: VerificationObservationErrorPlan,
    valid_times: tuple[str, ...],
    acquisition_valid_times_by_source: tuple[tuple[str, ...], ...],
    acquisition_time_offset_seconds_by_source: Tensor,
    source_availability_by_time: Tensor,
    range_km_by_source: Tensor,
    elevation_deg_by_source: Tensor,
    beam_blockage_fraction_by_source: Tensor,
    attenuation_qc_score_by_source: Tensor,
) -> Tensor:
    """Derive deterministic eligible-first mosaic source scores."""

    if (
        type(plan) is not VerificationObservationErrorPlan
        or plan.contract != "verification-observation-error-plan-v12"
        or plan.source_assignment_algorithm_digest
        != OBSERVATION_SOURCE_SELECTION_ALGORITHM_V1_DIGEST
    ):
        raise ValueError("product-owned observation source selection is invalid")
    valid_datetimes = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in valid_times
    )
    nominal_ages = acquisition_time_offset_seconds_by_source.new_tensor(
        [
            [
                (
                    valid_datetimes[index]
                    - datetime.fromisoformat(value.replace("Z", "+00:00"))
                ).total_seconds()
                for index, value in enumerate(row)
            ]
            for row in acquisition_valid_times_by_source
        ]
    )[:, :, None, None]
    ages = nominal_ages - acquisition_time_offset_seconds_by_source
    availability = source_availability_by_time[:, :, None, None].expand_as(
        range_km_by_source
    )
    minimum_elevation = cast(float, plan.minimum_elevation_deg)
    maximum_elevation = cast(float, plan.maximum_elevation_deg)
    maximum_range = cast(float, plan.maximum_range_km)
    maximum_age = cast(float, plan.maximum_acquisition_age_seconds)
    maximum_blockage = cast(float, plan.maximum_beam_blockage_fraction)
    minimum_attenuation = cast(float, plan.minimum_attenuation_qc_score)
    eligible = (
        availability
        & (ages >= 0.0)
        & (ages <= maximum_age)
        & (range_km_by_source <= maximum_range)
        & (elevation_deg_by_source >= minimum_elevation)
        & (elevation_deg_by_source <= maximum_elevation)
        & (beam_blockage_fraction_by_source <= maximum_blockage)
        & (attenuation_qc_score_by_source >= minimum_attenuation)
    )
    range_quality = 1.0 / (1.0 + range_km_by_source)
    elevation_span = maximum_elevation - minimum_elevation
    if elevation_span > 0.0:
        elevation_midpoint = 0.5 * (minimum_elevation + maximum_elevation)
        elevation_quality = torch.clamp(
            1.0
            - 2.0
            * torch.abs(elevation_deg_by_source - elevation_midpoint)
            / elevation_span,
            min=0.0,
            max=1.0,
        )
    else:
        elevation_quality = torch.ones_like(elevation_deg_by_source)
    time_quality = torch.exp(
        -torch.pow(
            ages / cast(float, plan.temporal_quality_decay_scale_seconds),
            cast(float, plan.temporal_quality_decay_power),
        )
    )
    score = (
        1.0
        + range_quality
        + elevation_quality
        + time_quality
        + (1.0 - beam_blockage_fraction_by_source)
        + attenuation_qc_score_by_source
    )
    return torch.where(eligible, score, torch.zeros_like(score))


@dataclass(frozen=True)
class VerificationObservationMaskEvidence:
    """Exact raw fields from which product-owned mask algorithms replay."""

    source_identity: VerificationObservationSourceIdentity
    source_registry_artifact_digest: str
    ordered_source_digests: tuple[str, ...]
    reflectivity_dbz_by_source: Tensor
    detection_limit_dbz_by_source: Tensor
    acquisition_time_offset_seconds_by_source: Tensor
    observation_report_kind_by_source: Tensor
    source_assignment_scores: Tensor
    source_availability_by_time: Tensor
    range_km_by_source: Tensor
    elevation_deg_by_source: Tensor
    beam_blockage_fraction_by_source: Tensor
    attenuation_qc_score_by_source: Tensor
    range_elevation_validity_domain_digest: str
    beam_blockage_visibility_mask_digest: str
    spatial_correlation_block_digest: str
    geometry_contract_digest: str | None = None
    contract: str = "verification-observation-mask-evidence-v10"
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        score_shape = self.source_assignment_scores.shape
        frame_shape = score_shape[1:]
        if (
            self.contract not in {
                "verification-observation-mask-evidence-v4",
                "verification-observation-mask-evidence-v5",
                "verification-observation-mask-evidence-v6",
                "verification-observation-mask-evidence-v7",
                "verification-observation-mask-evidence-v9",
                "verification-observation-mask-evidence-v10",
            }
            or type(self.source_identity) is not VerificationObservationSourceIdentity
            or self.source_identity.contract
            != (
                "verification-observation-source-identity-v3"
                if self.contract
                in {
                    "verification-observation-mask-evidence-v5",
                    "verification-observation-mask-evidence-v6",
                    "verification-observation-mask-evidence-v7",
                    "verification-observation-mask-evidence-v9",
                    "verification-observation-mask-evidence-v10",
                }
                else "verification-observation-source-identity-v2"
            )
            or self.source_assignment_scores.ndim != 4
            or len(self.source_identity.valid_times) != frame_shape[0]
            or not self.source_assignment_scores.is_floating_point()
            or not bool(torch.all(torch.isfinite(self.source_assignment_scores)))
            or not bool(torch.all(self.source_assignment_scores >= 0.0))
            or self.source_availability_by_time.dtype is not torch.bool
            or self.source_availability_by_time.shape
            != torch.Size((score_shape[0], frame_shape[0]))
            or self.source_availability_by_time.device
            != self.source_assignment_scores.device
            or len(self.ordered_source_digests) != score_shape[0]
            or len(set(self.ordered_source_digests)) != score_shape[0]
            or len(self.source_identity.acquisition_valid_times_by_source)
            != score_shape[0]
        ):
            raise ValueError("verification observation mask evidence is invalid")
        if self.contract in {
            "verification-observation-mask-evidence-v5",
            "verification-observation-mask-evidence-v6",
            "verification-observation-mask-evidence-v7",
            "verification-observation-mask-evidence-v9",
            "verification-observation-mask-evidence-v10",
        }:
            if not isinstance(self.geometry_contract_digest, str):
                raise ValueError("verification geometry identity is missing")
            _require_sha256(
                "verification geometry contract",
                self.geometry_contract_digest,
            )
        elif self.geometry_contract_digest is not None:
            raise ValueError("legacy verification evidence cannot claim geometry")
        _require_sha256(
            "mosaic observation source registry artifact",
            self.source_registry_artifact_digest,
        )
        for source_digest in self.ordered_source_digests:
            _require_sha256("ordered observation source", source_digest)
        if (
            self.contract
            in {
                "verification-observation-mask-evidence-v9",
                "verification-observation-mask-evidence-v10",
            }
            and self.source_assignment_scores.dtype
            not in {torch.float32, torch.float64}
        ):
            raise ValueError(
                "scientific observation evidence requires float32 or float64"
            )
        for tensor in (
            self.reflectivity_dbz_by_source,
            self.detection_limit_dbz_by_source,
            self.acquisition_time_offset_seconds_by_source,
            self.range_km_by_source,
            self.elevation_deg_by_source,
            self.beam_blockage_fraction_by_source,
            self.attenuation_qc_score_by_source,
        ):
            if (
                tensor.shape != score_shape
                or not tensor.is_floating_point()
                or tensor.dtype != self.source_assignment_scores.dtype
                or tensor.device != self.source_assignment_scores.device
                or not bool(torch.all(torch.isfinite(tensor)))
            ):
                raise ValueError("verification observation mask evidence is invalid")
        if (
            bool(
                torch.any(
                    self.acquisition_time_offset_seconds_by_source > 0.0
                )
            )
            or not bool(torch.all(self.range_km_by_source >= 0.0))
            or not bool(
                torch.all(
                    (self.beam_blockage_fraction_by_source >= 0.0)
                    & (self.beam_blockage_fraction_by_source <= 1.0)
                )
            )
            or not bool(
                torch.all(
                    (self.attenuation_qc_score_by_source >= 0.0)
                    & (self.attenuation_qc_score_by_source <= 1.0)
                )
            )
            or self.observation_report_kind_by_source.dtype is not torch.uint8
            or self.observation_report_kind_by_source.shape != score_shape
            or self.observation_report_kind_by_source.device
            != self.source_assignment_scores.device
            or not bool(
                torch.all(
                    (self.observation_report_kind_by_source >= 0)
                    & (
                        self.observation_report_kind_by_source
                        <= int(
                            VerificationObservationReportKind.BELOW_DETECTION_CENSORED
                        )
                    )
                )
            )
        ):
            raise ValueError("verification observation mask evidence is invalid")
        for name in (
            "range_elevation_validity_domain_digest",
            "beam_blockage_visibility_mask_digest",
            "spatial_correlation_block_digest",
        ):
            _require_sha256(name, getattr(self, name))
        self.source_identity.validate_integrity()
        expected_upstream_digest = _verification_observation_upstream_artifact_digest(
            contract=(
                (
                    "typed-upstream-verification-observation-v10"
                    if self.contract == "verification-observation-mask-evidence-v10"
                    else (
                        "typed-upstream-verification-observation-v9"
                        if self.contract == "verification-observation-mask-evidence-v9"
                        else (
                            "typed-upstream-verification-observation-v7"
                            if self.contract == "verification-observation-mask-evidence-v7"
                            else (
                                "typed-upstream-verification-observation-v6"
                                if self.contract == "verification-observation-mask-evidence-v6"
                                else "typed-upstream-verification-observation-v5"
                            )
                        )
                    )
                )
                if self.contract
                in {
                    "verification-observation-mask-evidence-v5",
                    "verification-observation-mask-evidence-v6",
                    "verification-observation-mask-evidence-v7",
                    "verification-observation-mask-evidence-v9",
                    "verification-observation-mask-evidence-v10",
                }
                else "typed-upstream-verification-observation-v4"
            ),
            valid_times=self.source_identity.valid_times,
            acquisition_valid_times_by_source=(
                self.source_identity.acquisition_valid_times_by_source
            ),
            grid_contract_digest=self.source_identity.grid_contract_digest,
            radar_product_digest=self.source_identity.radar_product_digest,
            native_verification_source_identity_digest=(
                self.source_identity.native_verification_source_identity_digest
            ),
            source_registry_artifact_digest=(
                self.source_registry_artifact_digest
            ),
            ordered_source_digests=self.ordered_source_digests,
            reflectivity_dbz_by_source=self.reflectivity_dbz_by_source,
            detection_limit_dbz_by_source=self.detection_limit_dbz_by_source,
            acquisition_time_offset_seconds_by_source=(
                self.acquisition_time_offset_seconds_by_source
            ),
            observation_report_kind_by_source=(
                self.observation_report_kind_by_source
            ),
            source_assignment_scores=self.source_assignment_scores,
            source_availability_by_time=self.source_availability_by_time,
            range_km_by_source=self.range_km_by_source,
            elevation_deg_by_source=self.elevation_deg_by_source,
            beam_blockage_fraction_by_source=(
                self.beam_blockage_fraction_by_source
            ),
            attenuation_qc_score_by_source=(
                self.attenuation_qc_score_by_source
            ),
            range_elevation_validity_domain_digest=(
                self.range_elevation_validity_domain_digest
            ),
            beam_blockage_visibility_mask_digest=(
                self.beam_blockage_visibility_mask_digest
            ),
            spatial_correlation_block_digest=(
                self.spatial_correlation_block_digest
            ),
            geometry_contract_digest=self.geometry_contract_digest,
        )
        if (
            self.source_identity.upstream_verification_artifact_digest
            != expected_upstream_digest
        ):
            raise ValueError("verification source does not attest raw mask evidence")
        for name in (
            "reflectivity_dbz_by_source",
            "detection_limit_dbz_by_source",
            "acquisition_time_offset_seconds_by_source",
            "observation_report_kind_by_source",
            "source_assignment_scores",
            "source_availability_by_time",
            "range_km_by_source",
            "elevation_deg_by_source",
            "beam_blockage_fraction_by_source",
            "attenuation_qc_score_by_source",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        object.__setattr__(self, "evidence_digest", json_digest(self.payload))

    def validate_integrity(self) -> None:
        self.source_identity.validate_integrity()
        expected_upstream_digest = _verification_observation_upstream_artifact_digest(
            contract=(
                (
                    "typed-upstream-verification-observation-v10"
                    if self.contract == "verification-observation-mask-evidence-v10"
                    else (
                        "typed-upstream-verification-observation-v9"
                        if self.contract == "verification-observation-mask-evidence-v9"
                        else (
                            "typed-upstream-verification-observation-v7"
                            if self.contract == "verification-observation-mask-evidence-v7"
                            else (
                                "typed-upstream-verification-observation-v6"
                                if self.contract == "verification-observation-mask-evidence-v6"
                                else "typed-upstream-verification-observation-v5"
                            )
                        )
                    )
                )
                if self.contract
                in {
                    "verification-observation-mask-evidence-v5",
                    "verification-observation-mask-evidence-v6",
                    "verification-observation-mask-evidence-v7",
                    "verification-observation-mask-evidence-v9",
                    "verification-observation-mask-evidence-v10",
                }
                else "typed-upstream-verification-observation-v4"
            ),
            valid_times=self.source_identity.valid_times,
            acquisition_valid_times_by_source=(
                self.source_identity.acquisition_valid_times_by_source
            ),
            grid_contract_digest=self.source_identity.grid_contract_digest,
            radar_product_digest=self.source_identity.radar_product_digest,
            native_verification_source_identity_digest=(
                self.source_identity.native_verification_source_identity_digest
            ),
            source_registry_artifact_digest=(
                self.source_registry_artifact_digest
            ),
            ordered_source_digests=self.ordered_source_digests,
            reflectivity_dbz_by_source=self.reflectivity_dbz_by_source,
            detection_limit_dbz_by_source=self.detection_limit_dbz_by_source,
            acquisition_time_offset_seconds_by_source=(
                self.acquisition_time_offset_seconds_by_source
            ),
            observation_report_kind_by_source=(
                self.observation_report_kind_by_source
            ),
            source_assignment_scores=self.source_assignment_scores,
            source_availability_by_time=self.source_availability_by_time,
            range_km_by_source=self.range_km_by_source,
            elevation_deg_by_source=self.elevation_deg_by_source,
            beam_blockage_fraction_by_source=(
                self.beam_blockage_fraction_by_source
            ),
            attenuation_qc_score_by_source=(
                self.attenuation_qc_score_by_source
            ),
            range_elevation_validity_domain_digest=(
                self.range_elevation_validity_domain_digest
            ),
            beam_blockage_visibility_mask_digest=(
                self.beam_blockage_visibility_mask_digest
            ),
            spatial_correlation_block_digest=(
                self.spatial_correlation_block_digest
            ),
            geometry_contract_digest=self.geometry_contract_digest,
        )
        if (
            self.source_identity.upstream_verification_artifact_digest
            != expected_upstream_digest
            or self.evidence_digest != json_digest(self.payload)
        ):
            raise ValueError("verification observation mask evidence mismatch")

    def validate_against_registry(
        self,
        source_registry: MosaicObservationSourceRegistry,
    ) -> None:
        """Validate the entire source cube, including never-selected rows."""

        self.validate_integrity()
        source_registry.validate_integrity()
        expected_limits = _registered_detection_limit_field(
            source_registry=source_registry,
            range_km_by_source=self.range_km_by_source,
            elevation_deg_by_source=self.elevation_deg_by_source,
        )
        kinds = self.observation_report_kind_by_source
        detected = kinds == int(VerificationObservationReportKind.DETECTED_ECHO)
        clear = kinds == int(VerificationObservationReportKind.CONFIRMED_CLEAR)
        censored = kinds == int(
            VerificationObservationReportKind.BELOW_DETECTION_CENSORED
        )
        ages = _verification_acquisition_age_seconds_by_source(
            source_identity=self.source_identity,
            acquisition_time_offset_seconds_by_source=(
                self.acquisition_time_offset_seconds_by_source
            ),
        )
        if (
            source_registry.contract
            != (
                (
                    "mosaic-observation-source-registry-v6"
                    if self.contract
                    == "verification-observation-mask-evidence-v10"
                    else (
                        "mosaic-observation-source-registry-v5"
                        if self.contract
                        in {
                            "verification-observation-mask-evidence-v6",
                            "verification-observation-mask-evidence-v7",
                            "verification-observation-mask-evidence-v9",
                        }
                        else "mosaic-observation-source-registry-v4"
                    )
                )
                if self.contract
                in {
                    "verification-observation-mask-evidence-v5",
                    "verification-observation-mask-evidence-v6",
                    "verification-observation-mask-evidence-v7",
                    "verification-observation-mask-evidence-v9",
                    "verification-observation-mask-evidence-v10",
                }
                else "mosaic-observation-source-registry-v3"
            )
            or self.source_registry_artifact_digest
            != source_registry.registry_digest
            or self.ordered_source_digests
            != tuple(
                source.source_digest for source in source_registry.ordered_sources
            )
            or not bool(torch.equal(self.detection_limit_dbz_by_source, expected_limits))
            or bool(torch.any(detected & (self.reflectivity_dbz_by_source <= expected_limits)))
            or bool(torch.any(clear & (self.reflectivity_dbz_by_source >= expected_limits)))
            or bool(torch.any(censored & (self.reflectivity_dbz_by_source > expected_limits)))
            or bool(torch.any(ages < 0.0))
        ):
            raise ValueError(
                "verification source cube disagrees with its preregistered registry"
            )

    def validate_against_plan_registry_and_geometry(
        self,
        *,
        plan: VerificationObservationErrorPlan,
        source_registry: MosaicObservationSourceRegistry,
        geometry: RadarObservationGeometryContract,
    ) -> None:
        """Recompute all product-owned geometry and source-selection fields."""

        self.validate_against_registry(source_registry)
        source_registry.validate_against_plan(plan)
        geometry.validate_integrity()
        if (
            self.contract != "verification-observation-mask-evidence-v10"
            or plan.contract != "verification-observation-error-plan-v12"
            or self.source_identity.contract
            != "verification-observation-source-identity-v3"
            or self.source_identity.acquisition_timestamp_reference
            != plan.acquisition_timestamp_reference
            or self.geometry_contract_digest != geometry.geometry_digest
            or geometry.geometry_digest != plan.geometry_contract_digest
            or geometry.grid_contract_digest
            != self.source_identity.grid_contract_digest
            or source_registry.projected_crs_digest
            != geometry.projected_crs_digest
            or source_registry.metric_domain_digest
            != CURRENT_RADAR_METRIC_DOMAIN.digest
        ):
            raise ValueError("verification evidence generation is not current")
        expected_range, expected_elevation = (
            _registered_observation_geometry_fields(
                source_registry=source_registry,
                geometry=geometry,
                time_count=len(self.source_identity.valid_times),
                output_dtype=self.reflectivity_dbz_by_source.dtype,
            )
        )
        expected_scores = _product_owned_source_assignment_scores(
            plan=plan,
            valid_times=self.source_identity.valid_times,
            acquisition_valid_times_by_source=(
                self.source_identity.acquisition_valid_times_by_source
            ),
            acquisition_time_offset_seconds_by_source=(
                self.acquisition_time_offset_seconds_by_source
            ),
            source_availability_by_time=self.source_availability_by_time,
            range_km_by_source=expected_range,
            elevation_deg_by_source=expected_elevation,
            beam_blockage_fraction_by_source=(
                self.beam_blockage_fraction_by_source
            ),
            attenuation_qc_score_by_source=(
                self.attenuation_qc_score_by_source
            ),
        )
        if (
            not torch.equal(self.range_km_by_source, expected_range)
            or not torch.equal(
                self.elevation_deg_by_source,
                expected_elevation,
            )
            or not torch.equal(
                self.source_assignment_scores,
                expected_scores,
            )
        ):
            raise ValueError(
                "verification geometry/source selection is not product-derived"
            )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "source_identity_digest": self.source_identity.identity_digest,
            "source_registry_artifact_digest": (
                self.source_registry_artifact_digest
            ),
            "ordered_source_digests": list(self.ordered_source_digests),
            "reflectivity_dbz_by_source_digest": tensor_digest(
                self.reflectivity_dbz_by_source
            ),
            "detection_limit_dbz_by_source_digest": tensor_digest(
                self.detection_limit_dbz_by_source
            ),
            "acquisition_time_offset_seconds_by_source_digest": tensor_digest(
                self.acquisition_time_offset_seconds_by_source
            ),
            "observation_report_kind_by_source_digest": tensor_digest(
                self.observation_report_kind_by_source
            ),
            "source_assignment_scores_digest": tensor_digest(
                self.source_assignment_scores
            ),
            "source_availability_by_time_digest": tensor_digest(
                self.source_availability_by_time
            ),
            "range_km_by_source_digest": tensor_digest(
                self.range_km_by_source
            ),
            "elevation_deg_by_source_digest": tensor_digest(
                self.elevation_deg_by_source
            ),
            "beam_blockage_fraction_by_source_digest": tensor_digest(
                self.beam_blockage_fraction_by_source
            ),
            "attenuation_qc_score_by_source_digest": tensor_digest(
                self.attenuation_qc_score_by_source
            ),
            "range_elevation_validity_domain_digest": (
                self.range_elevation_validity_domain_digest
            ),
            "beam_blockage_visibility_mask_digest": (
                self.beam_blockage_visibility_mask_digest
            ),
            "spatial_correlation_block_digest": (
                self.spatial_correlation_block_digest
            ),
            **(
                {"geometry_contract_digest": self.geometry_contract_digest}
                if self.contract
                in {
                    "verification-observation-mask-evidence-v5",
                    "verification-observation-mask-evidence-v6",
                    "verification-observation-mask-evidence-v7",
                    "verification-observation-mask-evidence-v9",
                    "verification-observation-mask-evidence-v10",
                }
                else {}
            ),
        }

    @property
    def frames_dbz(self) -> Tensor:
        return _selected_verification_spatial_evidence(self)[2].detach().clone()

    @property
    def detection_limit_dbz(self) -> Tensor:
        return _selected_verification_spatial_evidence(self)[3].detach().clone()

    @property
    def acquisition_time_offset_seconds(self) -> Tensor:
        return _selected_verification_spatial_evidence(self)[4].detach().clone()

    @property
    def below_detection_reported(self) -> Tensor:
        return (
            _selected_verification_spatial_evidence(self)[5]
            == int(
                VerificationObservationReportKind.BELOW_DETECTION_CENSORED
            )
        ).detach().clone()

    @property
    def observation_report_kind(self) -> Tensor:
        return _selected_verification_spatial_evidence(self)[5].detach().clone()

    @classmethod
    def issue(
        cls,
        *,
        plan: VerificationObservationErrorPlan,
        geometry: RadarObservationGeometryContract,
        valid_times: tuple[str, ...],
        acquisition_valid_times_by_source: tuple[tuple[str, ...], ...],
        grid_contract_digest: str,
        radar_product_digest: str,
        native_verification_source_identity_digest: str,
        source_registry: MosaicObservationSourceRegistry,
        source_authority_id: str,
        source_authority_private_key: Ed25519PrivateKey,
        source_observed_at: str,
        reflectivity_dbz_by_source: Tensor,
        acquisition_time_offset_seconds_by_source: Tensor,
        observation_report_kind_by_source: Tensor,
        source_availability_by_time: Tensor,
        beam_blockage_fraction_by_source: Tensor,
        attenuation_qc_score_by_source: Tensor,
        range_elevation_validity_domain_digest: str,
        beam_blockage_visibility_mask_digest: str,
        spatial_correlation_block_digest: str,
    ) -> VerificationObservationMaskEvidence:
        canonical_valid_times = tuple(
            _canonical_verification_time(value) for value in valid_times
        )
        canonical_acquisition_times_by_source = tuple(
            tuple(_canonical_verification_time(value) for value in row)
            for row in acquisition_valid_times_by_source
        )
        if type(source_registry) is not MosaicObservationSourceRegistry:
            raise ValueError("mosaic observation source registry is required")
        source_registry.validate_integrity()
        source_registry.validate_against_plan(plan)
        geometry.validate_integrity()
        if (
            plan.contract != "verification-observation-error-plan-v12"
            or source_registry.contract
            != "mosaic-observation-source-registry-v6"
            or source_registry.projected_crs_digest
            != geometry.projected_crs_digest
            or geometry.geometry_digest != plan.geometry_contract_digest
            or geometry.grid_contract_digest != grid_contract_digest
        ):
            raise ValueError("verification geometry disagrees with its plan")
        ordered_source_digests = tuple(
            source.source_digest for source in source_registry.ordered_sources
        )
        range_km_by_source, elevation_deg_by_source = (
            _registered_observation_geometry_fields(
                source_registry=source_registry,
                geometry=geometry,
                time_count=len(canonical_valid_times),
                output_dtype=reflectivity_dbz_by_source.dtype,
            )
        )
        source_assignment_scores = _product_owned_source_assignment_scores(
            plan=plan,
            valid_times=canonical_valid_times,
            acquisition_valid_times_by_source=(
                canonical_acquisition_times_by_source
            ),
            acquisition_time_offset_seconds_by_source=(
                acquisition_time_offset_seconds_by_source
            ),
            source_availability_by_time=source_availability_by_time,
            range_km_by_source=range_km_by_source,
            elevation_deg_by_source=elevation_deg_by_source,
            beam_blockage_fraction_by_source=(
                beam_blockage_fraction_by_source
            ),
            attenuation_qc_score_by_source=attenuation_qc_score_by_source,
        )
        registered_limits = _registered_detection_limit_field(
            source_registry=source_registry,
            range_km_by_source=range_km_by_source,
            elevation_deg_by_source=elevation_deg_by_source,
        )
        upstream_digest = _verification_observation_upstream_artifact_digest(
            contract="typed-upstream-verification-observation-v10",
            valid_times=canonical_valid_times,
            acquisition_valid_times_by_source=(
                canonical_acquisition_times_by_source
            ),
            grid_contract_digest=grid_contract_digest,
            radar_product_digest=radar_product_digest,
            native_verification_source_identity_digest=(
                native_verification_source_identity_digest
            ),
            source_registry_artifact_digest=source_registry.registry_digest,
            ordered_source_digests=ordered_source_digests,
            reflectivity_dbz_by_source=reflectivity_dbz_by_source,
            detection_limit_dbz_by_source=registered_limits,
            acquisition_time_offset_seconds_by_source=(
                acquisition_time_offset_seconds_by_source
            ),
            observation_report_kind_by_source=(
                observation_report_kind_by_source
            ),
            source_assignment_scores=source_assignment_scores,
            source_availability_by_time=source_availability_by_time,
            range_km_by_source=range_km_by_source,
            elevation_deg_by_source=elevation_deg_by_source,
            beam_blockage_fraction_by_source=(
                beam_blockage_fraction_by_source
            ),
            attenuation_qc_score_by_source=(
                attenuation_qc_score_by_source
            ),
            range_elevation_validity_domain_digest=(
                range_elevation_validity_domain_digest
            ),
            beam_blockage_visibility_mask_digest=(
                beam_blockage_visibility_mask_digest
            ),
            spatial_correlation_block_digest=spatial_correlation_block_digest,
            geometry_contract_digest=geometry.geometry_digest,
        )
        source_identity = VerificationObservationSourceIdentity.issue(
            valid_times=canonical_valid_times,
            acquisition_valid_times_by_source=(
                canonical_acquisition_times_by_source
            ),
            grid_contract_digest=grid_contract_digest,
            radar_product_digest=radar_product_digest,
            native_verification_source_identity_digest=(
                native_verification_source_identity_digest
            ),
            upstream_verification_artifact_digest=upstream_digest,
            source_authority_id=source_authority_id,
            source_authority_private_key=source_authority_private_key,
            source_observed_at=source_observed_at,
            acquisition_timestamp_reference=cast(
                Literal["volume_end"],
                plan.acquisition_timestamp_reference,
            ),
        )
        evidence = cls(
            source_identity=source_identity,
            source_registry_artifact_digest=source_registry.registry_digest,
            ordered_source_digests=ordered_source_digests,
            reflectivity_dbz_by_source=reflectivity_dbz_by_source,
            detection_limit_dbz_by_source=registered_limits,
            acquisition_time_offset_seconds_by_source=(
                acquisition_time_offset_seconds_by_source
            ),
            observation_report_kind_by_source=(
                observation_report_kind_by_source
            ),
            source_assignment_scores=source_assignment_scores,
            source_availability_by_time=source_availability_by_time,
            range_km_by_source=range_km_by_source,
            elevation_deg_by_source=elevation_deg_by_source,
            beam_blockage_fraction_by_source=(
                beam_blockage_fraction_by_source
            ),
            attenuation_qc_score_by_source=(
                attenuation_qc_score_by_source
            ),
            range_elevation_validity_domain_digest=(
                range_elevation_validity_domain_digest
            ),
            beam_blockage_visibility_mask_digest=(
                beam_blockage_visibility_mask_digest
            ),
            spatial_correlation_block_digest=spatial_correlation_block_digest,
            geometry_contract_digest=geometry.geometry_digest,
        )
        evidence.validate_against_plan_registry_and_geometry(
            plan=plan,
            source_registry=source_registry,
            geometry=geometry,
        )
        return evidence


def _selected_verification_spatial_evidence(
    raw_evidence: VerificationObservationMaskEvidence,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    maximum_scores, selected_indices = torch.max(
        raw_evidence.source_assignment_scores,
        dim=0,
    )

    def selected(values: Tensor) -> Tensor:
        return torch.gather(
            values,
            0,
            selected_indices.unsqueeze(0),
        ).squeeze(0)

    return (
        maximum_scores,
        selected_indices,
        selected(raw_evidence.reflectivity_dbz_by_source),
        selected(raw_evidence.detection_limit_dbz_by_source),
        selected(raw_evidence.acquisition_time_offset_seconds_by_source),
        selected(raw_evidence.observation_report_kind_by_source),
        selected(raw_evidence.range_km_by_source),
        selected(raw_evidence.elevation_deg_by_source),
        selected(raw_evidence.beam_blockage_fraction_by_source),
        selected(raw_evidence.attenuation_qc_score_by_source),
    )


def _derive_verification_observation_masks(
    *,
    plan: VerificationObservationErrorPlan,
    raw_evidence: VerificationObservationMaskEvidence,
    source_registry: MosaicObservationSourceRegistry,
    geometry: RadarObservationGeometryContract,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
]:
    if (
        type(plan) is not VerificationObservationErrorPlan
        or plan.contract != "verification-observation-error-plan-v12"
        or plan.mask_derivation_algorithm_digest
        != OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST
        or type(raw_evidence) is not VerificationObservationMaskEvidence
        or raw_evidence.contract != "verification-observation-mask-evidence-v10"
        or type(source_registry) is not MosaicObservationSourceRegistry
        or source_registry.contract != "mosaic-observation-source-registry-v6"
        or type(geometry) is not RadarObservationGeometryContract
        or geometry.contract != "radar-observation-geometry-v6"
    ):
        raise ValueError("deterministic observation-mask derivation is invalid")
    plan.validate_integrity()
    raw_evidence.validate_against_plan_registry_and_geometry(
        plan=plan,
        source_registry=source_registry,
        geometry=geometry,
    )
    source_registry.validate_against_plan(plan)
    if (
        raw_evidence.source_identity.source_authority_id
        != plan.verification_source_authority_id
        or raw_evidence.source_identity.source_authority_public_key_hex
        != plan.verification_source_authority_public_key_hex
        or raw_evidence.source_assignment_scores.shape[0]
        != len(source_registry.ordered_sources)
        or raw_evidence.source_registry_artifact_digest
        != source_registry.registry_digest
        or raw_evidence.ordered_source_digests
        != tuple(
            source.source_digest for source in source_registry.ordered_sources
        )
    ):
        raise ValueError("observation source-assignment evidence disagrees with registry")
    (
        maximum_scores,
        selected_indices,
        selected_frames_dbz,
        selected_detection_limit_dbz,
        selected_acquisition_time_offset_seconds,
        selected_report_kind,
        selected_range_km,
        selected_elevation_deg,
        selected_beam_blockage_fraction,
        selected_attenuation_qc_score,
    ) = _selected_verification_spatial_evidence(raw_evidence)
    assigned = maximum_scores > 0.0
    source_map = torch.where(
        assigned,
        selected_indices,
        torch.full_like(selected_indices, -1),
    )
    availability = raw_evidence.source_availability_by_time[:, :, None, None]
    availability = availability.expand_as(
        raw_evidence.source_assignment_scores
    )
    selected_available = torch.gather(
        availability,
        0,
        selected_indices.unsqueeze(0),
    ).squeeze(0)
    acquisition_ages_by_source = _verification_acquisition_age_seconds_by_source(
        source_identity=raw_evidence.source_identity,
        acquisition_time_offset_seconds_by_source=(
            raw_evidence.acquisition_time_offset_seconds_by_source
        ),
    )
    selected_age_seconds = torch.gather(
        acquisition_ages_by_source,
        0,
        selected_indices.unsqueeze(0),
    ).squeeze(0)
    acquisition_time_valid = selected_age_seconds <= cast(
        float, plan.maximum_acquisition_age_seconds
    )
    spatial_metric_maximum_age_seconds = (
        cast(float, plan.spatial_metric_maximum_displacement_fraction_cells)
        * geometry.grid_spacing_m
        / cast(float, plan.spatial_metric_reference_speed_mps)
    )
    spatial_metric_valid = (
        selected_age_seconds <= spatial_metric_maximum_age_seconds
    )
    source_present = assigned & selected_available
    range_elevation_valid = (
        (selected_range_km <= cast(float, plan.maximum_range_km))
        & (
            selected_elevation_deg
            >= cast(float, plan.minimum_elevation_deg)
        )
        & (
            selected_elevation_deg
            <= cast(float, plan.maximum_elevation_deg)
        )
    )
    beam_blocked = (
        selected_beam_blockage_fraction
        > cast(float, plan.maximum_beam_blockage_fraction)
    )
    attenuation_qc_valid = (
        selected_attenuation_qc_score
        >= cast(float, plan.minimum_attenuation_qc_score)
    )
    confirmed_clear = (
        selected_report_kind
        == int(VerificationObservationReportKind.CONFIRMED_CLEAR)
    )
    below_detection_censored = (
        selected_report_kind
        == int(VerificationObservationReportKind.BELOW_DETECTION_CENSORED)
    )
    eligible = (
        source_present
        & range_elevation_valid
        & ~beam_blocked
        & acquisition_time_valid
        & attenuation_qc_valid
    )
    confirmed_clear &= eligible
    below_detection_censored &= eligible
    exported_source_map = (
        None
        if plan.radar_source_kind == "single_site"
        else source_map.detach().clone()
    )
    return (
        selected_frames_dbz,
        selected_detection_limit_dbz,
        selected_acquisition_time_offset_seconds,
        selected_age_seconds,
        source_present,
        range_elevation_valid,
        beam_blocked,
        acquisition_time_valid,
        attenuation_qc_valid,
        spatial_metric_valid,
        confirmed_clear,
        below_detection_censored,
        exported_source_map,
    )


@dataclass(frozen=True)
class VerificationObservationMaskDerivationArtifact:
    """Replayable raw-evidence-to-mask artifact for confirmatory research."""

    plan: VerificationObservationErrorPlan
    raw_evidence: VerificationObservationMaskEvidence
    source_registry: MosaicObservationSourceRegistry
    geometry: RadarObservationGeometryContract
    _selected_frames_dbz: Tensor = field(repr=False)
    _selected_detection_limit_dbz: Tensor = field(repr=False)
    _selected_acquisition_time_offset_seconds: Tensor = field(repr=False)
    _selected_acquisition_age_seconds: Tensor = field(repr=False)
    _source_present_mask: Tensor = field(repr=False)
    _range_elevation_valid_mask: Tensor = field(repr=False)
    _beam_blocked_mask: Tensor = field(repr=False)
    _acquisition_time_valid_mask: Tensor = field(repr=False)
    _attenuation_qc_valid_mask: Tensor = field(repr=False)
    _spatial_metric_valid_mask: Tensor = field(repr=False)
    _confirmed_clear_mask: Tensor = field(repr=False)
    _below_detection_censored_mask: Tensor = field(repr=False)
    _source_radar_index_map: Tensor | None = field(repr=False)
    contract: str = "verification-observation-mask-derivation-artifact-v10"
    artifact_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract not in {
            "verification-observation-mask-derivation-artifact-v9",
            "verification-observation-mask-derivation-artifact-v10",
        }:
            raise ValueError("verification observation mask artifact is invalid")
        expected = _derive_verification_observation_masks(
            plan=self.plan,
            raw_evidence=self.raw_evidence,
            source_registry=self.source_registry,
            geometry=self.geometry,
        )
        actual = self._output_tensors
        if any(
            (left is None) != (right is None)
            or (
                left is not None
                and right is not None
                and not bool(torch.equal(left, right))
            )
            for left, right in zip(expected, actual)
        ):
            raise ValueError("verification observation mask replay mismatch")
        for name in (
            "_selected_frames_dbz",
            "_selected_detection_limit_dbz",
            "_selected_acquisition_time_offset_seconds",
            "_selected_acquisition_age_seconds",
            "_source_present_mask",
            "_range_elevation_valid_mask",
            "_beam_blocked_mask",
            "_acquisition_time_valid_mask",
            "_attenuation_qc_valid_mask",
            "_spatial_metric_valid_mask",
            "_confirmed_clear_mask",
            "_below_detection_censored_mask",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        if self._source_radar_index_map is not None:
            object.__setattr__(
                self,
                "_source_radar_index_map",
                self._source_radar_index_map.detach().clone(),
            )
        object.__setattr__(self, "artifact_digest", json_digest(self.payload))

    @property
    def _output_tensors(
        self,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
    ]:
        return (
            self._selected_frames_dbz,
            self._selected_detection_limit_dbz,
            self._selected_acquisition_time_offset_seconds,
            self._selected_acquisition_age_seconds,
            self._source_present_mask,
            self._range_elevation_valid_mask,
            self._beam_blocked_mask,
            self._acquisition_time_valid_mask,
            self._attenuation_qc_valid_mask,
            self._spatial_metric_valid_mask,
            self._confirmed_clear_mask,
            self._below_detection_censored_mask,
            self._source_radar_index_map,
        )

    def validate_replay(self) -> None:
        expected = _derive_verification_observation_masks(
            plan=self.plan,
            raw_evidence=self.raw_evidence,
            source_registry=self.source_registry,
            geometry=self.geometry,
        )
        if (
            self.artifact_digest != json_digest(self.payload)
            or any(
                (left is None) != (right is None)
                or (
                    left is not None
                    and right is not None
                    and not bool(torch.equal(left, right))
                )
                for left, right in zip(expected, self._output_tensors)
            )
        ):
            raise ValueError("verification observation mask replay mismatch")

    def _snapshot(self, value: Tensor | None) -> Tensor | None:
        self.validate_replay()
        return None if value is None else value.detach().clone()

    @property
    def source_present_mask(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._source_present_mask))

    @property
    def selected_frames_dbz(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._selected_frames_dbz))

    @property
    def selected_detection_limit_dbz(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._selected_detection_limit_dbz))

    @property
    def selected_acquisition_time_offset_seconds(self) -> Tensor:
        return cast(
            Tensor,
            self._snapshot(self._selected_acquisition_time_offset_seconds),
        )

    @property
    def selected_acquisition_age_seconds(self) -> Tensor:
        return cast(
            Tensor,
            self._snapshot(self._selected_acquisition_age_seconds),
        )

    @property
    def range_elevation_valid_mask(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._range_elevation_valid_mask))

    @property
    def beam_blocked_mask(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._beam_blocked_mask))

    @property
    def attenuation_qc_valid_mask(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._attenuation_qc_valid_mask))

    @property
    def acquisition_time_valid_mask(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._acquisition_time_valid_mask))

    @property
    def spatial_metric_valid_mask(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._spatial_metric_valid_mask))

    @property
    def confirmed_clear_mask(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._confirmed_clear_mask))

    @property
    def below_detection_censored_mask(self) -> Tensor:
        return cast(Tensor, self._snapshot(self._below_detection_censored_mask))

    @property
    def source_radar_index_map(self) -> Tensor | None:
        return self._snapshot(self._source_radar_index_map)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "plan_digest": self.plan.plan_digest,
            "raw_evidence_digest": self.raw_evidence.evidence_digest,
            "source_registry_digest": self.source_registry.registry_digest,
            "geometry_contract_digest": self.geometry.geometry_digest,
            "mask_derivation_algorithm_digest": (
                OBSERVATION_MASK_DERIVATION_ALGORITHM_V10_DIGEST
                if self.contract
                == "verification-observation-mask-derivation-artifact-v10"
                else OBSERVATION_MASK_DERIVATION_ALGORITHM_V9_DIGEST
            ),
            "selected_frames_dbz_digest": tensor_digest(
                self._selected_frames_dbz
            ),
            "selected_detection_limit_dbz_digest": tensor_digest(
                self._selected_detection_limit_dbz
            ),
            "selected_acquisition_time_offset_seconds_digest": tensor_digest(
                self._selected_acquisition_time_offset_seconds
            ),
            "selected_acquisition_age_seconds_digest": tensor_digest(
                self._selected_acquisition_age_seconds
            ),
            "source_present_mask_digest": tensor_digest(
                self._source_present_mask
            ),
            "range_elevation_valid_mask_digest": tensor_digest(
                self._range_elevation_valid_mask
            ),
            "beam_blocked_mask_digest": tensor_digest(self._beam_blocked_mask),
            "acquisition_time_valid_mask_digest": tensor_digest(
                self._acquisition_time_valid_mask
            ),
            "attenuation_qc_valid_mask_digest": tensor_digest(
                self._attenuation_qc_valid_mask
            ),
            "spatial_metric_valid_mask_digest": tensor_digest(
                self._spatial_metric_valid_mask
            ),
            "confirmed_clear_mask_digest": tensor_digest(
                self._confirmed_clear_mask
            ),
            "below_detection_censored_mask_digest": tensor_digest(
                self._below_detection_censored_mask
            ),
            "source_radar_index_map_digest": (
                None
                if self._source_radar_index_map is None
                else tensor_digest(self._source_radar_index_map)
            ),
        }


def derive_verification_observation_masks(
    *,
    plan: VerificationObservationErrorPlan,
    raw_evidence: VerificationObservationMaskEvidence,
    source_registry: MosaicObservationSourceRegistry,
    geometry: RadarObservationGeometryContract,
) -> VerificationObservationMaskDerivationArtifact:
    """Derive and seal product-owned observation masks from raw evidence."""

    derived = _derive_verification_observation_masks(
        plan=plan,
        raw_evidence=raw_evidence,
        source_registry=source_registry,
        geometry=geometry,
    )
    return VerificationObservationMaskDerivationArtifact(
        plan=plan,
        raw_evidence=raw_evidence,
        source_registry=source_registry,
        geometry=geometry,
        _selected_frames_dbz=derived[0],
        _selected_detection_limit_dbz=derived[1],
        _selected_acquisition_time_offset_seconds=derived[2],
        _selected_acquisition_age_seconds=derived[3],
        _source_present_mask=derived[4],
        _range_elevation_valid_mask=derived[5],
        _beam_blocked_mask=derived[6],
        _acquisition_time_valid_mask=derived[7],
        _attenuation_qc_valid_mask=derived[8],
        _spatial_metric_valid_mask=derived[9],
        _confirmed_clear_mask=derived[10],
        _below_detection_censored_mask=derived[11],
        _source_radar_index_map=derived[12],
    )


@dataclass(frozen=True)
class VerificationObservationDerivationInputs:
    """Typed raw inputs consumed by the deterministic scientific derivation."""

    frames_dbz: Tensor
    source_present_mask: Tensor
    range_elevation_valid_mask: Tensor
    beam_blocked_mask: Tensor
    acquisition_time_valid_mask: Tensor
    attenuation_qc_valid_mask: Tensor
    confirmed_clear_mask: Tensor
    below_detection_censored_mask: Tensor
    upstream_verification_artifact_digest: str
    range_elevation_validity_domain_digest: str
    beam_blockage_visibility_mask_digest: str
    spatial_correlation_block_digest: str
    spatial_metric_valid_mask: Tensor | None = None
    detection_limit_dbz: Tensor | None = None
    acquisition_time_offset_seconds: Tensor | None = None
    acquisition_age_seconds: Tensor | None = None
    source_radar_index_map: Tensor | None = None
    source_identity: VerificationObservationSourceIdentity | None = None
    mask_derivation: VerificationObservationMaskDerivationArtifact | None = None
    contract: str = "verification-observation-derivation-inputs-v1"
    raw_verification_identity_digest: str = field(init=False)
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        masks = (
            self.source_present_mask,
            self.range_elevation_valid_mask,
            self.beam_blocked_mask,
            self.acquisition_time_valid_mask,
            self.attenuation_qc_valid_mask,
            self.confirmed_clear_mask,
            self.below_detection_censored_mask,
        )
        if (
            self.contract
            not in {
                "verification-observation-derivation-inputs-v1",
                "verification-observation-derivation-inputs-v2",
                "verification-observation-derivation-inputs-v3",
                "verification-observation-derivation-inputs-v4",
                "verification-observation-derivation-inputs-v5",
                "verification-observation-derivation-inputs-v6",
                "verification-observation-derivation-inputs-v7",
                "verification-observation-derivation-inputs-v8",
                "verification-observation-derivation-inputs-v10",
                "verification-observation-derivation-inputs-v11",
            }
            or self.frames_dbz.ndim != 3
            or not self.frames_dbz.is_floating_point()
            or not bool(torch.all(torch.isfinite(self.frames_dbz)))
            or any(
                mask.dtype is not torch.bool
                or mask.shape != self.frames_dbz.shape
                or mask.device != self.frames_dbz.device
                for mask in masks
            )
            or (
                self.spatial_metric_valid_mask is not None
                and (
                    self.spatial_metric_valid_mask.dtype is not torch.bool
                    or self.spatial_metric_valid_mask.shape != self.frames_dbz.shape
                    or self.spatial_metric_valid_mask.device
                    != self.frames_dbz.device
                )
            )
            or (
                self.source_radar_index_map is not None
                and (
                    self.source_radar_index_map.dtype is not torch.int64
                    or self.source_radar_index_map.shape != self.frames_dbz.shape
                    or self.source_radar_index_map.device != self.frames_dbz.device
                )
            )
        ):
            raise ValueError("verification observation derivation inputs are invalid")
        if self.contract == "verification-observation-derivation-inputs-v1":
            if (
                self.source_identity is not None
                or self.mask_derivation is not None
                or self.detection_limit_dbz is not None
                or self.acquisition_time_offset_seconds is not None
                or self.acquisition_age_seconds is not None
                or self.spatial_metric_valid_mask is not None
            ):
                raise ValueError("legacy derivation inputs cannot claim mask replay")
        elif self.contract in {
            "verification-observation-derivation-inputs-v2",
            "verification-observation-derivation-inputs-v3",
            "verification-observation-derivation-inputs-v4",
            "verification-observation-derivation-inputs-v5",
            "verification-observation-derivation-inputs-v6",
            "verification-observation-derivation-inputs-v7",
            "verification-observation-derivation-inputs-v8",
        }:
            raise ValueError("legacy observation derivation inputs are audit-only")
        else:
            if (
                type(self.source_identity) is not VerificationObservationSourceIdentity
                or type(self.mask_derivation)
                is not VerificationObservationMaskDerivationArtifact
                or self.mask_derivation.contract
                != (
                    "verification-observation-mask-derivation-artifact-v10"
                    if self.contract
                    == "verification-observation-derivation-inputs-v11"
                    else "verification-observation-mask-derivation-artifact-v9"
                )
                or self.detection_limit_dbz is None
                or self.acquisition_time_offset_seconds is None
                or self.acquisition_age_seconds is None
                or self.spatial_metric_valid_mask is None
                or self.detection_limit_dbz.shape != self.frames_dbz.shape
                or self.acquisition_time_offset_seconds.shape
                != self.frames_dbz.shape
                or self.acquisition_age_seconds.shape != self.frames_dbz.shape
                or self.detection_limit_dbz.dtype != self.frames_dbz.dtype
                or self.acquisition_time_offset_seconds.dtype
                != self.frames_dbz.dtype
                or self.acquisition_age_seconds.dtype != self.frames_dbz.dtype
                or self.detection_limit_dbz.device != self.frames_dbz.device
                or self.acquisition_time_offset_seconds.device
                != self.frames_dbz.device
                or self.acquisition_age_seconds.device != self.frames_dbz.device
                or not bool(torch.all(torch.isfinite(self.detection_limit_dbz)))
                or not bool(
                    torch.all(
                        torch.isfinite(self.acquisition_time_offset_seconds)
                    )
                )
                or not bool(torch.all(torch.isfinite(self.acquisition_age_seconds)))
                or bool(torch.any(self.acquisition_time_offset_seconds > 0.0))
                or bool(torch.any(self.acquisition_age_seconds < 0.0))
            ):
                raise ValueError("mask derivation evidence is required")
            source_identity = cast(
                VerificationObservationSourceIdentity,
                self.source_identity,
            )
            mask_derivation = cast(
                VerificationObservationMaskDerivationArtifact,
                self.mask_derivation,
            )
            source_identity.validate_integrity()
            mask_derivation.validate_replay()
            raw_evidence = mask_derivation.raw_evidence
            derived_values = (
                mask_derivation.selected_frames_dbz,
                mask_derivation.selected_detection_limit_dbz,
                mask_derivation.selected_acquisition_time_offset_seconds,
                mask_derivation.selected_acquisition_age_seconds,
                mask_derivation.source_present_mask,
                mask_derivation.range_elevation_valid_mask,
                mask_derivation.beam_blocked_mask,
                mask_derivation.acquisition_time_valid_mask,
                mask_derivation.attenuation_qc_valid_mask,
                mask_derivation.spatial_metric_valid_mask,
                mask_derivation.confirmed_clear_mask,
                mask_derivation.below_detection_censored_mask,
                mask_derivation.source_radar_index_map,
            )
            supplied_values = (
                self.frames_dbz,
                self.detection_limit_dbz,
                self.acquisition_time_offset_seconds,
                self.acquisition_age_seconds,
                self.source_present_mask,
                self.range_elevation_valid_mask,
                self.beam_blocked_mask,
                self.acquisition_time_valid_mask,
                self.attenuation_qc_valid_mask,
                self.spatial_metric_valid_mask,
                self.confirmed_clear_mask,
                self.below_detection_censored_mask,
                self.source_radar_index_map,
            )
            if (
                source_identity.identity_digest
                != raw_evidence.source_identity.identity_digest
                or self.upstream_verification_artifact_digest
                != source_identity.upstream_verification_artifact_digest
                or self.range_elevation_validity_domain_digest
                != raw_evidence.range_elevation_validity_domain_digest
                or self.beam_blockage_visibility_mask_digest
                != raw_evidence.beam_blockage_visibility_mask_digest
                or self.spatial_correlation_block_digest
                != raw_evidence.spatial_correlation_block_digest
                or any(
                    (expected is None) != (actual is None)
                    or (
                        expected is not None
                        and actual is not None
                        and not bool(torch.equal(expected, actual))
                    )
                    for expected, actual in zip(
                        derived_values,
                        supplied_values,
                    )
                )
            ):
                raise ValueError(
                    "verification derivation inputs disagree with mask replay"
                )
        for name in (
            "upstream_verification_artifact_digest",
            "range_elevation_validity_domain_digest",
            "beam_blockage_visibility_mask_digest",
            "spatial_correlation_block_digest",
        ):
            _require_sha256(name, getattr(self, name))
        frames = self.frames_dbz.detach().clone()
        object.__setattr__(self, "frames_dbz", frames)
        if self.detection_limit_dbz is not None:
            object.__setattr__(
                self,
                "detection_limit_dbz",
                self.detection_limit_dbz.detach().clone(),
            )
        if self.acquisition_time_offset_seconds is not None:
            object.__setattr__(
                self,
                "acquisition_time_offset_seconds",
                self.acquisition_time_offset_seconds.detach().clone(),
            )
        if self.acquisition_age_seconds is not None:
            object.__setattr__(
                self,
                "acquisition_age_seconds",
                self.acquisition_age_seconds.detach().clone(),
            )
        for name in (
            "source_present_mask",
            "range_elevation_valid_mask",
            "beam_blocked_mask",
            "acquisition_time_valid_mask",
            "attenuation_qc_valid_mask",
            "confirmed_clear_mask",
            "below_detection_censored_mask",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        if self.spatial_metric_valid_mask is not None:
            object.__setattr__(
                self,
                "spatial_metric_valid_mask",
                self.spatial_metric_valid_mask.detach().clone(),
            )
        if self.source_radar_index_map is not None:
            object.__setattr__(
                self,
                "source_radar_index_map",
                self.source_radar_index_map.detach().clone(),
            )
        object.__setattr__(
            self,
            "raw_verification_identity_digest",
            json_digest(self.identity_payload),
        )
        object.__setattr__(self, "content_digest", json_digest(self.payload))

    def validate_integrity(self) -> None:
        if self.contract in {
            "verification-observation-derivation-inputs-v10",
            "verification-observation-derivation-inputs-v11",
        }:
            if (
                type(self.source_identity) is not VerificationObservationSourceIdentity
                or type(self.mask_derivation)
                is not VerificationObservationMaskDerivationArtifact
            ):
                raise ValueError("verification observation derivation input mismatch")
            self.source_identity.validate_integrity()
            self.mask_derivation.validate_replay()
            expected_values = (
                *(
                    self.mask_derivation.selected_frames_dbz,
                    self.mask_derivation.selected_detection_limit_dbz,
                    self.mask_derivation.selected_acquisition_time_offset_seconds,
                    self.mask_derivation.selected_acquisition_age_seconds,
                ),
                self.mask_derivation.source_present_mask,
                self.mask_derivation.range_elevation_valid_mask,
                self.mask_derivation.beam_blocked_mask,
                self.mask_derivation.acquisition_time_valid_mask,
                self.mask_derivation.attenuation_qc_valid_mask,
                self.mask_derivation.spatial_metric_valid_mask,
                self.mask_derivation.confirmed_clear_mask,
                self.mask_derivation.below_detection_censored_mask,
                self.mask_derivation.source_radar_index_map,
            )
            actual_values = (
                *(
                    self.frames_dbz,
                    self.detection_limit_dbz,
                    self.acquisition_time_offset_seconds,
                    self.acquisition_age_seconds,
                ),
                self.source_present_mask,
                self.range_elevation_valid_mask,
                self.beam_blocked_mask,
                self.acquisition_time_valid_mask,
                self.attenuation_qc_valid_mask,
                self.spatial_metric_valid_mask,
                self.confirmed_clear_mask,
                self.below_detection_censored_mask,
                self.source_radar_index_map,
            )
            if any(
                (expected is None) != (actual is None)
                or (
                    expected is not None
                    and actual is not None
                    and not bool(torch.equal(expected, actual))
                )
                for expected, actual in zip(expected_values, actual_values)
            ):
                raise ValueError("verification observation derivation input mismatch")
        if (
            self.raw_verification_identity_digest
            != json_digest(self.identity_payload)
            or self.content_digest != json_digest(self.payload)
        ):
            raise ValueError("verification observation derivation input mismatch")

    @property
    def identity_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract": "typed-verification-observation-source-v1",
            "frames_dbz_digest": tensor_digest(self.frames_dbz),
            "source_present_mask_digest": tensor_digest(self.source_present_mask),
            "range_elevation_valid_mask_digest": tensor_digest(
                self.range_elevation_valid_mask
            ),
            "beam_blocked_mask_digest": tensor_digest(self.beam_blocked_mask),
            "attenuation_qc_valid_mask_digest": tensor_digest(
                self.attenuation_qc_valid_mask
            ),
            "below_detection_censored_mask_digest": tensor_digest(
                self.below_detection_censored_mask
            ),
            "source_radar_index_map_digest": (
                None
                if self.source_radar_index_map is None
                else tensor_digest(self.source_radar_index_map)
            ),
            "upstream_verification_artifact_digest": (
                self.upstream_verification_artifact_digest
            ),
            "range_elevation_validity_domain_digest": (
                self.range_elevation_validity_domain_digest
            ),
            "beam_blockage_visibility_mask_digest": (
                self.beam_blockage_visibility_mask_digest
            ),
            "spatial_correlation_block_digest": (
                self.spatial_correlation_block_digest
            ),
        }
        if self.contract in {
            "verification-observation-derivation-inputs-v10",
            "verification-observation-derivation-inputs-v11",
        }:
            source_identity = cast(
                VerificationObservationSourceIdentity,
                self.source_identity,
            )
            mask_derivation = cast(
                VerificationObservationMaskDerivationArtifact,
                self.mask_derivation,
            )
            generation_payload: dict[str, object] = {
                    "contract": (
                        "typed-verification-observation-source-v10"
                        if self.contract
                        == "verification-observation-derivation-inputs-v11"
                        else "typed-verification-observation-source-v9"
                    ),
                    "verification_source_identity_digest": (
                        source_identity.identity_digest
                    ),
                    "valid_times": list(source_identity.valid_times),
                    "grid_contract_digest": source_identity.grid_contract_digest,
                    "radar_product_digest": source_identity.radar_product_digest,
                    "native_verification_source_identity_digest": (
                        source_identity.native_verification_source_identity_digest
                    ),
                    "source_acquisition_time_identity_digest": (
                        source_identity.source_acquisition_time_identity_digest
                    ),
                    "mask_derivation_artifact_digest": (
                        mask_derivation.artifact_digest
                    ),
                }
            generation_payload.update(
                {
                        "detection_limit_dbz_digest": tensor_digest(
                            cast(Tensor, self.detection_limit_dbz)
                        ),
                        "acquisition_time_offset_seconds_digest": tensor_digest(
                            cast(Tensor, self.acquisition_time_offset_seconds)
                        ),
                        "acquisition_age_seconds_digest": tensor_digest(
                            cast(Tensor, self.acquisition_age_seconds)
                        ),
                        "acquisition_time_valid_mask_digest": tensor_digest(
                            self.acquisition_time_valid_mask
                        ),
                        "spatial_metric_valid_mask_digest": tensor_digest(
                            cast(Tensor, self.spatial_metric_valid_mask)
                        ),
                        "confirmed_clear_mask_digest": tensor_digest(
                            self.confirmed_clear_mask
                        ),
                }
            )
            payload.update(generation_payload)
        return payload

    @property
    def payload(self) -> dict[str, object]:
        payload = self.identity_payload | {
            "contract": self.contract,
            "raw_verification_identity_digest": (
                self.raw_verification_identity_digest
            ),
        }
        if self.contract in {
            "verification-observation-derivation-inputs-v10",
            "verification-observation-derivation-inputs-v11",
        }:
            source_identity = cast(
                VerificationObservationSourceIdentity,
                self.source_identity,
            )
            mask_derivation = cast(
                VerificationObservationMaskDerivationArtifact,
                self.mask_derivation,
            )
            payload.update(
                {
                    "source_identity": source_identity.payload
                    | {"identity_digest": source_identity.identity_digest},
                    "mask_derivation_artifact_digest": (
                        mask_derivation.artifact_digest
                    ),
                }
            )
        return payload

    @classmethod
    def from_mask_derivation(
        cls,
        artifact: VerificationObservationMaskDerivationArtifact,
    ) -> VerificationObservationDerivationInputs:
        """Create current inputs only from a replayed source composition."""

        if type(artifact) is not VerificationObservationMaskDerivationArtifact:
            raise ValueError("mask derivation artifact is required")
        artifact.validate_replay()
        raw_evidence = artifact.raw_evidence
        return cls(
            frames_dbz=artifact.selected_frames_dbz,
            detection_limit_dbz=artifact.selected_detection_limit_dbz,
            acquisition_time_offset_seconds=(
                artifact.selected_acquisition_time_offset_seconds
            ),
            acquisition_age_seconds=(
                artifact.selected_acquisition_age_seconds
            ),
            source_present_mask=artifact.source_present_mask,
            range_elevation_valid_mask=artifact.range_elevation_valid_mask,
            beam_blocked_mask=artifact.beam_blocked_mask,
            acquisition_time_valid_mask=artifact.acquisition_time_valid_mask,
            attenuation_qc_valid_mask=artifact.attenuation_qc_valid_mask,
            spatial_metric_valid_mask=artifact.spatial_metric_valid_mask,
            confirmed_clear_mask=artifact.confirmed_clear_mask,
            below_detection_censored_mask=(
                artifact.below_detection_censored_mask
            ),
            source_radar_index_map=artifact.source_radar_index_map,
            upstream_verification_artifact_digest=(
                raw_evidence.source_identity.upstream_verification_artifact_digest
            ),
            range_elevation_validity_domain_digest=(
                raw_evidence.range_elevation_validity_domain_digest
            ),
            beam_blockage_visibility_mask_digest=(
                raw_evidence.beam_blockage_visibility_mask_digest
            ),
            spatial_correlation_block_digest=(
                raw_evidence.spatial_correlation_block_digest
            ),
            source_identity=raw_evidence.source_identity,
            mask_derivation=artifact,
            contract="verification-observation-derivation-inputs-v11",
        )


class VerificationCellState(IntEnum):
    """Per-cell observation semantics retained for scientific verification."""

    OBSERVED_CLEAR = 0
    OBSERVED_ECHO = 1
    SOURCE_MISSING = 2
    QC_INVALID = 3
    BEAM_BLOCKED = 4
    BELOW_DETECTION_CENSORED = 5
    MOSAIC_SOURCE_UNASSIGNED = 6
    STALE_ACQUISITION = 7


def _validate_verification_cell_states(
    *,
    frames_dbz: Tensor,
    valid_mask: Tensor,
    quality_weight: Tensor,
    observation_std_dbz: Tensor,
    observation_state_code: Tensor,
    minimum_detectable_echo_dbz: float,
    radar_source_kind: Literal["single_site", "mosaic"],
    source_radar_index_map: Tensor | None,
    detection_limit_dbz: Tensor | None = None,
) -> None:
    if (
        observation_state_code.dtype is not torch.uint8
        or observation_state_code.shape != frames_dbz.shape
        or observation_state_code.device != frames_dbz.device
    ):
        raise ValueError("verification observation-state tensor is invalid")
    detection_threshold = (
        torch.full_like(frames_dbz, minimum_detectable_echo_dbz)
        if detection_limit_dbz is None
        else detection_limit_dbz
    )
    if (
        detection_threshold.shape != frames_dbz.shape
        or detection_threshold.dtype != frames_dbz.dtype
        or detection_threshold.device != frames_dbz.device
        or not bool(torch.all(torch.isfinite(detection_threshold)))
    ):
        raise ValueError("verification detection-limit tensor is invalid")
    observed_clear = observation_state_code == VerificationCellState.OBSERVED_CLEAR
    observed_echo = observation_state_code == VerificationCellState.OBSERVED_ECHO
    censored = (
        observation_state_code
        == VerificationCellState.BELOW_DETECTION_CENSORED
    )
    recognized = torch.zeros_like(valid_mask)
    for state in VerificationCellState:
        recognized |= observation_state_code == state
    expected_valid = observed_clear | observed_echo | censored
    invalid = ~expected_valid
    mosaic_unassigned = (
        observation_state_code
        == VerificationCellState.MOSAIC_SOURCE_UNASSIGNED
    )
    if radar_source_kind == "single_site":
        source_semantics_valid = (
            source_radar_index_map is None
            and not bool(torch.any(mosaic_unassigned))
        )
    else:
        source_semantics_valid = (
            source_radar_index_map is not None
            and source_radar_index_map.dtype is torch.int64
            and source_radar_index_map.shape == frames_dbz.shape
            and source_radar_index_map.device == frames_dbz.device
            and bool(torch.all(source_radar_index_map >= -1))
            and bool(
                torch.equal(
                    source_radar_index_map == -1,
                    mosaic_unassigned,
                )
            )
            and bool(
                torch.all(
                    source_radar_index_map.masked_select(expected_valid) >= 0
                )
            )
        )
    if (
        not source_semantics_valid
        or not bool(torch.all(recognized))
        or not bool(torch.equal(valid_mask, expected_valid))
        or bool(torch.any(observed_clear & (frames_dbz >= detection_threshold)))
        or bool(torch.any(observed_echo & (frames_dbz <= detection_threshold)))
        or bool(torch.any(censored & (frames_dbz > detection_threshold)))
        or not bool(torch.all(quality_weight.masked_select(invalid) == 0.0))
        or not bool(torch.all(observation_std_dbz.masked_select(invalid) == 0.0))
        or not bool(torch.all(observation_std_dbz.masked_select(expected_valid) > 0.0))
    ):
        raise ValueError("verification observation-state semantics are invalid")


@dataclass(frozen=True)
class VerificationObservationErrorContract:
    """Source-aware error semantics and exact weighting tensors for scoring."""

    radar_source_kind: Literal["single_site", "mosaic"]
    source_calibration_epochs: tuple[tuple[str, str], ...]
    range_elevation_validity_domain_digest: str
    beam_blockage_visibility_mask_digest: str
    attenuation_qc_digest: str
    censoring_rule_digest: str
    spatial_correlation_block_digest: str
    quality_weight_interpretation_digest: str
    observation_error_model_digest: str
    minimum_detectable_echo_dbz: float
    observation_error_reference_std_dbz: float
    valid_mask_digest: str
    quality_weight_digest: str
    observation_std_dbz_digest: str
    observation_state_code_digest: str
    observation_error_plan_digest: str
    source_radar_index_map_digest: str | None = None
    detection_limit_dbz_digest: str | None = None
    acquisition_time_offset_seconds_digest: str | None = None
    acquisition_age_seconds_digest: str | None = None
    spatial_metric_valid_mask_digest: str | None = None
    source_registry_digest: str | None = None
    calibration_registry_digest: str | None = None
    observation_error_derivation_digest: str | None = None
    observation_mask_derivation_digest: str | None = None
    verification_source_identity_digest: str | None = None
    source_acquisition_time_identity_digest: str | None = None
    spatial_correlation_role: Literal["diagnostic_only"] = "diagnostic_only"
    missing_data_taxonomy: tuple[str, ...] = (
        "observed_clear",
        "observed_echo",
        "source_missing",
        "qc_invalid",
        "beam_blocked",
        "below_detection_censored",
        "mosaic_source_unassigned",
    )
    metric_weight_rule: str = "quality-times-normalized-inverse-variance-v1"
    contract: str = "verification-observation-error-contract-v3"
    contract_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract not in {
                "verification-observation-error-contract-v3",
                "verification-observation-error-contract-v4",
                "verification-observation-error-contract-v5",
                "verification-observation-error-contract-v6",
                "verification-observation-error-contract-v7",
                "verification-observation-error-contract-v8",
                "verification-observation-error-contract-v9",
                "verification-observation-error-contract-v10",
                "verification-observation-error-contract-v11",
                "verification-observation-error-contract-v13",
                "verification-observation-error-contract-v14",
            }
            or self.radar_source_kind not in {"single_site", "mosaic"}
            or self.metric_weight_rule
            != "quality-times-normalized-inverse-variance-v1"
            or self.spatial_correlation_role != "diagnostic_only"
            or self.missing_data_taxonomy
            != (
                (
                    "observed_clear",
                    "observed_echo",
                    "source_missing",
                    "qc_invalid",
                    "beam_blocked",
                    "below_detection_censored",
                    "mosaic_source_unassigned",
                    "stale_acquisition",
                )
                if self.contract in {
                    "verification-observation-error-contract-v8",
                    "verification-observation-error-contract-v9",
                    "verification-observation-error-contract-v10",
                    "verification-observation-error-contract-v11",
                    "verification-observation-error-contract-v13",
                    "verification-observation-error-contract-v14",
                }
                else (
                    "observed_clear",
                    "observed_echo",
                    "source_missing",
                    "qc_invalid",
                    "beam_blocked",
                    "below_detection_censored",
                    "mosaic_source_unassigned",
                )
            )
            or not math.isfinite(self.minimum_detectable_echo_dbz)
            or not math.isfinite(self.observation_error_reference_std_dbz)
            or self.observation_error_reference_std_dbz <= 0.0
            or not self.source_calibration_epochs
            or len({item[0] for item in self.source_calibration_epochs})
            != len(self.source_calibration_epochs)
        ):
            raise ValueError("verification observation-error contract is invalid")
        if self.contract == "verification-observation-error-contract-v3":
            if (
                tuple(sorted(self.source_calibration_epochs))
                != self.source_calibration_epochs
                or self.source_registry_digest is not None
                or self.calibration_registry_digest is not None
                or self.observation_error_derivation_digest is not None
                or self.observation_mask_derivation_digest is not None
                or self.verification_source_identity_digest is not None
                or self.source_acquisition_time_identity_digest is not None
            ):
                raise ValueError(
                    "exploratory observation-error contract is invalid"
                )
        elif self.contract == "verification-observation-error-contract-v4":
            for name in (
                "source_registry_digest",
                "calibration_registry_digest",
                "observation_error_derivation_digest",
            ):
                value = getattr(self, name)
                if value is None:
                    raise ValueError(
                        "deterministic observation-error lineage is incomplete"
                    )
                _require_sha256(name, value)
            if any(
                value is not None
                for value in (
                    self.observation_mask_derivation_digest,
                    self.verification_source_identity_digest,
                    self.source_acquisition_time_identity_digest,
                )
            ):
                raise ValueError("v4 observation-error lineage is invalid")
        else:
            for name in (
                "source_registry_digest",
                "calibration_registry_digest",
                "observation_error_derivation_digest",
                "observation_mask_derivation_digest",
                "verification_source_identity_digest",
                "source_acquisition_time_identity_digest",
            ):
                value = getattr(self, name)
                if value is None:
                    raise ValueError(
                        "confirmatory observation-error lineage is incomplete"
                    )
                _require_sha256(name, value)
            if self.contract in {
                "verification-observation-error-contract-v6",
                "verification-observation-error-contract-v7",
                "verification-observation-error-contract-v8",
                "verification-observation-error-contract-v9",
                "verification-observation-error-contract-v10",
                "verification-observation-error-contract-v11",
                "verification-observation-error-contract-v13",
                "verification-observation-error-contract-v14",
            }:
                for name in (
                    "detection_limit_dbz_digest",
                    "acquisition_time_offset_seconds_digest",
                ):
                    value = getattr(self, name)
                    if value is None:
                        raise ValueError(
                            "source-composed observation lineage is incomplete"
                        )
                    _require_sha256(name, value)
                if self.contract in {
                    "verification-observation-error-contract-v8",
                    "verification-observation-error-contract-v9",
                    "verification-observation-error-contract-v10",
                    "verification-observation-error-contract-v11",
                    "verification-observation-error-contract-v13",
                    "verification-observation-error-contract-v14",
                }:
                    if self.acquisition_age_seconds_digest is None:
                        raise ValueError(
                            "absolute acquisition-age lineage is incomplete"
                        )
                    _require_sha256(
                        "acquisition_age_seconds_digest",
                        self.acquisition_age_seconds_digest,
                    )
                elif self.acquisition_age_seconds_digest is not None:
                    raise ValueError("legacy observation-error lineage is invalid")
                if self.contract in {
                    "verification-observation-error-contract-v9",
                    "verification-observation-error-contract-v10",
                    "verification-observation-error-contract-v11",
                    "verification-observation-error-contract-v13",
                    "verification-observation-error-contract-v14",
                }:
                    if self.spatial_metric_valid_mask_digest is None:
                        raise ValueError("spatial metric lineage is incomplete")
                    _require_sha256(
                        "spatial_metric_valid_mask_digest",
                        self.spatial_metric_valid_mask_digest,
                    )
                elif self.spatial_metric_valid_mask_digest is not None:
                    raise ValueError("legacy spatial metric lineage is invalid")
            elif (
                self.detection_limit_dbz_digest is not None
                or self.acquisition_time_offset_seconds_digest is not None
                or self.acquisition_age_seconds_digest is not None
                or self.spatial_metric_valid_mask_digest is not None
            ):
                raise ValueError("legacy observation-error lineage is invalid")
        for name in (
            "range_elevation_validity_domain_digest",
            "beam_blockage_visibility_mask_digest",
            "attenuation_qc_digest",
            "censoring_rule_digest",
            "spatial_correlation_block_digest",
            "quality_weight_interpretation_digest",
            "observation_error_model_digest",
            "valid_mask_digest",
            "quality_weight_digest",
            "observation_std_dbz_digest",
            "observation_state_code_digest",
            "observation_error_plan_digest",
        ):
            _require_sha256(name, getattr(self, name))
        for source_digest, calibration_epoch_digest in self.source_calibration_epochs:
            _require_sha256("verification source radar", source_digest)
            _require_sha256("radar calibration epoch", calibration_epoch_digest)
        if self.radar_source_kind == "single_site":
            if len(self.source_calibration_epochs) != 1 or (
                self.source_radar_index_map_digest is not None
            ):
                raise ValueError("single-site observation-error source is invalid")
        elif self.source_radar_index_map_digest is None:
            raise ValueError("mosaic observation-error source map is required")
        else:
            _require_sha256(
                "verification source radar index map",
                self.source_radar_index_map_digest,
            )
        object.__setattr__(self, "contract_digest", json_digest(self.payload))

    @classmethod
    def from_tensors(
        cls,
        *,
        plan: VerificationObservationErrorPlan,
        valid_mask: Tensor,
        quality_weight: Tensor,
        observation_std_dbz: Tensor,
        frames_dbz: Tensor,
        observation_state_code: Tensor,
        source_radar_index_map: Tensor | None,
        source_calibration_epochs: tuple[tuple[str, str], ...],
        range_elevation_validity_domain_digest: str,
        beam_blockage_visibility_mask_digest: str,
        spatial_correlation_block_digest: str,
    ) -> VerificationObservationErrorContract:
        if type(plan) is not VerificationObservationErrorPlan:
            raise ValueError("verification observation-error plan is invalid")
        _validate_verification_cell_states(
            frames_dbz=frames_dbz,
            valid_mask=valid_mask,
            quality_weight=quality_weight,
            observation_std_dbz=observation_std_dbz,
            observation_state_code=observation_state_code,
            minimum_detectable_echo_dbz=plan.minimum_detectable_echo_dbz,
            radar_source_kind=plan.radar_source_kind,
            source_radar_index_map=source_radar_index_map,
        )
        return cls(
            radar_source_kind=plan.radar_source_kind,
            source_calibration_epochs=tuple(sorted(source_calibration_epochs)),
            range_elevation_validity_domain_digest=(
                range_elevation_validity_domain_digest
            ),
            beam_blockage_visibility_mask_digest=(
                beam_blockage_visibility_mask_digest
            ),
            attenuation_qc_digest=plan.attenuation_qc_digest,
            censoring_rule_digest=plan.censoring_rule_digest,
            spatial_correlation_block_digest=spatial_correlation_block_digest,
            quality_weight_interpretation_digest=(
                plan.quality_weight_interpretation_digest
            ),
            observation_error_model_digest=plan.observation_error_model_digest,
            minimum_detectable_echo_dbz=plan.minimum_detectable_echo_dbz,
            observation_error_reference_std_dbz=(
                plan.observation_error_reference_std_dbz
            ),
            valid_mask_digest=tensor_digest(valid_mask),
            quality_weight_digest=tensor_digest(quality_weight),
            observation_std_dbz_digest=tensor_digest(observation_std_dbz),
            observation_state_code_digest=tensor_digest(
                observation_state_code
            ),
            observation_error_plan_digest=plan.plan_digest,
            spatial_correlation_role=plan.spatial_correlation_role,
            source_radar_index_map_digest=(
                None
                if source_radar_index_map is None
                else tensor_digest(source_radar_index_map)
            ),
        )

    def validate_against_plan(
        self,
        plan: VerificationObservationErrorPlan,
    ) -> None:
        plan.validate_integrity()
        if (
            type(plan) is not VerificationObservationErrorPlan
            or self.contract_digest != json_digest(self.payload)
            or self.observation_error_plan_digest != plan.plan_digest
            or (
                self.contract == "verification-observation-error-contract-v4"
                and (
                    plan.contract != "verification-observation-error-plan-v2"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v5"
                and (
                    plan.contract != "verification-observation-error-plan-v3"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v6"
                and (
                    plan.contract != "verification-observation-error-plan-v4"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v7"
                and (
                    plan.contract != "verification-observation-error-plan-v5"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v10"
                and (
                    plan.contract != "verification-observation-error-plan-v8"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v14"
                and (
                    plan.contract != "verification-observation-error-plan-v12"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v13"
                and (
                    plan.contract != "verification-observation-error-plan-v11"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v11"
                and (
                    plan.contract != "verification-observation-error-plan-v9"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v9"
                and (
                    plan.contract != "verification-observation-error-plan-v7"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or (
                self.contract == "verification-observation-error-contract-v8"
                and (
                    plan.contract != "verification-observation-error-plan-v6"
                    or self.source_registry_digest
                    != plan.source_registry_digest
                    or self.calibration_registry_digest
                    != plan.calibration_registry_digest
                )
            )
            or self.radar_source_kind != plan.radar_source_kind
            or self.attenuation_qc_digest != plan.attenuation_qc_digest
            or self.censoring_rule_digest != plan.censoring_rule_digest
            or self.quality_weight_interpretation_digest
            != plan.quality_weight_interpretation_digest
            or self.observation_error_model_digest
            != plan.observation_error_model_digest
            or self.minimum_detectable_echo_dbz
            != plan.minimum_detectable_echo_dbz
            or self.observation_error_reference_std_dbz
            != plan.observation_error_reference_std_dbz
            or self.spatial_correlation_role != plan.spatial_correlation_role
        ):
            raise ValueError(
                "verification contract disagrees with its observation-error plan"
            )

    @property
    def scientific_evidence_mode(
        self,
    ) -> Literal["exploratory_only", "deterministic_replay"]:
        if self.contract in {
            "verification-observation-error-contract-v4",
            "verification-observation-error-contract-v5",
            "verification-observation-error-contract-v6",
            "verification-observation-error-contract-v7",
            "verification-observation-error-contract-v8",
            "verification-observation-error-contract-v9",
            "verification-observation-error-contract-v10",
            "verification-observation-error-contract-v11",
            "verification-observation-error-contract-v13",
            "verification-observation-error-contract-v14",
        }:
            return "deterministic_replay"
        return "exploratory_only"

    @property
    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract": self.contract,
            "radar_source_kind": self.radar_source_kind,
            "source_calibration_epochs": [
                list(item) for item in self.source_calibration_epochs
            ],
            "range_elevation_validity_domain_digest": (
                self.range_elevation_validity_domain_digest
            ),
            "beam_blockage_visibility_mask_digest": (
                self.beam_blockage_visibility_mask_digest
            ),
            "attenuation_qc_digest": self.attenuation_qc_digest,
            "censoring_rule_digest": self.censoring_rule_digest,
            "spatial_correlation_block_digest": (
                self.spatial_correlation_block_digest
            ),
            "spatial_correlation_role": self.spatial_correlation_role,
            "quality_weight_interpretation_digest": (
                self.quality_weight_interpretation_digest
            ),
            "observation_error_model_digest": self.observation_error_model_digest,
            "minimum_detectable_echo_dbz": self.minimum_detectable_echo_dbz,
            "observation_error_reference_std_dbz": (
                self.observation_error_reference_std_dbz
            ),
            "valid_mask_digest": self.valid_mask_digest,
            "quality_weight_digest": self.quality_weight_digest,
            "observation_std_dbz_digest": self.observation_std_dbz_digest,
            "observation_state_code_digest": (
                self.observation_state_code_digest
            ),
            "observation_error_plan_digest": (
                self.observation_error_plan_digest
            ),
            "source_radar_index_map_digest": self.source_radar_index_map_digest,
            "missing_data_taxonomy": list(self.missing_data_taxonomy),
            "metric_weight_rule": self.metric_weight_rule,
        }
        if self.contract in {
            "verification-observation-error-contract-v4",
            "verification-observation-error-contract-v5",
            "verification-observation-error-contract-v6",
            "verification-observation-error-contract-v7",
            "verification-observation-error-contract-v8",
            "verification-observation-error-contract-v9",
            "verification-observation-error-contract-v10",
            "verification-observation-error-contract-v11",
            "verification-observation-error-contract-v13",
            "verification-observation-error-contract-v14",
        }:
            payload.update(
                {
                    "source_registry_digest": self.source_registry_digest,
                    "calibration_registry_digest": (
                        self.calibration_registry_digest
                    ),
                    "observation_error_derivation_digest": (
                        self.observation_error_derivation_digest
                    ),
                    "scientific_evidence_mode": self.scientific_evidence_mode,
                }
            )
        if self.contract in {
            "verification-observation-error-contract-v5",
            "verification-observation-error-contract-v6",
            "verification-observation-error-contract-v7",
            "verification-observation-error-contract-v8",
            "verification-observation-error-contract-v10",
            "verification-observation-error-contract-v11",
            "verification-observation-error-contract-v13",
            "verification-observation-error-contract-v14",
        }:
            payload.update(
                {
                    "observation_mask_derivation_digest": (
                        self.observation_mask_derivation_digest
                    ),
                    "verification_source_identity_digest": (
                        self.verification_source_identity_digest
                    ),
                    "source_acquisition_time_identity_digest": (
                        self.source_acquisition_time_identity_digest
                    ),
                }
            )
        if self.contract in {
            "verification-observation-error-contract-v6",
            "verification-observation-error-contract-v7",
            "verification-observation-error-contract-v8",
            "verification-observation-error-contract-v10",
            "verification-observation-error-contract-v11",
            "verification-observation-error-contract-v13",
            "verification-observation-error-contract-v14",
        }:
            payload.update(
                {
                    "detection_limit_dbz_digest": (
                        self.detection_limit_dbz_digest
                    ),
                    "acquisition_time_offset_seconds_digest": (
                        self.acquisition_time_offset_seconds_digest
                    ),
                }
            )
            if self.contract in {
                "verification-observation-error-contract-v8",
                "verification-observation-error-contract-v10",
                "verification-observation-error-contract-v11",
                "verification-observation-error-contract-v13",
                "verification-observation-error-contract-v14",
            }:
                payload["acquisition_age_seconds_digest"] = (
                    self.acquisition_age_seconds_digest
                )
            if self.contract in {
                "verification-observation-error-contract-v10",
                "verification-observation-error-contract-v11",
                "verification-observation-error-contract-v13",
                "verification-observation-error-contract-v14",
            }:
                payload["spatial_metric_valid_mask_digest"] = (
                    self.spatial_metric_valid_mask_digest
                )
        return payload


def _derive_verification_observation_error_tensors(
    *,
    plan: VerificationObservationErrorPlan,
    raw_inputs: VerificationObservationDerivationInputs,
    source_registry: MosaicObservationSourceRegistry,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    if (
        type(plan) is not VerificationObservationErrorPlan
        or (plan.contract, raw_inputs.contract)
        not in {
            (
                "verification-observation-error-plan-v2",
                "verification-observation-derivation-inputs-v1",
            ),
            (
                "verification-observation-error-plan-v11",
                "verification-observation-derivation-inputs-v10",
            ),
            (
                "verification-observation-error-plan-v12",
                "verification-observation-derivation-inputs-v11",
            ),
        }
        or plan.derivation_algorithm_digest
        != (
            OBSERVATION_ERROR_DERIVATION_ALGORITHM_V11_DIGEST
            if plan.contract == "verification-observation-error-plan-v12"
            else (
                OBSERVATION_ERROR_DERIVATION_ALGORITHM_V10_DIGEST
                if plan.contract == "verification-observation-error-plan-v11"
                else OBSERVATION_ERROR_DERIVATION_ALGORITHM_DIGEST
            )
        )
        or type(raw_inputs) is not VerificationObservationDerivationInputs
        or type(source_registry) is not MosaicObservationSourceRegistry
    ):
        raise ValueError("deterministic observation-error derivation is invalid")
    plan.validate_integrity()
    raw_inputs.validate_integrity()
    source_registry.validate_against_plan(plan)
    source_registry.validate_source_map(
        raw_inputs.source_radar_index_map,
        expected_shape=raw_inputs.frames_dbz.shape,
        expected_device=raw_inputs.frames_dbz.device,
    )
    frames = raw_inputs.frames_dbz
    detection_limit = (
        raw_inputs.detection_limit_dbz
        if raw_inputs.contract
        in {
            "verification-observation-derivation-inputs-v10",
            "verification-observation-derivation-inputs-v11",
        }
        else torch.full_like(frames, plan.minimum_detectable_echo_dbz)
    )
    assert detection_limit is not None
    if raw_inputs.source_radar_index_map is None:
        source_indices = torch.zeros_like(frames, dtype=torch.int64)
        mosaic_unassigned = torch.zeros_like(frames, dtype=torch.bool)
        exported_source_map = None
    else:
        source_indices = raw_inputs.source_radar_index_map
        mosaic_unassigned = source_indices == -1
        exported_source_map = source_indices.detach().clone()
    assigned = ~mosaic_unassigned
    source_missing = assigned & ~raw_inputs.source_present_mask
    beam_blocked = (
        assigned
        & raw_inputs.source_present_mask
        & raw_inputs.beam_blocked_mask
    )
    current_inputs = raw_inputs.contract in {
        "verification-observation-derivation-inputs-v10",
        "verification-observation-derivation-inputs-v11",
    }
    if current_inputs:
        stale_acquisition = (
            assigned
            & raw_inputs.source_present_mask
            & ~beam_blocked
            & raw_inputs.range_elevation_valid_mask
            & ~raw_inputs.acquisition_time_valid_mask
        )
        qc_invalid = (
            assigned
            & raw_inputs.source_present_mask
            & ~beam_blocked
            & ~stale_acquisition
            & (
                ~raw_inputs.range_elevation_valid_mask
                | ~raw_inputs.attenuation_qc_valid_mask
            )
        )
        eligible = (
            assigned
            & raw_inputs.source_present_mask
            & raw_inputs.range_elevation_valid_mask
            & ~raw_inputs.beam_blocked_mask
            & raw_inputs.acquisition_time_valid_mask
            & raw_inputs.attenuation_qc_valid_mask
        )
    else:
        stale_acquisition = torch.zeros_like(frames, dtype=torch.bool)
        qc_invalid = (
            assigned
            & raw_inputs.source_present_mask
            & ~beam_blocked
            & (
                ~raw_inputs.range_elevation_valid_mask
                | ~raw_inputs.attenuation_qc_valid_mask
            )
        )
        eligible = (
            assigned
            & raw_inputs.source_present_mask
            & raw_inputs.range_elevation_valid_mask
            & ~raw_inputs.beam_blocked_mask
            & raw_inputs.attenuation_qc_valid_mask
        )
    if bool(
        torch.any(
            raw_inputs.below_detection_censored_mask
            & (
                ~eligible
                | (frames > detection_limit)
            )
        )
    ):
        raise ValueError("below-detection censoring input is invalid")
    censored = eligible & raw_inputs.below_detection_censored_mask
    if current_inputs:
        observed_clear = eligible & raw_inputs.confirmed_clear_mask
        observed_echo = eligible & ~censored & ~observed_clear
    else:
        observed_echo = eligible & ~censored & (frames >= detection_limit)
        observed_clear = eligible & ~censored & ~observed_echo
    valid_mask = observed_clear | observed_echo | censored
    observation_state = torch.full_like(
        frames,
        int(VerificationCellState.QC_INVALID),
        dtype=torch.uint8,
    )
    for mask, state in (
        (observed_clear, VerificationCellState.OBSERVED_CLEAR),
        (observed_echo, VerificationCellState.OBSERVED_ECHO),
        (censored, VerificationCellState.BELOW_DETECTION_CENSORED),
        (qc_invalid, VerificationCellState.QC_INVALID),
        (beam_blocked, VerificationCellState.BEAM_BLOCKED),
        (source_missing, VerificationCellState.SOURCE_MISSING),
        (stale_acquisition, VerificationCellState.STALE_ACQUISITION),
        (
            mosaic_unassigned,
            VerificationCellState.MOSAIC_SOURCE_UNASSIGNED,
        ),
    ):
        observation_state = torch.where(
            mask,
            torch.full_like(observation_state, int(state)),
            observation_state,
        )
    lookup_indices = source_indices.clamp_min(0)
    quality_lookup = frames.new_tensor(
        [source.quality_weight for source in source_registry.ordered_sources]
    )
    observation_std_lookup = frames.new_tensor(
        [
            source.observation_std_dbz
            for source in source_registry.ordered_sources
        ]
    )
    baseline_quality = quality_lookup[lookup_indices]
    baseline_std = observation_std_lookup[lookup_indices]
    if raw_inputs.contract in {
        "verification-observation-derivation-inputs-v10",
        "verification-observation-derivation-inputs-v11",
    }:
        mask_derivation = cast(
            VerificationObservationMaskDerivationArtifact,
            raw_inputs.mask_derivation,
        )
        evidence = mask_derivation.raw_evidence
        (
            _,
            _,
            _,
            _,
            _,
            _,
            selected_range_km,
            selected_elevation_deg,
            selected_beam_blockage_fraction,
            selected_attenuation_qc_score,
        ) = _selected_verification_spatial_evidence(evidence)
        range_fraction = (
            selected_range_km / cast(float, plan.maximum_range_km)
        ).clamp(min=0.0, max=1.0)
        elevation_scale = max(
            abs(cast(float, plan.minimum_elevation_deg)),
            abs(cast(float, plan.maximum_elevation_deg)),
            torch.finfo(frames.dtype).eps,
        )
        elevation_fraction = (
            selected_elevation_deg.abs() / elevation_scale
        ).clamp(max=1.0)
        blockage_fraction = selected_beam_blockage_fraction
        attenuation_penalty = 1.0 - selected_attenuation_qc_score
        range_reliability = (1.0 - 0.5 * range_fraction).clamp(
            min=0.5,
            max=1.0,
        )
        spatial_quality = (
            selected_attenuation_qc_score
            * (1.0 - blockage_fraction)
            * range_reliability
        ).clamp(min=0.0, max=1.0)
        spatial_std_multiplier = (
            1.0
            + range_fraction
            + elevation_fraction
            + blockage_fraction
            + attenuation_penalty
        )
        baseline_quality = baseline_quality * spatial_quality
        baseline_std = baseline_std * spatial_std_multiplier
        acquisition_age = cast(Tensor, raw_inputs.acquisition_age_seconds)
        temporal_decay = torch.exp(
            -torch.pow(
                acquisition_age
                / cast(float, plan.temporal_quality_decay_scale_seconds),
                cast(float, plan.temporal_quality_decay_power),
            )
        )
        temporal_error = (
            cast(float, plan.temporal_error_growth_dbz_per_second)
            * acquisition_age
        )
        baseline_quality = baseline_quality * temporal_decay
        baseline_std = torch.sqrt(
            baseline_std.square() + temporal_error.square()
        )
    quality_weight = torch.where(
        valid_mask,
        baseline_quality,
        torch.zeros_like(frames),
    )
    observation_std_dbz = torch.where(
        valid_mask,
        baseline_std,
        torch.zeros_like(frames),
    )
    _validate_verification_cell_states(
        frames_dbz=frames,
        valid_mask=valid_mask,
        quality_weight=quality_weight,
        observation_std_dbz=observation_std_dbz,
        observation_state_code=observation_state,
        minimum_detectable_echo_dbz=plan.minimum_detectable_echo_dbz,
        radar_source_kind=plan.radar_source_kind,
        source_radar_index_map=exported_source_map,
        detection_limit_dbz=detection_limit,
    )
    return (
        valid_mask,
        quality_weight,
        observation_std_dbz,
        observation_state,
        exported_source_map,
    )


@dataclass(frozen=True)
class ObservationErrorDerivationArtifact:
    """Replayable plan-to-tensor evidence for offline scientific validation."""

    plan: VerificationObservationErrorPlan
    raw_inputs: VerificationObservationDerivationInputs
    source_registry: MosaicObservationSourceRegistry
    _valid_mask: Tensor = field(repr=False)
    _quality_weight: Tensor = field(repr=False)
    _observation_std_dbz: Tensor = field(repr=False)
    _observation_state_code: Tensor = field(repr=False)
    _source_radar_index_map: Tensor | None = field(repr=False)
    contract: str = "observation-error-derivation-artifact-v1"
    artifact_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract not in {
            "observation-error-derivation-artifact-v1",
            "observation-error-derivation-artifact-v2",
            "observation-error-derivation-artifact-v3",
            "observation-error-derivation-artifact-v4",
            "observation-error-derivation-artifact-v5",
            "observation-error-derivation-artifact-v6",
            "observation-error-derivation-artifact-v7",
            "observation-error-derivation-artifact-v8",
            "observation-error-derivation-artifact-v10",
            "observation-error-derivation-artifact-v11",
        }:
            raise ValueError("observation-error derivation artifact is invalid")
        expected_contract = {
            (
                "verification-observation-error-plan-v11",
                "verification-observation-derivation-inputs-v10",
            ): "observation-error-derivation-artifact-v10",
            (
                "verification-observation-error-plan-v12",
                "verification-observation-derivation-inputs-v11",
            ): "observation-error-derivation-artifact-v11",
            (
                "verification-observation-error-plan-v9",
                "verification-observation-derivation-inputs-v8",
            ): "observation-error-derivation-artifact-v8",
            (
                "verification-observation-error-plan-v8",
                "verification-observation-derivation-inputs-v7",
            ): "observation-error-derivation-artifact-v7",
            (
                "verification-observation-error-plan-v7",
                "verification-observation-derivation-inputs-v6",
            ): "observation-error-derivation-artifact-v6",
            (
                "verification-observation-error-plan-v6",
                "verification-observation-derivation-inputs-v5",
            ): "observation-error-derivation-artifact-v5",
            (
                "verification-observation-error-plan-v5",
                "verification-observation-derivation-inputs-v4",
            ): "observation-error-derivation-artifact-v4",
            (
                "verification-observation-error-plan-v4",
                "verification-observation-derivation-inputs-v3",
            ): "observation-error-derivation-artifact-v3",
            (
                "verification-observation-error-plan-v3",
                "verification-observation-derivation-inputs-v2",
            ): "observation-error-derivation-artifact-v2",
        }.get(
            (self.plan.contract, self.raw_inputs.contract),
            "observation-error-derivation-artifact-v1",
        )
        if self.contract != expected_contract:
            raise ValueError("observation-error derivation generation is invalid")
        derived = _derive_verification_observation_error_tensors(
            plan=self.plan,
            raw_inputs=self.raw_inputs,
            source_registry=self.source_registry,
        )
        supplied = (
            self._valid_mask,
            self._quality_weight,
            self._observation_std_dbz,
            self._observation_state_code,
            self._source_radar_index_map,
        )
        if any(
            (expected is None) != (actual is None)
            or (
                expected is not None
                and actual is not None
                and not bool(torch.equal(expected, actual))
            )
            for expected, actual in zip(derived, supplied)
        ):
            raise ValueError("observation-error derivation replay mismatch")
        for name in (
            "_valid_mask",
            "_quality_weight",
            "_observation_std_dbz",
            "_observation_state_code",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        if self._source_radar_index_map is not None:
            object.__setattr__(
                self,
                "_source_radar_index_map",
                self._source_radar_index_map.detach().clone(),
            )
        object.__setattr__(self, "artifact_digest", json_digest(self.payload))

    def validate_replay(self) -> None:
        derived = _derive_verification_observation_error_tensors(
            plan=self.plan,
            raw_inputs=self.raw_inputs,
            source_registry=self.source_registry,
        )
        current = (
            self._valid_mask,
            self._quality_weight,
            self._observation_std_dbz,
            self._observation_state_code,
            self._source_radar_index_map,
        )
        if (
            self.artifact_digest != json_digest(self.payload)
            or any(
                (expected is None) != (actual is None)
                or (
                    expected is not None
                    and actual is not None
                    and not bool(torch.equal(expected, actual))
                )
                for expected, actual in zip(derived, current)
            )
        ):
            raise ValueError("observation-error derivation replay mismatch")

    @property
    def valid_mask(self) -> Tensor:
        self.validate_replay()
        return self._valid_mask.detach().clone()

    @property
    def quality_weight(self) -> Tensor:
        self.validate_replay()
        return self._quality_weight.detach().clone()

    @property
    def observation_std_dbz(self) -> Tensor:
        self.validate_replay()
        return self._observation_std_dbz.detach().clone()

    @property
    def observation_state_code(self) -> Tensor:
        self.validate_replay()
        return self._observation_state_code.detach().clone()

    @property
    def source_radar_index_map(self) -> Tensor | None:
        self.validate_replay()
        if self._source_radar_index_map is None:
            return None
        return self._source_radar_index_map.detach().clone()

    @property
    def observation_error_contract(self) -> VerificationObservationErrorContract:
        self.validate_replay()
        source_identity = self.raw_inputs.source_identity
        mask_derivation = self.raw_inputs.mask_derivation
        is_confirmatory = self.contract in {
            "observation-error-derivation-artifact-v2",
            "observation-error-derivation-artifact-v3",
            "observation-error-derivation-artifact-v4",
            "observation-error-derivation-artifact-v5",
            "observation-error-derivation-artifact-v6",
            "observation-error-derivation-artifact-v7",
            "observation-error-derivation-artifact-v8",
            "observation-error-derivation-artifact-v10",
            "observation-error-derivation-artifact-v11",
        }
        is_source_composed = (
            self.contract in {
                "observation-error-derivation-artifact-v3",
                "observation-error-derivation-artifact-v4",
                "observation-error-derivation-artifact-v5",
                "observation-error-derivation-artifact-v6",
                "observation-error-derivation-artifact-v7",
                "observation-error-derivation-artifact-v8",
                "observation-error-derivation-artifact-v10",
                "observation-error-derivation-artifact-v11",
            }
        )
        if is_confirmatory and (
            type(source_identity) is not VerificationObservationSourceIdentity
            or type(mask_derivation)
            is not VerificationObservationMaskDerivationArtifact
        ):
            raise ValueError("confirmatory observation-error lineage is incomplete")
        return VerificationObservationErrorContract(
            radar_source_kind=self.plan.radar_source_kind,
            source_calibration_epochs=(
                self.source_registry.source_calibration_epochs
            ),
            range_elevation_validity_domain_digest=(
                self.raw_inputs.range_elevation_validity_domain_digest
            ),
            beam_blockage_visibility_mask_digest=(
                self.raw_inputs.beam_blockage_visibility_mask_digest
            ),
            attenuation_qc_digest=self.plan.attenuation_qc_digest,
            censoring_rule_digest=self.plan.censoring_rule_digest,
            spatial_correlation_block_digest=(
                self.raw_inputs.spatial_correlation_block_digest
            ),
            quality_weight_interpretation_digest=(
                self.plan.quality_weight_interpretation_digest
            ),
            observation_error_model_digest=(
                self.plan.observation_error_model_digest
            ),
            minimum_detectable_echo_dbz=(
                self.plan.minimum_detectable_echo_dbz
            ),
            observation_error_reference_std_dbz=(
                self.plan.observation_error_reference_std_dbz
            ),
            valid_mask_digest=tensor_digest(self._valid_mask),
            quality_weight_digest=tensor_digest(self._quality_weight),
            observation_std_dbz_digest=tensor_digest(
                self._observation_std_dbz
            ),
            observation_state_code_digest=tensor_digest(
                self._observation_state_code
            ),
            observation_error_plan_digest=self.plan.plan_digest,
            source_radar_index_map_digest=(
                None
                if self._source_radar_index_map is None
                else tensor_digest(self._source_radar_index_map)
            ),
            detection_limit_dbz_digest=(
                tensor_digest(cast(Tensor, self.raw_inputs.detection_limit_dbz))
                if is_source_composed
                else None
            ),
            acquisition_time_offset_seconds_digest=(
                tensor_digest(
                    cast(
                        Tensor,
                        self.raw_inputs.acquisition_time_offset_seconds,
                    )
                )
                if is_source_composed
                else None
            ),
            acquisition_age_seconds_digest=(
                tensor_digest(
                    cast(Tensor, self.raw_inputs.acquisition_age_seconds)
                )
                if self.contract
                in {
                    "observation-error-derivation-artifact-v5",
                    "observation-error-derivation-artifact-v6",
                    "observation-error-derivation-artifact-v7",
                    "observation-error-derivation-artifact-v8",
                    "observation-error-derivation-artifact-v10",
                    "observation-error-derivation-artifact-v11",
                }
                else None
            ),
            spatial_metric_valid_mask_digest=(
                tensor_digest(
                    cast(Tensor, self.raw_inputs.spatial_metric_valid_mask)
                )
                if self.contract
                in {
                    "observation-error-derivation-artifact-v6",
                    "observation-error-derivation-artifact-v7",
                    "observation-error-derivation-artifact-v8",
                    "observation-error-derivation-artifact-v10",
                    "observation-error-derivation-artifact-v11",
                }
                else None
            ),
            source_registry_digest=self.source_registry.source_registry_digest,
            calibration_registry_digest=(
                self.source_registry.calibration_registry_digest
            ),
            observation_error_derivation_digest=self.artifact_digest,
            observation_mask_derivation_digest=(
                None
                if not is_confirmatory
                else cast(
                    VerificationObservationMaskDerivationArtifact,
                    mask_derivation,
                ).artifact_digest
            ),
            verification_source_identity_digest=(
                None
                if not is_confirmatory
                else cast(
                    VerificationObservationSourceIdentity,
                    source_identity,
                ).identity_digest
            ),
            source_acquisition_time_identity_digest=(
                None
                if not is_confirmatory
                else cast(
                    VerificationObservationSourceIdentity,
                    source_identity,
                ).source_acquisition_time_identity_digest
            ),
            missing_data_taxonomy=(
                (
                    "observed_clear",
                    "observed_echo",
                    "source_missing",
                    "qc_invalid",
                    "beam_blocked",
                    "below_detection_censored",
                    "mosaic_source_unassigned",
                    "stale_acquisition",
                )
                if self.contract
                in {
                    "observation-error-derivation-artifact-v5",
                    "observation-error-derivation-artifact-v6",
                    "observation-error-derivation-artifact-v7",
                    "observation-error-derivation-artifact-v8",
                    "observation-error-derivation-artifact-v10",
                    "observation-error-derivation-artifact-v11",
                }
                else (
                    "observed_clear",
                    "observed_echo",
                    "source_missing",
                    "qc_invalid",
                    "beam_blocked",
                    "below_detection_censored",
                    "mosaic_source_unassigned",
                )
            ),
            spatial_correlation_role=self.plan.spatial_correlation_role,
            contract=(
                "verification-observation-error-contract-v14"
                if self.contract == "observation-error-derivation-artifact-v11"
                else (
                    "verification-observation-error-contract-v13"
                    if self.contract == "observation-error-derivation-artifact-v10"
                    else (
                        "verification-observation-error-contract-v11"
                        if self.contract == "observation-error-derivation-artifact-v8"
                        else (
                            "verification-observation-error-contract-v10"
                            if self.contract == "observation-error-derivation-artifact-v7"
                            else (
                                "verification-observation-error-contract-v9"
                                if self.contract == "observation-error-derivation-artifact-v6"
                                else (
                                    "verification-observation-error-contract-v8"
                                    if self.contract == "observation-error-derivation-artifact-v5"
                                    else (
                                        "verification-observation-error-contract-v7"
                                        if self.contract == "observation-error-derivation-artifact-v4"
                                        else (
                                            "verification-observation-error-contract-v6"
                                            if is_source_composed
                                            else (
                                                "verification-observation-error-contract-v5"
                                                if is_confirmatory
                                                else "verification-observation-error-contract-v4"
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
        )

    @property
    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract": self.contract,
            "plan_digest": self.plan.plan_digest,
            "raw_input_digest": self.raw_inputs.content_digest,
            "raw_verification_identity_digest": (
                self.raw_inputs.raw_verification_identity_digest
            ),
            "source_registry_digest": self.source_registry.registry_digest,
            "derivation_algorithm_digest": (
                self.plan.derivation_algorithm_digest
            ),
            "valid_mask_digest": tensor_digest(self._valid_mask),
            "quality_weight_digest": tensor_digest(self._quality_weight),
            "observation_std_dbz_digest": tensor_digest(
                self._observation_std_dbz
            ),
            "observation_state_code_digest": tensor_digest(
                self._observation_state_code
            ),
            "source_radar_index_map_digest": (
                None
                if self._source_radar_index_map is None
                else tensor_digest(self._source_radar_index_map)
            ),
        }
        if self.contract in {
            "observation-error-derivation-artifact-v2",
            "observation-error-derivation-artifact-v3",
            "observation-error-derivation-artifact-v4",
            "observation-error-derivation-artifact-v5",
            "observation-error-derivation-artifact-v6",
            "observation-error-derivation-artifact-v7",
            "observation-error-derivation-artifact-v8",
            "observation-error-derivation-artifact-v10",
            "observation-error-derivation-artifact-v11",
        }:
            source_identity = cast(
                VerificationObservationSourceIdentity,
                self.raw_inputs.source_identity,
            )
            mask_derivation = cast(
                VerificationObservationMaskDerivationArtifact,
                self.raw_inputs.mask_derivation,
            )
            payload.update(
                {
                    "verification_source_identity_digest": (
                        source_identity.identity_digest
                    ),
                    "source_acquisition_time_identity_digest": (
                        source_identity.source_acquisition_time_identity_digest
                    ),
                    "observation_mask_derivation_digest": (
                        mask_derivation.artifact_digest
                    ),
                }
            )
        return payload


def derive_verification_observation_error(
    *,
    plan: VerificationObservationErrorPlan,
    raw_verification_source: VerificationObservationDerivationInputs,
    source_registry: MosaicObservationSourceRegistry,
) -> ObservationErrorDerivationArtifact:
    """Derive and seal replayable observation-error tensors."""

    if raw_verification_source.contract in {
        "verification-observation-derivation-inputs-v2",
        "verification-observation-derivation-inputs-v3",
        "verification-observation-derivation-inputs-v4",
        "verification-observation-derivation-inputs-v5",
        "verification-observation-derivation-inputs-v6",
        "verification-observation-derivation-inputs-v7",
        "verification-observation-derivation-inputs-v8",
    }:
        raise ValueError("legacy observation derivation inputs are audit-only")

    derived = _derive_verification_observation_error_tensors(
        plan=plan,
        raw_inputs=raw_verification_source,
        source_registry=source_registry,
    )
    return ObservationErrorDerivationArtifact(
        plan=plan,
        raw_inputs=raw_verification_source,
        source_registry=source_registry,
        _valid_mask=derived[0],
        _quality_weight=derived[1],
        _observation_std_dbz=derived[2],
        _observation_state_code=derived[3],
        _source_radar_index_map=derived[4],
        contract=(
            "observation-error-derivation-artifact-v11"
            if raw_verification_source.contract
            == "verification-observation-derivation-inputs-v11"
            else (
                "observation-error-derivation-artifact-v10"
                if raw_verification_source.contract
                == "verification-observation-derivation-inputs-v10"
                else "observation-error-derivation-artifact-v1"
            )
        ),
    )


_SUPPORTED_VERIFICATION_BUNDLE_CONTRACTS = frozenset(
    f"radar-verification-bundle-v{generation}"
    for generation in range(1, 18)
)
_OBSERVATION_ERROR_VERIFICATION_BUNDLE_CONTRACTS = frozenset(
    f"radar-verification-bundle-v{generation}"
    for generation in range(6, 18)
)


@dataclass(frozen=True)
class VerificationBundle:
    """Content-addressed future radar/QC bundle for delayed verification."""

    frames_dbz: Tensor
    valid_mask: Tensor
    valid_times: tuple[str, ...]
    grid_contract_digest: str
    radar_product_digest: str
    qc_pipeline_digest: str
    mask_policy_digest: str | None = None
    censor_policy_digest: str | None = None
    reflectivity_resolution_dbz: float | None = None
    quantization_origin_dbz: float | None = None
    threshold_bin_convention: str | None = None
    floor_representation_contract_digest: str | None = None
    quality_weight: Tensor | None = None
    observation_std_dbz: Tensor | None = None
    observation_state_code: Tensor | None = None
    source_radar_index_map: Tensor | None = None
    detection_limit_dbz: Tensor | None = None
    acquisition_time_offset_seconds: Tensor | None = None
    acquisition_age_seconds: Tensor | None = None
    spatial_metric_valid_mask: Tensor | None = None
    observation_error_contract: VerificationObservationErrorContract | None = None
    observation_error_derivation: ObservationErrorDerivationArtifact | None = None
    contract: str = "radar-verification-bundle-v1"
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract not in _SUPPORTED_VERIFICATION_BUNDLE_CONTRACTS:
            raise ValueError("unsupported verification bundle contract")
        if self.frames_dbz.ndim != 3 or not self.frames_dbz.is_floating_point():
            raise ValueError(
                "verification frames must be floating with shape [lead,H,W]"
            )
        if (
            self.valid_mask.dtype != torch.bool
            or self.valid_mask.shape != self.frames_dbz.shape
            or self.valid_mask.device != self.frames_dbz.device
        ):
            raise ValueError(
                "verification valid_mask must be boolean and match frames"
            )
        if bool(torch.any(self.valid_mask & ~torch.isfinite(self.frames_dbz))):
            raise ValueError("valid verification cells must contain finite dBZ")
        if (
            not isinstance(self.valid_times, tuple)
            or len(self.valid_times) != self.frames_dbz.shape[0]
        ):
            raise ValueError(
                "verification valid_times must match the lead dimension"
            )
        canonical_times = tuple(
            _canonical_verification_time(value) for value in self.valid_times
        )
        parsed_times = tuple(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in canonical_times
        )
        if any(
            later <= earlier
            for earlier, later in zip(parsed_times, parsed_times[1:])
        ):
            raise ValueError("verification valid_times must be increasing")
        for name, value in (
            ("grid_contract_digest", self.grid_contract_digest),
            ("radar_product_digest", self.radar_product_digest),
            ("qc_pipeline_digest", self.qc_pipeline_digest),
        ):
            _require_sha256(name, value)
        measurement_values = (
            self.mask_policy_digest,
            self.censor_policy_digest,
            self.reflectivity_resolution_dbz,
            self.quantization_origin_dbz,
            self.threshold_bin_convention,
            self.floor_representation_contract_digest,
        )
        if self.contract == "radar-verification-bundle-v1":
            if any(value is not None for value in measurement_values):
                raise ValueError(
                    "legacy verification bundles cannot claim v2 measurement lineage"
                )
        else:
            for name in (
                "mask_policy_digest",
                "censor_policy_digest",
                "floor_representation_contract_digest",
            ):
                value = getattr(self, name)
                if value is None:
                    raise ValueError("v2 verification measurement lineage is incomplete")
                _require_sha256(name, value)
            if (
                self.reflectivity_resolution_dbz is None
                or not math.isfinite(self.reflectivity_resolution_dbz)
                or self.reflectivity_resolution_dbz <= 0.0
                or self.quantization_origin_dbz is None
                or not math.isfinite(self.quantization_origin_dbz)
                or self.threshold_bin_convention
                != (
                    "threshold_edge_centered_bins"
                    if self.contract == "radar-verification-bundle-v2"
                    else "nearest_rounding_threshold_censor"
                )
            ):
                raise ValueError("v2 verification quantization contract is invalid")
        error_values = (
            self.quality_weight,
            self.observation_std_dbz,
            self.observation_state_code,
            self.observation_error_contract,
        )
        if (
            self.contract
            not in _OBSERVATION_ERROR_VERIFICATION_BUNDLE_CONTRACTS
        ):
            if any(value is not None for value in error_values) or (
                self.source_radar_index_map is not None
            ) or self.observation_error_derivation is not None or (
                self.detection_limit_dbz is not None
                or self.acquisition_time_offset_seconds is not None
                or self.acquisition_age_seconds is not None
                or self.spatial_metric_valid_mask is not None
            ):
                raise ValueError("legacy verification cannot claim observation error")
        else:
            if any(value is None for value in error_values):
                raise ValueError("verification observation-error lineage is incomplete")
            quality = cast(Tensor, self.quality_weight)
            observation_std = cast(Tensor, self.observation_std_dbz)
            observation_state = cast(Tensor, self.observation_state_code)
            source_radar_index_map = self.source_radar_index_map
            error_contract = cast(
                VerificationObservationErrorContract,
                self.observation_error_contract,
            )
            derivation = self.observation_error_derivation
            source_composition_tensors = (
                self.detection_limit_dbz,
                self.acquisition_time_offset_seconds,
            )
            if self.contract in {
                "radar-verification-bundle-v11",
                "radar-verification-bundle-v12",
                "radar-verification-bundle-v13",
                "radar-verification-bundle-v14",
                "radar-verification-bundle-v16",
                "radar-verification-bundle-v17",
            }:
                source_composition_tensors += (self.acquisition_age_seconds,)
            if self.contract in {
                "radar-verification-bundle-v9",
                "radar-verification-bundle-v10",
                "radar-verification-bundle-v11",
                "radar-verification-bundle-v12",
                "radar-verification-bundle-v13",
                "radar-verification-bundle-v14",
                "radar-verification-bundle-v16",
                "radar-verification-bundle-v17",
            }:
                if any(value is None for value in source_composition_tensors):
                    raise ValueError(
                        "source-composed verification tensors are incomplete"
                    )
                for tensor in source_composition_tensors:
                    assert tensor is not None
                    if (
                        tensor.shape != self.frames_dbz.shape
                        or tensor.dtype != self.frames_dbz.dtype
                        or tensor.device != self.frames_dbz.device
                        or not bool(torch.all(torch.isfinite(tensor)))
                    ):
                        raise ValueError(
                            "source-composed verification tensor is invalid"
                        )
                assert self.acquisition_time_offset_seconds is not None
                if bool(torch.any(self.acquisition_time_offset_seconds > 0.0)):
                    raise ValueError(
                        "verification acquisition offset cannot be future-dated"
                    )
                if (
                    self.contract
                    in {
                        "radar-verification-bundle-v11",
                        "radar-verification-bundle-v12",
                        "radar-verification-bundle-v13",
                        "radar-verification-bundle-v14",
                        "radar-verification-bundle-v16",
                        "radar-verification-bundle-v17",
                    }
                    and self.acquisition_age_seconds is not None
                    and bool(torch.any(self.acquisition_age_seconds < 0.0))
                ):
                    raise ValueError(
                        "verification acquisition age cannot be negative"
                    )
                if self.contract in {
                    "radar-verification-bundle-v12",
                    "radar-verification-bundle-v13",
                    "radar-verification-bundle-v14",
                    "radar-verification-bundle-v16",
                    "radar-verification-bundle-v17",
                }:
                    spatial_mask = self.spatial_metric_valid_mask
                    if (
                        spatial_mask is None
                        or spatial_mask.dtype is not torch.bool
                        or spatial_mask.shape != self.frames_dbz.shape
                        or spatial_mask.device != self.frames_dbz.device
                    ):
                        raise ValueError("spatial metric mask is invalid")
            elif any(value is not None for value in source_composition_tensors):
                raise ValueError("legacy verification cannot claim source composition")
            if (
                type(error_contract) is not VerificationObservationErrorContract
                or quality.shape != self.frames_dbz.shape
                or observation_std.shape != self.frames_dbz.shape
                or not quality.is_floating_point()
                or not observation_std.is_floating_point()
                or quality.dtype != self.frames_dbz.dtype
                or observation_std.dtype != self.frames_dbz.dtype
                or quality.device != self.frames_dbz.device
                or observation_std.device != self.frames_dbz.device
                or not bool(torch.all(torch.isfinite(quality)))
                or not bool(torch.all(torch.isfinite(observation_std)))
                or not bool(torch.all((quality >= 0.0) & (quality <= 1.0)))
                or not bool(torch.all(quality.masked_select(~self.valid_mask) == 0.0))
                or not bool(torch.all(observation_std.masked_select(self.valid_mask) > 0.0))
                or not bool(torch.all(observation_std.masked_select(~self.valid_mask) == 0.0))
                or error_contract.valid_mask_digest != tensor_digest(self.valid_mask)
                or error_contract.quality_weight_digest != tensor_digest(quality)
                or error_contract.observation_std_dbz_digest
                != tensor_digest(observation_std)
                or error_contract.observation_state_code_digest
                != tensor_digest(observation_state)
                or error_contract.source_radar_index_map_digest
                != (
                    None
                    if source_radar_index_map is None
                    else tensor_digest(source_radar_index_map)
                )
                or error_contract.attenuation_qc_digest != self.qc_pipeline_digest
                or error_contract.censoring_rule_digest != self.censor_policy_digest
            ):
                raise ValueError("verification observation-error tensors are invalid")
            if self.contract == "radar-verification-bundle-v6":
                if (
                    derivation is not None
                    or error_contract.scientific_evidence_mode
                    != "exploratory_only"
                ):
                    raise ValueError(
                        "v6 verification is exploratory observation evidence"
                    )
            elif self.contract == "radar-verification-bundle-v7":
                if type(derivation) is not ObservationErrorDerivationArtifact:
                    raise ValueError(
                        "deterministic observation-error derivation is required"
                    )
                derivation.validate_replay()
                derived_contract = derivation.observation_error_contract
                derived_source_map = derivation.source_radar_index_map
                if (
                    error_contract.scientific_evidence_mode
                    != "deterministic_replay"
                    or error_contract.contract_digest
                    != derived_contract.contract_digest
                    or error_contract.observation_error_derivation_digest
                    != derivation.artifact_digest
                    or tensor_digest(self.frames_dbz)
                    != tensor_digest(derivation.raw_inputs.frames_dbz)
                    or tensor_digest(self.valid_mask)
                    != tensor_digest(derivation.valid_mask)
                    or tensor_digest(quality)
                    != tensor_digest(derivation.quality_weight)
                    or tensor_digest(observation_std)
                    != tensor_digest(derivation.observation_std_dbz)
                    or tensor_digest(observation_state)
                    != tensor_digest(derivation.observation_state_code)
                    or (
                        None
                        if source_radar_index_map is None
                        else tensor_digest(source_radar_index_map)
                    )
                    != (
                        None
                        if derived_source_map is None
                        else tensor_digest(derived_source_map)
                    )
                ):
                    raise ValueError(
                        "verification disagrees with observation-error replay"
                    )
                if (
                    derivation.contract
                    != "observation-error-derivation-artifact-v1"
                    or error_contract.contract
                    != "verification-observation-error-contract-v4"
                ):
                    raise ValueError("v7 observation-error generation is invalid")
            elif self.contract == "radar-verification-bundle-v8":
                if (
                    type(derivation) is not ObservationErrorDerivationArtifact
                    or derivation.contract
                    != "observation-error-derivation-artifact-v2"
                    or error_contract.contract
                    != "verification-observation-error-contract-v5"
                    or type(derivation.raw_inputs.source_identity)
                    is not VerificationObservationSourceIdentity
                ):
                    raise ValueError(
                        "confirmatory observation-error derivation is required"
                    )
                derivation.validate_replay()
                source_identity = cast(
                    VerificationObservationSourceIdentity,
                    derivation.raw_inputs.source_identity,
                )
                derived_contract = derivation.observation_error_contract
                derived_source_map = derivation.source_radar_index_map
                if (
                    error_contract.scientific_evidence_mode
                    != "deterministic_replay"
                    or error_contract.contract_digest
                    != derived_contract.contract_digest
                    or error_contract.observation_error_derivation_digest
                    != derivation.artifact_digest
                    or error_contract.verification_source_identity_digest
                    != source_identity.identity_digest
                    or self.valid_times != source_identity.valid_times
                    or self.grid_contract_digest
                    != source_identity.grid_contract_digest
                    or self.radar_product_digest
                    != source_identity.radar_product_digest
                    or tensor_digest(self.frames_dbz)
                    != tensor_digest(derivation.raw_inputs.frames_dbz)
                    or tensor_digest(self.valid_mask)
                    != tensor_digest(derivation.valid_mask)
                    or tensor_digest(quality)
                    != tensor_digest(derivation.quality_weight)
                    or tensor_digest(observation_std)
                    != tensor_digest(derivation.observation_std_dbz)
                    or tensor_digest(observation_state)
                    != tensor_digest(derivation.observation_state_code)
                    or (
                        None
                        if source_radar_index_map is None
                        else tensor_digest(source_radar_index_map)
                    )
                    != (
                        None
                        if derived_source_map is None
                        else tensor_digest(derived_source_map)
                    )
                ):
                    raise ValueError(
                        "verification disagrees with confirmatory source replay"
                    )
            elif self.contract == "radar-verification-bundle-v9":
                if (
                    type(derivation) is not ObservationErrorDerivationArtifact
                    or derivation.contract
                    != "observation-error-derivation-artifact-v3"
                    or error_contract.contract
                    != "verification-observation-error-contract-v6"
                    or type(derivation.raw_inputs.source_identity)
                    is not VerificationObservationSourceIdentity
                ):
                    raise ValueError(
                        "source-composed observation derivation is required"
                    )
                derivation.validate_replay()
                source_identity = cast(
                    VerificationObservationSourceIdentity,
                    derivation.raw_inputs.source_identity,
                )
                derived_contract = derivation.observation_error_contract
                derived_source_map = derivation.source_radar_index_map
                derived_detection_limit = cast(
                    Tensor,
                    derivation.raw_inputs.detection_limit_dbz,
                )
                derived_time_offset = cast(
                    Tensor,
                    derivation.raw_inputs.acquisition_time_offset_seconds,
                )
                if (
                    error_contract.scientific_evidence_mode
                    != "deterministic_replay"
                    or error_contract.contract_digest
                    != derived_contract.contract_digest
                    or error_contract.observation_error_derivation_digest
                    != derivation.artifact_digest
                    or error_contract.verification_source_identity_digest
                    != source_identity.identity_digest
                    or error_contract.detection_limit_dbz_digest
                    != tensor_digest(derived_detection_limit)
                    or error_contract.acquisition_time_offset_seconds_digest
                    != tensor_digest(derived_time_offset)
                    or self.valid_times != source_identity.valid_times
                    or self.grid_contract_digest
                    != source_identity.grid_contract_digest
                    or self.radar_product_digest
                    != source_identity.radar_product_digest
                    or not bool(
                        torch.equal(self.frames_dbz, derivation.raw_inputs.frames_dbz)
                    )
                    or not bool(torch.equal(self.valid_mask, derivation.valid_mask))
                    or not bool(torch.equal(quality, derivation.quality_weight))
                    or not bool(
                        torch.equal(observation_std, derivation.observation_std_dbz)
                    )
                    or not bool(
                        torch.equal(
                            observation_state,
                            derivation.observation_state_code,
                        )
                    )
                    or not bool(
                        torch.equal(
                            cast(Tensor, self.detection_limit_dbz),
                            derived_detection_limit,
                        )
                    )
                    or not bool(
                        torch.equal(
                            cast(Tensor, self.acquisition_time_offset_seconds),
                            derived_time_offset,
                        )
                    )
                    or (
                        None
                        if source_radar_index_map is None
                        else tensor_digest(source_radar_index_map)
                    )
                    != (
                        None
                        if derived_source_map is None
                        else tensor_digest(derived_source_map)
                    )
                ):
                    raise ValueError(
                        "verification disagrees with source-composed replay"
                    )
            else:
                current_temporal_generation = (
                    self.contract
                    in {
                        "radar-verification-bundle-v11",
                        "radar-verification-bundle-v12",
                        "radar-verification-bundle-v13",
                        "radar-verification-bundle-v14",
                        "radar-verification-bundle-v16",
                        "radar-verification-bundle-v17",
                    }
                )
                if (
                    type(derivation) is not ObservationErrorDerivationArtifact
                    or derivation.contract
                    != (
                        "observation-error-derivation-artifact-v6"
                        if self.contract == "radar-verification-bundle-v12"
                        else (
                            "observation-error-derivation-artifact-v7"
                            if self.contract == "radar-verification-bundle-v13"
                            else (
                                "observation-error-derivation-artifact-v8"
                                if self.contract == "radar-verification-bundle-v14"
                                else (
                                    "observation-error-derivation-artifact-v11"
                                    if self.contract == "radar-verification-bundle-v17"
                                    else (
                                        "observation-error-derivation-artifact-v10"
                                        if self.contract
                                        == "radar-verification-bundle-v16"
                                        else (
                                            "observation-error-derivation-artifact-v5"
                                            if current_temporal_generation
                                            else "observation-error-derivation-artifact-v4"
                                        )
                                    )
                                )
                            )
                        )
                    )
                    or error_contract.contract
                    != (
                        "verification-observation-error-contract-v9"
                        if self.contract == "radar-verification-bundle-v12"
                        else (
                            "verification-observation-error-contract-v10"
                            if self.contract == "radar-verification-bundle-v13"
                            else (
                                "verification-observation-error-contract-v11"
                                if self.contract == "radar-verification-bundle-v14"
                                else (
                                    "verification-observation-error-contract-v14"
                                    if self.contract == "radar-verification-bundle-v17"
                                    else (
                                        "verification-observation-error-contract-v13"
                                        if self.contract == "radar-verification-bundle-v16"
                                        else (
                                            "verification-observation-error-contract-v8"
                                            if current_temporal_generation
                                            else "verification-observation-error-contract-v7"
                                        )
                                    )
                                )
                            )
                        )
                    )
                    or type(derivation.raw_inputs.source_identity)
                    is not VerificationObservationSourceIdentity
                ):
                    raise ValueError(
                        "preregistered temporal observation derivation is required"
                    )
                derivation.validate_replay()
                source_identity = cast(
                    VerificationObservationSourceIdentity,
                    derivation.raw_inputs.source_identity,
                )
                derived_contract = derivation.observation_error_contract
                derived_source_map = derivation.source_radar_index_map
                derived_detection_limit = cast(
                    Tensor, derivation.raw_inputs.detection_limit_dbz
                )
                derived_time_offset = cast(
                    Tensor,
                    derivation.raw_inputs.acquisition_time_offset_seconds,
                )
                derived_age = (
                    cast(Tensor, derivation.raw_inputs.acquisition_age_seconds)
                    if current_temporal_generation
                    else None
                )
                if (
                    error_contract.scientific_evidence_mode
                    != "deterministic_replay"
                    or error_contract.contract_digest
                    != derived_contract.contract_digest
                    or error_contract.observation_error_derivation_digest
                    != derivation.artifact_digest
                    or error_contract.verification_source_identity_digest
                    != source_identity.identity_digest
                    or error_contract.detection_limit_dbz_digest
                    != tensor_digest(derived_detection_limit)
                    or error_contract.acquisition_time_offset_seconds_digest
                    != tensor_digest(derived_time_offset)
                    or (
                        current_temporal_generation
                        and error_contract.acquisition_age_seconds_digest
                        != tensor_digest(cast(Tensor, derived_age))
                    )
                    or (
                        self.contract
                        in {
                            "radar-verification-bundle-v12",
                            "radar-verification-bundle-v13",
                            "radar-verification-bundle-v14",
                            "radar-verification-bundle-v16",
                            "radar-verification-bundle-v17",
                        }
                        and (
                            error_contract.spatial_metric_valid_mask_digest
                            != tensor_digest(
                                cast(
                                    Tensor,
                                    derivation.raw_inputs.spatial_metric_valid_mask,
                                )
                            )
                            or not bool(
                                torch.equal(
                                    cast(Tensor, self.spatial_metric_valid_mask),
                                    cast(
                                        Tensor,
                                        derivation.raw_inputs.spatial_metric_valid_mask,
                                    ),
                                )
                            )
                        )
                    )
                    or self.valid_times != source_identity.valid_times
                    or self.grid_contract_digest
                    != source_identity.grid_contract_digest
                    or self.radar_product_digest
                    != source_identity.radar_product_digest
                    or not bool(
                        torch.equal(
                            self.frames_dbz,
                            derivation.raw_inputs.frames_dbz,
                        )
                    )
                    or not bool(torch.equal(self.valid_mask, derivation.valid_mask))
                    or not bool(torch.equal(quality, derivation.quality_weight))
                    or not bool(
                        torch.equal(
                            observation_std,
                            derivation.observation_std_dbz,
                        )
                    )
                    or not bool(
                        torch.equal(
                            observation_state,
                            derivation.observation_state_code,
                        )
                    )
                    or not bool(
                        torch.equal(
                            cast(Tensor, self.detection_limit_dbz),
                            derived_detection_limit,
                        )
                    )
                    or not bool(
                        torch.equal(
                            cast(Tensor, self.acquisition_time_offset_seconds),
                            derived_time_offset,
                        )
                    )
                    or (
                        current_temporal_generation
                        and not bool(
                            torch.equal(
                                cast(Tensor, self.acquisition_age_seconds),
                                cast(Tensor, derived_age),
                            )
                        )
                    )
                    or (
                        None
                        if source_radar_index_map is None
                        else tensor_digest(source_radar_index_map)
                    )
                    != (
                        None
                        if derived_source_map is None
                        else tensor_digest(derived_source_map)
                    )
                    or (
                        self.contract
                        in {
                            "radar-verification-bundle-v14",
                            "radar-verification-bundle-v16",
                            "radar-verification-bundle-v17",
                        }
                        and (
                            type(derivation.raw_inputs.mask_derivation)
                            is not VerificationObservationMaskDerivationArtifact
                            or cast(
                                VerificationObservationMaskDerivationArtifact,
                                derivation.raw_inputs.mask_derivation,
                            ).geometry.contract
                            != (
                                "radar-observation-geometry-v6"
                                if self.contract == "radar-verification-bundle-v17"
                                else (
                                    "radar-observation-geometry-v5"
                                    if self.contract == "radar-verification-bundle-v16"
                                    else "radar-observation-geometry-v3"
                                )
                            )
                        )
                    )
                ):
                    raise ValueError(
                        "verification disagrees with preregistered temporal replay"
                    )
            _validate_verification_cell_states(
                frames_dbz=self.frames_dbz,
                valid_mask=self.valid_mask,
                quality_weight=quality,
                observation_std_dbz=observation_std,
                observation_state_code=observation_state,
                minimum_detectable_echo_dbz=(
                    error_contract.minimum_detectable_echo_dbz
                ),
                radar_source_kind=error_contract.radar_source_kind,
                source_radar_index_map=source_radar_index_map,
                detection_limit_dbz=(
                    cast(Tensor, self.detection_limit_dbz)
                    if self.contract in {
                        "radar-verification-bundle-v9",
                        "radar-verification-bundle-v10",
                        "radar-verification-bundle-v11",
                        "radar-verification-bundle-v12",
                        "radar-verification-bundle-v13",
                        "radar-verification-bundle-v14",
                        "radar-verification-bundle-v16",
                        "radar-verification-bundle-v17",
                    }
                    else None
                ),
            )
        frames = self.frames_dbz.detach().clone()
        valid = self.valid_mask.detach().clone()
        object.__setattr__(self, "frames_dbz", frames)
        object.__setattr__(self, "valid_mask", valid)
        if self.quality_weight is not None:
            object.__setattr__(self, "quality_weight", self.quality_weight.detach().clone())
        if self.observation_std_dbz is not None:
            object.__setattr__(
                self,
                "observation_std_dbz",
                self.observation_std_dbz.detach().clone(),
            )
        if self.observation_state_code is not None:
            object.__setattr__(
                self,
                "observation_state_code",
                self.observation_state_code.detach().clone(),
            )
        if self.source_radar_index_map is not None:
            object.__setattr__(
                self,
                "source_radar_index_map",
                self.source_radar_index_map.detach().clone(),
            )
        for name in (
            "detection_limit_dbz",
            "acquisition_time_offset_seconds",
            "acquisition_age_seconds",
            "spatial_metric_valid_mask",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.detach().clone())
        object.__setattr__(self, "valid_times", canonical_times)
        object.__setattr__(
            self,
            "content_digest",
            _verification_content_digest(
                self.contract,
                frames,
                valid,
                canonical_times,
                self.grid_contract_digest,
                self.radar_product_digest,
                self.qc_pipeline_digest,
                self.mask_policy_digest,
                self.censor_policy_digest,
                self.reflectivity_resolution_dbz,
                self.quantization_origin_dbz,
                self.threshold_bin_convention,
                self.floor_representation_contract_digest,
                self.quality_weight,
                self.observation_std_dbz,
                self.observation_state_code,
                self.source_radar_index_map,
                self.detection_limit_dbz,
                self.acquisition_time_offset_seconds,
                self.acquisition_age_seconds,
                self.spatial_metric_valid_mask,
                self.observation_error_contract,
                self.observation_error_derivation,
            ),
        )

    def validate_integrity(self) -> None:
        if self.observation_error_derivation is not None:
            self.observation_error_derivation.validate_replay()
        expected = _verification_content_digest(
            self.contract,
            self.frames_dbz,
            self.valid_mask,
            self.valid_times,
            self.grid_contract_digest,
            self.radar_product_digest,
            self.qc_pipeline_digest,
            self.mask_policy_digest,
            self.censor_policy_digest,
            self.reflectivity_resolution_dbz,
            self.quantization_origin_dbz,
            self.threshold_bin_convention,
            self.floor_representation_contract_digest,
            self.quality_weight,
            self.observation_std_dbz,
            self.observation_state_code,
            self.source_radar_index_map,
            self.detection_limit_dbz,
            self.acquisition_time_offset_seconds,
            self.acquisition_age_seconds,
            self.spatial_metric_valid_mask,
            self.observation_error_contract,
            self.observation_error_derivation,
        )
        if expected != self.content_digest:
            raise ValueError("verification bundle content digest mismatch")

    @property
    def metric_weight(self) -> Tensor:
        self.validate_integrity()
        if self.contract not in {
            "radar-verification-bundle-v6",
            "radar-verification-bundle-v7",
            "radar-verification-bundle-v8",
            "radar-verification-bundle-v9",
            "radar-verification-bundle-v10",
            "radar-verification-bundle-v11",
            "radar-verification-bundle-v12",
            "radar-verification-bundle-v13",
            "radar-verification-bundle-v14",
            "radar-verification-bundle-v16",
            "radar-verification-bundle-v17",
        }:
            return self.valid_mask.to(self.frames_dbz)
        quality = cast(Tensor, self.quality_weight)
        observation_std = cast(Tensor, self.observation_std_dbz)
        error_contract = cast(
            VerificationObservationErrorContract,
            self.observation_error_contract,
        )
        observation_state = cast(Tensor, self.observation_state_code)
        reference = observation_std.new_tensor(
            error_contract.observation_error_reference_std_dbz
        )
        inverse_variance = torch.where(
            self.valid_mask,
            (reference / observation_std.clamp_min(torch.finfo(observation_std.dtype).tiny)).square().clamp(max=1.0),
            torch.zeros_like(observation_std),
        )
        point_observation = observation_state != (
            VerificationCellState.BELOW_DETECTION_CENSORED
        )
        return (quality * inverse_variance * point_observation).detach().clone()

    @property
    def intensity_metric_weight(self) -> Tensor:
        """Weight only quantitative echo intensity, never clear categories."""

        weight = self.metric_weight
        if self.contract not in {
            "radar-verification-bundle-v11",
            "radar-verification-bundle-v12",
            "radar-verification-bundle-v13",
            "radar-verification-bundle-v14",
            "radar-verification-bundle-v16",
            "radar-verification-bundle-v17",
        }:
            return weight
        state = cast(Tensor, self.observation_state_code)
        return torch.where(
            state == VerificationCellState.OBSERVED_ECHO,
            weight,
            torch.zeros_like(weight),
        ).detach().clone()

    @property
    def spatial_metric_weight(self) -> Tensor:
        """Return metric weight restricted to preregistered spatial-age support."""

        weight = self.metric_weight
        if self.contract not in {
            "radar-verification-bundle-v12",
            "radar-verification-bundle-v13",
            "radar-verification-bundle-v14",
            "radar-verification-bundle-v16",
            "radar-verification-bundle-v17",
        }:
            return weight
        return torch.where(
            cast(Tensor, self.spatial_metric_valid_mask),
            weight,
            torch.zeros_like(weight),
        ).detach().clone()

    @property
    def fso_metric_weight(self) -> Tensor:
        """Frozen current FSO domain: quantitative echo with spatial support."""

        weight = self.intensity_metric_weight
        if self.contract not in {
            "radar-verification-bundle-v12",
            "radar-verification-bundle-v13",
            "radar-verification-bundle-v14",
            "radar-verification-bundle-v16",
            "radar-verification-bundle-v17",
        }:
            return weight
        return torch.where(
            cast(Tensor, self.spatial_metric_valid_mask),
            weight,
            torch.zeros_like(weight),
        ).detach().clone()


VerificationInput = Tensor | VerificationBundle

CONTEXT_FEATURE_NAMES_V13 = (
    "motion_dy",
    "motion_dx",
    "motion_speed",
    "log_growth",
    "motion_disagreement",
    "growth_disagreement",
    "motion_pair_conflict",
    "growth_pair_conflict",
    "tendency_pair_count",
    "tendency_source_observation",
    "tendency_source_background",
    "state_path_pair_count",
    "state_path_source_observation",
    "state_path_source_background",
    "state_path_conflict",
    "state_path_extrapolated",
    "state_path_age_available",
    "state_path_age_minutes",
    "state_path_psr_available",
    "log1p_state_path_minimum_psr",
    "growth_overlap_support_available",
    "log1p_minimum_growth_overlap_support",
    "growth_overlap_area_available",
    "log1p_minimum_growth_overlap_area_km2",
    "current_state_support_fraction",
    "background_contribution_fraction",
    "latest_observation_coverage",
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_integrated_echo",
    *tuple(
        f"motion_pair_selection_{selection.value.lower()}"
        for selection in TendencyPairSelection
    ),
    *tuple(
        f"growth_pair_selection_{selection.value.lower()}"
        for selection in TendencyPairSelection
    ),
    *tuple(
        f"state_path_mode_{selection.value.lower()}"
        for selection in TendencyPairSelection
    ),
    "phase_correlation_psr_available",
    "log1p_minimum_phase_correlation_psr",
    "projected_velocity_available",
    "projected_velocity_x_mps",
    "projected_velocity_y_mps",
    "projected_speed_mps",
    "motion_disagreement_mps_available",
    "motion_disagreement_mps",
    "area_weighted_echo_available",
    "log1p_linear_reflectivity_integral_km2",
    "grid_spacing_available",
    "grid_column_spacing_m",
    "grid_row_spacing_m",
)
CONTEXT_FEATURE_NAMES = (
    *CONTEXT_FEATURE_NAMES_V13,
    "observation_path_pair_count",
    "observation_path_conflict",
    "observation_path_extrapolated",
    "observation_path_age_available",
    "observation_path_age_minutes",
    "observation_path_psr_available",
    "log1p_observation_path_minimum_psr",
    "background_path_pair_count",
    "background_path_conflict",
    "background_path_extrapolated",
    "background_path_age_available",
    "background_path_age_minutes",
    "background_path_psr_available",
    "log1p_background_path_minimum_psr",
)


@dataclass(frozen=True)
class SensitivityConfig:
    """Fixed metric and compression choices for one sensitivity contract."""

    metric_names: tuple[str, ...] = DEFAULT_METRICS
    metric_domain: FSOMetricDomain = "issued"
    require_verification_lineage: bool = False
    required_verification_radar_product_digest: str | None = None
    required_verification_qc_pipeline_digest: str | None = None
    full_map_lead_minutes: tuple[int, ...] = (30, 60, 120, 180)
    tile_size: int = 16
    tile_size_m: float | None = None
    soft_fss_temperature_dbz: float = 2.0
    soft_fss_window: int = 9
    soft_fss_window_m: float | None = None
    minimum_fss_truth_mass: float = 0.5
    active_margin_dbz: float = 0.1
    linearity_delta: tuple[float, float, float] = (0.05, -0.04, 0.005)
    pair_conflict_trust_penalty: float = 0.5
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if not isinstance(self.metric_names, tuple):
            raise TypeError("metric_names must be a tuple")
        unknown = set(self.metric_names) - set(SUPPORTED_METRICS)
        if unknown:
            raise ValueError(f"unsupported metrics: {sorted(unknown)}")
        if not self.metric_names:
            raise ValueError("at least one metric is required")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names must be unique")
        if self.metric_domain not in (
            "issued",
            "radar_dynamics_anchored",
            "confidence_weighted",
        ):
            raise ValueError("unsupported FSO metric domain")
        if type(self.require_verification_lineage) is not bool:
            raise TypeError("require_verification_lineage must be Boolean")
        required_lineage = (
            self.required_verification_radar_product_digest,
            self.required_verification_qc_pipeline_digest,
        )
        if (required_lineage[0] is None) != (required_lineage[1] is None):
            raise ValueError(
                "verification radar product and QC digests must be paired"
            )
        if required_lineage[0] is not None:
            if not self.require_verification_lineage:
                raise ValueError(
                    "approved verification identities require lineage"
                )
            _require_sha256(
                "required_verification_radar_product_digest",
                required_lineage[0],
            )
            _require_sha256(
                "required_verification_qc_pipeline_digest",
                cast(str, required_lineage[1]),
            )
        if not isinstance(self.full_map_lead_minutes, tuple):
            raise TypeError("full_map_lead_minutes must be a tuple")
        if len(set(self.full_map_lead_minutes)) != len(
            self.full_map_lead_minutes
        ):
            raise ValueError("full_map_lead_minutes must be unique")
        if any(
            type(minutes) is not int or minutes <= 0
            for minutes in self.full_map_lead_minutes
        ):
            raise ValueError("full-map leads must be positive integers")
        if type(self.tile_size) is not int or self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        for name, value in (
            ("tile_size_m", self.tile_size_m),
            ("soft_fss_window_m", self.soft_fss_window_m),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(self.soft_fss_temperature_dbz)
            or self.soft_fss_temperature_dbz <= 0
        ):
            raise ValueError("soft_fss_temperature_dbz must be positive")
        if (
            type(self.soft_fss_window) is not int
            or self.soft_fss_window <= 0
            or self.soft_fss_window % 2 == 0
        ):
            raise ValueError("soft_fss_window must be a positive odd integer")
        if (
            not math.isfinite(self.minimum_fss_truth_mass)
            or self.minimum_fss_truth_mass <= 0
        ):
            raise ValueError("minimum_fss_truth_mass must be positive")
        if (
            not math.isfinite(self.active_margin_dbz)
            or self.active_margin_dbz <= 0
        ):
            raise ValueError("active_margin_dbz must be positive")
        if len(self.linearity_delta) != 3 or not all(
            math.isfinite(value) for value in self.linearity_delta
        ):
            raise ValueError("linearity_delta must contain three finite values")
        if (
            not math.isfinite(self.pair_conflict_trust_penalty)
            or not 0.0 < self.pair_conflict_trust_penalty <= 1.0
        ):
            raise ValueError("pair_conflict_trust_penalty must be in (0, 1]")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def digest(self) -> str:
        return dataclass_digest(self)

    @classmethod
    def for_automated_learning(
        cls,
        *,
        radar_product_digest: str,
        qc_pipeline_digest: str,
    ) -> SensitivityConfig:
        """Return the fail-closed metric policy for automated learning."""

        return cls(
            metric_names=(
                "log_echo_mse",
                "soft_fss_error_35",
                "centroid_error_m2",
            ),
            metric_domain="radar_dynamics_anchored",
            require_verification_lineage=True,
            required_verification_radar_product_digest=(
                radar_product_digest
            ),
            required_verification_qc_pipeline_digest=qc_pipeline_digest,
            tile_size_m=16_000.0,
            soft_fss_window_m=9_000.0,
        )


VariationalPreconditioner = Literal[
    "none",
    "prior_smoothness_diagonal",
]


@dataclass(frozen=True)
class VariationalAdjointConfig:
    """Execution and local-validity budget for delayed P1 adjoints."""

    lead_minutes: tuple[int, ...] | None = None
    pcg_relative_tolerance: float | None = None
    maximum_pcg_iterations: int | None = None
    maximum_normal_products: int = 10_000
    maximum_whitener_total_operations: int = 100_000_000_000
    maximum_materialized_output_bytes: int = 2 * 1024**3
    warm_start_by_metric: bool = True
    preconditioner: VariationalPreconditioner = (
        "prior_smoothness_diagonal"
    )
    minimum_detection_margin_dbz: float = 1.0e-3
    minimum_remap_fraction_margin: float = 1.0e-4
    minimum_output_cap_margin_dbz: float = 1.0e-3
    minimum_publication_margin: float = 1.0e-4
    minimum_neural_prior_valid_margin: float = 1.0e-3
    minimum_neural_prior_support_margin: float = 1.0e-3
    require_active_set_margin: bool = False
    minimum_reachability_margin: float = 1.0e-3
    minimum_unresolved_amplitude_fraction_margin: float = 1.0e-4
    minimum_amplitude_confidence_margin: float = 1.0e-3
    minimum_motion_saturation_margin_fraction: float = 1.0e-3
    minimum_motion_speed_saturation_margin_mps: float = 0.0
    minimum_growth_saturation_margin_per_step: float = 1.0e-4
    require_feasibility_margin: bool = False
    gauss_newton_probe_count: int = 4
    gauss_newton_probe_seed: int = 0
    maximum_gauss_newton_relative_curvature_defect: float = 0.25
    require_gauss_newton_reliability: bool = False
    maximum_detected_delta_dbz: float = 0.5
    maximum_censor_delta_dbz: float = 0.5
    maximum_observation_weight_delta: float = 0.1
    maximum_background_delta_dbz: float = 0.5
    maximum_perturbed_pixel_count: int = 4096
    maximum_perturbed_fraction: float = 0.05
    maximum_perturbed_area_km2: float | None = None
    maximum_whitened_perturbation_l2: float = 8.0
    perturbation_tile_size: int = 16
    perturbation_tile_size_m: float | None = None
    maximum_per_tile_whitened_norm: float = 4.0
    maximum_observation_weight_l2: float = 1.0
    minimum_observation_multiplier: float = 0.5
    require_baseline_dynamics_branch_validity: bool = False

    def __post_init__(self) -> None:
        if self.lead_minutes is not None:
            if not isinstance(self.lead_minutes, tuple):
                raise TypeError("adjoint lead_minutes must be a tuple")
            if not self.lead_minutes:
                raise ValueError("adjoint lead_minutes cannot be empty")
            if len(set(self.lead_minutes)) != len(self.lead_minutes):
                raise ValueError("adjoint lead_minutes must be unique")
            if any(
                type(value) is not int or value <= 0
                for value in self.lead_minutes
            ):
                raise ValueError(
                    "adjoint lead_minutes must contain positive integers"
                )
            if tuple(sorted(self.lead_minutes)) != self.lead_minutes:
                raise ValueError("adjoint lead_minutes must be increasing")
        if self.pcg_relative_tolerance is not None and (
            isinstance(self.pcg_relative_tolerance, bool)
            or not math.isfinite(self.pcg_relative_tolerance)
            or self.pcg_relative_tolerance <= 0.0
        ):
            raise ValueError("adjoint PCG tolerance must be positive")
        if self.maximum_pcg_iterations is not None and (
            type(self.maximum_pcg_iterations) is not int
            or self.maximum_pcg_iterations <= 0
        ):
            raise ValueError("adjoint PCG iterations must be positive")
        for name, value in (
            ("maximum_normal_products", self.maximum_normal_products),
            (
                "maximum_whitener_total_operations",
                self.maximum_whitener_total_operations,
            ),
            (
                "maximum_materialized_output_bytes",
                self.maximum_materialized_output_bytes,
            ),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warm_start_by_metric) is not bool:
            raise TypeError("warm_start_by_metric must be Boolean")
        if self.preconditioner not in (
            "none",
            "prior_smoothness_diagonal",
        ):
            raise ValueError("unsupported variational preconditioner")
        margins = (
            self.minimum_detection_margin_dbz,
            self.minimum_remap_fraction_margin,
            self.minimum_output_cap_margin_dbz,
            self.minimum_publication_margin,
            self.minimum_neural_prior_valid_margin,
            self.minimum_neural_prior_support_margin,
            self.minimum_reachability_margin,
            self.minimum_unresolved_amplitude_fraction_margin,
            self.minimum_amplitude_confidence_margin,
            self.minimum_motion_saturation_margin_fraction,
            self.minimum_motion_speed_saturation_margin_mps,
            self.minimum_growth_saturation_margin_per_step,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
            for value in margins
        ):
            raise ValueError("active-set margins must be finite and nonnegative")
        if type(self.require_active_set_margin) is not bool:
            raise TypeError("require_active_set_margin must be Boolean")
        if type(self.require_feasibility_margin) is not bool:
            raise TypeError("require_feasibility_margin must be Boolean")
        if (
            type(self.gauss_newton_probe_count) is not int
            or self.gauss_newton_probe_count <= 0
        ):
            raise ValueError("gauss_newton_probe_count must be positive")
        if (
            type(self.gauss_newton_probe_seed) is not int
            or self.gauss_newton_probe_seed < 0
        ):
            raise ValueError("gauss_newton_probe_seed cannot be negative")
        if (
            isinstance(
                self.maximum_gauss_newton_relative_curvature_defect,
                bool,
            )
            or not math.isfinite(
                self.maximum_gauss_newton_relative_curvature_defect
            )
            or self.maximum_gauss_newton_relative_curvature_defect < 0.0
        ):
            raise ValueError(
                "maximum Gauss-Newton curvature defect must be nonnegative"
            )
        if type(self.require_gauss_newton_reliability) is not bool:
            raise TypeError("require_gauss_newton_reliability must be Boolean")
        if type(self.require_baseline_dynamics_branch_validity) is not bool:
            raise TypeError(
                "require_baseline_dynamics_branch_validity must be Boolean"
            )
        perturbation_limits = (
            self.maximum_detected_delta_dbz,
            self.maximum_censor_delta_dbz,
            self.maximum_observation_weight_delta,
            self.maximum_background_delta_dbz,
            self.maximum_whitened_perturbation_l2,
            self.maximum_per_tile_whitened_norm,
            self.maximum_observation_weight_l2,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0.0
            for value in perturbation_limits
        ):
            raise ValueError("local FSOI perturbation limits must be positive")
        if (
            type(self.maximum_perturbed_pixel_count) is not int
            or self.maximum_perturbed_pixel_count <= 0
        ):
            raise ValueError(
                "maximum_perturbed_pixel_count must be a positive integer"
            )
        if (
            type(self.perturbation_tile_size) is not int
            or self.perturbation_tile_size <= 0
        ):
            raise ValueError(
                "perturbation_tile_size must be a positive integer"
            )
        for name, value in (
            ("maximum_perturbed_area_km2", self.maximum_perturbed_area_km2),
            ("perturbation_tile_size_m", self.perturbation_tile_size_m),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(self.maximum_perturbed_fraction)
            or not 0.0 < self.maximum_perturbed_fraction <= 1.0
        ):
            raise ValueError("maximum_perturbed_fraction must be in (0, 1]")
        if (
            not math.isfinite(self.minimum_observation_multiplier)
            or not 0.0 < self.minimum_observation_multiplier <= 1.0
        ):
            raise ValueError(
                "minimum_observation_multiplier must be in (0, 1]"
            )

    @property
    def digest(self) -> str:
        return dataclass_digest(self)

    @classmethod
    def for_automated_learning(cls) -> VariationalAdjointConfig:
        """Return local-validity gates required by automated learning."""

        return cls(
            lead_minutes=(30, 60, 120, 180),
            require_active_set_margin=True,
            require_feasibility_margin=True,
            require_gauss_newton_reliability=True,
            require_baseline_dynamics_branch_validity=True,
            maximum_perturbed_area_km2=256.0,
            perturbation_tile_size_m=16_000.0,
        )


@dataclass(frozen=True)
class DirectSensitivity:
    maps: Tensor
    norm: Tensor
    tile_norm: Tensor
    whitened_tile_norm: Tensor | None = None
    impact: Tensor | None = None
    tile_impact: Tensor | None = None
    reward: Tensor | None = None


@dataclass(frozen=True)
class VariationalSensitivityChannel:
    """One frozen-model observation-parameter sensitivity channel."""

    maps: Tensor
    norm_by_time: Tensor
    tile_norm_by_time: Tensor


@dataclass(frozen=True)
class VariationalObservationSensitivity:
    """Separated frozen-structure P1 observation sensitivities.

    ``detected_dbz`` is the observation-residual path retained by the
    original frozen-GN contract. ``initial_background_dbz`` is the direct
    and implicit path through the accepted first-frame values used to build
    the P1 initial background. ``baseline_dynamics_dbz`` is the direct and
    implicit path through continuous P0 displacement/growth, with pair,
    integer FFT peak, and every other discrete selection frozen.
    ``frozen_structure_input_dbz`` is the sum of these three dBZ paths.
    """

    detected_dbz: VariationalSensitivityChannel
    censor_threshold_dbz: VariationalSensitivityChannel
    observation_weight: VariationalSensitivityChannel
    initial_background_dbz: VariationalSensitivityChannel
    baseline_dynamics_dbz: VariationalSensitivityChannel
    frozen_structure_input_dbz: VariationalSensitivityChannel
    baseline_branch_trusted_frozen_structure_input_dbz: (
        VariationalSensitivityChannel | None
    )

    @property
    def trusted_frozen_structure_input_dbz(
        self,
    ) -> VariationalSensitivityChannel | None:
        """Compatibility alias for the narrower branch-trust name."""

        return self.baseline_branch_trusted_frozen_structure_input_dbz


@dataclass(frozen=True)
class VariationalActiveSetMargins:
    """Distances to discrete or piecewise-smooth FSO contract boundaries."""

    detection_classification_dbz: float | None
    analysis_remap_fraction: float
    forecast_remap_fraction: float
    output_cap_dbz: float | None
    publication_support: float
    publication_confidence: float | None
    neural_prior_valid_probability: float | None
    neural_prior_support_probability: float | None
    low_local_validity: bool


@dataclass(frozen=True)
class VariationalFeasibilityMargins:
    """Accepted P1 interiority relative to hard feasibility boundaries."""

    reachability_support: float
    unresolved_amplitude_fraction: float
    amplitude_confidence: float | None
    motion_saturation_fraction: float
    motion_speed_saturation_mps: float | None
    growth_saturation_per_step: float
    low_interior_validity: bool


def _variational_feasibility_margins(
    margins: AnalysisFeasibilityMargins,
    config: VariationalAdjointConfig,
) -> VariationalFeasibilityMargins:
    low_interior_validity = (
        margins.reachability_support
        < config.minimum_reachability_margin
        or margins.unresolved_amplitude_fraction
        < config.minimum_unresolved_amplitude_fraction_margin
        or margins.amplitude_confidence is None
        or margins.amplitude_confidence
        < config.minimum_amplitude_confidence_margin
        or margins.motion_saturation_fraction
        < config.minimum_motion_saturation_margin_fraction
        or (
            config.minimum_motion_speed_saturation_margin_mps > 0.0
            and (
                margins.motion_speed_saturation_mps is None
                or margins.motion_speed_saturation_mps
                < config.minimum_motion_speed_saturation_margin_mps
            )
        )
        or margins.growth_saturation_per_step
        < config.minimum_growth_saturation_margin_per_step
    )
    return VariationalFeasibilityMargins(
        reachability_support=margins.reachability_support,
        unresolved_amplitude_fraction=(
            margins.unresolved_amplitude_fraction
        ),
        amplitude_confidence=margins.amplitude_confidence,
        motion_saturation_fraction=margins.motion_saturation_fraction,
        motion_speed_saturation_mps=margins.motion_speed_saturation_mps,
        growth_saturation_per_step=margins.growth_saturation_per_step,
        low_interior_validity=low_interior_validity,
    )


@dataclass(frozen=True)
class VariationalGaussNewtonDiagnostics:
    """Random-probe defect of exact frozen curvature from GN curvature."""

    relative_curvature_defect: Tensor
    maximum_relative_curvature_defect: float
    reliable: bool
    normal_products: int
    exact_hessian_products: int


CURRENT_VARIATIONAL_FSO_CONTRACT = "p1-variational-fso-v23"
EXPLORATORY_VARIATIONAL_FSO_CONTRACT = (
    "p1-variational-fso-exploratory-v1"
)
CURRENT_VARIATIONAL_FSOI_CONTRACT = "p1-linearized-observation-impact-v19"
EXPLORATORY_VARIATIONAL_FSOI_CONTRACT = (
    "p1-linearized-observation-impact-exploratory-v1"
)
_FSO_VERIFICATION_CONTRACTS = {
    CURRENT_VARIATIONAL_FSO_CONTRACT: "radar-verification-bundle-v17",
    EXPLORATORY_VARIATIONAL_FSO_CONTRACT: "legacy-verification-tensor-v1",
}
_FSOI_FSO_CONTRACTS = {
    CURRENT_VARIATIONAL_FSOI_CONTRACT: CURRENT_VARIATIONAL_FSO_CONTRACT,
    EXPLORATORY_VARIATIONAL_FSOI_CONTRACT: (
        EXPLORATORY_VARIATIONAL_FSO_CONTRACT
    ),
}


@dataclass(frozen=True)
class VariationalFSO:
    """Digest-bound P1 FSO under one frozen final IRLS/GN model."""

    contract: str
    forecast_run_digest: str
    analysis_input_digest: str
    sensitivity_config_digest: str
    adjoint_config_digest: str
    linearization_contract: str
    linearization_digest: str
    verification_contract: str
    verification_bundle_digest: str
    verification_lineage_complete: bool
    verification_valid_times: tuple[str, ...] | None
    verification_grid_contract_digest: str | None
    verification_radar_product_digest: str | None
    verification_qc_pipeline_digest: str | None
    metric_contract_digest: str
    algorithm_bundle_digest: str
    numerical_runtime_digest: str
    variational_fso_digest: str
    sensitivity_scope: str
    baseline_dynamics_frozen: bool
    baseline_pair_selection_frozen: bool
    baseline_dynamics_branch_status: BaselineDynamicsBranchStatus
    metric_names: tuple[str, ...]
    metric_domain: FSOMetricDomain
    metric_domain_digest: str
    lead_minutes: tuple[int, ...]
    full_map_lead_minutes: tuple[int, ...]
    tile_size: int
    tile_shape_yx: TileShape
    forecast_scores: Tensor
    metric_available: Tensor
    metric_domain_weight_sum: Tensor
    metric_domain_weight_fraction: Tensor
    forecast_cap_active_mask: Tensor
    observation: VariationalObservationSensitivity
    adjoint_iterations: Tensor
    adjoint_relative_residual: Tensor
    adjoint_true_residual_norm: Tensor
    adjoint_normal_products: Tensor
    adjoint_warm_started: Tensor
    total_normal_products: int
    whitener_operations_per_apply: int
    observed_whitener_apply_count: int
    materialized_output_bytes: int
    neural_prior_adjoint_direction_maximum_defect: float
    active_set_margins: VariationalActiveSetMargins
    feasibility_margins: VariationalFeasibilityMargins
    gauss_newton_diagnostics: VariationalGaussNewtonDiagnostics


@dataclass(frozen=True)
class VariationalObservationPerturbation:
    """Explicit first-order perturbation applied to the P1 observation model.

    ``detected_dbz`` perturbs detected reflectivity values,
    ``censor_threshold_dbz`` perturbs the censor threshold for censored events,
    ``observation_weight`` perturbs the unit objective multiplier for each
    valid observation. Optional ``initial_background_dbz`` independently
    perturbs the accepted first-frame values used by the P1 initial
    background, while optional ``baseline_dynamics_dbz`` perturbs the input
    dBZ values that generated continuous P0 motion/growth under the retained
    pair/peak selection. All channels are local perturbations; full observation
    removal requires a separate re-solve rather than this first-order contract.
    """

    detected_dbz: Tensor
    censor_threshold_dbz: Tensor
    observation_weight: Tensor
    initial_background_dbz: Tensor | None = None
    baseline_dynamics_dbz: Tensor | None = None
    physical_radar_dbz_delta: Tensor | None = None
    perturbation_semantics: PerturbationSemantics = "augmented_parameter"
    contract: str = "p1-observation-perturbation-v7"

    @classmethod
    def from_radar_dbz_delta(
        cls,
        delta_dbz: Tensor,
        linearization: AnalysisLinearization,
        *,
        neural_prior_runner: NeuralPriorInferenceRunner | None = None,
        neural_prior_application: NeuralPriorApplication | None = None,
    ) -> VariationalObservationPerturbation:
        """Map one physical radar-value change through retained input paths."""

        observations = linearization.observations
        frozen = linearization.frozen
        _validate_perturbation_tensor(
            "physical_radar_dbz_delta",
            delta_dbz,
            observations,
            observations.detected_mask,
            active_domain="detected observations",
        )
        active_delta = torch.where(
            observations.detected_mask,
            delta_dbz,
            torch.zeros_like(delta_dbz),
        )
        _ = _physical_radar_input_margins(
            active_delta,
            observations,
            frozen,
        )
        detected, background, dynamics = _physical_radar_channels(
            active_delta,
            observations,
            frozen,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        return cls(
            detected_dbz=detected,
            censor_threshold_dbz=torch.zeros_like(active_delta),
            observation_weight=torch.zeros_like(active_delta),
            initial_background_dbz=background,
            baseline_dynamics_dbz=dynamics,
            physical_radar_dbz_delta=active_delta,
            perturbation_semantics="physical_radar_value",
        )

    @classmethod
    def from_censor_threshold_delta(
        cls,
        delta_dbz: Tensor,
        linearization: AnalysisLinearization,
    ) -> VariationalObservationPerturbation:
        """Perturb the threshold that defines retained censored events."""

        observations = linearization.observations
        _validate_perturbation_tensor(
            "censor_threshold_dbz",
            delta_dbz,
            observations,
            observations.censored_mask,
            active_domain="censored observations",
        )
        zeros = torch.zeros_like(delta_dbz)
        return cls(
            detected_dbz=zeros,
            censor_threshold_dbz=delta_dbz,
            observation_weight=zeros,
        )

    @classmethod
    def from_censored_event_weight_delta(
        cls,
        delta_weight: Tensor,
        linearization: AnalysisLinearization,
    ) -> VariationalObservationPerturbation:
        """Perturb inclusion weight only for retained censored events."""

        observations = linearization.observations
        _validate_perturbation_tensor(
            "observation_weight",
            delta_weight,
            observations,
            observations.censored_mask,
            active_domain="censored observations",
        )
        zeros = torch.zeros_like(delta_weight)
        return cls(
            detected_dbz=zeros,
            censor_threshold_dbz=zeros,
            observation_weight=delta_weight,
        )

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "detected_dbz": tensor_digest(self.detected_dbz),
                "censor_threshold_dbz": tensor_digest(
                    self.censor_threshold_dbz
                ),
                "observation_weight": tensor_digest(
                    self.observation_weight
                ),
                "initial_background_dbz": (
                    None
                    if self.initial_background_dbz is None
                    else tensor_digest(self.initial_background_dbz)
                ),
                "baseline_dynamics_dbz": (
                    None
                    if self.baseline_dynamics_dbz is None
                    else tensor_digest(self.baseline_dynamics_dbz)
                ),
                "physical_radar_dbz_delta": (
                    None
                    if self.physical_radar_dbz_delta is None
                    else tensor_digest(self.physical_radar_dbz_delta)
                ),
                "perturbation_semantics": self.perturbation_semantics,
            }
        )


@dataclass(frozen=True)
class VariationalPerturbationDiagnostics:
    perturbed_pixel_count: int
    perturbed_fraction: float
    perturbed_area_km2: float | None
    whitened_l2: float
    maximum_per_tile_whitened_norm: float
    observation_weight_l2: float
    minimum_input_floor_margin_dbz: float | None
    minimum_input_ceiling_margin_dbz: float | None
    directional_classification_valid: bool
    baseline_dynamics_branch_status: BaselineDynamicsBranchStatus
    baseline_dynamics_branch_signature_digest: str | None


def _physical_radar_channels(
    delta_dbz: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    detected = torch.where(
        observations.detected_mask,
        delta_dbz,
        torch.zeros_like(delta_dbz),
    )
    background = torch.zeros_like(delta_dbz)
    if frozen.observation_derived_initial_background:
        background[0] = torch.where(
            observations.detected_mask[0] & frozen.observed_mask[0],
            delta_dbz[0],
            torch.zeros_like(delta_dbz[0]),
        )
    elif frozen.neural_prior_dependency == "radar_dependent":
        if neural_prior_runner is None:
            raise ValueError("radar-dependent prior perturbation requires a runner")
        _validate_retained_prior_runner(
            frozen,
            neural_prior_runner,
            neural_prior_application,
        )
        background[0] = torch.where(
            _neural_prior_derivative_mask(frozen),
            neural_prior_runner.jvp(
                _require_bound_neural_prior_input(frozen),
                delta_dbz,
            ),
            torch.zeros_like(background[0]),
        )
    dynamics = torch.zeros_like(delta_dbz)
    if frozen.baseline_metadata.tendency_source is TendencySource.OBSERVATION:
        dynamics = torch.where(
            observations.detected_mask & frozen.observed_mask,
            delta_dbz,
            dynamics,
        )
    return detected, background, dynamics


def _validate_retained_prior_runner(
    frozen: FrozenOuterState,
    runner: NeuralPriorInferenceRunner,
    application: NeuralPriorApplication | None,
) -> None:
    raw = frozen.neural_prior_raw_background_dbz
    execution_digest = frozen.neural_prior_execution_contract_digest
    if raw is None or execution_digest is None:
        raise ValueError("neural-prior restart state is incomplete")
    if application is not None:
        if application.application_digest != frozen.neural_prior_application_digest:
            raise ValueError("neural-prior perturbation application mismatch")
        runner.reproduce(
            application,
            _require_bound_neural_prior_input(frozen),
        )
    runner.validate_retained_output(
        _require_bound_neural_prior_input(frozen),
        raw,
        execution_contract_digest=execution_digest,
    )


def _neural_prior_derivative_mask(
    frozen: FrozenOuterState,
) -> Tensor:
    """Return the retained interior branch of the consumed prior output."""

    valid = frozen.neural_prior_valid_mask
    raw = frozen.neural_prior_raw_background_dbz
    if valid is None or raw is None:
        raise ValueError("neural-prior derivative requires retained prior state")
    return valid


def _require_bound_neural_prior_input(
    frozen: FrozenOuterState,
) -> BoundNeuralPriorInput:
    bound_input = frozen.neural_prior_bound_input
    if bound_input is None:
        raise ValueError("neural-prior bound input is missing from restart state")
    return bound_input


def _neural_prior_support_margin(
    frozen: FrozenOuterState,
    application: NeuralPriorApplication | None,
) -> float | None:
    """Distance from retained soft support to its contracted hard branch."""

    if frozen.neural_prior_dependency is None:
        return None
    if application is None or (
        application.application_digest != frozen.neural_prior_application_digest
    ):
        return None
    selected = application.support_probability.masked_select(
        application.valid_mask
    )
    if selected.numel() == 0:
        return None
    return float(
        torch.amin(
            torch.abs(
                selected
                - application.state_contract.support_decision_probability
            )
        ).detach()
    )


def _neural_prior_valid_margin(
    frozen: FrozenOuterState,
    application: NeuralPriorApplication | None,
) -> float | None:
    """Distance from probabilistic validity to its retained hard branch."""

    if frozen.neural_prior_dependency is None:
        return None
    if application is None or (
        application.application_digest != frozen.neural_prior_application_digest
    ):
        return None
    if application.inference_evidence.validity_contract == "exogenous_static":
        return math.inf
    selected = application.valid_probability
    if selected.numel() == 0:
        return None
    return float(
        torch.amin(
            torch.abs(
                selected
                - application.state_contract.valid_decision_probability
            )
        ).detach()
    )


def _physical_radar_input_margins(
    delta_dbz: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> tuple[float | None, float | None]:
    """Fail closed when a physical perturbation crosses input clipping."""

    changed_mask = delta_dbz != 0
    if not bool(torch.any(changed_mask)):
        return None, None
    changed = (observations.dbz + delta_dbz).masked_select(changed_mask)
    floor_margin = float(
        torch.amin(changed - frozen.nowcast_config.min_dbz).detach()
    )
    ceiling_margin = float(
        torch.amin(frozen.nowcast_config.max_dbz - changed).detach()
    )
    if floor_margin < 0.0 or ceiling_margin < 0.0:
        raise ValueError("physical radar perturbation crosses input clamp")
    return floor_margin, ceiling_margin


@dataclass(frozen=True)
class VariationalImpactChannel:
    """Signed first-order metric change from one perturbation channel."""

    maps: Tensor
    sum_by_time: Tensor
    tile_sum_by_time: Tensor


@dataclass(frozen=True)
class VariationalObservationImpact:
    """Component and total signed P1 observation-impact estimates."""

    detected_dbz: VariationalImpactChannel
    censor_threshold_dbz: VariationalImpactChannel
    observation_weight: VariationalImpactChannel
    initial_background_dbz: VariationalImpactChannel
    baseline_dynamics_dbz: VariationalImpactChannel
    total: VariationalImpactChannel
    baseline_branch_trusted_total: VariationalImpactChannel | None

    @property
    def trusted_total(self) -> VariationalImpactChannel | None:
        """Compatibility alias; this certifies only the baseline branch."""

        return self.baseline_branch_trusted_total


@dataclass(frozen=True)
class VariationalFSOI:
    """Explicit-perturbation first-order impact derived from P1 FSO."""

    contract: str
    fso: VariationalFSO
    perturbation: VariationalObservationPerturbation
    perturbation_contract: str
    perturbation_digest: str
    perturbation_diagnostics: VariationalPerturbationDiagnostics
    baseline_dynamics_branch_status: BaselineDynamicsBranchStatus
    observation: VariationalObservationImpact
    variational_fsoi_digest: str


@dataclass(frozen=True)
class ObservationRemovalConfig:
    """Budget for an explicit, nonlocal observation-denial experiment."""

    maximum_removed_observation_count: int = 4096
    maximum_removed_fraction: float = 0.05
    maximum_removed_area_km2: float | None = 256.0
    maximum_whitener_total_operations: int = 100_000_000_000
    contract: str = "p1-observation-removal-config-v1"

    def __post_init__(self) -> None:
        if self.contract != "p1-observation-removal-config-v1":
            raise ValueError("unsupported observation-removal config")
        if (
            type(self.maximum_removed_observation_count) is not int
            or self.maximum_removed_observation_count <= 0
        ):
            raise ValueError("maximum removed observation count must be positive")
        if (
            isinstance(self.maximum_removed_fraction, bool)
            or not math.isfinite(self.maximum_removed_fraction)
            or not 0.0 < self.maximum_removed_fraction <= 1.0
        ):
            raise ValueError("maximum removed fraction must be in (0, 1]")
        if self.maximum_removed_area_km2 is not None and (
            isinstance(self.maximum_removed_area_km2, bool)
            or not math.isfinite(self.maximum_removed_area_km2)
            or self.maximum_removed_area_km2 <= 0.0
        ):
            raise ValueError("maximum removed area must be positive")
        if (
            type(self.maximum_whitener_total_operations) is not int
            or self.maximum_whitener_total_operations <= 0
        ):
            raise ValueError("removal whitener operation budget must be positive")

    @property
    def digest(self) -> str:
        return dataclass_digest(self)


@dataclass(frozen=True)
class ObservationRemovalRequest:
    """A set of accepted observations removed from the full P1 input."""

    removal_mask: Tensor
    linearization_digest: str
    contract: str = "p1-observation-removal-request-v1"
    request_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-observation-removal-request-v1":
            raise ValueError("unsupported observation-removal request")
        if (
            not isinstance(self.removal_mask, Tensor)
            or self.removal_mask.dtype is not torch.bool
            or self.removal_mask.ndim != 3
        ):
            raise TypeError("removal mask must be a Boolean [time, y, x] Tensor")
        if not bool(torch.any(self.removal_mask)):
            raise ValueError("removal mask must select at least one observation")
        _require_sha256("linearization_digest", self.linearization_digest)
        owned = self.removal_mask.detach().clone()
        object.__setattr__(self, "removal_mask", owned)
        object.__setattr__(
            self,
            "request_digest",
            json_digest(
                {
                    "contract": self.contract,
                    "removal_mask": tensor_digest(owned),
                    "linearization_digest": self.linearization_digest,
                }
            ),
        )


@dataclass(frozen=True)
class ObservationRemovalImpact:
    """Resolved forecast-error change after a full observation denial."""

    request: ObservationRemovalRequest
    nominal_scores: Tensor
    removed_scores: Tensor
    metric_change: Tensor
    metric_available: Tensor
    lead_minutes: tuple[int, ...]
    metric_names: tuple[str, ...]
    metric_domain: FSOMetricDomain
    nominal_forecast_digest: str
    removed_forecast_digest: str
    removed_linearization_digest: str
    verification_bundle_digest: str
    sensitivity_config_digest: str
    removal_config_digest: str
    removed_observation_count: int
    removed_fraction: float
    removed_area_km2: float | None
    whitener_operations_per_apply: int
    observed_whitener_apply_count: int
    observed_whitener_total_operations: int
    contract: str = "p1-resolved-observation-removal-impact-v1"
    observation_removal_impact_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-resolved-observation-removal-impact-v1":
            raise ValueError("unsupported observation-removal impact")
        for name in (
            "nominal_scores",
            "removed_scores",
            "metric_change",
            "metric_available",
        ):
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            object.__setattr__(self, name, value.detach().clone())
        expected_shape = (len(self.lead_minutes), len(self.metric_names))
        if any(
            getattr(self, name).shape != expected_shape
            for name in (
                "nominal_scores",
                "removed_scores",
                "metric_change",
                "metric_available",
            )
        ):
            raise ValueError("observation-removal metric shapes disagree")
        if self.metric_available.dtype is not torch.bool:
            raise TypeError("observation-removal availability must be Boolean")
        available = self.metric_available
        if not torch.allclose(
            self.metric_change[available],
            (self.removed_scores - self.nominal_scores)[available],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("observation-removal metric change is inconsistent")
        for name in (
            "nominal_forecast_digest",
            "removed_forecast_digest",
            "removed_linearization_digest",
            "verification_bundle_digest",
            "sensitivity_config_digest",
            "removal_config_digest",
        ):
            _require_sha256(name, getattr(self, name))
        if type(self.removed_observation_count) is not int or (
            self.removed_observation_count <= 0
        ):
            raise ValueError("removed observation count must be positive")
        if not 0.0 < self.removed_fraction <= 1.0:
            raise ValueError("removed observation fraction must be in (0, 1]")
        if self.removed_area_km2 is not None and self.removed_area_km2 <= 0.0:
            raise ValueError("removed observation area must be positive")
        for name in (
            "whitener_operations_per_apply",
            "observed_whitener_apply_count",
            "observed_whitener_total_operations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.observed_whitener_total_operations != (
            self.whitener_operations_per_apply
            * self.observed_whitener_apply_count
        ):
            raise ValueError("observation-removal whitener accounting mismatch")
        object.__setattr__(
            self,
            "observation_removal_impact_digest",
            _observation_removal_impact_digest(self),
        )


def _observation_removal_impact_digest(
    impact: ObservationRemovalImpact,
) -> str:
    return json_digest(
        {
            "contract": impact.contract,
            "request_digest": impact.request.request_digest,
            "nominal_scores": tensor_digest(impact.nominal_scores),
            "removed_scores": tensor_digest(impact.removed_scores),
            "metric_change": tensor_digest(impact.metric_change),
            "metric_available": tensor_digest(impact.metric_available),
            "lead_minutes": list(impact.lead_minutes),
            "metric_names": list(impact.metric_names),
            "metric_domain": impact.metric_domain,
            "nominal_forecast_digest": impact.nominal_forecast_digest,
            "removed_forecast_digest": impact.removed_forecast_digest,
            "removed_linearization_digest": (
                impact.removed_linearization_digest
            ),
            "verification_bundle_digest": impact.verification_bundle_digest,
            "sensitivity_config_digest": impact.sensitivity_config_digest,
            "removal_config_digest": impact.removal_config_digest,
            "removed_observation_count": impact.removed_observation_count,
            "removed_fraction": impact.removed_fraction,
            "removed_area_km2": impact.removed_area_km2,
            "whitener_operations_per_apply": (
                impact.whitener_operations_per_apply
            ),
            "observed_whitener_apply_count": (
                impact.observed_whitener_apply_count
            ),
            "observed_whitener_total_operations": (
                impact.observed_whitener_total_operations
            ),
        }
    )


@dataclass(frozen=True)
class MetricTaylorThreshold:
    """Dimensionally correct Taylor limits for one forecast metric."""

    metric_name: str
    maximum_absolute_error: float
    material_impact_threshold: float
    ranking_scale: float | None = None
    ranking_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.metric_name not in SUPPORTED_METRICS:
            raise ValueError("unsupported Taylor metric")
        if (
            isinstance(self.maximum_absolute_error, bool)
            or not math.isfinite(self.maximum_absolute_error)
            or self.maximum_absolute_error < 0.0
        ):
            raise ValueError("metric Taylor absolute error must be nonnegative")
        if (
            isinstance(self.material_impact_threshold, bool)
            or not math.isfinite(self.material_impact_threshold)
            or self.material_impact_threshold <= 0.0
        ):
            raise ValueError("metric material impact threshold must be positive")
        if self.ranking_scale is not None and (
            isinstance(self.ranking_scale, bool)
            or not math.isfinite(self.ranking_scale)
            or self.ranking_scale <= 0.0
        ):
            raise ValueError("metric ranking scale must be positive")
        if (
            isinstance(self.ranking_weight, bool)
            or not math.isfinite(self.ranking_weight)
            or self.ranking_weight < 0.0
        ):
            raise ValueError("metric ranking weight must be nonnegative")

    @property
    def effective_ranking_scale(self) -> float:
        return (
            self.material_impact_threshold
            if self.ranking_scale is None
            else self.ranking_scale
        )


DEFAULT_METRIC_TAYLOR_THRESHOLDS = (
    MetricTaylorThreshold("log_echo_mse", 1.0e-6, 1.0e-6),
    MetricTaylorThreshold("soft_fss_error_35", 1.0e-6, 1.0e-6),
    MetricTaylorThreshold("centroid_error_m2", 1.0, 1.0),
)


@dataclass(frozen=True)
class AutomatedLearningPolicy:
    """One externally approved bundle for automated FSOI learning."""

    sensitivity_config: SensitivityConfig
    adjoint_config: VariationalAdjointConfig
    algorithm_bundle_digest: str
    numerical_runtime_digest: str
    metric_taylor_thresholds: tuple[
        MetricTaylorThreshold, ...
    ] = DEFAULT_METRIC_TAYLOR_THRESHOLDS
    maximum_linearity_relative_error: float = 0.1
    ranking_objective: CandidateRankingObjective = "expected_error_reduction"
    ranking_lead_weights: tuple[float, ...] = ()
    maximum_candidate_count: int = 10_000
    maximum_learning_candidates_to_validate: int = 32
    maximum_total_robust_resolves: int = 64
    maximum_candidate_bytes: int = 64 * 1024**2
    maximum_candidate_nonzeros: int = 1_000_000
    maximum_candidate_scoring_operations: int = 1_000_000_000
    maximum_candidate_ranking_wall_seconds: float = 300.0
    maximum_learning_pcg_iterations: int = 100_000
    maximum_learning_wall_seconds: float = 3_600.0
    maximum_whitener_total_operations: int = 100_000_000_000
    contract: str = "p1-automated-learning-policy-v8"

    def __post_init__(self) -> None:
        if self.contract != "p1-automated-learning-policy-v8":
            raise ValueError("unsupported automated-learning policy")
        _require_sha256(
            "algorithm_bundle_digest",
            self.algorithm_bundle_digest,
        )
        _require_sha256(
            "numerical_runtime_digest",
            self.numerical_runtime_digest,
        )
        if (
            isinstance(self.maximum_linearity_relative_error, bool)
            or not math.isfinite(self.maximum_linearity_relative_error)
            or not 0.0 < self.maximum_linearity_relative_error < 1.0
        ):
            raise ValueError(
                "maximum_linearity_relative_error must be in (0, 1)"
            )
        sensitivity = self.sensitivity_config
        if (
            sensitivity.metric_domain != "radar_dynamics_anchored"
            or not sensitivity.require_verification_lineage
            or sensitivity.required_verification_radar_product_digest is None
            or sensitivity.required_verification_qc_pipeline_digest is None
            or sensitivity.tile_size_m is None
            or (
                "soft_fss_error_35" in sensitivity.metric_names
                and sensitivity.soft_fss_window_m is None
            )
            or "centroid_error" in sensitivity.metric_names
        ):
            raise ValueError(
                "automated learning requires physical metrics and approved "
                "verification lineage"
            )
        adjoint = self.adjoint_config
        if not all(
            (
                adjoint.require_active_set_margin,
                adjoint.require_feasibility_margin,
                adjoint.require_gauss_newton_reliability,
                adjoint.require_baseline_dynamics_branch_validity,
                adjoint.maximum_perturbed_area_km2 is not None,
                adjoint.perturbation_tile_size_m is not None,
            )
        ):
            raise ValueError(
                "automated learning requires every local-validity gate"
            )
        if adjoint.lead_minutes != sensitivity.full_map_lead_minutes:
            raise ValueError(
                "automated learning requires a full map for every adjoint lead"
            )
        adjoint_leads = adjoint.lead_minutes
        if adjoint_leads is None:
            raise ValueError("automated learning requires explicit adjoint leads")
        if self.ranking_objective not in (
            "absolute_influence",
            "expected_error_reduction",
            "two_sided_diagnostic",
        ):
            raise ValueError("unsupported candidate ranking objective")
        if not isinstance(self.ranking_lead_weights, tuple):
            raise TypeError("ranking_lead_weights must be a tuple")
        if self.ranking_lead_weights and len(self.ranking_lead_weights) != len(
            adjoint_leads
        ):
            raise ValueError("ranking lead weights must match adjoint leads")
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
            for value in self.ranking_lead_weights
        ):
            raise ValueError("ranking lead weights must be nonnegative")
        if self.ranking_lead_weights and not any(self.ranking_lead_weights):
            raise ValueError("at least one ranking lead weight must be positive")
        for name, value in (
            ("maximum_candidate_count", self.maximum_candidate_count),
            (
                "maximum_learning_candidates_to_validate",
                self.maximum_learning_candidates_to_validate,
            ),
            ("maximum_total_robust_resolves", self.maximum_total_robust_resolves),
            ("maximum_candidate_bytes", self.maximum_candidate_bytes),
            ("maximum_candidate_nonzeros", self.maximum_candidate_nonzeros),
            (
                "maximum_candidate_scoring_operations",
                self.maximum_candidate_scoring_operations,
            ),
            (
                "maximum_learning_pcg_iterations",
                self.maximum_learning_pcg_iterations,
            ),
            (
                "maximum_whitener_total_operations",
                self.maximum_whitener_total_operations,
            ),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_learning_candidates_to_validate > self.maximum_candidate_count:
            raise ValueError(
                "learning validation count cannot exceed candidate count"
            )
        required_resolves = 2 * self.maximum_learning_candidates_to_validate
        if self.maximum_total_robust_resolves < required_resolves:
            raise ValueError(
                "robust-resolve budget must cover full/half candidate checks"
            )
        if (
            self.adjoint_config.maximum_whitener_total_operations
            > self.maximum_whitener_total_operations
        ):
            raise ValueError(
                "adjoint whitener budget cannot exceed the learning budget"
            )
        if (
            isinstance(self.maximum_learning_wall_seconds, bool)
            or not math.isfinite(self.maximum_learning_wall_seconds)
            or self.maximum_learning_wall_seconds <= 0.0
        ):
            raise ValueError("maximum_learning_wall_seconds must be positive")
        if (
            isinstance(self.maximum_candidate_ranking_wall_seconds, bool)
            or not math.isfinite(self.maximum_candidate_ranking_wall_seconds)
            or self.maximum_candidate_ranking_wall_seconds <= 0.0
        ):
            raise ValueError(
                "maximum_candidate_ranking_wall_seconds must be positive"
            )
        if not isinstance(self.metric_taylor_thresholds, tuple):
            raise TypeError("metric_taylor_thresholds must be a tuple")
        names = tuple(
            threshold.metric_name
            for threshold in self.metric_taylor_thresholds
        )
        if len(set(names)) != len(names):
            raise ValueError("metric Taylor thresholds must be unique")
        missing = set(sensitivity.metric_names) - set(names)
        if missing:
            raise ValueError(
                f"missing metric Taylor thresholds: {sorted(missing)}"
            )
        if not any(
            self.threshold_for(name).ranking_weight > 0.0
            for name in sensitivity.metric_names
        ):
            raise ValueError("at least one ranking metric weight must be positive")
        object.__setattr__(
            self,
            "metric_taylor_thresholds",
            tuple(
                sorted(
                    self.metric_taylor_thresholds,
                    key=lambda threshold: threshold.metric_name,
                )
            ),
        )

    def threshold_for(self, metric_name: str) -> MetricTaylorThreshold:
        for threshold in self.metric_taylor_thresholds:
            if threshold.metric_name == metric_name:
                return threshold
        raise ValueError(f"missing Taylor threshold for {metric_name}")

    @property
    def ranking_adjoint_config(self) -> VariationalAdjointConfig:
        """Use strict numerics while deferring candidate-specific branch checks."""

        return replace(
            self.adjoint_config,
            require_baseline_dynamics_branch_validity=False,
        )

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "contract": self.contract,
                "sensitivity_config_digest": self.sensitivity_config.digest,
                "adjoint_config_digest": self.adjoint_config.digest,
                "algorithm_bundle_digest": self.algorithm_bundle_digest,
                "numerical_runtime_digest": self.numerical_runtime_digest,
                "metric_taylor_thresholds": [
                    {
                        "metric_name": threshold.metric_name,
                        "maximum_absolute_error": (
                            threshold.maximum_absolute_error
                        ),
                        "material_impact_threshold": (
                            threshold.material_impact_threshold
                        ),
                        "ranking_scale": threshold.effective_ranking_scale,
                        "ranking_weight": threshold.ranking_weight,
                    }
                    for threshold in self.metric_taylor_thresholds
                ],
                "maximum_linearity_relative_error": (
                    self.maximum_linearity_relative_error
                ),
                "ranking_objective": self.ranking_objective,
                "ranking_lead_weights": list(self.resolved_ranking_lead_weights),
                "maximum_candidate_count": self.maximum_candidate_count,
                "maximum_learning_candidates_to_validate": (
                    self.maximum_learning_candidates_to_validate
                ),
                "maximum_total_robust_resolves": (
                    self.maximum_total_robust_resolves
                ),
                "maximum_candidate_bytes": self.maximum_candidate_bytes,
                "maximum_candidate_nonzeros": self.maximum_candidate_nonzeros,
                "maximum_candidate_scoring_operations": (
                    self.maximum_candidate_scoring_operations
                ),
                "maximum_candidate_ranking_wall_seconds": (
                    self.maximum_candidate_ranking_wall_seconds
                ),
                "maximum_learning_pcg_iterations": (
                    self.maximum_learning_pcg_iterations
                ),
                "maximum_learning_wall_seconds": self.maximum_learning_wall_seconds,
                "maximum_whitener_total_operations": (
                    self.maximum_whitener_total_operations
                ),
                "perturbation_semantics": "physical_radar_value",
            }
        )

    @property
    def resolved_ranking_lead_weights(self) -> tuple[float, ...]:
        if self.ranking_lead_weights:
            return self.ranking_lead_weights
        lead_minutes = self.adjoint_config.lead_minutes
        if lead_minutes is None:
            raise RuntimeError("automated learning policy lacks adjoint leads")
        return (1.0,) * len(lead_minutes)


@dataclass(frozen=True)
class SparseRadarPerturbation:
    """Sparse physical radar-value delta used by candidate ranking."""

    flat_indices: Tensor
    delta_values: Tensor
    shape: tuple[int, int, int]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.flat_indices.ndim != 1
            or self.flat_indices.dtype != torch.int64
        ):
            raise ValueError("sparse candidate indices must be int64 and 1-D")
        if (
            self.delta_values.ndim != 1
            or not self.delta_values.is_floating_point()
            or self.delta_values.shape != self.flat_indices.shape
        ):
            raise ValueError("sparse candidate values must be floating and 1-D")
        if len(self.shape) != 3 or any(
            type(value) is not int or value <= 0 for value in self.shape
        ):
            raise ValueError("sparse candidate shape must be positive [3,H,W]")
        if self.shape[0] != 3:
            raise ValueError("sparse radar candidates require three input times")
        indices = self.flat_indices.detach().clone()
        values = self.delta_values.detach().clone()
        if indices.numel() == 0:
            raise ValueError("sparse candidate cannot be empty")
        if not bool(torch.all(torch.isfinite(values))) or bool(
            torch.any(values == 0)
        ):
            raise ValueError("sparse candidate values must be finite and nonzero")
        size = math.prod(self.shape)
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= size)):
            raise ValueError("sparse candidate index is outside its shape")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("sparse candidate indices must be unique")
        order = torch.argsort(indices)
        indices = indices[order]
        values = values[order]
        object.__setattr__(self, "flat_indices", indices)
        object.__setattr__(self, "delta_values", values)
        object.__setattr__(
            self,
            "digest",
            _sparse_radar_perturbation_digest(self),
        )

    @classmethod
    def from_dense(cls, delta_dbz: Tensor) -> SparseRadarPerturbation:
        if delta_dbz.ndim != 3 or not delta_dbz.is_floating_point():
            raise ValueError("dense radar candidate must be floating [3,H,W]")
        flat = delta_dbz.reshape(-1)
        indices = torch.nonzero(flat != 0, as_tuple=False).flatten()
        return cls(
            indices.to(torch.int64),
            flat[indices],
            (delta_dbz.shape[0], delta_dbz.shape[1], delta_dbz.shape[2]),
        )

    @property
    def nonzero_count(self) -> int:
        return self.flat_indices.numel()

    @property
    def retained_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.flat_indices, self.delta_values)
        )

    def materialize(self, reference: Tensor) -> Tensor:
        if self.digest != _sparse_radar_perturbation_digest(self):
            raise ValueError("sparse candidate digest mismatch")
        if tuple(reference.shape) != self.shape:
            raise ValueError("sparse candidate shape mismatch")
        result = reference.new_zeros(self.shape).reshape(-1)
        result.index_copy_(
            0,
            self.flat_indices.to(reference.device),
            self.delta_values.to(dtype=reference.dtype, device=reference.device),
        )
        return result.reshape(self.shape)


def _sparse_radar_perturbation_digest(
    perturbation: SparseRadarPerturbation,
) -> str:
    return json_digest(
        {
            "contract": "sparse-radar-perturbation-v1",
            "shape": list(perturbation.shape),
            "flat_indices": tensor_digest(perturbation.flat_indices),
            "delta_values": tensor_digest(perturbation.delta_values),
        }
    )


@dataclass(frozen=True)
class VariationalCandidatePrecheck:
    candidate_id: str
    perturbation_digest: str
    admissible: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        _require_sha256("candidate perturbation digest", self.perturbation_digest)
        if self.admissible == (self.rejection_reason is not None):
            raise ValueError("candidate precheck status and reason disagree")


@dataclass(frozen=True)
class VariationalCandidateScore:
    """One admissible candidate's dimensionless frozen-domain score."""

    candidate_id: str
    perturbation: SparseRadarPerturbation
    predicted_metric_change: Tensor
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a nonempty string")
        if not isinstance(self.predicted_metric_change, Tensor):
            raise TypeError("candidate prediction must be a Tensor")
        object.__setattr__(
            self,
            "predicted_metric_change",
            self.predicted_metric_change.detach().clone(),
        )
        if not math.isfinite(self.score) or self.score < 0.0:
            raise ValueError("candidate score must be finite and nonnegative")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("candidate rank must be a positive integer")


@dataclass(frozen=True)
class VariationalCandidateRanking:
    """Content-bound ranking produced from one shared FSO solve."""

    fso: VariationalFSO
    scores: tuple[VariationalCandidateScore, ...]
    prechecks: tuple[VariationalCandidatePrecheck, ...]
    policy_digest: str
    ranking_objective: CandidateRankingObjective
    candidate_count: int
    scoring_operations: int
    whitener_operations_per_apply: int
    observed_whitener_apply_count: int
    contract: str = "p1-variational-candidate-ranking-v2"
    ranking_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-variational-candidate-ranking-v2":
            raise ValueError("unsupported candidate ranking")
        if tuple(score.rank for score in self.scores) != tuple(
            range(1, len(self.scores) + 1)
        ):
            raise ValueError("candidate ranks must be contiguous")
        identifiers = tuple(score.candidate_id for score in self.scores)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate identifiers must be unique")
        if self.candidate_count != len(self.prechecks):
            raise ValueError("candidate count and prechecks disagree")
        precheck_by_id = {value.candidate_id: value for value in self.prechecks}
        if len(precheck_by_id) != len(self.prechecks):
            raise ValueError("candidate precheck identifiers must be unique")
        if any(
            score.candidate_id not in precheck_by_id
            or not precheck_by_id[score.candidate_id].admissible
            or precheck_by_id[score.candidate_id].perturbation_digest
            != score.perturbation.digest
            for score in self.scores
        ):
            raise ValueError("ranked candidates must pass their bound precheck")
        if type(self.scoring_operations) is not int or self.scoring_operations < 0:
            raise ValueError("candidate scoring operations must be nonnegative")
        if self.whitener_operations_per_apply < 0 or (
            self.observed_whitener_apply_count < 0
        ):
            raise ValueError("candidate whitener telemetry must be nonnegative")
        _require_sha256("candidate ranking policy digest", self.policy_digest)
        object.__setattr__(
            self,
            "ranking_digest",
            _variational_candidate_ranking_digest(self),
        )


def _variational_candidate_ranking_digest(
    ranking: VariationalCandidateRanking,
) -> str:
    return json_digest(
        {
            "contract": ranking.contract,
            "fso_digest": ranking.fso.variational_fso_digest,
            "policy_digest": ranking.policy_digest,
            "ranking_objective": ranking.ranking_objective,
            "candidate_count": ranking.candidate_count,
            "scoring_operations": ranking.scoring_operations,
            "whitener_operations_per_apply": (
                ranking.whitener_operations_per_apply
            ),
            "observed_whitener_apply_count": (
                ranking.observed_whitener_apply_count
            ),
            "prechecks": [dataclass_digest(value) for value in ranking.prechecks],
            "scores": [
                {
                    "candidate_id": score.candidate_id,
                    "perturbation_digest": _sparse_radar_perturbation_digest(
                        score.perturbation
                    ),
                    "prediction": tensor_digest(
                        score.predicted_metric_change
                    ),
                    "score": score.score,
                    "rank": score.rank,
                }
                for score in ranking.scores
            ],
        }
    )


@dataclass(frozen=True)
class RankedLearningOutcome:
    """One selected result with its complete candidate-ranking lineage."""

    candidate_id: str
    candidate_rank: int
    candidate_score: float
    ranking_digest: str
    result: VariationalLearningImpact

    def __post_init__(self) -> None:
        if not self.candidate_id or self.candidate_rank <= 0:
            raise ValueError("ranked learning outcome identity is invalid")
        if not math.isfinite(self.candidate_score) or self.candidate_score < 0.0:
            raise ValueError("ranked learning outcome score is invalid")
        _require_sha256("ranking_digest", self.ranking_digest)
        evidence = self.result.approval_evidence
        if evidence is not None and (
            evidence.selection_mode != "ranked_top_k"
            or evidence.candidate_id != self.candidate_id
            or evidence.candidate_rank != self.candidate_rank
            or evidence.candidate_score != self.candidate_score
            or evidence.ranking_digest != self.ranking_digest
        ):
            raise ValueError("ranked outcome and approval evidence disagree")


@dataclass(frozen=True)
class LearningEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    policy_digest: str

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise TypeError("learning eligibility must be Boolean")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason
            for reason in self.reasons
        ):
            raise ValueError("learning rejection reasons must be strings")
        if self.eligible == bool(self.reasons):
            raise ValueError("learning eligibility and reasons disagree")
        _require_sha256("learning policy digest", self.policy_digest)


@dataclass(frozen=True)
class FirstOrderValidation:
    """Full/half-step checks on one frozen-domain Taylor prediction."""

    source_fsoi_digest: str
    nominal_forecast_digest: str
    nominal_input_bundle_digest: str
    nominal_full_analysis_input_digest: str
    full_step_prediction: Tensor
    full_step_resolved_metric_change: Tensor
    full_step_absolute_error: Tensor
    half_step_prediction: Tensor
    half_step_resolved_metric_change: Tensor
    half_step_absolute_error: Tensor
    metric_available: Tensor
    full_step_resolved_analysis_converged: bool
    half_step_resolved_analysis_converged: bool
    active_branch_valid: bool
    full_step_valid: bool
    half_step_valid: bool
    sign_consistent_for_material_impacts: bool
    material_metric_count: int
    maximum_material_impact: float
    aggregate_material_impact_norm: float
    first_order_valid: bool
    full_step_analysis_digest: str | None
    half_step_analysis_digest: str | None
    full_step_forecast_digest: str | None
    half_step_forecast_digest: str | None
    full_step_input_bundle_digest: str | None = None
    half_step_input_bundle_digest: str | None = None
    full_step_pcg_iterations: int = 0
    half_step_pcg_iterations: int = 0
    observed_whitener_apply_count: int = 0
    frozen_domain_state_effect: Tensor | None = None
    issuance_policy_effect: Tensor | None = None
    end_to_end_issuance_effect: Tensor | None = None
    coverage_before: Tensor | None = None
    coverage_after: Tensor | None = None
    newly_issued_fraction: Tensor | None = None
    withdrawn_fraction: Tensor | None = None
    background_fallback_before: Tensor | None = None
    background_fallback_after: Tensor | None = None
    metric_domain_contract: FirstOrderMetricDomain = "frozen_metric_domain"
    contract: str = "p1-first-order-validation-v5"
    validation_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-first-order-validation-v5":
            raise ValueError("unsupported first-order validation contract")
        if self.metric_domain_contract not in (
            "frozen_metric_domain",
            "resolved_issuance_domain",
        ):
            raise ValueError("unsupported first-order metric domain")
        for name in (
            "source_fsoi_digest",
            "nominal_forecast_digest",
            "nominal_input_bundle_digest",
            "nominal_full_analysis_input_digest",
        ):
            _require_sha256(name, getattr(self, name))
        tensor_names = (
            "full_step_prediction",
            "full_step_resolved_metric_change",
            "full_step_absolute_error",
            "half_step_prediction",
            "half_step_resolved_metric_change",
            "half_step_absolute_error",
            "metric_available",
        )
        for name in tensor_names:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            object.__setattr__(self, name, value.detach().clone())
        issuance_names = (
            "frozen_domain_state_effect",
            "issuance_policy_effect",
            "end_to_end_issuance_effect",
            "coverage_before",
            "coverage_after",
            "newly_issued_fraction",
            "withdrawn_fraction",
            "background_fallback_before",
            "background_fallback_after",
        )
        issuance = tuple(getattr(self, name) for name in issuance_names)
        if self.metric_domain_contract == "resolved_issuance_domain":
            if any(value is None for value in issuance):
                raise ValueError("resolved issuance decomposition is incomplete")
            for name, value in zip(issuance_names, issuance, strict=True):
                assert value is not None
                object.__setattr__(self, name, value.detach().clone())
            state = self.frozen_domain_state_effect
            policy = self.issuance_policy_effect
            total = self.end_to_end_issuance_effect
            assert state is not None and policy is not None and total is not None
            if state.shape != self.full_step_prediction.shape or (
                policy.shape != state.shape or total.shape != state.shape
            ):
                raise ValueError("resolved issuance metric shapes disagree")
            if not torch.allclose(
                state + policy, total, rtol=0.0, atol=0.0, equal_nan=True
            ):
                raise ValueError("resolved issuance effects do not close")
            if not torch.allclose(
                total,
                self.full_step_resolved_metric_change,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise ValueError("end-to-end effect and resolved change disagree")
            for name in issuance_names[3:]:
                value = getattr(self, name)
                assert value is not None
                if value.shape != (self.full_step_prediction.shape[0],) or bool(
                    torch.any((value < 0.0) | (value > 1.0))
                ):
                    raise ValueError("resolved issuance coverage is invalid")
        elif any(value is not None for value in issuance):
            raise ValueError("frozen-domain validation cannot carry issuance effects")
        if type(self.material_metric_count) is not int or (
            self.material_metric_count < 0
        ):
            raise ValueError("material_metric_count must be nonnegative")
        for name in (
            "maximum_material_impact",
            "aggregate_material_impact_norm",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.material_metric_count == 0 and (
            self.maximum_material_impact != 0.0
            or self.aggregate_material_impact_norm != 0.0
        ):
            raise ValueError("empty material signal must have zero magnitude")
        if (
            self.aggregate_material_impact_norm
            < self.maximum_material_impact
        ):
            raise ValueError("material impact norm cannot be below its maximum")
        if self.first_order_valid and self.material_metric_count == 0:
            raise ValueError("first-order validity requires material impact")
        for name in ("full_step_pcg_iterations", "half_step_pcg_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            type(self.observed_whitener_apply_count) is not int
            or self.observed_whitener_apply_count < 0
        ):
            raise ValueError("observed whitener apply count must be nonnegative")
        for name in (
            "full_step_analysis_digest",
            "half_step_analysis_digest",
            "full_step_forecast_digest",
            "half_step_forecast_digest",
            "full_step_input_bundle_digest",
            "half_step_input_bundle_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(name, value)
        input_digests = (
            self.full_step_input_bundle_digest,
            self.half_step_input_bundle_digest,
        )
        if self.metric_domain_contract == "resolved_issuance_domain":
            if any(value is None for value in input_digests):
                raise ValueError("resolved issuance input lineage is incomplete")
        elif any(value is not None for value in input_digests):
            raise ValueError("frozen validation cannot carry resolved inputs")
        object.__setattr__(
            self,
            "validation_digest",
            first_order_validation_digest(self),
        )

    @property
    def total_resolved_pcg_iterations(self) -> int:
        return self.full_step_pcg_iterations + self.half_step_pcg_iterations


@dataclass(frozen=True)
class LearningApprovalEvidence:
    """Immutable identities that justify one eligible learning impact."""

    policy_digest: str
    trust_store_digest: str
    fsoi_digest: str
    full_step_analysis_digest: str
    half_step_analysis_digest: str
    full_step_forecast_digest: str
    half_step_forecast_digest: str
    first_order_validation_digest: str
    learning_impact_digest: str
    approved_action_digest: str | None = None
    nominal_input_bundle_digest: str | None = None
    nominal_full_analysis_input_digest: str | None = None
    selection_mode: LearningSelectionMode = "direct"
    candidate_id: str | None = None
    candidate_rank: int | None = None
    candidate_score: float | None = None
    candidate_perturbation_digest: str | None = None
    ranking_digest: str | None = None
    ranking_policy_digest: str | None = None
    ranking_objective: CandidateRankingObjective | None = None
    whitener_operations_per_apply: int = 0
    observed_whitener_apply_count: int = 0
    observed_whitener_total_operations: int = 0
    contract: str = "p1-learning-approval-evidence-v4"

    def __post_init__(self) -> None:
        if self.contract not in (
            "p1-learning-approval-evidence-v1",
            "p1-learning-approval-evidence-v2",
            "p1-learning-approval-evidence-v3",
            "p1-learning-approval-evidence-v4",
        ):
            raise ValueError("unsupported learning approval evidence")
        for name, value in (
            ("policy_digest", self.policy_digest),
            ("trust_store_digest", self.trust_store_digest),
            ("fsoi_digest", self.fsoi_digest),
            ("full_step_analysis_digest", self.full_step_analysis_digest),
            ("half_step_analysis_digest", self.half_step_analysis_digest),
            ("full_step_forecast_digest", self.full_step_forecast_digest),
            ("half_step_forecast_digest", self.half_step_forecast_digest),
            (
                "first_order_validation_digest",
                self.first_order_validation_digest,
            ),
            ("learning_impact_digest", self.learning_impact_digest),
        ):
            _require_sha256(name, value)
        action_values = (
            self.approved_action_digest,
            self.nominal_input_bundle_digest,
            self.nominal_full_analysis_input_digest,
        )
        if self.contract in (
            "p1-learning-approval-evidence-v3",
            "p1-learning-approval-evidence-v4",
        ):
            if self.contract == "p1-learning-approval-evidence-v3":
                action_values = action_values[:2]
            if any(value is None for value in action_values):
                raise ValueError("learning action lineage is incomplete")
            _require_sha256(
                "approved_action_digest",
                cast(str, self.approved_action_digest),
            )
            _require_sha256(
                "nominal_input_bundle_digest",
                cast(str, self.nominal_input_bundle_digest),
            )
            if self.contract == "p1-learning-approval-evidence-v4":
                _require_sha256(
                    "nominal_full_analysis_input_digest",
                    cast(str, self.nominal_full_analysis_input_digest),
                )
        elif any(value is not None for value in action_values):
            raise ValueError("legacy learning evidence cannot carry action lineage")
        ranked_values = (
            self.candidate_id,
            self.candidate_rank,
            self.candidate_score,
            self.candidate_perturbation_digest,
            self.ranking_digest,
            self.ranking_policy_digest,
            self.ranking_objective,
        )
        if self.contract == "p1-learning-approval-evidence-v1":
            if self.selection_mode != "direct" or any(
                value is not None for value in ranked_values
            ):
                raise ValueError("legacy learning evidence cannot carry ranking")
            if any(
                (
                    self.whitener_operations_per_apply,
                    self.observed_whitener_apply_count,
                    self.observed_whitener_total_operations,
                )
            ):
                raise ValueError("legacy learning evidence cannot carry telemetry")
        elif self.selection_mode == "direct":
            if any(value is not None for value in ranked_values):
                raise ValueError("direct learning evidence cannot carry ranking")
        elif self.selection_mode == "ranked_top_k":
            if any(value is None for value in ranked_values):
                raise ValueError("ranked learning evidence is incomplete")
            if not isinstance(self.candidate_id, str) or not self.candidate_id:
                raise ValueError("ranked candidate_id must be nonempty")
            if type(self.candidate_rank) is not int or self.candidate_rank <= 0:
                raise ValueError("ranked candidate_rank must be positive")
            if (
                isinstance(self.candidate_score, bool)
                or not math.isfinite(cast(float, self.candidate_score))
                or cast(float, self.candidate_score) < 0.0
            ):
                raise ValueError("ranked candidate_score must be nonnegative")
            _require_sha256(
                "candidate_perturbation_digest",
                cast(str, self.candidate_perturbation_digest),
            )
            _require_sha256("ranking_digest", cast(str, self.ranking_digest))
            _require_sha256(
                "ranking_policy_digest",
                cast(str, self.ranking_policy_digest),
            )
        else:
            raise ValueError("unsupported learning selection mode")
        for name in (
            "whitener_operations_per_apply",
            "observed_whitener_apply_count",
            "observed_whitener_total_operations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.observed_whitener_total_operations != (
            self.whitener_operations_per_apply
            * self.observed_whitener_apply_count
        ):
            raise ValueError("learning whitener operation accounting mismatch")

    @property
    def digest(self) -> str:
        if self.contract == "p1-learning-approval-evidence-v1":
            return json_digest(
                {
                    "policy_digest": self.policy_digest,
                    "trust_store_digest": self.trust_store_digest,
                    "fsoi_digest": self.fsoi_digest,
                    "full_step_analysis_digest": (
                        self.full_step_analysis_digest
                    ),
                    "half_step_analysis_digest": (
                        self.half_step_analysis_digest
                    ),
                    "full_step_forecast_digest": (
                        self.full_step_forecast_digest
                    ),
                    "half_step_forecast_digest": (
                        self.half_step_forecast_digest
                    ),
                    "first_order_validation_digest": (
                        self.first_order_validation_digest
                    ),
                    "learning_impact_digest": self.learning_impact_digest,
                    "contract": self.contract,
                }
            )
        return dataclass_digest(self)


@dataclass(frozen=True)
class VariationalLearningImpact:
    eligibility: LearningEligibility
    fsoi: VariationalFSOI | None
    first_order_validation: FirstOrderValidation | None
    frozen_domain_learning_impact: VariationalImpactChannel | None
    approval_evidence: LearningApprovalEvidence | None = None
    contract: str = "p1-variational-learning-impact-v2"
    learning_result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract != "p1-variational-learning-impact-v2":
            raise ValueError("unsupported variational learning impact")
        if self.frozen_domain_learning_impact is not None:
            object.__setattr__(
                self,
                "frozen_domain_learning_impact",
                _clone_variational_impact_channel(
                    self.frozen_domain_learning_impact
                ),
            )
        complete = (
            self.fsoi is not None
            and self.first_order_validation is not None
            and self.first_order_validation.first_order_valid
            and self.frozen_domain_learning_impact is not None
            and self.approval_evidence is not None
        )
        if self.eligibility.eligible != complete:
            raise ValueError("learning eligibility and impact disagree")
        if self.first_order_validation is not None and self.fsoi is None:
            raise ValueError("first-order validation requires FSOI")
        if not self.eligibility.eligible and self.approval_evidence is not None:
            raise ValueError("rejected learning impact cannot carry approval")
        object.__setattr__(
            self,
            "learning_result_digest",
            variational_learning_impact_digest(self),
        )
        validate_variational_learning_impact(self)


@dataclass
class _VariationalChannelAccumulator:
    maps: Tensor
    by_time: Tensor
    tile_by_time: Tensor


@dataclass(frozen=True)
class _VariationalAdjointSensitivity:
    detected_dbz: Tensor
    censor_threshold_dbz: Tensor
    observation_weight: Tensor


@dataclass(frozen=True)
class _VariationalAdjointSolve:
    sensitivity: _VariationalAdjointSensitivity
    solution: Tensor
    iterations: int
    relative_residual: float
    true_residual_norm: float
    normal_products: int
    warm_started: bool


@dataclass(frozen=True)
class _FrozenBaselineDynamicsPath:
    """Reusable VJP for observation-derived baseline motion and growth."""

    active_mask: Tensor
    nominal_dynamics: Tensor
    observation_pullback: Callable[[Tensor], tuple[Tensor]]


@dataclass(frozen=True)
class P0TendencyBranchSignature:
    pair_spans: tuple[tuple[int, int], ...]
    motion_pair_spans: tuple[tuple[int, int], ...]
    growth_pair_spans: tuple[tuple[int, int], ...]
    integer_peak_yx_by_pair: tuple[tuple[int, int], ...]
    peak_is_search_interior_by_pair: tuple[bool, ...]
    pair_available_by_span: tuple[bool, ...]
    growth_evidence_available_by_span: tuple[bool, ...]
    motion_remap_cells: tuple[RemapCell, RemapCell]
    motion_selection: TendencyPairSelection
    growth_selection: TendencyPairSelection
    motion_conflict: bool
    growth_conflict: bool


@dataclass
class _NormalProductBudget:
    maximum: int
    used: int = 0

    def apply(
        self,
        operator: Callable[[Tensor], Tensor],
        value: Tensor,
    ) -> Tensor:
        if self.used >= self.maximum:
            raise ValueError("P1 FSO normal-product budget exhausted")
        self.used += 1
        return operator(value)


def _metric_tile_shape(
    config: SensitivityConfig,
    grid: RadarGridTimeContract | None,
) -> TileShape:
    if config.tile_size_m is None:
        return config.tile_size, config.tile_size
    if grid is None:
        raise ValueError("physical sensitivity settings require a grid contract")
    if not grid.grid_axes_are_orthogonal:
        raise ValueError(
            "physical sensitivity tiles require orthogonal grid axes"
        )
    assert grid.pixel_to_projected_matrix_m is not None
    (a, b), (c, d) = grid.pixel_to_projected_matrix_m
    row_spacing = math.hypot(b, d)
    column_spacing = math.hypot(a, c)
    return (
        max(1, math.floor(config.tile_size_m / row_spacing + 0.5)),
        max(1, math.floor(config.tile_size_m / column_spacing + 0.5)),
    )


def _perturbation_tile_size(
    config: VariationalAdjointConfig,
    grid: RadarGridTimeContract | None,
) -> TileShape:
    if config.perturbation_tile_size_m is None:
        return config.perturbation_tile_size, config.perturbation_tile_size
    if grid is None:
        raise ValueError("physical perturbation tiles require a grid contract")
    if not grid.grid_axes_are_orthogonal:
        raise ValueError(
            "physical perturbation tiles require orthogonal grid axes"
        )
    assert grid.pixel_to_projected_matrix_m is not None
    (a, b), (c, d) = grid.pixel_to_projected_matrix_m
    return (
        max(
            1,
            math.floor(
                config.perturbation_tile_size_m / math.hypot(b, d) + 0.5
            ),
        ),
        max(
            1,
            math.floor(
                config.perturbation_tile_size_m / math.hypot(a, c) + 0.5
            ),
        ),
    )


def _metric_domain_weight(
    result: ForecastResult,
    verification_finite: Tensor,
    forecast_index: int,
    domain: FSOMetricDomain,
) -> Tensor:
    """Freeze the spatial domain/weight used by one forecast metric."""

    if verification_finite.dtype is not torch.bool:
        raise TypeError("verification_finite must be Boolean")
    if verification_finite.shape != result.valid_mask[forecast_index].shape:
        raise ValueError("verification domain must match one forecast lead")
    if domain == "radar_dynamics_anchored":
        eligible = result.radar_dynamics_anchored_valid_mask[forecast_index]
        weight = eligible.to(result.state.echo_linear)
    elif domain == "confidence_weighted":
        eligible = result.valid_mask[forecast_index]
        weight = result.forecast_confidence[forecast_index]
    elif domain == "issued":
        eligible = result.valid_mask[forecast_index]
        weight = eligible.to(result.state.echo_linear)
    else:
        raise ValueError("unsupported FSO metric domain")
    return torch.where(
        verification_finite & eligible,
        weight,
        torch.zeros_like(weight),
    ).detach()


def _metric_domain_digest(
    domain: FSOMetricDomain,
    lead_minutes: tuple[int, ...],
    weights: tuple[Tensor, ...],
) -> str:
    return json_digest(
        {
            "version": "p1-fso-metric-domain-v1",
            "domain": domain,
            "lead_minutes": list(lead_minutes),
            "weight_digests": [tensor_digest(weight) for weight in weights],
        }
    )


def _deterministic_unit_probe(
    reference: Tensor,
    *,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bits = torch.randint(
        0,
        2,
        reference.shape,
        generator=generator,
        dtype=torch.int8,
        device="cpu",
    )
    probe = (2 * bits - 1).to(
        dtype=reference.dtype,
        device=reference.device,
    )
    return probe / torch.linalg.vector_norm(probe)


def _gauss_newton_curvature_diagnostics(
    control: Tensor,
    residual_fn: Callable[[Tensor], Tensor],
    normal_product: Callable[[Tensor], Tensor],
    budget: _NormalProductBudget,
    config: VariationalAdjointConfig,
) -> VariationalGaussNewtonDiagnostics:
    """Compare exact frozen-objective and GN curvature on fixed probes."""

    def objective(value: Tensor) -> Tensor:
        residual = residual_fn(value)
        return 0.5 * torch.dot(residual, residual)

    objective_gradient = torch.func.grad(objective)
    defects = control.new_empty(config.gauss_newton_probe_count)
    products_before = budget.used
    for index in range(config.gauss_newton_probe_count):
        probe = _deterministic_unit_probe(
            control,
            seed=config.gauss_newton_probe_seed + index,
        )
        exact_product = cast(
            Tensor,
            torch.func.jvp(
                objective_gradient,
                (control,),
                (probe,),
            )[1],
        )
        gn_product = budget.apply(normal_product, probe)
        if not bool(
            torch.all(torch.isfinite(exact_product))
            & torch.all(torch.isfinite(gn_product))
        ):
            raise ValueError("P1 frozen curvature probe is not finite")
        denominator = torch.linalg.vector_norm(gn_product)
        defects[index] = (
            torch.linalg.vector_norm(exact_product - gn_product)
            / denominator
        )
    maximum = float(torch.amax(defects).detach())
    reliable = (
        math.isfinite(maximum)
        and maximum
        <= config.maximum_gauss_newton_relative_curvature_defect
    )
    diagnostics = VariationalGaussNewtonDiagnostics(
        relative_curvature_defect=defects.detach(),
        maximum_relative_curvature_defect=maximum,
        reliable=reliable,
        normal_products=budget.used - products_before,
        exact_hessian_products=config.gauss_newton_probe_count,
    )
    if config.require_gauss_newton_reliability and not reliable:
        raise ValueError("P1 Gauss-Newton curvature approximation is unreliable")
    return diagnostics


def _adjoint_lead_indices(
    config: VariationalAdjointConfig,
    all_lead_minutes: tuple[int, ...],
) -> tuple[int, ...]:
    if config.lead_minutes is None:
        return tuple(range(len(all_lead_minutes)))
    positions = {minutes: index for index, minutes in enumerate(all_lead_minutes)}
    missing = set(config.lead_minutes) - set(all_lead_minutes)
    if missing:
        raise ValueError(f"adjoint leads are outside the forecast: {sorted(missing)}")
    return tuple(positions[minutes] for minutes in config.lead_minutes)


def _prior_smoothness_diagonal_preconditioner(
    control: Tensor,
    frozen: FrozenOuterState,
) -> Callable[[Tensor], Tensor]:
    diagonal = torch.ones_like(control)
    field_size = frozen.active_field_index.numel()
    if field_size > 0 and frozen.smooth_edge_left_index.numel() > 0:
        edge_weight = (
            frozen.analysis_config.field_smoothness_weight
            * frozen.smooth_edge_physical_weight
        ).to(dtype=control.dtype, device=control.device)
        field_diagonal = diagonal[:field_size]
        field_diagonal.scatter_add_(
            0,
            frozen.smooth_edge_left_index,
            edge_weight,
        )
        field_diagonal.scatter_add_(
            0,
            frozen.smooth_edge_right_index,
            edge_weight,
        )

    def apply(value: Tensor) -> Tensor:
        return value / diagonal

    return apply


def _variational_preconditioner(
    control: Tensor,
    frozen: FrozenOuterState,
    config: VariationalAdjointConfig,
) -> Callable[[Tensor], Tensor] | None:
    if config.preconditioner == "none":
        return None
    return _prior_smoothness_diagonal_preconditioner(control, frozen)


def _variational_materialized_output_bytes(
    reference: Tensor,
    *,
    selected_count: int,
    lead_count: int,
    metric_count: int,
    height: int,
    width: int,
    tile_rows: int,
    tile_columns: int,
    include_impact: bool,
    gauss_newton_probe_count: int,
) -> int:
    channel_elements = (
        selected_count * metric_count * 3 * height * width
        + lead_count * metric_count * 3
        + lead_count * metric_count * 3 * tile_rows * tile_columns
    )
    # Six sensitivity channels are always materialized. Explicit FSOI adds
    # six signed impact channels (five parameters plus their total).
    channel_count = 12 if include_impact else 6
    float_elements = (
        channel_count * channel_elements
        + 3 * lead_count * metric_count
        + 2 * lead_count
        + gauss_newton_probe_count
    )
    bool_elements = (
        selected_count * height * width
        + 2 * lead_count * metric_count
    )
    int64_elements = 2 * lead_count * metric_count
    return (
        float_elements * reference.element_size()
        + bool_elements
        + int64_elements * 8
    )


def _minimum_masked_value(values: Tensor, mask: Tensor) -> float | None:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        return None
    result = float(torch.amin(selected).detach())
    return result if math.isfinite(result) else None


def _remap_fraction_margin(
    displacement_yx: Tensor,
    cell: RemapCell,
) -> float:
    cell_tensor = displacement_yx.new_tensor((cell.y, cell.x))
    fraction = displacement_yx - cell_tensor
    margin = torch.minimum(fraction, 1.0 - fraction)
    return max(0.0, float(torch.amin(margin).detach()))


def _analysis_remap_margin(
    displacement_yx: Tensor,
    cells: tuple[RemapCell, RemapCell],
) -> float:
    return min(
        _remap_fraction_margin((index + 1) * displacement_yx, cell)
        for index, cell in enumerate(cells)
    )


def _publication_margins(
    result: ForecastResult,
    forecast_indices: tuple[int, ...],
) -> tuple[float, float | None]:
    config = result.run.config
    support_margins = [
        torch.abs(
            result.forecast_source_support[list(forecast_indices)]
            - config.min_publish_support
        )
    ]
    if config.minimum_publish_verified_support is not None:
        support_margins.append(
            torch.abs(
                result.forecast_verified_support[list(forecast_indices)]
                - config.minimum_publish_verified_support
            )
        )
    if config.minimum_publish_observation_verified_support is not None:
        support_margins.append(
            torch.abs(
                result.forecast_observation_verified_support[
                    list(forecast_indices)
                ]
                - config.minimum_publish_observation_verified_support
            )
        )
    support_margin = min(
        float(torch.amin(values).detach()) for values in support_margins
    )
    if config.maximum_publish_background_fraction is not None:
        support_margin = min(
            support_margin,
            abs(
                result.metadata.background_contribution_fraction
                - config.maximum_publish_background_fraction
            ),
        )
    confidence_margin = None
    if config.minimum_publish_confidence is not None:
        confidence_margin = float(
            torch.amin(
                torch.abs(
                    result.forecast_confidence[list(forecast_indices)]
                    - config.minimum_publish_confidence
                )
            ).detach()
        )
    return support_margin, confidence_margin


def _variational_channel_digest_values(
    channel: VariationalSensitivityChannel,
) -> dict[str, str]:
    return {
        "maps": tensor_digest(channel.maps),
        "norm_by_time": tensor_digest(channel.norm_by_time),
        "tile_norm_by_time": tensor_digest(channel.tile_norm_by_time),
    }


def _variational_impact_digest_values(
    channel: VariationalImpactChannel,
) -> dict[str, str]:
    return {
        "maps": tensor_digest(channel.maps),
        "sum_by_time": tensor_digest(channel.sum_by_time),
        "tile_sum_by_time": tensor_digest(channel.tile_sum_by_time),
    }


def _clone_variational_impact_channel(
    channel: VariationalImpactChannel,
) -> VariationalImpactChannel:
    return VariationalImpactChannel(
        maps=channel.maps.detach().clone(),
        sum_by_time=channel.sum_by_time.detach().clone(),
        tile_sum_by_time=channel.tile_sum_by_time.detach().clone(),
    )


def _variational_impact_digest(channel: VariationalImpactChannel) -> str:
    return json_digest(
        {
            "contract": "p1-frozen-domain-learning-impact-v1",
            **_variational_impact_digest_values(channel),
        }
    )


def first_order_validation_digest(
    validation: FirstOrderValidation,
) -> str:
    """Content digest for full/half-step Taylor validation evidence."""

    return json_digest(
        {
            "contract": validation.contract,
            "source_fsoi_digest": validation.source_fsoi_digest,
            "nominal_forecast_digest": validation.nominal_forecast_digest,
            "nominal_input_bundle_digest": (
                validation.nominal_input_bundle_digest
            ),
            "nominal_full_analysis_input_digest": (
                validation.nominal_full_analysis_input_digest
            ),
            "metric_domain_contract": validation.metric_domain_contract,
            "full_step_prediction": tensor_digest(
                validation.full_step_prediction
            ),
            "full_step_resolved_metric_change": tensor_digest(
                validation.full_step_resolved_metric_change
            ),
            "full_step_absolute_error": tensor_digest(
                validation.full_step_absolute_error
            ),
            "half_step_prediction": tensor_digest(
                validation.half_step_prediction
            ),
            "half_step_resolved_metric_change": tensor_digest(
                validation.half_step_resolved_metric_change
            ),
            "half_step_absolute_error": tensor_digest(
                validation.half_step_absolute_error
            ),
            "metric_available": tensor_digest(validation.metric_available),
            "full_step_resolved_analysis_converged": (
                validation.full_step_resolved_analysis_converged
            ),
            "half_step_resolved_analysis_converged": (
                validation.half_step_resolved_analysis_converged
            ),
            "active_branch_valid": validation.active_branch_valid,
            "full_step_valid": validation.full_step_valid,
            "half_step_valid": validation.half_step_valid,
            "sign_consistent_for_material_impacts": (
                validation.sign_consistent_for_material_impacts
            ),
            "material_metric_count": validation.material_metric_count,
            "maximum_material_impact": validation.maximum_material_impact,
            "aggregate_material_impact_norm": (
                validation.aggregate_material_impact_norm
            ),
            "first_order_valid": validation.first_order_valid,
            "full_step_analysis_digest": (
                validation.full_step_analysis_digest
            ),
            "half_step_analysis_digest": (
                validation.half_step_analysis_digest
            ),
            "full_step_forecast_digest": (
                validation.full_step_forecast_digest
            ),
            "half_step_forecast_digest": (
                validation.half_step_forecast_digest
            ),
            "full_step_input_bundle_digest": (
                validation.full_step_input_bundle_digest
            ),
            "half_step_input_bundle_digest": (
                validation.half_step_input_bundle_digest
            ),
            "full_step_pcg_iterations": validation.full_step_pcg_iterations,
            "half_step_pcg_iterations": validation.half_step_pcg_iterations,
            "observed_whitener_apply_count": (
                validation.observed_whitener_apply_count
            ),
            "frozen_domain_state_effect": _optional_tensor_digest(
                validation.frozen_domain_state_effect
            ),
            "issuance_policy_effect": _optional_tensor_digest(
                validation.issuance_policy_effect
            ),
            "end_to_end_issuance_effect": _optional_tensor_digest(
                validation.end_to_end_issuance_effect
            ),
            "coverage_before": _optional_tensor_digest(
                validation.coverage_before
            ),
            "coverage_after": _optional_tensor_digest(validation.coverage_after),
            "newly_issued_fraction": _optional_tensor_digest(
                validation.newly_issued_fraction
            ),
            "withdrawn_fraction": _optional_tensor_digest(
                validation.withdrawn_fraction
            ),
            "background_fallback_before": _optional_tensor_digest(
                validation.background_fallback_before
            ),
            "background_fallback_after": _optional_tensor_digest(
                validation.background_fallback_after
            ),
        }
    )


def _optional_tensor_digest(value: Tensor | None) -> str | None:
    return None if value is None else tensor_digest(value)


def variational_learning_impact_digest(
    learning: VariationalLearningImpact,
) -> str:
    """Content digest for the final eligible or rejected learning result."""

    impact = learning.frozen_domain_learning_impact
    return json_digest(
        {
            "contract": learning.contract,
            "eligibility": {
                "eligible": learning.eligibility.eligible,
                "reasons": list(learning.eligibility.reasons),
                "policy_digest": learning.eligibility.policy_digest,
            },
            "fsoi_digest": (
                None
                if learning.fsoi is None
                else learning.fsoi.variational_fsoi_digest
            ),
            "first_order_validation_digest": (
                None
                if learning.first_order_validation is None
                else learning.first_order_validation.validation_digest
            ),
            "frozen_domain_learning_impact_digest": (
                None if impact is None else _variational_impact_digest(impact)
            ),
            "approval_evidence_digest": (
                None
                if learning.approval_evidence is None
                else learning.approval_evidence.digest
            ),
        }
    )


@dataclass(frozen=True)
class _ResolvedVerification:
    frames_dbz: Tensor
    valid_mask: Tensor
    metric_weight: Tensor
    contract: str
    content_digest: str
    lineage_complete: bool
    valid_times: tuple[str, ...] | None
    grid_contract_digest: str | None
    radar_product_digest: str | None
    qc_pipeline_digest: str | None


def _verification_content_digest(
    contract: str,
    frames_dbz: Tensor,
    valid_mask: Tensor,
    valid_times: tuple[str, ...] | None,
    grid_contract_digest: str | None,
    radar_product_digest: str | None,
    qc_pipeline_digest: str | None,
    mask_policy_digest: str | None = None,
    censor_policy_digest: str | None = None,
    reflectivity_resolution_dbz: float | None = None,
    quantization_origin_dbz: float | None = None,
    threshold_bin_convention: str | None = None,
    floor_representation_contract_digest: str | None = None,
    quality_weight: Tensor | None = None,
    observation_std_dbz: Tensor | None = None,
    observation_state_code: Tensor | None = None,
    source_radar_index_map: Tensor | None = None,
    detection_limit_dbz: Tensor | None = None,
    acquisition_time_offset_seconds: Tensor | None = None,
    acquisition_age_seconds: Tensor | None = None,
    spatial_metric_valid_mask: Tensor | None = None,
    observation_error_contract: VerificationObservationErrorContract | None = None,
    observation_error_derivation: ObservationErrorDerivationArtifact | None = None,
) -> str:
    payload: dict[str, object] = {
        "version": "verification-bundle-content-v2",
        "contract": contract,
        "frames_dbz": tensor_digest(frames_dbz),
        "valid_mask": tensor_digest(valid_mask),
        "valid_times": None if valid_times is None else list(valid_times),
        "grid_contract_digest": grid_contract_digest,
        "radar_product_digest": radar_product_digest,
        "qc_pipeline_digest": qc_pipeline_digest,
    }
    if contract in {
        "radar-verification-bundle-v2",
        "radar-verification-bundle-v3",
        "radar-verification-bundle-v4",
        "radar-verification-bundle-v5",
        "radar-verification-bundle-v6",
        "radar-verification-bundle-v7",
        "radar-verification-bundle-v8",
        "radar-verification-bundle-v9",
        "radar-verification-bundle-v10",
        "radar-verification-bundle-v11",
        "radar-verification-bundle-v12",
        "radar-verification-bundle-v13",
        "radar-verification-bundle-v14",
        "radar-verification-bundle-v16",
        "radar-verification-bundle-v17",
    }:
        payload.update(
            {
                "version": (
                    "verification-bundle-content-v3"
                    if contract == "radar-verification-bundle-v2"
                    else (
                        "verification-bundle-content-v4"
                        if contract == "radar-verification-bundle-v3"
                        else (
                            "verification-bundle-content-v5"
                            if contract == "radar-verification-bundle-v4"
                            else (
                                "verification-bundle-content-v6"
                                if contract == "radar-verification-bundle-v5"
                                else (
                                    "verification-bundle-content-v7"
                                    if contract == "radar-verification-bundle-v6"
                                    else (
                                        "verification-bundle-content-v8"
                                        if contract == "radar-verification-bundle-v7"
                                        else (
                                            "verification-bundle-content-v9"
                                            if contract
                                            == "radar-verification-bundle-v8"
                                            else "verification-bundle-content-v10"
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                "mask_policy_digest": mask_policy_digest,
                "censor_policy_digest": censor_policy_digest,
                "reflectivity_resolution_dbz": reflectivity_resolution_dbz,
                "quantization_origin_dbz": quantization_origin_dbz,
                "threshold_bin_convention": threshold_bin_convention,
                "floor_representation_contract_digest": (
                    floor_representation_contract_digest
                ),
            }
        )
        if contract == "radar-verification-bundle-v10":
            payload["version"] = "verification-bundle-content-v11"
        elif contract == "radar-verification-bundle-v11":
            payload["version"] = "verification-bundle-content-v12"
        elif contract == "radar-verification-bundle-v12":
            payload["version"] = "verification-bundle-content-v13"
        elif contract == "radar-verification-bundle-v13":
            payload["version"] = "verification-bundle-content-v14"
        elif contract == "radar-verification-bundle-v14":
            payload["version"] = "verification-bundle-content-v15"
        elif contract == "radar-verification-bundle-v16":
            payload["version"] = "verification-bundle-content-v16"
        elif contract == "radar-verification-bundle-v17":
            payload["version"] = "verification-bundle-content-v17"
    if contract in {
        "radar-verification-bundle-v4",
        "radar-verification-bundle-v5",
        "radar-verification-bundle-v6",
        "radar-verification-bundle-v7",
        "radar-verification-bundle-v8",
        "radar-verification-bundle-v9",
        "radar-verification-bundle-v10",
        "radar-verification-bundle-v11",
        "radar-verification-bundle-v12",
        "radar-verification-bundle-v13",
        "radar-verification-bundle-v14",
        "radar-verification-bundle-v16",
        "radar-verification-bundle-v17",
    }:
        payload.update(
            {
                "quality_weight": (
                    None if quality_weight is None else tensor_digest(quality_weight)
                ),
                "observation_std_dbz": (
                    None
                    if observation_std_dbz is None
                    else tensor_digest(observation_std_dbz)
                ),
                "observation_error_contract": (
                    None
                    if observation_error_contract is None
                    else observation_error_contract.payload
                    | {"contract_digest": observation_error_contract.contract_digest}
                ),
            }
        )
    if contract in {
        "radar-verification-bundle-v6",
        "radar-verification-bundle-v7",
        "radar-verification-bundle-v8",
        "radar-verification-bundle-v9",
        "radar-verification-bundle-v10",
        "radar-verification-bundle-v11",
        "radar-verification-bundle-v12",
        "radar-verification-bundle-v13",
        "radar-verification-bundle-v14",
        "radar-verification-bundle-v16",
        "radar-verification-bundle-v17",
    }:
        payload["observation_state_code"] = (
            None
            if observation_state_code is None
            else tensor_digest(observation_state_code)
        )
        payload["source_radar_index_map"] = (
            None
            if source_radar_index_map is None
            else tensor_digest(source_radar_index_map)
        )
    if contract in {
        "radar-verification-bundle-v7",
        "radar-verification-bundle-v8",
        "radar-verification-bundle-v9",
        "radar-verification-bundle-v10",
        "radar-verification-bundle-v11",
        "radar-verification-bundle-v12",
        "radar-verification-bundle-v13",
        "radar-verification-bundle-v14",
        "radar-verification-bundle-v16",
        "radar-verification-bundle-v17",
    }:
        payload["observation_error_derivation_digest"] = (
            None
            if observation_error_derivation is None
            else observation_error_derivation.artifact_digest
        )
    if contract in {
        "radar-verification-bundle-v9",
        "radar-verification-bundle-v10",
        "radar-verification-bundle-v11",
        "radar-verification-bundle-v12",
        "radar-verification-bundle-v13",
        "radar-verification-bundle-v14",
        "radar-verification-bundle-v16",
        "radar-verification-bundle-v17",
    }:
        payload.update(
            {
                "detection_limit_dbz": (
                    None
                    if detection_limit_dbz is None
                    else tensor_digest(detection_limit_dbz)
                ),
                "acquisition_time_offset_seconds": (
                    None
                    if acquisition_time_offset_seconds is None
                    else tensor_digest(acquisition_time_offset_seconds)
                ),
            }
        )
        if contract in {
            "radar-verification-bundle-v11",
            "radar-verification-bundle-v12",
            "radar-verification-bundle-v13",
            "radar-verification-bundle-v14",
            "radar-verification-bundle-v16",
            "radar-verification-bundle-v17",
        }:
            payload["acquisition_age_seconds"] = (
                None
                if acquisition_age_seconds is None
                else tensor_digest(acquisition_age_seconds)
            )
        if contract in {
            "radar-verification-bundle-v12",
            "radar-verification-bundle-v13",
            "radar-verification-bundle-v14",
            "radar-verification-bundle-v16",
            "radar-verification-bundle-v17",
        }:
            payload["spatial_metric_valid_mask"] = (
                None
                if spatial_metric_valid_mask is None
                else tensor_digest(spatial_metric_valid_mask)
            )
    return json_digest(payload)


def _validate_current_verification_projected_grid(
    verification: VerificationBundle,
    result: ForecastResult,
) -> None:
    """Bind current verification geometry to the forecast's one grid authority."""

    grid = result.run.grid_time_contract
    if grid is None or result.run.grid_time_contract_digest is None:
        raise ValueError(
            "verification lineage requires a forecast grid/time contract"
        )
    if verification.grid_contract_digest != result.run.grid_time_contract_digest:
        raise ValueError("verification and forecast grid contracts disagree")
    if verification.contract != "radar-verification-bundle-v17":
        return
    grid.validate_current_metric_domain_evidence()
    derivation = verification.observation_error_derivation
    if (
        type(derivation) is not ObservationErrorDerivationArtifact
        or type(derivation.raw_inputs.mask_derivation)
        is not VerificationObservationMaskDerivationArtifact
    ):
        raise ValueError("current verification geometry lineage is incomplete")
    geometry = cast(
        VerificationObservationMaskDerivationArtifact,
        derivation.raw_inputs.mask_derivation,
    ).geometry
    if (
        grid.spatial_grid_contract != "radar-spatial-grid-identity-v5"
        or geometry.contract != "radar-observation-geometry-v6"
        or geometry.grid_contract_digest != grid.digest
        or type(geometry.projected_grid_identity)
        is not RadarSpatialGridIdentity
        or geometry.projected_grid_identity.digest != grid.spatial_grid_digest
        or geometry.projected_grid_identity.metric_domain_evidence_digest
        != CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
        or geometry.projected_grid_identity.shape_yx
        != tuple(verification.frames_dbz.shape[-2:])
    ):
        raise ValueError(
            "verification and forecast projected-grid geometry disagree"
        )


def _resolve_verification(
    verification: VerificationInput,
    result: ForecastResult,
    sensitivity_config: SensitivityConfig,
) -> _ResolvedVerification:
    if isinstance(verification, VerificationBundle):
        verification.validate_integrity()
        resolved = _ResolvedVerification(
            frames_dbz=verification.frames_dbz,
            valid_mask=verification.valid_mask,
            metric_weight=verification.fso_metric_weight,
            contract=verification.contract,
            content_digest=verification.content_digest,
            lineage_complete=True,
            valid_times=verification.valid_times,
            grid_contract_digest=verification.grid_contract_digest,
            radar_product_digest=verification.radar_product_digest,
            qc_pipeline_digest=verification.qc_pipeline_digest,
        )
    else:
        valid = torch.isfinite(verification)
        contract = "legacy-verification-tensor-v1"
        resolved = _ResolvedVerification(
            frames_dbz=verification,
            valid_mask=valid,
            metric_weight=valid.to(verification),
            contract=contract,
            content_digest=_verification_content_digest(
                contract,
                verification,
                valid,
                None,
                None,
                None,
                None,
            ),
            lineage_complete=False,
            valid_times=None,
            grid_contract_digest=None,
            radar_product_digest=None,
            qc_pipeline_digest=None,
        )
    if sensitivity_config.require_verification_lineage and not (
        resolved.lineage_complete
    ):
        raise ValueError(
            "complete verification lineage requires VerificationBundle"
        )
    if not resolved.lineage_complete:
        return resolved
    if not isinstance(verification, VerificationBundle):
        raise ValueError("complete verification requires its typed bundle")
    _validate_current_verification_projected_grid(verification, result)
    grid = result.run.grid_time_contract
    assert grid is not None
    issue_time = datetime.fromisoformat(
        grid.valid_times[-1].replace("Z", "+00:00")
    )
    expected_times = tuple(
        (issue_time + timedelta(minutes=minutes))
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
        for minutes in range(
            result.run.config.interval_minutes,
            result.run.config.horizon_minutes + 1,
            result.run.config.interval_minutes,
        )
    )
    if resolved.valid_times != expected_times:
        raise ValueError(
            "verification valid times do not match forecast issue and leads"
        )
    required_product = (
        sensitivity_config.required_verification_radar_product_digest
    )
    required_qc = sensitivity_config.required_verification_qc_pipeline_digest
    if required_product is not None and (
        resolved.radar_product_digest != required_product
        or resolved.qc_pipeline_digest != required_qc
    ):
        raise ValueError("verification product or QC identity is not approved")
    return resolved


def _validate_verification_lineage_fields(
    *,
    fso_contract: str,
    contract: str,
    content_digest: str,
    lineage_complete: bool,
    valid_times: tuple[str, ...] | None,
    grid_contract_digest: str | None,
    radar_product_digest: str | None,
    qc_pipeline_digest: str | None,
) -> None:
    _require_sha256("verification_bundle_digest", content_digest)
    expected_contract = _FSO_VERIFICATION_CONTRACTS.get(fso_contract)
    if expected_contract is None:
        raise ValueError("unsupported P1 FSO contract")
    if contract != expected_contract:
        raise ValueError(
            "FSO verification generation disagrees with its contract"
        )
    if type(lineage_complete) is not bool:
        raise TypeError("verification_lineage_complete must be Boolean")
    lineage_values = (
        grid_contract_digest,
        radar_product_digest,
        qc_pipeline_digest,
    )
    if not lineage_complete:
        if contract != "legacy-verification-tensor-v1":
            raise ValueError("incomplete verification must use legacy contract")
        if valid_times is not None or any(
            value is not None for value in lineage_values
        ):
            raise ValueError("incomplete verification cannot claim lineage")
        return
    if contract != "radar-verification-bundle-v17":
        raise ValueError("complete verification has the wrong contract")
    if valid_times is None or not valid_times:
        raise ValueError("complete verification requires valid times")
    canonical_times = tuple(
        _canonical_verification_time(value) for value in valid_times
    )
    parsed_times = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in canonical_times
    )
    if canonical_times != valid_times or any(
        later <= earlier
        for earlier, later in zip(parsed_times, parsed_times[1:])
    ):
        raise ValueError(
            "verification valid times must be canonical UTC and increasing"
        )
    for name, value in zip(
        (
            "verification_grid_contract_digest",
            "verification_radar_product_digest",
            "verification_qc_pipeline_digest",
        ),
        lineage_values,
    ):
        if value is None:
            raise ValueError(f"{name} is required")
        _require_sha256(name, value)


def _metric_contract_digest(config: SensitivityConfig) -> str:
    return json_digest(
        {
            "version": "p1-forecast-metric-contract-v4",
            "metric_names": list(config.metric_names),
            "metric_domain": config.metric_domain,
            "sensitivity_config_digest": config.digest,
        }
    )


def variational_fso_digest(fso: VariationalFSO) -> str:
    """Content digest for a complete frozen-model FSO result."""

    trusted_input = (
        fso.observation.baseline_branch_trusted_frozen_structure_input_dbz
    )
    return json_digest(
        {
            "version": "p1-variational-fso-digest-v17",
            "contract": fso.contract,
            "forecast_run_digest": fso.forecast_run_digest,
            "analysis_input_digest": fso.analysis_input_digest,
            "sensitivity_config_digest": fso.sensitivity_config_digest,
            "adjoint_config_digest": fso.adjoint_config_digest,
            "linearization_contract": fso.linearization_contract,
            "linearization_digest": fso.linearization_digest,
            "verification_contract": fso.verification_contract,
            "verification_bundle_digest": fso.verification_bundle_digest,
            "verification_lineage_complete": (
                fso.verification_lineage_complete
            ),
            "verification_valid_times": (
                None
                if fso.verification_valid_times is None
                else list(fso.verification_valid_times)
            ),
            "verification_grid_contract_digest": (
                fso.verification_grid_contract_digest
            ),
            "verification_radar_product_digest": (
                fso.verification_radar_product_digest
            ),
            "verification_qc_pipeline_digest": (
                fso.verification_qc_pipeline_digest
            ),
            "metric_contract_digest": fso.metric_contract_digest,
            "algorithm_bundle_digest": fso.algorithm_bundle_digest,
            "numerical_runtime_digest": fso.numerical_runtime_digest,
            "sensitivity_scope": fso.sensitivity_scope,
            "baseline_dynamics_frozen": fso.baseline_dynamics_frozen,
            "baseline_pair_selection_frozen": (
                fso.baseline_pair_selection_frozen
            ),
            "baseline_dynamics_branch_status": (
                fso.baseline_dynamics_branch_status
            ),
            "metric_names": list(fso.metric_names),
            "metric_domain": fso.metric_domain,
            "metric_domain_digest": fso.metric_domain_digest,
            "lead_minutes": list(fso.lead_minutes),
            "full_map_lead_minutes": list(fso.full_map_lead_minutes),
            "tile_size": fso.tile_size,
            "tile_shape_yx": list(fso.tile_shape_yx),
            "forecast_scores": tensor_digest(fso.forecast_scores),
            "metric_available": tensor_digest(fso.metric_available),
            "metric_domain_weight_sum": tensor_digest(
                fso.metric_domain_weight_sum
            ),
            "metric_domain_weight_fraction": tensor_digest(
                fso.metric_domain_weight_fraction
            ),
            "forecast_cap_active_mask": tensor_digest(
                fso.forecast_cap_active_mask
            ),
            "observation": {
                "detected_dbz": _variational_channel_digest_values(
                    fso.observation.detected_dbz
                ),
                "censor_threshold_dbz": (
                    _variational_channel_digest_values(
                        fso.observation.censor_threshold_dbz
                    )
                ),
                "observation_weight": _variational_channel_digest_values(
                    fso.observation.observation_weight
                ),
                "initial_background_dbz": (
                    _variational_channel_digest_values(
                        fso.observation.initial_background_dbz
                    )
                ),
                "baseline_dynamics_dbz": (
                    _variational_channel_digest_values(
                        fso.observation.baseline_dynamics_dbz
                    )
                ),
                "frozen_structure_input_dbz": (
                    _variational_channel_digest_values(
                        fso.observation.frozen_structure_input_dbz
                    )
                ),
                "baseline_branch_trusted_frozen_structure_input_dbz": (
                    None
                    if trusted_input is None
                    else _variational_channel_digest_values(trusted_input)
                ),
            },
            "adjoint_iterations": tensor_digest(fso.adjoint_iterations),
            "adjoint_relative_residual": tensor_digest(
                fso.adjoint_relative_residual
            ),
            "adjoint_true_residual_norm": tensor_digest(
                fso.adjoint_true_residual_norm
            ),
            "adjoint_normal_products": tensor_digest(
                fso.adjoint_normal_products
            ),
            "adjoint_warm_started": tensor_digest(
                fso.adjoint_warm_started
            ),
            "total_normal_products": fso.total_normal_products,
            "whitener_operations_per_apply": (
                fso.whitener_operations_per_apply
            ),
            "observed_whitener_apply_count": (
                fso.observed_whitener_apply_count
            ),
            "materialized_output_bytes": fso.materialized_output_bytes,
            "neural_prior_adjoint_direction_maximum_defect": (
                fso.neural_prior_adjoint_direction_maximum_defect
            ),
            "active_set_margins": {
                "detection_classification_dbz": (
                    fso.active_set_margins.detection_classification_dbz
                ),
                "analysis_remap_fraction": (
                    fso.active_set_margins.analysis_remap_fraction
                ),
                "forecast_remap_fraction": (
                    fso.active_set_margins.forecast_remap_fraction
                ),
                "output_cap_dbz": fso.active_set_margins.output_cap_dbz,
                "publication_support": (
                    fso.active_set_margins.publication_support
                ),
                "publication_confidence": (
                    fso.active_set_margins.publication_confidence
                ),
                "neural_prior_valid_probability": (
                    fso.active_set_margins.neural_prior_valid_probability
                ),
                "neural_prior_support_probability": (
                    fso.active_set_margins.neural_prior_support_probability
                ),
                "low_local_validity": (
                    fso.active_set_margins.low_local_validity
                ),
            },
            "feasibility_margins": {
                "reachability_support": (
                    fso.feasibility_margins.reachability_support
                ),
                "unresolved_amplitude_fraction": (
                    fso.feasibility_margins
                    .unresolved_amplitude_fraction
                ),
                "amplitude_confidence": (
                    fso.feasibility_margins.amplitude_confidence
                ),
                "motion_saturation_fraction": (
                    fso.feasibility_margins.motion_saturation_fraction
                ),
                "motion_speed_saturation_mps": (
                    fso.feasibility_margins
                    .motion_speed_saturation_mps
                ),
                "growth_saturation_per_step": (
                    fso.feasibility_margins.growth_saturation_per_step
                ),
                "low_interior_validity": (
                    fso.feasibility_margins.low_interior_validity
                ),
            },
            "gauss_newton_diagnostics": {
                "relative_curvature_defect": tensor_digest(
                    fso.gauss_newton_diagnostics.relative_curvature_defect
                ),
                "maximum_relative_curvature_defect": (
                    fso.gauss_newton_diagnostics
                    .maximum_relative_curvature_defect
                ),
                "reliable": fso.gauss_newton_diagnostics.reliable,
                "normal_products": (
                    fso.gauss_newton_diagnostics.normal_products
                ),
                "exact_hessian_products": (
                    fso.gauss_newton_diagnostics.exact_hessian_products
                ),
            },
        }
    )


def variational_fsoi_digest(fsoi: VariationalFSOI) -> str:
    """Content digest for one explicit first-order impact product."""

    return json_digest(
        {
            "version": "p1-variational-fsoi-digest-v16",
            "contract": fsoi.contract,
            "variational_fso_digest": fsoi.fso.variational_fso_digest,
            "perturbation_contract": fsoi.perturbation_contract,
            "perturbation_digest": fsoi.perturbation_digest,
            "perturbation_diagnostics": {
                "perturbed_pixel_count": (
                    fsoi.perturbation_diagnostics.perturbed_pixel_count
                ),
                "perturbed_fraction": (
                    fsoi.perturbation_diagnostics.perturbed_fraction
                ),
                "perturbed_area_km2": (
                    fsoi.perturbation_diagnostics.perturbed_area_km2
                ),
                "whitened_l2": fsoi.perturbation_diagnostics.whitened_l2,
                "maximum_per_tile_whitened_norm": (
                    fsoi.perturbation_diagnostics
                    .maximum_per_tile_whitened_norm
                ),
                "observation_weight_l2": (
                    fsoi.perturbation_diagnostics.observation_weight_l2
                ),
                "minimum_input_floor_margin_dbz": (
                    fsoi.perturbation_diagnostics
                    .minimum_input_floor_margin_dbz
                ),
                "minimum_input_ceiling_margin_dbz": (
                    fsoi.perturbation_diagnostics
                    .minimum_input_ceiling_margin_dbz
                ),
                "directional_classification_valid": (
                    fsoi.perturbation_diagnostics
                    .directional_classification_valid
                ),
                "baseline_dynamics_branch_status": (
                    fsoi.perturbation_diagnostics
                    .baseline_dynamics_branch_status
                ),
                "baseline_dynamics_branch_signature_digest": (
                    fsoi.perturbation_diagnostics
                    .baseline_dynamics_branch_signature_digest
                ),
            },
            "baseline_dynamics_branch_status": (
                fsoi.baseline_dynamics_branch_status
            ),
            "observation": {
                "detected_dbz": _variational_impact_digest_values(
                    fsoi.observation.detected_dbz
                ),
                "censor_threshold_dbz": _variational_impact_digest_values(
                    fsoi.observation.censor_threshold_dbz
                ),
                "observation_weight": _variational_impact_digest_values(
                    fsoi.observation.observation_weight
                ),
                "initial_background_dbz": (
                    _variational_impact_digest_values(
                        fsoi.observation.initial_background_dbz
                    )
                ),
                "baseline_dynamics_dbz": (
                    _variational_impact_digest_values(
                        fsoi.observation.baseline_dynamics_dbz
                    )
                ),
                "total": _variational_impact_digest_values(
                    fsoi.observation.total
                ),
                "baseline_branch_trusted_total": (
                    None
                    if fsoi.observation.baseline_branch_trusted_total is None
                    else _variational_impact_digest_values(
                        fsoi.observation.baseline_branch_trusted_total
                    )
                ),
            },
        }
    )


def validate_variational_fso(fso: VariationalFSO) -> None:
    """Reject any mutation of a content-addressed FSO result."""

    if fso.contract not in _FSO_VERIFICATION_CONTRACTS:
        raise ValueError("unsupported P1 FSO contract")
    if (
        fso.sensitivity_scope
        != "residual_plus_input_dependent_initial_state_and_baseline_with_frozen_selection"
        or fso.baseline_dynamics_frozen is not False
        or fso.baseline_pair_selection_frozen is not True
    ):
        raise ValueError("unsupported P1 FSO sensitivity scope")
    trusted_branch = fso.baseline_dynamics_branch_status in (
        "not_applicable",
        "certified",
    )
    trusted_input = (
        fso.observation.baseline_branch_trusted_frozen_structure_input_dbz
    )
    if fso.baseline_dynamics_branch_status not in (
        "not_applicable",
        "unknown",
        "certified",
        "invalid",
    ) or (
        (trusted_input is not None) != trusted_branch
    ):
        raise ValueError("invalid P1 FSO baseline branch trust contract")
    _validate_verification_lineage_fields(
        fso_contract=fso.contract,
        contract=fso.verification_contract,
        content_digest=fso.verification_bundle_digest,
        lineage_complete=fso.verification_lineage_complete,
        valid_times=fso.verification_valid_times,
        grid_contract_digest=fso.verification_grid_contract_digest,
        radar_product_digest=fso.verification_radar_product_digest,
        qc_pipeline_digest=fso.verification_qc_pipeline_digest,
    )
    if variational_fso_digest(fso) != fso.variational_fso_digest:
        raise ValueError("P1 FSO result digest mismatch")


def validate_variational_fsoi(fsoi: VariationalFSOI) -> None:
    """Reject any mutation or cross-binding in a P1 impact result."""

    expected_fso_contract = _FSOI_FSO_CONTRACTS.get(fsoi.contract)
    if expected_fso_contract is None:
        raise ValueError("unsupported P1 FSOI contract")
    validate_variational_fso(fsoi.fso)
    if fsoi.fso.contract != expected_fso_contract:
        raise ValueError("P1 FSOI generation disagrees with its FSO")
    if fsoi.perturbation.digest != fsoi.perturbation_digest:
        raise ValueError("P1 FSOI perturbation digest mismatch")
    if (
        fsoi.baseline_dynamics_branch_status
        != fsoi.perturbation_diagnostics.baseline_dynamics_branch_status
    ):
        raise ValueError("P1 FSOI baseline branch status mismatch")
    expected_trusted = fsoi.baseline_dynamics_branch_status in (
        "not_applicable",
        "certified",
    )
    if (
        fsoi.observation.baseline_branch_trusted_total is not None
    ) != expected_trusted:
        raise ValueError("invalid P1 FSOI trusted-total contract")
    if variational_fsoi_digest(fsoi) != fsoi.variational_fsoi_digest:
        raise ValueError("P1 FSOI result digest mismatch")


def validate_variational_learning_impact(
    learning: VariationalLearningImpact,
    *,
    expected_trust_store_digest: str | None = None,
) -> None:
    """Reject mutation or cross-binding in a final learning decision."""

    if learning.contract != "p1-variational-learning-impact-v2":
        raise ValueError("unsupported variational learning impact")
    if expected_trust_store_digest is not None:
        _require_sha256(
            "expected_trust_store_digest",
            expected_trust_store_digest,
        )
    validation = learning.first_order_validation
    if learning.fsoi is not None:
        validate_variational_fsoi(learning.fsoi)
    if learning.eligibility.eligible and (
        learning.fsoi is None
        or learning.fsoi.contract != CURRENT_VARIATIONAL_FSOI_CONTRACT
    ):
        raise ValueError("eligible learning requires current verification FSOI")
    if learning.fsoi is not None and validation is not None and (
        validation.source_fsoi_digest
        != learning.fsoi.variational_fsoi_digest
        or validation.nominal_forecast_digest
        != learning.fsoi.fso.forecast_run_digest
    ):
        raise ValueError("first-order validation lineage mismatch")
    if validation is not None and (
        first_order_validation_digest(validation)
        != validation.validation_digest
    ):
        raise ValueError("first-order validation digest mismatch")
    if not learning.eligibility.eligible:
        if learning.approval_evidence is not None:
            raise ValueError("rejected learning impact carries approval")
    else:
        fsoi = learning.fsoi
        impact = learning.frozen_domain_learning_impact
        evidence = learning.approval_evidence
        if fsoi is None or validation is None or impact is None or evidence is None:
            raise ValueError("eligible learning impact is incomplete")
        if not validation.first_order_valid:
            raise ValueError("eligible learning impact failed validation")
        expected = {
            "policy_digest": learning.eligibility.policy_digest,
            "fsoi_digest": fsoi.variational_fsoi_digest,
            "full_step_analysis_digest": validation.full_step_analysis_digest,
            "half_step_analysis_digest": validation.half_step_analysis_digest,
            "full_step_forecast_digest": validation.full_step_forecast_digest,
            "half_step_forecast_digest": validation.half_step_forecast_digest,
            "first_order_validation_digest": validation.validation_digest,
            "learning_impact_digest": _variational_impact_digest(impact),
            "approved_action_digest": fsoi.perturbation_digest,
            "nominal_input_bundle_digest": (
                validation.nominal_input_bundle_digest
            ),
        }
        if any(
            value is None or getattr(evidence, name) != value
            for name, value in expected.items()
        ):
            raise ValueError("learning approval evidence mismatch")
        if (
            expected_trust_store_digest is not None
            and evidence.trust_store_digest != expected_trust_store_digest
        ):
            raise ValueError("learning trust-store digest mismatch")
    if (
        variational_learning_impact_digest(learning)
        != learning.learning_result_digest
    ):
        raise ValueError("variational learning result digest mismatch")


def _new_variational_channel_accumulator(
    reference: Tensor,
    *,
    selected_count: int,
    lead_count: int,
    metric_count: int,
    height: int,
    width: int,
    tile_rows: int,
    tile_columns: int,
) -> _VariationalChannelAccumulator:
    return _VariationalChannelAccumulator(
        maps=reference.new_full(
            (selected_count, metric_count, 3, height, width),
            float("nan"),
        ),
        by_time=reference.new_full(
            (lead_count, metric_count, 3),
            float("nan"),
        ),
        tile_by_time=reference.new_full(
            (
                lead_count,
                metric_count,
                3,
                tile_rows,
                tile_columns,
            ),
            float("nan"),
        ),
    )


def _record_variational_channel(
    accumulator: _VariationalChannelAccumulator,
    values: Tensor,
    *,
    lead_index: int,
    metric_index: int,
    selected_index: int | None,
    tile_size: TileShape,
    signed_sum: bool,
) -> None:
    if signed_sum:
        accumulator.by_time[lead_index, metric_index] = values.reshape(
            3,
            -1,
        ).sum(dim=1)
        tile_function = _tile_sum
    else:
        accumulator.by_time[lead_index, metric_index] = (
            torch.linalg.vector_norm(values.reshape(3, -1), dim=1)
        )
        tile_function = _tile_l2
    accumulator.tile_by_time[lead_index, metric_index] = torch.stack(
        tuple(tile_function(values[index], tile_size) for index in range(3))
    )
    if selected_index is not None:
        accumulator.maps[selected_index, metric_index] = values


def _sensitivity_channel(
    accumulator: _VariationalChannelAccumulator,
) -> VariationalSensitivityChannel:
    return VariationalSensitivityChannel(
        maps=accumulator.maps,
        norm_by_time=accumulator.by_time,
        tile_norm_by_time=accumulator.tile_by_time,
    )


def _impact_channel(
    accumulator: _VariationalChannelAccumulator,
) -> VariationalImpactChannel:
    return VariationalImpactChannel(
        maps=accumulator.maps,
        sum_by_time=accumulator.by_time,
        tile_sum_by_time=accumulator.tile_by_time,
    )


@dataclass(frozen=True)
class SensitivitySnapshot:
    forecast_run_digest: str
    nowcast_config_digest: str
    sensitivity_config_digest: str
    grid_time_contract_digest: str | None
    verification_contract: str
    verification_bundle_digest: str
    verification_lineage_complete: bool
    verification_valid_times: tuple[str, ...] | None
    verification_grid_contract_digest: str | None
    verification_radar_product_digest: str | None
    verification_qc_pipeline_digest: str | None
    metric_names: tuple[str, ...]
    lead_minutes: tuple[int, ...]
    full_map_lead_minutes: tuple[int, ...]
    tile_size: int
    tile_shape_yx: TileShape
    context_feature_names: tuple[str, ...]
    context_features: Tensor
    analysis_control: Tensor
    forecast_scores: Tensor
    metric_available: Tensor
    control_sensitivity: Tensor
    forecast_sensitivity: Tensor
    forecast_cap_active_mask: Tensor
    forecast_confidence: Tensor
    path_evidence_by_metric: Tensor
    observation_source_fraction_by_metric: Tensor
    observation_verified_evidence_by_metric: Tensor
    background_verified_evidence_by_metric: Tensor
    direct: DirectSensitivity
    latest_sensitivity_mask: Tensor
    observation_std_dbz: Tensor | None
    observation_innovation_dbz: Tensor | None
    observation_innovation_mask: Tensor | None
    baseline_scores: Tensor | None
    reward_epsilon: float
    trust_components: dict[str, float]
    trust_score: float

    @property
    def impact_available(self) -> bool:
        return self.direct.impact is not None

    @property
    def reward_available(self) -> bool:
        return self.direct.reward is not None

    @property
    def whitened_tile_norm_available(self) -> bool:
        return self.direct.whitened_tile_norm is not None

    @property
    def observation_evidence_by_metric(self) -> Tensor:
        """Return the legacy observation-source fraction diagnostic."""

        return self.observation_source_fraction_by_metric


def compute_sensitivity_snapshot(
    latest_frame_dbz: Tensor,
    result: ForecastResult,
    verification_frames_dbz: VerificationInput,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    latest_background_dbz: Tensor | None = None,
    observation_std_dbz: float | Tensor | None = None,
    baseline_scores: Tensor | None = None,
) -> SensitivitySnapshot:
    """Compute M0 forecast/control/direct-observation sensitivities.

    The direct observation sensitivity is with respect to latest-frame dBZ
    inside a frozen active set. The FFT motion analysis is intentionally
    excluded: its discrete peak selection has no valid local derivative.
    Baseline-normalized reward remains disabled until baseline run, metric,
    verification, and valid-domain lineage can be verified together.
    """

    sensitivity_config = sensitivity_config or SensitivityConfig()
    if baseline_scores is not None:
        raise ValueError(
            "baseline_scores require a verified lineage contract; "
            "normalized reward is disabled until that contract exists"
        )
    nowcast_config = result.run.config
    result.validate_issuance()
    verification_bundle = _resolve_verification(
        verification_frames_dbz,
        result,
        sensitivity_config,
    )
    verification_frames = verification_bundle.frames_dbz
    result.run.validate_latest_frame(latest_frame_dbz)
    result.run.validate_latest_background(latest_background_dbz)
    latest_observation_mask = result.run.latest_observation_mask
    state = result.state
    metadata = result.metadata
    if metadata.data_status is DataStatus.UNAVAILABLE:
        raise ValueError("sensitivity is undefined for an unissued forecast")
    if metadata.dynamics_source is DynamicsSource.P1_VARIATIONAL:
        raise ValueError("M0 direct sensitivity requires a P0 state")
    if (
        2 * sensitivity_config.active_margin_dbz
        >= nowcast_config.max_dbz - nowcast_config.min_dbz
    ):
        raise ValueError("active_margin_dbz leaves no differentiable range")
    _validate_inputs(
        latest_frame_dbz,
        verification_frames,
        state,
        nowcast_config,
        latest_background_dbz,
    )

    height, width = state.echo_linear.shape
    grid_time_contract = result.run.grid_time_contract
    tile_shape_yx = _metric_tile_shape(
        sensitivity_config,
        grid_time_contract,
    )
    lead_minutes = tuple(
        range(
            nowcast_config.interval_minutes,
            nowcast_config.horizon_minutes + 1,
            nowcast_config.interval_minutes,
        )
    )
    full_map_indices = _full_map_indices(
        sensitivity_config.full_map_lead_minutes,
        lead_minutes,
    )
    metric_count = len(sensitivity_config.metric_names)
    lead_count = len(lead_minutes)
    tile_rows = math.ceil(height / tile_shape_yx[0])
    tile_columns = math.ceil(width / tile_shape_yx[1])

    clean_verification = torch.nan_to_num(
        verification_frames,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    )
    verification_finite = verification_bundle.valid_mask
    issued_valid = torch.isfinite(result.forecast_dbz)
    if issued_valid.shape != verification_finite.shape:
        raise ValueError("issued forecast must match verification shape")
    if result.valid_mask.shape != verification_finite.shape:
        raise ValueError("forecast valid_mask must match verification shape")
    if not torch.equal(result.valid_mask, issued_valid):
        raise ValueError("forecast valid_mask must match issued finite values")
    verification_valid = verification_finite & issued_valid
    truth_linear = dbz_to_echo(
        clean_verification,
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    issued_echo = dbz_to_echo(
        torch.nan_to_num(
            result.forecast_dbz,
            nan=nowcast_config.min_dbz,
            posinf=nowcast_config.max_dbz,
            neginf=nowcast_config.min_dbz,
        ),
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    control = torch.cat(
        (state.displacement_yx, state.log_growth_per_step.reshape(1))
    )
    echo = state.echo_linear
    clean_latest, latest_active = _frozen_observation(
        latest_frame_dbz,
        latest_observation_mask,
        nowcast_config,
        sensitivity_config,
    )
    if not bool(torch.any(latest_active)):
        raise ValueError(
            "M0 direct sensitivity requires a valid latest observation"
        )
    observation_std, whitening_available = _observation_std(
        observation_std_dbz,
        latest_frame_dbz,
        sensitivity_config.epsilon,
    )

    score_shape = (lead_count, metric_count)
    forecast_scores = echo.new_full(score_shape, float("nan"))
    metric_available = torch.zeros(
        score_shape,
        dtype=torch.bool,
        device=echo.device,
    )
    control_sensitivity = echo.new_full(
        (lead_count, metric_count, 3),
        float("nan"),
    )
    direct_norm = echo.new_zeros((lead_count, metric_count))
    tile_direct_norm = echo.new_zeros(
        (lead_count, metric_count, tile_rows, tile_columns)
    )
    tile_shape = (lead_count, metric_count, tile_rows, tile_columns)
    if whitening_available:
        tile_whitened_norm = echo.new_zeros(tile_shape)
    else:
        tile_whitened_norm = None
    selected_count = len(full_map_indices)
    forecast_maps = echo.new_full(
        (selected_count, metric_count, height, width),
        float("nan"),
    )
    direct_maps = echo.new_zeros(
        (selected_count, metric_count, height, width)
    )
    selected_cap_masks = torch.zeros(
        (selected_count, height, width),
        dtype=torch.bool,
        device=echo.device,
    )
    all_cap_masks = torch.zeros(
        (lead_count, height, width),
        dtype=torch.bool,
        device=echo.device,
    )
    forecast_confidence = result.forecast_confidence
    forecast_source_support = result.forecast_source_support
    forecast_observation_support = result.forecast_observation_source_support
    forecast_verified_support = result.forecast_verified_support
    confidence_decay = torch.where(
        forecast_verified_support > 0,
        forecast_confidence
        / forecast_verified_support.clamp_min(
            torch.finfo(forecast_verified_support.dtype).tiny
        ),
        0.0,
    )
    observation_verified_confidence = (
        result.forecast_observation_verified_support * confidence_decay
    )
    background_verified_confidence = (
        result.forecast_background_verified_support * confidence_decay
    )
    if not torch.allclose(
        observation_verified_confidence + background_verified_confidence,
        forecast_confidence,
        rtol=1.0e-5,
        atol=nowcast_config.contract_absolute_tolerance,
    ):
        raise ValueError("forecast evidence channels do not close")
    path_evidence_by_metric = echo.new_full(score_shape, float("nan"))
    observation_source_fraction_by_metric = echo.new_full(
        score_shape,
        float("nan"),
    )
    observation_verified_evidence_by_metric = echo.new_full(
        score_shape,
        float("nan"),
    )
    background_verified_evidence_by_metric = echo.new_full(
        score_shape,
        float("nan"),
    )
    innovation, innovation_mask = _dbz_innovation(
        latest_frame_dbz,
        latest_background_dbz,
        latest_observation_mask,
        nowcast_config,
    )
    impact_input_available = (
        innovation is not None
        and innovation_mask is not None
        and bool(torch.any(innovation_mask & latest_active))
    )
    if not impact_input_available:
        innovation = None
        innovation_mask = None
        tile_impact = None
        observation_impact = None
    else:
        tile_impact = echo.new_zeros(
            (lead_count, metric_count, tile_rows, tile_columns)
        )
        observation_impact = echo.new_zeros((lead_count, metric_count))
    selected_position = {
        index: position for position, index in enumerate(full_map_indices)
    }

    for lead_index in range(lead_count):
        truth = truth_linear[lead_index]
        valid = _metric_domain_weight(
            result,
            verification_finite[lead_index],
            lead_index,
            sensitivity_config.metric_domain,
        )
        lead_cell = freeze_remap_cell(
            (lead_index + 1) * state.displacement_yx
        )
        latent_prediction = _forecast_linear_at_step_core(
            state,
            lead_index + 1,
            nowcast_config,
            lead_cell,
        )
        prediction, cap_active = _freeze_output_cap(
            latent_prediction,
            nowcast_config,
        )
        nominal_valid = issued_valid[lead_index]
        if not torch.allclose(
            prediction[nominal_valid],
            issued_echo[lead_index][nominal_valid],
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError(
                "sensitivity model disagrees with the issued forecast"
            )
        all_cap_masks[lead_index] = cap_active
        if lead_index in selected_position:
            selected_cap_masks[selected_position[lead_index]] = cap_active

        for metric_index, metric_name in enumerate(
            sensitivity_config.metric_names
        ):
            if not _metric_has_support(
                metric_name,
                prediction,
                truth,
                valid,
                nowcast_config,
                sensitivity_config,
            ):
                direct_norm[lead_index, metric_index] = float("nan")
                tile_direct_norm[lead_index, metric_index] = float("nan")
                if tile_whitened_norm is not None:
                    tile_whitened_norm[lead_index, metric_index] = float("nan")
                if lead_index in selected_position:
                    position = selected_position[lead_index]
                    direct_maps[position, metric_index] = float("nan")
                if observation_impact is not None and tile_impact is not None:
                    observation_impact[lead_index, metric_index] = float("nan")
                    tile_impact[lead_index, metric_index] = float("nan")
                continue

            metric_available[lead_index, metric_index] = True
            metric = lambda forecast: forecast_metric(
                metric_name,
                forecast,
                truth,
                valid,
                nowcast_config,
                sensitivity_config,
                grid_time_contract,
            )
            score = metric(prediction)

            def score_from_state(
                candidate_control: Tensor,
                candidate_latest_dbz: Tensor,
            ) -> Tensor:
                candidate_echo = _active_dbz_to_echo(
                    candidate_latest_dbz,
                    clean_latest,
                    echo,
                    latest_active,
                    nowcast_config,
                )
                candidate_state = _state_from_control(
                    state,
                    candidate_control,
                    candidate_echo,
                )
                candidate = _forecast_linear_at_step_core(
                    candidate_state,
                    lead_index + 1,
                    nowcast_config,
                    lead_cell,
                )
                return metric(_apply_output_cap(candidate, cap_active, nowcast_config))

            control_gradient, direct_gradient = torch.func.grad(
                score_from_state,
                argnums=(0, 1),
            )(control, clean_latest)
            forecast_gradient = torch.func.grad(metric)(prediction)
            whitened_gradient = direct_gradient * observation_std
            evidence_weight = torch.abs(forecast_gradient.detach())
            evidence = _metric_evidence_ratios(
                evidence_weight,
                forecast_source_support[lead_index],
                forecast_confidence[lead_index],
                forecast_observation_support[lead_index],
                observation_verified_confidence[lead_index],
                background_verified_confidence[lead_index],
                sensitivity_config.epsilon,
            )
            if evidence is not None:
                (
                    path_evidence_by_metric[lead_index, metric_index],
                    observation_source_fraction_by_metric[
                        lead_index, metric_index
                    ],
                    observation_verified_evidence_by_metric[
                        lead_index, metric_index
                    ],
                    background_verified_evidence_by_metric[
                        lead_index, metric_index
                    ],
                ) = evidence

            forecast_scores[lead_index, metric_index] = score.detach()
            control_sensitivity[lead_index, metric_index] = (
                control_gradient.detach()
            )
            direct_norm[lead_index, metric_index] = torch.linalg.vector_norm(
                direct_gradient.detach()
            )
            tile_direct_norm[lead_index, metric_index] = _tile_l2(
                direct_gradient.detach(),
                tile_shape_yx,
            )
            if tile_whitened_norm is not None:
                tile_whitened_norm[lead_index, metric_index] = _tile_l2(
                    whitened_gradient.detach(),
                    tile_shape_yx,
                )

            if lead_index in selected_position:
                position = selected_position[lead_index]
                forecast_maps[position, metric_index] = forecast_gradient.detach()
                direct_maps[position, metric_index] = direct_gradient.detach()

            if observation_impact is not None and tile_impact is not None:
                if innovation is None or innovation_mask is None:
                    raise RuntimeError(
                        "impact storage requires an observation innovation"
                    )
                contribution = torch.where(
                    innovation_mask,
                    direct_gradient.detach() * innovation,
                    torch.zeros_like(direct_gradient),
                )
                tiles = _tile_sum(
                    contribution,
                    tile_shape_yx,
                )
                tile_impact[lead_index, metric_index] = tiles
                observation_impact[lead_index, metric_index] = tiles.sum()

    has_metric_support = bool(torch.any(metric_available))
    if not has_metric_support:
        observation_impact = None
        tile_impact = None

    trust_components = _trust_components(
        state,
        metadata,
        control,
        echo,
        truth_linear,
        verification_valid,
        control_sensitivity,
        metric_available,
        all_cap_masks,
        observation_verified_evidence_by_metric,
        nowcast_config,
        sensitivity_config,
        grid_time_contract,
    )
    trust_score = math.prod(trust_components.values())

    return SensitivitySnapshot(
        forecast_run_digest=result.forecast_run_digest,
        nowcast_config_digest=nowcast_config.digest,
        sensitivity_config_digest=sensitivity_config.digest,
        grid_time_contract_digest=result.run.grid_time_contract_digest,
        verification_contract=verification_bundle.contract,
        verification_bundle_digest=verification_bundle.content_digest,
        verification_lineage_complete=(
            verification_bundle.lineage_complete
        ),
        verification_valid_times=verification_bundle.valid_times,
        verification_grid_contract_digest=(
            verification_bundle.grid_contract_digest
        ),
        verification_radar_product_digest=(
            verification_bundle.radar_product_digest
        ),
        verification_qc_pipeline_digest=(
            verification_bundle.qc_pipeline_digest
        ),
        metric_names=sensitivity_config.metric_names,
        lead_minutes=lead_minutes,
        full_map_lead_minutes=sensitivity_config.full_map_lead_minutes,
        tile_size=max(tile_shape_yx),
        tile_shape_yx=tile_shape_yx,
        context_feature_names=CONTEXT_FEATURE_NAMES,
        context_features=extract_context_features(
            latest_frame_dbz,
            state,
            metadata,
            nowcast_config,
            latest_observation_mask=latest_observation_mask,
            grid_time_contract=result.run.grid_time_contract,
        ),
        analysis_control=control.detach(),
        forecast_scores=forecast_scores,
        metric_available=metric_available,
        control_sensitivity=control_sensitivity,
        forecast_sensitivity=forecast_maps,
        forecast_cap_active_mask=selected_cap_masks,
        forecast_confidence=forecast_confidence.detach(),
        path_evidence_by_metric=path_evidence_by_metric.detach(),
        observation_source_fraction_by_metric=(
            observation_source_fraction_by_metric.detach()
        ),
        observation_verified_evidence_by_metric=(
            observation_verified_evidence_by_metric.detach()
        ),
        background_verified_evidence_by_metric=(
            background_verified_evidence_by_metric.detach()
        ),
        direct=DirectSensitivity(
            maps=direct_maps,
            norm=direct_norm,
            tile_norm=tile_direct_norm,
            whitened_tile_norm=tile_whitened_norm,
            impact=observation_impact,
            tile_impact=tile_impact,
            reward=None,
        ),
        latest_sensitivity_mask=latest_active,
        observation_std_dbz=(
            observation_std.detach()
            if whitening_available
            else None
        ),
        observation_innovation_dbz=(
            innovation
            if innovation is not None
            else None
        ),
        observation_innovation_mask=(
            innovation_mask
            if innovation_mask is not None
            else None
        ),
        baseline_scores=None,
        reward_epsilon=sensitivity_config.epsilon,
        trust_components=trust_components,
        trust_score=trust_score,
    )


def compute_sensitivity_snapshot_from_run(
    result: ForecastResult,
    verification_frames_dbz: VerificationInput,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    observation_std_dbz: float | Tensor | None = None,
    baseline_scores: Tensor | None = None,
) -> SensitivitySnapshot:
    """Compute delayed M0 using the exact inputs embedded in ``result``."""

    return compute_sensitivity_snapshot(
        result.run.latest_frame_dbz,
        result,
        verification_frames_dbz,
        sensitivity_config=sensitivity_config,
        latest_background_dbz=result.run.latest_background_dbz,
        observation_std_dbz=observation_std_dbz,
        baseline_scores=baseline_scores,
    )


def compute_variational_fso(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    adjoint_config: VariationalAdjointConfig | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalFSO:
    """Compute frozen-final P1 forecast sensitivity to observations."""

    resolved_adjoint = adjoint_config or VariationalAdjointConfig()
    operations_per_apply = _analysis_whitener_operations_per_apply(analysis)
    with _count_observation_whitener_applies(
        operations_per_apply=operations_per_apply,
        maximum_total_operations=(
            resolved_adjoint.maximum_whitener_total_operations
        ),
    ) as counter:
        fso, _, _ = _compute_variational_products(
            result,
            analysis,
            verification_frames_dbz,
            sensitivity_config=sensitivity_config,
            adjoint_config=resolved_adjoint,
            observation_perturbation=None,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
    return _bind_fso_whitener_telemetry(fso, analysis, counter[0])


def compute_variational_fsoi(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    observation_perturbation: VariationalObservationPerturbation,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    adjoint_config: VariationalAdjointConfig | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalFSOI:
    """Compute signed first-order impact for an explicit perturbation."""

    resolved_adjoint = adjoint_config or VariationalAdjointConfig()
    operations_per_apply = _analysis_whitener_operations_per_apply(analysis)
    with _count_observation_whitener_applies(
        operations_per_apply=operations_per_apply,
        maximum_total_operations=(
            resolved_adjoint.maximum_whitener_total_operations
        ),
    ) as counter:
        fso, observation_impact, perturbation_diagnostics = (
            _compute_variational_products(
                result,
                analysis,
                verification_frames_dbz,
                sensitivity_config=sensitivity_config,
                adjoint_config=resolved_adjoint,
                observation_perturbation=observation_perturbation,
                neural_prior_runner=neural_prior_runner,
                neural_prior_application=neural_prior_application,
            )
        )
    fso = _bind_fso_whitener_telemetry(fso, analysis, counter[0])
    if observation_impact is None:
        raise RuntimeError("variational FSOI impact was not materialized")
    if perturbation_diagnostics is None:
        raise RuntimeError("variational perturbation was not validated")
    fsoi = VariationalFSOI(
        contract=(
            CURRENT_VARIATIONAL_FSOI_CONTRACT
            if fso.contract == CURRENT_VARIATIONAL_FSO_CONTRACT
            else EXPLORATORY_VARIATIONAL_FSOI_CONTRACT
        ),
        fso=fso,
        perturbation=observation_perturbation,
        perturbation_contract=observation_perturbation.contract,
        perturbation_digest=observation_perturbation.digest,
        perturbation_diagnostics=perturbation_diagnostics,
        baseline_dynamics_branch_status=(
            perturbation_diagnostics.baseline_dynamics_branch_status
        ),
        observation=observation_impact,
        variational_fsoi_digest="",
    )
    return replace(
        fsoi,
        variational_fsoi_digest=variational_fsoi_digest(fsoi),
    )


def compute_variational_observation_removal_impact(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    removal_mask: Tensor,
    *,
    sensitivity_config: SensitivityConfig | None = None,
    removal_config: ObservationRemovalConfig | None = None,
) -> ObservationRemovalImpact:
    """Rebuild P0/P1 and forecast after removing accepted observations.

    This is a nonlinear denial experiment, not a first-order FSOI. The
    analysis background, active support, common-bias whitener, robust optimum,
    posterior, confidence, and issuance domain are all recomputed.
    """

    result.validate_issuance()
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("observation removal requires a P1 linearization")
    validate_analysis_linearization_content(analysis.control, linearization)
    request = ObservationRemovalRequest(
        removal_mask=removal_mask,
        linearization_digest=linearization.linearization_digest,
    )
    observations = linearization.observations
    frozen = linearization.frozen
    mask = request.removal_mask.to(observations.valid_mask.device)
    if mask.shape != observations.valid_mask.shape:
        raise ValueError("removal mask must match the observation grid")
    if bool(torch.any(mask & ~observations.valid_mask)):
        raise ValueError("only accepted observations can be removed")
    removal = removal_config or ObservationRemovalConfig()
    removed_count = int(torch.count_nonzero(mask))
    valid_count = int(torch.count_nonzero(observations.valid_mask))
    removed_fraction = removed_count / max(1, valid_count)
    if removed_count > removal.maximum_removed_observation_count:
        raise ValueError("observation removal exceeds its count budget")
    if removed_fraction > removal.maximum_removed_fraction:
        raise ValueError("observation removal exceeds its fraction budget")
    grid = frozen.grid_time_contract
    union_count = int(torch.count_nonzero(torch.any(mask, dim=0)))
    removed_area_km2 = (
        None if grid is None else union_count * grid.cell_area_m2 / 1.0e6
    )
    if removal.maximum_removed_area_km2 is not None:
        if grid is None or removed_area_km2 is None:
            raise ValueError("physical removal budget requires a grid contract")
        try:
            grid.validate_projected_area_maximum(
                removed_area_km2,
                removal.maximum_removed_area_km2,
            )
        except ValueError as error:
            raise ValueError(
                "observation removal exceeds or is uncertain against its area "
                "budget"
            ) from error

    original_qc = ~observations.qc_rejected_mask
    changed_qc = original_qc & ~mask
    manifest = _operational_manifest_from_run(result)
    identity = _operational_identity_from_run(result)
    operations_per_apply = _observation_whitener_operations_per_apply(
        observations
    )
    with _count_observation_whitener_applies(
        operations_per_apply=operations_per_apply,
        maximum_total_operations=removal.maximum_whitener_total_operations,
    ) as counter:
        removed_forecast, removed_analysis = variational_nowcast(
            frozen.input_frames_dbz,
            nowcast_config=frozen.nowcast_config,
            analysis_config=frozen.analysis_config,
            observation_std_dbz=observations.std_dbz,
            quality_weight=observations.quality_weight,
            qc_mask=changed_qc,
            observation_common_bias_group_index=(
                observations.common_bias_group_index
            ),
            observation_common_bias_mode_weights=(
                observations.common_bias_mode_weights
            ),
            background_frames_dbz=frozen.background_frames_dbz,
            background_age_minutes=frozen.background_age_minutes,
            grid_time_contract=grid,
            operational_calibration_manifest=manifest,
            operational_calibration_approval_digest=(
                result.run.operational_calibration_approval_digest
            ),
            operational_data_identity=identity,
        )
    removed_linearization = removed_analysis.linearization
    if (
        not removed_analysis.converged
        or removed_analysis.used_fallback
        or removed_analysis.degraded
        or not removed_analysis.p1_forecast_eligible
        or removed_linearization is None
    ):
        raise RuntimeError(
            "observation-removal analysis did not produce an eligible P1"
        )
    removed_forecast.validate_issuance()
    config = sensitivity_config or SensitivityConfig()
    leads = config.full_map_lead_minutes
    if not leads:
        raise ValueError("observation removal requires at least one lead")
    interval = result.run.config.interval_minutes
    if any(
        minutes % interval != 0
        or minutes > result.run.config.horizon_minutes
        for minutes in leads
    ):
        raise ValueError("observation-removal leads must be issued forecast leads")
    verification = _resolve_verification(
        verification_frames_dbz,
        result,
        config,
    )
    _ = _resolve_verification(
        verification_frames_dbz,
        removed_forecast,
        config,
    )
    nominal_scores, nominal_available = _resolved_forecast_scores(
        result,
        analysis.state,
        verification,
        leads,
        config,
    )
    removed_scores, removed_available = _resolved_forecast_scores(
        removed_forecast,
        removed_analysis.state,
        verification,
        leads,
        config,
    )
    available = nominal_available & removed_available
    metric_change = torch.where(
        available,
        removed_scores - nominal_scores,
        torch.full_like(nominal_scores, float("nan")),
    )
    return ObservationRemovalImpact(
        request=request,
        nominal_scores=nominal_scores,
        removed_scores=removed_scores,
        metric_change=metric_change,
        metric_available=available,
        lead_minutes=leads,
        metric_names=config.metric_names,
        metric_domain=config.metric_domain,
        nominal_forecast_digest=_forecast_result_content_digest(result),
        removed_forecast_digest=_forecast_result_content_digest(
            removed_forecast
        ),
        removed_linearization_digest=(
            removed_linearization.linearization_digest
        ),
        verification_bundle_digest=verification.content_digest,
        sensitivity_config_digest=config.digest,
        removal_config_digest=removal.digest,
        removed_observation_count=removed_count,
        removed_fraction=removed_fraction,
        removed_area_km2=removed_area_km2,
        whitener_operations_per_apply=operations_per_apply,
        observed_whitener_apply_count=counter[0],
        observed_whitener_total_operations=operations_per_apply * counter[0],
    )


def validate_observation_removal_impact(
    impact: ObservationRemovalImpact,
) -> None:
    """Validate a resolved denial result before durable use."""

    expected_request = json_digest(
        {
            "contract": impact.request.contract,
            "removal_mask": tensor_digest(impact.request.removal_mask),
            "linearization_digest": impact.request.linearization_digest,
        }
    )
    if impact.request.request_digest != expected_request:
        raise ValueError("observation-removal request digest mismatch")
    if (
        impact.observation_removal_impact_digest
        != _observation_removal_impact_digest(impact)
    ):
        raise ValueError("observation-removal impact digest mismatch")


def _forecast_result_content_digest(result: ForecastResult) -> str:
    return json_digest(
        {
            "contract": "forecast-result-content-v1",
            "forecast_run_digest": result.forecast_run_digest,
            "forecast_dbz_digest": result.forecast_dbz_digest,
            "valid_mask_digest": result.valid_mask_digest,
            "state_metadata_digest": result.state_metadata_digest,
        }
    )


def _operational_manifest_from_run(
    result: ForecastResult,
) -> OperationalCalibrationManifest | None:
    value = result.run.operational_calibration_manifest_json
    return None if value is None else OperationalCalibrationManifest.from_json(value)


def _operational_identity_from_run(
    result: ForecastResult,
) -> OperationalDataIdentity | None:
    value = result.run.operational_data_identity_json
    return None if value is None else OperationalDataIdentity.from_json(value)


def _resolved_forecast_scores(
    result: ForecastResult,
    state: RadarState,
    verification: _ResolvedVerification,
    leads: tuple[int, ...],
    config: SensitivityConfig,
    *,
    domain_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    nowcast = result.run.config
    clean_truth = torch.nan_to_num(
        verification.frames_dbz,
        nan=nowcast.min_dbz,
        posinf=nowcast.max_dbz,
        neginf=nowcast.min_dbz,
    ).clamp(nowcast.min_dbz, nowcast.max_dbz)
    truth_linear = dbz_to_echo(
        clean_truth,
        min_dbz=nowcast.min_dbz,
        max_dbz=nowcast.max_dbz,
    )
    finite_truth = verification.valid_mask & torch.isfinite(
        verification.frames_dbz
    )
    scores = state.echo_linear.new_full(
        (len(leads), len(config.metric_names)),
        float("nan"),
    )
    available = torch.zeros_like(scores, dtype=torch.bool)
    for lead_index, minutes in enumerate(leads):
        step = minutes // nowcast.interval_minutes
        forecast_index = step - 1
        forecast, _ = _freeze_output_cap(
            forecast_linear_at_step(state, step, nowcast),
            nowcast,
        )
        weight = (
            _metric_domain_weight(
                result,
                finite_truth[forecast_index],
                forecast_index,
                config.metric_domain,
            )
            if domain_weights is None
            else domain_weights[lead_index]
        )
        if domain_weights is None:
            weight = weight * verification.metric_weight[forecast_index].to(weight)
        for metric_index, name in enumerate(config.metric_names):
            if not _metric_has_support(
                name,
                forecast,
                truth_linear[forecast_index],
                weight,
                nowcast,
                config,
            ):
                continue
            scores[lead_index, metric_index] = forecast_metric(
                name,
                forecast,
                truth_linear[forecast_index],
                weight,
                nowcast,
                config,
                result.run.grid_time_contract,
            )
            available[lead_index, metric_index] = True
    return scores.detach(), available.detach()


def _resolved_forecast_domain_weights(
    result: ForecastResult,
    verification: _ResolvedVerification,
    leads: tuple[int, ...],
    config: SensitivityConfig,
) -> Tensor:
    """Return one frozen metric-domain weight per requested lead."""

    finite = verification.valid_mask & torch.isfinite(verification.frames_dbz)
    weights = []
    for minutes in leads:
        forecast_index = minutes // result.run.config.interval_minutes - 1
        weights.append(
            _metric_domain_weight(
                result,
                finite[forecast_index],
                forecast_index,
                config.metric_domain,
            )
            * verification.metric_weight[forecast_index].to(result.state.echo_linear)
        )
    return torch.stack(weights).detach()


def validate_variational_fsoi_issuance_impact(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    fsoi: VariationalFSOI,
    *,
    policy: AutomatedLearningPolicy,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> FirstOrderValidation:
    """Re-solve a physical FSOI on the changed issuance domain.

    This is a research diagnostic. Automated learning remains bound to the
    smoother ``frozen_metric_domain`` contract.
    """

    validate_variational_fsoi(fsoi)
    result.validate_issuance()
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("issuance validation requires a linearization")
    if fsoi.perturbation.perturbation_semantics != "physical_radar_value":
        raise ValueError("issuance validation requires a physical perturbation")
    if fsoi.fso.forecast_run_digest != result.forecast_run_digest:
        raise ValueError("issuance validation forecast mismatch")
    if fsoi.fso.linearization_digest != linearization.linearization_digest:
        raise ValueError("issuance validation linearization mismatch")
    if fsoi.fso.sensitivity_config_digest != policy.sensitivity_config.digest:
        raise ValueError("issuance validation sensitivity policy mismatch")
    if fsoi.fso.adjoint_config_digest != policy.adjoint_config.digest:
        raise ValueError("issuance validation adjoint policy mismatch")
    return _validate_first_order_learning_impact(
        result,
        analysis,
        verification_frames_dbz,
        fsoi,
        policy,
        metric_domain_contract="resolved_issuance_domain",
        maximum_whitener_total_operations=(
            policy.maximum_whitener_total_operations
        ),
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )


def _bind_fso_whitener_telemetry(
    fso: VariationalFSO,
    analysis: AnalysisResult | P1LinearizationState,
    apply_count: int,
) -> VariationalFSO:
    linearization = analysis.linearization
    if linearization is None:
        raise RuntimeError("P1 FSO lacks a linearization")
    updated = replace(
        fso,
        whitener_operations_per_apply=(
            _observation_whitener_operations_per_apply(
                linearization.observations
            )
        ),
        observed_whitener_apply_count=apply_count,
        variational_fso_digest="",
    )
    return replace(
        updated,
        variational_fso_digest=variational_fso_digest(updated),
    )


def _analysis_whitener_operations_per_apply(
    analysis: AnalysisResult | P1LinearizationState,
) -> int:
    linearization = analysis.linearization
    if linearization is None:
        return 0
    return _observation_whitener_operations_per_apply(
        linearization.observations
    )


def score_candidate_perturbations(
    fso: VariationalFSO,
    analysis: AnalysisResult | P1LinearizationState,
    candidates: Iterable[
        tuple[
            str,
            SparseRadarPerturbation | VariationalObservationPerturbation,
        ]
    ],
    *,
    policy: AutomatedLearningPolicy,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalCandidateRanking:
    """Stream, precheck, and rank physical-radar candidates with one FSO."""

    validate_variational_fso(fso)
    if fso.contract != CURRENT_VARIATIONAL_FSO_CONTRACT:
        raise ValueError(
            "automated candidate ranking requires current verification FSO"
        )
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("candidate ranking requires a linearization")
    if fso.linearization_digest != linearization.linearization_digest:
        raise ValueError("candidate FSO linearization mismatch")
    if fso.sensitivity_config_digest != policy.sensitivity_config.digest:
        raise ValueError("candidate FSO sensitivity policy mismatch")
    if fso.adjoint_config_digest != policy.ranking_adjoint_config.digest:
        raise ValueError("candidate FSO adjoint policy mismatch")
    if fso.lead_minutes != fso.full_map_lead_minutes:
        raise ValueError("candidate ranking requires a full map for every lead")
    sensitivity = fso.observation.frozen_structure_input_dbz
    maps = sensitivity.maps
    if maps.shape[:2] != fso.forecast_scores.shape:
        raise ValueError("candidate FSO map coverage is incomplete")
    scales = maps.new_tensor(
        tuple(
            policy.threshold_for(name).effective_ranking_scale
            for name in fso.metric_names
        )
    )
    metric_weights = maps.new_tensor(
        tuple(
            policy.threshold_for(name).ranking_weight
            for name in fso.metric_names
        )
    )
    lead_weights = maps.new_tensor(policy.resolved_ranking_lead_weights)
    weighted_scale = lead_weights[:, None] * metric_weights[None, :]
    flat_maps = maps.reshape(*maps.shape[:2], -1)
    top: list[tuple[str, SparseRadarPerturbation, Tensor, float]] = []
    prechecks: list[VariationalCandidatePrecheck] = []
    identifiers: set[str] = set()
    candidate_count = 0
    scoring_operations = 0
    whitener_apply_count = fso.observed_whitener_apply_count
    if (
        fso.whitener_operations_per_apply * whitener_apply_count
        > policy.maximum_whitener_total_operations
    ):
        raise ValueError("common-bias total operation budget exhausted")
    started = time.monotonic()
    for candidate_id, candidate in candidates:
        candidate_count += 1
        if time.monotonic() - started > (
            policy.maximum_candidate_ranking_wall_seconds
        ):
            raise ValueError("candidate ranking wall-time budget exhausted")
        if candidate_count > policy.maximum_candidate_count:
            raise ValueError("learning candidate count exceeds its policy budget")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("learning candidate identifiers must be nonempty")
        if candidate_id in identifiers:
            raise ValueError("learning candidate identifiers must be unique")
        identifiers.add(candidate_id)
        sparse = _sparse_candidate(candidate)
        reason = _candidate_precheck_reason(
            sparse,
            linearization,
            policy,
        )
        if reason is not None:
            prechecks.append(
                VariationalCandidatePrecheck(
                    candidate_id=candidate_id,
                    perturbation_digest=sparse.digest,
                    admissible=False,
                    rejection_reason=reason,
                )
            )
            continue
        operations = sparse.nonzero_count * maps.shape[0] * maps.shape[1]
        scoring_operations += operations
        if scoring_operations > policy.maximum_candidate_scoring_operations:
            raise ValueError("candidate scoring operation budget exhausted")
        indices = sparse.flat_indices.to(maps.device)
        values = sparse.delta_values.to(dtype=maps.dtype, device=maps.device)
        prediction = (
            flat_maps.index_select(-1, indices) * values[None, None, :]
        ).sum(dim=-1)
        prediction = torch.where(
            fso.metric_available,
            prediction,
            torch.full_like(prediction, float("nan")),
        )
        score = _candidate_ranking_score(
            prediction,
            fso.metric_available,
            scales,
            weighted_scale,
            policy.ranking_objective,
        )
        if score == 0.0:
            prechecks.append(
                VariationalCandidatePrecheck(
                    candidate_id=candidate_id,
                    perturbation_digest=sparse.digest,
                    admissible=True,
                    rejection_reason=None,
                )
            )
            continue
        would_enter = len(top) < policy.maximum_learning_candidates_to_validate
        if not would_enter:
            worst = top[-1]
            would_enter = (-score, candidate_id) < (-worst[3], worst[0])
        if would_enter:
            reason, precheck_applies = _candidate_full_precheck_reason(
                sparse,
                linearization,
                policy,
                neural_prior_runner=neural_prior_runner,
                neural_prior_application=neural_prior_application,
            )
            whitener_apply_count += precheck_applies
            if (
                fso.whitener_operations_per_apply * whitener_apply_count
                > policy.maximum_whitener_total_operations
            ):
                raise ValueError("common-bias total operation budget exhausted")
        prechecks.append(
            VariationalCandidatePrecheck(
                candidate_id=candidate_id,
                perturbation_digest=sparse.digest,
                admissible=reason is None,
                rejection_reason=reason,
            )
        )
        if reason is not None or not would_enter:
            continue
        top.append((candidate_id, sparse, prediction.detach(), score))
        top.sort(key=lambda item: (-item[3], item[0]))
        del top[policy.maximum_learning_candidates_to_validate :]
    if candidate_count == 0:
        raise ValueError("at least one learning candidate is required")
    return VariationalCandidateRanking(
        fso=fso,
        prechecks=tuple(prechecks),
        policy_digest=policy.digest,
        ranking_objective=policy.ranking_objective,
        candidate_count=candidate_count,
        scoring_operations=scoring_operations,
        whitener_operations_per_apply=fso.whitener_operations_per_apply,
        observed_whitener_apply_count=whitener_apply_count,
        scores=tuple(
            VariationalCandidateScore(
                candidate_id=candidate_id,
                perturbation=perturbation,
                predicted_metric_change=prediction,
                score=score,
                rank=rank,
            )
            for rank, (candidate_id, perturbation, prediction, score) in (
                enumerate(top, start=1)
            )
        ),
    )


def _sparse_candidate(
    candidate: SparseRadarPerturbation | VariationalObservationPerturbation,
) -> SparseRadarPerturbation:
    if isinstance(candidate, SparseRadarPerturbation):
        return candidate
    delta = candidate.physical_radar_dbz_delta
    if candidate.perturbation_semantics != "physical_radar_value" or delta is None:
        raise ValueError("candidate must be a physical radar perturbation")
    return SparseRadarPerturbation.from_dense(delta)


def _candidate_precheck_reason(
    candidate: SparseRadarPerturbation,
    linearization: AnalysisLinearization,
    policy: AutomatedLearningPolicy,
) -> str | None:
    if candidate.retained_bytes > policy.maximum_candidate_bytes:
        return "candidate_byte_budget_exceeded"
    if candidate.nonzero_count > policy.maximum_candidate_nonzeros:
        return "candidate_nonzero_budget_exceeded"
    observations = linearization.observations
    frozen = linearization.frozen
    if candidate.shape != tuple(observations.dbz.shape):
        return "candidate_perturbation_shape_mismatch"
    indices = candidate.flat_indices.to(observations.dbz.device)
    values = candidate.delta_values.to(
        dtype=observations.dbz.dtype,
        device=observations.dbz.device,
    )
    detected = observations.detected_mask.reshape(-1).index_select(0, indices)
    if not bool(torch.all(detected)):
        return "physical_radar_delta_outside_detected_observations"
    config = policy.adjoint_config
    if bool(torch.any(torch.abs(values) > config.maximum_detected_delta_dbz)):
        return "detected_dbz_exceeds_local_perturbation_limit"
    nominal = observations.dbz.reshape(-1).index_select(0, indices)
    changed = nominal + values
    if bool(torch.any(changed < frozen.nowcast_config.min_dbz)) or bool(
        torch.any(changed > frozen.nowcast_config.max_dbz)
    ):
        return "physical_radar_perturbation_crosses_input_clamp"
    if bool(
        torch.any(
            changed
            < frozen.analysis_config.detection_limit_dbz
            + config.minimum_detection_margin_dbz
        )
    ):
        return "observation_perturbation_crosses_classification_branch"
    count = candidate.nonzero_count
    valid_count = max(1, int(torch.count_nonzero(observations.valid_mask)))
    if count > config.maximum_perturbed_pixel_count:
        return "observation_perturbation_exceeds_pixel_budget"
    if count / valid_count > config.maximum_perturbed_fraction:
        return "observation_perturbation_exceeds_area_fraction"
    grid = frozen.grid_time_contract
    if config.maximum_perturbed_area_km2 is not None:
        if grid is None:
            return "physical_perturbation_area_requires_grid_contract"
        area_km2 = count * grid.cell_area_m2 / 1.0e6
        area_status = grid.projected_area_maximum_status(
            area_km2,
            config.maximum_perturbed_area_km2,
        )
        if area_status == "exceeds":
            return "observation_perturbation_exceeds_physical_area_budget"
        if area_status == "uncertain":
            return (
                "observation_perturbation_area_budget_is_geodetically_uncertain"
            )
    if observations.common_bias_mode_weights is None:
        quality = observations.quality_weight.reshape(-1).index_select(0, indices)
        std = observations.std_dbz.reshape(-1).index_select(0, indices)
        energy = quality * (values / std).square()
        if math.sqrt(float(torch.sum(energy).detach())) > (
            config.maximum_whitened_perturbation_l2
        ):
            return "observation_perturbation_exceeds_whitened_trust_radius"
        if _sparse_maximum_tile_norm(
            indices,
            energy,
            observations.dbz.shape,
            _perturbation_tile_size(config, grid),
        ) > config.maximum_per_tile_whitened_norm:
            return "observation_perturbation_exceeds_tile_trust_radius"
    return None


def _candidate_full_precheck_reason(
    candidate: SparseRadarPerturbation,
    linearization: AnalysisLinearization,
    policy: AutomatedLearningPolicy,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None,
    neural_prior_application: NeuralPriorApplication | None,
) -> tuple[str | None, int]:
    counter = [0]
    try:
        dense = candidate.materialize(linearization.observations.dbz)
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            dense,
            linearization,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        with _count_observation_whitener_applies() as counter:
            _validate_variational_observation_perturbation(
                perturbation,
                linearization.observations,
                linearization.frozen,
                policy.adjoint_config,
                neural_prior_runner=neural_prior_runner,
                neural_prior_application=neural_prior_application,
            )
    except (TypeError, ValueError) as error:
        return str(error), counter[0]
    return None, counter[0]


def _sparse_maximum_tile_norm(
    flat_indices: Tensor,
    energy: Tensor,
    shape: torch.Size,
    tile_shape: TileShape,
) -> float:
    _, height, width = shape
    tile_height, tile_width = tile_shape
    tile_columns = math.ceil(width / tile_width)
    spatial = flat_indices % (height * width)
    frame = torch.div(flat_indices, height * width, rounding_mode="floor")
    row = torch.div(spatial, width, rounding_mode="floor")
    column = spatial % width
    tile = (
        frame * math.ceil(height / tile_height) * tile_columns
        + torch.div(row, tile_height, rounding_mode="floor") * tile_columns
        + torch.div(column, tile_width, rounding_mode="floor")
    )
    totals = energy.new_zeros(int(torch.amax(tile).detach()) + 1)
    totals.scatter_add_(0, tile, energy)
    return math.sqrt(float(torch.amax(totals).detach()))


def _candidate_ranking_score(
    prediction: Tensor,
    available: Tensor,
    scales: Tensor,
    weights: Tensor,
    objective: CandidateRankingObjective,
) -> float:
    normalized = prediction / scales[None, :]
    normalized = torch.where(available, normalized, 0.0)
    if objective == "expected_error_reduction":
        normalized = torch.clamp(-normalized, min=0.0)
    elif objective == "two_sided_diagnostic":
        benefit = torch.clamp(-normalized, min=0.0)
        harm = torch.clamp(normalized, min=0.0)
        benefit_norm = torch.sum(weights * benefit.square())
        harm_norm = torch.sum(weights * harm.square())
        return math.sqrt(float(torch.maximum(benefit_norm, harm_norm).detach()))
    elif objective != "absolute_influence":
        raise ValueError("unsupported candidate ranking objective")
    value = torch.where(available, weights * normalized.square(), 0.0).sum()
    return math.sqrt(float(value.detach()))


def validate_top_k_learning_impacts(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    ranking: VariationalCandidateRanking,
    *,
    policy: AutomatedLearningPolicy,
    policy_trust_store_path: str | Path,
    maximum_candidates_to_validate: int | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> tuple[RankedLearningOutcome, ...]:
    """Run full/half robust re-solves only for the highest-ranked candidates."""

    validate_variational_fso(ranking.fso)
    if ranking.fso.contract != CURRENT_VARIATIONAL_FSO_CONTRACT:
        raise ValueError(
            "automated learning requires current verification FSO"
        )
    if ranking.ranking_digest != _variational_candidate_ranking_digest(ranking):
        raise ValueError("variational candidate ranking digest mismatch")
    if ranking.policy_digest != policy.digest:
        raise ValueError("candidate ranking policy mismatch")
    limit = (
        policy.maximum_learning_candidates_to_validate
        if maximum_candidates_to_validate is None
        else maximum_candidates_to_validate
    )
    if type(limit) is not int or limit <= 0:
        raise ValueError("maximum_candidates_to_validate must be positive")
    if limit > policy.maximum_learning_candidates_to_validate:
        raise ValueError("learning candidate count exceeds its policy budget")
    resolve_limit = policy.maximum_total_robust_resolves
    if 2 * limit > resolve_limit:
        raise ValueError("full/half robust resolves exceed their policy budget")
    trust_store = _load_learning_policy_trust_store(policy_trust_store_path)
    rejection = _learning_context_rejection(
        result,
        analysis,
        ranking.fso,
        verification_frames_dbz,
        policy,
        trust_store,
    )
    if rejection is not None:
        return tuple(
            RankedLearningOutcome(
                candidate_id=scored.candidate_id,
                candidate_rank=scored.rank,
                candidate_score=scored.score,
                ranking_digest=ranking.ranking_digest,
                result=_rejected_learning_impact(policy, rejection),
            )
            for scored in ranking.scores[:limit]
        )
    linearization = analysis.linearization
    if linearization is None:
        raise RuntimeError("approved learning context lacks a linearization")
    outcomes: list[RankedLearningOutcome] = []
    started = time.monotonic()
    robust_resolves = 0
    total_pcg_iterations = 0
    for scored in ranking.scores[:limit]:
        dense = scored.perturbation.materialize(linearization.observations.dbz)
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            dense,
            linearization,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        with _count_observation_whitener_applies() as fsoi_counter:
            fsoi = _variational_fsoi_from_precomputed_fso(
                ranking.fso,
                linearization,
                perturbation,
                policy.adjoint_config,
                neural_prior_runner=neural_prior_runner,
                neural_prior_application=neural_prior_application,
            )
        robust_resolves += 2
        result_for_candidate = _learning_impact_from_fsoi(
            result,
            analysis,
            verification_frames_dbz,
            fsoi,
            policy,
            trust_store.content_digest,
            selection=_LearningSelection(
                mode="ranked_top_k",
                candidate_id=scored.candidate_id,
                candidate_rank=scored.rank,
                candidate_score=scored.score,
                candidate_perturbation_digest=scored.perturbation.digest,
                ranking_digest=ranking.ranking_digest,
                ranking_policy_digest=ranking.policy_digest,
                ranking_objective=ranking.ranking_objective,
                observed_whitener_apply_count=(
                    ranking.observed_whitener_apply_count + fsoi_counter[0]
                ),
            ),
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        validation = result_for_candidate.first_order_validation
        if validation is not None:
            total_pcg_iterations += validation.total_resolved_pcg_iterations
        if total_pcg_iterations > policy.maximum_learning_pcg_iterations:
            raise ValueError("learning PCG iteration budget exhausted")
        if time.monotonic() - started > policy.maximum_learning_wall_seconds:
            raise ValueError("learning wall-time budget exhausted")
        outcomes.append(
            RankedLearningOutcome(
                candidate_id=scored.candidate_id,
                candidate_rank=scored.rank,
                candidate_score=scored.score,
                ranking_digest=ranking.ranking_digest,
                result=result_for_candidate,
            )
        )
    if robust_resolves > resolve_limit:
        raise RuntimeError("learning robust-resolve accounting failed")
    return tuple(outcomes)


def compute_variational_fsoi_for_learning(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    observation_perturbation: VariationalObservationPerturbation,
    *,
    policy: AutomatedLearningPolicy,
    policy_trust_store_path: str | Path,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalLearningImpact:
    """Compute frozen-domain FSOI under a root-owned learning policy."""

    trust_store = _load_learning_policy_trust_store(
        policy_trust_store_path
    )
    if policy.digest not in trust_store.approved_policy_digests:
        return _rejected_learning_impact(policy, "unapproved_learning_policy")
    if (
        observation_perturbation.perturbation_semantics
        != "physical_radar_value"
    ):
        return _rejected_learning_impact(
            policy,
            "physical_radar_perturbation_required",
        )
    linearization = analysis.linearization
    if linearization is None:
        return _rejected_learning_impact(policy, "linearization_required")
    if linearization.algorithm_bundle_digest != policy.algorithm_bundle_digest:
        return _rejected_learning_impact(
            policy,
            "algorithm_bundle_not_approved",
        )
    if linearization.numerical_runtime_digest != policy.numerical_runtime_digest:
        return _rejected_learning_impact(
            policy,
            "numerical_runtime_not_approved",
        )
    try:
        fsoi = compute_variational_fsoi(
            result,
            analysis,
            verification_frames_dbz,
            observation_perturbation,
            sensitivity_config=policy.sensitivity_config,
            adjoint_config=policy.adjoint_config,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
    except ValueError as error:
        return _rejected_learning_impact(policy, str(error))
    return _learning_impact_from_fsoi(
        result,
        analysis,
        verification_frames_dbz,
        fsoi,
        policy,
        trust_store.content_digest,
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )


@dataclass(frozen=True)
class _LearningSelection:
    mode: LearningSelectionMode
    candidate_id: str | None = None
    candidate_rank: int | None = None
    candidate_score: float | None = None
    candidate_perturbation_digest: str | None = None
    ranking_digest: str | None = None
    ranking_policy_digest: str | None = None
    ranking_objective: CandidateRankingObjective | None = None
    observed_whitener_apply_count: int = 0


def _learning_impact_from_fsoi(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    fsoi: VariationalFSOI,
    policy: AutomatedLearningPolicy,
    trust_store_digest: str,
    *,
    selection: _LearningSelection | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalLearningImpact:
    selection = selection or _LearningSelection(mode="direct")
    impact = fsoi.observation.baseline_branch_trusted_total
    if impact is None:
        return _rejected_learning_impact(
            policy,
            "baseline_dynamics_branch_not_certified",
            fsoi=fsoi,
        )
    base_whitener_applies = (
        selection.observed_whitener_apply_count
        if selection.mode == "ranked_top_k"
        else fsoi.fso.observed_whitener_apply_count
    )
    base_whitener_operations = (
        fsoi.fso.whitener_operations_per_apply * base_whitener_applies
    )
    remaining_whitener_operations = (
        policy.maximum_whitener_total_operations - base_whitener_operations
    )
    if remaining_whitener_operations <= 0:
        return _rejected_learning_impact(
            policy,
            "common_bias_total_operation_budget_exhausted",
            fsoi=fsoi,
        )
    validation_started = time.monotonic()
    try:
        validation = _validate_first_order_learning_impact(
            result,
            analysis,
            verification_frames_dbz,
            fsoi,
            policy,
            maximum_whitener_total_operations=(
                remaining_whitener_operations
            ),
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
    except ValueError as error:
        if str(error) != (
            "common-bias whitener total operation budget exhausted"
        ):
            raise
        return _rejected_learning_impact(
            policy,
            "common_bias_total_operation_budget_exhausted",
            fsoi=fsoi,
        )
    if (
        validation.source_fsoi_digest != fsoi.variational_fsoi_digest
        or validation.nominal_forecast_digest != result.forecast_run_digest
        or validation.nominal_input_bundle_digest
        != result.run.input_bundle_digest
        or validation.nominal_full_analysis_input_digest
        != result.run.full_analysis_input_digest
    ):
        raise ValueError("first-order validation lineage mismatch")
    if (
        validation.total_resolved_pcg_iterations
        > policy.maximum_learning_pcg_iterations
    ):
        return _rejected_learning_impact(
            policy,
            "learning_pcg_iteration_budget_exhausted",
            fsoi=fsoi,
            first_order_validation=validation,
        )
    if (
        time.monotonic() - validation_started
        > policy.maximum_learning_wall_seconds
    ):
        return _rejected_learning_impact(
            policy,
            "learning_wall_time_budget_exhausted",
            fsoi=fsoi,
            first_order_validation=validation,
        )
    total_whitener_applies = (
        base_whitener_applies + validation.observed_whitener_apply_count
    )
    total_whitener_operations = (
        fsoi.fso.whitener_operations_per_apply * total_whitener_applies
    )
    if total_whitener_operations > policy.maximum_whitener_total_operations:
        return _rejected_learning_impact(
            policy,
            "common_bias_total_operation_budget_exhausted",
            fsoi=fsoi,
            first_order_validation=validation,
        )
    if not validation.first_order_valid:
        no_material_signal = validation.material_metric_count == 0 and all(
            (
                validation.full_step_resolved_analysis_converged,
                validation.half_step_resolved_analysis_converged,
                validation.active_branch_valid,
                validation.full_step_valid,
                validation.half_step_valid,
                validation.sign_consistent_for_material_impacts,
            )
        )
        reason = (
            "no_material_learning_signal"
            if no_material_signal
            else "first_order_validation_failed"
        )
        return _rejected_learning_impact(
            policy,
            reason,
            fsoi=fsoi,
            first_order_validation=validation,
        )
    owned_impact = _clone_variational_impact_channel(impact)
    analysis_digests = (
        validation.full_step_analysis_digest,
        validation.half_step_analysis_digest,
    )
    forecast_digests = (
        validation.full_step_forecast_digest,
        validation.half_step_forecast_digest,
    )
    if any(value is None for value in (*analysis_digests, *forecast_digests)):
        raise RuntimeError("eligible learning validation lacks resolved digests")
    evidence = LearningApprovalEvidence(
        policy_digest=policy.digest,
        trust_store_digest=trust_store_digest,
        fsoi_digest=fsoi.variational_fsoi_digest,
        full_step_analysis_digest=cast(str, analysis_digests[0]),
        half_step_analysis_digest=cast(str, analysis_digests[1]),
        full_step_forecast_digest=cast(str, forecast_digests[0]),
        half_step_forecast_digest=cast(str, forecast_digests[1]),
        first_order_validation_digest=validation.validation_digest,
        learning_impact_digest=_variational_impact_digest(owned_impact),
        approved_action_digest=fsoi.perturbation_digest,
        nominal_input_bundle_digest=validation.nominal_input_bundle_digest,
        nominal_full_analysis_input_digest=(
            validation.nominal_full_analysis_input_digest
        ),
        selection_mode=selection.mode,
        candidate_id=selection.candidate_id,
        candidate_rank=selection.candidate_rank,
        candidate_score=selection.candidate_score,
        candidate_perturbation_digest=(
            selection.candidate_perturbation_digest
        ),
        ranking_digest=selection.ranking_digest,
        ranking_policy_digest=selection.ranking_policy_digest,
        ranking_objective=selection.ranking_objective,
        whitener_operations_per_apply=(
            fsoi.fso.whitener_operations_per_apply
        ),
        observed_whitener_apply_count=total_whitener_applies,
        observed_whitener_total_operations=total_whitener_operations,
    )
    return VariationalLearningImpact(
        eligibility=LearningEligibility(
            eligible=True,
            reasons=(),
            policy_digest=policy.digest,
        ),
        fsoi=fsoi,
        first_order_validation=validation,
        frozen_domain_learning_impact=owned_impact,
        approval_evidence=evidence,
    )


def _learning_context_rejection(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    fso: VariationalFSO,
    verification_input: VerificationInput,
    policy: AutomatedLearningPolicy,
    trust_store: _LearningPolicyTrustStore,
) -> str | None:
    if policy.digest not in trust_store.approved_policy_digests:
        return "unapproved_learning_policy"
    if fso.contract != CURRENT_VARIATIONAL_FSO_CONTRACT:
        return "current_verification_FSO_required"
    linearization = analysis.linearization
    if linearization is None:
        return "linearization_required"
    if fso.forecast_run_digest != result.forecast_run_digest:
        return "candidate_FSO_forecast_mismatch"
    if fso.linearization_digest != linearization.linearization_digest:
        return "candidate_FSO_linearization_mismatch"
    if fso.sensitivity_config_digest != policy.sensitivity_config.digest:
        return "candidate_FSO_sensitivity_policy_mismatch"
    if fso.adjoint_config_digest != policy.ranking_adjoint_config.digest:
        return "candidate_FSO_adjoint_policy_mismatch"
    if linearization.algorithm_bundle_digest != policy.algorithm_bundle_digest:
        return "algorithm_bundle_not_approved"
    if linearization.numerical_runtime_digest != policy.numerical_runtime_digest:
        return "numerical_runtime_not_approved"
    try:
        _validate_variational_fso_lineage(result, analysis, linearization)
        verification = _resolve_verification(
            verification_input,
            result,
            policy.sensitivity_config,
        )
    except ValueError as error:
        return str(error)
    if verification.content_digest != fso.verification_bundle_digest:
        return "candidate_FSO_verification_mismatch"
    return None


def _variational_fsoi_from_precomputed_fso(
    fso: VariationalFSO,
    linearization: AnalysisLinearization,
    perturbation: VariationalObservationPerturbation,
    adjoint_config: VariationalAdjointConfig,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalFSOI:
    """Materialize one impact from retained maps without solving another adjoint."""

    if fso.linearization_digest != linearization.linearization_digest:
        raise ValueError("precomputed FSO linearization mismatch")
    ranking_config = replace(
        adjoint_config,
        require_baseline_dynamics_branch_validity=False,
    )
    if fso.adjoint_config_digest != ranking_config.digest:
        raise ValueError("precomputed FSO adjoint config mismatch")
    if fso.lead_minutes != fso.full_map_lead_minutes:
        raise ValueError("precomputed FSO does not retain every lead map")
    diagnostics = _validate_variational_observation_perturbation(
        perturbation,
        linearization.observations,
        linearization.frozen,
        adjoint_config,
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )
    if (
        adjoint_config.require_baseline_dynamics_branch_validity
        and diagnostics.baseline_dynamics_branch_status
        not in ("not_applicable", "certified")
    ):
        raise ValueError("P1 FSOI baseline dynamics branch is not certified")
    observations = linearization.observations
    component_pairs = (
        (fso.observation.detected_dbz, perturbation.detected_dbz),
        (
            fso.observation.censor_threshold_dbz,
            perturbation.censor_threshold_dbz,
        ),
        (fso.observation.observation_weight, perturbation.observation_weight),
        (
            fso.observation.initial_background_dbz,
            _initial_background_perturbation(perturbation, observations),
        ),
        (
            fso.observation.baseline_dynamics_dbz,
            _baseline_dynamics_perturbation(perturbation, observations),
        ),
    )
    components = tuple(
        _impact_from_precomputed_sensitivity(channel, delta, fso.tile_shape_yx)
        for channel, delta in component_pairs
    )
    total = _sum_impact_channels(components)
    trusted = diagnostics.baseline_dynamics_branch_status in (
        "not_applicable",
        "certified",
    )
    observation = VariationalObservationImpact(
        detected_dbz=components[0],
        censor_threshold_dbz=components[1],
        observation_weight=components[2],
        initial_background_dbz=components[3],
        baseline_dynamics_dbz=components[4],
        total=total,
        baseline_branch_trusted_total=total if trusted else None,
    )
    fsoi = VariationalFSOI(
        contract=(
            CURRENT_VARIATIONAL_FSOI_CONTRACT
            if fso.contract == CURRENT_VARIATIONAL_FSO_CONTRACT
            else EXPLORATORY_VARIATIONAL_FSOI_CONTRACT
        ),
        fso=fso,
        perturbation=perturbation,
        perturbation_contract=perturbation.contract,
        perturbation_digest=perturbation.digest,
        perturbation_diagnostics=diagnostics,
        baseline_dynamics_branch_status=(
            diagnostics.baseline_dynamics_branch_status
        ),
        observation=observation,
        variational_fsoi_digest="",
    )
    return replace(
        fsoi,
        variational_fsoi_digest=variational_fsoi_digest(fsoi),
    )


def _impact_from_precomputed_sensitivity(
    sensitivity: VariationalSensitivityChannel,
    delta: Tensor,
    tile_size: TileShape,
) -> VariationalImpactChannel:
    maps = sensitivity.maps * delta
    if maps.ndim != 5 or maps.shape[2] != 3:
        raise ValueError("precomputed sensitivity maps have an invalid shape")
    lead_count, metric_count, _, height, width = maps.shape
    accumulator = _new_variational_channel_accumulator(
        maps,
        selected_count=lead_count,
        lead_count=lead_count,
        metric_count=metric_count,
        height=height,
        width=width,
        tile_rows=math.ceil(height / tile_size[0]),
        tile_columns=math.ceil(width / tile_size[1]),
    )
    for lead in range(lead_count):
        for metric in range(metric_count):
            _record_variational_channel(
                accumulator,
                maps[lead, metric],
                lead_index=lead,
                metric_index=metric,
                selected_index=lead,
                tile_size=tile_size,
                signed_sum=True,
            )
    return _impact_channel(accumulator)


def _sum_impact_channels(
    channels: tuple[VariationalImpactChannel, ...],
) -> VariationalImpactChannel:
    if not channels:
        raise ValueError("at least one impact channel is required")
    return VariationalImpactChannel(
        maps=sum(
            (channel.maps for channel in channels),
            torch.zeros_like(channels[0].maps),
        ),
        sum_by_time=sum(
            (channel.sum_by_time for channel in channels),
            torch.zeros_like(channels[0].sum_by_time),
        ),
        tile_sum_by_time=sum(
            (channel.tile_sum_by_time for channel in channels),
            torch.zeros_like(channels[0].tile_sum_by_time),
        ),
    )


def _rejected_learning_impact(
    policy: AutomatedLearningPolicy,
    reason: str,
    *,
    fsoi: VariationalFSOI | None = None,
    first_order_validation: FirstOrderValidation | None = None,
) -> VariationalLearningImpact:
    return VariationalLearningImpact(
        eligibility=LearningEligibility(
            eligible=False,
            reasons=(reason,),
            policy_digest=policy.digest,
        ),
        fsoi=fsoi,
        first_order_validation=first_order_validation,
        frozen_domain_learning_impact=None,
    )


def _validate_first_order_learning_impact(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_input: VerificationInput,
    fsoi: VariationalFSOI,
    policy: AutomatedLearningPolicy,
    *,
    metric_domain_contract: FirstOrderMetricDomain = "frozen_metric_domain",
    maximum_whitener_total_operations: int | None = None,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> FirstOrderValidation:
    """Re-solve full and half perturbations on one explicit metric domain."""

    full_prediction = (
        fsoi.observation.total.sum_by_time.sum(dim=-1).detach()
    )
    half_prediction = 0.5 * full_prediction
    operations_per_apply = _analysis_whitener_operations_per_apply(analysis)
    with _count_observation_whitener_applies(
        operations_per_apply=operations_per_apply,
        maximum_total_operations=maximum_whitener_total_operations,
    ) as whitener_counter:
        full = _resolve_learning_step(
            result,
            analysis,
            verification_input,
            fsoi,
            policy,
            scale=1.0,
            metric_domain_contract=metric_domain_contract,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        half = _resolve_learning_step(
            result,
            analysis,
            verification_input,
            fsoi,
            policy,
            scale=0.5,
            metric_domain_contract=metric_domain_contract,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
    available = fsoi.fso.metric_available
    full_error = torch.abs(full.metric_change - full_prediction)
    half_error = torch.abs(half.metric_change - half_prediction)
    full_step_valid = _taylor_step_is_valid(
        full_prediction,
        full.metric_change,
        available,
        policy,
    )
    half_step_valid = _taylor_step_is_valid(
        half_prediction,
        half.metric_change,
        available,
        policy,
    )
    sign_consistent = _material_impact_signs_are_consistent(
        (
            (full_prediction, full.metric_change),
            (half_prediction, half.metric_change),
        ),
        available,
        fsoi.fso.metric_names,
        policy,
    )
    material_count, maximum_material, aggregate_material = (
        _material_impact_summary(
            (
                (full_prediction, full.metric_change),
                (half_prediction, half.metric_change),
            ),
            available,
            fsoi.fso.metric_names,
            policy,
        )
    )
    branch_valid = full.active_branch_valid and half.active_branch_valid
    first_order_valid = (
        full.analysis_converged
        and half.analysis_converged
        and branch_valid
        and full_step_valid
        and half_step_valid
        and sign_consistent
        and material_count > 0
    )
    if result.run.full_analysis_input_digest is None:
        raise ValueError("learning validation requires full input identity")
    return FirstOrderValidation(
        source_fsoi_digest=fsoi.variational_fsoi_digest,
        nominal_forecast_digest=result.forecast_run_digest,
        nominal_input_bundle_digest=result.run.input_bundle_digest,
        nominal_full_analysis_input_digest=(
            result.run.full_analysis_input_digest
        ),
        full_step_prediction=full_prediction,
        full_step_resolved_metric_change=full.metric_change,
        full_step_absolute_error=full_error,
        half_step_prediction=half_prediction,
        half_step_resolved_metric_change=half.metric_change,
        half_step_absolute_error=half_error,
        metric_available=available.detach().clone(),
        full_step_resolved_analysis_converged=full.analysis_converged,
        half_step_resolved_analysis_converged=half.analysis_converged,
        active_branch_valid=branch_valid,
        full_step_valid=full_step_valid,
        half_step_valid=half_step_valid,
        sign_consistent_for_material_impacts=sign_consistent,
        material_metric_count=material_count,
        maximum_material_impact=maximum_material,
        aggregate_material_impact_norm=aggregate_material,
        first_order_valid=first_order_valid,
        full_step_analysis_digest=full.analysis_digest,
        half_step_analysis_digest=half.analysis_digest,
        full_step_forecast_digest=full.forecast_digest,
        half_step_forecast_digest=half.forecast_digest,
        full_step_input_bundle_digest=full.input_bundle_digest,
        half_step_input_bundle_digest=half.input_bundle_digest,
        full_step_pcg_iterations=full.pcg_iterations,
        half_step_pcg_iterations=half.pcg_iterations,
        observed_whitener_apply_count=whitener_counter[0],
        frozen_domain_state_effect=full.frozen_domain_state_effect,
        issuance_policy_effect=full.issuance_policy_effect,
        end_to_end_issuance_effect=full.end_to_end_issuance_effect,
        coverage_before=full.coverage_before,
        coverage_after=full.coverage_after,
        newly_issued_fraction=full.newly_issued_fraction,
        withdrawn_fraction=full.withdrawn_fraction,
        background_fallback_before=full.background_fallback_before,
        background_fallback_after=full.background_fallback_after,
        metric_domain_contract=metric_domain_contract,
    )


@dataclass(frozen=True)
class _ResolvedLearningStep:
    metric_change: Tensor
    frozen_domain_state_effect: Tensor | None
    issuance_policy_effect: Tensor | None
    end_to_end_issuance_effect: Tensor | None
    coverage_before: Tensor | None
    coverage_after: Tensor | None
    newly_issued_fraction: Tensor | None
    withdrawn_fraction: Tensor | None
    background_fallback_before: Tensor | None
    background_fallback_after: Tensor | None
    analysis_converged: bool
    active_branch_valid: bool
    analysis_digest: str | None
    forecast_digest: str | None
    input_bundle_digest: str | None
    pcg_iterations: int


def _resolve_learning_step(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_input: VerificationInput,
    fsoi: VariationalFSOI,
    policy: AutomatedLearningPolicy,
    *,
    scale: float,
    metric_domain_contract: FirstOrderMetricDomain,
    neural_prior_runner: NeuralPriorInferenceRunner | None,
    neural_prior_application: NeuralPriorApplication | None,
) -> _ResolvedLearningStep:
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("first-order validation requires a linearization")
    perturbation = fsoi.perturbation
    physical_delta = perturbation.physical_radar_dbz_delta
    if physical_delta is None:
        raise ValueError("first-order validation requires a physical delta")
    delta = scale * physical_delta
    observations = linearization.observations
    frozen = linearization.frozen
    changed_frames = frozen.input_frames_dbz + delta
    changed_prior: NeuralPriorApplication | None = None
    if frozen.neural_prior_dependency is not None:
        if neural_prior_runner is None:
            raise ValueError("neural-prior re-solve requires an inference runner")
        _validate_retained_prior_runner(
            frozen,
            neural_prior_runner,
            neural_prior_application,
        )
        preliminary_run = ForecastRunContract.from_inputs(
            result.run.config,
            changed_frames,
            observations.valid_mask,
            frozen.background_frames_dbz,
            frozen.background_age_minutes,
            observation_quality_weight=(
                observations.quality_weight * observations.valid_mask
            ),
            observation_std_dbz=observations.std_dbz,
            grid_time_contract=frozen.grid_time_contract,
            analysis_config_json=result.run.analysis_config_json,
            analysis_config_digest=result.run.analysis_config_digest,
            analysis_input_digest=result.run.analysis_input_digest,
            operational_calibration_manifest_json=(
                result.run.operational_calibration_manifest_json
            ),
            operational_calibration_manifest_digest=(
                result.run.operational_calibration_manifest_digest
            ),
            operational_calibration_approval_digest=(
                result.run.operational_calibration_approval_digest
            ),
            operational_data_identity_json=(result.run.operational_data_identity_json),
            operational_data_identity_digest=(
                result.run.operational_data_identity_digest
            ),
            input_plan_json=result.run.input_plan_json,
            input_plan_digest=result.run.input_plan_digest,
        )
        changed_prior = neural_prior_runner.infer(
            changed_frames,
            input_run=preliminary_run,
            role=cast(Literal["candidate", "parent"], frozen.neural_prior_role),
        )
        retained_raw = frozen.neural_prior_raw_background_dbz
        assert retained_raw is not None
        if frozen.neural_prior_dependency == "exogenous" and not torch.equal(
            changed_prior.initial_background_dbz,
            retained_raw,
        ):
            raise ValueError("exogenous neural prior changed with the radar input")
        changed_observations, changed_frozen = prepare_analysis(
            changed_frames,
            nowcast_config=result.run.config,
            analysis_config=frozen.analysis_config,
            observation_std_dbz=observations.std_dbz,
            quality_weight=observations.quality_weight,
            qc_mask=observations.valid_mask,
            observation_common_bias_group_index=(observations.common_bias_group_index),
            observation_common_bias_mode_weights=(
                observations.common_bias_mode_weights
            ),
            background_frames_dbz=frozen.background_frames_dbz,
            background_age_minutes=frozen.background_age_minutes,
            grid_time_contract=frozen.grid_time_contract,
            neural_prior=changed_prior,
        )
    else:
        changed_observations = replace(observations, dbz=observations.dbz + delta)
        background_delta = (
            scale * _initial_background_perturbation(perturbation, observations)[0]
        )
        baseline_dynamics = torch.cat(
            (
                frozen.baseline_state.displacement_yx,
                frozen.baseline_state.log_growth_per_step.reshape(1),
            )
        )
        if frozen.baseline_metadata.tendency_source is TendencySource.OBSERVATION:
            baseline_dynamics = _baseline_dynamics_from_observation(
                changed_observations.dbz,
                frozen,
            )
        changed_frozen = replace(
            frozen,
            initial_background_dbz=(frozen.initial_background_dbz + background_delta),
            baseline_state=RadarState(
                echo_linear=frozen.baseline_state.echo_linear,
                displacement_yx=baseline_dynamics[:2],
                log_growth_per_step=baseline_dynamics[2],
            ),
            baseline_frames_dbz=frozen.baseline_frames_dbz + delta,
        )
    if not torch.equal(
        changed_frozen.active_field_index,
        frozen.active_field_index,
    ):
        metric_change = torch.full_like(fsoi.fso.forecast_scores, float("nan"))
        return _ResolvedLearningStep(
            metric_change=metric_change,
            frozen_domain_state_effect=None,
            issuance_policy_effect=None,
            end_to_end_issuance_effect=None,
            coverage_before=None,
            coverage_after=None,
            newly_issued_fraction=None,
            withdrawn_fraction=None,
            background_fallback_before=None,
            background_fallback_after=None,
            analysis_converged=False,
            active_branch_valid=False,
            analysis_digest=None,
            forecast_digest=None,
            input_bundle_digest=None,
            pcg_iterations=0,
        )
    resolved = solve_analysis(
        changed_observations,
        changed_frozen,
        control=analysis.control,
    )
    resolved_linearization = resolved.linearization
    converged = (
        resolved.converged
        and not resolved.used_fallback
        and not resolved.degraded
        and resolved.p1_forecast_eligible
        and resolved_linearization is not None
    )
    metric_change = torch.full_like(
        fsoi.fso.forecast_scores,
        float("nan"),
    )
    resolved_domain = metric_domain_contract == "resolved_issuance_domain"
    state_effect = torch.full_like(metric_change, float("nan")) if resolved_domain else None
    policy_effect = torch.full_like(metric_change, float("nan")) if resolved_domain else None
    total_effect = torch.full_like(metric_change, float("nan")) if resolved_domain else None
    coverage_before = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    coverage_after = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    newly_issued = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    withdrawn = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    fallback_before = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    fallback_after = metric_change.new_zeros(len(fsoi.fso.lead_minutes)) if resolved_domain else None
    branch_valid = converged and resolved_linearization is not None
    if resolved_linearization is not None:
        branch_valid = (
            branch_valid
            and resolved_linearization.frozen.analysis_remap_cells
            == frozen.analysis_remap_cells
        )
        if frozen.neural_prior_dependency is not None:
            resolved_prior_valid = (
                resolved_linearization.frozen.neural_prior_valid_mask
            )
            retained_prior_valid = frozen.neural_prior_valid_mask
            branch_valid = (
                branch_valid
                and resolved_prior_valid is not None
                and retained_prior_valid is not None
                and torch.equal(resolved_prior_valid, retained_prior_valid)
            )
    if not converged:
        return _ResolvedLearningStep(
            metric_change=metric_change,
            frozen_domain_state_effect=state_effect,
            issuance_policy_effect=policy_effect,
            end_to_end_issuance_effect=total_effect,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            newly_issued_fraction=newly_issued,
            withdrawn_fraction=withdrawn,
            background_fallback_before=fallback_before,
            background_fallback_after=fallback_after,
            analysis_converged=False,
            active_branch_valid=branch_valid,
            analysis_digest=None,
            forecast_digest=None,
            input_bundle_digest=None,
            pcg_iterations=resolved.pcg_iterations,
        )
    if resolved_linearization is None:
        raise RuntimeError("converged learning re-solve lacks linearization")
    resolved_forecast = None
    resolved_run = None
    if metric_domain_contract == "resolved_issuance_domain":
        resolved_analysis_config_json = result.run.analysis_config_json
        resolved_analysis_config_digest = result.run.analysis_config_digest
        resolved_analysis_input_digest = result.run.analysis_input_digest
        if changed_prior is not None:
            (
                resolved_analysis_config_json,
                resolved_analysis_config_digest,
                resolved_analysis_input_digest,
            ) = _analysis_input_lineage(
                changed_observations,
                changed_frozen.analysis_config,
                neural_prior_application_digest=(changed_prior.application_digest),
            )
        resolved_run = ForecastRunContract.from_inputs(
            result.run.config,
            changed_frames,
            changed_observations.valid_mask,
            frozen.background_frames_dbz,
            frozen.background_age_minutes,
            observation_quality_weight=(
                changed_observations.quality_weight
                * changed_observations.valid_mask
            ),
            observation_std_dbz=changed_observations.std_dbz,
            grid_time_contract=frozen.grid_time_contract,
            analysis_config_json=resolved_analysis_config_json,
            analysis_config_digest=resolved_analysis_config_digest,
            analysis_input_digest=resolved_analysis_input_digest,
            operational_calibration_manifest_json=(
                result.run.operational_calibration_manifest_json
            ),
            operational_calibration_manifest_digest=(
                result.run.operational_calibration_manifest_digest
            ),
            operational_calibration_approval_digest=(
                result.run.operational_calibration_approval_digest
            ),
            operational_data_identity_json=(
                result.run.operational_data_identity_json
            ),
            operational_data_identity_digest=(
                result.run.operational_data_identity_digest
            ),
            neural_prior_digest=(
                None if changed_prior is None else changed_prior.neural_prior_digest
            ),
            prior_application_digest=(
                None if changed_prior is None else changed_prior.application_digest
            ),
            prior_model_contract_digest=(
                None if changed_prior is None else changed_prior.model_contract_digest
            ),
            prior_feature_schema_digest=(
                None if changed_prior is None else changed_prior.feature_schema_digest
            ),
            prior_training_manifest_digest=(
                None
                if changed_prior is None
                else changed_prior.training_manifest_digest
            ),
            prior_inference_evidence_digest=(
                None
                if changed_prior is None
                else changed_prior.inference_evidence.evidence_digest
            ),
            prior_inference_algorithm_digest=(
                None
                if changed_prior is None
                else changed_prior.inference_evidence.inference_algorithm_digest
            ),
            prior_numerical_runtime_digest=(
                None
                if changed_prior is None
                else changed_prior.inference_evidence.numerical_runtime_digest
            ),
            prior_dependency=(
                None if changed_prior is None else changed_prior.dependency
            ),
            prior_role=None if changed_prior is None else changed_prior.role,
            input_plan_json=result.run.input_plan_json,
            input_plan_digest=result.run.input_plan_digest,
        )
        resolved_forecast = forecast_from_state(
            resolved.state,
            resolved.metadata,
            result.run.config,
            run=resolved_run,
        )
        resolved_forecast.validate_issuance()

    verification = _resolve_verification(
        verification_input,
        result,
        policy.sensitivity_config,
    )
    config = result.run.config
    clean_truth = torch.nan_to_num(
        verification.frames_dbz,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    truth_linear = dbz_to_echo(
        clean_truth,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    finite_truth = verification.valid_mask & torch.isfinite(
        verification.frames_dbz
    )
    available = fsoi.fso.metric_available
    changed_forecasts: list[dict[str, str | int]] = []
    for lead_index, minutes in enumerate(fsoi.fso.lead_minutes):
        step = minutes // config.interval_minutes
        forecast_index = step - 1
        if freeze_remap_cell(step * resolved.state.displacement_yx) != (
            freeze_remap_cell(step * analysis.state.displacement_yx)
        ):
            branch_valid = False
        nominal_linear = forecast_linear_at_step(analysis.state, step, config)
        changed_linear = forecast_linear_at_step(resolved.state, step, config)
        nominal_capped, nominal_cap_active = _freeze_output_cap(
            nominal_linear,
            config,
        )
        changed_capped, changed_cap_active = _freeze_output_cap(
            changed_linear,
            config,
        )
        if not torch.equal(nominal_cap_active, changed_cap_active):
            branch_valid = False
        changed_forecasts.append(
            {
                "lead_minutes": minutes,
                "forecast_digest": tensor_digest(changed_capped),
            }
        )
        nominal_weight = _metric_domain_weight(
            result,
            finite_truth[forecast_index],
            forecast_index,
            policy.sensitivity_config.metric_domain,
        )
        changed_weight = (
            nominal_weight
            if not resolved_domain
            else _metric_domain_weight(
                cast(ForecastResult, resolved_forecast),
                finite_truth[forecast_index],
                forecast_index,
                policy.sensitivity_config.metric_domain,
            )
        )
        if resolved_domain:
            assert coverage_before is not None
            assert coverage_after is not None
            assert newly_issued is not None
            assert withdrawn is not None
            assert fallback_before is not None
            assert fallback_after is not None
            valid_count = torch.count_nonzero(finite_truth[forecast_index]).clamp_min(1)
            nominal_support = nominal_weight > 0
            changed_support = changed_weight > 0
            coverage_before[lead_index] = torch.count_nonzero(nominal_support).to(metric_change) / valid_count
            coverage_after[lead_index] = torch.count_nonzero(changed_support).to(metric_change) / valid_count
            newly_issued[lead_index] = torch.count_nonzero(changed_support & ~nominal_support).to(metric_change) / valid_count
            withdrawn[lead_index] = torch.count_nonzero(nominal_support & ~changed_support).to(metric_change) / valid_count
            nominal_fallback = result.background_fallback_mask[forecast_index]
            changed_fallback = cast(
                ForecastResult, resolved_forecast
            ).background_fallback_mask[forecast_index]
            fallback_before[lead_index] = torch.count_nonzero(
                nominal_fallback & finite_truth[forecast_index]
            ).to(metric_change) / valid_count
            fallback_after[lead_index] = torch.count_nonzero(
                changed_fallback & finite_truth[forecast_index]
            ).to(metric_change) / valid_count
        for metric_index, metric_name in enumerate(fsoi.fso.metric_names):
            if not bool(available[lead_index, metric_index]):
                continue
            nominal_score = forecast_metric(
                metric_name,
                nominal_capped,
                truth_linear[forecast_index],
                nominal_weight,
                config,
                policy.sensitivity_config,
                frozen.grid_time_contract,
            )
            if not torch.allclose(
                nominal_score,
                fsoi.fso.forecast_scores[lead_index, metric_index],
                rtol=0.0,
                atol=config.contract_absolute_tolerance,
            ):
                raise ValueError(
                    "first-order validation does not reproduce the nominal metric"
                )
            changed_nominal_domain_score = forecast_metric(
                metric_name,
                changed_capped,
                truth_linear[forecast_index],
                nominal_weight,
                config,
                policy.sensitivity_config,
                frozen.grid_time_contract,
            )
            if resolved_domain:
                changed_resolved_domain_score = forecast_metric(
                    metric_name,
                    changed_capped,
                    truth_linear[forecast_index],
                    changed_weight,
                    config,
                    policy.sensitivity_config,
                    frozen.grid_time_contract,
                )
                assert state_effect is not None
                assert policy_effect is not None
                assert total_effect is not None
                state_effect[lead_index, metric_index] = (
                    changed_nominal_domain_score - nominal_score
                ).detach()
                policy_effect[lead_index, metric_index] = (
                    changed_resolved_domain_score - changed_nominal_domain_score
                ).detach()
                total_effect[lead_index, metric_index] = (
                    changed_resolved_domain_score - nominal_score
                ).detach()
                metric_change[lead_index, metric_index] = total_effect[
                    lead_index, metric_index
                ]
            else:
                metric_change[lead_index, metric_index] = (
                    changed_nominal_domain_score - nominal_score
                ).detach()
    return _ResolvedLearningStep(
        metric_change=metric_change,
        frozen_domain_state_effect=state_effect,
        issuance_policy_effect=policy_effect,
        end_to_end_issuance_effect=total_effect,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        newly_issued_fraction=newly_issued,
        withdrawn_fraction=withdrawn,
        background_fallback_before=fallback_before,
        background_fallback_after=fallback_after,
        analysis_converged=converged,
        active_branch_valid=branch_valid,
        analysis_digest=resolved_linearization.linearization_digest,
        forecast_digest=(
            cast(ForecastResult, resolved_forecast).forecast_run_digest
            if metric_domain_contract == "resolved_issuance_domain"
            else json_digest(
                {
                    "contract": "p1-resolved-learning-forecast-v1",
                    "forecasts": changed_forecasts,
                }
            )
        ),
        input_bundle_digest=(
            None if resolved_run is None else resolved_run.input_bundle_digest
        ),
        pcg_iterations=resolved.pcg_iterations,
    )


def _taylor_step_is_valid(
    prediction: Tensor,
    actual: Tensor,
    available: Tensor,
    policy: AutomatedLearningPolicy,
) -> bool:
    error = torch.abs(actual - prediction)
    scale = torch.maximum(torch.abs(actual), torch.abs(prediction))
    absolute_error = prediction.new_tensor(
        tuple(
            policy.threshold_for(name).maximum_absolute_error
            for name in policy.sensitivity_config.metric_names
        )
    )
    tolerance = (
        absolute_error
        + policy.maximum_linearity_relative_error * scale
    )
    selected = error.masked_select(available)
    selected_tolerance = tolerance.masked_select(available)
    return (
        selected.numel() > 0
        and bool(torch.all(torch.isfinite(selected)))
        and bool(torch.all(selected <= selected_tolerance))
    )


def _material_impact_signs_are_consistent(
    steps: tuple[tuple[Tensor, Tensor], ...],
    available: Tensor,
    metric_names: tuple[str, ...],
    policy: AutomatedLearningPolicy,
) -> bool:
    materiality = available.new_tensor(
        tuple(
            policy.threshold_for(name).material_impact_threshold
            for name in metric_names
        ),
        dtype=steps[0][0].dtype,
    )
    for prediction, actual in steps:
        material = available & (
            torch.maximum(torch.abs(prediction), torch.abs(actual))
            >= materiality
        )
        if bool(torch.any(material & (prediction * actual <= 0.0))):
            return False
    return True


def _material_impact_summary(
    steps: tuple[tuple[Tensor, Tensor], ...],
    available: Tensor,
    metric_names: tuple[str, ...],
    policy: AutomatedLearningPolicy,
) -> tuple[int, float, float]:
    magnitudes = torch.stack(
        tuple(
            torch.maximum(torch.abs(prediction), torch.abs(actual))
            for prediction, actual in steps
        )
    ).amax(dim=0)
    thresholds = magnitudes.new_tensor(
        tuple(
            policy.threshold_for(name).material_impact_threshold
            for name in metric_names
        )
    )
    selected = magnitudes.masked_select(
        available & (magnitudes >= thresholds)
    )
    if selected.numel() == 0:
        return 0, 0.0, 0.0
    return (
        int(selected.numel()),
        float(torch.amax(selected).detach()),
        float(torch.linalg.vector_norm(selected).detach()),
    )


def _compute_variational_products(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    verification_frames_dbz: VerificationInput,
    *,
    sensitivity_config: SensitivityConfig | None,
    adjoint_config: VariationalAdjointConfig | None,
    observation_perturbation: VariationalObservationPerturbation | None,
    neural_prior_runner: NeuralPriorInferenceRunner | None,
    neural_prior_application: NeuralPriorApplication | None,
) -> tuple[
    VariationalFSO,
    VariationalObservationImpact | None,
    VariationalPerturbationDiagnostics | None,
]:
    """Compute frozen-final P1 forecast sensitivity to observations.

    The final IRLS weights, remap cells, active controls, and observation
    classification are held fixed. Each forecast-metric adjoint uses the same
    matrix-free Gauss--Newton normal operator as the accepted analysis.
    """

    sensitivity_config = sensitivity_config or SensitivityConfig()
    adjoint_config = adjoint_config or VariationalAdjointConfig()
    if observation_perturbation is not None:
        adjoint_config = replace(
            adjoint_config,
            require_active_set_margin=True,
            require_feasibility_margin=True,
            require_gauss_newton_reliability=True,
        )
    result.validate_issuance()
    verification_bundle = _resolve_verification(
        verification_frames_dbz,
        result,
        sensitivity_config,
    )
    verification_frames = verification_bundle.frames_dbz
    if result.metadata.dynamics_source is not DynamicsSource.P1_VARIATIONAL:
        raise ValueError("variational FSO requires an accepted P1 forecast")
    if (
        analysis.used_fallback
        or not analysis.converged
        or analysis.degraded
        or not analysis.final_linearization_stationary
        or not analysis.final_robust_stationary
        or not analysis.final_irls_fixed_point
        or not analysis.p1_forecast_eligible
        or not analysis.posterior_eligible
        or not analysis.fso_eligible
    ):
        raise ValueError("variational FSO requires a converged P1 analysis")
    linearization = analysis.linearization
    if linearization is None:
        raise ValueError("P1 analysis does not retain a final linearization")
    if linearization.contract != P1_LINEARIZATION_CONTRACT:
        raise ValueError("unsupported P1 linearization contract")
    _validate_variational_fso_lineage(result, analysis, linearization)
    feasibility_margins = _variational_feasibility_margins(
        linearization.feasibility_margins,
        adjoint_config,
    )
    if (
        adjoint_config.require_feasibility_margin
        and feasibility_margins.low_interior_validity
    ):
        raise ValueError(
            "P1 FSO feasibility margin is below its requirement"
        )

    nowcast_config = result.run.config
    observations = linearization.observations
    frozen = linearization.frozen
    control = analysis.control
    validated_prior_input = None
    if frozen.neural_prior_dependency == "radar_dependent":
        if neural_prior_runner is None:
            raise ValueError("radar-dependent prior FSO requires an inference runner")
        _validate_retained_prior_runner(
            frozen,
            neural_prior_runner,
            neural_prior_application,
        )
        validated_prior_input = neural_prior_runner.validated_bound_input(
            _require_bound_neural_prior_input(frozen)
        )
    perturbation_diagnostics = None
    if observation_perturbation is not None:
        perturbation_diagnostics = _validate_variational_observation_perturbation(
            observation_perturbation,
            observations,
            frozen,
            adjoint_config,
            neural_prior_runner=neural_prior_runner,
            neural_prior_application=neural_prior_application,
        )
        if (
            adjoint_config.require_baseline_dynamics_branch_validity
            and perturbation_diagnostics.baseline_dynamics_branch_status
            not in ("not_applicable", "certified")
        ):
            raise ValueError(
                "P1 FSOI baseline dynamics branch is not certified"
            )
    _validate_inputs(
        observations.dbz[-1],
        verification_frames,
        analysis.state,
        nowcast_config,
        None,
    )
    height, width = analysis.state.echo_linear.shape
    all_lead_minutes = tuple(
        range(
            nowcast_config.interval_minutes,
            nowcast_config.horizon_minutes + 1,
            nowcast_config.interval_minutes,
        )
    )
    forecast_indices = _adjoint_lead_indices(
        adjoint_config,
        all_lead_minutes,
    )
    lead_minutes = tuple(
        all_lead_minutes[index] for index in forecast_indices
    )
    full_map_indices = _full_map_indices(
        sensitivity_config.full_map_lead_minutes,
        lead_minutes,
    )
    selected_position = {
        index: position for position, index in enumerate(full_map_indices)
    }
    lead_count = len(lead_minutes)
    metric_count = len(sensitivity_config.metric_names)
    tile_shape_yx = _metric_tile_shape(
        sensitivity_config,
        frozen.grid_time_contract,
    )
    tile_rows = math.ceil(height / tile_shape_yx[0])
    tile_columns = math.ceil(width / tile_shape_yx[1])
    selected_count = len(full_map_indices)
    materialized_output_bytes = _variational_materialized_output_bytes(
        control,
        selected_count=selected_count,
        lead_count=lead_count,
        metric_count=metric_count,
        height=height,
        width=width,
        tile_rows=tile_rows,
        tile_columns=tile_columns,
        include_impact=observation_perturbation is not None,
        gauss_newton_probe_count=adjoint_config.gauss_newton_probe_count,
    )
    if (
        materialized_output_bytes
        > adjoint_config.maximum_materialized_output_bytes
    ):
        raise ValueError(
            "P1 FSO materialized output exceeds its byte budget"
        )
    score_shape = (lead_count, metric_count)
    forecast_scores = control.new_full(score_shape, float("nan"))
    metric_available = torch.zeros(
        score_shape,
        dtype=torch.bool,
        device=control.device,
    )
    channel_shape = {
        "selected_count": selected_count,
        "lead_count": lead_count,
        "metric_count": metric_count,
        "height": height,
        "width": width,
        "tile_rows": tile_rows,
        "tile_columns": tile_columns,
    }
    detected_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    censor_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    weight_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    initial_background_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    baseline_dynamics_sensitivity = _new_variational_channel_accumulator(
        control,
        **channel_shape,
    )
    frozen_structure_input_sensitivity = (
        _new_variational_channel_accumulator(
            control,
            **channel_shape,
        )
    )
    impact_accumulators: tuple[
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
        _VariationalChannelAccumulator,
    ] | None = None
    if observation_perturbation is not None:
        impact_accumulators = (
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
            _new_variational_channel_accumulator(control, **channel_shape),
        )
    selected_cap_masks = torch.zeros(
        (selected_count, height, width),
        dtype=torch.bool,
        device=control.device,
    )
    adjoint_iterations = torch.zeros(
        score_shape,
        dtype=torch.int64,
        device=control.device,
    )
    adjoint_relative_residual = control.new_full(
        score_shape,
        float("nan"),
    )
    adjoint_true_residual_norm = control.new_full(
        score_shape,
        float("nan"),
    )
    adjoint_normal_products = torch.zeros(
        score_shape,
        dtype=torch.int64,
        device=control.device,
    )
    adjoint_warm_started = torch.zeros(
        score_shape,
        dtype=torch.bool,
        device=control.device,
    )
    neural_prior_adjoint_direction_maximum_defect = 0.0

    clean_verification = torch.nan_to_num(
        verification_frames,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    )
    verification_finite = verification_bundle.valid_mask
    if verification_finite.shape != result.valid_mask.shape:
        raise ValueError("verification frames must match the forecast shape")
    metric_domain_weights = tuple(
        _metric_domain_weight(
            result,
            verification_finite[forecast_index],
            forecast_index,
            sensitivity_config.metric_domain,
        )
        for forecast_index in forecast_indices
    )
    metric_domain_weight_sum = torch.stack(
        tuple(weight.sum() for weight in metric_domain_weights)
    )
    metric_domain_weight_fraction = metric_domain_weight_sum / float(
        height * width
    )
    metric_domain_digest = _metric_domain_digest(
        sensitivity_config.metric_domain,
        lead_minutes,
        metric_domain_weights,
    )
    truth_linear = dbz_to_echo(
        clean_verification,
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    issued_echo = dbz_to_echo(
        torch.nan_to_num(
            result.forecast_dbz,
            nan=nowcast_config.min_dbz,
            posinf=nowcast_config.max_dbz,
            neginf=nowcast_config.min_dbz,
        ),
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )

    residual_fn, normal_product = _variational_normal_operator(
        control,
        observations,
        frozen,
    )
    baseline_dynamics_path = _prepare_frozen_baseline_dynamics_path(
        observations,
        frozen,
    )
    baseline_dynamics_branch_status: BaselineDynamicsBranchStatus = (
        "not_applicable" if baseline_dynamics_path is None else "unknown"
    )
    baseline_dynamics_trusted = baseline_dynamics_branch_status in (
        "not_applicable",
        "certified",
    )
    if (
        adjoint_config.require_baseline_dynamics_branch_validity
        and observation_perturbation is None
        and not baseline_dynamics_trusted
    ):
        raise ValueError(
            "P1 FSO baseline dynamics branch margins are unavailable"
        )
    normal_product_budget = _NormalProductBudget(
        maximum=adjoint_config.maximum_normal_products
    )
    gauss_newton_diagnostics = _gauss_newton_curvature_diagnostics(
        control,
        residual_fn,
        normal_product,
        normal_product_budget,
        adjoint_config,
    )
    preconditioner = _variational_preconditioner(
        control,
        frozen,
        adjoint_config,
    )
    warm_solutions: dict[int, Tensor] = {}

    observation_count = observations.dbz.numel()
    base_observation_scale = (
        torch.sqrt(observations.quality_weight)
        * frozen.irls_sqrt_weight
        / observations.std_dbz
    )
    detected_observation_scale = torch.where(
        observations.detected_mask,
        base_observation_scale,
        torch.zeros_like(observations.dbz),
    )
    final_trajectory = _analysis_trajectory(control, frozen)
    analyzed_dbz = echo_to_dbz(
        final_trajectory.frames_linear,
        min_dbz=nowcast_config.min_dbz,
    )
    censor_response = torch.sigmoid(
        (
            analyzed_dbz
            - frozen.analysis_config.detection_limit_dbz
        )
        / frozen.analysis_config.censor_temperature_dbz
    )
    censor_error = (
        frozen.analysis_config.censor_temperature_dbz
        * F.softplus(
            (
                analyzed_dbz
                - frozen.analysis_config.detection_limit_dbz
            )
            / frozen.analysis_config.censor_temperature_dbz
        )
    )
    censor_cross_scale = base_observation_scale * (
        censor_response
        + (
            censor_error
            / frozen.analysis_config.censor_temperature_dbz
        )
        * (1.0 - censor_response)
    )
    censor_observation_scale = torch.where(
        observations.censored_mask,
        censor_cross_scale,
        torch.zeros_like(observations.dbz),
    )
    weighted_observation_residual = residual_fn(control)[
        :observation_count
    ].reshape_as(observations.dbz).detach()
    final_state = RadarState(
        echo_linear=final_trajectory.frames_linear[-1],
        displacement_yx=final_trajectory.displacement_yx,
        log_growth_per_step=final_trajectory.log_growth_per_step,
    )
    detection_margin = _minimum_masked_value(
        torch.abs(
            observations.dbz
            - frozen.analysis_config.detection_limit_dbz
        ),
        observations.valid_mask,
    )
    analysis_remap_margin = _analysis_remap_margin(
        final_trajectory.displacement_yx,
        frozen.analysis_remap_cells,
    )
    publication_support_margin, publication_confidence_margin = (
        _publication_margins(result, forecast_indices)
    )
    prior_support_margin = _neural_prior_support_margin(
        frozen,
        neural_prior_application,
    )
    prior_valid_margin = _neural_prior_valid_margin(
        frozen,
        neural_prior_application,
    )
    forecast_remap_margin = math.inf
    output_cap_margin: float | None = None

    for lead_index, forecast_index in enumerate(forecast_indices):
        truth = truth_linear[forecast_index]
        valid = metric_domain_weights[lead_index]
        forecast_step = forecast_index + 1
        lead_cell = freeze_remap_cell(
            forecast_step * analysis.state.displacement_yx
        )
        forecast_remap_margin = min(
            forecast_remap_margin,
            _remap_fraction_margin(
                forecast_step * analysis.state.displacement_yx,
                lead_cell,
            ),
        )
        latent_prediction = _forecast_linear_at_step_core(
            final_state,
            forecast_step,
            nowcast_config,
            lead_cell,
        )
        prediction, cap_active = _freeze_output_cap(
            latent_prediction,
            nowcast_config,
        )
        nominal_valid = result.valid_mask[forecast_index]
        latent_dbz = echo_to_dbz(
            latent_prediction,
            min_dbz=nowcast_config.min_dbz,
        )
        lead_cap_margin = _minimum_masked_value(
            torch.abs(latent_dbz - nowcast_config.max_dbz),
            nominal_valid,
        )
        if lead_cap_margin is not None:
            output_cap_margin = (
                lead_cap_margin
                if output_cap_margin is None
                else min(output_cap_margin, lead_cap_margin)
            )
        if not torch.allclose(
            prediction[nominal_valid],
            issued_echo[forecast_index][nominal_valid],
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError(
                "P1 FSO model disagrees with the issued forecast"
            )
        if lead_index in selected_position:
            selected_cap_masks[selected_position[lead_index]] = cap_active

        for metric_index, metric_name in enumerate(
            sensitivity_config.metric_names
        ):

            score_from_control: Callable[[Tensor], Tensor] = (
                lambda candidate_control: _variational_forecast_score(
                    candidate_control,
                    frozen,
                    forecast_step,
                    lead_cell,
                    cap_active,
                    metric_name,
                    truth,
                    valid,
                    nowcast_config,
                    sensitivity_config,
                )
            )

            if not _metric_has_support(
                metric_name,
                prediction,
                truth,
                valid,
                nowcast_config,
                sensitivity_config,
            ):
                continue

            metric_available[lead_index, metric_index] = True
            score = score_from_control(control)
            rhs = cast(
                Tensor,
                torch.func.grad(score_from_control)(control),
            ).detach()
            initial = (
                warm_solutions.get(metric_index)
                if adjoint_config.warm_start_by_metric
                else None
            )
            adjoint_solve = _variational_observation_adjoint(
                rhs,
                control,
                observations,
                frozen,
                residual_fn,
                normal_product,
                detected_observation_scale,
                censor_observation_scale,
                weighted_observation_residual,
                observation_count,
                adjoint_config=adjoint_config,
                preconditioner=preconditioner,
                initial=initial,
                budget=normal_product_budget,
            )
            if adjoint_config.warm_start_by_metric:
                warm_solutions[metric_index] = adjoint_solve.solution
            observation_sensitivity = adjoint_solve.sensitivity
            background_sensitivity = (
                _frozen_initial_background_observation_sensitivity(
                    adjoint_solve.solution,
                    control,
                    observations,
                    frozen,
                    forecast_step=forecast_step,
                    lead_cell=lead_cell,
                    cap_active=cap_active,
                    metric_name=metric_name,
                    truth=truth,
                    valid=valid,
                    nowcast_config=nowcast_config,
                    sensitivity_config=sensitivity_config,
                )
            )
            prior_input_sensitivity = background_sensitivity
            if frozen.neural_prior_dependency == "exogenous":
                prior_input_sensitivity = torch.zeros_like(background_sensitivity)
            elif frozen.neural_prior_dependency == "radar_dependent":
                assert neural_prior_runner is not None
                assert validated_prior_input is not None
                prior_cotangent = torch.where(
                    _neural_prior_derivative_mask(frozen),
                    background_sensitivity[0],
                    torch.zeros_like(background_sensitivity[0]),
                )
                prior_log_std_cotangent = (
                    _frozen_neural_prior_log_std_sensitivity(
                        adjoint_solve.solution,
                        control,
                        observations,
                        frozen,
                    )
                )
                with neural_prior_runner.derivative_session(
                    validated_prior_input
                ) as derivative_input:
                    neural_prior_adjoint_direction_maximum_defect = max(
                        neural_prior_adjoint_direction_maximum_defect,
                        neural_prior_runner.validate_adjoint_direction(
                            derivative_input,
                            prior_cotangent,
                            prior_log_std_cotangent,
                        ),
                    )
                    prior_input_sensitivity = neural_prior_runner.vjp_components(
                        derivative_input,
                        prior_cotangent,
                        prior_log_std_cotangent,
                    )
            dynamics_sensitivity = (
                _frozen_baseline_dynamics_observation_sensitivity(
                    baseline_dynamics_path,
                    adjoint_solve.solution,
                    control,
                    observations,
                    frozen,
                    forecast_step=forecast_step,
                    lead_cell=lead_cell,
                    cap_active=cap_active,
                    metric_name=metric_name,
                    truth=truth,
                    valid=valid,
                    nowcast_config=nowcast_config,
                    sensitivity_config=sensitivity_config,
                )
            )
            frozen_structure_input_sensitivity_values = (
                observation_sensitivity.detected_dbz
                + prior_input_sensitivity
                + dynamics_sensitivity
            )

            forecast_scores[lead_index, metric_index] = score.detach()
            adjoint_iterations[lead_index, metric_index] = (
                adjoint_solve.iterations
            )
            adjoint_relative_residual[lead_index, metric_index] = (
                adjoint_solve.relative_residual
            )
            adjoint_true_residual_norm[
                lead_index,
                metric_index,
            ] = adjoint_solve.true_residual_norm
            adjoint_normal_products[lead_index, metric_index] = (
                adjoint_solve.normal_products
            )
            adjoint_warm_started[lead_index, metric_index] = (
                adjoint_solve.warm_started
            )
            selected_index = selected_position.get(lead_index)
            sensitivity_channels = (
                (detected_sensitivity, observation_sensitivity.detected_dbz),
                (
                    censor_sensitivity,
                    observation_sensitivity.censor_threshold_dbz,
                ),
                (
                    weight_sensitivity,
                    observation_sensitivity.observation_weight,
                ),
                (
                    initial_background_sensitivity,
                    background_sensitivity,
                ),
                (
                    baseline_dynamics_sensitivity,
                    dynamics_sensitivity,
                ),
                (
                    frozen_structure_input_sensitivity,
                    frozen_structure_input_sensitivity_values,
                ),
            )
            for accumulator, values in sensitivity_channels:
                _record_variational_channel(
                    accumulator,
                    values,
                    lead_index=lead_index,
                    metric_index=metric_index,
                    selected_index=selected_index,
                    tile_size=tile_shape_yx,
                    signed_sum=False,
                )

            if (
                observation_perturbation is not None
                and impact_accumulators is not None
            ):
                background_impact = (
                    prior_input_sensitivity
                    * observation_perturbation.physical_radar_dbz_delta
                    if observation_perturbation.perturbation_semantics
                    == "physical_radar_value"
                    and observation_perturbation.physical_radar_dbz_delta
                    is not None
                    else background_sensitivity
                    * _initial_background_perturbation(
                        observation_perturbation,
                        observations,
                    )
                )
                component_impacts = (
                    observation_sensitivity.detected_dbz
                    * observation_perturbation.detected_dbz,
                    observation_sensitivity.censor_threshold_dbz
                    * observation_perturbation.censor_threshold_dbz,
                    observation_sensitivity.observation_weight
                    * observation_perturbation.observation_weight,
                    background_impact,
                    dynamics_sensitivity
                    * _baseline_dynamics_perturbation(
                        observation_perturbation,
                        observations,
                    ),
                )
                total_impact = (
                    component_impacts[0]
                    + component_impacts[1]
                    + component_impacts[2]
                    + component_impacts[3]
                    + component_impacts[4]
                )
                for accumulator, values in zip(
                    impact_accumulators,
                    (*component_impacts, total_impact),
                    strict=True,
                ):
                    _record_variational_channel(
                        accumulator,
                        values,
                        lead_index=lead_index,
                        metric_index=metric_index,
                        selected_index=selected_index,
                        tile_size=tile_shape_yx,
                        signed_sum=True,
                    )

    low_local_validity = (
        detection_margin is None
        or detection_margin < adjoint_config.minimum_detection_margin_dbz
        or analysis_remap_margin
        < adjoint_config.minimum_remap_fraction_margin
        or forecast_remap_margin
        < adjoint_config.minimum_remap_fraction_margin
        or output_cap_margin is None
        or output_cap_margin < adjoint_config.minimum_output_cap_margin_dbz
        or publication_support_margin
        < adjoint_config.minimum_publication_margin
        or (
            publication_confidence_margin is not None
            and publication_confidence_margin
            < adjoint_config.minimum_publication_margin
        )
        or (
            frozen.neural_prior_dependency is not None
            and (
                prior_valid_margin is None
                or prior_valid_margin
                < adjoint_config.minimum_neural_prior_valid_margin
                or prior_support_margin is None
                or prior_support_margin
                < adjoint_config.minimum_neural_prior_support_margin
            )
        )
    )
    active_set_margins = VariationalActiveSetMargins(
        detection_classification_dbz=detection_margin,
        analysis_remap_fraction=analysis_remap_margin,
        forecast_remap_fraction=forecast_remap_margin,
        output_cap_dbz=output_cap_margin,
        publication_support=publication_support_margin,
        publication_confidence=publication_confidence_margin,
        neural_prior_valid_probability=prior_valid_margin,
        neural_prior_support_probability=prior_support_margin,
        low_local_validity=low_local_validity,
    )
    if adjoint_config.require_active_set_margin and low_local_validity:
        raise ValueError("P1 FSO active-set margin is below its requirement")

    frozen_structure_channel = _sensitivity_channel(
        frozen_structure_input_sensitivity
    )
    observation_sensitivity = VariationalObservationSensitivity(
        detected_dbz=_sensitivity_channel(detected_sensitivity),
        censor_threshold_dbz=_sensitivity_channel(censor_sensitivity),
        observation_weight=_sensitivity_channel(weight_sensitivity),
        initial_background_dbz=_sensitivity_channel(
            initial_background_sensitivity
        ),
        baseline_dynamics_dbz=_sensitivity_channel(
            baseline_dynamics_sensitivity
        ),
        frozen_structure_input_dbz=frozen_structure_channel,
        baseline_branch_trusted_frozen_structure_input_dbz=(
            frozen_structure_channel if baseline_dynamics_trusted else None
        ),
    )
    if verification_bundle.contract == "radar-verification-bundle-v17":
        fso_contract = CURRENT_VARIATIONAL_FSO_CONTRACT
    elif verification_bundle.contract == "legacy-verification-tensor-v1":
        fso_contract = EXPLORATORY_VARIATIONAL_FSO_CONTRACT
    else:
        raise ValueError(
            "verification generation is audit-only for variational FSO"
        )
    fso = VariationalFSO(
        contract=fso_contract,
        forecast_run_digest=result.forecast_run_digest,
        analysis_input_digest=cast(str, result.run.analysis_input_digest),
        sensitivity_config_digest=sensitivity_config.digest,
        adjoint_config_digest=adjoint_config.digest,
        linearization_contract=linearization.contract,
        linearization_digest=linearization.linearization_digest,
        verification_contract=verification_bundle.contract,
        verification_bundle_digest=verification_bundle.content_digest,
        verification_lineage_complete=(
            verification_bundle.lineage_complete
        ),
        verification_valid_times=verification_bundle.valid_times,
        verification_grid_contract_digest=(
            verification_bundle.grid_contract_digest
        ),
        verification_radar_product_digest=(
            verification_bundle.radar_product_digest
        ),
        verification_qc_pipeline_digest=(
            verification_bundle.qc_pipeline_digest
        ),
        metric_contract_digest=_metric_contract_digest(sensitivity_config),
        algorithm_bundle_digest=linearization.algorithm_bundle_digest,
        numerical_runtime_digest=linearization.numerical_runtime_digest,
        variational_fso_digest="",
        sensitivity_scope=(
            "residual_plus_input_dependent_initial_state_and_baseline_with_frozen_selection"
        ),
        baseline_dynamics_frozen=False,
        baseline_pair_selection_frozen=True,
        baseline_dynamics_branch_status=baseline_dynamics_branch_status,
        metric_names=sensitivity_config.metric_names,
        metric_domain=sensitivity_config.metric_domain,
        metric_domain_digest=metric_domain_digest,
        lead_minutes=lead_minutes,
        full_map_lead_minutes=sensitivity_config.full_map_lead_minutes,
        tile_size=max(tile_shape_yx),
        tile_shape_yx=tile_shape_yx,
        forecast_scores=forecast_scores,
        metric_available=metric_available,
        metric_domain_weight_sum=metric_domain_weight_sum,
        metric_domain_weight_fraction=metric_domain_weight_fraction,
        forecast_cap_active_mask=selected_cap_masks,
        observation=observation_sensitivity,
        adjoint_iterations=adjoint_iterations,
        adjoint_relative_residual=adjoint_relative_residual,
        adjoint_true_residual_norm=adjoint_true_residual_norm,
        adjoint_normal_products=adjoint_normal_products,
        adjoint_warm_started=adjoint_warm_started,
        total_normal_products=normal_product_budget.used,
        whitener_operations_per_apply=0,
        observed_whitener_apply_count=0,
        materialized_output_bytes=materialized_output_bytes,
        neural_prior_adjoint_direction_maximum_defect=(
            neural_prior_adjoint_direction_maximum_defect
        ),
        active_set_margins=active_set_margins,
        feasibility_margins=feasibility_margins,
        gauss_newton_diagnostics=gauss_newton_diagnostics,
    )
    fso = replace(fso, variational_fso_digest=variational_fso_digest(fso))
    if validated_prior_input is not None:
        validated_prior_input.validate_completion()
    if impact_accumulators is None:
        return fso, None, None
    total_impact = _impact_channel(impact_accumulators[5])
    if perturbation_diagnostics is None:
        raise RuntimeError("variational perturbation diagnostics are missing")
    impact_branch_trusted = (
        perturbation_diagnostics.baseline_dynamics_branch_status
        in ("not_applicable", "certified")
    )
    return (
        fso,
        VariationalObservationImpact(
            detected_dbz=_impact_channel(impact_accumulators[0]),
            censor_threshold_dbz=_impact_channel(impact_accumulators[1]),
            observation_weight=_impact_channel(impact_accumulators[2]),
            initial_background_dbz=_impact_channel(
                impact_accumulators[3]
            ),
            baseline_dynamics_dbz=_impact_channel(
                impact_accumulators[4]
            ),
            total=total_impact,
            baseline_branch_trusted_total=(
                total_impact if impact_branch_trusted else None
            ),
        ),
        perturbation_diagnostics,
    )


def _variational_state(
    control: Tensor,
    frozen: FrozenOuterState,
) -> RadarState:
    trajectory = _analysis_trajectory(control, frozen)
    return RadarState(
        echo_linear=trajectory.frames_linear[-1],
        displacement_yx=trajectory.displacement_yx,
        log_growth_per_step=trajectory.log_growth_per_step,
    )


def _variational_forecast_score(
    control: Tensor,
    frozen: FrozenOuterState,
    step: int,
    lead_cell: RemapCell,
    cap_active: Tensor,
    metric_name: str,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> Tensor:
    latent = _forecast_linear_at_step_core(
        _variational_state(control, frozen),
        step,
        nowcast_config,
        lead_cell,
    )
    return forecast_metric(
        metric_name,
        _apply_output_cap(latent, cap_active, nowcast_config),
        truth,
        valid,
        nowcast_config,
        sensitivity_config,
        frozen.grid_time_contract,
    )


def _variational_normal_operator(
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    residual_fn: Callable[[Tensor], Tensor] = lambda value: residual_vector(
        value,
        observations,
        frozen,
    )
    residual_vjp = torch.func.vjp(residual_fn, control)
    pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        residual_vjp[1],
    )

    def normal_product(direction: Tensor) -> Tensor:
        jacobian_direction = cast(
            Tensor,
            torch.func.jvp(residual_fn, (control,), (direction,))[1],
        )
        return pullback(jacobian_direction)[0]

    return residual_fn, normal_product


def _variational_observation_adjoint(
    rhs: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    residual_fn: Callable[[Tensor], Tensor],
    normal_product: Callable[[Tensor], Tensor],
    detected_observation_scale: Tensor,
    censor_observation_scale: Tensor,
    weighted_observation_residual: Tensor,
    observation_count: int,
    *,
    adjoint_config: VariationalAdjointConfig,
    preconditioner: Callable[[Tensor], Tensor] | None,
    initial: Tensor | None,
    budget: _NormalProductBudget,
) -> _VariationalAdjointSolve:
    products_before = budget.used

    def counted_normal_product(value: Tensor) -> Tensor:
        return budget.apply(normal_product, value)

    relative_tolerance = (
        frozen.analysis_config.pcg_relative_tolerance
        if adjoint_config.pcg_relative_tolerance is None
        else adjoint_config.pcg_relative_tolerance
    )
    maximum_iterations = (
        frozen.analysis_config.maximum_pcg_iterations
        if adjoint_config.maximum_pcg_iterations is None
        else adjoint_config.maximum_pcg_iterations
    )
    try:
        adjoint = pcg(
            counted_normal_product,
            rhs,
            preconditioner=preconditioner,
            initial=initial,
            rtol=relative_tolerance,
            max_iterations=maximum_iterations,
        )
    except (ArithmeticError, RuntimeError, ValueError) as error:
        if str(error) == "P1 FSO normal-product budget exhausted":
            raise ValueError(
                "P1 FSO normal-product budget exhausted"
            ) from error
        raise ValueError("P1 FSO adjoint solve failed") from error
    if not adjoint.converged or not bool(
        torch.all(torch.isfinite(adjoint.solution))
    ):
        raise ValueError("P1 FSO adjoint solve did not converge")
    rhs_norm = float(torch.linalg.vector_norm(rhs.detach()).cpu())
    true_residual_norm = adjoint.relative_residual * rhs_norm

    jacobian_adjoint = cast(
        Tensor,
        torch.func.jvp(
            residual_fn,
            (control,),
            (adjoint.solution,),
        )[1],
    )
    prediction_response = jacobian_adjoint[:observation_count].reshape_as(
        observations.dbz
    )
    # Detected dBZ changes only the residual offset. The censored threshold
    # also changes the observation Jacobian, so its cross scale includes the
    # (dJ/dL).T r contribution assembled above. The objective-weight channel
    # differentiates alpha * 0.5 * r_i**2 at alpha=1.
    if frozen.analysis_config.observation_common_bias_std_dbz > 0.0:
        sensitivity = _correlated_observation_parameter_sensitivity(
            adjoint.solution,
            control,
            observations,
            frozen,
        )
    else:
        sensitivity = _VariationalAdjointSensitivity(
            detected_dbz=(
                detected_observation_scale * prediction_response
            ).detach(),
            censor_threshold_dbz=(
                censor_observation_scale * prediction_response
            ).detach(),
            observation_weight=(
                -weighted_observation_residual * prediction_response
            ).detach(),
        )
    return _VariationalAdjointSolve(
        sensitivity=sensitivity,
        solution=adjoint.solution.detach(),
        iterations=adjoint.iterations,
        relative_residual=adjoint.relative_residual,
        true_residual_norm=true_residual_norm,
        normal_products=budget.used - products_before,
        warm_started=initial is not None,
    )


def _frozen_initial_background_observation_sensitivity(
    adjoint_solution: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    forecast_step: int,
    lead_cell: RemapCell,
    cap_active: Tensor,
    metric_name: str,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> Tensor:
    """Differentiate accepted first-frame values through the P1 background.

    The active field, P0-derived baseline dynamics, remap cells, observation
    classes, and every other frozen structure remain fixed. The result has the
    observation shape and is nonzero only where the first frame supplied the
    P1 initial background.
    """

    initial_background = frozen.initial_background_dbz.detach()

    def frozen_with_background(candidate: Tensor) -> FrozenOuterState:
        return replace(frozen, initial_background_dbz=candidate)

    def stationarity_from_background(candidate: Tensor) -> Tensor:
        def objective(candidate_control: Tensor) -> Tensor:
            residual = residual_vector(
                candidate_control,
                observations,
                frozen_with_background(candidate),
            )
            return 0.5 * torch.dot(residual, residual)

        return cast(Tensor, torch.func.grad(objective)(control))

    stationarity_pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        torch.func.vjp(
            stationarity_from_background,
            initial_background,
        )[1],
    )
    implicit = -stationarity_pullback(adjoint_solution)[0]

    def score_from_background(candidate: Tensor) -> Tensor:
        return _variational_forecast_score(
            control,
            frozen_with_background(candidate),
            forecast_step,
            lead_cell,
            cap_active,
            metric_name,
            truth,
            valid,
            nowcast_config,
            sensitivity_config,
        )

    direct = cast(
        Tensor,
        torch.func.grad(score_from_background)(initial_background),
    )
    accepted_first_frame = (
        observations.valid_mask[0] & frozen.observed_mask[0]
    )
    if not frozen.observation_derived_initial_background:
        if frozen.neural_prior_valid_mask is None:
            raise ValueError("neural-prior background validity is missing")
        accepted_first_frame = frozen.neural_prior_valid_mask
    first_frame = torch.where(
        accepted_first_frame,
        direct + implicit,
        torch.zeros_like(initial_background),
    )
    return torch.cat(
        (
            first_frame.unsqueeze(0),
            torch.zeros_like(observations.dbz[1:]),
        ),
        dim=0,
    ).detach()


def _frozen_neural_prior_log_std_sensitivity(
    adjoint_solution: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> Tensor:
    """Differentiate P1 stationarity through the spatial prior precision."""

    prior_std = frozen.neural_prior_std_dbz
    prior_valid = frozen.neural_prior_valid_mask
    if prior_std is None or prior_valid is None:
        raise ValueError("neural-prior uncertainty state is missing")
    log_std = torch.log(prior_std.detach())

    def stationarity(candidate_log_std: Tensor) -> Tensor:
        candidate_frozen = replace(
            frozen,
            neural_prior_std_dbz=torch.exp(candidate_log_std),
        )

        def objective(candidate_control: Tensor) -> Tensor:
            residual = residual_vector(
                candidate_control,
                observations,
                candidate_frozen,
            )
            return 0.5 * torch.dot(residual, residual)

        return cast(Tensor, torch.func.grad(objective)(control))

    pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        torch.func.vjp(stationarity, log_std)[1],
    )
    sensitivity = -pullback(adjoint_solution)[0]
    return torch.where(
        prior_valid,
        sensitivity,
        torch.zeros_like(sensitivity),
    ).detach()


def _prepare_frozen_baseline_dynamics_path(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> _FrozenBaselineDynamicsPath | None:
    """Freeze P0 pair/peak branches and retain their continuous VJP."""

    if (
        frozen.baseline_metadata.tendency_source
        is not TendencySource.OBSERVATION
    ):
        return None
    observation_dbz = observations.dbz.detach()

    def dynamics_from_observation(candidate: Tensor) -> Tensor:
        return _baseline_dynamics_from_observation(candidate, frozen)

    nominal_dynamics, pullback = cast(
        tuple[Tensor, Callable[[Tensor], tuple[Tensor]]],
        torch.func.vjp(
            dynamics_from_observation,
            observation_dbz,
        ),
    )
    expected = torch.cat(
        (
            frozen.baseline_state.displacement_yx,
            frozen.baseline_state.log_growth_per_step.reshape(1),
        )
    )
    tolerance = frozen.nowcast_config.contract_absolute_tolerance
    if not torch.allclose(
        nominal_dynamics,
        expected,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "frozen P0 dynamics path does not reproduce the baseline state"
        )
    active = observations.detected_mask & frozen.observed_mask
    return _FrozenBaselineDynamicsPath(
        active_mask=active,
        nominal_dynamics=nominal_dynamics.detach(),
        observation_pullback=pullback,
    )


def _baseline_dynamics_from_observation(
    frames_dbz: Tensor,
    frozen: FrozenOuterState,
) -> Tensor:
    floor = frames_dbz.new_full((), frozen.nowcast_config.min_dbz)
    clean = torch.where(frozen.observed_mask, frames_dbz, floor)
    linear = dbz_to_echo(
        clean,
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    )
    estimate = _estimate_source_tendencies(
        clean,
        frozen.observed_mask,
        linear,
        frozen.nowcast_config,
        frozen.grid_time_contract,
    )
    return torch.cat(
        (
            estimate.displacement_yx,
            estimate.log_growth_per_step.reshape(1),
        )
    )


def _frozen_baseline_dynamics_observation_sensitivity(
    path: _FrozenBaselineDynamicsPath | None,
    adjoint_solution: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    forecast_step: int,
    lead_cell: RemapCell,
    cap_active: Tensor,
    metric_name: str,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> Tensor:
    """Differentiate continuous P0 dynamics with pair/peak selection fixed."""

    if path is None:
        return torch.zeros_like(observations.dbz)

    def frozen_with_dynamics(candidate: Tensor) -> FrozenOuterState:
        state = RadarState(
            echo_linear=frozen.baseline_state.echo_linear,
            displacement_yx=candidate[:2],
            log_growth_per_step=candidate[2],
        )
        return replace(frozen, baseline_state=state)

    def stationarity_from_dynamics(candidate: Tensor) -> Tensor:
        def objective(candidate_control: Tensor) -> Tensor:
            residual = residual_vector(
                candidate_control,
                observations,
                frozen_with_dynamics(candidate),
            )
            return 0.5 * torch.dot(residual, residual)

        return cast(Tensor, torch.func.grad(objective)(control))

    stationarity_pullback = cast(
        Callable[[Tensor], tuple[Tensor]],
        torch.func.vjp(
            stationarity_from_dynamics,
            path.nominal_dynamics,
        )[1],
    )
    implicit = -stationarity_pullback(adjoint_solution)[0]

    def score_from_dynamics(candidate: Tensor) -> Tensor:
        return _variational_forecast_score(
            control,
            frozen_with_dynamics(candidate),
            forecast_step,
            lead_cell,
            cap_active,
            metric_name,
            truth,
            valid,
            nowcast_config,
            sensitivity_config,
        )

    direct = cast(
        Tensor,
        torch.func.grad(score_from_dynamics)(path.nominal_dynamics),
    )
    observation_gradient = path.observation_pullback(direct + implicit)[0]
    return torch.where(
        path.active_mask,
        observation_gradient,
        torch.zeros_like(observation_gradient),
    ).detach()


def _correlated_observation_parameter_sensitivity(
    adjoint_solution: Tensor,
    control: Tensor,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
) -> _VariationalAdjointSensitivity:
    """Differentiate frozen stationarity through a non-diagonal whitener."""

    detected_values = observations.dbz.detach()
    censor_threshold = observations.dbz.new_full(
        observations.dbz.shape,
        frozen.analysis_config.detection_limit_dbz,
    )
    observation_multiplier = torch.ones_like(observations.dbz)

    def stationarity(
        candidate_detected: Tensor,
        candidate_threshold: Tensor,
        candidate_multiplier: Tensor,
    ) -> Tensor:
        def observation_objective(candidate_control: Tensor) -> Tensor:
            trajectory = _analysis_trajectory(candidate_control, frozen)
            prediction = echo_to_dbz(
                trajectory.frames_linear,
                min_dbz=frozen.nowcast_config.min_dbz,
            )
            detected_error = prediction - candidate_detected
            censored_error = (
                frozen.analysis_config.censor_temperature_dbz
                * F.softplus(
                    (
                        prediction - candidate_threshold
                    )
                    / frozen.analysis_config.censor_temperature_dbz
                )
            )
            error = torch.where(
                observations.detected_mask,
                detected_error,
                torch.where(
                    observations.censored_mask,
                    censored_error,
                    torch.zeros_like(prediction),
                ),
            )
            standardized = (
                torch.sqrt(observations.quality_weight)
                * torch.sqrt(candidate_multiplier)
                * error
                / observations.std_dbz
            )
            whitened = _apply_observation_error_whitener(
                standardized,
                observations,
                frozen.analysis_config,
                whitener=frozen.observation_whitener,
            )
            residual = frozen.irls_sqrt_weight * whitened
            return 0.5 * torch.dot(residual.flatten(), residual.flatten())

        return cast(
            Tensor,
            torch.func.grad(observation_objective)(control),
        )

    vjp_result = torch.func.vjp(
        stationarity,
        detected_values,
        censor_threshold,
        observation_multiplier,
    )
    pullback = cast(
        Callable[[Tensor], tuple[Tensor, Tensor, Tensor]],
        vjp_result[1],
    )
    detected_gradient, censor_gradient, weight_gradient = pullback(
        adjoint_solution
    )
    return _VariationalAdjointSensitivity(
        detected_dbz=torch.where(
            observations.detected_mask,
            -detected_gradient,
            torch.zeros_like(detected_gradient),
        ).detach(),
        censor_threshold_dbz=torch.where(
            observations.censored_mask,
            -censor_gradient,
            torch.zeros_like(censor_gradient),
        ).detach(),
        observation_weight=torch.where(
            observations.valid_mask,
            -weight_gradient,
            torch.zeros_like(weight_gradient),
        ).detach(),
    )


def _validate_variational_observation_perturbation(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    config: VariationalAdjointConfig,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> VariationalPerturbationDiagnostics:
    if perturbation.contract != "p1-observation-perturbation-v7":
        raise ValueError("unsupported P1 observation perturbation contract")
    if perturbation.perturbation_semantics not in (
        "augmented_parameter",
        "physical_radar_value",
    ):
        raise ValueError("unsupported observation perturbation semantics")
    channels = (
        (
            "detected_dbz",
            perturbation.detected_dbz,
            observations.detected_mask,
        ),
        (
            "censor_threshold_dbz",
            perturbation.censor_threshold_dbz,
            observations.censored_mask,
        ),
        (
            "observation_weight",
            perturbation.observation_weight,
            observations.valid_mask,
        ),
    )
    for name, values, active_mask in channels:
        _validate_perturbation_tensor(
            name,
            values,
            observations,
            active_mask,
        )
    _require_local_perturbation(
        "detected_dbz",
        perturbation.detected_dbz,
        config.maximum_detected_delta_dbz,
    )
    _require_local_perturbation(
        "censor_threshold_dbz",
        perturbation.censor_threshold_dbz,
        config.maximum_censor_delta_dbz,
    )
    _require_local_perturbation(
        "observation_weight",
        perturbation.observation_weight,
        config.maximum_observation_weight_delta,
    )
    if perturbation.initial_background_dbz is not None:
        values = perturbation.initial_background_dbz
        active = torch.zeros_like(observations.valid_mask)
        if frozen.neural_prior_dependency == "radar_dependent":
            if frozen.neural_prior_valid_mask is None:
                raise ValueError("neural-prior perturbation lacks a valid mask")
            active[0] = frozen.neural_prior_valid_mask
        else:
            active[0] = observations.valid_mask[0] & frozen.observed_mask[0]
        _validate_perturbation_tensor(
            "initial_background_dbz",
            values,
            observations,
            active,
            active_domain="accepted first-frame observations",
        )
    if perturbation.baseline_dynamics_dbz is not None:
        values = perturbation.baseline_dynamics_dbz
        active = observations.valid_mask & frozen.observed_mask
        _validate_perturbation_tensor(
            "baseline_dynamics_dbz",
            values,
            observations,
            active,
            active_domain="accepted observations",
        )
    for name, values in (
        ("initial_background_dbz", perturbation.initial_background_dbz),
        ("baseline_dynamics_dbz", perturbation.baseline_dynamics_dbz),
    ):
        if values is not None:
            _require_local_perturbation(
                name,
                values,
                config.maximum_background_delta_dbz,
            )
    physical_delta = _validate_physical_radar_semantics(
        perturbation,
        observations,
        frozen,
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )
    _validate_directional_classification(
        perturbation,
        observations,
        frozen,
        config,
        physical_delta,
    )
    return _perturbation_diagnostics(
        perturbation,
        observations,
        frozen,
        config,
        physical_delta,
    )


def _validate_perturbation_tensor(
    name: str,
    values: Tensor,
    observations: AnalysisObservations,
    active_mask: Tensor,
    *,
    active_domain: str = "its active mask",
) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} perturbation must be a Tensor")
    if values.shape != observations.dbz.shape:
        raise ValueError(f"{name} perturbation shape mismatch")
    if values.dtype != observations.dbz.dtype:
        raise ValueError(f"{name} perturbation dtype mismatch")
    if values.device != observations.dbz.device:
        raise ValueError(f"{name} perturbation device mismatch")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError(f"{name} perturbation must be finite")
    if bool(torch.any(values.masked_select(~active_mask) != 0)):
        raise ValueError(
            f"{name} perturbation must be zero outside {active_domain}"
        )


def _validate_physical_radar_semantics(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    *,
    neural_prior_runner: NeuralPriorInferenceRunner | None = None,
    neural_prior_application: NeuralPriorApplication | None = None,
) -> Tensor | None:
    physical_delta = perturbation.physical_radar_dbz_delta
    if perturbation.perturbation_semantics == "augmented_parameter":
        if physical_delta is not None:
            raise ValueError(
                "augmented parameter perturbation cannot carry a physical "
                "dBZ delta"
            )
        return None
    if physical_delta is None:
        raise ValueError(
            "physical radar perturbation requires its source dBZ delta"
        )
    _validate_perturbation_tensor(
        "physical_radar_dbz_delta",
        physical_delta,
        observations,
        observations.detected_mask,
        active_domain="detected observations",
    )
    detected, background, dynamics = _physical_radar_channels(
        physical_delta,
        observations,
        frozen,
        neural_prior_runner=neural_prior_runner,
        neural_prior_application=neural_prior_application,
    )
    expected = (
        detected,
        torch.zeros_like(physical_delta),
        torch.zeros_like(physical_delta),
        background,
        dynamics,
    )
    actual = (
        perturbation.detected_dbz,
        perturbation.censor_threshold_dbz,
        perturbation.observation_weight,
        perturbation.initial_background_dbz,
        perturbation.baseline_dynamics_dbz,
    )
    if any(
        value is None or not torch.equal(value, canonical)
        for value, canonical in zip(actual, expected, strict=True)
    ):
        raise ValueError(
            "physical radar perturbation channels are inconsistent"
        )
    _ = _physical_radar_input_margins(
        physical_delta,
        observations,
        frozen,
    )
    return physical_delta


def _validate_directional_classification(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    config: VariationalAdjointConfig,
    physical_delta: Tensor | None,
) -> None:
    delta_dbz = (
        perturbation.detected_dbz
        if physical_delta is None
        else physical_delta
    )
    changed_dbz = observations.dbz + delta_dbz
    changed_limit = (
        frozen.analysis_config.detection_limit_dbz
        + perturbation.censor_threshold_dbz
    )
    margin = config.minimum_detection_margin_dbz
    changed_classification = (delta_dbz != 0) | (
        perturbation.censor_threshold_dbz != 0
    )
    changed_detected = observations.detected_mask & changed_classification
    changed_censored = observations.censored_mask & changed_classification
    detected_valid = torch.all(
        changed_dbz.masked_select(changed_detected)
        >= changed_limit.masked_select(changed_detected) + margin
    )
    censored_valid = torch.all(
        changed_dbz.masked_select(changed_censored)
        <= changed_limit.masked_select(changed_censored) - margin
    )
    if not bool(detected_valid & censored_valid):
        raise ValueError(
            "observation perturbation crosses the detected/censored branch"
        )
    if bool(
        torch.any(
            1.0 + perturbation.observation_weight
            < config.minimum_observation_multiplier
        )
    ):
        raise ValueError(
            "observation perturbation crosses the weight-multiplier branch"
        )


def _perturbation_diagnostics(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    config: VariationalAdjointConfig,
    physical_delta: Tensor | None,
) -> VariationalPerturbationDiagnostics:
    floor_margin, ceiling_margin = (
        (None, None)
        if physical_delta is None
        else _physical_radar_input_margins(
            physical_delta,
            observations,
            frozen,
        )
    )
    baseline_delta = _baseline_dynamics_perturbation(
        perturbation,
        observations,
    )
    baseline_branch_status, baseline_signature_digest = (
        _baseline_dynamics_branch_certification(
            observations,
            frozen,
            baseline_delta,
        )
    )
    if baseline_branch_status == "invalid":
        raise ValueError(
            "observation perturbation crosses the frozen P0 tendency branch"
        )

    dbz_channels = (
        (physical_delta,)
        if physical_delta is not None
        else (
            perturbation.detected_dbz,
            perturbation.censor_threshold_dbz,
            perturbation.initial_background_dbz,
            perturbation.baseline_dynamics_dbz,
        )
    )
    active = perturbation.observation_weight != 0
    whitened_energy = torch.zeros_like(observations.dbz)
    for values in dbz_channels:
        if values is None:
            continue
        active |= values != 0
        standardized = (
            torch.sqrt(observations.quality_weight)
            * values
            / observations.std_dbz
        )
        whitened = _apply_observation_error_whitener(
            standardized,
            observations,
            frozen.analysis_config,
            whitener=frozen.observation_whitener,
        )
        whitened_energy += whitened.square()

    pixel_count = int(torch.count_nonzero(active).detach())
    valid_count = max(1, int(torch.count_nonzero(observations.valid_mask)))
    fraction = pixel_count / valid_count
    grid = frozen.grid_time_contract
    area_km2 = (
        None
        if grid is None
        else pixel_count * grid.cell_area_m2 / 1.0e6
    )
    whitened_l2 = math.sqrt(float(torch.sum(whitened_energy).detach()))
    tile_norm = _maximum_tile_norm(
        whitened_energy,
        _perturbation_tile_size(config, grid),
    )
    weight_l2 = float(
        torch.linalg.vector_norm(perturbation.observation_weight).detach()
    )
    limits = (
        (pixel_count, config.maximum_perturbed_pixel_count, "pixel budget"),
        (fraction, config.maximum_perturbed_fraction, "area fraction"),
        (
            whitened_l2,
            config.maximum_whitened_perturbation_l2,
            "whitened trust radius",
        ),
        (
            tile_norm,
            config.maximum_per_tile_whitened_norm,
            "per-tile trust radius",
        ),
        (
            weight_l2,
            config.maximum_observation_weight_l2,
            "weight trust radius",
        ),
    )
    for value, limit, name in limits:
        if value > limit:
            raise ValueError(f"observation perturbation exceeds its {name}")
    if config.maximum_perturbed_area_km2 is not None:
        if grid is None or area_km2 is None:
            raise ValueError(
                "physical perturbation area requires a grid contract"
            )
        try:
            grid.validate_projected_area_maximum(
                area_km2,
                config.maximum_perturbed_area_km2,
            )
        except ValueError as error:
            raise ValueError(
                "observation perturbation exceeds or is uncertain against its "
                "physical area budget"
            ) from error
    return VariationalPerturbationDiagnostics(
        perturbed_pixel_count=pixel_count,
        perturbed_fraction=fraction,
        perturbed_area_km2=area_km2,
        whitened_l2=whitened_l2,
        maximum_per_tile_whitened_norm=tile_norm,
        observation_weight_l2=weight_l2,
        minimum_input_floor_margin_dbz=floor_margin,
        minimum_input_ceiling_margin_dbz=ceiling_margin,
        directional_classification_valid=True,
        baseline_dynamics_branch_status=baseline_branch_status,
        baseline_dynamics_branch_signature_digest=(
            baseline_signature_digest
        ),
    )


def _maximum_tile_norm(energy: Tensor, tile_size: TileShape) -> float:
    maximum = 0.0
    tile_height, tile_width = tile_size
    for frame in energy:
        for row in range(0, frame.shape[0], tile_height):
            for column in range(0, frame.shape[1], tile_width):
                tile = frame[
                    row : row + tile_height,
                    column : column + tile_width,
                ]
                maximum = max(
                    maximum,
                    math.sqrt(float(torch.sum(tile).detach())),
                )
    return maximum


def _baseline_branch_is_stable(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    delta_dbz: Tensor,
) -> bool:
    if (
        frozen.baseline_metadata.tendency_source
        is not TendencySource.OBSERVATION
    ):
        return not bool(torch.any(delta_dbz != 0))
    status, _ = _baseline_dynamics_branch_certification(
        observations,
        frozen,
        delta_dbz,
    )
    return status == "certified"


def _baseline_dynamics_branch_certification(
    observations: AnalysisObservations,
    frozen: FrozenOuterState,
    delta_dbz: Tensor,
) -> tuple[BaselineDynamicsBranchStatus, str | None]:
    if (
        frozen.baseline_metadata.tendency_source
        is not TendencySource.OBSERVATION
    ):
        return "not_applicable", None
    nominal = _p0_tendency_branch_signature(observations.dbz, frozen)
    signature_digest = dataclass_digest(nominal)
    if not bool(torch.any(delta_dbz != 0)):
        return "certified", signature_digest
    for scale in (0.5, 1.0):
        changed = observations.dbz + scale * delta_dbz
        if _p0_tendency_branch_signature(changed, frozen) != nominal:
            return "invalid", signature_digest
    return "certified", signature_digest


def _p0_tendency_branch_signature(
    frames_dbz: Tensor,
    frozen: FrozenOuterState,
) -> P0TendencyBranchSignature:
    floor = frames_dbz.new_full((), frozen.nowcast_config.min_dbz)
    clean = torch.where(frozen.observed_mask, frames_dbz, floor)
    linear = dbz_to_echo(
        clean,
        min_dbz=frozen.nowcast_config.min_dbz,
        max_dbz=frozen.nowcast_config.max_dbz,
    )
    estimate = _estimate_source_tendencies(
        clean,
        frozen.observed_mask,
        linear,
        frozen.nowcast_config,
        frozen.grid_time_contract,
    )
    pair_spans = ((0, 1), (1, 2), (0, 2))
    peaks: list[tuple[int, int]] = []
    interiors: list[bool] = []
    available: list[bool] = []
    growth_available: list[bool] = []
    for previous, current in pair_spans:
        common = frozen.observed_mask[previous] & frozen.observed_mask[current]
        previous_dbz = torch.where(common, clean[previous], floor)
        current_dbz = torch.where(common, clean[current], floor)
        step_span = current - previous
        limits = motion_displacement_limits_yx(
            frozen.nowcast_config,
            frozen.grid_time_contract,
            previous_dbz,
        )
        _, _, interior, peak = _phase_correlation_details(
            previous_dbz,
            current_dbz,
            frozen.nowcast_config,
            max_displacement_yx=limits * step_span,
            grid_time_contract=frozen.grid_time_contract,
        )
        pair = _estimate_available_pair(
            clean,
            frozen.observed_mask,
            linear,
            previous,
            current,
            frozen.nowcast_config,
            frozen.grid_time_contract,
        )
        peaks.append(peak)
        interiors.append(interior)
        available.append(pair is not None)
        growth_available.append(pair is not None and pair[1].available)
    return P0TendencyBranchSignature(
        pair_spans=pair_spans,
        motion_pair_spans=estimate.motion_pair_spans,
        growth_pair_spans=estimate.growth_pair_spans,
        integer_peak_yx_by_pair=tuple(peaks),
        peak_is_search_interior_by_pair=tuple(interiors),
        pair_available_by_span=tuple(available),
        growth_evidence_available_by_span=tuple(growth_available),
        motion_remap_cells=(
            freeze_remap_cell(estimate.displacement_yx),
            freeze_remap_cell(2.0 * estimate.displacement_yx),
        ),
        motion_selection=estimate.motion_pair_selection,
        growth_selection=estimate.growth_pair_selection,
        motion_conflict=estimate.motion_pair_conflict,
        growth_conflict=estimate.growth_pair_conflict,
    )


def _require_local_perturbation(
    name: str,
    values: Tensor,
    maximum_absolute_value: float,
) -> None:
    if bool(torch.any(torch.abs(values) > maximum_absolute_value)):
        raise ValueError(
            f"{name} perturbation exceeds the local first-order limit"
        )


def _initial_background_perturbation(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
) -> Tensor:
    values = perturbation.initial_background_dbz
    return torch.zeros_like(observations.dbz) if values is None else values


def _baseline_dynamics_perturbation(
    perturbation: VariationalObservationPerturbation,
    observations: AnalysisObservations,
) -> Tensor:
    values = perturbation.baseline_dynamics_dbz
    return torch.zeros_like(observations.dbz) if values is None else values


def _validate_variational_fso_lineage(
    result: ForecastResult,
    analysis: AnalysisResult | P1LinearizationState,
    linearization: AnalysisLinearization,
) -> None:
    frozen = linearization.frozen
    observations = linearization.observations
    validate_analysis_linearization_content(
        analysis.control,
        linearization,
    )
    if linearization.forecast_run_digest != result.forecast_run_digest:
        raise ValueError("P1 linearization forecast run mismatch")
    if frozen.nowcast_config != result.run.config:
        raise ValueError("P1 linearization nowcast config mismatch")
    config_digest = dataclass_digest(frozen.analysis_config)
    if config_digest != result.run.analysis_config_digest:
        raise ValueError("P1 linearization analysis config mismatch")
    if (
        frozen.neural_prior_application_digest != result.run.prior_application_digest
        or frozen.neural_prior_dependency != result.run.prior_dependency
    ):
        raise ValueError("P1 linearization neural-prior lineage mismatch")
    input_digest = json_digest(
        {
            "version": "p1-analysis-input-v2",
            "analysis_config_digest": config_digest,
            "observation_std_dbz": tensor_digest(observations.std_dbz),
            "quality_weight": tensor_digest(observations.quality_weight),
            "neural_prior_application_digest": (
                result.run.prior_application_digest
            ),
        }
    )
    if input_digest != result.run.analysis_input_digest:
        raise ValueError("P1 linearization input lineage mismatch")
    if not torch.equal(
        analysis.active_field_index,
        frozen.active_field_index,
    ):
        raise ValueError("P1 linearization active controls mismatch")
    trajectory = _analysis_trajectory(analysis.control, frozen)
    state_values = (
        (analysis.state.echo_linear, trajectory.frames_linear[-1]),
        (analysis.state.displacement_yx, trajectory.displacement_yx),
        (
            analysis.state.log_growth_per_step,
            trajectory.log_growth_per_step,
        ),
    )
    if any(
        not torch.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=result.run.config.contract_absolute_tolerance,
        )
        for actual, expected in state_values
    ):
        raise ValueError("P1 linearization does not reproduce the analysis")
    stationarity = _linearization_stationarity(
        analysis.control,
        observations,
        frozen,
    )
    stored_stationarity = (
        ("residual norm", linearization.residual_norm, stationarity.residual_norm),
        ("gradient norm", linearization.gradient_norm, stationarity.gradient_norm),
        (
            "field gradient RMS",
            linearization.field_gradient_rms,
            stationarity.field_gradient_rms,
        ),
        (
            "field gradient maximum",
            linearization.field_gradient_max,
            stationarity.field_gradient_max,
        ),
        (
            "dynamics gradient maximum",
            linearization.dynamics_gradient_max,
            stationarity.dynamics_gradient_max,
        ),
        (
            "relative stationarity",
            linearization.relative_stationarity,
            stationarity.relative_stationarity,
        ),
    )
    tolerance = 64.0 * torch.finfo(analysis.control.dtype).eps
    for name, stored, actual in stored_stationarity:
        if not (
            math.isfinite(stored)
            and math.isfinite(actual)
            and math.isclose(
                stored,
                actual,
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
        ):
            raise ValueError(f"P1 linearization {name} mismatch")
    analysis_stationarity = (
        analysis.linearization_residual_norm,
        analysis.linearization_gradient_norm,
        analysis.linearization_field_gradient_rms,
        analysis.linearization_field_gradient_max,
        analysis.linearization_dynamics_gradient_max,
        analysis.linearization_relative_stationarity,
        analysis.robust_gradient_norm,
        analysis.robust_field_gradient_rms,
        analysis.robust_field_gradient_max,
        analysis.robust_dynamics_gradient_max,
        analysis.robust_relative_stationarity,
        analysis.irls_relative_weight_change,
        analysis.linearization_polish_iterations,
    )
    retained_stationarity = (
        linearization.residual_norm,
        linearization.gradient_norm,
        linearization.field_gradient_rms,
        linearization.field_gradient_max,
        linearization.dynamics_gradient_max,
        linearization.relative_stationarity,
        linearization.robust_gradient_norm,
        linearization.robust_field_gradient_rms,
        linearization.robust_field_gradient_max,
        linearization.robust_dynamics_gradient_max,
        linearization.robust_relative_stationarity,
        linearization.irls_relative_weight_change,
        linearization.polish_iterations,
    )
    if analysis_stationarity != retained_stationarity:
        raise ValueError("P1 analysis linearization diagnostics mismatch")
    if not _stationarity_is_acceptable(
        stationarity,
        block_tolerance=(
            frozen.analysis_config
            .final_linearization_relative_stationarity_tolerance
        ),
        field_max_tolerance=(
            frozen.analysis_config.final_field_gradient_max_tolerance
        ),
    ):
        raise ValueError("P1 final linearization is not stationary")
    robust = _robust_stationarity(
        analysis.control,
        observations,
        frozen,
    )
    refreshed = freeze_irls_weights(
        analysis.control,
        observations,
        frozen,
    )
    weight_change = _relative_irls_weight_change(frozen, refreshed)
    robust_diagnostics = (
        (
            "robust gradient norm",
            linearization.robust_gradient_norm,
            robust.gradient_norm,
        ),
        (
            "robust field gradient RMS",
            linearization.robust_field_gradient_rms,
            robust.field_gradient_rms,
        ),
        (
            "robust field gradient maximum",
            linearization.robust_field_gradient_max,
            robust.field_gradient_max,
        ),
        (
            "robust dynamics gradient maximum",
            linearization.robust_dynamics_gradient_max,
            robust.dynamics_gradient_max,
        ),
        (
            "robust relative stationarity",
            linearization.robust_relative_stationarity,
            robust.relative_stationarity,
        ),
        (
            "IRLS relative weight change",
            linearization.irls_relative_weight_change,
            weight_change,
        ),
    )
    for name, stored, actual in robust_diagnostics:
        if not math.isclose(
            stored,
            actual,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError(f"P1 linearization {name} mismatch")
    if (
        not _stationarity_is_acceptable(
            robust,
            block_tolerance=(
                frozen.analysis_config
                .final_robust_relative_stationarity_tolerance
            ),
            field_max_tolerance=(
                frozen.analysis_config.final_field_gradient_max_tolerance
            ),
        )
        or weight_change
        > frozen.analysis_config.final_irls_relative_weight_tolerance
    ):
        raise ValueError("P1 linearization is not a robust IRLS fixed point")


def forecast_metric(
    name: str,
    forecast_linear: Tensor,
    truth_linear: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> Tensor:
    """Evaluate one differentiable forecast metric."""

    if name == "log_echo_mse":
        floor = 10.0 ** (nowcast_config.min_dbz / 10.0)
        difference = torch.log(forecast_linear + floor) - torch.log(
            truth_linear + floor
        )
        return _masked_mean(difference.square(), valid)
    if name == "soft_fss_error_35":
        return _soft_fss_error(
            forecast_linear,
            truth_linear,
            valid,
            nowcast_config,
            sensitivity_config,
            grid_time_contract,
        )
    if name == "centroid_error":
        forecast_center = _soft_centroid(forecast_linear, valid)
        truth_center = _soft_centroid(truth_linear, valid)
        return torch.sum((forecast_center - truth_center).square())
    if name == "centroid_error_m2":
        if grid_time_contract is None:
            raise ValueError("centroid_error_m2 requires a grid contract")
        forecast_center = _soft_projected_centroid(
            forecast_linear,
            valid,
            grid_time_contract,
        )
        truth_center = _soft_projected_centroid(
            truth_linear,
            valid,
            grid_time_contract,
        )
        return torch.sum((forecast_center - truth_center).square())
    raise ValueError(f"unsupported metric: {name}")


def extract_context_features(
    latest_frame_dbz: Tensor,
    state: RadarState,
    metadata: ForecastMetadata,
    config: NowcastConfig,
    *,
    latest_observation_mask: Tensor,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> Tensor:
    """Extract a small auditable context vector for later retrieval."""

    latest_valid = (
        torch.isfinite(latest_frame_dbz) & latest_observation_mask
    )
    latest = torch.nan_to_num(
        latest_frame_dbz,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    active = latest_valid & (latest >= config.echo_threshold_dbz)
    strong = latest_valid & (latest >= 35.0)
    valid_values = latest[latest_valid]
    active_values = latest[active]
    if active_values.numel():
        q90 = torch.quantile(active_values, 0.9)
    else:
        q90 = latest.new_tensor(config.min_dbz)
    if valid_values.numel():
        latest_mean = valid_values.mean()
        latest_max = valid_values.max()
    else:
        latest_mean = latest.new_tensor(config.min_dbz)
        latest_max = latest.new_tensor(config.min_dbz)

    border_width = max(1, min(latest.shape) // 16)
    border = torch.zeros_like(active)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    active_count = active.sum().clamp_min(1)
    boundary_fraction = (active & border).sum() / active_count

    linear = dbz_to_echo(
        latest,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    center = torch.nan_to_num(
        _soft_centroid(linear, latest_valid),
        nan=0.0,
    )
    motion = state.displacement_yx
    valid_count = latest_valid.sum().clamp_min(1)
    support_fraction = metadata.source_support.to(latest).mean()
    tendency_observation = latest.new_tensor(
        float(metadata.tendency_source is TendencySource.OBSERVATION)
    )
    tendency_background = latest.new_tensor(
        float(metadata.tendency_source is TendencySource.BACKGROUND)
    )
    state_path_observation = latest.new_tensor(
        float(metadata.state_path_source is TendencySource.OBSERVATION)
    )
    state_path_background = latest.new_tensor(
        float(metadata.state_path_source is TendencySource.BACKGROUND)
    )
    pair_selection_features = tuple(
        latest.new_tensor(
            float(metadata.motion_pair_selection is selection)
        )
        for selection in TendencyPairSelection
    ) + tuple(
        latest.new_tensor(
            float(metadata.growth_pair_selection is selection)
        )
        for selection in TendencyPairSelection
    )
    state_path_selection_features = tuple(
        latest.new_tensor(float(metadata.state_path_mode is selection))
        for selection in TendencyPairSelection
    )
    state_path_age_available = metadata.state_path_age_minutes is not None
    state_path_psr_available = math.isfinite(metadata.state_path_minimum_psr)
    growth_support_available = math.isfinite(
        metadata.minimum_growth_overlap_support
    )
    growth_area_available = math.isfinite(
        metadata.minimum_growth_overlap_area_km2
    )
    observation_path_age_available = (
        metadata.observation_path.age_minutes is not None
    )
    observation_path_psr_available = math.isfinite(
        metadata.observation_path.minimum_psr
    )
    background_path_age_available = (
        metadata.background_path.age_minutes is not None
    )
    background_path_psr_available = math.isfinite(
        metadata.background_path.minimum_psr
    )
    psr_available = metadata.tendency_pair_count > 0 and bool(
        torch.isfinite(metadata.minimum_phase_correlation_psr)
    )
    finite_minimum_psr = torch.nan_to_num(
        metadata.minimum_phase_correlation_psr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    disagreement_mps_available = bool(
        torch.isfinite(metadata.motion_disagreement_mps)
    )
    finite_disagreement_mps = torch.nan_to_num(
        metadata.motion_disagreement_mps,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    grid_available = grid_time_contract is not None
    if grid_time_contract is None:
        projected_velocity = latest.new_zeros(2)
        area_weighted_echo = latest.new_zeros(())
        grid_spacing = latest.new_zeros(2)
    else:
        projected_velocity = grid_time_contract.projected_velocity_xy(
            state.displacement_yx,
            config.interval_minutes,
        ).to(latest)
        area_weighted_echo = linear[latest_valid].sum() * (
            grid_time_contract.cell_area_m2 / 1.0e6
        )
        grid_spacing = latest.new_tensor(
            (grid_time_contract.dx_m, grid_time_contract.dy_m)
        )
    return torch.stack(
        (
            motion[0],
            motion[1],
            torch.linalg.vector_norm(motion),
            state.log_growth_per_step,
            metadata.motion_disagreement_px,
            metadata.growth_disagreement,
            latest.new_tensor(float(metadata.motion_pair_conflict)),
            latest.new_tensor(float(metadata.growth_pair_conflict)),
            latest.new_tensor(float(metadata.tendency_pair_count)),
            tendency_observation,
            tendency_background,
            latest.new_tensor(float(metadata.state_path_pair_count)),
            state_path_observation,
            state_path_background,
            latest.new_tensor(float(metadata.state_path_conflict)),
            latest.new_tensor(float(metadata.state_path_extrapolated)),
            latest.new_tensor(float(state_path_age_available)),
            latest.new_tensor(metadata.state_path_age_minutes or 0.0),
            latest.new_tensor(float(state_path_psr_available)),
            latest.new_tensor(
                math.log1p(metadata.state_path_minimum_psr)
                if state_path_psr_available
                else 0.0
            ),
            latest.new_tensor(float(growth_support_available)),
            latest.new_tensor(
                math.log1p(metadata.minimum_growth_overlap_support)
                if growth_support_available
                else 0.0
            ),
            latest.new_tensor(float(growth_area_available)),
            latest.new_tensor(
                math.log1p(metadata.minimum_growth_overlap_area_km2)
                if growth_area_available
                else 0.0
            ),
            support_fraction,
            latest.new_tensor(metadata.background_contribution_fraction),
            metadata.coverage_by_frame[-1].to(latest),
            latest_mean,
            latest_max,
            q90,
            active.sum().to(latest.dtype) / valid_count,
            strong.sum().to(latest.dtype) / valid_count,
            boundary_fraction.to(latest.dtype),
            center[0],
            center[1],
            torch.log1p(linear[latest_valid].sum()),
            *pair_selection_features,
            *state_path_selection_features,
            latest.new_tensor(float(psr_available)),
            torch.log1p(finite_minimum_psr).to(latest),
            latest.new_tensor(float(grid_available)),
            projected_velocity[0],
            projected_velocity[1],
            torch.linalg.vector_norm(projected_velocity),
            latest.new_tensor(float(disagreement_mps_available)),
            finite_disagreement_mps.to(latest),
            latest.new_tensor(float(grid_available)),
            torch.log1p(area_weighted_echo),
            latest.new_tensor(float(grid_available)),
            grid_spacing[0],
            grid_spacing[1],
            latest.new_tensor(float(metadata.observation_path.pair_count)),
            latest.new_tensor(float(metadata.observation_path.conflict)),
            latest.new_tensor(float(metadata.observation_path.extrapolated)),
            latest.new_tensor(float(observation_path_age_available)),
            latest.new_tensor(metadata.observation_path.age_minutes or 0.0),
            latest.new_tensor(float(observation_path_psr_available)),
            latest.new_tensor(
                math.log1p(metadata.observation_path.minimum_psr)
                if observation_path_psr_available
                else 0.0
            ),
            latest.new_tensor(float(metadata.background_path.pair_count)),
            latest.new_tensor(float(metadata.background_path.conflict)),
            latest.new_tensor(float(metadata.background_path.extrapolated)),
            latest.new_tensor(float(background_path_age_available)),
            latest.new_tensor(metadata.background_path.age_minutes or 0.0),
            latest.new_tensor(float(background_path_psr_available)),
            latest.new_tensor(
                math.log1p(metadata.background_path.minimum_psr)
                if background_path_psr_available
                else 0.0
            ),
        )
    ).detach()


def _state_from_control(
    template: RadarState,
    control: Tensor,
    echo: Tensor,
) -> RadarState:
    return RadarState(
        echo_linear=echo,
        displacement_yx=control[:2],
        log_growth_per_step=control[2],
    )


def _freeze_output_cap(
    forecast: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor, Tensor]:
    """Apply the issued dBZ cap and freeze its nominal active set."""

    maximum = _maximum_linear_echo(forecast, config)
    active = (forecast < maximum).detach()
    return torch.where(active, forecast, maximum), active


def _apply_output_cap(
    forecast: Tensor,
    active: Tensor,
    config: NowcastConfig,
) -> Tensor:
    maximum = _maximum_linear_echo(forecast, config)
    return torch.where(active, forecast, maximum)


def _maximum_linear_echo(reference: Tensor, config: NowcastConfig) -> Tensor:
    floor = 10.0 ** (config.min_dbz / 10.0)
    maximum = 10.0 ** (config.max_dbz / 10.0) - floor
    return reference.new_tensor(maximum)


def _metric_has_support(
    name: str,
    forecast: Tensor,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> bool:
    """Decide metric support once, before any differentiation."""

    if not bool(torch.any(valid)):
        return False
    if name == "soft_fss_error_35":
        floor = 10.0 ** (nowcast_config.min_dbz / 10.0)
        truth_dbz = 10.0 * torch.log10(truth + floor)
        truth_event = torch.sigmoid(
            (truth_dbz - 35.0)
            / sensitivity_config.soft_fss_temperature_dbz
        )
        truth_mass = torch.sum(truth_event * valid.to(truth.dtype))
        return bool(truth_mass >= sensitivity_config.minimum_fss_truth_mass)
    if name in ("centroid_error", "centroid_error_m2"):
        forecast_mass = torch.sum(
            torch.log1p(forecast) * valid.to(forecast.dtype)
        )
        truth_mass = torch.sum(torch.log1p(truth) * valid.to(truth.dtype))
        return bool(
            (forecast_mass > sensitivity_config.epsilon)
            & (truth_mass > sensitivity_config.epsilon)
        )
    return True


def _soft_fss_error(
    forecast: Tensor,
    truth: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> Tensor:
    floor = 10.0 ** (nowcast_config.min_dbz / 10.0)
    forecast_dbz = 10.0 * torch.log10(forecast + floor)
    truth_dbz = 10.0 * torch.log10(truth + floor)
    temperature = sensitivity_config.soft_fss_temperature_dbz
    forecast_event = torch.sigmoid((forecast_dbz - 35.0) / temperature)
    truth_event = torch.sigmoid((truth_dbz - 35.0) / temperature)

    valid_float = valid.to(forecast.dtype)
    local_valid = _soft_fss_average(
        valid_float,
        sensitivity_config,
        grid_time_contract,
    )
    denominator = local_valid.clamp_min(sensitivity_config.epsilon)
    forecast_fraction = _soft_fss_average(
        forecast_event * valid_float,
        sensitivity_config,
        grid_time_contract,
    ) / denominator
    truth_fraction = _soft_fss_average(
        truth_event * valid_float,
        sensitivity_config,
        grid_time_contract,
    ) / denominator
    numerator = _weighted_mean(
        (forecast_fraction - truth_fraction).square(),
        local_valid,
        sensitivity_config.epsilon,
    )
    reference = _weighted_mean(
        forecast_fraction.square() + truth_fraction.square(),
        local_valid,
        sensitivity_config.epsilon,
    )
    return numerator / (reference + sensitivity_config.epsilon)


def _soft_fss_average(
    values: Tensor,
    config: SensitivityConfig,
    grid: RadarGridTimeContract | None,
) -> Tensor:
    if config.soft_fss_window_m is None:
        window = config.soft_fss_window
        return F.avg_pool2d(
            values[None, None],
            window,
            stride=1,
            padding=window // 2,
        )[0, 0]
    if grid is None:
        raise ValueError("physical FSS requires a grid contract")
    return _affine_footprint_average(
        values,
        grid,
        0.5 * config.soft_fss_window_m,
    )


def _affine_footprint_average(
    values: Tensor,
    grid: RadarGridTimeContract,
    radius_m: float,
) -> Tensor:
    """Average over the exact projected-distance footprint of one grid."""

    radius_y, radius_x = grid.pixel_radius_yx(radius_m)
    offsets = grid.pixel_offsets_within_distance(
        radius_m,
        maximum_radius_yx=(radius_y, radius_x),
    )
    kernel = values.new_zeros((2 * radius_y + 1, 2 * radius_x + 1))
    for row, column in offsets:
        kernel[row + radius_y, column + radius_x] = 1.0
    kernel /= len(offsets)
    return F.conv2d(
        values[None, None],
        kernel[None, None],
        padding=(radius_y, radius_x),
    )[0, 0]


def _weighted_mean(values: Tensor, weights: Tensor, epsilon: float) -> Tensor:
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(epsilon)


def _soft_centroid(echo: Tensor, valid: Tensor) -> Tensor:
    height, width = echo.shape
    y = torch.linspace(-1.0, 1.0, height, dtype=echo.dtype, device=echo.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=echo.dtype, device=echo.device)
    weights = torch.log1p(echo) * valid.to(echo.dtype)
    total = weights.sum()
    safe_total = total.clamp_min(torch.finfo(echo.dtype).eps)
    center = torch.stack(
        (
            torch.sum(weights * y[:, None]) / safe_total,
            torch.sum(weights * x[None, :]) / safe_total,
        )
    )
    return torch.where(
        total > torch.finfo(echo.dtype).eps,
        center,
        torch.full_like(center, float("nan")),
    )


def _soft_projected_centroid(
    echo: Tensor,
    valid: Tensor,
    grid: RadarGridTimeContract,
) -> Tensor:
    """Return the echo centroid in projected metres; origin cancels in errors."""

    height, width = echo.shape
    row = torch.arange(height, dtype=echo.dtype, device=echo.device)
    column = torch.arange(width, dtype=echo.dtype, device=echo.device)
    weights = torch.log1p(echo) * valid.to(echo.dtype)
    total = weights.sum()
    safe_total = total.clamp_min(torch.finfo(echo.dtype).eps)
    center_column_row = torch.stack(
        (
            torch.sum(weights * column[None, :]) / safe_total,
            torch.sum(weights * row[:, None]) / safe_total,
        )
    )
    assert grid.pixel_to_projected_matrix_m is not None
    matrix = echo.new_tensor(grid.pixel_to_projected_matrix_m)
    center = matrix @ center_column_row
    return torch.where(
        total > torch.finfo(echo.dtype).eps,
        center,
        torch.full_like(center, float("nan")),
    )


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    weights = valid.to(values.dtype)
    count = weights.sum()
    mean = torch.sum(values * weights) / count.clamp_min(
        torch.finfo(values.dtype).tiny
    )
    return torch.where(
        count > 0,
        mean,
        torch.full_like(mean, float("nan")),
    )


def _tile_l2(values: Tensor, tile_size: TileShape) -> Tensor:
    tiles = _as_tiles(values, tile_size)
    return torch.sqrt(torch.sum(tiles.square(), dim=(-1, -2)))


def _tile_sum(values: Tensor, tile_size: TileShape) -> Tensor:
    return torch.sum(_as_tiles(values, tile_size), dim=(-1, -2))


def _as_tiles(values: Tensor, tile_size: TileShape) -> Tensor:
    height, width = values.shape
    tile_height, tile_width = tile_size
    tile_rows = math.ceil(height / tile_height)
    tile_columns = math.ceil(width / tile_width)
    padded = F.pad(
        values,
        (
            0,
            tile_columns * tile_width - width,
            0,
            tile_rows * tile_height - height,
        ),
    )
    return padded.reshape(
        tile_rows,
        tile_height,
        tile_columns,
        tile_width,
    ).permute(0, 2, 1, 3)


def _frozen_observation(
    latest_frame: Tensor,
    accepted: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> tuple[Tensor, Tensor]:
    finite = torch.isfinite(latest_frame)
    clean = torch.nan_to_num(
        latest_frame,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    ).clamp(nowcast_config.min_dbz, nowcast_config.max_dbz)

    margin = sensitivity_config.active_margin_dbz
    latest_active = (
        finite
        & accepted
        & (clean > nowcast_config.min_dbz + margin)
        & (clean < nowcast_config.max_dbz - margin)
    )
    return clean.detach(), latest_active.detach()


def _active_dbz_to_echo(
    candidate_dbz: Tensor,
    nominal_dbz: Tensor,
    nominal_echo: Tensor,
    active: Tensor,
    config: NowcastConfig,
) -> Tensor:
    """Apply dBZ perturbations only where the frozen active set permits."""

    safe_dbz = torch.where(
        active,
        candidate_dbz,
        torch.zeros_like(candidate_dbz),
    )
    nominal_safe_dbz = torch.where(
        active,
        nominal_dbz,
        torch.zeros_like(nominal_dbz),
    )
    candidate_echo = dbz_to_echo(
        safe_dbz,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    nominal_active_echo = dbz_to_echo(
        nominal_safe_dbz,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    ).detach()
    perturbation = torch.where(
        active,
        candidate_echo - nominal_active_echo,
        torch.zeros_like(candidate_echo),
    )
    return nominal_echo.detach() + perturbation


def _observation_std(
    value: float | Tensor | None,
    frames: Tensor,
    epsilon: float,
) -> tuple[Tensor, bool]:
    if value is None:
        return torch.ones_like(frames), False
    if isinstance(value, (int, float)):
        result = torch.full_like(frames, float(value))
    else:
        result = value.to(dtype=frames.dtype, device=frames.device)
        if result.ndim == 0:
            result = torch.full_like(frames, float(result))
        elif result.shape != frames.shape:
            raise ValueError("observation_std_dbz must match frames shape")
    if not bool(torch.all(torch.isfinite(result))) or bool(
        torch.any(result <= epsilon)
    ):
        raise ValueError("observation_std_dbz must be finite and positive")
    return result, True


def _dbz_innovation(
    latest_frame: Tensor,
    background: Tensor | None,
    accepted: Tensor,
    config: NowcastConfig,
) -> tuple[Tensor | None, Tensor | None]:
    if background is None:
        return None, None
    valid = (
        torch.isfinite(latest_frame)
        & torch.isfinite(background)
        & accepted
    )
    clean_frame = torch.nan_to_num(
        latest_frame,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    clean_background = torch.nan_to_num(
        background,
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    innovation = torch.where(
        valid,
        clean_frame - clean_background,
        torch.full_like(latest_frame, float("nan")),
    )
    return innovation.detach(), valid.detach()


def _full_map_indices(
    selected_minutes: tuple[int, ...],
    all_minutes: tuple[int, ...],
) -> tuple[int, ...]:
    unknown = set(selected_minutes) - set(all_minutes)
    if unknown:
        raise ValueError(f"full-map leads outside forecast horizon: {sorted(unknown)}")
    return tuple(all_minutes.index(value) for value in selected_minutes)


def _metric_evidence_ratios(
    sensitivity_weight: Tensor,
    source_support: Tensor,
    forecast_confidence: Tensor,
    observation_source_support: Tensor,
    observation_verified_confidence: Tensor,
    background_verified_confidence: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    denominator = (sensitivity_weight * source_support).sum()
    if float(denominator) <= epsilon:
        return None

    def weighted_fraction(evidence: Tensor) -> Tensor:
        return (sensitivity_weight * evidence).sum() / denominator

    return (
        weighted_fraction(forecast_confidence),
        weighted_fraction(observation_source_support),
        weighted_fraction(observation_verified_confidence),
        weighted_fraction(background_verified_confidence),
    )


def _trust_components(
    template: RadarState,
    metadata: ForecastMetadata,
    control: Tensor,
    echo: Tensor,
    truth: Tensor,
    valid: Tensor,
    gradients: Tensor,
    metric_available: Tensor,
    cap_masks: Tensor,
    observation_verified_evidence_by_metric: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
    grid_time_contract: RadarGridTimeContract | None,
) -> dict[str, float]:
    verification_quality = valid.to(echo.dtype).mean().clamp(0.0, 1.0)
    support_quality = metric_available.to(echo.dtype).mean()
    conflict_count = int(metadata.motion_pair_conflict) + int(
        metadata.growth_pair_conflict
    )
    pair_consistency_quality = (
        sensitivity_config.pair_conflict_trust_penalty**conflict_count
    )
    evidence_available = metric_available & torch.isfinite(
        observation_verified_evidence_by_metric
    )
    if bool(torch.any(evidence_available)):
        observation_verified_evidence_quality = float(
            observation_verified_evidence_by_metric[evidence_available]
            .mean()
            .clamp(0.0, 1.0)
        )
    else:
        observation_verified_evidence_quality = 0.0
    if not bool(torch.any(metric_available)):
        return {
            "linearity": 0.0,
            "verification": float(verification_quality),
            "metric_support": 0.0,
            "pair_consistency": pair_consistency_quality,
            "observation_verified_evidence": (
                observation_verified_evidence_quality
            ),
        }

    delta = control.new_tensor(sensitivity_config.linearity_delta)
    predicted_change = torch.sum(gradients[metric_available].mean(dim=0) * delta)

    def aggregate(candidate_control: Tensor) -> Tensor:
        candidate_state = _state_from_control(
            template,
            candidate_control,
            echo,
        )
        scores: list[Tensor] = []
        for lead_index in range(nowcast_config.forecast_steps):
            latent_forecast = forecast_linear_at_step(
                candidate_state,
                lead_index + 1,
                nowcast_config,
            )
            forecast = _apply_output_cap(
                latent_forecast,
                cap_masks[lead_index],
                nowcast_config,
            )
            for metric_index, name in enumerate(
                sensitivity_config.metric_names
            ):
                if not bool(metric_available[lead_index, metric_index]):
                    continue
                scores.append(
                    forecast_metric(
                        name,
                        forecast,
                        truth[lead_index],
                        valid[lead_index],
                        nowcast_config,
                        sensitivity_config,
                        grid_time_contract,
                    )
                )
        return torch.stack(scores).mean()

    actual_change = aggregate(control + delta) - aggregate(control)
    linearity_error = torch.abs(actual_change - predicted_change) / (
        torch.abs(actual_change)
        + torch.abs(predicted_change)
        + sensitivity_config.epsilon
    )
    linearity_quality = torch.nan_to_num(
        torch.exp(-linearity_error / 0.25),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    return {
        "linearity": float(linearity_quality.detach()),
        "verification": float(verification_quality.detach()),
        "metric_support": float(support_quality.detach()),
        "pair_consistency": pair_consistency_quality,
        "observation_verified_evidence": (
            observation_verified_evidence_quality
        ),
    }


def _validate_inputs(
    latest_frame: Tensor,
    verification: Tensor,
    state: RadarState,
    config: NowcastConfig,
    background: Tensor | None,
) -> None:
    if latest_frame.ndim != 2:
        raise ValueError("latest_frame_dbz must have shape [height, width]")
    expected = (config.forecast_steps, *latest_frame.shape)
    if tuple(verification.shape) != expected:
        raise ValueError(f"verification_frames_dbz must have shape {expected}")
    if tuple(state.echo_linear.shape) != tuple(latest_frame.shape):
        raise ValueError("state grid must match frame grid")
    if background is not None and background.shape != latest_frame.shape:
        raise ValueError(
            "latest_background_dbz must match latest_frame_dbz shape"
        )
    if (
        not latest_frame.is_floating_point()
        or not verification.is_floating_point()
    ):
        raise TypeError(
            "latest frame and verification must be floating-point tensors"
        )
    if background is not None and not background.is_floating_point():
        raise TypeError("latest_background_dbz must be floating-point")
    if state.displacement_yx.shape != (2,):
        raise ValueError("state displacement must have shape [2]")
    if state.log_growth_per_step.ndim != 0:
        raise ValueError("state log growth must be scalar")
    if latest_frame.device != verification.device:
        raise ValueError(
            "latest frame and verification must use the same device"
        )
    state_tensors = (
        state.echo_linear,
        state.displacement_yx,
        state.log_growth_per_step,
    )
    if any(tensor.device != latest_frame.device for tensor in state_tensors):
        raise ValueError("state and latest frame must use the same device")
    if any(not tensor.is_floating_point() for tensor in state_tensors):
        raise TypeError("state tensors must be floating-point")
    if background is not None and background.device != latest_frame.device:
        raise ValueError(
            "background and latest frame must use the same device"
        )
