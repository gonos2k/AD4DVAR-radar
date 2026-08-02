"""Forecast sensitivities that can be stored as conditional experience."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ._digest import dataclass_digest
from .nowcast import (
    DataStatus,
    ForecastMetadata,
    ForecastResult,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    TendencyPairSelection,
    TendencySource,
    _forecast_linear_at_step_core,
    forecast_linear_at_step,
)
from .physics import dbz_to_echo, freeze_remap_cell


SUPPORTED_METRICS = (
    "log_echo_mse",
    "soft_fss_error_35",
    "centroid_error",
)

CONTEXT_FEATURE_NAMES = (
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


@dataclass(frozen=True)
class SensitivityConfig:
    """Fixed metric and compression choices for one sensitivity contract."""

    metric_names: tuple[str, ...] = SUPPORTED_METRICS
    full_map_lead_minutes: tuple[int, ...] = (30, 60, 120, 180)
    tile_size: int = 16
    soft_fss_temperature_dbz: float = 2.0
    soft_fss_window: int = 9
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
class SensitivitySnapshot:
    forecast_run_digest: str
    nowcast_config_digest: str
    sensitivity_config_digest: str
    grid_time_contract_digest: str | None
    metric_names: tuple[str, ...]
    lead_minutes: tuple[int, ...]
    full_map_lead_minutes: tuple[int, ...]
    tile_size: int
    context_feature_names: tuple[str, ...]
    context_features: Tensor
    analysis_control: Tensor
    forecast_scores: Tensor
    metric_available: Tensor
    control_sensitivity: Tensor
    forecast_sensitivity: Tensor
    forecast_cap_active_mask: Tensor
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


def compute_sensitivity_snapshot(
    latest_frame_dbz: Tensor,
    result: ForecastResult,
    verification_frames_dbz: Tensor,
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
    """

    sensitivity_config = sensitivity_config or SensitivityConfig()
    nowcast_config = result.run.config
    result.validate_issuance()
    result.run.validate_latest_frame(latest_frame_dbz)
    result.run.validate_latest_background(latest_background_dbz)
    latest_observation_mask = result.run.latest_observation_mask
    state = result.state
    metadata = result.metadata
    if metadata.data_status is DataStatus.UNAVAILABLE:
        raise ValueError("sensitivity is undefined for an unissued forecast")
    if metadata.provenance != "p0_support_merged":
        raise ValueError("M0 direct sensitivity requires a P0 state")
    if (
        2 * sensitivity_config.active_margin_dbz
        >= nowcast_config.max_dbz - nowcast_config.min_dbz
    ):
        raise ValueError("active_margin_dbz leaves no differentiable range")
    _validate_inputs(
        latest_frame_dbz,
        verification_frames_dbz,
        state,
        nowcast_config,
        latest_background_dbz,
    )

    height, width = state.echo_linear.shape
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
    tile_rows = math.ceil(height / sensitivity_config.tile_size)
    tile_columns = math.ceil(width / sensitivity_config.tile_size)

    clean_verification = torch.nan_to_num(
        verification_frames_dbz,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    )
    verification_valid = torch.isfinite(verification_frames_dbz)
    issued_valid = torch.isfinite(result.forecast_dbz)
    if issued_valid.shape != verification_valid.shape:
        raise ValueError("issued forecast must match verification shape")
    if result.valid_mask.shape != verification_valid.shape:
        raise ValueError("forecast valid_mask must match verification shape")
    if not torch.equal(result.valid_mask, issued_valid):
        raise ValueError("forecast valid_mask must match issued finite values")
    verification_valid = verification_valid & issued_valid
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
        valid = verification_valid[lead_index]
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

            forecast_scores[lead_index, metric_index] = score.detach()
            control_sensitivity[lead_index, metric_index] = (
                control_gradient.detach()
            )
            direct_norm[lead_index, metric_index] = torch.linalg.vector_norm(
                direct_gradient.detach()
            )
            tile_direct_norm[lead_index, metric_index] = _tile_l2(
                direct_gradient.detach(),
                sensitivity_config.tile_size,
            )
            if tile_whitened_norm is not None:
                tile_whitened_norm[lead_index, metric_index] = _tile_l2(
                    whitened_gradient.detach(),
                    sensitivity_config.tile_size,
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
                    sensitivity_config.tile_size,
                )
                tile_impact[lead_index, metric_index] = tiles
                observation_impact[lead_index, metric_index] = tiles.sum()

    has_metric_support = bool(torch.any(metric_available))
    if not has_metric_support:
        observation_impact = None
        tile_impact = None

    reward = None
    if baseline_scores is not None:
        baseline_scores = baseline_scores.to(
            dtype=echo.dtype,
            device=echo.device,
        )
        if baseline_scores.shape != forecast_scores.shape:
            raise ValueError("baseline_scores must match forecast_scores shape")
        if not bool(torch.all(torch.isfinite(baseline_scores))) or bool(
            torch.any(baseline_scores < 0)
        ):
            raise ValueError("baseline_scores must be finite and non-negative")
        if observation_impact is not None:
            reward = -observation_impact / (
                baseline_scores + sensitivity_config.epsilon
            )

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
        nowcast_config,
        sensitivity_config,
    )
    trust_score = math.prod(trust_components.values())

    return SensitivitySnapshot(
        forecast_run_digest=result.forecast_run_digest,
        nowcast_config_digest=nowcast_config.digest,
        sensitivity_config_digest=sensitivity_config.digest,
        grid_time_contract_digest=result.run.grid_time_contract_digest,
        metric_names=sensitivity_config.metric_names,
        lead_minutes=lead_minutes,
        full_map_lead_minutes=sensitivity_config.full_map_lead_minutes,
        tile_size=sensitivity_config.tile_size,
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
        direct=DirectSensitivity(
            maps=direct_maps,
            norm=direct_norm,
            tile_norm=tile_direct_norm,
            whitened_tile_norm=tile_whitened_norm,
            impact=observation_impact,
            tile_impact=tile_impact,
            reward=reward,
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
        baseline_scores=(
            baseline_scores.detach()
            if baseline_scores is not None
            else None
        ),
        reward_epsilon=sensitivity_config.epsilon,
        trust_components=trust_components,
        trust_score=trust_score,
    )


def compute_sensitivity_snapshot_from_run(
    result: ForecastResult,
    verification_frames_dbz: Tensor,
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


def forecast_metric(
    name: str,
    forecast_linear: Tensor,
    truth_linear: Tensor,
    valid: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
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
        )
    if name == "centroid_error":
        forecast_center = _soft_centroid(forecast_linear, valid)
        truth_center = _soft_centroid(truth_linear, valid)
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
    if name == "centroid_error":
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
) -> Tensor:
    floor = 10.0 ** (nowcast_config.min_dbz / 10.0)
    forecast_dbz = 10.0 * torch.log10(forecast + floor)
    truth_dbz = 10.0 * torch.log10(truth + floor)
    temperature = sensitivity_config.soft_fss_temperature_dbz
    forecast_event = torch.sigmoid((forecast_dbz - 35.0) / temperature)
    truth_event = torch.sigmoid((truth_dbz - 35.0) / temperature)

    window = sensitivity_config.soft_fss_window
    padding = window // 2
    valid_float = valid.to(forecast.dtype)
    local_valid = F.avg_pool2d(
        valid_float[None, None],
        window,
        stride=1,
        padding=padding,
    )[0, 0]
    denominator = local_valid.clamp_min(sensitivity_config.epsilon)
    forecast_fraction = F.avg_pool2d(
        (forecast_event * valid_float)[None, None],
        window,
        stride=1,
        padding=padding,
    )[0, 0] / denominator
    truth_fraction = F.avg_pool2d(
        (truth_event * valid_float)[None, None],
        window,
        stride=1,
        padding=padding,
    )[0, 0] / denominator
    local_mask = local_valid > 0.0
    numerator = _masked_mean(
        (forecast_fraction - truth_fraction).square(),
        local_mask,
    )
    reference = _masked_mean(
        forecast_fraction.square() + truth_fraction.square(),
        local_mask,
    )
    return numerator / (reference + sensitivity_config.epsilon)


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


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    weights = valid.to(values.dtype)
    count = weights.sum()
    mean = torch.sum(values * weights) / count.clamp_min(1.0)
    return torch.where(
        count > 0,
        mean,
        torch.full_like(mean, float("nan")),
    )


def _tile_l2(values: Tensor, tile_size: int) -> Tensor:
    tiles = _as_tiles(values, tile_size)
    return torch.sqrt(torch.sum(tiles.square(), dim=(-1, -2)))


def _tile_sum(values: Tensor, tile_size: int) -> Tensor:
    return torch.sum(_as_tiles(values, tile_size), dim=(-1, -2))


def _as_tiles(values: Tensor, tile_size: int) -> Tensor:
    height, width = values.shape
    tile_rows = math.ceil(height / tile_size)
    tile_columns = math.ceil(width / tile_size)
    padded = F.pad(
        values,
        (0, tile_columns * tile_size - width, 0, tile_rows * tile_size - height),
    )
    return padded.reshape(
        tile_rows,
        tile_size,
        tile_columns,
        tile_size,
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
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> dict[str, float]:
    verification_quality = valid.to(echo.dtype).mean().clamp(0.0, 1.0)
    support_quality = metric_available.to(echo.dtype).mean()
    conflict_count = int(metadata.motion_pair_conflict) + int(
        metadata.growth_pair_conflict
    )
    pair_consistency_quality = (
        sensitivity_config.pair_conflict_trust_penalty**conflict_count
    )
    if not bool(torch.any(metric_available)):
        return {
            "linearity": 0.0,
            "verification": float(verification_quality),
            "metric_support": 0.0,
            "pair_consistency": pair_consistency_quality,
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
