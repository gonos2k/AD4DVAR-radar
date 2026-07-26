"""Forecast sensitivities that can be stored as conditional experience."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .nowcast import (
    NowcastConfig,
    RadarState,
    dbz_to_linear,
    forecast_linear_at_step,
)


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
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_echo_mass",
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
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class SensitivitySnapshot:
    """M0 sensitivity result for one completed forecast cycle."""

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
    direct_observation_sensitivity: Tensor
    direct_observation_sensitivity_norm: Tensor
    tile_direct_sensitivity_norm: Tensor
    tile_whitened_direct_sensitivity_norm: Tensor
    direct_observation_impact: Tensor
    tile_direct_observation_impact: Tensor
    direct_normalized_reward: Tensor
    latest_sensitivity_mask: Tensor
    observation_std_dbz: Tensor
    observation_innovation_dbz: Tensor
    observation_innovation_mask: Tensor
    baseline_scores: Tensor
    reward_epsilon: float
    trust_components: dict[str, float]
    trust_score: float
    impact_available: bool
    reward_available: bool
    whitened_tile_norm_available: bool
    indirect_observation_sensitivity_available: bool = False
    promotion_eligible: bool = False


def compute_sensitivity_snapshot(
    frames_dbz: Tensor,
    state: RadarState,
    verification_frames_dbz: Tensor,
    *,
    nowcast_config: NowcastConfig | None = None,
    sensitivity_config: SensitivityConfig | None = None,
    background_frames_dbz: Tensor | None = None,
    observation_std_dbz: float | Tensor | None = None,
    baseline_scores: Tensor | None = None,
    qc_mask: Tensor | None = None,
) -> SensitivitySnapshot:
    """Compute M0 forecast/control/direct-observation sensitivities.

    The direct observation sensitivity is with respect to latest-frame dBZ
    inside a frozen active set. The FFT motion analysis is intentionally
    excluded: its discrete peak selection has no valid local derivative.
    """

    nowcast_config = nowcast_config or NowcastConfig()
    sensitivity_config = sensitivity_config or SensitivityConfig()
    if state.provenance != "p0_fft_latest":
        raise ValueError(
            "M0 direct sensitivity requires a P0 latest-frame state"
        )
    if (
        2 * sensitivity_config.active_margin_dbz
        >= nowcast_config.max_dbz - nowcast_config.min_dbz
    ):
        raise ValueError("active_margin_dbz leaves no differentiable range")
    _validate_inputs(
        frames_dbz,
        verification_frames_dbz,
        state,
        nowcast_config,
        background_frames_dbz,
        qc_mask,
    )

    height, width = state.echo_amplitude.shape
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
    truth_linear = dbz_to_linear(clean_verification, nowcast_config)
    control = torch.cat(
        (state.displacement_yx, state.log_growth_per_step.reshape(1))
    )
    amplitude = state.echo_amplitude
    clean_frames, latest_active = _frozen_observations(
        frames_dbz,
        nowcast_config,
        sensitivity_config,
        qc_mask,
    )
    observation_std, whitening_available = _observation_std(
        observation_std_dbz,
        frames_dbz,
        sensitivity_config.epsilon,
    )

    score_shape = (lead_count, metric_count)
    forecast_scores = amplitude.new_full(score_shape, float("nan"))
    metric_available = torch.zeros(
        score_shape,
        dtype=torch.bool,
        device=amplitude.device,
    )
    control_sensitivity = amplitude.new_full(
        (lead_count, metric_count, 3),
        float("nan"),
    )
    direct_norm = amplitude.new_zeros((lead_count, metric_count, 3))
    tile_direct_norm = amplitude.new_zeros(
        (lead_count, metric_count, 3, tile_rows, tile_columns)
    )
    tile_shape = (lead_count, metric_count, 3, tile_rows, tile_columns)
    if whitening_available:
        tile_whitened_norm = amplitude.new_zeros(tile_shape)
    else:
        tile_whitened_norm = amplitude.new_full(tile_shape, float("nan"))
    selected_count = len(full_map_indices)
    forecast_maps = amplitude.new_full(
        (selected_count, metric_count, height, width),
        float("nan"),
    )
    direct_maps = amplitude.new_zeros(
        (selected_count, metric_count, 3, height, width)
    )
    selected_cap_masks = torch.zeros(
        (selected_count, height, width),
        dtype=torch.bool,
        device=amplitude.device,
    )
    all_cap_masks = torch.zeros(
        (lead_count, height, width),
        dtype=torch.bool,
        device=amplitude.device,
    )
    innovation, innovation_mask = _dbz_innovation(
        frames_dbz,
        background_frames_dbz,
        qc_mask,
        nowcast_config,
    )
    impact_input_available = (
        innovation is not None
        and bool(torch.any(innovation_mask[2] & latest_active))
    )
    if not impact_input_available:
        tile_impact = amplitude.new_full(
            (lead_count, metric_count, 3, tile_rows, tile_columns),
            float("nan"),
        )
        observation_impact = amplitude.new_full(
            (lead_count, metric_count, 3),
            float("nan"),
        )
    else:
        tile_impact = amplitude.new_zeros(
            (lead_count, metric_count, 3, tile_rows, tile_columns)
        )
        observation_impact = amplitude.new_zeros((lead_count, metric_count, 3))
    selected_position = {
        index: position for position, index in enumerate(full_map_indices)
    }

    for lead_index in range(lead_count):
        truth = truth_linear[lead_index]
        valid = verification_valid[lead_index]
        latent_prediction = forecast_linear_at_step(
            state,
            lead_index + 1,
            nowcast_config,
        )
        prediction, cap_active = _freeze_output_cap(
            latent_prediction,
            nowcast_config,
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
                direct_norm[lead_index, metric_index, 2] = float("nan")
                tile_direct_norm[lead_index, metric_index, 2] = float("nan")
                if whitening_available:
                    tile_whitened_norm[
                        lead_index, metric_index, 2
                    ] = float("nan")
                if lead_index in selected_position:
                    position = selected_position[lead_index]
                    direct_maps[position, metric_index, 2] = float("nan")
                if impact_input_available:
                    observation_impact[
                        lead_index, metric_index, 2
                    ] = float("nan")
                    tile_impact[lead_index, metric_index, 2] = float("nan")
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
                candidate_amplitude = _active_dbz_to_amplitude(
                    candidate_latest_dbz,
                    clean_frames[2],
                    amplitude,
                    latest_active,
                    nowcast_config,
                )
                candidate_state = _state_from_control(
                    state,
                    candidate_control,
                    candidate_amplitude,
                )
                candidate = forecast_linear_at_step(
                    candidate_state,
                    lead_index + 1,
                    nowcast_config,
                )
                return metric(_apply_output_cap(candidate, cap_active, nowcast_config))

            control_gradient, direct_gradient = torch.func.grad(
                score_from_state,
                argnums=(0, 1),
            )(control, clean_frames[2])
            forecast_gradient = torch.func.grad(metric)(prediction)
            whitened_gradient = direct_gradient * observation_std[2]

            forecast_scores[lead_index, metric_index] = score.detach()
            control_sensitivity[lead_index, metric_index] = (
                control_gradient.detach()
            )
            direct_norm[lead_index, metric_index, 2] = torch.linalg.vector_norm(
                direct_gradient.detach()
            )
            tile_direct_norm[lead_index, metric_index, 2] = _tile_l2(
                direct_gradient.detach(),
                sensitivity_config.tile_size,
            )
            if whitening_available:
                tile_whitened_norm[lead_index, metric_index, 2] = _tile_l2(
                    whitened_gradient.detach(),
                    sensitivity_config.tile_size,
                )

            if lead_index in selected_position:
                position = selected_position[lead_index]
                forecast_maps[position, metric_index] = forecast_gradient.detach()
                direct_maps[position, metric_index, 2] = direct_gradient.detach()

            if impact_input_available:
                contribution = torch.where(
                    innovation_mask[2],
                    direct_gradient.detach() * innovation[2],
                    torch.zeros_like(direct_gradient),
                )
                tiles = _tile_sum(
                    contribution,
                    sensitivity_config.tile_size,
                )
                tile_impact[lead_index, metric_index, 2] = tiles
                observation_impact[lead_index, metric_index, 2] = tiles.sum()

    has_metric_support = bool(torch.any(metric_available))
    impact_available = impact_input_available and has_metric_support
    if not impact_available:
        observation_impact.fill_(float("nan"))
        tile_impact.fill_(float("nan"))

    reward = amplitude.new_full(observation_impact.shape, float("nan"))
    if baseline_scores is not None:
        baseline_scores = baseline_scores.to(
            dtype=amplitude.dtype,
            device=amplitude.device,
        )
        if baseline_scores.shape != forecast_scores.shape:
            raise ValueError("baseline_scores must match forecast_scores shape")
        if not bool(torch.all(torch.isfinite(baseline_scores))) or bool(
            torch.any(baseline_scores < 0)
        ):
            raise ValueError("baseline_scores must be finite and non-negative")
        if impact_available:
            reward = -observation_impact / (
                baseline_scores[..., None] + sensitivity_config.epsilon
            )
    reward_available = impact_available and baseline_scores is not None

    trust_components = _trust_components(
        state,
        control,
        amplitude,
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
        metric_names=sensitivity_config.metric_names,
        lead_minutes=lead_minutes,
        full_map_lead_minutes=sensitivity_config.full_map_lead_minutes,
        tile_size=sensitivity_config.tile_size,
        context_feature_names=CONTEXT_FEATURE_NAMES,
        context_features=extract_context_features(
            frames_dbz,
            state,
            nowcast_config,
        ),
        analysis_control=control.detach(),
        forecast_scores=forecast_scores,
        metric_available=metric_available,
        control_sensitivity=control_sensitivity,
        forecast_sensitivity=forecast_maps,
        forecast_cap_active_mask=selected_cap_masks,
        direct_observation_sensitivity=direct_maps,
        direct_observation_sensitivity_norm=direct_norm,
        tile_direct_sensitivity_norm=tile_direct_norm,
        tile_whitened_direct_sensitivity_norm=tile_whitened_norm,
        direct_observation_impact=observation_impact,
        tile_direct_observation_impact=tile_impact,
        direct_normalized_reward=reward,
        latest_sensitivity_mask=latest_active,
        observation_std_dbz=(
            observation_std
            if whitening_available
            else amplitude.new_full(frames_dbz.shape, float("nan"))
        ),
        observation_innovation_dbz=(
            innovation
            if innovation is not None
            else amplitude.new_full(frames_dbz.shape, float("nan"))
        ),
        observation_innovation_mask=(
            innovation_mask
            if innovation_mask is not None
            else torch.zeros_like(frames_dbz, dtype=torch.bool)
        ),
        baseline_scores=(
            baseline_scores.detach()
            if baseline_scores is not None
            else amplitude.new_full(forecast_scores.shape, float("nan"))
        ),
        reward_epsilon=sensitivity_config.epsilon,
        trust_components=trust_components,
        trust_score=trust_score,
        impact_available=impact_available,
        reward_available=reward_available,
        whitened_tile_norm_available=whitening_available,
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
    frames_dbz: Tensor,
    state: RadarState,
    config: NowcastConfig,
) -> Tensor:
    """Extract a small auditable context vector for later retrieval."""

    latest = torch.nan_to_num(
        frames_dbz[2],
        nan=config.min_dbz,
        posinf=config.max_dbz,
        neginf=config.min_dbz,
    ).clamp(config.min_dbz, config.max_dbz)
    active = latest >= config.echo_threshold_dbz
    strong = latest >= 35.0
    active_values = latest[active]
    if active_values.numel():
        q90 = torch.quantile(active_values, 0.9)
    else:
        q90 = latest.new_tensor(config.min_dbz)

    border_width = max(1, min(latest.shape) // 16)
    border = torch.zeros_like(active)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    active_count = active.sum().clamp_min(1)
    boundary_fraction = (active & border).sum() / active_count

    linear = dbz_to_linear(latest, config)
    center = torch.nan_to_num(
        _soft_centroid(linear, torch.ones_like(active)),
        nan=0.0,
    )
    motion = state.displacement_yx
    return torch.stack(
        (
            motion[0],
            motion[1],
            torch.linalg.vector_norm(motion),
            state.log_growth_per_step,
            state.motion_disagreement_px,
            state.growth_disagreement,
            latest.mean(),
            latest.max(),
            q90,
            active.to(latest.dtype).mean(),
            strong.to(latest.dtype).mean(),
            boundary_fraction.to(latest.dtype),
            center[0],
            center[1],
            torch.log1p(linear.sum()),
        )
    ).detach()


def _state_from_control(
    template: RadarState,
    control: Tensor,
    amplitude: Tensor,
) -> RadarState:
    return RadarState(
        echo_amplitude=amplitude,
        displacement_yx=control[:2],
        log_growth_per_step=control[2],
        pair_displacements_yx=template.pair_displacements_yx,
        pair_log_growth=template.pair_log_growth,
        provenance=template.provenance,
        forecast_status=template.forecast_status,
        data_coverage_fraction=template.data_coverage_fraction,
        latest_data_coverage_fraction=(
            template.latest_data_coverage_fraction
        ),
        background_used=template.background_used,
        background_age_minutes=template.background_age_minutes,
        forecast_source_mask=template.forecast_source_mask,
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


def _frozen_observations(
    frames: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
    qc_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Freeze cleaning/QC choices and return the latest differentiable mask."""

    finite = torch.isfinite(frames)
    clean = torch.nan_to_num(
        frames,
        nan=nowcast_config.min_dbz,
        posinf=nowcast_config.max_dbz,
        neginf=nowcast_config.min_dbz,
    ).clamp(nowcast_config.min_dbz, nowcast_config.max_dbz)
    if qc_mask is None:
        accepted = torch.ones_like(finite)
    else:
        accepted = qc_mask

    margin = sensitivity_config.active_margin_dbz
    latest_active = (
        finite[2]
        & accepted[2]
        & (clean[2] > nowcast_config.min_dbz + margin)
        & (clean[2] < nowcast_config.max_dbz - margin)
    )
    return clean.detach(), latest_active.detach()


def _active_dbz_to_amplitude(
    candidate_dbz: Tensor,
    nominal_dbz: Tensor,
    nominal_amplitude: Tensor,
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
    floor = 10.0 ** (config.min_dbz / 10.0)
    candidate_echo = torch.pow(10.0, safe_dbz / 10.0) - floor
    nominal_echo = torch.pow(10.0, nominal_safe_dbz / 10.0) - floor
    candidate_amplitude = torch.sqrt(candidate_echo.clamp_min(config.epsilon))
    nominal_active_amplitude = torch.sqrt(
        nominal_echo.clamp_min(config.epsilon)
    ).detach()
    perturbation = torch.where(
        active,
        candidate_amplitude - nominal_active_amplitude,
        torch.zeros_like(candidate_amplitude),
    )
    return nominal_amplitude.detach() + perturbation


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
    frames: Tensor,
    background: Tensor | None,
    qc_mask: Tensor | None,
    config: NowcastConfig,
) -> tuple[Tensor | None, Tensor | None]:
    if background is None:
        return None, None
    valid = torch.isfinite(frames) & torch.isfinite(background)
    if qc_mask is not None:
        valid = valid & qc_mask
    clean_frames = torch.nan_to_num(
        frames,
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
        clean_frames - clean_background,
        torch.full_like(frames, float("nan")),
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
    control: Tensor,
    amplitude: Tensor,
    truth: Tensor,
    valid: Tensor,
    gradients: Tensor,
    metric_available: Tensor,
    cap_masks: Tensor,
    nowcast_config: NowcastConfig,
    sensitivity_config: SensitivityConfig,
) -> dict[str, float]:
    verification_quality = valid.to(amplitude.dtype).mean().clamp(0.0, 1.0)
    support_quality = metric_available.to(amplitude.dtype).mean()
    if not bool(torch.any(metric_available)):
        return {
            "linearity": 0.0,
            "verification": float(verification_quality),
            "metric_support": 0.0,
        }

    delta = control.new_tensor(sensitivity_config.linearity_delta)
    predicted_change = torch.sum(gradients[metric_available].mean(dim=0) * delta)

    def aggregate(candidate_control: Tensor) -> Tensor:
        candidate_state = _state_from_control(
            template,
            candidate_control,
            amplitude,
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
    }


def _validate_inputs(
    frames: Tensor,
    verification: Tensor,
    state: RadarState,
    config: NowcastConfig,
    background: Tensor | None,
    qc_mask: Tensor | None,
) -> None:
    if frames.ndim != 3 or frames.shape[0] != 3:
        raise ValueError("frames_dbz must have shape [3, height, width]")
    expected = (config.forecast_steps, *frames.shape[1:])
    if tuple(verification.shape) != expected:
        raise ValueError(f"verification_frames_dbz must have shape {expected}")
    if tuple(state.echo_amplitude.shape) != tuple(frames.shape[1:]):
        raise ValueError("state grid must match frame grid")
    if background is not None and background.shape != frames.shape:
        raise ValueError("background_frames_dbz must match frames_dbz shape")
    if not frames.is_floating_point() or not verification.is_floating_point():
        raise TypeError("frames and verification must be floating-point tensors")
    if background is not None and not background.is_floating_point():
        raise TypeError("background_frames_dbz must be floating-point")
    if qc_mask is not None:
        if qc_mask.shape != frames.shape:
            raise ValueError("qc_mask must match frames_dbz shape")
        if qc_mask.dtype != torch.bool:
            raise TypeError("qc_mask must be boolean")
    if state.displacement_yx.shape != (2,):
        raise ValueError("state displacement must have shape [2]")
    if state.log_growth_per_step.ndim != 0:
        raise ValueError("state log growth must be scalar")
    if frames.device != verification.device:
        raise ValueError("frames and verification must use the same device")
    state_tensors = (
        state.echo_amplitude,
        state.displacement_yx,
        state.log_growth_per_step,
    )
    if any(tensor.device != frames.device for tensor in state_tensors):
        raise ValueError("state and frames must use the same device")
    if any(not tensor.is_floating_point() for tensor in state_tensors):
        raise TypeError("state tensors must be floating-point")
    if background is not None and background.device != frames.device:
        raise ValueError("background and frames must use the same device")
    if qc_mask is not None and qc_mask.device != frames.device:
        raise ValueError("qc_mask and frames must use the same device")
