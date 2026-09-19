"""Small, exploratory mean-only prior learning experiment.

The experiment keeps the prior standard deviation and hard support/validity
decisions constant. Every finite-difference value comes from a fresh P1 solve;
this is not a deployment or promotion path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn

from advar.nowcast import (
    ForecastRunContract,
    NowcastConfig,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    RadarGridTimeContract,
    radar_projected_crs_semantic_digest,
)
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.sensitivity import (
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    SensitivityConfig,
    VariationalAdjointConfig,
    VariationalFSOI,
    VariationalObservationPerturbation,
    compute_variational_fso,
    compute_variational_fsoi,
    forecast_metric,
)
from advar.variational import (
    AnalysisConfig,
    NeuralPriorInferenceRunner,
    NeuralPriorProbabilityContract,
    NeuralPriorStateContract,
    neural_prior_state_censor_policy_digest,
    variational_nowcast,
)

MIN_DBZ = -10.0
MAX_DBZ = 70.0
INTERVAL_MINUTES = 10
HORIZON_MINUTES = 20
GRID_SHAPE = (16, 16)
METRIC = "log_echo_mse"
FD_STEPS = (1.0e-3, 5.0e-4, 2.5e-4)
TAYLOR_RELATIVE_TOLERANCE = 0.10
PRIOR_REGION = (7, 9, 7, 9)


class MeanOnlyPrior(nn.Module):
    """One trainable mean offset and fixed uncertainty/support heads."""

    def __init__(self, offset: float = 0.5) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset, dtype=torch.float64))

    def forward(self, features: Tensor) -> tuple[Tensor, ...]:
        mean = features + self.offset
        fixed_std = torch.ones_like(mean)
        fixed_valid = torch.zeros_like(mean)
        y0, y1, x0, x1 = PRIOR_REGION
        fixed_valid[y0:y1, x0:x1] = 1.0
        fixed_support = torch.ones_like(mean)
        fixed_probability = torch.ones_like(mean)
        return (
            mean,
            fixed_std,
            fixed_valid,
            fixed_support,
            fixed_probability,
            mean,
            fixed_std,
        )


@dataclass(frozen=True)
class LearningRun:
    initial_offset: float
    updated_offset: float
    parameter_gradient: float
    objective_before: float
    objective_after: float
    heldout_before: float
    heldout_after: float
    finite_difference_steps: tuple[float, ...]
    finite_difference_gradients: tuple[float, ...]
    half_taylor_error: float
    full_taylor_error: float
    fsoi_predicted_impact: float
    full_resolved_impact: float
    reload_max_abs_error: float
    next_assimilation_max_abs_error: float
    analysis_relative_stationarity_before: float
    analysis_relative_stationarity_after: float
    analysis_truth_mae_before: float
    analysis_truth_mae_after: float
    next_window_forecast_before: float
    next_window_forecast_after: float
    next_window_analysis_truth_mae_before: float
    next_window_analysis_truth_mae_after: float
    gB_norm: float
    fso_contract: str
    fsoi_contract: str


@dataclass(frozen=True)
class _Run:
    model: MeanOnlyPrior
    runner: NeuralPriorInferenceRunner
    application: object
    result: object
    analysis: object


def _grid(start_minutes: int = 0) -> RadarGridTimeContract:
    origin = datetime(2026, 8, 5, tzinfo=timezone.utc)
    times = tuple(
        (origin + timedelta(minutes=start_minutes + 10 * index))
        .isoformat()
        .replace("+00:00", "Z")
        for index in range(3)
    )
    return RadarGridTimeContract(
        valid_times=times,
        background_valid_times=times,
        dx_m=1000.0,
        dy_m=1000.0,
        projection="EPSG:5179",
        grid_hash="4" * 64,
        spatial_grid_contract="radar-spatial-grid-identity-v6",
        grid_shape_yx=GRID_SHAPE,
        projected_crs_digest=radar_projected_crs_semantic_digest("EPSG:5179"),
        metric_domain_digest=CURRENT_RADAR_METRIC_DOMAIN.digest,
        metric_domain_evidence_digest=CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest,
        cell_center_origin_xy_m=(1_000_000.0, 2_000_000.0),
        grid_coordinate_dtype=RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        cell_center_convention=RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    )


def _frames(center_offset: float = 0.0) -> Tensor:
    coordinate = torch.arange(GRID_SHAPE[0], dtype=torch.float64)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.stack(
        tuple(
            20.0
            + 5.0 * torch.exp(-((y - center) ** 2 + (x - center) ** 2) / 8.0)
            for center in (6.0 + center_offset, 6.5 + center_offset, 7.0 + center_offset)
        )
    )


def _truths(center_offset: float = 0.0) -> Tensor:
    coordinate = torch.arange(GRID_SHAPE[0], dtype=torch.float64)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.stack(
        (
            20.0
            + 5.0
            * torch.exp(
                -(
                    (y - (7.5 + center_offset)) ** 2
                    + (x - (7.5 + center_offset)) ** 2
                )
                / 8.0
            ),
            20.0
            + 5.0
            * torch.exp(
                -(
                    (y - (8.0 + center_offset)) ** 2
                    + (x - (8.0 + center_offset)) ** 2
                )
                / 8.0
            ),
        )
    )


def _analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        censored_background_policy="floor",
        maximum_outer_iterations=100,
        maximum_pcg_iterations=200,
        pcg_relative_tolerance=1.0e-8,
        initial_increment_scale_dbz=1.0,
        field_smoothness_weight=0.0,
        maximum_final_linearization_polish_iterations=4,
        pseudo_huber_delta=1.0e6,
        echo_transform_scale_dbz=10.0,
    )


def _sensitivity_config() -> SensitivityConfig:
    return SensitivityConfig(
        metric_names=(METRIC,),
        metric_domain="radar_dynamics_anchored",
        full_map_lead_minutes=(INTERVAL_MINUTES,),
        tile_size=4,
        tile_size_m=4000.0,
    )


def _adjoint_config() -> VariationalAdjointConfig:
    return VariationalAdjointConfig(
        lead_minutes=(INTERVAL_MINUTES,),
        pcg_relative_tolerance=1.0e-8,
        maximum_pcg_iterations=200,
        perturbation_tile_size=4,
        perturbation_tile_size_m=4000.0,
        require_active_set_margin=True,
        require_feasibility_margin=True,
        require_gauss_newton_reliability=True,
    )


def _prior_runner(model: MeanOnlyPrior, frames: Tensor) -> NeuralPriorInferenceRunner:
    return NeuralPriorInferenceRunner(
        model.eval(),
        lambda features: features[0],
        example_frames=frames,
        model_contract_digest="4" * 64,
        feature_schema_digest="5" * 64,
        training_manifest_digest="6" * 64,
        state_contract=NeuralPriorStateContract(
            state_product_digest="a" * 64,
            state_qc_pipeline_digest="9" * 64,
            state_mask_policy_digest="3" * 64,
            state_censor_policy_digest=neural_prior_state_censor_policy_digest(
                detection_limit_dbz=5.0,
                censor_temperature_dbz=1.0,
                censored_background_policy="floor",
                minimum_dbz=MIN_DBZ,
                maximum_dbz=MAX_DBZ,
            ),
            support_threshold_dbz=5.0,
            minimum_state_dbz=MIN_DBZ,
            maximum_state_dbz=MAX_DBZ,
            minimum_state_std_dbz=0.1,
            maximum_state_std_dbz=20.0,
        ),
        probability_contract=NeuralPriorProbabilityContract(
            support_threshold_dbz=5.0,
            support_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=MIN_DBZ,
        ),
        dependency="radar_dependent",
    )


def _run_at_offset(
    offset: float,
    frames: Tensor,
    grid: RadarGridTimeContract,
) -> _Run:
    return _run_model(MeanOnlyPrior(offset), frames, grid)


def _run_model(
    model: MeanOnlyPrior,
    frames: Tensor,
    grid: RadarGridTimeContract,
) -> _Run:
    runner = _prior_runner(model, frames)
    mask = torch.ones_like(frames, dtype=torch.bool)
    run = ForecastRunContract.from_inputs(
        NowcastConfig(horizon_minutes=HORIZON_MINUTES),
        frames,
        mask,
        frames.clone(),
        0.0,
        grid_time_contract=grid,
    )
    application = runner.infer(frames, input_run=run, role="candidate")
    result, analysis = variational_nowcast(
        frames,
        nowcast_config=run.config,
        analysis_config=_analysis_config(),
        qc_mask=mask,
        background_frames_dbz=frames,
        background_age_minutes=0.0,
        grid_time_contract=grid,
        neural_prior=application,
    )
    return _Run(model, runner, application, result, analysis)


def _metric_weight(result: object, truth: Tensor, lead: int) -> Tensor:
    finite = torch.isfinite(truth)
    issued = result.valid_mask[lead]
    anchored = result.radar_dynamics_anchored_valid_mask[lead]
    weight = finite & issued & anchored
    if int(torch.count_nonzero(weight)) == 0:
        raise RuntimeError("fixed verification domain has no scored pixels")
    return weight.to(truth)


def _score(result: object, truth: Tensor, weight: Tensor, lead: int) -> float:
    value = forecast_metric(
        METRIC,
        dbz_to_echo(result.forecast_dbz[lead], min_dbz=MIN_DBZ, max_dbz=MAX_DBZ),
        dbz_to_echo(truth, min_dbz=MIN_DBZ, max_dbz=MAX_DBZ),
        weight,
        result.run.config,
        _sensitivity_config(),
        result.run.grid_time_contract,
    )
    return float(value.detach())


def _assert_stable(
    nominal: _Run,
    candidate: _Run,
    train_weight: Tensor,
    hold_weight: Tensor,
    truths: Tensor,
) -> None:
    a = nominal.analysis.linearization
    b = candidate.analysis.linearization
    if a is None or b is None:
        raise RuntimeError("P1 linearization is missing")
    for name in (
        "converged", "used_fallback", "degraded", "final_linearization_stationary",
        "final_robust_stationary", "final_irls_fixed_point", "p1_forecast_eligible",
        "posterior_eligible", "fso_eligible",
    ):
        if getattr(nominal.analysis, name) is not getattr(candidate.analysis, name):
            raise RuntimeError(f"P1 status changed: {name}")
    def same(left: object, right: object) -> bool:
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            return torch.equal(left, right)
        if isinstance(left, tuple) and isinstance(right, tuple):
            return len(left) == len(right) and all(same(x, y) for x, y in zip(left, right))
        return left == right

    for left, right, name in (
        (a.frozen.neural_prior_valid_mask, b.frozen.neural_prior_valid_mask, "prior valid"),
        (a.frozen.initial_support_mask, b.frozen.initial_support_mask, "support"),
        (a.frozen.active_field_index, b.frozen.active_field_index, "active field"),
        (a.frozen.analysis_remap_cells, b.frozen.analysis_remap_cells, "analysis remap"),
        (a.frozen.detected_masks, b.frozen.detected_masks, "detection class"),
        (a.frozen.observed_mask, b.frozen.observed_mask, "observation class"),
        (
            a.frozen.baseline_state.displacement_yx,
            b.frozen.baseline_state.displacement_yx,
            "baseline motion",
        ),
        (
            a.frozen.baseline_state.log_growth_per_step,
            b.frozen.baseline_state.log_growth_per_step,
            "baseline growth",
        ),
        (nominal.result.valid_mask, candidate.result.valid_mask, "issued domain"),
        (
            nominal.result.radar_dynamics_anchored_valid_mask,
            candidate.result.radar_dynamics_anchored_valid_mask,
            "metric domain",
        ),
        (nominal.application.state_std_dbz, candidate.application.state_std_dbz, "prior std"),
        (
            nominal.application.state_valid_probability,
            candidate.application.state_valid_probability,
            "prior validity probability",
        ),
        (
            nominal.application.state_support_probability,
            candidate.application.state_support_probability,
            "prior support probability",
        ),
        (
            nominal.application.event_probability,
            candidate.application.event_probability,
            "prior event probability",
        ),
    ):
        if left is None or right is None or not same(left, right):
            raise RuntimeError(f"branch changed: {name}")
    candidate_prior = b.frozen.neural_prior_valid_mask
    if candidate_prior is None:
        raise RuntimeError("candidate prior-valid mask is missing")
    if not torch.allclose(
        b.frozen.initial_background_dbz[candidate_prior],
        candidate.application.initial_background_dbz[candidate_prior],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("candidate accepted prior differs from raw mean")
    if not torch.equal(train_weight, _metric_weight(candidate.result, truths[0], 0)):
        raise RuntimeError("training metric weight changed")
    if not torch.equal(hold_weight, _metric_weight(candidate.result, truths[1], 1)):
        raise RuntimeError("held-out metric weight changed")


def _relative_error(actual: float, predicted: float) -> float:
    return abs(actual - predicted) / max(abs(predicted), 1.0e-10)


def _analysis_relative_stationarity(run: _Run) -> float:
    linearization = run.analysis.linearization
    if linearization is None:
        raise RuntimeError("analysis error requested without linearization")
    values = (linearization.relative_stationarity, linearization.robust_relative_stationarity)
    if not all(torch.isfinite(torch.tensor(value)) for value in values):
        raise RuntimeError("analysis stationarity is nonfinite")
    return float(max(abs(value) for value in values))


def _analysis_truth_mae(run: _Run, truth: Tensor) -> float:
    analysis_dbz = echo_to_dbz(
        run.analysis.state.echo_linear,
        min_dbz=MIN_DBZ,
    )
    valid = torch.isfinite(analysis_dbz) & torch.isfinite(truth)
    if not bool(torch.any(valid)):
        raise RuntimeError("analysis truth has no finite overlap")
    return float((analysis_dbz[valid] - truth[valid]).abs().mean().detach())


def _atomic_save(model: MeanOnlyPrior, checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{checkpoint.name}.",
        dir=checkpoint.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_mean_only_learning(
    *,
    checkpoint_path: str | Path,
    learning_rate: float = 0.25,
) -> LearningRun:
    """Run one local FSO-guided optimizer step and independent reanalyses."""

    if not torch.isfinite(torch.tensor(learning_rate)) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    torch.manual_seed(20260909)
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        model = MeanOnlyPrior()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        initial_offset = float(model.offset.detach())
    else:
        model = MeanOnlyPrior()
        initial_offset = float(model.offset.detach())
    frames, truths, grid = _frames(), _truths(), _grid()
    nominal = _run_model(model, frames, grid)
    train_weight = _metric_weight(nominal.result, truths[0], 0)
    hold_weight = _metric_weight(nominal.result, truths[1], 1)
    fso = compute_variational_fso(
        nominal.result, nominal.analysis, truths,
        sensitivity_config=_sensitivity_config(), adjoint_config=_adjoint_config(),
        neural_prior_runner=nominal.runner, neural_prior_application=nominal.application,
    )
    prior_valid = nominal.analysis.linearization.frozen.neural_prior_valid_mask
    if prior_valid is None:
        raise RuntimeError("FSO did not retain neural-prior validity")
    gB = fso.observation.initial_background_dbz.maps[0, 0, 0]
    gB = torch.where(prior_valid, gB, torch.zeros_like(gB))
    if not bool(torch.isfinite(gB).all()) or float(torch.linalg.vector_norm(gB)) == 0.0:
        raise RuntimeError("prior-only gB is empty or nonfinite")
    accepted = nominal.analysis.linearization.frozen.initial_background_dbz
    raw = nominal.application.initial_background_dbz
    if not torch.allclose(accepted[prior_valid], raw[prior_valid], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("accepted prior background differs from raw mean")
    nominal_train = _score(nominal.result, truths[0], train_weight, 0)
    nominal_holdout = _score(nominal.result, truths[1], hold_weight, 1)

    nominal.model.zero_grad()
    torch.sum(nominal.model(frames[0])[0] * gB).backward()
    if nominal.model.offset.grad is None or not bool(torch.isfinite(nominal.model.offset.grad)):
        raise RuntimeError("mean-only parameter gradient is unavailable")
    parameter_gradient = float(nominal.model.offset.grad.detach())

    fd_values = []
    for step in FD_STEPS:
        plus = _run_at_offset(initial_offset + step, frames, grid)
        minus = _run_at_offset(initial_offset - step, frames, grid)
        _assert_stable(nominal, plus, train_weight, hold_weight, truths)
        _assert_stable(nominal, minus, train_weight, hold_weight, truths)
        fd_values.append(
            (
                _score(plus.result, truths[0], train_weight, 0)
                - _score(minus.result, truths[0], train_weight, 0)
            )
            / (2.0 * step)
        )
    if abs(fd_values[1] - fd_values[2]) > 0.10 * max(abs(fd_values[2]), 1.0e-10):
        raise RuntimeError("central finite difference is not step-stable")

    delta_theta = -learning_rate * parameter_gradient
    half = _run_at_offset(initial_offset + 0.5 * delta_theta, frames, grid)
    full = _run_at_offset(initial_offset + delta_theta, frames, grid)
    _assert_stable(nominal, half, train_weight, hold_weight, truths)
    _assert_stable(nominal, full, train_weight, hold_weight, truths)
    half_change = _score(half.result, truths[0], train_weight, 0) - nominal_train
    full_change = _score(full.result, truths[0], train_weight, 0) - nominal_train
    half_error = _relative_error(half_change, 0.5 * parameter_gradient * delta_theta)
    full_error = _relative_error(full_change, parameter_gradient * delta_theta)
    if max(half_error, full_error) > TAYLOR_RELATIVE_TOLERANCE:
        raise RuntimeError("parameter Taylor check failed")

    delta_background = torch.zeros_like(frames)
    delta_background[0] = prior_valid.to(frames) * delta_theta
    perturbation = VariationalObservationPerturbation(
        detected_dbz=torch.zeros_like(frames), censor_threshold_dbz=torch.zeros_like(frames),
        observation_weight=torch.zeros_like(frames), initial_background_dbz=delta_background,
        perturbation_semantics="augmented_parameter",
    )
    fsoi: VariationalFSOI = compute_variational_fsoi(
        nominal.result, nominal.analysis, truths, perturbation,
        sensitivity_config=_sensitivity_config(), adjoint_config=_adjoint_config(),
        neural_prior_runner=nominal.runner, neural_prior_application=nominal.application,
    )
    fsoi_predicted = float(fsoi.observation.initial_background_dbz.sum_by_time[0, 0, 0])
    if abs(fsoi_predicted - parameter_gradient * delta_theta) > 1.0e-10:
        raise RuntimeError("augmented FSOI and gB chain disagree")
    if abs(fsoi_predicted - full_change) > 0.01 * max(abs(full_change), 1.0e-10):
        raise RuntimeError("augmented FSOI and resolved impact disagree")

    optimizer = torch.optim.SGD(nominal.model.parameters(), lr=learning_rate)
    optimizer.step()
    updated_offset = float(nominal.model.offset.detach())
    updated = _run_at_offset(updated_offset, frames, grid)
    _assert_stable(nominal, updated, train_weight, hold_weight, truths)
    updated_train = _score(updated.result, truths[0], train_weight, 0)
    updated_holdout = _score(updated.result, truths[1], hold_weight, 1)
    if not updated_train < nominal_train:
        raise RuntimeError("optimizer step did not reduce training objective")

    analysis_truth_before = _analysis_truth_mae(nominal, frames[2])
    analysis_truth_after = _analysis_truth_mae(updated, frames[2])
    next_frames = _frames(0.5)
    next_truths = _truths(0.5)
    next_grid = _grid(10)
    next_memory = _run_at_offset(updated_offset, next_frames, next_grid)
    next_before = _run_at_offset(initial_offset, next_frames, next_grid)
    next_train_weight = _metric_weight(next_memory.result, next_truths[0], 0)
    next_forecast_before = _score(
        next_before.result,
        next_truths[0],
        next_train_weight,
        0,
    )
    next_forecast_after = _score(
        next_memory.result,
        next_truths[0],
        next_train_weight,
        0,
    )
    next_analysis_before = _analysis_truth_mae(
        next_before,
        next_frames[2],
    )
    next_analysis_after = _analysis_truth_mae(next_memory, next_frames[2])
    _atomic_save(nominal.model, checkpoint)
    reloaded = MeanOnlyPrior()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    reload_error = float(
        (reloaded(frames[0])[0] - nominal.model(frames[0])[0])
        .abs()
        .max()
        .detach()
    )
    reloaded_run = _run_model(reloaded, frames, grid)
    if not torch.equal(reloaded_run.result.valid_mask, updated.result.valid_mask):
        raise RuntimeError("reloaded assimilation validity changed")
    finite = updated.result.valid_mask
    next_error = float(
        (reloaded_run.result.forecast_dbz[finite] - updated.result.forecast_dbz[finite])
        .abs()
        .max()
        .detach()
    )
    next_reloaded = _run_model(reloaded, next_frames, next_grid)
    if not torch.equal(next_memory.result.valid_mask, next_reloaded.result.valid_mask):
        raise RuntimeError("reloaded shifted assimilation validity changed")
    shifted_finite = next_memory.result.valid_mask
    shifted_error = float(
        (next_memory.result.forecast_dbz[shifted_finite]
         - next_reloaded.result.forecast_dbz[shifted_finite])
        .abs()
        .max()
        .detach()
    )
    if shifted_error != 0.0:
        raise RuntimeError("reloaded shifted assimilation forecast changed")
    return LearningRun(
        initial_offset=initial_offset, updated_offset=updated_offset,
        parameter_gradient=parameter_gradient, objective_before=nominal_train,
        objective_after=updated_train, heldout_before=nominal_holdout,
        heldout_after=updated_holdout, finite_difference_steps=FD_STEPS,
        finite_difference_gradients=tuple(fd_values), half_taylor_error=half_error,
        full_taylor_error=full_error, fsoi_predicted_impact=fsoi_predicted,
        full_resolved_impact=full_change, reload_max_abs_error=reload_error,
        next_assimilation_max_abs_error=max(next_error, shifted_error),
        gB_norm=float(torch.linalg.vector_norm(gB)),
        analysis_relative_stationarity_before=_analysis_relative_stationarity(nominal),
        analysis_relative_stationarity_after=_analysis_relative_stationarity(updated),
        analysis_truth_mae_before=analysis_truth_before,
        analysis_truth_mae_after=analysis_truth_after,
        next_window_forecast_before=next_forecast_before,
        next_window_forecast_after=next_forecast_after,
        next_window_analysis_truth_mae_before=next_analysis_before,
        next_window_analysis_truth_mae_after=next_analysis_after,
        fso_contract=fso.contract, fsoi_contract=fsoi.contract,
    )
