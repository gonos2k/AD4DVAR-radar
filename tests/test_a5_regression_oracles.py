"""Small, independent regression oracles for the remaining A5 gaps."""

import math
from dataclasses import replace

import torch

import advar.sensitivity as sensitivity
import advar.variational as variational
from advar.nowcast import (
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    NowcastConfig,
    RadarGridTimeContract,
    radar_projected_crs_semantic_digest,
)
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.sensitivity import SensitivityConfig
from test_sensitivity import _current_verification_bundle


def test_zero_background_vector_update_binds_control_and_both_ad_paths() -> None:
    background = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    control = torch.tensor((0.3, -0.2), dtype=torch.float64, requires_grad=True)
    limit = 10.0

    def decode(baseline: torch.Tensor, increment: torch.Tensor) -> torch.Tensor:
        return variational._bounded_vector_update(
            baseline,
            increment,
            scale=1.0,
            limit=limit,
        )

    norm = torch.linalg.vector_norm(control)
    normalized_norm = norm / limit
    expected = torch.tanh(normalized_norm) / normalized_norm * control
    torch.testing.assert_close(decode(background, control), expected)

    background_jacobian, control_jacobian = torch.func.jacrev(
        decode,
        argnums=(0, 1),
    )(background, control)
    assert bool(torch.isfinite(background_jacobian).all())
    assert bool(torch.isfinite(control_jacobian).all())
    # At zero background both arguments enter the smooth interior map with the
    # same scale, so their first derivatives coincide.
    torch.testing.assert_close(background_jacobian, control_jacobian)
    radial_derivative = 1.0 / torch.cosh(normalized_norm).square()
    torch.testing.assert_close(
        control_jacobian @ control,
        radial_derivative * control,
    )
    assert torch.autograd.gradcheck(decode, (background, control))
    assert torch.autograd.gradgradcheck(decode, (background, control))


def test_canonical_observation_residual_binds_detected_value_and_censor_limit() -> None:
    frames = torch.full((3, 3, 3), 20.0, dtype=torch.float64)
    frames[:, 0, 1] = 0.0
    analysis_config = variational.AnalysisConfig(
        censored_background_policy="detection_limit",
    )
    observations, frozen = variational.prepare_analysis(
        frames,
        analysis_config=analysis_config,
    )
    control = variational.initial_control(frozen)
    changed = replace(observations, dbz=observations.dbz + 0.25)
    original_residual = variational.observation_residual_dbz(
        control,
        observations,
        frozen,
    )
    changed_residual = variational.observation_residual_dbz(
        control,
        changed,
        frozen,
    )

    detected = observations.detected_mask
    censored = observations.censored_mask
    assert bool(torch.any(detected))
    assert bool(torch.any(censored))
    torch.testing.assert_close(
        (changed_residual - original_residual)[detected],
        torch.full_like(original_residual[detected], -0.25),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        (changed_residual - original_residual)[censored],
        torch.zeros_like(original_residual[censored]),
        rtol=0.0,
        atol=1.0e-12,
    )

    trajectory = variational._analysis_trajectory(control, frozen)
    prediction_dbz = echo_to_dbz(
        trajectory.frames_linear,
        min_dbz=frozen.nowcast_config.min_dbz,
    )
    temperature = frozen.analysis_config.censor_temperature_dbz
    threshold = frozen.analysis_config.detection_limit_dbz
    expected_censored = temperature * torch.nn.functional.softplus(
        (prediction_dbz - threshold) / temperature
    )
    torch.testing.assert_close(
        original_residual[censored],
        expected_censored[censored],
        rtol=0.0,
        atol=1.0e-12,
    )


def _current_two_lead_p1_fixture():
    coordinates = torch.arange(8, dtype=torch.float64)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frames = torch.stack(
        tuple(
            -10.0
            + 40.0
            * torch.exp(
                -((y - center).square() + (x - center).square()) / 4.0
            )
            for center in (3.0, 3.5, 4.0)
        )
    )
    grid = RadarGridTimeContract(
        valid_times=(
            "2026-08-05T00:00:00Z",
            "2026-08-05T00:10:00Z",
            "2026-08-05T00:20:00Z",
        ),
        dx_m=1000.0,
        dy_m=1000.0,
        projection="EPSG:5179",
        grid_hash="a" * 64,
        spatial_grid_contract="radar-spatial-grid-identity-v6",
        grid_shape_yx=(8, 8),
        projected_crs_digest=radar_projected_crs_semantic_digest("EPSG:5179"),
        metric_domain_digest=CURRENT_RADAR_METRIC_DOMAIN.digest,
        metric_domain_evidence_digest=CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest,
        cell_center_origin_xy_m=(1_000_000.0, 2_000_000.0),
        grid_coordinate_dtype=RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        cell_center_convention=RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    )
    return variational.variational_nowcast(
        frames,
        nowcast_config=NowcastConfig(horizon_minutes=20),
        analysis_config=variational.AnalysisConfig(
            censored_background_policy="detection_limit",
            maximum_outer_iterations=12,
            maximum_pcg_iterations=100,
            pcg_relative_tolerance=1.0e-8,
        ),
        grid_time_contract=grid,
    ) + (grid,)


def test_current_v22_censored_fso_uses_typed_nonuniform_weights_and_sensitivity() -> None:
    forecast, analysis, grid = _current_two_lead_p1_fixture()
    truth = forecast.forecast_dbz.clone()
    truth[0, 1, 1] = float("nan")
    truth[0, 1, 2] += 1.0
    truth[0, 2, 3] -= 2.0
    truth[1, 1, 2] += 3.0
    blocked_first_source = torch.zeros_like(truth)
    blocked_first_source[0, 4:] = 1.0
    blocked_first_source[1, :4] = 1.0
    blockage = torch.stack((blocked_first_source, 1.0 - blocked_first_source))
    bundle = _current_verification_bundle(
        truth,
        valid_times=(
            "2026-08-05T00:30:00Z",
            "2026-08-05T00:40:00Z",
        ),
        grid_time_contract=grid,
        source_quality_weights=(1.0, 0.25),
        beam_blockage_fraction_by_source=blockage,
    )
    config = SensitivityConfig(
        metric_names=("log_echo_mse",),
        full_map_lead_minutes=(10,),
        tile_size=4,
        require_verification_lineage=True,
    )
    fso = sensitivity.compute_variational_fso(
        forecast,
        analysis,
        bundle,
        sensitivity_config=config,
    )

    floor = 10.0 ** (forecast.run.config.min_dbz / 10.0)
    forecast_linear = dbz_to_echo(
        forecast.forecast_dbz,
        min_dbz=forecast.run.config.min_dbz,
        max_dbz=forecast.run.config.max_dbz,
    )
    clean_truth = torch.nan_to_num(
        truth,
        nan=forecast.run.config.min_dbz,
        posinf=forecast.run.config.max_dbz,
        neginf=forecast.run.config.min_dbz,
    ).clamp(forecast.run.config.min_dbz, forecast.run.config.max_dbz)
    truth_linear = dbz_to_echo(
        clean_truth,
        min_dbz=forecast.run.config.min_dbz,
        max_dbz=forecast.run.config.max_dbz,
    )
    expected_scores = []
    unweighted_scores = []
    expected_weights = []
    for lead in range(2):
        weight = bundle.fso_metric_weight[lead] * forecast.valid_mask[lead].to(
            forecast_linear
        )
        residual = (
            torch.log(forecast_linear[lead] + floor)
            - torch.log(truth_linear[lead] + floor)
        )
        expected_scores.append(
            (weight * residual.square()).sum() / weight.sum()
        )
        unweighted_scores.append(residual[weight > 0.0].square().mean())
        expected_weights.append(weight)
    expected = torch.stack(expected_scores)
    unweighted = torch.stack(unweighted_scores)
    weights = torch.stack(expected_weights)
    assert bool(torch.all(torch.isfinite(weights)))
    assert float(weights[0].sum()) > 0.0
    assert float(weights[0][weights[0] > 0.0].amax()) > float(
        weights[0][weights[0] > 0.0].amin()
    )
    assert int(torch.count_nonzero(weights[0] > 0.0)) >= 2
    assert float(weights[0, 1, 1]) == 0.0
    assert int(torch.count_nonzero(
        (
            torch.log(forecast_linear[0] + floor)
            - torch.log(truth_linear[0] + floor)
        )[weights[0] > 0.0]
    )) >= 2
    torch.testing.assert_close(fso.forecast_scores[:, 0], expected)
    assert not math.isclose(
        float(expected[0]),
        float(unweighted[0]),
        rel_tol=1.0e-6,
        abs_tol=1.0e-9,
    )
    detected_maps = fso.observation.detected_dbz.maps
    assert bool(torch.isfinite(detected_maps).all())
    assert float(torch.linalg.vector_norm(detected_maps)) > 0.0
