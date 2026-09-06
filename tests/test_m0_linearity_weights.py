"""M0 Taylor diagnostics must use the score's fixed real weights."""

import math
from dataclasses import replace

import pytest
import torch
from test_sensitivity import _current_verification_bundle, result_for

from advar.nowcast import (
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    forecast_from_state,
    radar_projected_crs_semantic_digest,
)
from advar.physics import dbz_to_echo
from advar.sensitivity import SensitivityConfig, compute_sensitivity_snapshot


@pytest.mark.parametrize("delta", [1.0e-3, 1.0e-4, 1.0e-5])
@pytest.mark.parametrize(
    "domain, verification_case",
    [
        ("issued", "nonuniform"),
        ("confidence_weighted", "unit"),
        ("radar_dynamics_anchored", "unit"),
        ("issued", "unit"),
        ("issued", "uniform"),
        ("issued", "missing"),
    ],
)
def test_m0_linearity_uses_score_weights_and_boolean_coverage(
    domain: str,
    verification_case: str,
    delta: float,
) -> None:
    config = NowcastConfig(horizon_minutes=20)
    latest = torch.full((8, 8), 20.0, dtype=torch.float64)
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
    echo = dbz_to_echo(latest, min_dbz=config.min_dbz, max_dbz=config.max_dbz)
    state = RadarState(echo, latest.new_zeros(2), latest.new_zeros(()))
    result = result_for(
        state,
        config,
        frames=torch.stack((latest, latest, latest)),
        grid_time_contract=grid,
    )
    # Distinct spatial evidence creates a real confidence weight and a smaller
    # anchored domain while retaining full forecast issuance.
    support = torch.ones_like(latest)
    support[4:6] = 0.1
    support[6:] = 0.0
    result = forecast_from_state(
        state,
        replace(
            result.metadata,
            local_motion_verified_support=support,
            local_growth_verified_support=support,
            local_dynamics_verified_support=support,
        ),
        config,
        run=result.run,
    )
    truth = torch.full_like(result.forecast_dbz, 10.0)
    truth[:, 4:] = 30.0
    truth[1] += 2.0
    qc = torch.ones_like(truth)
    quality_weights = (1.0,)
    blockage = None
    if verification_case == "nonuniform":
        quality_weights = (1.0, 0.01)
        first_blocked = torch.zeros_like(truth)
        first_blocked[0, 4:] = 1.0
        first_blocked[1, :4] = 1.0
        blockage = torch.stack((first_blocked, 1.0 - first_blocked))
    elif verification_case == "uniform":
        quality_weights = (0.25,)
    elif verification_case == "missing":
        truth[:, 0, 0] = float("nan")
        qc[:, 7, 7] = 0.0
    bundle = _current_verification_bundle(
        truth,
        valid_times=("2026-08-05T00:30:00Z", "2026-08-05T00:40:00Z"),
        grid_time_contract=grid,
        attenuation_qc_score=qc,
        source_quality_weights=quality_weights,
        beam_blockage_fraction_by_source=blockage,
    )
    sensitivity_config = SensitivityConfig(
        metric_names=("log_echo_mse",),
        metric_domain=domain,
        full_map_lead_minutes=(10, 20),
        tile_size=4,
        linearity_delta=(0.0, 0.0, delta),
        require_verification_lineage=True,
    )
    forecast_before = result.forecast_dbz.clone()
    snapshot = compute_sensitivity_snapshot(
        latest,
        result,
        bundle,
        sensitivity_config=sensitivity_config,
    )

    # Independently specify the fixed evaluation measure, without the helper
    # under test. Coverage always counts Boolean eligible verification cells.
    coverage = bundle.valid_mask & result.valid_mask & (bundle.fso_metric_weight > 0)
    weight = bundle.fso_metric_weight * coverage
    if verification_case == "nonuniform":
        assert weight[0, 0, 0] > weight[0, 7, 7] > 0.0
        assert 0.0 < weight[1, 0, 0] < weight[1, 7, 7]
    elif verification_case == "uniform":
        assert bool(torch.all((weight > 0.0) & (weight < 1.0)))
    if domain == "confidence_weighted":
        weight = weight * result.forecast_confidence
        assert not torch.allclose(weight[:, :4].mean(), weight[:, 4:].mean())
    elif domain == "radar_dynamics_anchored":
        weight = weight * result.radar_dynamics_anchored_valid_mask
        assert int((weight > 0).sum()) < int(coverage.sum())
    denominator = weight.sum(dim=(-2, -1))
    floor = 10.0 ** (config.min_dbz / 10.0)
    truth_echo = dbz_to_echo(
        torch.nan_to_num(bundle.frames_dbz, nan=config.min_dbz),
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    residual = torch.log(echo + floor) - torch.log(truth_echo + floor)
    expected_scores = (weight * residual.square()).sum(dim=(-2, -1)) / denominator
    growth_sum = latest.new_tensor([1.0, 1.0 + math.exp(-10.0 / 60.0)])
    expected_growth_gradient = (
        (2.0 * weight * residual * echo / (echo + floor)).sum(dim=(-2, -1))
        / denominator
        * growth_sum
    )
    perturbed_echo = echo * torch.exp(delta * growth_sum[:, None, None])
    perturbed_residual = torch.log(perturbed_echo + floor) - torch.log(
        truth_echo + floor
    )
    perturbed_scores = (weight * perturbed_residual.square()).sum(
        dim=(-2, -1)
    ) / denominator
    actual_change = (perturbed_scores - expected_scores).mean()
    predicted_change = delta * expected_growth_gradient.mean()
    expected_linearity = torch.exp(
        -torch.abs(actual_change - predicted_change)
        / (
            torch.abs(actual_change)
            + torch.abs(predicted_change)
            + sensitivity_config.epsilon
        )
        / 0.25
    )
    torch.testing.assert_close(snapshot.forecast_scores[:, 0], expected_scores)
    torch.testing.assert_close(
        snapshot.control_sensitivity[:, 0, 2], expected_growth_gradient
    )
    expected_forecast_gradient = (
        2.0 * weight * residual / (echo + floor) / denominator[:, None, None]
    )
    torch.testing.assert_close(
        snapshot.forecast_sensitivity[:, 0], expected_forecast_gradient
    )
    assert snapshot.trust_components["linearity"] == pytest.approx(
        float(expected_linearity), abs=1.0e-9
    )
    assert snapshot.trust_components["verification"] == float(
        coverage.to(latest).mean()
    )
    assert snapshot.trust_score == pytest.approx(
        math.prod(snapshot.trust_components.values())
    )
    assert abs(float(predicted_change)) > delta * 0.1
    assert float(expected_linearity) > 0.99
    assert torch.equal(result.forecast_dbz, forecast_before)
    assert bool(torch.all(snapshot.metric_available))
