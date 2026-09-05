from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_sensitivity import _current_verification_bundle

from advar.nowcast import (
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    NowcastConfig,
    RadarGridTimeContract,
    radar_projected_crs_semantic_digest,
)
from advar.sensitivity import (
    CURRENT_VARIATIONAL_FSOI_CONTRACT,
    FirstOrderValidation,
    LearningApprovalEvidence,
    LearningEligibility,
    SensitivityConfig,
    VariationalImpactChannel,
    VariationalLearningImpact,
    VariationalObservationPerturbation,
    _metric_domain_weight,
    _resolve_verification,
    _resolved_forecast_scores,
    _variational_impact_digest,
    compute_sensitivity_snapshot,
    compute_variational_fso,
    compute_variational_fsoi,
    forecast_metric,
    validate_variational_learning_impact,
)
from advar.variational import AnalysisConfig, variational_nowcast
from advar.physics import dbz_to_echo
from advar.nowcast import RadarState


def _current_p1_fixture() -> tuple[object, object, RadarGridTimeContract]:
    torch.set_num_threads(1)
    coordinates = torch.arange(8, dtype=torch.float64)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frames = torch.stack(
        tuple(
            -10.0
            + 40.0 * torch.exp(
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
    forecast, analysis = variational_nowcast(
        frames,
        nowcast_config=NowcastConfig(horizon_minutes=10),
        analysis_config=AnalysisConfig(
            censored_background_policy="detection_limit",
            maximum_outer_iterations=12,
            maximum_pcg_iterations=100,
            pcg_relative_tolerance=1.0e-8,
        ),
        grid_time_contract=grid,
    )
    return forecast, analysis, grid


def test_current_v22_censored_weight_binds_p1_domain_and_score() -> None:
    forecast, analysis, grid = _current_p1_fixture()
    truth = forecast.forecast_dbz.clone()
    cell = tuple(int(value) for value in torch.nonzero(forecast.valid_mask[0])[0])
    truth[(0, *cell)] = float("nan")
    bundle = _current_verification_bundle(
        truth,
        valid_times=("2026-08-05T00:30:00Z",),
        grid_time_contract=grid,
    )
    config = SensitivityConfig(
        metric_names=("log_echo_mse",),
        full_map_lead_minutes=(10,),
        require_verification_lineage=True,
    )

    fso = compute_variational_fso(
        forecast,
        analysis,
        bundle,
        sensitivity_config=config,
    )
    resolved = _resolve_verification(bundle, forecast, config)
    expected_scores, expected_available = _resolved_forecast_scores(
        forecast,
        analysis.state,
        resolved,
        (10,),
        config,
    )

    assert float(bundle.fso_metric_weight.sum()) == 48.0
    assert float(fso.metric_domain_weight_sum[0]) == 48.0
    assert torch.equal(fso.metric_available, expected_available)
    torch.testing.assert_close(fso.forecast_scores, expected_scores)
    assert fso.forecast_scores[0, 0].abs() < 1.0e-20

    linearization = analysis.linearization
    assert linearization is not None
    delta = torch.zeros_like(linearization.observations.dbz)
    observation_cell = tuple(
        int(value)
        for value in torch.nonzero(
            linearization.observations.detected_mask,
            as_tuple=False,
        )[0]
    )
    delta[observation_cell] = 0.01
    fsoi = compute_variational_fsoi(
        forecast,
        analysis,
        bundle,
        VariationalObservationPerturbation.from_radar_dbz_delta(
            delta,
            linearization,
        ),
        sensitivity_config=config,
    )
    assert float(fsoi.fso.metric_domain_weight_sum[0]) == 48.0


def test_current_v22_spatial_gate_produces_no_fso_support() -> None:
    forecast, analysis, grid = _current_p1_fixture()
    truth = forecast.forecast_dbz.clone()
    truth[0, 1, 1] += 5.0
    bundle = _current_verification_bundle(
        truth,
        valid_times=("2026-08-05T00:30:00Z",),
        grid_time_contract=grid,
        acquisition_time_offset_seconds=-60.0,
    )
    config = SensitivityConfig(
        metric_names=("log_echo_mse",),
        full_map_lead_minutes=(10,),
        require_verification_lineage=True,
    )

    fso = compute_variational_fso(
        forecast,
        analysis,
        bundle,
        sensitivity_config=config,
    )

    assert int(bundle.spatial_metric_valid_mask.sum()) == 0
    assert float(bundle.fso_metric_weight.sum()) == 0.0
    assert float(fso.metric_domain_weight_sum[0]) == 0.0
    assert not bool(fso.metric_available[0, 0])
    assert torch.isnan(fso.forecast_scores[0, 0])


def test_current_v22_m0_uses_censored_and_fractional_weights() -> None:
    from test_sensitivity import result_for

    _, _, grid = _current_p1_fixture()
    nowcast_config = NowcastConfig(horizon_minutes=10)
    latest = torch.full((8, 8), 20.0, dtype=torch.float64)
    state = RadarState(
        echo_linear=dbz_to_echo(
            latest,
            min_dbz=nowcast_config.min_dbz,
            max_dbz=nowcast_config.max_dbz,
        ),
        displacement_yx=torch.zeros(2, dtype=torch.float64),
        log_growth_per_step=torch.zeros((), dtype=torch.float64),
    )
    result = result_for(
        state,
        nowcast_config,
        frames=torch.stack((latest, latest, latest)),
        grid_time_contract=grid,
    )
    truth = result.forecast_dbz.clone()
    truth[0, 0, 0] = float("nan")
    bundle = _current_verification_bundle(
        truth,
        valid_times=("2026-08-05T00:30:00Z",),
        grid_time_contract=grid,
    )
    sensitivity_config = SensitivityConfig(
        metric_names=("log_echo_mse",),
        full_map_lead_minutes=(10,),
        tile_size=4,
        require_verification_lineage=True,
    )

    snapshot = compute_sensitivity_snapshot(
        latest,
        result,
        bundle,
        sensitivity_config=sensitivity_config,
    )
    resolved = _resolve_verification(bundle, result, sensitivity_config)
    weight = _metric_domain_weight(
        result,
        resolved.valid_mask[0],
        0,
        "issued",
        verification_metric_weight=resolved.metric_weight[0],
    )
    truth_linear = dbz_to_echo(
        torch.nan_to_num(
            bundle.frames_dbz,
            nan=nowcast_config.min_dbz,
            posinf=nowcast_config.max_dbz,
            neginf=nowcast_config.min_dbz,
        ),
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    forecast_linear = dbz_to_echo(
        result.forecast_dbz,
        min_dbz=nowcast_config.min_dbz,
        max_dbz=nowcast_config.max_dbz,
    )
    expected_score = forecast_metric(
        "log_echo_mse",
        forecast_linear[0],
        truth_linear[0],
        weight,
        nowcast_config,
        sensitivity_config,
        grid,
    )

    torch.testing.assert_close(snapshot.forecast_scores[0, 0], expected_score)
    assert snapshot.forecast_scores[0, 0].abs() < 1.0e-20
    assert snapshot.forecast_sensitivity[0, 0, 0, 0] == 0.0
    fractional = _metric_domain_weight(
        result,
        resolved.valid_mask[0],
        0,
        "issued",
        verification_metric_weight=0.25 * resolved.metric_weight[0],
    )
    torch.testing.assert_close(fractional, 0.25 * weight)


def test_metric_domain_weight_detaches_fractional_typed_verification_weight() -> None:
    from test_sensitivity import result_for
    from advar.nowcast import RadarState
    from advar.physics import dbz_to_echo

    config = NowcastConfig(horizon_minutes=10)
    latest = torch.full((2, 2), 20.0, dtype=torch.float64)
    state = RadarState(
        echo_linear=dbz_to_echo(
            latest,
            min_dbz=config.min_dbz,
            max_dbz=config.max_dbz,
        ),
        displacement_yx=torch.zeros(2, dtype=torch.float64),
        log_growth_per_step=torch.zeros((), dtype=torch.float64),
    )
    result = result_for(state, config)
    finite = torch.ones_like(latest, dtype=torch.bool)
    typed_weight = torch.full_like(latest, 0.25, requires_grad=True)

    combined = _metric_domain_weight(
        result,
        finite,
        0,
        "issued",
        verification_metric_weight=typed_weight,
    )

    torch.testing.assert_close(combined, torch.full_like(latest, 0.25))
    assert not combined.requires_grad


def test_v4_learning_approval_binds_full_analysis_input_digest() -> None:
    def digest(character: str) -> str:
        return character * 64
    validation = FirstOrderValidation(
        source_fsoi_digest=digest("b"),
        nominal_forecast_digest=digest("c"),
        nominal_input_bundle_digest=digest("d"),
        nominal_full_analysis_input_digest=digest("e"),
        full_step_prediction=torch.ones((1, 1)),
        full_step_resolved_metric_change=torch.ones((1, 1)),
        full_step_absolute_error=torch.zeros((1, 1)),
        half_step_prediction=torch.ones((1, 1)),
        half_step_resolved_metric_change=torch.ones((1, 1)),
        half_step_absolute_error=torch.zeros((1, 1)),
        metric_available=torch.ones((1, 1), dtype=torch.bool),
        full_step_resolved_analysis_converged=True,
        half_step_resolved_analysis_converged=True,
        active_branch_valid=True,
        full_step_valid=True,
        half_step_valid=True,
        sign_consistent_for_material_impacts=True,
        material_metric_count=1,
        maximum_material_impact=1.0,
        aggregate_material_impact_norm=1.0,
        first_order_valid=True,
        full_step_analysis_digest=digest("f"),
        half_step_analysis_digest=digest("0"),
        full_step_forecast_digest=digest("1"),
        half_step_forecast_digest=digest("2"),
    )
    impact = VariationalImpactChannel(
        maps=torch.zeros((1, 1, 3, 1, 1)),
        sum_by_time=torch.zeros((1, 1, 3)),
        tile_sum_by_time=torch.zeros((1, 1, 3, 1, 1)),
    )
    fake_fso = SimpleNamespace(
        contract="fake-fso",
        forecast_run_digest=digest("c"),
    )
    fake_fsoi = SimpleNamespace(
        contract=CURRENT_VARIATIONAL_FSOI_CONTRACT,
        fso=fake_fso,
        variational_fsoi_digest=digest("b"),
        perturbation_digest=digest("3"),
    )
    eligibility = LearningEligibility(
        eligible=True,
        reasons=(),
        policy_digest=digest("4"),
    )

    def evidence(full_analysis_digest: str) -> LearningApprovalEvidence:
        return LearningApprovalEvidence(
            policy_digest=digest("4"),
            trust_store_digest=digest("5"),
            fsoi_digest=digest("b"),
            full_step_analysis_digest=digest("f"),
            half_step_analysis_digest=digest("0"),
            full_step_forecast_digest=digest("1"),
            half_step_forecast_digest=digest("2"),
            first_order_validation_digest=validation.validation_digest,
            learning_impact_digest=_variational_impact_digest(impact),
            approved_action_digest=digest("3"),
            nominal_input_bundle_digest=digest("d"),
            nominal_full_analysis_input_digest=full_analysis_digest,
        )

    with (
        patch("advar.sensitivity.validate_variational_fso", return_value=None),
        patch("advar.sensitivity.validate_variational_fsoi", return_value=None),
    ):
        legitimate = VariationalLearningImpact(
            eligibility=eligibility,
            fsoi=fake_fsoi,
            first_order_validation=validation,
            frozen_domain_learning_impact=impact,
            approval_evidence=evidence(digest("e")),
        )
        forged = object.__new__(VariationalLearningImpact)
        for field in fields(VariationalLearningImpact):
            object.__setattr__(forged, field.name, getattr(legitimate, field.name))
        object.__setattr__(forged, "approval_evidence", evidence(digest("6")))

        validate_variational_learning_impact(legitimate)
        with pytest.raises(ValueError, match="learning approval evidence mismatch"):
            validate_variational_learning_impact(forged)
