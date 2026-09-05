from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import advar.sensitivity as sensitivity
import test_sensitivity as fixtures
from test_a2_fso import _current_p1_fixture


def test_interior_peak_switch_is_not_certified_by_matching_samples() -> None:
    fixtures.VariationalFSOTests.setUpClass()
    case = fixtures.VariationalFSOTests
    linearization = case.analysis.linearization
    assert linearization is not None
    observations = linearization.observations
    frozen = linearization.frozen
    generator = torch.Generator().manual_seed(22)
    for _ in range(81):
        delta = torch.where(
            observations.detected_mask,
            0.15 * torch.randn(
                observations.dbz.shape,
                generator=generator,
                dtype=observations.dbz.dtype,
            ),
            torch.zeros_like(observations.dbz),
        )
    nominal = sensitivity._p0_tendency_branch_signature(observations.dbz, frozen)
    assert [
        sensitivity._p0_tendency_branch_signature(
            observations.dbz + scale * delta, frozen,
        ) == nominal
        for scale in (0.25, 0.5, 1.0)
    ] == [False, True, True]
    perturbation = sensitivity.VariationalObservationPerturbation.from_radar_dbz_delta(
        delta, linearization,
    )
    config = sensitivity.VariationalAdjointConfig(
        maximum_perturbed_fraction=1.0,
        maximum_perturbed_pixel_count=100,
    )
    exploratory = sensitivity.compute_variational_fsoi(
        case.forecast, case.analysis, case.verification, perturbation,
        sensitivity_config=case.sensitivity_config,
        adjoint_config=config,
    )
    assert exploratory.baseline_dynamics_branch_status == "unknown"
    assert exploratory.observation.baseline_branch_trusted_total is None
    with pytest.raises(ValueError, match="baseline dynamics branch is not certified"):
        sensitivity.compute_variational_fsoi(
            case.forecast, case.analysis, case.verification, perturbation,
            sensitivity_config=case.sensitivity_config,
            adjoint_config=replace(config, require_baseline_dynamics_branch_validity=True),
        )
    assert sensitivity._baseline_dynamics_branch_certification(
        observations, frozen, torch.zeros_like(delta),
    )[0] == "certified"


def test_current_learning_rejects_nonzero_uncertified_p0_path() -> None:
    forecast, analysis, grid = _current_p1_fixture()
    linearization = analysis.linearization
    assert linearization is not None
    verification = fixtures._current_verification_bundle(
        forecast.forecast_dbz - 0.5,
        valid_times=("2026-08-05T00:30:00Z",),
        grid_time_contract=grid,
    )
    policy = sensitivity.AutomatedLearningPolicy(
        sensitivity_config=replace(
            sensitivity.SensitivityConfig.for_automated_learning(
                radar_product_digest="5" * 64,
                qc_pipeline_digest="6" * 64,
            ),
            metric_names=("log_echo_mse",),
            full_map_lead_minutes=(10,),
            tile_size=4,
        ),
        adjoint_config=replace(
            sensitivity.VariationalAdjointConfig.for_automated_learning(),
            lead_minutes=(10,),
        ),
        algorithm_bundle_digest=linearization.algorithm_bundle_digest,
        numerical_runtime_digest=linearization.numerical_runtime_digest,
    )
    delta = torch.zeros_like(linearization.observations.dbz)
    index = tuple(torch.nonzero(linearization.observations.detected_mask)[0])
    delta[index] = 5.0e-6
    perturbation = sensitivity.VariationalObservationPerturbation.from_radar_dbz_delta(
        delta, linearization,
    )
    # Only policy authorization is a fixture; the physical branch and all
    # current geometry, verification, and numerical checks run normally.
    with patch.object(
        sensitivity,
        "_load_learning_policy_trust_store",
        return_value=sensitivity._LearningPolicyTrustStore(
            frozenset((policy.digest,)), "7" * 64,
        ),
    ):
        learning = sensitivity.compute_variational_fsoi_for_learning(
            forecast, analysis, verification, perturbation,
            policy=policy, policy_trust_store_path="/unused-fixture",
        )
    assert not learning.eligibility.eligible
    assert learning.eligibility.reasons == (
        "P1 FSOI baseline dynamics branch is not certified",
    )
    assert learning.first_order_validation is None
