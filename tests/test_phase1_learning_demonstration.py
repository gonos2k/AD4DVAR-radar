"""Unmocked synthetic completion path for Phase 1 automated learning."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import test_sensitivity as sensitivity_tests  # noqa: E402
import advar.sensitivity as sensitivity  # noqa: E402
from advar.nowcast import (  # noqa: E402
    ForecastRunContract,
    NowcastConfig,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    RadarGridTimeContract,
    radar_projected_crs_semantic_digest,
)
from advar.physics import dbz_to_echo  # noqa: E402
from advar.sensitivity import (  # noqa: E402
    AutomatedLearningPolicy,
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    MetricTaylorThreshold,
    SensitivityConfig,
    VariationalAdjointConfig,
    VariationalObservationPerturbation,
    compute_variational_fsoi_for_learning,
    forecast_metric,
)
from advar.variational import (  # noqa: E402
    AnalysisConfig,
    NeuralPriorInferenceRunner,
    NeuralPriorProbabilityContract,
    NeuralPriorStateContract,
    neural_prior_state_censor_policy_digest,
    variational_nowcast,
)


class _RadarDependentPrior(nn.Module):
    """Small seven-head prior with nonzero mean and standard-deviation JVPs."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.tensor(1.0, dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mean = features + self.anchor
        std = torch.exp(0.002 * features)
        valid = torch.ones_like(mean)
        support = (mean >= 5.0).to(mean)
        probability = torch.ones_like(mean)
        return (mean, std, valid, support, probability, mean, std)


def _build_case() -> dict[str, object]:
    coordinates = torch.arange(16, dtype=torch.float64)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frames = torch.stack(
        tuple(
            20.0
            + 5.0 * torch.exp(-((y - center) ** 2 + (x - center) ** 2) / 8.0)
            for center in (6.0, 6.5, 7.0)
        )
    )
    background = frames.clone()
    observation_mask = torch.ones_like(frames, dtype=torch.bool)
    # The missing outer rings make P0 use the complete background tendency,
    # while the interior retains enough latest observations for P1 dynamics.
    observation_mask[1] = False
    observation_mask[1, 2:14, 2:14] = True
    observation_mask[2] = False
    observation_mask[2, 2:14, 2:14] = True

    nowcast_config = NowcastConfig(horizon_minutes=10)
    analysis_config = AnalysisConfig(
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
    prior = NeuralPriorInferenceRunner(
        _RadarDependentPrior().eval(),
        lambda features: features[0],
        example_frames=frames,
        state_contract=NeuralPriorStateContract(
            state_product_digest="a" * 64,
            state_qc_pipeline_digest="9" * 64,
            state_mask_policy_digest="3" * 64,
            state_censor_policy_digest=neural_prior_state_censor_policy_digest(
                detection_limit_dbz=5.0,
                censor_temperature_dbz=1.0,
                censored_background_policy="floor",
                minimum_dbz=-10.0,
                maximum_dbz=70.0,
            ),
            support_threshold_dbz=5.0,
            minimum_state_dbz=-10.0,
            maximum_state_dbz=70.0,
            minimum_state_std_dbz=0.1,
            maximum_state_std_dbz=20.0,
        ),
        probability_contract=NeuralPriorProbabilityContract(
            support_threshold_dbz=5.0,
            support_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
        ),
        model_contract_digest="4" * 64,
        feature_schema_digest="5" * 64,
        training_manifest_digest="6" * 64,
        dependency="radar_dependent",
    )
    observation_times = (
        "2026-08-05T00:00:00Z",
        "2026-08-05T00:10:00Z",
        "2026-08-05T00:20:00Z",
    )
    grid = RadarGridTimeContract(
        valid_times=observation_times,
        background_valid_times=observation_times,
        dx_m=1000.0,
        dy_m=1000.0,
        projection="EPSG:5179",
        grid_hash="4" * 64,
        spatial_grid_contract="radar-spatial-grid-identity-v6",
        grid_shape_yx=(16, 16),
        projected_crs_digest=radar_projected_crs_semantic_digest("EPSG:5179"),
        metric_domain_digest=CURRENT_RADAR_METRIC_DOMAIN.digest,
        metric_domain_evidence_digest=CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest,
        cell_center_origin_xy_m=(1_000_000.0, 2_000_000.0),
        grid_coordinate_dtype=RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        cell_center_convention=RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    )
    input_run = ForecastRunContract.from_inputs(
        nowcast_config,
        frames,
        observation_mask,
        background,
        0.0,
        grid_time_contract=grid,
    )
    application = prior.infer(frames, input_run=input_run, role="candidate")
    forecast, analysis = variational_nowcast(
        frames,
        nowcast_config=nowcast_config,
        analysis_config=analysis_config,
        qc_mask=observation_mask,
        background_frames_dbz=background,
        background_age_minutes=0.0,
        grid_time_contract=grid,
        neural_prior=application,
    )
    verification = sensitivity_tests._current_verification_bundle(
        torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        ),
        valid_times=("2026-08-05T00:30:00Z",),
        grid_time_contract=grid,
    )
    linearization = analysis.linearization
    if linearization is None:
        raise AssertionError("synthetic P1 did not retain a linearization")
    sensitivity_config = replace(
        SensitivityConfig.for_automated_learning(
            radar_product_digest="5" * 64,
            qc_pipeline_digest="6" * 64,
        ),
        metric_names=("log_echo_mse",),
        full_map_lead_minutes=(10,),
        tile_size=4,
        tile_size_m=4000.0,
        metric_domain="radar_dynamics_anchored",
    )
    adjoint_config = replace(
        VariationalAdjointConfig.for_automated_learning(),
        lead_minutes=(10,),
        maximum_perturbed_area_km2=256.0,
        perturbation_tile_size_m=4000.0,
        # Keep the production default 0.25 curvature-defect gate.
    )
    policy = AutomatedLearningPolicy(
        sensitivity_config=sensitivity_config,
        adjoint_config=adjoint_config,
        algorithm_bundle_digest=linearization.algorithm_bundle_digest,
        numerical_runtime_digest=linearization.numerical_runtime_digest,
        metric_taylor_thresholds=(
            MetricTaylorThreshold("log_echo_mse", 1.0e-6, 1.0e-8),
        ),
    )
    index = torch.nonzero(
        linearization.observations.detected_mask[0], as_tuple=False
    )[0]
    location = (0, int(index[0]), int(index[1]))
    return {
        "frames": frames,
        "background": background,
        "observation_mask": observation_mask,
        "nowcast_config": nowcast_config,
        "analysis_config": analysis_config,
        "prior": prior,
        "grid": grid,
        "application": application,
        "forecast": forecast,
        "analysis": analysis,
        "verification": verification,
        "linearization": linearization,
        "policy": policy,
        "location": location,
    }


class Phase1LearningDemonstrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = _build_case()
        defaults = AnalysisConfig()
        assert cls.case["analysis_config"].final_linearization_relative_stationarity_tolerance == defaults.final_linearization_relative_stationarity_tolerance
        assert cls.case["analysis_config"].final_robust_relative_stationarity_tolerance == defaults.final_robust_relative_stationarity_tolerance
        assert cls.case["analysis_config"].final_field_gradient_max_tolerance == defaults.final_field_gradient_max_tolerance
        assert cls.case["policy"].adjoint_config.maximum_gauss_newton_relative_curvature_defect == VariationalAdjointConfig().maximum_gauss_newton_relative_curvature_defect
        if cls.case["policy"].algorithm_bundle_digest != cls.case["linearization"].algorithm_bundle_digest:
            raise AssertionError("learning policy is not bound to the retained P1 algorithm")
        if cls.case["policy"].numerical_runtime_digest != cls.case["linearization"].numerical_runtime_digest:
            raise AssertionError("learning policy is not bound to the retained runtime")
        cls.trust_store = sensitivity._LearningPolicyTrustStore(
            approved_policy_digests=frozenset((cls.case["policy"].digest,)),
            content_digest="7" * 64,
        )

    def _run_learning(self, delta_value: float):
        case = self.case
        delta = torch.zeros_like(case["frames"])
        delta[case["location"]] = delta_value
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            delta,
            case["linearization"],
            neural_prior_runner=case["prior"],
            neural_prior_application=case["application"],
        )
        # The numerical path is unmocked. This fixture only supplies the
        # external policy boundary because a normal user-owned /tmp file is
        # intentionally rejected by the root-ownership trust-store contract.
        with patch.object(
            sensitivity,
            "_load_learning_policy_trust_store",
            return_value=self.trust_store,
        ):
            result = compute_variational_fsoi_for_learning(
                case["forecast"],
                case["analysis"],
                case["verification"],
                perturbation,
                policy=case["policy"],
                policy_trust_store_path=(
                    "/tmp/advar-phase1-completion/learning-policies.json"
                ),
                neural_prior_runner=case["prior"],
                neural_prior_application=case["application"],
            )
        return result, delta

    def test_radar_prior_p1_fsoi_real_resolves_and_approves(self) -> None:
        case = self.case
        learning, delta = self._run_learning(-0.1)
        self.assertTrue(learning.eligibility.eligible, learning.eligibility.reasons)
        self.assertEqual(learning.eligibility.reasons, ())
        self.assertIsNotNone(learning.fsoi)
        self.assertIsNotNone(learning.first_order_validation)
        self.assertIsNotNone(learning.approval_evidence)
        assert learning.fsoi is not None
        assert learning.first_order_validation is not None
        self.assertEqual(learning.fsoi.baseline_dynamics_branch_status, "not_applicable")
        self.assertGreater(
            float(torch.linalg.vector_norm(learning.fsoi.observation.initial_background_dbz.sum_by_time)),
            0.0,
        )
        validation = learning.first_order_validation
        self.assertTrue(validation.full_step_resolved_analysis_converged)
        self.assertTrue(validation.half_step_resolved_analysis_converged)
        self.assertTrue(validation.active_branch_valid)
        self.assertTrue(validation.full_step_valid)
        self.assertTrue(validation.half_step_valid)
        self.assertTrue(validation.sign_consistent_for_material_impacts)
        self.assertTrue(validation.first_order_valid)
        self.assertEqual(validation.material_metric_count, 1)
        self.assertGreater(validation.maximum_material_impact, 1.0e-8)
        self.assertGreater(validation.aggregate_material_impact_norm, 1.0e-8)

        # Independently rebuild the full changed forecast and evaluate the
        # metric, providing an oracle separate from the learning validator.
        changed_frames = case["frames"] + delta
        changed_run = ForecastRunContract.from_inputs(
            case["nowcast_config"],
            changed_frames,
            case["observation_mask"],
            case["background"],
            0.0,
            grid_time_contract=case["grid"],
        )
        changed_prior = case["prior"].infer(
            changed_frames, input_run=changed_run, role="candidate"
        )
        changed_forecast, _ = variational_nowcast(
            changed_frames,
            nowcast_config=case["nowcast_config"],
            analysis_config=case["analysis_config"],
            qc_mask=case["observation_mask"],
            background_frames_dbz=case["background"],
            background_age_minutes=0.0,
            grid_time_contract=case["grid"],
            neural_prior=changed_prior,
        )
        verification = case["verification"]
        weight = sensitivity._metric_domain_weight(
            case["forecast"],
            verification.valid_mask[0],
            0,
            "radar_dynamics_anchored",
            verification_metric_weight=verification.fso_metric_weight[0],
        )
        truth = dbz_to_echo(
            verification.frames_dbz,
            min_dbz=case["nowcast_config"].min_dbz,
            max_dbz=case["nowcast_config"].max_dbz,
        )
        nominal_linear = dbz_to_echo(
            case["forecast"].forecast_dbz[0],
            min_dbz=case["nowcast_config"].min_dbz,
            max_dbz=case["nowcast_config"].max_dbz,
        )
        changed_linear = dbz_to_echo(
            changed_forecast.forecast_dbz[0],
            min_dbz=case["nowcast_config"].min_dbz,
            max_dbz=case["nowcast_config"].max_dbz,
        )
        nominal_score = forecast_metric(
            "log_echo_mse", nominal_linear, truth[0], weight,
            case["nowcast_config"], case["policy"].sensitivity_config,
            case["grid"],
        )
        changed_score = forecast_metric(
            "log_echo_mse", changed_linear, truth[0], weight,
            case["nowcast_config"], case["policy"].sensitivity_config,
            case["grid"],
        )
        self.assertLess(float(changed_score), float(nominal_score))
        self.assertAlmostEqual(
            float(changed_score - nominal_score),
            float(validation.full_step_resolved_metric_change[0, 0]),
            delta=2.0e-8,
        )

    def test_unmaterial_delta_is_rejected_after_real_validation(self) -> None:
        learning, _ = self._run_learning(1.0e-6)
        self.assertFalse(learning.eligibility.eligible)
        self.assertEqual(
            learning.eligibility.reasons,
            ("no_material_learning_signal",),
        )
        self.assertIsNotNone(learning.first_order_validation)
        assert learning.first_order_validation is not None
        self.assertEqual(learning.first_order_validation.material_metric_count, 0)
        self.assertTrue(
            learning.first_order_validation.full_step_resolved_analysis_converged
        )
        self.assertTrue(
            learning.first_order_validation.half_step_resolved_analysis_converged
        )


if __name__ == "__main__":
    unittest.main()
