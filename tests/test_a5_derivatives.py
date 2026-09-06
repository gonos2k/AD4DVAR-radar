from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import test_sensitivity as sensitivity_tests  # noqa: E402
import test_variational as variational_tests  # noqa: E402
import advar.sensitivity as sensitivity_module  # noqa: E402
from advar.nowcast import ForecastRunContract  # noqa: E402
from advar.sensitivity import (  # noqa: E402
    ObservationRemovalConfig,
    SensitivityConfig,
    VariationalAdjointConfig,
    VariationalObservationPerturbation,
    compute_variational_fso,
    compute_variational_fsoi,
    compute_variational_observation_removal_impact,
)
from advar.variational import (  # noqa: E402
    AnalysisConfig,
    NeuralPriorInferenceRunner,
    variational_nowcast,
)


class _PartialRadarPrior(nn.Module):
    """A differentiable radar prior with one retained support cell."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = value[0] + self.anchor
        validity = torch.full_like(state, 0.8)
        support = torch.zeros_like(state)
        support[3, 3] = 1.0
        event_probability = torch.full_like(state, 0.8)
        return (
            state,
            torch.ones_like(state),
            validity,
            support,
            event_probability,
            state,
            torch.ones_like(state),
        )


class A5DerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sensitivity_tests.VariationalFSOTests.setUpClass()
        cls.fixture = sensitivity_tests.VariationalFSOTests

    def test_censored_observations_drive_frozen_p0_dynamics(self) -> None:
        fixture = self.fixture
        nowcast_config = replace(
            fixture.nowcast_config,
            echo_threshold_dbz=0.0,
        )
        analysis_config = replace(
            fixture.analysis_config,
            detection_limit_dbz=5.0,
            maximum_outer_iterations=20,
        )
        forecast, analysis = variational_nowcast(
            fixture.frames,
            nowcast_config=nowcast_config,
            analysis_config=analysis_config,
        )
        linearization = analysis.linearization
        self.assertIsNotNone(linearization)
        assert linearization is not None
        observations = linearization.observations
        frozen = linearization.frozen
        mid_censored = observations.censored_mask & (observations.dbz >= 0.0)
        self.assertGreater(int(torch.count_nonzero(mid_censored)), 0)

        path = sensitivity_module._prepare_frozen_baseline_dynamics_path(
            observations,
            frozen,
        )
        self.assertIsNotNone(path)
        assert path is not None
        torch.testing.assert_close(
            path.active_mask,
            observations.valid_mask & frozen.observed_mask,
        )

        verification = torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        )
        fso = compute_variational_fso(
            forecast,
            analysis,
            verification,
            sensitivity_config=SensitivityConfig(
                metric_names=("log_echo_mse",),
                full_map_lead_minutes=(10,),
                tile_size=4,
            ),
        )
        dynamics_map = fso.observation.baseline_dynamics_dbz.maps[0, 0]
        self.assertGreater(float(dynamics_map[mid_censored].abs().max()), 0.0)
        self.assertEqual(fso.baseline_dynamics_branch_status, "unknown")

    def _partial_prior_case(self):
        fixture = self.fixture
        analysis_config = AnalysisConfig(
            detection_limit_dbz=5.0,
            censored_background_policy="floor",
            maximum_outer_iterations=20,
            maximum_pcg_iterations=100,
            pcg_relative_tolerance=1.0e-8,
        )
        state_contract = replace(
            variational_tests._PRIOR_STATE_CONTRACT,
            valid_decision_probability=0.6,
            support_decision_probability=0.7,
        )
        runner = NeuralPriorInferenceRunner(
            _PartialRadarPrior().eval(),
            lambda value: value,
            example_frames=fixture.frames,
            state_contract=state_contract,
            probability_contract=variational_tests._PRIOR_PROBABILITY_CONTRACT,
            model_contract_digest="4" * 64,
            feature_schema_digest="5" * 64,
            training_manifest_digest="6" * 64,
            dependency="radar_dependent",
        )
        input_run = ForecastRunContract.from_inputs(
            fixture.nowcast_config,
            fixture.frames,
            torch.ones_like(fixture.frames, dtype=torch.bool),
            None,
        )
        application = runner.infer(
            fixture.frames,
            input_run=input_run,
            role="candidate",
        )
        forecast, analysis = variational_nowcast(
            fixture.frames,
            nowcast_config=fixture.nowcast_config,
            analysis_config=analysis_config,
            neural_prior=application,
        )
        return runner, application, forecast, analysis

    def test_partial_radar_prior_keeps_identity_fallback_in_fsoi(self) -> None:
        runner, application, forecast, analysis = self._partial_prior_case()
        linearization = analysis.linearization
        self.assertIsNotNone(linearization)
        assert linearization is not None
        observations = linearization.observations
        frozen = linearization.frozen
        prior_valid = frozen.neural_prior_valid_mask
        self.assertIsNotNone(prior_valid)
        assert prior_valid is not None
        fallback = (
            observations.detected_mask[0]
            & frozen.observed_mask[0]
            & ~prior_valid
        )
        self.assertGreater(int(torch.count_nonzero(fallback)), 0)

        index = torch.nonzero(fallback, as_tuple=False)[0]
        location = (0, int(index[0]), int(index[1]))
        delta = torch.zeros_like(self.fixture.frames)
        delta[location] = 0.01
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            delta,
            linearization,
            neural_prior_runner=runner,
            neural_prior_application=application,
        )
        self.assertEqual(float(perturbation.initial_background_dbz[location]), 0.01)

        verification = torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        )
        fsoi = compute_variational_fsoi(
            forecast,
            analysis,
            verification,
            perturbation,
            sensitivity_config=self.fixture.sensitivity_config,
            adjoint_config=VariationalAdjointConfig(
                maximum_gauss_newton_relative_curvature_defect=1.0e9,
            ),
            neural_prior_runner=runner,
            neural_prior_application=application,
        )
        background_map = fsoi.fso.observation.initial_background_dbz.maps[0, 0]
        self.assertNotEqual(float(background_map[location]), 0.0)

    def test_observation_removal_reinfers_exogenous_prior(self) -> None:
        fixture = self.fixture
        analysis_config = AnalysisConfig(
            detection_limit_dbz=5.0,
            censored_background_policy="floor",
            maximum_outer_iterations=12,
            maximum_pcg_iterations=100,
            pcg_relative_tolerance=1.0e-8,
        )
        runner = variational_tests._prior_runner(
            0.0,
            "candidate",
            example_frames=fixture.frames,
            dependency="exogenous",
        )
        input_run = ForecastRunContract.from_inputs(
            fixture.nowcast_config,
            fixture.frames,
            torch.ones_like(fixture.frames, dtype=torch.bool),
            None,
        )
        application = runner.infer(
            fixture.frames,
            input_run=input_run,
            role="candidate",
        )
        forecast, analysis = variational_nowcast(
            fixture.frames,
            nowcast_config=fixture.nowcast_config,
            analysis_config=analysis_config,
            neural_prior=application,
        )
        linearization = analysis.linearization
        self.assertIsNotNone(linearization)
        assert linearization is not None
        observations = linearization.observations
        index = torch.nonzero(observations.valid_mask, as_tuple=False)[0]
        location = tuple(int(value) for value in index)
        removal_mask = torch.zeros_like(observations.valid_mask)
        removal_mask[location] = True
        sensitivity_config = SensitivityConfig(
            metric_names=("log_echo_mse",),
            full_map_lead_minutes=(10,),
            tile_size=4,
        )
        verification = torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        )
        impact = compute_variational_observation_removal_impact(
            forecast,
            analysis,
            verification,
            removal_mask,
            sensitivity_config=sensitivity_config,
            removal_config=ObservationRemovalConfig(
                maximum_removed_area_km2=None,
            ),
            neural_prior_runner=runner,
            neural_prior_application=application,
        )

        changed_qc = ~removal_mask
        changed_run = ForecastRunContract.from_inputs(
            fixture.nowcast_config,
            fixture.frames,
            changed_qc,
            None,
        )
        changed_application = runner.infer(
            fixture.frames,
            input_run=changed_run,
            role="candidate",
        )
        self.assertEqual(
            changed_application.neural_prior_digest,
            application.neural_prior_digest,
        )
        self.assertNotEqual(
            changed_application.application_digest,
            application.application_digest,
        )
        expected_forecast, expected_analysis = variational_nowcast(
            fixture.frames,
            nowcast_config=fixture.nowcast_config,
            analysis_config=analysis_config,
            qc_mask=changed_qc,
            neural_prior=changed_application,
        )
        expected_fso = compute_variational_fso(
            expected_forecast,
            expected_analysis,
            verification,
            sensitivity_config=sensitivity_config,
        )
        torch.testing.assert_close(
            impact.removed_scores,
            expected_fso.forecast_scores,
            rtol=0.0,
            atol=0.0,
        )

    def test_observation_removal_requires_and_preserves_source_history(self) -> None:
        fixture = self.fixture
        source_available_mask = torch.ones_like(
            fixture.frames,
            dtype=torch.bool,
        )
        source_available_mask[0, 0, 0] = False
        forecast, analysis = variational_nowcast(
            fixture.frames,
            nowcast_config=fixture.nowcast_config,
            analysis_config=fixture.analysis_config,
            source_available_mask=source_available_mask,
        )
        linearization = analysis.linearization
        self.assertIsNotNone(linearization)
        assert linearization is not None
        observations = linearization.observations
        self.assertFalse(bool(observations.valid_mask[0, 0, 0]))
        index = torch.nonzero(observations.valid_mask, as_tuple=False)[0]
        location = tuple(int(value) for value in index)
        removal_mask = torch.zeros_like(observations.valid_mask)
        removal_mask[location] = True
        removal_config = ObservationRemovalConfig(maximum_removed_area_km2=None)

        with self.assertRaisesRegex(ValueError, "source availability mask"):
            compute_variational_observation_removal_impact(
                forecast,
                analysis,
                forecast.forecast_dbz,
                removal_mask,
                sensitivity_config=fixture.sensitivity_config,
                removal_config=removal_config,
            )
        impact = compute_variational_observation_removal_impact(
            forecast,
            analysis,
            forecast.forecast_dbz,
            removal_mask,
            sensitivity_config=fixture.sensitivity_config,
            removal_config=removal_config,
            source_available_mask=source_available_mask,
        )
        self.assertEqual(impact.removed_observation_count, 1)


if __name__ == "__main__":
    unittest.main()
