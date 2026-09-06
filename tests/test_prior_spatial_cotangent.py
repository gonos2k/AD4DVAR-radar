"""Spatial chain-rule regressions for the radar-dependent P1 prior."""

from unittest.mock import patch
import unittest

import torch

import test_a5_derivatives as fixtures
import advar.sensitivity as sensitivity
from advar.sensitivity import (
    VariationalAdjointConfig,
    VariationalObservationPerturbation,
    compute_variational_fso,
    compute_variational_fsoi,
)


class PriorSpatialCotangentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures.A5DerivativeTests.setUpClass()
        cls.fixture = fixtures.A5DerivativeTests.fixture
        cls.runner, cls.application, cls.forecast, cls.analysis = (
            fixtures.A5DerivativeTests()._partial_prior_case()
        )

    def _impact(self, delta, verification):
        linearization = self.analysis.linearization
        assert linearization is not None
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            delta,
            linearization,
            neural_prior_runner=self.runner,
            neural_prior_application=self.application,
        )
        return compute_variational_fsoi(
            self.forecast,
            self.analysis,
            verification,
            perturbation,
            sensitivity_config=self.fixture.sensitivity_config,
            adjoint_config=VariationalAdjointConfig(
                maximum_gauss_newton_relative_curvature_defect=1.0e9,
            ),
            neural_prior_runner=self.runner,
            neural_prior_application=self.application,
        )

    def test_identity_prior_and_fallback_preserve_spatial_gradient(self) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        prior_valid = linearization.frozen.neural_prior_valid_mask
        assert prior_valid is not None
        self.assertTrue(bool(prior_valid[3, 3]))
        self.assertFalse(bool(prior_valid[2, 3]))
        self.assertTrue(bool(linearization.observations.detected_mask[0, 2, 3]))
        delta = torch.zeros_like(self.fixture.frames)
        delta[0, 3, 3] = 0.01  # Prior-provided cell outside the first row.
        delta[0, 2, 3] = -0.02  # Observation fallback in the same input.
        row_error = torch.linspace(-0.4, 0.7, 8, dtype=delta.dtype)[:, None]
        verification = self.forecast.forecast_dbz - row_error
        impact = self._impact(delta, verification)
        channels = impact.fso.observation
        background = channels.initial_background_dbz.maps[0, 0]
        self.assertGreater(float(background[0, 3, 3].abs()), 1.0e-8)
        self.assertGreater(
            float((background[0, 3, 3] - background[0, 0, 3]).abs()), 1.0e-8,
        )
        # P(Y)=Y[0] with constant prior std; fallback is also the identity.
        # This uses the background-field derivative, not the VJP's own input.
        expected = (
            channels.detected_dbz.maps[0, 0]
            + channels.baseline_dynamics_dbz.maps[0, 0]
            + background
        )
        torch.testing.assert_close(
            channels.frozen_structure_input_dbz.maps[0, 0], expected,
            rtol=1.0e-8, atol=1.0e-10,
        )
        torch.testing.assert_close(
            impact.observation.initial_background_dbz.maps[0, 0],
            background * delta,
            rtol=1.0e-8, atol=1.0e-10,
        )
        torch.testing.assert_close(
            impact.observation.total.sum_by_time[0, 0].sum(),
            (expected * delta).sum(), rtol=1.0e-8, atol=1.0e-10,
        )
        standalone = compute_variational_fso(
            self.forecast,
            self.analysis,
            verification,
            sensitivity_config=self.fixture.sensitivity_config,
            adjoint_config=VariationalAdjointConfig(
                maximum_gauss_newton_relative_curvature_defect=1.0e9,
            ),
            neural_prior_runner=self.runner,
            neural_prior_application=self.application,
        )
        torch.testing.assert_close(
            standalone.observation.frozen_structure_input_dbz.maps[0, 0],
            expected, rtol=1.0e-8, atol=1.0e-10,
        )

    def test_composed_quadratic_oracle_reaches_nonfirst_prior_rows(self) -> None:
        # Exercise the actual P1 caller/runner with an analytically solved
        # background subproblem: J(c,B)=||c-B||²/2, c*=B,
        # E(c,B)=.25<G,B>+.75<G,c>. Thus the full background gradient is G.
        # The other P1 score paths remain real; only this subproblem is replaced.
        linearization = self.analysis.linearization
        assert linearization is not None
        frozen = linearization.frozen
        observations = linearization.observations
        prior_valid = frozen.neural_prior_valid_mask
        assert prior_valid is not None
        active = prior_valid | (
            observations.valid_mask[0] & frozen.observed_mask[0] & ~prior_valid
        )
        self.assertTrue(bool(active[2, 3]))
        delta = torch.zeros_like(self.fixture.frames)
        delta[0, 3, 3] = 0.01
        delta[0, 2, 3] = -0.02
        verification = self.forecast.forecast_dbz - 0.5
        gradient = torch.arange(1.0, 65.0, dtype=delta.dtype).reshape(8, 8)
        gradient[3, 3] = -7.0
        zero_first = gradient.clone()
        zero_first[0] = 0.0
        original_helper = sensitivity._frozen_initial_background_observation_sensitivity

        for field in (gradient, zero_first, gradient.flip(0)):
            with self.subTest(zero_first_row=bool((field[0] == 0).all()), prior_value=float(field[3, 3])):
                def quadratic_helper(*args, **kwargs):
                    background = frozen.initial_background_dbz

                    def residual(control, obs, state):
                        return (control - state.initial_background_dbz).reshape(-1)

                    def score(control, state, *unused):
                        return (field * (0.25 * state.initial_background_dbz + 0.75 * control)).sum()

                    with patch.object(sensitivity, "residual_vector", residual), patch.object(
                        sensitivity, "_variational_forecast_score", score,
                    ):
                        result = original_helper(
                            0.75 * field, background, observations, frozen, **kwargs,
                        )
                    torch.testing.assert_close(result[1], field, rtol=0.0, atol=0.0)
                    self.assertEqual(result[0].shape, self.fixture.frames.shape)
                    self.assertEqual(result[1].shape, background.shape)
                    return result

                with patch.object(
                    sensitivity, "_frozen_initial_background_observation_sensitivity",
                    quadratic_helper,
                ):
                    impact = self._impact(delta, verification)
                expected_map = torch.zeros_like(delta)
                expected_map[0] = torch.where(active, field, 0.0) * delta[0]
                torch.testing.assert_close(
                    impact.observation.initial_background_dbz.maps[0, 0],
                    expected_map, rtol=1.0e-9, atol=1.0e-11,
                )

                def composed_score(frames):
                    background = torch.where(active, frames[0], frozen.initial_background_dbz)
                    optimum = background  # Exact minimizer of the quadratic J.
                    return (field * (0.25 * background + 0.75 * optimum)).sum()

                actual_change = (
                    composed_score(self.fixture.frames + delta)
                    - composed_score(self.fixture.frames - delta)
                ) / 2.0
                torch.testing.assert_close(
                    impact.observation.initial_background_dbz.sum_by_time[0, 0].sum(),
                    actual_change, rtol=1.0e-8, atol=1.0e-10,
                )
                torch.testing.assert_close(
                    impact.observation.initial_background_dbz.maps[0, 0, 0, 3, 3],
                    field[3, 3] * 0.01, rtol=1.0e-9, atol=1.0e-11,
                )


if __name__ == "__main__":
    unittest.main()
