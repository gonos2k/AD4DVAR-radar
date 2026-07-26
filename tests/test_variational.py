from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.matrix_free import (  # noqa: E402
    PCGResult,
    gauss_newton_hvp,
    jvp,
    pcg as matrix_free_pcg,
    vjp,
)
from advar.nowcast import (  # noqa: E402
    NowcastConfig,
    RadarState,
    RemapCell,
    advect,
    linear_to_dbz,
)
from advar.sensitivity import compute_sensitivity_snapshot  # noqa: E402
from advar.variational import (  # noqa: E402
    AnalysisConfig,
    FrozenOuterState,
    analysis_trajectory,
    freeze_irls_weights,
    initial_control,
    observation_residual_dbz,
    prepare_analysis,
    residual_vector,
    robust_objective,
    solve_analysis,
    variational_nowcast,
    whitened_observation_residual,
)


class VariationalAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nowcast_config = NowcastConfig()
        self.analysis_config = AnalysisConfig(
            maximum_outer_iterations=5,
            maximum_pcg_iterations=50,
            pcg_relative_tolerance=1.0e-7,
        )

    def stationary_problem(
        self,
        value_dbz: float = 20.0,
        *,
        height: int = 4,
        width: int = 5,
    ):
        frames = torch.full(
            (3, height, width),
            value_dbz,
            dtype=torch.float64,
        )
        return prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )

    def test_three_observation_blocks_are_explicit(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(observations)
        residual = observation_residual_dbz(
            control,
            observations,
            frozen,
        )
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            atol=1.0e-10,
            rtol=0.0,
        )

        changed_dbz = observations.dbz.clone()
        changed_dbz[1] -= 1.0
        changed = replace(observations, dbz=changed_dbz)
        changed_residual = observation_residual_dbz(
            control,
            changed,
            frozen,
        )
        torch.testing.assert_close(
            changed_residual[0],
            torch.zeros_like(changed_residual[0]),
            atol=1.0e-10,
            rtol=0.0,
        )
        torch.testing.assert_close(
            changed_residual[1],
            torch.ones_like(changed_residual[1]),
            atol=1.0e-10,
            rtol=0.0,
        )
        torch.testing.assert_close(
            changed_residual[2],
            torch.zeros_like(changed_residual[2]),
            atol=1.0e-10,
            rtol=0.0,
        )

    def test_legacy_frozen_outer_state_constructor_remains_valid(self) -> None:
        observations, frozen = self.stationary_problem()
        legacy = FrozenOuterState(
            frozen.initial_background_dbz,
            frozen.initial_support_mask,
            frozen.baseline_state,
            frozen.irls_sqrt_weight,
            frozen.nowcast_config,
            frozen.analysis_config,
        )

        self.assertIsNone(legacy.analysis_remap_cells)
        trajectory = analysis_trajectory(
            initial_control(observations),
            legacy,
        )
        self.assertEqual(trajectory.frames_linear.shape, (3, 4, 5))

    def test_detected_censored_and_once_whitened_residuals(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(observations)

        observed_dbz = observations.dbz.clone()
        observed_dbz[0] -= 2.0
        detected = replace(observations, dbz=observed_dbz)
        whitened = whitened_observation_residual(
            control,
            detected,
            frozen,
        )
        torch.testing.assert_close(
            whitened[0],
            torch.ones_like(whitened[0]),
            atol=1.0e-10,
            rtol=0.0,
        )

        doubled_std = replace(
            detected,
            std_dbz=2.0 * detected.std_dbz,
        )
        halved = whitened_observation_residual(
            control,
            doubled_std,
            frozen,
        )
        torch.testing.assert_close(
            halved[0],
            0.5 * whitened[0],
            atol=1.0e-10,
            rtol=0.0,
        )

        censored_mask = observations.censored_mask.clone()
        detected_mask = observations.detected_mask.clone()
        censored_mask[1:] = True
        detected_mask[1:] = False
        low_a = observations.dbz.clone()
        low_b = observations.dbz.clone()
        low_a[1:] = 0.0
        low_b[1:] = -5.0
        censored_a = replace(
            observations,
            dbz=low_a,
            detected_mask=detected_mask,
            censored_mask=censored_mask,
        )
        censored_b = replace(censored_a, dbz=low_b)
        residual_a = observation_residual_dbz(
            control,
            censored_a,
            frozen,
        )
        residual_b = observation_residual_dbz(
            control,
            censored_b,
            frozen,
        )
        torch.testing.assert_close(residual_a[1:], residual_b[1:])
        self.assertTrue(bool(torch.all(residual_a[1:] > 0)))

    def test_frozen_irls_matches_true_robust_gradient(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(observations)
        control[-1] = 0.2
        frozen = freeze_irls_weights(control, observations, frozen)
        original_weight = frozen.irls_sqrt_weight.clone()
        residual_fn = lambda value: residual_vector(
            value,
            observations,
            frozen,
        )

        residual, pullback = torch.func.vjp(residual_fn, control)
        irls_gradient = pullback(residual)[0]
        true_gradient = torch.func.grad(
            lambda value: robust_objective(
                value,
                observations,
                frozen,
            )
        )(control)
        torch.testing.assert_close(
            irls_gradient,
            true_gradient,
            atol=1.0e-9,
            rtol=1.0e-9,
        )

        direction = torch.linspace(
            -0.1,
            0.1,
            control.numel(),
            dtype=control.dtype,
        )
        gauss_newton_hvp(residual_fn, control, direction)
        torch.testing.assert_close(frozen.irls_sqrt_weight, original_weight)

    def test_residual_derivative_and_gauss_newton_contracts(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(observations)
        control[-3:] = torch.tensor(
            [0.05, -0.04, 0.03],
            dtype=control.dtype,
        )
        frozen = freeze_irls_weights(control, observations, frozen)
        function = lambda value: residual_vector(
            value,
            observations,
            frozen,
        )
        torch.manual_seed(21)
        direction = torch.randn_like(control)
        cotangent = torch.randn_like(function(control))
        _, tangent = jvp(function, control, direction)
        _, adjoint = vjp(function, control, cotangent)

        torch.testing.assert_close(
            torch.dot(tangent, cotangent),
            torch.dot(direction, adjoint),
            atol=1.0e-8,
            rtol=1.0e-8,
        )
        epsilon = 1.0e-5
        finite_difference = (
            function(control + epsilon * direction)
            - function(control - epsilon * direction)
        ) / (2.0 * epsilon)
        torch.testing.assert_close(
            tangent,
            finite_difference,
            atol=2.0e-7,
            rtol=2.0e-5,
        )

        second = torch.randn_like(control)
        hv = gauss_newton_hvp(function, control, direction)
        hw = gauss_newton_hvp(function, control, second)
        torch.testing.assert_close(
            torch.dot(direction, hw),
            torch.dot(hv, second),
            atol=1.0e-8,
            rtol=1.0e-8,
        )
        self.assertGreaterEqual(
            float(torch.dot(direction, hv)),
            -1.0e-10 * float(torch.dot(direction, direction)),
        )

    def test_analysis_operator_has_gradient_above_output_cap(self) -> None:
        observations, frozen = self.stationary_problem(value_dbz=70.0)
        control = initial_control(observations)
        control[0] = 2.0

        value = lambda candidate: observation_residual_dbz(
            candidate,
            observations,
            frozen,
        )[0, 0, 0]
        gradient = torch.func.grad(value)(control)
        self.assertGreater(float(value(control)), 0.0)
        self.assertGreater(abs(float(gradient[0])), 1.0e-6)

    def test_joint_analysis_reduces_manufactured_trajectory_error(self) -> None:
        height, width = 6, 6
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float64),
            torch.arange(width, dtype=torch.float64),
            indexing="ij",
        )
        initial = 2.0e4 * torch.exp(
            -((y - 2.7) ** 2 + (x - 3.1) ** 2) / 2.0
        )
        displacement = torch.tensor([0.45, -0.35], dtype=torch.float64)
        growth = torch.tensor(0.025, dtype=torch.float64)
        truth = torch.stack(
            (
                initial,
                advect(initial, displacement) * torch.exp(growth),
                advect(initial, 2.0 * displacement) * torch.exp(2.0 * growth),
            )
        )
        observed = linear_to_dbz(truth, self.nowcast_config)
        observed = observed.clone()
        observed[0, 2, 3] += 3.0
        observed[2, 3, 2] += 6.0

        observations, frozen = prepare_analysis(
            observed,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            observation_std_dbz=1.5,
        )
        zero = initial_control(observations)
        baseline = analysis_trajectory(zero, frozen)
        result = solve_analysis(observations, frozen)

        self.assertFalse(result.used_fallback, result.reason)
        self.assertLess(result.final_objective, result.initial_objective)
        baseline_error = torch.mean(
            (
                linear_to_dbz(
                    baseline.frames_linear,
                    self.nowcast_config,
                )
                - linear_to_dbz(truth, self.nowcast_config)
            )
            ** 2
        )
        analysis_error = torch.mean(
            (
                linear_to_dbz(
                    result.analyzed_frames_linear,
                    self.nowcast_config,
                )
                - linear_to_dbz(truth, self.nowcast_config)
            )
            ** 2
        )
        self.assertLess(float(analysis_error), float(baseline_error))
        torch.testing.assert_close(
            result.state.echo_linear,
            result.analyzed_frames_linear[-1],
        )

    def test_analysis_can_cross_zero_into_a_negative_remap_cell(self) -> None:
        height, width = 6, 6
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float64),
            torch.arange(width, dtype=torch.float64),
            indexing="ij",
        )
        initial = 2.0e4 * torch.exp(
            -((y - 2.7) ** 2 + (x - 3.1) ** 2) / 2.0
        )
        displacement = torch.tensor([-0.35, 0.0], dtype=torch.float64)
        truth = torch.stack(
            (
                initial,
                advect(initial, displacement),
                advect(initial, 2.0 * displacement),
            )
        )
        observations, frozen = prepare_analysis(
            linear_to_dbz(truth, self.nowcast_config),
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            observation_std_dbz=1.0,
        )
        baseline = frozen.baseline_state
        zero_motion = RadarState(
            echo_amplitude=baseline.echo_amplitude,
            displacement_yx=torch.zeros_like(baseline.displacement_yx),
            log_growth_per_step=torch.zeros_like(
                baseline.log_growth_per_step
            ),
            pair_displacements_yx=torch.zeros_like(
                baseline.pair_displacements_yx
            ),
            pair_log_growth=torch.zeros_like(baseline.pair_log_growth),
            provenance=baseline.provenance,
        )
        frozen = replace(
            frozen,
            baseline_state=zero_motion,
            analysis_remap_cells=(RemapCell(0, 0), RemapCell(0, 0)),
        )

        result = solve_analysis(observations, frozen)

        self.assertFalse(result.used_fallback, result.reason)
        self.assertLess(float(result.state.displacement_yx[0]), -0.1)
        self.assertLess(result.final_objective, result.initial_objective)

    def test_invalid_observations_use_qc_cleaned_baseline_fallback(self) -> None:
        frames = torch.full((3, 5, 5), 20.0, dtype=torch.float64)
        qc_mask = torch.zeros_like(frames, dtype=torch.bool)
        forecast, result = variational_nowcast(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            qc_mask=qc_mask,
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "no_valid_observations")
        torch.testing.assert_close(
            forecast,
            torch.full_like(forecast, self.nowcast_config.min_dbz),
        )

    def test_pcg_failure_uses_baseline_fallback(self) -> None:
        observations, frozen = self.stationary_problem()
        changed_dbz = observations.dbz.clone()
        changed_dbz[1] -= 1.0
        changed = replace(observations, dbz=changed_dbz)

        with patch(
            "advar.variational.pcg",
            side_effect=RuntimeError("synthetic linear failure"),
        ):
            result = solve_analysis(changed, frozen)

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "pcg_failed")
        torch.testing.assert_close(
            result.state.echo_linear,
            frozen.baseline_state.echo_linear,
        )

    def test_rejected_lm_trial_retries_with_more_damping(self) -> None:
        observations, frozen = self.stationary_problem()
        changed_dbz = observations.dbz.clone()
        changed_dbz[1] -= 1.0
        changed = replace(observations, dbz=changed_dbz)
        frozen = replace(
            frozen,
            analysis_config=replace(
                self.analysis_config,
                maximum_outer_iterations=1,
                maximum_damping_retries=2,
            ),
        )
        calls = 0

        def bad_then_real(operator, rhs, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return PCGResult(
                    solution=-rhs,
                    converged=True,
                    iterations=1,
                    relative_residual=0.0,
                )
            return matrix_free_pcg(operator, rhs, **kwargs)

        with patch("advar.variational.pcg", side_effect=bad_then_real):
            result = solve_analysis(changed, frozen)

        self.assertGreaterEqual(calls, 2)
        self.assertFalse(result.used_fallback, result.reason)
        self.assertLess(result.final_objective, result.initial_objective)

    def test_low_precision_frames_are_rejected_before_fft(self) -> None:
        frames = torch.full((3, 4, 4), 20.0, dtype=torch.float16)

        with self.assertRaisesRegex(TypeError, "float32 or float64"):
            prepare_analysis(frames)

    def test_m0_sensitivity_rejects_p1_analysis_state(self) -> None:
        frames = torch.full((3, 4, 4), 20.0, dtype=torch.float64)
        _, result = variational_nowcast(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        verification = torch.full(
            (self.nowcast_config.forecast_steps, 4, 4),
            20.0,
            dtype=torch.float64,
        )

        with self.assertRaisesRegex(ValueError, "requires a P0"):
            compute_sensitivity_snapshot(
                frames,
                result.state,
                verification,
                nowcast_config=self.nowcast_config,
            )


if __name__ == "__main__":
    unittest.main()
