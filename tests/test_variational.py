from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from collections.abc import Callable
from typing import cast

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
    DataStatus,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    TendencySource,
    forecast_from_state,
)
from advar.physics import (  # noqa: E402
    RemapCell,
    dbz_to_echo,
    echo_to_dbz,
    remap,
)
from advar.sensitivity import compute_sensitivity_snapshot  # noqa: E402
import advar.variational as variational_module  # noqa: E402
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


def advect(echo: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    return remap(echo, displacement)


def linear_to_dbz(
    echo: torch.Tensor,
    config: NowcastConfig,
) -> torch.Tensor:
    return echo_to_dbz(
        echo,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )


class VariationalAnalysisTests(unittest.TestCase):
    nowcast_config = NowcastConfig()
    analysis_config = AnalysisConfig(
        maximum_outer_iterations=5,
        maximum_pcg_iterations=50,
        pcg_relative_tolerance=1.0e-7,
    )

    def test_p1_run_lineage_covers_config_std_and_quality(self) -> None:
        frames = torch.full((3, 6, 6), 20.0, dtype=torch.float64)
        base_config = AnalysisConfig(
            maximum_outer_iterations=1,
            maximum_pcg_iterations=2,
        )
        changed_config = replace(base_config, pseudo_huber_delta=3.0)
        base, _ = variational_nowcast(
            frames,
            analysis_config=base_config,
        )
        variants = (
            variational_nowcast(
                frames,
                analysis_config=changed_config,
            )[0],
            variational_nowcast(
                frames,
                analysis_config=base_config,
                observation_std_dbz=3.0,
            )[0],
            variational_nowcast(
                frames,
                analysis_config=base_config,
                quality_weight=0.5,
            )[0],
        )

        self.assertIsNotNone(base.run.analysis_config_json)
        self.assertIsNotNone(base.run.analysis_config_digest)
        self.assertIsNotNone(base.run.analysis_input_digest)
        for variant in variants:
            self.assertNotEqual(
                variant.run.input_bundle_digest,
                base.run.input_bundle_digest,
            )
            self.assertNotEqual(
                variant.forecast_run_digest,
                base.forecast_run_digest,
            )

    def test_p1_run_preserves_grid_time_contract(self) -> None:
        frames = torch.full((3, 4, 4), 20.0, dtype=torch.float64)
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="e" * 64,
        )

        forecast, _ = variational_nowcast(
            frames,
            analysis_config=AnalysisConfig(
                maximum_outer_iterations=1,
                maximum_pcg_iterations=2,
            ),
            grid_time_contract=contract,
        )

        self.assertEqual(forecast.run.grid_time_contract, contract)
        self.assertEqual(
            forecast.run.grid_time_contract_digest,
            contract.digest,
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

    def active_field_position(
        self,
        frozen: FrozenOuterState,
        row: int,
        column: int,
    ) -> int:
        width = frozen.initial_background_dbz.shape[1]
        flat_index = row * width + column
        matches = torch.nonzero(
            frozen.active_field_index == flat_index,
            as_tuple=False,
        ).flatten()
        self.assertEqual(matches.numel(), 1)
        return int(matches[0])

    def test_three_observation_blocks_are_explicit(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(frozen)
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

    def test_detected_censored_and_once_whitened_residuals(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(frozen)

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

    def test_missing_qc_rejected_and_observed_clear_are_distinct(self) -> None:
        frames = torch.full((3, 4, 5), 20.0, dtype=torch.float64)
        frames[0, 0, 0] = torch.nan
        frames[1, 0, 1] = self.nowcast_config.min_dbz
        qc_mask = torch.ones_like(frames, dtype=torch.bool)
        qc_mask[2, 0, 2] = False

        observations, _ = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            qc_mask=qc_mask,
        )

        self.assertTrue(observations.missing_mask[0, 0, 0])
        self.assertFalse(observations.valid_mask[0, 0, 0])
        self.assertTrue(observations.observed_clear_mask[1, 0, 1])
        self.assertTrue(observations.valid_mask[1, 0, 1])
        self.assertTrue(observations.qc_rejected_mask[2, 0, 2])
        self.assertFalse(observations.valid_mask[2, 0, 2])

    def test_frozen_irls_matches_true_robust_gradient(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(frozen)
        control[-1] = 0.2
        frozen = freeze_irls_weights(control, observations, frozen)
        original_weight = frozen.irls_sqrt_weight.clone()
        residual_fn: Callable[[torch.Tensor], torch.Tensor] = lambda value: (
            residual_vector(
                value,
                observations,
                frozen,
            )
        )

        vjp_result = torch.func.vjp(residual_fn, control)
        residual = cast(torch.Tensor, vjp_result[0])
        pullback = cast(
            Callable[[torch.Tensor], tuple[torch.Tensor]],
            vjp_result[1],
        )
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
        control = initial_control(frozen)
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

    def test_solver_reuses_one_vjp_pullback_per_outer_iteration(self) -> None:
        observations, frozen = self.stationary_problem()
        changed_dbz = observations.dbz.clone()
        changed_dbz[1] -= 1.0
        changed = replace(observations, dbz=changed_dbz)
        original_vjp = torch.func.vjp
        vjp_calls = 0

        def counted_vjp(function, *primals, **kwargs):
            nonlocal vjp_calls
            vjp_calls += 1
            return original_vjp(function, *primals, **kwargs)

        with patch(
            "advar.variational.torch.func.vjp",
            side_effect=counted_vjp,
        ):
            result = solve_analysis(changed, frozen)

        self.assertGreater(result.outer_iterations, 0)
        self.assertGreater(result.pcg_iterations, 0)
        self.assertEqual(vjp_calls, result.outer_iterations)

    def test_returned_analysis_records_local_identifiability(self) -> None:
        coordinates = torch.arange(8, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        blob = -10.0 + 40.0 * torch.exp(
            -((y - 3.5).square() + (x - 3.5).square()) / 4.0
        )
        frames = torch.stack((blob, blob - 1.0, blob))
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )

        result = solve_analysis(observations, frozen)

        self.assertFalse(result.used_fallback, result.reason)
        self.assertIsNotNone(result.dynamics_reduced_hessian_eigenvalues)
        self.assertIsNotNone(
            result.dynamics_reduced_hessian_condition_number
        )
        eigenvalues = result.dynamics_reduced_hessian_eigenvalues
        assert eigenvalues is not None
        condition_number = result.dynamics_reduced_hessian_condition_number
        assert condition_number is not None
        diagnostic_frozen = freeze_irls_weights(
            result.control,
            observations,
            frozen,
        )
        field_size = diagnostic_frozen.active_field_index.numel()

        def observation_residual(value: torch.Tensor) -> torch.Tensor:
            return (
                whitened_observation_residual(
                    value,
                    observations,
                    diagnostic_frozen,
                )
                * diagnostic_frozen.irls_sqrt_weight
            ).reshape(-1)

        dynamics_columns = []
        for dynamics_index in range(3):
            direction = torch.zeros_like(result.control)
            direction[field_size + dynamics_index] = 1.0
            jvp_result = torch.func.jvp(
                observation_residual,
                (result.control,),
                (direction,),
            )
            dynamics_columns.append(cast(torch.Tensor, jvp_result[1]))
        expected_hessian = torch.stack(
            tuple(
                torch.stack(
                    tuple(
                        torch.dot(left, right)
                        for right in dynamics_columns
                    )
                )
                for left in dynamics_columns
            )
        ) + torch.eye(3, dtype=torch.float64)
        expected_eigenvalues = torch.linalg.eigvalsh(expected_hessian)
        torch.testing.assert_close(
            torch.tensor(eigenvalues, dtype=torch.float64),
            expected_eigenvalues,
        )
        self.assertGreaterEqual(eigenvalues[0], 1.0)
        self.assertLessEqual(eigenvalues[0], eigenvalues[1])
        self.assertLessEqual(eigenvalues[1], eigenvalues[2])
        self.assertAlmostEqual(
            condition_number,
            eigenvalues[2] / eigenvalues[0],
        )
        self.assertIsNotNone(result.field_growth_jacobian_cosine)
        self.assertGreaterEqual(
            result.field_growth_jacobian_cosine or 0.0,
            0.0,
        )
        self.assertLessEqual(result.field_growth_jacobian_cosine or 0.0, 1.0)
        self.assertIsNotNone(result.field_motion_jacobian_cosine_yx)
        motion_cosines = result.field_motion_jacobian_cosine_yx
        assert motion_cosines is not None
        for cosine in motion_cosines:
            self.assertIsNotNone(cosine)
            assert cosine is not None
            self.assertGreaterEqual(cosine, 0.0)
            self.assertLessEqual(cosine, 1.0)

    def test_ad_hot_path_has_no_boundary_validation(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(frozen)
        direction = torch.ones_like(control)
        residual_fn = lambda value: residual_vector(
            value,
            observations,
            frozen,
        )

        with patch(
            "advar.variational._validate_observations",
            side_effect=AssertionError("observation validation entered"),
        ), patch(
            "advar.variational.validate_physical_echo",
            side_effect=AssertionError("physical audit entered"),
        ):
            residual = residual_fn(control)
            product = gauss_newton_hvp(
                residual_fn,
                control,
                direction,
            )

        self.assertTrue(bool(torch.all(torch.isfinite(residual))))
        self.assertTrue(bool(torch.all(torch.isfinite(product))))

    def test_public_trajectory_refreezes_stale_remap_cells(self) -> None:
        observations, frozen = self.stationary_problem()
        control = initial_control(frozen)
        expected = analysis_trajectory(control, frozen)
        stale = replace(
            frozen,
            analysis_remap_cells=(
                RemapCell(4, 4),
                RemapCell(-4, -4),
            ),
        )

        actual = analysis_trajectory(control, stale)

        torch.testing.assert_close(
            actual.frames_linear,
            expected.frames_linear,
        )
        torch.testing.assert_close(
            actual.displacement_yx,
            expected.displacement_yx,
        )
        torch.testing.assert_close(
            actual.log_growth_per_step,
            expected.log_growth_per_step,
        )

    def test_solver_rejects_stale_active_field_index_before_evaluation(
        self,
    ) -> None:
        observations, frozen = self.stationary_problem()
        frozen = replace(
            frozen,
            active_field_index=torch.tensor(
                [0],
                dtype=torch.long,
                device=frozen.initial_background_dbz.device,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "active_field_index must enumerate initial support",
        ):
            solve_analysis(observations, frozen)

    def test_analysis_operator_has_gradient_above_output_cap(self) -> None:
        observations, frozen = self.stationary_problem(value_dbz=70.0)
        control = initial_control(frozen)
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
        zero = initial_control(frozen)
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
            echo_linear=baseline.echo_linear,
            displacement_yx=torch.zeros_like(baseline.displacement_yx),
            log_growth_per_step=torch.zeros_like(
                baseline.log_growth_per_step
            ),
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

    def test_analysis_support_follows_analyzed_displacement(self) -> None:
        height, width = 6, 6
        frames = torch.full(
            (3, height, width),
            torch.nan,
            dtype=torch.float64,
        )
        frames[0, 1, 2] = 20.0
        background = torch.full_like(frames, torch.nan)
        background[0, 1, 4] = self.nowcast_config.min_dbz
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )
        control = initial_control(frozen)
        motion_limit = self.nowcast_config.max_displacement_px
        control[-3] = (
            motion_limit
            * torch.atanh(control.new_tensor(1.0 / motion_limit))
            / self.analysis_config.motion_increment_scale_px
        )

        result = variational_module._analysis_result(
            control,
            observations,
            frozen,
            1.0,
            0.5,
            1,
            0,
            True,
            "test_support_closure",
        )

        torch.testing.assert_close(
            result.state.displacement_yx,
            control.new_tensor((1.0, 0.0)),
            atol=1.0e-12,
            rtol=0.0,
        )
        displacement = 2.0 * result.state.displacement_yx
        observation_support = remap(
            frozen.observed_mask[0].to(dtype=control.dtype),
            displacement,
        )
        background_support = remap(
            frozen.background_mask[0].to(dtype=control.dtype),
            displacement,
        )
        expected_support = (
            observation_support
            + (1.0 - observation_support) * background_support
        )
        torch.testing.assert_close(
            result.metadata.source_support,
            expected_support,
        )
        self.assertTrue(result.metadata.background_used)
        self.assertAlmostEqual(
            result.metadata.background_contribution_fraction,
            0.5,
        )
        self.assertEqual(result.metadata.background_age_minutes, 10.0)

        run = ForecastRunContract.from_inputs(
            self.nowcast_config,
            frames,
            observations.valid_mask,
            background,
            10.0,
        )
        forecast = forecast_from_state(
            result.state,
            result.metadata,
            self.nowcast_config,
            run=run,
        )
        expected_valid = (
            remap(
                expected_support,
                result.state.displacement_yx,
            )
            >= self.nowcast_config.min_publish_support
        )
        torch.testing.assert_close(forecast.valid_mask[0], expected_valid)
        self.assertTrue(forecast.valid_mask[0, 4, 2])
        self.assertTrue(torch.isfinite(forecast.forecast_dbz[0, 4, 2]))

    def test_analysis_preserves_background_tendency_provenance(self) -> None:
        observations, frozen = self.stationary_problem()
        frozen = replace(
            frozen,
            background_age_minutes=10.0,
            baseline_metadata=replace(
                frozen.baseline_metadata,
                background_used=True,
                background_age_minutes=10.0,
                tendency_source=TendencySource.BACKGROUND,
            ),
        )

        result = variational_module._analysis_result(
            initial_control(frozen),
            observations,
            frozen,
            1.0,
            0.5,
            1,
            0,
            True,
            "test_background_tendency_provenance",
        )

        self.assertFalse(result.used_fallback)
        self.assertEqual(
            result.metadata.background_state_support_fraction,
            0.0,
        )
        self.assertTrue(result.metadata.background_tendency_used)
        self.assertTrue(result.metadata.background_used)
        self.assertEqual(result.metadata.background_age_minutes, 10.0)

    def test_causal_support_back_advects_later_detection(self) -> None:
        detected = torch.zeros((3, 7, 9), dtype=torch.bool)
        detected[2, 3, 6] = True
        observed = torch.zeros_like(detected)
        observed[0, 3, 4] = True

        support, seed = variational_module._causal_control_and_seed_support(
            detected,
            observed,
            torch.zeros_like(detected),
            torch.tensor((0.0, 1.0), dtype=torch.float64),
            self.analysis_config.minimum_control_reachability,
            0,
        )

        self.assertTrue(support[3, 4])
        self.assertEqual(int(support.sum()), 1)
        torch.testing.assert_close(seed, support)

    def test_tiny_causal_tail_does_not_open_control_support(self) -> None:
        detected = torch.zeros((3, 7, 9), dtype=torch.bool)
        detected[2, 3, 4] = True
        observed = torch.zeros_like(detected)
        observed[0, 3, 3] = True
        displacement = torch.tensor(
            (0.0, 0.999995),
            dtype=torch.float64,
        )
        precursor = remap(
            detected[2].to(dtype=displacement.dtype),
            -2.0 * displacement,
        )

        self.assertGreater(
            float(precursor[3, 3]),
            self.nowcast_config.epsilon,
        )
        self.assertLess(
            float(precursor[3, 3]),
            self.analysis_config.minimum_control_reachability,
        )
        support, seed = variational_module._causal_control_and_seed_support(
            detected,
            observed,
            torch.zeros_like(detected),
            displacement,
            self.analysis_config.minimum_control_reachability,
            0,
        )
        self.assertFalse(bool(torch.any(support)))
        self.assertFalse(bool(torch.any(seed)))

    def test_bilinear_quarter_weights_open_control_support(self) -> None:
        detected = torch.zeros((3, 7, 9), dtype=torch.bool)
        detected[2, 3, 4] = True
        observed = torch.zeros_like(detected)
        observed[0, 2:4, 3:5] = True

        support, seed = variational_module._causal_control_and_seed_support(
            detected,
            observed,
            torch.zeros_like(detected),
            torch.tensor((0.25, 0.25), dtype=torch.float64),
            self.analysis_config.minimum_control_reachability,
            0,
        )

        self.assertEqual(int(support.sum()), 4)
        torch.testing.assert_close(seed, support)

    def test_tiny_causal_tail_is_not_representable(self) -> None:
        _, frozen = self.stationary_problem(height=7, width=9)
        support = torch.zeros_like(frozen.initial_support_mask)
        support[3, 3] = True
        detected = torch.zeros_like(frozen.detected_masks)
        detected[2, 3, 4] = True
        frozen = replace(
            frozen,
            initial_support_mask=support,
            detected_masks=detected,
        )

        self.assertFalse(
            variational_module._analysis_window_is_representable(
                frozen,
                torch.tensor((0.0, 0.999995), dtype=torch.float64),
            )
        )

    def test_transient_intermediate_echo_must_be_representable(self) -> None:
        _, frozen = self.stationary_problem(height=7, width=9)
        support = torch.zeros_like(frozen.initial_support_mask)
        support[3, 3] = True
        detected = torch.zeros_like(frozen.detected_masks)
        detected[1, 3, 4] = True
        frozen = replace(
            frozen,
            initial_support_mask=support,
            detected_masks=detected,
        )

        self.assertFalse(
            variational_module._analysis_window_is_representable(
                frozen,
                torch.tensor((0.0, 2.0), dtype=torch.float64),
            )
        )

    def test_reachability_margin_changes_sign_at_threshold(self) -> None:
        _, frozen = self.stationary_problem(height=7, width=9)
        support = torch.zeros_like(frozen.initial_support_mask)
        support[3, 3] = True
        detected = torch.zeros_like(frozen.detected_masks)
        detected[1, 3, 4] = True
        frozen = replace(
            frozen,
            initial_support_mask=support,
            detected_masks=detected,
        )

        below = variational_module._analysis_window_reachability_margin(
            frozen,
            torch.tensor((0.0, 0.249999), dtype=torch.float64),
        )
        above = variational_module._analysis_window_reachability_margin(
            frozen,
            torch.tensor((0.0, 0.250001), dtype=torch.float64),
        )

        self.assertLess(below, 0.0)
        self.assertGreater(above, 0.0)

    def test_later_echo_opens_anchored_causal_control_support(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[:, 2, 2] = 20.0
        frames[0, 4, 6] = 4.9
        frames[1, 4, 6] = 12.0
        frames[2, 4, 6] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )

        self.assertFalse(observations.detected_mask[0, 4, 6])
        self.assertTrue(frozen.initial_support_mask[4, 6])

        control = initial_control(frozen)
        self.assertEqual(
            control.numel(),
            int(torch.count_nonzero(frozen.initial_support_mask)) + 3,
        )
        self.assertLess(
            control.numel(),
            frozen.initial_background_dbz.numel() + 3,
        )
        baseline = analysis_trajectory(control, frozen).frames_linear[2]
        changed = control.clone()
        changed[self.active_field_position(frozen, 4, 6)] = 1.0
        response = analysis_trajectory(changed, frozen).frames_linear[2]
        self.assertGreater(float(response[4, 6]), float(baseline[4, 6]))

    def test_causal_envelope_preserves_only_initial_anchors(self) -> None:
        detected = torch.zeros((3, 7, 9), dtype=torch.bool)
        detected[2, 3, 6] = True
        observed = torch.zeros_like(detected)
        observed[0, 3, 4] = True
        observed[0, 3, 6] = True

        support, seed = variational_module._causal_control_and_seed_support(
            detected,
            observed,
            torch.zeros_like(detected),
            torch.zeros(2, dtype=torch.float64),
            self.analysis_config.minimum_control_reachability,
            self.analysis_config.causal_support_dilation_px,
        )

        self.assertTrue(support[3, 4])
        self.assertTrue(support[3, 6])
        self.assertFalse(support[3, 5])
        self.assertEqual(int(support.sum()), 2)
        self.assertFalse(seed[3, 4])
        self.assertTrue(seed[3, 6])
        self.assertEqual(int(seed.sum()), 1)

        observations, frozen = self.stationary_problem(
            value_dbz=self.nowcast_config.min_dbz,
            height=7,
            width=9,
        )
        frozen = replace(
            frozen,
            initial_support_mask=support,
            active_field_index=torch.nonzero(
                support.flatten(),
                as_tuple=False,
            ).flatten(),
            causal_only_mask=support,
            causal_seed_mask=seed,
            detected_masks=detected,
        )
        warm = variational_module._warm_started_control(
            observations,
            frozen,
        )
        self.assertEqual(warm.numel(), 5)
        self.assertEqual(int(torch.count_nonzero(warm)), 1)
        self.assertEqual(
            float(warm[self.active_field_position(frozen, 3, 4)]),
            0.0,
        )
        self.assertGreater(
            float(warm[self.active_field_position(frozen, 3, 6)]),
            1.0,
        )
        control_count, seed_count, seed_cost = (
            variational_module._causal_seed_diagnostics(frozen)
        )
        self.assertEqual(control_count, 2)
        self.assertEqual(seed_count, 1)
        self.assertGreater(seed_cost, 0.0)

        margin = variational_module._analysis_window_reachability_margin(
            frozen,
            torch.tensor((0.0, 1.0), dtype=torch.float64),
        )
        self.assertAlmostEqual(
            margin,
            1.0 - self.analysis_config.minimum_control_reachability,
        )

    def test_floor_precursor_uses_prior_charged_warm_start(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[:, 2, 2] = 20.0
        frames[1, 4, 6] = 6.0
        frames[2, 4, 6] = 7.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            observation_std_dbz=0.5,
        )

        warm = variational_module._warm_started_control(
            observations,
            frozen,
        )
        warm_trajectory = analysis_trajectory(warm, frozen)
        warm_initial_dbz = echo_to_dbz(
            warm_trajectory.frames_linear[0],
            min_dbz=self.nowcast_config.min_dbz,
        )

        self.assertTrue(frozen.causal_only_mask[4, 6])
        self.assertTrue(frozen.causal_seed_mask[4, 6])
        self.assertGreater(
            abs(float(warm[self.active_field_position(frozen, 4, 6)])),
            1.0,
        )
        self.assertAlmostEqual(float(warm_initial_dbz[4, 6]), 4.0, places=6)
        self.assertGreater(float(torch.dot(warm, warm)), 0.0)
        zero = initial_control(frozen)
        zero_frozen = variational_module._freeze_analysis_remap_cells(
            zero,
            frozen,
        )
        reference_cost, _ = variational_module._evaluate_control(
            zero,
            observations,
            zero_frozen,
        )
        warm_cost, _ = variational_module._evaluate_control(
            warm,
            observations,
            frozen,
        )
        self.assertTrue(torch.isfinite(warm_cost))
        _, seed_count, seed_prior_cost = (
            variational_module._causal_seed_diagnostics(frozen)
        )
        self.assertEqual(seed_count, 1)
        self.assertAlmostEqual(
            seed_prior_cost,
            0.5 * float(torch.dot(warm, warm)),
        )

        result = solve_analysis(observations, frozen)

        self.assertFalse(result.used_fallback, result.reason)
        self.assertAlmostEqual(result.initial_objective, float(reference_cost))
        self.assertLess(result.final_objective, result.initial_objective)
        self.assertIsNotNone(result.minimum_reachability_margin)
        self.assertGreaterEqual(result.minimum_reachability_margin or 0.0, 0.0)
        self.assertGreater(result.causal_control_cell_count, 0)
        self.assertGreater(result.causal_seed_cell_count, 0)
        self.assertGreater(result.causal_seed_prior_cost, 0.0)
        torch.testing.assert_close(
            result.active_field_index,
            frozen.active_field_index,
        )
        self.assertEqual(
            result.control.numel(),
            result.active_field_index.numel() + 3,
        )
        self.assertEqual(
            result.amplitude_diagnostics_source,
            "returned_analysis",
        )

    def test_seed_warm_start_cannot_override_zero_control_reference(
        self,
    ) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[:, 2, 2] = 20.0
        frames[1, 4, 6] = 6.0
        frames[2, 4, 6] = 7.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            observation_std_dbz=3.0,
        )
        zero = initial_control(frozen)
        zero_frozen = variational_module._freeze_analysis_remap_cells(
            zero,
            frozen,
        )
        reference_cost, _ = variational_module._evaluate_control(
            zero,
            observations,
            zero_frozen,
        )
        warm = variational_module._warm_started_control(
            observations,
            frozen,
        )
        warm_cost, _ = variational_module._evaluate_control(
            warm,
            observations,
            frozen,
        )
        self.assertGreater(float(warm_cost), float(reference_cost))

        result = solve_analysis(observations, frozen)

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "no_improvement_over_zero_control")
        self.assertAlmostEqual(result.initial_objective, float(reference_cost))
        self.assertEqual(result.final_objective, result.initial_objective)
        self.assertEqual(int(torch.count_nonzero(result.control)), 0)
        self.assertEqual(result.causal_seed_cell_count, 1)

    def test_analysis_must_improve_zero_control_reference(self) -> None:
        observations, frozen = self.stationary_problem()

        result = variational_module._analysis_result(
            initial_control(frozen),
            observations,
            frozen,
            10.0,
            20.0,
            1,
            0,
            True,
            "synthetic_warm_start_improvement",
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "no_improvement_over_zero_control")
        self.assertEqual(result.initial_objective, 10.0)
        self.assertEqual(result.final_objective, 10.0)
        self.assertEqual(int(torch.count_nonzero(result.control)), 0)

    def test_later_detected_echo_is_not_published_as_clear(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[:, 2, 2] = 20.0
        frames[0, 4, 6] = 4.9
        frames[1, 4, 6] = 12.0
        frames[2, 4, 6] = 20.0

        forecast, result = variational_nowcast(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "unresolved_growth_or_emergence")
        self.assertIsNotNone(result.minimum_reachability_margin)
        self.assertGreaterEqual(result.minimum_reachability_margin or 0.0, 0.0)
        self.assertTrue(forecast.valid_mask[0, 4, 6])
        self.assertGreater(
            float(forecast.forecast_dbz[0, 4, 6]),
            self.analysis_config.detection_limit_dbz,
        )
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        zero = initial_control(frozen)
        zero_frozen = variational_module._freeze_analysis_remap_cells(
            zero,
            frozen,
        )
        reference_cost, _ = variational_module._evaluate_control(
            zero,
            observations,
            zero_frozen,
        )
        self.assertAlmostEqual(
            result.initial_objective,
            float(reference_cost),
        )
        self.assertAlmostEqual(
            result.final_objective,
            float(reference_cost),
        )
        self.assertEqual(int(torch.count_nonzero(result.control)), 0)
        self.assertIsNotNone(result.unresolved_amplitude_fraction)
        self.assertGreater(
            result.unresolved_amplitude_fraction or 0.0,
            self.analysis_config.maximum_unresolved_amplitude_fraction,
        )
        self.assertGreater(result.causal_control_cell_count, 0)
        self.assertEqual(result.causal_seed_cell_count, 0)
        self.assertEqual(result.causal_seed_prior_cost, 0.0)
        self.assertEqual(
            result.amplitude_diagnostics_source,
            "rejected_candidate",
        )

    def test_amplitude_fraction_tolerates_one_spatial_outlier(self) -> None:
        frames = torch.full((3, 32, 32), -10.0, dtype=torch.float64)
        frames[2] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        trajectory = analysis_trajectory(
            initial_control(frozen),
            frozen,
        )
        prediction_dbz = frames.clone()
        prediction_dbz[2, 15:18, 15:18] = 10.0
        trajectory = replace(
            trajectory,
            frames_linear=dbz_to_echo(
                prediction_dbz,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
        )

        fraction = variational_module._unresolved_amplitude_fraction(
            observations,
            frozen,
            trajectory,
        )

        self.assertAlmostEqual(fraction, 1.0 / (32.0 * 32.0))
        self.assertLess(
            fraction,
            self.analysis_config.maximum_unresolved_amplitude_fraction,
        )

    def test_low_quality_echo_does_not_hard_veto_amplitude(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[2, 3, 4] = 20.0
        quality = torch.ones_like(frames)
        quality[2, 3, 4] = 0.01
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            quality_weight=quality,
        )
        trajectory = analysis_trajectory(
            initial_control(frozen),
            frozen,
        )
        prediction_dbz = torch.full_like(frames, -10.0)
        prediction_dbz[2, 3, 4] = 10.0
        trajectory = replace(
            trajectory,
            frames_linear=dbz_to_echo(
                prediction_dbz,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
        )

        fraction = variational_module._unresolved_amplitude_fraction(
            observations,
            frozen,
            trajectory,
        )

        self.assertEqual(fraction, 0.0)

    def test_transient_intermediate_amplitude_is_checked(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[1, 3, 4] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        trajectory = analysis_trajectory(
            initial_control(frozen),
            frozen,
        )

        fraction = variational_module._unresolved_amplitude_fraction(
            observations,
            frozen,
            trajectory,
        )

        self.assertEqual(fraction, 1.0)

    def test_local_echo_closes_subpixel_amplitude_error(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[2, 3, 4] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        trajectory = analysis_trajectory(
            initial_control(frozen),
            frozen,
        )
        prediction_dbz = torch.full_like(frames, -10.0)
        prediction_dbz[2, 3, 5] = 20.0
        trajectory = replace(
            trajectory,
            frames_linear=dbz_to_echo(
                prediction_dbz,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
        )

        fraction = variational_module._unresolved_amplitude_fraction(
            observations,
            frozen,
            trajectory,
        )

        self.assertEqual(fraction, 0.0)

    def test_amplitude_fraction_is_gated_per_time(self) -> None:
        frames = torch.full((3, 32, 32), -10.0, dtype=torch.float64)
        frames[1, 0, :10] = 20.0
        frames[2].flatten()[:1000] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        prediction_dbz = frames.clone()
        prediction_dbz[1] = self.nowcast_config.min_dbz
        trajectory = replace(
            analysis_trajectory(initial_control(frozen), frozen),
            frames_linear=dbz_to_echo(
                prediction_dbz,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
        )

        diagnostics = variational_module._amplitude_diagnostics(
            observations,
            frozen,
            trajectory,
        )

        torch.testing.assert_close(
            diagnostics.unresolved_fraction_by_time,
            torch.tensor((1.0, 0.0), dtype=torch.float64),
        )
        self.assertEqual(
            float(diagnostics.maximum_unresolved_fraction),
            1.0,
        )
        self.assertEqual(
            variational_module._unresolved_amplitude_fraction(
                observations,
                frozen,
                trajectory,
            ),
            1.0,
        )

    def test_local_max_does_not_hide_integrated_echo_deficit(self) -> None:
        frames = torch.full((3, 9, 9), -10.0, dtype=torch.float64)
        frames[2, 3:6, 3:6] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        prediction_dbz = torch.full_like(frames, -10.0)
        prediction_dbz[2, 4, 4] = 20.0
        trajectory = replace(
            analysis_trajectory(initial_control(frozen), frozen),
            frames_linear=dbz_to_echo(
                prediction_dbz,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
        )

        diagnostics = variational_module._amplitude_diagnostics(
            observations,
            frozen,
            trajectory,
        )

        self.assertEqual(
            float(diagnostics.unresolved_fraction_by_time[1]),
            0.0,
        )
        self.assertAlmostEqual(
            float(diagnostics.integrated_echo_ratio_by_time[1]),
            1.0 / 9.0,
        )
        self.assertAlmostEqual(
            float(
                diagnostics
                .displacement_tolerant_soft_echo_area_ratio_by_time[1]
            ),
            1.0 / 9.0,
            places=5,
        )
        self.assertTrue(
            diagnostics.degrades_confidence(self.analysis_config)
        )

    def test_continuous_violation_allows_progress_before_threshold(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[2, 3, 4] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            observation_std_dbz=2.0,
        )

        def diagnostics(predicted_dbz: float):
            prediction = torch.full_like(frames, -10.0)
            prediction[2, 3, 4] = predicted_dbz
            trajectory = replace(
                analysis_trajectory(initial_control(frozen), frozen),
                frames_linear=dbz_to_echo(
                    prediction,
                    min_dbz=self.nowcast_config.min_dbz,
                    max_dbz=self.nowcast_config.max_dbz,
                ),
            )
            return variational_module._amplitude_diagnostics(
                observations,
                frozen,
                trajectory,
            )

        current = diagnostics(12.0)
        improved = diagnostics(13.6)
        worsened = diagnostics(11.0)

        self.assertEqual(float(current.maximum_unresolved_fraction), 1.0)
        self.assertEqual(float(improved.maximum_unresolved_fraction), 1.0)
        self.assertAlmostEqual(float(current.maximum_violation_score), 1.0)
        self.assertAlmostEqual(
            float(improved.maximum_violation_score),
            0.04,
        )
        self.assertTrue(
            variational_module._amplitude_trial_is_admissible(
                current,
                improved,
                self.analysis_config.maximum_unresolved_amplitude_fraction,
                torch.float64,
            )
        )
        self.assertFalse(
            variational_module._amplitude_trial_is_admissible(
                current,
                worsened,
                self.analysis_config.maximum_unresolved_amplitude_fraction,
                torch.float64,
            )
        )

    def test_small_float32_violation_can_make_relative_progress(self) -> None:
        current = self._synthetic_amplitude_diagnostics(
            (1.0e-8, 0.0),
            dtype=torch.float32,
        )
        improved = self._synthetic_amplitude_diagnostics(
            (2.5e-9, 0.0),
            dtype=torch.float32,
        )

        self.assertTrue(
            variational_module._amplitude_trial_is_admissible(
                current,
                improved,
                0.01,
                torch.float32,
            )
        )

    def test_violation_merit_breaks_time_axis_maximum_ties(self) -> None:
        current = self._synthetic_amplitude_diagnostics((1.0, 1.0))
        improved = self._synthetic_amplitude_diagnostics((0.5, 1.0))

        self.assertTrue(
            variational_module._amplitude_trial_is_admissible(
                current,
                improved,
                0.01,
                torch.float64,
            )
        )

    def test_effective_pixel_count_is_quality_scale_invariant(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[2, 3, 4] = 20.0

        counts: list[float] = []
        for quality_value in (1.0, 1.0e-6):
            quality = torch.ones_like(frames)
            quality[2, 3, 4] = quality_value
            observations, frozen = prepare_analysis(
                frames,
                nowcast_config=self.nowcast_config,
                analysis_config=self.analysis_config,
                quality_weight=quality,
            )
            diagnostics = variational_module._amplitude_diagnostics(
                observations,
                frozen,
                analysis_trajectory(initial_control(frozen), frozen),
            )
            counts.append(
                float(diagnostics.effective_pixel_count_by_time[1])
            )

        self.assertEqual(counts, [1.0, 1.0])

    def test_low_absolute_quality_cannot_hard_veto_amplitude(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[2, 3, 4] = 20.0
        quality = torch.ones_like(frames)
        quality[2, 3, 4] = 1.0e-3
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            quality_weight=quality,
        )
        prediction = torch.full_like(frames, -10.0)
        trajectory = replace(
            analysis_trajectory(initial_control(frozen), frozen),
            frames_linear=dbz_to_echo(
                prediction,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
        )

        diagnostics = variational_module._amplitude_diagnostics(
            observations,
            frozen,
            trajectory,
        )

        self.assertEqual(
            float(diagnostics.unresolved_fraction_by_time[1]),
            1.0,
        )
        self.assertAlmostEqual(
            float(diagnostics.total_quality_weight_by_time[1]),
            1.0e-3,
        )
        self.assertFalse(
            bool(diagnostics.information_sufficient_by_time[1])
        )
        self.assertEqual(float(diagnostics.maximum_unresolved_fraction), 1.0)
        self.assertEqual(
            float(diagnostics.maximum_gated_unresolved_fraction),
            0.0,
        )
        self.assertTrue(diagnostics.has_insufficient_information)

        result = variational_module._analysis_result(
            initial_control(frozen),
            observations,
            frozen,
            1.0,
            0.5,
            1,
            0,
            True,
            "test_insufficient_amplitude_information",
        )
        self.assertFalse(result.used_fallback)
        self.assertTrue(result.degraded)
        self.assertTrue(result.insufficient_amplitude_information)
        self.assertEqual(
            result.amplitude_information_sufficient_by_time,
            (True, False),
        )

    def test_operational_policy_falls_back_on_insufficient_information(
        self,
    ) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[2, 3, 4] = 20.0
        quality = torch.ones_like(frames)
        quality[2, 3, 4] = 1.0e-3
        config = replace(
            self.analysis_config,
            amplitude_information_policy="operational_fallback",
        )
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=config,
            quality_weight=quality,
        )

        result = solve_analysis(observations, frozen)

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "insufficient_amplitude_information")
        self.assertTrue(result.insufficient_amplitude_information)
        self.assertEqual(result.final_objective, result.initial_objective)
        torch.testing.assert_close(
            result.control,
            torch.zeros_like(result.control),
        )

    def test_continuous_violation_does_not_double_weight_quality(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[2, 3, 3] = 20.0
        frames[2, 3, 5] = 20.0
        quality = torch.ones_like(frames)
        quality[2, 3, 5] = 0.25
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            observation_std_dbz=2.0,
            quality_weight=quality,
        )
        prediction = torch.full_like(frames, -10.0)
        trajectory = replace(
            analysis_trajectory(initial_control(frozen), frozen),
            frames_linear=dbz_to_echo(
                prediction,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
        )

        diagnostics = variational_module._amplitude_diagnostics(
            observations,
            frozen,
            trajectory,
        )

        effective_count = 1.25**2 / (1.0 + 0.25**2)
        expected = (
            12.0**2 + 7.0**2 + 4.5**2 + 3.5**2
        ) / effective_count
        self.assertAlmostEqual(
            float(diagnostics.violation_score_by_time[1]),
            expected,
        )

    def test_effective_pixel_threshold_can_disable_small_sample_veto(
        self,
    ) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[2, 3, 4] = 20.0
        config = replace(
            self.analysis_config,
            minimum_amplitude_effective_pixel_count=2.0,
        )
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=config,
        )

        diagnostics = variational_module._amplitude_diagnostics(
            observations,
            frozen,
            analysis_trajectory(initial_control(frozen), frozen),
        )

        self.assertEqual(
            float(diagnostics.effective_pixel_count_by_time[1]),
            1.0,
        )
        self.assertFalse(
            bool(diagnostics.information_sufficient_by_time[1])
        )
        self.assertEqual(float(diagnostics.maximum_unresolved_fraction), 1.0)
        self.assertEqual(
            float(diagnostics.maximum_gated_unresolved_fraction),
            0.0,
        )

    def test_established_echo_excess_growth_is_diagnosed(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[:, 3, 4] = torch.tensor((5.1, 5.1, 40.0))
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            observation_std_dbz=0.1,
        )
        diagnostics = variational_module._amplitude_diagnostics(
            observations,
            frozen,
            analysis_trajectory(initial_control(frozen), frozen),
        )

        torch.testing.assert_close(
            diagnostics.established_echo_excess_growth_fraction_by_time,
            torch.tensor((0.0, 1.0), dtype=torch.float64),
        )
        self.assertGreater(
            float(diagnostics.maximum_growth_envelope_ratio_by_time[1]),
            100.0,
        )
        self.assertEqual(
            float(diagnostics.maximum_unresolved_fraction),
            0.0,
        )

        result = variational_module._analysis_result(
            initial_control(frozen),
            observations,
            frozen,
            1.0,
            0.5,
            1,
            0,
            True,
            "test_established_growth_diagnostic",
        )
        self.assertFalse(result.used_fallback)
        self.assertTrue(result.degraded)
        self.assertEqual(
            result.established_echo_excess_growth_fraction,
            1.0,
        )
        self.assertEqual(
            result.established_echo_excess_growth_fraction_by_time,
            (0.0, 1.0),
        )
        self.assertIsNotNone(result.maximum_growth_envelope_ratio)

    def test_established_echo_inside_growth_envelope_is_not_excess(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[:, 3, 4] = torch.tensor((10.0, 11.0, 12.0))
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            observation_std_dbz=0.1,
        )
        diagnostics = variational_module._amplitude_diagnostics(
            observations,
            frozen,
            analysis_trajectory(initial_control(frozen), frozen),
        )

        torch.testing.assert_close(
            diagnostics.established_echo_excess_growth_fraction_by_time,
            torch.zeros(2, dtype=torch.float64),
        )
        self.assertLessEqual(
            float(torch.max(diagnostics.maximum_growth_envelope_ratio_by_time)),
            1.0,
        )

    def _synthetic_amplitude_diagnostics(
        self,
        violation_by_time: tuple[float, float],
        *,
        dtype: torch.dtype = torch.float64,
    ):
        zeros = torch.zeros(2, dtype=dtype)
        return variational_module._AmplitudeDiagnostics(
            unresolved_fraction_by_time=torch.ones(2, dtype=dtype),
            unresolved_pixel_fraction_by_time=torch.ones(2, dtype=dtype),
            violation_score_by_time=torch.tensor(
                violation_by_time,
                dtype=dtype,
            ),
            integrated_echo_ratio_by_time=zeros.clone(),
            displacement_tolerant_soft_echo_area_ratio_by_time=zeros.clone(),
            effective_pixel_count_by_time=zeros.clone(),
            bad_quality_weight_by_time=zeros.clone(),
            total_quality_weight_by_time=zeros.clone(),
            information_sufficient_by_time=torch.ones(2, dtype=torch.bool),
            established_echo_excess_growth_fraction_by_time=torch.full(
                (2,),
                math.nan,
                dtype=dtype,
            ),
            maximum_growth_envelope_ratio_by_time=torch.full(
                (2,),
                math.nan,
                dtype=dtype,
            ),
        )

    def test_unresolved_amplitude_fraction_must_be_bounded(self) -> None:
        for value in (-0.1, 1.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"must be in \[0, 1\]"):
                    AnalysisConfig(
                        maximum_unresolved_amplitude_fraction=value
                    )

    def test_amplitude_information_thresholds_must_be_positive(self) -> None:
        for field_name in (
            "minimum_amplitude_total_quality_weight",
            "minimum_amplitude_effective_pixel_count",
        ):
            for value in (0.0, -1.0, float("nan")):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, "must be positive"):
                        if (
                            field_name
                            == "minimum_amplitude_total_quality_weight"
                        ):
                            AnalysisConfig(
                                minimum_amplitude_total_quality_weight=value
                            )
                        else:
                            AnalysisConfig(
                                minimum_amplitude_effective_pixel_count=value
                            )

    def test_amplitude_policy_and_confidence_thresholds_are_validated(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "amplitude_information_policy",
        ):
            AnalysisConfig(
                amplitude_information_policy=cast(
                    variational_module.AmplitudeInformationPolicy,
                    "invalid",
                )
            )
        for field_name in (
            "minimum_integrated_echo_ratio_for_confidence",
            "minimum_soft_echo_area_ratio_for_confidence",
            "maximum_established_excess_growth_fraction_for_confidence",
        ):
            for value in (-0.1, 1.1, float("nan")):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"must be in \[0, 1\]",
                    ):
                        if field_name == (
                            "minimum_integrated_echo_ratio_for_confidence"
                        ):
                            AnalysisConfig(
                                minimum_integrated_echo_ratio_for_confidence=(
                                    value
                                )
                            )
                        elif field_name == (
                            "minimum_soft_echo_area_ratio_for_confidence"
                        ):
                            AnalysisConfig(
                                minimum_soft_echo_area_ratio_for_confidence=(
                                    value
                                )
                            )
                        else:
                            AnalysisConfig(
                                maximum_established_excess_growth_fraction_for_confidence=(
                                    value
                                )
                            )

    def test_latest_amplitude_threshold_constructor_remains_supported(
        self,
    ) -> None:
        config = AnalysisConfig(maximum_latest_detected_error_std=4.5)

        self.assertEqual(config.maximum_detected_error_std, 4.5)

    def test_unrepresentable_latest_echo_falls_back_to_p0(self) -> None:
        frames = torch.full((3, 7, 9), -10.0, dtype=torch.float64)
        frames[:, 2, 2] = 20.0
        frames[0, 4, 6] = 4.9
        frames[1, 4, 6] = 12.0
        frames[2, 4, 6] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        stale_support = torch.zeros_like(frozen.initial_support_mask)
        stale_support[2, 2] = True
        frozen = replace(
            frozen,
            initial_support_mask=stale_support,
            active_field_index=torch.nonzero(
                stale_support.flatten(),
                as_tuple=False,
            ).flatten(),
        )

        result = variational_module._analysis_result(
            initial_control(frozen),
            observations,
            frozen,
            1.0,
            0.5,
            1,
            0,
            True,
            "test_unrepresentable_echo",
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "unrepresentable_analysis_window")
        self.assertEqual(result.amplitude_diagnostics_source, "unavailable")
        torch.testing.assert_close(
            result.state.echo_linear,
            frozen.baseline_state.echo_linear,
        )
        self.assertGreater(
            float(
                echo_to_dbz(
                    result.state.echo_linear,
                    min_dbz=self.nowcast_config.min_dbz,
                )[4, 6]
            ),
            self.analysis_config.detection_limit_dbz,
        )

    def test_later_background_does_not_expand_p1_support(self) -> None:
        frames = torch.full((3, 7, 9), torch.nan, dtype=torch.float64)
        frames[0, 2, 2] = 20.0
        background = torch.full_like(frames, torch.nan)
        background[2, 4, 6] = self.nowcast_config.min_dbz
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        result = variational_module._analysis_result(
            initial_control(frozen),
            observations,
            frozen,
            1.0,
            0.5,
            1,
            0,
            True,
            "test_later_background_support",
        )

        self.assertFalse(result.used_fallback)
        self.assertEqual(float(result.metadata.source_support[4, 6]), 0.0)

    def test_infeasible_lm_trial_is_rejected_before_evaluation(self) -> None:
        observations, frozen = self.stationary_problem()
        changed_dbz = observations.dbz.clone()
        changed_dbz[1] -= 1.0
        changed = replace(observations, dbz=changed_dbz)
        frozen = replace(
            frozen,
            analysis_config=replace(
                self.analysis_config,
                maximum_outer_iterations=1,
                maximum_damping_retries=0,
            ),
        )
        evaluate = variational_module._evaluate_control

        with (
            patch(
                "advar.variational._analysis_window_is_representable",
                return_value=False,
            ) as representable,
            patch(
                "advar.variational._evaluate_control",
                wraps=evaluate,
            ) as evaluate_control,
        ):
            result = solve_analysis(changed, frozen)

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "no_accepted_step")
        self.assertEqual(representable.call_count, 12)
        self.assertEqual(evaluate_control.call_count, 1)

    def test_invalid_observations_without_background_are_unavailable(
        self,
    ) -> None:
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
        self.assertEqual(
            result.metadata.data_status,
            DataStatus.UNAVAILABLE,
        )
        self.assertTrue(bool(torch.all(torch.isnan(forecast.forecast_dbz))))

    def test_missing_initial_frame_falls_back_to_time_aware_baseline(
        self,
    ) -> None:
        frames = torch.full((3, 8, 8), 25.0, dtype=torch.float64)
        frames[0] = torch.nan

        forecast, result = variational_nowcast(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "no_initial_state_support")
        self.assertEqual(result.active_field_index.numel(), 0)
        self.assertEqual(result.control.numel(), 3)
        self.assertGreater(float(result.state.echo_linear.sum()), 0.0)
        self.assertTrue(bool(torch.all(torch.isfinite(forecast.forecast_dbz))))
        torch.testing.assert_close(
            forecast.forecast_dbz[0],
            frames[-1],
            atol=0.02,
            rtol=0.0,
        )

    def test_invalid_observations_use_stale_background(self) -> None:
        frames = torch.full((3, 5, 5), 20.0, dtype=torch.float64)
        qc_mask = torch.zeros_like(frames, dtype=torch.bool)
        background = torch.full_like(frames, 25.0)
        forecast, result = variational_nowcast(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            qc_mask=qc_mask,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "no_valid_observations")
        self.assertEqual(
            result.metadata.data_status,
            DataStatus.STALE_BACKGROUND,
        )
        self.assertEqual(float(result.metadata.coverage_by_frame.mean()), 0.0)
        self.assertTrue(
            bool(torch.all(torch.isfinite(forecast.forecast_dbz)))
        )
        torch.testing.assert_close(forecast.forecast_dbz[0], background[-1])
        torch.testing.assert_close(
            echo_to_dbz(
                result.analyzed_frames_linear,
                min_dbz=self.nowcast_config.min_dbz,
            ),
            background,
        )

    def test_analysis_background_preserves_valid_observation(self) -> None:
        frames = torch.full((3, 5, 5), torch.nan, dtype=torch.float64)
        frames[0, 2, 2] = 15.0
        background = torch.full_like(frames, 25.0)

        _, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        self.assertEqual(float(frozen.initial_background_dbz[2, 2]), 15.0)
        self.assertEqual(float(frozen.initial_background_dbz[0, 0]), 25.0)

    def test_negative_analysis_trajectory_fails_closed(self) -> None:
        frames = torch.full((3, 16, 16), 20.0, dtype=torch.float64)
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        negative = -torch.ones(16, 16, dtype=torch.float64)
        with patch(
            "advar.variational.advance",
            return_value=negative,
        ):
            result = solve_analysis(observations, frozen)

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "positivity_violation")
        self.assertGreaterEqual(
            float(result.analyzed_frames_linear.min()),
            0.0,
        )
        self.assertGreaterEqual(result.audit.minimum_before_fix, 0.0)

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
        self.assertFalse(result.degraded)
        self.assertEqual(result.metadata.data_status, DataStatus.OBSERVED)

    def test_later_pcg_failure_preserves_accepted_analysis(self) -> None:
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
            analysis_config=replace(
                self.analysis_config,
                maximum_outer_iterations=2,
                maximum_damping_retries=0,
                gradient_tolerance=1.0e-12,
                step_tolerance=1.0e-12,
            ),
            observation_std_dbz=1.0,
        )
        baseline = frozen.baseline_state
        zero_motion = RadarState(
            echo_linear=baseline.echo_linear,
            displacement_yx=torch.zeros_like(baseline.displacement_yx),
            log_growth_per_step=torch.zeros_like(
                baseline.log_growth_per_step
            ),
        )
        frozen = replace(
            frozen,
            baseline_state=zero_motion,
            analysis_remap_cells=(RemapCell(0, 0), RemapCell(0, 0)),
        )
        calls = 0

        def first_real_then_fail(operator, rhs, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return matrix_free_pcg(operator, rhs, **kwargs)
            raise RuntimeError("synthetic later linear failure")

        with patch(
            "advar.variational.pcg",
            side_effect=first_real_then_fail,
        ):
            result = solve_analysis(observations, frozen)

        self.assertEqual(calls, 2)
        self.assertEqual(result.outer_iterations, 2)
        self.assertEqual(result.reason, "pcg_failed")
        self.assertFalse(result.converged)
        self.assertFalse(result.used_fallback)
        self.assertTrue(result.degraded)
        self.assertGreater(float(torch.linalg.vector_norm(result.control)), 0)
        self.assertLess(result.final_objective, result.initial_objective)
        self.assertEqual(
            result.metadata.provenance,
            "p1_variational_analysis",
        )

        final_frozen = freeze_irls_weights(
            result.control,
            observations,
            frozen,
        )
        expected = analysis_trajectory(result.control, final_frozen)
        torch.testing.assert_close(
            result.analyzed_frames_linear,
            expected.frames_linear,
        )
        torch.testing.assert_close(
            result.state.echo_linear,
            expected.frames_linear[-1],
        )
        self.assertAlmostEqual(
            result.final_objective,
            float(
                robust_objective(
                    result.control,
                    observations,
                    final_frozen,
                )
            ),
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
        frames[1] = 19.0
        forecast, analysis = variational_nowcast(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        verification = torch.full(
            (self.nowcast_config.forecast_steps, 4, 4),
            20.0,
            dtype=torch.float64,
        )

        self.assertFalse(analysis.used_fallback, analysis.reason)
        with self.assertRaisesRegex(ValueError, "requires a P0"):
            compute_sensitivity_snapshot(
                frames[-1],
                forecast,
                verification,
            )


if __name__ == "__main__":
    unittest.main()
