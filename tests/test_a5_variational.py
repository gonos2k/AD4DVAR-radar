"""Focused regression tests for the A5 variational boundaries."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import advar.variational as variational_module
from advar.nowcast import NowcastConfig, RadarGridTimeContract
from advar.variational import AnalysisConfig, prepare_analysis


class A5VariationalTests(unittest.TestCase):
    def test_exact_speed_boundary_exposes_inward_generalized_jvp(self) -> None:
        background = torch.tensor((30.0, 0.0), dtype=torch.float64)
        zero = torch.zeros_like(background)
        inward = torch.tensor((-1.0, 0.0), dtype=background.dtype)

        value, tangent = torch.func.jvp(
            lambda control: variational_module._bounded_vector_update(
                background,
                control,
                scale=2.0,
                limit=30.0,
            ),
            (zero,),
            (inward,),
        )

        torch.testing.assert_close(value, background, rtol=0.0, atol=0.0)
        torch.testing.assert_close(tangent, 2.0 * inward)
        # The forward projection remains feasible for an outward trial even
        # though the boundary uses the interior-side generalized Jacobian.
        outward = variational_module._bounded_vector_update(
            background,
            -inward,
            scale=2.0,
            limit=30.0,
        )
        self.assertLessEqual(float(torch.linalg.vector_norm(outward)), 30.0)

    def test_extreme_pseudo_huber_delta_retains_quadratic_cost_and_gradient(
        self,
    ) -> None:
        cases = ((torch.float32, 2.0e38), (torch.float64, 1.0e308))
        for dtype, delta in cases:
            with self.subTest(dtype=dtype, delta=delta):
                frames = torch.stack(
                    tuple(
                        torch.full((3, 3), value, dtype=dtype)
                        for value in (20.0, 21.0, 20.0)
                    )
                )
                observations, frozen = prepare_analysis(
                    frames,
                    analysis_config=AnalysisConfig(
                        pseudo_huber_delta=delta,
                    ),
                )
                control = variational_module.initial_control(frozen)
                residual = variational_module._whitened_observation_residual(
                    control,
                    observations,
                    frozen,
                )
                objective = variational_module.robust_objective(
                    control,
                    observations,
                    frozen,
                )
                gradient = torch.func.grad(
                    variational_module.robust_objective,
                    argnums=0,
                )(control, observations, frozen)

                expected = 0.5 * residual.to(torch.float64).square().sum()
                torch.testing.assert_close(
                    objective.to(torch.float64),
                    expected,
                    rtol=2.0e-6,
                    atol=2.0e-12,
                )
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)

    def test_pseudo_huber_cost_and_irls_have_independent_value_oracles(self) -> None:
        residual = torch.tensor(
            (-3.0, -0.25, 0.0, 0.25, 3.0),
            dtype=torch.float64,
            requires_grad=True,
        )
        delta = 2.0
        expected_cost = delta**2 * (
            torch.sqrt(1.0 + (residual / delta).square()) - 1.0
        )
        actual_cost = variational_module._pseudo_huber_cost(residual, delta)
        torch.testing.assert_close(actual_cost, expected_cost)
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: variational_module._pseudo_huber_cost(value, delta).sum(),
            (residual,),
        ))
        self.assertTrue(torch.autograd.gradgradcheck(
            lambda value: variational_module._pseudo_huber_cost(value, delta).sum(),
            (residual.detach().requires_grad_(),),
        ))
        split_residual = torch.tensor(
            (delta * (1.0 - 1.0e-4), delta * (1.0 + 1.0e-4)),
            dtype=torch.float64,
            requires_grad=True,
        )
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: variational_module._pseudo_huber_cost(value, delta).sum(),
            (split_residual,),
        ))
        self.assertTrue(torch.autograd.gradgradcheck(
            lambda value: variational_module._pseudo_huber_cost(value, delta).sum(),
            (split_residual.detach().requires_grad_(),),
        ))

        near_branch_delta = torch.tensor(2.0e19, dtype=torch.float32)
        near_branch_residual = torch.nextafter(
            near_branch_delta,
            torch.full_like(near_branch_delta, torch.inf),
        ).reshape(1).requires_grad_()
        near_branch_cost = variational_module._pseudo_huber_cost(
            near_branch_residual,
            float(near_branch_delta),
        ).sum()
        near_branch_gradient = torch.autograd.grad(
            near_branch_cost,
            near_branch_residual,
        )[0]
        self.assertTrue(bool(torch.isfinite(near_branch_cost)))
        self.assertTrue(bool(torch.isfinite(near_branch_gradient).all()))
        expected_near_branch_gradient = torch.tensor(
            2.0e19 / (2.0**0.5),
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            near_branch_gradient,
            expected_near_branch_gradient.reshape(1),
            rtol=3.0e-6,
            atol=0.0,
        )

        extreme_residual = torch.tensor(
            (0.0, 1.0e19, torch.finfo(torch.float32).max),
            dtype=torch.float32,
        )
        extreme_delta = 1.0e19
        actual_weight = variational_module._pseudo_huber_irls_sqrt_weight(
            extreme_residual,
            extreme_delta,
        )
        expected_weight = torch.pow(
            1.0
            + (
                extreme_residual.to(torch.float64)
                / extreme_delta
            ).square(),
            -0.25,
        ).to(torch.float32)
        torch.testing.assert_close(actual_weight, expected_weight, rtol=1.0e-5, atol=1.0e-12)
        self.assertTrue(bool(torch.isfinite(actual_weight).all()))

        underflowing_ratio_weight = variational_module._pseudo_huber_irls_sqrt_weight(
            torch.tensor((torch.finfo(torch.float32).max,), dtype=torch.float32),
            1.0e-30,
        )
        expected_underflowing_ratio_weight = torch.sqrt(
            torch.tensor(
                1.0e-30 / torch.finfo(torch.float32).max,
                dtype=torch.float64,
            )
        ).to(torch.float32)
        torch.testing.assert_close(
            underflowing_ratio_weight,
            expected_underflowing_ratio_weight.reshape(1),
            rtol=1.0e-5,
            atol=0.0,
        )
        self.assertGreater(float(underflowing_ratio_weight), 0.0)

    def test_subnormal_pseudo_huber_delta_is_rejected_at_public_dtype_boundary(
        self,
    ) -> None:
        frames = torch.full((3, 2, 2), 20.0, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "normal minimum"):
            prepare_analysis(
                frames,
                analysis_config=AnalysisConfig(pseudo_huber_delta=1.0e-40),
            )

    def test_physical_vector_scales_are_rejected_at_public_dtype_boundary(
        self,
    ) -> None:
        frames = torch.full((3, 2, 2), 20.0, dtype=torch.float32)
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "representable"):
            prepare_analysis(
                frames,
                nowcast_config=NowcastConfig(
                    maximum_motion_speed_mps=1.0,
                ),
                analysis_config=AnalysisConfig(
                    motion_increment_scale_mps=1.0e-40,
                ),
                grid_time_contract=contract,
            )

    def test_vector_update_rejects_unrepresentable_limit_and_ratio(self) -> None:
        background = torch.zeros(2, dtype=torch.float32)
        control = torch.zeros_like(background)
        with self.assertRaisesRegex(ValueError, "representable"):
            variational_module._bounded_vector_update(
                background,
                control,
                scale=1.0,
                limit=1.0e-40,
            )
        with self.assertRaisesRegex(ValueError, "ratio"):
            variational_module._bounded_vector_update(
                background,
                control,
                scale=1.0e20,
                limit=1.0e-20,
            )

    def test_extreme_pixel_footprints_are_clipped_to_grid(self) -> None:
        frames = torch.full((3, 2, 3), 20.0, dtype=torch.float64)
        _, frozen = prepare_analysis(
            frames,
            analysis_config=AnalysisConfig(
                causal_support_dilation_px=10**9,
                amplitude_displacement_tolerance_px=10**9,
            ),
        )

        self.assertEqual(frozen.amplitude_displacement_tolerance_yx, (1, 2))

    def test_extreme_common_bias_tile_is_one_bounded_full_grid_tile(self) -> None:
        frames = torch.full((3, 2, 3), 20.0, dtype=torch.float64)
        config = AnalysisConfig(
            observation_common_bias_std_dbz=1.0,
            observation_common_bias_tile_size_px=10**9,
        )
        observations, _ = prepare_analysis(frames, analysis_config=config)
        values = torch.arange(frames.numel(), dtype=frames.dtype).reshape_as(frames)

        whitened = variational_module._apply_observation_error_whitener(
            values,
            observations,
            config,
        )

        self.assertEqual(whitened.shape, frames.shape)
        self.assertTrue(bool(torch.isfinite(whitened).all()))


if __name__ == "__main__":
    unittest.main()
