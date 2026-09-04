"""Regression oracles for the September 2026 numerical review."""

import math
from fractions import Fraction
from importlib import import_module
import unittest

import torch

import advar.promotion as promotion_module
import advar.variational as variational_module
from advar.diagnostics import audit_transport
from advar.ensemble_sensitivity import EnsembleFSOStatistics, compute_ensemble_fso

nowcast_module = import_module("advar.nowcast")


class NumericalReviewTests(unittest.TestCase):
    def test_zero_background_velocity_derivatives(self) -> None:
        for control in ((0.0, 0.0), (0.3, -0.2)):
            with self.subTest(control=control):
                background = torch.zeros(2, dtype=torch.float64, requires_grad=True)
                increment = torch.tensor(control, dtype=torch.float64)

                def decode(value: torch.Tensor) -> torch.Tensor:
                    return variational_module._bounded_vector_update(
                        value, increment, scale=1.0, limit=10.0
                    )

                self.assertTrue(torch.autograd.gradcheck(decode, (background,)))
                self.assertTrue(torch.autograd.gradgradcheck(decode, (background,)))
                if control == (0.0, 0.0):
                    torch.testing.assert_close(
                        torch.func.jacrev(decode)(background),
                        torch.eye(2, dtype=torch.float64),
                    )

    def test_large_control_has_finite_velocity_jacobian(self) -> None:
        background = torch.zeros(2, dtype=torch.float32)
        control = torch.tensor([1e20, -1e20], dtype=torch.float32)

        def decode(value: torch.Tensor) -> torch.Tensor:
            return variational_module._bounded_vector_update(
                background, value, scale=1.0, limit=10.0
            )

        jacobian = torch.func.jacrev(decode)(control)
        self.assertTrue(bool(torch.isfinite(jacobian).all()))
        torch.testing.assert_close(
            decode(control),
            control.new_tensor([10 / math.sqrt(2), -10 / math.sqrt(2)]),
        )

    def test_small_float32_residual_cost(self) -> None:
        generator = torch.Generator().manual_seed(23)
        frames = torch.full((3, 4, 4), 20.0) + 0.0003 * torch.randn(
            (3, 4, 4), generator=generator
        )
        observations, frozen = variational_module.prepare_analysis(frames)
        control = variational_module.initial_control(frozen)
        residual = variational_module.whitened_observation_residual(
            control, observations, frozen
        ).to(torch.float64)
        delta = frozen.analysis_config.pseudo_huber_delta
        expected = (
            delta**2 * (torch.sqrt(1 + (residual / delta).square()) - 1)
        ).sum()
        cost = variational_module.robust_objective(control, observations, frozen)
        self.assertGreater(float(cost.detach()), 0.0)
        torch.testing.assert_close(
            cost.to(torch.float64), expected, rtol=1e-5, atol=1e-12
        )

    def test_small_float32_residual_solve(self) -> None:
        # Enough cells for the improvement to exceed the separate final
        # acceptance threshold; each residual still exercises cancellation.
        generator = torch.Generator().manual_seed(23)
        frames = torch.full((3, 16, 16), 20.0) + 0.0003 * torch.randn(
            (3, 16, 16), generator=generator
        )
        observations, frozen = variational_module.prepare_analysis(frames)
        result = variational_module.solve_analysis(observations, frozen)
        self.assertFalse(result.used_fallback)
        self.assertLess(result.final_objective, result.initial_objective)

    def test_valid_grid_cell_area_encloses_nominal_and_exact_value(self) -> None:
        matrix = ((1900.0, 700.0), (-1200.0, 100.0))
        grid = nowcast_module.RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=math.hypot(1900, 1200),
            dy_m=math.hypot(700, 100),
            projection="EPSG:5179",
            grid_hash="8" * 64,
            pixel_to_projected_matrix_m=matrix,
            spatial_grid_contract="radar-spatial-grid-identity-v6",
            grid_shape_yx=(2, 2),
            projected_crs_digest=nowcast_module.radar_projected_crs_semantic_digest(
                "EPSG:5179"
            ),
            metric_domain_digest=nowcast_module.CURRENT_RADAR_METRIC_DOMAIN.digest,
            metric_domain_evidence_digest=(
                nowcast_module.CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
            ),
            cell_center_origin_xy_m=(1_000_000.0, 2_000_000.0),
            grid_coordinate_dtype=nowcast_module.RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
            cell_center_convention=(
                nowcast_module.RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION
            ),
        )
        area = grid.cell_area_value_m2
        self.assertLessEqual(area.lower, area.nominal)
        self.assertGreaterEqual(area.upper, area.nominal)
        self.assertLessEqual(Fraction(area.lower), Fraction(1_030_000))
        self.assertGreaterEqual(Fraction(area.upper), Fraction(1_030_000))

    def test_truncated_gaussian_left_tail_value_and_gradients(self) -> None:
        location = torch.tensor([45.0], dtype=torch.float64, requires_grad=True)
        scale = torch.ones(1, dtype=torch.float64, requires_grad=True)

        def score(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            return promotion_module._truncated_gaussian_diagnostics(
                mean, std, mean.new_tensor([5.0]), support_threshold_dbz=0.0
            )[0]

        torch.testing.assert_close(
            score(location, scale), location.new_tensor([794.6334302644476])
        )
        self.assertTrue(torch.autograd.gradcheck(score, (location, scale)))

    def test_normal_interval_both_tails_have_finite_gradients(self) -> None:
        lower = torch.tensor(
            [-40.25, 39.75], dtype=torch.float64, requires_grad=True
        )
        upper = torch.tensor(
            [-39.75, 40.25], dtype=torch.float64, requires_grad=True
        )
        log_mass = promotion_module._standard_normal_log_interval_mass
        torch.testing.assert_close(
            log_mass(lower, upper),
            lower.new_full((2,), -794.6334302644476),
        )
        self.assertTrue(torch.autograd.gradcheck(log_mass, (lower, upper)))

    def test_ensemble_centering_uses_each_metric_scale(self) -> None:
        def statistics(projections: torch.Tensor) -> EnsembleFSOStatistics:
            return EnsembleFSOStatistics.from_diagonal_r(
                innovation=torch.ones(1),
                inverse_observation_variance=torch.ones(1),
                analysis_observation_perturbations=torch.tensor(
                    [[-1.0], [0.0], [1.0]]
                ),
                forecast_error_projection_by_member=projections,
                lead_minutes=(60,),
                metric_names=("centroid_error_m2", "log_echo_mse"),
                verification_reference_digest="3" * 64,
                observation_error_model_digest="4" * 64,
                observation_ids=("o0",),
                ensemble_member_ids=("m0", "m1", "m2"),
            )

        projections = torch.tensor([[[-1e8, 1.0]], [[0.0, 1.0]], [[1e8, 1.0]]])
        with self.assertRaisesRegex(ValueError, "centered"):
            statistics(projections)
        centered = projections - projections.mean(dim=0)
        result = compute_ensemble_fso(statistics(centered))
        self.assertEqual(float(result.total_impact[0, 1]), 0.0)
        self.assertEqual(float(result.total_impact_jackknife_std[0, 1]), 0.0)

    def test_float32_transport_without_boundary_echo_has_zero_outflow(self) -> None:
        generator = torch.Generator().manual_seed(0)
        echo = torch.zeros((512, 512), dtype=torch.float32)
        echo[8:-8, 8:-8] = 10 ** (7 * torch.rand((496, 496), generator=generator))
        audit = audit_transport(echo, torch.tensor([1.0, 1.0]))
        self.assertEqual(audit.boundary_outflow_integral, 0.0)
        self.assertLessEqual(
            audit.echo_budget_error / audit.echo_integral_before, 1e-14
        )

    def test_transport_outflow_counts_corner_once(self) -> None:
        echo = torch.zeros((4, 4), dtype=torch.float32)
        echo[-1, -1] = 8.0
        cases = (((0.5, 0.5), 6.0), ((5.0, 0.0), 8.0), ((0.0, -5.0), 8.0))
        for displacement, expected in cases:
            with self.subTest(displacement=displacement):
                audit = audit_transport(echo, torch.tensor(displacement))
                self.assertEqual(audit.boundary_outflow_integral, expected)
                self.assertEqual(audit.echo_budget_error, 0.0)
