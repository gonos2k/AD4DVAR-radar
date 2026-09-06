"""Independent A5 oracles for posterior uncertainty, stationarity and M0."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_sensitivity import _current_verification_bundle, result_for  # noqa: E402

from advar.nowcast import (  # noqa: E402
    CURRENT_RADAR_METRIC_DOMAIN,
    CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
    NowcastConfig,
    RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
    RadarGridTimeContract,
    RadarState,
    radar_projected_crs_semantic_digest,
)
from advar.physics import dbz_to_echo  # noqa: E402
from advar.sensitivity import (  # noqa: E402
    SensitivityConfig,
    compute_sensitivity_snapshot,
)
from advar.variational import (  # noqa: E402
    AnalysisConfig,
    analysis_trajectory,
    residual_vector,
    robust_objective,
    variational_nowcast,
    whitened_observation_residual,
)
import advar.variational as variational_module  # noqa: E402


def _p1_case() -> tuple[
    torch.Tensor,
    NowcastConfig,
    AnalysisConfig,
    RadarGridTimeContract,
]:
    coordinates = torch.arange(8, dtype=torch.float64)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    blob = -10.0 + 40.0 * torch.exp(
        -((y - 3.5).square() + (x - 3.5).square()) / 4.0
    )
    frames = torch.stack((blob, blob - 1.0, blob))
    nowcast_config = NowcastConfig(horizon_minutes=20)
    analysis_config = AnalysisConfig(
        censored_background_policy="detection_limit",
        maximum_outer_iterations=5,
        maximum_pcg_iterations=50,
        pcg_relative_tolerance=1.0e-7,
    )
    grid = RadarGridTimeContract(
        valid_times=(
            "2026-08-04T00:00:00Z",
            "2026-08-04T00:10:00Z",
            "2026-08-04T00:20:00Z",
        ),
        dx_m=1000.0,
        dy_m=1000.0,
        projection="EPSG:5179",
        grid_hash="f" * 64,
    )
    return frames, nowcast_config, analysis_config, grid


def _current_grid() -> RadarGridTimeContract:
    return RadarGridTimeContract(
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
        grid_shape_yx=(4, 4),
        projected_crs_digest=radar_projected_crs_semantic_digest("EPSG:5179"),
        metric_domain_digest=CURRENT_RADAR_METRIC_DOMAIN.digest,
        metric_domain_evidence_digest=(
            CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
        ),
        cell_center_origin_xy_m=(1_000_000.0, 2_000_000.0),
        grid_coordinate_dtype=RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
        cell_center_convention=RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
    )


def _finite_difference_jacobian(
    function,
    point: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """Build a Jacobian without invoking the implementation's AD oracle."""

    columns: list[torch.Tensor] = []
    for index in range(point.numel()):
        direction = torch.zeros_like(point)
        direction[index] = step
        columns.append(
            (function(point + direction) - function(point - direction))
            / (2.0 * step)
        )
    return torch.stack(columns, dim=1)


def _independent_posterior_covariance(
    control: torch.Tensor,
    observations,
    frozen,
) -> torch.Tensor:
    """Reconstruct the Schur-complement posterior from finite differences."""

    field_size = frozen.active_field_index.numel()
    step = 1.0e-5

    def weighted_observation(value: torch.Tensor) -> torch.Tensor:
        return (
            whitened_observation_residual(value, observations, frozen)
            * frozen.irls_sqrt_weight
        ).reshape(-1)

    def full_residual(value: torch.Tensor) -> torch.Tensor:
        return residual_vector(value, observations, frozen)

    observation_jacobian = _finite_difference_jacobian(
        weighted_observation,
        control,
        step,
    )
    full_jacobian = _finite_difference_jacobian(
        full_residual,
        control,
        step,
    )
    dynamics = observation_jacobian[:, field_size:]
    field = observation_jacobian[:, :field_size]
    field_normal = full_jacobian[:, :field_size].mT @ full_jacobian[
        :, :field_size
    ]
    cross = field.mT @ dynamics
    data_gram = dynamics.mT @ dynamics
    conditioned = data_gram - cross.mT @ torch.linalg.solve(field_normal, cross)
    conditioned = 0.5 * (conditioned + conditioned.mT)
    regularized = conditioned + torch.eye(3, dtype=control.dtype)
    return torch.linalg.solve(regularized, torch.eye(3, dtype=control.dtype))


class A5UncertaintyOracleTests(unittest.TestCase):
    def test_physical_posterior_metadata_matches_independent_covariance(self) -> None:
        frames, nowcast_config, analysis_config, grid = _p1_case()
        frames_before = frames.clone()
        _, analysis = variational_nowcast(
            frames,
            nowcast_config=nowcast_config,
            analysis_config=analysis_config,
            grid_time_contract=grid,
        )

        self.assertFalse(analysis.used_fallback, analysis.reason)
        self.assertIsNotNone(analysis.linearization)
        retained = analysis.linearization
        assert retained is not None
        control = analysis.control
        observations = retained.observations
        frozen = retained.frozen
        field_size = frozen.active_field_index.numel()
        covariance = _independent_posterior_covariance(
            control,
            observations,
            frozen,
        )

        def physical_dynamics(dynamics: torch.Tensor) -> torch.Tensor:
            candidate = torch.cat((control[:field_size], dynamics))
            trajectory = analysis_trajectory(candidate, frozen)
            contract = frozen.grid_time_contract
            assert contract is not None
            velocity = contract.projected_velocity_xy(
                trajectory.displacement_yx,
                frozen.nowcast_config.interval_minutes,
            )
            return torch.cat(
                (velocity, trajectory.log_growth_per_step.reshape(1))
            )

        physical_jacobian = _finite_difference_jacobian(
            physical_dynamics,
            control[field_size:],
            1.0e-5,
        )
        physical_covariance = physical_jacobian @ covariance @ physical_jacobian.mT
        physical_covariance = 0.5 * (
            physical_covariance + physical_covariance.mT
        )
        velocity_variance = torch.linalg.eigvalsh(
            physical_covariance[:2, :2]
        )[-1]
        expected_velocity_uncertainty = torch.sqrt(velocity_variance.clamp_min(0.0))
        expected_growth_uncertainty = torch.sqrt(
            physical_covariance[2, 2].clamp_min(0.0)
        )

        self.assertGreater(float(torch.linalg.vector_norm(covariance)), 0.0)
        self.assertGreater(float(expected_velocity_uncertainty), 0.0)
        self.assertGreater(float(expected_growth_uncertainty), 0.0)
        self.assertGreater(float(torch.abs(physical_covariance[0, 2])), 0.0)
        torch.testing.assert_close(
            analysis.metadata.posterior_velocity_uncertainty_mps,
            expected_velocity_uncertainty,
            rtol=2.0e-4,
            atol=2.0e-6,
        )
        torch.testing.assert_close(
            analysis.metadata.posterior_log_growth_uncertainty_per_step,
            expected_growth_uncertainty,
            rtol=2.0e-4,
            atol=2.0e-6,
        )
        torch.testing.assert_close(frames, frames_before)

    def test_retained_linearization_matches_fresh_weighted_and_robust_gradients(
        self,
    ) -> None:
        frames, nowcast_config, analysis_config, grid = _p1_case()
        _, analysis = variational_nowcast(
            frames,
            nowcast_config=nowcast_config,
            analysis_config=analysis_config,
            grid_time_contract=grid,
        )
        self.assertFalse(analysis.used_fallback, analysis.reason)
        retained = analysis.linearization
        self.assertIsNotNone(retained)
        assert retained is not None
        control = analysis.control
        observations = retained.observations
        frozen = retained.frozen
        field_size = frozen.active_field_index.numel()

        weighted_cost = lambda value: 0.5 * torch.dot(
            residual_vector(value, observations, frozen),
            residual_vector(value, observations, frozen),
        )
        weighted_gradient = torch.func.grad(weighted_cost)(control)
        robust_gradient = torch.func.grad(
            lambda value: robust_objective(value, observations, frozen)
        )(control)

        def stats(gradient: torch.Tensor) -> tuple[float, float, float, float]:
            field = gradient[:field_size]
            dynamics = gradient[field_size:]
            field_rms = float(torch.sqrt(torch.mean(field.square())))
            field_max = float(torch.amax(torch.abs(field)))
            dynamics_max = float(torch.amax(torch.abs(dynamics)))
            return (
                float(torch.linalg.vector_norm(gradient)),
                field_rms,
                field_max,
                dynamics_max,
            )

        weighted_stats = stats(weighted_gradient)
        robust_stats = stats(robust_gradient)
        weighted_residual_norm = float(
            torch.linalg.vector_norm(residual_vector(control, observations, frozen))
        )
        self.assertTrue(all(math.isfinite(value) for value in weighted_stats))
        self.assertTrue(all(math.isfinite(value) for value in robust_stats))
        self.assertGreater(weighted_residual_norm, 0.0)

        for actual, expected in zip(
            (weighted_residual_norm, *weighted_stats),
            (
                retained.residual_norm,
                retained.gradient_norm,
                retained.field_gradient_rms,
                retained.field_gradient_max,
                retained.dynamics_gradient_max,
            ),
        ):
            self.assertTrue(
                math.isclose(
                    actual,
                    expected,
                    rel_tol=2.0e-6,
                    abs_tol=2.0e-9,
                )
            )
        for actual, expected in zip(
            robust_stats,
            (
                retained.robust_gradient_norm,
                retained.robust_field_gradient_rms,
                retained.robust_field_gradient_max,
                retained.robust_dynamics_gradient_max,
            ),
        ):
            self.assertTrue(
                math.isclose(
                    actual,
                    expected,
                    rel_tol=2.0e-6,
                    abs_tol=2.0e-9,
                )
            )
        self.assertTrue(
            math.isclose(
                max(weighted_stats[1], weighted_stats[3]),
                retained.relative_stationarity,
                rel_tol=2.0e-6,
                abs_tol=2.0e-9,
            )
        )
        self.assertTrue(
            math.isclose(
                max(robust_stats[1], robust_stats[3]),
                retained.robust_relative_stationarity,
                rel_tol=2.0e-6,
                abs_tol=2.0e-9,
            )
        )
        self.assertLessEqual(
            retained.robust_relative_stationarity,
            analysis_config.final_robust_relative_stationarity_tolerance,
        )
        self.assertLessEqual(
            retained.relative_stationarity,
            analysis_config.final_linearization_relative_stationarity_tolerance,
        )

    def test_m0_whitened_map_and_tile_norm_use_positive_known_std(self) -> None:
        config = NowcastConfig(horizon_minutes=10)
        latest = torch.full((4, 4), 20.0, dtype=torch.float64)
        latest_before = latest.clone()
        grid = _current_grid()
        state = RadarState(
            dbz_to_echo(latest, min_dbz=config.min_dbz, max_dbz=config.max_dbz),
            torch.zeros(2, dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
        )
        result = result_for(
            state,
            config,
            frames=torch.stack((latest, latest, latest)),
            grid_time_contract=grid,
        )
        truth = torch.full((1, 4, 4), 21.0, dtype=torch.float64)
        truth_before = truth.clone()
        verification = _current_verification_bundle(
            truth,
            valid_times=("2026-08-05T00:30:00Z",),
            grid_time_contract=grid,
        )
        known_std = torch.full_like(latest, 2.0)
        known_std_before = known_std.clone()
        sensitivity_config = SensitivityConfig(
            metric_names=("log_echo_mse",),
            metric_domain="issued",
            full_map_lead_minutes=(10,),
            tile_size=2,
            linearity_delta=(0.0, 0.0, 1.0e-4),
            require_verification_lineage=True,
        )
        snapshot = compute_sensitivity_snapshot(
            latest,
            result,
            verification,
            sensitivity_config=sensitivity_config,
            observation_std_dbz=known_std,
        )

        metric_weight = verification.metric_weight[0]
        valid = (
            verification.valid_mask[0]
            & result.valid_mask[0]
            & (metric_weight > 0.0)
        )
        weight = torch.where(valid, metric_weight, torch.zeros_like(metric_weight))
        floor = 10.0 ** (config.min_dbz / 10.0)
        truth_echo = dbz_to_echo(
            verification.frames_dbz[0],
            min_dbz=config.min_dbz,
            max_dbz=config.max_dbz,
        )
        residual = torch.log(result.state.echo_linear + floor) - torch.log(
            truth_echo + floor
        )
        expected_map = (
            2.0
            * (math.log(10.0) / 10.0)
            * residual
            * weight
            / weight.sum()
        )
        direct_map = snapshot.direct.maps[0, 0]
        self.assertTrue(bool(torch.all(known_std > 0.0)))
        self.assertGreater(float(torch.linalg.vector_norm(residual)), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(direct_map)), 0.0)
        torch.testing.assert_close(direct_map, expected_map, rtol=2.0e-6, atol=2.0e-9)
        torch.testing.assert_close(
            snapshot.direct.norm[0, 0],
            torch.linalg.vector_norm(expected_map),
            rtol=2.0e-6,
            atol=2.0e-9,
        )
        whitened_tiles = snapshot.direct.whitened_tile_norm
        self.assertIsNotNone(whitened_tiles)
        assert whitened_tiles is not None
        expected_tiles = torch.stack(
            tuple(
                torch.stack(
                    tuple(
                        2.0
                        * torch.linalg.vector_norm(
                            expected_map[
                                row : row + 2,
                                column : column + 2,
                            ]
                        )
                        for column in (0, 2)
                    )
                )
                for row in (0, 2)
            )
        )
        torch.testing.assert_close(
            whitened_tiles[0, 0],
            expected_tiles,
            rtol=2.0e-6,
            atol=2.0e-9,
        )
        torch.testing.assert_close(latest, latest_before)
        torch.testing.assert_close(truth, truth_before)
        torch.testing.assert_close(known_std, known_std_before)

    def test_posterior_saturation_guard_is_two_sided_at_sigma_boundary(self) -> None:
        """Exercise the predicate directly; full operational lineage is out of scope."""

        config = NowcastConfig(p1_posterior_saturation_sigma_multiplier=2.0)
        velocity_uncertainty = torch.tensor(0.25, dtype=torch.float64)
        growth_uncertainty = torch.tensor(0.02, dtype=torch.float64)
        motion_boundary = (
            config.p1_motion_saturation_safe_margin_mps
            + config.p1_posterior_saturation_sigma_multiplier
            * float(velocity_uncertainty)
        )
        growth_boundary = (
            config.p1_growth_saturation_safe_margin_per_step
            + config.p1_posterior_saturation_sigma_multiplier
            * float(growth_uncertainty)
        )
        above_motion = motion_boundary + 1.0

        def safe(motion: float | None, growth: float) -> bool:
            return variational_module._posterior_saturation_is_safe(
                motion,
                torch.tensor(growth, dtype=torch.float64),
                velocity_uncertainty,
                growth_uncertainty,
                config,
            )

        tolerance = config.contract_absolute_tolerance
        self.assertFalse(safe(motion_boundary - 2.0 * tolerance, growth_boundary + 1.0))
        self.assertTrue(safe(motion_boundary, growth_boundary + 1.0))
        self.assertFalse(safe(above_motion, growth_boundary - 2.0 * tolerance))
        self.assertTrue(safe(above_motion, growth_boundary))
        self.assertFalse(safe(None, growth_boundary + 1.0))
        self.assertFalse(
            variational_module._posterior_saturation_is_safe(
                above_motion,
                torch.tensor(growth_boundary, dtype=torch.float64),
                torch.tensor(float("nan"), dtype=torch.float64),
                growth_uncertainty,
                config,
            )
        )


if __name__ == "__main__":
    unittest.main()
