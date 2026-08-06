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

from advar._digest import tensor_digest  # noqa: E402
from advar.matrix_free import (  # noqa: E402
    PCGResult,
    gauss_newton_hvp,
    jvp,
    pcg as matrix_free_pcg,
    vjp,
)
from advar.nowcast import (  # noqa: E402
    DataStatus,
    DynamicsSource,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    TendencyPairSelection,
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
        censored_background_policy="detection_limit",
        maximum_outer_iterations=5,
        maximum_pcg_iterations=50,
        pcg_relative_tolerance=1.0e-7,
    )

    def test_block_stationarity_is_resolution_stable(self) -> None:
        field = torch.tensor((2.0, -2.0), dtype=torch.float64)
        dynamics = torch.tensor((0.1, -0.2, 0.3), dtype=torch.float64)
        small = torch.cat((field, dynamics))
        large = torch.cat((field.repeat(16), dynamics))

        small_values = variational_module._block_stationarity(small, 2)
        large_values = variational_module._block_stationarity(large, 32)

        self.assertEqual(small_values, large_values)
        self.assertEqual(small_values, (2.0, 0.3, 2.0))

    def test_posterior_saturation_requires_uncertainty_clearance(self) -> None:
        config = NowcastConfig(
            p1_posterior_saturation_sigma_multiplier=2.0,
        )
        velocity_uncertainty = torch.tensor(0.1, dtype=torch.float64)
        growth_uncertainty = torch.tensor(0.01, dtype=torch.float64)
        growth_margin = torch.tensor(
            config.p1_growth_saturation_safe_margin_per_step + 0.03,
            dtype=torch.float64,
        )

        self.assertFalse(
            variational_module._posterior_saturation_is_safe(
                2.1,
                growth_margin,
                velocity_uncertainty,
                growth_uncertainty,
                config,
            )
        )
        self.assertTrue(
            variational_module._posterior_saturation_is_safe(
                2.3,
                growth_margin,
                velocity_uncertainty,
                growth_uncertainty,
                config,
            )
        )

    def test_censored_background_is_independent_of_storage_value(self) -> None:
        low = torch.full((3, 5, 6), -10.0, dtype=torch.float64)
        high = torch.full_like(low, 4.9)
        low[:, 2, 3] = 20.0
        high[:, 2, 3] = 20.0

        low_observations, low_frozen = prepare_analysis(
            low,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        high_observations, high_frozen = prepare_analysis(
            high,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )

        self.assertTrue(
            torch.equal(
                low_observations.censored_mask,
                high_observations.censored_mask,
            )
        )
        torch.testing.assert_close(
            low_frozen.initial_background_dbz,
            high_frozen.initial_background_dbz,
        )

    def test_default_censored_background_uses_clear_sky_floor(self) -> None:
        frames = torch.full((3, 4, 5), 4.9, dtype=torch.float64)
        frames[:, 1, 2] = 20.0

        _, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=AnalysisConfig(),
        )

        expected = torch.full_like(frames[0], self.nowcast_config.min_dbz)
        expected[1, 2] = 20.0
        torch.testing.assert_close(frozen.initial_background_dbz, expected)

    def test_external_censored_background_requires_complete_coverage(
        self,
    ) -> None:
        frames = torch.full((3, 4, 5), -10.0, dtype=torch.float64)
        config = replace(
            self.analysis_config,
            censored_background_policy="external_background",
        )
        with self.assertRaisesRegex(ValueError, "background coverage"):
            prepare_analysis(
                frames,
                nowcast_config=self.nowcast_config,
                analysis_config=config,
            )
        background = torch.full_like(frames, 1.5)
        _, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=config,
            background_frames_dbz=background,
            background_age_minutes=0.0,
        )
        torch.testing.assert_close(frozen.initial_background_dbz, background[0])

        high_background = torch.full_like(frames, 20.0)
        _, capped = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=config,
            background_frames_dbz=high_background,
            background_age_minutes=0.0,
        )
        detection_limit = torch.full_like(
            capped.initial_background_dbz,
            config.detection_limit_dbz,
        )
        self.assertTrue(
            bool(capped.initial_background_dbz.lt(detection_limit).all())
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

    def test_p1_confidence_uses_physical_posterior_uncertainty(self) -> None:
        coordinates = torch.arange(8, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        blob = -10.0 + 40.0 * torch.exp(
            -((y - 3.5).square() + (x - 3.5).square()) / 4.0
        )
        frames = torch.stack((blob, blob - 1.0, blob))
        contract = RadarGridTimeContract(
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

        forecast, analysis = variational_nowcast(
            frames,
            analysis_config=self.analysis_config,
            grid_time_contract=contract,
        )

        self.assertFalse(analysis.used_fallback, analysis.reason)
        self.assertTrue(
            bool(
                torch.isfinite(
                    analysis.metadata.posterior_velocity_uncertainty_mps
                )
            )
        )
        self.assertTrue(
            bool(
                torch.isfinite(
                    analysis.metadata
                    .posterior_log_growth_uncertainty_per_step
                )
            )
        )
        self.assertTrue(
            bool(
                torch.isfinite(
                    analysis.metadata.p1_velocity_saturation_uncertainty_mps
                )
            )
        )
        expected_velocity_uncertainty = torch.sqrt(
            analysis.metadata.posterior_velocity_uncertainty_mps.square()
            + analysis.metadata.posterior_velocity_uncertainty_mps.new_tensor(
                self.nowcast_config.forecast_velocity_uncertainty_mps**2
            )
            + analysis.metadata.p1_velocity_saturation_uncertainty_mps.square()
        )
        torch.testing.assert_close(
            forecast.forecast_velocity_uncertainty_mps,
            expected_velocity_uncertainty,
        )
        self.assertTrue(
            bool(torch.all(forecast.radar_dynamics_anchored_valid_mask))
        )

        stale_pair_diagnostics = replace(
            analysis.metadata,
            motion_disagreement_mps=torch.tensor(100.0),
            growth_disagreement=torch.tensor(100.0),
            maximum_growth_saturation_excess=torch.tensor(100.0),
        )
        changed = forecast_from_state(
            analysis.state,
            stale_pair_diagnostics,
            self.nowcast_config,
            run=forecast.run,
        )
        torch.testing.assert_close(
            changed.forecast_confidence,
            forecast.forecast_confidence,
        )

    def test_p1_without_physical_posterior_has_zero_confidence(self) -> None:
        coordinates = torch.arange(8, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        blob = -10.0 + 40.0 * torch.exp(
            -((y - 3.5).square() + (x - 3.5).square()) / 4.0
        )
        frames = torch.stack((blob, blob - 1.0, blob))

        forecast, analysis = variational_nowcast(
            frames,
            analysis_config=self.analysis_config,
        )

        self.assertFalse(analysis.used_fallback, analysis.reason)
        self.assertTrue(
            bool(
                torch.isnan(
                    analysis.metadata.posterior_velocity_uncertainty_mps
                )
            )
        )
        self.assertFalse(bool(torch.any(forecast.forecast_confidence)))
        self.assertFalse(
            bool(torch.any(forecast.radar_dynamics_anchored_valid_mask))
        )

    def stationary_problem(
        self,
        value_dbz: float = 20.0,
        *,
        height: int = 4,
        width: int = 5,
        analysis_config: AnalysisConfig | None = None,
    ):
        frames = torch.full(
            (3, height, width),
            value_dbz,
            dtype=torch.float64,
        )
        return prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=analysis_config or self.analysis_config,
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

    def test_common_bias_whitener_matches_dense_inverse_square_root(
        self,
    ) -> None:
        observations, frozen = self.stationary_problem(height=3, width=5)
        values = torch.linspace(
            -1.0,
            1.0,
            observations.dbz.numel(),
            dtype=observations.dbz.dtype,
        ).reshape_as(observations.dbz)

        unchanged = variational_module._apply_observation_error_whitener(
            values,
            observations,
            frozen.analysis_config,
        )
        self.assertIs(unchanged, values)
        unchanged_tiled = (
            variational_module._apply_observation_error_whitener(
                values,
                observations,
                replace(
                    frozen.analysis_config,
                    observation_common_bias_tile_size_px=2,
                ),
            )
        )
        self.assertIs(unchanged_tiled, values)

        for scope in ("per_frame", "all_times"):
            for tile_size in (0, 2):
                with self.subTest(scope=scope, tile_size=tile_size):
                    config = replace(
                        frozen.analysis_config,
                        observation_common_bias_std_dbz=1.5,
                        observation_common_bias_scope=scope,
                        observation_common_bias_tile_size_px=tile_size,
                    )
                    actual = (
                        variational_module._apply_observation_error_whitener(
                            values,
                            observations,
                            config,
                        )
                    )
                    mode = (
                        torch.sqrt(observations.quality_weight)
                        / observations.std_dbz
                    )
                    covariance = torch.eye(
                        values.numel(),
                        dtype=values.dtype,
                    )
                    height, width = values.shape[-2:]
                    spatial_size = max(height, width) if tile_size == 0 else tile_size
                    for row in range(0, height, spatial_size):
                        for column in range(0, width, spatial_size):
                            frame_groups = (
                                tuple((index,) for index in range(values.shape[0]))
                                if scope == "per_frame"
                                else (tuple(range(values.shape[0])),)
                            )
                            for frame_group in frame_groups:
                                bias_mode = torch.zeros_like(mode)
                                for frame in frame_group:
                                    bias_mode[
                                        frame,
                                        row : row + spatial_size,
                                        column : column + spatial_size,
                                    ] = mode[
                                        frame,
                                        row : row + spatial_size,
                                        column : column + spatial_size,
                                    ]
                                flattened_mode = bias_mode.flatten()
                                covariance = covariance + 1.5**2 * torch.outer(
                                    flattened_mode,
                                    flattened_mode,
                                )
                    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                    expected = eigenvectors @ (
                        torch.rsqrt(eigenvalues)
                        * (eigenvectors.mT @ values.flatten())
                    )
                    torch.testing.assert_close(
                        actual,
                        expected.reshape_as(values),
                        rtol=1.0e-12,
                        atol=1.0e-12,
                    )

        raw_groups = torch.tensor(
            (
                (10, 10, 20, 20, -1),
                (10, 10, 20, 30, 30),
                (40, 40, 20, 30, 30),
            ),
            dtype=torch.long,
        )
        for scope in ("per_frame", "all_times"):
            with self.subTest(group_scope=scope):
                canonical = (
                    variational_module._canonical_common_bias_group_index(
                        raw_groups,
                        frame_shape=tuple(observations.dbz.shape),
                        temporal_scope=scope,
                        device=observations.dbz.device,
                    )
                )
                grouped_observations = replace(
                    observations,
                    common_bias_group_index=canonical,
                )
                config = replace(
                    frozen.analysis_config,
                    observation_common_bias_std_dbz=1.5,
                    observation_common_bias_scope=scope,
                    observation_common_bias_group_map_digest=(
                        tensor_digest(canonical)
                    ),
                )
                actual = (
                    variational_module._apply_observation_error_whitener(
                        values,
                        grouped_observations,
                        config,
                    )
                )
                mode = (
                    torch.sqrt(observations.quality_weight)
                    / observations.std_dbz
                )
                covariance = torch.eye(
                    values.numel(),
                    dtype=values.dtype,
                )
                for group in torch.unique(canonical[canonical >= 0]):
                    bias_mode = torch.where(
                        canonical == group,
                        mode,
                        torch.zeros_like(mode),
                    ).flatten()
                    covariance = covariance + 1.5**2 * torch.outer(
                        bias_mode,
                        bias_mode,
                    )
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                expected = eigenvectors @ (
                    torch.rsqrt(eigenvalues)
                    * (eigenvectors.mT @ values.flatten())
                )
                torch.testing.assert_close(
                    actual,
                    expected.reshape_as(values),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )

        relabeled_groups = torch.where(
            raw_groups < 0,
            raw_groups,
            raw_groups * 7 + 3,
        )
        self.assertEqual(
            variational_module.observation_common_bias_group_map_digest(
                raw_groups,
                temporal_scope="all_times",
            ),
            variational_module.observation_common_bias_group_map_digest(
                relabeled_groups,
                temporal_scope="all_times",
            ),
        )

        x_fraction = torch.linspace(
            0.0,
            1.0,
            observations.dbz.shape[-1],
            dtype=observations.dbz.dtype,
        ).expand(observations.dbz.shape[-2], -1)
        mode_weights = torch.stack(
            (torch.sqrt(1.0 - x_fraction), torch.sqrt(x_fraction))
        )
        for scope in ("per_frame", "all_times"):
            with self.subTest(overlapping_scope=scope):
                canonical_weights = (
                    variational_module._canonical_common_bias_mode_weights(
                        mode_weights,
                        frame_shape=(
                            observations.dbz.shape[0],
                            observations.dbz.shape[1],
                            observations.dbz.shape[2],
                        ),
                        dtype=observations.dbz.dtype,
                        device=observations.dbz.device,
                    )
                )
                weighted_observations = replace(
                    observations,
                    common_bias_mode_weights=canonical_weights,
                )
                mode_digest = (
                    variational_module
                    .observation_common_bias_mode_weights_digest(mode_weights)
                )
                config = replace(
                    frozen.analysis_config,
                    observation_common_bias_std_dbz=1.5,
                    observation_common_bias_scope=scope,
                    observation_common_bias_mode_weights_digest=mode_digest,
                )
                actual = (
                    variational_module._apply_observation_error_whitener(
                        values,
                        weighted_observations,
                        config,
                    )
                )
                base_mode = (
                    torch.sqrt(observations.quality_weight)
                    / observations.std_dbz
                )
                covariance = torch.eye(
                    values.numel(),
                    dtype=values.dtype,
                )
                frame_groups = (
                    tuple((index,) for index in range(values.shape[0]))
                    if scope == "per_frame"
                    else (tuple(range(values.shape[0])),)
                )
                weights_by_frame = (
                    variational_module._common_bias_mode_weights_by_frame(
                        canonical_weights
                    )
                )
                for frame_group in frame_groups:
                    for mode_index in range(canonical_weights.shape[-3]):
                        bias_mode = torch.zeros_like(values)
                        for frame in frame_group:
                            weight_frame = (
                                0
                                if weights_by_frame.shape[0] == 1
                                else frame
                            )
                            bias_mode[frame] = (
                                1.5
                                * base_mode[frame]
                                * weights_by_frame[weight_frame, mode_index]
                            )
                        flattened_mode = bias_mode.flatten()
                        covariance = covariance + torch.outer(
                            flattened_mode,
                            flattened_mode,
                        )
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                expected = eigenvectors @ (
                    torch.rsqrt(eigenvalues)
                    * (eigenvectors.mT @ values.flatten())
                )
                torch.testing.assert_close(
                    actual,
                    expected.reshape_as(values),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )

        self.assertEqual(
            variational_module.observation_common_bias_mode_weights_digest(
                mode_weights
            ),
            variational_module.observation_common_bias_mode_weights_digest(
                mode_weights.clone()
            ),
        )

        precision = variational_module._observation_marginal_precision(
            observations,
            replace(
                frozen.analysis_config,
                observation_common_bias_std_dbz=1.5,
            ),
        )
        torch.testing.assert_close(
            precision,
            observations.quality_weight
            / (
                observations.std_dbz.square()
                + 1.5**2 * observations.quality_weight
            ),
        )
        grouped_precision = variational_module._observation_marginal_precision(
            replace(
                observations,
                common_bias_group_index=(
                    variational_module._canonical_common_bias_group_index(
                        raw_groups,
                        frame_shape=(
                            observations.dbz.shape[0],
                            observations.dbz.shape[1],
                            observations.dbz.shape[2],
                        ),
                        temporal_scope="all_times",
                        device=observations.dbz.device,
                    )
                ),
            ),
            replace(
                frozen.analysis_config,
                observation_common_bias_std_dbz=1.5,
            ),
        )
        torch.testing.assert_close(
            grouped_precision[:, 0, -1],
            observations.quality_weight[:, 0, -1]
            / observations.std_dbz[:, 0, -1].square(),
        )

    def test_overlapping_common_bias_factorization_is_frozen_once(
        self,
    ) -> None:
        frames = torch.full((3, 4, 4), 20.0, dtype=torch.float64)
        mode_weights = torch.full(
            (2, 4, 4),
            0.5,
            dtype=torch.float64,
        )
        original = (
            variational_module._low_rank_inverse_sqrt_correction_from_gram
        )
        with patch.object(
            variational_module,
            "_low_rank_inverse_sqrt_correction_from_gram",
            wraps=original,
        ) as correction:
            observations, frozen = prepare_analysis(
                frames,
                nowcast_config=NowcastConfig(),
                analysis_config=AnalysisConfig(
                    observation_common_bias_std_dbz=1.0
                ),
                observation_common_bias_mode_weights=mode_weights,
            )
            count_after_freeze = correction.call_count
            control = initial_control(frozen)
            residual_vector(control, observations, frozen)
            residual_vector(control, observations, frozen)

        self.assertEqual(count_after_freeze, 1)
        self.assertEqual(correction.call_count, 1)
        assert observations.common_bias_mode_weights is not None
        self.assertEqual(
            observations.common_bias_mode_weights.shape,
            (2, 4, 4),
        )
        self.assertEqual(
            frozen.observation_whitener.overlapping_correction.shape,
            (3, 2, 2),
        )

    def test_common_bias_storage_budgets_fail_closed(self) -> None:
        frames = torch.full((3, 4, 4), 20.0, dtype=torch.float64)
        mode_weights = torch.full(
            (2, 4, 4),
            0.5,
            dtype=torch.float64,
        )
        base = AnalysisConfig(observation_common_bias_std_dbz=1.0)
        with self.assertRaisesRegex(ValueError, "mode weights.*byte budget"):
            prepare_analysis(
                frames,
                analysis_config=replace(
                    base,
                    maximum_common_bias_mode_weight_bytes=1,
                ),
                observation_common_bias_mode_weights=mode_weights,
            )
        with self.assertRaisesRegex(ValueError, "whitener.*byte budget"):
            prepare_analysis(
                frames,
                analysis_config=replace(
                    base,
                    maximum_frozen_whitener_bytes=1,
                ),
                observation_common_bias_mode_weights=mode_weights,
            )
        with self.assertRaisesRegex(ValueError, "linearization.*byte budget"):
            prepare_analysis(
                frames,
                analysis_config=replace(
                    base,
                    maximum_linearization_bytes=1,
                ),
                observation_common_bias_mode_weights=mode_weights,
            )

    def test_common_bias_contract_is_validated_and_changes_lineage(self) -> None:
        for value in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "common_bias"):
                    AnalysisConfig(observation_common_bias_std_dbz=value)
        with self.assertRaisesRegex(ValueError, "common_bias_scope"):
            AnalysisConfig(
                observation_common_bias_scope=cast(
                    variational_module.ObservationCommonBiasScope,
                    "invalid",
                )
            )
        for value in (-1, True, 1.5):
            with self.subTest(tile_size=value):
                with self.assertRaisesRegex(ValueError, "tile_size"):
                    AnalysisConfig(
                        observation_common_bias_tile_size_px=cast(int, value)
                    )
        for digest in ("bad", "A" * 64, cast(str, 1)):
            with self.subTest(group_digest=digest):
                with self.assertRaisesRegex(ValueError, "group_map_digest"):
                    AnalysisConfig(
                        observation_common_bias_group_map_digest=digest
                    )
                with self.assertRaisesRegex(ValueError, "mode_weights_digest"):
                    AnalysisConfig(
                        observation_common_bias_mode_weights_digest=digest
                    )

        frames = torch.full((3, 4, 4), 20.0, dtype=torch.float64)
        diagonal, _ = variational_nowcast(
            frames,
            analysis_config=AnalysisConfig(maximum_outer_iterations=1),
        )
        correlated, _ = variational_nowcast(
            frames,
            analysis_config=AnalysisConfig(
                maximum_outer_iterations=1,
                observation_common_bias_std_dbz=1.0,
            ),
        )
        tiled, _ = variational_nowcast(
            frames,
            analysis_config=AnalysisConfig(
                maximum_outer_iterations=1,
                observation_common_bias_std_dbz=1.0,
                observation_common_bias_tile_size_px=2,
            ),
        )
        group_map = torch.tensor(
            (
                (0, 0, 1, 1),
                (0, 0, 1, 1),
                (2, 2, 3, 3),
                (2, 2, 3, 3),
            ),
            dtype=torch.long,
        )
        grouped, _ = variational_nowcast(
            frames,
            analysis_config=AnalysisConfig(
                maximum_outer_iterations=1,
                observation_common_bias_std_dbz=1.0,
            ),
            observation_common_bias_group_index=group_map,
        )
        mode_weights = torch.zeros((2, 4, 4), dtype=torch.float64)
        mode_weights[0, :, :2] = 1.0
        mode_weights[1, :, 2:] = 1.0
        overlapping, _ = variational_nowcast(
            frames,
            analysis_config=AnalysisConfig(
                maximum_outer_iterations=1,
                observation_common_bias_std_dbz=1.0,
            ),
            observation_common_bias_mode_weights=mode_weights,
        )
        self.assertNotEqual(
            diagonal.run.analysis_config_digest,
            correlated.run.analysis_config_digest,
        )
        self.assertNotEqual(
            diagonal.forecast_run_digest,
            correlated.forecast_run_digest,
        )
        self.assertNotEqual(
            correlated.run.analysis_config_digest,
            tiled.run.analysis_config_digest,
        )
        self.assertNotEqual(
            correlated.forecast_run_digest,
            tiled.forecast_run_digest,
        )
        self.assertNotEqual(
            correlated.run.analysis_config_digest,
            grouped.run.analysis_config_digest,
        )
        self.assertNotEqual(
            grouped.run.analysis_config_digest,
            overlapping.run.analysis_config_digest,
        )
        grouped_observations, grouped_frozen = prepare_analysis(
            frames,
            analysis_config=AnalysisConfig(
                observation_common_bias_std_dbz=1.0,
            ),
            observation_common_bias_group_index=group_map,
        )
        self.assertIsNotNone(grouped_observations.common_bias_group_index)
        expected_group_digest = (
            variational_module.observation_common_bias_group_map_digest(
                group_map
            )
        )
        assert grouped.run.analysis_config_json is not None
        self.assertIn(expected_group_digest, grouped.run.analysis_config_json)
        self.assertEqual(
            grouped_frozen.analysis_config
            .observation_common_bias_group_map_digest,
            expected_group_digest,
        )

        with self.assertRaisesRegex(ValueError, "requires positive"):
            prepare_analysis(
                frames,
                observation_common_bias_group_index=group_map,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            prepare_analysis(
                frames,
                analysis_config=AnalysisConfig(
                    observation_common_bias_std_dbz=1.0,
                    observation_common_bias_tile_size_px=2,
                ),
                observation_common_bias_group_index=group_map,
            )
        correlated_config = AnalysisConfig(
            observation_common_bias_std_dbz=1.0,
        )
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            prepare_analysis(
                frames,
                analysis_config=correlated_config,
                observation_common_bias_group_index=group_map.to(
                    dtype=torch.float64
                ),
            )
        invalid_groups = group_map.clone()
        invalid_groups[0, 0] = -2
        with self.assertRaisesRegex(ValueError, "-1 or nonnegative"):
            prepare_analysis(
                frames,
                analysis_config=correlated_config,
                observation_common_bias_group_index=invalid_groups,
            )
        with self.assertRaisesRegex(ValueError, "at least one group"):
            prepare_analysis(
                frames,
                analysis_config=correlated_config,
                observation_common_bias_group_index=torch.full_like(
                    group_map,
                    -1,
                ),
            )
        invalid_only_group = torch.full_like(group_map, -1)
        invalid_only_group[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "valid observation"):
            prepare_analysis(
                frames,
                analysis_config=correlated_config,
                observation_common_bias_group_index=invalid_only_group,
                qc_mask=torch.zeros_like(frames, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            prepare_analysis(
                frames,
                analysis_config=replace(
                    correlated_config,
                    observation_common_bias_group_map_digest="0" * 64,
                ),
                observation_common_bias_group_index=group_map,
            )
        with self.assertRaisesRegex(ValueError, "require positive"):
            prepare_analysis(
                frames,
                observation_common_bias_mode_weights=mode_weights,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            prepare_analysis(
                frames,
                analysis_config=correlated_config,
                observation_common_bias_group_index=group_map,
                observation_common_bias_mode_weights=mode_weights,
            )
        invalid_mode_weights = torch.ones_like(mode_weights)
        with self.assertRaisesRegex(ValueError, "sum to at most one"):
            prepare_analysis(
                frames,
                analysis_config=correlated_config,
                observation_common_bias_mode_weights=invalid_mode_weights,
            )
        mode_digest = (
            variational_module.observation_common_bias_mode_weights_digest(
                mode_weights
            )
        )
        prepared_modes, prepared_mode_frozen = prepare_analysis(
            frames,
            analysis_config=replace(
                correlated_config,
                observation_common_bias_mode_weights_digest=mode_digest,
            ),
            observation_common_bias_mode_weights=mode_weights,
        )
        self.assertIsNotNone(prepared_modes.common_bias_mode_weights)
        self.assertEqual(
            prepared_mode_frozen.analysis_config
            .observation_common_bias_mode_weights_digest,
            mode_digest,
        )

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
        self.assertTrue(observations.censored_mask[1, 0, 1])
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

    def test_field_smoothness_prior_uses_only_active_edges(self) -> None:
        _, frozen = self.stationary_problem()
        width = frozen.initial_background_dbz.shape[1]
        active_index = torch.tensor(
            (0, 1, width + 2),
            dtype=torch.int64,
        )
        active_mask = torch.zeros_like(frozen.initial_support_mask)
        active_mask.flatten()[active_index] = True
        left, right, physical_weight = (
            variational_module._active_smoothness_graph(
                active_mask,
                active_index,
                frozen.initial_background_dbz,
                None,
            )
        )
        frozen = replace(
            frozen,
            active_field_index=active_index,
            smooth_edge_left_index=left,
            smooth_edge_right_index=right,
            smooth_edge_physical_weight=physical_weight,
        )
        control = initial_control(frozen)
        control[:3] = torch.tensor(
            (0.0, 2.0, 100.0),
            dtype=control.dtype,
        )

        smoothness = variational_module._field_smoothness_residual(
            control,
            frozen,
        )

        self.assertEqual(smoothness.numel(), 1)
        torch.testing.assert_close(
            smoothness,
            torch.tensor(
                (
                    2.0
                    * math.sqrt(self.analysis_config.field_smoothness_weight),
                ),
                dtype=control.dtype,
            ),
        )

    def test_constant_active_field_has_zero_smoothness_cost(self) -> None:
        _, frozen = self.stationary_problem()
        control = initial_control(frozen)
        field_size = frozen.active_field_index.numel()
        control[:field_size] = 3.0

        residual = variational_module._field_smoothness_residual(
            control,
            frozen,
        )

        self.assertGreater(residual.numel(), 0)
        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_physical_smoothness_graph_weights_anisotropic_edges(self) -> None:
        active_mask = torch.ones((2, 2), dtype=torch.bool)
        active_index = torch.arange(4, dtype=torch.long)
        reference = torch.zeros((2, 2), dtype=torch.float64)
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=250.0,
            dy_m=2000.0,
            projection="EPSG:5179",
            grid_hash="a" * 64,
        )

        left, right, weight = variational_module._active_smoothness_graph(
            active_mask,
            active_index,
            reference,
            contract,
        )

        torch.testing.assert_close(left, torch.tensor((0, 1, 0, 2)))
        torch.testing.assert_close(right, torch.tensor((2, 3, 1, 3)))
        torch.testing.assert_close(
            weight,
            torch.tensor((0.125, 0.125, 8.0, 8.0), dtype=torch.float64),
        )

    def test_field_smoothness_rejects_sheared_grid(self) -> None:
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="c" * 64,
            pixel_to_projected_matrix_m=(
                (1000.0, 800.0),
                (0.0, -600.0),
            ),
        )

        with self.assertRaisesRegex(ValueError, "orthogonal"):
            prepare_analysis(
                frames,
                nowcast_config=self.nowcast_config,
                analysis_config=self.analysis_config,
                grid_time_contract=contract,
            )

        _, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=replace(
                self.analysis_config,
                field_smoothness_weight=0.0,
            ),
            grid_time_contract=contract,
        )
        self.assertFalse(contract.grid_axes_are_orthogonal)
        self.assertEqual(frozen.smooth_edge_left_index.numel(), 112)

    def test_projected_velocity_control_is_isotropic_on_anisotropic_grid(
        self,
    ) -> None:
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=250.0,
            dy_m=2000.0,
            projection="EPSG:5179",
            grid_hash="b" * 64,
            pixel_to_projected_matrix_m=(
                (0.0, -2000.0),
                (250.0, 0.0),
            ),
        )
        nowcast = replace(
            self.nowcast_config,
            maximum_motion_speed_mps=100.0,
            minimum_phase_correlation_psr=0.0,
        )
        config = replace(
            self.analysis_config,
            motion_increment_scale_mps=2.0,
        )
        _, frozen = prepare_analysis(
            frames,
            nowcast_config=nowcast,
            analysis_config=config,
            grid_time_contract=contract,
        )
        field_size = frozen.active_field_index.numel()
        control_x = initial_control(frozen)
        control_y = initial_control(frozen)
        control_x[field_size] = 0.5
        control_y[field_size + 1] = 0.5

        velocity_x = contract.projected_velocity_xy(
            analysis_trajectory(control_x, frozen).displacement_yx,
            nowcast.interval_minutes,
        )
        velocity_y = contract.projected_velocity_xy(
            analysis_trajectory(control_y, frozen).displacement_yx,
            nowcast.interval_minutes,
        )

        self.assertGreater(float(velocity_x[0]), 0.0)
        self.assertGreater(float(velocity_y[1]), 0.0)
        torch.testing.assert_close(velocity_x[1], velocity_x.new_zeros(()))
        torch.testing.assert_close(velocity_y[0], velocity_y.new_zeros(()))
        torch.testing.assert_close(velocity_x[0], velocity_y[1])

    def test_radial_velocity_control_is_rotation_equivariant(self) -> None:
        background = torch.zeros(2, dtype=torch.float64)
        axis_control = torch.tensor((20.0, 0.0), dtype=torch.float64)
        diagonal_control = torch.tensor(
            (20.0 / math.sqrt(2.0), 20.0 / math.sqrt(2.0)),
            dtype=torch.float64,
        )

        axis_velocity = variational_module._bounded_vector_update(
            background,
            axis_control,
            scale=2.0,
            limit=30.0,
        )
        diagonal_velocity = variational_module._bounded_vector_update(
            background,
            diagonal_control,
            scale=2.0,
            limit=30.0,
        )

        torch.testing.assert_close(
            torch.linalg.vector_norm(axis_velocity),
            torch.linalg.vector_norm(diagonal_velocity),
        )
        self.assertLess(float(torch.linalg.vector_norm(axis_velocity)), 30.0)
        torch.testing.assert_close(axis_velocity[1], axis_velocity.new_zeros(()))
        torch.testing.assert_close(diagonal_velocity[0], diagonal_velocity[1])

    def test_radial_velocity_control_preserves_baseline_and_local_scale(
        self,
    ) -> None:
        baseline = torch.tensor((12.0, -5.0), dtype=torch.float64)
        decoded = variational_module._bounded_vector_update(
            baseline,
            torch.zeros_like(baseline),
            scale=2.0,
            limit=30.0,
        )
        torch.testing.assert_close(decoded, baseline)

        zero = torch.zeros(2, dtype=torch.float64)
        direction = torch.tensor((1.0, -2.0), dtype=torch.float64)
        value, tangent = torch.func.jvp(
            lambda control: variational_module._bounded_vector_update(
                zero,
                control,
                scale=2.0,
                limit=30.0,
            ),
            (zero,),
            (direction,),
        )
        torch.testing.assert_close(value, zero)
        torch.testing.assert_close(tangent, 2.0 * direction)

    def test_bounded_controls_allow_inward_updates_at_saturated_baselines(
        self,
    ) -> None:
        vector = torch.tensor((30.0, 0.0), dtype=torch.float64)
        zero_vector = variational_module._bounded_vector_update(
            vector,
            torch.zeros_like(vector),
            scale=2.0,
            limit=30.0,
        )
        inward_vector = variational_module._bounded_vector_update(
            vector,
            torch.tensor((-1.0, 0.0), dtype=vector.dtype),
            scale=2.0,
            limit=30.0,
        )
        outward_vector = variational_module._bounded_vector_update(
            vector,
            torch.tensor((1.0, 0.0), dtype=vector.dtype),
            scale=2.0,
            limit=30.0,
        )
        torch.testing.assert_close(zero_vector, vector, rtol=0.0, atol=0.0)
        self.assertLess(float(torch.linalg.vector_norm(inward_vector)), 30.0)
        torch.testing.assert_close(outward_vector, vector, rtol=0.0, atol=0.0)

        scalar = torch.tensor(1.0, dtype=torch.float64)
        zero_scalar = variational_module._bounded_update(
            scalar,
            scalar.new_zeros(()),
            scale=0.1,
            limit=1.0,
        )
        inward_scalar = variational_module._bounded_update(
            scalar,
            scalar.new_tensor(-1.0),
            scale=0.1,
            limit=1.0,
        )
        outward_scalar = variational_module._bounded_update(
            scalar,
            scalar.new_tensor(1.0),
            scale=0.1,
            limit=1.0,
        )
        torch.testing.assert_close(zero_scalar, scalar, rtol=0.0, atol=0.0)
        self.assertLess(float(inward_scalar), 1.0)
        torch.testing.assert_close(outward_scalar, scalar, rtol=0.0, atol=0.0)

    def test_radial_velocity_control_keeps_large_controls_inside_speed_ball(
        self,
    ) -> None:
        background = torch.tensor((10.0, 5.0), dtype=torch.float64)
        for control in (
            torch.tensor((1.0e6, 0.0), dtype=torch.float64),
            torch.tensor((0.0, -1.0e6), dtype=torch.float64),
            torch.tensor((1.0e6, 1.0e6), dtype=torch.float64),
        ):
            with self.subTest(control=tuple(control.tolist())):
                velocity = variational_module._bounded_vector_update(
                    background,
                    control,
                    scale=2.0,
                    limit=30.0,
                )
                self.assertTrue(bool(torch.all(torch.isfinite(velocity))))
                self.assertLessEqual(
                    float(torch.linalg.vector_norm(velocity)),
                    30.0
                    * (1.0 + 10.0 * torch.finfo(velocity.dtype).eps),
                )

    def test_solver_reuses_pullbacks_and_field_diagnostic_adds_two(self) -> None:
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
        self.assertEqual(
            vjp_calls,
            result.outer_iterations
            + result.linearization_polish_iterations
            + 5,
        )

    def test_final_linearization_is_polished_to_stationarity(self) -> None:
        coordinates = torch.arange(8, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        frames = torch.stack(
            tuple(
                -10.0
                + 40.0
                * torch.exp(
                    -(
                        (y - 3.5).square()
                        + (x - center_x).square()
                    )
                    / 4.0
                )
                for center_x in (3.0, 3.5, 4.0)
            )
        )
        config = AnalysisConfig(
            censored_background_policy="detection_limit",
            maximum_outer_iterations=8,
            maximum_pcg_iterations=100,
            pcg_relative_tolerance=1.0e-8,
        )

        _, result = variational_nowcast(
            frames,
            nowcast_config=NowcastConfig(horizon_minutes=10),
            analysis_config=config,
        )

        self.assertIn(
            result.reason,
            (
                "step_tolerance",
                "final_linearization_stationary",
                "final_robust_irls_fixed_point",
            ),
        )
        self.assertGreater(result.linearization_polish_iterations, 0)
        self.assertIsNotNone(result.linearization)
        assert result.linearization is not None
        self.assertLessEqual(
            result.linearization.relative_stationarity,
            config.final_linearization_relative_stationarity_tolerance,
        )
        self.assertEqual(
            result.linearization_relative_stationarity,
            result.linearization.relative_stationarity,
        )
        self.assertEqual(
            result.linearization_gradient_norm,
            result.linearization.gradient_norm,
        )
        self.assertTrue(result.final_robust_stationary)
        self.assertTrue(result.final_irls_fixed_point)
        self.assertTrue(result.p1_forecast_eligible)
        self.assertTrue(result.posterior_eligible)

    def test_final_linearization_tracks_remap_branch_changes(self) -> None:
        _, frozen = self.stationary_problem()
        baseline = replace(
            frozen.baseline_state,
            displacement_yx=torch.tensor(
                (0.9999, 0.25),
                dtype=torch.float64,
            ),
        )
        frozen = replace(frozen, baseline_state=baseline)
        control = initial_control(frozen)
        frozen = variational_module._freeze_analysis_remap_cells(
            control,
            frozen,
        )

        self.assertTrue(
            variational_module._analysis_remap_cells_match(control, frozen)
        )
        crossed = control.clone()
        crossed[-3] = 1.0e-3
        self.assertFalse(
            variational_module._analysis_remap_cells_match(crossed, frozen)
        )

    def test_retained_linearization_owns_its_tensor_storage(self) -> None:
        coordinates = torch.arange(8, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        blob = -10.0 + 40.0 * torch.exp(
            -((y - 3.5).square() + (x - 3.5).square()) / 4.0
        )
        frames = torch.stack((blob, blob - 1.0, blob))
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=NowcastConfig(horizon_minutes=10),
            analysis_config=AnalysisConfig(
                censored_background_policy="detection_limit",
                maximum_outer_iterations=8,
                maximum_pcg_iterations=100,
                pcg_relative_tolerance=1.0e-8,
            ),
        )
        result = solve_analysis(observations, frozen)
        linearization = result.linearization
        assert linearization is not None

        self.assertNotEqual(
            linearization.observations.dbz.data_ptr(),
            observations.dbz.data_ptr(),
        )
        self.assertNotEqual(
            linearization.frozen.irls_sqrt_weight.data_ptr(),
            frozen.irls_sqrt_weight.data_ptr(),
        )
        retained_dbz = linearization.observations.dbz.clone()
        retained_irls = linearization.frozen.irls_sqrt_weight.clone()
        observations.dbz.add_(100.0)
        frozen.irls_sqrt_weight.zero_()
        torch.testing.assert_close(linearization.observations.dbz, retained_dbz)
        torch.testing.assert_close(
            linearization.frozen.irls_sqrt_weight,
            retained_irls,
        )

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
        self.assertIsNotNone(
            result.regularized_dynamics_hessian_eigenvalues
        )
        self.assertIsNotNone(
            result.regularized_dynamics_hessian_condition_number
        )
        eigenvalues = result.regularized_dynamics_hessian_eigenvalues
        assert eigenvalues is not None
        condition_number = (
            result.regularized_dynamics_hessian_condition_number
        )
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
        expected_gram = expected_hessian - torch.eye(3, dtype=torch.float64)
        expected_data_eigenvalues = torch.linalg.eigvalsh(expected_gram)
        expected_eigenvalues = torch.linalg.eigvalsh(expected_hessian)
        torch.testing.assert_close(
            torch.tensor(eigenvalues, dtype=torch.float64),
            expected_eigenvalues,
        )
        data_eigenvalues = result.dynamics_data_gram_eigenvalues
        assert data_eigenvalues is not None
        torch.testing.assert_close(
            torch.tensor(data_eigenvalues, dtype=torch.float64),
            expected_data_eigenvalues,
        )
        self.assertAlmostEqual(
            result.dynamics_data_information_trace or 0.0,
            float(torch.trace(expected_gram)),
        )
        self.assertEqual(result.dynamics_data_numerical_rank, 3)
        expected_data_share = expected_data_eigenvalues / (
            1.0 + expected_data_eigenvalues
        )
        self.assertAlmostEqual(
            result.dynamics_data_effective_dimension or 0.0,
            float(torch.sum(expected_data_share)),
        )
        data_share = result.dynamics_data_to_prior_ratio_by_mode
        assert data_share is not None
        torch.testing.assert_close(
            torch.tensor(data_share, dtype=torch.float64),
            expected_data_share,
        )
        observation_field_columns = []
        full_field_columns = []
        full_residual = lambda value: residual_vector(
            value,
            observations,
            diagnostic_frozen,
        )
        for field_index in range(field_size):
            direction = torch.zeros_like(result.control)
            direction[field_index] = 1.0
            observation_field_columns.append(
                cast(
                    torch.Tensor,
                    torch.func.jvp(
                        observation_residual,
                        (result.control,),
                        (direction,),
                    )[1],
                )
            )
            full_field_columns.append(
                cast(
                    torch.Tensor,
                    torch.func.jvp(
                        full_residual,
                        (result.control,),
                        (direction,),
                    )[1],
                )
            )
        observation_field = torch.stack(observation_field_columns, dim=1)
        full_field = torch.stack(full_field_columns, dim=1)
        dynamics_jacobian = torch.stack(dynamics_columns, dim=1)
        field_dynamics = observation_field.mT @ dynamics_jacobian
        expected_conditioned_gram = expected_gram - field_dynamics.mT @ (
            torch.linalg.solve(full_field.mT @ full_field, field_dynamics)
        )
        expected_conditioned_gram = 0.5 * (
            expected_conditioned_gram + expected_conditioned_gram.mT
        )
        expected_conditioned_eigenvalues = torch.linalg.eigvalsh(
            expected_conditioned_gram
        ).clamp_min(0.0)
        conditioned_eigenvalues = (
            result.field_conditioned_dynamics_data_gram_eigenvalues
        )
        assert conditioned_eigenvalues is not None
        torch.testing.assert_close(
            torch.tensor(conditioned_eigenvalues, dtype=torch.float64),
            expected_conditioned_eigenvalues,
            atol=2.0e-7,
            rtol=2.0e-6,
        )
        self.assertAlmostEqual(
            result.field_conditioned_dynamics_data_information_trace or 0.0,
            float(torch.sum(expected_conditioned_eigenvalues)),
            places=7,
        )
        expected_conditioned_dimension = torch.sum(
            expected_conditioned_eigenvalues
            / (1.0 + expected_conditioned_eigenvalues)
        )
        self.assertAlmostEqual(
            result.field_conditioned_dynamics_data_effective_dimension or 0.0,
            float(expected_conditioned_dimension),
            places=7,
        )
        self.assertLessEqual(
            result.field_conditioned_dynamics_data_information_trace or 0.0,
            result.dynamics_data_information_trace or 0.0,
        )
        self.assertLessEqual(
            result.field_conditioning_maximum_relative_residual or 0.0,
            self.analysis_config.pcg_relative_tolerance,
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
        self.assertIsNotNone(result.field_motion_jacobian_cosine_by_control)
        motion_cosines = result.field_motion_jacobian_cosine_by_control
        assert motion_cosines is not None
        for cosine in motion_cosines:
            self.assertIsNotNone(cosine)
            assert cosine is not None
            self.assertGreaterEqual(cosine, 0.0)
            self.assertLessEqual(cosine, 1.0)
        self.assertGreaterEqual(result.field_smoothness_prior_cost, 0.0)
        self.assertIsNotNone(result.motion_saturation_margin_yx)
        assert result.motion_saturation_margin_yx is not None
        self.assertTrue(
            all(margin >= 0.0 for margin in result.motion_saturation_margin_yx)
        )
        self.assertIsNotNone(result.growth_saturation_margin)
        self.assertGreaterEqual(result.growth_saturation_margin or 0.0, 0.0)

    def test_data_identifiability_reports_zero_information_without_data(
        self,
    ) -> None:
        observations, frozen = self.stationary_problem()
        no_information = replace(
            observations,
            quality_weight=torch.zeros_like(observations.quality_weight),
        )
        control = initial_control(frozen)
        diagnostics = variational_module._identifiability_diagnostics(
            control,
            no_information,
            frozen,
            analysis_trajectory(control, frozen),
        )

        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertEqual(
            diagnostics.dynamics_data_gram_eigenvalues,
            (0.0,) * 3,
        )
        self.assertEqual(diagnostics.dynamics_data_information_trace, 0.0)
        self.assertEqual(diagnostics.dynamics_data_numerical_rank, 0)
        self.assertEqual(diagnostics.dynamics_data_effective_dimension, 0.0)
        self.assertEqual(
            diagnostics.dynamics_data_to_prior_ratio_by_mode,
            (0.0,) * 3,
        )
        self.assertEqual(
            diagnostics.field_conditioned_dynamics_data_gram_eigenvalues,
            (0.0,) * 3,
        )
        self.assertEqual(
            diagnostics.field_conditioned_dynamics_data_information_trace,
            0.0,
        )
        self.assertEqual(
            diagnostics.field_conditioned_dynamics_data_effective_dimension,
            0.0,
        )
        torch.testing.assert_close(
            diagnostics.field_conditioned_dynamics_posterior_covariance,
            torch.eye(3, dtype=torch.float64),
        )
        self.assertEqual(
            diagnostics.regularized_dynamics_hessian_eigenvalues,
            (1.0,) * 3,
        )
        self.assertEqual(
            diagnostics.regularized_dynamics_hessian_condition_number,
            1.0,
        )

    def test_degraded_analysis_skips_field_conditioned_identifiability(
        self,
    ) -> None:
        observations, frozen = self.stationary_problem()

        with patch(
            "advar.variational._field_conditioned_dynamics_gram",
            side_effect=AssertionError(
                "degraded analysis entered field conditioning"
            ),
        ):
            result = variational_module._analysis_result(
                initial_control(frozen),
                observations,
                frozen,
                1.0,
                0.5,
                1,
                0,
                False,
                "maximum_iterations",
                degraded=True,
            )

        self.assertFalse(result.used_fallback)
        self.assertTrue(result.degraded)
        self.assertIsNotNone(result.dynamics_data_gram_eigenvalues)
        self.assertIsNone(
            result.field_conditioned_dynamics_data_gram_eigenvalues
        )
        self.assertIsNone(
            result.field_conditioned_dynamics_data_effective_dimension
        )

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
            ((0, 0),),
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
        expected_background_support = (
            (1.0 - observation_support) * background_support
        )
        torch.testing.assert_close(
            result.metadata.source_support,
            expected_support,
        )
        torch.testing.assert_close(
            result.metadata.observation_source_support,
            observation_support,
        )
        torch.testing.assert_close(
            result.metadata.background_source_support,
            expected_background_support,
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
        forecast.validate_issuance()

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
                state_path_source=TendencySource.OBSERVATION,
                state_path_mode=TendencyPairSelection.RECENT,
                state_path_pair_count=1,
                state_path_minimum_psr=12.0,
                state_path_age_minutes=10.0,
                minimum_growth_overlap_support=5.0,
                minimum_growth_overlap_area_km2=2.0,
            ),
        )

        result = variational_module._analysis_result(
            initial_control(frozen),
            observations,
            frozen,
            1.0,
            0.5,
            1,
            ((0, 0),),
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
        torch.testing.assert_close(
            result.metadata.verified_source_support,
            result.metadata.source_support,
        )
        torch.testing.assert_close(
            result.metadata.observation_verified_source_support,
            result.metadata.verified_source_support,
        )
        self.assertEqual(
            result.metadata.dynamics_source,
            DynamicsSource.P1_VARIATIONAL,
        )
        self.assertEqual(
            result.metadata.state_path_source,
            TendencySource.NONE,
        )
        self.assertEqual(
            result.metadata.state_path_mode,
            TendencyPairSelection.NONE,
        )
        self.assertEqual(result.metadata.state_path_pair_count, 0)
        self.assertTrue(math.isnan(result.metadata.state_path_minimum_psr))
        self.assertIsNone(result.metadata.state_path_age_minutes)
        self.assertTrue(
            math.isnan(result.metadata.minimum_growth_overlap_support)
        )
        self.assertTrue(
            math.isnan(result.metadata.minimum_growth_overlap_area_km2)
        )

    def test_p1_verification_excludes_local_latest_observation_mismatch(
        self,
    ) -> None:
        observations, frozen = self.stationary_problem()
        trajectory = analysis_trajectory(initial_control(frozen), frozen)
        mismatched_frames = trajectory.frames_linear.clone()
        mismatched_frames[-1, 0, 0] = dbz_to_echo(
            torch.tensor(
                self.nowcast_config.min_dbz,
                dtype=mismatched_frames.dtype,
            ),
            min_dbz=self.nowcast_config.min_dbz,
            max_dbz=self.nowcast_config.max_dbz,
        )
        trajectory = replace(
            trajectory,
            frames_linear=mismatched_frames,
        )

        verified = variational_module._local_analysis_verified_support(
            trajectory,
            observations,
            torch.ones_like(observations.dbz[-1]),
            frozen,
        )

        expected = torch.ones_like(verified)
        expected[0, 0] = 0.0
        torch.testing.assert_close(verified, expected)

    def test_p1_verification_excludes_zero_quality_observation(
        self,
    ) -> None:
        observations, frozen = self.stationary_problem()
        quality_weight = observations.quality_weight.clone()
        quality_weight[-1, 0, 0] = 0.0
        observations = replace(
            observations,
            quality_weight=quality_weight,
        )

        verified = variational_module._local_analysis_verified_support(
            analysis_trajectory(initial_control(frozen), frozen),
            observations,
            torch.ones_like(observations.dbz[-1]),
            frozen,
        )

        expected = torch.ones_like(verified)
        expected[0, 0] = 0.0
        torch.testing.assert_close(verified, expected)

    def test_p1_verification_excludes_negligible_precision(self) -> None:
        observations, frozen = self.stationary_problem()
        quality_weight = observations.quality_weight.clone()
        quality_weight[-1, 0, 0] = 1.0e-8
        observations = replace(
            observations,
            quality_weight=quality_weight,
        )
        trajectory = analysis_trajectory(initial_control(frozen), frozen)
        mismatched_frames = trajectory.frames_linear.clone()
        mismatched_frames[-1, 0, 0] = dbz_to_echo(
            torch.tensor(
                self.nowcast_config.min_dbz,
                dtype=mismatched_frames.dtype,
            ),
            min_dbz=self.nowcast_config.min_dbz,
            max_dbz=self.nowcast_config.max_dbz,
        )
        trajectory = replace(
            trajectory,
            frames_linear=mismatched_frames,
        )

        verified = variational_module._local_analysis_verified_support(
            trajectory,
            observations,
            torch.ones_like(observations.dbz[-1]),
            frozen,
        )

        self.assertEqual(float(verified[0, 0]), 0.0)

    def test_p1_verification_enforces_absolute_detected_error(self) -> None:
        observations, frozen = self.stationary_problem()
        frozen = replace(
            frozen,
            analysis_config=replace(
                frozen.analysis_config,
                maximum_latest_detected_error_std=100.0,
                maximum_local_analysis_verification_error_dbz=6.0,
            ),
        )
        trajectory = analysis_trajectory(initial_control(frozen), frozen)
        mismatched_frames = trajectory.frames_linear.clone()
        mismatched_frames[-1, 0, 0] = dbz_to_echo(
            torch.tensor(
                self.nowcast_config.min_dbz,
                dtype=mismatched_frames.dtype,
            ),
            min_dbz=self.nowcast_config.min_dbz,
            max_dbz=self.nowcast_config.max_dbz,
        )
        trajectory = replace(
            trajectory,
            frames_linear=mismatched_frames,
        )

        verified = variational_module._local_analysis_verified_support(
            trajectory,
            observations,
            torch.ones_like(observations.dbz[-1]),
            frozen,
        )

        self.assertEqual(float(verified[0, 0]), 0.0)

    def test_censored_p1_verification_requires_precision(self) -> None:
        observations, frozen = self.stationary_problem()
        detected = observations.detected_mask.clone()
        censored = observations.censored_mask.clone()
        quality = observations.quality_weight.clone()
        detected[-1, 0, 0] = False
        censored[-1, 0, 0] = True
        quality[-1, 0, 0] = 1.0e-8
        observations = replace(
            observations,
            detected_mask=detected,
            censored_mask=censored,
            quality_weight=quality,
        )
        trajectory = analysis_trajectory(initial_control(frozen), frozen)
        frames = trajectory.frames_linear.clone()
        frames[-1, 0, 0] = dbz_to_echo(
            torch.tensor(
                self.nowcast_config.min_dbz,
                dtype=frames.dtype,
            ),
            min_dbz=self.nowcast_config.min_dbz,
            max_dbz=self.nowcast_config.max_dbz,
        )
        trajectory = replace(trajectory, frames_linear=frames)

        verified = variational_module._local_analysis_verified_support(
            trajectory,
            observations,
            torch.ones_like(observations.dbz[-1]),
            frozen,
        )

        self.assertEqual(float(verified[0, 0]), 0.0)

    def test_p1_latest_state_fit_does_not_certify_wrong_motion(self) -> None:
        frames = torch.full(
            (3, 7, 9),
            self.nowcast_config.min_dbz,
            dtype=torch.float64,
        )
        frames[0, 3, 2] = 20.0
        frames[1, 3, 4] = 20.0
        frames[2, 3, 4] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        trajectory = variational_module.AnalysisTrajectory(
            frames_linear=dbz_to_echo(
                frames,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
            displacement_yx=torch.tensor(
                (0.0, 2.0),
                dtype=torch.float64,
            ),
            log_growth_per_step=torch.zeros((), dtype=torch.float64),
        )

        state, motion, growth, dynamics = (
            variational_module._local_analysis_evidence_supports(
                trajectory,
                observations,
                torch.ones_like(frames[-1]),
                frozen,
            )
        )

        self.assertEqual(float(state[3, 4]), 1.0)
        self.assertEqual(float(motion[3, 4]), 0.0)
        self.assertEqual(float(growth[3, 4]), 0.0)
        self.assertEqual(float(dynamics[3, 4]), 0.0)

    def test_p1_motion_fit_does_not_certify_wrong_growth(self) -> None:
        frames = torch.full(
            (3, 7, 9),
            self.nowcast_config.min_dbz,
            dtype=torch.float64,
        )
        frames[0, 3, 4] = 20.0
        frames[1, 3, 4] = 20.0
        frames[2, 3, 4] = 10.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        trajectory = variational_module.AnalysisTrajectory(
            frames_linear=dbz_to_echo(
                frames,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
            displacement_yx=torch.zeros(2, dtype=torch.float64),
            log_growth_per_step=torch.tensor(0.25, dtype=torch.float64),
        )

        state, motion, growth, dynamics = (
            variational_module._local_analysis_evidence_supports(
                trajectory,
                observations,
                torch.ones_like(frames[-1]),
                frozen,
            )
        )

        self.assertEqual(float(state[3, 4]), 1.0)
        self.assertEqual(float(motion[3, 4]), 1.0)
        self.assertEqual(float(growth[3, 4]), 0.0)
        self.assertEqual(float(dynamics[3, 4]), 0.0)

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
            ((0, 0),),
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
            ((0, 0),),
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
            ((0, 0),),
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
            variational_module._rectangular_offsets_yx(
                self.analysis_config.causal_support_dilation_px,
                self.analysis_config.causal_support_dilation_px,
            ),
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
            analysis_config=replace(
                self.analysis_config,
                censored_background_policy="floor",
            ),
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
            analysis_config=replace(
                self.analysis_config,
                censored_background_policy="floor",
            ),
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
        frames[:, 2:4, 2:4] = 20.0
        frames[1, 4, 6] = 6.0
        frames[2, 4, 6] = 7.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=replace(
                self.analysis_config,
                censored_background_policy="floor",
            ),
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
        self.assertEqual(
            result.metadata.dynamics_source,
            DynamicsSource.P0_FALLBACK,
        )
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

    def test_confidence_detects_bidirectional_echo_and_area_errors(
        self,
    ) -> None:
        diagnostics = replace(
            self._synthetic_amplitude_diagnostics((0.0, 0.0)),
            unresolved_fraction_by_time=torch.zeros(2, dtype=torch.float64),
            integrated_echo_ratio_by_time=torch.tensor(
                (0.25, 4.0),
                dtype=torch.float64,
            ),
            displacement_tolerant_soft_echo_area_ratio_by_time=torch.tensor(
                (4.0, 0.25),
                dtype=torch.float64,
            ),
        )

        self.assertTrue(diagnostics.degrades_confidence(self.analysis_config))

    def test_small_failed_precursor_object_is_not_diluted_by_large_object(
        self,
    ) -> None:
        frames = torch.full((3, 32, 32), -10.0, dtype=torch.float64)
        frames[:, 29, 29] = 20.0
        frames[2, 10:25, 10:25] = 20.0
        frames[2, 2, 2] = 20.0
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        prediction_dbz = frames.clone()
        prediction_dbz[2, 2, 2] = self.nowcast_config.min_dbz
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

        self.assertLess(
            float(diagnostics.unresolved_fraction_by_time[1]),
            self.analysis_config.maximum_unresolved_amplitude_fraction,
        )
        self.assertEqual(
            int(diagnostics.precursor_object_count_by_time[1]),
            2,
        )
        self.assertEqual(
            float(
                diagnostics.maximum_object_unresolved_fraction_by_time[1]
            ),
            1.0,
        )
        self.assertLess(
            float(
                diagnostics.minimum_object_integrated_echo_ratio_by_time[1]
            ),
            self.analysis_config.minimum_integrated_echo_ratio_for_confidence,
        )
        self.assertTrue(diagnostics.degrades_confidence(self.analysis_config))

    def test_overlapping_object_footprints_share_prediction_once(self) -> None:
        frames = torch.full((3, 9, 9), -10.0, dtype=torch.float64)
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        precursor = torch.zeros((9, 9), dtype=torch.bool)
        precursor[4, 3] = True
        precursor[4, 5] = True
        unresolved = torch.zeros_like(precursor)
        quality = torch.ones((9, 9), dtype=torch.float64)
        observed_dbz = torch.full((9, 9), -10.0, dtype=torch.float64)
        observed_dbz[precursor] = 20.0
        prediction_dbz = torch.full_like(observed_dbz, -10.0)
        prediction_dbz[4, 4] = 20.0
        prediction_echo = dbz_to_echo(
            prediction_dbz,
            min_dbz=self.nowcast_config.min_dbz,
            max_dbz=self.nowcast_config.max_dbz,
        )

        diagnostics = variational_module._precursor_object_diagnostics(
            precursor,
            unresolved,
            quality,
            observed_dbz,
            prediction_dbz,
            prediction_echo,
            torch.ones_like(precursor),
            frozen,
            enabled=True,
        )

        self.assertEqual(int(diagnostics[0]), 2)
        self.assertEqual(int(diagnostics[1]), 0)
        self.assertEqual(float(diagnostics[2]), 0.0)
        self.assertAlmostEqual(float(diagnostics[3]), 0.5)
        self.assertAlmostEqual(float(diagnostics[4]), 0.5)
        self.assertAlmostEqual(float(diagnostics[5]), 0.5, places=5)
        self.assertAlmostEqual(float(diagnostics[6]), 0.5, places=5)
        self.assertAlmostEqual(float(diagnostics[7]), 0.5)

    def test_matching_group_preserves_two_predicted_objects(self) -> None:
        frames = torch.full((3, 9, 9), -10.0, dtype=torch.float64)
        _, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        precursor = torch.zeros((9, 9), dtype=torch.bool)
        precursor[4, 3] = True
        precursor[4, 5] = True
        observed_dbz = torch.full((9, 9), -10.0, dtype=torch.float64)
        observed_dbz[precursor] = 20.0
        prediction_dbz = observed_dbz.clone()

        diagnostics = variational_module._precursor_object_diagnostics(
            precursor,
            torch.zeros_like(precursor),
            torch.ones((9, 9), dtype=torch.float64),
            observed_dbz,
            prediction_dbz,
            dbz_to_echo(
                prediction_dbz,
                min_dbz=self.nowcast_config.min_dbz,
                max_dbz=self.nowcast_config.max_dbz,
            ),
            torch.ones_like(precursor),
            frozen,
            enabled=True,
        )

        self.assertAlmostEqual(float(diagnostics[7]), 1.0)

    def test_established_echo_cannot_fill_precursor_object(self) -> None:
        frames = torch.full((3, 9, 9), -10.0, dtype=torch.float64)
        _, frozen = prepare_analysis(
            frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
        )
        precursor = torch.zeros((9, 9), dtype=torch.bool)
        precursor[4, 5] = True
        quality = torch.ones((9, 9), dtype=torch.float64)
        observed_dbz = torch.full((9, 9), -10.0, dtype=torch.float64)
        observed_dbz[precursor] = 20.0
        prediction_dbz = torch.full_like(observed_dbz, -10.0)
        prediction_dbz[4, 4] = 20.0
        prediction_echo = dbz_to_echo(
            prediction_dbz,
            min_dbz=self.nowcast_config.min_dbz,
            max_dbz=self.nowcast_config.max_dbz,
        )
        precursor_attribution = torch.ones_like(precursor)
        precursor_attribution[4, 4] = False

        diagnostics = variational_module._precursor_object_diagnostics(
            precursor,
            torch.zeros_like(precursor),
            quality,
            observed_dbz,
            prediction_dbz,
            prediction_echo,
            precursor_attribution,
            frozen,
            enabled=True,
        )

        self.assertEqual(int(diagnostics[0]), 1)
        self.assertEqual(float(diagnostics[3]), 0.0)
        self.assertEqual(float(diagnostics[4]), 0.0)
        self.assertLess(
            float(diagnostics[6]),
            self.analysis_config.minimum_soft_echo_area_ratio_for_confidence,
        )
        self.assertEqual(float(diagnostics[7]), 0.0)

    def test_object_count_collapse_degrades_confidence(self) -> None:
        diagnostics = replace(
            self._synthetic_amplitude_diagnostics((0.0, 0.0)),
            minimum_object_count_ratio_by_time=torch.tensor(
                (1.0, 0.5),
                dtype=torch.float64,
            ),
        )

        self.assertTrue(diagnostics.degrades_confidence(self.analysis_config))

    def test_operational_confidence_policy_falls_back_on_under_and_over(
        self,
    ) -> None:
        observations, frozen = self.stationary_problem()
        frozen = replace(
            frozen,
            analysis_config=replace(
                frozen.analysis_config,
                amplitude_confidence_policy="operational_fallback",
            ),
        )
        for ratio in (0.25, 4.0):
            with self.subTest(integrated_echo_ratio=ratio):
                diagnostics = replace(
                    self._synthetic_amplitude_diagnostics((0.0, 0.0)),
                    unresolved_fraction_by_time=torch.zeros(
                        2,
                        dtype=torch.float64,
                    ),
                    integrated_echo_ratio_by_time=torch.tensor(
                        (1.0, ratio),
                        dtype=torch.float64,
                    ),
                    displacement_tolerant_soft_echo_area_ratio_by_time=(
                        torch.ones(2, dtype=torch.float64)
                    ),
                )

                with patch.object(
                    variational_module,
                    "_amplitude_diagnostics",
                    return_value=diagnostics,
                ):
                    result = variational_module._analysis_result(
                        initial_control(frozen),
                        observations,
                        frozen,
                        1.0,
                        0.5,
                        1,
                        0,
                        True,
                        "test_operational_confidence",
                    )

                self.assertTrue(result.used_fallback)
                self.assertEqual(
                    result.reason,
                    "amplitude_confidence_failure",
                )
                self.assertTrue(result.amplitude_confidence_failed)
                self.assertEqual(
                    result.final_objective,
                    result.initial_objective,
                )

    def test_research_confidence_policy_returns_degraded_analysis(self) -> None:
        observations, frozen = self.stationary_problem()
        diagnostics = replace(
            self._synthetic_amplitude_diagnostics((0.0, 0.0)),
            unresolved_fraction_by_time=torch.zeros(2, dtype=torch.float64),
            integrated_echo_ratio_by_time=torch.tensor(
                (1.0, 4.0),
                dtype=torch.float64,
            ),
            displacement_tolerant_soft_echo_area_ratio_by_time=torch.ones(
                2,
                dtype=torch.float64,
            ),
        )

        with patch.object(
            variational_module,
            "_amplitude_diagnostics",
            return_value=diagnostics,
        ):
            result = variational_module._analysis_result(
                initial_control(frozen),
                observations,
                frozen,
                1.0,
                0.5,
                1,
                0,
                True,
                "test_research_confidence",
            )

        self.assertFalse(result.used_fallback)
        self.assertTrue(result.degraded)
        self.assertTrue(result.amplitude_confidence_failed)

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
            precursor_object_count_by_time=torch.zeros(2, dtype=torch.long),
            insufficient_object_count_by_time=torch.zeros(
                2,
                dtype=torch.long,
            ),
            maximum_object_unresolved_fraction_by_time=zeros.clone(),
            minimum_object_integrated_echo_ratio_by_time=torch.full(
                (2,),
                math.nan,
                dtype=dtype,
            ),
            maximum_object_integrated_echo_ratio_by_time=torch.full(
                (2,),
                math.nan,
                dtype=dtype,
            ),
            minimum_object_soft_echo_area_ratio_by_time=torch.full(
                (2,),
                math.nan,
                dtype=dtype,
            ),
            maximum_object_soft_echo_area_ratio_by_time=torch.full(
                (2,),
                math.nan,
                dtype=dtype,
            ),
            minimum_object_count_ratio_by_time=torch.full(
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

    def test_final_linearization_settings_are_validated(self) -> None:
        for field_name in (
            "final_linearization_relative_stationarity_tolerance",
            "final_robust_relative_stationarity_tolerance",
            "final_irls_relative_weight_tolerance",
        ):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, "must be positive"):
                        if field_name == (
                            "final_linearization_relative_stationarity_tolerance"
                        ):
                            AnalysisConfig(
                                final_linearization_relative_stationarity_tolerance=value
                            )
                        elif field_name == (
                            "final_robust_relative_stationarity_tolerance"
                        ):
                            AnalysisConfig(
                                final_robust_relative_stationarity_tolerance=value
                            )
                        else:
                            AnalysisConfig(
                                final_irls_relative_weight_tolerance=value
                            )
        for value in (-1, 1.5, True):
            with self.subTest(polish_iterations=value):
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    AnalysisConfig(
                        maximum_final_linearization_polish_iterations=value
                    )

        for field_name in (
            "maximum_common_bias_mode_weight_bytes",
            "maximum_frozen_whitener_bytes",
            "maximum_linearization_bytes",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    AnalysisConfig(**{field_name: 0})

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
        with self.assertRaisesRegex(
            ValueError,
            "amplitude_confidence_policy",
        ):
            AnalysisConfig(
                amplitude_confidence_policy=cast(
                    variational_module.AmplitudeConfidencePolicy,
                    "invalid",
                )
            )
        for field_name in (
            "minimum_integrated_echo_ratio_for_confidence",
            "minimum_soft_echo_area_ratio_for_confidence",
            "maximum_established_excess_growth_fraction_for_confidence",
            "minimum_object_count_ratio_for_confidence",
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
                        elif field_name == (
                            "maximum_established_excess_growth_fraction_for_confidence"
                        ):
                            AnalysisConfig(
                                maximum_established_excess_growth_fraction_for_confidence=(
                                    value
                                )
                            )
                        else:
                            AnalysisConfig(
                                minimum_object_count_ratio_for_confidence=value
                            )
        for field_name in (
            "maximum_integrated_echo_ratio_for_confidence",
            "maximum_soft_echo_area_ratio_for_confidence",
        ):
            for value in (0.9, float("nan"), float("inf")):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        "must be finite and at least 1",
                    ):
                        AnalysisConfig(**{field_name: value})

    def test_operational_analysis_requires_complete_physical_contract(
        self,
    ) -> None:
        config = AnalysisConfig(
            execution_mode="operational",
            operational_calibration_id="test-calibration-v1",
            amplitude_information_policy="operational_fallback",
            amplitude_confidence_policy="operational_fallback",
            motion_increment_scale_mps=2.0,
            causal_support_uncertainty_m=1000.0,
            amplitude_displacement_tolerance_m=1000.0,
        )
        frames = torch.full((3, 7, 9), 20.0, dtype=torch.float64)

        with self.assertRaisesRegex(ValueError, "grid/time contract"):
            prepare_analysis(
                frames,
                nowcast_config=NowcastConfig(
                    maximum_motion_speed_mps=30.0
                ),
                analysis_config=config,
            )

        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "pair confidence"):
            prepare_analysis(
                frames,
                nowcast_config=NowcastConfig(maximum_motion_speed_mps=30.0),
                analysis_config=config,
                grid_time_contract=contract,
            )

        with self.assertRaisesRegex(ValueError, "motion saturation margin"):
            prepare_analysis(
                frames,
                nowcast_config=NowcastConfig(
                    maximum_motion_speed_mps=1.0,
                    pair_echo_dilation_m=1000.0,
                    phase_correlation_sidelobe_radius_m=1000.0,
                ),
                analysis_config=config,
                grid_time_contract=contract,
            )

        groups = torch.zeros(frames.shape[-2:], dtype=torch.long)
        groups[:, frames.shape[-1] // 2 :] = 1
        physical_nowcast = NowcastConfig(
            maximum_motion_speed_mps=30.0,
            pair_echo_dilation_m=1000.0,
            phase_correlation_sidelobe_radius_m=1000.0,
        )
        grouped_config = replace(
            config,
            observation_common_bias_std_dbz=1.0,
        )
        with self.assertRaisesRegex(ValueError, "requires its digest"):
            prepare_analysis(
                frames,
                nowcast_config=physical_nowcast,
                analysis_config=grouped_config,
                observation_common_bias_group_index=groups,
                grid_time_contract=contract,
            )
        group_digest = (
            variational_module.observation_common_bias_group_map_digest(
                groups
            )
        )
        grouped_observations, _ = prepare_analysis(
            frames,
            nowcast_config=physical_nowcast,
            analysis_config=replace(
                grouped_config,
                observation_common_bias_group_map_digest=group_digest,
            ),
            observation_common_bias_group_index=groups,
            grid_time_contract=contract,
        )
        assert grouped_observations.common_bias_group_index is not None
        self.assertEqual(
            tensor_digest(grouped_observations.common_bias_group_index),
            group_digest,
        )

    def test_field_smoothness_weight_must_be_nonnegative(self) -> None:
        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "cannot be negative"):
                    AnalysisConfig(field_smoothness_weight=value)

    def test_physical_motion_increment_requires_positive_complete_contract(
        self,
    ) -> None:
        for value in (0.0, -1.0, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    AnalysisConfig(motion_increment_scale_mps=value)

        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        config = AnalysisConfig(motion_increment_scale_mps=2.0)
        with self.assertRaisesRegex(ValueError, "grid/time contract"):
            prepare_analysis(frames, analysis_config=config)

        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "physical motion limit"):
            prepare_analysis(
                frames,
                analysis_config=config,
                grid_time_contract=contract,
            )

    def test_physical_distance_settings_resolve_against_grid(self) -> None:
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=500.0,
            projection="EPSG:5179",
            grid_hash="e" * 64,
        )
        observations, frozen = prepare_analysis(
            frames,
            nowcast_config=NowcastConfig(
                maximum_motion_speed_mps=10.0
            ),
            analysis_config=AnalysisConfig(
                causal_support_uncertainty_m=1000.0,
                amplitude_displacement_tolerance_m=750.0,
            ),
            grid_time_contract=contract,
        )

        self.assertEqual(observations.dbz.shape, frames.shape)
        self.assertEqual(
            frozen.amplitude_displacement_tolerance_yx,
            (1, 0),
        )
        self.assertEqual(
            frozen.amplitude_displacement_offsets_yx,
            ((-1, 0), (0, 0), (1, 0)),
        )
        torch.testing.assert_close(
            frozen.motion_limits_yx,
            torch.tensor((12.0, 6.0), dtype=frames.dtype),
        )

    def test_exact_physical_footprint_excludes_diagonal_causal_anchor(
        self,
    ) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="1" * 64,
        )
        offsets = contract.pixel_offsets_within_distance(
            1000.0,
            maximum_radius_yx=(6, 8),
        )
        detected = torch.zeros((3, 7, 9), dtype=torch.bool)
        detected[2, 3, 3] = True
        observed = torch.zeros_like(detected)
        observed[0, 3, 4] = True
        observed[0, 4, 4] = True

        support, _ = variational_module._causal_control_and_seed_support(
            detected,
            observed,
            torch.zeros_like(detected),
            torch.zeros(2, dtype=torch.float64),
            self.analysis_config.minimum_control_reachability,
            offsets,
        )

        self.assertTrue(support[3, 4])
        self.assertFalse(support[4, 4])

    def test_exact_physical_footprint_excludes_diagonal_local_maximum(
        self,
    ) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="2" * 64,
        )
        offsets = contract.pixel_offsets_within_distance(
            1000.0,
            maximum_radius_yx=(4, 4),
        )
        value = torch.zeros((5, 5), dtype=torch.float64)
        value[2, 2] = 20.0

        local = variational_module._footprint_maximum(value, offsets)

        self.assertEqual(float(local[2, 1]), 20.0)
        self.assertEqual(float(local[1, 1]), 0.0)

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
        torch.testing.assert_close(
            result.metadata.path_verified_source_support,
            torch.zeros_like(result.metadata.source_support),
        )
        torch.testing.assert_close(
            result.metadata.verified_source_support,
            torch.zeros_like(result.metadata.source_support),
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

    def test_operational_analysis_rejects_degraded_candidate(self) -> None:
        observations, frozen = self.stationary_problem()
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-08-03T00:00:00Z",
                "2026-08-03T00:10:00Z",
                "2026-08-03T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="d" * 64,
        )
        frozen = replace(
            frozen,
            nowcast_config=replace(
                frozen.nowcast_config,
                maximum_motion_speed_mps=30.0,
                pair_echo_dilation_m=1000.0,
                phase_correlation_sidelobe_radius_m=1000.0,
            ),
            analysis_config=AnalysisConfig(
                execution_mode="operational",
                operational_calibration_id="test-calibration-v1",
                amplitude_information_policy="operational_fallback",
                amplitude_confidence_policy="operational_fallback",
                motion_increment_scale_mps=2.0,
                causal_support_uncertainty_m=1000.0,
                amplitude_displacement_tolerance_m=1000.0,
            ),
            grid_time_contract=contract,
        )

        result = variational_module._analysis_result(
            initial_control(frozen),
            observations,
            frozen,
            1.0,
            0.5,
            1,
            ((0, 0),),
            False,
            "pcg_failed",
            degraded=True,
        )

        self.assertTrue(result.used_fallback)
        self.assertFalse(result.degraded)
        self.assertEqual(result.reason, "degraded_operational_analysis")
        self.assertEqual(
            result.metadata.dynamics_source,
            DynamicsSource.P0_FALLBACK,
        )

    def test_operational_analysis_rejects_saturated_dynamics(self) -> None:
        observations, frozen = self.stationary_problem()
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-08-03T00:00:00Z",
                "2026-08-03T00:10:00Z",
                "2026-08-03T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="e" * 64,
        )
        frozen = replace(
            frozen,
            nowcast_config=replace(
                frozen.nowcast_config,
                maximum_motion_speed_mps=30.0,
                pair_echo_dilation_m=1000.0,
                phase_correlation_sidelobe_radius_m=1000.0,
                p1_motion_saturation_safe_margin_mps=2.0,
            ),
            analysis_config=AnalysisConfig(
                execution_mode="operational",
                operational_calibration_id="test-calibration-v1",
                amplitude_information_policy="operational_fallback",
                amplitude_confidence_policy="operational_fallback",
                motion_increment_scale_mps=2.0,
                causal_support_uncertainty_m=1000.0,
                amplitude_displacement_tolerance_m=1000.0,
            ),
            grid_time_contract=contract,
        )

        with patch.object(
            variational_module,
            "_motion_speed_saturation_margin",
            return_value=1.0,
        ):
            result = variational_module._analysis_result(
                initial_control(frozen),
                observations,
                frozen,
                1.0,
                0.5,
                1,
                ((0, 0),),
                True,
                "converged",
            )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.reason, "dynamics_saturation_margin")

    def test_p1_saturation_uncertainty_grows_toward_decoder_limit(self) -> None:
        reference = torch.zeros((), dtype=torch.float64)
        config = NowcastConfig(
            forecast_velocity_uncertainty_mps=1.0,
            forecast_log_growth_uncertainty_per_step=0.05,
            p1_motion_saturation_safe_margin_mps=2.0,
            p1_growth_saturation_safe_margin_per_step=0.1,
            p1_saturation_uncertainty_multiplier=4.0,
        )

        safe = variational_module._p1_saturation_uncertainty(
            reference,
            reference,
            2.0,
            reference.new_tensor(0.1),
            config,
        )
        saturated = variational_module._p1_saturation_uncertainty(
            reference,
            reference,
            0.0,
            reference.new_zeros(()),
            config,
        )

        torch.testing.assert_close(safe[0], reference)
        torch.testing.assert_close(safe[1], reference)
        torch.testing.assert_close(saturated[0], reference.new_tensor(4.0))
        torch.testing.assert_close(saturated[1], reference.new_tensor(0.2))

    def test_saturation_uncertainty_counteracts_small_decoder_jacobian(
        self,
    ) -> None:
        reference = torch.zeros((), dtype=torch.float64)

        def decode(control: torch.Tensor) -> torch.Tensor:
            return variational_module._bounded_update(
                reference,
                control,
                1.0,
                1.0,
            )

        center_control = reference.clone()
        saturated_control = reference.new_tensor(math.atanh(0.99))
        center_jacobian = torch.func.jacrev(decode)(center_control)
        saturated_jacobian = torch.func.jacrev(decode)(saturated_control)
        growth_margin = 1.0 - torch.abs(decode(saturated_control))
        config = NowcastConfig(
            max_log_growth_per_step=1.0,
            p1_growth_saturation_safe_margin_per_step=0.1,
        )
        saturation = variational_module._p1_saturation_uncertainty(
            reference,
            reference,
            config.p1_motion_saturation_safe_margin_mps,
            growth_margin,
            config,
        )[1]

        self.assertLess(
            float(saturated_jacobian),
            0.05 * float(center_jacobian),
        )
        self.assertGreater(float(saturation), 0.0)

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
