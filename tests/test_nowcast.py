from dataclasses import replace
from pathlib import Path
from importlib import import_module
import math
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.diagnostics import (  # noqa: E402
    EchoPositivityError,
    audit_transport,
    validate_physical_echo,
)
from advar.nowcast import (  # noqa: E402
    DataStatus,
    DynamicsSource,
    ForecastMetadata,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    StatePathProvenance,
    TendencyPairSelection,
    TendencySource,
    estimate_state as estimate_state_with_metadata,
    forecast_from_state as forecast_result_from_state,
    forecast_linear_from_state,
    nowcast,
)
from advar.physics import (  # noqa: E402
    FORECAST_INTEGRATOR_VERSION,
    FrozenCellMismatchError,
    RemapCell,
    dbz_to_echo,
    echo_to_dbz,
    remap,
    remap_core,
)


def dbz_to_linear(dbz: torch.Tensor, config: NowcastConfig) -> torch.Tensor:
    return dbz_to_echo(
        dbz,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )


def linear_to_dbz(echo: torch.Tensor, config: NowcastConfig) -> torch.Tensor:
    echo, _ = validate_physical_echo(echo, name="test conversion")
    return echo_to_dbz(
        echo,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )


def advect(
    echo: torch.Tensor,
    displacement: torch.Tensor,
    *,
    frozen_cell: RemapCell | None = None,
) -> torch.Tensor:
    return remap(echo, displacement, cell=frozen_cell)


def estimate_state(
    frames: torch.Tensor,
    config: NowcastConfig,
) -> RadarState:
    return estimate_state_with_metadata(frames, config)[0]


def observed_metadata(state: RadarState) -> ForecastMetadata:
    return ForecastMetadata(
        data_status=DataStatus.OBSERVED,
        coverage_by_frame=torch.ones(
            3,
            dtype=state.echo_linear.dtype,
            device=state.echo_linear.device,
        ),
        background_used=False,
        background_contribution_fraction=0.0,
        background_age_minutes=None,
        source_support=torch.ones_like(state.echo_linear),
        observation_source_support=torch.ones_like(state.echo_linear),
        background_source_support=torch.zeros_like(state.echo_linear),
        path_verified_source_support=torch.ones_like(state.echo_linear),
        verified_source_support=torch.ones_like(state.echo_linear),
        observation_verified_source_support=torch.ones_like(
            state.echo_linear
        ),
        background_verified_source_support=torch.zeros_like(
            state.echo_linear
        ),
        motion_disagreement_px=state.echo_linear.new_zeros(()),
        motion_disagreement_mps=state.echo_linear.new_full((), torch.nan),
        growth_disagreement=state.echo_linear.new_zeros(()),
        minimum_phase_correlation_psr=state.echo_linear.new_tensor(10.0),
        tendency_pair_count=2,
        tendency_source=TendencySource.OBSERVATION,
        state_path_source=TendencySource.OBSERVATION,
        state_path_age_minutes=0.0,
        observation_path=StatePathProvenance(age_minutes=0.0),
        minimum_growth_overlap_support=float(state.echo_linear.numel()),
    )


def forecast_from_state(
    state: RadarState,
    config: NowcastConfig,
) -> torch.Tensor:
    latest = linear_to_dbz(state.echo_linear, config)
    frames = torch.stack((latest, latest, latest))
    return forecast_result_from_state(
        state,
        observed_metadata(state),
        config,
        run=ForecastRunContract.from_inputs(
            config,
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        ),
    ).forecast_dbz


class NowcastTests(unittest.TestCase):
    config: NowcastConfig = NowcastConfig()
    echo: torch.Tensor = torch.empty(0)

    def setUp(self) -> None:
        self.config = NowcastConfig()
        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32),
            indexing="ij",
        )
        self.echo = 1.0e5 * torch.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 40.0)

    @staticmethod
    def _growth_evidence(
        nowcast_module: object,
        value: torch.Tensor,
        *,
        available: bool = True,
        overlap_support: float = 8.0,
        overlap_area_km2: float | None = None,
        mean_echo: float = 1.0,
        alignment_log_error: float = 0.0,
    ) -> object:
        previous_integral = value.new_tensor(
            overlap_support * mean_echo
        )
        return nowcast_module._GrowthEvidence(  # type: ignore[attr-defined]
            value=value,
            available=available,
            overlap_support=value.new_tensor(overlap_support),
            overlap_area_km2=(
                value.new_full((), torch.nan)
                if overlap_area_km2 is None
                else value.new_tensor(overlap_area_km2)
            ),
            aligned_previous_integral=previous_integral,
            current_integral=previous_integral * torch.exp(value),
            alignment_log_error=value.new_tensor(alignment_log_error),
        )

    def _pair_estimates(
        self,
        nowcast_module: object,
        *pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[object, ...]:
        return tuple(
            None
            if pair is None
            else (
                pair[0],
                self._growth_evidence(nowcast_module, pair[1]),
                pair[2],
            )
            for pair in pairs
        )

    def _moving_gaussian_frames(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float64),
            torch.arange(64, dtype=torch.float64),
            indexing="ij",
        )
        displacement = torch.tensor([1.2, -0.8], dtype=torch.float64)
        echoes = torch.stack(
            [
                1.0e3
                * torch.exp(
                    -(
                        (y - (24.0 + step * displacement[0])) ** 2
                        + (x - (34.0 + step * displacement[1])) ** 2
                    )
                    / 50.0
                )
                for step in range(3)
            ]
        )
        return linear_to_dbz(echoes, self.config), displacement

    def test_dbz_linear_round_trip(self) -> None:
        dbz = torch.tensor([-10.0, 0.0, 20.0, 45.0, 70.0])
        restored = linear_to_dbz(dbz_to_linear(dbz, self.config), self.config)
        torch.testing.assert_close(restored, dbz, atol=1.0e-4, rtol=1.0e-5)

    def test_stationary_echo_stays_stationary(self) -> None:
        dbz = linear_to_dbz(self.echo, self.config)
        result = nowcast(torch.stack((dbz, dbz, dbz)), self.config)
        forecast, state = result.forecast_dbz, result.state

        torch.testing.assert_close(
            state.displacement_yx,
            torch.zeros(2),
            atol=0.1,
            rtol=0.0,
        )
        torch.testing.assert_close(forecast[0], dbz, atol=0.02, rtol=0.0)
        self.assertEqual(forecast.shape, (18, 64, 64))
        torch.testing.assert_close(
            result.metadata.source_support,
            torch.ones_like(result.metadata.source_support),
        )
        self.assertTrue(bool(torch.all(result.valid_mask)))
        self.assertFalse(hasattr(result, "forecast_linear"))

    def test_translation_is_recovered(self) -> None:
        displacement = torch.tensor([2.0, -3.0])
        echo_1 = advect(self.echo, displacement)
        echo_2 = advect(echo_1, displacement)
        frames = linear_to_dbz(
            torch.stack((self.echo, echo_1, echo_2)),
            self.config,
        )

        state = estimate_state(frames, self.config)
        torch.testing.assert_close(
            state.displacement_yx,
            displacement,
            atol=0.35,
            rtol=0.0,
        )

        expected = advect(echo_2, displacement)
        first_forecast = forecast_from_state(state, self.config)[0]
        torch.testing.assert_close(
            dbz_to_linear(first_forecast, self.config),
            expected,
            atol=50.0,
            rtol=0.02,
        )

    def test_motion_improves_an_independent_analytic_gaussian_truth(
        self,
    ) -> None:
        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32),
            indexing="ij",
        )
        displacement = torch.tensor([0.45, -0.35])
        truth = torch.stack(
            tuple(
                2.0e4
                * torch.exp(
                    -(
                        (y - (30.2 + step * displacement[0])) ** 2
                        + (x - (31.4 + step * displacement[1])) ** 2
                    )
                    / 8.0
                )
                for step in range(4)
            )
        )
        frames = linear_to_dbz(truth[:3], self.config)
        state = estimate_state(frames, self.config)
        forecast = forecast_from_state(
            state,
            NowcastConfig(horizon_minutes=10),
        )[0]
        verification = linear_to_dbz(truth[3], self.config)

        torch.testing.assert_close(
            state.displacement_yx,
            displacement,
            atol=0.2,
            rtol=0.0,
        )
        issued = torch.isfinite(forecast)
        forecast_error = torch.mean(
            torch.abs(forecast[issued] - verification[issued])
        )
        persistence_error = torch.mean(
            torch.abs(frames[2][issued] - verification[issued])
        )
        self.assertLess(float(forecast_error), float(persistence_error))

    def test_motion_is_recovered_without_suppressing_edge_echo(self) -> None:
        y, x = torch.meshgrid(
            torch.arange(16, dtype=torch.float32),
            torch.arange(16, dtype=torch.float32),
            indexing="ij",
        )
        echo_0 = 1.0e5 * torch.exp(-((y - 4) ** 2 + (x - 11) ** 2) / 8.0)
        displacement = torch.tensor([1.0, -2.0])
        echo_1 = advect(echo_0, displacement)
        echo_2 = advect(echo_1, displacement)
        frames = linear_to_dbz(
            torch.stack((echo_0, echo_1, echo_2)),
            self.config,
        )

        state = estimate_state(frames, self.config)
        torch.testing.assert_close(
            state.displacement_yx,
            displacement,
            atol=0.2,
            rtol=0.0,
        )

    def test_motion_is_recovered_in_a_two_row_frame(self) -> None:
        echo_0 = torch.zeros(2, 8)
        echo_0[:, 2:4] = 1.0e5
        displacement = torch.tensor([0.0, 1.0])
        echo_1 = advect(echo_0, displacement)
        echo_2 = advect(echo_1, displacement)
        frames = linear_to_dbz(
            torch.stack((echo_0, echo_1, echo_2)),
            self.config,
        )

        state = estimate_state(frames, self.config)
        torch.testing.assert_close(state.displacement_yx, displacement)

    def test_boundary_outflow_is_not_treated_as_decay(self) -> None:
        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32),
            indexing="ij",
        )
        echo_0 = 1.0e5 * torch.exp(-((y - 4) ** 2 + (x - 32) ** 2) / 20.0)
        displacement = torch.tensor([-2.0, 0.0])
        echo_1 = advect(echo_0, displacement).clamp_min(0.0)
        echo_2 = advect(echo_1, displacement).clamp_min(0.0)
        frames = linear_to_dbz(
            torch.stack((echo_0, echo_1, echo_2)),
            self.config,
        )

        state = estimate_state(frames, self.config)
        self.assertLess(abs(float(state.log_growth_per_step)), 0.005)

    def test_growth_is_recovered(self) -> None:
        factor = 1.1
        frames = linear_to_dbz(
            torch.stack((self.echo, self.echo * factor, self.echo * factor**2)),
            self.config,
        )
        state = estimate_state(frames, self.config)
        self.assertAlmostEqual(
            float(state.log_growth_per_step),
            float(torch.log(torch.tensor(factor))),
            places=3,
        )

    def test_missing_pixel_is_not_published_as_observed_clear(self) -> None:
        dbz = linear_to_dbz(self.echo, self.config)
        frames = torch.stack((dbz, dbz, dbz))
        frames[:, 0, 0] = torch.nan
        result = nowcast(frames, self.config)
        forecast, metadata = result.forecast_dbz, result.metadata

        self.assertEqual(
            metadata.data_status,
            DataStatus.PARTIAL,
        )
        self.assertTrue(bool(torch.all(torch.isnan(forecast[:, 0, 0]))))
        self.assertTrue(bool(torch.all(torch.isfinite(forecast[:, 1:, 1:]))))

    def test_all_missing_without_background_is_unavailable(self) -> None:
        frames = torch.full((3, 8, 8), torch.nan)
        result = nowcast(frames, self.config)
        forecast, metadata = result.forecast_dbz, result.metadata

        self.assertEqual(metadata.data_status, DataStatus.UNAVAILABLE)
        self.assertEqual(float(metadata.coverage_by_frame.mean()), 0.0)
        self.assertEqual(metadata.tendency_source, TendencySource.NONE)
        self.assertTrue(bool(torch.all(torch.isnan(forecast))))

    def test_all_missing_uses_time_aligned_stale_background(self) -> None:
        frames = torch.full((3, 8, 8), torch.nan)
        background = torch.full((3, 8, 8), 20.0)
        result = nowcast(
            frames,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        forecast, metadata = result.forecast_dbz, result.metadata
        self.assertEqual(metadata.data_status, DataStatus.STALE_BACKGROUND)
        self.assertTrue(metadata.background_used)
        self.assertEqual(metadata.background_contribution_fraction, 1.0)
        self.assertEqual(metadata.background_age_minutes, 10.0)
        self.assertEqual(
            metadata.tendency_source,
            TendencySource.NONE,
        )
        self.assertFalse(metadata.background_tendency_used)
        self.assertTrue(torch.isnan(metadata.minimum_phase_correlation_psr))
        self.assertEqual(
            metadata.state_path_source,
            TendencySource.BACKGROUND,
        )
        self.assertEqual(
            metadata.state_path_mode,
            TendencyPairSelection.NONE,
        )
        self.assertEqual(metadata.state_path_age_minutes, 10.0)
        self.assertTrue(bool(torch.all(torch.isfinite(forecast))))
        torch.testing.assert_close(forecast[0], background[-1])

    def test_background_requires_an_explicit_age(self) -> None:
        frames = torch.full((3, 8, 8), torch.nan)
        background = torch.full((3, 8, 8), 20.0)

        with self.assertRaisesRegex(
            ValueError,
            "background_age_minutes is required",
        ):
            nowcast(
                frames,
                self.config,
                background_frames_dbz=background,
            )

    def test_latest_only_background_uses_stationary_persistence(self) -> None:
        frames = torch.full((3, 8, 8), torch.nan)
        background = torch.full((3, 8, 8), torch.nan)
        background[-1] = 20.0
        result = nowcast(
            frames,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        torch.testing.assert_close(
            result.state.displacement_yx,
            torch.zeros(2),
        )
        torch.testing.assert_close(
            result.state.log_growth_per_step,
            torch.zeros(()),
        )
        torch.testing.assert_close(result.forecast_dbz[0], background[-1])

    def test_old_background_without_latest_source_is_unavailable(self) -> None:
        frames = torch.full((3, 8, 8), torch.nan)
        background = torch.full((3, 8, 8), torch.nan)
        background[0] = 20.0
        result = nowcast(
            frames,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=20.0,
        )

        forecast, metadata = result.forecast_dbz, result.metadata
        self.assertFalse(metadata.background_used)
        self.assertEqual(metadata.data_status, DataStatus.UNAVAILABLE)
        self.assertTrue(bool(torch.all(torch.isnan(forecast))))

    def test_unused_background_does_not_publish_stale_age(self) -> None:
        frames = torch.full((3, 8, 8), 20.0)
        background = torch.full_like(frames, 25.0)
        result = nowcast(
            frames,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        metadata = result.metadata
        self.assertFalse(metadata.background_used)
        self.assertFalse(metadata.background_tendency_used)
        self.assertEqual(metadata.background_contribution_fraction, 0.0)
        self.assertEqual(metadata.background_state_support_fraction, 0.0)
        torch.testing.assert_close(
            metadata.background_source_support,
            torch.zeros_like(metadata.background_source_support),
        )
        self.assertIsNone(metadata.background_path.age_minutes)
        self.assertIsNone(metadata.background_age_minutes)
        self.assertEqual(metadata.data_status, DataStatus.OBSERVED)

    def test_tiny_actual_background_contribution_preserves_provenance(
        self,
    ) -> None:
        config = replace(self.config, epsilon=0.02)
        frames = torch.full((3, 8, 8), torch.nan, dtype=torch.float64)
        frames[-1] = 20.0
        frames[-1, 0, 0] = torch.nan
        background = torch.full_like(frames, torch.nan)
        background[-1] = 25.0

        result = nowcast(
            frames,
            config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        metadata = result.metadata
        self.assertLess(metadata.background_contribution_fraction, config.epsilon)
        self.assertTrue(metadata.background_used)
        self.assertFalse(metadata.background_tendency_used)
        self.assertEqual(metadata.background_age_minutes, 10.0)
        self.assertGreater(
            float(metadata.background_source_support[0, 0]),
            config.support_presence_threshold,
        )
        self.assertEqual(metadata.background_path.age_minutes, 10.0)
        torch.testing.assert_close(
            metadata.source_support,
            metadata.observation_source_support
            + metadata.background_source_support,
        )
        result.validate_issuance()

    def test_background_tendency_preserves_usage_and_age(self) -> None:
        background, _ = self._moving_gaussian_frames()
        frames = torch.full_like(background, torch.nan)
        frames[2] = background[2]

        result = nowcast(
            frames,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        metadata = result.metadata
        self.assertEqual(metadata.tendency_source, TendencySource.BACKGROUND)
        self.assertTrue(metadata.background_tendency_used)
        self.assertTrue(metadata.background_used)
        self.assertEqual(metadata.background_state_support_fraction, 0.0)
        self.assertEqual(metadata.background_age_minutes, 10.0)
        self.assertGreaterEqual(
            float(metadata.minimum_phase_correlation_psr),
            self.config.minimum_phase_correlation_psr,
        )

    def test_phase_correlation_psr_accepts_a_distinct_shift(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        frames, displacement = self._moving_gaussian_frames()

        shift, psr, search_interior = (
            nowcast_module._phase_correlation_shift_and_psr(
                frames[0],
                frames[1],
                self.config,
            )
        )

        torch.testing.assert_close(
            shift,
            displacement,
            atol=0.15,
            rtol=0.0,
        )
        self.assertGreaterEqual(
            float(psr),
            self.config.minimum_phase_correlation_psr,
        )
        self.assertTrue(search_interior)

    def test_physical_psr_is_translation_invariant(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        config = NowcastConfig(
            phase_correlation_sidelobe_radius_m=1000.0,
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
            grid_hash="7" * 64,
        )
        correlation = torch.arange(64, dtype=torch.float64).reshape(8, 8)
        correlation[0, 0] = 100.0
        shifted = torch.roll(correlation, shifts=(2, 3), dims=(0, 1))

        origin_psr = nowcast_module._peak_to_sidelobe_ratio(
            correlation,
            0,
            0,
            config,
            contract,
        )
        shifted_psr = nowcast_module._peak_to_sidelobe_ratio(
            shifted,
            2,
            3,
            config,
            contract,
        )

        torch.testing.assert_close(shifted_psr, origin_psr)

    def test_phase_correlation_rejects_peak_outside_motion_range(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        previous = linear_to_dbz(self.echo, self.config)
        current = torch.roll(previous, shifts=30, dims=1)

        shift, psr, search_interior = (
            nowcast_module._phase_correlation_shift_and_psr(
                previous,
                current,
                self.config,
            )
        )

        torch.testing.assert_close(
            shift,
            torch.tensor([0.0, 30.0]),
            atol=0.1,
            rtol=0.0,
        )
        self.assertGreaterEqual(
            float(psr),
            self.config.minimum_phase_correlation_psr,
        )
        self.assertFalse(search_interior)

    def test_phase_correlation_rejects_peak_at_search_boundary(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        previous = linear_to_dbz(self.echo, self.config)
        current = torch.roll(previous, shifts=20, dims=1)

        shift, psr, search_interior = (
            nowcast_module._phase_correlation_shift_and_psr(
                previous,
                current,
                self.config,
            )
        )

        torch.testing.assert_close(
            shift,
            torch.tensor([0.0, 20.0]),
            atol=0.1,
            rtol=0.0,
        )
        self.assertGreaterEqual(
            float(psr),
            self.config.minimum_phase_correlation_psr,
        )
        self.assertFalse(search_interior)

    def test_subpixel_search_limit_accepts_zero_motion(self) -> None:
        config = NowcastConfig(max_displacement_px=0.25)
        frame = linear_to_dbz(self.echo, config)

        state, metadata = estimate_state_with_metadata(
            torch.stack((frame, frame, frame)),
            config,
        )

        torch.testing.assert_close(
            state.displacement_yx,
            torch.zeros_like(state.displacement_yx),
        )
        self.assertEqual(metadata.tendency_pair_count, 2)
        self.assertGreaterEqual(
            float(metadata.minimum_phase_correlation_psr),
            config.minimum_phase_correlation_psr,
        )

    def test_conflicting_motion_without_psr_advantage_uses_persistence(
        self,
    ) -> None:
        first = linear_to_dbz(self.echo, self.config)
        second = torch.roll(first, shifts=10, dims=1)
        frames = torch.stack((first, second, first))

        state, metadata = estimate_state_with_metadata(frames, self.config)

        torch.testing.assert_close(
            state.displacement_yx,
            torch.zeros_like(state.displacement_yx),
            atol=0.1,
            rtol=0.0,
        )
        self.assertAlmostEqual(
            float(metadata.motion_disagreement_px),
            20.0,
            places=1,
        )
        self.assertEqual(metadata.motion_pair_count, 0)
        self.assertEqual(metadata.growth_pair_count, 0)
        self.assertEqual(
            metadata.motion_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertEqual(
            metadata.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertTrue(metadata.motion_pair_conflict)
        self.assertTrue(metadata.growth_pair_conflict)
        self.assertEqual(metadata.tendency_pair_count, 0)
        self.assertTrue(torch.isnan(metadata.minimum_phase_correlation_psr))

    def test_conflicting_growth_without_psr_advantage_uses_persistence(
        self,
    ) -> None:
        factor = math.exp(self.config.max_log_growth_per_step)
        frames = linear_to_dbz(
            torch.stack((self.echo, self.echo * factor, self.echo)),
            self.config,
        )

        state, metadata = estimate_state_with_metadata(frames, self.config)

        self.assertAlmostEqual(
            float(state.log_growth_per_step),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            float(metadata.growth_disagreement),
            2.0 * self.config.max_log_growth_per_step,
            places=3,
        )
        self.assertEqual(metadata.motion_pair_count, 2)
        self.assertEqual(metadata.growth_pair_count, 0)
        self.assertEqual(
            metadata.motion_pair_selection,
            TendencyPairSelection.BLENDED,
        )
        self.assertEqual(
            metadata.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertFalse(metadata.motion_pair_conflict)
        self.assertTrue(metadata.growth_pair_conflict)
        self.assertEqual(metadata.tendency_pair_count, 2)

    def test_opposite_motion_below_search_limit_is_not_blended(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        values = torch.zeros((3, 2, 2), dtype=torch.float64)
        masks = torch.ones_like(values, dtype=torch.bool)
        estimates = (
            (
                torch.tensor([9.0, 0.0], dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(12.0, dtype=torch.float64),
            ),
            (
                torch.tensor([-9.0, 0.0], dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(12.0, dtype=torch.float64),
            ),
        )

        with patch.object(
            nowcast_module,
            "_estimate_available_pair",
            side_effect=self._pair_estimates(nowcast_module, *estimates),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                values,
                masks,
                values,
                self.config,
                None,
            )

        torch.testing.assert_close(
            tendency.displacement_yx,
            torch.zeros(2, dtype=torch.float64),
        )
        self.assertEqual(
            tendency.motion_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertEqual(tendency.motion_pair_count, 0)
        self.assertEqual(tendency.growth_pair_count, 0)
        self.assertTrue(tendency.motion_pair_conflict)
        self.assertTrue(tendency.growth_pair_conflict)
        self.assertEqual(
            tendency.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertEqual(float(tendency.log_growth_per_step), 0.0)
        torch.testing.assert_close(
            tendency.source_displacement_yx,
            torch.tensor(
                ((0.0, 0.0), (-9.0, 0.0), (0.0, 0.0)),
                dtype=torch.float64,
            ),
        )
        self.assertTrue(bool(torch.all(tendency.source_usable)))

    def test_growth_is_reestimated_with_selected_motion(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        previous = torch.zeros((8, 8), dtype=torch.float64)
        current = torch.zeros_like(previous)
        previous[2:6, 1:3] = 10.0
        current[2:6, 3:5] = 10.0
        previous_mask = torch.zeros_like(previous, dtype=torch.bool)
        current_mask = torch.zeros_like(previous, dtype=torch.bool)
        previous_mask[1:7, 0:5] = True
        current_mask[1:7, 2:7] = True
        linear = torch.stack((previous, current, previous))
        masks = torch.stack((previous_mask, current_mask, previous_mask))
        estimates = (
            (
                torch.tensor([0.0, 2.0], dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(12.0, dtype=torch.float64),
            ),
            (
                torch.tensor([0.0, -2.0], dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(12.0, dtype=torch.float64),
            ),
        )

        with patch.object(
            nowcast_module,
            "_estimate_available_pair",
            side_effect=self._pair_estimates(nowcast_module, *estimates),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                linear,
                masks,
                linear,
                self.config,
                None,
            )

        torch.testing.assert_close(
            tendency.displacement_yx,
            torch.zeros(2, dtype=torch.float64),
        )
        self.assertAlmostEqual(
            float(tendency.growth_disagreement),
            2.0 * self.config.max_log_growth_per_step,
        )
        self.assertTrue(tendency.growth_pair_conflict)
        self.assertEqual(
            tendency.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )

    def test_direction_turn_uses_each_source_path_for_current_state(
        self,
    ) -> None:
        nowcast_module = import_module("advar.nowcast")
        shape = (16, 16)
        linear = torch.zeros((3, *shape), dtype=torch.float64)
        masks = torch.zeros_like(linear, dtype=torch.bool)
        linear[0, 8, 4] = 1.0
        linear[1, 8, 8] = 2.0
        masks[0, 8, 4] = True
        masks[1, 8, 8] = True
        estimates = (
            (
                torch.tensor([0.0, 4.0], dtype=torch.float64),
                torch.tensor(0.04, dtype=torch.float64),
                torch.tensor(12.0, dtype=torch.float64),
            ),
            (
                torch.tensor([0.0, -4.0], dtype=torch.float64),
                torch.tensor(-0.04, dtype=torch.float64),
                torch.tensor(12.0, dtype=torch.float64),
            ),
        )

        with patch.object(
            nowcast_module,
            "_estimate_available_pair",
            side_effect=self._pair_estimates(nowcast_module, *estimates),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                linear,
                torch.ones_like(masks),
                linear,
                self.config,
                None,
            )

        current, support, _, _, _ = nowcast_module._merge_source_frames(
            linear,
            masks,
            tendency.source_displacement_yx,
            tendency.source_log_growth,
            tendency.source_usable,
            self.config,
        )

        self.assertEqual(float(support[8, 4]), 1.0)
        self.assertEqual(float(support[8, 8]), 0.0)
        self.assertAlmostEqual(
            float(current[8, 4]),
            2.0 * math.exp(-0.04),
        )
        torch.testing.assert_close(
            tendency.source_log_growth,
            torch.tensor((0.0, -0.04, 0.0), dtype=torch.float64),
        )

    def test_source_support_cannot_return_after_leaving_boundary(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        linear = torch.zeros((3, 5, 5), dtype=torch.float64)
        masks = torch.zeros_like(linear, dtype=torch.bool)
        linear[0, 2, 0] = 1.0
        masks[0, 2, 0] = True
        endpoint_displacements = linear.new_zeros((3, 2))
        source_growth = linear.new_zeros(3)
        source_usable = torch.tensor((True, False, False))
        support_segments = linear.new_zeros((3, 2, 2))
        support_segments[0, 0] = linear.new_tensor((0.0, -1.0))
        support_segments[0, 1] = linear.new_tensor((0.0, 1.0))

        endpoint_state, endpoint_support, _, _, _ = (
            nowcast_module._merge_source_frames(
                linear,
                masks,
                endpoint_displacements,
                source_growth,
                source_usable,
                self.config,
            )
        )
        path_state, path_support, _, _, _ = nowcast_module._merge_source_frames(
            linear,
            masks,
            endpoint_displacements,
            source_growth,
            source_usable,
            self.config,
            source_support_displacements_yx=support_segments,
        )

        self.assertEqual(float(endpoint_support.sum()), 1.0)
        self.assertEqual(float(endpoint_state.sum()), 1.0)
        self.assertEqual(float(path_support.sum()), 0.0)
        self.assertEqual(float(path_state.sum()), 0.0)

    def test_future_fallback_does_not_replace_observation_paths(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        values = torch.zeros((3, 16, 16), dtype=torch.float64)
        masks = torch.ones_like(values, dtype=torch.bool)
        earlier = (
            torch.tensor([0.0, 4.0], dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(12.0, dtype=torch.float64),
        )
        recent = (
            torch.tensor([0.0, -4.0], dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(12.0, dtype=torch.float64),
        )
        with patch.object(
            nowcast_module,
            "_estimate_available_pair",
            side_effect=self._pair_estimates(
                nowcast_module,
                earlier,
                recent,
            ),
        ):
            observation_paths = nowcast_module._estimate_source_tendencies(
                values,
                masks,
                values,
                self.config,
                None,
            )
        zero = values.new_zeros(())
        background_future = nowcast_module._single_pair_tendency(
            values.new_zeros(2),
            self._growth_evidence(nowcast_module, zero),
            values.new_tensor(20.0),
            selection=TendencyPairSelection.RECENT,
            source_pair_index=1,
        )
        prepared = nowcast_module.prepare_input(values, self.config)

        with patch.object(
            nowcast_module,
            "_estimate_source_tendencies",
            side_effect=(observation_paths, background_future),
        ):
            future, source, observation, background = (
                nowcast_module._estimate_time_normalized_tendencies(
                    prepared,
                    values,
                    values,
                    self.config,
                    None,
                )
            )

        self.assertFalse(observation_paths.future_available)
        self.assertTrue(observation_paths.reconstruction_available)
        self.assertIs(future, background_future)
        self.assertEqual(source, TendencySource.BACKGROUND)
        self.assertIs(observation, observation_paths)
        self.assertIs(background, background_future)
        torch.testing.assert_close(
            observation.source_displacement_yx[1],
            recent[0],
        )

    def test_background_future_does_not_move_observation_state(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        frames = torch.full((3, 16, 16), torch.nan, dtype=torch.float64)
        frames[1, 8, 8] = 20.0
        background = torch.full_like(frames, self.config.min_dbz)
        prepared = nowcast_module.prepare_input(
            frames,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )
        zero = frames.new_zeros(())
        earlier = (
            frames.new_tensor([0.0, 4.0]),
            zero,
            frames.new_tensor(10.0),
        )
        recent = (
            frames.new_tensor([0.0, -4.0]),
            zero,
            frames.new_tensor(12.0),
        )
        with patch.object(
            nowcast_module,
            "_estimate_available_pair",
            side_effect=self._pair_estimates(
                nowcast_module,
                earlier,
                recent,
            ),
        ):
            observation_paths = nowcast_module._estimate_source_tendencies(
                prepared.frames_dbz,
                prepared.observed_mask,
                torch.zeros_like(frames),
                self.config,
                None,
            )
        background_future = nowcast_module._single_pair_tendency(
            frames.new_zeros(2),
            self._growth_evidence(nowcast_module, zero),
            frames.new_tensor(20.0),
            selection=TendencyPairSelection.RECENT,
            source_pair_index=1,
        )

        with patch.object(
            nowcast_module,
            "_estimate_time_normalized_tendencies",
            return_value=(
                background_future,
                TendencySource.BACKGROUND,
                observation_paths,
                background_future,
            ),
        ):
            state, metadata = nowcast_module.estimate_prepared_state(
                prepared,
                self.config,
            )

        self.assertEqual(metadata.tendency_source, TendencySource.BACKGROUND)
        self.assertEqual(
            metadata.state_path_source,
            TendencySource.OBSERVATION,
        )
        self.assertEqual(
            metadata.state_path_mode,
            TendencyPairSelection.RECENT,
        )
        self.assertEqual(metadata.state_path_pair_count, 1)
        self.assertEqual(metadata.state_path_minimum_psr, 12.0)
        self.assertFalse(metadata.state_path_conflict)
        self.assertFalse(metadata.state_path_extrapolated)
        self.assertEqual(metadata.state_path_age_minutes, 10.0)
        self.assertGreater(float(state.echo_linear[8, 4]), 10.0)
        self.assertLess(float(state.echo_linear[8, 8]), 1.0)

    def test_observation_path_is_published_without_future_tendency(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        frames = torch.full((3, 16, 16), torch.nan, dtype=torch.float64)
        frames[1, 8, 8] = 20.0
        prepared = nowcast_module.prepare_input(frames, self.config)
        zero = frames.new_zeros(())
        estimates = self._pair_estimates(
            nowcast_module,
            (frames.new_tensor((0.0, 4.0)), zero, frames.new_tensor(10.0)),
            (frames.new_tensor((0.0, -4.0)), zero, frames.new_tensor(12.0)),
        )
        with patch.object(
            nowcast_module,
            "_estimate_available_pair",
            side_effect=estimates,
        ):
            observation_paths = nowcast_module._estimate_source_tendencies(
                prepared.frames_dbz,
                prepared.observed_mask,
                torch.zeros_like(frames),
                self.config,
                None,
            )
        background_paths = nowcast_module._estimate_source_tendencies(
            prepared.background_frames_dbz,
            prepared.background_mask,
            torch.zeros_like(frames),
            self.config,
            None,
        )

        with patch.object(
            nowcast_module,
            "_estimate_time_normalized_tendencies",
            return_value=(
                observation_paths,
                TendencySource.NONE,
                observation_paths,
                background_paths,
            ),
        ):
            state, metadata = nowcast_module.estimate_prepared_state(
                prepared,
                self.config,
            )

        self.assertEqual(metadata.tendency_source, TendencySource.NONE)
        self.assertEqual(
            metadata.state_path_source,
            TendencySource.OBSERVATION,
        )
        self.assertEqual(
            metadata.state_path_mode,
            TendencyPairSelection.RECENT,
        )
        self.assertEqual(metadata.state_path_pair_count, 1)
        self.assertGreater(float(state.echo_linear[8, 4]), 10.0)
        self.assertLess(float(state.echo_linear[8, 8]), 1.0)

    def test_complete_latest_frame_has_direct_path_provenance(self) -> None:
        frames, _ = self._moving_gaussian_frames()

        result = nowcast(frames, self.config)

        self.assertEqual(
            result.metadata.state_path_source,
            TendencySource.OBSERVATION,
        )
        self.assertEqual(
            result.metadata.state_path_mode,
            TendencyPairSelection.NONE,
        )
        self.assertEqual(result.metadata.state_path_pair_count, 0)
        self.assertTrue(math.isnan(result.metadata.state_path_minimum_psr))
        self.assertFalse(result.metadata.state_path_extrapolated)
        self.assertEqual(result.metadata.state_path_age_minutes, 0.0)

    def test_conflicting_pairs_choose_clearly_higher_psr_pair(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        values = torch.zeros((3, 2, 2), dtype=torch.float64)
        masks = torch.ones_like(values, dtype=torch.bool)
        estimates = (
            (
                torch.tensor([9.0, 0.0], dtype=torch.float64),
                torch.tensor(0.2, dtype=torch.float64),
                torch.tensor(30.0, dtype=torch.float64),
            ),
            (
                torch.tensor([-9.0, 0.0], dtype=torch.float64),
                torch.tensor(-0.2, dtype=torch.float64),
                torch.tensor(8.1, dtype=torch.float64),
            ),
        )

        with (
            patch.object(
                nowcast_module,
                "_estimate_available_pair",
                side_effect=self._pair_estimates(
                    nowcast_module,
                    *estimates,
                ),
            ),
            patch.object(
                nowcast_module,
                "_growth_evidence_aligned_with_motion",
                side_effect=(
                    self._growth_evidence(nowcast_module, estimates[0][1]),
                    self._growth_evidence(nowcast_module, estimates[1][1]),
                ),
            ),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                values,
                masks,
                values,
                self.config,
                None,
            )

        torch.testing.assert_close(
            tendency.displacement_yx,
            estimates[0][0],
        )
        torch.testing.assert_close(
            tendency.log_growth_per_step,
            estimates[0][1],
        )
        self.assertEqual(
            tendency.motion_pair_selection,
            TendencyPairSelection.EARLIER,
        )
        self.assertEqual(
            tendency.growth_pair_selection,
            TendencyPairSelection.EARLIER,
        )
        self.assertEqual(tendency.tendency_pair_count, 1)
        self.assertEqual(float(tendency.minimum_phase_correlation_psr), 30.0)
        self.assertTrue(tendency.motion_pair_conflict)
        self.assertTrue(tendency.growth_pair_conflict)

    def test_high_confidence_long_pair_replaces_lone_adjacent_pair(
        self,
    ) -> None:
        nowcast_module = import_module("advar.nowcast")
        values = torch.zeros((3, 2, 2), dtype=torch.float64)
        masks = torch.ones_like(values, dtype=torch.bool)
        adjacent = (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor(0.01, dtype=torch.float64),
            torch.tensor(8.1, dtype=torch.float64),
        )
        long = (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor(0.01, dtype=torch.float64),
            torch.tensor(30.0, dtype=torch.float64),
        )

        with (
            patch.object(
                nowcast_module,
                "_estimate_available_pair",
                side_effect=self._pair_estimates(
                    nowcast_module,
                    adjacent,
                    None,
                    long,
                ),
            ),
            patch.object(
                nowcast_module,
                "_growth_evidence_aligned_with_motion",
                side_effect=(
                    self._growth_evidence(nowcast_module, adjacent[1]),
                    self._growth_evidence(nowcast_module, long[1]),
                ),
            ),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                values,
                masks,
                values,
                self.config,
                None,
            )

        torch.testing.assert_close(tendency.displacement_yx, long[0])
        torch.testing.assert_close(tendency.log_growth_per_step, long[1])
        self.assertEqual(
            tendency.motion_pair_selection,
            TendencyPairSelection.LONG,
        )
        self.assertEqual(
            tendency.growth_pair_selection,
            TendencyPairSelection.LONG,
        )
        self.assertEqual(tendency.tendency_pair_count, 1)
        self.assertEqual(float(tendency.minimum_phase_correlation_psr), 30.0)
        self.assertFalse(tendency.motion_pair_conflict)
        self.assertFalse(tendency.growth_pair_conflict)
        torch.testing.assert_close(
            tendency.source_displacement_yx[0],
            2.0 * long[0],
        )
        self.assertEqual(
            tendency.source_usable.tolist(),
            [True, False, True],
        )

    def test_recent_pair_excludes_unconnected_earliest_source(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        values = torch.zeros((3, 2, 2), dtype=torch.float64)
        masks = torch.ones_like(values, dtype=torch.bool)
        recent = (
            torch.tensor([1.0, -2.0], dtype=torch.float64),
            torch.tensor(0.05, dtype=torch.float64),
            torch.tensor(12.0, dtype=torch.float64),
        )

        with patch.object(
            nowcast_module,
            "_estimate_available_pair",
            side_effect=self._pair_estimates(
                nowcast_module,
                None,
                recent,
                None,
            ),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                values,
                masks,
                values,
                self.config,
                None,
            )

        self.assertEqual(
            tendency.motion_pair_selection,
            TendencyPairSelection.RECENT,
        )
        self.assertEqual(tendency.source_usable.tolist(), [False, True, True])
        torch.testing.assert_close(
            tendency.source_displacement_yx[1],
            recent[0],
        )
        self.assertEqual(float(tendency.source_log_growth[1]), 0.05)

    def test_long_pair_conflict_without_confidence_advantage_is_independent(
        self,
    ) -> None:
        nowcast_module = import_module("advar.nowcast")
        values = torch.zeros((3, 2, 2), dtype=torch.float64)
        masks = torch.ones_like(values, dtype=torch.bool)
        adjacent = (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor(0.2, dtype=torch.float64),
            torch.tensor(12.0, dtype=torch.float64),
        )
        long = (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor(-0.2, dtype=torch.float64),
            torch.tensor(24.0, dtype=torch.float64),
        )

        with (
            patch.object(
                nowcast_module,
                "_estimate_available_pair",
                side_effect=self._pair_estimates(
                    nowcast_module,
                    adjacent,
                    None,
                    long,
                ),
            ),
            patch.object(
                nowcast_module,
                "_growth_evidence_aligned_with_motion",
                side_effect=(
                    self._growth_evidence(nowcast_module, adjacent[1]),
                    self._growth_evidence(nowcast_module, long[1]),
                ),
            ),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                values,
                masks,
                values,
                self.config,
                None,
            )

        torch.testing.assert_close(tendency.displacement_yx, adjacent[0])
        self.assertEqual(float(tendency.log_growth_per_step), 0.0)
        self.assertEqual(
            tendency.motion_pair_selection,
            TendencyPairSelection.SINGLE,
        )
        self.assertEqual(
            tendency.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertEqual(tendency.motion_pair_count, 1)
        self.assertEqual(tendency.growth_pair_count, 0)
        self.assertEqual(tendency.tendency_pair_count, 1)
        self.assertEqual(float(tendency.minimum_phase_correlation_psr), 12.0)
        self.assertFalse(tendency.motion_pair_conflict)
        self.assertTrue(tendency.growth_pair_conflict)
        torch.testing.assert_close(
            tendency.source_log_growth,
            torch.zeros(3, dtype=torch.float64),
        )
        self.assertEqual(tendency.source_usable.tolist(), [False, False, True])

    def test_long_pair_confidence_ignores_irrelevant_domain_coverage(
        self,
    ) -> None:
        nowcast_module = import_module("advar.nowcast")
        values = torch.zeros((3, 2, 2), dtype=torch.float64)
        masks = torch.ones_like(values, dtype=torch.bool)
        masks[1, 0] = False
        adjacent = (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(12.0, dtype=torch.float64),
        )
        long = (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(20.0, dtype=torch.float64),
        )

        with (
            patch.object(
                nowcast_module,
                "_estimate_available_pair",
                side_effect=self._pair_estimates(
                    nowcast_module,
                    adjacent,
                    None,
                    long,
                ),
            ),
            patch.object(
                nowcast_module,
                "_growth_evidence_aligned_with_motion",
                side_effect=(
                    self._growth_evidence(nowcast_module, adjacent[1]),
                    self._growth_evidence(nowcast_module, long[1]),
                ),
            ),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                values,
                masks,
                values,
                self.config,
                None,
            )

        self.assertEqual(
            tendency.motion_pair_selection,
            TendencyPairSelection.SINGLE,
        )
        self.assertEqual(
            tendency.growth_pair_selection,
            TendencyPairSelection.SINGLE,
        )

    def test_equal_zero_pair_confidence_has_no_false_advantage(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        adjacent = torch.tensor(1.0, dtype=torch.float64)
        long = torch.tensor(2.0, dtype=torch.float64)
        zero = torch.zeros((), dtype=torch.float64)

        selected = nowcast_module._select_single_adjacent_or_long_component(
            adjacent,
            long,
            zero,
            zero,
            inconsistent=False,
            config=self.config,
        )
        conflict = nowcast_module._select_single_adjacent_or_long_component(
            adjacent,
            long,
            zero,
            zero,
            inconsistent=True,
            config=self.config,
        )

        self.assertEqual(selected[3], TendencyPairSelection.SINGLE)
        self.assertEqual(conflict[3], TendencyPairSelection.PERSISTENCE)

    def test_pair_velocity_disagreement_is_resolution_invariant(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        config = NowcastConfig(
            maximum_pair_velocity_disagreement_mps=10.0,
        )
        common = {
            "valid_times": (
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            "projection": "EPSG:5179",
        }
        fine = RadarGridTimeContract(
            dx_m=250.0,
            dy_m=250.0,
            grid_hash="1" * 64,
            **common,
        )
        coarse = RadarGridTimeContract(
            dx_m=1000.0,
            dy_m=1000.0,
            grid_hash="2" * 64,
            **common,
        )

        for velocity, expected in ((6.0, False), (12.0, True)):
            with self.subTest(velocity=velocity):
                fine_difference = torch.tensor(
                    [0.0, velocity * 600.0 / fine.dx_m]
                )
                coarse_difference = torch.tensor(
                    [0.0, velocity * 600.0 / coarse.dx_m]
                )
                fine_disagreement_mps = (
                    nowcast_module._motion_disagreement_mps(
                        torch.zeros(2),
                        fine_difference,
                        config,
                        fine,
                    )
                )
                coarse_disagreement_mps = (
                    nowcast_module._motion_disagreement_mps(
                        torch.zeros(2),
                        coarse_difference,
                        config,
                        coarse,
                    )
                )
                self.assertEqual(float(fine_disagreement_mps), velocity)
                self.assertEqual(float(coarse_disagreement_mps), velocity)
                self.assertEqual(
                    nowcast_module._motion_pairs_are_inconsistent(
                        torch.linalg.vector_norm(fine_difference),
                        fine_disagreement_mps,
                        config,
                    ),
                    expected,
                )
                self.assertEqual(
                    nowcast_module._motion_pairs_are_inconsistent(
                        torch.linalg.vector_norm(coarse_difference),
                        coarse_disagreement_mps,
                        config,
                    ),
                    expected,
                )

    def test_conflicting_component_uses_recent_pair_with_psr_advantage(
        self,
    ) -> None:
        nowcast_module = import_module("advar.nowcast")
        selected, indices, selection = nowcast_module._combine_pair_component(
            torch.tensor(9.0),
            torch.tensor(-9.0),
            torch.tensor(8.1),
            torch.tensor(30.0),
            inconsistent=True,
            config=self.config,
        )

        self.assertEqual(float(selected), -9.0)
        self.assertEqual(indices, (1,))
        self.assertEqual(selection, TendencyPairSelection.RECENT)

    def test_consistent_component_uses_psr_and_recency_weights(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        selected, indices, selection = nowcast_module._combine_pair_component(
            torch.tensor(0.0),
            torch.tensor(10.0),
            torch.tensor(10.0),
            torch.tensor(30.0),
            inconsistent=False,
            config=self.config,
        )

        expected = (self.config.recent_weight * 30.0 * 10.0) / (
            (1.0 - self.config.recent_weight) * 10.0
            + self.config.recent_weight * 30.0
        )
        self.assertAlmostEqual(float(selected), expected, places=5)
        self.assertEqual(indices, (0, 1))
        self.assertEqual(selection, TendencyPairSelection.BLENDED)

    def test_low_psr_pairs_fail_closed_to_persistence(self) -> None:
        generator = torch.Generator().manual_seed(7)
        frames = 10.0 + 30.0 * torch.rand(
            3,
            32,
            32,
            generator=generator,
            dtype=torch.float64,
        )

        state, metadata = estimate_state_with_metadata(frames, self.config)

        torch.testing.assert_close(
            state.displacement_yx,
            torch.zeros_like(state.displacement_yx),
        )
        torch.testing.assert_close(
            state.log_growth_per_step,
            torch.zeros_like(state.log_growth_per_step),
        )
        self.assertEqual(metadata.tendency_pair_count, 0)
        self.assertEqual(metadata.tendency_source, TendencySource.NONE)
        self.assertTrue(torch.isnan(metadata.minimum_phase_correlation_psr))

    def test_missing_first_frame_uses_remaining_observation_pair(
        self,
    ) -> None:
        dbz = linear_to_dbz(self.echo, self.config)
        frames = torch.stack((torch.full_like(dbz, torch.nan), dbz, dbz))
        result = nowcast(frames, self.config)
        forecast, metadata = result.forecast_dbz, result.metadata

        self.assertEqual(
            metadata.data_status,
            DataStatus.PARTIAL,
        )
        self.assertEqual(float(metadata.coverage_by_frame[-1]), 1.0)
        self.assertTrue(bool(torch.all(torch.isfinite(forecast))))

    def test_moving_echo_uses_only_time_normalized_available_pairs(self) -> None:
        frames, _ = self._moving_gaussian_frames()
        complete_state, complete_metadata = estimate_state_with_metadata(
            frames,
            self.config,
        )
        self.assertEqual(complete_metadata.tendency_pair_count, 2)
        tolerance = 0.15 * torch.linalg.vector_norm(
            complete_state.displacement_yx
        )

        for missing_index in range(3):
            with self.subTest(missing_index=missing_index):
                partial = frames.clone()
                partial[missing_index] = torch.nan
                state, metadata = estimate_state_with_metadata(
                    partial,
                    self.config,
                )
                error = torch.linalg.vector_norm(
                    state.displacement_yx
                    - complete_state.displacement_yx
                )
                self.assertLessEqual(float(error), float(tolerance))
                self.assertEqual(metadata.tendency_pair_count, 1)
                self.assertLess(
                    abs(float(state.log_growth_per_step)),
                    1.0e-4,
                )

    def test_near_echo_qc_hole_rejects_mask_dominated_motion(self) -> None:
        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float64),
            torch.arange(64, dtype=torch.float64),
            indexing="ij",
        )
        displacement = torch.tensor([1.2, -0.8], dtype=torch.float64)
        frames = torch.stack(
            [
                -10.0
                + 50.0
                * torch.exp(
                    -(
                        (y - (28.0 + step * displacement[0])) ** 2
                        + (x - (34.0 + step * displacement[1])) ** 2
                    )
                    / 50.0
                )
                for step in range(3)
            ]
        )

        for hole_size in (1, 2, 3, 6):
            with self.subTest(hole_size=hole_size):
                qc_mask = torch.ones_like(frames, dtype=torch.bool)
                qc_mask[
                    :,
                    28 : 28 + hole_size,
                    34 : 34 + hole_size,
                ] = False
                state, metadata = estimate_state_with_metadata(
                    frames,
                    self.config,
                    qc_mask=qc_mask,
                )

                torch.testing.assert_close(
                    state.displacement_yx,
                    torch.zeros_like(state.displacement_yx),
                )
                torch.testing.assert_close(
                    state.log_growth_per_step,
                    torch.zeros_like(state.log_growth_per_step),
                )
                self.assertEqual(metadata.tendency_pair_count, 0)

    def test_irregular_mask_rejects_mask_dominated_motion(self) -> None:
        frames, _ = self._moving_gaussian_frames()
        y, x = torch.meshgrid(
            torch.arange(64),
            torch.arange(64),
            indexing="ij",
        )
        mask = ((3 * y + 5 * x) % 10) < 7
        qc_mask = mask.expand_as(frames)

        state, metadata = estimate_state_with_metadata(
            frames,
            self.config,
            qc_mask=qc_mask,
        )

        torch.testing.assert_close(
            state.displacement_yx,
            torch.zeros_like(state.displacement_yx),
        )
        torch.testing.assert_close(
            state.log_growth_per_step,
            torch.zeros_like(state.log_growth_per_step),
        )
        self.assertEqual(metadata.tendency_pair_count, 0)

    def test_far_qc_hole_does_not_reject_echo_pair(self) -> None:
        frames, _ = self._moving_gaussian_frames()
        complete = estimate_state(frames, self.config)
        qc_mask = torch.ones_like(frames, dtype=torch.bool)
        qc_mask[:, 0, 0] = False

        partial, metadata = estimate_state_with_metadata(
            frames,
            self.config,
            qc_mask=qc_mask,
        )

        torch.testing.assert_close(
            partial.displacement_yx,
            complete.displacement_yx,
            atol=0.05,
            rtol=0.0,
        )
        self.assertEqual(metadata.tendency_pair_count, 2)

    def test_middle_missing_normalizes_twenty_minute_growth(self) -> None:
        factor = 1.25
        frames = linear_to_dbz(
            torch.stack((self.echo, self.echo * factor, self.echo * factor**2)),
            self.config,
        )
        frames[1] = torch.nan

        state = estimate_state(frames, self.config)

        self.assertAlmostEqual(
            float(state.log_growth_per_step),
            float(torch.log(torch.tensor(factor))),
            places=3,
        )

    def test_middle_missing_rejects_motion_at_search_boundary(self) -> None:
        echoes = torch.zeros((3, 64, 64), dtype=torch.float64)
        for step in range(3):
            echoes[step, 32, 5 + 20 * step] = 1.0e5
        frames = linear_to_dbz(echoes, self.config)
        frames[1] = torch.nan

        state = estimate_state(frames, self.config)

        torch.testing.assert_close(
            state.displacement_yx,
            torch.zeros_like(state.displacement_yx),
        )

    def test_pair_without_common_echo_does_not_slow_valid_motion(self) -> None:
        echoes = torch.zeros((3, 64, 64), dtype=torch.float64)
        for step in range(3):
            echoes[step, 32, 20 + 4 * step] = 1.0e5
        frames = linear_to_dbz(echoes, self.config)
        frames[2, :, 28] = torch.nan

        state = estimate_state(frames, self.config)

        torch.testing.assert_close(
            state.displacement_yx,
            torch.tensor([0.0, 4.0], dtype=torch.float64),
            atol=0.1,
            rtol=0.0,
        )
        self.assertLess(
            abs(float(state.log_growth_per_step)),
            1.0e-5,
        )

    def test_observation_pair_precedes_conflicting_background_pair(self) -> None:
        frames, _ = self._moving_gaussian_frames()
        observation_pair = frames.clone()
        observation_pair[2] = torch.nan
        background = frames[0].expand_as(frames).clone()

        state, metadata = estimate_state_with_metadata(
            observation_pair,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )
        expected = estimate_state(
            observation_pair,
            self.config,
        ).displacement_yx

        torch.testing.assert_close(
            state.displacement_yx,
            expected,
            atol=0.05,
            rtol=0.0,
        )
        self.assertEqual(metadata.tendency_pair_count, 1)
        self.assertEqual(
            metadata.tendency_source,
            TendencySource.OBSERVATION,
        )
        self.assertTrue(metadata.background_used)
        self.assertEqual(metadata.background_contribution_fraction, 1.0)

    def test_unconnected_observation_does_not_precede_current_background(
        self,
    ) -> None:
        frames = torch.full((3, 8, 8), torch.nan, dtype=torch.float64)
        frames[1] = 30.0
        background = torch.full_like(frames, self.config.min_dbz)

        result = nowcast(
            frames,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        torch.testing.assert_close(
            result.forecast_dbz[0],
            torch.full((8, 8), self.config.min_dbz, dtype=torch.float64),
        )

    def test_direct_clear_observation_precedes_echo_background(self) -> None:
        frames = torch.full((3, 8, 8), torch.nan, dtype=torch.float64)
        frames[2, 4, 4] = self.config.min_dbz
        background = torch.full_like(frames, 30.0)

        result = nowcast(
            frames,
            self.config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        self.assertAlmostEqual(
            float(result.forecast_dbz[0, 4, 4]),
            self.config.min_dbz,
            places=5,
        )
        self.assertGreater(float(result.forecast_dbz[0, 4, 5]), 29.0)

    def test_partial_latest_frame_is_completed_from_previous_state(self) -> None:
        frames = torch.full((3, 8, 8), 20.0)
        frames[2] = torch.nan
        frames[2, 4, 4] = 20.0

        result = nowcast(frames, self.config)

        torch.testing.assert_close(
            result.metadata.source_support,
            torch.ones_like(result.metadata.source_support),
        )
        expected_verified = torch.zeros_like(
            result.metadata.verified_source_support
        )
        expected_verified[4, 4] = 1.0
        torch.testing.assert_close(
            result.metadata.verified_source_support,
            expected_verified,
        )
        self.assertTrue(bool(torch.all(torch.isfinite(result.forecast_dbz))))
        torch.testing.assert_close(
            result.forecast_dbz[0],
            torch.full((8, 8), 20.0),
        )

    def test_local_path_verification_rejects_opposing_echo_motion(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        config = replace(self.config, pair_echo_dilation_px=1)
        linear = torch.zeros((3, 16, 16), dtype=torch.float64)
        masks = torch.zeros_like(linear, dtype=torch.bool)
        echo = dbz_to_linear(linear.new_tensor(20.0), config)
        linear[1, 3:5, 3:5] = echo
        linear[1, 11:13, 11:13] = echo
        linear[2, 3:5, 5:7] = echo
        linear[2, 11:13, 9:11] = echo
        masks[1:] = linear[1:] > 0
        growth = nowcast_module._aligned_growth_evidence(
            linear[1],
            linear[2],
            masks[1],
            masks[2],
            linear.new_tensor((0.0, 2.0)),
            config,
            max_log_growth=config.max_log_growth_per_step,
            grid_time_contract=None,
        )
        paths = nowcast_module._single_pair_tendency(
            linear.new_tensor((0.0, 2.0)),
            growth,
            linear.new_tensor(20.0),
            selection=TendencyPairSelection.RECENT,
            source_pair_index=1,
        )

        path_verified, state_verified = (
            nowcast_module._source_verification_masks(
                linear,
                masks,
                paths,
                config,
                None,
            )
        )

        self.assertTrue(bool(torch.all(path_verified[1, 3:5, 5:7])))
        self.assertFalse(bool(torch.any(path_verified[1, 11:13, 13:15])))
        torch.testing.assert_close(state_verified, path_verified)

    def test_local_state_verification_rejects_amplitude_mismatch(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        config = replace(self.config, pair_echo_dilation_px=0)
        linear = torch.zeros((3, 16, 16), dtype=torch.float64)
        masks = torch.zeros_like(linear, dtype=torch.bool)
        linear[1, 3:5, 3:5] = dbz_to_linear(
            linear.new_tensor(20.0),
            config,
        )
        linear[1, 11:13, 11:13] = dbz_to_linear(
            linear.new_tensor(40.0),
            config,
        )
        linear[2, 3:5, 3:5] = dbz_to_linear(
            linear.new_tensor(20.0),
            config,
        )
        linear[2, 11:13, 11:13] = dbz_to_linear(
            linear.new_tensor(10.0),
            config,
        )
        masks[1:] = linear[1:] > 0
        zero_motion = linear.new_zeros(2)
        growth = nowcast_module._aligned_growth_evidence(
            linear[1],
            linear[2],
            masks[1],
            masks[2],
            zero_motion,
            config,
            max_log_growth=config.max_log_growth_per_step,
            grid_time_contract=None,
        )
        self.assertTrue(growth.available)
        paths = nowcast_module._single_pair_tendency(
            zero_motion,
            growth,
            linear.new_tensor(20.0),
            selection=TendencyPairSelection.RECENT,
            source_pair_index=1,
        )
        paths = replace(
            paths,
            source_log_growth=torch.zeros_like(paths.source_log_growth),
        )

        path_verified, state_verified = (
            nowcast_module._source_verification_masks(
                linear,
                masks,
                paths,
                config,
                None,
            )
        )

        self.assertTrue(bool(torch.all(path_verified[1, 3:5, 3:5])))
        self.assertTrue(bool(torch.all(path_verified[1, 11:13, 11:13])))
        self.assertTrue(bool(torch.all(state_verified[1, 3:5, 3:5])))
        self.assertFalse(bool(torch.any(state_verified[1, 11:13, 11:13])))

    def test_path_without_growth_evidence_is_not_state_verified(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        linear = torch.zeros((3, 8, 8), dtype=torch.float64)
        masks = torch.zeros_like(linear, dtype=torch.bool)
        echo = dbz_to_linear(linear.new_tensor(20.0), self.config)
        linear[1, 4, 3] = echo
        linear[2, 4, 4] = echo
        masks[1:] = linear[1:] > 0
        growth = nowcast_module._aligned_growth_evidence(
            linear[1],
            linear[2],
            masks[1],
            masks[2],
            linear.new_tensor((0.0, 1.0)),
            self.config,
            max_log_growth=self.config.max_log_growth_per_step,
            grid_time_contract=None,
        )
        self.assertFalse(growth.available)
        paths = nowcast_module._single_pair_tendency(
            linear.new_tensor((0.0, 1.0)),
            growth,
            linear.new_tensor(20.0),
            selection=TendencyPairSelection.RECENT,
            source_pair_index=1,
        )

        path_verified, state_verified = (
            nowcast_module._source_verification_masks(
                linear,
                masks,
                paths,
                self.config,
                None,
            )
        )

        self.assertTrue(bool(path_verified[1, 4, 4]))
        self.assertFalse(bool(state_verified[1, 4, 4]))

    def test_operational_publication_excludes_unverified_persistence(
        self,
    ) -> None:
        frames = torch.full((3, 8, 8), 20.0)
        frames[2] = torch.nan
        frames[2, 4, 4] = 20.0
        config = replace(
            self.config,
            minimum_publish_verified_support=0.95,
        )

        result = nowcast(frames, config)

        self.assertEqual(int(result.valid_mask[0].sum()), 1)
        self.assertTrue(bool(result.valid_mask[0, 4, 4]))
        self.assertFalse(bool(result.valid_mask[0, 3, 3]))
        torch.testing.assert_close(
            result.forecast_verified_support[0],
            result.metadata.verified_source_support,
        )

    def test_sparse_recent_frames_do_not_discard_complete_older_state(
        self,
    ) -> None:
        frames = torch.full((3, 8, 8), 20.0)
        frames[1:] = torch.nan
        frames[1, 3, 3] = 20.0
        frames[2, 4, 4] = 20.0

        result = nowcast(frames, self.config)

        torch.testing.assert_close(
            result.metadata.source_support,
            torch.ones_like(result.metadata.source_support),
        )
        self.assertTrue(bool(torch.all(torch.isfinite(result.forecast_dbz))))
        torch.testing.assert_close(
            result.forecast_dbz[0],
            torch.full((8, 8), 20.0),
        )

    def test_fractional_sparse_merge_preserves_constant_echo(self) -> None:
        echo = torch.ones(8, 8, dtype=torch.float64)
        frame = linear_to_dbz(echo, self.config)
        frames = torch.stack((frame, frame, torch.full_like(frame, torch.nan)))
        frames[1, :, 1::2] = torch.nan
        nowcast_module = import_module("advar.nowcast")
        zero = echo.new_zeros(())
        observation_paths = nowcast_module._single_pair_tendency(
            echo.new_tensor([0.0, 0.5]),
            self._growth_evidence(nowcast_module, zero),
            echo.new_tensor(10.0),
        )
        tendencies = (
            observation_paths,
            TendencySource.OBSERVATION,
            observation_paths,
            observation_paths,
        )

        with patch.object(
            nowcast_module,
            "_estimate_time_normalized_tendencies",
            return_value=tendencies,
        ):
            state, metadata = estimate_state_with_metadata(frames, self.config)

        valid = metadata.source_support > self.config.epsilon
        torch.testing.assert_close(
            state.echo_linear[valid],
            torch.ones_like(state.echo_linear[valid]),
        )

    def test_fractional_support_is_not_promoted_to_full_support(self) -> None:
        frames = torch.full((3, 8, 8), torch.nan, dtype=torch.float64)
        frames[1, 3, 3] = linear_to_dbz(
            torch.tensor(1.0, dtype=torch.float64),
            self.config,
        )
        nowcast_module = import_module("advar.nowcast")
        zero = frames.new_zeros(())
        observation_paths = nowcast_module._single_pair_tendency(
            frames.new_tensor([0.5, 0.5]),
            self._growth_evidence(nowcast_module, zero),
            frames.new_tensor(10.0),
        )
        tendencies = (
            observation_paths,
            TendencySource.OBSERVATION,
            observation_paths,
            observation_paths,
        )

        with patch.object(
            nowcast_module,
            "_estimate_time_normalized_tendencies",
            return_value=tendencies,
        ):
            state, metadata = estimate_state_with_metadata(frames, self.config)

        self.assertIsNotNone(metadata.source_support)
        support = metadata.source_support
        assert support is not None
        nonzero = support[support > 0]
        self.assertEqual(nonzero.numel(), 4)
        torch.testing.assert_close(
            nonzero,
            torch.full_like(nonzero, 0.25),
        )
        torch.testing.assert_close(support.sum(), support.new_tensor(1.0))
        torch.testing.assert_close(
            state.echo_linear[support > 0],
            torch.ones(4, dtype=torch.float64),
        )

    def test_fractional_observation_support_blends_with_background(self) -> None:
        config = replace(self.config, epsilon=0.02)
        frames = torch.full((3, 8, 8), torch.nan, dtype=torch.float64)
        observation_echo = torch.tensor(100.0, dtype=torch.float64)
        background_echo = torch.tensor(1.0, dtype=torch.float64)
        frames[1, 3, 3] = linear_to_dbz(observation_echo, config)
        background = torch.full_like(
            frames,
            float(linear_to_dbz(background_echo, config)),
        )
        nowcast_module = import_module("advar.nowcast")
        zero = frames.new_zeros(())
        observation_paths = nowcast_module._single_pair_tendency(
            frames.new_tensor([0.0, 0.99]),
            self._growth_evidence(nowcast_module, zero),
            frames.new_tensor(10.0),
        )
        tendencies = (
            observation_paths,
            TendencySource.OBSERVATION,
            observation_paths,
            observation_paths,
        )

        with patch.object(
            nowcast_module,
            "_estimate_time_normalized_tendencies",
            return_value=tendencies,
        ):
            state, metadata = estimate_state_with_metadata(
                frames,
                config,
                background_frames_dbz=background,
                background_age_minutes=10.0,
            )

        expected = 0.99 * observation_echo + 0.01 * background_echo
        torch.testing.assert_close(state.echo_linear[3, 4], expected)
        self.assertTrue(metadata.background_used)
        self.assertGreater(metadata.background_contribution_fraction, 0.9)
        self.assertLess(metadata.observation_state_support_fraction, 0.1)
        self.assertLess(
            metadata.observation_state_support_fraction,
            config.epsilon,
        )
        self.assertAlmostEqual(
            metadata.observation_state_support_fraction
            + metadata.background_state_support_fraction,
            1.0,
        )
        self.assertEqual(metadata.background_age_minutes, 10.0)
        self.assertEqual(
            metadata.observation_path.mode,
            TendencyPairSelection.SINGLE,
        )
        self.assertTrue(metadata.observation_path.extrapolated)
        self.assertEqual(metadata.observation_path.age_minutes, 10.0)
        self.assertEqual(
            metadata.background_path.mode,
            TendencyPairSelection.NONE,
        )
        self.assertEqual(metadata.background_path.age_minutes, 10.0)
        result = forecast_result_from_state(
            state,
            metadata,
            config,
            run=ForecastRunContract.from_inputs(
                config,
                frames,
                torch.isfinite(frames),
                background,
                background_age_minutes=10.0,
            ),
        )
        result.validate_issuance()

    def test_publication_mask_uses_advected_fractional_support(self) -> None:
        config = NowcastConfig(
            horizon_minutes=20,
            min_publish_support=0.6,
        )
        echo = torch.ones(8, 8, dtype=torch.float64)
        support = torch.zeros_like(echo)
        support[3:5, 3:5] = torch.tensor(
            [[1.0, 0.8], [0.6, 0.4]],
            dtype=torch.float64,
        )
        state = RadarState(
            echo_linear=echo,
            displacement_yx=torch.tensor([0.0, 0.5], dtype=torch.float64),
            log_growth_per_step=torch.zeros((), dtype=torch.float64),
        )
        metadata = ForecastMetadata(
            data_status=DataStatus.PARTIAL,
            coverage_by_frame=torch.ones(3, dtype=torch.float64),
            background_used=False,
            background_contribution_fraction=0.0,
            background_age_minutes=None,
            source_support=support,
            observation_source_support=support,
            background_source_support=torch.zeros_like(support),
            path_verified_source_support=support,
            verified_source_support=support,
            observation_verified_source_support=support,
            background_verified_source_support=torch.zeros_like(support),
            motion_disagreement_px=torch.zeros((), dtype=torch.float64),
            motion_disagreement_mps=torch.full(
                (), torch.nan, dtype=torch.float64
            ),
            growth_disagreement=torch.zeros((), dtype=torch.float64),
            minimum_phase_correlation_psr=torch.tensor(
                10.0,
                dtype=torch.float64,
            ),
            tendency_pair_count=1,
            tendency_source=TendencySource.OBSERVATION,
            state_path_source=TendencySource.OBSERVATION,
            state_path_age_minutes=0.0,
            observation_path=StatePathProvenance(age_minutes=0.0),
            motion_pair_count=1,
            motion_pair_selection=TendencyPairSelection.SINGLE,
        )

        latest = linear_to_dbz(state.echo_linear, config)
        frames = torch.stack((latest, latest, latest))
        result = forecast_result_from_state(
            state,
            metadata,
            config,
            run=ForecastRunContract.from_inputs(
                config,
                frames,
                torch.ones_like(frames, dtype=torch.bool),
                None,
            ),
        )
        expected = torch.stack(
            [
                remap(support, step * state.displacement_yx)
                >= config.min_publish_support
                for step in range(1, config.forecast_steps + 1)
            ]
        )

        torch.testing.assert_close(result.valid_mask, expected)
        self.assertTrue(
            bool(torch.all(torch.isfinite(result.forecast_dbz[expected])))
        )
        self.assertTrue(
            bool(torch.all(torch.isnan(result.forecast_dbz[~expected])))
        )

    def test_full_support_uses_an_unknown_inflow_boundary(self) -> None:
        config = NowcastConfig(horizon_minutes=10)
        echo = torch.ones(8, 8, dtype=torch.float64)
        state = RadarState(
            echo_linear=echo,
            displacement_yx=torch.tensor([0.0, 1.0], dtype=torch.float64),
            log_growth_per_step=torch.zeros((), dtype=torch.float64),
        )
        latest = linear_to_dbz(echo, config)
        frames = torch.stack((latest, latest, latest))
        result = forecast_result_from_state(
            state,
            observed_metadata(state),
            config,
            run=ForecastRunContract.from_inputs(
                config,
                frames,
                torch.ones_like(frames, dtype=torch.bool),
                None,
            ),
        )

        self.assertEqual(int(result.valid_mask.sum()), 8 * 7)
        self.assertTrue(
            torch.equal(
                torch.isfinite(result.forecast_dbz),
                result.valid_mask,
            )
        )
        self.assertTrue(bool(torch.all(~result.valid_mask[0, :, 0])))
        self.assertTrue(bool(torch.all(result.valid_mask[0, :, 1:])))

    def test_analysis_window_growth_does_not_decay(self) -> None:
        echo = torch.ones(8, 8, dtype=torch.float64)
        frame = linear_to_dbz(echo, self.config)
        frames = torch.stack(
            (
                frame,
                torch.full_like(frame, torch.nan),
                torch.full_like(frame, torch.nan),
            )
        )
        nowcast_module = import_module("advar.nowcast")
        growth = torch.log(echo.new_tensor(1.2))
        zero = echo.new_zeros(())
        observation_paths = nowcast_module._single_pair_tendency(
            echo.new_zeros(2),
            self._growth_evidence(nowcast_module, growth),
            echo.new_tensor(10.0),
        )
        tendencies = (
            observation_paths,
            TendencySource.OBSERVATION,
            observation_paths,
            observation_paths,
        )

        with patch.object(
            nowcast_module,
            "_estimate_time_normalized_tendencies",
            return_value=tendencies,
        ):
            state, _ = estimate_state_with_metadata(frames, self.config)

        torch.testing.assert_close(
            state.echo_linear,
            torch.full_like(echo, 1.2**2),
        )

    def test_growth_uses_normalized_advected_support(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        previous_mask = torch.zeros(8, 8, dtype=torch.bool)
        previous_mask[:, ::2] = True
        current_mask = torch.ones_like(previous_mask)
        previous = 10.0 * previous_mask.to(torch.float64)
        current = torch.full_like(previous, 10.0)

        evidence = nowcast_module._aligned_growth_evidence(
            previous,
            current,
            previous_mask,
            current_mask,
            torch.tensor([0.0, 0.5], dtype=torch.float64),
            self.config,
            grid_time_contract=None,
        )

        self.assertTrue(evidence.available)
        torch.testing.assert_close(
            evidence.value,
            torch.zeros_like(evidence.value),
        )
        torch.testing.assert_close(
            evidence.alignment_log_error,
            torch.zeros_like(evidence.alignment_log_error),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_growth_with_three_pixels_of_overlap_is_unavailable(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        previous = torch.zeros((4, 4), dtype=torch.float64)
        current = torch.zeros_like(previous)
        previous_mask = torch.zeros_like(previous, dtype=torch.bool)
        current_mask = torch.zeros_like(previous, dtype=torch.bool)
        previous_mask[0, :3] = True
        current_mask[0, :3] = True
        previous[previous_mask] = 10.0
        current[current_mask] = 20.0

        evidence = nowcast_module._aligned_growth_evidence(
            previous,
            current,
            previous_mask,
            current_mask,
            previous.new_zeros(2),
            self.config,
            grid_time_contract=None,
        )

        self.assertFalse(evidence.available)
        self.assertEqual(float(evidence.overlap_support), 3.0)
        self.assertEqual(float(evidence.value), 0.0)
        self.assertEqual(float(evidence.aligned_previous_integral), 30.0)
        self.assertEqual(float(evidence.current_integral), 60.0)

    def test_clear_sky_does_not_inflate_growth_evidence(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        config = NowcastConfig(minimum_growth_overlap_support=0.5)
        previous = torch.zeros((64, 64), dtype=torch.float64)
        current = torch.zeros_like(previous)
        previous[32, 32] = 10.0
        current[32, 32] = 10.0
        mask = torch.ones_like(previous, dtype=torch.bool)

        evidence = nowcast_module._aligned_growth_evidence(
            previous,
            current,
            mask,
            mask,
            previous.new_zeros(2),
            config,
            grid_time_contract=None,
        )

        self.assertTrue(evidence.available)
        self.assertEqual(float(evidence.overlap_support), 1.0)
        self.assertEqual(float(evidence.aligned_previous_integral), 10.0)
        self.assertEqual(float(evidence.current_integral), 10.0)
        self.assertEqual(float(evidence.value), 0.0)

    def test_fractional_support_weights_strong_current_echo(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        config = NowcastConfig(
            minimum_growth_overlap_support=0.005,
            max_log_growth_per_step=math.log(100.0),
        )
        previous = torch.zeros((5, 5), dtype=torch.float64)
        previous[2, 2] = 10.0
        previous_mask = previous > 0
        current = torch.zeros_like(previous)
        current[2, 2] = 100.0
        current_mask = torch.zeros_like(previous_mask)
        current_mask[2, 2] = True

        evidence = nowcast_module._aligned_growth_evidence(
            previous,
            current,
            previous_mask,
            current_mask,
            previous.new_tensor((0.0, 0.99)),
            config,
            grid_time_contract=None,
        )

        self.assertTrue(evidence.available)
        torch.testing.assert_close(
            evidence.overlap_support,
            previous.new_tensor(0.01),
        )
        torch.testing.assert_close(
            evidence.aligned_previous_integral,
            previous.new_tensor(0.1),
        )
        torch.testing.assert_close(
            evidence.current_integral,
            previous.new_tensor(1.0),
        )

    def test_tiny_support_tail_does_not_force_growth_limit(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        previous = torch.zeros((5, 14), dtype=torch.float64)
        previous[2, 1:11] = 10.0
        previous_mask = previous > 0
        displacement = previous.new_tensor((0.0, 0.99))
        moved_support = remap(
            previous_mask.to(dtype=previous.dtype),
            displacement,
        )
        aligned = remap(previous, displacement) / moved_support.clamp_min(
            self.config.epsilon
        )
        current = aligned.clone()
        current[2, 1] = 1000.0
        current_mask = moved_support > self.config.epsilon

        evidence = nowcast_module._aligned_growth_evidence(
            previous,
            current,
            previous_mask,
            current_mask,
            displacement,
            self.config,
            grid_time_contract=None,
        )

        expected_growth = math.log(109.9 / 100.0)
        self.assertTrue(evidence.available)
        self.assertAlmostEqual(float(evidence.value), expected_growth)
        self.assertLess(
            float(evidence.value),
            self.config.max_log_growth_per_step,
        )

    def test_unavailable_growth_is_not_blended_as_zero(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        available = self._growth_evidence(
            nowcast_module,
            torch.tensor(0.05, dtype=torch.float64),
        )
        unavailable = self._growth_evidence(
            nowcast_module,
            torch.tensor(0.0, dtype=torch.float64),
            available=False,
        )

        value, indices, selection, disagreement, conflict = (
            nowcast_module._combine_adjacent_growth_evidence(
                available,
                unavailable,
                torch.tensor(12.0, dtype=torch.float64),
                torch.tensor(30.0, dtype=torch.float64),
                TendencyPairSelection.BLENDED,
                self.config,
            )
        )

        self.assertEqual(float(value), 0.05)
        self.assertEqual(indices, (0,))
        self.assertEqual(selection, TendencyPairSelection.EARLIER)
        self.assertEqual(float(disagreement), 0.0)
        self.assertFalse(conflict)

    def test_growth_confidence_uses_echo_evidence_not_only_motion_psr(
        self,
    ) -> None:
        nowcast_module = import_module("advar.nowcast")
        sparse_high_psr = self._growth_evidence(
            nowcast_module,
            torch.tensor(0.2, dtype=torch.float64),
            overlap_support=4.0,
            mean_echo=10.0,
        )
        broad_lower_psr = self._growth_evidence(
            nowcast_module,
            torch.tensor(-0.2, dtype=torch.float64),
            overlap_support=400.0,
            mean_echo=10.0,
        )

        value, indices, selection, _, conflict = (
            nowcast_module._combine_adjacent_growth_evidence(
                sparse_high_psr,
                broad_lower_psr,
                torch.tensor(30.0, dtype=torch.float64),
                torch.tensor(10.0, dtype=torch.float64),
                TendencyPairSelection.BLENDED,
                self.config,
            )
        )

        self.assertTrue(conflict)
        self.assertEqual(indices, (1,))
        self.assertEqual(selection, TendencyPairSelection.RECENT)
        torch.testing.assert_close(value, broad_lower_psr.value)

    def test_growth_confidence_penalizes_shape_misalignment(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        aligned = self._growth_evidence(
            nowcast_module,
            torch.tensor(0.05, dtype=torch.float64),
            alignment_log_error=0.0,
        )
        deformed = self._growth_evidence(
            nowcast_module,
            torch.tensor(0.05, dtype=torch.float64),
            alignment_log_error=9.0,
        )
        psr = torch.tensor(12.0, dtype=torch.float64)

        aligned_confidence = nowcast_module._growth_evidence_confidence(
            aligned,
            psr,
            span_penalty=1.0,
            config=self.config,
        )
        deformed_confidence = nowcast_module._growth_evidence_confidence(
            deformed,
            psr,
            span_penalty=1.0,
            config=self.config,
        )

        torch.testing.assert_close(
            deformed_confidence,
            aligned_confidence / 10.0,
        )
        extinction = replace(
            aligned,
            current_integral=aligned.value.new_zeros(()),
        )
        extinction_confidence = (
            nowcast_module._growth_evidence_confidence(
                extinction,
                psr,
                span_penalty=1.0,
                config=self.config,
            )
        )
        self.assertGreater(float(extinction_confidence), 0.0)

    def test_growth_confidence_uses_physical_area_across_resolutions(
        self,
    ) -> None:
        nowcast_module = import_module("advar.nowcast")
        value = torch.tensor(0.05, dtype=torch.float64)
        fine = self._growth_evidence(
            nowcast_module,
            value,
            overlap_support=64.0,
            overlap_area_km2=4.0,
            mean_echo=10.0,
        )
        coarse = self._growth_evidence(
            nowcast_module,
            value,
            overlap_support=1.0,
            overlap_area_km2=4.0,
            mean_echo=10.0,
        )
        psr = torch.tensor(12.0, dtype=torch.float64)

        fine_confidence = nowcast_module._growth_evidence_confidence(
            fine,
            psr,
            span_penalty=1.0,
            config=self.config,
        )
        coarse_confidence = nowcast_module._growth_evidence_confidence(
            coarse,
            psr,
            span_penalty=1.0,
            config=self.config,
        )

        torch.testing.assert_close(fine_confidence, coarse_confidence)

    def test_unavailable_growth_records_no_pair_selection(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        unavailable = self._growth_evidence(
            nowcast_module,
            torch.tensor(0.0, dtype=torch.float64),
            available=False,
        )

        estimate = nowcast_module._single_pair_tendency(
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            unavailable,
            torch.tensor(12.0, dtype=torch.float64),
        )

        self.assertTrue(estimate.future_available)
        self.assertEqual(estimate.motion_pair_count, 1)
        self.assertEqual(estimate.growth_pair_count, 0)
        self.assertEqual(
            estimate.growth_pair_selection,
            TendencyPairSelection.NONE,
        )
        self.assertFalse(estimate.growth_pair_conflict)

    def test_missing_aligned_previous_echo_is_not_maximum_growth(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        previous = torch.zeros((4, 4), dtype=torch.float64)
        current = torch.full_like(previous, 10.0)
        mask = torch.ones_like(previous, dtype=torch.bool)

        evidence = nowcast_module._aligned_growth_evidence(
            previous,
            current,
            mask,
            mask,
            previous.new_zeros(2),
            self.config,
            grid_time_contract=None,
        )

        self.assertFalse(evidence.available)
        self.assertEqual(float(evidence.value), 0.0)
        self.assertEqual(float(evidence.aligned_previous_integral), 0.0)
        self.assertEqual(float(evidence.current_integral), 160.0)

    def test_growth_overlap_area_is_resolution_invariant(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        config = NowcastConfig(
            minimum_growth_overlap_support=0.5,
            minimum_growth_overlap_area_km2=3.5,
        )
        common_contract = {
            "valid_times": (
                "2026-08-02T00:00:00Z",
                "2026-08-02T00:10:00Z",
                "2026-08-02T00:20:00Z",
            ),
            "projection": "EPSG:5179",
            "grid_hash": "9" * 64,
        }
        fine_contract = RadarGridTimeContract(
            dx_m=250.0,
            dy_m=250.0,
            **common_contract,
        )
        coarse_contract = RadarGridTimeContract(
            dx_m=2000.0,
            dy_m=2000.0,
            **common_contract,
        )
        fine_previous = torch.full((8, 8), 10.0, dtype=torch.float64)
        fine_current = torch.full((8, 8), 12.0, dtype=torch.float64)
        coarse_previous = torch.full((2, 2), 10.0, dtype=torch.float64)
        coarse_current = torch.full((2, 2), 12.0, dtype=torch.float64)
        fine_mask = torch.ones_like(fine_previous, dtype=torch.bool)
        coarse_mask = torch.zeros_like(coarse_previous, dtype=torch.bool)
        coarse_mask[0, 0] = True

        fine_evidence = nowcast_module._aligned_growth_evidence(
            fine_previous,
            fine_current,
            fine_mask,
            fine_mask,
            fine_previous.new_zeros(2),
            config,
            grid_time_contract=fine_contract,
        )
        coarse_evidence = nowcast_module._aligned_growth_evidence(
            coarse_previous,
            coarse_current,
            coarse_mask,
            coarse_mask,
            coarse_previous.new_zeros(2),
            config,
            grid_time_contract=coarse_contract,
        )

        self.assertTrue(fine_evidence.available)
        self.assertTrue(coarse_evidence.available)
        torch.testing.assert_close(
            fine_evidence.overlap_area_km2,
            coarse_evidence.overlap_area_km2,
        )
        torch.testing.assert_close(
            fine_evidence.value,
            coarse_evidence.value,
        )
        self.assertAlmostEqual(float(fine_evidence.value), math.log(1.2))
        self.assertEqual(float(fine_evidence.overlap_area_km2), 4.0)

    def test_high_pair_psr_does_not_replace_missing_growth_evidence(
        self,
    ) -> None:
        nowcast_module = import_module("advar.nowcast")
        values = torch.zeros((3, 4, 4), dtype=torch.float64)
        masks = torch.ones_like(values, dtype=torch.bool)
        first = (
            values.new_tensor([0.0, 1.0]),
            values.new_tensor(0.05),
            values.new_tensor(12.0),
        )
        second = (
            values.new_tensor([0.0, 1.0]),
            values.new_tensor(0.0),
            values.new_tensor(30.0),
        )
        first_evidence = self._growth_evidence(
            nowcast_module,
            first[1],
        )
        second_evidence = self._growth_evidence(
            nowcast_module,
            second[1],
            available=False,
        )

        with (
            patch.object(
                nowcast_module,
                "_estimate_available_pair",
                side_effect=self._pair_estimates(
                    nowcast_module,
                    first,
                    second,
                ),
            ),
            patch.object(
                nowcast_module,
                "_growth_evidence_aligned_with_motion",
                side_effect=(first_evidence, second_evidence),
            ),
        ):
            tendency = nowcast_module._estimate_source_tendencies(
                values,
                masks,
                values,
                self.config,
                None,
            )

        self.assertEqual(float(tendency.log_growth_per_step), 0.05)
        self.assertEqual(tendency.growth_pair_count, 1)
        self.assertEqual(
            tendency.growth_pair_selection,
            TendencyPairSelection.EARLIER,
        )
        self.assertEqual(float(tendency.growth_disagreement), 0.0)

    def test_earlier_pair_does_not_extrapolate_missing_latest_frame(self) -> None:
        dbz = linear_to_dbz(self.echo, self.config)
        frames = torch.stack((dbz, dbz, torch.full_like(dbz, torch.nan)))
        result = nowcast(frames, self.config)
        forecast, metadata = result.forecast_dbz, result.metadata

        self.assertEqual(
            metadata.data_status,
            DataStatus.PARTIAL,
        )
        self.assertEqual(float(metadata.coverage_by_frame[-1]), 0.0)
        self.assertFalse(bool(torch.any(torch.isfinite(forecast))))
        self.assertFalse(bool(torch.any(metadata.source_support)))

    def test_empty_echo_uses_persistence_fallback(self) -> None:
        frames = torch.full((3, 32, 32), self.config.min_dbz)
        result = nowcast(frames, self.config)
        forecast, state = result.forecast_dbz, result.state

        self.assertEqual(result.metadata.tendency_pair_count, 0)
        torch.testing.assert_close(state.displacement_yx, torch.zeros(2))
        torch.testing.assert_close(state.log_growth_per_step, torch.zeros(()))
        torch.testing.assert_close(
            forecast,
            torch.full((18, 32, 32), self.config.min_dbz),
        )

    def test_long_lead_uses_one_direct_warp(self) -> None:
        displacement = torch.tensor([0.2, -0.3])
        dbz = linear_to_dbz(self.echo, self.config)
        frames = torch.stack((dbz, dbz, dbz))
        state = estimate_state(frames, self.config)
        state = type(state)(
            echo_linear=state.echo_linear,
            displacement_yx=displacement,
            log_growth_per_step=torch.zeros(()),
        )

        forecast = forecast_from_state(state, self.config)
        expected = linear_to_dbz(
            advect(state.echo_linear, 18 * displacement),
            self.config,
        )
        issued = torch.isfinite(forecast[-1])
        self.assertFalse(bool(torch.all(issued)))
        torch.testing.assert_close(forecast[-1][issued], expected[issued])

    def test_full_forecast_remaps_each_lead_once_even_with_audit(self) -> None:
        state = RadarState(
            echo_linear=self.echo,
            displacement_yx=torch.tensor([0.2, -0.3]),
            log_growth_per_step=torch.zeros(()),
        )
        nowcast_module = import_module("advar.nowcast")
        with patch.object(nowcast_module, "remap_core", wraps=remap_core) as kernel:
            latest = linear_to_dbz(state.echo_linear, self.config)
            frames = torch.stack((latest, latest, latest))
            result = forecast_result_from_state(
                state,
                observed_metadata(state),
                self.config,
                run=ForecastRunContract.from_inputs(
                    self.config,
                    frames,
                    torch.ones_like(frames, dtype=torch.bool),
                    None,
                ),
                audit=True,
            )

        self.assertEqual(kernel.call_count, self.config.forecast_steps)
        assert result.audit is not None
        self.assertEqual(len(result.audit.transport), self.config.forecast_steps)

    def test_forecast_config_must_match_its_run_contract(self) -> None:
        state = RadarState(
            echo_linear=self.echo,
            displacement_yx=torch.zeros(2),
            log_growth_per_step=torch.zeros(()),
        )
        latest = linear_to_dbz(state.echo_linear, self.config)
        frames = torch.stack((latest, latest, latest))
        run = ForecastRunContract.from_inputs(
            self.config,
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )

        with self.assertRaisesRegex(ValueError, "run contract"):
            forecast_result_from_state(
                state,
                observed_metadata(state),
                NowcastConfig(max_dbz=60.0),
                run=run,
            )

    def test_forecast_rejects_out_of_contract_state_dynamics(self) -> None:
        latest = linear_to_dbz(self.echo, self.config)
        frames = torch.stack((latest, latest, latest))
        run = ForecastRunContract.from_inputs(
            self.config,
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        states = (
            RadarState(
                echo_linear=self.echo,
                displacement_yx=torch.tensor(
                    (self.config.max_displacement_px + 1.0, 0.0)
                ),
                log_growth_per_step=torch.zeros(()),
            ),
            RadarState(
                echo_linear=self.echo,
                displacement_yx=torch.zeros(2),
                log_growth_per_step=torch.tensor(
                    self.config.max_log_growth_per_step + 0.1
                ),
            ),
        )

        for state in states:
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, "configured limit"):
                    forecast_result_from_state(
                        state,
                        observed_metadata(state),
                        self.config,
                        run=run,
                    )

    def test_issuance_rejects_digest_consistent_invalid_dynamics(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        result = nowcast(frames, self.config)
        invalid_state = replace(
            result.state,
            displacement_yx=torch.tensor(
                (self.config.max_displacement_px + 1.0, 0.0),
                dtype=torch.float64,
            ),
        )
        state_digest = nowcast_module.state_metadata_digest(
            invalid_state,
            result.metadata,
        )
        run_digest = nowcast_module._forecast_run_identity_digest(
            result.run,
            state_digest,
            result.forecast_dbz_digest,
            result.valid_mask_digest,
        )
        invalid = replace(
            result,
            state=invalid_state,
            state_metadata_digest=state_digest,
            forecast_run_digest=run_digest,
        )

        with self.assertRaisesRegex(ValueError, "configured limit"):
            invalid.validate_issuance()

    def test_forecast_rejects_state_grid_outside_run_contract(self) -> None:
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        issued = nowcast(frames, self.config)
        state = replace(
            issued.state,
            echo_linear=issued.state.echo_linear[:4, :4].clone(),
        )
        metadata = replace(
            issued.metadata,
            source_support=issued.metadata.source_support[:4, :4].clone(),
        )

        with self.assertRaisesRegex(ValueError, "run input grid"):
            forecast_result_from_state(
                state,
                metadata,
                self.config,
                run=issued.run,
            )

    def test_issuance_rejects_digest_consistent_invalid_structure(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        issued = nowcast(frames, self.config)

        def reissue(**changes: object) -> object:
            candidate = replace(issued, **changes)
            state_digest = nowcast_module.state_metadata_digest(
                candidate.state,
                candidate.metadata,
            )
            forecast_digest = nowcast_module.tensor_digest(
                candidate.forecast_dbz
            )
            valid_digest = nowcast_module.tensor_digest(candidate.valid_mask)
            return replace(
                candidate,
                state_metadata_digest=state_digest,
                forecast_dbz_digest=forecast_digest,
                valid_mask_digest=valid_digest,
                forecast_run_digest=(
                    nowcast_module._forecast_run_identity_digest(
                        candidate.run,
                        state_digest,
                        forecast_digest,
                        valid_digest,
                    )
                ),
            )

        def without_origin(value: torch.Tensor) -> torch.Tensor:
            updated = value.clone()
            updated[0, 0] = 0.0
            return updated

        metadata_without_origin = replace(
            issued.metadata,
            source_support=without_origin(issued.metadata.source_support),
            observation_source_support=without_origin(
                issued.metadata.observation_source_support
            ),
            path_verified_source_support=without_origin(
                issued.metadata.path_verified_source_support
            ),
            verified_source_support=without_origin(
                issued.metadata.verified_source_support
            ),
            observation_verified_source_support=without_origin(
                issued.metadata.observation_verified_source_support
            ),
        )
        cases = (
            (
                reissue(
                    forecast_dbz=issued.forecast_dbz[:-1].clone(),
                    valid_mask=issued.valid_mask[:-1].clone(),
                ),
                "lead shape",
            ),
            (
                reissue(
                    forecast_dbz=issued.forecast_dbz.clone() + 1.0,
                ),
                "forecast does not close against the issued state",
            ),
            (
                reissue(
                    metadata=metadata_without_origin,
                ),
                "valid mask does not close against the issued state",
            ),
            (
                reissue(
                    metadata=replace(
                        issued.metadata,
                        source_support=(
                            issued.metadata.source_support[:4, :4].clone()
                        ),
                    )
                ),
                "source_support",
            ),
            (
                reissue(
                    metadata=replace(
                        issued.metadata,
                        verified_source_support=(
                            issued.metadata.verified_source_support + 0.1
                        ),
                    )
                ),
                "verified_source_support",
            ),
            (
                reissue(
                    metadata=replace(
                        issued.metadata,
                        observation_path=StatePathProvenance(
                            age_minutes=10.0
                        ),
                    )
                ),
                "aggregate state path provenance",
            ),
            (
                reissue(
                    metadata=replace(
                        issued.metadata,
                        provenance="p1_variational_analysis",
                        dynamics_source=DynamicsSource.P1_VARIATIONAL,
                    )
                ),
                "P1 metadata",
            ),
            (
                reissue(
                    metadata=replace(
                        issued.metadata,
                        background_used=True,
                        background_age_minutes=10.0,
                    )
                ),
                "background usage provenance",
            ),
        )

        for candidate, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    candidate.validate_issuance()

    def test_p1_issuance_rejects_verified_support_without_evidence(
        self,
    ) -> None:
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        issued = nowcast(frames, self.config)
        observation_verified = (
            issued.metadata.observation_verified_source_support.clone()
        )
        observation_verified[0, 0] = 0.0
        metadata = replace(
            issued.metadata,
            provenance="p1_variational_analysis",
            dynamics_source=DynamicsSource.P1_VARIATIONAL,
            state_path_source=TendencySource.NONE,
            state_path_mode=TendencyPairSelection.NONE,
            state_path_pair_count=0,
            state_path_minimum_psr=math.nan,
            state_path_conflict=False,
            state_path_extrapolated=False,
            state_path_age_minutes=None,
            observation_path=StatePathProvenance(),
            background_path=StatePathProvenance(),
            minimum_growth_overlap_support=math.nan,
            minimum_growth_overlap_area_km2=math.nan,
            observation_verified_source_support=observation_verified,
        )
        candidate = forecast_result_from_state(
            issued.state,
            metadata,
            self.config,
            run=issued.run,
        )

        with self.assertRaisesRegex(
            ValueError,
            "verified_source_support must equal source evidence channels",
        ):
            candidate.validate_issuance()

    def test_issuance_rejects_growth_count_evidence_mismatch(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        issued = nowcast(frames, self.config)

        def reissue(metadata: ForecastMetadata) -> object:
            state_digest = nowcast_module.state_metadata_digest(
                issued.state,
                metadata,
            )
            return replace(
                issued,
                metadata=metadata,
                state_metadata_digest=state_digest,
                forecast_run_digest=(
                    nowcast_module._forecast_run_identity_digest(
                        issued.run,
                        state_digest,
                        issued.forecast_dbz_digest,
                        issued.valid_mask_digest,
                    )
                ),
            )

        unused_with_evidence = replace(
            issued.metadata,
            minimum_growth_overlap_support=4.0,
        )
        used_without_evidence = replace(
            issued.metadata,
            growth_pair_count=1,
            growth_pair_selection=TendencyPairSelection.SINGLE,
            tendency_pair_count=1,
            minimum_phase_correlation_psr=issued.forecast_dbz.new_tensor(10.0),
        )

        for metadata, message in (
            (unused_with_evidence, "unused growth pairs"),
            (used_without_evidence, "used growth pairs"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    reissue(metadata).validate_issuance()

    def test_fractional_advection_is_non_negative_and_does_not_gain_mass(
        self,
    ) -> None:
        echo = torch.zeros(32, 32, dtype=torch.float64)
        echo[16, 16] = 1.0e5
        moved = advect(
            echo,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )

        self.assertGreaterEqual(float(moved.min()), 0.0)
        torch.testing.assert_close(moved.sum(), echo.sum())

    def test_roundoff_tolerance_cannot_create_negative_transport_weight(
        self,
    ) -> None:
        echo = torch.zeros(7, 7, dtype=torch.float32)
        echo[3, 3] = 1.0

        moved = advect(
            echo,
            torch.tensor([-1.0e-7, 0.0], dtype=torch.float32),
            frozen_cell=RemapCell(0, 0),
        )

        self.assertGreaterEqual(float(moved.min()), 0.0)

    def test_low_precision_tolerance_does_not_hide_stale_cell(self) -> None:
        for dtype, value in (
            (torch.float16, -0.03125),
            (torch.bfloat16, -0.125),
        ):
            with self.subTest(dtype=dtype):
                echo = torch.ones(4, 4, dtype=dtype)
                with self.assertRaises(FrozenCellMismatchError):
                    advect(
                        echo,
                        torch.tensor([value, 0.0], dtype=dtype),
                        frozen_cell=RemapCell(0, 0),
                    )

    def test_physical_echo_rejects_material_negative_values(self) -> None:
        echo = torch.ones(4, 4, dtype=torch.float64)
        echo[1, 2] = -1.0e-6

        with self.assertRaises(EchoPositivityError):
            validate_physical_echo(echo, name="transport input")
        with self.assertRaises(EchoPositivityError):
            linear_to_dbz(echo, self.config)

    def test_low_precision_echo_rejects_material_negative_values(self) -> None:
        for dtype, value in (
            (torch.float16, -0.03125),
            (torch.bfloat16, -0.25),
        ):
            with self.subTest(dtype=dtype):
                echo = torch.ones(4, 4, dtype=dtype)
                echo[1, 2] = value
                with self.assertRaises(EchoPositivityError):
                    validate_physical_echo(
                        echo,
                        name="low-precision test",
                    )

    def test_high_dynamic_range_does_not_hide_negative_echo(self) -> None:
        for dtype, positive in (
            (torch.float32, 1.0e7),
            (torch.float64, 1.0e15),
        ):
            with self.subTest(dtype=dtype):
                echo = torch.tensor(
                    [[positive, -1.0], [0.0, 0.0]],
                    dtype=dtype,
                )
                with self.assertRaises(EchoPositivityError):
                    validate_physical_echo(echo, name="dynamic range test")

    def test_roundoff_negative_echo_is_corrected_and_reported(self) -> None:
        echo = torch.ones(4, 4, dtype=torch.float64)
        echo[1, 2] = -1.0e-15

        corrected, diagnostics = validate_physical_echo(
            echo,
            name="roundoff test",
        )

        self.assertEqual(float(corrected[1, 2]), 0.0)
        self.assertEqual(diagnostics.corrected_count, 1)

    def test_nonfinite_physical_echo_is_rejected(self) -> None:
        echo = torch.ones(4, 4, dtype=torch.float64)
        echo[0, 0] = torch.nan

        with self.assertRaises(EchoPositivityError):
            validate_physical_echo(echo, name="nonfinite test")

    def test_stale_frozen_remap_cell_is_rejected(self) -> None:
        echo = torch.ones(4, 4, dtype=torch.float64)

        with self.assertRaises(FrozenCellMismatchError):
            advect(
                echo,
                torch.tensor([1.2, 0.3], dtype=torch.float64),
                frozen_cell=RemapCell(0, 0),
            )

    def test_large_displacement_does_not_expand_cell_tolerance(self) -> None:
        echo = torch.ones(4, 4, dtype=torch.float32)

        with self.assertRaises(FrozenCellMismatchError):
            advect(
                echo,
                torch.tensor(
                    [1_000_000.25, 0.0],
                    dtype=torch.float32,
                ),
                frozen_cell=RemapCell(999_998, 0),
            )

    def test_nonfinite_displacement_is_rejected(self) -> None:
        echo = torch.ones(4, 4, dtype=torch.float64)
        for displacement in (
            torch.tensor([torch.inf, 0.0], dtype=torch.float64),
            torch.tensor([0.0, torch.nan], dtype=torch.float64),
        ):
            with self.subTest(displacement=displacement.tolist()):
                with self.assertRaises(ValueError):
                    advect(
                        echo,
                        displacement,
                        frozen_cell=RemapCell(0, 0),
                    )

    def test_transport_diagnostics_close_boundary_budget(self) -> None:
        echo = torch.zeros(8, 8, dtype=torch.float64)
        echo[0, 3] = 10.0
        diagnostics = audit_transport(
            echo,
            torch.tensor([-0.25, 0.5], dtype=torch.float64),
        )

        self.assertLess(diagnostics.echo_budget_error, 1.0e-14)
        self.assertGreater(diagnostics.boundary_outflow_integral, 0.0)

    def test_transport_diagnostics_preserve_roundoff_correction(self) -> None:
        echo = torch.ones(4, 4, dtype=torch.float64)
        echo[1, 1] = -1.0e-15
        corrected, positivity = validate_physical_echo(
            echo,
            name="transport input",
        )
        audit_transport(corrected, torch.zeros(2, dtype=torch.float64))
        self.assertEqual(positivity.corrected_count, 1)
        self.assertGreater(positivity.corrected_integral, 0.0)

    def test_integrator_contract_names_positive_local_remap(self) -> None:
        self.assertEqual(
            FORECAST_INTEGRATOR_VERSION,
            "local-conservative-slice-remap-v2",
        )

    def test_growth_and_decay_keep_echo_positive_for_eighteen_steps(self) -> None:
        for growth in (
            -self.config.max_log_growth_per_step,
            self.config.max_log_growth_per_step,
        ):
            with self.subTest(growth=growth):
                state = RadarState(
                    echo_linear=self.echo.to(torch.float64),
                    displacement_yx=torch.tensor(
                        [0.35, -0.45],
                        dtype=torch.float64,
                    ),
                    log_growth_per_step=torch.tensor(
                        growth,
                        dtype=torch.float64,
                    ),
                )
                forecast = forecast_linear_from_state(state, self.config)

                self.assertTrue(bool(torch.all(torch.isfinite(forecast))))
                self.assertGreaterEqual(float(forecast.min()), 0.0)

    def test_fractional_impulse_has_only_four_local_destinations(self) -> None:
        echo = torch.zeros(32, 32, dtype=torch.float64)
        echo[16, 16] = 1.0e5
        moved = advect(
            echo,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )
        expected = torch.zeros_like(echo)
        expected[16:18, 16:18] = 2.5e4

        torch.testing.assert_close(moved, expected)
        self.assertEqual(int(torch.count_nonzero(moved)), 4)

        footprint = torch.zeros_like(echo, dtype=torch.bool)
        footprint[16:18, 16:18] = True
        moved_dbz = linear_to_dbz(moved, self.config)
        self.assertEqual(
            int(torch.count_nonzero((moved_dbz > 5.0) & ~footprint)),
            0,
        )

    def test_sharp_echoes_remain_local_and_conserve_echo_integral(self) -> None:
        displacement = torch.tensor(
            [0.25, -0.75],
            dtype=torch.float64,
        )
        cases = []

        rectangle = torch.zeros(32, 32, dtype=torch.float64)
        rectangle[10:15, 12:19] = 3.0e4
        cases.append((rectangle, (10, 15, 11, 18)))

        horizontal_band = torch.zeros(32, 32, dtype=torch.float64)
        horizontal_band[14:16, 8:24] = 2.0e4
        cases.append((horizontal_band, (14, 16, 7, 23)))

        vertical_band = torch.zeros(32, 32, dtype=torch.float64)
        vertical_band[8:24, 14:16] = 2.0e4
        cases.append((vertical_band, (8, 24, 13, 15)))

        for echo, (y_start, y_stop, x_start, x_stop) in cases:
            with self.subTest(bounds=(y_start, y_stop, x_start, x_stop)):
                moved = advect(echo, displacement)
                support = torch.zeros_like(echo, dtype=torch.bool)
                support[y_start : y_stop + 1, x_start : x_stop + 1] = True

                self.assertTrue(bool(torch.all(moved[~support] == 0)))
                self.assertGreaterEqual(float(moved.min()), 0.0)
                torch.testing.assert_close(
                    moved.sum(),
                    echo.sum(),
                    atol=1.0e-10,
                    rtol=1.0e-12,
                )

    def test_boundary_outflow_closes_the_echo_integral_budget(self) -> None:
        cases = (
            ((0, 10), (-0.5, 0.0)),
            ((31, 10), (0.5, 0.0)),
            ((10, 0), (0.0, -0.5)),
            ((10, 31), (0.0, 0.5)),
        )
        for source, displacement in cases:
            with self.subTest(source=source, displacement=displacement):
                echo = torch.zeros(32, 32, dtype=torch.float64)
                echo[source] = 1.0e5
                moved = advect(
                    echo,
                    torch.tensor(displacement, dtype=torch.float64),
                )
                expected_outflow = 0.5 * float(echo.sum())
                actual_outflow = float(echo.sum() - moved.sum())

                self.assertAlmostEqual(
                    actual_outflow,
                    expected_outflow,
                    places=9,
                )
                self.assertGreaterEqual(float(moved.min()), 0.0)

    def test_direct_one_pixel_warp_avoids_repeated_half_pixel_diffusion(
        self,
    ) -> None:
        echo = torch.zeros(16, 16, dtype=torch.float64)
        echo[8, 8] = 1.0
        half = torch.tensor([0.0, 0.5], dtype=torch.float64)

        repeated = advect(advect(echo, half), half)
        direct = advect(echo, 2.0 * half)

        torch.testing.assert_close(repeated.sum(), echo.sum())
        torch.testing.assert_close(direct.sum(), echo.sum())
        self.assertEqual(int(torch.count_nonzero(repeated)), 3)
        self.assertEqual(int(torch.count_nonzero(direct)), 1)

        columns = torch.arange(16, dtype=torch.float64)[None, :]
        repeated_center = torch.sum(repeated * columns) / repeated.sum()
        direct_center = torch.sum(direct * columns) / direct.sum()
        torch.testing.assert_close(repeated_center, direct_center)
        self.assertFalse(torch.equal(repeated, direct))

    def test_echo_does_not_wrap_back_after_leaving_domain(self) -> None:
        echo = torch.zeros(32, 32)
        echo[16, 16] = 1.0e5
        moved = advect(echo, torch.tensor([80.0, 0.0]))
        torch.testing.assert_close(moved, torch.zeros_like(moved))

    def test_advection_rejects_integer_echo_and_displacement(self) -> None:
        with self.assertRaisesRegex(TypeError, "echo"):
            advect(
                torch.ones(4, 4, dtype=torch.int64),
                torch.tensor([0.5, 0.5]),
            )
        with self.assertRaisesRegex(TypeError, "displacement"):
            advect(
                torch.ones(4, 4),
                torch.tensor([0, 1]),
            )

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            nowcast(torch.zeros(2, 32, 32), self.config)

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NowcastConfig(epsilon=-1.0)
        for field_name in (
            "support_presence_threshold",
            "contract_absolute_tolerance",
            "ratio_regularizer",
        ):
            for value in (0.0, -1.0, float("nan")):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(ValueError):
                        NowcastConfig(**{field_name: value})
        with self.assertRaises(ValueError):
            NowcastConfig(echo_threshold_dbz=float("nan"))
        with self.assertRaises(TypeError):
            NowcastConfig(interval_minutes=True)
        with self.assertRaises(ValueError):
            NowcastConfig(pair_echo_dilation_px=-1)
        with self.assertRaises(TypeError):
            NowcastConfig(pair_echo_dilation_px=True)
        with self.assertRaises(ValueError):
            NowcastConfig(min_publish_support=0.0)
        with self.assertRaises(ValueError):
            NowcastConfig(min_publish_support=1.01)
        for value in (0.0, -1.0, 1.01, float("nan"), True):
            with self.subTest(minimum_publish_verified_support=value):
                with self.assertRaises(ValueError):
                    NowcastConfig(minimum_publish_verified_support=value)
        with self.assertRaises(ValueError):
            NowcastConfig(maximum_background_age_minutes=0.0)
        with self.assertRaises(ValueError):
            NowcastConfig(minimum_phase_correlation_psr=-1.0)
        with self.assertRaises(ValueError):
            NowcastConfig(maximum_pair_motion_disagreement_px=0.0)
        with self.assertRaises(ValueError):
            NowcastConfig(maximum_pair_velocity_disagreement_mps=0.0)
        with self.assertRaises(ValueError):
            NowcastConfig(maximum_pair_growth_disagreement=0.0)
        with self.assertRaises(ValueError):
            NowcastConfig(minimum_growth_overlap_support=0.0)
        for value in (0.0, -1.0, float("nan"), True):
            with self.subTest(minimum_growth_overlap_area_km2=value):
                with self.assertRaises(ValueError):
                    NowcastConfig(minimum_growth_overlap_area_km2=value)
        for value in (0.0, -1.0):
            with self.subTest(minimum_pair_psr_advantage=value):
                with self.assertRaises(ValueError):
                    NowcastConfig(minimum_pair_psr_advantage=value)
        for value in (1.0, 0.0, -1.0):
            with self.subTest(minimum_pair_confidence_ratio=value):
                with self.assertRaises(ValueError):
                    NowcastConfig(minimum_pair_confidence_ratio=value)
        for value in (0.0, -1.0, 1.01):
            with self.subTest(long_pair_confidence_penalty=value):
                with self.assertRaises(ValueError):
                    NowcastConfig(long_pair_confidence_penalty=value)
        with self.assertRaises(ValueError):
            NowcastConfig(phase_correlation_sidelobe_radius_px=-1)
        with self.assertRaises(TypeError):
            NowcastConfig(phase_correlation_sidelobe_radius_px=True)
        for field_name in (
            "pair_echo_dilation_m",
            "phase_correlation_sidelobe_radius_m",
        ):
            for value in (-1.0, float("nan"), True):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, "nonnegative"):
                        NowcastConfig(**{field_name: value})
        for value in (0.0, -1.0, float("nan"), True):
            with self.subTest(maximum_motion_speed_mps=value):
                with self.assertRaises(ValueError):
                    NowcastConfig(maximum_motion_speed_mps=value)

    def test_physical_pair_settings_require_grid_contract(self) -> None:
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        config = NowcastConfig(
            pair_echo_dilation_m=1000.0,
            phase_correlation_sidelobe_radius_m=1000.0,
            minimum_growth_overlap_area_km2=1.0,
        )

        with self.assertRaisesRegex(ValueError, "grid/time contract"):
            estimate_state_with_metadata(frames, config)

    def test_physical_pair_echo_neighborhood_uses_exact_distance(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="9" * 64,
        )
        config = NowcastConfig(pair_echo_dilation_m=1000.0)
        previous = torch.full((5, 5), -10.0, dtype=torch.float64)
        current = previous.clone()
        previous[2, 2] = 20.0
        current[2, 2] = 20.0
        common = torch.ones((5, 5), dtype=torch.bool)
        common[1, 1] = False

        self.assertTrue(
            nowcast_module._has_complete_echo_neighborhood(
                previous,
                current,
                common,
                config,
                contract,
            )
        )
        common[2, 1] = False
        self.assertFalse(
            nowcast_module._has_complete_echo_neighborhood(
                previous,
                current,
                common,
                config,
                contract,
            )
        )

    def test_dilation_ignores_offsets_outside_small_grid(self) -> None:
        nowcast_module = import_module("advar.nowcast")
        mask = torch.tensor(
            [[True, False], [False, False]],
            dtype=torch.bool,
        )

        dilated = nowcast_module._dilate_mask(
            mask,
            ((0, 0), (0, 1), (3, 0), (0, -3), (-4, 4)),
        )

        self.assertTrue(
            torch.equal(
                dilated,
                torch.tensor(
                    [[True, True], [False, False]],
                    dtype=torch.bool,
                ),
            )
        )

    def test_grid_time_contract_is_canonical_and_part_of_run_identity(
        self,
    ) -> None:
        frames = torch.full((3, 4, 4), 20.0)
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T09:00:00+09:00",
                "2026-07-31T09:10:00+09:00",
                "2026-07-31T09:20:00+09:00",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="a" * 64,
        )
        shifted = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
                "2026-07-31T00:30:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="a" * 64,
        )

        result = nowcast(frames, grid_time_contract=contract)
        shifted_result = nowcast(frames, grid_time_contract=shifted)

        self.assertEqual(
            contract.valid_times,
            (
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
        )
        self.assertEqual(result.run.grid_time_contract, contract)
        self.assertEqual(result.run.grid_time_contract_digest, contract.digest)
        torch.testing.assert_close(
            result.displacement_mps_yx,
            result.state.displacement_yx
            * result.state.displacement_yx.new_tensor((1000.0, 1000.0))
            / 600.0,
        )
        torch.testing.assert_close(
            result.projected_velocity_mps_xy,
            torch.stack(
                (
                    result.state.displacement_yx[1] * 1000.0 / 600.0,
                    -result.state.displacement_yx[0] * 1000.0 / 600.0,
                )
            ),
        )
        self.assertNotEqual(
            result.forecast_run_digest,
            shifted_result.forecast_run_digest,
        )

    def test_grid_affine_maps_row_col_to_projected_velocity(self) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=500.0,
            projection="EPSG:5179",
            grid_hash="b" * 64,
            pixel_to_projected_matrix_m=(
                (0.0, -500.0),
                (1000.0, 0.0),
            ),
        )
        displacement = torch.tensor((2.0, 3.0), dtype=torch.float64)

        torch.testing.assert_close(
            contract.projected_displacement_xy(displacement),
            torch.tensor((-1000.0, 3000.0), dtype=torch.float64),
        )
        torch.testing.assert_close(
            contract.displacement_yx_from_projected_xy(
                torch.tensor((-1000.0, 3000.0), dtype=torch.float64)
            ),
            displacement,
        )
        velocity = contract.projected_velocity_xy(displacement, 10)
        torch.testing.assert_close(
            contract.displacement_yx_from_projected_velocity(velocity, 10),
            displacement,
        )
        self.assertEqual(
            contract.maximum_displacement_yx(10.0, 10),
            (12.0, 6.0),
        )
        self.assertEqual(contract.pixel_radius_yx(1000.0), (2, 1))

        sheared = RadarGridTimeContract(
            valid_times=contract.valid_times,
            dx_m=1000.0,
            dy_m=1000.0,
            projection=contract.projection,
            grid_hash="c" * 64,
            pixel_to_projected_matrix_m=(
                (1000.0, 800.0),
                (0.0, 600.0),
            ),
        )
        self.assertEqual(sheared.pixel_radius_yx(1000.0), (1, 1))
        sheared_offsets = sheared.pixel_offsets_within_distance(
            1000.0,
            maximum_radius_yx=(7, 7),
        )
        self.assertIn((1, -1), sheared_offsets)
        self.assertNotIn((1, 1), sheared_offsets)

    def test_physical_footprint_uses_exact_projected_distance(self) -> None:
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

        self.assertEqual(
            contract.pixel_offsets_within_distance(
                1.0,
                maximum_radius_yx=(7, 7),
            ),
            ((0, 0),),
        )
        one_kilometre = contract.pixel_offsets_within_distance(
            1000.0,
            maximum_radius_yx=(7, 7),
        )
        self.assertIn((0, 1), one_kilometre)
        self.assertIn((1, 0), one_kilometre)
        self.assertNotIn((1, 1), one_kilometre)

    def test_physical_footprint_rejects_radius_larger_than_grid(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "larger than the analysis grid"):
            contract.pixel_offsets_within_distance(
                4000.0,
                maximum_radius_yx=(2, 2),
            )

    def test_grid_affine_rejects_singular_or_inconsistent_geometry(
        self,
    ) -> None:
        common = {
            "valid_times": (
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            "dx_m": 1000.0,
            "dy_m": 1000.0,
            "projection": "EPSG:5179",
            "grid_hash": "d" * 64,
        }
        cases = (
            (
                ((1000.0, 1000.0), (0.0, 0.0)),
                "invertible",
            ),
            (
                ((500.0, 0.0), (0.0, -1000.0)),
                "agree with dx_m and dy_m",
            ),
            (
                ((1000.0, 1000.0), (0.0, 1.0e-6)),
                "well-conditioned",
            ),
        )
        for matrix, message in cases:
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, message):
                    RadarGridTimeContract(
                        **common,
                        pixel_to_projected_matrix_m=matrix,
                    )

    def test_physical_motion_limit_requires_grid_contract(self) -> None:
        frames = torch.full((3, 8, 8), 20.0)

        with self.assertRaisesRegex(ValueError, "grid/time contract"):
            nowcast(
                frames,
                NowcastConfig(maximum_motion_speed_mps=10.0),
            )

    def test_grid_time_contract_rejects_invalid_time_and_background_age(
        self,
    ) -> None:
        frames = torch.full((3, 4, 4), 20.0)
        with self.assertRaisesRegex(ValueError, "timezone"):
            RadarGridTimeContract(
                valid_times=(
                    "2026-07-31T00:00:00",
                    "2026-07-31T00:10:00Z",
                    "2026-07-31T00:20:00Z",
                ),
                dx_m=1000.0,
                dy_m=1000.0,
                projection="EPSG:5179",
                grid_hash="a" * 64,
            )
        wrong_interval = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:05:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "interval_minutes"):
            nowcast(frames, grid_time_contract=wrong_interval)

        background = frames - 1.0
        background_contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="a" * 64,
            background_valid_times=(
                "2026-07-30T23:50:00Z",
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
            ),
        )
        with self.assertRaisesRegex(ValueError, "latest valid times"):
            nowcast(
                frames,
                background_frames_dbz=background,
                background_age_minutes=20.0,
                grid_time_contract=background_contract,
            )
        with self.assertRaisesRegex(ValueError, "maximum_background_age"):
            nowcast(
                frames,
                NowcastConfig(maximum_background_age_minutes=5.0),
                background_frames_dbz=background,
                background_age_minutes=10.0,
            )


if __name__ == "__main__":
    unittest.main()
