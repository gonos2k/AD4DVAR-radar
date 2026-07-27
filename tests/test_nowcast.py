from pathlib import Path
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
    ForecastMetadata,
    NowcastConfig,
    RadarState,
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
        background_age_minutes=None,
        source_mask=None,
        pair_displacements_yx=torch.stack(
            (state.displacement_yx, state.displacement_yx)
        ),
        pair_log_growth=torch.stack(
            (state.log_growth_per_step, state.log_growth_per_step)
        ),
    )


def forecast_from_state(
    state: RadarState,
    config: NowcastConfig,
) -> torch.Tensor:
    return forecast_result_from_state(
        state,
        observed_metadata(state),
        config,
    ).forecast_dbz


class NowcastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NowcastConfig()
        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32),
            indexing="ij",
        )
        self.echo = 1.0e5 * torch.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 40.0)

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
        forecast_error = torch.mean(torch.abs(forecast - verification))
        persistence_error = torch.mean(torch.abs(frames[2] - verification))
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
        self.assertEqual(metadata.background_age_minutes, 10.0)
        self.assertTrue(bool(torch.all(torch.isfinite(forecast))))
        torch.testing.assert_close(forecast[0], background[-1])

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
        self.assertTrue(metadata.background_used)
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
        self.assertIsNone(metadata.background_age_minutes)
        self.assertEqual(metadata.data_status, DataStatus.OBSERVED)

    def test_missing_first_frame_uses_neighboring_observation_only_as_fill(
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

    def test_missing_latest_frame_uses_mask_aware_persistence(self) -> None:
        dbz = linear_to_dbz(self.echo, self.config)
        frames = torch.stack((dbz, dbz, torch.full_like(dbz, torch.nan)))
        result = nowcast(frames, self.config)
        forecast, metadata = result.forecast_dbz, result.metadata

        self.assertEqual(
            metadata.data_status,
            DataStatus.PARTIAL,
        )
        self.assertEqual(float(metadata.coverage_by_frame[-1]), 0.0)
        self.assertTrue(bool(torch.all(torch.isfinite(forecast))))
        torch.testing.assert_close(forecast[0], dbz, atol=0.02, rtol=0.0)

    def test_empty_echo_uses_persistence_fallback(self) -> None:
        frames = torch.full((3, 32, 32), self.config.min_dbz)
        result = nowcast(frames, self.config)
        forecast, state = result.forecast_dbz, result.state

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
        torch.testing.assert_close(forecast[-1], expected)

    def test_full_forecast_remaps_each_lead_once_even_with_audit(self) -> None:
        state = RadarState(
            echo_linear=self.echo,
            displacement_yx=torch.tensor([0.2, -0.3]),
            log_growth_per_step=torch.zeros(()),
        )
        with patch("advar.nowcast.remap_core", wraps=remap_core) as kernel:
            result = forecast_result_from_state(
                state,
                observed_metadata(state),
                self.config,
                audit=True,
            )

        self.assertEqual(kernel.call_count, self.config.forecast_steps)
        self.assertEqual(len(result.audit.transport), self.config.forecast_steps)

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
        with self.assertRaises(ValueError):
            NowcastConfig(echo_threshold_dbz=float("nan"))
        with self.assertRaises(TypeError):
            NowcastConfig(interval_minutes=True)


if __name__ == "__main__":
    unittest.main()
