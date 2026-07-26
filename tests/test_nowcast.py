from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.nowcast import (  # noqa: E402
    NowcastConfig,
    advect,
    dbz_to_linear,
    estimate_state,
    forecast_from_state,
    linear_to_dbz,
    nowcast,
)


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
        forecast, state = nowcast(torch.stack((dbz, dbz, dbz)), self.config)

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

    def test_nan_input_produces_finite_forecast(self) -> None:
        dbz = linear_to_dbz(self.echo, self.config)
        frames = torch.stack((dbz, dbz, dbz))
        frames[:, 0, 0] = torch.nan
        forecast, _ = nowcast(frames, self.config)
        self.assertTrue(bool(torch.all(torch.isfinite(forecast))))

    def test_empty_echo_uses_persistence_fallback(self) -> None:
        frames = torch.full((3, 32, 32), self.config.min_dbz)
        forecast, state = nowcast(frames, self.config)

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
            echo_amplitude=state.echo_amplitude,
            displacement_yx=displacement,
            log_growth_per_step=torch.zeros(()),
            pair_displacements_yx=state.pair_displacements_yx,
            pair_log_growth=state.pair_log_growth,
        )

        forecast = forecast_from_state(state, self.config)
        expected = linear_to_dbz(
            advect(state.echo_linear, 18 * displacement),
            self.config,
        )
        torch.testing.assert_close(forecast[-1], expected)

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
