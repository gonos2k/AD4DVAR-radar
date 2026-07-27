from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.matrix_free import hvp, jvp, vjp  # noqa: E402
from advar.nowcast import (  # noqa: E402
    NowcastConfig,
    RadarState,
    advect,
    freeze_remap_cell,
    forecast_from_state,
)


class MatrixFreeTests(unittest.TestCase):
    def test_jvp_vjp_dot_product_identity(self) -> None:
        torch.manual_seed(7)
        matrix = torch.randn(9, 5, dtype=torch.float64)
        point = torch.randn(5, dtype=torch.float64)
        direction = torch.randn(5, dtype=torch.float64)
        cotangent = torch.randn(9, dtype=torch.float64)

        function = lambda value: torch.sin(matrix @ value)
        _, jacobian_vector = jvp(function, point, direction)
        _, vector_jacobian = vjp(function, point, cotangent)

        left = torch.dot(jacobian_vector, cotangent)
        right = torch.dot(direction, vector_jacobian)
        torch.testing.assert_close(left, right)

    def test_hvp_is_symmetric(self) -> None:
        torch.manual_seed(11)
        point = torch.randn(6, dtype=torch.float64)
        v = torch.randn(6, dtype=torch.float64)
        w = torch.randn(6, dtype=torch.float64)
        loss = lambda value: torch.sum(torch.exp(value) + value**4)

        hv = hvp(loss, point, v)
        hw = hvp(loss, point, w)
        torch.testing.assert_close(torch.dot(hv, w), torch.dot(v, hw))

    def test_forecast_operator_jvp_vjp_identity(self) -> None:
        torch.manual_seed(13)
        config = NowcastConfig(horizon_minutes=20)
        y, x = torch.meshgrid(
            torch.arange(12, dtype=torch.float64),
            torch.arange(12, dtype=torch.float64),
            indexing="ij",
        )
        echo = 1.0e3 * torch.exp(-((y - 6) ** 2 + (x - 6) ** 2) / 8.0)
        pair_motion = torch.zeros(2, 2, dtype=torch.float64)
        pair_growth = torch.zeros(2, dtype=torch.float64)

        def forecast(control: torch.Tensor) -> torch.Tensor:
            state = RadarState(
                echo_amplitude=torch.sqrt(echo),
                displacement_yx=control[:2],
                log_growth_per_step=control[2],
                pair_displacements_yx=pair_motion,
                pair_log_growth=pair_growth,
            )
            return forecast_from_state(state, config)

        point = torch.tensor([0.3, -0.2, 0.01], dtype=torch.float64)
        direction = torch.randn_like(point)
        cotangent = torch.randn_like(forecast(point))
        _, jacobian_vector = jvp(forecast, point, direction)
        _, vector_jacobian = vjp(forecast, point, cotangent)

        left = torch.sum(jacobian_vector * cotangent)
        right = torch.dot(direction, vector_jacobian)
        torch.testing.assert_close(left, right, atol=1.0e-8, rtol=1.0e-8)

    def test_forecast_metric_hvp_is_symmetric(self) -> None:
        config = NowcastConfig(horizon_minutes=20)
        y, x = torch.meshgrid(
            torch.arange(8, dtype=torch.float64),
            torch.arange(8, dtype=torch.float64),
            indexing="ij",
        )
        echo = 1.0e3 * torch.exp(-((y - 4) ** 2 + (x - 4) ** 2) / 6.0)
        pair_motion = torch.zeros(2, 2, dtype=torch.float64)
        pair_growth = torch.zeros(2, dtype=torch.float64)

        def loss(control: torch.Tensor) -> torch.Tensor:
            state = RadarState(
                echo_amplitude=torch.sqrt(echo),
                displacement_yx=control[:2],
                log_growth_per_step=control[2],
                pair_displacements_yx=pair_motion,
                pair_log_growth=pair_growth,
            )
            forecast = forecast_from_state(state, config)
            return torch.mean(forecast**2)

        point = torch.tensor([0.3, -0.2, 0.01], dtype=torch.float64)
        v = torch.tensor([0.2, -0.1, 0.03], dtype=torch.float64)
        w = torch.tensor([-0.1, 0.4, -0.02], dtype=torch.float64)
        hv = hvp(loss, point, v)
        hw = hvp(loss, point, w)
        torch.testing.assert_close(
            torch.dot(hv, w),
            torch.dot(v, hw),
            atol=1.0e-8,
            rtol=1.0e-8,
        )

    def test_advection_jvp_matches_finite_difference_in_frozen_cell(self) -> None:
        field = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)

        for point in (
            torch.tensor([0.25, 0.35], dtype=torch.float64),
            torch.tensor([1.25, -1.75], dtype=torch.float64),
        ):
            with self.subTest(point=point.tolist()):
                cell = freeze_remap_cell(point)
                function = lambda motion: advect(
                    field,
                    motion,
                    frozen_cell=cell,
                )
                _, product = jvp(function, point, direction)
                epsilon = 1.0e-5
                finite_difference = (
                    function(point + epsilon * direction)
                    - function(point - epsilon * direction)
                ) / (2.0 * epsilon)
                torch.testing.assert_close(
                    product,
                    finite_difference,
                    atol=1.0e-7,
                    rtol=1.0e-7,
                )

    def test_positive_state_gate_does_not_clamp_signed_jvp(self) -> None:
        field = torch.zeros(4, 4, dtype=torch.float64)
        field[1, 1] = 1.0
        point = torch.tensor([0.25, 0.25], dtype=torch.float64)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        cell = freeze_remap_cell(point)

        _, product = jvp(
            lambda motion: advect(field, motion, frozen_cell=cell),
            point,
            direction,
        )

        self.assertTrue(bool(torch.any(product < 0.0)))
        self.assertTrue(bool(torch.any(product > 0.0)))

    def test_advection_hvp_matches_finite_difference_in_frozen_cell(self) -> None:
        field = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        point = torch.tensor([0.25, 0.35], dtype=torch.float64)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        cell = freeze_remap_cell(point)
        loss = lambda motion: torch.mean(
            advect(field, motion, frozen_cell=cell) ** 2
        )

        product = hvp(loss, point, direction)
        gradient = torch.func.grad(loss)
        epsilon = 1.0e-5
        finite_difference = (
            gradient(point + epsilon * direction)
            - gradient(point - epsilon * direction)
        ) / (2.0 * epsilon)
        torch.testing.assert_close(
            product,
            finite_difference,
            atol=1.0e-2,
            rtol=1.0e-3,
        )

    def test_advection_is_continuous_at_an_integer_cell_boundary(self) -> None:
        field = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        point = torch.zeros(2, dtype=torch.float64)
        epsilon = 1.0e-7
        left = advect(
            field,
            torch.tensor([-epsilon, 0.0], dtype=torch.float64),
        )
        center = advect(field, point)
        right = advect(
            field,
            torch.tensor([epsilon, 0.0], dtype=torch.float64),
        )

        torch.testing.assert_close(left, center, atol=2.0e-6, rtol=0.0)
        torch.testing.assert_close(right, center, atol=2.0e-6, rtol=0.0)

    def test_forecast_jvp_matches_finite_difference_inside_remap_cell(self) -> None:
        config = NowcastConfig(horizon_minutes=10)
        y, x = torch.meshgrid(
            torch.arange(12, dtype=torch.float64),
            torch.arange(12, dtype=torch.float64),
            indexing="ij",
        )
        amplitude = torch.exp(-((y - 6) ** 2 + (x - 6) ** 2) / 16.0)
        pair_motion = torch.zeros(2, 2, dtype=torch.float64)
        pair_growth = torch.zeros(2, dtype=torch.float64)

        def function(motion: torch.Tensor) -> torch.Tensor:
            state = RadarState(
                echo_amplitude=amplitude,
                displacement_yx=motion,
                log_growth_per_step=torch.zeros((), dtype=torch.float64),
                pair_displacements_yx=pair_motion,
                pair_log_growth=pair_growth,
            )
            return forecast_from_state(state, config)

        point = torch.tensor([1.25, 0.2], dtype=torch.float64)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        _, product = jvp(function, point, direction)
        epsilon = 1.0e-5
        finite_difference = (
            function(point + epsilon * direction)
            - function(point - epsilon * direction)
        ) / (2.0 * epsilon)
        torch.testing.assert_close(
            product,
            finite_difference,
            atol=1.0e-7,
            rtol=1.0e-6,
        )

    def test_forecast_hvp_matches_finite_difference_inside_remap_cell(self) -> None:
        config = NowcastConfig(horizon_minutes=10)
        y, x = torch.meshgrid(
            torch.arange(12, dtype=torch.float64),
            torch.arange(12, dtype=torch.float64),
            indexing="ij",
        )
        amplitude = torch.exp(-((y - 6) ** 2 + (x - 6) ** 2) / 16.0)
        pair_motion = torch.zeros(2, 2, dtype=torch.float64)
        pair_growth = torch.zeros(2, dtype=torch.float64)

        def loss(control: torch.Tensor) -> torch.Tensor:
            state = RadarState(
                echo_amplitude=amplitude,
                displacement_yx=control[:2],
                log_growth_per_step=control[2],
                pair_displacements_yx=pair_motion,
                pair_log_growth=pair_growth,
            )
            return torch.mean(forecast_from_state(state, config) ** 2)

        point = torch.tensor([1.25, 0.2, 0.01], dtype=torch.float64)
        direction = torch.tensor([0.2, -0.1, 0.03], dtype=torch.float64)
        product = hvp(loss, point, direction)
        gradient = torch.func.grad(loss)
        epsilon = 1.0e-4
        finite_difference = (
            gradient(point + epsilon * direction)
            - gradient(point - epsilon * direction)
        ) / (2.0 * epsilon)
        torch.testing.assert_close(
            product,
            finite_difference,
            atol=1.0e-8,
            rtol=1.0e-8,
        )


if __name__ == "__main__":
    unittest.main()
