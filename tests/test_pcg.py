from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.matrix_free import (  # noqa: E402
    gauss_newton_hvp,
    gauss_newton_value_and_gradient,
    jvp,
    pcg,
    vjp,
)


class PCGTests(unittest.TestCase):
    def test_solves_known_spd_system(self) -> None:
        matrix = torch.tensor(
            [[4.0, 1.0], [1.0, 3.0]],
            dtype=torch.float64,
        )
        rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)

        result = pcg(
            lambda value: matrix @ value,
            rhs,
            rtol=1.0e-12,
        )

        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations, 2)
        self.assertLess(result.relative_residual, 1.0e-12)
        torch.testing.assert_close(
            result.solution,
            torch.linalg.solve(matrix, rhs),
        )
        with self.assertRaises(FrozenInstanceError):
            setattr(result, "iterations", 10)

    def test_uses_inverse_preconditioner(self) -> None:
        diagonal = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        rhs = torch.tensor([2.0, -8.0, 18.0], dtype=torch.float64)

        result = pcg(
            lambda value: diagonal * value,
            rhs,
            preconditioner=lambda value: value / diagonal,
            rtol=1.0e-12,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        torch.testing.assert_close(result.solution, rhs / diagonal)

    def test_iteration_limit_reports_recomputed_true_residual(self) -> None:
        matrix = torch.tensor(
            [[1.0, 0.0], [0.0, 4.0]],
            dtype=torch.float64,
        )
        rhs = torch.ones(2, dtype=torch.float64)
        operator_calls = 0

        def operator(value: torch.Tensor) -> torch.Tensor:
            nonlocal operator_calls
            operator_calls += 1
            return matrix @ value

        result = pcg(
            operator,
            rhs,
            rtol=0.0,
            max_iterations=1,
        )
        true_relative_residual = float(
            torch.linalg.vector_norm(rhs - matrix @ result.solution)
            / torch.linalg.vector_norm(rhs)
        )

        self.assertFalse(result.converged)
        self.assertEqual(operator_calls, 2)
        self.assertAlmostEqual(
            result.relative_residual,
            true_relative_residual,
        )

    def test_zero_rhs_returns_exact_zero_without_operator_call(self) -> None:
        calls = 0

        def operator(value: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return value

        result = pcg(
            operator,
            torch.zeros(4, dtype=torch.float64),
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.relative_residual, 0.0)
        self.assertEqual(calls, 0)
        torch.testing.assert_close(
            result.solution,
            torch.zeros(4, dtype=torch.float64),
        )

    def test_low_precision_norm_does_not_overflow_to_false_convergence(self) -> None:
        rhs = torch.full((1000,), 60_000.0, dtype=torch.float16)

        result = pcg(lambda value: value, rhs, rtol=1.0e-3)

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.relative_residual, 0.0)
        torch.testing.assert_close(result.solution, rhs)

    def test_tiny_float64_system_does_not_underflow_inner_product(self) -> None:
        rhs = torch.tensor([1.0e-300], dtype=torch.float64)

        result = pcg(lambda value: value, rhs)

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.relative_residual, 0.0)
        torch.testing.assert_close(result.solution, rhs)

    def test_unrepresentable_solution_raises_controlled_error(self) -> None:
        rhs = torch.ones(2, dtype=torch.float64)

        with self.assertRaisesRegex(RuntimeError, "step is not finite"):
            pcg(lambda value: 1.0e-310 * value, rhs)

    @unittest.skipUnless(
        torch.backends.mps.is_available(),
        "MPS is not available",
    )
    def test_mps_solve_does_not_require_float64(self) -> None:
        rhs = torch.full((4,), 1.0e38, device="mps")

        result = pcg(lambda value: value, rhs)

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.relative_residual, 0.0)
        torch.testing.assert_close(result.solution, rhs)

    @unittest.skipUnless(
        torch.backends.mps.is_available(),
        "MPS is not available",
    )
    def test_mps_extreme_scale_matches_cpu_oracle(self) -> None:
        diagonal_cpu = torch.logspace(-12, 12, 64, dtype=torch.float32)
        rhs_cpu = torch.linspace(0.5, 1.5, 64, dtype=torch.float32)

        def solve(diagonal: torch.Tensor, rhs: torch.Tensor):
            return pcg(
                lambda value: diagonal * value,
                rhs,
                preconditioner=lambda value: value / diagonal,
                rtol=1.0e-5,
            )

        cpu = solve(diagonal_cpu, rhs_cpu)
        mps = solve(diagonal_cpu.to("mps"), rhs_cpu.to("mps"))

        self.assertEqual(cpu.converged, mps.converged)
        self.assertEqual(cpu.iterations, mps.iterations)
        torch.testing.assert_close(
            mps.solution.cpu(),
            cpu.solution,
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    def test_rejects_invalid_operator_shape_and_configuration(self) -> None:
        rhs = torch.ones(3, dtype=torch.float64)

        with self.assertRaisesRegex(ValueError, "max_iterations"):
            pcg(lambda value: value, rhs, max_iterations=0)
        with self.assertRaisesRegex(ValueError, "shape"):
            pcg(lambda value: value[:-1], rhs)
        with self.assertRaisesRegex(RuntimeError, "positive definite"):
            pcg(lambda value: -value, rhs)


class GaussNewtonTests(unittest.TestCase):
    def test_value_gradient_and_hvp_match_linear_algebra(self) -> None:
        matrix = torch.tensor(
            [
                [1.0, 2.0, -1.0],
                [0.5, -1.0, 3.0],
                [2.0, 0.0, 1.0],
                [-1.0, 1.0, 0.5],
            ],
            dtype=torch.float64,
        )
        target = torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=torch.float64)
        point = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64)
        direction = torch.tensor([-0.3, 0.7, 0.1], dtype=torch.float64)
        residual_fn = lambda value: matrix @ value - target

        value, gradient = gauss_newton_value_and_gradient(
            residual_fn,
            point,
        )
        product = gauss_newton_hvp(
            residual_fn,
            point,
            direction,
            damping=0.2,
        )
        residual = residual_fn(point)

        torch.testing.assert_close(value, 0.5 * torch.dot(residual, residual))
        torch.testing.assert_close(gradient, matrix.T @ residual)
        torch.testing.assert_close(
            product,
            matrix.T @ (matrix @ direction) + 0.2 * direction,
        )

    def test_gauss_newton_operator_is_symmetric_psd(self) -> None:
        torch.manual_seed(23)
        point = torch.randn(5, dtype=torch.float64)
        left = torch.randn_like(point)
        right = torch.randn_like(point)

        def residual_fn(value: torch.Tensor) -> torch.Tensor:
            return torch.cat((torch.sin(value), value[:3] ** 2))

        h_left = gauss_newton_hvp(residual_fn, point, left)
        h_right = gauss_newton_hvp(residual_fn, point, right)

        torch.testing.assert_close(
            torch.dot(left, h_right),
            torch.dot(h_left, right),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        self.assertGreaterEqual(float(torch.dot(left, h_left)), -1.0e-12)

    def test_gauss_newton_hvp_supports_vmap(self) -> None:
        point = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64)
        directions = torch.eye(3, dtype=torch.float64)
        residual_fn = lambda value: torch.sin(value) + value**2

        products = torch.func.vmap(
            lambda direction: gauss_newton_hvp(
                residual_fn,
                point,
                direction,
            )
        )(directions)
        expected = torch.stack(
            [
                gauss_newton_hvp(residual_fn, point, direction)
                for direction in directions
            ]
        )

        torch.testing.assert_close(products, expected)

    def test_jvp_vjp_adjoint_identity_for_residual(self) -> None:
        torch.manual_seed(29)
        matrix = torch.randn(7, 4, dtype=torch.float64)
        point = torch.randn(4, dtype=torch.float64)
        direction = torch.randn_like(point)
        cotangent = torch.randn(7, dtype=torch.float64)
        residual_fn = lambda value: torch.tanh(matrix @ value)

        _, jacobian_vector = jvp(residual_fn, point, direction)
        _, vector_jacobian = vjp(residual_fn, point, cotangent)

        torch.testing.assert_close(
            torch.dot(jacobian_vector, cotangent),
            torch.dot(direction, vector_jacobian),
            atol=1.0e-12,
            rtol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
