import itertools
import unittest

import torch
import torch.nn.functional as F

from advar.physics import RemapCell, remap_core, shift_zero


def padded_reference(echo: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    height, width = echo.shape[-2:]
    padded = F.pad(echo, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)))
    y, x = max(-dy, 0), max(-dx, 0)
    return padded[..., y:y + height, x:x + width]


class ShiftZeroTests(unittest.TestCase):
    def test_values_jvp_and_vjp_match_original_for_81_shifts(self) -> None:
        generator = torch.Generator().manual_seed(812)
        echo = torch.randn(2, 5, 4, dtype=torch.float64, generator=generator).transpose(-1, -2)
        direction = torch.randn(echo.shape, dtype=echo.dtype, generator=generator)
        cotangent = torch.randn(echo.shape, dtype=echo.dtype, generator=generator)
        for dy, dx in itertools.product((-6, -5, -4, -1, 0, 1, 4, 5, 6), repeat=2):
            with self.subTest(dy=dy, dx=dx):
                original = lambda value: padded_reference(value, dy, dx)
                bounded = lambda value: shift_zero(value, dy, dx)
                expected, expected_jvp = torch.func.jvp(original, (echo,), (direction,))
                actual, actual_jvp = torch.func.jvp(bounded, (echo,), (direction,))
                _, expected_vjp = torch.func.vjp(original, echo)
                _, actual_vjp = torch.func.vjp(bounded, echo)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_jvp, expected_jvp, rtol=0, atol=0)
                torch.testing.assert_close(actual_vjp(cotangent)[0], expected_vjp(cotangent)[0], rtol=0, atol=0)

    def test_storage_is_bounded_by_output_size(self) -> None:
        echo = torch.ones(4, 4)
        for dy, dx in ((1000, 1000), (-1000, 0), (0, 1000), (1, -2), (0, 0)):
            with self.subTest(dy=dy, dx=dx):
                output = shift_zero(echo, dy, dx)
                self.assertEqual(output.shape, echo.shape)
                self.assertEqual(output.untyped_storage().nbytes(), 64)
                if abs(dy) >= 4 or abs(dx) >= 4:
                    self.assertEqual(int(torch.count_nonzero(output)), 0)

    def test_out_of_domain_shift_preserves_backward_connection(self) -> None:
        echo = torch.ones(4, 4, requires_grad=True)
        shift_zero(echo, 1000, -1000).sum().backward()
        self.assertIsNotNone(echo.grad)
        torch.testing.assert_close(echo.grad, torch.zeros_like(echo))

    def test_out_of_domain_discards_nonfinite_values_without_mutating_input(self) -> None:
        echo = torch.tensor([[torch.nan, torch.inf], [-torch.inf, 1.0]])
        original = echo.clone()
        output = shift_zero(echo, 2, 0)
        torch.testing.assert_close(output, torch.zeros_like(echo))
        torch.testing.assert_close(echo, original, equal_nan=True)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_domain_exit_preserves_shift_and_remap_derivatives(self) -> None:
        echo = torch.arange(40, device="mps", dtype=torch.float32).reshape(2, 5, 4)
        echo = echo.transpose(-1, -2).requires_grad_()
        for dy, dx in ((4, 0), (0, -5), (1000, 1000)):
            with self.subTest(dy=dy, dx=dx):
                function = lambda value: shift_zero(value, dy, dx)
                output, tangent = torch.func.jvp(function, (echo,), (torch.ones_like(echo),))
                _, pullback = torch.func.vjp(function, echo)
                torch.testing.assert_close(output, torch.zeros_like(echo))
                torch.testing.assert_close(tangent, torch.zeros_like(echo))
                torch.testing.assert_close(pullback(torch.ones_like(output))[0], torch.zeros_like(echo))
                gradient, = torch.autograd.grad(function(echo).sum(), echo)
                torch.testing.assert_close(gradient, torch.zeros_like(echo))

        for motion, cell in (((4.25, 0.5), RemapCell(4, 0)), ((3.25, -0.5), RemapCell(3, -1))):
            with self.subTest(motion=motion):
                displacement = echo.new_tensor(motion)
                function = lambda field, motion: remap_core(field, motion, cell)
                output, tangent = torch.func.jvp(
                    function, (echo, displacement), (torch.ones_like(echo), torch.ones_like(displacement)),
                )
                _, pullback = torch.func.vjp(function, echo, displacement)
                gradients = pullback(torch.ones_like(output))
                cpu_echo, cpu_displacement = echo.detach().cpu(), displacement.cpu()
                expected, expected_tangent = torch.func.jvp(
                    function, (cpu_echo, cpu_displacement),
                    (torch.ones_like(cpu_echo), torch.ones_like(cpu_displacement)),
                )
                _, cpu_pullback = torch.func.vjp(function, cpu_echo, cpu_displacement)
                expected_gradients = cpu_pullback(torch.ones_like(expected))
                torch.testing.assert_close(output.cpu(), expected)
                torch.testing.assert_close(tangent.cpu(), expected_tangent)
                for actual, expected_gradient in zip(gradients, expected_gradients):
                    torch.testing.assert_close(actual.cpu(), expected_gradient)


if __name__ == "__main__":
    unittest.main()
