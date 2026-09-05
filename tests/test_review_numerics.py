"""Independent values and derivatives for the full-review numerical fixes."""

import math
import unittest

import torch

from advar.ensemble_sensitivity import PrecisionOperatorArtifact
from advar.matrix_free import gauss_newton_value_and_gradient
from advar.metrics import critical_success_index, mae, rmse
from advar.nowcast import ForecastRunContract, NowcastConfig, nowcast
from advar.promotion import (
    _standard_normal_log_interval_mass,
    _truncated_gaussian_diagnostics,
)


class NumericalFollowupTests(unittest.TestCase):
    def test_gauss_newton_representable_half_sum_and_gradient(self):
        point = torch.full((4,), 1e19, dtype=torch.float32)
        value, gradient = gauss_newton_value_and_gradient(lambda x: x, point)
        oracle = 0.5 * point.double().square().sum()
        torch.testing.assert_close(value.double(), oracle, rtol=2e-7, atol=0.0)
        torch.testing.assert_close(gradient, point)

    def test_metrics_large_values_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            limit = torch.finfo(dtype).max
            for metric in (mae, rmse):
                with self.subTest(dtype=dtype, metric=metric.__name__):
                    x = torch.tensor([limit, -limit], dtype=dtype, requires_grad=True)
                    value = metric(x, torch.zeros_like(x))
                    torch.testing.assert_close(value, x.new_tensor(limit))
                    gradient = torch.autograd.grad(value, x)[0]
                    torch.testing.assert_close(gradient, x.new_tensor([0.5, -0.5]))

    def test_metrics_representable_result_after_overflowing_raw_difference(self):
        for dtype in (torch.float32, torch.float64):
            limit = torch.finfo(dtype).max
            x = torch.tensor([limit, 0., 0., 0.], dtype=dtype, requires_grad=True)
            truth = x.new_tensor([-limit, 0., 0., 0.])
            for metric, expected, derivative in ((mae, limit / 2, 0.25), (rmse, limit, 0.5)):
                with self.subTest(dtype=dtype, metric=metric.__name__):
                    value = metric(x, truth)
                    torch.testing.assert_close(value, x.new_tensor(expected))
                    gradient = torch.autograd.grad(value, x)[0]
                    torch.testing.assert_close(gradient, x.new_tensor([derivative, 0., 0., 0.]))

    def test_metrics_preserve_cancellation_and_small_values(self):
        for dtype in (torch.float32, torch.float64):
            small = 1e-20 if dtype == torch.float32 else 1e-200
            x = torch.tensor([torch.finfo(dtype).max, small], dtype=dtype)
            truth = x.new_tensor([torch.finfo(dtype).max, 0.])
            torch.testing.assert_close(mae(x, truth), x.new_tensor(small / 2), atol=0.0, rtol=2e-6)
            torch.testing.assert_close(rmse(x, truth), x.new_tensor(small / math.sqrt(2)), atol=0.0, rtol=2e-6)

    def test_metrics_ordinary_derivatives_and_zero_subgradient(self):
        x = torch.tensor([0., -2., 3.], dtype=torch.float64, requires_grad=True)
        truth = x.new_tensor([1., 0.5, -4.])
        for metric in (mae, rmse):
            self.assertTrue(torch.autograd.gradcheck(lambda y: metric(y, truth), (x,)))
            self.assertTrue(torch.autograd.gradgradcheck(lambda y: metric(y, truth), (x,)))
            zero = torch.zeros_like(x, requires_grad=True)
            value = metric(zero, torch.zeros_like(zero))
            torch.testing.assert_close(value, zero.new_tensor(0.))
            torch.testing.assert_close(torch.autograd.grad(value, zero)[0], torch.zeros_like(zero))

    def test_narrow_normal_interval_values_and_endpoint_derivatives(self):
        # Reference values: 100-digit integration of exp(-x*x/2)/sqrt(2*pi)
        # between the exact binary64 endpoints (erfc differences in the tails).
        for lower, upper, expected in (
            (-1e-16, 1e-16, -37.06715284054946),
            (40., 40.000001, -814.7344690936271),
            (-40.000001, -40., -814.7344690936271),
        ):
            bounds = torch.tensor([lower, upper], dtype=torch.float64, requires_grad=True)
            value = _standard_normal_log_interval_mass(bounds[:1], bounds[1:]).sum()
            self.assertAlmostEqual(float(value.detach()), expected, places=11)
            gradient = torch.autograd.grad(value, bounds)[0]
            log_density = -0.5 * bounds.detach().square() - 0.5 * math.log(2 * math.pi)
            oracle = torch.exp(log_density - expected) * bounds.new_tensor([-1., 1.])
            torch.testing.assert_close(gradient, oracle, rtol=2e-11, atol=0.0)

    def test_narrow_normal_parameter_derivatives(self):
        point = torch.tensor([0.1, math.log(1e-3)], dtype=torch.float64, requires_grad=True)
        def interval(parameters):
            center, log_width = parameters.unbind()
            half_width = 0.5 * log_width.exp()
            return _standard_normal_log_interval_mass(center - half_width, center + half_width)
        self.assertTrue(torch.autograd.gradcheck(interval, (point,), eps=1e-6))
        self.assertTrue(torch.autograd.gradgradcheck(interval, (point,), eps=1e-6))
        # Endpoint perturbations must be smaller than the tiny interval itself.
        bounds = torch.tensor([-5e-17, 5e-17], dtype=torch.float64, requires_grad=True)
        tiny_interval = lambda edges: _standard_normal_log_interval_mass(edges[:1], edges[1:])
        self.assertTrue(torch.autograd.gradcheck(tiny_interval, (bounds,), eps=1e-22))
        # The same finite-difference step must also resolve the cotangent input.
        cotangent = bounds.new_tensor([1e-16], requires_grad=True)
        self.assertTrue(torch.autograd.gradgradcheck(
            tiny_interval, (bounds,), grad_outputs=(cotangent,), eps=1e-22
        ))

    def test_truncated_gaussian_large_scale_has_finite_likelihood(self):
        scale = torch.tensor([1e16], dtype=torch.float64, requires_grad=True)
        nll, pit = _truncated_gaussian_diagnostics(
            torch.zeros_like(scale), scale, torch.full_like(scale, 5.),
            support_threshold_dbz=0.,
        )
        # Near zero the density is 1/sqrt(2*pi); truncation retains half its mass.
        expected = math.log(1e16) + 0.5 * math.log(2 * math.pi)
        torch.testing.assert_close(nll, scale.new_tensor([expected]), rtol=0.0, atol=2e-13)
        self.assertTrue(bool(torch.isfinite(pit).all()))
        torch.testing.assert_close(torch.autograd.grad(nll.sum(), scale)[0], scale.new_tensor([1e-16]), rtol=1e-12, atol=0.)

    def test_invalid_csi_threshold_is_rejected(self):
        for threshold in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ValueError, "threshold_dbz must be finite"):
                critical_success_index(torch.tensor([40.]), torch.tensor([0.]), threshold)

    def test_unsupported_precision_is_rejected_at_boundary(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.assertRaisesRegex(TypeError, "float32 or float64"):
                nowcast(torch.zeros((3, 2, 2), dtype=dtype))
            with self.assertRaisesRegex(TypeError, "float32 or float64"):
                PrecisionOperatorArtifact(
                    precision=torch.eye(2, dtype=dtype), covariance=torch.eye(2, dtype=dtype),
                    observation_ids=("a", "b"), forecast_run_digest="1"*64,
                    observation_error_model_digest="2"*64, calibration_manifest_digest="3"*64,
                )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_companion_devices_are_checked_before_tensor_operations(self):
        frames = torch.zeros((3, 2, 2))
        mask = torch.ones_like(frames, dtype=torch.bool)
        for kwargs in (
            {"observation_quality_weight": torch.ones_like(frames, device="mps")},
            {"observation_std_dbz": torch.ones_like(frames, device="mps")},
        ):
            with self.assertRaisesRegex(ValueError, "match the frames"):
                ForecastRunContract.from_inputs(NowcastConfig(), frames, mask, None, **kwargs)
        with self.assertRaisesRegex(ValueError, "observation_masks"):
            ForecastRunContract.from_inputs(NowcastConfig(), frames, mask.to("mps"), None)


if __name__ == "__main__":
    unittest.main()
