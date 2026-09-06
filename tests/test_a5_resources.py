"""Independent block-whitening oracle on rectangular and skinny grids."""

import unittest
from unittest.mock import patch

import torch

from advar import variational


def _dense_blocks(values, mode, tile_size, bias_std, scope):
    output = torch.empty_like(values)
    frames, height, width = values.shape
    times = (
        [(index,) for index in range(frames)]
        if scope == "per_frame"
        else [tuple(range(frames))]
    )
    for time in times:
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                block = (list(time), slice(y, y + tile_size), slice(x, x + tile_size))
                vector = values[block].flatten()
                direction = mode[block].flatten()
                covariance = torch.eye(vector.numel(), dtype=values.dtype)
                covariance += bias_std**2 * torch.outer(direction, direction)
                eigenvalues, vectors = torch.linalg.eigh(covariance)
                inverse_root = (vectors * eigenvalues.rsqrt()) @ vectors.mT
                output[block] = (inverse_root @ vector).reshape_as(values[block])
    return output


class A5ResourceTests(unittest.TestCase):
    def test_skinny_grid_whitening_keeps_blocks_and_bounded_padding(self):
        generator = torch.Generator().manual_seed(34)
        for shape, tile_size in (
            ((3, 2, 101), 100),
            ((3, 101, 2), 100),
            ((3, 2, 3), 10**9),
        ):
            values = torch.randn(shape, generator=generator, dtype=torch.float64)
            mode = torch.rand(shape, generator=generator, dtype=torch.float64)
            for scope in ("per_frame", "all_times"):
                with self.subTest(shape=shape, tile_size=tile_size, scope=scope):
                    original_pad = variational.F.pad
                    padded_sizes = []

                    def bounded_pad(tensor, padding):
                        left, right, top, bottom = padding
                        padded_size = (
                            tensor.shape[0]
                            * (tensor.shape[1] + top + bottom)
                            * (tensor.shape[2] + left + right)
                        )
                        self.assertLessEqual(padded_size, 4 * tensor.numel())
                        padded_sizes.append(padded_size)
                        return original_pad(tensor, padding)

                    def apply(value):
                        return variational._apply_tiled_observation_error_whitener(
                            value,
                            mode,
                            bias_std=0.7,
                            tile_size=tile_size,
                            temporal_scope=scope,
                        )

                    with patch.object(variational.F, "pad", side_effect=bounded_pad):
                        result = apply(values)
                    self.assertEqual(len(padded_sizes), 2)
                    expected = _dense_blocks(values, mode, tile_size, 0.7, scope)
                    torch.testing.assert_close(result, expected, rtol=1e-11, atol=1e-11)
                    direction = torch.randn(
                        shape, generator=generator, dtype=torch.float64
                    )
                    _, tangent = torch.func.jvp(apply, (values,), (direction,))
                    expected_tangent = _dense_blocks(
                        direction, mode, tile_size, 0.7, scope
                    )
                    torch.testing.assert_close(
                        tangent, expected_tangent, rtol=1e-11, atol=1e-11
                    )


if __name__ == "__main__":
    unittest.main()
