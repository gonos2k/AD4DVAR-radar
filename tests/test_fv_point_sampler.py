import pytest
import torch

from advar.fv_point_sampler import point_dbz_bilinear


def _coords(values: list[tuple[float, float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_bilinear_sampler_preserves_constants_and_reproduces_affine_fields() -> None:
    rows = torch.arange(6, dtype=torch.float64)[:, None]
    columns = torch.arange(7, dtype=torch.float64)[None, :]
    constant = torch.full((2, 6, 7), 12.5, dtype=torch.float64)
    affine = 2 * rows - 3 * columns + 4
    field = torch.stack((affine.expand(6, 7), (affine + 8).expand(6, 7)))
    points = _coords([(1.25, 2.5), (3.75, 4.125)])

    sampled_constant = point_dbz_bilinear(constant, points)
    sampled_affine = point_dbz_bilinear(field, points)
    expected_affine = torch.stack(
        (
            2 * points[:, 0] - 3 * points[:, 1] + 4,
            2 * points[:, 0] - 3 * points[:, 1] + 12,
        )
    )
    torch.testing.assert_close(sampled_constant, torch.full((2, 2), 12.5, dtype=torch.float64))
    torch.testing.assert_close(sampled_affine, expected_affine, atol=1e-14, rtol=1e-14)


def test_bilinear_sampler_autograd_is_the_weighted_adjoint() -> None:
    field = torch.arange(30, dtype=torch.float64).reshape(5, 6).requires_grad_()
    points = _coords([(1.25, 2.5), (2.75, 1.125), (1.5, 4.25)])
    dual = torch.tensor([0.5, -2.0, 1.25], dtype=torch.float64)

    (point_dbz_bilinear(field, points) * dual).sum().backward()

    expected = torch.zeros_like(field)
    for (row, column), coefficient in zip(points.tolist(), dual.tolist()):
        row0, column0 = int(row), int(column)
        dr, dc = row - row0, column - column0
        for r, c, weight in (
            (row0, column0, (1 - dr) * (1 - dc)),
            (row0, column0 + 1, (1 - dr) * dc),
            (row0 + 1, column0, dr * (1 - dc)),
            (row0 + 1, column0 + 1, dr * dc),
        ):
            expected[r, c] += coefficient * weight

    assert field.grad is not None
    torch.testing.assert_close(field.grad, expected, atol=1e-14, rtol=1e-14)
    assert bool(torch.isfinite(field.grad).all())


def test_bilinear_sampler_uses_nonnegative_unit_sum_weights() -> None:
    point = _coords([(1.25, 2.75)])
    weights = []
    for row, column in ((1, 2), (1, 3), (2, 2), (2, 3)):
        basis = torch.zeros((4, 5), dtype=torch.float64)
        basis[row, column] = 1.0
        weights.append(point_dbz_bilinear(basis, point)[0])
    weight_vector = torch.stack(weights)
    assert bool((weight_vector >= 0).all())
    torch.testing.assert_close(weight_vector.sum(), torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "point",
    [(-0.1, 1.5), (0.0, 1.5), (1.5, 0.0), (4.0, 1.5), (1.5, 5.0), (4.1, 1.5)],
)
def test_bilinear_sampler_rejects_outside_and_edge_points(point: tuple[float, float]) -> None:
    field = torch.zeros((5, 6), dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly inside"):
        point_dbz_bilinear(field, _coords([point]))


def test_bilinear_sampler_averages_dbz_values_directly() -> None:
    # Equal geometric weights average 0 and 20 dBZ to 10 dBZ. Averaging
    # linear reflectivity first would produce about 17 dBZ instead.
    field = torch.zeros((4, 4), dtype=torch.float64)
    field[:, 2:] = 20.0
    point = _coords([(1.5, 1.5)])
    sampled = point_dbz_bilinear(field, point)
    assert sampled.item() == pytest.approx(10.0)


def test_bilinear_sampler_validates_shape_dtype_and_finiteness() -> None:
    field = torch.zeros((5, 6), dtype=torch.float64)
    valid_points = _coords([(1.5, 2.5)])
    with pytest.raises(ValueError, match="shape \\[N, 2\\]"):
        point_dbz_bilinear(field, torch.zeros((2,), dtype=torch.float64))
    with pytest.raises(ValueError, match="CPU float64"):
        point_dbz_bilinear(field.float(), valid_points)
    with pytest.raises(ValueError, match="finite"):
        point_dbz_bilinear(field, _coords([(float("nan"), 2.0)]))
    with pytest.raises(ValueError, match="finite"):
        point_dbz_bilinear(field + float("inf"), valid_points)
