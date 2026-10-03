"""Pure interval-jet and paired-projection checks; no FV experiment is run."""
from typing import Any, cast

import pytest

from examples.weather_scenarios import fv_active_face_normal_reference as base
from examples.weather_scenarios import fv_point_paired_face_normal_reference as paired


def _fixture_basis() -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    wx = paired._W_X
    wy = paired._W_Y
    basis = []
    for k in range(5):
        mode = [[float((k + 1) * (r + 2) + (c + 1) * (k + 2)) for c in range(6)]
                for r in range(5)]
        mode[4][4], mode[3][4] = wx[k], 0.0
        mode[3][1], mode[3][0] = -wy[k], 0.0
        basis.append(mode)
    first = {}
    for k in (1, 2, 3, 4):
        ratio = wx[k] / wx[0]
        first[k] = [[basis[k][r][c] - ratio * basis[0][r][c] for c in range(6)]
                    for r in range(5)]
    projected = []
    for k in (2, 3, 4):
        qy_pivot = -(first[1][3][1] - first[1][3][0])
        qy_weight = -(first[k][3][1] - first[k][3][0])
        ratio = qy_weight / qy_pivot
        projected.append([[first[k][r][c] - ratio * first[1][r][c] for c in range(6)]
                          for r in range(5)])
    return basis, projected


def test_paired_projection_matches_rational_flux_rows_and_both_exact_zeros():
    basis, projected = _fixture_basis()
    fixture: dict[str, Any] = {
        "psi_basis": basis,
        "projected_psi_basis": projected,
        "normal_weights": [list(paired._W_X), list(paired._W_Y)],
        "coefficient_limits": [0.11, 0.08, 0.7, 0.6, 0.5],
        "normal_scales": [0.11, 0.08],
    }
    wx, wy = paired.validate_projection(fixture)
    assert wx == list(paired._W_X)
    assert wy == list(paired._W_Y)
    assert all(mode[4][4] == mode[3][4] for mode in projected)
    assert all(mode[3][0] == mode[3][1] for mode in projected)

    broken = {**fixture, "projected_psi_basis": [[row[:] for row in mode]
                                                  for mode in projected]}
    broken["projected_psi_basis"][0][0][0] += 0.125
    with pytest.raises(ValueError, match="rational formula"):
        paired.validate_projection(broken)


@pytest.mark.parametrize("sign", [-1, 1])
def test_sector_tag_selects_positive_and_negative_parts_even_with_zero_derivative(sign):
    from mpmath import iv

    context = cast(Any, iv)
    iv.dps = 50
    mp = paired._SectorJetMath(context, 1)
    event = paired.SectorEventJet(context.zero, context.zero, 1, True, sign)
    positive = max(event, mp.zero)
    negative = min(event, mp.zero)
    assert (positive is event) == (sign > 0)
    assert (negative is event) == (sign < 0)
    if sign > 0:
        assert positive.derivative == iv.zero and negative.derivative == iv.zero
    else:
        assert positive.derivative == iv.zero and negative.derivative == iv.zero


def test_two_event_tags_are_independent_and_unary_chains_strip_tags():
    from mpmath import iv

    context = cast(Any, iv)
    iv.dps = 50
    mp = paired._SectorJetMath(context, 1)
    qx = paired.SectorEventJet(context.zero, context.one, 1, True, -1)
    qy = paired.SectorEventJet(context.zero, context.zero, 1, True, 1)
    assert min(qx, mp.zero) is qx
    assert max(qy, mp.zero) is qy
    transformed = mp.exp(qx * 2 + 1)
    assert type(transformed) is base.IntervalJet
    assert transformed.value == context.exp(context.one)
    assert transformed.derivative == 2 * context.exp(context.one)
    with pytest.raises(paired.NormalReferenceRefusal, match="two independent tagged events"):
        qx < qy
    moving_zero = base.IntervalJet(context.zero, context.one, 1)
    with pytest.raises(paired.NormalReferenceRefusal, match="nonconstant untagged zero"):
        qx < moving_zero


def test_bilinear_sampler_uses_exact_coordinate_cells_and_preserves_jet_derivatives():
    from mpmath import iv

    context = cast(Any, iv)
    iv.dps = 60
    mp = paired._SectorJetMath(context, 1)
    coordinates: list[list[float]] = [[0.35, 0.65], [1.25, 2.4], [2.5, 3.1], [1.7, 1.45]]
    field = []
    for i in range(4):
        row = []
        for j in range(5):
            row.append(base.IntervalJet(context.mpf(2 + 3*i - 4*j),
                                        context.mpf(-1 + 2*i + 5*j), 1))
        field.append(row)
    samples = paired._point_samples(field, coordinates, mp)
    for sample, (row, column) in zip(samples, coordinates, strict=True):
        expected_value = context.mpf(2) + 3 * context.mpf(row) - 4 * context.mpf(column)
        expected_derivative = -1 + 2 * context.mpf(row) + 5 * context.mpf(column)
        assert bool(sample.value.a <= expected_value.a)
        assert bool(sample.value.b >= expected_value.b)
        assert bool(sample.derivative.a <= expected_derivative.a)
        assert bool(sample.derivative.b >= expected_derivative.b)


@pytest.mark.parametrize("normal_axis", [0, 1])
def test_selected_flux_binary_diagnostics_keep_exact_zero_and_axis_derivatives(normal_axis):
    from mpmath import iv

    context = cast(Any, iv)
    iv.dps = 60
    dx = context.mpf("0.11") if normal_axis == 0 else context.zero
    dy = context.mpf("0.08") if normal_axis == 1 else context.zero
    qx = paired.SectorEventJet(context.zero, dx, 1, True, -1)
    qy = paired.SectorEventJet(context.zero, dy, 1, True, 1)
    result = paired._selected_flux_binary(qx, qy)
    exact_zero = {"lower": [0, 0, 0, 0], "upper": [0, 0, 0, 0]}
    assert result["selected_flux_values_binary"] == [exact_zero, exact_zero]
    assert result["selected_flux_derivatives_binary"] == [paired.normal._mpi(dx), paired.normal._mpi(dy)]


def test_invalid_projection_row_is_not_treated_as_a_branch_refusal():
    fixture = {
        "psi_basis": [], "projected_psi_basis": [], "normal_weights": [],
        "coefficient_limits": [], "normal_scales": [],
    }
    with pytest.raises(ValueError) as error:
        paired.validate_projection(fixture)
    assert type(error.value) is ValueError
