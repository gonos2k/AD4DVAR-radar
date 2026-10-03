"""Independent interval chart, sampler and captured-fixture regressions."""
from fractions import Fraction
from typing import Any, cast

import pytest

from examples.weather_scenarios import fv_active_face_normal_reference as base
from examples.weather_scenarios import fv_point_paired_face_normal_reference as paired
from examples.weather_scenarios import fv_point_single_face_normal_reference as single


def _projection_fixture() -> dict[str, Any]:
    weights = single._WEIGHTS
    basis = []
    for k in range(5):
        mode = [[float((k + 1) * (r + 2) + (c + 1) * (k + 2)) for c in range(6)]
                for r in range(5)]
        mode[4][4], mode[3][4] = float(weights[k]), 0.0
        mode[3][1], mode[3][0] = float(k + 3), 0.0
        basis.append(mode)
    projected = []
    for k in (1, 2, 3, 4):
        ratio = float(weights[k] / weights[0])
        projected.append([[basis[k][r][c] - ratio * basis[0][r][c] for c in range(6)]
                          for r in range(5)])
    return {"psi_basis": basis, "projected_psi_basis": projected,
            "normal_weights": [float(v) for v in weights],
            "coefficient_limits": [0.11, 0.08, 0.7, 0.6, 0.5], "normal_scale": 0.11}


def test_single_projection_enforces_only_qx_zero_and_matches_exact_fraction_formula():
    fixture = _projection_fixture()
    single.validate_projection(fixture)
    projected = fixture["projected_psi_basis"]
    assert all(mode[4][4] == mode[3][4] for mode in projected)
    assert any(mode[3][1] != mode[3][0] for mode in projected)
    broken = {**fixture, "projected_psi_basis": [[row[:] for row in mode]
                                                  for mode in projected]}
    broken["projected_psi_basis"][0][0][0] += 0.125
    with pytest.raises(ValueError, match="exact qx-face formula"):
        single.validate_projection(broken)


def test_single_face_lift_restores_pivot_prior_and_preserves_open_domain():
    from mpmath import iv

    context = cast(Any, iv)
    iv.dps = 60
    mp = paired._SectorJetMath(context, 1)
    tangent = [base.IntervalJet(context.zero, context.zero, 1) for _ in range(25)]
    eta = base.IntervalJet(context.zero, context.one, 1)
    fixture = _projection_fixture()
    full, _, alpha0 = single._lift(tangent, eta, fixture, mp)
    assert len(full) == 26
    assert alpha0.value == context.zero and alpha0.derivative == context.mpf(0.11)
    assert full[20].value == context.zero and full[20].derivative == context.one
    assert full[21:25] == tangent[20:24]
    full_prior = mp.fsum(value**2 for value in full) / 2
    tangent_prior = mp.fsum(value**2 for value in tangent) / 2
    assert full_prior.value == tangent_prior.value

    outside = tangent[:]
    outside[21] = base.IntervalJet(context.mpf(100), context.zero, 1)
    with pytest.raises(single.SingleFaceNormalRefusal, match="pivot coefficient domain"):
        single._lift(outside, base.IntervalJet(context.zero, context.zero, 1), fixture, mp)


def test_normal_orientation_requires_left_negative_and_right_positive_slopes():
    assert single.strict_normal_orientation((Fraction(-2), Fraction(-1)),
                                            (Fraction(1, 10), Fraction(1, 5)))
    assert not single.strict_normal_orientation((Fraction(-2), Fraction(-1)),
                                                (Fraction(-1, 5), Fraction(-1, 10)))
    assert not single.strict_normal_orientation((Fraction(-2), Fraction(0)),
                                                (Fraction(1, 10), Fraction(1, 5)))


def test_point_bilinear_helper_preserves_affine_values_and_interval_derivatives():
    from mpmath import iv

    context = cast(Any, iv)
    iv.dps = 60
    mp = paired._SectorJetMath(context, 1)
    coordinates: list[list[float]] = [[0.35, 0.65], [1.25, 2.4], [2.5, 3.1], [1.7, 1.45]]
    field = [[base.IntervalJet(context.mpf(2 + 3*i - 4*j),
                               context.mpf(-1 + 2*i + 5*j), 1)
              for j in range(5)] for i in range(4)]
    samples = paired._point_samples(field, coordinates, mp)
    for sample, (row, col) in zip(samples, coordinates, strict=True):
        expected = context.mpf(2) + 3*context.mpf(row) - 4*context.mpf(col)
        derivative = -1 + 2*context.mpf(row) + 5*context.mpf(col)
        assert bool(sample.value.a <= expected.a) and bool(sample.value.b >= expected.b)
        assert bool(sample.derivative.a <= derivative.a) and bool(sample.derivative.b >= derivative.b)


def test_untagged_exact_tie_remains_an_error():
    from mpmath import iv

    context = cast(Any, iv)
    iv.dps = 50
    untagged = base.IntervalJet(context.zero, context.zero, 1)
    with pytest.raises(ValueError, match="nonsmooth tie"):
        untagged >= 0


def test_captured_dyadic_normal_weights_reach_complete_interval_evaluation():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "graphify-out/fv-root-cause-20260919/R2_POINT_SINGLE_QX_NORMAL_INPUT_20261003_RELEASE2.json"
    capture = json.loads(path.read_text())
    result = single.evaluate(capture["fixture"], capture["tangent"], -1, dps=80)
    assert len(result["branch_choices"]) == 36
    assert result["side"] == -1
