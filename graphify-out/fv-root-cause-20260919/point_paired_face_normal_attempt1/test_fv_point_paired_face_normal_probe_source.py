"""Pure wiring checks for paired normal capture auditing; no reference evaluation."""
from fractions import Fraction
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_paired_face_normal_probe as probe


def _native_row() -> dict[str, Any]:
    qx, qy = [[1] * 6 for _ in range(4)], [[1] * 5 for _ in range(5)]
    qx[3][4], qy[3][0] = 0, 0
    return {"step": 0, "stage": 0,
            "x": {"slope_sign": [[1, 0, -1], [1, 1, 1]],
                 "choose_left": [[True, False, False], [True, False, True]]},
            "y": {"slope_sign": [[1] * 3, [-1, 0, 1]],
                 "choose_left": [[True] * 3, [False, False, True]]},
            "qx_sign": qx, "qy_sign": qy}


def _branch_row(sides: tuple[int, int]) -> dict[str, Any]:
    row = _native_row()
    row["qx_sign"][3][4], row["qy_sign"][3][0] = sides
    return {"step": 0, "stage": 0,
            "x": [1, 0, 2, 1, 2, 1], "y": [1, 1, 1, 2, 0, 1],
            "qx_sign": [x for line in row["qx_sign"] for x in line],
            "qy_sign": [x for line in row["qy_sign"] for x in line]}


def test_limiter_bins_other_faces_and_selected_orthant_are_checked():
    native: list[dict[str, Any]] = [_native_row() for _ in range(36)]
    for index, row in enumerate(native):
        row["step"], row["stage"] = index // 2, index % 2
    branch: list[dict[str, Any]] = [_branch_row((-1, 1)) for _ in range(36)]
    for index, row in enumerate(branch):
        row["step"], row["stage"] = index // 2, index % 2
    probe.audit_trace(native, branch, (-1, 1))
    branch[4]["qx_sign"][0] = -1
    with pytest.raises(probe.UnsupportedNormal, match="native other faces"):
        probe.audit_trace(native, branch, (-1, 1))


def test_malformed_native_trace_is_not_classified_as_unsupported_normal():
    with pytest.raises(ValueError, match="36 analysis"):
        probe.audit_trace([_native_row()], [], (1, 1))


def test_binary_interval_intersection_and_strict_orientation_use_exact_bounds():
    a = {"J_binary": {"lower": [0, 1, -1, 1], "upper": [0, 1, 1, 1]}}
    b = {"objective_mpi": {"lower": [0, 1, 0, 1], "upper": [0, 3, 0, 2]}}
    overlap = probe.intersection([probe.interval(a, "objective"), probe.interval(b, "objective")])
    assert overlap is not None and overlap[:2] == (Fraction(1), Fraction(2))
    assert probe.strict_orientation((Fraction(-2), Fraction(-1)), (Fraction(1), Fraction(3)))
    assert not probe.strict_orientation((Fraction(-2), Fraction(0)), (Fraction(1), Fraction(3)))


def test_native_objective_crosscheck_uses_positive_component_scale_without_unit_floor():
    row = {"J_binary": {"lower": [0, 3, -1, 2], "upper": [0, 3, -1, 2]},
           "robust_cost_bounds": ["0.2", "0.2"], "prior_cost_bounds": ["0.3", "0.3"],
           "smooth_cost_bounds": ["0", "0"]}
    passed = probe.primal_crosscheck(1.5, row)
    failed = probe.primal_crosscheck(1.5001, row)
    assert passed["passed"] and passed["scale"] == 3.5
    assert not failed["passed"] and failed["budget"] < 1e-10


def _mpi_point(value):
    numerator, denominator = value.as_integer_ratio()
    exponent = -(denominator.bit_length() - 1)
    raw = [0, numerator, exponent, numerator.bit_length()]
    return {"lower": raw, "upper": raw}


def test_evaluation_set_rejects_duplicate_or_unknown_axis_and_fluxes_are_pinned():
    rows: list[dict[str, Any]] = [
        {"normal_axis": axis, "requested_sides": [sx, sy], "sector_sides": [sx, sy]}
        for axis in (0, 1) for sx in (-1, 1) for sy in (-1, 1)
    ]
    probe.audit_evaluation_set(rows)
    with pytest.raises(ValueError, match="each axis/orthant once"):
        probe.audit_evaluation_set(rows[:-1] + [rows[0]])
    with pytest.raises(ValueError, match="unknown axis"):
        probe.audit_evaluation_set([dict(rows[0], normal_axis=2)])

    zero = _mpi_point(0.0)
    x, y = _mpi_point(0.11), _mpi_point(0.08)
    probe.audit_fluxes({"selected_flux_values_binary": [zero, zero],
                        "selected_flux_derivatives_binary": [x, zero]}, 0, [0.11, 0.08])
    probe.audit_fluxes({"selected_flux_values_binary": [zero, zero],
                        "selected_flux_derivatives_binary": [zero, y]}, 1, [0.11, 0.08])
    with pytest.raises(ValueError, match="not exact zero"):
        probe.audit_fluxes({"selected_flux_values_binary": [x, zero],
                            "selected_flux_derivatives_binary": [x, zero]}, 0, [0.11, 0.08])
