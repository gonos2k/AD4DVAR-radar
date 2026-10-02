"""Small analytic checks for the independent fixed-case FV reference."""
import math

import mpmath

from examples.weather_scenarios.fv_slice_precision_reference import evaluate


def _grid(value, height=4, width=5):
    return [[value for _ in range(width)] for _ in range(height)]


def _fixture():
    valid = [[[True] * 5 for _ in range(4)] for _ in range(3)]
    valid[1][1][2] = False
    valid[2][2][3] = False
    mode = [[[1.0 if valid[t][i][j] else 0.0 for j in range(5)]
             for i in range(4)] for t in range(3)]
    no_edges = [[0.0] * 4, [0.0] * 4, [0.0] * 5, [0.0] * 5]
    boundaries = [[[list(edge) for edge in no_edges] for _stage in range(2)]
                  for _step in range(2)]
    return {
        "background_dbz": _grid(20.0),
        "psi_basis": [[[0.0] * 6 for _ in range(5)] for _ in range(5)],
        "coefficient_limits": [1.0] * 5,
        "growth_limit": 0.01,
        "floor_dbz": 0.0,
        "echo_floor": 1.0,
        "echo_exponent_factor": math.log(10.0) / 10.0,
        "dbz_log_factor": 10.0 / math.log(10.0),
        "transform_scale": 1.0,
        "increment_ratio": 4.0,
        "transform_epsilon": 1e-6,
        "substeps": 1,
        "dt": 60.0,
        "area": 100.0,
        "boundary_echo": boundaries,
        "observation_dbz": [_grid(20.0) for _ in range(3)],
        "valid_mask": valid,
        "std_dbz": [_grid(1.0) for _ in range(3)],
        "quality_weight": [_grid(1.0) for _ in range(3)],
        "whitener_mode": mode,
        "bias_std": 0.25,
        "per_frame": True,
        "smooth_left_index": [],
        "smooth_right_index": [],
        "smooth_physical_weight": [],
        "smooth_weight": 0.0,
        "robust_delta": 2.0,
    }


def test_constant_flow_growth_whitener_and_branch_trace_match_closed_form():
    fixture = _fixture()
    control = [0.0] * 25 + [0.1]
    result = evaluate(fixture, control, dps=60)
    mp = mpmath.mp
    with mp.workdps(60):
        gamma = mp.mpf(fixture["growth_limit"]) * mp.tanh(mp.mpf(control[-1]))
        initial_echo = mp.mpf(fixture["echo_floor"]) * mp.expm1(
            mp.mpf(fixture["echo_exponent_factor"]) * mp.mpf(20.0)
        )
        active_counts = (20, 19, 19)
        residuals = []
        for lead in range(3):
            expected_dbz = mp.mpf(fixture["floor_dbz"]) + mp.mpf(
                fixture["dbz_log_factor"]
            ) * mp.log1p(initial_echo * mp.exp(lead * gamma))
            actual = mp.mpf(result["predicted_dbz"][lead][0][0])
            assert mp.almosteq(actual, expected_dbz, rel_eps=mp.mpf("1e-55"))
            shrink = 1 / mp.sqrt(
                1 + mp.mpf(fixture["bias_std"]) ** 2 * active_counts[lead]
            )
            residuals.append((expected_dbz - 20) * shrink)
        for t in range(3):
            assert mp.almosteq(mp.mpf(result["whitened_residual"][t][0][0]),
                               residuals[t], rel_eps=mp.mpf("1e-55"))
        expected_robust = mp.fsum(
            active_counts[t] * r**2 / (mp.sqrt(1 + (r / 2) ** 2) + 1)
            for t, r in enumerate(residuals)
        )
        expected_objective = expected_robust + mp.mpf(control[-1]) ** 2 / 2
        assert mp.almosteq(mp.mpf(result["objective"]), expected_objective,
                           rel_eps=mp.mpf("1e-55"))
    assert len(result["branch_choices"]) == 4
    assert all(row[axis] == [0] * 6 for row in result["branch_choices"] for axis in ("x", "y"))
    assert all(not any(row["qx_sign"]) and not any(row["qy_sign"])
               for row in result["branch_choices"])
