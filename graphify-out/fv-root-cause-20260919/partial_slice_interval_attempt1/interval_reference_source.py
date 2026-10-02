"""Independent mpmath reference for one frozen two-hole FV objective point.

This module intentionally imports no ADVAR/production code. It evaluates the
declared 3x4x5 full-known minmod profile from a captured fixture; it is not a
general FV solver or a branch-admission certificate.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any


@contextmanager
def _arithmetic(dps: int, interval: bool):
    if interval:
        from mpmath import iv
        previous = iv.dps
        iv.dps = dps
        try:
            yield iv
        finally:
            iv.dps = previous
    else:
        from mpmath import mp
        with mp.workdps(dps):
            yield mp


def _decide(comparison: bool | None) -> bool:
    if comparison is None:
        raise ValueError("interval arithmetic could not certify a numerical branch")
    return comparison


def _tanh(value: Any, mp: Any) -> Any:
    if hasattr(mp, "tanh"):
        return mp.tanh(value)
    exponential_minus_one = mp.expm1(2 * value)
    return exponential_minus_one / (exponential_minus_one + 2)


def _mp(value: object, mp: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError("fixture numbers and controls must be int, float, or decimal strings")
    return mp.mpf(value)


def _grid(values: object, height: int, width: int, mp: Any, name: str) -> list[list[Any]]:
    if not isinstance(values, (list, tuple)) or len(values) != height:
        raise ValueError(f"{name} must have {height} rows")
    result = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError(f"{name} rows must have width {width}")
        result.append([_mp(value, mp) for value in row])
    return result


def _cube(values: object, times: int, height: int, width: int, mp: Any,
          name: str) -> list[list[list[Any]]]:
    if not isinstance(values, (list, tuple)) or len(values) != times:
        raise ValueError(f"{name} must have {times} times")
    return [_grid(frame, height, width, mp, name) for frame in values]


def _sign(value: Any) -> int:
    return 1 if _decide(value > 0) else (-1 if _decide(value < 0) else 0)


def _choice(left: Any, right: Any) -> int:
    if _decide(left * right <= 0):
        return 0
    if _decide(abs(left) < abs(right)):
        return 1
    if _decide(abs(right) < abs(left)):
        return 2
    return 3


def _softplus(value: Any, mp: Any) -> Any:
    # Match torch.nn.functional.softplus's default beta=1, threshold=20 split.
    return value if _decide(value > 20) else mp.log1p(mp.exp(value))


def _euler_minmod(
    echo: list[list[Any]], qx: list[list[Any]], qy: list[list[Any]],
    edges: list[list[Any]], dt: Any, area: Any, step: int, stage: int,
    trace: list[dict[str, Any]], mp: Any,
) -> list[list[Any]]:
    height, width = len(echo), len(echo[0])
    sx = [[mp.zero for _ in range(width)] for _ in range(height)]
    sy = [[mp.zero for _ in range(width)] for _ in range(height)]
    choices_x: list[int] = []
    choices_y: list[int] = []
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            ax = echo[row][column] - echo[row][column - 1]
            bx = echo[row][column + 1] - echo[row][column]
            ay = echo[row][column] - echo[row - 1][column]
            by = echo[row + 1][column] - echo[row][column]
            choice_x, choice_y = _choice(ax, bx), _choice(ay, by)
            choices_x.append(choice_x)
            choices_y.append(choice_y)
            sx[row][column] = (
                mp.zero if choice_x == 0 else
                (ax if choice_x == 1 else (bx if choice_x == 2 else ax))
            )
            sy[row][column] = (
                mp.zero if choice_y == 0 else
                (ay if choice_y == 1 else (by if choice_y == 2 else ay))
            )
    trace.append({
        "step": step,
        "stage": stage,
        "x": choices_x,
        "y": choices_y,
        "qx_sign": [_sign(value) for row in qx for value in row],
        "qy_sign": [_sign(value) for row in qy for value in row],
    })

    left, right, bottom, top = edges
    left_face = [[echo[i][j] - sx[i][j] / 2 for j in range(width)] for i in range(height)]
    right_face = [[echo[i][j] + sx[i][j] / 2 for j in range(width)] for i in range(height)]
    bottom_face = [[echo[i][j] - sy[i][j] / 2 for j in range(width)] for i in range(height)]
    top_face = [[echo[i][j] + sy[i][j] / 2 for j in range(width)] for i in range(height)]
    updated = [[mp.zero for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            left_source = left[i] if j == 0 else right_face[i][j - 1]
            right_source = right[i] if j == width - 1 else left_face[i][j + 1]
            bottom_source = bottom[j] if i == 0 else top_face[i - 1][j]
            top_source = top[j] if i == height - 1 else bottom_face[i + 1][j]
            incoming = (
                max(qx[i][j], mp.zero) * left_source
                - min(qx[i][j + 1], mp.zero) * right_source
                + max(qy[i][j], mp.zero) * bottom_source
                - min(qy[i + 1][j], mp.zero) * top_source
            )
            outgoing = (
                max(qx[i][j + 1], mp.zero) * right_face[i][j]
                - min(qx[i][j], mp.zero) * left_face[i][j]
                + max(qy[i + 1][j], mp.zero) * top_face[i][j]
                - min(qy[i][j], mp.zero) * bottom_face[i][j]
            )
            updated[i][j] = echo[i][j] - dt * (outgoing / area) + dt * (incoming / area)
    return updated


def evaluate(fixture: dict[str, Any], control: Sequence[float | str], dps: int,
             *, interval: bool = False) -> dict[str, Any]:
    """Evaluate the frozen 20+5+1-control, two-interval objective in mpmath.

    Floats in the fixture are converted from their exact binary64 values;
    decimal-string controls remain decimal inputs for high-precision chart
    endpoints. The supported fixture has three fully-known times with two
    missing cells, per-frame ungrouped common-bias whitening, positive growth,
    and no neural-prior standard deviations.
    """
    if type(dps) is not int or dps < 20:
        raise ValueError("dps must be an integer of at least 20")
    with _arithmetic(dps, interval) as mp:
        background = fixture["background_dbz"]
        height, width = len(background), len(background[0])
        if (height, width) != (4, 5):
            raise ValueError("reference supports the fixed 4x5 field only")
        control_mp = [_mp(value, mp) for value in control]
        flow_limits = [_mp(value, mp) for value in fixture["coefficient_limits"]]
        if len(control_mp) != 26 or len(flow_limits) != 5:
            raise ValueError("reference requires the original 20+5+1 control layout")
        if fixture.get("per_frame") is not True or fixture.get("tile_size", fixture.get("tile_size_px", 0)) != 0:
            raise ValueError("reference supports ungrouped per-frame whitening only")
        if fixture.get("common_bias_group_index") is not None:
            raise ValueError("grouped common-bias whitening is outside this reference")
        if fixture.get("neural_prior_std_dbz", fixture.get("neural_prior_std")) is not None:
            raise ValueError("reference requires the default no-neural-prior contract")
        if fixture.get("censored_mask") is not None and any(
            bool(value) for frame in fixture["censored_mask"] for row in frame for value in row
        ):
            raise ValueError("reference supports detected and missing cells, no censoring")

        background_mp = _grid(background, height, width, mp, "background_dbz")
        obs = _cube(fixture["observation_dbz"], 3, height, width, mp, "observation_dbz")
        valid = fixture["valid_mask"]
        if len(valid) != 3 or any(len(frame) != height for frame in valid):
            raise ValueError("valid_mask must have shape [3,4,5]")
        valid_mp = [[[bool(value) for value in row] for row in frame] for frame in valid]
        if sum(sum(sum(row) for row in frame) for frame in valid_mp) != 58:
            raise ValueError("reference expects 58 detected and two missing observations")
        if "detected_mask" in fixture and fixture["detected_mask"] != valid:
            raise ValueError("reference requires every valid observation to be detected")
        std = _cube(fixture["std_dbz"], 3, height, width, mp, "std_dbz")
        quality = _cube(fixture["quality_weight"], 3, height, width, mp, "quality_weight")
        mode = _cube(fixture["whitener_mode"], 3, height, width, mp, "whitener_mode")

        floor_dbz = _mp(fixture["floor_dbz"], mp)
        transform_scale = _mp(fixture["transform_scale"], mp)
        epsilon = _mp(fixture["transform_epsilon"], mp)
        increment_ratio = _mp(fixture["increment_ratio"], mp)
        echo_floor = _mp(fixture["echo_floor"], mp)
        echo_factor = _mp(fixture["echo_exponent_factor"], mp)
        dbz_factor = _mp(fixture["dbz_log_factor"], mp)

        initial_echo: list[list[Any]] = []
        for i in range(height):
            echo_row = []
            for j in range(width):
                offset = (background_mp[i][j] - floor_dbz) / transform_scale
                if not _decide(offset >= epsilon):
                    offset = epsilon
                latent = offset + mp.log(-mp.expm1(-offset))
                analyzed = floor_dbz + transform_scale * _softplus(
                    latent + increment_ratio * control_mp[i * width + j], mp,
                )
                echo_row.append(echo_floor * mp.expm1(echo_factor * (analyzed - floor_dbz)))
            initial_echo.append(echo_row)

        psi_basis = fixture["psi_basis"]
        if len(psi_basis) != 5:
            raise ValueError("psi_basis must contain five streamfunction modes")
        basis = [_grid(item, height + 1, width + 1, mp, "psi_basis") for item in psi_basis]
        coeff = [flow_limits[k] * _tanh(control_mp[height * width + k], mp) for k in range(5)]
        psi = [[mp.fsum(coeff[k] * basis[k][i][j] for k in range(5))
                for j in range(width + 1)] for i in range(height + 1)]
        qx = [[psi[i + 1][j] - psi[i][j] for j in range(width + 1)]
              for i in range(height)]
        qy = [[-(psi[i][j + 1] - psi[i][j]) for j in range(width)]
              for i in range(height + 1)]
        face_flux = qy[2][0]

        growth_limit = _mp(fixture["growth_limit"], mp)
        interval_growth = growth_limit * _tanh(control_mp[-1], mp)
        if _decide(interval_growth <= 0):
            raise ValueError("reference supports positive growth only")
        substeps = fixture["substeps"]
        if type(substeps) is not int or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        step_growth = interval_growth / substeps
        growth_factor = mp.exp(step_growth)
        dt = _mp(fixture["dt"], mp)
        area = _mp(fixture["area"], mp)
        boundaries = fixture["boundary_echo"]
        if len(boundaries) != 2 * substeps:
            raise ValueError("boundary_echo must cover two intervals of substeps")
        edge_steps = []
        for step_edges in boundaries:
            if len(step_edges) != 2 or any(len(stage) != 4 for stage in step_edges):
                raise ValueError("boundary_echo entries must be [step][stage][left,right,bottom,top]")
            converted = []
            for stage in step_edges:
                converted.append([
                    [_mp(value, mp) for value in edge]
                    for edge in stage
                ])
            if any(len(converted[stage][edge]) != (height if edge < 2 else width)
                   for stage in range(2) for edge in range(4)):
                raise ValueError("boundary edge lengths do not match the 4x5 field")
            edge_steps.append(converted)

        frames = [initial_echo]
        trace: list[dict[str, Any]] = []
        echo = initial_echo
        for step, step_edges in enumerate(edge_steps):
            grown = [[value * growth_factor for value in row] for row in echo]
            grown_edges = [[value * growth_factor for value in edge]
                           for edge in step_edges[0]]
            stage1 = _euler_minmod(grown, qx, qy, grown_edges, dt, area,
                                   step, 0, trace, mp)
            stage2 = _euler_minmod(stage1, qx, qy, step_edges[1], dt, area,
                                   step, 1, trace, mp)
            echo = [[(grown[i][j] + stage2[i][j]) / 2 for j in range(width)]
                    for i in range(height)]
            if (step + 1) % substeps == 0:
                frames.append(echo)

        predicted_dbz = [[[
            floor_dbz + dbz_factor * mp.log1p(value / echo_floor)
            for value in row
        ] for row in frame] for frame in frames]
        standardized = [[[mp.zero for _ in range(width)] for _ in range(height)] for _ in range(3)]
        for t in range(3):
            for i in range(height):
                for j in range(width):
                    if valid_mp[t][i][j]:
                        error = predicted_dbz[t][i][j] - obs[t][i][j]
                        standardized[t][i][j] = mp.sqrt(quality[t][i][j]) * error / std[t][i][j]

        whitened = [[[mp.zero for _ in range(width)] for _ in range(height)] for _ in range(3)]
        bias = _mp(fixture["bias_std"], mp)
        for t in range(3):
            norm2 = mp.fsum(mode[t][i][j] ** 2 for i in range(height) for j in range(width))
            projection = mp.fsum(mode[t][i][j] * standardized[t][i][j]
                                 for i in range(height) for j in range(width))
            coefficient = (mp.zero if norm2 == 0 or bias == 0 else
                           (1 - 1 / mp.sqrt(1 + bias**2 * norm2)) / norm2)
            for i in range(height):
                for j in range(width):
                    whitened[t][i][j] = standardized[t][i][j] - coefficient * mode[t][i][j] * projection

        delta = _mp(fixture["robust_delta"], mp)
        robust_cost = mp.fsum(
            whitened[t][i][j] ** 2 / (
                mp.sqrt(1 + (whitened[t][i][j] / delta) ** 2) + 1
            )
            for t in range(3) for i in range(height) for j in range(width)
            if valid_mp[t][i][j]
        )
        prior_cost = mp.fsum(value**2 for value in control_mp) / 2
        smooth_weight = _mp(fixture["smooth_weight"], mp)
        left_indices = fixture["smooth_left_index"]
        right_indices = fixture["smooth_right_index"]
        physical_weights = fixture["smooth_physical_weight"]
        if not (len(left_indices) == len(right_indices) == len(physical_weights)):
            raise ValueError("smoothness edge indices and weights must have equal lengths")
        smooth_cost = mp.zero
        for left, right, physical in zip(left_indices, right_indices, physical_weights, strict=True):
            difference = control_mp[right] - control_mp[left]
            smooth_cost += smooth_weight * _mp(physical, mp) * difference**2 / 2
        objective = robust_cost + prior_cost + smooth_cost

        def strings(value: Any) -> Any:
            if isinstance(value, list):
                return [strings(item) for item in value]
            if isinstance(value, dict):
                return {key: strings(item) for key, item in value.items()}
            if isinstance(value, mp.mpf):
                return mp.nstr(value, n=dps)
            return value

        return {
            "arithmetic": "interval" if interval else "point",
            "objective": mp.nstr(objective, n=dps),
            "objective_interval_binary": [list(bound) for bound in objective._mpi_] if interval else None,
            "robust_observation_cost": mp.nstr(robust_cost, n=dps),
            "control_prior_cost": mp.nstr(prior_cost, n=dps),
            "field_smoothness_cost": mp.nstr(smooth_cost, n=dps),
            "whitened_residual": strings(whitened),
            "predicted_dbz": strings(predicted_dbz),
            "face_flux_qy_2_0": mp.nstr(face_flux, n=dps),
            "branch_choices": trace,
            "scope": "independent high-precision fixed-input calculation; no production imports or branch certificate",
        }
