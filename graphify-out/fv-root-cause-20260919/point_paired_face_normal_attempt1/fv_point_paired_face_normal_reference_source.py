"""Independent first-order interval reference at the paired point-FV face event.

Fixture contract (all numeric arrays are nested int/float/decimal-string values):
``background_dbz[4][5]``, ``psi_basis[5][5][6]``,
``projected_psi_basis[3][5][6]``, ``coefficient_limits[5]``,
``normal_weights[2][5]``, ``normal_scales[2]``, transform scalars
``floor_dbz``, ``transform_scale``, ``transform_epsilon``, ``increment_ratio``,
``echo_floor``, ``echo_exponent_factor``, ``dbz_log_factor``, ``growth_limit``,
``substeps``, ``dt``, ``area``, ``boundary_echo[2*substeps][2][4][edge]``,
``point_dbz[3][4]``, ``std_dbz[3][4]``, ``quality_weight[3][4]``,
``coordinates[4][2]``, symmetric ``whitener[4][4]``, ``robust_delta``, ``smooth_weight``,
``smooth_left_index``, ``smooth_right_index``, and ``smooth_physical_weight``.

This evaluates directional derivatives only. It is neither an optimizer nor a
normal-minimum/response certificate, and imports no production model code.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from examples.weather_scenarios import fv_active_face_normal_reference as normal
from examples.weather_scenarios import fv_slice_precision_reference as reference


_W_X = (1.0, 0.0, 4.0, -3.5, 16.0)
_W_Y = (0.0, -1.0, -3.0, -0.5, -3.0)
_KEEP_PAIR = (2, 3, 4)


class NormalReferenceRefusal(ValueError):
    """A supported strict-domain or interval branch could not be certified."""


def _certify(comparison: bool | None) -> bool:
    try:
        return reference._decide(comparison)
    except ValueError as error:
        raise NormalReferenceRefusal(str(error)) from error


@dataclass(frozen=True, eq=False)
class SectorEventJet(normal.IntervalJet):
    """Exact-zero face flux whose selected sector is explicit per event."""

    sector_sign: int = 1

    def __post_init__(self) -> None:
        if self.sector_sign not in (-1, 1) or not normal._is_exact_zero(self.value):
            raise ValueError("sector event must be an exact zero with sign +/-1")

    def _order(self, other: Any) -> int:
        right = self._coerce(other)
        greater = self.value > right.value
        if greater is True:
            return 1
        less = self.value < right.value
        if less is True:
            return -1
        equal = self.value == right.value
        if equal is True and normal._is_exact_zero(self.value):
            if isinstance(right, SectorEventJet):
                raise NormalReferenceRefusal("cannot order two independent tagged events")
            if right.event or not normal._is_exact_zero(right.derivative):
                raise NormalReferenceRefusal("selected event ties a nonconstant untagged zero")
            return self.sector_sign
        if equal is True:
            raise NormalReferenceRefusal("untagged nonsmooth tie in interval comparison")
        raise NormalReferenceRefusal("interval arithmetic cannot decide a branch comparison")


@dataclass(frozen=True, eq=False)
class _SectorZeroJet(normal.IntervalJet):
    """Zero sentinel that lets builtin max/min inspect a tagged event from either side."""

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, SectorEventJet) and normal._is_exact_zero(other.value):
            return other.sector_sign < 0
        return self._order(other) > 0

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, SectorEventJet) and normal._is_exact_zero(other.value):
            return other.sector_sign > 0
        return self._order(other) < 0


class _SectorJetMath(normal._JetMath):
    @property
    def zero(self) -> _SectorZeroJet:
        return _SectorZeroJet(self.context.zero, self.context.zero, self.side)


def _fraction(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError("fixture numbers must be int, float, or decimal strings")
    return Fraction.from_float(float(value)) if isinstance(value, float) else Fraction(value)


def _matrix(values: Any, rows: int, columns: int, name: str) -> list[list[Any]]:
    if (not isinstance(values, (list, tuple)) or len(values) != rows
            or any(not isinstance(row, (list, tuple)) or len(row) != columns for row in values)):
        raise ValueError(f"{name} must have shape [{rows},{columns}]")
    return [list(row) for row in values]


def _basis(values: Any, count: int, name: str) -> list[list[list[Any]]]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must contain {count} modes")
    return [_matrix(_matrix(mode, 5, 6, name), 5, 6, name) for mode in values]


def validate_projection(fixture: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Check captured weights and the paired projection in exact rationals."""
    basis = _basis(fixture["psi_basis"], 5, "psi_basis")
    paired = _basis(fixture["projected_psi_basis"], 3, "projected_psi_basis")
    b = [[[ _fraction(value) for value in row] for row in mode] for mode in basis]
    p = [[[ _fraction(value) for value in row] for row in mode] for mode in paired]
    wx, wy = tuple(map(_fraction, _W_X)), tuple(map(_fraction, _W_Y))
    captured_weights = fixture["normal_weights"]
    if not isinstance(captured_weights, (list, tuple)) or len(captured_weights) != 2:
        raise ValueError("normal_weights must contain the two pinned five-entry rows")
    if (tuple(map(_fraction, captured_weights[0])) != wx
            or tuple(map(_fraction, captured_weights[1])) != wy):
        raise ValueError("captured normal weights differ from the pinned paired rows")
    for k, mode in enumerate(b):
        if mode[4][4] - mode[3][4] != wx[k]:
            raise ValueError("psi basis q_x[3,4] weights differ from the pinned row")
        if -(mode[3][1] - mode[3][0]) != wy[k]:
            raise ValueError("psi basis q_y[3,0] weights differ from the pinned row")
    for i, k in enumerate(_KEEP_PAIR):
        for r in range(5):
            for c in range(6):
                # The pivot block is diag(1,-1), so eliminating both pivot
                # coefficients gives this exact rational basis projection.
                expected = (b[k][r][c] - wx[k] / wx[0] * b[0][r][c]
                            - wy[k] / wy[1] * b[1][r][c])
                if p[i][r][c] != expected:
                    raise ValueError("paired projected basis differs from the pinned rational formula")
        if p[i][3][0] != p[i][3][1] or p[i][4][4] != p[i][3][4]:
            raise ValueError("second projection did not preserve both exact selected zeros")
    limits = fixture["coefficient_limits"]
    if not isinstance(limits, (list, tuple)) or len(limits) != 5:
        raise ValueError("coefficient_limits must have length five")
    limit_f = tuple(map(_fraction, limits))
    if any(value <= 0 for value in limit_f):
        raise ValueError("coefficient limits must be positive")
    normal_scales = fixture["normal_scales"]
    if not isinstance(normal_scales, (list, tuple)) or len(normal_scales) != 2:
        raise ValueError("normal_scales must contain qx and qy scales")
    scales = tuple(map(_fraction, normal_scales))
    expected_scales = (abs(wx[0]) * limit_f[0], abs(wy[1]) * limit_f[1])
    if scales != expected_scales or scales != (_fraction(0.11), _fraction(0.08)):
        raise ValueError("paired face scales must match the original pivot rows and (.11,.08)")
    return list(wx), list(wy)


def _paired_projected_basis(fixture: dict[str, Any], mp: _SectorJetMath):
    return [reference._grid(mode, 5, 6, mp, "projected_psi_basis")
            for mode in fixture["projected_psi_basis"]]


def _selected_flux_binary(qx: SectorEventJet, qy: SectorEventJet) -> dict[str, list[dict[str, list[int]]]]:
    return {
        "selected_flux_values_binary": [normal._mpi(qx.value), normal._mpi(qy.value)],
        "selected_flux_derivatives_binary": [normal._mpi(qx.derivative), normal._mpi(qy.derivative)],
    }


def _point_samples(field: list[list[Any]], coordinates: list[list[Any]], mp: _SectorJetMath):
    h, w = len(field), len(field[0])
    result = []
    for row, column in coordinates:
        r, c = mp.context.mpf(row), mp.context.mpf(column)
        if not (_certify(r > 0) and _certify(r < h - 1)
                and _certify(c > 0) and _certify(c < w - 1)):
            raise NormalReferenceRefusal("bilinear observation point leaves the strict interior stencil")
        row_fraction, column_fraction = _fraction(row), _fraction(column)
        i = row_fraction.numerator // row_fraction.denominator
        j = column_fraction.numerator // column_fraction.denominator
        fr, fc = r - i, c - j
        result.append(field[i][j] * (1-fr) * (1-fc) + field[i][j+1] * (1-fr) * fc
                      + field[i+1][j] * fr * (1-fc) + field[i+1][j+1] * fr * fc)
    return result


def _classify_interval_failure(error: ValueError) -> NormalReferenceRefusal | None:
    message = str(error)
    if any(token in message for token in (
        "interval arithmetic could not certify", "interval arithmetic cannot decide",
        "event-side derivative sign is interval-ambiguous", "nonsmooth tie",
    )):
        return NormalReferenceRefusal(message)
    return None


def _euler_minmod(*args: Any, **kwargs: Any):
    try:
        return reference._euler_minmod(*args, **kwargs)
    except ValueError as error:
        refusal = _classify_interval_failure(error)
        if refusal is None:
            raise
        raise refusal from error


def evaluate(
    fixture: dict[str, Any], tangent: list[float | str], normal_axis: int,
    sides: tuple[int, int], dps: int = 80,
) -> dict[str, Any]:
    """Evaluate J and one normal derivative in one of the four paired sectors.

    ``normal_axis=0`` differentiates the qx[3,4] coordinate; ``1`` differentiates
    qy[3,0]. ``sides=(sx,sy)`` chooses the strict sector at both exact-zero faces.
    """
    if sides[0] not in (-1, 1) or sides[1] not in (-1, 1):
        raise ValueError("sector sides must each be -1 or +1")
    if normal_axis not in (0, 1) or len(tangent) != 24:
        raise ValueError("normal_axis must be 0/1 and tangent must have length 24")
    if type(dps) is not int or dps < 30:
        raise ValueError("dps must be an integer of at least 30")
    validate_projection(fixture)
    with reference._arithmetic(dps, True) as iv:
        mp = _SectorJetMath(iv, 1)
        z = mp.zero
        eta_x = normal._jet(iv.zero, 1, iv.one if normal_axis == 0 else iv.zero)
        eta_y = normal._jet(iv.zero, 1, iv.one if normal_axis == 1 else iv.zero)
        tangent_values = [reference._mp(value, mp) for value in tangent]
        tangent_jets = [normal._jet(value, 1) for value in tangent_values]
        limits = [reference._mp(value, mp) for value in fixture["coefficient_limits"]]
        wx = [reference._mp(value, mp) for value in _W_X]
        wy = [reference._mp(value, mp) for value in _W_Y]
        alpha = {k: limits[k] * mp.tanh(tangent_jets[20 + k - 2]) for k in (2, 3, 4)}
        alpha0 = (reference._mp(fixture["normal_scales"][0], mp) * eta_x
                  - mp.fsum(wx[k] * alpha[k] for k in (1, 2, 3, 4) if k in alpha)) / wx[0]
        alpha1 = (reference._mp(fixture["normal_scales"][1], mp) * eta_y
                  - mp.fsum(wy[k] * alpha[k] for k in (2, 3, 4))) / wy[1]
        ratio0, ratio1 = alpha0 / limits[0], alpha1 / limits[1]
        if not (_certify(ratio0 > -1) and _certify(ratio0 < 1)
                and _certify(ratio1 > -1) and _certify(ratio1 < 1)):
            raise NormalReferenceRefusal("paired lift leaves an original strict coefficient domain")
        pivot0, pivot1 = mp.atanh(ratio0), mp.atanh(ratio1)
        original_control = (tangent_jets[:20] + [pivot0, pivot1]
                            + tangent_jets[20:23] + [tangent_jets[23]])

        height, width = 4, 5
        background = reference._grid(fixture["background_dbz"], height, width, mp, "background_dbz")
        floor_dbz = reference._mp(fixture["floor_dbz"], mp)
        transform_scale = reference._mp(fixture["transform_scale"], mp)
        epsilon = reference._mp(fixture["transform_epsilon"], mp)
        increment_ratio = reference._mp(fixture["increment_ratio"], mp)
        echo_floor = reference._mp(fixture["echo_floor"], mp)
        echo_factor = reference._mp(fixture["echo_exponent_factor"], mp)
        dbz_factor = reference._mp(fixture["dbz_log_factor"], mp)
        echo = []
        for i in range(height):
            row = []
            for j in range(width):
                offset = (background[i][j] - floor_dbz) / transform_scale
                if not _certify(offset >= epsilon):
                    offset = epsilon
                latent = offset + mp.log(-mp.expm1(-offset))
                try:
                    softplus = reference._softplus(
                        latent + increment_ratio * tangent_jets[i * width + j], mp)
                except ValueError as error:
                    refusal = _classify_interval_failure(error)
                    if refusal is None:
                        raise
                    raise refusal from error
                analyzed = floor_dbz + transform_scale * softplus
                row.append(echo_floor * mp.expm1(echo_factor * (analyzed - floor_dbz)))
            echo.append(row)

        basis = _paired_projected_basis(fixture, mp)
        original_basis = [reference._grid(mode, 5, 6, mp, "psi_basis") for mode in fixture["psi_basis"]]
        scale_x = reference._mp(fixture["normal_scales"][0], mp)
        scale_y = reference._mp(fixture["normal_scales"][1], mp)
        psi = [[mp.fsum(alpha[k + 2] * basis[k][i][j] for k in range(3))
                + scale_x * eta_x * original_basis[0][i][j] / wx[0]
                + scale_y * eta_y * original_basis[1][i][j] / wy[1]
                for j in range(width + 1)] for i in range(height + 1)]
        qx = [[psi[i + 1][j] - psi[i][j] for j in range(width + 1)] for i in range(height)]
        qy = [[-(psi[i][j + 1] - psi[i][j]) for j in range(width)] for i in range(height + 1)]
        qx[3][4] = SectorEventJet((scale_x * eta_x).value,
                                  (scale_x * eta_x).derivative, 1, True, sides[0])
        qy[3][0] = SectorEventJet((scale_y * eta_y).value,
                                  (scale_y * eta_y).derivative, 1, True, sides[1])

        growth_limit = reference._mp(fixture["growth_limit"], mp)
        growth = growth_limit * mp.tanh(tangent_jets[23])
        if not _certify(growth > 0):
            raise NormalReferenceRefusal("normal reference requires the fixed positive-growth branch")
        substeps = fixture["substeps"]
        if type(substeps) is not int or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        growth_factor = mp.exp(growth / substeps)
        dt, area = reference._mp(fixture["dt"], mp), reference._mp(fixture["area"], mp)
        boundaries = fixture["boundary_echo"]
        if not isinstance(boundaries, (list, tuple)) or len(boundaries) != 2 * substeps:
            raise ValueError("boundary_echo must cover two intervals of fixed substeps")
        if not (_certify(dt > 0) and _certify(area > 0)):
            raise ValueError("time step and cell area must be positive")
        frames = [echo]
        trace: list[dict[str, Any]] = []
        for step, step_edges in enumerate(boundaries):
            if len(step_edges) != 2 or any(len(stage) != 4 for stage in step_edges):
                raise ValueError("boundary_echo entries must have [stage][left,right,bottom,top]")
            edges = [[[reference._mp(value, mp) for value in edge] for edge in stage]
                     for stage in step_edges]
            if any(len(edges[stage][edge]) != (height if edge < 2 else width)
                   for stage in range(2) for edge in range(4)):
                raise ValueError("boundary edge lengths do not match the fixed 4x5 profile")
            grown = [[value * growth_factor for value in row] for row in echo]
            grown_edges = [[value * growth_factor for value in edge] for edge in edges[0]]
            stage1 = _euler_minmod(grown, qx, qy, grown_edges, dt, area,
                                   step, 0, trace, mp)
            stage2 = _euler_minmod(stage1, qx, qy, edges[1], dt, area,
                                   step, 1, trace, mp)
            echo = [[(grown[i][j] + stage2[i][j]) / 2 for j in range(width)]
                    for i in range(height)]
            if (step + 1) % substeps == 0:
                frames.append(echo)
        predicted_dbz = [[floor_dbz + dbz_factor * mp.log1p(value / echo_floor)
                          for row in frame for value in row] for frame in frames]

        coordinates = _matrix(fixture["coordinates"], 4, 2, "coordinates")
        observations = _matrix(fixture["point_dbz"], 3, 4, "point_dbz")
        std = _matrix(fixture["std_dbz"], 3, 4, "std_dbz")
        quality = _matrix(fixture["quality_weight"], 3, 4, "quality_weight")
        whitener = _matrix(fixture["whitener"], 4, 4, "whitener")
        if any(_fraction(whitener[i][j]) != _fraction(whitener[j][i])
               for i in range(4) for j in range(4)):
            raise ValueError("captured native point whitener must be exactly symmetric")
        obs_mp = [[reference._mp(value, mp) for value in row] for row in observations]
        std_mp = [[reference._mp(value, mp) for value in row] for row in std]
        quality_mp = [[reference._mp(value, mp) for value in row] for row in quality]
        if any(not _certify(value > 0) for row in std_mp for value in row):
            raise ValueError("point observation standard deviations must be positive")
        if any(not _certify(value > 0) for row in quality_mp for value in row):
            raise ValueError("point quality weights must be positive")
        if any(not _certify(value <= 1) for row in quality_mp for value in row):
            raise ValueError("point quality weights must not exceed one")
        delta = reference._mp(fixture["robust_delta"], mp)
        if not _certify(delta > 0):
            raise ValueError("robust_delta must be positive")
        residuals = []
        for t in range(3):
            samples = _point_samples([predicted_dbz[t][i * width:(i + 1) * width]
                                      for i in range(height)], coordinates, mp)
            residuals.append([mp.sqrt(quality_mp[t][k]) * (samples[k] - obs_mp[t][k]) / std_mp[t][k]
                              for k in range(4)])
        white = [[mp.fsum(reference._mp(whitener[i][j], mp) * residuals[t][j]
                          for j in range(4)) for i in range(4)] for t in range(3)]
        robust = mp.fsum(value**2 / (mp.sqrt(1 + (value / delta)**2) + 1)
                         for row in white for value in row)
        prior = mp.fsum(value**2 for value in original_control) / 2
        smooth_weight = reference._mp(fixture["smooth_weight"], mp)
        left, right = fixture["smooth_left_index"], fixture["smooth_right_index"]
        smooth_weights = fixture["smooth_physical_weight"]
        if not (len(left) == len(right) == len(smooth_weights)):
            raise ValueError("field-prior edge arrays must have equal lengths")
        smooth = z
        for i, j, weight in zip(left, right, smooth_weights, strict=True):
            if type(i) is not int or type(j) is not int or not (0 <= i < 20 and 0 <= j < 20):
                raise ValueError("smoothness edges must index the 20 field controls")
            difference = original_control[j] - original_control[i]
            smooth += smooth_weight * reference._mp(weight, mp) * difference**2 / 2
        objective = robust + prior + smooth
        return {
            "normal_axis": normal_axis, "sector_sides": list(sides),
            "objective_interval": mp.nstr(objective.value, n=dps),
            "sigma_eta_interval": mp.nstr(objective.derivative, n=dps),
            "objective_mpi": normal._mpi(objective.value),
            "sigma_eta_mpi": normal._mpi(objective.derivative),
            "J_binary": normal._mpi(objective.value),
            "sigma_eta_binary": normal._mpi(objective.derivative),
            "objective_bounds": normal._bounds(objective.value, dps),
            "sigma_eta_bounds": normal._bounds(objective.derivative, dps),
            "interval_bound_representation": "mpmath.iv._mpi exact binary endpoint tuples are authoritative",
            "selected_flux_values": [mp.nstr(qx[3][4].value, n=dps), mp.nstr(qy[3][0].value, n=dps)],
            "selected_flux_derivatives": [mp.nstr(qx[3][4].derivative, n=dps),
                                          mp.nstr(qy[3][0].derivative, n=dps)],
            **_selected_flux_binary(qx[3][4], qy[3][0]),
            "robust_cost_interval": mp.nstr(robust.value, n=dps),
            "prior_cost_interval": mp.nstr(prior.value, n=dps),
            "smooth_cost_interval": mp.nstr(smooth.value, n=dps),
            "robust_cost_bounds": normal._bounds(robust.value, dps),
            "prior_cost_bounds": normal._bounds(prior.value, dps),
            "smooth_cost_bounds": normal._bounds(smooth.value, dps),
            "branch_choices": trace,
            "scope": "directional interval derivative at one fixed paired-zero point; no optimizer or normal-minimum claim",
        }
