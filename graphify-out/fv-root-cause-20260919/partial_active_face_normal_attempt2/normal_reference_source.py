"""First-order interval reference for the two one-sided derivatives at qy[2,0]=0.

This uses the independent fixed-profile Euler primitive and interval forward
jets. It evaluates one fixed 4x5 point; it is not a normal certificate or an
optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from examples.weather_scenarios import fv_slice_precision_reference as reference


@dataclass(frozen=True, eq=False)
class IntervalJet:
    """An interval value and its interval derivative in the increasing-eta direction."""

    value: Any
    derivative: Any
    side: int
    event: bool = False

    def _coerce(self, other: Any) -> IntervalJet:
        if isinstance(other, IntervalJet):
            if other.side != self.side:
                raise ValueError("jet sides differ")
            return other
        return IntervalJet(self.value.ctx.mpf(other), self.value.ctx.zero, self.side)

    def __add__(self, other: Any) -> IntervalJet:
        right = self._coerce(other)
        return IntervalJet(self.value + right.value, self.derivative + right.derivative,
                           self.side)

    def __radd__(self, other: Any) -> IntervalJet:
        return self + other

    def __neg__(self) -> IntervalJet:
        return IntervalJet(-self.value, -self.derivative, self.side)

    def __sub__(self, other: Any) -> IntervalJet:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Any) -> IntervalJet:
        return self._coerce(other) - self

    def __mul__(self, other: Any) -> IntervalJet:
        right = self._coerce(other)
        return IntervalJet(self.value * right.value,
                           self.derivative * right.value + self.value * right.derivative,
                           self.side)

    def __rmul__(self, other: Any) -> IntervalJet:
        return self * other

    def __truediv__(self, other: Any) -> IntervalJet:
        right = self._coerce(other)
        return IntervalJet(self.value / right.value,
                           (self.derivative * right.value - self.value * right.derivative)
                           / (right.value * right.value),
                           self.side)

    def __rtruediv__(self, other: Any) -> IntervalJet:
        return self._coerce(other) / self

    def __pow__(self, exponent: int) -> IntervalJet:
        if type(exponent) is not int:
            raise TypeError("interval jets support integer powers only")
        if exponent == 0:
            return IntervalJet(self.value.ctx.one, self.value.ctx.zero, self.side)
        return IntervalJet(self.value**exponent,
                           exponent * self.value ** (exponent - 1) * self.derivative,
                           self.side)

    def __eq__(self, other: object) -> bool:
        right = self._coerce(other)
        equal = self.value == right.value
        if equal is None:
            raise ValueError("interval equality is undecidable")
        return bool(equal)

    def _order(self, other: Any) -> int:
        right = self._coerce(other)
        greater = self.value > right.value
        if greater is True:
            return 1
        less = self.value < right.value
        if less is True:
            return -1
        equal = self.value == right.value
        if equal is True:
            if self.event or right.event:
                if not (_is_exact_zero(self.value) and _is_exact_zero(right.value)):
                    raise ValueError("event-side comparison is only valid at an exact zero point")
                directed = self.side * (self.derivative - right.derivative)
                if (directed > 0) is True:
                    return 1
                if (directed < 0) is True:
                    return -1
                if (directed == 0) is True:
                    return 0
                raise ValueError("event-side derivative sign is interval-ambiguous")
            raise ValueError("nonsmooth tie is not the tagged selected face event")
        raise ValueError("interval arithmetic cannot decide a branch comparison")

    def __lt__(self, other: Any) -> bool:
        return self._order(other) < 0

    def __le__(self, other: Any) -> bool:
        return self._order(other) <= 0

    def __gt__(self, other: Any) -> bool:
        return self._order(other) > 0

    def __ge__(self, other: Any) -> bool:
        return self._order(other) >= 0

    def __abs__(self) -> IntervalJet:
        order = self._order(self.value.ctx.zero)
        if order > 0:
            return self
        if order < 0:
            return -self
        raise ValueError("absolute value is nonsmooth at zero")


class _JetMath:
    """Small mpmath.iv adapter used by the existing scalar Euler primitive."""

    def __init__(self, context: Any, side: int) -> None:
        self.context = context
        self.side = side
        self._zero = IntervalJet(context.zero, context.zero, side)

    @property
    def zero(self) -> IntervalJet:
        return self._zero

    def mpf(self, value: object) -> Any:
        return _jet(self.context.mpf(value), self.side)

    def fsum(self, values: Any) -> Any:
        entries = list(values)
        if not any(isinstance(value, IntervalJet) for value in entries):
            return self.context.fsum(entries)
        result: Any = self.zero
        for value in entries:
            result = result + value
        return result

    def _unary(self, function: Any, derivative: Any, value: Any) -> Any:
        if not isinstance(value, IntervalJet):
            return function(value)
        primal = function(value.value)
        return IntervalJet(primal, derivative(value.value, primal) * value.derivative,
                           value.side)

    def exp(self, value: Any) -> Any:
        return self._unary(self.context.exp, lambda x, y: y, value)

    def expm1(self, value: Any) -> Any:
        return self._unary(self.context.expm1, lambda x, y: self.context.exp(x), value)

    def log(self, value: Any) -> Any:
        return self._unary(self.context.log, lambda x, y: 1 / x, value)

    def log1p(self, value: Any) -> Any:
        return self._unary(self.context.log1p, lambda x, y: 1 / (1 + x), value)

    def sqrt(self, value: Any) -> Any:
        return self._unary(self.context.sqrt, lambda x, y: 1 / (2 * y), value)

    def tanh(self, value: Any) -> Any:
        def primal(x: Any) -> Any:
            numerator = self.context.expm1(2 * x)
            return numerator / (numerator + 2)
        return self._unary(primal, lambda x, y: 1 - y * y, value)

    def atanh(self, value: Any) -> Any:
        if not isinstance(value, IntervalJet):
            return (self.context.log1p(value) - self.context.log1p(-value)) / 2
        primal = (self.context.log1p(value.value) - self.context.log1p(-value.value)) / 2
        derivative = value.derivative / (1 - value.value * value.value)
        return IntervalJet(primal, derivative, value.side)

    def nstr(self, value: Any, n: int) -> str:
        return self.context.nstr(value.value if isinstance(value, IntervalJet) else value, n=n)


def _jet(value: Any, side: int, derivative: Any = None, *, event: bool = False) -> IntervalJet:
    if isinstance(value, IntervalJet):
        if value.side != side:
            raise ValueError("jet sides differ")
        if derivative is None and not event:
            return value
        return IntervalJet(value.value, value.derivative if derivative is None else derivative,
                           side, value.event or event)
    if derivative is None:
        derivative = value.ctx.zero
    return IntervalJet(value, derivative, side, event)


def _validate_projection(fixture: dict[str, Any]) -> tuple[list[list[list[Any]]], list[list[list[Any]]], list[Any], Any]:
    basis_raw = fixture["psi_basis"]
    projected_raw = fixture["projected_psi_basis"]
    weights_raw = fixture["face_weights"]
    if len(basis_raw) != 5 or len(projected_raw) != 4 or len(weights_raw) != 5:
        raise ValueError("normal reference requires five basis modes, four projected modes, and five weights")
    if any(len(mode) != 5 or any(len(row) != 6 for row in mode) for mode in basis_raw + projected_raw):
        raise ValueError("streamfunction bases must have fixed 5x6 shape")
    pivot = 1
    keep = (0, 2, 3, 4)
    limits = fixture["coefficient_limits"]
    weights_float = [float(value) for value in weights_raw]
    pivot_weight = weights_float[pivot]
    if pivot_weight == 0:
        raise ValueError("selected face pivot weight must be nonzero")
    projected_float = [[[float(value) for value in row] for row in mode] for mode in projected_raw]
    basis_float = [[[float(value) for value in row] for row in mode] for mode in basis_raw]
    fraction = lambda value: Fraction.from_float(float(value))
    for mode_index, original_index in enumerate(keep):
        for row in range(5):
            for column in range(6):
                expected = (fraction(basis_float[original_index][row][column])
                            - fraction(weights_float[original_index]) / fraction(pivot_weight)
                            * fraction(basis_float[pivot][row][column]))
                actual = fraction(projected_float[mode_index][row][column])
                if actual != expected:
                    raise ValueError("captured projected basis differs from the pinned projection formula")
        if projected_float[mode_index][2][0] != projected_float[mode_index][2][1]:
            raise ValueError("projected mode does not give an exact selected-face structural zero")
    for actual, mode in zip(weights_float, basis_float, strict=True):
        expected = fraction(mode[2][0]) - fraction(mode[2][1])
        if fraction(actual) != expected:
            raise ValueError("captured face weights differ from q_y[2,0] basis flux")
    flow_limits = [float(value) for value in limits]
    face_scale = float(fixture["face_scale"])
    if fraction(face_scale) != abs(fraction(pivot_weight)) * fraction(flow_limits[pivot]):
        raise ValueError("captured face scale differs from abs(w_p)*L_p")
    return basis_float, projected_float, weights_raw, fixture["face_scale"]


def evaluate_side(fixture: dict[str, Any], tangent: list[float | str], side: int,
                  dps: int = 80) -> dict[str, Any]:
    """Return an interval for J and dJ/deta on one strict side at eta=0."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    if type(dps) is not int or dps < 30:
        raise ValueError("dps must be an integer of at least 30")
    if len(tangent) != 25:
        raise ValueError("tangent must contain the 25 fixed nonpivot coordinates")
    if (fixture.get("per_frame") is not True
            or fixture.get("tile_size", fixture.get("tile_size_px", 0)) != 0
            or fixture.get("common_bias_group_index") is not None
            or fixture.get("neural_prior_std_dbz", fixture.get("neural_prior_std")) is not None):
        raise ValueError("reference requires the pinned per-frame, ungrouped, no-neural-prior contract")
    if fixture.get("censored_mask") is not None and any(
        bool(value) for frame in fixture["censored_mask"] for row in frame for value in row
    ):
        raise ValueError("reference supports detected and missing observations, no censoring")
    basis_float, projected_float, weights_raw, scale_raw = _validate_projection(fixture)
    with reference._arithmetic(dps, True) as iv:
        mp = _JetMath(iv, side)
        zero = iv.zero
        raw_tangent = [reference._mp(value, mp) for value in tangent]
        tangent_jets = [_jet(value, side) for value in raw_tangent]
        eta = _jet(zero, side, iv.one)
        height, width = 4, 5
        flow_limits = [reference._mp(value, mp) for value in fixture["coefficient_limits"]]
        weights = [reference._mp(value, mp) for value in weights_raw]
        scale = reference._mp(scale_raw, mp)
        keep = (0, 2, 3, 4)
        alpha = {index: flow_limits[index] * mp.tanh(tangent_jets[20 + position])
                 for position, index in enumerate(keep)}
        pivot_alpha = (scale * eta - mp.fsum(weights[index] * alpha[index] for index in keep)) / weights[1]
        pivot_ratio = pivot_alpha / flow_limits[1]
        if not (reference._decide(pivot_ratio > -1) and reference._decide(pivot_ratio < 1)):
            raise ValueError("zero-face lift leaves the original strict pivot coefficient domain")
        pivot_latent = mp.atanh(pivot_ratio)
        original_control = (tangent_jets[:20] + [tangent_jets[20], pivot_latent]
                            + tangent_jets[21:24] + [tangent_jets[24]])

        background = fixture["background_dbz"]
        if len(background) != height or any(len(row) != width for row in background):
            raise ValueError("normal reference supports the fixed 4x5 field only")
        background_mp = reference._grid(background, height, width, mp, "background_dbz")
        initial_echo: list[list[Any]] = []
        floor_dbz = reference._mp(fixture["floor_dbz"], mp)
        transform_scale = reference._mp(fixture["transform_scale"], mp)
        epsilon = reference._mp(fixture["transform_epsilon"], mp)
        increment_ratio = reference._mp(fixture["increment_ratio"], mp)
        echo_floor = reference._mp(fixture["echo_floor"], mp)
        echo_factor = reference._mp(fixture["echo_exponent_factor"], mp)
        dbz_factor = reference._mp(fixture["dbz_log_factor"], mp)
        for i in range(height):
            echo_row = []
            for j in range(width):
                offset = (background_mp[i][j] - floor_dbz) / transform_scale
                if not reference._decide(offset >= epsilon):
                    offset = epsilon
                latent = offset + mp.log(-mp.expm1(-offset))
                analyzed = floor_dbz + transform_scale * reference._softplus(
                    latent + increment_ratio * tangent_jets[i * width + j], mp,
                )
                echo_row.append(echo_floor * mp.expm1(echo_factor * (analyzed - floor_dbz)))
            initial_echo.append(echo_row)

        basis = [reference._grid(mode, 5, 6, mp, "psi_basis") for mode in basis_float]
        projected = [reference._grid(mode, 5, 6, mp, "projected_psi_basis") for mode in projected_float]
        pivot_term = scale * eta / weights[1]
        psi = [[mp.fsum(alpha[index] * projected[position][i][j]
                        for position, index in enumerate(keep))
                + pivot_term * basis[1][i][j]
                for j in range(width + 1)] for i in range(height + 1)]
        qx = [[psi[i + 1][j] - psi[i][j] for j in range(width + 1)]
              for i in range(height)]
        qy = [[-(psi[i][j + 1] - psi[i][j]) for j in range(width)]
              for i in range(height + 1)]
        # The captured projected vertices are exactly equal, so their four
        # terms have zero selected-face flux. Preserve the remaining identity
        # q_y[2,0] = face_scale * eta instead of widening 0 by interval dependency.
        selected = scale * eta
        qy[2][0] = IntervalJet(selected.value, selected.derivative, side, event=True)

        growth_limit = reference._mp(fixture["growth_limit"], mp)
        interval_growth = growth_limit * mp.tanh(tangent_jets[-1])
        if not reference._decide(interval_growth > 0):
            raise ValueError("normal reference supports the pinned positive-growth branch only")
        substeps = fixture["substeps"]
        if type(substeps) is not int or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        growth_factor = mp.exp(interval_growth / substeps)
        dt = reference._mp(fixture["dt"], mp)
        area = reference._mp(fixture["area"], mp)
        boundaries = fixture["boundary_echo"]
        if len(boundaries) != 2 * substeps:
            raise ValueError("boundary_echo must cover two intervals of substeps")
        edge_steps = []
        for step_edges in boundaries:
            if len(step_edges) != 2 or any(len(stage) != 4 for stage in step_edges):
                raise ValueError("boundary_echo entries must be [step][stage][left,right,bottom,top]")
            converted = [[ [reference._mp(value, mp) for value in edge] for edge in stage ]
                         for stage in step_edges]
            if any(len(converted[stage][edge]) != (height if edge < 2 else width)
                   for stage in range(2) for edge in range(4)):
                raise ValueError("boundary edge lengths do not match the fixed field")
            edge_steps.append(converted)
        frames = [initial_echo]
        trace: list[dict[str, Any]] = []
        echo = initial_echo
        for step, step_edges in enumerate(edge_steps):
            grown = [[value * growth_factor for value in row] for row in echo]
            grown_edges = [[value * growth_factor for value in edge] for edge in step_edges[0]]
            stage1 = reference._euler_minmod(grown, qx, qy, grown_edges, dt, area,
                                            step, 0, trace, mp)
            stage2 = reference._euler_minmod(stage1, qx, qy, step_edges[1], dt, area,
                                            step, 1, trace, mp)
            echo = [[(grown[i][j] + stage2[i][j]) / 2 for j in range(width)]
                    for i in range(height)]
            if (step + 1) % substeps == 0:
                frames.append(echo)
        predicted_dbz = [[[
            floor_dbz + dbz_factor * mp.log1p(value / echo_floor)
            for value in row
        ] for row in frame] for frame in frames]

        observations = reference._cube(fixture["observation_dbz"], 3, height, width, mp, "observation_dbz")
        std = reference._cube(fixture["std_dbz"], 3, height, width, mp, "std_dbz")
        quality = reference._cube(fixture["quality_weight"], 3, height, width, mp, "quality_weight")
        mode = reference._cube(fixture["whitener_mode"], 3, height, width, mp, "whitener_mode")
        valid = fixture["valid_mask"]
        if (len(valid) != 3 or any(len(frame) != height or any(len(row) != width for row in frame)
                                   for frame in valid)
                or sum(sum(sum(bool(value) for value in row) for row in frame) for frame in valid) != 58):
            raise ValueError("normal reference requires 58 detected cells and two missing cells")
        valid_mask = [[[bool(value) for value in row] for row in frame] for frame in valid]
        missing = [(t, i, j) for t in range(3) for i in range(height) for j in range(width)
                   if not valid_mask[t][i][j]]
        if missing != [(1, 1, 2), (2, 2, 3)]:
            raise ValueError("normal reference requires the two pinned missing-observation cells")
        if "detected_mask" in fixture and fixture["detected_mask"] != valid:
            raise ValueError("reference requires every valid observation to be detected")
        standardized = [[[zero for _ in range(width)] for _ in range(height)] for _ in range(3)]
        for t in range(3):
            for i in range(height):
                for j in range(width):
                    if valid_mask[t][i][j]:
                        standardized[t][i][j] = (mp.sqrt(quality[t][i][j])
                                                  * (predicted_dbz[t][i][j] - observations[t][i][j])
                                                  / std[t][i][j])
        bias = reference._mp(fixture["bias_std"], mp)
        whitened = [[[zero for _ in range(width)] for _ in range(height)] for _ in range(3)]
        for t in range(3):
            norm2 = mp.fsum(mode[t][i][j] ** 2 for i in range(height) for j in range(width))
            projection_value = mp.fsum(mode[t][i][j] * standardized[t][i][j]
                                       for i in range(height) for j in range(width))
            coefficient = (zero if norm2 == 0 or bias == 0 else
                           (1 - 1 / mp.sqrt(1 + bias**2 * norm2)) / norm2)
            for i in range(height):
                for j in range(width):
                    whitened[t][i][j] = standardized[t][i][j] - coefficient * mode[t][i][j] * projection_value

        delta = reference._mp(fixture["robust_delta"], mp)
        robust_cost = mp.fsum(
            whitened[t][i][j] ** 2 / (mp.sqrt(1 + (whitened[t][i][j] / delta) ** 2) + 1)
            for t in range(3) for i in range(height) for j in range(width) if valid_mask[t][i][j]
        )
        prior_cost = mp.fsum(value**2 for value in original_control) / 2
        smooth_cost = zero
        smooth_weight = reference._mp(fixture["smooth_weight"], mp)
        for left, right, weight in zip(fixture["smooth_left_index"], fixture["smooth_right_index"],
                                       fixture["smooth_physical_weight"], strict=True):
            difference = original_control[right] - original_control[left]
            smooth_cost += smooth_weight * reference._mp(weight, mp) * difference**2 / 2
        objective = robust_cost + prior_cost + smooth_cost
        selected_flux = qy[2][0]
        objective_bounds = _bounds(objective.value, dps)
        sigma_bounds = _bounds(objective.derivative, dps)
        return {
            "side": side,
            "objective_interval": mp.nstr(objective.value, n=dps),
            "sigma_eta_interval": mp.nstr(objective.derivative, n=dps),
            "objective_mpi": _mpi(objective.value),
            "sigma_eta_mpi": _mpi(objective.derivative),
            "sigma_eta_interval_binary": [list(bound) for bound in objective.derivative._mpi_],
            "objective_interval_binary": [list(bound) for bound in objective.value._mpi_],
            "objective_bounds": objective_bounds,
            "sigma_eta_bounds": sigma_bounds,
            "interval_bound_representation": "mpmath.iv._mpi exact binary endpoint tuples are authoritative; decimal bounds are presentation only",
            "face_flux_interval": mp.nstr(selected_flux.value, n=dps),
            "face_flux_derivative_interval": mp.nstr(selected_flux.derivative, n=dps),
            "robust_cost_interval": mp.nstr(robust_cost.value, n=dps),
            "prior_cost_interval": mp.nstr(prior_cost.value, n=dps),
            "smooth_cost_interval": mp.nstr(smooth_cost.value, n=dps),
            "branch_choices": trace,
            "scope": "first derivative at the approximate fixed tangent point; one-sided branch extension only; no normal certificate",
        }


def _bounds(value: Any, dps: int) -> list[str]:
    from mpmath import mp
    lower, upper = value._mpi_
    with mp.workdps(dps):
        lower_value, upper_value = mp.make_mpf(lower), mp.make_mpf(upper)
        if not mp.isfinite(lower_value) or not mp.isfinite(upper_value):
            raise ValueError("interval result is not finite")
        return [str(mp.nstr(lower_value, n=dps)), str(mp.nstr(upper_value, n=dps))]


def _mpi(value: Any) -> dict[str, list[int]]:
    lower, upper = value._mpi_
    return {"lower": [int(component) for component in lower],
            "upper": [int(component) for component in upper]}


def _is_exact_zero(value: Any) -> bool:
    return value._mpi_ == ((0, 0, 0, 0), (0, 0, 0, 0))
