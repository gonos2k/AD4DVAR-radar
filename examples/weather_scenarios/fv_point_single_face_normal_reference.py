"""Independent one-sided interval derivative on q_x[3,4]=0.

The fixture uses the paired-reference point-cost schema, except
``normal_weights`` is one five-entry qx row, ``normal_scale`` is scalar, and
``projected_psi_basis`` contains four qx-projected modes. This module imports
only independent arithmetic helpers, not Torch or production model code.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from examples.weather_scenarios import fv_point_paired_face_normal_reference as paired
from examples.weather_scenarios import fv_slice_precision_reference as reference

SingleFaceNormalRefusal = paired.NormalReferenceRefusal
_WEIGHTS = (Fraction(1), Fraction(0), Fraction(4), Fraction(-7, 2), Fraction(16))


def validate_projection(fixture: dict[str, Any]) -> None:
    basis = paired._basis(fixture["psi_basis"], 5, "psi_basis")
    projected = paired._basis(fixture["projected_psi_basis"], 4, "projected_psi_basis")
    weights_raw = fixture["normal_weights"]
    if not isinstance(weights_raw, (list, tuple)) or len(weights_raw) != 5:
        raise ValueError("normal_weights must be one five-entry qx row")
    weights = tuple(map(paired._fraction, weights_raw))
    if weights != _WEIGHTS:
        raise ValueError("captured qx weights differ from the pinned row")
    b = [[[paired._fraction(value) for value in row] for row in mode] for mode in basis]
    p = [[[paired._fraction(value) for value in row] for row in mode] for mode in projected]
    for k, mode in enumerate(b):
        if mode[4][4] - mode[3][4] != weights[k]:
            raise ValueError("psi basis q_x[3,4] weights differ from the pinned row")
    for i, k in enumerate((1, 2, 3, 4)):
        for row in range(5):
            for column in range(6):
                expected = b[k][row][column] - weights[k] / weights[0] * b[0][row][column]
                if p[i][row][column] != expected:
                    raise ValueError("projected basis differs from exact qx-face formula")
        if p[i][4][4] != p[i][3][4]:
            raise ValueError("projected mode does not structurally zero q_x[3,4]")
    limits = fixture["coefficient_limits"]
    if not isinstance(limits, (list, tuple)) or len(limits) != 5:
        raise ValueError("coefficient_limits must have length five")
    scale = paired._fraction(fixture["normal_scale"])
    if (scale != abs(weights[0] * paired._fraction(limits[0]))
            or scale != paired._fraction(0.11)):
        raise ValueError("normal_scale must match the original pivot and equal .11")


def _lift(tangent: list[Any], eta: Any, fixture: dict[str, Any], mp: Any):
    limits = [reference._mp(value, mp) for value in fixture["coefficient_limits"]]
    weights = [reference._mp(float(value), mp) for value in _WEIGHTS]
    scale = reference._mp(fixture["normal_scale"], mp)
    flow_latents = tangent[20:24]
    alpha = [limits[k] * mp.tanh(flow_latents[k - 1]) for k in (1, 2, 3, 4)]
    alpha0 = (scale * eta - mp.fsum(weights[k] * alpha[k - 1] for k in (1, 2, 3, 4))) / weights[0]
    ratio = alpha0 / limits[0]
    if not (paired._certify(ratio > -1) and paired._certify(ratio < 1)):
        raise SingleFaceNormalRefusal("qx-face lift leaves original pivot coefficient domain")
    pivot = mp.atanh(ratio)
    full_control = tangent[:20] + [pivot] + flow_latents + [tangent[24]]
    return full_control, alpha, alpha0


def strict_normal_orientation(left: tuple[Fraction, Fraction],
                              right: tuple[Fraction, Fraction]) -> bool:
    """A strict cusp minimum needs a negative left slope and positive right slope."""
    return left[1] < 0 < right[0]


def evaluate(fixture: dict[str, Any], tangent: list[float | str], side: int,
             dps: int = 80) -> dict[str, Any]:
    """Evaluate J and dJ/deta on one strict side of the selected qx face."""
    if side not in (-1, 1) or len(tangent) != 25:
        raise ValueError("side must be +/-1 and tangent must have length 25")
    if type(dps) is not int or dps < 30:
        raise ValueError("dps must be an integer of at least 30")
    validate_projection(fixture)
    with reference._arithmetic(dps, True) as iv:
        mp = paired._SectorJetMath(iv, 1)
        tangent_values = [reference._mp(value, mp) for value in tangent]
        tangent_jets = [paired.normal._jet(value, 1) for value in tangent_values]
        eta = paired.normal._jet(iv.zero, 1, iv.one)
        full_control, alpha, _ = _lift(tangent_jets, eta, fixture, mp)
        basis = [reference._grid(mode, 5, 6, mp, "projected_psi_basis")
                 for mode in fixture["projected_psi_basis"]]
        original = [reference._grid(mode, 5, 6, mp, "psi_basis")
                    for mode in fixture["psi_basis"]]
        # These fixed dyadic weights are exactly representable as binary64.
        weights = [reference._mp(float(value), mp) for value in _WEIGHTS]
        scale = reference._mp(fixture["normal_scale"], mp)
        psi = [[mp.fsum(alpha[k] * basis[k][i][j] for k in range(4))
                + scale * eta * original[0][i][j] / weights[0]
                for j in range(6)] for i in range(5)]
        qx = [[psi[i + 1][j] - psi[i][j] for j in range(6)] for i in range(4)]
        qy = [[-(psi[i][j + 1] - psi[i][j]) for j in range(5)] for i in range(5)]
        qx[3][4] = paired.SectorEventJet((scale * eta).value,
                                         (scale * eta).derivative, 1, True, side)

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
                if not paired._certify(offset >= epsilon):
                    offset = epsilon
                latent = offset + mp.log(-mp.expm1(-offset))
                try:
                    softplus = reference._softplus(
                        latent + increment_ratio * tangent_jets[i * width + j], mp)
                except ValueError as error:
                    refusal = paired._classify_interval_failure(error)
                    if refusal is None:
                        raise
                    raise refusal from error
                analyzed = floor_dbz + transform_scale * softplus
                row.append(echo_floor * mp.expm1(echo_factor * (analyzed - floor_dbz)))
            echo.append(row)

        growth = reference._mp(fixture["growth_limit"], mp) * mp.tanh(tangent_jets[24])
        if not paired._certify(growth > 0):
            raise SingleFaceNormalRefusal("single-face reference requires positive growth")
        substeps = fixture["substeps"]
        if type(substeps) is not int or substeps < 1:
            raise ValueError("substeps must be positive")
        growth_factor = mp.exp(growth / substeps)
        dt, area = reference._mp(fixture["dt"], mp), reference._mp(fixture["area"], mp)
        if not (paired._certify(dt > 0) and paired._certify(area > 0)):
            raise ValueError("dt and area must be positive")
        boundaries = fixture["boundary_echo"]
        if not isinstance(boundaries, (list, tuple)) or len(boundaries) != 2 * substeps:
            raise ValueError("boundary_echo must cover two intervals")
        frames, trace = [echo], []
        for step, stages in enumerate(boundaries):
            if len(stages) != 2 or any(len(stage) != 4 for stage in stages):
                raise ValueError("boundary entries require two stages and four edges")
            edges = [[[reference._mp(value, mp) for value in edge] for edge in stage]
                     for stage in stages]
            if any(len(edges[s][e]) != (height if e < 2 else width)
                   for s in range(2) for e in range(4)):
                raise ValueError("boundary edge length differs from the fixed 4x5 grid")
            grown = [[value * growth_factor for value in row] for row in echo]
            grown_edges = [[value * growth_factor for value in edge] for edge in edges[0]]
            stage1 = paired._euler_minmod(grown, qx, qy, grown_edges, dt, area,
                                          step, 0, trace, mp)
            stage2 = paired._euler_minmod(stage1, qx, qy, edges[1], dt, area,
                                          step, 1, trace, mp)
            echo = [[(grown[i][j] + stage2[i][j]) / 2 for j in range(width)]
                    for i in range(height)]
            if (step + 1) % substeps == 0:
                frames.append(echo)
        predicted = [[floor_dbz + dbz_factor * mp.log1p(value / echo_floor)
                      for row in frame for value in row] for frame in frames]

        coordinates = paired._matrix(fixture["coordinates"], 4, 2, "coordinates")
        points = paired._matrix(fixture["point_dbz"], 3, 4, "point_dbz")
        std_raw = paired._matrix(fixture["std_dbz"], 3, 4, "std_dbz")
        quality_raw = paired._matrix(fixture["quality_weight"], 3, 4, "quality_weight")
        whitener = paired._matrix(fixture["whitener"], 4, 4, "whitener")
        if any(paired._fraction(whitener[i][j]) != paired._fraction(whitener[j][i])
               for i in range(4) for j in range(4)):
            raise ValueError("captured point whitener must be symmetric")
        observations = [[reference._mp(v, mp) for v in row] for row in points]
        std = [[reference._mp(v, mp) for v in row] for row in std_raw]
        quality = [[reference._mp(v, mp) for v in row] for row in quality_raw]
        paired._validate_point_weights(std, quality, mp)
        residuals = []
        for t in range(3):
            field = [predicted[t][i * width:(i + 1) * width] for i in range(height)]
            samples = paired._point_samples(field, coordinates, mp)
            residuals.append([mp.sqrt(quality[t][k]) * (samples[k] - observations[t][k]) / std[t][k]
                              for k in range(4)])
        white = [[mp.fsum(reference._mp(whitener[i][j], mp) * residuals[t][j]
                          for j in range(4)) for i in range(4)] for t in range(3)]
        delta = reference._mp(fixture["robust_delta"], mp)
        if not paired._certify(delta > 0):
            raise ValueError("robust_delta must be positive")
        robust = mp.fsum(v**2 / (mp.sqrt(1 + (v / delta)**2) + 1)
                         for row in white for v in row)
        prior = mp.fsum(v**2 for v in full_control) / 2
        smooth_weight = reference._mp(fixture["smooth_weight"], mp)
        left, right, smooth_weights = (fixture[k] for k in
                                       ("smooth_left_index", "smooth_right_index", "smooth_physical_weight"))
        if not (len(left) == len(right) == len(smooth_weights)):
            raise ValueError("field-prior edge arrays must have equal lengths")
        smooth = mp.zero
        for i, j, w in zip(left, right, smooth_weights, strict=True):
            if type(i) is not int or type(j) is not int or not (0 <= i < 20 and 0 <= j < 20):
                raise ValueError("smoothness edges must index field controls")
            d = full_control[j] - full_control[i]
            smooth += smooth_weight * reference._mp(w, mp) * d**2 / 2
        objective = robust + prior + smooth
        fluxes = {
            "selected_flux_values_binary": [paired.normal._mpi(qx[3][4].value),
                                            paired.normal._mpi(qy[3][0].value)],
            "selected_flux_derivatives_binary": [paired.normal._mpi(qx[3][4].derivative),
                                                 paired.normal._mpi(qy[3][0].derivative)],
        }
        return {
            "side": side,
            "objective_mpi": paired.normal._mpi(objective.value),
            "sigma_eta_mpi": paired.normal._mpi(objective.derivative),
            "J_binary": paired.normal._mpi(objective.value),
            "sigma_eta_binary": paired.normal._mpi(objective.derivative),
            "objective_bounds": paired.normal._bounds(objective.value, dps),
            "sigma_eta_bounds": paired.normal._bounds(objective.derivative, dps),
            "selected_flux_values_binary": fluxes["selected_flux_values_binary"],
            "selected_flux_derivatives_binary": fluxes["selected_flux_derivatives_binary"],
            "selected_flux_values": [mp.nstr(qx[3][4].value, n=dps), mp.nstr(qy[3][0].value, n=dps)],
            "selected_flux_derivatives": [mp.nstr(qx[3][4].derivative, n=dps),
                                          mp.nstr(qy[3][0].derivative, n=dps)],
            "robust_cost_bounds": paired.normal._bounds(robust.value, dps),
            "prior_cost_bounds": paired.normal._bounds(prior.value, dps),
            "smooth_cost_bounds": paired.normal._bounds(smooth.value, dps),
            "branch_choices": trace,
            "scope": "directional interval derivative on one qx-zero side; qy is an ordinary flux",
        }
