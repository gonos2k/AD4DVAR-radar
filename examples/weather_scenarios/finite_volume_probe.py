"""Measure the actual FV baseline with prescribed flow, not inferred forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from advar import transport


def _validate_oracle_case(size, spacing, elapsed, velocity, omega, growth_rate, quadrature_order):
    if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
        raise ValueError("size must be a positive integer")
    if isinstance(spacing, bool) or not isinstance(spacing, Real) or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a positive finite number")
    if isinstance(elapsed, bool) or not isinstance(elapsed, Real) or not math.isfinite(elapsed):
        raise ValueError("elapsed must be finite")
    if not isinstance(velocity, (tuple, list)) or len(velocity) != 2:
        raise ValueError("velocity must be (vx, vy)")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in velocity):
        raise ValueError("velocity must contain finite numbers")
    if isinstance(omega, bool) or not isinstance(omega, Real) or not math.isfinite(omega):
        raise ValueError("omega must be finite")
    if isinstance(growth_rate, bool) or not isinstance(growth_rate, Real) or not math.isfinite(growth_rate):
        raise ValueError("growth_rate must be finite")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, Integral) or quadrature_order <= 0:
        raise ValueError("quadrature_order must be a positive integer")
    if omega != 0 and (velocity[0] != 0 or velocity[1] != 0):
        raise ValueError("simultaneous translation and rotation are not supported")
    exponent = float(growth_rate) * float(elapsed)
    if not math.isfinite(exponent):
        raise ValueError("growth_rate * elapsed must be finite")
    try:
        growth = math.exp(exponent)
    except OverflowError:
        raise ValueError("growth factor is not finite") from None
    if not math.isfinite(growth):
        raise ValueError("growth factor is not finite")
    return int(size), float(spacing), float(elapsed), (float(velocity[0]), float(velocity[1])), float(omega), float(growth_rate), int(quadrature_order)


def _characteristic_echo(y, x, elapsed, velocity, omega, growth_rate):
    sample_x = x - velocity[0] * elapsed
    sample_y = y - velocity[1] * elapsed
    cosine, sine = math.cos(omega * elapsed), math.sin(omega * elapsed)
    departure_x = cosine * sample_x + sine * sample_y
    departure_y = -sine * sample_x + cosine * sample_y
    radius2 = ((departure_x + 5000) / 3000).square() + ((departure_y - 2000) / 4500).square()
    profile = torch.exp(-radius2 / 2) * (1 - radius2 / 9).clamp_min(0).pow(4)
    return profile * math.exp(growth_rate * elapsed)


def characteristic_echo(y, x, elapsed, velocity=(0.0, 0.0), omega=0.0, growth_rate=0.0):
    """Pointwise compact-profile echo evaluated at inverse-characteristic coordinates."""
    if not isinstance(y, torch.Tensor) or not isinstance(x, torch.Tensor) or y.shape != x.shape:
        raise ValueError("x and y must be same-shaped tensors")
    if y.dtype != torch.float64 or x.dtype != y.dtype or not bool(torch.isfinite(y).all() & torch.isfinite(x).all()):
        raise ValueError("x and y must be finite float64 tensors")
    _, _, elapsed, velocity, omega, growth_rate, _ = _validate_oracle_case(
        1, 1.0, elapsed, velocity, omega, growth_rate, 1
    )
    return _characteristic_echo(y, x, elapsed, velocity, omega, growth_rate)


def cell_averages(size, spacing, elapsed, velocity, omega, growth_rate, quadrature_order=8):
    """Independent characteristic solution integrated by tensor Gauss quadrature."""
    size, spacing, elapsed, velocity, omega, growth_rate, quadrature_order = _validate_oracle_case(
        size, spacing, elapsed, velocity, omega, growth_rate, quadrature_order
    )
    centers = (torch.arange(size, dtype=torch.float64) + 0.5 - size / 2) * spacing
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    field = torch.zeros_like(x)
    for ny, wy in zip(nodes, weights):
        for nx, wx in zip(nodes, weights):
            field += float(wx * wy / 4) * _characteristic_echo(
                y + float(ny) * spacing / 2,
                x + float(nx) * spacing / 2,
                elapsed, velocity, omega, growth_rate,
            )
    return field


def face_averages(size, spacing, elapsed, velocity, omega, growth_rate, quadrature_order=8):
    """Return characteristic averages on left, right, bottom, and top faces."""
    size, spacing, elapsed, velocity, omega, growth_rate, quadrature_order = _validate_oracle_case(
        size, spacing, elapsed, velocity, omega, growth_rate, quadrature_order
    )
    centers = (torch.arange(size, dtype=torch.float64) + 0.5 - size / 2) * spacing
    boundary = size * spacing / 2
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)

    def average(fixed, varying, fixed_first):
        result = torch.zeros_like(varying)
        for node, weight in zip(nodes, weights):
            offset = varying + float(node) * spacing / 2
            y, x = (fixed, offset) if fixed_first else (offset, fixed)
            result += float(weight / 2) * _characteristic_echo(y, x, elapsed, velocity, omega, growth_rate)
        return result

    return (
        average(torch.full_like(centers, -boundary), centers, False),
        average(torch.full_like(centers, boundary), centers, False),
        average(torch.full_like(centers, -boundary), centers, True),
        average(torch.full_like(centers, boundary), centers, True),
    )


def moments(field, spacing):
    coordinates = (torch.arange(field.shape[0], dtype=field.dtype) + 0.5 - field.shape[0] / 2) * spacing
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    total = field.sum()
    center = torch.stack(((field * x).sum(), (field * y).sum())) / total
    variance = torch.stack((
        (field * (x - center[0]).square()).sum(),
        (field * (y - center[1]).square()).sum(),
    )) / total
    return center, variance.sqrt()


def run_case(name, size, leads, velocity, omega, growth_rate):
    # A fixed 48 km square permits spatial refinement without changing the case.
    spacing, interval = 48000 / size, 600.0
    vertices = (torch.arange(size + 1, dtype=torch.float64) - size / 2) * spacing
    y, x = torch.meshgrid(vertices, vertices, indexing="ij")
    psi = velocity[0] * y - velocity[1] * x - omega * (x.square() + y.square()) / 2
    psi = psi - psi[0, 0]
    qx, qy = transport.face_volume_fluxes(psi)
    speed_bound = 3.0  # Prescribed case bound, not chosen from a fitted trajectory.
    substeps = max(1, math.ceil(interval * 4 * speed_bound / spacing / 0.9))
    dt = interval / substeps
    echo = cell_averages(size, spacing, 0, velocity, omega, growth_rate)
    support = torch.ones_like(echo)
    records = []
    elapsed_solver = 0.0
    for lead in range(1, leads + 1):
        start = time.perf_counter()
        for _ in range(substeps):
            result = transport.finite_volume_step(
                echo, support, qx, qy, dt_seconds=dt,
                spacing_yx=(spacing, spacing), log_growth=growth_rate * dt,
            )
            echo, support = result.echo, result.support
        elapsed_solver += time.perf_counter() - start
        truth = cell_averages(size, spacing, lead * interval, velocity, omega, growth_rate)
        center, width = moments(echo, spacing)
        true_center, true_width = moments(truth, spacing)
        records.append({
            "lead_minutes": lead * interval / 60,
            "relative_echo_l2": float(torch.linalg.vector_norm(echo - truth) / torch.linalg.vector_norm(truth)),
            "center_error_m": float(torch.linalg.vector_norm(center - true_center)),
            "axis_width_relative_error": ((width - true_width) / true_width).tolist(),
            "echo_integral": float(echo.sum()) * spacing**2,
            "oracle_echo_integral": float(truth.sum()) * spacing**2,
            "support_mean": float(support.mean()),
            "minimum_echo": float(echo.min()),
        })
    finer_truth = cell_averages(size, spacing, leads * interval, velocity, omega, growth_rate, 16)
    quadrature_error = float(torch.linalg.vector_norm(truth - finer_truth) / torch.linalg.vector_norm(finer_truth))
    return {"case": name, "prescribed_velocity_xy_mps": velocity, "omega_per_second": omega,
            "final_oracle_relative_l2_8_vs_16": quadrature_error,
            "growth_per_second": growth_rate, "substeps_per_lead": substeps,
            "total_substeps": leads * substeps, "solver_wall_seconds": elapsed_solver, "leads": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--leads", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 16 <= args.size <= 256 or not 1 <= args.leads <= 18:
        parser.error("size must be 16..256 and leads 1..18")
    cases = [
        ("translation", (0.5, 0.0), 0.0, 0.0),
        ("rotation_positive", (0.0, 0.0), math.pi / 4 / 10800, 0.0),
        ("rotation_negative_decay", (0.0, 0.0), -math.pi / 4 / 10800, -0.04 / 600),
    ]
    with torch.no_grad():
        results = [run_case(name, args.size, args.leads, velocity, omega, growth)
                   for name, velocity, omega, growth in cases]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"scope": "prescribed-flow FV numerical baseline; no P0/P1/FSO learning or forecast skill claim",
              "python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
              "dtype": "float64", "threads": torch.get_num_threads(), "size": args.size,
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__).resolve(), Path(transport.__file__).resolve())},
              "domain_side_m": 48000, "oracle": "compact profile, inverse characteristic, 8x8 Gauss cell quadrature",
              "whole_process_peak_rss_bytes": int(peak_rss if sys.platform == "darwin" else 1024 * peak_rss),
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
