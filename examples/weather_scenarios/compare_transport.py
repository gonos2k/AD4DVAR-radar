"""Development comparison of known-field transport; not inferred forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import resource
import sys
import time

import torch
from advar import transport

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finite_volume_probe import cell_averages, moments
import muscl_experiment


CASES = (
    ("translation", (0.5, 0.0), 0.0, 0.0),
    ("rotation_positive", (0.0, 0.0), math.pi / 4 / 10800, 0.0),
    ("rotation_negative_decay", (0.0, 0.0), -math.pi / 4 / 10800, -0.04 / 600),
)


def principal_moments(field, spacing):
    coordinates = (torch.arange(field.shape[0], dtype=field.dtype) + .5 - field.shape[0] / 2) * spacing
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    center, _ = moments(field, spacing)
    x, y = x - center[0], y - center[1]
    mass = field.sum()
    xx, xy, yy = ((field * a).sum() / mass for a in (x * x, x * y, y * y))
    covariance = torch.stack((torch.stack((xx, xy)), torch.stack((xy, yy))))
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    angle = torch.atan2(vectors[1, -1], vectors[0, -1])
    return eigenvalues.sqrt(), angle


def run_case(case, size, scheme, time_multiplier):
    name, velocity, omega, rate = case
    spacing, interval = 48000 / size, 600.
    vertices = (torch.arange(size + 1, dtype=torch.float64) - size / 2) * spacing
    y, x = torch.meshgrid(vertices, vertices, indexing="ij")
    psi = velocity[0] * y - velocity[1] * x - omega * (x * x + y * y) / 2
    qx, qy = transport.face_volume_fluxes(psi - psi[0, 0])
    # Same conservative schedule for both schemes. This is a prescribed bound.
    substeps = math.ceil(interval * 4 * 3. / spacing / .5) * time_multiplier
    dt = interval / substeps
    outgoing = (qx[:, 1:].relu() + (-qx[:, :-1]).relu()
                + qy[1:].relu() + (-qy[:-1]).relu())
    actual_cfl = float((dt * (outgoing / spacing**2)).max())
    echo = cell_averages(size, spacing, 0., velocity, omega, rate)
    support = torch.ones_like(echo)
    zeros = tuple(torch.zeros(size, dtype=echo.dtype) for _ in range(4))
    ones = tuple(torch.ones(size, dtype=echo.dtype) for _ in range(4))
    seconds, budget_error = 0., 0.
    records = []
    for lead in range(1, 19):
        started = time.perf_counter()
        for _ in range(substeps):
            previous_mass = float(echo.sum()) * spacing**2
            if scheme == "donorcell":
                result = transport.finite_volume_step(
                    echo, support, qx, qy, dt_seconds=dt, spacing_yx=(spacing, spacing),
                    log_growth=rate * dt, boundary_echo=(zeros, zeros),
                    boundary_support=(ones, ones), max_courant=.5,
                )
                updated, outflow = result.echo, result.transport_outflow
            else:
                updated, outflow = muscl_experiment.muscl_step(
                    echo, qx, qy, dt_seconds=dt, spacing_yx=(spacing, spacing), log_growth=rate * dt,
                )
            restored_mass = float(updated.sum()) * spacing**2 * math.exp(-rate * dt)
            residual = abs(restored_mass - previous_mass + float(outflow)) / previous_mass
            budget_error = max(budget_error, residual)
            echo = updated
        seconds += time.perf_counter() - started
        truth = cell_averages(size, spacing, lead * interval, velocity, omega, rate)
        center, width = moments(echo, spacing)
        true_center, true_width = moments(truth, spacing)
        principal, angle = principal_moments(echo, spacing)
        true_principal, true_angle = principal_moments(truth, spacing)
        # An ellipse orientation has period pi. The prescribed ellipses are anisotropic.
        angle_error = float(torch.abs((angle - true_angle + math.pi / 2) % math.pi - math.pi / 2))
        records.append({
            "lead_minutes": lead * 10, "relative_echo_l2": float(torch.linalg.vector_norm(echo-truth) / torch.linalg.vector_norm(truth)),
            "center_error_m": float(torch.linalg.vector_norm(center-true_center)),
            "axis_width_relative_error": ((width-true_width)/true_width).tolist(),
            "principal_width_relative_error": ((principal-true_principal)/true_principal).tolist(),
            "orientation_error_degrees": math.degrees(angle_error), "minimum_echo": float(echo.min()),
        })
    return {"case": name, "scheme": scheme, "size": size, "spacing_m": spacing,
            "substeps_per_lead": substeps, "actual_max_cfl": actual_cfl,
            "wall_seconds_including_budget_diagnostics": seconds,
            "max_relative_transformed_budget_residual": budget_error, "leads": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[48, 64, 128, 256])
    parser.add_argument("--time-multiplier", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if any(n < 16 or n > 256 for n in args.sizes) or args.time_multiplier not in (1, 2, 4):
        parser.error("sizes must be 16..256 and time-multiplier 1, 2, or 4")
    source_paths = [Path(__file__), Path(muscl_experiment.__file__), Path(transport.__file__),
                    Path(__file__).with_name("finite_volume_probe.py")]
    report = {"status": "running", "scope": "development cases; full known field and known zero exterior; no forecast/FSO/support attribution claim",
              "python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(),
              "device": "cpu", "dtype": "float64", "domain_side_m": 48000,
              "sizes": args.sizes, "time_multiplier": args.time_multiplier,
              "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              "cost_scope": "wall time includes budget diagnostics; donorcell includes support/audits omitted by MUSCL, so not a production cost comparison",
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for size in args.sizes:
            for case in CASES:
                for scheme in ("donorcell", "minmod"):
                    result = run_case(case, size, scheme, args.time_multiplier)
                    report["results"].append(result)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(size, case[0], scheme, result["leads"][-1]["relative_echo_l2"], flush=True)
    report["status"] = "complete"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["whole_process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
