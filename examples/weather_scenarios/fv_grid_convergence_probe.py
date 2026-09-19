"""Small prescribed-flow FV grid and derivative convergence probe.

This probe exercises the current donor-cell transport core against independent
characteristic cell averages.  It is intentionally limited to a complete
known field with known-zero exterior; it does not evaluate P1, FSO, FSOI, or
forecast skill.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import torch

from advar import transport

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from affine_holdout import affine_cell_averages, affine_face_fluxes  # noqa: E402
from finite_volume_probe import cell_averages  # noqa: E402


SIDE = 48000.0
INTERVAL = 600.0
MAX_COURANT = 0.5
LEADS = 3


def _translation_case() -> dict:
    return {
        "name": "translation",
        "kind": "characteristic",
        "velocity": (0.5, 0.0),
        "omega": 0.0,
        "growth_rate": 0.0,
        "speed_bound_mps": 0.5,
    }


def _rotation_case() -> dict:
    return {
        "name": "rotation",
        "kind": "characteristic",
        "velocity": (0.0, 0.0),
        "omega": math.pi / 4 / 10800,
        "growth_rate": 0.0,
        "speed_bound_mps": 3.0,
    }


def _strain_case() -> dict:
    # This is the frozen CP3b area-preserving strain, reused as an analytic
    # oracle only; the minmod candidate is not part of this probe.
    return {
        "name": "area_preserving_strain",
        "kind": "affine",
        "A_per_second": [[1.688162562906987e-05, 0.0], [0.0, -1.688162562906987e-05]],
        "b_mps": [0.0, 0.0],
        "growth_per_second": 3.240740740740741e-05,
        "initial_center_xy_m": [-3000.0, 2000.0],
        "initial_scale_xy_m": [2600.0, 4300.0],
        "initial_angle_degrees": -10.0,
        "speed_bound_mps": 0.5,
    }


CASES = (_translation_case(), _rotation_case(), _strain_case())


def _flow(case: dict, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if case["kind"] == "characteristic":
        spacing = SIDE / size
        vertices = (torch.arange(size + 1, dtype=torch.float64) - size / 2) * spacing
        y, x = torch.meshgrid(vertices, vertices, indexing="ij")
        psi = (case["velocity"][0] * y - case["velocity"][1] * x
               - case["omega"] * (x.square() + y.square()) / 2)
        return transport.face_volume_fluxes(psi - psi[0, 0])
    return affine_face_fluxes(case, size, SIDE)


def _oracle(case: dict, size: int, elapsed: float, amplitude: float = 1.0) -> torch.Tensor:
    spacing = SIDE / size
    if case["kind"] == "characteristic":
        velocity = tuple(amplitude * value for value in case["velocity"])
        return cell_averages(
            size, spacing, elapsed, velocity, amplitude * case["omega"],
            case["growth_rate"], 8,
        )
    varied = copy.deepcopy(case)
    varied["A_per_second"] = [[amplitude * value for value in row]
                               for row in case["A_per_second"]]
    varied["b_mps"] = [amplitude * value for value in case["b_mps"]]
    return affine_cell_averages(varied, size, SIDE, elapsed, 8)


def _schedule(qx: torch.Tensor, qy: torch.Tensor, size: int, speed_bound: float) -> tuple[int, float, float]:
    spacing = SIDE / size
    substeps = max(1, math.ceil(INTERVAL * 4.0 * speed_bound / spacing / MAX_COURANT))
    dt = INTERVAL / substeps
    outgoing = (qx[:, 1:].clamp_min(0.0) - qx[:, :-1].clamp_max(0.0)
                + qy[1:, :].clamp_min(0.0) - qy[:-1, :].clamp_max(0.0))
    actual_cfl = float((dt * outgoing / spacing**2).max())
    if actual_cfl > MAX_COURANT + 1e-12:
        raise RuntimeError(f"CFL schedule violated: {actual_cfl}")
    return substeps, dt, actual_cfl


def _advance(echo: torch.Tensor, qx: torch.Tensor, qy: torch.Tensor,
             substeps: int, dt: float, growth_rate: float) -> torch.Tensor:
    size = echo.shape[0]
    zeros = tuple(torch.zeros(size, dtype=echo.dtype) for _ in range(4))
    ones = tuple(torch.ones(size, dtype=echo.dtype) for _ in range(4))
    support = torch.ones_like(echo)
    spacing = SIDE / size
    for _ in range(substeps):
        result = transport.finite_volume_step(
            echo, support, qx, qy, dt_seconds=dt,
            spacing_yx=(spacing, spacing), log_growth=growth_rate * dt,
            boundary_echo=(zeros, zeros), boundary_support=(ones, ones),
            max_courant=MAX_COURANT,
        )
        echo = result.echo
    return echo


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(torch.finfo(a.dtype).tiny))


def _derivative_record(case: dict, size: int, qx: torch.Tensor, qy: torch.Tensor,
                       substeps: int, dt: float, leads: int) -> dict:
    """Compare core JVP with the independent characteristic/affine derivative."""
    total_time = leads * INTERVAL
    initial = _oracle(case, size, 0.0)

    def trajectory(amplitude: torch.Tensor) -> torch.Tensor:
        # The branch decisions (face signs and schedule) are fixed at the
        # nominal amplitude; this is the differentiable local model being checked.
        return _advance(initial, amplitude * qx, amplitude * qy, substeps * leads,
                        dt, float(case.get("growth_rate", case.get("growth_per_second", 0.0))))

    nominal = torch.ones((), dtype=torch.float64, requires_grad=True)
    _, jvp = torch.func.jvp(trajectory, (nominal,), (torch.ones_like(nominal),))
    h = 1e-4
    oracle_derivative = (_oracle(case, size, total_time, 1.0 + h)
                         - _oracle(case, size, total_time, 1.0 - h)) / (2.0 * h)
    return {
        "jvp_relative_to_independent_oracle": _relative(jvp, oracle_derivative),
        "jvp_norm": float(torch.linalg.vector_norm(jvp)),
        "oracle_derivative_norm": float(torch.linalg.vector_norm(oracle_derivative)),
        "oracle_fd_step": h,
    }


def run_case(case: dict, size: int, leads: int = LEADS) -> dict:
    spacing = SIDE / size
    qx, qy = _flow(case, size)
    substeps, dt, actual_cfl = _schedule(qx, qy, size, case["speed_bound_mps"])
    growth_rate = float(case.get("growth_rate", case.get("growth_per_second", 0.0)))
    echo = _oracle(case, size, 0.0)
    records = []
    budget_max = 0.0
    started = time.perf_counter()
    zeros = tuple(torch.zeros(size, dtype=torch.float64) for _ in range(4))
    ones = tuple(torch.ones(size, dtype=torch.float64) for _ in range(4))
    for lead in range(1, leads + 1):
        lead_start_mass = float(echo.sum()) * spacing**2
        transformed_outflow = 0.0
        for substep in range(substeps):
            old_mass = float(echo.sum()) * spacing**2
            result = transport.finite_volume_step(
                echo, torch.ones_like(echo), qx, qy, dt_seconds=dt,
                spacing_yx=(spacing, spacing), log_growth=growth_rate * dt,
                boundary_echo=(zeros, zeros), boundary_support=(ones, ones),
                max_courant=MAX_COURANT,
            )
            echo = result.echo
            restored_mass = math.exp(-growth_rate * dt) * float(echo.sum()) * spacing**2
            residual = abs(restored_mass - old_mass + float(result.transport_outflow)) / max(old_mass, 1e-300)
            budget_max = max(budget_max, residual)
            transformed_outflow += math.exp(-substep * growth_rate * dt) * float(result.transport_outflow)
        lead_residual = abs(math.exp(-growth_rate * INTERVAL) * float(echo.sum()) * spacing**2
                            - lead_start_mass + transformed_outflow) / max(lead_start_mass, 1e-300)
        budget_max = max(budget_max, lead_residual)
        truth = _oracle(case, size, lead * INTERVAL)
        records.append({
            "lead_minutes": lead * INTERVAL / 60.0,
            "relative_echo_l2": _relative(echo, truth),
            "minimum_echo": float(echo.min()),
            "transformed_budget_residual": lead_residual,
        })
    derivative = _derivative_record(case, size, qx, qy, substeps, dt, leads)
    return {
        "case": case["name"], "size": size, "spacing_m": spacing,
        "substeps_per_lead": substeps, "actual_max_cfl": actual_cfl,
        "wall_seconds": time.perf_counter() - started,
        "max_transformed_budget_residual": budget_max,
        "leads": records, "derivative": derivative,
    }


def run_probe(sizes: tuple[int, ...] = (32, 64, 128), *, leads: int = LEADS) -> dict:
    if type(leads) is not int or not 1 <= leads <= 18:
        raise ValueError("leads must be an integer in [1, 18]")
    started = time.perf_counter()
    with torch.no_grad():
        results = [run_case(case, size, leads) for size in sizes for case in CASES]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "status": "complete", "scope": "prescribed-flow FV PDE grid/JVP convergence only; no P1/FSO/FSOI, learning, or forecast-skill claim",
        "boundary_condition": "known-zero exterior with complete known initial field",
        "oracle": "existing characteristic cell_averages plus existing CP3b affine_cell_averages for area-preserving strain",
        "cases": [case["name"] for case in CASES], "sizes": list(sizes),
        "domain_side_m": SIDE, "interval_seconds": INTERVAL, "leads": leads,
        "max_courant": MAX_COURANT, "scheme": "current donorcell SSPRK2",
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads(),
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path(__file__), Path(transport.__file__),
                                        HERE / "finite_volume_probe.py", HERE / "affine_holdout.py")},
        "results": results,
        "wall_seconds": time.perf_counter() - started,
        "whole_process_peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--leads", type=int, default=LEADS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if any(size < 16 or size > 128 for size in args.sizes):
        parser.error("sizes must be in [16, 128]")
    if not 1 <= args.leads <= 18:
        parser.error("leads must be in [1, 18]")
    report = run_probe(tuple(args.sizes), leads=args.leads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
