"""Frozen CP3b affine transport holdout (research-only known-field comparison)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from advar import transport


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from compare_transport import moments, principal_moments
import muscl_experiment  # noqa: E402  (the candidate is a frozen local source)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("checkpoint") != "CP3b" or manifest.get("status") != "frozen before holdout execution":
        raise ValueError("manifest is not the frozen CP3b manifest")
    if manifest.get("candidate") != "minmod" or manifest.get("grid_size") != 128:
        raise ValueError("CP3b candidate/grid contract is invalid")
    return manifest


def _validate_sources(manifest: dict) -> dict[str, str]:
    paths = {
        "candidate": HERE / "muscl_experiment.py",
        "transport": ROOT / "src" / "advar" / "transport.py",
        "holdout": HERE / "affine_holdout.py",
        "compare_helper": HERE / "compare_transport.py",
        "oracle_helper": HERE / "finite_volume_probe.py",
    }
    actual = {str(path): _sha256(path) for path in paths.values()}
    expected = manifest["candidate_sha256"]
    for relative, digest in expected.items():
        path = (ROOT / relative).resolve()
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"frozen source hash mismatch: {relative}")
    return actual


def _flow_arrays(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    A = torch.tensor(case["A_per_second"], dtype=dtype)
    b = torch.tensor(case["b_mps"], dtype=dtype)
    trace_scale = float(torch.abs(A).max())
    if abs(float(torch.trace(A))) > 256 * torch.finfo(dtype).eps * trace_scale:
        raise ValueError(f"{case['name']}: A must have trace zero")
    return A, b


def _case_arrays(case: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    A, b = _flow_arrays(case)
    dtype = torch.float64
    center = torch.tensor(case["initial_center_xy_m"], dtype=dtype)
    angle = math.radians(float(case["initial_angle_degrees"]))
    return A, b, center, angle


def _geometry_checks(case: dict, side: float, total_time: float, speed_bound: float) -> dict:
    A, b, center, _ = _case_arrays(case)
    scales = torch.tensor(case["initial_scale_xy_m"], dtype=torch.float64)
    symmetric = 0.5 * (A + A.T)
    strain_rate = max(0.0, float(torch.linalg.eigvalsh(symmetric)[-1]))
    bound = math.exp(strain_rate * total_time) * (
        float(torch.linalg.vector_norm(center)) + 3.0 * float(scales.max())
        + total_time * float(torch.linalg.vector_norm(b))
    )
    if not math.isfinite(bound) or bound >= side / 2:
        raise ValueError(f"{case['name']}: analytic support bound leaves the domain")

    n = int(128)
    spacing = side / n
    vertices = (torch.arange(n + 1, dtype=torch.float64) - n / 2) * spacing
    y, x = torch.meshgrid(vertices, vertices, indexing="ij")
    u = A[0, 0] * x + A[0, 1] * y + b[0]
    v = A[1, 0] * x + A[1, 1] * y + b[1]
    component_speed = float(torch.maximum(torch.abs(u), torch.abs(v)).max())
    if component_speed > speed_bound + 256 * torch.finfo(torch.float64).eps * max(1.0, speed_bound):
        raise ValueError(f"{case['name']}: component speed exceeds frozen U3 bound")
    return {
        "trace_A_per_second": float(torch.trace(A)),
        "computed_strain_rate_norm_per_second": strain_rate,
        "computed_support_radius_bound_m": bound,
        "declared_support_radius_bound_m": case.get("support_radius_bound_m"),
        "computed_max_component_speed_mps": component_speed,
        "declared_max_component_speed_mps": case.get("max_component_speed_bound_mps"),
        "analytic_support_inside_domain": True,
    }


def affine_face_fluxes(case: dict, size: int, side: float):
    """Return shared volume fluxes for the manifest's stationary affine flow."""
    spacing = side / size
    vertices = (torch.arange(size + 1, dtype=torch.float64) - size / 2) * spacing
    y, x = torch.meshgrid(vertices, vertices, indexing="ij")
    A, b = _flow_arrays(case)
    psi = (
        0.5 * A[0, 1] * y.square() + A[0, 0] * x * y
        - 0.5 * A[1, 0] * x.square() + b[0] * y - b[1] * x
    )
    return transport.face_volume_fluxes(psi - psi[0, 0])


def affine_cell_averages(case: dict, size: int, side: float, elapsed: float, order: int = 8) -> torch.Tensor:
    """Independent augmented-matrix characteristic solution and cell average."""
    if order not in (8, 16):
        raise ValueError("quadrature order must be 8 or 16")
    dtype = torch.float64
    spacing = side / size
    centers = (torch.arange(size, dtype=dtype) + 0.5 - size / 2) * spacing
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    A, b, center, angle = _case_arrays(case)
    H = torch.zeros((3, 3), dtype=dtype)
    H[:2, :2] = A
    H[:2, 2] = b
    back = torch.linalg.matrix_exp(-float(elapsed) * H)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    c, s = math.cos(angle), math.sin(angle)
    field = torch.zeros_like(x)
    for ny, wy in zip(nodes, weights):
        for nx, wx in zip(nodes, weights):
            arrival = torch.stack((x + float(nx) * spacing / 2, y + float(ny) * spacing / 2,
                                   torch.ones_like(x)))
            departure = torch.einsum("ij,jhw->ihw", back, arrival)[:2]
            dx, dy = departure[0] - center[0], departure[1] - center[1]
            local_x, local_y = c * dx + s * dy, -s * dx + c * dy
            scales = torch.tensor(case["initial_scale_xy_m"], dtype=dtype)
            r2 = (local_x / scales[0]).square() + (local_y / scales[1]).square()
            profile = torch.exp(-r2 / 2) * torch.clamp(1 - r2 / 9, min=0).pow(4)
            field += float(wx * wy / 4) * profile
    return field * math.exp(float(case["growth_per_second"]) * float(elapsed))


def _orientation_error(actual: torch.Tensor, truth: torch.Tensor, threshold: float, spacing: float):
    actual_axes, actual_angle = principal_moments(actual, spacing)
    truth_axes, truth_angle = principal_moments(truth, spacing)
    anisotropy = float((truth_axes[1].square() - truth_axes[0].square()) / truth_axes.square().sum())
    if anisotropy < threshold:
        return None, False, anisotropy, actual_axes, truth_axes
    error = torch.abs((actual_angle - truth_angle + math.pi / 2) % math.pi - math.pi / 2)
    return math.degrees(float(error)), True, anisotropy, actual_axes, truth_axes


def _run_case(case: dict, manifest: dict, scheme: str) -> dict:
    size, side = int(manifest["grid_size"]), float(manifest["domain_side_m"])
    interval, leads = float(manifest["lead_interval_seconds"]), int(manifest["lead_count"])
    spacing = side / size
    qx, qy = affine_face_fluxes(case, size, side)
    outgoing = qx[:, 1:].clamp_min(0) - qx[:, :-1].clamp_max(0) + qy[1:, :].clamp_min(0) - qy[:-1, :].clamp_max(0)
    substeps = math.ceil(interval * 4 * float(manifest["speed_bound_mps"]) / spacing / float(manifest["max_courant"]))
    dt = interval / substeps
    actual_cfl = float(dt * (outgoing / spacing**2).max())
    if actual_cfl > float(manifest["max_courant"]) + 1e-12:
        raise ValueError(f"{case['name']}: fixed schedule violates CFL")
    echo = affine_cell_averages(case, size, side, 0.0, 8)
    zeros = tuple(torch.zeros(size, dtype=torch.float64) for _ in range(4))
    ones = tuple(torch.ones(size, dtype=torch.float64) for _ in range(4))
    area = spacing**2
    records, budget_max, seconds = [], 0.0, 0.0
    growth_rate = float(case["growth_per_second"])
    for lead in range(1, leads + 1):
        lead_budget_max, lead_outflow = 0.0, 0.0
        lead_start_mass = float(echo.sum()) * area
        for substep in range(substeps):
            old_mass = float(echo.sum()) * area
            started = time.perf_counter()
            if scheme == "donorcell":
                result = transport.finite_volume_step(
                    echo, torch.ones_like(echo), qx, qy, dt_seconds=dt,
                    spacing_yx=(spacing, spacing), log_growth=float(case["growth_per_second"]) * dt,
                    boundary_echo=(zeros, zeros), boundary_support=(ones, ones), max_courant=float(manifest["max_courant"]),
                )
                updated, outflow = result.echo, result.transport_outflow
            else:
                updated, outflow = muscl_experiment.muscl_step(
                    echo, qx, qy, dt_seconds=dt, spacing_yx=(spacing, spacing),
                    log_growth=float(case["growth_per_second"]) * dt,
                )
            seconds += time.perf_counter() - started
            residual = abs(math.exp(-float(case["growth_per_second"]) * dt) * float(updated.sum()) * area - old_mass + float(outflow)) / max(old_mass, np.finfo(float).tiny)
            budget_max = max(budget_max, residual)
            lead_budget_max = max(lead_budget_max, residual)
            # Local r resets at every step; convert all terms to this lead's
            # start before accumulating a transformed mass budget.
            lead_outflow += math.exp(-substep * growth_rate * dt) * float(outflow)
            echo = updated
        lead_residual = abs(math.exp(-growth_rate * interval) * float(echo.sum()) * area
                            - lead_start_mass + lead_outflow) / lead_start_mass
        budget_max = max(budget_max, lead_residual)
        truth = affine_cell_averages(case, size, side, lead * interval, 8)
        center, width = moments(echo, spacing)
        true_center, true_width = moments(truth, spacing)
        orientation, covered, anisotropy, principal, true_principal = _orientation_error(
            echo, truth, float(manifest["thresholds"]["truth_anisotropy_minimum"]), spacing
        )
        records.append({
            "lead_minutes": lead * interval / 60,
            "relative_echo_l2": float(torch.linalg.vector_norm(echo - truth) / torch.linalg.vector_norm(truth)),
            "center_error_m": float(torch.linalg.vector_norm(center - true_center)),
            "center_error_pixels": float(torch.linalg.vector_norm(center - true_center) / spacing),
            "axis_width_relative_error": ((width - true_width) / true_width).tolist(),
            "principal_width_relative_error": ((principal - true_principal) / true_principal).tolist(),
            "orientation_error_degrees": orientation,
            "orientation_covered": covered,
            "truth_anisotropy": anisotropy,
            "minimum_echo": float(echo.min()),
            "transformed_outflow": lead_outflow,
            "transformed_budget_residual": lead_budget_max,
            "lead_transformed_budget_residual": lead_residual,
        })
    truth8 = affine_cell_averages(case, size, side, leads * interval, 8)
    truth16 = affine_cell_averages(case, size, side, leads * interval, 16)
    oracle_error = float(torch.linalg.vector_norm(truth8 - truth16) / torch.linalg.vector_norm(truth16))
    return {"case": case["name"], "scheme": scheme, "size": size, "spacing_m": spacing,
            "substeps_per_lead": substeps, "actual_max_cfl": actual_cfl,
            "wall_seconds": seconds, "max_relative_transformed_budget_residual": budget_max,
            "final_oracle_relative_l2_8_vs_16": oracle_error, "leads": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    sources = _validate_sources(manifest)
    total_time = float(manifest["lead_interval_seconds"]) * int(manifest["lead_count"])
    geometry = {case["name"]: _geometry_checks(case, float(manifest["domain_side_m"]), total_time, float(manifest["speed_bound_mps"])) for case in manifest["cases"]}
    report = {
        "status": "running", "scope": "CP3b frozen affine holdout; full known field and known-zero exterior; no P1/FSO/production claim",
        "boundary_condition": "known-zero exterior with no inflow",
        "budget_coordinates": "transformed_outflow is relative to each lead start; cross-lead sums require growth reweighting",
        "known_complete_field": True, "known_zero_exterior": True, "no_inflow": True,
        "candidate_hash_matches": True,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": _sha256(args.manifest),
        "candidate": "minmod", "candidate_grid": 128, "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads(), "source_sha256": sources,
        "geometry_checks": geometry, "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for case in manifest["cases"]:
            for scheme in ("donorcell", "minmod"):
                result = _run_case(case, manifest, scheme)
                report["results"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(case["name"], scheme, result["leads"][-1]["relative_echo_l2"], flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["status"] = "complete"
    report["whole_process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
