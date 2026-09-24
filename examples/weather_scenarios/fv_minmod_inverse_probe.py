"""Small matched minmod inverse oracle; not a general exact-response API.

All nine initial cells, two constant-flow coefficients and growth are controls.
Known boundaries are fixed. Strict interior limiter/flow branches are inspected
at every Euler stage; perimeter slopes are identically zero by construction.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import hashlib
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import torch
from advar import transport as t, variational as v
from advar.nowcast import NowcastConfig
from advar.physics import dbz_to_echo, echo_to_dbz

_spec = importlib.util.spec_from_file_location(
    "fv_dense_oracle", Path(__file__).with_name("fv_sensitivity_probe.py")
)
assert _spec is not None and _spec.loader is not None
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


def make_case():
    y, x = torch.meshgrid(torch.arange(4, dtype=torch.float64),
                          torch.arange(4, dtype=torch.float64), indexing="ij")
    basis = torch.stack((y, -x))
    q = 30 * torch.exp(.2*x[:3, :3] + .3*y[:3, :3]
                       + .04*x[:3, :3]**2 + .07*y[:3, :3]**2)
    edges = (q[:, 0], q[:, -1], q[0], q[-1])
    ones = tuple(torch.ones_like(e) for e in edges)
    boundary, support = ((edges, edges),)*27, ((ones, ones),)*27
    limits = q.new_tensor([.15, .15])
    spec = v.FVAnalysisTransport(
        psi_basis=basis, coefficient_limits=limits, substeps_per_interval=9,
        spacing_yx=(10., 10.), boundary_echo=boundary[:18],
        boundary_support=support[:18], reconstruction="minmod", replay=True,
    )
    truth, _ = t.finite_volume_trajectory(
        q, torch.ones_like(q), limits*q.new_tensor([.3, .2]).tanh(),
        q.new_tensor(.008), psi_basis=basis, leads=2,
        substeps_per_interval=9, interval_seconds=60., spacing_yx=(10., 10.),
        boundary_echo=boundary[:18], boundary_support=support[:18],
        reconstruction="minmod",
    )
    obs, frozen = v.prepare_analysis(
        echo_to_dbz(truth, min_dbz=-10.),
        nowcast_config=NowcastConfig(interval_minutes=1, horizon_minutes=1),
        analysis_config=v.AnalysisConfig(field_smoothness_weight=0.),
        observation_std_dbz=.1, fv_transport=spec,
    )
    return obs, frozen, boundary[18:], support[18:]


def make_spatial_case(*, smoke_only: bool = False):
    """Build the bounded 4x5 joint minmod fixture (26 controls).

    ``smoke_only`` limits the optimizer budget for a cost check; it is never
    used as a stationarity or response result.
    """
    y, x = torch.meshgrid(
        torch.arange(5, dtype=torch.float64),
        torch.arange(6, dtype=torch.float64),
        indexing="ij",
    )
    basis = torch.stack(
        (y, x, x * y, 0.5 * (x.square() - y.square()), x.square() * y)
    )
    cell_y, cell_x = y[:-1, :-1], x[:-1, :-1]
    # Scale the existing smooth joint fixture above the detected-data floor;
    # the scale is part of the synthetic observation, not a learned offset.
    q = 20 * (
        1.5
        + 0.04 * cell_y
        + 0.03 * cell_x
        + 0.01 * cell_x * cell_y
        + 0.001 * cell_x.square()
        + 0.002 * cell_y.square()
        + 0.0003 * cell_x * cell_y.square()
    )
    limits = q.new_tensor([0.11, 0.08, 0.07, 0.04, 0.03])
    truth_coefficients = limits * q.new_tensor([0.7, -0.6, 0.5, 0.4, -0.3])
    edges = (q[:, 0], q[:, -1], q[0], q[-1])
    ones = tuple(torch.ones_like(edge) for edge in edges)
    boundary = ((edges, edges),) * 27
    support = ((ones, ones),) * 27
    transport = v.FVAnalysisTransport(
        psi_basis=basis,
        coefficient_limits=limits,
        substeps_per_interval=9,
        spacing_yx=(10.0, 10.0),
        boundary_echo=boundary[:18],
        boundary_support=support[:18],
        reconstruction="minmod",
        replay=True,
    )
    truth, _ = t.finite_volume_trajectory(
        q,
        torch.ones_like(q),
        truth_coefficients,
        q.new_tensor(0.008),
        psi_basis=basis,
        leads=2,
        substeps_per_interval=9,
        interval_seconds=60.0,
        spacing_yx=(10.0, 10.0),
        boundary_echo=boundary[:18],
        boundary_support=support[:18],
        reconstruction="minmod",
    )
    analysis_config = v.AnalysisConfig(field_smoothness_weight=0.0)
    if smoke_only:
        analysis_config = replace(
            analysis_config,
            maximum_outer_iterations=1,
            maximum_pcg_iterations=8,
        )
    obs, frozen = v.prepare_analysis(
        echo_to_dbz(truth, min_dbz=-10.0),
        nowcast_config=NowcastConfig(
            interval_minutes=1, horizon_minutes=1, min_dbz=-10.0
        ),
        analysis_config=analysis_config,
        observation_std_dbz=0.1,
        fv_transport=transport,
    )
    return obs, frozen, boundary[18:], support[18:]


def inspect_branches(call):
    """Inspect actual RK states, outside AD; strict tests imply a local branch.

    No finite segment certificate is inferred from endpoint inspections.
    This bounded oracle requires all face fluxes nonzero and all interior
    input slopes nonzero and same-sign ties absent. Opposite-sign slopes give
    a locally constant zero limiter; nominal zero face flow is still rejected.
    Signed and zero log growth use the core's representability checks; growth
    sign is not a minmod or upwind branch switch.
    """
    stages = []
    face_signatures = []
    original = t._euler_minmod

    def inspect(q, qx, qy, *args):
        pairs = ((q[1:-1, 1:-1]-q[1:-1, :-2],
                  q[1:-1, 2:]-q[1:-1, 1:-1]),
                 (q[1:-1, 1:-1]-q[:-2, 1:-1],
                  q[2:, 1:-1]-q[1:-1, 1:-1]))
        scale = q.abs().max()
        if not bool(torch.isfinite(scale)) or bool(scale <= 0):
            raise ValueError("minmod joint oracle left its strict smooth branch")
        slope_tolerance = 128 * torch.finfo(q.dtype).eps * scale
        choices = []
        margins = []
        for left, right in pairs:
            positive = (left > slope_tolerance) & (right > slope_tolerance)
            negative = (left < -slope_tolerance) & (right < -slope_tolerance)
            active = positive | negative
            difference = (left - right).abs()
            if (bool((left.abs() <= slope_tolerance).any())
                    or bool((right.abs() <= slope_tolerance).any())
                    or bool((active & (difference <= slope_tolerance)).any())):
                raise ValueError("minmod joint oracle left its strict smooth branch")
            # The selected operand is min(left, right) for positive slopes and
            # max(left, right) for negative slopes. Record it per cell rather
            # than collapsing the whole grid with .all().
            choose_left = active & torch.where(positive, left < right, left > right)
            slope_sign = torch.where(positive, 1, torch.where(negative, -1, 0))
            choices.append({
                "choose_left": choose_left.tolist(),
                "slope_sign": slope_sign.tolist(),
                "left_sign": torch.sign(left).to(torch.int8).tolist(),
                "right_sign": torch.sign(right).to(torch.int8).tolist(),
            })
            margins.extend((left.abs(), right.abs(), difference))
        flux = torch.cat((qx.flatten(), qy.flatten()))
        flux_scale = flux.abs().max()
        face_tolerance = 128 * torch.finfo(q.dtype).eps * flux_scale
        if bool((flux_scale <= 0) or (flux.abs() <= face_tolerance).any()):
            raise ValueError("minmod joint oracle left its strict smooth branch")
        stages.append((float(torch.stack(margins).min() / scale), choices))
        face_signatures.append({
            "qx": torch.sign(qx).to(torch.int8).tolist(),
            "qy": torch.sign(qy).to(torch.int8).tolist(),
        })
        return original(q, qx, qy, *args)

    with patch.object(t, "_euler_minmod", inspect):
        call()
    if not stages:
        raise RuntimeError("no RK stages inspected")
    return {
        "euler_stages": len(stages),
        "minimum_scaled_slope_margin": min(s[0] for s in stages),
        "choices": [s[1] for s in stages],
        "face_signs": face_signatures,
    }


def run_probe(*, spatial=False, checkpoint=None):
    started = time.monotonic()
    obs, frozen, boundary, support = make_spatial_case() if spatial else make_case()
    shape = frozen.initial_background_dbz.shape
    pattern = torch.linspace(-.2, .3, shape.numel(), dtype=obs.dbz.dtype).reshape(shape)
    p = torch.cat((obs.dbz.flatten(), obs.dbz.new_tensor([.02])))
    report = {
        "status": "running", "shape": list(shape), "substeps_per_interval": 9,
        "scope": "local minmod robust inverse; fixed support/boundaries/precision; synthetic conditional score",
        "finite_path_certified": False, "general_minmod_response_eligible": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(t.__file__), Path(v.__file__),
                                       Path(_oracle.__file__))},
        "timings": {}, "reanalysis": [],
    }

    def save():
        report["elapsed_seconds"] = time.monotonic()-started
        if checkpoint is not None:
            checkpoint.write_text(json.dumps(report, indent=2)+"\n")

    def contract(parameters):
        y, theta = parameters[:-1].reshape_as(obs.dbz), parameters[-1]
        return replace(frozen, initial_background_dbz=y[0]+theta*pattern)

    def objective(c, parameters):
        return v.robust_objective(c, replace(obs, dbz=parameters[:-1].reshape_as(obs.dbz)), contract(parameters))

    def forecast(c, parameters):
        return echo_to_dbz(v.forecast_fv_analysis(
            c, contract(parameters), leads=1, boundary_start_interval=2,
            boundary_echo=boundary, boundary_support=support,
        ).frames_linear[-1], min_dbz=-10.)

    gradient = torch.func.grad(objective)
    seed = v.initial_control(frozen)
    dynamics = [.1, -.08, .06, .04, -.03, .008] if spatial else [.1, .1, .01]
    seed[-len(dynamics):] = seed.new_tensor(dynamics)
    report["controls"] = seed.numel()
    report["initial_branch"] = inspect_branches(lambda: forecast(seed, p))
    report["seed_objective"] = float(objective(seed, p))
    save()
    start_control = seed
    if spatial:
        stamp = time.monotonic()
        gn = v.solve_analysis(obs, contract(p), control=seed)
        report["timings"]["product_gn"] = time.monotonic()-stamp
        report["product_gn"] = {
            "reason": gn.reason, "reference_objective": gn.initial_objective,
            "objective": float(objective(gn.control, p)),
            "gradient_max": float(gradient(gn.control, p).abs().max()),
            "outer_iterations": gn.outer_iterations, "pcg_iterations": gn.pcg_iterations,
            "control": gn.control.tolist(), "default_solver_tolerances": True,
        }
        start_control = gn.control
        save()
        print("Product GN:", report["product_gn"]["reason"], flush=True)

    # Optimization may move between smooth branches. Local response needs a
    # smooth final root, not proof that all optimizer segments stayed in one.
    stamp = time.monotonic()
    control = _oracle.polish(objective, start_control, p)
    report["timings"]["nominal_polish"] = time.monotonic()-stamp
    branches = inspect_branches(lambda: forecast(control, p))
    report.update(control=control.tolist(), stationary_branch=branches,
                  gradient_max=float(gradient(control, p).abs().max()),
                  objective=float(objective(control, p)))
    if spatial:
        report["product_gn"]["objective_gap_to_oracle"] = report["product_gn"]["objective"]-report["objective"]
        report["product_gn"]["control_distance_to_oracle"] = float((gn.control-control).norm())
    save()
    print("Nominal minmod stationary solve complete", flush=True)
    verification = forecast(control, p).detach()+.1*pattern

    def score(c, parameters):
        return (forecast(c, parameters)-verification).square().mean()

    stamp = time.monotonic()
    H = _oracle.dense_hessian(gradient, control, p)
    eigenvalues = torch.linalg.eigvalsh(H)
    torch.linalg.cholesky(H)
    rhs, direct = torch.func.grad(score, argnums=(0, 1))(control, p)
    if not bool(rhs.norm() > 0):
        raise ValueError("response experiment requires a nonzero score gradient")
    adjoint = torch.linalg.solve(H.T, rhs)
    _, pullback = torch.func.vjp(lambda z: gradient(control, z), p)
    sensitivity = direct-pullback(adjoint)[0]
    residual = (H.T@adjoint-rhs).norm()
    report.update(hessian_min_eigenvalue=float(eigenvalues[0]),
                  hessian_condition_number=float(eigenvalues[-1]/eigenvalues[0]),
                  adjoint_relative_residual=float(residual/rhs.norm()),
                  adjoint_backward_error=float(residual/(H.norm()*adjoint.norm()+rhs.norm())),
                  nominal_score=float(score(control, p)))
    report["timings"]["hessian_adjoint"] = time.monotonic()-stamp
    save()
    n = obs.dbz.numel()
    directions = (
        ("observation", torch.cat((torch.sin(torch.arange(n, dtype=p.dtype)), p.new_zeros(1)))),
        ("background_parameter", torch.cat((p.new_zeros(n), p.new_ones(1)))),
    )
    for name, direction in directions:
        for h in (1e-3, 5e-4):
            row = {"direction": name, "h": h,
                   "adjoint": float(torch.dot(sensitivity, direction)), "endpoints": []}
            report["reanalysis"].append(row)
            for sign in (-1, 1):
                stamp = time.monotonic()
                changed = p+sign*h*direction
                try:
                    refined = _oracle.polish(objective, control, changed)
                except RuntimeError as error:
                    report.update(status="failed", failure={
                        "direction": name, "h": h, "sign": sign,
                        "reason": str(error),
                        "elapsed_seconds": time.monotonic()-stamp,
                    })
                    save()
                    raise
                trace = inspect_branches(lambda: forecast(refined, changed))
                same_branch = (trace["choices"], trace["face_signs"]) == (branches["choices"], branches["face_signs"])
                row["endpoints"].append({
                    "sign": sign, "control": refined.tolist(), "score": float(score(refined, changed)),
                    "objective": float(objective(refined, changed)),
                    "gradient_max": float(gradient(refined, changed).abs().max()),
                    "branch": trace, "same_local_branch": same_branch,
                    "elapsed_seconds": time.monotonic()-stamp,
                })
                save()
                if not same_branch:
                    raise ValueError("reanalysis endpoint changed its local branch; no response claim")
            fd = (row["endpoints"][1]["score"]-row["endpoints"][0]["score"])/(2*h)
            row.update(central_reanalysis=fd, absolute_error=abs(fd-row["adjoint"]))
            save()
            print(json.dumps({k: z for k, z in row.items() if k != "endpoints"}), flush=True)
    report["status"] = "complete"
    save()
    return report


def run_spatial_smoke():
    """Run one bounded 4x5 robust/minmod solve for endpoint diagnostics.

    This is a smoke comparison only.  It does not run dense inverse response
    or certify an implicit path; the 26-control dense oracle remains within its
    hard 32-control limit for a later, explicitly bounded experiment.
    """
    started = time.monotonic()
    obs, frozen, boundary, support = make_spatial_case(smoke_only=True)
    seed = v.initial_control(frozen)
    seed[-6:] = seed.new_tensor([0.1, -0.08, 0.06, 0.04, -0.03, 0.008])

    def objective(control):
        return v.robust_objective(control, obs, frozen)

    def forecast(control):
        return v.forecast_fv_analysis(
            control,
            frozen,
            leads=1,
            boundary_start_interval=2,
            boundary_echo=boundary,
            boundary_support=support,
        )

    def endpoint(control):
        started_endpoint = time.monotonic()
        branch = inspect_branches(lambda: forecast(control))
        score = float(objective(control).detach())
        gradient = torch.func.grad(objective)(control)
        return {
            "control": control.tolist(),
            "score": score,
            "gradient_max": float(gradient.abs().max()),
            "branch": branch,
            "elapsed_seconds": time.monotonic() - started_endpoint,
        }

    initial = endpoint(seed)
    solve_started = time.monotonic()
    result = v.solve_analysis(obs, frozen, control=seed)
    final = endpoint(result.control)
    return {
        "scope": "4x5 local minmod smoke only; fixed complete support/boundaries; echo scale 20; 60-second intervals",
        "controls": int(result.control.numel()),
        "substeps_per_interval": 9,
        "detected_observations": bool(obs.detected_mask.all()),
        "initial_endpoint": initial,
        "solve_analysis": {
            "elapsed_seconds": time.monotonic() - solve_started,
            "supplied_seed_used": True,
            "reference_objective_is_default_zero_control": True,
            "supplied_seed_objective": initial["score"],
            "initial_objective": result.initial_objective,
            "final_objective": result.final_objective,
            "selected_gradient_norm": result.selected_gradient_norm,
            "outer_iterations": result.outer_iterations,
            "pcg_iterations": result.pcg_iterations,
            "reason": result.reason,
            "improved": result.improved,
        },
        "final_endpoint": final,
        "elapsed_seconds": time.monotonic() - started,
        "dense_oracle_control_limit": 32,
        "response_oracle_run": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial-smoke", action="store_true")
    parser.add_argument("--spatial", action="store_true")
    args = parser.parse_args()
    result = run_spatial_smoke() if args.spatial_smoke else run_probe(spatial=args.spatial, checkpoint=args.output)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("scope", "elapsed_seconds")}, indent=2))
