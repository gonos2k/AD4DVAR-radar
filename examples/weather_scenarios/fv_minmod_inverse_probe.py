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


def inspect_branches(call):
    """Inspect actual RK states, outside AD; strict tests imply a local branch.

    No finite segment certificate is inferred from endpoint inspections.
    This bounded oracle requires all face fluxes nonzero and all interior
    slopes positive and unequal, so no nominal-zero relevance shortcut is used.
    """
    stages = []
    original = t._euler_minmod
    original_step = t.finite_volume_step

    def inspect_growth(*args, **kwargs):
        growth = kwargs["log_growth"]
        if not bool(growth > 128*torch.finfo(growth.dtype).eps):
            raise ValueError("minmod joint oracle requires strict positive growth")
        return original_step(*args, **kwargs)

    def inspect(q, qx, qy, *args):
        pairs = ((q[1:-1, 1:-1]-q[1:-1, :-2],
                  q[1:-1, 2:]-q[1:-1, 1:-1]),
                 (q[1:-1, 1:-1]-q[:-2, 1:-1],
                  q[2:, 1:-1]-q[1:-1, 1:-1]))
        scale = q.abs().max()
        margins = torch.stack([z.min()/scale for a, b in pairs
                               for z in (a, b, (a-b).abs())])
        flux = torch.cat((qx.flatten(), qy.flatten()))
        tolerance = 128*torch.finfo(q.dtype).eps
        if bool((flux <= tolerance*flux.abs().max()).any()) or bool((margins <= tolerance).any()):
            raise ValueError("minmod joint oracle left its strict smooth branch")
        stages.append((float(margins.min()), tuple(bool((a < b).all()) for a, b in pairs)))
        return original(q, qx, qy, *args)

    with patch.object(t, "_euler_minmod", inspect), patch.object(t, "finite_volume_step", inspect_growth):
        call()
    if not stages:
        raise RuntimeError("no RK stages inspected")
    return {"euler_stages": len(stages), "minimum_scaled_slope_margin": min(s[0] for s in stages),
            "choices": [list(s[1]) for s in stages]}


def run_probe():
    started = time.monotonic()
    obs, frozen, boundary, support = make_case()
    theta = obs.dbz.new_tensor(.02)
    pattern = torch.linspace(-.2, .3, 9, dtype=obs.dbz.dtype).reshape(3, 3)

    def contract(p):
        y, parameter = p[:-1].reshape_as(obs.dbz), p[-1]
        return replace(frozen, initial_background_dbz=y[0]+parameter*pattern)

    p = torch.cat((obs.dbz.flatten(), theta[None]))

    def objective(c, p):
        return v.robust_objective(c, replace(obs, dbz=p[:-1].reshape_as(obs.dbz)), contract(p))

    def forecast(c, p):
        return echo_to_dbz(v.forecast_fv_analysis(
            c, contract(p), leads=1, boundary_start_interval=2,
            boundary_echo=boundary, boundary_support=support,
        ).frames_linear[-1], min_dbz=-10.)

    # A declared nonzero local seed, not the generating truth control.
    seed = v.initial_control(frozen)
    seed[-3:] = seed.new_tensor([.1, .1, .01])
    initial_branch = inspect_branches(lambda: forecast(seed, p))

    def check_step(start, stop, parameters):
        before = inspect_branches(lambda: forecast(start, parameters))
        after = inspect_branches(lambda: forecast(stop, parameters))
        if before["choices"] != after["choices"]:
            raise ValueError("Newton endpoints changed limiter choices")

    control = _oracle.polish(objective, seed, p,
                            check_step=lambda a, b: check_step(a, b, p))
    branches = inspect_branches(lambda: forecast(control, p))
    if initial_branch["choices"] != branches["choices"]:
        raise ValueError("refinement changed limiter choices")
    print("Nominal minmod stationary solve complete", flush=True)
    verification = forecast(control, p).detach() + .1*pattern

    def score(c, p):
        return (forecast(c, p)-verification).square().mean()

    gradient = torch.func.grad(objective)
    H = _oracle.dense_hessian(gradient, control, p)
    rhs, direct = torch.func.grad(score, argnums=(0, 1))(control, p)
    adjoint = torch.linalg.solve(H.T, rhs)
    _, pullback = torch.func.vjp(lambda z: gradient(control, z), p)
    sensitivity = direct-pullback(adjoint)[0]
    rows = []
    for name, direction in (("observation", torch.cat((torch.sin(torch.arange(27, dtype=p.dtype)), p.new_zeros(1)))),
                            ("background_parameter", torch.cat((p.new_zeros(27), p.new_ones(1))))):
        predicted = torch.dot(sensitivity, direction)
        for h in (1e-3, 5e-4):
            values = []
            for sign in (-1, 1):
                changed = p+sign*h*direction
                refined = _oracle.polish(objective, control, changed,
                                        check_step=lambda a, b: check_step(a, b, changed))
                trace = inspect_branches(lambda: forecast(refined, changed))
                if trace["choices"] != branches["choices"]:
                    raise ValueError("perturbed solution changed RK limiter branches")
                values.append(float(score(refined, changed)))
            fd = (values[1]-values[0])/(2*h)
            rows.append({"direction": name, "h": h, "adjoint": float(predicted),
                         "central_reanalysis": fd, "absolute_error": abs(fd-float(predicted))})
            print(json.dumps(rows[-1]), flush=True)
    return {"scope": "3x3 local minmod robust inverse; fixed complete support/boundaries; dense oracle",
            "controls": control.numel(), "substeps_per_interval": 9,
            "initial_branch": initial_branch, "stationary_branch": branches,
            "gradient_max": float(gradient(control, p).abs().max()),
            "hessian_min_eigenvalue": float(torch.linalg.eigvalsh(H).min()),
            "adjoint_relative_residual": float((H.T@adjoint-rhs).norm()/rhs.norm()),
            "reanalysis": rows, "elapsed_seconds": time.monotonic()-started,
            "control": control.tolist(), "general_minmod_response_eligible": False,
            "finite_path_certified": False,
            "environment": {"python": platform.python_version(), "torch": torch.__version__},
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__), Path(t.__file__), Path(v.__file__),
                                           Path(_oracle.__file__))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe()
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
