"""Bounded local-path diagnostic for the saved 4x5 minmod stationary point.

This probes only the local implicit Newton path. It does not certify a finite
branch segment or relax the response gates; every predictor and polished
endpoint must reproduce all 54 RK-stage limiter and face-sign signatures.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import torch

from advar import variational as v

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "graphify-out/fv-root-cause-20260919/minmod_spatial_inverse.json"
PROBE = ROOT / "examples/weather_scenarios/fv_minmod_inverse_probe.py"
MEASURED_PROBE = ROOT / "graphify-out/fv-root-cause-20260919/minmod_spatial_inverse_measured.py"
ORACLE = ROOT / "examples/weather_scenarios/fv_sensitivity_probe.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_branch(left: dict, right: dict) -> bool:
    return (left["choices"], left["face_signs"]) == (
        right["choices"], right["face_signs"]
    )


def _write(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")


def run(output: Path) -> dict:
    started = time.monotonic()
    probe_path = PROBE
    probe = _load_module(probe_path, "fv_minmod_inverse_for_local_path")
    oracle = _load_module(ORACLE, "fv_dense_oracle_for_local_path")
    saved = json.loads(SAVED.read_text())
    obs, frozen, boundary, support = probe.make_spatial_case()
    shape = frozen.initial_background_dbz.shape
    if tuple(shape) != tuple(saved["shape"]):
        raise ValueError("saved shape does not match current spatial fixture")
    control = torch.tensor(saved["control"], dtype=torch.float64)
    if control.numel() != int(saved["controls"]):
        raise ValueError("saved control dimension does not match report")
    pattern = torch.linspace(-0.2, 0.3, shape.numel(), dtype=obs.dbz.dtype).reshape(shape)
    parameters = torch.cat((obs.dbz.flatten(), obs.dbz.new_tensor([0.02])))

    def contract(p: torch.Tensor):
        y, theta = p[:-1].reshape_as(obs.dbz), p[-1]
        return replace(frozen, initial_background_dbz=y[0] + theta * pattern)

    def objective(c: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return v.robust_objective(
            c, replace(obs, dbz=p[:-1].reshape_as(obs.dbz)), contract(p)
        )

    def forecast(c: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return probe.echo_to_dbz(
            v.forecast_fv_analysis(
                c,
                contract(p),
                leads=1,
                boundary_start_interval=2,
                boundary_echo=boundary,
                boundary_support=support,
            ).frames_linear[-1],
            min_dbz=-10.0,
        )

    gradient = torch.func.grad(objective)
    current_hashes = {
        str(path): _hash(path)
        for path in (probe_path, ROOT / "src/advar/transport.py", ROOT / "src/advar/variational.py", ORACLE)
    }
    saved_hashes = saved.get("source_sha256", {})
    # The archived producer has location-relative imports. Load today's module
    # only after checking the reused fixture/tracer AST against that archive.
    reused_names = {"make_spatial_case", "inspect_branches"}
    def reused_functions(path):
        return {node.name: ast.dump(node) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in reused_names}
    fixture_exact = (reused_functions(PROBE) == reused_functions(MEASURED_PROBE)
                     and _hash(MEASURED_PROBE) == saved_hashes.get(str(PROBE)))
    core_paths = (ROOT / "src/advar/transport.py", ROOT / "src/advar/variational.py", ORACLE)
    source_identity = {
        "saved": saved_hashes,
        "current": current_hashes,
        "core_exact": all(_hash(path) == saved_hashes.get(str(path)) for path in core_paths),
        "fixture_probe_exact": fixture_exact,
    }

    nominal_objective = float(objective(control, parameters))
    nominal_gradient = gradient(control, parameters)
    nominal_forecast = forecast(control, parameters).detach()
    nominal_branch = probe.inspect_branches(lambda: forecast(control, parameters))
    saved_branch = saved.get("stationary_branch", {})
    saved_branch_exact = _same_branch(nominal_branch, saved_branch) if saved_branch else False
    report_time = time.monotonic()
    report: dict = {
        "status": "running",
        "finite_path_certified": False,
        "general_minmod_response_eligible": False,
        "scope": "local predictor/corrector only; fixed minmod support/boundaries/precision",
        "saved_report": str(SAVED),
        "saved_report_sha256": _hash(SAVED),
        "producer_sha256": _hash(Path(__file__)),
        "derivative_relative_tolerance": 1e-4,
        "source_identity": {**source_identity, "fixture_source_used": str(probe_path), "current_fixture_source": str(PROBE), "measured_fixture_available": MEASURED_PROBE.exists()},
        "input_identity": {
            "shape": list(shape),
            "parameters_sha256": hashlib.sha256(parameters.numpy().tobytes()).hexdigest(),
            "controls": int(control.numel()),
            "substeps_per_interval": 9,
            "saved_gradient_max": saved.get("gradient_max"),
            "current_gradient_max": float(nominal_gradient.abs().max()),
            "saved_objective": saved.get("objective"),
            "current_objective": nominal_objective,
            "objective_abs_difference": abs(nominal_objective - float(saved["objective"])),
            "saved_branch_exact": saved_branch_exact,
            "saved_branch_available": bool(saved_branch),
        },
        "direction": {
            "observation": "sin(arange(obs.dbz.numel())) with theta component zero",
            "observation_components": int(obs.dbz.numel()),
            "theta": "final parameter component",
        },
        "nominal_branch": nominal_branch,
        "nominal_control": control.tolist(),
        "hessian": {},
        "tangents": {},
        "pairs": [],
        "elapsed_seconds": 0.0,
    }
    # The saved control is the only starting point; no nominal solve is run.
    if not source_identity["core_exact"] or not source_identity["fixture_probe_exact"]:
        raise ValueError("core or measured fixture source identity mismatch")
    if not torch.isfinite(nominal_gradient).all() or float(nominal_gradient.abs().max()) >= 1e-10:
        raise ValueError("saved/current nominal gradient is not at the unchanged 1e-10 gate")
    if abs(nominal_objective - float(saved["objective"])) > 1e-8:
        raise ValueError("saved/current nominal objective mismatch")
    if not saved_branch_exact:
        raise ValueError("saved/current nominal RK branch signature mismatch")
    _write(output, report)

    H = oracle.dense_hessian(gradient, control, parameters)
    torch.linalg.cholesky(H)
    report["hessian"] = {
        "matrix": H.tolist(),
        "min_eigenvalue": float(torch.linalg.eigvalsh(H)[0]),
        "condition_number": float(torch.linalg.cond(H)),
    }
    directions = {
        "observation_sine60": torch.cat(
            (torch.sin(torch.arange(obs.dbz.numel(), dtype=parameters.dtype)), parameters.new_zeros(1))
        ),
        "theta": torch.cat((parameters.new_zeros(obs.dbz.numel()), parameters.new_ones(1))),
    }
    tangents: dict[str, torch.Tensor] = {}
    crosses: dict[str, torch.Tensor] = {}
    for name, direction in directions.items():
        cross = torch.func.jvp(
            lambda p: gradient(control, p), (parameters,), (direction,)
        )[1]
        tangent = torch.linalg.solve(H, -cross)
        tangents[name] = tangent
        crosses[name] = cross
        report["tangents"][name] = {
            "parameter_direction": direction.tolist(),
            "cross_gradient": cross.tolist(),
            "control_tangent": tangent.tolist(),
            "cross_norm": float(cross.norm()),
            "tangent_norm": float(tangent.norm()),
        }
    verification = nominal_forecast + 0.1 * pattern

    def score(c: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (forecast(c, p) - verification).square().mean()

    rhs, direct = torch.func.grad(score, argnums=(0, 1))(control, parameters)
    adjoint = torch.linalg.solve(H.T, rhs)
    report["adjoint"] = {
        "rhs": rhs.tolist(),
        "direct": direct.tolist(),
        "solution": adjoint.tolist(),
        "relative_residual": float((H.T @ adjoint - rhs).norm() / rhs.norm()),
    }
    report["timings"] = {"setup_and_nominal_seconds": report_time - started, "hessian_and_tangent_seconds": time.monotonic() - report_time}
    _write(output, report)

    response_sensitivity = {
        name: float(direct.dot(direction) - adjoint.dot(crosses[name]))
        for name, direction in directions.items()
    }
    report["adjoint"]["response_sensitivity"] = response_sensitivity
    observation_tangent = tangents["observation_sine60"]
    accepted = 0
    for j in range(12):
        h = 1e-3 * (2.0 ** (-j))
        endpoint_records = []
        predictors_ok = True
        predictor_data = []
        for sign in (-1, 1):
            changed = parameters + sign * h * directions["observation_sine60"]
            predictor = (control + sign * h * observation_tangent).detach()
            predictor_displacement = float((predictor - control).norm())
            try:
                predictor_branch = probe.inspect_branches(lambda: forecast(predictor, changed))
                predictor_same = _same_branch(predictor_branch, nominal_branch)
            except (RuntimeError, ValueError) as error:
                predictor_branch, predictor_same = {"error": str(error)}, False
            predictor_data.append((sign, changed, predictor, predictor_branch, predictor_same, predictor_displacement))
            if not predictor_same:
                predictors_ok = False
        if not predictors_ok:
            endpoint_records = [
                {
                    "sign": sign,
                    "predictor": predictor.tolist(),
                    "predictor_displacement": predictor_displacement,
                    "predictor_branch": predictor_branch,
                    "predictor_same_branch": predictor_same,
                }
                for sign, _changed, predictor, predictor_branch, predictor_same, predictor_displacement in predictor_data
            ]
        else:
            for sign, changed, predictor, predictor_branch, predictor_same, predictor_displacement in predictor_data:
                def check_step(_start: torch.Tensor, trial: torch.Tensor) -> None:
                    trial_branch = probe.inspect_branches(lambda: forecast(trial, changed))
                    if not _same_branch(trial_branch, nominal_branch):
                        raise ValueError("Newton trial changed the nominal RK branch")

                try:
                    refined = oracle.polish(objective, predictor, changed, check_step=check_step)
                    branch = probe.inspect_branches(lambda: forecast(refined, changed))
                    same = _same_branch(branch, nominal_branch)
                    endpoint_records.append({
                        "sign": sign,
                        "predictor": predictor.tolist(),
                        "predictor_displacement": predictor_displacement,
                        "predictor_branch": predictor_branch,
                        "predictor_same_branch": True,
                        "control": refined.tolist(),
                        "objective": float(objective(refined, changed)),
                        "gradient_max": float(gradient(refined, changed).abs().max()),
                        "score": float(score(refined, changed)),
                        "branch": branch,
                        "same_local_branch": same,
                    })
                    if not same:
                        predictors_ok = False
                except (RuntimeError, ValueError) as error:
                    predictors_ok = False
                    endpoint_records.append({
                        "sign": sign,
                        "predictor": predictor.tolist(),
                        "predictor_displacement": predictor_displacement,
                        "predictor_branch": predictor_branch,
                        "predictor_same_branch": True,
                        "error": str(error),
                    })
        pair = {"j": j, "h": h, "predictors_ok": predictors_ok, "endpoints": endpoint_records}
        if len(endpoint_records) == 2 and all("score" in item for item in endpoint_records):
            minus, plus = endpoint_records
            central = (plus["score"] - minus["score"]) / (2.0 * h)
            predicted = response_sensitivity["observation_sine60"]
            pair.update({
                "central_reanalysis": central,
                "response_sensitivity": predicted,
                "absolute_error": abs(central - predicted),
                "relative_error_actual_denominator": abs(central - predicted) / abs(central) if central else None,
                "relative_error_adjoint_denominator": abs(central - predicted) / abs(predicted) if predicted else None,
                "derivative_pass": bool(math.isfinite(central) and predicted != 0 and
                                        abs(central - predicted) <= 1e-4 * abs(predicted)),
            })
        report["pairs"].append(pair)
        _write(output, report)
        if predictors_ok and len(endpoint_records) == 2 and all(
            item.get("same_local_branch", False) for item in endpoint_records
        ):
            accepted += 1
        else:
            accepted = 0
        if accepted >= 2:
            report["status"] = "complete"
            report["accepted_successive_pairs"] = accepted
            break
    else:
        report["status"] = "incomplete"
        report["accepted_successive_pairs"] = accepted
    report["elapsed_seconds"] = time.monotonic() - started
    _write(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ("status", "accepted_successive_pairs", "elapsed_seconds") if k in result}, indent=2))
