"""One fixed-branch tangent correction for the original R2 point input."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

import torch
from torch import Tensor

from advar import matrix_free, variational as v
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from advar.transport import observe_minmod_stages
from examples.weather_scenarios import fv_point_event_diagnostic as event
from examples.weather_scenarios.fv_active_face_stationarity_probe import ActiveFaceRefusal, _slope_choices
from examples.weather_scenarios.fv_partial_signed_face_slices import _true_residual_monitor
from examples.weather_scenarios.fv_point_active_face_binding import FVPointActiveFaceBinding

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "graphify-out/fv-root-cause-20260919"
EVENT = OUT / "point_event_diagnostic_attempt1/event.json"
PLAN = OUT / "R2_POINT_ACTIVE_FACE_STATIONARITY_PLAN_20261003.md"
EVENT_SHA256 = "fde001e2f2de78f2a812437c7f485826621a0f7004060c19a0e57788901b82b9"
PLAN_SHA256 = "1887d997f68305bb9a5ebe156c2d89c506edeef5d4d3bceefe06b31b2a3fd278"
SOURCE_PATHS = tuple(dict.fromkeys((*event.merit.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_event_diagnostic.py",
    "examples/weather_scenarios/fv_point_active_face_binding.py",
    "examples/weather_scenarios/fv_point_active_face_stationarity.py",
    "examples/weather_scenarios/fv_active_face_stationarity_probe.py",
    "examples/weather_scenarios/fv_partial_signed_face_slices.py",
    "examples/weather_scenarios/fv_face_flux_coordinates.py")))


class PointFaceRefusal(ValueError):
    """A known numerical branch or restricted-curvature qualification refusal."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_record(q: Tensor, qx: Tensor, qy: Tensor, index: int) -> tuple[dict[str, Any], float, float]:
    if (q.shape != (4, 5) or qx.shape != (4, 6) or qy.shape != (5, 5)
            or not all(bool(torch.isfinite(a).all()) for a in (q, qx, qy))):
        raise RuntimeError("point active-face stage violates its fixed shape/finiteness contract")
    if not torch.equal(qx[3, 4], qx.new_zeros(())):
        raise RuntimeError("point selected structural zero was not preserved")
    flux = torch.cat((qx.flatten(), qy.flatten()))
    other = torch.cat((flux[:22], flux[23:])).abs()
    scale = flux.abs().max()
    if not bool(scale > 0) or bool((other <= 128 * torch.finfo(q.dtype).eps * scale).any()):
        raise PointFaceRefusal("nonselected point face is zero or unresolved")
    face_margin = float(other.min() / scale)
    try:
        choices, slope_margin = _slope_choices(q)
    except ActiveFaceRefusal as error:
        raise PointFaceRefusal(str(error)) from error
    if min(face_margin, slope_margin) <= 1e-4:
        raise PointFaceRefusal("point nonselected face/limiter lacks the original 1e-4 margin")
    return {"step": index // 2, "stage": index % 2, **choices,
            "qx_sign": qx.sign().to(torch.int8).tolist(),
            "qy_sign": qy.sign().to(torch.int8).tolist()}, slope_margin, face_margin


def trace(binding: FVPointActiveFaceBinding, control: Tensor, p: Tensor) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        binding.lift(control)
    except RuntimeError as error:
        if str(error) != "face coordinate is outside the strict representable chart domain":
            raise
        raise PointFaceRefusal(str(error)) from error
    contract = binding.reduced_problem.contract(p)
    cfg, analysis = contract.nowcast_config, contract.analysis_config
    background = contract.initial_background_dbz
    offset = (background - cfg.min_dbz) / analysis.echo_transform_scale_dbz
    margin = 64 * torch.finfo(p.dtype).eps * (
        (background.abs() + abs(cfg.min_dbz)) / analysis.echo_transform_scale_dbz
        + analysis.transform_epsilon)
    if not bool(((offset - analysis.transform_epsilon > margin) & (background < cfg.max_dbz)).all()):
        raise PointFaceRefusal("point background left the fixed smooth transform branch")
    analysis_stages: list[tuple[Tensor, Tensor, Tensor]] = []
    forecast_stages: list[tuple[Tensor, Tensor, Tensor]] = []
    with observe_minmod_stages(lambda q, x, y: analysis_stages.append((q, x, y))):
        v.analysis_trajectory(control, contract)
    with observe_minmod_stages(lambda q, x, y: forecast_stages.append((q, x, y))):
        binding.forecast(control, p)
    if len(analysis_stages) != 36 or len(forecast_stages) != 54:
        raise RuntimeError("point analysis/forecast stage count mismatch")
    if any(not all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
           for left, right in zip(analysis_stages, forecast_stages[:36], strict=True)):
        raise RuntimeError("point objective analysis differs from the forecast replay")
    rows, slopes, faces = [], [], []
    for i, (q, x, y) in enumerate(forecast_stages):
        row, slope, face = stage_record(q, x, y, i)
        rows.append(row)
        slopes.append(slope)
        faces.append(face)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"signature_sha256": digest, "euler_stages": 54,
            "minimum_scaled_slope_margin": min(slopes),
            "minimum_scaled_other_face_margin": min(faces)}, rows


def curvature(objective: Any, control: Tensor, p: Tensor) -> dict[str, Any]:
    gradient = torch.func.grad(objective)
    columns = [torch.func.jvp(lambda c: gradient(c, p), (control,), (d,))[1]
               for d in torch.eye(25, dtype=control.dtype)]
    h = torch.stack(columns, dim=1)
    if h.shape != (25, 25):
        raise ValueError("point restricted Hessian shape is malformed")
    if not bool(torch.isfinite(h).all()):
        raise PointFaceRefusal("point restricted Hessian is nonfinite")
    scale = h.abs().max()
    normalized = h / scale if float(scale) > 0 else h
    norm = torch.linalg.vector_norm(normalized)
    symmetry = float(torch.linalg.vector_norm(normalized-normalized.T) / norm) if float(norm) > 0 else 0.0
    eigen = torch.linalg.eigvalsh(.5 * (normalized+normalized.T)) * scale
    if not bool(torch.isfinite(eigen).all()):
        raise PointFaceRefusal("point restricted eigenvalues exceed the representable scale")
    floor = 128 * torch.finfo(h.dtype).eps * max(float(eigen.abs().max()), torch.finfo(h.dtype).tiny)
    qualified = float(eigen[0]) > floor and symmetry <= 128 * torch.finfo(h.dtype).eps * 25
    return {"hessian": h.tolist(), "eigenvalues": eigen.tolist(), "symmetry_relative": symmetry,
            "positive_curvature_floor": floor, "qualified": qualified,
            "scope": "numerical restricted25 curvature; not full26/global curvature"}


def audit_solves(objective: Any, p: Tensor, solves: list[dict[str, Any]], signature: str | None) -> None:
    gradient = torch.func.grad(objective)
    for row in solves:
        if (row.get("eta") != 0.0 or row.get("input_signature") != signature
                or row.get("rtol") != 1e-10 or row.get("max_iterations") != 104):
            raise ValueError("saved point Newton solve changed its branch or numerical budget")
        if row.get("error") is not None:
            continue
        if (not isinstance(row.get("converged"), bool) or type(row.get("iterations")) is not int
                or not 0 <= row["iterations"] <= 104):
            raise ValueError("saved point PCG convergence metadata is malformed")
        c, rhs, step = (torch.tensor(row[k], dtype=torch.float64)
                        for k in ("input_tangent", "rhs", "solution"))
        if any(a.shape != (25,) or not bool(torch.isfinite(a).all()) for a in (c, rhs, step)):
            raise ValueError("saved point Newton vectors are malformed")
        g = gradient(c, p)
        if not torch.equal(rhs, -g):
            raise ValueError("saved point Newton RHS differs from fresh gradient")
        r = torch.func.jvp(lambda z: gradient(z, p), (c,), (step,))[1] + g
        scale = torch.linalg.vector_norm(g)
        if not bool(torch.isfinite(scale) & (scale > 0)):
            raise ValueError("saved point Newton solve has no finite positive gradient norm")
        relative = float(torch.linalg.vector_norm(r) / scale)
        if not bool(torch.isfinite(r).all()) or not math.isfinite(relative) or relative > 1e-10:
            raise ValueError("fresh point Newton residual exceeds original tolerance")
        row["post_solve_fresh_relative_residual"] = relative


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("point active-face output must be fresh")
    if sha(EVENT) != EVENT_SHA256 or sha(PLAN) != PLAN_SHA256:
        raise ValueError("point active-face event/plan differs from the declared bytes")
    saved = json.loads(EVENT.read_text())
    if not saved.get("single_face_event_candidate") or saved.get("response_validation") != "not_performed":
        raise ValueError("point event diagnosis is not the declared candidate")
    event_sha, plan_sha = sha(EVENT), sha(PLAN)
    source_before = {name: sha(ROOT/name) for name in SOURCE_PATHS}
    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"}
    if environment != saved["runtime"]:
        raise ValueError("point active-face runtime differs from diagnosed points")
    problem, warm, p, direction = event.preflight.fixed_problem()
    input_before = event.preflight._input_identity(problem, warm, p, direction)
    if input_before != saved["input_before"]:
        raise ValueError("point active-face statistical input changed")
    original_control = torch.tensor(saved["points"][0]["control"], dtype=torch.float64)
    if event._tensor_sha(original_control) != event.ACCEPTED_SHA:
        raise ValueError("point active-face start is not the declared historical control")
    binding = FVPointActiveFaceBinding(problem)
    # Original-vector conversion is preflight: without a representable chart
    # coordinate there is no numerical correction or tangent result to report.
    start = binding.tangent_coordinates(original_control)
    fixed_p = p.clone()
    state: dict[str, Any] = {"tangent": start.clone(), "signature": None}
    solves: list[dict[str, Any]] = []; trials: list[dict[str, Any]] = []
    last = start.clone()

    def branch_check(c: Tensor, parameters: Tensor):
        if not torch.equal(parameters, fixed_p):
            raise RuntimeError("point refiner changed fixed observations/background parameter")
        branch, _ = trace(binding, c, parameters)
        if branch["signature_sha256"] != state["signature"]:
            raise PointFaceRefusal("point nonselected branch changed in fixed-face correction")
        state["tangent"] = c.detach().clone()
        return branch, "fixed qx[3,4]=0; all other point branches strict"

    def observer(row: dict[str, Any]):
        nonlocal last
        trials.append(row)
        if row.get("accepted"):
            last = torch.tensor(row["candidate_control"], dtype=torch.float64)
        report["last_accepted_tangent"] = last.tolist()
        save()

    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "running", "numerical_status": "running",
        "source_before": source_before, "input_before": input_before, "runtime": environment,
        "event_sha256": event_sha, "plan_sha256": plan_sha, "parameters_sha256": event._tensor_sha(p),
        "start_tangent": start.tolist(),
        "full_root_claim": False, "normal_validation": "not_performed",
        "response_validation": "not_performed", "physical_validated": False,
        "linear_solves": solves, "trials": trials}

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
        temporary.replace(output)

    save()
    try:
        start_branch, start_trace = trace(binding, start, p)
        state["signature"] = start_branch["signature_sha256"]
        report.update(start_control=binding.lift(start).tolist(), start_branch=start_branch,
            start_trace=start_trace, start_gradient=torch.func.grad(binding.objective)(start, p).tolist(),
            start_objective=float(binding.objective(start, p)))
        report["start_curvature"] = curvature(binding.objective, start, p)
        if not report["start_curvature"]["qualified"]:
            raise PointFaceRefusal("point initial tangent curvature is not numerically SPD")
        with matrix_free.observe_pcg_calls(_true_residual_monitor(solves, 0.0, state)):
            result = refine_stationary(binding.objective, start, p, branch_check=branch_check,
                max_iterations=4, max_backtracks=16, pcg_max_iterations=104, trial_observer=observer)
        last = result.control.clone()
        final_branch, final_trace = trace(binding, last, p)
        g = torch.func.grad(binding.objective)(last, p)
        if float(g.abs().max()) >= 1e-10 or not torch.isfinite(g).all():
            raise PointFaceRefusal("fresh point tangent gradient exceeds original tolerance")
        report["final_curvature"] = curvature(binding.objective, last, p)
        if not report["final_curvature"]["qualified"]:
            raise PointFaceRefusal("point final tangent curvature is not numerically SPD")
        report.update(numerical_status="tangent_stationary_candidate", tangent_gradient=g.tolist(),
            tangent_gradient_max=float(g.abs().max()), newton_iterations=result.iterations,
            refiner_hvp_count=result.hvp_count, final_branch=final_branch, final_trace=final_trace,
            final_objective=float(binding.objective(last,p)))
    except RefinementNumericalRefusal as error:
        report.update(numerical_status="numerical_refusal", reason=str(error))
    except PointFaceRefusal as error:
        report.update(numerical_status="branch_or_curvature_refusal", reason=str(error))
    audit_solves(binding.objective, p, solves, state["signature"])
    source_after = {name: sha(ROOT/name) for name in SOURCE_PATHS}
    input_after = event.preflight._input_identity(problem, warm, p, direction)
    runtime_after = {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"}
    if (source_after != source_before or input_after != input_before or not torch.equal(p, fixed_p)
            or runtime_after != environment or sha(EVENT) != event_sha or sha(PLAN) != plan_sha):
        raise ValueError("point active-face source/input changed during correction")
    try:
        last_control = binding.lift(last).tolist()
    except RuntimeError as error:
        if (str(error) != "face coordinate is outside the strict representable chart domain"
                or report["numerical_status"] != "branch_or_curvature_refusal"):
            raise
        last_control = None
    report.update(phase="finished", source_after=source_after, input_after=input_after,
        source_unchanged=True, input_unchanged=True, last_accepted_tangent=last.tolist(),
        last_accepted_control=last_control, runtime_after=runtime_after)
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
