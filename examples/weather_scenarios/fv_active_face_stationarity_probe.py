"""Branch-local tangent refinement on the structural q_y[2,0]=0 face."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import torch
from torch import Tensor

from advar import matrix_free, variational as v
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from advar.transport import observe_minmod_stages
from examples.weather_scenarios import fv_partial_signed_face_slices as slices
from examples.weather_scenarios.fv_active_face_binding import FVActiveFaceBinding
from tests.test_fv_research_partial_observation import _problem


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
RAW = EVIDENCE / "partial_signed_face_slices_attempt1/signed_face_slices.json"
RAW_SHA256 = "2e331247833dcf9cda71891bc99283551e5490bb9701fbde3b29eb18fcfbaee9"
CORRECTION = EVIDENCE / "partial_slice_interval_correction_attempt1/correction.json"
CORRECTION_SHA256 = "375407c927296a58f59bb6d0286bfb1a2f11a3047a8b3978a113e2ca04476858"
PLAN = EVIDENCE / "R4_ACTIVE_FACE_STATIONARITY_PLAN_20261003.md"
SELF_PATH = "examples/weather_scenarios/fv_active_face_stationarity_probe.py"
BINDING_PATH = "examples/weather_scenarios/fv_active_face_binding.py"
STATIONARITY_TOLERANCE = 1e-10
TRUE_RESIDUAL_TOLERANCE = 1e-10
PCG_MAX_ITERATIONS = 80
MAX_BACKTRACKS = 16
MAX_NEWTON_ITERATIONS = 2
FACE_MARGIN_THRESHOLD = 1e-4


class ActiveFaceRefusal(ValueError):
    """The selected zero face or another branch violates its fixed contract."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slope_choices(q: Tensor) -> tuple[dict[str, object], float]:
    scale = q.abs().max()
    if not bool(torch.isfinite(scale)) or float(scale) <= 0:
        raise ActiveFaceRefusal("minmod state has no finite positive scale")
    tolerance = 128 * torch.finfo(q.dtype).eps * scale
    choices = []
    margins = []
    for left, right in (
        (q[1:-1, 1:-1] - q[1:-1, :-2], q[1:-1, 2:] - q[1:-1, 1:-1]),
        (q[1:-1, 1:-1] - q[:-2, 1:-1], q[2:, 1:-1] - q[1:-1, 1:-1]),
    ):
        positive = (left > tolerance) & (right > tolerance)
        negative = (left < -tolerance) & (right < -tolerance)
        active = positive | negative
        difference = (left - right).abs()
        if (bool((left.abs() <= tolerance).any())
                or bool((right.abs() <= tolerance).any())
                or bool((active & (difference <= tolerance)).any())):
            raise ActiveFaceRefusal("another limiter input is zero or tied")
        choices.append({
            "choose_left": (active & torch.where(positive, left < right, left > right)).tolist(),
            "slope_sign": torch.where(positive, 1, torch.where(negative, -1, 0)).tolist(),
            "left_sign": torch.sign(left).to(torch.int8).tolist(),
            "right_sign": torch.sign(right).to(torch.int8).tolist(),
        })
        margins.extend((left.abs(), right.abs(), difference))
    return {"x": choices[0], "y": choices[1]}, float(torch.stack(margins).min() / scale)


def _inspect_stage(q: Tensor, qx: Tensor, qy: Tensor, step: int, stage: int) -> tuple[dict[str, Any], float, float]:
    if q.shape != (4, 5) or qx.shape != (4, 6) or qy.shape != (5, 5):
        raise ActiveFaceRefusal("observer stage does not match the fixed 4x5 transport layout")
    face = qy[2, 0]
    if not torch.equal(face, torch.zeros_like(face)):
        raise ActiveFaceRefusal("selected q_y[2,0] face is not an exact structural zero")
    flux = torch.cat((qx.flatten(), qy.flatten()))
    selected = qx.numel() + 2 * qy.shape[1]
    other = torch.cat((flux[:selected], flux[selected + 1:])).abs()
    flux_scale = flux.abs().max()
    if not bool(torch.isfinite(flux_scale)) or float(flux_scale) <= 0:
        raise ActiveFaceRefusal("non-selected face-flux scale is not positive finite")
    tolerance = 128 * torch.finfo(q.dtype).eps * flux_scale
    if bool((other <= tolerance).any()):
        raise ActiveFaceRefusal("a non-selected face flux is zero or unresolved")
    face_margin = float(other.min() / flux_scale)
    if face_margin <= FACE_MARGIN_THRESHOLD:
        raise ActiveFaceRefusal("another face lacks the fixed 1e-4 scaled margin")
    choices, slope_margin = _slope_choices(q)
    if slope_margin <= FACE_MARGIN_THRESHOLD:
        raise ActiveFaceRefusal("another limiter input lacks the fixed 1e-4 scaled margin")
    row = {"step": step, "stage": stage, **choices,
           "qx_sign": torch.sign(qx).to(torch.int8).tolist(),
           "qy_sign": torch.sign(qy).to(torch.int8).tolist()}
    return row, slope_margin, face_margin


def _trace_face(binding: FVActiveFaceBinding, tangent: Tensor, parameters: Tensor) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding.lift(tangent)
    _, _, reduced_frozen = binding._fixed_state(parameters)
    analysis_rows = []
    with observe_minmod_stages(lambda q, qx, qy: analysis_rows.append((q.clone(), qx.clone(), qy.clone()))):
        v._analysis_trajectory(tangent, reduced_frozen)
    forecast_rows = []
    with observe_minmod_stages(lambda q, qx, qy: forecast_rows.append((q.clone(), qx.clone(), qy.clone()))):
        binding.forecast(tangent, parameters)
    if len(analysis_rows) != 36 or len(forecast_rows) != 54:
        raise ActiveFaceRefusal("expected 36 analysis and 54 analysis-plus-forecast stages")
    if any(not (torch.equal(a[0], f[0]) and torch.equal(a[1], f[1]) and torch.equal(a[2], f[2]))
           for a, f in zip(analysis_rows, forecast_rows[:36], strict=True)):
        raise ActiveFaceRefusal("analysis observer differs from forecast's analysis replay")
    trace, slope_margins, face_margins = [], [], []
    for index, (q, qx, qy) in enumerate(analysis_rows + forecast_rows[36:]):
        row, slope, face = _inspect_stage(q, qx, qy, index // 2, index % 2)
        trace.append(row)
        slope_margins.append(slope)
        face_margins.append(face)
    if min(face_margins) <= FACE_MARGIN_THRESHOLD:
        raise ActiveFaceRefusal("another face lacks the fixed 1e-4 scaled margin")
    digest = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ({"signature_sha256": digest, "euler_stages": len(trace),
             "minimum_scaled_slope_margin": min(slope_margins),
             "minimum_scaled_face_flux_margin": min(face_margins),
             "selected_face": "qy[2,0]=0"}, trace)


def _audit_linear_solves(binding: FVActiveFaceBinding, parameters: Tensor,
                         solves: list[dict[str, Any]], signature: str) -> int:
    gradient = torch.func.grad(binding.objective)
    audited = 0
    for row in solves:
        if (row.get("eta") != 0.0 or row.get("input_signature") != signature
                or row.get("error") is not None):
            raise ActiveFaceRefusal("Newton solve is not bound to the zero-face branch")
        tangent = torch.tensor(row["input_tangent"], dtype=torch.float64)
        rhs = torch.tensor(row["rhs"], dtype=torch.float64)
        step = torch.tensor(row["solution"], dtype=torch.float64)
        saved = torch.tensor(row["independently_recomputed_residual"], dtype=torch.float64)
        if (tangent.shape != (25,) or rhs.shape != (25,) or step.shape != (25,) or saved.shape != (25,)
                or not all(bool(torch.isfinite(x).all()) for x in (tangent, rhs, step, saved))
                or row["max_iterations"] != PCG_MAX_ITERATIONS or row["rtol"] != TRUE_RESIDUAL_TOLERANCE
                or not isinstance(row.get("iterations"), int)
                or not 0 <= row["iterations"] <= PCG_MAX_ITERATIONS):
            raise ActiveFaceRefusal("saved Newton PCG vectors or budget are malformed")
        branch, _ = _trace_face(binding, tangent, parameters)
        if branch["signature_sha256"] != signature:
            raise ActiveFaceRefusal("saved Newton solve input is outside the active-face branch")
        fresh_gradient = gradient(tangent, parameters)
        fresh_rhs = -fresh_gradient
        rhs_scale = torch.linalg.vector_norm(fresh_gradient)
        if not bool(torch.isfinite(rhs_scale)) or float(rhs_scale) <= 0:
            raise ActiveFaceRefusal("Newton RHS has no positive finite norm")
        if float(torch.linalg.vector_norm(rhs - fresh_rhs)) > 128 * torch.finfo(rhs.dtype).eps * float(rhs_scale):
            raise ActiveFaceRefusal("saved Newton RHS differs from fresh active-face gradient")
        hessian_step = torch.func.jvp(lambda z: gradient(z, parameters), (tangent,), (step,))[1]
        residual = hessian_step + fresh_gradient
        relative = torch.linalg.vector_norm(residual) / rhs_scale
        residual_scale = torch.linalg.vector_norm(hessian_step) + rhs_scale
        if (float(torch.linalg.vector_norm(residual - saved))
                > 128 * torch.finfo(rhs.dtype).eps * float(residual_scale)
                or float(relative) > TRUE_RESIDUAL_TOLERANCE
                or row.get("converged") is not True):
            raise ActiveFaceRefusal("independent active-face PCG residual audit failed")
        audited += 1
    return audited


def _check_fixed_branches(problem: Any, parameters: Tensor) -> None:
    if parameters.shape != (61,) or not bool(torch.isfinite(parameters).all()):
        raise ActiveFaceRefusal("fixed parameters must be a finite 61-vector")
    background = problem.contract(parameters).initial_background_dbz
    cfg = problem.frozen.nowcast_config
    analysis = problem.frozen.analysis_config
    scale = analysis.echo_transform_scale_dbz
    offset = (background - cfg.min_dbz) / scale
    margin = 64 * torch.finfo(background.dtype).eps * (
        (background.abs() + abs(cfg.min_dbz)) / scale + analysis.transform_epsilon
    )
    if not bool(((offset - analysis.transform_epsilon > margin)
                 & (background < cfg.max_dbz)).all()):
        raise ActiveFaceRefusal("fixed background is outside the smooth transform branch")
    active = problem._observation_values(parameters)[problem.observations.valid_mask]
    if not bool(((active > analysis.detection_limit_dbz) & (active < cfg.max_dbz)).all()):
        raise ActiveFaceRefusal("fixed observations do not retain the detected observation branch")


def main(output: Path) -> None:
    slices._fresh_path(output)
    _source_guard = (RAW_SHA256, CORRECTION_SHA256)
    if _sha(RAW) != _source_guard[0] or _sha(CORRECTION) != _source_guard[1]:
        raise slices.SliceIdentityRefusal("pinned signed-slice or interval-correction record changed")
    raw, correction = json.loads(RAW.read_text()), json.loads(CORRECTION.read_text())
    slices._validate_runtime(raw["environment"])
    if (correction.get("status") != "tangent_stationary_candidate"
            or correction.get("historical_trial_accepted") is not False):
        raise slices.SliceIdentityRefusal("new +2 slice correction is not the pinned start endpoint")
    plan_sha = _sha(PLAN)
    original_sources = slices._sources()
    if original_sources != raw["source_after"]:
        raise slices.SliceIdentityRefusal("original 25 measured source paths changed")
    diagnostic_paths = (SELF_PATH, BINDING_PATH)
    diagnostic_sources = {path: _sha(ROOT / path) for path in diagnostic_paths}
    problem, parameters, identity = slices._current_problem()
    fixed_parameters = parameters.clone()
    if (identity != correction["input_identity"]
            or slices._tensor_sha(parameters) != correction["final_endpoint"]["parameters_sha256"]):
        raise slices.SliceIdentityRefusal("current problem/parameters differ from pinned corrected endpoint")
    if parameters.dtype != torch.float64 or parameters.device.type != "cpu":
        raise slices.SliceIdentityRefusal("current fixed parameters are not CPU FP64")
    _check_fixed_branches(problem, parameters)
    binding = FVActiveFaceBinding(problem)
    correction_control = torch.tensor(correction["final_endpoint"]["control"], dtype=torch.float64)
    if slices._tensor_sha(correction_control) != correction["final_endpoint"]["control_sha256"]:
        raise slices.SliceIdentityRefusal("corrected start control/hash mismatch")
    tangent = binding.tangent_coordinates(correction_control)
    lifted_start = binding.lift(tangent)
    state: dict[str, Any] = {"tangent": tangent.clone(), "signature": None}
    start_branch, start_trace = _trace_face(binding, tangent, parameters)
    state["signature"] = start_branch["signature_sha256"]
    state["start_trace"] = start_trace

    def branch_check(value: Tensor, p: Tensor):
        if not torch.equal(p, fixed_parameters):
            raise ActiveFaceRefusal("parameters changed during active-face refinement")
        _check_fixed_branches(problem, p)
        branch, _ = _trace_face(binding, value, p)
        if branch["signature_sha256"] != state["signature"]:
            raise ActiveFaceRefusal("active-face trace changed within the refinement")
        state["tangent"] = value.detach().clone()
        return branch, "fixed q_y[2,0]=0 with all other faces/limiters strict"

    trials: list[dict[str, Any]] = []
    solves: list[dict[str, Any]] = []
    last_accepted = tangent.clone()

    def observe_trial(row: dict[str, Any]) -> None:
        nonlocal last_accepted
        trials.append({key: row[key] for key in (
            "iteration", "backtrack", "step_scale", "accepted", "rejection", "branch_reason",
            "objective", "gradient_max", "armijo_ratio", "linear_relative_residual", "hvp_count",
        ) if key in row})
        if row.get("accepted") is True:
            last_accepted = torch.tensor(row["candidate_control"], dtype=torch.float64)

    report: dict[str, Any] = {
        "scope": "branch-local active-face stationarity diagnostic; no full-root/minimum/response claim",
        "plan_sha256": plan_sha, "original_source_before": original_sources,
        "diagnostic_source_before": diagnostic_sources, "input_identity": identity,
        "parameters_sha256": slices._tensor_sha(parameters),
        "historical_interval_correction_sha256": CORRECTION_SHA256,
        "historical_trial_accepted": False, "full_root_claim": False,
        "response_validation": "not_performed", "face_support": binding.support,
        "runtime": raw["environment"],
        "start_control": lifted_start.tolist(), "start_control_sha256": slices._tensor_sha(lifted_start),
        "start_tangent": tangent.tolist(), "start_branch": start_branch,
        "start_trace": start_trace, "trials": trials, "linear_solves": solves,
    }
    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    report["status"] = "running"
    save()
    try:
        with matrix_free.observe_pcg_calls(slices._true_residual_monitor(solves, 0.0, state)):
            result = refine_stationary(
                binding.objective, tangent, parameters, branch_check=branch_check,
                max_iterations=MAX_NEWTON_ITERATIONS, max_backtracks=MAX_BACKTRACKS,
                pcg_max_iterations=PCG_MAX_ITERATIONS, trial_observer=observe_trial,
            )
        tangent_final = result.control.detach().clone()
        branch_final, trace_final = _trace_face(binding, tangent_final, parameters)
        tangent_gradient = torch.func.grad(binding.objective)(tangent_final, parameters)
        fresh_gradient_max = float(tangent_gradient.abs().max())
        try:
            slices._assert_scalar_match("active-face tangent gradient", result.gradient_max, fresh_gradient_max)
        except slices.SliceAuditRefusal as error:
            raise ActiveFaceRefusal(str(error)) from error
        if fresh_gradient_max >= STATIONARITY_TOLERANCE:
            raise ActiveFaceRefusal("fresh tangent gradient does not meet or match the refiner gate")
        full_control = binding.lift(tangent_final)
        full_gradient = torch.func.grad(problem.objective, argnums=0)(full_control, parameters)
        hessian_gradient = torch.func.grad(binding.objective)
        columns = []
        for index in range(25):
            direction = torch.zeros_like(tangent_final)
            direction[index] = 1.0
            columns.append(torch.func.jvp(
                lambda value: hessian_gradient(value, parameters),
                (tangent_final,), (direction,),
            )[1])
        hessian = torch.stack(columns, dim=1)
        if not bool(torch.isfinite(hessian).all()):
            raise ActiveFaceRefusal("restricted tangent Hessian contains nonfinite entries")
        symmetric = 0.5 * (hessian + hessian.T)
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        hessian_norm = torch.linalg.vector_norm(hessian)
        symmetry_norm = torch.linalg.vector_norm(hessian - hessian.T)
        symmetry_relative = float(symmetry_norm / hessian_norm) if float(hessian_norm) > 0 else float(symmetry_norm)
        symmetry_tolerance = 128 * torch.finfo(hessian.dtype).eps * hessian.shape[0]
        eigen_scale = float(eigenvalues.abs().max())
        curvature_floor = 128 * torch.finfo(hessian.dtype).eps * max(eigen_scale, torch.finfo(hessian.dtype).tiny)
        curvature_qualified = float(eigenvalues.min()) > curvature_floor and symmetry_relative <= symmetry_tolerance
        audited = _audit_linear_solves(binding, parameters, solves, start_branch["signature_sha256"])
        report.update(
            status="tangent_stationary_candidate", final_tangent=tangent_final.tolist(),
            final_control=full_control.tolist(), final_control_sha256=slices._tensor_sha(full_control),
            final_branch=branch_final, final_trace=trace_final,
            tangent_gradient=tangent_gradient.tolist(),
            tangent_gradient_max=fresh_gradient_max,
            refiner_gradient_max=result.gradient_max,
            full_gradient=full_gradient.tolist(), full_gradient_max=float(full_gradient.abs().max()),
            original_full_gradient_gate_passed=float(full_gradient.abs().max()) < STATIONARITY_TOLERANCE,
            hessian=hessian.tolist(), hessian_symmetry_relative=symmetry_relative,
            symmetric_hessian_eigenvalues=eigenvalues.tolist(),
            tangent_hessian_min_eigenvalue=float(eigenvalues.min()),
            tangent_hessian_max_eigenvalue=float(eigenvalues.max()),
            tangent_hessian_condition=(float(eigenvalues.max()/eigenvalues.min())
                if float(eigenvalues.min()) > 0 and math.isfinite(float(eigenvalues.max()/eigenvalues.min())) else None),
            hessian_symmetry_tolerance=symmetry_tolerance,
            hessian_positive_curvature_floor=curvature_floor,
            hessian_scope="restricted25 classical Hessian with all nonselected branches strict; structural-zero face contributes zero along tangent directions",
            tangent_curvature_qualified=curvature_qualified,
            tangent_curvature_qualification=("positive_definite_on_restricted25" if curvature_qualified
                                             else "not_positive_definite_on_restricted25"),
            full_gradient_scope="ambient native selected-face diagnostic at the eta=0 lift; not a full26 classical-root certificate",
            newton_iterations=result.iterations, refiner_hvp_count=result.hvp_count,
            independently_audited_pcg_solves=audited,
            one_sided_normal_certificate="not_performed",
        )
    except (RefinementNumericalRefusal, ActiveFaceRefusal) as error:
        report.update(status=("numerical_refusal" if isinstance(error, RefinementNumericalRefusal)
                              else "branch_refusal"), reason=str(error),
                      last_accepted_tangent=last_accepted.tolist(),
                      last_accepted_control=binding.lift(last_accepted).tolist())
    after_sources = slices._sources()
    diagnostic_after = {path: _sha(ROOT / path) for path in diagnostic_paths}
    identity_after = slices.prior._preflight_identity()
    if (after_sources != original_sources or diagnostic_after != diagnostic_sources
            or _sha(PLAN) != plan_sha or _sha(RAW) != RAW_SHA256 or _sha(CORRECTION) != CORRECTION_SHA256
            or identity_after != identity or not torch.equal(parameters, fixed_parameters)
            or slices._tensor_sha(parameters) != report["parameters_sha256"]):
        raise slices.SliceIdentityRefusal("active-face sources/inputs changed during refinement")
    report["original_source_after"] = after_sources
    report["diagnostic_source_after"] = diagnostic_after
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
