"""Bounded conditional response validation on the fixed qy[2,0]=0 family."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import torch
from torch import Tensor
from mpmath import mp

from advar import local_response, matrix_free
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from examples.weather_scenarios import (
    fv_active_face_normal_probe as normal_probe,
    fv_active_face_normal_reference as normal_reference,
    fv_active_face_stationarity_probe as stationarity_probe,
    fv_partial_signed_face_slices as slices,
    fv_slice_precision_probe as precision_probe,
)
from examples.weather_scenarios.fv_active_face_binding import FVActiveFaceBinding
from tests.test_fv_research_partial_observation import _problem


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
RAW = EVIDENCE / "partial_signed_face_slices_attempt1/signed_face_slices.json"
RAW_SHA256 = "2e331247833dcf9cda71891bc99283551e5490bb9701fbde3b29eb18fcfbaee9"
STATIONARITY = EVIDENCE / "partial_active_face_stationarity_attempt1/stationarity.json"
STATIONARITY_SHA256 = "9f345554d0ee604057032c4c7031ad7e883d00dd35d010bea2f2970418ba2a4a"
NORMAL_INPUT = EVIDENCE / "R4_ACTIVE_FACE_NORMAL_INPUT_20261003.json"
NORMAL_INPUT_SHA256 = "5694b56e958058adb4004291ea03dc4b5c85841a5acd1627603f1ac0d016174e"
NORMAL_RESULT = EVIDENCE / "partial_active_face_normal_attempt3/normal.json"
NORMAL_RESULT_SHA256 = "14d0b62ebaf995cb980458a965470c52ddfcc17d0e3e03c73d0afa969d411881"
PLAN = EVIDENCE / "R4_ACTIVE_FACE_RESPONSE_PLAN_20261003.md"
PLAN_SHA256 = "17d501487096da7f65d0214d676d63bef0890a84b6773c9c9e4ee9845b194948"
SELF_PATH = "examples/weather_scenarios/fv_active_face_response_probe.py"
STATIONARITY_PATH = "examples/weather_scenarios/fv_active_face_stationarity_probe.py"
NORMAL_PROBE_PATH = "examples/weather_scenarios/fv_active_face_normal_probe.py"
NORMAL_REFERENCE_PATH = "examples/weather_scenarios/fv_active_face_normal_reference.py"
PRECISION_PROBE_PATH = "examples/weather_scenarios/fv_slice_precision_probe.py"
PRECISION_REFERENCE_PATH = "examples/weather_scenarios/fv_slice_precision_reference.py"
BINDING_PATH = "examples/weather_scenarios/fv_active_face_binding.py"
LOCAL_RESPONSE_PATH = "src/advar/local_response.py"
MATRIX_FREE_PATH = "src/advar/matrix_free.py"
STATIONARITY_TOLERANCE = 1e-10
LINEAR_TOLERANCE = 1e-10
PCG_MAX_ITERATIONS = 80
NEWTON_MAX_ITERATIONS = 2
MAX_BACKTRACKS = 16
STEP_SIZES = (0.00025, 0.000125)
DIRECTION_NAME = "middle_time_common_bias_plus1"
PRECISION_PROBE_SHA256 = "d89ecf687e4d85f30e4e35c0317fda00cc9d4e7ecbf3273764bc983510b5817f"
SELF_CANONICAL_SHA256 = "1a7536075391ad1bc91260080e3989048d4ddd8e83edf1e7cf5a63dde0a9e22b"
_SELF_PIN = re.compile(rb'(?m)^SELF_CANONICAL_SHA256 = "[0-9a-f]{64}"$')


class ActiveFaceResponseRefusal(ValueError):
    """Pinned input, branch, or restricted-response contract was not met."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_source_sha256(source: bytes) -> str:
    normalized, count = _SELF_PIN.subn(b'SELF_CANONICAL_SHA256 = "<canonical-self-pin>"', source)
    if count != 1:
        raise slices.SliceIdentityRefusal("response producer self-pin is missing or ambiguous")
    return hashlib.sha256(normalized).hexdigest()


def _require_source_sha256(path: Path, expected: str) -> None:
    if _sha(path) != expected:
        raise slices.SliceIdentityRefusal(f"reviewed source pin changed: {path.name}")


def _require_canonical_source_sha256(source: bytes, expected: str) -> None:
    if _canonical_source_sha256(source) != expected:
        raise slices.SliceIdentityRefusal("reviewed response producer self-pin changed")


def middle_time_direction(parameters: Tensor, valid_mask: Tensor) -> Tensor:
    """Unit dBZ shift of all 20 middle-frame slots; the missing slot is ignored by H."""
    if (parameters.shape != (61,) or parameters.dtype != torch.float64
            or parameters.device.type != "cpu" or not bool(torch.isfinite(parameters).all())
            or valid_mask.shape != (3, 4, 5) or valid_mask.device.type != "cpu"
            or valid_mask.dtype is not torch.bool):
        raise ValueError("direction requires CPU FP64 61 parameters and a 3x4x5 mask")
    if (bool(valid_mask[0].logical_not().any())
            or torch.nonzero(~valid_mask, as_tuple=False).tolist() != [[1, 1, 2], [2, 2, 3]]):
        raise ValueError("direction requires the pinned two-hole observation mask")
    direction = torch.zeros_like(parameters)
    direction[20:40] = 1.0
    return direction


def parameter_endpoint(parameters: Tensor, direction: Tensor, sign: int, step: float) -> Tensor:
    if (sign not in (-1, 1) or parameters.shape != (61,) or direction.shape != parameters.shape
            or parameters.dtype != torch.float64 or direction.dtype != parameters.dtype
            or parameters.device.type != "cpu" or direction.device.type != "cpu"
            or not bool(torch.isfinite(parameters).all()) or not bool(torch.isfinite(direction).all())
            or not math.isfinite(step) or step <= 0):
        raise ValueError("parameter endpoint requires sign ±1, CPU FP64 vectors, and a positive step")
    return parameters + (sign * step) * direction


def _interval_bounds(interval_binary: list[list[int]]) -> tuple[Any, Any]:
    if len(interval_binary) != 2:
        raise ValueError("normal derivative interval must have two exact endpoint tuples")
    return mp.make_mpf(tuple(interval_binary[0])), mp.make_mpf(tuple(interval_binary[1]))


def normal_orientation(minus: list[list[int]], plus: list[list[int]]) -> tuple[bool, dict[str, str]]:
    lower_minus, upper_minus = _interval_bounds(minus)
    lower_plus, upper_plus = _interval_bounds(plus)
    with mp.workdps(100):
        ok = bool(upper_minus < 0 < lower_plus)
        margins = {"negative_side_upper": str(mp.nstr(upper_minus, 50)),
                   "positive_side_lower": str(mp.nstr(lower_plus, 50))}
    return ok, margins


def _hessian_audit(binding: FVActiveFaceBinding, tangent: Tensor, parameters: Tensor) -> dict[str, Any]:
    gradient = torch.func.grad(binding.objective)
    columns = []
    for index in range(25):
        direction = torch.zeros_like(tangent)
        direction[index] = 1.0
        columns.append(torch.func.jvp(lambda value: gradient(value, parameters),
                                      (tangent,), (direction,))[1])
    hessian = torch.stack(columns, dim=1)
    if not bool(torch.isfinite(hessian).all()):
        raise ActiveFaceResponseRefusal("restricted tangent Hessian is nonfinite")
    scale = torch.linalg.vector_norm(hessian)
    symmetry = torch.linalg.vector_norm(hessian - hessian.T)
    relative = float(symmetry / scale) if float(scale) > 0 else float(symmetry)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    minimum, maximum = float(eigenvalues[0]), float(eigenvalues[-1])
    tolerance = 128 * torch.finfo(hessian.dtype).eps * hessian.shape[0]
    floor = 128 * torch.finfo(hessian.dtype).eps * max(abs(minimum), abs(maximum), torch.finfo(hessian.dtype).tiny)
    qualified = relative <= tolerance and minimum > floor
    return {"hvp_columns": len(columns), "symmetry_relative": relative,
            "symmetry_tolerance": tolerance, "lambda_min": minimum, "lambda_max": maximum,
            "positive_curvature_floor": floor,
            "condition": maximum / minimum if minimum > 0 and math.isfinite(maximum / minimum) else None,
            "numerical_spd_qualified": qualified}


def _active_fixture(problem: Any, binding: FVActiveFaceBinding, parameters: Tensor) -> dict[str, Any]:
    fixture, _, _ = precision_probe.fixture_for(problem, parameters)
    fixture.update(projected_psi_basis=binding.transport.psi_basis.tolist(),
                   face_weights=binding.chart.weights.tolist(),
                   face_scale=float(binding.chart.face_scale))
    return fixture


def _normal_pair(problem: Any, binding: FVActiveFaceBinding, parameters: Tensor,
                 tangent: Tensor, trace: list[dict[str, Any]]) -> dict[str, Any]:
    fixture = _active_fixture(problem, binding, parameters)
    try:
        sides = [normal_reference.evaluate_side(fixture, tangent.tolist(), side, dps=80)
                 for side in (-1, 1)]
    except ValueError as error:
        raise ActiveFaceResponseRefusal(f"one-sided interval branch/domain refusal: {error}") from error
    for result in sides:
        if result["branch_choices"] != normal_probe.expected_trace(trace, result["side"]):
            raise ActiveFaceResponseRefusal("one-sided interval reference changed an analysis branch")
    orientation, margins = normal_orientation(
        sides[0]["sigma_eta_interval_binary"], sides[1]["sigma_eta_interval_binary"],
    )
    return {"sides": sides, "strict_negative_positive_orientation": orientation,
            "normal_margins": margins,
            "physical_flux_scale": float(binding.chart.face_scale),
            "scope": "interval derivatives at this approximate restricted point only"}


def _branch_checker(problem: Any, binding: FVActiveFaceBinding, parameters: Tensor,
                    expected_signature: str):
    fixed = parameters.clone()

    def check(tangent: Tensor, candidate_parameters: Tensor):
        if not torch.equal(candidate_parameters, fixed):
            raise ActiveFaceResponseRefusal("branch check received different fixed parameters")
        stationarity_probe._check_fixed_branches(problem, candidate_parameters)
        summary, _ = stationarity_probe._trace_face(binding, tangent, candidate_parameters)
        if summary["signature_sha256"] != expected_signature:
            raise ActiveFaceResponseRefusal("54-stage other-branch signature changed")
        return summary, "fixed structural q_y[2,0]=0; all other 54-stage branches match base"

    return check


def _endpoint_refinement(problem: Any, binding: FVActiveFaceBinding, tangent_start: Tensor,
                          parameters: Tensor, base_signature: str) -> dict[str, Any]:
    refinement_started = time.perf_counter()
    state: dict[str, Any] = {"tangent": tangent_start.clone(), "signature": base_signature}
    branch_check = _branch_checker(problem, binding, parameters, base_signature)
    trials: list[dict[str, Any]] = []
    solves: list[dict[str, Any]] = []
    last_accepted = tangent_start.clone()

    def observe_trial(row: dict[str, Any]) -> None:
        nonlocal last_accepted
        trials.append({key: row[key] for key in (
            "iteration", "backtrack", "step_scale", "accepted", "rejection", "branch_reason",
            "objective", "gradient_max", "linear_relative_residual", "hvp_count",
        ) if key in row})
        if row.get("accepted") is True:
            last_accepted = torch.tensor(row["candidate_control"], dtype=torch.float64)

    stateful_check = branch_check
    # The monitor records the exact tangent/RHS/solution/residual for every PCG solve.
    def checked_branch(value: Tensor, p: Tensor):
        result = stateful_check(value, p)
        state["tangent"] = value.detach().clone()
        return result

    try:
        with matrix_free.observe_pcg_calls(stationarity_probe.slices._true_residual_monitor(
                solves, 0.0, state)):
            result = refine_stationary(
                binding.objective, tangent_start, parameters, branch_check=checked_branch,
                max_iterations=NEWTON_MAX_ITERATIONS, max_backtracks=MAX_BACKTRACKS,
                pcg_max_iterations=PCG_MAX_ITERATIONS, trial_observer=observe_trial,
            )
        tangent = result.control.detach().clone()
        gradient = torch.func.grad(binding.objective)(tangent, parameters)
        gradient_max = float(gradient.abs().max())
        stationarity_probe.slices._assert_scalar_match("endpoint fresh tangent gradient",
                                                        result.gradient_max, gradient_max)
        if gradient_max >= STATIONARITY_TOLERANCE:
            raise ActiveFaceResponseRefusal("endpoint tangent gradient exceeds 1e-10")
        branch, trace = stationarity_probe._trace_face(binding, tangent, parameters)
        if branch["signature_sha256"] != base_signature:
            raise ActiveFaceResponseRefusal("endpoint final branch differs from base signature")
        audited = stationarity_probe._audit_linear_solves(binding, parameters, solves, base_signature)
        full_control = binding.lift(tangent)
        full_gradient = torch.func.grad(problem.objective, argnums=0)(full_control, parameters)
        return {"status": "tangent_stationary", "tangent_start": tangent_start.tolist(),
                "tangent": tangent.tolist(), "tangent_sha256": slices._tensor_sha(tangent),
                "control": full_control.tolist(), "control_sha256": slices._tensor_sha(full_control),
                "parameters": parameters.tolist(), "parameters_sha256": slices._tensor_sha(parameters),
                "objective": float(binding.objective(tangent, parameters)),
                "score": float(binding.score(tangent, parameters)),
                "tangent_gradient": gradient.tolist(), "tangent_gradient_max": gradient_max,
                "ambient_full_gradient_max_diagnostic": float(full_gradient.abs().max()),
                "original_full26_gradient_gate_passed": float(full_gradient.abs().max()) < STATIONARITY_TOLERANCE,
                "branch": branch, "trace": trace, "trials": trials, "linear_solves": solves,
                "independently_audited_pcg_solves": audited,
                "newton_iterations": result.iterations, "refiner_hvp_count": result.hvp_count,
                "last_accepted_tangent": last_accepted.tolist(),
                "refinement_seconds": time.perf_counter() - refinement_started}
    except (stationarity_probe.ActiveFaceRefusal, ActiveFaceResponseRefusal,
            RefinementNumericalRefusal, slices.SliceAuditRefusal) as error:
        return {"status": "refinement_refusal", "reason": f"{type(error).__name__}: {error}",
                "tangent_start": tangent_start.tolist(), "last_accepted_tangent": last_accepted.tolist(),
                "trials": trials, "linear_solves": solves,
                "refinement_seconds": time.perf_counter() - refinement_started}


def main(output: Path) -> None:
    slices._fresh_path(output)
    pinned = ((RAW, RAW_SHA256), (STATIONARITY, STATIONARITY_SHA256),
              (NORMAL_INPUT, NORMAL_INPUT_SHA256), (NORMAL_RESULT, NORMAL_RESULT_SHA256),
              (PLAN, PLAN_SHA256))
    if any(_sha(path) != digest for path, digest in pinned):
        raise slices.SliceIdentityRefusal("pinned R4 response inputs or plan changed")
    self_source = Path(__file__)
    _require_canonical_source_sha256(self_source.read_bytes(), SELF_CANONICAL_SHA256)
    _require_source_sha256(ROOT / PRECISION_PROBE_PATH, PRECISION_PROBE_SHA256)
    raw, stationarity = json.loads(RAW.read_text()), json.loads(STATIONARITY.read_text())
    normal_input, normal_result = json.loads(NORMAL_INPUT.read_text()), json.loads(NORMAL_RESULT.read_text())
    slices._validate_runtime(raw["environment"])
    original_sources = slices._sources()
    if original_sources != raw["source_after"] or original_sources != stationarity["original_source_after"]:
        raise slices.SliceIdentityRefusal("measured original source set changed")
    diagnostic_paths = (SELF_PATH, STATIONARITY_PATH, NORMAL_PROBE_PATH, NORMAL_REFERENCE_PATH,
                        PRECISION_PROBE_PATH, PRECISION_REFERENCE_PATH, BINDING_PATH,
                        LOCAL_RESPONSE_PATH, MATRIX_FREE_PATH)
    diagnostic_before = {path: _sha(ROOT / path) for path in diagnostic_paths}
    if (stationarity["diagnostic_source_after"].get(BINDING_PATH) != diagnostic_before[BINDING_PATH]
            or stationarity["diagnostic_source_after"].get(STATIONARITY_PATH) != diagnostic_before[STATIONARITY_PATH]
            or normal_result["source_sha256"].get(NORMAL_PROBE_PATH) != diagnostic_before[NORMAL_PROBE_PATH]
            or normal_result["source_sha256"].get(NORMAL_REFERENCE_PATH) != diagnostic_before[NORMAL_REFERENCE_PATH]):
        raise slices.SliceIdentityRefusal("measured active-face diagnostic source changed")
    problem, parameters, identity = slices._current_problem()
    fixed_parameters = parameters.clone()
    if (identity != stationarity["input_identity"]
            or normal_input.get("stationarity_sha256") != STATIONARITY_SHA256
            or normal_input.get("tangent") != stationarity.get("final_tangent")
            or normal_input.get("parameters_sha256") != stationarity["parameters_sha256"]
            or normal_input.get("input_identity") != identity
            or slices._tensor_sha(parameters) != stationarity["parameters_sha256"]):
        raise slices.SliceIdentityRefusal("fixed current problem/parameter identity differs from stationarity")
    if (stationarity.get("status") != "tangent_stationary_candidate"
            or stationarity.get("tangent_gradient_max", math.inf) >= STATIONARITY_TOLERANCE
            or stationarity.get("full_root_claim") is not False
            or normal_result.get("strict_opposite_orientation_at_approximate_point") is not True):
        raise slices.SliceIdentityRefusal("pinned active-face stationarity/normal prerequisite is absent")
    binding = FVActiveFaceBinding(problem)
    tangent = torch.tensor(stationarity["final_tangent"], dtype=torch.float64)
    if tangent.shape != (25,) or not bool(torch.isfinite(tangent).all()):
        raise slices.SliceIdentityRefusal("pinned base tangent is malformed")
    saved_control = torch.tensor(stationarity["final_control"], dtype=torch.float64)
    if (saved_control.shape != (26,)
            or slices._tensor_sha(saved_control) != stationarity["final_control_sha256"]):
        raise slices.SliceIdentityRefusal("pinned stationarity full control/hash mismatch")
    base_control = binding.lift(tangent)
    if slices._tensor_sha(base_control) != stationarity["final_control_sha256"]:
        raise slices.SliceIdentityRefusal("fresh chart lift differs from pinned stationarity control")
    stationarity_probe._check_fixed_branches(problem, parameters)
    base_branch, base_trace = stationarity_probe._trace_face(binding, tangent, parameters)
    if (base_branch["signature_sha256"] != stationarity["final_branch"]["signature_sha256"]
            or base_trace != stationarity["final_trace"]):
        raise slices.SliceIdentityRefusal("fresh baseline trace differs from pinned stationarity trace")
    objective_gradient = torch.func.grad(binding.objective)
    base_gradient = objective_gradient(tangent, parameters)
    base_gradient_max = float(base_gradient.abs().max())
    slices._assert_scalar_match("baseline fresh tangent gradient",
                                stationarity["tangent_gradient_max"], base_gradient_max)
    if base_gradient_max >= STATIONARITY_TOLERANCE:
        raise ActiveFaceResponseRefusal("baseline tangent root fails the original gradient gate")
    base_hessian_started = time.perf_counter()
    base_hessian = _hessian_audit(binding, tangent, parameters)
    base_hessian_seconds = time.perf_counter() - base_hessian_started
    base_normal_started = time.perf_counter()
    base_normal = _normal_pair(problem, binding, parameters, tangent, base_trace)
    base_normal_seconds = time.perf_counter() - base_normal_started
    for recomputed, archived in zip(base_normal["sides"], normal_result["sides"], strict=True):
        if recomputed["sigma_eta_interval_binary"] != archived["sigma_eta_interval_binary"]:
            raise slices.SliceIdentityRefusal("fresh baseline normal enclosure differs from pinned result")
    if (not base_hessian["numerical_spd_qualified"]
            or not base_normal["strict_negative_positive_orientation"]):
        raise ActiveFaceResponseRefusal("fresh baseline tangent-curvature/normal gate is unsupported")
    base_full_gradient = torch.func.grad(problem.objective, argnums=0)(base_control, parameters)
    direction = middle_time_direction(parameters, problem.observations.valid_mask)
    branch = _branch_checker(problem, binding, parameters, base_branch["signature_sha256"])

    score_on_fixed_verification = binding.score

    identity_record = {
        "fixed_problem": identity, "parameters_sha256": slices._tensor_sha(parameters),
        "base_tangent_sha256": slices._tensor_sha(tangent),
        "base_branch_sha256": base_branch["signature_sha256"],
        "direction_sha256": slices._tensor_sha(direction),
        "response_variant": "restricted_active_face_local_optimizer_conditional",
    }
    started = time.perf_counter()
    response = local_response.compute_local_response(
        binding.objective, score_on_fixed_verification, tangent, parameters,
        {DIRECTION_NAME: direction}, branch_check=branch, input_identity=identity_record,
    )
    response_seconds = time.perf_counter() - started
    direct_value = float(response.direct[DIRECTION_NAME])
    if direct_value != 0.0:
        raise ActiveFaceResponseRefusal("middle-time direction unexpectedly has a direct score effect")
    missing_indices = (27, 53)
    missing_gradients = {str(index): [float(value[index]) for value in (
        response.direct_gradient, response.indirect_gradient, response.total_gradient,
    )] for index in missing_indices}
    if any(component != 0.0 for triplet in missing_gradients.values() for component in triplet):
        raise ActiveFaceResponseRefusal("missing observation parameter affected the response")

    cross = response.mixed_gradients[DIRECTION_NAME]
    tangent_rhs = -cross
    tangent_hvps = 0

    def tangent_hessian_vector(delta: Tensor) -> Tensor:
        nonlocal tangent_hvps
        tangent_hvps += 1
        value = torch.func.jvp(lambda c: objective_gradient(c, parameters), (tangent,), (delta,))[1]
        if not bool(torch.isfinite(value).all()):
            raise ActiveFaceResponseRefusal("tangent predictor HVP is nonfinite")
        return value

    predictor_started = time.perf_counter()
    predictor = matrix_free.pcg(tangent_hessian_vector, tangent_rhs,
                                rtol=LINEAR_TOLERANCE, max_iterations=PCG_MAX_ITERATIONS)
    predictor_seconds = time.perf_counter() - predictor_started
    predicted_tangent = predictor.solution
    predictor_residual = tangent_hessian_vector(predicted_tangent) - tangent_rhs
    rhs_norm = float(torch.linalg.vector_norm(tangent_rhs))
    true_predictor_relative = float(torch.linalg.vector_norm(predictor_residual)) / rhs_norm if rhs_norm > 0 else float(torch.linalg.vector_norm(predictor_residual))
    if (not predictor.converged or not bool(torch.isfinite(predicted_tangent).all())
            or not math.isfinite(true_predictor_relative) or true_predictor_relative > LINEAR_TOLERANCE):
        raise ActiveFaceResponseRefusal("matrix-free tangent predictor failed its true residual gate")

    report: dict[str, Any] = {
        "scope": "conditional restricted-active-face local-optimizer response at an approximate point; not a smooth26 response proof",
        "response_variant": "restricted_active_face_local_optimizer_conditional",
        "original_smooth26_response_supported": False, "physical_validated": False,
        "full_root_claim": False, "uniform_parameter_neighborhood_proved": False,
        "response_validation": "running", "plan_sha256": PLAN_SHA256,
        "raw_sha256": RAW_SHA256, "stationarity_sha256": STATIONARITY_SHA256,
        "normal_input_sha256": NORMAL_INPUT_SHA256, "normal_result_sha256": NORMAL_RESULT_SHA256,
        "original_source_before": original_sources, "diagnostic_source_before": diagnostic_before,
        "input_identity": identity, "parameters_sha256": slices._tensor_sha(parameters),
        "base_tangent": tangent.tolist(), "base_tangent_sha256": slices._tensor_sha(tangent),
        "base_control": base_control.tolist(), "base_control_sha256": slices._tensor_sha(base_control),
        "base_branch": base_branch,
        "base_trace": base_trace, "base_gradient": base_gradient.tolist(),
        "base_gradient_max": base_gradient_max, "base_hessian": base_hessian,
        "base_objective": float(binding.objective(tangent, parameters)),
        "base_score": float(score_on_fixed_verification(tangent, parameters)),
        "base_normal_recomputed": base_normal,
        "base_hessian_seconds": base_hessian_seconds,
        "base_normal_seconds": base_normal_seconds,
        "ambient_full_gradient_max_diagnostic": float(base_full_gradient.abs().max()),
        "original_full26_gradient_gate_passed": float(base_full_gradient.abs().max()) < STATIONARITY_TOLERANCE,
        "direction": direction.tolist(), "direction_sha256": slices._tensor_sha(direction),
        "active_middle_direction_count": int(problem.observations.valid_mask[1].sum()),
        "inactive_missing_direction_slots": list(missing_indices),
        "base_response": {
            "direct_gradient": response.direct_gradient.tolist(),
            "indirect_gradient": response.indirect_gradient.tolist(),
            "total_gradient": response.total_gradient.tolist(),
            "direct": direct_value, "indirect": float(response.indirect[DIRECTION_NAME]),
            "total": float(response.total[DIRECTION_NAME]),
            "score_control_gradient": response.score_control_gradient.tolist(),
            "adjoint": response.adjoint.tolist(),
            "mixed_gradient": cross.tolist(), "gradient_max": response.gradient_max,
            "true_adjoint_residual": response.true_adjoint_residual,
            "true_adjoint_relative_residual": response.true_adjoint_relative_residual,
            "pcg_relative_residual": response.pcg_relative_residual,
            "pcg_iterations": response.pcg_iterations, "hvp_count": response.hvp_count,
            "scope": response.scope, "seconds": response_seconds,
            "missing_parameter_gradient_triplets": missing_gradients,
        },
        "tangent_predictor": {
            "rhs": tangent_rhs.tolist(), "solution": predicted_tangent.tolist(),
            "converged": predictor.converged, "iterations": predictor.iterations,
            "pcg_relative_residual": predictor.relative_residual,
            "true_residual": predictor_residual.tolist(),
            "true_relative_residual": true_predictor_relative,
            "max_iterations": PCG_MAX_ITERATIONS, "rtol": LINEAR_TOLERANCE,
            "hvp_count": tangent_hvps, "seconds": predictor_seconds,
        },
        "endpoints": [],
        "timings": {"base_response_seconds": response_seconds,
                    "tangent_predictor_seconds": predictor_seconds},
    }

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    report["response_validation"] = "base_response_ready"
    save()
    for step in STEP_SIZES:
        for sign in (-1, 1):
            endpoint_started = time.perf_counter()
            endpoint: dict[str, Any] = {"step": step, "sign": sign}
            parameters_endpoint = parameter_endpoint(parameters, direction, sign, step)
            tangent_start = tangent + (sign * step) * predicted_tangent
            endpoint["parameters_sha256"] = slices._tensor_sha(parameters_endpoint)
            endpoint["parameters"] = parameters_endpoint.tolist()
            endpoint["tangent_start"] = tangent_start.tolist()
            try:
                stationarity_probe._check_fixed_branches(problem, parameters_endpoint)
                endpoint.update(_endpoint_refinement(problem, binding, tangent_start,
                                                    parameters_endpoint, base_branch["signature_sha256"]))
                if endpoint["status"] == "tangent_stationary":
                    hessian_started = time.perf_counter()
                    endpoint["hessian_audit"] = _hessian_audit(
                        binding, torch.tensor(endpoint["tangent"], dtype=torch.float64), parameters_endpoint,
                    )
                    endpoint["hessian_seconds"] = time.perf_counter() - hessian_started
                    normal_started = time.perf_counter()
                    endpoint["normal"] = _normal_pair(
                        problem, binding, parameters_endpoint,
                        torch.tensor(endpoint["tangent"], dtype=torch.float64), endpoint["trace"],
                    )
                    endpoint["normal_seconds"] = time.perf_counter() - normal_started
                    endpoint["qualification_passed"] = bool(
                        endpoint["hessian_audit"]["numerical_spd_qualified"]
                        and endpoint["normal"]["strict_negative_positive_orientation"]
                    )
            except (stationarity_probe.ActiveFaceRefusal,
                    ActiveFaceResponseRefusal, RefinementNumericalRefusal) as error:
                endpoint.update(status="endpoint_refusal", reason=f"{type(error).__name__}: {error}",
                                last_accepted_tangent=endpoint.get("last_accepted_tangent", tangent_start.tolist()))
            endpoint["total_seconds"] = time.perf_counter() - endpoint_started
            report["endpoints"].append(endpoint)
            save()

    by_key = {(row["step"], row["sign"]): row for row in report["endpoints"]}
    finite_differences = []
    for step in STEP_SIZES:
        negative, positive = by_key[(step, -1)], by_key[(step, 1)]
        if negative.get("status") != "tangent_stationary" or positive.get("status") != "tangent_stationary":
            finite_differences.append({"step": step, "status": "endpoint_refusal"})
            continue
        if (negative.get("qualification_passed") is not True
                or positive.get("qualification_passed") is not True):
            finite_differences.append({"step": step, "status": "endpoint_qualification_not_met"})
            continue
        difference = (positive["score"] - negative["score"]) / (2 * step)
        predicted = float(response.total[DIRECTION_NAME])
        absolute_error = abs(difference - predicted)
        scale = max(abs(difference), abs(predicted), torch.finfo(torch.float64).tiny)
        relative_error = absolute_error / scale
        finite_differences.append({"step": step, "central_score_difference": difference,
                                   "predicted_response": predicted,
                                   "absolute_error": absolute_error,
                                   "relative_error": relative_error,
                                   "tolerance": 1e-4,
                                   "passed": relative_error <= 1e-4})
    decreasing_error = (len(finite_differences) == 2
                        and all(item.get("passed") is True for item in finite_differences)
                        and finite_differences[1]["absolute_error"] < finite_differences[0]["absolute_error"])
    all_endpoints = all(row.get("qualification_passed") is True for row in report["endpoints"])
    report["score_finite_difference"] = finite_differences
    report["finite_difference_errors_decrease"] = decreasing_error
    report["response_validation"] = ("restricted_active_face_response_numerically_supported_at_tested_points"
                                     if all_endpoints and decreasing_error
                                     else "conditional_response_not_qualified")
    report["timings"]["total_seconds"] = time.perf_counter() - started

    after_sources = slices._sources()
    diagnostic_after = {path: _sha(ROOT / path) for path in diagnostic_paths}
    identity_after = slices.prior._preflight_identity()
    if (after_sources != original_sources or diagnostic_after != diagnostic_before
            or identity_after != identity or not torch.equal(parameters, fixed_parameters)
            or any(_sha(path) != digest for path, digest in pinned)):
        raise slices.SliceIdentityRefusal("response source/input/parameter pins changed during calculation")
    report["original_source_after"] = after_sources
    report["diagnostic_source_after"] = diagnostic_after
    report["parameters_sha256_after"] = slices._tensor_sha(parameters)
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
