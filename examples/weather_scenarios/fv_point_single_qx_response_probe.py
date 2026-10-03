"""Bounded conditional response check on the original point qx[3,4]=0 face."""
from __future__ import annotations

import argparse, hashlib, json, math, platform
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import mpmath, torch
from torch import Tensor

from advar import local_response, matrix_free
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from examples.weather_scenarios import fv_point_active_face_stationarity as stationarity
from examples.weather_scenarios import fv_point_response_preflight as preflight
from examples.weather_scenarios import fv_point_single_qx_normal_probe as normal_probe
from examples.weather_scenarios import fv_point_single_face_normal_reference as normal_reference
from examples.weather_scenarios.fv_point_active_face_binding import FVPointActiveFaceBinding
from advar.fv_point_research_problem import FVPointResearchProblem

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "graphify-out/fv-root-cause-20260919"
RELEASE = OUT / "point_qy_release_attempt2/release.json"
RELEASE_SHA256 = "e3f4f083ac9bfb002d38e4e2a0352672f179775d18b22fbf26fd294bedd2ad24"
NORMAL_INPUT = OUT / "R2_POINT_SINGLE_QX_NORMAL_INPUT_20261003_RELEASE2.json"
NORMAL_INPUT_SHA256 = "dd8519440ac989944e787e9365115b210e3f1aa22a19048f0f8113990d6313e4"
NORMAL_RESULT = OUT / "point_single_qx_normal_attempt2/normal.json"
NORMAL_RESULT_SHA256 = "ae6e65aaab17de0ab5e8d578dcba6d61e220e4365e46fd096a0dcc507974fb38"
PLAN = OUT / "R2_POINT_SINGLE_QX_RESPONSE_PLAN_20261003.md"
PLAN_SHA256 = "1aae37e936ce9ecd0a8439fbd78a87470dca68169773dab5152844f1c8b24408"
SOURCE_PATHS = tuple(dict.fromkeys((
    *preflight.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_event_diagnostic.py",
    "examples/weather_scenarios/fv_point_active_face_binding.py",
    "examples/weather_scenarios/fv_point_active_face_stationarity.py",
    "examples/weather_scenarios/fv_point_single_face_normal_reference.py",
    "examples/weather_scenarios/fv_point_single_qx_normal_probe.py",
    "examples/weather_scenarios/fv_point_paired_face_normal_probe.py",
    "examples/weather_scenarios/fv_point_paired_face_normal_reference.py",
    "examples/weather_scenarios/fv_active_face_normal_reference.py",
    "examples/weather_scenarios/fv_slice_precision_probe.py",
    "examples/weather_scenarios/fv_slice_precision_reference.py",
    "examples/weather_scenarios/fv_point_single_qx_response_probe.py",
    "tests/test_fv_point_single_qx_response_probe.py",
)))
STATIONARITY_TOLERANCE = 1e-10
LINEAR_TOLERANCE = 1e-10
PCG_MAX_ITERATIONS = 104
STEP_SIZES = (0.00025, 0.000125)
DIRECTION_NAME = "middle_four_qx_face_plus1"

class ResponseRefusal(ValueError): """Pinned input, branch, curvature, or independent-normal gate failed."""

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

def qx_direction(parameters: Tensor) -> Tensor:
    if (parameters.shape != (13,) or parameters.dtype != torch.float64
            or parameters.device.type != "cpu" or not bool(torch.isfinite(parameters).all())):
        raise ValueError("qx response direction requires a finite CPU FP64 length-13 parameter vector")
    direction = torch.zeros_like(parameters)
    direction[4:8] = 1.0
    return direction

def parameter_endpoint(parameters: Tensor, direction: Tensor, sign: int, step: float) -> Tensor:
    if (sign not in (-1, 1) or parameters.shape != (13,) or direction.shape != (13,)
            or parameters.dtype != torch.float64 or direction.dtype != torch.float64
            or parameters.device.type != "cpu" or direction.device.type != "cpu"
            or not bool(torch.isfinite(parameters).all()) or not bool(torch.isfinite(direction).all())
            or not math.isfinite(step) or step <= 0):
        raise ValueError("endpoint requires sign +/-1, finite CPU FP64 vectors, and positive step")
    return parameters + (sign * step) * direction

def _fixture(problem: FVPointResearchProblem, parameters: Tensor,
             captured: dict[str, Any]) -> dict[str, Any]:
    """Refresh only p-dependent values while preserving captured fixed inputs."""
    fixture = deepcopy(captured)
    fixture["background_dbz"] = problem.contract(parameters).initial_background_dbz.tolist()
    fixture["point_dbz"] = parameters[:-1].reshape_as(problem.observation_dbz).tolist()
    return fixture

def _source_inventory(*maps: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(set(SOURCE_PATHS).union(*(mapping.keys() for mapping in maps))))

def _validate_runtimes(release: dict[str, Any], normal: dict[str, Any]) -> dict[str, Any]:
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"}
    normal_runtime = {"python": platform.python_version(), "mpmath": mpmath.__version__}
    if runtime != release.get("runtime") or normal_runtime != normal.get("runtime") or mpmath.__version__ != "1.3.0":
        raise ValueError("runtime differs from the pinned release or normal qualification")
    return runtime

def _normal_pair(problem: Any, parameters: Tensor, tangent: Tensor,
                 native_trace: list[dict[str, Any]], captured: dict[str, Any],
                 binding: FVPointActiveFaceBinding) -> dict[str, Any]:
    fixture = _fixture(problem, parameters, captured)
    sides = []
    for side in (-1, 1):
        row = normal_probe.evaluate_side(fixture, tangent.tolist(), side)
        normal_probe.audit_trace(native_trace[:36], row.get("branch_choices", []), side)
        normal_probe.audit_flux(row)
        comparison = normal_probe.audit.primal_crosscheck(float(binding.objective(tangent, parameters)), row)
        if not comparison["passed"]:
            raise ResponseRefusal("independent interval objective does not cross-check the actual-p native objective")
        row["native_objective_crosscheck"] = comparison
        sides.append(row)
    left, right = (normal_probe.audit.interval(row, "sigma")[:2] for row in sides)
    oriented = normal_reference.strict_normal_orientation(left, right)
    return {"sides": sides, "strict_negative_positive_orientation": oriented,
            "scope": "independent two-sided qx normal at this actual-parameter approximate tangent"}

def _branch_checker(problem: Any, binding: FVPointActiveFaceBinding, parameters: Tensor,
                    expected: str):
    fixed = parameters.clone()

    def check(tangent: Tensor, candidate: Tensor):
        if not torch.equal(candidate, fixed):
            raise ResponseRefusal("branch check received different fixed parameters")
        try:
            control = binding.lift(tangent)
            binding.chart.to_face_coordinates(control)
        except RuntimeError as error:
            if str(error) not in ("face coordinate is outside the strict representable chart domain",
                                  "original control is outside the strict representable chart domain"):
                raise
            raise stationarity.PointFaceRefusal(str(error)) from error
        branch, _ = stationarity.trace(binding, tangent, candidate)
        if branch["signature_sha256"] != expected:
            raise ResponseRefusal("54-stage nonselected branch signature changed")
        return branch, "fixed q_x[3,4]=0; all other 54-stage branches match release2"
    return check

def _refine(problem: Any, binding: FVPointActiveFaceBinding, start: Tensor,
            parameters: Tensor, signature: str) -> dict[str, Any]:
    state: dict[str, Any] = {"tangent": start.clone(), "signature": signature}
    solves: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    last = start.clone()
    branch_check = _branch_checker(problem, binding, parameters, signature)

    def check(value: Tensor, p: Tensor):
        if not torch.equal(p, parameters):
            raise ResponseRefusal("endpoint refinement changed its actual parameters")
        branch, scope = branch_check(value, p)
        state["tangent"] = value.detach().clone()
        return branch, scope

    def observe(row: dict[str, Any]) -> None:
        nonlocal last
        trials.append({k: row[k] for k in ("iteration", "backtrack", "step_scale", "accepted",
                      "rejection", "branch_reason", "objective", "gradient_max",
                      "linear_relative_residual", "hvp_count") if k in row})
        if row.get("accepted") is True:
            last = torch.tensor(row["candidate_control"], dtype=torch.float64)

    try:
        from examples.weather_scenarios.fv_partial_signed_face_slices import _true_residual_monitor
        with matrix_free.observe_pcg_calls(_true_residual_monitor(solves, 0.0, state)):
            result = refine_stationary(binding.objective, start, parameters, branch_check=check,
                max_iterations=2, max_backtracks=16, pcg_max_iterations=PCG_MAX_ITERATIONS,
                trial_observer=observe)
        tangent = result.control.detach().clone()
        gradient = torch.func.grad(binding.objective)(tangent, parameters)
        gradient_max = float(gradient.abs().max())
        if not math.isfinite(gradient_max) or gradient_max >= STATIONARITY_TOLERANCE:
            raise ResponseRefusal("fresh endpoint tangent gradient exceeds 1e-10")
        branch, trace = stationarity.trace(binding, tangent, parameters)
        if branch["signature_sha256"] != signature:
            raise ResponseRefusal("refined endpoint changed the 54-stage branch")
        stationarity.audit_solves(binding.objective, parameters, solves, signature)
        control = binding.lift(tangent)
        full_gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
        return {"status": "tangent_stationary", "tangent_start": start.tolist(),
                "tangent": tangent.tolist(), "tangent_sha256": tensor_sha(tangent),
                "control": control.tolist(), "control_sha256": tensor_sha(control),
                "parameters": parameters.tolist(), "parameters_sha256": tensor_sha(parameters),
                "objective": float(binding.objective(tangent, parameters)),
                "score": float(binding.score(tangent, parameters)),
                "tangent_gradient": gradient.tolist(), "tangent_gradient_max": gradient_max,
                "ambient_full_gradient_max_diagnostic": float(full_gradient.abs().max()),
                "original_full26_gradient_gate_passed": float(full_gradient.abs().max()) < STATIONARITY_TOLERANCE,
                "branch": branch, "trace": trace, "trials": trials, "linear_solves": solves,
                "newton_iterations": result.iterations, "refiner_hvp_count": result.hvp_count,
                "last_accepted_tangent": last.tolist()}
    except (ResponseRefusal, stationarity.PointFaceRefusal, RefinementNumericalRefusal) as error:
        return {"status": "refinement_refusal", "reason": f"{type(error).__name__}: {error}",
                "tangent_start": start.tolist(), "last_accepted_tangent": last.tolist(),
                "trials": trials, "linear_solves": solves}

def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("response output must be fresh")
    if (sha(RELEASE) != RELEASE_SHA256 or sha(NORMAL_INPUT) != NORMAL_INPUT_SHA256
            or sha(PLAN) != PLAN_SHA256):
        raise ValueError("release2, normal capture, or response plan pin changed")
    if NORMAL_RESULT_SHA256.startswith("PENDING_") or not NORMAL_RESULT.exists():
        raise ValueError("root-owned independent normal-result pin is still pending")
    if sha(NORMAL_RESULT) != NORMAL_RESULT_SHA256:
        raise ValueError("independent normal-result pin changed")
    release, captured, normal_result = (json.loads(p.read_text()) for p in (RELEASE, NORMAL_INPUT, NORMAL_RESULT))
    if (release.get("phase") != "finished" or release.get("numerical_status") != "tangent_stationary_candidate"
            or release.get("response_validation") != "not_performed" or release.get("full_root_claim") is not False
            or normal_result.get("phase") != "finished"
            or normal_result.get("numerical_status") != "oriented_single_qx_normal"
            or normal_result.get("strict_single_qx_orientation") is not True
            or normal_result.get("response_validation") != "not_performed"
            or normal_result.get("release_report_sha256") != RELEASE_SHA256
            or normal_result.get("input_sha256") != NORMAL_INPUT_SHA256
            or captured.get("release_report_sha256") != RELEASE_SHA256
            or captured.get("input_identity") != release.get("input_after")
            or captured.get("tangent") != release.get("last_accepted_tangent")
            or captured.get("native_objective") != release.get("final_objective")
            or captured.get("native_signature_sha256") != release.get("final_branch", {}).get("signature_sha256")):
        raise ValueError("release2 and independent-normal capture provenance is incomplete")
    runtime = _validate_runtimes(release, normal_result)
    if (normal_result.get("input_identity_before") != captured.get("input_identity")
            or normal_result.get("input_identity_after") != captured.get("input_identity")):
        raise ValueError("normal report fixed-input identity differs from captured fixture")
    captured_sources = captured.get("source_sha256", {})
    release_sources = release.get("source_after", {})
    normal_sources_before = normal_result.get("source_before", {})
    normal_sources = normal_result.get("source_after", {})
    if normal_sources_before != normal_sources:
        raise ValueError("independent normal source-before/source-after record changed")
    source_paths = _source_inventory(captured_sources, release_sources, normal_sources_before, normal_sources)
    before = {name: sha(ROOT / name) for name in source_paths}
    if any(before.get(name) != digest for name, digest in captured_sources.items()):
        raise ValueError("normal capture source map changed")
    if any(before.get(name) != digest for name, digest in release_sources.items()):
        raise ValueError("release2 source map changed")
    if any(before.get(name) != digest for name, digest in normal_sources.items()):
        raise ValueError("independent normal source map changed")
    if any(before.get(name) != digest for name, digest in normal_sources_before.items()):
        raise ValueError("independent normal pre-run source map changed")
    problem, warm, parameters, direction = preflight.fixed_problem()
    fixed_parameters = parameters.clone()
    input_before = preflight._input_identity(problem, warm, parameters, direction)
    binding = FVPointActiveFaceBinding(problem)
    tangent = torch.tensor(release["last_accepted_tangent"], dtype=torch.float64)
    if tangent.shape != (25,) or tensor_sha(parameters) != captured.get("parameters_sha256"):
        raise ValueError("release tangent or actual original parameter vector is malformed")
    if input_before != release.get("input_after"):
        raise ValueError("fixed point input identity differs from release2")
    branch, trace = stationarity.trace(binding, tangent, parameters)
    if (branch["signature_sha256"] != release["final_branch"]["signature_sha256"]
            or trace != release["final_trace"]):
        raise ValueError("fresh point tangent trace differs from release2")
    gradient = torch.func.grad(binding.objective)(tangent, parameters)
    gradient_max = float(gradient.abs().max())
    if not math.isfinite(gradient_max) or gradient_max >= STATIONARITY_TOLERANCE:
        raise ResponseRefusal("fresh release2 tangent gradient exceeds 1e-10")
    curvature = stationarity.curvature(binding.objective, tangent, parameters)
    if not curvature["qualified"]:
        raise ResponseRefusal("fresh 25D tangent Hessian failed SPD audit")
    if not normal_result.get("evaluations") or len(normal_result["evaluations"]) != 2:
        raise ValueError("independent normal result omitted its two side evaluations")
    base_normal = _normal_pair(problem, parameters, tangent, trace, captured["fixture"], binding)
    if not base_normal["strict_negative_positive_orientation"]:
        raise ResponseRefusal("fresh actual-parameter two-sided normal failed orientation")
    direction = qx_direction(parameters)
    base_control = binding.lift(tangent)
    signature = branch["signature_sha256"]
    identity = {"release_report_sha256": RELEASE_SHA256,
                "normal_input_sha256": NORMAL_INPUT_SHA256,
                "normal_result_sha256": NORMAL_RESULT_SHA256,
                "base_tangent_sha256": tensor_sha(tangent),
                "parameters_sha256": tensor_sha(parameters),
                "branch_sha256": signature,
                "response_variant": "restricted_active_face_local_optimizer_conditional"}
    response = local_response.compute_local_response(
        binding.objective, binding.score, tangent, parameters, {DIRECTION_NAME: direction},
        branch_check=_branch_checker(problem, binding, parameters, signature), input_identity=identity)
    if float(response.direct[DIRECTION_NAME]) != 0.0:
        raise ResponseRefusal("middle four observed-dBZ parameters unexpectedly affect the score directly")
    rhs = -response.mixed_gradients[DIRECTION_NAME]
    gradient_function = torch.func.grad(binding.objective)
    hvps = 0

    def hessian_vector(vector: Tensor) -> Tensor:
        nonlocal hvps
        hvps += 1
        value = torch.func.jvp(lambda c: gradient_function(c, parameters), (tangent,), (vector,))[1]
        if not bool(torch.isfinite(value).all()):
            raise ResponseRefusal("predictor Hessian-vector product is nonfinite")
        return value

    predictor = matrix_free.pcg(hessian_vector, rhs, rtol=LINEAR_TOLERANCE,
                                max_iterations=PCG_MAX_ITERATIONS)
    residual = hessian_vector(predictor.solution) - rhs
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    true_relative = float(torch.linalg.vector_norm(residual)) / rhs_norm if rhs_norm > 0 else float(torch.linalg.vector_norm(residual))
    if (not predictor.converged or not math.isfinite(true_relative) or true_relative > LINEAR_TOLERANCE
            or not bool(torch.isfinite(predictor.solution).all())):
        raise ResponseRefusal("matrix-free tangent predictor failed its true residual gate")
    report: dict[str, Any] = {
        "phase": "running",
        "scope": "conditional response on the qx[3,4]=0 face at a tested approximate point",
        "response_variant": "restricted_active_face_local_optimizer_conditional",
        "original_full26_classical_root_supported": False, "physical_validated": False,
        "full_root_claim": False, "uniform_parameter_neighborhood_proved": False,
        "response_validation": "running", "plan_sha256": PLAN_SHA256,
        "release_report_sha256": RELEASE_SHA256, "normal_input_sha256": NORMAL_INPUT_SHA256,
        "normal_result_sha256": NORMAL_RESULT_SHA256, "source_before": before,
        "input_identity": release["input_after"], "parameters_sha256": tensor_sha(parameters),
        "base_tangent": tangent.tolist(), "base_tangent_sha256": tensor_sha(tangent),
        "base_control": base_control.tolist(), "base_control_sha256": tensor_sha(base_control),
        "base_branch": branch, "base_trace": trace, "base_gradient": gradient.tolist(),
        "base_gradient_max": gradient_max, "base_curvature": curvature,
        "base_objective": float(binding.objective(tangent, parameters)),
        "base_score": float(binding.score(tangent, parameters)), "base_normal": base_normal,
        "direction": direction.tolist(), "direction_sha256": tensor_sha(direction),
        "base_response": {"direct_gradient": response.direct_gradient.tolist(),
            "indirect_gradient": response.indirect_gradient.tolist(), "total_gradient": response.total_gradient.tolist(),
            "direct": float(response.direct[DIRECTION_NAME]), "indirect": float(response.indirect[DIRECTION_NAME]),
            "total": float(response.total[DIRECTION_NAME]), "score_control_gradient": response.score_control_gradient.tolist(),
            "adjoint": response.adjoint.tolist(), "mixed_gradient": response.mixed_gradients[DIRECTION_NAME].tolist(),
            "gradient_max": response.gradient_max, "true_adjoint_residual": response.true_adjoint_residual,
            "true_adjoint_relative_residual": response.true_adjoint_relative_residual,
            "pcg_relative_residual": response.pcg_relative_residual, "pcg_iterations": response.pcg_iterations,
            "hvp_count": response.hvp_count, "scope": response.scope},
        "tangent_predictor": {"rhs": rhs.tolist(), "solution": predictor.solution.tolist(),
            "converged": predictor.converged, "iterations": predictor.iterations,
            "pcg_relative_residual": predictor.relative_residual, "true_residual": residual.tolist(),
            "true_relative_residual": true_relative, "max_iterations": PCG_MAX_ITERATIONS,
            "rtol": LINEAR_TOLERANCE, "hvp_count": hvps}, "endpoints": []}

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    report["response_validation"] = "base_response_ready"
    save()
    for step in STEP_SIZES:
        for sign in (-1, 1):
            p_end = parameter_endpoint(parameters, direction, sign, step)
            start = tangent + (sign * step) * predictor.solution
            endpoint: dict[str, Any] = {"step": step, "sign": sign,
                "parameters": p_end.tolist(), "parameters_sha256": tensor_sha(p_end),
                "tangent_start": start.tolist()}
            try:
                endpoint.update(_refine(problem, binding, start, p_end, signature))
                if endpoint["status"] == "tangent_stationary":
                    end_tangent = torch.tensor(endpoint["tangent"], dtype=torch.float64)
                    endpoint["curvature"] = stationarity.curvature(binding.objective, end_tangent, p_end)
                    endpoint["normal"] = _normal_pair(problem, p_end, end_tangent,
                                                        endpoint["trace"], captured["fixture"], binding)
                    endpoint["qualification_passed"] = bool(
                        endpoint["curvature"]["qualified"]
                        and endpoint["normal"]["strict_negative_positive_orientation"])
            except (ResponseRefusal, stationarity.PointFaceRefusal,
                    normal_probe.Refusal, RefinementNumericalRefusal) as error:
                endpoint.update(status="endpoint_refusal", reason=f"{type(error).__name__}: {error}")
            report["endpoints"].append(endpoint)
            save()
    by_key = {(row["step"], row["sign"]): row for row in report["endpoints"]}
    differences = []
    predicted = float(response.total[DIRECTION_NAME])
    for step in STEP_SIZES:
        minus, plus = by_key[(step, -1)], by_key[(step, 1)]
        if not all(row.get("qualification_passed") is True for row in (minus, plus)):
            differences.append({"step": step, "status": "endpoint_not_qualified"})
            continue
        measured = (plus["score"] - minus["score"]) / (2 * step)
        error = abs(measured - predicted)
        relative = error / max(abs(measured), abs(predicted), torch.finfo(torch.float64).tiny)
        differences.append({"step": step, "central_score_difference": measured,
            "predicted_response": predicted, "absolute_error": error,
            "relative_error": relative, "tolerance": 1e-4, "passed": relative <= 1e-4})
    report["score_finite_difference"] = differences
    report["finite_difference_errors_decrease"] = bool(
        len(differences) == 2 and all(row.get("passed") is True for row in differences)
        and differences[1]["absolute_error"] < differences[0]["absolute_error"])
    report["response_validation"] = (
        "restricted_active_face_response_numerically_supported_at_tested_points"
        if len(report["endpoints"]) == 4
        and all(row.get("qualification_passed") is True for row in report["endpoints"])
        and report["finite_difference_errors_decrease"]
        else "conditional_response_not_qualified")
    after = {name: sha(ROOT / name) for name in source_paths}
    input_after = preflight._input_identity(problem, warm, parameters, direction)
    if (after != before or sha(RELEASE) != RELEASE_SHA256 or sha(NORMAL_INPUT) != NORMAL_INPUT_SHA256
            or sha(NORMAL_RESULT) != NORMAL_RESULT_SHA256 or sha(PLAN) != PLAN_SHA256
            or input_after != input_before or not torch.equal(parameters, fixed_parameters)):
        raise ValueError("source or pinned evidence changed during response calculation")
    runtime_after = _validate_runtimes(release, normal_result)
    report.update(phase="finished", source_after=after, source_unchanged=True,
                  input_before=input_before, input_after=input_after, input_unchanged=True,
                  runtime=runtime, runtime_after=runtime_after,
                  normal_runtime=normal_result["runtime"], normal_runtime_after={
                      "python": platform.python_version(), "mpmath": mpmath.__version__})
    save()
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
