"""Four guarded tangent-slice solves for the pinned two-hole FV point."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import matrix_free
from advar.local_refinement import (
    _NonFiniteEvaluation,
    RefinementCallbackError,
    RefinementNumericalRefusal,
    RefinementTrial,
    refine_stationary,
)
from advar.transport import bounded_fv_coefficients, face_volume_fluxes
from examples.weather_scenarios import fv_face_flux_coordinates as face_chart
from examples.weather_scenarios import fv_partial_alternate_root_probe as seed_probe
from examples.weather_scenarios import fv_partial_field_correction_probe as field_probe
from examples.weather_scenarios import fv_partial_reanalysis_preflight as preflight
from examples.weather_scenarios import fv_partial_sector_root_probe as prior
from examples.weather_scenarios.fv86_resource_runner import run_guarded as run_guarded_child
from examples.weather_scenarios.fv_point_sector_policy import _digest, _key
from tests.test_fv_research_partial_observation import _problem


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "R4_ORIGINAL_TWO_HOLE_SIGNED_ETA_DESIGN_20261001.md"
PLAN_SHA256 = "878d2316bed7e43238c8a4572d9ca865713278ceedf66cd4999729505978540b"
PROBE_SOURCE_PIN = "134fc0419255d3a4a7536be5dd86d98cc304539650b06e8c3d55180f40e49209"
RUNNER_PATH = "examples/weather_scenarios/fv86_resource_runner.py"
RUNNER_SHA256 = "6242598d8b11a74ed44fcd419d83fd14cfcd637f34ee4746e1f5432554aeeb6d"
FIELD_ARCHIVE = EVIDENCE / "partial_field_correction_attempt1"
FIELD_CONTROL_PATH = FIELD_ARCHIVE / "field_correction.json"
FIELD_RUN_PATH = FIELD_ARCHIVE / "field_correction.run.json"
FIELD_RESOURCE_PATH = FIELD_ARCHIVE / "field_correction.resource.json"
FIELD_CONTROL_SHA256 = "8ad5b5e3d56a074ad79a28fa0f23c0cbdae745c26efed5101cd826fe2d9442cf"
FIELD_RUN_SHA256 = "30e660ad211a090a81055b040de06cf691f57b5c1a2ec59269ccc1a03ddf709d"
FIELD_RESOURCE_SHA256 = "c32e698f49a88b440898c7cf13f91bf8e1fa2ebc4f625b726ba848623225a6cc"
FIELD_FINAL_CONTROL_SHA256 = "a78d1b8e0cd8f997572f5e2a913f7023dcd6333c124ac4b7c9414358f8484e26"
FIELD_FINAL_SIGNATURE_SHA256 = "50d3b1a4bad6761b14806c62708400d87d16e506ab08ef545f7ba9d870bece6d"
PARAMETERS_SHA256 = "e3a45fac86e622332c6afd3472b46d58ef97cd37db1445260c86923a3c423bf9"
ETA0 = 9.714529998761101e-5
ETA_FACTORS = (-2.0, -1.0, 1.0, 2.0)
CONTROL_SIZE = 26
FIELD_COUNT = 20
GROWTH_COUNT = 1
PIVOT_FLOW_INDEX = 1
PIVOT_CONTROL_INDEX = FIELD_COUNT + PIVOT_FLOW_INDEX
STATIONARITY_TOLERANCE = 1e-10
TRUE_RESIDUAL_TOLERANCE = 1e-10
OBJECTIVE_ROUNDOFF_FACTOR = 128.0
MAX_NEWTON = 8
MAX_BACKTRACKS = 16
MAX_PCG = 80
WALL_SECONDS = 300
RSS_LIMIT_BYTES = 1024**3
SELF_PATH = "examples/weather_scenarios/fv_partial_signed_face_slices.py"
SOURCE_PATHS = tuple(dict.fromkeys((
    *seed_probe.SOURCE_PATHS,
    *prior.SOURCE_PATHS,
    "examples/weather_scenarios/fv_partial_field_correction_probe.py",
    "examples/weather_scenarios/fv_face_flux_coordinates.py",
    RUNNER_PATH,
    SELF_PATH,
)))
RAW_ARCHIVES = tuple(dict.fromkeys((
    seed_probe.SEED_ARCHIVE.resolve(),
    seed_probe.seed_gate.ARCHIVE.resolve(),
    FIELD_ARCHIVE.resolve(),
)))
_KNOWN_BRANCH_REFUSALS = frozenset({
    "minmod joint oracle left its strict smooth branch",
    "partial sector core margin resolvability refused",
})


class SliceIdentityRefusal(ValueError):
    """Pinned source, input, runtime, or archive identity did not match."""


class SliceBranchRefusal(ValueError):
    """A fixed-eta trial left its fresh strict branch or chart domain."""


class SliceAuditRefusal(ValueError):
    """A parent recomputation differed from a reported endpoint."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_probe_sha256(source: bytes) -> str:
    text = source.decode("utf-8")
    lines = text.splitlines(keepends=True)
    pin_lines = [i for i, line in enumerate(lines) if line.startswith("PROBE_SOURCE_PIN = ")]
    if len(pin_lines) != 1:
        raise SliceIdentityRefusal("probe source must contain one canonical pin declaration")
    index = pin_lines[0]
    newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
    lines[index] = f'PROBE_SOURCE_PIN = "<PIN>"{newline}'
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _verify_probe_bytes(source: bytes, expected_raw_sha256: str, expected_canonical_sha256: str) -> None:
    if (hashlib.sha256(source).hexdigest() != expected_raw_sha256
            or _canonical_probe_sha256(source) != expected_canonical_sha256):
        raise SliceIdentityRefusal("probe source does not match its caller and reviewed canonical pins")


def _verify_probe_file(expected_raw_sha256: str) -> None:
    try:
        source = Path(__file__).read_bytes()
    except OSError as error:
        raise SliceIdentityRefusal(f"probe source is unreadable: {error}") from error
    _verify_probe_bytes(source, expected_raw_sha256, PROBE_SOURCE_PIN)


def _verify_static_pins(expected_plan: str, expected_runner: str) -> None:
    if expected_plan != PLAN_SHA256 or expected_runner != RUNNER_SHA256:
        raise SliceIdentityRefusal("caller plan/runner pins differ from reviewed constants")
    try:
        current_plan, current_runner = _sha(PLAN), _sha(ROOT / RUNNER_PATH)
    except OSError as error:
        raise SliceIdentityRefusal(f"pinned plan or runner is unreadable: {error}") from error
    if current_plan != expected_plan or current_runner != expected_runner:
        raise SliceIdentityRefusal("reviewed plan or resource runner source changed")


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _runtime() -> dict[str, str]:
    return {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"}


def _assert_scalar_match(name: str, saved: float, actual: float) -> None:
    tolerance = (128 * torch.finfo(torch.float64).eps
                 * max(abs(saved), abs(actual), torch.finfo(torch.float64).tiny))
    if not math.isfinite(saved) or not math.isfinite(actual) or abs(saved - actual) > tolerance:
        raise SliceAuditRefusal(f"fresh endpoint {name} differs")


def _assert_payload_match(name: str, saved: Any, actual: Any) -> None:
    if isinstance(saved, dict) and isinstance(actual, dict) and saved.keys() == actual.keys():
        for key in saved:
            _assert_payload_match(f"{name}.{key}", saved[key], actual[key])
    elif isinstance(saved, list) and isinstance(actual, list) and len(saved) == len(actual):
        for index, (left, right) in enumerate(zip(saved, actual)):
            _assert_payload_match(f"{name}[{index}]", left, right)
    elif (isinstance(saved, (int, float)) and not isinstance(saved, bool)
          and isinstance(actual, (int, float)) and not isinstance(actual, bool)):
        if isinstance(saved, int) and isinstance(actual, int):
            if saved != actual:
                raise SliceAuditRefusal(f"fresh endpoint {name} differs")
        else:
            _assert_scalar_match(name, float(saved), float(actual))
    elif saved != actual:
        raise SliceAuditRefusal(f"fresh endpoint {name} differs")


def _wrapper_completion_status(child: dict[str, Any], resource: dict[str, Any],
                               audit_passed: bool) -> str:
    if child.get("execution_status") != "child_completed":
        return "preflight_refused" if child.get("execution_status") == "preflight_or_audit_refused" else "child_guard_refused"
    if resource.get("exit_code") != 0 or resource.get("resource_termination") is not None:
        return "guard_refused"
    return "completed" if audit_passed else "parent_audit_refused"


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _fresh_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("signed-face output path must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(path.resolve(strict=False).is_relative_to(archive) for archive in RAW_ARCHIVES):
        raise ValueError("signed-face output must be outside every input archive")


def _validate_runtime(archived: object, current: dict[str, str] | None = None) -> None:
    if archived != (current or _runtime()):
        raise SliceIdentityRefusal("pinned field-correction runtime differs from this host")


def _current_problem() -> tuple[Any, Tensor, dict[str, Any]]:
    try:
        input_identity = prior._preflight_identity()
        problem, _, parameters = _problem()
    except (OSError, ValueError, RuntimeError) as error:
        raise SliceIdentityRefusal(
            f"current source/input preflight refused: {type(error).__name__}: {error}"
        ) from error
    if problem.identity != input_identity["current_problem_identity"]:
        raise SliceIdentityRefusal("current problem identity differs from source-bound preflight")
    return problem, parameters, input_identity


def _field_endpoint_identity(problem: Any, parameters: Tensor) -> tuple[dict[str, Any], Tensor]:
    """Load the real pinned field endpoint only after archive/source/runtime checks."""
    try:
        if (_sha(FIELD_CONTROL_PATH) != FIELD_CONTROL_SHA256
                or _sha(FIELD_RUN_PATH) != FIELD_RUN_SHA256
                or _sha(FIELD_RESOURCE_PATH) != FIELD_RESOURCE_SHA256):
            raise SliceIdentityRefusal("hash-pinned field-correction records changed")
        field = json.loads(FIELD_CONTROL_PATH.read_text())
    except SliceIdentityRefusal:
        raise
    except (OSError, ValueError) as error:
        raise SliceIdentityRefusal(f"pinned field-correction records unreadable: {error}") from error
    _validate_runtime(field.get("environment"))
    try:
        seed, seed_control = seed_probe._seed_evidence()
        current_input = prior._preflight_identity()
        current_sources = _sources()
        current_raw = seed_probe._seed_record_hashes()
    except (OSError, ValueError, RuntimeError) as error:
        raise SliceIdentityRefusal(
            f"original seed/source/input/runtime gate refused: {type(error).__name__}: {error}"
        ) from error
    archived_sources = field.get("source_before")
    checks = (
        isinstance(archived_sources, dict)
        and field.get("source_before") == field.get("source_after")
        and all(current_sources.get(name) == digest for name, digest in archived_sources.items()),
        field.get("input_before") == field.get("input_after") == current_input,
        field.get("seed_raw_hashes_before") == field.get("seed_raw_hashes_after") == current_raw,
        field.get("parameters_sha256") == PARAMETERS_SHA256 == _tensor_sha(parameters),
        field.get("seed_control_sha256") == _tensor_sha(seed_control),
        field.get("archive_manifest_sha256") == seed_probe.SEED_MANIFEST_SHA256,
        field.get("final_control_sha256") == FIELD_FINAL_CONTROL_SHA256,
        field.get("final_branch", {}).get("signature_sha256") == FIELD_FINAL_SIGNATURE_SHA256,
        field.get("numerical_status") == "block_stationary_candidate"
        and field.get("full_stationarity") is False
        and field.get("response_validation") == "not_performed",
    )
    if not all(checks):
        raise SliceIdentityRefusal("pinned field-correction endpoint identity/qualification changed")
    if not isinstance(seed, dict) or seed.get("selected_control_sha256") != seed_probe.SEED_CONTROL_SHA256:
        raise SliceIdentityRefusal("PR212 seed certificate did not validate on this runtime")
    control = torch.tensor(field.get("final_control"), dtype=torch.float64)
    if (control.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(control).all())
            or _tensor_sha(control) != FIELD_FINAL_CONTROL_SHA256):
        raise SliceIdentityRefusal("archived field-correction control failed its pinned hash")
    return field, control


def _chart_for_problem(problem: Any) -> face_chart.FVFaceFluxCoordinateChart:
    spec = problem.frozen.fv_transport
    if spec is None or problem.frozen.active_field_index.numel() != FIELD_COUNT:
        raise SliceIdentityRefusal("pinned problem no longer has the 20+5+1 control layout")
    chart = face_chart.FVFaceFluxCoordinateChart.from_basis(
        field_count=FIELD_COUNT, growth_count=GROWTH_COUNT,
        coefficient_limits=spec.coefficient_limits, psi_basis=spec.psi_basis,
        axis="y", face=(2, 0), pivot_index=PIVOT_FLOW_INDEX,
    )
    if chart.control_count != CONTROL_SIZE:
        raise SliceIdentityRefusal("signed-face chart control count changed")
    return chart


def _slice_control(
    chart: face_chart.FVFaceFluxCoordinateChart, tangent: Tensor, eta: Tensor,
) -> Tensor:
    if tangent.shape != (CONTROL_SIZE - 1,) or eta.numel() != 1:
        raise ValueError("signed-face slice expects 25 tangent controls and scalar eta")
    coordinates = torch.cat((tangent[:PIVOT_CONTROL_INDEX], eta.reshape(1),
                             tangent[PIVOT_CONTROL_INDEX:]))
    return chart.from_face_coordinates(coordinates)


def _tangent_coordinates(
    chart: face_chart.FVFaceFluxCoordinateChart, control: Tensor,
) -> tuple[Tensor, Tensor]:
    coordinates = chart.to_face_coordinates(control)
    eta = coordinates[PIVOT_CONTROL_INDEX]
    tangent = torch.cat((coordinates[:PIVOT_CONTROL_INDEX],
                         coordinates[PIVOT_CONTROL_INDEX + 1:]))
    return tangent, eta


def _slice_objective(
    objective: Callable[[Tensor, Tensor], Tensor],
    chart: face_chart.FVFaceFluxCoordinateChart,
    eta: Tensor,
) -> Callable[[Tensor, Tensor], Tensor]:
    def transformed(tangent: Tensor, parameters: Tensor) -> Tensor:
        return objective(_slice_control(chart, tangent, eta), parameters)
    return transformed


def _face_flux(problem: Any, control: Tensor) -> Tensor:
    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("face flux requires the fixed FV transport basis")
    coefficient = bounded_fv_coefficients(
        control[FIELD_COUNT:FIELD_COUNT + 5],
        psi_basis=spec.psi_basis,
        coefficient_limits=spec.coefficient_limits,
        dt_seconds=60.0 * problem.frozen.nowcast_config.interval_minutes
        / spec.substeps_per_interval,
        spacing_yx=spec.spacing_yx,
        reconstruction=spec.reconstruction,
    )
    psi = torch.einsum("k,kij->ij", coefficient, spec.psi_basis)
    _, qy = face_volume_fluxes(psi)
    return qy[2, 0]


def _eta_roundoff_tolerance(
    chart: face_chart.FVFaceFluxCoordinateChart, control: Tensor,
) -> float:
    """Scale FP64 face-coordinate tolerance by the absolute contributing terms."""
    flow = control[chart.field_count:chart.field_count + chart.flow_count]
    normalized_terms = chart.normalized_weights * torch.tanh(flow)
    return 128 * torch.finfo(control.dtype).eps * float(normalized_terms.abs().sum())


def _branch_summary(problem: Any, control: Tensor, parameters: Tensor) -> dict[str, Any]:
    try:
        branch, _, face_margin = preflight.branch_with_face_margin(problem, control, parameters)
    except ValueError as error:
        if str(error) in _KNOWN_BRANCH_REFUSALS:
            raise SliceBranchRefusal(str(error)) from error
        raise
    summary = field_probe._branch_summary(branch, face_margin)
    if (summary["euler_stages"] != 54
            or not math.isfinite(summary["minimum_scaled_slope_margin"])
            or summary["minimum_scaled_slope_margin"] <= 0.0
            or not math.isfinite(face_margin) or face_margin <= 0.0):
        raise SliceBranchRefusal("full-trace strict branch margins are not positive")
    return summary


def _require_slice_signature(expected: str | None, actual: str) -> None:
    if expected is not None and actual != expected:
        raise SliceBranchRefusal("trial signature differs from this slice start")


def _endpoint(
    point_id: str,
    kind: str,
    control: Tensor,
    eta: Tensor,
    problem: Any,
    chart: face_chart.FVFaceFluxCoordinateChart,
    parameters: Tensor,
    branch: dict[str, Any],
    *,
    slice_gradient: Tensor | None = None,
) -> dict[str, Any]:
    if control.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(control).all()):
        raise ValueError("endpoint control must be a finite 26-vector")
    j = problem.objective(control, parameters)
    g = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    if j.ndim != 0 or not bool(torch.isfinite(j)) or not bool(torch.isfinite(g).all()):
        raise RefinementNumericalRefusal("endpoint objective or full gradient is nonfinite")
    tangent, eta_from_chart = _tangent_coordinates(chart, control)
    production_eta = _face_flux(problem, control) / chart.face_scale
    coordinate_tolerance = _eta_roundoff_tolerance(chart, control)
    if (abs(float(eta_from_chart - eta)) > coordinate_tolerance
            or abs(float(production_eta - eta)) > coordinate_tolerance):
        raise SliceAuditRefusal("endpoint face coordinate differs from requested eta")
    eta_tensor = eta.reshape(()).clone()
    c_eta = torch.func.jvp(
        lambda value: _slice_control(chart, tangent, value),
        (eta_tensor,), (torch.ones_like(eta_tensor),),
    )[1]
    face_gradient = torch.func.grad(lambda value: _face_flux(problem, value))(control)
    face_norm = torch.linalg.vector_norm(face_gradient)
    if (not bool(torch.isfinite(c_eta).all()) or not bool(torch.isfinite(face_gradient).all())
            or not bool(torch.isfinite(face_norm)) or float(face_norm) <= 0.0):
        raise RefinementNumericalRefusal("endpoint face normal or chart tangent is not finite/nonzero")
    sigma = torch.dot(g, c_eta)
    sigma_floor = 128 * torch.finfo(control.dtype).eps * torch.linalg.vector_norm(g) * torch.linalg.vector_norm(c_eta)
    normal_slope = torch.dot(g, face_gradient / face_norm)
    face_flux = _face_flux(problem, control)
    flux_normal_derivative = sigma / chart.face_scale
    diagnostic_scalars = torch.stack((j, face_flux, sigma, sigma_floor,
                                      flux_normal_derivative, normal_slope))
    if not bool(torch.isfinite(diagnostic_scalars).all()):
        raise RefinementNumericalRefusal("endpoint diagnostics contain a nonfinite scalar")
    blocks = field_probe._gradient_blocks(g)
    output = {
        "point_id": point_id,
        "kind": kind,
        "control": control.detach().tolist(),
        "control_sha256": _tensor_sha(control),
        "requested_eta": float(eta),
        "chart_eta": float(eta_from_chart),
        "production_eta": float(production_eta),
        "face_flux": float(face_flux),
        "objective": float(j),
        "parameters_sha256": _tensor_sha(parameters),
        "full_gradient": g.detach().tolist(),
        "gradient_blocks": blocks,
        "slice_gradient": None if slice_gradient is None else slice_gradient.detach().tolist(),
        "slice_gradient_inf": None if slice_gradient is None else float(slice_gradient.abs().max()),
        "slice_gradient_l2": None if slice_gradient is None else float(torch.linalg.vector_norm(slice_gradient)),
        "face_gradient_l2": float(face_norm),
        "envelope_sigma": float(sigma),
        "envelope_sigma_roundoff_floor": float(sigma_floor),
        "envelope_sigma_resolved": bool(sigma.abs() > sigma_floor),
        "physical_flux_normal_derivative": float(flux_normal_derivative),
        "unit_face_normal_slope": float(normal_slope),
        "branch": branch,
        "original_full_gradient_gate_passed": blocks["full_inf"] < STATIONARITY_TOLERANCE,
        "response_validation": "not_performed",
    }
    return output


def _strict_slice_branch(
    problem: Any,
    chart: face_chart.FVFaceFluxCoordinateChart,
    eta: Tensor,
    expected_signature: str | None,
):
    def check(tangent: Tensor, parameters: Tensor) -> tuple[dict[str, Any], str]:
        try:
            control = _slice_control(chart, tangent, eta)
        except RuntimeError as error:
            message = str(error)
            if "strict representable chart domain" in message or "must be finite" in message:
                raise SliceBranchRefusal(message) from error
            raise
        try:
            summary = _branch_summary(problem, control, parameters)
            _require_slice_signature(expected_signature, summary["signature_sha256"])
            return summary, "strict full-trace branch"
        except ValueError as error:
            if isinstance(error, SliceBranchRefusal) or str(error) in _KNOWN_BRANCH_REFUSALS:
                raise SliceBranchRefusal(str(error)) from error
            raise RefinementCallbackError(f"unexpected strict-branch failure: {error}") from error
    return check


def _true_residual_monitor(records: list[dict[str, Any]], eta: float, state: dict[str, Any]):
    def factory(original: Callable[..., Any]):
        def solve(operator: Callable[[Tensor], Tensor], rhs: Tensor, **kwargs: Any):
            calls = 0
            row: dict[str, Any] = {
                "eta": eta,
                "input_tangent": state["tangent"].detach().tolist(),
                "input_signature": state["signature"],
                "rhs": rhs.detach().tolist(),
                "rtol": kwargs.get("rtol"),
                "max_iterations": kwargs.get("max_iterations"),
            }

            def measured(direction: Tensor) -> Tensor:
                nonlocal calls
                calls += 1
                return operator(direction)

            try:
                result = original(measured, rhs, **kwargs)
                row.update(solution=result.solution.detach().tolist(),
                           converged=result.converged, iterations=result.iterations,
                           reported_relative_residual=float(result.relative_residual))
                rhs_norm = torch.linalg.vector_norm(rhs)
                residual = measured(result.solution) - rhs
                residual_norm = torch.linalg.vector_norm(residual)
                relative = residual_norm / rhs_norm if float(rhs_norm) > 0 else residual_norm
                row.update(
                    true_residual=residual.detach().tolist(),
                    independently_recomputed_residual=residual.detach().tolist(),
                    independently_recomputed_residual_l2=float(residual_norm),
                    independently_recomputed_relative_residual=float(relative),
                    measured_operator_calls=calls - 1,
                    independent_residual_hvp_calls=1,
                )
                if (not bool(torch.isfinite(relative))
                        or float(relative) > TRUE_RESIDUAL_TOLERANCE):
                    raise RefinementNumericalRefusal(
                        "independently recomputed tangent PCG residual exceeds tolerance"
                    )
                return result
            except (RuntimeError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                row["total_operator_calls"] = calls
                records.append(row)
        return solve
    return factory


def _run_slice(
    index: int,
    factor: float,
    start_control: Tensor,
    problem: Any,
    chart: face_chart.FVFaceFluxCoordinateChart,
    parameters: Tensor,
    report: dict[str, Any],
    save: Callable[[], None],
) -> tuple[dict[str, Any], Tensor]:
    eta = start_control.new_tensor(ETA0 * factor)
    tangent, _ = _tangent_coordinates(chart, start_control)
    slice_record: dict[str, Any] = {
        "index": index, "factor": factor, "eta": float(eta),
        "status": "running", "start_control": start_control.detach().tolist(),
        "start_control_sha256": _tensor_sha(start_control),
        "start_tangent": tangent.detach().tolist(),
        "last_accepted_control": start_control.detach().tolist(),
        "last_accepted_control_sha256": _tensor_sha(start_control),
        "linear_solves": [], "accepted_steps": [], "rejected_trials": [],
        "refusal": None,
    }
    report["slices"].append(slice_record)
    save()
    objective = _slice_objective(problem.objective, chart, eta)
    last_accepted_tangent = tangent.detach().clone()
    last_accepted_control = start_control.detach().clone()
    try:
        start_branch = _branch_summary(problem, start_control, parameters)
        slice_record["start_branch"] = start_branch
        start_objective = problem.objective(start_control, parameters)
        if start_objective.ndim != 0 or not bool(torch.isfinite(start_objective)):
            raise _NonFiniteEvaluation("slice start objective is nonfinite or nonscalar")
        slice_record["start_objective"] = float(start_objective)
        signature = start_branch["signature_sha256"]
        branch_check = _strict_slice_branch(problem, chart, eta, signature)
        solver_state: dict[str, Any] = {"tangent": tangent.detach().clone(), "signature": signature}

        def accept(trial: RefinementTrial):
            return field_probe.objective_gate(trial)

        def observe_trial(row: dict[str, Any]) -> None:
            nonlocal last_accepted_tangent, last_accepted_control
            if row.get("accepted") is not True:
                slice_record["rejected_trials"].append({
                    key: row[key] for key in
                    ("iteration", "backtrack", "step_scale", "rejection", "branch_reason",
                     "policy_reason", "objective", "gradient_max", "linear_relative_residual",
                     "armijo_ratio", "normalized_slope", "hvp_count") if key in row
                })
                save()
                return
            candidate_tangent = torch.tensor(row["candidate_control"], dtype=torch.float64)
            candidate = _slice_control(chart, candidate_tangent, eta)
            branch_summary = _branch_summary(problem, candidate, parameters)
            if branch_summary["signature_sha256"] != signature:
                raise SliceAuditRefusal("accepted trial lost its slice signature")
            last_accepted_tangent = candidate_tangent.detach().clone()
            last_accepted_control = candidate.detach().clone()
            solver_state["tangent"] = last_accepted_tangent
            slice_record["last_accepted_control"] = last_accepted_control.tolist()
            slice_record["last_accepted_control_sha256"] = _tensor_sha(last_accepted_control)
            slice_record["accepted_steps"].append({
                "iteration": row["iteration"], "backtrack": row["backtrack"],
                "step_scale": row["step_scale"], "control_sha256": _tensor_sha(candidate),
                "objective": row["objective"], "gradient_max": row["gradient_max"],
                "signature_sha256": branch_summary["signature_sha256"],
            })
            save()

        try:
            with matrix_free.observe_pcg_calls(
                _true_residual_monitor(slice_record["linear_solves"], float(eta), solver_state),
            ):
                result = refine_stationary(
                    objective, tangent, parameters,
                    branch_check=branch_check,
                    max_iterations=MAX_NEWTON,
                    max_backtracks=MAX_BACKTRACKS,
                    pcg_max_iterations=MAX_PCG,
                    trial_acceptance=accept,
                    trial_observer=observe_trial,
                )
            final_tangent = result.control.detach().clone()
            final_control = _slice_control(chart, final_tangent, eta)
            final_branch = _branch_summary(problem, final_control, parameters)
            final_grad_x = torch.func.grad(objective)(final_tangent, parameters)
            fresh_gradient_max = float(final_grad_x.abs().max())
            _assert_scalar_match("refiner slice gradient max", result.gradient_max,
                                 fresh_gradient_max)
            final = _endpoint(
                f"slice-{index}-final", "final_endpoint", final_control, eta,
                problem, chart, parameters, final_branch,
                slice_gradient=final_grad_x,
            )
            slice_record["final_endpoint"] = final
            slice_record["refiner_gradient_max"] = result.gradient_max
            slice_record["fresh_slice_gradient_max"] = fresh_gradient_max
            slice_record["refiner_iterations"] = result.iterations
            slice_record["refiner_hvp_count"] = result.hvp_count
            slice_record["status"] = "tangent_stationary_candidate"
            if fresh_gradient_max >= STATIONARITY_TOLERANCE:
                slice_record["status"] = "slice_refused"
                slice_record["refusal"] = "fresh tangent-gradient gate failed after refiner return"
            next_control = final_control.detach().clone()
        except (RefinementNumericalRefusal, RefinementCallbackError) as error:
            slice_record["status"] = "slice_refused"
            slice_record["refusal"] = f"{type(error).__name__}: {error}"
            next_control = last_accepted_control
            last_branch = _branch_summary(problem, next_control, parameters)
            last_gradient = torch.func.grad(objective)(last_accepted_tangent, parameters)
            slice_record["final_endpoint"] = _endpoint(
                f"slice-{index}-last-accepted", "last_accepted_on_refusal",
                next_control, eta, problem, chart, parameters, last_branch,
                slice_gradient=last_gradient,
            )
    except _NonFiniteEvaluation as error:
        slice_record["status"] = "slice_start_refused"
        slice_record["start_refusal_kind"] = "nonfinite_initial_evaluation"
        slice_record["refusal"] = f"{type(error).__name__}: {error}"
        slice_record["start_tangent"] = tangent.detach().tolist()
        next_control = start_control.detach().clone()
    except (SliceBranchRefusal, RefinementNumericalRefusal,
            RefinementCallbackError) as error:
        slice_record["status"] = (
            "slice_refused" if "start_branch" in slice_record else "slice_start_refused"
        )
        slice_record["refusal"] = f"{type(error).__name__}: {error}"
        if "start_branch" not in slice_record:
            slice_record["start_refusal_kind"] = "strict_branch"
        next_control = (last_accepted_control.detach().clone()
                        if "start_branch" in slice_record else start_control.detach().clone())
        if "start_branch" in slice_record and "final_endpoint" not in slice_record:
            last_branch = _branch_summary(problem, next_control, parameters)
            last_gradient = torch.func.grad(objective)(last_accepted_tangent, parameters)
            slice_record["final_endpoint"] = _endpoint(
                f"slice-{index}-last-accepted", "last_accepted_on_refusal",
                next_control, eta, problem, chart, parameters, last_branch,
                slice_gradient=last_gradient,
            )
    save()
    return slice_record, next_control


def _fresh_endpoint_audit(
    point: dict[str, Any], problem: Any, chart: face_chart.FVFaceFluxCoordinateChart,
    parameters: Tensor,
) -> dict[str, Any]:
    control = torch.tensor(point.get("control"), dtype=torch.float64)
    if (control.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(control).all())
            or _tensor_sha(control) != point.get("control_sha256")):
        raise SliceAuditRefusal("reported endpoint control/hash is malformed")
    eta = control.new_tensor(point["requested_eta"])
    try:
        branch = _branch_summary(problem, control, parameters)
    except SliceBranchRefusal as error:
        raise SliceAuditRefusal(f"fresh endpoint no longer has a strict branch: {error}") from error
    tangent, _ = _tangent_coordinates(chart, control)
    slice_grad = torch.func.grad(_slice_objective(problem.objective, chart, eta))(tangent, parameters)
    fresh = _endpoint(point["point_id"], point["kind"], control, eta, problem,
                      chart, parameters, branch, slice_gradient=slice_grad)
    _assert_payload_match("endpoint", point, fresh)
    if point.get("parameters_sha256") != _tensor_sha(parameters):
        raise SliceAuditRefusal("endpoint parameter identity differs")
    return {"point_id": point["point_id"], "passed": True,
            "control_sha256": point["control_sha256"],
            "fresh_branch_sha256": branch["signature_sha256"],
            "slice_gradient_inf": fresh["slice_gradient_inf"]}


def _audit_linear_solves(
    slice_record: dict[str, Any], problem: Any,
    chart: face_chart.FVFaceFluxCoordinateChart, parameters: Tensor,
) -> int:
    eta = parameters.new_tensor(float(slice_record["eta"]))
    gradient = torch.func.grad(_slice_objective(problem.objective, chart, eta))
    start_signature = slice_record.get("start_branch", {}).get("signature_sha256")
    is_candidate = slice_record.get("status") == "tangent_stationary_candidate"
    refusal_status = slice_record.get("status") in {"slice_refused", "slice_start_refused"}
    audited = 0
    for row in slice_record.get("linear_solves", []):
        if (row.get("rtol") != TRUE_RESIDUAL_TOLERANCE
                or row.get("max_iterations") != MAX_PCG):
            raise SliceAuditRefusal("saved tangent PCG budget/tolerance differs from the plan")
        tangent_values, rhs_values = row.get("input_tangent"), row.get("rhs")
        if not isinstance(tangent_values, list) or not isinstance(rhs_values, list):
            raise SliceAuditRefusal("saved PCG input coordinates/RHS are missing")
        tangent = torch.tensor(tangent_values, dtype=torch.float64)
        rhs = torch.tensor(rhs_values, dtype=torch.float64)
        if (tangent.shape != (CONTROL_SIZE - 1,) or rhs.shape != tangent.shape
                or not bool(torch.isfinite(tangent).all()) or not bool(torch.isfinite(rhs).all())
                or row.get("input_signature") != start_signature
                or row.get("eta") != float(eta)):
            raise SliceAuditRefusal("saved PCG input coordinates/RHS/signature are invalid")
        current_gradient = gradient(tangent, parameters)
        expected_rhs = -current_gradient
        fresh_gradient_norm = torch.linalg.vector_norm(current_gradient)
        rhs_error = torch.linalg.vector_norm(rhs - expected_rhs)
        rhs_tolerance = 128 * torch.finfo(torch.float64).eps * float(fresh_gradient_norm)
        if float(rhs_error) > rhs_tolerance:
            raise SliceAuditRefusal("saved PCG RHS differs from fresh tangent gradient")
        has_solution = isinstance(row.get("solution"), list)
        if not has_solution:
            if row.get("error") is None or is_candidate:
                raise SliceAuditRefusal("PCG record omitted solution without a refusal")
            audited += 1
            continue
        if row.get("true_residual") is None and row.get("error") is not None and not is_candidate:
            audited += 1
            continue
        solution_values, residual_values = row.get("solution"), row.get("true_residual")
        if not isinstance(solution_values, list) or not isinstance(residual_values, list):
            if refusal_status and row.get("error") is not None:
                audited += 1
                continue
            raise SliceAuditRefusal("reported PCG solve omitted its solution/residual vectors")
        solution = torch.tensor(solution_values, dtype=torch.float64)
        saved_residual = torch.tensor(residual_values, dtype=torch.float64)
        if (solution.shape != tangent.shape or saved_residual.shape != tangent.shape
                or not bool(torch.isfinite(solution).all())
                or not bool(torch.isfinite(saved_residual).all())):
            raise SliceAuditRefusal("saved PCG solution or true residual is malformed")
        hessian_solution = torch.func.jvp(
            lambda value: gradient(value, parameters), (tangent,), (solution,),
        )[1]
        residual = hessian_solution + current_gradient
        if float(fresh_gradient_norm) <= 0.0:
            raise SliceAuditRefusal("saved PCG RHS is zero at a solve endpoint")
        relative = torch.linalg.vector_norm(residual) / fresh_gradient_norm
        residual_scale = torch.linalg.vector_norm(hessian_solution) + fresh_gradient_norm
        residual_tolerance = 128 * torch.finfo(torch.float64).eps * float(residual_scale)
        if not torch.allclose(saved_residual, residual, rtol=0.0, atol=residual_tolerance):
            raise SliceAuditRefusal("fresh PCG true residual vector differs")
        _assert_scalar_match("PCG true relative residual",
                             float(row.get("independently_recomputed_relative_residual", math.inf)),
                             float(relative))
        residual_refusal = "independently recomputed tangent PCG residual exceeds tolerance"
        if (row.get("converged") is True and float(relative) > TRUE_RESIDUAL_TOLERANCE
                and (not refusal_status or row.get("error") is None
                     or residual_refusal not in str(row.get("error")))):
            raise SliceAuditRefusal("fresh PCG true residual exceeds its convergence gate")
        if (is_candidate and (row.get("converged") is not True
                              or float(relative) > TRUE_RESIDUAL_TOLERANCE
                              or row.get("error") is not None)):
            raise SliceAuditRefusal("slice candidate contains a failed PCG solve")
        audited += 1
    return audited


def _audit_candidate_counts(slice_record: dict[str, Any]) -> None:
    """Require one independently auditable solve and accepted update per Newton step."""
    iterations = slice_record.get("refiner_iterations")
    hvp_count = slice_record.get("refiner_hvp_count")
    solves = slice_record.get("linear_solves")
    accepted = slice_record.get("accepted_steps")
    if (not isinstance(iterations, int) or iterations < 0
            or not isinstance(hvp_count, int) or hvp_count < 0
            or not isinstance(solves, list) or not isinstance(accepted, list)):
        raise SliceAuditRefusal("stationary candidate omitted refiner counts or evidence lists")
    if len(solves) != iterations or len(accepted) != iterations:
        raise SliceAuditRefusal(
            "stationary candidate solve/accepted-step counts differ from refiner iterations"
        )
    if iterations == 0 and hvp_count != 0:
        raise SliceAuditRefusal("zero-step stationary candidate reports Hessian-vector products")


def _audit_nonfinite_start(
    start_control: Tensor, problem: Any,
    chart: face_chart.FVFaceFluxCoordinateChart, parameters: Tensor,
    eta: float,
) -> None:
    tangent, _ = _tangent_coordinates(chart, start_control)
    objective = _slice_objective(problem.objective, chart, parameters.new_tensor(eta))
    value = objective(tangent, parameters)
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        return
    gradient = torch.func.grad(objective)(tangent, parameters)
    if not bool(torch.isfinite(gradient).all()):
        return
    raise SliceAuditRefusal("child reported nonfinite startup at a finite objective and gradient")


def _audit_slice(
    slice_record: dict[str, Any], problem: Any,
    chart: face_chart.FVFaceFluxCoordinateChart, parameters: Tensor,
) -> tuple[list[dict[str, Any]], int]:
    factor_value = slice_record.get("factor")
    if not isinstance(factor_value, (int, float)):
        raise SliceAuditRefusal("slice factor is missing or malformed")
    factor, status = float(factor_value), slice_record.get("status")
    eta_expected = ETA0 * factor
    refusal = slice_record.get("refusal")
    if status == "slice_start_refused":
        if not isinstance(refusal, str) or not refusal:
            raise SliceAuditRefusal("slice-start refusal lacks a reason")
        refusal_kind = slice_record.get("start_refusal_kind")
        if refusal_kind == "chart_domain":
            tangent_values = slice_record.get("start_tangent")
            if not isinstance(tangent_values, list):
                raise SliceAuditRefusal("chart-domain refusal omitted tangent coordinates")
            try:
                _slice_control(chart, torch.tensor(tangent_values, dtype=torch.float64),
                               parameters.new_tensor(eta_expected))
            except RuntimeError as error:
                if "strict representable chart domain" not in str(error):
                    raise SliceAuditRefusal(f"unexpected chart-start refusal: {error}") from error
            else:
                raise SliceAuditRefusal("child refused a representable chart start")
            return [], 0
        start_control = torch.tensor(slice_record.get("start_control"), dtype=torch.float64)
        if (start_control.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(start_control).all())
                or _tensor_sha(start_control) != slice_record.get("start_control_sha256")):
            raise SliceAuditRefusal("slice-start refusal lacks a finite control/hash")
        start_chart_eta = chart.to_face_coordinates(start_control)[PIVOT_CONTROL_INDEX]
        start_production_eta = _face_flux(problem, start_control) / chart.face_scale
        eta_tol = _eta_roundoff_tolerance(chart, start_control)
        if (abs(float(start_chart_eta) - eta_expected) > eta_tol
                or abs(float(start_production_eta) - eta_expected) > eta_tol):
            raise SliceAuditRefusal("slice-start refusal control differs from its requested eta")
        if refusal_kind == "strict_branch":
            try:
                _branch_summary(problem, start_control, parameters)
            except SliceBranchRefusal:
                pass
            else:
                raise SliceAuditRefusal("child refused a start whose strict branch passes")
        elif refusal_kind == "nonfinite_initial_evaluation":
            _branch_summary(problem, start_control, parameters)
            _audit_nonfinite_start(start_control, problem, chart, parameters, eta_expected)
        else:
            raise SliceAuditRefusal("slice-start refusal has an unknown cause")
        return [], 0

    start_control = torch.tensor(slice_record.get("start_control"), dtype=torch.float64)
    if (start_control.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(start_control).all())
            or _tensor_sha(start_control) != slice_record.get("start_control_sha256")):
        raise SliceAuditRefusal("slice start control/hash is malformed")
    start_chart_eta = chart.to_face_coordinates(start_control)[PIVOT_CONTROL_INDEX]
    start_production_eta = _face_flux(problem, start_control) / chart.face_scale
    eta_tol = _eta_roundoff_tolerance(chart, start_control)
    if (abs(float(start_chart_eta) - eta_expected) > eta_tol
            or abs(float(start_production_eta) - eta_expected) > eta_tol):
        raise SliceAuditRefusal("slice start differs from its predeclared eta")
    start_branch = _branch_summary(problem, start_control, parameters)
    _assert_payload_match("slice start branch", slice_record.get("start_branch"), start_branch)
    _assert_scalar_match("slice start objective", float(slice_record["start_objective"]),
                         float(problem.objective(start_control, parameters)))
    signature = start_branch["signature_sha256"]
    for step in slice_record.get("accepted_steps", []):
        if (step.get("signature_sha256") != signature
                or not all(math.isfinite(float(step.get(key, math.nan)))
                           for key in ("objective", "gradient_max", "step_scale"))):
            raise SliceAuditRefusal("accepted inner step changed signature or has invalid metrics")
    linear_count = _audit_linear_solves(slice_record, problem, chart, parameters)
    final = slice_record.get("final_endpoint")
    if status not in {"tangent_stationary_candidate", "slice_refused"}:
        raise SliceAuditRefusal(f"unknown slice status: {status}")
    if not isinstance(final, dict):
        raise SliceAuditRefusal("slice result/refusal omitted its final or last-accepted endpoint")
    if (slice_record.get("last_accepted_control_sha256") != final.get("control_sha256")
            or _tensor_sha(torch.tensor(slice_record.get("last_accepted_control"),
                                        dtype=torch.float64)) != final.get("control_sha256")):
        raise SliceAuditRefusal("saved last-accepted control differs from the slice endpoint")
    if (final.get("requested_eta") != eta_expected
            or final.get("branch", {}).get("signature_sha256") != signature):
        raise SliceAuditRefusal("final endpoint eta/signature differs from its slice")
    endpoint_audit = _fresh_endpoint_audit(final, problem, chart, parameters)
    if status == "tangent_stationary_candidate":
        _audit_candidate_counts(slice_record)
        if (endpoint_audit["slice_gradient_inf"] >= STATIONARITY_TOLERANCE
                or refusal is not None):
            raise SliceAuditRefusal("tangent-stationary status disagrees with fresh gradient/refusal")
        _assert_scalar_match("refiner fresh slice gradient max",
                             float(slice_record["fresh_slice_gradient_max"]),
                             endpoint_audit["slice_gradient_inf"])
        _assert_scalar_match("refiner reported slice gradient max",
                             float(slice_record["refiner_gradient_max"]),
                             endpoint_audit["slice_gradient_inf"])
        if any(row.get("converged") is not True or row.get("error") is not None
               or float(row.get("independently_recomputed_relative_residual", math.inf))
               > TRUE_RESIDUAL_TOLERANCE for row in slice_record.get("linear_solves", [])):
            raise SliceAuditRefusal("tangent-stationary status includes a failed PCG solve")
    elif not isinstance(refusal, str) or not refusal:
        raise SliceAuditRefusal("slice refusal omitted its numerical reason")
    return [endpoint_audit], linear_count


def _run_child(output: Path, *, expected_plan_sha256: str,
               expected_probe_sha256: str, expected_runner_sha256: str) -> dict[str, Any]:
    _fresh_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("signed-face temporary output must be fresh")
    report: dict[str, Any] = {
        "execution_status": "running", "numerical_status": "running",
        "scope": "four signed-eta 25-coordinate slice solves; no full root or response",
        "full_root_claim": False, "response_validation": "not_performed",
        "plan_sha256": expected_plan_sha256,
        "reviewed_probe_sha256": expected_probe_sha256,
        "reviewed_runner_sha256": expected_runner_sha256,
        "environment": _runtime(), "source_before": None, "input_before": None,
        "raw_hashes_before": None, "parameters_sha256": None,
        "field_correction_control_sha256": None,
        "eta0": ETA0, "predeclared_eta_factors": list(ETA_FACTORS),
        "slices": [], "refusal": None,
    }

    def save() -> None:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    try:
        _verify_static_pins(expected_plan_sha256, expected_runner_sha256)
        _verify_probe_file(expected_probe_sha256)
        try:
            source_before = _sources()
        except OSError as error:
            raise SliceIdentityRefusal(f"current source snapshot failed: {error}") from error
        problem, parameters, input_before = _current_problem()
        field, control = _field_endpoint_identity(problem, parameters)
        if (_tensor_sha(parameters) != PARAMETERS_SHA256
                or field["input_before"] != input_before):
            raise SliceIdentityRefusal("parameter/current-input identity differs from pinned endpoint")
        try:
            archived_base_branch = _branch_summary(problem, control, parameters)
        except SliceBranchRefusal as error:
            raise SliceIdentityRefusal(
                f"pinned archived base control no longer passes strict branch checks: {error}"
            ) from error
        saved_base_branch = field["final_branch"]
        if (archived_base_branch["signature_sha256"] != FIELD_FINAL_SIGNATURE_SHA256
                or archived_base_branch["signature_sha256"]
                != saved_base_branch.get("signature_sha256")
                or archived_base_branch["euler_stages"] != 54
                or abs(archived_base_branch["minimum_scaled_face_flux_margin"]
                       - saved_base_branch["minimum_scaled_face_flux_margin"])
                > 128 * torch.finfo(torch.float64).eps
                or abs(archived_base_branch["minimum_scaled_slope_margin"]
                       - saved_base_branch["minimum_scaled_slope_margin"])
                > 128 * torch.finfo(torch.float64).eps):
            raise SliceIdentityRefusal("fresh archived-control branch differs before coordinate conversion")
        chart = _chart_for_problem(problem)
        archived_eta_chart = chart.to_face_coordinates(control)[PIVOT_CONTROL_INDEX]
        archived_face_flux = _face_flux(problem, control)
        archived_eta_production = archived_face_flux / chart.face_scale
        eta_tolerance = _eta_roundoff_tolerance(chart, control)
        if abs(float(archived_eta_production) - ETA0) > eta_tolerance:
            raise SliceIdentityRefusal("production eta differs from pinned face-only eta0")
        if abs(float(archived_eta_chart) - ETA0) > eta_tolerance:
            raise SliceIdentityRefusal("chart eta differs from pinned face-only eta0 beyond roundoff")
        base_tangent, _ = _tangent_coordinates(chart, control)
        projected_positive_start = _slice_control(
            chart, base_tangent, control.new_tensor(ETA0),
        )
        roundtrip = chart.from_face_coordinates(chart.to_face_coordinates(control))
        raw_before = seed_probe._seed_record_hashes()
        field_hashes_before = {str(path): _sha(path) for path in
                               (FIELD_CONTROL_PATH, FIELD_RUN_PATH, FIELD_RESOURCE_PATH)}
        report.update(
            source_before=source_before, input_before=input_before,
            raw_hashes_before=raw_before, field_record_hashes_before=field_hashes_before,
            parameters_sha256=_tensor_sha(parameters),
            field_correction_control_sha256=_tensor_sha(control),
            archived_base_branch=archived_base_branch,
            eta0=float(ETA0),
            archived_production_face_flux=float(archived_face_flux),
            archived_production_eta=float(archived_eta_production),
            archived_chart_eta=float(archived_eta_chart),
            chart_eta_roundoff_delta=float(archived_eta_chart - archived_eta_production),
            chart_roundtrip_control_linf=float((roundtrip - control).abs().max()),
            positive_eta0_start_control_sha256=_tensor_sha(projected_positive_start),
            positive_eta0_start_displacement_linf=float((projected_positive_start - control).abs().max()),
        )
        save()
        working_by_sign: dict[int, Tensor] = {1: control.detach().clone(), -1: control.detach().clone()}
        process_order = (-1.0, -2.0, 1.0, 2.0)
        for index, factor in enumerate(process_order):
            sign = -1 if factor < 0 else 1
            previous = working_by_sign[sign]
            if abs(factor) == 1.0:
                start_tangent = base_tangent
            else:
                start_tangent = _tangent_coordinates(chart, previous)[0]
            try:
                start = _slice_control(chart, start_tangent,
                                       control.new_tensor(ETA0 * factor))
            except RuntimeError as error:
                if "strict representable chart domain" not in str(error):
                    raise
                report["slices"].append({
                    "index": index, "factor": factor, "eta": ETA0 * factor,
                    "status": "slice_start_refused", "start_refusal_kind": "chart_domain",
                    "start_tangent": start_tangent.tolist(),
                    "refusal": f"chart_domain: {error}", "linear_solves": [],
                    "accepted_steps": [], "rejected_trials": [],
                })
                report["numerical_status"] = "partial_slice_refusal"
                save()
                continue
            record, next_control = _run_slice(
                index, factor, start, problem, chart, parameters, report, save,
            )
            working_by_sign[sign] = next_control
            if record["status"] != "tangent_stationary_candidate":
                report["numerical_status"] = "partial_slice_refusal"
            save()
        if report["numerical_status"] == "running":
            report["numerical_status"] = "four_slice_stationarity_candidates"
        _verify_static_pins(expected_plan_sha256, expected_runner_sha256)
        _verify_probe_file(expected_probe_sha256)
        try:
            source_after = _sources()
            input_after = prior._preflight_identity()
            raw_after = seed_probe._seed_record_hashes()
            field_hashes_after = {str(path): _sha(path) for path in
                                  (FIELD_CONTROL_PATH, FIELD_RUN_PATH, FIELD_RESOURCE_PATH)}
        except (OSError, ValueError, RuntimeError) as error:
            raise SliceIdentityRefusal(
                f"post-slice source/input/archive check refused: {type(error).__name__}: {error}"
            ) from error
        if (source_after != source_before or input_after != input_before
                or raw_after != raw_before or field_hashes_after != field_hashes_before):
            raise SliceIdentityRefusal("source/input/raw/archive identities changed during slices")
        report.update(source_after=source_after, input_after=input_after,
                      raw_hashes_after=raw_after, field_record_hashes_after=field_hashes_after,
                      execution_status="child_completed")
        save()
    except (SliceIdentityRefusal, SliceAuditRefusal) as error:
        report.update(execution_status="preflight_or_audit_refused",
                      numerical_status="identity_refused", refusal=f"{type(error).__name__}: {error}")
        save()
    return report


def _parent_audit(child: dict[str, Any]) -> dict[str, Any]:
    """Independently recompute every saved endpoint before accepting evidence."""
    _verify_static_pins(str(child.get("plan_sha256")),
                        str(child.get("reviewed_runner_sha256")))
    _verify_probe_file(str(child.get("reviewed_probe_sha256")))
    problem, parameters, input_before = _current_problem()
    try:
        current_sources = _sources()
        current_raw = seed_probe._seed_record_hashes()
        current_field_records = {str(path): _sha(path) for path in
                                 (FIELD_CONTROL_PATH, FIELD_RUN_PATH, FIELD_RESOURCE_PATH)}
    except (OSError, ValueError, RuntimeError) as error:
        raise SliceIdentityRefusal(f"parent source/archive snapshot failed: {error}") from error
    if (problem.identity != input_before["current_problem_identity"]
            or child.get("parameters_sha256") != _tensor_sha(parameters)
            or child.get("input_before") != input_before
            or child.get("input_after") != input_before
            or child.get("source_before") != current_sources
            or child.get("source_after") != current_sources
            or child.get("raw_hashes_before") != current_raw
            or child.get("raw_hashes_after") != current_raw
            or child.get("field_record_hashes_before") != current_field_records
            or child.get("field_record_hashes_after") != current_field_records):
        raise SliceAuditRefusal("parent audit current source/input/parameter identity differs")
    _, archived_control = _field_endpoint_identity(problem, parameters)
    if child.get("field_correction_control_sha256") != _tensor_sha(archived_control):
        raise SliceAuditRefusal("parent audit field-control archive identity differs")
    archived_eta_production = float(_face_flux(problem, archived_control)
                                    / _chart_for_problem(problem).face_scale)
    _assert_scalar_match("pinned eta0", ETA0, archived_eta_production)
    chart = _chart_for_problem(problem)
    endpoint_audits = []
    linear_solve_count = 0
    for slice_record in child.get("slices", []):
        points, solves = _audit_slice(slice_record, problem, chart, parameters)
        endpoint_audits.extend(points)
        linear_solve_count += solves
    if child.get("execution_status") == "child_completed":
        slice_records = child.get("slices", [])
        factors = sorted(float(row.get("factor")) for row in slice_records)
        if factors != [-2.0, -1.0, 1.0, 2.0]:
            raise SliceAuditRefusal("completed child omitted a predeclared signed eta slice")
        expected_status = (
            "four_slice_stationarity_candidates"
            if all(row.get("status") == "tangent_stationary_candidate" for row in slice_records)
            else "partial_slice_refusal"
        )
        if child.get("numerical_status") != expected_status:
            raise SliceAuditRefusal("child aggregate numerical status disagrees with slice outcomes")
    try:
        final_identity = (_sources(), prior._preflight_identity(),
                          seed_probe._seed_record_hashes())
    except (OSError, ValueError, RuntimeError) as error:
        raise SliceAuditRefusal(f"parent final source/input/raw audit refused: {error}") from error
    if (final_identity[0] != child.get("source_before")
            or final_identity[1] != input_before
            or final_identity[2] != child.get("raw_hashes_before")):
        raise SliceAuditRefusal("parent audit detected source/input/raw identity drift")
    _verify_static_pins(str(child.get("plan_sha256")),
                        str(child.get("reviewed_runner_sha256")))
    _verify_probe_file(str(child.get("reviewed_probe_sha256")))
    return {"passed": True, "endpoint_count": len(endpoint_audits),
            "linear_solve_count": linear_solve_count, "endpoints": endpoint_audits,
            "input_identity_sha256": problem.identity["fixed_problem_sha256"],
            "parameters_sha256": _tensor_sha(parameters)}


def run_guarded(
    output_dir: Path,
    *,
    expected_plan_sha256: str,
    expected_probe_sha256: str,
    expected_runner_sha256: str,
) -> dict[str, Any]:
    _fresh_path(output_dir)
    output_dir.mkdir(parents=True)
    child_output = output_dir / "signed_face_slices.json"
    run_path = output_dir / "signed_face_slices.run.json"
    resource_path = output_dir / "signed_face_slices.resource.json"
    log_path = output_dir / "signed_face_slices.log"
    run: dict[str, Any] = {
        "execution_status": "preflight_refused",
        "numerical_status": "not_a_root_or_response_result",
        "full_root_claim": False,
        "response_validation": "not_performed",
        "parent_audit": {"passed": False, "reason": "pins_not_validated"},
    }
    try:
        _verify_static_pins(expected_plan_sha256, expected_runner_sha256)
        _verify_probe_file(expected_probe_sha256)
    except SliceIdentityRefusal as error:
        run["parent_audit"] = {"passed": False, "reason": f"{type(error).__name__}: {error}"}
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
        return run
    command = [sys.executable, str(Path(__file__)), "--child",
               "--output", str(child_output),
               "--expected-plan-sha256", expected_plan_sha256,
               "--expected-probe-sha256", expected_probe_sha256,
               "--expected-runner-sha256", expected_runner_sha256]
    resource = run_guarded_child(
        command, wall_seconds=WALL_SECONDS, rss_bytes=RSS_LIMIT_BYTES,
        report_path=resource_path, log_path=log_path,
    )
    run.update({
        "execution_status": "child_guard_refused",
        "numerical_status": "not_a_root_or_response_result",
        "resource": resource,
        "child_json_sha256": _sha(child_output) if child_output.is_file() else None,
        "parent_audit": {"passed": False, "reason": "not_run"},
    })
    if child_output.is_file():
        child = json.loads(child_output.read_text())
        run["child"] = child
        if child.get("execution_status") == "preflight_or_audit_refused":
            run["numerical_status"] = child.get("numerical_status")
            run["execution_status"] = _wrapper_completion_status(child, resource, False)
            run["parent_audit"] = {"passed": False, "reason": child.get("refusal")}
        elif resource.get("exit_code") != 0 or resource.get("resource_termination") is not None:
            run["execution_status"] = _wrapper_completion_status(child, resource, False)
        elif child.get("source_before") is None:
            run["execution_status"] = _wrapper_completion_status(child, resource, False)
        else:
            try:
                audit = _parent_audit(child)
                run["parent_audit"] = audit
                run["execution_status"] = _wrapper_completion_status(
                    child, resource, bool(audit.get("passed")),
                )
                run["numerical_status"] = child.get("numerical_status")
            except (SliceAuditRefusal, SliceIdentityRefusal) as error:
                run["execution_status"] = "parent_audit_refused"
                run["parent_audit"] = {"passed": False, "reason": f"{type(error).__name__}: {error}"}
    temporary = run_path.with_suffix(run_path.suffix + ".tmp")
    temporary.write_text(json.dumps(run, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(run_path)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    args = parser.parse_args(argv)
    pins = {
        "expected_plan_sha256": args.expected_plan_sha256,
        "expected_probe_sha256": args.expected_probe_sha256,
        "expected_runner_sha256": args.expected_runner_sha256,
    }
    if args.child:
        if args.output is None or args.launch:
            parser.error("child mode requires --output and cannot be combined with --launch")
        _run_child(args.output, **pins)
    elif args.launch:
        if args.output_dir is None or args.output is not None:
            parser.error("launch mode requires --output-dir and cannot be combined with --output")
        result = run_guarded(args.output_dir, **pins)
        print(json.dumps({
            "execution_status": result["execution_status"],
            "numerical_status": result["numerical_status"],
            "response_validation": result["response_validation"],
            "parent_audit": result["parent_audit"],
        }, sort_keys=True))
    else:
        parser.error("select exactly one of --child or --launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
