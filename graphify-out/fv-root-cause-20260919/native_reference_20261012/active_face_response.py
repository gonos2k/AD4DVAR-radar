"""Gated reoptimized response on the qualified y[4,3] face chart."""
from __future__ import annotations

import hashlib
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
COMMON_PATH = HERE / "perturbation_response.py"
_common_spec = importlib.util.spec_from_file_location("native_face_response_common", COMMON_PATH)
if _common_spec is None or _common_spec.loader is None:
    raise RuntimeError("shared native-reference response helpers cannot be loaded")
common = importlib.util.module_from_spec(_common_spec)
sys.modules[_common_spec.name] = common
_common_spec.loader.exec_module(common)

FACE = ("y", 4, 3)
CONTROL_SIZE = 26
TANGENT_SIZE = 25
STATIONARITY_TOL = 1e-10
ROUNDING_MULTIPLIER = 128.0
TOTAL_WALL_CAP_SECONDS = 3600.0
PARAMETERS = ((0, "observation_dbz_0"), (12, "background_beta"))
QUALIFIER_PATH = HERE / "qualify_candidate.py"
_RED_QUALIFIER: Any | None = None


def _load_red_qualifier() -> Any:
    global _RED_QUALIFIER
    if _RED_QUALIFIER is not None:
        return _RED_QUALIFIER
    spec = importlib.util.spec_from_file_location("active_face_native_qualifier", QUALIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("pinned native qualifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _RED_QUALIFIER = module
    return _RED_QUALIFIER


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned file missing or unsafe: {path}")
    return _sha_bytes(path.read_bytes())


def _tensor_sha(value: Tensor) -> str:
    return common._tensor_sha(value)


def _write(output: Path, record: dict[str, Any]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


def _gate(output: Path, status: str, reason: str, *,
        base_sha: str | None = None, phase_plan_sha: str | None = None) -> dict[str, Any]:
    return _write(output, {"status": status, "reason": reason, "minimum_claim": False,
        "response_claim": False, "selection_use": False,
        "base_control_sha256": base_sha, "phase_plan_sha256": phase_plan_sha,
        "native_J_evaluations_started": 0, "forecast_score_evaluations_started": 0})


def _phase_plan(plan_path: Path, expected_sha: str, qualification: dict[str, Any],
        control: Tensor, parameters: Tensor) -> dict[str, Any]:
    if plan_path.is_symlink() or _sha(plan_path) != expected_sha:
        raise ValueError("active-face phase plan seal mismatch")
    plan = json.loads(plan_path.read_text())
    qualification_sha = hashlib.sha256(json.dumps(qualification, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    required = {
        "phase_kind": "active_face_reoptimized_parameter_response",
        "status": "frozen_active_face_response_plan",
        "root_launch_clearance": True,
        "continuation_plan_sha256": common.CONTINUATION_PLAN_SHA,
        "base_control_sha256": _tensor_sha(control),
        "base_parameters_sha256": _tensor_sha(parameters),
        "qualification_sha256": qualification_sha,
        "runner_sha256": _sha(SELF),
        "common_runner_sha256": _sha(COMMON_PATH),
    }
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise ValueError(f"active-face phase plan mismatch: {key}")
    budget = plan.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("active-face phase plan is missing its cumulative budget")
    elapsed_before = float(budget.get("charged_seconds_before", -1.0))
    phase_seconds = float(budget.get("phase_seconds", -1.0))
    native_j_before = int(budget.get("native_J_successes_before", -1))
    phase_j_limit = int(budget.get("phase_native_J_limit", -1))
    if (not math.isfinite(elapsed_before) or elapsed_before < 0
            or not math.isfinite(phase_seconds) or phase_seconds <= 0
            or elapsed_before + phase_seconds > TOTAL_WALL_CAP_SECONDS
            or native_j_before < 0 or phase_j_limit < 1):
        raise ValueError("active-face phase budget is invalid or exceeds the original wall cap")
    if plan.get("fixed_problem_sha256") != common.FIXED_PROBLEM_SHA:
        raise ValueError("active-face phase plan has the wrong fixed-problem identity")
    if (type(plan.get("armijo_iterations")) is not int or plan["armijo_iterations"] < 1
            or type(plan.get("armijo_backtracks")) is not int or plan["armijo_backtracks"] < 1):
        raise ValueError("active-face phase plan needs positive Armijo iteration and backtrack limits")
    return plan


def _full_control(chart: Any, tangent: Tensor) -> Tensor:
    pivot = int(chart.layout["face_flux_coordinate"])
    chart_coordinates = torch.cat((tangent[:pivot], tangent.new_zeros(1), tangent[pivot:]))
    return chart.from_face_coordinates(chart_coordinates)


def _tangent_coordinates(chart: Any, control: Tensor) -> Tensor:
    pivot = int(chart.layout["face_flux_coordinate"])
    full = chart.to_face_coordinates(control)
    if abs(float(full[pivot])) > 128 * torch.finfo(full.dtype).eps:
        raise ValueError("qualified active control does not lie on the chart's Q=0 face")
    return torch.cat((full[:pivot], full[pivot + 1:]))


def _event_control(record: dict[str, Any]) -> Tensor:
    value = record.get("event_control")
    if not isinstance(value, list):
        raise ValueError("active qualification is missing its event_control")
    result = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if result.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(result).all()):
        raise ValueError("qualified active event control is malformed")
    return result


def qualify_active_point(problem: Any, control: Tensor, parameters: Tensor,
        budget: dict[str, Any], *, chart: Any, base_parameters: Tensor,
        base_branch_signature: str) -> dict[str, Any]:
    """Fresh first-order active-face qualification for one explicit p vector.

    This computes no score, Hessian, or curvature claim. It audits the actual
    360-stage branch, two original-objective full gradients, the face normal,
    and chart tangents, while charging every native-J evaluation to `budget`.
    """
    if (control.shape != (CONTROL_SIZE,) or parameters.shape != (13,)
            or base_parameters.shape != (13,) or control.dtype != torch.float64
            or parameters.dtype != torch.float64 or base_parameters.dtype != torch.float64
            or control.device.type != "cpu" or parameters.device.type != "cpu"
            or base_parameters.device.type != "cpu"):
        raise ValueError("active point qualification requires CPU FP64 control[26] and parameters[13]")
    index = budget.get("declared_parameter_index")
    delta = budget.get("declared_parameter_delta")
    if index not in (0, 12) or not isinstance(delta, (int, float)):
        raise ValueError("active point callback must receive one declared parameter transition")
    expected_parameters = base_parameters.clone()
    expected_parameters[index] += float(delta)
    if not torch.equal(expected_parameters, parameters):
        raise ValueError("perturbed parameters differ from the single declared transition")
    if (budget.get("base_parameters_sha256") != _tensor_sha(base_parameters)
            or budget.get("face_pivot_index") != chart.pivot_index):
        raise ValueError("active point budget does not bind the base parameters/chart pivot")

    def check_budget() -> None:
        deadline = float(budget["phase_deadline_monotonic"])
        if time.monotonic() >= deadline:
            raise TimeoutError("active point qualification exceeded the shared phase deadline")
        if int(budget["native_J_successes_completed"]) >= int(budget["max_phase_native_J_successes"]):
            raise TimeoutError("active point qualification exhausted the shared phase objective allowance")

    continuation_parent, continuation = common._validate_parent_plan(problem)
    del continuation_parent
    if continuation.get("source_transition", {}).get("src/advar/transport.py", {}).get("after") != common.TRANSPORT_CURRENT_SHA:
        raise ValueError("active point lacks the pinned support-only transport transition")
    frozen = problem.contract(parameters)
    spec = frozen.fv_transport
    if spec is None:
        raise ValueError("active point fixed problem has no FV transport")
    from advar import transport

    coefficients = spec.coefficient_limits * torch.tanh(control[20:25])
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    qx, qy = transport.face_volume_fluxes(psi)
    q_value = qy[FACE[1], FACE[2]]
    all_flux = torch.cat((qx.reshape(-1), qy.reshape(-1)))
    flux_scale = all_flux.abs().max()
    if not bool(torch.isfinite(flux_scale) & (flux_scale > 0)):
        raise ValueError("active point face flux scale is unresolved")
    q_residual_ratio = float(q_value.abs() / flux_scale)
    chart_coords = chart.to_face_coordinates(control)
    pivot_coordinate = int(chart.layout["face_flux_coordinate"])
    tangent = torch.cat((chart_coords[:pivot_coordinate], chart_coords[pivot_coordinate + 1:]))
    lifted = _full_control(chart, tangent)
    if _tensor_sha(lifted) != _tensor_sha(control):
        raise ValueError("active point is not exactly representable by the base Q=0 chart")

    qualifier = _load_red_qualifier()
    check_budget()
    budget["native_J_successes_started"] += 1
    branch = qualifier._analysis_branch(problem, control, parameters,
        deadline=float(budget["phase_deadline_monotonic"]))
    branch_j = branch.get("native_J_on_audited_path")
    if branch.get("objective_finite") is not True or not isinstance(branch_j, (float, int)):
        raise ValueError("actual 360-stage branch audit did not produce a finite native objective")
    budget["native_J_successes_completed"] += 1
    if (branch.get("classification") != "singleface"
            or branch.get("stage_count") != 360
            or branch.get("unique_nearzero_face_entities") != ["y:4:3"]
            or branch.get("unique_near_tie_slope_entities") != []
            or branch.get("actual_selector_signature_sha256") != base_branch_signature
            or q_residual_ratio > ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps):
        return {"status": "unsupported_perturbation_branch", "event_control": control.tolist(),
            "event_control_sha256": _tensor_sha(control), "input_parameters_sha256": _tensor_sha(parameters),
            "event_branch": branch, "same_actual_branch": False,
            "native_J": float(branch_j), "selection_use": False}

    full_gradients: dict[int, Tensor] = {}
    for side in (-1, 1):
        check_budget()
        budget["native_J_successes_started"] += 1
        with transport.selected_face_extension(FACE[0], FACE[1], FACE[2], side):
            full_gradients[side] = torch.func.grad(problem.objective, argnums=0)(control, parameters)
        if not bool(torch.isfinite(full_gradients[side]).all()):
            raise ValueError("active point full gradient is nonfinite")
        budget["native_J_successes_completed"] += 1
    from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart
    normal_chart = FVFaceFluxCoordinateChart.from_basis(field_count=20, growth_count=1,
        coefficient_limits=spec.coefficient_limits.clone(), psi_basis=spec.psi_basis,
        axis=FACE[0], face=(FACE[1], FACE[2]), pivot_index=chart.pivot_index)
    if normal_chart.identity["chart_sha256"] != chart.identity["chart_sha256"]:
        raise ValueError("active point chart differs from the frozen base chart")
    jacobian = torch.func.jacrev(lambda z: _full_control(chart, z))(tangent)
    tangent_gradients = {side: jacobian.T @ full_gradients[side] for side in (-1, 1)}
    normal = torch.func.grad(lambda c: face_flux(c, spec))(control)
    normal_norm = torch.linalg.vector_norm(normal)
    if not bool(torch.isfinite(normal_norm) & (normal_norm > torch.finfo(control.dtype).tiny)):
        raise ValueError("active face normal is unresolved")
    slopes = {side: float(torch.dot(full_gradients[side], normal) / normal_norm) for side in (-1, 1)}
    budgets = {side: ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps
        * float(torch.linalg.vector_norm(full_gradients[side])) for side in (-1, 1)}
    try:
        theta = qualifier.tangent.minimum_mixture_weight(full_gradients[-1], full_gradients[1])
    except ValueError as error:
        return {"status": "unsupported_perturbation_minmix", "event_control": control.tolist(),
            "event_control_sha256": _tensor_sha(control), "input_parameters_sha256": _tensor_sha(parameters),
            "event_branch": branch, "native_J": float(branch_j), "reason": str(error),
            "selection_use": False}
    mixed = (1 - theta) * full_gradients[-1] + theta * full_gradients[1]
    tangent_mixed = jacobian.T @ mixed
    tangent_inf = float(tangent_mixed.abs().max())
    full_mixed_inf = float(mixed.abs().max())
    agreement = float(torch.linalg.vector_norm(tangent_gradients[1] - tangent_gradients[-1]))
    tangent_scale = max(float(tangent_gradients[1].norm()), float(tangent_gradients[-1].norm()),
        torch.finfo(control.dtype).tiny)
    agreement_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * tangent_scale
    tangent_stationary = all(float(tangent_gradients[side].abs().max()) < STATIONARITY_TOL
        for side in (-1, 1)) and full_mixed_inf < STATIONARITY_TOL
    outward = slopes[-1] < -budgets[-1] and slopes[1] > budgets[1]
    status = ("singleface_first_order_qualified" if tangent_stationary and outward
        and agreement <= agreement_budget else "singleface_first_order_not_qualified")
    return {
        "status": status, "event_control": control.tolist(),
        "event_control_sha256": _tensor_sha(control),
        "input_parameters_sha256": _tensor_sha(parameters),
        "native_J": float(branch_j), "tangent_gradient_inf": tangent_inf,
        "full_mixed_gradient_inf": full_mixed_inf,
        "tangent_gradient_inf_by_side": {str(side): float(tangent_gradients[side].abs().max())
            for side in (-1, 1)},
        "full_side_gradient_inf_by_side": {str(side): float(full_gradients[side].abs().max())
            for side in (-1, 1)},
        "tangent_agreement_inf": agreement, "tangent_agreement_budget": agreement_budget,
        "tangent_stationary": tangent_stationary,
        "minimum_mixture_theta": float(theta), "one_sided_normal_slopes": {
            str(side): slopes[side] for side in (-1, 1)},
        "one_sided_normal_roundoff_budgets": {str(side): budgets[side] for side in (-1, 1)},
        "one_sided_normal_pass": outward,
        "same_actual_branch": branch.get("actual_selector_signature_sha256") == base_branch_signature,
        "actual_selector_signature_sha256": branch.get("actual_selector_signature_sha256"),
        "event_face": {"axis": FACE[0], "row": FACE[1], "column": FACE[2],
            "q_value": float(q_value), "chart_residual_ratio": q_residual_ratio,
            "chart": chart.identity},
        "event_branch": branch, "curvature_computed": False,
        "selection_use": False,
    }


def face_flux(control: Tensor, spec: Any) -> Tensor:
    from advar import transport

    coefficients = spec.coefficient_limits * torch.tanh(control[20:25])
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    _qx, qy = transport.face_volume_fluxes(psi)
    return qy[FACE[1], FACE[2]]


def _reduced_hessian(record: dict[str, Any], side: str) -> Tensor:
    entries = record.get("reduced_hessians_by_side", record.get("reduced_curvature_by_side"))
    if not isinstance(entries, dict) or not isinstance(entries.get(side), dict):
        raise ValueError("qualified base receipt is missing its exact reduced Hessian matrix")
    entry = entries[side]
    raw = entry.get("hessian", entry.get("matrix"))
    if not isinstance(raw, list):
        raise ValueError("RED qualification must store each exact 25x25 reduced Hessian")
    matrix = torch.as_tensor(raw, dtype=torch.float64, device="cpu")
    if matrix.shape != (TANGENT_SIZE, TANGENT_SIZE) or not bool(torch.isfinite(matrix).all()):
        raise ValueError("stored reduced Hessian must be finite 25x25 FP64")
    digest = entry.get("hessian_sha256", entry.get("matrix_sha256"))
    if digest != _tensor_sha(matrix) or entry.get("positive_definite") is not True:
        raise ValueError("stored reduced Hessian digest or positive-curvature claim failed")
    return matrix


def _qualified_perturbed_face(value: Any, control: Tensor, parameters: Tensor,
        chart: Any, base_signature: str) -> bool:
    if not isinstance(value, dict):
        return False
    face = value.get("event_face")
    branch = value.get("event_branch")
    parameter_sha = value.get("input_parameters_sha256")
    branch_signature = value.get("actual_selector_signature_sha256")
    if isinstance(branch, dict):
        branch_signature = branch.get("actual_selector_signature_sha256", branch_signature)
    if not isinstance(face, dict) or not isinstance(branch, dict):
        return False
    chart_record = face.get("chart")
    if not isinstance(chart_record, dict):
        return False
    face_entities = branch.get("unique_nearzero_face_entities")
    slope_entities = branch.get("unique_near_tie_slope_entities")
    full_gradients = value.get("full_side_gradient_inf_by_side")
    if not (isinstance(full_gradients, dict)
            and isinstance(value.get("one_sided_normal_slopes"), dict)
            and isinstance(value.get("one_sided_normal_roundoff_budgets"), dict)):
        return False
    try:
        gradient_inf = float(value["tangent_gradient_inf"])
        theta = float(value["minimum_mixture_theta"])
        q_value = float(face["q_value"])
        residual = float(face["chart_residual_ratio"])
        slopes = value["one_sided_normal_slopes"]
        budgets = value["one_sided_normal_roundoff_budgets"]
        full_minus, full_plus = float(full_gradients["-1"]), float(full_gradients["1"])
        gm, gp = float(slopes["-1"]), float(slopes["1"])
        bm, bp = float(budgets["-1"]), float(budgets["1"])
    except (KeyError, TypeError, ValueError):
        return False
    return (value.get("status") == "singleface_first_order_qualified"
        and parameter_sha == _tensor_sha(parameters)
        and value.get("event_control_sha256") == _tensor_sha(control)
        and value.get("tangent_stationary") is True
        and value.get("one_sided_normal_pass") is True
        and value.get("same_actual_branch") is True
        and isinstance(full_gradients, dict)
        and math.isfinite(full_minus) and math.isfinite(full_plus)
        and branch_signature == base_signature
        and branch.get("classification") == "singleface"
        and branch.get("stage_count", branch.get("actual_analysis_stage_count")) == 360
        and face_entities == ["y:4:3"] and slope_entities == []
        and face.get("axis") == FACE[0] and face.get("row") == FACE[1] and face.get("column") == FACE[2]
        and chart_record.get("chart_sha256") == chart.identity.get("chart_sha256")
        and math.isfinite(gradient_inf) and gradient_inf < STATIONARITY_TOL
        and math.isfinite(theta) and 0.0 <= theta <= 1.0
        and math.isfinite(q_value) and math.isfinite(residual)
        and residual <= ROUNDING_MULTIPLIER * torch.finfo(torch.float64).eps
        and gm < -bm and gp > bp)


def _exact_base_response(problem: Any, chart: Any, tangent: Tensor,
        parameters: Tensor, theta: Tensor, hessian: Tensor,
        parameter_index: int,
        charge_objective: Callable[[], None],
        charge_score: Callable[[], None]) -> dict[str, float]:
    from advar import transport

    def phi_side(z: Tensor, p: Tensor, selected_side: int) -> Tensor:
        control = _full_control(chart, z)
        with transport.selected_face_extension(FACE[0], FACE[1], FACE[2], selected_side):
            return problem.objective(control, p)

    def score(z: Tensor, p: Tensor) -> Tensor:
        control = _full_control(chart, z)
        charge_score()
        return problem.score(control, p)

    def mixed_gradient(z: Tensor, p: Tensor) -> Tensor:
        gm = torch.func.grad(lambda zz, pp: phi_side(zz, pp, -1), argnums=0)(z, p)
        gp = torch.func.grad(lambda zz, pp: phi_side(zz, pp, 1), argnums=0)(z, p)
        return (1.0 - theta) * gm + theta * gp

    p = parameters.detach().clone().requires_grad_(True)
    charge_score()
    ez = torch.func.grad(score, argnums=0)(tangent, p)
    charge_score()
    ep = torch.func.grad(score, argnums=1)(tangent, p)[parameter_index]
    direction = torch.zeros_like(p)
    direction[parameter_index] = 1.0
    charge_objective()
    charge_objective()
    fp = torch.func.jvp(lambda pp: mixed_gradient(tangent, pp), (p,), (direction,))[1]
    lam = torch.linalg.solve(hessian.T, ez)
    prediction = ep - torch.dot(lam, fp)
    if not bool(torch.isfinite(prediction)):
        raise ValueError("active-face exact adjoint response is nonfinite")
    return {"E_p": float(ep.detach()), "lambda_F_zp": float(torch.dot(lam, fp).detach()),
        "predicted_dE_dp": float(prediction.detach()),
        "linear_system_residual_norm": float(torch.linalg.vector_norm(hessian.T @ lam - ez).detach())}


def run_active_face_response(problem: Any, parameters: Tensor, qualification: dict[str, Any], *,
        phase_plan_path: Path, phase_plan_sha256: str,
        qualify_face: Callable[[Tensor, Tensor, dict[str, Any]], dict[str, Any]] | None = None,
        output: Path) -> dict[str, Any]:
    """Run eight actual local chart reoptimizations after base and launch gates."""
    if output.exists():
        raise ValueError("active-face response output must be fresh")
    invocation_start = time.monotonic()
    runner_sha_before = _sha(SELF)
    if (parameters.shape != (13,) or parameters.dtype != torch.float64
            or parameters.device.type != "cpu" or not bool(torch.isfinite(parameters).all())):
        raise ValueError("active-face parameter vector must be finite CPU FP64 length 13")
    if (qualification.get("status") == "qualified_original_native_J_two_face_minimum"
            or str(qualification.get("status", "")).startswith("two_face")):
        two_face_path = HERE / "active_two_face_response.py"
        two_face_spec = importlib.util.spec_from_file_location("native_active_twoface_response", two_face_path)
        if two_face_spec is None or two_face_spec.loader is None:
            raise RuntimeError("two-face active response driver cannot be loaded")
        two_face = importlib.util.module_from_spec(two_face_spec)
        sys.modules[two_face_spec.name] = two_face
        two_face_spec.loader.exec_module(two_face)
        return two_face.run_two_face_response(problem, parameters, qualification,
            phase_plan_path=phase_plan_path, phase_plan_sha256=phase_plan_sha256, output=output)
    if (qualification.get("status") != "singleface_local_minimum_qualified"
            or qualification.get("minimum_claim") is not True
            or qualification.get("P2_passed") is not True):
        return _gate(output, "gate_pending_base",
            "RED has not qualified the active face as a two-sided local minimum",
            base_sha=qualification.get("event_control_sha256"),
            phase_plan_sha=phase_plan_sha256)
    from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart

    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("active-face response requires the fixed minmod FV transport")
    chart_record = qualification.get("event_face", {}).get("chart", {})
    pivot_index = chart_record.get("pivot_flow_index")
    if type(pivot_index) is not int:
        return _gate(output, "gate_pending_base", "base qualification has no pinned chart pivot")
    chart = FVFaceFluxCoordinateChart.from_basis(field_count=20, growth_count=1,
        coefficient_limits=spec.coefficient_limits.clone(), psi_basis=spec.psi_basis,
        axis=FACE[0], face=(FACE[1], FACE[2]), pivot_index=pivot_index)
    if not common._qualified_singleface_base(qualification,
            _event_control(qualification), parameters, chart):
        return _gate(output, "gate_pending_base",
            "RED has not qualified the exact qy[4,3] event as a two-sided local minimum",
            base_sha=qualification.get("event_control_sha256"))
    base_control = _event_control(qualification)
    base_sha = _tensor_sha(base_control)
    try:
        plan = _phase_plan(phase_plan_path, phase_plan_sha256, qualification, base_control, parameters)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return _gate(output, "gate_pending_launch_clearance", str(error), base_sha=base_sha,
            phase_plan_sha=phase_plan_sha256)
    parent_plan, continuation = common._validate_parent_plan(problem)
    del parent_plan
    if continuation.get("source_transition", {}).get("src/advar/transport.py", {}).get("after") != common.TRANSPORT_CURRENT_SHA:
        return _gate(output, "gate_pending_source_transition", "approved support-only transition is not active",
            base_sha=base_sha, phase_plan_sha=phase_plan_sha256)

    side_signature = qualification.get("event_branch", {}).get("actual_selector_signature_sha256")
    if not isinstance(side_signature, str):
        return _gate(output, "gate_pending_base", "base event branch signature is missing", base_sha=base_sha)
    continuation_bundle_sha_before = _sha(common.CONTINUATION_WORK / "source.tar.gz")
    qualification_sha_before = hashlib.sha256(json.dumps(qualification, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    parameters_sha_before = _tensor_sha(parameters)
    face_qualifier = qualify_face
    if face_qualifier is None:
        def default_face_qualifier(control: Tensor, p: Tensor,
                shared_budget: dict[str, Any]) -> dict[str, Any]:
            return qualify_active_point(problem, control, p, shared_budget,
                chart=chart, base_parameters=parameters, base_branch_signature=side_signature)
        face_qualifier = default_face_qualifier
    tangent = _tangent_coordinates(chart, base_control)
    chart_base = _full_control(chart, tangent)
    if _tensor_sha(chart_base) != base_sha:
        return _gate(output, "gate_pending_base", "base control does not round-trip exactly through the Q=0 chart",
            base_sha=base_sha)

    try:
        hminus, hplus = _reduced_hessian(qualification, "-1"), _reduced_hessian(qualification, "1")
        theta_value = qualification.get("minimum_mixture_theta",
            qualification.get("minmix_theta", qualification.get("theta")))
        if theta_value is None:
            raise ValueError("qualified base lacks its mixture weight")
        theta = float(theta_value)
    except (ValueError, TypeError):
        return _gate(output, "gate_pending_base", "qualified base lacks exact two-sided reduced Hessian or mixture weight",
            base_sha=base_sha)
    theta_tensor = tangent.new_tensor(theta)
    if not (ROUNDING_MULTIPLIER * torch.finfo(tangent.dtype).eps < theta
            < 1.0 - ROUNDING_MULTIPLIER * torch.finfo(tangent.dtype).eps):
        return _gate(output, "gate_pending_base", "base minimum-mixture weight is not resolved in (0,1)",
            base_sha=base_sha, phase_plan_sha=phase_plan_sha256)
    metric = 0.5 * ((1.0 - theta_tensor) * hminus + theta_tensor * hplus
        + ((1.0 - theta_tensor) * hminus + theta_tensor * hplus).T)
    try:
        factor = torch.linalg.cholesky(metric)
    except RuntimeError:
        return _gate(output, "gate_pending_base", "mixture of the stored side Hessians is not positive definite",
            base_sha=base_sha)

    budget_plan = plan["budget"]
    phase_deadline = invocation_start + float(budget_plan["phase_seconds"])
    native_j_before = int(budget_plan["native_J_successes_before"])
    phase_limit = int(budget_plan["phase_native_J_limit"])
    budget: dict[str, Any] = {"native_J_successes_before": native_j_before,
        "native_J_successes_started": 0, "native_J_successes_completed": 0,
        "score_evaluations_started": 0, "score_evaluations_completed": 0,
        "max_phase_native_J_successes": phase_limit,
        "phase_deadline_monotonic": phase_deadline}

    def budget_failure(reason: str) -> dict[str, Any]:
        return _write(output, {"status": "partial_budget_exhausted", "reason": reason,
            "minimum_claim": False, "response_claim": False, "selection_use": False,
            "base_control_sha256": base_sha, "base_parameters_sha256": _tensor_sha(parameters),
            "phase_plan_sha256": phase_plan_sha256, "budget": budget,
            "elapsed_seconds": time.monotonic() - invocation_start})

    def check_budget() -> None:
        if time.monotonic() >= phase_deadline:
            budget_failure("active-face shared original-wall budget exhausted")
            raise TimeoutError("active-face shared original-wall budget exhausted")
        if int(budget["native_J_successes_completed"]) >= phase_limit:
            budget_failure("active-face phase native-J evaluation allowance exhausted")
            raise TimeoutError("active-face phase native-J evaluation allowance exhausted")

    def charge_objective() -> None:
        check_budget()
        budget["native_J_successes_started"] += 1
        budget["native_J_successes_completed"] += 1

    def charge_score() -> None:
        check_budget()
        budget["score_evaluations_started"] += 1
        budget["score_evaluations_completed"] += 1

    from advar import transport

    def phi(z: Tensor, p: Tensor) -> Tensor:
        control = _full_control(chart, z)
        with transport.selected_face_extension(FACE[0], FACE[1], FACE[2], 1):
            check_budget()
            budget["native_J_successes_started"] += 1
            value = problem.objective(control, p)
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError("face-restricted native J must be a finite scalar")
        budget["native_J_successes_completed"] += 1
        return value

    def phi_for_ad(z: Tensor, p: Tensor) -> Tensor:
        control = _full_control(chart, z)
        with transport.selected_face_extension(FACE[0], FACE[1], FACE[2], 1):
            return problem.objective(control, p)

    def value_and_grad(z: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        value = phi(z, p)
        check_budget()
        budget["native_J_successes_started"] += 1
        gradient = torch.func.grad(phi_for_ad, argnums=0)(z, p)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("active-face tangent native gradient is nonfinite")
        budget["native_J_successes_completed"] += 1
        return value, gradient

    try:
        base_native_j = float(phi(tangent, parameters))
    except TimeoutError:
        return _read_json(output)
    except (ValueError, RuntimeError) as error:
        return _write(output, {"status": "base_face_objective_failed", "reason": str(error),
            "minimum_claim": False, "response_claim": False, "selection_use": False,
            "base_control_sha256": base_sha, "base_parameters_sha256": _tensor_sha(parameters),
            "phase_plan_sha256": phase_plan_sha256, "budget": budget})
    rows: list[dict[str, Any]] = []
    for parameter_index, label in PARAMETERS:
        if parameter_index == 0:
            h = float(problem.observation_std_dbz.reshape(-1)[0]) / 100.0
        else:
            h = 1.0e-3
        try:
            adjoint = _exact_base_response(problem, chart, tangent, parameters, theta_tensor,
                metric, parameter_index, charge_objective, charge_score)
        except TimeoutError:
            return _read_json(output)
        except (ValueError, RuntimeError) as error:
            return _write(output, {"status": "base_adjoint_failed", "reason": str(error),
                "parameter_index": parameter_index, "minimum_claim": False,
                "response_claim": False, "selection_use": False,
                "base_control_sha256": base_sha, "base_parameters_sha256": _tensor_sha(parameters),
                "phase_plan_sha256": phase_plan_sha256, "budget": budget})
        points: list[dict[str, Any]] = []
        for scale in (1.0, -1.0, 0.5, -0.5):
            if not math.isfinite(h) or h <= 0:
                raise ValueError("active-face perturbation step must be finite and positive")
            p = parameters.clone()
            delta = scale * h
            p[parameter_index] += delta
            budget["base_parameters_sha256"] = _tensor_sha(parameters)
            budget["declared_parameter_index"] = parameter_index
            budget["declared_parameter_delta"] = delta
            budget["face_pivot_index"] = chart.pivot_index
            z = tangent.clone()  # Each sign starts at the qualified base; no linear predictor is used.
            trace: list[dict[str, Any]] = []
            converged = False
            for iteration in range(int(plan["armijo_iterations"])):
                check_budget()
                current_value, gradient = value_and_grad(z, p)
                grad_inf = float(gradient.abs().max())
                trace.append({"iteration": iteration, "objective": float(current_value),
                    "tangent_gradient_inf": grad_inf})
                if grad_inf < STATIONARITY_TOL:
                    converged = True
                    break
                direction = torch.cholesky_solve((-gradient).reshape(-1, 1), factor).reshape(-1)
                slope = float(torch.dot(gradient, direction))
                if not math.isfinite(slope) or slope >= 0:
                    break
                accepted = False
                for backtrack in range(int(plan["armijo_backtracks"])):
                    alpha = 0.5 ** backtrack
                    trial = z + alpha * direction
                    try:
                        trial_value = phi(trial, p)
                    except (ValueError, RuntimeError):
                        continue
                    if float(trial_value) <= float(current_value) + 1e-4 * alpha * slope:
                        z = trial
                        trace[-1]["accepted_alpha"] = alpha
                        accepted = True
                        break
                if not accepted:
                    break
            if not converged:
                partial = {"parameter_index": parameter_index, "scale": scale,
                    "status": "reoptimization_not_stationary", "trace": trace,
                    "budget": budget}
                _write(output, {"status": "partial_reoptimization_failure", "minimum_claim": False,
                    "response_claim": False, "selection_use": False, "base_control_sha256": base_sha,
                    "base_parameters_sha256": _tensor_sha(parameters), "partial_point": partial,
                    "budget": budget})
                return _read_json(output)
            candidate_control = _full_control(chart, z)
            check_budget()
            try:
                qualification_point = face_qualifier(candidate_control, p, budget)
            except TimeoutError:
                if output.exists():
                    return _read_json(output)
                return budget_failure("active point qualification exceeded the shared phase budget")
            except (ValueError, RuntimeError) as error:
                partial = {"parameter_index": parameter_index, "scale": scale,
                    "status": "perturbed_point_qualification_error", "reason": str(error),
                    "trace": trace, "budget": budget}
                return _write(output, {"status": "partial_branch_gate_failure", "minimum_claim": False,
                    "response_claim": False, "selection_use": False, "base_control_sha256": base_sha,
                    "base_parameters_sha256": _tensor_sha(parameters), "partial_point": partial,
                    "budget": budget})
            check_budget()
            actual_control = _event_control(qualification_point)
            if not _qualified_perturbed_face(qualification_point, actual_control, p, chart, side_signature):
                partial = {"parameter_index": parameter_index, "scale": scale,
                    "status": "perturbed_face_or_branch_gate_failed", "trace": trace,
                    "qualification": qualification_point, "budget": budget}
                _write(output, {"status": "partial_branch_gate_failure", "minimum_claim": False,
                    "response_claim": False, "selection_use": False, "base_control_sha256": base_sha,
                    "base_parameters_sha256": _tensor_sha(parameters), "partial_point": partial,
                    "budget": budget})
                return _read_json(output)
            check_budget()
            charge_score()
            score = problem.score(actual_control, p)
            if score.ndim != 0 or not bool(torch.isfinite(score)):
                raise ValueError("active-face future score must be a finite scalar")
            points.append({"scale": scale, "parameter_delta": delta,
                "optimizer_tangent_sha256": _tensor_sha(z),
                "qualified_control_sha256": _tensor_sha(actual_control),
                "parameters_sha256": _tensor_sha(p),
                "native_J": float(qualification_point["native_J"]), "score": float(score),
                "tangent_gradient_inf": float(qualification_point["tangent_gradient_inf"]),
                "branch_signature": side_signature, "armijo_trace": trace,
                "curvature_certified": False})
        by_scale = {float(item["scale"]): item for item in points}
        derivative_h = (float(by_scale[1.0]["score"]) - float(by_scale[-1.0]["score"])) / (2.0 * h)
        derivative_half = (float(by_scale[0.5]["score"]) - float(by_scale[-0.5]["score"])) / h
        rows.append({"parameter_index": parameter_index, "parameter": label, "h": h,
            "exact_active_face_adjoint": adjoint,
            "reoptimized_dE_dp_h": derivative_h,
            "reoptimized_dE_dp_h_over_2": derivative_half,
            "central_difference_refinement_change": derivative_half - derivative_h,
            "points": points})

    _parent_after, continuation_after = common._validate_parent_plan(problem)
    del _parent_after
    source_closure = (
        continuation_after.get("parent_native_plan_sha256") == common.REFERENCE_PLAN_SHA
        and _sha(common.CONTINUATION_WORK / "source.tar.gz") == continuation_bundle_sha_before
        and _tensor_sha(parameters) == parameters_sha_before
        and _tensor_sha(base_control) == base_sha
        and hashlib.sha256(json.dumps(qualification, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest() == qualification_sha_before
        and _sha(SELF) == runner_sha_before
    )
    elapsed = time.monotonic() - invocation_start
    if not source_closure:
        return _write(output, {"status": "source_input_closure_failed",
            "minimum_claim": False, "response_claim": False, "selection_use": False,
            "base_control_sha256": base_sha, "base_parameters_sha256": _tensor_sha(parameters),
            "phase_plan_sha256": phase_plan_sha256, "budget": budget})
    result = {"status": "completed", "phase_plan_sha256": phase_plan_sha256,
        "phase_plan_path": str(phase_plan_path), "base_control_sha256": base_sha,
        "base_parameters_sha256": _tensor_sha(parameters),
        "base_qualification_sha256": hashlib.sha256(json.dumps(qualification, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "continuation_plan_sha256": common.CONTINUATION_PLAN_SHA,
        "source_input_runtime_closure": True,
        "transport_support_only_transition": {"path": "src/advar/transport.py",
            "before": common.TRANSPORT_OLD_SHA, "after": common.TRANSPORT_CURRENT_SHA},
        "chart_identity": chart.identity, "face": FACE,
        "base_native_J": base_native_j, "base_mixture_theta": theta,
        "base_reduced_hessian_source": "RED exact same-point two-sided 25x25 curvature receipts; theta-weighted SPD preconditioner",
        "phase_budget": {**budget, "charged_seconds_before": budget_plan["charged_seconds_before"],
            "phase_seconds": budget_plan["phase_seconds"], "elapsed_seconds": elapsed,
            "global_elapsed_after_seconds": float(budget_plan["charged_seconds_before"]) + elapsed,
            "native_J_successes_after": native_j_before + int(budget["native_J_successes_completed"]),
            "no_new_3600_second_grant": True},
        "optimizer": "base-Hessian-preconditioned actual native-J Armijo on the fixed 25D chart; every sign warm-starts from base",
        "scope": "local conditional active-face response at one P2-qualified qy[4,3] base; not global or perturbed-minimum certification",
        "base_qualification_status": qualification.get("status"),
        "base_active_face_minimum_qualified": True,
        "perturbed_curvature_certified": False, "selection_use": False,
        "implicit_equation": "theta-weighted two-side tangent gradient with theta fixed at the qualified base; tangent-gradient agreement is required at every qualification",
        "minimum_claim": False, "response_claim": True, "parameters": rows}
    return _write(output, result)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True,
        help="RED same-point two-sided active-face minimum receipt")
    parser.add_argument("--phase-plan", type=Path, required=True,
        help="separately frozen cumulative-budget active-response plan")
    parser.add_argument("--phase-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as current_cycle

    case = current_cycle.load_fixed_case()
    qualification = _read_json(args.qualification)
    result = run_active_face_response(case["problem"], case["parameters"], qualification,
        phase_plan_path=args.phase_plan, phase_plan_sha256=args.phase_plan_sha256,
        qualify_face=None, output=args.output)
    print(json.dumps({key: result.get(key) for key in (
        "status", "reason", "base_control_sha256", "phase_plan_sha256",
        "response_claim", "phase_budget")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
