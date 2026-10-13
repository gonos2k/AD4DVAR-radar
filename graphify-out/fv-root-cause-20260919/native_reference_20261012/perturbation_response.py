"""Exact local adjoint and gated native Powell perturbation responses.

This module is intentionally inert on import.  Its expensive path is available
only after the caller supplies an eligible, independently qualified base.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import tarfile
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor


PARAMETERS = ((0, "observation_dbz_0", "observation_std_dbz[0,0]/100"),
    (12, "background_beta", "fixed background-pattern coefficient"))
POWELL = {"method": "Powell", "maxfev_per_start": 1000,
    "maxiter_per_start": 100, "xtol": 1e-11, "ftol": 1e-13}
QUALIFIED_STATUS = "qualified_regular_local_minimum"
ROOT = Path(__file__).resolve().parents[3]
REFERENCE_WORK = Path(__file__).resolve().parent
REFERENCE_PLAN_SHA = "b6f3a08e02ec6da684ea8ea7c9611e6eedf3b8de9de8a9ebc6830347dd13d9ac"
CONTINUATION_WORK = REFERENCE_WORK / "continuation3"
CONTINUATION_PLAN_SHA = "67fa44ac8c6c08f79df99a9fe2c5bee10b385a7769bb65ad6caafafeb9c52a01"
FIXED_PROBLEM_SHA = "16fa95b534c3b61cf5d1506770cbfc77edb527c40d1f4c536f5130cba24e1fc3"
TRANSPORT_OLD_SHA = "b0335b7a72ecd22084d5ab8ea0e46707240be15ef0039f4b31f96adbc937ee1a"
TRANSPORT_CURRENT_SHA = "c9dde24c12237d5ffee6b8ca0766b4d5619abf6110f3e858ee447bf49c1f7da9"
SCIPY_VERSION = "1.15.3"


def _tensor_sha(value: Tensor) -> str:
    if (value.dtype != torch.float64 or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())):
        raise ValueError("reference tensors must be finite CPU FP64")
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def _qualified_control(value: Any, fallback: Tensor) -> Tensor:
    if not isinstance(value, dict):
        return fallback
    candidate = value.get("qualified_control", value.get("control"))
    if isinstance(candidate, Tensor):
        point = candidate
    elif isinstance(candidate, (list, tuple)):
        point = torch.as_tensor(candidate, dtype=torch.float64, device="cpu")
    else:
        return fallback
    if (point.shape != (26,) or point.dtype != torch.float64 or point.device.type != "cpu"
            or not bool(torch.isfinite(point).all())):
        raise ValueError("qualified callback control must be finite CPU FP64 length 26")
    return point


def _qualified(value: Any, control: Tensor, parameters: Tensor, *,
        candidate_control: Tensor | None = None) -> bool:
    if not isinstance(value, dict):
        return False
    input_after = value.get("input_after")
    input_parameters = input_after.get("parameters_sha256") if isinstance(input_after, dict) else None
    branch = value.get("final_branch_before_hessian")
    branch_signature = (branch.get("actual_selector_signature_sha256")
        if isinstance(branch, dict) else value.get("branch_signature"))
    raw_gradient = value.get("native_gradient_norm", value.get("gradient_inf"))
    if raw_gradient is None:
        return False
    try:
        gradient_norm = float(raw_gradient)
    except (TypeError, ValueError):
        return False
    status_ok = value.get("status") in (QUALIFIED_STATUS, "smooth_local_minimum_qualified")
    if candidate_control is None:
        candidate_control = control
    raw_candidate_digest = value.get("candidate_control_sha256")
    control_digest = value.get("control_sha256")
    has_returned_control = isinstance(value.get("qualified_control", value.get("control")), (Tensor, list, tuple))
    control_binding_ok = (
        control_digest == _tensor_sha(control)
        if control_digest is not None else
        (raw_candidate_digest == _tensor_sha(candidate_control) and has_returned_control)
    )
    parameters_digest = value.get("parameters_sha256", input_parameters)
    positive_definite = value.get("positive_definite", value.get("full_hessian_positive_definite"))
    regular_branch = value.get("regular_branch", value.get("branch_stable"))
    if isinstance(branch, dict):
        regular_branch = (regular_branch is True
            and branch.get("classification") == "smooth_regular")
    return (status_ok
        and value.get("minimum_claim") is True
        and control_binding_ok
        and (raw_candidate_digest is None or raw_candidate_digest == _tensor_sha(candidate_control))
        and parameters_digest == _tensor_sha(parameters)
        and math.isfinite(gradient_norm) and gradient_norm < 1e-10
        and positive_definite is True and regular_branch is True
        and isinstance(branch_signature, str))


def _branch_signature(value: dict[str, Any]) -> str:
    branch = value.get("final_branch_before_hessian")
    signature = (branch.get("actual_selector_signature_sha256")
        if isinstance(branch, dict) else value.get("branch_signature"))
    if not isinstance(signature, str):
        raise ValueError("qualification is missing its actual selector signature")
    return signature


def _qualified_singleface_base(value: Any, event_control: Tensor, parameters: Tensor,
        chart: Any, *, stationarity_tolerance: float = 1e-10) -> bool:
    """Check the narrow RED qy[4,3] active-event qualification contract.

    This is a gate only; the smooth full-control Powell runner does not consume
    an active-face record. Active reoptimization needs its own frozen phase plan.
    """
    if not isinstance(value, dict):
        return False
    face = value.get("event_face")
    chart_record = face.get("chart") if isinstance(face, dict) else None
    branch = value.get("event_branch")
    branch_signature = branch.get("actual_selector_signature_sha256") if isinstance(branch, dict) else None
    input_after = value.get("input_after")
    parameter_sha = value.get("input_parameters_sha256")
    if parameter_sha is None and isinstance(input_after, dict):
        parameter_sha = input_after.get("parameters_sha256")
    theta = value.get("minimum_mixture_theta", value.get("minmix_theta", value.get("theta")))
    slopes = value.get("one_sided_normal_slopes")
    budgets = value.get("one_sided_normal_roundoff_budgets")
    tangent = value.get("tangent_gradient_inf_by_side")
    full_gradients = value.get("full_side_gradient_inf_by_side")
    curvature = value.get("reduced_curvature_by_side")
    if not (isinstance(face, dict) and isinstance(chart_record, dict)
            and isinstance(branch, dict) and isinstance(slopes, dict)
            and isinstance(budgets, dict) and isinstance(tangent, dict)
            and isinstance(full_gradients, dict) and isinstance(curvature, dict)
            and theta is not None):
        return False
    try:
        mix = float(theta)
        q_value = float(face["q_value"])
        chart_residual = float(face["chart_residual_ratio"])
        normal_minus, normal_plus = float(slopes["-1"]), float(slopes["1"])
        budget_minus, budget_plus = float(budgets["-1"]), float(budgets["1"])
        tangent_minus, tangent_plus = float(tangent["-1"]), float(tangent["1"])
        full_minus, full_plus = float(full_gradients["-1"]), float(full_gradients["1"])
    except (KeyError, TypeError, ValueError):
        return False
    chart_identity = chart.identity
    chart_sha = chart_identity.get("chart_sha256") if isinstance(chart_identity, dict) else None
    curvature_ok = all(isinstance(curvature.get(side), dict)
        and curvature[side].get("positive_definite") is True for side in ("-1", "1"))
    return (value.get("status") == "singleface_local_minimum_qualified"
        and value.get("minimum_claim") is True and value.get("P2_passed") is True
        and value.get("tangent_stationary") is True and value.get("one_sided_normal_pass") is True
        and parameter_sha == _tensor_sha(parameters)
        and value.get("event_control_sha256") == _tensor_sha(event_control)
        and branch.get("classification") == "singleface"
        and branch.get("stage_count", branch.get("actual_analysis_stage_count")) == 360
        and branch.get("unique_nearzero_face_entities") == ["y:4:3"]
        and branch.get("unique_near_tie_slope_entities") == []
        and isinstance(face, dict) and face.get("axis") == "y"
        and face.get("row") == 4 and face.get("column") == 3
        and isinstance(chart_record, dict) and chart_record.get("chart_sha256") == chart_sha
        and chart_record.get("pivot_flow_index") == chart.pivot_index
        and isinstance(branch_signature, str)
        and 128.0 * torch.finfo(torch.float64).eps < mix
        and mix < 1.0 - 128.0 * torch.finfo(torch.float64).eps
        and math.isfinite(q_value) and math.isfinite(chart_residual)
        and chart_residual <= 128.0 * torch.finfo(torch.float64).eps
        and math.isfinite(normal_minus) and normal_minus < -budget_minus
        and math.isfinite(normal_plus) and normal_plus > budget_plus
        and math.isfinite(tangent_minus) and tangent_minus < stationarity_tolerance
        and math.isfinite(tangent_plus) and tangent_plus < stationarity_tolerance
        and math.isfinite(full_minus) and math.isfinite(full_plus)
        and curvature_ok)


def _gate_pending(output: Path, reason: str) -> dict[str, Any]:
    result = {"status": "gate_pending_base", "reason": reason,
        "physical_calls": 0, "objective_evaluations": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _write_status(output: Path, status: str, reason: str, budget: dict[str, Any]) -> dict[str, Any]:
    result = {"status": status, "reason": reason, "selection_use": False,
        "objective_evaluations_cumulative": int(budget["objective_evaluations"]),
        "objective_evaluation_limit": int(budget["max_objective_evaluations"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _require_budget(budget: dict[str, Any]) -> None:
    required = {"global_started_at", "total_seconds", "objective_evaluations",
        "max_objective_evaluations"}
    if not required.issubset(budget):
        raise ValueError(f"shared cumulative budget missing {sorted(required - set(budget))}")
    if (not math.isfinite(float(budget["global_started_at"]))
            or float(budget["total_seconds"]) <= 0
            or int(budget["objective_evaluations"]) < 0
            or int(budget["max_objective_evaluations"]) < 1):
        raise ValueError("invalid shared cumulative budget")


def _validate_parent_plan(problem: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = REFERENCE_WORK / "PLAN.json"
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if (digest != REFERENCE_PLAN_SHA
            or (REFERENCE_WORK / "PLAN_SHA256.txt").read_text().strip() != digest):
        raise ValueError("frozen native-reference plan seal mismatch")
    parent_plan = json.loads(plan_path.read_text())
    continuation_path = CONTINUATION_WORK / "CONTINUATION_PLAN.json"
    continuation_digest = hashlib.sha256(continuation_path.read_bytes()).hexdigest()
    continuation_sha_file = CONTINUATION_WORK / "CONTINUATION_PLAN_SHA256.txt"
    if (continuation_digest != CONTINUATION_PLAN_SHA
            or continuation_sha_file.read_text().strip() != continuation_digest):
        raise ValueError("approved support-transition continuation plan seal mismatch")
    plan = json.loads(continuation_path.read_text())
    optimizer = parent_plan.get("optimizer", {})
    transition = plan.get("source_transition", {}).get("src/advar/transport.py")
    if (optimizer.get("method") != POWELL["method"]
            or optimizer.get("maxfev_per_start") != POWELL["maxfev_per_start"]
            or optimizer.get("maxiter_per_start") != POWELL["maxiter_per_start"]
            or optimizer.get("xtol") != POWELL["xtol"]
            or optimizer.get("ftol") != POWELL["ftol"]
            or parent_plan.get("fixed_problem_identity", {}).get("fixed_problem_sha256") != FIXED_PROBLEM_SHA
            or plan.get("parent_native_plan_sha256") != digest
            or transition != {"before": TRANSPORT_OLD_SHA, "after": TRANSPORT_CURRENT_SHA}
            or plan.get("source_files", {}).get("src/advar/transport.py") != TRANSPORT_CURRENT_SHA
            or plan.get("extra_source_files", {}).get(
                "graphify-out/fv-root-cause-20260919/native_reference_20261012/reference_search.py"
            ) != hashlib.sha256((REFERENCE_WORK / "reference_search.py").read_bytes()).hexdigest()
            or problem.identity.get("fixed_problem_sha256") != FIXED_PROBLEM_SHA):
        raise ValueError("frozen Powell policy, continuation source transition, or native problem identity mismatch")
    source_map = {**plan["source_files"], **plan["extra_source_files"]}
    for name, expected in source_map.items():
        path = ROOT / name
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(ROOT.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"continuation source pin mismatch: {name}")
    for name, expected in plan["archive_files"].items():
        path = ROOT / name
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(ROOT.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"continuation archive pin mismatch: {name}")
    bundle = CONTINUATION_WORK / "source.tar.gz"
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != plan.get("source_bundle_sha256"):
        raise ValueError("continuation source bundle hash mismatch")
    with tarfile.open(bundle, "r:gz") as archive:
        members: dict[str, bytes] = {}
        for item in archive.getmembers():
            if item.isfile():
                stream = archive.extractfile(item)
                if stream is None:
                    raise ValueError(f"unreadable continuation source bundle member: {item.name}")
                members[item.name] = stream.read()
    if set(members) != set(source_map):
        raise ValueError("continuation source bundle member set mismatch")
    if any(hashlib.sha256(members[name]).hexdigest() != expected
            for name, expected in source_map.items()):
        raise ValueError("continuation source bundle member hash mismatch")
    if (plan.get("runtime_identity", {}).get("python") != platform.python_version()
            or plan.get("runtime_identity", {}).get("torch") != torch.__version__
            or plan.get("runtime_identity", {}).get("device") != "CPU FP64"):
        raise ValueError("runtime differs from the approved continuation plan")
    return parent_plan, plan


def _adjoint_response(problem: Any, control: Tensor, parameters: Tensor,
        parameter_index: int, budget: dict[str, Any], budget_check: Callable[[], None]
        ) -> dict[str, float]:
    """Compute E_p - lambda.T F_zp with the exact native objective Hessian."""
    z = control.detach().clone().requires_grad_(True)
    p = parameters.detach().clone().requires_grad_(True)

    def objective(value: Tensor, parameter: Tensor) -> Tensor:
        budget_check()
        result = problem.objective(value, parameter)
        budget["objective_evaluations"] = int(budget["objective_evaluations"]) + 1
        return result

    def objective_z(value: Tensor) -> Tensor:
        return objective(value, p)

    def objective_zp(value: Tensor, parameter: Tensor) -> Tensor:
        return objective(value, parameter)

    gradient_z = torch.autograd.grad(objective_z(z), z, create_graph=True)[0]
    fz = torch.autograd.functional.jacobian(
        lambda value: torch.autograd.grad(
            objective(value, p), value, create_graph=True)[0],
        z, create_graph=False, strict=True, vectorize=False)
    fzp = torch.autograd.functional.jacobian(
        lambda parameter: torch.autograd.grad(
            objective_zp(z, parameter), z, create_graph=True)[0],
        p, create_graph=False, strict=True, vectorize=False)
    budget_check()
    score = problem.score(z, p)
    ez = torch.autograd.grad(score, z, create_graph=True)[0]
    budget_check()
    ep = torch.autograd.grad(problem.score(z, p), p, create_graph=False)[0]
    # F_z is symmetric for this scalar C2 objective; solve the stated transpose system.
    lam = torch.linalg.solve(fz.T, ez)
    prediction = ep[parameter_index] - torch.dot(lam, fzp[:, parameter_index])
    stationarity = torch.linalg.vector_norm(gradient_z)
    if not bool(torch.isfinite(prediction)):
        raise ValueError("exact adjoint response is nonfinite")
    return {"E_p": float(ep[parameter_index]),
        "lambda_F_zp": float(torch.dot(lam, fzp[:, parameter_index])),
        "predicted_dE_dp": float(prediction),
        "stationarity_norm": float(stationarity),
        "linear_system_residual_norm": float(torch.linalg.vector_norm(fz.T @ lam - ez))}


def _parameter_step(problem: Any, parameters: Tensor, index: int) -> tuple[float, str]:
    if index == 0:
        standard_deviation = problem.observation_std_dbz.reshape(-1)[0]
        step = float(standard_deviation) / 100.0
        return step, "first observed dBZ; h=its fixed noise standard deviation/100"
    if index == 12:
        return 1.0e-3, "background-pattern coefficient; h=1e-3 coefficient units"
    raise ValueError("unsupported native parameter index")


def run_reoptimized_response(problem: Any, base_control: Tensor,
        base_parameters: Tensor, qualification: dict[str, Any], *,
        qualify: Callable[[Tensor, Tensor], dict[str, Any]],
        budget: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run two four-point response grids, only after base and point qualification.

    ``budget`` is the same mutable cumulative budget used by the parent reference
    search. Powell's frozen per-start limits are upper bounds; global limits are
    enforced by the shared counter and deadline, never reset here.
    """
    if (base_control.shape != (26,) or base_parameters.shape != (13,)
            or base_control.dtype != torch.float64 or base_parameters.dtype != torch.float64
            or base_control.device.type != "cpu" or base_parameters.device.type != "cpu"):
        raise ValueError("native reference dimensions must be CPU FP64 controls[26], parameters[13]")
    qualified_base = _qualified_control(qualification, base_control)
    if not _qualified(qualification, qualified_base, base_parameters, candidate_control=base_control):
        return _gate_pending(output, "supplied base has not passed the regular local-minimum qualification")
    _require_budget(budget)

    def time_check() -> None:
        if time.monotonic() - float(budget["global_started_at"]) >= float(budget["total_seconds"]):
            raise TimeoutError("shared total reference budget reached")

    def budget_check() -> None:
        time_check()
        if int(budget["objective_evaluations"]) >= int(budget["max_objective_evaluations"]):
            raise TimeoutError("shared cumulative objective-evaluation budget reached")

    if int(budget["objective_evaluations"]) >= int(budget["max_objective_evaluations"]):
        return _write_status(output, "budget_exhausted_no_response",
            "the shared native-objective evaluation cap is already closed; no new phase grant is created", budget)
    _parent_plan, plan = _validate_parent_plan(problem)
    time_check()
    independent = qualify(qualified_base, base_parameters)
    time_check()
    base_control = _qualified_control(independent, qualified_base)
    if not _qualified(independent, base_control, base_parameters,
            candidate_control=qualified_base):
        return _gate_pending(output, "fresh base qualification did not confirm the supplied eligible base")
    import scipy
    from scipy.optimize import minimize
    if scipy.__version__ != SCIPY_VERSION:
        raise ValueError(f"SciPy {SCIPY_VERSION} required by the shared Powell runtime, got {scipy.__version__}")

    base_signature = _branch_signature(independent)
    pattern = problem.background_pattern
    pattern_norm = float(torch.linalg.vector_norm(pattern))
    pattern_max = float(pattern.abs().max())
    if not math.isfinite(pattern_norm) or pattern_norm == 0:
        raise ValueError("background pattern must have finite nonzero norm")
    rows: list[dict[str, Any]] = []
    all_started = time.monotonic()

    for index, label, step_note in PARAMETERS:
        h, physical_step = _parameter_step(problem, base_parameters, index)
        if not math.isfinite(h) or h <= 0:
            raise ValueError(f"invalid perturbation size for parameter {index}")
        adjoint = _adjoint_response(problem, base_control, base_parameters, index,
            budget, budget_check)
        grid: list[dict[str, Any]] = []
        for scale in (1.0, -1.0, 0.5, -0.5):
            budget_check()
            delta = scale * h
            parameters = base_parameters.clone()
            parameters[index] += delta
            budget_check()
            start_value = float(problem.objective(base_control, parameters).detach())
            budget["objective_evaluations"] = int(budget["objective_evaluations"]) + 1
            radius = math.sqrt(2.0 * start_value)
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("identity-prior Powell box requires positive finite start objective")
            calls = 0

            def objective(vector: Any) -> float:
                nonlocal calls
                budget_check()
                candidate = torch.as_tensor(vector, dtype=torch.float64, device="cpu")
                with torch.no_grad():
                    value = problem.objective(candidate, parameters)
                if value.ndim != 0 or not bool(torch.isfinite(value)):
                    raise ValueError("native objective must return a finite scalar")
                calls += 1
                budget["objective_evaluations"] = int(budget["objective_evaluations"]) + 1
                return float(value)

            remaining = int(budget["max_objective_evaluations"]) - int(budget["objective_evaluations"])
            if remaining < 1:
                raise TimeoutError("shared cumulative objective-evaluation budget exhausted")
            result = minimize(objective, base_control.detach().numpy().copy(), method="Powell",
                bounds=[(-radius, radius)] * 26,
                options={"maxfev": min(1000, remaining),
                    "maxiter": 100, "xtol": 1e-11, "ftol": 1e-13, "disp": False})
            optimizer_candidate = torch.as_tensor(result.x, dtype=torch.float64, device="cpu")
            q = qualify(optimizer_candidate, parameters)
            time_check()
            candidate = _qualified_control(q, optimizer_candidate)
            if (not _qualified(q, candidate, parameters, candidate_control=optimizer_candidate)
                    or _branch_signature(q) != base_signature):
                raise ValueError(f"perturbed Powell point {index}:{scale:g} failed stationarity/branch qualification")
            native_j = q.get("qualified_native_J", q.get("native_J"))
            if native_j is None or not math.isfinite(float(native_j)):
                raise ValueError("qualified callback must provide native_J at its returned control")
            with torch.no_grad():
                score = problem.score(candidate, parameters)
            time_check()
            if score.ndim != 0 or not bool(torch.isfinite(score)):
                raise ValueError("terminal score must be a finite scalar")
            raw_gradient = q.get("native_gradient_norm", q.get("gradient_inf"))
            if raw_gradient is None:
                raise ValueError("qualified callback is missing its native gradient norm")
            grid.append({"scale": scale, "parameter_delta": delta,
                "optimizer_control_sha256": _tensor_sha(optimizer_candidate),
                "control_sha256": _tensor_sha(candidate), "parameters_sha256": _tensor_sha(parameters),
                "objective": float(native_j), "score": float(score),
                "stationarity_norm": float(raw_gradient),
                "branch_signature": _branch_signature(q), "powell_success": bool(result.success),
                "powell_message": str(result.message), "objective_calls": calls,
                "iterations": int(result.nit)})
        by_scale = {float(item["scale"]): item for item in grid}
        d_h = (float(by_scale[1.0]["score"]) - float(by_scale[-1.0]["score"])) / (2.0 * h)
        d_half = (float(by_scale[0.5]["score"]) - float(by_scale[-0.5]["score"])) / h
        rows.append({"parameter_index": index, "parameter": label,
            "step_rationale": step_note, "h": h, "physical_perturbation": physical_step,
            "observations": len(grid), "reoptimized_dE_dp_h": d_h,
            "reoptimized_dE_dp_h_over_2": d_half,
            "central_difference_refinement_change": d_half - d_h,
            "exact_adjoint": adjoint, "powell": grid})

    result = {"status": "completed", "base_control_sha256": _tensor_sha(base_control),
        "base_parameters_sha256": _tensor_sha(base_parameters),
        "base_branch_signature": base_signature, "selection_use": False,
        "reoptimized_points": 8, "terminal_score_evaluations": 8,
        "objective_evaluations_cumulative": int(budget["objective_evaluations"]),
        "elapsed_seconds": time.monotonic() - all_started,
        "background_pattern_rms": float(torch.sqrt(torch.mean(pattern.square()))),
        "background_pattern_max_abs": pattern_max,
        "background_delta_max_abs_at_h": 1.0e-3 * pattern_max,
        "optimizer": POWELL, "scipy_version": SCIPY_VERSION, "parameters": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result
