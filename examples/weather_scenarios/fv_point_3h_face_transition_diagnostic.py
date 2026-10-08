"""Bounded two-sided donor-trace diagnostic at a selected FV face transition."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport
from examples.weather_scenarios import fv_point_3h_model_guided_continuation as model
from examples.weather_scenarios import fv_point_3h_qy32_diagnostic as qy
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as guard_policy
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FACE_TRANSITION_DIAGNOSTIC_PLAN_20261008.json"
MODEL_PLAN = EVIDENCE / "MODEL_GUIDED_FULL_SPACE_PLAN_20261008.json"
MODEL_PLAN_SHA = "9fc5e046ca484737ebd040469990f10cc918fcd9dc245d0762e65f297151440c"
MODEL_STEP = EVIDENCE / "model_guided_resume_20261008_attempt1/step.json"
MODEL_RUN = MODEL_STEP.with_suffix(".run.json")
MODEL_RESOURCE = MODEL_STEP.with_suffix(".resource.json")
MODEL_STEP_SHA = "66aa57c3056220a7057a5b3c3cca2a09c06d44ed34279b055174e6930ab1838e"
MODEL_RUN_SHA = "604d447997c76f358fda4c8f0c895c563c1f2772171de511cf1015d19f2cf6bd"
MODEL_RESOURCE_SHA = "67b7294e85b19022b205be03f8cb2f4198be5cd8d50c21935d92354d682344e5"
CONTROL_SHA = "2cdccade81a8cf218dd7cad97adf71f6675f8445481dd8f6394ec337ea8e8007"
PARAMETERS_SHA = qy.PARAMETERS_SHA
SELF = "examples/weather_scenarios/fv_point_3h_face_transition_diagnostic.py"
TEST = "tests/test_fv_point_3h_face_transition_diagnostic.py"
ETAS = (-2e-6, -1e-6, 0.0, 1e-6, 2e-6)
NONZERO_ETAS = (-2e-6, -1e-6, 1e-6, 2e-6)
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 240.0, 300.0, 1024**3
DEFAULT_FACE = {"axis": "y", "row": 4, "column": 3}


def policy() -> dict[str, Any]:
    return {"etas": list(ETAS),
            "internal_seconds": INTERNAL_SECONDS, "outer_seconds": WALL_SECONDS,
            "rss_bytes": RSS_BYTES, "guarded_launches": 1,
            "hvp_calls": 0, "pcg_solves": 0, "optimizer_steps": 0,
            "score_computed": False, "response_computed": False}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _face_weights(problem: Any, axis: str, row: int, column: int) -> Tensor:
    spec = problem.frozen.fv_transport
    if axis not in {"x", "y"} or spec is None:
        raise ValueError("face requires x/y axis and the fixed FV transport profile")
    basis, limits = spec.psi_basis, spec.coefficient_limits
    if basis.ndim != 3 or basis.shape[0] != 5 or limits.shape != (5,):
        raise ValueError("face diagnostic requires the five-coefficient frozen streamfunction")
    height, width = basis.shape[1] - 1, basis.shape[2] - 1
    if ((axis == "x" and not (0 <= row < height and 0 <= column <= width))
            or (axis == "y" and not (0 <= row <= height and 0 <= column < width))):
        raise ValueError("requested face index is outside the FV face grid")
    if axis == "x":
        delta = basis[:, row + 1, column] - basis[:, row, column]
    else:
        delta = -(basis[:, row, column + 1] - basis[:, row, column])
    weights = delta * limits
    if not bool(torch.isfinite(weights).all()) or not bool(torch.any(weights != 0)):
        raise ValueError("requested face has no finite streamfunction sensitivity")
    return weights


def _face_value(control: Tensor, weights: Tensor) -> Tensor:
    if control.shape != (26,) or weights.shape != (5,) or control.dtype != torch.float64:
        raise ValueError("face value requires full FP64 control and five weights")
    return torch.dot(weights, torch.tanh(control[20:25]))


def _face_normal(control: Tensor, weights: Tensor) -> Tensor:
    normal = torch.zeros_like(control)
    normal[20:25] = weights * (1.0 - torch.tanh(control[20:25]).square())
    return normal


def _pivot(control: Tensor, weights: Tensor) -> int:
    normal = _face_normal(control, weights)[20:25]
    pivot = int(torch.argmax(normal.abs()))
    scale = float(normal.abs().max())
    if not math.isfinite(scale) or scale <= 128 * torch.finfo(control.dtype).eps:
        raise ValueError("face chart has no resolved pivot derivative")
    return 20 + pivot


def _chart(control: Tensor, weights: Tensor, pivot: int, eta: float) -> Tensor:
    if (control.shape != (26,) or weights.shape != (5,) or pivot not in range(20, 25)
            or not math.isfinite(eta)):
        raise ValueError("face chart inputs are malformed")
    result = control.clone()
    pflow = pivot - 20
    retained = [index for index in range(5) if index != pflow]
    numerator = eta - torch.dot(weights[retained], torch.tanh(control[20:25][retained]))
    argument = numerator / weights[pflow]
    if not bool(torch.isfinite(argument)) or not bool(argument.abs() < 1):
        raise ValueError("requested face coordinate lies outside the real pivot chart")
    result[pivot] = torch.atanh(argument)
    return result


def _chart_jacobian(control: Tensor, weights: Tensor, pivot: int,
                    eta: float) -> tuple[Tensor, Tensor, Tensor]:
    point = _chart(control, weights, pivot, eta)
    retained = [index for index in range(26) if index != pivot]
    z = torch.zeros((26, 25), dtype=control.dtype, device=control.device)
    z[retained, torch.arange(25)] = 1
    normal = _face_normal(point, weights)
    for col, coordinate in enumerate(retained):
        if 20 <= coordinate < 25:
            z[pivot, col] = -normal[coordinate] / normal[pivot]
    gamma_eta = torch.zeros_like(control)
    gamma_eta[pivot] = 1.0 / normal[pivot]
    return point, z, gamma_eta


def _geometry(control: Tensor, weights: Tensor, pivot: int,
              eta: float, gradient: Tensor) -> dict[str, Any]:
    point, z, gamma_eta = _chart_jacobian(control, weights, pivot, eta)
    normal = _face_normal(point, weights)
    q, r = torch.linalg.qr(z, mode="reduced")
    tangent = q @ (q.T @ gradient)
    covector = z.T @ gradient
    normal_component = float((normal @ gradient) / torch.linalg.vector_norm(normal))
    direct_tangent = gradient - normal * (normal @ gradient) / (normal @ normal)
    return {"tangent_covector": covector.tolist(),
            "tangent_gradient_norm": float(torch.linalg.vector_norm(tangent)),
            "tangent_gradient_norm_projection": float(torch.linalg.vector_norm(direct_tangent)),
            "normal_gradient_magnitude": abs(normal_component),
            "unit_normal_gradient": float((normal @ gradient) / torch.linalg.vector_norm(normal)),
            "intrinsic_normal_slope": float((normal @ gradient) / (normal @ normal)),
            "eta_transverse_slope": float(gamma_eta @ gradient),
            "normal_dot_chart_eta": float(normal @ gamma_eta),
            "normal_dot_chart_tangent_max": float((normal @ z).abs().max()),
            "chart_metric_condition": float(torch.linalg.cond(z.T @ z)),
            "chart_qr_diagonal": torch.diagonal(r).abs().tolist()}


def _donors(q: Tensor, qx: Tensor, qy: Tensor, sx: Tensor, sy: Tensor,
            raw_edges: tuple[Tensor, Tensor, Tensor, Tensor],
            effective_edges: tuple[Tensor, Tensor, Tensor, Tensor],
            axis: str, row: int, column: int) -> dict[str, Any]:
    left, right, bottom, top = effective_edges
    raw_left, raw_right, raw_bottom, raw_top = raw_edges
    if axis == "x":
        flux = qx[row, column]
        minus = left[row] if column == 0 else q[row, column - 1] + 0.5 * sx[row, column - 1]
        plus = right[row] if column == q.shape[1] else q[row, column] - 0.5 * sx[row, column]
        external = column in {0, q.shape[1]}
        raw_boundary = raw_left[row] if column == 0 else raw_right[row] if column == q.shape[1] else None
    elif axis == "y":
        flux = qy[row, column]
        minus = bottom[column] if row == 0 else q[row - 1, column] + 0.5 * sy[row - 1, column]
        plus = top[column] if row == q.shape[0] else q[row, column] - 0.5 * sy[row, column]
        external = row in {0, q.shape[0]}
        raw_boundary = raw_bottom[column] if row == 0 else raw_top[column] if row == q.shape[0] else None
    else:
        raise ValueError("face axis must be x or y")
    selected = minus if bool(flux > 0) else plus if bool(flux < 0) else None
    return {"flux": float(flux), "minus_donor": float(minus), "plus_donor": float(plus),
            "selected_donor": None if selected is None else float(selected),
            "selected_side": "minus" if bool(flux > 0) else "plus" if bool(flux < 0) else "zero_flux",
            "external_face": external,
            "raw_boundary_edge": None if raw_boundary is None else float(raw_boundary)}


def _effective_edges(raw: tuple[Tensor, Tensor, Tensor, Tensor],
                     stage: int, growth_log: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if stage % 2 == 1 or not bool(growth_log > 0):
        return raw
    return (transport._scale_by_growth(raw[0], growth_log),
            transport._scale_by_growth(raw[1], growth_log),
            transport._scale_by_growth(raw[2], growth_log),
            transport._scale_by_growth(raw[3], growth_log))


def _stage_schedule(problem: Any, parameters: Tensor, growth_log: Tensor) -> tuple[Any, Any, int, int]:
    fv = problem.contract(parameters).fv_transport
    substeps = fv.substeps_per_interval
    analysis = fv.boundary_echo
    future = problem.future_boundary_echo
    analysis_count = 4 * substeps
    expected_future = 2 * problem.frozen.nowcast_config.forecast_steps * substeps
    if (len(analysis) != 2 * substeps or len(future) * 2 != expected_future
            or len(analysis) * 2 != analysis_count):
        raise ValueError("analysis/future boundary schedules differ from the traced RK stage profile")
    scale = torch.exp(growth_log)
    if not bool(torch.isfinite(scale)):
        raise ValueError("per-substep growth scale is nonfinite")
    if bool(growth_log > 0):
        limit = torch.finfo(growth_log.dtype).max / scale
        for schedule in (analysis, future):
            for stages in schedule:
                for edges in stages:
                    if bool((torch.cat(edges).abs() > limit).any()):
                        raise ValueError("a prescribed boundary edge is not scalable at positive growth")
    return analysis, future, analysis_count, expected_future


def _schedule_edges(analysis: Any, future: Any, analysis_count: int, stage: int,
                    growth_log: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor],
                                                  tuple[Tensor, Tensor, Tensor, Tensor], str]:
    segment = "analysis" if stage < analysis_count else "future"
    local = stage if stage < analysis_count else stage - analysis_count
    step_index, rk_stage = divmod(local, 2)
    schedule = analysis if segment == "analysis" else future
    raw = tuple(schedule[step_index][rk_stage])
    return raw, _effective_edges(raw, rk_stage, growth_log), segment


def _verify_face_value(problem: Any, control: Tensor, weights: Tensor,
                       face: dict[str, Any], eta: float) -> float:
    qx, qy_values = qy.production_fluxes(problem, control)
    actual = qx[face["row"], face["column"]] if face["axis"] == "x" else qy_values[face["row"], face["column"]]
    formula = _face_value(control, weights)
    scale = (weights * torch.tanh(control[20:25])).abs().sum()
    budget = 128 * torch.finfo(control.dtype).eps * max(
        float(scale), abs(float(actual)), abs(float(formula)), abs(eta), torch.finfo(control.dtype).tiny)
    if (not bool(torch.isfinite(actual) & torch.isfinite(formula))
            or abs(float(actual) - float(formula)) > budget
            or abs(float(actual) - eta) > budget):
        raise ValueError("chart eta differs from production face flux")
    return float(actual)


def _cost_only(problem: Any, control: Tensor, parameters: Tensor, weights: Tensor,
               face: dict[str, Any], deadline: float) -> dict[str, Any]:
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic budget expired before eta-zero cost")
    eta = _verify_face_value(problem, control, weights, face, 0.0)
    value = problem.objective(control, parameters)
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic budget expired after eta-zero cost")
    if value.shape != () or not bool(torch.isfinite(value)):
        raise ValueError("eta-zero original full-control J is nonfinite")
    return {"requested_eta": 0.0, "realized_eta": eta,
        "objective": float(value), "gradient_computed": False,
        "branch_checked": False, "control": control.tolist(),
        "control_sha256": _tensor_sha(control)}


def _measure(problem: Any, control: Tensor, parameters: Tensor, weights: Tensor,
             pivot: int, eta: float, deadline: float, face: dict[str, Any],
             margin_collector: Any,
             stage_schedules: tuple[Any, Any, int, int]) -> dict[str, Any]:
    realized_eta = _verify_face_value(problem, control, weights, face, eta)
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired before objective")
    objective = problem.objective(control, parameters)
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired after objective")
    if not bool(torch.isfinite(objective)):
        raise ValueError("original full-control J is nonfinite")
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired before gradient")
    gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    phi = torch.dot(gradient, gradient) / 2
    if (gradient.shape != (26,) or not bool(torch.isfinite(gradient).all())
            or not bool(torch.isfinite(phi))):
        raise ValueError("original full-control gradient or Phi is nonfinite")
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired after gradient")
    spec = problem.frozen.fv_transport
    cfg = problem.frozen.nowcast_config
    growth = cfg.max_log_growth_per_step * torch.tanh(control[25]) / spec.substeps_per_interval
    analysis_schedule, future_schedule, analysis_count, future_count = stage_schedules
    traces: list[dict[str, Any]] = []
    signature = None
    scope = None
    try:
        def observe(q: Tensor, qx: Tensor, qy_values: Tensor) -> None:
            stage = margin_collector.stages
            margin_collector(q, qx, qy_values)
            sx, sy = transport._muscl_slopes(q)
            raw_edges, effective_edges, segment = _schedule_edges(
                analysis_schedule, future_schedule, analysis_count, stage, growth)
            donor = _donors(q, qx, qy_values, sx, sy, raw_edges, effective_edges,
                            face["axis"], face["row"], face["column"])
            if stage < analysis_count:
                if donor["external_face"]:
                    raw_value = donor["raw_boundary_edge"]
                    if face["axis"] == "x" and face["column"] == 0:
                        effective_value = float(effective_edges[0][face["row"]])
                    elif face["axis"] == "x":
                        effective_value = float(effective_edges[1][face["row"]])
                    elif face["row"] == 0:
                        effective_value = float(effective_edges[2][face["column"]])
                    else:
                        effective_value = float(effective_edges[3][face["column"]])
                    donor.update(raw_boundary_edge=raw_value, effective_boundary_edge=effective_value,
                                 boundary_growth_gap=(None if raw_value is None
                                                      else effective_value - raw_value))
                traces.append({"stage": stage, "segment": segment,
                    "growth_log_per_step": float(growth), "face": donor,
                    "raw_edges": [edge.tolist() for edge in raw_edges],
                    "effective_edges": [edge.tolist() for edge in effective_edges]})
        with torch.no_grad(), transport.observe_minmod_stages(observe):
            signature, scope = problem.branch_check(control, parameters)
    except ValueError as error:
        margins = margin_collector.report()
        return {"requested_eta": eta, "realized_eta": realized_eta,
            "objective": float(objective), "gradient": gradient.tolist(),
            "phi": float(phi), "branch": {"status": "branch_refused", "reason": str(error)},
            "branch_margins": margins, "donor_trace": traces, "geometry": _geometry(
                control, weights, pivot, eta, gradient),
            "observer_stage_count": margin_collector.stages,
            "branch_signature": None}
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic time budget expired after strict branch/donor capture")
    margins = margin_collector.report()
    expected_stages = problem.layout["euler_stages"]
    analysis_stages = 2 * (len(problem.layout["observation_times_seconds"]) - 1) * spec.substeps_per_interval
    if analysis_count != analysis_stages or analysis_stages + future_count != expected_stages:
        raise ValueError("boundary schedule partition differs from the full branch signature stages")
    observed_stages = margin_collector.stages
    passed = (signature.get("euler_stages") == expected_stages == observed_stages
        and len(traces) == analysis_count
        and len(signature.get("choices", [])) == expected_stages
        and len(signature.get("face_signs", [])) == expected_stages
        and seed._valid_margins(margins, complete=True))
    branch = {"status": "passed_strict_branch" if passed else "branch_or_margin_refused",
        "euler_stages": signature.get("euler_stages"),
        "choice_stage_count": len(signature.get("choices", [])),
        "face_sign_stage_count": len(signature.get("face_signs", [])),
        "signature_sha256": qy._digest({"choices": signature.get("choices"),
                                         "face_signs": signature.get("face_signs")})}
    return {"requested_eta": eta, "realized_eta": float(_face_value(control, weights)),
        "control": control.tolist(), "control_sha256": _tensor_sha(control),
        "objective": float(objective), "gradient": gradient.tolist(),
        "gradient_sha256": _tensor_sha(gradient), "gradient_inf": float(gradient.abs().max()),
        "gradient_norm": float(torch.linalg.vector_norm(gradient)), "phi": float(phi),
        "geometry": _geometry(control, weights, pivot, eta, gradient), "branch": branch,
        "branch_scope": scope, "branch_margins": margins,
        "branch_partition": (qy._partition(signature, analysis_stages) if passed else None),
        "branch_signature": {"choices": signature.get("choices"),
                             "face_signs": signature.get("face_signs")},
        "donor_trace": traces, "analysis_stage_count": analysis_stages,
        "future_stage_count": expected_stages - analysis_stages,
        "observer_stage_count": observed_stages}


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    path = path.resolve()
    if (path != PLAN.resolve() or path.is_symlink()
            or not path.resolve().is_relative_to(ROOT.resolve()) or _sha(path) != digest):
        raise ValueError("face-transition plan identity mismatch")
    plan = json.loads(path.read_text())
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if (plan.get("experiment_kind") != "face_transition_diagnostic"
            or plan.get("producing_plan") != MODEL_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != MODEL_PLAN_SHA
            or plan.get("policy") != policy() or not isinstance(plan.get("face"), dict)
            or plan["face"].get("axis") not in {"x", "y"}
            or not isinstance(sources, dict) or not isinstance(archives, dict)):
        raise ValueError("face-transition scope, face, policy, or plan pin maps changed")
    producer = model._load_plan(MODEL_PLAN, MODEL_PLAN_SHA)
    required_sources = set(producer["source_files"]) | {SELF, TEST}
    required_archives = set(producer["archive_files"]) | {
        MODEL_PLAN.relative_to(ROOT).as_posix(), MODEL_STEP.relative_to(ROOT).as_posix(),
        MODEL_RUN.relative_to(ROOT).as_posix(), MODEL_RESOURCE.relative_to(ROOT).as_posix()}
    if (set(sources) != required_sources or set(archives) != required_archives
            or any(sources[name] != value for name, value in producer["source_files"].items())
            or any(archives[name] != value for name, value in producer["archive_files"].items())
            or sources.get(SELF) != _sha(ROOT / SELF)
            or sources.get(TEST) != _sha(ROOT / TEST)):
        raise ValueError("face-transition plan changed model lineage or omitted its focused producer/tests")
    required = {MODEL_PLAN.relative_to(ROOT).as_posix(): MODEL_PLAN_SHA,
        MODEL_STEP.relative_to(ROOT).as_posix(): MODEL_STEP_SHA,
        MODEL_RUN.relative_to(ROOT).as_posix(): MODEL_RUN_SHA,
        MODEL_RESOURCE.relative_to(ROOT).as_posix(): MODEL_RESOURCE_SHA}
    if any(archives.get(name) != value for name, value in required.items()):
        raise ValueError("face-transition plan omits the current model endpoint receipt set")
    for name, value in {**sources, **archives}.items():
        model.tangent._pinned_path(name, value)
    return plan


def _validate_model_source(plan: dict[str, Any]) -> dict[str, Any]:
    if _sha(MODEL_PLAN) != MODEL_PLAN_SHA:
        raise ValueError("frozen model-guided resume plan changed")
    model_plan = model._load_plan(MODEL_PLAN, MODEL_PLAN_SHA)
    if not {**model_plan["source_files"], **model_plan["archive_files"]}:
        raise ValueError("model-guided plan has no source pins")
    for path, digest in ((MODEL_STEP, MODEL_STEP_SHA), (MODEL_RUN, MODEL_RUN_SHA),
                         (MODEL_RESOURCE, MODEL_RESOURCE_SHA)):
        if (plan["archive_files"].get(path.relative_to(ROOT).as_posix()) != digest
                or _sha(path) != digest):
            raise ValueError("face diagnostic plan omits the current model endpoint receipts")
    raw = json.loads(MODEL_STEP.read_text())
    parent = json.loads(MODEL_RUN.read_text())
    resource = json.loads(MODEL_RESOURCE.read_text())
    steps = raw.get("iterations", [])
    if (raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("optimizer_steps_applied") != 3
            or raw.get("current_control_sha256") != CONTROL_SHA
            or raw.get("numerical_status") != "max_iterations_completed"
            or raw.get("candidate_committed") is not True
            or len(steps) != 3 or any(item.get("committed") is not True for item in steps)
            or steps[-1].get("accepted_control_sha256") != CONTROL_SHA
            or steps[-1].get("accepted_control") != raw.get("current_control")
            or steps[-1].get("accepted_gradient") != raw.get("current_state", {}).get("gradient")
            or steps[-1].get("accepted_phi") != raw.get("current_state", {}).get("phi")
            or steps[-1].get("accepted_objective") != raw.get("current_state", {}).get("objective")
            or steps[-1].get("trials", [])[-1].get("branch", {}).get("signature_sha256")
            != raw.get("current_state", {}).get("branch", {}).get("signature_sha256")
            or not model.tangent._check_fixed_input(raw.get("input_after", {}),
                raw.get("input_before", {}), CONTROL_SHA)
            or raw.get("source_unchanged") is not True
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime") != raw.get("runtime_after")
            or parent.get("execution_status") != "completed"
            or parent.get("child_sha256") != MODEL_STEP_SHA
            or parent.get("numerical_status") != raw.get("numerical_status")
            or guard_policy.execution_status(resource) != "completed"):
        raise ValueError("current PR260 model-guided endpoint/receipts are not closed")
    expected_sources = {**model_plan["source_files"], **model_plan["archive_files"],
        MODEL_PLAN.relative_to(ROOT).as_posix(): MODEL_PLAN_SHA}
    if raw.get("source_before") != expected_sources:
        raise ValueError("current model run source receipt differs from its producer plan")
    return raw


def _gradient_pair(minus: Tensor, plus: Tensor, normal: Tensor) -> dict[str, Any]:
    delta = plus - minus
    delta_squared = float(delta @ delta)
    theta = (0.0 if delta_squared == 0 else
        min(1.0, max(0.0, -float(minus @ delta) / delta_squared)))
    gradient = minus + theta * delta
    direction = -gradient
    coefficient = float((normal @ delta) / (normal @ normal))
    tangent_jump = delta - coefficient * normal
    return {"minus_gradient": minus.tolist(), "plus_gradient": plus.tolist(),
        "gradient_jump": delta.tolist(),
        "gradient_jump_l2": float(torch.linalg.vector_norm(delta)),
        "zero_chart_normal_coefficient": coefficient,
        "zero_chart_tangent_jump": tangent_jump.tolist(),
        "zero_chart_tangent_jump_l2": float(torch.linalg.vector_norm(tangent_jump)),
        "phi_jump_identity": float(0.5 * (delta @ (plus + minus))),
        "minimum_norm_convex_hull": {"theta": theta, "gradient": gradient.tolist(),
            "direction": direction.tolist(),
            "gradient_norm": float(torch.linalg.vector_norm(gradient)),
            "minus_directional_pairing": float(minus @ direction),
            "plus_directional_pairing": float(plus @ direction),
            "scope": "finite two-gradient segment; diagnostic only, no step/root claim"}}


def _gradient_jump(control: Tensor, weights: Tensor, pivot: int,
                   samples: dict[float, dict[str, Any]]) -> dict[str, Any]:

    gm = torch.tensor(samples[-1e-6]["gradient"], dtype=torch.float64)
    gp = torch.tensor(samples[1e-6]["gradient"], dtype=torch.float64)
    gm_outer = torch.tensor(samples[-2e-6]["gradient"], dtype=torch.float64)
    gp_outer = torch.tensor(samples[2e-6]["gradient"], dtype=torch.float64)
    normal0 = _face_normal(_chart(control, weights, pivot, 0.0), weights)
    return {"normal_at_zero_face_chart": normal0.tolist(),
        "inner_pair": _gradient_pair(gm, gp, normal0),
        "outer_pair": _gradient_pair(gm_outer, gp_outer, normal0),
        "eta_zero_chart_geometry_using_finite_plus_inner_gradient": _geometry(
            control, weights, pivot, 0.0, gp),
        "zero_gradient_evaluated": False}


def _signature_differences(samples: dict[float, dict[str, Any]], analysis_stages: int) -> dict[str, Any]:
    pairs = {"same_negative_side": (-2e-6, -1e-6),
             "inner_cross_side": (-1e-6, 1e-6),
             "same_positive_side": (1e-6, 2e-6),
             "outer_cross_side": (-2e-6, 2e-6)}
    result = {}
    for label, (left_eta, right_eta) in pairs.items():
        left = samples[left_eta].get("branch_signature")
        right = samples[right_eta].get("branch_signature")
        if left is None or right is None:
            result[label] = None
            continue
        if any(len(left.get(key, [])) != len(right.get(key, [])) for key in ("choices", "face_signs")):
            result[label] = {"complete_signature_pair": False,
                "left_choice_stages": len(left.get("choices", [])),
                "right_choice_stages": len(right.get("choices", [])),
                "left_face_sign_stages": len(left.get("face_signs", [])),
                "right_face_sign_stages": len(right.get("face_signs", []))}
            continue
        result[label] = {}
        for key in ("choices", "face_signs"):
            a, b = left[key], right[key]
            changed = [index for index, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
            result[label][key] = {"changed_count": len(changed),
                "analysis_changed_count": sum(index < analysis_stages for index in changed),
                "future_changed_count": sum(index >= analysis_stages for index in changed),
                "changed_indices": changed}
    return result


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "setup_not_complete", "plan_sha256": plan_sha,
        "policy": policy(), "face": None, "eta_zero_cost_only": None,
        "samples": [], "optimizer_steps_applied": 0, "hvp_calls": 0, "pcg_solves": 0,
        "score_computed": False, "response_computed": False,
        "source_before": None, "source_after": None, "source_unchanged": None,
        "input_before": None, "input_after": None, "fixed_input_unchanged": None,
        "runtime": None, "runtime_after": None}
    guard_policy._write(output, record)
    source_names: set[str] = set()
    problem = original = control = parameters = truth = None
    identity: dict[str, Any] = {}

    def close() -> None:
        if source_names:
            after = {name: _sha(ROOT / name) for name in sorted(source_names)}
            record["source_after"] = after
            record["source_unchanged"] = record["source_before"] == after
        if all(value is not None for value in (problem, original, control, parameters, truth)):
            final_identity = seed._input_identity(cast(Any, problem), cast(Tensor, original),
                cast(Tensor, control), cast(Tensor, parameters), cast(Tensor, truth))
            record["input_after"] = final_identity
            record["fixed_input_unchanged"] = model.tangent._check_fixed_input(
                final_identity, identity, _tensor_sha(cast(Tensor, control)))
        record["runtime_after"] = guard_policy.blocks.runtime_identity()
        record["elapsed_seconds"] = time.monotonic() - started

    try:
        plan = _load_plan(plan_path, plan_sha)
        model_raw = _validate_model_source(plan)
        source_names = set(plan["source_files"]) | set(plan["archive_files"]) | {
            plan_path.resolve().relative_to(ROOT.resolve()).as_posix()}
        expected = {**plan["source_files"], **plan["archive_files"],
                    plan_path.resolve().relative_to(ROOT.resolve()).as_posix(): plan_sha}
        before = {name: _sha(ROOT / name) for name in sorted(source_names)}
        if before != expected:
            raise ValueError("face diagnostic prelaunch pins changed")
        record["source_before"] = before
        problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
        control = torch.tensor(model_raw["current_control"], dtype=torch.float64)
        if (_tensor_sha(control) != CONTROL_SHA or _tensor_sha(parameters) != PARAMETERS_SHA):
            raise ValueError("reconstructed current model control/parameter hashes changed")
        identity = seed._input_identity(problem, original, control, parameters, truth)
        if identity != model_raw["input_after"]:
            raise ValueError("reconstructed current model fixed input differs from PR260 receipt")
        runtime = guard_policy.blocks.runtime_identity()
        if runtime != model_raw["runtime"]:
            raise ValueError("face diagnostic runtime differs from current model endpoint")
        face = plan.get("face", DEFAULT_FACE)
        weights = _face_weights(problem, face["axis"], face["row"], face["column"])
        pivot = _pivot(control, weights)
        base_eta = float(qy.production_fluxes(problem, control)[0 if face["axis"] == "x" else 1]
                         [face["row"], face["column"]])
        if not math.isfinite(base_eta):
            raise ValueError("selected production face flux is nonfinite")
        base_eta = _verify_face_value(problem, control, weights, face, base_eta)
        record.update(face=face, pivot_control_index=pivot, face_weights=weights.tolist(),
            base_eta=base_eta, base_control_sha256=CONTROL_SHA, parameters_sha256=PARAMETERS_SHA,
            input_before=identity, runtime=runtime)
        spec = problem.frozen.fv_transport
        growth = (problem.frozen.nowcast_config.max_log_growth_per_step
                  * torch.tanh(control[25]) / spec.substeps_per_interval)
        stage_schedules = _stage_schedule(problem, parameters, growth)
        expected_euler = problem.layout["euler_stages"]
        if stage_schedules[2] + stage_schedules[3] != expected_euler:
            raise ValueError("validated boundary schedule does not cover the frozen FV stage count")
        zero_control = _chart(control, weights, pivot, 0.0)
        record["eta_zero_cost_only"] = _cost_only(
            problem, zero_control, parameters, weights, face, deadline)
        guard_policy._write(output, record)
        for eta in NONZERO_ETAS:
            if time.monotonic() >= deadline:
                raise TimeoutError("240-second face diagnostic budget expired")
            point = _chart(control, weights, pivot, eta)
            if time.monotonic() >= deadline:
                raise TimeoutError("240-second face diagnostic budget expired after chart reconstruction")
            collector = seed._MarginCollector()
            measured = _measure(problem, point, parameters, weights, pivot, eta, deadline,
                                face, collector, stage_schedules)
            record["samples"].append(measured)
            guard_policy._write(output, record)
        samples = {float(item["requested_eta"]): item for item in record["samples"]}
        signatures = _signature_differences(samples, 2 * (len(problem.layout["observation_times_seconds"])-1)
                                            * problem.frozen.fv_transport.substeps_per_interval)
        record["branch_side_differences"] = signatures
        record["gradient_jump"] = _gradient_jump(control, weights, pivot, samples)
        if time.monotonic() >= deadline:
            raise TimeoutError("240-second budget expired after face jump summaries")
        close()
        if (record["source_unchanged"] is not True or record["fixed_input_unchanged"] is not True
                or record["runtime_after"] != record["runtime"]):
            raise ValueError("face diagnostic final identity closure failed")
        if time.monotonic() >= deadline:
            raise TimeoutError("240-second budget expired after final source/input/runtime closure")
        all_strict = all(sample["branch"].get("status") == "passed_strict_branch"
                         for sample in record["samples"])
        record.update(phase="finished", execution_status="completed",
                      numerical_status="finite_two_sided_donor_trace_diagnostic"
                      if all_strict else "branch_refusal")
        guard_policy._write(output, record)
        return record
    except TimeoutError as error:
        close()
        closure = (record["source_unchanged"] is True
            and record["fixed_input_unchanged"] is True and record["runtime_after"] == record["runtime"])
        record.update(phase="finished", execution_status="completed" if closure else "failed",
            numerical_status="budget_refusal" if closure else "integrity_refusal", refusal=str(error))
        guard_policy._write(output, record)
        return record
    except Exception as error:
        close()
        record.update(phase="finished", execution_status="failed",
            numerical_status="diagnostic_error", refusal=f"{type(error).__name__}: {error}")
        guard_policy._write(output, record)
        raise


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    if plan_path != PLAN.resolve() or _sha(plan_path) != plan_sha:
        raise ValueError("caller face-transition plan identity mismatch")
    _load_plan(plan_path, plan_sha)
    parent_path = output.with_suffix(".run.json")
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, parent_path)):
        raise ValueError("face diagnostic report/resource/log paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
        "--plan", str(plan_path), "--plan-sha256", plan_sha, "--output", str(output)]
    resource_result = run_guarded_diagnostic(command, wall_seconds=WALL_SECONDS,
        rss_bytes=RSS_BYTES, report_path=resource, log_path=log)
    execution = guard_policy.execution_status(resource_result)
    parent = {"execution_status": execution, "resource": resource_result,
              "child_sha256": None, "child_read_error": None, "numerical_status": "not_reached"}
    guard_policy._write(parent_path, parent)
    try:
        child = json.loads(output.read_text())
        if not isinstance(child, dict):
            raise ValueError("face diagnostic child report must be a JSON object")
        parent.update(child_sha256=_sha(output), numerical_status=child.get("numerical_status"))
        closed = (child.get("phase") == "finished" and child.get("execution_status") == "completed"
            and child.get("source_unchanged") is True and child.get("fixed_input_unchanged") is True
            and child.get("runtime_after") == child.get("runtime")
            and child.get("plan_sha256") == plan_sha and child.get("base_control_sha256") == CONTROL_SHA)
        if execution == "completed" and not closed:
            parent.update(execution_status="failed", execution_failure_reason="child identity closure failed")
    except (OSError, ValueError) as error:
        parent.update(execution_status="failed" if execution == "completed" else execution,
            child_read_error=f"{type(error).__name__}: {error}")
        guard_policy._write(parent_path, parent)
        raise RuntimeError(f"face-transition report could not be verified: {parent_path}") from error
    guard_policy._write(parent_path, parent)
    if parent["execution_status"] == "failed":
        raise RuntimeError(f"face-transition diagnostic failed; receipts: {parent_path}, {output}")
    return {"parent": parent, "child": child}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "face_transition_attempt1/diagnostic.json")
    parser.add_argument("--resource", type=Path, default=EVIDENCE / "face_transition_attempt1/diagnostic.resource.json")
    parser.add_argument("--log", type=Path, default=EVIDENCE / "face_transition_attempt1/diagnostic.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
                             args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
