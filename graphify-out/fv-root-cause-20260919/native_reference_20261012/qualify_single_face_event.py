"""Conditional one-face event qualification for the actual y[4,3] Newton ray.

This is a one-candidate active-event analysis, not a generic nonsmooth solver.
It projects the stored dense-Newton ray onto the one observed Qy[4,3]=0 face,
uses the repository's face-flux chart and selected-face extension, takes
checkpointed exact chart-Newton corrections, then checks native-J tangential
stationarity, the two one-sided normal slopes, and exact reduced curvature on
both incident smooth pieces. It computes no score or full forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
QUALIFIER_PATH = HERE / "qualify_candidate.py"
_spec = importlib.util.spec_from_file_location("native_qualifier", QUALIFIER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("native qualifier source cannot be loaded")
qualifier = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = qualifier
_spec.loader.exec_module(qualifier)

from advar import matrix_free, transport  # noqa: E402
from examples.weather_scenarios.fv_face_flux_coordinates import (  # noqa: E402
    FVFaceFluxCoordinateChart, face_weights_from_basis,
)

CONTROL_SIZE, FACE, PIVOT_FACE = 26, ("y", 4, 3), ("y", 4, 3)
FLOW_START, FLOW_COUNT, FIELD_COUNT, GROWTH_COUNT = 20, 5, 20, 1
STATIONARITY_TOL = qualifier.STATIONARITY_TOL
ROUNDING_MULT = 128.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def _face_value(control: Tensor, spec: Any) -> Tensor:
    coefficients = spec.coefficient_limits * torch.tanh(control[FLOW_START:FLOW_START + FLOW_COUNT])
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    qx, qy = transport.face_volume_fluxes(psi)
    return qy[FACE[1], FACE[2]]


def _bracket_event(control: Tensor, direction: Tensor, spec: Any,
                   *, deadline: float) -> tuple[float, float, float]:
    value0 = float(_face_value(control, spec))
    high = 2.0**-24
    value_high = float(_face_value(control + high * direction, spec))
    attempts = 0
    while value0 * value_high > 0 and attempts < 32:
        if time.monotonic() >= deadline:
            raise TimeoutError("face event bracket budget exhausted")
        high *= 2.0
        value_high = float(_face_value(control + high * direction, spec))
        attempts += 1
    if value0 == 0 or value0 * value_high > 0:
        raise ValueError("dense Newton ray does not bracket the selected y[4,3] face")
    low, value_low = 0.0, value0
    for _ in range(80):
        if time.monotonic() >= deadline:
            raise TimeoutError("face event bisection budget exhausted")
        mid = 0.5 * (low + high)
        value_mid = float(_face_value(control + mid * direction, spec))
        if value_mid == 0.0:
            return mid, low, high
        if value_low * value_mid > 0:
            low, value_low = mid, value_mid
        else:
            high = mid
    return 0.5 * (low + high), low, high


def _newton_event_control(control: Tensor, step: Tensor, alpha: float,
                          spec: Any, pivot_index: int) -> tuple[Tensor, Tensor, FVFaceFluxCoordinateChart]:
    ray_control = control + alpha * step
    chart = FVFaceFluxCoordinateChart.from_basis(
        field_count=FIELD_COUNT, growth_count=GROWTH_COUNT,
        coefficient_limits=spec.coefficient_limits,
        psi_basis=spec.psi_basis, axis=FACE[0], face=(FACE[1], FACE[2]),
        pivot_index=pivot_index,
    )
    coordinates = chart.to_face_coordinates(ray_control)
    pivot = FIELD_COUNT + pivot_index
    coordinates[pivot] = coordinates.new_zeros(())
    event_control = chart.from_face_coordinates(coordinates)
    return event_control, coordinates, chart


def run(candidate_json: Path, hessian_json: Path, continuation_plan: Path,
        continuation_sha256: str, output: Path, *, remaining_seconds: float = 300.0,
        compute_reduced_curvature: bool = False) -> dict[str, Any]:
    if output.exists():
        raise ValueError("one-face event output must be fresh")
    if not math.isfinite(remaining_seconds) or not 0 < remaining_seconds <= 540:
        raise ValueError("remaining_seconds must be in (0,540]")
    started = time.monotonic()
    deadline = started + remaining_seconds
    _base_plan, source_before, transition = qualifier._execution_plan(
        continuation_plan_path=continuation_plan,
        expected_continuation_sha256=continuation_sha256,
    )
    candidate_digest, hessian_digest = _sha(candidate_json), _sha(hessian_json)
    candidate = json.loads(candidate_json.read_text())
    hraw = json.loads(hessian_json.read_text())
    control = torch.as_tensor(candidate["best_control"], dtype=torch.float64)
    case = qualifier.cycle.load_fixed_case()
    parameters = case["parameters"]
    problem, original, truth = case["problem"], case["original"], case["truth"]
    if (_tensor_sha(control) != hraw.get("control_sha256")
            or _tensor_sha(parameters) != hraw.get("parameters_sha256")
            or hraw.get("hvp_columns_completed") != CONTROL_SIZE
            or hraw.get("qualification_source_sha256") != _sha(QUALIFIER_PATH)
            or hraw.get("source_transition") != transition
            or hraw.get("full_hessian_positive_definite") is not True
            or hraw.get("branch_stable") is not True):
        raise ValueError("one-face event inputs differ from the full-H checkpoint")
    hessian = torch.as_tensor(hraw["hessian"], dtype=torch.float64)
    gradient = torch.as_tensor(hraw["native_gradient"], dtype=torch.float64)
    if (hessian.shape != (CONTROL_SIZE, CONTROL_SIZE)
            or _tensor_sha(hessian) != hraw.get("hessian_sha256")
            or gradient.shape != (CONTROL_SIZE,)):
        raise ValueError("full-H checkpoint matrix or gradient is malformed")
    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("pinned point profile has no FV transport")
    spec_before = _sha(ROOT / "src/advar/transport.py")
    if spec_before != "c9dde24c12237d5ffee6b8ca0766b4d5619abf6110f3e858ee447bf49c1f7da9":
        raise ValueError("support-only transport transition is not the approved c9dde24 source")

    # The face normal is obtained from the fixed linear streamfunction basis;
    # choose the largest representable coefficient derivative as chart pivot.
    weights = face_weights_from_basis(spec.psi_basis, FACE[0], (FACE[1], FACE[2]))
    fractions = torch.tanh(control[FLOW_START:FLOW_START + FLOW_COUNT])
    normal_weights = weights * spec.coefficient_limits * (1.0 - fractions.square())
    pivot_index = int(normal_weights.abs().argmax())
    if not bool(normal_weights[pivot_index].abs() > 0):
        raise ValueError("selected face has no finite control-space normal")
    newton_step = torch.linalg.solve(hessian, -gradient)
    if not bool(torch.isfinite(newton_step).all()) or not float(torch.dot(gradient, newton_step)) < 0:
        raise ValueError("stored exact H does not yield a finite descending Newton correction")
    alpha, alpha_low, alpha_high = _bracket_event(control, newton_step, spec, deadline=deadline)
    event_control, face_coordinates, chart = _newton_event_control(
        control, newton_step, alpha, spec, pivot_index,
    )
    actual_face_flux = float(_face_value(event_control, spec))
    all_flux = torch.cat(tuple(part.flatten() for part in transport.face_volume_fluxes(
        torch.einsum("k,kij->ij", spec.coefficient_limits * torch.tanh(
            event_control[FLOW_START:FLOW_START + FLOW_COUNT]), spec.psi_basis))))
    face_scale = float(all_flux.abs().max())
    chart_residual_ratio = abs(actual_face_flux) / max(face_scale, torch.finfo(event_control.dtype).tiny)
    if chart_residual_ratio > 128 * torch.finfo(event_control.dtype).eps:
        raise ValueError("face chart projection does not preserve the production qy[4,3]=0 within FP64 tolerance")

    event_branch = qualifier._analysis_branch(problem, event_control, parameters, deadline=deadline)
    expected_face = f"y:{FACE[1]}:{FACE[2]}"
    if (event_branch["classification"] != "singleface"
            or event_branch["unique_nearzero_face_entities"] != [expected_face]
            or event_branch["unique_near_tie_slope_entities"]):
        unsupported = {
            "status": "unsupported_event_not_singleface",
            "base_control_sha256": _tensor_sha(control),
            "event_control_sha256": _tensor_sha(event_control),
            "native_J": event_branch.get("native_J_on_audited_path"),
            "event_branch": event_branch,
            "source_transition": transition,
            "event_alpha": alpha,
            "chart_face_flux_residual_ratio": chart_residual_ratio,
            "minimum_claim": False, "HVP_calls": 0,
        }
        _write(output, unsupported)
        return unsupported

    pivot = chart.layout["face_flux_coordinate"]
    y0 = torch.cat((face_coordinates[:pivot], face_coordinates[pivot + 1:]))

    def lift(y: Tensor) -> Tensor:
        coords = torch.cat((y[:pivot], y.new_zeros(1), y[pivot:]))
        return chart.from_face_coordinates(coords)

    # Correct the actual face point with the exact reduced Hessian of J∘Γ.
    # This is a local constrained qualification step, not a full-space trial.
    # Persist every HVP boundary so a timeout cannot erase completed work.
    progress_path = output.with_suffix(output.suffix + ".progress.json")
    chart_side = +1
    def chart_objective(y: Tensor, p: Tensor) -> Tensor:
        full = lift(y)
        with transport.selected_face_extension(FACE[0], FACE[1], FACE[2], chart_side):
            return problem.objective(full, p)

    face_newton: list[dict[str, Any]] = []
    eye = torch.eye(CONTROL_SIZE - 1, dtype=event_control.dtype)
    for iteration in range(4):
        if time.monotonic() >= deadline:
            raise TimeoutError("remaining face-Newton budget expired before gradient")
        native_gradient = torch.func.grad(chart_objective, argnums=0)(y0, parameters)
        grad_inf = float(native_gradient.abs().max())
        step_record: dict[str, Any] = {
            "iteration": iteration, "control": lift(y0).tolist(),
            "control_sha256": _tensor_sha(lift(y0)),
            "gradient_inf": grad_inf, "hvp_started": 0, "hvp_returned": 0,
            "hessian_columns": [], "line_search": [],
        }
        face_newton.append(step_record)
        _write(progress_path, {
            "status": "face_newton_in_progress", "phase": "gradient",
            "event_control": lift(y0).tolist(), "face_newton": face_newton,
            "source_transition": transition, "continuation_plan_sha256": continuation_sha256,
            "candidate_receipt_sha256": candidate_digest, "H_checkpoint_sha256": hessian_digest,
        })
        if grad_inf < STATIONARITY_TOL:
            step_record["status"] = "stationary"
            break
        gradient_fn = torch.func.grad(chart_objective, argnums=0)
        columns: list[Tensor] = []
        for column_index in range(CONTROL_SIZE - 1):
            if time.monotonic() >= deadline:
                step_record["status"] = "budget_exhausted"
                _write(progress_path, {
                    "status": "face_newton_partial", "phase": "reduced_hessian",
                    "event_control": lift(y0).tolist(), "face_newton": face_newton,
                    "source_transition": transition,
                    "continuation_plan_sha256": continuation_sha256,
                    "candidate_receipt_sha256": candidate_digest,
                    "H_checkpoint_sha256": hessian_digest,
                })
                raise TimeoutError("remaining face-Newton budget expired during reduced-H columns")
            step_record["hvp_started"] += 1
            _write(progress_path, {
                "status": "face_newton_in_progress", "phase": "reduced_hessian_column_started",
                "column": column_index, "event_control": lift(y0).tolist(),
                "face_newton": face_newton, "source_transition": transition,
                "continuation_plan_sha256": continuation_sha256,
                "candidate_receipt_sha256": candidate_digest,
                "H_checkpoint_sha256": hessian_digest,
            })
            col = torch.func.jvp(lambda y: gradient_fn(y, parameters), (y0,),
                                 (eye[column_index],))[1]
            columns.append(col)
            step_record["hvp_returned"] += 1
            step_record["hessian_columns"].append(col.tolist())
            _write(progress_path, {
                "status": "face_newton_in_progress", "phase": "reduced_hessian_column_returned",
                "column": column_index, "event_control": lift(y0).tolist(),
                "face_newton": face_newton, "source_transition": transition,
                "continuation_plan_sha256": continuation_sha256,
                "candidate_receipt_sha256": candidate_digest,
                "H_checkpoint_sha256": hessian_digest,
            })
        hessian_face = torch.stack(columns, dim=1)
        hnorm = torch.linalg.matrix_norm(hessian_face)
        symmetry = float(torch.linalg.matrix_norm(hessian_face-hessian_face.T) /
                         torch.clamp_min(hnorm, torch.finfo(hessian_face.dtype).tiny))
        step_record["hessian_symmetry_relative"] = symmetry
        if symmetry > qualifier.blocks.SYMMETRY_TOL:
            step_record["status"] = "unsupported_hessian_asymmetry"
            break
        symmetric_hessian = 0.5 * (hessian_face + hessian_face.T)
        try:
            correction = torch.linalg.solve(symmetric_hessian, -native_gradient)
        except RuntimeError:
            step_record["status"] = "singular_reduced_hessian"
            break
        slope = float(torch.dot(native_gradient, correction))
        step_record["newton_slope"] = slope
        if not math.isfinite(slope) or slope >= 0:
            step_record["status"] = "not_descent"
            break
        base_value = float(chart_objective(y0, parameters))
        for armijo_index in range(30):
            scale = 0.5 ** armijo_index
            trial_y = y0 + scale * correction
            trial_value = float(chart_objective(trial_y, parameters))
            accepted = trial_value <= base_value + 1e-4 * scale * slope
            step_record["line_search"].append({"scale": scale, "native_J": trial_value,
                                                "accepted": accepted})
            if accepted:
                y0 = trial_y
                step_record["accepted_scale"] = scale
                step_record["accepted_native_J"] = trial_value
                step_record["status"] = "accepted"
                break
        else:
            step_record["status"] = "no_armijo_step"
            break
        _write(progress_path, {
            "status": "face_newton_in_progress", "phase": "iteration_completed",
            "event_control": lift(y0).tolist(), "face_newton": face_newton,
            "source_transition": transition, "continuation_plan_sha256": continuation_sha256,
            "candidate_receipt_sha256": candidate_digest, "H_checkpoint_sha256": hessian_digest,
        })

    event_control = lift(y0)
    face_coordinates = chart.to_face_coordinates(event_control)
    actual_face_flux = float(_face_value(event_control, spec))
    all_flux = torch.cat(tuple(part.flatten() for part in transport.face_volume_fluxes(
        torch.einsum("k,kij->ij", spec.coefficient_limits * torch.tanh(
            event_control[FLOW_START:FLOW_START + FLOW_COUNT]), spec.psi_basis))))
    face_scale = float(all_flux.abs().max())
    chart_residual_ratio = abs(actual_face_flux) / max(face_scale, torch.finfo(event_control.dtype).tiny)
    if chart_residual_ratio > 128 * torch.finfo(event_control.dtype).eps:
        raise ValueError("face chart Newton left production qy[4,3]=0 beyond FP64 tolerance")
    # At a corrected point the branch signature may legitimately differ from
    # the ray's prior signature; all downstream gates use this actual signature.
    event_branch = qualifier._analysis_branch(problem, event_control, parameters, deadline=deadline)

    sides = (-1, 1)
    gradients_full: dict[int, Tensor] = {}
    tangent_gradients: dict[int, Tensor] = {}
    side_objectives: dict[int, Any] = {}

    for side in sides:
        def objective(y: Tensor, p: Tensor, *, _side: int = side) -> Tensor:
            full = lift(y)
            with transport.selected_face_extension(FACE[0], FACE[1], FACE[2], _side):
                return problem.objective(full, p)

        side_objectives[side] = objective
        tangent_gradients[side] = torch.func.grad(objective, argnums=0)(y0, parameters)
        with transport.selected_face_extension(FACE[0], FACE[1], FACE[2], side):
            gradients_full[side] = torch.func.grad(problem.objective, argnums=0)(event_control, parameters)

    q_gradient = torch.func.grad(lambda c: _face_value(c, spec))(event_control)
    q_normal_norm = torch.linalg.vector_norm(q_gradient)
    if not bool(torch.isfinite(q_normal_norm) & (q_normal_norm > 0)):
        raise ValueError("selected active face has an unresolved normal")
    normal_slopes = {side: float(torch.dot(gradients_full[side], q_gradient) / q_normal_norm)
                     for side in sides}
    normal_budgets = {side: ROUNDING_MULT * torch.finfo(gradient.dtype).eps
        * float(gradients_full[side].norm()) for side in sides}

    tangent_stationary = all(float(g.abs().max()) < STATIONARITY_TOL for g in tangent_gradients.values())
    one_sided_normal_pass = (normal_slopes[+1] > normal_budgets[+1]
                             and normal_slopes[-1] < -normal_budgets[-1])
    partial = {
        "phase": "singleface_point_measured",
        "status": "singleface_event_partial",
        "scope": "conditional one-face event on qy[4,3]; original native 26-control J; not generic nonsmooth support",
        "base_candidate_control_sha256": _tensor_sha(control),
        "event_control": event_control.tolist(), "event_control_sha256": _tensor_sha(event_control),
        "event_projection_l2": float(torch.linalg.vector_norm(event_control-control)),
        "event_alpha": alpha, "event_alpha_bracket": [alpha_low, alpha_high],
        "event_face": {"axis": "y", "row": 4, "column": 3,
            "q_value": float(_face_value(event_control, spec)),
            "chart_residual_ratio": chart_residual_ratio,
            "normal_l2": float(q_normal_norm), "chart": chart.identity,
            "pivot_flow_index": pivot_index},
        "event_native_J": event_branch["native_J_on_audited_path"],
        "event_branch": event_branch,
        "face_newton": face_newton,
        "tangent_coordinates": y0.tolist(),
        "tangent_gradient_inf_by_side": {str(side): float(tangent_gradients[side].abs().max()) for side in sides},
        "full_side_gradient_inf_by_side": {str(side): float(gradients_full[side].abs().max()) for side in sides},
        "one_sided_normal_slopes": {str(side): normal_slopes[side] for side in sides},
        "one_sided_normal_roundoff_budgets": {str(side): normal_budgets[side] for side in sides},
        "tangent_stationary": tangent_stationary,
        "one_sided_normal_pass": one_sided_normal_pass,
        "minimum_claim": False, "B_or_response_computed": False,
        "HVP_calls": 0,
        "source_transition": transition,
        "continuation_plan_sha256": continuation_sha256,
        "input_parameters_sha256": _tensor_sha(parameters),
        "H_checkpoint_sha256": hessian_digest,
        "candidate_receipt_sha256": candidate_digest,
    }
    _write(output, partial)
    if not compute_reduced_curvature or not tangent_stationary or not one_sided_normal_pass:
        partial["status"] = "singleface_first_order_not_qualified"
        partial["reason"] = ("P2 reduced curvature was not run: explicit request required"
            if not compute_reduced_curvature else "tangential stationarity or strict one-sided normal-slope gate failed")
        _write(output, partial)
        return partial
    reduced_hessians: dict[int, Tensor] = {}
    reduced_eigs: dict[int, Tensor] = {}
    reduced_metrics: dict[int, dict[str, Any]] = {}
    if tangent_stationary and one_sided_normal_pass:
        eye = torch.eye(CONTROL_SIZE - 1, dtype=event_control.dtype)
        for side in sides:
            objective = side_objectives[side]
            gradient_fn = torch.func.grad(objective, argnums=0)
            columns = []
            for column_index in range(CONTROL_SIZE - 1):
                if time.monotonic() >= deadline:
                    _write(progress_path, {
                        "status": "two_sided_reduced_hessian_partial",
                        "side": side, "column": column_index,
                        "event_control": event_control.tolist(),
                        "Hessian_columns_started": len(columns) + column_index,
                        "Hessian_columns_returned": sum(len(v.get("columns", [])) for v in reduced_metrics.values()) + len(columns),
                        "source_transition": transition,
                        "continuation_plan_sha256": continuation_sha256,
                        "candidate_receipt_sha256": candidate_digest,
                        "H_checkpoint_sha256": hessian_digest,
                    })
                    raise TimeoutError("remaining one-face P2 budget expired during reduced-H columns")
                _write(progress_path, {
                    "status": "two_sided_reduced_hessian_in_progress", "side": side,
                    "column_started": column_index, "event_control": event_control.tolist(),
                    "Hessian_columns_started": (0 if side == -1 else CONTROL_SIZE-1) + column_index + 1,
                    "Hessian_columns_returned": (0 if side == -1 else CONTROL_SIZE-1) + len(columns),
                    "source_transition": transition,
                    "continuation_plan_sha256": continuation_sha256,
                    "candidate_receipt_sha256": candidate_digest,
                    "H_checkpoint_sha256": hessian_digest,
                })
                column = torch.func.jvp(
                    lambda y: gradient_fn(y, parameters), (y0,), (eye[column_index],),
                )[1]
                columns.append(column)
                _write(progress_path, {
                    "status": "two_sided_reduced_hessian_in_progress", "side": side,
                    "column_returned": column_index, "event_control": event_control.tolist(),
                    "Hessian_columns_started": (0 if side == -1 else CONTROL_SIZE-1) + column_index + 1,
                    "Hessian_columns_returned": (0 if side == -1 else CONTROL_SIZE-1) + len(columns),
                    "source_transition": transition,
                    "continuation_plan_sha256": continuation_sha256,
                    "candidate_receipt_sha256": candidate_digest,
                    "H_checkpoint_sha256": hessian_digest,
                })
            hessian = torch.stack(columns, dim=1)
            hnorm = torch.linalg.matrix_norm(hessian)
            symmetry = float(torch.linalg.matrix_norm(hessian-hessian.T) /
                             torch.clamp_min(hnorm, torch.finfo(hessian.dtype).tiny))
            eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian+hessian.T))
            floor = max(qualifier.blocks.CURVATURE_REL_TOL * float(eigenvalues.abs().max()),
                        ROUNDING_MULT * torch.finfo(hessian.dtype).eps *
                        max(float(hnorm), torch.finfo(hessian.dtype).tiny))
            reduced_hessians[side], reduced_eigs[side] = hessian, eigenvalues
            reduced_metrics[side] = {"hessian_sha256": _tensor_sha(hessian),
                "symmetry_relative": symmetry, "eigenvalues": eigenvalues.tolist(),
                "positive_floor": floor,
                "positive_definite": symmetry <= qualifier.blocks.SYMMETRY_TOL and bool(eigenvalues[0] > floor)}

    p2 = (tangent_stationary and one_sided_normal_pass
          and set(reduced_metrics) == set(sides)
          and all(row["positive_definite"] for row in reduced_metrics.values()))
    _, source_after, _ = qualifier._execution_plan(
        continuation_plan_path=continuation_plan,
        expected_continuation_sha256=continuation_sha256,
    )
    input_after = qualifier.cycle.shared._input_identity(problem, original, control, parameters, truth)
    record = {
        "status": "singleface_local_minimum_qualified" if p2 else "singleface_event_not_minimum_qualified",
        "scope": "conditional one-face event on qy[4,3]; original native 26-control J; not generic nonsmooth support",
        "base_candidate_control_sha256": _tensor_sha(control),
        "event_control": event_control.tolist(), "event_control_sha256": _tensor_sha(event_control),
        "event_projection_l2": float(torch.linalg.vector_norm(event_control-control)),
        "event_face": {"axis": "y", "row": 4, "column": 3,
                       "q_value": float(_face_value(event_control, spec)),
                       "chart_residual_ratio": chart_residual_ratio,
                       "normal_l2": float(q_normal_norm), "chart": chart.identity,
                       "pivot_flow_index": pivot_index},
        "event_branch": event_branch,
        "tangent_stationary": tangent_stationary,
        "tangent_gradient_inf_by_side": {str(side): float(tangent_gradients[side].abs().max()) for side in sides},
        "one_sided_normal_slopes": {str(side): normal_slopes[side] for side in sides},
        "one_sided_normal_roundoff_budgets": {str(side): normal_budgets[side] for side in sides},
        "one_sided_normal_pass": one_sided_normal_pass,
        "reduced_curvature_by_side": {str(side): reduced_metrics[side] for side in reduced_metrics},
        "refinement": {"status": "not_run; no CG tangent refinement"},
        "P2_passed": p2, "minimum_claim": p2,
        "B_or_response_computed": False,
        "Hessian_columns": (CONTROL_SIZE-1) * len(reduced_metrics),
        "source_transition": transition,
        "continuation_plan_sha256": continuation_sha256,
        "input_parameters_sha256": _tensor_sha(parameters),
        "H_checkpoint_sha256": hessian_digest,
        "candidate_receipt_sha256": candidate_digest,
        "source_archive_closure": source_before == source_after,
        "elapsed_seconds": time.monotonic()-started,
    }
    _write(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--hessian-json", type=Path, required=True)
    parser.add_argument("--continuation-plan", type=Path, required=True)
    parser.add_argument("--continuation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remaining-seconds", type=float, default=300.0)
    parser.add_argument("--compute-reduced-curvature", action="store_true")
    args = parser.parse_args()
    result = run(args.candidate_json, args.hessian_json, args.continuation_plan,
                 args.continuation_sha256, args.output,
                 remaining_seconds=args.remaining_seconds,
                 compute_reduced_curvature=args.compute_reduced_curvature)
    print(json.dumps({k: result.get(k) for k in (
        "status", "event_face", "tangent_stationary", "one_sided_normal_pass",
        "reduced_curvature_by_side", "P2_passed", "minimum_claim",
        "Hessian_columns", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
