"""Bounded first-order diagnostic at the actual two-face intersection."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/continuation3/attempt1/retained_graph_face_qualification.json"
QUALIFIER = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/qualify_candidate.py"
CONTINUATION = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/continuation3/CONTINUATION_PLAN.json"
PLAN = HERE / "TWO_FACE_PLAN.json"
OUTPUT = HERE / "two_face_diagnostic.json"
PLAN_SHA = HERE / "TWO_FACE_PLAN_SHA256.txt"
FACE1 = ("y", 4, 3)
FACE2 = ("y", 2, 1)
EXPECTED_FACES = ["y:2:1", "y:4:3"]
ROUNDING_MULTIPLIER = 128.0

spec = importlib.util.spec_from_file_location("two_face_native_qualifier", QUALIFIER)
if spec is None or spec.loader is None:
    raise RuntimeError("native qualifier cannot be loaded")
qualifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qualifier
spec.loader.exec_module(qualifier)

from advar import transport  # noqa: E402
from examples.weather_scenarios import fv_point_3h_active_intersection_diagnostic as geometry  # noqa: E402
from examples.weather_scenarios.fv_face_flux_coordinates import face_weights_from_basis  # noqa: E402


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return _sha_bytes(value.detach().contiguous().cpu().numpy().tobytes())


def _write(record: dict[str, Any]) -> None:
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def _read_plan() -> tuple[dict[str, Any], str]:
    raw = PLAN.read_bytes()
    digest = _sha_bytes(raw)
    if PLAN_SHA.read_text().strip() != digest:
        raise ValueError("two-face plan sidecar does not match its bytes")
    plan = json.loads(raw)
    if plan.get("status") != "frozen_two_face_diagnostic":
        raise ValueError("two-face plan is not frozen")
    for name, expected in plan["source_files"].items():
        path = ROOT / name
        if not path.is_file() or _sha_bytes(path.read_bytes()) != expected:
            raise ValueError(f"pinned source/input changed: {name}")
    if _sha_bytes(INPUT.read_bytes()) != plan["input_sha256"]:
        raise ValueError("current-point input changed after plan freeze")
    return plan, digest


def _fluxes(control: Tensor, basis: Tensor, limits: Tensor) -> tuple[Tensor, Tensor]:
    psi = torch.einsum("k,kij->ij", limits * torch.tanh(control[20:25]), basis)
    return transport.face_volume_fluxes(psi)


def _chart(limits: Tensor, basis: Tensor) -> tuple[Tensor, list[int], list[int], dict[str, Any]]:
    rows = []
    for axis, row, column in (FACE1, FACE2):
        weights = face_weights_from_basis(basis, axis, (row, column))
        rows.append(weights * limits)
    matrix = torch.stack(rows)
    pairs = []
    for pair in itertools.combinations(range(5), 2):
        determinant = torch.linalg.det(matrix[:, list(pair)])
        pairs.append((abs(float(determinant)), pair, float(determinant)))
    magnitude, pivots_tuple, signed_det = max(pairs, key=lambda item: item[0])
    if not math.isfinite(magnitude) or magnitude <= 128 * torch.finfo(matrix.dtype).eps * float(torch.linalg.matrix_norm(matrix) ** 2):
        raise ValueError("two face constraints have no numerically resolved pivot pair")
    pivots = list(pivots_tuple)
    free = [i for i in range(5) if i not in pivots]
    metadata = {
        "faces": [list(FACE1), list(FACE2)],
        "constraint_matrix": matrix.tolist(),
        "pivot_flow_indices": pivots,
        "free_flow_indices": free,
        "pivot_determinant_signed": signed_det,
        "pivot_determinant_abs": magnitude,
    }
    return matrix, pivots, free, metadata


def _make_intersection_chart(control: Tensor, matrix: Tensor,
                             pivots: list[int], free: list[int]):
    pivot_matrix = matrix[:, pivots]
    free_matrix = matrix[:, free]

    def lift(coordinates: Tensor) -> Tensor:
        if coordinates.shape != (24,):
            raise ValueError("two-face chart coordinates must have length 24")
        free_z = torch.tanh(coordinates[20:23])
        pivot_z = torch.linalg.solve(pivot_matrix, -(free_matrix @ free_z))
        z = coordinates.new_zeros(5)
        z = z.index_copy(0, torch.tensor(free), free_z)
        z = z.index_copy(0, torch.tensor(pivots), pivot_z)
        flow = torch.atanh(z)
        return torch.cat((coordinates[:20], flow, coordinates[23:24]))

    current_flow_fraction = torch.tanh(control[20:25])
    initial = torch.cat((
        control[:20],
        control[20 + torch.tensor(free)],
        control[25:26],
    ))
    projected = lift(initial)
    return lift, initial, projected, current_flow_fraction


def _face_value(control: Tensor, basis: Tensor, limits: Tensor,
                face: tuple[str, int, int]) -> Tensor:
    qx, qy = _fluxes(control, basis, limits)
    return qx[face[1], face[2]] if face[0] == "x" else qy[face[1], face[2]]


def run() -> dict[str, Any]:
    plan, plan_sha = _read_plan()
    if OUTPUT.exists():
        raise ValueError("output already exists; this bounded phase cannot resume implicitly")
    if (plan["budget"]["cumulative_native_J_calls_before"] != 249
            or plan["budget"]["phase_native_J_value_cap"] != 12
            or plan["budget"]["phase_gradient_cap"] != 12
            or plan["budget"]["phase_hessian_cap"] != 0):
        raise ValueError("two-face cumulative budget differs from the frozen declaration")
    start = time.monotonic()
    deadline = start + float(plan["budget"]["phase_wall_seconds"])
    prior = json.loads(INPUT.read_text())
    control = torch.tensor(prior["control"], dtype=torch.float64)
    if _tensor_sha(control) != prior["control_sha256"]:
        raise ValueError("candidate control bytes do not match the retained v9 result")
    case = qualifier.cycle.load_fixed_case()
    problem, original, truth, parameters = (
        case["problem"], case["original"], case["truth"], case["parameters"]
    )
    if _tensor_sha(parameters) != prior["input_parameters_sha256"]:
        raise ValueError("fixed parameter vector differs from the retained v9 input")
    continuation = json.loads(CONTINUATION.read_text())
    continuation_sha = _sha_bytes(CONTINUATION.read_bytes())
    _, source_before, source_transition = qualifier._execution_plan(
        continuation_plan_path=CONTINUATION,
        expected_continuation_sha256=continuation_sha,
    )
    fixed_identity = qualifier.cycle.shared._input_identity(
        problem, original, control, parameters, truth
    )
    if not isinstance(fixed_identity, dict):
        raise ValueError("fixed-case identity audit did not return a mapping")
    transport_spec = problem.frozen.fv_transport
    limits = transport_spec.coefficient_limits
    basis = transport_spec.psi_basis
    matrix, pivots, free, chart_metadata = _chart(limits, basis)
    lift, initial, projected, original_z = _make_intersection_chart(
        control, matrix, pivots, free
    )
    if not bool(torch.isfinite(projected).all()):
        raise ValueError("two-face intersection chart produced nonfinite control")
    projected_sha = _tensor_sha(projected)
    q1 = _face_value(projected, basis, limits, FACE1)
    q2 = _face_value(projected, basis, limits, FACE2)
    qx, qy = _fluxes(projected, basis, limits)
    flux_scale = torch.maximum(qx.abs().max(), qy.abs().max())
    face_roundoff = ROUNDING_MULTIPLIER * torch.finfo(projected.dtype).eps * max(
        float(flux_scale), torch.finfo(projected.dtype).tiny
    )
    face_residuals = [float(q1), float(q2)]
    if max(abs(v) for v in face_residuals) > face_roundoff:
        raise ValueError("projected control does not resolve both face constraints")
    chart_jacobian = torch.func.jacrev(lift)(initial)
    if chart_jacobian.shape != (26, 24) or not bool(torch.isfinite(chart_jacobian).all()):
        raise ValueError("intersection chart Jacobian is malformed")
    normal_matrix = torch.func.jacrev(
        lambda c: torch.stack((_face_value(c, basis, limits, FACE1),
                               _face_value(c, basis, limits, FACE2)))
    )(projected)
    singular_values = torch.linalg.svdvals(normal_matrix)
    if float(singular_values[-1]) <= 128 * torch.finfo(projected.dtype).eps * float(singular_values[0]):
        raise ValueError("two face normals are dependent at the projected control")
    tangent_residual = float(torch.linalg.matrix_norm(normal_matrix @ chart_jacobian))

    state: dict[str, Any] = {
        "status": "in_progress",
        "phase": "geometry_preflight_complete",
        "plan_sha256": plan_sha,
        "input_path": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": _sha_bytes(INPUT.read_bytes()),
        "input_control_sha256": prior["control_sha256"],
        "projected_control_sha256": projected_sha,
        "source_before": source_before,
        "source_transition": source_transition,
        "continuation_plan_sha256": continuation_sha,
        "fixed_case_identity": fixed_identity,
        "chart": chart_metadata,
        "projected_control": projected.tolist(),
        "projection": {
            "control_euclidean_norm": float(torch.linalg.vector_norm(projected - control)),
            "flow_fraction_euclidean_norm": float(torch.linalg.vector_norm(
                torch.tanh(projected[20:25]) - original_z
            )),
            "face_values": face_residuals,
            "face_roundoff_budget": face_roundoff,
            "normal_singular_values": singular_values.tolist(),
            "normal_times_tangent_jacobian_frobenius": tangent_residual,
            "tangent_chart_shape": list(chart_jacobian.shape),
        },
        "limits": plan["budget"],
        "native_J_value_calls_started": 0,
        "native_J_value_calls_completed": 0,
        "gradient_calls_started": 0,
        "gradient_calls_completed": 0,
        "hessian_calls_started": 0,
        "hessian_calls_completed": 0,
        "sector_gradients": [],
    }
    _write(state)

    if time.monotonic() >= deadline:
        raise TimeoutError("phase deadline elapsed before actual branch audit")
    state["native_J_value_calls_started"] = 1
    state["phase"] = "actual_intersection_branch_started"
    _write(state)
    branch = qualifier._analysis_branch(problem, projected, parameters, deadline=deadline)
    state["native_J_value_calls_completed"] = 1
    near_faces = branch.get("unique_nearzero_face_entities")
    supported = (
        branch.get("stage_count") == qualifier.ANALYSIS_STAGES
        and near_faces == EXPECTED_FACES
        and not branch.get("unique_near_tie_slope_entities")
        and branch.get("objective_finite") is True
    )
    state.update(
        phase="actual_intersection_branch_complete",
        branch=branch,
        native_J=branch.get("native_J_on_audited_path"),
        expected_face_entities=EXPECTED_FACES,
        actual_face_entities=near_faces,
        exactly_two_expected_faces=supported,
    )
    _write(state)
    if not supported:
        state.update(status="unsupported_actual_two_face_branch", elapsed_seconds=time.monotonic() - start)
        _write(state)
        return state

    # Offset only the second face's signed volume while remaining tangent to
    # the first face. The two first-face extensions supply the other sector bit.
    first_row, second_row = matrix[0], matrix[1]
    second_transverse = second_row - torch.dot(second_row, first_row) / torch.dot(first_row, first_row) * first_row
    second_norm_sq = torch.dot(second_row, second_transverse)
    if not bool(torch.isfinite(second_norm_sq) and second_norm_sq > 0):
        raise ValueError("second face has no resolved transverse direction")
    epsilon_flux = 1.0e-7 * float(flux_scale)
    if epsilon_flux <= 8.0 * face_roundoff:
        epsilon_flux = 16.0 * face_roundoff
    z0 = torch.tanh(projected[20:25])
    sector_eps = {}
    for side2 in (-1, 1):
        z = z0 + (side2 * epsilon_flux / float(second_norm_sq)) * second_transverse
        if bool((z.abs() >= 1).any()):
            raise ValueError("sector offset leaves the open tanh-flow domain")
        c = torch.cat((projected[:20], torch.atanh(z), projected[25:26]))
        actual1 = _face_value(c, basis, limits, FACE1)
        actual2 = _face_value(c, basis, limits, FACE2)
        if abs(float(actual1)) > 2 * face_roundoff or float(actual2) * side2 <= 8 * face_roundoff:
            raise ValueError("sector offset did not realize the requested Q2 side while holding Q1")
        sector_eps[str(side2)] = {"control": c, "control_sha256": _tensor_sha(c),
                                  "face_values": [float(actual1), float(actual2)]}

    hull_vectors = []
    for side1 in (-1, 1):
        for side2 in (-1, 1):
            if time.monotonic() >= deadline:
                state.update(status="phase_wall_budget_exhausted", phase="sector_gradient_budget_gate",
                             elapsed_seconds=time.monotonic() - start)
                _write(state)
                return state
            if state["gradient_calls_started"] >= plan["budget"]["phase_gradient_cap"]:
                state.update(status="gradient_cap_exhausted", phase="sector_gradient_cap_gate")
                _write(state)
                return state
            offset = sector_eps[str(side2)]
            record = {
                "side_q1_extension": side1,
                "side_q2_reachable_offset": side2,
                "epsilon_q2": epsilon_flux,
                "control_sha256": offset["control_sha256"],
                "face_values": offset["face_values"],
                "gradient_started": True,
                "gradient_returned": False,
            }
            state["sector_gradients"].append(record)
            state["gradient_calls_started"] += 1
            state["phase"] = "sector_gradient_started"
            _write(state)
            with transport.selected_face_extension(FACE1[0], FACE1[1], FACE1[2], side1):
                full_gradient = torch.func.grad(problem.objective, argnums=0)(
                    offset["control"], parameters
                )
            state["gradient_calls_completed"] += 1
            tangent_gradient = chart_jacobian.T @ full_gradient
            if not bool(torch.isfinite(tangent_gradient).all()):
                raise ValueError("sector tangent gradient is nonfinite")
            record.update(
                gradient_returned=True,
                full_gradient_inf=float(full_gradient.abs().max()),
                tangent_gradient_inf=float(tangent_gradient.abs().max()),
                tangent_gradient=tangent_gradient.tolist(),
                full_gradient_sha256=_tensor_sha(full_gradient),
            )
            hull_vectors.append(tangent_gradient)
            state["phase"] = "sector_gradient_returned"
            state["elapsed_seconds"] = time.monotonic() - start
            _write(state)

    gradients = torch.stack(hull_vectors)
    hull = geometry.minimum_sampled_hull(gradients)
    _, source_after, _ = qualifier._execution_plan(
        continuation_plan_path=CONTINUATION,
        expected_continuation_sha256=continuation_sha,
    )
    state.update(
        status="two_face_first_order_diagnostic_complete",
        phase="sector_hull_complete",
        sampled_sector_gradient_hull=hull,
        source_after=source_after,
        source_archive_closure=source_before == source_after,
        elapsed_seconds=time.monotonic() - start,
        total_native_J_value_calls=plan["budget"]["cumulative_native_J_calls_before"] + state["native_J_value_calls_completed"],
        limitation="Four finite epsilon sector-gradient samples and their convex hull are diagnostic, not limiting-gradient or minimum certification.",
        minimum_claim=False,
        response_claim=False,
    )
    _write(state)
    return state


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
