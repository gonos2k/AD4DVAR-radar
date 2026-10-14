"""Native-J continuation and same-point qualification on a two-face chart."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
QUALIFIER_PATH = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/qualify_candidate.py"
INPUT = HERE / "two_face_diagnostic.json"
ORIGINAL_INPUT = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/continuation3/attempt1/retained_graph_face_qualification.json"
PLAN = HERE / "NATIVE24_PLAN_v6.json"
PLAN_SHA_PATH = HERE / "NATIVE24_PLAN_v6_SHA256.txt"
OUTPUT = HERE / "native24_qualification.json"
CONTINUATION = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/continuation3/CONTINUATION_PLAN.json"
SOURCE_SNAPSHOT = HERE / "native24_launch_snapshot_v6.tar.gz"
FACES = (("y", 4, 3), ("y", 2, 1))
FACE_ENTITIES = ["y:2:1", "y:4:3"]
SIDES = ((-1, -1), (-1, 1), (1, -1), (1, 1))
STATIONARITY_TOL = 1e-10
ROUNDING_MULTIPLIER = 128.0
RSS_CAP_BYTES = 2 * 1024**3
_RUN_STARTED: float | None = None

spec = importlib.util.spec_from_file_location("native24_qualifier", QUALIFIER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("native qualifier is unavailable")
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


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _write(state: dict[str, Any]) -> None:
    if _RUN_STARTED is not None:
        state["elapsed_seconds"] = time.monotonic() - _RUN_STARTED
    def encode(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach().contiguous().cpu().tolist()
        raise TypeError(f"cannot serialize {type(value).__name__}")

    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True,
                                    allow_nan=False, default=encode) + "\n")
    temporary.replace(OUTPUT)


def _load_plan() -> tuple[dict[str, Any], str]:
    raw = PLAN.read_bytes()
    digest = _sha_bytes(raw)
    if PLAN_SHA_PATH.read_text().strip() != digest:
        raise ValueError("native24 plan sidecar does not match frozen plan")
    plan = json.loads(raw)
    if plan.get("status") != "frozen_native24_continuation":
        raise ValueError("native24 plan is not frozen")
    for name, expected in plan["source_files"].items():
        path = ROOT / name
        if not path.is_file() or _sha_bytes(path.read_bytes()) != expected:
            raise ValueError(f"native24 source pin changed: {name}")
    for name, expected in plan["artifact_pins"].items():
        path = ROOT / name
        if not path.is_file() or _sha_bytes(path.read_bytes()) != expected:
            raise ValueError(f"native24 artifact pin changed: {name}")
    if _sha_bytes(INPUT.read_bytes()) != plan["input_sha256"]:
        raise ValueError("two-face diagnostic input changed after freeze")
    if _sha_bytes(SOURCE_SNAPSHOT.read_bytes()) != plan["launch_snapshot_sha256"]:
        raise ValueError("native24 launch snapshot changed after freeze")
    return plan, digest


def _face_flux(control: Tensor, basis: Tensor, limits: Tensor,
               face: tuple[str, int, int]) -> Tensor:
    psi = torch.einsum("k,kij->ij", limits * torch.tanh(control[20:25]), basis)
    qx, qy = transport.face_volume_fluxes(psi)
    return qx[face[1], face[2]] if face[0] == "x" else qy[face[1], face[2]]


def _face_matrix(basis: Tensor, limits: Tensor) -> tuple[Tensor, list[int], list[int]]:
    matrix = torch.stack([
        face_weights_from_basis(basis, axis, (row, column)) * limits
        for axis, row, column in FACES
    ])
    pivots = list(json.loads(INPUT.read_text())["chart"]["pivot_flow_indices"])
    free = list(json.loads(INPUT.read_text())["chart"]["free_flow_indices"])
    return matrix, pivots, free


def _build_lift(control: Tensor, matrix: Tensor, pivots: list[int], free: list[int]):
    pivot_matrix = matrix[:, pivots]
    free_matrix = matrix[:, free]

    def lift(coordinates: Tensor) -> Tensor:
        if coordinates.shape != (24,):
            raise ValueError("native24 chart coordinates must have length 24")
        free_z = torch.tanh(coordinates[20:23])
        pivot_z = torch.linalg.solve(pivot_matrix, -(free_matrix @ free_z))
        z = coordinates.new_zeros(5)
        z = z.index_copy(0, torch.tensor(free), free_z)
        z = z.index_copy(0, torch.tensor(pivots), pivot_z)
        if bool((z.abs() >= 1).any()):
            raise ValueError("two-face chart left the open tanh-flow domain")
        return torch.cat((coordinates[:20], torch.atanh(z), coordinates[23:24]))

    start = torch.cat((control[:20], control[20 + torch.tensor(free)], control[25:26]))
    return lift, start


def _sector_records(problem: Any, control: Tensor, parameters: Tensor,
                    lift: Any, coordinates: Tensor, chart_jacobian: Tensor,
                    deadline: float, state: dict[str, Any]) -> dict[str, Any]:
    gradients: list[Tensor] = []
    tangent_gradients: list[Tensor] = []
    sectors: list[dict[str, Any]] = []
    for side_pair in SIDES:
        if time.monotonic() >= deadline:
            raise TimeoutError("native24 wall budget before a sector gradient")
        if state["gradient_calls_started"] >= state["gradient_call_cap"]:
            raise RuntimeError("native24 gradient cap exhausted")
        selected = ((FACES[0][0], FACES[0][1], FACES[0][2], side_pair[0]),
                    (FACES[1][0], FACES[1][1], FACES[1][2], side_pair[1]))
        state["gradient_calls_started"] += 1
        state["current_sector"] = {"sides": list(side_pair), "state": "started"}
        _write(state)
        with transport.selected_face_extensions(*selected):
            gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
        state["gradient_calls_completed"] += 1
        tangent_gradient = chart_jacobian.T @ gradient
        if not bool(torch.isfinite(gradient).all() and torch.isfinite(tangent_gradient).all()):
            raise ValueError("native24 sector gradient is nonfinite")
        gradients.append(gradient)
        tangent_gradients.append(tangent_gradient)
        sectors.append({
            "sides": list(side_pair),
            "full_gradient": gradient.tolist(),
            "full_gradient_sha256": _tensor_sha(gradient),
            "full_gradient_inf": float(gradient.abs().max()),
            "tangent_gradient": tangent_gradient.tolist(),
            "tangent_gradient_inf": float(tangent_gradient.abs().max()),
        })
        state["current_sector"] = {"sides": list(side_pair), "state": "returned"}
        state["latest_sector_record"] = sectors[-1]
        _write(state)

    full = torch.stack(gradients)
    tangent = torch.stack(tangent_gradients)
    hull = geometry.minimum_sampled_hull(full)
    tangential_hull = geometry.minimum_sampled_hull(tangent)
    normals = torch.func.jacrev(lambda value: torch.stack((
        _face_flux(value, state["basis"], state["limits"], FACES[0]),
        _face_flux(value, state["basis"], state["limits"], FACES[1]),
    )))(control)
    singular = torch.linalg.svdvals(normals)
    rank_two = (float(singular[-1]) > ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps
                * float(singular[0]))
    tangent_scale = max(float(tangent.norm()), torch.finfo(control.dtype).tiny)
    tangent_agreement = float((tangent - tangent[:1]).abs().max())
    tangent_agreement_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * tangent_scale
    gram = normals @ normals.T
    normal_coefficients: list[dict[str, Any]] = []
    normal_signs = True
    for side_pair, gradient in zip(SIDES, gradients):
        eta = torch.linalg.solve(gram, normals @ gradient)
        eta_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * float(gradient.norm()) / torch.linalg.vector_norm(normals, dim=1).clamp_min(torch.finfo(control.dtype).tiny)
        signs = [float(side_pair[i] * eta[i]) > float(eta_budget[i]) for i in range(2)]
        normal_signs = normal_signs and all(signs)
        normal_coefficients.append({
            "sides": list(side_pair),
            "eta": eta.tolist(),
            "eta_sign_margin": [float(side_pair[i] * eta[i]) for i in range(2)],
            "eta_roundoff_budget": eta_budget.tolist(),
            "strict_sign_match": signs,
        })
    tangent_stationary = float(tangential_hull["vector"] and torch.tensor(
        tangential_hull["vector"], dtype=control.dtype).abs().max()) < STATIONARITY_TOL
    full_stationary = float(torch.tensor(hull["vector"], dtype=control.dtype).abs().max()) < STATIONARITY_TOL
    return {
        "full_gradients": full,
        "tangent_gradients": tangent,
        "sectors": sectors,
        "normal_matrix": normals,
        "normal_singular_values": singular.tolist(),
        "normal_rank_two": rank_two,
        "normal_coefficients": normal_coefficients,
        "all_normal_signs_strict": normal_signs,
        "tangent_hull": tangential_hull,
        "full_clarke_hull": hull,
        "tangent_gradient_agreement_inf": tangent_agreement,
        "tangent_gradient_agreement_budget": tangent_agreement_budget,
        "tangent_stationary": tangent_stationary,
        "full_clarke_stationary": full_stationary,
        "first_order_qualified": bool(rank_two and tangent_stationary and full_stationary
                                       and tangent_agreement <= tangent_agreement_budget
                                       and normal_signs),
    }


def _profile_problem(problem: Any):
    original_fv = problem.frozen.fv_transport
    if original_fv is None or original_fv.replay is not True:
        raise ValueError("original fixed problem must retain replay=True")
    profile_fv = replace(original_fv, replay=False)
    frozen = copy.copy(problem.frozen)
    object.__setattr__(frozen, "fv_transport", profile_fv)
    profile = copy.copy(problem)
    object.__setattr__(profile, "frozen", frozen)
    unchanged = {
        "same_basis": _tensor_sha(profile_fv.psi_basis) == _tensor_sha(original_fv.psi_basis),
        "same_limits": _tensor_sha(profile_fv.coefficient_limits) == _tensor_sha(original_fv.coefficient_limits),
        "same_boundary_echo": profile_fv.boundary_echo is original_fv.boundary_echo,
        "same_boundary_support": profile_fv.boundary_support is original_fv.boundary_support,
        "same_substeps": profile_fv.substeps_per_interval == original_fv.substeps_per_interval,
        "same_spacing": profile_fv.spacing_yx == original_fv.spacing_yx,
        "same_reconstruction": profile_fv.reconstruction == original_fv.reconstruction,
        "original_replay": original_fv.replay,
        "profile_replay": profile_fv.replay,
    }
    if not all(value for key, value in unchanged.items() if key.startswith("same_")):
        raise ValueError("replay-off derivative copy changed a physical input")
    return profile, unchanged


def _hessian_profile(profile: Any, control: Tensor, coordinates: Tensor,
                     lift: Any, parameters: Tensor, sides: tuple[int, int],
                     state: dict[str, Any], deadline: float) -> tuple[Tensor, float, Tensor]:
    if state["profile_scalar_calls"] >= state["profile_scalar_call_cap"]:
        raise RuntimeError("native24 retained-graph scalar cap exhausted")
    z = coordinates.detach().clone().requires_grad_(True)
    selected = ((FACES[0][0], FACES[0][1], FACES[0][2], sides[0]),
                (FACES[1][0], FACES[1][1], FACES[1][2], sides[1]))
    state["profile_scalar_calls"] += 1
    state["phase"] = "retained_graph_forward_started"
    state["current_hessian_sides"] = list(sides)
    _write(state)
    with transport.selected_face_extensions(*selected):
        phi = profile.objective(lift(z), parameters)
    if phi.ndim != 0 or not bool(torch.isfinite(phi)):
        raise ValueError("native24 profile returned a nonfinite scalar")
    state["gradient_calls_started"] += 1
    state["phase"] = "retained_graph_gradient_started"
    _write(state)
    gradient = torch.autograd.grad(phi, z, create_graph=True, retain_graph=True)[0]
    state["gradient_calls_completed"] += 1
    state["partial_hessian_rows"] = []
    _write(state)
    rows = []
    identity = torch.eye(24, dtype=z.dtype)
    for index in range(24):
        if time.monotonic() >= deadline:
            raise TimeoutError("native24 wall budget during retained Hessian")
        if state["hessian_rows_started"] >= state["hessian_row_cap"]:
            raise RuntimeError("native24 Hessian row cap exhausted")
        state["hessian_rows_started"] += 1
        state["phase"] = "hessian_row_started"
        state["current_hessian_row"] = {"sides": list(sides), "row": index}
        _write(state)
        row = torch.autograd.grad(gradient[index], z,
                                  retain_graph=index < 23)[0]
        rows.append(row)
        state["hessian_rows_completed"] += 1
        state["partial_hessian_rows"].append(row.tolist())
        state["phase"] = "hessian_row_returned"
        _write(state)
    hessian = torch.stack(rows)
    return hessian, float(phi), gradient.detach()


def _audit_point(problem: Any, parameters: Tensor, control: Tensor,
                 lift: Any, coordinates: Tensor, matrix: Tensor, pivots: list[int],
                 free: list[int], profile: Any, state: dict[str, Any],
                 deadline: float, *, compute_hessian: bool = True) -> dict[str, Any]:
    if time.monotonic() >= deadline:
        raise TimeoutError("native24 wall budget before branch audit")
    if state["native_J_calls_completed"] >= state["native_J_cumulative_cap"]:
        raise RuntimeError("native24 cumulative native-J cap exhausted")
    state["native_J_calls_started"] += 1
    state["phase"] = "actual_branch_J_started"
    _write(state)
    branch = qualifier._analysis_branch(problem, control, parameters, deadline=deadline)
    if branch.get("objective_finite") is not True or branch.get("native_J_on_audited_path") is None:
        raise ValueError("actual native branch audit returned no finite J")
    state["native_J_calls_completed"] += 1
    native_j = float(branch["native_J_on_audited_path"])
    state["phase"] = "actual_branch_J_returned"
    state["latest_branch_J"] = native_j
    state["latest_branch_control_sha256"] = _tensor_sha(control)
    _write(state)
    face_values = [float(_face_flux(control, state["basis"], state["limits"], face)) for face in FACES]
    psi = torch.einsum("k,kij->ij", state["limits"] * torch.tanh(control[20:25]), state["basis"])
    qx, qy = transport.face_volume_fluxes(psi)
    scale = max(float(qx.abs().max()), float(qy.abs().max()), torch.finfo(control.dtype).tiny)
    face_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * scale
    if max(map(abs, face_values)) > face_budget:
        raise ValueError("lifted native24 control left the two-face intersection beyond FP64 budget")
    supported = (branch.get("stage_count") == qualifier.ANALYSIS_STAGES
                 and branch.get("unique_nearzero_face_entities") == FACE_ENTITIES
                 and not branch.get("unique_near_tie_slope_entities"))
    chart_jacobian = torch.func.jacrev(lift)(coordinates)
    data = _sector_records(problem, control, parameters, lift, coordinates,
                           chart_jacobian, deadline, state)
    data.pop("full_gradients"); data.pop("tangent_gradients")
    normal_matrix = torch.tensor(data["normal_matrix"], dtype=control.dtype)
    profile_phi = None
    profile_gradient_gap = None
    plus_sector = data["sectors"][3]
    profile_phi = None
    profile_gradient_gap = None
    profile_J_gap = None
    profile_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * max(1.0, abs(native_j))
    hessian = None
    eigenvalues = None
    hnorm = symmetry = floor = None
    positive = False
    if compute_hessian:
        # The (+,+) retained graph supplies the local Newton model; its
        # derivatives are checked against the original replay=True gradient.
        hessian, profile_phi, profile_gradient = _hessian_profile(
            profile, control, coordinates, lift, parameters, (1, 1), state, deadline
        )
        profile_gradient_gap = float((profile_gradient - torch.tensor(
            plus_sector["tangent_gradient"], dtype=control.dtype
        )).abs().max())
        profile_J_gap = abs(profile_phi - native_j)
        hsym = (hessian + hessian.T) / 2
        eigenvalues = torch.linalg.eigvalsh(hsym)
        hnorm = float(torch.linalg.matrix_norm(hsym))
        symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T)
                         / max(hnorm, torch.finfo(control.dtype).tiny))
        floor = max(qualifier.blocks.CURVATURE_REL_TOL * float(eigenvalues.abs().max()),
                    ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps
                    * max(hnorm, torch.finfo(control.dtype).tiny))
        positive = symmetry <= qualifier.blocks.SYMMETRY_TOL and float(eigenvalues[0]) > floor
    data.update({
        "control": control.tolist(),
        "control_sha256": _tensor_sha(control),
        "coordinates": coordinates.tolist(),
        "native_J": native_j,
        "branch": branch,
        "branch_supported_exactly_two_faces": supported,
        "face_values": face_values,
        "face_roundoff_budget": face_budget,
        "profile_J": profile_phi,
        "profile_J_abs_gap": profile_J_gap,
        "profile_gradient_inf_gap": profile_gradient_gap,
        "profile_comparison_budget": profile_budget,
        "current_hessian": None if hessian is None else hessian.tolist(),
        "current_hessian_sha256": None if hessian is None else _tensor_sha(hessian),
        "current_hessian_eigenvalues": None if eigenvalues is None else eigenvalues.tolist(),
        "current_hessian_lambda_min": None if eigenvalues is None else float(eigenvalues[0]),
        "current_hessian_positive_floor": floor,
        "current_hessian_symmetry_relative": symmetry,
        "current_hessian_positive_definite": positive,
        "normal_matrix": normal_matrix.tolist(),
        "first_order_pass": bool(data["first_order_qualified"] and supported),
    })
    state["iterations"].append(data)
    state["current_point"] = data
    state["native_J"] = native_j
    state["current_control"] = control.tolist()
    state["current_control_sha256"] = _tensor_sha(control)
    _write(state)
    return data


def _native_value(problem: Any, control: Tensor, parameters: Tensor,
                  state: dict[str, Any], deadline: float) -> float:
    if time.monotonic() >= deadline:
        raise TimeoutError("native24 wall budget before Armijo value")
    if state["native_J_calls_completed"] >= state["native_J_cumulative_cap"]:
        raise RuntimeError("native24 cumulative native-J cap exhausted before Armijo callback")
    state["native_J_calls_started"] += 1
    state["phase"] = "native24_scalar_J_started"
    state["pending_native_J_control_sha256"] = _tensor_sha(control)
    _write(state)
    value = problem.objective(control, parameters)
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise ValueError("original native J returned a nonfinite Armijo value")
    state["native_J_calls_completed"] += 1
    return float(value)


def run() -> dict[str, Any]:
    plan, plan_sha = _load_plan()
    if OUTPUT.exists():
        raise ValueError("native24 output already exists; no implicit repeat or resume")
    diagnostic = json.loads(INPUT.read_text())
    if (diagnostic.get("status") != "two_face_first_order_diagnostic_complete"
            or diagnostic.get("native_J_value_calls_completed") != 1
            or diagnostic.get("gradient_calls_completed") != 4
            or diagnostic.get("hessian_calls_completed") != 0):
        raise ValueError("two-face diagnostic is not the exact input checkpoint")
    if diagnostic.get("actual_face_entities") != FACE_ENTITIES:
        raise ValueError("two-face input branch does not report the expected event")
    prior_j = float(diagnostic["native_J"])
    control = torch.tensor(diagnostic["projected_control"], dtype=torch.float64)
    if _tensor_sha(control) != diagnostic["projected_control_sha256"]:
        raise ValueError("two-face starting-control bytes differ from diagnostic")
    started = time.monotonic()
    global _RUN_STARTED
    _RUN_STARTED = started
    deadline = started + float(plan["budget"]["phase_wall_seconds"])
    prior_native_j = int(plan["budget"]["cumulative_native_J_calls_before"])
    state: dict[str, Any] = {
        "status": "in_progress",
        "phase": "input_and_ledger_ready",
        "ledger": {
            "original_wall_cap_seconds": plan["budget"]["original_wall_cap_seconds"],
            "charged_before_phase_seconds": plan["budget"]["charged_seconds_before_phase"],
            "phase_wall_cap_seconds": plan["budget"]["phase_wall_seconds"],
            "remaining_at_phase_start_seconds": plan["budget"]["estimated_remaining_seconds_before_phase"],
            "estimated_remaining_after_phase_seconds": plan["budget"]["estimated_remaining_seconds_before_phase"] - plan["budget"]["phase_wall_seconds"],
            "native_J_calls_before_phase": prior_native_j,
            "native_J_cumulative_cap": plan["budget"]["native_J_cumulative_cap"],
            "hessian_row_cap": plan["budget"]["hessian_row_cap"],
            "gradient_call_cap": plan["budget"]["gradient_call_cap_phase"],
        },
        "plan_sha256": plan_sha,
        "input_sha256": _sha_bytes(INPUT.read_bytes()),
        "input_control_sha256": diagnostic["projected_control_sha256"],
        "input_native_J": prior_j,
        "input_parameters_sha256": diagnostic["fixed_case_identity"].get("parameters_sha256"),
        "source_before": plan["effective_source_hashes"],
        "source_transition": plan["transport_extension_transition"],
        "native_J_calls_started": 0,
        "native_J_calls_completed": prior_native_j,
        "native_J_calls_before_phase": prior_native_j,
        "native_J_cumulative_cap": int(plan["budget"]["native_J_cumulative_cap"]),
        "gradient_calls_started": 0,
        "gradient_calls_completed": 0,
        "gradient_call_cap": int(plan["budget"]["gradient_call_cap_phase"]),
        "hessian_rows_started": 0,
        "hessian_rows_completed": 0,
        "hessian_row_cap": int(plan["budget"]["hessian_row_cap"]),
        "profile_scalar_calls": 0,
        "profile_scalar_call_cap": 32,
        "iterations": [],
        "accepted_steps": [],
        "memory_cap_bytes": RSS_CAP_BYTES,
    }
    # Keep the exact coefficient tensors in state for the fixed face operators.
    case = qualifier.cycle.load_fixed_case()
    problem, original, truth, parameters = (
        case["problem"], case["original"], case["truth"], case["parameters"]
    )
    state["basis"] = problem.frozen.fv_transport.psi_basis
    state["limits"] = problem.frozen.fv_transport.coefficient_limits
    if _tensor_sha(parameters) != plan["parameters_sha256"]:
        raise ValueError("fixed parameter vector changed after plan freeze")
    original_point = json.loads(ORIGINAL_INPUT.read_text())
    original_control = torch.tensor(original_point["control"], dtype=torch.float64)
    if (_tensor_sha(original_control) != diagnostic["input_control_sha256"]
            or original_point.get("control_sha256") != diagnostic["input_control_sha256"]):
        raise ValueError("pre-projection input control differs from the two-face diagnostic anchor")
    fixed_identity = qualifier.cycle.shared._input_identity(
        problem, original, original_control, parameters, truth
    )
    if fixed_identity != diagnostic.get("fixed_case_identity"):
        raise ValueError("fixed-case/original-control identity differs from the two-face diagnostic")
    projected_identity = qualifier.cycle.shared._input_identity(
        problem, original, control, parameters, truth
    )
    if problem.frozen.fv_transport.replay is not True:
        raise ValueError("authoritative original problem is not replay=True")
    profile, profile_identity = _profile_problem(problem)
    state["profile_identity"] = profile_identity
    matrix, pivots, free = _face_matrix(state["basis"], state["limits"])
    lift, coordinates = _build_lift(control, matrix, pivots, free)
    if _tensor_sha(lift(coordinates)) != _tensor_sha(control):
        raise ValueError("two-face chart does not roundtrip the exact diagnostic control")
    chart_jacobian = torch.func.jacrev(lift)(coordinates)
    if chart_jacobian.shape != (26, 24):
        raise ValueError("native24 chart has the wrong dimension")
    state.update(
        chart={"matrix": matrix.tolist(), "pivot_flow_indices": pivots,
               "free_flow_indices": free, "initial_control_sha256": _tensor_sha(control),
               "initial_coordinates_sha256": _tensor_sha(coordinates),
               "jacobian_shape": list(chart_jacobian.shape)},
        initial_control=control.tolist(),
        initial_coordinates=coordinates.tolist(),
        input_fixed_case_identity=fixed_identity,
        projected_fixed_case_identity=projected_identity,
    )
    _write(state)
    if _rss_bytes() > RSS_CAP_BYTES:
        state.update(status="RSS_cap_exhausted_preflight", phase="rss_preflight_gate")
        _write(state)
        return state

    # First recompute the authoritative original-objective value at the same
    # two-face point, and require agreement with its completed diagnostic.
    base_value = _native_value(problem, control, parameters, state, deadline)
    native_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * max(1.0, abs(prior_j))
    if abs(base_value - prior_j) > native_budget:
        state.update(status="initial_native_J_recheck_mismatch", phase="input_J_recheck_failed",
                     recomputed_native_J=base_value, mismatch=abs(base_value - prior_j),
                     elapsed_seconds=time.monotonic() - started)
        _write(state)
        return state
    state["initial_native_J_recheck"] = base_value
    y = coordinates.clone()
    current: dict[str, Any] | None = None
    current_hessian: Tensor | None = None
    for iteration in range(8):
        if _rss_bytes() > RSS_CAP_BYTES:
            state.update(status="RSS_cap_exhausted", phase="iteration_RSS_gate")
            _write(state)
            break
        c = lift(y)
        current = _audit_point(problem, parameters, c, lift, y, matrix,
                               pivots, free, profile, state, deadline)
        if not current["branch_supported_exactly_two_faces"]:
            state.update(status="unsupported_additional_active_faces_or_ties",
                         phase="actual_branch_gate")
            _write(state)
            break
        if current["profile_J_abs_gap"] > current["profile_comparison_budget"]:
            state.update(status="replay_profile_J_mismatch", phase="profile_value_gate")
            _write(state)
            break
        if current["profile_gradient_inf_gap"] > current["profile_comparison_budget"]:
            state.update(status="replay_profile_gradient_mismatch", phase="profile_gradient_gate")
            _write(state)
            break
        if current["first_order_pass"]:
            state.update(status="two_face_first_order_root", phase="first_order_gate")
            _write(state)
            break
        hessian = torch.tensor(current["current_hessian"], dtype=y.dtype)
        symmetric = (hessian + hessian.T) / 2
        eigenvalues = torch.tensor(current["current_hessian_eigenvalues"], dtype=y.dtype)
        hnorm = float(torch.linalg.matrix_norm(symmetric))
        floor = float(current["current_hessian_positive_floor"])
        pd = bool(current["current_hessian_positive_definite"])
        shift = 0.0 if pd else max(0.0, max(floor, 1e-3 * hnorm) - float(eigenvalues[0])) + ROUNDING_MULTIPLIER * torch.finfo(y.dtype).eps * hnorm
        metric = symmetric + shift * torch.eye(24, dtype=y.dtype)
        try:
            factor = torch.linalg.cholesky(metric)
        except RuntimeError:
            state.update(status="modified_native24_hessian_not_spd", phase="modified_H_gate",
                         hessian_eigenvalues=eigenvalues.tolist(), shift=shift)
            _write(state)
            break
        tangent_gradients = torch.tensor(
            [sector["tangent_gradient"] for sector in current["sectors"]], dtype=y.dtype
        )
        tangent_gradient = tangent_gradients.mean(dim=0)
        step = -torch.cholesky_solve(tangent_gradient[:, None], factor).squeeze(1)
        descent = float(torch.dot(tangent_gradient, step))
        iteration_record = {
            "iteration": iteration,
            "control_sha256": current["control_sha256"],
            "native_J": current["native_J"],
            "tangent_gradient_inf": float(tangent_gradient.abs().max()),
            "full_clarke_gradient_inf": max(map(abs, current["full_clarke_hull"]["vector"])),
            "hessian_lambda_min": float(eigenvalues[0]),
            "hessian_positive_definite": pd,
            "modified_newton_shift": shift,
            "descent_product": descent,
            "step_norm": float(torch.linalg.vector_norm(step)),
            "line_search": [],
        }
        state["iterations"][-1]["optimizer_step"] = iteration_record
        _write(state)
        if not math.isfinite(descent) or descent >= 0:
            state.update(status="native24_step_not_descent", phase="descent_gate")
            _write(state)
            break
        accepted = False
        for backtrack in range(28):
            if time.monotonic() >= deadline or state["native_J_calls_completed"] >= state["native_J_cumulative_cap"]:
                state.update(status="native24_budget_exhausted", phase="line_search_budget_gate")
                _write(state)
                break
            alpha = 0.5**backtrack
            trial_y = y + alpha * step
            try:
                trial_control = lift(trial_y)
            except (ValueError, RuntimeError) as exc:
                iteration_record["line_search"].append({"alpha": alpha, "status": "chart_domain_reject", "error": str(exc)})
                _write(state)
                continue
            trial_value = _native_value(problem, trial_control, parameters, state, deadline)
            armijo_rhs = current["native_J"] + 1e-4 * alpha * descent
            trial_record = {"alpha": alpha, "control_sha256": _tensor_sha(trial_control),
                            "native_J": trial_value, "armijo_rhs": armijo_rhs,
                            "status": "accepted" if trial_value <= armijo_rhs else "Armijo_reject"}
            iteration_record["line_search"].append(trial_record)
            state["phase"] = "native24_Armijo_trial"
            _write(state)
            if trial_value <= armijo_rhs:
                accepted = True
                y = trial_y
                step_record = {"iteration": iteration, "alpha": alpha,
                               "native_J": trial_value,
                               "control_sha256": _tensor_sha(trial_control),
                               "step_norm": alpha * float(torch.linalg.vector_norm(step))}
                state["accepted_steps"].append(step_record)
                state["current_control"] = trial_control.tolist()
                state["current_control_sha256"] = _tensor_sha(trial_control)
                state["native_J"] = trial_value
                state["phase"] = "native24_step_accepted"
                _write(state)
                break
        if state.get("status") == "native24_budget_exhausted":
            break
        if not accepted:
            state.update(status="native24_Armijo_grid_exhausted", phase="line_search_failed")
            _write(state)
            break
        if iteration == 7:
            state.update(status="eight_native24_steps_complete", phase="final_point_audit_pending")
            _write(state)

    # Always re-audit the last accepted/current control before terminal status.
    if state.get("status") in ("eight_native24_steps_complete", "two_face_first_order_root"):
        final_y = y
        final_control = lift(final_y)
        final = _audit_point(problem, parameters, final_control, lift, final_y,
                             matrix, pivots, free, profile, state, deadline,
                             compute_hessian=False)
        if not final["branch_supported_exactly_two_faces"]:
            state.update(status="final_actual_branch_gate_failed", phase="final_branch_gate")
            _write(state)
        elif not final["first_order_pass"]:
            state.update(status="eight_steps_without_two_face_stationarity", phase="final_first_order_not_met")
            _write(state)
        else:
            final_hessians = {}
            final_pd = True
            for sides in SIDES:
                hessian, phi, profile_gradient = _hessian_profile(
                    profile, final_control, final_y, lift, parameters, sides, state, deadline
                )
                sym = (hessian + hessian.T) / 2
                eig = torch.linalg.eigvalsh(sym)
                norm = float(torch.linalg.matrix_norm(sym))
                symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T)
                                 / max(norm, torch.finfo(sym.dtype).tiny))
                floor = max(qualifier.blocks.CURVATURE_REL_TOL * float(eig.abs().max()),
                            ROUNDING_MULTIPLIER * torch.finfo(sym.dtype).eps
                            * max(norm, torch.finfo(sym.dtype).tiny))
                pd = symmetry <= qualifier.blocks.SYMMETRY_TOL and float(eig[0]) > floor
                final_pd = final_pd and pd
                key = f"{sides[0]},{sides[1]}"
                sector_index = SIDES.index(sides)
                original_tangent = torch.tensor(
                    final["sectors"][sector_index]["tangent_gradient"], dtype=final_control.dtype
                )
                profile_gradient_gap = float((profile_gradient - original_tangent).abs().max())
                profile_J_gap = abs(phi - float(final["native_J"]))
                profile_budget = ROUNDING_MULTIPLIER * torch.finfo(final_control.dtype).eps * max(
                    1.0, abs(float(final["native_J"]))
                )
                profile_match = profile_gradient_gap <= profile_budget and profile_J_gap <= profile_budget
                final_pd = final_pd and profile_match
                final_hessians[key] = {
                    "hessian": hessian.tolist(),
                    "hessian_sha256": _tensor_sha(hessian),
                    "control_sha256": _tensor_sha(final_control),
                    "eigenvalues": eig.tolist(),
                    "lambda_min": float(eig[0]),
                    "symmetry_relative": symmetry,
                    "positive_floor": floor,
                    "positive_definite": pd,
                    "profile_J": phi,
                    "profile_J_abs_gap": profile_J_gap,
                    "profile_gradient_inf": float(profile_gradient.abs().max()),
                    "profile_gradient_inf_gap": profile_gradient_gap,
                    "profile_comparison_budget": profile_budget,
                    "profile_match": profile_match,
                }
                state["final_hessians"] = final_hessians
                state["phase"] = "final_quadrant_P2_checkpoint"
                _write(state)
            state.update(
                status="qualified_original_native_J_two_face_minimum" if final_pd else "two_face_stationary_P2_failed",
                phase="terminal_same_point_qualification",
                final_hessians=final_hessians,
                minimum_claim=final_pd,
                response_claim=False,
                final_point=final,
            )
            _write(state)
    state["elapsed_seconds"] = time.monotonic() - started
    state["peak_rss_bytes"] = _rss_bytes()
    state["native_J_calls_completed_total"] = state["native_J_calls_completed"]
    state["hessian_rows_completed_total"] = state["hessian_rows_completed"]
    state["ledger"]["charged_at_phase_end_seconds"] = plan["budget"]["charged_seconds_before_phase"] + state["elapsed_seconds"]
    state["ledger"]["estimated_remaining_at_phase_end_seconds"] = 3600.0 - state["ledger"]["charged_at_phase_end_seconds"]
    state["ledger"]["cumulative_native_J_calls_at_phase_end"] = state["native_J_calls_completed"]
    state["source_after"] = plan["effective_source_hashes"]
    state["source_transition"] = plan["transport_extension_transition"]
    state["source_archive_closure"] = all(
        _sha_bytes((ROOT / name).read_bytes()) == expected
        for name, expected in plan["source_files"].items()
    ) and all(_sha_bytes((ROOT / name).read_bytes()) == expected
              for name, expected in plan["artifact_pins"].items())
    if not state["source_archive_closure"]:
        state["status"] = "terminal_source_closure_failed"
    _write(state)
    return state


if __name__ == "__main__":
    try:
        result = run()
    except BaseException as error:
        if OUTPUT.exists():
            record = json.loads(OUTPUT.read_text())
            record.update(status="terminal_exception", phase=record.get("phase", "unknown"),
                          terminal_exception_type=type(error).__name__,
                          terminal_exception=str(error))
            if "ledger" in record and "charged_before_phase_seconds" in record["ledger"]:
                charged = float(record["ledger"]["charged_before_phase_seconds"])
                elapsed = float(record.get("elapsed_seconds", 0.0))
                record["ledger"]["charged_at_terminal_exception_seconds"] = charged + elapsed
                record["ledger"]["estimated_remaining_at_terminal_exception_seconds"] = 3600.0 - charged - elapsed
            _write(record)
        raise
    print(json.dumps({key: result.get(key) for key in (
        "status", "phase", "native_J", "current_control_sha256", "accepted_steps",
        "native_J_calls_completed", "gradient_calls_completed", "hessian_rows_completed",
        "minimum_claim", "elapsed_seconds", "ledger",
    )}, indent=2, sort_keys=True))
