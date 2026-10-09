"""Compare baseline and robust-GN Jacobi tangent directions at one saved endpoint."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, cast

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport, variational as v
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_resume as resume
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_PRECONDITION_COMPARISON_PLAN_20261009.json"
RESUME_PLAN = EVIDENCE / "TANGENT_RESUME_PLAN_20261009.json"
RESUME_PLAN_SHA = "d5c4fa93ff4c3fddedb6d458fedea916c0f06d32edfe8bf55d2fd44449ac9d10"
RESUME_ARCHIVE = EVIDENCE / "TANGENT_RESUME_ARCHIVE_20261009.json"
BASE_ARCHIVE = EVIDENCE / "tangent_resume_20261009_attempt1/step.json.gz"
BASE_RUN = EVIDENCE / "tangent_resume_20261009_attempt1/step.run.json"
BASE_RESOURCE = EVIDENCE / "tangent_resume_20261009_attempt1/step.resource.json"
SOURCE_MANIFEST = EVIDENCE / "precond_source_20261009/manifest.json"
BASE_CONTROL_SHA = "33cb86ca73a404e6ff9260acbf63ee97f248ac0c1f529427eb1af1ec85a43586"
BASE_THETA = 0.3295008210283633
BASE_OBJECTIVE = 0.06124426330405451
BASE_F_SQUARED = 0.005145254919599806
FACE = tangent.FACE
SELF = "examples/weather_scenarios/fv_point_3h_tangent_precondition_comparison.py"
TEST = "tests/test_fv_point_3h_tangent_precondition_comparison.py"
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 1024**3, "direction_models": 2,
    "candidate_grid_per_direction": 16, "radius": 0.05,
    "jacobian_row_vjp_calls": 24, "hvp_calls": 4,
    "max_accepted_iterations": 1, "optimizer_steps": 1,
    "pcg_solves": 0, "dense_solves": 0, "diagonal_floor": 1.0,
    "root_claim": False, "minimum_claim": False,
    "response_claim": False, "score_claim": False,
}
METHOD = {
    "observation_rows": 12,
    "residual_space": "quality_standardized_then_same_time_correlation_whitened_point_dbz",
    "robust_loss": "pseudo_huber_delta_2",
    "field_smoothness_weight": 0.0,
    "neural_prior": False,
    "diagonal_floor": 1.0,
    "directions": ["baseline_tangent", "robust_gn_jacobi"],
    "candidate_selection": "actual_F_squared_then_native_J_then_baseline_with_128_epsilon_ties",
}
ANALYSIS_STAGES = 360
OBSERVATION_ROWS = 12
PRIOR_FLOOR = 1.0
J_C1 = F_C1 = 1e-4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _close(a: float, b: float, dtype: torch.dtype, factor: float = 128.0) -> bool:
    scale = max(abs(a), abs(b), torch.finfo(dtype).tiny)
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= factor * torch.finfo(dtype).eps * scale


def _residual_pair_roundoff_budget(problem: Any, parameters: Tensor,
        residual_minus: Tensor, residual_plus: Tensor) -> tuple[Tensor, Tensor]:
    """Scale rowwise parity tolerance by the standardized/whitened construction."""
    quality = problem.quality_weight
    std = problem.observation_std_dbz
    observed = parameters[:-1].reshape_as(problem.observation_dbz)
    if residual_minus.numel() == OBSERVATION_ROWS and residual_plus.numel() == OBSERVATION_ROWS:
        residual_minus = residual_minus.reshape(3, OBSERVATION_ROWS // 3)
        residual_plus = residual_plus.reshape(3, OBSERVATION_ROWS // 3)
    if (quality.shape != (3, OBSERVATION_ROWS // 3) or std.shape != quality.shape
            or observed.shape != quality.shape
            or residual_minus.shape != quality.shape or residual_plus.shape != quality.shape):
        raise ValueError("fixed residual-pair roundoff inputs do not have the 3x4 profile")
    whitened_observed = quality.sqrt() * observed / std
    whitener = getattr(problem, "_correlation_whitener", None)
    correlation = getattr(problem, "observation_correlation", None)
    if whitener is None:
        transform = torch.eye(quality.shape[1], dtype=quality.dtype, device=quality.device)
        correlation_norm = 1.0
    else:
        if (whitener.shape != (quality.shape[1], quality.shape[1])
                or whitener.dtype != quality.dtype or whitener.device != quality.device):
            raise ValueError("fixed point residual whitener has an invalid layout")
        transform = whitener
        if correlation is None or correlation.shape != transform.shape:
            raise ValueError("fixed point whitener is missing its correlation matrix")
        correlation_norm = float(torch.linalg.matrix_norm(correlation, ord=float("inf")))
    epsilon = torch.finfo(quality.dtype).eps
    scales = torch.empty_like(quality)
    for time_index in range(quality.shape[0]):
        residual_norm = max(float(torch.linalg.vector_norm(residual_minus[time_index])),
            float(torch.linalg.vector_norm(residual_plus[time_index])))
        raw_residual_bound = math.sqrt(correlation_norm) * residual_norm
        scaled_prediction_bound = (2.0 * whitened_observed[time_index].abs()
            + raw_residual_bound)
        scales[time_index] = transform.abs() @ scaled_prediction_bound
    return scales * (128.0 * epsilon), scales


def _projected_row_parity(jacobian_minus: Tensor, jacobian_plus: Tensor,
        projector: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    projected_minus = jacobian_minus @ projector
    projected_plus = jacobian_plus @ projector
    ambient_minus = torch.linalg.vector_norm(jacobian_minus, dim=1)
    ambient_plus = torch.linalg.vector_norm(jacobian_plus, dim=1)
    projected_minus_norm = torch.linalg.vector_norm(projected_minus, dim=1)
    projected_plus_norm = torch.linalg.vector_norm(projected_plus, dim=1)
    scale = torch.stack((ambient_minus, ambient_plus,
        projected_minus_norm, projected_plus_norm)).amax(dim=0)
    scale = scale.clamp_min(torch.finfo(jacobian_minus.dtype).tiny)
    error = torch.linalg.vector_norm(projected_minus - projected_plus, dim=1)
    budget = 128.0 * torch.finfo(jacobian_minus.dtype).eps * scale
    return error, budget, ambient_minus, ambient_plus, torch.stack((projected_minus_norm, projected_plus_norm))


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("precondition-comparison plan identity mismatch")
    if _sha(RESUME_PLAN) != RESUME_PLAN_SHA:
        raise ValueError("accepted tangent-resume plan digest changed")
    prior = json.loads(RESUME_PLAN.read_text())
    snapshots = json.loads(SOURCE_MANIFEST.read_text())["snapshots"]
    plan = json.loads(path.read_text())
    prior_sources, prior_archives = prior["source_files"], prior["archive_files"]
    expected_sources = set(prior_sources) | {SELF, TEST}
    expected_archives = set(prior_archives) | {
        RESUME_PLAN.relative_to(ROOT).as_posix(), RESUME_ARCHIVE.relative_to(ROOT).as_posix(),
        BASE_ARCHIVE.relative_to(ROOT).as_posix(), BASE_RUN.relative_to(ROOT).as_posix(),
        BASE_RESOURCE.relative_to(ROOT).as_posix(), SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        *[snapshot["archive_path"] for snapshot in snapshots.values()],
    }
    if (plan.get("experiment_kind") != "current_tangent_precondition_comparison"
            or plan.get("policy") != POLICY
            or plan.get("producing_plan") != RESUME_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != RESUME_PLAN_SHA
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA
            or plan.get("face") != FACE
            or plan.get("method") != METHOD
            or plan.get("producer_source_snapshots") != snapshots
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 131
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 116):
        raise ValueError("precondition-comparison plan scope or policy changed")
    sources, archives = plan["source_files"], plan["archive_files"]
    if any(archives.get(name) != value for name, value in prior_archives.items()):
        raise ValueError("inherited tangent-resume archive pin changed")
    snapshot_names = set(snapshots)
    if any(sources.get(name) != value for name, value in prior_sources.items()
           if name not in snapshot_names):
        raise ValueError("inherited tangent-resume source pin changed")
    if any(prior_sources.get(name) != snapshot["sha256"] for name, snapshot in snapshots.items()):
        raise ValueError("pre-comparison source snapshot differs from the resume plan")
    if any(sources.get(name) != _sha(ROOT / name) for name in snapshots):
        raise ValueError("current modified runner/library source pins differ from live files")
    for name, snapshot in snapshots.items():
        if (archives.get(snapshot["archive_path"]) != snapshot["sha256"]
                or _sha(ROOT / snapshot["archive_path"]) != snapshot["sha256"]):
            raise ValueError(f"pre-comparison source archive changed: {name}")
    for name, value in {**sources, **archives}.items():
        path_value = ROOT / name
        if path_value.is_symlink() or not path_value.resolve().is_relative_to(ROOT.resolve()) or _sha(path_value) != value:
            raise ValueError(f"precondition-comparison source/archive pin mismatch: {name}")
    archive = json.loads(RESUME_ARCHIVE.read_text())
    if (plan.get("direction_archive") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or plan.get("direction_archive_sha256") != _sha(BASE_ARCHIVE)
            or plan.get("direction_run") != BASE_RUN.relative_to(ROOT).as_posix()
            or plan.get("direction_run_sha256") != _sha(BASE_RUN)
            or plan.get("direction_resource") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or plan.get("direction_resource_sha256") != _sha(BASE_RESOURCE)
            or plan.get("direction_archive_manifest") != RESUME_ARCHIVE.relative_to(ROOT).as_posix()
            or plan.get("direction_archive_manifest_sha256") != _sha(RESUME_ARCHIVE)
            or plan.get("direction_child_sha256") != archive.get("raw_sha256")):
        raise ValueError("precondition-comparison starting receipt pins changed")
    return plan


def pseudo_huber_second_derivative(residual: Tensor, delta: float) -> Tensor:
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("pseudo-Huber scale must be finite and positive")
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("pseudo-Huber residual rows must be finite")
    delta_tensor = residual.new_tensor(delta)
    result = (delta_tensor / torch.hypot(delta_tensor, residual)).pow(3)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("pseudo-Huber second derivative is nonfinite")
    return result


def residual_jacobian_rows(
    residual_fn: Callable[[Tensor], Tensor], point: Tensor, *,
    on_row_start: Callable[[int, Tensor], None] | None = None,
    on_row_complete: Callable[[int, Tensor, Tensor], None] | None = None,
) -> tuple[Tensor, Tensor]:
    """Build residual rows with scalar pullbacks; FV outputs do not vmap."""
    result = torch.func.vjp(residual_fn, point)
    values = cast(Tensor, result[0])
    pullback = cast(Callable[[Tensor], tuple[Tensor, ...]], result[1])
    if values.ndim != 1 or values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("row Jacobian requires a nonempty finite vector residual")
    rows: list[Tensor] = []
    for index in range(values.numel()):
        if on_row_start is not None:
            on_row_start(index, values)
        cotangent = torch.zeros_like(values)
        cotangent[index] = 1.0
        row = pullback(cotangent)[0]
        if row.shape != point.shape or not bool(torch.isfinite(row).all()):
            raise ValueError("residual pullback row is malformed or nonfinite")
        if on_row_complete is not None:
            on_row_complete(index, values, row)
        rows.append(row)
    del pullback
    return values, torch.stack(rows)


def projected_gn_jacobi_diagonal(
    normal: Tensor, gminus: Tensor, gplus: Tensor, theta: float,
    residual: Tensor, jacobian_minus: Tensor, jacobian_plus: Tensor,
    control: Tensor, weights: Tensor, delta: float,
    *, prior_diagonal: Tensor | None = None, floor: float = PRIOR_FLOOR,
) -> tuple[Tensor, dict[str, Any]]:
    """Build diag(P H_GN P - mu P Q_cc P) without forming a Hessian."""
    size = control.numel()
    tensors = (normal, gminus, gplus, residual, jacobian_minus, jacobian_plus, control, weights)
    if (not all(bool(torch.isfinite(value).all()) for value in tensors)
            or not 0.0 <= theta <= 1.0 or not math.isfinite(floor) or floor <= 0
            or any(value.dtype != control.dtype or value.device != control.device for value in tensors)
            or normal.shape != (size,) or gminus.shape != (size,) or gplus.shape != (size,)
            or jacobian_minus.shape != (OBSERVATION_ROWS, size)
            or jacobian_plus.shape != (OBSERVATION_ROWS, size)
            or residual.shape != (OBSERVATION_ROWS,) or control.shape != (26,) or weights.shape != (5,)):
        raise ValueError("projected GN Jacobi products violate the fixed 12x26 FP64 contract")
    normal_norm = torch.linalg.vector_norm(normal)
    if not bool(torch.isfinite(normal_norm) & (normal_norm > torch.finfo(control.dtype).tiny)):
        raise ValueError("projected GN Jacobi normal is unresolved")
    unit_normal = normal / normal_norm
    projector = torch.eye(size, dtype=control.dtype, device=control.device) - torch.outer(unit_normal, unit_normal)
    robust_curvature = pseudo_huber_second_derivative(residual, delta)
    projected_minus = jacobian_minus @ projector
    projected_plus = jacobian_plus @ projector
    data_diagonal = ((1.0 - theta) * (robust_curvature[:, None] * projected_minus.square()).sum(dim=0)
        + theta * (robust_curvature[:, None] * projected_plus.square()).sum(dim=0))
    prior = torch.ones_like(control) if prior_diagonal is None else prior_diagonal
    if (prior.shape != (size,) or prior.dtype != control.dtype or prior.device != control.device
            or not bool(torch.isfinite(prior).all()) or not bool((prior > 0).all())):
        raise ValueError("original-prior diagonal must be finite, positive and control-shaped")
    prior_diagonal_projected = projector.square() @ prior
    flow = torch.tanh(control[20:25])
    face_curvature_diagonal = torch.zeros_like(control)
    face_curvature_diagonal[20:25] = -2.0 * weights * flow * (1.0 - flow.square())
    projected_face_diagonal = projector.square() @ face_curvature_diagonal
    mixed = (1.0 - theta) * gminus + theta * gplus
    mu = torch.dot(normal, mixed) / torch.dot(normal, normal)
    raw_diagonal = data_diagonal + prior_diagonal_projected - mu * projected_face_diagonal
    floored_diagonal = torch.maximum(raw_diagonal, raw_diagonal.new_full((size,), floor))
    inverse_diagonal = floored_diagonal.reciprocal()
    direction = -projector @ (inverse_diagonal * (projector @ mixed))
    if not all(bool(torch.isfinite(value).all()) for value in (
            robust_curvature, projector, data_diagonal, projected_face_diagonal,
            raw_diagonal, floored_diagonal, inverse_diagonal, direction)):
        raise ValueError("projected GN Jacobi diagonal or direction is nonfinite")
    diagnostics = {
        "mu": float(mu), "robust_curvature": robust_curvature.tolist(),
        "data_diagonal": data_diagonal.tolist(),
        "projected_prior_diagonal": prior_diagonal_projected.tolist(),
        "projected_face_curvature_diagonal": projected_face_diagonal.tolist(),
        "raw_diagonal": raw_diagonal.tolist(),
        "floored_diagonal": floored_diagonal.tolist(),
        "inverse_diagonal": inverse_diagonal.tolist(),
        "floor": floor, "floor_component_count": int(torch.count_nonzero(raw_diagonal < floor)),
        "minimum_raw_diagonal": float(raw_diagonal.min()),
        "maximum_raw_diagonal": float(raw_diagonal.max()),
        "projected_minus_row_norms": torch.linalg.vector_norm(projected_minus, dim=1).tolist(),
        "projected_plus_row_norms": torch.linalg.vector_norm(projected_plus, dim=1).tolist(),
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "direction_normal_residual": float(torch.dot(normal, direction)),
        "tangent_preconditioner_positive_on_nonzero_projected_gradient": bool(
            torch.dot(projector @ mixed, inverse_diagonal * (projector @ mixed)) > 0),
        "scope": "projected robust Gauss-Newton Jacobi search metric; not full-Hessian curvature",
    }
    return direction, diagnostics


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    """Load and close the archived 33cb endpoint without reconstructing FV outputs."""
    if _sha(RESUME_PLAN) != RESUME_PLAN_SHA:
        raise ValueError("accepted tangent-resume plan digest changed")
    prior = json.loads(RESUME_PLAN.read_text())
    archive = json.loads(RESUME_ARCHIVE.read_text())
    parent, resource = json.loads(BASE_RUN.read_text()), json.loads(BASE_RESOURCE.read_text())
    raw_bytes = gzip.decompress(BASE_ARCHIVE.read_bytes())
    child_sha = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    source_receipt = {**prior["source_files"], **prior["archive_files"],
        RESUME_PLAN.relative_to(ROOT).as_posix(): RESUME_PLAN_SHA}
    if (archive.get("raw_path") != "graphify-out/fv-root-cause-20260919/tangent_resume_20261009_attempt1/step.json"
            or archive.get("raw_sha256") != child_sha
            or archive.get("gzip_path") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or archive.get("gzip_sha256") != _sha(BASE_ARCHIVE)
            or archive.get("run_path") != BASE_RUN.relative_to(ROOT).as_posix()
            or archive.get("run_sha256") != _sha(BASE_RUN)
            or archive.get("resource_path") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or archive.get("resource_sha256") != _sha(BASE_RESOURCE)
            or archive.get("guarded_launch_count") != 1
            or archive.get("raw_bytes") != len(raw_bytes)
            or parent.get("child_sha256") != child_sha
            or parent.get("execution_status") != "completed"
            or parent.get("child_read_error") is not None
            or parent.get("resource") != resource
            or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != resume.POLICY["outer_seconds"]
            or resource.get("rss_limit_bytes") != resume.POLICY["rss_bytes"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > resume.POLICY["outer_seconds"]
            or resource.get("sampled_peak_rss_bytes", resume.POLICY["rss_bytes"]) >= resume.POLICY["rss_bytes"]
            or raw.get("plan_sha256") != RESUME_PLAN_SHA
            or raw.get("plan_hashes", {}).get("source_files") != prior["source_files"]
            or raw.get("plan_hashes", {}).get("archive_files") != prior["archive_files"]
            or raw.get("source_before") != source_receipt or raw.get("source_after") != source_receipt
            or raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("candidate_type") != "tangent_gradient_correction"
            or raw.get("accepted_iterations") != 3 or raw.get("optimizer_steps_applied") != 3
            or raw.get("hvp_calls_started") != 6 or raw.get("hvp_calls_completed") != 6
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True
            or raw.get("deadline_passed") is not True
            or raw.get("runtime") != raw.get("runtime_after")
            or not geometry.model._fixed_input(raw.get("input_after", {}),
                raw.get("input_before", {}), BASE_CONTROL_SHA)):
        raise ValueError("33cb accepted tangent-resume archive/input/resource closure is invalid")
    chain_sha, chain_theta, last, accepted = resume._validate_iteration_chain(
        raw.get("iterations", []), raw.get("hvp_history", []),
        resume.BASE_CONTROL_SHA, resume.BASE_THETA)
    closure = raw.get("last_confirmed_closure")
    try:
        accepted_gradients = {side: torch.as_tensor(accepted["side_gradients"][str(side)], dtype=torch.float64)
            for side in (-1, 1)}
        closed_gradients = {side: torch.as_tensor(closure["side_gradients"][str(side)], dtype=torch.float64)
            for side in (-1, 1)}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("33cb endpoint lacks both accepted/final-repeat side gradients") from error
    accepted_traces, closed_traces = accepted.get("branch_trace", {}), closure.get("branch_trace", {})
    if (chain_sha != BASE_CONTROL_SHA or chain_theta != BASE_THETA
            or closure != last.get("final_repeat")
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or raw.get("current_control") != accepted.get("control")
            or raw.get("current_theta") != BASE_THETA
            or not isinstance(closure, dict)
            or not all(closure.get(key) is True for key in (
                "native_objective_matches_proposal", "side_objectives_match_native",
                "merit_matches_proposal", "face_audit_passed", "branch_pair_passed",
                "trace_matches_proposal", "source_unchanged", "fixed_input_unchanged",
                "runtime_unchanged", "deadline_passed"))
            or not shared._gradient_pair_match(closed_gradients, accepted_gradients)
            or any(accepted_traces.get(str(side), {}).get("signature_sha256")
                != closed_traces.get(str(side), {}).get("signature_sha256") for side in (-1, 1))
            or not _close(float(accepted.get("objective", math.nan)), BASE_OBJECTIVE, torch.float64)
            or not _close(float(accepted.get("F_squared", math.nan)), BASE_F_SQUARED, torch.float64)):
        raise ValueError("33cb current control/theta/J/F² or independent final repeat changed")
    control = torch.as_tensor(raw["current_control"], dtype=torch.float64)
    return {"raw": raw, "control": control, "theta": BASE_THETA,
        "objective": BASE_OBJECTIVE, "accepted": accepted, "child_sha256": child_sha,
        "input_identity": raw["input_after"], "runtime": raw["runtime_after"]}


def _counted_residual_rows(context: dict[str, Any], control: Tensor,
                           sign: int, theta: float) -> tuple[Tensor, Tensor]:
    problem, parameters = context["problem"], context["parameters"]
    record, output = context["record"], context["output"]
    deadline, plan = context["deadline"], context["plan"]
    def residual_fn(value: Tensor) -> Tensor:
        return problem.observation_residual(value, parameters).reshape(-1)

    with transport.selected_face_extension(FACE["axis"], FACE["row"], FACE["column"], sign):
        shared._deadline(deadline, float(plan["policy"]["internal_seconds"]))
        def before_row(row_index: int, residual: Tensor) -> None:
            shared._deadline(deadline, float(plan["policy"]["internal_seconds"]))
            limit = int(plan["policy"]["jacobian_row_vjp_calls"])
            if record["jacobian_rows_started"] >= limit:
                raise ValueError("plan Jacobian-row VJP limit reached")
            receipt: dict[str, Any] = {"side": sign, "row": row_index,
                "base_control_sha256": _tensor_sha(control), "theta": theta,
                "status": "started"}
            record["jacobian_rows_started"] += 1
            record["jacobian_row_history"].append(receipt)
            record["current_jacobian_row"] = receipt
            tangent._write(output, record)

        def after_row(row_index: int, residual: Tensor, gradient: Tensor) -> None:
            record["jacobian_rows_completed"] += 1
            receipt = record["jacobian_row_history"][-1]
            receipt.update(status="completed", residual_value=float(residual[row_index]),
                gradient=gradient.tolist())
            tangent._write(output, record)
            shared._deadline(deadline, float(plan["policy"]["internal_seconds"]))

        values, rows = residual_jacobian_rows(residual_fn, control,
            on_row_start=before_row, on_row_complete=after_row)
        if values.shape != (OBSERVATION_ROWS,):
            raise ValueError("current whitened point residual does not have twelve rows")
    shared._deadline(deadline, float(plan["policy"]["internal_seconds"]))
    return values.detach(), rows


def _direction_model_factory(context: dict[str, Any]):
    def build(control: Tensor, theta: float, gminus: Tensor, gplus: Tensor,
              normal: Tensor, chart_jacobian: Tensor, pivot: int) -> list[dict[str, Any]]:
        problem = context["problem"]
        record = context["record"]
        frozen = problem.frozen
        observed = context["model_state"].get("observed")
        if (observed is None or not observed["pair"]["passed"]
                or problem.layout["controls"] != 26 or problem.layout["parameters"] != 13
                or tuple(problem.observation_dbz.shape) != (3, 4)
                or problem.observation_status is not None
                or not bool(problem._detected_mask.all())
                or frozen.neural_prior_std_dbz is not None
                or frozen.neural_prior_valid_mask is not None
                or frozen.neural_prior_dependency is not None
                or frozen.analysis_config.field_smoothness_weight != 0.0):
            raise ValueError("preconditioned comparison requires the fixed 12-row plain-prior point profile")
        parameters = context["parameters"]
        contract = problem.contract(parameters)
        prior_residual = v._control_prior_residual(control, contract)
        if not torch.equal(prior_residual, control):
            raise ValueError("fixed profile no longer has the identity control-prior residual")
        residual_minus, rows_minus = _counted_residual_rows(context, control, -1, theta)
        residual_plus, rows_plus = _counted_residual_rows(context, control, 1, theta)
        residual_pair_budget, residual_pair_scale = _residual_pair_roundoff_budget(
            problem, parameters, residual_minus, residual_plus)
        residual_errors = (residual_minus - residual_plus).abs().reshape_as(residual_pair_budget)
        residual_error = float(residual_errors.max())
        if bool((residual_errors > residual_pair_budget).any()):
            raise ValueError("the two selected-face whitened residual vectors do not agree")
        projector = torch.eye(control.numel(), dtype=control.dtype, device=control.device)
        unit_normal = normal / torch.linalg.vector_norm(normal)
        projector = projector - torch.outer(unit_normal, unit_normal)
        row_error, row_budget, ambient_row_norms_minus, ambient_row_norms_plus, projected_row_norms = (
            _projected_row_parity(rows_minus, rows_plus, projector))
        if bool((row_error > row_budget).any()):
            row_index = int(torch.argmax(row_error / row_budget))
            raise ValueError(f"projected side residual Jacobian row {row_index} differs across the face")
        projected_row_norms_minus, projected_row_norms_plus = projected_row_norms

        delta = float(contract.analysis_config.pseudo_huber_delta)
        robust_cost = v._pseudo_huber_cost(residual_minus, delta).sum()
        smooth_cost = v._field_smoothness_prior_cost(control, contract)
        reconstructed_objective = robust_cost + 0.5 * torch.dot(prior_residual, prior_residual) + smooth_cost
        native_j = float(observed["native_j"])
        side_objectives = {side: float(observed["side"][side][0]) for side in (-1, 1)}
        if (not _close(float(reconstructed_objective), native_j, control.dtype)
                or any(not _close(float(reconstructed_objective), value, control.dtype)
                    for value in side_objectives.values())
                or float(smooth_cost) != 0.0):
            raise ValueError("whitened row data cost and fixed regularizer reconstruction do not match J")
        mixed = (1.0 - theta) * gminus + theta * gplus
        baseline = tangent.tangent_direction(gminus, gplus, normal,
            chart_jacobian, theta, pivot=pivot)[2]
        euclidean_baseline = -projector @ mixed
        baseline_error = float(torch.linalg.vector_norm(baseline - euclidean_baseline))
        baseline_budget = 128 * torch.finfo(control.dtype).eps * max(
            float(torch.linalg.vector_norm(baseline)),
            float(torch.linalg.vector_norm(euclidean_baseline)), torch.finfo(control.dtype).tiny)
        if baseline_error > baseline_budget:
            raise ValueError("existing retained-chart direction differs from the projected baseline")
        direction, diagnostics = projected_gn_jacobi_diagonal(normal, gminus, gplus,
            theta, residual_minus, rows_minus, rows_plus, control, context["weights"],
            delta, floor=PRIOR_FLOOR)
        diagnostics.update({"residual_minus": residual_minus.tolist(),
            "residual_plus": residual_plus.tolist(),
            "residual_pair_max_error": residual_error,
            "residual_pair_budget": residual_pair_budget.tolist(),
            "residual_pair_construction_scale": residual_pair_scale.tolist(),
            "projected_row_max_errors": row_error.tolist(),
            "projected_row_budgets": row_budget.tolist(),
            "ambient_minus_row_norms": ambient_row_norms_minus.tolist(),
            "ambient_plus_row_norms": ambient_row_norms_plus.tolist(),
            "projected_minus_row_norms": projected_row_norms_minus.tolist(),
            "projected_plus_row_norms": projected_row_norms_plus.tolist(),
            "reconstructed_objective": float(reconstructed_objective),
            "side_objectives": {str(side): side_objectives[side] for side in (-1, 1)},
            "baseline_chart_projection_error": baseline_error,
            "baseline_chart_projection_budget": baseline_budget,
            "jacobian_row_vjp_calls": record["jacobian_rows_completed"]})
        return [
            {"name": "baseline_tangent", "direction": baseline,
                "direction_override": None,
                "diagnostics": {"scope": "existing projected tangent-gradient baseline"}},
            {"name": "robust_gn_jacobi", "direction": direction,
                "direction_override": direction, "diagnostics": diagnostics},
        ]
    return build


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child_impl(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA,
        direction_model_factory=_direction_model_factory)


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA,
        direction_model_factory=_direction_model_factory, initial_theta=BASE_THETA)


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    return tangent.run(plan_path, plan_sha, output, resource, log,
        plan_loader=_load_plan, child_script=Path(__file__),
        base_control_sha256=BASE_CONTROL_SHA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path,
        default=EVIDENCE / "tangent_precondition_comparison_20261009_attempt1/step.json")
    parser.add_argument("--resource", type=Path,
        default=EVIDENCE / "tangent_precondition_comparison_20261009_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path,
        default=EVIDENCE / "tangent_precondition_comparison_20261009_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


