"""One bounded full-space Newton correction at a selected FV upwind kink.

The probe solves the coupled stationarity/face equations using both fixed
one-sided smooth extensions of the original objective. It can report one
accepted correction or a refusal; neither outcome certifies a minimum or a
smooth response.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as face_geometry
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "NONSMOOTH_COUPLED_FACE_PLAN_20261009.json"
PRODUCER_PLAN = EVIDENCE / "CORRECTION_RESUME_PLAN_20261008.json"
PRODUCER_PLAN_SHA = "9c29f4c0882272dccd68e2a35abf9a1fe9ff5b2fc5d5adc1502fe9a6ee8e9130"
BASE_DIR = EVIDENCE / "correction_resume_20261008_attempt1"
BASE_STEP = BASE_DIR / "step.json"
BASE_RUN = BASE_DIR / "step.run.json"
BASE_RESOURCE = BASE_DIR / "step.resource.json"
BASE_STEP_SHA = "7bd91fa946b2c4f63380fb5df5dc2c47b4d2bfb2942ee5a5f579e703a419d83b"
BASE_RUN_SHA = "bc8da6c11d0a210154290904d585a521328bde08c7dac7a4c76e2599ea759322"
BASE_RESOURCE_SHA = "2db8c1712600f89a0a9f84efa2ef1b77ff0ee9e0b4de1c28b273111451c812f4"
BASE_CONTROL_SHA = "95a55578eea1d3b9600a210e7bdba37c21d7d7a1baf07244afc513a46df8c842"
SOURCE_MANIFEST = EVIDENCE / "kink_source_20261009/manifest.json"
FACE: dict[str, Any] = {"axis": "y", "row": 4, "column": 3}
SELF = "examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py"
TEST = "tests/test_fv_point_3h_nonsmooth_coupled_probe.py"
CORE_TEST = "tests/test_fv_selected_face_extension.py"
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 600.0, 660.0, 1024**3
MAX_HVP = 56
MAX_CANDIDATES = 8
RADIUS = 0.05
J_C1 = 1.0e-4
F_C1 = 1.0e-4
Q_ROUNDOFF = 128
FACE_PARITY_ETA = 1.0e-8
ANALYSIS_STAGES = 360


def policy() -> dict[str, Any]:
    return {
        "guarded_launches": 1, "internal_seconds": INTERNAL_SECONDS,
        "outer_seconds": WALL_SECONDS, "rss_bytes": RSS_BYTES,
        "hvp_calls": MAX_HVP, "basis_hvps_per_side": 26,
        "parity_hvps": 4, "dense_solves": 1,
        "max_candidates": MAX_CANDIDATES, "radius": RADIUS,
        "optimizer_steps": 1, "pcg_solves": 0,
        "gradient_scale": 1.0, "face_constraint_scale": "max_abs_face_weight",
        "root_claim": False, "minimum_claim": False,
        "response_claim": False,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


_face_weights = face_geometry._face_weights
_face_normal = face_geometry._face_normal
_chart = face_geometry._chart
_chart_jacobian = face_geometry._chart_jacobian


def _face_value(control: Tensor, weights: Tensor) -> Tensor:
    return torch.dot(weights, torch.tanh(control[20:25]))


def _production_face_value(problem: Any, control: Tensor, axis: str,
                           row: int, column: int) -> Tensor:
    qx, qy = face_geometry.qy.production_fluxes(problem, control)
    return qx[row, column] if axis == "x" else qy[row, column]


def _flow_cancellation_scale(control: Tensor, weights: Tensor) -> Tensor:
    return torch.sum((weights * torch.tanh(control[20:25])).abs())


def _face_audit(problem: Any, control: Tensor, weights: Tensor,
                face: dict[str, Any], eta: float) -> tuple[Tensor, float, bool]:
    try:
        actual = control.new_tensor(face_geometry._verify_face_value(
            problem, control, weights, face, eta))
        verified = True
    except ValueError:
        actual = _production_face_value(problem, control, **face)
        verified = False
    formula = _face_value(control, weights)
    scale = (weights * torch.tanh(control[20:25])).abs().sum()
    budget = 128 * torch.finfo(control.dtype).eps * torch.stack((
        scale, actual.abs(), formula.abs(), actual.new_tensor(abs(eta)),
        actual.new_tensor(torch.finfo(control.dtype).tiny))).max()
    passed = (verified and bool(torch.isfinite(actual) & torch.isfinite(formula))
        and abs(float(actual - formula)) <= float(budget)
        and abs(float(actual) - eta) <= float(budget))
    return actual, float(budget), passed


def _face_sign_copy(value: Tensor, axis: str,
                    target: dict[str, Any] = FACE) -> list[Any]:
    signs = torch.sign(value).to(torch.int8)
    result = signs.tolist()
    if axis == target["axis"]:
        result[target["row"]][target["column"]] = None
    return result


def _pivot(control: Tensor, weights: Tensor) -> int:
    flow_normal = _face_normal(control, weights)[20:25]
    index = int(torch.argmax(flow_normal.abs()))
    if float(flow_normal[index].abs()) <= 128 * torch.finfo(control.dtype).eps:
        raise ValueError("selected-face chart has no resolved pivot derivative")
    return 20 + index


def _analysis_trace(problem: Any, control: Tensor, parameters: Tensor,
                    side: int, *, target: dict[str, Any] = FACE) -> dict[str, Any]:
    """Observe the 360 analysis stages, omitting the selected flux from gates."""
    expected = ANALYSIS_STAGES
    stages: list[dict[str, Any]] = []
    observed = 0
    nonfinite = False
    minima: dict[str, float | None] = {
        "x_slope_abs_over_q": None, "y_slope_abs_over_q": None,
        "active_limiter_gap_over_q": None,
        "x_other_face_flux_abs_over_scale": None, "y_other_face_flux_abs_over_scale": None,
    }
    max_flux = 0.0

    def observe(q: Tensor, qx: Tensor, qy: Tensor) -> None:
        nonlocal nonfinite, max_flux, observed
        observed += 1
        if observed > expected:
            nonfinite = True
            return
        q_scale = q.abs().max()
        if not bool(torch.isfinite(q_scale)) or not bool(q_scale > 0):
            nonfinite = True
            return
        tolerance = 128 * torch.finfo(q.dtype).eps * q_scale
        pairs = ((q[1:-1, 1:-1] - q[1:-1, :-2], q[1:-1, 2:] - q[1:-1, 1:-1]),
                 (q[1:-1, 1:-1] - q[:-2, 1:-1], q[2:, 1:-1] - q[1:-1, 1:-1]))
        choices = []
        for orientation, (left, right) in zip(("x", "y"), pairs):
            if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
                nonfinite = True
                continue
            active = (((left > tolerance) & (right > tolerance))
                      | ((left < -tolerance) & (right < -tolerance)))
            gap = (left - right).abs()
            for name, values, mask in ((f"{orientation}_slope_abs_over_q", left.abs(), None),
                                       (f"{orientation}_slope_abs_over_q", right.abs(), None),
                                       ("active_limiter_gap_over_q", gap, active)):
                selected = values if mask is None else values[mask]
                if selected.numel():
                    value = float(selected.min() / q_scale)
                    old = minima[name]
                    minima[name] = value if old is None else min(old, value)
            if (bool((left.abs() <= tolerance).any()) or bool((right.abs() <= tolerance).any())
                    or bool((active & (gap <= tolerance)).any())):
                nonfinite = True
            positive, negative = (left > tolerance) & (right > tolerance), (left < -tolerance) & (right < -tolerance)
            choices.append({
                "choose_left": (active & torch.where(positive, left < right, left > right)).tolist(),
                "slope_sign": torch.where(positive, 1, torch.where(negative, -1, 0)).tolist(),
                "left_sign": torch.sign(left).to(torch.int8).tolist(),
                "right_sign": torch.sign(right).to(torch.int8).tolist(),
            })
        fluxes = {"x": qx, "y": qy}
        signatures: dict[str, Any] = {}
        selected_axis = target["axis"]
        for axis, value in fluxes.items():
            if not bool(torch.isfinite(value).all()):
                nonfinite = True
            scale = value.abs().max()
            max_flux = max(max_flux, float(scale))
            mask = torch.ones_like(value, dtype=torch.bool)
            if axis == selected_axis:
                mask[target["row"], target["column"]] = False
            other = value[mask]
            if other.numel():
                if not bool(torch.isfinite(scale)) or not bool(scale > 0):
                    nonfinite = True
                else:
                    minimum = float(other.abs().min() / scale)
                    key = f"{axis}_other_face_flux_abs_over_scale"
                    old = minima[key]
                    minima[key] = minimum if old is None else min(old, minimum)
            if bool((other.abs() <= 128 * torch.finfo(value.dtype).eps * scale).any()):
                nonfinite = True
            # This copy-only mask never touches the production flux tensor.
            signatures[axis] = _face_sign_copy(value, axis, target)
        stages.append({"choices": choices, "face_signs": signatures,
                       "target_flux": float(fluxes[selected_axis][target["row"], target["column"]])})

    with transport.selected_face_extension(target["axis"], target["row"], target["column"], side):
        with torch.no_grad(), transport.observe_minmod_stages(observe):
            problem.objective(control, parameters)
    if observed != expected or len(stages) != expected:
        nonfinite = True
    return {"stage_count": len(stages), "observed_stage_count": observed,
            "expected_stage_count": expected,
            "choices": [item["choices"] for item in stages],
            "face_signs": [item["face_signs"] for item in stages],
            "target_fluxes": [item["target_flux"] for item in stages],
            "minimum_margins": minima, "maximum_face_flux": max_flux,
            "excluded_face_from_sign_gate": {**target, "diagnostic_copy_only": True},
            "nonfinite_or_tie": nonfinite,
            "signature_sha256": _canonical_sha({"choices": [item["choices"] for item in stages],
                "face_signs": [item["face_signs"] for item in stages]}),
            "side": side}


def _trace_gate(minus: dict[str, Any], plus: dict[str, Any], *, dtype: torch.dtype) -> dict[str, Any]:
    eps = torch.finfo(dtype).eps
    near = 128 * eps * max(float(minus["maximum_face_flux"]), float(plus["maximum_face_flux"]), 1e-300)
    target_max = max(max(map(abs, minus["target_fluxes"]), default=0.0),
                     max(map(abs, plus["target_fluxes"]), default=0.0))
    same_other = (minus["choices"] == plus["choices"] and minus["face_signs"] == plus["face_signs"])
    passed = (minus["stage_count"] == plus["stage_count"] == 360
              and not minus["nonfinite_or_tie"] and not plus["nonfinite_or_tie"]
              and same_other and target_max <= near)
    return {"passed": passed, "target_flux_bound": near,
            "target_flux_max_abs": target_max, "same_other_branch_signature": same_other}


def _fresh_native(problem: Any, control: Tensor, parameters: Tensor) -> tuple[Tensor, dict[str, Any]]:
    objective = problem.objective(control, parameters)
    branch, scope = problem.branch_check(control, parameters)
    signature_sha = _canonical_sha({"choices": branch.get("choices"),
                                    "face_signs": branch.get("face_signs")})
    return objective, {"signature": branch, "signature_sha256": signature_sha, "scope": scope}


def _min_norm_mix(gminus: Tensor, gplus: Tensor) -> tuple[float, Tensor, dict[str, Any]]:
    delta = gplus - gminus
    denominator = torch.dot(delta, delta)
    raw = (torch.zeros((), dtype=gminus.dtype) if not bool(denominator > 0)
           else -torch.dot(gminus, delta) / denominator)
    theta = float(raw.clamp(0, 1))
    mixed = (1 - theta) * gminus + theta * gplus
    return theta, mixed, {"unclipped_theta": float(raw), "theta": theta,
        "endpoint_norms": [float(torch.linalg.vector_norm(gminus)), float(torch.linalg.vector_norm(gplus))],
        "mixed_norm": float(torch.linalg.vector_norm(mixed))}


def _coupled_matrix(hminus: Tensor, hplus: Tensor, gminus: Tensor,
                    gplus: Tensor, normal: Tensor, q: Tensor,
                    theta: float, qscale: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    hm = (1 - theta) * hminus + theta * hplus
    size = gminus.numel()
    matrix = hm.new_zeros((size + 1, size + 1))
    matrix[:size, :size] = hm
    matrix[:size, size] = gplus - gminus
    matrix[size, :size] = normal / qscale
    residual = torch.cat(((1 - theta) * gminus + theta * gplus, (q / qscale).reshape(1)))
    return matrix, residual, hm


def _solve_coupled(matrix: Tensor, residual: Tensor) -> tuple[Tensor, dict[str, Any]]:
    if matrix.shape != (27, 27) or residual.shape != (27,):
        raise ValueError("coupled step requires the full 27-unknown KKT-like system")
    delta = torch.linalg.solve(matrix, -residual)
    remainder = matrix @ delta + residual
    denominator = torch.linalg.vector_norm(matrix) * torch.linalg.vector_norm(delta) + torch.linalg.vector_norm(residual)
    relative = torch.linalg.vector_norm(remainder) / denominator.clamp_min(torch.finfo(matrix.dtype).tiny)
    if not bool(torch.isfinite(delta).all() & torch.isfinite(relative)):
        raise ValueError("dense coupled solve returned a nonfinite result")
    return delta, {"scaled_residual_relative": float(relative), "passed": bool(relative <= 1e-10),
                   "method": "torch.linalg.solve dense pivoted LU", "unknowns": 27}


def _diagnostic_curvature(hmix: Tensor, mixed_gradient: Tensor, control: Tensor,
                          weights: Tensor, normal: Tensor) -> dict[str, Any]:
    flow = torch.tanh(control[20:25])
    qh = torch.zeros_like(hmix)
    qh[20:25, 20:25] = torch.diag(-2 * weights * flow * (1 - flow.square()))
    tangent = torch.linalg.qr(normal[:, None], mode="complete").Q[:, 1:]
    nn = torch.dot(normal, normal)
    lam = torch.dot(normal, mixed_gradient) / nn
    projected = tangent.T @ (hmix - lam * qh) @ tangent
    eigenvalues = torch.linalg.eigvalsh(projected)
    return {"scope": "base-point diagnostic only; not a candidate curvature certificate",
            "lagrange_multiplier": float(lam), "minimum_tangent_eigenvalue": float(eigenvalues.min()),
            "maximum_tangent_eigenvalue": float(eigenvalues.max()),
            "tangent_eigenvalues": eigenvalues.tolist()}


def _path_direction(control: Tensor, weights: Tensor, pivot: int,
                    delta: Tensor) -> tuple[Tensor, Tensor]:
    retained = [i for i in range(control.numel()) if i != pivot]
    _, z, _ = _chart_jacobian(control, weights, pivot, 0.0)
    direction = z @ delta[retained]
    return direction, direction[retained]


def _residual_measure(gradient: Tensor, q: Tensor, qscale: Tensor) -> Tensor:
    return torch.cat((gradient, (q / qscale).reshape(1)))


def _candidate_path(problem: Any, base: Tensor, weights: Tensor, pivot: int,
                    eta0: float, delta: Tensor, alpha: float) -> tuple[Tensor, float]:
    # The final unknown is theta, the convex weight between side gradients.
    # The candidate remains on the selected zero-face chart.
    candidate = _chart(base + alpha * delta[:26], weights, pivot, eta0)
    return candidate, eta0


def _source_maps_valid(plan: dict[str, Any]) -> None:
    inherited = json.loads(PRODUCER_PLAN.read_text())
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    expected_sources = set(inherited["source_files"]) | {SELF, TEST, CORE_TEST}
    expected_archives = set(inherited["archive_files"]) | {
        PRODUCER_PLAN.relative_to(ROOT).as_posix(), BASE_STEP.relative_to(ROOT).as_posix(),
        BASE_RUN.relative_to(ROOT).as_posix(), BASE_RESOURCE.relative_to(ROOT).as_posix(),
        SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
    }
    snapshots = plan.get("producer_source_snapshots")
    if (not isinstance(sources, dict) or not isinstance(archives, dict)
            or set(sources) != expected_sources or len(sources) != 122
            or len(archives) != 78 or not isinstance(snapshots, dict)
            or set(snapshots) != {"src/advar/transport.py"}
            or set(archives) != expected_archives | {snapshots["src/advar/transport.py"].get("archive_path")}
            or not SOURCE_MANIFEST.is_file()):
        raise ValueError("nonsmooth plan source/archive scope changed")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if snapshots != manifest.get("snapshots"):
        raise ValueError("transport source snapshot map differs from immutable manifest")
    for name, value in inherited["source_files"].items():
        if name in snapshots:
            snap = snapshots[name]
            if (snap.get("sha256") != value or archives.get(snap.get("archive_path")) != value
                    or sources.get(name) != _sha(ROOT / name)):
                raise ValueError("transport snapshot does not preserve inherited producer bytes")
        elif sources.get(name) != value:
            raise ValueError(f"inherited source pin changed: {name}")
    if any(archives.get(name) != value for name, value in inherited["archive_files"].items()):
        raise ValueError("inherited archive pin changed")
    if any(sources.get(name) != _sha(ROOT / name) for name in (SELF, TEST, CORE_TEST)):
        raise ValueError("current probe or transport contract tests differ from plan pins")
    if archives.get(PRODUCER_PLAN.relative_to(ROOT).as_posix()) != PRODUCER_PLAN_SHA:
        raise ValueError("plan does not archive the frozen correction producer")
    if archives.get(SOURCE_MANIFEST.relative_to(ROOT).as_posix()) != _sha(SOURCE_MANIFEST):
        raise ValueError("plan does not pin its immutable transport source manifest")
    for name, value in {**sources, **archives}.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or path.is_symlink() or _sha(path) != value:
            raise ValueError(f"source/archive pin mismatch: {name}")


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("nonsmooth coupled-face plan must not be a symbolic link")
    path = path.resolve()
    if path != PLAN.resolve() or path.is_symlink() or _sha(path) != digest:
        raise ValueError("nonsmooth coupled-face plan identity mismatch")
    plan = json.loads(path.read_text())
    if (plan.get("experiment_kind") != "nonsmooth_coupled_face_step"
            or plan.get("policy") != policy()
            or plan.get("producing_plan") != PRODUCER_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != PRODUCER_PLAN_SHA
            or plan.get("base_step") != BASE_STEP.relative_to(ROOT).as_posix()
            or plan.get("base_run") != BASE_RUN.relative_to(ROOT).as_posix()
            or plan.get("base_resource") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("face") != FACE
            or _sha(PRODUCER_PLAN) != PRODUCER_PLAN_SHA):
        raise ValueError("nonsmooth plan policy, face, or producer lineage changed")
    _source_maps_valid(plan)
    for artifact, known in ((BASE_STEP, BASE_STEP_SHA), (BASE_RUN, BASE_RUN_SHA),
                            (BASE_RESOURCE, BASE_RESOURCE_SHA)):
        if _sha(artifact) != known or plan["archive_files"].get(artifact.relative_to(ROOT).as_posix()) != known:
            raise ValueError("accepted correction receipt hash differs from the plan")
    if (plan.get("base_step_sha256") != BASE_STEP_SHA or plan.get("base_run_sha256") != BASE_RUN_SHA
            or plan.get("base_resource_sha256") != BASE_RESOURCE_SHA):
        raise ValueError("nonsmooth plan omits the accepted correction receipt pins")
    return plan


def _load_base(plan: dict[str, Any]) -> dict[str, Any]:
    producer = json.loads(PRODUCER_PLAN.read_text())
    expected_sources = {**producer["source_files"], **producer["archive_files"],
        PRODUCER_PLAN.relative_to(ROOT).as_posix(): PRODUCER_PLAN_SHA}
    raw, parent, resource = (json.loads(BASE_STEP.read_text()), json.loads(BASE_RUN.read_text()),
                             json.loads(BASE_RESOURCE.read_text()))
    state, iterations = raw.get("current_state"), raw.get("iterations")
    if not isinstance(state, dict) or not isinstance(iterations, list):
        raise ValueError("accepted correction receipt has no closed endpoint")
    expected_resource = {"exit_code": 0, "resource_termination": None, "monitor_error": None,
        "received_sigterm": False, "elapsed_seconds": resource.get("elapsed_seconds"),
        "wall_limit_seconds": 300.0, "rss_limit_bytes": RSS_BYTES}
    resource_ok = all(resource.get(k) == v for k, v in expected_resource.items())
    resource_ok = resource_ok and isinstance(resource.get("sampled_peak_rss_bytes"), (int, float)) \
        and resource["sampled_peak_rss_bytes"] < RSS_BYTES \
        and 0 < float(resource.get("elapsed_seconds", 0)) <= 300
    input_before, input_after = raw.get("input_before"), raw.get("input_after")
    fixed_keys = ("parameters_sha256", "terminal_truth_sha256", "archived_input")
    input_closed = (isinstance(input_before, dict) and isinstance(input_after, dict)
        and all(input_before.get(key) == input_after.get(key) for key in fixed_keys)
        and input_after.get("control_sha256") == BASE_CONTROL_SHA)
    sequence_closed = (len(iterations) == 3 and all(
        it.get("index") == index and it.get("committed") is True
        and it.get("hvp_calls_started") == 1 and it.get("hvp_calls_completed") == 1
        and (index == 0 or it.get("base_control_sha256") == iterations[index - 1].get("accepted_control_sha256"))
        and sum(trial.get("status") == "accepted" for trial in it.get("trials", [])) == 1
        and next(trial for trial in it.get("trials", []) if trial.get("status") == "accepted").get("control_sha256")
            == it.get("accepted_control_sha256")
        for index, it in enumerate(iterations)))
    final_iteration = iterations[-1] if iterations else {}
    final_trial = next((trial for trial in final_iteration.get("trials", [])
                        if trial.get("status") == "accepted"), {})
    if (raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "max_iterations_completed"
            or raw.get("plan_sha256") != PRODUCER_PLAN_SHA
            or raw.get("base_control_sha256") != producer.get("base_control_sha256")
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or raw.get("optimizer_steps_applied") != 3 or raw.get("hvp_calls") != 3
            or raw.get("hvp_calls_completed") != 3 or raw.get("candidate_committed") is not True
            or raw.get("active_candidate_committed") is not False
            or raw.get("fixed_input_unchanged") is not True or not input_closed
            or raw.get("runtime_after") != raw.get("runtime")
            or raw.get("source_unchanged") is not True or raw.get("source_before") != raw.get("source_after")
            or raw.get("source_before") != expected_sources
            or iterations[0].get("base_control_sha256") != raw.get("base_control_sha256")
            or not sequence_closed
            or state.get("control_sha256") != BASE_CONTROL_SHA
            or len(iterations) != 3
            or final_iteration.get("accepted_control") != raw.get("current_control")
            or final_trial.get("objective") != state.get("objective")
            or final_trial.get("gradient") != state.get("gradient")
            or final_trial.get("phi") != state.get("phi")
            or final_trial.get("branch", {}).get("signature_sha256")
                != state.get("branch", {}).get("signature_sha256")
            or iterations[-1].get("accepted_control_sha256") != BASE_CONTROL_SHA
            or parent.get("execution_status") != "completed"
            or parent.get("child_read_error") is not None
            or parent.get("child_sha256") != BASE_STEP_SHA
            or parent.get("numerical_status") != raw.get("numerical_status")
            or parent.get("resource") != resource or not resource_ok
            or _tensor_sha(torch.tensor(raw.get("current_control"), dtype=torch.float64)) != BASE_CONTROL_SHA):
        raise ValueError("PR26495a accepted endpoint or closed receipts are invalid")
    return {"raw": raw, "state": state, "control": torch.tensor(raw["current_control"], dtype=torch.float64),
            "expected_objective": state["objective"], "expected_gradient": torch.tensor(state["gradient"], dtype=torch.float64),
            "runtime": raw["runtime"], "input_identity": raw["input_after"]}


def _input_identity(problem: Any, original: Tensor, control: Tensor,
                    parameters: Tensor, truth: Tensor) -> dict[str, Any]:
    return seed._input_identity(problem, original, control, parameters, truth)


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("600-second internal deadline reached")


def _counted_hvp(record: dict[str, Any], output: Path, deadline: float,
                 label: dict[str, Any], call: Any) -> Tensor:
    _deadline(deadline)
    record["hvp_calls_started"] = int(record.get("hvp_calls_started", 0)) + 1
    record["current_hvp"] = {**label, "status": "started"}
    _write(output, record)
    result = call()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("HVP returned a nonfinite component")
    record["hvp_calls_completed"] = int(record.get("hvp_calls_completed", 0)) + 1
    record["current_hvp"] = {**label, "status": "completed"}
    record.setdefault("hvp_history", []).append(record["current_hvp"])
    _write(output, record)
    _deadline(deadline)
    return result


def _runtime() -> dict[str, str]:
    return face_geometry.guard_policy.blocks.runtime_identity()


def _source_hashes(plan: dict[str, Any], plan_path: Path, plan_sha: str) -> dict[str, str]:
    names = set(plan["source_files"]) | set(plan["archive_files"])
    names.add(plan_path.relative_to(ROOT).as_posix())
    return {name: _sha(ROOT / name) for name in sorted(names)}


def _finish_without_commit(record: dict[str, Any], *, output: Path, problem: Any,
                           original: Tensor, original_control: Tensor, parameters: Tensor,
                           truth: Tensor, input_before: dict[str, Any], runtime_before: dict[str, Any],
                           source_before: dict[str, str], plan: dict[str, Any],
                           plan_path: Path, plan_sha: str, deadline: float) -> dict[str, Any]:
    after = _input_identity(problem, original, original_control, parameters, truth)
    source_after = _source_hashes(plan, plan_path, plan_sha)
    fixed_ok = face_geometry.model._fixed_input(after, input_before, _tensor_sha(original_control))
    runtime_after = _runtime()
    source_ok = source_before == source_after
    record.update(phase="finished", execution_status="completed",
        optimizer_steps_applied=0, candidate_committed=False, active_candidate_committed=False,
        current_control=original_control.tolist(), current_control_sha256=_tensor_sha(original_control),
        input_before=input_before, input_after=after, fixed_input_unchanged=bool(fixed_ok),
        runtime=runtime_before, runtime_after=runtime_after, runtime_unchanged=runtime_before == runtime_after,
        source_before=source_before, source_after=source_after, source_unchanged=source_ok,
        deadline_passed=time.monotonic() < deadline)
    if not all((fixed_ok, runtime_before == runtime_after, source_ok)):
        record.update(execution_status="failed", numerical_status="diagnostic_closure_failed",
            refusal="source, fixed-input, or runtime closure changed during refusal")
    _write(output, record)
    return record


def _eval_side(problem: Any, control: Tensor, parameters: Tensor, side: int,
               hvp_direction: Tensor | None = None, *, hvp_call: Any = None
               ) -> tuple[Tensor, Tensor, Tensor | None]:
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    with transport.selected_face_extension(FACE["axis"], FACE["row"], FACE["column"], side):
        objective = problem.objective(control, parameters)
        gradient = gradient_fn(control, parameters)
        hvp = None
        if hvp_direction is not None:
            call = lambda: torch.func.jvp(lambda value: gradient_fn(value, parameters),
                (control,), (hvp_direction,))[1]
            hvp = call() if hvp_call is None else hvp_call(call)
    return objective, gradient, hvp


def _eval_native(problem: Any, control: Tensor, parameters: Tensor,
                 hvp_direction: Tensor | None = None, *, hvp_call: Any = None
                 ) -> tuple[Tensor, Tensor, Tensor | None]:
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    objective = problem.objective(control, parameters)
    gradient = gradient_fn(control, parameters)
    call = lambda: torch.func.jvp(lambda value: gradient_fn(value, parameters),
        (control,), (hvp_direction,))[1]
    hvp = (None if hvp_direction is None else call() if hvp_call is None else hvp_call(call))
    return objective, gradient, hvp


def _parity_gate(native: tuple[Tensor, Tensor, Tensor | None], extended: tuple[Tensor, Tensor, Tensor | None],
                 *, eps: float = 2e-9) -> dict[str, Any]:
    errors = []
    for a, b in zip(native, extended):
        if a is None or b is None:
            raise ValueError("parity requires completed native and extension products")
        scale = max(float(a.norm()), float(b.norm()), torch.finfo(a.dtype).tiny)
        errors.append(float((a - b).norm()) / scale)
    return {"relative_errors": errors, "tolerance": eps,
            "passed": all(math.isfinite(v) and v <= eps for v in errors)}


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path,
                    closure_state: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()
    deadline = start + INTERNAL_SECONDS
    plan = _load_plan(plan_path, plan_sha)
    base = _load_base(plan)
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "preflight", "plan_sha256": plan_sha,
        "base_control_sha256": BASE_CONTROL_SHA, "face": FACE,
        "policy": policy(), "hvp_calls_started": 0, "hvp_calls_completed": 0,
        "hvp_history": [],
        "hvp_columns": [], "iterations": [], "candidate_committed": False,
        "active_candidate_committed": False, "optimizer_steps_applied": 0,
        "full_smooth_root": False, "minimum_claim": False, "response_claim": False}
    source_before = _source_hashes(plan, plan_path, plan_sha)
    record["source_before"] = source_before
    record["plan_hashes"] = {"source_files": plan["source_files"],
                              "archive_files": plan["archive_files"]}
    closure_state.update(plan=plan, plan_path=plan_path, plan_sha=plan_sha,
                         source_before=source_before, control=base["control"])
    _write(output, record)
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    original_base_control = base["control"].clone()
    input_before = _input_identity(problem, original, original_base_control, parameters, truth)
    runtime_before = _runtime()
    if (input_before != base["input_identity"] or runtime_before != base["runtime"]
            or _tensor_sha(parameters) != base["raw"]["parameters_sha256"]):
        raise ValueError("reconstructed fixed input/runtime differs from PR26495a receipt")
    closure_state.update(problem=problem, original=original, parameters=parameters, truth=truth,
                         input_before=input_before, runtime=runtime_before)
    fresh_j, fresh_branch = _fresh_native(problem, original_base_control, parameters)
    native_base_gradient = torch.func.grad(problem.objective, argnums=0)(original_base_control, parameters)
    if not torch.equal(native_base_gradient, base["expected_gradient"]):
        raise ValueError("fresh native gradient differs from accepted PR264 nonzero-face point")
    objective_tol = 128 * torch.finfo(original_base_control.dtype).eps * max(
        abs(float(fresh_j)), abs(float(base["expected_objective"])), torch.finfo(original_base_control.dtype).tiny)
    if (abs(float(fresh_j) - base["expected_objective"]) > objective_tol
            or fresh_branch["signature_sha256"]
                != base["state"].get("branch", {}).get("signature_sha256")):
        raise ValueError("fresh native J/full branch differs from accepted last state")
    record.update(numerical_status="fresh_native_state_closed", base_control_sha256=BASE_CONTROL_SHA,
        fresh_native={
        "objective": float(fresh_j),
        "receipt_gradient_inf": float(base["expected_gradient"].abs().max()),
        "gradient": native_base_gradient.tolist(), "gradient_receipt_matches": True,
        "branch": fresh_branch["signature"], "branch_scope": fresh_branch["scope"]})
    _write(output, record)

    weights = _face_weights(problem, **FACE)
    pivot = _pivot(original_base_control, weights)
    control = _chart(original_base_control, weights, pivot, 0.0)
    record.update(face_control=control.tolist(), face_control_sha256=_tensor_sha(control),
        face_chart={"retained_coordinates": [i for i in range(26) if i != pivot],
                    "pivot": pivot, "eta": 0.0,
                    "scope": "same 25 retained coordinates with analytic pivot reconstruction"})
    face_native_j = problem.objective(control, parameters)
    record["face_point_native_objective"] = float(face_native_j)
    face_scale = weights.abs().max()
    face0 = _face_value(control, weights)
    cancellation_scale = _flow_cancellation_scale(control, weights)
    production_face0, face_bound, face_audit_passed = _face_audit(
        problem, control, weights, FACE, 0.0)
    if not face_audit_passed:
        record.update(numerical_status="face_qualification_refused",
            refusal="production face Q or weighted-coordinate reduction exceeds its 128-epsilon cancellation bound",
            face_value=float(face0), production_face_value=float(production_face0),
            face_reduction_residual=float(production_face0 - face0),
            weighted_flow_cancellation_scale=float(cancellation_scale), face_roundoff_bound=face_bound,
            phase="finished", execution_status="completed")
        return _finish_without_commit(record, output=output, problem=problem, original=original,
            original_control=original_base_control, parameters=parameters, truth=truth,
            input_before=input_before, runtime_before=runtime_before, source_before=source_before,
            plan=plan, plan_path=plan_path, plan_sha=plan_sha, deadline=deadline)
    traces = {side: _analysis_trace(problem, control, parameters, side) for side in (-1, 1)}
    trace_gate = _trace_gate(traces[-1], traces[1], dtype=control.dtype)
    record.update(face_qualification={"value": float(face0), "roundoff_bound": face_bound,
        "weights": weights.tolist(), "qscale": float(face_scale), "pivot": pivot,
        "traces": {str(k): v for k, v in traces.items()}, "gate": trace_gate,
        "production_value": float(production_face0),
        "production_reduction_residual": float(production_face0 - face0),
        "weighted_flow_cancellation_scale": float(cancellation_scale)})
    if not trace_gate["passed"]:
        record.update(numerical_status="branch_support_refused",
            refusal="selected-face traces do not share the same other-face/minmod analysis signature",
            phase="finished", execution_status="completed")
        return _finish_without_commit(record, output=output, problem=problem, original=original,
            original_control=original_base_control, parameters=parameters, truth=truth,
            input_before=input_before, runtime_before=runtime_before, source_before=source_before,
            plan=plan, plan_path=plan_path, plan_sha=plan_sha, deadline=deadline)

    # Compare extension and native derivatives at the same chart points before
    # spending any Hessian products. The tiny signed offsets remain on one chart.
    eta_parity = (-FACE_PARITY_ETA, FACE_PARITY_ETA)
    normal = _face_normal(control, weights)
    parity_direction = normal / torch.linalg.vector_norm(normal)
    parity_results: dict[str, Any] = {}
    for side, eta in zip((-1, 1), eta_parity):
        _deadline(deadline)
        point = _chart(control, weights, pivot, eta)
        probe_q, probe_bound, probe_face_passed = _face_audit(problem, point, weights, FACE, eta)
        if not probe_face_passed or not bool(side * probe_q > 0):
            record.update(numerical_status="extension_parity_refused",
                refusal="parity probe lacks its requested production face value and side")
            return _finish_without_commit(record, output=output, problem=problem, original=original,
                original_control=original_base_control, parameters=parameters, truth=truth,
                input_before=input_before, runtime_before=runtime_before, source_before=source_before,
                plan=plan, plan_path=plan_path, plan_sha=plan_sha, deadline=deadline)
        def parity_hvp(call: Any, *, who: str, sign: int = side) -> Tensor:
            return _counted_hvp(record, output, deadline,
                {"phase": "parity", "side": sign, "operator": who}, call)
        native = _eval_native(problem, point, parameters, parity_direction,
            hvp_call=lambda call: parity_hvp(call, who="native"))
        extended = _eval_side(problem, point, parameters, side, parity_direction,
            hvp_call=lambda call: parity_hvp(call, who="selected_extension"))
        assert native[2] is not None and extended[2] is not None
        parity = _parity_gate(native, extended)
        # Branch support around the face must remain the qualified base support.
        trace = _analysis_trace(problem, point, parameters, side)
        supported = (not trace["nonfinite_or_tie"]
            and trace["stage_count"] == ANALYSIS_STAGES
            and trace["choices"] == traces[side]["choices"]
            and trace["face_signs"] == traces[side]["face_signs"])
        parity_results[str(side)] = {"eta": eta, "production_face_value": float(probe_q),
            "production_face_roundoff_bound": float(probe_bound), "production_side_passed": True,
            "parity": parity,
            "analysis_trace_matches_base": supported}
        if not parity["passed"] or not supported:
            record.update(numerical_status="extension_parity_refused", parity=parity_results,
                refusal="native/selected-side J-g-HVP or same-chart analysis support mismatch",
                phase="finished", execution_status="completed")
            return _finish_without_commit(record, output=output, problem=problem, original=original,
                original_control=original_base_control, parameters=parameters, truth=truth,
                input_before=input_before, runtime_before=runtime_before, source_before=source_before,
                plan=plan, plan_path=plan_path, plan_sha=plan_sha, deadline=deadline)

    side_data: dict[int, dict[str, Any]] = {}
    for side in (-1, 1):
        _deadline(deadline)
        j, g, _ = _eval_side(problem, control, parameters, side)
        side_data[side] = {"objective": float(j), "gradient": g}
    face_objective_tol = 128 * torch.finfo(control.dtype).eps * max(
        abs(float(face_native_j)), abs(side_data[-1]["objective"]),
        abs(side_data[1]["objective"]), torch.finfo(control.dtype).tiny)
    if max(abs(side_data[-1]["objective"] - float(face_native_j)),
           abs(side_data[1]["objective"] - float(face_native_j))) > face_objective_tol:
        record.update(numerical_status="extension_value_refused",
            refusal="one-sided objective extension does not equal original objective at the exact face",
            phase="finished", execution_status="completed")
        return _finish_without_commit(record, output=output, problem=problem, original=original,
            original_control=original_base_control, parameters=parameters, truth=truth,
            input_before=input_before, runtime_before=runtime_before, source_before=source_before,
            plan=plan, plan_path=plan_path, plan_sha=plan_sha, deadline=deadline)

    hessians: dict[int, Tensor] = {}
    counts = {"started": int(record["hvp_calls_started"]),
              "completed": int(record["hvp_calls_completed"])}
    for side in (-1, 1):
        cols = []
        for column in range(26):
            _deadline(deadline)
            direction = torch.zeros_like(control)
            direction[column] = 1
            record.update(numerical_status="hessian_columns", current_side=side, current_column=column)
            gfn = torch.func.grad(problem.objective, argnums=0)
            def calculate() -> Tensor:
                with transport.selected_face_extension(FACE["axis"], FACE["row"], FACE["column"], side):
                    return torch.func.jvp(lambda c: gfn(c, parameters), (control,), (direction,))[1]
            hv = _counted_hvp(record, output, deadline,
                {"phase": "basis", "side": side, "column": column}, calculate)
            counts["started"] = int(record["hvp_calls_started"])
            counts["completed"] = int(record["hvp_calls_completed"])
            cols.append(hv)
            record["hvp_calls_started"] = counts["started"]
            record["hvp_calls_completed"] = counts["completed"]
            record["hvp_columns"].append({"side": side, "column": column})
            _write(output, record)
        hessians[side] = torch.stack(cols, dim=1)
    hminus, hplus = hessians[-1], hessians[1]
    symmetry = {}
    for side, hessian in hessians.items():
        relative = float(torch.linalg.vector_norm(hessian - hessian.T) /
                         torch.linalg.vector_norm(hessian).clamp_min(torch.finfo(control.dtype).tiny))
        symmetry[str(side)] = {"relative": relative, "tolerance": 1e-9,
                               "passed": math.isfinite(relative) and relative <= 1e-9}
    if not all(row["passed"] for row in symmetry.values()):
        record.update(numerical_status="hessian_symmetry_refused", symmetry_audit=symmetry,
            refusal="one-sided Hessian symmetry audit failed; matrices were not symmetrized",
            phase="finished", execution_status="completed")
        return _finish_without_commit(record, output=output, problem=problem, original=original,
            original_control=original_base_control, parameters=parameters, truth=truth,
            input_before=input_before, runtime_before=runtime_before, source_before=source_before,
            plan=plan, plan_path=plan_path, plan_sha=plan_sha, deadline=deadline)

    theta, mixed_gradient, mix_info = _min_norm_mix(side_data[-1]["gradient"], side_data[1]["gradient"])
    matrix, residual, hmix = _coupled_matrix(hminus, hplus, side_data[-1]["gradient"],
        side_data[1]["gradient"], normal, face0, theta, face_scale)
    delta, solve_info = _solve_coupled(matrix, residual)
    record.update(numerical_status="coupled_system_solved", symmetry_audit=symmetry,
        hessian_minus=hminus.tolist(), hessian_plus=hplus.tolist(),
        mix=mix_info, theta=theta, matrix=matrix.tolist(), residual=residual.tolist(),
        delta=delta.tolist(), solve=solve_info,
        curvature_diagnostic=_diagnostic_curvature(hmix, mixed_gradient, control, weights, normal),
        hessian_scope="full 26x26 ambient FP64 one-sided Hessians; exact selected branch contexts",
        hvp_calls_started=counts["started"], hvp_calls_completed=counts["completed"])
    _write(output, record)
    if not solve_info["passed"]:
        record.update(numerical_status="coupled_solve_refused", refusal="27-equation solve residual exceeds 1e-10",
            phase="finished", execution_status="completed")
        return _finish_without_commit(record, output=output, problem=problem, original=original,
            original_control=original_base_control, parameters=parameters, truth=truth,
            input_before=input_before, runtime_before=runtime_before, source_before=source_before,
            plan=plan, plan_path=plan_path, plan_sha=plan_sha, deadline=deadline)

    path_direction, _ = _path_direction(control, weights, pivot, delta)
    path_norm = torch.linalg.vector_norm(path_direction)
    tangent_slopes = [float(torch.dot(side_data[s]["gradient"], path_direction)) for s in (-1, 1)]
    coupled_tangent = torch.cat((path_direction, delta[-1:].clone()))
    merit_direction = matrix @ coupled_tangent
    directional_model = torch.dot(residual, merit_direction)
    max_alpha = min(1.0, RADIUS / max(float(path_norm), torch.finfo(control.dtype).tiny))
    alphas = [max_alpha / (2**i) for i in range(MAX_CANDIDATES)]
    base_f_norm2 = torch.dot(residual, residual)
    iteration: dict[str, Any] = {"index": 0, "base_control_sha256": BASE_CONTROL_SHA,
        "alpha_limit": max_alpha, "trials": [], "proposal": None, "accepted": None,
        "hvp_calls_started": counts["started"], "hvp_calls_completed": counts["completed"]}
    # The actual chart tangent defines the trust radius and merit first order model.
    if not bool(torch.isfinite(path_direction).all()) or float(path_norm) == 0:
        iteration["refusal"] = "coupled direction has no finite nonzero chart tangent"
    elif max(tangent_slopes) >= 0 or float(torch.dot(mixed_gradient, path_direction)) >= 0:
        iteration["refusal"] = "one-sided or selected subgradient does not descend along the exact chart tangent"
    elif not bool(directional_model < 0):
        iteration["refusal"] = "coupled residual has no descending scaled merit model"
    else:
        for index, alpha in enumerate(alphas):
            _deadline(deadline)
            candidate, eta = _candidate_path(problem, control, weights, pivot, 0.0, delta, alpha)
            theta_trial = theta + alpha * float(delta[-1])
            if not 0.0 <= theta_trial <= 1.0:
                iteration["trials"].append({"alpha": alpha, "theta": theta_trial,
                    "status": "subgradient_weight_refused"})
                continue
            actual_delta = candidate - control
            actual_norm = torch.linalg.vector_norm(actual_delta)
            if float(actual_norm) > RADIUS * (1 + 64 * torch.finfo(control.dtype).eps):
                iteration["trials"].append({"alpha": alpha, "status": "radius_refused",
                    "actual_path_norm": float(actual_norm)})
                continue
            trial: dict[str, Any] = {"alpha": alpha, "eta": eta,
                "actual_path_norm": float(actual_norm), "control": candidate.tolist(),
                "control_sha256": _tensor_sha(candidate)}
            jnative = problem.objective(candidate, parameters)
            branch_trace = {side: _analysis_trace(problem, candidate, parameters, side) for side in (-1, 1)}
            branch_ok = all(not branch_trace[s]["nonfinite_or_tie"]
                and branch_trace[s]["stage_count"] == ANALYSIS_STAGES
                and branch_trace[s]["choices"] == traces[s]["choices"]
                and branch_trace[s]["face_signs"] == traces[s]["face_signs"] for s in (-1, 1))
            gs = {side: _eval_side(problem, candidate, parameters, side)[1] for side in (-1, 1)}
            qtrial = _face_value(candidate, weights)
            qtrial_production, q_bound_tensor, q_audit = _face_audit(
                problem, candidate, weights, FACE, 0.0)
            ftrial_vec = _residual_measure((1 - theta_trial) * gs[-1] + theta_trial * gs[1],
                qtrial, face_scale)
            actual_j_slope = max(tangent_slopes)
            j_required = float(face_native_j) + J_C1 * alpha * actual_j_slope
            f_slope = float(torch.dot(residual, merit_direction))
            f_required = float(base_f_norm2) + 2 * F_C1 * alpha * f_slope
            f_trial2 = torch.dot(ftrial_vec, ftrial_vec)
            q_bound = Q_ROUNDOFF / 128 * float(q_bound_tensor)
            j_pass = float(jnative) <= j_required
            f_pass = bool(float(f_trial2) <= f_required)
            q_pass = abs(float(qtrial)) <= q_bound
            q_pass = q_pass and q_audit and abs(float(qtrial_production)) <= q_bound \
                and abs(float(qtrial_production - qtrial)) <= q_bound
            trial.update({"objective": float(jnative), "theta": theta_trial,
                "J_armijo_passed": j_pass,
                "J_armijo_bound": j_required, "J_actual_path_slope": actual_j_slope,
                "F_squared": float(f_trial2), "F_squared_armijo_bound": f_required,
                "F_squared_first_order_slope": f_slope, "F_squared_armijo_passed": f_pass,
                "face_value": float(qtrial), "face_roundoff_bound": q_bound,
                "production_face_value": float(qtrial_production),
                "face_reduction_residual": float(qtrial_production - qtrial),
                "face_roundoff_passed": q_pass, "branch_support_passed": branch_ok,
                "side_gradients": {str(s): gs[s].tolist() for s in (-1, 1)},
                "branch_trace": {str(k): v for k, v in branch_trace.items()},
                "status": "accepted" if all((j_pass, f_pass, q_pass, branch_ok)) else "rejected"})
            iteration["trials"].append(trial)
            if trial["status"] == "accepted":
                iteration["proposal"] = trial
                break
    proposal = iteration["proposal"]
    evaluation_control = (torch.tensor(proposal["control"], dtype=torch.float64)
                          if proposal is not None else control)
    _deadline(deadline)
    final_native_j = problem.objective(evaluation_control, parameters)
    final_side = {side: _eval_side(problem, evaluation_control, parameters, side)
                  for side in (-1, 1)}
    final_trace = {side: _analysis_trace(problem, evaluation_control, parameters, side)
                   for side in (-1, 1)}
    eval_face = _face_value(evaluation_control, weights)
    eval_production_face, eval_q_bound_tensor, eval_face_audit = _face_audit(
        problem, evaluation_control, weights, FACE, 0.0)
    eval_q_bound = Q_ROUNDOFF / 128 * float(eval_q_bound_tensor)
    objective_match = (proposal is None or abs(float(final_native_j) - proposal["objective"])
        <= 128 * torch.finfo(control.dtype).eps * max(abs(float(final_native_j)),
            abs(proposal["objective"]), torch.finfo(control.dtype).tiny))
    gradient_match = True
    objective_extension_match = True
    branch_match = True
    merit_match = True
    if proposal is not None:
        saved_gradients = {s: torch.tensor(proposal["side_gradients"][str(s)], dtype=control.dtype)
                           for s in (-1, 1)}
        gradient_match = all(
            float((final_side[s][1] - saved_gradients[s]).abs().max()) <=
            128 * torch.finfo(control.dtype).eps * max(float(final_side[s][1].abs().max()),
                float(saved_gradients[s].abs().max()), torch.finfo(control.dtype).tiny)
            for s in (-1, 1))
        objective_extension_match = all(abs(float(final_side[s][0]) - float(final_native_j)) <=
            128 * torch.finfo(control.dtype).eps * max(abs(float(final_side[s][0])),
                abs(float(final_native_j)), torch.finfo(control.dtype).tiny) for s in (-1, 1))
        branch_match = all(not final_trace[s]["nonfinite_or_tie"]
            and final_trace[s]["signature_sha256"] == proposal["branch_trace"][str(s)]["signature_sha256"]
            for s in (-1, 1))
        final_f = _residual_measure((1 - proposal["theta"]) * final_side[-1][1]
            + proposal["theta"] * final_side[1][1], eval_face, face_scale)
        merit_value = float(torch.dot(final_f, final_f))
        merit_match = abs(merit_value - proposal["F_squared"]) <= 128 * torch.finfo(control.dtype).eps * max(
            abs(merit_value), abs(proposal["F_squared"]), torch.finfo(control.dtype).tiny)
    final_input_control = (evaluation_control if proposal is not None else original_base_control)
    final_input = _input_identity(problem, original, final_input_control, parameters, truth)
    final_fixed = face_geometry.model._fixed_input(final_input, input_before, _tensor_sha(final_input_control))
    final_runtime = _runtime()
    final_sources = _source_hashes(plan, plan_path, plan_sha)
    final_source_ok = final_sources == source_before
    final_deadline = time.monotonic() < deadline
    final_face_ok = (eval_face_audit and abs(float(eval_face)) <= eval_q_bound
                     and abs(float(eval_production_face)) <= eval_q_bound)
    final_repeat = {"native_objective": float(final_native_j),
        "side_objectives": {str(s): float(final_side[s][0]) for s in (-1, 1)},
        "side_gradients_finite": all(bool(torch.isfinite(final_side[s][1]).all()) for s in (-1, 1)),
        "gradient_matches_proposal": gradient_match, "objective_matches_proposal": objective_match,
        "side_objectives_match_native": objective_extension_match,
        "branch_matches_proposal": branch_match, "merit_matches_proposal": merit_match,
        "face_value": float(eval_face), "production_face_value": float(eval_production_face),
        "face_roundoff_bound": eval_q_bound, "face_roundoff_passed": final_face_ok,
        "side_trace_signatures": {str(s): final_trace[s]["signature_sha256"] for s in (-1, 1)},
        "source_unchanged": final_source_ok, "fixed_input_unchanged": bool(final_fixed),
        "runtime_unchanged": final_runtime == runtime_before,
        "deadline_passed": final_deadline}
    closure_ok = all((final_repeat["side_gradients_finite"], objective_match,
        objective_extension_match, branch_match, merit_match, final_face_ok, final_source_ok,
        bool(final_fixed), final_runtime == runtime_before, final_deadline))
    if proposal is not None and closure_ok:
        iteration["accepted"] = proposal
        iteration["candidate_type"] = "one_nonsmooth_coupled_step"
    elif proposal is not None:
        proposal["status"] = "final_closure_refused"
        iteration["refusal"] = "independent final J/g/F/trace/source/input/runtime/deadline repeat failed"
    record["iterations"] = [iteration]
    committed = iteration["accepted"] is not None
    final_control = evaluation_control if committed else original_base_control
    final_control_sha = _tensor_sha(final_control)
    final_input_after = _input_identity(problem, original, final_control, parameters, truth)
    fixed_ok = face_geometry.model._fixed_input(final_input_after, input_before, final_control_sha)
    runtime_after = _runtime()
    source_after = _source_hashes(plan, plan_path, plan_sha)
    source_ok = source_after == source_before
    record.update(phase="finished", execution_status="completed",
        numerical_status=("one_nonsmooth_coupled_step_accepted" if committed else "coupled_step_refused"),
        optimizer_steps_applied=int(committed), candidate_committed=committed,
        active_candidate_committed=False,
        candidate_type="one_nonsmooth_coupled_step" if committed else None,
        current_control=final_control.tolist(), current_control_sha256=final_control_sha,
        input_before=input_before, input_after=final_input_after,
        fixed_input_unchanged=bool(fixed_ok), runtime=runtime_before,
        runtime_after=runtime_after, runtime_unchanged=runtime_after == runtime_before,
        source_before=source_before, source_after=source_after, source_unchanged=source_ok,
        deadline_passed=final_deadline, final_repeat=final_repeat,
        elapsed_seconds=time.monotonic() - start,
        full_smooth_root=False, minimum_claim=False, response_claim=False)
    if not all((fixed_ok, runtime_after == runtime_before, source_ok)):
        record.update(execution_status="failed", numerical_status="diagnostic_closure_failed",
            refusal="source, fixed-input, or runtime closure changed after final repeat",
            candidate_committed=False, optimizer_steps_applied=0,
            current_control=original_base_control.tolist(), current_control_sha256=BASE_CONTROL_SHA)
    _write(output, record)
    return record


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    """Return a durable refusal record for every caught child-side failure."""
    closure_state: dict[str, Any] = {}
    try:
        if not output.exists():
            _write(output, {"phase": "running", "execution_status": "running",
                "numerical_status": "preflight", "plan_sha256": plan_sha,
                "base_control_sha256": BASE_CONTROL_SHA, "candidate_committed": False,
                "active_candidate_committed": False, "optimizer_steps_applied": 0,
                "hvp_calls_started": 0, "hvp_calls_completed": 0})
        return _run_child_impl(plan_path, plan_sha, output, closure_state)
    except Exception as error:
        try:
            record = json.loads(output.read_text())
            if not isinstance(record, dict):
                record = {}
        except (OSError, ValueError):
            record = {}
        record.update(phase="finished", numerical_status=(
            "internal_deadline_refused" if isinstance(error, TimeoutError) else "prerequisite_or_execution_refused"),
            refusal=f"{type(error).__name__}: {error}",
            execution_status="failed", candidate_committed=False,
            active_candidate_committed=False, optimizer_steps_applied=0,
            full_smooth_root=False, minimum_claim=False, response_claim=False,
            candidate_type=None)
        try:
            plan = closure_state["plan"]
            plan_path = closure_state["plan_path"]
            plan_sha = closure_state["plan_sha"]
            control = closure_state["control"]
            source_before = closure_state["source_before"]
            problem = closure_state["problem"]
            original = closure_state["original"]
            parameters = closure_state["parameters"]
            truth = closure_state["truth"]
            input_before = closure_state["input_before"]
            runtime_before = closure_state["runtime"]
            input_after = _input_identity(problem, original, control, parameters, truth)
            runtime_after = _runtime()
            source_after = _source_hashes(plan, plan_path, plan_sha)
            fixed_ok = face_geometry.model._fixed_input(input_after, input_before, BASE_CONTROL_SHA)
            source_ok = source_before == source_after
            runtime_ok = runtime_before == runtime_after
            record.update(base_control_sha256=BASE_CONTROL_SHA,
                current_control=control.tolist(), current_control_sha256=BASE_CONTROL_SHA,
                input_before=input_before, input_after=input_after,
                fixed_input_unchanged=bool(fixed_ok), runtime=runtime_before,
                runtime_after=runtime_after, runtime_unchanged=runtime_ok,
                source_before=source_before, source_after=source_after,
                source_unchanged=source_ok, deadline_passed=not isinstance(error, TimeoutError))
            if isinstance(error, TimeoutError) and fixed_ok and source_ok and runtime_ok:
                record["execution_status"] = "completed"
            if not (fixed_ok and source_ok and runtime_ok):
                record["execution_status"] = "failed"
                record["numerical_status"] = "diagnostic_closure_failed"
                record["refusal"] = "source, fixed-input, or runtime closure failed after error"
        except Exception as close_error:
            record.update(execution_status="failed", source_unchanged=False,
                fixed_input_unchanged=False, runtime_unchanged=False,
                closure_error=f"{type(close_error).__name__}: {close_error}")
        _write(output, record)
        return record


def _execution_status(resource: dict[str, Any]) -> str:
    elapsed = float(resource.get("elapsed_seconds", WALL_SECONDS + 1))
    peak = int(resource.get("sampled_peak_rss_bytes", RSS_BYTES + 1))
    if (resource.get("wall_limit_seconds") == WALL_SECONDS
            and resource.get("rss_limit_bytes") == RSS_BYTES
            and resource.get("exit_code") == 0 and resource.get("resource_termination") is None
            and resource.get("monitor_error") is None and not resource.get("received_sigterm")
            and elapsed <= WALL_SECONDS and peak < RSS_BYTES):
        return "completed"
    termination = resource.get("resource_termination")
    if termination == "wall_time_limit" or elapsed > WALL_SECONDS:
        return "wall_timeout"
    if termination == "rss_limit" or peak >= RSS_BYTES:
        return "rss_limit"
    if termination == "cancelled" or resource.get("received_sigterm"):
        return "cancelled"
    if termination in {"rss_monitor_unavailable", "resource_monitor_error"} \
            or resource.get("monitor_error") is not None:
        return "monitor_failure"
    return "failed"


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    if plan_path.is_symlink():
        raise ValueError("nonsmooth coupled-face plan must not be a symbolic link")
    plan_path = plan_path.resolve()
    plan = _load_plan(plan_path, plan_sha)
    if any(p.exists() or p.is_symlink() for p in (output, resource, log, output.with_suffix(".run.json"))):
        raise ValueError("output/resource/log paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
        "--plan", str(plan_path), "--plan-sha256", plan_sha, "--output", str(output)]
    resource_result = run_guarded_diagnostic(command, wall_seconds=WALL_SECONDS,
        rss_bytes=RSS_BYTES, report_path=resource, log_path=log)
    status = _execution_status(resource_result)
    parent = {"execution_status": status, "resource": resource_result,
        "child_sha256": _sha(output) if output.exists() else None,
        "numerical_status": "not_reached", "child_read_error": None}
    child: dict[str, Any] | None = None
    if output.exists():
        try:
            child = json.loads(output.read_text())
            if not isinstance(child, dict):
                raise ValueError("child report must be a JSON object")
        except (OSError, ValueError) as error:
            parent.update(execution_status="failed",
                child_read_error=f"{type(error).__name__}: {error}")
            _write(output.with_suffix(".run.json"), parent)
            return {"parent": parent, "child": None}
        parent["numerical_status"] = child.get("numerical_status")
        iterations = child.get("iterations", [])
        accepted = [row for row in iterations if row.get("accepted") is not None] if isinstance(iterations, list) else []
        current_sha = child.get("current_control_sha256")
        closed = (child.get("phase") == "finished" and child.get("execution_status") == "completed"
            and child.get("plan_sha256") == plan_sha
            and child.get("base_control_sha256") == BASE_CONTROL_SHA
            and child.get("source_unchanged") is True
            and child.get("fixed_input_unchanged") is True
            and child.get("runtime_unchanged") is True
            and child.get("runtime_after") == child.get("runtime")
            and child.get("source_after") == child.get("source_before")
            and child.get("active_candidate_committed") is False
            and isinstance(iterations, list)
            and child.get("optimizer_steps_applied") == int(child.get("candidate_committed") is True)
            and child.get("candidate_committed") is (len(accepted) == 1)
            and child.get("hvp_calls_started") == child.get("hvp_calls_completed")
            and child.get("hvp_calls_completed", 0) <= MAX_HVP
            and ((not accepted and current_sha == BASE_CONTROL_SHA)
                 or (len(accepted) == 1 and child.get("candidate_type") == "one_nonsmooth_coupled_step"
                     and accepted[0].get("accepted", {}).get("control_sha256") == current_sha)))
        if status == "completed" and not closed:
            parent.update(execution_status="failed", execution_failure_reason="child closure refused")
        elif status != "completed":
            parent["execution_status"] = "failed" if status == "completed" else status
    _write(output.with_suffix(".run.json"), parent)
    return {"parent": parent, "child": child if output.exists() else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "nonsmooth_coupled_face_attempt1/step.json")
    parser.add_argument("--resource", type=Path, default=EVIDENCE / "nonsmooth_coupled_face_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path, default=EVIDENCE / "nonsmooth_coupled_face_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
