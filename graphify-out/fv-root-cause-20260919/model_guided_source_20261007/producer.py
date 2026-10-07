"""Three full-space model-guided original-J iterations after the Qy32 step."""
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

from examples.weather_scenarios import fv_point_3h_qy32_tangent_step as tangent
from examples.weather_scenarios import fv_point_3h_qy32_diagnostic as qy
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as guard_policy
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "MODEL_GUIDED_FULL_SPACE_PLAN_20261007.json"
TANGENT_PLAN = EVIDENCE / "QY32_TANGENT_STEP_PLAN_20261007.json"
TANGENT_PLAN_SHA = "70f9560c0355fa6b65927ac70fb4de2c055f338d5dd276c8fa453fc30379ecf9"
STEP = EVIDENCE / "qy32_tangent_step_attempt1/step.json"
STEP_PARENT = STEP.with_suffix(".run.json")
STEP_RESOURCE = STEP.with_suffix(".resource.json")
STEP_SHA = "061c51d2b016f024e24ef7e1afb30be1c1169836305c92e97d806301d3040c53"
STEP_PARENT_SHA = "e84cd4b87a22ebceeb5d625d15eb6043d5434b5a855e66703e0513ddda9487d7"
STEP_RESOURCE_SHA = "ab0d550cf94d75df79bf3017cc1b3583736afe5f665e0c7f4d879557b5e1c316"
CONTROL_SHA = "058895848fdeb348e8d6bb1f22ac13e3f2dadfa48bff46522ffbf94fedf72ffa"
PARAMETERS_SHA = qy.PARAMETERS_SHA
SELF = "examples/weather_scenarios/fv_point_3h_model_guided_continuation.py"
TEST = "tests/test_fv_point_3h_model_guided_continuation.py"
MAX_ITERATIONS, MAX_HVP, MAX_CANDIDATES = 3, 3, 16
RADIUS, C1_J, C1_PHI, ROOT_GINF = 0.05, 1e-4, 1e-4, 1e-10
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 240.0, 300.0, 1024**3


def policy() -> dict[str, Any]:
    return {"max_iterations": MAX_ITERATIONS, "max_current_hvp": MAX_HVP,
            "max_candidates_per_iteration": MAX_CANDIDATES, "radius": RADIUS,
            "j_armijo_c1": C1_J, "phi_armijo_c1": C1_PHI,
            "root_gradient_inf": ROOT_GINF, "internal_seconds": INTERNAL_SECONDS,
            "outer_seconds": WALL_SECONDS, "rss_bytes": RSS_BYTES,
            "guarded_launches": 1, "pcg_solves": 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_alpha(gradient: Tensor, h_direction: Tensor, radius: float = RADIUS) -> float:
    """Bound a full-gradient model step by control radius and Phi-model minimizer."""
    if (gradient.shape != (26,) or h_direction.shape != (26,)
            or gradient.dtype != torch.float64 or h_direction.dtype != torch.float64
            or not bool(torch.isfinite(gradient).all() & torch.isfinite(h_direction).all())
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError("model step needs finite FP64 26-vectors and a positive radius")
    direction = -gradient
    direction_norm = float(torch.linalg.vector_norm(direction))
    h_norm = float(torch.linalg.vector_norm(h_direction))
    if not math.isfinite(direction_norm) or direction_norm <= 0:
        raise guard_policy.StepRefusal("full-gradient direction has zero or nonfinite norm")
    if not math.isfinite(h_norm) or h_norm <= 0:
        raise guard_policy.StepRefusal("current HVP has zero or nonfinite norm")
    checks = tangent.resolved_direction_checks(gradient, direction, h_direction)
    scaled_h = h_direction / h_norm
    scaled_dot = float((gradient / torch.linalg.vector_norm(gradient)) @ scaled_h)
    # −gᵀHd / ||Hd||², arranged to avoid squaring a large HVP norm.
    phi_minimizer = (float(torch.linalg.vector_norm(gradient)) / h_norm) * (-scaled_dot)
    radius_alpha = radius / direction_norm
    alpha = min(1.0, radius_alpha, phi_minimizer)
    if not math.isfinite(alpha) or alpha <= 0 or checks["g_dot_Hd"] >= 0:
        raise guard_policy.StepRefusal("model-guided alpha is not a resolved positive step")
    return alpha


def first_order_model(objective: float, phi: float, gradient: Tensor,
                      direction: Tensor, h_direction: Tensor, alpha: float) -> dict[str, Any]:
    """Record exact linearized-gradient Phi and Taylor J models for a straight path."""
    linear_gradient = gradient + alpha * h_direction
    j_slope = float(gradient @ direction)
    d_h_d = float(direction @ h_direction)
    return {"predicted_J_first_order": objective + alpha * j_slope,
            "predicted_J_quadratic": objective + alpha * j_slope + 0.5 * alpha**2 * d_h_d,
            "predicted_gradient": linear_gradient.tolist(),
            "predicted_Phi_from_linear_gradient": float(torch.dot(linear_gradient, linear_gradient) / 2),
            "scope": "first-order gradient model and local quadratic J model along full-space straight path"}


def merit_acceptance(base_j: float, base_phi: float, trial_j: float, trial_phi: float,
                    alpha: float, g_dot_d: float, g_dot_h_d: float) -> dict[str, Any]:
    """Apply dual Armijo and require a representable actual Phi decrease."""
    result = tangent.dual_armijo(base_j, base_phi, trial_j, trial_phi,
                                 alpha, g_dot_d, g_dot_h_d)
    phi_budget = 128 * torch.finfo(torch.float64).eps * max(
        abs(base_phi), abs(trial_phi), torch.finfo(torch.float64).tiny)
    resolved = base_phi - trial_phi > phi_budget
    result.update(phi_decrease_roundoff_budget=phi_budget,
                  phi_decrease_resolved=resolved,
                  accepted=result["accepted"] and resolved)
    return result


def _fixed_input(candidate: dict[str, Any], base: dict[str, Any], control_sha: str) -> bool:
    return tangent._check_fixed_input(candidate, base, control_sha)


def _compact_state(control: Tensor, measured: dict[str, Any]) -> dict[str, Any]:
    compact = tangent._compact_measure(measured)
    compact.pop("static_flux_signs", None)
    return {"control": control.tolist(), "control_sha256": tangent._tensor_sha(control),
            "objective": compact["objective"], "phi": compact["phi"],
            "gradient": compact["gradient"], "gradient_inf": compact["gradient_inf"],
            "gradient_l2": compact["gradient_norm"], "branch": compact["branch"],
            "branch_partition": compact["branch_partition"]}


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    path = path.resolve()
    if path != PLAN.resolve() or path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()) or _sha(path) != digest:
        raise ValueError("model-guided continuation plan identity mismatch")
    plan = json.loads(path.read_text())
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if (plan.get("experiment_kind") != "model_guided_full_space_continuation"
            or plan.get("producing_plan") != TANGENT_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != TANGENT_PLAN_SHA
            or plan.get("policy") != policy()
            or not isinstance(sources, dict) or not isinstance(archives, dict)):
        raise ValueError("model-guided plan scope, policy, or pin maps changed")
    old = cast(dict[str, Any], tangent._load_plan(TANGENT_PLAN, TANGENT_PLAN_SHA))
    if (not set(old["source_files"]) <= set(sources)
            or any(sources[name] != value for name, value in old["source_files"].items())
            or any(archives.get(name) != value for name, value in old["archive_files"].items())):
        raise ValueError("model-guided plan changed inherited tangent-step evidence pins")
    if sources.get(SELF) != _sha(ROOT / SELF) or sources.get(TEST) != _sha(ROOT / TEST):
        raise ValueError("model-guided plan must pin its producer and tests")
    for name, value in {**sources, **archives}.items():
        tangent._pinned_path(name, value)
    for path_, value in ((TANGENT_PLAN, TANGENT_PLAN_SHA), (STEP, STEP_SHA),
                         (STEP_PARENT, STEP_PARENT_SHA), (STEP_RESOURCE, STEP_RESOURCE_SHA)):
        if archives.get(path_.relative_to(ROOT).as_posix()) != value:
            raise ValueError("model-guided plan omits an exact accepted tangent-step receipt")
    return plan


def _load_base() -> dict[str, Any]:
    _ = tangent._load_plan(TANGENT_PLAN, TANGENT_PLAN_SHA)
    if _sha(STEP) != STEP_SHA or _sha(STEP_PARENT) != STEP_PARENT_SHA or _sha(STEP_RESOURCE) != STEP_RESOURCE_SHA:
        raise ValueError("accepted PR #258 tangent step artifact changed")
    raw = json.loads(STEP.read_text())
    parent = json.loads(STEP_PARENT.read_text())
    resource = json.loads(STEP_RESOURCE.read_text())
    trials = raw.get("trials", [])
    accepted = [trial for trial in trials if trial.get("status") == "accepted"]
    if (raw.get("phase") != "finished"
            or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "one_qy32_tangent_dual_armijo_step_accepted"
            or raw.get("optimizer_steps_applied") != 1
            or raw.get("candidate_committed") is not True
            or raw.get("plan_sha256") != TANGENT_PLAN_SHA
            or raw.get("base_control_sha256") != qy.ENDPOINT_SHA
            or raw.get("accepted_control_sha256") != CONTROL_SHA
            or raw.get("parameters_sha256") != PARAMETERS_SHA
            or not tangent._check_fixed_input(raw.get("input_after", {}),
                raw.get("input_before", {}), CONTROL_SHA)
            or len(accepted) != 1 or accepted[-1].get("control_sha256") != CONTROL_SHA
            or accepted[-1].get("J_armijo_passed") is not True
            or accepted[-1].get("Phi_armijo_passed") is not True
            or accepted[-1].get("strict_point_passed") is not True
            or accepted[-1].get("branch", {}).get("status") != "passed_strict_branch"
            or accepted[-1].get("objective") != raw.get("accepted_objective")
            or accepted[-1].get("phi") != raw.get("accepted_phi")
            or accepted[-1].get("gradient") != raw.get("accepted_gradient")
            or accepted[-1].get("branch_signature_sha256")
            != raw.get("accepted_branch", {}).get("signature_sha256")
            or raw.get("source_unchanged") is not True
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime") != raw.get("runtime_after")
            or parent.get("execution_status") != "completed"
            or parent.get("child_sha256") != STEP_SHA
            or parent.get("numerical_status") != raw.get("numerical_status")
            or guard_policy.execution_status(resource) != "completed"
            or any(_sha(ROOT / name) != value for name, value in raw.get("source_before", {}).items())):
        raise ValueError("PR #258 accepted endpoint or closed receipts are invalid")
    return raw


def _static_zero_faces(problem: Any, control: Tensor) -> list[dict[str, Any]]:
    qx, qy_values = qy.production_fluxes(problem, control)
    zeros = []
    for axis, values in (("qx", qx), ("qy", qy_values)):
        for row, col in torch.nonzero(values == 0, as_tuple=False).tolist():
            zeros.append({"axis": axis, "row": row, "column": col})
    return zeros


def _accepted_state(record: dict[str, Any], control: Tensor,
                    measured: dict[str, Any], iteration: dict[str, Any]) -> None:
    state = _compact_state(control, measured)
    record["iterations"].append(iteration)
    record["current_control"] = control.tolist()
    record["current_control_sha256"] = state["control_sha256"]
    record["current_state"] = {key: value for key, value in state.items() if key not in {"control"}}


def _commit_iteration(record: dict[str, Any], control: Tensor,
                      measured: dict[str, Any], iteration: dict[str, Any],
                      *, closure_ok: bool) -> bool:
    if not closure_ok:
        return False
    iteration["committed"] = True
    _accepted_state(record, control, measured, iteration)
    record.update(optimizer_steps_applied=len(record["iterations"]), active_iteration=None,
                  active_candidate_committed=False, candidate_committed=True)
    return True


def _mark_active_not_committed(record: dict[str, Any], reason: str) -> None:
    active = record.get("active_iteration")
    if isinstance(active, dict) and active.get("trials"):
        tangent._mark_not_committed({"optimizer_steps_applied": 0,
                                     "trials": active["trials"]}, reason)
    record["active_candidate_committed"] = False
    record["candidate_committed"] = bool(record.get("iterations"))


def _current_hvp(gradient_fn: Any, control: Tensor, parameters: Tensor,
                 direction: Tensor, deadline: float,
                 on_start: Any, on_complete: Any) -> Tensor:
    if time.monotonic() >= deadline:
        raise TimeoutError("240-second budget expired before current HVP")
    on_start()
    result = torch.func.jvp(gradient_fn, (control, parameters),
                            (direction, torch.zeros_like(parameters)))[1]
    on_complete()
    if time.monotonic() >= deadline:
        raise TimeoutError("240-second budget expired after current HVP")
    return result


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "setup_not_complete", "plan_sha256": plan_sha,
        "policy": policy(), "source_before": None, "source_after": None,
        "source_unchanged": None, "input_before": None, "input_after": None,
        "fixed_input_unchanged": None, "runtime": None, "runtime_after": None,
        "iterations": [], "active_iteration": None, "hvp_calls": 0,
        "hvp_calls_completed": 0, "pcg_solves": 0, "optimizer_steps_applied": 0,
        "candidate_committed": False, "active_candidate_committed": False,
        "full_root_claim": False, "full_Hessian_computed": False, "response_computed": False,
        "score_computed": False, "reanalysis_computed": False}
    guard_policy._write(output, record)
    source_names: set[str] = set()
    problem = None
    original = None
    control = None
    parameters = None
    truth = None
    base_identity: dict[str, Any] = {}

    def close(final_control: Tensor | None) -> None:
        if source_names:
            after = {name: _sha(ROOT / name) for name in sorted(source_names)}
            record["source_after"] = after
            record["source_unchanged"] = record["source_before"] == after
        if all(value is not None for value in (problem, original, control, parameters, truth)):
            target = final_control if final_control is not None else control
            identity = seed._input_identity(cast(Any, problem), cast(Tensor, original),
                cast(Tensor, target), cast(Tensor, parameters), cast(Tensor, truth))
            record["input_after"] = identity
            record["fixed_input_unchanged"] = _fixed_input(identity, base_identity,
                tangent._tensor_sha(cast(Tensor, target)))
        record["runtime_after"] = guard_policy.blocks.runtime_identity()
        record["elapsed_seconds"] = time.monotonic() - started

    try:
        plan = _load_plan(plan_path, plan_sha)
        tangent_raw = _load_base()
        source_names = set(plan["source_files"]) | set(plan["archive_files"]) | {
            plan_path.resolve().relative_to(ROOT.resolve()).as_posix()}
        expected = {**plan["source_files"], **plan["archive_files"],
                    plan_path.resolve().relative_to(ROOT.resolve()).as_posix(): plan_sha}
        before = {name: _sha(ROOT / name) for name in sorted(source_names)}
        if before != expected:
            raise ValueError("prelaunch source/archive hashes differ from plan pins")
        record["source_before"] = before
        problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
        control = torch.tensor(tangent_raw["accepted_control"], dtype=torch.float64)
        if (_sha(TANGENT_PLAN) != TANGENT_PLAN_SHA
                or tangent._tensor_sha(control) != CONTROL_SHA
                or tangent._tensor_sha(parameters) != PARAMETERS_SHA):
            raise ValueError("reconstructed accepted PR #258 control/parameters changed")
        base_identity = seed._input_identity(problem, original, control, parameters, truth)
        if base_identity != tangent_raw["input_after"]:
            raise ValueError("reconstructed original fixed input differs from accepted tangent receipt")
        runtime = guard_policy.blocks.runtime_identity()
        if runtime != tangent_raw["runtime"]:
            raise ValueError("runtime differs from accepted tangent endpoint")
        record.update(base_control_sha256=CONTROL_SHA, parameters_sha256=PARAMETERS_SHA,
                      current_control=control.tolist(), current_control_sha256=CONTROL_SHA,
                      input_before=base_identity, runtime=runtime)
        current_problem = problem
        current_parameters = parameters
        current_original = original
        current_truth = truth
        gradient_fn = torch.func.grad(current_problem.objective, argnums=0)
        current_measure = qy._measure(current_problem, current_parameters, control,
            float(qy.production_qy32(current_problem, control)), gradient_fn,
            with_gradient=True, deadline=deadline)
        saved_trial = [item for item in tangent_raw["trials"] if item.get("status") == "accepted"][-1]
        baseline_ok = (current_measure["branch"].get("status") == "passed_strict_branch"
            and current_measure["branch"].get("signature_sha256") == saved_trial["branch_signature_sha256"]
            and tangent._metric(float(tangent_raw["accepted_objective"]), current_measure["objective"])["passed"]
            and tangent._metric(float(tangent_raw["accepted_phi"]), current_measure["phi"])["passed"]
            and all(tangent._metric(a, float(b))["passed"] for a, b in
                    zip(tangent_raw["accepted_gradient"], current_measure["gradient"], strict=True)))
        if not baseline_ok:
            raise ValueError("fresh full J/g/Phi/branch differs from accepted PR #258 endpoint")
        current_state = _compact_state(control, current_measure)
        record.update(current_state={key: value for key, value in current_state.items() if key != "control"},
                      current_control=control.tolist(), current_control_sha256=CONTROL_SHA,
                      numerical_status="iterating")
        current_measure.pop("branch_signature", None)
        current_measure.pop("static_flux_signs", None)
        guard_policy._write(output, record)
        for iteration_index in range(MAX_ITERATIONS):
            if time.monotonic() >= deadline:
                raise TimeoutError("240-second budget expired before a new model-guided iteration")
            gradient = torch.tensor(record["current_state"]["gradient"], dtype=torch.float64)
            gradient_inf = float(gradient.abs().max())
            if gradient_inf <= ROOT_GINF:
                record.update(phase="finished", execution_status="completed",
                              numerical_status="root_pending_audit", root_pending_audit=True)
                break
            static_zeros = _static_zero_faces(current_problem, control)
            iteration: dict[str, Any] = {"index": iteration_index, "base_control_sha256": tangent._tensor_sha(control),
                                         "base_objective": record["current_state"]["objective"],
                                         "base_phi": record["current_state"]["phi"],
                                         "base_gradient": gradient.tolist(),
                                         "base_gradient_inf": gradient_inf,
                                         "static_zero_faces": static_zeros, "trials": [],
                                         "hvp_calls_started": 0, "hvp_calls_completed": 0}
            record["active_iteration"] = iteration
            record["active_candidate_committed"] = False
            if static_zeros:
                iteration["status"] = "static_face_zero_refusal"
                record.update(phase="finished", execution_status="completed",
                              numerical_status="static_face_zero_refusal")
                break
            direction = -gradient
            if record["hvp_calls"] >= MAX_HVP:
                raise guard_policy.StepRefusal("three-current-HVP cap reached")
            def product_started() -> None:
                record["hvp_calls"] += 1
                iteration["hvp_calls_started"] = 1
                guard_policy._write(output, record)
            def product_completed() -> None:
                record["hvp_calls_completed"] += 1
                iteration["hvp_calls_completed"] = 1
                guard_policy._write(output, record)
            h_direction = _current_hvp(gradient_fn, control, current_parameters,
                direction, deadline, product_started, product_completed)
            if record["hvp_calls"] > MAX_HVP:
                raise RuntimeError("model-guided continuation exceeded its current-HVP cap")
            if h_direction.shape != (26,) or not bool(torch.isfinite(h_direction).all()):
                raise guard_policy.StepRefusal("current-point HVP is nonfinite")
            try:
                slopes = tangent.resolved_direction_checks(gradient, direction, h_direction)
                alpha0 = model_alpha(gradient, h_direction, RADIUS)
            except guard_policy.StepRefusal as error:
                iteration.update(status="direction_refusal", refusal=str(error), H_direction=h_direction.tolist())
                record.update(phase="finished", execution_status="completed",
                              numerical_status="direction_refusal")
                break
            iteration.update(direction=direction.tolist(), H_direction=h_direction.tolist(),
                             direction_checks=slopes, alpha_start=alpha0,
                             model_alpha_scope="min(1, radius/||d||, −gᵀHd/||Hd||²)")
            record["active_iteration"] = iteration
            guard_policy._write(output, record)
            accepted: Tensor | None = None
            accepted_measure: dict[str, Any] | None = None
            for trial_index in range(MAX_CANDIDATES):
                if time.monotonic() >= deadline:
                    raise TimeoutError("240-second budget expired during full-space candidate search")
                alpha = alpha0 * 0.5**trial_index
                candidate = control + alpha * direction
                delta = candidate - control
                displacement = float(torch.linalg.vector_norm(delta))
                trial: dict[str, Any] = {"index": trial_index, "alpha": alpha,
                    "control_sha256": tangent._tensor_sha(candidate), "displacement_l2": displacement,
                    "new_hvp": 0}
                if displacement == 0.0:
                    trial.update(status="zero_displacement_refusal")
                    iteration["trials"].append(trial)
                    iteration["status"] = "zero_displacement_refusal"
                    record.update(phase="finished", execution_status="completed",
                                  numerical_status="zero_displacement_refusal")
                    break
                if displacement > RADIUS:
                    trial.update(status="rejected", refusal="actual full-control radius exceeded before FV")
                    iteration["trials"].append(trial)
                    guard_policy._write(output, record)
                    continue
                zeros = _static_zero_faces(current_problem, candidate)
                trial["static_zero_faces"] = zeros
                if zeros:
                    trial.update(status="rejected", refusal="static zero face before candidate gradient")
                    iteration["trials"].append(trial)
                    guard_policy._write(output, record)
                    continue
                eta = float(qy.production_qy32(current_problem, candidate))
                measured = qy._measure(current_problem, current_parameters, candidate, eta,
                    gradient_fn, with_gradient=True, deadline=deadline)
                armijo = merit_acceptance(float(record["current_state"]["objective"]),
                    float(record["current_state"]["phi"]), measured["objective"], measured["phi"],
                    alpha, slopes["g_dot_d"], slopes["g_dot_Hd"])
                trial.update(objective=measured["objective"], phi=measured["phi"],
                    gradient=measured["gradient"], gradient_inf=measured["gradient_inf"],
                    gradient_l2=measured["gradient_norm"], branch=measured["branch"],
                    branch_partition=measured["branch_partition"], eta=eta,
                    strict_point_passed=measured["branch"].get("status") == "passed_strict_branch",
                    model=first_order_model(float(record["current_state"]["objective"]),
                        float(record["current_state"]["phi"]), gradient, direction,
                        h_direction, alpha), **armijo)
                measured.pop("branch_signature", None)
                measured.pop("static_flux_signs", None)
                strict_passed = trial["strict_point_passed"]
                trial["dual_armijo_passed"] = armijo["accepted"]
                trial["accepted"] = strict_passed and armijo["accepted"]
                trial["status"] = "accepted" if trial["accepted"] else "rejected"
                iteration["trials"].append(trial)
                guard_policy._write(output, record)
                if not armijo["phi_decrease_resolved"]:
                    continue
                if trial["status"] == "accepted":
                    accepted, accepted_measure = candidate, measured
                    break
            if record.get("numerical_status") == "zero_displacement_refusal":
                break
            if accepted is None or accepted_measure is None:
                iteration.setdefault("status", "dual_armijo_refusal")
                record.update(phase="finished", execution_status="completed",
                              numerical_status=iteration["status"])
                break
            # Fresh endpoint check precedes each irreversible state update.
            final_eta = float(qy.production_qy32(current_problem, accepted))
            final_measure = qy._measure(current_problem, current_parameters, accepted,
                final_eta, gradient_fn, with_gradient=True, deadline=deadline)
            trial = iteration["trials"][-1]
            recheck = (final_measure["branch"].get("status") == "passed_strict_branch"
                and final_measure["branch"].get("signature_sha256") == trial["branch"].get("signature_sha256")
                and tangent._metric(float(trial["objective"]), final_measure["objective"])["passed"]
                and tangent._metric(float(trial["phi"]), final_measure["phi"])["passed"]
                and all(tangent._metric(a, float(b))["passed"] for a, b in
                        zip(trial["gradient"], final_measure["gradient"], strict=True)))
            if not recheck:
                tangent._mark_not_committed({"optimizer_steps_applied": 0, "trials": iteration["trials"]},
                                            "fresh endpoint recheck failed")
                raise ValueError("fresh full-space endpoint recheck failed")
            final_gradient = torch.tensor(final_measure["gradient"], dtype=torch.float64)
            delta_norm = float(torch.linalg.vector_norm(accepted - control))
            iteration.update(status="accepted", accepted_control=accepted.tolist(),
                accepted_control_sha256=tangent._tensor_sha(accepted),
                accepted_objective=final_measure["objective"], accepted_phi=final_measure["phi"],
                accepted_gradient=final_gradient.tolist(),
                accepted_gradient_inf=float(final_gradient.abs().max()),
                accepted_gradient_l2=float(torch.linalg.vector_norm(final_gradient)),
                accepted_displacement_l2=delta_norm, accepted_geometry=final_measure["geometry"])
            source_after = {name: _sha(ROOT / name) for name in sorted(source_names)}
            input_after = seed._input_identity(current_problem, current_original, accepted,
                                               current_parameters, current_truth)
            runtime_after = guard_policy.blocks.runtime_identity()
            closed = (source_after == before and _fixed_input(input_after, base_identity,
                        tangent._tensor_sha(accepted)) and runtime_after == runtime)
            if not closed:
                tangent._mark_not_committed({"optimizer_steps_applied": 0,
                    "trials": iteration["trials"]}, "iteration commit closure failed")
                raise ValueError("iteration failed source/input/runtime closure")
            if time.monotonic() >= deadline:
                tangent._mark_not_committed({"optimizer_steps_applied": 0,
                    "trials": iteration["trials"]}, "iteration closure exceeded internal budget")
                raise TimeoutError("240-second budget expired after iteration closure")
            if not _commit_iteration(record, accepted, final_measure, iteration, closure_ok=closed):
                raise ValueError("iteration commit gate refused the provisional endpoint")
            control = accepted
            current_measure = final_measure
            final_measure.pop("branch_signature", None)
            final_measure.pop("static_flux_signs", None)
            record.update(source_after=source_after, source_unchanged=True,
                input_after=input_after, fixed_input_unchanged=True, runtime_after=runtime_after,
                candidate_committed=True)
            gradient_inf = record["current_state"]["gradient_inf"]
            if gradient_inf <= ROOT_GINF:
                record.update(phase="finished", execution_status="completed",
                              numerical_status="root_pending_audit", root_pending_audit=True)
                break
            guard_policy._write(output, record)
        if record.get("phase") != "finished":
            record.update(phase="finished", execution_status="completed",
                          numerical_status="max_iterations_completed")
        close(control)
        if (record.get("source_unchanged") is not True
                or record.get("fixed_input_unchanged") is not True
                or record.get("runtime_after") != record.get("runtime")):
            raise ValueError("final model-guided source/input/runtime closure refused")
        if time.monotonic() >= deadline:
            raise TimeoutError("240-second budget expired after final receipt closure")
        record["elapsed_seconds"] = time.monotonic() - started
        guard_policy._write(output, record)
        return record
    except TimeoutError as error:
        close(control)
        closure = (record.get("source_unchanged") is True
                   and record.get("fixed_input_unchanged") is True
                   and record.get("runtime_after") == record.get("runtime"))
        record.update(phase="finished", execution_status="completed" if closure else "failed",
                      numerical_status="budget_refusal" if closure else "integrity_refusal",
                      refusal=str(error),
                      optimizer_steps_applied=len(record["iterations"]))
        _mark_active_not_committed(record, str(error))
        guard_policy._write(output, record)
        return record
    except guard_policy.StepRefusal as error:
        close(control)
        closure = (record.get("source_unchanged") is True
                   and record.get("fixed_input_unchanged") is True
                   and record.get("runtime_after") == record.get("runtime"))
        record.update(phase="finished", execution_status="completed" if closure else "failed",
                      numerical_status="direction_refusal" if closure else "integrity_refusal",
                      refusal=str(error),
                      optimizer_steps_applied=len(record["iterations"]))
        _mark_active_not_committed(record, str(error))
        guard_policy._write(output, record)
        return record
    except Exception as error:
        close(control)
        record.update(phase="finished", execution_status="failed",
                      numerical_status="continuation_error",
                      refusal=f"{type(error).__name__}: {error}",
                      optimizer_steps_applied=len(record["iterations"]))
        _mark_active_not_committed(record, str(error))
        guard_policy._write(output, record)
        raise


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    if plan_path != PLAN.resolve() or _sha(plan_path) != plan_sha:
        raise ValueError("caller model-guided plan identity mismatch")
    _load_plan(plan_path, plan_sha)
    parent_path = output.with_suffix(".run.json")
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, parent_path)):
        raise ValueError("model-guided report/resource/log paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
               "--plan", str(plan_path), "--plan-sha256", plan_sha, "--output", str(output)]
    resource_result = run_guarded_diagnostic(command, wall_seconds=WALL_SECONDS,
        rss_bytes=RSS_BYTES, report_path=resource, log_path=log)
    execution = guard_policy.execution_status(resource_result)
    parent: dict[str, Any] = {"execution_status": execution, "resource": resource_result,
        "child_sha256": None, "child_read_error": None, "numerical_status": "not_reached"}
    guard_policy._write(parent_path, parent)
    try:
        child = json.loads(output.read_text())
        if not isinstance(child, dict):
            raise ValueError("model-guided child report must be a JSON object")
        parent.update(child_sha256=_sha(output), numerical_status=child.get("numerical_status"))
        closed = (child.get("phase") == "finished" and child.get("execution_status") == "completed"
            and child.get("source_unchanged") is True and child.get("fixed_input_unchanged") is True
            and child.get("runtime_after") == child.get("runtime")
            and child.get("plan_sha256") == plan_sha and child.get("base_control_sha256") == CONTROL_SHA)
        if execution == "completed" and not closed:
            parent.update(execution_status="failed", execution_failure_reason="child closure refused")
        iterations = child.get("iterations", [])
        current_sha = child.get("current_control_sha256")
        count_closed = (isinstance(iterations, list)
            and child.get("optimizer_steps_applied") == len(iterations)
            and child.get("candidate_committed") is (len(iterations) > 0)
            and child.get("active_candidate_committed") is False
            and (iterations or child.get("current_control_sha256") == CONTROL_SHA)
            and (not iterations or (iterations[-1].get("committed") is True
                and iterations[-1].get("accepted_control_sha256") == current_sha
                and iterations[-1].get("accepted_control") == child.get("current_control"))))
        if not count_closed:
            parent.update(execution_status="failed", execution_failure_reason="iteration count does not close")
    except (OSError, ValueError) as error:
        parent.update(execution_status="failed" if execution == "completed" else execution,
            child_read_error=f"{type(error).__name__}: {error}")
        guard_policy._write(parent_path, parent)
        raise RuntimeError(f"model-guided child report could not be verified: {parent_path}") from error
    guard_policy._write(parent_path, parent)
    if parent.get("execution_status") == "failed":
        raise RuntimeError(f"model-guided child failed; receipts: {parent_path}, {output}")
    return {"parent": parent, "child": child}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "model_guided_attempt1/step.json")
    parser.add_argument("--resource", type=Path, default=EVIDENCE / "model_guided_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path, default=EVIDENCE / "model_guided_attempt1/step.log")
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
