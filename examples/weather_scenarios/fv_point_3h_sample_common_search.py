"""One bounded original-J search along the sampled common full-gradient direction."""
from __future__ import annotations

import argparse
import gzip
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

from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as guard_policy
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as face
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "SAMPLE_COMMON_COST_SEARCH_PLAN_20261008.json"
FACE_PLAN = face.PLAN
FACE_PLAN_SHA = "6b79d07ce609ebe35879dcb3f1cec95649a786155c392e7ea041063b4e19283e"
FACE_DIR = EVIDENCE / "face_transition_attempt1"
FACE_ARCHIVE = FACE_DIR / "archive.json"
FACE_DIAGNOSTIC = FACE_DIR / "diagnostic.json.gz"
FACE_PARENT = FACE_DIR / "diagnostic.run.json"
FACE_RESOURCE = FACE_DIR / "diagnostic.resource.json"
FACE_RAW_SHA = "ffa9473e5c0852c8f525665d8e7ac4255ad19820aca9951ef6c137ea905798ce"
FACE_GZIP_SHA = "1a6f02acf0bdfc70466067c73cc804658d4de38c5567be362dd71fd92ff9490e"
MODEL_STEP_SHA = face.MODEL_STEP_SHA
CONTROL_SHA = face.CONTROL_SHA
PARAMETERS_SHA = face.PARAMETERS_SHA
SELF = "examples/weather_scenarios/fv_point_3h_sample_common_search.py"
TEST = "tests/test_fv_point_3h_sample_common_search.py"
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 240.0, 300.0, 1024**3
RADIUS, MAX_BACKTRACKS = 0.05, 16
ALPHA_C1 = guard_policy.block_step.ARMIJO_C1
ETA_MINUS, ETA_PLUS = -1e-6, 1e-6


class SearchRefusal(RuntimeError):
    """Known refusal before a candidate can be committed."""


def policy() -> dict[str, Any]:
    return {"internal_seconds": INTERNAL_SECONDS, "outer_seconds": WALL_SECONDS,
            "rss_bytes": RSS_BYTES, "guarded_launches": 1,
            "candidate_limit": MAX_BACKTRACKS, "hvp_calls": 1,
            "pcg_solves": 0, "optimizer_steps_max": 1,
            "radius": RADIUS, "j_armijo_c1": ALPHA_C1,
            "acceptance": "original_J_Armijo_and_strict_endpoint_branch",
            "phi_is_diagnostic_only": True, "score_computed": False,
            "response_computed": False}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    guard_policy._write(path, value)


def sample_direction(diagnostic: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
    """Reconstruct the minimum-norm point on the two sampled gradient segment."""
    if diagnostic.get("base_control_sha256") != CONTROL_SHA:
        raise ValueError("sampled direction is not attached to the current model control")
    samples = {float(item["requested_eta"]): item for item in diagnostic.get("samples", [])}
    if ETA_MINUS not in samples or ETA_PLUS not in samples:
        raise ValueError("sampled direction requires the fixed inner gradient pair")
    minus = torch.tensor(samples[ETA_MINUS].get("gradient"), dtype=torch.float64)
    plus = torch.tensor(samples[ETA_PLUS].get("gradient"), dtype=torch.float64)
    if (minus.shape != (26,) or plus.shape != (26,)
            or not bool(torch.isfinite(minus).all() & torch.isfinite(plus).all())):
        raise ValueError("sampled gradients must be finite CPU FP64 26-vectors")
    delta = plus - minus
    denominator = torch.dot(delta, delta)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("sampled gradient segment has no resolved length")
    theta = min(1.0, max(0.0, -float(torch.dot(minus, delta) / denominator)))
    common_gradient = minus + theta * delta
    direction = -common_gradient
    norm = float(torch.linalg.vector_norm(direction))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("sampled common direction has no positive finite norm")
    return direction, {"theta": theta, "gradient": common_gradient.tolist(),
        "gradient_norm": norm, "direction": direction.tolist(),
        "direction_norm": norm, "minus_slope": float(torch.dot(minus, direction)),
        "plus_slope": float(torch.dot(plus, direction)),
        "scope": "finite sampled-gradient segment only; direction diagnostic, no root claim"}


def initial_scale(gradient: Tensor, direction: Tensor, h_direction: Tensor,
                  *, radius: float = RADIUS) -> tuple[float, dict[str, float | bool]]:
    """Choose a capped scale; curvature affects the cap only when resolved positive."""
    if (gradient.shape != (26,) or direction.shape != gradient.shape
            or h_direction.shape != gradient.shape
            or any(value.dtype != torch.float64 or value.device.type != "cpu"
                   for value in (gradient, direction, h_direction))
            or not bool(torch.isfinite(gradient).all() & torch.isfinite(direction).all()
                        & torch.isfinite(h_direction).all())
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError("scale inputs must be finite 26-vectors and positive radius")
    direction_norm = float(torch.linalg.vector_norm(direction))
    if not math.isfinite(direction_norm) or direction_norm <= 0:
        raise ValueError("direction norm must be positive and finite")
    gtd, slope_budget = descent_pairing(gradient, direction)
    d_h_d = float(torch.dot(direction, h_direction))
    curvature_budget = 128 * torch.finfo(direction.dtype).eps * max(
        direction_norm * float(torch.linalg.vector_norm(h_direction)),
        torch.finfo(direction.dtype).tiny)
    phi_slope = float(gradient @ h_direction)
    if not all(math.isfinite(x) for x in (d_h_d, curvature_budget, phi_slope)):
        raise SearchRefusal("direction curvature, resolution budget or Phi diagnostic is nonfinite")
    curvature_resolved = d_h_d > curvature_budget
    caps = [1.0, radius / direction_norm]
    if curvature_resolved:
        caps.append(-gtd / d_h_d)
    alpha = min(caps)
    if not math.isfinite(alpha) or alpha <= 0:
        raise SearchRefusal("initial sample-common scale is not positive and finite")
    return alpha, {"g_dot_d": gtd, "g_dot_d_resolution_budget": slope_budget,
        "d_dot_Hd": d_h_d, "g_dot_Hd_diagnostic": phi_slope,
        "curvature_resolution_budget": curvature_budget,
        "curvature_cap_used": curvature_resolved,
        "direction_norm": direction_norm, "initial_alpha": alpha,
        "initial_step_norm": alpha * direction_norm}


def descent_pairing(gradient: Tensor, direction: Tensor) -> tuple[float, float]:
    if (gradient.shape != (26,) or direction.shape != (26,)
            or gradient.dtype != torch.float64 or direction.dtype != torch.float64
            or gradient.device.type != "cpu" or direction.device.type != "cpu"
            or not bool(torch.isfinite(gradient).all() & torch.isfinite(direction).all())):
        raise ValueError("descent pairing requires finite CPU FP64 26-vectors")
    gtd = float(torch.dot(gradient, direction))
    scale = float(torch.linalg.vector_norm(gradient)) * float(torch.linalg.vector_norm(direction))
    budget = 128 * torch.finfo(gradient.dtype).eps * max(scale, torch.finfo(gradient.dtype).tiny)
    if not math.isfinite(gtd) or gtd >= -budget:
        raise SearchRefusal("fresh base gradient has no scale-resolved descent pairing")
    return gtd, budget


def actual_step_metrics(alpha_init: float, helper_alpha: float,
                        direction_norm: float) -> dict[str, float]:
    if (not all(math.isfinite(value) and 0 < value <= 1
                for value in (alpha_init, helper_alpha))
            or not math.isfinite(direction_norm) or direction_norm <= 0):
        raise ValueError("actual step metrics require finite positive bounded scales and norm")
    actual_alpha = alpha_init * helper_alpha
    return {"actual_alpha": actual_alpha,
            "actual_step_norm": actual_alpha * direction_norm}


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    path = path.resolve()
    if path != PLAN.resolve() or path.is_symlink() or not path.is_relative_to(ROOT.resolve()) or _sha(path) != digest:
        raise ValueError("sample-common plan identity mismatch")
    plan = json.loads(path.read_text())
    prior = face._load_plan(FACE_PLAN, FACE_PLAN_SHA)
    expected_sources = set(prior["source_files"]) | {SELF, TEST}
    expected_archives = set(prior["archive_files"]) | {
        FACE_PLAN.relative_to(ROOT).as_posix(), FACE_DIAGNOSTIC.relative_to(ROOT).as_posix(),
        FACE_ARCHIVE.relative_to(ROOT).as_posix(), FACE_PARENT.relative_to(ROOT).as_posix(),
        FACE_RESOURCE.relative_to(ROOT).as_posix()}
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if (plan.get("experiment_kind") != "sample_common_cost_search"
            or plan.get("producing_plan") != FACE_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != FACE_PLAN_SHA
            or plan.get("policy") != policy()
            or plan.get("base_control_sha256") != CONTROL_SHA
            or not isinstance(sources, dict) or not isinstance(archives, dict)
            or set(sources) != expected_sources or set(archives) != expected_archives
            or any(sources.get(name) != digest for name, digest in prior["source_files"].items())
            or any(archives.get(name) != digest for name, digest in prior["archive_files"].items())
            or sources.get(SELF) != _sha(ROOT / SELF) or sources.get(TEST) != _sha(ROOT / TEST)):
        raise ValueError("sample-common scope or lineage pin maps changed")
    if _sha(FACE_PLAN) != FACE_PLAN_SHA:
        raise ValueError("pinned face-transition plan changed")
    for name, value in {**sources, **archives}.items():
        face.model.tangent._pinned_path(name, value)
    if (_sha(FACE_DIAGNOSTIC) != FACE_GZIP_SHA
            or archives.get(FACE_DIAGNOSTIC.relative_to(ROOT).as_posix()) != FACE_GZIP_SHA):
        raise ValueError("pinned face-transition compressed diagnostic changed")
    return plan


def _load_sample_diagnostic() -> tuple[dict[str, Any], dict[str, Any]]:
    archive = json.loads(FACE_ARCHIVE.read_text())
    parent = json.loads(FACE_PARENT.read_text())
    resource = json.loads(FACE_RESOURCE.read_text())
    if (archive.get("raw_sha256") != FACE_RAW_SHA
            or archive.get("archive_sha256") != FACE_GZIP_SHA
            or archive.get("byte_equality") is not True
            or parent.get("execution_status") != "completed"
            or parent.get("child_read_error") is not None
            or parent.get("child_sha256") != FACE_RAW_SHA
            or parent.get("numerical_status") != "finite_two_sided_donor_trace_diagnostic"
            or parent.get("resource") != resource
            or guard_policy.execution_status(resource) != "completed"
            or resource.get("wall_limit_seconds") != WALL_SECONDS
            or resource.get("rss_limit_bytes") != RSS_BYTES
            or resource.get("elapsed_seconds", math.inf) > WALL_SECONDS
            or resource.get("sampled_peak_rss_bytes", math.inf) > RSS_BYTES):
        raise ValueError("face diagnostic archive, completion, or resource receipt is not closed")
    raw_bytes = gzip.decompress(FACE_DIAGNOSTIC.read_bytes())
    if hashlib.sha256(raw_bytes).hexdigest() != FACE_RAW_SHA:
        raise ValueError("decompressed face diagnostic bytes differ from the pinned raw digest")
    diagnostic = json.loads(raw_bytes)
    del raw_bytes
    if (diagnostic.get("phase") != "finished" or diagnostic.get("execution_status") != "completed"
            or diagnostic.get("numerical_status") != "finite_two_sided_donor_trace_diagnostic"
            or diagnostic.get("base_control_sha256") != CONTROL_SHA
            or diagnostic.get("parameters_sha256") != PARAMETERS_SHA
            or diagnostic.get("plan_sha256") != FACE_PLAN_SHA
            or diagnostic.get("source_unchanged") is not True
            or diagnostic.get("source_before") != diagnostic.get("source_after")
            or diagnostic.get("fixed_input_unchanged") is not True
            or diagnostic.get("input_before") != diagnostic.get("input_after")
            or diagnostic.get("runtime") != diagnostic.get("runtime_after")
            or diagnostic.get("optimizer_steps_applied") != 0
            or diagnostic.get("hvp_calls") != 0 or diagnostic.get("pcg_solves") != 0
            or diagnostic.get("score_computed") is not False
            or diagnostic.get("response_computed") is not False):
        raise ValueError("face diagnostic strict closure or read-only scope changed")
    return diagnostic, {"archive": archive, "parent": parent, "resource": resource}


def _branch(problem: Any, control: Tensor, parameters: Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    collector = seed._MarginCollector()
    with torch.no_grad():
        try:
            with face.transport.observe_minmod_stages(collector):
                signature, scope = problem.branch_check(control, parameters)
            analysis_stages = (2 * (len(problem.layout["observation_times_seconds"]) - 1)
                               * problem.frozen.fv_transport.substeps_per_interval)
            partition = face.qy._partition(signature, analysis_stages)
        except ValueError as error:
            return {"status": "branch_refused", "reason": str(error)}, collector.report()
    choices, signs = signature.get("choices", []), signature.get("face_signs", [])
    compact = {"choices": choices, "face_signs": signs}
    branch = {"status": "passed_strict_branch" if len(choices) == 3600 and len(signs) == 3600
              and collector.stages == 3600 and seed._valid_margins(collector.report(), complete=True)
              else "branch_or_margin_refused",
              "euler_stages": signature.get("euler_stages"),
              "choice_stage_count": len(choices), "face_sign_stage_count": len(signs),
              "signature_sha256": face.qy._digest(compact), "scope": scope,
              "partition": partition}
    return branch, collector.report()


def _fresh(problem: Any, control: Tensor, parameters: Tensor,
           gradient_fn: Any, deadline: float) -> tuple[Tensor, Tensor, Tensor]:
    if time.monotonic() >= deadline:
        raise SearchRefusal("internal time budget expired before objective")
    objective = problem.objective(control, parameters)
    if time.monotonic() >= deadline:
        raise SearchRefusal("internal time budget expired after objective")
    gradient = gradient_fn(control, parameters)
    phi = torch.dot(gradient, gradient) / 2
    if (objective.shape != () or objective.dtype != torch.float64 or objective.device.type != "cpu"
            or gradient.shape != (26,) or gradient.dtype != torch.float64
            or gradient.device.type != "cpu" or not bool(torch.isfinite(objective))
            or not bool(torch.isfinite(gradient).all() & torch.isfinite(phi))):
        raise SearchRefusal("fresh original J/full gradient/Phi is nonfinite")
    if time.monotonic() >= deadline:
        raise SearchRefusal("internal time budget expired after full gradient")
    return objective, gradient, phi


def _candidate_objective(problem: Any, base: Tensor, parameters: Tensor):
    def evaluate(candidate: Tensor, p: Tensor) -> Tensor:
        if float(torch.linalg.vector_norm(candidate - base)) > RADIUS * (1 + 64 * torch.finfo(candidate.dtype).eps):
            raise ValueError("candidate exceeded the whole-control radius before FV evaluation")
        return problem.objective(candidate, p)
    return evaluate


def _final_recheck(tentative: dict[str, Any], objective: Tensor, gradient: Tensor,
                   phi: Tensor, branch: dict[str, Any], margins: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {"objective": audit._metric(float(tentative["objective"]), float(objective)),
              "phi_consistency_diagnostic": audit._metric(float(tentative["phi"]), float(phi)),
              "gradient_components": [audit._metric(float(a), float(b)) for a, b in
                  zip(torch.tensor(tentative["gradient"], dtype=torch.float64), gradient, strict=True)]}
    passed = (all(item["passed"] for item in checks["gradient_components"])
              and checks["objective"]["passed"]
              and checks["phi_consistency_diagnostic"]["passed"]
              and branch.get("status") == "passed_strict_branch"
              and branch.get("signature_sha256") == tentative.get("branch", {}).get("signature_sha256")
              and seed._valid_margins(margins, complete=True))
    return {"checks": checks, "passed": passed,
            "phi_role": "consistency diagnostic; decrease is never an acceptance criterion"}


def _mark_uncommitted(record: dict[str, Any], reason: str) -> None:
    if record.get("trials") and record["trials"][-1].get("status") == "accepted":
        record["trials"][-1].update(status="candidate_not_committed", commit_refusal=reason)
    record["optimizer_steps_applied"] = 0
    record["candidate_committed"] = False


def _run_child(plan_path: Path, plan_sha256: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    record: dict[str, Any] = {"schema": "advar.point3h-sample-common-search.v1",
        "phase": "running", "execution_status": "running", "numerical_status": "setup_not_complete",
        "plan_sha256": plan_sha256, "policy": policy(), "samples": None, "trials": [],
        "optimizer_steps_applied": 0, "hvp_calls": 0, "hvp_calls_started": 0,
        "hvp_calls_completed": 0, "pcg_solves": 0,
        "candidate_committed": False,
        "score_computed": False, "response_computed": False, "full_root_claim": False,
        "source_before": None, "source_after": None, "source_unchanged": None,
        "input_before": None, "input_after": None, "fixed_input_unchanged": None,
        "runtime": None, "runtime_after": None}
    _write(output, record)
    source_names: set[str] = set()
    source_before: dict[str, str] = {}
    problem = original = parameters = truth = control = final_control = None
    identity: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None

    def close_partial() -> bool:
        if not source_names or not source_before:
            return False
        source_after = {name: _sha(ROOT / name) for name in sorted(source_names)}
        record["source_after"] = source_after
        record["source_unchanged"] = source_after == source_before
        runtime_after = guard_policy.blocks.runtime_identity()
        record["runtime_after"] = runtime_after
        if (isinstance(original, Tensor) and isinstance(parameters, Tensor)
                and isinstance(truth, Tensor) and isinstance(final_control, Tensor)
                and identity is not None):
            input_after = seed._input_identity(problem, original, final_control, parameters, truth)
            record["input_after"] = input_after
            record["fixed_input_unchanged"] = face.model.tangent._check_fixed_input(
                input_after, identity, _tensor_sha(cast(Tensor, final_control)))
        else:
            record["fixed_input_unchanged"] = False
        return (record["source_unchanged"] is True and runtime_after == runtime
                and record["fixed_input_unchanged"] is True)

    try:
        plan = _load_plan(plan_path, plan_sha256)
        diagnostic, diagnostic_receipts = _load_sample_diagnostic()
        direction, direction_audit = sample_direction(diagnostic)
        del diagnostic
        source_names = set(plan["source_files"]) | set(plan["archive_files"]) | {
            plan_path.resolve().relative_to(ROOT.resolve()).as_posix()}
        source_before = {name: _sha(ROOT / name) for name in sorted(source_names)}
        expected = {**plan["source_files"], **plan["archive_files"],
                    plan_path.resolve().relative_to(ROOT.resolve()).as_posix(): plan_sha256}
        if source_before != expected:
            raise ValueError("sample-common prelaunch source/archive pins changed")
        record["source_before"] = source_before
        model_raw = face._validate_model_source(plan)
        problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
        control = torch.tensor(model_raw["current_control"], dtype=torch.float64)
        final_control = control
        identity = seed._input_identity(problem, original, control, parameters, truth)
        runtime = guard_policy.blocks.runtime_identity()
        record.update(base_control=control.tolist(), base_control_sha256=_tensor_sha(control),
                      input_before=identity, runtime=runtime)
        if (_tensor_sha(control) != CONTROL_SHA or _tensor_sha(parameters) != PARAMETERS_SHA
                or identity != model_raw["input_after"] or runtime != model_raw["runtime"]):
            raise ValueError("reconstructed current model input/control/runtime differs from PR260")
        base_branch, base_margins = _branch(problem, control, parameters)
        if (base_branch.get("status") != "passed_strict_branch"
                or base_branch.get("signature_sha256") != model_raw["current_state"]["branch"]["signature_sha256"]):
            raise SearchRefusal("fresh current base branch differs from the committed model endpoint")
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        base_j, gradient, base_phi = _fresh(problem, control, parameters, gradient_fn, deadline)
        current = model_raw["current_state"]
        base_checks: dict[str, Any] = {"objective": audit._metric(float(current["objective"]), float(base_j)),
            "phi": audit._metric(float(current["phi"]), float(base_phi)),
            "gradient_components": [audit._metric(float(a), float(b)) for a, b in
                zip(torch.tensor(current["gradient"], dtype=torch.float64), gradient, strict=True)]}
        if (not base_checks["objective"]["passed"]
                or not base_checks["phi"]["passed"]
                or not all(item["passed"] for item in base_checks["gradient_components"])):
            raise SearchRefusal("fresh base J/full gradient/Phi differs from the committed model endpoint")
        gtd, slope_budget = descent_pairing(gradient, direction)
        record["sample_direction_pairing"] = {
            "g_dot_d": gtd, "resolution_budget": slope_budget,
            "resolved_descent": True}
        if time.monotonic() >= deadline:
            raise SearchRefusal("internal budget expired before the single fresh HVP")
        record["hvp_calls_started"] = 1
        _write(output, record)
        h_direction = torch.func.jvp(gradient_fn, (control, parameters),
                                     (direction, torch.zeros_like(parameters)))[1]
        record["hvp_calls"] = 1
        record["hvp_calls_completed"] = 1
        _write(output, record)
        if time.monotonic() >= deadline:
            raise SearchRefusal("internal budget expired after the single fresh HVP")
        if h_direction.shape != (26,) or not bool(torch.isfinite(h_direction).all()):
            raise SearchRefusal("fresh HVP direction is nonfinite or malformed")
        alpha_init, scale_audit = initial_scale(gradient, direction, h_direction)
        scaled_step = alpha_init * direction
        if float(torch.linalg.vector_norm(scaled_step)) > RADIUS * (1 + 64 * torch.finfo(control.dtype).eps):
            raise SearchRefusal("initial scaled sample-common step exceeds whole-control radius")
        record.update(base_control=control.tolist(), base_control_sha256=_tensor_sha(control),
            parameters_sha256=_tensor_sha(parameters), base_objective=float(base_j),
            base_gradient=gradient.tolist(), base_phi_diagnostic=float(base_phi),
            base_branch=base_branch, base_margins=base_margins, base_checks=base_checks,
            sample_direction=direction_audit, scale_audit=scale_audit,
            H_direction=h_direction.tolist(),
            diagnostic_receipts=diagnostic_receipts, input_before=identity, runtime=runtime,
            policy={**policy(), "initial_alpha": alpha_init,
                    "actual_alpha": "helper_alpha times initial_alpha",
                    "phi_acceptance_role": "diagnostic_only"})
        _write(output, record)
        strict_branch_fn = lambda point, p: _branch(problem, point, p)

        def record_trial(row: dict[str, Any]) -> None:
            metrics = actual_step_metrics(alpha_init, row["alpha"],
                                          direction_audit["direction_norm"])
            metrics["actual_step_norm"] = float(torch.linalg.vector_norm(
                torch.tensor(row["control"], dtype=control.dtype) - control))
            record["trials"].append({**row, **metrics})
            _write(output, record)

        accepted, trials, status = guard_policy.bounded_original_j_search(
            control, parameters, base_j, gradient, scaled_step,
            _candidate_objective(problem, control, parameters), gradient_fn,
            strict_branch_fn, base_branch["signature_sha256"], deadline=deadline,
            on_trial=record_trial)
        if len(record["trials"]) < len(trials):
            record["trials"].extend(trials[len(record["trials"]):])
        final_control = control
        pending_accept: dict[str, Any] | None = None
        if status == "one_original_J_step_accepted" and accepted is not None:
            tentative = record["trials"][-1]
            final_j, final_g, final_phi = _fresh(problem, accepted, parameters, gradient_fn, deadline)
            final_branch, final_margins = _branch(problem, accepted, parameters)
            final_check = _final_recheck(tentative, final_j, final_g, final_phi,
                                         final_branch, final_margins)
            if not final_check["passed"]:
                raise SearchRefusal("accepted candidate failed independent final J/g/Phi/branch recheck")
            pending_accept = {"numerical_status": status, "optimizer_steps_applied": 1,
                "candidate_committed": True,
                "accepted_control": accepted.tolist(),
                "accepted_control_sha256": _tensor_sha(accepted),
                "accepted_objective": float(final_j), "accepted_gradient": final_g.tolist(),
                "accepted_phi_diagnostic": float(final_phi), "accepted_branch": final_branch,
                "accepted_margins": final_margins, "final_recheck": final_check,
                **actual_step_metrics(alpha_init, float(tentative["alpha"]),
                                      direction_audit["direction_norm"]),
                "actual_step_norm": float(torch.linalg.vector_norm(accepted - control))}
            final_control = accepted
        else:
            record.update(numerical_status=status, refusal=status)
        if time.monotonic() >= deadline:
            raise SearchRefusal("internal budget expired before final source/input/runtime closure")
        if not close_partial():
            raise ValueError("final source/input/runtime closure failed")
        if time.monotonic() >= deadline:
            raise SearchRefusal("internal budget expired before candidate commit")
        if pending_accept is not None:
            record.update(pending_accept)
        record.update(phase="finished", execution_status="completed",
            elapsed_seconds=time.monotonic() - started)
    except (SearchRefusal, guard_policy.StepRefusal, TimeoutError) as error:
        _mark_uncommitted(record, f"{type(error).__name__}: {error}")
        closed = close_partial()
        record.update(phase="finished", execution_status="completed" if closed else "failed",
            numerical_status="search_refusal", refusal=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.monotonic() - started)
    except Exception as error:
        _mark_uncommitted(record, f"{type(error).__name__}: {error}")
        close_partial()
        record.update(phase="finished", execution_status="failed",
            numerical_status="program_error", refusal=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.monotonic() - started)
        _write(output, record)
        raise
    _write(output, record)
    return record


def run(plan_path: Path, plan_sha256: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    _load_plan(plan_path, plan_sha256)
    parent_path = output.with_suffix(".run.json")
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, parent_path)):
        raise ValueError("sample-common report/resource/log paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
        "--plan", str(plan_path), "--plan-sha256", plan_sha256, "--output", str(output),
        "--resource", str(resource), "--log", str(log)]
    resource_result = run_guarded_diagnostic(command, wall_seconds=WALL_SECONDS,
        rss_bytes=RSS_BYTES, report_path=resource, log_path=log)
    parent = {"execution_status": guard_policy.execution_status(resource_result),
        "resource": resource_result, "child_sha256": None, "child_read_error": None,
        "numerical_status": "not_reached"}
    _write(parent_path, parent)
    try:
        child = json.loads(output.read_text())
        if not isinstance(child, dict):
            raise ValueError("sample-common child report must be a JSON object")
        parent.update(child_sha256=_sha(output), numerical_status=child.get("numerical_status"))
        closed = (child.get("phase") == "finished"
            and child.get("execution_status") == "completed"
            and child.get("source_unchanged") is True
            and child.get("fixed_input_unchanged") is True
            and child.get("runtime_after") == child.get("runtime")
            and child.get("plan_sha256") == plan_sha256
            and child.get("base_control_sha256") == CONTROL_SHA)
        if parent["execution_status"] == "completed" and not closed:
            parent.update(execution_status="failed", execution_failure_reason="child identity closure failed")
    except (OSError, ValueError) as error:
        parent.update(execution_status="failed" if parent["execution_status"] == "completed"
                      else parent["execution_status"],
                      child_read_error=f"{type(error).__name__}: {error}")
        _write(parent_path, parent)
        raise RuntimeError(f"sample-common report could not be verified: {parent_path}") from error
    _write(parent_path, parent)
    if parent["execution_status"] == "failed":
        raise RuntimeError(f"sample-common guarded run failed; receipts: {parent_path}, {output}")
    return {"parent": parent, "child": child}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        run(args.plan, args.plan_sha256, args.output, args.resource, args.log)


if __name__ == "__main__":
    main()
