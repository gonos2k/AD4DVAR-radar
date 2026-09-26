"""Bounded relinearized joint J/Phi continuation from the pinned PR #221 point.

This exploratory search checkpoints every endpoint trial. It does not certify
the connecting path, a root, or a response.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios import fv_point_3h_first_branch_probe as first_branch
from examples.weather_scenarios import fv_point_3h_merit_step_probe as merit_step
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed_linear

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_3H_MERIT_CONTINUATION_PLAN.md"
SEED_REPORT = EVIDENCE / "point_3h_merit_step_attempt1/point_3h_merit_step.json"
SEED_EVIDENCE = EVIDENCE / "FV_POINT_3H_MERIT_STEP_EVIDENCE.json"
PLAN_SHA256 = "33a3b3d558b83711d066bea8e3d2d92de4779e630811b2d4b2a2d7246e0ba521"
SEED_REPORT_SHA256 = "024b89d346ae0136542888527d3bb528d7f638c1e3be44f3aa7cdb2232d10f3a"
SEED_EVIDENCE_SHA256 = "df026542d85910a61a9181ae46df8f5c7e6ac8d8bfa80dbc318e33d429b85a8a"
SEED_CONTROL_SHA256 = "9f8d3ae0d77d8b24f1565ab622cf8fe7bbfd83766fbd67964a6a5bd52920a078"
SEED_SIGNATURE_SHA256 = "c1d60b87479fea5937ae44d4740cb0de3adf10e62fc50018cab69abb12ca47bb"
SEED_J = 0.9319504904669574
SEED_PHI = 2.6187151084118883
SEED_GRADIENT_INF = 1.2605349625093007
EXPECTED_CONTROLS = 26
EXPECTED_STAGES = 3600
EXPECTED_PARAMETERS = 13
MAX_ACCEPTED_EPOCHS = 8
MAX_TOTAL_TRIALS = 64
MAX_TRIALS_PER_EPOCH = 8
INITIAL_ALPHA = 0.02
ARMijo_C1 = 1.0e-4
ROUNDING_MULTIPLIER = 128.0
SYMMETRY_TOLERANCE = 1.0e-8
INTERNAL_BUDGET_SECONDS = 540.0
WALL_SECONDS = 600
REAP_GRACE_SECONDS = 2
SAMPLED_RSS_BYTES = 1024**3
DEFAULT_DIRECTORY = EVIDENCE / "point_3h_merit_continuation_attempt1"

SOURCE_PATHS = tuple(dict.fromkeys((
    *seed_linear.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_3h_merit_step_probe.py",
    "examples/weather_scenarios/fv_point_3h_merit_continuation.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _source_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _finite_tensor(value: object, reference: Tensor) -> bool:
    return (isinstance(value, Tensor) and value.shape == reference.shape
            and value.dtype == reference.dtype and value.device == reference.device
            and bool(torch.isfinite(value).all()))


def _finite_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _continuation_link(current: dict[str, Any],
                       previous: dict[str, Any] | None) -> bool:
    """Bind each fresh epoch point to the pinned seed or prior accepted trial."""
    if previous is None:
        expected = {
            "control_sha256": SEED_CONTROL_SHA256,
            "branch_signature_sha256": SEED_SIGNATURE_SHA256,
            "objective": SEED_J, "phi": SEED_PHI,
            "gradient_inf": SEED_GRADIENT_INF,
        }
    else:
        expected = {
            "control_sha256": previous.get("control_sha256"),
            "branch_signature_sha256": previous.get("branch_signature_sha256"),
            "objective": previous.get("objective"), "phi": previous.get("phi"),
            "gradient_inf": previous.get("gradient_inf"),
        }
    return all(current.get(key) == value for key, value in expected.items())


def _budget_exhausted(started: float, now: float | None = None) -> bool:
    return (time.monotonic() if now is None else now) - started >= INTERNAL_BUDGET_SECONDS


def _trial_slots(total_trials: int) -> int:
    return min(MAX_TRIALS_PER_EPOCH, max(0, MAX_TOTAL_TRIALS - total_trials))


def _transpose_consistency(hg: Tensor, htg: Tensor) -> dict[str, Any]:
    """Check one exact JVP/VJP pair at a single current control."""
    record: dict[str, Any] = {"status": "refused", "tolerance": SYMMETRY_TOLERANCE}
    if (hg.ndim != 1 or htg.shape != hg.shape or hg.dtype != torch.float64
            or htg.dtype != hg.dtype or hg.device.type != "cpu"
            or htg.device != hg.device or not bool(torch.isfinite(hg).all())
            or not bool(torch.isfinite(htg).all())):
        record["reason"] = "invalid_or_nonfinite_hessian_products"
        return record
    difference = torch.linalg.vector_norm(hg - htg)
    hg_norm = torch.linalg.vector_norm(hg)
    htg_norm = torch.linalg.vector_norm(htg)
    denominator = torch.maximum(torch.maximum(hg_norm, htg_norm),
                                torch.as_tensor(torch.finfo(hg.dtype).tiny, dtype=hg.dtype))
    ratio = difference / denominator
    values = (float(difference), float(hg_norm), float(htg_norm), float(ratio))
    record.update(difference_l2=values[0] if math.isfinite(values[0]) else None,
                  hg_l2=values[1] if math.isfinite(values[1]) else None,
                  htg_l2=values[2] if math.isfinite(values[2]) else None,
                  relative_difference=values[3] if math.isfinite(values[3]) else None)
    if not all(math.isfinite(value) for value in values):
        record["reason"] = "nonfinite_transpose_comparison"
    elif values[3] > SYMMETRY_TOLERANCE:
        record["reason"] = "transpose_consistency_failed"
    else:
        record.update(status="passed", reason=None)
    return record


def _full_current_branch(problem: Any, control: Tensor,
                         parameters: Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the seed-linear helper's separate full margin pass and strict oracle."""
    branch, margins = seed_linear._full_branch_with_margins(
        problem, control, parameters,
    )
    valid_branch = (
        branch.get("status") == "passed_strict_branch"
        and branch.get("euler_stages") == EXPECTED_STAGES
        and branch.get("choice_stage_count") == EXPECTED_STAGES
        and branch.get("face_sign_stage_count") == EXPECTED_STAGES
    )
    valid_margins = seed_linear._valid_margins(margins, complete=True)
    if not valid_branch or not valid_margins:
        return {**branch, "status": "branch_or_margin_refused"}, margins
    return branch, margins


def _fresh_merit(problem: Any, control: Tensor, parameters: Tensor,
                 gradient_fn: Any) -> tuple[Tensor, Tensor, Tensor] | None:
    objective = problem.objective(control, parameters)
    gradient = gradient_fn(control, parameters)
    phi = torch.dot(gradient, gradient) / 2.0 if isinstance(gradient, Tensor) else None
    if (not isinstance(objective, Tensor) or objective.shape != ()
            or not _finite_tensor(gradient, control)
            or not isinstance(phi, Tensor) or phi.shape != ()
            or not bool(torch.isfinite(objective)) or not bool(torch.isfinite(phi))):
        return None
    return objective, gradient, phi


def _branch_is_complete(signature: dict[str, Any]) -> bool:
    return (signature.get("euler_stages") == EXPECTED_STAGES
            and len(signature.get("choices", [])) == EXPECTED_STAGES
            and len(signature.get("face_signs", [])) == EXPECTED_STAGES)


def _strict_endpoint(problem: Any, control: Tensor,
                     parameters: Tensor) -> dict[str, Any]:
    """Run the original complete strict endpoint oracle once."""
    try:
        signature, scope = problem.branch_check(control, parameters)
    except ValueError as error:
        return {"status": "branch_refused", "reason": str(error)}
    if not _branch_is_complete(signature):
        return {"status": "branch_incomplete",
                "euler_stages": signature.get("euler_stages"),
                "choice_stage_count": len(signature.get("choices", [])),
                "face_sign_stage_count": len(signature.get("face_signs", []))}
    return {"status": "passed_strict_branch",
            "euler_stages": EXPECTED_STAGES,
            "choice_stage_count": EXPECTED_STAGES,
            "face_sign_stage_count": EXPECTED_STAGES,
            "signature_sha256": merit_step._signature_sha(signature),
            "scope": scope}


def _seed_control() -> Tensor:
    if _sha(SEED_REPORT) != SEED_REPORT_SHA256:
        raise ValueError("pinned PR #221 raw report SHA256 changed")
    raw = json.loads(SEED_REPORT.read_text())
    if (raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "candidate_accepted_branch_changed"
            or raw.get("accepted_candidate", {}).get("control_sha256") != SEED_CONTROL_SHA256
            or raw.get("accepted_candidate", {}).get("branch_signature_sha256") != SEED_SIGNATURE_SHA256
            or raw.get("accepted_candidate", {}).get("objective") != SEED_J
            or raw.get("accepted_candidate", {}).get("phi") != SEED_PHI
            or raw.get("accepted_candidate", {}).get("gradient_inf") != SEED_GRADIENT_INF):
        raise ValueError("pinned PR #221 accepted endpoint evidence changed")
    control = torch.as_tensor(raw["accepted_candidate"]["control"], dtype=torch.float64)
    if control.shape != (EXPECTED_CONTROLS,) or seed_linear._tensor_sha(control) != SEED_CONTROL_SHA256:
        raise ValueError("pinned PR #221 control shape, dtype, or SHA256 changed")
    return control


def run_probe(output: Path) -> dict[str, Any]:
    """Run the bounded relinearized endpoint search inside the one guarded child."""
    started = time.monotonic()
    report: dict[str, Any] = {
        "pid": os.getpid(), "execution_status": "running",
        "execution_phase": "source_check", "numerical_status": "not_reached",
        "accepted_epochs": [], "candidate_attempts": [], "accepted_candidate": None,
        "response_computed": False, "adjoint_computed": False,
        "score_computed": False, "pcg_calls": 0,
        "path_certified": False, "root_certified": False,
        "plan_sha256": PLAN_SHA256, "seed_report_sha256": SEED_REPORT_SHA256,
        "seed_evidence_sha256": SEED_EVIDENCE_SHA256,
        "seed_control_sha256": SEED_CONTROL_SHA256,
        "wall_limit_seconds": WALL_SECONDS,
        "termination_reap_grace_seconds": REAP_GRACE_SECONDS,
        "rss_limit_bytes": SAMPLED_RSS_BYTES,
        "internal_budget_seconds": INTERNAL_BUDGET_SECONDS,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
    }
    _write_json(output, report)
    source_before: dict[str, str] = {}
    archive_before = input_before = None
    problem = original = control = parameters = truth = None
    branch_oracle_calls = 0
    current_branch_calls = 0
    current_branch_call_unrecorded = False
    exact_hvp_calls = exact_vjp_calls = 0
    phase_seconds: dict[str, float] = {}
    try:
        tick = time.monotonic()
        source_before = _source_hashes()
        if (not first_branch._sources_match_archive(source_before)
                or _sha(PLAN) != PLAN_SHA256
                or _sha(SEED_REPORT) != SEED_REPORT_SHA256
                or _sha(SEED_EVIDENCE) != SEED_EVIDENCE_SHA256):
            raise ValueError("pinned source, continuation plan, or PR #221 report changed")
        archive_before = first_branch._archive_input_identity()
        if _canonical_sha(archive_before) != seed_linear.ARCHIVED_INPUT_SHA256:
            raise ValueError("fixed archived input identity changed")
        phase_seconds["source_and_archive_identity"] = time.monotonic() - tick

        tick = time.monotonic()
        problem, original, base_control, parameters, truth, input_before = seed_linear._prepare_fixed_seed()
        control = _seed_control()
        if (base_control.shape != control.shape or control.dtype != torch.float64
                or parameters.shape != (EXPECTED_PARAMETERS,)
                or problem.layout["controls"] != EXPECTED_CONTROLS
                or problem.layout["parameters"] != EXPECTED_PARAMETERS
                or problem.layout["euler_stages"] != EXPECTED_STAGES
                or problem.frozen.nowcast_config.forecast_steps != 18):
            raise ValueError("fixed PR #204 3-hour problem layout changed")
        input_before = seed_linear._input_identity(problem, original, control, parameters, truth)
        field_count = int(problem.frozen.active_field_index.numel())
        if field_count != 20:
            raise ValueError("fixed PR #204 latent control partition changed")
        report.update(input_before=input_before, archive_before=archive_before,
                      layout=problem.layout,
                      pinned_seed={"control_sha256": seed_linear._tensor_sha(control),
                                   "parameters_sha256": seed_linear._tensor_sha(parameters),
                                   "terminal_truth_sha256": seed_linear._tensor_sha(truth),
                                   "objective": SEED_J, "phi": SEED_PHI,
                                   "gradient_inf": SEED_GRADIENT_INF,
                                   "branch_signature_sha256": SEED_SIGNATURE_SHA256})
        phase_seconds["fixture_reconstruction"] = time.monotonic() - tick
        _write_json(output, report)

        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        current_metrics: dict[str, Any] | None = None
        for epoch in range(MAX_ACCEPTED_EPOCHS):
            if len(report["candidate_attempts"]) >= MAX_TOTAL_TRIALS:
                report["numerical_status"] = "trial_limit"
                break
            if _budget_exhausted(started):
                report["numerical_status"] = "internal_budget_exhausted"
                break
            report.update(execution_phase="strict_current_branch", current_epoch=epoch)
            report.pop("current", None)
            report["current_control_sha256"] = seed_linear._tensor_sha(control)
            tick = time.monotonic()
            branch_oracle_calls += 1
            current_branch_calls += 1
            current_branch_call_unrecorded = True
            branch, margins = _full_current_branch(problem, control, parameters)
            phase_seconds["strict_branch_and_margin_pass"] = phase_seconds.get(
                "strict_branch_and_margin_pass", 0.0) + time.monotonic() - tick
            report.update(current_branch=branch, current_branch_margins=margins,
                          branch_status=branch.get("status"))
            if _budget_exhausted(started):
                report["numerical_status"] = "internal_budget_exhausted"
                _write_json(output, report)
                break
            if branch.get("status") != "passed_strict_branch":
                report["numerical_status"] = "current_branch_or_margin_refused"
                break
            current_signature_sha = branch["signature_sha256"]
            tick = time.monotonic()
            metrics = _fresh_merit(problem, control, parameters, gradient_fn)
            phase_seconds["fresh_current_merit"] = phase_seconds.get("fresh_current_merit", 0.0) + time.monotonic() - tick
            if _budget_exhausted(started):
                report["numerical_status"] = "internal_budget_exhausted"
                _write_json(output, report)
                break
            if metrics is None:
                report["numerical_status"] = "current_merit_refused"
                break
            objective, gradient, phi = metrics
            current_metrics = {"objective": float(objective), "phi": float(phi),
                               "gradient_l2": float(torch.linalg.vector_norm(gradient)),
                               "gradient_inf": float(gradient.abs().max()),
                               "control_sha256": seed_linear._tensor_sha(control),
                               "branch_signature_sha256": current_signature_sha}
            if epoch == 0 and not _continuation_link(current_metrics, None):
                report["numerical_status"] = "pinned_seed_merit_mismatch"
                break
            report["current"] = current_metrics
            if current_metrics["gradient_inf"] < 1.0e-4:
                report["numerical_status"] = "handoff_candidate_only"
                break

            tick = time.monotonic()
            exact_hvp_calls += 1
            hg = torch.func.jvp(lambda value: gradient_fn(value, parameters),
                                (control,), (gradient,))[1]
            exact_vjp_calls += 1
            pullback = torch.func.vjp(
                lambda value: gradient_fn(value, parameters), control,
            )[1]
            htg = pullback(gradient)[0]
            phase_seconds["fresh_hg_and_htg"] = phase_seconds.get("fresh_hg_and_htg", 0.0) + time.monotonic() - tick
            if _budget_exhausted(started):
                report["numerical_status"] = "internal_budget_exhausted"
                _write_json(output, report)
                break
            transpose = _transpose_consistency(hg, htg)
            report["transpose_consistency"] = transpose
            if transpose["status"] != "passed":
                report["numerical_status"] = "transpose_consistency_refused"
                break
            direction, direction_record = merit_step._joint_direction(gradient, hg)
            report["direction"] = direction_record
            if direction is None:
                report["numerical_status"] = "no_joint_descent_direction"
                break
            # Use the independently computed VJP slope in the declared Phi slope.
            slope_phi = torch.dot(htg, direction)
            slope_j = torch.dot(gradient, direction)
            report["true_slope_j"] = (
                float(slope_j) if bool(torch.isfinite(slope_j)) else None
            )
            report["true_slope_phi"] = (
                float(slope_phi) if bool(torch.isfinite(slope_phi)) else None
            )
            if (not bool(torch.isfinite(slope_phi)) or not bool(torch.isfinite(slope_j))
                    or float(slope_phi) >= 0.0 or float(slope_j) >= 0.0):
                report["true_slope_refusal_reason"] = (
                    "nonfinite_true_slope" if not bool(torch.isfinite(slope_phi))
                    or not bool(torch.isfinite(slope_j)) else "true_slope_not_descent"
                )
                report["numerical_status"] = "true_slope_refused"
                break
            epoch_record: dict[str, Any] = {
                "epoch": epoch, "current": current_metrics,
                "current_branch": branch, "current_branch_margins": margins,
                "direction": direction_record,
                "transpose_consistency": transpose,
                "true_slope_j": float(slope_j), "true_slope_phi": float(slope_phi),
                "trials": [], "accepted": False,
            }
            report["accepted_epochs"].append(epoch_record)
            current_branch_call_unrecorded = False
            accepted = False
            for index in range(_trial_slots(len(report["candidate_attempts"]))):
                if _budget_exhausted(started):
                    report["numerical_status"] = "internal_budget_exhausted"
                    break
                alpha = INITIAL_ALPHA / (2 ** index)
                candidate_control = control + alpha * direction
                attempt: dict[str, Any] = {
                    "epoch": epoch, "trial": index, "alpha": alpha,
                    "control_sha256": seed_linear._tensor_sha(candidate_control),
                    "branch_status": "not_checked", "accepted": False,
                    "path_certified": False,
                }
                report["execution_phase"] = "endpoint_trial"
                if not _finite_tensor(candidate_control, control):
                    attempt.update(branch_status="refused_nonfinite_control",
                                   refusal_reason="candidate control is invalid or nonfinite")
                else:
                    tick = time.monotonic()
                    branch_oracle_calls += 1
                    branch = _strict_endpoint(problem, candidate_control, parameters)
                    phase_seconds["strict_endpoint_oracle"] = phase_seconds.get(
                        "strict_endpoint_oracle", 0.0) + time.monotonic() - tick
                    attempt.update(branch=branch, branch_status=branch.get("status"))
                    if branch.get("status") == "passed_strict_branch":
                        attempt["branch_signature_sha256"] = branch.get("signature_sha256")
                    if _budget_exhausted(started):
                        attempt.update(objective_status="not_evaluated_internal_budget",
                                       refusal_reason="internal phase budget expired after strict endpoint check")
                        report["numerical_status"] = "internal_budget_exhausted"
                    elif branch.get("status") == "passed_strict_branch":
                        candidate_signature_sha = branch["signature_sha256"]
                        if not isinstance(candidate_signature_sha, str) or len(candidate_signature_sha) != 64:
                            attempt.update(branch_status="incomplete_strict_branch",
                                           refusal_reason="strict endpoint signature is incomplete")
                        else:
                            same_signature = candidate_signature_sha == current_signature_sha
                            attempt["branch_signature_sha256"] = candidate_signature_sha
                            tick = time.monotonic()
                            candidate_metrics = _fresh_merit(problem, candidate_control,
                                                             parameters, gradient_fn)
                            phase_seconds["fresh_candidate_merit"] = phase_seconds.get("fresh_candidate_merit", 0.0) + time.monotonic() - tick
                            if _budget_exhausted(started):
                                report["numerical_status"] = "internal_budget_exhausted"
                                if candidate_metrics is None:
                                    attempt.update(objective_status="nonfinite_or_invalid",
                                                   refusal_reason="endpoint merit invalid at phase budget boundary")
                                else:
                                    candidate_j, candidate_gradient, candidate_phi = candidate_metrics
                                    attempt.update(
                                        objective_status="complete_after_budget",
                                        objective=float(candidate_j), phi=float(candidate_phi),
                                        gradient_l2=float(torch.linalg.vector_norm(candidate_gradient)),
                                        gradient_inf=float(candidate_gradient.abs().max()),
                                        refusal_reason="internal phase budget expired before acceptance gate",
                                    )
                            elif candidate_metrics is None:
                                attempt.update(objective_status="nonfinite_or_invalid",
                                               refusal_reason="fresh endpoint J/g/Phi contract failed")
                            else:
                                candidate_j, candidate_gradient, candidate_phi = candidate_metrics
                                gate = merit_step._candidate_gate(
                                    seed_j=float(objective), seed_phi=float(phi),
                                    candidate_j=float(candidate_j), candidate_phi=float(candidate_phi),
                                    slope_j=float(slope_j), slope_phi=float(slope_phi),
                                    alpha=alpha, same_signature=same_signature,
                                )
                                attempt.update(
                                    objective_status="finite", objective=float(candidate_j),
                                    phi=float(candidate_phi),
                                    gradient_l2=float(torch.linalg.vector_norm(candidate_gradient)),
                                    gradient_inf=float(candidate_gradient.abs().max()),
                                    branch_changed=not same_signature, gate=gate,
                                    accepted=gate["accepted"],
                                )
                                if gate["accepted"]:
                                    accepted = True
                                    epoch_record["accepted"] = True
                                    epoch_record["accepted_trial"] = index
                                    control = candidate_control
                                    report["accepted_candidate"] = {
                                        "epoch": epoch, "trial": index, "alpha": alpha,
                                        "control": control.tolist(),
                                        "control_sha256": seed_linear._tensor_sha(control),
                                        "branch_signature_sha256": candidate_signature_sha,
                                        "objective": float(candidate_j), "phi": float(candidate_phi),
                                        "gradient_inf": float(candidate_gradient.abs().max()),
                                        "branch_changed": not same_signature,
                                        "path_certified": False,
                                    }
                report["candidate_attempts"].append(attempt)
                epoch_record["trials"].append(attempt)
                _write_json(output, report)
                if accepted:
                    break
            _write_json(output, report)
            if report.get("numerical_status") == "internal_budget_exhausted":
                break
            if not accepted:
                report["numerical_status"] = (
                    "trial_limit" if len(report["candidate_attempts"]) >= MAX_TOTAL_TRIALS
                    else "epoch_line_search_refused"
                )
                break
            accepted_record = report.get("accepted_candidate")
            if (isinstance(accepted_record, dict)
                    and accepted_record["gradient_inf"] < 1.0e-4):
                report["numerical_status"] = "handoff_candidate_only"
                break
        else:
            report["numerical_status"] = "accepted_epoch_limit"
        if (report["accepted_candidate"] is not None
                and report["numerical_status"] in {
                    "accepted_epoch_limit", "handoff_candidate_only", "trial_limit",
                }
                and not _budget_exhausted(started)):
            report.update(execution_phase="final_accepted_branch")
            tick = time.monotonic()
            branch_oracle_calls += 1
            final_branch, final_margins = _full_current_branch(
                problem, control, parameters,
            )
            phase_seconds["strict_branch_and_margin_pass"] = phase_seconds.get(
                "strict_branch_and_margin_pass", 0.0) + time.monotonic() - tick
            report["final_accepted_branch"] = final_branch
            report["final_accepted_branch_margins"] = final_margins
            if _budget_exhausted(started):
                report["numerical_status"] = "internal_budget_exhausted"
            elif final_branch.get("status") != "passed_strict_branch":
                report["numerical_status"] = "final_accepted_branch_or_margin_refused"
        elif (report["accepted_candidate"] is not None
              and report["numerical_status"] in {
                  "accepted_epoch_limit", "handoff_candidate_only", "trial_limit",
              }):
            report["numerical_status"] = "internal_budget_exhausted"
        if report["numerical_status"] == "not_reached":
            report["numerical_status"] = (
                "trial_limit" if len(report["candidate_attempts"]) >= MAX_TOTAL_TRIALS
                else "accepted_epoch_limit"
            )
        report.update(execution_status="completed", execution_phase="finished")
    except Exception as error:
        report.update(execution_status="failed", execution_phase="failed",
                      numerical_status="continuation_error",
                      error=f"{type(error).__name__}: {error}")
    finally:
        report["child_elapsed_seconds"] = time.monotonic() - started
        report["child_phase_seconds"] = phase_seconds
        report["costs"] = {
            "accepted_epoch_count": sum(bool(item.get("accepted"))
                                         for item in report["accepted_epochs"]),
            "epoch_count": len(report["accepted_epochs"]),
            "trial_count": len(report["candidate_attempts"]),
            "branch_oracle_calls": branch_oracle_calls,
            "current_branch_calls": current_branch_calls,
            "current_branch_call_unrecorded": current_branch_call_unrecorded,
            "exact_hvp_calls": exact_hvp_calls, "exact_vjp_calls": exact_vjp_calls,
            "response_computed": False, "adjoint_computed": False,
            "score_computed": False, "pcg_calls": 0,
        }
        report["source_before"] = source_before
        report["source_after"] = _source_hashes()
        report["source_unchanged"] = report["source_after"] == source_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["seed_report_unchanged"] = _sha(SEED_REPORT) == SEED_REPORT_SHA256
        report["seed_evidence_unchanged"] = _sha(SEED_EVIDENCE) == SEED_EVIDENCE_SHA256
        try:
            report["archive_unchanged"] = archive_before is not None and first_branch._archive_input_identity() == archive_before
        except (OSError, ValueError, json.JSONDecodeError):
            report["archive_unchanged"] = False
        if (problem is not None and isinstance(original, Tensor)
                and isinstance(control, Tensor) and isinstance(parameters, Tensor)
                and isinstance(truth, Tensor) and isinstance(input_before, dict)):
            after_identity = seed_linear._input_identity(problem, original, control,
                                                         parameters, truth)
            report["input_after"] = after_identity
            # The control is intentionally updated by the continuation.
            report["fixed_input_unchanged"] = (
                after_identity.get("parameters_sha256") == input_before.get("parameters_sha256")
                and after_identity.get("terminal_truth_sha256") == input_before.get("terminal_truth_sha256")
                and after_identity.get("archived_input") == input_before.get("archived_input")
            )
        else:
            report["input_after"] = None
            report["fixed_input_unchanged"] = False
        if not all(report.get(key) is True for key in (
            "source_unchanged", "plan_unchanged", "seed_report_unchanged",
            "seed_evidence_unchanged",
            "archive_unchanged", "fixed_input_unchanged",
        )):
            report.update(execution_status="failed", execution_phase="identity_refused",
                          numerical_status="identity_changed")
        _write_json(output, report)
    return report


def _valid_resource(resource: dict[str, Any], command: list[str]) -> bool:
    elapsed, peak = resource.get("elapsed_seconds"), resource.get("sampled_peak_rss_bytes")
    return (resource.get("command") == command and resource.get("exit_code") == 0
            and resource.get("resource_termination") is None
            and resource.get("monitor_error") is None
            and resource.get("wall_limit_seconds") == WALL_SECONDS
            and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
            and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
            and type(peak) is int and 0 < peak <= SAMPLED_RSS_BYTES
            and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
            and math.isfinite(elapsed) and 0 <= elapsed <= WALL_SECONDS)


def _valid_child(child: object, sources: dict[str, str]) -> bool:
    if not isinstance(child, dict):
        return False
    costs = child.get("costs")
    attempts = child.get("candidate_attempts")
    epochs = child.get("accepted_epochs")
    if not isinstance(costs, dict) or not isinstance(attempts, list) or not isinstance(epochs, list):
        return False
    known_statuses = {
        "pinned_seed_merit_mismatch",
        "handoff_candidate_only", "transpose_consistency_refused",
        "no_joint_descent_direction", "true_slope_refused",
        "current_branch_or_margin_refused", "current_merit_refused",
        "internal_budget_exhausted", "epoch_line_search_refused",
        "trial_limit", "accepted_epoch_limit",
        "final_accepted_branch_or_margin_refused",
    }
    if not (child.get("execution_status") == "completed"
            and child.get("execution_phase") == "finished"
            and child.get("numerical_status") in known_statuses
            and child.get("source_before") == child.get("source_after") == sources
            and child.get("source_unchanged") is True
            and child.get("plan_sha256") == PLAN_SHA256 and child.get("plan_unchanged") is True
            and child.get("seed_report_sha256") == SEED_REPORT_SHA256
            and child.get("seed_report_unchanged") is True
            and child.get("seed_evidence_sha256") == SEED_EVIDENCE_SHA256
            and child.get("seed_evidence_unchanged") is True
            and child.get("seed_control_sha256") == SEED_CONTROL_SHA256
            and child.get("archive_unchanged") is True
            and child.get("fixed_input_unchanged") is True
            and child.get("response_computed") is False
            and child.get("adjoint_computed") is False
            and child.get("score_computed") is False and child.get("pcg_calls") == 0
            and child.get("path_certified") is False and child.get("root_certified") is False
            and type(child.get("pid")) is int and child["pid"] > 0
            and child.get("wall_limit_seconds") == WALL_SECONDS
            and child.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
            and len(epochs) <= MAX_ACCEPTED_EPOCHS and len(attempts) <= MAX_TOTAL_TRIALS
            and costs.get("accepted_epoch_count") == sum(
                item.get("accepted") is True for item in epochs if isinstance(item, dict))
            and costs.get("epoch_count") == len(epochs)
            and costs.get("trial_count") == len(attempts)
            and type(costs.get("current_branch_calls")) is int
            and costs["current_branch_calls"] == (
                len(epochs) + int(costs.get("current_branch_call_unrecorded") is True))
            and type(costs.get("current_branch_call_unrecorded")) is bool
            and (costs.get("current_branch_call_unrecorded") is False
                 or child.get("numerical_status") in {
                     "pinned_seed_merit_mismatch", "current_branch_or_margin_refused",
                     "current_merit_refused", "transpose_consistency_refused",
                     "no_joint_descent_direction", "true_slope_refused",
                     "handoff_candidate_only", "internal_budget_exhausted",
                 })
            and (costs.get("current_branch_call_unrecorded") is False
                 or child.get("current_epoch") == len(epochs))
            and type(costs.get("branch_oracle_calls")) is int
            and costs["branch_oracle_calls"] == (
                costs["current_branch_calls"]
                + sum(item.get("branch_status") != "refused_nonfinite_control"
                      for item in attempts)
                + int("final_accepted_branch" in child))
            and type(costs.get("exact_hvp_calls")) is int
            and costs.get("exact_hvp_calls") == costs.get("exact_vjp_calls")
            and len(epochs) <= costs["exact_hvp_calls"] <= len(epochs) + 1):
        return False
    if any(not isinstance(item, dict) or item.get("path_certified") is not False for item in attempts):
        return False
    epoch_trials: list[dict[str, Any]] = []
    last_accepted: dict[str, Any] | None = None
    expected_trial_count = 0
    previous_accepted: dict[str, Any] | None = None
    for epoch_index, epoch in enumerate(epochs):
        if (not isinstance(epoch, dict) or epoch.get("epoch") != epoch_index
                or not isinstance(epoch.get("current"), dict)
                or not isinstance(epoch.get("trials"), list)
                or len(epoch["trials"]) > MAX_TRIALS_PER_EPOCH
                or type(epoch.get("accepted")) is not bool
                or (len(epoch["trials"]) == 0
                    and not (child.get("numerical_status") == "internal_budget_exhausted"
                             and epoch_index == len(epochs) - 1
                             and epoch.get("accepted") is False))):
            return False
        current = epoch["current"]
        if not _continuation_link(current, previous_accepted):
            return False
        current_branch = epoch.get("current_branch")
        if (not isinstance(current_branch, dict)
                or current_branch.get("status") != "passed_strict_branch"
                or current_branch.get("signature_sha256")
                != current.get("branch_signature_sha256")
                or not seed_linear._valid_margins(
                    epoch.get("current_branch_margins"), complete=True)):
            return False
        direction = epoch.get("direction")
        transpose = epoch.get("transpose_consistency")
        if (not isinstance(direction, dict) or direction.get("status") != "passed"
                or not isinstance(transpose, dict) or transpose.get("status") != "passed"
                or not _finite_number(transpose.get("relative_difference"))
                or transpose["relative_difference"] > SYMMETRY_TOLERANCE):
            return False
        slope_j, slope_phi = epoch.get("true_slope_j"), epoch.get("true_slope_phi")
        signature = current.get("branch_signature_sha256")
        if (not all(_finite_number(current.get(name)) for name in
                    ("objective", "phi", "gradient_l2", "gradient_inf"))
                or not isinstance(slope_j, (int, float)) or isinstance(slope_j, bool)
                or not math.isfinite(slope_j) or slope_j >= 0
                or not isinstance(slope_phi, (int, float)) or isinstance(slope_phi, bool)
                or not math.isfinite(slope_phi) or slope_phi >= 0
                or not isinstance(signature, str) or len(signature) != 64):
            return False
        accepted_in_epoch = []
        for trial_index, trial in enumerate(epoch["trials"]):
            if (not isinstance(trial, dict) or trial.get("epoch") != epoch_index
                    or trial.get("trial") != trial_index
                    or trial.get("alpha") != INITIAL_ALPHA / 2**trial_index
                    or trial.get("path_certified") is not False):
                return False
            global_index = expected_trial_count + trial_index
            if global_index >= len(attempts) or attempts[global_index] != trial:
                return False
            if trial.get("accepted") is True:
                accepted_in_epoch.append(trial)
            if trial.get("branch_status") == "passed_strict_branch":
                branch = trial.get("branch")
                trial_signature = branch.get("signature_sha256") if isinstance(branch, dict) else None
                if (not isinstance(trial_signature, str) or len(trial_signature) != 64
                        or trial.get("branch_signature_sha256") != trial_signature):
                    return False
                if trial.get("objective_status") == "finite":
                    if not all(_finite_number(trial.get(name))
                               for name in ("objective", "phi", "gradient_l2", "gradient_inf")):
                        return False
                    same_signature = trial_signature == signature
                    recomputed = merit_step._candidate_gate(
                        seed_j=current["objective"], seed_phi=current["phi"],
                        candidate_j=trial["objective"], candidate_phi=trial["phi"],
                        slope_j=float(slope_j), slope_phi=float(slope_phi), alpha=trial["alpha"],
                        same_signature=same_signature,
                    )
                    if (trial.get("branch_changed") is not (not same_signature)
                            or trial.get("gate") != recomputed
                            or trial.get("accepted") is not recomputed["accepted"]):
                        return False
                elif trial.get("objective_status") == "complete_after_budget":
                    if (child.get("numerical_status") != "internal_budget_exhausted"
                            or not all(_finite_number(trial.get(name))
                                       for name in ("objective", "phi", "gradient_l2", "gradient_inf"))
                            or trial.get("accepted") is True or "gate" in trial):
                        return False
                elif trial.get("objective_status") == "not_evaluated_internal_budget":
                    if (child.get("numerical_status") != "internal_budget_exhausted"
                            or trial.get("accepted") is True or "objective" in trial
                            or "gate" in trial):
                        return False
                elif (trial.get("objective_status") != "nonfinite_or_invalid"
                      or trial.get("accepted") is True or "gate" in trial):
                    return False
            elif trial.get("accepted") is True or "objective" in trial or "gate" in trial:
                return False
        if len(accepted_in_epoch) > 1 or epoch.get("accepted") is not bool(accepted_in_epoch):
            return False
        if accepted_in_epoch:
            if accepted_in_epoch[0] is not epoch["trials"][-1]:
                return False
            last_accepted = accepted_in_epoch[0]
            previous_accepted = accepted_in_epoch[0]
        elif epoch_index != len(epochs) - 1:
            return False
        expected_trial_count += len(epoch["trials"])
        epoch_trials.extend(epoch["trials"])
    if expected_trial_count != len(attempts):
        return False
    accepted = child.get("accepted_candidate")
    if last_accepted is None:
        terminal_current = child.get("current")
        terminal_branch = child.get("current_branch")
        terminal_margin_ok = seed_linear._valid_margins(
            child.get("current_branch_margins"), complete=True,
        )
        if child.get("numerical_status") == "handoff_candidate_only":
            return (accepted is None and isinstance(terminal_current, dict)
                    and _continuation_link(terminal_current, None)
                    and isinstance(terminal_branch, dict)
                    and terminal_branch.get("status") == "passed_strict_branch"
                    and terminal_margin_ok
                    and _finite_number(terminal_current.get("gradient_inf"))
                    and terminal_current["gradient_inf"] < 1.0e-4)
        if child.get("numerical_status") == "pinned_seed_merit_mismatch":
            return (accepted is None and isinstance(terminal_current, dict)
                    and isinstance(terminal_branch, dict)
                    and terminal_branch.get("status") == "passed_strict_branch"
                    and terminal_margin_ok
                    and not _continuation_link(terminal_current, None))
        if child.get("numerical_status") in {
            "transpose_consistency_refused", "no_joint_descent_direction",
            "true_slope_refused",
        }:
            return (accepted is None and isinstance(terminal_current, dict)
                    and isinstance(terminal_branch, dict)
                    and terminal_branch.get("status") == "passed_strict_branch"
                    and terminal_margin_ok
                    and _continuation_link(terminal_current, None))
        return accepted is None and child.get("numerical_status") not in {
            "accepted_epoch_limit", "final_accepted_branch_or_margin_refused",
        }
    assert last_accepted is not None
    accepted_trial = last_accepted
    if not isinstance(accepted, dict):
        return False
    accepted_hash = last_accepted.get("control_sha256")
    control_values = accepted.get("control")
    if (not isinstance(control_values, list)
            or len(control_values) != EXPECTED_CONTROLS
            or not all(_finite_number(value) for value in control_values)):
        return False
    try:
        control = torch.as_tensor(control_values, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError):
        return False
    if (control.ndim != 1 or control.shape != (EXPECTED_CONTROLS,)
            or control.dtype != torch.float64 or control.device.type != "cpu"
            or not bool(torch.isfinite(control).all())):
        return False
    final_margin_ok = True
    if child.get("numerical_status") in {
        "handoff_candidate_only", "accepted_epoch_limit", "trial_limit",
    }:
        final_branch = child.get("final_accepted_branch")
        final_margin_ok = (
            isinstance(final_branch, dict)
            and final_branch.get("status") == "passed_strict_branch"
            and final_branch.get("signature_sha256") == accepted.get("branch_signature_sha256")
            and seed_linear._valid_margins(
                child.get("final_accepted_branch_margins"), complete=True)
        )
    accepted_count = sum(item.get("accepted") is True for item in epochs)
    status = child.get("numerical_status")
    terminal_current = child.get("current")
    terminal_branch = child.get("current_branch")
    terminal_margins_ok = seed_linear._valid_margins(
        child.get("current_branch_margins"), complete=True,
    )
    if status in {"transpose_consistency_refused", "no_joint_descent_direction",
                  "true_slope_refused"}:
        if (not isinstance(terminal_current, dict)
                or not _continuation_link(terminal_current, last_accepted)
                or not isinstance(terminal_branch, dict)
                or terminal_branch.get("status") != "passed_strict_branch"
                or not terminal_margins_ok):
            return False
        if status == "transpose_consistency_refused":
            transpose = child.get("transpose_consistency")
            if (not isinstance(transpose, dict) or transpose.get("status") != "refused"
                    or transpose.get("reason") not in {
                        "invalid_or_nonfinite_hessian_products",
                        "nonfinite_transpose_comparison", "transpose_consistency_failed",
                    }):
                return False
        elif status == "no_joint_descent_direction":
            direction = child.get("direction")
            if (not isinstance(direction, dict)
                    or direction.get("status") != "no_joint_descent_direction"
                    or direction.get("reason") not in {
                        "invalid_or_nonfinite_gradient_or_hvp",
                        "nonfinite_direction_scale_or_curvature",
                        "curvature_gate_failed", "nonfinite_or_non_descent_slope",
                    }):
                return False
        else:
            direction = child.get("direction")
            if not isinstance(direction, dict) or direction.get("status") != "passed":
                return False
            true_j, true_phi = child.get("true_slope_j"), child.get("true_slope_phi")
            if (not _finite_number(true_j) or not _finite_number(true_phi)
                    or child.get("true_slope_refusal_reason") not in {
                        "nonfinite_true_slope", "true_slope_not_descent",
                    }):
                return False
            if isinstance(true_j, (int, float)) and isinstance(true_phi, (int, float)):
                if true_j < 0 and true_phi < 0:
                    return False
    if status == "current_merit_refused":
        if ("current" in child or not isinstance(terminal_branch, dict)
                or terminal_branch.get("status") != "passed_strict_branch"
                or not terminal_margins_ok):
            return False
    if status == "current_branch_or_margin_refused":
        if (not isinstance(terminal_branch, dict)
                or terminal_branch.get("status") == "passed_strict_branch"):
            return False
    if status in {"current_merit_refused", "current_branch_or_margin_refused"}:
        expected_current_hash = (last_accepted.get("control_sha256")
                                 if last_accepted is not None
                                 else SEED_CONTROL_SHA256)
        if child.get("current_control_sha256") != expected_current_hash:
            return False
    if status == "accepted_epoch_limit" and accepted_count != MAX_ACCEPTED_EPOCHS:
        return False
    if status == "trial_limit" and len(attempts) != MAX_TOTAL_TRIALS:
        return False
    if status == "epoch_line_search_refused":
        if (not epochs or epochs[-1].get("accepted") is not False
                or len(epochs[-1]["trials"]) != MAX_TRIALS_PER_EPOCH
                or any(item.get("accepted") is True for item in epochs[-1]["trials"])):
            return False
    if status == "handoff_candidate_only" and accepted.get("gradient_inf", 1.0) >= 1.0e-4:
        return False
    if status == "final_accepted_branch_or_margin_refused":
        failed_final = child.get("final_accepted_branch")
        if (not isinstance(failed_final, dict)
                or failed_final.get("status") == "passed_strict_branch"):
            return False
    return (
        accepted.get("path_certified") is False
        and accepted.get("control_sha256") == accepted_hash
        and seed_linear._tensor_sha(control) == accepted_hash
        and accepted.get("epoch") == accepted_trial.get("epoch")
        and accepted.get("trial") == accepted_trial.get("trial")
        and accepted.get("alpha") == accepted_trial.get("alpha")
        and accepted.get("objective") == accepted_trial.get("objective")
        and accepted.get("phi") == accepted_trial.get("phi")
        and accepted.get("gradient_inf") == accepted_trial.get("gradient_inf")
        and accepted.get("branch_signature_sha256") == accepted_trial.get("branch_signature_sha256")
        and accepted.get("branch_changed") == accepted_trial.get("branch_changed")
        and type(accepted.get("branch_changed")) is bool
        and final_margin_ok
    )


def run(directory: Path) -> dict[str, Any]:
    """Launch one bounded child; parent only checks source, archive, and report."""
    started = time.monotonic()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("continuation output directory must be fresh and empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_3h_merit_continuation.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output)]
    sources_before = _source_hashes()
    if (not first_branch._sources_match_archive(sources_before)
            or _sha(PLAN) != PLAN_SHA256 or _sha(SEED_REPORT) != SEED_REPORT_SHA256
            or _sha(SEED_EVIDENCE) != SEED_EVIDENCE_SHA256):
        raise ValueError("continuation source or pinned evidence changed before launch")
    archive_before = first_branch._archive_input_identity()
    if _canonical_sha(archive_before) != seed_linear.ARCHIVED_INPUT_SHA256:
        raise ValueError("fixed archived input identity changed before launch")
    prelaunch = time.monotonic() - started
    resource = run_guarded(command, wall_seconds=WALL_SECONDS,
                           rss_bytes=SAMPLED_RSS_BYTES,
                           report_path=directory / "point_3h_merit_continuation.resource.json",
                           log_path=directory / "point_3h_merit_continuation.log")
    try:
        child = json.loads(output.read_text())
        read_error = None
    except (OSError, json.JSONDecodeError) as error:
        child, read_error = None, f"{type(error).__name__}: {error}"
    sources_after = _source_hashes()
    try:
        archive_unchanged = first_branch._archive_input_identity() == archive_before
    except (OSError, ValueError, json.JSONDecodeError):
        archive_unchanged = False
    child_valid = _valid_child(child, sources_before)
    if isinstance(child, dict):
        child_valid = child_valid and child.get("pid") == resource.get("child_pid")
    ok = (_valid_resource(resource, command) and child_valid
          and sources_before == sources_after and archive_unchanged
          and _sha(PLAN) == PLAN_SHA256 and _sha(SEED_REPORT) == SEED_REPORT_SHA256
          and _sha(SEED_EVIDENCE) == SEED_EVIDENCE_SHA256)
    result = {
        "execution_status": "completed" if ok else "failed",
        "execution_phase": "finished" if ok else "guard_or_evidence_failure",
        "numerical_status": child.get("numerical_status", "not_reached") if isinstance(child, dict) else "not_reached",
        "accepted_candidate": child.get("accepted_candidate") if isinstance(child, dict) else None,
        "accepted_epochs": child.get("accepted_epochs", []) if isinstance(child, dict) else [],
        "candidate_attempts": child.get("candidate_attempts", []) if isinstance(child, dict) else [],
        "child_elapsed_seconds": child.get("child_elapsed_seconds") if isinstance(child, dict) else None,
        "child_phase_seconds": child.get("child_phase_seconds") if isinstance(child, dict) else None,
        "parent_prelaunch_seconds": prelaunch, "guarded_wait_seconds": resource.get("elapsed_seconds"),
        "parent_elapsed_seconds": time.monotonic() - started,
        "child_read_error": read_error, "resource": resource,
        "source_sha256": sources_before, "source_unchanged": sources_before == sources_after,
        "plan_sha256": PLAN_SHA256, "plan_unchanged": _sha(PLAN) == PLAN_SHA256,
        "seed_report_sha256": SEED_REPORT_SHA256,
        "seed_report_unchanged": _sha(SEED_REPORT) == SEED_REPORT_SHA256,
        "seed_evidence_sha256": SEED_EVIDENCE_SHA256,
        "seed_evidence_unchanged": _sha(SEED_EVIDENCE) == SEED_EVIDENCE_SHA256,
        "archive_unchanged": archive_unchanged, "child_report_valid": bool(child_valid),
        "scope": "bounded endpoint J/Phi continuation; no connecting-path, root, or response claim",
    }
    _write_json(directory / "point_3h_merit_continuation.run.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = run_probe(args.output)
        code = 0 if result.get("execution_status") == "completed" else 2
    elif args.directory is not None:
        result = run(args.directory)
        code = 0 if result["execution_status"] == "completed" else 1
    else:
        parser.error("provide --output for child mode or --directory for guarded mode")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(code)
