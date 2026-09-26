"""Try one pinned branch-aware joint J/Phi descent step at the 3-hour seed.

This is a bounded exploratory candidate search. It never qualifies a root or
computes a score, adjoint, parameter VJP, or response.
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
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed_linear

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_3H_MERIT_STEP_PLAN.md"
PLAN_SHA256 = "9af6bc4b0a30694f109b144579297704ecda4e1ce888fdb8d621dc9a1f7abfb6"
HESSIAN_EVIDENCE = EVIDENCE / "FV_POINT_3H_SEED_HESSIAN_EVIDENCE.json"
HESSIAN_EVIDENCE_SHA256 = "77bf193a2f891748199c12c0a318bd47fcb0f199a23da3f0e926d7fa4ebf8ed0"
SEED_HESSIAN_RELATIVE_SYMMETRY = 1.0949379174674463e-15
EXPECTED_SEED_BRANCH_SIGNATURE_SHA256 = (
    "80a1fac22b536bfedcedfda506e11d85f4b9f2a73c5610c73fa8dfdb210dfe51"
)
ARCHIVED_INPUT_SHA256 = seed_linear.ARCHIVED_INPUT_SHA256
WALL_SECONDS = 300
REAP_GRACE_SECONDS = 2
SAMPLED_RSS_BYTES = 1024**3
EXPECTED_CONTROLS = 26
EXPECTED_STAGES = 3600
EXPECTED_PARAMETERS = 13
STATIONARITY_TOLERANCE = 1.0e-10
ARMijo_C1 = 1.0e-4
INITIAL_ALPHA = 0.02
MAX_BACKTRACKS = 8
ROUNDING_MULTIPLIER = 128.0
DEFAULT_DIRECTORY = EVIDENCE / "point_3h_merit_step_attempt1"

SOURCE_PATHS = tuple(dict.fromkeys((
    *seed_linear.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_3h_merit_step_probe.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _safe_float(value: Tensor | float) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _source_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _compact_signature(signature: dict[str, Any]) -> dict[str, Any]:
    return {"choices": signature.get("choices"),
            "face_signs": signature.get("face_signs")}


def _signature_sha(signature: dict[str, Any]) -> str:
    return _canonical_sha(_compact_signature(signature))


def _roundoff_floor(value: float, dtype: torch.dtype = torch.float64) -> float:
    return ROUNDING_MULTIPLIER * torch.finfo(dtype).eps * max(
        abs(value), torch.finfo(dtype).tiny,
    )


def _joint_direction(gradient: Tensor, hessian_gradient: Tensor) -> tuple[Tensor | None, dict[str, Any]]:
    """Return the declared -Hg direction only when both merit slopes descend."""
    record: dict[str, Any] = {"status": "no_joint_descent_direction"}
    if (gradient.ndim != 1 or hessian_gradient.shape != gradient.shape
            or gradient.dtype != torch.float64
            or hessian_gradient.dtype != gradient.dtype
            or gradient.device.type != "cpu" or hessian_gradient.device != gradient.device
            or not bool(torch.isfinite(gradient).all())
            or not bool(torch.isfinite(hessian_gradient).all())):
        record["reason"] = "invalid_or_nonfinite_gradient_or_hvp"
        return None, record
    gradient_norm = torch.linalg.vector_norm(gradient)
    hg_norm = torch.linalg.vector_norm(hessian_gradient)
    curvature = torch.dot(gradient, hessian_gradient)
    threshold = ROUNDING_MULTIPLIER * torch.finfo(gradient.dtype).eps * gradient_norm * hg_norm
    values = (float(gradient_norm), float(hg_norm), float(curvature), float(threshold))
    if not all(math.isfinite(value) for value in values):
        record["reason"] = "nonfinite_direction_scale_or_curvature"
        return None, record
    record.update(gradient_l2=values[0], hg_l2=values[1],
                  curvature_dot=values[2], curvature_threshold=values[3])
    if values[0] == 0.0 or values[1] == 0.0 or values[2] <= values[3]:
        record["reason"] = "curvature_gate_failed"
        return None, record
    direction = -hessian_gradient / hg_norm
    slope_j = torch.dot(gradient, direction)
    slope_phi = torch.dot(hessian_gradient, direction)
    if (not bool(torch.isfinite(direction).all())
            or not bool(torch.isfinite(slope_j)) or not bool(torch.isfinite(slope_phi))
            or float(slope_j) >= 0.0 or float(slope_phi) >= 0.0):
        record["reason"] = "nonfinite_or_non_descent_slope"
        return None, record
    record.update(status="passed", direction_sha256=_tensor_sha(direction),
                  slope_j=float(slope_j), slope_phi=float(slope_phi),
                  direction_l2=float(torch.linalg.vector_norm(direction)),
                  direction_max_abs=float(direction.abs().max()))
    return direction, record


def _block_norms(value: Tensor, field_count: int) -> dict[str, float]:
    return {
        "field_l2": float(torch.linalg.vector_norm(value[:field_count])),
        "dynamics_l2": float(torch.linalg.vector_norm(value[field_count:])),
        "all_l2": float(torch.linalg.vector_norm(value)),
        "max_abs_component": float(value.abs().max()),
    }


def _candidate_gate(*, seed_j: float, seed_phi: float, candidate_j: float,
                    candidate_phi: float, slope_j: float, slope_phi: float,
                    alpha: float, same_signature: bool,
                    dtype: torch.dtype = torch.float64) -> dict[str, Any]:
    """Apply strict roundoff decreases and, on the seed branch, both Armijo tests."""
    floors = {"j": _roundoff_floor(seed_j, dtype),
              "phi": _roundoff_floor(seed_phi, dtype)}
    decrease_j = seed_j - candidate_j
    decrease_phi = seed_phi - candidate_phi
    actual = decrease_j > floors["j"] and decrease_phi > floors["phi"]
    armijo_j = seed_j + ARMijo_C1 * alpha * slope_j
    armijo_phi = seed_phi + ARMijo_C1 * alpha * slope_phi
    armijo = candidate_j <= armijo_j and candidate_phi <= armijo_phi
    accepted = actual and (armijo if same_signature else True)
    reasons: list[str] = []
    if not decrease_j > floors["j"]:
        reasons.append("analysis_objective_not_decreased_beyond_roundoff")
    if not decrease_phi > floors["phi"]:
        reasons.append("stationarity_merit_not_decreased_beyond_roundoff")
    if same_signature and actual and not armijo:
        if candidate_j > armijo_j:
            reasons.append("analysis_objective_armijo_failed")
        if candidate_phi > armijo_phi:
            reasons.append("stationarity_merit_armijo_failed")
    return {
        "accepted": accepted, "branch_changed": not same_signature,
        "actual_decrease": actual, "armijo_required": same_signature,
        "armijo_passed": armijo if same_signature else None,
        "roundoff_floors": floors, "decrease_j": decrease_j,
        "decrease_phi": decrease_phi, "armijo_limits": {
            "j": armijo_j, "phi": armijo_phi,
        }, "refusal_reasons": reasons,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _fixed_identity() -> tuple[Any, Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    problem, original, control, parameters, truth, identity = seed_linear._prepare_fixed_seed()
    if (control.shape != (EXPECTED_CONTROLS,) or parameters.shape != (EXPECTED_PARAMETERS,)
            or control.dtype != torch.float64 or parameters.dtype != torch.float64
            or control.device.type != "cpu" or parameters.device.type != "cpu"
            or problem.layout["controls"] != EXPECTED_CONTROLS
            or problem.layout["parameters"] != EXPECTED_PARAMETERS
            or problem.layout["euler_stages"] != EXPECTED_STAGES
            or problem.frozen.nowcast_config.forecast_steps != 18):
        raise ValueError("fixed shifted three-hour seed layout changed")
    return problem, original, control, parameters, truth, identity


def run_probe(output: Path) -> dict[str, Any]:
    """Rebuild the pinned seed and try the one declared backtracking sequence."""
    started = time.monotonic()
    phase_seconds: dict[str, float] = {}
    branch_oracle_calls = 0
    exact_hvp_calls = 0
    source_before: dict[str, str] = {}
    archive_before: dict[str, Any] | None = None
    input_before: dict[str, Any] | None = None
    problem = original = control = parameters = truth = None
    report: dict[str, Any] = {
        "pid": os.getpid(), "execution_status": "running",
        "execution_phase": "source_check", "numerical_status": "not_reached",
        "branch_status": "not_performed", "direction_status": "not_attempted",
        "accepted_candidate": None, "candidate_attempts": [],
        "stationarity_passed": "not_tested", "response_computed": False,
        "adjoint_computed": False, "pcg_calls": 0, "score_computed": False,
        "finite_segment_certified": False, "segment_certified": False,
        "plan_sha256": PLAN_SHA256,
        "hessian_evidence_sha256": HESSIAN_EVIDENCE_SHA256,
        "archive_input_sha256": ARCHIVED_INPUT_SHA256,
        "wall_limit_seconds": WALL_SECONDS,
        "termination_reap_grace_seconds": REAP_GRACE_SECONDS,
        "rss_limit_bytes": SAMPLED_RSS_BYTES,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
    }
    _write_json(output, report)
    try:
        tick = time.monotonic()
        source_before = _source_hashes()
        if (not first_branch._sources_match_archive(source_before)
                or _sha(PLAN) != PLAN_SHA256
                or _sha(HESSIAN_EVIDENCE) != HESSIAN_EVIDENCE_SHA256):
            raise ValueError("pinned source or merit-step plan identity changed")
        archive_before = first_branch._archive_input_identity()
        if _canonical_sha(archive_before) != ARCHIVED_INPUT_SHA256:
            raise ValueError("fixed archived input identity changed")
        phase_seconds["source_and_archive_identity"] = time.monotonic() - tick

        tick = time.monotonic()
        problem, original, control, parameters, truth, input_before = _fixed_identity()
        if input_before.get("archived_input") != archive_before:
            raise ValueError("reconstructed seed differs from pinned archive")
        field_count = int(problem.frozen.active_field_index.numel())
        if field_count != 20:
            raise ValueError("fixed 3-hour latent control layout changed")
        report.update(execution_phase="strict_seed_branch", input_before=input_before,
                      archive_before=archive_before,
                      hessian_evidence_relative_symmetry=SEED_HESSIAN_RELATIVE_SYMMETRY,
                      layout=problem.layout,
                      candidate_seed={
                          "control_sha256": _tensor_sha(control),
                          "parameters_sha256": _tensor_sha(parameters),
                          "terminal_truth_sha256": _tensor_sha(truth),
                      })
        _write_json(output, report)
        try:
            branch_oracle_calls += 1
            seed_signature, branch_scope = problem.branch_check(control, parameters)
        except ValueError as error:
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status="seed_branch_refused", branch_status="refused",
                          refusal=f"{type(error).__name__}: {error}")
            return report
        compact_seed = _compact_signature(seed_signature)
        seed_signature_sha = _canonical_sha(compact_seed)
        if seed_signature_sha != EXPECTED_SEED_BRANCH_SIGNATURE_SHA256:
            raise ValueError("fixed PR #218/#220 seed branch signature changed")
        if (seed_signature.get("euler_stages") != EXPECTED_STAGES
                or len(seed_signature.get("choices", [])) != EXPECTED_STAGES
                or len(seed_signature.get("face_signs", [])) != EXPECTED_STAGES):
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status="seed_branch_incomplete", branch_status="incomplete")
            return report
        report.update(branch_status="passed_strict", seed_branch_signature_sha256=seed_signature_sha,
                      branch_scope=branch_scope)
        phase_seconds["strict_seed_branch"] = time.monotonic() - tick

        tick = time.monotonic()
        objective = problem.objective(control, parameters)
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        gradient = gradient_fn(control, parameters)
        phase_seconds["seed_objective_gradient"] = time.monotonic() - tick
        if (not isinstance(objective, Tensor) or objective.shape != ()
                or objective.dtype != control.dtype or objective.device != control.device
                or not isinstance(gradient, Tensor) or gradient.shape != control.shape
                or gradient.dtype != control.dtype or gradient.device != control.device
                or not bool(torch.isfinite(objective)) or not bool(torch.isfinite(gradient).all())):
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status="seed_objective_or_gradient_invalid",
                          branch_status="passed_strict")
            return report
        tick = time.monotonic()
        exact_hvp_calls = 1
        hg = torch.func.jvp(
            lambda value: gradient_fn(value, parameters), (control,), (gradient,),
        )[1]
        phase_seconds["one_exact_hg_hvp"] = time.monotonic() - tick
        if (not isinstance(hg, Tensor) or hg.shape != control.shape
                or hg.dtype != control.dtype or hg.device != control.device
                or not bool(torch.isfinite(hg).all())):
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status="nonfinite_hg", branch_status="passed_strict")
            return report
        seed_phi = float(torch.dot(gradient, gradient)) / 2.0
        if not math.isfinite(seed_phi):
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status="nonfinite_seed_merit",
                          branch_status="passed_strict", seed={
                              "objective": float(objective), "phi": None,
                              "gradient_l2": float(torch.linalg.vector_norm(gradient)),
                              "gradient_inf": float(gradient.abs().max()),
                          })
            return report
        direction, direction_record = _joint_direction(gradient, hg)
        report.update(direction_status=direction_record["status"],
                      direction=direction_record,
                      seed={"objective": float(objective),
                            "gradient_l2": float(torch.linalg.vector_norm(gradient)),
                            "gradient_inf": float(gradient.abs().max()),
                            "gradient_blocks": _block_norms(gradient, field_count),
                            "phi": seed_phi,
                            "hg_blocks": _block_norms(hg, field_count)})
        if direction is None:
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status="no_joint_descent_direction",
                          branch_status="passed_strict")
            return report

        seed_j = float(objective)
        slope_j = float(torch.dot(gradient, direction))
        slope_phi = float(torch.dot(hg, direction))
        report.update(execution_phase="line_search", numerical_status="searching",
                      line_search={"alpha_grid": [INITIAL_ALPHA / (2**k)
                                                   for k in range(MAX_BACKTRACKS)],
                                   "maximum_candidates": MAX_BACKTRACKS,
                                   "armijo_c1": ARMijo_C1,
                                   "roundoff_multiplier": ROUNDING_MULTIPLIER})
        _write_json(output, report)
        for backtrack in range(MAX_BACKTRACKS):
            alpha = INITIAL_ALPHA / (2**backtrack)
            candidate_control = control + alpha * direction
            attempt: dict[str, Any] = {
                "backtrack": backtrack, "alpha": alpha,
                "control_sha256": _tensor_sha(candidate_control),
                "branch_status": "not_checked", "objective_status": "not_evaluated",
                "accepted": False, "segment_certified": False,
            }
            if (candidate_control.shape != control.shape
                    or not bool(torch.isfinite(candidate_control).all())):
                attempt.update(branch_status="refused_nonfinite_control",
                               refusal_reason="candidate control is nonfinite")
                report["candidate_attempts"].append(attempt)
                continue
            try:
                branch_oracle_calls += 1
                candidate_signature, _ = problem.branch_check(candidate_control, parameters)
            except ValueError as error:
                attempt.update(branch_status="refused_strict_branch",
                               refusal_reason=f"{type(error).__name__}: {error}")
                report["candidate_attempts"].append(attempt)
                continue
            if (candidate_signature.get("euler_stages") != EXPECTED_STAGES
                    or len(candidate_signature.get("choices", [])) != EXPECTED_STAGES
                    or len(candidate_signature.get("face_signs", [])) != EXPECTED_STAGES):
                attempt.update(branch_status="incomplete_strict_branch",
                               refusal_reason="candidate branch signature is incomplete")
                report["candidate_attempts"].append(attempt)
                continue
            candidate_signature_sha = _signature_sha(candidate_signature)
            same_signature = candidate_signature_sha == seed_signature_sha
            attempt.update(branch_status="passed_strict", branch_signature_sha256=candidate_signature_sha,
                           branch_changed=not same_signature)
            tick = time.monotonic()
            candidate_j_tensor = problem.objective(candidate_control, parameters)
            candidate_gradient = gradient_fn(candidate_control, parameters)
            candidate_phi_tensor = torch.dot(candidate_gradient, candidate_gradient) / 2.0
            phase_seconds["candidate_objective_gradient"] = (
                phase_seconds.get("candidate_objective_gradient", 0.0)
                + time.monotonic() - tick
            )
            if (candidate_j_tensor.shape != () or candidate_gradient.shape != control.shape
                    or candidate_j_tensor.dtype != control.dtype
                    or candidate_j_tensor.device != control.device
                    or candidate_gradient.dtype != control.dtype
                    or candidate_gradient.device != control.device
                    or not bool(torch.isfinite(candidate_j_tensor))
                    or not bool(torch.isfinite(candidate_gradient).all())
                    or not bool(torch.isfinite(candidate_phi_tensor))):
                attempt.update(objective_status="nonfinite_or_invalid",
                               refusal_reason="fresh candidate J/g/Phi contract failed")
                report["candidate_attempts"].append(attempt)
                continue
            candidate_j = float(candidate_j_tensor)
            candidate_phi = float(candidate_phi_tensor)
            gate = _candidate_gate(
                seed_j=seed_j, seed_phi=seed_phi,
                candidate_j=candidate_j, candidate_phi=candidate_phi,
                slope_j=slope_j, slope_phi=slope_phi, alpha=alpha,
                same_signature=same_signature,
            )
            attempt.update(objective_status="finite", objective=candidate_j,
                           gradient_l2=float(torch.linalg.vector_norm(candidate_gradient)),
                           gradient_inf=float(candidate_gradient.abs().max()), phi=candidate_phi,
                           gradient_blocks=_block_norms(candidate_gradient, field_count),
                           gate=gate, accepted=gate["accepted"])
            report["candidate_attempts"].append(attempt)
            if gate["accepted"]:
                report["accepted_candidate"] = {
                    "control": candidate_control.tolist(),
                    "control_sha256": _tensor_sha(candidate_control),
                    "branch_signature_sha256": candidate_signature_sha,
                    "branch_changed": not same_signature,
                    "alpha": alpha, "backtrack": backtrack,
                    "objective": candidate_j, "phi": candidate_phi,
                    "gradient_inf": float(candidate_gradient.abs().max()),
                    "gradient_blocks": _block_norms(candidate_gradient, field_count),
                    "acceptance": gate,
                    "segment_certified": False,
                }
                report.update(execution_status="completed", execution_phase="finished",
                              numerical_status=("candidate_accepted_branch_changed"
                                                if not same_signature
                                                else "candidate_accepted_same_signature"))
                break
        else:
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status="line_search_refused")
        if report["execution_status"] == "running":
            report.update(execution_status="completed", execution_phase="finished")
        report["candidate_refusals"] = [
            item for item in report["candidate_attempts"] if not item["accepted"]
        ]
    except Exception as error:
        report.update(execution_status="failed", execution_phase="failed",
                      numerical_status="probe_error",
                      error=f"{type(error).__name__}: {error}")
    finally:
        report["child_elapsed_seconds"] = time.monotonic() - started
        report["child_phase_seconds"] = phase_seconds
        report["costs"] = {
            "phase_seconds": phase_seconds,
            "candidate_count": len(report["candidate_attempts"]),
            "branch_oracle_calls": branch_oracle_calls,
            "exact_hvp_calls": exact_hvp_calls,
            "newton_steps": 0, "pcg_calls": 0,
            "response_computed": False, "score_computed": False,
        }
        report["source_before"] = source_before
        report["source_after"] = _source_hashes()
        report["source_unchanged"] = report["source_after"] == source_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["hessian_evidence_unchanged"] = (
            _sha(HESSIAN_EVIDENCE) == HESSIAN_EVIDENCE_SHA256
        )
        try:
            report["archive_unchanged"] = (
                archive_before is not None
                and first_branch._archive_input_identity() == archive_before
            )
        except (OSError, ValueError, json.JSONDecodeError):
            report["archive_unchanged"] = False
        if (problem is not None and isinstance(original, Tensor)
                and isinstance(control, Tensor) and isinstance(parameters, Tensor)
                and isinstance(truth, Tensor)):
            report["input_after"] = seed_linear._input_identity(
                problem, original, control, parameters, truth,
            )
            report["input_unchanged"] = report["input_after"] == input_before
        else:
            report["input_after"] = None
            report["input_unchanged"] = False
        if not all(report.get(name) is True for name in (
            "source_unchanged", "plan_unchanged", "hessian_evidence_unchanged",
            "archive_unchanged", "input_unchanged",
        )):
            report.update(execution_status="failed", execution_phase="identity_refused",
                          numerical_status="identity_changed")
        _write_json(output, report)
    return report


def _finite_number(value: object, *, minimum: float | None = None) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (minimum is None or value >= minimum))


def _valid_resource(resource: dict[str, Any], command: list[str]) -> bool:
    elapsed, peak = resource.get("elapsed_seconds"), resource.get("sampled_peak_rss_bytes")
    return (
        resource.get("command") == command and resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(peak) is int and 0 < peak <= SAMPLED_RSS_BYTES
        and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
        and math.isfinite(elapsed) and 0 <= elapsed <= WALL_SECONDS
    )


def _valid_child(child: object, expected_sources: dict[str, str],
                 expected_archive: dict[str, Any]) -> bool:
    if not isinstance(child, dict):
        return False
    input_before = child.get("input_before")
    seed = child.get("candidate_seed")
    if not isinstance(input_before, dict) or not isinstance(seed, dict):
        return False
    common = (
        child.get("execution_status") == "completed"
        and child.get("execution_phase") == "finished"
        and child.get("source_before") == child.get("source_after") == expected_sources
        and child.get("source_unchanged") is True
        and child.get("plan_sha256") == PLAN_SHA256
        and child.get("plan_unchanged") is True
        and child.get("hessian_evidence_sha256") == HESSIAN_EVIDENCE_SHA256
        and child.get("hessian_evidence_unchanged") is True
        and child.get("archive_before") == expected_archive
        and child.get("archive_input_sha256") == ARCHIVED_INPUT_SHA256
        and child.get("archive_unchanged") is True
        and child.get("input_before") == child.get("input_after")
        and child.get("input_unchanged") is True
        and input_before.get("archived_input") == expected_archive
        and input_before.get("control_sha256") == seed_linear.SHIFTED_SEED_CONTROL_SHA256
        and input_before.get("parameters_sha256") == seed_linear.EXPECTED_PARAMETERS_SHA256
        and input_before.get("terminal_truth_sha256") == seed_linear.EXPECTED_TRUTH_SHA256
        and input_before.get("changed_control_index") == seed_linear.EXPECTED_CHANGED_CONTROL
        and input_before.get("changed_control_indices") == [seed_linear.EXPECTED_CHANGED_CONTROL]
        and _canonical_sha(input_before.get("archived_input")) == ARCHIVED_INPUT_SHA256
        and seed.get("control_sha256") == seed_linear.SHIFTED_SEED_CONTROL_SHA256
        and seed.get("parameters_sha256") == seed_linear.EXPECTED_PARAMETERS_SHA256
        and seed.get("terminal_truth_sha256") == seed_linear.EXPECTED_TRUTH_SHA256
        and child.get("response_computed") is False
        and child.get("adjoint_computed") is False
        and child.get("score_computed") is False
        and child.get("pcg_calls") == 0
        and child.get("finite_segment_certified") is False
        and child.get("segment_certified") is False
        and type(child.get("pid")) is int and child["pid"] > 0
        and child.get("wall_limit_seconds") == WALL_SECONDS
        and child.get("termination_reap_grace_seconds") == REAP_GRACE_SECONDS
        and child.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and isinstance(child.get("candidate_attempts"), list)
        and len(child["candidate_attempts"]) <= MAX_BACKTRACKS
        and child.get("hessian_evidence_relative_symmetry") == SEED_HESSIAN_RELATIVE_SYMMETRY
        and isinstance(child.get("costs"), dict)
        and child["costs"].get("candidate_count") == len(child["candidate_attempts"])
        and child["costs"].get("branch_oracle_calls") == 1 + sum(
            item.get("branch_status") != "refused_nonfinite_control"
            for item in child["candidate_attempts"] if isinstance(item, dict)
        )
        and child["costs"].get("newton_steps") == 0
        and child["costs"].get("exact_hvp_calls") in (0, 1)
    )
    if not common:
        return False
    status = child.get("numerical_status")
    attempts = child["candidate_attempts"]
    costs = child["costs"]
    if status in {"seed_branch_refused", "seed_branch_incomplete"}:
        return (not attempts and costs["exact_hvp_calls"] == 0
                and child.get("direction_status") == "not_attempted")
    if status == "seed_objective_or_gradient_invalid":
        return (not attempts and costs["exact_hvp_calls"] == 0
                and child.get("branch_status") == "passed_strict")
    if status in {"nonfinite_hg", "nonfinite_seed_merit", "no_joint_descent_direction"}:
        expected_hvps = 1
        if not (not attempts and costs["exact_hvp_calls"] == expected_hvps
                and child.get("branch_status") == "passed_strict"):
            return False
        if status == "no_joint_descent_direction":
            direction_record = child["direction"]
            return (child.get("direction_status") == "no_joint_descent_direction"
                    and isinstance(direction_record, dict)
                    and direction_record.get("reason") in {
                        "invalid_or_nonfinite_gradient_or_hvp",
                        "nonfinite_direction_scale_or_curvature",
                        "curvature_gate_failed", "nonfinite_or_non_descent_slope",
                    })
        return True

    seed = child.get("seed")
    direction = child.get("direction")
    if (not isinstance(seed, dict) or not isinstance(direction, dict)
            or costs["exact_hvp_calls"] != 1
            or child.get("branch_status") != "passed_strict"
            or child.get("seed_branch_signature_sha256")
            != EXPECTED_SEED_BRANCH_SIGNATURE_SHA256
            or child.get("direction_status") != "passed"):
        return False
    finite_seed_fields = ("objective", "phi", "gradient_l2", "gradient_inf")
    if not all(_finite_number(seed.get(key), minimum=0.0 if key != "objective" else None)
               for key in finite_seed_fields):
        return False
    if not all(_finite_number(direction.get(key)) for key in (
        "slope_j", "slope_phi", "curvature_dot", "curvature_threshold",
        "gradient_l2", "hg_l2",
    )):
        return False
    if not (direction["slope_j"] < 0 and direction["slope_phi"] < 0
            and direction["curvature_dot"] > direction["curvature_threshold"]
            and isinstance(direction.get("direction_sha256"), str)
            and len(direction["direction_sha256"]) == 64):
        return False
    expected_alphas = [INITIAL_ALPHA / 2**index for index in range(MAX_BACKTRACKS)]
    if any(
        not isinstance(item, dict) or item.get("backtrack") != index
        or item.get("alpha") != expected_alphas[index]
        for index, item in enumerate(attempts)
    ):
        return False
    for trial in attempts:
        if trial.get("branch_status") == "passed_strict":
            branch_hash = trial.get("branch_signature_sha256")
            if (not isinstance(branch_hash, str) or len(branch_hash) != 64
                    or type(trial.get("branch_changed")) is not bool
                    or trial["branch_changed"]
                    != (branch_hash != EXPECTED_SEED_BRANCH_SIGNATURE_SHA256)):
                return False
        if trial.get("objective_status") == "finite":
            if not all(_finite_number(trial.get(key), minimum=0.0 if key != "objective" else None)
                       for key in ("objective", "phi", "gradient_l2", "gradient_inf")):
                return False
            changed_trial = trial.get("branch_changed")
            if type(changed_trial) is not bool:
                return False
            branch_hash = trial.get("branch_signature_sha256")
            if trial.get("branch_status") != "passed_strict":
                return False
            expected_gate = _candidate_gate(
                seed_j=float(seed["objective"]), seed_phi=float(seed["phi"]),
                candidate_j=float(trial["objective"]), candidate_phi=float(trial["phi"]),
                slope_j=float(direction["slope_j"]),
                slope_phi=float(direction["slope_phi"]),
                alpha=float(trial["alpha"]), same_signature=not changed_trial,
            )
            if trial.get("gate") != expected_gate:
                return False
        elif trial.get("accepted") is True:
            return False
    accepted = [item for item in attempts if item.get("accepted") is True]
    candidate = child.get("accepted_candidate")
    if status == "line_search_refused":
        return len(attempts) == MAX_BACKTRACKS and not accepted and candidate is None
    if status not in {"candidate_accepted_same_signature", "candidate_accepted_branch_changed"}:
        return False
    if len(accepted) != 1 or accepted[0] is not attempts[-1] or not isinstance(candidate, dict):
        return False
    trial = attempts[-1]
    trial_signature_hash = trial.get("branch_signature_sha256")
    changed = (isinstance(trial_signature_hash, str)
               and trial_signature_hash != EXPECTED_SEED_BRANCH_SIGNATURE_SHA256)
    if status == "candidate_accepted_branch_changed" and not changed:
        return False
    if status == "candidate_accepted_same_signature" and changed:
        return False
    if (candidate.get("branch_changed") is not changed
            or trial.get("branch_changed") is not changed
            or candidate.get("segment_certified") is not False
            or trial.get("segment_certified") is not False
            or candidate.get("control_sha256") != trial.get("control_sha256")
            or candidate.get("branch_signature_sha256") != trial.get("branch_signature_sha256")
            or candidate.get("alpha") != trial.get("alpha")
            or candidate.get("backtrack") != trial.get("backtrack")
            or candidate.get("objective") != trial.get("objective")
            or candidate.get("phi") != trial.get("phi")):
        return False
    values = candidate.get("control")
    control_hash = candidate.get("control_sha256")
    signature_hash = candidate.get("branch_signature_sha256")
    if (not isinstance(values, list) or len(values) != EXPECTED_CONTROLS
            or not isinstance(control_hash, str) or len(control_hash) != 64
            or not isinstance(signature_hash, str) or len(signature_hash) != 64):
        return False
    if not all(_finite_number(value) for value in values):
        return False
    tensor = torch.tensor(values, dtype=torch.float64)
    if seed_linear._tensor_sha(tensor) != candidate.get("control_sha256"):
        return False
    gate = trial.get("gate")
    return (
        isinstance(gate, dict) and gate.get("accepted") is True
        and _finite_number(trial.get("objective"))
        and _finite_number(trial.get("phi"), minimum=0.0)
        and _finite_number(trial.get("gradient_inf"), minimum=0.0)
        and gate.get("branch_changed") is changed
        and gate.get("actual_decrease") is True
        and gate.get("armijo_required") is (not changed)
        and (gate.get("armijo_passed") is None if changed
             else gate.get("armijo_passed") is True)
    )


def run(directory: Path) -> dict[str, Any]:
    """Launch one child; parent validates records without constructing the FV case."""
    started = time.monotonic()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("merit-step output directory must be fresh and empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_3h_merit_step.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output)]
    source_before = _source_hashes()
    if (not first_branch._sources_match_archive(source_before)
            or _sha(PLAN) != PLAN_SHA256
            or _sha(HESSIAN_EVIDENCE) != HESSIAN_EVIDENCE_SHA256):
        raise ValueError("merit-step source or plan identity changed before launch")
    archive_before = first_branch._archive_input_identity()
    if _canonical_sha(archive_before) != ARCHIVED_INPUT_SHA256:
        raise ValueError("merit-step archived input identity changed before launch")
    prelaunch_seconds = time.monotonic() - started
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "point_3h_merit_step.resource.json",
        log_path=directory / "point_3h_merit_step.log",
    )
    try:
        child = json.loads(output.read_text())
        read_error = None
    except (OSError, json.JSONDecodeError) as error:
        child, read_error = None, f"{type(error).__name__}: {error}"
    source_after = _source_hashes()
    try:
        archive_unchanged = first_branch._archive_input_identity() == archive_before
    except (OSError, ValueError, json.JSONDecodeError):
        archive_unchanged = False
    plan_unchanged = _sha(PLAN) == PLAN_SHA256
    hessian_evidence_unchanged = _sha(HESSIAN_EVIDENCE) == HESSIAN_EVIDENCE_SHA256
    child_valid = _valid_child(child, source_before, archive_before)
    if isinstance(child, dict):
        child_valid = child_valid and child.get("pid") == resource.get("child_pid")
    execution_ok = (
        _valid_resource(resource, command) and child_valid
        and source_before == source_after and archive_unchanged and plan_unchanged
        and hessian_evidence_unchanged
    )
    result: dict[str, Any] = {
        "execution_status": "completed" if execution_ok else "failed",
        "execution_phase": "finished" if execution_ok else "guard_or_evidence_failure",
        "numerical_status": child.get("numerical_status", "not_reached")
        if isinstance(child, dict) else "not_reached",
        "accepted_candidate": child.get("accepted_candidate")
        if isinstance(child, dict) else None,
        "candidate_attempts": child.get("candidate_attempts", [])
        if isinstance(child, dict) else [],
        "child_read_error": read_error, "resource": resource,
        "source_sha256": source_before, "source_unchanged": source_before == source_after,
        "plan_sha256": PLAN_SHA256, "plan_unchanged": plan_unchanged,
        "hessian_evidence_sha256": HESSIAN_EVIDENCE_SHA256,
        "hessian_evidence_unchanged": hessian_evidence_unchanged,
        "archive_unchanged": archive_unchanged,
        "child_report_valid": bool(child_valid),
        "parent_prelaunch_seconds": prelaunch_seconds,
        "parent_elapsed_seconds": time.monotonic() - started,
        "scope": "one pinned joint J/Phi descent direction and at most eight endpoint trials; candidate only; no root, response, PCG, or score",
    }
    _write_json(directory / "point_3h_merit_step.run.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--directory", type=Path)
    arguments = parser.parse_args()
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        report = run_probe(arguments.output)
        raise SystemExit(0 if report.get("execution_status") == "completed" else 2)
    if arguments.directory is None:
        parser.error("--output or --directory is required")
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
