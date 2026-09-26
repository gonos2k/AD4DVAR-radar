"""One guarded stationarity/linear-feasibility check at a fixed 3-hour seed.

This probe never changes the seed or applies a Newton step.  A successful PCG
solve is only a one-right-hand-side feasibility result, not a global curvature
certificate or an eligible implicit response.
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
from typing import Any, cast

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import matrix_free, transport
from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios import fv_point_3h_first_branch_probe as first_branch
from examples.weather_scenarios import fv_point_3h_shifted_branch_probe as shifted_seed
from examples.weather_scenarios.fv_point_3h_forward_case import make_case


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_3H_SEED_LINEAR_PLAN.md"
PLAN_SHA256 = "7f801b06d5772d0cb4a7a757f7c856c9c6d2cceddaf067adaa2666f87ea64d1f"
SHIFTED_PLAN_SHA256 = "3fa9bc123e9aedddfb027abd373960e71db0b7b7693bb83a3877bbc35d8db6f6"
ARCHIVED_INPUT_SHA256 = "00665e879d0e370e15f2bf3ef253a2267eb59335c2f51036289ff22632808744"
SHIFTED_SEED_CONTROL_SHA256 = "97ce3a636b25d79ca0aab208035e4299f9f4ad26ea00d557531e7a9eded2080b"
EXPECTED_PARAMETERS_SHA256 = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
EXPECTED_TRUTH_SHA256 = "b927c1a39d16f05285cccb7a9ff48d182eea539bfba746f7aff446aee0974610"
SHIFTED_HELPER_SHA256 = "fd4f36c5802a1ad0a2e40bb8a4a8e2fac2dd0d6aa0c6418922c87c9149e10c11"
WALL_SECONDS = 600
REAP_GRACE_SECONDS = 2
SAMPLED_RSS_BYTES = 1024**3
EXPECTED_STAGES = 3600
EXPECTED_ANALYSIS_STAGES = 360
EXPECTED_CONTROLS = 26
EXPECTED_PARAMETERS = 13
EXPECTED_CHANGED_CONTROL = 21
STATIONARITY_TOLERANCE = 1.0e-10
LINEAR_RTOL = 1.0e-10
LINEAR_MAX_ITERATIONS = 104
DEFAULT_DIRECTORY = EVIDENCE / "point_3h_seed_linear_attempt1"

SOURCE_PATHS = tuple(dict.fromkeys((
    *first_branch.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_3h_shifted_branch_probe.py",
    "examples/weather_scenarios/fv_point_3h_seed_linear_probe.py",
)))

_KNOWN_PCG_REFUSALS = {
    "operator must be symmetric positive definite",
    "preconditioner must be positive definite",
}
_KNOWN_PCG_NONFINITE = {
    "PCG step is not finite",
    "residual norm is not finite",
    "true residual norm is not finite",
    "PCG direction update is not finite",
}


class _NonfiniteDerivative(ValueError):
    """An explicitly checked HVP returned non-finite components."""


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
    result = float(value)
    return result if math.isfinite(result) else None


def _source_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _input_identity(problem: Any, original: Tensor, control: Tensor,
                    parameters: Tensor, truth: Tensor) -> dict[str, Any]:
    return shifted_seed._input_identity(
        problem, original, control, parameters, truth,
    )


def _prepare_fixed_seed() -> tuple[Any, Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    problem, original, control, parameters, truth, identity = (
        shifted_seed._prepare_shifted_case()
    )
    layout = problem.layout
    if (control.shape != (EXPECTED_CONTROLS,)
            or parameters.shape != (EXPECTED_PARAMETERS,)
            or control.dtype != torch.float64 or parameters.dtype != torch.float64
            or control.device.type != "cpu" or parameters.device.type != "cpu"
            or layout["controls"] != EXPECTED_CONTROLS
            or layout["parameters"] != EXPECTED_PARAMETERS
            or layout["euler_stages"] != EXPECTED_STAGES
            or layout["observation_times_seconds"] != (0.0, 600.0, 1200.0)
            or layout["forecast_time_seconds"] != 12000.0
            or problem.frozen.nowcast_config.forecast_steps != 18):
        raise ValueError("fixed PR #204 point 3-hour profile changed")
    if (identity["changed_control_index"] != EXPECTED_CHANGED_CONTROL
            or identity["changed_control_indices"] != [EXPECTED_CHANGED_CONTROL]
            or identity["control_sha256"] != SHIFTED_SEED_CONTROL_SHA256
            or identity["parameters_sha256"] != EXPECTED_PARAMETERS_SHA256
            or identity["terminal_truth_sha256"] != EXPECTED_TRUTH_SHA256
            or _canonical_sha(identity["archived_input"]) != ARCHIVED_INPUT_SHA256):
        raise ValueError("fixed PR #218 shifted seed/input identity changed")
    return problem, original, control, parameters, truth, identity


def _minimum_record(value: Tensor, stage: int) -> dict[str, Any]:
    minimum = value.min()
    return {"value": _safe_float(minimum), "stage_index": stage,
            "stage_context": first_branch._stage_context(stage)}


class _MarginCollector:
    """Keep only global minima and the flux scale across admitted stages."""

    def __init__(self) -> None:
        self.stages = 0
        self.minima: dict[str, dict[str, Any] | None] = {
            name: None for name in (
                "x_left_slope_over_q_scale",
                "x_right_slope_over_q_scale",
                "x_active_limiter_gap_over_q_scale",
                "y_left_slope_over_q_scale",
                "y_right_slope_over_q_scale",
                "y_active_limiter_gap_over_q_scale",
            )
        }
        self.minimum_face: tuple[float, int, str, list[int]] | None = None
        self.maximum_face_flux = 0.0
        self.nonfinite = False

    def _update(self, name: str, values: Tensor, stage: int,
                mask: Tensor | None = None) -> None:
        selected = values if mask is None else values[mask]
        if selected.numel() == 0:
            return
        if not bool(torch.isfinite(selected).all()):
            self.nonfinite = True
            return
        candidate = float(selected.min())
        previous = self.minima[name]
        if previous is None or candidate < previous["value"]:
            self.minima[name] = _minimum_record(selected, stage)

    def __call__(self, q: Tensor, qx: Tensor, qy: Tensor) -> None:
        stage = self.stages
        self.stages += 1
        q_scale = q.abs().max()
        if not bool(torch.isfinite(q_scale)) or float(q_scale) <= 0:
            self.nonfinite = True
            return
        tolerance = 128 * torch.finfo(q.dtype).eps * q_scale
        for orientation, left, right in (
            ("x", q[1:-1, 1:-1] - q[1:-1, :-2],
             q[1:-1, 2:] - q[1:-1, 1:-1]),
            ("y", q[1:-1, 1:-1] - q[:-2, 1:-1],
             q[2:, 1:-1] - q[1:-1, 1:-1]),
        ):
            self._update(f"{orientation}_left_slope_over_q_scale",
                         left.abs() / q_scale, stage)
            self._update(f"{orientation}_right_slope_over_q_scale",
                         right.abs() / q_scale, stage)
            active = (((left > tolerance) & (right > tolerance))
                      | ((left < -tolerance) & (right < -tolerance)))
            self._update(f"{orientation}_active_limiter_gap_over_q_scale",
                         (left - right).abs() / q_scale, stage, active)

        for orientation, faces in (("x", qx), ("y", qy)):
            if faces.numel() == 0:
                continue
            absolute = faces.abs()
            if not bool(torch.isfinite(absolute).all()):
                self.nonfinite = True
                continue
            self.maximum_face_flux = max(self.maximum_face_flux, float(absolute.max()))
            flat_index = int(torch.argmin(absolute))
            candidate = float(absolute.reshape(-1)[flat_index])
            if self.minimum_face is None or candidate < self.minimum_face[0]:
                index = [int(value) for value in torch.unravel_index(
                    torch.tensor(flat_index), faces.shape,
                )]
                self.minimum_face = (candidate, stage, orientation, index)

    def report(self) -> dict[str, Any]:
        face: dict[str, Any] | None = None
        if self.minimum_face is not None and self.maximum_face_flux > 0:
            value, stage, orientation, index = self.minimum_face
            face = {
                "value": value / self.maximum_face_flux,
                "absolute_minimum": value,
                "global_maximum_absolute_flux": self.maximum_face_flux,
                "stage_index": stage,
                "stage_context": first_branch._stage_context(stage),
                "orientation": orientation,
                "index": index,
            }
        return {
            "observed_stage_count": self.stages,
            "expected_stage_count": EXPECTED_STAGES,
            "complete": self.stages == EXPECTED_STAGES and not self.nonfinite,
            "nonfinite": self.nonfinite,
            "normalized_minima": self.minima,
            "absolute_face_flux_over_global_maximum": face,
            "flux_global_maximum_absolute": self.maximum_face_flux,
        }


def _full_branch_with_margins(problem: Any, control: Tensor,
                              parameters: Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    collector = _MarginCollector()
    # A separate full pass ensures the first strictly rejected stage is still
    # included in diagnostics; the strict oracle itself short-circuits there.
    with torch.no_grad(), transport.observe_minmod_stages(collector):
        problem.forecast(control, parameters)
    margins = collector.report()
    try:
        signature, scope = problem.branch_check(control, parameters)
    except ValueError as error:
        return ({"status": "branch_refused", "reason": str(error)}, margins)
    compact = {"choices": signature.get("choices"),
               "face_signs": signature.get("face_signs")}
    branch = {
        "status": "passed_strict_branch",
        "euler_stages": signature.get("euler_stages"),
        "choice_stage_count": len(signature.get("choices", [])),
        "face_sign_stage_count": len(signature.get("face_signs", [])),
        "signature_sha256": _canonical_sha(compact),
        "scope": scope,
    }
    if not margins["complete"]:
        branch["strict_status"] = branch["status"]
        branch["status"] = "branch_diagnostic_incomplete"
        return branch, margins
    if (branch["euler_stages"] != EXPECTED_STAGES
            or branch["choice_stage_count"] != EXPECTED_STAGES
            or branch["face_sign_stage_count"] != EXPECTED_STAGES):
        return ({"status": "branch_diagnostic_incomplete"}, margins)
    return branch, margins


def _gradient_norms(gradient: Tensor, field_count: int) -> dict[str, float]:
    return {
        "field_l2": float(torch.linalg.vector_norm(gradient[:field_count])),
        "field_inf": float(gradient[:field_count].abs().max()),
        "dynamics_l2": float(torch.linalg.vector_norm(gradient[field_count:])),
        "dynamics_inf": float(gradient[field_count:].abs().max()),
        "all_l2": float(torch.linalg.vector_norm(gradient)),
        "all_inf": float(gradient.abs().max()),
    }


def _linear_preflight(problem: Any, control: Tensor,
                      parameters: Tensor, gradient: Tensor) -> dict[str, Any]:
    calls = 0
    objective = lambda value: problem.objective(value, parameters)
    gradient_fn = torch.func.grad(objective)

    def hessian_vector(direction: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        product = torch.func.jvp(
            gradient_fn, (control,), (direction,),
        )[1]
        if not isinstance(product, Tensor) or not bool(torch.isfinite(product).all()):
            raise _NonfiniteDerivative("exact Hessian-vector product is nonfinite")
        return product

    result: dict[str, Any] = {
        "status": "not_attempted", "rtol": LINEAR_RTOL,
        "max_iterations": LINEAR_MAX_ITERATIONS,
        "preconditioner": "none", "hvp_calls": 0,
        "iterations": 0, "pcg_converged": None,
        "pcg_reported_relative_residual": None,
        "true_relative_residual": None,
        "curvature_certificate": "not_proved_by_pcg",
    }
    if gradient.abs().max() < STATIONARITY_TOLERANCE:
        result["status"] = "skipped_stationary_zero_rhs"
        result["curvature_certificate"] = "unverified_zero_rhs_skipped"
        return result

    try:
        solve = matrix_free.pcg(
            hessian_vector, -gradient, rtol=LINEAR_RTOL,
            max_iterations=LINEAR_MAX_ITERATIONS,
        )
    except _NonfiniteDerivative as error:
        result.update(status="nonfinite_derivative", refusal=str(error),
                      hvp_calls=calls)
        return result
    except RuntimeError as error:
        if str(error) not in _KNOWN_PCG_REFUSALS:
            if str(error) in _KNOWN_PCG_NONFINITE:
                result.update(status="nonfinite_derivative", refusal=str(error),
                              hvp_calls=calls)
                return result
            raise
        result.update(status="pcg_refused_non_spd", refusal=str(error),
                      hvp_calls=calls)
        return result

    result["iterations"] = solve.iterations
    result["pcg_converged"] = solve.converged
    result["pcg_reported_relative_residual"] = solve.relative_residual
    if (solve.solution.shape != control.shape or solve.solution.dtype != control.dtype
            or solve.solution.device != control.device
            or not bool(torch.isfinite(solve.solution).all())):
        result.update(status="nonfinite_derivative", hvp_calls=calls)
        return result

    # One extra exact HVP recomputes the true residual independently of PCG.
    try:
        true_product = hessian_vector(solve.solution)
    except _NonfiniteDerivative as error:
        result.update(status="nonfinite_derivative", refusal=str(error),
                      hvp_calls=calls)
        return result
    denominator = float(torch.linalg.vector_norm(gradient))
    numerator = float(torch.linalg.vector_norm(true_product + gradient))
    true_relative = numerator / denominator if denominator > 0 else math.inf
    result["true_relative_residual"] = _safe_float(true_relative)
    result["hvp_calls"] = calls
    if not solve.converged:
        result["status"] = "pcg_iteration_budget"
    elif not math.isfinite(true_relative):
        result["status"] = "nonfinite_derivative"
    elif true_relative <= LINEAR_RTOL:
        result["status"] = "linear_pass"
    else:
        result["status"] = "linear_residual_failed"
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def run_probe(output: Path) -> dict[str, Any]:
    """Rebuild, inspect and solve one RHS entirely inside the guarded child."""
    child_started = time.monotonic()
    report: dict[str, Any] = {
        "pid": os.getpid(), "execution_status": "running",
        "execution_phase": "source_check", "numerical_status": "not_reached",
        "branch_status": "not_performed", "stationarity_status": "not_tested",
        "linear_status": "not_attempted", "response_computed": False,
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "plan_sha256": PLAN_SHA256,
        "shifted_plan_sha256": SHIFTED_PLAN_SHA256,
        "archive_input_sha256": ARCHIVED_INPUT_SHA256,
        "hvp_calls": 0,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
    }
    _write_json(output, report)
    problem = original = control = parameters = truth = None
    source_before: dict[str, str] = {}
    input_before: dict[str, Any] | None = None
    archive_before: dict[str, Any] | None = None
    phase_seconds: dict[str, float] = {}
    try:
        phase_tick = time.monotonic()
        source_before = _source_hashes()
        if (not first_branch._sources_match_archive(source_before)
                or source_before.get(
                    "examples/weather_scenarios/fv_point_3h_shifted_branch_probe.py"
                ) != SHIFTED_HELPER_SHA256):
            raise ValueError("fixed PR #204/#218 source identity changed")
        if _sha(PLAN) != PLAN_SHA256 or _sha(shifted_seed.PLAN) != SHIFTED_PLAN_SHA256:
            raise ValueError("fixed seed or shifted-seed plan identity changed")
        archive_before = first_branch._archive_input_identity()
        if _canonical_sha(archive_before) != ARCHIVED_INPUT_SHA256:
            raise ValueError("fixed PR #204 archived input identity changed")

        report.update(execution_phase="fixture", source_before=source_before,
                      archive_before=archive_before)
        _write_json(output, report)
        problem, original, control, parameters, truth, input_before = _prepare_fixed_seed()
        phase_seconds["source_and_fixture"] = time.monotonic() - phase_tick
        report.update(execution_phase="strict_branch", input_before=input_before,
                      layout=problem.layout,
                      candidate={
                          "changed_control_index": EXPECTED_CHANGED_CONTROL,
                          "changed_control_indices": [EXPECTED_CHANGED_CONTROL],
                          "flow_fraction": -0.59,
                          "control_sha256": _tensor_sha(control),
                          "parameters_sha256": _tensor_sha(parameters),
                          "terminal_truth_sha256": _tensor_sha(truth),
                      })
        _write_json(output, report)

        phase_tick = time.monotonic()
        branch, margins = _full_branch_with_margins(problem, control, parameters)
        phase_seconds["full_forecast_and_strict_branch"] = time.monotonic() - phase_tick
        report["branch"] = branch
        report["branch_margins"] = margins
        report["branch_status"] = branch["status"]
        if branch["status"] != "passed_strict_branch":
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status=("branch_refused" if branch["status"] == "branch_refused"
                                            else "branch_diagnostic_incomplete"))
        else:
            report.update(execution_phase="analysis_objective")
            _write_json(output, report)
            phase_tick = time.monotonic()
            objective = problem.objective(control, parameters)
            gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
            phase_seconds["analysis_objective_and_gradient"] = time.monotonic() - phase_tick
            field_count = int(problem.frozen.active_field_index.numel())
            finite_contract = (
                isinstance(objective, Tensor) and objective.shape == ()
                and objective.dtype == torch.float64 and objective.device.type == "cpu"
                and isinstance(gradient, Tensor) and gradient.shape == (EXPECTED_CONTROLS,)
                and gradient.dtype == torch.float64 and gradient.device.type == "cpu"
                and bool(torch.isfinite(objective)) and bool(torch.isfinite(gradient).all())
            )
            if not finite_contract:
                report.update(execution_status="completed", execution_phase="finished",
                              numerical_status="nonfinite_derivative",
                              stationarity_status="not_tested")
            else:
                norms = _gradient_norms(gradient, field_count)
                stationary = norms["all_inf"] < STATIONARITY_TOLERANCE
                report["analysis_objective"] = float(objective)
                report["analysis_euler_stages_by_contract"] = EXPECTED_ANALYSIS_STAGES
                report["gradient_norms"] = norms
                report["stationarity_status"] = (
                    "passed" if stationary else "not_stationary"
                )
                if stationary:
                    phase_tick = time.monotonic()
                    linear = _linear_preflight(problem, control, parameters, gradient)
                    phase_seconds["linear_preflight"] = time.monotonic() - phase_tick
                else:
                    report.update(execution_phase="linear_solve")
                    _write_json(output, report)
                    phase_tick = time.monotonic()
                    linear = _linear_preflight(problem, control, parameters, gradient)
                    phase_seconds["linear_preflight"] = time.monotonic() - phase_tick
                report["linear"] = linear
                report["hvp_calls"] = linear["hvp_calls"]
                report["linear_status"] = linear["status"]
                report["numerical_status"] = (
                    "stationarity_pass" if stationary else linear["status"]
                )
                report.update(execution_status="completed", execution_phase="finished")
    except Exception as error:
        report.update(execution_status="failed", execution_phase="failed",
                      numerical_status="not_reached",
                      error=f"{type(error).__name__}: {error}")
    finally:
        report["child_elapsed_seconds"] = time.monotonic() - child_started
        report["child_phase_seconds"] = phase_seconds
        report["wall_limit_seconds"] = WALL_SECONDS
        report["rss_limit_bytes"] = SAMPLED_RSS_BYTES
        report["termination_reap_grace_seconds"] = REAP_GRACE_SECONDS
        report["source_after"] = _source_hashes()
        report["source_unchanged"] = report["source_after"] == source_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        report["shifted_plan_unchanged"] = _sha(shifted_seed.PLAN) == SHIFTED_PLAN_SHA256
        if (problem is not None and original is not None and control is not None
                and parameters is not None and truth is not None):
            report["input_after"] = _input_identity(
                problem, original, control, parameters, truth,
            )
            report["input_unchanged"] = report["input_after"] == input_before
        else:
            report["input_after"] = None
            report["input_unchanged"] = False
        try:
            report["archive_unchanged"] = (
                archive_before is not None
                and first_branch._archive_input_identity() == archive_before
            )
        except (OSError, ValueError, json.JSONDecodeError):
            report["archive_unchanged"] = False
        if not all(report.get(key) is True for key in (
            "source_unchanged", "plan_unchanged", "shifted_plan_unchanged",
            "input_unchanged", "archive_unchanged",
        )):
            report.update(execution_status="failed", execution_phase="identity_refused",
                          numerical_status="not_reached")
        _write_json(output, report)
    return report


def _valid_resource(value: dict[str, Any], command: list[str]) -> bool:
    elapsed, peak = value.get("elapsed_seconds"), value.get("sampled_peak_rss_bytes")
    return (
        value.get("command") == command and value.get("exit_code") == 0
        and value.get("resource_termination") is None and value.get("monitor_error") is None
        and value.get("wall_limit_seconds") == WALL_SECONDS
        and value.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(value.get("rss_samples")) is int and value["rss_samples"] > 0
        and type(peak) is int and 0 < peak <= SAMPLED_RSS_BYTES
        and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
        and math.isfinite(elapsed) and 0 <= elapsed <= WALL_SECONDS
    )


def _finite_number(value: object, *, minimum: float | None = None) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and (minimum is None or value >= minimum)
    )


def _valid_margins(value: object, *, complete: bool) -> bool:
    if not isinstance(value, dict):
        return False
    stage_count = value.get("observed_stage_count")
    if (value.get("complete") is not complete
            or value.get("expected_stage_count") != EXPECTED_STAGES
            or type(stage_count) is not int
            or not 0 <= stage_count <= EXPECTED_STAGES
            or (complete and stage_count != EXPECTED_STAGES)
            or (not complete and stage_count == EXPECTED_STAGES)
            or (complete and value.get("nonfinite") is not False)):
        return False
    minima = value.get("normalized_minima")
    names = {
        "x_left_slope_over_q_scale", "x_right_slope_over_q_scale",
        "x_active_limiter_gap_over_q_scale", "y_left_slope_over_q_scale",
        "y_right_slope_over_q_scale", "y_active_limiter_gap_over_q_scale",
    }
    if not isinstance(minima, dict) or set(minima) != names:
        return False
    for item in minima.values():
        if item is None:
            continue
        if (not isinstance(item, dict)
                or not _finite_number(item.get("value"), minimum=0.0)
                or type(item.get("stage_index")) is not int
                or not 0 <= item["stage_index"] < EXPECTED_STAGES
                or item.get("stage_context") != first_branch._stage_context(item["stage_index"])):
            return False
    face = value.get("absolute_face_flux_over_global_maximum")
    return (
        isinstance(face, dict)
        and _finite_number(face.get("value"), minimum=0.0)
        and face["value"] <= 1.0
        and _finite_number(face.get("absolute_minimum"), minimum=0.0)
        and _finite_number(face.get("global_maximum_absolute_flux"), minimum=0.0)
        and type(face.get("stage_index")) is int
        and 0 <= face["stage_index"] < EXPECTED_STAGES
        and face.get("stage_context") == first_branch._stage_context(face["stage_index"])
        and face.get("orientation") in ("x", "y")
        and isinstance(face.get("index"), list)
        and len(face["index"]) == 2
        and all(type(component) is int for component in face["index"])
    )


def _valid_child(child: object, expected_sources: dict[str, str],
                 expected_archive: dict[str, Any]) -> bool:
    if not isinstance(child, dict):
        return False
    layout = child.get("layout")
    candidate = child.get("candidate")
    input_before = child.get("input_before")
    branch_record = child.get("branch")
    if not all(isinstance(value, dict) for value in
               (layout, candidate, input_before, branch_record)):
        return False
    layout = cast(dict[str, Any], layout)
    candidate = cast(dict[str, Any], candidate)
    input_before = cast(dict[str, Any], input_before)
    branch_record = cast(dict[str, Any], branch_record)
    phase_seconds = child.get("child_phase_seconds")
    if not isinstance(phase_seconds, dict) or not all(
        isinstance(name, str) and _finite_number(seconds, minimum=0.0)
        for name, seconds in phase_seconds.items()
    ):
        return False
    common = (
        child.get("execution_status") == "completed"
        and child.get("execution_phase") == "finished"
        and child.get("source_before") == child.get("source_after") == expected_sources
        and child.get("source_unchanged") is True
        and child.get("plan_sha256") == PLAN_SHA256 and child.get("plan_unchanged") is True
        and child.get("shifted_plan_sha256") == SHIFTED_PLAN_SHA256
        and child.get("shifted_plan_unchanged") is True
        and child.get("archive_input_sha256") == ARCHIVED_INPUT_SHA256
        and child.get("archive_before") == expected_archive
        and child.get("archive_unchanged") is True
        and child.get("input_before") == child.get("input_after")
        and child.get("input_unchanged") is True
        and child.get("response_computed") is False
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
        and type(child.get("pid")) is int and child["pid"] > 0
        and child.get("wall_limit_seconds") == WALL_SECONDS
        and child.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and child.get("termination_reap_grace_seconds") == REAP_GRACE_SECONDS
        and _finite_number(child.get("child_elapsed_seconds"), minimum=0.0)
        and layout.get("controls") == EXPECTED_CONTROLS
        and layout.get("parameters") == EXPECTED_PARAMETERS
        and layout.get("euler_stages") == EXPECTED_STAGES
        and candidate.get("control_sha256") == SHIFTED_SEED_CONTROL_SHA256
        and candidate.get("parameters_sha256") == EXPECTED_PARAMETERS_SHA256
        and candidate.get("terminal_truth_sha256") == EXPECTED_TRUTH_SHA256
        and input_before.get("archived_input") == expected_archive
        and input_before.get("control_sha256") == SHIFTED_SEED_CONTROL_SHA256
        and input_before.get("parameters_sha256") == EXPECTED_PARAMETERS_SHA256
        and input_before.get("terminal_truth_sha256") == EXPECTED_TRUTH_SHA256
        and input_before.get("changed_control_index") == EXPECTED_CHANGED_CONTROL
        and input_before.get("changed_control_indices") == [EXPECTED_CHANGED_CONTROL]
        and child.get("branch_status") == branch_record.get("status")
    )
    if not common:
        return False
    status = child.get("numerical_status")
    branch = branch_record
    margins = child.get("branch_margins")
    if not isinstance(margins, dict):
        return False
    if status == "branch_refused":
        margins_valid = (
            _valid_margins(margins, complete=True)
            if margins.get("complete") is True
            else _valid_margins(margins, complete=False)
        )
        return (branch.get("status") == "branch_refused"
                and margins_valid
                and child.get("analysis_objective") is None
                and child.get("linear_status") == "not_attempted")
    if status == "branch_diagnostic_incomplete":
        return (branch.get("status") == "branch_diagnostic_incomplete"
                and branch.get("strict_status") == "passed_strict_branch"
                and _valid_margins(margins, complete=False)
                and child.get("analysis_objective") is None
                and child.get("linear_status") == "not_attempted")
    if (branch.get("status") != "passed_strict_branch"
            or not _valid_margins(margins, complete=True)):
        return False
    if status == "stationarity_pass":
        norms = child.get("gradient_norms", {})
        return (child.get("stationarity_status") == "passed"
                and norms.get("all_inf", math.inf) < STATIONARITY_TOLERANCE
                and child.get("linear", {}).get("status") == "skipped_stationary_zero_rhs"
                and child.get("hvp_calls") == 0)
    if status == "nonfinite_derivative":
        if child.get("stationarity_status") == "not_tested":
            return (child.get("analysis_objective") is None
                    and child.get("linear_status") == "not_attempted")
        return (child.get("stationarity_status") == "not_stationary"
                and child.get("gradient_norms", {}).get("all_inf", 0.0)
                    >= STATIONARITY_TOLERANCE
                and child.get("linear", {}).get("status") == "nonfinite_derivative"
                and type(child.get("hvp_calls")) is int and child["hvp_calls"] >= 1)
    if status in {"pcg_refused_non_spd", "pcg_iteration_budget", "linear_pass",
                  "linear_residual_failed"}:
        linear = child.get("linear", {})
        common_linear = (
            child.get("stationarity_status") == "not_stationary"
            and child.get("analysis_objective") is not None
            and child.get("gradient_norms", {}).get("all_inf", 0.0)
                >= STATIONARITY_TOLERANCE
            and linear.get("status") == status
            and linear.get("rtol") == LINEAR_RTOL
            and linear.get("max_iterations") == LINEAR_MAX_ITERATIONS
            and linear.get("preconditioner") == "none"
            and linear.get("curvature_certificate") == "not_proved_by_pcg"
            and child.get("hvp_calls") == linear.get("hvp_calls")
        )
        if not common_linear:
            return False
        if status == "pcg_refused_non_spd":
            return linear.get("refusal") in _KNOWN_PCG_REFUSALS
        if status == "pcg_iteration_budget":
            true_residual = linear.get("true_relative_residual")
            return (linear.get("pcg_converged") is False
                    and isinstance(true_residual, (int, float))
                    and math.isfinite(true_residual))
        true_residual = linear.get("true_relative_residual")
        if not isinstance(true_residual, (int, float)) or not math.isfinite(true_residual):
            return False
        if status == "linear_pass":
            return linear.get("pcg_converged") is True and true_residual <= LINEAR_RTOL
        return linear.get("pcg_converged") is True and true_residual > LINEAR_RTOL
    return False


def run(directory: Path) -> dict[str, Any]:
    """Launch exactly one guarded child; never reconstruct the fixture here."""
    parent_started = time.monotonic()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("seed-linear output directory must be fresh and empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_3h_seed_linear.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output)]
    source_before = _source_hashes()
    if (not first_branch._sources_match_archive(source_before)
            or source_before.get(
                "examples/weather_scenarios/fv_point_3h_shifted_branch_probe.py"
            ) != SHIFTED_HELPER_SHA256
            or _sha(PLAN) != PLAN_SHA256
            or _sha(shifted_seed.PLAN) != SHIFTED_PLAN_SHA256):
        raise ValueError("seed-linear source or plan identity changed before launch")
    archive_identity = first_branch._archive_input_identity()
    parent_prelaunch_seconds = time.monotonic() - parent_started
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "point_3h_seed_linear.resource.json",
        log_path=directory / "point_3h_seed_linear.log",
    )
    child: object = None
    read_error = None
    try:
        loaded = json.loads(output.read_text())
        child = loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    current_sources = _source_hashes()
    try:
        archive_unchanged = first_branch._archive_input_identity() == archive_identity
    except (OSError, ValueError, json.JSONDecodeError):
        archive_unchanged = False
    resource_ok = _valid_resource(resource, command)
    child_ok = (
        _valid_child(child, source_before, archive_identity)
        and isinstance(child, dict)
        and child.get("pid") == resource.get("child_pid")
        and current_sources == source_before
        and _sha(PLAN) == PLAN_SHA256
        and _sha(shifted_seed.PLAN) == SHIFTED_PLAN_SHA256
        and archive_unchanged
    )
    result: dict[str, Any] = {
        "execution_status": "completed" if resource_ok and child_ok else "failed",
        "execution_phase": "finished" if resource_ok and child_ok else "guard_or_evidence_failure",
        "numerical_status": child.get("numerical_status", "not_reached")
        if isinstance(child, dict) else "not_reached",
        "branch_status": child.get("branch_status", "not_performed")
        if isinstance(child, dict) else "not_performed",
        "stationarity_status": child.get("stationarity_status", "not_tested")
        if isinstance(child, dict) else "not_tested",
        "linear_status": child.get("linear_status", "not_attempted")
        if isinstance(child, dict) else "not_attempted",
        "child_elapsed_seconds": child.get("child_elapsed_seconds")
        if isinstance(child, dict) else None,
        "child_phase_seconds": child.get("child_phase_seconds")
        if isinstance(child, dict) else None,
        "parent_prelaunch_seconds": parent_prelaunch_seconds,
        "guarded_wait_seconds": resource.get("elapsed_seconds"),
        "parent_elapsed_seconds": time.monotonic() - parent_started,
        "child_read_error": read_error,
        "resource": resource,
        "source_sha256": source_before,
        "source_unchanged": current_sources == source_before,
        "plan_sha256": PLAN_SHA256,
        "shifted_plan_sha256": SHIFTED_PLAN_SHA256,
        "archive_unchanged": archive_unchanged,
        "child_report_valid": bool(child_ok),
        "scope": "fixed PR #218 point 3-hour seed; strict branch, analysis stationarity and one linear RHS only; no step, adjoint or response",
    }
    _write_json(directory / "point_3h_seed_linear.run.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = run_probe(args.output)
        code = 0 if result["execution_status"] == "completed" else 2
    elif args.directory is not None:
        result = run(args.directory)
        code = 0 if result["execution_status"] == "completed" else 1
    else:
        parser.error("provide --output for child mode or --directory for guarded mode")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(code)
