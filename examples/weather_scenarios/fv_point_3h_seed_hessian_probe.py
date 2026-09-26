"""One guarded exact-Hessian audit at the fixed shifted three-hour seed."""
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
from examples.weather_scenarios import fv_point_3h_shifted_branch_probe as shifted_seed


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_3H_SEED_HESSIAN_PLAN.md"
PLAN_SHA256 = "06af1a0aa1a17795cfa685a3117a66cd1895c0a851f67694abe3efaa122af831"
ARCHIVED_INPUT_SHA256 = seed_linear.ARCHIVED_INPUT_SHA256
CONTROL_SHA256 = seed_linear.SHIFTED_SEED_CONTROL_SHA256
PARAMETERS_SHA256 = seed_linear.EXPECTED_PARAMETERS_SHA256
TRUTH_SHA256 = seed_linear.EXPECTED_TRUTH_SHA256
WALL_SECONDS = 600
REAP_GRACE_SECONDS = 2
SAMPLED_RSS_BYTES = 1024**3
EXPECTED_CONTROLS = 26
EXPECTED_STAGES = 3600
SYMMETRY_TOLERANCE = 1.0e-8
EIGENPAIR_TOLERANCE = 1.0e-8
CURVATURE_RELATIVE_TOLERANCE = 1.0e-8
ROUNDING_TOLERANCE_MULTIPLIER = 128.0
DEFAULT_DIRECTORY = EVIDENCE / "point_3h_seed_hessian_attempt1"
SOURCE_PATHS = tuple(dict.fromkeys((*seed_linear.SOURCE_PATHS,
                                    "examples/weather_scenarios/fv_point_3h_seed_hessian_probe.py")))


class _InvalidHVP(ValueError):
    """An exact HVP violates its declared tensor contract."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _hessian_audit(problem: Any, control: Tensor, parameters: Tensor) -> dict[str, Any]:
    """Build exact HVP columns, then qualify the least-curved eigenpair freshly."""
    gradient = torch.func.grad(problem.objective, argnums=0)
    calls = 0

    def hvp(direction: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        value = torch.func.jvp(
            lambda c: gradient(c, parameters), (control,), (direction,),
        )[1]
        if (not isinstance(value, Tensor) or value.shape != control.shape
                or value.dtype != control.dtype or value.device != control.device
                or not bool(torch.isfinite(value).all())):
            raise _InvalidHVP("exact Hessian-vector product violates finite tensor contract")
        return value

    result: dict[str, Any] = {
        "status": "running", "hvp_column_count": EXPECTED_CONTROLS,
        "hvp_calls": 0, "symmetry_tolerance": SYMMETRY_TOLERANCE,
        "eigenpair_residual_tolerance": EIGENPAIR_TOLERANCE,
        "curvature_relative_tolerance": CURVATURE_RELATIVE_TOLERANCE,
        "hessian_shape": [EXPECTED_CONTROLS, EXPECTED_CONTROLS],
        "hessian_dtype": str(control.dtype), "hessian_device": str(control.device),
    }
    try:
        basis = torch.eye(control.numel(), dtype=control.dtype, device=control.device)
        columns = [hvp(basis[index]) for index in range(control.numel())]
        matrix = torch.stack(columns, dim=1)
        result["hvp_calls"] = calls
        if (matrix.shape != (EXPECTED_CONTROLS, EXPECTED_CONTROLS)
                or matrix.dtype != control.dtype or matrix.device != control.device
                or not bool(torch.isfinite(matrix).all())):
            result.update(status="nonfinite_or_invalid_hessian")
            return result
        norm = torch.linalg.matrix_norm(matrix)
        norm_value = float(norm)
        result["hessian_frobenius_norm"] = norm_value
        result["hessian_sha256"] = _tensor_sha(matrix)
        if not math.isfinite(norm_value):
            result.update(status="nonfinite_hessian_scale", inertia_assigned=False,
                          symmetry_relative=None, eigenvalues=None,
                          lambda_min=None, lambda_max=None, lambda_ratio=None)
            return result
        if norm_value == 0.0:
            result.update(status="zero_scale_inconclusive", inertia_assigned=False,
                          symmetry_relative=None, eigenvalues=None,
                          lambda_min=None, lambda_max=None, lambda_ratio=None)
            return result

        symmetry = float(torch.linalg.matrix_norm(matrix - matrix.T) / norm)
        result["symmetry_relative"] = symmetry
        if not math.isfinite(symmetry) or symmetry > SYMMETRY_TOLERANCE:
            result.update(status="symmetry_failed", inertia_assigned=False,
                          eigenvalues=None, lambda_min=None, lambda_max=None,
                          lambda_ratio=None)
            return result

        symmetric = 0.5 * (matrix + matrix.T)
        try:
            eigenvalues_only = torch.linalg.eigvalsh(symmetric)
            eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        except RuntimeError as error:
            result.update(status="eigensolver_failed",
                          refusal=f"{type(error).__name__}: {error}",
                          inertia_assigned=False)
            return result
        if (not bool(torch.isfinite(eigenvalues_only).all())
                or not bool(torch.isfinite(eigenvalues).all())
                or not bool(torch.isfinite(eigenvectors).all())):
            result.update(status="eigensolver_nonfinite", inertia_assigned=False)
            return result
        if not torch.allclose(eigenvalues_only, eigenvalues, rtol=1e-12, atol=1e-14):
            result.update(status="eigensolver_inconsistent", inertia_assigned=False)
            return result
        minimum, maximum = float(eigenvalues[0]), float(eigenvalues[-1])
        spectral_scale = float(eigenvalues.abs().max())
        result.update(
            eigenvalues=[float(item) for item in eigenvalues],
            lambda_min=minimum,
            lambda_max=maximum,
            spectral_scale=spectral_scale,
            lambda_ratio=(minimum / maximum
                          if maximum != 0.0 and math.isfinite(maximum) else None),
        )
        if spectral_scale == 0.0:
            result.update(status="zero_spectral_scale_inconclusive", inertia_assigned=False)
            return result

        minimum_vector = eigenvectors[:, 0].clone()
        pivot = int(torch.argmax(minimum_vector.abs()))
        if float(minimum_vector[pivot]) < 0.0:
            minimum_vector = -minimum_vector
        result["minimum_eigenvector_sha256"] = _tensor_sha(minimum_vector)

        fresh_product = hvp(minimum_vector)
        result["hvp_calls"] = calls
        residual = fresh_product - eigenvalues[0] * minimum_vector
        residual_norm = float(torch.linalg.vector_norm(residual))
        relative_residual = float(torch.linalg.vector_norm(residual) / norm)
        asymmetry_allowance = float(torch.linalg.matrix_norm(matrix - matrix.T)) / 2.0
        rounding_allowance = (
            ROUNDING_TOLERANCE_MULTIPLIER * torch.finfo(control.dtype).eps * norm_value
        )
        uncertainty = max(
            CURVATURE_RELATIVE_TOLERANCE * spectral_scale,
            asymmetry_allowance + residual_norm + rounding_allowance,
        )
        result["fresh_eigenpair_relative_residual"] = relative_residual
        result["fresh_eigenpair_residual_norm"] = residual_norm
        result["symmetry_absolute_allowance"] = asymmetry_allowance
        result["rounding_allowance"] = rounding_allowance
        result["curvature_threshold"] = uncertainty
        if not math.isfinite(relative_residual) or relative_residual > EIGENPAIR_TOLERANCE:
            result.update(status="eigenpair_residual_failed", inertia_assigned=False)
            return result

        if minimum < -uncertainty:
            verdict = "locally_negative_curvature"
        elif minimum > uncertainty:
            verdict = "locally_positive_curvature"
        else:
            verdict = "near_zero_inconclusive"
        result.update(status="curvature_verified", curvature_verdict=verdict,
                      inertia_assigned=True)
        return result
    except _InvalidHVP as error:
        result.update(status="hvp_or_eigensolver_failed",
                      refusal=f"{type(error).__name__}: {error}", hvp_calls=calls)
        return result


def _input_expected(identity: object, archive: dict[str, Any]) -> bool:
    if not isinstance(identity, dict):
        return False
    return (
        identity.get("archived_input") == archive
        and identity.get("control_sha256") == CONTROL_SHA256
        and identity.get("parameters_sha256") == PARAMETERS_SHA256
        and identity.get("terminal_truth_sha256") == TRUTH_SHA256
        and identity.get("changed_control_index") == seed_linear.EXPECTED_CHANGED_CONTROL
        and identity.get("changed_control_indices") == [seed_linear.EXPECTED_CHANGED_CONTROL]
        and _canonical_sha(identity.get("archived_input")) == ARCHIVED_INPUT_SHA256
    )


def run_probe(output: Path) -> dict[str, Any]:
    """Reconstruct and audit the pinned seed wholly within the guarded child."""
    started = time.monotonic()
    report: dict[str, Any] = {
        "pid": os.getpid(), "execution_status": "running", "execution_phase": "source_check",
        "numerical_status": "not_reached", "branch_status": "not_performed",
        "gradient_status": "not_tested", "hessian_status": "not_attempted",
        "newton_steps": 0, "pcg_calls": 0, "terminal_score_computed": False,
        "response_computed": False, "adjoint_computed": False,
        "plan_sha256": PLAN_SHA256, "archive_input_sha256": ARCHIVED_INPUT_SHA256,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "device": "CPU FP64"},
    }
    source_before: dict[str, str] = {}
    archive_before: dict[str, Any] | None = None
    input_before: dict[str, Any] | None = None
    phase_seconds: dict[str, float] = {}
    problem = original = control = parameters = truth = None
    _write_json(output, report)
    try:
        tick = time.monotonic()
        source_before = _source_hashes()
        if (not first_branch._sources_match_archive(source_before)
                or _sha(PLAN) != PLAN_SHA256):
            raise ValueError("pinned source or Hessian plan identity changed")
        archive_before = first_branch._archive_input_identity()
        if _canonical_sha(archive_before) != ARCHIVED_INPUT_SHA256:
            raise ValueError("pinned three-hour archived input changed")
        report.update(execution_phase="fixture", source_before=source_before,
                      archive_before=archive_before)
        _write_json(output, report)

        problem, original, control, parameters, truth, input_before = seed_linear._prepare_fixed_seed()
        if not _input_expected(input_before, archive_before):
            raise ValueError("shifted candidate/input identity differs from pinned records")
        phase_seconds["source_and_fixture"] = time.monotonic() - tick
        report.update(execution_phase="strict_branch", input_before=input_before,
                      layout=problem.layout,
                      candidate={"control_sha256": CONTROL_SHA256,
                                 "parameters_sha256": PARAMETERS_SHA256,
                                 "terminal_truth_sha256": TRUTH_SHA256,
                                 "changed_control_index": seed_linear.EXPECTED_CHANGED_CONTROL})
        _write_json(output, report)

        tick = time.monotonic()
        branch, margins = seed_linear._full_branch_with_margins(problem, control, parameters)
        phase_seconds["strict_branch"] = time.monotonic() - tick
        report["branch"] = branch
        report["branch_margins"] = margins
        report["branch_status"] = branch.get("status")
        if branch.get("status") != "passed_strict_branch":
            report.update(execution_status="completed", execution_phase="finished",
                          numerical_status="branch_refused"
                          if branch.get("status") == "branch_refused"
                          else "branch_diagnostic_incomplete")
        else:
            report.update(execution_phase="analysis_gradient")
            _write_json(output, report)
            tick = time.monotonic()
            objective = problem.objective(control, parameters)
            gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
            phase_seconds["analysis_objective_and_gradient"] = time.monotonic() - tick
            if (not isinstance(objective, Tensor) or objective.shape != ()
                    or objective.dtype != torch.float64 or objective.device.type != "cpu"
                    or gradient.shape != (EXPECTED_CONTROLS,) or gradient.dtype != torch.float64
                    or gradient.device.type != "cpu" or not bool(torch.isfinite(objective))
                    or not bool(torch.isfinite(gradient).all())):
                report.update(execution_status="completed", execution_phase="finished",
                              numerical_status="nonfinite_or_invalid_gradient",
                              gradient_status="failed")
            else:
                report["analysis_objective"] = float(objective)
                report["gradient_sha256"] = _tensor_sha(gradient)
                report["gradient_inf_norm"] = float(gradient.abs().max())
                report["gradient_status"] = "finite"
                report.update(execution_phase="exact_hessian")
                _write_json(output, report)
                tick = time.monotonic()
                hessian = _hessian_audit(problem, control, parameters)
                phase_seconds["exact_hessian_audit"] = time.monotonic() - tick
                report["hessian"] = hessian
                report["hessian_status"] = hessian["status"]
                report.update(execution_status="completed", execution_phase="finished",
                              numerical_status=hessian["status"])
    except Exception as error:
        report.update(execution_status="failed", execution_phase="failed",
                      numerical_status="not_reached",
                      error=f"{type(error).__name__}: {error}")
    finally:
        report["child_elapsed_seconds"] = time.monotonic() - started
        report["child_phase_seconds"] = phase_seconds
        report["wall_limit_seconds"] = WALL_SECONDS
        report["rss_limit_bytes"] = SAMPLED_RSS_BYTES
        report["termination_reap_grace_seconds"] = REAP_GRACE_SECONDS
        report["source_after"] = _source_hashes()
        report["source_unchanged"] = report["source_after"] == source_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        if (problem is not None and original is not None and control is not None
                and parameters is not None and truth is not None):
            report["input_after"] = seed_linear._input_identity(
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
            "source_unchanged", "plan_unchanged", "input_unchanged", "archive_unchanged",
        )):
            report.update(execution_status="failed", execution_phase="identity_refused",
                          numerical_status="identity_changed")
        _write_json(output, report)
    return report


def _finite_number(value: object, *, minimum: float | None = None) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (minimum is None or value >= minimum))


def _finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return float(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-24)


def _valid_curvature_claim(hessian: dict[str, Any]) -> bool:
    eigenvalues = hessian.get("eigenvalues")
    if (not isinstance(eigenvalues, list) or len(eigenvalues) != EXPECTED_CONTROLS
            or not all(_finite_number(value) for value in eigenvalues)):
        return False
    values = [float(value) for value in eigenvalues]
    if values != sorted(values):
        return False

    minimum_value = _finite_float(hessian.get("lambda_min"))
    maximum_value = _finite_float(hessian.get("lambda_max"))
    scale_value = _finite_float(hessian.get("spectral_scale"))
    matrix_norm_value = _finite_float(hessian.get("hessian_frobenius_norm"))
    if (minimum_value is None or maximum_value is None
            or scale_value is None or matrix_norm_value is None):
        return False
    minimum = float(minimum_value)
    maximum = float(maximum_value)
    scale = float(scale_value)
    matrix_norm = float(matrix_norm_value)
    if (matrix_norm <= 0.0 or scale <= 0.0 or not _close(minimum, values[0])
            or not _close(maximum, values[-1])
            or not _close(scale, max(abs(value) for value in values))):
        return False

    ratio = hessian.get("lambda_ratio")
    ratio_value = _finite_float(ratio)
    if maximum == 0.0:
        if ratio is not None:
            return False
    elif ratio_value is None or not _close(ratio_value, minimum / maximum):
        return False

    symmetry = hessian.get("symmetry_relative")
    residual_relative = hessian.get("fresh_eigenpair_relative_residual")
    residual_norm = hessian.get("fresh_eigenpair_residual_norm")
    symmetry_allowance = hessian.get("symmetry_absolute_allowance")
    rounding_allowance = hessian.get("rounding_allowance")
    threshold = hessian.get("curvature_threshold")
    symmetry_value = _finite_float(symmetry)
    residual_relative_value = _finite_float(residual_relative)
    residual_norm_value = _finite_float(residual_norm)
    symmetry_allowance_value = _finite_float(symmetry_allowance)
    rounding_allowance_value = _finite_float(rounding_allowance)
    threshold_value = _finite_float(threshold)
    components = (symmetry_value, residual_relative_value, residual_norm_value,
                  symmetry_allowance_value, rounding_allowance_value, threshold_value)
    if any(value is None or value < 0.0 for value in components):
        return False
    assert symmetry_value is not None and residual_relative_value is not None
    assert residual_norm_value is not None and symmetry_allowance_value is not None
    assert rounding_allowance_value is not None and threshold_value is not None
    if (symmetry_value > SYMMETRY_TOLERANCE
            or residual_relative_value > EIGENPAIR_TOLERANCE
            or not _close(residual_relative_value, residual_norm_value / matrix_norm)
            or not _close(symmetry_allowance_value, symmetry_value * matrix_norm / 2.0)
            or not _close(
                rounding_allowance_value,
                ROUNDING_TOLERANCE_MULTIPLIER * torch.finfo(torch.float64).eps * matrix_norm,
            )):
        return False
    declared_floor = max(
        CURVATURE_RELATIVE_TOLERANCE * scale,
        symmetry_allowance_value + residual_norm_value + rounding_allowance_value,
    )
    delta = threshold_value
    if not _close(delta, declared_floor):
        return False
    expected_verdict = (
        "locally_negative_curvature" if minimum < -delta
        else "locally_positive_curvature" if minimum > delta
        else "near_zero_inconclusive"
    )
    return (
        hessian.get("curvature_verdict") == expected_verdict
        and hessian.get("inertia_assigned") is True
        and isinstance(hessian.get("hessian_sha256"), str)
        and len(hessian["hessian_sha256"]) == 64
        and isinstance(hessian.get("minimum_eigenvector_sha256"), str)
        and len(hessian["minimum_eigenvector_sha256"]) == 64
        and hessian.get("hvp_calls") == EXPECTED_CONTROLS + 1
    )


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


def _valid_child(child: object, source: dict[str, str], archive: dict[str, Any]) -> bool:
    if not isinstance(child, dict):
        return False
    layout = child.get("layout")
    input_before = child.get("input_before")
    branch = child.get("branch")
    margins = child.get("branch_margins")
    phase_seconds = child.get("child_phase_seconds")
    common = (
        child.get("execution_status") == "completed"
        and child.get("execution_phase") == "finished"
        and child.get("source_before") == child.get("source_after") == source
        and child.get("source_unchanged") is True
        and child.get("plan_sha256") == PLAN_SHA256 and child.get("plan_unchanged") is True
        and child.get("archive_input_sha256") == ARCHIVED_INPUT_SHA256
        and child.get("archive_before") == archive and child.get("archive_unchanged") is True
        and input_before == child.get("input_after") and child.get("input_unchanged") is True
        and _input_expected(input_before, archive)
        and isinstance(layout, dict) and layout.get("controls") == EXPECTED_CONTROLS
        and layout.get("parameters") == seed_linear.EXPECTED_PARAMETERS
        and layout.get("euler_stages") == EXPECTED_STAGES
        and isinstance(branch, dict) and isinstance(margins, dict)
        and isinstance(phase_seconds, dict)
        and all(_finite_number(value, minimum=0.0) for value in phase_seconds.values())
        and child.get("terminal_score_computed") is False
        and child.get("response_computed") is False
        and child.get("adjoint_computed") is False
        and child.get("newton_steps") == 0 and child.get("pcg_calls") == 0
        and child.get("pid", 0) > 0
        and child.get("wall_limit_seconds") == WALL_SECONDS
        and child.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and child.get("termination_reap_grace_seconds") == REAP_GRACE_SECONDS
        and _finite_number(child.get("child_elapsed_seconds"), minimum=0.0)
    )
    if not common:
        return False
    if not isinstance(branch, dict) or not isinstance(margins, dict):
        return False
    status = child.get("numerical_status")
    if child.get("branch_status") != branch.get("status"):
        return False
    if status in {"branch_refused", "branch_diagnostic_incomplete"}:
        margins_valid = (
            type(margins.get("observed_stage_count")) is int
            and 0 <= margins["observed_stage_count"] <= EXPECTED_STAGES
            and margins.get("expected_stage_count") == EXPECTED_STAGES
            and margins.get("complete") is (margins["observed_stage_count"] == EXPECTED_STAGES)
        )
        return (branch.get("status") == status and margins_valid
                and child.get("gradient_status") == "not_tested"
                and child.get("hessian_status") == "not_attempted")
    hessian = child.get("hessian")
    if (branch.get("status") != "passed_strict_branch"
            or branch.get("euler_stages") != EXPECTED_STAGES
            or branch.get("choice_stage_count") != EXPECTED_STAGES
            or branch.get("face_sign_stage_count") != EXPECTED_STAGES
            or not isinstance(branch.get("signature_sha256"), str)
            or len(branch["signature_sha256"]) != 64
            or margins.get("complete") is not True
            or margins.get("observed_stage_count") != EXPECTED_STAGES
            or margins.get("expected_stage_count") != EXPECTED_STAGES
            or child.get("gradient_status") != "finite"
            or not isinstance(hessian, dict)
            or hessian.get("status") != status
            or hessian.get("hvp_column_count") != EXPECTED_CONTROLS
            or hessian.get("hessian_shape") != [EXPECTED_CONTROLS, EXPECTED_CONTROLS]
            or hessian.get("hessian_dtype") != "torch.float64"
            or hessian.get("hessian_device") != "cpu"):
        return False
    if status == "curvature_verified":
        return _valid_curvature_claim(hessian)
    if status == "symmetry_failed":
        return (hessian.get("inertia_assigned") is False
                and hessian.get("eigenvalues") is None
                and _finite_number(hessian.get("symmetry_relative"), minimum=0.0)
                and hessian["symmetry_relative"] > SYMMETRY_TOLERANCE)
    if status in {"zero_scale_inconclusive", "zero_spectral_scale_inconclusive"}:
        return hessian.get("inertia_assigned") is False
    return status in {"nonfinite_or_invalid_hessian", "nonfinite_hessian_scale",
                      "eigensolver_failed", "eigensolver_nonfinite",
                      "eigensolver_inconsistent", "eigenpair_residual_failed",
                      "hvp_or_eigensolver_failed"}


def run(directory: Path) -> dict[str, Any]:
    """Launch one child; never rebuild the FV fixture in the parent."""
    started = time.monotonic()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("seed-Hessian output directory must be fresh and empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_3h_seed_hessian.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output)]
    source_before = _source_hashes()
    if not first_branch._sources_match_archive(source_before) or _sha(PLAN) != PLAN_SHA256:
        raise ValueError("seed-Hessian source or plan identity changed before launch")
    archive_before = first_branch._archive_input_identity()
    if _canonical_sha(archive_before) != ARCHIVED_INPUT_SHA256:
        raise ValueError("seed-Hessian archived input identity changed before launch")
    prelaunch_seconds = time.monotonic() - started
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "point_3h_seed_hessian.resource.json",
        log_path=directory / "point_3h_seed_hessian.log",
    )
    try:
        loaded = json.loads(output.read_text())
        child = loaded if isinstance(loaded, dict) else None
        read_error = None
    except (OSError, json.JSONDecodeError) as error:
        child = None
        read_error = f"{type(error).__name__}: {error}"
    current_sources = _source_hashes()
    try:
        archive_unchanged = first_branch._archive_input_identity() == archive_before
    except (OSError, ValueError, json.JSONDecodeError):
        archive_unchanged = False
    resource_ok = _valid_resource(resource, command)
    child_ok = (
        _valid_child(child, source_before, archive_before)
        and isinstance(child, dict) and child.get("pid") == resource.get("child_pid")
    )
    plan_unchanged = _sha(PLAN) == PLAN_SHA256
    source_unchanged = current_sources == source_before
    environment_ok = source_unchanged and plan_unchanged and archive_unchanged
    execution_ok = resource_ok and child_ok and environment_ok
    result: dict[str, Any] = {
        "execution_status": "completed" if execution_ok else "failed",
        "execution_phase": "finished" if execution_ok else "guard_or_evidence_failure",
        "numerical_status": child.get("numerical_status", "not_reached")
        if isinstance(child, dict) else "not_reached",
        "branch_status": child.get("branch_status", "not_performed")
        if isinstance(child, dict) else "not_performed",
        "gradient_status": child.get("gradient_status", "not_tested")
        if isinstance(child, dict) else "not_tested",
        "hessian_status": child.get("hessian_status", "not_attempted")
        if isinstance(child, dict) else "not_attempted",
        "child_elapsed_seconds": child.get("child_elapsed_seconds")
        if isinstance(child, dict) else None,
        "child_phase_seconds": child.get("child_phase_seconds")
        if isinstance(child, dict) else None,
        "parent_prelaunch_seconds": prelaunch_seconds,
        "guarded_wait_seconds": resource.get("elapsed_seconds"),
        "parent_elapsed_seconds": time.monotonic() - started,
        "child_read_error": read_error,
        "resource": resource, "source_sha256": source_before,
        "source_unchanged": source_unchanged,
        "plan_sha256": PLAN_SHA256, "plan_unchanged": plan_unchanged,
        "archive_unchanged": archive_unchanged,
        "child_report_valid": bool(child_ok),
        "scope": "one fixed PR #218 shifted point seed; strict branch and exact Hessian audit only; no step, PCG, score or response",
    }
    _write_json(directory / "point_3h_seed_hessian.run.json", result)
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
