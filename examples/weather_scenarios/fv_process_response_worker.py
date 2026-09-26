"""One source-bound FV local response in its own Python process."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
from uuid import UUID

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import matrix_free, transport
from advar.local_response import compute_local_response
from examples.weather_scenarios import fv_concurrent_response_probe as shared


SOURCE_PATHS = shared.SOURCE_PATHS + (
    "examples/weather_scenarios/fv_process_response_worker.py",
    "examples/weather_scenarios/fv_process_response_runner.py",
)


def _hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def _canonical_attempt_id(value: str) -> str:
    """Require the lowercase canonical spelling of an RFC 4122 UUID4."""
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("attempt_id must be a canonical UUID4") from error
    if str(parsed) != value or parsed.version != 4 or parsed.variant != "specified in RFC 4122":
        raise ValueError("attempt_id must be a canonical UUID4")
    return value


def _parse_attempt_id(value: str) -> str:
    try:
        return _canonical_attempt_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def run(case_id: str, output: Path, attempt_id: str | None = None) -> dict[str, Any]:
    if attempt_id is not None:
        attempt_id = _canonical_attempt_id(attempt_id)
    if case_id not in ("fv4x5", "fv8x10"):
        raise ValueError("worker case must be fv4x5 or fv8x10")
    source_before = _hashes()
    problem, control, parameters, direction, expected = (
        shared._small_case() if case_id == "fv4x5" else shared._large_case()
    )
    identity = {
        "problem": problem.identity["fixed_problem_sha256"],
        "control": shared._tensor_identity(control),
        "parameters": shared._tensor_identity(parameters),
        "verification": shared._tensor_identity(problem.verification),
        "direction": shared._tensor_identity(direction),
    }
    report: dict[str, Any] = {
        "case_id": case_id, "status": "running", "phase": "response",
        "pid": os.getpid(), "executable": sys.executable,
        "python": platform.python_version(), "torch": torch.__version__,
        "source_before": source_before,
        "archived_report_sha256": shared.ARCHIVED,
        "input_identity": identity, "nonlinear_analyses": 0,
        "nonlinear_reanalyses": 0, "physical_validation": "not_performed",
    }
    if attempt_id is not None:
        report["attempt_id"] = attempt_id

    def save() -> None:
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    save()
    try:
        stage_count = 0

        def count_stage(q: Tensor, qx: Tensor, qy: Tensor) -> None:
            nonlocal stage_count
            stage_count += 1

        def checked_branch(c: Tensor, p: Tensor):
            with transport.observe_minmod_stages(count_stage):
                return problem.branch_check(c, p)

        solves: list[dict[str, Any]] = []

        def monitor(original):
            def solve(operator, rhs, **kwargs):
                entry: dict[str, Any] = {
                    "rhs": shared._tensor_identity(rhs),
                    "rtol": kwargs.get("rtol"),
                    "max_iterations": kwargs.get("max_iterations"),
                    "hvp_calls": 0,
                    "hvp_event_times": [],
                    "hvp_intervals": [],
                }
                started = time.monotonic()

                def measured(value):
                    hvp_started = time.monotonic()
                    result = operator(value)
                    hvp_ended = time.monotonic()
                    entry["hvp_calls"] += 1
                    entry["hvp_event_times"].append(hvp_ended)
                    entry["hvp_intervals"].append([hvp_started, hvp_ended])
                    return result

                try:
                    result = original(measured, rhs, **kwargs)
                    entry.update(converged=result.converged, iterations=result.iterations,
                                 relative_residual=result.relative_residual)
                    return result
                finally:
                    entry["seconds"] = time.monotonic() - started
                    solves.append(entry)

            return solve

        report["response_started_monotonic"] = time.monotonic()
        with matrix_free.observe_pcg_calls(monitor):
            response = compute_local_response(
                problem.objective, problem.score, control, parameters,
                {"middle_time_bias": direction},
                branch_check=checked_branch, input_identity=identity,
            )
        report["response_ended_monotonic"] = time.monotonic()
        total = float(response.total["middle_time_bias"])
        relative_difference = abs(total - expected) / abs(expected)
        expected_stages = 54 if case_id == "fv4x5" else 108
        expected_branch = problem.expected_branch
        if expected_branch is None:
            raise ValueError("worker requires a pinned archived branch")
        choices_sha256 = hashlib.sha256(json.dumps(
            response.branch_signature["choices"], sort_keys=True
        ).encode()).hexdigest()
        faces_sha256 = hashlib.sha256(json.dumps(
            response.branch_signature["face_signs"], sort_keys=True
        ).encode()).hexdigest()
        expected_choices_sha256 = hashlib.sha256(json.dumps(
            expected_branch["choices"], sort_keys=True
        ).encode()).hexdigest()
        expected_faces_sha256 = hashlib.sha256(json.dumps(
            expected_branch["face_signs"], sort_keys=True
        ).encode()).hexdigest()
        if (response.gradient_max >= 1e-10
                or response.true_adjoint_relative_residual > 1e-10
                or response.branch_signature["euler_stages"] != expected_stages
                or stage_count != expected_stages
                or choices_sha256 != expected_choices_sha256
                or faces_sha256 != expected_faces_sha256
                or relative_difference > 1e-6
                or len(solves) != 1
                or not solves[0].get("converged")
                or solves[0]["hvp_calls"] != response.hvp_count
                or solves[0]["iterations"] != response.pcg_iterations
                or solves[0]["relative_residual"] != response.pcg_relative_residual):
            raise ValueError(f"{case_id} response or diagnostic gate failed")
        report["response"] = {
            "branch_euler_stages": response.branch_signature["euler_stages"],
            "stage_observer_events": stage_count,
            "branch_choices_sha256": choices_sha256,
            "archived_branch_choices_sha256": expected_choices_sha256,
            "branch_face_signs_sha256": faces_sha256,
            "archived_branch_face_signs_sha256": expected_faces_sha256,
            "gradient_max": response.gradient_max,
            "true_adjoint_residual": response.true_adjoint_residual,
            "true_adjoint_relative_residual": response.true_adjoint_relative_residual,
            "pcg_relative_residual": response.pcg_relative_residual,
            "pcg_iterations": response.pcg_iterations,
            "hvp_count": response.hvp_count,
            "direct": float(response.direct["middle_time_bias"]),
            "indirect": float(response.indirect["middle_time_bias"]),
            "total": total, "archived_total": expected,
            "relative_difference": relative_difference,
            "pcg_monitor": solves[0],
        }
        source_after = _hashes()
        report["source_after"] = source_after
        identity_after = {
            "problem": problem.identity["fixed_problem_sha256"],
            "control": shared._tensor_identity(control),
            "parameters": shared._tensor_identity(parameters),
            "verification": shared._tensor_identity(problem.verification),
            "direction": shared._tensor_identity(direction),
        }
        report["input_identity_after"] = identity_after
        if (source_before != source_after
                or identity != identity_after
                or any(shared._hash(shared.EVIDENCE / name) != expected
                       for name, expected in shared.ARCHIVED.items())):
            raise RuntimeError("worker source or fixed input changed during response")
        report.update(status="completed", phase="finished", source_unchanged=True,
                      inputs_unchanged=True)
    except Exception as error:
        report.update(status="execution_error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("fv4x5", "fv8x10"), required=True)
    parser.add_argument("--attempt-id", type=_parse_attempt_id)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.case, arguments.output, arguments.attempt_id)
