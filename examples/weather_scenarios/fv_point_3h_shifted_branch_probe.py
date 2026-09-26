"""One fixed alternate-control feasibility check for the point 3-hour branch."""
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

from advar import transport
from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios import fv_point_3h_first_branch_probe as first_branch
from examples.weather_scenarios.fv_point_3h_forward_case import make_case


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_3H_SHIFTED_BRANCH_PLAN.md"
PLAN_SHA256 = "3fa9bc123e9aedddfb027abd373960e71db0b7b7693bb83a3877bbc35d8db6f6"
ARCHIVED_INPUT_SHA256 = first_branch.ARCHIVED_INPUT_SHA256
WALL_SECONDS = 180
SAMPLED_RSS_BYTES = 768 * 1024**2
EXPECTED_STAGES = 3600
EXPECTED_FRACTIONS = (0.7, -0.6, 0.5, 0.4, -0.3)
SHIFTED_FRACTIONS = (0.7, -0.59, 0.5, 0.4, -0.3)
FLOW_FRACTION_INDEX = 1
DEFAULT_DIRECTORY = EVIDENCE / "point_3h_shifted_branch_attempt1"
SOURCE_PATHS = tuple(dict.fromkeys((
    *first_branch.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_3h_forward_probe.py",
    "examples/weather_scenarios/fv_point_3h_forward_case.py",
    "examples/weather_scenarios/fv_point_3h_shifted_branch_probe.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _prepare_shifted_case() -> tuple[Any, Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    problem, original_control, parameters, truth = make_case()
    base_identity = first_branch._input_identity(problem, original_control, parameters, truth)
    if _canonical_sha(base_identity) != ARCHIVED_INPUT_SHA256:
        raise ValueError("fixed PR #204 input identity changed")

    field_size = problem.frozen.active_field_index.numel()
    flow_offset = field_size
    flow_index = flow_offset + FLOW_FRACTION_INDEX
    spec = problem.frozen.fv_transport
    if (spec is None or original_control.shape != (26,) or parameters.shape != (13,)
            or problem.layout["euler_stages"] != EXPECTED_STAGES
            or problem.frozen.nowcast_config.forecast_steps != 18
            or flow_index >= original_control.numel()):
        raise ValueError("fixed PR #204 point 3-hour layout changed")

    original_fractions = torch.tanh(original_control[flow_offset:flow_offset + 5])
    expected = original_control.new_tensor(EXPECTED_FRACTIONS)
    if not torch.allclose(original_fractions, expected, rtol=0, atol=2e-15):
        raise ValueError("original flow fractions differ from the declared candidate")
    control = _shift_control(original_control, field_size)
    changed = torch.nonzero(control != original_control, as_tuple=False).flatten().tolist()
    shifted_fractions = torch.tanh(control[flow_offset:flow_offset + 5])
    if changed != [flow_index] or not torch.allclose(
        shifted_fractions, control.new_tensor(SHIFTED_FRACTIONS), rtol=0, atol=2e-15
    ):
        raise ValueError("shifted candidate changed more than the declared flow fraction")

    identity = {
        "archived_input": base_identity,
        "control_sha256": _tensor_sha(control),
        "parameters_sha256": _tensor_sha(parameters),
        "terminal_truth_sha256": _tensor_sha(truth),
        "changed_control_index": flow_index,
        "changed_control_indices": changed,
        "original_flow_fractions": [float(value) for value in original_fractions],
        "shifted_flow_fractions": [float(value) for value in shifted_fractions],
    }
    return problem, original_control, control, parameters, truth, identity


def _shift_control(original_control: Tensor, field_size: int) -> Tensor:
    """Change only the predeclared second flow fraction at control[21]."""
    if field_size != 20 or original_control.shape != (26,):
        raise ValueError("shifted candidate requires the fixed 20+6 control layout")
    flow_index = field_size + FLOW_FRACTION_INDEX
    original_fractions = torch.tanh(original_control[field_size:field_size + 5])
    if not torch.allclose(
        original_fractions, original_control.new_tensor(EXPECTED_FRACTIONS),
        rtol=0, atol=2e-15,
    ):
        raise ValueError("original flow fractions differ from the declared candidate")
    control = original_control.clone()
    control[flow_index] = math.atanh(SHIFTED_FRACTIONS[FLOW_FRACTION_INDEX])
    changed = torch.nonzero(control != original_control, as_tuple=False).flatten().tolist()
    shifted_fractions = torch.tanh(control[field_size:field_size + 5])
    if changed != [flow_index] or not torch.allclose(
        shifted_fractions, control.new_tensor(SHIFTED_FRACTIONS), rtol=0, atol=2e-15
    ):
        raise ValueError("shifted candidate changed more than the declared flow fraction")
    return control


def _static_face_summary(control: Tensor, problem: Any) -> dict[str, Any]:
    """Apply production coefficient and face-flux operators before any rollout."""
    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("fixed point three-hour profile has no FV transport")
    field_size = problem.frozen.active_field_index.numel()
    dynamic = control[field_size:]
    coefficients = transport.bounded_fv_coefficients(
        dynamic[:spec.coefficient_limits.numel()],
        psi_basis=spec.psi_basis,
        coefficient_limits=spec.coefficient_limits,
        dt_seconds=(problem.frozen.nowcast_config.interval_minutes * 60.0
                    / spec.substeps_per_interval),
        spacing_yx=spec.spacing_yx,
        reconstruction=spec.reconstruction,
        max_courant=spec.max_courant,
    )
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    qx, qy = transport.face_volume_fluxes(psi)
    flux = torch.cat((qx.reshape(-1), qy.reshape(-1)))
    maximum = flux.abs().max()
    threshold = 128 * torch.finfo(control.dtype).eps * maximum
    near_zero = flux.abs() <= threshold
    where = torch.nonzero(near_zero, as_tuple=False)
    first = None
    if where.numel():
        flat = int(where[0, 0])
        if flat < qx.numel():
            first = {"orientation": "x", "index": list(torch.unravel_index(
                torch.tensor(flat), qx.shape))}
        else:
            first = {"orientation": "y", "index": list(torch.unravel_index(
                torch.tensor(flat - qx.numel()), qy.shape))}
        first = {key: (value if key == "orientation" else [int(v) for v in value])
                 for key, value in first.items()}
    return {
        "status": "passed" if bool(torch.isfinite(flux).all()) and not bool(near_zero.any()) else "refused_static_face",
        "coefficient_fractions": [float(value) for value in
                                  coefficients / spec.coefficient_limits],
        "face_count": int(flux.numel()),
        "near_zero_face_count": int(torch.count_nonzero(near_zero)),
        "maximum_abs_flux": float(maximum),
        "minimum_abs_flux": float(flux.abs().min()),
        "strict_threshold": float(threshold),
        "first_near_zero_face": first,
        "face_flux_sha256": _tensor_sha(flux),
    }


def _branch_record(problem: Any, control: Tensor, parameters: Tensor) -> dict[str, Any]:
    try:
        signature, scope = problem.branch_check(control, parameters)
    except ValueError as error:
        return {"status": "refused_full_oracle", "reason": str(error)}
    compact_signature = {
        "choices": signature.get("choices"),
        "face_signs": signature.get("face_signs"),
    }
    return {
        "status": "passed_strict_oracle",
        "euler_stages": signature.get("euler_stages"),
        "choice_stage_count": len(signature.get("choices", [])),
        "face_sign_stage_count": len(signature.get("face_signs", [])),
        "signature_sha256": _canonical_sha(compact_signature),
        "scope": scope,
    }


def _input_identity(problem: Any, original: Tensor, control: Tensor,
                    parameters: Tensor, truth: Tensor) -> dict[str, Any]:
    return {
        "archived_input": first_branch._input_identity(
            problem, original, parameters, truth
        ),
        "control_sha256": _tensor_sha(control),
        "parameters_sha256": _tensor_sha(parameters),
        "terminal_truth_sha256": _tensor_sha(truth),
        "changed_control_index": int(problem.frozen.active_field_index.numel())
        + FLOW_FRACTION_INDEX,
        "changed_control_indices": torch.nonzero(
            control != original, as_tuple=False
        ).flatten().tolist(),
        "original_flow_fractions": [float(value) for value in torch.tanh(
            original[problem.frozen.active_field_index.numel():
                     problem.frozen.active_field_index.numel() + 5]
        )],
        "shifted_flow_fractions": [float(value) for value in torch.tanh(
            control[problem.frozen.active_field_index.numel():
                    problem.frozen.active_field_index.numel() + 5]
        )],
    }


def run_probe(output: Path) -> dict[str, Any]:
    """Verify provenance, then run the one predeclared candidate diagnostic."""
    started = time.monotonic()
    source_before = _source_hashes()
    if not first_branch._sources_match_archive(source_before):
        raise ValueError("fixed PR #204 source identity changed")
    if _sha(PLAN) != PLAN_SHA256:
        raise ValueError("shifted-branch execution plan changed")
    archive_identity = first_branch._archive_input_identity()
    problem, original, control, parameters, truth, input_before = _prepare_shifted_case()
    if input_before["archived_input"] != archive_identity:
        raise ValueError("shifted candidate is not bound to the archived input")

    report: dict[str, Any] = {
        "pid": os.getpid(), "execution_status": "running",
        "execution_phase": "running", "diagnostic_status": "not_performed",
        "stationarity_passed": "not_tested", "response_computed": False,
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "forecast_score_computed": False, "plan_sha256": PLAN_SHA256,
        "archive_input_sha256": ARCHIVED_INPUT_SHA256,
        "source_before": source_before, "input_before": input_before,
        "layout": problem.layout,
        "scope": "single alternate latent flow fraction; fixed PR #204 point 3-hour inputs; branch feasibility only",
        "candidate": {
            "changed_control_indices": input_before["changed_control_indices"],
            "changed_control_index": input_before["changed_control_index"],
            "original_fraction": EXPECTED_FRACTIONS[FLOW_FRACTION_INDEX],
            "shifted_fraction": SHIFTED_FRACTIONS[FLOW_FRACTION_INDEX],
            "original_control_sha256": _tensor_sha(original),
            "shifted_control_sha256": _tensor_sha(control),
            "parameters_sha256": _tensor_sha(parameters),
            "terminal_truth_sha256": _tensor_sha(truth),
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
    }

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        static = _static_face_summary(control, problem)
        report["static_face_gate"] = static
        if static["status"] != "passed":
            report["diagnostic_status"] = "refused_static_face"
        else:
            diagnostic = first_branch.observe_first_failure(
                lambda: problem.forecast(control, parameters)
            )
            report["trajectory_diagnostic"] = diagnostic
            if diagnostic["status"] == "refused_first_branch":
                report["diagnostic_status"] = "refused_first_branch"
            elif diagnostic["euler_stages"] != EXPECTED_STAGES:
                raise ValueError("pointwise pass did not observe all 3,600 stages")
            else:
                report["full_strict_oracle"] = _branch_record(problem, control, parameters)
                report["diagnostic_status"] = report["full_strict_oracle"]["status"]
        report["execution_status"] = "completed"
        report["execution_phase"] = "finished"
    except Exception as error:
        report["execution_status"] = "failed"
        report["execution_phase"] = "failed"
        report["diagnostic_status"] = "not_performed"
        report["diagnostic_error"] = f"{type(error).__name__}: {error}"
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _source_hashes()
        report["input_after"] = _input_identity(
            problem, original, control, parameters, truth
        )
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256
        try:
            report["archive_unchanged"] = (
                first_branch._archive_input_identity() == archive_identity
            )
        except (OSError, ValueError, json.JSONDecodeError):
            report["archive_unchanged"] = False
        if (not report["source_unchanged"] or not report["input_unchanged"]
                or not report["plan_unchanged"] or not report["archive_unchanged"]):
            report.update(execution_status="failed", execution_phase="identity_refused",
                          diagnostic_status="not_performed")
        save()
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


def _valid_static_gate(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    status = value.get("status")
    near_zero = value.get("near_zero_face_count")
    threshold = value.get("strict_threshold")
    minimum = value.get("minimum_abs_flux")
    maximum = value.get("maximum_abs_flux")
    return (
        status in ("passed", "refused_static_face")
        and type(near_zero) is int and near_zero >= 0
        and type(value.get("face_count")) is int and value["face_count"] > 0
        and isinstance(threshold, (int, float)) and math.isfinite(threshold) and threshold >= 0
        and isinstance(minimum, (int, float)) and math.isfinite(minimum) and minimum >= 0
        and isinstance(maximum, (int, float)) and math.isfinite(maximum) and maximum >= 0
        and isinstance(value.get("face_flux_sha256"), str)
        and isinstance(value.get("coefficient_fractions"), list)
        and len(value["coefficient_fractions"]) == len(SHIFTED_FRACTIONS)
        and all(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=2e-15)
                for actual, expected in zip(value["coefficient_fractions"], SHIFTED_FRACTIONS))
        and (status != "passed" or maximum > 0 and near_zero == 0 and minimum > threshold)
        and (status != "refused_static_face" or near_zero > 0)
    )


def _valid_candidate_identity(value: object, expected_archive: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    original = value.get("original_flow_fractions")
    shifted = value.get("shifted_flow_fractions")
    hashes = (value.get("control_sha256"), value.get("parameters_sha256"),
              value.get("terminal_truth_sha256"))
    return (
        value.get("archived_input") == expected_archive
        and value.get("changed_control_index") == 21
        and value.get("changed_control_indices") == [21]
        and isinstance(original, list) and len(original) == len(EXPECTED_FRACTIONS)
        and all(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=2e-15)
                for actual, expected in zip(original, EXPECTED_FRACTIONS))
        and isinstance(shifted, list) and len(shifted) == len(SHIFTED_FRACTIONS)
        and all(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=2e-15)
                for actual, expected in zip(shifted, SHIFTED_FRACTIONS))
        and all(isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in hashes)
    )


def _valid_candidate_report(candidate: object, input_identity: object) -> bool:
    if not isinstance(candidate, dict) or not isinstance(input_identity, dict):
        return False
    archived = input_identity.get("archived_input")
    original_fractions = input_identity.get("original_flow_fractions")
    shifted_fractions = input_identity.get("shifted_flow_fractions")
    if (not isinstance(archived, dict)
            or not isinstance(original_fractions, list)
            or not isinstance(shifted_fractions, list)
            or len(original_fractions) != len(EXPECTED_FRACTIONS)
            or len(shifted_fractions) != len(SHIFTED_FRACTIONS)):
        return False
    return (
        candidate.get("original_control_sha256") == archived.get("control_sha256")
        and candidate.get("shifted_control_sha256") == input_identity.get("control_sha256")
        and candidate.get("parameters_sha256") == input_identity.get("parameters_sha256")
        and candidate.get("terminal_truth_sha256") == input_identity.get("terminal_truth_sha256")
        and candidate.get("changed_control_index") == input_identity.get("changed_control_index")
        and candidate.get("changed_control_indices") == input_identity.get("changed_control_indices")
        and isinstance(candidate.get("original_fraction"), (int, float))
        and math.isfinite(candidate["original_fraction"])
        and math.isclose(candidate["original_fraction"], original_fractions[FLOW_FRACTION_INDEX],
                         rel_tol=0, abs_tol=2e-15)
        and isinstance(candidate.get("shifted_fraction"), (int, float))
        and math.isfinite(candidate["shifted_fraction"])
        and math.isclose(candidate["shifted_fraction"], shifted_fractions[FLOW_FRACTION_INDEX],
                         rel_tol=0, abs_tol=2e-15)
    )


def _valid_child(child: object, expected_sources: dict[str, str],
                 expected_archive: dict[str, Any]) -> bool:
    if not isinstance(child, dict):
        return False
    status = child.get("diagnostic_status")
    common_valid = (
        child.get("execution_status") == "completed"
        and child.get("execution_phase") == "finished"
        and child.get("stationarity_passed") == "not_tested"
        and child.get("response_computed") is False
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
        and child.get("forecast_score_computed") is False
        and type(child.get("pid")) is int and child["pid"] > 0
        and child.get("source_before") == child.get("source_after") == expected_sources
        and child.get("source_unchanged") is True
        and child.get("input_before") == child.get("input_after")
        and _valid_candidate_identity(child.get("input_before"), expected_archive)
        and _valid_candidate_report(child.get("candidate"), child.get("input_before"))
        and child.get("input_unchanged") is True
        and child.get("plan_sha256") == PLAN_SHA256
        and child.get("plan_unchanged") is True
        and child.get("archive_input_sha256") == ARCHIVED_INPUT_SHA256
        and child.get("archive_unchanged") is True
        and first_branch._valid_layout(child.get("layout"))
        and _valid_static_gate(child.get("static_face_gate"))
    )
    if not common_valid:
        return False

    static = child["static_face_gate"]
    trajectory = child.get("trajectory_diagnostic")
    full = child.get("full_strict_oracle")
    if status == "refused_static_face":
        return (static["status"] == "refused_static_face"
                and trajectory is None and full is None)
    if static["status"] != "passed":
        return False
    if status == "refused_first_branch":
        if not isinstance(trajectory, dict) or full is not None:
            return False
        stage_index = trajectory.get("stage_index")
        failure = trajectory.get("first_failure")
        return (
            trajectory.get("status") == "refused_first_branch"
            and type(stage_index) is int and 0 <= stage_index < EXPECTED_STAGES
            and trajectory.get("stages_observed") == stage_index + 1
            and trajectory.get("accepted_stages_before_failure") == stage_index
            and trajectory.get("stage_context") == first_branch._stage_context(stage_index)
            and isinstance(failure, dict)
            and failure.get("predicate") in {
                "field_scale_finite_positive", "x_left_slope_nonzero",
                "x_right_slope_nonzero", "x_active_limiter_operand_difference",
                "y_left_slope_nonzero", "y_right_slope_nonzero",
                "y_active_limiter_operand_difference", "x_face_flux_nonzero",
                "y_face_flux_nonzero",
            }
        )
    if status not in ("refused_full_oracle", "passed_strict_oracle"):
        return False
    if (not isinstance(trajectory, dict)
            or trajectory.get("status") != "passed_pointwise"
            or trajectory.get("euler_stages") != EXPECTED_STAGES
            or trajectory.get("stages_observed") != EXPECTED_STAGES
            or trajectory.get("first_failure") is not None
            or not isinstance(full, dict)
            or full.get("status") != status):
        return False
    if status == "refused_full_oracle":
        return isinstance(full.get("reason"), str) and bool(full["reason"])
    return (
        full.get("euler_stages") == EXPECTED_STAGES
        and full.get("choice_stage_count") == EXPECTED_STAGES
        and full.get("face_sign_stage_count") == EXPECTED_STAGES
        and isinstance(full.get("signature_sha256"), str)
        and len(full["signature_sha256"]) == 64
    )


def run(directory: Path) -> dict[str, Any]:
    """Run one guarded child and preserve execution/diagnostic distinctions."""
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("shifted-branch output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_3h_shifted_branch.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output)]
    source_before = _source_hashes()
    if not first_branch._sources_match_archive(source_before):
        raise ValueError("fixed PR #204 source identity changed")
    if _sha(PLAN) != PLAN_SHA256:
        raise ValueError("shifted-branch execution plan changed")
    archive_identity = first_branch._archive_input_identity()
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "point_3h_shifted_branch.resource.json",
        log_path=directory / "point_3h_shifted_branch.log",
    )
    child: object = None
    read_error = None
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    current_sources = _source_hashes()
    try:
        archive_unchanged = first_branch._archive_input_identity() == archive_identity
    except (OSError, ValueError, json.JSONDecodeError):
        archive_unchanged = False
    valid = (
        _valid_resource(resource, command)
        and _valid_child(child, source_before, archive_identity)
        and isinstance(child, dict)
        and child.get("pid") == resource.get("child_pid")
        and current_sources == source_before
        and _sha(PLAN) == PLAN_SHA256
        and archive_unchanged
    )
    result: dict[str, Any] = {
        "execution_status": "completed" if valid else "failed",
        "diagnostic_status": child.get("diagnostic_status", "not_performed")
        if isinstance(child, dict) else "not_performed",
        "first_failure": child.get("trajectory_diagnostic", {}).get("first_failure")
        if isinstance(child, dict) and isinstance(child.get("trajectory_diagnostic"), dict)
        else None,
        "static_face_gate": child.get("static_face_gate")
        if isinstance(child, dict) else None,
        "full_strict_oracle": child.get("full_strict_oracle")
        if isinstance(child, dict) else None,
        "stationarity_passed": "not_tested", "response_computed": False,
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "child_read_error": read_error, "resource": resource,
        "source_sha256": source_before,
        "input_identity": child.get("input_before") if isinstance(child, dict) else None,
        "plan_sha256": PLAN_SHA256,
        "archive_unchanged": archive_unchanged,
        "scope": "single alternate-control feasibility diagnostic; no analysis or sensitivity solve",
    }
    (directory / "point_3h_shifted_branch.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = run_probe(args.output)
        exit_code = 0 if result["execution_status"] == "completed" else 2
    elif args.directory is not None:
        result = run(args.directory)
        exit_code = 0 if result["execution_status"] == "completed" else 1
    else:
        parser.error("provide --output for child mode or --directory for guarded mode")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(exit_code)
