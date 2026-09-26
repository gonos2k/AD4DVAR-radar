"""Locate two archived face-flux events and inspect finite off-event sides."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, TypedDict

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport
from examples.weather_scenarios import fv_partial_sector_root_probe as prior
from tests.test_fv_research_partial_observation import _problem


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_PARTIAL_FACE_EVENT_PLAN.md"
PLAN_SHA256 = "bf6a5634062233167d68a927b7c079da069dcbc7f245daf9d08b28153ebe4d50"
ARCHIVE = EVIDENCE / "partial_sector_root_attempt1"
ARCHIVE_MANIFEST_SHA256 = "d80362f8acbb0feb1c82d60d5e9cde61a4863564d7f8f64e49467ba39d12cddd"
RAW_FILES = (
    "partial_sector_root.json", "partial_sector_root.run.json",
    "partial_sector_root.resource.json", "partial_sector_root.log",
)
FACE_INDEX = (2, 0)
MAX_BISECTIONS = 64
EVENT_PARAMETER_TOLERANCE = 2.0**-48
SIDE_OFFSETS = (2.0**-8, 2.0**-10)
STATUS_ONLY_SOURCES = frozenset({
    "examples/weather_scenarios/fv_partial_sector_root_probe.py",
    "examples/weather_scenarios/fv_partial_sector_root_runner.py",
})
SOURCE_PATHS = tuple(dict.fromkeys((
    *prior.SOURCE_PATHS,
    "examples/weather_scenarios/fv_partial_face_event_probe.py",
    "examples/weather_scenarios/fv_partial_face_event_runner.py",
)))


class EventBracket(TypedDict):
    left_t: float
    right_t: float
    left_value: float
    right_value: float
    iterations: int
    stop_reason: str


def bisect_zero(
    value_at: Callable[[float], float], *, left: float = 0.0,
    right: float = 1.0, max_iterations: int = MAX_BISECTIONS,
    parameter_tolerance: float = EVENT_PARAMETER_TOLERANCE,
) -> EventBracket:
    """Bracket a scalar sign event without evaluating derivatives."""
    if (not math.isfinite(left) or not math.isfinite(right) or left >= right
            or max_iterations < 1 or not math.isfinite(parameter_tolerance)
            or parameter_tolerance <= 0.0):
        raise ValueError("event bracket and stopping limits must be finite and ordered")
    f_left, f_right = float(value_at(left)), float(value_at(right))
    if not math.isfinite(f_left) or not math.isfinite(f_right):
        raise ValueError("event endpoint values must be finite")
    if f_left == 0.0 or f_right == 0.0:
        raise ValueError("endpoint event is unsupported; endpoints must be off-event")
    if math.copysign(1.0, f_left) == math.copysign(1.0, f_right):
        raise ValueError("event endpoints must have opposite signs")

    iterations = 0
    while right - left > parameter_tolerance and iterations < max_iterations:
        width = right - left
        midpoint = left + width * 0.5
        if midpoint == left or midpoint == right:
            break
        f_midpoint = float(value_at(midpoint))
        if not math.isfinite(f_midpoint):
            raise ValueError("event midpoint value must be finite")
        if f_midpoint == 0.0:
            return {
                "left_t": midpoint, "right_t": midpoint,
                "left_value": 0.0, "right_value": 0.0,
                "iterations": iterations + 1, "stop_reason": "exact_zero",
            }
        iterations += 1
        if math.copysign(1.0, f_midpoint) == math.copysign(1.0, f_left):
            left, f_left = midpoint, f_midpoint
        else:
            right, f_right = midpoint, f_midpoint

    if right - left <= parameter_tolerance:
        stop_reason = "bracket_width"
    elif left + (right - left) * 0.5 in (left, right):
        stop_reason = "float64_resolution"
    else:
        raise ValueError("event bisection iteration limit exhausted")
    return {
        "left_t": left, "right_t": right,
        "left_value": f_left, "right_value": f_right,
        "iterations": iterations, "stop_reason": stop_reason,
    }


def event_side_parameters(event_t: float) -> list[dict[str, Any]]:
    """Return only predeclared finite offsets that lie inside the chord."""
    return [
        {key: point[key] for key in ("side", "offset", "t")}
        for point in event_side_schedule(event_t) if point["available"]
    ]


def event_side_schedule(
    event_t: float,
) -> list[dict[str, Any]]:
    """Keep every predeclared offset, including unavailable chord points."""
    if not math.isfinite(event_t) or not 0.0 < event_t < 1.0:
        raise ValueError("event parameter must be a finite interior event")
    points = []
    for side, sign in (("left", -1.0), ("right", 1.0)):
        for offset in SIDE_OFFSETS:
            t = event_t + sign * offset
            available = 0.0 < t < 1.0
            points.append({
                "side": side, "offset": offset,
                "t": t if available else None, "available": available,
            })
    return points


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _check_archive() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = ARCHIVE / "manifest.json"
    if _sha(manifest_path) != ARCHIVE_MANIFEST_SHA256:
        raise ValueError("archived PR209 manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for name in RAW_FILES:
        path = ARCHIVE / name
        expected = manifest["file_sha256"][str(path.relative_to(ROOT))]
        if _sha(path) != expected:
            raise ValueError(f"archived PR209 raw file changed: {name}")

    child = json.loads((ARCHIVE / RAW_FILES[0]).read_text())
    parent = json.loads((ARCHIVE / RAW_FILES[1]).read_text())
    if (child.get("numerical_status") != "root_refused"
            or parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != "root_refused"
            or child.get("response_validation") != "not_performed"
            or child.get("source_before") != child.get("source_after")
            or child.get("source_before") != manifest["source_before_after_sha256"]):
        raise ValueError("archived PR209 refusal identity changed")
    return manifest, child


def _accepted_chords(child: dict[str, Any]) -> list[dict[str, Any]]:
    seed = child["gauss_newton"]
    start_values = seed["control"]
    start_sha = seed["control_sha256"]
    start_branch = child["seed_branch"]
    accepted = [row for row in child["trial_records"] if row.get("accepted") is True]
    if len(accepted) < 2:
        raise ValueError("archive lacks the first two accepted PR209 trials")
    chords = []
    for expected_iteration, row in enumerate(accepted[:2], start=1):
        if row.get("iteration") != expected_iteration:
            raise ValueError("first accepted PR209 trials are not iterations one and two")
        end_values = row.get("candidate_control")
        end_sha = row.get("candidate_control_sha256")
        if not isinstance(end_values, list) or not isinstance(end_sha, str):
            raise ValueError("accepted PR209 endpoint is incomplete")
        start = torch.tensor(start_values, dtype=torch.float64)
        end = torch.tensor(end_values, dtype=torch.float64)
        if (start.shape != (26,) or end.shape != (26,)
                or not bool(torch.isfinite(start).all())
                or not bool(torch.isfinite(end).all())
                or _tensor_sha(start) != start_sha or _tensor_sha(end) != end_sha):
            raise ValueError("accepted PR209 endpoint control hash or shape mismatch")
        chords.append({
            "label": f"accepted_iteration_{expected_iteration}",
            "start": start, "end": end,
            "start_sha256": start_sha, "end_sha256": end_sha,
            "start_signature_sha256": start_branch["signature_sha256"],
            "end_signature_sha256": row["branch"]["signature_sha256"],
        })
        start_values, start_sha, start_branch = (
            end_values, end_sha, row["branch"]
        )
    return chords


def _qy_20(control: Tensor, spec: Any, interval_minutes: float) -> float:
    limits = spec.coefficient_limits
    coefficients = transport.bounded_fv_coefficients(
        control[-(limits.numel() + 1):-1],
        psi_basis=spec.psi_basis, coefficient_limits=limits,
        dt_seconds=interval_minutes * 60.0 / spec.substeps_per_interval,
        spacing_yx=spec.spacing_yx, reconstruction=spec.reconstruction,
        max_courant=spec.max_courant,
    )
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    _, qy = transport.face_volume_fluxes(psi)
    value = float(qy[FACE_INDEX])
    if not math.isfinite(value):
        raise ValueError("production q_y[2,0] face flux is nonfinite")
    return value


def _branch_with_face_margin(
    problem: Any, control: Tensor, parameters: Tensor,
) -> tuple[dict[str, Any], list[float], str]:
    face_margins: list[float] = []

    def record(_echo: Tensor, qx: Tensor, qy: Tensor) -> None:
        flux = torch.cat((qx.flatten(), qy.flatten())).abs()
        maximum = flux.max()
        if (not bool(torch.isfinite(flux).all())
                or not bool(torch.isfinite(maximum)) or bool(maximum <= 0)):
            face_margins.append(math.nan)
            return
        face_margins.append(float(flux.min() / maximum))

    with transport.observe_minmod_stages(record):
        branch, scope = problem.branch_check(control, parameters)
    if len(face_margins) != branch["euler_stages"]:
        raise RuntimeError("branch trace and face-margin trace have different lengths")
    if not all(math.isfinite(value) for value in face_margins):
        raise ValueError("off-event face-margin trace contains nonfinite values")
    return branch, face_margins, scope


def _sample_side(
    problem: Any, parameters: Tensor, start: Tensor, direction: Tensor,
    point: dict[str, Any], chord_label: str,
) -> dict[str, Any]:
    t = float(point["t"])
    control = start + t * direction
    result: dict[str, Any] = {
        "chord": chord_label, "side": point["side"],
        "offset": point["offset"], "t": t,
        "control_sha256": _tensor_sha(control),
        "finite": None,
    }
    try:
        branch, face_margins, scope = _branch_with_face_margin(
            problem, control, parameters
        )
    except ValueError as error:
        if str(error) not in prior._KNOWN_BRANCH_REFUSALS:
            raise RuntimeError("unexpected branch callback failure") from error
        result.update(status="branch_refused", refusal=f"{type(error).__name__}: {error}")
        return result

    signature = {"choices": branch["choices"], "face_signs": branch["face_signs"]}
    signature_sha = hashlib.sha256(json.dumps(
        signature, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    result.update(
        status="branch_admitted", branch_scope=scope,
        euler_stages=branch["euler_stages"],
        minimum_scaled_slope_margin=branch["minimum_scaled_slope_margin"],
        minimum_scaled_face_flux_margin=min(face_margins),
        face_margins_by_stage=face_margins,
        branch_signature=signature,
        branch_signature_sha256=signature_sha,
        response_margin_qualified=(
            branch["minimum_scaled_slope_margin"] > 1.0e-4
            and min(face_margins) > 1.0e-4
        ),
    )
    if branch["euler_stages"] != 54:
        result.update(status="unsupported_stage_count",
                      refusal="expected 54 complete branch stages")
        return result

    gradient_function = torch.func.grad(problem.objective, argnums=0)
    objective = problem.objective(control, parameters)
    gradient = gradient_function(control, parameters)
    if (not bool(torch.isfinite(objective))
            or not bool(torch.isfinite(gradient).all())):
        result.update(status="unsupported_nonfinite_objective_or_gradient", finite=False)
        return result
    jvp_result = torch.func.jvp(
        lambda value: gradient_function(value, parameters),
        (control,), (direction,),
    )
    hessian_direction = jvp_result[1]
    if not bool(torch.isfinite(hessian_direction).all()):
        result.update(status="unsupported_nonfinite_jvp", finite=False)
        return result

    merit = 0.5 * torch.dot(gradient, gradient)
    gradient_dot_direction = torch.dot(gradient, direction)
    merit_directional_derivative = torch.dot(gradient, hessian_direction)
    scalars = (merit, gradient_dot_direction, merit_directional_derivative)
    if not all(bool(torch.isfinite(value)) for value in scalars):
        result.update(status="unsupported_nonfinite_directional_quantities", finite=False)
        return result
    result.update(
        status="finite_off_event_derivatives",
        objective=float(objective),
        gradient_inf=float(gradient.abs().max()),
        gradient_l2=float(torch.linalg.vector_norm(gradient)),
        gradient_merit=float(merit),
        gradient_dot_chord=float(gradient_dot_direction),
        hessian_times_chord=hessian_direction.tolist(),
        merit_directional_derivative=float(merit_directional_derivative),
        finite=True,
    )
    return result


def run(
    output: Path, *, expected_plan_sha256: str,
    expected_probe_sha256: str,
) -> dict[str, Any]:
    archive_root = ARCHIVE.resolve(strict=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if (output.resolve(strict=False).is_relative_to(archive_root)
            or temporary.resolve(strict=False).is_relative_to(archive_root)
            or output.exists() or output.is_symlink()
            or temporary.exists() or temporary.is_symlink()):
        raise ValueError("face-event output must be fresh and outside the raw archive")
    if (_sha(PLAN) != PLAN_SHA256 or expected_plan_sha256 != PLAN_SHA256
            or _sha(Path(__file__)) != expected_probe_sha256):
        raise ValueError("reviewed face-event plan or probe source changed before run")

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest, child = _check_archive()
    source_before = _sources()
    for name, digest in child["source_before"].items():
        if name not in STATUS_ONLY_SOURCES and _sha(ROOT / name) != digest:
            raise ValueError(f"archived PR209 source changed: {name}")
    input_before = prior._preflight_identity()
    if child["input_before"] != child["input_after"] or child["input_before"] != input_before:
        raise ValueError("archived PR209 fixed-input identity changed")

    problem, _, parameters = _problem()
    if problem.identity != input_before["current_problem_identity"]:
        raise ValueError("current fixed FV problem identity changed")
    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("fixed research problem lacks FV transport")
    interval_minutes = problem.frozen.nowcast_config.interval_minutes
    chords = _accepted_chords(child)
    chord_records = []
    side_records = []
    side_comparisons = []
    for chord in chords:
        start, end = chord["start"], chord["end"]
        direction = end - start
        flux_at = lambda t: _qy_20(start + t * direction, spec, interval_minutes)
        start_flux, end_flux = flux_at(0.0), flux_at(1.0)
        if start_flux * end_flux >= 0.0:
            raise ValueError(f"{chord['label']} archived endpoints do not cross q_y[2,0]")
        bracket = bisect_zero(flux_at)
        event_t = 0.5 * (float(bracket["left_t"]) + float(bracket["right_t"]))
        chord_records.append({
            "label": chord["label"],
            "start_control_sha256": chord["start_sha256"],
            "end_control_sha256": chord["end_sha256"],
            "start_signature_sha256": chord["start_signature_sha256"],
            "end_signature_sha256": chord["end_signature_sha256"],
            "face": {"kind": "qy", "index": list(FACE_INDEX)},
            "start_face_flux": start_flux, "end_face_flux": end_flux,
            "event_bracket": bracket, "event_parameter": event_t,
            "event_face_flux": flux_at(event_t),
            "event_derivative_evaluated": False,
        })
        for point in event_side_schedule(event_t):
            if not point["available"]:
                side_records.append({
                    "chord": chord["label"], "side": point["side"],
                    "offset": point["offset"], "t": None,
                    "status": "unavailable_outside_segment", "finite": None,
                })
            else:
                side_records.append(_sample_side(
                    problem, parameters, start, direction, point, chord["label"]
                ))
        for side in ("left", "right"):
            pair = [sample for sample in side_records
                    if sample["chord"] == chord["label"] and sample["side"] == side]
            if len(pair) != 2:
                side_comparisons.append({
                    "chord": chord["label"], "side": side,
                    "status": "incomplete_offsets",
                    "selected_face_attribution": "undetermined",
                })
            elif any("branch_signature_sha256" not in sample for sample in pair):
                side_comparisons.append({
                    "chord": chord["label"], "side": side,
                    "status": "unsupported_sample",
                    "selected_face_attribution": "undetermined",
                })
            else:
                signatures_match = (
                    pair[0]["branch_signature_sha256"]
                    == pair[1]["branch_signature_sha256"]
                )
                side_comparisons.append({
                    "chord": chord["label"], "side": side,
                    "status": "same_signature" if signatures_match else "signature_changed",
                    "signature_sha256_by_offset": {
                        str(sample["offset"]): sample["branch_signature_sha256"]
                        for sample in pair
                    },
                    "selected_face_attribution": (
                        "not_confounded_by_signature_change"
                        if signatures_match else "confounded_by_other_branch_changes"
                    ),
                })

    input_after = prior._preflight_identity()
    source_after = _sources()
    if (_sha(PLAN) != PLAN_SHA256
            or _sha(Path(__file__)) != expected_probe_sha256
            or _sha(ARCHIVE / "manifest.json") != ARCHIVE_MANIFEST_SHA256
            or source_before != source_after or input_before != input_after
            or any(_sha(ARCHIVE / name) != manifest["file_sha256"][
                str((ARCHIVE / name).relative_to(ROOT))] for name in RAW_FILES)):
        raise ValueError("source, input, plan, or raw PR209 archive changed during probe")

    report = {
        "status": "face_event_diagnostic_only",
        "scope": "two archived PR209 accepted chords; q_y[2,0] event and finite off-event sides",
        "archive_manifest_sha256": ARCHIVE_MANIFEST_SHA256,
        "plan_sha256": PLAN_SHA256,
        "reviewed_probe_sha256": expected_probe_sha256,
        "source_before": source_before, "source_after": source_after,
        "input_before": input_before, "input_after": input_after,
        "chord_count": len(chord_records), "chords": chord_records,
        "off_event_sample_count": len(side_records), "off_event_samples": side_records,
        "same_side_offset_comparisons": side_comparisons,
        "max_bisections": MAX_BISECTIONS,
        "event_parameter_tolerance": EVENT_PARAMETER_TOLERANCE,
        "side_offsets": list(SIDE_OFFSETS),
        "event_derivatives_evaluated": False,
        "new_gn_root_adjoint_reanalysis_runs": 0,
        "root_claim": False, "response_claim": False,
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "elapsed_seconds": time.monotonic() - started,
        "pid": os.getpid(),
    }
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    args = parser.parse_args()
    result = run(
        args.output,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_probe_sha256=args.expected_probe_sha256,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "chord_count", "off_event_sample_count", "elapsed_seconds",
    )}, indent=2, sort_keys=True))
