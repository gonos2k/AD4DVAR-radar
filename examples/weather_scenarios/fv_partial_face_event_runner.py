"""Run the bounded PR209 face-event diagnostic under the shared RSS guard."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, TypeGuard

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_partial_face_event_probe as probe
from examples.weather_scenarios.fv86_resource_runner import run_guarded


PLAN_SHA256 = probe.PLAN_SHA256
WALL_SECONDS = 300
SAMPLED_RSS_BYTES = 1024**3
RUNNER_PATH = "examples/weather_scenarios/fv_partial_face_event_runner.py"


def _identity(
    *, expected_plan_sha256: str, expected_probe_sha256: str,
    expected_runner_sha256: str,
) -> dict[str, Any]:
    if (expected_plan_sha256 != PLAN_SHA256
            or probe._sha(probe.PLAN) != PLAN_SHA256
            or probe._sha(Path(probe.__file__)) != expected_probe_sha256
            or probe._sha(Path(__file__)) != expected_runner_sha256):
        raise ValueError("reviewed source or plan changed")
    sources = probe._sources()
    if sources.get(RUNNER_PATH) != expected_runner_sha256:
        raise ValueError("runner is absent from the probe's source identity")
    manifest, _ = probe._check_archive()
    raw_archive = {
        name: probe._sha(probe.ARCHIVE / name)
        for name in probe.RAW_FILES
    }
    if any(raw_archive[name] != manifest["file_sha256"][
            str((probe.ARCHIVE / name).relative_to(ROOT))]
           for name in probe.RAW_FILES):
        raise ValueError("raw PR209 archive identity changed")
    return {
        "plan_sha256": PLAN_SHA256,
        "probe_sha256": expected_probe_sha256,
        "runner_sha256": expected_runner_sha256,
        "archive_manifest_sha256": probe.ARCHIVE_MANIFEST_SHA256,
        "raw_archive_sha256": raw_archive,
        "sources": sources,
    }


def _valid_resource(resource: Any, command: list[str]) -> bool:
    return (
        isinstance(resource, dict)
        and resource.get("command") == command
        and resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("child_pid")) is int
        and resource["child_pid"] > 0
        and type(resource.get("rss_samples")) is int
        and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
        and type(resource.get("elapsed_seconds")) in (int, float)
        and math.isfinite(resource["elapsed_seconds"])
        and 0 <= resource["elapsed_seconds"] <= WALL_SECONDS
    )


def _finite_number(value: Any) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(value)


def _sha256_text(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _expected_chords() -> list[dict[str, str]]:
    _, archived = probe._check_archive()
    seed = archived["gauss_newton"]
    accepted = [row for row in archived["trial_records"] if row.get("accepted") is True]
    if len(accepted) < 2:
        return []
    expected = []
    start_sha = seed["control_sha256"]
    start_signature = archived["seed_branch"]["signature_sha256"]
    for iteration, row in enumerate(accepted[:2], start=1):
        if row.get("iteration") != iteration:
            return []
        expected.append({
            "label": f"accepted_iteration_{iteration}",
            "start_control_sha256": start_sha,
            "end_control_sha256": row["candidate_control_sha256"],
            "start_signature_sha256": start_signature,
            "end_signature_sha256": row["branch"]["signature_sha256"],
        })
        start_sha = row["candidate_control_sha256"]
        start_signature = row["branch"]["signature_sha256"]
    return expected


def _archived_chords() -> list[dict[str, Any]]:
    _, archived = probe._check_archive()
    return probe._accepted_chords(archived)


def _current_input_identity() -> dict[str, Any]:
    return probe.prior._preflight_identity()


def _current_problem() -> Any:
    problem, _, _ = probe._problem()
    return problem


def _valid_chord(chord: Any, expected: dict[str, str]) -> bool:
    if not isinstance(chord, dict):
        return False
    bracket = chord.get("event_bracket")
    if not isinstance(bracket, dict):
        return False
    left_t, right_t = bracket.get("left_t"), bracket.get("right_t")
    left_value, right_value = bracket.get("left_value"), bracket.get("right_value")
    event_t = chord.get("event_parameter")
    start_flux_value = chord.get("start_face_flux")
    end_flux_value = chord.get("end_face_flux")
    if (not _finite_number(left_t) or not _finite_number(right_t)
            or not _finite_number(left_value) or not _finite_number(right_value)
            or not _finite_number(event_t) or not _finite_number(start_flux_value)
            or not _finite_number(end_flux_value)
            or not _finite_number(chord.get("event_face_flux"))):
        return False
    left_t, right_t = float(left_t), float(right_t)
    left_value, right_value = float(left_value), float(right_value)
    event_t = float(event_t)
    start_flux = float(start_flux_value)
    end_flux = float(end_flux_value)
    if not (0.0 < left_t <= right_t < 1.0 and 0.0 < event_t < 1.0):
        return False
    if not math.isclose(event_t, (left_t + right_t) * 0.5, rel_tol=0.0, abs_tol=1e-15):
        return False
    if start_flux * end_flux >= 0.0:
        return False
    if left_value * right_value > 0.0:
        return False
    if chord.get("event_derivative_evaluated") is not False:
        return False
    stop_reason = bracket.get("stop_reason")
    iterations = bracket.get("iterations")
    if type(iterations) is not int or not 1 <= iterations <= probe.MAX_BISECTIONS:
        return False
    if stop_reason == "exact_zero":
        if not (left_t == right_t and left_value == right_value == 0.0):
            return False
    elif stop_reason == "bracket_width":
        if (right_t - left_t > probe.EVENT_PARAMETER_TOLERANCE
                or left_value * right_value >= 0.0):
            return False
    elif stop_reason == "float64_resolution":
        if (left_t + (right_t - left_t) * 0.5 not in (left_t, right_t)
                or left_value * right_value >= 0.0):
            return False
    else:
        return False
    return all(chord.get(key) == value for key, value in expected.items()) and (
        chord.get("face") == {"kind": "qy", "index": [2, 0]}
    )


def _valid_signature(sample: dict[str, Any]) -> bool:
    signature = sample.get("branch_signature")
    if not isinstance(signature, dict):
        return False
    choices, face_signs = signature.get("choices"), signature.get("face_signs")
    if (not isinstance(choices, list) or len(choices) != 54
            or not isinstance(face_signs, list) or len(face_signs) != 54):
        return False

    def matrix(value: Any, rows: int, columns: int, *, kind: str) -> bool:
        if not isinstance(value, list) or len(value) != rows:
            return False
        for row in value:
            if not isinstance(row, list) or len(row) != columns:
                return False
            for entry in row:
                if kind == "bool":
                    if type(entry) is not bool:
                        return False
                elif type(entry) is not int or entry not in {-1, 0, 1}:
                    return False
                elif kind == "nonzero_sign" and entry == 0:
                    return False
        return True

    for stage in choices:
        if not isinstance(stage, list) or len(stage) != 2:
            return False
        for axis in stage:
            if not isinstance(axis, dict):
                return False
            if (not matrix(axis.get("choose_left"), 2, 3, kind="bool")
                    or not matrix(axis.get("slope_sign"), 2, 3, kind="sign")
                    or not matrix(axis.get("left_sign"), 2, 3, kind="sign")
                    or not matrix(axis.get("right_sign"), 2, 3, kind="sign")):
                return False
    for stage in face_signs:
        if (not isinstance(stage, dict)
                or not matrix(stage.get("qx"), 4, 6, kind="nonzero_sign")
                or not matrix(stage.get("qy"), 5, 5, kind="nonzero_sign")):
            return False
    try:
        digest = hashlib.sha256(json.dumps(
            signature, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
    except (TypeError, ValueError):
        return False
    return sample.get("branch_signature_sha256") == digest


def _valid_sample(sample: Any, event_by_chord: dict[str, float]) -> bool:
    if not isinstance(sample, dict):
        return False
    label, side, offset = sample.get("chord"), sample.get("side"), sample.get("offset")
    if not isinstance(label, str) or label not in event_by_chord or side not in ("left", "right"):
        return False
    if not _finite_number(offset) or offset not in probe.SIDE_OFFSETS:
        return False
    sign = -1.0 if side == "left" else 1.0
    expected_t = event_by_chord[label] + sign * offset
    available = 0.0 < expected_t < 1.0
    status = sample.get("status")
    if not available:
        return (status == "unavailable_outside_segment" and sample.get("t") is None
                and sample.get("finite") is None)
    if not _finite_number(sample.get("t")) or float(sample["t"]) != expected_t:
        return False
    if not _sha256_text(sample.get("control_sha256")):
        return False
    if status == "branch_refused":
        refusal = sample.get("refusal")
        known_refusals = {
            f"ValueError: {reason}" for reason in probe.prior._KNOWN_BRANCH_REFUSALS
        }
        return sample.get("finite") is None and refusal in known_refusals
    if status not in {
        "finite_off_event_derivatives",
        "unsupported_nonfinite_objective_or_gradient",
        "unsupported_nonfinite_jvp",
        "unsupported_nonfinite_directional_quantities",
    }:
        return False
    slope_margin = sample.get("minimum_scaled_slope_margin")
    face_margin = sample.get("minimum_scaled_face_flux_margin")
    if not _finite_number(slope_margin) or not _finite_number(face_margin):
        return False
    slope_margin, face_margin = float(slope_margin), float(face_margin)
    stage_margins = sample.get("face_margins_by_stage")
    if (sample.get("euler_stages") != 54
            or slope_margin <= 0.0
            or face_margin <= 0.0
            or type(sample.get("response_margin_qualified")) is not bool
            or sample["response_margin_qualified"] != (
                slope_margin > 1.0e-4 and face_margin > 1.0e-4
            )
            or not isinstance(stage_margins, list) or len(stage_margins) != 54
            or not _valid_signature(sample)):
        return False
    if (not all(_finite_number(value) and value > 0.0 for value in stage_margins)
            or not math.isclose(min(float(value) for value in stage_margins), face_margin,
                                rel_tol=1e-12, abs_tol=1e-15)):
        return False
    if status == "finite_off_event_derivatives":
        objective = sample.get("objective")
        gradient_inf = sample.get("gradient_inf")
        gradient_l2 = sample.get("gradient_l2")
        merit = sample.get("gradient_merit")
        gradient_dot_chord = sample.get("gradient_dot_chord")
        merit_derivative = sample.get("merit_directional_derivative")
        hvp = sample.get("hessian_times_chord")
        if (not _finite_number(objective) or not _finite_number(gradient_inf)
                or not _finite_number(gradient_l2) or not _finite_number(merit)
                or not _finite_number(gradient_dot_chord)
                or not _finite_number(merit_derivative)):
            return False
        gradient_inf = float(gradient_inf)
        gradient_l2 = float(gradient_l2)
        merit = float(merit)
        return (
            sample.get("finite") is True
            and gradient_inf >= 0.0 and gradient_l2 >= gradient_inf
            and merit >= 0.0
            and math.isclose(merit, 0.5 * gradient_l2**2,
                             rel_tol=1e-10, abs_tol=1e-14)
            and isinstance(hvp, list) and len(hvp) == 26
            and all(_finite_number(value) for value in hvp)
        )
    return sample.get("finite") is False


def _valid_child(
    child: Any, *, identity: dict[str, Any], expected_probe_sha256: str,
    child_pid: Any, current_input_identity: dict[str, Any], problem: Any,
) -> bool:
    if not isinstance(child, dict):
        return False
    chords = child.get("chords")
    if not isinstance(chords, list) or len(chords) != 2:
        return False
    expected = _expected_chords()
    if len(expected) != 2 or not all(_valid_chord(row, pin)
                                     for row, pin in zip(chords, expected)):
        return False
    archived_chords = _archived_chords()
    if len(archived_chords) != 2:
        return False
    if (child.get("input_before") != current_input_identity
            or child.get("input_after") != current_input_identity
            or problem.identity != current_input_identity.get("current_problem_identity")):
        return False
    spec = problem.frozen.fv_transport
    if spec is None:
        return False
    interval = problem.frozen.nowcast_config.interval_minutes
    control_by_label: dict[str, tuple[Any, Any]] = {}
    for row, archive in zip(chords, archived_chords):
        label = row["label"]
        start, end = archive["start"], archive["end"]
        direction = end - start
        control_by_label[label] = (start, direction)
        bracket = row["event_bracket"]
        event_t = row["event_parameter"]
        for t, field in (
            (0.0, "start_face_flux"), (1.0, "end_face_flux"),
            (bracket["left_t"], "left_value"),
            (bracket["right_t"], "right_value"),
            (event_t, "event_face_flux"),
        ):
            actual = probe._qy_20(start + float(t) * direction, spec, interval)
            recorded = (row[field] if field in row else bracket[field])
            if actual != recorded:
                return False
    event_by_chord: dict[str, float] = {}
    for row in chords:
        label, event_t = row.get("label"), row.get("event_parameter")
        if not isinstance(label, str) or not _finite_number(event_t):
            return False
        event_by_chord[label] = float(event_t)
    samples = child.get("off_event_samples")
    if (not isinstance(samples, list) or len(samples) != 8
            or not all(_valid_sample(sample, event_by_chord) for sample in samples)):
        return False
    keys = [(sample["chord"], sample["side"], sample["offset"]) for sample in samples]
    expected_keys = [
        (label, side, offset)
        for label in event_by_chord
        for side in ("left", "right") for offset in probe.SIDE_OFFSETS
    ]
    if len(set(keys)) != 8 or set(keys) != set(expected_keys):
        return False
    for sample in samples:
        if sample["status"] == "unavailable_outside_segment":
            continue
        start, direction = control_by_label[sample["chord"]]
        control = start + float(sample["t"]) * direction
        if probe._tensor_sha(control) != sample["control_sha256"]:
            return False
    comparisons = child.get("same_side_offset_comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 4:
        return False
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            return False
        label, side = comparison.get("chord"), comparison.get("side")
        if label not in event_by_chord or side not in ("left", "right"):
            return False
        pair = [sample for sample in samples
                if sample["chord"] == label and sample["side"] == side]
        hashes = [sample.get("branch_signature_sha256") for sample in pair]
        if all(isinstance(value, str) for value in hashes):
            same = hashes[0] == hashes[1]
            status = "same_signature" if same else "signature_changed"
            attribution = ("not_confounded_by_signature_change" if same
                          else "confounded_by_other_branch_changes")
            mapping = {str(sample["offset"]): sample["branch_signature_sha256"]
                       for sample in pair}
            if (comparison.get("status") != status
                    or comparison.get("selected_face_attribution") != attribution
                    or comparison.get("signature_sha256_by_offset") != mapping):
                return False
        elif (comparison.get("status") != "unsupported_sample"
              or comparison.get("selected_face_attribution") != "undetermined"):
            return False
    comparison_keys = [(row.get("chord"), row.get("side")) for row in comparisons]
    if len(set(comparison_keys)) != 4 or set(comparison_keys) != {
        (label, side) for label in event_by_chord for side in ("left", "right")
    }:
        return False
    return (
        child.get("status") == "face_event_diagnostic_only"
        and child.get("pid") == child_pid
        and child.get("plan_sha256") == identity["plan_sha256"]
        and child.get("reviewed_probe_sha256") == expected_probe_sha256
        and child.get("archive_manifest_sha256") == identity["archive_manifest_sha256"]
        and child.get("source_before") == child.get("source_after") == identity["sources"]
        and isinstance(child.get("input_before"), dict)
        and child.get("input_before") == child.get("input_after") == current_input_identity
        and child.get("chord_count") == 2
        and child.get("off_event_sample_count") == 8
        and child.get("event_derivatives_evaluated") is False
        and child.get("new_gn_root_adjoint_reanalysis_runs") == 0
        and child.get("root_claim") is False
        and child.get("response_claim") is False
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
    )


def _read_child(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(value, dict):
        return None, "child result is not a JSON object"
    return value, None


def run(
    directory: Path, *, expected_plan_sha256: str,
    expected_probe_sha256: str, expected_runner_sha256: str,
) -> dict[str, Any]:
    resolved = directory.resolve(strict=False)
    archive = probe.ARCHIVE.resolve(strict=True)
    if resolved.is_relative_to(archive) or directory.is_symlink():
        raise ValueError("face-event output must be outside frozen raw archive")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("face-event output directory must be empty")

    identity_before = _identity(
        expected_plan_sha256=expected_plan_sha256,
        expected_probe_sha256=expected_probe_sha256,
        expected_runner_sha256=expected_runner_sha256,
    )
    input_identity_before = _current_input_identity()
    problem = _current_problem()
    if problem.identity != input_identity_before.get("current_problem_identity"):
        raise ValueError("fresh current input identity differs from fixed FV problem")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "partial_face_event.json"
    resource_path = directory / "partial_face_event.resource.json"
    log_path = directory / "partial_face_event.log"
    run_path = directory / "partial_face_event.run.json"
    command = [
        sys.executable, str(Path(probe.__file__)),
        "--output", str(output),
        "--expected-plan-sha256", expected_plan_sha256,
        "--expected-probe-sha256", expected_probe_sha256,
    ]

    resource: dict[str, Any] | None = None
    guard_error = None
    try:
        resource = run_guarded(
            command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
            report_path=resource_path, log_path=log_path,
        )
    except Exception as error:
        guard_error = f"{type(error).__name__}: {error}"
    child, child_read_error = _read_child(output)

    input_identity_after = None
    input_identity_error = None
    try:
        input_identity_after = _current_input_identity()
    except (OSError, ValueError, RuntimeError) as error:
        input_identity_error = f"{type(error).__name__}: {error}"
    inputs_unchanged = input_identity_after == input_identity_before

    identity_after = None
    identity_error = None
    try:
        identity_after = _identity(
            expected_plan_sha256=expected_plan_sha256,
            expected_probe_sha256=expected_probe_sha256,
            expected_runner_sha256=expected_runner_sha256,
        )
    except (OSError, ValueError) as error:
        identity_error = f"{type(error).__name__}: {error}"
    identities_unchanged = identity_after == identity_before

    resource_termination = resource.get("resource_termination") if resource else None
    child_output_exists = output.exists()
    if not identities_unchanged or not inputs_unchanged:
        execution_status = "identity_changed"
    elif resource_termination is not None:
        execution_status = "resource_limited"
    elif guard_error is not None or resource is None:
        execution_status = "guard_failed"
    elif resource.get("exit_code") != 0:
        execution_status = "child_nonzero_exit" if child_output_exists else "child_exception"
    elif not _valid_resource(resource, command):
        execution_status = "resource_report_invalid"
    elif child is None:
        execution_status = "child_result_missing_or_invalid"
    elif not _valid_child(
        child, identity=identity_before,
        expected_probe_sha256=expected_probe_sha256,
        child_pid=resource.get("child_pid"),
        current_input_identity=input_identity_before,
        problem=problem,
    ):
        execution_status = "child_result_invalid"
    else:
        execution_status = "completed"

    result = {
        "execution_status": execution_status,
        "numerical_status": child.get("status") if child else "not_reached",
        "resource": resource,
        "guard_error": guard_error,
        "child_read_error": child_read_error,
        "child_output_exists": child_output_exists,
        "identity_before": identity_before,
        "identity_after": identity_after,
        "identity_error": identity_error,
        "identities_unchanged": identities_unchanged,
        "input_identity_before": input_identity_before,
        "input_identity_after": input_identity_after,
        "input_identity_error": input_identity_error,
        "inputs_unchanged": inputs_unchanged,
        "plan_sha256": PLAN_SHA256,
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "scope": "guarded PR209 two-chord face-event diagnostic only",
    }
    run_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    args = parser.parse_args()
    report = run(
        args.directory,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_probe_sha256=args.expected_probe_sha256,
        expected_runner_sha256=args.expected_runner_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["execution_status"] == "completed" else 1)
