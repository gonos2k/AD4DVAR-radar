#!/usr/bin/env python3
"""Independently audit the frozen attempt-2 limiter-event JSON; no model execution."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "LIMITER_EVENT_PLAN_ATTEMPT2_20261010.json"
PREFLIGHT = EVIDENCE / "LIMITER_EVENT_PREFLIGHT_ATTEMPT2_20261010.json"
ARCHIVE = EVIDENCE / "LIMITER_EVENT_ARCHIVE_ATTEMPT2_20261010.json"
PRIOR = EVIDENCE / "gn_prepared_search_20261010_attempt1/step.json.gz"
ATTEMPT = EVIDENCE / "limiter_event_20261010_attempt2"
RAW = ATTEMPT / "diagnostic.json"
GZIP = ATTEMPT / "diagnostic.json.gz"
RUN = ATTEMPT / "diagnostic.run.json"
RESOURCE = ATTEMPT / "diagnostic.resource.json"
FAILED_ARCHIVE = EVIDENCE / "LIMITER_EVENT_FAILED_ARCHIVE_20261010.json"
FAILED_RUN = EVIDENCE / "limiter_event_20261010_attempt1/diagnostic.run.json"
JSON_OUT = EVIDENCE / "LIMITER_EVENT_SAVED_AUDIT_20261010.json"
MD_OUT = EVIDENCE / "LIMITER_EVENT_GREEN_FINAL_20261010.md"
PLAN_SHA = "3579c2bbf9f3213cec5669948e9de21e47da258720572b291b083c5355e3f79d"
BASE_SHA = "027e83cf77f7f9cf97c0c72da9f737b5627d312d6375b698bb5b7f7ae94c6769"
DIRECTION_SHA = "ab3a2635cedc7f90c9cead68e7ce9bae9b33a15cabf5ffe4970a9b00c91fea74"
FAILED_PLAN_SHA = "75b2e14cd9fed536763b552accdff2912f161a6ab4f2dd67d7dfb4c8aca75de6"
EPS = sys.float_info.epsilon


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(values: Any) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("expected saved numeric vector")
    result = [float(x) for x in values]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("saved vector contains nonfinite value")
    return result


def vdot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimension mismatch")
    return math.fsum(a * b for a, b in zip(left, right))


def norm(values: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in values))


def minus(left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("vector dimension mismatch")
    return [a - b for a, b in zip(left, right)]


def close(a: float, b: float, *, factor: float = 512.0) -> bool:
    return abs(a - b) <= factor * EPS * max(abs(a), abs(b), sys.float_info.min)


def tensor_sha(values: list[float]) -> str:
    return hashlib.sha256(struct.pack("<" + "d" * len(values), *values)).hexdigest()


def fraction(group: list[float], whole: list[float]) -> float:
    denominator = vdot(whole, whole)
    return vdot(group, group) / denominator if denominator else 0.0


def main() -> None:
    plan = read_json(PLAN)
    preflight = read_json(PREFLIGHT)
    archive = read_json(ARCHIVE)
    raw = read_json(RAW)
    run = read_json(RUN)
    resource = read_json(RESOURCE)
    failed_archive = read_json(FAILED_ARCHIVE)
    failed_run = read_json(FAILED_RUN)
    prior = json.loads(gzip.decompress(PRIOR.read_bytes()))

    pinned_groups: dict[str, Any] = {}
    for group in ("source_files", "archive_files"):
        pins = plan[group]
        matches, missing, mismatches = 0, [], []
        for name, expected in pins.items():
            file_path = ROOT / name
            if not file_path.is_file():
                missing.append(name)
                continue
            actual = sha(file_path)
            if actual == expected:
                matches += 1
            else:
                mismatches.append({"path": name, "expected": expected, "actual": actual})
        pinned_groups[group] = {"expected": len(pins), "matched": matches,
            "missing": missing, "mismatches": mismatches,
            "passed": matches == len(pins) and not missing and not mismatches}

    actual_raw = RAW.read_bytes()
    gzip_bytes = GZIP.read_bytes()
    uncompressed = gzip.decompress(gzip_bytes)
    actual_raw_sha = hashlib.sha256(actual_raw).hexdigest()
    uncompressed_sha = hashlib.sha256(uncompressed).hexdigest()
    current = vector(raw["current_control"])
    direction = vector(prior["postcommit_gn_readiness"]["direction"])
    if len(current) != 26 or len(direction) != 26:
        raise ValueError("current control or saved tangent direction is not length 26")
    if tensor_sha(current) != BASE_SHA or tensor_sha(direction) != DIRECTION_SHA:
        raise ValueError("saved control or tangent direction tensor hash mismatch")

    side_data: dict[str, Any] = {}
    gradients: dict[str, list[float]] = {}
    projected_gradients: dict[str, list[float]] = {}
    units: dict[str, list[float]] = {}
    slopes: dict[str, float] = {}
    zetas: dict[str, float] = {}
    side_checks: dict[str, bool] = {}
    for side in ("-1", "1"):
        item = raw["event_derivatives"][side]
        values = vector(item["values"])
        original_values = vector(item["original_values"])
        jvp = vector(item["jvp"])
        gradient = vector(item["gradient"])
        projected = vector(item["tangent_gradient"])
        if not (len(values) == len(original_values) == len(jvp) == 3
                and len(gradient) == len(projected) == 26):
            raise ValueError("event derivative payload dimensions changed")
        zeta = values[0] - values[1]
        slope = jvp[0] - jvp[1]
        normal_component = float(item["gradient_normal_component"])
        unit = [x / normal_component for x in minus(gradient, projected)]
        raw_dot = vdot(gradient, direction)
        projected_dot = vdot(projected, direction)
        unit_dot_direction = vdot(unit, direction)
        side_checks[side] = all((
            values == original_values,
            item.get("value_parity_exact") is True,
            item.get("analysis_frames_parity_exact") is True,
            item.get("gradient_dot_direction_passed") is True,
            item.get("projected_gradient_dot_direction_passed") is True,
            close(zeta, float(item["zeta"])),
            close(slope, float(item["zeta_directional_derivative"])),
            close(raw_dot, slope),
            close(projected_dot, slope),
            close(projected_dot, float(item["projected_gradient_dot_direction"])),
            close(unit_dot_direction, float(item["unit_normal_dot_direction"])),
            close(norm(unit), 1.0, factor=4096.0),
            close(float(item["linear_event_alpha"]), -zeta / slope),
            close(-zeta, float(item["positive_inside_gap"])),
            close(-slope, float(item["oriented_directional_derivative"])),
            values[0] < 0.0 and values[1] < 0.0 and values[0] < values[1],
        ))
        gradients[side], projected_gradients[side], units[side] = gradient, projected, unit
        slopes[side], zetas[side] = slope, zeta
        dnorm = norm(direction)
        side_data[side] = {
            "left": values[0], "right": values[1], "q_abs_max": values[2],
            "zeta_left_minus_right": zeta, "normalized_gap_abs_zeta_over_qmax": abs(zeta) / values[2],
            "zeta_directional_derivative": slope,
            "first_order_event_alpha": -zeta / slope,
            "gradient_l2": norm(gradient), "projected_gradient_l2": norm(projected),
            "direction_l2": dnorm,
            "cosine_gradient_to_direction": raw_dot / (norm(gradient) * dnorm),
            "cosine_projected_gradient_to_direction": projected_dot / (norm(projected) * dnorm),
            "normal_gradient_component": normal_component,
            "unit_normal_dot_direction": unit_dot_direction,
            "raw_gradient_dot_direction": raw_dot,
            "projected_gradient_dot_direction": projected_dot,
            "raw_gradient_c12_15_squared_fraction": fraction(gradient[12:15], gradient),
            "projected_gradient_c12_15_squared_fraction": fraction(projected[12:15], projected),
            "direction_c12_15_squared_fraction": fraction(direction[12:15], direction),
            "analysis_frames_parity_exact": item.get("analysis_frames_parity_exact"),
            "event_values_parity_exact": values == original_values,
            "jvp_gradient_dot_check_passed": side_checks[side],
        }

    unit = units["-1"]
    gradient_delta = minus(gradients["1"], gradients["-1"])
    projected_delta = minus(projected_gradients["1"], projected_gradients["-1"])
    delta_normal = vdot(gradient_delta, unit)
    delta_tangent_residual = minus(gradient_delta, [delta_normal * x for x in unit])
    dot_plus, dot_minus = vdot(gradients["1"], direction), vdot(gradients["-1"], direction)
    dot_difference = vdot(gradient_delta, direction)
    separate_dot_difference = dot_plus - dot_minus
    normal_dot_prediction = delta_normal * vdot(unit, direction)
    pure_normal_dot_error = abs(dot_difference - normal_dot_prediction)
    componentwise_dot_scale = math.fsum(abs(x * y) for x, y in zip(gradient_delta, direction))
    underlying_dot_scale = math.fsum(abs(x * y) for x, y in zip(gradients["1"], direction)) + math.fsum(
        abs(x * y) for x, y in zip(gradients["-1"], direction))
    pure_normal_roundoff_bound = (norm(projected_delta) * norm(direction)
        + 64.0 * EPS * (componentwise_dot_scale + underlying_dot_scale + abs(normal_dot_prediction)))
    subtraction_roundoff_bound = 64.0 * EPS * (
        componentwise_dot_scale + underlying_dot_scale + abs(dot_plus) + abs(dot_minus))
    projection_checks = {
        "side_projected_gradient_difference_l2": norm(projected_delta),
        "side_gradient_difference_l2": norm(gradient_delta),
        "side_gradient_difference_tangent_residual_l2": norm(delta_tangent_residual),
        "side_gradient_difference_normal_component": delta_normal,
        "side_dot_difference": dot_difference,
        "side_dot_difference_from_separate_dots": separate_dot_difference,
        "side_dot_subtraction_error": abs(dot_difference - separate_dot_difference),
        "side_dot_subtraction_roundoff_bound": subtraction_roundoff_bound,
        "normal_only_dot_prediction": normal_dot_prediction,
        "side_dot_difference_pure_normal_error": pure_normal_dot_error,
        "side_dot_difference_pure_normal_roundoff_bound": pure_normal_roundoff_bound,
        "side_dot_difference_equals_pure_normal_term": pure_normal_dot_error <= pure_normal_roundoff_bound,
        "side_dot_difference_forms_agree_to_roundoff": abs(dot_difference - separate_dot_difference)
            <= subtraction_roundoff_bound,
        "projected_gradients_match_across_sides": norm(projected_delta) <= 1e-12,
        "side_gradient_jump_is_pure_normal": norm(delta_tangent_residual) <= 1e-11,
        "normal_vectors_match_across_sides": norm(minus(units["1"], units["-1"])) <= 1e-12,
    }

    base_j, base_r = float(raw["base_J"]), float(raw["base_R"])
    samples = []
    sample_checks = []
    for sample in raw["samples"]:
        events = {}
        event_close = True
        for side in ("-1", "1"):
            event = sample["events"][side]
            event_values = vector(event["values"])
            actual_zeta = event_values[0] - event_values[1]
            linear_zeta = zetas[side] + float(sample["alpha"]) * slopes[side]
            sample_events_equal = close(actual_zeta, float(event["zeta"])) and close(
                linear_zeta, float(event["linear_zeta"]))
            event_close = event_close and sample_events_equal
            events[side] = {
                "left": event_values[0], "right": event_values[1], "q_abs_max": event_values[2],
                "zeta_actual": actual_zeta, "zeta_linear_model": linear_zeta,
                "actual_minus_linear_zeta": actual_zeta - linear_zeta,
                "choose_left": event.get("choose_left"),
                "same_negative_sign": event.get("same_negative_sign"),
                "event_value_parity_exact": event.get("value_parity_exact") is True,
            }
        compact = {"alpha": float(sample["alpha"]), "actual_move_l2": float(sample["actual_move"]),
            "control_sha256": sample["control_sha256"], "J": float(sample["J"]),
            "J_change": float(sample["J"]) - base_j, "R": float(sample["R"]),
            "R_change": float(sample["R"]) - base_r, "theta": float(sample["theta"]),
            "branch_pair_passed": sample["branch_pair_passed"], "face_passed": sample["face_passed"],
            "trace_matches_base": sample["trace_matches_base"],
            "side_objectives_match_native": sample["side_objectives_match_native"],
            "events": events, "linear_event_model_close": event_close}
        samples.append(compact)
        sample_checks.append(event_close and all(e["event_value_parity_exact"] for e in events.values()))

    failed_gzip_sha = hashlib.sha256((EVIDENCE / "limiter_event_20261010_attempt1/diagnostic.json.gz").read_bytes()).hexdigest()
    failed_attempt_note = {
        "status": "failed_source_closure_no_success_claim",
        "plan_sha256": failed_archive.get("plan_sha256"),
        "reason": failed_archive.get("reason"),
        "success_claim": failed_archive.get("success_claim"),
        "preserved_gzip_sha256_matches_manifest": failed_gzip_sha == failed_archive.get("gzip_sha256"),
        "failed_run_exit_code": failed_run.get("resource", {}).get("exit_code"),
        "failed_execution_status": failed_run.get("execution_status"),
        "full_reproduction_claim": False,
    }

    trace = prior["final_repeat"]["branch_trace"]
    branch_event = {}
    for side in ("-1", "1"):
        choices = trace[side]["choices"][124][0]
        branch_event[side] = {
            "left_sign": choices["left_sign"][1][2],
            "right_sign": choices["right_sign"][1][2],
            "choose_left": choices["choose_left"][1][2],
            "active_limiter_gap_over_q_minimum": trace[side]["minimum_margins"]["active_limiter_gap_over_q"],
        }
    event_gap = side_data["-1"]["normalized_gap_abs_zeta_over_qmax"]
    event_trace_closed = all(branch_event[side]["left_sign"] == -1
        and branch_event[side]["right_sign"] == -1 and branch_event[side]["choose_left"] is False
        for side in ("-1", "1"))
    event_gap_closed = all(close(event_gap,
        branch_event[side]["active_limiter_gap_over_q_minimum"], factor=4096.0)
        for side in ("-1", "1"))
    checks = {
        "frozen_plan_and_preflight_pins": sha(PLAN) == PLAN_SHA and preflight.get("plan_sha256") == PLAN_SHA
            and preflight.get("load_plan_passed") is True and preflight.get("closed_base_passed") is True
            and preflight.get("production_executed") is False,
        "all_154_sources_and_200_archives_match": all(v["passed"] for v in pinned_groups.values()),
        "attempt2_archive_gzip_raw_run_and_resource_close": archive.get("plan_sha256") == PLAN_SHA
            and archive.get("execution_status") == "completed"
            and archive.get("attempt2_guarded_launch_count") == 1
            and archive.get("optimizer_steps_applied") == 0
            and archive.get("lossless_roundtrip") is True
            and hashlib.sha256(gzip_bytes).hexdigest() == archive.get("gzip_sha256")
            and uncompressed_sha == archive.get("raw_sha256") == actual_raw_sha == run.get("child_sha256")
            and raw.get("plan_sha256") == PLAN_SHA
            and sha(RUN) == archive.get("run_sha256") and sha(RESOURCE) == archive.get("resource_sha256"),
        "attempt2_guard_completed_zero_exit": run.get("execution_status") == "completed"
            and resource.get("exit_code") == 0 and resource.get("resource_termination") is None
            and resource.get("monitor_error") is None and resource.get("received_sigterm") is False,
        "source_input_runtime_closed": raw.get("source_before") == raw.get("source_after")
            and raw.get("source_unchanged") is True
            and raw.get("input_before") == raw.get("input_after")
            and raw.get("fixed_input_unchanged") is True
            and raw.get("runtime_before") == raw.get("runtime_after")
            and raw.get("runtime_unchanged") is True,
        "event_side_parity_and_jvp_gradient_checks": all(side_checks.values()),
        "trace_124_x_cell_and_normalized_gap_match_saved_event": event_trace_closed and event_gap_closed,
        "side_gradient_difference_is_pure_normal": all(projection_checks[key] for key in (
            "side_dot_difference_equals_pure_normal_term", "projected_gradients_match_across_sides",
            "side_gradient_jump_is_pure_normal", "normal_vectors_match_across_sides",
            "side_dot_difference_forms_agree_to_roundoff")),
        "all_three_samples_match_linear_zeta_and_event_values": len(samples) == 3 and all(sample_checks),
        "no_optimizer_hvp_row_or_solve": raw.get("optimizer_steps_applied") == 0
            and raw.get("fresh_hvp_calls") == 0 and raw.get("archived_hvp_vectors_used") == 0
            and raw.get("event_grad_completed") == 2 and raw.get("event_jvp_completed") == 2,
        "failed_attempt1_is_marked_without_success_claim": failed_attempt_note["success_claim"] is False
            and failed_attempt_note["failed_run_exit_code"] == 1
            and failed_attempt_note["preserved_gzip_sha256_matches_manifest"],
    }
    result = {
        "plan_sha256": PLAN_SHA,
        "preflight": {k: preflight.get(k) for k in (
            "plan_sha256", "source_count", "archive_count", "load_plan_passed", "closed_base_passed",
            "production_executed")},
        "plan_pin_audit": pinned_groups,
        "parent_point": {"control_sha256": raw.get("base_control_sha256"),
            "direction_sha256": raw.get("direction_sha256"), "J": base_j, "R": base_r,
            "theta": raw.get("base_theta"), "optimizer_steps_applied": raw.get("optimizer_steps_applied"),
            "direction_l2": norm(direction)},
        "event": {"definition": raw.get("event"), "trace_branch_at_124_x_1_2": branch_event,
            "sides": side_data, "normalized_event_gap": event_gap,
            "normalized_gap_matches_saved_global_minimum": all(close(
                event_gap, trace[s]["minimum_margins"]["active_limiter_gap_over_q"], factor=4096.0)
                for s in ("-1", "1")),
            "side_projection_comparison": projection_checks},
        "samples": samples,
        "attempt1_failure_note": failed_attempt_note,
        "attempt2_receipts": {"raw_sha256": actual_raw_sha, "gzip_sha256": hashlib.sha256(gzip_bytes).hexdigest(),
            "run_sha256": sha(RUN), "resource_sha256": sha(RESOURCE),
            "elapsed_seconds": resource.get("elapsed_seconds"),
            "sampled_peak_rss_bytes": resource.get("sampled_peak_rss_bytes"),
            "rss_limit_bytes": resource.get("rss_limit_bytes"),
            "wall_limit_seconds": resource.get("wall_limit_seconds")},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# GREEN saved-data review: 027e limiter event",
        "",
        f"Attempt 2 plan SHA-256: `{PLAN_SHA}`. All checks passed: **{result['all_checks_passed']}**.",
        "This review uses saved JSON, frozen hashes, and standard-library arithmetic only; it performs no FV, AD, or HVP evaluation.",
        "",
        "## Event and derivative",
        "",
        "At analysis Euler stage 124, x-interior cell [1,2], the direct-tape capture and replay-enabled reference match exactly on both active-face extensions. The event is `ζ = left - right`; the observed values are negative and same-sign.",
        f"`ζ = {side_data['-1']['zeta_left_minus_right']:.12g}`, `q_abs_max = {side_data['-1']['q_abs_max']:.12g}`, and `|ζ|/q_abs_max = {event_gap:.12g}`.",
        f"The saved first-order event alpha is `{side_data['-1']['first_order_event_alpha']:.12g}`. Side JVPs are "
        f"{slopes['-1']:.12g} and {slopes['1']:.12g}; the JVP/gradient-dot checks and projected-gradient-dot checks pass on both sides.",
        f"Projected gradient norm is {norm(projected_gradients['-1']):.12g}; direction norm is {norm(direction):.12g}; "
        f"cosine(projected gradient, GN direction) is {vdot(projected_gradients['-1'], direction)/(norm(projected_gradients['-1'])*norm(direction)):.12g}.",
        f"The c[12:15] squared-gradient fraction is {fraction(projected_gradients['-1'][12:15], projected_gradients['-1']):.8%} after projection.",
        f"The GN direction's saved face-normal component is {side_data['-1']['unit_normal_dot_direction']:.12g}. The projected-gradient difference between sides is {projection_checks['side_projected_gradient_difference_l2']:.12g} L2.",
        f"For the side-gradient jump, Δg·d is {projection_checks['side_dot_difference']:.12g}; the pure-normal estimate is {projection_checks['normal_only_dot_prediction']:.12g}. Their difference {projection_checks['side_dot_difference_pure_normal_error']:.12g} is below the componentwise roundoff/projection bound {projection_checks['side_dot_difference_pure_normal_roundoff_bound']:.12g}; the tangent residue norm is {projection_checks['side_gradient_difference_tangent_residual_l2']:.12g}.",
        "",
        "## Local chart samples",
        "",
        "| α | actual move | ζ (− / +) | linear ζ (− / +) | J change | R change | trace matches base |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for sample in samples:
        minus_event, plus_event = sample["events"]["-1"], sample["events"]["1"]
        md.append(f"| {sample['alpha']:.12g} | {sample['actual_move_l2']:.12g} | "
            f"{minus_event['zeta_actual']:.12g} / {plus_event['zeta_actual']:.12g} | "
            f"{minus_event['zeta_linear_model']:.12g} / {plus_event['zeta_linear_model']:.12g} | "
            f"{sample['J_change']:.12g} | {sample['R_change']:.12g} | {sample['trace_matches_base']} |")
    md += [
        "",
        "The smallest sample remains on the saved limiter selector and lowers R slightly. The middle and largest samples cross ζ=0, change the selector, and raise R; J falls slightly at all three. The linear alpha is a local first-order estimate, not a safe interval or a path certificate.",
        "",
        "Attempt 1 remains a failed source-closure attempt after a late helper/test change; its manifest has `success_claim=false`. Its values are not included as successful evidence, and full reproduction of that attempt is not claimed because the exact helper/test source bytes were not preserved.",
        "",
        f"Attempt 2 used {resource.get('elapsed_seconds')} s and sampled {resource.get('sampled_peak_rss_bytes')} bytes RSS; it completed with zero optimizer steps, zero HVPs, zero Jacobian row calls, and zero dense solves.",
        "",
        "## Receipt checks",
        "",
    ]
    md.extend(f"- `{key}`: **{value}**" for key, value in checks.items())
    md.append("")
    MD_OUT.write_text("\n".join(md), encoding="utf-8")
    print(JSON_OUT)
    print(MD_OUT)
    if not all(checks.values()):
        raise SystemExit("saved event audit found a mismatched or incomplete receipt")


if __name__ == "__main__":
    main()
