"""Analyze the mixing-minimum attempt from saved JSON arrays only.

This script reads receipts and controls/gradients already written by the child.
It never imports the model package, computes an FV output, differentiates, or
runs an HVP. Run only after the root-owned attempt receipts are complete.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

EVIDENCE = Path(__file__).resolve().parent
ATTEMPT = EVIDENCE / "tangent_mixing_minimum_20261010_attempt1"
PLAN = EVIDENCE / "TANGENT_MIXING_MINIMUM_PLAN_20261010.json"
PRODUCER_ARCHIVE = EVIDENCE / "tangent_precondition_comparison_20261009_attempt1/step.json.gz"
PRODUCER_SHA = "c64480cfdc8b011b63b8bec4fd0e704187b28657d47e5e51b63f606406e0c4cf"
EXPECTED_BASE_SHA = "0da57dbc5a4c101c2f24204835a746514b110f4548fe64de40455f4478dfbed1"
FACE_SCALE = 0.84
EPS128 = 128.0 * 2.220446049250313e-16


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def vector_sha(values: list[float]) -> str:
    return sha_bytes(struct.pack(f"<{len(values)}d", *(float(x) for x in values)))


def dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("saved vectors have mismatched lengths")
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def norm(values: list[float]) -> float:
    return math.sqrt(math.fsum(float(x) * float(x) for x in values))


def mixed_gradient(side_gradients: dict[str, list[float]], theta: float) -> list[float]:
    minus = side_gradients["-1"]
    plus = side_gradients["1"]
    return [(1.0 - theta) * float(a) + theta * float(b) for a, b in zip(minus, plus)]


def gradient_summary(values: list[float]) -> dict[str, float]:
    return {"l2": norm(values), "linf": max((abs(float(x)) for x in values), default=0.0)}


def close(a: float, b: float, *, factor: float = 128.0) -> bool:
    scale = max(abs(a), abs(b), float.fromhex("0x1.0p-1022"))
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= factor * 2.220446049250313e-16 * scale


def load_child(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("child receipt must be a JSON object")
    return data, value


def prior_endpoint() -> tuple[bytes, dict[str, Any]]:
    data, raw = load_child(PRODUCER_ARCHIVE)
    if sha_bytes(data) != PRODUCER_SHA:
        raise ValueError("pinned PR270 producer raw digest changed")
    endpoint = raw.get("last_confirmed_closure")
    if not isinstance(endpoint, dict):
        raise ValueError("PR270 archive has no last confirmed endpoint")
    control = endpoint.get("control")
    if not isinstance(control, list) or vector_sha(control) != EXPECTED_BASE_SHA:
        raise ValueError("PR270 archive endpoint control does not match the mixing base")
    return data, raw


def static_qy(control: list[float], row: int, column: int) -> float:
    """Saved-control static flux from the original five-mode basis, for context."""
    amplitudes = (0.11, 0.08, 0.07, 0.04, 0.03)
    flow = [math.tanh(float(control[20 + index])) for index in range(5)]

    def psi(y: int, x: int) -> float:
        basis = (float(y), float(x), float(x * y),
            0.5 * float(x * x - y * y), float(x * x * y))
        return math.fsum(amplitudes[k] * flow[k] * basis[k] for k in range(5))

    return -(psi(row, column + 1) - psi(row, column))


def collect(child_path: Path, run_path: Path, resource_path: Path, plan_path: Path) -> dict[str, Any]:
    raw_bytes, child = load_child(child_path)
    parent = json.loads(run_path.read_text())
    resource = json.loads(resource_path.read_text())
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    prior_bytes, prior = prior_endpoint()
    prior_endpoint_record = prior["last_confirmed_closure"]
    base_control = [float(x) for x in prior_endpoint_record["control"]]
    base_sha = vector_sha(base_control)
    if base_sha != EXPECTED_BASE_SHA:
        raise ValueError("loaded PR270 base changed after digest validation")

    iterations = child.get("iterations", [])
    item = iterations[0] if isinstance(iterations, list) and iterations else {}
    arms = item.get("model_comparisons", []) if isinstance(item, dict) else []
    arm = next((value for value in arms if value.get("name") == "candidate_mixing_minimum"), None)
    trials = arm.get("trials", []) if isinstance(arm, dict) else []
    accepted_trials = [trial for trial in trials if trial.get("accepted") is True]
    selected_trial = accepted_trials[0] if len(accepted_trials) == 1 else None
    final_repeat = item.get("final_repeat") if isinstance(item, dict) else None
    if not isinstance(final_repeat, dict):
        final_repeat = None

    terminal_control = child.get("current_control")
    if not isinstance(terminal_control, list):
        terminal_control = child.get("last_confirmed_control", base_control)
    terminal_control = [float(x) for x in terminal_control]
    terminal_sha = vector_sha(terminal_control)
    movement = [value - base for value, base in zip(terminal_control, base_control)]
    movement_norm = norm(movement)
    movement_max = max((abs(value) for value in movement), default=0.0)
    accepted = child.get("accepted_iterations") == 1 and selected_trial is not None
    no_op_trials = [trial for trial in trials if trial.get("status") == "zero_control_movement_refused"]
    transaction_contract = {
        "accepted_commit_has_nonzero_control_movement": (not accepted or
            (movement_norm > 0.0 and terminal_sha != base_sha)),
        "unaccepted_run_preserves_carried_base": (accepted or
            (terminal_sha == base_sha and child.get("current_theta") == plan["initial_theta"]
                and movement_norm == 0.0)),
        "zero_movement_trials_refused": all(trial.get("accepted") is not True for trial in no_op_trials),
        "zero_movement_refusal_count": len(no_op_trials),
    }

    initial_f = child.get("initial_F", [])
    carried_f2 = float(child.get("initial_F_squared", float("nan")))
    if initial_f and math.isfinite(carried_f2):
        carried_f2_vector = dot([float(x) for x in initial_f], [float(x) for x in initial_f])
    else:
        carried_f2_vector = None
    working_base = child.get("base_mixing_minimum", {})
    working_f2 = working_base.get("F_squared") if isinstance(working_base, dict) else None
    working_theta = working_base.get("working_theta_star") if isinstance(working_base, dict) else None
    selected_f2 = final_repeat.get("F_squared") if accepted and final_repeat else None
    selected_psi = final_repeat.get("Psi") if accepted and final_repeat else None
    selected_theta = final_repeat.get("recomputed_mixing_theta_star") if accepted and final_repeat else None
    working_gradient: list[float] | None = None
    selected_gradient: list[float] | None = None
    selected_native_j = final_repeat.get("objective") if accepted and final_repeat else None
    p2_gradient_gate: dict[str, Any] = {"passed": False, "sides": {}}
    if working_theta is not None:
        working_gradient = mixed_gradient(prior_endpoint_record["side_gradients"], float(working_theta))
    if accepted and final_repeat and selected_theta is not None:
        selected_gradient = mixed_gradient(final_repeat["side_gradients"], float(selected_theta))
        trial_gradients = selected_trial.get("side_gradients", {})
        side_checks = {}
        for side in ("-1", "1"):
            actual = [float(x) for x in final_repeat["side_gradients"][side]]
            expected = [float(x) for x in trial_gradients[side]]
            error = max((abs(a - b) for a, b in zip(actual, expected)), default=0.0)
            scale = max(max((abs(x) for x in actual), default=0.0),
                max((abs(x) for x in expected), default=0.0), float.fromhex("0x1.0p-1022"))
            budget = 128.0 * 2.220446049250313e-16 * scale
            side_checks[side] = {"max_abs_error": error, "budget": budget,
                "matched": error <= budget and len(actual) == len(expected)}
        p2_gradient_gate = {"passed": all(value["matched"] for value in side_checks.values()),
            "sides": side_checks}

    derivative = None
    if isinstance(arm, dict) and item:
        direction = [float(x) for x in arm.get("direction", [])]
        hminus = [float(x) for x in arm.get("hminus_direction", [])]
        hplus = [float(x) for x in arm.get("hplus_direction", [])]
        residual = [float(x) for x in arm.get("residual", [])]
        residual_direction = [float(x) for x in arm.get("residual_direction", [])]
        gminus = [float(x) for x in prior_endpoint_record["side_gradients"]["-1"]]
        gplus = [float(x) for x in prior_endpoint_record["side_gradients"]["1"]]
        jump = [plus - minus for minus, plus in zip(gminus, gplus)]
        theta_star = float(working_theta)
        mixed = [gm + theta_star * j for gm, j in zip(gminus, jump)]
        hmix = [(1.0 - theta_star) * hm + theta_star * hp
            for hm, hp in zip(hminus, hplus)]
        delta_h = [hp - hm for hm, hp in zip(hminus, hplus)]
        jump_squared = dot(jump, jump)
        theta_numerator = dot(jump, hmix) + dot(mixed, delta_h)
        theta_prime_formula = -theta_numerator / jump_squared if jump_squared else None
        f_dot_df = dot(residual, residual_direction)
        q_base = float(child.get("face_qualification", {}).get("value", 0.0))
        ndotd = residual_direction[-1] * FACE_SCALE if residual_direction else 0.0
        envelope_slope = dot(mixed, hmix) + q_base * ndotd / (FACE_SCALE * FACE_SCALE)
        df_formula = [hm + j * float(arm.get("delta_theta", 0.0))
            for hm, j in zip(hmix, jump)] + [ndotd / FACE_SCALE]
        derivative = {
            "theta_star": theta_star,
            "theta_prime_saved": arm.get("delta_theta"),
            "theta_prime_recomputed": theta_prime_formula,
            "theta_prime_numerator": arm.get("delta_theta_numerator"),
            "theta_prime_denominator": arm.get("delta_theta_denominator"),
            "theta_prime_formula_error": (None if theta_prime_formula is None else
                float(arm.get("delta_theta", 0.0)) - theta_prime_formula),
            "F_dot_DF": f_dot_df,
            "envelope_slope_from_saved_arrays": envelope_slope,
            "saved_envelope_slope": arm.get("gates", {}).get("envelope_slope"),
            "residual_direction_max_formula_error": max(
                (abs(a - b) for a, b in zip(residual_direction, df_formula)), default=0.0),
            "residual_direction_norm_squared": dot(residual_direction, residual_direction),
            "direction_norm": norm(direction),
            "linear_theta_prediction_for_accepted_alpha": (
                None if selected_trial is None else selected_trial.get("linear_theta_prediction")),
            "actual_candidate_theta_star": (None if selected_trial is None else
                selected_trial.get("candidate_theta_star", selected_trial.get("theta"))),
        }
        df2 = derivative["residual_direction_norm_squared"]
        derivative["model_merit_alpha_cap"] = (
            -f_dot_df / df2 if isinstance(df2, (int, float)) and df2 > 0 and f_dot_df < 0 else None)
        derivative["F_dot_DF_matches_envelope"] = close(f_dot_df, envelope_slope)

    closure_keys = ("side_gradients_finite", "native_objective_matches_proposal",
        "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
        "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
        "fixed_input_unchanged", "runtime_unchanged", "deadline_passed",
        "mixing_minimum_valid", "minimum_theta_matches_proposal")
    closure_gates = ({key: final_repeat.get(key) is True for key in closure_keys}
        if final_repeat else {key: False for key in closure_keys})
    trial_summaries = []
    for trial in trials:
        control = trial.get("control", [])
        actual_theta = trial.get("candidate_theta_star", trial.get("theta"))
        linear_theta = trial.get("linear_theta_prediction")
        trial_summaries.append({
            "alpha": trial.get("alpha"),
            "linear_theta_prediction": linear_theta,
            "actual_candidate_theta_star": actual_theta,
            "theta_prediction_error": (float(actual_theta) - float(linear_theta)
                if isinstance(actual_theta, (int, float)) and isinstance(linear_theta, (int, float))
                else None),
            "objective_J": trial.get("objective"),
            "F_squared": trial.get("F_squared"),
            "Psi": trial.get("Psi"),
            "actual_path_norm": trial.get("actual_path_norm"),
            "control_sha256": vector_sha(control) if control else None,
            "face_value": trial.get("face_value"),
            "face_passed": trial.get("face_audit_passed"),
            "branch_pair_passed": trial.get("branch_pair_passed"),
            "J_armijo_passed": trial.get("J_armijo_passed"),
            "F_squared_armijo_passed": trial.get("F_squared_armijo_passed"),
            "mixing_minimum_valid": trial.get("mixing_minimum_valid"),
            "minimum_theta_refusal": trial.get("minimum_theta_refusal"),
            "accepted": trial.get("accepted"), "status": trial.get("status"),
        })

    history = child.get("hvp_history", [])
    side_counts = {str(side): sum(1 for row in history if row.get("side") == side) for side in (-1, 1)}
    expected_direction_sha = (vector_sha([float(x) for x in arm.get("direction", [])])
        if isinstance(arm, dict) and arm.get("direction") else None)
    hvp_provenance = {"count": len(history), "sides": side_counts,
        "all_completed": len(history) == child.get("hvp_calls_completed")
            and all(row.get("status") == "completed" for row in history),
        "base_control_sha_matches": all(row.get("base_control_sha256") == base_sha for row in history),
        "reference_theta_matches": all(row.get("theta") == item.get("theta") for row in history),
        "working_theta_matches": all(row.get("working_theta") == item.get("working_theta") for row in history),
        "direction_sha_matches": all(row.get("direction_sha256") == expected_direction_sha for row in history),
        "selected_face_operator": all(row.get("operator") == "selected_face_extension"
            and row.get("scope") == "one current chart tangent" for row in history)}
    source_before = child.get("source_before", {})
    source_after = child.get("source_after", {})
    resource_closed = (resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("received_sigterm") is False
        and resource.get("wall_limit_seconds") == plan["policy"]["outer_seconds"]
        and resource.get("rss_limit_bytes") == plan["policy"]["rss_bytes"]
        and resource.get("elapsed_seconds", math.inf) <= plan["policy"]["outer_seconds"]
        and resource.get("sampled_peak_rss_bytes", math.inf) < plan["policy"]["rss_bytes"])

    output = {
        "scope": "Saved JSON arrays/receipts/static flux arithmetic only; no FV, AD, gradient, or HVP generation",
        "plan_sha256": sha_bytes(plan_bytes),
        "raw_sha256": sha_bytes(raw_bytes),
        "producer_raw_sha256": sha_bytes(prior_bytes),
        "execution_status": child.get("execution_status"),
        "numerical_status": child.get("numerical_status"),
        "base": {"control_sha256": base_sha,
            "carried_theta": plan["initial_theta"],
            "native_J": prior_endpoint_record["objective"],
            "carried_F_squared": prior_endpoint_record["F_squared"],
            "carried_mixed_gradient_norms": gradient_summary(
                [float(x) for x in child.get("initial_F", [])[:26]]),
            "new_fresh_carried_F_squared": carried_f2,
            "initial_F_vector_squared": carried_f2_vector,
            "working_theta_star": working_theta,
            "working_minimum_F_squared": working_f2,
            "working_Psi": working_base.get("Psi") if isinstance(working_base, dict) else None,
            "working_minimum_gradient_norms": (gradient_summary(working_gradient)
                if working_gradient is not None else None),
            "working_F_squared_decrease_from_carried_percent": (
                100.0 * (1.0 - float(working_f2) / carried_f2)
                if isinstance(working_f2, (int, float)) and carried_f2 else None),
            "working_merit_is_optimizer_step": False},
        "selected_endpoint": {"accepted": accepted,
            "control_sha256": terminal_sha,
            "theta": child.get("current_theta"),
            "theta_star": selected_theta,
            "native_J": (final_repeat.get("objective") if final_repeat else None),
            "actual_minimum_F_squared": selected_f2,
            "actual_minimum_Psi": selected_psi,
            "actual_minimum_gradient_norms": (gradient_summary(selected_gradient)
                if selected_gradient is not None else None),
            "F_squared_decrease_from_carried_percent": (
                100.0 * (1.0 - float(selected_f2) / carried_f2)
                if isinstance(selected_f2, (int, float)) and carried_f2 else None),
            "F_squared_decrease_from_working_minimum_percent": (
                100.0 * (1.0 - float(selected_f2) / float(working_f2))
                if isinstance(selected_f2, (int, float)) and isinstance(working_f2, (int, float))
                    and working_f2 else None),
            "native_J_decrease_percent": (100.0 * (1.0 - float(selected_native_j)
                / float(prior_endpoint_record["objective"]))
                if isinstance(selected_native_j, (int, float)) and prior_endpoint_record["objective"] else None),
            "control_movement_l2": movement_norm,
            "control_movement_linf": movement_max,
            "control_movement": movement,
            "static_Q32_base": static_qy(base_control, 3, 2),
            "static_Q43_base": static_qy(base_control, 4, 3),
            "static_Q32_selected": static_qy(terminal_control, 3, 2),
            "static_Q43_selected": static_qy(terminal_control, 4, 3),
            "base_target_face_Q": child.get("face_qualification", {}).get("value"),
            "target_face_Q": (child.get("final_face_audit", {}).get("value") if accepted else None),
            "target_face_production_Q": (child.get("final_face_audit", {}).get("production_value") if accepted else None)},
        "line_model": derivative,
        "candidate_trials": trial_summaries,
        "closure_gates": closure_gates,
        "per_side_gradient_P2_match": p2_gradient_gate,
        "transaction_contract": transaction_contract,
        "hvp_provenance": hvp_provenance,
        "execution_gates": {"parent_completed": parent.get("execution_status") == "completed",
            "parent_child_digest_matches": parent.get("child_sha256") == sha_bytes(raw_bytes),
            "source_receipt_matches": source_before == source_after and child.get("source_unchanged") is True,
            "fixed_input_unchanged": child.get("fixed_input_unchanged") is True,
            "runtime_unchanged": child.get("runtime_unchanged") is True,
            "deadline_passed": child.get("deadline_passed") is True,
            "resource_closed": resource_closed},
        "accounting": {"guarded_launches": plan["policy"]["guarded_launches"],
            "hvp_started": child.get("hvp_calls_started"),
            "hvp_completed": child.get("hvp_calls_completed"),
            "hvp_sides": side_counts,
            "hvp_directions": sorted({row.get("direction_model") for row in history}),
            "jacobian_rows_started": child.get("jacobian_rows_started"),
            "jacobian_rows_completed": child.get("jacobian_rows_completed"),
            "pcg_solves": plan["policy"]["pcg_solves"],
            "dense_solves": plan["policy"]["dense_solves"],
            "optimizer_steps_applied": child.get("optimizer_steps_applied"),
            "accepted_iterations": child.get("accepted_iterations"),
            "sampled_peak_rss_bytes": resource.get("sampled_peak_rss_bytes"),
            "elapsed_seconds": resource.get("elapsed_seconds")},
        "claims": {"full_smooth_root": child.get("full_smooth_root"),
            "minimum_claim": child.get("minimum_claim"),
            "response_claim": child.get("response_claim"),
            "score_claim": plan["policy"].get("score_claim")},
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, default=ATTEMPT / "step.json")
    parser.add_argument("--run", type=Path, default=ATTEMPT / "step.run.json")
    parser.add_argument("--resource", type=Path, default=ATTEMPT / "step.resource.json")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "MIXMIN_RESULT_20261010.json")
    args = parser.parse_args()
    result = collect(args.child, args.run, args.resource, args.plan)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "plan_sha256", "raw_sha256", "execution_status", "numerical_status",
        "base", "selected_endpoint", "per_side_gradient_P2_match",
        "hvp_provenance", "accounting", "execution_gates", "claims")},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
