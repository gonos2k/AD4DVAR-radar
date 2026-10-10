"""Reanalyze saved local-window arrays without running the FV model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "GN_LOCAL_WINDOW_PLAN_20261010.json"
ATTEMPT = EVIDENCE / "gn_local_window_20261010_attempt2"
FAILED_ATTEMPT = EVIDENCE / "gn_local_window_20261010_attempt1"
MEMORY = EVIDENCE / "tangent_coupled_gn_memory_resume_20261010_attempt1"
MEMORY_MANIFEST = EVIDENCE / "STREAMGN_MEMORY_ARCHIVE_20261010.json"
PREFLIGHT_MANIFEST = EVIDENCE / "GN_LOCAL_WINDOW_PREFLIGHT_20261010.json"
PLAN_SHA256 = "f0bce6629d3de1071276ce3cf93dcec70a2cba79f8d5661f847f7826670a92a9"
PREFLIGHT_MANIFEST_SHA256 = "2fba6416002eeb7c4b2aee78c8cc38cacf2983fb2585423d99b7dd759597b8c9"
BASE_SHA256 = "2ec34eeeef531b94e21b5b4372ced2e07257a5407541072d300f5a3f513e1fcf"
DIRECTION_SHA256 = "0240d2e18fb8dfd4012bd7a0c8d3dd75f13cc12c81c5c941db9afafb3c3d6e94"
OLD_ALPHA0 = 0.7306593405044787
OLD_MIN_ALPHA = OLD_ALPHA0 / 2**15
SAMPLE_ALPHAS = tuple(OLD_MIN_ALPHA / (2**k) for k in range(1, 9))
EPS = 2.220446049250313e-16
FACE_WEIGHTS = (0.0, -0.08, -0.28, -0.14, -0.84)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"path is a symlink or outside the repository: {path}")
    return resolved


def _load_json(path: Path) -> Any:
    return json.loads(_inside(path).read_text())


def _close(a: float, b: float, scale: float | None = None) -> bool:
    unit = max(abs(a), abs(b), abs(scale or 0.0), 1e-300)
    return abs(a - b) <= 256 * EPS * unit


def _norm(values: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _chart_pivot(control: list[float]) -> int:
    flow = control[20:25]
    normal = [weight * (1.0 - math.tanh(value) ** 2)
        for weight, value in zip(FACE_WEIGHTS, flow)]
    return 20 + max(range(5), key=lambda index: abs(normal[index]))


def _verify_chart_point(base: list[float], direction: list[float], control: list[float],
                        alpha: float, pivot: int) -> dict[str, float]:
    retained = [index for index in range(26) if index != pivot]
    errors = [abs(control[index] - (base[index] + alpha * direction[index]))
        for index in retained]
    budgets = [256 * EPS * max(1.0, abs(base[index]), abs(alpha * direction[index]))
        for index in retained]
    if any(error > budget for error, budget in zip(errors, budgets)):
        raise ValueError("saved point does not follow retained coordinates of the fixed face chart")
    flow_indices = [index for index in range(20, 25) if index != pivot]
    pflow = pivot - 20
    argument = -math.fsum(FACE_WEIGHTS[index - 20] * math.tanh(control[index])
        for index in flow_indices) / FACE_WEIGHTS[pflow]
    if not math.isfinite(argument) or abs(argument) >= 1:
        raise ValueError("saved retained flow coordinates lie outside the real atanh chart")
    pivot_prediction = math.atanh(argument)
    pivot_error = abs(control[pivot] - pivot_prediction)
    pivot_budget = 256 * EPS * max(1.0, abs(control[pivot]), abs(pivot_prediction))
    face_q = math.fsum(weight * math.tanh(value)
        for weight, value in zip(FACE_WEIGHTS, control[20:25]))
    if pivot_error > pivot_budget:
        raise ValueError("saved pivot coordinate differs from the independent atanh reconstruction")
    return {"retained_max_abs_error": max(errors, default=0.0),
        "retained_max_budget": max(budgets, default=0.0),
        "pivot_control_error": pivot_error, "pivot_control_budget": pivot_budget,
        "static_face_q_recomputed": face_q}


def _first_difference(left: Any, right: Any, path: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                return path + (key,)
            found = _first_difference(left[key], right[key], path + (key,))
            if found is not None:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path + (min(len(left), len(right)),)
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, path + (index,))
            if found is not None:
                return found
        return None
    return None if left == right else path


def _choice_location(path: tuple[Any, ...] | None) -> dict[str, Any] | None:
    if path is None:
        return None
    orientation = path[1] if len(path) > 1 and isinstance(path[1], int) else None
    return {"stage": path[0] if path else None,
        "axis": ("x", "y")[orientation] if orientation in (0, 1) else None,
        "key": path[2] if len(path) > 2 else None,
        "row": path[3] if len(path) > 3 and isinstance(path[3], int) else None,
        "column": path[4] if len(path) > 4 and isinstance(path[4], int) else None}


def _sign_location(path: tuple[Any, ...] | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"stage": path[0] if path else None,
        "axis": path[1] if len(path) > 1 and path[1] in ("x", "y") else None,
        "row": path[2] if len(path) > 2 and isinstance(path[2], int) else None,
        "column": path[3] if len(path) > 3 and isinstance(path[3], int) else None}


def _trace_summary(base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    selector_path = _first_difference(base.get("choices"), current.get("choices"))
    sign_path = _first_difference(base.get("face_signs"), current.get("face_signs"))
    return {"signature_sha256": current.get("signature_sha256"),
        "choices_changed": selector_path is not None,
        "face_signs_changed": sign_path is not None,
        "first_changed_selector": _choice_location(selector_path),
        "first_changed_face_sign": _sign_location(sign_path),
        "stage_count": current.get("stage_count"),
        "minimum_margins": current.get("minimum_margins"),
        "nonfinite_or_tie": current.get("nonfinite_or_tie")}


def _canonical_signature(trace: dict[str, Any]) -> str:
    payload = {"choices": trace.get("choices"), "face_signs": trace.get("face_signs")}
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode())


def _validate_trace(trace: dict[str, Any]) -> None:
    if (trace.get("stage_count") != 360 or trace.get("observed_stage_count") != 360
            or trace.get("expected_stage_count") != 360
            or trace.get("nonfinite_or_tie") is not False
            or trace.get("signature_sha256") != _canonical_signature(trace)):
        raise ValueError("branch trace is incomplete, tied, or has an invalid signature")


def _validate_plan(plan_path: Path) -> dict[str, Any]:
    if _file_sha(_inside(plan_path)) != PLAN_SHA256:
        raise ValueError("frozen local-window plan digest changed")
    plan = _load_json(plan_path)
    if (plan.get("experiment_kind") != "gn_local_window_diagnostic"
            or plan.get("base_control_sha256") != BASE_SHA256
            or plan.get("direction_sha256") != DIRECTION_SHA256
            or plan.get("old_minimum_tested_alpha") != OLD_MIN_ALPHA
            or plan.get("sample_alphas") != list(SAMPLE_ALPHAS)
            or plan.get("preflight_manifest") != PREFLIGHT_MANIFEST.relative_to(ROOT).as_posix()
            or plan.get("preflight_manifest_sha256") != PREFLIGHT_MANIFEST_SHA256
            or plan.get("policy", {}).get("optimizer_steps") != 0
            or plan.get("policy", {}).get("jacobian_row_vjp_calls") != 0
            or plan.get("policy", {}).get("dense_solves") != 0
            or plan.get("policy", {}).get("candidate_samples") != 8):
        raise ValueError("frozen plan differs from the fixed saved-only analysis contract")
    if _file_sha(_inside(PREFLIGHT_MANIFEST)) != PREFLIGHT_MANIFEST_SHA256:
        raise ValueError("failed first preflight manifest changed")
    preflight = _load_json(PREFLIGHT_MANIFEST)
    if (preflight.get("execution_status") != "failed"
            or preflight.get("diagnostic_child_present") is not False
            or preflight.get("numerical_status") != "not_reached"
            or preflight.get("guarded_launches") != 1
            or preflight.get("failed_plan_sha256") != "32618a3bba6d70a6cf12e4624abfc2bfa2f30a6b5c7ba94d94fb3df4de32895c"):
        raise ValueError("first failed source-admission attempt is not preserved separately")
    if ((FAILED_ATTEMPT / "diagnostic.json").exists()
            or (FAILED_ATTEMPT / "diagnostic.json").is_symlink()):
        raise ValueError("first preflight failure unexpectedly has a numerical child")
    failed_run = _load_json(FAILED_ATTEMPT / "diagnostic.run.json")
    failed_resource = _load_json(FAILED_ATTEMPT / "diagnostic.resource.json")
    if (failed_run.get("execution_status") != "failed"
            or failed_run.get("child_sha256") is not None
            or failed_run.get("resource") != failed_resource
            or failed_resource.get("exit_code") != 1
            or failed_resource.get("resource_termination") is not None):
        raise ValueError("first failed attempt receipts do not close as a pre-model refusal")
    memory_plan_path = EVIDENCE / "TANGENT_COUPLED_GN_MEMORY_RESUME_PLAN_20261010.json"
    prior = _load_json(memory_plan_path)
    added_archives = {
        memory_plan_path.relative_to(ROOT).as_posix(),
        MEMORY_MANIFEST.relative_to(ROOT).as_posix(),
        (MEMORY / "step.json.gz").relative_to(ROOT).as_posix(),
        (MEMORY / "step.run.json").relative_to(ROOT).as_posix(),
        (MEMORY / "step.resource.json").relative_to(ROOT).as_posix(),
        PREFLIGHT_MANIFEST.relative_to(ROOT).as_posix(),
    }
    added_archives.update(item["archive_path"] for item in preflight["snapshots"].values())
    added_archives.update(item["path"] for item in preflight["receipts"].values())
    if (set(plan["source_files"]) != set(prior["source_files"]) | {
            "examples/weather_scenarios/fv_point_3h_gn_local_window_probe.py",
            "tests/test_fv_point_3h_gn_local_window_probe.py"}
            or set(plan["archive_files"]) != set(prior["archive_files"]) | added_archives
            or len(plan["source_files"]) != 146 or len(plan["archive_files"]) != 185
            or any(plan["source_files"].get(key) != value for key, value in prior["source_files"].items())
            or any(plan["archive_files"].get(key) != value for key, value in prior["archive_files"].items())):
        raise ValueError("successor plan does not preserve predecessor pins and failed preflight provenance")
    for relative, digest in {**plan["source_files"], **plan["archive_files"]}.items():
        path = _inside(ROOT / relative)
        if _file_sha(path) != digest:
            raise ValueError(f"plan pin changed: {relative}")
    return plan


def _load_predecessor(plan: dict[str, Any]
                      ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    gzip_path = MEMORY / "step.json.gz"
    manifest = _load_json(MEMORY_MANIFEST)
    if _file_sha(_inside(gzip_path)) != manifest.get("gzip_sha256"):
        raise ValueError("predecessor memory archive gzip digest changed")
    if manifest.get("gzip_sha256") != plan["archive_files"][gzip_path.relative_to(ROOT).as_posix()]:
        raise ValueError("predecessor memory archive differs from the frozen plan")
    raw_path = MEMORY / "step.json"
    if _file_sha(_inside(raw_path)) != manifest.get("raw_sha256"):
        raise ValueError("predecessor memory raw digest changed")
    with _inside(raw_path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    iteration = raw["iterations"][1]
    arm = next(item for item in iteration["model_comparisons"]
        if item.get("name") == "robust_gn_coupled")
    if (raw.get("last_confirmed_control_sha256") != BASE_SHA256
            or iteration.get("base_control_sha256") != BASE_SHA256
            or arm.get("accepted") is not False
            or hashlib.sha256(struct.pack("<26d", *arm["direction"])).hexdigest() != DIRECTION_SHA256):
        raise ValueError("predecessor does not contain the exact closed base and refused direction")
    closure = raw["last_confirmed_closure"]
    trials = iteration.get("trials", [])
    return closure, arm, trials


def _validate_candidate_trace_pair(sample: dict[str, Any], traces: dict[str, Any],
                                   base_traces: dict[str, Any], index: int) -> dict[str, Any]:
    pair = {side: traces[str(index)][side] for side in ("-1", "1")}
    for trace in pair.values():
        _validate_trace(trace)
    computed_delta = {side: _trace_summary(base_traces[side], pair[side]) for side in pair}
    if sample.get("trace_delta") != computed_delta:
        raise ValueError(f"sample {index} trace selector/sign delta is not reproducible")
    minus, plus = pair["-1"], pair["1"]
    flux_scale = max(float(minus.get("maximum_face_flux", 0.0)),
        float(plus.get("maximum_face_flux", 0.0)), 1e-300)
    target_max = max(max(map(abs, minus.get("target_fluxes", [])), default=0.0),
        max(map(abs, plus.get("target_fluxes", [])), default=0.0))
    gate = (minus["choices"] == plus["choices"]
        and minus["face_signs"] == plus["face_signs"]
        and target_max <= 128 * EPS * flux_scale)
    if bool(sample.get("strict_branch_pair_passed")) != gate:
        raise ValueError(f"sample {index} current branch-pair result differs from its traces")
    return {"current_pair_passed": gate,
        "minus": computed_delta["-1"], "plus": computed_delta["1"]}


def _analyze(plan_path: Path, diagnostic_path: Path, run_path: Path,
             resource_path: Path) -> dict[str, Any]:
    plan = _validate_plan(plan_path)
    preflight = _load_json(PREFLIGHT_MANIFEST)
    failed_resource = _load_json(FAILED_ATTEMPT / "diagnostic.resource.json")
    diagnostic_path, run_path, resource_path = map(_inside,
        (diagnostic_path, run_path, resource_path))
    diagnostic = _load_json(diagnostic_path)
    run = _load_json(run_path)
    resource = _load_json(resource_path)
    child_sha = _file_sha(diagnostic_path)
    if (run.get("execution_status") != "completed" or run.get("child_sha256") != child_sha
            or run.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("rss_limit_bytes") != plan["policy"]["rss_bytes"]
            or resource.get("wall_limit_seconds") != plan["policy"]["outer_seconds"]):
        raise ValueError("diagnostic child/run/resource receipts do not close")
    if (diagnostic.get("plan_sha256") != PLAN_SHA256
            or diagnostic.get("numerical_status") != "local_window_diagnostic_completed_no_acceptance_or_step_selection"
            or diagnostic.get("base_control_sha256") != BASE_SHA256
            or diagnostic.get("direction_sha256") != DIRECTION_SHA256
            or diagnostic.get("optimizer_steps_applied") != 0
            or diagnostic.get("candidate_committed") is not False
            or diagnostic.get("acceptance_claim") is not False
            or diagnostic.get("jacobian_rows_completed") != 0
            or diagnostic.get("dense_solves_completed") != 0
            or diagnostic.get("hvp_calls_completed") != 2
            or len(diagnostic.get("samples", [])) != 8
            or diagnostic.get("source_unchanged") is not True
            or diagnostic.get("source_before") != diagnostic.get("source_after")
            or diagnostic.get("fixed_input_unchanged") is not True
            or diagnostic.get("input_before") != diagnostic.get("input_after")
            or diagnostic.get("runtime_unchanged") is not True
            or diagnostic.get("runtime_before") != diagnostic.get("runtime_after")):
        raise ValueError("diagnostic output violates the frozen no-commit/eight-point contract")

    accepted_closure, refused_arm, old_trials = _load_predecessor(plan)
    if (len(old_trials) != 16
            or not all(item.get("J_armijo_passed") is True
                and item.get("F_squared_armijo_passed") is False
                and item.get("face_audit_passed") is True
                and item.get("branch_pair_passed") is True
                and item.get("mixing_minimum_valid") is True for item in old_trials)
            or not _close(float(old_trials[-1]["alpha"]), OLD_MIN_ALPHA)):
        raise ValueError("saved 16-point refused path or its smallest-alpha gate receipt changed")
    base_traces = diagnostic.get("base_full_branch_traces", {})
    baseline_trace_audit: dict[str, Any] = {}
    for side in ("-1", "1"):
        trace = base_traces.get(side)
        accepted_trace = accepted_closure.get("branch_trace", {}).get(side)
        if not isinstance(trace, dict) or not isinstance(accepted_trace, dict):
            raise ValueError("full baseline branch traces are missing")
        _validate_trace(trace)
        _validate_trace(accepted_trace)
        if trace != accepted_trace:
            raise ValueError(f"fresh baseline full branch trace differs from accepted closure ({side})")
        baseline_trace_audit[side] = {"signature_matches_closure": True,
            "stage_count": trace["stage_count"], "minimum_margins": trace["minimum_margins"]}
        fresh_gradient = [float(value) for value in diagnostic["base_gminus" if side == "-1" else "base_gplus"]]
        accepted_gradient = [float(value) for value in accepted_closure["side_gradients"][side]]
        if len(fresh_gradient) != 26 or any(not _close(a, b, max(abs(a), abs(b)))
                for a, b in zip(fresh_gradient, accepted_gradient)):
            raise ValueError(f"fresh base side gradient differs from accepted closure ({side})")

    base_control = [float(v) for v in diagnostic["base_control"]]
    direction = [float(v) for v in refused_arm["direction"]]
    g0 = [float(v) for v in diagnostic["fresh_residual"]]
    r = [float(v) for v in diagnostic["fresh_residual_direction"]]
    if (len(base_control) != 26 or len(direction) != 26 or len(g0) != 27 or len(r) != 27
            or base_control != [float(value) for value in accepted_closure["control"]]
            or hashlib.sha256(struct.pack("<26d", *base_control)).hexdigest() != BASE_SHA256
            or diagnostic.get("base_theta") != plan["initial_theta"]
            or diagnostic.get("base_objective") != plan["base_objective"]
            or diagnostic.get("base_F_squared") != plan["base_F_squared"]):
        raise ValueError("diagnostic base arrays do not match the frozen point dimensions or metrics")
    if (diagnostic.get("initial_side_gradients_match") is not True
            or diagnostic.get("initial_branch_traces_match") is not True
            or diagnostic.get("base_face_audit_passed") is not True
            or diagnostic.get("direction_chart_error", math.inf) > 1e-12
            or abs(float(diagnostic.get("direction_face_normal_dot", math.inf))) > 1e-12
            or not all(item.get("passed") is True for item in
                diagnostic.get("fresh_hvp_component_scaled_comparison", {}).values())
            or not all(item.get("passed") is True for item in
                diagnostic.get("fresh_model_G_r_component_scaled_comparison", {}).values())):
        raise ValueError("same-point closure, chart, HVP, or G/r check did not pass")
    hvp_history = diagnostic.get("hvp_history", [])
    if (len(hvp_history) != 2 or {item.get("side") for item in hvp_history} != {-1, 1}
            or any(item.get("base_control_sha256") != BASE_SHA256
                or item.get("direction_sha256") != DIRECTION_SHA256
                or item.get("operator") != "selected_face_extension"
                or item.get("status") != "completed"
                or item.get("stored_receipt", {}).get("theta") != plan["initial_theta"]
                or item.get("stored_receipt", {}).get("working_theta") != plan["initial_theta"]
                or item.get("stored_receipt", {}).get("direction_model") != "robust_gn_coupled"
                or item.get("stored_receipt", {}).get("operator") != "selected_face_extension"
                or item.get("stored_receipt", {}).get("status") != "completed"
                for item in hvp_history)):
        raise ValueError("fresh HVP receipts are not exactly the same-point −/+ pair")
    j_slope = max(sum(a * b for a, b in zip(diagnostic["base_gminus"], direction)),
        sum(a * b for a, b in zip(diagnostic["base_gplus"], direction)))
    f2_base = float(plan["base_F_squared"])
    j_base = float(plan["base_objective"])
    model_slope = float(diagnostic["merit_product"])
    traces = diagnostic.get("full_branch_traces", {})
    sample_reports = []
    chart_pivot = _chart_pivot(base_control)
    if chart_pivot not in range(20, 25):
        raise ValueError("fixed selected-face normal has no valid flow pivot")
    for index, (sample, expected_alpha) in enumerate(zip(diagnostic["samples"], SAMPLE_ALPHAS), 1):
        alpha = float(sample["alpha"])
        if not _close(alpha, expected_alpha):
            raise ValueError(f"sample {index} alpha differs from the frozen dyadic grid")
        control = [float(v) for v in sample["control"]]
        if len(control) != 26:
            raise ValueError(f"sample {index} control has the wrong dimension")
        chart_metrics = _verify_chart_point(base_control, direction, control, alpha, chart_pivot)
        if hashlib.sha256(struct.pack("<26d", *control)).hexdigest() != sample.get("control_sha256"):
            raise ValueError(f"sample {index} control digest does not match its stored vector")

        G = [float(v) for v in sample["G"]]
        E = [G[i] - g0[i] - alpha * r[i] for i in range(27)]
        recorded_E = [float(v) for v in sample["E"]]
        if any(not _close(a, b, max(abs(x), abs(y))) for a, b, x, y
               in zip(E, recorded_E, G, g0)):
            raise ValueError(f"sample {index} stored nonlinear residual remainder is inconsistent")
        E_over_h = [value / alpha for value in E]
        E_over_h2 = [value / (alpha * alpha) for value in E]
        for field, expected in (("E_over_h", E_over_h), ("E_over_h2", E_over_h2)):
            recorded = [float(v) for v in sample[field]]
            if any(not _close(a, b, max(abs(a), abs(b))) for a, b in zip(expected, recorded)):
                raise ValueError(f"sample {index} {field} array is inconsistent")

        actual_f2 = math.fsum(value * value for value in G)
        f2_rhs = f2_base + 2e-4 * alpha * model_slope
        actual_j = float(sample["actual_J"])
        j_rhs = j_base + 1e-4 * alpha * j_slope
        f2_pass = actual_f2 <= f2_rhs
        j_pass = actual_j <= j_rhs
        if (not _close(actual_f2, float(sample["actual_F_squared"]), f2_base)
                or not _close(f2_rhs, float(sample["actual_F2_armijo_rhs"]), f2_base)
                or not _close(j_rhs, float(sample["actual_J_armijo_rhs"]), j_base)
                or sample.get("actual_F2_armijo_passed") is not f2_pass
                or sample.get("actual_J_armijo_passed") is not j_pass):
            raise ValueError(f"sample {index} actual dual Armijo values are inconsistent")
        trace_audit = _validate_candidate_trace_pair(sample, traces, base_traces, index)
        q = float(sample["face_q"])
        production_q = float(sample["production_face_value"])
        face_bound = float(sample["face_roundoff_bound"])
        static_face_q = float(chart_metrics["static_face_q_recomputed"])
        if abs(static_face_q - q) > face_bound or abs(static_face_q) > face_bound:
            raise ValueError(f"sample {index} static flow-face equation differs from its saved Q")
        face_ok = (sample.get("face_audit_passed") is True and abs(q) <= face_bound
            and abs(production_q) <= face_bound)
        if not face_ok:
            raise ValueError(f"sample {index} selected-face value fails its saved audit budget")
        displacement = [control[i] - base_control[i] for i in range(26)]
        sample_reports.append({"index": index, "alpha": alpha,
            "chart_pivot_from_fixed_face_normal": chart_pivot,
            "retained_chart_max_abs_error": chart_metrics["retained_max_abs_error"],
            "retained_chart_max_budget": chart_metrics["retained_max_budget"],
            "pivot_atanh_error": chart_metrics["pivot_control_error"],
            "pivot_atanh_budget": chart_metrics["pivot_control_budget"],
            "static_face_q_recomputed": static_face_q,
            "actual_displacement_l2": _norm(displacement),
            "control_block_displacements": {
                "initial_field_20_l2": _norm(displacement[:20]),
                "flow_5_l2": _norm(displacement[20:25]),
                "growth_1_abs": abs(displacement[25]),
                "control_linf": max(map(abs, displacement))},
            "predicted_displacement_l2": alpha * _norm(direction),
            "G_l2": _norm(G), "E_l2": _norm(E),
            "E_over_h_l2": _norm(E_over_h), "E_over_h2_l2": _norm(E_over_h2),
            "actual_F_squared": actual_f2, "actual_F2_armijo_rhs": f2_rhs,
            "actual_F2_armijo_passed": f2_pass,
            "actual_J": actual_j, "actual_J_armijo_rhs": j_rhs,
            "actual_J_armijo_passed": j_pass,
            "face_q": q, "production_face_value": production_q,
            "face_roundoff_bound": face_bound, "face_audit_passed": face_ok,
            "control_sha256": sample["control_sha256"], "trace_audit": trace_audit})
    if chart_pivot not in range(20, 25):
        raise ValueError("fixed-face pivot derivation returned an invalid control")
    if diagnostic.get("source_unchanged") is not True or diagnostic.get("runtime_unchanged") is not True:
        raise ValueError("diagnostic source/runtime receipts did not close")
    return {"scope": "saved diagnostic arrays/receipts only; no FV, AD, HVP, optimizer, or step selection",
        "plan_sha256": PLAN_SHA256, "diagnostic_child_sha256": _file_sha(diagnostic_path),
        "analyzer_sha256": _file_sha(Path(__file__)),
        "base_control_sha256": BASE_SHA256, "direction_sha256": DIRECTION_SHA256,
        "old_minimum_tested_alpha": OLD_MIN_ALPHA,
        "sample_count": len(sample_reports), "shared_chart_pivot_verified_from_fixed_normal": chart_pivot,
        "baseline_full_trace_audit": baseline_trace_audit,
        "previous_16_candidate_refusal": {
            "candidate_count": len(old_trials),
            "all_J_armijo_passed": all(item.get("J_armijo_passed") is True for item in old_trials),
            "all_F2_armijo_failed": all(item.get("F_squared_armijo_passed") is False for item in old_trials),
            "old_min_alpha": float(old_trials[-1]["alpha"]),
            "old_min_alpha_F_squared": float(old_trials[-1]["F_squared"]),
            "old_min_alpha_F2_armijo_rhs": float(old_trials[-1]["F_squared_armijo_bound"])},
        "fresh_hvp_receipts": diagnostic["hvp_history"],
        "fresh_hvp_comparisons": diagnostic["fresh_hvp_component_scaled_comparison"],
        "fresh_model_G_r_comparisons": diagnostic["fresh_model_G_r_component_scaled_comparison"],
        "prior_source_admission_failure": {
            "plan_sha256": preflight["failed_plan_sha256"],
            "failure": preflight["failure"], "diagnostic_child_present": False,
            "exit_code": failed_resource["exit_code"],
            "sampled_peak_rss_bytes": failed_resource["sampled_peak_rss_bytes"]},
        "row_vjps_completed": diagnostic["jacobian_rows_completed"],
        "dense_solves_completed": diagnostic["dense_solves_completed"],
        "optimizer_steps_applied": diagnostic["optimizer_steps_applied"],
        "candidate_committed": diagnostic["candidate_committed"],
        "samples": sample_reports,
        "all_samples_failed_F2_armijo": all(not item["actual_F2_armijo_passed"] for item in sample_reports),
        "all_samples_passed_J_armijo": all(item["actual_J_armijo_passed"] for item in sample_reports)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=ATTEMPT / "diagnostic.json")
    parser.add_argument("--run", type=Path, default=ATTEMPT / "diagnostic.run.json")
    parser.add_argument("--resource", type=Path, default=ATTEMPT / "diagnostic.resource.json")
    parser.add_argument("--output", type=Path,
        default=EVIDENCE / "GN_LOCAL_WINDOW_ANALYSIS_20261010.json")
    args = parser.parse_args()
    report = _analyze(PLAN, args.diagnostic, args.run, args.resource)
    output = args.output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"analysis output must be a regular in-repository path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
