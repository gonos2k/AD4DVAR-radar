"""Compare two archived R2-O controls across the fixed 54-stage point branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any

import torch
from torch import Tensor

from advar import transport
from examples.weather_scenarios import fv_point_basin_probe as basin
from examples.weather_scenarios import fv_point_core_strict_root_probe as core
from examples.weather_scenarios import fv_point_merit_root_probe as merit
from examples.weather_scenarios import fv_point_response_preflight as preflight


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
RAW = EVIDENCE / "point_merit_root_attempt1/point_merit_root.json"
RAW_SHA256 = "929219cadd5e96e3c66f01abb6497b976462c213874686a9d8c3da92023d9c44"
PRIOR = EVIDENCE / "point_sector_root_attempt1/point_sector_root.json"
PRIOR_SHA256 = "c32d148cd08523661f4838ceebd518f57c26e6b92f0ebcf25d6493a4c1658aab"
PLAN = EVIDENCE / "R2_POINT_EVENT_DIAGNOSTIC_PLAN_20261003.md"
PLAN_SHA256 = "0f5995d73c2245934acf5869111e7112b8f3953f6b70c3051247b0f22a40eab5"
SELF = "examples/weather_scenarios/fv_point_event_diagnostic.py"
SELF_CANONICAL_SHA256 = "93aa60625613fc41de2245badca826daf86b96303de13e7466594344aca41025"
_SELF_PIN = re.compile(rb'(?m)^SELF_CANONICAL_SHA256 = "[0-9a-f]{64}"$')
EPS64 = torch.finfo(torch.float64).eps
ACCEPTED_SHA = "dd2f711b5bd111515b521d6448fa0ccde28999e327f56c2457566b98e284f263"
REFUSED_SHA = "bb51610ce59207707ea493bb02bc69baab565d208787650de7bdaf5c4fddde89"
SEED_SHA = "17fe09298738471223846280f77095f7f3bcb96d9e1d527dacf71a62b0628a8f"
SEED_SIGNATURE = "ae447d2aeb384f7c39f7c97ba9b034fe3cdd6b7222f5d58d2b8587b2972b1da5"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def canonical_source_sha256(source: bytes) -> str:
    normalized, count = _SELF_PIN.subn(b'SELF_CANONICAL_SHA256 = "<canonical-self-pin>"', source)
    if count != 1:
        raise ValueError("event diagnostic canonical source pin is missing or ambiguous")
    return hashlib.sha256(normalized).hexdigest()


def require_self_source_pin(source: bytes, expected: str) -> None:
    if canonical_source_sha256(source) != expected:
        raise ValueError("event diagnostic source pin changed")


def require_source_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    changed = changed_source_paths(before, after)
    if changed:
        raise ValueError(f"event diagnostic source changed: {changed}")


def changed_source_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    if before.keys() != after.keys():
        raise ValueError("source maps have different path sets")
    return sorted(path for path in before if before[path] != after[path])


def validate_trace(trace: list[dict[str, Any]]) -> None:
    if not isinstance(trace, list) or len(trace) != 54:
        raise ValueError("point trace must contain all 54 Euler stages")
    for index, row in enumerate(trace):
        if (row.get("step") != index // 2 or row.get("stage") != index % 2
                or len(row.get("face_signs", {}).get("qx", [])) != 24
                or len(row.get("face_signs", {}).get("qy", [])) != 25):
            raise ValueError("point trace stage or face-sign layout is malformed")
        limiters = row.get("limiters", {})
        if any(len(limiters.get(axis, [])) != 6 for axis in ("x", "y")):
            raise ValueError("point trace limiter layout is malformed")
        if any(set(cell) != {"choice", "slope_sign", "left_sign", "right_sign"}
               for axis in ("x", "y") for cell in limiters[axis]):
            raise ValueError("point trace limiter signature fields are incomplete")


def compare_traces(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    validate_trace(before)
    validate_trace(after)
    faces, limiters = [], []
    for stage, (left, right) in enumerate(zip(before, after, strict=True)):
        for axis, width in (("qx", 6), ("qy", 5)):
            for flat, (a, b) in enumerate(zip(left["face_signs"][axis], right["face_signs"][axis], strict=True)):
                if a != b:
                    faces.append({"step": stage // 2, "stage": stage % 2, "axis": axis,
                                  "row": flat // width, "column": flat % width,
                                  "before": a, "after": b})
        for axis in ("x", "y"):
            for cell, (a, b) in enumerate(zip(left["limiters"][axis], right["limiters"][axis], strict=True)):
                if a != b:
                    limiters.append({"step": stage // 2, "stage": stage % 2, "axis": axis,
                                     "row": cell // 3 + 1, "column": cell % 3 + 1,
                                     "before": a, "after": b})
    locations = {(row["axis"], row["row"], row["column"]) for row in faces}
    candidate = (len(faces) == 54 and len(locations) == 1
                 and locations == {("qx", 3, 4)} and not limiters)
    return {"face_sign_differences": faces, "limiter_differences": limiters,
            "single_qx_3_4_event_candidate": candidate}


def _stage_snapshot(q: Tensor, qx: Tensor, qy: Tensor, index: int) -> dict[str, Any]:
    pairs = ((q[1:-1, 1:-1] - q[1:-1, :-2], q[1:-1, 2:] - q[1:-1, 1:-1]),
             (q[1:-1, 1:-1] - q[:-2, 1:-1], q[2:, 1:-1] - q[1:-1, 1:-1]))
    limiters = {}
    for axis, (left, right) in zip(("x", "y"), pairs, strict=True):
        cells = []
        for a, b in zip(left.flatten(), right.flatten(), strict=True):
            if bool(a * b <= 0):
                choice = 0
            elif bool(a.abs() < b.abs()):
                choice = 1
            elif bool(b.abs() < a.abs()):
                choice = 2
            else:
                choice = 3
            cells.append({"choice": choice, "slope_sign": 1 if bool(a > 0 and b > 0) else
                          (-1 if bool(a < 0 and b < 0) else 0),
                          "left_sign": int(torch.sign(a)), "right_sign": int(torch.sign(b))})
        limiters[axis] = cells
    flux = torch.cat((qx.flatten(), qy.flatten()))
    flat_index = int(flux.abs().argmin())
    scale = float(flux.abs().max())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("point stage has no positive finite face-flux scale")
    if flat_index < qx.numel():
        axis, row, column = "qx", flat_index // qx.shape[1], flat_index % qx.shape[1]
    else:
        qy_index = flat_index - qx.numel()
        axis, row, column = "qy", qy_index // qy.shape[1], qy_index % qy.shape[1]
    return {"step": index // 2, "stage": index % 2, "limiters": limiters,
            "face_signs": {"qx": torch.sign(qx).to(torch.int8).flatten().tolist(),
                           "qy": torch.sign(qy).to(torch.int8).flatten().tolist()},
            "nearest_face": {"axis": axis, "row": row, "column": column,
                             "signed_flux": float(flux[flat_index]),
                             "scaled_abs_margin": float(flux.abs().min() / scale)}}


def _trace_point(problem: Any, control: Tensor, parameters: Tensor) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    snapshots = []
    index = 0

    def collect(q: Tensor, qx: Tensor, qy: Tensor) -> None:
        nonlocal index
        snapshots.append(_stage_snapshot(q, qx, qy, index))
        index += 1

    with transport.observe_minmod_stages(collect):
        branch, scope = core._core_branch(problem, control, parameters)
    validate_trace(snapshots)
    return branch, snapshots, scope


def core_trace_matches(branch: dict[str, Any], trace: list[dict[str, Any]]) -> bool:
    for index, row in enumerate(trace):
        for axis_index, axis in enumerate(("x", "y")):
            nested = branch["choices"][index][axis_index]
            expected = [{"choice": 0 if sign == 0 else (1 if left else 2),
                         "slope_sign": sign,
                         "left_sign": ls, "right_sign": rs}
                        for sign, left, ls, rs in zip(
                            (x for line in nested["slope_sign"] for x in line),
                            (x for line in nested["choose_left"] for x in line),
                            (x for line in nested["left_sign"] for x in line),
                            (x for line in nested["right_sign"] for x in line), strict=True)]
            if row["limiters"][axis] != expected:
                return False
        for axis in ("qx", "qy"):
            expected_signs = [value for line in branch["face_signs"][index][axis] for value in line]
            if row["face_signs"][axis] != expected_signs:
                return False
    return True


def _check_core_match(branch: dict[str, Any], trace: list[dict[str, Any]]) -> None:
    if not core_trace_matches(branch, trace):
        raise ValueError("captured observer branch fields disagree with core oracle")


def _gradient_budget(saved: float, actual: float) -> float:
    return 128 * EPS64 * max(abs(saved), abs(actual), torch.finfo(torch.float64).tiny)


def _metric_check(saved: float, actual: float) -> dict[str, float | bool]:
    budget = _gradient_budget(saved, actual)
    difference = abs(saved - actual)
    return {"archived": saved, "fresh": actual, "absolute_difference": difference,
            "comparison_budget": budget, "passed": difference <= budget}


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("point event diagnostic output must be fresh")
    if _sha(RAW) != RAW_SHA256 or _sha(PRIOR) != PRIOR_SHA256 or _sha(PLAN) != PLAN_SHA256:
        raise ValueError("pinned event diagnostic inputs or plan changed")
    prior = json.loads(RAW.read_text())
    if (prior.get("numerical_status") != "merit_refused"
            or prior.get("response_validation") != "not_performed"
            or prior.get("seed_control_sha256") != SEED_SHA
            or prior.get("seed_branch", {}).get("signature_sha256") != SEED_SIGNATURE
            or prior.get("source_unchanged") is not True
            or prior.get("input_unchanged") is not True):
        raise ValueError("the archived merit attempt is not the declared refusal")
    source_before = merit._sources()
    self_before = _sha(ROOT / SELF)
    require_self_source_pin((ROOT / SELF).read_bytes(), SELF_CANONICAL_SHA256)
    archived_source_paths = changed_source_paths(prior["source_before"], source_before)
    allowed = {"examples/weather_scenarios/fv86_resource_runner.py",
               "src/advar/fv_point_research_problem.py"}
    if set(archived_source_paths) != allowed:
        raise ValueError("unexpected drift from archived merit-root source set")
    archived_source_differences = {
        path: {"archived_sha256": prior["source_before"][path],
               "current_sha256": source_before[path]}
        for path in archived_source_paths
    }
    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"}
    if environment != prior["environment"]:
        raise ValueError("event diagnostic runtime differs from archived merit attempt")
    problem, warm, parameters, direction = preflight.fixed_problem()
    input_before = preflight._input_identity(problem, warm, parameters, direction)
    if input_before != prior["input_before"]:
        raise ValueError("current fixed original problem/parameters differ from archived input")
    trials = prior["trial_records"]
    accepted = next((row for row in reversed(trials)
                     if row.get("iteration") == 6 and row.get("backtrack") == 13
                     and row.get("accepted") is True), None)
    refused = next((row for row in reversed(trials)
                    if row.get("iteration") == 7 and row.get("backtrack") == 15
                    and row.get("accepted") is False), None)
    if accepted is None or refused is None:
        raise ValueError("declared last-accepted/refused trial records are missing")
    controls = []
    for row, expected_sha in ((accepted, ACCEPTED_SHA), (refused, REFUSED_SHA)):
        control = torch.tensor(row["candidate_control"], dtype=torch.float64)
        if control.shape != (26,) or _tensor_sha(control) != expected_sha:
            raise ValueError("archived event diagnostic control/hash mismatch")
        controls.append(control)
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    evaluations = []
    traces = []
    for control, row in zip(controls, (accepted, refused), strict=True):
        branch, trace, scope = _trace_point(problem, control, parameters)
        _check_core_match(branch, trace)
        signature = basin._signature_sha(branch)
        if signature != row["branch_signature_sha256"]:
            raise ValueError("fresh 54-stage signature differs from archived merit trial")
        value = problem.objective(control, parameters)
        gradient = gradient_fn(control, parameters)
        if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
            raise ValueError("reconstructed point objective/gradient is nonfinite")
        j, gmax, gn = float(value), float(gradient.abs().max()), float(torch.linalg.vector_norm(gradient))
        checks = {name: _metric_check(saved, actual) for name, saved, actual in (
            ("objective", row["objective"], j),
            ("gradient_max", row["gradient_max"], gmax),
            ("gradient_norm", row["gradient_norm"], gn),
        )}
        if not all(bool(check["passed"]) for check in checks.values()):
            raise ValueError("reconstructed objective/gradient differs from archived trial")
        frozen = problem.frozen.fv_transport
        if frozen is None:
            raise ValueError("point flow basis is absent")
        coefficients = frozen.coefficient_limits * torch.tanh(control[20:25])
        growth = problem.frozen.nowcast_config.max_log_growth_per_step * torch.tanh(control[25])
        psi_basis = frozen.psi_basis
        face_weights = psi_basis[:, 4, 4] - psi_basis[:, 3, 4]
        qx_3_4 = torch.dot(coefficients, face_weights)
        blocks = {"field_l2": float(torch.linalg.vector_norm(gradient[:20])),
                  "flow_l2": float(torch.linalg.vector_norm(gradient[20:25])),
                  "growth_abs": float(gradient[25].abs())}
        evaluations.append({"trial": {k: row.get(k) for k in (
                                "iteration", "backtrack", "step_scale", "accepted", "rejection",
                                "policy_reason", "objective", "gradient_norm", "gradient_max",
                                "normalized_slope", "linear_relative_residual")},
                            "control_sha256": _tensor_sha(control), "control": control.tolist(),
                            "objective": j, "gradient": gradient.tolist(), "gradient_norm": gn,
                            "gradient_max": gmax, "gradient_blocks": blocks,
                            "archived_metric_checks": checks,
                            "flow_latent": control[20:25].tolist(),
                            "flow_coefficients": coefficients.tolist(),
                            "qx_3_4_basis_weights": face_weights.tolist(),
                            "qx_3_4_from_coefficients": float(qx_3_4),
                            "growth_latent": float(control[25]), "growth_log_per_interval": float(growth),
                            "branch_signature_sha256": signature,
                            "minimum_scaled_slope_margin": branch["minimum_scaled_slope_margin"],
                            "minimum_scaled_face_flux_margin": branch["minimum_scaled_face_flux_margin"],
                            "nearest_face": min((stage["nearest_face"] for stage in trace),
                                key=lambda face: face["scaled_abs_margin"]),
                            "stage_trace": trace, "branch_scope": scope})
        traces.append(trace)
    differences = compare_traces(traces[0], traces[1])
    for evaluation in evaluations:
        evaluation["trace_differences_from_other_point"] = differences
    gradient_delta = (torch.tensor(evaluations[1]["gradient"], dtype=torch.float64)
                      - torch.tensor(evaluations[0]["gradient"], dtype=torch.float64))
    input_after = preflight._input_identity(problem, warm, parameters, direction)
    source_after = merit._sources()
    require_source_unchanged(source_before, source_after)
    environment_after = {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"}
    if (input_after != input_before or environment_after != environment
            or _sha(ROOT / SELF) != self_before
            or _sha(RAW) != RAW_SHA256 or _sha(PRIOR) != PRIOR_SHA256 or _sha(PLAN) != PLAN_SHA256):
        raise ValueError("event diagnostic source or input changed during evaluation")
    require_self_source_pin((ROOT / SELF).read_bytes(), SELF_CANONICAL_SHA256)
    result = {
        "pid": os.getpid(), "phase": "finished", "numerical_status": "diagnostic_only",
        "source_unchanged": True, "input_unchanged": True,
        "scope": "two saved finite-offset points only; no optimizer, response, constrained stationarity, or root claim",
        "raw_sha256": RAW_SHA256, "prior_report_sha256": PRIOR_SHA256,
        "plan_sha256": PLAN_SHA256, "runtime": environment, "runtime_after": environment_after,
        "source_before": source_before, "source_after": source_after,
        "event_diagnostic_source_sha256": self_before,
        "archived_current_source_differences": archived_source_differences,
        "input_before": input_before, "input_after": input_after,
        "points": evaluations, "indexed_branch_differences": differences,
        "gradient_component_difference_refused_minus_accepted": gradient_delta.tolist(),
        "gradient_difference_blocks": {
            "field_l2": float(torch.linalg.vector_norm(gradient_delta[:20])),
            "flow_l2": float(torch.linalg.vector_norm(gradient_delta[20:25])),
            "growth_abs": float(gradient_delta[25].abs()),
        },
        "single_face_event_candidate": differences["single_qx_3_4_event_candidate"],
        "root_absence_claim": False, "response_validation": "not_performed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
