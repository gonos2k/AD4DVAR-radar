"""Observe terminal E at two closed controls under one fixed new FV evaluator.

This is an observation-only paired score; it never selects or commits a point.
The caller supplies a frozen plan and separately admitted start/final receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
WORK = EVIDENCE / "netflux_future_observation_20261012"
PLAN = WORK / "PLAN.json"
ATTEMPT = WORK / "attempt1"
PARENT_PLAN = EVIDENCE / "netflux_continuous_20261012/PLAN.json"
PARENT_PLAN_SHA = "5fd879837c2e5898b16b01007aabae34d0eeea7da8b6d094d6d03b9dad366e07"
CASE = EVIDENCE / "netflux_current_cycle_20261012/fixed_case.pt"
CASE_MANIFEST = EVIDENCE / "netflux_current_cycle_20261012/FIXED_CASE_MANIFEST.json"
CASE_SHA = "1e810c23c73b93d89b2a10b46d9ddea44a8217b07dcd3ebfebb462858e3fa9b7"
CASE_MANIFEST_SHA = "82f998c995cd562bf7aee822008a00b1a0a726922554fc950ec605d7c0d3c905"
TRANSPORT = "src/advar/transport.py"
TRANSPORT_SHA = "b0335b7a72ecd22084d5ab8ea0e46707240be15ef0039f4b31f96adbc937ee1a"
NUMERICAL_VERSION = "minmod_net_flux_fp64_v1"
START_ENDPOINT_MODULE = "examples.weather_scenarios.fv_point_3h_netflux_endpoint"
FINAL_ENDPOINT_MODULE = "examples.weather_scenarios.fv_point_3h_netflux_continuous_endpoint"
START_CONTROL_SHA = "0aafdf8069fcb8d5adc4ec813646707a3d1e083804631b17facb24664098655b"
FINAL_CONTROL_SHA = "b6470069c9b8992bf19fa85ab848f5ef424a1e2e8e33a4909bcb0d036ef88660"
PARAMETERS_SHA = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
TRUTH_SHA = "b927c1a39d16f05285cccb7a9ff48d182eea539bfba746f7aff446aee0974610"
PROBLEM_SHA = "16fa95b534c3b61cf5d1506770cbfc77edb527c40d1f4c536f5130cba24e1fc3"
SELF = "examples/weather_scenarios/fv_point_3h_netflux_future_observation.py"
TEST = "tests/test_fv_point_3h_netflux_future_observation.py"
FINAL_ENDPOINT_SOURCE = "examples/weather_scenarios/fv_point_3h_netflux_continuous_endpoint.py"
NEW_SOURCES = {SELF, TEST, FINAL_ENDPOINT_SOURCE,
    "tests/test_fv_point_3h_netflux_continuous_endpoint.py"}
from examples.weather_scenarios import fv_point_3h_netflux_continuous_endpoint as final_endpoint  # noqa: E402
from examples.weather_scenarios import fv_point_3h_netflux_endpoint as start_endpoint  # noqa: E402

RECEIPTS = tuple(dict.fromkeys((PARENT_PLAN, *start_endpoint.RECEIPTS,
    *final_endpoint.RECEIPTS, CASE, CASE_MANIFEST)))
SCORE_PROFILE = {"controls": 26, "parameters": 13, "observation_shape": [3, 4],
    "observation_times_seconds": [0.0, 600.0, 1200.0],
    "forecast_steps": 18, "interval_minutes": 10, "forecast_time_seconds": 12000.0}
POLICY = {"guarded_launches": 1, "internal_seconds": 600.0,
    "outer_seconds": 660.0, "rss_bytes": 2 * 1024**3,
    "score_evaluations": 2, "optimizer_steps": 0,
    "candidate_evaluations": 0, "selection_use": False,
    "future_score_evaluations": 2, "score_order": "start_then_final"}
def _score_profile(problem: Any) -> dict[str, Any]:
    layout = problem.layout
    config = problem.frozen.nowcast_config
    steps = int(config.forecast_steps)
    interval = float(config.interval_minutes)
    profile = {"controls": int(layout["controls"]),
        "parameters": int(layout["parameters"]),
        "observation_shape": list(layout["observation_shape"]),
        "observation_times_seconds": list(layout["observation_times_seconds"]),
        "forecast_steps": steps, "interval_minutes": interval,
        "forecast_time_seconds": float(layout["forecast_time_seconds"])}
    if any(profile.get(key) != value for key, value in SCORE_PROFILE.items()):
        raise ValueError("fixed observation problem differs from the pinned terminal score profile")
    return profile


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned path is not a regular in-repository file: {path}")
    return _sha_bytes(path.read_bytes())


def _tensor_sha(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.device.type != "cpu":
        raise ValueError("observation tensor identity requires a CPU tensor")
    return _sha_bytes(value.detach().contiguous().numpy().tobytes())


def _source_hashes(plan: dict[str, Any], plan_path: Path, digest: str) -> dict[str, str]:
    pinned = {**plan.get("source_files", {}), **plan.get("archive_files", {})}
    pinned[plan_path.relative_to(ROOT).as_posix()] = digest
    return {name: _sha_file(ROOT / name) for name in sorted(pinned)}


def load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha_file(path) != digest:
        raise ValueError("future-observation plan identity mismatch")
    if _sha_file(PARENT_PLAN) != PARENT_PLAN_SHA:
        raise ValueError("continuous parent plan digest mismatch")
    parent = json.loads(PARENT_PLAN.read_text())
    plan = json.loads(path.read_text())
    expected_sources = set(parent["source_files"]) | NEW_SOURCES
    expected_archives = set(parent["archive_files"]) | {
        receipt.relative_to(ROOT).as_posix() for receipt in RECEIPTS}
    if (plan.get("experiment_kind") != "paired_new_transport_terminal_E_observation"
            or plan.get("numerical_version") != NUMERICAL_VERSION
            or plan.get("transport_sha256") != TRANSPORT_SHA
            or plan.get("fixed_case_sha256") != CASE_SHA
            or plan.get("start_control_sha256") != START_CONTROL_SHA
            or plan.get("final_control_sha256") != FINAL_CONTROL_SHA
            or plan.get("parameters_sha256") != PARAMETERS_SHA
            or plan.get("truth_sha256") != TRUTH_SHA
            or plan.get("problem_identity_sha256") != PROBLEM_SHA
            or plan.get("score_profile") != SCORE_PROFILE
            or plan.get("policy") != POLICY
            or set(plan.get("source_files", {})) != expected_sources
            or set(plan.get("archive_files", {})) != expected_archives
            or plan["archive_files"].get(CASE.relative_to(ROOT).as_posix()) != CASE_SHA
            or plan["archive_files"].get(CASE_MANIFEST.relative_to(ROOT).as_posix()) != CASE_MANIFEST_SHA):
        raise ValueError("future-observation plan changed numerical scope, inputs, or policy")
    for name, digest_value in parent["source_files"].items():
        if plan["source_files"].get(name) != digest_value:
            raise ValueError(f"future-observation inherited source changed: {name}")
    for name, digest_value in parent["archive_files"].items():
        if plan["archive_files"].get(name) != digest_value:
            raise ValueError(f"future-observation inherited archive changed: {name}")
    for name, digest_value in {**plan["source_files"], **plan["archive_files"]}.items():
        if _sha_file(ROOT / name) != digest_value:
            raise ValueError(f"future-observation pin mismatch: {name}")
    if _sha_file(ROOT / TRANSPORT) != TRANSPORT_SHA:
        raise ValueError("live transport is not the pinned new net-flux evaluator")
    return plan


def _endpoint_control(endpoint: dict[str, Any]) -> Tensor:
    control = endpoint.get("control")
    if control is None:
        raw = endpoint.get("raw")
        if isinstance(raw, dict):
            control = raw.get("current_control")
    tensor = torch.as_tensor(control, dtype=torch.float64)
    if (tensor.shape != (26,) or tensor.device.type != "cpu"
            or tensor.dtype != torch.float64 or not bool(torch.isfinite(tensor).all())):
        raise ValueError("admitted endpoint control must be finite CPU FP64 length 26")
    return tensor


def _endpoint_input(endpoint: dict[str, Any]) -> dict[str, Any]:
    value = endpoint.get("input_after")
    if not isinstance(value, dict) and isinstance(endpoint.get("raw"), dict):
        value = endpoint["raw"].get("input_after")
    if not isinstance(value, dict):
        raise ValueError("endpoint receipt is missing its closed fixed-input identity")
    return value


def observe_score_pair(problem: Any, parameters: Tensor, truth: Tensor,
        start_control: Tensor, final_control: Tensor, *,
        check: Callable[[str], None] | None = None,
        progress: Callable[[str, str, float | None], None] | None = None) -> dict[str, Any]:
    """Call the production score exactly twice with one fixed parameter/truth pair."""
    tensors = (parameters, truth, start_control, final_control)
    if (any(not isinstance(value, Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float64 or not bool(torch.isfinite(value).all())
            for value in tensors)
            or parameters.shape != (13,) or truth.shape != (4, 5)
            or start_control.shape != (26,) or final_control.shape != (26,)):
        raise ValueError("paired terminal score requires the fixed CPU FP64 net-flux profile")
    if not torch.equal(problem.verification_dbz, truth):
        raise ValueError("fixed-case terminal truth differs from the score's verification field")
    _score_profile(problem)

    initial = tuple(_tensor_sha(value) for value in tensors)
    verification_sha = _tensor_sha(problem.verification_dbz)
    values: list[float] = []
    for label, control in (("start", start_control), ("final", final_control)):
        if check is not None:
            check(f"before_{label}_score")
        if progress is not None:
            progress(label, "started", None)
        with torch.no_grad():
            score = problem.score(control, parameters)
        if not isinstance(score, Tensor) or score.ndim != 0 or not bool(torch.isfinite(score)):
            raise ValueError(f"{label} terminal score must be a finite scalar tensor")
        value = float(score)
        if value < 0.0:
            raise ValueError(f"{label} terminal mean-square score cannot be negative")
        values.append(value)
        if progress is not None:
            progress(label, "completed", value)
        if check is not None:
            check(f"after_{label}_score")
    if (initial != tuple(_tensor_sha(value) for value in tensors)
            or verification_sha != _tensor_sha(problem.verification_dbz)
            or not torch.equal(problem.verification_dbz, truth)):
        raise ValueError("score evaluation mutated a fixed input or endpoint control")
    return {"start_E": values[0], "final_E": values[1],
        "delta_E_final_minus_start": values[1] - values[0],
        "score_evaluations": 2, "selection_use": False,
        "forecast_steps": 18, "forecast_time_seconds_from_start": 12000.0,
        "forecast_scope": "last frame at +180 minutes after the +20 minute analysis window"}


def _load_endpoint(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    return module.load_endpoint()


def _load_case() -> dict[str, Any]:
    if _sha_file(CASE) != CASE_SHA:
        raise ValueError("fixed old-truth case asset digest changed")
    return torch.load(CASE, map_location="cpu", weights_only=False)


def _fixed_case_tensors(case: dict[str, Any], plan: dict[str, Any]
        ) -> tuple[Any, Tensor, Tensor, Tensor]:
    problem, original = case.get("problem"), case.get("original")
    parameters, truth = case.get("parameters"), case.get("truth")
    identity = getattr(problem, "identity", None)
    if (problem is None or not isinstance(original, Tensor)
            or not isinstance(parameters, Tensor) or not isinstance(truth, Tensor)
            or not isinstance(identity, dict)
            or identity.get("fixed_problem_sha256") != PROBLEM_SHA
            or _tensor_sha(parameters) != PARAMETERS_SHA
            or _tensor_sha(truth) != TRUTH_SHA
            or plan.get("parameters_sha256") != PARAMETERS_SHA
            or plan.get("truth_sha256") != TRUTH_SHA
            or not torch.equal(problem.verification_dbz, truth)):
        raise ValueError("fixed parameters/truth/problem differ from the frozen case identity")
    _score_profile(problem)
    return problem, original, parameters, truth


def _run_summary(resource: dict[str, Any], child_path: Path, digest: str) -> dict[str, Any]:
    from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle
    shared = cycle.shared
    status = shared._execution_status(resource, wall_seconds=float(POLICY["outer_seconds"]),
        rss_bytes=int(POLICY["rss_bytes"]))
    result: dict[str, Any] = {"execution_status": status, "resource": resource,
        "child_sha256": None, "child_receipt_status": "missing"}
    if not child_path.is_file() or child_path.is_symlink():
        result["execution_status"] = "failed"
        result["child_receipt_error"] = "child receipt is missing or not a regular file"
        return result
    raw_bytes = child_path.read_bytes()
    result["child_sha256"] = _sha_bytes(raw_bytes)
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        result.update(execution_status="failed", child_receipt_status="invalid",
            child_receipt_error=f"{type(error).__name__}: {error}")
        return result
    pair = raw.get("score_pair") if isinstance(raw, dict) else None
    values = (pair.get("start_E"), pair.get("final_E")) if isinstance(pair, dict) else ()
    scores_finite_nonnegative = (len(values) == 2 and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value)) and float(value) >= 0.0 for value in values))
    closure_ok = (isinstance(raw, dict)
        and raw.get("phase") == "finished"
        and raw.get("execution_status") == "completed"
        and raw.get("numerical_status") == "paired_terminal_score_observed"
        and raw.get("plan_sha256") == digest
        and raw.get("numerical_version") == NUMERICAL_VERSION
        and raw.get("transport_sha256") == TRANSPORT_SHA
        and raw.get("score_evaluations_started") == 2
        and raw.get("score_evaluations_completed") == 2
        and isinstance(pair, dict) and pair.get("score_evaluations") == 2
        and pair.get("selection_use") is False and scores_finite_nonnegative
        and raw.get("optimizer_steps") == 0 and raw.get("candidate_evaluations") == 0
        and raw.get("selection_use") is False
        and raw.get("root_claim") is False and raw.get("minimum_claim") is False
        and raw.get("response_claim") is False and raw.get("score_claim") is False
        and raw.get("source_unchanged") is True
        and raw.get("source_before") == raw.get("source_after")
        and raw.get("input_unchanged") is True
        and raw.get("input_before") == raw.get("input_after")
        and raw.get("runtime_unchanged") is True
        and raw.get("runtime_before") == raw.get("runtime_after")
        and raw.get("fixed_case_unchanged") is True
        and raw.get("fixed_case_sha256") == CASE_SHA)
    if status != "completed" or not closure_ok:
        result.update(execution_status="failed", child_receipt_status="invalid",
            child_receipt_error="resource guard or child observation closure did not pass")
    else:
        result["child_receipt_status"] = "closed"
    return result


def _run_child(plan_path: Path, digest: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    plan = load_plan(plan_path, digest)
    from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle
    shared = cycle.shared
    deadline = started + float(POLICY["internal_seconds"])
    source_before = _source_hashes(plan, plan_path, digest)
    manifest_before = _sha_file(CASE_MANIFEST)
    start_endpoint = _load_endpoint(START_ENDPOINT_MODULE)
    final_endpoint = _load_endpoint(FINAL_ENDPOINT_MODULE)
    start_control = _endpoint_control(start_endpoint)
    final_control = _endpoint_control(final_endpoint)
    if (_tensor_sha(start_control) != START_CONTROL_SHA
            or _tensor_sha(final_control) != plan.get("final_control_sha256")
            or torch.equal(start_control, final_control)):
        raise ValueError("paired endpoint receipt controls do not match the frozen plan")
    case = _load_case()
    problem, original, parameters, truth = _fixed_case_tensors(case, plan)
    params_sha, truth_sha = _tensor_sha(parameters), _tensor_sha(truth)
    if params_sha != PARAMETERS_SHA or truth_sha != TRUTH_SHA:
        raise ValueError("fixed parameters/truth changed after case admission")
    runtime_before = shared._runtime()
    manifest = json.loads(CASE_MANIFEST.read_text())
    if (runtime_before != manifest.get("runtime_identity")
            or runtime_before != start_endpoint.get("runtime_after")
            or runtime_before != final_endpoint.get("runtime_after")):
        raise ValueError("fixed-case runtime differs from the captured old-kernel runtime")
    start_input = shared._input_identity(problem, original, start_control, parameters, truth)
    final_input = shared._input_identity(problem, original, final_control, parameters, truth)
    for endpoint in (start_endpoint, final_endpoint):
        identity = _endpoint_input(endpoint)
        if (identity.get("parameters_sha256") != params_sha
                or identity.get("terminal_truth_sha256") != truth_sha
                or identity.get("archived_input") != start_input.get("archived_input")):
            raise ValueError("endpoint receipts do not share the fixed parameter/truth pair")
    if start_input != _endpoint_input(start_endpoint) or final_input != _endpoint_input(final_endpoint):
        raise ValueError("fixed problem/controls do not match the independently admitted endpoint inputs")
    input_before = {"start": start_input, "final": final_input}
    result: dict[str, Any] = {"phase": "observation_started", "execution_status": "running",
        "numerical_status": "paired_terminal_score_observation_in_progress",
        "plan_sha256": digest, "numerical_version": NUMERICAL_VERSION,
        "transport_sha256": TRANSPORT_SHA, "input_before": input_before,
        "runtime_before": runtime_before, "source_before": source_before,
        "fixed_case_sha256": CASE_SHA, "score_evaluations_started": 0,
        "score_evaluations_completed": 0, "score_pair_partial": {},
        "optimizer_steps": 0, "candidate_evaluations": 0,
        "selection_use": False, "root_claim": False, "minimum_claim": False,
        "response_claim": False, "score_claim": False}
    from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
    tangent._write(output, result)

    def check(stage: str) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"internal 600-second observation deadline reached at {stage}")
        if (_source_hashes(plan, plan_path, digest) != source_before
                or _sha_file(CASE_MANIFEST) != manifest_before or _sha_file(CASE) != CASE_SHA
                or shared._runtime() != runtime_before
                or shared._input_identity(problem, original, start_control, parameters, truth) != start_input
                or shared._input_identity(problem, original, final_control, parameters, truth) != final_input):
            raise ValueError(f"fixed source/input/runtime identity changed at {stage}")

    def progress(label: str, status: str, score: float | None) -> None:
        if status == "started":
            result["score_evaluations_started"] += 1
        else:
            result["score_evaluations_completed"] += 1
            result["score_pair_partial"][f"{label}_E"] = score
        result["last_score_stage"] = f"{label}_{status}"
        tangent._write(output, result)

    try:
        scores = observe_score_pair(problem, parameters, truth, start_control,
            final_control, check=check, progress=progress)
        check("after_score_pair")
    except TimeoutError as error:
        result.update(phase="budget_exhausted", execution_status="failed",
            numerical_status="paired_terminal_score_budget_exhausted", failure=str(error),
            elapsed_seconds=time.monotonic() - started)
        tangent._write(output, result)
        raise
    except Exception as error:
        result.update(phase="failed", execution_status="failed",
            numerical_status="paired_terminal_score_observation_failed",
            failure=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        tangent._write(output, result)
        raise
    try:
        input_after_start = shared._input_identity(problem, original, start_control, parameters, truth)
        input_after_final = shared._input_identity(problem, original, final_control, parameters, truth)
        source_after = _source_hashes(plan, plan_path, digest)
        manifest_after, case_after = _sha_file(CASE_MANIFEST), _sha_file(CASE)
        if (input_after_start != start_input or input_after_final != final_input
                or source_after != source_before or manifest_after != manifest_before
                or case_after != CASE_SHA or shared._runtime() != runtime_before):
            raise ValueError("source, fixed-case asset, input, or runtime changed during paired score observation")
    except Exception as error:
        result.update(phase="failed", execution_status="failed",
            numerical_status="paired_terminal_score_closure_failed",
            failure=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        tangent._write(output, result)
        raise
    result.update({"phase": "finished", "execution_status": "completed",
        "numerical_status": "paired_terminal_score_observed",
        "input_after": {"start": input_after_start, "final": input_after_final},
        "runtime_after": runtime_before, "source_after": source_after,
        "source_unchanged": True,
        "fixed_case_sha256": case_after, "fixed_case_unchanged": True,
        "score_pair": scores, "runtime_unchanged": True,
        "input_unchanged": True, "optimizer_steps": 0, "candidate_evaluations": 0,
        "root_claim": False, "minimum_claim": False, "response_claim": False,
        "score_claim": False, "selection_use": False,
        "elapsed_seconds": time.monotonic() - started})
    tangent._write(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--child", action="store_true")
    arguments = parser.parse_args()
    if arguments.child:
        _run_child(PLAN, arguments.plan_sha256, ATTEMPT / "score_pair.json")
        return
    plan = load_plan(PLAN, arguments.plan_sha256)
    if ATTEMPT.exists():
        raise ValueError("future-observation attempt directory must be fresh")
    ATTEMPT.mkdir(parents=True)
    from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic
    resource = run_guarded_diagnostic(
        [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
         "--plan-sha256", arguments.plan_sha256],
        wall_seconds=float(POLICY["outer_seconds"]), rss_bytes=int(POLICY["rss_bytes"]),
        report_path=ATTEMPT / "score_pair.resource.json",
        log_path=ATTEMPT / "score_pair.log")
    run = _run_summary(resource, ATTEMPT / "score_pair.json", arguments.plan_sha256)
    (ATTEMPT / "score_pair.run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(run, sort_keys=True))
    if run["execution_status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
