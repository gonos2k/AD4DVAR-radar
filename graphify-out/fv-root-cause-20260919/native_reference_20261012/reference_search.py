"""Two-start bounded native-J Powell reference search; no E/Q/R/HVP calls."""
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

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
PLAN = WORK / "PLAN.json"
PLAN_SHA_FILE = WORK / "PLAN_SHA256.txt"
ATTEMPT = WORK / "attempt1"
SELF = "graphify-out/fv-root-cause-20260919/native_reference_20261012/reference_search.py"
SIGNED = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012"
SIGNED_PLAN = SIGNED / "PLAN.json"
SIGNED_PLAN_SHA = "707bf06bf0432a3f46b6fcc8901908186d70336e075ea5dc72b3cf4658bcfe9b"
SIGNED_STEP = SIGNED / "attempt1/step.json"
FIXED_CASE = ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"
FIXED_CASE_SHA = "1e810c23c73b93d89b2a10b46d9ddea44a8217b07dcd3ebfebb462858e3fa9b7"
EXPECTED_FB0A_SHA = "fb0a2efaeefe3a8b07a9385a81260757e4923b181a9430c384cfb96b8d8ea7b4"
EXPECTED_FB0A_J = 0.060988878373729814
EXPECTED_PARAMETERS_SHA = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
EXPECTED_TRUTH_SHA = "b927c1a39d16f05285cccb7a9ff48d182eea539bfba746f7aff446aee0974610"
EXPECTED_PROBLEM_SHA = "16fa95b534c3b61cf5d1506770cbfc77edb527c40d1f4c536f5130cba24e1fc3"
MAXFEV_PER_START = 1000
MAXITER_PER_START = 100
SEARCH_SECONDS = 3000.0
TOTAL_SECONDS = 3600.0
OUTER_SECONDS = 3600.0
RSS_BYTES = 2 * 1024**3
STARTS = ("fb0a", "zero")

sys.path.insert(0, str(ROOT))
from examples.weather_scenarios import fv_point_3h_netflux_future_observation as score_observer  # noqa: E402
from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle  # noqa: E402
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed  # noqa: E402


class SearchDeadline(Exception):
    """The frozen native-J wall budget has been reached."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned file is missing, unsafe, or outside the repository: {path}")
    return _sha_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _tensor_sha(value: torch.Tensor) -> str:
    if value.dtype != torch.float64 or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
        raise ValueError("native reference tensors must be finite CPU FP64")
    return _sha_bytes(value.detach().contiguous().numpy().tobytes())


def _load_pinned_inputs() -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor,
        dict[str, torch.Tensor], dict[str, str]]:
    if _sha(SIGNED_PLAN) != SIGNED_PLAN_SHA:
        raise ValueError("signed baseline plan hash mismatch")
    signed_plan = _read_json(SIGNED_PLAN)
    step = _read_json(SIGNED_STEP)
    if (step.get("base_control_sha256") != "2a4dba130d28273a63e97d0de39a11f64fdee09b0ad754559bc8902ba6cddbca"
            or step.get("candidate_committed") is not True
            or step.get("current_control_sha256") != "96108bdcf2cd6abc05878bf2cfc9e2531f37cf82e152ad1a9eaf0a67de746411"):
        raise ValueError("signed-span input is not the final committed 2a4d trial")
    arms = {arm["name"]: arm for arm in step["arms"]}
    fb_receipt_path = ROOT / arms["baseline_tangent"]["trials"][0]["candidate_receipt"]
    fb_receipt = _read_json(fb_receipt_path)
    fb0a = torch.as_tensor(fb_receipt["control"], dtype=torch.float64)
    zero = torch.zeros_like(fb0a)
    if (_tensor_sha(fb0a) != EXPECTED_FB0A_SHA
            or fb_receipt.get("control_sha256") != EXPECTED_FB0A_SHA
            or not bool(fb_receipt.get("accepted"))):
        raise ValueError("fb0a start does not match the sealed independent gradient candidate")
    points = {"fb0a": fb0a, "zero": zero}
    case = score_observer._load_case()
    problem, original, parameters, truth = score_observer._fixed_case_tensors(case, {
        "parameters_sha256": EXPECTED_PARAMETERS_SHA,
        "truth_sha256": EXPECTED_TRUTH_SHA})
    if problem.identity.get("fixed_problem_sha256") != EXPECTED_PROBLEM_SHA:
        raise ValueError("fixed native-J problem identity mismatch")
    contract = problem.contract(parameters)
    from advar.variational import _control_prior_residual
    for name, point in points.items():
        if not torch.equal(_control_prior_residual(point, contract), point):
            raise ValueError(f"identity prior contract is absent for {name}")
    identity = {name: cycle.shared._input_identity(problem, original, point, parameters, truth)
        for name, point in points.items()}
    hashes = {
        str(SIGNED_PLAN.relative_to(ROOT)): _sha(SIGNED_PLAN),
        str((SIGNED / "PLAN_SHA256.txt").relative_to(ROOT)): _sha(SIGNED / "PLAN_SHA256.txt"),
        str((SIGNED / "source.tar.gz").relative_to(ROOT)): _sha(SIGNED / "source.tar.gz"),
        str(SIGNED_STEP.relative_to(ROOT)): _sha(SIGNED_STEP),
        str((SIGNED / "attempt1/step.run.json").relative_to(ROOT)): _sha(SIGNED / "attempt1/step.run.json"),
        str((SIGNED / "attempt1/step.resource.json").relative_to(ROOT)): _sha(SIGNED / "attempt1/step.resource.json"),
        fb_receipt_path.relative_to(ROOT).as_posix(): _sha(fb_receipt_path),
        str(FIXED_CASE.relative_to(ROOT)): _sha(FIXED_CASE),
        str(FIXED_CASE.with_name("FIXED_CASE_MANIFEST.json").relative_to(ROOT)):
            _sha(FIXED_CASE.with_name("FIXED_CASE_MANIFEST.json")),
    }
    return problem, original, parameters, truth, points, {
        "input_identities": identity,
        "input_receipt_hashes": hashes,
        "source_files": signed_plan["source_files"],
        "archive_files": signed_plan["archive_files"],
        "parameters_sha256": _tensor_sha(parameters),
        "truth_sha256": _tensor_sha(truth),
        "problem_identity": problem.identity,
        "runtime_identity": {"python": platform.python_version(),
            "torch": torch.__version__, "device": "CPU FP64"},
    }


def freeze_plan() -> dict[str, Any]:
    if PLAN.exists() or PLAN_SHA_FILE.exists():
        raise ValueError("native-reference plan already frozen")
    problem, original, parameters, truth, points, pins = _load_pinned_inputs()
    parent_plan = _read_json(SIGNED_PLAN)
    import tarfile
    for name, digest in parent_plan["source_files"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"native-J inherited source pin changed: {name}")
    for name, digest in parent_plan["archive_files"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"native-J inherited archive pin changed: {name}")
    with tarfile.open(SIGNED / "source.tar.gz", "r:gz") as source_tar:
        archived = {member.name: source_tar.extractfile(member).read()
            for member in source_tar.getmembers() if member.isfile()}
    if set(archived) != set(parent_plan["source_files"]):
        raise ValueError("signed source bundle member set mismatch")
    for name, digest in parent_plan["source_files"].items():
        if _sha_bytes(archived[name]) != digest:
            raise ValueError(f"signed source bundle hash mismatch: {name}")
    archive_files = {**parent_plan["archive_files"], **pins["input_receipt_hashes"]}
    plan = {
        "experiment_kind": "two_start_bounded_native_J_powell_reference_search",
        "source_files": parent_plan["source_files"],
        "archive_files": {k: archive_files[k] for k in sorted(archive_files)},
        "source_bundle_sha256": _sha(SIGNED / "source.tar.gz"),
        "observer_source_sha256": _sha(ROOT / SELF),
        "parent_signed_span_plan_sha256": SIGNED_PLAN_SHA,
        "parent_signed_source_count": len(parent_plan["source_files"]),
        "parent_signed_archive_count": len(parent_plan["archive_files"]),
        "points": {name: _tensor_sha(point) for name, point in points.items()},
        "input_identities": pins["input_identities"],
        "parameters_sha256": pins["parameters_sha256"],
        "terminal_truth_sha256": pins["truth_sha256"],
        "fixed_problem_identity": pins["problem_identity"],
        "runtime_identity": pins["runtime_identity"],
        "native_objective": "exact problem.objective(control,parameters), evaluated under torch.no_grad; no surrogate and no exception-to-penalty conversion",
        "identity_prior": "verified 0.5*||control||²; each start's component box is [-sqrt(2*Jstart),+sqrt(2*Jstart)]",
        "starts": [
            {"name": "fb0a", "expected_control_sha256": EXPECTED_FB0A_SHA,
                "expected_start_J": EXPECTED_FB0A_J},
            {"name": "zero", "expected_control_sha256": _tensor_sha(points["zero"]),
                "expected_start_J": None},
        ],
        "optimizer": {"method": "Powell", "maxfev_per_start": MAXFEV_PER_START,
            "maxiter_per_start": MAXITER_PER_START,
            "xtol": 1e-11, "ftol": 1e-13, "bounds": "componentwise start-centered identity-prior box"},
        "budget": {"total_seconds": TOTAL_SECONDS, "search_internal_seconds": SEARCH_SECONDS,
            "search_outer_seconds": OUTER_SECONDS, "reserved_parent_qualification_seconds": 600.0,
            "rss_bytes": RSS_BYTES, "max_objective_evaluations": 2000,
            "resume_rule": "same plan only; cumulative per-start function-evaluation and global-time counters are never reset"},
        "scope": {"Q": False, "R": False, "E": False, "HVP": False,
            "E_based_candidate_selection": False,
            "output": "report each start's Powell result independently; no cross-start winner selection"},
    }
    PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    digest = _sha(PLAN)
    PLAN_SHA_FILE.write_text(digest + "\n")
    return {"plan_sha256": digest, "source_count": len(plan["source_files"]),
        "archive_count": len(plan["archive_files"]), "observer_sha256": plan["observer_source_sha256"],
        "runtime": plan["runtime_identity"], "points": plan["points"]}


def load_plan(digest: str) -> dict[str, Any]:
    if _sha(PLAN) != digest or PLAN_SHA_FILE.read_text().strip() != digest:
        raise ValueError("native-reference plan seal mismatch")
    plan = _read_json(PLAN)
    parent = _read_json(SIGNED_PLAN)
    if (plan.get("experiment_kind") != "two_start_bounded_native_J_powell_reference_search"
            or plan.get("observer_source_sha256") != _sha(ROOT / SELF)
            or plan.get("source_files") != parent.get("source_files")
            or plan.get("parent_signed_span_plan_sha256") != SIGNED_PLAN_SHA
            or plan.get("parent_signed_source_count") != 237
            or plan.get("parent_signed_archive_count") != 306
            or plan.get("budget", {}).get("max_objective_evaluations") != 2000
            or plan.get("scope", {}).get("E") is not False):
        raise ValueError("native-reference source or policy scope changed")
    for group in ("source_files", "archive_files"):
        for name, expected in plan[group].items():
            if _sha(ROOT / name) != expected:
                raise ValueError(f"native-reference {group[:-6]} pin mismatch: {name}")
    return plan


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def _run_search(plan_sha: str) -> dict[str, Any]:
    from scipy.optimize import minimize

    plan = load_plan(plan_sha)
    problem, original, parameters, truth, points, pins = _load_pinned_inputs()
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"}
    if runtime != plan["runtime_identity"]:
        raise ValueError(f"optimizer runtime differs from frozen plan: {runtime}")
    if ATTEMPT.exists() and (ATTEMPT / "checkpoint.json").is_file():
        checkpoint_path = ATTEMPT / "checkpoint.json"
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get("plan_sha256") != plan_sha:
            raise ValueError("cannot resume under a different plan")
    else:
        ATTEMPT.mkdir(parents=True, exist_ok=True)
        checkpoint = {"plan_sha256": plan_sha, "status": "running",
            "started_monotonic": time.monotonic(), "cumulative_elapsed_seconds": 0.0,
            "starts": {name: {"status": "pending", "objective_evaluations": 0,
                "powell_function_calls": 0, "powell_iterations": 0,
                "best_J": None, "best_control": None} for name in STARTS},
            "constraints": {"q_calls": 0, "r_calls": 0, "e_calls": 0, "hvp_calls": 0,
                "selection_use": False}}
        _write_json(ATTEMPT / "checkpoint.json", checkpoint)
    start_clock = time.monotonic()
    total_elapsed_before = float(checkpoint.get("cumulative_elapsed_seconds", 0.0))
    global_deadline = start_clock + max(0.0, SEARCH_SECONDS - total_elapsed_before)
    jsonl = ATTEMPT / "evaluations.jsonl"

    from advar.variational import _control_prior_residual
    contract = problem.contract(parameters)
    if any(not torch.equal(_control_prior_residual(point, contract), point) for point in points.values()):
        raise ValueError("identity prior changed after plan freeze")

    for start_name in STARTS:
        state = checkpoint["starts"][start_name]
        if state.get("status") == "completed":
            continue
        if time.monotonic() >= global_deadline:
            break
        original_start = points[start_name]
        if state.get("best_control") is None:
            x0 = original_start.clone()
            tic = time.monotonic()
            with torch.no_grad():
                f0_tensor = problem.objective(x0, parameters)
            if (not isinstance(f0_tensor, torch.Tensor) or f0_tensor.ndim != 0
                    or not bool(torch.isfinite(f0_tensor))):
                raise ValueError(f"native J at {start_name} start is nonfinite or malformed")
            f0 = float(f0_tensor)
            elapsed = time.monotonic() - tic
            state["objective_evaluations"] += 1
            state["initial_J"] = f0
            if start_name == "fb0a":
                tolerance = 128 * torch.finfo(torch.float64).eps * max(
                    abs(f0), abs(EXPECTED_FB0A_J), torch.finfo(torch.float64).tiny)
                if abs(f0 - EXPECTED_FB0A_J) > tolerance:
                    raise ValueError(f"native fb0a J {f0:.17g} differs from sealed .060988878373729814")
            box = math.sqrt(2.0 * f0)
            if not math.isfinite(box) or box <= 0 or bool((x0.abs() > box).any()):
                raise ValueError(f"identity-prior box does not contain {start_name} start")
            state["box_radius"] = box
            state["best_J"] = f0
            state["best_control"] = x0.tolist()
            initial_record = {"kind": "initial_native_J", "objective_evaluation": 1,
                "J": f0, "control": x0.tolist(), "elapsed_seconds": elapsed,
                "control_sha256": _tensor_sha(x0)}
            with jsonl.open("a") as handle:
                handle.write(json.dumps({"start": start_name, **initial_record}, allow_nan=False) + "\n")
            _write_json(ATTEMPT / "checkpoint.json", checkpoint)
            _write_json(ATTEMPT / f"{start_name}.candidate_checkpoint.json", {
                "plan_sha256": plan_sha, "start": start_name,
                "status": "running", "objective_evaluations": state["objective_evaluations"],
                "powell_function_calls": state["powell_function_calls"],
                "best_J": state["best_J"], "best_control": state["best_control"],
                "box_radius": box})
        x0 = torch.as_tensor(state["best_control"], dtype=torch.float64)
        f0 = float(state["best_J"])
        box = float(state["box_radius"])
        cached_x = x0.numpy().copy()
        cached_f = f0
        cache_once = True
        options_maxfev = max(0, MAXFEV_PER_START - int(state["powell_function_calls"]))
        options_maxiter = max(0, MAXITER_PER_START - int(state["powell_iterations"]))
        if options_maxfev == 0 or options_maxiter == 0:
            state["status"] = "evaluation_or_iteration_cap_reached"
            state["message"] = "cumulative per-start SciPy limit reached"
            continue

        def objective(vector: Any) -> float:
            nonlocal cache_once
            if time.monotonic() >= global_deadline:
                raise SearchDeadline("cumulative 3000-second search budget reached")
            state["powell_function_calls"] += 1
            array = torch.as_tensor(vector, dtype=torch.float64)
            if cache_once and np_equal(vector, cached_x):
                cache_once = False
                value, elapsed, kind = cached_f, 0.0, "powell_start_cache"
            else:
                if int(state["objective_evaluations"]) >= MAXFEV_PER_START:
                    raise SearchDeadline(f"{start_name} 1000 objective-evaluation cap reached")
                tic = time.monotonic()
                with torch.no_grad():
                    result = problem.objective(array, parameters)
                if (not isinstance(result, torch.Tensor) or result.ndim != 0
                        or not bool(torch.isfinite(result))):
                    raise ValueError(f"native J at {start_name} evaluation is nonfinite or malformed")
                value = float(result)
                elapsed, kind = time.monotonic() - tic, "native_objective"
                state["objective_evaluations"] += 1
            record = {"kind": kind, "objective_evaluation": int(state["objective_evaluations"]),
                "powell_function_call": int(state["powell_function_calls"]),
                "J": value, "control": array.tolist(), "control_sha256": _tensor_sha(array),
                "evaluation_seconds": elapsed,
                "cumulative_elapsed_seconds": total_elapsed_before + time.monotonic() - start_clock}
            if state["best_J"] is None or value < float(state["best_J"]):
                state["best_J"], state["best_control"] = value, array.tolist()
            checkpoint["status"] = "running"
            checkpoint["cumulative_elapsed_seconds"] = record["cumulative_elapsed_seconds"]
            with jsonl.open("a") as handle:
                handle.write(json.dumps({"start": start_name, **record}, allow_nan=False) + "\n")
            _write_json(ATTEMPT / "checkpoint.json", checkpoint)
            _write_json(ATTEMPT / f"{start_name}.candidate_checkpoint.json", {
                "plan_sha256": plan_sha, "start": start_name,
                "status": "running", "objective_evaluations": state["objective_evaluations"],
                "powell_function_calls": state["powell_function_calls"],
                "best_J": state["best_J"], "best_control": state["best_control"],
                "box_radius": box})
            return value

        try:
            result = minimize(objective, x0.numpy(), method="Powell",
                bounds=[(-box, box)] * int(x0.numel()),
                options={"maxfev": options_maxfev, "maxiter": options_maxiter,
                    "xtol": 1e-11, "ftol": 1e-13, "disp": False})
        except SearchDeadline as error:
            state["status"] = "budget_stopped_resumable"
            state["message"] = str(error)
            checkpoint["cumulative_elapsed_seconds"] = total_elapsed_before + time.monotonic() - start_clock
            _write_json(ATTEMPT / "checkpoint.json", checkpoint)
            break
        state["powell_iterations"] += int(result.nit)
        state["scipy_nfev_this_launch"] = int(result.nfev)
        state["scipy_nit_this_launch"] = int(result.nit)
        state["scipy_success"] = bool(result.success)
        state["scipy_message"] = str(result.message)
        candidate = torch.as_tensor(result.x, dtype=torch.float64)
        # Final `result.fun` is from the same counted callback; no extra J call.
        state["scipy_result_J"] = float(result.fun)
        state["scipy_result_control"] = candidate.tolist()
        state["scipy_result_control_sha256"] = _tensor_sha(candidate)
        if float(result.fun) < float(state["best_J"]):
            state["best_J"], state["best_control"] = float(result.fun), candidate.tolist()
        state["status"] = "completed" if result.success else "stopped_by_powell_limit_or_tolerance"
        state["box_bounds"] = [-box, box]
        state["powell_options"] = {"maxfev_this_launch": options_maxfev,
            "maxiter_this_launch": options_maxiter,"xtol": 1e-11,"ftol": 1e-13}
        checkpoint["cumulative_elapsed_seconds"] = total_elapsed_before + time.monotonic() - start_clock
        _write_json(ATTEMPT / "checkpoint.json", checkpoint)

    all_done = all(checkpoint["starts"][name].get("status") not in ("pending", "running", "budget_stopped_resumable")
        for name in STARTS)
    checkpoint["status"] = "completed" if all_done else "budget_stopped_resumable"
    checkpoint["cumulative_elapsed_seconds"] = total_elapsed_before + time.monotonic() - start_clock
    checkpoint["constraints"] = {"q_calls": 0, "r_calls": 0, "e_calls": 0,
        "hvp_calls": 0, "selection_use": False}
    _write_json(ATTEMPT / "checkpoint.json", checkpoint)
    summary = {"plan_sha256": plan_sha, "status": checkpoint["status"],
        "cumulative_elapsed_seconds": checkpoint["cumulative_elapsed_seconds"],
        "starts": {name: checkpoint["starts"][name] for name in STARTS},
        "objective_evaluations_total": sum(int(checkpoint["starts"][n]["objective_evaluations"]) for n in STARTS),
        "powell_function_calls_total": sum(int(checkpoint["starts"][n]["powell_function_calls"]) for n in STARTS),
        "q_calls": 0, "r_calls": 0, "e_calls": 0, "hvp_calls": 0,
        "candidate_selection_by_E": False}
    _write_json(ATTEMPT / "summary.json", summary)
    return summary


def np_equal(a: Any, b: Any) -> bool:
    import numpy as np
    return np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze_plan(), sort_keys=True))
        return
    if not args.plan_sha256:
        raise SystemExit("--plan-sha256 is required")
    load_plan(args.plan_sha256)
    if args.child:
        result = _run_search(args.plan_sha256)
        print(json.dumps(result, sort_keys=True))
        return
    from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    resource = run_guarded_diagnostic(
        [sys.executable, str(Path(__file__).resolve()), "--child", "--plan-sha256", args.plan_sha256],
        wall_seconds=OUTER_SECONDS, rss_bytes=RSS_BYTES,
        report_path=ATTEMPT / "resource.json", log_path=ATTEMPT / "search.log")
    run = {"resource": resource,
        "execution_status": "completed" if resource.get("exit_code") == 0 else "failed",
        "plan_sha256": args.plan_sha256,
        "checkpoint_sha256": _sha(ATTEMPT / "checkpoint.json") if (ATTEMPT / "checkpoint.json").is_file() else None}
    _write_json(ATTEMPT / "run.json", run)
    print(json.dumps(run, sort_keys=True))
    if run["execution_status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
