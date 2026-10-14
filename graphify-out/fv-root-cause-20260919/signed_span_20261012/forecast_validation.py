"""Observe the fixed terminal score at three independently closed controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
ATTEMPT = WORK / "forecast_validation_attempt1"
PLAN = WORK / "FORECAST_PLAN.json"
PLAN_SHA = WORK / "FORECAST_PLAN_SHA256.txt"
BASE_PLAN_PATH = WORK / "PLAN.json"
BASE_PLAN_SHA = "707bf06bf0432a3f46b6fcc8901908186d70336e075ea5dc72b3cf4658bcfe9b"
BASE_PLAN_SHA_FILE = WORK / "PLAN_SHA256.txt"
BASE_SOURCE_TAR = WORK / "source.tar.gz"
BASE_STEP = WORK / "attempt1/step.json"
BASE_RUN = WORK / "attempt1/step.run.json"
BASE_RESOURCE = WORK / "attempt1/step.resource.json"
OLD_ENDPOINT_PATH = ROOT / "graphify-out/fv-root-cause-20260919/residual_direction_20261012/endpoint.py"
FIXED_CASE = ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"
FIXED_CASE_MANIFEST = FIXED_CASE.with_name("FIXED_CASE_MANIFEST.json")
OLD_ENDPOINT_SHA = "2a4dba130d28273a63e97d0de39a11f64fdee09b0ad754559bc8902ba6cddbca"
BASELINE_SHA = "fb0a2efaeefe3a8b07a9385a81260757e4923b181a9430c384cfb96b8d8ea7b4"
SIGNED_SHA = "96108bdcf2cd6abc05878bf2cfc9e2531f37cf82e152ad1a9eaf0a67de746411"
PARAMETERS_SHA = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
TRUTH_SHA = "b927c1a39d16f05285cccb7a9ff48d182eea539bfba746f7aff446aee0974610"
INPUT_BUDGET = {"internal_seconds": 60.0, "outer_seconds": 90.0,
    "rss_bytes": 2 * 1024**3, "score_calls": 3, "optimizer_steps": 0,
    "candidate_evaluations": 0, "hvp_calls": 0, "selection_use": False}
POINTS = ("2a4d", "fb0a", "96108")

sys.path.insert(0, str(ROOT))
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed  # noqa: E402
from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle  # noqa: E402
from examples.weather_scenarios import fv_point_3h_netflux_future_observation as score_observer  # noqa: E402


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned path is missing, symlinked, or outside the repository: {path}")
    return _sha_bytes(path.read_bytes())


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hash_tensor(value: torch.Tensor) -> str:
    if value.device.type != "cpu" or value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
        raise ValueError("fixed forecast tensors must be finite CPU FP64")
    return _sha_bytes(value.detach().contiguous().numpy().tobytes())


def _sealed_points() -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, str]]:
    base_plan = _json(BASE_PLAN_PATH)
    if _sha(BASE_PLAN_PATH) != BASE_PLAN_SHA or BASE_PLAN_SHA_FILE.read_text().strip() != BASE_PLAN_SHA:
        raise ValueError("signed-span parent plan seal mismatch")
    step = _json(BASE_STEP)
    if _sha(BASE_STEP) != _json(BASE_RUN)["child_sha256"]:
        raise ValueError("signed-span child receipt digest differs from its run record")
    if (step.get("base_control_sha256") != OLD_ENDPOINT_SHA
            or step.get("current_control_sha256") != SIGNED_SHA
            or step.get("candidate_committed") is not True
            or step.get("optimizer_steps_applied") != 1
            or step.get("hvp_started") != 0 or step.get("hvp_reused") != 4):
        raise ValueError("signed-span run is not the expected closed 2a4d-to-96108 trial")
    arms = {entry["name"]: entry for entry in step["arms"]}
    base_endpoint = _load_old_endpoint()
    c0 = torch.as_tensor(base_endpoint["control"], dtype=torch.float64)
    baseline = torch.as_tensor(arms["baseline_tangent"]["trials"][0]["control"], dtype=torch.float64)
    signed = torch.as_tensor(arms["two_direction_subspace"]["trials"][0]["control"], dtype=torch.float64)
    if (tuple(_hash_tensor(c) for c in (c0, baseline, signed)) != (OLD_ENDPOINT_SHA, BASELINE_SHA, SIGNED_SHA)
            or not torch.equal(signed, torch.as_tensor(step["current_control"], dtype=torch.float64))
            or not torch.equal(signed, torch.as_tensor(step["closed_steps"][0]["final_repeat"]["control"], dtype=torch.float64))):
        raise ValueError("three forecast controls do not match the separately closed receipts")
    receipt_files = {
        BASE_PLAN_PATH, BASE_PLAN_SHA_FILE, BASE_SOURCE_TAR, BASE_STEP, BASE_RUN, BASE_RESOURCE,
        ROOT / arms["baseline_tangent"]["trials"][0]["candidate_receipt"],
        ROOT / arms["two_direction_subspace"]["trials"][0]["candidate_receipt"],
        FIXED_CASE, FIXED_CASE_MANIFEST,
    }
    return dict(zip(POINTS, (c0, baseline, signed))), base_endpoint, {
        path.relative_to(ROOT).as_posix(): _sha(path) for path in sorted(receipt_files)}


def _load_old_endpoint() -> dict[str, Any]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("forecast_validation_parent_endpoint", OLD_ENDPOINT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot import the sealed PR304 endpoint loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_endpoint()


def _prepare_inputs() -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    case = score_observer._load_case()
    problem, original, parameters, truth = score_observer._fixed_case_tensors(case, {
        "parameters_sha256": PARAMETERS_SHA, "truth_sha256": TRUTH_SHA})
    if problem.identity != _json(PLAN)["fixed_problem_identity"]:
        raise ValueError("fixed original problem identity changed after freeze")
    layout = problem.layout
    if (tuple(layout["observation_times_seconds"]) != (0.0, 600.0, 1200.0)
            or layout["forecast_time_seconds"] != 12000.0
            or problem.frozen.nowcast_config.forecast_steps != 18
            or truth.shape != (4, 5)):
        raise ValueError("fixed old-problem terminal score or forecast timing changed")
    points, endpoint, _ = _sealed_points()
    identities = {name: seed._input_identity(problem, original, control, parameters, truth)
        for name, control in points.items()}
    expected_input = endpoint["input_after"]
    for name, identity in identities.items():
        if (identity.get("parameters_sha256") != PARAMETERS_SHA
                or identity.get("terminal_truth_sha256") != TRUTH_SHA
                or identity.get("archived_input") != expected_input.get("archived_input")):
            raise ValueError(f"{name} point does not share the fixed admitted parameters/truth/input")
    return problem, original, parameters, truth, points


def _source_hashes(plan: dict[str, Any]) -> dict[str, str]:
    hashes = {name: _sha(ROOT / name) for name in plan["source_files"]}
    hashes[SELF] = _sha(ROOT / SELF)
    return hashes


SELF = "graphify-out/fv-root-cause-20260919/signed_span_20261012/forecast_validation.py"


def freeze_plan() -> dict[str, Any]:
    if PLAN.exists() or PLAN_SHA.exists():
        raise ValueError("forecast plan already frozen")
    inherited = _json(BASE_PLAN_PATH)
    points, _, receipt_files = _sealed_points()
    for name, expected in inherited["source_files"].items():
        if _sha(ROOT / name) != expected:
            raise ValueError(f"inherited original source pin mismatch at freeze: {name}")
    for name, expected in inherited["archive_files"].items():
        if _sha(ROOT / name) != expected:
            raise ValueError(f"inherited original archive pin mismatch at freeze: {name}")
    import tarfile
    with tarfile.open(BASE_SOURCE_TAR, "r:gz") as source_tar:
        source_members = {member.name: source_tar.extractfile(member).read()
            for member in source_tar.getmembers() if member.isfile()}
    if set(source_members) != set(inherited["source_files"]):
        raise ValueError("signed-span original source bundle member set changed")
    if any(_sha_bytes(source_members[name]) != digest
            for name, digest in inherited["source_files"].items()):
        raise ValueError("signed-span original source bundle member digest mismatch")
    case = score_observer._load_case()
    problem, original, parameters, truth = score_observer._fixed_case_tensors(case, {
        "parameters_sha256": PARAMETERS_SHA, "truth_sha256": TRUTH_SHA})
    if _hash_tensor(parameters) != PARAMETERS_SHA or _hash_tensor(truth) != TRUTH_SHA:
        raise ValueError("fixed case parameters or terminal truth differ")
    shared_input = seed._input_identity(problem, original, points["2a4d"], parameters, truth)["archived_input"]
    manifest = _json(FIXED_CASE_MANIFEST)
    receipt_set = {**receipt_files, **inherited["archive_files"]}
    plan = {"experiment_kind": "three_point_old_operator_terminal_forecast_score",
        "source_files": inherited["source_files"],
        "archive_files": {k: _sha(ROOT / k) for k in sorted(receipt_set)},
        "source_bundle_sha256": _sha(BASE_SOURCE_TAR),
        "inherited_source_count": len(inherited["source_files"]),
        "inherited_archive_count": len(inherited["archive_files"]),
        "observer_source_sha256": _sha(ROOT / SELF),
        "signed_span_plan_sha256": BASE_PLAN_SHA,
        "signed_span_source_bundle_sha256": _sha(BASE_SOURCE_TAR),
        "signed_span_child_sha256": _sha(BASE_STEP),
        "signed_span_candidate_receipts": {k: v for k, v in receipt_files.items() if "candidate_01.json" in k},
        "points": {name: _hash_tensor(control) for name, control in points.items()},
        "parameters_sha256": PARAMETERS_SHA, "terminal_truth_sha256": TRUTH_SHA,
        "fixed_case_sha256": _sha(FIXED_CASE),
        "fixed_case_manifest_sha256": _sha(FIXED_CASE_MANIFEST),
        "fixed_problem_identity": problem.identity,
        "fixed_archived_input": shared_input,
        "runtime_identity": manifest["runtime_identity"],
        "timing": {"observation_times_seconds": list(problem.layout["observation_times_seconds"]),
            "absolute_forecast_time_seconds": float(problem.layout["forecast_time_seconds"]),
            "lead_minutes_after_analysis": 180, "forecast_steps": 18,
            "interval_minutes": 10},
        "truth_lead_frames": {"available": False,
            "reason": "the pinned original truth is one terminal 4x5 verification field; no aligned 18-lead truth tensor is present"},
        "policy": INPUT_BUDGET}
    PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    digest = _sha(PLAN)
    PLAN_SHA.write_text(digest + "\n")
    return {"plan_sha256": digest, "source_pins": len(plan["source_files"]),
        "archive_pins": len(plan["archive_files"]), "points": plan["points"]}


def load_plan(digest: str) -> dict[str, Any]:
    if _sha(PLAN) != digest or PLAN_SHA.read_text().strip() != digest:
        raise ValueError("forecast-validation plan seal mismatch")
    plan = _json(PLAN)
    inherited = _json(BASE_PLAN_PATH)
    if (plan.get("experiment_kind") != "three_point_old_operator_terminal_forecast_score"
            or plan.get("policy") != INPUT_BUDGET
            or plan.get("signed_span_plan_sha256") != BASE_PLAN_SHA
            or plan.get("observer_source_sha256") != _sha(ROOT / SELF)
            or plan.get("source_files") != inherited.get("source_files")
            or plan.get("inherited_source_count") != 237
            or plan.get("timing", {}).get("forecast_steps") != 18
            or plan.get("timing", {}).get("lead_minutes_after_analysis") != 180
            or plan.get("truth_lead_frames", {}).get("available") is not False):
        raise ValueError("forecast plan scope/source identity changed")
    for name, expected in plan["source_files"].items():
        if _sha(ROOT / name) != expected:
            raise ValueError(f"original source pin mismatch: {name}")
    for name, expected in plan["archive_files"].items():
        if _sha(ROOT / name) != expected:
            raise ValueError(f"forecast input/archive pin mismatch: {name}")
    if _sha(BASE_PLAN_PATH) != BASE_PLAN_SHA or _sha(BASE_SOURCE_TAR) != plan["source_bundle_sha256"]:
        raise ValueError("signed-span parent bundle changed")
    return plan


def _run_child(digest: str) -> dict[str, Any]:
    started = time.monotonic()
    plan = load_plan(digest)
    source_before = _source_hashes(plan)
    runtime_before = cycle.shared._runtime()
    if runtime_before != plan["runtime_identity"]:
        raise ValueError("runtime differs from the sealed original fixed case")
    manifest_before = _sha(FIXED_CASE_MANIFEST)
    problem, original, parameters, truth, points = _prepare_inputs()
    identities = {name: seed._input_identity(problem, original, control, parameters, truth)
        for name, control in points.items()}
    for name, control in points.items():
        if _hash_tensor(control) != plan["points"][name]:
            raise ValueError(f"point {name} control changed after plan freeze")
    input_before = {name: identities[name] for name in POINTS}
    result: dict[str, Any] = {"phase": "scoring", "status": "running",
        "plan_sha256": digest, "source_before": source_before,
        "runtime_before": runtime_before, "input_before": input_before,
        "score_calls_started": 0, "score_calls_completed": 0,
        "score_results": {}, "optimizer_steps": 0, "candidate_evaluations": 0,
        "hvp_calls": 0, "selection_use": False,
        "lead_frame_validation": plan["truth_lead_frames"],
        "claim_scope": "terminal 180-minute-after-analysis dBZ MSE only"}
    out = ATTEMPT / "forecast_validation.json"
    frame_path = ATTEMPT / "forecast_fields.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    deadline = started + INPUT_BUDGET["internal_seconds"]

    def check() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("60-second internal score-observation budget reached")
        if (_source_hashes(plan) != source_before or cycle.shared._runtime() != runtime_before
                or _sha(FIXED_CASE_MANIFEST) != manifest_before
                or _sha(FIXED_CASE) != plan["fixed_case_sha256"]):
            raise ValueError("source, runtime, or fixed case changed during terminal scoring")
        for name, control in points.items():
            if seed._input_identity(problem, original, control, parameters, truth) != input_before[name]:
                raise ValueError(f"fixed {name} input changed during score calls")

    from importlib import import_module
    from advar.physics import echo_to_dbz
    problem_module = import_module(problem.__class__.__module__)
    original_forecast = problem_module.v.forecast_fv_analysis
    captured: list[torch.Tensor] = []

    def capture_forecast(*args: Any, **kwargs: Any) -> Any:
        trajectory = original_forecast(*args, **kwargs)
        if torch.is_grad_enabled():
            raise ValueError("terminal score forecast unexpectedly entered an AD context")
        frames = trajectory.frames_linear
        if frames.ndim != 3 or frames.shape[0] != 19 or not bool(torch.isfinite(frames).all()):
            raise ValueError("forecast did not return 18 finite leads plus the analysis-end frame")
        captured.append(frames.detach().cpu().clone())
        return trajectory

    try:
        problem_module.v.forecast_fv_analysis = capture_forecast
        for name in POINTS:
            check()
            result["score_calls_started"] += 1
            out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
            with torch.no_grad():
                score = problem.score(points[name], parameters)
            if (not isinstance(score, torch.Tensor) or score.ndim != 0
                    or not bool(torch.isfinite(score)) or float(score) < 0.0
                    or len(captured) != result["score_calls_started"]):
                raise ValueError(f"terminal score at {name} is invalid or missed one forecast call")
            dBZ_frames = echo_to_dbz(captured[-1], min_dbz=problem.frozen.nowcast_config.min_dbz)
            residual = dBZ_frames[-1] - truth
            recomputed_mse = residual.square().mean()
            score_value = float(score)
            mse_value = float(recomputed_mse)
            budget = 128.0 * torch.finfo(truth.dtype).eps * max(
                abs(score_value), abs(mse_value), torch.finfo(truth.dtype).tiny)
            if abs(score_value - mse_value) > budget:
                raise ValueError(f"terminal score at {name} does not match captured final forecast frame")
            result["score_results"][name] = {"terminal_dbz_mse": score_value,
                "terminal_dbz_rmse": math.sqrt(score_value),
                "terminal_dbz_mae": float(residual.abs().mean()),
                "control_sha256": plan["points"][name],
                "forecast_steps": 18, "lead_minutes_after_analysis": 180,
                "score_domain": "fixed 4x5 off-grid verification points",
                "final_frame_score_matches_problem_score": True}
            result["score_calls_completed"] += 1
            out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        problem_module.v.forecast_fv_analysis = original_forecast
        check()
        lead_frames = {name: echo_to_dbz(frames[1:], min_dbz=problem.frozen.nowcast_config.min_dbz)
            for name, frames in zip(POINTS, captured)}
        torch.save({"controls": plan["points"], "field_frames_dbz": lead_frames,
            "lead_minutes_after_analysis": list(range(10, 181, 10)),
            "observation_times_seconds": list(problem.layout["observation_times_seconds"]),
            "forecast_absolute_terminal_seconds": float(problem.layout["forecast_time_seconds"]),
            "original_truth_lead_frames_available": False}, frame_path)
        values = {k: v["terminal_dbz_mse"] for k, v in result["score_results"].items()}
        result["contrasts"] = {
            "fb0a_minus_2a4d_mse": values["fb0a"] - values["2a4d"],
            "96108_minus_fb0a_mse": values["96108"] - values["fb0a"],
            "96108_minus_2a4d_mse": values["96108"] - values["2a4d"]}
        result.update(status="completed", phase="finished", source_after=_source_hashes(plan),
            runtime_after=cycle.shared._runtime(), input_after={name: seed._input_identity(
                problem, original, control, parameters, truth) for name, control in points.items()},
            source_unchanged=True, runtime_unchanged=True, input_unchanged=True,
            fixed_case_unchanged=True, optimizer_steps=0, candidate_evaluations=0,
            hvp_calls=0, selection_use=False, score_calls=3,
            forecast_calls=3,
            forecast_fields_path=frame_path.relative_to(ROOT).as_posix(),
            forecast_fields_sha256=_sha(frame_path),
            lead_frame_mse_mae_status="not_computed_no_original_truth_lead_frames",
            persistence_status="not_computed_no_original_truth_lead_frames",
            elapsed_seconds=time.monotonic() - started)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return result
    except Exception as error:
        problem_module.v.forecast_fv_analysis = original_forecast
        result.update(status="failed", phase="failed",
            failure=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        raise


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
        _run_child(args.plan_sha256)
        return
    if ATTEMPT.exists():
        raise ValueError("forecast validation attempt directory must be fresh")
    from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic
    ATTEMPT.mkdir(parents=True)
    resource = run_guarded_diagnostic([str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
        "--child", "--plan-sha256", args.plan_sha256],
        wall_seconds=INPUT_BUDGET["outer_seconds"], rss_bytes=INPUT_BUDGET["rss_bytes"],
        report_path=ATTEMPT / "forecast_validation.resource.json",
        log_path=ATTEMPT / "forecast_validation.log")
    step_path = ATTEMPT / "forecast_validation.json"
    run = {"resource": resource, "child_sha256": _sha(step_path) if step_path.is_file() else None,
        "execution_status": "completed" if resource.get("exit_code") == 0 and step_path.is_file()
            and _json(step_path).get("status") == "completed" else "failed"}
    (ATTEMPT / "forecast_validation.run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run, sort_keys=True))
    if run["execution_status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
