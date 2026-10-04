"""One-shot, three-launch coordinator for the pinned cd6b Hessian experiment."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from examples.weather_scenarios import fv_point_3h_endpoint_diagnostics as diagnostic
from examples.weather_scenarios.fv_diagnostic_guard import atomic_write_text

ROOT = diagnostic.ROOT.resolve()
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
ARCHIVE = EVIDENCE / "coupled_original_j_continuation_attempt1/coupled_original_j_continuation.json"
PLAN = EVIDENCE / "RESUMABLE_HESSIAN_IMPLEMENTATION_PLAN_20261004.md"
CONTROL_SHA = "cd6b5693e2bd32a8a41a3509b87b61e7cf647b03d864dfccc0a4cccce96e908d"
PARAMETERS_SHA = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
ARCHIVE_SHA = "3e2e1e247be0de1c6b5e27d4c835dbbe6b456f7a0c0dc2e0e90ac18d71fd8385"
PARENT_SHA = "5cf9c94f61f8154ab895f3c7cd3d85cc45e8841774554629e35def59785f0790"
RESOURCE_SHA = "6e642669e49a776a928eaa9832e470bb3f27286902c4527f5c1910a8135ffc96"
PLAN_SHA = "ce72cd6693731e97f0155cc6077d170c16f3e0d43264effd3ecb607d31ee2842"
MAX_LAUNCHES, TOTAL_SECONDS, LAUNCH_SECONDS = 3, 900.0, 300.0
RSS_BYTES = 1024**3
SOURCE_PINS = {
    diagnostic.SELF: "6e48e23d2ad6be3a54ade759a35d720a9d928163dda70eb0906decc057047c8c",
    diagnostic.TEST: "b43ba53cd220ceb3d16c94fd8f269ed827a6a722145d1d55dc23f62b41dcb459",
    diagnostic.CHECKPOINT: "7d16d92d3ec64c1c749470a4601d674809b3de30b21dc49472b1cf45832dde8c",
    diagnostic.CHECKPOINT_TEST: "33b01676d564cf0cc9a3c403df58a691b9853f011e64ae58a1224646b4ad269e",
    diagnostic.GUARD: "a1c6b71dcaa4a16952658c9057839cef405238390908476b04e793710f6d275c",
    diagnostic.GUARD_TEST: "70b91717617dfb1b32dcee5d7cb1b90c355fa09d143047c9b30c262043c437ec",
    "tests/test_fv_objective_score_dependency.py": "de41a42612c22aa63c5eb60aac4fe3b36986657a74afc3c8c2530e1513b66e03",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _relative_hashes(paths: set[str]) -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in sorted(paths)}


def _fresh(path: Path) -> bool:
    return not path.exists() and not path.is_symlink()


def run_checkpoint_experiment(*, directory: Path, archive: Path = ARCHIVE,
                              archive_sha256: str = ARCHIVE_SHA,
                              parent_sha256: str = PARENT_SHA,
                              resource_sha256: str = RESOURCE_SHA,
                              plan: Path = PLAN, plan_sha256: str = PLAN_SHA) -> dict[str, Any]:
    """Run the frozen same-point experiment once; every guarded launch is durable."""
    root, evidence = ROOT.resolve(), EVIDENCE.resolve()
    target = directory.resolve()
    if (not target.is_relative_to(evidence) or target == evidence
            or directory.exists() or directory.is_symlink()):
        raise ValueError("experiment directory must be a fresh child of the evidence tree")
    archive_path, plan_path = archive.resolve(), plan.resolve()
    if (archive_path != ARCHIVE.resolve() or plan_path != PLAN.resolve()
            or archive_sha256 != ARCHIVE_SHA or parent_sha256 != PARENT_SHA
            or resource_sha256 != RESOURCE_SHA or plan_sha256 != PLAN_SHA):
        raise ValueError("experiment must use the caller-pinned cd6b archive, sidecars and plan")
    if any(_sha(root / name) != digest for name, digest in SOURCE_PINS.items()):
        raise ValueError("reviewed endpoint/checkpoint dependency pin changed")
    raw, actual_archive_sha = diagnostic._read_object(archive_path)
    parent_path, resource_path = archive_path.with_suffix(".run.json"), archive_path.with_suffix(".resource.json")
    parent, actual_parent_sha = diagnostic._read_object(parent_path)
    resource, actual_resource_sha = diagnostic._read_object(resource_path)
    if (actual_archive_sha != archive_sha256 or actual_parent_sha != parent_sha256
            or actual_resource_sha != resource_sha256 or _sha(plan_path) != plan_sha256):
        raise ValueError("pinned original archive/sidecar/plan bytes changed")
    diagnostic._validate_parent_and_resource(raw, parent, resource)
    accepted = diagnostic.extract_last_accepted_trial(raw)
    if (raw.get("last_accepted_control_sha256") != CONTROL_SHA
            or raw.get("parameters_sha256") != PARAMETERS_SHA
            or accepted["control"] != raw.get("last_accepted_control")):
        raise ValueError("archive endpoint is not the declared cd6b control/parameter point")
    archive_relative = archive_path.relative_to(root).as_posix()
    parent_relative = parent_path.relative_to(root).as_posix()
    resource_relative = resource_path.relative_to(root).as_posix()
    plan_relative = plan_path.relative_to(root).as_posix()
    own_source = Path(__file__).resolve().relative_to(root).as_posix()
    own_test = "tests/test_fv_checkpoint_experiment.py"
    paths = set(raw["source_before"]) | set(SOURCE_PINS) | {
        archive_relative, parent_relative, resource_relative, plan_relative, own_source, own_test,
    }
    source_before = _relative_hashes(paths)
    if any(source_before.get(name) != digest for name, digest in raw["source_before"].items()):
        raise ValueError("current dependencies differ from the immutable endpoint archive")
    if (source_before[archive_relative] != archive_sha256
            or source_before[parent_relative] != parent_sha256
            or source_before[resource_relative] != resource_sha256
            or source_before[plan_relative] != plan_sha256):
        raise ValueError("experiment source map does not bind all original sidecars and the plan")

    target.mkdir(parents=False, exist_ok=False)
    with (target / ".experiment.claim").open("x", encoding="utf-8") as claim:
        claim.write(json.dumps({"archive_sha256": archive_sha256, "control_sha256": CONTROL_SHA,
            "parameters_sha256": PARAMETERS_SHA, "max_guarded_launches": MAX_LAUNCHES,
            "cumulative_outer_seconds": TOTAL_SECONDS}) + "\n")
    checkpoint_directory = target / "checkpoint"
    preflight = {"scope": "metadata-only; no FV/objective/branch/HVP evaluation",
        "source_before": source_before, "archive_sha256": archive_sha256,
        "parent_sha256": parent_sha256, "resource_sha256": resource_sha256,
        "plan_sha256": plan_sha256, "control_sha256": CONTROL_SHA,
        "parameters_sha256": PARAMETERS_SHA,
        "checkpoint_directory": checkpoint_directory.relative_to(root).as_posix(),
        "max_guarded_launches": MAX_LAUNCHES, "cumulative_outer_seconds": TOTAL_SECONDS,
        "per_launch_outer_seconds": LAUNCH_SECONDS, "per_kernel_attempt_seconds": 240.0,
        "max_kernel_attempts": MAX_LAUNCHES, "rss_limit_bytes": RSS_BYTES,
        "quota_scope": "kernel reservations exclude metadata/branch/J/g work; each guarded launch still consumes one slot"}
    _write(target / "preflight.json", preflight)
    ledger: dict[str, Any] = {"scope": "one immutable source/point checkpoint experiment; no optimization or response",
        "phase": "running", "source_before": source_before, "attempts": [],
        "max_guarded_launches": MAX_LAUNCHES, "cumulative_outer_seconds": TOTAL_SECONDS,
        "control_sha256": CONTROL_SHA, "parameters_sha256": PARAMETERS_SHA,
        "checkpoint_directory": checkpoint_directory.relative_to(root).as_posix(),
        "optimizer_step_applied": False, "response_computed": False, "physical_validated": False}
    _write(target / "experiment.json", ledger)
    started = time.monotonic()
    for index in range(1, MAX_LAUNCHES + 1):
        if time.monotonic() - started + LAUNCH_SECONDS > TOTAL_SECONDS:
            ledger["stop_reason"] = "insufficient cumulative allowance for another full guarded launch"
            break
        attempt_dir = target / f"attempt_{index}"
        row: dict[str, Any] = {"attempt": index, "reserved_outer_seconds": LAUNCH_SECONDS,
            "status": "reserved", "output_directory": attempt_dir.relative_to(root).as_posix()}
        ledger["attempts"].append(row)
        _write(target / "experiment.json", ledger)
        output = attempt_dir / "audit.json"
        try:
            record = diagnostic.guarded_run(archive=archive_path, archive_sha256=archive_sha256,
                parent_sha256=parent_sha256, resource_sha256=resource_sha256,
                plan=plan_path, output=output, checkpoint_directory=checkpoint_directory,
                checkpoint_max_attempts=MAX_LAUNCHES)
        except Exception as error:
            row.update(status="failed", failure=f"{type(error).__name__}: {error}")
            ledger["stop_reason"] = "guarded launch raised; no retry"
            _write(target / "experiment.json", ledger)
            break
        resource_actual = record.get("resource")
        parent_artifact, resource_artifact = output.with_suffix(".run.json"), output.with_suffix(".resource.json")
        missing = [str(path.relative_to(root)) for path in (parent_artifact, resource_artifact)
                   if not path.is_file()]
        row.update(status=record.get("execution_status", "failed"),
            numerical_status=record.get("numerical_status", "not_reached"),
            child_read_error=record.get("child_read_error"), child_terminal_phase=record.get("child_terminal_phase"),
            endpoint_identity_matches_archive=record.get("endpoint_identity_matches_archive"),
            source_unchanged=record.get("source_unchanged"), child_source_maps_match=record.get("child_source_maps_match"),
            child_runtime_matches=record.get("child_runtime_matches"), resource=resource_actual,
            parent_record=parent_artifact.relative_to(root).as_posix(),
            resource_record=resource_artifact.relative_to(root).as_posix(),
            parent_record_sha256=_sha(parent_artifact) if parent_artifact.is_file() else None,
            resource_record_sha256=_sha(resource_artifact) if resource_artifact.is_file() else None,
            missing_artifacts=missing,
            elapsed_seconds=resource_actual.get("elapsed_seconds") if isinstance(resource_actual, dict) else None,
            checkpoint_progress=record.get("checkpoint", {"status": "not_recorded"}))
        if missing:
            row["status"] = "failed"
            row["failure"] = "guard parent/resource artifact missing after launch"
        # Persist the completed parent outcome before any optional child-file inspection.
        _write(target / "experiment.json", ledger)
        if row["status"] == "completed":
            child_error = record.get("child_read_error")
            if child_error is not None:
                row.update(status="failed", child_read_error=child_error,
                           failure="guard parent reports child output could not be read")
            else:
                try:
                    child, child_sha = diagnostic._read_object(output)
                    if child_sha != record.get("child_sha256"):
                        raise ValueError("child report SHA differs from its parent record")
                    row["checkpoint_progress"] = child.get("checkpoint", {"status": "not_recorded"})
                    row["child_terminal_phase"] = child.get("phase")
                except (OSError, ValueError) as error:
                    row.update(status="failed", child_read_error=str(error),
                               failure="completed parent did not bind a readable child report")
            _write(target / "experiment.json", ledger)
        if row["status"] != "completed":
            ledger["stop_reason"] = "execution/resource/integrity failure; no retry"
            break
        if row["numerical_status"] == "endpoint_diagnostic_completed":
            ledger["stop_reason"] = "fresh curvature diagnostic completed"
            break
        if row["numerical_status"] != "checkpoint_budget_refusal":
            ledger["stop_reason"] = "nonbudget numerical refusal; no retry"
            break
    else:
        ledger["stop_reason"] = "declared three-launch limit exhausted"
    source_after = _relative_hashes(paths)
    ledger.update(phase="finished", source_after=source_after,
        source_unchanged=source_after == source_before,
        original_archive_unchanged=_sha(archive_path) == archive_sha256,
        elapsed_seconds=time.monotonic() - started,
        reserved_outer_seconds=LAUNCH_SECONDS * len(ledger["attempts"]))
    if not ledger["source_unchanged"] or not ledger["original_archive_unchanged"]:
        ledger["stop_reason"] = "source/archive changed; result publication refused"
    _write(target / "experiment.json", ledger)
    return ledger
