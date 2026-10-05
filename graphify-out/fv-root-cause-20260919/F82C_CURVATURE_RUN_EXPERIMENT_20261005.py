"""Frozen three-launch f82c curvature experiment; no optimizer or score."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.weather_scenarios.fv_diagnostic_guard import atomic_write_text, run_guarded_diagnostic
from examples.weather_scenarios.fv_point_3h_bounded_coupled_step import execution_status

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    if sha(plan_path) != args.plan_sha256:
        raise ValueError("frozen plan hash changed")
    plan = json.loads(plan_path.read_text())
    policy = plan["policy"]
    if (policy.get("max_guarded_launches") != 3 or policy.get("per_launch_wall_seconds") != 300.
            or policy.get("total_outer_reserved_seconds") != 900.):
        raise ValueError("coordinator requires the frozen three-launch/900-second policy")
    for name, digest in {**plan["source_files"], **plan["archive_files"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"frozen experiment input/source changed: {name}")
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT / "graphify-out/fv-root-cause-20260919"):
        raise ValueError("experiment directory must be inside evidence tree")
    directory.mkdir(parents=False, exist_ok=False)
    checkpoint = directory / "checkpoint"
    ledger: dict[str, Any] = {"phase": "running", "scope": plan["scope"],
        "plan_sha256": args.plan_sha256, "attempts": [], "reserved_outer_seconds": 0,
        "control_sha256": plan["control_sha256"], "parameters_sha256": plan["parameters_sha256"],
        "optimizer_step_applied": False, "response_computed": False, "physical_validated": False}
    path = directory / "experiment.json"
    started = time.monotonic()
    write(path, ledger)
    for index in range(1, 4):
        if ledger["reserved_outer_seconds"] + 300 > 900 or time.monotonic() - started + 300 > 900:
            ledger["stop_reason"] = "insufficient cumulative reservation or wall allowance"
            break
        attempt = directory / f"attempt_{index}"
        attempt.mkdir(exist_ok=False)
        row: dict[str, Any] = {"attempt": index, "status": "reserved", "reserved_outer_seconds": 300}
        ledger["attempts"].append(row)
        ledger["reserved_outer_seconds"] += 300
        write(path, ledger)
        command = [str(ROOT / ".venv/bin/python"), "-m", "examples.weather_scenarios.fv_point_3h_accepted_curvature",
            "--child", "--plan", str(plan_path), "--plan-sha256", args.plan_sha256,
            "--output-directory", str(attempt), "--checkpoint-directory", str(checkpoint)]
        resource_path = attempt / "audit.resource.json"
        parent_path = attempt / "audit.run.json"
        try:
            resource = run_guarded_diagnostic(command, wall_seconds=300., rss_bytes=1024**3,
                report_path=resource_path, log_path=attempt / "audit.log")
        except Exception as error:
            row.update(status="failed", failure=f"{type(error).__name__}: {error}")
            ledger["stop_reason"] = "guard failure; no retry"
            write(path, ledger)
            break
        parent: dict[str, Any] = {"phase": "finished", "execution_status": execution_status(resource),
            "resource": resource, "numerical_status": "not_reached", "child_read_error": None}
        write(parent_path, parent)
        row.update(status=parent["execution_status"], resource=resource, resource_sha256=sha(resource_path))
        write(path, ledger)
        try:
            child_path = attempt / "audit.json"
            child = json.loads(child_path.read_text())
            if not isinstance(child, dict):
                raise ValueError("child audit must be a JSON object")
            valid = (child.get("phase") == "finished" and child.get("source_unchanged") is True
                and child.get("fixed_input_unchanged") is True and child.get("runtime_after") == child.get("runtime")
                and child.get("control_sha256") == plan["control_sha256"]
                and child.get("parameters_sha256") == plan["parameters_sha256"]
                and child.get("plan_sha256") == args.plan_sha256)
            parent.update(child_sha256=sha(child_path), numerical_status=child.get("numerical_status"))
            if parent["execution_status"] == "completed" and not valid:
                parent.update(execution_status="failed", child_read_error="child input/source/runtime completion not verified")
            write(parent_path, parent)
            row.update(status=parent["execution_status"], numerical_status=parent["numerical_status"],
                parent_sha256=sha(parent_path), child_read_error=parent["child_read_error"])
            row.update(child_sha256=sha(child_path), completed_columns=child.get("hvp_columns_completed"),
                hvp_calls_total=child.get("hvp_calls_total"))
            if row["status"] != "completed":
                ledger["stop_reason"] = "execution failure; no retry"
                break
            if child.get("numerical_status") == "curvature_completed":
                ledger["stop_reason"] = "fresh accepted-point curvature completed"
                break
            if child.get("numerical_status") != "curvature_budget_refused":
                ledger["stop_reason"] = "non-budget numerical/input refusal; no retry"
                break
        except (OSError, ValueError, TypeError, AttributeError) as error:
            parent.update(execution_status="failed", child_read_error=f"{type(error).__name__}: {error}")
            write(parent_path, parent)
            row.update(status="failed", parent_sha256=sha(parent_path), failure=parent["child_read_error"])
            ledger["stop_reason"] = "missing/invalid attempt artifact; no retry"
            break
        write(path, ledger)
    else:
        ledger["stop_reason"] = "launch quota exhausted"
    ledger.update(phase="finished", elapsed_seconds=time.monotonic() - started,
        source_unchanged=all(sha(ROOT / name) == digest for name, digest in plan["source_files"].items()),
        original_archive_unchanged=all(sha(ROOT / name) == digest for name, digest in plan["archive_files"].items()))
    write(path, ledger)
    print(json.dumps({"stop_reason": ledger["stop_reason"], "attempts": len(ledger["attempts"])}))

if __name__ == "__main__":
    main()
