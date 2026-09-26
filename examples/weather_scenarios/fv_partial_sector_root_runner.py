"""Guard and independently classify one current-source partial FV root search."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, TypeGuard

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios.fv_point_basin_runner import _valid_curvature
from examples.weather_scenarios import fv_partial_sector_root_probe as probe


WALL_SECONDS = 600
SAMPLED_RSS_BYTES = 1024**3
ROOT_STATUSES = {"root_margin_qualified", "root_low_margin"}
REFUSALS = {"seed_curvature_refused", "root_refused"}


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def _branch(value: object) -> TypeGuard[dict[str, Any]]:
    return (isinstance(value, dict)
            and value.get("euler_stages") == 54
            and _digest(value.get("signature_sha256"))
            and all(type(value.get(key)) in (int, float)
                    and math.isfinite(value[key]) and value[key] > 0
                    for key in ("minimum_scaled_slope_margin",
                                "minimum_scaled_face_flux_margin")))


def _control(value: object) -> torch.Tensor | None:
    if (not isinstance(value, list) or len(value) != 26
            or not all(type(item) in (int, float) and math.isfinite(item)
                       for item in value)):
        return None
    return torch.tensor(value, dtype=torch.float64)


_BRANCH_FIELDS = ("signature_sha256", "euler_stages",
                  "minimum_scaled_slope_margin", "minimum_scaled_face_flux_margin")


def _branch_call_matches(call: object, control_sha: str, branch: dict[str, Any]) -> bool:
    return (isinstance(call, dict)
            and call.get("status") == "core_branch_admitted"
            and call.get("control_sha256") == control_sha
            and all(call.get(key) == branch[key] for key in _BRANCH_FIELDS))


def _valid_refinement_path(child: dict[str, Any], calls: list[Any]) -> bool:
    iterations = child["refinement_iterations"]
    trials = child.get("trial_records")
    solves = child.get("linear_solves")
    if not isinstance(trials, list) or not isinstance(solves, list):
        return False
    refinement = [entry for entry in solves if isinstance(entry, dict)
                  and entry.get("phase") == "sector_refinement"]
    accepted = [entry for entry in trials if isinstance(entry, dict)
                and entry.get("accepted") is True]
    policies = child.get("policy_records")
    if len(refinement) != iterations or len(accepted) != iterations:
        return False
    if not isinstance(policies, list):
        return False
    previous_signature = child["seed_branch"]["signature_sha256"]
    seed_norm = child.get("seed_gradient_norm")
    if (not isinstance(seed_norm, (int, float)) or isinstance(seed_norm, bool)
            or not math.isfinite(seed_norm) or seed_norm < 0):
        return False
    previous_norm = float(seed_norm)
    for index, (trial, solve) in enumerate(zip(accepted, refinement), start=1):
        candidate = _control(trial.get("candidate_control"))
        branch = trial.get("branch")
        candidate_sha = trial.get("candidate_control_sha256")
        if not (
            trial.get("iteration") == index
            and candidate is not None
            and isinstance(candidate_sha, str)
            and candidate_sha == probe._tensor_sha(candidate)
            and _branch(branch)
            and any(_branch_call_matches(call, candidate_sha, branch) for call in calls)
            and type(trial.get("linear_relative_residual")) in (int, float)
            and math.isfinite(trial["linear_relative_residual"])
            and 0 <= trial["linear_relative_residual"] <= 1e-10
            and type(trial.get("pcg_relative_residual")) in (int, float)
            and math.isfinite(trial["pcg_relative_residual"])
            and 0 <= trial["pcg_relative_residual"] <= 1e-10
            and type(trial.get("pcg_iterations")) is int
            and trial["pcg_iterations"] == solve.get("iterations")
            and trial["pcg_relative_residual"] == solve.get("relative_residual")
            and solve.get("converged") is True
            and type(trial.get("gradient_norm")) in (int, float)
            and math.isfinite(trial["gradient_norm"])
            and trial["gradient_norm"] >= 0
        ):
            return False
        candidate_norm = float(trial["gradient_norm"])
        switched = branch["signature_sha256"] != previous_signature
        if switched:
            old_merit = 0.5 * previous_norm * previous_norm
            new_merit = 0.5 * candidate_norm * candidate_norm
            delta = old_merit - new_merit
            floor = 128 * torch.finfo(torch.float64).eps * max(
                abs(old_merit), abs(new_merit), torch.finfo(torch.float64).tiny)
            matching = [record for record in policies if isinstance(record, dict)
                        and record.get("iteration") == index
                        and record.get("backtrack") == trial.get("backtrack")
                        and record.get("step_scale") == trial.get("step_scale")
                        and record.get("current_signature_sha256") == previous_signature
                        and record.get("candidate_signature_sha256") == branch["signature_sha256"]]
            if not (
                math.isfinite(old_merit) and math.isfinite(new_merit)
                and math.isfinite(delta) and math.isfinite(floor) and delta > floor
                and trial.get("acceptance_policy") == "trial_acceptance"
                and trial.get("policy_reason") == "merit_switch_decrease"
                and len(matching) == 1
                and matching[0].get("accepted") is True
                and matching[0].get("reason") == "merit_switch_decrease"
                and all(matching[0].get(key) == value for key, value in (
                    ("old_merit", old_merit), ("new_merit", new_merit),
                    ("delta", delta), ("floor", floor)
                ))
            ):
                return False
        elif not (
            "acceptance_policy" not in trial
            and type(trial.get("armijo_ratio")) in (int, float)
            and math.isfinite(trial["armijo_ratio"])
            and 0 <= trial["armijo_ratio"] <= 1
            and not any(isinstance(record, dict)
                        and record.get("iteration") == index
                        and record.get("backtrack") == trial.get("backtrack")
                        for record in policies)
        ):
            return False
        previous_signature = branch["signature_sha256"]
        previous_norm = candidate_norm
    last_hash = (accepted[-1]["candidate_control_sha256"] if accepted
                 else child["gauss_newton"]["control_sha256"])
    return last_hash == child.get("final_control_sha256")


def _valid_root(child: dict[str, Any]) -> bool:
    control = _control(child.get("control"))
    seed = child.get("seed_branch")
    final = child.get("final_branch")
    calls = child.get("branch_calls")
    if not (
        child.get("phase") == "finished"
        and control is not None
        and _branch(seed) and _branch(final)
        and _valid_curvature(child.get("seed_curvature"))
        and _valid_curvature(child.get("final_curvature"))
        and child.get("seed_curvature_control_sha256") == child["gauss_newton"]["control_sha256"]
        and child.get("final_curvature_control_sha256") == child.get("final_control_sha256")
        and child.get("final_control_sha256") == probe._tensor_sha(control)
        and type(child.get("final_gradient_max")) in (int, float)
        and math.isfinite(child["final_gradient_max"])
        and 0 <= child["final_gradient_max"] < 1e-10
        and type(child.get("refinement_iterations")) is int
        and 0 <= child["refinement_iterations"] <= 8
        and all(type(child.get(key)) in (int, float) and math.isfinite(child[key])
                for key in ("final_objective", "final_score"))
        and isinstance(calls, list) and len(calls) >= 2
        and isinstance(calls[-1], dict)
        and calls[-1].get("status") == "core_branch_admitted"
        and calls[-1].get("control_sha256") == child["final_control_sha256"]
        and _branch_call_matches(calls[-1], child["final_control_sha256"], final)
    ):
        return False
    high = (final["minimum_scaled_slope_margin"] > 1e-4
            and final["minimum_scaled_face_flux_margin"] > 1e-4)
    if (child.get("response_margin_qualified") is not high
            or child.get("numerical_status") != (
                "root_margin_qualified" if high else "root_low_margin")):
        return False
    return _valid_refinement_path(child, calls)


def run(directory: Path) -> dict[str, object]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("partial sector output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "partial_sector_root.json"
    command = [sys.executable,
               str(ROOT / "examples/weather_scenarios/fv_partial_sector_root_probe.py"),
               "--output", str(output)]
    source_before = probe._sources()
    plan_before = probe._sha(probe.PLAN)
    input_before = probe._preflight_identity()
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "partial_sector_root.resource.json",
        log_path=directory / "partial_sector_root.log",
    )
    child = None
    read_error = None
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    current = None
    provenance_error = None
    try:
        current = probe._preflight_identity()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        provenance_error = f"{type(error).__name__}: {error}"
    resource_ok = (
        resource.get("exit_code") in (0, 2)
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("command") == command
        and type(resource.get("child_pid")) is int and resource["child_pid"] > 0
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
        and type(resource.get("elapsed_seconds")) in (int, float)
        and math.isfinite(resource["elapsed_seconds"])
        and 0 <= resource["elapsed_seconds"] <= WALL_SECONDS
    )
    child_ok = (
        isinstance(child, dict)
        and child.get("numerical_status") in ROOT_STATUSES | REFUSALS
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("plan_unchanged") is True
        and child.get("warm_control_unchanged") is True
        and child.get("parameters_unchanged") is True
        and child.get("source_before") == child.get("source_after") == source_before == probe._sources()
        and child.get("input_before") == child.get("input_after") == input_before == current
        and child.get("plan_sha256") == plan_before == probe.PLAN_SHA256 == probe._sha(probe.PLAN)
        and type(child.get("pid")) is int and child["pid"] == resource.get("child_pid")
        and isinstance(child.get("gauss_newton"), dict)
        and (seed_control := _control(child["gauss_newton"].get("control"))) is not None
        and child["gauss_newton"].get("control_sha256") == probe._tensor_sha(seed_control)
        and _branch(child.get("seed_branch"))
        and isinstance(child.get("branch_calls"), list)
        and any(_branch_call_matches(call, child["gauss_newton"]["control_sha256"],
                                     child["seed_branch"])
                for call in child["branch_calls"])
        and ((child["numerical_status"] in ROOT_STATUSES
              and resource["exit_code"] == 0 and _valid_root(child))
             or (child["numerical_status"] in REFUSALS
                 and resource["exit_code"] == 2
                 and isinstance(child.get("refusal"), str)
                 and (child.get("phase") == "seed_curvature"
                      if child["numerical_status"] == "seed_curvature_refused"
                      else child.get("phase") in ("sector_refinement", "final_curvature"))))
    )
    result: dict[str, object] = {
        "execution_status": "completed" if resource_ok and child_ok else "failed",
        "numerical_status": child.get("numerical_status") if isinstance(child, dict) else "not_reached",
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "resource": resource,
        "child_read_error": read_error,
        "provenance_error": provenance_error,
        "scope": "one current-source two-hole partial FV nominal root search; no adjoint/reanalysis",
    }
    (directory / "partial_sector_root.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
