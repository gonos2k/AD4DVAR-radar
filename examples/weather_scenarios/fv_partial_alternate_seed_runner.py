"""Guard and classify one exploratory alternate-sector seed audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios.fv_point_basin_runner import _valid_curvature
from examples.weather_scenarios import fv_partial_alternate_seed_gate as gate


WALL_SECONDS = 120
SAMPLED_RSS_BYTES = 1024**3
VALID_STATUSES = {"seed_locally_spd", "seed_branch_refused", "seed_curvature_refused"}
_COMMON_CHILD_KEYS = frozenset({
    "pid", "phase", "numerical_status", "scope", "response_validation",
    "physical_validation", "environment", "archive_manifest_sha256",
    "plan_sha256", "reviewed_probe_sha256", "source_before", "source_after",
    "input_before", "input_after", "selected_iteration", "selected_backtrack",
    "eligible_candidate_count", "selected_control_sha256",
    "selected_signature_sha256", "selected_previously_accepted",
    "selected_previous_policy_reason", "new_product_gn_runs",
    "new_newton_refinement_runs", "new_adjoint_reanalysis_runs",
    "objective", "gradient_norm", "gradient_max", "elapsed_seconds",
    "raw_unchanged",
})


def _positive_branch(value: object) -> bool:
    return (isinstance(value, dict)
            and value.get("euler_stages") == 54
            and value.get("signature_sha256") == gate.SELECTED_SIGNATURE_SHA256
            and all(type(value.get(name)) in (int, float)
                    and math.isfinite(value[name]) and value[name] > 1e-4
                    for name in ("minimum_scaled_slope_margin",
                                 "minimum_scaled_face_flux_margin")))


def _valid_child(child: dict[str, Any], exit_code: int) -> bool:
    common = (child.get("selected_iteration") == 5
              and child.get("selected_backtrack") == 3
              and child.get("eligible_candidate_count") == 29
              and child.get("selected_control_sha256") == gate.SELECTED_CONTROL_SHA256
              and child.get("selected_signature_sha256") == gate.SELECTED_SIGNATURE_SHA256
              and child.get("selected_previously_accepted") is False
              and child.get("selected_previous_policy_reason") == "merit_switch_refused"
              and child.get("new_product_gn_runs") == 0
              and child.get("new_newton_refinement_runs") == 0
              and child.get("new_adjoint_reanalysis_runs") == 0
              and child.get("response_validation") == "not_performed"
              and child.get("physical_validation") == "not_performed"
              and child.get("raw_unchanged") is True
              and all(type(child.get(name)) in (int, float)
                      and math.isfinite(child[name])
                      for name in ("objective", "gradient_norm", "gradient_max")))
    if not common:
        return False
    status = child.get("numerical_status")
    if status == "seed_locally_spd":
        return (set(child) == _COMMON_CHILD_KEYS | {
                    "seed_branch", "curvature", "curvature_control_sha256"}
                and exit_code == 0 and child.get("phase") == "finished"
                and _positive_branch(child.get("seed_branch"))
                and _valid_curvature(child.get("curvature"))
                and child.get("curvature_control_sha256") == gate.SELECTED_CONTROL_SHA256
                and "refusal" not in child)
    if status == "seed_branch_refused":
        return (set(child) == _COMMON_CHILD_KEYS | {"refusal"}
                and exit_code == 2 and child.get("phase") == "seed_branch"
                and child.get("refusal") in {
                    "_SeedNumericalRefusal: branch: minmod joint oracle left its strict smooth branch",
                    "_SeedNumericalRefusal: branch: selected seed margin is at or below 1e-4",
                })
    if status == "seed_curvature_refused":
        return (set(child) == _COMMON_CHILD_KEYS | {"seed_branch", "refusal"}
                and exit_code == 2 and child.get("phase") == "seed_curvature"
                and _positive_branch(child.get("seed_branch"))
                and isinstance(child.get("refusal"), str)
                and child["refusal"].startswith("_SeedNumericalRefusal: curvature:"))
    return False


def _archive_matches_child(child: dict[str, Any]) -> bool:
    try:
        archive = gate.ARCHIVE
        manifest_path = archive / "manifest.json"
        if gate._sha(manifest_path) != gate.ARCHIVE_MANIFEST_SHA256:
            return False
        manifest = json.loads(manifest_path.read_text())
        if any(gate._sha(archive / name) != manifest["file_sha256"][
            str((archive / name).relative_to(ROOT))] for name in gate.RAW_FILES):
            return False
        prior = json.loads((archive / "partial_sector_root.json").read_text())
        selected = gate.select_seed(prior)
        row = selected["selected"]
        return (prior["input_before"] == prior["input_after"] == child["input_before"]
                and prior["source_before"] == prior["source_after"]
                == manifest["source_before_after_sha256"]
                and row["candidate_control_sha256"] == child["selected_control_sha256"]
                and row["branch"]["signature_sha256"] == child["selected_signature_sha256"]
                and all(math.isclose(child[name], row[name], rel_tol=1e-10, abs_tol=1e-12)
                        for name in ("objective", "gradient_norm", "gradient_max"))
                and ("seed_branch" not in child or all(
                    math.isclose(child["seed_branch"][name], row["branch"][name],
                                 rel_tol=1e-10, abs_tol=1e-12)
                    for name in ("minimum_scaled_slope_margin",
                                 "minimum_scaled_face_flux_margin"))))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def run(directory: Path, *, expected_plan_sha256: str,
        expected_probe_sha256: str,
        expected_runner_sha256: str) -> dict[str, Any]:
    if (directory.resolve(strict=False).is_relative_to(gate.ARCHIVE.resolve(strict=True))
            or directory.is_symlink()):
        raise ValueError("alternate seed output directory must be outside raw archive")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("alternate seed gate output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "alternate_seed.json"
    command = [
        sys.executable,
        str(ROOT / "examples/weather_scenarios/fv_partial_alternate_seed_gate.py"),
        "--output", str(output),
        "--expected-plan-sha256", expected_plan_sha256,
        "--expected-probe-sha256", expected_probe_sha256,
    ]
    source_before = gate._sources()
    if (gate._sha(gate.PLAN) != expected_plan_sha256
            or gate._sha(Path(gate.__file__)) != expected_probe_sha256
            or gate._sha(Path(__file__)) != expected_runner_sha256
            or source_before["examples/weather_scenarios/fv_partial_alternate_seed_runner.py"]
            != expected_runner_sha256):
        raise ValueError("reviewed alternate seed source or plan changed before child")
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "alternate_seed.resource.json",
        log_path=directory / "alternate_seed.log",
    )
    child: dict[str, Any] | None = None
    read_error = None
    try:
        parsed = json.loads(output.read_text())
        if isinstance(parsed, dict):
            child = parsed
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
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
        child is not None
        and child.get("numerical_status") in VALID_STATUSES
        and child.get("pid") == resource.get("child_pid")
        and child.get("archive_manifest_sha256") == gate.ARCHIVE_MANIFEST_SHA256
        and child.get("plan_sha256") == expected_plan_sha256 == gate._sha(gate.PLAN)
        and child.get("reviewed_probe_sha256") == expected_probe_sha256 == gate._sha(Path(gate.__file__))
        and gate._sha(Path(__file__)) == expected_runner_sha256
        and child.get("source_before") == child.get("source_after") == source_before == gate._sources()
        and child["source_before"].get(
            "examples/weather_scenarios/fv_partial_alternate_seed_runner.py")
        == expected_runner_sha256
        and child.get("input_before") == child.get("input_after")
        and _archive_matches_child(child)
        and _valid_child(child, resource.get("exit_code"))
    )
    execution_status = (
        "resource_limited" if resource.get("resource_termination") is not None else
        "completed" if resource_ok and child_ok else "failed"
    )
    result = {
        "execution_status": execution_status,
        "numerical_status": child.get("numerical_status") if child is not None else "not_reached",
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "resource": resource,
        "child_read_error": read_error,
        "scope": "one post-hoc selected seed branch/Hessian audit; no GN/root refinement/response",
    }
    (directory / "alternate_seed.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory,
                 expected_plan_sha256=arguments.expected_plan_sha256,
                 expected_probe_sha256=arguments.expected_probe_sha256,
                 expected_runner_sha256=arguments.expected_runner_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
