"""One archived alternate-sector seed branch and exact-curvature gate."""
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
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_partial_sector_root_probe as prior
from examples.weather_scenarios import fv_partial_reanalysis_preflight as preflight
from examples.weather_scenarios import fv_point_basin_probe as basin
from tests.test_fv_research_partial_observation import _problem


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_PARTIAL_ALTERNATE_SEED_GATE_PLAN.md"
ARCHIVE = EVIDENCE / "partial_sector_root_attempt1"
ARCHIVE_MANIFEST_SHA256 = "d80362f8acbb0feb1c82d60d5e9cde61a4863564d7f8f64e49467ba39d12cddd"
PRIOR_STATUS_SOURCE_SHA256 = "fddc3ddb20a3a961f73eb2d40dd379b25e87cd6109ca836476f45a1408350ccc"
PRIOR_RUNNER_STATUS_SOURCE_SHA256 = "2bc6607a1bc4c369cd11edbd359d7a3754352dc7df2b7377944811161e37b18e"
STATUS_ONLY_SOURCES = frozenset({
    "examples/weather_scenarios/fv_partial_sector_root_probe.py",
    "examples/weather_scenarios/fv_partial_sector_root_runner.py",
})
RAW_FILES = (
    "partial_sector_root.json", "partial_sector_root.run.json",
    "partial_sector_root.resource.json", "partial_sector_root.log",
)
SELECTED_CONTROL_SHA256 = "125e0200fdf85f996715ba1bc04aa3f631333614bf15cf2c16630b94c1897421"
SELECTED_SIGNATURE_SHA256 = "50d3b1a4bad6761b14806c62708400d87d16e506ab08ef545f7ba9d870bece6d"
SOURCE_PATHS = tuple(dict.fromkeys((
    *prior.SOURCE_PATHS,
    "examples/weather_scenarios/fv_partial_alternate_seed_gate.py",
    "examples/weather_scenarios/fv_partial_alternate_seed_runner.py",
)))


class _SeedNumericalRefusal(ValueError):
    """A declared branch or exact-Hessian failure at the selected seed."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def select_seed(child: dict[str, Any]) -> dict[str, Any]:
    """Rank only archived cross-signature candidates above both margins."""
    preceding_signature = child["seed_branch"]["signature_sha256"]
    eligible: list[dict[str, Any]] = []
    for trial in child["trial_records"]:
        branch = trial["branch"]
        maximum = trial["gradient_max"]
        control = torch.tensor(trial["candidate_control"], dtype=torch.float64)
        if control.shape != (26,) or not bool(torch.isfinite(control).all()):
            raise ValueError("archived trial control is not a finite 26-vector")
        if prior._tensor_sha(control) != trial["candidate_control_sha256"]:
            raise ValueError("archived trial control hash mismatch")
        if (branch["signature_sha256"] != preceding_signature
                and type(maximum) in (int, float) and math.isfinite(maximum)
                and maximum >= 0
                and branch["minimum_scaled_slope_margin"] > 1e-4
                and branch["minimum_scaled_face_flux_margin"] > 1e-4):
            eligible.append(trial)
        if trial["accepted"]:
            preceding_signature = branch["signature_sha256"]
    if len(eligible) != 29:
        raise ValueError("archived alternate-sector eligible set changed")
    eligible.sort(key=lambda trial: (
        trial["gradient_max"],
        -min(trial["branch"]["minimum_scaled_slope_margin"],
             trial["branch"]["minimum_scaled_face_flux_margin"]),
        trial["iteration"], trial["backtrack"],
    ))
    selected = eligible[0]
    if (selected["iteration"] != 5 or selected["backtrack"] != 3
            or selected["candidate_control_sha256"] != SELECTED_CONTROL_SHA256
            or selected["branch"]["signature_sha256"] != SELECTED_SIGNATURE_SHA256
            or selected["accepted"] is not False
            or selected.get("policy_reason") != "merit_switch_refused"):
        raise ValueError("predeclared alternate-sector seed changed")
    return {"eligible_count": len(eligible), "selected": selected}


def _close(name: str, measured: float, archived: float) -> None:
    if not (math.isfinite(measured) and math.isfinite(archived)
            and math.isclose(measured, archived, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError(f"selected seed {name} differs from archived measurement")


def run(output: Path, *, expected_plan_sha256: str,
        expected_probe_sha256: str) -> dict[str, Any]:
    archive_root = ARCHIVE.resolve(strict=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if (output.resolve(strict=False).is_relative_to(archive_root)
            or temporary.resolve(strict=False).is_relative_to(archive_root)
            or output.exists() or output.is_symlink()
            or temporary.exists() or temporary.is_symlink()):
        raise ValueError("alternate seed output must be fresh and outside raw archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest_path = ARCHIVE / "manifest.json"
    if _sha(manifest_path) != ARCHIVE_MANIFEST_SHA256:
        raise ValueError("archived root-attempt manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for name in RAW_FILES:
        path = ARCHIVE / name
        if _sha(path) != manifest["file_sha256"][str(path.relative_to(ROOT))]:
            raise ValueError(f"archived root-attempt raw file changed: {name}")
    child = json.loads((ARCHIVE / "partial_sector_root.json").read_text())
    if (child.get("numerical_status") != "root_refused"
            or child.get("source_before") != child.get("source_after")
            or child.get("source_before") != manifest["source_before_after_sha256"]):
        raise ValueError("archived root-attempt source or status changed")
    selected = select_seed(child)
    row = selected["selected"]
    source_before, plan_hash = _sources(), _sha(PLAN)
    if (plan_hash != expected_plan_sha256
            or source_before["examples/weather_scenarios/fv_partial_alternate_seed_gate.py"]
            != expected_probe_sha256
            or source_before["examples/weather_scenarios/fv_partial_sector_root_probe.py"]
            != PRIOR_STATUS_SOURCE_SHA256
            or source_before["examples/weather_scenarios/fv_partial_sector_root_runner.py"]
            != PRIOR_RUNNER_STATUS_SOURCE_SHA256):
        raise ValueError("reviewed alternate seed plan or probe changed")
    for name, digest in child["source_before"].items():
        if name not in STATUS_ONLY_SOURCES and _sha(ROOT / name) != digest:
            raise ValueError(f"archived objective or geometric source changed: {name}")
    input_before = prior._preflight_identity()
    if child["input_before"] != child["input_after"] or child["input_before"] != input_before:
        raise ValueError("archived and current fixed input differ")
    problem, _, parameters = _problem()
    if problem.identity != input_before["current_problem_identity"]:
        raise ValueError("current fixed problem identity changed")
    control = torch.tensor(row["candidate_control"], dtype=torch.float64)
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "seed_branch", "numerical_status": "running",
        "scope": "post-hoc archived alternate-sector seed gate only; no GN/refiner/response",
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "archive_manifest_sha256": ARCHIVE_MANIFEST_SHA256,
        "plan_sha256": plan_hash, "reviewed_probe_sha256": expected_probe_sha256,
        "source_before": source_before, "input_before": input_before,
        "selected_iteration": row["iteration"],
        "selected_backtrack": row["backtrack"],
        "eligible_candidate_count": selected["eligible_count"],
        "selected_control_sha256": SELECTED_CONTROL_SHA256,
        "selected_signature_sha256": SELECTED_SIGNATURE_SHA256,
        "selected_previously_accepted": False,
        "selected_previous_policy_reason": row["policy_reason"],
        "new_product_gn_runs": 0, "new_newton_refinement_runs": 0,
        "new_adjoint_reanalysis_runs": 0,
    }

    def save() -> None:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        objective = float(problem.objective(control, parameters))
        gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
        if gradient.shape != control.shape or not bool(torch.isfinite(gradient).all()):
            raise ValueError("selected seed gradient is invalid")
        norm = float(torch.linalg.vector_norm(gradient))
        maximum = float(gradient.abs().max())
        _close("objective", objective, row["objective"])
        _close("gradient_norm", norm, row["gradient_norm"])
        _close("gradient_max", maximum, row["gradient_max"])
        report.update(objective=objective, gradient_norm=norm, gradient_max=maximum)
        try:
            branch, _, face = preflight.branch_with_face_margin(problem, control, parameters)
        except ValueError as error:
            if str(error) != "minmod joint oracle left its strict smooth branch":
                raise
            raise _SeedNumericalRefusal(f"branch: {error}") from error
        summary = prior._branch_summary({**branch, "minimum_scaled_face_flux_margin": face})
        if (summary["euler_stages"] != 54
                or summary["signature_sha256"] != SELECTED_SIGNATURE_SHA256):
            raise ValueError("selected seed full branch signature changed")
        _close("slope_margin", summary["minimum_scaled_slope_margin"],
               row["branch"]["minimum_scaled_slope_margin"])
        _close("face_margin", face, row["branch"]["minimum_scaled_face_flux_margin"])
        if (summary["minimum_scaled_slope_margin"] <= 1e-4
                or summary["minimum_scaled_face_flux_margin"] <= 1e-4):
            raise _SeedNumericalRefusal("branch: selected seed margin is at or below 1e-4")
        report["seed_branch"] = summary
        report["phase"] = "seed_curvature"
        save()
        try:
            curvature = basin._hessian_audit(problem, control, parameters)
        except ValueError as error:
            if not str(error).startswith("basin terminal exact "):
                raise
            raise _SeedNumericalRefusal(f"curvature: {error}") from error
        report.update(phase="finished", numerical_status="seed_locally_spd",
                      curvature=curvature, curvature_control_sha256=SELECTED_CONTROL_SHA256)
    except _SeedNumericalRefusal as error:
        report.update(numerical_status=("seed_branch_refused" if report["phase"] == "seed_branch"
                                        else "seed_curvature_refused"),
                      refusal=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = prior._preflight_identity()
        report["raw_unchanged"] = (
            _sha(manifest_path) == ARCHIVE_MANIFEST_SHA256
            and all(_sha(ARCHIVE / name) == manifest["file_sha256"][
                str((ARCHIVE / name).relative_to(ROOT))] for name in RAW_FILES))
        if (report["source_before"] != report["source_after"]
                or report["input_before"] != report["input_after"]
                or _sha(PLAN) != plan_hash or _sha(Path(__file__)) != expected_probe_sha256
                or not report["raw_unchanged"]):
            report.update(phase="identity_recheck", numerical_status="identity_error")
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    arguments = parser.parse_args()
    result = run(arguments.output,
                 expected_plan_sha256=arguments.expected_plan_sha256,
                 expected_probe_sha256=arguments.expected_probe_sha256)
    print(json.dumps({key: result.get(key) for key in (
        "phase", "numerical_status", "gradient_max", "curvature", "refusal",
        "elapsed_seconds",
    )}, indent=2, sort_keys=True))
    raise SystemExit(0 if result["numerical_status"] == "seed_locally_spd" else 2)
