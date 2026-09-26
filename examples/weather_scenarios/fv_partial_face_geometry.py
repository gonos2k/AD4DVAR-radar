"""Read-only face-flux geometry for the archived two-hole root search."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport
from examples.weather_scenarios import fv_partial_sector_root_probe as prior
from tests.test_fv_research_partial_observation import _problem


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_PARTIAL_FACE_GEOMETRY_PLAN.md"
ARCHIVE = EVIDENCE / "partial_sector_root_attempt1"
ARCHIVE_MANIFEST_SHA256 = "d80362f8acbb0feb1c82d60d5e9cde61a4863564d7f8f64e49467ba39d12cddd"
STATUS_ONLY_SOURCES = frozenset({
    "examples/weather_scenarios/fv_partial_sector_root_probe.py",
    "examples/weather_scenarios/fv_partial_sector_root_runner.py",
})
RAW_FILES = (
    "partial_sector_root.json", "partial_sector_root.run.json",
    "partial_sector_root.resource.json", "partial_sector_root.log",
)
SOURCE_PATHS = tuple(dict.fromkeys((
    *prior.SOURCE_PATHS,
    "examples/weather_scenarios/fv_partial_face_geometry.py",
)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _face_geometry(control: torch.Tensor, spec: Any, interval_minutes: float) -> dict[str, Any]:
    limits = spec.coefficient_limits
    coefficients = transport.bounded_fv_coefficients(
        control[-(limits.numel() + 1):-1],
        psi_basis=spec.psi_basis, coefficient_limits=limits,
        dt_seconds=interval_minutes * 60.0 / spec.substeps_per_interval,
        spacing_yx=spec.spacing_yx, reconstruction=spec.reconstruction,
        max_courant=spec.max_courant,
    )
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    qx, qy = transport.face_volume_fluxes(psi)
    flux = torch.cat((qx.flatten(), qy.flatten()))
    absolute = flux.abs()
    minimum = float(absolute.min())
    maximum = float(absolute.max())
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("face flux extrema must be finite with positive maximum")
    tie_tolerance = 128 * torch.finfo(torch.float64).eps * maximum
    minimizers = torch.nonzero(absolute <= minimum + tie_tolerance).flatten().tolist()
    index = int(absolute.argmin())
    if index < qx.numel():
        face = {"kind": "qx", "index": list(divmod(index, qx.shape[1]))}
    else:
        face = {"kind": "qy", "index": list(divmod(index - qx.numel(), qy.shape[1]))}
    return {
        "coefficient_values": coefficients.tolist(),
        "minimum_absolute_flux": minimum,
        "maximum_absolute_flux": maximum,
        "ratio": minimum / maximum,
        "minimizing_face": face if len(minimizers) == 1 else None,
        "minimizer_count_at_roundoff_scale": len(minimizers),
        "minimum_signed_flux": float(flux[index]) if len(minimizers) == 1 else None,
        "minimum_sign": (int(torch.sign(flux[index])) if len(minimizers) == 1 else None),
    }


def run(output: Path, *, expected_plan_sha256: str,
        expected_probe_sha256: str) -> dict[str, Any]:
    archive_root = ARCHIVE.resolve(strict=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if (output.resolve(strict=False).is_relative_to(archive_root)
            or temporary.resolve(strict=False).is_relative_to(archive_root)
            or output.exists() or output.is_symlink()
            or temporary.exists() or temporary.is_symlink()):
        raise ValueError("face-geometry output must be fresh and outside the raw archive")
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
    child = json.loads((ARCHIVE / RAW_FILES[0]).read_text())
    parent = json.loads((ARCHIVE / RAW_FILES[1]).read_text())
    if (child.get("numerical_status") != "root_refused"
            or parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != "root_refused"
            or child.get("response_validation") != "not_performed"
            or child.get("source_before") != child.get("source_after")
            or child.get("source_before") != manifest["source_before_after_sha256"]):
        raise ValueError("archived refusal status changed")
    source_before, plan_hash = _sources(), _sha(PLAN)
    if (plan_hash != expected_plan_sha256
            or source_before["examples/weather_scenarios/fv_partial_face_geometry.py"]
            != expected_probe_sha256):
        raise ValueError("reviewed geometry plan or probe source changed before run")
    for name, digest in child["source_before"].items():
        if name not in STATUS_ONLY_SOURCES and _sha(ROOT / name) != digest:
            raise ValueError(f"archived geometric source changed: {name}")
    input_before = prior._preflight_identity()
    if child["input_before"] != child["input_after"] or child["input_before"] != input_before:
        raise ValueError("archived and current fixed inputs differ")
    problem, _, _ = _problem()
    spec = problem.frozen.fv_transport
    if spec is None or problem.identity != input_before["current_problem_identity"]:
        raise ValueError("current fixed FV transport identity changed")
    expected_stages = problem.layout["euler_stages"]
    if expected_stages != 54:
        raise ValueError("fixed case RK stage count changed")
    control_rows: list[tuple[str, dict[str, Any], list[float], str]] = [
        ("gn_seed", child["seed_branch"], child["gauss_newton"]["control"],
         child["gauss_newton"]["control_sha256"]),
    ]
    for trial in child["trial_records"]:
        control_rows.append((
            f"trial_{trial['iteration']}_{trial['backtrack']}",
            trial["branch"], trial["candidate_control"],
            trial["candidate_control_sha256"],
        ))
    if len(control_rows) != 53:
        raise ValueError("archived candidate count changed")
    entries = []
    for label, branch, values, digest in control_rows:
        control = torch.tensor(values, dtype=torch.float64)
        if control.shape != (26,) or not bool(torch.isfinite(control).all()):
            raise ValueError("archived control is not a finite 26-vector")
        if prior._tensor_sha(control) != digest:
            raise ValueError("archived control hash mismatch")
        geometry = _face_geometry(control, spec, problem.frozen.nowcast_config.interval_minutes)
        if (branch["euler_stages"] != expected_stages
                or abs(geometry["ratio"] - branch["minimum_scaled_face_flux_margin"]) > 1e-12):
            raise ValueError("recomputed face ratio disagrees with strict trace")
        entries.append({"label": label, "control_sha256": digest,
                        "signature_sha256": branch["signature_sha256"],
                        "reported_ratio": branch["minimum_scaled_face_flux_margin"],
                        "ratio_absolute_difference": abs(
                            geometry["ratio"] - branch["minimum_scaled_face_flux_margin"]),
                        **geometry})
    accepted = [trial for trial in child["trial_records"] if trial.get("accepted") is True]
    if len(accepted) != 8:
        raise ValueError("archived accepted trial count changed")
    previous_accepted = entries[0]
    for entry, trial in zip(entries[1:], child["trial_records"]):
        same_face = entry["minimizing_face"] == previous_accepted["minimizing_face"]
        entry.update({
            "iteration": trial["iteration"], "backtrack": trial["backtrack"],
            "accepted": trial["accepted"], "rejection": trial.get("rejection"),
            "policy_reason": trial.get("policy_reason"),
            "objective": trial.get("objective"),
            "gradient_norm": trial.get("gradient_norm"),
            "gradient_max": trial.get("gradient_max"),
            "signature_changed_from_preceding_accepted": (
                entry["signature_sha256"] != previous_accepted["signature_sha256"]),
            "minimum_face_sign_changed_from_preceding_accepted": (
                entry["minimum_sign"] != previous_accepted["minimum_sign"]
                if same_face and entry["minimum_sign"] is not None else None),
        })
        if trial["accepted"]:
            previous_accepted = entry
    face_counts = Counter(
        f"{entry['minimizing_face']['kind']}:{entry['minimizing_face']['index']}"
        if entry["minimizing_face"] is not None else "ambiguous" for entry in entries)
    report = {
        "status": "geometry_only",
        "scope": "archived 52 trials and GN seed; current-source face-flux derivation only",
        "archive_manifest_sha256": ARCHIVE_MANIFEST_SHA256,
        "plan_sha256": plan_hash,
        "reviewed_probe_sha256": expected_probe_sha256,
        "source_before": source_before,
        "source_after": _sources(),
        "input_before": input_before,
        "input_after": prior._preflight_identity(),
        "candidate_count": len(entries) - 1,
        "accepted_count": len(accepted),
        "minimum_face_counts": dict(face_counts),
        "accepted_same_min_face_sign_crossings": sum(
            entry["minimum_face_sign_changed_from_preceding_accepted"] is True
            for entry in entries[1:] if entry["accepted"]),
        "rejected_same_min_face_sign_crossings": sum(
            entry["minimum_face_sign_changed_from_preceding_accepted"] is True
            for entry in entries[1:] if not entry["accepted"]),
        "maximum_ratio_absolute_difference": max(
            entry["ratio_absolute_difference"] for entry in entries),
        "entries": entries,
        "elapsed_seconds": time.monotonic() - started,
        "pid": os.getpid(),
        "new_gn_root_adjoint_reanalysis_runs": 0,
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
    }
    if (report["source_before"] != report["source_after"]
            or report["input_before"] != report["input_after"]
            or _sha(PLAN) != plan_hash
            or _sha(Path(__file__)) != expected_probe_sha256
            or _sha(manifest_path) != ARCHIVE_MANIFEST_SHA256
            or any(_sha(ARCHIVE / name) != manifest["file_sha256"][
                str((ARCHIVE / name).relative_to(ROOT))] for name in RAW_FILES)):
        raise ValueError("source, fixed input, plan or raw archive changed during geometry audit")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
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
    print(json.dumps({key: result[key] for key in (
        "status", "candidate_count", "accepted_count", "minimum_face_counts",
        "maximum_ratio_absolute_difference", "elapsed_seconds",
    )}, indent=2, sort_keys=True))
