"""Measure finite-offset objective gradients tangent to an archived FV face event.

This diagnostic reuses the fixed PR209 controls and PR225 finite side samples.
It differentiates the smooth production face-flux map at each event and the
objective only at the eight archived off-event controls. It makes no response
or stationarity claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport
from examples.weather_scenarios import fv_partial_face_event_probe as face_event
from examples.weather_scenarios.fv86_resource_runner import run_guarded
from tests.test_fv_research_partial_observation import _problem

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_PARTIAL_FACE_TANGENT_PLAN.md"
PLAN_SHA256 = "1aac5f48f2dd02cf303131a7a42fbea1c8b69e87c0cc12943516b47434d9a900"
PR225_CHILD = EVIDENCE / "partial_face_event_attempt1/partial_face_event.json"
PR225_CHILD_SHA256 = "7b0798414c2f7036189db0afef5cbf101b53c8903c29acf4079b4de0f2b0ca1d"
PR225_EVIDENCE = EVIDENCE / "FV_PARTIAL_FACE_EVENT_ATTEMPT1_EVIDENCE.json"
PR225_EVIDENCE_SHA256 = "df72467af26217534a59435c9f689b337d2d2f96cf13c48c078b803af9f6bec4"
ARCHIVE = face_event.ARCHIVE
ARCHIVE_MANIFEST_SHA256 = "d80362f8acbb0feb1c82d60d5e9cde61a4863564d7f8f64e49467ba39d12cddd"
RAW_FILES = face_event.RAW_FILES
FACE_INDEX = (2, 0)
SIDE_OFFSETS = (2.0**-8, 2.0**-10)
SAMPLE_METADATA_FIELDS = (
    "chord", "side", "offset", "t", "control_sha256", "status",
    "response_margin_qualified", "branch_scope", "branch_signature",
    "branch_signature_sha256", "euler_stages", "minimum_scaled_face_flux_margin",
    "minimum_scaled_slope_margin", "face_margins_by_stage", "gradient_l2", "gradient_inf",
)
WALL_SECONDS = 120
SAMPLED_RSS_BYTES = 1024**3
SELF_PATH = "examples/weather_scenarios/fv_partial_face_tangent_probe.py"
RUNNER_PATH = "examples/weather_scenarios/fv86_resource_runner.py"
RUNNER_SHA256 = "6242598d8b11a74ed44fcd419d83fd14cfcd637f34ee4746e1f5432554aeeb6d"
SOURCE_PATHS = tuple(dict.fromkeys((*face_event._sources().keys(), RUNNER_PATH, SELF_PATH)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _require_runner_pin(sources: dict[str, str]) -> None:
    if sources.get(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError("guard runner source differs from its reviewed pin")


def _block_norms(value: Tensor) -> dict[str, float]:
    if value.shape != (26,):
        raise ValueError("optimizer control vectors must have 26 entries")
    return {
        "field_20": float(torch.linalg.vector_norm(value[:20])),
        "flow_5": float(torch.linalg.vector_norm(value[20:25])),
        "growth_1": float(torch.linalg.vector_norm(value[25:])),
    }


def tangent_gradient(gradient: Tensor, normal: Tensor) -> dict[str, Any]:
    """Project one 26-control gradient onto the Euclidean event tangent plane."""
    if gradient.shape != (26,) or normal.shape != (26,):
        raise ValueError("gradient and face normal must have 26 entries")
    if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(normal).all()):
        raise ValueError("gradient and face normal must be finite")
    normal_norm = torch.linalg.vector_norm(normal)
    if not bool(torch.isfinite(normal_norm)) or bool(normal_norm <= 0):
        raise ValueError("face normal must have finite nonzero norm")
    unit_normal = normal / normal_norm
    slope = torch.dot(gradient, unit_normal)
    tangent = gradient - slope * unit_normal
    return {
        "gradient": gradient.tolist(),
        "gradient_l2": float(torch.linalg.vector_norm(gradient)),
        "gradient_inf": float(gradient.abs().max()),
        "unit_normal_slope": float(slope),
        "tangent_gradient": tangent.tolist(),
        "tangent_l2": float(torch.linalg.vector_norm(tangent)),
        "tangent_block_l2": _block_norms(tangent),
    }


def segment_minimum(left: Tensor, right: Tensor) -> dict[str, Any]:
    """Return the minimum-norm point on one sampled gradient segment."""
    if left.shape != (26,) or right.shape != (26,):
        raise ValueError("sample gradients must have 26 entries")
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("sample gradients must be finite")
    difference = right - left
    difference_norm = torch.linalg.vector_norm(difference)
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(
        1.0, float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right))
    )
    if float(difference_norm) <= tolerance:
        return {
            "status": "degenerate_pair",
            "alpha": None,
            "minimum_norm": None,
            "difference_norm": float(difference_norm),
            "degeneracy_tolerance": tolerance,
        }
    alpha = torch.clamp(-torch.dot(left, difference) / difference.square().sum(), 0.0, 1.0)
    point = left + alpha * difference
    return {
        "status": "sampled_segment_minimum",
        "alpha": float(alpha),
        "minimum_norm": float(torch.linalg.vector_norm(point)),
        "difference_norm": float(difference_norm),
        "degeneracy_tolerance": tolerance,
    }


def _event_face(control: Tensor, spec: Any, interval_minutes: float) -> tuple[Tensor, Tensor]:
    limits = spec.coefficient_limits
    coefficients = transport.bounded_fv_coefficients(
        control[-(limits.numel() + 1):-1], psi_basis=spec.psi_basis,
        coefficient_limits=limits,
        dt_seconds=interval_minutes * 60.0 / spec.substeps_per_interval,
        spacing_yx=spec.spacing_yx, reconstruction=spec.reconstruction,
        max_courant=spec.max_courant,
    )
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    qx, qy = transport.face_volume_fluxes(psi)
    return qy[FACE_INDEX], torch.cat((qx.reshape(-1), qy.reshape(-1)))


def _validate_pr225() -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha(PLAN) != PLAN_SHA256:
        raise ValueError("frozen tangent plan changed")
    if _sha(PR225_CHILD) != PR225_CHILD_SHA256:
        raise ValueError("PR225 child raw hash changed")
    if _sha(PR225_EVIDENCE) != PR225_EVIDENCE_SHA256:
        raise ValueError("PR225 evidence hash changed")
    child = json.loads(PR225_CHILD.read_text())
    evidence = json.loads(PR225_EVIDENCE.read_text())
    files = evidence.get("files_sha256", {})
    if (files.get(str(PR225_CHILD.relative_to(ROOT))) != PR225_CHILD_SHA256
            or child.get("plan_sha256") != face_event.PLAN_SHA256
            or child.get("archive_manifest_sha256") != ARCHIVE_MANIFEST_SHA256
            or child.get("status") != "face_event_diagnostic_only"
            or child.get("response_validation") != "not_performed"
            or child.get("response_claim") is not False
            or child.get("root_claim") is not False
            or child.get("event_derivatives_evaluated") is not False
            or child.get("new_gn_root_adjoint_reanalysis_runs") != 0
            or child.get("off_event_sample_count") != 8
            or len(child.get("chords", [])) != 2
            or len(child.get("off_event_samples", [])) != 8):
        raise ValueError("PR225 child/evidence contract changed")
    return child, evidence


def _archived_controls(child: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, archived_pr209 = face_event._check_archive()
    if _sha(ARCHIVE / "manifest.json") != ARCHIVE_MANIFEST_SHA256:
        raise ValueError("PR209 archive manifest changed")
    chords = face_event._accepted_chords(archived_pr209)
    pr225_chords = child["chords"]
    if len(chords) != 2 or len(pr225_chords) != 2:
        raise ValueError("exactly two archived chords are required")
    controls: list[dict[str, Any]] = []
    samples = child["off_event_samples"]
    if len(samples) != 8:
        raise ValueError("exactly eight PR225 samples are required")
    for index, (chord, event_chord) in enumerate(zip(chords, pr225_chords, strict=True)):
        if event_chord.get("label") != chord["label"]:
            raise ValueError("PR225 chord ordering differs from PR209 archive")
        start, end = chord["start"], chord["end"]
        direction = end - start
        event_t = event_chord.get("event_parameter")
        if (type(event_t) not in (int, float) or not math.isfinite(event_t)
                or not 0.0 < event_t < 1.0):
            raise ValueError("PR225 event coordinate is invalid")
        controls.append({**chord, "event_t": float(event_t), "direction": direction})
        expected_points = [
            (side, offset, event_t + sign * offset)
            for side, sign in (("left", -1.0), ("right", 1.0))
            for offset in SIDE_OFFSETS
        ]
        rows = samples[index * 4:(index + 1) * 4]
        for row, (side, offset, t) in zip(rows, expected_points, strict=True):
            if (row.get("chord") != chord["label"] or row.get("side") != side
                    or row.get("offset") != offset or row.get("t") != t
                    or row.get("status") != "finite_off_event_derivatives"
                    or row.get("finite") is not True
                    or row.get("response_margin_qualified") is not False):
                raise ValueError("PR225 sample identity or eligibility status changed")
            control = start + float(t) * direction
            if _tensor_sha(control) != row.get("control_sha256"):
                raise ValueError("reconstructed PR225 control tensor hash changed")
            row["_control"] = control
    if (child.get("input_before") != child.get("input_after")
            or child.get("source_before") != child.get("source_after")):
        raise ValueError("PR225 before/after inputs or sources differ")
    return controls, samples


def _face_normal(
    control: Tensor, spec: Any, interval_minutes: float,
) -> tuple[Tensor, float, float, float]:
    control_for_ad = control.detach().clone().requires_grad_(True)
    face, all_flux = _event_face(control_for_ad, spec, interval_minutes)
    # Keep the production face scalar as a Tensor through autograd.
    normal = torch.autograd.grad(face, control_for_ad)[0]
    normal_norm = torch.linalg.vector_norm(normal)
    flux_scale = all_flux.detach().abs().max()
    threshold = 128.0 * torch.finfo(torch.float64).eps * max(1.0, float(normal_norm))
    if (not bool(torch.isfinite(normal).all()) or not bool(torch.isfinite(normal_norm))
            or float(normal_norm) <= threshold or not bool(torch.isfinite(flux_scale))):
        raise ValueError("event face normal is nonfinite or unresolved")
    return normal.detach(), float(normal_norm), float(flux_scale), float(face.detach())


def _gradient(
    problem: Any, control: Tensor, parameters: Tensor,
) -> Tensor:
    gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("off-event objective gradient is nonfinite")
    return gradient.detach()


def _check_gradient_scalars(gradient: Tensor, sample: dict[str, Any]) -> None:
    for key, actual in (("gradient_l2", torch.linalg.vector_norm(gradient)),
                        ("gradient_inf", gradient.abs().max())):
        reported = sample.get(key)
        if (not isinstance(reported, (float, int)) or isinstance(reported, bool)
                or not math.isfinite(reported)):
            raise ValueError(f"PR225 {key} scalar is missing or nonfinite")
        tolerance = 128.0 * torch.finfo(torch.float64).eps * max(1.0, abs(float(reported)))
        if abs(float(actual) - float(reported)) > tolerance:
            raise ValueError(f"recomputed {key} disagrees with PR225")


def _diagnostic(
    *, output: Path, expected_plan_sha256: str, expected_source_sha256: str,
) -> dict[str, Any]:
    started = time.monotonic()
    evidence_roots = (ARCHIVE.resolve(strict=True), PR225_CHILD.parent.resolve(strict=True))
    temporary = output.with_suffix(output.suffix + ".tmp")
    if (output.exists() or output.is_symlink() or temporary.exists() or temporary.is_symlink()
            or any(output.resolve(strict=False).is_relative_to(root) for root in evidence_roots)
            or any(temporary.resolve(strict=False).is_relative_to(root) for root in evidence_roots)):
        raise ValueError("diagnostic output must be fresh and outside archived evidence")
    if expected_plan_sha256 != PLAN_SHA256 or _sha(PLAN) != PLAN_SHA256:
        raise ValueError("reviewed tangent plan changed")
    if expected_source_sha256 != _sha(Path(__file__)):
        raise ValueError("reviewed tangent probe source changed")

    identity_paths = (PLAN, PR225_CHILD, PR225_EVIDENCE,
                      ARCHIVE / "manifest.json", *(ARCHIVE / name for name in RAW_FILES))
    identity_before_files = {_path_name(path): _sha(path) for path in identity_paths}
    source_before = _sources()
    _require_runner_pin(source_before)
    if source_before.get(SELF_PATH) != expected_source_sha256:
        raise ValueError("reviewed probe source SHA256 does not match child source")
    current_input_before = face_event.prior._preflight_identity()
    pr225, evidence = _validate_pr225()
    for name, digest in pr225.get("source_before", {}).items():
        if name not in face_event.STATUS_ONLY_SOURCES and _sha(ROOT / name) != digest:
            raise ValueError(f"PR225 source identity changed: {name}")
    if (pr225.get("input_before") != current_input_before
            or pr225.get("input_after") != current_input_before):
        raise ValueError("current fixed FV inputs differ from PR225")
    problem, _, parameters = _problem()
    if problem.identity != current_input_before["current_problem_identity"]:
        raise ValueError("current fixed FV problem identity changed")
    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("fixed research problem lacks FV transport")
    interval_minutes = problem.frozen.nowcast_config.interval_minutes
    chords, samples = _archived_controls(pr225)

    chord_results: list[dict[str, Any]] = []
    for chord in chords:
        event_control = chord["start"] + chord["event_t"] * chord["direction"]
        normal, normal_norm, flux_scale, face_value = _face_normal(
            event_control, spec, interval_minutes
        )
        chord_results.append({
            "label": chord["label"],
            "event_parameter": chord["event_t"],
            "event_control_sha256": _tensor_sha(event_control),
            "event_face_flux": face_value,
            "event_flux_scale_max_abs": flux_scale,
            "normal": normal.tolist(),
            "normal_l2": normal_norm,
            "normal_block_l2": _block_norms(normal),
            "flux_scale_max_abs": flux_scale,
            "normal_zero_guard": 128.0 * torch.finfo(torch.float64).eps
            * max(1.0, normal_norm),
        })

    gradient_records: list[dict[str, Any]] = []
    gradients_by_key: dict[tuple[str, str, float], Tensor] = {}
    for sample in samples:
        control = sample.pop("_control")
        chord = next(item for item in chord_results if item["label"] == sample["chord"])
        normal = torch.tensor(chord["normal"], dtype=torch.float64)
        gradient = _gradient(problem, control, parameters)
        _check_gradient_scalars(gradient, sample)
        decomposition = tangent_gradient(gradient, normal)
        record = {
            "chord": sample["chord"], "side": sample["side"],
            "offset": sample["offset"], "t": sample["t"],
            "control_sha256": sample["control_sha256"],
            "archived_status": sample["status"],
            "archived_response_margin_qualified": sample["response_margin_qualified"],
            "branch_scope": sample["branch_scope"],
            "branch_signature": sample["branch_signature"],
            "branch_signature_sha256": sample["branch_signature_sha256"],
            "euler_stages": sample["euler_stages"],
            "minimum_scaled_face_flux_margin": sample["minimum_scaled_face_flux_margin"],
            "minimum_scaled_slope_margin": sample["minimum_scaled_slope_margin"],
            "face_margins_by_stage": sample["face_margins_by_stage"],
            "gradient_block_l2": _block_norms(gradient),
            **decomposition,
        }
        gradient_records.append(record)
        gradients_by_key[(sample["chord"], sample["side"], sample["offset"])] = gradient

    hulls: list[dict[str, Any]] = []
    tangent_changes: list[dict[str, Any]] = []
    for chord in chords:
        label = chord["label"]
        by_side_offset: dict[tuple[str, float], dict[str, Any]] = {}
        for record in gradient_records:
            if record["chord"] == label:
                by_side_offset[(record["side"], record["offset"])] = record
        for offset in SIDE_OFFSETS:
            hulls.append({
                "chord": label, "offset": offset,
                **segment_minimum(
                    gradients_by_key[(label, "left", offset)],
                    gradients_by_key[(label, "right", offset)],
                ),
            })
        far, near = SIDE_OFFSETS
        for side in ("left", "right"):
            far_tangent = torch.tensor(
                by_side_offset[(side, far)]["tangent_gradient"], dtype=torch.float64
            )
            near_tangent = torch.tensor(
                by_side_offset[(side, near)]["tangent_gradient"], dtype=torch.float64
            )
            change = near_tangent - far_tangent
            tangent_changes.append({
                "chord": label, "side": side,
                "far_offset": far, "near_offset": near,
                "near_minus_far": change.tolist(),
                "change_l2": float(torch.linalg.vector_norm(change)),
            })

    input_after = json.loads(json.dumps(
        face_event.prior._preflight_identity(), allow_nan=False,
    ))
    source_after = _sources()
    identity_after_files = {_path_name(path): _sha(path) for path in identity_paths}
    if (input_after != current_input_before or source_after != source_before
            or identity_after_files != identity_before_files
            or source_after.get(SELF_PATH) != expected_source_sha256
            or source_after.get(RUNNER_PATH) != RUNNER_SHA256):
        raise ValueError("source, plan, PR209/225 evidence, or fixed input changed during diagnostic")

    report = {
        "status": "finite_offset_tangent_gradient_diagnostic_only",
        "scope": "two PR209 chords; PR225 archived event coordinates and eight finite controls",
        "plan_sha256": PLAN_SHA256,
        "probe_sha256": expected_source_sha256,
        "runner_sha256": RUNNER_SHA256,
        "pr225_child_sha256": PR225_CHILD_SHA256,
        "pr225_evidence_sha256": PR225_EVIDENCE_SHA256,
        "archive_manifest_sha256": ARCHIVE_MANIFEST_SHA256,
        "evidence_identity_before": identity_before_files,
        "evidence_identity_after": identity_after_files,
        "source_before": source_before,
        "source_after": source_after,
        "input_before": current_input_before,
        "input_after": input_after,
        "coordinate_metric": "Euclidean norm in the 26 existing optimizer control coordinates",
        "block_slices": {"field_20": [0, 20], "flow_5": [20, 25], "growth_1": [25, 26]},
        "event_normals": chord_results,
        "sample_count": len(gradient_records),
        "samples": gradient_records,
        "same_chord_sampled_gradient_hulls": hulls,
        "same_side_near_far_tangent_changes": tangent_changes,
        "sampled_hulls_scope": "finite-offset sampled gradients; not a Clarke subdifferential or limiting-gradient convergence proof",
        "event_objective_or_trajectory_derivatives_evaluated": False,
        "response_validation": "not_performed",
        "response_eligibility": "ineligible_archived_face_margin_below_1e-4",
        "physical_validation": "not_performed",
        "root_claim": False,
        "response_claim": False,
        "elapsed_seconds": time.monotonic() - started,
        "pid": os.getpid(),
    }
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


def _path_name(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sample_metadata(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: sample[key] for key in SAMPLE_METADATA_FIELDS} for sample in samples]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _valid_resource(resource: Any, command: list[str]) -> bool:
    return (
        isinstance(resource, dict) and resource.get("command") == command
        and type(resource.get("child_pid")) is int and resource["child_pid"] > 0
        and resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
        and type(resource.get("elapsed_seconds")) in (int, float)
        and math.isfinite(resource["elapsed_seconds"])
        and 0 <= resource["elapsed_seconds"] <= WALL_SECONDS
    )


def _finite_vector(value: Any, size: int) -> bool:
    return (
        isinstance(value, list) and len(value) == size
        and all(type(item) in (int, float) and math.isfinite(item) for item in value)
    )


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _close_number(actual: Any, expected: float) -> bool:
    return _finite_number(actual) and abs(float(actual) - expected) <= (
        128.0 * torch.finfo(torch.float64).eps * max(1.0, abs(expected))
    )


def _close_vector(actual: Any, expected: Tensor) -> bool:
    if not _finite_vector(actual, expected.numel()):
        return False
    return bool(torch.all(
        (torch.tensor(actual, dtype=torch.float64) - expected).abs()
        <= 128.0 * torch.finfo(torch.float64).eps * torch.maximum(
            torch.ones_like(expected), expected.abs(),
        )
    ))


def _close_blocks(actual: Any, expected: dict[str, float]) -> bool:
    return isinstance(actual, dict) and set(actual) == set(expected) and all(
        _close_number(actual.get(key), value) for key, value in expected.items()
    )


def _valid_child(
    child: Any,
    *,
    expected_source_sha256: str,
    expected_sources: dict[str, str],
    expected_input: dict[str, Any],
    expected_evidence: dict[str, str],
    expected_child_pid: int,
    expected_event_metadata: list[dict[str, Any]],
    expected_sample_metadata: list[dict[str, Any]],
) -> bool:
    if not isinstance(child, dict):
        return False
    samples = child.get("samples")
    normals = child.get("event_normals")
    hulls = child.get("same_chord_sampled_gradient_hulls")
    changes = child.get("same_side_near_far_tangent_changes")
    if not (
        child.get("status") == "finite_offset_tangent_gradient_diagnostic_only"
        and child.get("scope") == "two PR209 chords; PR225 archived event coordinates and eight finite controls"
        and child.get("plan_sha256") == PLAN_SHA256
        and child.get("probe_sha256") == expected_source_sha256
        and child.get("runner_sha256") == RUNNER_SHA256
        and type(expected_child_pid) is int and expected_child_pid > 0
        and child.get("pid") == expected_child_pid
        and child.get("pr225_child_sha256") == PR225_CHILD_SHA256
        and child.get("pr225_evidence_sha256") == PR225_EVIDENCE_SHA256
        and child.get("archive_manifest_sha256") == ARCHIVE_MANIFEST_SHA256
        and child.get("event_objective_or_trajectory_derivatives_evaluated") is False
        and child.get("root_claim") is False and child.get("response_claim") is False
        and child.get("response_validation") == "not_performed"
        and child.get("response_eligibility") == "ineligible_archived_face_margin_below_1e-4"
        and child.get("sample_count") == 8 and isinstance(samples, list) and len(samples) == 8
        and isinstance(normals, list) and len(normals) == 2
        and isinstance(hulls, list) and len(hulls) == 4
        and isinstance(changes, list) and len(changes) == 4
        and child.get("evidence_identity_before") == expected_evidence
        and child.get("evidence_identity_after") == expected_evidence
        and child.get("source_before") == expected_sources
        and child.get("source_after") == expected_sources
        and expected_sources.get(RUNNER_PATH) == RUNNER_SHA256
        and child.get("input_before") == expected_input
        and child.get("input_after") == expected_input
        and len(expected_event_metadata) == 2
        and len(expected_sample_metadata) == 8
    ):
        return False

    if [
        {key: normal.get(key) for key in metadata}
        for normal, metadata in zip(normals, expected_event_metadata, strict=True)
        if isinstance(normal, dict)
    ] != expected_event_metadata:
        return False
    normals_by_chord: dict[str, Tensor] = {}
    for normal in normals:
        if not isinstance(normal, dict):
            return False
        label = normal.get("label")
        vector = normal.get("normal")
        if (not isinstance(label, str) or label in normals_by_chord
                or not _finite_vector(vector, 26)):
            return False
        normal_tensor = torch.tensor(vector, dtype=torch.float64)
        norm = float(torch.linalg.vector_norm(normal_tensor))
        if (not math.isfinite(norm) or norm <= 128.0 * torch.finfo(torch.float64).eps
                or bool(torch.count_nonzero(normal_tensor[:20]))
                or bool(torch.count_nonzero(normal_tensor[25:]))
                or not _close_number(normal.get("normal_l2"), norm)
                or not _close_blocks(normal.get("normal_block_l2"), _block_norms(normal_tensor))
                or not _finite_number(normal.get("flux_scale_max_abs"))
                or normal["flux_scale_max_abs"] < 0
                or not _finite_number(normal.get("event_face_flux"))
                or not _close_number(
                    normal.get("normal_zero_guard"),
                    128.0 * torch.finfo(torch.float64).eps * max(1.0, norm),
                )):
            return False
        normals_by_chord[label] = normal_tensor
    if len(normals_by_chord) != 2:
        return False

    gradients: dict[tuple[str, str, float], Tensor] = {}
    tangents: dict[tuple[str, str, float], Tensor] = {}
    for sample, metadata in zip(samples, expected_sample_metadata, strict=True):
        report_keys = {
            "status": "archived_status",
            "response_margin_qualified": "archived_response_margin_qualified",
        }
        if not isinstance(sample, dict) or any(
            sample.get(report_keys.get(key, key)) != value for key, value in metadata.items()
        ):
            return False
        label = sample.get("chord")
        side = sample.get("side")
        offset = sample.get("offset")
        if (label not in normals_by_chord or side not in ("left", "right")
                or offset not in SIDE_OFFSETS or sample.get("archived_status") != "finite_off_event_derivatives"
                or sample.get("archived_response_margin_qualified") is not False):
            return False
        key = (label, side, offset)
        if key in gradients or not _finite_vector(sample.get("gradient"), 26):
            return False
        gradient = torch.tensor(sample["gradient"], dtype=torch.float64)
        expected = tangent_gradient(gradient, normals_by_chord[label])
        if (not _close_number(sample.get("gradient_l2"), expected["gradient_l2"])
                or not _close_number(sample.get("gradient_inf"), expected["gradient_inf"])
                or not _close_number(sample.get("gradient_l2"), float(metadata["gradient_l2"]))
                or not _close_number(sample.get("gradient_inf"), float(metadata["gradient_inf"]))
                or not _close_number(sample.get("unit_normal_slope"), expected["unit_normal_slope"])
                or not _finite_vector(sample.get("tangent_gradient"), 26)
                or not _close_vector(sample.get("tangent_gradient"), torch.tensor(
                    expected["tangent_gradient"], dtype=torch.float64,
                ))
                or not _close_number(sample.get("tangent_l2"), expected["tangent_l2"])
                or not _close_blocks(sample.get("gradient_block_l2"), _block_norms(gradient))
                or not _close_blocks(sample.get("tangent_block_l2"), expected["tangent_block_l2"])):
            return False
        gradients[key] = gradient
        tangents[key] = torch.tensor(expected["tangent_gradient"], dtype=torch.float64)
    if len(gradients) != 8:
        return False

    expected_hulls: list[dict[str, Any]] = []
    expected_changes: list[dict[str, Any]] = []
    for label in normals_by_chord:
        for offset in SIDE_OFFSETS:
            expected_hulls.append({
                "chord": label, "offset": offset,
                **segment_minimum(gradients[(label, "left", offset)], gradients[(label, "right", offset)]),
            })
        far, near = SIDE_OFFSETS
        for side in ("left", "right"):
            delta = tangents[(label, side, near)] - tangents[(label, side, far)]
            expected_changes.append({
                "chord": label, "side": side, "far_offset": far, "near_offset": near,
                "near_minus_far": delta.tolist(), "change_l2": float(torch.linalg.vector_norm(delta)),
            })
    for actual, expected in zip(hulls, expected_hulls, strict=True):
        if (not isinstance(actual, dict)
                or any(actual.get(key) != value for key, value in expected.items()
                       if key in ("chord", "offset", "status"))
                or not _close_number(actual.get("difference_norm"), expected["difference_norm"])
                or not _close_number(actual.get("degeneracy_tolerance"), expected["degeneracy_tolerance"])):
            return False
        for key in ("alpha", "minimum_norm"):
            if expected[key] is None:
                if actual.get(key) is not None:
                    return False
            elif not _close_number(actual.get(key), expected[key]):
                return False
    for actual, expected in zip(changes, expected_changes, strict=True):
        if (not isinstance(actual, dict)
                or any(actual.get(key) != expected[key]
                       for key in ("chord", "side", "far_offset", "near_offset"))
                or not _close_vector(actual.get("near_minus_far"), torch.tensor(
                    expected["near_minus_far"], dtype=torch.float64,
                ))
                or not _close_number(actual.get("change_l2"), expected["change_l2"])):
            return False
    return True


def run_guarded_diagnostic(directory: Path, *, expected_source_sha256: str) -> dict[str, Any]:
    """Launch one reviewed child, keeping execution and numerical status separate."""
    if directory.exists() or directory.is_symlink():
        raise ValueError("attempt directory must be fresh")
    directory = directory.resolve()
    if (directory == ARCHIVE.resolve() or directory.is_relative_to(ARCHIVE.resolve())
            or directory == PR225_CHILD.parent.resolve()
            or directory.is_relative_to(PR225_CHILD.parent.resolve())):
        raise ValueError("attempt directory must be outside PR209 and PR225 archives")
    source_sha = _sha(Path(__file__))
    if expected_source_sha256 != source_sha:
        raise ValueError("caller-reviewed probe SHA256 does not match current source")
    sources_before = _sources()
    _require_runner_pin(sources_before)
    if sources_before.get(SELF_PATH) != expected_source_sha256:
        raise ValueError("probe source identity differs from caller-reviewed SHA256")
    directory.mkdir(parents=True)
    output = directory / "partial_face_tangent.json"
    resource_path = directory / "partial_face_tangent.resource.json"
    log_path = directory / "partial_face_tangent.log"
    run_path = directory / "partial_face_tangent.run.json"
    if any(path.exists() or path.is_symlink()
           for path in (output, resource_path, log_path, run_path)):
        raise ValueError("attempt output paths must all be fresh")
    paths = (PLAN, PR225_CHILD, PR225_EVIDENCE,
             ARCHIVE / "manifest.json", *(ARCHIVE / name for name in RAW_FILES))
    identity_before = {_path_name(path): _sha(path) for path in paths}
    input_before = json.loads(json.dumps(
        face_event.prior._preflight_identity(), allow_nan=False,
    ))
    pr225, _ = _validate_pr225()
    archived_chords, archived_samples = _archived_controls(pr225)
    expected_event_metadata = []
    for chord in archived_chords:
        event_control = chord["start"] + chord["event_t"] * chord["direction"]
        expected_event_metadata.append({
            "label": chord["label"],
            "event_parameter": chord["event_t"],
            "event_control_sha256": _tensor_sha(event_control),
        })
    expected_sample_metadata = _sample_metadata(archived_samples)
    command = [
        sys.executable, str(Path(__file__)), "--child", "--output", str(output),
        "--expected-plan-sha256", PLAN_SHA256,
        "--expected-source-sha256", source_sha,
    ]
    guard_error = None
    resource = None
    try:
        resource = run_guarded(
            command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
            report_path=resource_path, log_path=log_path,
        )
    except Exception as error:
        guard_error = f"{type(error).__name__}: {error}"
    child = _read_json(output)
    identity_after = {_path_name(path): _sha(path) for path in paths}
    sources_after = _sources()
    input_after = face_event.prior._preflight_identity()
    unchanged = (
        identity_before == identity_after and sources_before == sources_after
        and input_before == input_after and source_sha == _sha(Path(__file__))
        and sources_after.get(SELF_PATH) == expected_source_sha256
        and sources_after.get(RUNNER_PATH) == RUNNER_SHA256
    )
    if not unchanged:
        execution_status = "identity_changed"
    elif guard_error is not None or resource is None:
        execution_status = "guard_failed"
    elif resource.get("resource_termination") is not None:
        execution_status = "resource_limited"
    elif resource.get("exit_code") != 0:
        execution_status = "child_nonzero_exit" if child else "child_exception"
    elif not _valid_resource(resource, command):
        execution_status = "resource_report_invalid"
    elif not _valid_child(
        child,
        expected_source_sha256=expected_source_sha256,
        expected_sources=sources_before,
        expected_input=input_before,
        expected_evidence=identity_before,
        expected_child_pid=resource["child_pid"],
        expected_event_metadata=expected_event_metadata,
        expected_sample_metadata=expected_sample_metadata,
    ):
        execution_status = "child_result_invalid"
    else:
        execution_status = "completed"
    result = {
        "execution_status": execution_status,
        "numerical_status": child.get("status") if child else "not_reached",
        "response_eligibility": child.get("response_eligibility") if child else "not_reached",
        "response_validation": "not_performed",
        "resource": resource,
        "guard_error": guard_error,
        "identity_before": identity_before,
        "identity_after": identity_after,
        "sources_before": sources_before,
        "sources_after": sources_after,
        "input_before": input_before,
        "input_after": input_after,
        "identities_unchanged": unchanged,
        "child_output_exists": output.exists(),
        "scope": "one guarded PR209/PR225 finite-offset tangent-gradient diagnostic",
    }
    run_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("--child requires --output")
        report = _diagnostic(
            output=args.output,
            expected_plan_sha256=args.expected_plan_sha256 or "",
            expected_source_sha256=args.expected_source_sha256,
        )
    else:
        if args.directory is None:
            parser.error("parent mode requires --directory")
        report = run_guarded_diagnostic(
            args.directory, expected_source_sha256=args.expected_source_sha256,
        )
    print(json.dumps({
        key: report.get(key) for key in (
            "execution_status", "numerical_status", "status", "response_eligibility"
        ) if key in report
    }, sort_keys=True))
    return 0 if report.get("execution_status", "completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
