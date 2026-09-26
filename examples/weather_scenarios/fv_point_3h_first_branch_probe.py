"""Locate the first strict minmod-branch refusal in the fixed 3-hour case.

This diagnostic reports pointwise predicates only. It does not certify a
stationary point, a finite perturbation segment, or a response.
"""
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
from typing import Any, Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport
from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios import fv_point_3h_forward_probe as forward_probe
from examples.weather_scenarios.fv_point_3h_forward_case import make_case


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_3H_FIRST_BRANCH_PLAN.md"
PLAN_SHA256 = "ed468d020ed1f4b7170c8b6089537b1535f85f0c1e635f717a0a12aba3d3fd32"
ARCHIVE = EVIDENCE / "point_3h_forward_attempt1"
ARCHIVE_MANIFEST = ARCHIVE / "manifest.json"
ARCHIVE_MANIFEST_SHA256 = "056963a6f264a97aa165501a1ff6c07bf62842f17ff26992037cb977e0469de8"
FORWARD_EVIDENCE = EVIDENCE / "FV_POINT_3H_FORWARD_EVIDENCE.json"
FORWARD_EVIDENCE_SHA256 = "a78f09769c5201eafcb7eaedd4f8e506266f80409476780e128d4b08608cfc4e"
POLICY_PLAN = EVIDENCE / "FV_POINT_3H_RESPONSE_POLICY_PLAN.md"
POLICY_PLAN_SHA256 = "d1d87f91306a2c4c3fed2b99c329314c04670a4ada56c79a884e72ebdd905b00"
POLICY_RESULTS = EVIDENCE / "FV_POINT_3H_RESPONSE_POLICY_RESULTS.md"
POLICY_RESULTS_SHA256 = "5d1cd984afc687122d42d075d3cc7659119709dfb9c6572d5d7c5fcdbd04f14e"
POLICY_RECORD = EVIDENCE / "FV_POINT_3H_RESPONSE_POLICY.json"
POLICY_RECORD_SHA256 = "7df3012761264b5a063b7a1a5218d2e7157cd2aa5b7eb0e8156f57a666fc921c"
POLICY_EVIDENCE = EVIDENCE / "FV_POINT_3H_RESPONSE_POLICY_EVIDENCE.json"
POLICY_EVIDENCE_SHA256 = "3f9a1d3bbb87636a204f4be1dfa5b3c8994d500951c51e3f2838b11ff86979e4"
ARCHIVED_INPUT_SHA256 = "00665e879d0e370e15f2bf3ef253a2267eb59335c2f51036289ff22632808744"
WALL_SECONDS = 180
SAMPLED_RSS_BYTES = 768 * 1024**2
EXPECTED_STAGES = 3600
DEFAULT_DIRECTORY = EVIDENCE / "point_3h_first_branch_attempt1"

# These sources define the archived PR #204 profile and its branch operator.
PINNED_ARCHIVE_SOURCES = {
    "examples/weather_scenarios/fv_long_horizon_case.py": "22feb5c49495b26c78caa826e73f34a9abb21a4dbf71cfa5e34fbaae884cbc85",
    "examples/weather_scenarios/fv_minmod_inverse_probe.py": "38c0590b0a8028ae01c203782976e08ed29df0f9c54bf784bd964f5abd6fbd6f",
    "examples/weather_scenarios/fv_point_3h_forward_case.py": "a23bab3d10bde9ca08661979df9e2c94a0de7bde6c7c732b5ed2b9ec769f61ec",
    "examples/weather_scenarios/fv_point_3h_forward_probe.py": "8ab664ebcb42d6c74d1b056a751df57941816ca1804538f9afd954b26f95ca75",
    "examples/weather_scenarios/fv_point_3h_forward_runner.py": "bf50195f2ff7d7ba7550e54ebb3473f0d501cde2663dd8d4f0e5281db97570af",
    "examples/weather_scenarios/fv_point_research_case.py": "358dd1bb46f7cf2ebb0fcf7bb33d4735b66419412d97d0b7847e7eab236f5f2e",
    "examples/weather_scenarios/fv_point_response_preflight.py": "6929668a5d2c95a64499ab598a0f045d66e47a80cf7a997513f448fff8ba3ccf",
    "examples/weather_scenarios/fv_point_response_preflight_runner.py": "7e25520af78418323a28dd3aec152040829bba597ab2902c4a80413c1088aab8",
    "examples/weather_scenarios/fv_sensitivity_probe.py": "919c6d94717fd96dd68d2f63d73b4c15d5e96e5d3d65efccf4deecdc7dce2b4f",
    "src/advar/fv_point_research_problem.py": "ffdb27fd40aa6943ab4c04860695303e4270291377a7b86535c06416704eec19",
    "src/advar/fv_point_sampler.py": "e3116a9250a5890521d66502aacd69c6ea32cc348fb961e6ffb88cf97750d3af",
    "src/advar/fv_research_problem.py": "487a429dadd1179db9d4e27a2523bf64e81fffe90c955ef46ea392aae313c3af",
    "src/advar/local_refinement.py": "47366f828777b6fc322386b6dda8209916310ec2d06590767c1a9512ac344321",
    "src/advar/local_response.py": "ed4ab262907da26bfdf6201997083125c1430ebccce9afa70982417d77a060f4",
    "src/advar/matrix_free.py": "d30f6d888a5ec2c693b6787930a29d604c5da58a7030114957a1e8bb3e7e788d",
    "src/advar/nowcast.py": "391612bb8714c33d82f426f2225cf8869b2cb45679d362f95f2a2fb7191da63b",
    "src/advar/physics.py": "00b49438c2099ce514ef24a54b89eb21565315150b47e9c880513187c44bbed3",
    "src/advar/transport.py": "604709c561700def1f78b1eaac0beeb304d0b82554ea20bb16f8bb4fb44ae6bf",
    "src/advar/variational.py": "e871557463090abf185f6f29ef73c3813cf503497cac25d10a4f500f31f1824a",
}
SOURCE_PATHS = tuple(PINNED_ARCHIVE_SOURCES) + (
    "examples/weather_scenarios/fv86_resource_runner.py",
    "examples/weather_scenarios/fv_point_3h_first_branch_probe.py",
)


class _FirstBranchRefusal(Exception):
    """Internal control flow used to stop at the first inadmissible stage."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_float(value: Tensor | float) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _source_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _archive_input_identity() -> dict[str, Any]:
    if (_sha(PLAN) != PLAN_SHA256
            or _sha(ARCHIVE_MANIFEST) != ARCHIVE_MANIFEST_SHA256
            or _sha(FORWARD_EVIDENCE) != FORWARD_EVIDENCE_SHA256
            or _sha(POLICY_PLAN) != POLICY_PLAN_SHA256
            or _sha(POLICY_RESULTS) != POLICY_RESULTS_SHA256
            or _sha(POLICY_RECORD) != POLICY_RECORD_SHA256
            or _sha(POLICY_EVIDENCE) != POLICY_EVIDENCE_SHA256):
        raise ValueError("fixed three-hour plan, policy record or archived identity changed")
    manifest = json.loads(ARCHIVE_MANIFEST.read_text())
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("archived forward manifest is incomplete")
    for name, expected in hashes.items():
        path = (ARCHIVE / name).resolve()
        if not path.is_relative_to(ARCHIVE.resolve()) or _sha(path) != expected:
            raise ValueError(f"archived forward artifact changed: {name}")
    child = json.loads((ARCHIVE / "point_3h_forward.json").read_text())
    identity = child.get("input_before")
    if not isinstance(identity, dict) or _canonical_sha(identity) != ARCHIVED_INPUT_SHA256:
        raise ValueError("PR #204 archived input identity changed")
    if child.get("input_after") != identity or child.get("input_unchanged") is not True:
        raise ValueError("PR #204 archived child input identity is inconsistent")
    policy = json.loads(POLICY_RECORD.read_text())
    if policy.get("input_identity") != identity or policy.get("stationary_root_absence_proved") is not False:
        raise ValueError("R5 response-policy record no longer binds the archived input")
    return identity


def _sources_match_archive(source: dict[str, str]) -> bool:
    return all(source.get(name) == digest
               for name, digest in PINNED_ARCHIVE_SOURCES.items())


def _first_index(mask: Tensor, offset: tuple[int, int] = (0, 0)) -> list[int] | None:
    positions = torch.nonzero(mask, as_tuple=False)
    if positions.numel() == 0:
        return None
    first = positions[0]
    return [int(first[0]) + offset[0], int(first[1]) + offset[1]]


def _violation(name: str, mask: Tensor, *, orientation: str | None = None,
               offset: tuple[int, int] = (0, 0), **details: Any) -> dict[str, Any]:
    count = int(torch.count_nonzero(mask))
    result: dict[str, Any] = {"predicate": name, "count": count,
                              "first_index": _first_index(mask, offset)}
    if orientation is not None:
        result["orientation"] = orientation
    result["index_space"] = (
        "full_field_interior_center_cell_0based" if name.endswith("slope_nonzero")
        or "active_limiter" in name else f"{orientation}_face_array_0based"
    )
    if details:
        result["values"] = details
    return result


def classify_stage(stage_index: int, q: Tensor, qx: Tensor,
                   qy: Tensor) -> dict[str, Any]:
    """Return ordered strict-oracle failures and all violations at a stage.

    The order matches ``fv_minmod_inverse_probe.inspect_branches``: field
    scale, x then y interior left/right/tie tests, then x and y face fluxes.
    Indices are zero-based; slope indices refer to their interior center cell.
    """
    scale_tensor = q.abs().max()
    scale_finite = bool(torch.isfinite(scale_tensor))
    scale = _safe_float(scale_tensor) if scale_finite else None
    violations: list[dict[str, Any]] = []
    if not scale_finite or scale is None or scale <= 0:
        violations.append({
            "predicate": "field_scale_finite_positive", "count": 1,
            "first_index": None, "values": {"scale": scale, "finite": scale_finite},
        })
        return {"stage_index": stage_index, "first_failure": violations[0],
                "violations": violations,
                "violation_counts": {"field_scale_finite_positive": 1},
                "simultaneous_violation_count": 1}

    tolerance = 128 * torch.finfo(q.dtype).eps * scale_tensor
    for orientation, left, right in (
        ("x", q[1:-1, 1:-1] - q[1:-1, :-2],
         q[1:-1, 2:] - q[1:-1, 1:-1]),
        ("y", q[1:-1, 1:-1] - q[:-2, 1:-1],
         q[2:, 1:-1] - q[1:-1, 1:-1]),
    ):
        positive = (left > tolerance) & (right > tolerance)
        negative = (left < -tolerance) & (right < -tolerance)
        active = positive | negative
        difference = (left - right).abs()
        checks = [
            (f"{orientation}_left_slope_nonzero", left.abs() <= tolerance,
             {"minimum_abs": _safe_float(left.abs().min()),
              "tolerance": _safe_float(tolerance)}),
            (f"{orientation}_right_slope_nonzero", right.abs() <= tolerance,
             {"minimum_abs": _safe_float(right.abs().min()),
              "tolerance": _safe_float(tolerance)}),
            (f"{orientation}_active_limiter_operand_difference",
             active & (difference <= tolerance),
             {"minimum_active_difference": (
                 _safe_float(difference[active].min()) if bool(active.any()) else None
             ), "tolerance": _safe_float(tolerance)}),
        ]
        for name, mask, values in checks:
            if bool(mask.any()):
                violations.append(_violation(name, mask, orientation=orientation,
                                             offset=(1, 1), **values))

    flux = torch.cat((qx.reshape(-1), qy.reshape(-1)))
    flux_scale = flux.abs().max()
    flux_tolerance = 128 * torch.finfo(q.dtype).eps * flux_scale
    for orientation, faces in (("x", qx), ("y", qy)):
        mask = faces.abs() <= flux_tolerance
        if bool(mask.any()):
            violations.append(_violation(
                f"{orientation}_face_flux_nonzero", mask,
                orientation=orientation,
                values_min_abs=_safe_float(faces.abs().min()),
                tolerance=_safe_float(flux_tolerance),
            ))
    return {
        "stage_index": stage_index,
        "first_failure": violations[0] if violations else None,
        "violations": violations,
        "violation_counts": {item["predicate"]: item["count"] for item in violations},
        "simultaneous_violation_count": len(violations),
        "simultaneous_violating_entries": sum(item["count"] for item in violations),
    }


def _stage_context(stage_index: int) -> dict[str, int | str | None]:
    if stage_index < 360:
        interval_index = stage_index // 180
        local_stage = stage_index % 180
        return {
            "phase": "analysis_replay", "interval_index": interval_index,
            "interval_start_seconds": interval_index * 600,
            "interval_end_seconds": (interval_index + 1) * 600,
            "forecast_lead_index": None,
            "substep_index": local_stage // 2,
            "ssprk_euler_stage_index": local_stage % 2,
        }
    future_index = stage_index - 360
    interval_index = future_index // 180
    local_stage = future_index % 180
    lead_index = interval_index + 1
    return {
        "phase": "future_forecast", "interval_index": interval_index,
        "interval_start_seconds": 1200 + interval_index * 600,
        "interval_end_seconds": 1200 + lead_index * 600,
        "forecast_lead_index": lead_index,
        "substep_index": local_stage // 2,
        "ssprk_euler_stage_index": local_stage % 2,
    }


def observe_first_failure(forecast: Callable[[], Tensor]) -> dict[str, Any]:
    """Observe a forecast and stop immediately after its first strict failure."""
    stages_observed = 0
    first_failure: dict[str, Any] | None = None

    def observe(q: Tensor, qx: Tensor, qy: Tensor) -> None:
        nonlocal stages_observed, first_failure
        result = classify_stage(stages_observed, q, qx, qy)
        stages_observed += 1
        if result["first_failure"] is not None:
            first_failure = result
            raise _FirstBranchRefusal

    try:
        with torch.no_grad(), transport.observe_minmod_stages(observe):
            forecast()
    except _FirstBranchRefusal:
        pass
    if first_failure is not None:
        return {"status": "refused_first_branch", "stages_observed": stages_observed,
                "accepted_stages_before_failure": first_failure["stage_index"],
                "stage_context": _stage_context(first_failure["stage_index"]),
                **first_failure}
    return {"status": "passed_pointwise", "stages_observed": stages_observed,
            "accepted_stages_before_failure": stages_observed,
            "euler_stages": stages_observed, "first_failure": None}


def _input_identity(problem: Any, control: Tensor, parameters: Tensor,
                    truth: Tensor) -> dict[str, Any]:
    return forward_probe._identity(problem, control, parameters, truth)


def _validate_profile(problem: Any, control: Tensor, parameters: Tensor,
                      truth: Tensor) -> dict[str, Any]:
    layout = problem.layout
    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("fixed point three-hour profile has no FV transport")
    steps = problem.frozen.nowcast_config.forecast_steps
    if not (
        control.shape == (26,) and parameters.shape == (13,)
        and truth.shape == (4, 5)
        and layout["controls"] == 26 and layout["parameters"] == 13
        and layout["observation_times_seconds"] == (0.0, 600.0, 1200.0)
        and layout["forecast_time_seconds"] == 12000.0
        and layout["euler_stages"] == EXPECTED_STAGES
        and steps == 18 and spec.substeps_per_interval == 90
        and len(spec.boundary_echo) == 180
        and len(problem.future_boundary_echo) == 1620
        and bool(problem.frozen.initial_support_mask.all())
        and torch.equal(problem.background_dbz, problem.frozen.initial_background_dbz)
    ):
        raise ValueError("fixed PR #204 26-control/13-parameter profile changed")
    return layout


def run_probe(output: Path) -> dict[str, Any]:
    """Run the one pinned diagnostic in its guarded child process."""
    started = time.monotonic()
    source_before = _source_hashes()
    if not _sources_match_archive(source_before):
        raise ValueError("fixed PR #204 source identity changed")
    archive_identity = _archive_input_identity()
    plan_before = _sha(PLAN)
    if source_before.get("examples/weather_scenarios/fv_minmod_inverse_probe.py") != PINNED_ARCHIVE_SOURCES[
        "examples/weather_scenarios/fv_minmod_inverse_probe.py"
    ]:
        raise ValueError("strict branch oracle source changed")
    problem, control, parameters, truth = make_case()
    layout = _validate_profile(problem, control, parameters, truth)
    input_before = _input_identity(problem, control, parameters, truth)
    if _canonical_sha(input_before) != ARCHIVED_INPUT_SHA256 or input_before != archive_identity:
        raise ValueError("PR #204 fixed input differs from archived forward profile")
    archive_before = {
        "manifest_sha256": _sha(ARCHIVE_MANIFEST),
        "forward_evidence_sha256": _sha(FORWARD_EVIDENCE),
        "policy_plan_sha256": _sha(POLICY_PLAN),
        "policy_results_sha256": _sha(POLICY_RESULTS),
        "policy_record_sha256": _sha(POLICY_RECORD),
        "policy_evidence_sha256": _sha(POLICY_EVIDENCE),
        "input_identity_sha256": _canonical_sha(archive_identity),
        "archived_profile_source_sha256": PINNED_ARCHIVE_SOURCES,
        "current_guard_source_sha256": source_before[
            "examples/weather_scenarios/fv86_resource_runner.py"
        ],
    }
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "first_branch_diagnostic",
        "execution_phase": "running", "execution_status": "running",
        "diagnostic_status": "not_performed", "stationarity_passed": "not_tested",
        "response_computed": False, "response_validation": "not_performed",
        "physical_validation": "not_performed", "forecast_score_computed": False,
        "plan_sha256": PLAN_SHA256, "archive": archive_before,
        "source_before": source_before, "input_before": input_before,
        "layout": layout,
        "scope": "fixed PR #204 point-observation 4x5 profile; first strict minmod predicate only; no solver or response",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
    }

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        diagnostic = observe_first_failure(lambda: problem.forecast(control, parameters))
        if diagnostic["status"] == "passed_pointwise" and diagnostic["euler_stages"] != EXPECTED_STAGES:
            raise ValueError("pointwise pass did not observe all 3,600 stages")
        report["diagnostic"] = diagnostic
        report["diagnostic_status"] = diagnostic["status"]
        report["execution_status"] = "completed"
        report["execution_phase"] = "finished"
        report["phase"] = "finished"
    except Exception as error:
        report["execution_status"] = "failed"
        report["execution_phase"] = "failed"
        report["phase"] = "failed"
        report["diagnostic_error"] = f"{type(error).__name__}: {error}"
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _source_hashes()
        report["input_after"] = _input_identity(problem, control, parameters, truth)
        report["source_unchanged"] = report["source_after"] == source_before
        report["input_unchanged"] = report["input_after"] == input_before
        report["plan_unchanged"] = _sha(PLAN) == PLAN_SHA256 == plan_before
        try:
            archive_after = _archive_input_identity()
            report["archive_unchanged"] = (
                _sha(ARCHIVE_MANIFEST) == archive_before["manifest_sha256"]
                and _sha(FORWARD_EVIDENCE) == archive_before["forward_evidence_sha256"]
                and _sha(POLICY_PLAN) == archive_before["policy_plan_sha256"]
                and _sha(POLICY_RESULTS) == archive_before["policy_results_sha256"]
                and _sha(POLICY_RECORD) == archive_before["policy_record_sha256"]
                and _sha(POLICY_EVIDENCE) == archive_before["policy_evidence_sha256"]
                and _canonical_sha(archive_after) == archive_before["input_identity_sha256"]
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            report["archive_unchanged"] = False
            report["archive_recheck_error"] = f"{type(error).__name__}: {error}"
        if not all(report[key] for key in (
            "source_unchanged", "input_unchanged", "plan_unchanged", "archive_unchanged"
        )):
            report.update(execution_status="failed", execution_phase="identity_refused",
                          diagnostic_status="not_performed")
        save()
    return report


def _valid_resource(value: dict[str, Any], command: list[str]) -> bool:
    elapsed = value.get("elapsed_seconds")
    peak = value.get("sampled_peak_rss_bytes")
    elapsed_valid = (
        isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
        and math.isfinite(elapsed) and 0 <= elapsed <= WALL_SECONDS
    )
    return (
        value.get("command") == command and value.get("exit_code") == 0
        and value.get("resource_termination") is None and value.get("monitor_error") is None
        and value.get("wall_limit_seconds") == WALL_SECONDS
        and value.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(value.get("rss_samples")) is int and value["rss_samples"] > 0
        and type(peak) is int and 0 < peak <= SAMPLED_RSS_BYTES
        and elapsed_valid
    )


def _valid_diagnostic(child: dict[str, Any]) -> bool:
    if not (
        child.get("execution_status") == "completed"
        and child.get("execution_phase") == "finished"
        and child.get("stationarity_passed") == "not_tested"
        and child.get("response_computed") is False
        and child.get("response_validation") == "not_performed"
        and child.get("physical_validation") == "not_performed"
        and child.get("forecast_score_computed") is False
        and isinstance(child.get("diagnostic"), dict)
    ):
        return False
    value = child["diagnostic"]
    if value.get("status") == "passed_pointwise":
        return (
            child.get("diagnostic_status") == "passed_pointwise"
            and value.get("euler_stages") == EXPECTED_STAGES
            and value.get("stages_observed") == EXPECTED_STAGES
            and value.get("first_failure") is None
        )
    first = value.get("first_failure")
    return (
        child.get("diagnostic_status") == "refused_first_branch"
        and value.get("status") == "refused_first_branch"
        and isinstance(first, dict)
        and type(value.get("stage_index")) is int
        and 0 <= value["stage_index"] < EXPECTED_STAGES
        and value.get("stages_observed") == value["stage_index"] + 1
        and value.get("accepted_stages_before_failure") == value["stage_index"]
        and first.get("predicate") in {
            "field_scale_finite_positive", "x_left_slope_nonzero",
            "x_right_slope_nonzero", "x_active_limiter_operand_difference",
            "y_left_slope_nonzero", "y_right_slope_nonzero",
            "y_active_limiter_operand_difference", "x_face_flux_nonzero",
            "y_face_flux_nonzero",
        }
        and type(value.get("simultaneous_violation_count")) is int
        and value["simultaneous_violation_count"] >= 1
        and isinstance(value.get("violations"), list)
        and bool(value["violations"])
        and value["first_failure"] == value["violations"][0]
        and isinstance(value.get("stage_context"), dict)
        and value["stage_context"] == _stage_context(value["stage_index"])
        and all(isinstance(item, dict) and type(item.get("count")) is int
                and item["count"] > 0 for item in value["violations"])
        and value["simultaneous_violation_count"] == len(value["violations"])
    )


def _valid_layout(layout: object) -> bool:
    return isinstance(layout, dict) and (
        layout.get("controls") == 26 and layout.get("parameters") == 13
        and layout.get("state_shape") == [4, 5]
        and layout.get("observation_shape") == [3, 4]
        and layout.get("observation_times_seconds") == [0.0, 600.0, 1200.0]
        and layout.get("forecast_time_seconds") == 12000.0
        and layout.get("euler_stages") == EXPECTED_STAGES
    )


def run(directory: Path) -> dict[str, Any]:
    """Launch exactly one child and separate execution from diagnostic result."""
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("first-branch output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_3h_first_branch.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output)]
    source_before = _source_hashes()
    if not _sources_match_archive(source_before):
        raise ValueError("fixed PR #204 source identity changed")
    archive_identity = _archive_input_identity()
    resource = run_guarded(
        command, wall_seconds=WALL_SECONDS, rss_bytes=SAMPLED_RSS_BYTES,
        report_path=directory / "point_3h_first_branch.resource.json",
        log_path=directory / "point_3h_first_branch.log",
    )
    child: dict[str, Any] | None = None
    read_error = None
    try:
        loaded = json.loads(output.read_text())
        child = loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    current_sources = _source_hashes()
    resource_ok = _valid_resource(resource, command)
    child_ok = bool(
        child is not None and _valid_diagnostic(child)
        and child.get("pid") == resource.get("child_pid")
        and child.get("source_unchanged") is True
        and _sources_match_archive(source_before)
        and child.get("source_before") == child.get("source_after") == source_before == current_sources
        and child.get("input_unchanged") is True
        and child.get("input_before") == child.get("input_after") == archive_identity
        and _canonical_sha(child.get("input_before")) == ARCHIVED_INPUT_SHA256
        and child.get("plan_sha256") == PLAN_SHA256 and child.get("plan_unchanged") is True
        and child.get("archive") == {
            "manifest_sha256": ARCHIVE_MANIFEST_SHA256,
            "forward_evidence_sha256": FORWARD_EVIDENCE_SHA256,
            "policy_plan_sha256": POLICY_PLAN_SHA256,
            "policy_results_sha256": POLICY_RESULTS_SHA256,
            "policy_record_sha256": POLICY_RECORD_SHA256,
            "policy_evidence_sha256": POLICY_EVIDENCE_SHA256,
            "input_identity_sha256": ARCHIVED_INPUT_SHA256,
            "archived_profile_source_sha256": PINNED_ARCHIVE_SOURCES,
            "current_guard_source_sha256": source_before[
                "examples/weather_scenarios/fv86_resource_runner.py"
            ],
        }
        and _valid_layout(child.get("layout"))
        and child.get("archive_unchanged") is True
        and _sha(PLAN) == PLAN_SHA256
    )
    completed = resource_ok and child_ok
    result: dict[str, Any] = {
        "execution_status": "completed" if completed else "failed",
        "diagnostic_status": child.get("diagnostic_status", "not_performed") if child else "not_performed",
        "first_failure": child.get("diagnostic", {}).get("first_failure") if child else None,
        "stationarity_passed": "not_tested", "response_computed": False,
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "child_read_error": read_error, "resource": resource,
        "scope": "fixed PR #204 3-hour point forecast; strict branch cause only; no solver or response",
    }
    (directory / "point_3h_first_branch.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = run_probe(args.output)
        exit_code = 0 if result["execution_status"] == "completed" else 2
    elif args.directory is not None:
        result = run(args.directory)
        exit_code = 0 if result["execution_status"] == "completed" else 1
    else:
        parser.error("provide --output for child mode or --directory for guarded mode")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(exit_code)
