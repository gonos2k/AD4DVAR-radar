"""Analytic and fake-boundary tests for the finite-offset tangent diagnostic."""
from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_partial_face_tangent_probe as probe


def test_tangent_projection_removes_only_the_normal_component():
    normal = torch.zeros(26, dtype=torch.float64)
    normal[0] = 3.0
    gradient = torch.zeros(26, dtype=torch.float64)
    gradient[0] = 4.0
    gradient[20] = 5.0

    result = probe.tangent_gradient(gradient, normal)

    assert result["unit_normal_slope"] == 4.0
    assert result["tangent_gradient"][:1] == [0.0]
    assert result["tangent_gradient"][20] == 5.0
    assert result["tangent_l2"] == 5.0
    assert result["tangent_block_l2"] == {"field_20": 0.0, "flow_5": 5.0, "growth_1": 0.0}


def test_segment_minimum_uses_clipped_analytic_minimizer():
    left = torch.zeros(26, dtype=torch.float64)
    right = torch.zeros(26, dtype=torch.float64)
    left[0] = -2.0
    right[0] = 2.0
    left[1] = right[1] = 3.0

    result = probe.segment_minimum(left, right)

    assert result["status"] == "sampled_segment_minimum"
    assert result["alpha"] == 0.5
    assert result["minimum_norm"] == 3.0


def test_segment_minimum_reports_degenerate_pair_without_division():
    left = torch.zeros(26, dtype=torch.float64)
    right = left.clone()

    result = probe.segment_minimum(left, right)

    assert result["status"] == "degenerate_pair"
    assert result["alpha"] is None
    assert result["minimum_norm"] is None


def test_fake_diagnostic_records_float_face_value_without_event_objective_ad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    def write(path: Path, value: bytes) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return hashlib.sha256(value).hexdigest()

    root = tmp_path / "root"
    archive = root / "archive"
    child_path = root / "pr225" / "child.json"
    evidence_path = root / "pr225" / "evidence.json"
    plan_path = root / "plan.md"
    manifest_path = archive / "manifest.json"
    plan_hash = write(plan_path, b"plan")
    child_hash = write(child_path, b"child")
    evidence_hash = write(evidence_path, b"evidence")
    manifest_hash = write(manifest_path, b"manifest")

    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "PLAN", plan_path)
    monkeypatch.setattr(probe, "PLAN_SHA256", plan_hash)
    monkeypatch.setattr(probe, "PR225_CHILD", child_path)
    monkeypatch.setattr(probe, "PR225_CHILD_SHA256", child_hash)
    monkeypatch.setattr(probe, "PR225_EVIDENCE", evidence_path)
    monkeypatch.setattr(probe, "PR225_EVIDENCE_SHA256", evidence_hash)
    monkeypatch.setattr(probe, "ARCHIVE", archive)
    monkeypatch.setattr(probe, "ARCHIVE_MANIFEST_SHA256", manifest_hash)
    monkeypatch.setattr(probe, "RAW_FILES", ())
    monkeypatch.setattr(probe, "SOURCE_PATHS", ())
    expected_source_sha = probe._sha(Path(probe.__file__))
    expected_sources = {
        probe.SELF_PATH: expected_source_sha,
        probe.RUNNER_PATH: probe.RUNNER_SHA256,
        "source.py": "fixed",
    }
    monkeypatch.setattr(probe, "_sources", lambda: expected_sources)
    input_identity = {"current_problem_identity": "fixed-problem"}
    monkeypatch.setattr(probe.face_event.prior, "_preflight_identity", lambda: input_identity)

    objective_calls: list[torch.Tensor] = []

    def fake_objective(control: torch.Tensor, _parameters: torch.Tensor) -> torch.Tensor:
        objective_calls.append(control)
        return 0.5 * control.square().sum()

    class FakeProblem:
        identity = "fixed-problem"
        frozen = SimpleNamespace(
            fv_transport=object(), nowcast_config=SimpleNamespace(interval_minutes=10),
        )
        objective = staticmethod(fake_objective)

    parameters = torch.zeros(1, dtype=torch.float64)
    monkeypatch.setattr(probe, "_problem", lambda: (FakeProblem(), None, parameters))
    monkeypatch.setattr(
        probe, "_event_face",
        lambda control, _spec, _interval: (control[20], control),
    )

    chords: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for chord_index in range(2):
        label = f"chord-{chord_index}"
        chords.append({
            "label": label, "event_t": 0.5,
            "start": torch.zeros(26, dtype=torch.float64),
            "direction": torch.full(
                (26,), float(chord_index + 1), dtype=torch.float64,
            ),
        })
        for side_index, side in enumerate(("left", "right")):
            for offset_index, offset in enumerate(probe.SIDE_OFFSETS):
                control = torch.zeros(26, dtype=torch.float64)
                control[0] = 0.1 * (chord_index + 1)
                control[1] = float(side_index + offset_index + 1)
                gradient = control
                samples.append({
                    "chord": label, "side": side, "offset": offset,
                    "t": 0.5, "control_sha256": f"{len(samples):064x}",
                    "status": "finite_off_event_derivatives", "finite": True,
                    "response_margin_qualified": False,
                    "branch_scope": "fixed fake branch scope",
                    "branch_signature": {"stage": len(samples)},
                    "branch_signature_sha256": f"{len(samples):064x}",
                    "euler_stages": 54,
                    "minimum_scaled_face_flux_margin": 2.0e-5,
                    "minimum_scaled_slope_margin": 3.0e-5,
                    "face_margins_by_stage": [2.0e-5],
                    "gradient_l2": float(torch.linalg.vector_norm(gradient)),
                    "gradient_inf": float(gradient.abs().max()),
                    "_control": control,
                })

    archived_child = {
        "source_before": {}, "input_before": input_identity, "input_after": input_identity,
    }
    monkeypatch.setattr(probe, "_validate_pr225", lambda: (archived_child, {}))
    monkeypatch.setattr(probe, "_archived_controls", lambda _child: (chords, samples))
    report = probe._diagnostic(
        output=tmp_path / "attempt" / "report.json",
        expected_plan_sha256=plan_hash,
        expected_source_sha256=expected_source_sha,
    )

    assert report["sample_count"] == 8
    assert [row["event_face_flux"] for row in report["event_normals"]] == [0.5, 1.0]
    assert all(row["normal"][20] == 1.0 for row in report["event_normals"])
    assert len(objective_calls) == 8
    assert report["event_objective_or_trajectory_derivatives_evaluated"] is False
    expected_sample_metadata = probe._sample_metadata(samples)
    expected_event_metadata = [
        {
            "label": chord["label"],
            "event_parameter": chord["event_t"],
            "event_control_sha256": probe._tensor_sha(
                chord["start"] + chord["event_t"] * chord["direction"],
            ),
        }
        for chord in chords
    ]
    assert probe._valid_child(
        report,
        expected_source_sha256=expected_source_sha,
        expected_sources=expected_sources,
        expected_input=input_identity,
        expected_evidence=report["evidence_identity_before"],
        expected_child_pid=report["pid"],
        expected_event_metadata=expected_event_metadata,
        expected_sample_metadata=expected_sample_metadata,
    )
    def valid(candidate: dict[str, Any]) -> bool:
        return probe._valid_child(
            candidate,
            expected_source_sha256=expected_source_sha,
            expected_sources=expected_sources,
            expected_input=input_identity,
            expected_evidence=report["evidence_identity_before"],
            expected_child_pid=report["pid"],
            expected_event_metadata=expected_event_metadata,
            expected_sample_metadata=expected_sample_metadata,
        )

    forged = copy.deepcopy(report)
    forged["response_claim"] = True
    assert not valid(forged)

    forged = copy.deepcopy(report)
    forged["event_normals"][0]["normal"] = [0.0] * 26
    forged["event_normals"][0]["normal_l2"] = 0.0
    forged["event_normals"][0]["normal_block_l2"] = {
        "field_20": 0.0, "flow_5": 0.0, "growth_1": 0.0,
    }
    assert not valid(forged)

    forged = copy.deepcopy(report)
    forged["event_normals"][0]["normal"] = [0.0] * 26
    forged["event_normals"][0]["normal"][0] = 1.0
    forged["event_normals"][0]["normal_l2"] = 1.0
    forged["event_normals"][0]["normal_block_l2"] = {
        "field_20": 1.0, "flow_5": 0.0, "growth_1": 0.0,
    }
    assert not valid(forged)

    forged = copy.deepcopy(report)
    forged["pid"] += 1
    assert not valid(forged)

    forged = copy.deepcopy(report)
    forged["samples"][0]["gradient"] = [0.0] * 26
    forged["samples"][0]["gradient_l2"] = 0.0
    forged["samples"][0]["gradient_inf"] = 0.0
    forged["samples"][0]["tangent_gradient"] = [0.0] * 26
    forged["samples"][0]["tangent_l2"] = 0.0
    forged["samples"][0]["unit_normal_slope"] = 0.0
    forged["samples"][0]["gradient_block_l2"] = {
        "field_20": 0.0, "flow_5": 0.0, "growth_1": 0.0,
    }
    forged["samples"][0]["tangent_block_l2"] = {
        "field_20": 0.0, "flow_5": 0.0, "growth_1": 0.0,
    }
    assert not valid(forged)

    forged = copy.deepcopy(report)
    forged["same_chord_sampled_gradient_hulls"] = None
    forged["same_side_near_far_tangent_changes"] = None
    assert not valid(forged)

    for key in ("source_before", "input_before", "evidence_identity_before"):
        forged = copy.deepcopy(report)
        forged[key] = {"forged": "identity"}
        assert not valid(forged)

    forged = copy.deepcopy(report)
    forged["samples"][0]["branch_signature"] = {"forged": True}
    assert not valid(forged)

    forged = copy.deepcopy(report)
    forged["same_side_near_far_tangent_changes"][0]["near_minus_far"] = [0.0] * 26
    assert not valid(forged)
    assert json.loads((tmp_path / "attempt" / "report.json").read_text()) == report


def test_guard_validators_reject_forged_resource_and_child():
    command = ["python", "probe.py"]
    resource = {
        "command": command, "child_pid": 123, "exit_code": 0, "resource_termination": None,
        "monitor_error": None, "wall_limit_seconds": probe.WALL_SECONDS,
        "rss_limit_bytes": probe.SAMPLED_RSS_BYTES, "rss_samples": 1,
        "sampled_peak_rss_bytes": 1, "elapsed_seconds": 1.0,
    }
    assert probe._valid_resource(resource, command)
    assert not probe._valid_resource({**resource, "child_pid": 0}, command)
    assert not probe._valid_resource({**resource, "sampled_peak_rss_bytes": 0}, command)
    assert not probe._valid_child(
        {}, expected_source_sha256="0" * 64, expected_sources={}, expected_input={},
        expected_evidence={}, expected_child_pid=123, expected_event_metadata=[],
        expected_sample_metadata=[],
    )


def test_guarded_launch_requires_a_fresh_attempt_directory(tmp_path: Path):
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "unrelated.txt").write_text("preserve")

    with pytest.raises(ValueError, match="attempt directory must be fresh"):
        probe.run_guarded_diagnostic(directory, expected_source_sha256="reviewed")

    assert (directory / "unrelated.txt").read_text() == "preserve"


def test_guarded_launch_requires_the_caller_reviewed_probe_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    directory = tmp_path / "attempt"
    monkeypatch.setattr(probe, "run_guarded", lambda *_args, **_kwargs: pytest.fail(
        "guard must not run when the caller pin is stale",
    ))

    with pytest.raises(ValueError, match="caller-reviewed probe SHA256"):
        probe.run_guarded_diagnostic(directory, expected_source_sha256="0" * 64)

    assert not directory.exists()
