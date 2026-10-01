"""Archived FV support/refusal accounting without new model executions."""
from __future__ import annotations

import hashlib
import json

import pytest

from examples.weather_scenarios import fv_support_matrix as matrix


def test_stratified_inventory_preserves_refusal_and_unmeasured_validation():
    result = matrix.run()
    assert result["cross_profile_fraction"] is None
    assert result["profile_count"] == 4
    assert {name: tuple(profile["declared_case_ids"])
            for name, profile in result["profiles"].items()} == matrix.CASE_IDS

    full = result["profiles"]["fv86_full_support"]
    assert full["axes"]["execution"]["status_counts"] == {"completed": 2}
    assert full["axes"]["response"]["status_counts"] == {"computed": 2}
    assert full["axes"]["response"]["computed_evidence_coverage_fraction_of_declared"] == 1
    assert full["axes"]["response"]["computed_fraction_given_attempt"] == 1
    assert full["axes"]["local_response_eligibility"]["eligible_evidence_coverage_fraction_of_declared"] == 1
    assert full["axes"]["response_validation"]["independent_validation_attempted_count"] == 0
    assert full["axes"]["response_validation"]["attempted_fraction_of_declared"] == 0
    assert full["axes"]["response_validation"]["validated_evidence_coverage_fraction_of_declared"] == 0
    assert full["axes"]["response_validation"]["pass_fraction_given_attempt"] is None

    long = result["profiles"]["long_regular_forward"]
    assert long["axes"]["forecast"]["status_counts"] == {
        "available_same_operator_synthetic": 2,
    }
    assert long["axes"]["branch"]["status_counts"] == {"refused": 2}
    assert long["axes"]["local_response_eligibility"]["assessed_count"] == 2
    assert long["axes"]["local_response_eligibility"]["eligible_fraction_given_assessed"] == 0
    assert long["axes"]["response"]["status_counts"] == {"not_attempted": 2}
    assert long["axes"]["response"]["computed_evidence_coverage_fraction_of_declared"] == 0
    assert long["axes"]["response"]["computed_fraction_given_attempt"] is None
    assert all(row["current_source_match"] is False for row in long["rows"])
    assert all(row["proof_level"] == (
        "historical_source_snapshot_bound_no_child_exit_record"
    ) for row in long["rows"])

    partial = result["profiles"]["partial_two_hole"]
    assert partial["axes"]["execution"]["status_counts"] == {"failed": 1}
    assert partial["axes"]["stationarity"]["status_counts"] == {
        "not_established_due_to_refinement_refusal": 1,
    }
    assert partial["axes"]["response"]["status_counts"] == {"not_attempted": 1}
    assert partial["axes"]["local_response_eligibility"]["eligible_fraction_given_assessed"] == 0
    assert partial["rows"][0]["proof_level"] == "preflight_bound_nonzero_exit_no_final_source_check"

    point = result["profiles"]["point_missing_fixed_control"]
    assert point["axes"]["execution"]["status_counts"] == {"recorded_only": 1}
    assert point["axes"]["stationarity"]["status_counts"] == {"not_assessed": 1}
    assert point["axes"]["local_response_eligibility"]["assessed_count"] == 0
    assert point["axes"]["local_response_eligibility"]["eligible_fraction_given_assessed"] is None
    assert point["rows"][0]["current_source_match"] is False
    assert point["rows"][0]["proof_level"] == (
        "historical_source_snapshot_bound_no_child_exit_record"
    )


def test_validation_fraction_remains_null_until_an_attempt_exists():
    rows = [{"response_validation": "not_performed"},
            {"response_validation": "not_established"}]
    no_attempt = matrix._counts(rows, "response_validation")
    assert no_attempt["declared_case_count"] == 2
    assert no_attempt["independent_validation_attempted_count"] == 0
    assert no_attempt["validated_evidence_coverage_fraction_of_declared"] == 0
    assert no_attempt["pass_fraction_given_attempt"] is None
    rows.append({"response_validation": "passed"})
    rows.append({"response_validation": "failed"})
    measured = matrix._counts(rows, "response_validation")
    assert measured["declared_case_count"] == 4
    assert measured["independent_validation_attempted_count"] == 2
    assert measured["attempted_fraction_of_declared"] == 0.5
    assert measured["validated_evidence_coverage_fraction_of_declared"] == 0.25
    assert measured["pass_fraction_given_attempt"] == 0.5


def test_manifest_mismatch_refuses_a_changed_measurement(tmp_path, monkeypatch):
    evidence = tmp_path / "graphify-out/fv-root-cause-20260919"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "HERE", evidence)
    original = b'{"value": 1}'
    (evidence / "measurement.json").write_bytes(original)
    (tmp_path / "source.txt").write_text("original")
    manifest = {"sha256": {
        "graphify-out/fv-root-cause-20260919/measurement.json": hashlib.sha256(original).hexdigest(),
        "source.txt": hashlib.sha256(b"original").hexdigest(),
    }}
    (evidence / "manifest.json").write_text(json.dumps(manifest))
    _, current_source_match = matrix._require_manifest("manifest.json", "measurement.json")
    assert current_source_match is True
    (evidence / "measurement.json").write_text('{"value": 2}')
    with pytest.raises(ValueError, match="differs from its manifest"):
        matrix._require_manifest("manifest.json", "measurement.json")
    (evidence / "measurement.json").write_bytes(original)
    (tmp_path / "source.txt").write_text("changed")
    with pytest.raises(ValueError, match="manifest source/evidence mismatch"):
        matrix._require_manifest("manifest.json", "measurement.json")


def _write_source_registry(evidence, *, path, source_hash, snapshot_path, monkeypatch):
    registry = {
        "schema_version": 1,
        "sources": [{
            "path": path,
            "sha256": source_hash,
            "snapshot": snapshot_path,
            "source_commit": "historical-commit",
        }],
    }
    registry_path = evidence / matrix.SOURCE_SNAPSHOT_REGISTRY
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(
        matrix, "SOURCE_SNAPSHOT_REGISTRY_SHA256", hashlib.sha256(registry_path.read_bytes()).hexdigest()
    )


def _write_snapshot_manifest(tmp_path, evidence, *, source=b"original"):
    artifact = b'{"value": 1}'
    (evidence / "measurement.json").write_bytes(artifact)
    (tmp_path / "source.txt").write_bytes(b"changed")
    manifest = {"sha256": {
        "graphify-out/fv-root-cause-20260919/measurement.json": hashlib.sha256(artifact).hexdigest(),
        "source.txt": hashlib.sha256(source).hexdigest(),
    }}
    (evidence / "manifest.json").write_text(json.dumps(manifest))
    return source


def test_manifest_accepts_only_registered_historical_source_snapshot(tmp_path, monkeypatch):
    evidence = tmp_path / "graphify-out/fv-root-cause-20260919"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "HERE", evidence)
    source = _write_snapshot_manifest(tmp_path, evidence)
    snapshot_path = "source_snapshots/source.py"
    snapshot = evidence / snapshot_path
    snapshot.parent.mkdir()
    snapshot.write_bytes(source)
    _write_source_registry(
        evidence,
        path="source.txt",
        source_hash=hashlib.sha256(source).hexdigest(),
        snapshot_path=snapshot_path,
        monkeypatch=monkeypatch,
    )

    _, current_source_match = matrix._require_manifest("manifest.json", "measurement.json")
    assert current_source_match is False


def test_manifest_refuses_tampered_historical_source_snapshot(tmp_path, monkeypatch):
    evidence = tmp_path / "graphify-out/fv-root-cause-20260919"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "HERE", evidence)
    source = _write_snapshot_manifest(tmp_path, evidence)
    snapshot_path = "source_snapshots/source.py"
    snapshot = evidence / snapshot_path
    snapshot.parent.mkdir()
    snapshot.write_bytes(source)
    _write_source_registry(
        evidence,
        path="source.txt",
        source_hash=hashlib.sha256(source).hexdigest(),
        snapshot_path=snapshot_path,
        monkeypatch=monkeypatch,
    )
    snapshot.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="historical source snapshot differs from its pinned hash"):
        matrix._require_manifest("manifest.json", "measurement.json")


def test_manifest_refuses_tampered_source_snapshot_registry(tmp_path, monkeypatch):
    evidence = tmp_path / "graphify-out/fv-root-cause-20260919"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "HERE", evidence)
    source = _write_snapshot_manifest(tmp_path, evidence)
    snapshot_path = "source_snapshots/source.py"
    snapshot = evidence / snapshot_path
    snapshot.parent.mkdir()
    snapshot.write_bytes(source)
    _write_source_registry(
        evidence,
        path="source.txt",
        source_hash=hashlib.sha256(source).hexdigest(),
        snapshot_path=snapshot_path,
        monkeypatch=monkeypatch,
    )
    registry = evidence / matrix.SOURCE_SNAPSHOT_REGISTRY
    registry.write_text(registry.read_text() + " ")

    with pytest.raises(ValueError, match="source snapshot registry differs from its pinned hash"):
        matrix._require_manifest("manifest.json", "measurement.json")


def test_manifest_refuses_unregistered_historical_source(tmp_path, monkeypatch):
    evidence = tmp_path / "graphify-out/fv-root-cause-20260919"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "HERE", evidence)
    source = _write_snapshot_manifest(tmp_path, evidence)
    snapshot_path = "source_snapshots/other.py"
    snapshot = evidence / snapshot_path
    snapshot.parent.mkdir()
    snapshot.write_bytes(source)
    _write_source_registry(
        evidence,
        path="other.txt",
        source_hash=hashlib.sha256(source).hexdigest(),
        snapshot_path=snapshot_path,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ValueError, match="manifest source/evidence mismatch: source.txt"):
        matrix._require_manifest("manifest.json", "measurement.json")


def test_manifest_refuses_snapshot_path_outside_evidence_root(tmp_path, monkeypatch):
    evidence = tmp_path / "graphify-out/fv-root-cause-20260919"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "HERE", evidence)
    source = _write_snapshot_manifest(tmp_path, evidence)
    snapshot_path = "../source.py"
    (evidence / snapshot_path).write_bytes(source)
    _write_source_registry(
        evidence,
        path="source.txt",
        source_hash=hashlib.sha256(source).hexdigest(),
        snapshot_path=snapshot_path,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ValueError, match="historical source snapshot is missing or outside the evidence root"):
        matrix._require_manifest("manifest.json", "measurement.json")
