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
    matrix._require_manifest("manifest.json", "measurement.json")
    (evidence / "measurement.json").write_text('{"value": 2}')
    with pytest.raises(ValueError, match="differs from its manifest"):
        matrix._require_manifest("manifest.json", "measurement.json")
    (evidence / "measurement.json").write_bytes(original)
    (tmp_path / "source.txt").write_text("changed")
    with pytest.raises(ValueError, match="manifest source/evidence mismatch"):
        matrix._require_manifest("manifest.json", "measurement.json")
