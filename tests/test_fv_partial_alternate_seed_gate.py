"""Frozen alternate seed selection and no-response parent classification."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from examples.weather_scenarios import fv_partial_alternate_seed_gate as gate
from examples.weather_scenarios import fv_partial_alternate_seed_runner as runner


def _archive():
    return json.loads((gate.ARCHIVE / "partial_sector_root.json").read_text())


def test_selection_is_one_previously_refused_cross_sector_candidate():
    result = gate.select_seed(_archive())
    row = result["selected"]
    assert result["eligible_count"] == 29
    assert (row["iteration"], row["backtrack"]) == (5, 3)
    assert row["candidate_control_sha256"] == gate.SELECTED_CONTROL_SHA256
    assert row["branch"]["signature_sha256"] == gate.SELECTED_SIGNATURE_SHA256
    assert row["accepted"] is False
    assert row["policy_reason"] == "merit_switch_refused"
    assert row["branch"]["minimum_scaled_face_flux_margin"] > 1e-4


def test_changed_archived_control_hash_is_rejected():
    child = _archive()
    child["trial_records"][0]["candidate_control_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="control hash mismatch"):
        gate.select_seed(child)


def _eligible_child() -> dict[str, Any]:
    archived = _archive()
    row = gate.select_seed(archived)["selected"]
    return {
        "pid": 123, "phase": "finished", "numerical_status": "seed_locally_spd",
        "scope": "test", "response_validation": "not_performed",
        "physical_validation": "not_performed", "environment": {},
        "archive_manifest_sha256": gate.ARCHIVE_MANIFEST_SHA256,
        "plan_sha256": "p", "reviewed_probe_sha256": "q",
        "source_before": {}, "source_after": {},
        "input_before": archived["input_before"], "input_after": archived["input_before"],
        "selected_iteration": 5, "selected_backtrack": 3,
        "eligible_candidate_count": 29,
        "selected_control_sha256": gate.SELECTED_CONTROL_SHA256,
        "selected_signature_sha256": gate.SELECTED_SIGNATURE_SHA256,
        "selected_previously_accepted": False,
        "selected_previous_policy_reason": "merit_switch_refused",
        "new_product_gn_runs": 0, "new_newton_refinement_runs": 0,
        "new_adjoint_reanalysis_runs": 0,
        "objective": row["objective"], "gradient_norm": row["gradient_norm"],
        "gradient_max": row["gradient_max"], "elapsed_seconds": 1.0,
        "raw_unchanged": True,
        "seed_branch": row["branch"],
        "curvature": {"hvp_columns": 26, "symmetry_relative": 0.0,
                      "lambda_min": 1.0, "lambda_max": 2.0, "lambda_ratio": 0.5},
        "curvature_control_sha256": gate.SELECTED_CONTROL_SHA256,
    }


@pytest.mark.parametrize("mutation", [
    "none", "response", "wrong_curvature_hash", "low_margin", "wrong_exit",
    "accepted_seed", "wrong_input", "newton_count",
])
def test_parent_requires_only_seed_evidence(mutation):
    child = _eligible_child()
    exit_code = 0
    if mutation == "response":
        child["response"] = {"total_gradient": [0.0]}
    elif mutation == "wrong_curvature_hash":
        child["curvature_control_sha256"] = "0" * 64
    elif mutation == "low_margin":
        child["seed_branch"]["minimum_scaled_face_flux_margin"] = 5e-5
    elif mutation == "wrong_exit":
        exit_code = 2
    elif mutation == "accepted_seed":
        child["selected_previously_accepted"] = True
    elif mutation == "wrong_input":
        child["input_before"] = {"wrong": True}
    elif mutation == "newton_count":
        child["new_newton_refinement_runs"] = 1
    assert runner._valid_child(child, exit_code) is (mutation in {"none", "wrong_input"})
    assert runner._archive_matches_child(child) is (mutation not in {"wrong_input", "low_margin"})


def test_gate_cannot_write_into_archived_raw_directory():
    destination = gate.ARCHIVE / "never_write_alternate_seed.json"
    assert not destination.exists()
    with pytest.raises(ValueError, match="outside raw archive"):
        gate.run(destination, expected_plan_sha256="x", expected_probe_sha256="y")
    assert not destination.exists()


def test_guarded_parent_cannot_create_directory_inside_raw_archive():
    directory = gate.ARCHIVE / "never_create_alternate_seed_run"
    assert not directory.exists()
    with pytest.raises(ValueError, match="outside raw archive"):
        runner.run(directory, expected_plan_sha256="x", expected_probe_sha256="y",
                   expected_runner_sha256="z")
    assert not directory.exists()


def test_direct_cli_cannot_create_directory_inside_raw_archive():
    directory = gate.ARCHIVE / "never_create_alternate_seed_cli"
    assert not directory.exists()
    completed = subprocess.run(
        [sys.executable, str(Path(gate.__file__)), "--output",
         str(directory / "alternate_seed.json"), "--expected-plan-sha256", "x",
         "--expected-probe-sha256", "y"],
        cwd=gate.ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "outside raw archive" in completed.stderr
    assert not directory.exists()


def test_reviewed_plan_and_probe_hashes_are_required_before_fv_work(monkeypatch, tmp_path):
    monkeypatch.setattr(gate.prior, "_preflight_identity", lambda: pytest.fail(
        "unreviewed source must fail before FV preflight"))
    with pytest.raises(ValueError, match="reviewed alternate seed plan or probe changed"):
        gate.run(tmp_path / "alternate_seed.json",
                 expected_plan_sha256="0" * 64, expected_probe_sha256="0" * 64)


def test_status_helper_source_mismatch_fails_before_fv_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "PRIOR_STATUS_SOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(gate.prior, "_preflight_identity", lambda: pytest.fail(
        "status helper mismatch must fail before FV preflight"))
    with pytest.raises(ValueError, match="reviewed alternate seed plan or probe changed"):
        gate.run(tmp_path / "alternate_seed.json",
                 expected_plan_sha256=gate._sha(gate.PLAN),
                 expected_probe_sha256=gate._sha(Path(gate.__file__)))


def test_guarded_parent_rejects_unreviewed_runner_before_child(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "run_guarded", lambda *_args, **_kwargs: pytest.fail(
        "unreviewed parent must not launch a child"))
    with pytest.raises(ValueError, match="reviewed alternate seed source or plan changed"):
        runner.run(tmp_path / "attempt",
                   expected_plan_sha256=gate._sha(gate.PLAN),
                   expected_probe_sha256=gate._sha(Path(gate.__file__)),
                   expected_runner_sha256="0" * 64)
