"""Small fail-closed checks for the PR213 alternate-root acceptance gates."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import json
import shutil
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_partial_alternate_root_probe as probe
from examples.weather_scenarios import fv_partial_alternate_root_runner as runner
from advar.local_refinement import RefinementTrial


def _branch(signature: str) -> dict[str, object]:
    return {"choices": [0, 1 if signature == "same" else 2],
            "face_signs": [1, -1]}


def _trial(*, same_signature: bool = False) -> RefinementTrial:
    current = _branch("same")
    candidate = _branch("same" if same_signature else "changed")
    return RefinementTrial(
        iteration=2, backtrack=3,
        current_objective=4.0, candidate_objective=8.0,
        current_gradient_norm=2.0, candidate_gradient_norm=1.0,
        current_gradient_max=1.5, candidate_gradient_max=0.8,
        current_branch=current, candidate_branch=candidate,
        step_scale=0.125, normalized_slope=-0.2, armijo_ratio=0.5,
    )


def test_changed_signature_uses_measured_phi_decrease_only() -> None:
    records: list[dict[str, object]] = []
    result = probe.measured_switch_acceptance(_trial(), records.append)

    assert result == (True, "measured_phi_decrease")
    old_phi, new_phi = records[0]["old_phi"], records[0]["new_phi"]
    decrease, floor = (records[0]["measured_phi_decrease"],
                       records[0]["required_decrease"])
    assert isinstance(old_phi, (int, float)) and old_phi == 2.0
    assert isinstance(new_phi, (int, float)) and new_phi == 0.5
    assert isinstance(decrease, (int, float)) and math.isfinite(decrease)
    assert isinstance(floor, (int, float)) and decrease > floor
    assert "objective" not in records[0]


def test_same_signature_keeps_refiners_normalized_armijo() -> None:
    assert probe.measured_switch_acceptance(_trial(same_signature=True)) is None


def test_seed_control_is_reconstructed_from_hash_pinned_pr209_raw_trial() -> None:
    control, row, prior = probe._archived_seed_control()

    assert control.shape == (26,)
    assert probe._tensor_sha(control) == probe.SEED_CONTROL_SHA256
    assert row["iteration"] == 5
    assert row["backtrack"] == 3
    assert row["candidate_control_sha256"] == probe.SEED_CONTROL_SHA256
    assert prior["input_before"] == prior["input_after"]


def test_corrupt_pr212_manifest_cannot_be_a_completed_numerical_refusal(monkeypatch) -> None:
    monkeypatch.setattr(probe, "SEED_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(probe._SeedGateRefusal, match="manifest hash changed"):
        probe._seed_evidence()
    assert not runner._valid_refusal(
        {"phase": "seed_recheck", "refusal": "manifest hash changed"},
        "seed_gate_refused", 2,
    )


@pytest.mark.parametrize("mutation", ["seed_status", "resource_exit", "source_pin"])
def test_resealed_wrong_pr212_evidence_is_execution_error(monkeypatch, tmp_path, mutation: str) -> None:
    archived = tmp_path / "partial_alternate_seed_attempt1"
    shutil.copytree(probe.SEED_ARCHIVE, archived)
    manifest_path = archived / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "seed_status":
        path = archived / "alternate_seed.json"
        value = json.loads(path.read_text())
        value["numerical_status"] = "seed_curvature_refused"
        path.write_text(json.dumps(value))
    elif mutation == "resource_exit":
        path = archived / "alternate_seed.resource.json"
        value = json.loads(path.read_text())
        value["exit_code"] = 2
        path.write_text(json.dumps(value))
    else:
        manifest["reviewed_probe_sha256"] = "0" * 64
    manifest["file_sha256"] = {
        f"partial_alternate_seed_attempt1/{name}": probe._sha(archived / name)
        for name in probe.SEED_RECORDS
    }
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "SEED_ARCHIVE", archived)
    monkeypatch.setattr(probe, "SEED_MANIFEST_SHA256", probe._sha(manifest_path))
    with pytest.raises(probe._SeedGateRefusal):
        probe._seed_evidence()
    assert not runner._valid_refusal(
        {"phase": "seed_recheck", "refusal": "bad PR212 evidence"},
        "seed_gate_refused", 2,
    )


@pytest.mark.parametrize("archive", [probe.SEED_ARCHIVE, probe.seed_gate.ARCHIVE])
def test_direct_probe_cannot_write_inside_either_raw_archive(archive: Path) -> None:
    directory = archive / "never_create_alternate_root"
    assert not directory.exists()
    with pytest.raises(ValueError, match="outside raw archives"):
        probe.run(directory / "result.json", expected_plan_sha256="x",
                  expected_probe_sha256="y", expected_runner_sha256="z")
    assert not directory.exists()


def test_changed_signature_rejects_non_decrease_at_roundoff_floor() -> None:
    trial = _trial()
    no_change = replace(trial, candidate_gradient_norm=trial.current_gradient_norm)
    result = probe.measured_switch_acceptance(no_change)
    assert result == (False, "measured_phi_refused")


def test_publication_requires_true_residual_and_switch_policy_evidence() -> None:
    control = [0.0] * 26
    control_hash = probe._tensor_sha(torch.tensor(control, dtype=torch.float64))
    accepted = {
        "iteration": 1, "backtrack": 0, "accepted": True,
        "candidate_control_sha256": control_hash, "candidate_control": control,
        "objective": 1.0, "gradient_norm": 0.5, "gradient_max": 0.4,
        "branch": {"euler_stages": 54, "signature_sha256": "b" * 64,
                   "minimum_scaled_slope_margin": 0.2,
                   "minimum_scaled_face_flux_margin": 0.3},
        "linear_relative_residual": 1e-12, "pcg_relative_residual": 1e-12,
        "pcg_iterations": 7, "armijo_ratio": 0.5,
    }
    solve = {"iteration": 1, "iterations": 7, "true_relative_residual": 1e-12,
             "reported_relative_residual": 1e-12, "rtol": 1e-10,
             "max_iterations": 104, "converged": True}
    branch_call = {"control_sha256": control_hash, "phase": "root_refinement",
                   "status": "positive_strict_branch",
                   "signature_sha256": "b" * 64,
                   "minimum_scaled_slope_margin": 0.2,
                   "minimum_scaled_face_flux_margin": 0.3}
    child: dict[str, Any] = {
        "trial_records": [accepted], "policy_records": [], "linear_solves": [solve],
        "branch_calls": [branch_call], "seed_gradient_norm": 1.0,
        "seed_signature_sha256": "b" * 64, "seed_control_sha256": "c" * 64,
    }
    assert runner._valid_candidate_trace(child, allow_solve_error=False)

    child["trial_records"][0]["linear_relative_residual"] = 1e-8
    assert not runner._valid_candidate_trace(child, allow_solve_error=False)

    child["trial_records"][0]["linear_relative_residual"] = 1e-12
    child["trial_records"][0]["candidate_control"] = None
    assert not runner._valid_candidate_trace(child, allow_solve_error=False)

    child["trial_records"][0]["candidate_control"] = control
    child["trial_records"][0]["backtrack"] = 1
    assert not runner._valid_candidate_trace(child, allow_solve_error=False)


@pytest.mark.parametrize("failure", ["nonconverged", "true_residual"])
def test_terminal_pcg_refusal_is_a_numerical_refusal(failure: str) -> None:
    nonconverged = failure == "nonconverged"
    solve: dict[str, Any] = {
        "iteration": 1, "iterations": 104, "rtol": 1e-10,
        "max_iterations": 104, "converged": not nonconverged,
        "reported_relative_residual": 2e-10,
        "true_relative_residual": 2e-10,
    }
    refusal = (
        "RefinementNumericalRefusal: matrix-free Newton PCG did not converge"
        if nonconverged else
        "RefinementNumericalRefusal: matrix-free Newton PCG true residual exceeds tolerance"
    )
    child: dict[str, Any] = {
        "phase": "root_refinement", "refusal": refusal,
        "trial_records": [], "policy_records": [], "linear_solves": [solve],
        "branch_calls": [], "seed_gradient_norm": 1.0,
        "seed_signature_sha256": "b" * 64, "seed_control_sha256": "c" * 64,
    }

    assert runner._valid_candidate_trace(child, allow_solve_error=True)
    assert runner._valid_refusal(child, "root_refused", 2)
    assert not runner._valid_candidate_trace(child, allow_solve_error=False)

    successful_root = {**child, "phase": "finished"}
    assert not runner._valid_candidate_trace(successful_root, allow_solve_error=False)


def test_terminal_pcg_refusal_rejects_unrelated_reason_and_followup_work() -> None:
    solve: dict[str, Any] = {
        "iteration": 1, "iterations": 104, "rtol": 1e-10,
        "max_iterations": 104, "converged": False,
        "reported_relative_residual": 2e-10,
        "true_relative_residual": 2e-10,
    }
    child: dict[str, Any] = {
        "phase": "root_refinement",
        "refusal": "RefinementNumericalRefusal: matrix-free Newton PCG did not converge",
        "trial_records": [], "policy_records": [], "linear_solves": [solve],
        "branch_calls": [], "seed_gradient_norm": 1.0,
        "seed_signature_sha256": "b" * 64, "seed_control_sha256": "c" * 64,
    }
    solve["error"] = "RuntimeError: unrelated runtime failure"
    assert not runner._valid_refusal(child, "root_refused", 2)
    solve.pop("error")
    child["refusal"] = "RuntimeError: unrelated runtime failure"
    assert not runner._valid_candidate_trace(child, allow_solve_error=True)
    child["refusal"] = (
        "RefinementNumericalRefusal: matrix-free Newton PCG did not converge"
    )
    child["trial_records"] = [{
        "iteration": 1, "backtrack": 0, "accepted": True,
        "candidate_control_sha256": "a" * 64,
    }]
    assert not runner._terminal_failed_solve(
        child, solve, child["trial_records"], child["linear_solves"])
    child["trial_records"] = []
    later_solve = {**solve, "iteration": 2, "converged": True}
    assert not runner._terminal_failed_solve(
        child, solve, [], [solve, later_solve])
    later_trial = {"iteration": 2, "backtrack": 0, "accepted": False}
    assert not runner._terminal_failed_solve(
        child, solve, [later_trial], [solve])


@pytest.mark.parametrize("field,value", [
    ("iterations", 0),
    ("iterations", 103),
    ("reported_relative_residual", 1e-10),
    ("true_relative_residual", 1e-10),
])
def test_nonconverged_pcg_refusal_requires_limit_and_both_residuals(
    field: str, value: object,
) -> None:
    solve: dict[str, Any] = {
        "iteration": 1, "iterations": 104, "rtol": 1e-10,
        "max_iterations": 104, "converged": False,
        "reported_relative_residual": 2e-10,
        "true_relative_residual": 2e-10,
    }
    child: dict[str, Any] = {
        "phase": "root_refinement",
        "refusal": "RefinementNumericalRefusal: matrix-free Newton PCG did not converge",
    }
    assert runner._terminal_failed_solve(child, solve, [], [solve])

    solve[field] = value
    assert not runner._terminal_failed_solve(child, solve, [], [solve])


def test_terminal_known_pcg_operator_refusal_is_accepted() -> None:
    solve: dict[str, Any] = {
        "iteration": 1, "rtol": 1e-10, "max_iterations": 104,
        "error": "RuntimeError: operator must be symmetric positive definite",
    }
    child: dict[str, Any] = {
        "phase": "root_refinement",
        "refusal": (
            "RefinementNumericalRefusal: operator must be symmetric positive definite"
        ),
        "trial_records": [], "policy_records": [], "linear_solves": [solve],
        "branch_calls": [], "seed_gradient_norm": 1.0,
        "seed_signature_sha256": "b" * 64, "seed_control_sha256": "c" * 64,
    }

    assert runner._valid_candidate_trace(child, allow_solve_error=True)
    assert runner._valid_refusal(child, "root_refused", 2)


def test_publication_requires_exact_policy_for_signature_switch() -> None:
    control = [0.0] * 26
    control_hash = probe._tensor_sha(torch.tensor(control, dtype=torch.float64))
    trial = {
        "iteration": 1, "backtrack": 0, "accepted": True,
        "candidate_control_sha256": control_hash, "candidate_control": control,
        "objective": 1.0, "gradient_norm": 0.5, "gradient_max": 0.4,
        "branch": {"euler_stages": 54, "signature_sha256": "b" * 64,
                   "minimum_scaled_slope_margin": 0.2,
                   "minimum_scaled_face_flux_margin": 0.3},
        "linear_relative_residual": 1e-12, "pcg_relative_residual": 1e-12,
        "pcg_iterations": 7, "armijo_ratio": 0.5,
    }
    solve = {"iteration": 1, "iterations": 7, "true_relative_residual": 1e-12,
             "reported_relative_residual": 1e-12, "rtol": 1e-10,
             "max_iterations": 104, "converged": True}
    policy = {"iteration": 1, "backtrack": 0,
              "current_signature_sha256": "c" * 64,
              "candidate_signature_sha256": "b" * 64,
              "old_phi": 0.5, "new_phi": 0.125,
              "measured_phi_decrease": 0.375,
              "required_decrease": 128 * torch.finfo(torch.float64).eps * 0.5,
              "accepted": True, "reason": "measured_phi_decrease"}
    child: dict[str, Any] = {
        "trial_records": [trial], "policy_records": [policy], "linear_solves": [solve],
        "branch_calls": [{"control_sha256": control_hash,
            "phase": "root_refinement",
            "status": "positive_strict_branch",
            "signature_sha256": "b" * 64,
            "minimum_scaled_slope_margin": 0.2,
            "minimum_scaled_face_flux_margin": 0.3}],
        "seed_gradient_norm": 1.0, "seed_signature_sha256": "c" * 64,
        "seed_control_sha256": "d" * 64,
    }
    assert runner._valid_candidate_trace(child, allow_solve_error=False)
    child["policy_records"] = []
    assert not runner._valid_candidate_trace(child, allow_solve_error=False)


def test_root_success_and_low_margin_are_distinct() -> None:
    branch: dict[str, object] = {
        "euler_stages": 54, "signature_sha256": "b" * 64,
        "minimum_scaled_slope_margin": 2e-4,
        "minimum_scaled_face_flux_margin": 3e-4,
    }
    control = [0.0] * 26
    control_hash = probe._tensor_sha(torch.tensor(control, dtype=torch.float64))
    accepted_trial: dict[str, Any] = {
        "iteration": 1, "backtrack": 0, "accepted": True,
        "candidate_control": control, "candidate_control_sha256": control_hash,
        "objective": 0.1, "gradient_norm": 0.5, "gradient_max": 0.4,
        "branch": branch,
        "linear_relative_residual": 1e-12, "pcg_relative_residual": 1e-12,
        "pcg_iterations": 7, "armijo_ratio": 0.5,
    }
    solve: dict[str, Any] = {
        "iteration": 1, "iterations": 7, "true_relative_residual": 1e-12,
        "reported_relative_residual": 1e-12, "rtol": 1e-10,
        "max_iterations": 104, "converged": True,
    }
    branch_calls: list[dict[str, object]] = [{
        "control_sha256": control_hash, "phase": "root_refinement",
        "status": "positive_strict_branch",
        "signature_sha256": "b" * 64,
        "minimum_scaled_slope_margin": 2e-4,
        "minimum_scaled_face_flux_margin": 3e-4,
    }, {
        "control_sha256": control_hash, "phase": "final_branch",
        "status": "positive_strict_branch", "signature_sha256": "b" * 64,
        "minimum_scaled_slope_margin": 2e-4,
        "minimum_scaled_face_flux_margin": 3e-4,
    }]
    child: dict[str, Any] = {
        "phase": "finished", "control": control,
        "final_control_sha256": control_hash,
        "final_curvature_control_sha256": control_hash,
        "final_gradient_max": 1e-11, "final_gradient_norm": 1e-10,
        "final_objective": 0.1, "refinement_iterations": 1,
        "final_curvature": {"symmetry_relative": 1e-14, "lambda_min": 1.0,
                             "lambda_max": 2.0, "lambda_ratio": 0.5,
                             "hvp_columns": 26},
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "final_branch": branch, "trial_records": [accepted_trial],
        "policy_records": [], "linear_solves": [solve],
        "seed_gradient_norm": 1.0,
        "seed_control_sha256": control_hash, "seed_signature_sha256": "b" * 64,
        "branch_calls": branch_calls,
        "response_margin_qualified": True,
    }
    assert runner._valid_root(child, "root_margin_qualified")
    branch["minimum_scaled_face_flux_margin"] = 8e-5
    accepted_trial["branch"]["minimum_scaled_face_flux_margin"] = 8e-5
    branch_calls[0]["minimum_scaled_face_flux_margin"] = 8e-5
    branch_calls[1]["minimum_scaled_face_flux_margin"] = 8e-5
    child["response_margin_qualified"] = False
    assert runner._valid_root(child, "root_low_margin")
    assert not runner._valid_root(child, "root_margin_qualified")

    child["refinement_iterations"] = 0
    child["trial_records"] = []
    child["linear_solves"] = []
    child["seed_control_sha256"] = control_hash
    assert not runner._valid_root(child, "root_low_margin")

    child["refinement_iterations"] = 1
    child["trial_records"] = [accepted_trial]
    child["linear_solves"] = [solve]
    branch_calls.pop()
    assert not runner._valid_root(child, "root_low_margin")
