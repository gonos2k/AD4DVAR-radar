from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_first_branch_probe as first_branch
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed_linear
from examples.weather_scenarios import fv_point_3h_merit_step_probe as probe


def test_joint_direction_is_exactly_negative_hg_and_both_slopes_are_negative() -> None:
    gradient = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    hg = torch.tensor([3.0, -0.5, 0.25], dtype=torch.float64)

    direction, record = probe._joint_direction(gradient, hg)

    assert direction is not None
    assert torch.equal(direction, -hg / torch.linalg.vector_norm(hg))
    assert record["status"] == "passed"
    assert record["slope_j"] == pytest.approx(float(torch.dot(gradient, direction)))
    assert record["slope_j"] < 0
    assert record["slope_phi"] == pytest.approx(-float(torch.linalg.vector_norm(hg)))
    assert record["slope_phi"] < 0


def test_joint_direction_refuses_when_curvature_gate_does_not_clear_roundoff() -> None:
    gradient = torch.tensor([1.0, 0.0], dtype=torch.float64)
    hg = torch.tensor([0.0, 1.0], dtype=torch.float64)

    direction, record = probe._joint_direction(gradient, hg)

    assert direction is None
    assert record["status"] == "no_joint_descent_direction"
    assert record["reason"] == "curvature_gate_failed"


def test_changed_signature_accepts_only_actual_joint_decrease_without_armijo() -> None:
    gate = probe._candidate_gate(
        seed_j=10.0, seed_phi=5.0, candidate_j=9.0, candidate_phi=4.0,
        slope_j=-1.0, slope_phi=-1.0, alpha=0.02, same_signature=False,
    )

    assert gate["accepted"] is True
    assert gate["branch_changed"] is True
    assert gate["actual_decrease"] is True
    assert gate["armijo_required"] is False
    assert gate["armijo_passed"] is None


def test_same_signature_refuses_candidate_that_decreases_but_fails_armijo() -> None:
    gate = probe._candidate_gate(
        seed_j=10.0, seed_phi=5.0, candidate_j=9.999999, candidate_phi=4.999999,
        slope_j=-1.0, slope_phi=-1.0, alpha=0.02, same_signature=True,
    )

    assert gate["actual_decrease"] is True
    assert gate["armijo_required"] is True
    assert gate["accepted"] is False
    assert "analysis_objective_armijo_failed" in gate["refusal_reasons"]
    assert "stationarity_merit_armijo_failed" in gate["refusal_reasons"]


def test_same_signature_refuses_without_roundoff_resolvable_joint_decrease() -> None:
    gate = probe._candidate_gate(
        seed_j=1.0, seed_phi=1.0, candidate_j=1.0, candidate_phi=0.5,
        slope_j=-1.0, slope_phi=-1.0, alpha=0.02, same_signature=True,
    )

    assert gate["accepted"] is False
    assert "analysis_objective_not_decreased_beyond_roundoff" in gate["refusal_reasons"]


def _fake_child(source: dict[str, str], archive: dict[str, object]) -> dict[str, object]:
    identity = {
        "archived_input": archive,
        "control_sha256": seed_linear.SHIFTED_SEED_CONTROL_SHA256,
        "parameters_sha256": seed_linear.EXPECTED_PARAMETERS_SHA256,
        "terminal_truth_sha256": seed_linear.EXPECTED_TRUTH_SHA256,
        "changed_control_index": seed_linear.EXPECTED_CHANGED_CONTROL,
        "changed_control_indices": [seed_linear.EXPECTED_CHANGED_CONTROL],
    }
    return {
        "pid": 456, "execution_status": "completed", "execution_phase": "finished",
        "numerical_status": "seed_branch_refused", "candidate_attempts": [],
        "direction_status": "not_attempted", "branch_status": "refused",
        "accepted_candidate": None, "candidate_seed": {
            "control_sha256": seed_linear.SHIFTED_SEED_CONTROL_SHA256,
            "parameters_sha256": seed_linear.EXPECTED_PARAMETERS_SHA256,
            "terminal_truth_sha256": seed_linear.EXPECTED_TRUTH_SHA256,
        },
        "source_before": source, "source_after": source, "source_unchanged": True,
        "plan_sha256": probe.PLAN_SHA256, "plan_unchanged": True,
        "hessian_evidence_sha256": probe.HESSIAN_EVIDENCE_SHA256,
        "hessian_evidence_unchanged": True,
        "hessian_evidence_relative_symmetry": probe.SEED_HESSIAN_RELATIVE_SYMMETRY,
        "archive_input_sha256": probe.ARCHIVED_INPUT_SHA256,
        "archive_before": archive, "archive_unchanged": True,
        "input_before": identity, "input_after": identity, "input_unchanged": True,
        "response_computed": False, "adjoint_computed": False,
        "score_computed": False, "pcg_calls": 0,
        "finite_segment_certified": False, "segment_certified": False,
        "wall_limit_seconds": probe.WALL_SECONDS,
        "termination_reap_grace_seconds": probe.REAP_GRACE_SECONDS,
        "rss_limit_bytes": probe.SAMPLED_RSS_BYTES,
        "costs": {"candidate_count": 0, "branch_oracle_calls": 1,
                  "exact_hvp_calls": 0, "newton_steps": 0},
    }


def _fake_resource(command: list[str]) -> dict[str, object]:
    return {
        "command": command, "child_pid": 456, "exit_code": 0,
        "resource_termination": None, "monitor_error": None,
        "elapsed_seconds": 1.0, "sampled_peak_rss_bytes": 10_000_000,
        "rss_samples": 4, "wall_limit_seconds": probe.WALL_SECONDS,
        "rss_limit_bytes": probe.SAMPLED_RSS_BYTES,
    }


def _fake_line_search_child(source: dict[str, str],
                            archive: dict[str, object]) -> dict[str, object]:
    child = _fake_child(source, archive)
    child.update(
        numerical_status="line_search_refused", branch_status="passed_strict",
        seed_branch_signature_sha256=probe.EXPECTED_SEED_BRANCH_SIGNATURE_SHA256,
        direction_status="passed",
        seed={"objective": 2.0, "phi": 1.0, "gradient_l2": 1.0,
              "gradient_inf": 1.0},
        direction={"status": "passed", "slope_j": -1.0, "slope_phi": -1.0,
                   "curvature_dot": 1.0, "curvature_threshold": 1e-14,
                   "gradient_l2": 1.0, "hg_l2": 1.0,
                   "direction_sha256": "a" * 64},
        candidate_attempts=[
            {"backtrack": index, "alpha": probe.INITIAL_ALPHA / 2**index,
             "accepted": False, "branch_status": "refused_strict_branch"}
            for index in range(probe.MAX_BACKTRACKS)
        ],
        accepted_candidate=None,
        costs={"candidate_count": probe.MAX_BACKTRACKS,
               "branch_oracle_calls": 1 + probe.MAX_BACKTRACKS,
               "exact_hvp_calls": 1, "newton_steps": 0},
    )
    return child


def test_parent_rejects_accepted_status_without_a_matching_accepted_trial() -> None:
    source = probe._source_hashes()
    archive = first_branch._archive_input_identity()
    child = _fake_line_search_child(source, archive)
    assert probe._valid_child(child, source, archive)

    child["numerical_status"] = "candidate_accepted_same_signature"

    assert not probe._valid_child(child, source, archive)


def test_parent_recomputes_gate_instead_of_trusting_forged_candidate_decrease() -> None:
    source = probe._source_hashes()
    archive = first_branch._archive_input_identity()
    child = _fake_line_search_child(source, archive)
    attempts = child["candidate_attempts"]
    assert isinstance(attempts, list)
    first = attempts[0]
    assert isinstance(first, dict)
    first.update(
        branch_status="passed_strict", branch_changed=False,
        objective_status="finite", objective=2.1, phi=0.9,
        gradient_l2=1.0, gradient_inf=1.0, accepted=False,
        gate={"accepted": True, "branch_changed": False,
              "actual_decrease": True, "armijo_required": True,
              "armijo_passed": True},
    )

    assert not probe._valid_child(child, source, archive)


def test_parent_rejects_same_signature_trial_falsely_marked_branch_changed() -> None:
    source = probe._source_hashes()
    archive = first_branch._archive_input_identity()
    child = _fake_line_search_child(source, archive)
    control = torch.zeros(probe.EXPECTED_CONTROLS, dtype=torch.float64)
    control_sha = probe._tensor_sha(control)
    signature_sha = probe.EXPECTED_SEED_BRANCH_SIGNATURE_SHA256
    alpha = probe.INITIAL_ALPHA
    seed_j, seed_phi = 2.0, 1.0
    candidate_j, candidate_phi = 1.999999, 0.999999
    forged_gate = probe._candidate_gate(
        seed_j=seed_j, seed_phi=seed_phi, candidate_j=candidate_j,
        candidate_phi=candidate_phi, slope_j=-1.0, slope_phi=-1.0,
        alpha=alpha, same_signature=False,
    )
    assert forged_gate["accepted"]
    assert candidate_j > forged_gate["armijo_limits"]["j"]
    assert candidate_phi > forged_gate["armijo_limits"]["phi"]
    attempt = {
        "backtrack": 0, "alpha": alpha, "accepted": True,
        "branch_status": "passed_strict", "branch_signature_sha256": signature_sha,
        "branch_changed": True, "segment_certified": False,
        "control_sha256": control_sha, "objective_status": "finite",
        "objective": candidate_j, "phi": candidate_phi,
        "gradient_l2": 1.0, "gradient_inf": 1.0, "gate": forged_gate,
    }
    candidate = {
        "control": control.tolist(), "control_sha256": control_sha,
        "branch_signature_sha256": signature_sha, "branch_changed": True,
        "alpha": alpha, "backtrack": 0, "objective": candidate_j,
        "phi": candidate_phi, "gradient_inf": 1.0,
        "segment_certified": False,
    }
    child.update(
        numerical_status="candidate_accepted_branch_changed",
        seed_branch_signature_sha256=signature_sha,
        seed={"objective": seed_j, "phi": seed_phi,
              "gradient_l2": 1.0, "gradient_inf": 1.0},
        direction={"slope_j": -1.0, "slope_phi": -1.0,
                   "curvature_dot": 1.0, "curvature_threshold": 1e-14,
                   "gradient_l2": 1.0, "hg_l2": 1.0,
                   "direction_sha256": "a" * 64},
        candidate_attempts=[attempt], accepted_candidate=candidate,
        costs={"candidate_count": 1, "branch_oracle_calls": 2,
               "exact_hvp_calls": 1, "newton_steps": 0},
    )

    assert not probe._valid_child(child, source, archive)


@pytest.mark.parametrize("mutation", ["child_identity", "guard_exit"])
def test_parent_guard_rejects_fake_child_or_resource_record_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    source = probe._source_hashes()
    archive = first_branch._archive_input_identity()

    def fake_guard(command: list[str], *, report_path: Path, log_path: Path,
                   wall_seconds: int, rss_bytes: int) -> dict[str, object]:
        assert wall_seconds == probe.WALL_SECONDS
        assert rss_bytes == probe.SAMPLED_RSS_BYTES
        child = _fake_child(source, archive)
        resource = _fake_resource(command)
        if mutation == "child_identity":
            child["source_unchanged"] = False
            Path(command[-1]).write_text(json.dumps(child))
        else:
            Path(command[-1]).write_text(json.dumps(child))
            resource["exit_code"] = 1
        report_path.write_text(json.dumps(resource))
        log_path.write_text("fake guarded child\n")
        return resource

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    monkeypatch.setattr(seed_linear, "_prepare_fixed_seed",
                        lambda: pytest.fail("parent must not build the FV fixture"))

    result = probe.run(tmp_path / f"attempt-{mutation}")

    assert result["execution_status"] == "failed"
    if mutation == "child_identity":
        assert result["child_report_valid"] is False
    else:
        assert result["child_report_valid"] is True
