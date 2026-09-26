from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_shifted_branch_probe as probe


def test_shift_changes_only_second_flow_latent_at_control_21():
    control = torch.zeros(26, dtype=torch.float64)
    control[20:25] = torch.atanh(torch.tensor(
        probe.EXPECTED_FRACTIONS, dtype=torch.float64,
    ))
    before = control.clone()

    shifted = probe._shift_control(control, field_size=20)

    assert torch.equal(control, before)
    assert torch.nonzero(shifted != control, as_tuple=False).flatten().tolist() == [21]
    assert torch.equal(shifted[torch.arange(26) != 21], control[torch.arange(26) != 21])
    assert torch.allclose(
        torch.tanh(shifted[20:25]),
        torch.tensor(probe.SHIFTED_FRACTIONS, dtype=torch.float64),
        rtol=0,
        atol=2e-15,
    )


def test_shift_rejects_unexpected_original_fractions():
    with pytest.raises(ValueError, match="original flow fractions"):
        probe._shift_control(torch.zeros(26, dtype=torch.float64), field_size=20)


def _static_problem() -> tuple[SimpleNamespace, torch.Tensor]:
    dtype = torch.float64
    y, x = torch.meshgrid(torch.arange(3, dtype=dtype),
                          torch.arange(3, dtype=dtype), indexing="ij")
    basis = torch.stack((y, x))
    limits = torch.tensor([0.01, 0.01], dtype=dtype)
    spec = SimpleNamespace(
        coefficient_limits=limits, psi_basis=basis,
        substeps_per_interval=90, spacing_yx=(10.0, 10.0),
        reconstruction="minmod", max_courant=0.5,
    )
    frozen = SimpleNamespace(
        active_field_index=torch.empty(0, dtype=torch.int64),
        fv_transport=spec,
        nowcast_config=SimpleNamespace(interval_minutes=10),
    )
    return SimpleNamespace(frozen=frozen), torch.tensor(
        [math.atanh(0.3), math.atanh(-0.4)], dtype=dtype,
    )


def test_static_face_gate_uses_production_operators_and_finds_nonzero_fluxes():
    problem, control = _static_problem()

    result = probe._static_face_summary(control, problem)

    assert result["status"] == "passed"
    assert result["near_zero_face_count"] == 0
    assert result["face_count"] == 12
    assert result["minimum_abs_flux"] > result["strict_threshold"]
    assert len(result["face_flux_sha256"]) == 64


def _source_identity(monkeypatch, tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    source = {"probe.py": "source-hash"}
    archived = {"problem": "fixed"}
    monkeypatch.setattr(probe, "_source_hashes", lambda: source)
    monkeypatch.setattr(probe.first_branch, "_sources_match_archive", lambda _value: True)
    monkeypatch.setattr(probe.first_branch, "_archive_input_identity", lambda: archived)
    monkeypatch.setattr(probe, "_sha", lambda path: probe.PLAN_SHA256 if path == probe.PLAN else "archive")
    return source, archived


def _prepared_case(monkeypatch, source, archived):
    problem = SimpleNamespace(
        layout={"controls": 26, "parameters": 13, "euler_stages": 3600},
        frozen=SimpleNamespace(active_field_index=torch.arange(20)),
    )
    original = torch.zeros(26, dtype=torch.float64)
    control = original.clone()
    control[21] = 0.2
    archived["control_sha256"] = probe._tensor_sha(original)
    parameters = torch.zeros(13, dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)

    def identity(_problem, _original, candidate, p, target):
        return {
            "archived_input": archived,
            "control_sha256": probe._tensor_sha(candidate),
            "parameters_sha256": probe._tensor_sha(p),
            "terminal_truth_sha256": probe._tensor_sha(target),
            "changed_control_index": 21,
            "changed_control_indices": [21],
            "original_flow_fractions": list(probe.EXPECTED_FRACTIONS),
            "shifted_flow_fractions": list(probe.SHIFTED_FRACTIONS),
        }

    input_identity = identity(problem, original, control, parameters, truth)
    monkeypatch.setattr(probe, "_prepare_shifted_case", lambda: (
        problem, original, control, parameters, truth, input_identity,
    ))
    monkeypatch.setattr(probe, "_input_identity", identity)
    monkeypatch.setattr(probe, "_static_face_summary", lambda *_args: {
        "status": "passed", "near_zero_face_count": 0,
        "face_count": 12, "strict_threshold": 1e-15,
        "minimum_abs_flux": 0.01, "maximum_abs_flux": 0.2,
        "coefficient_fractions": list(probe.SHIFTED_FRACTIONS),
        "face_flux_sha256": "a" * 64,
    })
    return problem, original, control, parameters, truth, input_identity


def test_child_recomputes_live_input_identity_after_diagnostic_mutation(monkeypatch, tmp_path):
    source, archived = _source_identity(monkeypatch, tmp_path)
    _, _, control, parameters, _, before = _prepared_case(monkeypatch, source, archived)

    def mutate_then_refuse(_forecast):
        control[21].add_(0.1)
        parameters[0].add_(1.0)
        return {"status": "refused_first_branch", "stage_index": 0}

    monkeypatch.setattr(probe.first_branch, "observe_first_failure", mutate_then_refuse)
    report = probe.run_probe(tmp_path / "raw.json")

    assert report["execution_status"] == "failed"
    assert report["execution_phase"] == "identity_refused"
    assert report["diagnostic_status"] == "not_performed"
    assert report["input_before"] == before
    assert report["input_after"] != before
    assert report["input_unchanged"] is False


def test_static_face_refusal_stops_before_candidate_forecast(monkeypatch, tmp_path):
    source, archived = _source_identity(monkeypatch, tmp_path)
    _prepared_case(monkeypatch, source, archived)
    monkeypatch.setattr(probe, "_static_face_summary", lambda *_args: {
        "status": "refused_static_face", "near_zero_face_count": 1,
        "face_count": 12, "strict_threshold": 1e-15,
        "minimum_abs_flux": 0.0, "maximum_abs_flux": 0.2,
        "coefficient_fractions": list(probe.SHIFTED_FRACTIONS),
        "face_flux_sha256": "a" * 64,
    })
    monkeypatch.setattr(
        probe.first_branch, "observe_first_failure",
        lambda *_args: pytest.fail("candidate rollout must not run after static refusal"),
    )

    report = probe.run_probe(tmp_path / "static-refusal.json")

    assert report["execution_status"] == "completed"
    assert report["diagnostic_status"] == "refused_static_face"
    assert report["stationarity_passed"] == "not_tested"
    assert report["response_computed"] is False


def test_full_oracle_record_is_compact_and_hashes_signature():
    signature = {
        "euler_stages": 3600,
        "choices": [{"choose_left": [[True]]}] * 3600,
        "face_signs": [{"qx": [[1]], "qy": [[-1]]}] * 3600,
    }
    problem = SimpleNamespace(branch_check=lambda _control, _parameters: (
        signature, "strict fixture branch",
    ))

    result = probe._branch_record(
        problem, torch.zeros(26, dtype=torch.float64), torch.zeros(13, dtype=torch.float64),
    )

    assert result["status"] == "passed_strict_oracle"
    assert result["euler_stages"] == 3600
    assert result["choice_stage_count"] == 3600
    assert result["face_sign_stage_count"] == 3600
    assert len(result["signature_sha256"]) == 64
    assert "choices" not in result and "face_signs" not in result


def _valid_child(
    source: dict[str, str], input_identity: dict[str, Any], *,
    diagnostic_status: str = "passed_strict_oracle",
) -> dict[str, Any]:
    archived_input = input_identity["archived_input"]
    child: dict[str, Any] = {
        "pid": 101, "execution_status": "completed", "execution_phase": "finished",
        "diagnostic_status": diagnostic_status, "stationarity_passed": "not_tested",
        "response_computed": False, "response_validation": "not_performed",
        "physical_validation": "not_performed", "forecast_score_computed": False,
        "source_before": source, "source_after": source, "source_unchanged": True,
        "layout": {
            "controls": 26, "parameters": 13, "state_shape": [4, 5],
            "observation_shape": [3, 4],
            "observation_times_seconds": [0.0, 600.0, 1200.0],
            "forecast_time_seconds": 12000.0, "euler_stages": 3600,
        },
        "input_before": input_identity, "input_after": input_identity,
        "candidate": {
            "changed_control_indices": input_identity["changed_control_indices"],
            "changed_control_index": input_identity["changed_control_index"],
            "original_fraction": input_identity["original_flow_fractions"][1],
            "shifted_fraction": input_identity["shifted_flow_fractions"][1],
            "original_control_sha256": archived_input["control_sha256"],
            "shifted_control_sha256": input_identity["control_sha256"],
            "parameters_sha256": input_identity["parameters_sha256"],
            "terminal_truth_sha256": input_identity["terminal_truth_sha256"],
        },
        "input_unchanged": True, "plan_sha256": probe.PLAN_SHA256,
        "plan_unchanged": True, "archive_input_sha256": probe.ARCHIVED_INPUT_SHA256,
        "archive_unchanged": True,
        "static_face_gate": {
            "status": "passed", "near_zero_face_count": 0, "face_count": 12,
            "strict_threshold": 1e-15, "minimum_abs_flux": 0.01,
            "maximum_abs_flux": 0.2,
            "coefficient_fractions": list(probe.SHIFTED_FRACTIONS),
            "face_flux_sha256": "a" * 64,
        },
        "trajectory_diagnostic": {
            "status": "passed_pointwise", "euler_stages": 3600,
            "stages_observed": 3600, "first_failure": None,
        },
        "full_strict_oracle": {
            "status": "passed_strict_oracle", "euler_stages": 3600,
            "choice_stage_count": 3600, "face_sign_stage_count": 3600,
            "signature_sha256": "b" * 64,
        },
    }
    return child


def test_guarded_parent_accepts_completed_pointwise_branch_pass(monkeypatch, tmp_path):
    source, archived = _source_identity(monkeypatch, tmp_path)
    _, _, _, _, _, input_identity = _prepared_case(monkeypatch, source, archived)
    child = _valid_child(source, input_identity)
    monkeypatch.setattr(probe, "_prepare_shifted_case", lambda: pytest.fail(
        "parent must not reconstruct the FV fixture outside the resource guard"
    ))

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path):
        assert wall_seconds == 180
        assert rss_bytes == 768 * 1024**2
        Path(command[-1]).write_text(json.dumps(child))
        return {
            "command": command, "child_pid": 101, "exit_code": 0,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 180, "elapsed_seconds": 3.0,
            "rss_limit_bytes": 768 * 1024**2, "rss_samples": 2,
            "sampled_peak_rss_bytes": 300_000_000,
        }

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / "guarded")

    assert result["execution_status"] == "completed"
    assert result["diagnostic_status"] == "passed_strict_oracle"
    assert result["stationarity_passed"] == "not_tested"
    assert result["response_computed"] is False


def test_guarded_parent_keeps_numerical_refusal_distinct_from_execution_failure(
    monkeypatch, tmp_path,
):
    source, archived = _source_identity(monkeypatch, tmp_path)
    _, _, _, _, _, input_identity = _prepared_case(monkeypatch, source, archived)
    child = _valid_child(source, input_identity, diagnostic_status="refused_first_branch")
    child.pop("full_strict_oracle")
    child["trajectory_diagnostic"] = {
        "status": "refused_first_branch", "stage_index": 0,
        "stages_observed": 1, "accepted_stages_before_failure": 0,
        "stage_context": probe.first_branch._stage_context(0),
        "first_failure": {"predicate": "x_left_slope_nonzero"},
    }

    def fake_guard(command, **_kwargs):
        Path(command[-1]).write_text(json.dumps(child))
        return {
            "command": command, "child_pid": 101, "exit_code": 0,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 180, "elapsed_seconds": 3.0,
            "rss_limit_bytes": 768 * 1024**2, "rss_samples": 2,
            "sampled_peak_rss_bytes": 300_000_000,
        }

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / "guarded-refusal")

    assert result["execution_status"] == "completed"
    assert result["diagnostic_status"] == "refused_first_branch"
    assert result["first_failure"]["predicate"] == "x_left_slope_nonzero"


def test_guarded_parent_rejects_nonzero_child_exit(monkeypatch, tmp_path):
    source, archived = _source_identity(monkeypatch, tmp_path)
    _, _, _, _, _, input_identity = _prepared_case(monkeypatch, source, archived)
    child = _valid_child(source, input_identity)

    def fake_guard(command, **_kwargs):
        Path(command[-1]).write_text(json.dumps(child))
        return {
            "command": command, "child_pid": 101, "exit_code": 2,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 180, "elapsed_seconds": 3.0,
            "rss_limit_bytes": 768 * 1024**2, "rss_samples": 2,
            "sampled_peak_rss_bytes": 300_000_000,
        }

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / "guarded-failure")

    assert result["execution_status"] == "failed"
    assert result["diagnostic_status"] == "passed_strict_oracle"


@pytest.mark.parametrize("contradiction", ["static", "full_oracle"])
def test_guarded_parent_rejects_status_evidence_contradictions(
    monkeypatch, tmp_path, contradiction,
):
    source, archived = _source_identity(monkeypatch, tmp_path)
    _, _, _, _, _, input_identity = _prepared_case(monkeypatch, source, archived)
    child = _valid_child(source, input_identity)
    if contradiction == "static":
        child["diagnostic_status"] = "refused_static_face"
        child.pop("trajectory_diagnostic")
        child.pop("full_strict_oracle")
    else:
        child["trajectory_diagnostic"]["stages_observed"] = 3599

    def fake_guard(command, **_kwargs):
        Path(command[-1]).write_text(json.dumps(child))
        return {
            "command": command, "child_pid": 101, "exit_code": 0,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 180, "elapsed_seconds": 3.0,
            "rss_limit_bytes": 768 * 1024**2, "rss_samples": 2,
            "sampled_peak_rss_bytes": 300_000_000,
        }

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / f"contradiction-{contradiction}")

    assert result["execution_status"] == "failed"


@pytest.mark.parametrize("field", [
    "original_control_sha256", "shifted_control_sha256", "parameters_sha256",
    "terminal_truth_sha256", "changed_control_index", "changed_control_indices",
    "original_fraction", "shifted_fraction",
])
def test_guarded_parent_rejects_candidate_report_mutations(monkeypatch, tmp_path, field):
    source, archived = _source_identity(monkeypatch, tmp_path)
    _, _, _, _, _, input_identity = _prepared_case(monkeypatch, source, archived)
    child = _valid_child(source, input_identity)
    if field.endswith("sha256"):
        child["candidate"][field] = "f" * 64
    elif field == "changed_control_indices":
        child["candidate"][field] = [20]
    elif field == "changed_control_index":
        child["candidate"][field] = 20
    else:
        child["candidate"][field] += 0.01

    def fake_guard(command, **_kwargs):
        Path(command[-1]).write_text(json.dumps(child))
        return {
            "command": command, "child_pid": 101, "exit_code": 0,
            "resource_termination": None, "monitor_error": None,
            "wall_limit_seconds": 180, "elapsed_seconds": 3.0,
            "rss_limit_bytes": 768 * 1024**2, "rss_samples": 2,
            "sampled_peak_rss_bytes": 300_000_000,
        }

    monkeypatch.setattr(probe, "run_guarded", fake_guard)
    result = probe.run(tmp_path / f"candidate-mutation-{field}")

    assert result["execution_status"] == "failed"
