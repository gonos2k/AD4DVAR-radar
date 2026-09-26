from __future__ import annotations

import json
import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest

import torch

from examples.weather_scenarios import fv_point_3h_merit_continuation as continuation
from examples.weather_scenarios import fv_point_3h_merit_step_probe as merit_step


def test_transpose_check_accepts_exact_quadratic_jvp_and_vjp() -> None:
    hessian = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    control = torch.tensor([0.4, -0.3], dtype=torch.float64)
    direction = torch.tensor([0.2, 0.8], dtype=torch.float64)
    gradient = lambda value: hessian @ value
    hg = torch.func.jvp(gradient, (control,), (direction,))[1]
    pullback = torch.func.vjp(gradient, control)[1]
    htg = pullback(direction)[0]

    result = continuation._transpose_consistency(hg, htg)

    assert result["status"] == "passed"
    assert result["relative_difference"] == 0.0


def test_transpose_check_refuses_mismatch_and_nonfinite_products() -> None:
    hg = torch.tensor([1.0, 2.0], dtype=torch.float64)
    mismatched = continuation._transpose_consistency(
        hg, torch.tensor([1.0, 2.01], dtype=torch.float64),
    )
    nonfinite = continuation._transpose_consistency(
        hg, torch.tensor([float("nan"), 2.0], dtype=torch.float64),
    )

    assert mismatched["reason"] == "transpose_consistency_failed"
    assert nonfinite["reason"] == "invalid_or_nonfinite_hessian_products"


def test_strict_endpoint_uses_one_complete_oracle_call() -> None:
    signature = {
        "euler_stages": continuation.EXPECTED_STAGES,
        "choices": [0] * continuation.EXPECTED_STAGES,
        "face_signs": [0] * continuation.EXPECTED_STAGES,
    }

    class FakeProblem:
        calls = 0

        def branch_check(self, control, parameters):
            self.calls += 1
            return signature, "fake full strict oracle"

    problem = FakeProblem()
    branch = continuation._strict_endpoint(
        problem, torch.zeros(continuation.EXPECTED_CONTROLS, dtype=torch.float64),
        torch.zeros(continuation.EXPECTED_PARAMETERS, dtype=torch.float64),
    )

    assert problem.calls == 1
    assert branch["status"] == "passed_strict_branch"
    assert branch["euler_stages"] == continuation.EXPECTED_STAGES


def test_pinned_seed_checks_raw_report_and_control_tensor_sha() -> None:
    control = continuation._seed_control()

    assert control.shape == (continuation.EXPECTED_CONTROLS,)
    assert control.dtype == torch.float64
    assert continuation.seed_linear._tensor_sha(control) == continuation.SEED_CONTROL_SHA256


def test_tail_policy_pins_pr223_refused_epoch_endpoint_without_fv() -> None:
    policy = continuation.TAIL_POLICY
    raw = continuation._seed_report(policy)
    control = continuation._seed_control(policy)

    assert raw["numerical_status"] == "epoch_line_search_refused"
    assert control.shape == (continuation.EXPECTED_CONTROLS,)
    assert continuation.seed_linear._tensor_sha(control) == policy.seed_control_sha256
    assert policy.alpha_grid == tuple(0.000078125 / 2**k for k in range(12))
    assert (policy.max_accepted_epochs, policy.max_total_trials,
            policy.internal_budget_seconds, policy.wall_seconds,
            policy.sampled_rss_bytes) == (2, 24, 240.0, 300, 1024**3)
    with pytest.raises(ValueError, match="accepted endpoint evidence changed"):
        continuation._seed_report(replace(policy, seed_report_status="candidate_accepted_branch_changed"))


def test_default_policy_preserves_historical_numeric_gates_and_cli_defaults() -> None:
    policy = continuation.DEFAULT_POLICY

    assert policy.plan == continuation.PLAN
    assert policy.plan_sha256 == continuation.PLAN_SHA256
    assert policy.seed_report_sha256 == continuation.SEED_REPORT_SHA256
    assert policy.seed_evidence_sha256 == continuation.SEED_EVIDENCE_SHA256
    assert policy.seed_control_sha256 == continuation.SEED_CONTROL_SHA256
    assert policy.seed_signature_sha256 == continuation.SEED_SIGNATURE_SHA256
    assert policy.alpha_grid == tuple(0.02 / 2**k for k in range(8))
    assert (policy.max_accepted_epochs, policy.max_total_trials,
            policy.internal_budget_seconds, policy.wall_seconds,
            policy.sampled_rss_bytes) == (8, 64, 540.0, 600, 1024**3)
    assert policy.output_directory == continuation.DEFAULT_DIRECTORY
    with pytest.raises(FrozenInstanceError):
        setattr(policy, "max_total_trials", 1)


def test_epoch_link_requires_seed_or_immediately_prior_accepted_point() -> None:
    current = {
        "control_sha256": continuation.SEED_CONTROL_SHA256,
        "branch_signature_sha256": continuation.SEED_SIGNATURE_SHA256,
        "objective": continuation.SEED_J, "phi": continuation.SEED_PHI,
        "gradient_inf": continuation.SEED_GRADIENT_INF,
    }
    assert continuation._continuation_link(current, None)
    assert not continuation._continuation_link({**current, "phi": float(current["phi"]) + 1}, None)

    next_point = {**current, "control_sha256": "a" * 64, "objective": 0.8}
    assert continuation._continuation_link(next_point, next_point)
    assert not continuation._continuation_link(current, next_point)


def test_internal_budget_and_total_trial_cap_are_hard_limits() -> None:
    started = 100.0
    assert not continuation._budget_exhausted(started, now=started + 539.999)
    assert continuation._budget_exhausted(started, now=started + 540.0)
    assert continuation._trial_slots(0) == continuation.MAX_TRIALS_PER_EPOCH == 8
    assert continuation._trial_slots(56) == 8
    assert continuation._trial_slots(63) == 1
    assert continuation._trial_slots(64) == 0


def test_analytic_two_epoch_relinearizes_after_branch_switch(tmp_path, monkeypatch) -> None:
    initial = torch.zeros(continuation.EXPECTED_CONTROLS, dtype=torch.float64)
    initial[0], initial[1] = 1.0, 0.2
    parameters = torch.zeros(continuation.EXPECTED_PARAMETERS, dtype=torch.float64)
    truth = torch.zeros(1, dtype=torch.float64)
    threshold = 0.99
    archive_identity = {"fixture": "analytic"}
    input_identity = {
        "parameters_sha256": "p" * 64,
        "terminal_truth_sha256": "t" * 64,
        "archived_input": archive_identity,
    }

    class QuadraticProblem:
        layout = {"controls": continuation.EXPECTED_CONTROLS,
                  "parameters": continuation.EXPECTED_PARAMETERS,
                  "euler_stages": continuation.EXPECTED_STAGES}
        frozen = SimpleNamespace(
            active_field_index=torch.arange(20),
            nowcast_config=SimpleNamespace(forecast_steps=18),
        )
        objective_calls = 0

        def objective(self, control, fixed_parameters):
            self.objective_calls += 1
            x, y = control[0], control[1]
            branch_b = x < threshold
            # The two branch formulas match at x=threshold. Branch B has a
            # different exact Hessian, so the second epoch must relinearize.
            value_a = 0.5 * x.square() + 0.5 * y.square()
            value_b = x.square() - threshold * x + 0.5 * threshold**2 + 0.5 * y.square()
            return torch.where(branch_b, value_b, value_a)

    problem = QuadraticProblem()
    control_hash = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
    policy = replace(
        continuation.DEFAULT_POLICY, max_accepted_epochs=2,
        seed_control_sha256=control_hash, seed_signature_sha256="a" * 64,
        seed_j=0.52, seed_phi=0.52, seed_gradient_inf=1.0,
    )
    monkeypatch.setattr(continuation, "_seed_control", lambda _policy=policy: initial.clone())
    monkeypatch.setattr(continuation, "_source_hashes", lambda: {"fake.py": "s" * 64})
    monkeypatch.setattr(continuation.first_branch, "_sources_match_archive", lambda _: True)
    monkeypatch.setattr(continuation.first_branch, "_archive_input_identity",
                        lambda: archive_identity)
    original_canonical = continuation._canonical_sha
    monkeypatch.setattr(
        continuation, "_canonical_sha",
        lambda value: continuation.seed_linear.ARCHIVED_INPUT_SHA256
        if value is archive_identity else original_canonical(value),
    )
    monkeypatch.setattr(
        continuation, "_sha",
        lambda path: (continuation.PLAN_SHA256 if path == continuation.PLAN
                      else continuation.SEED_REPORT_SHA256 if path == continuation.SEED_REPORT
                      else continuation.SEED_EVIDENCE_SHA256),
    )
    monkeypatch.setattr(
        continuation.seed_linear, "_prepare_fixed_seed",
        lambda: (problem, initial.clone(), initial.clone(), parameters, truth, input_identity),
    )
    monkeypatch.setattr(
        continuation.seed_linear, "_input_identity",
        lambda _problem, _original, current, _parameters, _truth: {
            **input_identity, "control_sha256": continuation.seed_linear._tensor_sha(current),
        },
    )
    monkeypatch.setattr(continuation.seed_linear, "_valid_margins",
                        lambda _value, *, complete: complete)

    def branch_record(current):
        signature = "b" * 64 if float(current[0]) < threshold else "a" * 64
        return {"status": "passed_strict_branch", "euler_stages": continuation.EXPECTED_STAGES,
                "choice_stage_count": continuation.EXPECTED_STAGES,
                "face_sign_stage_count": continuation.EXPECTED_STAGES,
                "signature_sha256": signature}

    current_branch_calls = 0
    strict_endpoint_calls = 0

    def full_current_branch(_problem, current, _parameters):
        nonlocal current_branch_calls
        current_branch_calls += 1
        return branch_record(current), {"complete": True}

    def strict_endpoint(_problem, current, _parameters):
        nonlocal strict_endpoint_calls
        strict_endpoint_calls += 1
        return branch_record(current)

    monkeypatch.setattr(continuation, "_full_current_branch", full_current_branch)
    monkeypatch.setattr(continuation, "_strict_endpoint", strict_endpoint)
    jvp_bases = []
    vjp_bases = []
    real_jvp, real_vjp = torch.func.jvp, torch.func.vjp

    def record_jvp(function, primals, tangents, *args, **kwargs):
        jvp_bases.append(primals[0].clone())
        return real_jvp(function, primals, tangents, *args, **kwargs)

    def record_vjp(function, *primals, **kwargs):
        vjp_bases.append(primals[0].clone())
        return real_vjp(function, *primals, **kwargs)

    monkeypatch.setattr(torch.func, "jvp", record_jvp)
    monkeypatch.setattr(torch.func, "vjp", record_vjp)

    result = continuation.run_probe(tmp_path / "analytic-child.json", policy)

    assert result["execution_status"] == "completed"
    assert len(result["accepted_epochs"]) == 2
    source_identity = {"fake.py": "s" * 64}
    assert continuation._valid_child(result, source_identity, policy)
    first_accept = result["accepted_epochs"][0]["trials"][-1]
    assert first_accept["accepted"] is True
    assert first_accept["branch_changed"] is True
    assert (result["accepted_epochs"][1]["current"]["control_sha256"]
            == first_accept["control_sha256"])
    assert len(jvp_bases) == len(vjp_bases) == 2
    torch.testing.assert_close(jvp_bases[0], initial)
    torch.testing.assert_close(vjp_bases[0], initial)
    assert float(jvp_bases[1][0]) < threshold
    torch.testing.assert_close(jvp_bases[1], vjp_bases[1])
    forged = copy.deepcopy(result)
    forged["accepted_epochs"][1]["current"]["objective"] += 0.1
    assert not continuation._valid_child(forged, source_identity, policy)
    forged_alpha = copy.deepcopy(result)
    forged_alpha["accepted_epochs"][0]["trials"][0]["alpha"] *= 2
    forged_alpha["candidate_attempts"][0]["alpha"] *= 2
    assert not continuation._valid_child(forged_alpha, source_identity, policy)
    forged_handoff = copy.deepcopy(result)
    forged_handoff["numerical_status"] = "handoff_candidate_only"
    forged_handoff["accepted_candidate"]["gradient_inf"] = 0.0
    assert result["accepted_candidate"]["gradient_inf"] >= 1.0e-4
    assert not continuation._valid_child(forged_handoff, source_identity, policy)
    nested_control = copy.deepcopy(result)
    control_values = nested_control["accepted_candidate"]["control"]
    nested_control["accepted_candidate"]["control"] = [
        control_values[:13], control_values[13:],
    ]
    assert not continuation._valid_child(nested_control, source_identity, policy)
    nonfinite_control = copy.deepcopy(result)
    nonfinite_values = list(nonfinite_control["accepted_candidate"]["control"])
    nonfinite_values[0] = float("nan")
    nonfinite_hash = continuation.seed_linear._tensor_sha(
        torch.tensor(nonfinite_values, dtype=torch.float64),
    )
    nonfinite_trial = nonfinite_control["accepted_epochs"][-1]["trials"][-1]
    nonfinite_trial["control_sha256"] = nonfinite_hash
    nonfinite_control["candidate_attempts"][-1]["control_sha256"] = nonfinite_hash
    nonfinite_control["accepted_candidate"]["control"] = nonfinite_values
    nonfinite_control["accepted_candidate"]["control_sha256"] = nonfinite_hash
    assert not continuation._valid_child(nonfinite_control, source_identity, policy)

    tail_cap_policy = replace(
        policy, name="tail", alpha_grid=continuation.TAIL_POLICY.alpha_grid,
        max_total_trials=len(continuation.TAIL_POLICY.alpha_grid),
    )

    def refuse_tail_endpoints(*_args):
        return {"status": "branch_refused", "reason": "analytic refusal"}

    monkeypatch.setattr(
        continuation, "_strict_endpoint", refuse_tail_endpoints,
    )
    capped_tail = continuation.run_probe(tmp_path / "tail-cap-child.json", tail_cap_policy)
    assert capped_tail["numerical_status"] == "trial_limit"
    assert len(capped_tail["candidate_attempts"]) == tail_cap_policy.max_total_trials
    assert capped_tail["accepted_candidate"] is None
    assert len(capped_tail["accepted_epochs"]) == 1
    assert capped_tail["accepted_epochs"][0]["accepted"] is False
    assert continuation._valid_child(capped_tail, source_identity, tail_cap_policy)
    forged_tail = copy.deepcopy(capped_tail)
    forged_tail["numerical_status"] = "epoch_line_search_refused"
    assert not continuation._valid_child(forged_tail, source_identity, tail_cap_policy)
    monkeypatch.setattr(continuation, "_strict_endpoint", strict_endpoint)

    original_budget_check = continuation._budget_exhausted
    for stop_check in (2, 3, 4, 6, 7):
        current_branch_calls = 0
        strict_endpoint_calls = 0
        objective_calls_before = problem.objective_calls
        jvp_bases.clear()
        vjp_bases.clear()
        check_count = 0

        def fake_clock(_started, *, policy=None, stop_at=stop_check):
            nonlocal check_count
            check_count += 1
            return check_count >= stop_at

        monkeypatch.setattr(continuation, "_budget_exhausted", fake_clock)
        stopped = continuation.run_probe(tmp_path / f"fake-clock-{stop_check}.json", policy)
        assert stopped["numerical_status"] == "internal_budget_exhausted"
        assert stopped["accepted_candidate"] is None
        assert stopped["costs"]["current_branch_calls"] == 1
        assert stopped["costs"]["branch_oracle_calls"] == 1 + strict_endpoint_calls
        if stop_check == 2:
            assert problem.objective_calls == objective_calls_before
            assert stopped["costs"]["exact_hvp_calls"] == 0
        elif stop_check == 3:
            assert stopped["costs"]["exact_hvp_calls"] == 0
        elif stop_check == 4:
            assert stopped["costs"]["exact_hvp_calls"] == 1
            assert not stopped["accepted_epochs"]
        elif stop_check == 6:
            assert stopped["candidate_attempts"][0]["objective_status"] == "not_evaluated_internal_budget"
        else:
            assert stopped["candidate_attempts"][0]["objective_status"] == "complete_after_budget"
            assert stopped["candidate_attempts"][0]["accepted"] is False
        assert continuation._valid_child(stopped, source_identity, policy), stop_check

    monkeypatch.setattr(continuation, "_budget_exhausted", original_budget_check)
    capped_policy = replace(policy, max_total_trials=1)
    capped = continuation.run_probe(tmp_path / "trial-cap-after-accept.json", capped_policy)
    assert capped["numerical_status"] == "trial_limit"
    assert len(capped["candidate_attempts"]) == 1
    assert len(capped["accepted_epochs"]) == 1
    assert capped["accepted_epochs"][0]["accepted"] is True
    assert continuation._valid_child(capped, source_identity, capped_policy)

    current_branch_calls = 0
    strict_endpoint_calls = 0

    def refuse_second_current(_problem, current, _parameters):
        nonlocal current_branch_calls
        current_branch_calls += 1
        if current_branch_calls == 2:
            return {"status": "branch_or_margin_refused"}, {"complete": False}
        return branch_record(current), {"complete": True}

    monkeypatch.setattr(continuation, "_full_current_branch", refuse_second_current)
    refusal = continuation.run_probe(
        tmp_path / "current-branch-refusal-after-accept.json", policy,
    )
    assert refusal["numerical_status"] == "current_branch_or_margin_refused"
    assert len(refusal["accepted_epochs"]) == 1
    assert refusal["costs"]["current_branch_calls"] == 2
    assert refusal["costs"]["branch_oracle_calls"] == 3
    assert continuation._valid_child(refusal, source_identity, policy)

    handoff_policy = replace(policy, seed_phi=0.0, seed_gradient_inf=0.0)
    monkeypatch.setattr(
        continuation, "_fresh_merit",
        lambda *_args: (torch.tensor(handoff_policy.seed_j, dtype=torch.float64),
                        torch.zeros_like(initial),
                        torch.tensor(0.0, dtype=torch.float64)),
    )
    current_branch_calls = 0
    handoff = continuation.run_probe(
        tmp_path / "initial-handoff-candidate.json", handoff_policy,
    )
    assert handoff["numerical_status"] == "handoff_candidate_only"
    assert handoff["accepted_epochs"] == []
    assert handoff["costs"]["exact_hvp_calls"] == 0
    assert handoff["costs"]["current_branch_call_unrecorded"] is True
    assert continuation._valid_child(handoff, source_identity, handoff_policy)


def test_candidate_gate_requires_dual_decrease_and_only_same_branch_armijo() -> None:
    changed = merit_step._candidate_gate(
        seed_j=4.0, seed_phi=5.0, candidate_j=3.0, candidate_phi=4.0,
        slope_j=-1.0, slope_phi=-1.0, alpha=0.1, same_signature=False,
    )
    changed_bad = merit_step._candidate_gate(
        seed_j=4.0, seed_phi=5.0, candidate_j=3.0, candidate_phi=5.1,
        slope_j=-1.0, slope_phi=-1.0, alpha=0.1, same_signature=False,
    )
    same = merit_step._candidate_gate(
        seed_j=4.0, seed_phi=5.0, candidate_j=3.9999999, candidate_phi=4.9999999,
        slope_j=-1.0, slope_phi=-1.0, alpha=0.1, same_signature=True,
    )

    assert changed["accepted"] is True
    assert changed_bad["accepted"] is False
    assert same["accepted"] is False
    assert same["armijo_required"] is True


def test_fake_guard_failure_is_reported_without_child_reconstruction(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(continuation.first_branch, "_sources_match_archive", lambda _: True)

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path):
        if "--profile" in command:
            profile_index = command.index("--profile")
            assert command[profile_index + 1] == "tail"
            assert Path(command[command.index("--output") + 1]).name == "point_3h_merit_tail.json"
            assert report_path.name == "point_3h_merit_tail.resource.json"
            assert log_path.name == "point_3h_merit_tail.log"
            assert wall_seconds == continuation.TAIL_POLICY.wall_seconds == 300
            assert rss_bytes == continuation.TAIL_POLICY.sampled_rss_bytes == 1024**3
            assert continuation.TAIL_POLICY.internal_budget_seconds == 240.0
        else:
            assert report_path.name == "point_3h_merit_continuation.resource.json"
            assert wall_seconds == continuation.WALL_SECONDS
            assert rss_bytes == continuation.SAMPLED_RSS_BYTES
        return {
            "command": command, "exit_code": -9,
            "resource_termination": "wall_limit", "monitor_error": None,
            "wall_limit_seconds": wall_seconds, "rss_limit_bytes": rss_bytes,
            "rss_samples": 1, "sampled_peak_rss_bytes": 1234,
            "elapsed_seconds": float(wall_seconds), "child_pid": 123,
        }

    monkeypatch.setattr(continuation, "run_guarded", fake_guard)
    result = continuation.run(tmp_path / "guard-refusal")

    assert result["execution_status"] == "failed"
    assert result["child_report_valid"] is False
    persisted = json.loads((tmp_path / "guard-refusal" /
                            "point_3h_merit_continuation.run.json").read_text())
    assert persisted["resource"]["resource_termination"] == "wall_limit"

    tail_result = continuation.run(
        tmp_path / "tail-guard-refusal", continuation.TAIL_POLICY,
    )
    assert tail_result["execution_status"] == "failed"
    tail_persisted = json.loads((tmp_path / "tail-guard-refusal" /
                                 "point_3h_merit_tail.run.json").read_text())
    assert tail_persisted["resource"]["wall_limit_seconds"] == 300
    assert tail_persisted["resource"]["rss_limit_bytes"] == 1024**3
