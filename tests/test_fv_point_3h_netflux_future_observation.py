from __future__ import annotations

import json
from types import SimpleNamespace
import uuid

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_netflux_future_observation as observer


class ScoreOnlyProblem:
    def __init__(self, truth: torch.Tensor, score_values: tuple[float, float] = (1.25, 0.75)):
        self.verification_dbz = truth.clone()
        self.layout = {"controls": 26, "parameters": 13,
            "observation_shape": (3, 4),
            "observation_times_seconds": (0.0, 600.0, 1200.0),
            "forecast_time_seconds": 12000.0}
        self.frozen = SimpleNamespace(nowcast_config=SimpleNamespace(
            forecast_steps=18, interval_minutes=10))
        self.score_values = score_values
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def score(self, control: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        self.calls.append((control.clone(), parameters.clone()))
        return control.new_tensor(self.score_values[len(self.calls) - 1])


def _inputs() -> tuple[ScoreOnlyProblem, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    truth = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    parameters = torch.linspace(-0.5, 0.5, 13, dtype=torch.float64)
    start = torch.zeros(26, dtype=torch.float64)
    final = torch.zeros(26, dtype=torch.float64)
    start[3] = 0.1
    final[3] = -0.2
    return ScoreOnlyProblem(truth), parameters, truth, start, final


def test_observer_calls_only_two_terminal_scores_with_same_fixed_inputs() -> None:
    problem, parameters, truth, start, final = _inputs()
    before = tuple(x.clone() for x in (parameters, truth, start, final))

    result = observer.observe_score_pair(problem, parameters, truth, start, final)

    assert result["start_E"] == 1.25
    assert result["final_E"] == 0.75
    assert result["delta_E_final_minus_start"] == -0.5
    assert result["score_evaluations"] == 2
    assert result["selection_use"] is False
    assert result["forecast_steps"] == 18
    assert result["forecast_time_seconds_from_start"] == 12000.0
    assert result["forecast_scope"] == "last frame at +180 minutes after the +20 minute analysis window"
    assert len(problem.calls) == 2
    assert torch.equal(problem.calls[0][0], start)
    assert torch.equal(problem.calls[1][0], final)
    assert all(torch.equal(call[1], parameters) for call in problem.calls)
    assert all(torch.equal(a, b) for a, b in zip(before, (parameters, truth, start, final), strict=True))
    assert not ({"selected_control", "candidate", "accepted"} & set(result))


def test_observer_checks_deadline_and_records_order_around_each_score() -> None:
    problem, parameters, truth, start, final = _inputs()
    events: list[str] = []

    result = observer.observe_score_pair(problem, parameters, truth, start, final,
        check=events.append,
        progress=lambda label, status, _score: events.append(f"{label}_{status}"))

    assert events == ["before_start_score", "start_started", "start_completed",
        "after_start_score", "before_final_score", "final_started",
        "final_completed", "after_final_score"]
    assert result["score_evaluations"] == 2


def test_observer_stops_before_second_score_when_deadline_guard_fails() -> None:
    problem, parameters, truth, start, final = _inputs()
    events: list[str] = []

    def check(stage: str) -> None:
        events.append(stage)
        if stage == "before_final_score":
            raise TimeoutError("budget")

    with pytest.raises(TimeoutError, match="budget"):
        observer.observe_score_pair(problem, parameters, truth, start, final,
            check=check,
            progress=lambda label, status, _score: events.append(f"{label}_{status}"))
    assert len(problem.calls) == 1
    assert "final_started" not in events


def test_observer_rejects_profile_mismatch_before_scoring() -> None:
    problem, parameters, truth, start, final = _inputs()
    problem.frozen.nowcast_config.forecast_steps = 17

    with pytest.raises(ValueError, match="terminal score profile"):
        observer.observe_score_pair(problem, parameters, truth, start, final)
    assert problem.calls == []


def test_plan_declares_only_four_new_sources_and_disables_selection() -> None:
    assert observer.NEW_SOURCES == {
        observer.SELF, observer.TEST, observer.FINAL_ENDPOINT_SOURCE,
        "tests/test_fv_point_3h_netflux_continuous_endpoint.py"}
    assert observer.POLICY["score_evaluations"] == 2
    assert observer.POLICY["optimizer_steps"] == 0
    assert observer.POLICY["candidate_evaluations"] == 0
    assert observer.POLICY["selection_use"] is False


def test_load_plan_accepts_exact_physical_source_and_archive_pins(monkeypatch) -> None:
    parent = json.loads(observer.PARENT_PLAN.read_text())
    source_names = set(parent["source_files"]) | observer.NEW_SOURCES
    archive_names = set(parent["archive_files"]) | {
        receipt.relative_to(observer.ROOT).as_posix() for receipt in observer.RECEIPTS}
    plan = {
        "experiment_kind": "paired_new_transport_terminal_E_observation",
        "numerical_version": observer.NUMERICAL_VERSION,
        "transport_sha256": observer.TRANSPORT_SHA,
        "fixed_case_sha256": observer.CASE_SHA,
        "start_control_sha256": observer.START_CONTROL_SHA,
        "final_control_sha256": observer.FINAL_CONTROL_SHA,
        "parameters_sha256": observer.PARAMETERS_SHA,
        "truth_sha256": observer.TRUTH_SHA,
        "problem_identity_sha256": observer.PROBLEM_SHA,
        "score_profile": observer.SCORE_PROFILE,
        "policy": observer.POLICY,
        "source_files": {name: observer._sha_file(observer.ROOT / name)
            for name in sorted(source_names)},
        "archive_files": {name: observer._sha_file(observer.ROOT / name)
            for name in sorted(archive_names)},
    }
    path = observer.WORK / f"observer-plan-test-{uuid.uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, sort_keys=True))
    monkeypatch.setattr(observer, "PLAN", path)
    try:
        loaded = observer.load_plan(path, observer._sha_file(path))
        assert set(loaded["source_files"]) == source_names
        assert loaded["source_files"][observer.FINAL_ENDPOINT_SOURCE] == observer._sha_file(
            observer.ROOT / observer.FINAL_ENDPOINT_SOURCE)
    finally:
        path.unlink(missing_ok=True)


def test_pinned_case_metadata_admits_actual_problem_without_forward_evaluation() -> None:
    case = observer._load_case()
    problem, original, parameters, truth = observer._fixed_case_tensors(case, {
        "parameters_sha256": observer.PARAMETERS_SHA,
        "truth_sha256": observer.TRUTH_SHA,
    })

    assert problem.identity["fixed_problem_sha256"] == observer.PROBLEM_SHA
    assert original.shape == (26,)
    assert parameters.shape == (13,)
    assert truth.shape == (4, 5)
    assert observer._score_profile(problem) == observer.SCORE_PROFILE
    assert getattr(problem, "score_calls", 0) == 0


def _valid_resource(exit_code: int = 0) -> dict[str, object]:
    return {"exit_code": exit_code, "elapsed_seconds": 2.0,
        "wall_limit_seconds": observer.POLICY["outer_seconds"],
        "rss_limit_bytes": observer.POLICY["rss_bytes"],
        "sampled_peak_rss_bytes": 1024, "resource_termination": None,
        "monitor_error": None, "received_sigterm": False}


def _valid_child(digest: str) -> dict[str, object]:
    inputs = {"start": {"control": "a"}, "final": {"control": "b"}}
    sources = {"source.py": "pinned"}
    return {"phase": "finished", "execution_status": "completed",
        "numerical_status": "paired_terminal_score_observed", "plan_sha256": digest,
        "numerical_version": observer.NUMERICAL_VERSION,
        "transport_sha256": observer.TRANSPORT_SHA,
        "score_evaluations_started": 2, "score_evaluations_completed": 2,
        "score_pair": {"start_E": 1.0, "final_E": 0.75,
            "score_evaluations": 2, "selection_use": False},
        "optimizer_steps": 0, "candidate_evaluations": 0, "selection_use": False,
        "root_claim": False, "minimum_claim": False,
        "response_claim": False, "score_claim": False,
        "source_unchanged": True, "source_before": sources, "source_after": sources,
        "input_unchanged": True, "input_before": inputs, "input_after": inputs,
        "runtime_unchanged": True, "runtime_before": {"torch": "fixed"},
        "runtime_after": {"torch": "fixed"}, "fixed_case_unchanged": True,
        "fixed_case_sha256": observer.CASE_SHA}


def test_parent_run_requires_guard_and_closed_child_receipt(tmp_path) -> None:
    digest = "plan-digest"
    child = tmp_path / "score_pair.json"
    child.write_text(json.dumps(_valid_child(digest)))

    result = observer._run_summary(_valid_resource(), child, digest)

    assert result["execution_status"] == "completed"
    assert result["child_receipt_status"] == "closed"
    assert result["child_sha256"] == observer._sha_bytes(child.read_bytes())


def test_parent_run_rejects_guard_failure_even_with_closed_child(tmp_path) -> None:
    child = tmp_path / "score_pair.json"
    child.write_text(json.dumps(_valid_child("plan-digest")))

    result = observer._run_summary(_valid_resource(exit_code=1), child, "plan-digest")

    assert result["execution_status"] == "failed"
    assert result["child_receipt_status"] == "invalid"


def test_parent_run_rejects_missing_child_receipt(tmp_path) -> None:
    result = observer._run_summary(_valid_resource(), tmp_path / "missing.json", "plan-digest")

    assert result["execution_status"] == "failed"
    assert result["child_receipt_status"] == "missing"


def test_parent_run_rejects_child_plan_hash_mismatch(tmp_path) -> None:
    child = tmp_path / "score_pair.json"
    child.write_text(json.dumps(_valid_child("different-plan")))

    result = observer._run_summary(_valid_resource(), child, "plan-digest")

    assert result["execution_status"] == "failed"
    assert result["child_receipt_status"] == "invalid"


def test_observer_rejects_a_truth_that_is_not_the_problem_verification() -> None:
    problem, parameters, truth, start, final = _inputs()
    wrong_truth = truth.clone()
    wrong_truth[0, 0] += 1.0

    with pytest.raises(ValueError, match="verification field"):
        observer.observe_score_pair(problem, parameters, wrong_truth, start, final)
    assert problem.calls == []


@pytest.mark.parametrize("values", [(-1.0, 1.0), (float("nan"), 1.0)])
def test_observer_rejects_nonfinite_or_negative_terminal_scores(values) -> None:
    truth = torch.zeros((4, 5), dtype=torch.float64)
    problem = ScoreOnlyProblem(truth, values)
    parameters = torch.zeros(13, dtype=torch.float64)
    start = torch.zeros(26, dtype=torch.float64)
    final = torch.ones(26, dtype=torch.float64)

    with pytest.raises(ValueError, match="score"):
        observer.observe_score_pair(problem, parameters, truth, start, final)


def test_endpoint_control_accepts_only_finite_cpu_fp64_vector() -> None:
    expected = torch.zeros(26, dtype=torch.float64)
    loaded = observer._endpoint_control({"control": expected.tolist()})
    assert torch.equal(loaded, expected)
    with pytest.raises(ValueError, match="length 26"):
        observer._endpoint_control({"control": [0.0] * 25})
