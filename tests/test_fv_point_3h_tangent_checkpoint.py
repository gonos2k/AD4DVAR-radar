from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def test_interrupted_child_checkpoint_matches_latest_confirmed_commit(tmp_path, monkeypatch):
    base_control = torch.zeros(26, dtype=torch.float64)
    committed_control = base_control.clone()
    committed_control[0] = 0.01
    base_theta = 0.3504554198817298
    committed_theta = 0.4
    runtime = {"python": "mock-runtime"}
    source_hashes = {"source.py": "mock-source-digest"}
    traces = {str(side): {"signature_sha256": f"trace-{side}"} for side in (-1, 1)}
    gradients = {str(side): [0.0] * 26 for side in (-1, 1)}

    def input_identity(_problem, _original, control, _parameters, _truth):
        return {"control_sha256": tangent._tensor_sha(control)}

    base = {
        "control": base_control,
        "theta": base_theta,
        "objective": 1.0,
        "accepted": {
            "F_squared": 0.0,
            "side_gradients": gradients,
            "branch_trace": traces,
        },
        "raw": {
            "input_after": input_identity(None, None, base_control, None, None),
            "runtime_after": runtime,
        },
    }

    def observed(_probe, _problem, _control, _parameters, _weights):
        zero = torch.zeros(26, dtype=torch.float64)
        return {
            "native_j": torch.tensor(1.0, dtype=torch.float64),
            "side": {
                -1: (torch.tensor(1.0, dtype=torch.float64), zero.clone()),
                1: (torch.tensor(1.0, dtype=torch.float64), zero.clone()),
            },
            "traces": {-1: traces["-1"], 1: traces["1"]},
            "pair": {"passed": True},
            "q": torch.tensor(0.0, dtype=torch.float64),
            "production_q": torch.tensor(0.0, dtype=torch.float64),
            "face_bound": 1e-12,
            "face_ok": True,
            "side_objectives_match_native": True,
            "side_gradients_finite": True,
        }

    monkeypatch.setattr(tangent, "_load_plan", lambda *_: {
        "source_files": {}, "archive_files": {},
    })
    monkeypatch.setattr(tangent, "_load_current_base", lambda _plan: base)
    monkeypatch.setattr(tangent.shared, "_source_hashes", lambda *_: source_hashes)
    monkeypatch.setattr(tangent.shared, "_runtime", lambda: runtime)
    monkeypatch.setattr(tangent.shared, "_input_identity", input_identity)
    monkeypatch.setattr(tangent, "_observe", observed)
    monkeypatch.setattr(tangent.seed, "_prepare_fixed_seed", lambda: (
        SimpleNamespace(), base_control.clone(), None, torch.zeros(1), torch.zeros(1), None,
    ))
    monkeypatch.setattr(tangent.geometry, "_face_weights", lambda *_args, **_kwargs:
        torch.tensor([0.84, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    monkeypatch.setattr(tangent.geometry.model, "_fixed_input", lambda *_args: True)

    def stop_after_durable_commit(
        initial_control, theta, _current_products, _side_hvp, _chart_candidate,
        _candidate_eval, final_repeat, *, committed_progress, **_kwargs,
    ):
        proposal = {
            "control": committed_control.tolist(),
            "theta": committed_theta,
            "objective": 1.0,
            "F_squared": 0.0,
            "side_gradients": gradients,
            "branch_trace": traces,
        }
        repeat = final_repeat(proposal)
        assert tangent.fresh_final_closure(proposal, repeat)
        assert torch.equal(initial_control, base_control)
        assert theta == base_theta
        item = {
            "index": 0,
            "base_control_sha256": tangent._tensor_sha(base_control),
            "accepted": True,
            "committed_control": committed_control.tolist(),
            "committed_theta": committed_theta,
            "final_repeat": repeat,
        }
        committed_progress([item])
        # SystemExit models an external termination after the atomic checkpoint
        # and bypasses the child's Exception recovery/final-report write.
        raise SystemExit("simulated guard stop during the next iteration")

    monkeypatch.setattr(tangent, "bounded_continuation", stop_after_durable_commit)
    output = tmp_path / "child.json"

    try:
        tangent._run_child_impl(tmp_path / "plan.json", "mock-plan-digest", output)
    except SystemExit as error:
        assert "simulated guard stop" in str(error)
    else:
        raise AssertionError("mock child did not stop after its checkpoint")

    checkpoint = json.loads(output.read_text())
    expected_digest = tangent._tensor_sha(committed_control)
    assert checkpoint["phase"] == "running"
    assert checkpoint["accepted_iterations"] == 1
    assert checkpoint["optimizer_steps_applied"] == 1
    assert checkpoint["current_control"] == committed_control.tolist()
    assert checkpoint["current_control_sha256"] == expected_digest
    assert checkpoint["current_theta"] == committed_theta
    assert checkpoint["last_confirmed_control"] == committed_control.tolist()
    assert checkpoint["last_confirmed_control_sha256"] == expected_digest
    assert checkpoint["last_confirmed_theta"] == committed_theta
    assert checkpoint["last_confirmed_iterations"] == 1


@pytest.mark.parametrize(
    ("confirmed_theta", "confirmed_count"),
    [(99.0, 1), (0.4, -4), (0.4, True)],
    ids=("theta-out-of-domain", "negative-count", "boolean-count"),
)
def test_malformed_recovery_checkpoint_is_not_promoted(
    tmp_path, monkeypatch, confirmed_theta, confirmed_count,
):
    candidate = torch.linspace(-0.1, 0.2, 26, dtype=torch.float64)
    candidate_sha = tangent._tensor_sha(candidate)

    def fail_after_writing_checkpoint(_plan, _plan_sha, output):
        tangent._write(output, {
            "last_confirmed_control": candidate.tolist(),
            "last_confirmed_control_sha256": candidate_sha,
            "last_confirmed_theta": confirmed_theta,
            "last_confirmed_iterations": confirmed_count,
        })
        raise TimeoutError("synthetic stop with malformed durable state")

    monkeypatch.setattr(tangent, "_run_child_impl", fail_after_writing_checkpoint)
    result = tangent._run_child(tmp_path / "plan.json", "mock-plan-digest",
                                tmp_path / "child.json")

    assert result["execution_status"] == "failed"
    assert result["numerical_status"] == "invalid_checkpoint"
    assert result["refusal"].startswith("invalid last-confirmed checkpoint;")
    assert result["candidate_committed"] is False
    assert result["optimizer_steps_applied"] == 0
    assert result["accepted_iterations"] == 0
    assert result["current_control_sha256"] == tangent.BASE_CONTROL_SHA
    assert "current_control" not in result
    assert "current_theta" not in result
    assert "last_confirmed_control" not in result
    assert "last_confirmed_theta" not in result
    assert "last_confirmed_iterations" not in result
