from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def test_write_stream_matches_existing_json_bytes(tmp_path):
    path = tmp_path / "checkpoint.json"
    value = {
        "unicode": "기상 ☁️ café",
        "negative_zero": -0.0,
        "nested": {"items": [1, {"enabled": True, "value": None}]},
    }

    tangent._write(path, value)

    expected = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    assert b"\\u" in path.read_bytes()
    assert b"-0.0" in path.read_bytes()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_write_rejects_nonfinite_values_and_preserves_previous_file(tmp_path, value):
    path = tmp_path / "checkpoint.json"
    previous = b"previous checkpoint\n"
    path.write_bytes(previous)

    with pytest.raises(ValueError):
        tangent._write(path, {"nested": {"value": value}})

    assert path.read_bytes() == previous
    assert not path.with_name(path.name + f".{tangent.os.getpid()}.tmp").exists()


def test_write_partial_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    previous = b"previous checkpoint\n"
    path.write_bytes(previous)
    temp = path.with_name(path.name + f".{tangent.os.getpid()}.tmp")
    original_open = Path.open

    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream
            self.writes = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.stream.close()

        def write(self, chunk):
            self.writes += 1
            if self.writes == 2:
                raise OSError("synthetic partial write failure")
            return self.stream.write(chunk)

    def fail_after_partial_temp_write(self, *args, **kwargs):
        stream = original_open(self, *args, **kwargs)
        return PartialWriter(stream) if self == temp else stream

    monkeypatch.setattr(Path, "open", fail_after_partial_temp_write)
    with pytest.raises(OSError, match="partial write"):
        tangent._write(path, {"nested": {"value": 1}})

    assert path.read_bytes() == previous
    assert not temp.exists()


def test_write_replace_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    previous = b"previous checkpoint\n"
    path.write_bytes(previous)
    temp = path.with_name(path.name + f".{tangent.os.getpid()}.tmp")

    def fail_replace(_source, _target):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(tangent.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        tangent._write(path, {"new": True})

    assert path.read_bytes() == previous
    assert not temp.exists()


def test_committed_progress_drops_comparison_checkpoint_and_keeps_closed_history(
    tmp_path, monkeypatch,
):
    base_control = torch.zeros(26, dtype=torch.float64)
    committed_control = base_control.clone()
    committed_control[0] = 0.01
    base_theta, committed_theta = 0.35, 0.4
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
        "accepted": {"F_squared": 0.0, "side_gradients": gradients, "branch_trace": traces},
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
        "source_files": {}, "archive_files": {}, "policy": tangent.POLICY,
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

    earlier_failure = {"index": 0, "accepted": False, "refusal": "earlier closure failed"}
    original_write = tangent._write
    progress_records = []
    failed_commit_snapshots = []
    fail_commit_once = True

    def fail_first_commit_write(path, value):
        nonlocal fail_commit_once
        if "comparison_progress" in value:
            progress_records.append(value)
        if value.get("accepted_iterations") == 1 and fail_commit_once:
            fail_commit_once = False
            failed_commit_snapshots.append(value)
            raise OSError("synthetic committed checkpoint write failure")
        original_write(path, value)

    monkeypatch.setattr(tangent, "_write", fail_first_commit_write)

    def stop_after_commit(
        initial_control, theta, _current_products, _side_hvp, _chart_candidate,
        _candidate_eval, final_repeat, *, direction_progress, committed_progress, **_kwargs,
    ):
        direction_progress({"index": 1, "model_comparisons": [{"status": "partial"}]})
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
        accepted = {
            "index": 1,
            "base_control_sha256": tangent._tensor_sha(base_control),
            "accepted": True,
            "committed_control": committed_control.tolist(),
            "committed_theta": committed_theta,
            "final_repeat": repeat,
        }
        previous_checkpoint = output.read_bytes()
        with pytest.raises(OSError, match="checkpoint write failure"):
            committed_progress([earlier_failure, accepted])
        assert output.read_bytes() == previous_checkpoint
        assert "comparison_progress" in progress_records[-1]
        assert "comparison_progress" not in failed_commit_snapshots[0]
        committed_progress([earlier_failure, accepted])
        raise SystemExit("simulated stop after durable commit")

    monkeypatch.setattr(tangent, "bounded_continuation", stop_after_commit)
    def unused_models(_context: dict[str, Any]) -> Callable[
            [torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
            list[dict[str, Any]]]:
        return lambda *_args: []
    output = tmp_path / "child.json"
    with pytest.raises(SystemExit, match="durable commit"):
        tangent._run_child_impl(tmp_path / "plan.json", "mock-plan-digest", output,
            direction_model_factory=unused_models)

    checkpoint = json.loads(output.read_text())
    expected_digest = tangent._tensor_sha(committed_control)
    assert "comparison_progress" not in checkpoint
    assert len(checkpoint["iterations"]) == 2
    assert checkpoint["iterations"][0] == earlier_failure
    assert checkpoint["iterations"][1]["accepted"] is True
    assert checkpoint["iterations"][1]["committed_theta"] == committed_theta
    assert checkpoint["accepted_iterations"] == 1
    assert checkpoint["optimizer_steps_applied"] == 1
    assert checkpoint["candidate_committed"] is True
    assert checkpoint["current_control"] == committed_control.tolist()
    assert checkpoint["current_control_sha256"] == expected_digest
    assert checkpoint["current_theta"] == committed_theta
    assert checkpoint["last_confirmed_control"] == committed_control.tolist()
    assert checkpoint["last_confirmed_control_sha256"] == expected_digest
    assert checkpoint["last_confirmed_theta"] == committed_theta
    assert checkpoint["last_confirmed_iterations"] == 1
    assert checkpoint["last_confirmed_closure"] == checkpoint["iterations"][1]["final_repeat"]
