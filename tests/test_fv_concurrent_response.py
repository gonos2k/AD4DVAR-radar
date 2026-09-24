"""Fast preflight for the bounded concurrent FV response integration probe."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from examples.weather_scenarios import (
    fv_concurrent_response_probe as probe,
    fv_concurrent_response_runner as runner,
)


def test_runner_and_probe_bind_the_same_sources():
    assert runner.SOURCE_PATHS == probe.SOURCE_PATHS


@pytest.mark.parametrize(
    ("case_factory", "archive_names", "direction_slice", "expected_stages"),
    [
        (
            probe._small_case,
            ("minmod_middle_time_bias_final.json", "minmod_parameter_vjp.json"),
            slice(20, 40),
            54,
        ),
        (
            probe._large_case,
            ("fv86_seed_a.json", "fv86_reanalysis.json"),
            slice(80, 160),
            108,
        ),
    ],
)
def test_case_archives_direction_branch_and_nominal_gradient(
    case_factory, archive_names, direction_slice, expected_stages
):
    for name in archive_names:
        assert probe._hash(probe.EVIDENCE / name) == probe.ARCHIVED[name]

    problem, control, parameters, direction, _ = case_factory()

    assert torch.count_nonzero(direction).item() == direction_slice.stop - direction_slice.start
    expected_direction = torch.zeros_like(direction)
    expected_direction[direction_slice] = 1.0
    assert torch.equal(direction, expected_direction)

    branch, _ = problem.branch_check(control, parameters)
    assert branch["euler_stages"] == expected_stages
    assert branch["choices"] == problem.expected_branch["choices"]
    assert branch["face_signs"] == problem.expected_branch["face_signs"]

    # This checks the archived stationary control directly. It does not call
    # local_response or enter the PCG/HVP response path.
    gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    assert bool(torch.isfinite(gradient).all())
    assert gradient.abs().max().item() < 1e-10


@pytest.mark.parametrize(
    ("exit_code", "source_changes", "elapsed", "malformed", "expected_status"),
    [
        (0, False, 1.0, False, "completed"),
        (1, False, 1.0, False, "failed"),
        (0, True, 1.0, False, "failed"),
        (0, False, runner.WALL_SECONDS + 0.01, False, "failed"),
        (0, False, 1.0, True, "failed"),
    ],
)
def test_runner_classifies_exit_and_source_identity_without_child_run(
    tmp_path, monkeypatch, exit_code, source_changes, elapsed, malformed, expected_status
):
    before = {name: f"hash-{name}" for name in runner.SOURCE_PATHS}
    after = dict(before)
    if source_changes:
        after[runner.SOURCE_PATHS[0]] = "changed"
    hashes = iter((before, after))
    monkeypatch.setattr(runner, "_hashes", lambda: next(hashes))

    def fake_run_guarded(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        payload = (
            json.dumps(
                {
                    "status": "completed",
                    "phase": "finished",
                    "source_unchanged": True,
                    "inputs_unchanged": True,
                    "source_sha256": before,
                    "workers": {"fv4x5": {}, "fv8x10": {}},
                }
            )
        )
        output_path.write_text("{" if malformed else payload)
        return {
            "exit_code": exit_code,
            "resource_termination": None,
            "monitor_error": None,
            "elapsed_seconds": elapsed,
        }

    monkeypatch.setattr(runner, "run_guarded", fake_run_guarded)
    result = runner.run(tmp_path)

    assert result["execution_status"] == expected_status
    assert json.loads((tmp_path / "fv_concurrent_response.run.json").read_text())["execution_status"] == expected_status
    assert bool(result["child_read_error"]) is malformed
