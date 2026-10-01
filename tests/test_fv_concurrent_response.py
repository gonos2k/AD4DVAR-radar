"""Fast preflight for the bounded concurrent FV response integration probe."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from examples.weather_scenarios import (
    fv_concurrent_response_probe as probe,
    fv_concurrent_response_runner as runner,
)
_SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "fv_response_test_support", Path(__file__).with_name("fv_response_test_support.py"),
)
assert _SUPPORT_SPEC is not None and _SUPPORT_SPEC.loader is not None
_SUPPORT = importlib.util.module_from_spec(_SUPPORT_SPEC)
_SUPPORT_SPEC.loader.exec_module(_SUPPORT)
portable_case = getattr(_SUPPORT, "portable_case")


def test_runner_and_probe_bind_the_same_sources():
    assert runner.SOURCE_PATHS == probe.SOURCE_PATHS


@pytest.mark.parametrize(
    ("case_id", "archive_names", "direction_slice", "expected_stages"),
    [
        ("fv4x5", ("minmod_middle_time_bias_final.json", "minmod_parameter_vjp.json"),
         slice(20, 40), 54),
        ("fv8x10", ("fv86_seed_a.json", "fv86_reanalysis.json"),
         slice(80, 160), 108),
    ],
)
def test_portable_case_uses_frozen_control_data_with_current_inputs(
    case_id, archive_names, direction_slice, expected_stages,
):
    for name in archive_names:
        assert probe._hash(probe.EVIDENCE / name) == probe.ARCHIVED[name]
    problem, control, parameters, direction, stages = portable_case(case_id)

    assert control.ndim == parameters.ndim == direction.ndim == 1
    assert control.dtype == parameters.dtype == direction.dtype == torch.float64
    assert torch.isfinite(control).all() and torch.isfinite(parameters).all()
    assert torch.count_nonzero(direction).item() == direction_slice.stop - direction_slice.start
    expected_direction = torch.zeros_like(direction)
    expected_direction[direction_slice] = 1.0
    assert torch.equal(direction, expected_direction)

    branch, _ = problem.branch_check(control, parameters)
    assert branch["euler_stages"] == stages == expected_stages

    # The frozen control is only a seed here; this gradient uses fresh current
    # inputs and does not enter the PCG/HVP response path or certify the archive.
    gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize(
    ("case_id", "shape", "message"),
    [
        ("fv4x5", (26,), "4x5 archived stationary input identity mismatch"),
        ("fv4x5", (61,), "4x5 archived stationary input identity mismatch"),
        ("fv4x5", (4, 5), "4x5 verification differs from archived score"),
        ("fv8x10", (241,), "8x10 archived input/verification identity mismatch"),
        ("fv8x10", (8, 10), "8x10 archived input/verification identity mismatch"),
    ],
)
def test_production_case_constructors_reject_archived_identity_drift(
    monkeypatch, case_id, shape, message,
):
    original_hash = probe._bytes_hash

    def drifted_hash(value):
        if tuple(value.shape) == shape:
            return "0" * 64
        return original_hash(value)

    monkeypatch.setattr(probe, "_bytes_hash", drifted_hash)
    factory = probe._small_case if case_id == "fv4x5" else probe._large_case
    with pytest.raises(ValueError, match=message):
        factory()


def test_production_small_case_rejects_regenerated_input_drift(monkeypatch):
    make_case = probe.small_fixture.make_spatial_case

    def drifted_case(*, smoke_only=False):
        observations, frozen, boundary, support = make_case(smoke_only=smoke_only)
        changed_dbz = observations.dbz.clone()
        changed_dbz[0, 0, 0] = torch.nextafter(
            changed_dbz[0, 0, 0], torch.full_like(changed_dbz[0, 0, 0], torch.inf)
        )
        return probe.replace(observations, dbz=changed_dbz), frozen, boundary, support

    monkeypatch.setattr(probe.small_fixture, "make_spatial_case", drifted_case)
    with pytest.raises(ValueError, match="4x5 archived stationary input identity mismatch"):
        probe._small_case()


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
