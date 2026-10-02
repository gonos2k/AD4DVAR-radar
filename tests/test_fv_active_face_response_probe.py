"""Synthetic parameter-direction, endpoint, and interval-orientation checks."""
from typing import Any, cast
from pathlib import Path

import pytest
import torch

from mpmath import iv

from examples.weather_scenarios.fv_active_face_response_probe import (
    PRECISION_PROBE_SHA256,
    SELF_CANONICAL_SHA256,
    _require_canonical_source_sha256,
    _require_source_sha256,
    middle_time_direction,
    normal_orientation,
    parameter_endpoint,
)
from examples.weather_scenarios import fv_active_face_response_probe as probe


def _mask() -> torch.Tensor:
    valid = torch.ones((3, 4, 5), dtype=torch.bool)
    valid[1, 1, 2] = False
    valid[2, 2, 3] = False
    return valid


def test_middle_time_direction_shifts_20_slots_but_only_19_are_active():
    parameters = torch.zeros(61, dtype=torch.float64)
    valid = _mask()
    direction = middle_time_direction(parameters, valid)
    assert direction.shape == (61,)
    assert torch.equal(direction[:20], torch.zeros(20, dtype=torch.float64))
    assert torch.equal(direction[20:40], torch.ones(20, dtype=torch.float64))
    assert torch.equal(direction[40:], torch.zeros(21, dtype=torch.float64))
    assert int(valid[1].sum()) == 19
    assert direction[27] == 1.0  # Missing slot is dropped by the observation operator.


def test_signed_parameter_endpoints_are_symmetric_and_do_not_mutate_inputs():
    parameters = torch.linspace(-0.2, 0.3, 61, dtype=torch.float64)
    direction = torch.zeros_like(parameters)
    direction[20:40] = 1.0
    original = parameters.clone()
    step = 0.00025
    minus = parameter_endpoint(parameters, direction, -1, step)
    plus = parameter_endpoint(parameters, direction, 1, step)
    torch.testing.assert_close((plus + minus) / 2, parameters, rtol=0.0, atol=1e-16)
    torch.testing.assert_close((plus - minus) / (2 * step), direction, rtol=1e-12, atol=1e-12)
    assert torch.equal(parameters, original)


def test_interval_normal_orientation_requires_strict_opposite_signs():
    interval = cast(Any, iv)
    negative = [list(bound) for bound in interval.mpf(["-0.4", "-0.3"])._mpi_]
    positive = [list(bound) for bound in interval.mpf(["0.2", "0.3"])._mpi_]
    oriented, _ = normal_orientation(negative, positive)
    assert oriented
    ambiguous = [list(bound) for bound in interval.mpf(["-0.1", "0.1"])._mpi_]
    oriented, _ = normal_orientation(negative, ambiguous)
    assert not oriented


def test_reviewed_producer_and_fixture_generator_pins_reject_pre_run_edits(tmp_path):
    producer = probe.__file__
    assert producer is not None
    source = Path(producer).read_bytes()
    _require_canonical_source_sha256(source, SELF_CANONICAL_SHA256)
    changed = source.replace(b"middle_time_common_bias_plus1", b"middle_time_common_bias_plus2", 1)
    with pytest.raises(ValueError, match="self-pin changed"):
        _require_canonical_source_sha256(changed, SELF_CANONICAL_SHA256)

    fixture_source = tmp_path / "fixture.py"
    fixture_source.write_bytes(Path("examples/weather_scenarios/fv_slice_precision_probe.py").read_bytes())
    _require_source_sha256(fixture_source, PRECISION_PROBE_SHA256)
    fixture_source.write_bytes(fixture_source.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="reviewed source pin changed"):
        _require_source_sha256(fixture_source, PRECISION_PROBE_SHA256)
