from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_3h_exploration_correction_cycle as cycle


def _receipt_plan() -> dict[str, Any]:
    pairs = ((cycle.SAMPLE_STEP, "base_step_sha256", cycle.SAMPLE_STEP_SHA),
        (cycle.SAMPLE_RUN, "base_run_sha256", cycle.SAMPLE_RUN_SHA),
        (cycle.SAMPLE_RESOURCE, "base_resource_sha256", cycle.SAMPLE_RESOURCE_SHA),
        (cycle.PR260_STEP, "cycle_reference_step_sha256", cycle.PR260_STEP_SHA),
        (cycle.PR260_RUN, "cycle_reference_run_sha256", cycle.PR260_RUN_SHA),
        (cycle.PR260_RESOURCE, "cycle_reference_resource_sha256", cycle.PR260_RESOURCE_SHA))
    archives = {path.relative_to(cycle.ROOT).as_posix(): digest
                for path, _, digest in pairs}
    return {key: digest for _, key, digest in pairs} | {"archive_files": archives}


def test_receipt_loader_normalizes_accepted_phi_and_historical_reference():
    loaded = cycle._load_base(_receipt_plan())
    assert loaded["accepted_control_sha256"] == cycle.BASE_CONTROL_SHA
    assert loaded["accepted_phi"] == pytest.approx(0.02995458290516701)
    assert loaded["current_state"]["gradient"] == loaded["accepted_gradient"]
    assert loaded["cycle_reference_control_sha256"] == cycle.CYCLE_REFERENCE_SHA
    assert loaded["cycle_reference"]["objective"] == pytest.approx(0.061347638323085395)


def test_receipt_loader_rejects_wrong_receipt_hash():
    plan = _receipt_plan()
    plan["base_step_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt hash"):
        cycle._load_base(plan)


@pytest.mark.parametrize("damage", ["uncommitted", "changed_gradient"])
def test_receipt_loader_rejects_uncommitted_or_inconsistent_trial(monkeypatch, damage):
    original_loads = cycle.json.loads
    sample_text = cycle.SAMPLE_STEP.read_text()

    def altered_loads(value, *args, **kwargs):
        parsed = original_loads(value, *args, **kwargs)
        if value != sample_text:
            return parsed
        if damage == "uncommitted":
            parsed["candidate_committed"] = False
        else:
            parsed["accepted_gradient"][0] += 1e-4
        return parsed

    monkeypatch.setattr(cycle.json, "loads", altered_loads)
    with pytest.raises(ValueError, match="accepted sample endpoint"):
        cycle._load_base(_receipt_plan())


@pytest.mark.parametrize("current,expected", [
    ({"objective": 0.9, "phi": 0.09}, False),
    ({"objective": 0.8, "phi": 0.4}, False),
    ({"objective": 0.8, "phi": 0.08}, True),
])
def test_recovery_requires_resolved_j_and_phi_below_reference(current, expected):
    reference = {"objective": 0.85, "phi": 0.085}
    start = {"objective": 1.0, "phi": 0.5}
    result = cycle.recovery_assessment(reference, start, current)
    assert result["recovered"] is expected
    assert result["status"] == ("recovered" if expected else "unrecovered")
    assert result["metrics"]["objective"]["delta_from_correction_start"] < 0
    assert result["metrics"]["phi"]["delta_from_correction_start"] < 0


def test_recovery_does_not_call_phi_decrease_recovered_when_still_above_reference():
    result = cycle.recovery_assessment(
        {"objective": 0.85, "phi": 0.085},
        {"objective": 1.0, "phi": 0.5},
        {"objective": 0.8, "phi": 0.09})
    assert result["metrics"]["phi"]["correction_decrease_resolved"] is True
    assert result["metrics"]["phi"]["below_preexploration_reference"] is False
    assert result["recovered"] is False
