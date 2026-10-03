"""Synthetic branch-trace comparison and source-pin tests; no FV execution."""
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_event_diagnostic as event


def _trace() -> list[dict[str, Any]]:
    stages = []
    for index in range(54):
        cell = {"choice": 1, "slope_sign": 1, "left_sign": 1, "right_sign": 1}
        stages.append({"step": index // 2, "stage": index % 2,
                       "limiters": {"x": [cell.copy() for _ in range(6)],
                                    "y": [cell.copy() for _ in range(6)]},
                       "face_signs": {"qx": [1] * 24, "qy": [1] * 25}})
    return stages


def test_unchanged_trace_and_repeated_named_upwind_flip():
    before = _trace()
    assert event.compare_traces(before, _trace()) == {
        "face_sign_differences": [], "limiter_differences": [],
        "single_qx_3_4_event_candidate": False,
    }
    single = _trace()
    single[17]["face_signs"]["qx"][22] = -1
    assert event.compare_traces(before, single)["single_qx_3_4_event_candidate"] is False

    after = _trace()
    for row in after:
        row["face_signs"]["qx"][22] = -1
    result = event.compare_traces(before, after)
    assert result["single_qx_3_4_event_candidate"] is True
    assert len(result["face_sign_differences"]) == 54
    assert {(row["axis"], row["row"], row["column"], row["before"], row["after"])
            for row in result["face_sign_differences"]} == {("qx", 3, 4, 1, -1)}
    assert result["limiter_differences"] == []


def test_limiter_selector_or_input_sign_change_is_reported_and_not_single_face_event():
    before = _trace()
    after = _trace()
    after[3]["limiters"]["y"][4]["right_sign"] = -1
    result = event.compare_traces(before, after)
    assert result["face_sign_differences"] == []
    assert result["single_qx_3_4_event_candidate"] is False
    assert result["limiter_differences"] == [{
        "step": 1, "stage": 1, "axis": "y", "row": 2, "column": 2,
        "before": {"choice": 1, "slope_sign": 1, "left_sign": 1, "right_sign": 1},
        "after": {"choice": 1, "slope_sign": 1, "left_sign": 1, "right_sign": -1},
    }]


def test_malformed_stage_count_and_source_pin_mutation_refuse(tmp_path):
    with pytest.raises(ValueError, match="54 Euler stages"):
        event.validate_trace(_trace()[:-1])

    producer = Path(event.__file__).read_bytes()
    event.require_self_source_pin(producer, event.SELF_CANONICAL_SHA256)
    with pytest.raises(ValueError, match="source pin changed"):
        event.require_self_source_pin(producer.replace(b"root_absence_claim", b"root_claim", 1),
                                      event.SELF_CANONICAL_SHA256)
    assert event.changed_source_paths({"a.py": "a", "b.py": "b"},
                                      {"a.py": "a", "b.py": "changed"}) == ["b.py"]


def test_stage_snapshot_indexes_nearest_qx_face_without_running_model():
    q = torch.arange(20, dtype=torch.float64).reshape(4, 5) + 1
    qx = torch.ones((4, 6), dtype=torch.float64)
    qx[3, 4] = -1e-8
    qy = torch.ones((5, 5), dtype=torch.float64)
    stage = event._stage_snapshot(q, qx, qy, 10)
    assert stage["step"] == 5 and stage["stage"] == 0
    assert stage["nearest_face"]["axis"] == "qx"
    assert (stage["nearest_face"]["row"], stage["nearest_face"]["column"]) == (3, 4)
    assert stage["nearest_face"]["signed_flux"] == -1e-8


def test_snapshot_matches_original_core_oracle_list_of_two_axis_maps():
    trace = _trace()
    branch = {"choices": [], "face_signs": []}
    for row in trace:
        choices = []
        for axis in ("x", "y"):
            cells = row["limiters"][axis]
            def grid(field):
                return [[cell[field] for cell in cells[start:start + 3]] for start in (0, 3)]
            choices.append({"choose_left": [[cell["choice"] == 1 for cell in cells[start:start + 3]]
                                             for start in (0, 3)],
                            "slope_sign": grid("slope_sign"),
                            "left_sign": grid("left_sign"),
                            "right_sign": grid("right_sign")})
        branch["choices"].append(choices)
        branch["face_signs"].append({"qx": [row["face_signs"]["qx"][start:start + 6]
                                             for start in (0, 6, 12, 18)],
                                     "qy": [row["face_signs"]["qy"][start:start + 5]
                                             for start in (0, 5, 10, 15, 20)]})
    assert event.core_trace_matches(branch, trace)
    branch["choices"][0][0]["left_sign"][0][0] = -1
    assert not event.core_trace_matches(branch, trace)
