from __future__ import annotations

import json
import math
import sys

import pytest

from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit


def test_component_scaled_metric_uses_declared_fp64_budget() -> None:
    passed = audit._metric(3.0, 3.0 + 1.0e-14)
    failed = audit._metric(3.0, 3.0 + 1.0e-10)
    assert passed["passed"] is True
    assert passed["budget"] == pytest.approx(128 * 2.220446049250313e-16 * 3.0)
    assert failed["passed"] is False


def test_metric_rejects_nonfinite_difference() -> None:
    assert audit._metric(1.0, math.inf)["passed"] is False


def test_metric_near_zero_uses_component_scale_without_unit_floor() -> None:
    exact_zero = audit._metric(0.0, 0.0)
    tiny_difference = audit._metric(0.0, 1.0e-300)
    small_relative_difference = audit._metric(1.0e-300, 1.1e-300)
    assert exact_zero["passed"] is True
    assert tiny_difference["passed"] is False
    assert small_relative_difference["passed"] is False
    assert exact_zero["budget"] == 128 * math.ulp(1.0) * sys.float_info.min


def test_preflight_source_superset_accepts_extra_pinned_paths() -> None:
    child = {"a.py": "a"}
    parent = {"a.py": "a", "extra.py": "b"}
    assert audit._source_subset_matches(child, parent)
    assert not audit._source_subset_matches({"a.py": "wrong"}, parent)


def test_archived_preflight_is_a_pinned_superset_of_child_sources() -> None:
    raw = json.loads(audit.RAW.read_text())
    preflight = json.loads(audit.PREFLIGHT.read_text())
    child, parent = raw["source_before"], preflight["source_sha256"]
    assert child == raw["source_after"]
    assert child != parent
    assert audit._source_subset_matches(child, parent)
    assert not audit._source_subset_matches({**child, next(iter(child)): "bad-hash"}, parent)


def test_endpoint_identity_allows_control_change_but_pins_fixed_inputs() -> None:
    base = {"parameters_sha256": "p", "terminal_truth_sha256": "t",
            "archived_input": {"problem": "fixed"}, "control_sha256": "old"}
    candidate = {**base, "control_sha256": "accepted"}
    assert audit._fixed_input_matches(candidate, base, "accepted")
    assert not audit._fixed_input_matches({**candidate, "parameters_sha256": "other"}, base, "accepted")
    assert not audit._fixed_input_matches({**candidate, "archived_input": {"problem": "other"}}, base, "accepted")
    assert not audit._fixed_input_matches({**candidate, "terminal_truth_sha256": "other"}, base, "accepted")


def test_archived_endpoint_identity_pins_fixed_problem_and_new_control() -> None:
    raw = json.loads(audit.RAW.read_text())
    base = raw["input_after"]["identity"]
    candidate = {**base, "control_sha256": audit.ACCEPTED_CONTROL_SHA}
    assert audit._fixed_input_matches(candidate, base, audit.ACCEPTED_CONTROL_SHA)
    assert not audit._fixed_input_matches({**candidate, "parameters_sha256": "other"}, base,
                                          audit.ACCEPTED_CONTROL_SHA)


def test_plan_and_frozen_source_names_are_distinct_from_output() -> None:
    assert audit.SELF.endswith("fv_point_3h_accepted_endpoint_audit.py")
    assert audit.TEST == "tests/test_fv_point_3h_accepted_endpoint_audit.py"
    assert audit.DEFAULT_OUT.name == "audit.json"
    assert audit.INTERNAL_SECONDS < 600
