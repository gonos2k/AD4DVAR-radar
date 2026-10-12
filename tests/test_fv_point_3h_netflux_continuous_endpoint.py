from __future__ import annotations

import copy
import json

import pytest

from examples.weather_scenarios import fv_point_3h_netflux_continuation as prior
from examples.weather_scenarios import fv_point_3h_netflux_continuous_endpoint as endpoint


def _records():
    plan = endpoint.continuation.load_plan(endpoint.PLAN, endpoint.PLAN_SHA)
    manifest = endpoint._json(endpoint._read(endpoint.RECEIPT_MANIFEST,
        endpoint.RECEIPT_MANIFEST_SHA), "receipt manifest")
    members = endpoint._tar_members(endpoint._read(endpoint.RECEIPTS_TAR,
        endpoint.RECEIPTS_TAR_SHA), set(manifest["files"]), "continuous receipts")
    raw = json.loads(members["attempt1/step.json"])
    run = json.loads(members["attempt1/step.run.json"])
    resource = json.loads(members["attempt1/step.resource.json"])
    start = endpoint.continuation.endpoint.load_endpoint()
    return plan, raw, run, resource, members, start


def test_endpoint_is_receipt_only_and_returns_last_confirmed_prefix(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("continuous endpoint must not regenerate data or evaluate FV")

    monkeypatch.setattr(prior.previous.old.previous.seed, "_prepare_fixed_seed", forbidden)
    monkeypatch.setattr(prior.tangent, "_observe", forbidden)
    loaded = endpoint.load_endpoint()
    assert endpoint.tangent._tensor_sha(loaded["control"]) == endpoint.BASE_SHA
    assert loaded["raw"]["numerical_status"] == "budget_exhausted"
    assert loaded["raw"]["deadline_passed"] is False
    assert len(loaded["raw"]["closed_steps"]) == 11
    assert loaded["raw"]["optimizer_steps_applied"] == 11
    assert loaded["raw"]["last_confirmed_control"] == loaded["raw"]["current_control"]
    assert loaded["repeat"]["control"] == loaded["raw"]["current_control"]
    assert loaded["raw"]["current_point"]["hvp_completed"] == 1
    assert "side_HVP" not in loaded["raw"]["current_point"]
    assert loaded["raw"]["current_point"]["base_control_sha256"] == endpoint.BASE_SHA
    assert len(loaded["receipt_files"]) == len(endpoint.RECEIPTS)


def test_internal_budget_stop_is_admitted_only_with_all_11_closed_prefixes():
    plan, raw, run, resource, members, start = _records()
    damaged = copy.deepcopy(raw)
    damaged["closed_steps"].pop()
    with pytest.raises(ValueError, match="11-step closed prefix"):
        endpoint._validate_child(plan, damaged, run, resource, members, start)


def test_endpoint_rejects_a_damaged_committed_step_p2():
    plan, raw, run, resource, members, start = _records()
    damaged = copy.deepcopy(raw)
    damaged["closed_steps"][4]["minimum_mixture_P2_match"]["passed"] = False
    with pytest.raises(ValueError, match="P2/HVP/model closure"):
        endpoint._validate_child(plan, damaged, run, resource, members, start)


def test_endpoint_rejects_treating_the_partial_hvp_as_complete():
    plan, raw, run, resource, members, start = _records()
    damaged = copy.deepcopy(raw)
    damaged["current_point"]["hvp_completed"] = 2
    with pytest.raises(ValueError, match="partial point"):
        endpoint._validate_child(plan, damaged, run, resource, members, start)


def test_endpoint_rejects_any_committed_step_armijo_or_repeat_mismatch():
    plan, raw, run, resource, members, start = _records()
    damaged = copy.deepcopy(raw)
    damaged["closed_steps"][2]["proposal"]["J_armijo_passed"] = False
    with pytest.raises(ValueError, match="original-J/P2/previous-P2"):
        endpoint._validate_child(plan, damaged, run, resource, members, start)
