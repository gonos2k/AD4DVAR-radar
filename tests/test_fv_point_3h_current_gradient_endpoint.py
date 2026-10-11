from __future__ import annotations

import copy

import pytest

from examples.weather_scenarios import fv_point_3h_current_gradient_endpoint as endpoint


def test_endpoint_admission_is_receipt_only(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("endpoint admission must not prepare or evaluate FV")

    monkeypatch.setattr(endpoint.tangent, "_observe", forbidden)
    loaded = endpoint.load_endpoint()
    assert endpoint.tangent._tensor_sha(loaded["control"]) == endpoint.BASE_SHA
    assert loaded["raw"]["candidate_committed"] is True
    assert len(loaded["raw"]["closed_steps"]) == 3
    assert loaded["raw"]["current_control"] == loaded["raw"]["last_confirmed_control"]
    assert loaded["raw"]["current_full_G"] == loaded["raw"]["last_confirmed_full_G"]
    assert loaded["parent_plan"]["base_control_sha256"] != endpoint.BASE_SHA
    assert set(loaded["receipt_files"]) == {
        path.relative_to(endpoint.ROOT).as_posix() for path in endpoint.RECEIPTS}


@pytest.mark.parametrize("damage", ["p2", "proposal", "full_g"])
def test_each_persisted_commit_rejects_corrupted_closure(damage):
    loaded = endpoint.load_endpoint()
    for saved_step in loaded["raw"]["closed_steps"]:
        step = copy.deepcopy(saved_step)
        if damage == "p2":
            step["previous_P2_match"]["passed"] = False
        elif damage == "proposal":
            step["proposal"]["J_armijo_passed"] = False
        else:
            step["full_G_check"]["G"][0] += 1e-4
        with pytest.raises(ValueError):
            endpoint._close_step(step, None, step["base_control_sha256"])

