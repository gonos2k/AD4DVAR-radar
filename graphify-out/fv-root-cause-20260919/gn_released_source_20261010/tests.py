from __future__ import annotations

from types import SimpleNamespace

import torch

from examples.weather_scenarios import fv_point_3h_gn_released_resume as released


def test_parent_p2_archive_is_the_closed_15516_point():
    parent = released._load_parent()
    control = torch.tensor(parent["current_control"], dtype=torch.float64)
    assert released.tangent._tensor_sha(control) == released.BASE_SHA
    assert parent["current_control_sha256"] == released.BASE_SHA
    assert parent["final_repeat"]["control"] == parent["current_control"]
    assert parent["proposal"]["control"] == parent["current_control"]


def test_trace124_snapshot_records_selector_change_without_freezing_it():
    parent = released._load_parent()
    for side in ("-1", "1"):
        trace = parent["final_repeat"]["branch_trace"][side]
        snapshot = released._event_selector_snapshot(trace)
        assert snapshot["left_sign"] == snapshot["right_sign"] == snapshot["slope_sign"] == -1
        assert snapshot["choose_left"] is True
        assert snapshot["matches_old_negative_right_selector"] is False


def test_gn_model_contract_matches_the_full_current_residual_and_theta():
    residual = torch.linspace(-0.5, 0.5, 27, dtype=torch.float64)
    model = SimpleNamespace(residual=residual.clone(), theta=0.6882137924515473)
    check = released._base_model_closure(model, residual, model.theta)
    assert check["passed"] is True

    changed = residual.clone()
    changed[2] += 1e-7
    check = released._base_model_closure(SimpleNamespace(residual=changed, theta=model.theta),
        residual, model.theta)
    assert check["passed"] is False
    check = released._base_model_closure(SimpleNamespace(residual=residual, theta=model.theta + 1e-4),
        residual, model.theta)
    assert check["passed"] is False
