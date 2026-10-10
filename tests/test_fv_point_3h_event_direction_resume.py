from __future__ import annotations

import copy
from typing import Any

import torch

from examples.weather_scenarios import fv_point_3h_event_direction_resume as resume
from examples.weather_scenarios import fv_point_3h_event_direction_comparison as comparison
from types import SimpleNamespace


def test_parent_receipt_is_the_closed_48b0_point_and_event_support():
    parent = resume._load_parent()
    control = torch.tensor(parent["current_control"], dtype=torch.float64)
    assert resume.tangent._tensor_sha(control) == resume.BASE_SHA
    assert parent["selection"]["arm"] == "event_orthogonal"
    assert parent["selected_direction_sha256"] == resume.BASE_EVENT_DIRECTION_SHA

    for side in ("-1", "1"):
        values = torch.tensor(parent["selected_event_values"][side]["values"], dtype=torch.float64)
        trace = parent["final_repeat"]["branch_trace"][side]
        assert resume._event_support_reason(values, trace) is None


def test_changed_event_support_releases_the_event_arm_but_keeps_gn_fallback():
    values = torch.tensor([-0.33, -0.32, 39.0], dtype=torch.float64)
    row = [0, 0, -1, 0]
    choice = {"left_sign": [[0] * 4, row.copy(), [0] * 4],
        "right_sign": [[0] * 4, row.copy(), [0] * 4],
        "slope_sign": [[0] * 4, row.copy(), [0] * 4],
        "choose_left": [[False] * 4, [False] * 4, [False] * 4]}
    trace: dict[str, Any] = {"choices": [[choice, choice] for _ in range(125)]}
    assert resume._event_support_reason(values, trace) is None

    changed: dict[str, Any] = copy.deepcopy(trace)
    changed["choices"][124][0]["choose_left"][1][2] = True
    reason = resume._event_support_reason(values, changed)
    assert reason is not None

    arms = {"prepared_gn": {"supported": True,
                "accepted": {"objective": 1.0, "F_squared": 1.0}},
        "event_orthogonal": {"supported": False, "unsupported_reason": reason, "accepted": None}}
    selected = comparison._select_arm_candidate(arms, torch.float64)
    assert selected is not None and selected["arm"] == "prepared_gn"
    assert selected["comparison_complete"] is False
    assert selected["selection_basis"] == "single_supported_arm_fallback"


def test_each_arm_model_must_reproduce_fresh_full_residual_and_theta():
    residual = torch.linspace(-0.4, 0.5, 27, dtype=torch.float64)
    model = SimpleNamespace(residual=residual.clone(), theta=0.485)
    check = resume._base_model_closure(model, residual, 0.485)
    assert check["passed"] is True

    wrong_residual = residual.clone()
    wrong_residual[4] += 1e-8
    check = resume._base_model_closure(SimpleNamespace(residual=wrong_residual, theta=0.485),
        residual, 0.485)
    assert check["passed"] is False

    check = resume._base_model_closure(SimpleNamespace(residual=residual, theta=0.486), residual, 0.485)
    assert check["passed"] is False
