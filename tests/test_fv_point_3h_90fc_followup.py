from __future__ import annotations

from copy import deepcopy
from typing import Any
import json
import hashlib
import tempfile
from pathlib import Path

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_90fc_followup as followup
from examples.weather_scenarios import fv_point_3h_dual_merit_continuation as continuation
from examples.weather_scenarios import fv_point_3h_accepted_curvature as curvature


def _receipt(monkeypatch) -> dict[str, Any]:
    control = [0.0] * 26
    control_sha = curvature.tensor_sha(torch.tensor(control, dtype=torch.float64))
    monkeypatch.setattr(followup, "CONTROL_SHA", control_sha)
    branch = {"signature_sha256": "branch-final"}
    margins = {"complete": True}
    gradient = [0.0] * 26
    trial = {"status": "accepted", "control": control,
        "control_sha256": control_sha, "objective": 0.063,
        "phi": 0.4, "gradient": gradient, "gradient_inf": 0.0,
        "branch": branch, "margins": margins, "J_armijo_passed": True,
        "Phi_armijo_passed": True, "strict_point_passed": True}
    iteration = {"status": "accepted", "accepted_control": control,
        "accepted_control_sha256": control_sha}
    return {"phase": "finished", "execution_status": "completed",
        "numerical_status": "budget_refusal", "optimizer_steps_applied": 2,
        "accepted_control": control, "accepted_control_sha256": control_sha,
        "trials": [{"status": "rejected"},
            {**deepcopy(trial), "control_sha256": "first-accepted"}, deepcopy(trial)],
        "iterations": [deepcopy(iteration), deepcopy(iteration)],
        "current_state": {"objective": 0.063, "phi": 0.4, "gradient": gradient,
            "gradient_inf": 0.0, "branch": branch, "margins": margins}}


def test_final_endpoint_must_match_last_accepted_row_and_current_state(monkeypatch):
    receipt = _receipt(monkeypatch)

    assert followup._accepted_endpoint(receipt) == receipt["trials"][-1]

    mismatched = deepcopy(receipt)
    mismatched["current_state"]["phi"] = 0.5
    with pytest.raises(ValueError, match="final accepted trial"):
        followup._accepted_endpoint(mismatched)


def test_rejects_unaccepted_or_wrong_endpoint_history(monkeypatch):
    receipt = _receipt(monkeypatch)
    receipt["trials"][-1]["control_sha256"] = "wrong"
    with pytest.raises(ValueError, match="final accepted trial"):
        followup._accepted_endpoint(receipt)

    receipt = _receipt(monkeypatch)
    receipt["optimizer_steps_applied"] = 1
    with pytest.raises(ValueError, match="exactly two"):
        followup._accepted_endpoint(receipt)


def test_adapter_delegates_to_existing_loop_with_pinned_base(monkeypatch, tmp_path):
    calls = []

    def delegated(*args, **kwargs):
        calls.append((args, kwargs))
        return {"delegated": True}

    monkeypatch.setattr(continuation, "run", delegated)
    plan = tmp_path / "plan.json"
    output = tmp_path / "step.json"

    assert followup.run(plan, "plan-sha", output) == {"delegated": True}
    assert calls == [((plan, "plan-sha", output),
                      {"base_loader": followup.load_base, "base_path": followup.BASE})]


def test_real_archived_endpoint_uses_final_control_and_gradient_values():
    raw = json.loads(followup.BASE.read_text())
    final = followup._accepted_endpoint(raw)

    assert final["control_sha256"] == followup.CONTROL_SHA
    assert "gradient_inf" not in final
    assert continuation.execution_status(json.loads(followup.BASE.with_suffix(".resource.json").read_text())) == "completed"
    assert json.loads(followup.BASE.with_suffix(".run.json").read_text())["execution_status"] == "failed"


def test_new_plan_cannot_omit_an_inherited_model_dependency():
    plan = json.loads(followup.PLAN.read_text())
    del plan["source_files"]["src/advar/physics.py"]
    with tempfile.TemporaryDirectory(prefix="followup-plan-test-", dir=followup.ROOT) as stage:
        path = Path(stage) / "plan.json"
        path.write_text(json.dumps(plan))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="omits current kernel sources"):
            followup._require_pins(path, digest)
