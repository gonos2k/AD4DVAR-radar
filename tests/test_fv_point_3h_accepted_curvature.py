from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_accepted_curvature as curvature


ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / "graphify-out/fv-root-cause-20260919/bounded_coupled_original_j_20261005_attempt1/step.json"


def _step() -> dict[str, Any]:
    return json.loads(STEP.read_text())


def test_pinned_step_selects_only_the_accepted_f82c_full_point() -> None:
    step = _step()
    accepted = curvature._accepted_trial(step)
    assert step["accepted_control_sha256"] == curvature.CONTROL_SHA
    assert accepted["control_sha256"] == curvature.CONTROL_SHA
    assert accepted["branch_signature_changed"] is True
    assert accepted["strict_point_passed"] is True
    assert accepted["branch"]["euler_stages"] == 3600


def test_endpoint_metrics_check_every_gradient_component() -> None:
    accepted = curvature._accepted_trial(_step())
    gradient = torch.tensor(accepted["gradient"], dtype=torch.float64)
    checks = curvature._endpoint_metrics(
        accepted, torch.tensor(accepted["objective"], dtype=torch.float64),
        gradient, torch.tensor(accepted["phi"], dtype=torch.float64))
    assert curvature._endpoint_metrics_pass(checks)

    changed = gradient.clone()
    changed[0] += 1e-6
    # Keep the gradient infinity norm unchanged so this specifically exercises
    # componentwise identity with the Hessian's base-point gradient.
    assert float(changed.abs().max()) == float(gradient.abs().max())
    mismatch = curvature._endpoint_metrics(
        accepted, torch.tensor(accepted["objective"], dtype=torch.float64),
        changed, torch.tensor(accepted["phi"], dtype=torch.float64))
    assert mismatch["gradient_inf"]["passed"]
    assert not mismatch["gradient_components"][0]["passed"]
    assert not curvature._endpoint_metrics_pass(mismatch)


def test_accepted_step_rejects_control_or_branch_identity_drift() -> None:
    step = _step()
    altered = copy.deepcopy(step)
    altered["trials"][0]["control_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="accepted strict f82c endpoint"):
        curvature._accepted_trial(altered)

    altered = copy.deepcopy(step)
    altered["trials"][0]["armijo_passed"] = False
    with pytest.raises(ValueError, match="accepted strict f82c endpoint"):
        curvature._accepted_trial(altered)


def test_branch_partition_reports_analysis_and_future_as_distinct_hashes() -> None:
    signature = {"choices": [i % 2 for i in range(3600)],
                 "face_signs": [(-1) ** i for i in range(3600)]}
    partition = curvature._branch_partition(signature, (0.0, 600.0, 1200.0), 90)
    assert partition["analysis_stages"] == 360
    assert partition["future_stages"] == 3240
    altered = copy.deepcopy(signature)
    altered["choices"][359] = 99
    assert curvature._branch_partition(altered, (0.0, 600.0, 1200.0), 90)["analysis_signature_sha256"] != partition["analysis_signature_sha256"]
    altered = copy.deepcopy(signature)
    altered["face_signs"][360] = 99
    changed = curvature._branch_partition(altered, (0.0, 600.0, 1200.0), 90)
    assert changed["analysis_signature_sha256"] == partition["analysis_signature_sha256"]
    assert changed["future_signature_sha256"] != partition["future_signature_sha256"]


def test_branch_partition_refuses_incomplete_signatures() -> None:
    with pytest.raises(curvature.CurvatureRefusal, match="cannot be partitioned"):
        curvature._branch_partition({"choices": [0] * 3600, "face_signs": [0] * 3599},
                                    (0.0, 600.0, 1200.0), 90)


def test_expired_branch_setup_does_not_start_objective_or_hvp(tmp_path, monkeypatch):
    step = _step()
    accepted = curvature._accepted_trial(step)
    monkeypatch.setattr(curvature, "ROOT", tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(curvature, "_load_inputs", lambda *args:
                        ({"source_files": {}, "archive_files": {}}, step, accepted, {"source_before": {}}))
    parameters = torch.tensor(step["parameters"], dtype=torch.float64)
    identity = step["input_after"]
    monkeypatch.setattr(curvature.seed, "_prepare_fixed_seed", lambda:
        (object(), object(), None, parameters, object(), {"archived_input": identity["archived_input"]}))
    monkeypatch.setattr(curvature.seed, "_input_identity", lambda *args: identity)
    clock = {"now": 0.}
    monkeypatch.setattr(curvature.time, "monotonic", lambda: clock["now"])

    def branch(*args):
        clock["now"] = 241.
        return accepted["branch"], accepted["margins"]

    def forbidden(*args):
        raise AssertionError("expired branch setup cannot start J/g or HVP")

    monkeypatch.setattr(curvature.tail, "_full_current_branch", branch)
    monkeypatch.setattr(curvature.tail, "_fresh_merit", forbidden)
    monkeypatch.setattr(curvature.checkpoint_probe, "checkpoint_hessian", forbidden)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    result = curvature.run(plan_path=plan, plan_sha256=curvature.sha(plan),
                           output=attempt / "audit.json", checkpoint_directory=tmp_path / "checkpoint")
    assert result["numerical_status"] == "setup_budget_refused"
    assert result["hvp_calls_total"] == 0
    assert result["fixed_input_unchanged"] is True
