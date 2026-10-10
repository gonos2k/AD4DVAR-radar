from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_memory_resume as memory


def _raw_packet() -> dict[str, Any]:
    import gzip
    import json
    return json.loads(gzip.decompress(memory.PARENT_ARCHIVE.read_bytes()))


def test_saved_memory_predecessor_admits_exact_two_step_p2_prefix_and_anchors():
    base = memory._load_current_base({})
    raw = _raw_packet()
    closure, control = memory._closed_two_step_prefix(raw)
    assert base["child_sha256"] == memory.PARENT_RAW_SHA
    assert tangent._tensor_sha(base["control"]) == memory.BASE_CONTROL_SHA
    assert torch.equal(base["control"], control)
    assert base["theta"] == memory.BASE_THETA
    assert base["objective"] == memory.BASE_OBJECTIVE
    assert closure["F_squared"] == memory.BASE_F_SQUARED
    assert base["anchor_provenance"] == memory.PROVENANCE
    assert set(base["raw"]) == {"input_after", "runtime_after"}
    assert memory.POLICY == {**memory.partial.POLICY, "rss_bytes": 2 * 1024**3}
    assert memory.RSS_TRANSITION["predecessor_sampled_peak_rss_bytes"] > memory.partial.POLICY["rss_bytes"]
    assert memory.RSS_TRANSITION["next_rss_limit_bytes"] == 2 * 1024**3
    assert raw["accepted_iterations"] == raw["optimizer_steps_applied"] == 2
    assert raw["hvp_calls_completed"] == 4
    assert raw["jacobian_rows_completed"] == 72
    assert raw["dense_solves_completed"] == 3


@pytest.mark.parametrize("mutation", ["missing_final", "side_gradient_drift"])
def test_two_step_prefix_requires_individual_gradient_p2_closure(mutation):
    raw = _raw_packet()
    if mutation == "missing_final":
        raw["iterations"][1]["final_repeat"] = None
    else:
        trial = next(trial for arm in raw["iterations"][1]["model_comparisons"]
            for trial in arm["trials"] if trial.get("accepted") is True)
        trial["side_gradients"]["-1"][0] += 1e-3
    with pytest.raises(ValueError, match="P2 closure"):
        memory._closed_two_step_prefix(raw)


@pytest.mark.parametrize("mutation", ["third_hvp", "third_candidate", "incomplete_rows"])
def test_uncommitted_third_point_boundary_rejects_new_work(mutation):
    raw = _raw_packet()
    if mutation == "third_hvp":
        raw["hvp_history"].append({**raw["hvp_history"][-1],
            "base_control_sha256": memory.BASE_CONTROL_SHA, "side": -1})
        raw["hvp_calls_started"] += 1
        raw["hvp_calls_completed"] += 1
    elif mutation == "third_candidate":
        raw["iterations"].append({"accepted": False, "base_control_sha256": memory.BASE_CONTROL_SHA})
    else:
        row = next(row for row in reversed(raw["jacobian_row_history"])
            if row["base_control_sha256"] == memory.BASE_CONTROL_SHA)
        row["status"] = "started"
    with pytest.raises(ValueError):
        memory._validate_two_step_packet(raw)


def test_fresh_side_gradient_mismatch_refuses_before_gn_model_or_hvp(tmp_path, monkeypatch):
    base = memory._load_current_base({})
    accepted = base["accepted"]
    gradients = {side: torch.as_tensor(accepted["side_gradients"][str(side)], dtype=torch.float64)
        for side in (-1, 1)}
    gradients[-1][0] += 1e-3
    traces = {side: {"signature_sha256": accepted["branch_trace"][str(side)]["signature_sha256"]}
        for side in (-1, 1)}
    observation = {"native_j": torch.tensor(memory.BASE_OBJECTIVE, dtype=torch.float64),
        "side": {side: (torch.tensor(memory.BASE_OBJECTIVE, dtype=torch.float64), gradients[side])
            for side in (-1, 1)}, "traces": traces, "pair": {"passed": True},
        "q": torch.tensor(0.0, dtype=torch.float64), "production_q": torch.tensor(0.0, dtype=torch.float64),
        "face_bound": 1e-12, "face_ok": True, "side_objectives_match_native": True,
        "side_gradients_finite": True}
    calls = {"gn_model": 0}

    class ToyProblem:
        pass

    problem = ToyProblem()
    parameters = torch.zeros(1, dtype=torch.float64)
    original = torch.zeros((4, 5), dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)
    plan = {"policy": memory.POLICY, "source_files": {}, "archive_files": {}}
    monkeypatch.setattr(memory, "_load_plan", lambda *_: plan)
    monkeypatch.setattr(memory, "_load_current_base", lambda _: base)
    monkeypatch.setattr(tangent.seed, "_prepare_fixed_seed", lambda: (
        problem, original, None, parameters, truth, {}))
    monkeypatch.setattr(shared, "_input_identity", lambda *_: base["raw"]["input_after"])
    monkeypatch.setattr(shared, "_runtime", lambda: base["raw"]["runtime_after"])
    monkeypatch.setattr(shared, "_source_hashes", lambda *_: {})
    monkeypatch.setattr(shared, "_deadline", lambda *_: None)
    monkeypatch.setattr(tangent, "_observe", lambda *_: observation)
    monkeypatch.setattr(geometry, "_face_weights", lambda *_args, **_kwargs: torch.full((5,), 0.84))

    def model_factory(_context):
        def build(*_args):
            calls["gn_model"] += 1
            return []
        return build

    with pytest.raises(ValueError, match="fresh J/F/side gradients"):
        tangent._run_child_impl(Path("plan.json"), "synthetic", tmp_path / "child.json",
            plan_loader=memory._load_plan, base_loader=memory._load_current_base,
            base_control_sha256=memory.BASE_CONTROL_SHA,
            direction_model_factory=model_factory)
    assert calls["gn_model"] == 0

