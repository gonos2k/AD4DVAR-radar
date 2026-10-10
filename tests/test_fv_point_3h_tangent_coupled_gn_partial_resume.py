from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import tempfile
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_partial_resume as partial
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_resume as gn_resume


def _raw_receipt() -> dict[str, Any]:
    import gzip
    import json
    return json.loads(gzip.decompress(partial.PARTIAL_ARCHIVE.read_bytes()))


def test_saved_partial_receipt_admits_only_the_confirmed_prefix_and_derived_anchors():
    base = partial._load_current_base({})
    assert base["child_sha256"] == partial.PARTIAL_RAW_SHA
    assert base["control"].shape == (26,)
    assert base["theta"] == partial.BASE_THETA
    assert base["objective"] == partial.BASE_OBJECTIVE
    assert base["accepted"]["F_squared"] == partial.BASE_F_SQUARED
    assert base["anchor_provenance"] == {
        "input": "derived_from_pr273_input_after_control_sha_and_saved_control_flow_diagnostics_per_fixed_input_contract",
        "runtime": "reused_from_pr273_runtime_after"}
    assert set(base["raw"]) == {"input_after", "runtime_after"}
    pr273 = gn_resume._load_current_base({})
    expected_input = dict(pr273["raw"]["input_after"])
    expected_input["control_sha256"] = partial.BASE_CONTROL_SHA
    expected_input["shifted_flow_fractions"] = torch.tanh(base["control"][20:25]).tolist()
    assert base["raw"]["input_after"] == expected_input
    assert {key for key in set(expected_input) | set(pr273["raw"]["input_after"])
        if expected_input.get(key) != pr273["raw"]["input_after"].get(key)} == {
            "control_sha256", "shifted_flow_fractions"}
    raw = _raw_receipt()
    assert raw["jacobian_rows_started"] == 26
    assert raw["jacobian_rows_completed"] == 25
    closure, control = partial._closed_first_commit(raw)
    assert tangent._tensor_sha(control) == partial.BASE_CONTROL_SHA
    assert closure["F_squared"] == partial.BASE_F_SQUARED
    assert len(raw["iterations"]) == 1


@pytest.mark.parametrize("mutation", ["missing_closure", "unclosed_first_row"])
def test_partial_prefix_refuses_missing_or_invalid_first_closure(mutation):
    raw = _raw_receipt()
    if mutation == "missing_closure":
        raw["last_confirmed_closure"] = None
    else:
        raw["iterations"][0]["final_repeat"]["fixed_input_unchanged"] = False
        raw["last_confirmed_closure"]["fixed_input_unchanged"] = False
    with pytest.raises(ValueError):
        partial._closed_first_commit(raw)


def test_unfinished_second_row_is_not_in_confirmed_prefix_hash():
    raw = _raw_receipt()
    closure, control = partial._closed_first_commit(raw)
    before = tangent._tensor_sha(control)
    changed = copy.deepcopy(raw)
    changed["jacobian_row_history"][-1]["gradient"] = [9.0] * 26
    changed_closure, changed_control = partial._closed_first_commit(changed)
    assert changed_closure == closure
    assert tangent._tensor_sha(changed_control) == before == partial.BASE_CONTROL_SHA


@pytest.mark.parametrize("mutation", ["side", "theta"])
def test_loader_rejects_resealed_receipt_with_invalid_unfinished_tail(mutation):
    raw = _raw_receipt()
    if mutation == "side":
        raw["jacobian_row_history"][-1]["side"] = 1
    else:
        raw["jacobian_row_history"][-1]["theta"] = 0.123
    raw_bytes = (json.dumps(raw, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    compressed = gzip.compress(raw_bytes)
    raw_sha = partial.hashlib.sha256(raw_bytes).hexdigest()
    gzip_sha = partial.hashlib.sha256(compressed).hexdigest()

    with tempfile.TemporaryDirectory(prefix="partial-gn-tail-", dir=partial.ROOT) as temp:
        directory = Path(temp)
        raw_path = directory / "step.json"
        archive_path = directory / "step.json.gz"
        run_path = directory / "step.run.json"
        resource_path = directory / "step.resource.json"
        manifest_path = directory / "archive.json"
        raw_path.write_bytes(raw_bytes)
        archive_path.write_bytes(compressed)
        resource = json.loads(partial.PARTIAL_RESOURCE.read_text())
        run = json.loads(partial.PARTIAL_RUN.read_text())
        run["child_sha256"] = raw_sha
        run_bytes = (json.dumps(run, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        run_path.write_bytes(run_bytes)
        resource_path.write_bytes(partial.PARTIAL_RESOURCE.read_bytes())
        manifest = json.loads(partial.PARTIAL_MANIFEST.read_text())
        manifest.update(raw_path=raw_path.relative_to(partial.ROOT).as_posix(),
            gzip_path=archive_path.relative_to(partial.ROOT).as_posix(), raw_sha256=raw_sha,
            gzip_sha256=gzip_sha, run_path=run_path.relative_to(partial.ROOT).as_posix(),
            run_sha256=partial.hashlib.sha256(run_bytes).hexdigest(),
            resource_path=resource_path.relative_to(partial.ROOT).as_posix(),
            raw_bytes=len(raw_bytes), gzip_bytes=len(compressed))
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(partial, "PARTIAL_RAW", raw_path)
            monkeypatch.setattr(partial, "PARTIAL_ARCHIVE", archive_path)
            monkeypatch.setattr(partial, "PARTIAL_RUN", run_path)
            monkeypatch.setattr(partial, "PARTIAL_RESOURCE", resource_path)
            monkeypatch.setattr(partial, "PARTIAL_MANIFEST", manifest_path)
            monkeypatch.setattr(partial, "PARTIAL_RAW_SHA", raw_sha)
            monkeypatch.setattr(partial, "PARTIAL_GZIP_SHA", gzip_sha)
            monkeypatch.setattr(partial, "PARTIAL_RUN_SHA", partial.hashlib.sha256(run_bytes).hexdigest())
            with pytest.raises(ValueError, match="prefix/tail boundary"):
                partial._load_current_base({})


def test_fresh_gradient_mismatch_is_rejected_before_gn_model_or_hvp(tmp_path, monkeypatch):
    base = partial._load_current_base({})
    accepted = base["accepted"]
    gradients = {side: torch.as_tensor(accepted["side_gradients"][str(side)], dtype=torch.float64)
        for side in (-1, 1)}
    gradients[-1][0] += 1e-3
    traces = {side: {"signature_sha256": accepted["branch_trace"][str(side)]["signature_sha256"]}
        for side in (-1, 1)}
    observation = {"native_j": torch.tensor(partial.BASE_OBJECTIVE, dtype=torch.float64),
        "side": {side: (torch.tensor(partial.BASE_OBJECTIVE, dtype=torch.float64), gradients[side])
            for side in (-1, 1)}, "traces": traces, "pair": {"passed": True},
        "q": torch.tensor(0.0, dtype=torch.float64), "production_q": torch.tensor(0.0, dtype=torch.float64),
        "face_bound": 1e-12, "face_ok": True, "side_objectives_match_native": True,
        "side_gradients_finite": True}

    source_count = {"gn_model": 0}

    class ToyProblem:
        pass

    problem = ToyProblem()
    params = torch.zeros(1, dtype=torch.float64)
    original = torch.zeros((4, 5), dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)
    plan = {"policy": partial.POLICY, "source_files": {}, "archive_files": {}}
    monkeypatch.setattr(partial, "_load_plan", lambda *_: plan)
    monkeypatch.setattr(partial, "_load_current_base", lambda _: base)
    monkeypatch.setattr(tangent.seed, "_prepare_fixed_seed", lambda: (
        problem, original, None, params, truth, {}))
    monkeypatch.setattr(shared, "_input_identity", lambda *_: base["raw"]["input_after"])
    monkeypatch.setattr(shared, "_runtime", lambda: base["raw"]["runtime_after"])
    monkeypatch.setattr(shared, "_source_hashes", lambda *_: {})
    monkeypatch.setattr(shared, "_deadline", lambda *_: None)
    monkeypatch.setattr(tangent, "_observe", lambda *_: observation)
    monkeypatch.setattr(geometry, "_face_weights", lambda *_args, **_kwargs: torch.full((5,), 0.84))

    def model_factory(_context):
        def build(*_args):
            source_count["gn_model"] += 1
            return []
        return build

    with pytest.raises(ValueError, match="fresh J/F/side gradients"):
        tangent._run_child_impl(Path("plan.json"), "synthetic", tmp_path / "child.json",
            plan_loader=partial._load_plan, base_loader=partial._load_current_base,
            base_control_sha256=partial.BASE_CONTROL_SHA,
            direction_model_factory=model_factory)
    assert source_count["gn_model"] == 0

