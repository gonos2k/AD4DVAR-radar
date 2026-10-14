from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012"
               / "signed_trial.py")
spec = importlib.util.spec_from_file_location("signed_span_test_module", MODULE_PATH)
assert spec is not None and spec.loader is not None
signed = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = signed
spec.loader.exec_module(signed)


def test_cached_hvp_key_refuses_wrong_control_and_direction_without_fallback(monkeypatch):
    dtype = torch.float64
    control = torch.zeros(26, dtype=dtype)
    direction = torch.ones(26, dtype=dtype)
    pair = torch.full_like(control, 2.0)
    context = {"model_state": {"observed": {"side": {
        -1: (torch.tensor(0.0, dtype=dtype), torch.zeros_like(control)),
        1: (torch.tensor(0.0, dtype=dtype), torch.ones_like(control)),
    }}}, "weights": torch.ones(5, dtype=dtype)}
    model = {"baseline_tangent": {"direction": direction,
        "side_HVP": {"-1": pair, "1": pair}}}
    monkeypatch.setattr(signed, "_CACHE", {"control": control, "theta": 0.5,
        "fresh_models": model})
    monkeypatch.setattr(signed, "_CONTEXT", context)
    monkeypatch.setattr(signed, "_verify_context", lambda *args: {})
    jvp_calls = 0
    original_jvp = torch.func.jvp

    def counted_jvp(*args, **kwargs):
        nonlocal jvp_calls
        jvp_calls += 1
        return original_jvp(*args, **kwargs)

    monkeypatch.setattr(torch.func, "jvp", counted_jvp)
    record = {"commits": 0}
    spec_ok = {"name": "baseline_tangent", "direction": direction}
    torch.testing.assert_close(signed.hvp_pair(context, control, spec_ok, -1), pair)
    with pytest.raises(ValueError, match="control"):
        signed.hvp_pair(context, control + 1.0, spec_ok, -1)
    with pytest.raises(ValueError, match="direction"):
        signed.hvp_pair(context, control,
            {"name": "baseline_tangent", "direction": direction + 1.0}, -1)
    assert jvp_calls == 0
    assert signed.POLICY["new_hvp_cap"] == 0
    assert record["commits"] == 0


def test_signed_solver_accepts_only_contracted_radii(monkeypatch):
    dtype = torch.float64
    G = torch.tensor([1.0, 0.0], dtype=dtype)
    D = torch.eye(2, dtype=dtype)
    V = torch.eye(2, dtype=dtype)
    observed_radii = []

    def solve(g, d, v, radius):
        observed_radii.append(radius)
        coeff = torch.tensor([radius, 0.0], dtype=dtype)
        direction = d @ coeff
        residual = v @ coeff
        predicted = torch.linalg.vector_norm(g + residual).square()
        return signed.SubspaceModel(coeff, direction, residual, predicted, {})

    monkeypatch.setattr(signed, "solve_two_direction_subspace", solve)
    signed._signed_solve(G, D, V, signed.RADIUS / 2)
    assert observed_radii == [signed.RADIUS / 2] * 4
    with pytest.raises(ValueError, match="trust region"):
        signed._signed_solve(G, D, V, signed.RADIUS * 2)


@pytest.mark.parametrize("wrong_key", ["control", "direction"])
def test_executor_wrong_cached_key_stops_before_jvp_candidates_or_commit(
        monkeypatch, tmp_path, wrong_key):
    harness_path = ROOT / "tests/test_fv_point_3h_netflux_subspace_trial.py"
    harness_spec = importlib.util.spec_from_file_location("synthetic_runner_harness", harness_path)
    assert harness_spec is not None and harness_spec.loader is not None
    harness = importlib.util.module_from_spec(harness_spec)
    sys.modules[harness_spec.name] = harness
    harness_spec.loader.exec_module(harness)

    runner = signed.trial._run_child

    def run_signed(path, digest, output):
        return runner(path, digest, output, experiment=signed)

    monkeypatch.setattr(signed.trial, "_run_child", run_signed)
    monkeypatch.setattr(signed, "load_plan", lambda *_args: {})
    monkeypatch.setattr(signed, "endpoint", signed.trial.endpoint)
    monkeypatch.setattr(signed, "ATTEMPT", tmp_path / "attempt")

    def specs(context, control, *_args):
        signed._CONTEXT = context
        d = torch.zeros_like(control)
        d[0] = 1.0
        cached_control = (control + torch.nn.functional.one_hot(
            torch.tensor(0), num_classes=control.numel()).to(control)
            if wrong_key == "control" else control)
        cached_direction = (d + torch.nn.functional.one_hot(
            torch.tensor(1), num_classes=control.numel()).to(control)
            if wrong_key == "direction" else d)
        cached_model = {"direction": cached_direction,
            "side_HVP": {"-1": torch.zeros_like(control), "1": torch.zeros_like(control)}}
        signed._CACHE = {"control": cached_control,
            "fresh_models": {"baseline_tangent": cached_model}}
        return [{"name": "baseline_tangent", "direction": d},
            {"name": "residual_merit", "direction": -d}]

    monkeypatch.setattr(signed, "direction_specs", specs)
    monkeypatch.setattr(signed, "_verify_context", lambda *_args: {})
    raw, counts = harness._run(monkeypatch, tmp_path)
    assert raw["execution_status"] == "failed"
    assert raw["hvp_started"] == 0 and raw["hvp_completed"] == 0
    assert raw["candidate_evaluations_started"] == 0
    assert raw["optimizer_steps_applied"] == 0 and raw["candidate_committed"] is False
    assert counts["HVP"] == 0
