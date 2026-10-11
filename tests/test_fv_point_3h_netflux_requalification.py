from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_current_gradient_endpoint as endpoint
from examples.weather_scenarios import fv_point_3h_netflux_requalification as requal

DTYPE = torch.float64


def _base() -> dict[str, Any]:
    raw_bytes = gzip.decompress(endpoint.BASE_GZIP.read_bytes())
    raw = json.loads(raw_bytes)
    repeat = raw["closed_steps"][-1]["final_repeat"]
    return {"control": torch.tensor(raw["current_control"], dtype=DTYPE),
        "raw": raw, "repeat": repeat,
        "full_G": torch.tensor(raw["current_full_G"], dtype=DTYPE),
        "input_after": raw["input_after"], "runtime_after": raw["runtime_after"]}


def _install_mocks(monkeypatch, base, *, fault: str | None = None):
    normal = torch.zeros(26, dtype=DTYPE)
    normal[0] = 1.0
    gm, gp = torch.zeros(26, dtype=DTYPE), torch.zeros(26, dtype=DTYPE)
    gm[0], gp[0] = -1.0, 1.0
    if fault == "release_unresolved_mixture":
        gm[0], gp[0] = 1.0, 1.0
    gm[1] = gp[1] = 0.25
    traces = {-1: {"signature_sha256": "minus"}, 1: {"signature_sha256": "plus"}}
    observed = {"native_j": torch.tensor(1.0, dtype=DTYPE),
        "side": {-1: (torch.tensor(1.0, dtype=DTYPE), gm),
                 1: (torch.tensor(1.0, dtype=DTYPE), gp)},
        "traces": traces, "q": torch.tensor(0.0, dtype=DTYPE),
        "production_q": 0.0, "face_bound": 1e-14,
        "pair": {"passed": fault != "branch"}, "face_ok": True,
        "side_objectives_match_native": fault != "cost",
        "side_gradients_finite": True}

    class Backend:
        def __init__(self):
            old = base["raw"]["source_after"]
            self.source_anchor = {**old, "src/advar/transport.py": "new-transport-sha"}
            self.plan_sha = "new-plan-sha"
            self.control, self.problem = base["control"].clone(), object()
            self.parameters, self.weights = torch.zeros(1, dtype=DTYPE), torch.zeros(1)
            self.original, self.truth = object(), object()
            self.input_anchor, self.runtime_anchor = base["input_after"], base["runtime_after"]
            self.repeat_calls = 0

        def _check(self, _control):
            return None

        def repeat(self, _bundle, trial):
            self.repeat_calls += 1
            side_gradients = trial["side_gradients"]
            if fault == "full_g":
                side_gradients = json.loads(json.dumps(side_gradients))
                side_gradients["-1"][1] += 1e-5
            repeat = {"control": trial["control"].tolist(), "theta": trial["theta"],
                "objective": trial["objective"], "F_squared": trial["F_squared"],
                "side_gradients": side_gradients,
                "mixing_minimum_valid": True, "minimum_theta_matches_proposal": True,
                "native_objective_matches_proposal": fault != "repeat_cost",
                "side_objectives_match_native": True, "side_gradients_finite": True,
                "merit_matches_proposal": True, "face_audit_passed": True,
                "branch_pair_passed": True, "trace_matches_proposal": fault != "repeat_trace",
                "fixed_input_unchanged": True, "runtime_unchanged": True,
                "source_unchanged": True, "deadline_passed": True,
                "branch_trace": trial["branch_trace"]}
            proposal = {"control": trial["control"].tolist()}
            return proposal, repeat

    backend = Backend()
    monkeypatch.setattr(requal.tangent, "_observe", lambda *_args: observed)
    monkeypatch.setattr(requal.geometry, "_face_normal", lambda *_args: normal)
    monkeypatch.setattr(requal.geometry, "_pivot", lambda *_args: 0)
    chart = torch.zeros((26, 25), dtype=DTYPE)
    chart[1:, :] = torch.eye(25, dtype=DTYPE)
    monkeypatch.setattr(requal.geometry, "_chart_jacobian", lambda *_args: (None, chart, None))
    monkeypatch.setattr(requal.shared, "_input_identity", lambda *_args: base["input_after"])
    monkeypatch.setattr(requal.shared, "_runtime", lambda: base["runtime_after"])

    def fresh_full_g(repeat, expected, _weights):
        minus = torch.tensor(repeat["side_gradients"]["-1"], dtype=DTYPE)
        plus = torch.tensor(repeat["side_gradients"]["1"], dtype=DTYPE)
        actual = torch.cat(((1 - repeat["theta"]) * minus + repeat["theta"] * plus,
                            torch.zeros(1, dtype=DTYPE)))
        match = requal.finite._full_G_match(actual, expected)
        merit = float(torch.dot(actual, actual))
        return {"G": actual.tolist(), "component_match": match,
            "merit_match": abs(merit - repeat["F_squared"]) <= 1e-12,
            "passed": match["passed"] and abs(merit - repeat["F_squared"]) <= 1e-12}

    monkeypatch.setattr(requal.previous, "fresh_full_G", fresh_full_g)
    return backend, observed


def test_requalification_closes_fresh_repeat_without_gating_on_old_p2(monkeypatch, tmp_path):
    base = _base()
    backend, _ = _install_mocks(monkeypatch, base)
    result = requal.requalify_base(backend, base, tmp_path / "requalified.json")
    assert result["status"] == "new_evaluator_current_P2_requalified"
    assert result["control_sha256"] == requal.BASE_SHA
    assert result["new_transport_source_sha256"] != result["old_transport_source_sha256"]
    assert result["new_repeat_matches_first_full_G"]
    assert backend.repeat_calls == 1
    assert json.loads((tmp_path / "requalified.json").read_text())["repeat"] == result["repeat"]


@pytest.mark.parametrize("fault,match", [
    ("branch", "pair, face"),
    ("cost", "pair, face, cost"),
    ("repeat_cost", "independent net-flux repeat"),
    ("repeat_trace", "independent net-flux repeat"),
    ("full_g", "independent net-flux repeat full-G"),
])
def test_requalification_rejects_invalid_new_evaluator_closures(
        monkeypatch, tmp_path, fault, match):
    base = _base()
    backend, _ = _install_mocks(monkeypatch, base, fault=fault)
    with pytest.raises(ValueError, match=match):
        requal.requalify_base(backend, base, tmp_path / "failed.json")


def test_requalification_rejects_reused_transport_identity(monkeypatch, tmp_path):
    base = _base()
    backend, _ = _install_mocks(monkeypatch, base)
    backend.source_anchor["src/advar/transport.py"] = base["raw"]["source_after"][
        "src/advar/transport.py"]
    with pytest.raises(ValueError, match="changed transport source"):
        requal.requalify_base(backend, base, tmp_path / "failed.json")


def test_release_gate_precedes_unresolved_mixture(monkeypatch, tmp_path):
    base = _base()
    backend, _ = _install_mocks(monkeypatch, base, fault="release_unresolved_mixture")

    def mixture_must_not_run(*_args, **_kwargs):
        raise AssertionError("native face release must be checked before mixture minimization")

    monkeypatch.setattr(requal.tangent, "minimum_mixture_weight", mixture_must_not_run)
    output = tmp_path / "release.json"
    with pytest.raises(ValueError, match="new evaluator native face release required"):
        requal.requalify_base(backend, base, output)
    saved = json.loads(output.read_text())
    assert saved["status"] == "native_face_release_required_no_Q0_commit"
    assert saved["outward_derivatives"]["release_required"]
    assert backend.repeat_calls == 0


def test_requalification_rejects_changed_trusted_input_identity(monkeypatch, tmp_path):
    base = _base()
    backend, _ = _install_mocks(monkeypatch, base)
    monkeypatch.setattr(requal.shared, "_input_identity", lambda *_args: {"replaced": True})
    with pytest.raises(ValueError, match="input/runtime"):
        requal.requalify_base(backend, base, tmp_path / "failed.json")
