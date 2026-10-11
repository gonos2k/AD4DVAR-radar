from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_current_gradient_resume as resume

DTYPE = torch.float64


def _case(minus_normal: float, plus_normal: float, *, theta: float = 0.5):
    c = torch.zeros(26, dtype=DTYPE)
    minus, plus, normal = c.clone(), c.clone(), c.clone()
    minus[0] = minus_normal
    plus[0] = plus_normal
    minus[1] = plus[1] = 0.25
    normal[0] = 1.0
    G = torch.cat(((1 - theta) * minus + theta * plus, torch.zeros(1, dtype=DTYPE)))
    traces = {-1: {"signature_sha256": "minus"}, 1: {"signature_sha256": "plus"}}
    observed: dict[str, Any] = {"native_j": torch.tensor(1., dtype=DTYPE),
        "side": {-1: (torch.tensor(1., dtype=DTYPE), minus),
                 1: (torch.tensor(1., dtype=DTYPE), plus)},
        "traces": traces, "q": torch.tensor(0., dtype=DTYPE),
        "pair": {"passed": True}, "face_ok": True,
        "side_objectives_match_native": True, "side_gradients_finite": True}
    repeat = {"theta": theta, "objective": 1.,
        "side_gradients": {str(s): observed["side"][s][1].tolist() for s in (-1, 1)},
        "branch_trace": {str(s): traces[s] for s in (-1, 1)}}
    base = {"control": c, "theta": theta, "full_G": G, "repeat": repeat,
        "raw": {"terminal_future_score": 7.4}, "input_after": {"fixed": True},
        "runtime_after": {"fixed": True}}
    return observed, base, normal


def _run(monkeypatch, tmp_path, observed, base, normal):
    class Backend:
        def __init__(self, *_args):
            self.control = base["control"]
            self.parameters = torch.zeros(1, dtype=DTYPE)
            self.weights = normal
            self.source_anchor = {"source": "frozen"}
            self.input_anchor = base["input_after"]
            self.runtime_anchor = base["runtime_after"]
            self.problem, self.original, self.truth = object(), object(), object()
            self.timings = {}
        def _check(self, _control):
            return None

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no chart, HVP or search before qualification/face decision")

    monkeypatch.setattr(resume, "load_plan", lambda *_args: {})
    monkeypatch.setattr(resume.endpoint, "load_endpoint", lambda: base)
    monkeypatch.setattr(resume, "CurrentBackend", Backend)
    monkeypatch.setattr(resume.tangent, "_observe", lambda *_args: observed)
    monkeypatch.setattr(resume.geometry, "_face_normal", lambda *_args: normal)
    monkeypatch.setattr(resume.geometry, "_pivot", forbidden)
    monkeypatch.setattr(resume.geometry, "_chart_jacobian", forbidden)
    monkeypatch.setattr(resume.search, "search", forbidden)
    monkeypatch.setattr(resume.shared, "_input_identity", lambda *_args: base["input_after"])
    monkeypatch.setattr(resume.shared, "_source_hashes", lambda *_args: {"source": "frozen"})
    monkeypatch.setattr(resume.shared, "_runtime", lambda: base["runtime_after"])
    monkeypatch.setattr(resume.geometry.model, "_fixed_input", lambda *_args: True)
    output = tmp_path / "step.json"
    return output, lambda: resume._run_child(tmp_path / "plan.json", "synthetic", output)


@pytest.mark.parametrize("mn,pn", [(1., 2.), (-2., -1.), (1., 1.)])
def test_child_classifies_release_before_unsupported_mixture(monkeypatch, tmp_path, mn, pn):
    observed, base, normal = _case(mn, pn)
    with pytest.raises(ValueError):
        resume.tangent.minimum_mixture_weight(observed["side"][-1][1], observed["side"][1][1])
    output, run = _run(monkeypatch, tmp_path, observed, base, normal)
    result = run()
    assert result["numerical_status"] == "native_face_release_required_no_Q0_commit"
    assert result["current_point"]["previous_P2_match"]["passed"]
    assert result["current_point"]["outward_derivatives"]["release_required"]
    assert result["hvp_started"] == result["candidate_evaluations_started"] == 0
    assert result["optimizer_steps_applied"] == 0
    assert not result["candidate_committed"]
    assert result["current_control"] == result["last_confirmed_control"] == base["control"].tolist()
    assert json.loads(output.read_text())["current_full_G"] == base["full_G"].tolist()


@pytest.mark.parametrize("damage", ["J", "side", "trace", "full_G"])
def test_child_p2_mismatch_cannot_become_face_release(monkeypatch, tmp_path, damage):
    observed, base, normal = _case(1., 2.)
    if damage == "J":
        observed["native_j"] += 1e-4
    elif damage == "side":
        observed["side"][-1][1][2] += 1e-4
    elif damage == "trace":
        observed["traces"] = copy.deepcopy(observed["traces"])
        observed["traces"][1]["signature_sha256"] = "changed"
    else:
        base["full_G"][2] += 1e-4
    output, run = _run(monkeypatch, tmp_path, observed, base, normal)
    with pytest.raises(ValueError, match="previous closed P2"):
        run()
    result = json.loads(output.read_text())
    assert result["execution_status"] == "failed"
    assert "outward_derivatives" not in result["current_point"]
    assert result["hvp_started"] == result["candidate_evaluations_started"] == 0
    assert not result["candidate_committed"]
    assert result["current_control"] == result["last_confirmed_control"]


@pytest.mark.parametrize("gate", ["face_ok", "side_objectives_match_native", "side_gradients_finite", "pair"])
def test_child_invalid_native_pair_cannot_become_release(monkeypatch, tmp_path, gate):
    observed, base, normal = _case(1., 2.)
    observed[gate] = {"passed": False} if gate == "pair" else False
    _, run = _run(monkeypatch, tmp_path, observed, base, normal)
    result = run()
    assert result["numerical_status"] == "unsupported_current_native_pair"
    assert "outward_derivatives" not in result["current_point"]
    assert result["hvp_started"] == result["candidate_evaluations_started"] == 0
    assert not result["candidate_committed"]


def test_child_unresolved_jump_without_release_is_supported_unresolved_state(monkeypatch, tmp_path):
    observed, base, normal = _case(0., 0.)
    _, run = _run(monkeypatch, tmp_path, observed, base, normal)
    result = run()
    assert result["numerical_status"] == "unsupported_current_mixture"
    assert "jump is unresolved" in result["termination_reason"]
    assert not result["current_point"]["outward_derivatives"]["release_required"]
    assert result["hvp_started"] == result["candidate_evaluations_started"] == 0
    assert not result["candidate_committed"]


def test_child_retained_interior_reaches_chart_only_after_p2_and_release(monkeypatch, tmp_path):
    observed, base, normal = _case(-1., 1.)
    output, run = _run(monkeypatch, tmp_path, observed, base, normal)
    with pytest.raises(AssertionError, match="no chart"):
        run()
    result = json.loads(output.read_text())
    assert result["current_point"]["minimum_mixture_P2_match"]["passed"]
    assert not result["current_point"]["outward_derivatives"]["release_required"]


def test_child_reminimized_theta_mismatch_still_fails(monkeypatch, tmp_path):
    observed, base, normal = _case(-1., 1., theta=0.25)
    output, run = _run(monkeypatch, tmp_path, observed, base, normal)
    with pytest.raises(ValueError, match="minimum mixture differs"):
        run()
    result = json.loads(output.read_text())
    assert result["current_point"]["previous_P2_match"]["passed"]
    assert not result["current_point"]["minimum_mixture_P2_match"]["passed"]
    assert result["hvp_started"] == result["candidate_evaluations_started"] == 0
    assert not result["candidate_committed"]
