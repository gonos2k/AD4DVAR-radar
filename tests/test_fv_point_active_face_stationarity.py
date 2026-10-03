"""Small branch/domain/linear-audit regressions; no FV optimization."""
import pytest
import torch
import json
from typing import Any, cast

from examples.weather_scenarios import fv_point_active_face_stationarity as probe


def _stage():
    y, x = torch.meshgrid(torch.arange(4, dtype=torch.float64),
                          torch.arange(5, dtype=torch.float64), indexing="ij")
    q = 1.5 + .04*y + .03*x + .01*x*y + .001*x*x + .002*y*y + .0003*x*y*y
    qx = torch.ones((4, 6), dtype=torch.float64)
    qx[3, 4] = 0
    return q, qx, torch.ones((5, 5), dtype=torch.float64)


def test_only_named_structural_zero_is_exempt_from_other_face_margin():
    q, x, y = _stage()
    row, slope, face = probe.stage_record(q, x, y, 7)
    assert row["qx_sign"][3][4] == 0 and row["stage"] == 1
    assert slope > 1e-4 and face == 1
    y[0, 0] = 0
    with pytest.raises(probe.PointFaceRefusal, match="nonselected"):
        probe.stage_record(q, x, y, 7)


def test_limiter_refusal_is_numeric_but_broken_structural_zero_is_invariant_error():
    q, x, y = _stage()
    with pytest.raises(probe.PointFaceRefusal, match="zero or tied"):
        probe.stage_record(torch.ones_like(q), x, y, 0)
    x[3, 4] = 1e-300
    with pytest.raises(RuntimeError, match="structural zero"):
        probe.stage_record(q, x, y, 0)


def test_known_pivot_domain_refuses_but_unrelated_program_error_propagates():
    class Binding:
        def __init__(self, message):
            self.message = message
        def lift(self, _control):
            raise RuntimeError(self.message)
    c = torch.zeros(25, dtype=torch.float64)
    p = torch.zeros(13, dtype=torch.float64)
    with pytest.raises(probe.PointFaceRefusal):
        probe.trace(cast(Any, Binding("face coordinate is outside the strict representable chart domain")), c, p)
    with pytest.raises(RuntimeError, match="programming mistake"):
        probe.trace(cast(Any, Binding("programming mistake")), c, p)


def test_zero_rhs_cannot_pass_true_relative_residual_audit_as_nan():
    objective = lambda c, _p: .5 * c.square().sum()
    row: dict[str, Any] = {"input_tangent": [0.]*25, "rhs": [0.]*25, "solution": [0.]*25}
    row.update(eta=0.0, input_signature="test", rtol=1e-10, max_iterations=104,
               converged=True, iterations=1)
    with pytest.raises(ValueError, match="positive gradient norm"):
        probe.audit_solves(objective, torch.zeros(13, dtype=torch.float64), [row], "test")


def test_saved_newton_branch_or_budget_metadata_cannot_be_rebound():
    objective = lambda c, _p: .5 * c.square().sum()
    row = {"eta": 1.0, "input_signature": "other", "rtol": 1e-6,
           "max_iterations": 105, "error": "operator failed"}
    with pytest.raises(ValueError, match="branch or numerical budget"):
        probe.audit_solves(objective, torch.zeros(13, dtype=torch.float64), [row], "fixed")


def test_curvature_gate_distinguishes_known_positive_and_negative_direction():
    d = torch.arange(1, 26, dtype=torch.float64)
    c = torch.zeros(25, dtype=torch.float64)
    p = torch.zeros(13, dtype=torch.float64)
    positive = probe.curvature(lambda z, _p: .5*(d*z.square()).sum(), c, p)
    assert positive["qualified"] and positive["eigenvalues"][0] == 1
    d[0] = -1
    negative = probe.curvature(lambda z, _p: .5*(d*z.square()).sum(), c, p)
    assert not negative["qualified"] and negative["eigenvalues"][0] == -1


def test_initial_expected_refusal_keeps_terminal_record_without_optimization(monkeypatch, tmp_path):
    saved = json.loads(probe.EVENT.read_text())
    monkeypatch.setattr(probe.platform, "python_version", lambda: saved["runtime"]["python"])
    monkeypatch.setattr(probe.torch, "__version__", saved["runtime"]["torch"])
    p = torch.zeros(13, dtype=torch.float64)
    monkeypatch.setattr(probe.event.preflight, "fixed_problem",
                        lambda: (object(), torch.zeros(26, dtype=torch.float64), p, p.clone()))
    monkeypatch.setattr(probe.event.preflight, "_input_identity", lambda *_args: saved["input_before"])
    class Binding:
        def __init__(self, _problem):
            pass
        def tangent_coordinates(self, _control):
            return torch.zeros(25, dtype=torch.float64)
        def lift(self, _control):
            raise RuntimeError("face coordinate is outside the strict representable chart domain")
        def objective(self, control, _parameters):
            return .5 * control.square().sum()
    monkeypatch.setattr(probe, "FVPointActiveFaceBinding", Binding)
    def refuse(*_args):
        raise probe.PointFaceRefusal("known initial domain refusal")
    monkeypatch.setattr(probe, "trace", refuse)
    def unexpected_optimizer(*_args, **_kwargs):
        raise AssertionError("initial refusal must occur before optimization")
    monkeypatch.setattr(probe, "refine_stationary", unexpected_optimizer)
    target = tmp_path / "refusal.json"
    result = probe.run(target)
    assert result["phase"] == "finished"
    assert result["numerical_status"] == "branch_or_curvature_refusal"
    assert result["last_accepted_control"] is None
    assert result["response_validation"] == "not_performed"
    assert json.loads(target.read_text()) == result
