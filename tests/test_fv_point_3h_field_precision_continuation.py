"""Pure artifact, field-restriction, and fresh-Hff gate tests; no FV execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_field_precision_continuation as probe


def test_continuation_starts_only_at_attempt1_last_accepted_full_control():
    probe.require_pins()
    raw=json.loads(probe.RAW.read_text())
    start=probe.select_last_accepted_control(raw)
    assert len([row for row in raw["trials"] if row.get("accepted") is True])==4
    assert probe.tensor_sha(start)=="ecfbbbd4919b8b783c93dce4a8dbb27598eac8eb9ca698bcf7ecd510f737b38b"
    assert probe.tensor_sha(start)!=raw["start_control_sha256"]
    assert torch.equal(start[20:],torch.tensor(raw["fixed_dynamics"],dtype=torch.float64))
    assert raw["field_issuance"] is False and raw["numerical_status"]=="continuation_refused"


def test_field_restriction_keeps_full_original_objective_and_frozen_dynamics_prior():
    matrix=torch.diag(torch.arange(1,27,dtype=torch.float64))
    p=torch.zeros(13,dtype=torch.float64); d=torch.linspace(-.3,.2,6,dtype=torch.float64)
    def original(control,_p): return .5*control@matrix@control + torch.sin(control).sum()
    reduced=probe.field_core.field_objective(SimpleNamespace(objective=original),d)
    f=torch.linspace(.1,.4,20,dtype=torch.float64); full=torch.cat((f,d))
    torch.testing.assert_close(reduced(f,p),original(full,p),rtol=0,atol=0)
    torch.testing.assert_close(torch.func.grad(reduced,argnums=0)(f,p),
        (matrix@full+torch.cos(full))[:20],rtol=1e-12,atol=1e-12)


def test_fresh_twenty_dimensional_hff_and_independent_21st_hvp_are_audited():
    generator=torch.Generator().manual_seed(13)
    q=torch.randn((20,20),generator=generator,dtype=torch.float64)
    h=q.T@q+torch.eye(20,dtype=torch.float64); p=torch.zeros(13,dtype=torch.float64)
    objective=lambda f,_p: .5*f@h@f
    result=probe.field_hessian_audit(objective,torch.zeros(20,dtype=torch.float64),p,float("inf"))
    torch.testing.assert_close(torch.tensor(result["hessian"],dtype=torch.float64),h,rtol=1e-12,atol=1e-12)
    assert result["hvp_columns"]==20 and result["independent_eigenpair_hvp"]
    assert result["lambda_min"]>result["positive_floor"] and result["condition"]<1e8
    assert result["eigenpair_relative_residual"]<=1e-8


def test_fresh_hff_gate_refuses_indefinite_field_curvature():
    h=torch.eye(20,dtype=torch.float64); h[0,0]=-1.
    objective=lambda f,_p: .5*f@h@f
    with pytest.raises(probe.ContinuationRefusal,match="resolved SPD"):
        probe.field_hessian_audit(objective,torch.zeros(20,dtype=torch.float64),
                                  torch.zeros(13,dtype=torch.float64),float("inf"))


def test_fresh_hff_eigensolver_failure_is_a_known_numerical_refusal(monkeypatch):
    h=torch.eye(20,dtype=torch.float64); objective=lambda f,_p: .5*f@h@f
    def fail(_matrix): raise torch.linalg.LinAlgError("synthetic eigensolver failure")
    monkeypatch.setattr(torch.linalg,"eigh",fail)
    with pytest.raises(probe.ContinuationRefusal,match="fresh Hff eigensolver failed"):
        probe.field_hessian_audit(objective,torch.zeros(20,dtype=torch.float64),
                                  torch.zeros(13,dtype=torch.float64),float("inf"))


def test_terminal_record_keeps_field_and_dynamics_gradient_residuals():
    original=lambda control,_p: .5*torch.dot(control,control)
    control=torch.linspace(-.2,.3,26,dtype=torch.float64)
    p=torch.zeros(13,dtype=torch.float64); problem=SimpleNamespace(objective=original)
    start=control.clone(); start[:20]=0
    mid=torch.cat((.5*control[:20],control[20:])); quarter=torch.cat((.25*control[:20],control[20:]))
    refusal={"field_issuance":False,"trials":[
        {"accepted":True,"candidate_full_control":mid.tolist()},
        {"accepted":False,"candidate_full_control":quarter.tolist()},
        {"accepted":True,"candidate_full_control":control.tolist()}]}
    selected,source=probe.select_terminal_control(refusal,start)
    row=probe.terminal_diagnostics(problem,selected,p)
    assert source=="last_accepted_precision_trial" and refusal["field_issuance"] is False
    torch.testing.assert_close(torch.tensor(row["terminal_full_gradient"],dtype=torch.float64),control)
    torch.testing.assert_close(torch.tensor(row["terminal_dynamics_gradient"],dtype=torch.float64),control[20:])
    assert row["terminal_gradient_blocks"]["field"]["l2"]==float(control[:20].norm())
    assert row["terminal_field_gradient_max"]==float(control[:20].abs().max())
    success={"field_issuance":True,"corrected_control":control.tolist()}
    selected_success,success_source=probe.select_terminal_control(success,start)
    success_row=probe.terminal_diagnostics(problem,selected_success,p)
    assert success_source=="corrected_field" and success_row["terminal_full_gradient"]==control.tolist()
    json.dumps({"refusal":row,"success":success_row},allow_nan=False)
    eligible={"terminal_diagnostics_status":"completed","terminal_diagnostics_after_wall_deadline":False,
              "terminal_field_gradient_max":.5e-10}
    assert probe.terminal_issuance_gate(eligible)==(True,True)
    late={**eligible,"terminal_diagnostics_after_wall_deadline":True}
    assert probe.terminal_issuance_gate(late)==(True,False)
    bad_gradient={**eligible,"terminal_field_gradient_max":2e-10}
    assert probe.terminal_issuance_gate(bad_gradient)==(False,False)
    expired=probe.terminal_diagnostics(problem,control,p,deadline=0.)
    assert expired["terminal_diagnostics_status"]=="post_audit_budget_refused"
    assert "terminal_full_gradient" not in expired
    json.dumps(expired,allow_nan=False)
    invalid=SimpleNamespace(objective=lambda c,_p: c.sum()*torch.tensor(float("nan")))
    bad=probe.terminal_diagnostics(invalid,control,p)
    assert bad["terminal_diagnostics_status"]=="numerical_refusal"
    assert "terminal_full_gradient" not in bad
    json.dumps(bad,allow_nan=False)
