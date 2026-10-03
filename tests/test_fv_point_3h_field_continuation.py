"""Pure artifact, field-restriction, and fresh-Hff gate tests; no FV execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_field_continuation as probe


def test_continuation_starts_only_at_attempt1_last_accepted_full_control():
    probe.require_pins()
    raw=json.loads(probe.RAW.read_text())
    start=probe.select_last_accepted_control(raw)
    assert probe.tensor_sha(start)=="b7a3e0274e99b299576457105010cceb247e4aba0edde6bbbc3e78dede868739"
    assert probe.tensor_sha(start)!=raw["accepted_control_sha256"]
    assert torch.equal(start[20:],torch.tensor(raw["fixed_dynamics"],dtype=torch.float64))
    assert raw["field_issuance"] is False and raw["numerical_status"]=="field_refused"


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
