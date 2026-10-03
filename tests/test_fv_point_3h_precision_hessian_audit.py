"""Pure terminal selection and coupled full-H/Schur audit checks; no FV run."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_precision_hessian_audit as probe


def test_terminal_control_is_the_latest_accepted_precision_point():
    probe.require_pins()
    raw=json.loads(probe.RAW.read_text())
    control=probe.terminal_control(raw)
    assert raw["source_before"]==raw["source_after"]
    accepted=[trial for trial in raw["trials"] if trial.get("accepted") is True]
    assert len(accepted)==5
    assert torch.equal(control,torch.tensor(accepted[-1]["candidate_full_control"],dtype=torch.float64))
    assert torch.equal(control[20:],torch.tensor(raw["fixed_dynamics"],dtype=torch.float64))


def test_fresh_full_hvp_and_schur_report_keeps_nonsymmetric_hessian_indefinite():
    h=torch.eye(26,dtype=torch.float64)
    h[20:,20:]=torch.diag(torch.tensor([-1.,2.,3.,4.,5.,6.],dtype=torch.float64))
    h[0,20]=h[20,0]=1.
    control=torch.ones(26,dtype=torch.float64); p=torch.zeros(13,dtype=torch.float64)
    objective=lambda c,_p: .5*c@h@c
    gradient=h@control
    report=probe.fresh_hessian(objective,control,p,gradient,float("inf"))
    saved=torch.tensor(report["hessian"],dtype=torch.float64)
    torch.testing.assert_close(saved,h,rtol=0,atol=0)
    assert report["hvp_columns"]==26 and report["hvp_calls_total"]==27
    assert min(report["eigenvalues"])<0
    assert report["block_schur"]["status"]=="schur_evaluated"
    assert report["block_schur"]["hff_condition"]==1.0
    eliminated=report["block_schur"]["linearized_eliminated_dynamics_residual"]
    assert eliminated[0]==-2.0  # Includes nonzero g_f; this is a linearized residual.
    assert "not a profiled gradient" in report["block_schur"]["scope"]
    json.dumps(report,allow_nan=False)


def test_fresh_hessian_audit_refuses_overrun_after_eigenpair_or_schur(monkeypatch):
    h=torch.eye(26,dtype=torch.float64); control=torch.ones(26,dtype=torch.float64)
    p=torch.zeros(13,dtype=torch.float64); objective=lambda c,_p: .5*c@h@c
    for expire_after,expected_calls in ((27,0),(28,1)):
        calls=[]; schur_calls=[]
        def clock():
            calls.append(None)
            return 2.0 if len(calls)>expire_after else 0.0
        monkeypatch.setattr(probe,"time",SimpleNamespace(monotonic=clock))
        monkeypatch.setattr(probe.blocks,"schur_diagnostic",lambda *_args: (schur_calls.append(None) or {"status":"schur_evaluated"}))
        with pytest.raises(probe.AuditRefusal,match="240-second internal budget exhausted"):
            probe.fresh_hessian(objective,control,p,h@control,1.0)
        assert len(schur_calls)==expected_calls
