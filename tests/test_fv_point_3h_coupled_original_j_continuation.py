"""Synthetic coupled-J/modified-operator continuation checks; no FV execution."""
import json

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_coupled_original_j_continuation as probe


def _indefinite():
    h=torch.eye(26,dtype=torch.float64)
    h[20:,20:]=torch.diag(torch.tensor([-1.,2.,3.,4.,5.,6.],dtype=torch.float64))
    h[0,20]=h[20,0]=.5
    return h


def test_pinned_r8_start_and_fixed_input_contract():
    probe.require_pins()
    probe.require_pins()
    raw=json.loads(probe.R8.read_text()); accepted=json.loads(probe.AUDIT.read_text())
    assert raw["accepted_control_sha256"]=="3aa913cd71d5916b4f3146efad75b0f5dd6766f6bdc81ee6e8c6706cee03fc32"
    assert raw["parameters_sha256"]==accepted["parameters_sha256"]
    assert raw["input_before"]["parameters_sha256"]==accepted["input_after"]["parameters_sha256"]
    assert raw["input_before"]["archived_input"]==accepted["input_after"]["archived_input"]
    assert raw["input_before"]["control_sha256"]=="a890f344503cb3da02919de45f1c5b244beac300b1d4d2ca36f403c653e42b6d"
    assert raw["input_before"]==raw["input_after"]
    assert probe.audit._fixed_input_matches(raw["input_before"],accepted["input_after"],raw["control_sha256"])


def test_dynamics_shift_makes_indefinite_original_system_preconditionable():
    h=_indefinite(); g=torch.linspace(-.4,.7,26,dtype=torch.float64)
    schur=probe.blocks.schur_diagnostic(h,g)
    mu,preconditioner,audit=probe.choose_dynamics_shift(h,schur,schur["full_hessian_eigenvalues"])
    assert min(torch.linalg.eigvalsh(h))<0 and mu>0
    hmu=h.clone(); hmu[20:,20:]+=mu*torch.eye(6,dtype=torch.float64)
    assert float(torch.linalg.eigvalsh(hmu)[0])>0
    step,audit_solve=probe.block_probe.solve_newton_direction(
        probe.modified_operator(lambda v:h@v,mu),g,preconditioner)
    assert audit_solve["true_relative_residual"]<=1e-10
    assert float((h@step+g).norm())>1e-4
    assert audit["full_spd_resolved"] is False


def test_mu_zero_is_used_only_for_resolved_full_and_schur_spd():
    h=4*torch.eye(26,dtype=torch.float64); g=torch.ones(26,dtype=torch.float64)
    schur=probe.blocks.schur_diagnostic(h,g)
    mu,_,audit=probe.choose_dynamics_shift(h,schur,schur["full_hessian_eigenvalues"])
    assert mu==0 and audit["full_spd_resolved"] and audit["schur_spd_resolved"]


def test_original_objective_armijo_and_refusal_message_scope():
    h=_indefinite(); c=torch.linspace(-.1,.2,26,dtype=torch.float64); p=torch.zeros(13,dtype=torch.float64)
    j=lambda x:.5*x@h@x
    g=h@c; s=-g; alpha=2.**-15
    assert probe.block_probe.armijo_holds(float(j(c)),float(j(c+alpha*s)),alpha,float(g@s))
    assert probe.refusal_message(probe.block_probe.StepRefusal("300-second internal budget exhausted during PCG"))=="1200-second continuation budget exhausted during modified PCG"
    assert probe.refusal_message(probe.hess_probe.AuditRefusal("240-second internal Hessian audit budget exhausted"))=="1200-second continuation budget exhausted during fresh Hessian audit"
    assert probe.refusal_message(probe.block_probe.StepRefusal("PCG or fresh original-H true residual gate failed"))=="PCG or fresh modified-A_mu true residual gate failed"
    assert probe.refusal_message(probe.hess_probe.AuditRefusal("240-second internal budget exhausted after independent 27th HVP"))=="1200-second continuation budget exhausted after independent 27th HVP"
    assert probe.refusal_message(probe.hess_probe.AuditRefusal("240-second internal budget exhausted after Schur diagnostic"))=="1200-second continuation budget exhausted after Schur diagnostic"
    assert probe.refusal_message(probe.block_probe.StepRefusal("unrelated numeric refusal"))=="unrelated numeric refusal"
    assert probe.refusal_message(ValueError("programming defect"))=="programming defect"


def test_candidate_metrics_serializes_nonfinite_and_propagates_bad_shapes():
    finite=torch.zeros(26,dtype=torch.float64)
    assert probe.candidate_metrics(torch.tensor(float("nan"),dtype=torch.float64),finite)["status"]=="nonfinite_objective"
    assert probe.candidate_metrics(torch.tensor(1.,dtype=torch.float64),torch.full((26,),float("inf"),dtype=torch.float64))["status"]=="nonfinite_gradient"
    with pytest.raises(ValueError,match="candidate gradient"):
        probe.candidate_metrics(torch.tensor(1.,dtype=torch.float64),torch.zeros(25,dtype=torch.float64))


def test_accepted_epoch_four_can_finish_from_its_measured_gradient():
    gradient=torch.zeros(26,dtype=torch.float64); gradient[25]=.5e-10
    metrics=probe.candidate_metrics(torch.tensor(1.,dtype=torch.float64),gradient)
    assert probe.MAX_EPOCHS==4 and probe.accepted_endpoint_stationary(metrics)
    gradient[25]=2e-10
    assert not probe.accepted_endpoint_stationary(probe.candidate_metrics(torch.tensor(1.,dtype=torch.float64),gradient))


def test_fresh_hessian_is_recomputed_at_each_current_control():
    def objective(control,parameters):
        return .5*torch.dot(control,control)+control[0]**3/6
    parameters=torch.empty(0,dtype=torch.float64)
    left=torch.zeros(26,dtype=torch.float64); right=left.clone(); right[0]=.4
    gl=torch.func.grad(objective,argnums=0)(left,parameters)
    gr=torch.func.grad(objective,argnums=0)(right,parameters)
    first=probe.hess_probe.fresh_hessian(objective,left,parameters,gl,float("inf"))
    second=probe.hess_probe.fresh_hessian(objective,right,parameters,gr,float("inf"))
    assert first["hessian_sha256"]!=second["hessian_sha256"]
    assert first["hvp_calls_total"]==second["hvp_calls_total"]==27
