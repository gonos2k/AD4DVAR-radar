"""Synthetic shifted-block PCG/Armijo checks; no FV model execution."""
import json
from pathlib import Path

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_block_modified_newton_search as probe
from examples.weather_scenarios import fv_point_3h_schur_newton_step as schur


def _indefinite_original():
    h=torch.eye(26,dtype=torch.float64)
    h[20:,20:]=torch.diag(torch.tensor([-1.,2.,3.,4.,5.,6.],dtype=torch.float64))
    h[0,20]=h[20,0]=.5
    return h


def test_pinned_terminal_hessian_is_indefinite_and_terminal_inputs_are_frozen():
    probe.require_pins()
    import json
    raw=json.loads(probe.RAW.read_text())
    assert raw["terminal_control_sha256"]=="a890f344503cb3da02919de45f1c5b244beac300b1d4d2ca36f403c653e42b6d"
    assert min(raw["full_hessian_eigenvalues"])<0
    assert min(raw["block_schur"]["hff_eigenvalues"])>0
    assert min(raw["block_schur"]["schur_eigenvalues"])<0


def test_terminal_precision_raw_identity_matches_changed_control_not_base_control():
    raw=json.loads(probe.RAW.read_text())
    accepted=json.loads(probe.BASE_AUDIT.read_text())
    terminal=raw["input_after"]
    assert terminal["control_sha256"]==raw["terminal_control_sha256"]
    assert accepted["input_after"]["control_sha256"]!=terminal["control_sha256"]
    assert probe.terminal_input_matches(raw,terminal,accepted["input_after"],raw["terminal_control_sha256"])
    bad={**raw,"input_before":accepted["input_after"]}
    assert not probe.terminal_input_matches(bad,terminal,accepted["input_after"],raw["terminal_control_sha256"])


def test_shifted_block_preconditioner_solves_modified_system_not_original_newton():
    h=_indefinite_original(); g=torch.linspace(-.4,.7,26,dtype=torch.float64)
    schur_result=probe.blocks.schur_diagnostic(h,g)
    assert schur_result["status"]=="schur_evaluated" and schur_result["schur_eigenvalues"][0]<0
    mu=max(0.,1.-min(schur_result["schur_eigenvalues"]))
    hmu=h.clone(); hmu[20:,20:]+=mu*torch.eye(6,dtype=torch.float64)
    preconditioner,audit=probe.schur_probe.block_inverse_preconditioner(hmu)
    assert audit["full"]["lambda_min"]>0 and mu>0
    operator=probe.modified_operator(lambda v:h@v,mu)
    step,solve=probe.schur_probe.solve_newton_direction(operator,g,preconditioner)
    torch.testing.assert_close(operator(step)+g,torch.zeros_like(g),rtol=1e-10,atol=1e-10)
    original_residual=h@step+g
    assert float(original_residual.norm())>1e-4
    assert solve["true_relative_residual"]<=1e-10
    assert solve["g_dot_s"] < -solve["g_dot_s_rounding_budget"]


def test_armijo_condition_is_evaluated_on_unchanged_original_objective():
    h=_indefinite_original(); control=torch.linspace(-.1,.2,26,dtype=torch.float64)
    gradient=h@control; direction=-gradient
    base=float(.5*control@h@control)
    alpha=2.**-15; candidate=control+alpha*direction
    candidate_j=float(.5*candidate@h@candidate); slope=float(gradient@direction)
    assert probe.schur_probe.armijo_holds(base,candidate_j,alpha,slope)
    modified_candidate_j=float(.5*candidate@h@candidate + .5*alpha*alpha*direction[20:].square().sum())
    assert candidate_j!=modified_candidate_j


def test_nonfinite_trial_values_are_refusals_but_bad_callback_shapes_propagate():
    bad=torch.tensor(float("nan"),dtype=torch.float64)
    assert probe.candidate_metrics(bad,torch.zeros(26,dtype=torch.float64))["status"]=="nonfinite_objective"
    assert probe.candidate_metrics(torch.tensor(1.,dtype=torch.float64),torch.full((26,),float("inf"),dtype=torch.float64))["status"]=="nonfinite_gradient"
    with pytest.raises(ValueError,match="candidate J"):
        probe.candidate_metrics(torch.zeros(2,dtype=torch.float64),torch.zeros(26,dtype=torch.float64))
    with pytest.raises(ValueError,match="candidate gradient"):
        probe.candidate_metrics(torch.tensor(1.,dtype=torch.float64),torch.zeros(25,dtype=torch.float64))


def test_shared_pcg_deadline_is_translated_only_for_its_exact_budget_message():
    known=probe.schur_probe.StepRefusal("300-second internal budget exhausted during PCG")
    assert probe.refusal_message(known)=="240-second modified-Newton internal budget exhausted during PCG"
    residual=probe.schur_probe.StepRefusal("PCG or fresh original-H true residual gate failed")
    assert probe.refusal_message(residual)=="PCG or fresh modified-A_mu true residual gate failed"
    other=probe.schur_probe.StepRefusal("operator must be symmetric positive definite")
    assert probe.refusal_message(other)==str(other)
