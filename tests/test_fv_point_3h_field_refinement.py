"""Synthetic fixed-dynamics field-refinement wiring; no FV model construction."""
from types import SimpleNamespace
import json
from pathlib import Path

import pytest
import torch

from advar import matrix_free
from advar.local_refinement import refine_stationary
from examples.weather_scenarios import fv_point_3h_field_refinement as probe
from examples.weather_scenarios import fv_partial_signed_face_slices as slices


def _quadratic():
    generator=torch.Generator().manual_seed(41)
    matrix=torch.randn((26,26),generator=generator,dtype=torch.float64)
    return matrix.T@matrix+torch.eye(26,dtype=torch.float64),torch.linspace(-.3,.4,26,dtype=torch.float64)


def test_pinned_endpoint_audit_qualifies_hff_but_not_full_hessian():
    probe.require_pins()
    audit=json.loads((Path(__file__).resolve().parents[1]/"graphify-out/fv-root-cause-20260919"
        /"point_3h_accepted_endpoint_audit_attempt1/audit.json").read_text())
    probe.require_resolved_hff(audit)
    assert min(audit["full_hessian_eigenvalues"])<0
    invalid={**audit,"block_schur":{**audit["block_schur"],"hff_eigenvalues":[-1.]}}
    with pytest.raises(probe.FieldBudgetRefusal,match="resolved SPD Hff"):
        probe.require_resolved_hff(invalid)


def test_field_objective_keeps_all_original_prior_terms_and_fixed_dynamics_coupling():
    matrix,target=_quadratic(); d=torch.linspace(-.2,.3,6,dtype=torch.float64)
    def original(control, _p):
        difference=control-target
        return .5*difference@matrix@difference+.5*control@control
    problem=SimpleNamespace(objective=original); reduced=probe.field_objective(problem,d)
    field=torch.linspace(.1,.5,20,dtype=torch.float64); p=torch.zeros(13,dtype=torch.float64)
    full=torch.cat((field,d)); expected=original(full,p)
    torch.testing.assert_close(reduced(field,p),expected,rtol=0,atol=0)
    expected_gradient=(matrix@(full-target))[:20]+field
    torch.testing.assert_close(torch.func.grad(reduced,argnums=0)(field,p),expected_gradient,rtol=1e-12,atol=1e-12)
    d.fill_(99.0)
    torch.testing.assert_close(reduced(field,p),expected,rtol=0,atol=0)


def test_synthetic_fixed_dynamics_refinement_converges_and_audits_each_field_cg():
    matrix,target=_quadratic(); p=torch.zeros(13,dtype=torch.float64)
    fixed_d=torch.linspace(-.1,.2,6,dtype=torch.float64)
    def original(control, _parameters):
        delta=control-target
        return .5*delta@matrix@delta+.5*control@control
    objective=probe.field_objective(SimpleNamespace(objective=original),fixed_d)
    field=torch.zeros(20,dtype=torch.float64)
    state={"tangent":field.clone(),"signature":"start"}; solves=[]; trials=[]
    def branch(candidate, parameters):
        assert torch.equal(parameters,p)
        state.update(tangent=candidate.clone(),signature="side+" if candidate[0]>=0 else "side-")
        return {"signature":state["signature"]},"synthetic fixed-dynamics branch"
    with matrix_free.observe_pcg_calls(slices._true_residual_monitor(solves,0.,state)):
        result=refine_stationary(objective,field,p,branch_check=branch,max_iterations=4,
            max_backtracks=16,pcg_max_iterations=104,trial_observer=trials.append)
    audits=probe.audit_field_solves(objective,p,solves)
    full=torch.cat((result.control,fixed_d))
    expected=torch.linalg.solve(matrix[:20,:20]+torch.eye(20,dtype=torch.float64),
        matrix[:20,:]@(target-torch.cat((torch.zeros(20,dtype=torch.float64),fixed_d))))
    torch.testing.assert_close(result.control,expected,rtol=1e-9,atol=1e-10)
    assert torch.equal(full[20:],fixed_d)
    assert float(torch.func.grad(objective,argnums=0)(result.control,p).abs().max())<1e-10
    assert solves and trials and all(row["status"]=="passed" for row in audits)
    assert all(row["max_iterations"]==104 and row["rtol"]==1e-10 for row in solves)


def test_bad_independent_residual_is_a_json_safe_numerical_refusal():
    objective=lambda f,_p: .5*torch.dot(f,f)
    row={"rtol":1e-10,"max_iterations":104,"input_tangent":[1.]*20,
         "rhs":[-1.]*20,"solution":[0.]*20}
    result=probe.audit_field_solves(objective,torch.zeros(13,dtype=torch.float64),[row])
    assert result[0]["status"]=="residual_refused"
    assert result[0]["true_relative_residual"]==1.
    json.dumps(result,allow_nan=False)
