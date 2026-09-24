"""Full probe orchestration on an analytic 86/241 problem, no FV solve."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from examples.weather_scenarios import fv86_reanalysis_probe as probe


def toy(monkeypatch,tmp_path,mutation=None):
    dtype=torch.float64
    p=torch.ones(241,dtype=dtype)
    c=p[:86].clone()
    direction=torch.zeros_like(p);direction[80:160]=1
    branch={'euler_stages':108,'minimum_scaled_slope_margin':1.,'choices':[1],'face_signs':[1]}
    def objective(c,p):
        return .5*(c-p[:86]).square().sum()
    def score(c,p):
        return .5*c.square().sum()+.05*p.square().sum()
    case=SimpleNamespace(parameters=p,verification=torch.zeros(1,dtype=dtype),truth_initial_echo=torch.ones(1,dtype=dtype),
        observations=SimpleNamespace(dbz=p[:240].reshape(3,8,10)),definition={'toy':True},
        frozen=SimpleNamespace(fv_transport=SimpleNamespace(psi_basis=torch.ones(1,dtype=dtype))))
    fixture=SimpleNamespace(__file__=str(probe.EXAMPLES/'fv_scaled_research_case.py'),make_case=lambda:case,
        functions=lambda case,expected_branch=None:(objective,score,lambda c,p:(branch,'analytic')),
        actual_cfl=lambda case,c:0.)
    cached={'adjoint':c.tolist(),'score_control_gradient':c.tolist(),'direct_gradient':(.1*p).tolist(),
        'mixed_gradients':{'middle_time_bias':(-direction[:86]).tolist()},
        'direct':{'middle_time_bias':8.},'indirect':{'middle_time_bias':6.},'total':{'middle_time_bias':14.}}
    if mutation=='direct':cached['direct_gradient'][0]=.2
    if mutation=='adjoint':cached['adjoint'][0]=2.
    if mutation=='rhs':cached['score_control_gradient'][0]=2.
    if mutation=='mixed':
        cached['mixed_gradients']['middle_time_bias'][0]=1.
        cached['mixed_gradients']['middle_time_bias'][1]=-1.
    data={'mode':'seed_a','status':'eligible','source_sha256':{'src/advar/matrix_free.py':probe.sha256_file(probe.ROOT/'src/advar/matrix_free.py')},
        'input_identity':{'parameters':probe.digest(p),'verification':probe.digest(case.verification),
            'initial_echo':probe.digest(case.truth_initial_echo),'basis':probe.digest(case.frozen.fv_transport.psi_basis),
            'problem':probe.digest(case.definition)},
        'nominal_branch':branch,'final_score':float(score(c,p)),
        'branch_checks':[{'phase':'response','control_sha256':probe.digest(c),**probe.branch_summary(branch)}],
        'workflow':{'control':c.tolist(),'after':{'objective':0.},'response':cached}}
    path=tmp_path/'baseline.json';path.write_text(json.dumps(data))
    monkeypatch.setattr(probe,'BASELINE_PATH',path)
    monkeypatch.setattr(probe,'BASELINE_SHA256',probe.sha256_file(path))
    monkeypatch.setattr(probe,'load',lambda _:fixture)
    return p,c


def test_full_probe_varies_parameters_at_both_endpoints(monkeypatch,tmp_path):
    p,c=toy(monkeypatch,tmp_path)
    result=probe.run(tmp_path/'result.json')
    assert result['response_validation']=='passed'
    assert result['nonlinear_reanalyses']==4
    assert len(result['pairs'])==2
    assert all(row['passed'] for row in result['pairs'])
    for endpoint in result['endpoints']:
        pp=torch.tensor(endpoint['parameters'],dtype=p.dtype)
        cc=torch.tensor(endpoint['control'],dtype=p.dtype)
        assert not torch.equal(pp,p)
        torch.testing.assert_close(cc,pp[:86],rtol=0,atol=0)
    torch.testing.assert_close(p,torch.ones_like(p),rtol=0,atol=0)
    assert result['inputs_unchanged'] is True


@pytest.mark.parametrize('mutation',['direct','adjoint','rhs','mixed'])
def test_bad_cached_derivatives_fail_before_tangent(monkeypatch,tmp_path,mutation):
    toy(monkeypatch,tmp_path,mutation)
    path=tmp_path/'bad.json'
    with pytest.raises(ValueError):probe.run(path)
    result=json.loads(path.read_text())
    assert result['linear_solves']==[]
    assert result['response_validation']=='not_performed'


def test_pair_error_uses_declared_response_scale():
    slope,error=probe.pair_error(3.002,2.998,.001,2.,scale=2.)
    assert slope==pytest.approx(2.)
    assert error<1e-10


@pytest.mark.parametrize('reject_band,expected_pairs',[(0,3),(1,4)])
def test_refused_pairs_reset_consecutive_validation(monkeypatch,tmp_path,reject_band,expected_pairs):
    p,_=toy(monkeypatch,tmp_path)
    fixture=probe.load('fixture')
    original=fixture.functions
    def functions(case,expected_branch=None):
        objective,score,check=original(case,expected_branch)
        def guarded(c,pp):
            amplitude=float((pp-p).abs().max())
            reject=amplitude>.00075 if reject_band==0 else .0004<amplitude<.00075
            if reject:raise ValueError('synthetic branch refusal')
            return check(c,pp)
        return objective,score,guarded
    fixture.functions=functions
    result=probe.run(tmp_path/'result.json')
    assert result['response_validation']=='passed'
    assert len(result['pairs'])==expected_pairs
    assert all(pair['passed'] for pair in result['pairs'][-2:])
    assert any(endpoint['status']=='refused' for endpoint in result['endpoints'])
