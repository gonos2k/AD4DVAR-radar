from __future__ import annotations

import copy
import hashlib
import json
from typing import Any
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_netflux_continuation as runner


def _steps(ratio: float):
    steps = []
    for i in range(3):
        x = 0.25 * ratio ** i
        steps.append({'point_index': i, 'R': x*x, 'full_G': [0., x]+[0.]*25,
            'proposal': {'control_sha256': str(i)},
            'final_repeat': {'objective': .5*(ratio*x)**2, 'F_squared': (ratio*x)**2},
            'full_G_check': {'G': [0., ratio*x]+[0.]*25}})
    return steps


def test_rolling_stagnation_uses_three_commits_and_both_metrics():
    assert not runner.convergence_progress(_steps(.9999)[:2])['stagnation_detected']
    assert runner.convergence_progress(_steps(.9999))['stagnation_detected']
    assert not runner.convergence_progress(_steps(.99))['stagnation_detected']
    steps = _steps(.9999)
    regressed = _steps(.9999)
    regressed[-1]['full_G_check']['G'][1] = .26
    progress = runner.convergence_progress(regressed)
    assert progress['no_progress_detected'] and progress['window_metric_regression']
    assert not progress['stagnation_detected']
    assert progress['direction_comparison_required']
    regressed[-1]['final_repeat']['F_squared'] = regressed[0]['R'] * .99
    progress = runner.convergence_progress(regressed)
    assert progress['window_metric_regression'] and not progress['no_progress_detected']
    assert progress['direction_comparison_required']
    steps[-1]['final_repeat']['F_squared'] = steps[0]['R']*.99
    assert not runner.convergence_progress(steps)['stagnation_detected']


def _synthetic_loop(monkeypatch, tmp_path, *, fail_after_one=False, ratio=.9, initial=.25):
    dtype = torch.float64
    c = torch.zeros(26, dtype=dtype); c[1] = initial
    normal = torch.zeros_like(c); normal[0] = 1.
    counts = {'hvp': 0, 'search': 0, 'observe': 0}
    def observation(control):
        gm = torch.zeros_like(control); gp = gm.clone()
        gm[0], gp[0] = -1., 1.
        gm[1] = gp[1] = control[1]
        trace = {'signature_sha256': runner.tangent._tensor_sha(control)}
        return {'native_j': .5*control[1]**2, 'q': torch.tensor(0.,dtype=dtype),
            'side': {-1:(.5*control[1]**2,gm), 1:(.5*control[1]**2,gp)},
            'traces': {-1:trace, 1:trace}, 'pair': {'passed':True},
            'face_ok':True, 'side_objectives_match_native':True, 'side_gradients_finite':True}
    def repeat(control):
        obs=observation(control)
        return {'control':control.tolist(), 'theta':.5, 'objective':float(obs['native_j']),
            'F_squared':float(control[1]**2),
            'side_gradients':{str(s):obs['side'][s][1].tolist() for s in (-1,1)},
            'branch_trace':{str(s):obs['traces'][s] for s in (-1,1)}}
    G=torch.cat((.5*(observation(c)['side'][-1][1]+observation(c)['side'][1][1]),torch.zeros(1,dtype=dtype)))
    base={'control':c,'theta':.5,'full_G':G,'repeat':repeat(c),'raw':{},
        'input_after':{'fixed':True},'runtime_after':{'fixed':True}}
    class Backend:
        def __init__(self,*_args):
            self.control=c.clone(); self.parameters=torch.zeros(1,dtype=dtype);self.weights=normal
            self.source_anchor={'source':'frozen'};self.input_anchor=base['input_after']
            self.runtime_anchor=base['runtime_after'];self.original=self.truth=object();self.timings={}
            self.problem=SimpleNamespace(objective=lambda x,p: .5*x[1]**2);self.hvp_count=0
        def _check(self,_c): pass
        def chart(self,*_args): raise AssertionError('search stub controls chart')
        def evaluate(self,bundle,candidate,alpha):
            r=repeat(candidate)
            return {**r,'control':candidate.tolist(),'alpha':alpha,
                'G':[0.,float(candidate[1])]+[0.]*25}
        def repeat(self,bundle,selected):
            control=torch.as_tensor(selected['control'],dtype=dtype)
            r=repeat(control)
            return ({**selected,'control_sha256':runner.tangent._tensor_sha(control),
                'J_armijo_passed':True,'F_squared_armijo_passed':True,
                'J_armijo_bound':float(self.control[1]**2)*.5*(1.-.01),
                'F_squared_armijo_bound':float(self.control[1]**2)*(1.-.01)},r)
    def observe(_s,_p,control,*_args):
        counts['observe']+=1
        obs=observation(control)
        if fail_after_one and counts['observe']==2:
            obs=copy.deepcopy(obs);obs['traces'][1]['signature_sha256']='damaged'
        return obs
    def search(_model,control,evaluate,**_kwargs):
        counts['search']+=1
        candidate=control.clone();candidate[1]*=ratio
        selected=evaluate(candidate,.5,1-ratio)
        return selected,[{'accepted':True}],{}
    real_jvp=torch.func.jvp
    def jvp(*args,**kwargs):
        counts['hvp']+=1
        return real_jvp(*args,**kwargs)
    def full_g(r,expected,_weights):
        actual=[0.,r['control'][1]]+[0.]*25
        return {'G':actual,'passed':torch.equal(torch.tensor(actual,dtype=dtype),expected)}
    monkeypatch.setattr(runner,'_sha',lambda p:hashlib.sha256(p.read_bytes()).hexdigest())
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'ATTEMPT',tmp_path/'attempt')
    monkeypatch.setattr(runner,'load_plan',lambda *_a:{})
    monkeypatch.setattr(runner.endpoint,'load_endpoint',lambda:base)
    monkeypatch.setattr(runner,'CurrentBackend',Backend)
    monkeypatch.setattr(runner,'MAX_COMMITS',2)
    monkeypatch.setattr(runner.tangent,'_observe',observe)
    monkeypatch.setattr(runner.geometry,'_face_normal',lambda *_a:normal)
    monkeypatch.setattr(runner.geometry,'_pivot',lambda *_a:0)
    monkeypatch.setattr(runner.geometry,'_chart_jacobian',lambda *_a:(None,torch.eye(26,dtype=dtype)[:,1:],None))
    monkeypatch.setattr(runner.prepared.mixing,'minimum_tangent_model',lambda *_a,**_k:SimpleNamespace(
        gates={k:True for k in ('finite','both_side_gradients_descend','merit_descends','envelope_slope_matches_residual_dot')},merit_product=-1.,side_products={-1:-1.,1:-1.}))
    monkeypatch.setattr(runner.finite,'_base_model_closure',lambda *_a:{'passed':True})
    monkeypatch.setattr(runner.search,'search',search)
    monkeypatch.setattr(runner.torch.func,'jvp',jvp)
    monkeypatch.setattr(runner,'fresh_full_G',full_g)
    monkeypatch.setattr(runner.tangent,'fresh_final_closure',lambda *_a:True)
    monkeypatch.setattr(runner.shared,'_input_identity',lambda *_a:base['input_after'])
    monkeypatch.setattr(runner.shared,'_source_hashes',lambda *_a:{'source':'frozen'})
    monkeypatch.setattr(runner.shared,'_runtime',lambda:base['runtime_after'])
    monkeypatch.setattr(runner.geometry.model,'_fixed_input',lambda *_a:True)
    output=tmp_path/'step.json'
    return runner._run_child(tmp_path/'plan.json','test',output),counts,output


def test_two_steps_refresh_each_point_and_keep_new_version(monkeypatch,tmp_path):
    raw,counts,output=_synthetic_loop(monkeypatch,tmp_path)
    assert raw['numerical_status']=='commit_cap_reached'
    assert counts=={'hvp':4,'search':2,'observe':2}
    assert len(raw['closed_steps'])==raw['optimizer_steps_applied']==2
    assert raw['closed_steps'][1]['base_control_sha256']==raw['closed_steps'][0]['proposal']['control_sha256']
    assert raw['last_confirmed_numerical_version']==runner.VERSION
    assert raw['terminal_E_recomputed'] is False
    assert json.loads(output.read_text())==raw


def test_later_point_mismatch_preserves_first_commit(monkeypatch,tmp_path):
    raw,counts,output=_synthetic_loop(monkeypatch,tmp_path,fail_after_one=True)
    assert raw['execution_status']=='failed'
    assert raw['failure_stage']=='fresh_current_direction'
    assert counts=={'hvp':2,'search':1,'observe':2}
    assert raw['optimizer_steps_applied']==len(raw['closed_steps'])==1
    assert raw['current_control']==raw['last_confirmed_control']==raw['closed_steps'][0]['final_repeat']['control']
    assert raw['current_full_G']==raw['last_confirmed_full_G']
    assert json.loads(output.read_text())==raw


def test_postcommit_residual_candidate_precedes_cap(monkeypatch,tmp_path):
    raw,counts,_=_synthetic_loop(monkeypatch,tmp_path,ratio=.1,initial=2e-8)
    assert raw['numerical_status']=='residual_candidate_requires_qualification'
    assert raw['optimizer_steps_applied']==1
    assert counts['hvp']==2
    assert not raw['minimum_claim'] and not raw['root_claim']


@pytest.mark.parametrize('damage',['source','version','policy'])
def test_plan_refuses_any_further_numerical_or_policy_change(monkeypatch,tmp_path,damage):
    parent=json.loads(runner.PARENT_PLAN.read_text())
    monkeypatch.setattr(runner.endpoint,'load_endpoint',lambda:{'parent_plan':parent})
    plan: dict[str, Any]={'experiment_kind':'netflux_fixed_version_continuous_gradient','numerical_version':runner.VERSION,
        'base_control_sha256':runner.BASE_SHA,'policy':runner.POLICY,
        'source_files':{**parent['source_files'],**{p:runner._sha(runner.ROOT/p) for p in runner.NEW_SOURCES}},
        'archive_files':{**parent['archive_files'],**{p.relative_to(runner.ROOT).as_posix():runner._sha(p) for p in runner.endpoint.RECEIPTS}}}
    if damage=='source':plan['source_files']['src/advar/transport.py']='0'*64
    elif damage=='version':plan['numerical_version']='other'
    else:plan['policy']={**runner.POLICY,'max_commits':100}
    path=tmp_path/'PLAN.json';path.write_text(json.dumps(plan));monkeypatch.setattr(runner,'PLAN',path)
    monkeypatch.setattr(runner,'_sha',lambda p:hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='changed|mismatch'):
        runner.load_plan(path,runner._sha(path))


def test_backend_at_new_control_preserves_fixed_case_without_seed(monkeypatch,tmp_path):
    base=runner.endpoint.load_endpoint()
    def forbidden(*_args,**_kwargs):
        raise AssertionError('fixed-version continuation must not regenerate observations')
    monkeypatch.setattr(runner.previous.old.previous.seed,'_prepare_fixed_seed',forbidden)
    monkeypatch.setattr(runner.shared,'_source_hashes',lambda *_a:{runner.previous.TRANSPORT:runner.previous.NEW_TRANSPORT_SHA})
    monkeypatch.setattr(runner.CurrentBackend,'_check',lambda *_a:None)
    backend=runner.CurrentBackend({},tmp_path/'PLAN.json','plan',tmp_path,100.,base)
    assert backend.input_anchor==base['input_after']
    assert runner.tangent._tensor_sha(backend.control)==runner.BASE_SHA
    assert runner.tangent._tensor_sha(backend.truth)==base['input_after']['terminal_truth_sha256']
