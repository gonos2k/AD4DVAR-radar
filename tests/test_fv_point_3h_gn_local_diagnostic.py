from __future__ import annotations
import copy
from types import SimpleNamespace
import pytest
import torch
from examples.weather_scenarios import fv_point_3h_gn_local_diagnostic as probe


def test_closed_parent_is_final454_point_not_its_operator_base():
    parent=probe._load_parent()
    assert probe.tangent._tensor_sha(torch.tensor(parent['current_control'],dtype=torch.float64))==probe.BASE_SHA
    assert parent['final_repeat']['objective']==0.06102493753540203


def test_fixed_samples_all_evaluated_without_optimizer_acceptance():
    calls=[];receipts=[];model=SimpleNamespace(direction=torch.tensor([1.,0.],dtype=torch.float64),theta=.5)
    def evaluate(c,t,a):calls.append(a);return {'J_armijo_passed':True,'F_squared_armijo_passed':True}
    alphas=[.01/2**k for k in range(8)]
    samples,meta=probe.diagnostic_samples(model,torch.zeros(2,dtype=torch.float64),alphas,evaluate,
        chart_candidate=lambda c,d,a:c+a*d,progress=lambda t,m:receipts.append(copy.deepcopy((t,m))))
    assert calls==alphas and meta['evaluations_completed']==meta['slots_completed']==8
    assert not any(x['accepted'] for x in samples)
    assert all(x['J_armijo_passed'] for x in samples)


def test_local_timeout_saves_completed_and_interrupted_samples():
    calls=0;receipts=[];model=SimpleNamespace(direction=torch.tensor([1.,0.],dtype=torch.float64),theta=.5)
    def evaluate(*args):
        nonlocal calls
        calls+=1
        if calls==3:raise TimeoutError('deadline')
        return {'finite':True}
    with pytest.raises(TimeoutError):
        probe.diagnostic_samples(model,torch.zeros(2,dtype=torch.float64),[.01,.005,.0025],evaluate,
            chart_candidate=lambda c,d,a:c+a*d,progress=lambda t,m:receipts.append(copy.deepcopy((t,m))))
    samples,meta=receipts[-1]
    assert meta['slots_started']==3 and meta['evaluations_completed']==2
    assert samples[-1]['status']=='slot_interrupted' and not samples[-1]['accepted']
