from __future__ import annotations
import copy
from types import SimpleNamespace
import pytest
import torch
from examples.weather_scenarios import fv_point_3h_gn_local_extension as extension


def test_same_point_operator_receipt_and_new_grid_are_qualified():
    parent=extension._load_parent()
    assert extension.tangent._tensor_sha(torch.tensor(parent['current_control'],dtype=torch.float64))==extension.BASE_SHA
    assert extension.tangent._tensor_sha(torch.tensor(parent['gn_direction']['direction'],dtype=torch.float64))==parent['gn_direction']['direction_sha256']
    smallest=parent['sample_alphas'][-1];new=[smallest/2**k for k in range(1,9)]
    assert len(new)==8 and new[0]==smallest/2 and new[-1]==smallest/256


def test_small_window_keeps_all_gate_facts_but_never_accepts():
    model=SimpleNamespace(direction=torch.tensor([1.,0.],dtype=torch.float64),theta=.5)
    samples,meta=extension.diagnostic_samples(model,torch.zeros(2,dtype=torch.float64),[1e-5,5e-6],
        lambda *a:{'J_armijo_passed':True,'F_squared_armijo_passed':True,'accepted':True},
        chart_candidate=lambda c,d,a:c+a*d,progress=lambda *a:None)
    assert meta['evaluations_completed']==2 and all(s['J_armijo_passed'] for s in samples)
    assert not any(s['accepted'] for s in samples)


def test_interrupted_window_retains_started_completed_difference():
    receipts=[];calls=0
    model=SimpleNamespace(direction=torch.tensor([1.,0.],dtype=torch.float64),theta=.5)
    def evaluate(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise TimeoutError('deadline')
        return {'finite':True}
    with pytest.raises(TimeoutError):
        extension.diagnostic_samples(model,torch.zeros(2,dtype=torch.float64),[1e-5,5e-6],evaluate,
            chart_candidate=lambda c,d,a:c+a*d,progress=lambda t,m:receipts.append(copy.deepcopy((t,m))))
    t,m=receipts[-1];assert m['slots_started']==2 and m['evaluations_completed']==1
    assert t[-1]['status']=='slot_interrupted'
