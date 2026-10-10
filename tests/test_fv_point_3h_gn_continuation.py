from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_gn_continuation as continuation
from examples.weather_scenarios import fv_point_3h_event_direction_comparison as comparison


def test_closed_parent_is_bef459_and_whole_model_stays_current():
    parent = continuation._load_parent()
    control = torch.tensor(parent['current_control'], dtype=torch.float64)
    assert continuation.tangent._tensor_sha(control) == continuation.BASE_SHA
    residual = torch.linspace(-0.5, 0.5, 27, dtype=torch.float64)
    theta = parent['current_theta']
    assert continuation._base_model_closure(SimpleNamespace(residual=residual,theta=theta),residual,theta)['passed']
    changed=residual.clone();changed[0]+=1e-7
    assert not continuation._base_model_closure(SimpleNamespace(residual=changed,theta=theta),residual,theta)['passed']


def _closure(point: float):
    c=torch.full((26,),point,dtype=torch.float64)
    sides={'-1':[1.0]+[0.0]*25,'1':[-1.0]+[0.0]*25}
    proposal={'control':c.tolist(),'control_sha256':continuation.tangent._tensor_sha(c),
              'theta':.5,'objective':1.0,'F_squared':.25,'candidate_mixing_minimum':True,'side_gradients':sides}
    repeat={'control':c.tolist(),'theta':.5,'objective':1.0,'F_squared':.25,'side_gradients':sides,
            **{key:True for key in ('mixing_minimum_valid','minimum_theta_matches_proposal','side_gradients_finite',
                'native_objective_matches_proposal','side_objectives_match_native','merit_matches_proposal',
                'face_audit_passed','branch_pair_passed','trace_matches_proposal','source_unchanged',
                'fixed_input_unchanged','runtime_unchanged','deadline_passed')}}
    return proposal,repeat


def test_two_commits_update_current_last_count_and_history_atomically(tmp_path):
    output=tmp_path/'step.json'
    record={'last_confirmed_iterations':0,'closed_steps':[]}
    p,r=_closure(.1)
    record=continuation._commit_closed_point(record,{'point_index':0},p,r,output)
    p2,r2=_closure(.2)
    record=continuation._commit_closed_point(record,{'point_index':1},p2,r2,output)
    disk=json.loads(output.read_text())
    assert disk['current_control_sha256']==disk['last_confirmed_control_sha256']==p2['control_sha256']
    assert disk['last_confirmed_iterations']==disk['optimizer_steps_applied']==2
    assert [x['point_index'] for x in disk['closed_steps']]==[0,1]
    assert disk['closed_steps'][0]['final_repeat']==r
    assert disk['last_confirmed_closure']==r2


def test_failed_p2_or_write_keeps_previous_confirmed_prefix(tmp_path,monkeypatch):
    output=tmp_path/'step.json';p,r=_closure(.1)
    record=continuation._commit_closed_point({'last_confirmed_iterations':0,'closed_steps':[]},{},p,r,output)
    before=output.read_bytes();p2,r2=_closure(.2);r2['runtime_unchanged']=False
    with pytest.raises(ValueError):continuation._commit_closed_point(record,{},p2,r2,output)
    assert output.read_bytes()==before and record['last_confirmed_iterations']==1
    p2,r2=_closure(.2)
    def fail(*args):raise OSError('replace failed')
    monkeypatch.setattr(continuation.tangent,'_write',fail)
    with pytest.raises(OSError):continuation._commit_closed_point(record,{},p2,r2,output)
    assert output.read_bytes()==before and record['last_confirmed_iterations']==1


def test_commit_cap_refuses_fourth_point_without_replacing_file(tmp_path):
    output=tmp_path/'step.json';record={'last_confirmed_iterations':0,'closed_steps':[]}
    for i in range(3):
        p,r=_closure(.1*(i+1))
        record=continuation._commit_closed_point(record,{'point_index':i},p,r,output)
    before=output.read_bytes();p,r=_closure(.4)
    with pytest.raises(ValueError,match='cap'):
        continuation._commit_closed_point(record,{'point_index':3},p,r,output)
    assert output.read_bytes()==before
