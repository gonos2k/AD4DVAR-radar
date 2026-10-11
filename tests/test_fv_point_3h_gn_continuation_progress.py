from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_search_progress as tracked
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as frozen


def _model():
    return SimpleNamespace(direction=torch.tensor([1.,0.],dtype=torch.float64),theta=.5,
        gates={'finite':True,'both_side_gradients_descend':True,'merit_descends':True},
        merit_product=-.01,residual_direction=torch.tensor([1.,0.],dtype=torch.float64))


def _facts(accepted=False):
    keys=('mixing_minimum_valid','J_armijo_passed','branch_pair_passed','face_audit_passed',
          'side_objectives_match_native','side_gradients_finite','finite')
    return {**{k:True for k in keys},'F_squared_armijo_passed':accepted}


def test_timeout_keeps_two_completed_and_third_started_evaluation():
    receipts=[];calls=0
    def evaluate(*args):
        nonlocal calls
        calls+=1
        if calls==3:raise TimeoutError('deadline')
        return _facts()
    with pytest.raises(TimeoutError):
        tracked.search_candidates(_model(),torch.zeros(2,dtype=torch.float64),evaluate,
            chart_candidate=lambda c,d,a:c+a*d,progress=lambda t,m:receipts.append(copy.deepcopy((t,m))))
    trials,meta=receipts[-1]
    assert meta['slots_started']==meta['evaluations_started']==3
    assert meta['slots_completed']==meta['evaluations_completed']==2
    assert [x['status'] for x in trials]==['gate_refused','gate_refused','slot_interrupted']
    assert not any(x['accepted'] for x in trials)


def test_chart_failure_is_counted_without_claiming_function_evaluation():
    receipts=[]
    def chart(*args):raise ValueError('chart failed')
    with pytest.raises(ValueError):
        tracked.search_candidates(_model(),torch.zeros(2,dtype=torch.float64),lambda *a:_facts(),
            chart_candidate=chart,progress=lambda t,m:receipts.append(copy.deepcopy((t,m))))
    trials,meta=receipts[-1]
    assert meta['slots_started']==1 and meta['slots_completed']==0
    assert meta['evaluations_started']==meta['evaluations_completed']==0
    assert trials[0]['status']=='slot_failed'


def test_search_matches_frozen_first_pass_without_changing_acceptance():
    current=torch.zeros(2,dtype=torch.float64);chart=lambda c,d,a:c+a*d
    counts=[0,0]
    def evaluator(which):
        def evaluate(*args):
            counts[which]+=1;return _facts(counts[which]==4)
        return evaluate
    expected,trials,meta=frozen._search_candidates(_model(),current,evaluator(0),chart_candidate=chart)
    actual,newtrials,newmeta=tracked.search_candidates(_model(),current,evaluator(1),chart_candidate=chart,progress=lambda *a:None)
    assert actual is not None and expected is not None
    assert actual['alpha']==expected['alpha'] and actual['control']==expected['control']
    assert [x['accepted'] for x in newtrials]==[x['accepted'] for x in trials]
    assert newmeta['evaluated_count']==meta['evaluated_count']==4
    assert newmeta['evaluations_started']==newmeta['evaluations_completed']==4


def test_progress_receipt_does_not_add_slots_twice_across_points():
    record={'candidate_slots_used':9,'last_confirmed_iterations':1,'closed_steps':[{'P2':True}]}
    prior=record['candidate_slots_used']
    def progress(trials,meta):
        record.update(candidate_trials=copy.deepcopy(trials),search=copy.deepcopy(meta),
                      candidate_slots_used=prior+meta['slots_started'])
    tracked.search_candidates(_model(),torch.zeros(2,dtype=torch.float64),lambda *a:_facts(True),
        chart_candidate=lambda c,d,a:c+a*d,progress=progress)
    assert record['candidate_slots_used']==10
    assert record['last_confirmed_iterations']==1 and record['closed_steps']==[{'P2':True}]
