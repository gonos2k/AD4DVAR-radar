from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'graphify-out/fv-root-cause-20260919/subspace_rank_fix_20261012/radius_expansion_trial.py'
spec=importlib.util.spec_from_file_location('test_radius_expansion_driver',PATH)
assert spec and spec.loader
expansion=importlib.util.module_from_spec(spec);sys.modules[spec.name]=expansion;spec.loader.exec_module(expansion)


def test_current_endpoint_is_archived_7839_without_physical_evaluation(monkeypatch):
    def forbidden(*_args,**_kwargs):raise AssertionError('endpoint invoked physical model')
    monkeypatch.setattr(expansion.trial.comparison,'CurrentBackend',forbidden)
    monkeypatch.setattr(expansion.trial.tangent,'_observe',forbidden)
    base=expansion.load_endpoint()
    assert expansion.trial.tangent._tensor_sha(base['control'])==expansion.BASE_SHA
    assert base['input_after']['control_sha256']==expansion.BASE_SHA
    assert base['repeat']['F_squared']==pytest.approx(float(base['full_G'].square().sum()),rel=1e-14)


def test_endpoint_refuses_changed_confirmed_G_even_with_valid_candidate(monkeypatch):
    original=expansion.audit.load_archived_step
    def changed():
        plan,raw,pins=original();raw=copy.deepcopy(raw);raw['current_full_G'][0]+=0.01
        return plan,raw,pins
    monkeypatch.setattr(expansion.audit,'load_archived_step',changed)
    with pytest.raises(ValueError,match='differs'):
        expansion.load_endpoint()


def test_radius_selector_resolves_gain_from_supplied_vectors_and_obeys_cap():
    # Archived d19e vectors are a unit fixture for the selector, not new7839 operators.
    _,raw,_=expansion.audit.load_archived_step()
    G=torch.tensor(raw['closed_steps'][0]['full_G'],dtype=torch.float64)
    D=torch.stack([torch.tensor(m['direction'],dtype=G.dtype) for m in raw['fresh_models']],1)
    V=torch.stack([torch.tensor(m['residual_direction'],dtype=G.dtype) for m in raw['fresh_models']],1)
    record={};radius=expansion.choose_radius(G,D,V,record)
    assert radius is not None and expansion.INITIAL_RADIUS<radius<=2*expansion.INITIAL_RADIUS
    facts=record['current_KKT_radius_selection']
    assert facts['normalized_coefficients'][1]>facts['GN_coefficient_budget']
    assert facts['predicted_added_R_decrease']>facts['model_decrease_budget']
    assert facts['status']=='resolved_positive_GN_and_added_model_gain'


def test_rank_zero_skips_radius_comparison_without_invoking_solver(monkeypatch):
    def forbidden(*_args,**_kwargs):raise AssertionError('unqualified KKT called solver')
    monkeypatch.setattr(expansion.trial,'solve_two_direction_subspace',forbidden)
    record={};G=torch.ones(2,dtype=torch.float64)
    assert expansion.choose_radius(G,torch.eye(2,dtype=G.dtype),torch.zeros((2,2),dtype=G.dtype),record) is None
    assert record['current_KKT_radius_selection']['rank']==0


def test_shared_worker_keeps_base_when_no_radius_is_qualified(monkeypatch,tmp_path):
    from test_fv_point_3h_netflux_subspace_trial import _run
    monkeypatch.setattr(expansion.trial,'choose_radius',lambda *_args:None)
    raw,counts=_run(monkeypatch,tmp_path)
    assert raw['numerical_status']=='no_resolved_subspace_entry_within_frozen_radius_cap'
    assert raw['candidate_evaluations_started']==0 and counts['repeat']==0
    assert raw['optimizer_steps_applied']==0 and raw['current_control']==raw['last_confirmed_control']


@pytest.mark.parametrize('margin',['lambda_sign_resolved','mu1_sign_resolved'])
def test_unresolved_base_KKT_margin_skips_extra_comparison(monkeypatch,margin):
    _,raw,_=expansion.audit.load_archived_step()
    G=torch.tensor(raw['closed_steps'][0]['full_G'],dtype=torch.float64)
    D=torch.stack([torch.tensor(m['direction'],dtype=G.dtype) for m in raw['fresh_models']],1)
    V=torch.stack([torch.tensor(m['residual_direction'],dtype=G.dtype) for m in raw['fresh_models']],1)
    original=expansion.audit.axis0_kkt
    def unresolved(*args):
        kkt=original(*args);kkt[margin]=False;return kkt
    monkeypatch.setattr(expansion.audit,'axis0_kkt',unresolved)
    record={}
    assert expansion.choose_radius(G,D,V,record) is None
    assert record['current_KKT_radius_selection']['status']=='base_gradient_axis_not_resolved_KKT'
