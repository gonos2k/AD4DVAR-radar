from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from examples.weather_scenarios import fv_point_3h_netflux_continuation as runner

PATH=Path(__file__).resolve().parents[1]/'graphify-out/fv-root-cause-20260919/subspace_active_qr_20261012/cached_gradient_progress.py'
spec=importlib.util.spec_from_file_location('test_cached_progress_driver',PATH)
assert spec and spec.loader
progress=importlib.util.module_from_spec(spec);sys.modules[spec.name]=progress;spec.loader.exec_module(progress)


def test_shared_loop_reuses_only_initial_pair_and_refreshes_changed_point(monkeypatch,tmp_path):
    from test_fv_point_3h_netflux_continuation import _synthetic_loop
    def initial(_backend,c,d,G,theta,record,point):
        record['cached_HVP_calls_reused']=2;point['cached_HVP_calls_reused']=2
        return {-1:d.clone(),1:d.clone()}
    monkeypatch.setattr(runner,'initial_products',initial)
    raw,counts,*_rest=_synthetic_loop(monkeypatch,tmp_path)
    assert raw['optimizer_steps_applied']==2
    assert counts['hvp']==2 and raw['hvp_completed']==2
    assert raw['closed_steps'][0]['hvp_completed']==0
    assert raw['closed_steps'][1]['hvp_completed']==2
    assert raw['closed_steps'][0]['base_control_sha256']!=raw['closed_steps'][1]['base_control_sha256']


def test_actual_alpha_selects_the_same_hashed_receipt_after_contraction(monkeypatch,tmp_path):
    from test_fv_point_3h_netflux_continuation import _synthetic_loop
    def contracted(backend,bundle,candidate,requested):
        facts=backend.evaluate(bundle,candidate,requested/2)
        return {**facts,'requested_alpha':requested}
    monkeypatch.setattr(runner,'evaluate_candidate',contracted)
    raw,*_rest=_synthetic_loop(monkeypatch,tmp_path)
    assert raw['optimizer_steps_applied']==2,raw.get('termination_reason')
    for point in raw['closed_steps']:
        selected=point['selected_candidate'];facts=__import__('json').loads((tmp_path/selected['candidate_receipt']).read_text())
        assert selected['alpha']==facts['alpha']==selected['requested_alpha']/2
        assert point['proposal']['alpha']==selected['alpha']


def test_pinned_initial_products_refuse_changed_direction_and_DGd():
    admitted=progress.cache.admit();base=progress.load_endpoint()
    raw=admitted['step'];m=raw['fresh_models'][0];c=base['control'];d=torch.tensor(m['direction'],dtype=c.dtype)
    backend=SimpleNamespace(input_anchor=base['input_after'],runtime_anchor=base['runtime_after'])
    record,point={},{}
    pair=progress.initial_products(backend,c,d,base['full_G'],base['theta'],record,point)
    assert set(pair)=={-1,1} and record['cached_HVP_calls_reused']==2
    with pytest.raises(ValueError,match='requalification'):
        progress.initial_products(backend,c,d+1e-10,base['full_G'],base['theta'],{}, {})
    model=SimpleNamespace(residual=base['full_G'],residual_direction=torch.tensor(m['residual_direction'],dtype=c.dtype))
    progress.qualify_model(model,0,point)
    model.residual_direction=model.residual_direction+1e-8
    with pytest.raises(ValueError,match='DGd'):
        progress.qualify_model(model,0,point)


def test_chart_evaluator_forwards_actual_candidate_and_scaled_model_alpha(monkeypatch):
    monkeypatch.setattr(progress,'RADIUS',1.)
    c=torch.zeros(2,dtype=torch.float64);d=torch.tensor([1.,0.],dtype=c.dtype)
    received={}
    class Backend:
        def _check(self,*_a):pass
        def chart(self,bundle,current,direction,alpha):
            return current+alpha*direction+torch.tensor([0.,alpha*alpha],dtype=c.dtype)
        def evaluate(self,bundle,candidate,alpha):
            received.update(candidate=candidate.clone(),alpha=alpha)
            return {'alpha':alpha}
    bundle={'direction':d};backend=Backend()
    candidate=progress.candidate_chart(backend,bundle,c,d,1.)
    facts=progress.evaluate_candidate(backend,bundle,candidate,1.)
    assert torch.equal(received['candidate'],candidate) and not torch.equal(candidate,c)
    assert 0<received['alpha']<1. and candidate.norm()<=1.
    assert facts['alpha']==bundle['chart_alphas'][1.][0]
