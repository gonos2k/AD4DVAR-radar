"""Bounded original-J/R continuation from requalified cached7839 derivatives."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import torch

ROOT=Path(__file__).resolve().parents[3];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
CACHE_WORK=WORK.parent/'subspace_rank_fix_20261012'
sys.path.insert(0,str(CACHE_WORK))
import cached_operators as cache
from examples.weather_scenarios import fv_point_3h_netflux_continuation as continuation
from examples.weather_scenarios import fv_point_3h_netflux_subspace_trial as single

PLAN,ATTEMPT=WORK/'PROGRESS_PLAN.json',WORK/'progress1'
BASE_SHA=cache.BASE_SHA256;VERSION=continuation.VERSION
RADIUS=2*cache.BASE_RADIUS
MAX_COMMITS,MAX_HVP,MAX_CANDIDATES_PER_POINT=19,36,6
PRIOR_SECONDS=79.70184350013733
POLICY={**continuation.POLICY,'max_commits':MAX_COMMITS,'hvp_calls':MAX_HVP,'max_candidates':24,
    'max_candidates_per_point':6,'candidate_cap_per_point':6,'radius':RADIUS,
    'prior_preparation_seconds':PRIOR_SECONDS,'historical_HVP_calls':4,'total_HVP_cap_including_preparation':40,
    'historical_row_VJP_calls':24,'historical_GN_solves':1,'jacobian_row_vjp_calls':0,'dense_solves':0,
    'stop_on_rolling_comparison':False,'stagnation_action':'record only; no technique-selection gate or new GN comparison',
    'budget_scope':'one600s/660s including charged79.7018435s cachedpreparation,2GiB',
    'base_qualification':'archived7839 plus freshnative/P2/sidegradient/direction and cachedDGd/model closure',
    'candidate_policy':'gradient residual model alpha bounded2base then dyadicJR; chart actualalpha safeguard',
    'future_score_evaluations':0,'GN_gate':False,'chart_calls_per_candidate':33}
CHANGED_SOURCES={'examples/weather_scenarios/fv_point_3h_subspace_model.py','examples/weather_scenarios/fv_point_3h_netflux_continuation.py'}
NEW_SOURCES={str((WORK/n).relative_to(ROOT)) for n in ('cached_gradient_progress.py','cached_operators.py','current_model_audit.py')}|{
    'tests/test_fv_point_3h_active_qr_regression.py','tests/test_fv_point_3h_cached_current_model.py','tests/test_fv_point_3h_cached_gradient_progress.py'}
RECEIPTS=(cache.PLAN,cache.MANIFEST,cache.SOURCE_TAR,cache.RECEIPTS_TAR,cache.CLOSED_ENDPOINT_SNAPSHOT)


def load_endpoint()->dict[str,Any]:
    admitted=cache.admit();snapshot=admitted['endpoint_snapshot']['base'];raw=admitted['step']
    return {'control':torch.tensor(snapshot['control'],dtype=torch.float64),
        'full_G':torch.tensor(snapshot['full_G'],dtype=torch.float64),'theta':snapshot['theta'],
        'repeat':snapshot['repeat'],'input_after':snapshot['context'],
        'runtime_after':raw['runtime_after'],'parent_plan':admitted['plan']}

endpoint=SimpleNamespace(load_endpoint=load_endpoint)


def load_plan(path:Path,digest:str)->dict[str,Any]:
    if path.resolve()!=PLAN.resolve() or continuation._sha(path)!=digest:
        raise ValueError('progress plan identity mismatch')
    parent=load_endpoint()['parent_plan'];plan=json.loads(path.read_text())
    archives=set(parent['archive_files'])|{p.relative_to(ROOT).as_posix() for p in RECEIPTS}
    transition={n:{'before':parent['source_files'][n],'after':continuation._sha(ROOT/n)} for n in CHANGED_SOURCES}
    if (plan.get('experiment_kind')!='cached7839_original_JR_bounded_continuation'
            or plan.get('numerical_version')!=VERSION or plan.get('base_control_sha256')!=BASE_SHA
            or plan.get('policy')!=POLICY or plan.get('source_transition')!=transition
            or set(plan.get('source_files',{}))!=set(parent['source_files'])|NEW_SOURCES
            or set(plan.get('archive_files',{}))!=archives):
        raise ValueError('progress source/policy/identity scope changed')
    for name,d in parent['source_files'].items():
        if name not in CHANGED_SOURCES and plan['source_files'].get(name)!=d:
            raise ValueError(f'unapproved numerical input/source transition: {name}')
    if any(plan['archive_files'].get(n)!=h for n,h in parent['archive_files'].items()):
        raise ValueError('progress altered inherited archive pin')
    if any(continuation._sha(ROOT/n)!=h for n,h in {**plan['source_files'],**plan['archive_files']}.items()):
        raise ValueError('progress source/archive pin mismatch')
    return plan


def initial_products(backend:Any,c:torch.Tensor,direction:torch.Tensor,G:torch.Tensor,
        theta:float,record:dict[str,Any],point:dict[str,Any])->dict[int,torch.Tensor]:
    saved=cache.admit();raw=saved['step']
    baseline=next(m for m in raw['fresh_models'] if m['name']=='baseline_tangent')
    cached_direction=torch.tensor(baseline['direction'],dtype=c.dtype)
    if (continuation.tangent._tensor_sha(c)!=BASE_SHA or not torch.equal(direction,cached_direction)
            or theta!=raw['current_theta']
            or not continuation.finite._full_G_match(G,torch.tensor(raw['current_full_G'],dtype=c.dtype))['passed']
            or backend.input_anchor!=raw['input_after'] or backend.runtime_anchor!=raw['runtime_after']):
        raise ValueError('cached gradient operators failed current-point requalification')
    point.update(cached_HVP_calls_reused=2,cached_operator_pins=saved['pins'],
        cached_DGd=baseline['residual_direction'],cached_base_control_sha256=BASE_SHA)
    record.update(cached_HVP_calls_reused=2,prior_preparation_seconds=PRIOR_SECONDS,
        cached_historical_HVP_calls=4,cached_historical_row_VJP_calls=24,cached_historical_GN_solves=1)
    return {s:torch.tensor(baseline['side_HVP'][str(s)],dtype=c.dtype) for s in (-1,1)}


def qualify_model(model:Any,point_index:int,point:dict[str,Any])->None:
    if point_index==0:
        match=continuation.finite._full_G_match(model.residual_direction,
            torch.tensor(point['cached_DGd'],dtype=model.residual.dtype))
        point['cached_DGd_requalification']=match
        if not match['passed']:raise ValueError('cached HVPs do not reproduce qualified DGd')


def candidate_chart(backend:Any,bundle:dict[str,Any],c:torch.Tensor,
        direction:torch.Tensor,alpha:float)->torch.Tensor:
    candidate,factor,calls=single.chart_within_radius(backend,{**bundle,'direction':direction*alpha},c,RADIUS)
    bundle.setdefault('chart_alphas',{})[alpha]=(alpha*factor,calls)
    return candidate


def evaluate_candidate(backend:Any,bundle:dict[str,Any],candidate:torch.Tensor,alpha:float)->dict[str,Any]:
    actual_alpha,calls=bundle['chart_alphas'][alpha]
    facts=backend.evaluate(bundle,candidate,actual_alpha)
    facts.update(requested_alpha=alpha,chart_calls=calls)
    if 'G' in facts:
        fidelity=single.model_fidelity(bundle['G'],actual_alpha*bundle['model'].residual_direction,
            torch.tensor(facts['G'],dtype=candidate.dtype))
        facts['fidelity']={k:v if math.isfinite(v) else None for k,v in fidelity.items()}
    return facts


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--child',action='store_true');args=parser.parse_args()
    if args.child:
        result=continuation._run_child(PLAN,args.plan_sha256,ATTEMPT/'step.json',experiment=sys.modules[__name__])
        if result['execution_status']=='failed':raise SystemExit(1)
        return
    load_plan(PLAN,args.plan_sha256)
    if ATTEMPT.exists():raise ValueError('progress attempt must be fresh')
    ATTEMPT.mkdir()
    wall=POLICY['outer_seconds']-PRIOR_SECONDS
    resource=continuation.run_guarded_diagnostic([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child',
        '--plan-sha256',args.plan_sha256],wall_seconds=wall,rss_bytes=POLICY['rss_bytes'],
        report_path=ATTEMPT/'step.resource.json',log_path=ATTEMPT/'step.log')
    child=ATTEMPT/'step.json';status=continuation.shared._execution_status(resource,wall_seconds=wall,rss_bytes=POLICY['rss_bytes'])
    result={'execution_status':status,'resource':resource,'charged_prior_preparation_seconds':PRIOR_SECONDS,
        'total_guarded_seconds_including_preparation':resource['elapsed_seconds']+PRIOR_SECONDS,
        'child_sha256':continuation._sha(child) if child.exists() else None}
    continuation.tangent._write(ATTEMPT/'step.run.json',result);print(json.dumps(result))

if __name__=='__main__':main()
