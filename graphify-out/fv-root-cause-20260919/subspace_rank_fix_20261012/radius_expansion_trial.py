"""One KKT-selected radius comparison at freshly qualified control 7839625d."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import torch

ROOT=Path(__file__).resolve().parents[3]
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
import axis_radius_audit as audit
from expansion_policy import expansion_radius
from examples.weather_scenarios import fv_point_3h_netflux_subspace_trial as trial
from examples.weather_scenarios import fv_point_3h_current_gradient_endpoint as closure
PLAN,ATTEMPT=WORK/'EXPANSION_PLAN.json',WORK/'expansion1'
BASE_SHA='7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a'
INITIAL_RADIUS=trial.INITIAL_RADIUS
VERSION=trial.VERSION
CHANGED_SOURCES={'examples/weather_scenarios/fv_point_3h_subspace_model.py','examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py'}
NEW_SOURCES={str((WORK/n).relative_to(ROOT)) for n in ('radius_expansion_trial.py','axis_radius_audit.py','expansion_policy.py')}|{
    'tests/test_fv_point_3h_subspace_rank_regression.py','tests/test_fv_point_3h_subspace_radius_audit.py','tests/test_fv_point_3h_subspace_radius_expansion.py'}
RECEIPTS=(audit.PLAN,audit.RECEIPT_MANIFEST,audit.SOURCE_TAR,audit.RECEIPTS_TAR)
POLICY={**trial.POLICY,'base_qualification':'PR301_archived_source_receipt_then_fresh_current_P2',
    'radius_cap':2*INITIAL_RADIUS,'radius_selection':'fresh current full-rank axis KKT entry, 10percent past entry, min1.25base capped2base; require resolved positive GN and added model decrease; otherwise no FV',
    'numerical_source_transition':'rank-aware subspace solver and explicit experiment config only',
    'candidate_policy':'one KKT selected expanded radius per arm; halve rejected/poor model; never expand further',
    'future_score_evaluations':0}


def load_endpoint() -> dict[str,Any]:
    parent,raw,_=audit.load_archived_step()
    before=trial.endpoint.load_endpoint()
    arms=raw['arms'];winner=trial.comparison.choose_repeated_arm(arms,torch.float64)
    if (len(arms)!=2 or winner is None or raw['selected_direction']!=winner['name']
            or len(raw['closed_steps'])!=1 or not raw['comparison_complete']
            or raw['current_control_sha256']!=BASE_SHA or raw['last_confirmed_control_sha256']!=BASE_SHA):
        raise ValueError('PR301 winner/confirmed endpoint is not closed')
    point=raw['closed_steps'][0]
    for arm in arms:
        arm_point={**point,'proposal':arm['proposal'],'final_repeat':arm['repeat'],'full_G_check':arm['full_G_check']}
        closure._close_step(arm_point,before['full_G'],trial.endpoint.BASE_SHA)
    repeat,G,control_sha=closure._close_step(point,before['full_G'],trial.endpoint.BASE_SHA)
    if (control_sha!=BASE_SHA or repeat!=winner['repeat'] or point['proposal']!=winner['proposal']
            or repeat['control']!=raw['current_control'] or raw['current_control']!=raw['last_confirmed_control']
            or repeat['theta']!=raw['current_theta'] or raw['current_theta']!=raw['last_confirmed_theta']
            or G.tolist()!=raw['current_full_G'] or raw['current_full_G']!=raw['last_confirmed_full_G']):
        raise ValueError('PR301 endpoint differs from its closed winner')
    return {'control':torch.tensor(repeat['control'],dtype=torch.float64),'theta':repeat['theta'],
        'full_G':G,'repeat':repeat,'input_after':raw['input_after'],'runtime_after':raw['runtime_after'],'parent_plan':parent}

endpoint=SimpleNamespace(load_endpoint=load_endpoint)


def load_plan(path:Path,digest:str)->dict[str,Any]:
    if path.resolve()!=PLAN.resolve() or trial.comparison._sha(path)!=digest:
        raise ValueError('radius expansion plan identity mismatch')
    parent=load_endpoint()['parent_plan'];plan=json.loads(path.read_text())
    archives=set(parent['archive_files'])|{p.relative_to(ROOT).as_posix() for p in RECEIPTS}
    transition={n:{'before':parent['source_files'][n],'after':trial.comparison._sha(ROOT/n)} for n in CHANGED_SOURCES}
    if (plan.get('experiment_kind')!='fresh_7839_KKT_bounded_radius_comparison'
            or plan.get('policy')!=POLICY or plan.get('source_transition')!=transition
            or plan.get('base_control_sha256')!=BASE_SHA or plan.get('numerical_version')!=VERSION
            or set(plan.get('source_files',{}))!=set(parent['source_files'])|NEW_SOURCES
            or set(plan.get('archive_files',{}))!=archives):
        raise ValueError('radius expansion source/policy/scope changed')
    for name,h in parent['source_files'].items():
        if name not in CHANGED_SOURCES and plan['source_files'].get(name)!=h:
            raise ValueError(f'unapproved numerical source change: {name}')
    if any(plan['archive_files'].get(n)!=h for n,h in parent['archive_files'].items()):
        raise ValueError('radius expansion changed inherited archive')
    if any(trial.comparison._sha(ROOT/n)!=h for n,h in {**plan['source_files'],**plan['archive_files']}.items()):
        raise ValueError('radius expansion source/archive pin mismatch')
    return plan


def choose_radius(G:torch.Tensor,D:torch.Tensor,V:torch.Tensor,record:dict[str,Any])->float|None:
    geom=audit.residual_geometry(G,D,V)
    facts:dict[str,Any]={'control_sha256':BASE_SHA,'base_radius':INITIAL_RADIUS,'rank':geom.V_rank,
        'old_d19e_operators_reused':False}
    record['current_KKT_radius_selection']=facts
    if geom.V_rank!=2:
        facts['status']='full_rank_axis_KKT_premise_not_met';return None
    kkt=audit.axis0_kkt(geom.M,geom.Q,geom.c,INITIAL_RADIUS)
    entry=audit.axis0_activation(geom.M,geom.Q,geom.c)
    facts.update(current_axis_KKT=kkt,activation=entry)
    if not (kkt['axis0_kkt'] and kkt['lambda_sign_resolved'] and kkt['mu1_sign_resolved']):
        facts['status']='base_gradient_axis_not_resolved_KKT';return None
    critical=entry['critical_axis_activation_radius']
    if critical is None or critical<=INITIAL_RADIUS:
        facts['status']='no_resolved_future_axis_entry';return None
    radius=expansion_radius(INITIAL_RADIUS,entry['critical_axis_activation_radius']) if entry['inactive_axis_activates_before_axis_lambda_zero'] else None
    if radius is None:
        facts['status']='no_resolved_entry_inside_radius_cap';return None
    solution=trial.solve_two_direction_subspace(G,D,V,radius)
    beta=solution.coefficients*geom.scales
    axis_beta=min(radius/(float(geom.Q[0,0])**.5),max(0.,-float(geom.c[0])/float(geom.M[0,0])))
    axis_R=(G+geom.V_normalized[:,0]*axis_beta).square().sum()
    gain=axis_R-solution.predicted_R
    eps=128*torch.finfo(G.dtype).eps
    gain_budget=eps*torch.maximum(axis_R.abs(),solution.predicted_R.abs()).clamp_min(torch.finfo(G.dtype).tiny)
    beta_budget=eps*beta.norm().clamp_min(torch.finfo(G.dtype).tiny)
    facts.update(selected_radius=radius,normalized_coefficients=beta.tolist(),predicted_added_R_decrease=float(gain),
        model_decrease_budget=float(gain_budget),GN_coefficient_budget=float(beta_budget))
    if solution.diagnostics['reduced_domain'] or not bool(beta[1]>beta_budget and gain>gain_budget):
        facts['status']='no_resolved_additional_subspace_gain';return None
    facts['status']='resolved_positive_GN_and_added_model_gain'
    return radius


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--child',action='store_true');args=parser.parse_args()
    if args.child:
        result=trial._run_child(PLAN,args.plan_sha256,ATTEMPT/'step.json',experiment=sys.modules[__name__])
        if result['execution_status']=='failed':raise SystemExit(1)
        return
    load_plan(PLAN,args.plan_sha256)
    if ATTEMPT.exists():raise ValueError('bounded expansion attempt must be fresh')
    ATTEMPT.mkdir()
    resource=trial.previous.run_guarded_diagnostic([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
        '--child','--plan-sha256',args.plan_sha256],wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'],
        report_path=ATTEMPT/'step.resource.json',log_path=ATTEMPT/'step.log')
    child=ATTEMPT/'step.json';status=trial.shared._execution_status(resource,wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'])
    if not child.exists() or json.loads(child.read_text()).get('execution_status')!='completed':status='failed'
    result={'execution_status':status,'resource':resource,'child_sha256':trial.comparison._sha(child) if child.exists() else None}
    trial.tangent._write(ATTEMPT/'step.run.json',result);print(json.dumps(result))

if __name__=='__main__':main()
