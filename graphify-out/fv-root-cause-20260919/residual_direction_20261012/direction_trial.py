"""One fresh2a4d gradient-versus-(gradient,residual-merit) comparison."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[3];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
import endpoint
from examples.weather_scenarios import fv_point_3h_netflux_subspace_trial as trial
from examples.weather_scenarios.fv_point_3h_residual_direction import build_residual_direction
PLAN,ATTEMPT=WORK/'PLAN.json',WORK/'attempt1';BASE_SHA=endpoint.FINAL_SHA;VERSION=trial.VERSION
RADIUS=3.751435583391765e-5
DIRECTION_ORDER=('baseline_tangent','residual_merit')
POLICY={**trial.POLICY,'radius':RADIUS,'jacobian_row_vjp_calls':0,'dense_solves':0,
    'direction_columns':list(DIRECTION_ORDER),'selection':'actual repeated R then J; both arms required; no E',
    'base_qualification':'PR303closed2a4d then freshnative/P2/d/s/HVP4',
    'budget_scope':'one600internal/660outer/2GiB includes all preparation; no newGNrows',
    'residual_direction_assumption':'conditional regular smooth common face; actualDG[s] gates required'}
CHANGED_SOURCES={'examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py','tests/test_fv_point_3h_netflux_comparison_endpoint.py'}
NEW_SOURCES={str((WORK/n).relative_to(ROOT)) for n in ('direction_trial.py','endpoint.py')}|{
 'examples/weather_scenarios/fv_point_3h_residual_direction.py','tests/test_fv_point_3h_residual_direction.py',
 'tests/test_fv_point_3h_current_progress_endpoint.py','tests/test_fv_point_3h_residual_direction_trial.py'}
RECEIPTS=endpoint.RECEIPTS


def load_plan(path:Path,digest:str)->dict[str,Any]:
    if path.resolve()!=PLAN.resolve() or trial.comparison._sha(path)!=digest:raise ValueError('residual direction plan identity mismatch')
    parent=endpoint.load_endpoint()['progress_plan'];plan=json.loads(path.read_text())
    trans={n:{'before':parent['source_files'][n],'after':trial.comparison._sha(ROOT/n)} for n in CHANGED_SOURCES}
    archives=set(parent['archive_files'])|{p.relative_to(ROOT).as_posix() for p in RECEIPTS}
    if (plan.get('experiment_kind')!='fresh2a4d_gradient_residual_direction_comparison' or plan.get('policy')!=POLICY
        or plan.get('base_control_sha256')!=BASE_SHA or plan.get('numerical_version')!=VERSION
        or plan.get('source_transition')!=trans or set(plan.get('source_files',{}))!=set(parent['source_files'])|NEW_SOURCES
        or set(plan.get('archive_files',{}))!=archives):raise ValueError('residual direction scope changed')
    for n,h in parent['source_files'].items():
        if n not in CHANGED_SOURCES and plan['source_files'].get(n)!=h:raise ValueError('unapproved inherited source change')
    if any(plan['archive_files'].get(n)!=h for n,h in parent['archive_files'].items()):raise ValueError('inherited archive changed')
    if any(trial.comparison._sha(ROOT/n)!=h for n,h in {**plan['source_files'],**plan['archive_files']}.items()):raise ValueError('source/archive pin mismatch')
    return plan


def direction_specs(context:dict[str,Any],c:torch.Tensor,theta:float,gm:torch.Tensor,gp:torch.Tensor,
        normal:torch.Tensor,chart:torch.Tensor,pivot:int)->list[dict[str,Any]]:
    _,_,d,_=trial.tangent.tangent_direction(gm,gp,normal,chart,theta,pivot=pivot)
    return [{'name':'baseline_tangent','direction':d}]


def next_direction(model:Any,gm:torch.Tensor,gp:torch.Tensor,normal:torch.Tensor,chart:torch.Tensor,pivot:int)->dict[str,Any]:
    def action(vector):
        if not torch.equal(vector,model.direction):raise ValueError('seed DG reused for a different direction')
        return model.residual_direction[:gm.numel()]
    s,info=build_residual_direction(action,normal,gm,gp,chart,pivot,normalize_to=model.direction,seed_direction=model.direction)
    if not all(info[k] for k in ('finite','both_side_gradients_descend','merit_descends')):raise ValueError('residual direction local support/descent refused')
    return {'name':'residual_merit','direction':s}


def choose_radius(G:torch.Tensor,D:torch.Tensor,V:torch.Tensor,record:dict[str,Any])->float:
    record['direction_relation_scope']='conditional smooth-face hypothesis; finite candidates reevaluated'
    return RADIUS


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args()
    if a.child:
        r=trial._run_child(PLAN,a.plan_sha256,ATTEMPT/'step.json',experiment=sys.modules[__name__])
        if r['execution_status']=='failed':raise SystemExit(1)
        return
    load_plan(PLAN,a.plan_sha256)
    if ATTEMPT.exists():raise ValueError('comparison attempt must be fresh')
    ATTEMPT.mkdir()
    resource=trial.previous.run_guarded_diagnostic([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child',
        '--plan-sha256',a.plan_sha256],wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'],
        report_path=ATTEMPT/'step.resource.json',log_path=ATTEMPT/'step.log')
    status=trial.shared._execution_status(resource,wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'])
    child=ATTEMPT/'step.json';r={'execution_status':status,'resource':resource,'child_sha256':trial.comparison._sha(child) if child.exists() else None}
    trial.tangent._write(ATTEMPT/'step.run.json',r);print(json.dumps(r))

if __name__=='__main__':main()
