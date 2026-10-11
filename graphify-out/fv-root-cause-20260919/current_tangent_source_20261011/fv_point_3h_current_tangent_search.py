"""One current-point face/event tangent comparison, with original full residual."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from advar import transport, variational
from examples.weather_scenarios import fv_point_3h_active_intersection_diagnostic as parent
from examples.weather_scenarios import fv_point_3h_current_event_tangent as directions
from examples.weather_scenarios import fv_point_3h_gn_finite_continuation as finite
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared
from examples.weather_scenarios import fv_point_3h_policy_search as search
from examples.weather_scenarios.fv_point_3h_policy_backend import FVBackend
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

shared=parent.shared; tangent=parent.tangent; geometry=parent.geometry
EVIDENCE=parent.EVIDENCE
PLAN=EVIDENCE/'CURRENT_TANGENT_SEARCH_PLAN_20261011.json'
ATTEMPT=EVIDENCE/'current_tangent_search_20261011_attempt1'
PARENT_PLAN_SHA='eb3d2a9bc5b458de864d28cdae23c0240189dcd9dae53766624e736792bf814a'
ARM_ORDER=('face_tangent_gradient','event_tangent_gradient')
OWN_SOURCES={'examples/weather_scenarios/fv_point_3h_current_tangent_search.py',
    'examples/weather_scenarios/fv_point_3h_current_event_tangent.py',
    'tests/test_fv_point_3h_current_event_tangent.py','tests/test_fv_point_3h_current_tangent_search.py'}
POLICY: dict[str,Any]={'guarded_launches':1,'internal_seconds':600.,'outer_seconds':660.,'rss_bytes':2*1024**3,
    'event_gradient_calls':2,'hvp_calls':4,'candidate_cap_per_direction':24,'optimizer_steps':1,
    'jacobian_row_vjp_calls':0,'dense_solves':0,'radius':.05,'face_scale':.84,
    'arms':list(ARM_ORDER),'single_eligible_direction_fallback_allowed':True,'candidate_policy':'dyadic_JR','future_score_observation_only':True,
    'candidate_branch_scope':'any_current_strict_native_pair; selector changes allowed and recorded',
    'search_domain':'selected_Q0_conditional_one_step','root_claim':False,'minimum_claim':False,
    'response_claim':False,'score_claim':False}


def load_plan(path: Path,digest: str) -> dict[str,Any]:
    if path.resolve()!=PLAN.resolve() or parent._sha(path)!=digest:
        raise ValueError('current tangent plan identity mismatch')
    old=parent.load_plan(parent.PLAN,PARENT_PLAN_SHA)
    plan=json.loads(path.read_text())
    extra={str(p.relative_to(ROOT)) for p in (parent.PLAN,ROOT/'docs/FV_INVERSE_PROBLEM.md')}
    if (plan.get('experiment_kind')!='current_point_two_tangent_direction_search'
            or plan.get('base_control_sha256')!=parent.BASE_SHA or plan.get('policy')!=POLICY
            or set(plan.get('source_files',{}))!=set(old['source_files'])|OWN_SOURCES
            or set(plan.get('archive_files',{}))!=set(old['archive_files'])|extra):
        raise ValueError('current tangent plan scope changed')
    for group in ('source_files','archive_files'):
        if any(plan[group].get(name)!=sha for name,sha in old[group].items()):
            raise ValueError('inherited source/evidence pin changed')
        for name,sha in plan[group].items():
            if parent._sha(ROOT/name)!=sha:
                raise ValueError(f'current tangent pin changed: {name}')
    return plan


def choose_candidate(arms: list[dict[str,Any]]) -> dict[str,Any]|None:
    eligible=[a for a in arms if isinstance(a.get('candidate'),dict)
        and a['candidate'].get('accepted') is True]
    return min(eligible,key=lambda a:(a['candidate']['F_squared'],a['candidate']['objective'],a['order'])) if eligible else None


def event_trace_binding(values: Tensor,trace: dict[str,Any]) -> bool:
    """Bind the differentiated event scalar to this point's recorded selector."""
    if values.shape!=(3,) or not bool(torch.isfinite(values).all()):
        return False
    budget=128*torch.finfo(values.dtype).eps*values[2]
    zeta=values[0]-values[1]
    choice=trace['choices'][179][0]
    return (bool((values[:2]<-budget).all() and zeta.abs()>budget)
        and choice['left_sign'][1][2]==-1 and choice['right_sign'][1][2]==-1
        and choice['slope_sign'][1][2]==-1 and choice['choose_left'][1][2]==bool(zeta>0))


def repeat_full_G(repeat: dict[str,Any],proposal_G: list[float],weights: Tensor) -> dict[str,Any]:
    control=torch.tensor(repeat['control'],dtype=torch.float64)
    minus=torch.tensor(repeat['side_gradients']['-1'],dtype=control.dtype)
    plus=torch.tensor(repeat['side_gradients']['1'],dtype=control.dtype)
    vartheta=float(repeat['theta'])
    G=torch.cat(((1-vartheta)*minus+vartheta*plus,(shared._face_value(control,weights)/.84).reshape(1)))
    match=finite._full_G_match(G,torch.tensor(proposal_G,dtype=G.dtype))
    merit=float(torch.dot(G,G))
    budget=128*torch.finfo(G.dtype).eps*max(abs(merit),abs(repeat['F_squared']),torch.finfo(G.dtype).tiny)
    return {'G':G.tolist(),'component_match':match,'merit_match':abs(merit-repeat['F_squared'])<=budget,
        'passed':match['passed'] and abs(merit-repeat['F_squared'])<=budget}


def persist_commit(output: Path,record: dict[str,Any],proposal: dict[str,Any],
        repeat: dict[str,Any],full_G: dict[str,Any]) -> dict[str,Any]:
    if (not tangent.fresh_final_closure(proposal,repeat) or full_G['passed'] is not True
            or proposal.get('J_armijo_passed') is not True or proposal.get('F_squared_armijo_passed') is not True
            or repeat['objective']>proposal['J_armijo_bound'] or repeat['F_squared']>proposal['F_squared_armijo_bound']):
        raise ValueError('selected candidate final original J/R/full-G/P2 approval failed')
    committed=copy.deepcopy(record)
    control=torch.tensor(repeat['control'],dtype=torch.float64);sha=tangent._tensor_sha(control)
    committed.update(current_control=repeat['control'],current_control_sha256=sha,current_theta=repeat['theta'],
        current_full_G=full_G['G'],last_confirmed_control=repeat['control'],last_confirmed_control_sha256=sha,
        last_confirmed_theta=repeat['theta'],last_confirmed_full_G=full_G['G'],
        candidate_committed=True,optimizer_steps_applied=1,canonical_proposal=proposal,
        canonical_final_repeat=repeat,final_full_G_check=full_G,postcommit_readiness_performed=False)
    tangent._write(output,committed)
    return committed


class CurrentBackend(FVBackend):
    """Reuse only evaluation/repeat/check methods; never invoke the fec9 builder."""
    def __init__(self,plan: dict[str,Any],path: Path,digest: str,deadline: float):  # pyright: ignore[reportMissingSuperCall]
        # The frozen constructor pins fec9; only its evaluator methods are reused.
        self.plan,self.plan_path,self.plan_sha=plan,path.resolve(),digest
        self.deadline=deadline;self.output_dir=ATTEMPT/'operators'
        self.build_count=self.row_count=self.solve_count=self.hvp_count=0
        self.timings={key:0. for key in ('identity_seconds','own_write_seconds','candidate_observe_seconds',
            'repeat_seconds','row_solve_seconds_inclusive','pure_hvp_seconds')}
        self.parent=parent.load_base()
        self.control=torch.tensor(self.parent['current_control'],dtype=torch.float64)
        if tangent._tensor_sha(self.control)!=parent.BASE_SHA:
            raise ValueError('wrong current canonical control')
        self.problem,self.original,_,self.parameters,self.truth,_=parent.seed._prepare_fixed_seed()
        self.input_anchor=shared._input_identity(self.problem,self.original,self.control,self.parameters,self.truth)
        self.runtime_anchor=shared._runtime();self.source_anchor=shared._source_hashes(plan,self.plan_path,digest)
        if self.input_anchor!=self.parent['input_after'] or self.runtime_anchor!=self.parent['runtime_after']:
            raise ValueError('current canonical fixed input/runtime changed')
        self.weights=geometry._face_weights(self.problem,axis='y',row=4,column=3)
        self._check(self.control)


def _run_child(path: Path,digest: str,output: Path) -> dict[str,Any]:
    start=time.monotonic();deadline=start+POLICY['internal_seconds']
    plan=load_plan(path,digest)
    record: dict[str,Any]={'phase':'running','execution_status':'running','policy':POLICY,'plan_sha256':digest,
        'physical_problem':'original_J_local_minimization','search_domain':POLICY['search_domain'],
        'base_control_sha256':parent.BASE_SHA,'candidate_committed':False,'optimizer_steps_applied':0,
        'event_grad_started':0,'event_grad_completed':0,'hvp_started':0,'hvp_completed':0,
        'candidate_evaluations_started':0,'candidate_evaluations_completed':0,'arms':[],
        'root_claim':False,'minimum_claim':False,'response_claim':False,'score_claim':False,
        'original_problem_solved':False}
    write=lambda:tangent._write(output,record)
    write();backend=CurrentBackend(plan,path,digest,deadline)
    c=backend.control;base=backend.parent;closure=base['canonical_final_repeat']
    record.update(current_control=c.tolist(),current_control_sha256=parent.BASE_SHA,current_theta=base['current_theta'],
        current_full_G=base['current_full_G'],last_confirmed_control=c.tolist(),last_confirmed_control_sha256=parent.BASE_SHA,
        last_confirmed_theta=base['current_theta'],last_confirmed_full_G=base['current_full_G'],
        source_before=backend.source_anchor,input_before=backend.input_anchor,runtime_before=backend.runtime_anchor)

    def finish(status: str,error: str='',failed: bool=False) -> dict[str,Any]:
        canonical=torch.tensor(record['current_control'],dtype=torch.float64)
        inp=shared._input_identity(backend.problem,backend.original,canonical,backend.parameters,backend.truth)
        source=shared._source_hashes(plan,path.resolve(),digest);runtime=shared._runtime()
        input_ok=geometry.model._fixed_input(inp,backend.input_anchor,tangent._tensor_sha(canonical))
        record.update(phase='finished',execution_status='failed' if failed else 'completed',numerical_status=status,
            termination_reason=error,source_after=source,source_unchanged=source==backend.source_anchor,
            input_after=inp,fixed_input_unchanged=input_ok,runtime_after=runtime,runtime_unchanged=runtime==backend.runtime_anchor,
            deadline_passed=time.monotonic()<deadline,elapsed_seconds=time.monotonic()-start,
            backend_timing=backend.timings)
        write()
        if not all((record['source_unchanged'],input_ok,record['runtime_unchanged'])):
            raise ValueError('source/input/runtime changed')
        return record

    try:
        backend._check(c);obs=tangent._observe(shared,backend.problem,c,backend.parameters,backend.weights)
        theta=float(tangent.minimum_mixture_weight(obs['side'][-1][1],obs['side'][1][1]))
        G=torch.cat(((1-theta)*obs['side'][-1][1]+theta*obs['side'][1][1],(obs['q']/.84).reshape(1)))
        if (not finite._full_G_match(G,torch.tensor(base['current_full_G'],dtype=c.dtype))['passed']
                or not shared._gradient_pair_match({s:obs['side'][s][1] for s in (-1,1)},
                    {s:torch.tensor(closure['side_gradients'][str(s)],dtype=c.dtype) for s in (-1,1)})
                or theta!=closure['theta'] or float(obs['native_j'])!=closure['objective']
                or not obs['pair']['passed'] or not obs['face_ok'] or not obs['side_objectives_match_native']
                or any(obs['traces'][s]['signature_sha256']!=closure['branch_trace'][str(s)]['signature_sha256'] for s in (-1,1))):
            raise ValueError('current canonical J/full-G/vartheta/side-gradient/trace closure failed')
        record.update(base_requalified=True,base_J=float(obs['native_j']),base_R=float(torch.dot(G,G)),base_theta=theta)
        contract=backend.problem.contract(backend.parameters);fv=contract.fv_transport
        if fv is None:
            raise ValueError('FV contract missing')
        direct=replace(contract,fv_transport=replace(fv,replay=False))
        trajectory=lambda point:variational.analysis_trajectory(point,direct)
        ev_grads=[];ev_values=[]
        for side in (-1,1):
            backend._check(c);record['event_grad_started']+=1;write()
            function=lambda point,s=side:parent.event.raw_zeta(parent.event.event_values(trajectory,point,s))
            gradient=torch.func.grad(function)(c)
            values=parent.event.event_values(trajectory,c,side)
            if (not torch.equal(values,parent.original_event_values(backend.problem,c,backend.parameters,side))
                    or not bool(torch.isfinite(gradient).all() and torch.isfinite(values).all())
                    or not bool((values[:2]<-128*torch.finfo(c.dtype).eps*values[2]).all())):
                raise ValueError('current event derivative/value original parity failed')
            if not event_trace_binding(values,obs['traces'][side]):
                raise ValueError('current event scalar is not bound to its recorded trace179 selector')
            ev_grads.append(gradient);ev_values.append(values);record['event_grad_completed']+=1
        if not torch.equal(ev_values[0],ev_values[1]):
            raise ValueError('current external extensions do not share one event value')
        record['event_value_pair_match_exact']=True
        record['current_event_trace_binding_passed']=True
        record['base_zeta']=float(ev_values[0][0]-ev_values[0][1])
        normal=geometry._face_normal(c,backend.weights);pivot=geometry._pivot(c,backend.weights)
        _,chart,_=geometry._chart_jacobian(c,backend.weights,pivot,0.)
        built=directions.current_event_tangent(obs['side'][-1][1],obs['side'][1][1],normal,
            ev_grads[0],ev_grads[1],c,chart,pivot)
        record.update(event_values=[v.tolist() for v in ev_values],event_gradients=[g.tolist() for g in ev_grads],
            direction_audit=built['audit'],direction_reference_control_sha256=parent.BASE_SHA,
            event_scope='trace179 at current Q0 point; not the previous diagnostic intersection',
            base_future_score_scope='matched-twin terminal +180min dBZ MSE; observation only')
        with torch.no_grad():
            base_score=float(backend.problem.score(c,backend.parameters))
        if not math.isfinite(base_score):
            raise ValueError('base future diagnostic score nonfinite')
        record['base_future_score']=base_score;write();backend._check(c)
        if built['audit']['face_decision']!='retain_for_bounded_test':
            return finish('native_face_release_required_no_Q0_commit')
        bundles={}
        for index,name in enumerate(ARM_ORDER):
            direction=built['baseline_direction'] if index==0 else built['direction']
            arm={'name':name,'order':index,'direction':direction.tolist(),'direction_sha256':tangent._tensor_sha(direction),
                'supported':False,'candidate':None,'trials':[]}
            record['arms'].append(arm);write()
            products={}
            grad_fn=torch.func.grad(backend.problem.objective,argnums=0)
            for side in (-1,1):
                backend._check(c)
                if record['hvp_started']>=POLICY['hvp_calls']:
                    raise ValueError('fresh HVP budget exhausted')
                record['hvp_started']+=1;write();then=time.perf_counter()
                with transport.selected_face_extension('y',4,3,side):
                    products[side]=torch.func.jvp(lambda point:grad_fn(point,backend.parameters),(c,),(direction,))[1]
                backend.timings['pure_hvp_seconds']+=time.perf_counter()-then
                if products[side].shape!=c.shape or not bool(torch.isfinite(products[side]).all()):
                    raise ValueError('current exact-direction HVP nonfinite or malformed')
                record['hvp_completed']+=1;backend.hvp_count+=1;write()
            model=prepared.mixing.minimum_tangent_model(obs['side'][-1][1],obs['side'][1][1],
                products[-1],products[1],normal,chart,theta,float(obs['q']),pivot,.84,direction_override=direction)
            if not finite._base_model_closure(model,G,theta)['passed'] or not torch.equal(model.direction,direction):
                raise ValueError('current direction/model/full-G connection failed')
            arm.update(side_HVP={str(s):products[s].tolist() for s in (-1,1)},
                residual=model.residual.tolist(),residual_direction=model.residual_direction.tolist(),
                merit_product=model.merit_product,model_gates=model.gates)
            supported=all(model.gates.get(k) is True for k in ('finite','both_side_gradients_descend','merit_descends','envelope_slope_matches_residual_dot'))
            arm['supported']=supported;write()
            if not supported:
                arm['status']='direction_model_refused';continue
            bundle={'observed':obs,'theta':theta,'G':G,'R':record['base_R'],'J':record['base_J'],
                'direction':direction,'model':model,'pivot':pivot,'cost_slopes':model.side_products}
            bundles[name]=bundle
            receipts={}
            def evaluate(candidate: Tensor,_theta: float,alpha: float) -> dict[str,Any]:
                record['candidate_evaluations_started']+=1;write()
                facts=backend.evaluate(bundle,candidate,alpha)
                record['candidate_evaluations_completed']+=1
                number=record['candidate_evaluations_completed']
                file=ATTEMPT/name/f'candidate_{number:02d}.json'
                tangent._write(file,facts);receipts[alpha]=facts
                compact={k:v for k,v in facts.items() if k!='branch_trace'}
                compact.update(candidate_receipt=str(file.relative_to(ROOT)),candidate_receipt_sha256=parent._sha(file))
                return compact
            def progress(trials: list[dict[str,Any]],meta: dict[str,Any]) -> None:
                arm['trials']=trials;arm['search']=meta;write()
            selected,trials,meta=search.search(model,c,evaluate,
                chart_candidate=lambda point,direction,alpha:backend.chart(bundle,point,direction,alpha),progress=progress,
                policy={'name':'dyadic_JR','max_evaluations':24,'max_backtracks':24,'radius':.05})
            arm.update(candidate=selected,trials=trials,search=meta,status='eligible_candidate' if selected else 'candidate_search_exhausted')
            if selected is not None:
                selected['branch_trace']=receipts[selected['alpha']]['branch_trace']
            write()
        winner=choose_candidate(record['arms'])
        record['comparison_complete']=sum(a['candidate'] is not None for a in record['arms'])==2
        record['supported_directions']=[a['name'] for a in record['arms'] if a['supported']]
        record['selection_basis']='two_eligible_candidate_comparison' if record['comparison_complete'] else 'single_eligible_direction_fallback' if winner is not None else 'no_eligible_direction'
        if winner is None:
            return finish('no_eligible_current_tangent_candidate')
        trial=winner['candidate'];backend._check(c)
        proposal,repeat=backend.repeat(bundles[winner['name']],trial)
        proposal.update(selected_arm=winner['name'],acceptance_policy='original_JR_conditional_Q0',
            search_domain=POLICY['search_domain'])
        check=repeat_full_G(repeat,trial['G'],backend.weights)
        record['selected_direction']=winner['name']
        record=persist_commit(output,record,proposal,repeat,check)
        with torch.no_grad():
            final_score=float(backend.problem.score(torch.tensor(repeat['control'],dtype=c.dtype),backend.parameters))
        record['final_future_score']=final_score if math.isfinite(final_score) else None
        record['future_score_computed']=math.isfinite(final_score)
        return finish('one_conditional_current_tangent_candidate_committed')
    except TimeoutError as error:
        return finish('internal_time_budget_after_confirmed_prefix',str(error))
    except Exception as error:
        finish('current_tangent_failed_after_confirmed_prefix',str(error),True)
        raise


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--child',action='store_true')
    args=parser.parse_args();output=ATTEMPT/'step.json'
    if args.child:
        _run_child(PLAN,args.plan_sha256,output);return
    load_plan(PLAN,args.plan_sha256)
    if ATTEMPT.exists():
        raise ValueError('current tangent attempt must be fresh')
    ATTEMPT.mkdir()
    resource=run_guarded_diagnostic([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child','--plan-sha256',args.plan_sha256],
        wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'],report_path=ATTEMPT/'step.resource.json',log_path=ATTEMPT/'step.log')
    result={'execution_status':shared._execution_status(resource,wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes']),
        'resource':resource,'child_sha256':parent._sha(output) if output.exists() else None}
    tangent._write(ATTEMPT/'step.run.json',result);print(json.dumps(result))


if __name__=='__main__':
    main()
