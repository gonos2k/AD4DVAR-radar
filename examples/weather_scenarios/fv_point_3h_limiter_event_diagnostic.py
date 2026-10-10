"""Diagnose one observed minmod event at the closed PR279 point; never commit."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from advar import transport, variational
from examples.weather_scenarios import fv_point_3h_limiter_event_probe as event
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prior
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = prior.EVIDENCE
PLAN = EVIDENCE / 'LIMITER_EVENT_PLAN_ATTEMPT2_20261010.json'
ATTEMPT = EVIDENCE / 'limiter_event_20261010_attempt2'
SELF = 'examples/weather_scenarios/fv_point_3h_limiter_event_diagnostic.py'
HELPER = 'examples/weather_scenarios/fv_point_3h_limiter_event_probe.py'
TEST = 'tests/test_fv_point_3h_limiter_event_probe.py'
PRIOR_PLAN_SHA = '1392032f3f0b859f74f03551a98fa5465216cb7fd981250984b603a495526a1a'
BASE_SHA = '027e83cf77f7f9cf97c0c72da9f737b5627d312d6375b698bb5b7f7ae94c6769'
DIRECTION_SHA = 'ab3a2635cedc7f90c9cead68e7ce9bae9b33a15cabf5ffe4970a9b00c91fea74'
PARENT = EVIDENCE / 'gn_prepared_search_20261010_attempt1'
ARCHIVE = EVIDENCE / 'GN_PREPARED_SEARCH_ARCHIVE_20261010.json'
SAMPLE_ALPHAS = (1.742031084966799e-7, 6.968124339867196e-7, 1.3936248679734392e-6)
POLICY: dict[str, Any] = {'guarded_launches': 1, 'internal_seconds': 600.0, 'outer_seconds': 660.0,
    'rss_bytes': 2 * 1024**3, 'event_grad_calls': 2, 'event_jvp_calls': 2,
    'diagnostic_samples': 3, 'optimizer_steps': 0, 'hvp_calls': 0,
    'jacobian_row_vjp_calls': 0, 'dense_solves': 0,
    'root_claim': False, 'minimum_claim': False, 'response_claim': False, 'score_claim': False}


def _sha(path: Path) -> str:
    return prior._sha(path)


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError('event plan identity mismatch')
    old = prior._load_plan(prior.PLAN, PRIOR_PLAN_SHA)
    plan = json.loads(path.read_text())
    extra = [prior.PLAN, ARCHIVE, PARENT/'step.json.gz', PARENT/'step.run.json', PARENT/'step.resource.json']
    if (plan.get('experiment_kind') != 'current_limiter_event_diagnostic'
            or plan.get('policy') != POLICY or plan.get('event') != asdict(event.EVENT)
            or plan.get('base_control_sha256') != BASE_SHA or plan.get('direction_sha256') != DIRECTION_SHA
            or plan.get('sample_alphas') != list(SAMPLE_ALPHAS)
            or plan.get('predecessor_plan_sha256') != PRIOR_PLAN_SHA
            or set(plan.get('source_files', {})) != set(old['source_files']) | {SELF, HELPER, TEST}
            or set(plan.get('archive_files', {})) != set(old['archive_files']) | {x.relative_to(ROOT).as_posix() for x in extra}):
        raise ValueError('event plan policy or scope mismatch')
    for key in ('source_files', 'archive_files'):
        if any(plan[key].get(k) != v for k, v in old[key].items()):
            raise ValueError('inherited pin changed')
        for name, value in plan[key].items():
            if _sha(ROOT/name) != value:
                raise ValueError(f'event pin changed: {name}')
    return plan


def _load_base() -> dict[str, Any]:
    manifest = json.loads(ARCHIVE.read_text())
    blob = (PARENT/'step.json.gz').read_bytes()
    raw_bytes = gzip.decompress(blob)
    raw = json.loads(raw_bytes)
    run = json.loads((PARENT/'step.run.json').read_text())
    resource = json.loads((PARENT/'step.resource.json').read_text())
    if (hashlib.sha256(blob).hexdigest() != manifest['gzip_sha256']
            or hashlib.sha256(raw_bytes).hexdigest() != manifest['raw_sha256']
            or run['child_sha256'] != manifest['raw_sha256']
            or _sha(PARENT/'step.run.json') != manifest['run_sha256']
            or _sha(PARENT/'step.resource.json') != manifest['resource_sha256']
            or run['execution_status'] != 'completed' or resource['exit_code'] != 0
            or resource['resource_termination'] is not None
            or raw['current_control_sha256'] != BASE_SHA or raw['optimizer_steps_applied'] != 1
            or not raw['readiness_complete'] or not raw['candidate_committed']
            or raw['postcommit_gn_readiness']['direction_sha256'] != DIRECTION_SHA
            or not prior.tangent.fresh_final_closure(raw['proposal'], raw['final_repeat'])):
        raise ValueError('parent closed point invalid')
    for side in ('-1','1'):
        choice=raw['final_repeat']['branch_trace'][side]['choices'][124][0]
        if (choice['left_sign'][1][2] != -1 or choice['right_sign'][1][2] != -1
                or choice['slope_sign'][1][2] != -1 or choice['choose_left'][1][2] is not False):
            raise ValueError('saved limiter event selector/sign changed')
    return {key: raw[key] for key in ('current_control','current_theta','final_repeat',
        'postcommit_gn_readiness','input_after','runtime_after')}


def _plain_values(problem: Any, control: torch.Tensor, parameters: torch.Tensor, side: int) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    count = 0
    def observe(q: torch.Tensor, qx: torch.Tensor, qy: torch.Tensor) -> None:
        nonlocal count
        if count == event.EVENT.stage_index:
            row=event.EVENT.interior_row+1; column=event.EVENT.interior_column+1
            captured.append(torch.stack((q[row,column]-q[row,column-1],q[row,column+1]-q[row,column],q.abs().amax())))
        count += 1
    with torch.no_grad(), transport.selected_face_extension('y',4,3,side), transport.observe_minmod_stages(observe):
        problem.objective(control, parameters)
    if count != 360 or len(captured) != 1:
        raise ValueError('original event capture stage mismatch')
    return captured[0]


def _run_child(path: Path, digest: str, output: Path) -> None:
    start=time.monotonic(); deadline=start+POLICY['internal_seconds']
    plan=_load_plan(path,digest); path=path.resolve()
    source_before=prior.shared._source_hashes(plan,path,digest); base=_load_base()
    c=torch.tensor(base['current_control'],dtype=torch.float64)
    d=torch.tensor(base['postcommit_gn_readiness']['direction'],dtype=c.dtype)
    if prior.tangent._tensor_sha(c)!=BASE_SHA or prior.tangent._tensor_sha(d)!=DIRECTION_SHA:
        raise ValueError('current point/direction hash mismatch')
    record: dict[str, Any]={'phase':'running','policy':POLICY,'plan_sha256':digest,'base_control_sha256':BASE_SHA,
        'direction_sha256':DIRECTION_SHA,'event':asdict(event.EVENT),'optimizer_steps_applied':0,
        'event_grad_started':0,'event_grad_completed':0,'event_jvp_started':0,'event_jvp_completed':0,
        'current_control':c.tolist(),'current_theta':base['current_theta'],'event_derivatives':{},'samples':[]}
    write=lambda:prior.gni._atomic_write(output,record)
    check=lambda:prior.shared._deadline(deadline,POLICY['internal_seconds'])
    write();check()
    problem,original,_,parameters,truth,_=prior.seed._prepare_fixed_seed()
    inp=prior.shared._input_identity(problem,original,c,parameters,truth);runtime=prior.shared._runtime()
    if inp!=base['input_after'] or runtime!=base['runtime_after']:
        raise ValueError('original fixed input/runtime changed')
    weights=prior.geometry._face_weights(problem,axis='y',row=4,column=3)
    observed=prior.tangent._observe(prior.shared,problem,c,parameters,weights)
    closure=base['final_repeat']
    sides={s:observed['side'][s][1] for s in (-1,1)}
    refs={s:torch.tensor(closure['side_gradients'][str(s)],dtype=c.dtype) for s in (-1,1)}
    theta=float(prior.tangent.minimum_mixture_weight(sides[-1],sides[1]))
    G=torch.cat(((1-theta)*sides[-1]+theta*sides[1],(observed['q']/0.84).reshape(1)))
    if (not prior.shared._gradient_pair_match(sides,refs) or not observed['pair']['passed']
            or not observed['face_ok'] or not observed['side_objectives_match_native']
            or any(observed['traces'][s]['signature_sha256']!=closure['branch_trace'][str(s)]['signature_sha256'] for s in (-1,1))
            or not prior.local._scaled_error(G,torch.tensor(base['postcommit_gn_readiness']['residual'],dtype=c.dtype),G.abs())['passed']):
        raise ValueError('fresh base qualification failed')
    eps=128*torch.finfo(c.dtype).eps
    if abs(float(observed['native_j'])-closure['objective'])>eps*abs(closure['objective']) or abs(theta-closure['theta'])>eps*abs(theta):
        raise ValueError('fresh J/theta changed')
    contract=problem.contract(parameters); fv=contract.fv_transport
    if fv is None: raise ValueError('FV contract absent')
    direct=replace(contract,fv_transport=replace(fv,replay=False))
    trajectory=lambda point:variational.analysis_trajectory(point,direct)
    normal=prior.geometry._face_normal(c,weights); unit=normal/torch.linalg.vector_norm(normal)
    pivot=prior.geometry._pivot(c,weights)
    tangent_error=torch.dot(normal,d).abs()
    if not bool(torch.isfinite(d).all()) or float(tangent_error)>128*torch.finfo(c.dtype).eps*float(torch.linalg.vector_norm(normal)*torch.linalg.vector_norm(d)):
        raise ValueError('saved event direction is not tangent')
    record.update(source_before=source_before,input_before=inp,runtime_before=runtime,
        base_J=float(observed['native_j']),base_R=float(torch.dot(G,G)),base_theta=theta,
        original_replay=fv.replay,diagnostic_replay=False,
        differentiation_scope='same discrete trajectory with direct tape; original problem unchanged',
        archived_same_point_GN_direction_reused=True,archived_hvp_vectors_used=0,fresh_hvp_calls=0)
    for side in (-1,1):
        check(); plain=_plain_values(problem,c,parameters,side)
        function=lambda point,sign=side:event.event_values(trajectory,point,sign,event.EVENT,360)
        values=function(c)
        with torch.no_grad(), transport.selected_face_extension('y',4,3,side):
            original_frames=variational.analysis_trajectory(c,contract).frames_linear
            direct_frames=variational.analysis_trajectory(c,direct).frames_linear
        if not torch.equal(original_frames,direct_frames):
            raise ValueError('direct/original analysis frames differ')
        del original_frames,direct_frames
        if not torch.equal(values,plain): raise ValueError('direct/original event values differ')
        if not bool(torch.isfinite(values).all()) or not bool((values[:2]<0).all()) or not bool(values[0]<values[1]):
            raise ValueError('expected negative-slope right-selected event unavailable')
        record['event_jvp_started']+=1;write();check()
        jvp_result=torch.func.jvp(function,(c,),(d,))
        jvp_values,jvp=jvp_result[0],jvp_result[1]
        if not bool(torch.isfinite(jvp_values).all() & torch.isfinite(jvp).all()):
            raise ValueError('event JVP contains nonfinite values')
        record['event_jvp_completed']+=1
        record['event_grad_started']+=1;write();check()
        grad=torch.func.grad(lambda point:event.raw_zeta(function(point)))(c)
        record['event_grad_completed']+=1;check()
        slope=jvp[0]-jvp[1];dot=torch.dot(grad,d)
        if not prior.local._scaled_error(dot.reshape(1),slope.reshape(1),(grad.abs()*d.abs()).sum().reshape(1))['passed']:
            raise ValueError('event JVP and gradient product disagree')
        projected=grad-unit*torch.dot(unit,grad)
        projected_dot=torch.dot(projected,d)
        if not prior.local._scaled_error(projected_dot.reshape(1),slope.reshape(1),(projected.abs()*d.abs()).sum().reshape(1))['passed']:
            raise ValueError('projected event gradient and JVP disagree')
        zeta=values[0]-values[1]
        record['event_derivatives'][str(side)]={'values':values.tolist(),'original_values':plain.tolist(),
            'value_parity_exact':True,'analysis_frames_parity_exact':True,'jvp_values':jvp_values.tolist(),'jvp':jvp.tolist(),
            'zeta':float(zeta),'gradient':grad.tolist(),'tangent_gradient':projected.tolist(),
            'zeta_directional_derivative':float(slope),'gradient_dot_direction':float(dot),
            'positive_inside_gap':float(-zeta),'oriented_directional_derivative':float(-slope),
            'linear_event_alpha':float(-zeta/slope) if float(slope)>0 else None,
            'gradient_dot_direction_passed':True,'projected_gradient_dot_direction':float(projected_dot),
            'projected_gradient_dot_direction_passed':True,'unit_normal_dot_direction':float(torch.dot(unit,d)),
            'event_alpha_scope':'local first-order estimate, not path or safe interval certification','gradient_normal_component':float(torch.dot(unit,grad))}
        write()
    for alpha in SAMPLE_ALPHAS:
        check();candidate=prior.local._chart_candidate(c,weights,d,pivot,alpha)
        sample: dict[str, Any]={'alpha':alpha,'control_sha256':prior.tangent._tensor_sha(candidate),
            'actual_move':float(torch.linalg.vector_norm(candidate-c)),'events':{}}
        if not 0 < sample['actual_move'] <= 0.05 or not bool(torch.isfinite(candidate).all()):
            raise ValueError('diagnostic chart movement outside declared radius')
        trial=prior.tangent._observe(prior.shared,problem,candidate,parameters,weights)
        candidate_theta=float(prior.tangent.minimum_mixture_weight(trial['side'][-1][1],trial['side'][1][1]))
        candidate_G=torch.cat(((1-candidate_theta)*trial['side'][-1][1]+candidate_theta*trial['side'][1][1],(trial['q']/0.84).reshape(1)))
        sample.update(J=float(trial['native_j']),R=float(torch.dot(candidate_G,candidate_G)),theta=candidate_theta,
            branch_pair_passed=trial['pair']['passed'],face_passed=trial['face_ok'],
            side_objectives_match_native=trial['side_objectives_match_native'],
            trace_matches_base=all(trial['traces'][s]['signature_sha256']==observed['traces'][s]['signature_sha256'] for s in (-1,1)))
        for side in (-1,1):
            vals=_plain_values(problem,candidate,parameters,side); direct_vals=event.event_values(trajectory,candidate,side,event.EVENT,360)
            if not bool(torch.isfinite(vals).all() & torch.isfinite(direct_vals).all()):
                raise ValueError('sample event values nonfinite')
            if not torch.equal(vals,direct_vals):raise ValueError('sample direct/original event values differ')
            z=float(vals[0]-vals[1]);info=record['event_derivatives'][str(side)]
            sample['events'][str(side)]={'values':vals.tolist(),'zeta':z,'linear_zeta':info['zeta']+alpha*info['zeta_directional_derivative'],
                'choose_left':trial['traces'][side]['choices'][124][0]['choose_left'][1][2],
                'same_negative_sign':bool((vals[:2]<0).all()),'value_parity_exact':True}
        record['samples'].append(sample);write();check()
    after=prior.shared._input_identity(problem,original,c,parameters,truth)
    source_after=prior.shared._source_hashes(plan,path,digest);runtime_after=prior.shared._runtime()
    if inp!=after or source_before!=source_after or runtime!=runtime_after:raise ValueError('original source/input/runtime changed')
    record.update(phase='finished',execution_status='completed',numerical_status='limiter_event_diagnostic_complete_no_commit',
        source_after=source_after,input_after=after,runtime_after=runtime_after,source_unchanged=True,fixed_input_unchanged=True,
        runtime_unchanged=True,deadline_passed=time.monotonic()<deadline,elapsed_seconds=time.monotonic()-start,
        root_claim=False,minimum_claim=False,response_claim=False,score_claim=False)
    write()


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--child',action='store_true')
    args=parser.parse_args();output=ATTEMPT/'diagnostic.json'
    if args.child:_run_child(PLAN,args.plan_sha256,output);return
    _load_plan(PLAN,args.plan_sha256)
    if ATTEMPT.exists():raise ValueError('event attempt must be fresh')
    ATTEMPT.mkdir()
    resource=run_guarded_diagnostic([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child','--plan-sha256',args.plan_sha256],
        wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'],report_path=ATTEMPT/'diagnostic.resource.json',log_path=ATTEMPT/'diagnostic.log')
    result={'execution_status':prior.shared._execution_status(resource,wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes']),
        'resource':resource,'child_sha256':_sha(output) if output.exists() else None}
    prior.gni._atomic_write(ATTEMPT/'diagnostic.run.json',result);print(json.dumps(result))


if __name__=='__main__':main()
