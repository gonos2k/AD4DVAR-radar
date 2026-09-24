"""Summarize both predeclared runs, retaining refusals and original artifacts."""
import hashlib
import json
from pathlib import Path
import time
import sys
from typing import Any

import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.weather_scenarios.fv86_execution_probe import digest, load

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'graphify-out/fv-root-cause-20260919'


def execution_status(data: dict[str, Any], resource: dict[str, Any]) -> str:
    """Classify process completion separately from numerical outcome."""
    termination=resource.get('resource_termination')
    if resource.get('monitor_error') is not None:
        return 'failed'
    if termination in ('wall_time_limit','rss_limit'):
        return 'resource_limited'
    if termination is not None:
        return 'failed'
    exit_code=resource.get('exit_code')
    if (type(exit_code) is int and exit_code==0 and data.get('phase')=='finished'
            and data.get('source_unchanged') is True and data.get('status')!='execution_error'):
        return 'completed'
    return 'failed'


def numerical_status(data: dict[str, Any]) -> str:
    workflow=data.get('workflow')
    raw=workflow.get('status') if isinstance(workflow,dict) and 'status' in workflow else data.get('status')
    if raw in ('eligible','ineligible'):
        return raw
    return 'not_reached'


def status_summary(entries: list[dict[str, Any]]) -> dict[str, int]:
    completed=sum(entry['execution_status']=='completed' for entry in entries)
    numerical_eligible=sum(entry['numerical_status']=='eligible' for entry in entries)
    eligible=sum(entry['execution_status']=='completed' and entry['numerical_status']=='eligible'
                 for entry in entries)
    return {'declared_case_count':2,'execution_completed_count':completed,
            'numerical_eligible_count':numerical_eligible,'eligible_count':eligible}


def same_problem_and_source(raw: list[dict[str, Any]]) -> bool | None:
    if len(raw)!=2:
        return None
    first_identity=raw[0].get('input_identity')
    second_identity=raw[1].get('input_identity')
    first_source=raw[0].get('source_sha256')
    second_source=raw[1].get('source_sha256')
    if isinstance(first_identity,dict) and isinstance(second_identity,dict):
        common=first_identity.keys() & second_identity.keys()
        if any(first_identity[key]!=second_identity[key] for key in common):
            return False
    if isinstance(first_source,dict) and isinstance(second_source,dict):
        common=first_source.keys() & second_source.keys()
        if any(first_source[key]!=second_source[key] for key in common):
            return False
    required_identity={'parameters','verification','problem'}
    if (isinstance(first_identity,dict) and isinstance(second_identity,dict)
            and required_identity.issubset(first_identity)
            and required_identity.issubset(second_identity)
            and isinstance(first_source,dict) and first_source
            and isinstance(second_source,dict) and second_source):
        return first_identity==second_identity and first_source==second_source
    return None


def run(output_path: Path | None = None):
    started=time.monotonic()
    fixture=load('fv_scaled_research_case')
    case=fixture.make_case()
    objective=None
    objective_evaluation_count=0
    entries=[]
    raw=[]
    for mode in ('seed_a','seed_b'):
        data=json.loads((HERE/f'fv86_{mode}.json').read_text())
        resource=json.loads((HERE/f'fv86_{mode}.resource.json').read_text())
        raw.append(data)
        workflow=data.get('workflow') or {}
        if not isinstance(workflow,dict):
            workflow={}
        exec_status=execution_status(data,resource)
        num_status=numerical_status(data)
        identity=data.get('input_identity')
        required_identity=('parameters','verification','problem')
        if exec_status=='completed':
            if not isinstance(identity,dict) or any(key not in identity for key in required_identity):
                raise ValueError('completed run lacks measured problem identity')
            if (digest(case.parameters)!=identity['parameters']
                    or digest(case.verification)!=identity['verification']
                    or digest(case.definition)!=identity['problem']):
                raise ValueError('postprocessing input differs from measured problem')
        supplied_seed_objective=None
        seed_values=data.get('seed')
        if exec_status=='completed' and seed_values is not None:
            if objective is None:
                objective,_,_=fixture.functions(case)
            seed=torch.tensor(seed_values,dtype=case.parameters.dtype)
            supplied_seed_objective=float(objective(seed,case.parameters))
            objective_evaluation_count+=1
        response=workflow.get('response')
        legacy_status=('resource_limited' if exec_status=='resource_limited' else
                       'failed' if exec_status=='failed' else
                       'eligible' if num_status=='eligible' else 'ineligible')
        gn=data.get('gn') or {}
        refinement=data.get('refinement') or {}
        timings=data.get('timings') or {}
        entries.append({'mode':mode,'status':legacy_status,
            'execution_status':exec_status,'numerical_status':num_status,
            'response_validation':'not_performed',
            'exit_code':resource.get('exit_code'),'phase':data.get('phase'),
            'source_unchanged':data.get('source_unchanged'),
            'failure_category':resource.get('resource_termination') or data.get('failure_category'),
            'gn_reason':gn.get('reason'),
            'reference_zero_control_objective':gn.get('initial_objective'),
            'supplied_seed_objective':supplied_seed_objective,
            'gn_gradient_max':gn.get('gradient_max'),
            'final_gradient_max':workflow.get('after',{}).get('gradient_max'),
            'final_objective':workflow.get('after',{}).get('objective'),
            'adjoint_relative_residual':None if response is None else response['true_adjoint_relative_residual'],
            'refinement_iterations':refinement.get('iterations'),
            'refinement_hvp_count':refinement.get('hvp_count'),
            'timings':dict(gn_seconds=timings.get('gn_seconds'),**(workflow.get('timings') or {})),
            'wall_seconds':resource.get('elapsed_seconds'),
            'sampled_peak_rss_bytes':resource.get('sampled_peak_rss_bytes')})
    problem_source_match=same_problem_and_source(raw)
    summary: dict[str, Any] = {**status_summary(entries),
        'independently_validated_response_fraction':None,'nonlinear_reanalyses':0,
        'interpretation':'conditional discrete execution, not grid convergence or independent response accuracy',
        'cases':entries,'same_problem_and_source':problem_source_match,
        'gn_start_policy':'supplied seeds are preserved as inputs; product zero-reference fallback and initial candidate exploration remain active; internal first iterate not captured',
        'postprocessing_objective_evaluation_count':objective_evaluation_count,
        'postprocessing':f'{objective_evaluation_count} objective-only FV evaluations at available completed-run seeds, no optimizer rerun',
        'artifact_sha256':{},'source_sha256':{
            str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT/'src/advar/fv_research_problem.py',
                         ROOT/'examples/weather_scenarios/fv_scaled_research_case.py',
                         ROOT/'src/advar/variational.py', ROOT/'src/advar/transport.py',
                         ROOT/'src/advar/physics.py')}}
    for mode in ('preflight','seed_a','seed_b'):
        for suffix in ('.json','.resource.json','.log'):
            name=f'fv86_{mode}{suffix}'
            summary['artifact_sha256'][name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    for name in ('fv86_measured_producer.py.txt','fv86_measured_case.py.txt','fv86_preflight_runner.py.txt'):
        summary['artifact_sha256'][name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    if summary['eligible_count']==2 and problem_source_match is True:
        a,b=raw
        ac=torch.tensor(a['workflow']['control'],dtype=torch.float64)
        bc=torch.tensor(b['workflow']['control'],dtype=torch.float64)
        ag=torch.tensor(a['workflow']['response']['total_gradient'],dtype=torch.float64)
        bg=torch.tensor(b['workflow']['response']['total_gradient'],dtype=torch.float64)
        summary['between_starts']={'refined_control_l2':float((ac-bc).norm()),
            'total_gradient_relative_difference':float((ag-bg).norm()/ag.norm()),
            'score_absolute_difference':abs(a['final_score']-b['final_score']),
            'same_selectors':a['nominal_branch']['choices']==b['nominal_branch']['choices'],
            'same_face_signs':a['nominal_branch']['face_signs']==b['nominal_branch']['face_signs']}
    summary['postprocessing_seconds']=time.monotonic()-started
    target=Path(output_path) if output_path is not None else HERE/'fv86_execution_status_summary.json'
    target.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
