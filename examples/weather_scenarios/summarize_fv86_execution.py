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


def run():
    started=time.monotonic()
    fixture=load('fv_scaled_research_case')
    case=fixture.make_case()
    objective,_,_=fixture.functions(case)
    entries=[]
    raw=[]
    for mode in ('seed_a','seed_b'):
        data=json.loads((HERE/f'fv86_{mode}.json').read_text())
        resource=json.loads((HERE/f'fv86_{mode}.resource.json').read_text())
        raw.append(data)
        if (digest(case.parameters)!=data['input_identity']['parameters']
                or digest(case.verification)!=data['input_identity']['verification']
                or digest(case.definition)!=data['input_identity']['problem']):
            raise ValueError('postprocessing input differs from measured problem')
        workflow=data.get('workflow',{})
        response=workflow.get('response')
        seed=torch.tensor(data['seed'],dtype=case.parameters.dtype)
        status='resource_limited' if resource['resource_termination'] else data['status']
        entries.append({'mode':mode,'status':status,
            'failure_category':resource['resource_termination'] or data.get('failure_category'),
            'gn_reason':data.get('gn',{}).get('reason'),
            'reference_zero_control_objective':data.get('gn',{}).get('initial_objective'),
            'supplied_seed_objective':float(objective(seed,case.parameters)),
            'gn_gradient_max':data.get('gn',{}).get('gradient_max'),
            'final_gradient_max':workflow.get('after',{}).get('gradient_max'),
            'final_objective':workflow.get('after',{}).get('objective'),
            'adjoint_relative_residual':None if response is None else response['true_adjoint_relative_residual'],
            'refinement_iterations':data.get('refinement',{}).get('iterations'),
            'refinement_hvp_count':data.get('refinement',{}).get('hvp_count'),
            'timings':dict(gn_seconds=data['timings'].get('gn_seconds'),**workflow.get('timings',{})),
            'wall_seconds':resource['elapsed_seconds'],
            'sampled_peak_rss_bytes':resource['sampled_peak_rss_bytes']})
    if raw[0]['input_identity']!=raw[1]['input_identity'] or raw[0]['source_sha256']!=raw[1]['source_sha256']:
        raise ValueError('predeclared runs used different problems or sources')
    summary: dict[str, Any] = {'declared_case_count':2,'eligible_count':sum(r['status']=='eligible' for r in entries),
        'independently_validated_response_fraction':None,'nonlinear_reanalyses':0,
        'interpretation':'conditional discrete execution, not grid convergence or independent response accuracy',
        'cases':entries,'same_problem_and_source':True,
        'gn_start_policy':'supplied seeds are preserved as inputs; product zero-reference fallback and initial candidate exploration remain active; internal first iterate not captured',
        'postprocessing':'two objective-only FV evaluations at the supplied seeds, no optimizer rerun',
        'artifact_sha256':{},'source_sha256':{str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    for mode in ('preflight','seed_a','seed_b'):
        for suffix in ('.json','.resource.json','.log'):
            name=f'fv86_{mode}{suffix}'
            summary['artifact_sha256'][name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    for name in ('fv86_measured_producer.py.txt','fv86_measured_case.py.txt','fv86_preflight_runner.py.txt'):
        summary['artifact_sha256'][name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    if summary['eligible_count']==2:
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
    (HERE/'fv86_execution_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
