"""Status accounting for the archived FV86 runs; no scaled inverse solve."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/weather_scenarios/summarize_fv86_execution.py'


def load_summary():
    spec=importlib.util.spec_from_file_location('summarize_fv86_execution_status',SCRIPT)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(('data','resource','expected'),[
    ({'status':'eligible','phase':'finished','source_unchanged':True},{'exit_code':0},'completed'),
    ({'status':'eligible','phase':'finished','source_unchanged':True},{'exit_code':1},'failed'),
    ({'status':'eligible','phase':'finished','source_unchanged':True},{'exit_code':-9},'failed'),
    ({'status':'eligible','phase':'running','source_unchanged':True},{'exit_code':0},'failed'),
    ({'status':'eligible','phase':'finished'},{'exit_code':0},'failed'),
    ({'status':'eligible','phase':'finished','source_unchanged':False},{'exit_code':0},'failed'),
    ({'status':'execution_error','phase':'finished','source_unchanged':True},{'exit_code':0},'failed'),
    ({'status':'eligible','phase':'finished','source_unchanged':True},
     {'exit_code':-15,'resource_termination':'wall_time_limit'},'resource_limited'),
    ({'status':'eligible','phase':'finished','source_unchanged':True},
     {'exit_code':-15,'resource_termination':'rss_limit'},'resource_limited'),
    ({'status':'eligible','phase':'finished','source_unchanged':True},
     {'exit_code':0,'resource_termination':'unexpected_termination'},'failed'),
    ({'status':'eligible','phase':'finished','source_unchanged':True},
     {'exit_code':1,'resource_termination':'resource_monitor_error','monitor_error':'ps failed'},'failed'),
])
def test_execution_status_requires_successful_completed_unchanged_run(data,resource,expected):
    module=load_summary()
    assert module.execution_status(data,resource)==expected
    expected_numerical='eligible' if data.get('status')=='eligible' else 'not_reached'
    assert module.numerical_status(data)==expected_numerical


def test_summary_counts_only_completed_numerically_eligible_entries():
    module=load_summary()
    entries=[
        {'execution_status':'completed','numerical_status':'eligible'},
        {'execution_status':'failed','numerical_status':'eligible'},
    ]
    assert module.status_summary(entries)=={
        'declared_case_count':2,'execution_completed_count':1,
        'numerical_eligible_count':2,'eligible_count':1}


def test_numerical_status_uses_workflow_before_execution_error():
    module=load_summary()
    assert module.numerical_status({'status':'execution_error'})=='not_reached'
    assert module.numerical_status({'status':'execution_error',
                                    'workflow':{'status':'eligible'}})=='eligible'
    assert module.numerical_status({'status':'eligible',
                                    'workflow':{'status':'ineligible'}})=='ineligible'


def test_run_integrates_factory_and_writes_new_summary_path(tmp_path,monkeypatch):
    module=load_summary()
    evidence=tmp_path/'evidence'
    evidence.mkdir()
    monkeypatch.setattr(module,'HERE',evidence)

    case=SimpleNamespace(parameters=torch.tensor([1.0]),verification={'v':1},definition={'p':'case'})
    fixture=SimpleNamespace(make_case=lambda:case,
                            functions=lambda _case:(lambda control,_parameters:control.sum(),None,None))
    monkeypatch.setattr(module,'load',lambda _name:fixture)
    identities={'parameters':module.digest(case.parameters),
                'verification':module.digest(case.verification),
                'problem':module.digest(case.definition)}
    for mode,exit_code in (('seed_a',0),('seed_b',1)):
        data={'status':'eligible','phase':'finished','source_unchanged':True,
              'input_identity':identities,'source_sha256':{'same':'hash'},'seed':[float(exit_code+1)],
              'workflow':{'status':'eligible','timings':{}},'timings':{},'gn':{},'refinement':{}}
        (evidence/f'fv86_{mode}.json').write_text(json.dumps(data))
        resource={'exit_code':exit_code,'resource_termination':None,'monitor_error':None,
                  'elapsed_seconds':1.0,'sampled_peak_rss_bytes':100}
        (evidence/f'fv86_{mode}.resource.json').write_text(json.dumps(resource))
    for suffix in ('.json','.resource.json','.log'):
        (evidence/f'fv86_preflight{suffix}').write_text('{}')
    for mode in ('seed_a','seed_b'):
        (evidence/f'fv86_{mode}.log').write_text('{}')
    for name in ('fv86_measured_producer.py.txt','fv86_measured_case.py.txt','fv86_preflight_runner.py.txt'):
        (evidence/name).write_text('fixture')

    target=tmp_path/'status.json'
    module.run(target)
    summary=json.loads(target.read_text())
    assert summary['declared_case_count']==2
    assert summary['numerical_eligible_count']==2
    assert summary['eligible_count']==1
    assert summary['postprocessing_objective_evaluation_count']==1
    assert [case['execution_status'] for case in summary['cases']]==['completed','failed']
    assert [case['numerical_status'] for case in summary['cases']]==['eligible','eligible']
    assert [case['status'] for case in summary['cases']]==['eligible','failed']
    assert all(case['response_validation']=='not_performed' for case in summary['cases'])
    assert [(case['exit_code'],case['phase'],case['source_unchanged'])
            for case in summary['cases']]==[(0,'finished',True),(1,'finished',True)]
    assert not (evidence/'fv86_execution_summary.json').exists()
    module.run()
    assert (evidence/'fv86_execution_status_summary.json').exists()
    assert not (evidence/'fv86_execution_summary.json').exists()


def test_run_summarizes_resource_cutoffs_before_identity_or_seed(tmp_path,monkeypatch):
    module=load_summary()
    evidence=tmp_path/'evidence'
    evidence.mkdir()
    monkeypatch.setattr(module,'HERE',evidence)

    case=SimpleNamespace(parameters=torch.tensor([1.0]),verification={'v':1},definition={'p':'case'})
    objective_calls=[]
    def make_objective(_case):
        def objective(control,_parameters):
            objective_calls.append(control)
            return control.sum()
        return objective,None,None
    fixture=SimpleNamespace(make_case=lambda:case,functions=make_objective)
    monkeypatch.setattr(module,'load',lambda _name:fixture)
    for mode,termination in (('seed_a','wall_time_limit'),('seed_b','rss_limit')):
        data={'status':'running','phase':'prepare','source_unchanged':False,'timings':{}}
        (evidence/f'fv86_{mode}.json').write_text(json.dumps(data))
        resource={'exit_code':-15,'resource_termination':termination,'monitor_error':None,
                  'elapsed_seconds':10.0,'sampled_peak_rss_bytes':100}
        (evidence/f'fv86_{mode}.resource.json').write_text(json.dumps(resource))
        (evidence/f'fv86_{mode}.log').write_text('')
    for suffix in ('.json','.resource.json','.log'):
        (evidence/f'fv86_preflight{suffix}').write_text('{}')
    for name in ('fv86_measured_producer.py.txt','fv86_measured_case.py.txt','fv86_preflight_runner.py.txt'):
        (evidence/name).write_text('fixture')

    target=tmp_path/'early-status.json'
    module.run(target)
    summary=json.loads(target.read_text())
    assert summary['declared_case_count']==2
    assert summary['eligible_count']==0
    assert summary['numerical_eligible_count']==0
    assert summary['postprocessing_objective_evaluation_count']==0
    assert summary['same_problem_and_source'] is None
    assert [entry['execution_status'] for entry in summary['cases']]==['resource_limited']*2
    assert [entry['numerical_status'] for entry in summary['cases']]==['not_reached']*2
    assert [entry['supplied_seed_objective'] for entry in summary['cases']]==[None,None]
    assert objective_calls==[]
