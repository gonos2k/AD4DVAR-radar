"""Observation-only b647/d19e scores, separately guarded from candidate work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.weather_scenarios import fv_point_3h_netflux_future_observation as observer
from examples.weather_scenarios import fv_point_3h_netflux_comparison_endpoint as endpoint
from examples.weather_scenarios import fv_point_3h_netflux_direction_comparison as comparison

WORK = Path(__file__).resolve().parent
PLAN = WORK / 'OBSERVATION_PLAN.json'
ATTEMPT = WORK / 'observation1'
POLICY = observer.POLICY


def load_plan(digest):
    if observer._sha_file(PLAN) != digest:
        raise ValueError('observation plan identity mismatch')
    plan = json.loads(PLAN.read_text())
    parent = endpoint.load_endpoint()['parent_plan']
    expected_sources = set(parent['source_files']) | {
        endpoint.SELF, endpoint.TEST, Path(__file__).relative_to(ROOT).as_posix(),
        observer.SELF, observer.TEST}
    expected_archives = set(parent['archive_files']) | {p.relative_to(ROOT).as_posix() for p in endpoint.RECEIPTS}
    if (plan.get('experiment_kind') != 'observation_only_b647_d19e'
            or plan.get('policy') != POLICY or plan.get('numerical_version') != comparison.VERSION
            or plan.get('start_control_sha256') != comparison.BASE_SHA
            or plan.get('final_control_sha256') != endpoint.BASE_SHA
            or plan.get('parameters_sha256') != observer.PARAMETERS_SHA
            or plan.get('truth_sha256') != observer.TRUTH_SHA
            or plan.get('score_profile') != observer.SCORE_PROFILE
            or set(plan.get('source_files',{})) != expected_sources
            or set(plan.get('archive_files',{})) != expected_archives):
        raise ValueError('observation policy/input/source scope mismatch')
    for group in ('source_files','archive_files'):
        if any(plan[group].get(n) != h for n,h in parent[group].items()):
            raise ValueError('observation changed inherited pin')
    if any(observer._sha_file(ROOT/n) != h for n,h in {**plan['source_files'],**plan['archive_files']}.items()):
        raise ValueError('observation source/archive changed')
    return plan


def run_child(digest):
    started = time.monotonic(); deadline = started + POLICY['internal_seconds']
    plan = load_plan(digest)
    first, final = comparison.endpoint.load_endpoint(), endpoint.load_endpoint()
    case = observer._load_case()
    problem, original, parameters, truth = observer._fixed_case_tensors(case, plan)
    shared = comparison.shared
    before = observer._source_hashes(plan,PLAN,digest)
    runtime = shared._runtime()
    input_before = shared._input_identity(problem,original,final['control'],parameters,truth)
    if input_before != final['input_after'] or runtime != final['runtime_after']:
        raise ValueError('observation runtime/input differs from d19e receipt')
    record = {'phase':'running','execution_status':'running','plan_sha256':digest,
        'numerical_version':comparison.VERSION,'base_control_sha256':endpoint.BASE_SHA,
        'transport_sha256':before[observer.TRANSPORT],
        'fixed_case_sha256':before[observer.CASE.relative_to(ROOT).as_posix()],
        'fixed_case_manifest_sha256':before[observer.CASE_MANIFEST.relative_to(ROOT).as_posix()],
        'score_evaluations_started':0,'score_evaluations_completed':0,'optimizer_steps':0,
        'candidate_evaluations':0,'selection_use':False,'root_claim':False,'minimum_claim':False,
        'response_claim':False,'score_claim':False,'source_before':before,'input_before':input_before,
        'runtime_before':runtime,'score_history':[]}
    def write():
        comparison.tangent._write(ATTEMPT/'step.json',record)
    def check(stage):
        if time.monotonic() >= deadline:
            raise TimeoutError('observation global deadline exceeded')
        if (observer._source_hashes(plan,PLAN,digest) != before or shared._runtime() != runtime
                or shared._input_identity(problem,original,final['control'],parameters,truth) != input_before):
            raise ValueError(f'observation identity changed at {stage}')
    def progress(label,stage,value):
        key='score_evaluations_started' if stage=='started' else 'score_evaluations_completed'
        record[key] += 1
        record['score_history'].append({'point':label,'stage':stage,'E':value});write()
    write()
    try:
        record['score_pair'] = observer.observe_score_pair(problem,parameters,truth,first['control'],final['control'],
            check=check,progress=progress)
        check('final')
        after = observer._source_hashes(plan,PLAN,digest)
        final_input = shared._input_identity(problem,original,final['control'],parameters,truth)
        final_runtime = shared._runtime()
        record.update(phase='finished',execution_status='completed',numerical_status='paired_terminal_score_observed',
            source_after=after,source_unchanged=after == before,
            fixed_case_unchanged=all(after[p.relative_to(ROOT).as_posix()] == before[p.relative_to(ROOT).as_posix()]
                for p in (observer.CASE,observer.CASE_MANIFEST)),
            input_after=final_input,fixed_input_unchanged=final_input == input_before,
            runtime_after=final_runtime,runtime_unchanged=final_runtime == runtime,
            elapsed_seconds=time.monotonic()-started,deadline_passed=time.monotonic()<deadline)
    except Exception as error:
        record.update(phase='finished',execution_status='failed',termination_reason=f'{type(error).__name__}: {error}')
        write();raise
    write()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    if args.child:
        run_child(args.plan_sha256);return
    load_plan(args.plan_sha256)
    if ATTEMPT.exists():raise ValueError('observation attempt must be fresh')
    ATTEMPT.mkdir()
    resource=comparison.previous.run_guarded_diagnostic([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
        '--child','--plan-sha256',args.plan_sha256],wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'],
        report_path=ATTEMPT/'step.resource.json',log_path=ATTEMPT/'step.log')
    status=comparison.shared._execution_status(resource,wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'])
    child=ATTEMPT/'step.json'
    if not child.exists() or json.loads(child.read_text()).get('execution_status') != 'completed': status='failed'
    result={'execution_status':status,'resource':resource,'child_sha256':observer._sha_file(child) if child.exists() else None}
    comparison.tangent._write(ATTEMPT/'step.run.json',result);print(json.dumps(result))

if __name__ == '__main__':main()
