"""Separate terminal score observation for7839 versus closed2a4d; no optimizer."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
import endpoint
from examples.weather_scenarios import fv_point_3h_netflux_future_observation as observer
from examples.weather_scenarios import fv_point_3h_netflux_subspace_trial as trial
PLAN,ATTEMPT=WORK/'OBSERVATION_PLAN.json',WORK/'observation1'
POLICY=observer.POLICY


def load_plan(digest):
 if observer._sha_file(PLAN)!=digest:raise ValueError('observation plan hash mismatch')
 plan=json.loads(PLAN.read_text());base=endpoint.load_endpoint();parent=base['progress_plan']
 new={str((WORK/n).relative_to(ROOT)) for n in ('observe_current.py','endpoint.py')}|{observer.SELF,observer.TEST,'tests/test_fv_point_3h_current_progress_endpoint.py'}
 if (plan.get('experiment_kind')!='paired7839_2a4d_terminal_E_only' or plan.get('policy')!=POLICY
     or set(plan['source_files'])!=set(parent['source_files'])|new
     or set(plan['archive_files'])!=set(parent['archive_files'])|{p.relative_to(ROOT).as_posix() for p in endpoint.RECEIPTS}):raise ValueError('observation scope changed')
 changes={'tests/test_fv_point_3h_netflux_comparison_endpoint.py','examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py'}
 if any(plan['source_files'][n]!=h for n,h in parent['source_files'].items() if n not in changes):raise ValueError('numerical score source changed')
 if any(observer._sha_file(ROOT/n)!=h for n,h in {**plan['source_files'],**plan['archive_files']}.items()):raise ValueError('observation pins changed')
 return plan


def child(digest):
 started=time.monotonic();deadline=started+POLICY['internal_seconds'];plan=load_plan(digest);base=endpoint.load_endpoint()
 members=endpoint._tar_files(endpoint._read(endpoint.CLOSED_ENDPOINT_TAR,endpoint.CLOSED_ENDPOINT_TAR_SHA),'initialendpoint')
 initial=json.loads(members['CLOSED_ENDPOINT_SNAPSHOT.json'])['base'];start=__import__('torch').tensor(initial['control'],dtype=__import__('torch').float64)
 case=observer._load_case();profile={**plan,'parameters_sha256':observer.PARAMETERS_SHA,'truth_sha256':observer.TRUTH_SHA}
 problem,original,p,truth=observer._fixed_case_tensors(case,profile)
 sources=observer._source_hashes(plan,PLAN,digest);runtime=trial.shared._runtime()
 inp=trial.shared._input_identity(problem,original,base['control'],p,truth)
 if inp!=base['input_after'] or runtime!=base['runtime_after']:raise ValueError('observation input/runtime mismatch')
 r={'phase':'running','execution_status':'running','plan_sha256':digest,'score_evaluations_started':0,'score_evaluations_completed':0,'optimizer_steps':0,'candidate_evaluations':0,'selection_use':False,'source_before':sources,'input_before':inp,'runtime_before':runtime,'score_history':[]}
 def write():trial.tangent._write(ATTEMPT/'step.json',r)
 def check(stage):
  if time.monotonic()>=deadline:raise TimeoutError('observer deadline')
  if observer._source_hashes(plan,PLAN,digest)!=sources or trial.shared._runtime()!=runtime or trial.shared._input_identity(problem,original,base['control'],p,truth)!=inp:raise ValueError('observation identity changed')
 def progress(label,stage,value):
  r['score_evaluations_started' if stage=='started' else 'score_evaluations_completed']+=1;r['score_history'].append({'point':label,'stage':stage,'E':value});write()
 write()
 r['score_pair']=observer.observe_score_pair(problem,p,truth,start,base['control'],check=check,progress=progress);check('final')
 r.update(phase='finished',execution_status='completed',source_after=observer._source_hashes(plan,PLAN,digest),source_unchanged=True,input_after=inp,fixed_input_unchanged=True,runtime_after=runtime,runtime_unchanged=True,elapsed_seconds=time.monotonic()-started,start_control_sha256=endpoint.BASE_SHA,final_control_sha256=endpoint.FINAL_SHA,root_claim=False,minimum_claim=False,response_claim=False,forecast_skill_claim=False);write()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args()
 if a.child:child(a.plan_sha256);return
 load_plan(a.plan_sha256)
 if ATTEMPT.exists():raise ValueError('observer attempt must be fresh')
 ATTEMPT.mkdir();resource=trial.previous.run_guarded_diagnostic([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child','--plan-sha256',a.plan_sha256],wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'],report_path=ATTEMPT/'step.resource.json',log_path=ATTEMPT/'step.log')
 c=ATTEMPT/'step.json';status=trial.shared._execution_status(resource,wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'])
 if not c.exists() or json.loads(c.read_text()).get('execution_status')!='completed':status='failed'
 r={'execution_status':status,'resource':resource,'child_sha256':observer._sha_file(c) if c.exists() else None};trial.tangent._write(ATTEMPT/'step.run.json',r);print(json.dumps(r))

if __name__=='__main__':main()
