"""Verify the saved paired score arithmetic/resource closure, without forecasting."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from examples.weather_scenarios import fv_point_3h_netflux_future_observation as observer
w=Path(__file__).resolve().parent;path=w/'attempt1/score_pair.json'
r=json.loads(path.read_text());run=json.loads((w/'attempt1/score_pair.run.json').read_text());resource=json.loads((w/'attempt1/score_pair.resource.json').read_text())
assert run['child_sha256']==hashlib.sha256(path.read_bytes()).hexdigest() and run['resource']==resource
assert observer._run_summary(resource,path,r['plan_sha256'])['execution_status']=='completed'
plan=observer.load_plan(observer.PLAN,r['plan_sha256'])
assert r['source_before']==r['source_after']
assert r['input_before']==r['input_after'] and r['runtime_before']==r['runtime_after']
m=json.loads((w/'METRICS.json').read_text());scores=r['score_pair']
assert m['start_E']==scores['start_E'] and m['final_E']==scores['final_E']
delta=scores['final_E']-scores['start_E'];assert delta==scores['delta_E_final_minus_start']
assert m['decrease_percent']==-100*delta/scores['start_E']
assert m['start_control_sha256']==observer.START_CONTROL_SHA and m['final_control_sha256']==observer.FINAL_CONTROL_SHA
print('Stored pair verified:',scores['start_E'],scores['final_E'],'delta',delta,'decrease%',m['decrease_percent'])
print('Fixed source/archive pins:',len(plan['source_files']),len(plan['archive_files']))
