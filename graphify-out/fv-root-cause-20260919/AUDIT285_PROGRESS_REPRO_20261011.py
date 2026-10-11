from types import SimpleNamespace
import json
from pathlib import Path
import torch
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as search
model=SimpleNamespace(direction=torch.tensor([1.,0.],dtype=torch.float64),theta=.5,gates={'finite':True,'both_side_gradients_descend':True,'merit_descends':True},merit_product=-.01,residual_direction=torch.tensor([1.,0.],dtype=torch.float64))
record={'candidate_slots_used':9,'closed_steps':[{'previous_P2_preserved':True}]}
calls=0
def evaluate(*args):
 global calls
 calls+=1
 if calls==3:raise TimeoutError('synthetic timeout')
 return {'mixing_minimum_valid':True,'J_armijo_passed':True,'F_squared_armijo_passed':False,'branch_pair_passed':True,'face_audit_passed':True,'side_objectives_match_native':True,'side_gradients_finite':True,'finite':True}
try:
 accepted,trials,meta=search._search_candidates(model,torch.zeros(2,dtype=torch.float64),evaluate,chart_candidate=lambda c,d,a:c+a*d)
 record.update(candidate_trials=trials,search=meta)
 record['candidate_slots_used']+=meta['evaluated_count']
except TimeoutError:
 pass
result={'prior_recorded_slots':9,'actual_new_evaluations_completed':calls-1,'new_evaluation_started_at_timeout':calls,'recorded_slots_after_timeout':record['candidate_slots_used'],'current_trials_saved':'candidate_trials' in record,'closed_prefix_preserved':record['closed_steps'][0]['previous_P2_preserved'],'production_executed':False}
Path('graphify-out/fv-root-cause-20260919/AUDIT285_PROGRESS_REPRO_20261011.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
