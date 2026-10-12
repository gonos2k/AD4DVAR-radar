"""Reconstruct saved comparison metrics and approvals; no FV/operator rerun."""
from pathlib import Path
import hashlib,json,math,sys
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from examples.weather_scenarios import fv_point_3h_netflux_direction_comparison as d
w=Path(__file__).resolve().parent
raw_bytes=(w/'attempt1/step.json').read_bytes();raw=json.loads(raw_bytes)
run=json.loads((w/'attempt1/step.run.json').read_text());resource=json.loads((w/'attempt1/step.resource.json').read_text())
assert run['child_sha256']==hashlib.sha256(raw_bytes).hexdigest() and run['resource']==resource
assert raw['execution_status']=='completed' and raw['optimizer_steps_applied']==1
assert raw['source_unchanged'] and raw['fixed_input_unchanged'] and raw['runtime_unchanged'] and raw['deadline_passed']
digest=(w/'PLAN_SHA256.txt').read_text().strip();plan=d.load_plan(d.PLAN,digest)
base=d.endpoint.load_endpoint();R0=float(base['repeat']['F_squared']);J0=float(base['repeat']['objective']);max0=float(base['full_G'][:26].abs().max())
records=[]
for arm in raw['arms']:
 assert arm['status']=='independent_P2_fullG_eligible' and arm['hvp_completed']==arm['hvp_started']==2
 repeat=arm['repeat'];expected=torch.tensor(arm['full_G_check']['G'],dtype=torch.float64)
 check=d.previous.fresh_full_G(repeat,expected,torch.tensor([0.,-.08,-.28,-.14,-.84],dtype=torch.float64))
 assert check['passed'] and d.tangent.fresh_final_closure(arm['proposal'],repeat)
 assert repeat['objective']<=arm['proposal']['J_armijo_bound'] and repeat['F_squared']<=arm['proposal']['F_squared_armijo_bound']
 G=torch.tensor(arm['residual'],dtype=torch.float64);r=torch.tensor(arm['residual_direction'],dtype=torch.float64)
 chi=float(torch.dot(G,r)**2/(torch.dot(G,G)*torch.dot(r,r)))
 assert abs(chi-arm['efficiency']['chi'])<128*torch.finfo(G.dtype).eps
 alpha=float(arm['candidate']['alpha']);model_change=alpha*r;actual_change=expected-G
 error=float((actual_change-model_change).norm());reported=arm['candidate']['model_error_norm']
 assert abs(error-reported)<128*torch.finfo(G.dtype).eps*max(error,reported,1.)
 max1=float(expected[:26].abs().max())
 trials=[]
 for trial in arm['trials']:
  path=ROOT/trial['candidate_receipt'];data=path.read_bytes()
  assert hashlib.sha256(data).hexdigest()==trial['candidate_receipt_sha256']
  facts=json.loads(data)
  trials.append({k:facts[k] for k in ('alpha','objective','F_squared','J_armijo_passed','F_squared_armijo_passed','branch_signature_matches_base','model_error_norm')})
 records.append({'name':arm['name'],'inclusive_arm_seconds':arm['inclusive_arm_seconds'],
  'GN_preparation_seconds':arm.get('GN_preparation_seconds',0.),'chi':chi,
  'alpha_model_min':arm['efficiency']['unbounded_model_alpha'],'alpha_selected':alpha,
  'alpha_selected_over_model_min':alpha/arm['efficiency']['unbounded_model_alpha'],
  'candidate_count':len(trials),'J':float(repeat['objective']),'R':float(repeat['F_squared']),
  'G_norm':math.sqrt(float(repeat['F_squared'])),'mixed_gradient_max':max1,
  'decrease_percent':{'J':100*(J0-repeat['objective'])/J0,'R':100*(R0-repeat['F_squared'])/R0,'mixed_gradient_max':100*(max0-max1)/max0},
  'model_error_norm':error,'actual_G_delta_norm':float(actual_change.norm()),
  'model_error_over_actual_delta':error/float(actual_change.norm()),
  'displacement_l2':float(torch.linalg.vector_norm(torch.tensor(repeat['control'],dtype=torch.float64)-base['control'])),
  'trials':trials})
winner=d.choose_repeated_arm(raw['arms'],torch.float64);assert winner is not None
assert raw['selected_direction']==winner['name']
assert raw['current_control']==raw['last_confirmed_control']==winner['repeat']['control']
assert raw['current_full_G']==winner['full_G_check']['G']
assert raw['hvp_completed']==4 and raw['GN_rows_completed']==24 and raw['GN_solves_completed']==1
summary={'plan_sha256':digest,'raw_sha256':run['child_sha256'],'source_files':len(plan['source_files']),'archive_files':len(plan['archive_files']),
 'base_control_sha256':d.BASE_SHA,'current_control_sha256':raw['current_control_sha256'],'selected_direction':raw['selected_direction'],
 'start':{'J':J0,'R':R0,'G_norm':math.sqrt(R0),'mixed_gradient_max':max0},'arms':records,
 'common_preparation_seconds':raw['common_preparation_seconds'],'guard':resource,'HVP_completed':4,'GN_rows':24,'GN_solves':1,
 'candidate_count':raw['candidate_evaluations_completed'],'commits':1,'termination':raw['numerical_status'],
 'source_input_runtime_closure':True,'final_point_operator_readiness':False,'future_E_calls':0,
 'scope':'Stored gradients/G vectors, independent repeat/Armijo/fullG and receipt/source/resource closure arithmetic; no additional FV/HVP/forecast or exact-minimum validation.'}
(w/'METRICS.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('arms','guard')},indent=2))
for a in records:print(a['name'],a['decrease_percent'],'chi',a['chi'],'modelerror/actualdelta',a['model_error_over_actual_delta'])
