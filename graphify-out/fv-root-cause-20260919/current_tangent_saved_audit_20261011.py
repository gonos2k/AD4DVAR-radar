"""Saved scalar/vector arithmetic; no FV, HVP, optimization or score calls."""
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import torch
from examples.weather_scenarios import fv_point_3h_current_tangent_search as producer

EVIDENCE=Path(__file__).resolve().parent
FOLDER=EVIDENCE/'current_tangent_search_20261011_attempt1'
RAW=FOLDER/'step.json'
data=json.loads(RAW.read_text());base=producer.parent.load_base()
G0=torch.tensor(base['current_full_G'],dtype=torch.float64)
base_inf=float(G0[:-1].abs().max())
arm_results=[]
for arm in data['arms']:
    trial=arm['candidate'];G=torch.tensor(trial['G'],dtype=torch.float64)
    r=torch.tensor(arm['residual_direction'],dtype=torch.float64)
    prediction=G0+trial['alpha']*r;error=G-prediction;delta=G-G0
    assert math.isclose(float(error.norm()),trial['model_error_norm'],rel_tol=1e-12)
    trace_changes={side:producer.parent.branch_changes(trial['branch_trace'][side],base['canonical_final_repeat']['branch_trace'][side]) for side in ('-1','1')}
    arm_results.append({'name':arm['name'],'actual_candidate_count':arm['search']['evaluations_completed'],
        'alpha':trial['alpha'],'movement':trial['actual_path_norm'],
        'J':trial['objective'],'R':trial['F_squared'],'norm':math.sqrt(trial['F_squared']),
        'max_mixed_component':float(G[:-1].abs().max()),'vartheta':trial['theta'],
        'J_reduction_percent':100*(1-trial['objective']/data['base_J']),
        'R_reduction_percent':100*(1-trial['F_squared']/data['base_R']),
        'norm_reduction_percent':100*(1-math.sqrt(trial['F_squared']/data['base_R'])),
        'max_component_reduction_percent':100*(1-float(G[:-1].abs().max())/base_inf),
        'predicted_R':float(prediction@prediction),'model_error_norm':float(error.norm()),
        'actual_G_change_norm':float(delta.norm()),'error_over_actual_change':float(error.norm()/delta.norm()),
        'trace_changes':trace_changes,'model_support':arm['supported'],'J_armijo_passed':trial['J_armijo_passed'],
        'R_armijo_passed':trial['F_squared_armijo_passed']})
selected=next(a for a in arm_results if a['name']==data['selected_direction'])
resource=json.loads((FOLDER/'step.resource.json').read_text())
metrics: dict[str,Any]={'scope':'one current-point conditional Q0 step; not minimum/response or general direction-quality certification',
    'plan_sha256':data['plan_sha256'],'raw_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
    'base_control_sha256':data['base_control_sha256'],'current_control_sha256':data['current_control_sha256'],
    'base_J':data['base_J'],'base_R':data['base_R'],'base_norm':float(G0.norm()),'base_max_mixed_component':base_inf,
    'base_vartheta':data['base_theta'],'selected':selected,'arms':arm_results,
    'selection_basis':data['selection_basis'],'comparison_complete':data['comparison_complete'],
    'face_decision':data['direction_audit']['face_decision'],'outward_native_cost_slopes':data['direction_audit']['outward_native_cost_slopes'],
    'event_gradient_calls':data['event_grad_completed'],'HVP_calls':data['hvp_completed'],
    'actual_candidate_evaluations':data['candidate_evaluations_completed'],'optimizer_commits':data['optimizer_steps_applied'],
    'new_GN_solves':0,'new_observation_row_VJPs':0,'postcommit_readiness':False,
    'full_G_component_match':data['final_full_G_check']['component_match'],'full_G_merit_match':data['final_full_G_check']['merit_match'],
    'future_score':{'units':'dBZ squared','lead_minutes_after_last_observation':180,'scope':'matched-twin native forward observation only',
        'base':data['base_future_score'],'selected':data['final_future_score'],
        'change':data['final_future_score']-data['base_future_score'],
        'change_percent':100*(data['final_future_score']/data['base_future_score']-1),
        'used_for_selection':False,'independent_forecast_skill_certified':False},
    'guard_seconds':resource['elapsed_seconds'],'sampled_peak_rss_bytes':resource['sampled_peak_rss_bytes'],
    'exit_code':resource['exit_code'],'resource_termination':resource['resource_termination'],
    'source_input_runtime_unchanged':all(data[k] for k in ('source_unchanged','fixed_input_unchanged','runtime_unchanged')),
    'eligible_original_minimum':False,'exact_FV_response_completed':False,'historical_FV_milestone_percent':70}
(EVIDENCE/'CURRENT_TANGENT_METRICS_20261011.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({'selected':selected['name'],'R_reduction_percent':selected['R_reduction_percent'],
    'max_component_reduction_percent':selected['max_component_reduction_percent'],'E_change_percent':metrics['future_score']['change_percent']}))
