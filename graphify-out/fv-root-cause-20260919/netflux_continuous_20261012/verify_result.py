from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
from examples.weather_scenarios import fv_point_3h_netflux_continuation as runner
from examples.weather_scenarios import fv_point_3h_current_gradient_endpoint as legacy_endpoint

def verify() -> dict:
    w=runner.WORK;raw_path=w/'attempt1/step.json';raw=json.loads(raw_path.read_text())
    plan=json.loads(runner.PLAN.read_text());digest=hashlib.sha256(runner.PLAN.read_bytes()).hexdigest()
    runner.load_plan(runner.PLAN,digest)
    expected_sources={**plan['source_files'],**plan['archive_files'],runner.PLAN.relative_to(runner.ROOT).as_posix():digest}
    assert raw['plan_sha256']==digest and raw['source_before']==raw['source_after']==expected_sources
    assert raw['source_unchanged'] and raw['fixed_input_unchanged'] and raw['runtime_unchanged']
    assert raw['numerical_version']==runner.VERSION and raw['last_confirmed_numerical_version']==runner.VERSION
    assert raw['runtime_before']==raw['runtime_after']
    assert raw['current_control']==raw['last_confirmed_control']
    assert raw['current_full_G']==raw['last_confirmed_full_G']
    base=runner.endpoint.load_endpoint()
    previous_G=base['full_G'];previous_control=base['control'];previous_sha=runner.BASE_SHA
    previous_J=float(base['repeat']['objective']);previous_R=float(torch.dot(previous_G,previous_G))
    metrics=[]
    for index,step in enumerate(raw['closed_steps']):
        assert step['point_index']==index
        repeat,G,sha=legacy_endpoint._close_step(step,previous_G,previous_sha)
        control=torch.tensor(repeat['control'],dtype=torch.float64)
        J=float(repeat['objective']);R=float(torch.dot(G,G));m=float(G[:26].abs().max())
        assert J<previous_J and R<previous_R
        assert step['hvp_started']==step['hvp_completed']==2
        assert not step['outward_derivatives']['release_required']
        assert step['minimum_mixture_P2_match']['passed']
        assert all(step['model_gates'][k] for k in ('finite','both_side_gradients_descend','merit_descends','envelope_slope_matches_residual_dot'))
        alpha=float(step['proposal']['alpha'])
        actual_delta_norm=float(torch.linalg.vector_norm(G-previous_G))
        model_error=float(step['selected_candidate']['model_error_norm'])
        traces=repeat['branch_trace']
        oldtraces=base['repeat']['branch_trace'] if index==0 else raw['closed_steps'][index-1]['final_repeat']['branch_trace']
        metrics.append({'index':index+1,'control_sha256':sha,'J':J,'R':R,'G_norm':math.sqrt(R),'mixed_gradient_max':m,
            'J_decrease_percent':100*(previous_J-J)/previous_J,'R_decrease_percent':100*(previous_R-R)/previous_R,
            'displacement_l2':float(torch.linalg.vector_norm(control-previous_control)),
            'model_error_norm':model_error,'actual_G_delta_norm':actual_delta_norm,
            'model_error_over_actual_G_delta':model_error/actual_delta_norm,
            'branch_signature_matches_base':step['selected_candidate']['branch_signature_matches_base'],
            'alpha':alpha,'cost_drop_over_linear_prediction':(previous_J-J)/(alpha*previous_R),
            'candidates':len(step['trials']),
            'branch_signature_changed':{s:traces[s]['signature_sha256']!=oldtraces[s]['signature_sha256'] for s in ('-1','1')}})
        previous_G,previous_control,previous_sha,previous_J,previous_R=G,control,sha,J,R
    assert raw['current_control_sha256']==previous_sha==raw['last_confirmed_control_sha256']
    assert torch.equal(previous_G,torch.tensor(raw['current_full_G'],dtype=torch.float64))
    assert len(metrics)==raw['optimizer_steps_applied']
    run=json.loads((w/'attempt1/step.run.json').read_text());resource=json.loads((w/'attempt1/step.resource.json').read_text())
    assert run['child_sha256']==hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert run['resource']==resource
    before={'J':float(base['repeat']['objective']),'R':float(torch.dot(base['full_G'],base['full_G'])),'G_norm':float(base['full_G'].norm()),'mixed_gradient_max':float(base['full_G'][:26].abs().max())}
    after={'J':previous_J,'R':previous_R,'G_norm':math.sqrt(previous_R),'mixed_gradient_max':float(previous_G[:26].abs().max())}
    result={'plan_sha256':digest,'raw_sha256':run['child_sha256'],'base_control_sha256':runner.BASE_SHA,'current_control_sha256':previous_sha,
        'numerical_version':runner.VERSION,'start':before,'final':after,
        'decrease_percent':{k:100*(before[k]-after[k])/before[k] for k in before},'steps':metrics,
        'fresh_HVP':raw['hvp_completed'],'candidates':raw['candidate_evaluations_completed'],'commits':len(metrics),
        'numerical_status':raw['numerical_status'],'execution_status':raw['execution_status'],'failure_stage':raw['failure_stage'],
        'reason':raw['termination_reason'],'resource':resource,'progress':raw.get('convergence_progress'),
        'source_fixed_input_runtime_closure':True,'internal_deadline_passed':raw['deadline_passed'],
        'postcommit_readiness':raw.get('postcommit_readiness_performed',False),
        'terminal_attempt_provenance':{'operator_control_sha256':raw.get('current_point',{}).get('base_control_sha256'),
            'matches_final_control':raw.get('current_point',{}).get('base_control_sha256')==previous_sha,
            'point_HVP_completed':raw.get('current_point',{}).get('hvp_completed',0),
            'model_gates':raw.get('current_point',{}).get('model_gates'),
            'last_phase':raw['numerical_status'],'failure_stage':raw.get('failure_stage')},'future_E_calls':0,
        'verification_scope':'Stored raw receipts, same-version source/input/runtime pins, componentwise prefix/P2 reconstruction and arithmetic; no additional production FV/HVP/adjoint or forecast run.'}
    runner.tangent._write(w/'METRICS.json',result)
    return result
if __name__=='__main__':
    result=verify();print(json.dumps({k:v for k,v in result.items() if k not in ('steps','resource','progress')},indent=2));print('resource',result['resource']);print('steps',len(result['steps']))
