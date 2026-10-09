"""Saved-array resume arithmetic; no FV objective, AD or HVP generation."""
from pathlib import Path
import gzip,hashlib,importlib.util,json
import numpy as np
E=Path(__file__).resolve().parent
D=E/'tangent_resume_20261009_attempt1'
spec=importlib.util.spec_from_file_location('tangent_saved_math',E/'TANGENT_ANALYZE_20261009.py')
assert spec is not None and spec.loader is not None
saved=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved)

def payload(path):
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.json.gz').read_bytes())

def main():
    data=payload(D/'step.json');raw=json.loads(data)
    parent=json.loads((D/'step.run.json').read_text());resource=json.loads((D/'step.resource.json').read_text())
    previous=json.loads(gzip.decompress((E/'tangent_continuation_20261009_attempt1/step.json.gz').read_bytes()))
    base=next(t for t in previous['iterations'][-1]['trials'] if t['accepted'])
    assert parent['child_sha256']==hashlib.sha256(data).hexdigest() and parent['resource']==resource
    assert raw['source_before']==raw['source_after'] and raw['fixed_input_unchanged'] and raw['runtime_unchanged']
    prior=base; p,f,nu,n=saved.point(prior['control'],prior['theta'],prior['side_gradients'],prior['objective'])
    points=[p];steps=[]
    assert raw['base_control_sha256']==p['control_sha256']
    for item in raw['iterations']:
        if not item['accepted']:continue
        trial=next(t for t in item['trials'] if t['accepted'])
        p1,f1,nu1,n1=saved.point(trial['control'],trial['theta'],trial['side_gradients'],trial['objective'])
        assert item['base_control_sha256']==points[-1]['control_sha256'] and item['theta']==prior['theta']
        assert item['committed_control']==trial['control'] and item['committed_theta']==trial['theta']
        assert abs(p1['F_squared']-trial['F_squared'])<1e-15
        d=np.asarray(item['direction']);r=np.asarray(item['residual_direction']);alpha=trial['alpha']
        hm=np.asarray(item['hminus_direction']);hp=np.asarray(item['hplus_direction'])
        jump=np.asarray(prior['side_gradients']['1'])-np.asarray(prior['side_gradients']['-1'])
        expected=np.r_[(1-item['theta'])*hm+item['theta']*hp+jump*item['delta_theta'],n@d/.84]
        np.testing.assert_allclose(r,expected,atol=2e-14,rtol=2e-12)
        np.testing.assert_allclose(item['residual'],f,atol=2e-14,rtol=2e-12)
        assert all(item['final_repeat'][k] for k in ('trace_matches_proposal','face_audit_passed',
            'branch_pair_passed','source_unchanged','fixed_input_unchanged','runtime_unchanged','deadline_passed'))
        for side in ('-1','1'):
            np.testing.assert_array_equal(trial['side_gradients'][side],item['final_repeat']['side_gradients'][side])
        f2=float(f@f);pdir=float(f@r);r2=float(r@r);flin=f+alpha*r
        cos2=pdir*pdir/(f2*r2);cos=np.clip(-pdir/np.sqrt(f2*r2),-1,1)
        delta=np.asarray(trial['control'])-np.asarray(prior['control'])
        assert abs(np.linalg.norm(delta)-trial['actual_path_norm'])<1e-15
        assert abs(trial['theta']-item['theta']-alpha*item['delta_theta'])<1e-15
        assert trial['J_armijo_passed'] and trial['F_squared_armijo_passed']
        steps.append({'base_control_sha256':points[-1]['control_sha256'],'accepted_control_sha256':p1['control_sha256'],
            'alpha':alpha,'model_optimal_alpha':-pdir/r2,'actual_path_norm':float(np.linalg.norm(delta)),
            'candidate_slots':len(item['trials']),'evaluated_candidates':sum('objective' in t for t in item['trials']),
            'F_directional_derivative':pdir,'predicted_F_squared':float(flin@flin),
            'predicted_F_squared_decrease_percent':100*(f2-float(flin@flin))/f2,
            'actual_F_squared_decrease_percent':100*(f2-p1['F_squared'])/f2,
            'model_cos_squared':cos2,'model_angle_degrees':float(np.degrees(np.arccos(cos))),
            'finite_F_linearization_relative_error':float(np.linalg.norm(f1-flin)/np.linalg.norm(f1)),
            'branch_pair_signature':trial['branch_trace']['-1']['signature_sha256']})
        points.append(p1);prior=trial;f,nu,n=f1,nu1,n1
    assert len(steps)==raw['optimizer_steps_applied'] and raw['current_control_sha256']==points[-1]['control_sha256']
    assert raw['current_theta']==points[-1]['theta'] and raw['native_objective']==points[-1]['objective']
    assert raw['hvp_calls_started']==raw['hvp_calls_completed']==len(raw['hvp_history'])<=6
    result={'scope':'Saved-array/receipt/static geometry arithmetic; no production reexecution',
        'plan_sha256':raw['plan_sha256'],'raw_sha256':hashlib.sha256(data).hexdigest(),
        'execution_status':parent['execution_status'],'numerical_status':raw['numerical_status'],
        'accepted_iterations':len(steps),'current_HVP_calls':raw['hvp_calls_completed'],
        'points':points,'steps':steps,'resource':resource,
        'J_decrease_percent':100*(1-points[-1]['objective']/points[0]['objective']),
        'F_squared_decrease_percent':100*(1-points[-1]['F_squared']/points[0]['F_squared']),
        'F_norm_decrease_percent':100*(1-points[-1]['F_norm']/points[0]['F_norm']),
        'minimum_claim':False,'response_claim':False,'FV_completeness_external_estimate':70}
    (E/'TANGENT_RESUME_RESULT_20261009.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('points','steps')},indent=2))

if __name__=='__main__':main()
