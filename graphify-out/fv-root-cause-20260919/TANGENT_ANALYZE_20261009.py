"""Saved-array arithmetic only; does not evaluate FV or production derivatives."""
from pathlib import Path
import gzip, hashlib, json
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
E = Path(__file__).resolve().parent
D = E / 'tangent_continuation_20261009_attempt1'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def control_sha(c):
    return digest(np.asarray(c, dtype='<f8').tobytes())

def array(x):
    return np.asarray(x, dtype=np.float64)

def point(control, theta, gradients, objective):
    c = array(control); gm, gp = array(gradients['-1']), array(gradients['1'])
    weights = array([0., -.08, -.28, -.14, -.84])
    q = weights @ np.tanh(c[20:25])
    normal = np.zeros(26); normal[20:25] = weights * (1-np.tanh(c[20:25])**2)
    nu = normal / np.linalg.norm(normal)
    mixed = (1-theta)*gm + theta*gp
    a = nu @ mixed; tangent = mixed - a*nu
    f = np.r_[mixed, q/.84]
    return {'control_sha256': control_sha(c), 'objective': objective, 'theta': theta,
        'F_squared': float(f@f), 'F_norm': float(np.linalg.norm(f)),
        'mixed_gradient_inf': float(np.max(np.abs(mixed))),
        'tangent_gradient_norm': float(np.linalg.norm(tangent)), 'normal_mixture': float(a),
        'normal_minus': float(nu@gm), 'normal_plus': float(nu@gp),
        'static_face_value': float(q), 'block_gradient_norms': [float(np.linalg.norm(mixed[s]))
            for s in (slice(0,20),slice(20,25),slice(25,26))]}, f, nu, normal

def main():
    path=D/'step.json'
    payload=path.read_bytes() if path.exists() else gzip.decompress((D/'step.json.gz').read_bytes())
    raw=json.loads(payload)
    run=json.loads((D/'step.run.json').read_text()); resource=json.loads((D/'step.resource.json').read_text())
    old=json.loads(gzip.decompress((E/'candidate_requalification_20261009_attempt1/step.json.gz').read_bytes()))
    base=old['iterations'][0]['accepted']
    assert run['child_sha256']==digest(payload) and run['resource']==resource
    assert run['execution_status']==raw['execution_status']=='completed'
    assert raw['source_before']==raw['source_after'] and raw['source_unchanged']
    assert raw['runtime']==raw['runtime_after'] and raw['fixed_input_unchanged']
    prior=base; points=[]; steps=[]
    p,f,nu,n=point(prior['control'],prior['theta'],prior['side_gradients'],prior['objective']);points.append(p)
    assert abs(p['F_squared']-base['F_squared'])<1e-15
    for item in raw['iterations']:
        if not item['accepted']:
            continue
        trial=next(t for t in item['trials'] if t['accepted'])
        p1,f1,nu1,n1=point(trial['control'],trial['theta'],trial['side_gradients'],trial['objective'])
        assert item['base_control_sha256']==points[-1]['control_sha256']
        assert item['committed_control']==trial['control']
        assert abs(p1['F_squared']-trial['F_squared'])<1e-15
        d=array(item['direction']); hm=array(item['hminus_direction']);hp=array(item['hplus_direction'])
        jump=array(prior['side_gradients']['1'])-array(prior['side_gradients']['-1'])
        hmix=(1-item['theta'])*hm+item['theta']*hp
        modeled=np.r_[hmix+jump*item['delta_theta'], n@d/.84]
        np.testing.assert_allclose(modeled,array(item['residual_direction']),rtol=2e-12,atol=2e-14)
        np.testing.assert_allclose(f,array(item['residual']),rtol=2e-12,atol=2e-14)
        alpha=trial['alpha']; delta=array(trial['control'])-array(prior['control'])
        assert abs(np.linalg.norm(delta)-trial['actual_path_norm'])<1e-15
        assert abs(trial['theta']-(item['theta']+alpha*item['delta_theta']))<1e-15
        assert trial['J_armijo_passed'] and trial['F_squared_armijo_passed']
        assert all(item['final_repeat'][k] for k in ('trace_matches_proposal','source_unchanged',
            'fixed_input_unchanged','runtime_unchanged','deadline_passed'))
        for side in ('-1','1'):
            np.testing.assert_array_equal(trial['side_gradients'][side],item['final_repeat']['side_gradients'][side])
        flin=f+alpha*modeled
        steps.append({'base_control_sha256':points[-1]['control_sha256'], 'control_sha256':p1['control_sha256'],
            'alpha':alpha,'actual_path_norm':float(np.linalg.norm(delta)),
            'evaluated_candidates':sum('objective' in t for t in item['trials']),
            'candidate_slots':len(item['trials']), 'F_directional_derivative':float(f@modeled),
            'tangent_direction_face_derivative':float(n@d),
            'model_F_squared':float(flin@flin),
            'finite_F_linearization_relative_error':float(np.linalg.norm(f1-flin)/np.linalg.norm(f1)),
            'branch_pair_signature':trial['branch_trace']['-1']['signature_sha256']})
        points.append(p1); prior=trial; f,nu,n=f1,nu1,n1
    assert len(steps)==raw['optimizer_steps_applied']==3
    assert points[-1]['control_sha256']==raw['current_control_sha256']
    assert raw['hvp_calls_started']==raw['hvp_calls_completed']==len(raw['hvp_history'])==6
    result={'scope':'Saved-array, receipt hash, and static face geometry arithmetic; no FV/HVP generation',
        'plan_sha256':raw['plan_sha256'],'raw_sha256':digest(payload),'points':points,'steps':steps,
        'J_decrease_percent':100*(1-points[-1]['objective']/points[0]['objective']),
        'F_squared_decrease_percent':100*(1-points[-1]['F_squared']/points[0]['F_squared']),
        'F_norm_decrease_percent':100*(1-points[-1]['F_norm']/points[0]['F_norm']),
        'resource':resource,'accepted_iterations':len(steps),'hvp_calls':6,
        'minimum_claim':False,'response_claim':False,'FV_completeness_external_estimate':70}
    (E/'TANGENT_RESULT_20261009.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('points','steps')},indent=2))

if __name__=='__main__':
    main()
