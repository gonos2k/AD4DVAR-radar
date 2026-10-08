"""Saved control/gradient/HVP and receipt arithmetic; no production FV calls."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
path = HERE/'correction_resume_20261008_attempt1/step.json'
x = json.loads(path.read_text())
parent = json.loads(path.with_suffix('.run.json').read_text())
resource = json.loads(path.with_suffix('.resource.json').read_text())
old = json.loads((HERE/'exploration_correction_cycle_20261008_attempt1/step.json').read_text())
sha = lambda c: hashlib.sha256(np.array(c,dtype='<f8').tobytes()).hexdigest()
weights = np.array([0,-.08,-.28,-.14,-.84])

def metric(control, state):
    g=np.array(state['gradient']); c=np.array(control)
    n=np.zeros(26);n[20:25]=weights*(1-np.tanh(c[20:25])**2);n/=np.linalg.norm(n)
    return {'J':state['objective'],'Phi':float(g@g/2),'gInf':float(abs(g).max()),
        'gNorm':float(np.linalg.norm(g)),'control_sha256':sha(c),
        'Qy43_static':float(weights@np.tanh(c[20:25])),
        'Qy43_unit_normal_gradient':float(n@g),
        'Qy43_tangent_gradient_norm':float(np.linalg.norm(g-n*(n@g))),
        'blocks':{name:float(np.linalg.norm(g[s])) for name,s in [('field',slice(0,20)),('flow',slice(20,25)),('growth',slice(25,26))]}}

checks={'child_hash_matches':hashlib.sha256(path.read_bytes()).hexdigest()==parent['child_sha256'],
 'resource_matches':parent['resource']==resource,'source_closure':x['source_before']==x['source_after'],
 'input_closure':x['fixed_input_unchanged'] is True,'runtime_closure':x['runtime']==x['runtime_after'],
 'no_cycle_gate':'cycle_reference' not in x,'base_lineage':x['base_control_sha256']==old['current_control_sha256']}
control=np.array(old['current_control']);points=[metric(control,old['current_state'])];iterations=[]
for it in x['iterations']:
    c=np.array(it['accepted_control']);g=np.array(it['base_gradient']);d=np.array(it['direction']);w=np.array(it['H_direction']);ng=np.array(it['accepted_gradient']);trial=it['trials'][-1];a=trial['alpha']
    j_upper=it['base_objective']+1e-4*a*float(g@d);p_upper=it['base_phi']+1e-4*a*float(g@w)
    ok={'base_hash':sha(control)==it['base_control_sha256'],'accepted_hash':sha(c)==it['accepted_control_sha256'],
        'fresh_direction':bool(np.array_equal(d,-g)),'affine_control':bool(np.array_equal(c,control+a*d)),
        'J_armijo':it['accepted_objective']<=j_upper,'Phi_armijo':it['accepted_phi']<=p_upper,
        'Phi_arithmetic':bool(np.isclose(ng@ng/2,it['accepted_phi'],rtol=1e-14)),
        'strict_branch':trial['strict_point_passed'] is True,'committed':it['committed'] is True}
    assert all(ok.values()),ok
    state={'objective':it['accepted_objective'],'gradient':it['accepted_gradient']}
    points.append(metric(c,state));pred=g+a*w
    iterations.append({'index':it['index'],'checks':ok,'alpha_start':it['alpha_start'],'accepted_alpha':a,
        'candidate_count':len(it['trials']),'displacement':float(np.linalg.norm(c-control)),
        'g_dot_d':float(g@d),'g_dot_Hd':float(g@w),
        'gradient_linearization_relative_error':float(np.linalg.norm(ng-pred)/np.linalg.norm(ng)),
        'J_armijo_upper':j_upper,'Phi_armijo_upper':p_upper,
        'old_direction_at_new_point':float(ng@d),
        'analysis_signature':trial['branch_partition'].get('analysis_sha256'),
        'future_signature':trial['branch_partition'].get('future_sha256')})
    control=c
checks['current_control_hash']=sha(control)==x['current_control_sha256']
checks['count_closure']=len(x['iterations'])==x['optimizer_steps_applied']
assert all(checks.values()),checks
base,final=points[0],points[-1]
out={'scope':'saved vectors/static face geometry/receipt arithmetic only; no new FV/HVP',
 'checks':checks,'points':points,'iterations':iterations,
 'changes_vs_resume_base':{key:100*(final[key]/base[key]-1) for key in ['J','Phi','gNorm','gInf']},
 'numerical_status':x['numerical_status'],'full_root_claim':x['full_root_claim'],
 'HVP_started':x['hvp_calls'],'HVP_completed':x['hvp_calls_completed'],'PCG':x['pcg_solves'],
 'resource':{k:resource[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','exit_code']},
 'remaining':['eligible original3h root','curvature/original adjoint/reanalysis','independent synthetic future verification']}
(HERE/'CORRECTION_RESUME_RESULT_20261008.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'points':points,'changes':out['changes_vs_resume_base'],'status':out['numerical_status']},indent=2))
