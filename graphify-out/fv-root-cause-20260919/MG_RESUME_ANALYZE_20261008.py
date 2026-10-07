"""Verify saved arithmetic, not production objective or derivatives."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rpath=HERE/'model_guided_resume_20261008_attempt1/step.json'
r=json.loads(rpath.read_text())
old=json.loads((HERE/'model_guided_attempt1/step.json').read_text())
cbase=np.array(old['current_control'])
prevpartition=old['current_state']['branch_partition']
checks=[]
for row in r['iterations']:
    g=np.array(row['base_gradient']);d=np.array(row['direction']);hd=np.array(row['H_direction'])
    c=np.array(row['accepted_control']);g1=np.array(row['accepted_gradient']);t=row['trials'][-1];a=t['alpha']
    alpha_phi=-(g@hd)/(hd@hd);alpha0=min(1.,.05/np.linalg.norm(d),alpha_phi)
    gl=g+a*hd;partition=t['branch_partition']
    trials=[]
    for tr in row['trials']:
        a1=tr['alpha'];gtr=np.array(tr['gradient']);jlim=row['base_objective']+1e-4*a1*(g@d)
        plim=row['base_phi']+1e-4*a1*(g@hd)
        trials.append({'index':tr['index'],'control_hash_matches':hashlib.sha256((cbase+a1*d).tobytes()).hexdigest()==tr['control_sha256'],
                       'Phi_error':float(gtr@gtr/2-tr['phi']),'J_threshold_error':float(jlim-tr['j_limit']),
                       'Phi_threshold_error':float(plim-tr['phi_limit'])})
    checks.append({'index':row['index'],'negative_current_gradient':bool(np.array_equal(d,-g)),
        'accepted_hash_matches':hashlib.sha256(c.tobytes()).hexdigest()==row['accepted_control_sha256'],
        'affine_control_hash_matches':hashlib.sha256((cbase+a*d).tobytes()).hexdigest()==row['accepted_control_sha256'],
        'alpha_model_error':float(row['alpha_start']-alpha0),'Phi_error':float(g1@g1/2-row['accepted_phi']),
        'gradient_prediction_relative_error':float(np.linalg.norm(g1-gl)/np.linalg.norm(g1)),
        'actual_over_quadratic_J_reduction':float((row['base_objective']-row['accepted_objective'])/(row['base_objective']-t['model']['predicted_J_quadratic'])),
        'analysis_branch_changed':partition['analysis_sha256']!=prevpartition['analysis_sha256'],
        'future_branch_changed':partition['future_sha256']!=prevpartition['future_sha256'],
        'trials':trials})
    cbase,prevpartition=c,partition
result={'scope':'archived NumPy vector arithmetic only; no new FV/J/g/HVP','raw_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest(),
        'checks':checks,'source_closed':r['source_before']==r['source_after'],'fixed_input_unchanged':r['fixed_input_unchanged'],
        'runtime_closed':r['runtime']==r['runtime_after']}
(HERE/'MG_RESUME_INDEPENDENT_CHECKS_20261008.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
print([{k:x[k] for k in ['index','alpha_model_error','gradient_prediction_relative_error','analysis_branch_changed','future_branch_changed']} for x in checks])
