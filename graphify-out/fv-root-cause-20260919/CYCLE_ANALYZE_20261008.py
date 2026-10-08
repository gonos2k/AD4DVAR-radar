"""Independent saved-vector arithmetic; no FV or production derivative calls."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
path = HERE / 'exploration_correction_cycle_20261008_attempt1/step.json'
x = json.loads(path.read_text())
p = json.loads(path.with_suffix('.run.json').read_text())
r = json.loads(path.with_suffix('.resource.json').read_text())
old = json.loads((HERE/'sample_common_search_20261008_attempt1/step.json').read_text())
ref = json.loads((HERE/'model_guided_resume_20261008_attempt1/step.json').read_text())
eps = np.finfo(float).eps
sha = lambda c: hashlib.sha256(np.array(c,dtype='<f8').tobytes()).hexdigest()
checks = {'parent_child_hash':p['child_sha256']==hashlib.sha256(path.read_bytes()).hexdigest(),
          'resource_matches':p['resource']==r,
          'source_closure':x['source_before']==x['source_after'],
          'fixed_input_closure':x['fixed_input_unchanged'] is True,
          'runtime_closure':x['runtime']==x['runtime_after']}
control = np.array(old['accepted_control']); rows=[]
for it in x['iterations']:
 g,d,w = (np.array(it[k]) for k in ['base_gradient','direction','H_direction'])
 c=np.array(it['accepted_control']);ng=np.array(it['accepted_gradient']);trial=it['trials'][-1];a=trial['alpha']
 jbound=it['base_objective']+1e-4*a*float(g@d)
 phibound=it['base_phi']+1e-4*a*float(g@w)
 ok={'lineage':sha(control)==it['base_control_sha256'],
     'control_hash':sha(c)==it['accepted_control_sha256'],
     'fresh_negative_gradient':bool(np.array_equal(d,-g)),
     'affine_step':bool(np.array_equal(c,control+a*d)),
     'J_Armijo':it['accepted_objective']<=jbound,
     'Phi_Armijo':it['accepted_phi']<=phibound,
     'Phi_arithmetic':bool(np.isclose(ng@ng/2,it['accepted_phi'],rtol=1e-14)),
     'strict_branch':trial['strict_point_passed'] is True,
     'commit':it['committed'] is True}
 assert all(ok.values()),ok
 rows.append({'index':it['index'],'checks':ok,'J':it['accepted_objective'],'Phi':it['accepted_phi'],
 'gInf':float(abs(ng).max()),'gNorm':float(np.linalg.norm(ng)),'alpha':a,'candidate_count':len(it['trials']),
 'displacement':float(np.linalg.norm(c-control)),'J_armijo_upper':jbound,'Phi_armijo_upper':phibound,
 'g_dot_d':float(g@d),'g_dot_Hd':float(g@w),
 'gradient_linearization_relative_error':float(np.linalg.norm(ng-g-a*w)/np.linalg.norm(ng))})
 control=c
current=x['current_state'];cg=np.array(current['gradient'])
comparisons={}
for name,state in [('preexploration',ref['current_state']),('correction_start',{'objective':old['accepted_objective'],'phi':old['accepted_phi_diagnostic'],'gradient':old['accepted_gradient']})]:
 gg=np.array(state['gradient'])
 comparisons[name]={'J_change_percent':100*(current['objective']/state['objective']-1),
 'Phi_change_percent':100*(current['phi']/state['phi']-1),
 'gInf_before':float(abs(gg).max()),'gInf_after':float(abs(cg).max()),
 'gNorm_before':float(np.linalg.norm(gg)),'gNorm_after':float(np.linalg.norm(cg)),
 'blocks':{b:{'before':float(np.linalg.norm(gg[sl])),'after':float(np.linalg.norm(cg[sl]))} for b,sl in [('field',slice(0,20)),('flow',slice(20,25)),('growth',slice(25,26))]}}
assessment=x['cycle_assessment']; historical=ref['current_state']; recovery=True
for metric in ['objective','phi']:
 budget=128*eps*max(abs(historical[metric]),abs(current[metric]),np.finfo(float).tiny)
 recovery = recovery and historical[metric]-current[metric]>budget
 checks['assessment_'+metric]=assessment['metrics'][metric]['current']==current[metric] and assessment['metrics'][metric]['preexploration_reference']==historical[metric]
checks['recovered_metadata']=bool(assessment['recovered']==recovery)
checks['final_hash']=sha(control)==x['current_control_sha256']
assert all(checks.values()),checks
out={'scope':'saved vector/hash/receipt arithmetic only; no new FV/HVP','checks':checks,'iterations':rows,'comparisons':comparisons,'cycle_recovered':bool(recovery),'current_control_sha256':x['current_control_sha256'],'numerical_status':x['numerical_status'],'resource':{k:r[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','exit_code']},'counts':{'corrections':len(rows),'HVP_started':x['hvp_calls'],'HVP_completed':x['hvp_calls_completed'],'PCG':x['pcg_solves']},'not_a_root':current['gradient_inf']>1e-10,'remaining':['eligible original3h root','curvature/adjoint/reanalysis','independent synthetic future verification']}
(HERE/'CYCLE_RESULT_20261008.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'comparisons':comparisons,'current_control_sha256':out['current_control_sha256']},indent=2))
