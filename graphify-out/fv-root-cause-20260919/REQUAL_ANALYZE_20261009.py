"""Saved requalification state and receipt arithmetic; no production FV/AD."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
raw=HERE/'candidate_requalification_20261009_attempt1/step.json'
data=raw.read_bytes() if raw.exists() else gzip.decompress(raw.with_suffix('.json.gz').read_bytes())
x=json.loads(data);parent=json.loads(raw.with_suffix('.run.json').read_text());resource=json.loads(raw.with_suffix('.resource.json').read_text())
a=x['iterations'][0]['accepted'];proposal=x['iterations'][0]['proposal'];gminus=np.array(a['side_gradients']['-1']);gplus=np.array(a['side_gradients']['1']);theta=a['theta'];gm=(1-theta)*gminus+theta*gplus
c=np.array(a['control']);c0=np.array(x['face_control']);w=np.array(x['face_qualification']['weights']);qscale=max(abs(w));q=w@np.tanh(c[20:25]);F=np.r_[gm,q/qscale];F0=np.array(x['residual']);rem=np.array(x['solve']['actual_rhs_residual'])
sha=lambda c:hashlib.sha256(np.array(c,dtype='<f8').tobytes()).hexdigest()
checks={'child_hash':hashlib.sha256(data).hexdigest()==parent['child_sha256'],
 'resource_equal':resource==parent['resource'],'current_hash':sha(c)==x['current_control_sha256']==a['control_sha256'],
 'committed':x['candidate_committed'] is True and x['optimizer_steps_applied']==1,
 'individual_gradient_repeat':x['final_repeat']['gradient_matches_proposal'] is True,
 'all_final_checks':all(x['final_repeat'][k] for k in ['objective_matches_proposal','side_objectives_match_native','branch_matches_proposal','merit_matches_proposal','face_roundoff_passed','source_unchanged','fixed_input_unchanged','runtime_unchanged','deadline_passed']),
 'current_pair':x['final_repeat']['current_branch_pair_gate']['passed'] is True,
 'HVP_count':x['hvp_calls_started']==x['hvp_calls_completed']==2,
 'new_dense_solves':x['policy']['dense_solves']==0,
 'F_arithmetic':bool(np.isclose(F@F,a['F_squared'],rtol=1e-14)),
 'radius':bool(np.linalg.norm(c-c0)<=.05),
 'source_closure':x['source_before']==x['source_after'],'runtime_closure':x['runtime']==x['runtime_after'],
 'actual_direction_RHS_gate':float(np.linalg.norm(rem)/np.linalg.norm(F0))<=1e-10,
 'J_acceptance':a['J_armijo_passed'] is True,'F_acceptance':a['F_squared_armijo_passed'] is True}
assert all(checks.values()),checks
n=np.zeros(26);n[20:25]=w*(1-np.tanh(c[20:25])**2);un=n/np.linalg.norm(n)
out={'scope':'stored vectors, exact byte hashes and receipts only; no new FV/HVP','checks':checks,
 'candidate_type':x['candidate_type'],'inherited_numerical_status':x['numerical_status'],
 'current_control_sha256':x['current_control_sha256'],'alpha':a['alpha'],'theta':theta,
 'displacement_l2':float(np.linalg.norm(c-c0)),'base_face_J':x['face_point_native_objective'],'candidate_J':a['objective'],
 'J_change_percent_vs_face':100*(a['objective']/x['face_point_native_objective']-1),
 'base_scaled_F_norm':float(np.linalg.norm(F0)),'candidate_scaled_F_norm':float(np.linalg.norm(F)),
 'candidate_scaled_F_squared':float(F@F),'F_squared_change_percent':100*((F@F)/(F0@F0)-1),
 'mixed_gradient_inf':float(abs(gm).max()),'minus_gradient_inf':float(abs(gminus).max()),'plus_gradient_inf':float(abs(gplus).max()),
 'minus_unit_normal_gradient':float(un@gminus),'plus_unit_normal_gradient':float(un@gplus),
 'mixed_tangent_gradient_norm':float(np.linalg.norm(gm-un*(un@gm))),
 'weighted_Q_static':float(q),'production_Q':a['production_face_value'],'Q_roundoff_bound':a['face_roundoff_bound'],
 'actual_current_direction_rhs_relative':float(np.linalg.norm(rem)/np.linalg.norm(F0)),
 'archived_matrix_backward_error':x['solve']['archived_matrix_backward_error'],
 'candidate_count':len(x['iterations'][0]['trials']),
 'evaluated_candidates':sum(t.get('objective') is not None for t in x['iterations'][0]['trials']),
 'current_pair_signatures':x['final_repeat']['side_trace_signatures'],
 'branch_changes_from_base':a['branch_changes_from_base'],
 'support_margins':{s:t['minimum_margins'] for s,t in a['branch_trace'].items()},
 'resource':{k:resource[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','exit_code']},
 'not_a_smooth_or_nonsmooth_root':float(abs(gm).max())>1e-10,
 'remaining':['newpoint operator/Hessian refresh','minimum/curvature/active-structure qualification','appropriate coupled adjoint/reanalysis','independent synthetic future validation']}
(HERE/'REQUAL_RESULT_20261009.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['current_control_sha256','candidate_J','candidate_scaled_F_norm','J_change_percent_vs_face','F_squared_change_percent','mixed_gradient_inf','minus_unit_normal_gradient','plus_unit_normal_gradient']},indent=2))
