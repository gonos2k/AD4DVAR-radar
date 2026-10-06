import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path('/Users/yhlee/ADVAR');E=ROOT/'graphify-out/fv-root-cause-20260919';D=E/'e0b_repeat_dual_20261006_attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tensor_sha=lambda x:hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()
r=json.loads((D/'step.json').read_text());parent=json.loads((D/'step.run.json').read_text());resource=json.loads((D/'step.resource.json').read_text());planpath=E/'E0B_DUAL_MERIT_CONTINUATION_PLAN_20261006.json';plan=json.loads(planpath.read_text());base=json.loads((E/'a35_dual_merit_20261006_attempt1/step.json').read_text())
assert parent['child_sha256']==sha(D/'step.json')
assert r['plan_sha256']==sha(planpath)
assert r['source_before']==r['source_after'] and r['fixed_input_unchanged']
historical_sources={'examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py':D/'producer.py','tests/test_fv_point_3h_dual_merit_continuation.py':D/'producer_test.py'}
assert all(sha(historical_sources.get(n, ROOT/n))==v for n,v in {**plan['source_files'],**plan['archive_files']}.items())
assert tensor_sha(base['accepted_control'])==r['base_control_sha256']
points=[]; c=np.asarray(base['accepted_control'],dtype=np.float64); bt=next(t for t in base['trials'] if t['status']=='accepted');j0=bt['objective'];phi0=bt['phi']
for it in r['iterations']:
 assert tensor_sha(c)==it['base_control_sha256']
 g=np.asarray(it['base_state']['gradient']);s=np.asarray(it['direction']);hs=np.asarray(it['H_s']);residual=hs+g
 rel=float(np.linalg.norm(residual)/np.linalg.norm(g));assert rel<=plan['policy']['pcg_relative_tolerance']
 assert math.isclose(rel,it['solve']['true_relative_residual'],abs_tol=1e-14)
 assert float(g@s)<0 and float(g@hs)<0
 rows=it['trials'];accepted=[t for t in rows if t.get('accepted')];assert len(accepted)==1 and accepted[0] is rows[-1]
 for t in rows:
  assert tensor_sha(t['control'])==t['control_sha256']
  assert math.isclose(t['J_armijo_threshold'],it['base_state']['objective']+plan['policy']['j_armijo_c1']*t['alpha']*float(g@s),abs_tol=1e-14)
  assert math.isclose(t['Phi_armijo_threshold'],it['base_state']['phi']+plan['policy']['phi_armijo_c1']*t['alpha']*float(g@hs),abs_tol=1e-14)
 t=accepted[0];new=np.asarray(it['accepted_control']);np.testing.assert_array_equal(new,c+t['alpha']*s)
 assert tensor_sha(new)==it['accepted_control_sha256']
 gn=np.asarray(it['accepted_gradient']);phin=float(gn@gn/2);assert math.isclose(phin,it['accepted_phi'],abs_tol=1e-14)
 assert t['J_armijo_passed'] and t['Phi_armijo_passed'] and t['strict_point_passed']
 assert it['accepted_objective']<it['base_state']['objective'] and phin<it['base_state']['phi']
 points.append({'iteration':it['iteration'],'control_sha256':it['accepted_control_sha256'],'J':it['accepted_objective'],'Phi':phin,'gradient_inf':float(np.max(np.abs(gn))),'gradient_l2':float(np.linalg.norm(gn)),'step_l2':float(np.linalg.norm(new-c)),'alpha':t['alpha'],'candidates':len(rows),'PCG':it['solve']['iterations'],'HVP':it['solve']['hvp_calls_including_true_residual'],'true_relative_residual':rel,'model':it['diagnostics']['model_diagnostics']})
 c=new
assert tensor_sha(c)==r['accepted_control_sha256'] and len(points)==r['optimizer_steps_applied']
assert r['hvp_calls']<=plan['policy']['max_hvp_calls'] and r['hvp_calls_completed']<=r['hvp_calls']
final=r.get('current_state') or r.get('current_iteration',{}).get('base_state') or {'objective':j0,'phi':phi0,'gradient_inf':float(np.max(np.abs(bt['gradient'])))};result={'scope':'saved actual run source/input receipts and independent NumPy arithmetic only; no FV/HVP/PCG regeneration','raw_sha256':sha(D/'step.json'),'plan_sha256':sha(planpath),'execution':parent['execution_status'],'numerical':r['numerical_status'],'points':points,'initial_J':j0,'initial_Phi':phi0,'final_J':final['objective'],'final_Phi':final['phi'],'final_gradient_inf':final['gradient_inf'],'J_reduction_percent':100*(1-final['objective']/j0),'Phi_reduction_percent':100*(1-final['phi']/phi0),'committed_iterations':len(points),'HVP_started':r['hvp_calls'],'HVP_completed':r['hvp_calls_completed'],'PCG_completed_solve_iterations':r['pcg_iterations_completed'],'guard_seconds':resource['elapsed_seconds'],'sampled_peak_child_RSS_bytes':resource['sampled_peak_rss_bytes'],'remaining':['no global SPD or eligible stationary point','no new forecast score/adjoint/reanalysis/physical validation']}
(E/'E0B_REPEAT_DUAL_RESULT_20261006.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='points'},indent=2));print('points',[(p['iteration'],p['J'],p['Phi'],p['gradient_inf']) for p in points])
