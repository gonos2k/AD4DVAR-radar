"""Independent saved-vector/receipt arithmetic; no FV or production AD calls."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
raw=HERE/'nonsmooth_coupled_20261009_attempt1/step.json'
data=raw.read_bytes() if raw.exists() else gzip.decompress(raw.with_suffix('.json.gz').read_bytes())
x=json.loads(data)
parent=json.loads(raw.with_suffix('.run.json').read_text())
resource=json.loads(raw.with_suffix('.resource.json').read_text())
M=np.array(x['matrix']);F=np.array(x['residual']);delta=np.array(x['delta']);theta=x['theta']
gjump=M[:26,26];gm=F[:26]-theta*gjump;gp=F[:26]+(1-theta)*gjump
c=np.array(x['face_control']);w=np.array(x['face_qualification']['weights']);qscale=max(abs(w))
n=np.zeros(26);n[20:25]=w*(1-np.tanh(c[20:25])**2);un=n/np.linalg.norm(n)
sha=lambda a:hashlib.sha256(np.array(a,dtype='<f8').tobytes()).hexdigest()
gt=F[:26]-un*(un@F[:26]);jump_t=gjump-un*(un@gjump)
checks={'child_hash':hashlib.sha256(data).hexdigest()==parent['child_sha256'],
 'resource_receipt':parent['resource']==resource,
 'face_hash':sha(c)==x['face_control_sha256'],
 'final_hash':sha(x['current_control'])==x['current_control_sha256'],
 'no_commit':x['candidate_committed'] is False and x['optimizer_steps_applied']==0,
 'fixed_input':x['fixed_input_unchanged'] is True,
 'runtime':x['runtime']==x['runtime_after'],
 'source':x['source_before']==x['source_after'],
 'HVP_count':x['hvp_calls_started']==x['hvp_calls_completed']==56,
 'matrix_normal_row':bool(np.allclose(M[-1,:26],n/qscale,rtol=1e-14,atol=1e-16)),
 'gradient_norms':bool(np.allclose([np.linalg.norm(gm),np.linalg.norm(gp)],x['mix']['endpoint_norms'],rtol=1e-14)),
 'linear_solve':float(np.linalg.norm(M@delta+F)/np.linalg.norm(F))<=1e-10,
 'current_unchanged':x['current_control_sha256']==x['base_control_sha256'],
}
assert all(checks.values()),checks
base=x['face_qualification']['traces'];points=[]
for t in x['iterations'][0]['trials']:
 row={k:t.get(k) for k in ['alpha','status','actual_path_norm','objective','theta','J_armijo_passed','F_squared','F_squared_armijo_passed','face_roundoff_passed','branch_support_passed']}
 if 'control' in t:
  cc=np.array(t['control']);assert sha(cc)==t['control_sha256']
  row['Q_weighted_static']=float(w@np.tanh(cc[20:25]))
  row['J_change_percent_vs_face']=100*(t['objective']/x['face_point_native_objective']-1)
  row['F_squared_change_percent']=100*(t['F_squared']/(F@F)-1)
  row['trace_changes']={}
  for side, tr in t['branch_trace'].items():
   b=base[side];counts={};stages=[]
   for i,(before,after) in enumerate(zip(b['choices'],tr['choices'],strict=True)):
    changed=False
    for orientation,(a,z) in zip(('x','y'),zip(before,after,strict=True),strict=True):
     for key in a:
      size=int(np.count_nonzero(np.array(a[key])!=np.array(z[key])))
      if size:counts[orientation+'_'+key]=counts.get(orientation+'_'+key,0)+size;changed=True
    if changed:stages.append(i)
   face_changes=sum(a!=b for a,b in zip(b['face_signs'],tr['face_signs'],strict=True))
   strict=tr['stage_count']==360 and not tr['nonfinite_or_tie']
   assert strict
   row['trace_changes'][side]={'current_point_strict':strict,'changed_limiter_stage_count':len(stages),
       'changed_limiter_stages':stages,'selector_counts':counts,'changed_other_face_sign_stages':face_changes}
  same_pair=t['branch_trace']['-1']['signature_sha256']==t['branch_trace']['1']['signature_sha256']
  row['same_other_branch_between_candidate_sides']=same_pair
 points.append(row)
Hminus=np.array(x['hessian_minus']);Hplus=np.array(x['hessian_plus']);Hmix=(1-theta)*Hminus+theta*Hplus
assert np.allclose(M[:26,:26],Hmix,rtol=1e-14,atol=1e-14)
QH=np.zeros((26,26));QH[20:25,20:25]=np.diag(-2*w*np.tanh(c[20:25])*(1-np.tanh(c[20:25])**2))
mu=float(n@F[:26]/(n@n));T=np.linalg.qr(n[:,None],mode='complete')[0][:,1:]
eig=np.linalg.eigvalsh(T.T@(Hmix-mu*QH)@T)
out={'scope':'saved tensors/trace choices/hash arithmetic; no additional FV/AD/HVP',
 'checks':checks,'numerical_status':x['numerical_status'],'candidate_committed':False,
 'face_native_J':x['face_point_native_objective'],'theta':theta,'scaled_F_norm':float(np.linalg.norm(F)),
 'scaled_F_squared':float(F@F),'mixed_gradient_inf':float(abs(F[:26]).max()),
 'side_gradient_source':'algebraically reconstructed from saved F, jump column and theta; direct base side vectors were not archived',
 'minus_gradient':gm.tolist(),'plus_gradient':gp.tolist(),
 'minus_normal_gradient':float(un@gm),'plus_normal_gradient':float(un@gp),
 'mixed_tangent_norm':float(np.linalg.norm(gt)),'gradient_jump_tangent_norm':float(np.linalg.norm(jump_t)),
 'base_tangent_curvature_min':float(eig.min()),'base_tangent_curvature_max':float(eig.max()),
 'curvature_scope':'projected base diagnostic only, not endpoint/minimum/response certification',
 'linear_relative_residual_recomputed':float(np.linalg.norm(M@delta+F)/np.linalg.norm(F)),
 'candidate_count':len(points),'evaluated_candidates':sum('control' in t for t in x['iterations'][0]['trials']),
 'trials':points,'resource':{k:resource[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','exit_code']},
 'current_control_sha256':x['current_control_sha256'],
 'evidence_limits':['successful parity comparison values/errors not retained; completion follows enforced source gates and four completed parity products','no finite-path branch certificate','no new committed optimization point','minimum/response/forecast validations incomplete']}
(HERE/'KINK_RESULT_20261009.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['numerical_status','theta','scaled_F_norm','minus_normal_gradient','plus_normal_gradient','base_tangent_curvature_min','candidate_count','evaluated_candidates']},indent=2))
