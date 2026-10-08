"""Arithmetic of saved finite face probes; no production numerical calls."""
import hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
rawpath=HERE/'face_transition_attempt1/diagnostic.json'
r=json.loads(rawpath.read_text())
resource=json.loads(rawpath.with_suffix('.resource.json').read_text())
base=json.loads((HERE/'model_guided_resume_20261008_attempt1/step.json').read_text())
cb=np.array(base['current_control']);pivot=r['pivot_control_index']
normal=np.array(r['gradient_jump']['normal_at_zero_face_chart']);normal/=np.linalg.norm(normal)
points=[];checks=[];samples={s['requested_eta']:s for s in r['samples']}
for eta,s in samples.items():
 c=np.array(s['control']);g=np.array(s['gradient']);geo=s['geometry'];tr=s['donor_trace']
 gaps=np.array([t['face']['minus_donor']-t['face']['plus_donor'] for t in tr])
 points.append({'eta':eta,'J':s['objective'],'Phi':s['phi'],'gInf':s['gradient_inf'],
  'eta_chart_slope':geo['eta_transverse_slope'],'unit_normal_gradient':geo['unit_normal_gradient'],
  'tangent_norm':geo['tangent_gradient_norm'],'trace_stages':len(tr),
  'effective_donor_gap_min':float(gaps.min()),'effective_donor_gap_max':float(gaps.max()),
  'effective_donor_gap_rms':float(np.sqrt(np.mean(gaps**2))),
  'strict_branch':s['branch']['status']})
 checks.append({'eta':eta,'control_hash_matches':hashlib.sha256(c.tobytes()).hexdigest()==s['control_sha256'],
  'gradient_hash_matches':hashlib.sha256(g.tobytes()).hexdigest()==s['gradient_sha256'],
  'retained_controls_exact':bool(np.array_equal(np.delete(c,pivot),np.delete(cb,pivot))),
  'Phi_error':float(g@g/2-s['phi'])})
pairs={}
for name,minus,plus in [('inner',-1e-6,1e-6),('outer',-2e-6,2e-6)]:
 gm=np.array(samples[minus]['gradient']);gp=np.array(samples[plus]['gradient']);delta=gp-gm
 dn=normal*(normal@delta);dt=delta-dn;theta=float(np.clip(-(gm@delta)/(delta@delta),0,1))
 v=gm+theta*delta;direction=-v
 pairs[name]={'etas':[minus,plus],'gradient_jump_norm':float(np.linalg.norm(delta)),
  'tangent_jump_norm':float(np.linalg.norm(dt)),
  'normal_squared_fraction':float((dn@dn)/(delta@delta)),
  'Phi_jump':float((gp@gp-gm@gm)/2),'Phi_identity':float(delta@(gp+gm)/2),
  'Phi_normal_part':float(dn@(normal*(normal@(gp+gm)))/2),
  'Phi_tangent_part':float(dt@(gp+gm-normal*(normal@(gp+gm)))/2),
  'segment_theta':theta,'segment_gradient_norm':float(np.linalg.norm(v)),
  'sample_directional_pairings':[float(gm@direction),float(gp@direction)],
  'scope':'finite sampled-gradient algebra; no limiting/Clarke/root certificate or applied direction'}
left=samples[-1e-6]['branch_signature']['face_signs'];right=samples[1e-6]['branch_signature']['face_signs']
locations=set()
for a,b in zip(left,right,strict=True):
 for axis in ('qx','qy'):
  for i,(u,v) in enumerate(zip(a[axis],b[axis],strict=True)):
   for j,(x,y) in enumerate(zip(u,v,strict=True)):
    if x!=y:locations.add((axis,i,j))
summary={'raw_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),
 'scope':'one generic face diagnostic at fixed retained coordinates; no optimization/HVP/PCG',
 'base_control_sha256':r['base_control_sha256'],'zero_J':r['eta_zero_cost_only']['objective'],
 'points':points,'pairs':pairs,'inner_cross_changed_face_locations':[list(x) for x in sorted(locations)],
 'branch_pair_counts':{name:{kind:{key:value for key,value in changes.items() if key!='changed_indices'}
                       for kind,changes in pair.items()} for name,pair in r['branch_side_differences'].items()},
 'resource':{k:resource[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','exit_code']},
 'remaining':['eligible original3h root','current full curvature/original adjoint/reanalysis','independent future validation']}
(HERE/'FACE_TRANSITION_RESULT_20261008.json').write_text(json.dumps(summary,indent=2)+'\n')
(HERE/'FACE_TRANSITION_INDEPENDENT_CHECKS_20261008.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'pairs':pairs,'face_locations':summary['inner_cross_changed_face_locations']},indent=2))
