"""Resume bounded native-J minimization on the observed qy[4,3] face.

Uses the existing exact chart Hessian and a standard spectral-shift modified
Newton correction. This is local objective minimization, not an alternate
merit or face-release search. Source physics and frozen plans are unchanged.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, sys, time
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUTDIR=HERE/'continuation3/attempt1'
INPUT=OUTDIR/'face_chart_newton_resume.json'
OUTPUT=OUTDIR/'modified_newton_face_qualification_attempt2.json'
PLAN=HERE/'continuation3/CONTINUATION_PLAN.json'
PLAN_SHA='67fa44ac8c6c08f79df99a9fe2c5bee10b385a7769bb65ad6caafafeb9c52a01'
SELF=Path(__file__)
spec=importlib.util.spec_from_file_location('native_qualifier',HERE/'qualify_candidate.py')
if spec is None or spec.loader is None: raise RuntimeError('native qualifier is unavailable')
qualifier=importlib.util.module_from_spec(spec);sys.modules[spec.name]=qualifier;spec.loader.exec_module(qualifier)
from advar import transport
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart, face_weights_from_basis
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent

FACE=('y',4,3); NC,NT=26,25; STATIONARITY_TOL=1e-10; ROUNDING_MULT=128.
# The cumulative A budget remains anchored at the original 3600 s cap.
LEDGER={'original_cap_s':3600.0,'prior_charged_s':2039.9,
        'postresume_preflight_s':4.7,'reserved_forecast_and_verification_s':30.0}
LEDGER['available_for_this_run_s']=LEDGER['original_cap_s']-LEDGER['prior_charged_s']-LEDGER['postresume_preflight_s']
LEDGER['run_deadline_s']=LEDGER['available_for_this_run_s']-LEDGER['reserved_forecast_and_verification_s']

def sha_bytes(raw:bytes)->str: return hashlib.sha256(raw).hexdigest()
def tensor_sha(x:torch.Tensor)->str: return sha_bytes(x.detach().contiguous().cpu().numpy().tobytes())
def write(path:Path,record:dict[str,Any])->None:
 def json_default(value:Any)->Any:
  if isinstance(value,torch.Tensor):return value.detach().contiguous().cpu().tolist()
  raise TypeError(f'cannot serialize {type(value).__name__}')
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False,default=json_default)+'\n');tmp.replace(path)

def run()->dict[str,Any]:
 if OUTPUT.exists(): raise ValueError('modified-Newton qualification output must be fresh')
 started=time.monotonic();deadline=started+LEDGER['run_deadline_s'];script_sha_before=sha_bytes(SELF.read_bytes())
 prior=json.loads(INPUT.read_text()); control=torch.tensor(prior['control'],dtype=torch.float64)
 if tensor_sha(control)!=prior['control_sha256'] or prior['status']!='nonpositive_or_unresolved_face_curvature':
  raise ValueError('input must be the exact preserved indefinite-Hessian point')
 case=qualifier.cycle.load_fixed_case();problem,original,truth,parameters=case['problem'],case['original'],case['truth'],case['parameters']; fs=problem.frozen.fv_transport
 _,source_before,transition=qualifier._execution_plan(continuation_plan_path=PLAN,expected_continuation_sha256=PLAN_SHA)
 if transition!=prior['source_transition'] or sha_bytes(PLAN.read_bytes())!=PLAN_SHA:
  raise ValueError('source transition or continuation pin differs from the saved point')
 expected_parameter=prior['input_parameters_sha256'] if 'input_parameters_sha256' in prior else qualifier._tensor_sha(parameters)
 if tensor_sha(parameters)!=expected_parameter: raise ValueError('fixed parameters differ from saved point')
 specsha=sha_bytes((ROOT/'src/advar/transport.py').read_bytes())
 if specsha!='c9dde24c12237d5ffee6b8ca0766b4d5619abf6110f3e858ee447bf49c1f7da9': raise ValueError('approved support-only transport source changed')
 qweights=face_weights_from_basis(fs.psi_basis,'y',(4,3)); normalweights=qweights*fs.coefficient_limits*(1-torch.tanh(control[20:25]).square());pivot_i=int(normalweights.abs().argmax())
 chart=FVFaceFluxCoordinateChart.from_basis(field_count=20,growth_count=1,coefficient_limits=fs.coefficient_limits,psi_basis=fs.psi_basis,axis='y',face=(4,3),pivot_index=pivot_i)
 coords=chart.to_face_coordinates(control);pivot=chart.layout['face_flux_coordinate'];coords[pivot]=0.; y=torch.cat((coords[:pivot],coords[pivot+1:]));control=chart.from_face_coordinates(coords)
 def lift(v:torch.Tensor)->torch.Tensor: return chart.from_face_coordinates(torch.cat((v[:pivot],v.new_zeros(1),v[pivot:])))
 def qvalue(c:torch.Tensor)->torch.Tensor:
  psi=torch.einsum('k,kij->ij',fs.coefficient_limits*torch.tanh(c[20:25]),fs.psi_basis);return transport.face_volume_fluxes(psi)[1][4,3]
 def chart_obj(v:torch.Tensor,p:torch.Tensor,side:int)->torch.Tensor:
  with transport.selected_face_extension('y',4,3,side): return problem.objective(lift(v),p)
 def diagnose(v:torch.Tensor)->dict[str,Any]:
  c=lift(v);branch=qualifier._analysis_branch(problem,c,parameters,deadline=deadline);full={};tg={}
  for side in (-1,1):
   with transport.selected_face_extension('y',4,3,side): full[side]=torch.func.grad(problem.objective,argnums=0)(c,parameters)
   tg[side]=torch.func.grad(lambda u,p,s=side:chart_obj(u,p,s),argnums=0)(v,parameters)
  qg=torch.func.grad(qvalue)(c);qn=torch.linalg.vector_norm(qg)
  slopes={s:float(torch.dot(full[s],qg)/qn) for s in (-1,1)}
  budgets={s:ROUNDING_MULT*torch.finfo(c.dtype).eps*float(full[s].norm()) for s in (-1,1)}
  theta=tangent.minimum_mixture_weight(full[-1],full[1]);mix=(1-theta)*full[-1]+theta*full[1]
  agree=float((tg[-1]-tg[1]).abs().max());ginf=float(tg[1].abs().max())
  supported=(branch.get('classification')=='singleface' and branch.get('unique_nearzero_face_entities')==['y:4:3'] and not branch.get('unique_near_tie_slope_entities'))
  outward=slopes[1]>budgets[1] and slopes[-1]<-budgets[-1]
  eps=ROUNDING_MULT*torch.finfo(c.dtype).eps*max(float(tg[-1].norm()),float(tg[1].norm()),torch.finfo(c.dtype).tiny)
  return {'control':c,'control_sha256':tensor_sha(c),'native_J':float(problem.objective(c,parameters)),'branch':branch,'singleface_supported':supported,'q_value':float(qvalue(c)),'normal_l2':float(qn),'theta':float(theta),'theta_strict_interior':1e-8<theta<1-1e-8,'gradient_inf':ginf,'tangent_gradient_inf_by_side':{str(s):float(tg[s].abs().max()) for s in (-1,1)},'tangent_gradient_agreement_inf':agree,'tangent_agreement_budget':eps,'normal_slopes':{str(s):slopes[s] for s in (-1,1)},'normal_budgets':{str(s):budgets[s] for s in (-1,1)},'strict_outward':outward,'first_order_pass':ginf<STATIONARITY_TOL and agree<=eps,'mixed_gradient_inf':float(mix.abs().max())}
 # Reuse the exact cycle-1 H columns at this exact point. No near-face/full H reuse.
 old_cycle=prior['cycle_records'][-1]
 if old_cycle['point_sha256']!=tensor_sha(control) or not old_cycle.get('hessian_columns') or len(old_cycle['hessian_columns'])!=NT:
  raise ValueError('current point has no complete exact chart-H checkpoint')
 H=torch.tensor(old_cycle['hessian_columns'],dtype=torch.float64).T.contiguous()
 if tensor_sha(H)!=old_cycle.get('hessian_sha256'):
  raise ValueError('saved current-point chart-H digest mismatch')
 hvp_started=int(prior['current_hvp_started']);hvp_returned=int(prior['current_hvp_returned']);
 state={'phase':'loaded_exact_current_hessian','status':'in_progress','control':control.tolist(),'control_sha256':tensor_sha(control),'native_J':float(problem.objective(control,parameters)),'point_hessian':H.tolist(),'point_hessian_sha256':tensor_sha(H),'point_hessian_source_cycle':old_cycle['cycle'],'point_hessian_eigenvalues':old_cycle['hessian_eigenvalues'],'point_hessian_symmetry_relative':old_cycle['hessian_symmetry_relative'],'point_hessian_columns_reused':NT,'hvp_started_total':hvp_started,'hvp_returned_total':hvp_returned,'new_hvp_started':0,'new_hvp_returned':0,'accepted_steps':[],'cycles':[],'ledger':LEDGER,'continuation_plan_sha256':PLAN_SHA,'source_transition':transition,'source_hashes_before':source_before,'input_parameters_sha256':tensor_sha(parameters),'fixed_input_identity':qualifier.cycle.shared._input_identity(problem,original,control,parameters,truth),'audit_script_sha256_before':script_sha_before}
 def save(phase:str,**extra:Any)->None:
  c=lift(y);state.update(phase=phase,control=c.tolist(),control_sha256=tensor_sha(c),native_J=float(problem.objective(c,parameters)),hvp_started_total=hvp_started,hvp_returned_total=hvp_returned,new_hvp_started=state.get('new_hvp_started',0),new_hvp_returned=state.get('new_hvp_returned',0),elapsed_s=time.monotonic()-started)
  state.update(extra);write(OUTPUT,state)
 save('loaded_exact_current_hessian')
 current=diagnose(y);state['cycles'].append({'point':{k:v for k,v in current.items() if k not in ('control',)},'modified_hessian':{'eigenvalues':old_cycle['hessian_eigenvalues'],'symmetry_relative':old_cycle['hessian_symmetry_relative'],'sha256':tensor_sha(H),'source':'exact complete chart-H checkpoint at same control'}})
 save('current_point_diagnosed',current_point_diagnostics=state['cycles'][-1]['point'])
 eye=torch.eye(NT,dtype=y.dtype)
 for correction_index in range(2):
  if not current['singleface_supported'] or not current['strict_outward'] or not current['theta_strict_interior']:
   state['status']='unsupported_face_or_outward_gate';save('face_qualification_gate',current_point_diagnostics=state['cycles'][-1]['point']);break
  if current['first_order_pass']:
   state['status']='stationary_face_root';save('stationary_face_root',current_point_diagnostics=state['cycles'][-1]['point']);break
  hs=(H+H.T)/2; eig=torch.linalg.eigvalsh(hs); hnorm=float(torch.linalg.matrix_norm(hs))
  floor=max(qualifier.blocks.CURVATURE_REL_TOL*float(eig.abs().max()),ROUNDING_MULT*torch.finfo(y.dtype).eps*max(hnorm,torch.finfo(y.dtype).tiny))
  target=max(floor,1e-3*hnorm);mu=max(0.,target-float(eig[0]))+ROUNDING_MULT*torch.finfo(y.dtype).eps*max(hnorm,torch.finfo(y.dtype).tiny)
  shifted=hs+mu*torch.eye(NT,dtype=y.dtype);shift_eig=torch.linalg.eigvalsh(shifted)
  gf=torch.func.grad(lambda u,p:chart_obj(u,p,1),argnums=0);g=gf(y,parameters)
  step=torch.linalg.solve(shifted,-g); slope=float(torch.dot(g,step)); stepnorm=float(torch.linalg.vector_norm(step))
  row={'correction_index':correction_index,'point_sha256':current['control_sha256'],'native_J':current['native_J'],'gradient_inf':current['gradient_inf'],'original_H_eigenvalues':eig.tolist(),'original_H_lambda_min':float(eig[0]),'original_H_symmetry_relative':float(torch.linalg.matrix_norm(H-H.T)/max(hnorm,torch.finfo(y.dtype).tiny)),'original_H_floor':floor,'spectral_target_min_eigenvalue':target,'spectral_shift_mu':mu,'shifted_H_eigenvalues':shift_eig.tolist(),'shifted_H_condition_estimate':float(shift_eig[-1]/shift_eig[0]),'modified_direction_norm':stepnorm,'control_norm':float(torch.linalg.vector_norm(lift(y))),'newton_slope_g_dot_step':slope,'line_search':[],'hessian_columns_source':'reused exact point H' if correction_index==0 else 'fresh H at preceding accepted point'}
  state['cycles'][-1]['modified_hessian'].update({'shift_mu':mu,'shifted_eigenvalues':shift_eig.tolist(),'shifted_condition_estimate':row['shifted_H_condition_estimate'],'modified_step_norm':stepnorm,'g_dot_step':slope})
  if not math.isfinite(slope) or slope>=0: state['status']='modified_newton_not_descent';row['status']=state['status'];state['cycles'][-1]['modified_step']=row;save('modified_direction_refused');break
  base_j=current['native_J']; accepted=False
  for backtrack in range(30):
   if time.monotonic()>=deadline: state['status']='cumulative_budget_exhausted';row['status']=state['status'];state['cycles'][-1]['modified_step']=row;save('line_search_budget_exhausted');break
   alpha=0.5**backtrack;trial=y+alpha*step;trial_control=lift(trial);lower_bound=0.5*float(torch.dot(trial_control,trial_control));armijo_rhs=base_j+1e-4*alpha*slope
   trial_record={'alpha':alpha,'control_sha256':tensor_sha(trial_control),'control_norm':float(torch.linalg.vector_norm(trial_control)),'identity_prior_lower_bound':lower_bound,'armijo_rhs':armijo_rhs}
   if lower_bound>armijo_rhs:
    trial_record.update(native_J=None,status='rejected_by_identity_prior_lower_bound')
   else:
    trial_value=float(chart_obj(trial,parameters,1));trial_record['native_J']=trial_value;trial_record['armijo_passed']=trial_value<=armijo_rhs;trial_record['status']='accepted' if trial_value<=armijo_rhs else 'native_J_Armijo_rejected'
   row['line_search'].append(trial_record);state['cycles'][-1]['modified_step']=row;save('native_J_armijo_trial',correction_index=correction_index,backtrack=backtrack)
   if trial_record['status']=='accepted':
    y=trial;accepted=True;break
  if state.get('status')=='cumulative_budget_exhausted':break
  if not accepted:
   state['status']='native_J_armijo_grid_exhausted';row['status']=state['status'];save('native_J_armijo_exhausted');break
  state['accepted_steps'].append({'correction_index':correction_index,'alpha':alpha,'native_J':trial_record['native_J'],'control_sha256':tensor_sha(lift(y)),'step_norm':alpha*stepnorm})
  current=diagnose(y);row['status']='accepted';row['accepted_alpha']=alpha;row['accepted_native_J']=current['native_J'];state['cycles'][-1]['modified_step']=row
  state['cycles'].append({'point':{k:v for k,v in current.items() if k!='control'},'modified_hessian':None});save('accepted_modified_newton_step',current_point_diagnostics=state['cycles'][-1]['point'])
  if not current['singleface_supported'] or not current['strict_outward'] or not current['theta_strict_interior']:
   state['status']='unsupported_after_accepted_step';save('post_step_face_gate');break
  if current['first_order_pass']:
   state['status']='stationary_face_root';save('stationary_face_root');break
  if correction_index==1:
   state['status']='two_corrections_exhausted_without_stationarity';save('bounded_corrections_exhausted');break
  # Fresh 25-column exact chart Hessian at this new, branch-audited point.
  cols=[];gradient_fn=lambda z:torch.func.grad(lambda zz,p:chart_obj(zz,p,1),argnums=0)(z,parameters)
  for j in range(NT):
   if time.monotonic()>=deadline:state['status']='cumulative_budget_exhausted_during_H';save('fresh_H_budget_exhausted',pending_column=j);break
   state['new_hvp_started']+=1;hvp_started+=1;save('fresh_H_column_started',cycle=correction_index+1,column=j)
   col=torch.func.jvp(gradient_fn,(y,),(eye[j],))[1]
   cols.append(col);state['new_hvp_returned']+=1;hvp_returned+=1
   state['cycles'][-1].setdefault('hessian_columns',[]).append(col.tolist());save('fresh_H_column_returned',cycle=correction_index+1,column=j)
  if state['status']=='cumulative_budget_exhausted_during_H':break
  H=torch.stack(cols,dim=1);sy=(H+H.T)/2;ee=torch.linalg.eigvalsh(sy);hn=float(torch.linalg.matrix_norm(sy));sym=float(torch.linalg.matrix_norm(H-H.T)/max(hn,torch.finfo(H.dtype).tiny));fl=max(qualifier.blocks.CURVATURE_REL_TOL*float(ee.abs().max()),ROUNDING_MULT*torch.finfo(H.dtype).eps*max(hn,torch.finfo(H.dtype).tiny));state['cycles'][-1]['modified_hessian']={'eigenvalues':ee.tolist(),'lambda_min':float(ee[0]),'symmetry_relative':sym,'curvature_floor':fl,'hessian_sha256':tensor_sha(H),'hessian_columns':NT};save('fresh_H_complete')
 # If a stationary point was reached, compute both exact same-point incident-face
 # chart Hessians. A minimum claim requires both sides positive definite.
 if state['status']=='stationary_face_root':
  current=diagnose(y)
  if not current['first_order_pass'] or not current['singleface_supported'] or not current['strict_outward'] or not current['theta_strict_interior']:
   state['status']='stationary_gate_failed';save('final_first_order_gate');
  else:
   p2={};
   for side in (-1,1):
    gf=lambda z,s=side:torch.func.grad(lambda zz,p:chart_obj(zz,p,s),argnums=0)(z,parameters);cols=[]
    for j in range(NT):
     if time.monotonic()>=deadline:state['status']='cumulative_budget_exhausted_final_P2';save('final_P2_partial',side=side,column=j,completed_by_side={str(k):len(v.get('eigenvalues',[])) for k,v in p2.items()});break
     state['new_hvp_started']+=1;hvp_started+=1;save('final_P2_column_started',side=side,column=j)
     col=torch.func.jvp(gf,(y,),(eye[j],))[1];cols.append(col);state['new_hvp_returned']+=1;hvp_returned+=1;save('final_P2_column_returned',side=side,column=j)
    if state['status']=='cumulative_budget_exhausted_final_P2':break
    hh=torch.stack(cols,dim=1);symh=(hh+hh.T)/2;ev=torch.linalg.eigvalsh(symh);nn=float(torch.linalg.matrix_norm(symh));symerr=float(torch.linalg.matrix_norm(hh-hh.T)/max(nn,torch.finfo(hh.dtype).tiny));floor=max(qualifier.blocks.CURVATURE_REL_TOL*float(ev.abs().max()),ROUNDING_MULT*torch.finfo(hh.dtype).eps*max(nn,torch.finfo(hh.dtype).tiny));p2[str(side)]={'eigenvalues':ev.tolist(),'lambda_min':float(ev[0]),'symmetry_relative':symerr,'curvature_floor':floor,'positive_definite':bool(symerr<=qualifier.blocks.SYMMETRY_TOL and ev[0]>floor),'hessian_sha256':tensor_sha(hh)};save('final_P2_side_complete',side=side,P2_by_side=p2)
   if state['status']=='stationary_face_root':
    state['P2_by_side']=p2;state['minimum_claim']=len(p2)==2 and all(v['positive_definite'] for v in p2.values());state['status']='qualified_original_native_J_singleface_minimum' if state['minimum_claim'] else 'stationary_face_P2_failed';save('final_qualification',P2_by_side=p2,minimum_claim=state['minimum_claim'])
 _,source_after,_=qualifier._execution_plan(continuation_plan_path=PLAN,expected_continuation_sha256=PLAN_SHA)
 input_after=qualifier.cycle.shared._input_identity(problem,original,lift(y),parameters,truth)
 state['source_archive_closure']=source_before==source_after;state['source_hashes_after']=source_after;state['input_after']=input_after;state['final_parameters_sha256']=tensor_sha(parameters);state['script_sha256_after']=sha_bytes(SELF.read_bytes());state['script_unchanged']=script_sha_before==state['script_sha256_after'];state['elapsed_s']=time.monotonic()-started;state['ledger']['this_run_elapsed_s']=state['elapsed_s'];state['ledger']['charged_at_end_s']=LEDGER['prior_charged_s']+LEDGER['postresume_preflight_s']+state['elapsed_s'];state['ledger']['remaining_after_run_s']=LEDGER['original_cap_s']-state['ledger']['charged_at_end_s'];state['ledger']['unused_forecast_verification_reserve_s']=LEDGER['reserved_forecast_and_verification_s'];write(OUTPUT,state)
 return state
if __name__=='__main__':
 result=run();print(json.dumps({k:result.get(k) for k in ('status','control_sha256','native_J','hvp_started_total','hvp_returned_total','new_hvp_started','new_hvp_returned','accepted_steps','minimum_claim','elapsed_s','ledger')},indent=2))
