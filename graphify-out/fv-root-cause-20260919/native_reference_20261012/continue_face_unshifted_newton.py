"""Continue actual native-J minimization with exact face-chart Newton steps.

Uses unshifted Newton whenever the fresh reduced Hessian passes the resolved
SPD test; for an indefinite Hessian only, a spectral shift produces a descent
step. All trial acceptance is measured on the original objective restricted to
the already observed qy[4,3]=0 chart. No physics or source-plan changes occur.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, sys, time
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
OUTDIR=HERE/'continuation3/attempt1'; INPUT=OUTDIR/'modified_newton_face_qualification_attempt2.json'
OUTPUT=OUTDIR/'unshifted_newton_face_qualification.json'; PLAN=HERE/'continuation3/CONTINUATION_PLAN.json'
PLAN_SHA='67fa44ac8c6c08f79df99a9fe2c5bee10b385a7769bb65ad6caafafeb9c52a01';SELF=Path(__file__)
spec=importlib.util.spec_from_file_location('native_qualifier',HERE/'qualify_candidate.py')
if spec is None or spec.loader is None:raise RuntimeError('native qualifier is unavailable')
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
from advar import transport
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart,face_weights_from_basis
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
FACE=('y',4,3);NC,NT=26,25;TOL=1e-10;RM=128.

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def tsha(x:torch.Tensor)->str:return sha(x.detach().contiguous().cpu().numpy().tobytes())
def write(p:Path,d:dict[str,Any])->None:
 def default(v:Any)->Any:
  if isinstance(v,torch.Tensor):return v.detach().contiguous().cpu().tolist()
  raise TypeError(f'cannot serialize {type(v).__name__}')
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False,default=default)+'\n');tmp.replace(p)

def run()->dict[str,Any]:
 if OUTPUT.exists():raise ValueError('output must be fresh')
 prior=json.loads(INPUT.read_text()); control=torch.tensor(prior['control'],dtype=torch.float64)
 if tsha(control)!=prior['control_sha256'] or prior['status']!='two_corrections_exhausted_without_stationarity':raise ValueError('input is not the verified current point')
 ledger_prior=prior['ledger']; cumulative=float(ledger_prior['charged_at_end_s']); remaining=float(ledger_prior['remaining_after_run_s']);reserve=float(ledger_prior['unused_forecast_verification_reserve_s'])
 run_limit=max(0.,remaining-reserve); start=time.monotonic();deadline=start+run_limit;script_before=sha(SELF.read_bytes())
 case=q.cycle.load_fixed_case();problem,original,truth,p=case['problem'],case['original'],case['truth'],case['parameters'];fs=problem.frozen.fv_transport
 _,src_before,transition=q._execution_plan(continuation_plan_path=PLAN,expected_continuation_sha256=PLAN_SHA)
 if transition!=prior['source_transition'] or tsha(p)!=prior['input_parameters_sha256'] or sha(PLAN.read_bytes())!=PLAN_SHA:raise ValueError('pinned input/plan transition mismatch')
 if sha((ROOT/'src/advar/transport.py').read_bytes())!='c9dde24c12237d5ffee6b8ca0766b4d5619abf6110f3e858ee447bf49c1f7da9':raise ValueError('approved transport source transition changed')
 weights=face_weights_from_basis(fs.psi_basis,'y',(4,3));nw=weights*fs.coefficient_limits*(1-torch.tanh(control[20:25]).square());pivot_i=int(nw.abs().argmax());chart=FVFaceFluxCoordinateChart.from_basis(field_count=20,growth_count=1,coefficient_limits=fs.coefficient_limits,psi_basis=fs.psi_basis,axis='y',face=(4,3),pivot_index=pivot_i)
 coords=chart.to_face_coordinates(control);pivot=chart.layout['face_flux_coordinate'];coords[pivot]=0;y=torch.cat((coords[:pivot],coords[pivot+1:]));control=chart.from_face_coordinates(coords)
 def lift(z):return chart.from_face_coordinates(torch.cat((z[:pivot],z.new_zeros(1),z[pivot:])))
 def qv(c):
  psi=torch.einsum('k,kij->ij',fs.coefficient_limits*torch.tanh(c[20:25]),fs.psi_basis);return transport.face_volume_fluxes(psi)[1][4,3]
 def obj(z,side=1):
  with transport.selected_face_extension('y',4,3,side):return problem.objective(lift(z),p)
 def diag(z):
  c=lift(z);branch=q._analysis_branch(problem,c,p,deadline=deadline);full={};tg={}
  for side in (-1,1):
   with transport.selected_face_extension('y',4,3,side):full[side]=torch.func.grad(problem.objective,argnums=0)(c,p)
   tg[side]=torch.func.grad(lambda u,pp,s=side:obj(u,s),argnums=0)(z,p)
  n=torch.func.grad(qv)(c);nn=torch.linalg.vector_norm(n);sl={s:float(torch.dot(full[s],n)/nn) for s in (-1,1)};bud={s:RM*torch.finfo(c.dtype).eps*float(full[s].norm()) for s in (-1,1)}
  theta=tangent.minimum_mixture_weight(full[-1],full[1]);mix=(1-theta)*full[-1]+theta*full[1];agree=float((tg[-1]-tg[1]).abs().max());ginf=float(tg[1].abs().max());mixinf=float(mix.abs().max());agbudget=RM*torch.finfo(c.dtype).eps*max(float(tg[-1].norm()),float(tg[1].norm()),torch.finfo(c.dtype).tiny)
  single=(branch.get('classification')=='singleface' and branch.get('unique_nearzero_face_entities')==['y:4:3'] and not branch.get('unique_near_tie_slope_entities'));outward=sl[1]>bud[1] and sl[-1]<-bud[-1]
  return {'control':c,'control_sha256':tsha(c),'native_J':float(problem.objective(c,p)),'branch':branch,'singleface_supported':single,'face_q':float(qv(c)),'normal_l2':float(nn),'theta':float(theta),'strict_theta':1e-8<theta<1-1e-8,'gradient_inf':ginf,'mixed_gradient_inf':mixinf,'tangent_gradient_inf_by_side':{str(s):float(tg[s].abs().max()) for s in (-1,1)},'tangent_gradient_agreement_inf':agree,'tangent_agreement_budget':agbudget,'normal_slopes':{str(s):sl[s] for s in (-1,1)},'normal_budgets':{str(s):bud[s] for s in (-1,1)},'strict_outward':outward,'first_order_pass':ginf<TOL and mixinf<TOL and agree<=agbudget}
 state={'status':'in_progress','phase':'initial_point','control':lift(y).tolist(),'control_sha256':tsha(lift(y)),'native_J':float(problem.objective(lift(y),p)),'accepted_steps':[],'iterations':[],'hvp_started_total':prior['hvp_started_total'],'hvp_returned_total':prior['hvp_returned_total'],'new_hvp_started':0,'new_hvp_returned':0,'ledger':{'original_cap_s':3600.,'charged_before_this_run_s':cumulative,'estimated_remaining_before_this_run_s':remaining,'reserved_forecast_and_verification_s':reserve,'run_limit_s':run_limit},'continuation_plan_sha256':PLAN_SHA,'source_transition':transition,'source_hashes_before':src_before,'input_parameters_sha256':tsha(p),'prior_checkpoint_sha256':sha(INPUT.read_bytes()),'audit_script_sha256_before':script_before}
 calls_started=int(state['hvp_started_total']);calls_returned=int(state['hvp_returned_total'])
 def save(phase,**extra):
  state.update(phase=phase,control=lift(y).tolist(),control_sha256=tsha(lift(y)),native_J=float(problem.objective(lift(y),p)),hvp_started_total=calls_started,hvp_returned_total=calls_returned,new_hvp_started=state['new_hvp_started'],new_hvp_returned=state['new_hvp_returned'],elapsed_s=time.monotonic()-start)
  state.update(extra);write(OUTPUT,state)
 current=diag(y);state['iterations'].append({'point':{k:v for k,v in current.items() if k!='control'}});save('initial_point_audit')
 eye=torch.eye(NT,dtype=y.dtype);max_cycles=8;last_plus_H=None
 for cycle in range(max_cycles):
  if not current['singleface_supported'] or not current['strict_outward'] or not current['strict_theta']:
   state['status']='unsupported_face_or_outward_gate';save('active_face_gate');break
  if current['first_order_pass']:
   state['status']='stationary_face_root';save('stationary_face_root');break
  # Keep enough cumulative time for this exact H and the opposite-side P2 H.
  if deadline-time.monotonic()<520:
   state['status']='budget_reserved_for_final_P2';save('budget_reserve_gate');break
  gf=lambda z:torch.func.grad(lambda zz,pp:obj(zz,1),argnums=0)(z,p);Hcols=[];iterrow={'cycle':cycle,'point_sha256':current['control_sha256'],'native_J':current['native_J'],'gradient_inf':current['gradient_inf'],'mixed_gradient_inf':current['mixed_gradient_inf'],'hvp_columns_started':0,'hvp_columns_returned':0,'hessian_columns':[]}
  state['iterations'][-1]['fresh_hessian']=iterrow;save('fresh_H_started',cycle=cycle)
  for j in range(NT):
   if deadline-time.monotonic()<520:
    state['status']='budget_reserved_during_H';save('fresh_H_budget_reserve',cycle=cycle,pending_column=j);break
   iterrow['hvp_columns_started']+=1;calls_started+=1;state['new_hvp_started']+=1;save('fresh_H_column_started',cycle=cycle,column=j)
   col=torch.func.jvp(gf,(y,),(eye[j],))[1];Hcols.append(col);iterrow['hvp_columns_returned']+=1;state['new_hvp_returned']+=1;calls_returned+=1;iterrow['hessian_columns'].append(col.tolist());save('fresh_H_column_returned',cycle=cycle,column=j)
  if state['status']=='budget_reserved_during_H':break
  H=torch.stack(Hcols,dim=1);Hs=(H+H.T)/2;eig=torch.linalg.eigvalsh(Hs);hn=float(torch.linalg.matrix_norm(Hs));sym=float(torch.linalg.matrix_norm(H-H.T)/max(hn,torch.finfo(H.dtype).tiny));floor=max(q.blocks.CURVATURE_REL_TOL*float(eig.abs().max()),RM*torch.finfo(H.dtype).eps*max(hn,torch.finfo(H.dtype).tiny));spd=sym<=q.blocks.SYMMETRY_TOL and float(eig[0])>floor
  iterrow.update(hessian_sha256=tsha(H),eigenvalues=eig.tolist(),lambda_min=float(eig[0]),symmetry_relative=sym,positive_floor=floor,positive_definite=spd);last_plus_H=H;save('fresh_H_complete',cycle=cycle)
  if sym>q.blocks.SYMMETRY_TOL:state['status']='hessian_symmetry_refused';save('H_symmetry_gate');break
  # Use exact Newton on resolved SPD H. Only an indefinite H receives a
  # spectral shift, with positive floor 1e-3 times the local spectral scale.
  hscale=max(hn,torch.finfo(H.dtype).tiny);target=max(floor,1e-3*hscale);mu=0. if spd else max(0.,target-float(eig[0]))+RM*torch.finfo(H.dtype).eps*hscale
  Hmod=Hs+mu*torch.eye(NT,dtype=Hs.dtype);meig=torch.linalg.eigvalsh(Hmod);g=gf(y);step=torch.linalg.solve(Hmod,-g);slope=float(torch.dot(g,step));iterrow.update(spectral_shift_mu=mu,modified_eigenvalues=meig.tolist(),modified_condition_estimate=float(meig[-1]/meig[0]),step_norm=float(torch.linalg.vector_norm(step)),control_norm=float(torch.linalg.vector_norm(lift(y))),g_dot_step=slope,direction_kind='exact_newton' if mu==0 else 'spectrally_shifted_modified_newton',line_search=[]);save('modified_direction_ready',cycle=cycle)
  if not math.isfinite(slope) or slope>=0:state['status']='modified_direction_not_descent';save('descent_gate');break
  base=current['native_J'];accepted=False
  for bt in range(32):
   if deadline-time.monotonic()<520:state['status']='budget_reserved_during_line_search';save('line_search_budget_reserve',cycle=cycle,pending_trial=bt);break
   alpha=.5**bt;trial=y+alpha*step;ct=lift(trial);rhs=base+1e-4*alpha*slope;lower=.5*float(torch.dot(ct,ct));tr={'alpha':alpha,'control_sha256':tsha(ct),'control_norm':float(torch.linalg.vector_norm(ct)),'identity_prior_lower_bound':lower,'armijo_rhs':rhs}
   if lower>rhs:tr.update(native_J=None,status='identity_prior_bound_reject')
   else:
    val=float(obj(trial,1));tr['native_J']=val;tr['armijo_passed']=val<=rhs;tr['status']='accepted' if val<=rhs else 'native_J_Armijo_reject'
   iterrow['line_search'].append(tr);save('native_J_Armijo_trial',cycle=cycle,trial=bt)
   if tr['status']=='accepted':y=trial;accepted=True;state['accepted_steps'].append({'cycle':cycle,'alpha':alpha,'native_J':tr['native_J'],'control_sha256':tsha(lift(y)),'step_norm':alpha*iterrow['step_norm']});break
  if state['status'] in ('budget_reserved_during_line_search',):break
  if not accepted:state['status']='native_J_Armijo_grid_exhausted';save('Armijo_exhausted');break
  current=diag(y);state['iterations'].append({'point':{k:v for k,v in current.items() if k!='control'}});save('accepted_step_point_audit',cycle=cycle)
  if cycle==max_cycles-1 and not current['first_order_pass']:state['status']='bounded_cycle_limit_without_stationarity';save('cycle_limit');break
  if not current['singleface_supported'] or not current['strict_outward'] or not current['strict_theta']:
   state['status']='unsupported_after_step';save('post_step_active_face_gate');break
 # Final same-point two-sided reduced curvature; reuse + side only if exact at this same point.
 if state['status']=='stationary_face_root':
  current=diag(y)
  if not current['singleface_supported'] or not current['strict_outward'] or not current['strict_theta'] or not current['first_order_pass']:
   state['status']='final_first_order_gate_failed';save('final_first_order_gate')
  else:
   metrics={};
   for side in (-1,1):
    gf=lambda z,s=side:torch.func.grad(lambda zz,pp:obj(zz,s),argnums=0)(z,p);cols=[]
    if side==1 and last_plus_H is not None and state['iterations'][-1]['point']['control_sha256']==current['control_sha256']:
     cols=[last_plus_H[:,j] for j in range(NT)];state['P2_reused_samepoint_plus_H']=True
    for j in range(len(cols),NT):
     if deadline-time.monotonic()<30:state['status']='budget_exhausted_final_P2';save('final_P2_partial',side=side,pending_column=j);break
     calls_started+=1;calls_returned+=1;state['new_hvp_started']+=1;state['new_hvp_returned']+=1;save('final_P2_column_started',side=side,column=j)
     cols.append(torch.func.jvp(gf,(y,),(eye[j],))[1]);save('final_P2_column_returned',side=side,column=j)
    if state['status']=='budget_exhausted_final_P2':break
    HH=torch.stack(cols,dim=1);SS=(HH+HH.T)/2;ee=torch.linalg.eigvalsh(SS);nn=float(torch.linalg.matrix_norm(SS));se=float(torch.linalg.matrix_norm(HH-HH.T)/max(nn,torch.finfo(HH.dtype).tiny));fl=max(q.blocks.CURVATURE_REL_TOL*float(ee.abs().max()),RM*torch.finfo(HH.dtype).eps*max(nn,torch.finfo(HH.dtype).tiny));metrics[str(side)]={'hessian_sha256':tsha(HH),'eigenvalues':ee.tolist(),'lambda_min':float(ee[0]),'symmetry_relative':se,'positive_floor':fl,'positive_definite':se<=q.blocks.SYMMETRY_TOL and float(ee[0])>fl};save('final_P2_side_complete',side=side,P2_by_side=metrics)
   if state['status']=='stationary_face_root':state['P2_by_side']=metrics;state['minimum_claim']=len(metrics)==2 and all(v['positive_definite'] for v in metrics.values());state['status']='qualified_original_native_J_singleface_minimum' if state['minimum_claim'] else 'stationary_face_P2_failed';save('final_qualification',P2_by_side=metrics,minimum_claim=state['minimum_claim'],final_point_diagnostics={k:v for k,v in current.items() if k!='control'},native_J_recheck=float(problem.objective(lift(y),p)))
 _,src_after,_=q._execution_plan(continuation_plan_path=PLAN,expected_continuation_sha256=PLAN_SHA);identity_after=q.cycle.shared._input_identity(problem,original,lift(y),p,truth)
 state['source_archive_closure']=src_before==src_after;state['source_hashes_after']=src_after;state['input_after']=identity_after;state['script_sha256_after']=sha(SELF.read_bytes());state['script_unchanged']=script_before==state['script_sha256_after'];state['elapsed_s']=time.monotonic()-start;state['ledger'].update(run_elapsed_s=state['elapsed_s'],charged_at_end_s=cumulative+state['elapsed_s'],remaining_after_run_s=3600.-cumulative-state['elapsed_s'],reserve_preserved_s=max(0.,3600.-cumulative-state['elapsed_s']-reserve));write(OUTPUT,state);return state
if __name__=='__main__':
 r=run();print(json.dumps({k:r.get(k) for k in ('status','control_sha256','native_J','hvp_started_total','hvp_returned_total','new_hvp_started','new_hvp_returned','accepted_steps','minimum_claim','elapsed_s','ledger')},indent=2))
