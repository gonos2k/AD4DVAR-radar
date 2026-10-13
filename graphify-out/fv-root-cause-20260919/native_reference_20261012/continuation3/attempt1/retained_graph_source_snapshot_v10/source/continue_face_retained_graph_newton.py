"""Retained-graph exact reduced-H refinement under the frozen remaining A budget.

The only model-profile change is ``FVAnalysisTransport.replay: True -> False``
on a shallow computational copy. It changes checkpoint recomputation/tape
retention, not the FV substeps or source code. The original replay=True problem
remains the authority for native-J values, active-branch audits and outward
cost gradients. Retained-graph reverse rows provide exact reduced Hessians for
both incident chart extensions; final P2 is tested on the same point.
"""
from __future__ import annotations
import copy,hashlib,importlib.util,json,math,resource,sys,time
from dataclasses import replace
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=HERE/'continuation3/attempt1'
INPUT=OUT/'fixed_metric_face_qualification.json';BASE=OUT/'modified_newton_face_qualification_attempt2.json';PROBE=OUT/'replay_off_equivalence_probe.json'
QUAL_PLAN=HERE/'QUALIFICATION_CONTINUATION_PLAN_v10.json';CONT_PLAN=HERE/'continuation3/CONTINUATION_PLAN.json';OUTPUT=OUT/'retained_graph_face_qualification.json';SELF=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location('native_qualifier',HERE/'qualify_candidate.py')
if spec is None or spec.loader is None:raise RuntimeError('native qualifier unavailable')
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
from advar import transport
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart,face_weights_from_basis
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
NT=25;TOL=1e-10;RM=128.;FACE=('y',4,3);RSS_CAP=2*1024**3

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def tsha(x:torch.Tensor)->str:return sha(x.detach().contiguous().cpu().numpy().tobytes())
def rss_bytes()->int:
 value=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 return value if sys.platform=='darwin' else value*1024
def write(path:Path,state:dict[str,Any])->None:
 def default(v:Any)->Any:
  if isinstance(v,torch.Tensor):return v.detach().contiguous().cpu().tolist()
  raise TypeError(f'cannot serialize {type(v).__name__}')
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(state,indent=2,sort_keys=True,allow_nan=False,default=default)+'\n');tmp.replace(path)
def run()->dict[str,Any]:
 if OUTPUT.exists():raise ValueError('retained-graph output must be fresh')
 plan=json.loads(QUAL_PLAN.read_text());plan_sha=sha(QUAL_PLAN.read_bytes());inp=json.loads(INPUT.read_text());inp_sha=sha(INPUT.read_bytes());base=json.loads(BASE.read_text());base_sha=sha(BASE.read_bytes());probe=json.loads(PROBE.read_text());probe_sha=sha(PROBE.read_bytes())
 if plan.get('status')!='frozen' or plan.get('experiment_kind')!='native_J_same_face_retained_graph_newton':raise ValueError('retained-graph phase plan is not frozen')
 for name,digest in plan['source_files'].items():
  if sha((ROOT/name).read_bytes())!=digest:raise ValueError(f'source pin mismatch: {name}')
 for path,digest in ((INPUT,inp_sha),(BASE,base_sha),(PROBE,probe_sha)):
  if plan['inputs'].get(path.relative_to(ROOT).as_posix())!=digest:raise ValueError(f'input receipt changed: {path.name}')
 if inp.get('status')!='native_J_value_cap_exhausted' or inp.get('new_HVP_started')!=0 or inp.get('control_sha256')!=plan['input_control_sha256']:raise ValueError('current endpoint is not the frozen nonstationary control')
 started=time.monotonic();deadline=started+float(plan['budget']['phase_wall_cap_seconds']);native_cap=int(plan['budget']['cumulative_native_J_value_cap']);row_cap=int(plan['budget']['retained_Hessian_row_cap']);native_calls=int(inp['native_J_value_calls']);native_started=native_calls;grad_calls=0;row_started=row_returned=0;hvp_calls=0;profile_scalar_calls=0
 control=torch.tensor(inp['control'],dtype=torch.float64)
 if tsha(control)!=inp['control_sha256']:raise ValueError('fixed control bits changed')
 case=q.cycle.load_fixed_case();problem,original,truth,p=case['problem'],case['original'],case['truth'],case['parameters'];fv=problem.frozen.fv_transport
 _,source_before,transition=q._execution_plan(continuation_plan_path=CONT_PLAN,expected_continuation_sha256=plan['continuation_plan']['sha256'])
 if transition!=inp['source_transition'] or tsha(p)!=inp['input_parameters_sha256'] or not inp.get('source_archive_closure') or not inp.get('source_unchanged'):raise ValueError('original replay=True input/source closure changed')
 if q.cycle.shared._input_identity(problem,original,control,p,truth)!=inp['input_after']:raise ValueError('fixed original input identity changed')
 if fv is None or fv.replay is not True:raise ValueError('immutable original problem is not replay=True')
 # A shallow computational copy retains every physical tensor and derived
 # frozen field; only the public recomputation flag changes.
 fv_profile=replace(fv,replay=False);frozen_profile=copy.copy(problem.frozen);object.__setattr__(frozen_profile,'fv_transport',fv_profile)
 profile_problem=copy.copy(problem);object.__setattr__(profile_problem,'frozen',frozen_profile)
 physical_profile_identity={'same_basis_bytes':tsha(fv_profile.psi_basis)==tsha(fv.psi_basis),'same_limits_bytes':tsha(fv_profile.coefficient_limits)==tsha(fv.coefficient_limits),'same_boundary_object_ids':fv_profile.boundary_echo is fv.boundary_echo and fv_profile.boundary_support is fv.boundary_support,'same_substeps':fv_profile.substeps_per_interval==fv.substeps_per_interval,'same_spacing':fv_profile.spacing_yx==fv.spacing_yx,'same_reconstruction':fv_profile.reconstruction==fv.reconstruction,'same_courant':fv_profile.max_courant==fv.max_courant,'original_replay':fv.replay,'profile_replay':fv_profile.replay,'same_control_parameters_truth':True,'meaning':'checkpointing/tape-retention profile only; no FV formula, substep, physical tensor, data or source change'}
 if not all(v for k,v in physical_profile_identity.items() if k.startswith('same_')):raise ValueError('replay profile changed a physical field')
 weights=face_weights_from_basis(fv.psi_basis,'y',(4,3));nw=weights*fv.coefficient_limits*(1-torch.tanh(control[20:25]).square());pivot_flow=int(nw.abs().argmax());chart=FVFaceFluxCoordinateChart.from_basis(field_count=20,growth_count=1,coefficient_limits=fv.coefficient_limits,psi_basis=fv.psi_basis,axis='y',face=(4,3),pivot_index=pivot_flow)
 coords=chart.to_face_coordinates(control);pivot_obj=chart.layout.get('face_flux_coordinate')
 if not isinstance(pivot_obj,int) or isinstance(pivot_obj,bool):raise ValueError('chart pivot index is not a fixed integer')
 pivot=pivot_obj;coords[pivot]=0.;y=torch.cat((coords[:pivot],coords[pivot+1:]));lift_control=chart.from_face_coordinates(torch.cat((y[:pivot],y.new_zeros(1),y[pivot:])))
 if tsha(lift_control)!=tsha(control):raise ValueError('chart roundtrip changed pinned current control')
 def lift(u):return chart.from_face_coordinates(torch.cat((u[:pivot],u.new_zeros(1),u[pivot:])))
 def profile_obj(u,side):
  with transport.selected_face_extension('y',4,3,side):return profile_problem.objective(lift(u),p)
 def native_value(c):
  nonlocal native_calls
  if native_calls>=native_cap:raise RuntimeError('cumulative original-native-J cap exhausted before callback')
  native_calls+=1;v=problem.objective(c,p)
  if v.ndim!=0 or not bool(torch.isfinite(v)):raise ValueError('original native J is nonfinite')
  return float(v)
 def branch_audit(c):
  nonlocal native_calls
  if native_calls>=native_cap:raise RuntimeError('cumulative native-J cap exhausted before branch callback')
  br=q._analysis_branch(problem,c,p,deadline=deadline)
  if br.get('objective_finite') is not True or br.get('native_J_on_audited_path') is None:raise ValueError('original native branch objective is nonfinite')
  native_calls+=1
  return br,float(br['native_J_on_audited_path'])
 def differential(u,branch,native_j):
  nonlocal grad_calls
  c=lift(u);full={};tangent_grad={}
  for side in (-1,1):
   grad_calls+=1
   with transport.selected_face_extension('y',4,3,side):full[side]=torch.func.grad(problem.objective,argnums=0)(c,p)
  chart_jac=torch.func.jacrev(lift)(u)
  for side in (-1,1):tangent_grad[side]=chart_jac.T@full[side]
  def qvalue(x):
   psi=torch.einsum('k,kij->ij',fv.coefficient_limits*torch.tanh(x[20:25]),fv.psi_basis);return transport.face_volume_fluxes(psi)[1][4,3]
  qg=torch.func.grad(qvalue)(c);qn=torch.linalg.vector_norm(qg);slopes={s:float(torch.dot(full[s],qg)/qn) for s in (-1,1)};budgets={s:RM*torch.finfo(c.dtype).eps*float(full[s].norm()) for s in (-1,1)};theta=tangent.minimum_mixture_weight(full[-1],full[1]);mixed=(1-theta)*full[-1]+theta*full[1];agree=float((tangent_grad[-1]-tangent_grad[1]).abs().max());ginf=float(tangent_grad[1].abs().max());minf=float(mixed.abs().max());ab=RM*torch.finfo(c.dtype).eps*max(float(tangent_grad[-1].norm()),float(tangent_grad[1].norm()),torch.finfo(c.dtype).tiny)
  single=(branch.get('classification')=='singleface' and branch.get('unique_nearzero_face_entities')==['y:4:3'] and not branch.get('unique_near_tie_slope_entities'));out=slopes[1]>budgets[1] and slopes[-1]<-budgets[-1]
  return {'control':c,'control_sha256':tsha(c),'native_J':float(native_j),'branch':branch,'singleface_supported':single,'face_q':float(qvalue(c)),'normal_l2':float(qn),'theta':float(theta),'theta_strict':1e-8<theta<1-1e-8,'gradient_inf':ginf,'mixed_gradient_inf':minf,'tangent_gradient_inf_by_side':{str(s):float(tangent_grad[s].abs().max()) for s in (-1,1)},'tangent_gradient_agreement_inf':agree,'tangent_agreement_budget':ab,'normal_slopes':{str(s):slopes[s] for s in (-1,1)},'normal_budgets':{str(s):budgets[s] for s in (-1,1)},'strict_outward':out,'first_order_pass':ginf<TOL and minf<TOL and agree<=ab,'tangent_gradient_plus':tangent_grad[1],'tangent_gradient_minus':tangent_grad[-1]}
 def plain_diag(row):return {k:v for k,v in row.items() if k not in ('control','tangent_gradient_plus','tangent_gradient_minus')}
 # Current original branch/J is reused from a hash-closed accepted-point
 # audit; all gradients below are recomputed at the exact same control.
 old_point=inp['iterations'][-1]['point'];old_branch=old_point['branch']
 if old_point['control_sha256']!=tsha(control) or old_point['singleface_supported'] is not True:raise ValueError('saved actual branch audit is not for current single-face control')
 state={'status':'in_progress','phase':'current_differentials','control':control.tolist(),'control_sha256':tsha(control),'native_J':float(inp['current_native_J']),'current_point':None,'accepted_steps':[],'iterations':[],'native_J_value_calls_before_phase':native_calls,'cumulative_native_J_cap':native_cap,'gradient_calls':grad_calls,'profile_scalar_calls':profile_scalar_calls,'retained_Hessian_rows_started':0,'retained_Hessian_rows_returned':0,'retained_Hessian_row_cap':row_cap,'HVP_calls':0,'previous_phase_HVP_calls':int(inp['new_HVP_started']),'physical_profile_identity':physical_profile_identity,'replay_equivalence_probe_sha256':probe_sha,'replay_equivalence_probe':probe['equivalence'],'memory_cap_bytes':RSS_CAP,'plan_sha256':plan_sha,'continuation_plan_sha256':plan['continuation_plan']['sha256'],'source_before':source_before,'source_transition':transition,'input_control_sha256':tsha(control),'input_parameters_sha256':tsha(p),'input_previous_receipt_sha256':inp_sha,'profile_delta':'transport.replay true→false only on shallow computational clone'}
 def save(phase,**extra):
  state.update(phase=phase,control=lift(y).tolist(),control_sha256=tsha(lift(y)),native_J=state.get('native_J'),native_J_value_calls=native_calls,gradient_calls=grad_calls,profile_scalar_calls=profile_scalar_calls,retained_Hessian_rows_started=row_started,retained_Hessian_rows_returned=row_returned,hvp_calls=hvp_calls,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=rss_bytes());state.update(extra);write(OUTPUT,state)
 y=y.clone();current=differential(y,old_branch,float(inp['current_native_J']));state['native_J']=current['native_J'];state['current_point']=plain_diag(current);save('current_point_differentials')
 if not current['singleface_supported'] or not current['strict_outward'] or not current['theta_strict']:state['status']='current_face_gate_failed';save('current_face_gate')
 else:
  # One current-point retained graph supplies exact g and H. Reverse rows are
  # checkpointed individually and share this one forward graph.
  def build_graph(u,side):
   nonlocal profile_scalar_calls,grad_calls
   if profile_scalar_calls>=int(plan['budget']['profile_scalar_call_cap']):raise RuntimeError('retained-profile scalar cap exhausted before graph forward')
   z=u.detach().clone().requires_grad_(True)
   with transport.selected_face_extension('y',4,3,side):phi=profile_problem.objective(lift(z),p)
   profile_scalar_calls+=1
   if phi.ndim!=0 or not bool(torch.isfinite(phi)):raise ValueError('retained-profile objective is not a finite scalar')
   grad_calls+=1;g=torch.autograd.grad(phi,z,create_graph=True,retain_graph=True)[0]
   return z,phi,g
  z,phi,g=build_graph(y,+1);state['profile_J_current']=float(phi);state['profile_J_vs_native_abs']=abs(float(phi)-current['native_J']);state['gradient_profile_vs_original_chart_inf']=float((g-current['tangent_gradient_plus']).abs().max());state['gradient_profile_vs_original_chart_budget']=max(q.blocks.SYMMETRY_TOL*max(float(g.norm()),float(current['tangent_gradient_plus'].norm()),torch.finfo(y.dtype).tiny),RM*torch.finfo(y.dtype).eps)
  tolJ=RM*torch.finfo(y.dtype).eps*max(1.,abs(current['native_J']))
  if state['profile_J_vs_native_abs']>tolJ or state['gradient_profile_vs_original_chart_inf']>state['gradient_profile_vs_original_chart_budget']:state['status']='replay_profile_current_J_or_gradient_mismatch';save('profile_value_gate')
  else:
   Hrows=[];eye=torch.eye(NT,dtype=y.dtype)
   for i in range(NT):
    if time.monotonic()>=deadline or rss_bytes()>RSS_CAP or row_started>=row_cap:state['status']='current_H_row_budget_or_RSS_exhausted';save('current_H_row_partial',row=i);break
    row_started+=1;save('current_H_row_started',row=i)
    row=torch.autograd.grad(g[i],z,retain_graph=i<NT-1)[0];Hrows.append(row);row_returned+=1;state.setdefault('current_H_rows',[]).append(row.tolist());save('current_H_row_returned',row=i)
   if len(Hrows)==NT:
    Hcur=torch.stack(Hrows,dim=0);Hs=(Hcur+Hcur.T)/2;eig=torch.linalg.eigvalsh(Hs);hn=float(torch.linalg.matrix_norm(Hs));sym=float(torch.linalg.matrix_norm(Hcur-Hcur.T)/max(hn,torch.finfo(y.dtype).tiny));floor=max(q.blocks.CURVATURE_REL_TOL*float(eig.abs().max()),RM*torch.finfo(y.dtype).eps*max(hn,torch.finfo(y.dtype).tiny));pd=sym<=q.blocks.SYMMETRY_TOL and float(eig[0])>floor;target=max(floor,1e-3*hn);mu=0. if pd else max(0.,target-float(eig[0]))+RM*torch.finfo(y.dtype).eps*hn;Hmod=Hs+mu*torch.eye(NT,dtype=y.dtype);chol=torch.linalg.cholesky(Hmod);step=-torch.cholesky_solve(g.detach()[:,None],chol).squeeze(1);slope=float(torch.dot(g.detach(),step));state['current_H']={'hessian_sha256':tsha(Hcur),'hessian':Hcur.tolist(),'eigenvalues':eig.tolist(),'lambda_min':float(eig[0]),'symmetry_relative':sym,'positive_floor':floor,'positive_definite':pd,'shift_mu':mu,'direction_norm':float(torch.linalg.vector_norm(step)),'g_dot_step':slope,'control_sha256':tsha(lift(y))};save('current_H_complete')
    if slope>=0:state['status']='current_retained_H_step_not_descent';save('current_descent_gate')
    else:
     reserve_final=(2*NT)*0.724+2*8.8+30.0
     for iteration in range(8):
      if current['first_order_pass']:state['status']='stationary_face_root';break
      if time.monotonic()>=deadline-reserve_final or native_calls+2>native_cap:state['status']='budget_reserved_for_final_two_sided_H';save('newton_budget_gate');break
      accepted=False;trials=[];basej=current['native_J']
      for bt in range(28):
       if native_calls+2>native_cap or time.monotonic()>=deadline-reserve_final:break
       alpha=.5**bt;trial_y=y+alpha*step;trial_c=lift(trial_y);rhs=basej+1e-4*alpha*slope;value=native_value(trial_c);row={'alpha':alpha,'native_J':value,'control_sha256':tsha(trial_c),'control_norm':float(torch.linalg.vector_norm(trial_c)),'armijo_rhs':rhs,'armijo_passed':value<=rhs};trials.append(row);state['iterations'].append({'iteration':iteration,'trial':row});save('original_native_J_armijo_trial')
       if row['armijo_passed']:
        branch,bj=branch_audit(trial_c)
        tol=RM*torch.finfo(trial_c.dtype).eps*max(1.,abs(value))
        if abs(value-bj)>tol:state['status']='native_J_branch_audit_mismatch';save('branch_value_mismatch');break
        y=trial_y;current=differential(y,branch,value);state['native_J']=current['native_J'];state['accepted_steps'].append({'iteration':iteration,'alpha':alpha,'native_J':value,'control_sha256':current['control_sha256'],'step_norm':alpha*float(torch.linalg.vector_norm(step))});state['current_point']=plain_diag(current);state['iterations'].append({'iteration':iteration,'accepted_point':plain_diag(current),'trials':trials});save('accepted_native_J_newton_step');accepted=True;break
      if state['status']=='native_J_branch_audit_mismatch':break
      # Update current gradient exactly; retain Hcur only as a fixed metric.
      if accepted:
       step=-torch.cholesky_solve(current['tangent_gradient_plus'][:,None],chol).squeeze(1);slope=float(torch.dot(current['tangent_gradient_plus'],step));state['iterations'][-1]['fixed_metric_after_step']={'g_dot_step':slope,'direction_norm':float(torch.linalg.vector_norm(step))};save('fresh_gradient_fixed_metric_step')
      else:state['status']='original_native_J_armijo_grid_exhausted';save('armijo_grid_exhausted');break
      if not current['singleface_supported'] or not current['strict_outward'] or not current['theta_strict']:state['status']='post_step_face_gate_failed';save('post_step_face_gate');break
      if current['first_order_pass']:state['status']='stationary_face_root';break
     if state['status']=='in_progress':
      if current['first_order_pass']:state['status']='stationary_face_root';save('chord_cycle_stationary_root')
      else:state['status']='retained_graph_chord_cycle_limit';save('retained_graph_cycle_limit')
     if state['status']=='stationary_face_root' and current['first_order_pass']:
      # Final exact positive and negative reduced Hessians at the same point.
      matrices={}
      for side in (+1,-1):
       zz,ph,gg=build_graph(y,side);rows=[]
       if abs(float(ph)-current['native_J'])>tolJ:state['status']='final_profile_value_mismatch';save('final_profile_value_gate',side=side);break
       target_grad=current['tangent_gradient_plus'] if side==1 else current['tangent_gradient_minus'];grad_disagreement=float((gg-target_grad).abs().max());grad_scale=max(float(gg.norm()),float(target_grad.norm()),torch.finfo(y.dtype).tiny);grad_budget=max(q.blocks.SYMMETRY_TOL*grad_scale,RM*torch.finfo(y.dtype).eps);state.setdefault('final_profile_gradient_checks',{})[str(side)]={'max_abs':grad_disagreement,'budget':grad_budget,'passed':grad_disagreement<=grad_budget};save('final_profile_gradient_check',side=side)
       if grad_disagreement>grad_budget:state['status']='final_profile_gradient_mismatch';save('final_profile_gradient_refusal',side=side);break
       for i in range(NT):
        if time.monotonic()>=deadline or rss_bytes()>RSS_CAP or row_started>=row_cap:state['status']='final_H_rows_budget_or_RSS_exhausted';save('final_H_row_partial',side=side,row=i);break
        row_started+=1;save('final_H_row_started',side=side,row=i);r=torch.autograd.grad(gg[i],zz,retain_graph=i<NT-1)[0];rows.append(r);row_returned+=1;state.setdefault('final_H_rows_by_side',{}).setdefault(str(side),[]).append(r.tolist());save('final_H_row_returned',side=side,row=i)
       if len(rows)!=NT:break
       HH=torch.stack(rows,dim=0);SS=(HH+HH.T)/2;ee=torch.linalg.eigvalsh(SS);nh=float(torch.linalg.matrix_norm(SS));se=float(torch.linalg.matrix_norm(HH-HH.T)/max(nh,torch.finfo(HH.dtype).tiny));fl=max(q.blocks.CURVATURE_REL_TOL*float(ee.abs().max()),RM*torch.finfo(HH.dtype).eps*max(nh,torch.finfo(HH.dtype).tiny));matrices[str(side)]={'hessian_sha256':tsha(HH),'hessian':HH.tolist(),'eigenvalues':ee.tolist(),'lambda_min':float(ee[0]),'symmetry_relative':se,'positive_floor':fl,'positive_definite':se<=q.blocks.SYMMETRY_TOL and float(ee[0])>fl};save('final_samepoint_side_H_complete',side=side,P2_by_side=matrices)
      if state['status']=='stationary_face_root' and len(matrices)==2:
       replay_checks=[];chart_jac=torch.func.jacrev(lift)(y);grad_orig=torch.func.grad(problem.objective,argnums=0)
       for side in (+1,-1):
        if hvp_calls>=int(plan['budget']['additional_replay_on_HVP_checks']) or time.monotonic()>=deadline:state['status']='final_replay_HVP_check_budget_exhausted';save('final_replay_HVP_partial',side=side);break
        direction=eye[0];direction_control=torch.func.jvp(lift,(y,),(direction,))[1];hvp_calls+=1;save('final_replay_HVP_started',side=side)
        with transport.selected_face_extension('y',4,3,side):action=torch.func.jvp(lambda c:grad_orig(c,p),(lift(y),),(direction_control,))[1]
        chart_action=chart_jac.T@action;profile_action=torch.tensor(matrices[str(side)]['hessian'][0],dtype=y.dtype);err=float((chart_action-profile_action).norm());scale=max(float(chart_action.norm()),float(profile_action.norm()),torch.finfo(y.dtype).tiny);rel=err/scale;allowed=plan['budget']['replay_Hessian_relative_tolerance'];replay_checks.append({'side':side,'direction_chart_sha256':tsha(direction),'relative_error':rel,'allowed_relative_error':allowed,'passed':rel<=allowed,'replay_on_action_sha256':tsha(chart_action),'retained_row_sha256':tsha(profile_action)});save('final_replay_HVP_returned',side=side,check=replay_checks[-1])
       state['final_replay_Hessian_checks']=replay_checks
       if state['status']=='stationary_face_root' and len(replay_checks)==2 and all(x['passed'] for x in replay_checks):
        state['P2_by_side']=matrices;state['minimum_claim']=all(v['positive_definite'] for v in matrices.values());state['native_J_recheck']=current['native_J'];state['replay_profile_final_checks']=replay_checks;state['status']='qualified_original_native_J_singleface_minimum' if state['minimum_claim'] else 'stationary_face_P2_failed';save('final_two_sided_P2')
       elif state['status']=='stationary_face_root':state['status']='final_replay_profile_Hessian_equivalence_failed';save('final_replay_profile_refusal')
 _,source_after,_=q._execution_plan(continuation_plan_path=CONT_PLAN,expected_continuation_sha256=plan['continuation_plan']['sha256']);identity_after=q.cycle.shared._input_identity(problem,original,lift(y),p,truth)
 state['source_after']=source_after;state['source_archive_closure']=source_before==source_after;state['input_after']=identity_after;state['original_input_identity_unchanged']=identity_after==base['input_after'];state['script_sha256_after']=sha(SELF.read_bytes());state['script_unchanged']=state['script_sha256_after']==plan['source_files'][SELF.relative_to(ROOT).as_posix()];state['qualification_plan_sha256_after']=sha(QUAL_PLAN.read_bytes());state['source_unchanged']=state['script_unchanged'] and state['qualification_plan_sha256_after']==plan_sha;state['elapsed_seconds']=time.monotonic()-started;state['ledger'].update(run_elapsed_seconds=state['elapsed_seconds'],charged_at_end_seconds=plan['budget']['charged_before_phase_seconds']+state['elapsed_seconds'],remaining_after_phase_seconds=3600.-plan['budget']['charged_before_phase_seconds']-state['elapsed_seconds']);write(OUTPUT,state);return state
if __name__=='__main__':
 result=run();print(json.dumps({k:result.get(k) for k in ('status','control_sha256','native_J','native_J_value_calls','gradient_calls','profile_scalar_calls','retained_Hessian_rows_started','retained_Hessian_rows_returned','accepted_steps','minimum_claim','elapsed_seconds','peak_rss_bytes','ledger')},indent=2))
