"""Exact current-point chart Newton and common-face local-minimum qualification.

The chosen face chart sets the signed qy[4,3] volume coordinate to zero
identically in real arithmetic. ``selected_face_extension`` changes only the
allocation of that face flux between positive and negative donor terms; when
its signed flux is zero, both terms vanish. Therefore the two restricted
native-J scalar functions on the regular chart are identical, so their chart
gradients and Hessians are mathematically equal. The final record checks three
deterministic opposite-side Hessian actions against the full same-point
positive-side Hessian, while separately retaining the two strict outward
normal slopes. A mismatch refuses the minimum claim.
"""
from __future__ import annotations
import hashlib,importlib.util,json,math,sys,time
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=HERE/'continuation3/attempt1'
INPUT=OUT/'fixed_metric_face_qualification.json';BASE=OUT/'modified_newton_face_qualification_attempt2.json'
QUAL_PLAN=HERE/'QUALIFICATION_CONTINUATION_PLAN_v7.json';CONT_PLAN=HERE/'continuation3/CONTINUATION_PLAN.json';OUTPUT=OUT/'exact_newton_face_qualification.json';SELF=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location('native_qualifier',HERE/'qualify_candidate.py')
if spec is None or spec.loader is None:raise RuntimeError('native qualifier unavailable')
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
from advar import transport
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart,face_weights_from_basis
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
NT=25;TOL=1e-10;RM=128.;FACE=('y',4,3)
def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def tsha(x:torch.Tensor)->str:return sha(x.detach().contiguous().cpu().numpy().tobytes())
def write(path:Path,value:dict[str,Any])->None:
 def default(v:Any)->Any:
  if isinstance(v,torch.Tensor):return v.detach().contiguous().cpu().tolist()
  raise TypeError(f'cannot serialize {type(v).__name__}')
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=default)+'\n');tmp.replace(path)
def run()->dict[str,Any]:
 if OUTPUT.exists():raise ValueError('exact-Newton output must be fresh')
 plan=json.loads(QUAL_PLAN.read_text());plan_sha=sha(QUAL_PLAN.read_bytes());inp=json.loads(INPUT.read_text());inp_sha=sha(INPUT.read_bytes());base=json.loads(BASE.read_text());base_sha=sha(BASE.read_bytes())
 if plan.get('status')!='frozen' or plan.get('experiment_kind')!='native_J_same_face_exact_Newton':raise ValueError('same-face exact-Newton plan is not frozen')
 for name,digest in plan['source_files'].items():
  if sha((ROOT/name).read_bytes())!=digest:raise ValueError(f'qualification source pin changed: {name}')
 if plan['inputs'].get(INPUT.relative_to(ROOT).as_posix())!=inp_sha or plan['inputs'].get(BASE.relative_to(ROOT).as_posix())!=base_sha:raise ValueError('current point receipts differ from plan')
 if inp.get('phase')!='final_record' or inp.get('status')!='native_J_value_cap_exhausted' or inp.get('new_HVP_started')!=0:raise ValueError('input must be the closed nonstationary v6 candidate')
 if sha(CONT_PLAN.read_bytes())!=plan['continuation_plan']['sha256']:raise ValueError('original continuation source plan changed')
 started=time.monotonic();deadline=started+float(plan['budget']['phase_wall_cap_seconds']);native_cap=int(plan['budget']['cumulative_native_J_value_cap']);hvp_cap=int(plan['budget']['additional_HVP_cap']);native_calls=int(inp['native_J_value_calls']);gradient_calls=int(inp['gradient_calls']);hvp_started=hvp_returned=0
 if native_calls!=plan['budget']['native_J_value_calls_already_used'] or native_calls>=native_cap:raise ValueError('cumulative native-J budget differs from frozen plan')
 control=torch.tensor(inp['control'],dtype=torch.float64)
 if tsha(control)!=inp['control_sha256'] or tsha(torch.tensor(base['control'],dtype=torch.float64))!=tsha(control):raise ValueError('current c changed from its pinned fixed-metric endpoint')
 case=q.cycle.load_fixed_case();problem,original,truth,p=case['problem'],case['original'],case['truth'],case['parameters'];fs=problem.frozen.fv_transport
 _,source_before,transition=q._execution_plan(continuation_plan_path=CONT_PLAN,expected_continuation_sha256=plan['continuation_plan']['sha256'])
 if transition!=inp['source_transition'] or tsha(p)!=inp['input_parameters_sha256'] or not inp.get('source_archive_closure') or not inp.get('source_unchanged'):raise ValueError('current point source/input closure is not qualified')
 if q.cycle.shared._input_identity(problem,original,control,p,truth)!=base['input_after']:raise ValueError('p/data/runtime identity differs from the previous closed point')
 fs=problem.frozen.fv_transport;weights=face_weights_from_basis(fs.psi_basis,'y',(4,3));nw=weights*fs.coefficient_limits*(1-torch.tanh(control[20:25]).square());pivot_flow=int(nw.abs().argmax())
 chart=FVFaceFluxCoordinateChart.from_basis(field_count=20,growth_count=1,coefficient_limits=fs.coefficient_limits,psi_basis=fs.psi_basis,axis='y',face=(4,3),pivot_index=pivot_flow)
 coords=chart.to_face_coordinates(control);pivot_obj=chart.layout.get('face_flux_coordinate')
 if not isinstance(pivot_obj,int) or isinstance(pivot_obj,bool):raise ValueError('face chart pivot is not a fixed integer')
 pivot=pivot_obj;coords[pivot]=0.;y=torch.cat((coords[:pivot],coords[pivot+1:]));roundtrip=chart.from_face_coordinates(torch.cat((y[:pivot],y.new_zeros(1),y[pivot:])))
 if tsha(roundtrip)!=tsha(control):raise ValueError('chart roundtrip changed the pinned current control bits')
 def lift(v):return chart.from_face_coordinates(torch.cat((v[:pivot],v.new_zeros(1),v[pivot:])))
 def qvalue(c):
  psi=torch.einsum('k,kij->ij',fs.coefficient_limits*torch.tanh(c[20:25]),fs.psi_basis);return transport.face_volume_fluxes(psi)[1][4,3]
 def chart_obj(v,side=1):
  with transport.selected_face_extension('y',4,3,side):return problem.objective(lift(v),p)
 def native_value(fn):
  nonlocal native_calls
  if native_calls>=native_cap:raise RuntimeError('cumulative native-J value cap exhausted before objective callback')
  native_calls+=1;val=fn()
  if not isinstance(val,torch.Tensor) or val.ndim!=0 or not bool(torch.isfinite(val)):raise ValueError('native J callback is not a finite scalar')
  return float(val)
 def audit_branch(c):
  nonlocal native_calls
  if native_calls>=native_cap:raise RuntimeError('cumulative native-J cap exhausted before actual branch callback')
  branch=q._analysis_branch(problem,c,p,deadline=deadline)
  if branch.get('objective_finite') is not True or branch.get('native_J_on_audited_path') is None:raise ValueError('actual native-J branch audit is nonfinite')
  native_calls+=1
  return branch,float(branch['native_J_on_audited_path'])
 def differential(v,branch,native_j):
  nonlocal gradient_calls
  c=lift(v);full={};tg={}
  for side in (-1,1):
   gradient_calls+=1
   with transport.selected_face_extension('y',4,3,side):full[side]=torch.func.grad(problem.objective,argnums=0)(c,p)
   gradient_calls+=1;tg[side]=torch.func.grad(lambda z,pp,s=side:chart_obj(z,s),argnums=0)(v,p)
  ng=torch.func.grad(qvalue)(c);nn=torch.linalg.vector_norm(ng);slopes={s:float(torch.dot(full[s],ng)/nn) for s in (-1,1)};budgets={s:RM*torch.finfo(c.dtype).eps*float(full[s].norm()) for s in (-1,1)}
  theta=tangent.minimum_mixture_weight(full[-1],full[1]);mixed=(1-theta)*full[-1]+theta*full[1];agreement=float((tg[-1]-tg[1]).abs().max());ginf=float(tg[1].abs().max());minf=float(mixed.abs().max());agree_budget=RM*torch.finfo(c.dtype).eps*max(float(tg[-1].norm()),float(tg[1].norm()),torch.finfo(c.dtype).tiny)
  one=(branch.get('classification')=='singleface' and branch.get('unique_nearzero_face_entities')==['y:4:3'] and not branch.get('unique_near_tie_slope_entities'));out=slopes[1]>budgets[1] and slopes[-1]<-budgets[-1]
  return {'control':c,'control_sha256':tsha(c),'native_J':float(native_j),'branch':branch,'singleface_supported':one,'face_q':float(qvalue(c)),'normal_l2':float(nn),'theta':float(theta),'theta_strict':1e-8<theta<1-1e-8,'gradient_inf':ginf,'mixed_gradient_inf':minf,'tangent_gradient_inf_by_side':{str(s):float(tg[s].abs().max()) for s in (-1,1)},'tangent_gradient_agreement_inf':agreement,'tangent_agreement_budget':agree_budget,'normal_slopes':{str(s):slopes[s] for s in (-1,1)},'normal_budgets':{str(s):budgets[s] for s in (-1,1)},'strict_outward':out,'first_order_pass':ginf<TOL and minf<TOL and agreement<=agree_budget,'chart_gradient':tg[1]}
 def json_diag(diag):return {k:v for k,v in diag.items() if k not in ('control','chart_gradient')}
 cached=inp['iterations'][-1]['point'];branch0=cached['branch'];
 if not (cached['singleface_supported'] and cached['strict_outward'] and cached['strict_theta']):raise ValueError('pinned current face/outward/mixture gate failed')
 # Reuse the closed same-control selector audit, then freshly calculate both
 # full gradients and both chart gradients at that exact point.
 current=differential(y,branch0,float(inp['current_native_J']));state={'status':'in_progress','phase':'current_point_differentials','control':lift(y).tolist(),'control_sha256':tsha(lift(y)),'native_J':current['native_J'],'current_point':json_diag(current),'native_J_value_calls_before_phase':native_calls,'native_J_value_call_cap':native_cap,'gradient_calls_before_phase':gradient_calls,'additional_HVP_started':0,'additional_HVP_returned':0,'HVP_cap':hvp_cap,'previous_phase_control_sha256':inp['control_sha256'],'previous_phase_native_J_calls':inp['native_J_value_calls'],'previous_phase_HVP_started':inp['new_HVP_started'],'previous_phase_HVP_returned':inp['new_HVP_returned'],'accepted_steps':[],'iterations':[],'ledger':plan['budget'],'continuation_plan_sha256':plan['continuation_plan']['sha256'],'qualification_plan_sha256':plan_sha,'source_before':source_before,'source_transition':transition,'input_parameters_sha256':tsha(p),'input_receipt_sha256':sha(INPUT.read_bytes()),'prior_attempt2_receipt_sha256':base_sha,'common_chart_identity':{'face':['y',4,3],'chart':chart.identity,'phi_plus_equals_phi_minus_on_chart':'Q∘Γ=0; selected_face_extension only repartitions the signed zero face volume into plus/minus donor pieces; both primal face contributions vanish identically in exact arithmetic'},'P2_action_check_vectors':[]}
 def save(phase,**extra):
  state.update(phase=phase,control=lift(y).tolist(),control_sha256=tsha(lift(y)),native_J=current['native_J'],native_J_value_calls=native_calls,gradient_calls=gradient_calls,additional_HVP_started=hvp_started,additional_HVP_returned=hvp_returned,elapsed_seconds=time.monotonic()-started);state.update(extra);write(OUTPUT,state)
 save('current_point_differentials')
 if not current['singleface_supported'] or not current['strict_outward'] or not current['theta_strict']:
  state['status']='current_face_or_normal_gate_failed';save('current_face_gate');return state
 if current['first_order_pass']:
  state['status']='stationary_face_root';save('stationary_at_input')
 # Exact current-point H+ columns; it may be used as a fixed approximate
 # metric after a step, never as curvature evidence at the new point.
 Hcur_cols=[];eye=torch.eye(NT,dtype=y.dtype);gf=lambda z:torch.func.grad(lambda v,pp:chart_obj(v,1),argnums=0)(z,p)
 for j in range(NT):
  estimated_finish=(NT-j+28)*8.8+plan['budget']['reserved_forecast_and_verification_seconds']+20.
  if time.monotonic()>=deadline or deadline-time.monotonic()<estimated_finish or hvp_started>=hvp_cap:state['status']='current_H_budget_exhausted_preserving_final_P2';save('current_H_partial',column=j,estimated_finish_seconds=estimated_finish);return state
  hvp_started+=1;save('current_H_column_started',column=j)
  col=torch.func.jvp(gf,(y,),(eye[j],))[1];Hcur_cols.append(col);hvp_returned+=1;save('current_H_column_returned',column=j)
 Hcur=torch.stack(Hcur_cols,dim=1);Hs=(Hcur+Hcur.T)/2;ev=torch.linalg.eigvalsh(Hs);hn=float(torch.linalg.matrix_norm(Hs));sym=float(torch.linalg.matrix_norm(Hcur-Hcur.T)/max(hn,torch.finfo(Hcur.dtype).tiny));floor=max(q.blocks.CURVATURE_REL_TOL*float(ev.abs().max()),RM*torch.finfo(Hcur.dtype).eps*max(hn,torch.finfo(Hcur.dtype).tiny));pd=sym<=q.blocks.SYMMETRY_TOL and float(ev[0])>floor
 state['current_H']={'hessian_sha256':tsha(Hcur),'columns':Hcur.tolist(),'eigenvalues':ev.tolist(),'lambda_min':float(ev[0]),'symmetry_relative':sym,'positive_floor':floor,'positive_definite':pd,'control_sha256':current['control_sha256']};save('current_H_complete')
 if not current['first_order_pass']:
  target=max(floor,1e-3*hn);mu=0. if pd else max(0.,target-float(ev[0]))+RM*torch.finfo(Hcur.dtype).eps*hn;Hmod=Hs+mu*torch.eye(NT,dtype=y.dtype);metric_chol=torch.linalg.cholesky(Hmod);g=current['chart_gradient'];step=-torch.cholesky_solve(g[:,None],metric_chol).squeeze(1);slope=float(torch.dot(g,step));state['current_H'].update(spectral_target=target,shift_mu=mu,direction_norm=float(torch.linalg.vector_norm(step)),g_dot_step=slope,direction_kind='exact Newton' if mu==0 else 'modified Newton');save('exact_current_newton_direction')
  if not math.isfinite(slope) or slope>=0:state['status']='current_H_not_descent';save('current_H_descent_gate');
  else:
   for iteration in range(12):
    if current['first_order_pass']:state['status']='stationary_face_root';break
    # Preserve sufficient wall/HVP reserve for final H+ (25) and minus-side
    # identity actions (3), plus the external verification reserve.
    final_reserve=28*8.8+float(plan['budget']['reserved_forecast_and_verification_seconds'])
    if time.monotonic()>=deadline-final_reserve or hvp_started+28>hvp_cap:
     state['status']='wall_or_HVP_budget_reserved_for_final_P2';break
    if native_calls+2>native_cap:state['status']='native_J_value_cap_reserved_for_branch_audit';break
    g=current['chart_gradient'];slope=float(torch.dot(g,step));base_j=current['native_J'];accepted=False;trial_rows=[]
    for bt in range(28):
     if time.monotonic()>=deadline-final_reserve or native_calls+2>native_cap:break
     alpha=.5**bt;trial_y=y+alpha*step;trial_c=lift(trial_y);rhs=base_j+1e-4*alpha*slope
     lower=.5*float(torch.dot(trial_c,trial_c));row={'alpha':alpha,'control_sha256':tsha(trial_c),'control_norm':float(torch.linalg.vector_norm(trial_c)),'identity_prior_lower_bound':lower,'armijo_rhs':rhs}
     if lower>rhs:row.update(native_J=None,status='identity_prior_lower_bound_reject')
     else:
      val=native_value(lambda:problem.objective(trial_c,p));row['native_J']=val;row['armijo_passed']=val<=rhs;row['status']='accepted' if row['armijo_passed'] else 'native_J_Armijo_reject'
     trial_rows.append(row);state['iterations'].append({'iteration':iteration,'trial':row});save('native_J_armijo_trial')
     if row['status']=='accepted':
      branch,branch_j=audit_branch(trial_c);budget=128*torch.finfo(trial_c.dtype).eps*max(1.,abs(row['native_J']))
      if abs(branch_j-row['native_J'])>budget:state['status']='native_J_branch_audit_mismatch';save('accepted_trial_branch_mismatch');break
      y=trial_y;current=differential(y,branch,row['native_J']);accepted=True;state['accepted_steps'].append({'iteration':iteration,'alpha':alpha,'native_J':row['native_J'],'control_sha256':current['control_sha256'],'step_norm':alpha*float(torch.linalg.vector_norm(step))});state['iterations'].append({'iteration':iteration,'accepted_point':json_diag(current),'line_search':trial_rows});save('accepted_exact_newton_or_chord_step');break
    if state['status']=='native_J_branch_audit_mismatch':break
    if not accepted:state['status']='native_J_armijo_grid_exhausted';save('line_search_exhausted');break
    if not current['singleface_supported'] or not current['strict_outward'] or not current['theta_strict']:
     state['status']='post_step_face_gate_failed';save('post_step_face_gate');break
    if current['first_order_pass']:state['status']='stationary_face_root';save('stationary_face_root');break
    # Keep the exact current-point modified H as an explicitly approximate
    # descent metric after this accepted control change; rebuild g each time.
    gradient_calls+=1;g=current['chart_gradient'];step=-torch.cholesky_solve(g[:,None],metric_chol).squeeze(1);state['iterations'][-1]['fixed_metric_gradient_inf']=current['gradient_inf'];state['iterations'][-1]['fixed_metric_g_dot_step']=float(torch.dot(g,step));save('fresh_gradient_fixed_metric_direction')
 # Same-point final P2: exact plus-side chart H and 3 deterministic minus-side
 # Hessian actions. Require both mathematical identity certificate and tests.
 if state.get('status')=='stationary_face_root' and current['first_order_pass'] and current['singleface_supported'] and current['strict_outward'] and current['theta_strict']:
  Hplus_cols=[]
  for j in range(NT):
   if time.monotonic()>=deadline or hvp_started>=hvp_cap:state['status']='final_Hplus_budget_exhausted';save('final_Hplus_partial',column=j);break
   hvp_started+=1;save('final_Hplus_column_started',column=j);col=torch.func.jvp(gf,(y,),(eye[j],))[1];Hplus_cols.append(col);hvp_returned+=1;state.setdefault('final_Hplus_columns',[]).append(col.tolist());save('final_Hplus_column_returned',column=j)
  if len(Hplus_cols)==NT:
   Hplus=torch.stack(Hplus_cols,dim=1);Hsym=(Hplus+Hplus.T)/2;heig=torch.linalg.eigvalsh(Hsym);hnorm=float(torch.linalg.matrix_norm(Hsym));hsym=float(torch.linalg.matrix_norm(Hplus-Hplus.T)/max(hnorm,torch.finfo(Hplus.dtype).tiny));hfloor=max(q.blocks.CURVATURE_REL_TOL*float(heig.abs().max()),RM*torch.finfo(Hplus.dtype).eps*max(hnorm,torch.finfo(Hplus.dtype).tiny));pdplus=hsym<=q.blocks.SYMMETRY_TOL and float(heig[0])>hfloor;checks=[];vectors=[eye[0],eye[12],eye[24]]
   v=torch.arange(1,NT+1,dtype=y.dtype);vectors.append(torch.sin(v*math.sqrt(2.))+torch.cos(v*math.sqrt(3.)));vectors=vectors[:3]
   gfminus=lambda z:torch.func.grad(lambda vv,pp:chart_obj(vv,-1),argnums=0)(z,p)
   for k,direction in enumerate(vectors):
    if time.monotonic()>=deadline or hvp_started>=hvp_cap:state['status']='final_minus_action_budget_exhausted';save('final_minus_action_partial',action=k);break
    direction=direction/torch.linalg.vector_norm(direction);hvp_started+=1;save('final_minus_action_started',action=k);minus_action=torch.func.jvp(gfminus,(y,),(direction,))[1];hvp_returned+=1;plus_action=Hplus@direction;err=float(torch.linalg.vector_norm(minus_action-plus_action));scale=max(float(torch.linalg.vector_norm(plus_action)),float(torch.linalg.vector_norm(minus_action)),torch.finfo(y.dtype).tiny);rel=err/scale;budget=q.blocks.SYMMETRY_TOL+RM*torch.finfo(y.dtype).eps*max(1.,hnorm);checks.append({'action':k,'direction_sha256':tsha(direction),'relative_disagreement':rel,'allowed_relative_budget':budget,'passed':rel<=budget,'positive_action_sha256':tsha(plus_action),'negative_action_sha256':tsha(minus_action)});state['common_face_hessian_action_checks']=checks;save('final_minus_action_returned',action=k)
   if len(checks)==3 and all(c['passed'] for c in checks):
    state['common_face_identity_certificate']={'statement':'On Γ(u), qy[4,3] is identically zero in real arithmetic. The selected-face extension changes only the plus/minus split of this zero signed volume; both donor contributions vanish, so the restricted native-J scalars and all chart derivatives are identical.','source':'advar.transport._volume_parts / _euler_donorcell; examples.weather_scenarios.fv_face_flux_coordinates.FVFaceFluxCoordinateChart.from_face_coordinates; examples.weather_scenarios.fv_point_3h_netflux_current_cycle._analysis_branch','chart_face':['y',4,3],'same_point_control_sha256':tsha(lift(y)),'action_checks':checks}
    state['P2_by_common_chart']={'hessian_sha256':tsha(Hplus),'hessian':Hplus.tolist(),'eigenvalues':heig.tolist(),'lambda_min':float(heig[0]),'symmetry_relative':hsym,'positive_floor':hfloor,'positive_definite':pdplus,'opposite_side_action_checks':checks};state['minimum_claim']=bool(pdplus);state['native_J_recheck']=current['native_J'];state['P2_scope']='conditional real-arithmetic common-face local minimum; opposite-side FP64 curvature consistency audited by three deterministic H-action checks, not a separately materialized negative-side Hessian';state['status']='conditional_common_face_local_minimum_qualified' if pdplus else 'stationary_common_face_curvature_failed';save('final_common_face_P2')
   elif state['status']=='stationary_face_root':state['status']='common_face_Hessian_identity_check_failed';save('common_face_identity_refusal',action_checks=checks,positive_definite=pdplus)
 _,source_after,_=q._execution_plan(continuation_plan_path=CONT_PLAN,expected_continuation_sha256=plan['continuation_plan']['sha256']);identity_after=q.cycle.shared._input_identity(problem,original,lift(y),p,truth)
 state['source_after']=source_after;state['source_archive_closure']=source_before==source_after;state['input_after']=identity_after;state['script_sha256_after']=sha(SELF.read_bytes());state['script_unchanged']=state['script_sha256_after']==plan['source_files'][SELF.relative_to(ROOT).as_posix()];state['qualification_plan_sha256_after']=sha(QUAL_PLAN.read_bytes());state['source_unchanged']=state['script_unchanged'] and state['qualification_plan_sha256_after']==plan_sha;state['elapsed_seconds']=time.monotonic()-started;state['ledger'].update(run_elapsed_seconds=state['elapsed_seconds'],charged_at_end_seconds=plan['budget']['charged_before_phase_seconds']+state['elapsed_seconds'],remaining_after_phase_seconds=3600.-plan['budget']['charged_before_phase_seconds']-state['elapsed_seconds']);write(OUTPUT,state);return state
if __name__=='__main__':
 result=run();print(json.dumps({k:result.get(k) for k in ('status','control_sha256','native_J','native_J_value_calls','gradient_calls','additional_HVP_started','additional_HVP_returned','accepted_steps','minimum_claim','elapsed_seconds','ledger')},indent=2))
