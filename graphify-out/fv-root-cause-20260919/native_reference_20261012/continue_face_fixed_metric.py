"""Bounded native-J refinement using a previously verified SPD metric.

Every current-point gradient, active-face audit, mixture, outward-slope check,
and native-J Armijo value is fresh. The stored SPD Hessian is used only as an
approximate descent metric after its point changes; no exact-curvature claim is
made there. If the first-order gates pass, the final two-sided chart Hessians
are recomputed at that same point before any minimum claim.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, sys, time
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUTDIR=HERE/'continuation3/attempt1'
INPUT=OUTDIR/'modified_newton_face_qualification_attempt2.json';QUAL_PLAN=HERE/'QUALIFICATION_CONTINUATION_PLAN_v6.json'
OUTPUT=OUTDIR/'fixed_metric_face_qualification.json';CONT_PLAN=HERE/'continuation3/CONTINUATION_PLAN.json'
SELF=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location('native_qualifier',HERE/'qualify_candidate.py')
if spec is None or spec.loader is None:raise RuntimeError('native qualifier source is unavailable')
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
from advar import transport
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart,face_weights_from_basis
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
FACE=('y',4,3);NT=25;TOL=1e-10;RM=128.

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def tsha(x:torch.Tensor)->str:return sha(x.detach().contiguous().cpu().numpy().tobytes())
def write(p:Path,d:dict[str,Any])->None:
 def default(v:Any)->Any:
  if isinstance(v,torch.Tensor):return v.detach().contiguous().cpu().tolist()
  raise TypeError(f'cannot serialize {type(v).__name__}')
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False,default=default)+'\n');tmp.replace(p)

def run()->dict[str,Any]:
 plan=json.loads(QUAL_PLAN.read_text());plan_sha=sha(QUAL_PLAN.read_bytes());prior=json.loads(INPUT.read_text());input_sha=sha(INPUT.read_bytes())
 if plan.get('experiment_kind')!='bounded_native_J_face_qualification' or plan.get('status')!='frozen':raise ValueError('qualification continuation plan is not frozen')
 for name,digest in plan['source_files'].items():
  if sha((ROOT/name).read_bytes())!=digest:raise ValueError(f'qualification source pin changed: {name}')
 if plan['inputs'].get(INPUT.relative_to(ROOT).as_posix())!=input_sha or plan['input_control_sha256']!=prior.get('control_sha256'):raise ValueError('qualification input checkpoint differs from plan')
 resumed=OUTPUT.exists();partial=None;partial_sha=None
 if resumed:
  partial=json.loads(OUTPUT.read_text());partial_sha=sha(OUTPUT.read_bytes())
  if (plan['inputs'].get(OUTPUT.relative_to(ROOT).as_posix())!=partial_sha
          or partial.get('phase')!='terminal_preflight_error'
          or partial.get('status')!='resumable_after_preflight_error'
          or partial.get('accepted_steps')
          or partial.get('new_HVP_started')!=0
          or partial.get('new_HVP_returned')!=0
          or partial.get('control_sha256')!=plan.get('input_control_sha256')):
   raise ValueError('existing output is not the exact resumable initial audit checkpoint')
 else:
  if OUTPUT.exists():raise ValueError('qualification output exists and is not resumable')
 if plan['budget']['original_cap_seconds']!=3600.0 or plan['budget']['phase_wall_cap_seconds']<=0:raise ValueError('qualification cumulative budget malformed')
 started=time.monotonic();deadline=started+float(plan['budget']['phase_wall_cap_seconds']);native_cap=int(plan['budget']['native_J_value_calls']);hvp_cap=int(plan['budget']['new_HVP_calls']);native_calls=int(partial['native_J_value_calls']) if partial else 0;gradient_calls=int(partial['gradient_calls']) if partial else 0;hvp_started=int(partial['new_HVP_started']) if partial else 0;hvp_returned=int(partial['new_HVP_returned']) if partial else 0
 def native_value(fn)->float:
  nonlocal native_calls
  if native_calls>=native_cap:raise RuntimeError('qualification native-J value-call cap exhausted before callback')
  native_calls+=1;v=fn()
  if not isinstance(v,torch.Tensor) or v.ndim!=0 or not bool(torch.isfinite(v)):raise ValueError('native-J callback returned a nonfinite or malformed value')
  return float(v)
 control=torch.tensor(prior['control'],dtype=torch.float64)
 if tsha(control)!=prior['control_sha256'] or prior['status']!='two_corrections_exhausted_without_stationarity':raise ValueError('saved qualification point does not match frozen plan')
 case=q.cycle.load_fixed_case();problem,original,truth,p=case['problem'],case['original'],case['truth'],case['parameters'];fs=problem.frozen.fv_transport
 _,source_before,transition=q._execution_plan(continuation_plan_path=CONT_PLAN,expected_continuation_sha256=plan['continuation_plan']['sha256'])
 if transition!=prior['source_transition'] or tsha(p)!=prior['input_parameters_sha256']:raise ValueError('original fixed input/source transition changed')
 if resumed and (partial.get('source_before')!=source_before or partial.get('source_transition')!=transition or partial.get('preflight_error',{}).get('exception_type')!='KeyError'):raise ValueError('saved branch audit source closure/error receipt differs from resumed source pins')
 fixed_identity=q.cycle.shared._input_identity(problem,original,control,p,truth)
 if prior.get('input_after')!=fixed_identity:raise ValueError('fixed data/parameter/runtime identity differs from prior exact qualification point')
 if sha((ROOT/'src/advar/transport.py').read_bytes())!=transition['after']:raise ValueError('approved transport support change pin changed')
 weights=face_weights_from_basis(fs.psi_basis,'y',(4,3));nw=weights*fs.coefficient_limits*(1-torch.tanh(control[20:25]).square());pivot_i=int(nw.abs().argmax());chart=FVFaceFluxCoordinateChart.from_basis(field_count=20,growth_count=1,coefficient_limits=fs.coefficient_limits,psi_basis=fs.psi_basis,axis='y',face=(4,3),pivot_index=pivot_i)
 coords=chart.to_face_coordinates(control);pivot_value=chart.layout.get('face_flux_coordinate')
 if not isinstance(pivot_value,int) or isinstance(pivot_value,bool):raise ValueError('face chart pivot index is not a fixed integer')
 pivot=pivot_value;coords[pivot]=0.;y=torch.cat((coords[:pivot],coords[pivot+1:]));control=chart.from_face_coordinates(coords)
 def lift(z):return chart.from_face_coordinates(torch.cat((z[:pivot],z.new_zeros(1),z[pivot:])))
 def qv(c):
  psi=torch.einsum('k,kij->ij',fs.coefficient_limits*torch.tanh(c[20:25]),fs.psi_basis);return transport.face_volume_fluxes(psi)[1][4,3]
 def objective(z,side=1):
  with transport.selected_face_extension('y',4,3,side):return problem.objective(lift(z),p)
 def diagnose(z):
  nonlocal gradient_calls,native_calls
  c=lift(z)
  if native_calls>=native_cap:
   state['status']='native_J_value_cap_exhausted';save('branch_audit_native_value_cap_gate')
   raise RuntimeError('native-J value cap exhausted before branch-audit objective callback')
  branch=q._analysis_branch(problem,c,p,deadline=deadline)
  if branch.get('objective_finite') is True and branch.get('native_J_on_audited_path') is not None:native_calls+=1
  full={};tg={}
  for side in (-1,1):
   gradient_calls+=1
   with transport.selected_face_extension('y',4,3,side):full[side]=torch.func.grad(problem.objective,argnums=0)(c,p)
   gradient_calls+=1;tg[side]=torch.func.grad(lambda v,pp,s=side:objective(v,s),argnums=0)(z,p)
  qg=torch.func.grad(qv)(c);qn=torch.linalg.vector_norm(qg);sl={s:float(torch.dot(full[s],qg)/qn) for s in (-1,1)};bud={s:RM*torch.finfo(c.dtype).eps*float(full[s].norm()) for s in (-1,1)}
  theta=tangent.minimum_mixture_weight(full[-1],full[1]);mix=(1-theta)*full[-1]+theta*full[1];agree=float((tg[-1]-tg[1]).abs().max());ginf=float(tg[1].abs().max());mixinf=float(mix.abs().max());ab=RM*torch.finfo(c.dtype).eps*max(float(tg[-1].norm()),float(tg[1].norm()),torch.finfo(c.dtype).tiny)
  one=(branch.get('classification')=='singleface' and branch.get('unique_nearzero_face_entities')==['y:4:3'] and not branch.get('unique_near_tie_slope_entities'));out=sl[1]>bud[1] and sl[-1]<-bud[-1]
  jvalue=branch.get('native_J_on_audited_path')
  if jvalue is None or not branch.get('objective_finite'):raise ValueError('branch audit did not return a finite native-J value')
  return {'control':c,'control_sha256':tsha(c),'native_J':jvalue,'branch':branch,'singleface_supported':one,'face_q':float(qv(c)),'normal_l2':float(qn),'theta':float(theta),'strict_theta':1e-8<theta<1-1e-8,'gradient_inf':ginf,'mixed_gradient_inf':mixinf,'tangent_gradient_inf_by_side':{str(s):float(tg[s].abs().max()) for s in (-1,1)},'tangent_gradient_agreement_inf':agree,'tangent_agreement_budget':ab,'normal_slopes':{str(s):sl[s] for s in (-1,1)},'normal_budgets':{str(s):bud[s] for s in (-1,1)},'strict_outward':out,'first_order_pass':ginf<TOL and mixinf<TOL and agree<=ab}
 # The exact SPD H was computed at the preceding point c1. It is only an
 # approximate metric at c2, never treated as current curvature.
 metric_cycle=prior['cycles'][1];metric_point=metric_cycle['point'];metric_record=metric_cycle.get('modified_hessian')
 if not isinstance(metric_record,dict):raise ValueError('preceding point has no resolved metric Hessian record')
 metric_floor=metric_record.get('curvature_floor',math.inf)
 metric_pd=(metric_record is not None and metric_record.get('symmetry_relative',math.inf)<=q.blocks.SYMMETRY_TOL and metric_record.get('lambda_min',0.)>metric_floor)
 if not metric_pd or len(metric_cycle.get('hessian_columns',[]))!=NT:raise ValueError('no resolved SPD metric at the preceding point')
 Hmetric=torch.tensor(metric_cycle['hessian_columns'],dtype=torch.float64).T.contiguous();Hs=(Hmetric+Hmetric.T)/2
 if tsha(Hmetric)!=metric_record['hessian_sha256']:raise ValueError('saved SPD metric columns/hash are invalid')
 metric_shift=0.;metric_used=Hs
 if float(torch.linalg.eigvalsh(metric_used)[0])<=0:raise ValueError('fixed approximate metric is not SPD')
 state=(partial if partial is not None else {'status':'in_progress','phase':'input_ready','control':lift(y).tolist(),'control_sha256':tsha(lift(y)),'previous_native_J':prior['native_J'],'current_native_J':None,'accepted_steps':[],'iterations':[],'native_J_value_calls':0,'native_J_value_cap':native_cap,'gradient_calls':0,'new_HVP_started':0,'new_HVP_returned':0,'new_HVP_cap':hvp_cap,'previous_HVP_started':prior['hvp_started_total'],'previous_HVP_returned':prior['hvp_returned_total'],'ledger':plan['budget'],'source_transition':transition,'input_parameters_sha256':tsha(p),'prior_checkpoint_sha256':input_sha})
 state.update(ledger=plan['budget'],native_J_value_cap=native_cap,new_HVP_cap=hvp_cap,continuation_plan_sha256=plan['continuation_plan']['sha256'],qualification_plan_sha256=plan_sha,source_before=source_before,source_transition=transition,input_parameters_sha256=tsha(p),prior_checkpoint_sha256=input_sha,descent_metric={'source_point_sha256':metric_point['control_sha256'],'source_hessian_sha256':metric_record['hessian_sha256'],'source_hessian_lambda_min':metric_record['lambda_min'],'source_hessian_positive_floor':metric_record['curvature_floor'],'added_shift_from_prior_modified_newton':metric_shift,'effective_metric_sha256':tsha(metric_used),'interpretation':'fixed SPD local descent metric; not exact Hessian at current point'})
 def save(phase,**extra):
  state.update(phase=phase,control=lift(y).tolist(),control_sha256=tsha(lift(y)),current_native_J=state.get('current_native_J'),native_J_value_calls=native_calls,gradient_calls=gradient_calls,new_HVP_started=hvp_started,new_HVP_returned=hvp_returned,hvp_started_total=prior['hvp_started_total']+hvp_started,hvp_returned_total=prior['hvp_returned_total']+hvp_returned,elapsed_s=time.monotonic()-started);state.update(extra);write(OUTPUT,state)
 if resumed:
  current=dict(partial['iterations'][0]['point']);current['control']=lift(y);state['resumed_from_partial_sha256']=partial_sha;state['preflight_native_J_calls_reused']=native_calls;state['preflight_gradient_calls_reused']=gradient_calls;state['preflight_hvp_calls_reused']=hvp_started
  state.update(phase='resume_from_same_point_preflight',status='in_progress',ledger=plan['budget']);save('resume_from_same_point_preflight')
 else:
  current=diagnose(y);state['current_native_J']=current['native_J'];state['iterations'].append({'point':{k:v for k,v in current.items() if k!='control'},'metric_update':'reused SPD approximate metric from preceding exact point'});save('initial_current_point_audit')
 gf=torch.linalg.cholesky(metric_used);max_steps=int(plan['budget']['max_fixed_metric_gradient_steps']);stagnant=0;previous_norm=float(current['gradient_inf'])
 for iteration in range(max_steps):
  if not current['singleface_supported'] or not current['strict_outward'] or not current['strict_theta']:state['status']='unsupported_face_or_outward_gate';save('active_face_gate');break
  if current['first_order_pass']:state['status']='stationary_face_root';save('stationary_face_root');break
  if time.monotonic()>=deadline:state['status']='cumulative_wall_budget_exhausted';save('wall_budget_gate');break
  # Exact gradient at the current point; the fixed, previously qualified SPD
  # metric guarantees a descent product gᵀd<0 even as selectors change.
  gradient_calls+=1;g=torch.func.grad(lambda z,pp:objective(z,1),argnums=0)(y,p);step=-torch.cholesky_solve(g[:,None],gf).squeeze(1);slope=float(torch.dot(g,step));row={'iteration':iteration,'point_sha256':current['control_sha256'],'native_J':current['native_J'],'gradient_inf':current['gradient_inf'],'mixed_gradient_inf':current['mixed_gradient_inf'],'g_dot_step':slope,'step_norm':float(torch.linalg.vector_norm(step)),'line_search':[]};state['iterations'][-1]['step']=row;save('fixed_metric_direction_ready',iteration=iteration)
  if not math.isfinite(slope) or slope>=0:state['status']='fixed_metric_not_descent';save('descent_gate');break
  accepted=False
  for bt in range(32):
   if time.monotonic()>=deadline:state['status']='cumulative_wall_budget_exhausted';save('line_search_wall_gate',iteration=iteration);break
   alpha=.5**bt;trial=y+alpha*step;ct=lift(trial);rhs=current['native_J']+1e-4*alpha*slope;lower=.5*float(torch.dot(ct,ct));tr={'alpha':alpha,'control_sha256':tsha(ct),'control_norm':float(torch.linalg.vector_norm(ct)),'identity_prior_lower_bound':lower,'armijo_rhs':rhs}
   if lower>rhs:tr.update(native_J=None,status='identity_prior_lower_bound_reject')
   else:
    if native_calls+2>native_cap:state['status']='native_J_value_cap_exhausted';save('native_J_value_cap_gate',iteration=iteration,trial=bt);break
    tr['native_J']=native_value(lambda:objective(trial,1));tr['armijo_passed']=tr['native_J']<=rhs;tr['status']='accepted' if tr['armijo_passed'] else 'native_J_Armijo_reject'
   row['line_search'].append(tr);save('native_J_Armijo_trial',iteration=iteration,trial=bt)
   if tr['status']=='accepted':y=trial;accepted=True;break
  if state['status'] in ('cumulative_wall_budget_exhausted','native_J_value_cap_exhausted'):break
  if not accepted:state['status']='native_J_Armijo_grid_exhausted';save('Armijo_grid_exhausted');break
  state['accepted_steps'].append({'iteration':iteration,'alpha':alpha,'native_J':tr['native_J'],'control_sha256':tsha(lift(y)),'step_norm':alpha*row['step_norm']});current=diagnose(y);state['current_native_J']=current['native_J'];state['iterations'].append({'point':{k:v for k,v in current.items() if k!='control'},'metric_update':'same fixed approximate metric; exact first-order audit refreshed'});save('accepted_step_and_fresh_gradient')
  now_norm=float(torch.linalg.vector_norm(torch.tensor([current['gradient_inf']])))
  if now_norm>=.98*previous_norm:stagnant+=1
  else:stagnant=0
  previous_norm=now_norm
  if stagnant>=8:
   estimated_refresh_and_P2_seconds=(NT+50)*11.0+plan['budget']['reserved_forecast_and_verification_seconds']
   if hvp_started+NT+50>hvp_cap or deadline-time.monotonic()<estimated_refresh_and_P2_seconds:
    state['status']='fixed_metric_stagnation_no_budget_for_refresh_and_P2';save('stagnation_budget_gate',estimated_refresh_and_P2_seconds=estimated_refresh_and_P2_seconds);break
   cols=[];refresh={'at_control_sha256':current['control_sha256'],'hvp_started':0,'hvp_returned':0,'columns':[]}
   state['iterations'][-1]['metric_refresh']=refresh;gf_refresh=lambda z:torch.func.grad(lambda zz,pp:objective(zz,1),argnums=0)(z,p);eye=torch.eye(NT,dtype=y.dtype)
   for j in range(NT):
    required_after_column=(NT-j-1+50)*11.0+plan['budget']['reserved_forecast_and_verification_seconds']
    if time.monotonic()>=deadline or deadline-time.monotonic()<required_after_column or hvp_started+50>=hvp_cap:state['status']='refresh_H_budget_exhausted';save('refresh_H_partial',column=j,estimated_remaining_P2_seconds=required_after_column);break
    hvp_started+=1;refresh['hvp_started']+=1;save('refresh_H_column_started',column=j)
    col=torch.func.jvp(gf_refresh,(y,),(eye[j],))[1];cols.append(col);hvp_returned+=1;refresh['hvp_returned']+=1;refresh['columns'].append(col.tolist());save('refresh_H_column_returned',column=j)
   if state['status']=='refresh_H_budget_exhausted':break
   Hnew=torch.stack(cols,dim=1);Hsnew=(Hnew+Hnew.T)/2;eignew=torch.linalg.eigvalsh(Hsnew);hnnew=float(torch.linalg.matrix_norm(Hsnew));symnew=float(torch.linalg.matrix_norm(Hnew-Hnew.T)/max(hnnew,torch.finfo(Hnew.dtype).tiny));floornew=max(q.blocks.CURVATURE_REL_TOL*float(eignew.abs().max()),RM*torch.finfo(Hnew.dtype).eps*max(hnnew,torch.finfo(Hnew.dtype).tiny))
   if symnew>q.blocks.SYMMETRY_TOL:state['status']='refresh_H_asymmetric';save('refresh_H_symmetry_refusal',symmetry_relative=symnew);break
   pdnew=float(eignew[0])>floornew;targetnew=max(floornew,1e-3*hnnew);munew=0. if pdnew else max(0.,targetnew-float(eignew[0]))+RM*torch.finfo(Hnew.dtype).eps*hnnew
   metric_used=Hsnew+munew*torch.eye(NT,dtype=Hsnew.dtype);gf=torch.linalg.cholesky(metric_used);refresh.update(hessian_sha256=tsha(Hnew),eigenvalues=eignew.tolist(),symmetry_relative=symnew,positive_floor=floornew,positive_definite=pdnew,shift_mu=munew,metric_sha256=tsha(metric_used));save('refresh_H_complete');stagnant=0
 else:
  if current['first_order_pass']:state['status']='stationary_face_root';save('step_limit_stationary_root')
  else:state['status']='fixed_metric_step_limit_without_stationarity';save('step_limit_without_stationarity')
 # Same-point exact P2 is the only route to a local-minimum claim.
 if state['status']=='stationary_face_root':
  state['current_native_J']=current['native_J']
  if not current['singleface_supported'] or not current['strict_outward'] or not current['strict_theta'] or not current['first_order_pass']:
   state['status']='final_first_order_gate_failed';save('final_first_order_gate')
  else:
   metrics={};eye=torch.eye(NT,dtype=y.dtype);state['P2_columns_by_side']={}
   for side in (-1,1):
    cols=[];gf_side=lambda z,s=side:torch.func.grad(lambda v,pp:objective(v,s),argnums=0)(z,p)
    for j in range(NT):
     if time.monotonic()>=deadline or hvp_started>=hvp_cap:state['status']='P2_budget_exhausted';save('final_P2_partial',side=side,column=j);break
     hvp_started+=1;save('final_P2_column_started',side=side,column=j)
     col=torch.func.jvp(gf_side,(y,),(eye[j],))[1];cols.append(col);state['P2_columns_by_side'].setdefault(str(side),[]).append(col.tolist());hvp_returned+=1;save('final_P2_column_returned',side=side,column=j)
    if state['status']=='P2_budget_exhausted':break
    HH=torch.stack(cols,dim=1);SS=(HH+HH.T)/2;ev=torch.linalg.eigvalsh(SS);hn=float(torch.linalg.matrix_norm(SS));se=float(torch.linalg.matrix_norm(HH-HH.T)/max(hn,torch.finfo(HH.dtype).tiny));floor=max(q.blocks.CURVATURE_REL_TOL*float(ev.abs().max()),RM*torch.finfo(HH.dtype).eps*max(hn,torch.finfo(HH.dtype).tiny));metrics[str(side)]={'hessian_sha256':tsha(HH),'hessian':HH.tolist(),'eigenvalues':ev.tolist(),'lambda_min':float(ev[0]),'symmetry_relative':se,'positive_floor':floor,'positive_definite':se<=q.blocks.SYMMETRY_TOL and float(ev[0])>floor};save('final_P2_side_complete',side=side,P2_by_side=metrics)
   if state['status']=='stationary_face_root':
    state['P2_by_side']=metrics;state['minimum_claim']=len(metrics)==2 and all(v['positive_definite'] for v in metrics.values());state['status']='qualified_original_native_J_singleface_minimum' if state['minimum_claim'] else 'stationary_face_P2_failed'
    if native_calls>=native_cap:state['status']='P2_passed_native_J_recheck_cap_exhausted';state['minimum_claim']=False;save('final_qualification',P2_by_side=metrics,minimum_claim=False,final_point_diagnostics={k:v for k,v in current.items() if k!='control'})
    else:save('final_qualification',P2_by_side=metrics,minimum_claim=state['minimum_claim'],final_point_diagnostics={k:v for k,v in current.items() if k!='control'},native_J_recheck=current['native_J'])
 _,src_after,_=q._execution_plan(continuation_plan_path=CONT_PLAN,expected_continuation_sha256=plan['continuation_plan']['sha256']);identity_after=q.cycle.shared._input_identity(problem,original,lift(y),p,truth)
 state['source_after']=src_after;state['source_archive_closure']=source_before==src_after;state['input_after']=identity_after;state['script_sha256_after']=sha(SELF.read_bytes());state['script_unchanged']=state['script_sha256_after']==plan['source_files'][SELF.relative_to(ROOT).as_posix()];state['qualification_plan_sha256_after']=sha(QUAL_PLAN.read_bytes());state['source_unchanged']=state['script_unchanged'] and state['qualification_plan_sha256_after']==plan_sha;state['elapsed_s']=time.monotonic()-started;state['ledger']['run_elapsed_seconds']=state['elapsed_s'];state['ledger']['charged_at_end_seconds']=plan['budget']['charged_before_phase_seconds']+state['elapsed_s'];state['ledger']['remaining_after_phase_seconds']=3600.-state['ledger']['charged_at_end_seconds'];state['ledger']['reserved_forecast_and_verification_seconds']=plan['budget']['reserved_forecast_and_verification_seconds'];save('final_record');return state
if __name__=='__main__':
 r=run();print(json.dumps({k:r.get(k) for k in ('status','control_sha256','current_native_J','native_J_value_calls','gradient_calls','new_HVP_started','new_HVP_returned','accepted_steps','minimum_claim','elapsed_s','ledger')},indent=2))
