"""Audit exact accepted-prefix receipts; no physical model or operator calls."""
from pathlib import Path
import hashlib,json,math,sys
import torch
ROOT=Path(__file__).resolve().parents[3];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
import cached_gradient_progress as p
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
digest=(WORK/'PROGRESS_PLAN_SHA256.txt').read_text().strip();plan=p.load_plan(p.PLAN,digest)
s=json.loads((WORK/'progress1/step.json').read_text());r=json.loads((WORK/'progress1/step.run.json').read_text());resource=json.loads((WORK/'progress1/step.resource.json').read_text());base=p.load_endpoint()
checks={'raw_run_resource_link':r['child_sha256']==sha(WORK/'progress1/step.json') and r['resource']==resource,
 'source_before_after':s['source_before']==s['source_after']=={**plan['source_files'],**plan['archive_files'],p.PLAN.relative_to(ROOT).as_posix():digest},
 'input_runtime':s['fixed_input_unchanged'] and s['runtime_unchanged'] and s['runtime_before']==s['runtime_after'],
 'resource_guard':resource['exit_code']==0 and resource['resource_termination'] is None and resource['monitor_error'] is None and resource['sampled_peak_rss_bytes']<p.POLICY['rss_bytes'] and resource['elapsed_seconds']+p.PRIOR_SECONDS<=p.POLICY['outer_seconds'],
 'caps':s['candidate_evaluations_started']<=24 and s['hvp_started']<=36 and len(s['closed_steps'])<=19,
 'historical_accounting':s['cached_HVP_calls_reused']==2 and s['cached_historical_HVP_calls']==4 and s['hvp_completed']+4<=40 and r['charged_prior_preparation_seconds']==p.PRIOR_SECONDS,
 'no_claims_scores':all(s[k] is False for k in ('root_claim','minimum_claim','response_claim','score_claim','original_problem_solved','terminal_E_recomputed')) and plan['policy']['future_score_evaluations']==0,
 'budget_stop_explicit':s['numerical_status']=='budget_exhausted' and s['deadline_passed'] is False}
previous_sha=p.BASE_SHA;previous_G=base['full_G'];previous_J=base['repeat']['objective'];previous_R=base['repeat']['F_squared'];history=[]
for i,step in enumerate(s['closed_steps']):
 proposal,repeat,full,chosen=step['proposal'],step['final_repeat'],step['full_G_check'],step['selected_candidate']
 file=ROOT/chosen['candidate_receipt'];facts=json.loads(file.read_text())
 checks[f'{i}_receipt']=sha(file)==chosen['candidate_receipt_sha256'] and all(facts[k]==chosen[k] for k in ('alpha','objective','F_squared','theta','G'))
 checks[f'{i}_lineage']=step['point_index']==i and step['base_control_sha256']==previous_sha and p.continuation.finite._full_G_match(torch.tensor(step['full_G'],dtype=torch.float64),previous_G)['passed']
 checks[f'{i}_qualification']=step['previous_P2_match']['passed'] and step['minimum_mixture_P2_match']['passed'] and not step['outward_derivatives']['release_required'] and p.continuation.tangent.fresh_final_closure(proposal,repeat) and full['passed']
 checks[f'{i}_JR']=repeat['objective']<=proposal['J_armijo_bound'] and repeat['F_squared']<=proposal['F_squared_armijo_bound'] and repeat['objective']<previous_J and repeat['F_squared']<previous_R
 checks[f'{i}_radius']=chosen['actual_path_norm']<=p.RADIUS*(1+64*torch.finfo(torch.float64).eps) and chosen['chart_calls']<=33
 checks[f'{i}_repeated_G']=p.continuation.fresh_full_G(repeat,torch.tensor(full['G'],dtype=torch.float64),__import__('examples.weather_scenarios.fv_point_3h_netflux_endpoint',fromlist=['FACE_WEIGHTS']).FACE_WEIGHTS)['passed']
 if i==0:checks['initial_cache_requalification']=step['cached_HVP_calls_reused']==2 and step['cached_DGd_requalification']['passed'] and step['hvp_completed']==0
 else:checks[f'{i}_fresh_pair']=step['hvp_started']==step['hvp_completed']==2
 previous_sha=proposal['control_sha256'];previous_G=torch.tensor(full['G'],dtype=torch.float64);previous_J=repeat['objective'];previous_R=repeat['F_squared']
 history.append({'point_index':i,'control_sha256':previous_sha,'J':previous_J,'R':previous_R,'mixed_gradient_max':max(abs(x) for x in full['G'][:26]),'fidelity':chosen.get('fidelity')})
checks['last_confirmed_prefix']=s['current_control_sha256']==s['last_confirmed_control_sha256']==previous_sha and s['current_full_G']==s['last_confirmed_full_G']==previous_G.tolist() and p.continuation.tangent._tensor_sha(torch.tensor(s['current_control'],dtype=torch.float64))==previous_sha
checks['no_phantom_JVP']=sum(e['stage']=='qualified_complete' for e in s['hvp_history'])==s['hvp_completed'] and s['hvp_history'][-1]['stage']=='precheck'
report={'checks':checks,'all_checks_passed':all(checks.values()),'history':history,
 'time_caveat':{'charged_child_seconds':s['elapsed_seconds']+p.PRIOR_SECONDS,'internal_overrun_seconds':s['elapsed_seconds']+p.PRIOR_SECONDS-600,'charged_guarded_seconds':r['total_guarded_seconds_including_preparation'],'guarded_over_internal_seconds':r['total_guarded_seconds_including_preparation']-600,'outer_margin_seconds':660-r['total_guarded_seconds_including_preparation'],'cause':'not instrumented; no JVP/candidate/commit at final precheck'},
 'scope':'exact receipt/hash/KKT/native-P2/full-G and arithmetic audit; no independent FV/HVP regeneration or forecast evaluation'}
(WORK/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'all_checks_passed':report['all_checks_passed'],'failed':[k for k,v in checks.items() if not v],'accepted_points':len(history),'time_caveat':report['time_caveat']},indent=2))
assert report['all_checks_passed']
