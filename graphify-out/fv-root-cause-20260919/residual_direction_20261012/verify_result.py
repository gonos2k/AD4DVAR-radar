"""Stored-vector and receipt audit; performs no FV/HVP/score evaluation."""
from pathlib import Path
import hashlib
import json
import math
import sys
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import direction_trial as driver
trial=driver.trial
w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=(w/'PLAN_SHA256.txt').read_text().strip();plan=driver.load_plan(driver.PLAN,digest)
s=json.loads((w/'attempt1/step.json').read_text());run=json.loads((w/'attempt1/step.run.json').read_text())
checks={'raw_run_link':run['child_sha256']==sha(w/'attempt1/step.json'),
    'source_pin_closure':s.get('source_before')==s.get('source_after')=={**plan['source_files'],**plan['archive_files'],driver.PLAN.relative_to(ROOT).as_posix():digest},
    'fixed_input':s.get('fixed_input_unchanged') is True,'runtime':s.get('runtime_before')==s.get('runtime_after') and s.get('runtime_unchanged') is True,
    'caps':s['hvp_started']<=4 and s['hvp_completed']<=s['hvp_started'] and s['candidate_evaluations_started']<=12 and s['candidate_evaluations_completed']<=s['candidate_evaluations_started'] and s['optimizer_steps_applied']<=1,
    'no_future_selection':s['terminal_E_recomputed'] is False and plan['policy']['future_score_evaluations']==0,
    'no_root_or_minimum_claim':all(s[k] is False for k in ('root_claim','minimum_claim','response_claim','score_claim','original_problem_solved')),
    'resource_closed':run['resource'].get('exit_code')==0 and run['resource'].get('resource_termination') is None and run['resource'].get('monitor_error') is None and run['resource']['sampled_peak_rss_bytes']<plan['policy']['rss_bytes']}
base=driver.endpoint.load_endpoint()
checks['base_is_2a4d']=s['base_control_sha256']==driver.BASE_SHA
checks['fresh_HVP_base']=all(e['control_sha256']==driver.BASE_SHA for e in s['hvp_history'])
rows=s.get('GN_operator',{});metrics={'execution_status':s['execution_status'],'numerical_status':s['numerical_status'],
    'termination_reason':s['termination_reason'],'elapsed_seconds':s['elapsed_seconds'],'hvp_started':s['hvp_started'],'hvp_completed':s['hvp_completed'],
    'row_vjps':rows.get('jacobian_rows_completed',0),'GN_solves':rows.get('dense_solves_completed',0),
    'candidates':s['candidate_evaluations_completed'],'commits':s['optimizer_steps_applied'],'control_sha256':s['current_control_sha256'],'arms':[]}
if len(s['fresh_models'])==2:
    G=torch.tensor(base['full_G'],dtype=torch.float64) if isinstance(base['full_G'],list) else base['full_G']
    D=torch.stack([torch.tensor(m['direction'],dtype=G.dtype) for m in s['fresh_models']],1)
    V=torch.stack([torch.tensor(m['residual_direction'],dtype=G.dtype) for m in s['fresh_models']],1)
    for arm in s['arms']:
        for t in arm['trials']:
            p=ROOT/t['candidate_receipt'];payload=json.loads(p.read_text())
            checks[f"{arm['name']}_{t['index']}_receipt"]=sha(p)==t['candidate_receipt_sha256'] and payload=={k:v for k,v in t.items() if k not in ('candidate_receipt','candidate_receipt_sha256')}
            coeff=torch.tensor(t['model_coefficients'],dtype=G.dtype);r=torch.tensor(t['DGd'],dtype=G.dtype)
            checks[f"{arm['name']}_{t['index']}_DG_linearity"]=trial.finite._full_G_match(V@coeff,r)['passed']
            checks[f"{arm['name']}_{t['index']}_radius"]=t['actual_path_norm']<=t['radius'] and t['chart_calls']<=33 and all(x>=0 for x in t['coefficients'])
            if 'G' in t:
                fidelity=trial.model_fidelity(G,t['chart_alpha']*r,torch.tensor(t['G'],dtype=G.dtype))
                checks[f"{arm['name']}_{t['index']}_fidelity"]=all(math.isclose(t['fidelity'][k],v,rel_tol=1e-12,abs_tol=1e-14) for k,v in fidelity.items() if math.isfinite(v))
        if arm['status']=='independent_P2_fullG_eligible':
            t=arm['candidate'];p=arm['proposal'];rep=arm['repeat'];full=arm['full_G_check']
            checks[arm['name']+'_repeat']=trial.tangent.fresh_final_closure(p,rep) and full['passed'] and rep['objective']<=p['J_armijo_bound'] and rep['F_squared']<=p['F_squared_armijo_bound']
            metrics['arms'].append({'name':arm['name'],'J':rep['objective'],'R':rep['F_squared'],
                'J_decrease_percent':100*(s['base_J']-rep['objective'])/s['base_J'],
                'R_decrease_percent':100*(s['base_R']-rep['F_squared'])/s['base_R'],
                'mixed_gradient_max':max(abs(x) for x in full['G'][:26]),'candidate_count':len(arm['trials']),
                'coefficients':t['coefficients'],'actual_path_norm':t['actual_path_norm'],'fidelity':t['fidelity'],
                'branch_matches_base':t['branch_signature_matches_base'],'search_seconds':arm['search_seconds'],
                'inclusive_shared_preparation_seconds':s['common_preparation_seconds']+arm['search_seconds']})
        else:metrics['arms'].append({'name':arm['name'],'status':arm['status'],'candidate_count':len(arm['trials'])})
winner=trial.comparison.choose_repeated_arm(s['arms'],torch.float64)
checks['commit_requires_both_closed']=(s['optimizer_steps_applied']==0 if winner is None else s['selected_direction']==winner['name'] and s['optimizer_steps_applied']==1)
checks['current_control_hash']=trial.tangent._tensor_sha(torch.tensor(s['current_control'],dtype=torch.float64))==s['current_control_sha256']
if s['optimizer_steps_applied']==0:checks['base_preserved']=s['current_control']==base['control'].tolist() and s['current_control_sha256']==driver.BASE_SHA
else:
    checks['winner_lineage']=s['closed_steps'][0]['base_control_sha256']==driver.BASE_SHA and s['closed_steps'][0]['final_repeat']==winner['repeat']
    metrics.update(selected_direction=s['selected_direction'],final_R=float(torch.dot(torch.tensor(s['current_full_G'],dtype=torch.float64),torch.tensor(s['current_full_G'],dtype=torch.float64))))
report={'checks':checks,'all_checks_passed':all(checks.values()),'metrics':metrics,'scope':'stored-vector arithmetic and pinned receipt audit; no FV/HVP regeneration, no exact adjoint/minimum/independent forecast claim'}
(w/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');(w/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({'all_checks_passed':report['all_checks_passed'],'failed':[k for k,v in checks.items() if not v],'metrics':metrics},indent=2))
if not report['all_checks_passed']:raise SystemExit(1)
