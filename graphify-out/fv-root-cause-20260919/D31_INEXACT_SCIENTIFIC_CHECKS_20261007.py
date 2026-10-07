"""Replay saved arm arrays and receipts; never regenerate FV or HVPs."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/Users/yhlee/ADVAR')
E = ROOT / 'graphify-out/fv-root-cause-20260919'
D = E / 'd31_inexact_comparison_20261007_attempt1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tensor_sha = lambda x: hashlib.sha256(np.asarray(x, dtype=np.float64).tobytes()).hexdigest()
r = json.loads((D/'step.json').read_text())
p = json.loads((E/'D31_COMMITTED_BASE_COMPARISON_PLAN_20261007.json').read_text())
parent = json.loads((D/'step.run.json').read_text())
resource = json.loads((D/'step.resource.json').read_text())
base = json.loads((ROOT/p['base_step']).read_text())
assert parent['child_sha256'] == sha(D/'step.json')
assert r['source_before'] == r['source_after'] and r['fixed_input_unchanged']
assert all(sha(ROOT/n) == h for n,h in {**p['source_files'], **p['archive_files']}.items())
summary = {}
for mode, arm in r['arms'].items():
    assert arm.get('base_control_sha256', base['accepted_control_sha256']) == base['accepted_control_sha256']
    row = {'numerical_status': arm['numerical_status'], 'committed': arm['optimizer_steps_applied'],
           'costs': r['arm_counter_deltas'].get(mode), 'elapsed_seconds': arm.get('elapsed_seconds')}
    iterations = arm['iterations'] + ([arm['current_iteration']] if arm.get('current_iteration') else [])
    for it in iterations:
        if not it.get('solve'):
            continue
        g = np.asarray(it['base_state']['gradient']);s = np.asarray(it['direction']);hs = np.asarray(it['H_s'])
        rel = float(np.linalg.norm(hs+g)/np.linalg.norm(g))
        assert rel <= it['solve']['rtol'] and float(g@s) < 0 and float(g@hs) < 0
        assert math.isclose(rel, it['solve']['true_relative_residual'], abs_tol=1e-14)
        row.update(rtol=it['solve']['rtol'], true_relative_residual=rel,
                   pcg_iterations=it['solve']['iterations'], direction_norm=float(np.linalg.norm(s)))
        if it.get('status') == 'accepted':
            t = next(t for t in it['trials'] if t['status']=='accepted')
            candidate = np.asarray(it['accepted_control']);initial = np.asarray(base['accepted_control'])
            np.testing.assert_array_equal(candidate, initial+t['alpha']*s)
            assert tensor_sha(candidate)==it['accepted_control_sha256']
            assert t['J_armijo_passed'] and t['Phi_armijo_passed'] and t['strict_point_passed']
            assert math.isclose(.5*np.dot(t['gradient'],t['gradient']),t['phi'],abs_tol=1e-14)
            row.update(control_sha256=it['accepted_control_sha256'], J=t['objective'], Phi=t['phi'],
                       gradient_inf=max(map(abs,t['gradient'])), alpha=t['alpha'],
                       step_l2=it['displacement_l2'], candidates=len(it['trials']))
    summary[mode] = row
assert r['hvp_calls'] <= 90 and r['hvp_calls_completed'] <= r['hvp_calls']
assert sum(d['hvp_calls'] for d in r['arm_counter_deltas'].values()) <= r['hvp_calls']
# A raised setup/operator callback can leave a partial arm outside completed deltas.
result_unattributed = r['hvp_calls'] - sum(d['hvp_calls'] for d in r['arm_counter_deltas'].values())
result = {'scope':'saved-array/receipt arithmetic only; strict-first remaining-budget order, not equal reserves or global speedup',
          'execution':parent['execution_status'],'numerical':r['numerical_status'],
          'comparison_complete':r['comparison_complete'],'selected_arm':r['selected_arm'],
          'raw_sha256':sha(D/'step.json'),'arms':summary,'HVP':r['hvp_calls'],
          'partial_arm_HVP_outside_completed_deltas':result_unattributed,
          'guard_seconds':resource['elapsed_seconds'],'sampled_peak_RSS_bytes':resource['sampled_peak_rss_bytes'],
          'remaining':['eligible stationary point/full curvature/adjoint/reanalysis','independent forecast skill']}
(E/'D31_INEXACT_RESULT_20261007.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
