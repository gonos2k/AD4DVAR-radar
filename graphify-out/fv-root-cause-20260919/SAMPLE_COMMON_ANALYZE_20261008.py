"""Saved-vector arithmetic only; no production FV or derivative calls."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
raw = HERE / 'sample_common_search_20261008_attempt1/step.json'
x = json.loads(raw.read_text())
parent = json.loads(raw.with_suffix('.run.json').read_text())
resource = json.loads(raw.with_suffix('.resource.json').read_text())
c0, c1 = (np.array(x[k], dtype='<f8') for k in ('base_control', 'accepted_control'))
g0, g1 = (np.array(x[k]) for k in ('base_gradient', 'accepted_gradient'))
d = np.array(x['sample_direction']['direction']); hd = np.array(x['H_direction'])
a = x['actual_alpha']; trial = x['trials'][-1]
weights = np.array([0, -.08, -.28, -.14, -.84])
def geometry(c, g):
    n = np.zeros(26); n[20:25] = weights * (1-np.tanh(c[20:25])**2)
    unit = n / np.linalg.norm(n)
    return {'Qy43': float(weights @ np.tanh(c[20:25])),
            'normal_gradient': float(unit @ g),
            'tangent_gradient_norm': float(np.linalg.norm(g-unit*(unit@g)))}
checks = {
 'base_control_hash': hashlib.sha256(c0.tobytes()).hexdigest()==x['base_control_sha256'],
 'accepted_control_hash': hashlib.sha256(c1.tobytes()).hexdigest()==x['accepted_control_sha256'],
 'parent_raw_hash': hashlib.sha256(raw.read_bytes()).hexdigest()==parent['child_sha256'],
 'step_affine': bool(np.allclose(c1, c0+a*d, rtol=0, atol=1e-17)),
 'J_Armijo': x['accepted_objective'] <= x['base_objective'] + 1e-4*a*float(g0@d),
 'Phi_base': bool(np.isclose(g0@g0/2,x['base_phi_diagnostic'],rtol=1e-14)),
 'Phi_accepted': bool(np.isclose(g1@g1/2,x['accepted_phi_diagnostic'],rtol=1e-14)),
 'actual_alpha': a==trial['alpha']*x['scale_audit']['initial_alpha'],
 'radius': bool(np.linalg.norm(c1-c0)<=.05),
 'closure': x['source_unchanged'] is True and x['fixed_input_unchanged'] is True and x['runtime']==x['runtime_after'],
 'final_recheck': x['final_recheck']['passed'] is True,
 'single_HVP': x['hvp_calls_started']==x['hvp_calls_completed']==1,
 'committed': x['candidate_committed'] is True and x['optimizer_steps_applied']==1,
}
assert all(checks.values()), checks
out = {
 'scope':'saved vectors, static flow geometry and receipts only; no new FV/HVP',
 'checks':checks, 'J_change_percent':100*(x['accepted_objective']/x['base_objective']-1),
 'Phi_change_percent':100*(x['accepted_phi_diagnostic']/x['base_phi_diagnostic']-1),
 'gInf_before':float(abs(g0).max()), 'gInf_after':float(abs(g1).max()),
 'gradient_norm_before':float(np.linalg.norm(g0)), 'gradient_norm_after':float(np.linalg.norm(g1)),
 'actual_alpha':a,'actual_step_norm':float(np.linalg.norm(c1-c0)),
 'g_dot_d':float(g0@d),'d_dot_Hd':float(d@hd),'g_dot_Hd':float(g0@hd),
 'new_g_dot_old_direction':float(g1@d),
 'J_armijo_upper':float(x['base_objective']+1e-4*a*(g0@d)),
 'J_quadratic_model':float(x['base_objective']+a*(g0@d)+.5*a*a*(d@hd)),
 'Phi_linear_gradient_model':float(np.linalg.norm(g0+a*hd)**2/2),
 'gradient_linearization_error_norm':float(np.linalg.norm(g1-g0-a*hd)),
 'base_geometry':geometry(c0,g0),'accepted_geometry':geometry(c1,g1),
 'blocks':{name:{'before':float(np.linalg.norm(g0[sl])), 'after':float(np.linalg.norm(g1[sl]))}
           for name,sl in [('field',slice(0,20)),('flow',slice(20,25)),('growth',slice(25,26))]},
 'branch_signature_changed':trial['branch_signature_changed'],
 'resource':{k:resource[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','exit_code']},
 'accepted_control_sha256':x['accepted_control_sha256'],
 'remaining':['original eligible stationary point','curvature/original adjoint/reanalysis','independent synthetic future verification'],
}
(HERE/'SAMPLE_COMMON_RESULT_20261008.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
