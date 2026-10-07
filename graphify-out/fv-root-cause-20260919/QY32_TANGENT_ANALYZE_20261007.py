"""Arithmetic of stored tangent-step values; no objective, gradient or HVP calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from examples.weather_scenarios import fv_point_3h_qy32_diagnostic as qy
from examples.weather_scenarios import fv_point_3h_qy32_tangent_step as tangent

HERE = Path(__file__).resolve().parent
rawpath = HERE / 'qy32_tangent_step_attempt1/step.json'
r = json.loads(rawpath.read_text())
old = json.loads(qy.STEP.read_text())
c0 = np.array(old['accepted_control'])
c1 = np.array(r['accepted_control'])
g0, g1 = np.array(r['base_gradient']), np.array(r['accepted_gradient'])
d, hd = np.array(r['direction']['direction']), np.array(r['H_direction'])
dt = np.array(r['direction']['retained_direction'])
eta, alpha = r['eta0'], r['trials'][-1]['alpha']
normal = lambda c: np.r_[np.zeros(21), np.array([-.08, -.21, -.1, -.45]) * (1-np.tanh(c[21:25])**2), 0.]
n0 = normal(c0)
checks = []
for tr in r['trials']:
    if 'objective' not in tr:
        continue
    a, g = tr['alpha'], np.array(tr['gradient'])
    point = tangent.chart_candidate(torch.tensor(c0), eta, torch.tensor(dt), a).numpy()
    n = normal(point)
    jlim = r['base_objective'] + 1e-4*a*(g0 @ d)
    plim = r['base_phi'] + 1e-4*a*(g0 @ hd)
    checks.append({'index': tr['index'], 'Phi_error': float(g @ g/2-tr['phi']),
        'J_threshold_error': float(jlim-tr['j_limit']), 'Phi_threshold_error': float(plim-tr['phi_limit']),
        'acceptance_matches': bool(tr['accepted']) == bool(tr['objective'] <= jlim and tr['phi'] <= plim),
        'tangent_projection_error': float(np.linalg.norm(g-n*(n @ g)/(n @ n))-tr['geometry']['tangent_gradient_norm']),
        'static_reconstructed_control_hash_matches': hashlib.sha256(point.tobytes()).hexdigest() == tr['control_sha256'],
        'actual_norm_error': float(np.linalg.norm(point-c0)-tr['displacement_l2'])})
result = {'scope': 'independent NumPy arithmetic and static chart reconstruction of archived step; no new FV/J/g/HVP',
    'raw_sha256': hashlib.sha256(rawpath.read_bytes()).hexdigest(),
    'accepted_control_hash_matches': hashlib.sha256(c1.tobytes()).hexdigest() == r['accepted_control_sha256'],
    'normal_dot_direction': float(n0 @ d), 'gTd': float(g0 @ d), 'gTHd': float(g0 @ hd),
    'new_gradient_dot_old_direction': float(g1 @ d),
    'gradient_linear_error_l2': float(np.linalg.norm(g1-g0-alpha*hd)),
    'gradient_linear_relative_error': float(np.linalg.norm(g1-g0-alpha*hd)/np.linalg.norm(g1)),
    'finite_path_delta_minus_alpha_d_l2': float(np.linalg.norm(c1-c0-alpha*d)),
    'Phi_error_final': float(g1 @ g1/2-r['accepted_phi']), 'trials': checks,
    'source_closed': r['source_before'] == r['source_after'],
    'fixed_input_unchanged': r['fixed_input_unchanged'], 'runtime_closed': r['runtime'] == r['runtime_after']}
(HERE / 'QY32_TANGENT_INDEPENDENT_CHECKS_20261007.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'trials'}, indent=2))
print('all trial acceptance/hash checks', all(c['acceptance_matches'] and c['static_reconstructed_control_hash_matches'] for c in checks))
