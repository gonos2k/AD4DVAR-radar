"""Stored-value arithmetic only; no FV, objective, gradient or HVP evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rawpath = HERE / 'model_guided_attempt1/step.json'
r = json.loads(rawpath.read_text())
old = json.loads((HERE / 'qy32_tangent_step_attempt1/step.json').read_text())
resource = json.loads((HERE / 'model_guided_attempt1/step.resource.json').read_text())
g0 = np.array(old['accepted_gradient'])
gf = np.array(r['current_state']['gradient'])
points = []
checks = []
cbase = np.array(old['accepted_control'])
previous_partition = old['accepted_branch_partition']
for row in r['iterations']:
    g = np.array(row['accepted_gradient'])
    gb = np.array(row['base_gradient'])
    d = np.array(row['direction'])
    hd = np.array(row['H_direction'])
    c = np.array(row['accepted_control'])
    trial = row['trials'][-1]
    alpha = trial['alpha']
    slope_phi = float(gb @ hd)
    phi_alpha = -slope_phi / float(hd @ hd)
    capped = min(1., .05/float(np.linalg.norm(d)), phi_alpha)
    linear = gb + alpha*hd
    partition = trial['branch_partition']
    points.append({'index': row['index'], 'control_sha256': row['accepted_control_sha256'],
        'J': row['accepted_objective'], 'Phi': row['accepted_phi'],
        'gL2': float(np.linalg.norm(g)), 'gInf': float(abs(g).max()),
        'alpha_start': row['alpha_start'], 'accepted_alpha': alpha,
        'candidates': len(row['trials']), 'move': row['accepted_displacement_l2'], 'Qy32': trial['eta'],
        'tangent_gL2': row['accepted_geometry']['tangent_gradient_norm'],
        'normal_gradient': row['accepted_geometry']['euclidean_unit_normal_gradient'],
        'analysis_branch_changed': partition['analysis_sha256'] != previous_partition['analysis_sha256'],
        'future_branch_changed': partition['future_sha256'] != previous_partition['future_sha256'],
        'block_norms': {name: float(np.linalg.norm(g[a:b])) for name, (a,b) in
                       {'field': (0,20), 'flow': (20,25), 'growth': (25,26)}.items()}})
    checks.append({'index': row['index'], 'negative_current_gradient': bool(np.array_equal(d, -gb)),
        'accepted_control_hash_matches': hashlib.sha256(c.tobytes()).hexdigest() == row['accepted_control_sha256'],
        'affine_control_hash_matches': hashlib.sha256((cbase+alpha*d).tobytes()).hexdigest() == row['accepted_control_sha256'],
        'alpha_model_error': float(row['alpha_start']-capped),
        'Phi_error': float(g @ g/2-row['accepted_phi']),
        'displacement_error': float(np.linalg.norm(c-cbase)-row['accepted_displacement_l2']),
        'gradient_linear_relative_error': float(np.linalg.norm(g-linear)/np.linalg.norm(g)),
        'J_threshold_error': float(trial['j_limit']-(row['base_objective']+1e-4*alpha*float(gb @ d))),
        'Phi_threshold_error': float(trial['phi_limit']-(row['base_phi']+1e-4*alpha*slope_phi)),
        'actual_J_armijo': row['accepted_objective'] <= trial['j_limit'],
        'actual_Phi_armijo': row['accepted_phi'] <= trial['phi_limit']})
    cbase, previous_partition = c, partition
summary = {'scope': 'three bounded full26 model-guided gradient steps; not Newton nor final root',
    'raw_sha256': hashlib.sha256(rawpath.read_bytes()).hexdigest(),
    'initial_control_sha256': old['accepted_control_sha256'], 'final_control_sha256': r['current_control_sha256'],
    'base_J': old['accepted_objective'], 'base_Phi': old['accepted_phi'],
    'base_gL2': float(np.linalg.norm(g0)), 'base_gInf': float(abs(g0).max()),
    'final_J': r['current_state']['objective'], 'final_Phi': r['current_state']['phi'],
    'final_gL2': float(np.linalg.norm(gf)), 'final_gInf': float(abs(gf).max()),
    'J_percent_decrease': 100*(old['accepted_objective']-r['current_state']['objective'])/old['accepted_objective'],
    'Phi_percent_decrease': 100*(old['accepted_phi']-r['current_state']['phi'])/old['accepted_phi'],
    'gL2_percent_decrease': 100*(np.linalg.norm(g0)-np.linalg.norm(gf))/np.linalg.norm(g0),
    'gInf_percent_decrease': float(100*(abs(g0).max()-abs(gf).max())/abs(g0).max()),
    'hvp_started': r['hvp_calls'], 'hvp_completed': r['hvp_calls_completed'],
    'numerical_status': r['numerical_status'], 'iterations': points,
    'resource': {k: resource.get(k) for k in ['elapsed_seconds','sampled_peak_rss_bytes','exit_code','resource_termination']},
    'remaining': ['original full26 eligible3h root', 'current full curvature/original adjoint/VJP/reanalysis',
                  'independent synthetic future verification']}
(HERE / 'AMODEL_RESULT_20261007.json').write_text(json.dumps(summary, indent=2)+'\n')
(HERE / 'AMODEL_INDEPENDENT_CHECKS_20261007.json').write_text(json.dumps(checks, indent=2)+'\n')
print(json.dumps(summary, indent=2))
