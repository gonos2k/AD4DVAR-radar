"""Re-arithmetic of saved sector arrays; no model or derivative execution."""
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
RAW = EVIDENCE / 'active_intersection_20261011_attempt1/diagnostic.json'
record = json.loads(RAW.read_text())
resource = json.loads(RAW.with_name('diagnostic.resource.json').read_text())
basis = torch.tensor(record['normal_audit']['normal_basis'], dtype=torch.float64)
normal = basis[:, 0]
event_gradient = torch.tensor(record['root_event_gradients'], dtype=torch.float64).mean(dim=0)
w = event_gradient - normal * torch.dot(normal, event_gradient)
w = w / torch.linalg.vector_norm(w)
summaries = []
previous = None
for saved in record['sector_summaries']:
    samples = [sample for sample in record['sector_samples'] if sample['radius'] == saved['radius']]
    gradients = torch.tensor([sample['native_gradient'] for sample in samples], dtype=torch.float64)
    weights = torch.tensor(saved['sampled_hull']['weights'], dtype=torch.float64)
    mixture = weights @ gradients
    merit = float(torch.dot(mixture, mixture))
    assert abs(merit - saved['sampled_hull']['squared_norm']) <= 128 * torch.finfo(torch.float64).eps * merit
    assert all(sample['reachable_sector'] and sample['branch_changes']['only_declared_event_changed'] for sample in samples)
    limiter_jumps = []
    for index in (0, 2):
        jump = gradients[index + 1] - gradients[index]
        jump = jump - normal * torch.dot(normal, jump)
        perpendicular = jump - w * torch.dot(w, jump)
        limiter_jumps.append({'projected_norm': float(jump.norm()),
            'w_perpendicular_relative_norm': float(perpendicular.norm() / jump.norm())})
    external_jumps = []
    for index in (0, 1):
        jump = gradients[index + 2] - gradients[index]
        external_jumps.append({'norm': float(jump.norm()),
            'n_perpendicular_relative_norm': float((jump-normal*torch.dot(normal,jump)).norm()/jump.norm())})
    summaries.append({'radius': saved['radius'], 'sampled_hull_norm': math.sqrt(merit),
        'sampled_hull_weights': saved['sampled_hull']['weights'],
        'common_tangent_gradient_norm': saved['common_tangent_gradient_norm'],
        'common_tangent_projection_spread': saved['common_tangent_projection_spread'],
        'sampled_common_direction_cost_slopes': saved['sampled_cost_slopes'],
        'successive_sector_gradient_max_change': float((gradients-previous).norm(dim=1).max()) if previous is not None else None,
        'limiter_jumps': limiter_jumps,'external_jumps':external_jumps})
    previous = gradients
result = {'scope':'saved production arrays re-arithmetic; finite-radius gradients, not exact limiting or minimum certification',
    'raw_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(), 'base_control_sha256':record['base_control_sha256'],
    'canonical_control_unchanged':True, 'optimizer_commits':0, 'production_guards':1,
    'base_J':record['base_J'],'base_R':record['base_R'],
    'base_zeta':record['base_event_values'][0][0]-record['base_event_values'][0][1],
    'first_order_event_distance':record['first_order_event_distance'],
    'intersection_Q':record['intersection_Q'],'intersection_zeta':record['intersection_zeta'],
    'root_event_roundoff_bound':record['root_event_roundoff_bound'],
    'intersection_native_J':record['intersection_native_J'],
    'intersection_J_minus_canonical':record['intersection_native_J']-record['base_J'],
    'normal_singular_values':record['normal_audit']['normal_singular_values'],
    'event_gradient_calls':record['event_grad_completed'], 'native_sector_gradient_calls':record['native_gradient_completed'],
    'root_value_evaluations':len(record['root_evaluations']), 'native_sector_samples':len(record['sector_samples']),
    'tangent_cost_samples':record['tangent_cost_samples'], 'summaries':summaries,
    'guard_seconds':resource['elapsed_seconds'], 'sampled_peak_rss_bytes':resource['sampled_peak_rss_bytes'],
    'resource_termination':resource['resource_termination'],'exit_code':resource['exit_code'],
    'completion_percent':70,'actual_FV_adjoint_completed':False,'current_FV_forecast_score_completed':False}
(EVIDENCE/'ACTIVE_INTERSECTION_METRICS_20261011.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ('base_zeta','intersection_zeta','intersection_J_minus_canonical','guard_seconds','optimizer_commits')}))
