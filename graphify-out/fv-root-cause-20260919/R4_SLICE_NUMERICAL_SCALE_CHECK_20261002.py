"""Recalculate saved Newton model scales; no FV evaluation or policy change."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw = ROOT / 'partial_signed_face_slices_attempt1/signed_face_slices.json'
data = json.loads(raw.read_text())
slice_record = next(row for row in data['slices'] if row['factor'] == 2)
solve = slice_record['linear_solves'][-1]
rhs = solve['rhs']
step = solve['solution']
residual = solve['independently_recomputed_residual']
# g=-rhs and Hs=rhs+r, as independently audited by the parent.
linear_term = -math.fsum(b * s for b, s in zip(rhs, step, strict=True))
curvature_term = math.fsum(s * (b + r) for s, b, r in zip(step, rhs, residual, strict=True))
quadratic_delta = linear_term + 0.5 * curvature_term
objective = slice_record['accepted_steps'][-1]['objective']
trials = [row for row in slice_record['rejected_trials'] if row['iteration'] == 2]
assert linear_term < 0 and curvature_term > 0 and quadratic_delta < 0
assert len(trials) == 16 and all(row['rejection'] == 'policy' for row in trials)
result = {
    'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
    'scope': 'saved-vector arithmetic; not fresh FV evaluation, error bound or accepted endpoint',
    'linear_term': linear_term,
    'curvature_term': curvature_term,
    'quadratic_predicted_delta_full_step': quadratic_delta,
    'objective': objective,
    'objective_ulp': math.ulp(objective),
    'predicted_decrease_in_objective_ulps': -quadratic_delta / math.ulp(objective),
    'step_l2': math.sqrt(math.fsum(s * s for s in step)),
    'rhs_l2': math.sqrt(math.fsum(b * b for b in rhs)),
    'trials': [{
        'step_scale': row['step_scale'],
        'model_delta': row['step_scale'] * linear_term + 0.5 * row['step_scale']**2 * curvature_term,
        'recorded_objective_delta': row['objective'] - objective,
        'policy_allowance': 128 * 2**-52 * max(abs(objective), abs(row['objective'])),
        'armijo_ratio': row['armijo_ratio'],
    } for row in trials],
    'interpretation': 'local quadratic predicted decrease is below one objective ulp; observed increase is not proved to be roundoff; original refusal retained',
}
(ROOT / 'R4_SLICE_NUMERICAL_SCALE_CHECK_20261002.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'trials'}, indent=2))
