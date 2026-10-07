"""Arithmetic and signature comparisons only; does not evaluate FV or derivatives."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'qy32_diagnostic_attempt1/diagnostic.json'
OUT = HERE / 'QY32_RESULT_20261007.json'


def compare(left, right):
    a, b = left.get('branch_signature'), right.get('branch_signature')
    changes = None
    if isinstance(a, dict) and isinstance(b, dict):
        changes = {}
        for kind in ('choices', 'face_signs'):
            indices = [k for k, (x, y) in enumerate(zip(a[kind], b[kind], strict=True)) if x != y]
            changes[kind] = {'analysis_stage_indices': [k for k in indices if k < 360],
                             'future_stage_indices': [k for k in indices if k >= 360],
                             'stage_count': len(indices)}
    delta_eta = right['requested_eta'] - left['requested_eta']
    return {'etas': [left['requested_eta'], right['requested_eta']],
            'delta_J': right['objective'] - left['objective'],
            'delta_Phi': right['phi'] - left['phi'],
            'finite_interval_J_secant': (right['objective'] - left['objective']) / delta_eta,
            'branch_stage_changes': changes}


def main():
    report = json.loads(RAW.read_text())
    samples = {s['requested_eta']: s for s in report['samples']}
    summary = {'scope': 'finite fixed-retained-coordinate samples; no exact one-sided limit or single-face causal proof',
               'raw_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
               'numerical_status': report['numerical_status'],
               'execution_status': report['execution_status'],
               'source_unchanged': report['source_unchanged'],
               'fixed_input_unchanged': report['fixed_input_unchanged'],
               'base_control_sha256': report['base_control_sha256'],
               'zero_cost_only': report['eta_zero_cost_only'],
               'points': [], 'pairs': {},
               'optimizer_steps_applied': report['optimizer_steps_applied'],
               'hvp_calls': report['hvp_calls'], 'pcg_solves': report['pcg_solves'],
               'remaining': ['eligible original3h root', 'current full curvature', 'original adjoint/reanalysis',
                             'independent synthetic future score']}
    for eta, s in samples.items():
        summary['points'].append({k: s[k] for k in ('requested_eta', 'realized_qy32', 'objective', 'phi',
                                                   'gradient_norm', 'gradient_inf', 'geometry', 'branch', 'branch_partition')})
    for name, (a, b) in {'negative_same_side': (-2e-6, -1e-6), 'positive_same_side': (1e-6, 2e-6),
                         'inner_cross_side': (-1e-6, 1e-6)}.items():
        if a in samples and b in samples:
            summary['pairs'][name] = compare(samples[a], samples[b])
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'points': len(summary['points']), 'pairs': summary['pairs']}, indent=2))


if __name__ == '__main__':
    main()
