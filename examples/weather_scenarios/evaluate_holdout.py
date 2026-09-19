"""Evaluate a saved CP3b candidate report without rerunning transport."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Recorded before CP3b execution; do not loosen thresholds by replacing both inputs.
CP3B_MANIFEST_SHA256 = '10ce6bf792e17b8c0a42d34d4b4b72d108765d347b6ac34cae504fc601a7918e'
SOURCES = (
    'examples/weather_scenarios/muscl_experiment.py',
    'src/advar/transport.py',
    'examples/weather_scenarios/affine_holdout.py',
    'examples/weather_scenarios/compare_transport.py',
    'examples/weather_scenarios/finite_volume_probe.py',
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(report: dict, manifest: dict) -> dict:
    """Apply all candidate gates; signed widths use absolute errors."""
    failures, missing = [], []
    thresholds = manifest['thresholds']

    def bounded(label, value, lower, upper):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lower <= value <= upper):
            failures.append(label)

    if report['status'] != 'complete':
        failures.append('simulation not complete')
    if report['candidate'] != manifest['candidate'] or report['candidate_grid'] != manifest['grid_size']:
        failures.append('candidate/grid mismatch')
    if any(report[key] is not True for key in ('known_complete_field', 'known_zero_exterior', 'no_inflow')):
        failures.append('unsupported field/boundary scope')
    cases = [case['name'] for case in manifest['cases']]
    if any(report['geometry_checks'][name]['analytic_support_inside_domain'] is not True for name in cases):
        failures.append('analytic support leaves domain')
    results = [r for r in report['results'] if r['scheme'] == manifest['candidate']]
    if sorted(r['case'] for r in results) != sorted(cases):
        failures.append('missing, duplicate or unexpected candidate cases')
    coverage = {}
    for result in results:
        name = result['case']
        if result['size'] != manifest['grid_size']:
            failures.append(f'{name}: grid mismatch')
        bounded(f'{name}: CFL', result['actual_max_cfl'], 0, manifest['max_courant'])
        bounded(f'{name}: budget', result['max_relative_transformed_budget_residual'],
                0, thresholds['relative_transformed_budget_residual'])
        bounded(f'{name}: oracle', result['final_oracle_relative_l2_8_vs_16'],
                0, thresholds['oracle_relative_l2_8_vs_16'])
        expected = [(i + 1) * manifest['lead_interval_seconds'] / 60 for i in range(manifest['lead_count'])]
        if [lead['lead_minutes'] for lead in result['leads']] != expected:
            failures.append(f'{name}: missing, duplicate or unordered leads')
        covered = 0
        for lead in result['leads']:
            label = f"{name}/{lead['lead_minutes']}"
            for key in ('relative_echo_l2', 'center_error_pixels'):
                bounded(f'{label}: {key}', lead[key], 0, thresholds[key])
            widths = lead['axis_width_relative_error']
            if not isinstance(widths, list) or len(widths) != 2:
                failures.append(f'{label}: width axes')
            else:
                for width in widths:
                    bounded(f'{label}: width', width, -thresholds['axis_width_relative_error'],
                            thresholds['axis_width_relative_error'])
            bounded(f'{label}: minimum echo', lead['minimum_echo'], thresholds['minimum_echo'], math.inf)
            for key in ('transformed_budget_residual', 'lead_transformed_budget_residual'):
                bounded(f'{label}: {key}', lead[key], 0, thresholds['relative_transformed_budget_residual'])
            bounded(f'{label}: outflow', lead['transformed_outflow'], 0, math.inf)
            anisotropy = lead['truth_anisotropy']
            bounded(f'{label}: anisotropy', anisotropy, 0, 1)
            if lead['orientation_covered'] is not True:
                missing.append(f'{label}: orientation unavailable')
            else:
                bounded(f'{label}: orientation support', anisotropy,
                        thresholds['truth_anisotropy_minimum'], 1)
                bounded(f'{label}: angle', lead['orientation_error_degrees'],
                        0, thresholds['orientation_error_degrees'])
                covered += 1
        coverage[name] = covered
    return {'status': 'failed' if failures else 'incomplete' if missing else 'passed',
            'failures': failures, 'unevaluated': missing, 'orientation_coverage': coverage}


def evaluate_files(report_path: Path, manifest_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != CP3B_MANIFEST_SHA256 or report['manifest_sha256'] != CP3B_MANIFEST_SHA256:
        raise ValueError('manifest differs from frozen CP3b conditions')
    # Execution hashes establish reproducibility, not retrospective preregistration.
    for relative in SOURCES:
        path = ROOT / relative
        if report['source_sha256'].get(str(path)) != digest(path):
            raise ValueError(f'execution source differs: {relative}')
    for relative, expected in manifest['candidate_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'frozen candidate differs: {relative}')
    verdict = evaluate(report, manifest)
    verdict.update(report_sha256=digest(report_path), manifest_sha256=digest(manifest_path),
                   evaluator_sha256=digest(Path(__file__)),
                   scope='Saved candidate gates only; no inverse-flow, general-boundary or learning claim')
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        verdict = evaluate_files(args.report, args.manifest)
    except (KeyError, TypeError, ValueError, OSError) as error:
        verdict = {'status': 'failed', 'failures': [str(error)]}
    args.output.write_text(json.dumps(verdict, indent=2, allow_nan=False) + '\n')
    print(verdict['status'])
    raise SystemExit(0 if verdict['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
