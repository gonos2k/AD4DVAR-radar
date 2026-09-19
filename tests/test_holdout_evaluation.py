"""Candidate acceptance must reject small, targeted corruptions of saved evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/weather_scenarios'))
from evaluate_holdout import evaluate, evaluate_files

CHECKPOINT = ROOT / 'graphify-out/scenario-implementation-20260909/checkpoints'


@pytest.fixture
def inputs():
    return (json.loads((CHECKPOINT / 'CP3b_holdout.json').read_text()),
            json.loads((CHECKPOINT / 'CP3b_manifest.json').read_text()))


def candidate(report):
    return next(r for r in report['results'] if r['scheme'] == 'minmod')


def test_saved_candidate_passes_despite_failing_comparison_baseline(inputs):
    report, manifest = inputs
    assert any(lead['relative_echo_l2'] > .1 for r in report['results']
               if r['scheme'] == 'donorcell' for lead in r['leads'])
    assert evaluate(report, manifest)['status'] == 'passed'


@pytest.mark.parametrize('key,value', [
    ('axis_width_relative_error', [-.2, -.1]),
    ('axis_width_relative_error', [.1, 0]),
    ('relative_echo_l2', float('nan')),
    ('center_error_pixels', float('inf')),
    ('minimum_echo', -1e-20),
    ('lead_transformed_budget_residual', 1e-6),
    ('orientation_error_degrees', 3),
])
def test_candidate_corruption_fails(inputs, key, value):
    report, manifest = inputs
    candidate(report)['leads'][0][key] = value
    assert evaluate(report, manifest)['status'] == 'failed'


def test_unavailable_orientation_is_incomplete_not_passed(inputs):
    report, manifest = inputs
    lead = candidate(report)['leads'][0]
    lead.update(orientation_covered=False, orientation_error_degrees=None, truth_anisotropy=.01)
    assert evaluate(report, manifest)['status'] == 'incomplete'


@pytest.mark.parametrize('mutation', ['missing_case', 'duplicate_case', 'missing_lead', 'duplicate_lead'])
def test_incomplete_evidence_cannot_pass(inputs, mutation):
    report, manifest = inputs
    result = candidate(report)
    if mutation == 'missing_case':
        report['results'].remove(result)
    elif mutation == 'duplicate_case':
        report['results'].append(copy.deepcopy(result))
    elif mutation == 'missing_lead':
        result['leads'].pop()
    else:
        result['leads'][1] = copy.deepcopy(result['leads'][0])
    assert evaluate(report, manifest)['status'] == 'failed'


def test_manifest_drift_rejected(tmp_path, inputs):
    _, manifest = inputs
    manifest['thresholds']['relative_echo_l2'] = 1
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest differs'):
        evaluate_files(CHECKPOINT / 'CP3b_holdout.json', path)


def test_execution_helper_drift_rejected(tmp_path, inputs):
    report, _ = inputs
    report['source_sha256'][str(ROOT / 'examples/weather_scenarios/compare_transport.py')] = 'bad'
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='execution source differs'):
        evaluate_files(path, CHECKPOINT / 'CP3b_manifest.json')


def test_unknown_exterior_cannot_inherit_known_zero_acceptance(inputs):
    report, manifest = inputs
    report['known_zero_exterior'] = False
    assert evaluate(report, manifest)['status'] == 'failed'


def test_joint_report_and_manifest_relaxation_rejected(tmp_path, inputs):
    import hashlib
    report, manifest = inputs
    manifest['thresholds']['relative_echo_l2'] = 1
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    report['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='frozen CP3b'):
        evaluate_files(report_path, manifest_path)
