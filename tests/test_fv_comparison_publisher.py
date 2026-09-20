"""Reject mislabeled or incomparable archived transport tables before publication."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / 'graphify-out/fv-root-cause-20260919'
SPEC = spec_from_file_location('fv_comparison_publisher', EVIDENCE / 'publish_fv_response_evidence.py')
assert SPEC is not None and SPEC.loader is not None
PUBLISHER = module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


@pytest.fixture
def reports():
    return {scheme: json.loads((EVIDENCE / f'fv_comparison_{scheme}_180min.json').read_text())
            for scheme in ('donorcell', 'minmod')}


def test_archived_comparison_validates_independent_of_row_order(reports):
    reports['minmod']['results'].reverse()
    PUBLISHER._validate_transport_comparison(reports)


def test_donor_report_cannot_be_published_under_minmod_filename(reports):
    reports['minmod'] = deepcopy(reports['donorcell'])
    with pytest.raises(SystemExit, match='scheme/domain/time/boundary'):
        PUBLISHER._validate_transport_comparison(reports)


@pytest.mark.parametrize('key,value', [
    ('domain_side_m', 24000), ('interval_seconds', 300), ('leads', 17),
    ('sizes', [32, 64]), ('boundary_condition', 'periodic'),
])
def test_incompatible_global_conditions_are_rejected(reports, key, value):
    reports['minmod'][key] = value
    with pytest.raises(SystemExit, match='mismatch'):
        PUBLISHER._validate_transport_comparison(reports)


@pytest.mark.parametrize('key,value', [
    ('reconstruction', 'donorcell'), ('spacing_m', 1.0),
    ('echo_area_threshold', 0.2), ('substeps_per_lead', 0),
])
def test_incompatible_row_conditions_are_rejected(reports, key, value):
    reports['minmod']['results'][0][key] = value
    with pytest.raises(SystemExit, match='mismatch|CFL'):
        PUBLISHER._validate_transport_comparison(reports)


def test_duplicate_case_grid_cannot_hide_missing_case(reports):
    reports['minmod']['results'][1] = deepcopy(reports['minmod']['results'][0])
    with pytest.raises(SystemExit, match='missing or duplicate'):
        PUBLISHER._validate_transport_comparison(reports)


def test_changed_evaluation_time_is_rejected(reports):
    reports['minmod']['results'][0]['leads'][-1]['lead_minutes'] = 170
    with pytest.raises(SystemExit, match='row scheme/grid/time/threshold'):
        PUBLISHER._validate_transport_comparison(reports)


def test_shared_wrong_source_hash_is_not_sufficient(reports):
    for data in reports.values():
        key = next(iter(data['source_sha256']))
        data['source_sha256'][key] = '0' * 64
    with pytest.raises(SystemExit, match='measured revision'):
        PUBLISHER._validate_transport_comparison(reports)


def test_nonfinite_display_metric_is_rejected(reports):
    reports['minmod']['results'][-1]['derivative']['jvp_relative_to_independent_oracle'] = float('nan')
    with pytest.raises(SystemExit, match='displayed metrics'):
        PUBLISHER._validate_transport_comparison(reports)


def test_panel_cannot_bypass_scheme_validation(reports, tmp_path, monkeypatch):
    reports['minmod'] = deepcopy(reports['donorcell'])
    for scheme, data in reports.items():
        (tmp_path / f'fv_comparison_{scheme}_180min.json').write_text(json.dumps(data))
    monkeypatch.setattr(PUBLISHER, 'HERE', tmp_path)
    monkeypatch.setattr(PUBLISHER, 'LONG_HORIZON_PATH', tmp_path / 'absent.json')
    response = json.loads((EVIDENCE / 'rotation240_stable_response_18.json').read_text())
    with pytest.raises(SystemExit, match='scheme/domain/time/boundary'):
        PUBLISHER._panel(response, 1.0, None)


@pytest.mark.parametrize('corrupt', [False, True])
def test_joint_inverse_panel_preserves_local_scope(tmp_path, monkeypatch, corrupt):
    joint = json.loads((EVIDENCE / 'minmod_joint_inverse_final.json').read_text())
    if corrupt:
        joint['general_minmod_response_eligible'] = True
    (tmp_path / 'minmod_joint_inverse_final.json').write_text(json.dumps(joint))
    monkeypatch.setattr(PUBLISHER, 'HERE', tmp_path)
    monkeypatch.setattr(PUBLISHER, 'LONG_HORIZON_PATH', tmp_path / 'absent.json')
    response = json.loads((EVIDENCE / 'rotation240_stable_response_18.json').read_text())
    if corrupt:
        with pytest.raises(SystemExit, match='joint minmod report scope'):
            PUBLISHER._panel(response, 1.0, None)
    else:
        panel = PUBLISHER._panel(response, 1.0, None)
        assert 'fvMinmodJointInverse' in panel
        assert '일반 minmod FSOI' in panel
        assert 'background_parameter' in panel


@pytest.mark.parametrize('field,value,reason', [
    ('gradient_max', float('nan'), 'gradient_max'),
    ('gradient_max', 1e-6, 'gradient_max'),
    ('adjoint_relative_residual', float('inf'), 'adjoint_relative_residual'),
    ('adjoint_relative_residual', -1e-12, 'adjoint_relative_residual'),
    ('hessian_min_eigenvalue', -1., 'Hessian'),
    ('source_sha256', {}, 'identity'),
])
def test_joint_inverse_rejects_invalid_evidence(field, value, reason):
    joint = json.loads((EVIDENCE / 'minmod_joint_inverse_final.json').read_text())
    joint[field] = value
    with pytest.raises(SystemExit, match=reason):
        PUBLISHER._validate_joint_inverse(joint)


@pytest.mark.parametrize('mutation,reason', [
    ('zero_margin', 'branch margin'), ('negative_error', 'reanalysis numbers'),
    ('wrong_error', 'absolute_error'), ('source_hash', 'identity'),
])
def test_joint_inverse_panel_rejects_mutated_numeric_or_source_evidence(
    tmp_path, monkeypatch, mutation, reason,
):
    joint = json.loads((EVIDENCE / 'minmod_joint_inverse_final.json').read_text())
    if mutation == 'zero_margin':
        joint['stationary_branch']['minimum_scaled_slope_margin'] = 0.
    elif mutation == 'negative_error':
        joint['reanalysis'][0]['absolute_error'] = -1.
    elif mutation == 'wrong_error':
        joint['reanalysis'][0]['absolute_error'] = 1.
    else:
        joint['source_sha256'][next(iter(joint['source_sha256']))] = '0'*64
    (tmp_path / 'minmod_joint_inverse_final.json').write_text(json.dumps(joint))
    monkeypatch.setattr(PUBLISHER, 'HERE', tmp_path)
    monkeypatch.setattr(PUBLISHER, 'LONG_HORIZON_PATH', tmp_path / 'absent.json')
    response = json.loads((EVIDENCE / 'rotation240_stable_response_18.json').read_text())
    with pytest.raises(SystemExit, match=reason):
        PUBLISHER._panel(response, 1., None)


def test_local_path_panel_uses_completed_archived_pairs():
    panel = PUBLISHER._local_path_panel()
    assert 'fvMinmodLocalPath' in panel
    assert 'h=0.001' in panel
    assert '일반 minmod FSOI' in panel


@pytest.mark.parametrize('mutation', ['gradient', 'finite_path', 'derivative', 'source'])
def test_local_path_panel_rejects_changed_evidence(tmp_path, monkeypatch, mutation):
    data = json.loads((PUBLISHER.HERE / 'minmod_local_path.json').read_text())
    if mutation == 'gradient':
        data['pairs'][-1]['endpoints'][0]['gradient_max'] = float('nan')
    elif mutation == 'finite_path':
        data['finite_path_certified'] = True
    elif mutation == 'derivative':
        data['pairs'][-1]['derivative_pass'] = False
    else:
        data['producer_sha256'] = '0' * 64
    (tmp_path / 'minmod_local_path.json').write_text(json.dumps(data))
    monkeypatch.setattr(PUBLISHER, 'HERE', tmp_path)
    with pytest.raises(SystemExit, match='local path'):
        PUBLISHER._local_path_panel()


def test_additional_directions_panel_uses_measured_results():
    panel = PUBLISHER._additional_directions_panel()
    assert 'fvMinmodAdditionalDirections' in panel
    assert '중간 시각 공통 편향' in panel and '배경 θ' in panel


@pytest.mark.parametrize('mutation', ['direct', 'direction', 'step', 'score', 'branch', 'gradient', 'error'])
def test_additional_direction_evidence_rejects_mutation(mutation):
    import hashlib
    data = json.loads((EVIDENCE / 'minmod_theta_final.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pair = data['pairs'][-1]
    if mutation == 'direct':
        data['adjoint']['direct'][0] += .001
    elif mutation == 'direction':
        data['selected_direction'] = 'middle_time_bias'
    elif mutation == 'step':
        pair['h'] *= .9
    elif mutation == 'score':
        pair['endpoints'][0]['score'] += .001
    elif mutation == 'branch':
        pair['endpoints'][0]['branch']['face_signs'] = []
    elif mutation == 'gradient':
        pair['endpoints'][0]['gradient_max'] = float('nan')
    else:
        pair['absolute_error'] = -1
    with pytest.raises(SystemExit, match='direction result'):
        PUBLISHER._validate_direction_result(data, 'theta', fingerprint)


def test_matrix_free_panel_shows_separate_terms_and_scope():
    panel = PUBLISHER._matrix_free_panel()
    assert 'fvMinmodMatrixFree' in panel and '직접항' in panel and '간접항' in panel
    assert '사후 대조' in panel and '일반 minmod FSOI' in panel


@pytest.mark.parametrize('mutation', ['residual', 'total', 'source', 'scope', 'curvature', 'mixed'])
def test_matrix_free_panel_rejects_damaged_evidence(tmp_path, monkeypatch, mutation):
    data = json.loads((EVIDENCE/'minmod_matrix_free.json').read_text())
    if mutation == 'residual':
        data['actual_transpose_residual'] = 1e-3
    elif mutation == 'total':
        data['directions']['theta']['total'] += .001
    elif mutation == 'curvature':
        data['dense_min_eigenvalue'] = -1.
    elif mutation == 'mixed':
        data['directions']['theta']['mixed_gradient_max_difference'] = .1
    elif mutation == 'source':
        data['source_sha256'] = {}
    else:
        data['general_minmod_response_eligible'] = True
    (tmp_path/'minmod_matrix_free.json').write_text(json.dumps(data))
    monkeypatch.setattr(PUBLISHER,'HERE',tmp_path)
    with pytest.raises(SystemExit, match='matrix-free'):
        PUBLISHER._matrix_free_panel()
