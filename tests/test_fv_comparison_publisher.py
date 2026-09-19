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
