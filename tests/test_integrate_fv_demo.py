"""The original UI must preserve history and compare matching FV/P0 leads."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/weather_scenarios'))
from integrate_fv_demo import integrate


def test_integration_preserves_history_and_replaces_only_its_entries():
    legacy = {'id': 'legacy', 'metrics': [{'mae': 7.0}], 'observations': [[None]]}
    base = {'meta': {'grid_shape': [48, 48]}, 'scenarios': [legacy]}
    raw = {'cases': [{'id': 'pde_translation', 'name': '병진', 'meta': {'grid_shape': [12, 12]},
                     'methods': {name: {'metrics': [{'domain_pixels': 144, 'missing_pixels': missing, 'mae': score}]}
                                 for name, missing, score in [('p0', 12, None), ('fv', 0, 0.2)]}}]}
    html = lambda value: '<script id="demo-data" type="application/json">' + json.dumps(value) + '</script>'
    first = integrate(html(base), raw)
    assert first['scenarios'][0] == legacy
    assert len(first['scenarios']) == 3
    assert first['scenarios'][2]['meta']['grid_shape'] == [12, 12]
    assert first['scenarios'][2]['comparison'] == [dict(domain_pixels=144, p0_mae=None, fv_mae=0.2, p0_missing=12, fv_missing=0)]
    assert integrate(html(first), raw) == first


def test_mismatched_method_leads_are_not_silently_truncated():
    base = '<script id="demo-data" type="application/json">{"scenarios":[]}</script>'
    metric = dict(domain_pixels=1, missing_pixels=0, mae=0.)
    with pytest.raises(ValueError):
        integrate(base, {'cases': [{'methods': {'p0': {'metrics': [metric]}, 'fv': {'metrics': []}}}]})


@pytest.mark.parametrize('change', [{'domain_pixels': 2}, {'lead_minutes': 20}])
def test_mismatched_comparison_contract_is_rejected(change):
    base = '<script id="demo-data" type="application/json">{"scenarios":[]}</script>'
    metric = dict(domain_pixels=1, missing_pixels=0, mae=0., lead_minutes=10)
    with pytest.raises(ValueError, match='same domain and lead'):
        integrate(base, {'cases': [{'methods': {'p0': {'metrics': [metric]}, 'fv': {'metrics': [{**metric, **change}]}}}]})


def test_motion_diagnostics_use_each_methods_actual_saved_field():
    import math

    # A noncentral, anisotropic pattern is rotated exactly by 90 degrees.
    q = [[100. for _ in range(5)] for _ in range(5)]
    q[1][3], q[2][3], q[3][3] = 200., 500., 200.
    dbz = lambda grid: [[10 * math.log10(v + .1) for v in row] for row in grid]
    reference = dbz(q)
    rotated = dbz([list(row) for row in zip(*q[::-1])])
    missing = [[None] * 5 for _ in range(5)]
    metric = dict(domain_pixels=25, missing_pixels=0, mae=0., lead_minutes=10)
    raw = {'meta': {'spacing_m': 1., 'background_echo': 100., 'min_dbz': -10.},
           'cases': [{'id': 'rotation', 'name': 'rotation', 'observations': [reference], 'truth': [rotated],
                      'methods': {'p0': {'forecast': [reference], 'metrics': [metric]},
                                  'fv': {'forecast': [missing], 'metrics': [{**metric, 'mae': None, 'missing_pixels': 25}]}}}]}
    payload = integrate('<script id="demo-data" type="application/json">{"scenarios":[]}</script>', raw)
    p0, fv = payload['scenarios']
    assert p0['field_motion'][0]['truth']['polar_change_deg'] == pytest.approx(90)
    assert p0['field_motion'][0]['forecast']['polar_change_deg'] == pytest.approx(0)
    assert fv['field_motion'][0]['forecast']['polar_change_deg'] is None
