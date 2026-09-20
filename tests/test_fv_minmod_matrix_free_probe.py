"""The minmod adapter rejects inputs outside its declared research contract."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mf_probe', ROOT/'examples/weather_scenarios/fv_minmod_matrix_free_probe.py')
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.fixture(scope='module')
def case():
    obs, frozen, boundary, support = PROBE._load('fv_minmod_inverse_probe').make_spatial_case()
    saved = json.loads((PROBE.EVIDENCE/'minmod_middle_time_bias_final.json').read_text())
    return obs, frozen, boundary, support, saved


@pytest.mark.parametrize('mutation', ['donorcell', 'missing', 'boundary', 'pattern'])
def test_research_adapter_rejects_unsupported_inputs(case, mutation):
    obs, frozen, boundary, support, saved = case
    pattern = torch.ones((4,5), dtype=torch.float64)
    if mutation == 'donorcell':
        frozen = replace(frozen, fv_transport=replace(frozen.fv_transport, reconstruction='donorcell'))
    elif mutation == 'missing':
        obs = replace(obs, valid_mask=torch.zeros_like(obs.valid_mask))
    elif mutation == 'boundary':
        support = tuple(tuple(tuple(torch.zeros_like(e) for e in edges) for edges in stage) for stage in support)
    else:
        pattern = pattern.float()
    with pytest.raises(ValueError, match='research minmod'):
        PROBE.make_research_functions(obs,frozen,boundary,support,pattern,torch.ones((4,5),dtype=torch.float64),saved['nominal_branch'])


def test_branch_guard_rejects_changed_signature_and_transform_floor(case):
    obs, frozen, boundary, support, saved = case
    pattern = torch.linspace(-.2,.3,20,dtype=torch.float64).reshape(4,5)
    p = torch.cat((obs.dbz.flatten(),torch.tensor([.02],dtype=torch.float64)))
    c = torch.tensor(saved['nominal_control'],dtype=torch.float64)
    bad = {**saved['nominal_branch'],'choices': []}
    _, _, check = PROBE.make_research_functions(obs,frozen,boundary,support,pattern,pattern,bad)
    with pytest.raises(ValueError,match='branch identity'):
        check(c,p)
    p[:20] = frozen.nowcast_config.min_dbz
    with pytest.raises(ValueError,match='smooth branch'):
        check(c,p)


def test_archived_sources_relocate_without_basename_collision(tmp_path):
    import hashlib
    old = tmp_path/'absent-checkout'
    current = tmp_path/'new-checkout'
    fingerprints = {}
    for directory, content in [('src', b'core'), ('examples', b'example')]:
        relative = Path(directory)/'same.py'
        target = current/relative
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        fingerprints[str(old/relative)] = hashlib.sha256(content).hexdigest()
    assert not old.exists()
    PROBE.check_archived_sources(fingerprints, archived_root=old, root=current)
    (current/'src/same.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity mismatch'):
        PROBE.check_archived_sources(fingerprints, archived_root=old, root=current)


def test_archived_sources_reject_outside_root(tmp_path):
    for path in [tmp_path/'elsewhere/a.py', tmp_path/'old/../a.py']:
        with pytest.raises(ValueError):
            PROBE.check_archived_sources({str(path): 'unused'}, archived_root=tmp_path/'old', root=tmp_path/'new')


def test_actual_archived_gn_result_is_rejected_before_adjoint(case, monkeypatch):
    import advar.local_response as local
    obs, frozen, boundary, support, saved = case
    gn = json.loads((PROBE.EVIDENCE/'minmod_spatial_inverse.json').read_text())['product_gn']
    c = torch.tensor(gn['control'], dtype=torch.float64)
    p = torch.cat((obs.dbz.flatten(), obs.dbz.new_tensor([.02])))
    pattern = torch.linspace(-.2,.3,20,dtype=torch.float64).reshape(4,5)
    objective, score, check = PROBE.make_research_functions(obs,frozen,boundary,support,pattern,pattern,saved['nominal_branch'])
    def no_solve(*args, **kwargs):
        pytest.fail('unqualified GN result reached adjoint solver')
    monkeypatch.setattr(local, 'pcg', no_solve)
    with pytest.raises(ValueError, match='not stationary'):
        local.compute_local_response(objective,score,c,p,{'theta':torch.eye(61,dtype=p.dtype)[-1]},branch_check=check,input_identity={'source':'archived product GN'})
