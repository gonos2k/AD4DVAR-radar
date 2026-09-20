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
