"""Protect cached tangent/adjoint reuse with the measured 26-control system."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'local_path_probe', ROOT / 'examples/weather_scenarios/fv_minmod_local_path_probe.py')
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.fixture
def system():
    data = json.loads((ROOT / 'graphify-out/fv-root-cause-20260919/minmod_local_path_coarse.json').read_text())
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)
    return (tensor(data['hessian']['matrix']),
            {k: tensor(v['control_tangent']) for k, v in data['tangents'].items()},
            {k: tensor(v['cross_gradient']) for k, v in data['tangents'].items()},
            tensor(data['adjoint']['solution']), tensor(data['adjoint']['rhs']))


def test_measured_cached_linearization_is_resolved(system):
    assert max(PROBE._check_linearization(*system).values()) < 1e-12


@pytest.mark.parametrize('changed', ['tangent', 'adjoint', 'asymmetry', 'nan'])
def test_corrupted_cached_linearization_is_rejected(system, changed):
    H, tangents, crosses, adjoint, rhs = system
    if changed == 'tangent':
        tangents['observation_sine60'][0] += .01
    elif changed == 'adjoint':
        adjoint[0] += .01
    elif changed == 'asymmetry':
        H[0, 1] += 1.
    else:
        H[0, 0] = float('nan')
    with pytest.raises(ValueError, match='cached'):
        PROBE._check_linearization(H, tangents, crosses, adjoint, rhs)


def test_resume_counts_structural_and_derivative_pairs_separately():
    data = json.loads((ROOT / 'graphify-out/fv-root-cause-20260919/minmod_local_path_coarse.json').read_text())
    assert PROBE._consecutive_pairs(data['pairs'], require_derivative=False) == 2
    assert PROBE._consecutive_pairs(data['pairs'], require_derivative=True) == 0
