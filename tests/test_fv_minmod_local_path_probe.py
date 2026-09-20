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


def test_cached_score_derivatives_require_scaled_finite_agreement():
    fresh = torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64)
    assert PROBE._check_cached_score_derivative("rhs", fresh.clone(), fresh) == 0.0
    changed = fresh.clone()
    changed[0] += 1.0e-4
    with pytest.raises(ValueError, match="rhs"):
        PROBE._check_cached_score_derivative("rhs", changed, fresh)
    with pytest.raises(ValueError, match="shape/dtype"):
        PROBE._check_cached_score_derivative("rhs", fresh.to(torch.float32), fresh)
    with pytest.raises(ValueError, match="nonfinite"):
        PROBE._check_cached_score_derivative("rhs", torch.full_like(fresh, float("nan")), fresh)


def test_cached_parameter_direction_is_literal_and_typed():
    direction = torch.tensor([0.0, 1.0, -0.5], dtype=torch.float64)
    cached = {"parameter_direction": direction.tolist()}
    PROBE._check_cached_direction("test", cached, direction)
    changed = {"parameter_direction": [0.0, 1.0, 0.5]}
    with pytest.raises(ValueError, match="direction mismatch"):
        PROBE._check_cached_direction("test", changed, direction)
    with pytest.raises(ValueError, match="shape/dtype"):
        PROBE._check_cached_direction("test", {"parameter_direction": [0.0]}, direction)


def test_cache_payload_binds_numerical_contract_and_rejects_mutation():
    expected = {
        "contract": "fv-minmod-local-path-v1",
        "core_source_sha256": {"transport": "a"},
        "fixture_numerical_ast_sha256": "b",
        "shape": [4, 5],
        "controls": 26,
        "substeps_per_interval": 9,
        "parameters_sha256": "p",
        "control_sha256": "c",
        "direction_names": ["observation_sine60", "theta"],
        "saved_report_sha256": "s",
    }
    cache = {"cache_contract": expected["contract"], "cache_payload_identity": dict(expected)}
    cache.update(hessian={"matrix": [[1.]]}, tangents={}, adjoint={"rhs": [1.], "direct": [2.], "solution": [1.]})
    cache["linearization_sha256"] = PROBE._linearization_digest(cache)
    assert PROBE._check_cache_payload(cache, expected, "untrusted") == "versioned"
    cache["adjoint"]["direct"][0] += .001
    with pytest.raises(ValueError, match="payload digest"):
        PROBE._check_cache_payload(cache, expected, "untrusted")
    changed = {"cache_contract": expected["contract"], "cache_payload_identity": {**expected, "control_sha256": "changed"}}
    with pytest.raises(ValueError, match="payload identity"):
        PROBE._check_cache_payload(changed, expected, "untrusted")


def test_legacy_cache_requires_trusted_archive_and_producer():
    path = ROOT / "graphify-out/fv-root-cause-20260919/minmod_local_path_coarse.json"
    data = json.loads(path.read_text())
    expected = {"saved_report_sha256": data["saved_report_sha256"]}
    trusted = PROBE._hash(path)
    assert PROBE._check_cache_payload(data, expected, trusted) == "legacy-anchored"
    mutated = {**data, "producer_sha256": "0" * 64}
    with pytest.raises(ValueError, match="legacy cache"):
        PROBE._check_cache_payload(mutated, expected, trusted)


@pytest.mark.parametrize('mutation', ['direct', 'shape', 'tiny_component', 'verification', 'score'])
def test_direct_cache_rejects_changes_to_current_score(mutation):
    c = torch.tensor([.2, .3], dtype=torch.float64)
    p = torch.tensor([.4, .5], dtype=torch.float64)
    target = torch.tensor([.1, .2], dtype=torch.float64)
    def score(c, p):
        return ((c + p - target)**2).mean()
    rhs, direct = torch.func.grad(score, argnums=(0, 1))(c, p)
    cached = direct.clone()
    if mutation == 'direct':
        cached[0] += .001
    elif mutation == 'shape':
        cached = cached[:1]
    elif mutation == 'tiny_component':
        cached[0], direct[0] = 2e-20, 1e-20
    else:
        if mutation == 'verification':
            target = target + .01
        else:
            original = score
            score = lambda c, p: 2*original(c, p)
        rhs, direct = torch.func.grad(score, argnums=(0, 1))(c, p)
    with pytest.raises(ValueError, match='cached direct'):
        PROBE._check_cached_score_derivative('direct', cached, direct)


def test_calculation_identity_ignores_logging_but_detects_changed_equations(tmp_path):
    source = Path(PROBE.__file__).read_text()
    path = tmp_path / 'probe.py'
    expected = PROBE._calculation_hash(Path(PROBE.__file__))
    path.write_text(source.replace('report["status"] = "running"', 'report["status"] = "starting"'))
    assert PROBE._calculation_hash(path) == expected
    path.write_text(source.replace('0.1 * pattern', '0.2 * pattern'))
    assert PROBE._calculation_hash(path) != expected


def test_new_direction_clears_old_pairs_and_computes_its_cross(tmp_path, monkeypatch):
    # Stop before optimization: exercise actual preparation/cache/JVP wiring.
    class ReachedCorrector(BaseException):
        pass
    original_load = PROBE._load_module
    observed = []
    def load(path, name):
        module = original_load(path, name)
        if path == PROBE.ORACLE:
            def stop(objective, predictor, changed, **kwargs):
                observed.append(changed)
                raise ReachedCorrector
            module.polish = stop
        return module
    original_jvp = torch.func.jvp
    directions = []
    def jvp(func, primals, tangents, **kwargs):
        if primals[0].shape == (61,):
            directions.append(tangents[0].clone())
        return original_jvp(func, primals, tangents, **kwargs)
    monkeypatch.setattr(PROBE, '_load_module', load)
    monkeypatch.setattr(torch.func, 'jvp', jvp)
    output = tmp_path / 'new.json'
    with pytest.raises(ReachedCorrector):
        PROBE.run(output, resume=ROOT / 'graphify-out/fv-root-cause-20260919/minmod_local_path.json',
                  direction_name='middle_time_bias')
    data = json.loads(output.read_text())
    assert data['pairs'] == [] and data['resume_start_j'] == 0
    assert data['selected_direction'] == 'middle_time_bias'
    expected = torch.zeros(61, dtype=torch.float64)
    expected[20:40] = 1
    assert len(directions) == 1 and torch.equal(directions[0], expected)
    assert len(observed) == 1


def test_completed_direction_resume_validates_without_repeating_reanalysis(tmp_path, monkeypatch):
    original_load = PROBE._load_module
    def load(path, name):
        module = original_load(path, name)
        if path == PROBE.ORACLE:
            def unexpected(*args, **kwargs):
                pytest.fail('completed direction must not repeat reanalysis')
            module.polish = unexpected
        return module
    monkeypatch.setattr(PROBE, '_load_module', load)
    source = ROOT / 'graphify-out/fv-root-cause-20260919/minmod_theta.json'
    result = PROBE.run(tmp_path / 'validated.json', resume=source, direction_name='theta')
    assert result['status'] == 'complete'
    assert result['pairs'] == json.loads(source.read_text())['pairs']
    assert 'metadata_correction' not in result
    assert result['resume_provenance']['historical_metadata_correction']['old'] == 3
    assert result['cache_validation']['payload'] == 'versioned'
