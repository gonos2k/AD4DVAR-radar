from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle


def _plan() -> dict[str, Any]:
    parent = json.loads(cycle.PARENT_PLAN.read_text())
    sources = {**parent['source_files'], cycle.TRANSPORT: cycle.NEW_TRANSPORT_SHA}
    sources.update({name: cycle._sha(cycle.ROOT / name) for name in cycle.NEW_SOURCES})
    archives = {**parent['archive_files'], **{p.relative_to(cycle.ROOT).as_posix():
        cycle._sha(p) for p in cycle.EXTRA_ARCHIVES}}
    return {'experiment_kind': 'netflux_same_control_requalification_and_one_gradient_cycle',
        'base_control_sha256': cycle.BASE_SHA, 'policy': cycle.POLICY,
        'numerical_change': cycle.NUMERICAL_CHANGE,
        'source_files': sources, 'archive_files': archives}


@pytest.mark.parametrize('damage', ['none', 'other_source', 'legacy_transport', 'asset'])
def test_plan_only_allows_declared_transport_transition(monkeypatch, tmp_path, damage):
    plan = _plan()
    if damage == 'other_source':
        plan['source_files']['src/advar/physics.py'] = '0' * 64
    elif damage == 'legacy_transport':
        plan['source_files'][cycle.TRANSPORT] = cycle.OLD_TRANSPORT_SHA
    elif damage == 'asset':
        plan['archive_files'][cycle.CASE.relative_to(cycle.ROOT).as_posix()] = '0' * 64
    path = tmp_path / 'PLAN.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(cycle, 'PLAN', path)
    monkeypatch.setattr(cycle, '_sha', lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
    digest = cycle._sha(path)
    if damage == 'none':
        assert cycle.load_plan(path, digest) == plan
    else:
        with pytest.raises(ValueError, match='transition|scope'):
            cycle.load_plan(path, digest)


def test_asset_hash_is_checked_before_unpickling(monkeypatch):
    monkeypatch.setattr(cycle, '_sha', lambda _p: 'bad')
    def forbidden(*_args, **_kwargs):
        raise AssertionError('changed pickle must not be deserialized')
    monkeypatch.setattr(cycle.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='asset changed'):
        cycle.load_fixed_case()


def test_legacy_plan_refuses_the_changed_live_evaluator():
    with pytest.raises(ValueError, match="source/archive changed"):
        cycle.old.load_plan(cycle.PARENT_PLAN, cycle.PARENT_SHA)


def test_new_backend_uses_old_fixed_asset_without_regenerating_truth(monkeypatch, tmp_path):
    base = cycle.endpoint.load_endpoint()
    def forbidden(*_args, **_kwargs):
        raise AssertionError('new-kernel seed would change observations and truth')
    monkeypatch.setattr(cycle.old.previous.seed, '_prepare_fixed_seed', forbidden)
    monkeypatch.setattr(cycle.shared, '_source_hashes', lambda *_a: {cycle.TRANSPORT: cycle.NEW_TRANSPORT_SHA})
    monkeypatch.setattr(cycle.CurrentBackend, '_check', lambda *_a: None)
    backend = cycle.CurrentBackend({}, tmp_path / 'PLAN.json', 'plan', tmp_path, 100., base)
    assert backend.input_anchor == base['input_after']
    assert cycle.tangent._tensor_sha(backend.parameters) == base['input_after']['parameters_sha256']
    assert cycle.tangent._tensor_sha(backend.truth) == base['input_after']['terminal_truth_sha256']


def test_failed_new_qualification_preserves_legacy_confirmed_point(monkeypatch, tmp_path):
    base = cycle.endpoint.load_endpoint()
    class Backend:
        def __init__(self, *_a):
            self.control = base['control']
            self.problem = self.original = self.parameters = self.truth = object()
            self.source_anchor = {cycle.TRANSPORT: cycle.NEW_TRANSPORT_SHA}
            self.input_anchor, self.runtime_anchor = base['input_after'], base['runtime_after']
            self.timings = {}
    def fail(*_a):
        raise ValueError('independent new P2 mismatch')
    monkeypatch.setattr(cycle, 'load_plan', lambda *_a: {})
    monkeypatch.setattr(cycle, 'CurrentBackend', Backend)
    monkeypatch.setattr(cycle.requalification, 'requalify_base', fail)
    monkeypatch.setattr(cycle.shared, '_input_identity', lambda *_a: base['input_after'])
    monkeypatch.setattr(cycle.shared, '_runtime', lambda: base['runtime_after'])
    monkeypatch.setattr(cycle.shared, '_source_hashes', lambda *_a: {cycle.TRANSPORT: cycle.NEW_TRANSPORT_SHA})
    output = tmp_path / 'step.json'
    result = cycle._run_child(tmp_path / 'PLAN.json', 'plan', output)
    assert result['execution_status'] == 'failed'
    assert result['failure_stage'] == 'new_evaluator_base_requalification'
    assert result['current_control'] == result['last_confirmed_control'] == base['control'].tolist()
    assert result['last_confirmed_full_G'] == base['full_G'].tolist()
    assert result['current_full_G'] is None and not result['base_requalified']
    assert result['last_confirmed_numerical_version'] == 'legacy_separate_flux_fp64'
    assert result['hvp_started'] == result['candidate_evaluations_started'] == 0
    assert result['optimizer_steps_applied'] == 0 and not result['candidate_committed']
    assert json.loads(output.read_text()) == result


def test_child_failure_sets_nonzero_process_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(cycle.sys, 'argv', ['cycle', '--child', '--plan-sha256', 'digest'])
    monkeypatch.setattr(cycle, '_run_child', lambda *_a: {'execution_status': 'failed'})
    with pytest.raises(SystemExit) as error:
        cycle.main()
    assert error.value.code == 1
