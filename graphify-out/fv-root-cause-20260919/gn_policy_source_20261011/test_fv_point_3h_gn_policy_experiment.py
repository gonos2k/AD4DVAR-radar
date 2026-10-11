from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_gn_policy_experiment as runner


def _closure() -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    c = torch.zeros(26, dtype=torch.float64)
    sides = {'-1': [0.5] + [0.] * 25, '1': [0.5] + [0.] * 25}
    proposal: dict[str, Any] = {'control': c.tolist(), 'theta': .5, 'objective': .8,
        'F_squared': .25, 'side_gradients': sides, 'candidate_mixing_minimum': True,
        'F_squared_armijo_passed': False, 'acceptance_policy': 'bounded_J_Rcap',
        'control_sha256': runner.prior.tangent._tensor_sha(c),
        'J_armijo_passed': True, 'J_armijo_bound': .9, 'F_squared_armijo_bound': .1,
        'R_reference_initial': .5, 'R_cap': .505}
    repeat = {**copy.deepcopy(proposal), **{key: True for key in (
        'side_gradients_finite', 'native_objective_matches_proposal', 'side_objectives_match_native',
        'merit_matches_proposal', 'face_audit_passed', 'branch_pair_passed',
        'trace_matches_proposal', 'source_unchanged', 'fixed_input_unchanged',
        'runtime_unchanged', 'deadline_passed', 'mixing_minimum_valid', 'minimum_theta_matches_proposal')}}
    return proposal, repeat, [.5] + [0.] * 26


def _record_and_arm() -> tuple[dict[str, Any], dict[str, Any]]:
    arm: dict[str, Any] = {'status': 'endpoint_recovered', 'endpoint_P2_passed': True,
        'endpoint_J': .8, 'endpoint_R': .25, 'arm_order': 0,
        'policy': 'bounded_J_Rcap', 'points': [{}, {}, {}], 'canonical_promoted': False}
    record = {'base_J': 1., 'base_R': .5, 'current_control_sha256': 'initial',
        'candidate_committed': False, 'arms': [arm]}
    return record, arm


def test_unrecovered_nonmonotone_path_is_never_promotable():
    record, arm = _record_and_arm()
    arm['status'] = 'recovery_horizon_exhausted_noncanonical'
    assert runner.choose_endpoint([arm]) is None
    proposal, repeat, _ = _closure()
    repeat['F_squared'] = .51
    assert not runner.endpoint_recovered(1., .5, proposal, repeat, relative_margin=1e-8)


def test_recovered_endpoint_keeps_truthful_R_armijo_flag(tmp_path):
    record, arm = _record_and_arm()
    proposal, repeat, G = _closure()
    committed = runner.persist_canonical(tmp_path / 'step.json', record, arm, proposal, repeat, G)
    assert committed['candidate_committed'] is True
    assert committed['canonical_proposal']['F_squared_armijo_passed'] is False
    assert committed['selected_policy'] == 'bounded_J_Rcap'
    assert record['candidate_committed'] is False and record['current_control_sha256'] == 'initial'
    assert json.loads((tmp_path / 'step.json').read_text()) == committed


def test_atomic_failure_does_not_mutate_canonical_prefix(tmp_path, monkeypatch):
    record, arm = _record_and_arm()
    before = copy.deepcopy(record)
    proposal, repeat, G = _closure()
    def fail(*args):
        raise OSError('write failed')
    monkeypatch.setattr(runner.prior.tangent, '_write', fail)
    with pytest.raises(OSError):
        runner.persist_canonical(tmp_path / 'step.json', record, arm, proposal, repeat, G)
    assert record == before and arm['canonical_promoted'] is False


def test_endpoint_selection_uses_actual_original_R_then_J():
    _, arm = _record_and_arm()
    another = {**arm, 'endpoint_R': .24, 'endpoint_J': .9, 'arm_order': 1}
    assert runner.choose_endpoint([arm, another]) is another
    another['endpoint_R'] = .25
    assert runner.choose_endpoint([arm, another]) is arm


def test_recorded_endpoint_receipt_is_hash_bound_before_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    proposal, repeat, G = _closure()
    path = tmp_path / 'closed.json'
    runner.prior.tangent._write(path, {'policy': 'bounded_J_Rcap', 'proposal': proposal,
        'final_repeat': repeat, 'G': G})
    digest = runner._sha(path)
    _, winner = _record_and_arm()
    winner.update(points=[{'closed_receipt': 'closed.json', 'closed_sha256': digest,
        'endpoint_J': .8, 'endpoint_R': .25}], endpoint_receipt='closed.json',
        endpoint_receipt_sha256=digest, endpoint_control_sha256=proposal['control_sha256'])
    assert runner.read_bound_endpoint(winner, .5)['proposal'] == proposal
    altered = json.loads(path.read_text())
    altered['proposal']['R_cap'] = 2.
    runner.prior.tangent._write(path, altered)
    with pytest.raises(ValueError, match='receipt changed'):
        runner.read_bound_endpoint(winner, .5)
    new_digest = runner._sha(path)
    winner['points'][0]['closed_sha256'] = winner['endpoint_receipt_sha256'] = new_digest
    with pytest.raises(ValueError, match='cap'):
        runner.read_bound_endpoint(winner, .5)


def test_invalid_full_G_cannot_be_promoted(tmp_path):
    record, arm = _record_and_arm()
    proposal, repeat, G = _closure()
    G[0] = 1.
    with pytest.raises(ValueError, match='merit'):
        runner.persist_canonical(tmp_path / 'step.json', record, arm, proposal, repeat, G)


@pytest.mark.parametrize(('failure', 'expected_status'), [
    (ValueError('operator failure'), 'failed'),
    (TimeoutError('internal budget'), 'completed'),
])
def test_runner_terminal_status_preserves_canonical_prefix(tmp_path, monkeypatch, failure, expected_status):
    # The real child/writer path runs against a backend double; no FV or seed call.
    parent = runner.prior._load_parent()
    class BackendDouble:
        def __init__(self, *args):
            self.parent = parent
            self.problem = self.original = self.parameters = self.truth = None
            self.source_anchor = self.input_anchor = self.runtime_anchor = {}
            self.build_count = self.row_count = self.solve_count = self.hvp_count = 0
            self.timings = {}
        def build(self, *args):
            raise failure
    monkeypatch.setattr(runner, 'FVBackend', BackendDouble)
    monkeypatch.setattr(runner, '_load_plan', lambda *args: {'previous_actual_movement': 1e-7})
    monkeypatch.setattr(runner.prior.shared, '_source_hashes', lambda *args: {})
    monkeypatch.setattr(runner.prior.shared, '_input_identity', lambda *args: {})
    monkeypatch.setattr(runner.prior.shared, '_runtime', lambda: {})
    monkeypatch.setattr(runner.prior.geometry.model, '_fixed_input', lambda *args: True)
    output = tmp_path / 'step.json'
    if isinstance(failure, TimeoutError):
        runner._run_child(tmp_path / 'plan.json', 'test', output)
    else:
        with pytest.raises(ValueError, match='operator failure'):
            runner._run_child(tmp_path / 'plan.json', 'test', output)
    saved = json.loads(output.read_text())
    assert saved['execution_status'] == expected_status
    assert saved['candidate_committed'] is False and saved['optimizer_steps_applied'] == 0
    assert saved['current_control'] == parent['current_control']
    assert saved['current_full_G'] == parent['current_full_G']
