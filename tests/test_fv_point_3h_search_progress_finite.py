from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_gn_committed_continuation as old_runner
from examples.weather_scenarios import fv_point_3h_gn_finite_continuation as runner
from examples.weather_scenarios import fv_point_3h_search_progress as frozen
from examples.weather_scenarios import fv_point_3h_search_progress_finite as corrected


def _model():
    return SimpleNamespace(direction=torch.tensor([1., 0.], dtype=torch.float64), theta=.5,
        gates={'finite': True, 'both_side_gradients_descend': True, 'merit_descends': True},
        merit_product=-.01, residual_direction=torch.tensor([1., 0.], dtype=torch.float64))


def _facts(accepted: bool = True):
    keys = ('mixing_minimum_valid', 'J_armijo_passed', 'branch_pair_passed',
        'face_audit_passed', 'side_objectives_match_native', 'side_gradients_finite', 'finite')
    return {**{key: True for key in keys}, 'F_squared_armijo_passed': accepted}


CASES = [
    pytest.param([float('inf'), 0.], id='inf-coordinate'),
    pytest.param([float('nan'), 0.], id='nan-coordinate'),
    pytest.param([1e308, 1e308], id='finite-coordinate-norm-overflow'),
]


def _search_with_writer(helper, path, bad):
    record: dict[str, Any] = {'candidate_slots_used': 9, 'last_confirmed_iterations': 2,
        'current_control_sha256': 'closed', 'last_confirmed_control_sha256': 'closed',
        'closed_steps': [{'P2': True}, {'P2': True}]}
    chart_calls, evaluator_calls = [], []

    def chart(current, direction, alpha):
        chart_calls.append(alpha)
        return torch.tensor(bad, dtype=current.dtype) if len(chart_calls) == 1 else current + alpha * direction

    def evaluate(*args):
        evaluator_calls.append(args)
        return _facts()

    result = helper.search_candidates(_model(), torch.zeros(2, dtype=torch.float64), evaluate,
        chart_candidate=chart,
        progress=lambda trials, meta: runner._persist_search_progress(record, path, 9, trials, meta),
        limit=2)
    return result, record, chart_calls, evaluator_calls


@pytest.mark.parametrize('bad', CASES)
def test_nonfinite_chart_is_recorded_then_good_candidate_evaluated(tmp_path, bad):
    path = tmp_path / 'step.json'
    (accepted, trials, meta), record, charts, evaluations = _search_with_writer(corrected, path, bad)
    assert accepted is not None and accepted['index'] == 2
    assert len(charts) == 2 and len(evaluations) == 1
    assert trials[0]['status'] == 'nonfinite_chart_movement_refused'
    assert trials[0]['actual_path_norm'] is None and trials[0]['accepted'] is False
    assert meta['slots_started'] == meta['slots_completed'] == 2
    assert meta['evaluations_started'] == meta['evaluations_completed'] == 1
    saved = json.loads(path.read_text())
    assert saved == record
    assert saved['candidate_slots_used'] == 11
    assert saved['current_control_sha256'] == saved['last_confirmed_control_sha256'] == 'closed'
    assert saved['last_confirmed_iterations'] == 2 and saved['closed_steps'] == [{'P2': True}, {'P2': True}]


@pytest.mark.parametrize('bad', CASES)
def test_frozen_helper_reproduces_json_failure(tmp_path, bad):
    path = tmp_path / 'step.json'
    with pytest.raises(ValueError, match='Out of range float'):
        _search_with_writer(frozen, path, bad)
    saved = json.loads(path.read_text())
    assert saved['candidate_trials'][0]['status'] == 'slot_started'
    assert saved['search']['evaluations_started'] == saved['search']['evaluations_completed'] == 0
    assert saved['last_confirmed_iterations'] == 2


def test_finite_search_results_and_progress_receipts_are_unchanged(tmp_path):
    outputs = []
    for helper, name in [(frozen, 'old'), (corrected, 'new')]:
        path = tmp_path / (name + '.json')
        record: dict[str, Any] = {'candidate_slots_used': 0}
        calls = []

        def evaluate(*args):
            calls.append(args)
            return _facts(len(calls) == 3)

        result = helper.search_candidates(_model(), torch.zeros(2, dtype=torch.float64), evaluate,
            chart_candidate=lambda current, direction, alpha: current + alpha * direction,
            progress=lambda t, m: old_runner._persist_search_progress(record, path, 0, t, m))
        outputs.append((result, path.read_bytes()))
    assert outputs[0] == outputs[1]


def test_chart_exception_preserves_zero_evaluation_count(tmp_path):
    record: dict[str, Any] = {'candidate_slots_used': 3, 'last_confirmed_iterations': 2}
    path = tmp_path / 'step.json'

    def chart(*_args):
        raise ValueError('chart construction failed')

    with pytest.raises(ValueError, match='chart construction'):
        corrected.search_candidates(_model(), torch.zeros(2, dtype=torch.float64),
            lambda *_: pytest.fail('chart failure must not evaluate'), chart_candidate=chart,
            progress=lambda t, m: old_runner._persist_search_progress(record, path, 3, t, m))
    saved = json.loads(path.read_text())
    assert saved['search']['slots_started'] == 1 and saved['search']['slots_completed'] == 0
    assert saved['search']['evaluations_started'] == saved['search']['evaluations_completed'] == 0
    assert saved['candidate_trials'][0]['status'] == 'slot_failed'
    assert saved['last_confirmed_iterations'] == 2


def test_prospective_parent_admits_closed_prefix_without_partial_operators():
    parent = runner._load_parent()
    assert parent['current_control_sha256'] == runner.BASE_SHA
    assert runner.tangent._tensor_sha(torch.tensor(parent['current_control'], dtype=torch.float64)) == runner.BASE_SHA
    assert parent['final_repeat']['deadline_passed'] is True
    assert parent['current_full_G'] is not None
    assert not {'gn_direction', 'dense_solve_audit', 'jacobian_row_history'} & parent.keys()
    assert runner.search_progress is corrected
    assert runner.POLICY['jacobian_row_vjp_calls'] == 72


def test_new_atomic_checkpoint_drops_unqualified_top_solve_audit(tmp_path):
    # Saved P2 facts exercise only the checkpoint contract in a temporary file.
    parent = runner._load_parent()
    record = {'last_confirmed_iterations': 0, 'closed_steps': [], 'dense_solve_audit': {'stale': True}}
    committed = runner._commit_closed_point(record, {'accepted_full_G': parent['current_full_G']},
        parent['proposal'], parent['final_repeat'], tmp_path / 'step.json')
    assert 'dense_solve_audit' not in committed
    assert committed['current_control_sha256'] == runner.BASE_SHA
    assert committed['last_confirmed_closure'] == parent['final_repeat']
    assert record['dense_solve_audit'] == {'stale': True}
