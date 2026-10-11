from __future__ import annotations
import pytest
import torch
from examples.weather_scenarios import fv_point_3h_gn_local_candidate_commit as candidate
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared
from tests.test_fv_point_3h_gn_prepared_search import _proposal_and_repeat


def test_closed_same_point_operator_qualified_without_new_row_solve():
    parent=candidate._load_parent()
    assert candidate.tangent._tensor_sha(torch.tensor(parent['current_control'],dtype=torch.float64))==candidate.BASE_SHA
    assert candidate.POLICY['jacobian_row_vjp_calls']==candidate.POLICY['dense_solves']==0


def test_selection_does_not_exclude_valid_changed_selector():
    changed={'index':1,'eligible':True,'F_squared':.9,'objective':1.,'trace_matches_base':False}
    same={'index':2,'eligible':True,'F_squared':.99,'objective':1.,'trace_matches_base':True}
    assert candidate.select_candidate([changed,same]) is changed
    changed['eligible']=False
    assert candidate.select_candidate([changed,same]) is same
    same['eligible']=False
    assert candidate.select_candidate([changed,same]) is None


def test_selection_ties_use_J_then_fixed_candidate_order():
    first={'index':1,'eligible':True,'F_squared':1.,'objective':1.}
    second={'index':2,'eligible':True,'F_squared':1.,'objective':.9}
    assert candidate.select_candidate([first,second]) is second
    second['objective']=1.
    assert candidate.select_candidate([second,first]) is first


def test_P2_mismatch_refuses_promotion_of_diagnostic_candidate(tmp_path):
    proposal,repeat=_proposal_and_repeat();repeat['runtime_unchanged']=False
    with pytest.raises(ValueError):prepared._persist_closed_commit(tmp_path/'step.json',{},proposal,repeat)
    assert not (tmp_path/'step.json').exists()
