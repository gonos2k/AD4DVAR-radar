"""Exact matrix-free FV root refinement without objective subtraction."""
from pathlib import Path
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples/weather_scenarios'))
from fv_sensitivity_probe import make_case
from advar import fv_sensitivity as s
from advar import variational as v


@pytest.fixture(scope='module')
def near_stationary():
    observations, frozen, _, _ = make_case()
    frozen = replace(frozen, analysis_config=replace(
        frozen.analysis_config, maximum_outer_iterations=16,
        maximum_pcg_iterations=96, gradient_tolerance=1e-8,
        step_tolerance=1e-12, pcg_relative_tolerance=1e-10))
    result = v.solve_analysis(observations, frozen)
    return result.control, observations, frozen


def test_matrix_free_refinement_decreases_actual_stationary_residual(near_stationary):
    control, observations, frozen = near_stationary
    changed = control.clone()
    changed[-1] += 1e-5
    accepted = []
    refined, records = s.refine_fv_stationarity(
        changed, observations, frozen, gradient_tolerance=1e-10,
        on_step=lambda value, record: accepted.append((value, record)))
    gradient = torch.func.grad(v.robust_objective)(refined, observations, frozen)
    assert float(gradient.norm()) <= 1e-10
    assert records
    assert len(accepted) == len(records)
    assert not accepted[-1][0].requires_grad
    assert accepted[-1][0].grad_fn is None
    torch.testing.assert_close(accepted[-1][0], refined)
    assert all(r['gradient_after'] < r['gradient_before'] for r in records)
    assert all('gradient_max_before' in r and 'gradient_max_after' in r for r in records)
    assert all(r['face_margin'] > 0 for r in records)
    assert all(r['normal_products'] <= 64 for r in records)
    assert all(r['linear_relative_residual'] <= max(
        1e-6, r['linear_absolute_tolerance']/r['gradient_before']) for r in records)


def test_negative_curvature_cannot_be_polished_into_a_certificate(near_stationary, monkeypatch):
    control, observations, frozen = near_stationary
    monkeypatch.setattr(s, 'robust_objective', lambda c, *_: -0.5 * c.square().sum())
    with pytest.raises(RuntimeError, match='positive definite'):
        s.refine_fv_stationarity(control, observations, frozen)


def test_sensitive_face_tie_is_rejected_before_newton(near_stationary):
    control, observations, frozen = near_stationary
    tied = control.clone()
    tied[frozen.active_field_index.numel():-1] = 0
    with pytest.raises(ValueError, match='face sign'):
        s.refine_fv_stationarity(tied, observations, frozen)


def test_stopping_uses_max_component_residual_when_l2_is_larger(near_stationary, monkeypatch):
    control, observations, frozen = near_stationary
    tolerance = 1e-3
    component = 0.75 * tolerance

    def synthetic_objective(value, *_):
        return component * value.sum()

    monkeypatch.setattr(s, 'robust_objective', synthetic_objective)
    seen = []
    refined, records = s.refine_fv_stationarity(
        control, observations, frozen, gradient_tolerance=tolerance,
        on_step=lambda value, record: seen.append((value, record)))
    expected_max = component
    expected_norm = component * control.numel() ** 0.5
    assert expected_max < tolerance < expected_norm
    assert not records
    assert not seen
    torch.testing.assert_close(refined, control)
