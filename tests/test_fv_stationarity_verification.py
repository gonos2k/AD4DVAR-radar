"""Local first-order FV verification must not imply legacy FSO eligibility."""
from dataclasses import replace
from pathlib import Path
import runpy
import pytest
import torch
from advar import variational as v
from advar import fv_sensitivity as sensitivity
from advar.fv_sensitivity import verify_fv_stationarity

@pytest.fixture(scope='module')
def analysis():
    make_case = runpy.run_path(str(Path(__file__).parents[1] / 'examples/weather_scenarios/fv_sensitivity_probe.py'))['make_case']
    observations, frozen, _, _ = make_case()
    config = replace(frozen.analysis_config, maximum_outer_iterations=16,
                     maximum_pcg_iterations=96, gradient_tolerance=1e-10,
                     step_tolerance=1e-12, pcg_relative_tolerance=1e-10)
    frozen = replace(frozen, analysis_config=config)
    return v.solve_analysis(observations, frozen), observations, frozen


def test_explicit_verification_checks_residual_jvps_without_legacy_promotion(analysis):
    result, observations, frozen = analysis
    assert not result.stationarity_verified
    checked, evidence = verify_fv_stationarity(result, observations, frozen)
    assert checked.stationarity_verified
    assert evidence['gradient_norm'] <= evidence['tolerance']
    assert len(evidence['checks']) == 5
    assert all(item['face_box_margin'] > 0 for item in evidence['checks'])
    assert not hasattr(checked, 'linearization')
    assert not hasattr(checked, 'fso_eligible')
    torch.testing.assert_close(checked.control, result.control, rtol=0, atol=0)


def test_verification_rejects_nonstationary_control(analysis):
    result, observations, frozen = analysis
    changed = result.control.clone(); changed[-1] += 0.01
    with pytest.raises(ValueError, match='gradient exceeds'):
        verify_fv_stationarity(replace(result, control=changed), observations, frozen)


def test_verification_catches_a_wrong_residual_derivative(analysis, monkeypatch):
    result, observations, frozen = analysis
    original = sensitivity.residual_vector
    def broken(control, *args):
        residual = original(control, *args)
        return residual + 0.1 * (control.sum() - control.sum().detach())
    monkeypatch.setattr(sensitivity, 'residual_vector', broken)
    with pytest.raises(ValueError, match='gradient direction'):
        verify_fv_stationarity(result, observations, frozen)


def test_verification_rejects_missing_observations(analysis):
    result, observations, frozen = analysis
    mask = observations.valid_mask.clone(); mask[0, 0, 0] = False
    with pytest.raises(ValueError):
        verify_fv_stationarity(result, replace(observations, valid_mask=mask), frozen)


@pytest.mark.parametrize('name', ['support_frames', 'psi_coefficients', 'log_growth_per_step'])
def test_verification_rejects_stale_trajectory_components(analysis, name):
    result, observations, frozen = analysis
    changed = getattr(result.trajectory, name) + 0.01
    stale = replace(result, trajectory=replace(result.trajectory, **{name: changed}))
    with pytest.raises(ValueError, match=f'trajectory {name}'):
        verify_fv_stationarity(stale, observations, frozen)
