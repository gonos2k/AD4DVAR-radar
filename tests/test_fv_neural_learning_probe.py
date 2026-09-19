"""Actual neural observation-error learning, separate from FV neural priors."""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples/weather_scenarios'))
import fv_neural_learning_probe as probe


def assert_neural_evidence(report):
    assert report['control_count'] == 24
    assert report['parameter_count'] == 3
    assert not report['production_learning_eligible']
    assert report['stationarity_max'] < 1e-10
    assert report['finite_difference_stationarity_max'] < 1e-10
    assert report['adjoint_relative_residual'] < 1e-10
    slope = report['predicted_directional_slope']
    errors = [abs(value-slope) for value in report['finite_difference_slopes']]
    assert 3 < errors[0]/errors[1] < 5
    assert errors[1] < 1e-5*abs(slope)
    predicted = -0.25*sum(g*g for g in report['parameter_gradient'])
    actual = report['train_score_after']-report['train_score_before']
    assert actual < 0 and abs(actual-predicted) < 0.01*abs(predicted)
    assert report['initial_parameters'] != report['updated_parameters']
    assert report['holdout_gain_above_numerical_error']
    assert all(value == 0 for value in report['reload_errors'].values())
    assert report['same_face_signs'] and report['face_margin'] > 0
    for name in ('holdout_base_diagnostics','holdout_updated_diagnostics'):
        assert report[name]['gradient_max'] < 1e-10
        assert report[name]['hessian_min_eigenvalue'] > 0


def test_neural_features_are_not_a_duplicate_intercept_and_output_is_bounded():
    observations, _, _, _ = probe.oracle.make_case()
    features = probe._features(observations)
    design = torch.cat((features, torch.ones(features.shape[0], 1, dtype=features.dtype)), dim=1)
    assert torch.linalg.matrix_rank(design) == 3
    model = probe.TinyObservationStd()
    probe._set_flat(model, torch.tensor([1e3,-1e3,1e3], dtype=torch.float64))
    assert bool((model(features).abs() <= 0.25).all())


def test_neural_fv_update_and_checkpoint_reanalysis(tmp_path):
    assert_neural_evidence(probe.run_probe(checkpoint_path=tmp_path/'model.pt'))
