"""The small trained NN's feature path must survive the full FV reanalysis."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples/weather_scenarios'))
from fv_neural_observation_path_probe import run_probe


def assert_path_evidence(result):
    assert not result['parameter_retuning']
    assert not result['production_learning_eligible']
    assert not result['neural_prior_eligible']
    assert result['stationarity_max'] < 1e-10
    assert result['endpoint_stationarity_max'] < 1e-10
    assert result['adjoint_relative_residual'] < 1e-9
    assert result['hessian_min_eigenvalue'] > 0
    assert result['same_face_signs']
    assert min(result['face_branch_margins']) > 0
    slope = result['directional_isolated']
    assert slope < 0
    assert abs(result['isolated_minus_prior']) < abs(slope) * 1e-5
    assert max(result['finite_difference_errors']) < abs(slope) * 1e-3


def test_neural_feature_path_matches_reanalysis_without_changing_raw_observations():
    assert_path_evidence(run_probe())
