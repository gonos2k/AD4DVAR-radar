"""The constructed point problem has a real smooth stationary FV control."""

from dataclasses import replace

import pytest
import torch

from examples.weather_scenarios.fv_point_centered_prior_case import make_case


def test_centered_prior_is_stationary_and_strict_at_declared_control():
    case = make_case()
    gradient = torch.func.grad(case.objective, argnums=0)(case.control, case.parameters)
    assert case.control.shape == (26,) and case.parameters.shape == (13,)
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert case.objective(case.control, case.parameters) == 0
    branch, _ = case.branch_check(case.control, case.parameters)
    assert branch["euler_stages"] == 54
    assert branch["minimum_scaled_slope_margin"] > 1e-4
    assert branch["minimum_scaled_face_flux_margin"] > 1e-4
    assert case.direction.nonzero().flatten().tolist() == [4, 5, 6, 7]


def test_centered_prior_changes_only_fixed_dynamics_prior_mean():
    case = make_case()
    perturbation = torch.linspace(-0.02, 0.03, 26, dtype=torch.float64)
    control = case.control + perturbation
    dynamics = control[-6:]
    mean = case.dynamics_prior_mean
    expected = (case.problem.objective(control, case.parameters)
                - 0.5 * dynamics.square().sum()
                + 0.5 * (dynamics - mean).square().sum())
    torch.testing.assert_close(case.objective(control, case.parameters), expected,
                               rtol=0, atol=1e-15)
    assert case.identity["dynamics_prior_mean_sha256"] != case.identity["parameters_sha256"]


def test_branch_check_rejects_wrong_nominal_signature():
    case = make_case()
    changed = dict(case.nominal_branch)
    changed["choices"] = []
    wrong = replace(case, nominal_branch=changed)
    with pytest.raises(ValueError, match="qualified nominal branch"):
        wrong.branch_check(case.control, case.parameters)
