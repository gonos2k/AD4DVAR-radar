"""Bounded contract checks for the explicit 8x10 scaled FV fixture."""
import torch
import pytest

from advar import variational as v
from examples.weather_scenarios import fv_scaled_research_case as scaled


@pytest.fixture(scope="module")
def case():
    return scaled.make_case()


def prescribed_seed(case):
    control = v.initial_control(case.frozen)
    control[-6:] = control.new_tensor([0.1, -0.08, 0.06, 0.04, -0.03, 0.008])
    return control


def test_scaled_fixture_has_prescribed_geometry_counts_and_fixed_pattern(case):
    spec = case.frozen.fv_transport
    assert case.truth_initial_echo.shape == (8, 10)
    assert case.observations.dbz.shape == (3, 8, 10)
    assert spec is not None
    assert spec.psi_basis.shape == (5, 9, 11)
    assert torch.equal(spec.psi_basis[0, :, 0], torch.arange(9, dtype=torch.float64) / 2)
    assert torch.equal(spec.psi_basis[1, 0, :], torch.arange(11, dtype=torch.float64) / 2)
    assert spec.spacing_yx == (5.0, 5.0)
    assert torch.equal(spec.coefficient_limits, torch.tensor([0.11, 0.08, 0.07, 0.04, 0.03], dtype=torch.float64))
    assert case.parameters.shape == (241,)
    assert torch.equal(case.parameters[:-1], case.observations.dbz.flatten())
    assert case.parameters[-1] == pytest.approx(0.02)
    assert case.pattern.shape == (8, 10)
    assert torch.equal(case.pattern, torch.linspace(-0.2, 0.3, 80, dtype=torch.float64).reshape(8, 10))
    assert case.verification.shape == (8, 10)
    assert case.definition["observation_leads_seconds"] == [0, 60, 120]
    assert case.definition["verification_lead_seconds"] == 180
    assert case.definition["controls"] == 86
    assert case.definition["parameters"] == 241


def test_scaled_schedule_support_and_cfl_box_are_bounded(case):
    spec = case.frozen.fv_transport
    assert spec is not None
    assert len(spec.boundary_echo) == 36
    assert len(spec.boundary_support) == 36
    assert len(case.future_boundary_echo) == 18
    assert len(case.future_boundary_support) == 18
    assert case.definition["total_boundary_stage_pairs"] == 54
    assert case.definition["rk_stages"] == 108
    assert case.definition["coefficient_box_cfl_upper_bound"] < 0.5
    for stages in (*spec.boundary_support, *case.future_boundary_support):
        for edge in (*stages[0], *stages[1]):
            assert bool(torch.all(edge == 1))

    seed = prescribed_seed(case)
    coefficients = scaled.t.bounded_fv_coefficients(
        seed[-6:-1],
        psi_basis=spec.psi_basis,
        coefficient_limits=spec.coefficient_limits,
        dt_seconds=60.0 / 18,
        spacing_yx=(5.0, 5.0),
        reconstruction="minmod",
    )
    assert bool(torch.all(coefficients.abs() < spec.coefficient_limits))
    assert scaled.actual_cfl(case, seed) < 0.5


def test_scaled_functions_preserve_parameter_control_layout_and_strict_branch(case):
    objective, score, branch_check = scaled.functions(case)
    seed = prescribed_seed(case)
    branch, _ = branch_check(seed, case.parameters)
    assert branch["euler_stages"] == 108
    assert float(objective(seed, case.parameters)) > 0
    assert float(score(seed, case.parameters)) >= 0

    expected = {**branch, "choices": [], "face_signs": []}
    _, _, guarded = scaled.functions(case, expected_branch=expected)
    with pytest.raises(ValueError, match="branch identity"):
        guarded(seed, case.parameters)


def test_scaled_functions_refuse_wrong_layout_but_trace_zero_growth(case):
    _, _, branch_check = scaled.functions(case)
    seed = prescribed_seed(case)
    with pytest.raises(ValueError, match="layout mismatch"):
        branch_check(seed[:-1], case.parameters)
    with pytest.raises(ValueError, match="layout mismatch"):
        branch_check(seed, case.parameters[:-1])

    zero_growth = seed.clone()
    zero_growth[-1] = 0.0
    branch, _ = branch_check(zero_growth, case.parameters)
    assert branch["euler_stages"] == 108
