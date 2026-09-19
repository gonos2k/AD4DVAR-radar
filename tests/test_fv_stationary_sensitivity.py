"""Small C5 derivative-contract regressions; no long probe execution."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from advar.variational import initial_control, prepare_analysis

_ORACLE_SPEC = spec_from_file_location(
    "fv_sensitivity_probe",
    Path(__file__).parents[1] / "examples/weather_scenarios/fv_sensitivity_probe.py",
)
if _ORACLE_SPEC is None or _ORACLE_SPEC.loader is None:
    raise RuntimeError("FV sensitivity probe module is unavailable")
_ORACLE = module_from_spec(_ORACLE_SPEC)
_ORACLE_SPEC.loader.exec_module(_ORACLE)
dense_hessian = _ORACLE.dense_hessian
face_signs = _ORACLE.face_signs
make_case = _ORACLE.make_case
stationary_sensitivity = _ORACLE.stationary_sensitivity


def test_polish_uses_gradient_merit_when_cost_change_is_unresolvable():
    y = torch.tensor([1e-8], dtype=torch.float64)

    def objective(c, observations):
        # The constant erases every representable cost difference in this case.
        return c.new_tensor(1e20) + 0.5 * (c - observations).square().sum()

    start = torch.zeros_like(y)
    assert objective(start, y) == objective(y, y)
    result = _ORACLE.polish(objective, start, y)
    torch.testing.assert_close(result, y, rtol=0, atol=1e-20)


def test_polish_does_not_accept_a_zero_gradient_saddle():
    zero = torch.zeros(2, dtype=torch.float64)

    def objective(c, observations):
        return 0.5 * (c[0].square() - c[1].square())

    with pytest.raises(torch.linalg.LinAlgError):
        _ORACLE.polish(objective, zero, zero)


def test_dense_hessian_rejects_nonsymmetric_and_nonfinite_derivatives():
    zero = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="symmetric"):
        dense_hessian(lambda c, y: torch.stack((c[0] + c[1], c[1])), zero, zero)
    with pytest.raises(ValueError, match="finite"):
        dense_hessian(lambda c, y: c * float("nan"), zero, zero)


def test_polish_rejects_infinite_cost_even_with_zero_gradient():
    zero = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="objective is not finite"):
        _ORACLE.polish(lambda c, y: c.square().sum() + float("inf"), zero, zero)


def test_polish_checks_every_trial_branch():
    y = torch.ones(1, dtype=torch.float64)

    def reject(start, stop):
        raise ValueError("branch boundary")

    with pytest.raises(RuntimeError, match="refinement"):
        _ORACLE.polish(
            lambda c, y: 0.5 * (c - y).square().sum(),
            torch.zeros_like(y), y, check_step=reject,
        )


def test_stationary_sensitivity_scalar_quadratic_has_the_right_sign():
    stiffness = 3.0
    observation = torch.tensor([2.0], dtype=torch.float64)
    control = observation / (1.0 + stiffness)

    def objective(value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * (value - y).square().sum() + 0.5 * stiffness * value.square().sum()

    def score(value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * value.sum() + 3.0 * y.sum()

    sensitivity, direct, residual = stationary_sensitivity(
        objective,
        score,
        control,
        observation,
    )
    expected = torch.full_like(observation, 3.0 + 2.0 / (1.0 + stiffness))
    torch.testing.assert_close(sensitivity, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(direct, torch.full_like(observation, 3.0))
    assert residual < 1e-10


def test_stationary_sensitivity_rejects_a_nonstationary_point():
    observation = torch.tensor([2.0], dtype=torch.float64)

    def objective(value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * (value - y).square().sum() + 0.5 * value.square().sum()

    def score(value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return value.sum() + y.sum()

    with pytest.raises(ValueError, match="refined|stationary"):
        stationary_sensitivity(
            objective,
            score,
            torch.zeros_like(observation),
            observation,
        )


def test_face_signs_distinguishes_structural_and_sensitive_zero_faces():
    _, frozen, _, _ = make_case()
    specification = frozen.fv_transport
    assert specification is not None
    basis = torch.zeros_like(specification.psi_basis)
    basis[0, 1:, :] = 0.02 * torch.arange(
        basis.shape[1] - 1,
        dtype=basis.dtype,
    ).unsqueeze(1)
    structural = replace(
        frozen,
        fv_transport=replace(specification, psi_basis=basis),
    )
    control = initial_control(structural)
    field_size = frozen.active_field_index.numel()
    control[field_size] = 0.4
    signs = face_signs(control, structural)
    assert signs.numel() > 0

    sensitive_control = torch.zeros_like(control)
    with pytest.raises(ValueError, match="tie"):
        face_signs(sensitive_control, structural)


def test_prepare_analysis_full_detected_domain_uses_perturbed_first_frame():
    observations, frozen, _, _ = make_case()
    assert bool(observations.detected_mask.all())
    perturbation = torch.linspace(
        -0.02,
        0.02,
        frozen.input_frames_dbz.numel(),
        dtype=frozen.input_frames_dbz.dtype,
    ).reshape_as(frozen.input_frames_dbz)
    perturbed_frames = frozen.input_frames_dbz + perturbation
    perturbed_observations, perturbed_frozen = prepare_analysis(
        perturbed_frames,
        nowcast_config=frozen.nowcast_config,
        analysis_config=frozen.analysis_config,
        observation_std_dbz=0.1,
        fv_transport=frozen.fv_transport,
    )
    assert bool(perturbed_observations.detected_mask.all())
    torch.testing.assert_close(
        perturbed_frozen.initial_background_dbz,
        perturbed_observations.dbz[0],
        rtol=0,
        atol=0,
    )


def test_dense_hessian_oracle_has_a_small_control_cap():
    control = torch.zeros(33, dtype=torch.float64)
    observations = torch.zeros(33, dtype=torch.float64)

    def gradient(value: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return value

    with pytest.raises(ValueError, match="32"):
        dense_hessian(gradient, control, observations)


def test_face_box_rejects_possible_crossing_despite_matching_endpoints():
    _, frozen, _, _ = make_case()
    control = initial_control(frozen)
    size = frozen.active_field_index.numel()
    # A single x-flux is a difference of two independently variable coefficients.
    spec = frozen.fv_transport
    basis = torch.zeros_like(spec.psi_basis)
    y = torch.arange(basis.shape[1], dtype=basis.dtype)[:, None]
    basis[0] = y
    basis[1] = -y
    frozen = replace(frozen, fv_transport=replace(spec, psi_basis=basis))
    start, stop = control.clone(), control.clone()
    start[size : size + 2] = start.new_tensor([0.2, 0.1])
    stop[size : size + 2] = stop.new_tensor([0.4, 0.3])
    torch.testing.assert_close(face_signs(start, frozen), face_signs(stop, frozen))
    with pytest.raises(ValueError, match="box|cross"):
        _ORACLE.face_branch_margin(start, stop, frozen)
    close = start.clone()
    close[size] += 0.001
    assert _ORACLE.face_branch_margin(start, close, frozen) > 0


def test_nan_stationarity_is_not_accepted():
    x = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="stationary|refined"):
        stationary_sensitivity(
            lambda c, y: (c * y * float("nan")).sum(), lambda c, y: c.sum(), x, x
        )


def test_fv_total_observation_sensitivity_matches_stationary_reanalysis():
    import json

    report = _ORACLE.run_probe()
    print("FV_SENSITIVITY_REPORT=" + json.dumps(report))
    assert report["gn_api_curvature"] == "irls_gauss_newton"
    assert report["gn_api_score_error"] < 1e-12
    assert report["gn_api_direct_max_error"] < 1e-12
    assert report["gn_api_vs_dense_max"] < 1e-9
    assert report["gn_api_fixed_background_vs_dense_max"] < 1e-9
    assert 1e-8 < report["gn_api_vs_exact_relative"] < 0.01
    assert report["gn_api_normal_products"] <= 128
    assert report["gn_api_adjoint_relative_residual"] < 1e-10
    assert report["gradient_max"] < 1e-8
    assert report["adjoint_relative_residual"] < 1e-10
    assert report["pcg_vs_dense_sensitivity_max"] < 1e-10
    assert report["hessian_min_eigenvalue"] > 0
    assert report["hessian_symmetry_max"] < 1e-8
    assert report["hessian_vs_irls_gn_relative"] > 1e-6
    assert report["field_conditioned_data_dynamics_rank"] == 4
    assert report["polish_face_box_margin"] > 0
    assert report["total_vs_frozen_background_max"] > 1e-6
    assert report["direct_first_frame_norm"] > 0.01
    assert report["later_frame_partial_difference_max"] < 1e-10
    points = report["perturbations"]
    for point in points:
        assert point["face_box_margin"] > 0
        assert point["gradient_max"] < 1e-8
        assert point["actual_change"] * point["linear_change"] > 0
        assert point["actual_change"] * point["gn_linear_change"] > 0
        assert point["gn_taylor_error"] / abs(point["actual_change"]) < 0.1
    for large, small in zip(points, points[1:]):
        assert 3.5 < large["taylor_error"] / small["taylor_error"] < 4.5
        assert 3.5 < large["central_error"] / small["central_error"] < 4.5


def test_face_margin_accounts_for_cancellation_in_backend_vertex_sums():
    _, frozen, _, _ = make_case()
    spec = frozen.fv_transport
    basis = torch.zeros_like(spec.psi_basis)
    # A tiny represented row difference sits on large vertex values.
    basis[0, 1:] = 1e16
    basis[0, 2:] += 2.0
    limits = torch.zeros_like(spec.coefficient_limits)
    limits[0] = 1e-16
    frozen = replace(
        frozen, fv_transport=replace(spec, psi_basis=basis, coefficient_limits=limits)
    )
    control = initial_control(frozen)
    control[frozen.active_field_index.numel()] = 0.4
    with pytest.raises(ValueError, match="tie"):
        face_signs(control, frozen)
    with pytest.raises(ValueError, match="box|cross"):
        _ORACLE.face_branch_margin(control, control, frozen)
