"""FV residual integration; these tests do not claim public solver acceptance."""

from dataclasses import replace

import pytest
import torch

from advar.matrix_free import pcg
from advar.nowcast import NowcastConfig
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.transport import finite_volume_trajectory
from advar.variational import (
    AnalysisConfig, FVAnalysisTransport, analysis_trajectory, initial_control,
    prepare_analysis, residual_vector, solve_analysis,
)
from advar import variational as implementation


def _fixture(*, observation_std_dbz=None, reconstruction="donorcell", smooth=False):
    dtype = torch.float64
    y, x = torch.meshgrid(torch.arange(5, dtype=dtype), torch.arange(6, dtype=dtype), indexing="ij")
    basis = torch.stack((y, -x, -.5*(x.square()+y.square())))
    limits = torch.tensor([.15, .15, .02], dtype=dtype)
    q0 = dbz_to_echo(torch.tensor([[18., 21., 19., 23., 20.], [22., 18., 24., 19., 21.],
                                   [19., 24., 20., 22., 18.], [21., 20., 23., 18., 22.]], dtype=dtype),
                     min_dbz=-10.)
    if smooth:
        q0 = dbz_to_echo(18 + .2*x[:-1, :-1].square()
                         + .3*y[:-1, :-1].square()
                         + .05*x[:-1, :-1]*y[:-1, :-1], min_dbz=-10.)
    zero = (q0.new_full((4,), 60.), q0.new_full((4,), 70.),
            q0.new_full((5,), 80.), q0.new_full((5,), 90.))
    one = tuple(torch.ones_like(edge) for edge in zero)
    boundary = tuple((tuple(edge*(1+.01*i) for edge in zero),
                      tuple(edge*(1+.01*(i+1)) for edge in zero)) for i in range(4))
    support = ((one, one),) * 4
    specification = FVAnalysisTransport(
        psi_basis=basis, coefficient_limits=limits, substeps_per_interval=2,
        spacing_yx=(10., 10.), boundary_echo=boundary, boundary_support=support,
        reconstruction=reconstruction, replay=True,
    )
    target, _ = finite_volume_trajectory(
        q0, torch.ones_like(q0), limits * q0.new_tensor([.2, -.1, .15]).tanh(), q0.new_tensor(.01),
        psi_basis=basis, leads=2, substeps_per_interval=2, interval_seconds=60.,
        spacing_yx=(10., 10.), boundary_echo=boundary, boundary_support=support,
        reconstruction=reconstruction,
    )
    frames = echo_to_dbz(target, min_dbz=-10.)
    observations, frozen = prepare_analysis(
        frames, nowcast_config=NowcastConfig(interval_minutes=1, horizon_minutes=1, min_dbz=-10.),
        analysis_config=AnalysisConfig(field_smoothness_weight=0.), fv_transport=specification,
        observation_std_dbz=observation_std_dbz,
    )
    return observations, frozen, specification


def test_fv_control_layout_and_trajectory_do_not_fabricate_global_motion():
    _, frozen, specification = _fixture()
    control = initial_control(frozen)
    assert control.numel() == frozen.active_field_index.numel() + 4
    trajectory = analysis_trajectory(control, frozen)
    assert trajectory.displacement_yx is None
    assert trajectory.psi_coefficients.shape == (3,)
    assert trajectory.frames_linear.shape == (3, 4, 5)
    assert trajectory.support_frames.shape == (3, 4, 5)
    expected, expected_support = finite_volume_trajectory(
        trajectory.frames_linear[0], frozen.initial_support_mask.to(control.dtype),
        trajectory.psi_coefficients, trajectory.log_growth_per_step,
        psi_basis=specification.psi_basis, leads=2, substeps_per_interval=2,
        interval_seconds=60., spacing_yx=(10., 10.),
        boundary_echo=specification.boundary_echo, boundary_support=specification.boundary_support,
        reconstruction="donorcell", replay=False,
    )
    torch.testing.assert_close(trajectory.frames_linear, expected, rtol=0, atol=0)
    torch.testing.assert_close(trajectory.support_frames, expected_support, rtol=0, atol=0)


def test_existing_residual_has_fv_jvp_vjp_and_normal_operator():
    observations, frozen, _ = _fixture()
    control = initial_control(frozen)
    control[-4:] = control.new_tensor([.1, -.15, .2, .05])
    function = lambda c: residual_vector(c, observations, frozen)
    direction = .01 * torch.cos(torch.arange(control.numel(), dtype=control.dtype))
    value, tangent = torch.func.jvp(function, (control,), (direction,))
    _, pullback = torch.func.vjp(function, control)
    seed = torch.sin(value)
    torch.testing.assert_close((seed*tangent).sum(), (direction*pullback(seed)[0]).sum(), rtol=1e-11, atol=1e-13)
    h = 1e-5
    difference = (function(control+h*direction)-function(control-h*direction))/(2*h)
    torch.testing.assert_close(tangent, difference, rtol=3e-6, atol=3e-10)
    # Column assembly avoids requiring vmap support from the replay helper.
    columns = [torch.func.jvp(function, (control,), (e,))[1]
               for e in torch.eye(control.numel(), dtype=control.dtype)]
    jacobian = torch.stack(columns, dim=1)
    torch.testing.assert_close(pullback(tangent)[0], jacobian.T @ (jacobian @ direction), rtol=1e-11, atol=1e-13)


def test_existing_pcg_gn_step_reduces_the_fv_objective():
    observations, frozen, _ = _fixture()
    control = initial_control(frozen)
    function = lambda c: residual_vector(c, observations, frozen)
    residual, pullback = torch.func.vjp(function, control)
    gradient = pullback(residual)[0]
    def normal(v):
        return pullback(torch.func.jvp(function, (control,), (v,))[1])[0] + .1*v
    linear = pcg(normal, -gradient, rtol=1e-10, max_iterations=80)
    assert linear.converged
    before = implementation._evaluate_control(control, observations, frozen)[0]
    after = implementation._evaluate_control(control+.25*linear.solution, observations, frozen)[0]
    assert after < before


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
def test_shared_solver_returns_only_research_fv_analysis(scheme):
    observations, frozen, _ = _fixture(
        observation_std_dbz=.1, reconstruction=scheme, smooth=(scheme == "minmod"),
    )
    if scheme == "minmod":
        from advar.transport import _muscl_slopes
        initial = analysis_trajectory(initial_control(frozen), frozen).frames_linear[0]
        sx, sy = _muscl_slopes(initial)
        assert torch.count_nonzero(sx) > 0 and torch.count_nonzero(sy) > 0
    result = solve_analysis(observations, frozen)
    assert isinstance(result, implementation.FVAnalysisResult)
    assert result.improved
    assert result.final_objective < result.initial_objective
    assert result.outer_iterations > 0
    assert result.pcg_iterations > 0
    assert not result.stationarity_verified
    assert result.trajectory.displacement_yx is None
    assert result.trajectory.psi_coefficients.shape == (3,)
    actual, trajectory = implementation._evaluate_control(result.control, observations, frozen)
    assert result.final_objective == pytest.approx(float(actual), rel=1e-12)
    torch.testing.assert_close(result.trajectory.frames_linear, trajectory.frames_linear)
    assert torch.isfinite(result.control).all()
    actual_gradient = torch.func.grad(
        lambda c: implementation.robust_objective(c, observations, frozen),
    )(result.control)
    assert result.selected_gradient_norm == pytest.approx(
        float(torch.linalg.vector_norm(actual_gradient)), rel=1e-10, abs=1e-11,
    )


def test_public_fv_forecast_stays_blocked_until_same_backend_is_wired():
    _, frozen, specification = _fixture()
    with pytest.raises(NotImplementedError, match="FV|finite.volume|spatial"):
        implementation.variational_nowcast(
            frozen.input_frames_dbz, nowcast_config=frozen.nowcast_config,
            analysis_config=frozen.analysis_config, fv_transport=specification,
        )


def test_frozen_fv_inputs_do_not_follow_caller_mutation():
    observations, frozen, specification = _fixture()
    control = initial_control(frozen)
    before = residual_vector(control, observations, frozen)
    specification.boundary_echo[0][0][0].add_(100.)
    specification.coefficient_limits.mul_(100.)
    after = residual_vector(control, observations, frozen)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_legacy_control_layout_remains_available():
    _, fv_frozen, _ = _fixture()
    frozen = replace(fv_frozen, fv_transport=None)
    control = initial_control(frozen)
    assert control.numel() == frozen.active_field_index.numel()+3
    assert analysis_trajectory(control, frozen).displacement_yx.shape == (2,)


def test_cfl_certificate_uses_substep_seconds_not_observation_interval():
    _, frozen, _ = _fixture()
    fv = replace(frozen.fv_transport,
                 coefficient_limits=2 * frozen.fv_transport.coefficient_limits)
    frozen = replace(frozen, fv_transport=fv)
    control = initial_control(frozen)
    control[-4:-1] = 2.
    # This coefficient box is safe at 30 s, but not at the 60 s observation interval.
    trajectory = analysis_trajectory(control, frozen)
    assert torch.isfinite(trajectory.frames_linear).all()
    with pytest.raises(ValueError, match="CFL|Courant|courant"):
        implementation.bounded_fv_coefficients(
            control[-4:-1], psi_basis=fv.psi_basis,
            coefficient_limits=fv.coefficient_limits, dt_seconds=60.,
            spacing_yx=fv.spacing_yx, reconstruction=fv.reconstruction,
            max_courant=fv.max_courant,
        )


def test_fv_frozen_irls_gradient_matches_the_true_robust_objective():
    observations, frozen, _ = _fixture()
    control = initial_control(frozen)
    control[-4:] = control.new_tensor([.1, -.15, .2, .3])
    frozen = implementation.freeze_irls_weights(control, observations, frozen)
    weights = frozen.irls_sqrt_weight.clone()
    residual, pullback = torch.func.vjp(
        lambda c: residual_vector(c, observations, frozen), control,
    )
    actual = torch.func.grad(
        lambda c: implementation.robust_objective(c, observations, frozen),
    )(control)
    torch.testing.assert_close(pullback(residual)[0], actual, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(frozen.irls_sqrt_weight, weights, rtol=0, atol=0)


def test_fv_initial_exploration_uses_actual_cost_and_preserves_other_blocks():
    observations, frozen, _ = _fixture()
    base = initial_control(frozen)
    candidate, status = implementation._fv_initial_control_candidate(
        observations, frozen, control=base,
    )
    before = implementation._evaluate_control(base, observations, frozen)[0]
    after = implementation._evaluate_control(candidate, observations, frozen)[0]
    assert status == 'improved'
    assert after < before
    field_size = frozen.active_field_index.numel()
    torch.testing.assert_close(candidate[:field_size], base[:field_size], rtol=0, atol=0)
    torch.testing.assert_close(candidate[-1], base[-1], rtol=0, atol=0)
    torch.testing.assert_close(base, initial_control(frozen), rtol=0, atol=0)
    # Legacy phase-correlation motion is not an input to this FV cost search.
    changed = replace(frozen, baseline_state=replace(
        frozen.baseline_state,
        displacement_yx=base.new_tensor([4., -7.]),
    ))
    other, other_status = implementation._fv_initial_control_candidate(
        observations, changed, control=base,
    )
    assert other_status == status
    torch.testing.assert_close(candidate, other, rtol=0, atol=0)


def test_no_descent_in_constant_echo_is_not_reported_as_stationarity():
    _, _, specification = _fixture()
    q = torch.full((4, 5), 100., dtype=torch.float64)
    edges = (q.new_full((4,), 100.), q.new_full((4,), 100.),
             q.new_full((5,), 100.), q.new_full((5,), 100.))
    specification = replace(specification, boundary_echo=((edges, edges),)*4)
    frames = echo_to_dbz(q, min_dbz=-10.).expand(3, -1, -1).clone()
    observations, frozen = prepare_analysis(
        frames, nowcast_config=NowcastConfig(interval_minutes=1, horizon_minutes=1, min_dbz=-10.),
        analysis_config=AnalysisConfig(field_smoothness_weight=0.), fv_transport=specification,
    )
    candidate, status = implementation._fv_initial_control_candidate(observations, frozen)
    assert status == 'no_descent_found'
    torch.testing.assert_close(candidate, initial_control(frozen), rtol=0, atol=0)
    result = solve_analysis(observations, frozen)
    assert not result.improved
    assert not result.stationarity_verified
    assert result.reason == 'gradient_tolerance_unverified'


def test_initial_exploration_does_not_hide_internal_runtime_errors(monkeypatch):
    observations, frozen, _ = _fixture()
    def fail(*args, **kwargs):
        raise RuntimeError('unexpected internal failure')
    monkeypatch.setattr(implementation, '_evaluate_control', fail)
    with pytest.raises(RuntimeError, match='unexpected internal failure'):
        implementation._fv_initial_control_candidate(observations, frozen)


def test_invalid_initial_cost_is_not_a_no_descent_result(monkeypatch):
    observations, frozen, _ = _fixture()
    def invalid(*args, **kwargs):
        raise FloatingPointError('nonrepresentable initial cost')
    monkeypatch.setattr(implementation, '_evaluate_control', invalid)
    candidate, status = implementation._fv_initial_control_candidate(observations, frozen)
    assert status == 'invalid_initial_objective'
    torch.testing.assert_close(candidate, initial_control(frozen), rtol=0, atol=0)


def test_zero_selected_psi_gradient_escapes_using_actual_observation_cost():
    q = torch.full((2, 2), 100., dtype=torch.float64)
    basis = q.new_tensor([[[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]]])
    empty = tuple(q.new_zeros(2) for _ in range(4))
    known = tuple(q.new_ones(2) for _ in range(4))
    boundary = ((empty, empty),)*2
    support = ((known, known),)*2
    specification = FVAnalysisTransport(
        psi_basis=basis, coefficient_limits=q.new_tensor([.1]),
        substeps_per_interval=1, spacing_yx=(10., 10.),
        boundary_echo=boundary, boundary_support=support, reconstruction='donorcell',
    )
    # Symmetric observed decay gives a zero selected flow gradient; either
    # small signed outflow reduces its mismatch without identifying a sign.
    truth = torch.stack((q, .98*q, .9604*q))
    observations, frozen = prepare_analysis(
        echo_to_dbz(truth, min_dbz=-10.),
        nowcast_config=NowcastConfig(interval_minutes=1, horizon_minutes=1, min_dbz=-10.),
        analysis_config=AnalysisConfig(field_smoothness_weight=0.),
        observation_std_dbz=.1, fv_transport=specification,
    )
    base = initial_control(frozen)
    cost = lambda c: implementation.robust_objective(c, observations, frozen)
    selected = torch.func.grad(cost)(base)
    torch.testing.assert_close(selected[-2], base.new_zeros(()), rtol=0, atol=0)
    candidate, status = implementation._fv_initial_control_candidate(observations, frozen)
    assert status == 'improved'
    assert candidate[-2] != 0
    assert cost(candidate) < cost(base)
    torch.testing.assert_close(candidate[[0, 1, 2, 3, 5]], base[[0, 1, 2, 3, 5]], rtol=0, atol=0)


def test_fv_pcg_does_not_reclassify_internal_errors_as_numerical_failure(monkeypatch):
    observations, frozen, _ = _fixture()
    def internal_failure(*args, **kwargs):
        raise RuntimeError('unexpected operator implementation error')
    monkeypatch.setattr(implementation, 'pcg', internal_failure)
    with pytest.raises(RuntimeError, match='unexpected operator implementation error'):
        solve_analysis(observations, frozen)


def test_fv_trial_does_not_hide_internal_evaluation_errors(monkeypatch):
    observations, frozen, _ = _fixture()
    evaluate = implementation._evaluate_control
    solve_linear = implementation.pcg
    has_linear_step = False
    def linear(*args, **kwargs):
        nonlocal has_linear_step
        result = solve_linear(*args, **kwargs)
        has_linear_step = True
        return result
    def evaluation(*args, **kwargs):
        if has_linear_step:
            raise RuntimeError('unexpected trial implementation error')
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(implementation, 'pcg', linear)
    monkeypatch.setattr(implementation, '_evaluate_control', evaluation)
    with pytest.raises(RuntimeError, match='unexpected trial implementation error'):
        solve_analysis(observations, frozen)


def test_fv_trajectory_cannot_enter_legacy_fso_state_adapter():
    from advar.sensitivity import _variational_state
    _, frozen, _ = _fixture()
    with pytest.raises(ValueError, match='FV trajectory'):
        _variational_state(initial_control(frozen), frozen)
