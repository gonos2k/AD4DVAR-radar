"""Independent conservation and differentiation tests for the FV transport candidate.

The expected values in this file are formed from the face-flux definition and a
small, local SSPRK2 oracle.  They deliberately do not call implementation
helpers other than the two public transport entry points.
"""

import math

import pytest
import torch

from advar.transport import face_volume_fluxes, finite_volume_step


def _stages(left, right, bottom, top):
    """Build the explicit (start, end) stage traces for four whole faces."""
    return ((left[0], right[0], bottom[0], top[0]), (left[1], right[1], bottom[1], top[1]))


def _constant_boundary(H, W, value=0.0, *, dtype=torch.float64):
    left = torch.full((H,), value, dtype=dtype)
    right = torch.full((H,), value, dtype=dtype)
    bottom = torch.full((W,), value, dtype=dtype)
    top = torch.full((W,), value, dtype=dtype)
    return _stages((left, left), (right, right), (bottom, bottom), (top, top))


def _face_flux(q_left, q_right, q):
    return torch.where(q >= 0, q * q_left, q * q_right)


def _euler_oracle(echo, qx, qy, dt, spacing, boundary):
    """One upwind Euler stage, including known external values at stage zero."""
    H, W = echo.shape
    dy, dx = spacing
    left, right, bottom, top = boundary[0]
    fx = echo.new_empty((H, W + 1))
    fy = echo.new_empty((H + 1, W))
    fx[:, 0] = _face_flux(left, echo[:, 0], qx[:, 0])
    fx[:, -1] = _face_flux(echo[:, -1], right, qx[:, -1])
    if W > 1:
        fx[:, 1:-1] = _face_flux(echo[:, :-1], echo[:, 1:], qx[:, 1:-1])
    fy[0, :] = _face_flux(bottom, echo[0, :], qy[0, :])
    fy[-1, :] = _face_flux(echo[-1, :], top, qy[-1, :])
    if H > 1:
        fy[1:-1, :] = _face_flux(echo[:-1, :], echo[1:, :], qy[1:-1, :])
    return echo - dt * (fx[:, 1:] - fx[:, :-1] + fy[1:, :] - fy[:-1, :]) / (dy * dx)


def _ssprk2_oracle(echo, qx, qy, dt, spacing, boundary):
    first = _euler_oracle(echo, qx, qy, dt, spacing, boundary)
    # Constant boundary traces are used in this oracle, so the second stage has
    # the same external values as the first.
    second = _euler_oracle(first, qx, qy, dt, spacing, boundary)
    return 0.5 * (echo + second)


def _boundary_budget(echo, qx, qy, dt, spacing, boundary):
    """Independent outward boundary budget for a single RK stage."""
    H, W = echo.shape
    left, right, bottom, top = boundary[0]
    fx_left = _face_flux(left, echo[:, 0], qx[:, 0])
    fx_right = _face_flux(echo[:, -1], right, qx[:, -1])
    fy_bottom = _face_flux(bottom, echo[0, :], qy[0, :])
    fy_top = _face_flux(echo[-1, :], top, qy[-1, :])
    outward = torch.cat((-fx_left, fx_right, -fy_bottom, fy_top))
    return dt * torch.relu(-outward).sum(), dt * torch.relu(outward).sum()


def test_face_fluxes_cancel_discrete_divergence_and_preserve_gauge():
    torch.manual_seed(3)
    psi = torch.randn(4, 6, dtype=torch.float64)
    qx, qy = face_volume_fluxes(psi)
    divergence = qx[:, 1:] - qx[:, :-1] + qy[1:, :] - qy[:-1, :]
    torch.testing.assert_close(divergence, torch.zeros_like(divergence), atol=1e-14, rtol=0)
    shifted_qx, shifted_qy = face_volume_fluxes(psi + 12345.0)
    # The subtraction is gauge invariant algebraically; a large offset can
    # still leave one ulp of cancellation in floating point.
    torch.testing.assert_close(shifted_qx, qx, atol=2e-12, rtol=1e-12)
    torch.testing.assert_close(shifted_qy, qy, atol=2e-12, rtol=1e-12)


def test_face_fluxes_use_row_increase_as_positive_y_on_a_nonsquare_grid():
    H, W = 2, 5
    dy, dx = 3.0, 0.4
    y = torch.arange(H + 1, dtype=torch.float64)[:, None] * dy
    x = torch.arange(W + 1, dtype=torch.float64)[None, :] * dx
    psi = 1.75 * y - 2.25 * x + 11.0
    qx, qy = face_volume_fluxes(psi)
    torch.testing.assert_close(qx, torch.full_like(qx, 1.75 * dy))
    torch.testing.assert_close(qy, torch.full_like(qy, 2.25 * dx))


def test_ssprk2_matches_three_cell_upwind_oracle():
    echo = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.ones((1, 4), dtype=torch.float64)
    qy = torch.zeros((2, 3), dtype=torch.float64)
    boundary = _constant_boundary(1, 3, dtype=echo.dtype)
    boundary = ((torch.full((1,), 3.0, dtype=echo.dtype), boundary[0][1], boundary[0][2], boundary[0][3]),) * 2
    actual = finite_volume_step(
        echo,
        support,
        qx,
        qy,
        dt_seconds=0.25,
        spacing_yx=(1.0, 1.0),
        boundary_echo=boundary,
        boundary_support=_constant_boundary(1, 3, 1.0, dtype=echo.dtype),
    )
    expected = _ssprk2_oracle(echo, qx, qy, 0.25, (1.0, 1.0), boundary)
    torch.testing.assert_close(actual.echo, expected)
    assert float(actual.support.min()) >= 0.0
    assert float(actual.support.max()) <= 1.0


def test_cfl_one_is_admissible_when_requested_but_default_margin_rejects():
    echo = torch.ones((1, 1), dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.ones((1, 2), dtype=echo.dtype)
    qy = torch.zeros((2, 1), dtype=echo.dtype)
    boundary = _constant_boundary(1, 1, 1.0, dtype=echo.dtype)
    with pytest.raises(ValueError, match="CFL|courant"):
        finite_volume_step(echo, support, qx, qy, dt_seconds=1.0, spacing_yx=(1.0, 1.0), boundary_echo=boundary, boundary_support=boundary)
    result = finite_volume_step(echo, support, qx, qy, dt_seconds=1.0, spacing_yx=(1.0, 1.0), max_courant=1.0, boundary_echo=boundary, boundary_support=boundary)
    torch.testing.assert_close(result.echo, echo)


def test_cfl_multiple_requires_explicit_two_steps_and_matches_oracle():
    echo = torch.tensor([[0.2, 0.7, 1.4]], dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.ones((1, 4), dtype=echo.dtype)
    qy = torch.zeros((2, 3), dtype=echo.dtype)
    boundary = _constant_boundary(1, 3, dtype=echo.dtype)
    left = torch.full((1,), 2.0, dtype=echo.dtype)
    boundary = ((left, boundary[0][1], boundary[0][2], boundary[0][3]),) * 2
    with pytest.raises(ValueError, match="CFL|courant"):
        finite_volume_step(echo, support, qx, qy, dt_seconds=1.8, spacing_yx=(1.0, 1.0), boundary_echo=boundary, boundary_support=_constant_boundary(1, 3, 1.0, dtype=echo.dtype))
    stage = _ssprk2_oracle(echo, qx, qy, 0.9, (1.0, 1.0), boundary)
    expected = _ssprk2_oracle(stage, qx, qy, 0.9, (1.0, 1.0), boundary)
    first = finite_volume_step(echo, support, qx, qy, dt_seconds=0.9, spacing_yx=(1.0, 1.0), boundary_echo=boundary, boundary_support=_constant_boundary(1, 3, 1.0, dtype=echo.dtype))
    actual = finite_volume_step(first.echo, first.support, qx, qy, dt_seconds=0.9, spacing_yx=(1.0, 1.0), boundary_echo=boundary, boundary_support=_constant_boundary(1, 3, 1.0, dtype=echo.dtype))
    torch.testing.assert_close(actual.echo, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("log_growth", [-0.3, 0.0, 0.3])
def test_no_flow_growth_is_exact_and_source_quadrature_is_independent(log_growth):
    echo = torch.tensor([[0.5, 2.0], [1.0, 3.0]], dtype=torch.float64)
    support = torch.tensor([[0.1, 0.4], [0.8, 1.0]], dtype=torch.float64)
    qx = torch.zeros((2, 3), dtype=echo.dtype)
    qy = torch.zeros((3, 2), dtype=echo.dtype)
    actual = finite_volume_step(echo, support, qx, qy, dt_seconds=0.2, spacing_yx=(2.0, 1.5), log_growth=log_growth)
    torch.testing.assert_close(actual.echo, echo * math.exp(log_growth), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(actual.support, support, atol=0, rtol=0)
    expected_source = log_growth * (echo.sum() + actual.echo.sum()) * (2.0 * 1.5) / 2.0
    torch.testing.assert_close(actual.source_quadrature, expected_source, atol=1e-13, rtol=1e-13)


def test_unknown_clear_and_partial_inflow_support_are_separate_fields():
    echo = torch.zeros((1, 1), dtype=torch.float64)
    support = torch.zeros_like(echo)
    qx = torch.ones((1, 2), dtype=echo.dtype)
    qy = torch.zeros((2, 1), dtype=echo.dtype)
    zero_echo = _constant_boundary(1, 1, 0.0, dtype=echo.dtype)
    unknown = finite_volume_step(echo, support, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0), boundary_echo=zero_echo, boundary_support=_constant_boundary(1, 1, 0.0, dtype=echo.dtype))
    torch.testing.assert_close(unknown.echo, echo)
    torch.testing.assert_close(unknown.support, support)
    assert float(unknown.transport_inflow) == pytest.approx(0.0)
    known = finite_volume_step(echo, support, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0), boundary_echo=zero_echo, boundary_support=_constant_boundary(1, 1, 1.0, dtype=echo.dtype))
    assert float(known.echo) == pytest.approx(0.0)
    assert float(known.support) == pytest.approx(0.095)
    assert float(known.transport_inflow) == pytest.approx(0.0)
    boundary_echo = _constant_boundary(1, 1, 3.5, dtype=echo.dtype)
    partial = finite_volume_step(echo, support, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0), boundary_echo=boundary_echo, boundary_support=_constant_boundary(1, 1, 0.5, dtype=echo.dtype))
    # boundary_echo is already a whole-face contribution: support is audited
    # separately and must not multiply the echo contribution a second time.
    assert float(partial.echo) == pytest.approx(0.3325)
    assert float(partial.support) == pytest.approx(0.0475)
    assert float(partial.transport_inflow) == pytest.approx(0.35)


def test_time_varying_known_inflow_uses_start_and_end_traces_once():
    echo = torch.zeros((1, 1), dtype=torch.float64)
    support = torch.zeros_like(echo)
    qx = torch.ones((1, 2), dtype=echo.dtype)
    qy = torch.zeros((2, 1), dtype=echo.dtype)
    zero = torch.zeros(1, dtype=echo.dtype)
    one = torch.ones(1, dtype=echo.dtype)
    boundary_echo = _stages((zero, zero), (zero, zero), (zero, zero), (zero, zero))
    boundary_echo = _stages((zero, one), (zero, zero), (zero, zero), (zero, zero))
    boundary_support = _stages((one, one), (zero, zero), (zero, zero), (zero, zero))
    actual = finite_volume_step(echo, support, qx, qy, dt_seconds=0.2, spacing_yx=(1.0, 1.0), boundary_echo=boundary_echo, boundary_support=boundary_support)
    # SSPRK2 with b(t_n)=0 and b(t_n+dt)=1 gives dt/2 for this one-cell trace.
    assert float(actual.echo) == pytest.approx(0.1)
    assert float(actual.transport_inflow) == pytest.approx(0.1)


def test_boundary_jump_is_applied_only_after_the_previous_interval_ends():
    echo = torch.zeros((1, 1), dtype=torch.float64)
    support = torch.zeros_like(echo)
    qx = torch.ones((1, 2), dtype=echo.dtype)
    qy = torch.zeros((2, 1), dtype=echo.dtype)
    zero_edges = _constant_boundary(1, 1, 0.0, dtype=echo.dtype)
    known_edges = _constant_boundary(1, 1, 1.0, dtype=echo.dtype)
    first = finite_volume_step(
        echo, support, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0),
        boundary_echo=zero_edges, boundary_support=zero_edges,
    )
    torch.testing.assert_close(first.echo, echo)
    torch.testing.assert_close(first.support, support)
    second = finite_volume_step(
        first.echo, first.support, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0),
        boundary_echo=known_edges, boundary_support=known_edges,
    )
    # Independent one-cell SSPRK2 oracle: q1=0.1, q2=0.09, qnew=0.095.
    assert float(second.echo) == pytest.approx(0.095)
    assert float(second.support) == pytest.approx(0.095)


def test_physical_and_integrating_factor_inflow_budgets_use_stage_traces():
    echo = torch.full((1, 1), 5.0, dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.ones((1, 2), dtype=echo.dtype)
    qy = torch.zeros((2, 1), dtype=echo.dtype)
    boundary_echo = _stages(
        (torch.full((1,), 2.0, dtype=echo.dtype), torch.full((1,), 4.0, dtype=echo.dtype)),
        (torch.zeros(1, dtype=echo.dtype),) * 2,
        (torch.zeros(1, dtype=echo.dtype),) * 2,
        (torch.zeros(1, dtype=echo.dtype),) * 2,
    )
    boundary_support = _constant_boundary(1, 1, 1.0, dtype=echo.dtype)
    log_growth = 0.2
    result = finite_volume_step(
        echo,
        support,
        qx,
        qy,
        dt_seconds=0.1,
        spacing_yx=(1.0, 1.0),
        log_growth=log_growth,
        boundary_echo=boundary_echo,
        boundary_support=boundary_support,
    )
    expected_transport = 0.05 * (2.0 + 4.0 * math.exp(-log_growth))
    expected_physical = 0.05 * (2.0 + 4.0)
    r1 = 5.0 + 0.1 * (2.0 - 5.0)
    expected_transport_out = 0.05 * (5.0 + r1)
    expected_physical_out = 0.05 * (5.0 + math.exp(log_growth) * r1)
    assert float(result.transport_inflow) == pytest.approx(expected_transport)
    assert float(result.physical_inflow) == pytest.approx(expected_physical)
    assert float(result.transport_outflow) == pytest.approx(expected_transport_out)
    assert float(result.physical_outflow) == pytest.approx(expected_physical_out)


def test_r_budget_uses_nonunit_cell_area_and_local_growth_factor():
    echo = torch.full((1, 1), 5.0, dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.ones((1, 2), dtype=echo.dtype)
    qy = torch.zeros((2, 1), dtype=echo.dtype)
    left0 = torch.full((1,), 2.0, dtype=echo.dtype)
    left1 = torch.full((1,), 2.0, dtype=echo.dtype)
    zero = torch.zeros(1, dtype=echo.dtype)
    edges = _stages((left0, left1), (zero, zero), (zero, zero), (zero, zero))
    result = finite_volume_step(
        echo, support, qx, qy, dt_seconds=0.6, spacing_yx=(2.0, 3.0),
        log_growth=0.2, boundary_echo=edges, boundary_support=_constant_boundary(1, 1, 1.0, dtype=echo.dtype),
    )
    # C=dt*Q/A=.1, so r1=5+.1*(2-5)=4.7.  Boundary budgets retain Q*dt units;
    # cell area enters only the state update and the source mass audit.
    r1 = 4.7
    assert float(result.transport_inflow) == pytest.approx(0.3 * (2.0 + 2.0 * math.exp(-0.2)))
    assert float(result.transport_outflow) == pytest.approx(0.3 * (5.0 + r1))
    assert float(result.physical_inflow) == pytest.approx(0.6 * 2.0)
    assert float(result.physical_outflow) == pytest.approx(0.3 * (5.0 + math.exp(0.2) * r1))


def test_no_flow_source_quadrature_has_cubic_local_error():
    echo = torch.tensor([[0.4, 1.1], [0.7, 2.2]], dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.zeros((2, 3), dtype=echo.dtype)
    qy = torch.zeros((3, 2), dtype=echo.dtype)
    area = 2.0 * 1.5
    gamma = 0.8
    errors = []
    for dt in (0.04, 0.02):
        result = finite_volume_step(
            echo, support, qx, qy, dt_seconds=dt, spacing_yx=(2.0, 1.5),
            log_growth=gamma * dt,
        )
        exact_mass_change = echo.sum() * area * math.expm1(gamma * dt)
        errors.append(abs(float(result.source_quadrature) - float(exact_mass_change)))
    assert errors[0] > 0.0
    assert 7.5 < errors[0] / errors[1] < 8.5


def test_boundary_budget_corners_and_sides_has_no_double_face_weight():
    echo = torch.zeros((2, 2), dtype=torch.float64)
    support = torch.zeros_like(echo)
    qx = torch.ones((2, 3), dtype=echo.dtype)
    qy = torch.ones((3, 2), dtype=echo.dtype)
    edges = _constant_boundary(2, 2, 4.0, dtype=echo.dtype)
    support_edges = _constant_boundary(2, 2, 1.0, dtype=echo.dtype)
    result = finite_volume_step(echo, support, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0), boundary_echo=edges, boundary_support=support_edges)
    # Two incoming faces per corner are counted once each; no corner area factor exists.
    assert float(result.transport_inflow) == pytest.approx(0.1 * 16.0)
    assert float(result.echo.sum()) == pytest.approx(float(result.transport_inflow - result.transport_outflow), abs=1e-12)


def test_transport_budget_matches_source_free_mass_change_with_full_outflow():
    echo = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.tensor([[0.0, 0.2, 0.5], [0.0, 0.2, 0.5]], dtype=echo.dtype)
    # Choose Qy so every cell's discrete divergence cancels exactly.
    qy = torch.tensor([[0.0, 0.0], [-0.2, -0.3], [-0.4, -0.6]], dtype=echo.dtype)
    boundary = _constant_boundary(2, 2, 0.0, dtype=echo.dtype)
    result = finite_volume_step(echo, support, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0), boundary_echo=boundary, boundary_support=boundary)
    mass_change = float(result.echo.sum() - echo.sum())
    assert mass_change == pytest.approx(float(result.transport_inflow - result.transport_outflow), abs=1e-12)
    assert float(result.source_quadrature) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_positive_ssprk2_preserves_nonnegative_echo_across_supported_scales(dtype, scale):
    echo = torch.tensor([[scale, 2.0 * scale]], dtype=dtype)
    support = torch.ones_like(echo)
    # Constant +x Q has zero discrete divergence.  There is no known left
    # inflow, so every update is an outflow/transport positivity test.
    qx = torch.ones((1, 3), dtype=dtype)
    qy = torch.zeros((2, 2), dtype=dtype)
    result = finite_volume_step(
        echo,
        support,
        qx,
        qy,
        dt_seconds=0.9,
        spacing_yx=(1.0, 1.0),
        max_courant=0.9,
    )
    assert bool(torch.isfinite(result.echo).all())
    assert bool((result.echo >= 0).all())
    assert bool((result.support >= 0).all())


@pytest.mark.parametrize("dtype,seed", [(torch.float32, 11), (torch.float64, 8), (torch.float64, 13), (torch.float64, 17)])
def test_known_constant_field_is_preserved_for_random_streamfunction(dtype, seed):
    torch.manual_seed(seed)
    psi = torch.randn((5, 7), dtype=dtype)
    qx, qy = face_volume_fluxes(psi)
    spacing = (1.7, 0.9)
    area = spacing[0] * spacing[1]
    outgoing_rate = (
        torch.relu(qx[:, 1:]) + torch.relu(-qx[:, :-1])
        + torch.relu(qy[1:, :]) + torch.relu(-qy[:-1, :])
    ) / area
    dt = 0.8 / float(outgoing_rate.max())
    echo = torch.ones((4, 6), dtype=dtype)
    support = torch.ones_like(echo)
    edges = _constant_boundary(4, 6, 1.0, dtype=dtype)
    result = finite_volume_step(
        echo, support, qx, qy, dt_seconds=dt, spacing_yx=spacing,
        boundary_echo=edges, boundary_support=edges,
    )
    torch.testing.assert_close(result.echo, echo, rtol=8e-6 if dtype == torch.float32 else 1e-12, atol=8e-6 if dtype == torch.float32 else 1e-12)
    torch.testing.assert_close(result.support, support, rtol=8e-6 if dtype == torch.float32 else 1e-12, atol=8e-6 if dtype == torch.float32 else 1e-12)


def test_zero_face_flux_has_the_two_one_sided_upwind_derivatives():
    echo = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    support = torch.ones_like(echo)
    qy = torch.zeros((2, 2), dtype=echo.dtype)
    boundary = _constant_boundary(1, 2, 0.0, dtype=echo.dtype)

    def value(q):
        # A single Qx perturbation is completed to a divergence-free field by
        # the corresponding top-face Qy values from the shared streamfunction.
        psi = torch.zeros((2, 3), dtype=echo.dtype)
        psi[1, 1] = q
        qx, qy = face_volume_fluxes(psi)
        return finite_volume_step(echo, support, qx, qy, dt_seconds=0.01, spacing_yx=(1.0, 1.0), boundary_echo=boundary, boundary_support=boundary).echo

    eps = 1e-7
    positive_slope = (value(torch.tensor(eps)) - value(torch.tensor(0.0))) / eps
    negative_slope = (value(torch.tensor(-eps)) - value(torch.tensor(0.0))) / -eps
    assert float(positive_slope[0, 0]) == pytest.approx(-0.01 * 1.0, rel=1e-6)
    # The streamfunction completion contributes the opposite top-face trace
    # on the negative side, so this cell's total slope is -dt*(3-1).
    assert float(negative_slope[0, 0]) == pytest.approx(-0.01 * 2.0, rel=1e-6)


def test_jvp_vjp_and_finite_difference_agree_away_from_sign_boundaries():
    torch.manual_seed(19)
    echo = torch.tensor([[1.2, 2.0], [0.7, 1.8]], dtype=torch.float64)
    support = torch.ones_like(echo)
    psi = torch.tensor([[0.0, 1.2, 2.7], [2.2, 4.5, 6.4], [4.1, 8.0, 10.9]], dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(psi)
    cotangent = torch.randn_like(echo)

    def from_psi(value):
        qx, qy = face_volume_fluxes(value)
        return finite_volume_step(echo, support, qx, qy, dt_seconds=0.03, spacing_yx=(1.2, 0.8)).echo

    _, jvp = torch.func.jvp(from_psi, (psi,), (direction,))
    eps = 1e-6
    finite = (from_psi(psi + eps * direction) - from_psi(psi - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jvp, finite, atol=2e-8, rtol=2e-7)
    (vjp,) = torch.autograd.grad((from_psi(psi) * cotangent).sum(), (psi,))
    assert float((jvp * cotangent).sum().detach()) == pytest.approx(float((direction * vjp).sum().detach()), rel=2e-7, abs=2e-8)

    qx, qy = face_volume_fluxes(psi)
    qx = qx + 0.37
    qy = qy - 0.23
    def from_q(value):
        return finite_volume_step(value, support, qx, qy, dt_seconds=0.03, spacing_yx=(1.2, 0.8)).echo
    echo_direction = torch.randn_like(echo)
    _, qjvp = torch.func.jvp(from_q, (echo,), (echo_direction,))
    qfinite = (from_q(echo + eps * echo_direction) - from_q(echo - eps * echo_direction)) / (2 * eps)
    torch.testing.assert_close(qjvp, qfinite, atol=2e-8, rtol=2e-7)

    growth = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    def from_growth(value):
        return finite_volume_step(echo, support, torch.zeros_like(qx), torch.zeros_like(qy), dt_seconds=0.03, spacing_yx=(1.2, 0.8), log_growth=value).echo
    _, gjvp = torch.func.jvp(from_growth, (growth.detach(),), (torch.ones_like(growth),))
    gfinite = (from_growth(growth.detach() + eps) - from_growth(growth.detach() - eps)) / (2 * eps)
    torch.testing.assert_close(gjvp, gfinite, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize(
    "bad_call",
    [
        lambda e, s, qx, qy: finite_volume_step(e[:1], s, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0)),
        lambda e, s, qx, qy: finite_volume_step(e, s + 2.0, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0)),
        lambda e, s, qx, qy: finite_volume_step(e, s, qx[:, :-1], qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0)),
        lambda e, s, qx, qy: finite_volume_step(e, s, qx, qy, dt_seconds=-0.1, spacing_yx=(1.0, 1.0)),
        lambda e, s, qx, qy: finite_volume_step(e, s, qx, qy, dt_seconds=0.1, spacing_yx=(0.0, 1.0)),
        lambda e, s, qx, qy: finite_volume_step(e, s, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 0.0)),
        lambda e, s, qx, qy: finite_volume_step(e, s, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0), max_courant=0.0),
    ],
)
def test_step_rejects_invalid_shapes_units_and_step_controls(bad_call):
    echo = torch.ones((2, 2), dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.zeros((2, 3), dtype=echo.dtype)
    qy = torch.zeros((3, 2), dtype=echo.dtype)
    with pytest.raises((TypeError, ValueError)):
        bad_call(echo, support, qx, qy)


@pytest.mark.parametrize("bad_kind", ["echo_nan", "support_nan", "negative_echo", "support_gt_one", "qx_nan", "dtype"])
def test_step_rejects_nonfinite_negative_or_mixed_dtype_inputs(bad_kind):
    echo = torch.ones((2, 2), dtype=torch.float64)
    support = torch.ones_like(echo)
    qx = torch.zeros((2, 3), dtype=echo.dtype)
    qy = torch.zeros((3, 2), dtype=echo.dtype)
    if bad_kind == "echo_nan":
        echo[0, 0] = math.nan
    elif bad_kind == "support_nan":
        support[0, 0] = math.nan
    elif bad_kind == "negative_echo":
        echo[0, 0] = -1.0
    elif bad_kind == "support_gt_one":
        support[0, 0] = 1.1
    elif bad_kind == "qx_nan":
        qx[0, 0] = math.nan
    else:
        qy = qy.to(torch.float32)
    with pytest.raises((TypeError, ValueError)):
        finite_volume_step(echo, support, qx, qy, dt_seconds=0.1, spacing_yx=(1.0, 1.0))


@pytest.mark.parametrize("bad_psi", [torch.ones((3,)), torch.full((3, 3), math.nan)])
def test_face_fluxes_reject_wrong_shape_and_nonfinite_psi(bad_psi):
    with pytest.raises((TypeError, ValueError)):
        face_volume_fluxes(bad_psi)


@pytest.mark.parametrize("dtype,amplitude", [(torch.float32, 1e30), (torch.float64, 1e300)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_subnormal_face_flux_keeps_representable_echo_transfer(dtype, amplitude, sign):
    tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype))
    echo = torch.tensor([[amplitude, 0.]], dtype=dtype)
    if sign < 0:
        echo = echo.flip(1)
    result = finite_volume_step(
        echo, torch.ones_like(echo), (sign * tiny).expand(1, 3),
        torch.zeros(2, 2, dtype=dtype), dt_seconds=1., spacing_yx=(1., 1.),
    )
    # The O(Q²) correction is below representable precision, but Q*q is not.
    destination = result.echo[0, 1 if sign > 0 else 0]
    torch.testing.assert_close(destination, tiny * amplitude, atol=0, rtol=8 * torch.finfo(dtype).eps)


def test_zero_flux_preserves_midpoint_generalized_derivative():
    flux = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    echo = torch.tensor([[2., 0.]], dtype=flux.dtype)
    result = finite_volume_step(
        echo, torch.ones_like(echo), flux.expand(1, 3),
        torch.zeros(2, 2, dtype=flux.dtype), dt_seconds=1., spacing_yx=(1., 1.),
    )
    derivative, = torch.autograd.grad(result.echo[0, 1], flux)
    # Right/left directional slopes are 2 and 0; this is their midpoint.
    torch.testing.assert_close(derivative, torch.ones_like(derivative), atol=0, rtol=0)


def test_cfl_one_fp32_nonunit_area_uses_the_checked_coefficient():
    out = torch.tensor(0.00021806027507409453, dtype=torch.float32)
    echo = torch.ones(1, 1)
    qx = torch.stack((torch.zeros_like(out), out)).reshape(1, 2)
    qy = torch.stack((torch.zeros_like(out), -out)).reshape(2, 1)
    result = finite_volume_step(
        echo, echo, qx, qy, dt_seconds=132353.1803176046,
        spacing_yx=(28.860971450805664, 1.), max_courant=1.,
    )
    # The checked FP32 Courant coefficient is exactly one: both Euler
    # stages empty the cell, and SSPRK2 retains half the initial value.
    torch.testing.assert_close(result.echo, echo * .5, atol=0, rtol=0)
