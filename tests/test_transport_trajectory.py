"""Discrete and derivative oracles for the shared spatial-flow trajectory."""

import math

import pytest
import torch

from advar.transport import finite_volume_step, finite_volume_trajectory, face_volume_fluxes


def _schedule(packed, height, width):
    return tuple(tuple(tuple(row.split((height, height, width, width)))
                       for row in step) for step in packed)


@pytest.mark.parametrize("replay", [False, True])
def test_one_cell_trajectory_matches_time_dependent_boundary_recurrence(replay):
    q = torch.tensor([[.4]], dtype=torch.float64)
    basis = torch.tensor([[[0., 0.], [1., 1.]]], dtype=q.dtype)
    boundary = torch.arange(1, 33, dtype=q.dtype).reshape(4, 2, 4) / 10
    g = q.new_tensor(.12)
    frames, support = finite_volume_trajectory(
        q, torch.ones_like(q), q.new_tensor([.2]), g, psi_basis=basis,
        leads=2, substeps_per_interval=2, interval_seconds=1., spacing_yx=(1., 1.),
        boundary_echo=_schedule(boundary, 1, 1),
        boundary_support=_schedule(torch.ones_like(boundary), 1, 1), replay=replay,
    )
    # Euler coefficient C=.2*.5=.1; each substep has log growth .12/2.
    value = .4
    expected = [.4]
    for index in range(4):
        b0, b1 = boundary[index, :, 0].tolist()
        r1 = .9 * value + .1 * b0
        value = .5 * (math.exp(.06) * (value + .9 * r1) + .1 * b1)
        if index % 2 == 1:
            expected.append(value)
    torch.testing.assert_close(frames[:, 0, 0], q.new_tensor(expected), rtol=1e-13, atol=0)
    torch.testing.assert_close(support, torch.ones_like(support), rtol=1e-13, atol=0)


def _case(dtype=torch.float64, substeps=2):
    q = torch.tensor([[.3, .8, .6, 1.2], [.5, .9, 1.4, .7],
                      [1.1, .4, .9, 1.5]], dtype=dtype)
    y, x = torch.meshgrid(torch.arange(4, dtype=dtype), torch.arange(5, dtype=dtype), indexing="ij")
    basis = torch.stack((y, -x))
    boundaries = torch.linspace(.3, 1.3, 2*substeps*2*14, dtype=dtype).reshape(2*substeps, 2, 14)
    inputs = (q, q.new_tensor([.13, .07]), q.new_tensor(.03), boundaries)
    return basis, inputs


def _run(basis, inputs, replay, scheme):
    q, coefficients, growth, boundary = inputs
    return finite_volume_trajectory(
        q, torch.ones_like(q), coefficients, growth, psi_basis=basis,
        leads=2, substeps_per_interval=boundary.shape[0]//2, interval_seconds=.4, spacing_yx=(1.5, 2.),
        boundary_echo=_schedule(boundary, 3, 4),
        boundary_support=_schedule(torch.ones_like(boundary), 3, 4),
        replay=replay, reconstruction=scheme,
    )


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_trajectory_matches_direct_steps(dtype, scheme):
    basis, inputs = _case(dtype)
    q, coefficients, growth, boundary = inputs
    frames, supports = _run(basis, inputs, False, scheme)
    qx, qy = face_volume_fluxes((basis * coefficients[:, None, None]).sum(0))
    field, support = q, torch.ones_like(q)
    expected = [q]
    for index, edges in enumerate(_schedule(boundary, 3, 4)):
        result = finite_volume_step(
            field, support, qx, qy, dt_seconds=.2, spacing_yx=(1.5, 2.),
            log_growth=growth/2, boundary_echo=edges,
            boundary_support=tuple(tuple(torch.ones_like(edge) for edge in stage) for stage in edges),
            reconstruction=scheme, max_courant=.5,
        )
        field, support = result.echo, result.support
        if index % 2 == 1:
            expected.append(field)
    torch.testing.assert_close(frames, torch.stack(expected), rtol=0, atol=0)
    torch.testing.assert_close(supports[-1], support, rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
def test_forecast_continuation_uses_the_same_discrete_trajectory(scheme):
    basis, inputs = _case()
    q, coefficients, growth, boundary = inputs
    frames, supports = _run(basis, inputs, False, scheme)
    continued, continued_support = finite_volume_trajectory(
        frames[1], supports[1], coefficients, growth, psi_basis=basis,
        leads=1, substeps_per_interval=2, interval_seconds=.4, spacing_yx=(1.5, 2.),
        boundary_echo=_schedule(boundary[2:], 3, 4),
        boundary_support=_schedule(torch.ones_like(boundary[2:]), 3, 4),
        reconstruction=scheme, replay=True,
    )
    torch.testing.assert_close(continued, frames[1:], rtol=0, atol=0)
    torch.testing.assert_close(continued_support, supports[1:], rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["donorcell", "minmod"])
@pytest.mark.parametrize("substeps", [2, 9])
def test_replay_value_jvp_vjp_and_mixed_second_derivative(scheme, substeps):
    basis, inputs = _case(substeps=substeps)
    directions = tuple(.01 * torch.cos(torch.arange(v.numel(), dtype=v.dtype)).reshape(v.shape)
                       for v in inputs)

    def forward(replay, *values):
        return _run(basis, values, replay, scheme)[0]

    primal, tangent = torch.func.jvp(lambda *a: forward(False, *a), inputs, directions)
    replay_primal, replay_tangent = torch.func.jvp(lambda *a: forward(True, *a), inputs, directions)
    torch.testing.assert_close(replay_primal, primal, rtol=0, atol=0)
    torch.testing.assert_close(replay_tangent, tangent, rtol=1e-12, atol=1e-14)
    seed = torch.sin(primal)
    _, pullback = torch.func.vjp(lambda *a: forward(False, *a), *inputs)
    _, replay_pullback = torch.func.vjp(lambda *a: forward(True, *a), *inputs)
    gradients = pullback(seed)
    for actual, expected in zip(replay_pullback(seed), gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close((seed*tangent).sum(), sum((a*b).sum() for a,b in zip(directions, gradients)),
                               rtol=1e-12, atol=1e-14)
    assert gradients[3].abs().sum() > 0  # Boundary path must reach reverse AD.

    def gradient(replay):
        return torch.func.grad(lambda *a: forward(replay, *a).square().mean(), argnums=(0, 1, 2, 3))

    _, second = torch.func.jvp(gradient(False), inputs, directions)
    _, replay_second = torch.func.jvp(gradient(True), inputs, directions)
    for actual, expected in zip(replay_second, second):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-13)
    # Independent central difference on a non-tied branch checks the full map.
    h = 1e-5
    plus = forward(False, *(a+h*d for a,d in zip(inputs, directions)))
    minus = forward(False, *(a-h*d for a,d in zip(inputs, directions)))
    torch.testing.assert_close(tangent, (plus-minus)/(2*h), rtol=2e-6, atol=2e-10)
