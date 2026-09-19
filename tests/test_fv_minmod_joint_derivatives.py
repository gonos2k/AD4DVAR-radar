"""Small joint AD checks for the bounded minmod FV trajectory.

The test uses a fixed, fully known boundary schedule.  The streamfunction has
both signs on the faces, so the signed transport branch is exercised without
using a coefficient-sensitive zero-flow tie.  Nine substeps also exercises the
replay tape's block split at eight substeps.
"""

from __future__ import annotations

import torch

from advar.transport import face_volume_fluxes, finite_volume_trajectory


DTYPE = torch.float64
HEIGHT, WIDTH = 4, 5
SUBSTEPS = 9
COEFFICIENTS = 5


def _fixed_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yv = torch.arange(HEIGHT + 1, dtype=DTYPE)[:, None]
    xv = torch.arange(WIDTH + 1, dtype=DTYPE)[None, :]
    y = yv.expand(HEIGHT + 1, WIDTH + 1)
    x = xv.expand(HEIGHT + 1, WIDTH + 1)
    # Every slice is anchored at [0, 0], as required by the transport API.
    basis = torch.stack((y, x, x * y, 0.5 * (x.square() - y.square()), x.square() * y))
    return x, y, basis


def _edge_schedule(
    *, echo_value: float = 0.01
) -> tuple[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], ...]:
    echo_edges = (
        torch.full((HEIGHT,), echo_value, dtype=DTYPE),
        torch.full((HEIGHT,), 1.5 * echo_value, dtype=DTYPE),
        torch.full((WIDTH,), 2.0 * echo_value, dtype=DTYPE),
        torch.full((WIDTH,), 2.5 * echo_value, dtype=DTYPE),
    )
    return ((echo_edges, echo_edges),) * SUBSTEPS


def _support_schedule() -> tuple[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], ...]:
    support_edges = (
        torch.ones(HEIGHT, dtype=DTYPE),
        torch.ones(HEIGHT, dtype=DTYPE),
        torch.ones(WIDTH, dtype=DTYPE),
        torch.ones(WIDTH, dtype=DTYPE),
    )
    return ((support_edges, support_edges),) * SUBSTEPS


def _zero_edge_schedule() -> tuple[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], ...]:
    return _edge_schedule(echo_value=0.0)


def _case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, basis = _fixed_geometry()
    cell_y = y[:-1, :-1]
    cell_x = x[:-1, :-1]
    initial = 1.5 + 0.04 * cell_y + 0.03 * cell_x + 0.01 * cell_x * cell_y
    initial = initial + 0.001 * cell_x.square() + 0.002 * cell_y.square()
    initial = initial + 0.0003 * cell_x * cell_y.square()
    support = torch.ones_like(initial)
    coefficients = initial.new_tensor([0.11, -0.08, 0.07, 0.04, 0.0])
    growth = initial.new_tensor(0.03)
    return initial, support, basis, coefficients, growth


def _unpack(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    field_size = HEIGHT * WIDTH
    return (
        value[:field_size].reshape(HEIGHT, WIDTH),
        value[field_size : field_size + COEFFICIENTS],
        value[-1],
    )


def _trajectory(
    value: torch.Tensor,
    *,
    reconstruction: str,
    replay: bool,
    zero_boundary: bool = False,
) -> torch.Tensor:
    initial, coefficients, growth = _unpack(value)
    _, support, _, _, _ = _case()
    boundary = _zero_edge_schedule() if zero_boundary else _edge_schedule()
    echo_frames, _ = finite_volume_trajectory(
        initial,
        support,
        coefficients,
        growth,
        psi_basis=_fixed_geometry()[2],
        leads=1,
        substeps_per_interval=SUBSTEPS,
        interval_seconds=1.0,
        spacing_yx=(10.0, 10.0),
        boundary_echo=boundary,
        boundary_support=_support_schedule(),
        reconstruction=reconstruction,
        max_courant=0.5,
        replay=replay,
    )
    # Include both initial and final frames so initial-field, flow, and growth
    # all contribute to the scalar derivative below.
    return echo_frames


def _objective(
    value: torch.Tensor, *, reconstruction: str, replay: bool
) -> torch.Tensor:
    frames = _trajectory(value, reconstruction=reconstruction, replay=replay)
    return frames.square().mean() + 0.13 * value[-1].square()


def test_minmod_joint_gradient_hvp_and_replay_with_signed_flow() -> None:
    initial, _, basis, coefficients, growth = _case()
    value = torch.cat((initial.reshape(-1), coefficients, growth.reshape(1)))
    psi = torch.einsum("k,kij->ij", coefficients, basis)
    qx, qy = face_volume_fluxes(psi)
    assert float(qx.min()) < 0.0 < float(qx.max())
    assert float(qy.min()) < 0.0 < float(qy.max())
    # Every active face is away from a sign tie; extrema alone would not
    # exclude an interior zero face.
    assert bool(torch.all(qx.abs() > 1.0e-3))
    assert bool(torch.all(qy.abs() > 1.0e-3))
    assert SUBSTEPS > 8

    objective = lambda v, replay: _objective(
        v, reconstruction="minmod", replay=replay
    )
    gradient_no_replay = torch.func.grad(lambda v: objective(v, False))(value)
    gradient_replay = torch.func.grad(lambda v: objective(v, True))(value)
    direction = torch.linspace(-0.01, 0.01, value.numel(), dtype=DTYPE)
    hvp_no_replay = torch.func.jvp(
        torch.func.grad(lambda v: objective(v, False)), (value,), (direction,)
    )[1]
    hvp_replay = torch.func.jvp(
        torch.func.grad(lambda v: objective(v, True)), (value,), (direction,)
    )[1]
    torch.testing.assert_close(gradient_replay, gradient_no_replay, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(hvp_replay, hvp_no_replay, rtol=2e-12, atol=2e-13)

    step = 1.0e-4
    central_gradient = (
        objective(value + step * direction, False)
        - objective(value - step * direction, False)
    ) / (2.0 * step)
    central_hvp = (
        torch.func.grad(lambda v: objective(v, False))(value + step * direction)
        - torch.func.grad(lambda v: objective(v, False))(value - step * direction)
    ) / (2.0 * step)
    torch.testing.assert_close(
        central_gradient,
        gradient_no_replay @ direction,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    torch.testing.assert_close(central_hvp, hvp_no_replay, rtol=1.0e-8, atol=1.0e-10)

    hessian = torch.func.hessian(
        lambda v: objective(v, False)
    )(value)
    torch.testing.assert_close(hessian, hessian.T, rtol=2e-12, atol=2e-13)
    field_stop = HEIGHT * WIDTH
    flow_block = hessian[:field_stop, field_stop : field_stop + COEFFICIENTS]
    flow_growth = hessian[field_stop : field_stop + COEFFICIENTS, -1]
    assert float(torch.linalg.matrix_norm(flow_block)) > 1.0e-6
    assert float(torch.linalg.vector_norm(flow_growth)) > 1.0e-6

    donor_gradient = torch.func.grad(
        lambda v: _objective(v, reconstruction="donorcell", replay=True)
    )(value)
    donor_hvp = torch.func.jvp(
        torch.func.grad(
            lambda v: _objective(v, reconstruction="donorcell", replay=True)
        ),
        (value,),
        (direction,),
    )[1]
    assert float((gradient_replay - donor_gradient).norm()) > 1.0e-8
    assert float((hvp_replay - donor_hvp).norm()) > 1.0e-9


def test_one_sided_joint_state_and_gradient_taylor_slopes() -> None:
    initial, _, _, coefficients, growth = _case()
    value = torch.cat((initial.reshape(-1), coefficients, growth.reshape(1)))
    field_stop = HEIGHT * WIDTH
    flow_stop = field_stop + COEFFICIENTS

    def normalized(vector: torch.Tensor) -> torch.Tensor:
        return vector / torch.linalg.vector_norm(vector)

    field = torch.zeros_like(value)
    field[:field_stop] = normalized(
        torch.linspace(-1.0, 1.0, field_stop, dtype=DTYPE)
    )
    flow = torch.zeros_like(value)
    flow[field_stop:flow_stop] = normalized(
        value.new_tensor([0.2, -0.3, 0.4, -0.5, 0.6])
    )
    growth_direction = torch.zeros_like(value)
    growth_direction[-1] = 1.0
    directions = {
        "q": field,
        "flow": flow,
        "g": growth_direction,
        "qflow": normalized(field + flow),
        "qgrowth": normalized(field + growth_direction),
        "flowgrowth": normalized(flow + growth_direction),
    }

    objective = lambda v: _objective(v, reconstruction="minmod", replay=False)
    state = lambda v: _trajectory(v, reconstruction="minmod", replay=False)
    gradient = torch.func.grad(objective)
    base_gradient = gradient(value)
    base_state = state(value)
    for direction in directions.values():
        state_tangent = torch.func.jvp(state, (value,), (direction,))[1]
        gradient_tangent = torch.func.jvp(
            gradient, (value,), (direction,)
        )[1]
        previous_errors = None
        for step in (1.0e-4, 5.0e-5):
            plus_state = (state(value + step * direction) - base_state) / step
            minus_state = (base_state - state(value - step * direction)) / step
            plus_gradient = (gradient(value + step * direction) - base_gradient) / step
            minus_gradient = (base_gradient - gradient(value - step * direction)) / step

            # One-sided slope errors are O(h); admit the scaled subtraction
            # roundoff floor instead of forcing a ratio for a linear direction.
            errors = torch.stack(tuple(z.norm() for z in (
                plus_state-state_tangent, minus_state-state_tangent,
                plus_gradient-gradient_tangent, minus_gradient-gradient_tangent,
            )))
            scales = torch.stack((state_tangent.norm(), state_tangent.norm(),
                                  gradient_tangent.norm(), gradient_tangent.norm()))
            floors = 128*torch.finfo(DTYPE).eps/step * torch.stack((
                base_state.norm(), base_state.norm(),
                base_gradient.norm(), base_gradient.norm(),
            ))
            assert bool((errors <= 2e-4*scales + floors).all())
            if previous_errors is not None:
                assert bool((errors <= .6*previous_errors + floors).all())
            previous_errors = errors



def test_nominal_zero_flow_forward_identity_is_distinct_from_signed_flow() -> None:
    """Check zero-flux forward identity without a derivative safety claim."""
    initial, _, basis, _, _ = _case()
    zero_coefficients = torch.zeros(COEFFICIENTS, dtype=DTYPE)
    zero_psi = torch.einsum("k,kij->ij", zero_coefficients, basis)
    zero_qx, zero_qy = face_volume_fluxes(zero_psi)
    assert bool(torch.equal(zero_qx, torch.zeros_like(zero_qx)))
    assert bool(torch.equal(zero_qy, torch.zeros_like(zero_qy)))

    zero_value = torch.cat((initial.reshape(-1), zero_coefficients, torch.zeros(1, dtype=DTYPE)))
    zero_frames = _trajectory(
        zero_value,
        reconstruction="minmod",
        replay=True,
        zero_boundary=True,
    )
    torch.testing.assert_close(zero_frames, initial.expand_as(zero_frames))

    _, _, _, nominal_coefficients, _ = _case()
    nominal_value = torch.cat(
        (initial.reshape(-1), nominal_coefficients, torch.zeros(1, dtype=DTYPE))
    )
    nominal_frames = _trajectory(
        nominal_value,
        reconstruction="minmod",
        replay=True,
        zero_boundary=True,
    )
    assert float((nominal_frames[-1] - initial).abs().max()) > 1.0e-5
