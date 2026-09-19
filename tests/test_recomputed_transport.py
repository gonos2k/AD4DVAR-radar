"""Small derivative oracles for the matrix-free recomputation wrapper.

The finite-volume part deliberately uses a fixed, fully-known minmod branch.
The zero-flux case below checks first-order duality only: the max/min split has
a valid implementation derivative at zero, but it is not a classical twice
differentiable point.
"""

import torch

from advar.matrix_free import recompute
from advar.transport import face_volume_fluxes, finite_volume_step


def _scalar_block(state: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
    return torch.sin(state) + coefficient * state.square() + 0.1 * coefficient**3


def test_recompute_tuple_with_unused_output_and_shared_input() -> None:
    x = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)

    def block(a, b):
        return a * b, torch.sin(a + b)

    def score(value):
        used, _ = recompute(block, value, value)
        return used.square().sum()

    assert torch.autograd.gradcheck(score, (x,))
    assert torch.autograd.gradgradcheck(score, (x,))
    torch.testing.assert_close(torch.func.grad(score)(x), 4 * x**3)


def _plain_scalar_chain(
    state: torch.Tensor, coefficients: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    for coefficient in coefficients:
        state = _scalar_block(state, coefficient)
    return state


def _recomputed_scalar_chain(
    state: torch.Tensor, coefficients: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    for coefficient in coefficients:
        state = recompute(_scalar_block, state, coefficient)
    return state


def test_recompute_scalar_three_block_gradgrad_and_hvp_match_plain() -> None:
    dtype = torch.float64
    state = torch.tensor(0.37, dtype=dtype, requires_grad=True)
    coefficients = tuple(
        torch.tensor(value, dtype=dtype, requires_grad=True)
        for value in (0.11, -0.07, 0.05)
    )
    direction = torch.tensor(-0.43, dtype=dtype)

    def plain_value(value: torch.Tensor) -> torch.Tensor:
        return _plain_scalar_chain(value, coefficients).square()

    def recomputed_value(value: torch.Tensor) -> torch.Tensor:
        return _recomputed_scalar_chain(value, coefficients).square()

    torch.testing.assert_close(plain_value(state), recomputed_value(state))
    plain_gradient = torch.autograd.grad(
        plain_value(state), state, create_graph=True
    )[0]
    recomputed_gradient = torch.autograd.grad(
        recomputed_value(state), state, create_graph=True
    )[0]
    torch.testing.assert_close(plain_gradient, recomputed_gradient)
    plain_hvp = torch.autograd.grad(plain_gradient, state)[0] * direction
    recomputed_hvp = torch.autograd.grad(recomputed_gradient, state)[0] * direction
    torch.testing.assert_close(plain_hvp, recomputed_hvp)

    # Exercise the transform stack used by the FSO path as well as ordinary
    # reverse-mode gradgrad.
    torch.autograd.gradcheck(recomputed_value, (state,), eps=1.0e-6, atol=1.0e-8)
    torch.autograd.gradgradcheck(
        recomputed_value,
        (state,),
        eps=1.0e-5,
        atol=1.0e-7,
    )


def _boundary_tensor(
    height: int, width: int, *, value: float, dtype: torch.dtype
) -> torch.Tensor:
    result = torch.full((2, 4, max(height, width)), value, dtype=dtype)
    return result


def _fv_block(
    packed: torch.Tensor,
    psi: torch.Tensor,
    growth: torch.Tensor,
    boundary_echo: torch.Tensor,
    boundary_support: torch.Tensor,
) -> torch.Tensor:
    """One tensor-only block; non-tensor arguments stay fixed configuration."""

    height, width = packed.shape[-2:]
    echo, support = packed.unbind(0)
    qx, qy = face_volume_fluxes(psi)
    edge_lengths = (height, height, width, width)
    echo_edges = tuple(
        tuple(
            boundary_echo[stage, edge, : edge_lengths[edge]]
            for edge in range(4)
        )
        for stage in range(2)
    )
    support_edges = tuple(
        tuple(
            boundary_support[stage, edge, : edge_lengths[edge]]
            for edge in range(4)
        )
        for stage in range(2)
    )
    result = finite_volume_step(
        echo,
        support,
        qx,
        qy,
        dt_seconds=0.05,
        spacing_yx=(1.0, 1.0),
        log_growth=growth,
        boundary_echo=echo_edges,
        boundary_support=support_edges,
        max_courant=0.5,
        reconstruction="minmod",
    )
    return torch.stack((result.echo, result.support))


def _plain_fv_chain(*inputs: torch.Tensor) -> torch.Tensor:
    state = inputs[0]
    boundary_echo, boundary_support = inputs[-2:]
    for block in range(3):
        psi = inputs[1 + 2 * block]
        growth = inputs[2 + 2 * block]
        state = _fv_block(state, psi, growth, boundary_echo, boundary_support)
    return state


def _recomputed_fv_chain(*inputs: torch.Tensor) -> torch.Tensor:
    state = inputs[0]
    boundary_echo, boundary_support = inputs[-2:]
    for block in range(3):
        psi = inputs[1 + 2 * block]
        growth = inputs[2 + 2 * block]
        state = recompute(
            _fv_block,
            state,
            psi,
            growth,
            boundary_echo,
            boundary_support,
        )
    return state


def _fv_inputs(*, zero_flow: bool = False) -> tuple[torch.Tensor, ...]:
    dtype = torch.float64
    height, width = 3, 4
    y, x = torch.meshgrid(
        torch.arange(height + 1, dtype=dtype),
        torch.arange(width + 1, dtype=dtype),
        indexing="ij",
    )
    if zero_flow:
        psi_base = torch.zeros_like(y)
    else:
        psi_base = 0.03 * y * x + 0.005 * torch.sin(0.3 * y + 0.2 * x)
    packed = torch.stack(
        (
            1.0 + 0.03 * x[:-1, :-1].square() + 0.04 * y[:-1, :-1].square()
            + 0.02 * x[:-1, :-1] * y[:-1, :-1],
            torch.ones((height, width), dtype=dtype),
        )
    ).requires_grad_()
    values = [packed]
    for index, growth_value in enumerate((0.01, -0.005, 0.008)):
        values.append((psi_base * (1.0 + 0.07 * index)).requires_grad_())
        values.append(torch.tensor(growth_value, dtype=dtype, requires_grad=True))
    values.append(_boundary_tensor(height, width, value=0.0, dtype=dtype))
    values.append(_boundary_tensor(height, width, value=1.0, dtype=dtype))
    return tuple(values)


def _directions(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    # A constant psi direction is only a gauge shift and would miss flow AD.
    directions = [0.003 * torch.cos(torch.arange(value.numel(), dtype=value.dtype))
                  .reshape_as(value) for value in inputs]
    directions[0][1] = 0
    directions[-1] = torch.zeros_like(inputs[-1])
    return tuple(directions)


def test_recomputed_fv_three_blocks_match_plain_value_jvp_vjp_and_mixed() -> None:
    inputs = _fv_inputs()
    tangents = _directions(inputs)
    cotangent = torch.linspace(
        0.2,
        0.8,
        inputs[0].numel(),
        dtype=inputs[0].dtype,
    ).reshape_as(inputs[0])

    plain = _plain_fv_chain(*inputs)
    recomputed = _recomputed_fv_chain(*inputs)
    torch.testing.assert_close(plain, recomputed, atol=1.0e-12, rtol=1.0e-12)

    plain_vjp = torch.func.vjp(
        lambda *values: (_plain_fv_chain(*values) * cotangent).sum(),
        *inputs,
    )[1](torch.ones((), dtype=inputs[0].dtype))
    recomputed_vjp = torch.func.vjp(
        lambda *values: (_recomputed_fv_chain(*values) * cotangent).sum(),
        *inputs,
    )[1](torch.ones((), dtype=inputs[0].dtype))
    for expected, actual in zip(plain_vjp, recomputed_vjp):
        torch.testing.assert_close(expected, actual, atol=1.0e-11, rtol=1.0e-10)

    _, plain_jvp = torch.func.jvp(_plain_fv_chain, inputs, tangents)
    _, recomputed_jvp = torch.func.jvp(_recomputed_fv_chain, inputs, tangents)
    torch.testing.assert_close(plain_jvp, recomputed_jvp, atol=1.0e-11, rtol=1.0e-10)

    def plain_gradient(*values: torch.Tensor) -> torch.Tensor:
        return torch.func.grad(
            lambda packed: (
                _plain_fv_chain(packed, *values[1:]) * cotangent
            ).sum()
        )(values[0])

    def recomputed_gradient(*values: torch.Tensor) -> torch.Tensor:
        return torch.func.grad(
            lambda packed: (
                _recomputed_fv_chain(packed, *values[1:]) * cotangent
            ).sum()
        )(values[0])

    _, plain_mixed = torch.func.jvp(plain_gradient, inputs, tangents)
    _, recomputed_mixed = torch.func.jvp(recomputed_gradient, inputs, tangents)
    torch.testing.assert_close(
        plain_mixed,
        recomputed_mixed,
        atol=2.0e-10,
        rtol=2.0e-9,
    )


def test_recomputed_fv_zero_flow_tie_matches_plain_first_order_only() -> None:
    inputs = _fv_inputs(zero_flow=True)
    tangents = _directions(inputs)
    plain, plain_jvp = torch.func.jvp(_plain_fv_chain, inputs, tangents)
    recomputed, recomputed_jvp = torch.func.jvp(
        _recomputed_fv_chain, inputs, tangents
    )
    torch.testing.assert_close(plain, recomputed, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        plain_jvp,
        recomputed_jvp,
        atol=1.0e-11,
        rtol=1.0e-10,
    )

    cotangent = torch.ones_like(plain)
    plain_vjp = torch.func.vjp(_plain_fv_chain, *inputs)[1](cotangent)
    recomputed_vjp = torch.func.vjp(_recomputed_fv_chain, *inputs)[1](cotangent)
    for expected, actual in zip(plain_vjp, recomputed_vjp):
        torch.testing.assert_close(expected, actual, atol=1.0e-11, rtol=1.0e-10)
