"""Contract checks for the shared finite-volume trajectory interface."""

import torch
import pytest

from advar.transport import finite_volume_trajectory


def _edges(value: float, *, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return tuple(torch.full((1,), value, dtype=dtype) for _ in range(4))


def _call(*, basis: torch.Tensor, coefficients: torch.Tensor,
          support: torch.Tensor | None = None, boundary_echo=None,
          boundary_support=None, reconstruction: str = "minmod",
          interval_seconds=0.2, spacing_yx=(1.0, 1.0), replay=False):
    dtype = coefficients.dtype
    echo = torch.ones((1, 1), dtype=dtype)
    support = torch.ones_like(echo) if support is None else support
    zero_echo = (_edges(0.0, dtype=dtype),) * 2
    known_support = (_edges(1.0, dtype=dtype),) * 2
    return finite_volume_trajectory(
        echo,
        support,
        coefficients,
        torch.tensor(0.01, dtype=dtype),
        psi_basis=basis,
        leads=1,
        substeps_per_interval=1,
        interval_seconds=interval_seconds,
        spacing_yx=spacing_yx,
        boundary_echo=((zero_echo,) if boundary_echo is None else boundary_echo),
        boundary_support=((known_support,) if boundary_support is None
                          else boundary_support),
        reconstruction=reconstruction,
        replay=replay,
    )


def _basis(value: float = 1.0) -> torch.Tensor:
    return torch.tensor([[[0.0, 0.0], [value, 0.0]]], dtype=torch.float64)


def test_basis_anchor_and_full_rank_are_required() -> None:
    with pytest.raises(ValueError, match="anchor"):
        _call(basis=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64),
              coefficients=torch.tensor([0.1], dtype=torch.float64))

    dependent = torch.cat((_basis(), _basis()))
    with pytest.raises(ValueError, match="rank"):
        _call(basis=dependent,
              coefficients=torch.tensor([0.1, -0.2], dtype=torch.float64))


def test_empty_psi_basis_is_a_valid_zero_flow_case() -> None:
    frames, supports = _call(
        basis=torch.empty((0, 2, 2), dtype=torch.float64),
        coefficients=torch.empty((0,), dtype=torch.float64),
    )
    assert frames.shape == supports.shape == (2, 1, 1)
    torch.testing.assert_close(frames[0], torch.ones_like(frames[0]))


def test_schedule_length_and_fixed_timing_types_are_checked() -> None:
    kwargs = dict(
        basis=_basis(), coefficients=torch.tensor([0.1], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="length"):
        _call(**kwargs, boundary_echo=(), boundary_support=())
    with pytest.raises(TypeError, match="interval_seconds"):
        _call(**kwargs, interval_seconds=True)


def test_partial_support_is_allowed_for_donorcell_but_rejected_by_minmod() -> None:
    partial = torch.full((1, 1), 0.5, dtype=torch.float64)
    empty_basis = torch.empty((0, 2, 2), dtype=torch.float64)
    frames, supports = _call(
        basis=empty_basis,
        coefficients=torch.empty((0,), dtype=torch.float64),
        support=partial,
        reconstruction="donorcell",
    )
    torch.testing.assert_close(supports[0], partial)
    with pytest.raises(ValueError, match="support|known"):
        _call(
            basis=empty_basis,
            coefficients=torch.empty((0,), dtype=torch.float64),
            support=partial,
            reconstruction="minmod",
        )


def test_replay_preserves_support_boundary_derivative() -> None:
    dtype = torch.float64
    zero = torch.zeros(1, dtype=dtype)
    left0 = torch.tensor(0.4, dtype=dtype, requires_grad=True)
    left1 = torch.tensor(0.6, dtype=dtype, requires_grad=True)

    def support_stage(left: torch.Tensor):
        return (left.expand(1), zero, zero, zero)

    support_schedule = ((support_stage(left0), support_stage(left1)),)
    kwargs = dict(
        basis=_basis(),
        coefficients=torch.tensor([0.1], dtype=dtype),
        support=torch.full((1, 1), 0.5, dtype=dtype),
        boundary_support=support_schedule,
        reconstruction="donorcell",
        spacing_yx=(100.0, 100.0),
    )
    plain_support = _call(**kwargs, replay=False)[1]
    plain_grad = torch.autograd.grad(
        plain_support[-1].sum(), (left0, left1), retain_graph=True
    )
    replay_support = _call(**kwargs, replay=True)[1]
    replay_grad = torch.autograd.grad(replay_support[-1].sum(), (left0, left1))
    torch.testing.assert_close(replay_support, plain_support, atol=0, rtol=0)
    for actual, expected in zip(replay_grad, plain_grad):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert all(float(value.abs().sum()) > 0 for value in replay_grad)


def test_replay_freezes_mutable_spacing_metadata_before_backward() -> None:
    dtype = torch.float64
    # This anchored streamfunction produces nonzero boundary flux, so the
    # spacing enters the result and its ψ derivative.
    basis = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=dtype)
    coefficient = torch.tensor([0.1], dtype=dtype, requires_grad=True)
    echo = torch.ones((1, 1), dtype=dtype)
    support = torch.ones_like(echo)
    zero_echo = (_edges(0.0, dtype=dtype),) * 2
    known_support = (_edges(1.0, dtype=dtype),) * 2
    schedule_echo = (zero_echo,)
    schedule_support = (known_support,)

    spacing = [1.0, 1.0]
    echo_frames, _ = finite_volume_trajectory(
        echo,
        support,
        coefficient,
        torch.tensor(0.01, dtype=dtype),
        psi_basis=basis,
        leads=1,
        substeps_per_interval=1,
        interval_seconds=0.2,
        spacing_yx=spacing,
        boundary_echo=schedule_echo,
        boundary_support=schedule_support,
        reconstruction="minmod",
        replay=True,
    )
    # Replay happens during backward. Mutating the caller's list must not
    # change the fixed geometry used by the recorded forward evaluation.
    spacing[:] = [7.0, 11.0]
    actual = torch.autograd.grad(echo_frames[-1].sum(), coefficient)[0]

    reference_coefficient = torch.tensor([0.1], dtype=dtype, requires_grad=True)
    reference_frames, _ = finite_volume_trajectory(
        echo,
        support,
        reference_coefficient,
        torch.tensor(0.01, dtype=dtype),
        psi_basis=basis,
        leads=1,
        substeps_per_interval=1,
        interval_seconds=0.2,
        spacing_yx=(1.0, 1.0),
        boundary_echo=schedule_echo,
        boundary_support=schedule_support,
        reconstruction="minmod",
        replay=True,
    )
    expected = torch.autograd.grad(
        reference_frames[-1].sum(), reference_coefficient
    )[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
