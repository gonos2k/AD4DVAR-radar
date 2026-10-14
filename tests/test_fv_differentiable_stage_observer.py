"""Derivative capture at minmod stages outside checkpoint replay."""

import pytest
import torch

from advar import transport


DTYPE = torch.float64


def _trajectory(*, replay: bool) -> tuple[torch.Tensor, torch.Tensor]:
    echo = torch.tensor([[0.4]], dtype=DTYPE)
    support = torch.ones_like(echo)
    basis = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=DTYPE)
    coefficients = torch.tensor([0.1], dtype=DTYPE)
    growth = torch.tensor(0.0, dtype=DTYPE)
    zero = torch.zeros(1, dtype=DTYPE)
    one = torch.ones(1, dtype=DTYPE)
    zero_edges = (zero, zero, zero, zero)
    support_edges = (one, one, one, one)
    return transport.finite_volume_trajectory(
        echo,
        support,
        coefficients,
        growth,
        psi_basis=basis,
        leads=1,
        substeps_per_interval=1,
        interval_seconds=1.0,
        spacing_yx=(1.0, 1.0),
        boundary_echo=((zero_edges, zero_edges),),
        boundary_support=((support_edges, support_edges),),
        reconstruction="minmod",
        max_courant=0.5,
        replay=replay,
    )


def _stage_case():
    q = torch.tensor(
        [[1.0, 1.3, 1.9], [1.2, 1.7, 2.1], [1.5, 1.8, 2.4]],
        dtype=DTYPE,
        requires_grad=True,
    )
    support = torch.ones_like(q)
    qx = torch.full((3, 4), 0.01, dtype=DTYPE)
    qy = torch.zeros((4, 3), dtype=DTYPE)
    edges = (q[:, 0], q[:, -1], q[0], q[-1])
    known = tuple(torch.ones_like(edge) for edge in edges)
    return q, support, qx, qy, edges, known


def _step(q, support, qx, qy, edges, known):
    return transport.finite_volume_step(
        q,
        support,
        qx,
        qy,
        dt_seconds=0.1,
        spacing_yx=(1.0, 1.0),
        boundary_echo=(edges, edges),
        boundary_support=(known, known),
        reconstruction="minmod",
        max_courant=0.5,
    )


def test_default_observer_keeps_detached_isolated_snapshots():
    q, support, qx, qy, edges, known = _stage_case()
    inputs = (q.detach().clone(), qx.clone(), qy.clone())
    expected = _step(q, support, qx, qy, edges, known).echo
    received = []

    def mutate(stage_q, stage_qx, stage_qy):
        received.append((stage_q, stage_qx, stage_qy))
        stage_q.zero_()
        stage_qx.zero_()
        stage_qy.zero_()

    with transport.observe_minmod_stages(mutate):
        actual = _step(q, support, qx, qy, edges, known).echo

    assert len(received) == 2
    for snapshot, original in zip(received[0], (q, qx, qy)):
        assert not snapshot.requires_grad
        assert snapshot.data_ptr() != original.data_ptr()
    torch.testing.assert_close(q.detach(), inputs[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(qx, inputs[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(qy, inputs[2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_differentiable_stage_copy_preserves_reverse_gap_derivative():
    q, support, qx, qy, edges, known = _stage_case()
    stages = []
    with transport.observe_minmod_stages(
        lambda stage_q, _stage_qx, _stage_qy: stages.append(stage_q),
        differentiable=True,
    ):
        _step(q, support, qx, qy, edges, known)

    assert len(stages) == 2
    assert stages[0].requires_grad
    assert stages[0].data_ptr() != q.data_ptr()
    # Gap between the two x slopes at the center cell: right minus left.
    stage_gap = stages[0][1, 2] - 2.0 * stages[0][1, 1] + stages[0][1, 0]
    derivative, = torch.autograd.grad(stage_gap, q)
    expected = torch.zeros_like(q)
    expected[1] = q.new_tensor([1.0, -2.0, 1.0])
    torch.testing.assert_close(derivative, expected, rtol=0.0, atol=0.0)


def test_nested_observers_keep_connected_outer_copy_isolated_from_inner_mutation():
    q, support, qx, qy, edges, known = _stage_case()
    outer_values = []

    def mutate_inner(stage_q, _stage_qx, _stage_qy) -> None:
        stage_q.zero_()

    with transport.observe_minmod_stages(
        lambda stage_q, *_: outer_values.append(stage_q), differentiable=True,
    ):
        with transport.observe_minmod_stages(mutate_inner):
            _step(q, support, qx, qy, edges, known)

    assert len(outer_values) == 2
    assert outer_values[0].requires_grad
    torch.testing.assert_close(outer_values[0], q, rtol=0.0, atol=0.0)
    gap = outer_values[0][1, 2] - 2.0 * outer_values[0][1, 1] + outer_values[0][1, 0]
    derivative, = torch.autograd.grad(gap, q)
    assert torch.isfinite(derivative).all()
    assert torch.count_nonzero(derivative) == 3


def test_replay_refusal_tracks_outer_and_inner_differentiable_scopes():
    with transport.observe_minmod_stages(lambda *_: None, differentiable=True):
        with transport.observe_minmod_stages(lambda *_: None):
            with pytest.raises(ValueError, match="requires replay=False"):
                _trajectory(replay=True)
        # The inner detached observer has exited, but the outer capture remains.
        with pytest.raises(ValueError, match="requires replay=False"):
            _trajectory(replay=True)

    # A failed guard must not leak its context state.
    _trajectory(replay=True)


def test_inner_differentiable_scope_guards_only_its_lexical_extent():
    with transport.observe_minmod_stages(lambda *_: None):
        with transport.observe_minmod_stages(lambda *_: None, differentiable=True):
            with pytest.raises(ValueError, match="requires replay=False"):
                _trajectory(replay=True)
        _trajectory(replay=True)
